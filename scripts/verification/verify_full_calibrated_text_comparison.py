#!/usr/bin/env python3
"""Verify complete text-comparison manifests, metrics and dataset identity."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_full_calibrated_text_comparison as runner
from scripts.analysis import summarize_full_calibrated_text_comparison as summary
from scripts.runtime import identity


def reject(function, error=Exception):
    try:
        function()
    except error:
        return
    raise AssertionError("invalid full comparison input was accepted")


def verify_dataset_identity(root: Path) -> None:
    path = root / "dataset"
    dataset = Dataset.from_dict({"sentence": [str(index) for index in range(10)],
                                 "label": [index % 2 for index in range(10)]})
    dataset.save_to_disk(path)
    stored_fingerprint = str(load_from_disk(path)._fingerprint)
    identity = runner.dataset_identity(path, stored_fingerprint, 10)
    assert identity["samples"] == 10 and identity["files_sha256"]
    runner.verify_dataset_snapshot(identity)
    (path / "unexpected-cache.arrow").write_bytes(b"cache")
    reject(lambda: runner.verify_dataset_snapshot(identity), ValueError)
    (path / "unexpected-cache.arrow").unlink()
    reject(lambda: runner.dataset_identity(path, "wrong", 10), ValueError)
    reject(lambda: runner.dataset_identity(path, stored_fingerprint, 9), ValueError)


def verify_classification_parser() -> None:
    lines = ["Evaluation dataset fingerprint: abc"]
    for batch, (correct, total) in enumerate(((7, 8), (9, 10)), 1):
        lines.append(
            f"Evaluation progress: batch={batch}/2 correct={correct} total={total}/10 "
            f"accuracy={correct / total:.8f} elapsed_s={batch}.0 eta_s={2 - batch}.0"
        )
    lines += ["Correct/total: 9/10", "Prediction SHA256: " + "a" * 64,
              "Calibration[layer/site] values=100, underflows=0 (rate=0), overflows=0 (rate=0)"]
    result = runner.parse_classification("\n".join(lines), 10, {"layer/site"})
    assert result["accuracy"] == 0.9 and result["calibration_site_count"] == 1
    reject(lambda: runner.parse_classification("\n".join(lines[:-1]), 10, {"layer/site"}), ValueError)


def verify_gpt2_parser() -> None:
    rows = []
    for batch, samples in ((1, 8), (2, 10)):
        rows.append(json.dumps({
            "event": "evaluation_progress", "batch": batch, "total_batches": 2,
            "evaluated_samples": samples, "valid_loss_batches": batch,
            "average_loss": 3.0, "perplexity": 20.085536923187668,
            "loss_aggregation": "mean_of_batch_losses", "token_nll_sum": 30.0 * batch,
            "valid_token_count": 10 * batch, "token_weighted_loss": 3.0,
            "token_weighted_perplexity": 20.085536923187668,
        }, sort_keys=True))
    rows[1] = " 50%|progress| " + rows[1]
    text = "Evaluation dataset fingerprint: xyz\n" + "\n".join(rows)
    text += "\nCalibration[layer/site] values=100, underflows=0 (rate=0), overflows=0 (rate=0)\n"
    result = runner.parse_gpt2(text, 10, {"layer/site"})
    assert result["valid_token_count"] == 20
    assert result["token_weighted_perplexity"] == result["perplexity"]
    reject(lambda: runner.parse_gpt2(text.replace('"batch": 2', '"batch": 3'), 10, {"layer/site"}), ValueError)


def verify_commands(root: Path) -> None:
    source = Path(__file__).resolve().parents[2]
    args = SimpleNamespace(
        family="bert", source_root=source, python_bin="python", model_id="checkpoint",
        cache_dir="cache", calibration_dataset_path="training",
        calibration_dataset_fingerprint="train-fingerprint",
        evaluation_dataset_path="evaluation", evaluation_dataset_fingerprint="eval-fingerprint",
    )
    commands = runner.build_commands(args, root)
    assert list(commands) == ["collect", "ann", "snn"]
    assert "--calibration-dataset-path" in commands["collect"]
    assert commands["collect"].count("--calibration-mode") == 1
    assert commands["ann"][-2:] == ["--calibration-mode", "none"]
    assert commands["snn"][-2:] == ["--calibration-mode", "validate"]

    args.family = "roberta_large"
    commands = runner.build_commands(args, root)
    assert commands["collect"][2].endswith("error_analysis_roberta.py")
    assert runner.MODEL_CONFIG["roberta_large"]["sites"] == 218
    assert runner.MODEL_CONFIG["roberta_large"]["tag"] == runner.ROBERTA_LARGE_TAG
    assert runner.MODEL_CONFIG["gpt2"]["tag"] == runner.GPT2_COMPOSED_GELU_TAG
    assert set(runner.FAMILY_CONFIG) == {"bert", "roberta", "gpt2"}
    assert runner.gpu_lock_filename(
        host_label="local", physical_gpu=4, family="gpt2", slurm_job_id=None,
    ) == "gpu-4.lock"
    assert runner.gpu_lock_filename(
        host_label="ubai", physical_gpu=0, family="gpt2", slurm_job_id="123",
    ) == "ubai-123-gpt2.lock"
    reject(lambda: runner.gpu_lock_filename(
        host_label="ubai", physical_gpu=0, family="gpt2", slurm_job_id=None,
    ), ValueError)


def verify_frozen_source_identity(root: Path) -> None:
    source = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip()
    with patch.object(
        runner.subprocess,
        "check_output",
        side_effect=[commit + "\n", ""],
    ):
        hashes = runner.source_identity(source, commit, "gpt2")
    frozen = root / "source-identity.json"
    frozen.write_text(json.dumps({
        "source_commit": commit,
        "families": {"gpt2": hashes},
    }))
    with patch.object(runner.shutil, "which", return_value=None), patch.dict(
        os.environ,
        {"FROZEN_SOURCE_IDENTITY_PATH": str(frozen)},
    ):
        assert runner.source_identity(source, commit, "gpt2") == hashes
        changed = json.loads(frozen.read_text())
        changed["families"]["gpt2"][next(iter(hashes))] = "0" * 64
        frozen.write_text(json.dumps(changed))
        reject(lambda: runner.source_identity(source, commit, "gpt2"), ValueError)


def verify_collection_progress(root: Path) -> None:
    log = root / "bert-collect.log"
    log.write_text(
        "Calibration progress: pass=1/2 batch=5/625 samples=40/5000 elapsed_seconds=10.0\n"
    )
    output = root / "bert-progress"
    runner.update_progress(output, "bert", "collect", log)
    value = json.loads((output / "status.json").read_text())
    assert value["progress"]["batch"] == 5
    assert value["progress"]["total_batches"] == 1250
    assert (output / "progress" / "collect-batch-0005.json").is_file()


def evaluation_log(family: str, *, with_sites: bool) -> str:
    expected = runner.FAMILY_CONFIG[family]["evaluation_samples"]
    batches = (expected + runner.BATCH_SIZE - 1) // runner.BATCH_SIZE
    lines = ["Evaluation dataset fingerprint: evaluation"]
    if family == "gpt2":
        for batch in range(1, batches + 1):
            samples = min(batch * runner.BATCH_SIZE, expected)
            lines.append(json.dumps({
                "event": "evaluation_progress", "batch": batch, "total_batches": batches,
                "evaluated_samples": samples, "average_loss": 3.0, "perplexity": 20.0,
                "token_nll_sum": 30.0, "valid_token_count": 10,
                "token_weighted_loss": 3.0, "token_weighted_perplexity": 20.0,
            }, sort_keys=True))
    else:
        for batch in range(1, batches + 1):
            samples = min(batch * runner.BATCH_SIZE, expected)
            lines.append(
                f"Evaluation progress: batch={batch}/{batches} correct={samples} "
                f"total={samples}/{expected} accuracy=1.00000000 elapsed_s=1.0 eta_s=0.0"
            )
        lines.extend([f"Correct/total: {expected}/{expected}", "Prediction SHA256: " + "a" * 64])
    if with_sites:
        lines.extend(f"Calibration[layer/{index}] values=1, underflows=0 (rate=0), overflows=0 (rate=0)"
                     for index in range(runner.FAMILY_CONFIG[family]["sites"]))
    return "\n".join(lines) + "\n"


def verify_summarizer(root: Path) -> None:
    campaign = root / "campaign"
    for family in runner.FAMILY_CONFIG:
        output = campaign / family
        output.joinpath("logs").mkdir(parents=True)
        output.joinpath("phases").mkdir()
        sites = [{"module_name": "layer", "tensor_name": str(index)}
                 for index in range(runner.FAMILY_CONFIG[family]["sites"])]
        output.joinpath("calibration.json").write_text(json.dumps({
            "layers": sites, "metadata": {"dtype": "float64", "theta": 40.0,
                                             "model_options": [["text_calibration_policy_version", 1]]},
        }))
        phases = {}
        for phase in ("collect", "ann", "snn"):
            log = output / "logs" / f"{phase}.log"
            log.write_text("collected\n" if phase == "collect" else evaluation_log(
                family, with_sites=phase == "snn",
            ))
            phases[phase] = {"phase": phase, "elapsed_seconds": 1.0,
                             "log_file": str(log.relative_to(output)),
                             "log_sha256": identity.sha256_file(log)}
        calibration_sha = identity.sha256_file(output / "calibration.json")
        phases["collect"]["calibration_sha256"] = calibration_sha
        output.joinpath("manifest.json").write_text(json.dumps({
            "tag": runner.MODEL_CONFIG[family]["tag"],
            "family": family,
            "evaluator_family": runner.MODEL_CONFIG[family]["evaluator_family"],
            "source_commit": "a" * 40,
            "checkpoint_files_sha256": {"model": "b" * 64},
            "calibration_dataset": {"fingerprint": "training"},
            "evaluation_dataset": {"fingerprint": "evaluation"},
            "calibration_samples": 5000,
            "evaluation_samples": runner.FAMILY_CONFIG[family]["evaluation_samples"],
            "batch_size": 8, "theta": 40.0, "dtype": "float64",
        }))
        output.joinpath("result.json").write_text(json.dumps({
            "state": "complete", "family": family, "phases": phases,
            "calibration_sha256": calibration_sha,
        }))
    provenance = summary.build(campaign, campaign / "outputs", require_complete=True)
    assert provenance["complete"] and len(provenance["families"]) == 3
    assert set(provenance["files"]) == {"raw_runs.csv", "summary.csv", "calibration_sites.csv"}

    reused = root / "reused"
    for family in ("roberta", "gpt2"):
        source = campaign / family
        source_manifest = json.loads((source / "manifest.json").read_text())
        if family == "roberta":
            source_manifest.pop("evaluator_family")
            (source / "manifest.json").write_text(json.dumps(source_manifest))
        evidence, sites = runner.reused_calibration_evidence(
            source,
            family=family,
            evaluator_family=runner.MODEL_CONFIG[family]["evaluator_family"],
            checkpoint_files_sha256=source_manifest["checkpoint_files_sha256"],
            calibration_dataset=source_manifest["calibration_dataset"],
            expected_sites=runner.MODEL_CONFIG[family]["sites"],
        )
        assert len(sites) == runner.MODEL_CONFIG[family]["sites"]
        output = reused / family
        output.joinpath("logs").mkdir(parents=True)
        shutil.copyfile(source / "calibration.json", output / "calibration.json")
        source_result = json.loads((source / "result.json").read_text())
        phases = {}
        for phase in ("ann", "snn"):
            shutil.copyfile(source / "logs" / f"{phase}.log", output / "logs" / f"{phase}.log")
            phases[phase] = source_result["phases"][phase]
        manifest = dict(source_manifest)
        manifest.update(tag=runner.POWER_REUSE_TAG, calibration_reuse=evidence,
                        gelu_cubic_implementation="phi_nl_psi_ed_v1")
        output.joinpath("manifest.json").write_text(json.dumps(manifest))
        output.joinpath("result.json").write_text(json.dumps({
            "state": "complete", "family": family, "phases": phases,
            "calibration_sha256": evidence["calibration_sha256"],
            "calibration_reuse": evidence,
        }))
    reuse_provenance = summary.build(
        reused,
        reused / "outputs",
        require_complete=True,
        requested_models=("roberta", "gpt2"),
    )
    assert reuse_provenance["tag"] == runner.POWER_REUSE_TAG
    (campaign / "gpt2" / "result.json").unlink()
    reject(lambda: summary.build(campaign, campaign / "partial", require_complete=True), ValueError)


# @lat: [[text-calibration#Text Model Calibration#Complete Comparison Execution]]
def main() -> None:
    runtime = Path("/data/delayed-temporal/artifacts/runtime/verification-full-text-comparison")
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as temporary:
        root = Path(temporary)
        verify_dataset_identity(root)
        verify_classification_parser()
        verify_gpt2_parser()
        verify_commands(root)
        verify_frozen_source_identity(root)
        verify_collection_progress(root)
        verify_summarizer(root)
    print("Full calibrated text comparison checks passed")


if __name__ == "__main__":
    main()
