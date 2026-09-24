#!/usr/bin/env python3
"""Run one authenticated model replica under the screening median noise pair."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_full_calibrated_vit_comparison import (
    calibration_sites,
    parse_metric,
    source_identity,
    validate_snn_sites,
)
from scripts.experiments.run_vit_local_range_noise_condition import (
    parse_physical_counts,
)
from scripts.experiments.screening_median_model_sweep import (
    DEADLINE_MARGIN_SIGMA_RATIO,
    MODELS,
    canonical_alpha,
    load_screening_median,
    scaled_fractions,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity, local_gpu


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("screening median protocol schema differs")
    protocol_identity = payload.get("identity")
    if (
        not isinstance(protocol_identity, dict)
        or payload.get("protocol_id") != identity.json_sha256(protocol_identity)
    ):
        raise ValueError("screening median protocol identity differs")
    return resolved, payload


def _completed_vit_calibration(
    resource: dict[str, Any],
    *,
    expected_commit: str,
) -> tuple[Path, set[str]]:
    root = Path(resource["calibration_source"]).resolve(strict=True)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    calibration = (root / "calibration.json").resolve(strict=True)
    digest = identity.sha256_file(calibration)
    if (
        result.get("state") != "complete"
        or manifest.get("model_key") != resource["model_key"]
        or manifest.get("source_commit") != expected_commit
        or result.get("calibration_sha256") != digest
        or resource.get("calibration_sha256") != digest
    ):
        raise ValueError("ViT calibration source identity differs")
    return calibration, calibration_sites(calibration)


def _verify_model_resources(model: str, resource: dict[str, Any]) -> None:
    model_path = Path(resource["model_id"]).resolve(strict=True)
    if model == "cct7":
        checkpoint_sha256 = identity.sha256_file(model_path)
        dataset = identity.artifact_identity(
            Path(resource["dataset_root"]) / "cifar-10-batches-py"
        )
        expected_dataset = resource["evaluation_population"]
        if (
            dataset["aggregate_sha256"] != expected_dataset["aggregate_sha256"]
            or dataset["bytes"] != expected_dataset["bytes"]
        ):
            raise ValueError("CIFAR-10 dataset identity differs")
    else:
        checkpoint_sha256 = identity.artifact_identity(model_path)["aggregate_sha256"]
    if checkpoint_sha256 != resource["checkpoint_sha256"]:
        raise ValueError("model checkpoint checksum differs")
    if model != "cct7":
        from datasets import Dataset, load_from_disk

        preprocessing = Path(resource["image_preprocessing_config"]).resolve(strict=True)
        if (
            identity.sha256_file(preprocessing)
            != resource["image_preprocessing_sha256"]
        ):
            raise ValueError("ViT preprocessing identity differs")
        dataset = load_from_disk(resource["evaluation_dataset_path"])
        if (
            not isinstance(dataset, Dataset)
            or len(dataset) != 50_000
            or str(dataset._fingerprint)
            != resource["evaluation_dataset_fingerprint"]
        ):
            raise ValueError("ImageNet evaluation population identity differs")


def _aggregate_cct_counts(stats: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "events",
        "misses",
        "deadline_events",
        "outputs",
    )
    totals = {field: 0 for field in (*fields, "underflows", "overflows")}
    sites = []
    for site, counts in sorted(stats.items()):
        if not isinstance(counts, dict):
            raise ValueError("CCT Gaussian counter row is malformed")
        row = {"site": site}
        for field in fields:
            value = int(counts.get(field, 0))
            if value < 0:
                raise ValueError("CCT Gaussian counter is negative")
            totals[field] += value
            row[field] = value
        for source, destination in (
            ("output_underflows", "underflows"),
            ("output_overflows", "overflows"),
        ):
            value = int(counts.get(source, 0))
            if value < 0:
                raise ValueError("CCT Gaussian output counter is negative")
            totals[destination] += value
            row[destination] = value
        sites.append(row)
    if not sites or totals["events"] <= 0 or totals["outputs"] <= 0:
        raise ValueError("CCT Gaussian physical counters are missing")
    totals.update(
        miss_rate=totals["misses"] / totals["events"],
        deadline_event_rate=totals["deadline_events"] / totals["events"],
        underflow_rate=totals["underflows"] / totals["outputs"],
        overflow_rate=totals["overflows"] / totals["outputs"],
        site_count=len(sites),
        sites=sites,
    )
    return totals


def _vit_command(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    resource: dict[str, Any],
    calibration: Path,
    *,
    linear_fraction: float,
    log_fraction: float,
    run_id: str,
) -> list[str]:
    source_root = Path(protocol["resources"]["source_root"])
    wrapper = source_root / "scripts/analysis/evaluate_calibrated_vit.py"
    evaluation_selection = (
        ["--batch_size", "10", "--max_eval_batches", "50"]
        if args.evaluation_samples == 500
        else ["--batch_size", "32", "--quick-test"]
    )
    return [
        args.python_bin,
        "-u",
        str(wrapper),
        "--source-root",
        str(source_root),
        "--calibration-dataset-path",
        resource["calibration_dataset_path"],
        "--calibration-dataset-fingerprint",
        resource["calibration_dataset_fingerprint"],
        "--gelu-cubic-implementation",
        "phi_nl_psi_ed",
        "--gelu-cubic-floor",
        "1e-5",
        "--experiment_name",
        run_id,
        "--device",
        "cuda",
        "--model_backend",
        "spiking",
        "--model_id",
        resource["model_id"],
        "--dataset_id",
        "imagenet-1k",
        "--evaluation-dataset-path",
        resource["evaluation_dataset_path"],
        "--evaluation-split",
        "validation",
        "--image-preprocessing-config",
        resource["image_preprocessing_config"],
        *evaluation_selection,
        "--precision",
        "float64",
        "--source-commit",
        protocol["identity"]["source_commit"],
        "--checkpoint-sha256",
        resource["checkpoint_sha256"],
        "--no-tensorboard",
        "--report-clamp-stats",
        "--spiking-layernorm",
        "--spiking-ln-mul",
        "--spiking-ln-log",
        "--spiking-ln-expdiff",
        "--spiking-attention",
        "--spiking-mlp",
        "--no-spiking-mlp-exact-gelu",
        "--calibration-mode",
        "validate",
        "--calibration-path",
        str(calibration),
        "--calibration-samples",
        "5000",
        "--calibration-seed",
        "0",
        "--calibration-bins",
        "2048",
        "--calibration-lower-quantile",
        "0",
        "--calibration-upper-quantile",
        "1",
        "--calibration-margin-fraction",
        "0.05",
        "--gaussian-time-noise",
        "--time-noise-std-frac",
        "0",
        "--linear-time-noise-std-frac",
        repr(linear_fraction),
        "--log-time-noise-std-frac",
        repr(log_fraction),
        "--time-noise-mean",
        "0",
        "--time-noise-deadline-margin-std",
        repr(DEADLINE_MARGIN_SIGMA_RATIO),
        "--time-noise-seed",
        str(args.seed),
        "--no-mismatch-enabled",
        "--mismatch-range-std-frac",
        "0",
        "--mismatch-seed",
        "0",
        "--weight-noise-std",
        "0",
        "--bias-noise-std",
        "0",
    ]


def _cct_command(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    resource: dict[str, Any],
    *,
    linear_fraction: float,
    log_fraction: float,
) -> list[str]:
    source_root = Path(protocol["resources"]["source_root"])
    calibration = Path(resource["calibration_path"]).resolve(strict=True)
    if identity.sha256_file(calibration) != resource["calibration_sha256"]:
        raise ValueError("CCT calibration checksum differs")
    return [
        args.python_bin,
        "-u",
        str(source_root / "scripts/evaluation/error_analysis_cct.py"),
        "--checkpoint",
        resource["model_id"],
        "--dataset-root",
        resource["dataset_root"],
        "--output-dir",
        str(args.output_dir / "evaluation"),
        "--samples",
        str(args.evaluation_samples),
        "--calibration-samples",
        "5000",
        "--batch-size",
        "8",
        "--device",
        "cuda",
        "--noise-fractions",
        repr(linear_fraction),
        "--log-noise-fraction",
        repr(log_fraction),
        "--deadline-margin",
        repr(DEADLINE_MARGIN_SIGMA_RATIO),
        "--calibration-path",
        str(calibration),
        "--noise-only",
        "--seed",
        str(args.seed),
    ]


def _environment(
    source_root: Path,
    runtime: Path,
    gpu: int,
) -> dict[str, str]:
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TMPDIR=str(runtime),
        TMP=str(runtime),
        TEMP=str(runtime),
        WANDB_DIR=str(runtime),
        SOURCE_ROOT=str(source_root),
        PYTHONPATH=os.pathsep.join(
            map(
                str,
                (
                    source_root,
                    source_root / "src/transformers/src",
                    source_root / "src/spikingjelly",
                ),
            )
        ),
    )


def _admit_gpu(gpu: int) -> dict[str, Any]:
    policy = dict(local_gpu.DEFAULT_ADMISSION_POLICY)
    if gpu == 0:
        policy["max_memory_used_mib"] = 4096.0
    samples = []
    for check in range(2):
        sample = local_gpu.gpu_activity(gpu_ids=(gpu,))[gpu]
        samples.append(sample)
        if not local_gpu.gpu_available(sample, policy):
            raise RuntimeError(f"GPU {gpu} does not satisfy the admission policy")
        if check == 0:
            time.sleep(10)
    return {"policy": policy, "samples": samples}


# @lat: [[evaluation#Evaluation and Verification#Screening Median Model Timing Noise Sweep#Condition Execution]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--evaluation-samples", type=int, required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    protocol_path, protocol = _load_protocol(args.protocol)
    expected_samples = {
        "cct7": (500, 10_000),
        "imagenet_vit_small": (500, 5_000),
        "imagenet_vit_base": (500, 5_000),
    }[args.model]
    if args.evaluation_samples not in expected_samples:
        raise ValueError("evaluation sample count is outside the fixed protocol")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True
    ).strip() in {"tmpfs", "ramfs"}:
        raise RuntimeError("condition runtime must use a disk filesystem")

    alpha = canonical_alpha(args.alpha)
    pair = load_screening_median(
        Path(protocol["resources"]["hardware_summary"])
    )
    if pair.summary_sha256 != protocol["identity"]["hardware"]["summary_sha256"]:
        raise ValueError("protocol and hardware summary identities differ")
    linear_fraction, log_fraction = scaled_fractions(pair, alpha)
    resource = protocol["resources"]["models"][args.model]
    _verify_model_resources(args.model, resource)
    source_root = Path(protocol["resources"]["source_root"]).resolve(strict=True)
    expected_commit = protocol["identity"]["source_commit"]
    source_hashes = source_identity(source_root, expected_commit)
    for path in (
        Path(__file__).resolve(),
        source_root / "scripts/experiments/screening_median_model_sweep.py",
    ):
        source_hashes[str(path.relative_to(source_root))] = identity.sha256_file(path)

    run_id = f"{args.model}_alpha_{alpha.replace('.', 'p')}_seed_{args.seed}"
    if args.model == "cct7":
        command = _cct_command(
            args,
            protocol,
            resource,
            linear_fraction=linear_fraction,
            log_fraction=log_fraction,
        )
        sites = None
    else:
        calibration, sites = _completed_vit_calibration(
            resource,
            expected_commit=expected_commit,
        )
        command = _vit_command(
            args,
            protocol,
            resource,
            calibration,
            linear_fraction=linear_fraction,
            log_fraction=log_fraction,
            run_id=run_id,
        )
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": identity.sha256_file(protocol_path),
        "model": args.model,
        "alpha": alpha,
        "seed": args.seed,
        "evaluation_samples": args.evaluation_samples,
        "linear_time_noise_std_fraction": linear_fraction,
        "log_time_noise_std_fraction": log_fraction,
        "deadline_margin_sigma_ratio": DEADLINE_MARGIN_SIGMA_RATIO,
        "time_noise_scope": "model_wide",
        "noise_mean": 0.0,
        "hardware_summary_sha256": pair.summary_sha256,
        "phi_np_coordinate": pair.phi_np.physical_coordinate,
        "phi_nl_coordinate": pair.phi_nl.physical_coordinate,
        "source_commit": expected_commit,
        "source_hashes": source_hashes,
        "checkpoint_sha256": resource["checkpoint_sha256"],
        "calibration_sha256": resource["calibration_sha256"],
        "evaluation_population": resource["evaluation_population"],
        "physical_gpu": args.gpu,
        "command": command,
    }
    manifest_path = args.output_dir / "manifest.json"
    runtime_files.immutable_json(manifest_path, manifest)
    result_path = args.output_dir / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("state") == "complete":
            print(canonical(existing), flush=True)
            return

    lock_path = args.protocol.parent / "runtime/gpu-locks" / f"poseidon-gpu-{args.gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission = _admit_gpu(args.gpu)
        attempt = len(list(args.output_dir.joinpath("logs").glob("evaluation.attempt-*.log"))) + 1
        log_path = args.output_dir / "logs" / f"evaluation.attempt-{attempt:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_files.atomic_json(
            result_path,
            {"state": "running", "run_id": run_id, "admission": admission},
        )
        started = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=source_root,
                env=_environment(source_root, runtime, args.gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            failure = {
                "state": "failed",
                "run_id": run_id,
                "exit_status": completed.returncode,
                "log_file": str(log_path.relative_to(args.output_dir)),
                "admission": admission,
            }
            runtime_files.atomic_json(result_path, failure)
            raise RuntimeError(f"model evaluator exited with status {completed.returncode}")

    if args.model == "cct7":
        summary_path = args.output_dir / "evaluation/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dense_learned_modules") != []:
            raise ValueError("converted CCT retains dense learned modules")
        rows = summary.get("conditions", [])
        if len(rows) != 1:
            raise ValueError("noise-only CCT result must contain one condition")
        row = rows[0]
        if (
            row.get("samples") != args.evaluation_samples
            or not math.isclose(
                float(row.get("linear_time_noise_std_fraction")),
                linear_fraction,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
            or not math.isclose(
                float(row.get("log_time_noise_std_fraction")),
                log_fraction,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            )
        ):
            raise ValueError("CCT condition result differs from its manifest")
        metrics = {
            "correct": int(row["correct"]),
            "total": int(row["samples"]),
            "accuracy": float(row["accuracy"]),
            "prediction_sha256": row["prediction_sha256"],
            "physical_counts": _aggregate_cct_counts(row["gaussian_counts"]),
        }
    else:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        metrics = parse_metric(log_text, args.evaluation_samples)
        if sites is None:
            raise AssertionError("ViT calibration sites were not loaded")
        validate_snn_sites(log_text, sites)
        metrics["physical_counts"] = parse_physical_counts(log_text)

    if not math.isclose(
        metrics["correct"] / metrics["total"],
        metrics["accuracy"],
        rel_tol=0.0,
        abs_tol=5.0e-9,
    ):
        raise ValueError("model accuracy and correct count differ")
    result = {
        "state": "complete",
        "run_id": run_id,
        "elapsed_seconds": time.monotonic() - started,
        "log_file": str(log_path.relative_to(args.output_dir)),
        "log_sha256": identity.sha256_file(log_path),
        "metrics": metrics,
        "admission": admission,
    }
    source_identity(source_root, expected_commit)
    runtime_files.atomic_json(result_path, result)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
