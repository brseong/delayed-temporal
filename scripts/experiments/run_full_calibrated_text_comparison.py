#!/usr/bin/env python3
"""Run one complete calibrated text-model comparison with resumable evidence."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import local_gpu
from utils.transforms.functions import GELU_CUBIC_IMPLEMENTATION


ARTIFACTS = Path(os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", "/data/delayed-temporal/artifacts"))
TAG = "conversion_comparison_theta40_calibrated_float64_bounds3_v3"
GPT2_COMPOSED_GELU_TAG = "gpt2_theta40_calibrated_float64_bounds3_composed_gelu_v1"
FAMILY_CONFIG = {
    "bert": {"task": "sst2", "evaluation_samples": 872, "sites": 110, "activation": "gelu"},
    "roberta": {"task": "sst2", "evaluation_samples": 872, "sites": 110, "activation": "gelu"},
    "gpt2": {
        "task": "wikitext2", "evaluation_samples": 2891, "sites": 109,
        "activation": "gelu_new",
        "activation_implementation": "composed_gelu_new_v1",
    },
}
ROBERTA_LARGE_TAG = "roberta_large_theta40_calibrated_float64_bounds3_v1"
POWER_REUSE_TAG = "text_power_gelu_theta40_float64_reused_calibration_v1"
MODEL_CONFIG = {
    **{
        name: {**config, "evaluator_family": name, "tag": TAG}
        for name, config in FAMILY_CONFIG.items()
    },
    "gpt2": {
        **FAMILY_CONFIG["gpt2"], "evaluator_family": "gpt2",
        "tag": GPT2_COMPOSED_GELU_TAG,
    },
    "roberta_large": {
        "task": "sst2", "evaluation_samples": 872, "sites": 218,
        "activation": "gelu", "evaluator_family": "roberta", "tag": ROBERTA_LARGE_TAG,
    },
}
CALIBRATION_SAMPLES = 5000
BATCH_SIZE = 8


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_identity(source: Path, expected_commit: str, family: str) -> dict[str, str]:
    paths = sorted(source.joinpath("utils").rglob("*.py"))
    paths += [
        source / "scripts/evaluation" / "text_calibration_runtime.py",
        source / "scripts/evaluation" / f"error_analysis_{family}.py",
        source / "scripts/runtime" / "files.py",
        source / "scripts/runtime" / "identity.py",
        source / "scripts/runtime" / "local_gpu.py",
        Path(__file__).resolve(),
    ]
    hashes = {
        str(path.relative_to(source)): identity.sha256_file(path)
        for path in sorted(set(paths))
    }
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("text comparison requires a full source commit")
    git_path = shutil.which("git")
    if git_path is not None:
        head = subprocess.check_output(
            [git_path, "-C", str(source), "rev-parse", "HEAD"], text=True,
        ).strip()
        dirty = subprocess.check_output(
            [git_path, "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        if head != expected_commit or dirty:
            raise ValueError("text comparison requires the clean frozen source commit")
        return hashes

    frozen_path = os.environ.get("FROZEN_SOURCE_IDENTITY_PATH")
    if not frozen_path:
        raise RuntimeError("git or a frozen source identity is required")
    frozen = json.loads(Path(frozen_path).read_text())
    if (
        frozen.get("source_commit") != expected_commit
        or frozen.get("families", {}).get(family) != hashes
    ):
        raise ValueError("frozen source identity differs from the mounted source")
    return hashes


def checkpoint_identity(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): identity.sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def dataset_identity(path: Path, expected_fingerprint: str, expected_samples: int) -> dict[str, Any]:
    from datasets import load_from_disk

    dataset = load_from_disk(path)
    if str(dataset._fingerprint) != expected_fingerprint or len(dataset) != expected_samples:
        raise ValueError("self-contained text dataset identity differs")
    files = {
        str(item.relative_to(path)): identity.sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    return {"path": str(path), "fingerprint": expected_fingerprint,
            "samples": expected_samples, "files_sha256": files}


def verify_dataset_snapshot(snapshot: dict[str, Any]) -> None:
    """Reject tokenization or another phase that mutates the saved Dataset tree."""
    path = Path(snapshot["path"])
    files = {
        str(item.relative_to(path)): identity.sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if files != snapshot["files_sha256"]:
        raise ValueError("self-contained text dataset files changed during evaluation")


def build_commands(args: argparse.Namespace, output: Path) -> dict[str, list[str]]:
    family = MODEL_CONFIG[args.family]
    evaluator_family = family["evaluator_family"]
    evaluator = args.source_root / "scripts/evaluation" / f"error_analysis_{evaluator_family}.py"
    common = [
        args.python_bin, "-u", str(evaluator), "--model_id", args.model_id,
        "--task", family["task"], "--cache-dir", args.cache_dir,
        "--device", "cuda", "--dtype", "float64", "--theta", "40",
        "--batch_size", str(BATCH_SIZE), "--max_length", "128",
        "--no-tensorboard", "--no-gaussian-time-noise", "--report-clamp-stats",
        "--spiking-layernorm", "--spiking-attention", "--spiking-mlp",
        "--spiking-ln-mul", "--spiking-ln-log", "--spiking-ln-expdiff",
        "--activation", family["activation"],
        "--calibration-dataset-path", args.calibration_dataset_path,
        "--calibration-dataset-fingerprint", args.calibration_dataset_fingerprint,
        "--evaluation-dataset-path", args.evaluation_dataset_path,
        "--evaluation-dataset-fingerprint", args.evaluation_dataset_fingerprint,
    ]
    if evaluator_family == "gpt2":
        common += ["--tau-s", "1", "--attention-theta", "40"]
    calibration = [
        "--calibration-path", str(output / "calibration.json"),
        "--calibration-samples", str(CALIBRATION_SAMPLES), "--calibration-seed", "0",
        "--calibration-bins", "2048", "--calibration-lower-quantile", "0",
        "--calibration-upper-quantile", "1", "--calibration-margin-fraction", "0.05",
    ]
    return {
        "collect": [*common, "--experiment_name", output.name + "_collect",
                    "--model_backend", "spiking", *calibration, "--calibration-mode", "collect"],
        "ann": [*common, "--experiment_name", output.name + "_ann",
                "--model_backend", "hf", "--calibration-mode", "none"],
        "snn": [*common, "--experiment_name", output.name + "_snn",
                "--model_backend", "spiking", *calibration, "--calibration-mode", "validate"],
    }


def parse_sites(path: Path, expected: int) -> tuple[dict[str, Any], set[str]]:
    table = json.loads(path.read_text())
    rows = table["layers"]
    sites = {row["module_name"] + "/" + row["tensor_name"] for row in rows}
    if len(rows) != expected or len(sites) != expected:
        raise ValueError("calibration site population differs from the full comparison")
    metadata = table["metadata"]
    if metadata["dtype"] != "float64" or metadata["theta"] != 40.0:
        raise ValueError("calibration numerical contract differs")
    if dict(metadata["model_options"])["text_calibration_policy_version"] != 1:
        raise ValueError("text calibration policy differs")
    return metadata, sites


def reused_calibration_evidence(
    source: Path,
    *,
    family: str,
    evaluator_family: str,
    checkpoint_files_sha256: dict[str, str],
    calibration_dataset: dict[str, Any],
    expected_sites: int,
) -> tuple[dict[str, Any], set[str]]:
    """Authenticate a completed calibration artifact for evaluation reuse."""
    source = source.resolve(strict=True)
    source_manifest_path = source / "manifest.json"
    source_result_path = source / "result.json"
    calibration_path = source / "calibration.json"
    if not all(path.is_file() for path in (
        source_manifest_path, source_result_path, calibration_path,
    )):
        raise ValueError("reused calibration source is incomplete")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_result = json.loads(source_result_path.read_text())
    if (
        source_result.get("state") != "complete"
        or source_manifest.get("family") != family
        or source_manifest.get("evaluator_family") != evaluator_family
        or source_manifest.get("checkpoint_files_sha256") != checkpoint_files_sha256
        or {
            key: value for key, value in source_manifest.get("calibration_dataset", {}).items()
            if key != "path"
        } != {
            key: value for key, value in calibration_dataset.items() if key != "path"
        }
        or source_manifest.get("theta") != 40.0
        or source_manifest.get("dtype") != "float64"
    ):
        raise ValueError("reused calibration identity differs from the evaluation")
    calibration_sha256 = identity.sha256_file(calibration_path)
    collect = source_result.get("phases", {}).get("collect")
    if (
        not isinstance(collect, dict)
        or collect.get("calibration_sha256") != calibration_sha256
        or source_result.get("calibration_sha256") != calibration_sha256
    ):
        raise ValueError("reused calibration hash differs from its completed source")
    _, sites = parse_sites(calibration_path, expected_sites)
    return {
        "source_output": str(source),
        "source_tag": source_manifest["tag"],
        "source_commit": source_manifest["source_commit"],
        "source_manifest_sha256": identity.sha256_file(source_manifest_path),
        "source_result_sha256": identity.sha256_file(source_result_path),
        "calibration_path": str(calibration_path),
        "calibration_sha256": calibration_sha256,
        "collection_log_sha256": collect["log_sha256"],
    }, sites


def parse_classification(log: str, expected_samples: int, sites: set[str] | None) -> dict[str, Any]:
    finals = re.findall(r"^Correct/total: (\d+)/(\d+)\s*$", log, re.MULTILINE)
    digests = re.findall(r"^Prediction SHA256: ([0-9a-f]{64})\s*$", log, re.MULTILINE)
    fingerprints = re.findall(r"^Evaluation dataset fingerprint: (\S+)\s*$", log, re.MULTILINE)
    expected_batches = math.ceil(expected_samples / BATCH_SIZE)
    progress = re.findall(
        r"Evaluation progress: batch=(\d+)/(\d+) correct=(\d+) total=(\d+)/(\d+)", log,
    )
    if len(finals) != 1 or len(digests) != 1 or len(fingerprints) != 1 or len(progress) != expected_batches:
        raise ValueError("classification result is missing or incomplete")
    correct, total = map(int, finals[0])
    if total != expected_samples or not 0 <= correct <= total:
        raise ValueError("classification final count differs")
    result = {"correct": correct, "total": total, "accuracy": correct / total,
              "prediction_sha256": digests[0], "evaluation_dataset_fingerprint": fingerprints[0],
              "completed_batches": expected_batches}
    validate_site_reports(log, sites, result)
    return result


def json_event_rows(log: str, event: str) -> list[dict[str, Any]]:
    """Parse flushed JSON even when a progress bar prefix shares the output line."""
    rows = []
    for line in log.splitlines():
        start = line.find("{")
        if start < 0 or f'"event": "{event}"' not in line[start:]:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed {event} record") from error
        if value.get("event") != event:
            raise ValueError(f"mismatched {event} record")
        rows.append(value)
    return rows


def parse_gpt2(log: str, expected_samples: int, sites: set[str] | None) -> dict[str, Any]:
    fingerprints = re.findall(r"^Evaluation dataset fingerprint: (\S+)\s*$", log, re.MULTILINE)
    rows = json_event_rows(log, "evaluation_progress")
    expected_batches = math.ceil(expected_samples / BATCH_SIZE)
    if len(fingerprints) != 1 or len(rows) != expected_batches:
        raise ValueError("GPT-2 result is missing or incomplete")
    final = rows[-1]
    required = ("average_loss", "perplexity", "token_nll_sum", "valid_token_count",
                "token_weighted_loss", "token_weighted_perplexity")
    if final["evaluated_samples"] != expected_samples or final["batch"] != expected_batches:
        raise ValueError("GPT-2 final sample count differs")
    if any(isinstance(final[key], bool) or not isinstance(final[key], (int, float))
           or not math.isfinite(final[key]) for key in required):
        raise ValueError("GPT-2 metric is missing or nonfinite")
    result = {key: final[key] for key in required}
    result.update(total=expected_samples, completed_batches=expected_batches,
                  loss_aggregation="mean_of_batch_losses",
                  evaluation_dataset_fingerprint=fingerprints[0])
    validate_site_reports(log, sites, result)
    return result


def validate_site_reports(log: str, sites: set[str] | None, result: dict[str, Any]) -> None:
    if sites is None:
        return
    reports = re.findall(r"^Calibration\[([^\]]+)\] values=(\d+),", log, re.MULTILINE)
    if len(reports) != len(sites) or {site for site, _ in reports} != sites:
        raise ValueError("calibration execution reports are missing or duplicated")
    if any(int(count) <= 0 for _, count in reports):
        raise ValueError("calibration site was not executed")
    result["calibration_site_count"] = len(sites)


def parse_evaluation(log: str, family: str, sites: set[str] | None = None) -> dict[str, Any]:
    config = MODEL_CONFIG[family]
    samples = config["evaluation_samples"]
    return (parse_gpt2(log, samples, sites)
            if config["evaluator_family"] == "gpt2" else parse_classification(log, samples, sites))


def update_progress(output: Path, family: str, phase: str, log_path: Path) -> None:
    if not log_path.exists():
        return
    text = log_path.read_text(errors="replace")
    evaluator_family = MODEL_CONFIG[family]["evaluator_family"]
    if phase == "collect" and evaluator_family == "gpt2":
        rows = json_event_rows(text, "calibration_progress")
        progress = rows[-1] if rows else None
    elif phase == "collect":
        matches = re.findall(
            r"Calibration progress: pass=(\d+)/(\d+) batch=(\d+)/(\d+) "
            r"samples=(\d+)/(\d+) elapsed_seconds=([0-9.]+)", text,
        )
        progress = None
        if matches:
            current_pass, passes, batch, batches, samples, expected, elapsed = matches[-1]
            completed = (int(current_pass) - 1) * int(batches) + int(batch)
            total_batches = int(passes) * int(batches)
            progress = {"pass": int(current_pass), "passes": int(passes),
                        "batch": completed, "total_batches": total_batches,
                        "observed_samples": int(samples), "expected_samples": int(expected),
                        "elapsed_seconds": float(elapsed),
                        "estimated_remaining_seconds": float(elapsed) *
                        (total_batches - completed) / completed}
    elif evaluator_family == "gpt2":
        rows = json_event_rows(text, "evaluation_progress")
        progress = rows[-1] if rows else None
    else:
        matches = re.findall(
            r"Evaluation progress: batch=(\d+)/(\d+) correct=(\d+) total=(\d+)/(\d+) "
            r"accuracy=([0-9.]+) elapsed_s=([0-9.]+) eta_s=([0-9.]+)", text,
        )
        progress = None
        if matches:
            batch, batches, correct, total, expected, accuracy, elapsed, eta = matches[-1]
            progress = {"batch": int(batch), "total_batches": int(batches), "correct": int(correct),
                        "evaluated_samples": int(total), "expected_samples": int(expected),
                        "accuracy": float(accuracy), "elapsed_seconds": float(elapsed),
                        "estimated_remaining_seconds": float(eta)}
    value = {"family": family, "phase": phase, "state": "running", "progress": progress,
             "updated_at": time.time()}
    runtime_files.atomic_json(output / "status.json", value)
    if progress and "batch" in progress:
        snapshot = output / "progress" / f"{phase}-batch-{int(progress['batch']):04d}.json"
        if not snapshot.exists():
            runtime_files.new_json(snapshot, value)


def run_phase(command: list[str], *, output: Path, family: str, phase: str,
              environment: dict[str, str], lock_fd: int) -> tuple[Path, float]:
    attempts = len(list(output.joinpath("logs").glob(phase + ".attempt-*.log"))) + 1
    log_path = output / "logs" / f"{phase}.attempt-{attempts:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("x") as log:
        child = subprocess.Popen(command, cwd=environment["SOURCE_ROOT"], env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=(lock_fd,))
        try:
            while child.poll() is None:
                update_progress(output, family, phase, log_path)
                time.sleep(5)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=15)
            raise
    update_progress(output, family, phase, log_path)
    if child.returncode:
        raise RuntimeError(f"{phase} exited with status {child.returncode}; logs preserved")
    return log_path, time.monotonic() - started


# @lat: [[text-calibration#Text Model Calibration#Complete Comparison Execution]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(MODEL_CONFIG), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--calibration-dataset-path", required=True)
    parser.add_argument("--calibration-dataset-fingerprint", required=True)
    parser.add_argument("--evaluation-dataset-path", required=True)
    parser.add_argument("--evaluation-dataset-fingerprint", required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--campaign-extra-local-gpus", action="store_true")
    parser.add_argument("--host-label", choices=("local", "ubai"), default="local")
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--cache-dir", default="/root/.cache/huggingface/datasets")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--reuse-calibration-from", type=Path)
    args = parser.parse_args()

    if args.host_label == "local" and socket.gethostname() != "baekryun-cuda129":
        raise ValueError("local comparison must run on baekryun")
    if args.host_label == "local" and args.gpu < 4 and not args.campaign_extra_local_gpus:
        raise ValueError("GPU 0 through 3 require the campaign-specific override")
    args.source_root = args.source_root.resolve(strict=True)
    config = MODEL_CONFIG[args.family]
    evaluator_family = config["evaluator_family"]
    source_hashes = source_identity(args.source_root, args.expected_commit, evaluator_family)
    model = Path(args.model_id).resolve(strict=True)
    calibration_dataset = Path(args.calibration_dataset_path).resolve(strict=True)
    evaluation_dataset = Path(args.evaluation_dataset_path).resolve(strict=True)
    args.model_id = str(model)
    args.calibration_dataset_path = str(calibration_dataset)
    args.evaluation_dataset_path = str(evaluation_dataset)
    output = args.output_root.resolve()
    run_tag = POWER_REUSE_TAG if args.reuse_calibration_from is not None else config["tag"]
    allowed = ARTIFACTS / "logs/conversion_comparison" / run_tag / "text"
    if output.parent != allowed or output.name != args.family:
        raise ValueError("output path differs from the fixed full comparison layout")
    runtime = ((args.runtime_root / args.family) if args.runtime_root is not None else
               ARTIFACTS / "runtime" / run_tag / "text" / args.family).resolve()
    if args.host_label == "local":
        required_runtime = (ARTIFACTS / "runtime" / run_tag / "text").resolve()
        if runtime.parent != required_runtime:
            raise ValueError("local runtime differs from the fixed comparison layout")
    else:
        if "SLURM_JOB_ID" not in os.environ or not runtime.is_relative_to(Path("/enroot")):
            raise ValueError("UBAI runtime must be a Slurm-owned path below /enroot")
    checkpoint_files_sha256 = checkpoint_identity(model)
    calibration_dataset_identity = dataset_identity(
        calibration_dataset, args.calibration_dataset_fingerprint, CALIBRATION_SAMPLES,
    )
    evaluation_dataset_identity = dataset_identity(
        evaluation_dataset, args.evaluation_dataset_fingerprint,
        config["evaluation_samples"],
    )
    calibration_reuse = None
    reused_sites: set[str] | None = None
    if args.reuse_calibration_from is not None:
        calibration_reuse, reused_sites = reused_calibration_evidence(
            args.reuse_calibration_from,
            family=args.family,
            evaluator_family=evaluator_family,
            checkpoint_files_sha256=checkpoint_files_sha256,
            calibration_dataset=calibration_dataset_identity,
            expected_sites=config["sites"],
        )
    commands = build_commands(args, output)
    if calibration_reuse is not None:
        commands.pop("collect")
    manifest = {
        "schema_version": 1, "tag": run_tag, "family": args.family,
        "evaluator_family": evaluator_family,
        "source_root": str(args.source_root), "source_commit": args.expected_commit,
        "source_hashes": source_hashes, "model_id": str(model),
        "checkpoint_files_sha256": checkpoint_files_sha256,
        "calibration_dataset": calibration_dataset_identity,
        "evaluation_dataset": evaluation_dataset_identity,
        "calibration_samples": CALIBRATION_SAMPLES,
        "evaluation_samples": config["evaluation_samples"],
        "batch_size": BATCH_SIZE, "theta": 40.0, "dtype": "float64",
        "tau_s": 1.0, "tau_m": 1.0, "calibration_seed": 0,
        "calibration_bins": 2048, "calibration_quantiles": [0.0, 1.0],
        "calibration_margin_fraction": 0.05, "text_calibration_policy_version": 1,
        "output_bounds_version": 3, "noise": False, "wandb": False,
        "tensorboard": False, "host_label": args.host_label, "physical_gpu": args.gpu,
        "activation": config["activation"],
        "activation_implementation": config.get("activation_implementation", "model_default"),
        "gelu_cubic_implementation": GELU_CUBIC_IMPLEMENTATION,
        "calibration_reuse": calibration_reuse,
        "campaign_extra_local_gpus": bool(args.campaign_extra_local_gpus),
        "commands": commands, "runtime_dir": str(runtime),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if canonical(json.loads(manifest_path.read_text())) != canonical(manifest):
            raise ValueError("existing model manifest differs")
    else:
        runtime_files.new_json(manifest_path, manifest)
    if calibration_reuse is not None:
        calibration_path = output / "calibration.json"
        source_calibration_path = Path(calibration_reuse["calibration_path"])
        if calibration_path.exists():
            if identity.sha256_file(calibration_path) != calibration_reuse["calibration_sha256"]:
                raise ValueError("existing reused calibration hash differs")
        else:
            shutil.copyfile(source_calibration_path, calibration_path)
    runtime.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True).strip()
    if filesystem in {"tmpfs", "ramfs"}:
        raise RuntimeError("comparison runtime must use a disk filesystem")

    sys.path.insert(0, str(args.source_root))
    lock_path = ARTIFACTS / "runtime/gpu-locks" / f"gpu-{args.gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.host_label == "local":
            for check in range(2):
                sample = local_gpu.gpu_activity(gpu_ids=(args.gpu,))[args.gpu]
                if not local_gpu.gpu_available(sample):
                    raise RuntimeError(f"GPU {args.gpu} is occupied")
                if check == 0:
                    time.sleep(10)
            visible_gpu = str(args.gpu)
        else:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not visible or "," in visible:
                raise ValueError("each UBAI evaluator must receive exactly one GPU")
            visible_gpu = visible
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=visible_gpu, WANDB_MODE="disabled",
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TMPDIR=str(runtime),
            TMP=str(runtime), TEMP=str(runtime), WANDB_DIR=str(runtime),
            SOURCE_ROOT=str(args.source_root),
            PYTHONPATH=os.pathsep.join(map(str, (
                args.source_root, args.source_root / "src/transformers/src",
                args.source_root / "src/spikingjelly",
            ))),
        )
        state: dict[str, Any] = {"state": "running", "family": args.family, "phases": {}}
        if calibration_reuse is not None:
            state["calibration_reuse"] = calibration_reuse
        runtime_files.atomic_json(output / "result.json", state)
        try:
            sites: set[str] | None = reused_sites
            phases = ("ann", "snn") if calibration_reuse is not None else ("collect", "ann", "snn")
            for phase in phases:
                verify_dataset_snapshot(manifest["calibration_dataset"])
                verify_dataset_snapshot(manifest["evaluation_dataset"])
                phase_path = output / "phases" / f"{phase}.json"
                if phase_path.exists():
                    phase_result = json.loads(phase_path.read_text())
                    log_path = output / phase_result["log_file"]
                    if identity.sha256_file(log_path) != phase_result["log_sha256"]:
                        raise ValueError("completed phase log hash differs")
                    if phase == "collect":
                        _, sites = parse_sites(output / "calibration.json", config["sites"])
                    else:
                        phase_result["metrics"] = parse_evaluation(
                            log_path.read_text(), args.family, sites if phase == "snn" else None,
                        )
                    state["phases"][phase] = phase_result
                    continue
                if phase == "collect" and (output / "calibration.json").exists():
                    rejected = output / "rejected" / f"orphan-collect-{time.time_ns()}"
                    rejected.mkdir(parents=True)
                    os.replace(output / "calibration.json", rejected / "calibration.json")
                source_identity(args.source_root, args.expected_commit, evaluator_family)
                log_path, elapsed = run_phase(
                    commands[phase], output=output, family=args.family, phase=phase,
                    environment=environment, lock_fd=lock.fileno(),
                )
                log_text = log_path.read_text()
                phase_result = {"phase": phase, "elapsed_seconds": elapsed,
                                "log_file": str(log_path.relative_to(output)),
                                "log_sha256": identity.sha256_file(log_path)}
                if phase == "collect":
                    if log_text.count("Saved calibration artifact") != 1:
                        raise ValueError("calibration completion marker is missing or duplicated")
                    metadata, sites = parse_sites(output / "calibration.json", config["sites"])
                    phase_result.update(
                        calibration_sha256=identity.sha256_file(output / "calibration.json"),
                                        calibration_metadata=metadata,
                                        calibration_site_count=len(sites))
                else:
                    phase_result["metrics"] = parse_evaluation(
                        log_text, args.family, sites if phase == "snn" else None,
                    )
                runtime_files.new_json(phase_path, phase_result)
                state["phases"][phase] = phase_result
                runtime_files.atomic_json(output / "result.json", state)
                verify_dataset_snapshot(manifest["calibration_dataset"])
                verify_dataset_snapshot(manifest["evaluation_dataset"])
            ann = state["phases"]["ann"]["metrics"]
            snn = state["phases"]["snn"]["metrics"]
            if ann["evaluation_dataset_fingerprint"] != snn["evaluation_dataset_fingerprint"]:
                raise ValueError("ANN and SNN evaluation dataset fingerprints differ")
            state["state"] = "complete"
            state["calibration_sha256"] = (
                calibration_reuse["calibration_sha256"] if calibration_reuse is not None
                else state["phases"]["collect"]["calibration_sha256"]
            )
            source_identity(args.source_root, args.expected_commit, evaluator_family)
            verify_dataset_snapshot(manifest["calibration_dataset"])
            verify_dataset_snapshot(manifest["evaluation_dataset"])
            runtime_files.atomic_json(output / "result.json", state)
            runtime_files.atomic_json(output / "status.json", {"state": "complete", "family": args.family,
                                                   "updated_at": time.time()})
            print(canonical(state), flush=True)
        except BaseException as error:
            state.update(state="failed", error_type=type(error).__name__, error=str(error))
            runtime_files.atomic_json(output / "result.json", state)
            runtime_files.atomic_json(output / "status.json", {"state": "failed", "family": args.family,
                                                   "error_type": type(error).__name__,
                                                   "error": str(error), "updated_at": time.time()})
            raise


if __name__ == "__main__":
    main()
