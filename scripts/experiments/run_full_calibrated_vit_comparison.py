#!/usr/bin/env python3
"""Run one complete local-range calibrated ViT comparison on a direct GPU host."""

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

from scripts.runtime import ann_baseline
from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.runtime import local_gpu
from utils.transformers.calibration import (
    OPERATOR_BACKED_OUTPUT_HEAD_VERSION,
    VIT_CALIBRATION_POLICY_VERSION,
)


ARTIFACTS = Path(os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", "/data/delayed-temporal/artifacts"))
RUNTIME_ARTIFACTS = Path(
    os.environ.get("DELAYED_TEMPORAL_RUNTIME_ROOT", str(ARTIFACTS / "runtime"))
)
DEFAULT_ANN_BASELINE_CACHE_ROOT = Path(
    os.environ.get(
        "DELAYED_TEMPORAL_ANN_BASELINE_ROOT",
        "/data/delayed-temporal/artifacts/logs/ann_baselines/v1",
    )
)
TAG = "conversion_comparison_end_to_end_local_ranges_float64_v1"
MODEL_CONFIG = {
    "cifar10_vit_small": {"dataset_id": "cifar10", "split": "test", "samples": 10_000},
    "imagenet_vit_small": {"dataset_id": "imagenet-1k", "split": "validation", "samples": 5_000},
    "imagenet_vit_base": {"dataset_id": "imagenet-1k", "split": "validation", "samples": 5_000},
    "imagenet_vit_large": {"dataset_id": "imagenet-1k", "split": "validation", "samples": 5_000},
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_identity(source: Path, expected_commit: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("ViT comparison requires a full source commit")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if head != expected_commit or dirty:
        raise ValueError("ViT comparison requires the clean frozen source commit")
    paths = sorted(source.joinpath("utils").rglob("*.py")) + [
        source / "scripts/evaluation/error_analysis_vit.py",
        source / "scripts/analysis/evaluate_calibrated_vit.py",
        source / "scripts/analysis/gelu_cubic_phi_nl_vit.py",
        source / "scripts/runtime/ann_baseline.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(source)): identity.sha256_file(path) for path in sorted(set(paths))}


def dataset_identity(path: Path, fingerprint: str, samples: int) -> dict[str, Any]:
    from datasets import load_from_disk

    dataset = load_from_disk(path)
    if len(dataset) != samples or str(dataset._fingerprint) != fingerprint:
        raise ValueError("self-contained ViT dataset identity differs")
    return {"path": str(path), "fingerprint": fingerprint, "samples": samples}


def dense_baseline_identity(
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
    evaluation_dataset: dict[str, Any],
) -> dict[str, Any]:
    """Identify the dense ViT evaluation independently of converted-model code."""
    config = MODEL_CONFIG[args.model_key]
    preprocessing_sha256 = (
        identity.sha256_file(args.image_preprocessing_config)
        if args.image_preprocessing_config is not None else None
    )
    return ann_baseline.build_identity(
        model_key=args.model_key,
        model_family="vit",
        checkpoint_identity=checkpoint_sha256,
        evaluation_dataset=evaluation_dataset,
        evaluation_settings={
            "backend": "hf",
            "metric": "top1_accuracy",
            "split": config["split"],
            "evaluation_samples": config["samples"],
            "batch_size": args.batch_size,
            "precision": "float64",
            "image_preprocessing_sha256": preprocessing_sha256,
            "quick_test": config["dataset_id"] == "imagenet-1k",
        },
    )


def common_evaluator_args(args: argparse.Namespace, output: Path) -> list[str]:
    model = MODEL_CONFIG[args.model_key]
    values = [
        "--model_id", str(args.model_id), "--dataset_id", model["dataset_id"],
        "--evaluation-dataset-path", str(args.evaluation_dataset_path),
        "--evaluation-split", model["split"], "--device", "cuda", "--precision", "float64",
        "--batch_size", str(args.batch_size), "--source-commit", args.expected_commit,
        "--checkpoint-sha256", args.checkpoint_sha256, "--no-tensorboard", "--report-clamp-stats",
        "--spiking-layernorm", "--spiking-ln-mul", "--spiking-ln-log", "--spiking-ln-expdiff",
        "--spiking-attention", "--spiking-mlp", "--no-spiking-mlp-exact-gelu",
        "--no-gaussian-time-noise", "--time-noise-seed", "0", "--time-noise-std-frac", "0",
        "--time-noise-mean", "0", "--time-noise-deadline-margin-std", "0",
        "--no-mismatch-enabled", "--mismatch-range-std-frac", "0", "--mismatch-seed", "0",
        "--weight-noise-std", "0", "--bias-noise-std", "0",
    ]
    if args.image_preprocessing_config:
        values += ["--image-preprocessing-config", str(args.image_preprocessing_config)]
    if model["dataset_id"] == "imagenet-1k":
        values += ["--quick-test"]
    return values


def build_commands(args: argparse.Namespace, output: Path) -> dict[str, list[str]]:
    evaluator = args.source_root / "scripts/evaluation/error_analysis_vit.py"
    wrapper = args.source_root / "scripts/analysis/evaluate_calibrated_vit.py"
    common = common_evaluator_args(args, output)
    calibration = [
        "--calibration-path", str(output / "calibration.json"), "--calibration-samples", "5000",
        "--calibration-seed", "0", "--calibration-bins", "2048",
        "--calibration-lower-quantile", "0", "--calibration-upper-quantile", "1",
        "--calibration-margin-fraction", "0.05",
    ]
    wrapped = [
        args.python_bin, "-u", str(wrapper), "--source-root", str(args.source_root),
        "--calibration-dataset-path", str(args.calibration_dataset_path),
        "--calibration-dataset-fingerprint", args.calibration_dataset_fingerprint,
        "--gelu-cubic-implementation", "phi_nl_psi_ed", "--gelu-cubic-floor", "1e-5",
    ]
    return {
        "collect": [*wrapped, "--experiment_name", output.name + "_collect", *common,
                    "--model_backend", "spiking", *calibration, "--calibration-mode", "collect"],
        "ann": [args.python_bin, "-u", str(evaluator), "--experiment_name", output.name + "_ann",
                *common, "--model_backend", "hf", "--calibration-mode", "none"],
        "snn": [*wrapped, "--experiment_name", output.name + "_snn", *common,
                "--model_backend", "spiking", *calibration, "--calibration-mode", "validate"],
    }


def parse_metric(log: str, expected_samples: int) -> dict[str, Any]:
    correct = re.findall(r"^Correct: (\d+)\s*$", log, re.MULTILINE)
    totals = re.findall(r"^Evaluated samples: (\d+)\s*$", log, re.MULTILINE)
    digests = re.findall(r"^Prediction SHA256: ([0-9a-f]{64})\s*$", log, re.MULTILINE)
    accuracy = re.findall(r"^Accuracy: ([0-9.]+)\s*$", log, re.MULTILINE)
    if not all(len(rows) == 1 for rows in (correct, totals, digests, accuracy)):
        raise ValueError("ViT result markers are missing or duplicated")
    total = int(totals[0])
    value = float(accuracy[0])
    if total != expected_samples or not math.isclose(
        int(correct[0]) / total, value, rel_tol=0.0, abs_tol=5e-9,
    ):
        raise ValueError("ViT result count or accuracy differs")
    return {"correct": int(correct[0]), "total": total, "accuracy": value,
            "prediction_sha256": digests[0]}


def calibration_sites(path: Path) -> set[str]:
    table = json.loads(path.read_text())
    metadata = table["metadata"]
    options = dict(metadata["model_options"])
    if (
        metadata["dtype"] != "float64"
        or options.get("vit_calibration_policy_version") != VIT_CALIBRATION_POLICY_VERSION
        or options.get("operator_backed_output_head_version")
        != OPERATOR_BACKED_OUTPUT_HEAD_VERSION
    ):
        raise ValueError("ViT calibration policy differs")
    if options["output_bounds_version"] != 4 or "theta" in metadata or "theta" in options:
        raise ValueError("calibration artifact contains a legacy global range")
    rows = table["layers"]
    sites = {row["module_name"] + "/" + row["tensor_name"] for row in rows}
    if not rows or len(rows) != len(sites):
        raise ValueError("calibration site population is empty or duplicated")
    return sites


def validate_snn_sites(log: str, sites: set[str]) -> None:
    rows = re.findall(r"^Calibration\[([^\]]+)\] values=(\d+),", log, re.MULTILINE)
    if len(rows) != len(sites) or {site for site, _ in rows} != sites:
        raise ValueError("calibration execution reports are missing or duplicated")
    if any(int(count) <= 0 for _, count in rows):
        raise ValueError("a calibration site did not execute")


def run_phase(command: list[str], *, output: Path, phase: str,
              environment: dict[str, str], lock_fd: int) -> tuple[Path, float]:
    attempt = len(list(output.joinpath("logs").glob(phase + ".attempt-*.log"))) + 1
    log_path = output / "logs" / f"{phase}.attempt-{attempt:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("x") as log:
        child = subprocess.Popen(command, cwd=environment["SOURCE_ROOT"], env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 pass_fds=(lock_fd,))
        try:
            while child.poll() is None:
                runtime_files.atomic_json(output / "status.json", {
                    "state": "running", "phase": phase, "pid": child.pid,
                    "elapsed_seconds": time.monotonic() - started, "updated_at": time.time(),
                })
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
    if child.returncode:
        raise RuntimeError(f"{phase} exited with status {child.returncode}; logs preserved")
    return log_path, time.monotonic() - started


# @lat: [[evaluation#Evaluation and Verification#Local-Range Paper Re-evaluation]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-key", choices=tuple(MODEL_CONFIG), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--calibration-dataset-path", type=Path, required=True)
    parser.add_argument("--calibration-dataset-fingerprint", required=True)
    parser.add_argument("--evaluation-dataset-path", type=Path, required=True)
    parser.add_argument("--evaluation-dataset-fingerprint", required=True)
    parser.add_argument("--image-preprocessing-config", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--host-label", choices=("local", "poseidon"), required=True)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--ann-baseline-cache-root",
        type=Path,
        default=DEFAULT_ANN_BASELINE_CACHE_ROOT,
    )
    args = parser.parse_args()

    expected_host = "baekryun-cuda129" if args.host_label == "local" else "poseidon1"
    if socket.gethostname() != expected_host:
        raise ValueError(f"comparison requires host {expected_host}")
    args.source_root = args.source_root.resolve(strict=True)
    args.model_id = args.model_id.resolve(strict=True)
    args.calibration_dataset_path = args.calibration_dataset_path.resolve(strict=True)
    args.evaluation_dataset_path = args.evaluation_dataset_path.resolve(strict=True)
    if args.image_preprocessing_config:
        args.image_preprocessing_config = args.image_preprocessing_config.resolve(strict=True)
    source_hashes = source_identity(args.source_root, args.expected_commit)
    args.checkpoint_sha256 = identity.artifact_identity(args.model_id)["aggregate_sha256"]
    config = MODEL_CONFIG[args.model_key]
    calibration_dataset = dataset_identity(
        args.calibration_dataset_path, args.calibration_dataset_fingerprint, 5_000,
    )
    evaluation_dataset = dataset_identity(
        args.evaluation_dataset_path, args.evaluation_dataset_fingerprint,
        50_000 if config["dataset_id"] == "imagenet-1k" else config["samples"],
    )
    dense_identity = dense_baseline_identity(
        args,
        checkpoint_sha256=args.checkpoint_sha256,
        evaluation_dataset=evaluation_dataset,
    )
    dense_cache_root = args.ann_baseline_cache_root.expanduser().resolve()
    output = args.output_root.resolve()
    required_output = (ARTIFACTS / "logs/conversion_comparison" / TAG / "vit" / args.model_key).resolve()
    if output != required_output:
        raise ValueError("output path differs from the fixed comparison layout")
    runtime = args.runtime_root.resolve()
    required_runtime = (RUNTIME_ARTIFACTS / TAG / "vit" / args.model_key).resolve()
    if runtime != required_runtime:
        raise ValueError("runtime path differs from the fixed comparison layout")
    output.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True).strip()
    if filesystem in {"tmpfs", "ramfs"}:
        raise RuntimeError("comparison runtime must use a disk filesystem")
    commands = build_commands(args, output)
    manifest = {
        "schema_version": 1, "tag": TAG, "model_key": args.model_key,
        "source_commit": args.expected_commit, "source_hashes": source_hashes,
        "checkpoint_path": str(args.model_id), "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_dataset": calibration_dataset, "evaluation_dataset": evaluation_dataset,
        "evaluation_samples": config["samples"], "batch_size": args.batch_size,
        "range_contract": "operator_local_end_to_end_v1",
        "vit_calibration_policy_version": VIT_CALIBRATION_POLICY_VERSION,
        "operator_backed_output_head_version": OPERATOR_BACKED_OUTPUT_HEAD_VERSION,
        "output_bounds_version": 4, "dtype": "float64", "noise": False,
        "host_label": args.host_label, "physical_gpu": args.gpu,
        "runtime_dir": str(runtime), "commands": commands,
        "ann_baseline_identity": dense_identity,
        "ann_baseline_identity_sha256": ann_baseline.identity_sha256(dense_identity),
        "ann_baseline_cache_root": str(dense_cache_root),
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if canonical(json.loads(manifest_path.read_text())) != canonical(manifest):
            raise ValueError("existing ViT manifest differs")
    else:
        runtime_files.new_json(manifest_path, manifest)

    lock_path = ARTIFACTS / "runtime/gpu-locks" / f"{args.host_label}-gpu-{args.gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for check in range(2):
            sample = local_gpu.gpu_activity(gpu_ids=(args.gpu,))[args.gpu]
            if not local_gpu.gpu_available(sample):
                raise RuntimeError(f"GPU {args.gpu} is occupied")
            if check == 0:
                time.sleep(10)
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="disabled",
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
        state: dict[str, Any] = {"state": "running", "model_key": args.model_key, "phases": {}}
        runtime_files.atomic_json(output / "result.json", state)
        try:
            sites: set[str] | None = None
            for phase in ("collect", "ann", "snn"):
                phase_path = output / "phases" / f"{phase}.json"
                if phase_path.exists():
                    phase_result = json.loads(phase_path.read_text())
                    log_path = output / phase_result["log_file"]
                    if identity.sha256_file(log_path) != phase_result["log_sha256"]:
                        raise ValueError("completed phase log hash differs")
                    if phase == "collect":
                        sites = calibration_sites(output / "calibration.json")
                    elif phase == "ann":
                        phase_result["metrics"] = parse_metric(log_path.read_text(), config["samples"])
                        ann_baseline.publish(
                            cache_root=dense_cache_root,
                            baseline_identity=dense_identity,
                            source_log=log_path,
                            metrics=phase_result["metrics"],
                            elapsed_seconds=phase_result["elapsed_seconds"],
                        )
                    state["phases"][phase] = phase_result
                    continue
                if phase == "collect" and (output / "calibration.json").exists():
                    rejected = output / "rejected" / f"orphan-collect-{time.time_ns()}"
                    rejected.mkdir(parents=True)
                    os.replace(output / "calibration.json", rejected / "calibration.json")
                if phase == "ann":
                    cached = ann_baseline.load(
                        cache_root=dense_cache_root,
                        baseline_identity=dense_identity,
                        parse_metrics=lambda text: parse_metric(text, config["samples"]),
                    )
                    if cached is not None:
                        phase_result = ann_baseline.materialize_phase(
                            record=cached[0], source_log=cached[1], output=output,
                        )
                        runtime_files.new_json(phase_path, phase_result)
                        state["phases"][phase] = phase_result
                        runtime_files.atomic_json(output / "result.json", state)
                        continue
                source_identity(args.source_root, args.expected_commit)
                log_path, elapsed = run_phase(
                    commands[phase], output=output, phase=phase,
                    environment=environment, lock_fd=lock.fileno(),
                )
                log_text = log_path.read_text(errors="replace")
                phase_result: dict[str, Any] = {
                    "phase": phase, "elapsed_seconds": elapsed,
                    "log_file": str(log_path.relative_to(output)),
                    "log_sha256": identity.sha256_file(log_path),
                }
                if phase == "collect":
                    sites = calibration_sites(output / "calibration.json")
                    phase_result.update(
                        calibration_sha256=identity.sha256_file(output / "calibration.json"),
                        calibration_site_count=len(sites),
                    )
                else:
                    phase_result["metrics"] = parse_metric(log_text, config["samples"])
                    if phase == "ann":
                        record = ann_baseline.publish(
                            cache_root=dense_cache_root,
                            baseline_identity=dense_identity,
                            source_log=log_path,
                            metrics=phase_result["metrics"],
                            elapsed_seconds=elapsed,
                        )
                        phase_result["ann_baseline_identity_sha256"] = record["identity_sha256"]
                    if phase == "snn":
                        if sites is None:
                            raise RuntimeError("calibration sites are unavailable")
                        validate_snn_sites(log_text, sites)
                        phase_result["calibration_site_count"] = len(sites)
                runtime_files.new_json(phase_path, phase_result)
                state["phases"][phase] = phase_result
                runtime_files.atomic_json(output / "result.json", state)
            if state["phases"]["ann"]["metrics"]["total"] != state["phases"]["snn"]["metrics"]["total"]:
                raise ValueError("ANN and SNN totals differ")
            state.update(state="complete", calibration_sha256=state["phases"]["collect"]["calibration_sha256"])
            source_identity(args.source_root, args.expected_commit)
            runtime_files.atomic_json(output / "result.json", state)
            runtime_files.atomic_json(output / "status.json", {
                "state": "complete", "model_key": args.model_key, "updated_at": time.time(),
            })
            print(canonical(state), flush=True)
        except BaseException as error:
            state.update(state="failed", error_type=type(error).__name__, error=str(error))
            runtime_files.atomic_json(output / "result.json", state)
            runtime_files.atomic_json(output / "status.json", {
                "state": "failed", "model_key": args.model_key,
                "error_type": type(error).__name__, "error": str(error), "updated_at": time.time(),
            })
            raise


if __name__ == "__main__":
    main()
