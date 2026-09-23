#!/usr/bin/env python3
"""Evaluate one authenticated ViT-B local-window timing-noise condition."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
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
from scripts.experiments.run_full_calibrated_vit_comparison import (
    calibration_sites,
    parse_metric,
    source_identity,
    validate_snn_sites,
)


ARTIFACTS = Path(os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", "/data/delayed-temporal/artifacts"))
TAG = "vit_base_end_to_end_local_range_timing_noise_float64_v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_physical_counts(log: str) -> dict[str, Any]:
    pattern = re.compile(
        r"^Gaussian\[([^\]]+)\] events=(\d+), misses=(\d+) \(rate=([0-9.eE+-]+)\), "
        r"deadline_events=(\d+) \(rate=([0-9.eE+-]+)\), .*?outputs=(\d+), "
        r"underflows=(\d+) \(rate=([0-9.eE+-]+)\), overflows=(\d+) "
        r"\(rate=([0-9.eE+-]+)\)$",
        re.MULTILINE,
    )
    rows = []
    for match in pattern.finditer(log):
        rows.append({
            "site": match.group(1), "events": int(match.group(2)), "misses": int(match.group(3)),
            "deadline_events": int(match.group(5)), "outputs": int(match.group(7)),
            "underflows": int(match.group(8)), "overflows": int(match.group(10)),
        })
    if not rows:
        raise ValueError("Gaussian physical-count reports are missing")
    pooled = {key: sum(row[key] for row in rows) for key in (
        "events", "misses", "deadline_events", "outputs", "underflows", "overflows",
    )}
    pooled.update(
        miss_rate=pooled["misses"] / pooled["events"],
        deadline_event_rate=pooled["deadline_events"] / pooled["events"],
        underflow_rate=pooled["underflows"] / pooled["outputs"],
        overflow_rate=pooled["overflows"] / pooled["outputs"],
        site_count=len(rows), sites=rows,
    )
    return pooled


def completed_calibration(path: Path, expected_commit: str) -> tuple[Path, str, set[str]]:
    result_path = path / "result.json"
    manifest_path = path / "manifest.json"
    calibration_path = path / "calibration.json"
    if not all(item.is_file() for item in (result_path, manifest_path, calibration_path)):
        raise ValueError("ViT-B calibration source is incomplete")
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    calibration_sha256 = identity.sha256_file(calibration_path)
    if (
        result.get("state") != "complete"
        or manifest.get("model_key") != "imagenet_vit_base"
        or manifest.get("source_commit") != expected_commit
        or result.get("calibration_sha256") != calibration_sha256
    ):
        raise ValueError("ViT-B calibration source identity differs")
    return calibration_path, calibration_sha256, calibration_sites(calibration_path)


# @lat: [[noise#Noise Model#Local-Window Timing Noise Sweep]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--calibration-dataset-path", type=Path, required=True)
    parser.add_argument("--calibration-dataset-fingerprint", required=True)
    parser.add_argument("--evaluation-dataset-path", type=Path, required=True)
    parser.add_argument("--image-preprocessing-config", type=Path, required=True)
    parser.add_argument("--time-noise-std-frac", type=float, required=True)
    parser.add_argument("--deadline-margin-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()

    if socket.gethostname() != "poseidon1":
        raise ValueError("noise conditions are assigned to poseidon1")
    if not math.isfinite(args.time_noise_std_frac) or args.time_noise_std_frac <= 0:
        raise ValueError("timing-noise fraction must be finite and positive")
    if not math.isfinite(args.deadline_margin_ratio) or args.deadline_margin_ratio < 0:
        raise ValueError("deadline-margin ratio must be finite and non-negative")
    if not re.fullmatch(r"[a-z0-9_.-]+", args.run_id):
        raise ValueError("run identifier contains unsupported characters")
    args.source_root = args.source_root.resolve(strict=True)
    args.model_id = args.model_id.resolve(strict=True)
    args.calibration_source = args.calibration_source.resolve(strict=True)
    args.calibration_dataset_path = args.calibration_dataset_path.resolve(strict=True)
    args.evaluation_dataset_path = args.evaluation_dataset_path.resolve(strict=True)
    args.image_preprocessing_config = args.image_preprocessing_config.resolve(strict=True)
    source_hashes = source_identity(args.source_root, args.expected_commit)
    if identity.artifact_identity(args.model_id)["aggregate_sha256"] != args.checkpoint_sha256:
        raise ValueError("checkpoint contents differ")
    calibration_path, calibration_sha256, sites = completed_calibration(
        args.calibration_source, args.expected_commit,
    )
    output = (ARTIFACTS / "logs/noise_scan" / TAG / "runs" / args.run_id).resolve()
    runtime = (ARTIFACTS / "runtime" / TAG / args.run_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True).strip()
    if filesystem in {"tmpfs", "ramfs"}:
        raise RuntimeError("noise runtime must use a disk filesystem")
    local_calibration = output / "calibration.json"
    if local_calibration.exists():
        if identity.sha256_file(local_calibration) != calibration_sha256:
            raise ValueError("existing calibration copy differs")
    else:
        os.link(calibration_path, local_calibration)
    wrapper = args.source_root / "scripts/analysis/evaluate_calibrated_vit.py"
    command = [
        args.python_bin, "-u", str(wrapper), "--source-root", str(args.source_root),
        "--calibration-dataset-path", str(args.calibration_dataset_path),
        "--calibration-dataset-fingerprint", args.calibration_dataset_fingerprint,
        "--gelu-cubic-implementation", "phi_nl_psi_ed", "--gelu-cubic-floor", "1e-5",
        "--experiment_name", args.run_id, "--device", "cuda", "--model_backend", "spiking",
        "--model_id", str(args.model_id), "--dataset_id", "imagenet-1k",
        "--evaluation-dataset-path", str(args.evaluation_dataset_path),
        "--evaluation-split", "validation", "--image-preprocessing-config", str(args.image_preprocessing_config),
        "--quick-test", "--batch_size", "32", "--precision", "float64",
        "--source-commit", args.expected_commit, "--checkpoint-sha256", args.checkpoint_sha256,
        "--no-tensorboard", "--report-clamp-stats", "--spiking-layernorm", "--spiking-ln-mul",
        "--spiking-ln-log", "--spiking-ln-expdiff", "--spiking-attention", "--spiking-mlp",
        "--no-spiking-mlp-exact-gelu", "--calibration-mode", "validate",
        "--calibration-path", str(local_calibration), "--calibration-samples", "5000",
        "--calibration-seed", "0", "--calibration-bins", "2048",
        "--calibration-lower-quantile", "0", "--calibration-upper-quantile", "1",
        "--calibration-margin-fraction", "0.05", "--gaussian-time-noise",
        "--time-noise-std-frac", repr(args.time_noise_std_frac), "--time-noise-mean", "0",
        "--time-noise-deadline-margin-std", repr(args.deadline_margin_ratio),
        "--time-noise-seed", str(args.seed), "--no-mismatch-enabled",
        "--mismatch-range-std-frac", "0", "--mismatch-seed", "0",
        "--weight-noise-std", "0", "--bias-noise-std", "0",
    ]
    manifest = {
        "schema_version": 1, "tag": TAG, "run_id": args.run_id,
        "source_commit": args.expected_commit, "source_hashes": source_hashes,
        "checkpoint_path": str(args.model_id), "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_source": str(args.calibration_source),
        "calibration_sha256": calibration_sha256,
        "calibration_source_result_sha256": identity.sha256_file(args.calibration_source / "result.json"),
        "evaluation_dataset_path": str(args.evaluation_dataset_path),
        "calibration_dataset_fingerprint": args.calibration_dataset_fingerprint,
        "time_noise_std_fraction": args.time_noise_std_frac,
        "deadline_margin_sigma_ratio": args.deadline_margin_ratio,
        "seed": args.seed, "dtype": "float64",
        "range_contract": "operator_local_end_to_end_v1",
        "timing_noise_contract": "local_encoder_window_fraction_v1",
        "physical_gpu": args.gpu, "runtime_dir": str(runtime), "command": command,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if canonical(json.loads(manifest_path.read_text())) != canonical(manifest):
            raise ValueError("existing noise manifest differs")
    else:
        runtime_files.new_json(manifest_path, manifest)
    result_path = output / "result.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("state") == "complete":
        print(result_path.read_text(), flush=True)
        return

    lock_path = ARTIFACTS / "runtime/gpu-locks" / f"poseidon-gpu-{args.gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for check in range(2):
            if not local_gpu.gpu_available(local_gpu.gpu_activity(gpu_ids=(args.gpu,))[args.gpu]):
                raise RuntimeError(f"GPU {args.gpu} is occupied")
            if check == 0:
                time.sleep(10)
        attempt = len(list(output.joinpath("logs").glob("evaluation.attempt-*.log"))) + 1
        log_path = output / "logs" / f"evaluation.attempt-{attempt:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), WANDB_MODE="disabled",
            HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TMPDIR=str(runtime),
            TMP=str(runtime), TEMP=str(runtime), WANDB_DIR=str(runtime),
            PYTHONPATH=os.pathsep.join(map(str, (
                args.source_root, args.source_root / "src/transformers/src",
                args.source_root / "src/spikingjelly",
            ))),
        )
        started = time.monotonic()
        runtime_files.atomic_json(result_path, {"state": "running", "run_id": args.run_id})
        with log_path.open("x") as log:
            completed = subprocess.run(command, cwd=args.source_root, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            failure = {"state": "failed", "run_id": args.run_id,
                       "exit_status": completed.returncode, "log_file": str(log_path.relative_to(output))}
            runtime_files.atomic_json(result_path, failure)
            raise RuntimeError(f"noise evaluator exited with status {completed.returncode}")
        log_text = log_path.read_text(errors="replace")
        metrics = parse_metric(log_text, 5_000)
        validate_snn_sites(log_text, sites)
        metrics["physical_counts"] = parse_physical_counts(log_text)
        result = {
            "state": "complete", "run_id": args.run_id,
            "elapsed_seconds": time.monotonic() - started,
            "log_file": str(log_path.relative_to(output)), "log_sha256": identity.sha256_file(log_path),
            "metrics": metrics,
        }
        source_identity(args.source_root, args.expected_commit)
        runtime_files.atomic_json(result_path, result)
        print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
