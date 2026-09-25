#!/usr/bin/env python3
"""Run one authenticated ViT-B cumulative encoder-block timing-noise condition."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_full_calibrated_vit_comparison import (
    calibration_sites,
    parse_metric,
    source_identity,
    validate_snn_sites,
)
from scripts.experiments.run_vit_local_range_noise_condition import completed_calibration
from scripts.runtime import files as runtime_files
from scripts.runtime import identity, local_gpu


ARTIFACTS = Path(
    os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", str(ROOT / "artifacts"))
)
TAG = "vit_base_bss2_encoder_depth_noise_float64_v1"
HARDWARE_SUMMARY_SHA256 = (
    "639a908ac4b86440ef706afcd98467117cbf0f1e5424ca3e8de5f4ef8802f6c0"
)
CONDITIONS = ("clean", "screening-selected-coordinate", "screening-median")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hardware_conditions(path: Path) -> dict[str, dict[str, Any]]:
    """Read the two fixed measured conditions from the authenticated summary."""
    if identity.sha256_file(path) != HARDWARE_SUMMARY_SHA256:
        raise ValueError("BrainScaleS-2 encoder summary checksum differs")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("BrainScaleS-2 encoder summary schema differs")
    np_data, nl_data = payload["phi_np"], payload["phi_nl"]
    best_np = np_data["best_coordinate_remeasurement"]
    best_nl = nl_data["best_calibration_selected"]
    median_np = np_data["screening_median"]
    median_nl = nl_data["screening_median"]
    timing = {
        "phi_np_encoding_window_s": tuple(np_data["encoding_window_s"]),
        "phi_np_observation_deadline_s": float(np_data["observation_deadline_s"]),
        "phi_nl_encoding_window_s": tuple(nl_data["encoding_window_s"]),
        "phi_nl_observation_deadline_s": float(nl_data["observation_deadline_s"]),
    }
    conditions = {
        "screening-selected-coordinate": {
            "linear_time_std_fraction": float(best_np["validation_rt"]),
            "log_time_std_fraction": float(best_nl["validation_rt"]),
            "phi_np_coordinate": int(best_np["physical_coordinate"]),
            "phi_nl_coordinate": int(best_nl["physical_coordinate"]),
            "interpretation": "screening calibration-selected coordinate remeasurement",
            **timing,
        },
        "screening-median": {
            "linear_time_std_fraction": float(median_np["validation_rt"]),
            "log_time_std_fraction": float(median_nl["validation_rt"]),
            "phi_np_coordinate": int(median_np["physical_coordinate"]),
            "phi_nl_coordinate": int(median_nl["physical_coordinate"]),
            "interpretation": "median among usable screened coordinates",
            **timing,
        },
    }
    for condition in conditions.values():
        for key in ("linear_time_std_fraction", "log_time_std_fraction"):
            value = float(condition[key])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("measured timing-noise fraction must be positive")
    return conditions


def dataset_identity(
    path: Path,
    fingerprint: str,
    *,
    evaluation_samples: int,
) -> dict[str, Any]:
    """Authenticate the complete saved validation population before prefixing it."""
    from datasets import Dataset, load_from_disk

    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError("evaluation artifact must be one saved Dataset")
    if len(dataset) != 50_000 or str(dataset._fingerprint) != fingerprint:
        raise ValueError("ImageNet validation artifact identity differs")
    selected = dataset.select(range(evaluation_samples))
    return {
        "path": str(path),
        "fingerprint": fingerprint,
        "samples": len(dataset),
        "selection": "prefix",
        "selection_start": 0,
        "selection_stop": evaluation_samples,
        "selected_fingerprint": str(selected._fingerprint),
    }


def parse_scoped_counts(log: str, first_block_count: int) -> dict[str, Any]:
    """Parse Gaussian counters and prove that only the selected prefix executed."""
    pattern = re.compile(
        r"^Gaussian\[([^\]]+)\] events=(\d+), misses=(\d+) \(rate=([0-9.eE+-]+)\), "
        r"deadline_events=(\d+) \(rate=([0-9.eE+-]+)\), .*?outputs=(\d+), "
        r"underflows=(\d+) \(rate=([0-9.eE+-]+)\), overflows=(\d+) "
        r"\(rate=([0-9.eE+-]+)\)$",
        re.MULTILINE,
    )
    rows: list[dict[str, Any]] = []
    block_sites: dict[int, list[dict[str, Any]]] = {}
    for match in pattern.finditer(log):
        site = match.group(1)
        scoped = re.fullmatch(r"vit\.encoder\.block\.(\d+)/(.+)", site)
        if scoped is None:
            raise ValueError(f"Gaussian site is outside the ViT block scope: {site}")
        block = int(scoped.group(1))
        row = {
            "site": site,
            "block": block,
            "local_site": scoped.group(2),
            "events": int(match.group(2)),
            "misses": int(match.group(3)),
            "deadline_events": int(match.group(5)),
            "outputs": int(match.group(7)),
            "underflows": int(match.group(8)),
            "overflows": int(match.group(10)),
        }
        rows.append(row)
        block_sites.setdefault(block, []).append(row)
    if first_block_count == 0:
        if rows:
            raise ValueError("clean K=0 condition unexpectedly recorded Gaussian activity")
        return {
            "events": 0,
            "misses": 0,
            "deadline_events": 0,
            "outputs": 0,
            "underflows": 0,
            "overflows": 0,
            "miss_rate": 0.0,
            "active_blocks": [],
            "site_count": 0,
            "sites": [],
        }
    expected_blocks = set(range(first_block_count))
    if set(block_sites) != expected_blocks:
        raise ValueError("Gaussian block counters do not match the selected prefix")
    for block, block_rows in block_sites.items():
        event_sites = [row for row in block_rows if row["events"] > 0]
        has_linear = any(row["local_site"].startswith("linear.") for row in event_sites)
        has_log = any("log_" in row["local_site"] for row in event_sites)
        if not has_linear or not has_log:
            raise ValueError(
                f"block {block} did not execute both linear and logarithmic encoders"
            )
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "events",
            "misses",
            "deadline_events",
            "outputs",
            "underflows",
            "overflows",
        )
    }
    totals.update(
        miss_rate=totals["misses"] / totals["events"],
        active_blocks=sorted(block_sites),
        site_count=len(rows),
        sites=rows,
    )
    return totals


def validate_condition_arguments(args: argparse.Namespace) -> None:
    if not math.isfinite(args.measured_noise_scale) or not (
        0.0 < args.measured_noise_scale <= 1.0
    ):
        raise ValueError("measured_noise_scale must be in (0, 1]")
    if args.condition == "clean":
        if (
            args.first_block_count != 0
            or args.seed != 0
            or args.measured_noise_scale != 1.0
        ):
            raise ValueError("clean condition requires K=0, seed 0, and scale 1")
    elif not 1 <= args.first_block_count <= 12:
        raise ValueError("noisy condition requires K between 1 and 12")
    if args.evaluation_samples not in (500, 5000):
        raise ValueError("evaluation_samples must be 500 or 5000")


def validate_execution_arguments(
    args: argparse.Namespace,
    *,
    hostname: str,
    environment: dict[str, str],
) -> None:
    """Validate direct-host or scheduler-owned device selection."""
    if args.execution_mode == "local":
        if hostname != "baekryun-cuda129" or args.gpu is None:
            raise ValueError("local depth conditions require one baekryun GPU")
        return
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if (
        args.gpu is not None
        or not environment.get("SLURM_JOB_ID")
        or not visible
        or "," in visible
    ):
        raise ValueError("Slurm depth conditions require one allocated GPU")


# @lat: [[evaluation#Evaluation and Verification#ViT-B Cumulative Encoder Block Timing Noise#Condition Execution]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--calibration-dataset-path", type=Path, required=True)
    parser.add_argument("--calibration-dataset-fingerprint", required=True)
    parser.add_argument("--evaluation-dataset-path", type=Path, required=True)
    parser.add_argument("--evaluation-dataset-fingerprint", required=True)
    parser.add_argument("--image-preprocessing-config", type=Path, required=True)
    parser.add_argument("--hardware-summary", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--measured-noise-scale", type=float, default=1.0)
    parser.add_argument("--first-block-count", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--evaluation-samples", type=int, required=True)
    parser.add_argument("--execution-mode", choices=("local", "slurm"), default="local")
    parser.add_argument("--gpu", type=int, choices=range(8))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    validate_condition_arguments(args)
    validate_execution_arguments(
        args, hostname=socket.gethostname(), environment=dict(os.environ)
    )

    for name in (
        "source_root",
        "model_id",
        "calibration_source",
        "calibration_dataset_path",
        "evaluation_dataset_path",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.image_preprocessing_config = args.image_preprocessing_config.resolve(strict=True)
    args.hardware_summary = args.hardware_summary.resolve(strict=True)
    source_hashes = source_identity(args.source_root, args.expected_commit)
    runner_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    runner_dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if runner_dirty:
        raise ValueError("depth-condition runner requires a clean source checkout")
    runner_key = (
        str(Path(__file__).resolve().relative_to(args.source_root))
        if ROOT.resolve() == args.source_root
        else "runner:scripts/experiments/run_vit_bss2_depth_condition.py"
    )
    source_hashes[runner_key] = identity.sha256_file(Path(__file__).resolve())
    checkpoint_sha256 = identity.artifact_identity(args.model_id)["aggregate_sha256"]
    calibration_path, calibration_sha256, sites = completed_calibration(
        args.calibration_source, args.expected_commit
    )
    evaluation_dataset = dataset_identity(
        args.evaluation_dataset_path,
        args.evaluation_dataset_fingerprint,
        evaluation_samples=args.evaluation_samples,
    )
    measured = hardware_conditions(args.hardware_summary)
    if args.condition == "clean":
        linear_fraction = log_fraction = 0.0
        measured_condition: dict[str, Any] | None = None
    else:
        measured_condition = measured[args.condition]
        linear_fraction = (
            float(measured_condition["linear_time_std_fraction"])
            * args.measured_noise_scale
        )
        log_fraction = (
            float(measured_condition["log_time_std_fraction"])
            * args.measured_noise_scale
        )

    phase = "pilot" if args.evaluation_samples == 500 else "formal"
    scale_suffix = (
        ""
        if args.measured_noise_scale == 1.0
        else "_scale" + format(args.measured_noise_scale, ".12g").replace(".", "p")
    )
    run_id = (
        "clean_k00_seed0"
        if args.condition == "clean"
        else (
            f"{args.condition}{scale_suffix}_k{args.first_block_count:02d}"
            f"_seed{args.seed}"
        )
    )
    output = ARTIFACTS / "logs/bss2_vit_depth" / TAG / phase / "runs" / run_id
    runtime_base = args.runtime_root or ARTIFACTS / "runtime"
    runtime = runtime_base / TAG / phase / run_id
    output.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True
    ).strip()
    if filesystem in {"tmpfs", "ramfs"}:
        raise RuntimeError("depth-condition runtime must use a disk filesystem")
    local_calibration = output / "calibration.json"
    if local_calibration.exists():
        if identity.sha256_file(local_calibration) != calibration_sha256:
            raise ValueError("existing calibration copy differs")
    else:
        os.link(calibration_path, local_calibration)

    wrapper = args.source_root / "scripts/analysis/evaluate_calibrated_vit.py"
    command = [
        args.python_bin,
        "-u",
        str(wrapper),
        "--source-root",
        str(args.source_root),
        "--calibration-dataset-path",
        str(args.calibration_dataset_path),
        "--calibration-dataset-fingerprint",
        args.calibration_dataset_fingerprint,
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
        str(args.model_id),
        "--dataset_id",
        "imagenet-1k",
        "--evaluation-dataset-path",
        str(args.evaluation_dataset_path),
        "--evaluation-split",
        "validation",
        "--evaluation-samples",
        str(args.evaluation_samples),
        "--image-preprocessing-config",
        str(args.image_preprocessing_config),
        "--batch_size",
        "32",
        "--precision",
        "float64",
        "--source-commit",
        args.expected_commit,
        "--checkpoint-sha256",
        checkpoint_sha256,
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
        str(local_calibration),
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
        "4",
        "--time-noise-seed",
        str(args.seed),
        "--time-noise-vit-first-block-count",
        str(args.first_block_count),
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
    manifest = {
        "schema_version": 1,
        "tag": TAG,
        "phase": phase,
        "run_id": run_id,
        "condition": args.condition,
        "first_block_count": args.first_block_count,
        "seed": args.seed,
        "source_commit": args.expected_commit,
        "source_hashes": source_hashes,
        "checkpoint_path": str(args.model_id),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_source": str(args.calibration_source),
        "calibration_sha256": calibration_sha256,
        "calibration_source_result_sha256": identity.sha256_file(
            args.calibration_source / "result.json"
        ),
        "evaluation_dataset": evaluation_dataset,
        "evaluation_samples": args.evaluation_samples,
        "hardware_summary": str(args.hardware_summary),
        "hardware_summary_sha256": HARDWARE_SUMMARY_SHA256,
        "measured_condition": measured_condition,
        "linear_time_std_fraction": linear_fraction,
        "log_time_std_fraction": log_fraction,
        "deadline_margin_sigma_ratio": 4.0,
        "time_noise_scope": "vit_first_blocks",
        "dtype": "float64",
        "batch_size": 32,
        "physical_gpu": (
            args.gpu
            if args.execution_mode == "local"
            else os.environ.get("SLURM_JOB_GPUS", os.environ["CUDA_VISIBLE_DEVICES"])
        ),
        "runtime_dir": str(runtime),
        "command": command,
    }
    if args.execution_mode == "slurm":
        manifest.update(
            execution_mode="slurm",
            execution_host=socket.gethostname(),
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )
    if runner_commit != args.expected_commit:
        manifest["runner_source_commit"] = runner_commit
    if args.measured_noise_scale != 1.0:
        manifest["measured_noise_scale"] = args.measured_noise_scale
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if canonical(json.loads(manifest_path.read_text())) != canonical(manifest):
            raise ValueError("existing depth-condition manifest differs")
    else:
        runtime_files.new_json(manifest_path, manifest)
    result_path = output / "result.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("state") == "complete":
        print(result_path.read_text(), flush=True)
        return

    with ExitStack() as stack:
        if args.execution_mode == "local":
            lock_path = (
                ARTIFACTS / "runtime/gpu-locks" / f"local-gpu-{args.gpu}.lock"
            )
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(lock_path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for check in range(2):
                activity = local_gpu.gpu_activity(gpu_ids=(args.gpu,))[args.gpu]
                if not local_gpu.gpu_available(activity):
                    raise RuntimeError(f"GPU {args.gpu} is occupied")
                if check == 0:
                    time.sleep(10)
        attempt = len(list(output.joinpath("logs").glob("evaluation.attempt-*.log"))) + 1
        log_path = output / "logs" / f"evaluation.attempt-{attempt:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(
            os.environ,
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
            PYTHONPATH=os.pathsep.join(
                map(
                    str,
                    (
                        args.source_root,
                        args.source_root / "src/transformers/src",
                        args.source_root / "src/spikingjelly",
                    ),
                )
            ),
        )
        if args.execution_mode == "local":
            environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        started = time.monotonic()
        runtime_files.atomic_json(result_path, {"state": "running", "run_id": run_id})
        with log_path.open("x") as log:
            completed = subprocess.run(
                command,
                cwd=args.source_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            failure = {
                "state": "failed",
                "run_id": run_id,
                "exit_status": completed.returncode,
                "log_file": str(log_path.relative_to(output)),
            }
            runtime_files.atomic_json(result_path, failure)
            raise RuntimeError(
                f"depth-condition evaluator exited with status {completed.returncode}"
            )
        log_text = log_path.read_text(errors="replace")
        metrics = parse_metric(log_text, args.evaluation_samples)
        validate_snn_sites(log_text, sites)
        metrics["gaussian_counts"] = parse_scoped_counts(
            log_text, args.first_block_count
        )
        result = {
            "state": "complete",
            "run_id": run_id,
            "elapsed_seconds": time.monotonic() - started,
            "log_file": str(log_path.relative_to(output)),
            "log_sha256": identity.sha256_file(log_path),
            "metrics": metrics,
        }
        source_identity(args.source_root, args.expected_commit)
        runtime_files.atomic_json(result_path, result)
        print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
