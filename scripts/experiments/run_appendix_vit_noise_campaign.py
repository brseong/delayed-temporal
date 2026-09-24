#!/usr/bin/env python3
"""Prepare and distribute the authenticated Appendix ViT-B noise sweep."""

from __future__ import annotations

import argparse
import json
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
from scripts.runtime import identity, local_gpu
from scripts.experiments.run_vit_local_range_noise_condition import (
    APPENDIX_RAW_TIMESTAMP_TAG,
    ED_INTERNAL_NOISE_CONTRACT,
    RAW_TIMESTAMP_CONTRACT,
)


ARTIFACTS = Path(
    os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", "/data/delayed-temporal/artifacts")
)
BASELINE_TAG = "vit_base_appendix_local_range_raw_timestamp_baseline_float64_v2"
CAMPAIGN_TAG = "appendix_vit_base_local_range_raw_timestamp_float64_v2"
TRAIN_FINGERPRINT = "cabf903d14d1b1ac"
VALIDATION_FINGERPRINT = "746378cc7befed99"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def noise_cells() -> tuple[tuple[float, float], ...]:
    fractions = [10 ** (-5 + index / 8) for index in range(9)]
    ratios = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
    cells = {(fraction, 4.0) for fraction in fractions}
    cells.update((1e-5, ratio) for ratio in ratios)
    return tuple(sorted(cells))


def run_id(fraction: float, ratio: float, seed: int) -> str:
    fraction_text = f"{fraction:.12g}".replace(".", "p").replace("-", "m").replace("+", "p")
    ratio_text = f"{ratio:g}".replace(".", "p")
    return f"frac_{fraction_text}_ratio_{ratio_text}_seed_{seed}"


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def baseline_root() -> Path:
    return ARTIFACTS / "logs/conversion_comparison" / BASELINE_TAG / "vit/imagenet_vit_base"


def common_paths(source_root: Path) -> dict[str, Path]:
    dataset = ARTIFACTS / "assets/theta-selection-v1/datasets/imagenet_theta_selection_v1"
    return {
        "model": Path("/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k"),
        "train": dataset / "train_seed0_5000",
        "validation": dataset / "validation_50000",
        "preprocessing": source_root / "scripts/configs/vit_timm_preprocessing.json",
    }


def prepare(args: argparse.Namespace) -> None:
    if socket.gethostname() != "baekryun-cuda129":
        raise ValueError("Appendix baseline preparation runs on baekryun")
    paths = common_paths(args.source_root)
    output = baseline_root()
    runtime = ARTIFACTS / "runtime" / BASELINE_TAG / "vit/imagenet_vit_base"
    command = [
        args.python_bin, "-u",
        str(args.source_root / "scripts/experiments/run_full_calibrated_vit_comparison.py"),
        "--model-key", "imagenet_vit_base", "--source-root", str(args.source_root),
        "--expected-commit", args.expected_commit, "--campaign-tag", BASELINE_TAG,
        "--model-id", str(paths["model"]),
        "--calibration-dataset-path", str(paths["train"]),
        "--calibration-dataset-fingerprint", TRAIN_FINGERPRINT,
        "--evaluation-dataset-path", str(paths["validation"]),
        "--evaluation-dataset-fingerprint", VALIDATION_FINGERPRINT,
        "--image-preprocessing-config", str(paths["preprocessing"]),
        "--batch-size", "32", "--gpu", str(args.gpus[0]), "--host-label", "local",
        "--python-bin", args.python_bin, "--output-root", str(output),
        "--runtime-root", str(runtime),
    ]
    subprocess.run(command, cwd=args.source_root, check=True)


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = baseline_root()
    if not complete(root / "result.json"):
        raise ValueError("the fresh ViT-B baseline is incomplete")
    result = json.loads((root / "result.json").read_text())
    baseline_manifest = json.loads((root / "manifest.json").read_text())
    if (
        baseline_manifest.get("source_commit") != args.expected_commit
        or baseline_manifest.get("tag") != BASELINE_TAG
        or result.get("calibration_sha256") != identity.sha256_file(root / "calibration.json")
    ):
        raise ValueError("the fresh ViT-B baseline identity differs")
    slots = [
        *(("local", gpu) for gpu in args.local_gpus),
        *(("poseidon", gpu) for gpu in args.poseidon_gpus),
    ]
    if not slots:
        raise ValueError("at least one worker GPU is required")
    assignments = []
    for index, (fraction, ratio, seed) in enumerate(
        (fraction, ratio, seed)
        for fraction, ratio in noise_cells()
        for seed in range(3)
    ):
        host_label, physical_gpu = slots[index % len(slots)]
        assignments.append({
            "run_id": run_id(fraction, ratio, seed),
            "time_noise_std_fraction": fraction,
            "deadline_margin_sigma_ratio": ratio,
            "seed": seed,
            "host_label": host_label,
            "physical_gpu": physical_gpu,
        })
    if len(assignments) != 63 or len({row["run_id"] for row in assignments}) != 63:
        raise AssertionError("Appendix sweep must contain 63 unique replica runs")
    paths = common_paths(args.source_root)
    return {
        "schema_version": 1,
        "tag": CAMPAIGN_TAG,
        "noise_tag": APPENDIX_RAW_TIMESTAMP_TAG,
        "baseline_tag": BASELINE_TAG,
        "source_commit": args.expected_commit,
        "source_root": str(args.source_root),
        "condition_runner_sha256": identity.sha256_file(
            args.source_root / "scripts/experiments/run_vit_local_range_noise_condition.py"
        ),
        "evaluator_sha256": identity.sha256_file(
            args.source_root / "scripts/evaluation/error_analysis_vit.py"
        ),
        "checkpoint_path": str(paths["model"]),
        "checkpoint_sha256": baseline_manifest["checkpoint_sha256"],
        "calibration_source": str(root),
        "calibration_sha256": result["calibration_sha256"],
        "calibration_source_result_sha256": identity.sha256_file(root / "result.json"),
        "calibration_dataset_path": str(paths["train"]),
        "calibration_dataset_fingerprint": TRAIN_FINGERPRINT,
        "evaluation_dataset_path": str(paths["validation"]),
        "evaluation_dataset_fingerprint": VALIDATION_FINGERPRINT,
        "image_preprocessing_config": str(paths["preprocessing"]),
        "image_preprocessing_sha256": identity.sha256_file(paths["preprocessing"]),
        "dtype": "float64", "batch_size": 32, "evaluation_samples": 5_000,
        "raw_timestamp_contract": RAW_TIMESTAMP_CONTRACT,
        "exponential_difference_internal_noise": ED_INTERNAL_NOISE_CONTRACT,
        "wandb": False, "tensorboard": False,
        "runtime_filesystem": "disk_required", "tmpfs": False,
        "local_gpu_override": list(args.local_gpus),
        "poseidon_gpus": list(args.poseidon_gpus),
        "assignments": assignments,
    }


def write_manifest(args: argparse.Namespace) -> None:
    output = ARTIFACTS / "logs" / CAMPAIGN_TAG
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    path = output / "manifest.json"
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(manifest):
            raise ValueError("existing Appendix campaign manifest differs")
    else:
        runtime_files.new_json(path, manifest)
    print(path)


def task_command(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    row: dict[str, Any],
    gpu: int,
) -> list[str]:
    if row.get("host_label") != args.host_label or row.get("physical_gpu") != gpu:
        raise ValueError("task execution differs from its central host/GPU assignment")
    command = [
        args.python_bin, "-u",
        str(args.source_root / "scripts/experiments/run_vit_local_range_noise_condition.py"),
        "--source-root", str(args.source_root), "--expected-commit", args.expected_commit,
        "--model-id", manifest["checkpoint_path"],
        "--checkpoint-sha256", manifest["checkpoint_sha256"],
        "--calibration-source", manifest["calibration_source"],
        "--calibration-dataset-path", manifest["calibration_dataset_path"],
        "--calibration-dataset-fingerprint", manifest["calibration_dataset_fingerprint"],
        "--evaluation-dataset-path", manifest["evaluation_dataset_path"],
        "--evaluation-dataset-fingerprint", manifest["evaluation_dataset_fingerprint"],
        "--image-preprocessing-config", manifest["image_preprocessing_config"],
        "--time-noise-std-frac", repr(row["time_noise_std_fraction"]),
        "--deadline-margin-ratio", repr(row["deadline_margin_sigma_ratio"]),
        "--seed", str(row["seed"]), "--run-id", row["run_id"],
        "--gpu", str(gpu), "--host-label", args.host_label,
        "--campaign-tag", APPENDIX_RAW_TIMESTAMP_TAG, "--python-bin", args.python_bin,
    ]
    if args.host_label == "local":
        command.append("--allow-all-local-gpus")
    return command


def status_snapshot(
    path: Path,
    *,
    pending: list[dict[str, Any]],
    running: dict[int, dict[str, Any]],
    finished: list[str],
    failed: list[dict[str, Any]],
) -> None:
    runtime_files.atomic_json(path, {
        "state": "failed" if failed and not running and not pending else
                 "complete" if not running and not pending else "running",
        "pending": [row["run_id"] for row in pending],
        "running": {
            str(gpu): {"run_id": task["row"]["run_id"], "pid": task["process"].pid}
            for gpu, task in running.items()
        },
        "finished": finished, "failed": failed, "updated_at": time.time(),
    })


def worker(args: argparse.Namespace) -> None:
    expected_host = "baekryun-cuda129" if args.host_label == "local" else "poseidon1"
    if socket.gethostname() != expected_host:
        raise ValueError(f"worker requires host {expected_host}")
    manifest_path = ARTIFACTS / "logs" / CAMPAIGN_TAG / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("source_commit") != args.expected_commit
        or manifest.get("raw_timestamp_contract") != RAW_TIMESTAMP_CONTRACT
        or manifest.get("exponential_difference_internal_noise") != ED_INTERNAL_NOISE_CONTRACT
    ):
        raise ValueError("Appendix campaign identity differs")
    allowed = manifest["local_gpu_override" if args.host_label == "local" else "poseidon_gpus"]
    if list(args.gpus) != allowed:
        raise ValueError("worker GPU list differs from the frozen campaign manifest")
    runtime = ARTIFACTS / "runtime" / CAMPAIGN_TAG / args.host_label
    runtime.mkdir(parents=True, exist_ok=True)
    fstype = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True,
    ).strip()
    if fstype in {"tmpfs", "ramfs"}:
        raise RuntimeError("Appendix worker runtime must use disk storage")
    rows = [row for row in manifest["assignments"] if row["host_label"] == args.host_label]
    result_root = ARTIFACTS / "logs/noise_scan" / APPENDIX_RAW_TIMESTAMP_TAG / "runs"
    pending = [row for row in rows if not complete(result_root / row["run_id"] / "result.json")]
    finished = [row["run_id"] for row in rows if row not in pending]
    running: dict[int, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    status_path = ARTIFACTS / "logs" / CAMPAIGN_TAG / f"status-{args.host_label}.json"
    controller_logs = ARTIFACTS / "logs" / CAMPAIGN_TAG / f"controller-{args.host_label}"
    controller_logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for gpu, task in list(running.items()):
            returncode = task["process"].poll()
            if returncode is None:
                continue
            task["log"].close()
            del running[gpu]
            row = task["row"]
            if returncode == 0 and complete(result_root / row["run_id"] / "result.json"):
                finished.append(row["run_id"])
            else:
                failed.append({"run_id": row["run_id"], "exit_status": returncode})
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            activity = local_gpu.gpu_activity(gpu_ids=(gpu,))[gpu]
            if not local_gpu.gpu_available(activity):
                continue
            row_index = next(
                (index for index, row in enumerate(pending) if row["physical_gpu"] == gpu),
                None,
            )
            if row_index is None:
                continue
            row = pending.pop(row_index)
            log_path = controller_logs / f"{row['run_id']}.log"
            log = log_path.open("a")
            process = subprocess.Popen(
                task_command(args, manifest, row, gpu), cwd=args.source_root,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            running[gpu] = {"row": row, "process": process, "log": log}
        status_snapshot(
            status_path, pending=pending, running=running, finished=finished, failed=failed,
        )
        if pending or running:
            time.sleep(10)
    if failed:
        raise RuntimeError(f"{len(failed)} Appendix noise conditions failed")


# @lat: [[evaluation#Evaluation and Verification#Appendix Raw-Timestamp ViT-B Noise Rerun]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("prepare", "manifest", "worker"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host-label", choices=("local", "poseidon"), default="local")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--local-gpus", type=int, nargs="+", default=tuple(range(8)))
    parser.add_argument("--poseidon-gpus", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        raise ValueError("a full frozen source commit is required")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.source_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if actual_commit != args.expected_commit or dirty:
        raise ValueError("Appendix campaign requires the clean frozen source commit")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("worker GPU list contains duplicates")
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "manifest":
        write_manifest(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
