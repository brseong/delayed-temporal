#!/usr/bin/env python3
"""Schedule and validate the ViT-B cumulative encoder-block timing-noise campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_vit_bss2_depth_condition import (
    ARTIFACTS,
    CONDITIONS,
    HARDWARE_SUMMARY_SHA256,
    TAG,
    hardware_conditions,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity, local_gpu


PHASE_SAMPLES = {"pilot": 500, "formal": 5000}


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def expected_cells() -> tuple[tuple[str, int, int], ...]:
    """Return clean plus the fixed 72 noisy replica identities."""
    cells: list[tuple[str, int, int]] = [("clean", 0, 0)]
    for condition in CONDITIONS[1:]:
        for first_block_count in range(1, 13):
            for seed in range(3):
                cells.append((condition, first_block_count, seed))
    if len(cells) != 73 or len(set(cells)) != 73:
        raise AssertionError("depth campaign must contain 73 unique runs")
    return tuple(cells)


def run_id(condition: str, first_block_count: int, seed: int) -> str:
    if condition == "clean":
        return "clean_k00_seed0"
    return f"{condition}_k{first_block_count:02d}_seed{seed}"


def phase_root(phase: str) -> Path:
    return ARTIFACTS / "logs/bss2_vit_depth" / TAG / phase


def task_command(
    args: argparse.Namespace,
    *,
    phase: str,
    condition: str,
    first_block_count: int,
    seed: int,
    gpu: int | None = None,
) -> list[str]:
    command = [
        args.python_bin,
        "-u",
        str(args.source_root / "scripts/experiments/run_vit_bss2_depth_condition.py"),
        "--source-root",
        str(args.source_root),
        "--expected-commit",
        args.expected_commit,
        "--model-id",
        str(args.model_id),
        "--calibration-source",
        str(args.calibration_source),
        "--calibration-dataset-path",
        str(args.calibration_dataset_path),
        "--calibration-dataset-fingerprint",
        args.calibration_dataset_fingerprint,
        "--evaluation-dataset-path",
        str(args.evaluation_dataset_path),
        "--evaluation-dataset-fingerprint",
        args.evaluation_dataset_fingerprint,
        "--image-preprocessing-config",
        str(args.image_preprocessing_config),
        "--hardware-summary",
        str(args.hardware_summary),
        "--condition",
        condition,
        "--first-block-count",
        str(first_block_count),
        "--seed",
        str(seed),
        "--evaluation-samples",
        str(PHASE_SAMPLES[phase]),
        "--python-bin",
        args.python_bin,
    ]
    if gpu is not None:
        command += ["--gpu", str(gpu)]
    return command


def tasks(args: argparse.Namespace, phase: str) -> list[dict[str, Any]]:
    root = phase_root(phase)
    rows = []
    for condition, first_block_count, seed in expected_cells():
        name = run_id(condition, first_block_count, seed)
        rows.append(
            {
                "name": name,
                "condition": condition,
                "first_block_count": first_block_count,
                "seed": seed,
                "result": root / "runs" / name / "result.json",
                "command": task_command(
                    args,
                    phase=phase,
                    condition=condition,
                    first_block_count=first_block_count,
                    seed=seed,
                ),
            }
        )
    return rows


def validate_phase_artifacts(
    root: Path,
    *,
    phase: str,
    calibration_source: Path,
    hardware_summary: Path,
) -> None:
    """Gate formal release and summaries on all 73 authenticated results."""
    sample_count = PHASE_SAMPLES[phase]
    measured = hardware_conditions(hardware_summary)
    clean_prediction: str | None = None
    selected_fingerprint: str | None = None
    seen: set[tuple[str, int, int]] = set()
    for condition, first_block_count, seed in expected_cells():
        name = run_id(condition, first_block_count, seed)
        run_root = root / "runs" / name
        manifest_path, result_path = run_root / "manifest.json", run_root / "result.json"
        if not manifest_path.is_file() or not complete(result_path):
            raise ValueError(f"depth campaign run is incomplete: {name}")
        manifest = json.loads(manifest_path.read_text())
        result = json.loads(result_path.read_text())
        identity_tuple = (
            manifest.get("condition"),
            manifest.get("first_block_count"),
            manifest.get("seed"),
        )
        if identity_tuple != (condition, first_block_count, seed):
            raise ValueError(f"depth campaign identity differs: {name}")
        if identity_tuple in seen:
            raise ValueError("depth campaign contains a duplicate condition")
        seen.add(identity_tuple)
        if (
            manifest.get("evaluation_samples") != sample_count
            or manifest.get("hardware_summary_sha256") != HARDWARE_SUMMARY_SHA256
            or manifest.get("deadline_margin_sigma_ratio") != 4.0
            or manifest.get("time_noise_scope") != "vit_first_blocks"
            or result["metrics"].get("total") != sample_count
        ):
            raise ValueError(f"depth campaign contract differs: {name}")
        run_fingerprint = manifest.get("evaluation_dataset", {}).get(
            "selected_fingerprint"
        )
        if not run_fingerprint:
            raise ValueError(f"depth campaign selected dataset is unidentified: {name}")
        if selected_fingerprint is None:
            selected_fingerprint = run_fingerprint
        elif selected_fingerprint != run_fingerprint:
            raise ValueError("depth campaign runs use different evaluation prefixes")
        counts = result["metrics"].get("gaussian_counts", {})
        expected_blocks = list(range(first_block_count))
        if counts.get("active_blocks") != expected_blocks:
            raise ValueError(f"depth campaign active blocks differ: {name}")
        if condition == "clean":
            if counts.get("site_count") != 0:
                raise ValueError("clean depth condition contains Gaussian counters")
            clean_prediction = result["metrics"]["prediction_sha256"]
        else:
            expected = measured[condition]
            if (
                manifest.get("linear_time_std_fraction")
                != expected["linear_time_std_fraction"]
                or manifest.get("log_time_std_fraction")
                != expected["log_time_std_fraction"]
                or counts.get("site_count", 0) <= 0
                or counts.get("events", 0) <= 0
            ):
                raise ValueError(f"depth campaign measured condition differs: {name}")
    if seen != set(expected_cells()) or clean_prediction is None:
        raise ValueError("depth campaign condition population differs")
    if phase == "formal":
        source_result = json.loads((calibration_source / "result.json").read_text())
        expected_clean = source_result["phases"]["snn"]["metrics"]
        if (
            expected_clean.get("total") != sample_count
            or expected_clean.get("prediction_sha256") != clean_prediction
        ):
            raise ValueError("K=0 prediction digest differs from the clean SNN reference")


def status_snapshot(
    path: Path,
    *,
    phase: str,
    pending: list[dict[str, Any]],
    running: dict[int, dict[str, Any]],
    finished: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> None:
    runtime_files.atomic_json(
        path,
        {
            "state": (
                "failed"
                if failed and not running and not pending
                else "complete"
                if not running and not pending
                else "running"
            ),
            "phase": phase,
            "pending": [task["name"] for task in pending],
            "running": {
                str(gpu): {"name": task["name"], "pid": task["process"].pid}
                for gpu, task in running.items()
            },
            "finished": [task["name"] for task in finished],
            "failed": [
                {"name": task["name"], "exit_status": task["exit_status"]}
                for task in failed
            ],
            "updated_at": time.time(),
        },
    )


def run_phase(args: argparse.Namespace, phase: str) -> None:
    root = phase_root(phase)
    root.mkdir(parents=True, exist_ok=True)
    all_tasks = tasks(args, phase)
    pending = [task for task in all_tasks if not complete(task["result"])]
    finished = [task for task in all_tasks if complete(task["result"])]
    failed: list[dict[str, Any]] = []
    running: dict[int, dict[str, Any]] = {}
    while pending or running:
        for gpu, task in list(running.items()):
            returncode = task["process"].poll()
            if returncode is None:
                continue
            task["log"].close()
            del running[gpu]
            if returncode == 0 and complete(task["result"]):
                finished.append(task)
            else:
                task["exit_status"] = returncode
                failed.append(task)
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            activity = local_gpu.gpu_activity(gpu_ids=(gpu,))[gpu]
            if not local_gpu.gpu_available(activity):
                continue
            task = pending.pop(0)
            command = [*task["command"], "--gpu", str(gpu)]
            log_path = root / "controller-logs" / f"{task['name']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a")
            process = subprocess.Popen(
                command,
                cwd=args.source_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            task.update(process=process, log=log, gpu=gpu)
            running[gpu] = task
        status_snapshot(
            root / "status.json",
            phase=phase,
            pending=pending,
            running=running,
            finished=finished,
            failed=failed,
        )
        if pending or running:
            time.sleep(10)
    if failed:
        raise RuntimeError(f"{len(failed)} {phase} depth conditions failed")
    validate_phase_artifacts(
        root,
        phase=phase,
        calibration_source=args.calibration_source,
        hardware_summary=args.hardware_summary,
    )
    summary = args.source_root / "scripts/analysis/summarize_vit_bss2_depth.py"
    subprocess.run(
        [args.python_bin, str(summary), "--phase", phase, "--input-root", str(root)],
        cwd=args.source_root,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=("pilot", "formal", "all"), default="all")
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
    parser.add_argument("--gpus", type=int, nargs="+", default=(4, 5, 6, 7))
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    if (
        not args.gpus
        or len(set(args.gpus)) != len(args.gpus)
        or any(gpu not in range(4, 8) for gpu in args.gpus)
    ):
        raise ValueError("GPU list must contain unique indices from 4 through 7")
    for name in (
        "source_root",
        "model_id",
        "calibration_source",
        "calibration_dataset_path",
        "evaluation_dataset_path",
        "image_preprocessing_config",
        "hardware_summary",
    ):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.source_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if actual_commit != args.expected_commit or dirty:
        raise ValueError("depth campaign requires the clean frozen source commit")
    if identity.sha256_file(args.hardware_summary) != HARDWARE_SUMMARY_SHA256:
        raise ValueError("BrainScaleS-2 encoder summary checksum differs")

    output = ARTIFACTS / "logs/bss2_vit_depth" / TAG
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "tag": TAG,
        "source_root": str(args.source_root),
        "source_commit": args.expected_commit,
        "model_id": str(args.model_id),
        "calibration_source": str(args.calibration_source),
        "calibration_dataset_path": str(args.calibration_dataset_path),
        "calibration_dataset_fingerprint": args.calibration_dataset_fingerprint,
        "evaluation_dataset_path": str(args.evaluation_dataset_path),
        "evaluation_dataset_fingerprint": args.evaluation_dataset_fingerprint,
        "image_preprocessing_config": str(args.image_preprocessing_config),
        "phase": args.phase,
        "gpus": list(args.gpus),
        "hardware_summary": str(args.hardware_summary),
        "hardware_summary_sha256": HARDWARE_SUMMARY_SHA256,
        "conditions": list(CONDITIONS[1:]),
        "first_block_counts": list(range(13)),
        "seeds": [0, 1, 2],
        "pilot_samples": 500,
        "formal_samples": 5000,
        "deadline_margin_sigma_ratio": 4.0,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        comparable = {**existing, "phase": args.phase, "gpus": list(args.gpus)}
        if comparable != manifest:
            raise ValueError("existing depth-campaign manifest differs")
    else:
        runtime_files.new_json(manifest_path, manifest)

    if args.phase in ("pilot", "all"):
        run_phase(args, "pilot")
    if args.phase in ("formal", "all"):
        validate_phase_artifacts(
            phase_root("pilot"),
            phase="pilot",
            calibration_source=args.calibration_source,
            hardware_summary=args.hardware_summary,
        )
        run_phase(args, "formal")


if __name__ == "__main__":
    main()
