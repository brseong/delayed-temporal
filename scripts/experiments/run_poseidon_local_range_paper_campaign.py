#!/usr/bin/env python3
"""Schedule the local-range paper re-evaluation on free poseidon1 GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
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
from scripts.experiments.run_full_calibrated_vit_comparison import TAG as VIT_TAG
from scripts.experiments.run_vit_local_range_noise_condition import TAG as NOISE_TAG
from scripts.experiments.run_full_calibrated_text_comparison import (
    GPT2_COMPOSED_GELU_TAG,
    ROBERTA_LARGE_TAG,
    TAG as TEXT_TAG,
)


ARTIFACTS = Path(os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", "/data/delayed-temporal/artifacts"))
CAMPAIGN_TAG = "paper_end_to_end_local_range_poseidon_v1"
IMAGENET_TRAIN_FP = "cabf903d14d1b1ac"
IMAGENET_VAL_FP = "746378cc7befed99"
CIFAR_TRAIN_FP = "7201bcb7da71648f"
CIFAR_TEST_FP = "c07893a979afbb7d"
SST2_TRAIN_FP = "35d5874329335756"
SST2_VAL_FP = "e3182d90fd436f61"
WIKITEXT_TRAIN_FP = "f1506153809011c4"
WIKITEXT_TEST_FP = "38d46c7ecf7254ca"


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def checkpoint_hash(path: Path) -> str:
    return identity.artifact_identity(path)["aggregate_sha256"]


def noise_cells() -> tuple[tuple[float, float], ...]:
    """Return the exact unique Figure 4 cells in stable numeric order."""
    fractions = [10 ** (-5 + index / 8) for index in range(9)]
    ratios = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    cells = {(fraction, 4.0) for fraction in fractions}
    cells.update((1e-5, ratio) for ratio in ratios)
    return tuple(sorted(cells))


def vit_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    assets = ARTIFACTS / "assets"
    imagenet = assets / "theta-selection-v1/datasets/imagenet_theta_selection_v1"
    cifar = assets / "vit-conversion-comparison-v1"
    preprocessing = args.source_root / "scripts/configs/vit_timm_preprocessing.json"
    configurations = [
        ("cifar10_vit_small", cifar / "checkpoints/vit_small_patch16_224_cifar10",
         cifar / "datasets/cifar10/train_seed0_5000", CIFAR_TRAIN_FP,
         cifar / "datasets/cifar10/test_10000", CIFAR_TEST_FP, 32, None),
        ("imagenet_vit_small", Path("/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"),
         imagenet / "train_seed0_5000", IMAGENET_TRAIN_FP,
         imagenet / "validation_50000", IMAGENET_VAL_FP, 32, preprocessing),
        ("imagenet_vit_base", Path("/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k"),
         imagenet / "train_seed0_5000", IMAGENET_TRAIN_FP,
         imagenet / "validation_50000", IMAGENET_VAL_FP, 32, preprocessing),
        ("imagenet_vit_large", Path("/data/nas/vit_large_patch16_224.augreg_in21k_ft_in1k"),
         imagenet / "train_seed0_5000", IMAGENET_TRAIN_FP,
         imagenet / "validation_50000", IMAGENET_VAL_FP, 16, preprocessing),
    ]
    tasks = []
    for key, model, calibration, calibration_fp, evaluation, evaluation_fp, batch, config in configurations:
        output = ARTIFACTS / "logs/conversion_comparison" / VIT_TAG / "vit" / key
        command = [
            args.python_bin, "-u", str(args.source_root / "scripts/experiments/run_full_calibrated_vit_comparison.py"),
            "--model-key", key, "--source-root", str(args.source_root),
            "--expected-commit", args.expected_commit, "--model-id", str(model),
            "--calibration-dataset-path", str(calibration),
            "--calibration-dataset-fingerprint", calibration_fp,
            "--evaluation-dataset-path", str(evaluation),
            "--evaluation-dataset-fingerprint", evaluation_fp,
            "--batch-size", str(batch), "--host-label", "poseidon",
            "--python-bin", args.python_bin, "--output-root", str(output),
            "--runtime-root", str(ARTIFACTS / "runtime" / VIT_TAG / "vit" / key),
        ]
        if config is not None:
            command += ["--image-preprocessing-config", str(config)]
        tasks.append({"name": key, "kind": "table", "command": command,
                      "result": output / "result.json"})
    return tasks


def text_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    assets = ARTIFACTS / "assets/conversion-comparison-text-v1"
    configurations = [
        ("roberta", TEXT_TAG, assets / "checkpoints/roberta", assets / "sst2/train_seed0_5000",
         SST2_TRAIN_FP, assets / "sst2/validation_872", SST2_VAL_FP),
        ("roberta_large", ROBERTA_LARGE_TAG, assets / "checkpoints/roberta-large",
         assets / "sst2/train_seed0_5000", SST2_TRAIN_FP,
         assets / "sst2/validation_872", SST2_VAL_FP),
        ("gpt2", GPT2_COMPOSED_GELU_TAG, assets / "checkpoints/gpt2",
         assets / "wikitext2/train_nonempty_seed0_5000", WIKITEXT_TRAIN_FP,
         assets / "wikitext2/test_nonempty_2891", WIKITEXT_TEST_FP),
    ]
    tasks = []
    for family, tag, model, calibration, calibration_fp, evaluation, evaluation_fp in configurations:
        output = ARTIFACTS / "logs/conversion_comparison" / tag / "text" / family
        command = [
            args.python_bin, "-u", str(args.source_root / "scripts/experiments/run_full_calibrated_text_comparison.py"),
            "--family", family, "--source-root", str(args.source_root),
            "--expected-commit", args.expected_commit, "--model-id", str(model),
            "--calibration-dataset-path", str(calibration),
            "--calibration-dataset-fingerprint", calibration_fp,
            "--evaluation-dataset-path", str(evaluation),
            "--evaluation-dataset-fingerprint", evaluation_fp,
            "--host-label", "poseidon", "--python-bin", args.python_bin,
            "--output-root", str(output),
            "--runtime-root", str(ARTIFACTS / "runtime" / tag / "text"),
        ]
        tasks.append({"name": family, "kind": "table", "command": command,
                      "result": output / "result.json"})
    return tasks


def noise_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = args.noise_calibration_source or (
        ARTIFACTS / "logs/conversion_comparison" / VIT_TAG / "vit/imagenet_vit_base"
    )
    model = Path("/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k")
    calibration_dataset = ARTIFACTS / "assets/theta-selection-v1/datasets/imagenet_theta_selection_v1/train_seed0_5000"
    evaluation_dataset = ARTIFACTS / "assets/theta-selection-v1/datasets/imagenet_theta_selection_v1/validation_50000"
    preprocessing = args.source_root / "scripts/configs/vit_timm_preprocessing.json"
    tasks = []
    checkpoint = checkpoint_hash(model)
    for fraction, ratio in noise_cells():
        for seed in range(3):
            fraction_text = f"{fraction:.12g}".replace(".", "p").replace("-", "m").replace("+", "p")
            ratio_text = f"{ratio:g}".replace(".", "p")
            run_id = f"frac_{fraction_text}_ratio_{ratio_text}_seed_{seed}"
            result = ARTIFACTS / "logs/noise_scan" / NOISE_TAG / "runs" / run_id / "result.json"
            command = [
                args.python_bin, "-u", str(args.source_root / "scripts/experiments/run_vit_local_range_noise_condition.py"),
                "--source-root", str(args.source_root), "--expected-commit", args.expected_commit,
                "--model-id", str(model), "--checkpoint-sha256", checkpoint,
                "--calibration-source", str(base),
                "--calibration-dataset-path", str(calibration_dataset),
                "--calibration-dataset-fingerprint", IMAGENET_TRAIN_FP,
                "--evaluation-dataset-path", str(evaluation_dataset),
                "--image-preprocessing-config", str(preprocessing),
                "--time-noise-std-frac", repr(fraction),
                "--deadline-margin-ratio", repr(ratio), "--seed", str(seed),
                "--run-id", run_id, "--python-bin", args.python_bin,
            ]
            tasks.append({"name": run_id, "kind": "noise", "command": command, "result": result})
    if len(tasks) != 63:
        raise AssertionError("noise grid must contain exactly 63 unique replica runs")
    return tasks


def status_snapshot(path: Path, pending: list[dict[str, Any]], running: dict[int, dict[str, Any]],
                    finished: list[dict[str, Any]], failed: list[dict[str, Any]]) -> None:
    runtime_files.atomic_json(path, {
        "state": "failed" if failed and not running and not pending else
                 "complete" if not running and not pending else "running",
        "pending": [task["name"] for task in pending],
        "running": {str(gpu): {"name": task["name"], "pid": task["process"].pid,
                               "kind": task["kind"]} for gpu, task in running.items()},
        "finished": [task["name"] for task in finished],
        "failed": [{"name": task["name"], "exit_status": task["exit_status"]} for task in failed],
        "updated_at": time.time(),
    })


# @lat: [[evaluation#Evaluation and Verification#Local-Range Paper Re-evaluation]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--campaign-tag", default=CAMPAIGN_TAG)
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("--noise-calibration-source", type=Path)
    args = parser.parse_args()
    if os.uname().nodename != "poseidon1":
        raise ValueError("campaign supervisor must run on poseidon1")
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus or any(gpu not in range(4) for gpu in args.gpus):
        raise ValueError("poseidon GPU list must contain unique indices from 0 through 3")
    if not re.fullmatch(r"[a-z0-9_.-]+", args.campaign_tag):
        raise ValueError("campaign tag contains unsupported characters")
    if args.noise_only and args.noise_calibration_source is None:
        raise ValueError("noise-only execution requires an explicit calibration source")
    args.source_root = args.source_root.resolve(strict=True)
    if args.noise_calibration_source is not None:
        args.noise_calibration_source = args.noise_calibration_source.resolve(strict=True)
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_commit != args.expected_commit:
        raise ValueError("campaign source commit differs")
    output = ARTIFACTS / "logs" / args.campaign_tag
    runtime = ARTIFACTS / "runtime" / args.campaign_tag
    output.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(runtime)], text=True).strip() in {"tmpfs", "ramfs"}:
        raise RuntimeError("campaign runtime must use a disk filesystem")
    manifest = {
        "schema_version": 1, "tag": args.campaign_tag, "source_root": str(args.source_root),
        "source_commit": args.expected_commit, "gpus": list(args.gpus),
        "noise_only": args.noise_only,
        "noise_calibration_source": (
            str(args.noise_calibration_source) if args.noise_calibration_source else None
        ),
        "discrete_time_experiment": False, "wandb": False, "tensorboard": False,
        "runtime_root": str(runtime), "tmpfs": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("existing campaign manifest differs")
    else:
        runtime_files.new_json(manifest_path, manifest)

    table = [] if args.noise_only else vit_tasks(args) + text_tasks(args)
    noise = noise_tasks(args)
    pending = [task for task in table if not complete(task["result"])]
    finished = [task for task in table if complete(task["result"])]
    failed: list[dict[str, Any]] = []
    running: dict[int, dict[str, Any]] = {}
    noise_released = args.noise_only or complete(
        ARTIFACTS / "logs/conversion_comparison" / VIT_TAG / "vit/imagenet_vit_base/result.json"
    )
    if noise_released:
        pending.extend(task for task in noise if not complete(task["result"]))
        finished.extend(task for task in noise if complete(task["result"]))

    while pending or running or not noise_released:
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
        if not noise_released and complete(
            ARTIFACTS / "logs/conversion_comparison" / VIT_TAG / "vit/imagenet_vit_base/result.json"
        ):
            noise_released = True
            pending.extend(task for task in noise if not complete(task["result"]))
            finished.extend(task for task in noise if complete(task["result"]))
        base_failed = any(task["name"] == "imagenet_vit_base" for task in failed)
        if base_failed and not noise_released and not running and not pending:
            break
        for gpu in args.gpus:
            if gpu in running or not pending:
                continue
            activity = local_gpu.gpu_activity(gpu_ids=(gpu,))[gpu]
            if not local_gpu.gpu_available(activity):
                continue
            task = pending.pop(0)
            command = [*task["command"], "--gpu", str(gpu)]
            log_path = output / "controller-logs" / f"{task['name']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a")
            process = subprocess.Popen(command, cwd=args.source_root, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            task.update(process=process, log=log, gpu=gpu, controller_log=str(log_path))
            running[gpu] = task
        status_snapshot(output / "status.json", pending, running, finished, failed)
        if pending or running or not noise_released:
            time.sleep(10)
    status_snapshot(output / "status.json", pending, running, finished, failed)
    if failed:
        raise RuntimeError(f"{len(failed)} campaign tasks failed; see status.json")


if __name__ == "__main__":
    main()
