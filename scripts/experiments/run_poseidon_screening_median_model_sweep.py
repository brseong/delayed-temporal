#!/usr/bin/env python3
"""Prepare, run, resume, and summarize the screening median model sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_full_calibrated_vit_comparison import (
    TAG as VIT_PREPARE_TAG,
    source_identity,
)
from scripts.experiments.screening_median_model_sweep import (
    DEADLINE_MARGIN_SIGMA_RATIO,
    INITIAL_ALPHAS,
    MODELS,
    Cell,
    canonical_alphas,
    expected_cells,
    load_screening_median,
    protocol_id,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity, local_gpu


IMAGENET_CALIBRATION_FINGERPRINT = "cabf903d14d1b1ac"
IMAGENET_EVALUATION_FINGERPRINT = "746378cc7befed99"
DEFAULT_CALIBRATION_DATASET = Path(
    "/data/delayed-temporal/artifacts/assets/theta-selection-v1/"
    "datasets/imagenet_theta_selection_v1/train_seed0_5000"
)
DEFAULT_EVALUATION_DATASET = Path(
    "/data/delayed-temporal/artifacts/assets/theta-selection-v1/"
    "datasets/imagenet_theta_selection_v1/validation_50000"
)
DEFAULT_MODEL_PATHS = {
    "cct7": Path("/data/nas/cct_7_3x1_32_cifar10_300epochs.pth"),
    "imagenet_vit_small": Path("/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k"),
    "imagenet_vit_base": Path("/data/nas/vit_base_patch16_224.augreg2_in21k_ft_in1k"),
}


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _run_logged(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command exited with status {completed.returncode}: {' '.join(command)}"
        )


def _base_environment(source_root: Path, gpu: int, runtime: Path) -> dict[str, str]:
    runtime.mkdir(parents=True, exist_ok=True)
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        DELAYED_TEMPORAL_ARTIFACTS_ROOT=str(runtime.parents[1]),
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


def _admit_gpu_zero() -> dict[str, Any]:
    policy = dict(local_gpu.DEFAULT_ADMISSION_POLICY)
    policy["max_memory_used_mib"] = 4096.0
    samples = []
    for check in range(2):
        sample = local_gpu.gpu_activity(gpu_ids=(0,))[0]
        samples.append(sample)
        if not local_gpu.gpu_available(sample, policy):
            raise RuntimeError("GPU 0 does not satisfy the campaign admission policy")
        if check == 0:
            time.sleep(10)
    return {"policy": policy, "samples": samples}


def _prepare_cct(args: argparse.Namespace) -> None:
    output = args.output_root / "prepare/cct7"
    result_path = output / "prepare_result.json"
    manifest_path = output / "prepare_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_commit": args.expected_commit,
        "source_hashes": source_identity(args.source_root, args.expected_commit),
        "checkpoint_sha256": identity.sha256_file(args.cct_checkpoint),
        "dataset": identity.artifact_identity(
            args.cct_dataset_root / "cifar-10-batches-py"
        ),
        "calibration_samples": 5_000,
        "evaluation_samples": 10_000,
        "precision": "float64",
        "batch_size": 8,
    }
    if _complete(result_path):
        if not manifest_path.is_file():
            raise ValueError("completed CCT preparation has no immutable manifest")
        runtime_files.immutable_json(manifest_path, manifest)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("summary_sha256")
            != identity.sha256_file(output / "summary.json")
            or result.get("calibration_sha256")
            != identity.sha256_file(output / "calibration.json")
        ):
            raise ValueError("completed CCT preparation artifact checksum differs")
        return
    runtime_files.immutable_json(manifest_path, manifest)
    lock_path = args.output_root / "runtime/gpu-locks/poseidon-gpu-0.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission = _admit_gpu_zero()
        command = [
            args.python_bin,
            "-u",
            str(args.source_root / "scripts/evaluation/error_analysis_cct.py"),
            "--checkpoint",
            str(args.cct_checkpoint),
            "--dataset-root",
            str(args.cct_dataset_root),
            "--output-dir",
            str(output),
            "--samples",
            "10000",
            "--calibration-samples",
            "5000",
            "--batch-size",
            "8",
            "--device",
            "cuda",
            "--noise-fractions",
            "--log-noise-fraction",
            "0",
            "--deadline-margin",
            repr(DEADLINE_MARGIN_SIGMA_RATIO),
            "--seed",
            "0",
        ]
        started = time.monotonic()
        _run_logged(
            command,
            log_path=output / "logs/prepare.log",
            cwd=args.source_root,
            environment=_base_environment(
                args.source_root,
                0,
                args.runtime_root / "prepare/cct7",
            ),
        )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        clean = [
            row for row in summary["conditions"] if row["condition"] == "converted_clean"
        ]
        if (
            len(clean) != 1
            or summary.get("dense_learned_modules") != []
            or summary["ann_reference"]["samples"] != 10_000
            or clean[0]["samples"] != 10_000
        ):
            raise ValueError("CCT preparation result differs from the fixed protocol")
        runtime_files.atomic_json(
            result_path,
            {
                "state": "complete",
                "elapsed_seconds": time.monotonic() - started,
                "admission": admission,
                "command": command,
                "summary_sha256": identity.sha256_file(output / "summary.json"),
                "calibration_sha256": identity.sha256_file(output / "calibration.json"),
            },
        )


def _vit_prepare_paths(
    output_root: Path,
    runtime_root: Path,
    model: str,
) -> tuple[Path, Path]:
    output = output_root / "logs/conversion_comparison" / VIT_PREPARE_TAG / "vit" / model
    runtime = runtime_root / VIT_PREPARE_TAG / "vit" / model
    return output, runtime


def _prepare_vit(args: argparse.Namespace, model: str, gpu: int) -> None:
    output, runtime = _vit_prepare_paths(args.output_root, args.runtime_root, model)
    if _complete(output / "result.json"):
        return
    model_path = args.vit_small_model if model == "imagenet_vit_small" else args.vit_base_model
    command = [
        args.python_bin,
        "-u",
        str(args.source_root / "scripts/experiments/run_full_calibrated_vit_comparison.py"),
        "--model-key",
        model,
        "--source-root",
        str(args.source_root),
        "--expected-commit",
        args.expected_commit,
        "--model-id",
        str(model_path),
        "--calibration-dataset-path",
        str(args.calibration_dataset_path),
        "--calibration-dataset-fingerprint",
        IMAGENET_CALIBRATION_FINGERPRINT,
        "--evaluation-dataset-path",
        str(args.evaluation_dataset_path),
        "--evaluation-dataset-fingerprint",
        IMAGENET_EVALUATION_FINGERPRINT,
        "--image-preprocessing-config",
        str(model_path / "preprocessor_config.json"),
        "--batch-size",
        "32",
        "--gpu",
        str(gpu),
        "--host-label",
        "poseidon",
        "--python-bin",
        args.python_bin,
        "--output-root",
        str(output),
        "--runtime-root",
        str(runtime),
    ]
    environment = _base_environment(args.source_root, gpu, runtime)
    environment["DELAYED_TEMPORAL_ARTIFACTS_ROOT"] = str(args.output_root)
    environment["DELAYED_TEMPORAL_RUNTIME_ROOT"] = str(args.runtime_root)
    _run_logged(
        command,
        log_path=args.output_root / f"controller/prepare_{model}.log",
        cwd=args.source_root,
        environment=environment,
    )


def _prepare_models(args: argparse.Namespace) -> None:
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = (
            executor.submit(_prepare_cct, args),
            executor.submit(_prepare_vit, args, "imagenet_vit_small", 1),
            executor.submit(_prepare_vit, args, "imagenet_vit_base", 2),
        )
        for future in futures:
            future.result()


def _build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    pair = load_screening_median(args.hardware_summary)
    cct_root = args.output_root / "prepare/cct7"
    cct_dataset = identity.artifact_identity(
        args.cct_dataset_root / "cifar-10-batches-py"
    )
    cct_summary = json.loads((cct_root / "summary.json").read_text(encoding="utf-8"))
    cct_clean = next(
        row for row in cct_summary["conditions"] if row["condition"] == "converted_clean"
    )
    models: dict[str, dict[str, Any]] = {
        "cct7": {
            "model_key": "cct7",
            "checkpoint_sha256": identity.sha256_file(args.cct_checkpoint),
            "calibration_sha256": identity.sha256_file(cct_root / "calibration.json"),
            "evaluation_population": {
                "dataset": "cifar10",
                "split": "test",
                "selection": "complete-test",
                "samples": 10_000,
                "aggregate_sha256": cct_dataset["aggregate_sha256"],
                "bytes": cct_dataset["bytes"],
            },
            "ann_reference": cct_summary["ann_reference"],
            "converted_clean": cct_clean,
        }
    }
    resources: dict[str, dict[str, Any]] = {
        "cct7": {
            **models["cct7"],
            "model_id": str(args.cct_checkpoint),
            "dataset_root": str(args.cct_dataset_root),
            "calibration_path": str((cct_root / "calibration.json").resolve()),
        }
    }
    for model in ("imagenet_vit_small", "imagenet_vit_base"):
        root, _ = _vit_prepare_paths(args.output_root, args.runtime_root, model)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        calibration_path = root / "calibration.json"
        if (
            result.get("state") != "complete"
            or manifest.get("source_commit") != args.expected_commit
            or result.get("calibration_sha256")
            != identity.sha256_file(calibration_path)
        ):
            raise ValueError(f"{model} preparation identity differs")
        resource_identity = {
            "model_key": model,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "calibration_sha256": result["calibration_sha256"],
            "evaluation_population": manifest["evaluation_dataset"],
            "ann_reference": result["phases"]["ann"]["metrics"],
            "converted_clean": result["phases"]["snn"]["metrics"],
        }
        models[model] = resource_identity
        resources[model] = {
            **resource_identity,
            "model_id": manifest["checkpoint_path"],
            "calibration_source": str(root.resolve()),
            "calibration_dataset_path": str(args.calibration_dataset_path),
            "calibration_dataset_fingerprint": IMAGENET_CALIBRATION_FINGERPRINT,
            "evaluation_dataset_path": str(args.evaluation_dataset_path),
            "evaluation_dataset_fingerprint": IMAGENET_EVALUATION_FINGERPRINT,
            "image_preprocessing_config": str(
                Path(manifest["checkpoint_path"]) / "preprocessor_config.json"
            ),
        }
    hardware_identity = pair.as_dict()
    hardware_identity.pop("summary_path")
    identity_payload = {
        "contract": "screening_median_model_timing_noise_v1",
        "source_commit": args.expected_commit,
        "hardware": hardware_identity,
        "models": models,
        "precision": "float64",
        "deadline_margin_sigma_ratio": DEADLINE_MARGIN_SIGMA_RATIO,
        "time_noise_scope": "model_wide",
        "time_noise_mean": 0.0,
        "independent_event_noise": True,
        "weight_noise_std": 0.0,
        "bias_noise_std": 0.0,
        "static_mismatch": False,
    }
    return {
        "schema_version": 1,
        "protocol_id": protocol_id(identity_payload),
        "identity": identity_payload,
        "resources": {
            "source_root": str(args.source_root),
            "hardware_summary": str(args.hardware_summary),
            "models": resources,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    identity.verify_clean_checkout(args.source_root, args.expected_commit)
    _prepare_models(args)
    protocol = _build_protocol(args)
    runtime_files.immutable_json(args.output_root / "protocol.json", protocol)
    return protocol


def _load_protocol(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError("run --phase prepare before model conditions")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != identity.json_sha256(protocol.get("identity", {})):
        raise ValueError("campaign protocol identity differs")
    identity.verify_clean_checkout(args.source_root, args.expected_commit)
    if protocol["identity"]["source_commit"] != args.expected_commit:
        raise ValueError("campaign source commit differs")
    return protocol


def _cell_gpu(cell: Cell) -> int:
    if cell.model == "cct7":
        return 0
    if cell.model == "imagenet_vit_small":
        return 1
    try:
        alpha_index = INITIAL_ALPHAS.index(cell.alpha)
        parity = (alpha_index * len((0, 1, 2)) + cell.seed) % 2
    except ValueError:
        digest = hashlib.sha256(f"{cell.alpha}:{cell.seed}".encode()).digest()
        parity = digest[0] % 2
    return 2 + parity


def _verify_smoke_gate(args: argparse.Namespace) -> None:
    """Require all nine engineering smoke cells before any formal request."""

    expected = expected_cells(phase="smoke", alphas=("1",))
    incomplete = []
    for cell in expected:
        result_path = args.output_root / cell.relative_path / "result.json"
        if not _complete(result_path):
            incomplete.append(str(cell.relative_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result.get("metrics", {})
        counts = metrics.get("physical_counts", {})
        if (
            metrics.get("total") != 500
            or not math.isfinite(float(metrics.get("accuracy", float("nan"))))
            or int(counts.get("events", 0)) <= 0
            or int(counts.get("outputs", 0)) <= 0
        ):
            incomplete.append(str(cell.relative_path))
    if incomplete:
        raise RuntimeError(
            "formal execution requires nine complete smoke cells: "
            + ", ".join(incomplete)
        )


def _cell_command(
    args: argparse.Namespace,
    cell: Cell,
    *,
    gpu: int,
) -> list[str]:
    return [
        args.python_bin,
        "-u",
        str(args.source_root / "scripts/experiments/run_screening_median_model_condition.py"),
        "--protocol",
        str(args.output_root / "protocol.json"),
        "--model",
        cell.model,
        "--alpha",
        cell.alpha,
        "--seed",
        str(cell.seed),
        "--evaluation-samples",
        str(cell.evaluation_samples),
        "--gpu",
        str(gpu),
        "--output-dir",
        str(args.output_root / cell.relative_path),
        "--runtime-dir",
        str(args.runtime_root / cell.relative_path),
        "--python-bin",
        args.python_bin,
    ]


def _request(args: argparse.Namespace, phase: str, cells: tuple[Cell, ...]) -> Path:
    request = {
        "schema_version": 1,
        "protocol_id": _load_protocol(args)["protocol_id"],
        "phase": phase,
        "created_at_unix_ns": time.time_ns(),
        "alphas": sorted({cell.alpha for cell in cells}),
        "cells": [
            {
                "model": cell.model,
                "alpha": cell.alpha,
                "seed": cell.seed,
                "evaluation_samples": cell.evaluation_samples,
            }
            for cell in cells
        ],
    }
    path = args.output_root / "requests" / f"{request['created_at_unix_ns']}_{phase}.json"
    runtime_files.new_json(path, request)
    return path


def _run_cell_group(
    args: argparse.Namespace,
    rows: list[tuple[Cell, int]],
) -> None:
    for cell, gpu in rows:
        result = args.output_root / cell.relative_path / "result.json"
        if _complete(result):
            continue
        command = _cell_command(args, cell, gpu=gpu)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                subprocess.run(command, cwd=args.source_root, check=True)
                last_error = None
                break
            except subprocess.CalledProcessError as error:
                last_error = error
                if attempt == 0:
                    time.sleep(5)
        if last_error is not None:
            raise last_error


def run_cells(
    args: argparse.Namespace,
    *,
    phase: str,
    alphas: Iterable[str],
) -> None:
    _load_protocol(args)
    cells = expected_cells(phase=phase, alphas=alphas)
    _request(args, phase, cells)
    groups: dict[int, list[tuple[Cell, int]]] = {gpu: [] for gpu in args.gpus}
    for cell in cells:
        gpu = _cell_gpu(cell)
        if gpu not in groups:
            raise ValueError(f"required GPU {gpu} was not selected")
        groups[gpu].append((cell, gpu))
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [
            executor.submit(_run_cell_group, args, rows)
            for rows in groups.values()
            if rows
        ]
        for future in futures:
            future.result()


def aggregate(args: argparse.Namespace) -> None:
    _load_protocol(args)
    command = [
        args.python_bin,
        "-u",
        str(args.source_root / "scripts/analysis/summarize_screening_median_model_sweep.py"),
        "--input-root",
        str(args.output_root),
        "--output-dir",
        str(args.output_root),
    ]
    subprocess.run(command, cwd=args.source_root, check=True)


def status(args: argparse.Namespace) -> None:
    requested: dict[tuple[str, str, str, int, int], Cell] = {}
    for request_path in sorted(args.output_root.glob("requests/*.json")):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        phase = request["phase"]
        for row in request["cells"]:
            cell = Cell(
                phase=phase,
                model=row["model"],
                alpha=row["alpha"],
                seed=int(row["seed"]),
                evaluation_samples=int(row["evaluation_samples"]),
            )
            requested[cell.identity] = cell
    rows = []
    for cell in requested.values():
        run_root = args.output_root / cell.relative_path
        result_path = run_root / "result.json"
        manifest_path = run_root / "manifest.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            state = result.get("state", "unknown")
            log_file = result.get("log_file")
        else:
            state, log_file = "pending", None
        gpu = None
        if manifest_path.is_file():
            gpu = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "physical_gpu"
            )
        rows.append(
            {
                "phase": cell.phase,
                "model": cell.model,
                "alpha": cell.alpha,
                "seed": cell.seed,
                "state": state,
                "gpu": gpu,
                "run_root": str(run_root),
                "log_file": log_file,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        state = row["state"]
        counts[state] = counts.get(state, 0) + 1
    print(json.dumps({"counts": counts, "runs": rows}, indent=2, sort_keys=True))


# @lat: [[evaluation#Evaluation and Verification#Screening Median Model Timing Noise Sweep#Campaign Execution]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--phase",
        choices=("prepare", "smoke", "formal", "aggregate", "status", "all"),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--hardware-summary", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", default=INITIAL_ALPHAS)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/data/dtr/screening-median-v1"),
    )
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    parser.add_argument("--cct-checkpoint", type=Path, default=DEFAULT_MODEL_PATHS["cct7"])
    parser.add_argument(
        "--cct-dataset-root",
        type=Path,
        default=Path("/data/nas/SNN-Verification_Optimization"),
    )
    parser.add_argument(
        "--vit-small-model",
        type=Path,
        default=DEFAULT_MODEL_PATHS["imagenet_vit_small"],
    )
    parser.add_argument(
        "--vit-base-model",
        type=Path,
        default=DEFAULT_MODEL_PATHS["imagenet_vit_base"],
    )
    parser.add_argument(
        "--calibration-dataset-path",
        type=Path,
        default=DEFAULT_CALIBRATION_DATASET,
    )
    parser.add_argument(
        "--evaluation-dataset-path",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
    )
    args = parser.parse_args()
    args.source_root = args.source_root.resolve(strict=True)
    args.hardware_summary = args.hardware_summary.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runtime_root = args.runtime_root.resolve()
    if args.phase not in {"status", "aggregate"}:
        args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.phase in {"prepare", "all"}:
        args.cct_checkpoint = args.cct_checkpoint.resolve(strict=True)
        args.cct_dataset_root = args.cct_dataset_root.resolve(strict=True)
        args.vit_small_model = args.vit_small_model.resolve(strict=True)
        args.vit_base_model = args.vit_base_model.resolve(strict=True)
        args.calibration_dataset_path = args.calibration_dataset_path.resolve(strict=True)
        args.evaluation_dataset_path = args.evaluation_dataset_path.resolve(strict=True)
    args.alphas = canonical_alphas(args.alphas)
    if tuple(sorted(set(args.gpus))) != (0, 1, 2, 3):
        raise ValueError("the Poseidon campaign requires GPUs 0, 1, 2, and 3")
    if args.phase not in {"status", "aggregate"} and socket.gethostname() != "poseidon1":
        raise ValueError("model sweep execution requires poseidon1")

    if args.phase in {"prepare", "all"}:
        prepare(args)
    if args.phase in {"smoke", "all"}:
        run_cells(args, phase="smoke", alphas=("1",))
    if args.phase in {"formal", "all"}:
        _verify_smoke_gate(args)
        run_cells(args, phase="formal", alphas=args.alphas)
    if args.phase in {"aggregate", "all"}:
        aggregate(args)
    if args.phase == "status":
        status(args)


if __name__ == "__main__":
    main()
