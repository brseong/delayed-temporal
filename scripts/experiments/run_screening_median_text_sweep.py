#!/usr/bin/env python3
"""Prepare, run, resume, and summarize the screening-median text sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.screening_median_model_sweep import (
    DEADLINE_MARGIN_SIGMA_RATIO,
    load_screening_median,
    protocol_id,
)
from scripts.experiments.screening_median_text_sweep import (
    TEXT_INITIAL_ALPHAS,
    TEXT_MODELS,
    TextCell,
    expected_text_cells,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity, local_gpu


ASSETS = Path("/data/delayed-temporal/artifacts/assets/conversion-comparison-text-v1")
MODEL_CONFIG = {
    "roberta_base": {
        "family": "roberta",
        "evaluator_family": "roberta",
        "task": "sst2",
        "activation": "gelu",
        "model_id": ASSETS / "checkpoints/roberta",
        "calibration_dataset": ASSETS / "sst2/train_seed0_5000",
        "calibration_fingerprint": "35d5874329335756",
        "evaluation_dataset": ASSETS / "sst2/validation_872",
        "evaluation_fingerprint": "e3182d90fd436f61",
    },
    "gpt2": {
        "family": "gpt2",
        "evaluator_family": "gpt2",
        "task": "wikitext2",
        "activation": "gelu_new",
        "model_id": ASSETS / "checkpoints/gpt2",
        "calibration_dataset": ASSETS / "wikitext2/train_nonempty_seed0_5000",
        "calibration_fingerprint": "f1506153809011c4",
        "evaluation_dataset": ASSETS / "wikitext2/test_nonempty_2891",
        "evaluation_fingerprint": "38d46c7ecf7254ca",
    },
}


def complete(path: Path) -> bool:
    """Return whether an atomic result artifact is complete."""

    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def wait_for_gpu(gpu: int) -> None:
    """Wait for one locally idle GPU without disturbing another process."""

    while True:
        sample = local_gpu.gpu_activity(gpu_ids=(gpu,))[gpu]
        if local_gpu.gpu_available(sample):
            return
        time.sleep(15)


def preparation_command(args: argparse.Namespace, model: str, gpu: int) -> list[str]:
    """Build one complete clean text-model preparation command."""

    config = MODEL_CONFIG[model]
    output = args.output_root / "prepare" / model
    command = [
        args.python_bin,
        "-u",
        str(args.source_root / "scripts/experiments/run_full_calibrated_text_comparison.py"),
        "--family",
        config["family"],
        "--source-root",
        str(args.source_root),
        "--expected-commit",
        args.expected_commit,
        "--model-id",
        str(config["model_id"]),
        "--calibration-dataset-path",
        str(config["calibration_dataset"]),
        "--calibration-dataset-fingerprint",
        config["calibration_fingerprint"],
        "--evaluation-dataset-path",
        str(config["evaluation_dataset"]),
        "--evaluation-dataset-fingerprint",
        config["evaluation_fingerprint"],
        "--gpu",
        str(gpu),
        "--host-label",
        "local",
        "--python-bin",
        args.python_bin,
        "--output-root",
        str(output),
        "--runtime-root",
        str(args.runtime_root / "prepare" / model),
    ]
    if gpu < 4:
        command.append("--campaign-extra-local-gpus")
    return command


def prepare_one(args: argparse.Namespace, model: str, gpu: int) -> None:
    """Prepare a clean baseline and calibration after the selected GPU is idle."""

    output = args.output_root / "prepare" / model
    if complete(output / "result.json"):
        return
    wait_for_gpu(gpu)
    log_path = output / "controller.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(
            preparation_command(args, model, gpu),
            cwd=args.source_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def prepare(args: argparse.Namespace) -> None:
    """Prepare RoBERTa and GPT-2 concurrently, then freeze the protocol."""

    assignments = tuple(zip(TEXT_MODELS, args.gpus[: len(TEXT_MODELS)], strict=True))
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [executor.submit(prepare_one, args, model, gpu) for model, gpu in assignments]
        for future in futures:
            future.result()
    write_protocol(args)


def prepared_resource(args: argparse.Namespace, model: str) -> dict[str, Any]:
    """Build one immutable protocol resource from a complete preparation."""

    config = MODEL_CONFIG[model]
    root = args.output_root / "prepare" / model
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if result.get("state") != "complete" or manifest.get("source_commit") != args.expected_commit:
        raise ValueError(f"incomplete or mismatched text preparation: {model}")
    calibration = root / "calibration.json"
    collect = result["phases"]["collect"]
    evaluator_family = config["evaluator_family"]
    ann = result["phases"]["ann"]["metrics"]
    clean = result["phases"]["snn"]["metrics"]
    return {
        "model": model,
        "family": config["family"],
        "evaluator_family": evaluator_family,
        "task": config["task"],
        "activation": config["activation"],
        "model_id": str(config["model_id"].resolve(strict=True)),
        "checkpoint_sha256": identity.artifact_identity(config["model_id"])["aggregate_sha256"],
        "preparation_root": str(root.resolve(strict=True)),
        "calibration_sha256": identity.sha256_file(calibration),
        "calibration_site_count": int(collect["calibration_site_count"]),
        "calibration_dataset_path": str(config["calibration_dataset"].resolve(strict=True)),
        "calibration_dataset_fingerprint": config["calibration_fingerprint"],
        "evaluation_dataset_path": str(config["evaluation_dataset"].resolve(strict=True)),
        "evaluation_dataset_fingerprint": config["evaluation_fingerprint"],
        "evaluation_population": manifest["evaluation_dataset"],
        "cache_dir": "/data/nas/datasets",
        "ann_reference": ann,
        "converted_clean": clean,
    }


def write_protocol(args: argparse.Namespace) -> None:
    """Write or verify the text sweep protocol independently of alpha values."""

    pair = load_screening_median(args.hardware_summary)
    models = {model: prepared_resource(args, model) for model in TEXT_MODELS}
    protocol_identity = {
        "contract": "screening_median_text_timing_noise_v1",
        "source_commit": args.expected_commit,
        "hardware": {
            "summary_sha256": pair.summary_sha256,
            "phi_np": {"validation_rt": pair.phi_np.validation_rt},
            "phi_nl": {"validation_rt": pair.phi_nl.validation_rt},
        },
        "models": {
            model: {
                "checkpoint_sha256": resource["checkpoint_sha256"],
                "calibration_sha256": resource["calibration_sha256"],
                "evaluation_population": resource["evaluation_population"],
                "ann_reference": resource["ann_reference"],
                "converted_clean": resource["converted_clean"],
            }
            for model, resource in models.items()
        },
        "precision": "float64",
        "deadline_margin_sigma_ratio": DEADLINE_MARGIN_SIGMA_RATIO,
        "time_noise_scope": "all_encoder_outputs",
        "raw_timestamp_contract": "delivered_raw_timestamp_v1",
        "exponential_difference_internal_noise": "fixed_on_v1",
    }
    payload = {
        "schema_version": 1,
        "protocol_id": protocol_id(protocol_identity),
        "identity": protocol_identity,
        "resources": {
            "source_root": str(args.source_root),
            "hardware_summary": str(args.hardware_summary),
            "models": models,
        },
    }
    runtime_files.immutable_json(args.output_root / "protocol.json", payload)


def load_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Load the complete frozen protocol or require preparation first."""

    path = args.output_root / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError("run --phase prepare before text conditions")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != identity.json_sha256(payload.get("identity", {})):
        raise ValueError("text sweep protocol identity differs")
    return payload


def cell_command(args: argparse.Namespace, cell: TextCell, gpu: int) -> list[str]:
    """Build one authenticated text condition command."""

    return [
        args.python_bin,
        "-u",
        str(args.source_root / "scripts/experiments/run_screening_median_text_condition.py"),
        "--protocol",
        str(args.output_root / "protocol.json"),
        "--phase",
        cell.phase,
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


def run_group(args: argparse.Namespace, gpu: int, cells: list[TextCell]) -> None:
    """Run a stable per-GPU cell sequence with one safe retry."""

    for cell in cells:
        if complete(args.output_root / cell.relative_path / "result.json"):
            continue
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(2):
            wait_for_gpu(gpu)
            try:
                subprocess.run(cell_command(args, cell, gpu), cwd=args.source_root, check=True)
                last_error = None
                break
            except subprocess.CalledProcessError as error:
                last_error = error
                if attempt == 0:
                    time.sleep(5)
        if last_error is not None:
            raise last_error


def run_cells(args: argparse.Namespace, *, phase: str, alphas: Iterable[str]) -> None:
    """Distribute a deterministic cell population across local GPUs."""

    protocol = load_protocol(args)
    cells = expected_text_cells(phase=phase, alphas=alphas)
    assignments = tuple(
        (cell, args.gpus[index % len(args.gpus)])
        for index, cell in enumerate(cells)
    )
    request = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "phase": phase,
        "created_at_unix_ns": time.time_ns(),
        "alphas": sorted({cell.alpha for cell in cells}),
        "cells": [
            {
                "model": cell.model,
                "alpha": cell.alpha,
                "seed": cell.seed,
                "evaluation_samples": cell.evaluation_samples,
                "physical_gpu": gpu,
            }
            for cell, gpu in assignments
        ],
    }
    runtime_files.new_json(
        args.output_root / "requests" / f"{request['created_at_unix_ns']}_{phase}.json",
        request,
    )
    groups = {gpu: [] for gpu in args.gpus}
    for cell, gpu in assignments:
        groups[gpu].append(cell)
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [
            executor.submit(run_group, args, gpu, rows)
            for gpu, rows in groups.items()
            if rows
        ]
        for future in futures:
            future.result()


def verify_smoke_gate(args: argparse.Namespace) -> None:
    """Require all six measured-pair smoke replicas before formal inference."""

    missing = [
        str(cell.relative_path)
        for cell in expected_text_cells(phase="smoke", alphas=("1",))
        if not complete(args.output_root / cell.relative_path / "result.json")
    ]
    if missing:
        raise RuntimeError(f"formal text sweep requires complete smoke cells: {missing}")


def aggregate(args: argparse.Namespace) -> None:
    """Regenerate all text tables and figures from complete cells."""

    subprocess.run(
        [
            args.python_bin,
            "-u",
            str(args.source_root / "scripts/analysis/summarize_screening_median_text_sweep.py"),
            "--input-root",
            str(args.output_root),
            "--output-dir",
            str(args.output_root),
            "--vision-root",
            str(args.vision_root),
        ],
        cwd=args.source_root,
        check=True,
    )


# @lat: [[evaluation#Evaluation and Verification#Screening Median Text Timing Noise Sweep#Campaign Execution]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--phase", choices=("prepare", "smoke", "formal", "aggregate", "all"), required=True
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--hardware-summary", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", default=TEXT_INITIAL_ALPHAS)
    parser.add_argument("--gpus", type=int, nargs="+", default=tuple(range(8)))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--vision-root",
        type=Path,
        default=Path(
            "/data/delayed-temporal/artifacts/results/"
            "model_scale_screening_median_raw_timestamp_ed_v3"
        ),
    )
    parser.add_argument("--runtime-root", type=Path, default=Path("/data/dtr/text-sm1"))
    parser.add_argument("--python-bin", default="/opt/conda/envs/dt/bin/python")
    args = parser.parse_args()
    if socket.gethostname() != "baekryun-cuda129":
        raise ValueError("text sweep controller requires baekryun")
    if (
        len(set(args.gpus)) != len(args.gpus)
        or not args.gpus
        or any(gpu not in range(8) for gpu in args.gpus)
    ):
        raise ValueError("GPU list must contain unique indices from 0 through 7")
    if len(args.gpus) < len(TEXT_MODELS):
        raise ValueError("text preparation requires at least two GPUs")
    args.source_root = args.source_root.resolve(strict=True)
    args.hardware_summary = args.hardware_summary.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.vision_root = args.vision_root.resolve(strict=True)
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.source_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if actual_commit != args.expected_commit or dirty:
        raise ValueError("text sweep requires the clean frozen source commit")

    if args.phase in {"prepare", "all"}:
        prepare(args)
    if args.phase in {"smoke", "all"}:
        run_cells(args, phase="smoke", alphas=("1",))
    if args.phase in {"formal", "all"}:
        verify_smoke_gate(args)
        run_cells(args, phase="formal", alphas=args.alphas)
    if args.phase in {"aggregate", "all"}:
        aggregate(args)


if __name__ == "__main__":
    main()
