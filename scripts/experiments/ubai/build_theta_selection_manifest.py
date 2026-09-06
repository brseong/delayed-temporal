#!/usr/bin/env python3
"""Build immutable stage manifests for the UBAI theta selection pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASE_THETAS = (40, 80, 160, 320, 640, 1000, 1400, 2000, 2800, 4000)
FAMILY_PARTITIONS = {
    "rtx3090": "gpu1",
    "a10": "gpu2,gpu6",
    "rtx6000ada": "gpu3",
    "rtxa6000": "gpu4,gpu5",
}
FIELDS = (
    "run_id",
    "stage",
    "backend",
    "theta",
    "split",
    "expected_samples",
    "dataset_path",
    "dataset_fingerprint",
    "precision",
    "source_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "gpu_family",
    "quick_test",
    "benchmark_warmup_batches",
    "benchmark_measure_batches",
    "log_file",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("benchmark", "selection", "confirmation", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        default="/data/ubai-assets/datasets/imagenet_theta_selection_v1",
        help="Container-visible root holding train_seed0_5000 and validation_50000.",
    )
    parser.add_argument("--gpu-selection", type=Path)
    parser.add_argument("--gpu-family", choices=tuple(FAMILY_PARTITIONS))
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--extension", action="store_true")
    parser.add_argument(
        "--extra-theta",
        action="append",
        type=float,
        default=[],
        help="Add a positive theta candidate to a selection manifest; repeat as needed.",
    )
    return parser.parse_args()


def theta_text(theta: float) -> str:
    """Return a stable compact representation for manifest values and run IDs."""

    if theta <= 0:
        raise ValueError(f"theta must be positive: {theta}")
    return format(float(theta), ".15g")


def theta_run_id(prefix: str, theta: float) -> str:
    return f"{prefix}_{theta_text(theta).replace('.', 'p')}"


def row(
    *,
    run_id: str,
    stage: str,
    backend: str,
    theta: float,
    split: str,
    expected_samples: int,
    dataset_path: str,
    dataset_fingerprint: str,
    source_commit: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    gpu_family: str,
    quick_test: bool = False,
    benchmark: bool = False,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "stage": stage,
        "backend": backend,
        "theta": theta_text(theta),
        "split": split,
        "expected_samples": str(expected_samples),
        "dataset_path": dataset_path,
        "dataset_fingerprint": dataset_fingerprint,
        "precision": "float64",
        "source_commit": source_commit,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "gpu_family": gpu_family,
        "quick_test": "1" if quick_test else "0",
        "benchmark_warmup_batches": "5" if benchmark else "0",
        "benchmark_measure_batches": "20" if benchmark else "0",
        "log_file": f"{run_id}.log",
    }


def main() -> None:
    args = parse_arguments()
    if args.extra_theta and args.stage != "selection":
        raise ValueError("--extra-theta is valid only for selection manifests")
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    train = dataset_manifest["train_selection"]
    validation = dataset_manifest["validation"]
    train_path = str(Path(args.dataset_root) / "train_seed0_5000")
    validation_path = str(Path(args.dataset_root) / "validation_50000")
    if args.gpu_selection:
        gpu_family = json.loads(args.gpu_selection.read_text(encoding="utf-8"))["selected_family"]
    elif args.gpu_family:
        gpu_family = args.gpu_family
    else:
        raise ValueError("--gpu-family or --gpu-selection is required")

    common = {
        "source_commit": args.source_commit,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_sha256": args.checkpoint_sha256,
        "gpu_family": gpu_family,
    }
    rows: list[dict[str, str]] = []
    if args.stage == "benchmark":
        for replicate in range(2):
            rows.append(row(
                run_id=f"benchmark_{gpu_family}_rep_{replicate}",
                stage="benchmark",
                backend="spiking",
                theta=2000,
                split="train-selection-seed0-5000",
                expected_samples=640,
                dataset_path=train_path,
                dataset_fingerprint=train["fingerprint"],
                benchmark=True,
                **common,
            ))
    elif args.stage == "selection":
        thetas = sorted(set(BASE_THETAS) | set(args.extra_theta))
        if args.extension:
            thetas = sorted(set(thetas) | {5600, 8000})
        for theta in thetas:
            rows.append(row(
                run_id=theta_run_id("selection_theta", theta),
                stage="selection",
                backend="spiking",
                theta=theta,
                split="train-selection-seed0-5000",
                expected_samples=5000,
                dataset_path=train_path,
                dataset_fingerprint=train["fingerprint"],
                **common,
            ))
        rows.append(row(
            run_id="selection_dense_reference",
            stage="selection",
            backend="hf",
            theta=2000,
            split="train-selection-seed0-5000",
            expected_samples=5000,
            dataset_path=train_path,
            dataset_fingerprint=train["fingerprint"],
            **common,
        ))
    else:
        if args.selection_json is None:
            raise ValueError("confirmation/full manifest requires --selection-json")
        selection = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selected = float(selection["selected_theta"])
        if args.stage == "confirmation":
            rows.append(row(
                run_id=theta_run_id("replay_theta", selected),
                stage="replay",
                backend="spiking",
                theta=selected,
                split="train-selection-seed0-5000",
                expected_samples=5000,
                dataset_path=train_path,
                dataset_fingerprint=train["fingerprint"],
                **common,
            ))
            for theta_value in selection["validation_neighbors"]:
                theta = float(theta_value)
                rows.append(row(
                    run_id=theta_run_id("validation_theta", theta),
                    stage="validation",
                    backend="spiking",
                    theta=theta,
                    split="validation",
                    expected_samples=5000,
                    dataset_path=validation_path,
                    dataset_fingerprint=validation["quick_prefix_fingerprint"],
                    quick_test=True,
                    **common,
                ))
            rows.append(row(
                run_id="validation_dense_reference",
                stage="validation",
                backend="hf",
                theta=2000,
                split="validation",
                expected_samples=5000,
                dataset_path=validation_path,
                dataset_fingerprint=validation["quick_prefix_fingerprint"],
                quick_test=True,
                **common,
            ))
        else:
            for backend in ("spiking", "hf"):
                rows.append(row(
                    run_id=f"full_{backend}",
                    stage="full",
                    backend=backend,
                    theta=selected if backend == "spiking" else 2000,
                    split="validation",
                    expected_samples=50000,
                    dataset_path=validation_path,
                    dataset_fingerprint=validation["fingerprint"],
                    **common,
                ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            dialect="excel-tab",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)}\t{gpu_family}\t{FAMILY_PARTITIONS[gpu_family]}")


if __name__ == "__main__":
    main()
