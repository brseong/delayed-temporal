#!/usr/bin/env python3
"""Select the fastest reproducible UBAI GPU family from benchmark logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from summarize_theta_selection import parse_log, read_manifest


PARTITIONS = {
    "rtx3090": "gpu1",
    "a10": "gpu2,gpu6",
    "rtx6000ada": "gpu3",
    "rtxa6000": "gpu4,gpu5",
}
GPU_MODEL_MARKERS = {
    "rtx3090": ("RTX 3090",),
    "a10": (" A10", "A10 ", "A10-SXM"),
    "rtx6000ada": ("RTX 6000 Ada",),
    "rtxa6000": ("RTX A6000",),
}


def gpu_model_matches_family(gpu_model: str, family: str) -> bool:
    """Reject manifests that were scheduled on a different physical GPU family."""

    try:
        markers = GPU_MODEL_MARKERS[family]
    except KeyError as error:
        raise ValueError(f"unknown GPU family: {family}") from error
    return any(marker.casefold() in gpu_model.casefold() for marker in markers)


def choose_family(
    grouped: dict[str, list[Any]],
    *,
    availability: dict[str, int],
    pre_rejected: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Require replay parity, then choose throughput with a 5% capacity tie-break."""

    reference = grouped.get("rtxa6000", [])
    if len(reference) != 2:
        raise ValueError("RTX A6000 requires exactly two benchmark replicas")
    reference_digests = {run.prediction_sha256 for run in reference}
    reference_correct = {run.correct for run in reference}
    if len(reference_digests) != 1 or len(reference_correct) != 1:
        raise ValueError("RTX A6000 reference benchmark is not reproducible")
    reference_digest = next(iter(reference_digests))
    reference_correct_count = next(iter(reference_correct))

    eligible: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = dict(pre_rejected or {})
    for family, runs in grouped.items():
        if family not in PARTITIONS:
            rejected[family] = "unknown GPU family"
            continue
        if len(runs) != 2:
            rejected[family] = "requires exactly two complete replicas"
            continue
        if any(not gpu_model_matches_family(run.gpu_model, family) for run in runs):
            rejected[family] = "reported GPU model does not match manifest family"
            continue
        if any(run.samples != 640 for run in runs):
            rejected[family] = "benchmark must measure exactly 640 images"
            continue
        if any(run.benchmark_seconds_per_image is None for run in runs):
            rejected[family] = "missing benchmark timing"
            continue
        if len({run.prediction_sha256 for run in runs}) != 1:
            rejected[family] = "replica prediction digests differ"
            continue
        if any(run.prediction_sha256 != reference_digest for run in runs):
            rejected[family] = "predictions differ from RTX A6000 reference"
            continue
        if any(run.correct != reference_correct_count for run in runs):
            rejected[family] = "correct count differs from RTX A6000 reference"
            continue
        timings = [float(run.benchmark_seconds_per_image) for run in runs]
        if any(not timing > 0.0 for timing in timings):
            rejected[family] = "non-positive timing"
            continue
        eligible[family] = {
            "median_seconds_per_image": median(timings),
            "replica_seconds_per_image": timings,
            "peak_memory_bytes": max(
                int(run.benchmark_peak_memory_bytes or 0) for run in runs
            ),
            "available_gpus": int(availability.get(family, 0)),
            "partition": PARTITIONS[family],
        }
    if not eligible:
        raise ValueError("no reproducible UBAI GPU family passed the benchmark")

    fastest = min(
        eligible,
        key=lambda family: eligible[family]["median_seconds_per_image"],
    )
    fastest_time = eligible[fastest]["median_seconds_per_image"]
    tied = [
        family
        for family, record in eligible.items()
        if record["median_seconds_per_image"] <= fastest_time * 1.05
    ]
    selected = min(
        tied,
        key=lambda family: (
            -eligible[family]["available_gpus"],
            eligible[family]["median_seconds_per_image"],
            family,
        ),
    )
    return selected, {
        "format_version": 1,
        "selected_family": selected,
        "selected_partition": PARTITIONS[selected],
        "eligible": eligible,
        "rejected": rejected,
        "reference_prediction_sha256": reference_digest,
        "reference_correct": reference_correct_count,
        "tie_fraction": 0.05,
    }


def failed_run_reason(spec: dict[str, str], log_dir: Path) -> str:
    """Classify the latest identity-matching partial benchmark log."""

    prefix = f"{spec['log_file']}.partial."
    candidates = sorted(
        (path for path in log_dir.glob(f"{spec['log_file']}.partial.*") if path.name.startswith(prefix)),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        if spec["source_commit"] not in text or f"gpu_family: {spec['gpu_family']}" not in text:
            continue
        if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
            return "OOM at batch size 32"
        if "nan" in text.casefold():
            return "NaN reported by benchmark"
        return "incomplete or failed evaluator log"
    return "missing complete benchmark log"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--available",
        action="append",
        default=[],
        metavar="FAMILY=COUNT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    availability: dict[str, int] = {}
    for item in args.available:
        family, separator, count = item.partition("=")
        if not separator or family not in PARTITIONS or not count.isdigit():
            raise ValueError(f"invalid --available value: {item}")
        availability[family] = int(count)

    grouped: dict[str, list[Any]] = {}
    failures: dict[str, list[str]] = {}
    for manifest in args.manifest:
        for spec in read_manifest(manifest):
            try:
                run = parse_log(spec, args.log_dir)
            except (FileNotFoundError, UnicodeDecodeError, ValueError) as error:
                reason = failed_run_reason(spec, args.log_dir)
                failures.setdefault(spec["gpu_family"], []).append(
                    f"{spec['run_id']}: {reason} ({type(error).__name__})"
                )
                continue
            grouped.setdefault(spec["gpu_family"], []).append(run)
    pre_rejected = {
        family: "; ".join(reasons) for family, reasons in failures.items()
    }
    selected, payload = choose_family(
        grouped,
        availability=availability,
        pre_rejected=pre_rejected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{selected}\t{PARTITIONS[selected]}")


if __name__ == "__main__":
    main()
