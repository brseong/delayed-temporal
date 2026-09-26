#!/usr/bin/env python3
"""Plot time steps per phi operation for the source-frozen ViT clock sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = (
    ROOT
    / "artifacts/logs/clock_driven/"
    "vit_base_clock_driven_imagenet500_full_conversion_float64_fine_v3"
)
OUTPUT = ROOT / "artifacts/figures/ViT-clock-window-step-distribution"
RECORDS = CAMPAIGN / "diagnostics/window_step_distribution.json"
PLOTTED_STEP = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_count(lower: float, upper: float, width: float) -> int:
    """Match the source-frozen clock's first non-earlier edge rule."""
    from utils.transforms.clock import _scalar_ceil_steps

    return _scalar_ceil_steps(upper, width) - _scalar_ceil_steps(lower, width)


def _capture_windows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one clocked image and observe each declared phi operation window."""
    source = Path(experiment["source_root"])
    if not source.is_dir():
        raise FileNotFoundError(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / "src/transformers/src"))
    sys.path.insert(2, str(source / "src/spikingjelly"))

    from scripts.analysis import evaluate_calibrated_vit as wrapper
    from utils.transforms import noise

    calibration_path = CAMPAIGN / "calibration/vit_base_local_ranges.json"
    if _sha256(calibration_path) != experiment["calibration_sha256"]:
        raise ValueError("the frozen calibration differs from the clock sweep")

    captured: list[dict[str, Any]] = []
    original_quantize = noise.quantize_encoder_output
    original_argv = sys.argv

    def observe(time: Any, domain: Any, *, site: str) -> tuple[Any, Any]:
        captured.append({
            "site": site,
            "time_min": float(domain.min),
            "time_max": float(domain.max),
        })
        return original_quantize(time, domain, site=site)

    noise.quantize_encoder_output = observe
    os.environ.update({
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    sys.argv = [
        original_argv[0],
        "--source-root", str(source),
        "--calibration-dataset-path", experiment["calibration_dataset_path"],
        "--calibration-dataset-fingerprint", experiment["calibration_dataset_fingerprint"],
        "--gelu-cubic-implementation", "phi_nl_psi_ed",
        "--gelu-cubic-floor", "1e-5",
        "--experiment_name", "clock_window_step_distribution",
        "--device", "cuda",
        "--model_backend", "spiking",
        "--model_id", experiment["checkpoint_path"],
        "--dataset_id", "imagenet-1k",
        "--evaluation-dataset-path", str(CAMPAIGN / "assets/validation_first_500"),
        "--evaluation-split", "validation",
        "--evaluation-shard-count", "500",
        "--evaluation-shard-index", "0",
        "--max_eval_batches", "1",
        "--image-preprocessing-config", str(source / "scripts/configs/vit_timm_preprocessing.json"),
        "--batch_size", str(experiment["batch_size"]),
        "--precision", experiment["precision"],
        "--source-commit", experiment["source_commit"],
        "--checkpoint-sha256", experiment["checkpoint_sha256"],
        "--no-tensorboard",
        "--spiking-layernorm", "--spiking-ln-mul", "--spiking-ln-log",
        "--spiking-ln-expdiff", "--spiking-attention", "--spiking-mlp",
        "--no-spiking-mlp-exact-gelu",
        "--no-gaussian-time-noise", "--no-mismatch-enabled",
        "--clock-driven", "--clock-time-step", str(PLOTTED_STEP),
        "--clock-time-steps-per-window", "0",
        "--calibration-mode", "validate",
        "--calibration-path", str(calibration_path),
        "--calibration-samples", "5000", "--calibration-seed", "0",
        "--calibration-bins", "2048",
        "--calibration-lower-quantile", "0",
        "--calibration-upper-quantile", "1",
        "--calibration-margin-fraction", "0.05",
    ]
    try:
        wrapper.main()
    finally:
        noise.quantize_encoder_output = original_quantize
        sys.argv = original_argv
    if not captured:
        raise ValueError("the diagnostic did not observe any phi operation windows")
    return captured


def _validate_and_summarize(
    windows: list[dict[str, Any]], experiment: dict[str, Any],
    completed: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    by_run = {row["run_id"]: row for row in completed["shard_runs"]}
    summaries: dict[str, dict[str, float | int]] = {}
    for width in (PLOTTED_STEP,):
        width = float(width)
        run_id = f"dt_{width:g}_shard_00"
        reference = by_run[run_id]
        by_site: dict[str, list[int]] = defaultdict(list)
        all_counts = []
        for window in windows:
            count = _step_count(window["time_min"], window["time_max"], width)
            if count <= 0:
                raise ValueError("a phi operation window has no simulation interval")
            by_site[window["site"]].append(count)
            all_counts.append(count)
        if set(by_site) != set(reference["clock_sites"]):
            raise ValueError(f"phi operation sites differ from {run_id}")
        for site, counts in by_site.items():
            expected = reference["clock_sites"][site]
            if (
                len(counts) != expected["windows"]
                or min(counts) != expected["minimum_window_steps"]
                or max(counts) != expected["maximum_window_steps"]
            ):
                raise ValueError(
                    f"phi operation window steps differ at {run_id}: {site}: "
                    f"observed={(len(counts), min(counts), max(counts))}, "
                    f"expected={(expected['windows'], expected['minimum_window_steps'], expected['maximum_window_steps'])}"
                )
        if sum(count + 1 for count in all_counts) != reference["clock_updates"]["encoder"]["time_steps"]:
            raise ValueError(f"phi operation update total differs from {run_id}")
        ordered = sorted(all_counts)
        summaries[f"{width:g}"] = {
            "windows": len(ordered),
            "minimum": ordered[0],
            "maximum": ordered[-1],
            "mean": statistics.mean(ordered),
            "median": statistics.median(ordered),
            "p05": ordered[math.floor(0.05 * (len(ordered) - 1))],
            "p95": ordered[math.ceil(0.95 * (len(ordered) - 1))],
        }
    return summaries


def _plot(windows: list[dict[str, Any]], summaries: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    counts = np.array([
        _step_count(row["time_min"], row["time_max"], PLOTTED_STEP)
        for row in windows
    ])
    edges = np.geomspace(counts.min(), counts.max() + 1, 31)
    axis.hist(counts, bins=edges, color="#2166AC", edgecolor="white", linewidth=0.35)
    axis.axvline(float(np.median(counts)), color="#222222", linestyle="--", linewidth=1.2, label="Median")
    axis.axvline(float(np.mean(counts)), color="#666666", linestyle=":", linewidth=1.5, label="Mean")
    axis.set_xscale("log")
    axis.set_title(rf"Fixed global step width $\Delta t={PLOTTED_STEP:g}$")
    axis.set_xlabel(r"Time steps per $\phi$ operation")
    axis.set_ylabel(r"Number of $\phi$ operations")
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.98, 0.96,
        f"Median {summaries[f'{PLOTTED_STEP:g}']['median']:,.0f}\n"
        f"Mean {summaries[f'{PLOTTED_STEP:g}']['mean']:,.0f}",
        transform=axis.transAxes, ha="right", va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=250)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Capture one clean forward before plotting")
    args = parser.parse_args()
    experiment_path = CAMPAIGN / "experiment.json"
    completed_path = CAMPAIGN / "summary.json"
    experiment = json.loads(experiment_path.read_text())
    completed = json.loads(completed_path.read_text())
    source = Path(experiment["source_root"])
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))
    from scripts.analysis.evaluate_calibrated_vit import validate_source

    validate_source(source, experiment["source_commit"])
    if args.collect:
        if RECORDS.exists():
            raise FileExistsError(RECORDS)
        windows = _capture_windows(experiment)
    else:
        stored = json.loads(RECORDS.read_text())
        if (
            stored["experiment_sha256"] != _sha256(experiment_path)
            or stored["summary_sha256"] != _sha256(completed_path)
        ):
            raise ValueError("stored window data belong to a different experiment")
        windows = stored["windows"]
    summaries = _validate_and_summarize(windows, experiment, completed)
    if args.collect:
        RECORDS.parent.mkdir(parents=True, exist_ok=True)
        RECORDS.write_text(json.dumps({
            "experiment_sha256": _sha256(experiment_path),
            "summary_sha256": _sha256(completed_path),
            "source_commit": experiment["source_commit"],
            "calibration_sha256": experiment["calibration_sha256"],
            "evaluation_population": 1,
            "windows": windows,
            "summaries": summaries,
        }, indent=2, sort_keys=True) + "\n")
    _plot(windows, summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"Figure: {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
