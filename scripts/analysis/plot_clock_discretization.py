#!/usr/bin/env python3
"""Regenerate the ICLR two-panel discrete-time simulation figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analysis.plot_clock_time_step_sweep import (  # noqa: E402
    expected_population,
    expected_shards,
    load_verified_results,
)


CLOCK_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/logs/clock_driven/"
    "vit_base_clock_driven_imagenet500_theta20_float64_fine_v1"
)
WINDOW_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/logs/clock_driven/"
    "vit_base_clock_driven_window_steps_384_768_imagenet500_theta20_float64_v1"
)
OUTPUT_PREFIXES = (
    REPOSITORY_ROOT / "artifacts/figures/ViT-clock-discretization",
    REPOSITORY_ROOT / "paper/iclr_2027/figures/ViT-clock-discretization",
)
PLOTTED_WINDOW_STEPS = (128, 256, 384, 512, 768, 1024, 2048)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_extended_window_results(root: Path) -> list[dict[str, Any]]:
    """Load the verified fixed-window-count results including 384 and 768."""

    verification_path = root / "verification.json"
    if not verification_path.is_file():
        raise FileNotFoundError(
            f"extended sweep verification is missing: {verification_path}"
        )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "complete":
        raise ValueError("extended fixed-window sweep is not complete")

    expected_files = (
        ("summary.json", "summary_json_sha256"),
        ("summary.csv", "summary_csv_sha256"),
        ("raw_shards.csv", "raw_shards_csv_sha256"),
    )
    for filename, digest_field in expected_files:
        path = root / filename
        if not path.is_file() or _sha256(path) != verification.get(digest_field):
            raise ValueError(f"extended sweep artifact identity differs: {path}")

    results = verification.get("results")
    expected_steps = (None, 64, *PLOTTED_WINDOW_STEPS)
    if not isinstance(results, list) or len(results) != len(expected_steps):
        raise ValueError("extended sweep condition count differs")
    if verification.get("condition_count") != len(expected_steps):
        raise ValueError("extended sweep verification condition count differs")
    if verification.get("shard_run_count") != len(expected_steps) * expected_shards:
        raise ValueError("extended sweep shard count differs")
    if verification.get("evaluation_population") != expected_population:
        raise ValueError("extended sweep evaluation population differs")

    for expected_step, result in zip(expected_steps, results):
        if result.get("time_steps_per_window") != expected_step:
            raise ValueError("extended sweep condition order differs")
        expected_condition = (
            "continuous" if expected_step is None else f"steps_{expected_step}"
        )
        if result.get("condition") != expected_condition:
            raise ValueError("extended sweep condition name differs")
        correct = int(result["correct"])
        samples = int(result["samples"])
        if samples != expected_population or not 0 <= correct <= samples:
            raise ValueError("extended sweep task counts differ")
        if not math.isclose(
            float(result["accuracy"]),
            correct / samples,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("extended sweep accuracy differs from task counts")
    return results


def plot_figure(
    time_step_results: list[dict[str, Any]],
    window_results: list[dict[str, Any]],
    output_prefixes: tuple[Path, ...],
) -> None:
    """Render the paper figure after checking the shared reference condition."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    continuous = float(time_step_results[0]["accuracy"]) * 100.0
    window_continuous = float(window_results[0]["accuracy"]) * 100.0
    if not math.isclose(continuous, window_continuous, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("continuous-time references differ between sweeps")

    time_step_rows = time_step_results[1:]
    by_window_steps = {
        int(row["time_steps_per_window"]): row for row in window_results[1:]
    }
    if not set(PLOTTED_WINDOW_STEPS).issubset(by_window_steps):
        raise ValueError("a requested steps-per-window condition is missing")
    plotted_window_rows = [
        by_window_steps[steps] for steps in PLOTTED_WINDOW_STEPS
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#2166AC"
    grey = "#4D4D4D"
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)

    time_steps = [float(row["time_step"]) for row in time_step_rows]
    time_step_accuracy = [float(row["accuracy"]) * 100.0 for row in time_step_rows]
    discrete_line = axes[0].plot(
        time_steps,
        time_step_accuracy,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        color=blue,
        label="Discrete-time simulation",
        zorder=3,
    )[0]
    continuous_line = axes[0].axhline(
        continuous,
        color=grey,
        linestyle="--",
        linewidth=1.2,
        label="Continuous-time conversion",
        zorder=2,
    )
    axes[0].set_title(
        "(a) Same interval between simulation time steps",
        pad=6,
    )
    axes[0].set_xlabel("Time-step width")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_xticks(time_steps)
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].set_xlim(min(time_steps) - 0.004, max(time_steps) + 0.004)

    positions = list(range(len(PLOTTED_WINDOW_STEPS)))
    window_accuracy = [
        float(row["accuracy"]) * 100.0 for row in plotted_window_rows
    ]
    axes[1].plot(
        positions,
        window_accuracy,
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        color=blue,
        zorder=3,
    )
    axes[1].axhline(
        continuous,
        color=grey,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )
    axes[1].set_title(
        "(b) Same number of time steps in each time window",
        pad=6,
    )
    axes[1].set_xlabel("Number of time steps")
    axes[1].set_xticks(
        positions,
        labels=[str(steps) for steps in PLOTTED_WINDOW_STEPS],
    )
    axes[1].set_xlim(-0.45, len(positions) - 0.55)

    for axis in axes:
        axis.set_ylim(0.0, 100.0)
        axis.set_yticks((0, 20, 40, 60, 80, 100))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    figure.legend(
        [discrete_line, continuous_line],
        ["Discrete-time simulation", "Continuous-time conversion"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.4,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.88,
        bottom=0.27,
        wspace=0.20,
    )

    for output_prefix in output_prefixes:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_prefix.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.02,
        )
        figure.savefig(
            output_prefix.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-step-root", type=Path, default=CLOCK_ROOT)
    parser.add_argument("--window-root", type=Path, default=WINDOW_ROOT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        action="append",
        help=(
            "Output path without an extension; may be repeated. The default writes "
            "both the artifact and ICLR paper copies."
        ),
    )
    args = parser.parse_args()
    output_prefixes = tuple(args.output_prefix or OUTPUT_PREFIXES)
    _, time_step_results = load_verified_results(args.time_step_root.resolve())
    window_results = load_extended_window_results(args.window_root.resolve())
    plot_figure(
        time_step_results,
        window_results,
        tuple(path.resolve() for path in output_prefixes),
    )
    for output_prefix in output_prefixes:
        print(f"Figure PDF: {output_prefix.with_suffix('.pdf')}")
        print(f"Figure PNG: {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
