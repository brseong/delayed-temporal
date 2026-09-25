#!/usr/bin/env python3
"""Regenerate the ICLR overlaid discrete-time simulation figure."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analysis.plot_clock_time_step_sweep import (  # noqa: E402
    load_verified_results,
)


CLOCK_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/logs/clock_driven/"
    "vit_base_clock_driven_imagenet500_full_conversion_float64_fine_v3"
)
WINDOW_ROOT = (
    REPOSITORY_ROOT
    / "artifacts/logs/clock_driven/"
    "vit_base_clock_driven_window_steps_384_768_imagenet500_"
    "full_conversion_float64_v3"
)
OUTPUT_PREFIXES = (
    REPOSITORY_ROOT / "artifacts/figures/ViT-clock-discretization",
    REPOSITORY_ROOT / "paper/iclr_2027/figures/ViT-clock-discretization",
)
TIME_STEP_TAG = "vit_base_clock_driven_imagenet500_full_conversion_float64_fine_v3"
WINDOW_STEP_TAG = (
    "vit_base_clock_driven_window_steps_384_768_imagenet500_"
    "full_conversion_float64_v3"
)
PLOTTED_TIME_STEPS = tuple(index / 100.0 for index in range(1, 11))
PLOTTED_WINDOW_STEPS = (128, 256, 384, 512, 768, 1024, 2048)
ICLR_TEXT_WIDTH_INCHES = 5.5
FIGURE_WIDTH_FRACTION = 0.40
FIGURE_SIZE = (ICLR_TEXT_WIDTH_INCHES * FIGURE_WIDTH_FRACTION, 2.55)


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
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#2166AC"
    grey = "#4D4D4D"
    orange = "#D95F02"
    figure, bottom_axis = plt.subplots(figsize=FIGURE_SIZE)
    top_axis = bottom_axis.twiny()

    time_steps = [float(row["time_step"]) for row in time_step_rows]
    time_step_accuracy = [float(row["accuracy"]) * 100.0 for row in time_step_rows]
    time_step_line = bottom_axis.plot(
        time_steps,
        time_step_accuracy,
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        color=blue,
        label="Same interval between simulation time steps",
        zorder=3,
    )[0]
    continuous_line = bottom_axis.axhline(
        continuous,
        color=grey,
        linestyle="--",
        linewidth=1.1,
        label="Continuous time",
        zorder=2,
    )
    bottom_axis.set_ylabel("Top-1 accuracy (%)", labelpad=1)
    bottom_axis.set_xticks(time_steps)
    bottom_axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    bottom_axis.set_xlim(min(time_steps) - 0.004, max(time_steps) + 0.004)
    bottom_axis.tick_params(axis="x", colors=blue, labelrotation=55, pad=1)
    for label in bottom_axis.get_xticklabels():
        label.set_horizontalalignment("right")

    positions = list(range(len(PLOTTED_WINDOW_STEPS)))
    window_accuracy = [
        float(row["accuracy"]) * 100.0 for row in plotted_window_rows
    ]
    window_line = top_axis.plot(
        positions,
        window_accuracy,
        marker="s",
        markersize=3.8,
        linewidth=1.5,
        color=orange,
        label="Same number of time steps in each time window",
        zorder=3,
    )[0]
    top_axis.set_xticks(
        positions,
        labels=[str(steps) for steps in PLOTTED_WINDOW_STEPS],
    )
    top_axis.set_xlim(-0.45, len(positions) - 0.55)
    top_axis.tick_params(axis="x", colors=orange, pad=1)
    top_axis.spines["top"].set_visible(True)
    top_axis.spines["top"].set_color(orange)

    bottom_axis.set_ylim(-3.0, 100.0)
    bottom_axis.set_yticks((0, 20, 40, 60, 80, 100))
    bottom_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    figure.legend(
        [time_step_line, window_line, continuous_line],
        [
            "Same interval between\nsimulation time steps\n(bottom axis)",
            "Same number of time steps\nin each time window\n(top axis)",
            "Continuous time",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        handlelength=2.0,
        borderpad=0.35,
        labelspacing=0.45,
        fontsize=5.4,
    )
    figure.subplots_adjust(
        left=0.22,
        right=0.97,
        top=0.92,
        bottom=0.40,
    )

    for output_prefix in output_prefixes:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_prefix.with_suffix(".pdf"))
        figure.savefig(
            output_prefix.with_suffix(".png"),
            dpi=300,
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
    _, time_step_results = load_verified_results(
        args.time_step_root.resolve(),
        time_step_tag=TIME_STEP_TAG,
        time_step_grid=PLOTTED_TIME_STEPS,
    )
    _, window_results = load_verified_results(
        args.window_root.resolve(),
        window_step_tag=WINDOW_STEP_TAG,
        window_step_grid=PLOTTED_WINDOW_STEPS,
    )
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
