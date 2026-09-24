#!/usr/bin/env python3
"""Plot CCT timing-noise sweeps with provisional hardware candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FAMILIES = ("phi_np", "phi_nl", "both")
COLORS = {"phi_np": "#1f77b4", "phi_nl": "#ff7f0e", "both": "#2ca02c"}
LABELS = {
    "phi_np": r"$\phi_{\mathrm{NP}}$",
    "phi_nl": r"$\phi_{\mathrm{NL}}$",
    "both": r"$\phi_{\mathrm{NP}}+\phi_{\mathrm{NL}}$ (equal)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--clean-accuracy", type=float, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_candidates(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}
    for row in rows:
        family = row["condition_family"]
        if family not in FAMILIES:
            raise ValueError(f"unknown candidate condition_family: {family}")
        if family in candidates:
            raise ValueError(f"duplicate candidate condition_family: {family}")
        candidates[family] = row
    missing = set(FAMILIES).difference(candidates)
    if missing:
        raise ValueError(f"missing candidate families: {sorted(missing)}")
    return candidates


def sweep_points(
    reference: list[dict[str, str]],
    candidates: dict[str, dict[str, str]],
    family: str,
) -> list[tuple[float, float, float, float, bool]]:
    points = [
        (
            float(row["r_t"]),
            float(row["mean_accuracy"]) * 100,
            float(row["ci95_low"]) * 100,
            float(row["ci95_high"]) * 100,
            False,
        )
        for row in reference
        if row["condition_family"] == family
    ]
    if not points:
        raise ValueError(f"reference sweep has no rows for {family}")

    # The isolated candidates have a scalar r_t and therefore belong on their
    # respective sweep curves. The unequal joint candidate is handled as a
    # horizontal accuracy reference because it has two distinct fractions.
    if family in ("phi_np", "phi_nl"):
        row = candidates[family]
        points.append(
            (
                float(row["display_r_t"]),
                float(row["mean_accuracy"]) * 100,
                float(row["ci95_low"]) * 100,
                float(row["ci95_high"]) * 100,
                True,
            )
        )
    return sorted(points, key=lambda point: point[0])


def plot_sweep(
    ax: Any,
    reference: list[dict[str, str]],
    candidates: dict[str, dict[str, str]],
    family: str,
) -> None:
    points = sweep_points(reference, candidates, family)
    x = np.asarray([point[0] for point in points])
    mean = np.asarray([point[1] for point in points])
    low = np.asarray([point[2] for point in points])
    high = np.asarray([point[3] for point in points])
    is_candidate = np.asarray([point[4] for point in points])

    ax.plot(x, mean, color=COLORS[family], linewidth=2.8, zorder=3, label=LABELS[family])
    ax.fill_between(x, low, high, color=COLORS[family], alpha=0.12, zorder=1)
    ax.scatter(x[~is_candidate], mean[~is_candidate], color=COLORS[family], s=48, zorder=4)

    if is_candidate.any():
        candidate_x = x[is_candidate][0]
        candidate_mean = mean[is_candidate][0]
        candidate_low = low[is_candidate][0]
        candidate_high = high[is_candidate][0]
        ax.errorbar(
            candidate_x,
            candidate_mean,
            yerr=[
                [candidate_mean - candidate_low],
                [candidate_high - candidate_mean],
            ],
            fmt="none",
            ecolor=COLORS[family],
            elinewidth=2.2,
            capsize=6,
            capthick=2.0,
            zorder=6,
        )
        ax.scatter(
            [candidate_x],
            [candidate_mean],
            marker="*",
            s=720,
            color=COLORS[family],
            edgecolor="black",
            linewidth=1.8,
            zorder=7,
        )


# @lat: [[evaluation#Evaluation and Verification#Compact Transformer Diagnostic]]
def main() -> None:
    args = parse_args()
    reference = read_rows(args.reference_csv)
    candidates = require_candidates(read_rows(args.candidate_csv))

    figure, ax = plt.subplots(figsize=(12, 7.6), constrained_layout=True)
    for family in FAMILIES:
        plot_sweep(ax, reference, candidates, family)

    ax.axhline(args.clean_accuracy * 100, color="black", linestyle="--", linewidth=1.8, zorder=2)

    joint = candidates["both"]
    joint_accuracy = float(joint["mean_accuracy"]) * 100
    ax.axhline(
        joint_accuracy,
        color=COLORS["both"],
        linestyle=(0, (7, 3)),
        linewidth=2.6,
        zorder=2,
    )
    ax.text(
        1.05e-4,
        joint_accuracy + 1.4,
        r"observed $r_t$ pair",
        color=COLORS["both"],
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=13,
    )

    ax.set_xscale("log")
    ax.set_xlim(7.5e-5, 2.6e-2)
    ax.set_ylim(0, 101)
    ax.set_xlabel(r"Timing noise fraction $r_t$", fontsize=17)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=17)
    ax.grid(True, which="both", alpha=0.23, linewidth=1)
    ax.tick_params(axis="both", labelsize=15)

    handles, legend_labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8),
            Line2D(
                [0],
                [0],
                color=COLORS["phi_np"],
                marker="*",
                markersize=18,
                markeredgecolor="black",
                linestyle="none",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["phi_nl"],
                marker="*",
                markersize=18,
                markeredgecolor="black",
                linestyle="none",
            ),
        ]
    )
    legend_labels.extend(
        [
            "clean",
            r"observed $\phi_{\mathrm{NP}}$",
            r"observed $\phi_{\mathrm{NL}}$",
        ]
    )
    ax.legend(handles, legend_labels, loc="lower left", ncol=2, fontsize=12.5, frameon=False)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=220)
    figure.savefig(args.output_prefix.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
