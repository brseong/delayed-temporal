#!/usr/bin/env python3
"""Build the ICLR timing-noise figure from two completed ViT sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_NOISE_SCALES = tuple(1e-5 * 10 ** (index / 8) for index in range(9))
EXPECTED_MARGIN_RATIOS = (0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12)
MATCHED_PROVENANCE_FIELDS = (
    "checkpoint_sha256",
    "dataset_fingerprint",
    "source_commit",
    "theta",
    "validation_scope",
)


@dataclass(frozen=True)
class SweepData:
    clean_accuracy: float
    dense_accuracy: float
    rows: tuple[dict[str, float], ...]


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-15)


def _read_provenance(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "complete":
        raise ValueError(f"Sweep provenance is not complete: {path}")
    return data


def _read_summary(path: Path, *, axis: str) -> SweepData:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def baseline(backend: str) -> float:
        matches = [
            row for row in rows
            if row["stage"] == "baseline" and row["backend"] == backend
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {backend} baseline in {path}")
        return float(matches[0]["accuracy_mean"])

    stochastic = [
        row for row in rows
        if row["stage"] == "sigma_margin" and row["backend"] == "spiking"
    ]
    expected_values = (
        EXPECTED_NOISE_SCALES if axis == "time_noise_std_frac"
        else EXPECTED_MARGIN_RATIOS
    )
    stochastic.sort(key=lambda row: float(row[axis]))
    values = tuple(float(row[axis]) for row in stochastic)
    if len(values) != len(expected_values) or any(
        not _close(value, expected)
        for value, expected in zip(values, expected_values)
    ):
        raise ValueError(f"Unexpected {axis} grid in {path}")

    parsed: list[dict[str, float]] = []
    for row in stochastic:
        if int(row["replicas"]) != 3:
            raise ValueError(f"Incomplete stochastic condition in {path}")
        parsed.append({
            "time_noise_std_frac": float(row["time_noise_std_frac"]),
            "deadline_margin_std": float(row["deadline_margin_std"]),
            "theta": float(row["theta"]),
            "accuracy_mean": float(row["accuracy_mean"]),
            "accuracy_ci95_half_width": float(row["accuracy_ci95_half_width"]),
            "miss_rate": float(row["miss_rate"]),
        })
    return SweepData(baseline("spiking"), baseline("hf"), tuple(parsed))


def load_inputs(
    noise_summary: Path,
    margin_summary: Path,
    noise_provenance: Path,
    margin_provenance: Path,
) -> tuple[SweepData, SweepData]:
    """Load and cross-check the two completed slices used by the figure."""
    noise_meta = _read_provenance(noise_provenance)
    margin_meta = _read_provenance(margin_provenance)
    for field in MATCHED_PROVENANCE_FIELDS:
        if noise_meta.get(field) != margin_meta.get(field):
            raise ValueError(f"Sweep provenance mismatch for {field}")
    if noise_meta.get("runs") != 29 or noise_meta.get("stochastic_runs") != 27:
        raise ValueError("Timing-noise sweep does not contain 9 three-seed cells")
    if margin_meta.get("runs") != 41 or margin_meta.get("stochastic_runs") != 39:
        raise ValueError("Margin sweep does not contain 13 three-seed cells")

    noise = _read_summary(noise_summary, axis="time_noise_std_frac")
    margin = _read_summary(margin_summary, axis="deadline_margin_std")
    if not all(_close(row["theta"], 40.0) for row in noise.rows + margin.rows):
        raise ValueError("The paper figure requires theta=40")
    if not all(_close(row["deadline_margin_std"], 4.0) for row in noise.rows):
        raise ValueError("The timing-noise slice must fix the margin ratio at 4")
    if not all(_close(row["time_noise_std_frac"], 1e-5) for row in margin.rows):
        raise ValueError("The margin slice must fix timing noise at 1e-5")
    if not _close(noise.clean_accuracy, margin.clean_accuracy):
        raise ValueError("Clean spiking baselines differ between sweeps")
    if not _close(noise.dense_accuracy, margin.dense_accuracy):
        raise ValueError("Dense baselines differ between sweeps")
    return noise, margin


def plot_figure(noise: SweepData, margin: SweepData, output_prefix: Path) -> None:
    """Render the publication figure as vector PDF and review PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    blue = "#2166AC"
    orange = "#B35806"
    green = "#1B7837"
    gray = "#666666"
    fig, (noise_ax, margin_ax) = plt.subplots(
        1, 2, figsize=(7.05, 2.65), constrained_layout=True
    )

    noise_x = [row["time_noise_std_frac"] for row in noise.rows]
    noise_y = [100 * row["accuracy_mean"] for row in noise.rows]
    noise_error = [100 * row["accuracy_ci95_half_width"] for row in noise.rows]
    noise_ax.errorbar(
        noise_x, noise_y, yerr=noise_error, color=blue, marker="o",
        markersize=3.8, linewidth=1.4, capsize=2.2, label="Noisy accuracy",
    )
    noise_ax.axhline(
        100 * noise.clean_accuracy, color=green, linestyle="--", linewidth=1.1,
        label="Clean spiking",
    )
    noise_ax.axhline(
        100 * noise.dense_accuracy, color=gray, linestyle=":", linewidth=1.1,
        label="Dense reference",
    )
    noise_ax.set_xscale("log")
    noise_ax.set_xlim(8.8e-6, 1.13e-4)
    noise_ax.set_ylim(48, 88)
    noise_ax.set_xlabel(r"Timing noise $r_t$")
    noise_ax.set_ylabel("Top-1 accuracy (%)")
    noise_ax.set_title(r"(a) Timing-noise scale ($k=4$)", loc="left")
    noise_ax.grid(alpha=0.22, linewidth=0.6)
    noise_ax.legend(frameon=False, loc="lower left")

    margin_x = [row["deadline_margin_std"] for row in margin.rows]
    margin_y = [100 * row["accuracy_mean"] for row in margin.rows]
    margin_error = [100 * row["accuracy_ci95_half_width"] for row in margin.rows]
    accuracy_line = margin_ax.errorbar(
        margin_x, margin_y, yerr=margin_error, color=blue, marker="o",
        markersize=3.8, linewidth=1.4, capsize=2.2, label="Noisy accuracy",
    )
    margin_ax.axhline(
        100 * margin.clean_accuracy, color=green, linestyle="--", linewidth=1.1
    )
    margin_ax.axhline(
        100 * margin.dense_accuracy, color=gray, linestyle=":", linewidth=1.1
    )
    margin_ax.set_xlim(-0.35, 12.35)
    margin_ax.set_xticks((0, 2, 4, 6, 8, 10, 12))
    margin_ax.set_ylim(-2, 89)
    margin_ax.set_xlabel(r"Deadline margin $k=m/\sigma_t$")
    margin_ax.set_ylabel("Top-1 accuracy (%)", color=blue)
    margin_ax.tick_params(axis="y", colors=blue)
    margin_ax.set_title(r"(b) Deadline margin ($r_t=10^{-5}$)", loc="left")
    margin_ax.grid(alpha=0.22, linewidth=0.6)

    miss_ax = margin_ax.twinx()
    miss_percent = [100 * row["miss_rate"] for row in margin.rows]
    positive_rates = [value for value in miss_percent if value > 0]
    display_floor = min(positive_rates) / 5
    displayed_rates = [max(value, display_floor) for value in miss_percent]
    miss_line, = miss_ax.plot(
        margin_x, displayed_rates, color=orange, marker="s", markersize=3.5,
        linewidth=1.3, label="Deadline-miss rate",
    )
    zero_x = [x for x, value in zip(margin_x, miss_percent) if value == 0]
    if zero_x:
        miss_ax.scatter(
            zero_x, [display_floor] * len(zero_x), marker="v", s=20,
            facecolors="white", edgecolors=orange, linewidths=0.9, zorder=4,
        )
    miss_ax.set_yscale("log")
    miss_ax.set_ylim(display_floor / 2, max(miss_percent) * 2)
    miss_ax.set_ylabel("Deadline-miss rate (%)", color=orange)
    miss_ax.tick_params(axis="y", colors=orange)
    miss_ax.spines["right"].set_color(orange)
    margin_ax.legend(
        [accuracy_line, miss_line], ["Noisy accuracy", "Deadline-miss rate"],
        frameon=False, loc="center right",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("pdf", 300), ("png", 220)):
        fig.savefig(output_prefix.with_suffix(f".{extension}"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-summary", type=Path, required=True)
    parser.add_argument("--margin-summary", type=Path, required=True)
    parser.add_argument("--noise-provenance", type=Path, required=True)
    parser.add_argument("--margin-provenance", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    noise, margin = load_inputs(
        args.noise_summary,
        args.margin_summary,
        args.noise_provenance,
        args.margin_provenance,
    )
    plot_figure(noise, margin, args.output_prefix)


if __name__ == "__main__":
    main()
