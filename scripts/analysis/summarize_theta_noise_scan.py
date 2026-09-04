"""Combine validated per-theta Gaussian summaries into the appendix figure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_theta_summary(spec: str) -> list[dict[str, object]]:
    """Load one ``theta=summary.csv`` input and validate its Gaussian rows."""
    theta_text, separator, path_text = spec.partition("=")
    if not separator:
        raise ValueError(f"theta summary must use THETA=PATH syntax: {spec!r}")
    theta = float(theta_text)
    path = Path(path_text)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baselines = [row for row in rows if row["axis"] == "baseline"]
    gaussian = [row for row in rows if row["axis"] == "gaussian"]
    if len(baselines) != 1 or not gaussian:
        raise ValueError(f"theta summary requires one baseline and Gaussian rows: {path}")
    if any(row["axis"] == "mismatch" for row in rows):
        raise ValueError(f"theta appendix input must not contain mismatch rows: {path}")
    if any(not math.isclose(float(row["theta"]), theta) for row in rows):
        raise ValueError(f"declared theta does not match summary metadata: {path}")

    identity_fields = (
        "model_id",
        "dataset_id",
        "dataset_split",
        "evaluation_samples",
        "precision",
    )
    baseline = baselines[0]
    normalized: list[dict[str, object]] = []
    for row in gaussian:
        if any(row[field] != baseline[field] for field in identity_fields):
            raise ValueError(f"mixed evaluation identity inside theta summary: {path}")
        normalized.append(
            {
                "theta": theta,
                "time_noise_std_frac": float(row["magnitude"]),
                "replicas": int(row["replicas"]),
                "accuracy_mean": float(row["accuracy_mean"]),
                "accuracy_ci95_low": float(row["accuracy_ci95_low"]),
                "accuracy_ci95_high": float(row["accuracy_ci95_high"]),
                "baseline_accuracy": float(baseline["accuracy_mean"]),
                **{field: row[field] for field in identity_fields},
            }
        )
    return normalized


def summarize_theta_noise(
    inputs: Sequence[str], *, output_csv: Path, figure_prefix: Path
) -> list[dict[str, object]]:
    """Validate, combine, serialize, and plot the three theta transition scans."""
    rows = [row for spec in inputs for row in load_theta_summary(spec)]
    if {float(row["theta"]) for row in rows} != {40.0, 400.0, 2000.0}:
        raise ValueError("theta appendix requires exactly theta={40, 400, 2000}")
    identity_fields = (
        "model_id",
        "dataset_id",
        "dataset_split",
        "evaluation_samples",
        "precision",
    )
    if len({tuple(row[field] for field in identity_fields) for row in rows}) != 1:
        raise ValueError("theta summaries do not share one evaluation identity")

    rows.sort(key=lambda row: (float(row["theta"]), float(row["time_noise_std_frac"])))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_csv)

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.8}
    )
    fig, axis = plt.subplots(figsize=(5.4, 4.2))
    colors = {40.0: "#009E73", 400.0: "#E69F00", 2000.0: "#0072B2"}
    for theta in (40.0, 400.0, 2000.0):
        group = [row for row in rows if float(row["theta"]) == theta]
        x = [float(row["time_noise_std_frac"]) for row in group]
        y = [float(row["accuracy_mean"]) for row in group]
        low = [float(row["accuracy_ci95_low"]) for row in group]
        high = [float(row["accuracy_ci95_high"]) for row in group]
        axis.errorbar(
            x,
            y,
            yerr=(
                [mean - bound for mean, bound in zip(y, low, strict=True)],
                [bound - mean for mean, bound in zip(y, high, strict=True)],
            ),
            color=colors[theta],
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            capsize=2.5,
            label=rf"$\theta={int(theta)}$",
        )
    axis.set_xscale("log")
    axis.set_xlabel(r"time-noise fraction $r_t$")
    axis.set_ylabel("ImageNet top-1 accuracy (5k subset)")
    axis.set_ylim(-0.02, 0.9)
    axis.grid(True, which="both", color="#e8e8e8", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(figure_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return rows


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit theta-summary inputs and artifact destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, metavar="THETA=SUMMARY_CSV")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run theta-summary validation and publication rendering."""
    args = parse_arguments(argv)
    rows = summarize_theta_noise(
        args.input,
        output_csv=args.output_csv,
        figure_prefix=args.figure_prefix,
    )
    print(
        f"Validated {len(rows)} theta-noise points; wrote {args.output_csv} and "
        f"{args.figure_prefix}.{{png,pdf}}"
    )


if __name__ == "__main__":
    main()
