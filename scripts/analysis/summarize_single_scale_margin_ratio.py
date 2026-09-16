#!/usr/bin/env python3
"""Validate and plot one fixed timing-noise scale across margin ratios."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.analysis.summarize_sigma_margin_sweep import (
    CANONICAL_MARGINS,
    aggregate_runs,
    aggregate_sites,
    parse_run_log,
    read_manifest,
    write_csv,
)
from scripts.runtime import identity


def plot(summary: list[dict[str, object]], output_prefix: Path) -> None:
    """Plot task accuracy and pooled missed-event rate without notation aliases."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stochastic = [row for row in summary if row["stage"] == "sigma_margin"]
    stochastic.sort(key=lambda row: float(row["deadline_margin_std"]))
    ratios = np.array([float(row["deadline_margin_std"]) for row in stochastic])
    means = 100.0 * np.array([float(row["accuracy_mean"]) for row in stochastic])
    intervals = 100.0 * np.array(
        [float(row["accuracy_ci95_half_width"]) for row in stochastic]
    )
    missed = 100.0 * np.array([float(row["miss_rate"]) for row in stochastic])
    clean = 100.0 * next(
        float(row["accuracy_mean"])
        for row in summary
        if row["stage"] == "baseline" and row["backend"] == "spiking"
    )

    figure, (accuracy_axis, missed_axis) = plt.subplots(
        1, 2, figsize=(10.2, 3.8), constrained_layout=True
    )
    accuracy_axis.errorbar(
        ratios,
        means,
        yerr=intervals,
        color="#1f77b4",
        marker="o",
        capsize=3,
        linewidth=1.4,
        label="Mean ± 95% confidence interval (3 seeds)",
    )
    accuracy_axis.axhline(
        clean, color="#222222", linestyle="--", linewidth=1.2, label="Clean baseline"
    )
    accuracy_axis.set_xlabel("Margin / noise standard deviation")
    accuracy_axis.set_ylabel("ImageNet-1k top-1 accuracy (%)")
    accuracy_axis.set_ylim(0.0, 90.0)
    accuracy_axis.set_xticks(ratios)
    accuracy_axis.tick_params(axis="x", labelrotation=45)
    accuracy_axis.grid(True, color="#eeeeee")
    accuracy_axis.spines[["top", "right"]].set_visible(False)
    accuracy_axis.legend(frameon=False, fontsize=8, loc="center right")

    inset = accuracy_axis.inset_axes([0.48, 0.08, 0.48, 0.32])
    inset.errorbar(
        ratios, means, yerr=intervals, color="#1f77b4", marker="o",
        capsize=2, linewidth=1.0, markersize=3,
    )
    inset.set_ylim(0.0, 0.6)
    inset.set_xticks((0.0, 4.0, 8.0, 12.0))
    inset.set_ylabel("Accuracy (%)", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(True, color="#eeeeee")

    positive = missed[missed > 0.0]
    floor = min(positive.min() / 10.0, 1.0e-8) if positive.size else 1.0e-8
    plotted_missed = np.where(missed > 0.0, missed, floor)
    missed_axis.plot(ratios, plotted_missed, color="#d95f02", marker="o", linewidth=1.4)
    zero = missed == 0.0
    if bool(zero.any()):
        missed_axis.scatter(
            ratios[zero], plotted_missed[zero], marker="v", facecolors="none",
            edgecolors="#d95f02", label="No missed events observed",
        )
    missed_axis.set_yscale("log")
    missed_axis.set_xlabel("Margin / noise standard deviation")
    missed_axis.set_ylabel("Missed event rate (%)")
    missed_axis.set_xticks(ratios)
    missed_axis.tick_params(axis="x", labelrotation=45)
    missed_axis.grid(True, which="both", color="#eeeeee")
    missed_axis.spines[["top", "right"]].set_visible(False)
    if bool(zero.any()):
        missed_axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Timing noise scale fixed at 1 (absolute standard deviation 80)"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--site-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    args = parser.parse_args()

    specs = read_manifest(args.manifest, require_canonical=False)
    stochastic = [spec for spec in specs if spec.stage == "sigma_margin"]
    if {spec.time_noise_std_frac for spec in stochastic} != {1.0}:
        raise ValueError("single-scale manifest must use time_noise_std_frac=1")
    if {spec.theta for spec in specs} != {40.0}:
        raise ValueError("single-scale manifest must use theta=40")
    if {spec.deadline_margin_std for spec in stochastic} != set(CANONICAL_MARGINS):
        raise ValueError("single-scale manifest must contain all 13 margin ratios")

    runs = [parse_run_log(spec, args.log_dir) for spec in specs]
    summary = aggregate_runs(runs)
    site_rows = aggregate_sites(runs)
    raw_rows = []
    for run in runs:
        row = asdict(run)
        row.pop("sites")
        raw_rows.append(row)
    write_csv(args.raw_csv, raw_rows)
    write_csv(args.summary_csv, summary)
    write_csv(args.site_csv, site_rows)
    plot(summary, args.figure_prefix)

    first = specs[0]
    payload = {
        "format_version": 1,
        "status": "complete",
        "runs": len(runs),
        "stochastic_runs": len(stochastic),
        "theta": first.theta,
        "time_noise_std_frac": 1.0,
        "time_noise_std_abs": 80.0,
        "source_commit": first.source_commit,
        "checkpoint_sha256": first.checkpoint_sha256,
        "dataset_fingerprint": first.dataset_fingerprint,
        "manifest_sha256": identity.sha256_file(args.manifest),
        "raw_csv_sha256": identity.sha256_file(args.raw_csv),
        "summary_csv_sha256": identity.sha256_file(args.summary_csv),
        "site_csv_sha256": identity.sha256_file(args.site_csv),
        "paper_promotion_allowed": False,
    }
    args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.provenance_json.with_suffix(args.provenance_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.provenance_json)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
