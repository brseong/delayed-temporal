#!/usr/bin/env python3
"""Render a non-promotable snapshot from complete sigma-margin cells."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from scripts.analysis.summarize_sigma_margin_sweep import (
    CANONICAL_FRACTIONS,
    CANONICAL_MARGINS,
    ManifestRun,
    ParsedRun,
    aggregate_runs,
    aggregate_sites,
    parse_run_log,
    read_manifest,
    sha256_file,
    write_csv,
)


def collect_complete_cells(
    specs: Sequence[ManifestRun], log_dir: Path
) -> tuple[list[ParsedRun], dict[str, object]]:
    """Accept deterministic baselines and only complete three-replica cells."""

    valid: dict[str, ParsedRun] = {}
    rejected: dict[str, str] = {}
    for spec in specs:
        try:
            valid[spec.run_id] = parse_run_log(spec, log_dir)
        except (FileNotFoundError, UnicodeDecodeError, ValueError) as error:
            rejected[spec.run_id] = type(error).__name__

    baseline_specs = [spec for spec in specs if spec.stage == "baseline"]
    missing_baselines = [spec.run_id for spec in baseline_specs if spec.run_id not in valid]
    if missing_baselines:
        raise ValueError(
            "preview requires both deterministic baselines: "
            + ", ".join(missing_baselines)
        )

    cell_specs: dict[tuple[float, float], list[ManifestRun]] = {}
    for spec in specs:
        if spec.stage == "sigma_margin":
            cell_specs.setdefault(
                (spec.time_noise_std_frac, spec.deadline_margin_std), []
            ).append(spec)

    accepted = [valid[spec.run_id] for spec in baseline_specs]
    complete_cells: list[tuple[float, float]] = []
    for cell, replicas in sorted(cell_specs.items()):
        if (
            {spec.seed for spec in replicas} == {0, 1, 2}
            and all(spec.run_id in valid for spec in replicas)
        ):
            complete_cells.append(cell)
            accepted.extend(valid[spec.run_id] for spec in replicas)

    if not complete_cells:
        raise ValueError("preview requires at least one complete stochastic cell")
    metadata: dict[str, object] = {
        "format_version": 1,
        "status": "partial",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runs": len(specs),
        "valid_runs_at_snapshot": len(valid),
        "accepted_runs": len(accepted),
        "expected_stochastic_cells": len(cell_specs),
        "complete_stochastic_cells": len(complete_cells),
        "incomplete_stochastic_cells": len(cell_specs) - len(complete_cells),
        "rejected_or_missing_runs": len(rejected),
        "selection_rule": "deterministic baselines plus cells with seeds 0, 1, and 2",
        "paper_promotion_allowed": False,
    }
    return accepted, metadata


def plot_preview(
    summary: Sequence[dict[str, object]], figure_prefix: Path
) -> None:
    """Render the canonical grid while leaving incomplete cells blank."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    cells = [row for row in summary if row["stage"] == "sigma_margin"]
    index = {
        (float(row["time_noise_std_frac"]), float(row["deadline_margin_std"])): row
        for row in cells
    }
    matrices = []
    for field_name, scale in (
        ("accuracy_mean", 100.0),
        ("accuracy_ci95_half_width", 100.0),
        ("miss_rate", 100.0),
    ):
        matrices.append(np.array([
            [
                scale * float(index[(fraction, margin)][field_name])
                if (fraction, margin) in index else np.nan
                for fraction in CANONICAL_FRACTIONS
            ]
            for margin in CANONICAL_MARGINS
        ]))

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.65), constrained_layout=True)
    titles = (
        "Mean top-1 accuracy (%)",
        r"95% Student-$t$ confidence interval ($\pm$pp)",
        "Pooled event deadline-miss rate (%)",
    )
    cmaps = ("viridis", "magma", "cividis")
    for axis, matrix, title, cmap_name in zip(axes, matrices, titles, cmaps, strict=True):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#d9d9d9")
        image = axis.imshow(
            np.ma.masked_invalid(matrix), origin="lower", aspect="auto", cmap=cmap
        )
        axis.set_title(title)
        axis.set_xlabel(r"Timing scale $r_t$")
        axis.set_xticks(range(len(CANONICAL_FRACTIONS)))
        axis.set_xticklabels(
            [f"{value / 1e-10:g}" for value in CANONICAL_FRACTIONS],
            rotation=45,
            ha="right",
        )
        axis.set_yticks(range(len(CANONICAL_MARGINS)))
        axis.set_yticklabels([f"{value:g}" for value in CANONICAL_MARGINS])
        axis.set_ylabel(r"Deadline margin ratio $k=m/\sigma_t$")
        figure.colorbar(image, ax=axis, shrink=0.82)
    axes[0].text(
        1.0,
        -0.24,
        r"Tick labels show $r_t/10^{-10}$",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=8,
    )
    clean = next(
        float(row["accuracy_mean"])
        for row in summary
        if row["stage"] == "baseline" and row["backend"] == "spiking"
    )
    figure.suptitle(
        f"Partial result: cells with three completed replicas "
        f"(clean spiking baseline {100.0 * clean:.2f}%)"
    )
    figure.legend(
        handles=[Patch(facecolor="#d9d9d9", edgecolor="none", label="Not complete")],
        loc="outside upper right",
        frameon=False,
        fontsize=8,
    )
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(figure_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_preview_outputs(
    runs: Sequence[ParsedRun],
    metadata: dict[str, object],
    *,
    manifest: Path,
    raw_csv: Path,
    summary_csv: Path,
    site_csv: Path,
    snapshot_json: Path,
    figure_prefix: Path,
) -> None:
    """Aggregate one immutable partial snapshot and write its evidence files."""

    summary = aggregate_runs(runs)
    site_rows = aggregate_sites(runs)
    raw_rows = []
    for run in runs:
        row = asdict(run)
        row.pop("sites")
        raw_rows.append(row)
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary)
    write_csv(site_csv, site_rows)
    payload = {
        **metadata,
        "manifest_sha256": sha256_file(manifest),
        "raw_csv_sha256": sha256_file(raw_csv),
        "summary_csv_sha256": sha256_file(summary_csv),
        "site_csv_sha256": sha256_file(site_csv),
    }
    snapshot_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot_json.with_suffix(snapshot_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(snapshot_json)
    plot_preview(summary, figure_prefix)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--site-csv", type=Path, required=True)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    specs = read_manifest(args.manifest)
    runs, metadata = collect_complete_cells(specs, args.log_dir)
    write_preview_outputs(
        runs,
        metadata,
        manifest=args.manifest,
        raw_csv=args.raw_csv,
        summary_csv=args.summary_csv,
        site_csv=args.site_csv,
        snapshot_json=args.snapshot_json,
        figure_prefix=args.figure_prefix,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
