#!/usr/bin/env python3
"""Validate, summarize, and plot one ViT-B encoder-block timing-noise phase."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_vit_bss2_depth_campaign import (
    PHASE_SAMPLES,
    expected_cells,
    run_id,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity


T_CRITICAL_DF2_975 = 4.302652729911275
DISPLAY_NAMES = {
    "screening-selected-coordinate": "screening-selected coordinate",
    "screening-median": "screening median",
}
COLORS = {
    "screening-selected-coordinate": "#1f77b4",
    "screening-median": "#d62728",
}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(root: Path, phase: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    expected_samples = PHASE_SAMPLES[phase]
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    shared_identity: tuple[Any, ...] | None = None
    for condition, first_block_count, seed in expected_cells():
        name = run_id(condition, first_block_count, seed)
        run_root = root / "runs" / name
        manifest_path = run_root / "manifest.json"
        result_path = run_root / "result.json"
        if not manifest_path.is_file() or not result_path.is_file():
            raise ValueError(f"missing depth result: {name}")
        manifest = json.loads(manifest_path.read_text())
        result = json.loads(result_path.read_text())
        metrics = result.get("metrics", {})
        counts = metrics.get("gaussian_counts", {})
        if result.get("state") != "complete":
            raise ValueError(f"incomplete depth result: {name}")
        if (
            manifest.get("condition") != condition
            or manifest.get("first_block_count") != first_block_count
            or manifest.get("seed") != seed
            or manifest.get("evaluation_samples") != expected_samples
            or metrics.get("total") != expected_samples
            or counts.get("active_blocks") != list(range(first_block_count))
        ):
            raise ValueError(f"depth result contract differs: {name}")
        run_identity = (
            manifest.get("source_commit"),
            manifest.get("checkpoint_sha256"),
            manifest.get("calibration_sha256"),
            manifest.get("evaluation_dataset", {}).get("fingerprint"),
            manifest.get("evaluation_dataset", {}).get("selected_fingerprint"),
            manifest.get("hardware_summary_sha256"),
        )
        if shared_identity is None:
            shared_identity = run_identity
        elif shared_identity != run_identity:
            raise ValueError("depth results mix source or data identities")
        rows.append(
            {
                "phase": phase,
                "evaluation_samples": expected_samples,
                "condition": condition,
                "first_block_count": first_block_count,
                "seed": seed,
                "linear_time_std_fraction": manifest["linear_time_std_fraction"],
                "log_time_std_fraction": manifest["log_time_std_fraction"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "prediction_sha256": metrics["prediction_sha256"],
                "events": counts["events"],
                "misses": counts["misses"],
                "miss_rate": counts["miss_rate"],
                "site_count": counts["site_count"],
                "run_id": name,
            }
        )
        source_hashes[f"runs/{name}/manifest.json"] = identity.sha256_file(manifest_path)
        source_hashes[f"runs/{name}/result.json"] = identity.sha256_file(result_path)
    if len(rows) != 73:
        raise ValueError("depth summary requires exactly 73 runs")
    return rows, source_hashes


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    if len(clean_rows) != 1:
        raise ValueError("depth summary requires exactly one clean run")
    clean = float(clean_rows[0]["accuracy"])
    summary: list[dict[str, Any]] = []
    for condition in DISPLAY_NAMES:
        for first_block_count in range(13):
            if first_block_count == 0:
                selected = clean_rows
            else:
                selected = [
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["first_block_count"] == first_block_count
                ]
            expected_replicas = 1 if first_block_count == 0 else 3
            if len(selected) != expected_replicas:
                raise ValueError("depth summary has a missing or duplicate seed")
            values = [float(row["accuracy"]) for row in selected]
            mean = statistics.fmean(values)
            if len(values) == 1:
                low = high = mean
            else:
                half_width = (
                    T_CRITICAL_DF2_975
                    * statistics.stdev(values)
                    / math.sqrt(len(values))
                )
                low, high = mean - half_width, mean + half_width
            summary.append(
                {
                    "phase": selected[0]["phase"],
                    "evaluation_samples": selected[0]["evaluation_samples"],
                    "condition": condition,
                    "first_block_count": first_block_count,
                    "replicas": len(values),
                    "linear_time_std_fraction": (
                        selected[0]["linear_time_std_fraction"]
                        if first_block_count else 0.0
                    ),
                    "log_time_std_fraction": (
                        selected[0]["log_time_std_fraction"]
                        if first_block_count else 0.0
                    ),
                    "accuracy_mean": mean,
                    "accuracy_ci_low": low,
                    "accuracy_ci_high": high,
                    "accuracy_change_pp_mean": 100.0 * (mean - clean),
                    "accuracy_change_pp_ci_low": 100.0 * (low - clean),
                    "accuracy_change_pp_ci_high": 100.0 * (high - clean),
                    "pooled_events": sum(int(row["events"]) for row in selected),
                    "pooled_misses": sum(int(row["misses"]) for row in selected),
                }
            )
    return summary


def render(summary: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    for condition, label in DISPLAY_NAMES.items():
        rows = [row for row in summary if row["condition"] == condition]
        x = [int(row["first_block_count"]) for row in rows]
        mean = [100.0 * float(row["accuracy_mean"]) for row in rows]
        low = [100.0 * float(row["accuracy_ci_low"]) for row in rows]
        high = [100.0 * float(row["accuracy_ci_high"]) for row in rows]
        change = [float(row["accuracy_change_pp_mean"]) for row in rows]
        change_low = [float(row["accuracy_change_pp_ci_low"]) for row in rows]
        change_high = [float(row["accuracy_change_pp_ci_high"]) for row in rows]
        color = COLORS[condition]
        axes[0].plot(x, mean, marker="o", color=color, label=label)
        axes[0].fill_between(x, low, high, color=color, alpha=0.16)
        axes[1].plot(x, change, marker="o", color=color, label=label)
        axes[1].fill_between(x, change_low, change_high, color=color, alpha=0.16)
    for axis in axes:
        axis.set_xticks(range(13))
        axis.set_xlabel("Number of noisy encoder blocks from the input")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_ylabel("Top-1 accuracy change from clean (percentage points)")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# @lat: [[evaluation#Evaluation and Verification#ViT-B Cumulative Encoder Block Timing Noise#Summary Artifacts]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=tuple(PHASE_SAMPLES), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_root.resolve(strict=True)
    raw, source_hashes = load_runs(root, args.phase)
    summary = summarize(raw)
    raw_path = root / "depth_noise_raw.csv"
    summary_path = root / "depth_noise_summary.csv"
    write_csv(raw_path, raw, tuple(raw[0]))
    write_csv(summary_path, summary, tuple(summary[0]))
    figure = root / "depth_noise_accuracy"
    render(summary, figure)
    manifest = {
        "schema_version": 1,
        "phase": args.phase,
        "source_hashes": source_hashes,
        "raw_csv_sha256": identity.sha256_file(raw_path),
        "summary_csv_sha256": identity.sha256_file(summary_path),
        "figure_pdf_sha256": identity.sha256_file(figure.with_suffix(".pdf")),
        "figure_png_sha256": identity.sha256_file(figure.with_suffix(".png")),
        "interpretation": (
            "cumulative sensitivity to measured marginal encoder timing-noise scales; "
            "not causal attribution to an individual block"
        ),
    }
    runtime_files.atomic_json(root / "summary_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
