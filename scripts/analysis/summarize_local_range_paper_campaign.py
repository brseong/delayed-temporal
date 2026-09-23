#!/usr/bin/env python3
"""Validate and summarize the operator-local paper re-evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime import files as runtime_files
from scripts.runtime import identity
from scripts.experiments.run_full_calibrated_vit_comparison import TAG as VIT_TAG
from scripts.experiments.run_vit_local_range_noise_condition import TAG as NOISE_TAG
from scripts.experiments.run_full_calibrated_text_comparison import (
    GPT2_COMPOSED_GELU_TAG,
    ROBERTA_LARGE_TAG,
    TAG as TEXT_TAG,
)


T_CRITICAL_DF2 = 4.302652729911275
VIT_MODELS = (
    "cifar10_vit_small", "imagenet_vit_small", "imagenet_vit_base", "imagenet_vit_large",
)
TEXT_MODELS = (
    ("roberta", TEXT_TAG), ("roberta_large", ROBERTA_LARGE_TAG),
    ("gpt2", GPT2_COMPOSED_GELU_TAG),
)


def read_complete(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("state") != "complete":
        raise ValueError(f"result is incomplete: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table_rows(artifacts: Path) -> list[dict[str, Any]]:
    rows = []
    for model in VIT_MODELS:
        root = artifacts / "logs/conversion_comparison" / VIT_TAG / "vit" / model
        result = read_complete(root / "result.json")
        manifest = json.loads((root / "manifest.json").read_text())
        ann = result["phases"]["ann"]["metrics"]
        snn = result["phases"]["snn"]["metrics"]
        if ann["total"] != manifest["evaluation_samples"] or snn["total"] != manifest["evaluation_samples"]:
            raise ValueError(f"ViT sample count differs: {model}")
        rows.append({
            "model": model, "family": "vit", "samples": ann["total"],
            "ann_metric": ann["accuracy"], "snn_metric": snn["accuracy"],
            "metric": "accuracy", "calibration_sha256": result["calibration_sha256"],
            "source_commit": manifest["source_commit"],
        })
    for model, tag in TEXT_MODELS:
        root = artifacts / "logs/conversion_comparison" / tag / "text" / model
        result = read_complete(root / "result.json")
        manifest = json.loads((root / "manifest.json").read_text())
        ann = result["phases"]["ann"]["metrics"]
        snn = result["phases"]["snn"]["metrics"]
        if model == "gpt2":
            ann_metric = ann["token_weighted_perplexity"]
            snn_metric = snn["token_weighted_perplexity"]
            samples = ann["total"]
            metric = "token_weighted_perplexity"
        else:
            ann_metric = ann["accuracy"]
            snn_metric = snn["accuracy"]
            samples = ann["total"]
            metric = "accuracy"
        if samples != manifest["evaluation_samples"] or snn["total"] != samples:
            raise ValueError(f"text sample count differs: {model}")
        rows.append({
            "model": model, "family": "text", "samples": samples,
            "ann_metric": ann_metric, "snn_metric": snn_metric, "metric": metric,
            "calibration_sha256": result["calibration_sha256"],
            "source_commit": manifest["source_commit"],
        })
    if len({row["source_commit"] for row in rows}) != 1:
        raise ValueError("table results mix source commits")
    return rows


def noise_rows(artifacts: Path) -> list[dict[str, Any]]:
    runs = artifacts / "logs/noise_scan" / NOISE_TAG / "runs"
    rows = []
    identities = set()
    for result_path in sorted(runs.glob("*/result.json")):
        result = read_complete(result_path)
        root = result_path.parent
        manifest = json.loads((root / "manifest.json").read_text())
        if identity.sha256_file(root / result["log_file"]) != result["log_sha256"]:
            raise ValueError(f"noise log hash differs: {root.name}")
        metric = result["metrics"]
        counts = metric["physical_counts"]
        if metric["total"] != 5_000 or manifest["run_id"] != root.name:
            raise ValueError(f"noise run population or identity differs: {root.name}")
        identities.add((manifest["source_commit"], manifest["checkpoint_sha256"],
                        manifest["calibration_sha256"], manifest["evaluation_dataset_path"]))
        rows.append({
            "run_id": root.name, "time_noise_std_fraction": manifest["time_noise_std_fraction"],
            "deadline_margin_sigma_ratio": manifest["deadline_margin_sigma_ratio"],
            "seed": manifest["seed"], "correct": metric["correct"], "total": metric["total"],
            "accuracy": metric["accuracy"], "events": counts["events"], "misses": counts["misses"],
            "deadline_events": counts["deadline_events"], "outputs": counts["outputs"],
            "underflows": counts["underflows"], "overflows": counts["overflows"],
        })
    if len(rows) != 63 or len(identities) != 1:
        raise ValueError("noise evidence is incomplete or identity-mixed")
    cells: dict[tuple[float, float], set[int]] = {}
    for row in rows:
        key = (row["time_noise_std_fraction"], row["deadline_margin_sigma_ratio"])
        cells.setdefault(key, set()).add(row["seed"])
    if len(cells) != 21 or any(seeds != {0, 1, 2} for seeds in cells.values()):
        raise ValueError("noise cell or seed population differs")
    return rows


def noise_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["time_noise_std_fraction"], row["deadline_margin_sigma_ratio"]), []).append(row)
    summary = []
    for (fraction, ratio), replicas in sorted(groups.items()):
        accuracies = [row["accuracy"] for row in replicas]
        mean = statistics.fmean(accuracies)
        halfwidth = T_CRITICAL_DF2 * statistics.stdev(accuracies) / math.sqrt(3)
        events = sum(row["events"] for row in replicas)
        outputs = sum(row["outputs"] for row in replicas)
        summary.append({
            "time_noise_std_fraction": fraction, "deadline_margin_sigma_ratio": ratio,
            "replicas": 3, "accuracy_mean": mean, "accuracy_ci95_low": mean - halfwidth,
            "accuracy_ci95_high": mean + halfwidth, "events": events,
            "misses": sum(row["misses"] for row in replicas),
            "miss_rate": sum(row["misses"] for row in replicas) / events,
            "deadline_events": sum(row["deadline_events"] for row in replicas),
            "deadline_event_rate": sum(row["deadline_events"] for row in replicas) / events,
            "outputs": outputs, "underflows": sum(row["underflows"] for row in replicas),
            "overflows": sum(row["overflows"] for row in replicas),
            "saturation_rate": (sum(row["underflows"] + row["overflows"] for row in replicas) / outputs),
        })
    return summary


def render_figure(path: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fraction_rows = sorted((row for row in summary if row["deadline_margin_sigma_ratio"] == 4.0),
                           key=lambda row: row["time_noise_std_fraction"])
    ratio_rows = sorted((row for row in summary if row["time_noise_std_fraction"] == 1e-5),
                        key=lambda row: row["deadline_margin_sigma_ratio"])
    if len(fraction_rows) != 9 or len(ratio_rows) != 13:
        raise ValueError("plot grids are incomplete")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, rows, key, label in (
        (axes[0], fraction_rows, "time_noise_std_fraction", r"Timing-noise fraction $r_t$"),
        (axes[1], ratio_rows, "deadline_margin_sigma_ratio",
         r"Deadline margin / timing-noise std. $k$"),
    ):
        x = [row[key] for row in rows]
        y = [100 * row["accuracy_mean"] for row in rows]
        lower = [100 * (row["accuracy_mean"] - row["accuracy_ci95_low"]) for row in rows]
        upper = [100 * (row["accuracy_ci95_high"] - row["accuracy_mean"]) for row in rows]
        axis.errorbar(x, y, yerr=[lower, upper], marker="o", linewidth=1.5, capsize=2,
                      color="#2f5597", label="Top-1 accuracy")
        axis.set_xlabel(label)
        axis.set_ylabel("Top-1 accuracy (%)")
        axis.grid(True, alpha=0.25)
        if key == "time_noise_std_fraction":
            axis.set_xscale("log")
        else:
            secondary = axis.twinx()
            secondary.plot(x, [100 * row["miss_rate"] for row in rows], marker="s",
                           linewidth=1.2, color="#c65911", label="Deadline-miss rate")
            secondary.set_ylabel("Deadline-miss rate (%)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


# @lat: [[evaluation#Evaluation and Verification#Local-Range Paper Re-evaluation]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("/data/delayed-temporal/artifacts"))
    args = parser.parse_args()
    artifacts = args.artifacts_root.resolve(strict=True)
    output = artifacts / "results/paper_local_range_poseidon_v1"
    tables = table_rows(artifacts)
    raw = noise_rows(artifacts)
    summary = noise_summary(raw)
    write_csv(output / "table_results.csv", tables, list(tables[0]))
    write_csv(output / "noise_raw_runs.csv", raw, list(raw[0]))
    write_csv(output / "noise_summary.csv", summary, list(summary[0]))
    render_figure(artifacts / "figures/ViT-noise-eval-local-ranges", summary)
    runtime_files.atomic_json(output / "summary.json", {
        "state": "complete", "table_models": len(tables), "noise_runs": len(raw),
        "noise_cells": len(summary), "source_commit": tables[0]["source_commit"],
    })


if __name__ == "__main__":
    main()
