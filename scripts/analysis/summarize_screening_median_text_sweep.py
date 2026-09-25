#!/usr/bin/env python3
"""Aggregate screening-median RoBERTa and GPT-2 replica artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.screening_median_text_sweep import TEXT_MODELS, TEXT_SEEDS


T_CRITICAL_DF2_975 = 4.302652729696142


def load_rows(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load complete formal cells belonging to one authenticated protocol."""

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for result_path in sorted(root.joinpath("formal").rglob("result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("state") != "complete":
            continue
        manifest = json.loads((result_path.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != protocol.get("protocol_id"):
            raise ValueError("text result protocol differs")
        key = (manifest["model"], manifest["alpha"], int(manifest["seed"]))
        if key in seen:
            raise ValueError(f"duplicate text result cell: {key}")
        seen.add(key)
        metrics = result["metrics"]
        counts = metrics["physical_counts"]
        row = {
            "model": manifest["model"],
            "alpha": float(manifest["alpha"]),
            "alpha_text": manifest["alpha"],
            "seed": int(manifest["seed"]),
            "evaluation_samples": int(manifest["evaluation_samples"]),
            "linear_time_noise_std_fraction": float(manifest["linear_time_noise_std_fraction"]),
            "log_time_noise_std_fraction": float(manifest["log_time_noise_std_fraction"]),
            "events": int(counts["events"]),
            "misses": int(counts["misses"]),
            "deadline_events": int(counts["deadline_events"]),
            "outputs": int(counts["outputs"]),
            "underflows": int(counts["underflows"]),
            "overflows": int(counts["overflows"]),
            "accuracy": metrics.get("accuracy"),
            "prediction_sha256": metrics.get("prediction_sha256"),
            "token_weighted_loss": metrics.get("token_weighted_loss"),
            "token_weighted_perplexity": metrics.get("token_weighted_perplexity"),
            "valid_token_count": metrics.get("valid_token_count"),
        }
        rows.append(row)
    if not rows:
        raise ValueError("no complete text formal cells were found")
    return protocol, rows


def interval(values: list[float]) -> tuple[float, float, float]:
    """Return the three-seed mean and Student-t interval."""

    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError("text summary requires three finite replicas")
    mean = statistics.fmean(values)
    half_width = T_CRITICAL_DF2_975 * statistics.stdev(values) / math.sqrt(3)
    return mean, mean - half_width, mean + half_width


def summarize(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce each model and multiplier to task-native uncertainty metrics."""

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["alpha"]), []).append(row)
    alphas = {row["alpha"] for row in rows}
    expected = {(model, alpha) for model in TEXT_MODELS for alpha in alphas}
    if set(grouped) != expected:
        raise ValueError(f"text formal grid is incomplete: {sorted(expected - set(grouped))}")
    summary = []
    for (model, alpha), selected in sorted(grouped.items()):
        if {row["seed"] for row in selected} != set(TEXT_SEEDS):
            raise ValueError(f"text condition seeds differ: {(model, alpha)}")
        resource = protocol["resources"]["models"][model]
        events = sum(row["events"] for row in selected)
        misses = sum(row["misses"] for row in selected)
        base = {
            "model": model,
            "alpha": alpha,
            "replicas": 3,
            "linear_time_noise_std_fraction": selected[0]["linear_time_noise_std_fraction"],
            "log_time_noise_std_fraction": selected[0]["log_time_noise_std_fraction"],
            "pooled_events": events,
            "pooled_misses": misses,
            "pooled_miss_rate": misses / events,
        }
        if model == "roberta_base":
            clean = float(resource["converted_clean"]["accuracy"])
            mean, low, high = interval([float(row["accuracy"]) for row in selected])
            base.update(
                metric="accuracy",
                converted_clean_metric=clean,
                metric_mean=mean,
                metric_ci_low=low,
                metric_ci_high=high,
                metric_change_mean=100.0 * (mean - clean),
                metric_change_ci_low=100.0 * (low - clean),
                metric_change_ci_high=100.0 * (high - clean),
                metric_change_unit="percentage_points",
            )
        else:
            clean = float(resource["converted_clean"]["token_weighted_perplexity"])
            mean, low, high = interval(
                [float(row["token_weighted_perplexity"]) for row in selected]
            )
            loss_clean = float(resource["converted_clean"]["token_weighted_loss"])
            loss_mean, loss_low, loss_high = interval(
                [float(row["token_weighted_loss"]) for row in selected]
            )
            base.update(
                metric="token_weighted_perplexity",
                converted_clean_metric=clean,
                metric_mean=mean,
                metric_ci_low=low,
                metric_ci_high=high,
                metric_change_mean=100.0 * (mean / clean - 1.0),
                metric_change_ci_low=100.0 * (low / clean - 1.0),
                metric_change_ci_high=100.0 * (high / clean - 1.0),
                metric_change_unit="percent",
                converted_clean_token_weighted_loss=loss_clean,
                token_weighted_loss_mean=loss_mean,
                token_weighted_loss_ci_low=loss_low,
                token_weighted_loss_ci_high=loss_high,
                token_weighted_loss_change=loss_mean - loss_clean,
            )
        summary.append(base)
    return summary


def baselines(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Return dense and converted-clean task metrics."""

    rows = []
    for model in TEXT_MODELS:
        resource = protocol["resources"]["models"][model]
        ann = resource["ann_reference"]
        clean = resource["converted_clean"]
        rows.append(
            {
                "model": model,
                "metric": "accuracy" if model == "roberta_base" else "token_weighted_perplexity",
                "ann_metric": (
                    ann["accuracy"]
                    if model == "roberta_base"
                    else ann["token_weighted_perplexity"]
                ),
                "converted_clean_metric": (
                    clean["accuracy"]
                    if model == "roberta_base"
                    else clean["token_weighted_perplexity"]
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the union of row fields so task-native columns can coexist."""

    if not rows:
        raise ValueError("cannot write an empty text result table")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(summary: list[dict[str, Any]], output: Path) -> None:
    """Render separate task-native panels sharing the noise multiplier axis."""

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    specifications = (
        ("roberta_base", "RoBERTa-B", "Accuracy change from deterministic baseline (pp)"),
        ("gpt2", "GPT-2", "Corpus perplexity change from deterministic baseline (%)"),
    )
    for axis, (model, title, ylabel) in zip(axes, specifications, strict=True):
        selected = sorted(
            (row for row in summary if row["model"] == model),
            key=lambda row: row["alpha"],
        )
        x = [row["alpha"] for row in selected]
        mean = [row["metric_change_mean"] for row in selected]
        low = [row["metric_change_ci_low"] for row in selected]
        high = [row["metric_change_ci_high"] for row in selected]
        axis.plot(x, mean, marker="o", linewidth=2.0, label=title)
        axis.fill_between(x, low, high, alpha=0.15)
        axis.set_xscale("log")
        axis.axvline(1.0, color="black", linestyle=":", linewidth=1.4)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Noise scale multiplier")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25, which="both")
    figure.suptitle("Text-model sensitivity to measured encoder timing noise")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def render_accuracy_models(
    summary: list[dict[str, Any]], vision_root: Path, output: Path
) -> None:
    """Add RoBERTa-B to the existing accuracy-change model comparison."""

    with (vision_root / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        vision = list(csv.DictReader(handle))
    labels = {
        "cct7": "CCT-7",
        "imagenet_vit_small": "ViT-S/16",
        "imagenet_vit_base": "ViT-B/16",
        "roberta_base": "RoBERTa-B",
    }
    rows = [
        {
            "model": row["model"],
            "alpha": float(row["alpha"]),
            "mean": float(row["accuracy_change_pp_mean"]),
            "low": float(row["accuracy_change_pp_ci_low"]),
            "high": float(row["accuracy_change_pp_ci_high"]),
        }
        for row in vision
    ]
    rows += [
        {
            "model": row["model"],
            "alpha": float(row["alpha"]),
            "mean": float(row["metric_change_mean"]),
            "low": float(row["metric_change_ci_low"]),
            "high": float(row["metric_change_ci_high"]),
        }
        for row in summary
        if row["model"] == "roberta_base"
    ]
    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for model in labels:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: row["alpha"],
        )
        if not selected:
            raise ValueError(f"combined accuracy figure is missing {model}")
        x = [row["alpha"] for row in selected]
        mean = [row["mean"] for row in selected]
        low = [row["low"] for row in selected]
        high = [row["high"] for row in selected]
        axis.plot(x, mean, marker="o", linewidth=2.0, label=labels[model])
        axis.fill_between(x, low, high, alpha=0.15)
    axis.set_xscale("log")
    axis.axvline(1.0, color="black", linestyle=":", linewidth=1.4)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.text(
        1.0,
        0.02,
        "Measured screening median",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
    )
    axis.set_xlabel("Noise scale multiplier")
    axis.set_ylabel("Accuracy change from deterministic baseline (pp)")
    axis.set_title("Accuracy under measured encoder timing noise")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


# @lat: [[evaluation#Evaluation and Verification#Screening Median Text Timing Noise Sweep#Summary Artifacts]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-root", type=Path, required=True)
    args = parser.parse_args()
    protocol, raw = load_rows(args.input_root.resolve(strict=True))
    summary = summarize(protocol, raw)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "text_formal_raw.csv", raw)
    write_csv(output / "text_aggregate.csv", summary)
    write_csv(output / "text_baselines.csv", baselines(protocol))
    render(summary, output / "text_timing_noise_sensitivity")
    render_accuracy_models(
        summary,
        args.vision_root.resolve(strict=True),
        output / "model_timing_noise_accuracy_with_roberta",
    )


if __name__ == "__main__":
    main()
