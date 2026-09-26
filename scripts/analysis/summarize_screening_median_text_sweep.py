#!/usr/bin/env python3
"""Aggregate screening-median RoBERTa and GPT-2 replica artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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

from scripts.experiments.screening_median_text_sweep import TEXT_MODELS, TEXT_SEEDS


T_CRITICAL_DF2_975 = 4.302652729696142
PAPER_PANEL_WIDTH_IN = 3.35
PAPER_PANEL_HEIGHT_IN = 2.15
PAPER_PANEL_FONT_SIZE = 8.0
PAPER_PANEL_AXIS_LABEL_SIZE = 12.5
PAPER_PANEL_LEGEND_SIZE = 8.0


@dataclass(frozen=True)
class ComparisonFigureStyle:
    """Control the physical size and typography of the combined model panel."""

    width_in: float = PAPER_PANEL_WIDTH_IN
    height_in: float = PAPER_PANEL_HEIGHT_IN
    font_size: float = PAPER_PANEL_FONT_SIZE
    axis_label_size: float = PAPER_PANEL_AXIS_LABEL_SIZE
    legend_size: float = PAPER_PANEL_LEGEND_SIZE
    legend_columns: int = 5
    legend_above: bool = True
    gpt2_max_alpha: float | None = None
    show_title: bool = False
    compact_labels: bool = True


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


def summarize(
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Reduce each model and multiplier to task-native uncertainty metrics."""

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["alpha"]), []).append(row)
    alphas = {row["alpha"] for row in rows}
    expected = {(model, alpha) for model in TEXT_MODELS for alpha in alphas}
    if not allow_incomplete and set(grouped) != expected:
        raise ValueError(f"text formal grid is incomplete: {sorted(expected - set(grouped))}")
    summary = []
    for (model, alpha), selected in sorted(grouped.items()):
        if {row["seed"] for row in selected} != set(TEXT_SEEDS):
            if allow_incomplete:
                continue
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
            perplexities = [
                float(row["token_weighted_perplexity"]) for row in selected
            ]
            mean, low, high = interval(perplexities)
            inverse_mean, inverse_low, inverse_high = interval(
                [-100.0 * (1.0 - clean / value) for value in perplexities]
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
                relative_inverse_perplexity_change_mean=inverse_mean,
                relative_inverse_perplexity_change_ci_low=inverse_low,
                relative_inverse_perplexity_change_ci_high=inverse_high,
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
    figure.savefig(
        output.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def relative_inverse_perplexity_change(
    summary_row: dict[str, Any],
) -> tuple[float, float, float]:
    """Return seed-aggregated relative inverse-perplexity change and interval."""

    if summary_row.get("metric") != "token_weighted_perplexity":
        raise ValueError("inverse-perplexity change requires a perplexity row")
    values = (
        float(summary_row["relative_inverse_perplexity_change_mean"]),
        float(summary_row["relative_inverse_perplexity_change_ci_low"]),
        float(summary_row["relative_inverse_perplexity_change_ci_high"]),
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("inverse-perplexity change requires finite values")
    return values


def aligned_zero_limits(
    values: list[float], *, zero_position: float = 0.88
) -> tuple[float, float]:
    """Return limits that place zero at a fixed vertical axis fraction."""

    if not values or not 0.0 < zero_position < 1.0:
        raise ValueError("aligned limits require values and an interior zero position")
    negative = max(0.0, -min(values))
    positive = max(0.0, max(values))
    scale = max(
        negative / zero_position,
        positive / (1.0 - zero_position),
        1e-12,
    )
    return -scale * zero_position, scale * (1.0 - zero_position)


def render_model_comparison(
    summary: list[dict[str, Any]],
    vision_root: Path,
    output: Path,
    *,
    style: ComparisonFigureStyle | None = None,
) -> None:
    """Render classification accuracy and GPT-2 perplexity on aligned axes."""

    style = style or ComparisonFigureStyle()
    if (
        style.width_in <= 0
        or style.height_in <= 0
        or style.font_size <= 0
        or style.axis_label_size <= 0
        or style.legend_size <= 0
        or style.legend_columns < 1
    ):
        raise ValueError("comparison figure dimensions and typography must be positive")

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
    figure, axis = plt.subplots(
        figsize=(style.width_in, style.height_in), constrained_layout=True
    )
    accuracy_limits = []
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
        accuracy_limits.extend([*low, *high])
        axis.plot(
            x,
            mean,
            marker="o",
            markersize=0.55 * style.font_size,
            linewidth=0.18 * style.font_size,
            label=labels[model],
        )
        axis.fill_between(x, low, high, alpha=0.15)

    perplexity_rows = sorted(
        (
            row
            for row in summary
            if row["model"] == "gpt2"
            and (
                style.gpt2_max_alpha is None
                or float(row["alpha"]) <= style.gpt2_max_alpha
            )
        ),
        key=lambda row: row["alpha"],
    )
    if not perplexity_rows:
        raise ValueError("combined model figure is missing gpt2")
    perplexity_x = [row["alpha"] for row in perplexity_rows]
    perplexity_intervals = [
        relative_inverse_perplexity_change(row) for row in perplexity_rows
    ]
    perplexity_mean = [row[0] for row in perplexity_intervals]
    perplexity_low = [row[1] for row in perplexity_intervals]
    perplexity_high = [row[2] for row in perplexity_intervals]
    perplexity_axis = axis.twinx()
    perplexity_axis.plot(
        perplexity_x,
        perplexity_mean,
        color="black",
        linestyle="--",
        marker="s",
        markersize=0.55 * style.font_size,
        linewidth=0.18 * style.font_size,
        label="GPT-2",
    )
    perplexity_axis.fill_between(
        perplexity_x,
        perplexity_low,
        perplexity_high,
        color="black",
        alpha=0.10,
    )
    axis_label_size = style.axis_label_size
    tick_size = 0.80 * style.font_size
    if style.compact_labels:
        accuracy_label = "Accuracy change\n(pp)"
        perplexity_label = "Inverse-PPL\nchange (%)"
        x_label = r"Noise scale $\alpha$"
    else:
        accuracy_label = "Accuracy change from deterministic baseline (pp)"
        perplexity_label = "Relative inverse-perplexity change (%)"
        x_label = "Noise scale multiplier"
    perplexity_axis.set_ylabel(perplexity_label, fontsize=axis_label_size)
    axis.set_ylim(*aligned_zero_limits(accuracy_limits))
    perplexity_axis.set_ylim(
        *aligned_zero_limits([*perplexity_low, *perplexity_high])
    )
    axis.set_xscale("log")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel(x_label, fontsize=axis_label_size)
    axis.set_ylabel(accuracy_label, fontsize=axis_label_size)
    if style.show_title:
        axis.set_title(
            "Model sensitivity to measured encoder timing noise",
            fontsize=1.05 * style.font_size,
        )
    axis.grid(alpha=0.25, which="both")
    axis.tick_params(axis="both", labelsize=tick_size)
    perplexity_axis.tick_params(axis="both", labelsize=tick_size)
    accuracy_handles, accuracy_labels = axis.get_legend_handles_labels()
    perplexity_handles, perplexity_labels = perplexity_axis.get_legend_handles_labels()
    legend_options: dict[str, Any] = {}
    if style.legend_above:
        legend_options.update(loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axis.legend(
        accuracy_handles + perplexity_handles,
        accuracy_labels + perplexity_labels,
        frameon=False,
        fontsize=style.legend_size,
        ncol=style.legend_columns,
        columnspacing=0.4,
        handlelength=1.0,
        handletextpad=0.3,
        **legend_options,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(
        output.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


# @lat: [[evaluation#Evaluation and Verification#Screening Median Text Timing Noise Sweep#Summary Artifacts]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--comparison-width-in", type=float, default=PAPER_PANEL_WIDTH_IN
    )
    parser.add_argument(
        "--comparison-height-in", type=float, default=PAPER_PANEL_HEIGHT_IN
    )
    parser.add_argument(
        "--comparison-font-size", type=float, default=PAPER_PANEL_FONT_SIZE
    )
    parser.add_argument(
        "--comparison-axis-label-size", type=float,
        default=PAPER_PANEL_AXIS_LABEL_SIZE,
    )
    parser.add_argument(
        "--comparison-legend-size", type=float, default=PAPER_PANEL_LEGEND_SIZE
    )
    parser.add_argument("--comparison-legend-columns", type=int, default=5)
    parser.add_argument(
        "--comparison-legend-above",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--comparison-gpt2-max-alpha", type=float)
    parser.add_argument(
        "--comparison-title", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--comparison-compact-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    protocol, raw = load_rows(args.input_root.resolve(strict=True))
    summary = summarize(protocol, raw, allow_incomplete=args.allow_incomplete)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "text_formal_raw.csv", raw)
    write_csv(output / "text_aggregate.csv", summary)
    write_csv(output / "text_baselines.csv", baselines(protocol))
    render(summary, output / "text_timing_noise_sensitivity")
    render_model_comparison(
        summary,
        args.vision_root.resolve(strict=True),
        output / "model_timing_noise_model_comparison",
        style=ComparisonFigureStyle(
            width_in=args.comparison_width_in,
            height_in=args.comparison_height_in,
            font_size=args.comparison_font_size,
            axis_label_size=args.comparison_axis_label_size,
            legend_size=args.comparison_legend_size,
            legend_columns=args.comparison_legend_columns,
            legend_above=args.comparison_legend_above,
            gpt2_max_alpha=args.comparison_gpt2_max_alpha,
            show_title=args.comparison_title,
            compact_labels=args.comparison_compact_labels,
        ),
    )


if __name__ == "__main__":
    main()
