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
    "best-measured-coordinate": "best measured coordinate",
    "screening-median": "screening median",
}
COLORS = {
    "best-measured-coordinate": "#1f77b4",
    "screening-median": "#d62728",
}
SCALED_SPARSE_CONDITION = "screening-median"


def parse_max_first_block_counts(values: list[str]) -> dict[str, int] | None:
    """Parse one explicit stopping depth for each measured condition."""
    if not values:
        return None
    parsed: dict[str, int] = {}
    for value in values:
        condition, separator, raw_count = value.partition("=")
        if not separator or condition not in DISPLAY_NAMES or condition in parsed:
            raise ValueError("maximum block counts must uniquely name both conditions")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise ValueError("maximum block count must be an integer") from error
        if not 1 <= count <= 12:
            raise ValueError("maximum block count must be between 1 and 12")
        parsed[condition] = count
    if set(parsed) != set(DISPLAY_NAMES):
        raise ValueError("maximum block counts must uniquely name both conditions")
    return parsed


def selected_cells(
    max_first_block_counts: dict[str, int] | None,
) -> tuple[tuple[str, int, int], ...]:
    """Return the complete grid or the explicitly stopped contiguous prefixes."""
    cells = expected_cells()
    if max_first_block_counts is None:
        return cells
    return tuple(
        cell
        for cell in cells
        if cell[0] == "clean" or cell[1] <= max_first_block_counts[cell[0]]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(
    root: Path,
    phase: str,
    max_first_block_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    expected_samples = PHASE_SAMPLES[phase]
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    shared_identity: tuple[Any, ...] | None = None
    cells = selected_cells(max_first_block_counts)
    for condition, first_block_count, seed in cells:
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
    if len(rows) != len(cells):
        raise ValueError("depth summary run population differs")
    return rows, source_hashes


def scaled_run_id(
    condition: str,
    measured_noise_scale: float,
    first_block_count: int,
    seed: int,
) -> str:
    """Return the condition-run identity used by scaled depth evaluations."""
    scale = format(measured_noise_scale, ".12g").replace(".", "p")
    return f"{condition}_scale{scale}_k{first_block_count:02d}_seed{seed}"


def load_scaled_sparse_runs(
    root: Path,
    *,
    phase: str,
    condition: str,
    measured_noise_scales: tuple[float, ...],
    first_block_counts: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load one authenticated sparse prefix grid at multiple noise scales."""
    if condition != SCALED_SPARSE_CONDITION:
        raise ValueError("scaled sparse summaries require the screening median")
    if (
        not measured_noise_scales
        or len(set(measured_noise_scales)) != len(measured_noise_scales)
        or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in measured_noise_scales)
    ):
        raise ValueError("scaled sparse summaries require unique scales in (0, 1]")
    if (
        not first_block_counts
        or tuple(sorted(set(first_block_counts))) != first_block_counts
        or any(not 1 <= value <= 12 for value in first_block_counts)
    ):
        raise ValueError("scaled sparse depths must be unique, sorted, and between 1 and 12")

    expected_samples = PHASE_SAMPLES[phase]
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    shared_identity: tuple[Any, ...] | None = None
    for scale in measured_noise_scales:
        for first_block_count in first_block_counts:
            for seed in range(3):
                name = scaled_run_id(condition, scale, first_block_count, seed)
                run_root = root / "runs" / name
                manifest_path = run_root / "manifest.json"
                result_path = run_root / "result.json"
                if not manifest_path.is_file() or not result_path.is_file():
                    raise ValueError(f"missing scaled depth result: {name}")
                manifest = json.loads(manifest_path.read_text())
                result = json.loads(result_path.read_text())
                metrics = result.get("metrics", {})
                counts = metrics.get("gaussian_counts", {})
                measured = manifest.get("measured_condition", {})
                linear = float(manifest.get("linear_time_std_fraction", math.nan))
                logarithmic = float(manifest.get("log_time_std_fraction", math.nan))
                if (
                    result.get("state") != "complete"
                    or manifest.get("condition") != condition
                    or manifest.get("first_block_count") != first_block_count
                    or manifest.get("seed") != seed
                    or manifest.get("evaluation_samples") != expected_samples
                    or metrics.get("total") != expected_samples
                    or manifest.get("measured_noise_scale") != scale
                    or counts.get("active_blocks") != list(range(first_block_count))
                    or counts.get("site_count") != 26 * first_block_count
                    or counts.get("events", 0) <= 0
                    or not math.isclose(
                        linear,
                        float(measured.get("linear_time_std_fraction", math.nan)) * scale,
                        rel_tol=1e-15,
                        abs_tol=0.0,
                    )
                    or not math.isclose(
                        logarithmic,
                        float(measured.get("log_time_std_fraction", math.nan)) * scale,
                        rel_tol=1e-15,
                        abs_tol=0.0,
                    )
                ):
                    raise ValueError(f"scaled depth result contract differs: {name}")
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
                    raise ValueError("scaled depth results mix source or data identities")
                rows.append(
                    {
                        "phase": phase,
                        "evaluation_samples": expected_samples,
                        "condition": condition,
                        "measured_noise_scale": scale,
                        "first_block_count": first_block_count,
                        "seed": seed,
                        "linear_time_std_fraction": linear,
                        "log_time_std_fraction": logarithmic,
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
                source_hashes[f"runs/{name}/manifest.json"] = identity.sha256_file(
                    manifest_path
                )
                source_hashes[f"runs/{name}/result.json"] = identity.sha256_file(
                    result_path
                )
    expected_count = len(measured_noise_scales) * len(first_block_counts) * 3
    if len(rows) != expected_count:
        raise ValueError("scaled depth summary run population differs")
    return rows, source_hashes


def load_clean_reference(path: Path, *, expected_samples: int) -> dict[str, Any]:
    """Load the frozen deterministic converted-model result used at K=0."""
    payload = json.loads(path.read_text())
    metrics = payload.get("phases", {}).get("snn", {}).get("metrics", {})
    if (
        payload.get("state") != "complete"
        or metrics.get("total") != expected_samples
        or not math.isfinite(float(metrics.get("accuracy", math.nan)))
        or not isinstance(metrics.get("prediction_sha256"), str)
        or len(metrics["prediction_sha256"]) != 64
    ):
        raise ValueError("clean depth reference contract differs")
    return metrics


def validate_stopping_decision(
    root: Path,
    *,
    max_first_block_counts: dict[str, int],
    accuracy_threshold: float,
) -> dict[str, Any]:
    """Validate that each retained prefix ends at its first near-zero mean."""
    if not math.isfinite(accuracy_threshold) or not 0.0 <= accuracy_threshold <= 1.0:
        raise ValueError("accuracy threshold must be finite and between zero and one")
    rows, hashes = load_runs(root, "pilot", max_first_block_counts)
    decisions: dict[str, Any] = {}
    for condition, maximum in max_first_block_counts.items():
        means: dict[str, float] = {}
        for first_block_count in range(1, maximum + 1):
            values = [
                float(row["accuracy"])
                for row in rows
                if row["condition"] == condition
                and row["first_block_count"] == first_block_count
            ]
            if len(values) != 3:
                raise ValueError("stopping decision requires three seeds at every retained depth")
            means[str(first_block_count)] = statistics.fmean(values)
        if means[str(maximum)] > accuracy_threshold:
            raise ValueError("selected stopping depth does not meet the accuracy threshold")
        if any(means[str(index)] <= accuracy_threshold for index in range(1, maximum)):
            raise ValueError("selected stopping depth is not the first threshold crossing")
        decisions[condition] = {
            "max_first_block_count": maximum,
            "accuracy_means": means,
        }
    return {
        "rule": "stop after the mean accuracy is at most the stated threshold",
        "accuracy_threshold": accuracy_threshold,
        "conditions": decisions,
        "source_hashes": hashes,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    if len(clean_rows) != 1:
        raise ValueError("depth summary requires exactly one clean run")
    clean = float(clean_rows[0]["accuracy"])
    summary: list[dict[str, Any]] = []
    for condition in DISPLAY_NAMES:
        observed = sorted(
            {
                int(row["first_block_count"])
                for row in rows
                if row["condition"] == condition
            }
        )
        if not observed or observed != list(range(1, max(observed) + 1)):
            raise ValueError("depth summary requires a contiguous condition prefix")
        for first_block_count in range(max(observed) + 1):
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


def summarize_scaled_sparse(
    rows: list[dict[str, Any]],
    *,
    clean_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize three seeds per sparse depth and reuse one frozen clean point."""
    clean = float(clean_metrics["accuracy"])
    summary: list[dict[str, Any]] = []
    scales = sorted({float(row["measured_noise_scale"]) for row in rows})
    depths = sorted({int(row["first_block_count"]) for row in rows})
    for scale in scales:
        summary.append(
            {
                "phase": rows[0]["phase"],
                "evaluation_samples": rows[0]["evaluation_samples"],
                "condition": SCALED_SPARSE_CONDITION,
                "measured_noise_scale": scale,
                "first_block_count": 0,
                "replicas": 1,
                "linear_time_std_fraction": 0.0,
                "log_time_std_fraction": 0.0,
                "accuracy_mean": clean,
                "accuracy_ci_low": clean,
                "accuracy_ci_high": clean,
                "accuracy_change_pp_mean": 0.0,
                "accuracy_change_pp_ci_low": 0.0,
                "accuracy_change_pp_ci_high": 0.0,
                "pooled_events": 0,
                "pooled_misses": 0,
            }
        )
        for first_block_count in depths:
            selected = [
                row
                for row in rows
                if row["measured_noise_scale"] == scale
                and row["first_block_count"] == first_block_count
            ]
            if len(selected) != 3 or {int(row["seed"]) for row in selected} != {0, 1, 2}:
                raise ValueError("scaled depth summary has a missing or duplicate seed")
            values = [float(row["accuracy"]) for row in selected]
            mean = statistics.fmean(values)
            half_width = T_CRITICAL_DF2_975 * statistics.stdev(values) / math.sqrt(3)
            low, high = mean - half_width, mean + half_width
            summary.append(
                {
                    "phase": selected[0]["phase"],
                    "evaluation_samples": selected[0]["evaluation_samples"],
                    "condition": SCALED_SPARSE_CONDITION,
                    "measured_noise_scale": scale,
                    "first_block_count": first_block_count,
                    "replicas": 3,
                    "linear_time_std_fraction": selected[0][
                        "linear_time_std_fraction"
                    ],
                    "log_time_std_fraction": selected[0]["log_time_std_fraction"],
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


def render_scaled_sparse(summary: list[dict[str, Any]], output: Path) -> None:
    """Render the two fixed noise-scale curves on one accuracy axis."""
    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, scale in enumerate(
        sorted({float(row["measured_noise_scale"]) for row in summary})
    ):
        color = colors(index)
        rows = [row for row in summary if row["measured_noise_scale"] == scale]
        x = [int(row["first_block_count"]) for row in rows]
        mean = [100.0 * float(row["accuracy_mean"]) for row in rows]
        low = [100.0 * float(row["accuracy_ci_low"]) for row in rows]
        high = [100.0 * float(row["accuracy_ci_high"]) for row in rows]
        axis.plot(x, mean, marker="o", color=color, label=fr"$\alpha={scale:g}$")
        axis.fill_between(x, low, high, color=color, alpha=0.16)
    depths = sorted({int(row["first_block_count"]) for row in summary})
    axis.set_xticks(depths)
    axis.set_xlabel("Number of noisy encoder blocks from the input")
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, title="Noise scale relative to the screening median")
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    maximum = max(int(row["first_block_count"]) for row in summary)
    for axis in axes:
        axis.set_xticks(range(maximum + 1))
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
    parser.add_argument(
        "--max-first-block-count",
        action="append",
        default=[],
        metavar="CONDITION=K",
    )
    parser.add_argument("--stopping-decision-root", type=Path)
    parser.add_argument("--stopping-accuracy-threshold", type=float)
    parser.add_argument(
        "--scaled-sparse-condition",
        choices=(SCALED_SPARSE_CONDITION,),
    )
    parser.add_argument(
        "--measured-noise-scale",
        action="append",
        type=float,
        default=[],
    )
    parser.add_argument(
        "--first-block-count",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument("--clean-reference-result", type=Path)
    args = parser.parse_args()
    root = args.input_root.resolve(strict=True)
    scaled_options = (
        args.scaled_sparse_condition is not None,
        bool(args.measured_noise_scale),
        bool(args.first_block_count),
        args.clean_reference_result is not None,
    )
    if any(scaled_options):
        if not all(scaled_options):
            raise ValueError("scaled sparse summary options must be supplied together")
        if args.max_first_block_count or args.stopping_decision_root is not None:
            raise ValueError("scaled sparse summary does not use stopping options")
        if args.stopping_accuracy_threshold is not None:
            raise ValueError("scaled sparse summary does not use a stopping threshold")
        scales = tuple(sorted(args.measured_noise_scale))
        depths = tuple(args.first_block_count)
        clean_reference_path = args.clean_reference_result.resolve(strict=True)
        raw, source_hashes = load_scaled_sparse_runs(
            root,
            phase=args.phase,
            condition=args.scaled_sparse_condition,
            measured_noise_scales=scales,
            first_block_counts=depths,
        )
        clean_metrics = load_clean_reference(
            clean_reference_path,
            expected_samples=PHASE_SAMPLES[args.phase],
        )
        summary = summarize_scaled_sparse(raw, clean_metrics=clean_metrics)
        raw_path = root / "depth_noise_scaled_sparse_raw.csv"
        summary_path = root / "depth_noise_scaled_sparse_summary.csv"
        write_csv(raw_path, raw, tuple(raw[0]))
        write_csv(summary_path, summary, tuple(summary[0]))
        figure = root / "depth_noise_scaled_sparse_accuracy"
        render_scaled_sparse(summary, figure)
        manifest = {
            "schema_version": 1,
            "phase": args.phase,
            "condition": args.scaled_sparse_condition,
            "measured_noise_scales": scales,
            "first_block_counts": depths,
            "source_hashes": source_hashes,
            "clean_reference_result": str(clean_reference_path),
            "clean_reference_result_sha256": identity.sha256_file(
                clean_reference_path
            ),
            "clean_prediction_sha256": clean_metrics["prediction_sha256"],
            "raw_csv_sha256": identity.sha256_file(raw_path),
            "summary_csv_sha256": identity.sha256_file(summary_path),
            "figure_pdf_sha256": identity.sha256_file(figure.with_suffix(".pdf")),
            "figure_png_sha256": identity.sha256_file(figure.with_suffix(".png")),
            "interpretation": (
                "cumulative sensitivity to scaled measured marginal encoder timing "
                "noise fractions; not causal attribution to an individual block"
            ),
        }
        runtime_files.atomic_json(root / "scaled_sparse_summary_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    max_first_block_counts = parse_max_first_block_counts(args.max_first_block_count)
    stopping_options = (
        args.stopping_decision_root is not None,
        args.stopping_accuracy_threshold is not None,
    )
    if max_first_block_counts is None and any(stopping_options):
        raise ValueError("stopping evidence requires explicit maximum block counts")
    if max_first_block_counts is not None and not all(stopping_options):
        raise ValueError("explicit maximum block counts require stopping evidence")
    stopping_decision = None
    if max_first_block_counts is not None:
        decision_root = args.stopping_decision_root.resolve(strict=True)
        stopping_decision = validate_stopping_decision(
            decision_root,
            max_first_block_counts=max_first_block_counts,
            accuracy_threshold=args.stopping_accuracy_threshold,
        )
        stopping_decision["root"] = str(decision_root)
    raw, source_hashes = load_runs(root, args.phase, max_first_block_counts)
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
        "max_first_block_count_by_condition": max_first_block_counts,
        "stopping_decision": stopping_decision,
        "interpretation": (
            "cumulative sensitivity to measured marginal encoder timing-noise scales; "
            "not causal attribution to an individual block"
        ),
    }
    runtime_files.atomic_json(root / "summary_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
