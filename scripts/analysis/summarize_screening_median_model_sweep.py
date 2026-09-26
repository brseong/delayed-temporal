#!/usr/bin/env python3
"""Validate and plot completed screening median model timing-noise cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.screening_median_model_sweep import (
    MODELS,
    SEEDS,
    alpha_slug,
)
from scripts.runtime import files as runtime_files
from scripts.runtime import identity


T_CRITICAL_DF2_975 = 4.302652729911275
MODEL_LABELS = {
    "cct7": "CCT-7",
    "imagenet_vit_small": "ViT-S/16",
    "imagenet_vit_base": "ViT-B/16",
}


def _read_protocol(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if payload.get("protocol_id") != identity.json_sha256(payload.get("identity", {})):
        raise ValueError(f"protocol identity differs: {root}")
    return payload


def _cell_key(manifest: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(manifest["model"]),
        str(manifest["alpha"]),
        int(manifest["seed"]),
        int(manifest["evaluation_samples"]),
    )


def load_completed_cells(
    roots: Iterable[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Load a protocol-consistent union and reject conflicting duplicates."""

    resolved = tuple(root.resolve(strict=True) for root in roots)
    if not resolved:
        raise ValueError("at least one input root is required")
    protocols = [_read_protocol(root) for root in resolved]
    protocol_ids = {protocol["protocol_id"] for protocol in protocols}
    if len(protocol_ids) != 1:
        raise ValueError("input roots use different screening median protocols")
    protocol = protocols[0]
    cells: dict[tuple[str, str, int, int], tuple[dict[str, Any], str, str]] = {}
    source_hashes: dict[str, str] = {}
    for root in resolved:
        source_hashes[f"{root}/protocol.json"] = identity.sha256_file(root / "protocol.json")
        for manifest_path in sorted(root.glob("formal/*/alpha_*/seed_*/manifest.json")):
            result_path = manifest_path.with_name("result.json")
            if not result_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("state") != "complete":
                continue
            if manifest.get("protocol_id") != protocol["protocol_id"]:
                raise ValueError(f"cell protocol differs: {manifest_path}")
            key = _cell_key(manifest)
            manifest_hash = identity.sha256_file(manifest_path)
            result_hash = identity.sha256_file(result_path)
            if key in cells:
                _, previous_manifest, previous_result = cells[key]
                if (manifest_hash, result_hash) != (previous_manifest, previous_result):
                    raise ValueError(f"conflicting duplicate cell: {key}")
                continue
            cells[key] = (
                {"manifest": manifest, "result": result, "root": str(root)},
                manifest_hash,
                result_hash,
            )
            source_hashes[str(manifest_path)] = manifest_hash
            source_hashes[str(result_path)] = result_hash
    rows = []
    hardware = protocol["identity"]["hardware"]
    base_np = float(hardware["phi_np"]["validation_rt"])
    base_nl = float(hardware["phi_nl"]["validation_rt"])
    for key, (payload, _, _) in sorted(cells.items()):
        manifest, result = payload["manifest"], payload["result"]
        model, alpha_text, seed, samples = key
        if model not in MODELS or seed not in SEEDS:
            raise ValueError(f"cell identity is outside the protocol: {key}")
        expected_samples = 10_000 if model == "cct7" else 5_000
        metrics = result.get("metrics", {})
        counts = metrics.get("physical_counts", {})
        alpha = float(alpha_text)
        if (
            samples != expected_samples
            or metrics.get("total") != expected_samples
            or manifest.get("time_noise_scope") != "model_wide"
            or manifest.get("deadline_margin_sigma_ratio") != 4.0
            or not math.isclose(
                float(manifest["linear_time_noise_std_fraction"]),
                base_np * alpha,
                rel_tol=2.0e-15,
                abs_tol=1.0e-18,
            )
            or not math.isclose(
                float(manifest["log_time_noise_std_fraction"]),
                base_nl * alpha,
                rel_tol=2.0e-15,
                abs_tol=1.0e-18,
            )
        ):
            raise ValueError(f"cell contract differs: {key}")
        accuracy = float(metrics["accuracy"])
        if (
            not 0.0 <= accuracy <= 1.0
            or int(metrics["correct"]) / expected_samples != accuracy
        ):
            raise ValueError(f"cell accuracy differs: {key}")
        rows.append(
            {
                "model": model,
                "alpha": alpha,
                "alpha_text": alpha_text,
                "seed": seed,
                "evaluation_samples": samples,
                "linear_time_noise_std_fraction": manifest[
                    "linear_time_noise_std_fraction"
                ],
                "log_time_noise_std_fraction": manifest[
                    "log_time_noise_std_fraction"
                ],
                "accuracy": accuracy,
                "correct": int(metrics["correct"]),
                "prediction_sha256": metrics["prediction_sha256"],
                "events": int(counts["events"]),
                "misses": int(counts["misses"]),
                "deadline_events": int(counts["deadline_events"]),
                "outputs": int(counts["outputs"]),
                "underflows": int(counts["underflows"]),
                "overflows": int(counts["overflows"]),
                "source_root": payload["root"],
            }
        )
    return protocol, rows, source_hashes


def summarize(
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    allow_incomplete: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["alpha"]), []).append(row)
    alphas = {float(row["alpha"]) for row in rows}
    expected_groups = {(model, alpha) for model in MODELS for alpha in alphas}
    if not allow_incomplete and set(grouped) != expected_groups:
        missing = sorted(expected_groups.difference(grouped))
        raise ValueError(f"formal model/alpha cells are incomplete: {missing}")
    summary = []
    for (model, alpha), selected in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in selected}
        if seeds != set(SEEDS):
            if allow_incomplete:
                continue
            raise ValueError(f"missing or duplicate seeds for {(model, alpha)}")
        clean = float(
            protocol["resources"]["models"][model]["converted_clean"]["accuracy"]
        )
        values = [float(row["accuracy"]) for row in selected]
        mean = statistics.fmean(values)
        half_width = T_CRITICAL_DF2_975 * statistics.stdev(values) / math.sqrt(3)
        events = sum(int(row["events"]) for row in selected)
        misses = sum(int(row["misses"]) for row in selected)
        summary.append(
            {
                "model": model,
                "alpha": alpha,
                "replicas": 3,
                "linear_time_noise_std_fraction": selected[0][
                    "linear_time_noise_std_fraction"
                ],
                "log_time_noise_std_fraction": selected[0][
                    "log_time_noise_std_fraction"
                ],
                "converted_clean_accuracy": clean,
                "accuracy_mean": mean,
                "accuracy_ci_low": mean - half_width,
                "accuracy_ci_high": mean + half_width,
                "accuracy_change_pp_mean": 100.0 * (mean - clean),
                "accuracy_change_pp_ci_low": 100.0 * (mean - half_width - clean),
                "accuracy_change_pp_ci_high": 100.0 * (mean + half_width - clean),
                "pooled_events": events,
                "pooled_misses": misses,
                "pooled_miss_rate": misses / events,
            }
        )
    if not summary:
        raise ValueError("no complete three-seed formal conditions were found")
    return summary


def baseline_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        resource = protocol["resources"]["models"][model]
        ann = resource["ann_reference"]
        clean = resource["converted_clean"]
        rows.append(
            {
                "model": model,
                "ann_accuracy": ann["accuracy"],
                "converted_clean_accuracy": clean["accuracy"],
                "conversion_change_pp": 100.0 * (clean["accuracy"] - ann["accuracy"]),
                "ann_prediction_sha256": ann["prediction_sha256"],
                "converted_clean_prediction_sha256": clean["prediction_sha256"],
            }
        )
    return rows


def model_accuracy_rows(
    protocol: dict[str, Any],
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the compact paper table across all completed multipliers."""

    alphas = sorted({float(row["alpha"]) for row in summary})
    rows = []
    for model in MODELS:
        resource = protocol["resources"]["models"][model]
        ann = float(resource["ann_reference"]["accuracy"])
        clean = float(resource["converted_clean"]["accuracy"])
        row: dict[str, Any] = {
            "model": model,
            "ann_accuracy": ann,
            "converted_clean_accuracy": clean,
            "conversion_change_pp": 100.0 * (clean - ann),
        }
        selected = {
            float(item["alpha"]): item
            for item in summary
            if item["model"] == model
        }
        for alpha in alphas:
            if alpha not in selected:
                raise ValueError(f"paper table is missing {(model, alpha)}")
            item = selected[alpha]
            suffix = alpha_slug(str(alpha))
            row[f"alpha_{suffix}_accuracy_mean"] = item["accuracy_mean"]
            row[f"alpha_{suffix}_accuracy_ci_low"] = item["accuracy_ci_low"]
            row[f"alpha_{suffix}_accuracy_ci_high"] = item["accuracy_ci_high"]
        if 1.0 not in selected:
            raise ValueError(f"paper table is missing the measured pair for {model}")
        row["alpha_1_pooled_miss_rate"] = selected[1.0]["pooled_miss_rate"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(summary: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for model in MODELS:
        selected = sorted(
            (row for row in summary if row["model"] == model),
            key=lambda row: float(row["alpha"]),
        )
        if not selected:
            continue
        x = [float(row["alpha"]) for row in selected]
        mean = [float(row["accuracy_change_pp_mean"]) for row in selected]
        low = [float(row["accuracy_change_pp_ci_low"]) for row in selected]
        high = [float(row["accuracy_change_pp_ci_high"]) for row in selected]
        axis.plot(x, mean, marker="o", linewidth=2.0, label=MODEL_LABELS[model])
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
    axis.set_ylabel("Top-1 accuracy change from deterministic baseline (pp)")
    axis.set_title("Accuracy under measured encoder timing noise")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol, raw, source_hashes = load_completed_cells(args.input_root)
    summary = summarize(protocol, raw, allow_incomplete=args.allow_incomplete)
    baselines = baseline_rows(protocol)
    write_csv(output / "formal_raw.csv", raw)
    write_csv(output / "aggregate.csv", summary)
    write_csv(output / "baselines.csv", baselines)
    figure = output / "model_timing_noise_accuracy"
    if not args.allow_incomplete:
        accuracy_table = model_accuracy_rows(protocol, summary)
        write_csv(output / "model_accuracy_table.csv", accuracy_table)
        render(summary, figure)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "input_roots": [str(path.resolve()) for path in args.input_root],
        "source_hashes": source_hashes,
        "raw_sha256": identity.sha256_file(output / "formal_raw.csv"),
        "aggregate_sha256": identity.sha256_file(output / "aggregate.csv"),
        "baselines_sha256": identity.sha256_file(output / "baselines.csv"),
        "complete": not args.allow_incomplete,
    }
    if not args.allow_incomplete:
        manifest.update(
            model_accuracy_table_sha256=identity.sha256_file(
                output / "model_accuracy_table.csv"
            ),
            figure_png_sha256=identity.sha256_file(figure.with_suffix(".png")),
            figure_pdf_sha256=identity.sha256_file(figure.with_suffix(".pdf")),
        )
    runtime_files.atomic_json(output / "aggregate_sources.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
