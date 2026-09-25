#!/usr/bin/env python3
"""Validate and summarize the operator-local paper re-evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
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
from utils.transformers.calibration import (
    OPERATOR_BACKED_OUTPUT_HEAD_VERSION,
    TEXT_CALIBRATION_POLICY_VERSION,
    VIT_CALIBRATION_POLICY_VERSION,
)


T_CRITICAL_DF2 = 4.302652729911275
VIT_MODELS = (
    "cifar10_vit_small", "imagenet_vit_small", "imagenet_vit_base", "imagenet_vit_large",
)
VIT_SITE_COUNTS = {
    "cifar10_vit_small": 109,
    "imagenet_vit_small": 109,
    "imagenet_vit_base": 109,
    "imagenet_vit_large": 217,
}
TEXT_MODELS = (
    ("roberta", TEXT_TAG), ("roberta_large", ROBERTA_LARGE_TAG),
    ("gpt2", GPT2_COMPOSED_GELU_TAG),
)
TEXT_SITE_COUNTS = {"roberta": 110, "roberta_large": 218, "gpt2": 109}


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


def contains_legacy_range_key(value: Any) -> bool:
    """Return whether persisted evidence exposes a removed global range key."""
    legacy = {"theta", "attention_theta", "selected_theta"}
    if isinstance(value, dict):
        return any(
            str(key) in legacy
            or contains_legacy_range_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], str) and value[0] in legacy:
            return True
        return any(contains_legacy_range_key(item) for item in value)
    if isinstance(value, str):
        return value == "--theta" or value.startswith("--theta=") or value == "--attention-theta"
    return False


def validate_calibration(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_commit: str,
    expected_sites: int,
    family: str,
) -> None:
    if identity.sha256_file(path) != expected_sha256:
        raise ValueError(f"calibration hash differs: {path}")
    table = json.loads(path.read_text())
    if table.get("format_version") != 2 or contains_legacy_range_key(table):
        raise ValueError(f"calibration range contract differs: {path}")
    rows = table.get("layers", [])
    sites = {
        row["module_name"] + "/" + row["tensor_name"]
        for row in rows
    }
    metadata = table.get("metadata", {})
    options = dict(metadata.get("model_options", []))
    if (
        metadata.get("dtype") != "float64"
        or options.get("output_bounds_version") != 4
        or len(rows) != expected_sites
        or len(sites) != expected_sites
    ):
        raise ValueError(f"calibration identity or site population differs: {path}")
    if family == "vit" and (
        options.get("source_commit") != expected_source_commit
        or options.get("vit_calibration_policy_version")
        != VIT_CALIBRATION_POLICY_VERSION
        or options.get("operator_backed_output_head_version")
        != OPERATOR_BACKED_OUTPUT_HEAD_VERSION
    ):
        raise ValueError(f"ViT calibration source or policy differs: {path}")
    if family == "text" and (
        options.get("text_calibration_policy_version")
        != TEXT_CALIBRATION_POLICY_VERSION
        or options.get("operator_backed_output_head_version")
        != OPERATOR_BACKED_OUTPUT_HEAD_VERSION
    ):
        raise ValueError(f"text calibration policy differs: {path}")


def validate_pipeline(
    root: Path,
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
    expected_sites: int,
    family: str,
) -> None:
    if (
        manifest.get("range_contract") != "operator_local_end_to_end_v1"
        or manifest.get("output_bounds_version") != 4
        or manifest.get("dtype") != "float64"
        or manifest.get("noise") is not False
        or contains_legacy_range_key(manifest)
        or contains_legacy_range_key(result)
    ):
        raise ValueError(f"pipeline range contract differs: {root}")
    phases = result.get("phases", {})
    if set(phases) != {"collect", "ann", "snn"}:
        raise ValueError(f"pipeline phase population differs: {root}")
    for phase, record in phases.items():
        if record.get("phase") != phase:
            raise ValueError(f"phase identity differs: {root}/{phase}")
        log_path = root / record["log_file"]
        if identity.sha256_file(log_path) != record["log_sha256"]:
            raise ValueError(f"phase log hash differs: {root}/{phase}")
    calibration_sha256 = result.get("calibration_sha256")
    if calibration_sha256 != phases["collect"].get("calibration_sha256"):
        raise ValueError(f"completed calibration hash differs: {root}")
    if phases["collect"].get("calibration_site_count") != expected_sites:
        raise ValueError(f"collection site count differs: {root}")
    snn_metrics = phases["snn"].get("metrics", {})
    snn_sites = phases["snn"].get("calibration_site_count", snn_metrics.get("calibration_site_count"))
    if snn_sites != expected_sites:
        raise ValueError(f"SNN site count differs: {root}")
    validate_calibration(
        root / "calibration.json",
        expected_sha256=calibration_sha256,
        expected_source_commit=manifest["source_commit"],
        expected_sites=expected_sites,
        family=family,
    )


def validate_metric(value: float, *, accuracy: bool) -> None:
    if (
        not math.isfinite(value)
        or (accuracy and not 0.0 <= value <= 1.0)
        or (not accuracy and value <= 0)
    ):
        raise ValueError("table metric is outside its valid range")


def expected_noise_cells() -> set[tuple[float, float]]:
    """Return the exact 21 unique Figure 4 cells."""
    return {
        (10 ** (-5 + index / 8), 4.0) for index in range(9)
    } | {
        (1e-5, ratio)
        for ratio in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
    }


def table_rows(artifacts: Path) -> list[dict[str, Any]]:
    rows = []
    for model in VIT_MODELS:
        root = artifacts / "logs/conversion_comparison" / VIT_TAG / "vit" / model
        result = read_complete(root / "result.json")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("model_key") != model:
            raise ValueError(f"ViT model identity differs: {model}")
        validate_pipeline(
            root,
            manifest=manifest,
            result=result,
            expected_sites=VIT_SITE_COUNTS[model],
            family="vit",
        )
        ann = result["phases"]["ann"]["metrics"]
        snn = result["phases"]["snn"]["metrics"]
        if ann["total"] != manifest["evaluation_samples"] or snn["total"] != manifest["evaluation_samples"]:
            raise ValueError(f"ViT sample count differs: {model}")
        validate_metric(ann["accuracy"], accuracy=True)
        validate_metric(snn["accuracy"], accuracy=True)
        rows.append({
            "model": model, "family": "vit", "samples": ann["total"],
            "ann_metric": ann["accuracy"], "snn_metric": snn["accuracy"],
            "snn_minus_ann": snn["accuracy"] - ann["accuracy"],
            "metric": "accuracy", "calibration_sha256": result["calibration_sha256"],
            "source_commit": manifest["source_commit"],
        })
    for model, tag in TEXT_MODELS:
        root = artifacts / "logs/conversion_comparison" / tag / "text" / model
        result = read_complete(root / "result.json")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("family") != model:
            raise ValueError(f"text model identity differs: {model}")
        validate_pipeline(
            root,
            manifest=manifest,
            result=result,
            expected_sites=TEXT_SITE_COUNTS[model],
            family="text",
        )
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
        validate_metric(ann_metric, accuracy=model != "gpt2")
        validate_metric(snn_metric, accuracy=model != "gpt2")
        rows.append({
            "model": model, "family": "text", "samples": samples,
            "ann_metric": ann_metric, "snn_metric": snn_metric, "metric": metric,
            "snn_minus_ann": snn_metric - ann_metric,
            "calibration_sha256": result["calibration_sha256"],
            "source_commit": manifest["source_commit"],
        })
    if len({row["source_commit"] for row in rows}) != 1:
        raise ValueError("table results mix source commits")
    return rows


def noise_reference(
    artifacts: Path,
    calibration_source: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = calibration_source or (
        artifacts / "logs/conversion_comparison" / VIT_TAG / "vit/imagenet_vit_base"
    )
    root = root.resolve(strict=True)
    result = read_complete(root / "result.json")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("model_key") != "imagenet_vit_base":
        raise ValueError("noise calibration source is not the ViT-B pipeline")
    validate_pipeline(
        root,
        manifest=manifest,
        result=result,
        expected_sites=VIT_SITE_COUNTS["imagenet_vit_base"],
        family="vit",
    )
    return root, result, manifest


def noise_rows(
    artifacts: Path,
    calibration_source: Path | None = None,
    *,
    noise_tag: str = NOISE_TAG,
    require_raw_timestamp_contract: bool = False,
) -> list[dict[str, Any]]:
    runs = artifacts / "logs/noise_scan" / noise_tag / "runs"
    vit_root, vit_result, vit_manifest = noise_reference(artifacts, calibration_source)
    vit_result_path = vit_root / "result.json"
    expected_cells = expected_noise_cells()
    rows = []
    identities = set()
    run_ids = set()
    for result_path in sorted(runs.glob("*/result.json")):
        result = read_complete(result_path)
        root = result_path.parent
        manifest = json.loads((root / "manifest.json").read_text())
        if identity.sha256_file(root / result["log_file"]) != result["log_sha256"]:
            raise ValueError(f"noise log hash differs: {root.name}")
        metric = result["metrics"]
        counts = metric["physical_counts"]
        if (
            metric["total"] != 5_000
            or manifest["run_id"] != root.name
            or result.get("run_id") != root.name
            or manifest.get("range_contract") != "operator_local_end_to_end_v1"
            or manifest.get("timing_noise_contract") != "local_encoder_window_fraction_v1"
            or manifest.get("dtype") != "float64"
            or contains_legacy_range_key(manifest)
            or contains_legacy_range_key(result)
        ):
            raise ValueError(f"noise run population or identity differs: {root.name}")
        if require_raw_timestamp_contract and (
            manifest.get("raw_timestamp_contract") != "delivered_raw_timestamp_v1"
            or manifest.get("exponential_difference_internal_noise") != "fixed_on_v1"
            or manifest.get("schema_version") != 2
        ):
            raise ValueError(f"raw-timestamp or ED-noise contract differs: {root.name}")
        if root.name in run_ids:
            raise ValueError(f"duplicate noise run identifier: {root.name}")
        run_ids.add(root.name)
        if (
            manifest.get("source_commit") != vit_manifest.get("source_commit")
            or manifest.get("checkpoint_sha256") != vit_manifest.get("checkpoint_sha256")
            or manifest.get("calibration_sha256") != vit_result.get("calibration_sha256")
            or manifest.get("calibration_source_result_sha256") != identity.sha256_file(vit_result_path)
            or manifest.get("evaluation_dataset_path") != vit_manifest["evaluation_dataset"]["path"]
        ):
            raise ValueError(f"noise evidence differs from the ViT-B source: {root.name}")
        validate_metric(metric["accuracy"], accuracy=True)
        if any(
            isinstance(counts.get(key), bool)
            or not isinstance(counts.get(key), int)
            or counts[key] < 0
            for key in ("events", "misses", "deadline_events", "outputs", "underflows", "overflows")
        ):
            raise ValueError(f"noise physical counts are invalid: {root.name}")
        if (
            counts["events"] <= 0
            or counts["outputs"] <= 0
            or counts["misses"] > counts["events"]
            or counts["deadline_events"] > counts["events"]
            or counts["underflows"] + counts["overflows"] > counts["outputs"]
        ):
            raise ValueError(f"noise physical counts are inconsistent: {root.name}")
        identities.add((manifest["source_commit"], manifest["checkpoint_sha256"],
                        manifest["calibration_sha256"], manifest["evaluation_dataset_path"]))
        rows.append({
            "run_id": root.name, "time_noise_std_fraction": manifest["time_noise_std_fraction"],
            "deadline_margin_sigma_ratio": manifest["deadline_margin_sigma_ratio"],
            "seed": manifest["seed"], "correct": metric["correct"], "total": metric["total"],
            "accuracy": metric["accuracy"], "events": counts["events"], "misses": counts["misses"],
            "deadline_events": counts["deadline_events"], "outputs": counts["outputs"],
            "underflows": counts["underflows"], "overflows": counts["overflows"],
            "source_commit": manifest["source_commit"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "calibration_sha256": manifest["calibration_sha256"],
            "host_label": manifest.get("host_label"),
            "physical_gpu": manifest.get("physical_gpu"),
        })
    if len(rows) != 63 or len(identities) != 1:
        raise ValueError("noise evidence is incomplete or identity-mixed")
    cells: dict[tuple[float, float], set[int]] = {}
    for row in rows:
        key = (row["time_noise_std_fraction"], row["deadline_margin_sigma_ratio"])
        cells.setdefault(key, set()).add(row["seed"])
    if set(cells) != expected_cells or any(seeds != {0, 1, 2} for seeds in cells.values()):
        raise ValueError("noise cell or seed population differs")
    return rows


def validate_noise_campaign_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    noise_tag: str,
    reference_result: dict[str, Any],
    reference_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the central host/GPU assignment against completed replicas."""
    campaign = json.loads(path.read_text())
    assignments = campaign.get("assignments", [])
    expected = {row.get("run_id"): row for row in assignments}
    if (
        campaign.get("noise_tag") != noise_tag
        or campaign.get("source_commit") != reference_manifest.get("source_commit")
        or campaign.get("checkpoint_sha256") != reference_manifest.get("checkpoint_sha256")
        or campaign.get("calibration_sha256") != reference_result.get("calibration_sha256")
        or campaign.get("raw_timestamp_contract") != "delivered_raw_timestamp_v1"
        or campaign.get("exponential_difference_internal_noise") != "fixed_on_v1"
        or len(assignments) != 63
        or len(expected) != 63
        or None in expected
        or {row["run_id"] for row in rows} != set(expected)
    ):
        raise ValueError("central noise campaign identity or assignment population differs")
    for row in rows:
        assignment = expected[row["run_id"]]
        if any(
            row[key] != assignment[key]
            for key in (
                "time_noise_std_fraction", "deadline_margin_sigma_ratio", "seed",
                "host_label", "physical_gpu",
            )
        ):
            raise ValueError(f"noise execution differs from central assignment: {row['run_id']}")
    return campaign


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


def render_figure(
    path: Path,
    summary: list[dict[str, Any]],
    *,
    dense_accuracy: float,
    clean_spiking_accuracy: float,
) -> None:
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
        axis.axhline(100 * clean_spiking_accuracy, linestyle="--", linewidth=1.3,
                     color="#237a3b", label="Clean spiking")
        axis.axhline(100 * dense_accuracy, linestyle=":", linewidth=1.3,
                     color="#666666", label="Dense reference")
        axis.errorbar(x, y, yerr=[lower, upper], marker="o", linewidth=1.5, capsize=2,
                      color="#2f5597", label="Noisy accuracy")
        axis.set_xlabel(label)
        axis.set_ylabel("Top-1 accuracy (%)")
        axis.grid(True, alpha=0.25)
        if key == "time_noise_std_fraction":
            axis.set_xscale("log")
            axis.legend(loc="best")
        else:
            secondary = axis.twinx()
            rates = [100 * row["miss_rate"] for row in rows]
            nonzero = [rate for rate in rates if rate > 0]
            floor = min(nonzero) / 10 if nonzero else 1e-12
            plotted_rates = [rate if rate > 0 else floor for rate in rates]
            secondary.plot(x, plotted_rates, marker="s", linewidth=1.2,
                           color="#c65911", label="Deadline-miss rate")
            zero_x = [value for value, rate in zip(x, rates, strict=True) if rate == 0]
            if zero_x:
                secondary.scatter(zero_x, [floor] * len(zero_x), marker="v",
                                  facecolors="white", edgecolors="#c65911", zorder=3)
            secondary.set_yscale("log")
            secondary.set_ylabel("Deadline-miss rate (%)")
            handles, labels = axis.get_legend_handles_labels()
            extra_handles, extra_labels = secondary.get_legend_handles_labels()
            axis.legend(
                handles + extra_handles,
                labels + extra_labels,
                loc="center right",
                bbox_to_anchor=(0.98, 0.5),
                borderaxespad=0.0,
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


# @lat: [[evaluation#Evaluation and Verification#Local-Range Paper Re-evaluation]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("/data/delayed-temporal/artifacts"))
    parser.add_argument("--noise-calibration-source", type=Path)
    parser.add_argument("--noise-tag", default=NOISE_TAG)
    parser.add_argument("--output-tag", default="paper_end_to_end_local_range_poseidon_v1")
    parser.add_argument("--noise-only", action="store_true")
    parser.add_argument("--require-raw-timestamp-contract", action="store_true")
    parser.add_argument("--campaign-manifest", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_.-]+", args.noise_tag) or not re.fullmatch(
        r"[a-z0-9_.-]+", args.output_tag
    ):
        raise ValueError("noise or output tag contains unsupported characters")
    if args.require_raw_timestamp_contract and args.campaign_manifest is None:
        raise ValueError("the raw-timestamp reducer requires the central campaign manifest")
    artifacts = args.artifacts_root.resolve(strict=True)
    output = artifacts / "results" / args.output_tag
    tables = [] if args.noise_only else table_rows(artifacts)
    reference_root, reference_result, reference_manifest = noise_reference(
        artifacts, args.noise_calibration_source,
    )
    if tables:
        table_vit_base = next(row for row in tables if row["model"] == "imagenet_vit_base")
        table_vit_manifest = json.loads((
            artifacts / "logs/conversion_comparison" / VIT_TAG
            / "vit/imagenet_vit_base/manifest.json"
        ).read_text())
        if (
            reference_manifest.get("checkpoint_sha256")
            != table_vit_manifest.get("checkpoint_sha256")
            or reference_manifest.get("evaluation_dataset")
            != table_vit_manifest.get("evaluation_dataset")
        ):
            raise ValueError("noise reference does not match the Table 3 ViT-B checkpoint or dataset")
    else:
        table_vit_base = None
    raw = noise_rows(
        artifacts,
        reference_root,
        noise_tag=args.noise_tag,
        require_raw_timestamp_contract=args.require_raw_timestamp_contract,
    )
    campaign_manifest = None
    if args.campaign_manifest is not None:
        campaign_manifest = args.campaign_manifest.resolve(strict=True)
        validate_noise_campaign_manifest(
            campaign_manifest,
            raw,
            noise_tag=args.noise_tag,
            reference_result=reference_result,
            reference_manifest=reference_manifest,
        )
    summary = noise_summary(raw)
    if tables:
        write_csv(output / "table_results.csv", tables, list(tables[0]))
    write_csv(output / "noise_raw_runs.csv", raw, list(raw[0]))
    write_csv(output / "noise_summary.csv", summary, list(summary[0]))
    reference_ann = reference_result["phases"]["ann"]["metrics"]["accuracy"]
    reference_snn = reference_result["phases"]["snn"]["metrics"]["accuracy"]
    figure_stem = artifacts / "figures" / args.output_tag / "ViT-noise-eval"
    render_figure(
        figure_stem,
        summary,
        dense_accuracy=reference_ann,
        clean_spiking_accuracy=reference_snn,
    )
    runtime_files.atomic_json(output / "summary.json", {
        "state": "complete", "table_models": len(tables), "noise_runs": len(raw),
        "noise_cells": len(summary),
        "table_source_commit": tables[0]["source_commit"] if tables else None,
        "noise_source_commit": reference_manifest["source_commit"],
        "noise_calibration_sha256": reference_result["calibration_sha256"],
        "noise_calibration_source": str(reference_root),
        "table_vit_base_calibration_sha256": (
            table_vit_base["calibration_sha256"] if table_vit_base is not None else None
        ),
        "noise_tag": args.noise_tag,
        "raw_timestamp_contract_required": args.require_raw_timestamp_contract,
        "campaign_manifest": str(campaign_manifest) if campaign_manifest else None,
        "campaign_manifest_sha256": (
            identity.sha256_file(campaign_manifest) if campaign_manifest else None
        ),
        "noise_raw_runs_sha256": identity.sha256_file(output / "noise_raw_runs.csv"),
        "noise_summary_sha256": identity.sha256_file(output / "noise_summary.csv"),
        "figure_pdf_sha256": identity.sha256_file(figure_stem.with_suffix(".pdf")),
        "figure_png_sha256": identity.sha256_file(figure_stem.with_suffix(".png")),
    })


if __name__ == "__main__":
    main()
