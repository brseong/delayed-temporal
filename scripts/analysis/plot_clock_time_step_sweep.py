#!/usr/bin/env python3
"""Verify and plot the completed 500 image clock time step sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


repository_root = Path(__file__).resolve().parents[2]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from scripts.experiments.run_clock_driven_vit import (
    default_evaluation_samples as expected_population,
    default_shards as expected_shards,
    default_tag as expected_tag,
    default_time_steps as expected_time_steps,
)


default_root = Path("/data/delayed-temporal/artifacts/logs/clock_driven") / expected_tag


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)


def _condition_key(time_step: float | None) -> str:
    return "continuous" if time_step is None else f"dt_{time_step:g}"


def load_verified_results(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a complete campaign and reject mixed or incomplete evidence."""

    experiment_path = root / "experiment.json"
    summary_json_path = root / "summary.json"
    summary_csv_path = root / "summary.csv"
    for path in (experiment_path, summary_json_path, summary_csv_path):
        if not path.is_file():
            raise FileNotFoundError(f"required sweep artifact is missing: {path}")

    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("tag") != expected_tag:
        raise ValueError("sweep tag differs")
    if experiment.get("evaluation_population") != expected_population:
        raise ValueError("evaluation population differs")
    if experiment.get("evaluation_shards") != expected_shards:
        raise ValueError("evaluation shard count differs")
    time_steps = tuple(float(value) for value in experiment.get("time_steps", ()))
    if time_steps != expected_time_steps:
        raise ValueError("clock time step grid differs")
    if experiment.get("simulation") != "explicit_sequential_state_updates":
        raise ValueError("execution is not the explicit sequential simulation")

    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    if payload.get("tag") != expected_tag:
        raise ValueError("summary tag differs")
    conditions = payload.get("conditions")
    shard_runs = payload.get("shard_runs")
    if not isinstance(conditions, list) or not isinstance(shard_runs, list):
        raise ValueError("summary structure is incomplete")
    expected_conditions = (None,) + expected_time_steps
    if len(conditions) != len(expected_conditions):
        raise ValueError("summary does not contain every requested condition")
    if len(shard_runs) != len(expected_conditions) * expected_shards:
        raise ValueError("summary does not contain every requested shard")

    source_commit = experiment["source_commit"]
    calibration_sha256 = experiment["calibration_sha256"]
    dataset_path = experiment["evaluation_dataset_path"]
    verified: list[dict[str, Any]] = []
    for expected_step, condition in zip(expected_conditions, conditions):
        actual_step = condition.get("time_step")
        if actual_step is not None:
            actual_step = float(actual_step)
        if not _same_float(actual_step, expected_step):
            raise ValueError("summary condition order or clock time step differs")
        if condition.get("condition") != _condition_key(expected_step):
            raise ValueError("summary condition name differs")
        rows = [
            row for row in shard_runs
            if _same_float(
                None if row.get("time_step") is None else float(row["time_step"]),
                expected_step,
            )
        ]
        rows.sort(key=lambda row: int(row["shard_index"]))
        if [int(row["shard_index"]) for row in rows] != list(range(expected_shards)):
            raise ValueError("condition has a missing or duplicate shard")
        cursor = 0
        correct = 0
        samples = 0
        prediction_digests: list[str] = []
        for row in rows:
            if row.get("success") is not True:
                raise ValueError("condition contains an unsuccessful shard")
            if row.get("source_commit") != source_commit:
                raise ValueError("condition mixes source revisions")
            if row.get("calibration_sha256") != calibration_sha256:
                raise ValueError("condition mixes calibration tables")
            if row.get("evaluation_dataset_path") != dataset_path:
                raise ValueError("condition mixes evaluation datasets")
            if int(row["shard_count"]) != expected_shards:
                raise ValueError("shard count differs")
            start = int(row["shard_start"])
            stop = int(row["shard_stop"])
            row_samples = int(row["samples"])
            row_correct = int(row["correct"])
            if start != cursor or stop - start != row_samples:
                raise ValueError("shard coverage is not contiguous")
            if not 0 <= row_correct <= row_samples:
                raise ValueError("shard correct count is invalid")
            if not math.isclose(
                float(row["accuracy"]), row_correct / row_samples,
                rel_tol=0.0, abs_tol=1.0e-15,
            ):
                raise ValueError("shard accuracy differs from its counts")
            log_path = Path(row["log_path"])
            if not log_path.is_file() or _sha256(log_path) != row["log_sha256"]:
                raise ValueError("shard log identity differs")
            if expected_step is None:
                if row.get("clock_sites") or row.get("clock_updates"):
                    raise ValueError("continuous reference contains clock statistics")
            else:
                if not {"neg_linear_transform", "neg_log_transform"}.issubset(
                    row.get("clock_sites", {})
                ):
                    raise ValueError("clock encoder statistics are incomplete")
                updates = row.get("clock_updates", {})
                if set(updates) != {"encoder", "exponential", "pwm"}:
                    raise ValueError("clock state update statistics are incomplete")
                if any(
                    int(counts[field]) <= 0
                    for counts in updates.values()
                    for field in ("calls", "time_steps", "element_updates")
                ):
                    raise ValueError("clock state update count is not positive")
            cursor = stop
            correct += row_correct
            samples += row_samples
            prediction_digests.append(row["prediction_sha256"])
        if cursor != expected_population or samples != expected_population:
            raise ValueError("condition does not cover exactly 500 images")
        digest = hashlib.sha256("\n".join(prediction_digests).encode("ascii")).hexdigest()
        if digest != condition["ordered_shard_digest_sha256"]:
            raise ValueError("ordered prediction digest differs")
        if int(condition["correct"]) != correct or int(condition["samples"]) != samples:
            raise ValueError("condition counts differ from shard counts")
        accuracy = correct / samples
        if not math.isclose(
            float(condition["accuracy"]), accuracy, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ValueError("condition accuracy differs from shard counts")
        verified.append({
            "condition": condition["condition"],
            "time_step": expected_step,
            "correct": correct,
            "samples": samples,
            "accuracy": accuracy,
        })

    with summary_csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(verified):
        raise ValueError("CSV and JSON condition counts differ")
    for csv_row, result in zip(csv_rows, verified):
        if (
            csv_row["condition"] != result["condition"]
            or int(csv_row["correct"]) != result["correct"]
            or int(csv_row["samples"]) != result["samples"]
            or not math.isclose(
                float(csv_row["accuracy"]), result["accuracy"],
                rel_tol=0.0, abs_tol=1.0e-15,
            )
        ):
            raise ValueError("CSV and JSON summary values differ")
    return experiment, verified


def plot_results(results: list[dict[str, Any]], output_prefix: Path) -> None:
    """Render the verified accuracy curve as PDF and PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    continuous = results[0]["accuracy"] * 100.0
    clock_rows = results[1:]
    steps = [float(row["time_step"]) for row in clock_rows]
    accuracy = [row["accuracy"] * 100.0 for row in clock_rows]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    axis.plot(steps, accuracy, marker="o", color="#2166AC", label="Clock time step")
    axis.axhline(
        continuous,
        color="#4D4D4D",
        linestyle="--",
        linewidth=1.0,
        label="Continuous time",
    )
    axis.set_xlabel("Clock time step")
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_xticks(steps)
    axis.set_xlim(min(steps) - 0.003, max(steps) + 0.003)
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_verification(
    root: Path,
    experiment: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """Write the final machine readable completion evidence."""

    payload = {
        "status": "complete",
        "tag": expected_tag,
        "source_commit": experiment["source_commit"],
        "calibration_sha256": experiment["calibration_sha256"],
        "evaluation_population": expected_population,
        "condition_count": len(results),
        "clock_condition_count": len(results) - 1,
        "shard_run_count": len(results) * expected_shards,
        "summary_json_sha256": _sha256(root / "summary.json"),
        "summary_csv_sha256": _sha256(root / "summary.csv"),
        "experiment_sha256": _sha256(root / "experiment.json"),
        "results": results,
    }
    target = root / "verification.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_prefix = (
        args.output_prefix.resolve()
        if args.output_prefix is not None
        else root / "figures" / "clock_time_step_sweep"
    )
    experiment, results = load_verified_results(root)
    plot_results(results, output_prefix)
    write_verification(root, experiment, results)
    print(f"Verified {len(results)} conditions over 500 images")
    print(f"Figure PDF: {output_prefix.with_suffix('.pdf')}")
    print(f"Figure PNG: {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
