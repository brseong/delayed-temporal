#!/usr/bin/env python3
"""Audit the online W&B mirror against the authoritative sigma-margin CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def close(actual: object, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {label}: {actual!r}") from error
    if not math.isclose(value, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(f"{label} mismatch: {value} != {expected}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    with args.raw_csv.open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))
    if len(expected_rows) != 470:
        raise ValueError("authoritative CSV must contain 470 runs")
    expected = {row["run_id"]: row for row in expected_rows}
    if len(expected) != 470:
        raise ValueError("authoritative CSV run names are not unique")

    import wandb

    api = wandb.Api()
    remote_runs = list(api.runs(
        f"{args.entity}/{args.project}", filters={"group": args.group}, per_page=500,
    ))
    by_name = {run.name: run for run in remote_runs}
    if len(remote_runs) != 470 or len(by_name) != 470 or set(by_name) != set(expected):
        raise ValueError(
            f"W&B group must contain the same 470 unique names; got {len(remote_runs)}"
        )
    for run_id, row in expected.items():
        run = by_name[run_id]
        if run.state != "finished":
            raise ValueError(f"W&B run is not finished: {run_id} ({run.state})")
        config = dict(run.config)
        summary = dict(run.summary)
        close(config.get("theta"), float(row["theta"]), f"theta for {run_id}")
        close(
            config.get("time_noise_std_frac"),
            float(row["time_noise_std_frac"]),
            f"r_t for {run_id}",
        )
        close(
            config.get("time_noise_deadline_margin_std"),
            float(row["deadline_margin_std"]),
            f"deadline margin for {run_id}",
        )
        if config.get("precision") != "float64":
            raise ValueError(f"precision mismatch for {run_id}")
        close(summary.get("Final Accuracy"), float(row["accuracy"]), f"accuracy for {run_id}")
        if int(summary.get("Correct", -1)) != int(row["correct"]):
            raise ValueError(f"correct-count mismatch for {run_id}")
        if int(summary.get("Evaluated samples", -1)) != int(row["samples"]):
            raise ValueError(f"sample-count mismatch for {run_id}")
    print("W&B mirror matches all 470 authoritative CSV runs")


if __name__ == "__main__":
    main()
