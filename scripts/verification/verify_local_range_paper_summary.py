#!/usr/bin/env python3
"""Verify the local-range paper campaign reducer's rejection contracts."""

from __future__ import annotations

from collections.abc import Callable
import json
import math
from pathlib import Path
import tempfile

from scripts.analysis.summarize_local_range_paper_campaign import (
    contains_legacy_range_key,
    expected_noise_cells,
    noise_summary,
    validate_calibration,
    validate_metric,
)
from scripts.runtime import identity


def expect_value_error(callback: Callable[[], object]) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("invalid summary evidence was accepted")


# @lat: [[evaluation#Evaluation and Verification#Local-Range Paper Re-evaluation]]
def main() -> None:
    source_commit = "1" * 40
    table = {
        "format_version": 2,
        "metadata": {
            "dtype": "float64",
            "model_options": [
                ["source_commit", source_commit],
                ["output_bounds_version", 4],
                ["vit_calibration_policy_version", 2],
            ],
        },
        "layers": [
            {"module_name": "encoder.0", "tensor_name": "query"},
            {"module_name": "encoder.0", "tensor_name": "key"},
        ],
    }
    with tempfile.TemporaryDirectory(prefix="local-range-summary-") as directory:
        path = Path(directory) / "calibration.json"
        path.write_text(json.dumps(table, sort_keys=True))
        validate_calibration(
            path,
            expected_sha256=identity.sha256_file(path),
            expected_source_commit=source_commit,
            expected_sites=2,
            family="vit",
        )
        expect_value_error(lambda: validate_calibration(
            path,
            expected_sha256="0" * 64,
            expected_source_commit=source_commit,
            expected_sites=2,
            family="vit",
        ))
        table["metadata"]["theta"] = 40
        path.write_text(json.dumps(table, sort_keys=True))
        expect_value_error(lambda: validate_calibration(
            path,
            expected_sha256=identity.sha256_file(path),
            expected_source_commit=source_commit,
            expected_sites=2,
            family="vit",
        ))

    assert contains_legacy_range_key({"nested": [{"attention_theta": 40}]})
    assert contains_legacy_range_key({"model_options": [["theta", 40]]})
    assert contains_legacy_range_key({"command": ["python", "--theta", "40"]})
    assert not contains_legacy_range_key({"range_contract": "operator_local_v1"})
    assert len(expected_noise_cells()) == 21
    validate_metric(0.0, accuracy=True)
    validate_metric(1.0, accuracy=True)
    validate_metric(1.0, accuracy=False)
    for value, accuracy in ((-0.1, True), (1.1, True), (0.0, False), (math.nan, True)):
        expect_value_error(lambda value=value, accuracy=accuracy: validate_metric(
            value, accuracy=accuracy,
        ))

    replicas = [
        {
            "time_noise_std_fraction": 1e-5,
            "deadline_margin_sigma_ratio": 4.0,
            "accuracy": accuracy,
            "events": 100,
            "misses": misses,
            "deadline_events": misses,
            "outputs": 200,
            "underflows": 0,
            "overflows": 1,
        }
        for accuracy, misses in ((0.8, 1), (0.9, 2), (1.0, 3))
    ]
    summary = noise_summary(replicas)
    assert len(summary) == 1
    row = summary[0]
    assert math.isclose(row["accuracy_mean"], 0.9)
    assert row["accuracy_ci95_low"] < row["accuracy_mean"] < row["accuracy_ci95_high"]
    assert math.isclose(row["miss_rate"], 6 / 300)
    assert math.isclose(row["saturation_rate"], 3 / 600)
    print("Local-range paper summary verification passed")


if __name__ == "__main__":
    main()
