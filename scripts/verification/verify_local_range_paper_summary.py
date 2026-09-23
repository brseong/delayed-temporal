#!/usr/bin/env python3
"""Verify the local-range paper campaign reducer's rejection contracts."""

from __future__ import annotations

from collections.abc import Callable
from argparse import Namespace
import json
import math
from pathlib import Path
import tempfile

from scripts.analysis.summarize_local_range_paper_campaign import (
    contains_legacy_range_key,
    expected_noise_cells,
    noise_summary,
    render_figure,
    validate_calibration,
    validate_metric,
)
from scripts.analysis import summarize_local_range_paper_campaign as summary_module
from scripts.runtime import identity
from scripts.experiments import run_poseidon_local_range_paper_campaign as campaign


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

        text_table = {
            **table,
            "metadata": {
                **table["metadata"],
                "model_options": [
                    ["output_bounds_version", 4],
                    ["text_calibration_policy_version", 1],
                ],
            },
        }
        text_table["metadata"].pop("theta", None)
        path.write_text(json.dumps(text_table, sort_keys=True))
        validate_calibration(
            path,
            expected_sha256=identity.sha256_file(path),
            expected_source_commit=source_commit,
            expected_sites=2,
            family="text",
        )

    with tempfile.TemporaryDirectory(prefix="local-range-reference-") as directory:
        root = Path(directory)
        (root / "result.json").write_text(json.dumps({"state": "complete"}))
        (root / "manifest.json").write_text(json.dumps({
            "model_key": "imagenet_vit_base",
        }))
        original_validate_pipeline = summary_module.validate_pipeline
        summary_module.validate_pipeline = lambda *args, **kwargs: None
        try:
            resolved, _, manifest = summary_module.noise_reference(root, root)
            assert resolved == root.resolve()
            assert manifest["model_key"] == "imagenet_vit_base"
            (root / "manifest.json").write_text(json.dumps({"model_key": "other"}))
            expect_value_error(lambda: summary_module.noise_reference(root, root))
        finally:
            summary_module.validate_pipeline = original_validate_pipeline

    assert contains_legacy_range_key({"nested": [{"attention_theta": 40}]})
    assert contains_legacy_range_key({"model_options": [["theta", 40]]})
    assert contains_legacy_range_key({"command": ["python", "--theta", "40"]})
    assert not contains_legacy_range_key({"range_contract": "operator_local_v1"})
    assert len(expected_noise_cells()) == 21
    original_checkpoint_hash = campaign.checkpoint_hash
    campaign.checkpoint_hash = lambda _: "2" * 64
    try:
        explicit_calibration = Path("/data/calibration-source")
        tasks = campaign.noise_tasks(Namespace(
            source_root=Path("/data/source"),
            expected_commit=source_commit,
            python_bin="python",
            noise_calibration_source=explicit_calibration,
        ))
    finally:
        campaign.checkpoint_hash = original_checkpoint_hash
    assert len(tasks) == 63
    for task in tasks:
        command = task["command"]
        index = command.index("--calibration-source")
        assert command[index + 1] == str(explicit_calibration)
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

    figure_rows = [
        {
            "time_noise_std_fraction": fraction,
            "deadline_margin_sigma_ratio": ratio,
            "accuracy_mean": 0.9,
            "accuracy_ci95_low": 0.89,
            "accuracy_ci95_high": 0.91,
            "miss_rate": 0.0 if ratio >= 8 else 1e-4,
        }
        for fraction, ratio in sorted(expected_noise_cells())
    ]
    with tempfile.TemporaryDirectory(prefix="local-range-figure-") as directory:
        path = Path(directory) / "noise"
        render_figure(
            path,
            figure_rows,
            dense_accuracy=0.91,
            clean_spiking_accuracy=0.90,
        )
        assert path.with_suffix(".pdf").stat().st_size > 0
        assert path.with_suffix(".png").stat().st_size > 0
    print("Local-range paper summary verification passed")


if __name__ == "__main__":
    main()
