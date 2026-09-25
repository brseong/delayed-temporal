"""Verify the fixed ViT-B BrainScaleS-2 depth grid and summary contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = Path(os.environ.get("DELAYED_TEMPORAL_ARTIFACTS_ROOT", ROOT / "artifacts"))

from scripts.analysis.summarize_vit_bss2_depth import (
    load_clean_reference,
    load_runs,
    load_scaled_sparse_runs,
    parse_max_first_block_counts,
    render,
    render_scaled_sparse,
    scaled_run_id,
    selected_cells,
    summarize,
    summarize_scaled_sparse,
    validate_stopping_decision,
    write_csv,
)
from scripts.experiments.run_vit_bss2_depth_campaign import (
    expected_cells,
    run_id,
)
from scripts.experiments.run_vit_bss2_depth_condition import (
    hardware_conditions,
    parse_scoped_counts,
    validate_condition_arguments,
    validate_execution_arguments,
)


def must_reject(callable_) -> None:
    try:
        callable_()
    except (KeyError, TypeError, ValueError):
        return
    raise AssertionError("invalid depth-campaign input was accepted")


def verify_fixed_conditions() -> None:
    cells = expected_cells()
    assert len(cells) == len(set(cells)) == 73
    assert cells[0] == ("clean", 0, 0)
    assert sum(condition != "clean" for condition, _, _ in cells) == 72
    conditions = hardware_conditions(
        ARTIFACTS / "brainscales2-primitives/20260924T_best_median_screen_summary.json"
    )
    assert conditions["screening-selected-coordinate"]["linear_time_std_fraction"] == 0.008752792479355493
    assert conditions["screening-selected-coordinate"]["log_time_std_fraction"] == 0.015681047682163635
    assert conditions["screening-selected-coordinate"]["interpretation"] == (
        "screening calibration-selected coordinate remeasurement"
    )
    assert conditions["screening-median"]["linear_time_std_fraction"] == 0.010838111004060678
    assert conditions["screening-median"]["log_time_std_fraction"] == 0.024617376541590685
    assert conditions["screening-median"]["phi_np_encoding_window_s"] == (
        5.0e-6,
        91.0e-6,
    )
    assert conditions["screening-median"]["phi_np_observation_deadline_s"] == 300.0e-6
    assert conditions["screening-median"]["phi_nl_encoding_window_s"] == (
        5.0e-6,
        25.0e-6,
    )
    assert conditions["screening-median"]["phi_nl_observation_deadline_s"] == 60.0e-6
    validate_condition_arguments(
        SimpleNamespace(
            condition="clean",
            first_block_count=0,
            seed=0,
            evaluation_samples=500,
            measured_noise_scale=1.0,
        )
    )
    validate_condition_arguments(
        SimpleNamespace(
            condition="screening-median",
            first_block_count=12,
            seed=2,
            evaluation_samples=5000,
            measured_noise_scale=0.1,
        )
    )
    for candidate in (
        SimpleNamespace(
            condition="clean",
            first_block_count=1,
            seed=0,
            evaluation_samples=500,
            measured_noise_scale=1.0,
        ),
        SimpleNamespace(
            condition="screening-selected-coordinate",
            first_block_count=0,
            seed=0,
            evaluation_samples=500,
            measured_noise_scale=1.0,
        ),
        SimpleNamespace(
            condition="screening-median",
            first_block_count=13,
            seed=0,
            evaluation_samples=500,
            measured_noise_scale=1.0,
        ),
        SimpleNamespace(
            condition="screening-median",
            first_block_count=1,
            seed=0,
            evaluation_samples=501,
            measured_noise_scale=1.0,
        ),
        SimpleNamespace(
            condition="screening-median",
            first_block_count=1,
            seed=0,
            evaluation_samples=500,
            measured_noise_scale=0.0,
        ),
        SimpleNamespace(
            condition="clean",
            first_block_count=0,
            seed=0,
            evaluation_samples=500,
            measured_noise_scale=0.1,
        ),
    ):
        must_reject(lambda candidate=candidate: validate_condition_arguments(candidate))


def gaussian_line(site: str, events: int = 10) -> str:
    return (
        f"Gaussian[{site}] events={events}, misses=1 (rate=0.1), "
        "deadline_events=0 (rate=0), deadline_ulp_min=1e-16, "
        "deadline_ulp_max=1e-16, outputs=10, underflows=0 (rate=0), "
        "overflows=0 (rate=0)"
    )


def verify_scoped_count_gate() -> None:
    log = "\n".join(
        gaussian_line(f"vit.encoder.block.{block}/{site}")
        for block in range(2)
        for site in ("linear.data", "layernorm.log_sigma")
    )
    parsed = parse_scoped_counts(log, 2)
    assert parsed["active_blocks"] == [0, 1]
    assert parsed["events"] == 40 and parsed["misses"] == 4
    assert parse_scoped_counts("", 0)["site_count"] == 0
    must_reject(lambda: parse_scoped_counts(log, 1))
    must_reject(
        lambda: parse_scoped_counts(
            gaussian_line("linear.data") + "\n" + gaussian_line("layernorm.log_sigma"),
            1,
        )
    )
    must_reject(
        lambda: parse_scoped_counts(
            gaussian_line("vit.encoder.block.0/linear.data"), 1
        )
    )


def verify_execution_gate() -> None:
    validate_execution_arguments(
        SimpleNamespace(execution_mode="local", gpu=1),
        hostname="baekryun-cuda129",
        environment={},
    )
    validate_execution_arguments(
        SimpleNamespace(execution_mode="slurm", gpu=None),
        hostname="gpu-node",
        environment={"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "0"},
    )
    must_reject(
        lambda: validate_execution_arguments(
            SimpleNamespace(execution_mode="local", gpu=None),
            hostname="baekryun-cuda129",
            environment={},
        )
    )
    must_reject(
        lambda: validate_execution_arguments(
            SimpleNamespace(execution_mode="slurm", gpu=None),
            hostname="gpu-node",
            environment={"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "0,1"},
        )
    )


def write_fixture(root: Path) -> None:
    for condition, first_block_count, seed in expected_cells():
        name = run_id(condition, first_block_count, seed)
        run_root = root / "runs" / name
        run_root.mkdir(parents=True)
        manifest = {
            "condition": condition,
            "first_block_count": first_block_count,
            "seed": seed,
            "evaluation_samples": 500,
            "source_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "calibration_sha256": "c" * 64,
            "evaluation_dataset": {
                "fingerprint": "evaluation",
                "selected_fingerprint": "evaluation-prefix-500",
            },
            "hardware_summary_sha256": "d" * 64,
            "linear_time_std_fraction": 0.0 if condition == "clean" else 0.01,
            "log_time_std_fraction": 0.0 if condition == "clean" else 0.02,
        }
        counts = {
            "active_blocks": list(range(first_block_count)),
            "events": 100 * first_block_count,
            "misses": first_block_count,
            "miss_rate": 0.01 if first_block_count else 0.0,
            "site_count": 2 * first_block_count,
        }
        accuracy = 0.86 - 0.01 * first_block_count - 0.001 * seed
        result = {
            "state": "complete",
            "metrics": {
                "total": 500,
                "correct": round(accuracy * 500),
                "accuracy": accuracy,
                "prediction_sha256": f"{first_block_count + seed:064x}"[-64:],
                "gaussian_counts": counts,
            },
        }
        (run_root / "manifest.json").write_text(json.dumps(manifest))
        (run_root / "result.json").write_text(json.dumps(result))


def write_scaled_sparse_fixture(root: Path) -> Path:
    scales = (0.03, 0.1)
    depths = (1, 2)
    for scale in scales:
        for first_block_count in depths:
            for seed in range(3):
                name = scaled_run_id(
                    "screening-median", scale, first_block_count, seed
                )
                run_root = root / "runs" / name
                run_root.mkdir(parents=True)
                manifest = {
                    "condition": "screening-median",
                    "measured_noise_scale": scale,
                    "first_block_count": first_block_count,
                    "seed": seed,
                    "evaluation_samples": 5000,
                    "source_commit": "a" * 40,
                    "checkpoint_sha256": "b" * 64,
                    "calibration_sha256": "c" * 64,
                    "evaluation_dataset": {
                        "fingerprint": "evaluation",
                        "selected_fingerprint": "evaluation-prefix-5000",
                    },
                    "hardware_summary_sha256": "d" * 64,
                    "measured_condition": {
                        "linear_time_std_fraction": 0.01,
                        "log_time_std_fraction": 0.02,
                    },
                    "linear_time_std_fraction": 0.01 * scale,
                    "log_time_std_fraction": 0.02 * scale,
                }
                accuracy = 0.86 - scale * first_block_count - 0.001 * seed
                result = {
                    "state": "complete",
                    "metrics": {
                        "total": 5000,
                        "correct": round(accuracy * 5000),
                        "accuracy": accuracy,
                        "prediction_sha256": f"{first_block_count + seed:064x}"[-64:],
                        "gaussian_counts": {
                            "active_blocks": list(range(first_block_count)),
                            "events": 100 * first_block_count,
                            "misses": first_block_count,
                            "miss_rate": 0.01,
                            "site_count": 26 * first_block_count,
                        },
                    },
                }
                (run_root / "manifest.json").write_text(json.dumps(manifest))
                (run_root / "result.json").write_text(json.dumps(result))
    clean = root / "clean_reference.json"
    clean.write_text(
        json.dumps(
            {
                "state": "complete",
                "phases": {
                    "snn": {
                        "metrics": {
                            "total": 5000,
                            "correct": 4300,
                            "accuracy": 0.86,
                            "prediction_sha256": "e" * 64,
                        }
                    }
                },
            }
        )
    )
    return clean


def verify_summary_gate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_fixture(root)
        raw, hashes = load_runs(root, "pilot")
        summary = summarize(raw)
        assert len(raw) == 73 and len(hashes) == 146 and len(summary) == 26
        assert all(row["replicas"] == 1 for row in summary if row["first_block_count"] == 0)
        assert all(row["replicas"] == 3 for row in summary if row["first_block_count"] > 0)
        write_csv(root / "depth_noise_raw.csv", raw, tuple(raw[0]))
        write_csv(root / "depth_noise_summary.csv", summary, tuple(summary[0]))
        render(summary, root / "depth_noise_accuracy")
        assert (root / "depth_noise_accuracy.pdf").is_file()
        assert (root / "depth_noise_accuracy.png").is_file()

        scaled_root = root / "scaled"
        clean_path = write_scaled_sparse_fixture(scaled_root)
        scaled_raw, scaled_hashes = load_scaled_sparse_runs(
            scaled_root,
            phase="formal",
            condition="screening-median",
            measured_noise_scales=(0.03, 0.1),
            first_block_counts=(1, 2),
        )
        clean_metrics = load_clean_reference(clean_path, expected_samples=5000)
        scaled_summary = summarize_scaled_sparse(
            scaled_raw, clean_metrics=clean_metrics
        )
        assert len(scaled_raw) == 12 and len(scaled_hashes) == 24
        assert len(scaled_summary) == 6
        assert all(
            row["accuracy_mean"] == 0.86
            for row in scaled_summary
            if row["first_block_count"] == 0
        )
        render_scaled_sparse(scaled_summary, scaled_root / "scaled_accuracy")
        assert (scaled_root / "scaled_accuracy.pdf").is_file()
        assert (scaled_root / "scaled_accuracy.png").is_file()
        must_reject(
            lambda: load_scaled_sparse_runs(
                scaled_root,
                phase="formal",
                condition="screening-median",
                measured_noise_scales=(0.03, 0.2),
                first_block_counts=(1, 2),
            )
        )

        limits = parse_max_first_block_counts(
            ["screening-selected-coordinate=1", "screening-median=1"]
        )
        assert limits == {
            "screening-selected-coordinate": 1,
            "screening-median": 1,
        }
        assert len(selected_cells(limits)) == 7
        truncated, truncated_hashes = load_runs(root, "pilot", limits)
        truncated_summary = summarize(truncated)
        assert len(truncated) == 7
        assert len(truncated_hashes) == 14
        assert len(truncated_summary) == 4
        decision = validate_stopping_decision(
            root,
            max_first_block_counts=limits,
            accuracy_threshold=0.9,
        )
        assert decision["conditions"]["screening-selected-coordinate"][
            "max_first_block_count"
        ] == 1
        render(truncated_summary, root / "depth_noise_accuracy_truncated")
        assert (root / "depth_noise_accuracy_truncated.pdf").is_file()
        must_reject(
            lambda: parse_max_first_block_counts(["screening-selected-coordinate=1"])
        )
        must_reject(
            lambda: validate_stopping_decision(
                root,
                max_first_block_counts=limits,
                accuracy_threshold=0.1,
            )
        )

        wrong = root / "runs" / run_id("screening-median", 3, 1) / "manifest.json"
        payload = json.loads(wrong.read_text())
        payload["evaluation_dataset"]["fingerprint"] = "different"
        wrong.write_text(json.dumps(payload))
        must_reject(lambda: load_runs(root, "pilot"))


# @lat: [[evaluation#Evaluation and Verification#ViT-B Cumulative Encoder Block Timing Noise#Verification#Campaign artifacts]]
def main() -> None:
    verify_fixed_conditions()
    verify_scoped_count_gate()
    verify_execution_gate()
    verify_summary_gate()
    print("ViT-B BrainScaleS-2 depth campaign verification passed.")


if __name__ == "__main__":
    main()
