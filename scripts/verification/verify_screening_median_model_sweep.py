#!/usr/bin/env python3
"""Verify screening median selection, cells, merging, and summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.summarize_screening_median_model_sweep import (
    baseline_rows,
    load_completed_cells,
    model_accuracy_rows,
    render,
    summarize,
    write_csv,
)
from scripts.experiments.screening_median_model_sweep import (
    INITIAL_ALPHAS,
    MODELS,
    SEEDS,
    alpha_slug,
    canonical_alphas,
    expected_cells,
    load_screening_median,
    protocol_id,
    scaled_fractions,
)
from scripts.experiments.run_screening_median_model_condition import (
    _aggregate_cct_counts,
)
from scripts.experiments.run_poseidon_screening_median_model_sweep import (
    _assign_cells,
    _verify_smoke_gate,
)
from utils.transformers.models.spiking_cct import CCT7ForImageClassification


def must_reject(callable_) -> None:
    try:
        callable_()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return
    raise AssertionError("invalid screening median input was accepted")


def verify_measurement_and_grid(hardware_summary: Path) -> None:
    pair = load_screening_median(hardware_summary)
    assert pair.phi_np.physical_coordinate == 184
    assert pair.phi_nl.physical_coordinate == 1
    assert pair.phi_np.validation_rt == 0.010838111004060678
    assert pair.phi_nl.validation_rt == 0.024617376541590685
    assert pair.phi_np.encoding_window_duration_s == 86.0e-6
    assert pair.phi_np.observation_deadline_s == 300.0e-6
    assert pair.phi_nl.encoding_window_duration_s == 20.0e-6
    assert pair.phi_nl.observation_deadline_s == 60.0e-6
    linear, logarithmic = scaled_fractions(pair, "0.3")
    assert linear == pair.phi_np.validation_rt * 0.3
    assert logarithmic == pair.phi_nl.validation_rt * 0.3
    assert canonical_alphas(("1.0", "0.010", "0.01")) == ("0.01", "1")
    assert alpha_slug("0.0300") == "0p03"
    for invalid in ("0", "-1", "nan", "inf"):
        must_reject(lambda invalid=invalid: canonical_alphas((invalid,)))

    smoke = expected_cells(phase="smoke", alphas=("1",))
    formal = expected_cells(phase="formal", alphas=INITIAL_ALPHAS)
    assert len(smoke) == len(MODELS) * len(SEEDS) == 9
    assert len(formal) == len(MODELS) * len(SEEDS) * len(INITIAL_ALPHAS) == 54
    assert len({cell.identity for cell in formal}) == len(formal)
    slots = tuple(("local", gpu) for gpu in range(8)) + tuple(
        ("poseidon", gpu) for gpu in range(4)
    )
    assignments = _assign_cells(formal, slots)
    assert len(assignments) == len(formal)
    assert {(host, gpu) for _, host, gpu in assignments} == set(slots)
    assert assignments == _assign_cells(formal, slots)

    counts = _aggregate_cct_counts(
        {
            "linear.data": {
                "events": 10,
                "misses": 1,
                "deadline_events": 2,
                "outputs": 9,
                "output_underflows": 3,
                "output_overflows": 4,
            }
        }
    )
    assert counts["underflows"] == 3 and counts["overflows"] == 4

    cct = CCT7ForImageClassification(converted=True)
    assert cct.encoder.time_noise_vit_first_block_count is None
    preprocessing = json.loads(
        (ROOT / "scripts/configs/vit_timm_preprocessing.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    }.issubset(preprocessing["models"])


def verify_smoke_gate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = SimpleNamespace(output_root=root)
        must_reject(lambda: _verify_smoke_gate(args))
        for cell in expected_cells(phase="smoke", alphas=("1",)):
            result = root / cell.relative_path / "result.json"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "metrics": {
                            "total": 500,
                            "accuracy": 0.5,
                            "physical_counts": {"events": 10, "outputs": 10},
                        },
                    }
                ),
                encoding="utf-8",
            )
        _verify_smoke_gate(args)


def fixture_protocol() -> dict:
    models = {}
    for model in MODELS:
        samples = 10_000 if model == "cct7" else 5_000
        models[model] = {
            "model_key": model,
            "checkpoint_sha256": (model[0] * 64),
            "calibration_sha256": (model[-1] * 64),
            "evaluation_population": {"samples": samples},
            "ann_reference": {
                "accuracy": 0.9,
                "prediction_sha256": "a" * 64,
            },
            "converted_clean": {
                "accuracy": 0.89,
                "prediction_sha256": "b" * 64,
            },
        }
    identity_payload = {
        "contract": "screening_median_model_timing_noise_v1",
        "source_commit": "c" * 40,
        "hardware": {
            "summary_sha256": "d" * 64,
            "phi_np": {"validation_rt": 0.01},
            "phi_nl": {"validation_rt": 0.02},
        },
        "models": models,
        "precision": "float64",
        "deadline_margin_sigma_ratio": 4.0,
        "time_noise_scope": "model_wide",
        "raw_timestamp_contract": "gaussian_raw_timestamp_v1",
        "exponential_difference_internal_noise": "enabled_v1",
    }
    return {
        "schema_version": 1,
        "protocol_id": protocol_id(identity_payload),
        "identity": identity_payload,
        "resources": {"models": models},
    }


def write_cell(root: Path, protocol: dict, model: str, alpha: str, seed: int) -> None:
    samples = 10_000 if model == "cct7" else 5_000
    cell = root / "formal" / model / f"alpha_{alpha_slug(alpha)}" / f"seed_{seed}"
    cell.mkdir(parents=True)
    numeric_alpha = float(alpha)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "model": model,
        "alpha": alpha,
        "seed": seed,
        "evaluation_samples": samples,
        "linear_time_noise_std_fraction": 0.01 * numeric_alpha,
        "log_time_noise_std_fraction": 0.02 * numeric_alpha,
        "deadline_margin_sigma_ratio": 4.0,
        "time_noise_scope": "model_wide",
    }
    accuracy = 0.89 - 0.02 * numeric_alpha - 0.001 * seed
    result = {
        "state": "complete",
        "metrics": {
            "correct": round(accuracy * samples),
            "total": samples,
            "accuracy": round(accuracy * samples) / samples,
            "prediction_sha256": f"{seed + 1:064x}",
            "physical_counts": {
                "events": 100,
                "misses": seed,
                "deadline_events": seed + 1,
                "outputs": 100,
                "underflows": 0,
                "overflows": 0,
            },
        },
    }
    (cell / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (cell / "result.json").write_text(json.dumps(result), encoding="utf-8")


def write_root(root: Path, alphas: tuple[str, ...]) -> dict:
    protocol = fixture_protocol()
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    for model in MODELS:
        for alpha in alphas:
            for seed in SEEDS:
                write_cell(root, protocol, model, alpha, seed)
    return protocol


def verify_extension_and_merge() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        first = temporary_root / "first"
        second = temporary_root / "second"
        write_root(first, ("0.01", "1"))
        write_root(second, ("0.1",))
        protocol, raw, hashes = load_completed_cells((first, second))
        assert len(raw) == 27
        assert len(hashes) == 56
        summary = summarize(protocol, raw, allow_incomplete=False)
        assert len(summary) == 9
        baselines = baseline_rows(protocol)
        assert len(baselines) == 3
        accuracy_table = model_accuracy_rows(protocol, summary)
        assert len(accuracy_table) == 3
        assert all("alpha_1_pooled_miss_rate" in row for row in accuracy_table)
        write_csv(temporary_root / "raw.csv", raw)
        write_csv(temporary_root / "summary.csv", summary)
        render(summary, temporary_root / "figure")
        assert (temporary_root / "figure.png").is_file()
        assert (temporary_root / "figure.pdf").is_file()

        conflict = second / "formal/cct7/alpha_0p1/seed_0/manifest.json"
        payload = json.loads(conflict.read_text(encoding="utf-8"))
        payload["protocol_id"] = "f" * 64
        conflict.write_text(json.dumps(payload), encoding="utf-8")
        must_reject(lambda: load_completed_cells((first, second)))


def verify_protocol_excludes_execution_axes() -> None:
    base = {"source_commit": "a" * 40, "hardware": {}, "models": {}}
    first = protocol_id(base)
    assert first == protocol_id(dict(base))
    for key in ("alphas", "gpus", "runtime_root", "created_at", "requests"):
        must_reject(lambda key=key: protocol_id({**base, key: []}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--hardware-summary",
        type=Path,
        default=(
            ROOT
            / "artifacts/brainscales2-primitives/"
            "20260924T_best_median_screen_summary.json"
        ),
    )
    args = parser.parse_args()
    verify_measurement_and_grid(args.hardware_summary)
    verify_smoke_gate()
    verify_protocol_excludes_execution_axes()
    verify_extension_and_merge()
    print("Screening median model timing-noise sweep verification passed.")


if __name__ == "__main__":
    main()
