#!/usr/bin/env python3
"""Pure-Python verification for independent BSS-2 primitive characterization."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.hardware.brainscales2.primitive_artifacts import (
    load_primitive_observations,
    write_primitive_noise_artifacts,
)
from utils.hardware.brainscales2.backend import calibration_sha256
from utils.hardware.brainscales2.primitive_noise import (
    MockPrimitiveNoiseBackend,
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    default_primitive_coordinates,
    validate_primitive_observation,
)
from scripts.evaluation.brainscales2_primitive_noise import collect_observations


def rejects(action, exceptions=(ValueError, RuntimeError)) -> None:
    try:
        action()
    except exceptions:
        return
    raise AssertionError("invalid primitive observation was accepted")


# @lat: [[hardware#Independent Primitive Noise Verification#Synthetic transfer recovery]]
def verify_synthetic_transfer_recovery() -> None:
    config = PrimitiveNoiseConfig(repeats=32, calibration_repeats=16)
    backend = MockPrimitiveNoiseBackend()
    cases = (
        ("phi-np", "static", "slope_s_per_code", -20.0e-6 / 31.0, 0.12),
        ("phi-nl", "transfer", "tau_effective_s", 4.0e-6, 0.12),
        ("psi-int", "transfer", "gain", 0.98, 0.03),
        ("psi-ne", "transfer", "tau_effective_s", 10.0e-6, 0.18),
    )
    for primitive, stage, key, expected, relative_tolerance in cases:
        observation = backend.collect(primitive, config, stage=stage, quick=False)
        result = validate_primitive_observation(observation, config)
        recovered = torch.tensor(
            [parameters[key] for parameters in result.calibration_parameters]
        ).median()
        assert abs(float(recovered) - expected) <= abs(expected) * relative_tolerance
        assert result.noise["temporal_sigma"]["representative_median"] > 0


# @lat: [[hardware#Independent Primitive Noise Verification#Held-out validation isolation]]
def verify_held_out_validation_isolation() -> None:
    config = PrimitiveNoiseConfig(repeats=16, calibration_repeats=8)
    observation = MockPrimitiveNoiseBackend().collect(
        "psi-int", config, stage="transfer", quick=True
    )
    reference = validate_primitive_observation(observation, config)
    shifted_values = observation.observed.clone()
    shifted_values[config.calibration_repeats :] += 20.0
    shifted = replace(observation, observed=shifted_values)
    result = validate_primitive_observation(shifted, config)
    for changed, original in zip(
        result.calibration_parameters, reference.calibration_parameters
    ):
        assert changed.keys() == original.keys()
        torch.testing.assert_close(
            torch.tensor(list(changed.values())),
            torch.tensor(list(original.values())),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    assert result.validation_refit_parameters != reference.validation_refit_parameters
    assert not result.gates["normalized_rmse"]


# @lat: [[hardware#Independent Primitive Noise Verification#Temporal and fixed-pattern separation]]
def verify_temporal_and_fixed_pattern_separation() -> None:
    coordinates = default_primitive_coordinates(4)
    config = PrimitiveNoiseConfig(
        repeats=8,
        calibration_repeats=4,
        device_count=4,
        physical_coordinates=coordinates,
    )
    codes = torch.tensor([0.0, 15.0, 30.0], dtype=torch.float64)
    offsets = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64) * 1.0e-6
    mean = 25.0e-6 - codes * (20.0e-6 / 31.0)
    observed = (mean[None, :, None] + offsets[None, None, :]).repeat(8, 1, 1)
    delivered = torch.ones_like(observed, dtype=torch.bool)
    observation = PrimitiveObservation(
        primitive="phi-np",
        stage="static",
        output_kind="spike-time-s",
        input_code=codes,
        ideal_variable=codes,
        observed=observed,
        delivered=delivered,
        spike_count=torch.ones_like(observed, dtype=torch.int64),
        saturated=torch.zeros_like(delivered),
        physical_coordinates=coordinates,
    )
    result = validate_primitive_observation(observation, config)
    assert result.noise["temporal_sigma"]["representative_median"] < 1.0e-18
    fixed = result.noise["fixed_pattern_parameters"]["offset_s"]
    assert fixed["standard_deviation"] > 0


# @lat: [[hardware#Independent Primitive Noise Verification#Miss and saturation semantics]]
def verify_miss_and_saturation_semantics() -> None:
    config = PrimitiveNoiseConfig(repeats=16, calibration_repeats=8)
    backend = MockPrimitiveNoiseBackend()
    base = backend.collect("phi-np", config, stage="static", quick=True)
    observed = base.observed.clone()
    delivered = base.delivered.clone()
    count = base.spike_count.clone()
    delivered[:, 0, :] = False
    count[:, 0, :] = 0
    observed[:, 0, :] = float("nan")
    missed = replace(base, observed=observed, delivered=delivered, spike_count=count)
    result = validate_primitive_observation(missed, config)
    assert not result.gates["point_delivery"]
    assert bool(torch.isnan(missed.observed[:, 0]).all())

    invalid = observed.clone()
    invalid[:, 0, :] = config.deadline_s
    rejects(lambda: replace(missed, observed=invalid))

    multiple = replace(base, spike_count=torch.full_like(base.spike_count, 2))
    result = validate_primitive_observation(multiple, config)
    assert not result.gates["overall_delivery"]

    pwm = backend.collect("psi-int", config, stage="transfer", quick=True)
    saturated = pwm.saturated.clone()
    saturated[:, 0, :] = True
    result = validate_primitive_observation(replace(pwm, saturated=saturated), config)
    assert not result.gates["saturation"]

    ne = backend.collect("psi-ne", config, stage="transfer", quick=True)
    spike_count = ne.spike_count.clone()
    spike_count[0, 0, 0] = 1
    result = validate_primitive_observation(replace(ne, spike_count=spike_count), config)
    assert not result.gates["premature_spike"]

    endpoint_base = backend.collect("phi-np", config, stage="static", quick=False)
    endpoint_shifted = endpoint_base.observed.clone()
    endpoint_shifted[:, -1, :] += 10.0e-6
    base_result = validate_primitive_observation(endpoint_base, config)
    endpoint_result = validate_primitive_observation(
        replace(endpoint_base, observed=endpoint_shifted), config
    )
    assert (
        endpoint_result.noise["validation_normalized_rmse_maximum"]
        == base_result.noise["validation_normalized_rmse_maximum"]
    )


# @lat: [[hardware#Independent Primitive Noise Verification#NP stage gate]]
def verify_np_stage_gate() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    static = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="static", quick=True
    )
    validation = validate_primitive_observation(static, config)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        calibration = write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[static],
            validations=[validation],
        )
        assert not calibration["primitives"]["phi-np"]["validated"]
        assert calibration["primitives"]["phi-np"]["diagnostic_only"]
        assert not calibration["validated"]

    dynamic = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="dynamic", quick=True
    )
    failed_dynamic = replace(
        dynamic,
        spike_count=torch.full_like(dynamic.spike_count, 2),
    )
    failed_validation = validate_primitive_observation(failed_dynamic, config)
    with tempfile.TemporaryDirectory() as temporary:
        calibration = write_primitive_noise_artifacts(
            Path(temporary),
            config=config,
            observations=[static, failed_dynamic],
            validations=[validation, failed_validation],
        )
        np_result = calibration["primitives"]["phi-np"]
        assert not np_result["validated"] and np_result["diagnostic_only"]
        assert np_result["transfer_parameters"] is None
        assert np_result["noise"] is None

    original_collect = MockPrimitiveNoiseBackend.collect
    calls: list[tuple[str, str]] = []

    def failing_static(self, primitive, run_config, *, stage="transfer", quick=False):
        calls.append((primitive, stage))
        result = original_collect(
            self, primitive, run_config, stage=stage, quick=quick
        )
        if primitive == "phi-np" and stage == "static":
            return replace(
                result,
                spike_count=torch.full_like(result.spike_count, 2),
            )
        return result

    MockPrimitiveNoiseBackend.collect = failing_static
    try:
        observations, _ = collect_observations(
            SimpleNamespace(backend="mock", primitive="all", quick=True),
            config,
        )
    finally:
        MockPrimitiveNoiseBackend.collect = original_collect
    assert ("phi-np", "dynamic") not in calls
    assert ("phi-nl", "transfer") not in calls
    assert {item.primitive for item in observations} == {
        "phi-np",
        "psi-int",
        "psi-ne",
    }


# @lat: [[hardware#Independent Primitive Noise Verification#Artifact integrity]]
def verify_artifact_integrity() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    observation = MockPrimitiveNoiseBackend().collect(
        "psi-int", config, stage="transfer", quick=True
    )
    validation = validate_primitive_observation(observation, config)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[observation],
            validations=[validation],
        )
        loaded = load_primitive_observations(output, config)
        torch.testing.assert_close(
            loaded[0].observed, observation.observed, equal_nan=True
        )
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["replicas_are_pooled"] is False
        assert manifest["transformer_forward_included"] is False
        chunk = next((output / "raw").glob("*.pt"))
        chunk.write_bytes(chunk.read_bytes() + b"changed")
        rejects(lambda: load_primitive_observations(output, config), (ValueError,))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        calibration_path = root / "hagen.pbin"
        calibration_path.write_bytes(b"pinned calibration")
        hardware_config = replace(config, hagen_calibration_path=calibration_path)
        bad_checksum = replace(
            observation,
            metadata={
                "backend": "hagen-hardware",
                "chip_identifier": ["chip-a"],
                "calibration_sha256": "wrong",
            },
        )
        rejects(
            lambda: write_primitive_noise_artifacts(
                root / "bad-checksum",
                config=hardware_config,
                observations=[bad_checksum],
                validations=[validation],
            ),
            (ValueError,),
        )
        checksum = calibration_sha256(calibration_path)
        first = replace(
            observation,
            metadata={
                "backend": "hagen-hardware",
                "chip_identifier": ["chip-a"],
                "calibration_sha256": checksum,
            },
        )
        second = replace(
            observation,
            metadata={
                "backend": "hagen-hardware",
                "chip_identifier": ["chip-b"],
                "calibration_sha256": checksum,
            },
        )
        rejects(
            lambda: write_primitive_noise_artifacts(
                root / "bad-chip",
                config=hardware_config,
                observations=[first, second],
                validations=[validation, validation],
            ),
            (ValueError,),
        )

    rejects(
        lambda: PrimitiveNoiseConfig(
            device_count=2,
            physical_coordinates=(0, 0),
        )
    )
    rejects(
        lambda: PrimitiveNoiseConfig(
            device_count=2,
            physical_coordinates=(0, 512),
        )
    )


def main() -> None:
    verify_synthetic_transfer_recovery()
    verify_held_out_validation_isolation()
    verify_temporal_and_fixed_pattern_separation()
    verify_miss_and_saturation_semantics()
    verify_np_stage_gate()
    verify_artifact_integrity()
    print("BrainScaleS-2 primitive-noise checks passed")


if __name__ == "__main__":
    main()
