#!/usr/bin/env python3
"""Pure-Python verification for independent BSS-2 primitive characterization."""

from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import inspect
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
from utils.hardware.brainscales2.primitive_backend import PrimitiveHardwareBackend
from utils.hardware.brainscales2.primitive_correlation_worker import (
    _decode_correlation,
    _point_orders,
    _restore_point_order,
)
import utils.hardware.brainscales2.primitive_backend as primitive_backend_module
import utils.hardware.brainscales2.primitive_noise as primitive_noise_module
from utils.hardware.brainscales2.primitive_noise import (
    MockPrimitiveNoiseBackend,
    PRIMITIVES,
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    default_primitive_coordinates,
    validate_primitive_observation,
)
from utils.hardware.brainscales2.primitive_optimization import (
    build_encoder_operating_point_candidates,
    parse_current_stop_pair,
    parse_precharge_pair,
    score_encoder_operating_point,
)
from scripts.evaluation.brainscales2_primitive_noise import (
    build_parser,
    collect_observations,
    make_config,
    parse_reset_code_table,
    run,
)


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
        ("psi-ed", "transfer", "tau_effective_s", 10.0e-6, 0.18),
    )
    for primitive, stage, key, expected, relative_tolerance in cases:
        observation = backend.collect(primitive, config, stage=stage, quick=False)
        result = validate_primitive_observation(observation, config)
        recovered = torch.tensor(
            [parameters[key] for parameters in result.calibration_parameters]
        ).median()
        assert abs(float(recovered) - expected) <= abs(expected) * relative_tolerance
        if primitive == "phi-nl":
            lower_bound = torch.tensor(
                [
                    parameters["lower_bound_code"]
                    for parameters in result.calibration_parameters
                ]
            ).median()
            assert abs(float(lower_bound) + 8.0) <= 1.0
        assert result.noise["temporal_sigma"]["representative_median"] > 0
        assert len(result.moment_rows) == (
            2 * observation.point_count * observation.device_count
        )
        assert all("mean" in row and "variance" in row for row in result.moment_rows)
        assert ("monotonicity" in result.gates) == (primitive == "phi-np")
        assert "rank_monotonicity_minimum" in result.noise
        assert "adjacent_order_fraction_minimum" in result.noise
        if primitive == "phi-np":
            assert result.noise["monotonicity_assessment"] == {
                "acceptance_metric": "rank_monotonicity",
                "diagnostic_metric": "adjacent_order_fraction",
                "rationale": (
                    "Temporal jitter can reverse neighboring measured means without "
                    "breaking the global transfer direction."
                ),
            }


# @lat: [[hardware#Independent Primitive Noise Verification#Held-out validation isolation]]
def verify_held_out_validation_isolation() -> None:
    config = PrimitiveNoiseConfig(repeats=16, calibration_repeats=8)
    observation = MockPrimitiveNoiseBackend().collect(
        "psi-int", config, stage="transfer", quick=True
    )
    clean_values = (
        0.4
        + 0.98
        * observation.ideal_variable.reshape(1, -1, 1).expand_as(
            observation.observed
        )
    )
    observation = replace(
        observation,
        observed=clean_values,
        delivered=torch.ones_like(observation.delivered),
        saturated=torch.zeros_like(observation.saturated),
    )
    reference = validate_primitive_observation(observation, config)
    balanced_values = observation.observed.clone()
    balanced_offsets = torch.tensor(
        [20.0, -20.0] * (config.validation_repeats // 2),
        dtype=balanced_values.dtype,
    ).reshape(-1, 1, 1)
    balanced_values[config.calibration_repeats :] += balanced_offsets
    balanced = validate_primitive_observation(
        replace(observation, observed=balanced_values), config
    )
    torch.testing.assert_close(
        torch.tensor(
            balanced.noise["validation_normalized_rmse_maximum"]
        ),
        torch.tensor(
            reference.noise["validation_normalized_rmse_maximum"]
        ),
        rtol=1.0e-6,
        atol=1.0e-12,
    )
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


# @lat: [[hardware#Independent Primitive Noise Verification#Validated NP placement]]
def verify_validated_np_placement() -> None:
    coordinates = default_primitive_coordinates()
    assert coordinates == (
        3,
        11,
        13,
        1,
        134,
        141,
        128,
        130,
        273,
        261,
        262,
        274,
        390,
        386,
        388,
        392,
    )
    counts = [
        sum(base <= value < base + 128 for value in coordinates)
        for base in (0, 128, 256, 384)
    ]
    assert counts == [4, 4, 4, 4]
    assert len(coordinates) == len(set(coordinates))


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
    assert not result.gates["point_observed"]
    assert bool(torch.isnan(missed.observed[:, 0]).all())

    invalid = observed.clone()
    invalid[:, 0, :] = config.deadline_s
    rejects(lambda: replace(missed, observed=invalid))

    sparse_observed = base.observed.clone()
    sparse_delivered = base.delivered.clone()
    sparse_count = base.spike_count.clone()
    sparse_observed[1:config.calibration_repeats] = float("nan")
    sparse_observed[config.calibration_repeats + 1 :] = float("nan")
    sparse_delivered[1:config.calibration_repeats] = False
    sparse_delivered[config.calibration_repeats + 1 :] = False
    sparse_count[1:config.calibration_repeats] = 0
    sparse_count[config.calibration_repeats + 1 :] = 0
    sparse = replace(
        base,
        observed=sparse_observed,
        delivered=sparse_delivered,
        spike_count=sparse_count,
    )
    sparse_result = validate_primitive_observation(sparse, config)
    assert sparse_result.gates["point_observed"]
    assert sparse_result.gates["multiple_spike_rate"]
    assert sparse_result.noise["miss_rate"] > 0.5

    multiple_count = base.spike_count.clone()
    multiple_count[0, 0, 0] = 2
    multiple = replace(base, spike_count=multiple_count)
    base_result = validate_primitive_observation(base, config)
    result = validate_primitive_observation(multiple, config)
    assert not result.gates["multiple_spike_rate"]
    assert result.noise["multiple_spike_rate"] > 0.0
    assert result.noise["maximum_spike_count"] == 2
    assert result.noise["miss_rate"] == base_result.noise["miss_rate"]
    assert result.noise["exactly_one_spike_rate"] < 1.0
    tolerant = validate_primitive_observation(
        multiple,
        replace(config, multiple_spike_rate_limit=0.002),
    )
    assert tolerant.gates["multiple_spike_rate"]

    one_deadline_miss_observed = base.observed.clone()
    one_deadline_miss_delivered = base.delivered.clone()
    one_deadline_miss_count = base.spike_count.clone()
    one_deadline_miss_observed[0, 0, 0] = float("nan")
    one_deadline_miss_delivered[0, 0, 0] = False
    one_deadline_miss_count[0, 0, 0] = 0
    one_deadline_miss = replace(
        base,
        observed=one_deadline_miss_observed,
        delivered=one_deadline_miss_delivered,
        spike_count=one_deadline_miss_count,
    )
    within_limit = validate_primitive_observation(one_deadline_miss, config)
    assert within_limit.gates["deadline_miss_rate"]
    excessive_observed = base.observed.clone()
    excessive_delivered = base.delivered.clone()
    excessive_count = base.spike_count.clone()
    excessive_observed[:8, 0, 0] = float("nan")
    excessive_delivered[:8, 0, 0] = False
    excessive_count[:8, 0, 0] = 0
    excessive = validate_primitive_observation(
        replace(
            base,
            observed=excessive_observed,
            delivered=excessive_delivered,
            spike_count=excessive_count,
        ),
        config,
    )
    assert not excessive.gates["deadline_miss_rate"]

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
    torch.testing.assert_close(
        torch.tensor(
            endpoint_result.noise["validation_normalized_rmse_maximum"]
        ),
        torch.tensor(base_result.noise["validation_normalized_rmse_maximum"]),
        rtol=1.0e-6,
        atol=1.0e-12,
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Rank monotonicity gate]]
def verify_rank_monotonicity_gate() -> None:
    values = torch.arange(31, dtype=torch.float64)
    for left in (2, 7, 12, 17, 22):
        values[left], values[left + 1] = (
            values[left + 1].clone(),
            values[left].clone(),
        )
    assert primitive_noise_module._monotonicity(values, 1.0) < 0.95
    assert primitive_noise_module._rank_monotonicity(values, 1.0) >= 0.95

# @lat: [[hardware#Independent Primitive Noise Verification#Segmented PyNN recording decode]]
def verify_segmented_pynn_recording_decode() -> None:
    def train(
        values: list[float], source_index: int | None = None
    ) -> SimpleNamespace:
        annotations = (
            {} if source_index is None else {"source_index": source_index}
        )
        return SimpleNamespace(magnitude=values, annotations=annotations)

    segmented = [
        train([], 0),
        train([], 1),
        train([0.0052, 0.0061], 0),
        train([0.0054], 1),
        train([], 0),
        train([], 1),
        train([], 0),
        train([], 1),
        train([0.0653], 0),
        train([0.0655, 0.0660], 1),
        train([], 0),
        train([], 1),
    ]
    first, count = PrimitiveHardwareBackend._decode_pynn_spikes(
        segmented,
        repeats=2,
        devices=2,
        window_ms=0.060,
    )
    torch.testing.assert_close(
        first,
        torch.tensor(
            [[5.2e-6, 5.4e-6], [5.3e-6, 5.5e-6]], dtype=torch.float64
        ),
    )
    torch.testing.assert_close(
        count,
        torch.tensor([[2, 1], [1, 2]], dtype=torch.int64),
    )

    unsegmented, unsegmented_count = PrimitiveHardwareBackend._decode_pynn_spikes(
        [train([0.005]), train([0.006])],
        repeats=1,
        devices=2,
        window_ms=0.060,
    )
    torch.testing.assert_close(
        unsegmented,
        torch.tensor([[5.0e-6, 6.0e-6]], dtype=torch.float64),
    )
    assert bool((unsegmented_count == 1).all())

    rejects(
        lambda: PrimitiveHardwareBackend._decode_pynn_spikes(
            [train([], 0), train([], 0)],
            repeats=1,
            devices=2,
            window_ms=0.060,
        )
    )
    rejects(
        lambda: PrimitiveHardwareBackend._decode_pynn_spikes(
            [train([]), train([]), train([]), train([])],
            repeats=1,
            devices=2,
            window_ms=0.060,
        )
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Dynamic recording selection]]
def verify_dynamic_recording_selection() -> None:
    class Population:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def record(self, variable: str, *, device: str | None = None) -> None:
            self.calls.append((variable, device))

    static = Population()
    PrimitiveHardwareBackend._record_pynn_population(static, membrane=False)
    assert static.calls == [("spikes", None)]

    dynamic = Population()
    PrimitiveHardwareBackend._record_pynn_population(dynamic, membrane=True)
    assert dynamic.calls == [("spikes", None), ("v", "cadc")]


# @lat: [[hardware#Independent Primitive Noise Verification#Segmented dense recording decode]]
def verify_segmented_dense_recording_decode() -> None:
    def signal(
        source_id: int, times: list[float], values: list[float]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            magnitude=torch.tensor(values, dtype=torch.float64).reshape(-1, 1),
            times=SimpleNamespace(magnitude=times),
            annotations={"source_ids": [source_id]},
        )

    segment = SimpleNamespace(
        irregularlysampledsignals=[
            signal(0, [0.001, 0.003], [10.0, 30.0]),
            signal(1, [0.002, 0.004], [20.0, 40.0]),
            signal(0, [0.010], [100.0]),
            signal(1, [0.011], [110.0]),
        ],
        analogsignals=[],
    )
    selected = PrimitiveHardwareBackend._decode_pynn_precharge(
        segment,
        devices=2,
        sample_ms=0.0045,
    )
    torch.testing.assert_close(
        selected, torch.tensor([30.0, 40.0], dtype=torch.float64)
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Sparse precharge evidence]]
def verify_sparse_precharge_evidence() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    observation = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="dynamic", quick=True
    )
    sparse = torch.full_like(observation.precharge_cadc, torch.nan)
    sparse[0] = observation.precharge_cadc[0]
    validation = validate_primitive_observation(
        replace(observation, precharge_cadc=sparse), config
    )
    assert validation.gates["precharge_recorded"]
    assert "precharge_rank_monotonicity_minimum" in validation.noise

    sparse[:, :, 0] = torch.nan
    validation = validate_primitive_observation(
        replace(observation, precharge_cadc=sparse), config
    )
    assert not validation.gates["precharge_recorded"]


# @lat: [[hardware#Independent Primitive Noise Verification#Installed component resolution]]
def verify_installed_component_resolution() -> None:
    spike_source = object()
    static_synapse = object()
    pynn = SimpleNamespace(
        cells=SimpleNamespace(SpikeSourceArray=spike_source),
        synapses=SimpleNamespace(StaticSynapse=static_synapse),
    )
    assert (
        PrimitiveHardwareBackend._pynn_component(
            pynn, "SpikeSourceArray", "cells"
        )
        is spike_source
    )
    assert (
        PrimitiveHardwareBackend._pynn_component(
            pynn, "StaticSynapse", "synapses"
        )
        is static_synapse
    )
    rejects(
        lambda: PrimitiveHardwareBackend._pynn_component(
            SimpleNamespace(), "SpikeSourceArray", "cells"
        )
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Correlation recording decode]]
def verify_correlation_recording_decode() -> None:
    recording = [
        [
            SimpleNamespace(data=torch.tensor([150, 151])),
            SimpleNamespace(data=torch.tensor([145, 147])),
            SimpleNamespace(data=torch.tensor([140, 143])),
        ]
    ]
    decoded = _decode_correlation(recording, periods=3, devices=2)
    torch.testing.assert_close(
        decoded,
        torch.tensor(
            [[150.0, 151.0], [145.0, 147.0], [140.0, 143.0]],
            dtype=torch.float64,
        ),
    )
    rejects(
        lambda: _decode_correlation(recording, periods=2, devices=2)
    )
    rejects(
        lambda: _decode_correlation(recording, periods=3, devices=3)
    )


# @lat: [[hardware#Independent Primitive Noise Verification#Correlation worker boundary]]
def verify_correlation_worker_boundary() -> None:
    orders = _point_orders(repeats=3, point_count=5, seed=7, trial_start=11)
    assert orders == _point_orders(
        repeats=3, point_count=5, seed=7, trial_start=11
    )
    assert orders != _point_orders(
        repeats=3, point_count=5, seed=7, trial_start=12
    )
    canonical = torch.arange(30, dtype=torch.float64).reshape(3, 5, 2)
    scheduled = torch.empty_like(canonical)
    for trial, order in enumerate(orders):
        for acquisition_point, point in enumerate(order):
            scheduled[trial, acquisition_point] = canonical[trial, point]
    torch.testing.assert_close(
        _restore_point_order(scheduled, orders), canonical
    )
    rejects(
        lambda: _restore_point_order(scheduled, [[0, 1, 2, 3, 3]] * 3)
    )

    calls: list[tuple[int, int]] = []

    def fake_process(config, *, differences, repeats, trial_start):
        calls.append((trial_start, repeats))
        shape = (repeats, differences.numel(), config.device_count)
        response = 5.0 + 20.0 * torch.exp(differences / 10.0e-6)
        quiet = torch.full(shape, 180.0, dtype=torch.float64)
        stimulated = quiet - response.reshape(1, -1, 1)
        first = torch.full(shape, 35.0e-6, dtype=torch.float64)
        count = torch.ones(shape, dtype=torch.int64)
        if trial_start == 0:
            first[0, 0, 0] = config.deadline_s + 1.0e-6
            count[0, 1, 0] = 2
        return {
            "quiet": quiet,
            "stimulated": stimulated,
            "first": first,
            "count": count,
            "metadata": {"chip_identifier": ["synthetic-chip"]},
        }

    original_probe = primitive_backend_module.probe_primitive_capabilities
    with tempfile.TemporaryDirectory() as temporary:
        calibration = Path(temporary) / "correlation.pkl"
        calibration.write_bytes(b"correlation calibration")
        config = PrimitiveNoiseConfig(
            repeats=8,
            calibration_repeats=4,
            device_count=2,
            physical_coordinates=(0, 1),
            correlation_calibration_path=calibration,
            psi_ed_chunk_repeats=3,
        )
        backend = PrimitiveHardwareBackend()
        backend._run_psi_ed_process = fake_process  # type: ignore[method-assign]
        primitive_backend_module.probe_primitive_capabilities = lambda: {
            "psi-ed": True,
            "pynn_version": "synthetic",
        }
        try:
            observation = backend.collect("psi-ed", config, quick=True)
        finally:
            primitive_backend_module.probe_primitive_capabilities = original_probe
    assert calls == [(0, 3), (3, 3), (6, 2)]
    assert tuple(observation.observed.shape) == (8, 3, 2)
    assert int((~observation.delivered).sum()) == 1
    assert int((observation.spike_count > 1).sum()) == 1
    assert int(torch.isfinite(observation.first_spike_time_s).sum()) == 48
    finite_first = observation.first_spike_time_s[
        torch.isfinite(observation.first_spike_time_s)
    ]
    assert float(finite_first.max()) > config.deadline_s
    assert observation.metadata["multiple_spike_handling"] == "first-spike-time"
    assert observation.metadata["deadline_applies"] is True


# @lat: [[hardware#Independent Primitive Noise Verification#Nonlinear drive configuration]]
def verify_nonlinear_drive_configuration() -> None:
    config = PrimitiveNoiseConfig()
    assert config.exponential_input_weight == 63
    assert config.exponential_input_fan_in == 8
    assert config.to_manifest_dict()["exponential_input_fan_in"] == 8
    assert config.pynn_chunk_repeats == 8
    assert config.to_manifest_dict()["pynn_chunk_repeats"] == 8
    assert config.pynn_process_repeats == 32
    assert config.to_manifest_dict()["pynn_process_repeats"] == 32
    assert config.hagen_num_sends == 1024
    assert config.hagen_candidate_count == 128
    assert config.hagen_chunk_repeats == 16
    assert config.psi_ed_chunk_repeats == 32
    assert config.deadline_miss_rate_limit == 0.01
    manifest = config.to_manifest_dict()
    assert manifest["hagen_output_coordinates"] is None
    assert len(manifest["hagen_candidate_output_indices"]) == 128
    rejects(lambda: PrimitiveNoiseConfig(exponential_input_fan_in=0))
    rejects(lambda: PrimitiveNoiseConfig(pynn_chunk_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(pynn_process_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(precharge_input_fan_in=0))
    rejects(lambda: PrimitiveNoiseConfig(pynn_worker_timeout_s=0))
    rejects(lambda: PrimitiveNoiseConfig(reset_code_table=(300,) * 31))
    rejects(lambda: PrimitiveNoiseConfig(reset_code_table=(1023,) * 32))
    lookup = tuple(range(32))
    lookup_config = PrimitiveNoiseConfig(reset_code_table=lookup)
    assert PrimitiveHardwareBackend._reset_code(17, lookup_config) == 17
    assert lookup_config.to_manifest_dict()["reset_code_table"] == lookup
    vector_lookup = tuple(tuple(code for _ in range(16)) for code in range(32))
    vector_config = PrimitiveNoiseConfig(reset_code_table=vector_lookup)
    assert PrimitiveHardwareBackend._reset_code(17, vector_config) == (17,) * 16
    assert vector_config.to_manifest_dict()["reset_code_table"] == vector_lookup
    rejects(
        lambda: PrimitiveNoiseConfig(
            reset_code_table=tuple((code,) for code in range(32))
        )
    )
    assert parse_reset_code_table(list(range(32)), device_count=16) == lookup
    flat_vector = [code for code in range(32) for _ in range(16)]
    assert parse_reset_code_table(flat_vector, device_count=16) == vector_lookup
    rejects(lambda: parse_reset_code_table([0, 1], device_count=16))
    rejects(lambda: PrimitiveNoiseConfig(hagen_num_sends=0))
    rejects(lambda: PrimitiveNoiseConfig(hagen_candidate_count=15))
    rejects(lambda: PrimitiveNoiseConfig(hagen_chunk_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(psi_ed_chunk_repeats=0))
    rejects(lambda: PrimitiveNoiseConfig(deadline_miss_rate_limit=1.01))

    calls: list[tuple[int, int]] = []

    def fake_run(primitive, stage, code, config, *, repeats=None):
        assert primitive == "phi-np"
        assert stage == "static"
        assert repeats is not None
        calls.append((code, repeats))
        first = torch.full(
            (repeats, config.device_count),
            10.0e-6 - code * 0.1e-6,
            dtype=torch.float64,
        )
        count = torch.ones_like(first, dtype=torch.int64)
        return first, count, None, {"chip_identifier": ["synthetic-chip"]}

    original_probe = primitive_backend_module.probe_primitive_capabilities
    with tempfile.TemporaryDirectory() as directory:
        calibration = Path(directory) / "spiking.pbin"
        calibration.write_bytes(b"calibration")
        chunk_config = PrimitiveNoiseConfig(
            repeats=5,
            calibration_repeats=2,
            device_count=2,
            physical_coordinates=(0, 1),
            spiking_calibration_path=calibration,
            pynn_chunk_repeats=2,
            pynn_process_repeats=None,
        )
        backend = PrimitiveHardwareBackend()
        backend._run_pynn_code = fake_run  # type: ignore[method-assign]
        primitive_backend_module.probe_primitive_capabilities = lambda: {
            "phi-np": True,
            "pynn_version": "synthetic",
        }
        try:
            observation = backend.collect(
                "phi-np", chunk_config, stage="static", quick=True
            )
        finally:
            primitive_backend_module.probe_primitive_capabilities = original_probe
    assert calls == [(0, 2), (0, 2), (0, 1), (15, 2), (15, 2), (15, 1), (30, 2), (30, 2), (30, 1)]
    assert tuple(observation.observed.shape) == (5, 3, 2)
    assert observation.metadata["chunk_repeats"] == 2
    assert observation.metadata["chip_identifier"] == ["synthetic-chip"]
    calibration_indices = observation.metadata["calibration_acquisition_indices"]
    validation_indices = observation.metadata["validation_acquisition_indices"]
    assert len(calibration_indices) == 2
    assert len(validation_indices) == 3
    assert sorted(calibration_indices + validation_indices) == list(range(5))
    repeated_order = PrimitiveHardwareBackend._split_trial_order(5, 2, seed=0)
    assert calibration_indices == list(repeated_order[1])
    assert validation_indices == list(repeated_order[2])
    process_chunks = observation.metadata["per_code"][0]["chunks"]
    assert [(chunk["trial_start"], chunk["trial_stop"]) for chunk in process_chunks] == [
        (0, 5)
    ]
    assert [
        (chunk["trial_start"], chunk["trial_stop"])
        for chunk in process_chunks[0]["chunks"]
    ] == [(0, 2), (2, 4), (4, 5)]

    process_calls: list[tuple[int, int]] = []

    def fake_process(primitive, stage, code, config, *, repeats, trial_start):
        process_calls.append((code, repeats))
        first = torch.full(
            (repeats, config.device_count),
            10.0e-6 - code * 0.1e-6,
            dtype=torch.float64,
        )
        count = torch.ones_like(first, dtype=torch.int64)
        return first, count, None, {"chip_identifier": ["synthetic-chip"]}

    with tempfile.TemporaryDirectory() as directory:
        calibration = Path(directory) / "spiking.pbin"
        calibration.write_bytes(b"calibration")
        process_config = replace(
            chunk_config,
            spiking_calibration_path=calibration,
            pynn_process_repeats=2,
        )
        backend = PrimitiveHardwareBackend()
        backend._run_pynn_code_process = fake_process  # type: ignore[method-assign]
        primitive_backend_module.probe_primitive_capabilities = lambda: {
            "phi-np": True,
            "pynn_version": "synthetic",
        }
        try:
            isolated = backend.collect(
                "phi-np", process_config, stage="static", quick=True
            )
        finally:
            primitive_backend_module.probe_primitive_capabilities = original_probe
    assert process_calls == [
        (0, 2), (0, 2), (0, 1),
        (15, 2), (15, 2), (15, 1),
        (30, 2), (30, 2), (30, 1),
    ]
    assert tuple(isolated.observed.shape) == (5, 3, 2)
    assert isolated.metadata["process_repeats"] == 2

    worker_calls: list[list[str]] = []
    original_subprocess_run = primitive_backend_module.subprocess.run
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        calibration = root / "spiking.pbin"
        calibration.write_bytes(b"calibration")
        cache_config = PrimitiveNoiseConfig(
            repeats=8,
            calibration_repeats=4,
            device_count=2,
            physical_coordinates=(0, 1),
            spiking_calibration_path=calibration,
            pynn_worker_cache_dir=root / "cache",
        )

        def fake_subprocess_run(command, **_kwargs):
            worker_calls.append(command)
            response_path = Path(command[-1])
            torch.save(
                {
                    "first": torch.ones((8, 2), dtype=torch.float64),
                    "count": torch.ones((8, 2), dtype=torch.int64),
                    "precharge": None,
                    "metadata": {"chip_identifier": ["synthetic-chip"]},
                },
                response_path,
            )
            return SimpleNamespace(stdout="synthetic worker")

        primitive_backend_module.subprocess.run = fake_subprocess_run
        try:
            backend = PrimitiveHardwareBackend()
            first = backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                cache_config,
                repeats=8,
                trial_start=0,
            )
            second = backend._run_pynn_code_process(
                "phi-np",
                "static",
                15,
                cache_config,
                repeats=8,
                trial_start=0,
            )
        finally:
            primitive_backend_module.subprocess.run = original_subprocess_run
        assert len(worker_calls) == 1
        assert first[3]["worker_cache_hit"] is False
        assert second[3]["worker_cache_hit"] is True
        torch.testing.assert_close(first[0], second[0])


# @lat: [[hardware#Independent Primitive Noise Verification#Static reset separation]]
def verify_static_reset_separation() -> None:
    class Population:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def set(self, **parameters: object) -> None:
            self.calls.append(parameters)

    population = Population()
    PrimitiveHardwareBackend._set_population_reset(population, 300)
    PrimitiveHardwareBackend._set_population_reset(population, (301, 302))
    assert population.calls == [
        {"reset_v_reset": 300},
        {"reset_v_reset": [301, 302]},
    ]

    settings = SimpleNamespace(
        refractory_counters=list(range(512)),
        reset_holdoff=[value + 1 for value in range(512)],
        input_clock=[value % 2 for value in range(512)],
    )
    selected = PrimitiveHardwareBackend._selected_refractory_parameters(
        settings, [3, 134, 273, 390]
    )
    assert selected == {
        "refractory_period_refractory_time": [3, 134, 273, 390],
        "refractory_period_reset_holdoff": [4, 135, 274, 391],
        "refractory_period_input_clock": [1, 0, 1, 0],
    }
    malformed_settings = SimpleNamespace(
        refractory_counters=[1],
        reset_holdoff=settings.reset_holdoff,
        input_clock=settings.input_clock,
    )
    rejects(
        lambda: PrimitiveHardwareBackend._selected_refractory_parameters(
            malformed_settings, [0]
        ),
        (RuntimeError,),
    )

    applied: dict[str, object] = {}

    class RefractorySettings:
        refractory_counters = [31] * 512
        reset_holdoff = [0] * 512
        input_clock = [0] * 512
        fast_clock = 9
        slow_clock = 12

        def apply_to_chip(self, chip: object) -> None:
            applied["chip"] = chip

    def calculate_settings(targets: torch.Tensor) -> RefractorySettings:
        applied["targets"] = targets.clone()
        return RefractorySettings()

    original_import_module = primitive_backend_module.import_module

    def fake_import_module(name: str):
        if name == "numpy":
            return SimpleNamespace(
                full=lambda count, value: torch.full((count,), value)
            )
        if name == "quantities":
            return SimpleNamespace(s=1.0)
        if name == "calix.spiking.refractory_period":
            return SimpleNamespace(calculate_settings=calculate_settings)
        return original_import_module(name)

    primitive_backend_module.import_module = fake_import_module
    try:
        refractory_config = PrimitiveNoiseConfig(
            repeats=2,
            calibration_repeats=1,
            device_count=4,
            physical_coordinates=(3, 134, 273, 390),
            deadline_s=900.0e-6,
        )
        chip = object()
        _, refractory_metadata = (
            PrimitiveHardwareBackend._configure_first_spike_refractory(
                chip, refractory_config
            )
        )
    finally:
        primitive_backend_module.import_module = original_import_module
    assert applied["chip"] is chip
    torch.testing.assert_close(
        applied["targets"], torch.full((512,), 1.8e-3)
    )
    assert refractory_metadata["target_s"] == 1.8e-3

    source = inspect.getsource(
        PrimitiveHardwareBackend._run_pynn_code
    )
    trial_loop = source.index("for trial in range(total_trials)")
    initial_run = source.index("pynn.run(reference_ms, append)", trial_loop)
    ramp_enable = source.index(
        "population.set(constant_current_enable=True)", initial_run
    )
    state_retention = source[trial_loop:ramp_enable]
    assert "_set_population_reset" not in state_retention
    assert "config.reset_code_minimum" not in state_retention
    assert '"static_reset_policy"' in source
    assert '"input code retained for the full trial"' in source
    assert "2.0 * config.deadline_s" in source
    assert "_configure_first_spike_refractory" in source


# @lat: [[hardware#Independent Primitive Noise Verification#Hagen output qualification]]
def verify_hagen_output_qualification() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/primitive-output",
            "--hagen-chunk-repeats",
            "7",
        ]
    )
    config = make_config(args)
    assert config.hagen_chunk_repeats == 7
    assert config.to_manifest_dict()["hagen_chunk_repeats"] == 7

    class FakeHagenBackend:
        calls: list[int] = []

        def __init__(self, _config) -> None:
            pass

        def hardware_session(self):
            return nullcontext()

        def direct_linear(self, value, weight, *, avg):
            assert avg == 1
            self.calls.append(int(value.shape[0]))
            output = value @ weight.T
            return SimpleNamespace(
                value=output,
                metadata={"chip_identifier": ["synthetic-chip"]},
            )

    chunk_config = PrimitiveNoiseConfig(
        repeats=5,
        calibration_repeats=2,
        device_count=2,
        physical_coordinates=(0, 1),
        allow_environment_calibration=True,
        hagen_candidate_count=2,
        hagen_chunk_repeats=2,
    )
    original_backend = primitive_backend_module.HagenPWMBackend
    primitive_backend_module.HagenPWMBackend = FakeHagenBackend
    try:
        chunked = PrimitiveHardwareBackend().collect(
            "psi-int", chunk_config, quick=True
        )
    finally:
        primitive_backend_module.HagenPWMBackend = original_backend
    assert FakeHagenBackend.calls == [6, 6, 3] * 4
    assert tuple(chunked.observed.shape) == (5, 12, 2)
    assert chunked.metadata["chunk_repeats"] == 2
    assert chunked.metadata["chip_identifier"] == ["synthetic-chip"]
    assert [
        (chunk["trial_start"], chunk["trial_stop"])
        for chunk in chunked.metadata["per_drive"][0]["chunks"]
    ] == [(0, 2), (2, 4), (4, 5)]

    ideal = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    observed = torch.zeros((6, 3, 3), dtype=torch.float64)
    observed[:, :, 0] = ideal
    observed[:, :, 1] = ideal + torch.tensor([0.4, -0.3, 0.2])
    observed[:, :, 2] = ideal + torch.tensor([0.05, -0.05, 0.05])
    observed[3:, :, 0] += 50.0
    saturated = torch.zeros_like(observed, dtype=torch.bool)
    selected, scores = PrimitiveHardwareBackend._select_hagen_outputs(
        observed,
        saturated,
        ideal,
        calibration_repeats=3,
        selected_count=2,
    )
    assert selected == (0, 2)
    assert len(scores) == 3
    assert [int(row["output_index"]) for row in scores] == [0, 1, 2]
    observed[3:] -= 100.0
    repeated, _ = PrimitiveHardwareBackend._select_hagen_outputs(
        observed,
        saturated,
        ideal,
        calibration_repeats=3,
        selected_count=2,
    )
    assert repeated == selected


# @lat: [[hardware#Independent Primitive Noise Verification#Exponential response observation time]]
def verify_exponential_response_observation_time() -> None:
    parser = build_parser()
    default_args = parser.parse_args(["--output-dir", "/tmp/primitive-output"])
    default_config = make_config(default_args)
    assert default_config.observation_time_s == 40.0e-6
    assert default_config.psi_ne_input_fan_in == 2
    assert default_config.psi_ne_chunk_repeats == 4
    input_times = torch.tensor([5.0e-6, 25.0e-6], dtype=torch.float64)
    events = PrimitiveHardwareBackend._psi_ne_input_events(
        input_times,
        repeats=2,
        runtime_steps=61,
        dt_s=1.0e-6,
        fan_in=2,
    )
    assert tuple(events.shape) == (61, 8, 2)
    assert int(events[:, 0::2].sum()) == 0
    assert int(events[:, 1::2].sum()) == 8
    assert bool((events[5, 1, :] == 1).all())
    assert bool((events[25, 3, :] == 1).all())
    explicit_args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/primitive-output",
            "--observation-time",
            "45e-6",
            "--psi-ne-input-fan-in",
            "3",
            "--pynn-chunk-repeats",
            "5",
            "--psi-ne-chunk-repeats",
            "7",
        ]
    )
    explicit_config = make_config(explicit_args)
    assert explicit_config.observation_time_s == 45.0e-6
    assert explicit_config.psi_ne_input_fan_in == 3
    assert explicit_config.pynn_chunk_repeats == 5
    assert explicit_config.psi_ne_chunk_repeats == 7
    rejects(lambda: PrimitiveNoiseConfig(psi_ne_input_fan_in=0))
    rejects(lambda: PrimitiveNoiseConfig(psi_ne_chunk_repeats=0))

    calls: list[tuple[int, int]] = []

    def fake_process(
        config,
        *,
        input_times,
        runtime_steps,
        observation_step,
        repeats,
        trial_start,
    ):
        calls.append((trial_start, repeats))
        shape = (repeats, input_times.numel(), config.device_count)
        mean = 5.0 + 30.0 * torch.exp(
            -(config.observation_time_s - input_times) / 10.0e-6
        )
        observed = mean.reshape(1, -1, 1).expand(shape).clone()
        return {
            "baseline": torch.full(shape, -40.0, dtype=torch.float64),
            "observed": observed,
            "spike_count": torch.zeros(shape, dtype=torch.int64),
            "saturated": torch.zeros(shape, dtype=torch.bool),
            "calibration_loader": "synthetic-loader",
            "batch_count": 2 * repeats * input_times.numel(),
            "metadata": {"chip_identifier": ["synthetic-chip"]},
        }

    process_config = PrimitiveNoiseConfig(
        repeats=5,
        calibration_repeats=2,
        device_count=2,
        physical_coordinates=(0, 1),
        allow_environment_calibration=True,
        psi_ne_chunk_repeats=2,
    )
    original_probe = primitive_backend_module.probe_primitive_capabilities
    backend = PrimitiveHardwareBackend()
    backend._run_psi_ne_process = fake_process  # type: ignore[method-assign]
    primitive_backend_module.probe_primitive_capabilities = lambda: {
        "psi-ne": True,
        "hxtorch_version": "synthetic",
    }
    try:
        observation = backend.collect("psi-ne", process_config, quick=True)
    finally:
        primitive_backend_module.probe_primitive_capabilities = original_probe
    assert calls == [(0, 2), (2, 2), (4, 1)]
    assert tuple(observation.observed.shape) == (5, 3, 2)
    assert observation.metadata["chip_identifier"] == ["synthetic-chip"]
    assert [
        (chunk["trial_start"], chunk["trial_stop"])
        for chunk in observation.metadata["chunks"]
    ] == [(0, 2), (2, 4), (4, 5)]


# @lat: [[hardware#Independent Primitive Noise Verification#Insufficient fit preservation]]
def verify_insufficient_fit_preservation() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    base = MockPrimitiveNoiseBackend().collect(
        "phi-np", config, stage="static", quick=True
    )
    observed = base.observed.clone()
    delivered = base.delivered.clone()
    spike_count = base.spike_count.clone()
    observed[:, :, 0] = float("nan")
    delivered[:, :, 0] = False
    spike_count[:, :, 0] = 0
    missing_device = replace(
        base,
        observed=observed,
        delivered=delivered,
        spike_count=spike_count,
    )
    validation = validate_primitive_observation(missing_device, config)
    assert not validation.validated
    assert validation.gates["fit_available"] is False
    assert validation.device_statistics[0]["fit_available"] is False
    assert validation.device_statistics[1]["fit_available"] is True

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[missing_device],
            validations=[validation],
        )
        assert (output / "raw" / "index.json").is_file()
        payload = json.loads(
            (output / "primitive_noise_calibration.json").read_text()
        )
        assert not payload["validated"]


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
        "psi-ed",
    }


# @lat: [[hardware#Independent Primitive Noise Verification#Encoder operating point score]]
def verify_encoder_operating_point_score() -> None:
    config = PrimitiveNoiseConfig(
        repeats=32,
        calibration_repeats=16,
        device_count=4,
        physical_coordinates=default_primitive_coordinates(4),
    )
    backend = MockPrimitiveNoiseBackend()
    observations = [
        backend.collect("phi-np", config, stage="static", quick=False),
        backend.collect("phi-np", config, stage="dynamic", quick=False),
    ]
    validations = [
        validate_primitive_observation(observation, config)
        for observation in observations
    ]
    reference = score_encoder_operating_point(
        observations, validations, config, primitive="phi-np"
    )
    assert reference["selection_eligible"]
    assert reference["held_out_validated"]
    assert reference["selection_objective_rt"] > 0
    assert reference["validation_objective_rt"] > 0

    changed_values = observations[1].observed.clone()
    offsets = torch.tensor(
        [1.0, -1.0] * (config.validation_repeats // 2),
        dtype=changed_values.dtype,
    ).reshape(-1, 1, 1)
    changed_values[config.calibration_repeats :] += offsets * 1.0e-6
    changed_dynamic = replace(observations[1], observed=changed_values)
    changed_observations = [observations[0], changed_dynamic]
    changed_validations = [
        validations[0], validate_primitive_observation(changed_dynamic, config)
    ]
    changed = score_encoder_operating_point(
        changed_observations,
        changed_validations,
        config,
        primitive="phi-np",
    )
    assert changed["selection_objective_rt"] == reference["selection_objective_rt"]
    assert changed["validation_objective_rt"] > reference["validation_objective_rt"]

    screened_values = observations[0].observed.clone()
    screened_values[: config.calibration_repeats, 15, :] += 10.0e-6
    screened_static = replace(observations[0], observed=screened_values)
    screened_observations = [screened_static, observations[1]]
    screened_validations = [
        validate_primitive_observation(screened_static, config), validations[1]
    ]
    strict = score_encoder_operating_point(
        screened_observations,
        screened_validations,
        config,
        primitive="phi-np",
    )
    screening = score_encoder_operating_point(
        screened_observations,
        screened_validations,
        config,
        primitive="phi-np",
        screening=True,
    )
    assert not strict["selection_eligible"]
    assert screening["selection_eligible"]
    assert not screening["np_static"]["calibration_transfer"]["strict_eligible"]


# @lat: [[hardware#Independent Primitive Noise Verification#Resumable operating point search]]
def verify_resumable_operating_point_search() -> None:
    config = PrimitiveNoiseConfig(
        repeats=8,
        calibration_repeats=4,
        device_count=4,
        physical_coordinates=default_primitive_coordinates(4),
    )
    assert parse_precharge_pair("2:31") == (2, 31)
    rejects(lambda: parse_precharge_pair("2x31"), (ValueError,))
    parsed_current, parsed_stop = parse_current_stop_pair("512:40")
    assert parsed_current == 512
    torch.testing.assert_close(
        torch.tensor(parsed_stop), torch.tensor(40.0e-6), rtol=0.0, atol=1.0e-12
    )
    rejects(lambda: parse_current_stop_pair("512x40"), (ValueError,))
    candidates = build_encoder_operating_point_candidates(
        config,
        constant_current_codes=(512, 1022),
        threshold_codes=(600,),
        ramp_stop_times_s=(25.0e-6,),
        precharge_pairs=((1, 63),),
    )
    assert len(candidates) == 2
    assert candidates[0].apply(config).observation_time_s == 42.5e-6
    rejects(
        lambda: build_encoder_operating_point_candidates(
            config,
            constant_current_codes=(1023,),
            threshold_codes=(600,),
            ramp_stop_times_s=(25.0e-6,),
            precharge_pairs=((1, 63),),
        ),
        (ValueError,),
    )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "search"
        arguments = build_parser().parse_args(
            [
                "--phase",
                "optimize",
                "--primitive",
                "phi-np",
                "--backend",
                "mock",
                "--quick",
                "--device-count",
                "4",
                "--search-current-stop-pairs",
                "512:25",
                "1022:25",
                "--search-threshold-codes",
                "600",
                "--search-precharge-pairs",
                "1:63",
                "--output-dir",
                str(output),
            ]
        )
        run(arguments)
        selection_path = output / "selected_operating_point.json"
        first = selection_path.read_text(encoding="utf-8")
        assert (output / "search_manifest.json").is_file()
        assert (output / "operating_point_results.csv").is_file()
        assert len(list((output / "candidates").glob("*/candidate_result.json"))) == 2
        run(arguments)
        assert selection_path.read_text(encoding="utf-8") == first

    notebook = json.loads(
        (
            ROOT
            / "scripts"
            / "notebooks"
            / "ebrains_brainscales2_primitive_noise.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "RUN_OPERATING_POINT_SEARCH = False" in source
    assert "'--phase', 'optimize'" in source
    assert "selected_operating_point.json" in source
    assert notebook["metadata"]["language_info"]["version"] == "3.11"


# @lat: [[hardware#Independent Primitive Noise Verification#Artifact integrity]]
def verify_artifact_integrity() -> None:
    config = PrimitiveNoiseConfig(repeats=8, calibration_repeats=4)
    assert config.reset_code_minimum == 300
    assert config.reset_code_maximum == 900
    assert config.constant_current_code == 1022
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
        assert manifest["multiple_spike_handling"] == "first-spike-time"
        assert (output / "psi-int_transfer" / "moments.csv").is_file()
        chunk = next((output / "raw").glob("*.pt"))
        chunk.write_bytes(chunk.read_bytes() + b"changed")
        rejects(lambda: load_primitive_observations(output, config), (ValueError,))

    ed_observation = MockPrimitiveNoiseBackend().collect(
        "psi-ed", config, stage="transfer", quick=True
    )
    ed_validation = validate_primitive_observation(ed_observation, config)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        write_primitive_noise_artifacts(
            output,
            config=config,
            observations=[ed_observation],
            validations=[ed_validation],
        )
        loaded = load_primitive_observations(output, config)[0]
        torch.testing.assert_close(
            loaded.first_spike_time_s,
            ed_observation.first_spike_time_s,
            equal_nan=True,
        )
        moments = (output / "psi-ed_transfer" / "moments.csv").read_text()
        assert "mean" in moments and "variance" in moments

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
    verify_validated_np_placement()
    verify_miss_and_saturation_semantics()
    verify_rank_monotonicity_gate()
    verify_segmented_pynn_recording_decode()
    verify_dynamic_recording_selection()
    verify_segmented_dense_recording_decode()
    verify_sparse_precharge_evidence()
    verify_installed_component_resolution()
    verify_correlation_recording_decode()
    verify_correlation_worker_boundary()
    verify_nonlinear_drive_configuration()
    verify_static_reset_separation()
    verify_hagen_output_qualification()
    verify_exponential_response_observation_time()
    verify_insufficient_fit_preservation()
    verify_np_stage_gate()
    verify_encoder_operating_point_score()
    verify_resumable_operating_point_search()
    verify_artifact_integrity()
    print("BrainScaleS-2 primitive-noise checks passed")


if __name__ == "__main__":
    main()
