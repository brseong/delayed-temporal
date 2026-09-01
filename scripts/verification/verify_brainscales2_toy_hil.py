#!/usr/bin/env python3
"""Pure-Python regression checks for the toy BrainScaleS-2 HIL path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import ast
import csv
import json
import subprocess
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluation.brainscales2_toy_hil import (
    _aggregate_isolated_conditions,
    _apply_condition_worker_config,
    _evaluate_readout_ablations,
    _load_isolated_condition,
    _run_hagen_output,
    _run_isolated_pool_chunks,
    _run_temporal_pool,
    _run_worker_command_with_retries,
)
from utils.hardware.brainscales2.config import BrainScaleS2PoolConfig
from utils.hardware.brainscales2.hagen import HagenConfig, HagenPWMBackend, HagenResult
from utils.hardware.brainscales2.toy import (
    ARCHITECTURES,
    ToyMLP,
    convert_float_model,
    deserialize_converted_model,
    make_yin_yang_split,
    parameter_sha256,
    serialize_converted_model,
)
from utils.hardware.brainscales2.toy_artifacts import (
    ToyConditionEvaluation,
    activation_error_by_code,
    summarize_toy_evaluations,
    write_toy_artifacts,
)
from utils.hardware.brainscales2.toy_pooling import (
    _configure_grouped_synapse_weights,
    _grouped_input_channel_slice,
    GroupedHardwarePoolBackend,
    MockToyPoolBackend,
    ReplayToyPoolBackend,
    TimingCalibration,
    TimingCalibrationObservation,
    ToyPoolConfig,
    ToyPoolResult,
    calibrate_timing,
    concatenate_timing_calibration_observations,
    concatenate_toy_pool_results,
    decode_pool_observations,
    resolve_grouped_physical_coordinates,
)


def verify_deterministic_yin_yang_splits() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Deterministic datasets and frozen conversion]]
    first_x, first_y = make_yin_yang_split(120, 42)
    second_x, second_y = make_yin_yang_split(120, 42)
    other_x, _ = make_yin_yang_split(120, 41)
    torch.testing.assert_close(first_x, second_x)
    torch.testing.assert_close(first_y, second_y)
    assert first_x.shape == (120, 4)
    assert first_y.bincount(minlength=3).tolist() == [40, 40, 40]
    assert not torch.equal(first_x, other_x)
    assert bool(((first_x >= 0.0) & (first_x <= 1.0)).all())


def _converted_fixture() -> tuple[ToyMLP, object, torch.Tensor]:
    torch.manual_seed(7)
    model = ToyMLP(ARCHITECTURES["yy-30"])
    calibration_x, _ = make_yin_yang_split(96, 41)
    converted = convert_float_model(model, calibration_x)
    return model, converted, calibration_x


def verify_frozen_integer_conversion() -> None:
    model, converted, calibration_x = _converted_fixture()
    before = parameter_sha256(model)
    forward = converted.forward(calibration_x[:8])
    after = parameter_sha256(model)
    assert before == after == converted.manifest.source_parameter_sha256
    assert int(forward.input_uint5.min()) >= 0 and int(forward.input_uint5.max()) <= 31
    assert int(forward.hidden_uint5.min()) >= 0 and int(forward.hidden_uint5.max()) <= 31
    assert int(converted.first.weight_with_bias.min()) >= -63
    assert int(converted.first.weight_with_bias.max()) <= 63
    assert int(forward.logits_int8.min()) >= -128
    assert int(forward.logits_int8.max()) <= 127
    assert converted.first.weight_with_bias.shape == (30, 5)
    assert converted.second.weight_with_bias.shape == (3, 31)
    restored = deserialize_converted_model(serialize_converted_model(converted))
    restored_forward = restored.forward(calibration_x[:8])
    torch.testing.assert_close(forward.hidden_uint5, restored_forward.hidden_uint5)
    torch.testing.assert_close(forward.logits_int8, restored_forward.logits_int8)


def verify_grouped_placement() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Physical pool allocation]]
    local = resolve_grouped_physical_coordinates(30, 16, "local-pool", "dedicated")
    cross = resolve_grouped_physical_coordinates(30, 16, "cross-quadrant", "dedicated")
    assert local.shape == (30, 16) and cross.shape == (30, 16)
    assert torch.unique(local).numel() == 480
    assert torch.unique(cross).numel() == 480
    assert int(local.min()) >= 0 and int(local.max()) < 512
    for logical in range(30):
        assert torch.unique(local[logical] // 128).numel() == 1
        assert torch.unique(cross[logical] // 128).numel() == 4
    reused = resolve_grouped_physical_coordinates(
        128, 16, "cross-quadrant", "time-multiplexed"
    )
    assert bool((reused == reused[0]).all())
    try:
        ToyPoolConfig(pool_size=16, logical_neurons=128, mapping="dedicated")
    except ValueError as error:
        assert "512" in str(error)
    else:
        raise AssertionError("oversized dedicated mapping was accepted")


def verify_grouped_broadcast_fan_in() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Grouped broadcast fan-in]]
    dedicated = ToyPoolConfig(
        pool_size=2,
        logical_neurons=2,
        mapping="dedicated",
        inference_trials=1,
        calibration_trials=2,
    )
    assert _grouped_input_channel_slice(0, dedicated, 3) == slice(0, 3)
    assert _grouped_input_channel_slice(1, dedicated, 3) == slice(3, 6)
    weight = torch.full((4, 6), -1.0)
    _configure_grouped_synapse_weights(
        weight,
        dedicated,
        input_fan_in=3,
        synaptic_weight=7.0,
    )
    expected = torch.tensor(
        [
            [7, 7, 7, 0, 0, 0],
            [7, 7, 7, 0, 0, 0],
            [0, 0, 0, 7, 7, 7],
            [0, 0, 0, 7, 7, 7],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(weight, expected)

    reused = ToyPoolConfig(
        pool_size=2,
        logical_neurons=2,
        mapping="time-multiplexed",
        inference_trials=1,
        calibration_trials=2,
    )
    reused_weight = torch.zeros((2, 3))
    _configure_grouped_synapse_weights(
        reused_weight, reused, input_fan_in=3, synaptic_weight=5.0
    )
    torch.testing.assert_close(reused_weight, torch.full((2, 3), 5.0))


def verify_mock_and_all_miss_policy() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Miss-aware temporal decoding]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(4,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    hidden = torch.tensor([[0, 15], [31, 8]], dtype=torch.int32)
    config = ToyPoolConfig(
        pool_size=4,
        logical_neurons=2,
        inference_trials=3,
        calibration_trials=4,
        seed=19,
        miss_probability=0.0,
    )
    first = MockToyPoolBackend().run_uint5(hidden, config, spiking)
    second = MockToyPoolBackend().run_uint5(hidden, config, spiking)
    torch.testing.assert_close(first.first_spike_s, second.first_spike_s)
    torch.testing.assert_close(first.decoded_uint5, second.decoded_uint5)
    assert float((first.decoded_uint5 - hidden).abs().float().mean()) < 2.0

    coordinates = resolve_grouped_physical_coordinates(2, 4, "local-pool", "dedicated")
    all_miss = torch.full((1, 2, 2, 4), torch.nan, dtype=torch.float64)
    nominal = torch.full((2, 2), 15.0e-6, dtype=torch.float64)
    decoded = decode_pool_observations(
        all_miss,
        nominal,
        TimingCalibration(5.0e-6, torch.zeros((2, 4), dtype=torch.float64), 4),
        coordinates,
        ToyPoolConfig(
            pool_size=4,
            logical_neurons=2,
            inference_trials=1,
            calibration_trials=4,
        ),
        spiking,
    )
    assert bool(decoded.all_miss.all())
    assert int(decoded.decoded_uint5.sum()) == 0


def verify_mean_and_corrected_max_estimators() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Max estimator attribution]]
    spiking = BrainScaleS2PoolConfig(
        input_early_s=5.0e-6,
        input_late_s=25.0e-6,
        pool_sizes=(2,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    width = spiking.input_late_s - spiking.input_early_s
    code = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    nominal_codes = spiking.input_late_s - code / 31.0 * width
    nominal = nominal_codes[1].reshape(1, 1)
    coordinates = resolve_grouped_physical_coordinates(
        1, 2, "local-pool", "dedicated"
    )

    one_miss = torch.tensor(
        [[[[float(nominal), torch.nan]]]], dtype=torch.float64
    )
    identity_calibration = TimingCalibration(
        response_delay_s=0.0,
        neuron_offset_s=torch.zeros((1, 2), dtype=torch.float64),
        calibration_trials=4,
        nominal_code_time_s=nominal_codes,
        raw_max_expected_time_s=nominal_codes,
        analytic_max_correction_s=torch.zeros(3, dtype=torch.float64),
    )
    decoded: dict[str, int] = {}
    for estimator in ("mean", "raw-max"):
        result = decode_pool_observations(
            one_miss,
            nominal,
            identity_calibration,
            coordinates,
            ToyPoolConfig(
                pool_size=2,
                logical_neurons=1,
                inference_trials=1,
                calibration_trials=4,
                estimator=estimator,
            ),
            spiking,
        )
        decoded[estimator] = int(result.decoded_uint5[0, 0, 0])
    assert decoded["mean"] == 5
    assert decoded["raw-max"] == 10

    early_bias = 2.0e-6
    biased = torch.tensor(
        [[[[float(nominal - early_bias), float(nominal + 1.0e-6)]]]],
        dtype=torch.float64,
    )
    corrected_calibration = TimingCalibration(
        response_delay_s=0.0,
        neuron_offset_s=torch.zeros((1, 2), dtype=torch.float64),
        calibration_trials=4,
        nominal_code_time_s=nominal_codes,
        raw_max_expected_time_s=nominal_codes - early_bias,
        analytic_max_correction_s=torch.full((3,), early_bias, dtype=torch.float64),
    )
    for estimator in ("analytic-corrected-max", "empirical-corrected-max"):
        result = decode_pool_observations(
            biased,
            nominal,
            corrected_calibration,
            coordinates,
            ToyPoolConfig(
                pool_size=2,
                logical_neurons=1,
                inference_trials=1,
                calibration_trials=4,
                estimator=estimator,
            ),
            spiking,
        )
        assert int(result.decoded_uint5[0, 0, 0]) == 10


def verify_chunked_pool_aggregation() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Chunked pool aggregation]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(2,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=2,
        logical_neurons=2,
        inference_trials=2,
        calibration_trials=4,
        seed=29,
        miss_probability=0.0,
    )
    backend = MockToyPoolBackend()
    first = backend.run_uint5(
        torch.tensor([[0, 31]], dtype=torch.int32), config, spiking
    )
    second = backend.run_uint5(
        torch.tensor([[7, 19]], dtype=torch.int32), config, spiking
    )
    joined = concatenate_toy_pool_results([first, second])
    assert joined.first_spike_s.shape == (2, 2, 2, 2)
    assert joined.decoded_uint5.shape == (2, 2, 2)
    torch.testing.assert_close(
        joined.nominal_input_s,
        torch.cat((first.nominal_input_s, second.nominal_input_s), dim=0),
    )
    assert joined.metadata["chunked"] is True
    assert joined.metadata["sample_chunk_count"] == 2
    assert joined.metadata["calibration_strategy"] == "per-sample-chunk"


def verify_shared_split_timing_calibration() -> None:
    coordinates = resolve_grouped_physical_coordinates(
        2, 2, "local-pool", "dedicated"
    )
    code_times = torch.tensor([5.0e-6, 15.0e-6, 25.0e-6], dtype=torch.float64)
    offsets = torch.tensor(
        [[0.2e-6, -0.2e-6], [0.4e-6, -0.4e-6]], dtype=torch.float64
    )

    def observation(trials: int, marker: str) -> TimingCalibrationObservation:
        first = (
            code_times.reshape(1, -1, 1, 1)
            + 5.0e-6
            + offsets.reshape(1, 1, 2, 2)
        ).repeat(trials, 1, 1, 1)
        return TimingCalibrationObservation(
            first_spike_s=first,
            nominal_input_s=code_times,
            physical_coordinates=coordinates,
            metadata={"marker": marker},
        )

    joined = concatenate_timing_calibration_observations(
        [observation(2, "first"), observation(3, "second")]
    )
    timing = calibrate_timing(joined.first_spike_s, joined.nominal_input_s)
    assert joined.first_spike_s.shape == (5, 3, 2, 2)
    assert joined.metadata["calibration_strategy"] == "shared-split"
    assert joined.metadata["calibration_chunk_count"] == 2
    assert timing.calibration_trials == 5
    torch.testing.assert_close(timing.neuron_offset_s, offsets)
    assert abs(timing.response_delay_s - 5.0e-6) < 1.0e-12


def verify_m16_calibration_inference_batch_separation() -> None:
    spiking = BrainScaleS2PoolConfig(
        trials=8,
        pool_sizes=(16,),
        placements=("same-quadrant",),
        routings=("broadcast",),
        input_fan_in=4,
    )
    config = ToyPoolConfig(
        pool_size=16,
        logical_neurons=30,
        inference_trials=8,
        calibration_trials=4,
    )

    class RecordingHardwareBackend(GroupedHardwarePoolBackend):
        def __init__(self) -> None:
            self.batch_counts: list[int] = []

        def _run_inputs(
            self,
            inputs: torch.Tensor,
            pool_config: ToyPoolConfig,
            spiking_config: BrainScaleS2PoolConfig,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
            self.batch_counts.append(inputs.shape[1])
            coordinates = resolve_grouped_physical_coordinates(
                pool_config.logical_neurons,
                pool_config.pool_size,
                pool_config.placement,
                pool_config.mapping,
            )
            output_neurons = coordinates.numel()
            first = torch.full(
                (inputs.shape[1], output_neurons), 15.0e-6, dtype=torch.float64
            )
            count = torch.ones_like(first, dtype=torch.int64)
            return first, count, coordinates, {"backend": "recording"}

    backend = RecordingHardwareBackend()
    observation = backend.observe_timing_calibration(config, spiking)
    timing = calibrate_timing(
        observation.first_spike_s, observation.nominal_input_s
    )
    result = backend.run_uint5(
        torch.zeros((8, 30), dtype=torch.int32),
        config,
        spiking,
        timing_calibration=timing,
    )
    assert backend.batch_counts == [44, 64]
    assert result.first_spike_s.shape == (8, 8, 30, 16)
    assert result.metadata["calibration_strategy"] == "shared-split"


def verify_pool_size_aware_hardware_chunk_cap() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Pool-size-aware hardware chunk cap]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(8,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=8,
        logical_neurons=2,
        inference_trials=2,
        calibration_trials=4,
        seed=31,
        miss_probability=0.0,
    )

    class RecordingBackend:
        def __init__(self) -> None:
            self.sizes: list[int] = []
            self.delegate = MockToyPoolBackend()

        def run_uint5(
            self,
            hidden_uint5: torch.Tensor,
            pool_config: ToyPoolConfig,
            spiking_config: BrainScaleS2PoolConfig,
        ) -> object:
            self.sizes.append(hidden_uint5.shape[0])
            return self.delegate.run_uint5(hidden_uint5, pool_config, spiking_config)

    backend = RecordingBackend()
    hidden = torch.arange(65 * 2, dtype=torch.int32).reshape(65, 2).remainder(32)
    result = _run_temporal_pool(
        SimpleNamespace(
            pool_backend="hardware",
            pool_sample_chunk_size=64,
            pool_replica_sample_budget=256,
        ),
        backend,
        hidden,
        config,
        spiking,
    )
    assert backend.sizes == [32, 32, 1]
    assert result.decoded_uint5.shape[1] == 65
    assert result.metadata["requested_pool_sample_chunk_size"] == 64
    assert result.metadata["effective_pool_sample_chunk_size"] == 32
    assert result.metadata["pool_replica_sample_budget"] == 256


def verify_pool_chunk_process_isolation() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Pool chunk process isolation]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(8,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=8,
        logical_neurons=2,
        inference_trials=2,
        calibration_trials=4,
        seed=37,
        miss_probability=0.0,
    )
    hidden = torch.arange(35 * 2, dtype=torch.int32).reshape(35, 2).remainder(32)
    launched: list[tuple[str, int]] = []
    backend = MockToyPoolBackend()

    def launch(
        args: SimpleNamespace,
        chunk_dir: Path,
        hidden_chunk: torch.Tensor,
        pool_config: ToyPoolConfig,
        spiking_config: BrainScaleS2PoolConfig,
        timing_calibration: TimingCalibration,
        timing_calibration_path: Path,
    ) -> ToyPoolResult:
        chunk_dir.mkdir(parents=True)
        launched.append((chunk_dir.name, hidden_chunk.shape[0]))
        (chunk_dir / "worker_status.json").write_text(
            json.dumps({"status": "passed", "attempts": [{"attempt": 1}]}),
            encoding="utf-8",
        )
        result = backend.run_uint5(hidden_chunk, pool_config, spiking_config)
        return replace(
            result,
            metadata={**result.metadata, "calibration_strategy": "shared-split"},
        )

    calibration_launches: list[tuple[str, int]] = []

    def launch_calibration(
        args: SimpleNamespace,
        chunk_dir: Path,
        pool_config: ToyPoolConfig,
        spiking_config: BrainScaleS2PoolConfig,
    ) -> TimingCalibrationObservation:
        chunk_dir.mkdir(parents=True)
        calibration_launches.append((chunk_dir.name, pool_config.calibration_trials))
        (chunk_dir / "worker_status.json").write_text(
            json.dumps({"status": "passed", "attempts": [{"attempt": 1}]}),
            encoding="utf-8",
        )
        code_times = torch.linspace(
            spiking_config.input_late_s,
            spiking_config.input_early_s,
            11,
            dtype=torch.float64,
        )
        coordinates = resolve_grouped_physical_coordinates(
            pool_config.logical_neurons,
            pool_config.pool_size,
            pool_config.placement,
            pool_config.mapping,
        )
        first = (
            code_times.reshape(1, -1, 1, 1) + 5.0e-6
        ).repeat(
            pool_config.calibration_trials,
            1,
            pool_config.logical_neurons,
            pool_config.pool_size,
        )
        return TimingCalibrationObservation(
            first_spike_s=first,
            nominal_input_s=code_times,
            physical_coordinates=coordinates,
        )

    with TemporaryDirectory() as directory:
        result = _run_isolated_pool_chunks(
            SimpleNamespace(
                output_dir=Path(directory),
                pool_calibration_trial_chunk_size=2,
            ),
            hidden,
            config,
            spiking,
            16,
            launcher=launch,
            calibration_launcher=launch_calibration,
        )
    assert calibration_launches == [
        ("trials_0000_0002", 2),
        ("trials_0002_0004", 2),
    ]
    assert launched == [
        ("chunk_000000_000016", 16),
        ("chunk_000016_000032", 16),
        ("chunk_000032_000035", 3),
    ]
    assert result.decoded_uint5.shape[1] == 35
    assert result.metadata["chunk_process_isolation"] is True
    assert len(result.metadata["chunk_worker_dirs"]) == 3
    assert len(result.metadata["chunk_worker_status"]) == 3
    assert result.metadata["calibration_strategy"] == "shared-split"
    assert result.metadata["shared_timing_calibration"]["calibration_trials"] == 4


def verify_replay_split_and_reproducibility() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Held-out hardware replay]]
    path = (
        REPOSITORY_ROOT
        / "artifacts/brainscales2/20260829T084007Z/full_pooling/events.pt"
    )
    if not path.is_file():
        raise AssertionError("allowlisted accepted replay artifact is missing")
    backend = ReplayToyPoolBackend(path)
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(4,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    config = ToyPoolConfig(
        pool_size=4,
        logical_neurons=3,
        inference_trials=2,
        calibration_trials=4,
        seed=23,
    )
    hidden = torch.tensor([[0, 15, 31], [7, 21, 4]], dtype=torch.int32)
    first = backend.run_uint5(hidden, config, spiking)
    second = backend.run_uint5(hidden, config, spiking)
    torch.testing.assert_close(first.first_spike_s, second.first_spike_s, equal_nan=True)
    assert first.metadata["held_out_trials"] == 128
    assert first.metadata["scope"] == "rough-model-only"


class _FakeChunkedHagen:
    def __init__(self) -> None:
        self.config = SimpleNamespace(mode="hardware")
        self.row_counts: list[int] = []

    def output_layer(self, converted: object, hidden: torch.Tensor) -> HagenResult:
        del converted
        self.row_counts.append(hidden.shape[0])
        value = hidden.sum(dim=1, keepdim=True).to(torch.int8)
        return HagenResult(
            value=value,
            metadata={
                "backend": "fake-hardware",
                "input_shape": list(hidden.shape),
                "output_shape": list(value.shape),
                "elapsed_s": 0.25,
            },
        )


def verify_hagen_output_row_chunking() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Hagen output row chunking]]
    hagen = _FakeChunkedHagen()
    hidden = torch.arange(14, dtype=torch.int32).reshape(7, 2)
    result = _run_hagen_output(
        SimpleNamespace(hagen_row_chunk_size=3),
        hagen,
        None,
        hidden,
    )
    assert hagen.row_counts == [3, 3, 1]
    torch.testing.assert_close(
        result.value, hidden.sum(dim=1, keepdim=True).to(torch.int8)
    )
    assert result.metadata["chunked"] is True
    assert result.metadata["row_chunk_count"] == 3
    assert result.metadata["input_shape"] == [7, 2]
    assert result.metadata["output_shape"] == [7, 1]
    assert result.metadata["elapsed_s"] == 0.75


def verify_hagen_host_tiling() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Hagen tiling contract]]
    _, converted, _ = _converted_fixture()

    class FakePerceptron:
        @staticmethod
        def matmul(input_value: torch.Tensor, weight: torch.Tensor, **_: object) -> torch.Tensor:
            return input_value @ weight

    fake_hxtorch = SimpleNamespace(perceptron=FakePerceptron())
    backend = HagenPWMBackend(HagenConfig(mode="mock", tiling="host-128", tile_size=2))
    value = torch.tensor([[1, 2, 3, 4, 31]], dtype=torch.float32)
    result, schedule = backend._host_tiled_linear(  # noqa: SLF001 - contract probe
        fake_hxtorch,
        value,
        converted.first,
        avg=1,
    )
    expected = value.to(torch.int32) @ converted.first.weight_with_bias.T.to(torch.int32)
    torch.testing.assert_close(result.to(torch.int32), expected.clamp(-128, 127))
    assert [
        {"start": row["start"], "stop": row["stop"]} for row in schedule
    ] == [
        {"start": 0, "stop": 2},
        {"start": 2, "stop": 4},
        {"start": 4, "stop": 5},
    ]
    assert all(0.0 <= row["saturation_rate"] <= 1.0 for row in schedule)


def verify_host_mediated_implicit_relu_boundary() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Host-mediated implicit ReLU boundary]]
    hidden, metadata = HagenPWMBackend._implicit_lower_bound_uint5(  # noqa: SLF001
        torch.tensor([[-128.0, -3.0, 0.0, 3.0, 62.0, 127.0]]),
        shift=1,
    )
    torch.testing.assert_close(
        hidden,
        torch.tensor([[0, 0, 0, 2, 31, 31]], dtype=torch.int32),
    )
    assert metadata["relu_boundary"] == "implicit-lower-bound-host"
    assert metadata["converting_relu"] is None
    assert metadata["host_mediated_lower_bound"] is True
    assert metadata["lower_bound_v"] == 0.0
    assert metadata["upper_bound_v"] == 31.0
    assert metadata["lower_bound_clamped_values"] == 2
    assert metadata["upper_bound_clamped_values"] == 1


def verify_sigmoid_host_activation_adapter() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Sigmoid host activation adapter]]
    model, _, calibration_x = _converted_fixture()
    sigmoid_model = ToyMLP(model.architecture, activation="sigmoid")
    sigmoid_model.load_state_dict(model.state_dict())
    converted = convert_float_model(sigmoid_model, calibration_x)
    forward = converted.forward(calibration_x[:8])
    assert converted.manifest.activation == "sigmoid"
    assert converted.manifest.hidden_shift == 0
    assert converted.manifest.hidden_scale == 1.0 / 31.0
    assert int(forward.hidden_uint5.min()) >= 0
    assert int(forward.hidden_uint5.max()) <= 31
    restored = deserialize_converted_model(serialize_converted_model(converted))
    assert restored.manifest.activation == "sigmoid"
    torch.testing.assert_close(restored.forward(calibration_x[:8]).hidden_uint5, forward.hidden_uint5)

    hidden, metadata = HagenPWMBackend._host_sigmoid_uint5(  # noqa: SLF001
        torch.tensor([[-20.0, 0.0, 20.0]]),
        input_scale=1.0,
    )
    torch.testing.assert_close(hidden, torch.tensor([[0, 16, 31]], dtype=torch.int32))
    assert metadata["activation_adapter"] == "host-sigmoid-uint5"
    assert metadata["host_mediated_activation"] is True
    assert metadata["sigmoid_physical_subcircuit"] is False


def verify_condition_process_isolation_contract() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Condition process isolation]]
    with TemporaryDirectory() as directory:
        output = Path(directory)
        config_path = output / "worker_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(output / "checkpoint.pt"),
                    "output_dir": str(output / "worker"),
                    "condition_worker": True,
                }
            ),
            encoding="utf-8",
        )
        loaded_args = _apply_condition_worker_config(
            SimpleNamespace(
                condition_worker_config=config_path,
                checkpoint=None,
                output_dir=output,
                condition_worker=False,
            )
        )
        assert loaded_args.checkpoint == output / "checkpoint.pt"
        assert loaded_args.output_dir == output / "worker"
        assert loaded_args.condition_worker is True

        labels = torch.tensor([0, 1])
        baseline_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        hidden = torch.tensor([[0, 31], [15, 8]], dtype=torch.int32)
        spiking = BrainScaleS2PoolConfig(
            trials=4,
            pool_sizes=(1, 2),
            placements=("same-quadrant",),
            routings=("broadcast",),
        )
        worker_dirs: list[Path] = []
        for pool_size in (1, 2):
            result = MockToyPoolBackend().run_uint5(
                hidden,
                ToyPoolConfig(
                    pool_size=pool_size,
                    logical_neurons=2,
                    inference_trials=2,
                    calibration_trials=4,
                    seed=40 + pool_size,
                    miss_probability=0.0,
                ),
                spiking,
            )
            key = f"ttfs_M{pool_size}_local-pool_dedicated"
            evaluation = ToyConditionEvaluation(
                key=key,
                pool_size=pool_size,
                pooling_domain="ttfs",
                pool_result=result,
                nominal_hidden_uint5=hidden,
                logits=baseline_logits.reshape(1, 2, 2).repeat(2, 1, 1),
                oracle_miss_repair_logits=baseline_logits.reshape(1, 2, 2).repeat(
                    2, 1, 1
                ),
                torch_readout_logits=baseline_logits.reshape(1, 2, 2).repeat(
                    2, 1, 1
                ),
                torch_oracle_miss_repair_logits=baseline_logits.reshape(
                    1, 2, 2
                ).repeat(2, 1, 1),
                pwm_metadata={"first": {"shared": True}, "output": {}},
            )
            worker_dir = output / "condition_workers" / key
            write_toy_artifacts(
                worker_dir,
                labels=labels,
                float_logits=baseline_logits,
                ideal_logits=baseline_logits,
                ideal_hidden_uint5=hidden,
                evaluations=[evaluation],
                manifest={
                    "pool_sizes": [pool_size],
                    "placements": ["local-pool"],
                    "conditions": [
                        {
                            "key": key,
                            "pool_size": pool_size,
                            "pooling_domain": "ttfs",
                            "placement": result.placement,
                            "mapping": result.mapping,
                            "physical_coordinates": result.physical_coordinates,
                            "pool_metadata": result.metadata,
                            "pwm_metadata": evaluation.pwm_metadata,
                        }
                    ],
                },
                runtime={"elapsed_s": float(pool_size)},
                bootstrap_iterations=10,
            )
            (worker_dir / "worker_status.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "attempts": [{"attempt": 1, "status": "passed"}],
                    }
                ),
                encoding="utf-8",
            )
            restored, _, _, _ = _load_isolated_condition(worker_dir)
            assert restored.key == key
            worker_dirs.append(worker_dir)

        first_hidden_dir = output / "condition_workers" / "first_hidden_avg1"
        first_hidden_dir.mkdir(parents=True)
        _aggregate_isolated_conditions(
            SimpleNamespace(
                output_dir=output,
                pool_sizes=[1, 2],
                placements=["local-pool"],
                bootstrap_iterations=10,
                seed=0,
            ),
            worker_dirs,
            {1: first_hidden_dir},
        )
        master = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        isolation = master["condition_process_isolation"]
        assert isolation["enabled"] is True
        assert isolation["worker_count"] == 2
        assert isolation["resumable"] is True
        assert set(isolation["worker_status"]) == {
            "condition_workers/ttfs_M1_local-pool_dedicated",
            "condition_workers/ttfs_M2_local-pool_dedicated",
        }
        with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            conditions = {row["condition"] for row in csv.DictReader(handle)}
        assert "ttfs_M1_local-pool_dedicated" in conditions
        assert "ttfs_M2_local-pool_dedicated" in conditions


def verify_transient_worker_retry() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Transient worker retry]]
    with TemporaryDirectory() as directory:
        worker_dir = Path(directory)
        calls: list[list[str]] = []
        delays: list[float] = []

        def flaky_runner(
            command: list[str], *, cwd: Path, check: bool
        ) -> subprocess.CompletedProcess[str]:
            assert cwd == REPOSITORY_ROOT
            assert check is True
            calls.append(command)
            if len(calls) < 3:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)

        status = _run_worker_command_with_retries(
            ["worker"],
            worker_dir,
            max_attempts=3,
            retry_backoff_s=2.0,
            idle_timeout_s=10.0,
            runner=flaky_runner,
            sleeper=delays.append,
        )
        assert len(calls) == 3
        assert delays == [2.0, 4.0]
        assert status["status"] == "passed"
        assert [attempt["status"] for attempt in status["attempts"]] == [
            "failed",
            "failed",
            "passed",
        ]
        saved = json.loads(
            (worker_dir / "worker_status.json").read_text(encoding="utf-8")
        )
        assert saved == status

        exhausted_dir = worker_dir / "exhausted"
        exhausted_dir.mkdir()

        def failing_runner(
            command: list[str], *, cwd: Path, check: bool
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(7, command)

        try:
            _run_worker_command_with_retries(
                ["worker"],
                exhausted_dir,
                max_attempts=2,
                retry_backoff_s=0.0,
                idle_timeout_s=10.0,
                runner=failing_runner,
                sleeper=lambda _: None,
            )
        except subprocess.CalledProcessError as error:
            assert error.returncode == 7
        else:
            raise AssertionError("exhausted worker retry must re-raise")
        exhausted = json.loads(
            (exhausted_dir / "worker_status.json").read_text(encoding="utf-8")
        )
        assert exhausted["status"] == "failed"
        assert len(exhausted["attempts"]) == 2

        silent_dir = worker_dir / "silent"
        silent_dir.mkdir()
        try:
            _run_worker_command_with_retries(
                [
                    sys.executable,
                    "-c",
                    "import time; print('started', flush=True); time.sleep(30)",
                ],
                silent_dir,
                max_attempts=1,
                retry_backoff_s=0.0,
                idle_timeout_s=0.25,
            )
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError("silent worker must hit its idle watchdog")
        silent = json.loads(
            (silent_dir / "worker_status.json").read_text(encoding="utf-8")
        )
        assert silent["status"] == "failed"
        assert silent["attempts"][0]["failure"] == "idle-timeout"
        assert (silent_dir / "worker_attempt_1.log").is_file()


def verify_metrics_and_artifact_schema() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Network artifact contract]]
    spiking = BrainScaleS2PoolConfig(
        trials=4,
        pool_sizes=(2,),
        placements=("same-quadrant",),
        routings=("broadcast",),
    )
    hidden = torch.tensor([[0, 15], [31, 8]], dtype=torch.int32)
    result = MockToyPoolBackend().run_uint5(
        hidden,
        ToyPoolConfig(
            pool_size=2,
            logical_neurons=2,
            inference_trials=2,
            calibration_trials=4,
            seed=3,
            miss_probability=0.0,
        ),
        spiking,
    )
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]]]
    )
    evaluation = ToyConditionEvaluation(
        key="ttfs_M2_local-pool_dedicated",
        pool_size=2,
        pooling_domain="ttfs",
        pool_result=result,
        nominal_hidden_uint5=hidden,
        logits=logits,
        oracle_miss_repair_logits=logits,
        torch_readout_logits=logits,
        torch_oracle_miss_repair_logits=logits,
        pwm_metadata={"backend": "torch"},
    )
    labels = torch.tensor([0, 1])
    rows = summarize_toy_evaluations(labels, logits[0], logits[0], [evaluation], bootstrap_iterations=10)
    assert rows[-1]["accuracy"] == 1.0
    with TemporaryDirectory() as directory:
        output = Path(directory)
        write_toy_artifacts(
            output,
            labels=labels,
            float_logits=logits[0],
            ideal_logits=logits[0],
            ideal_hidden_uint5=hidden,
            evaluations=[evaluation],
            manifest={"task": "fixture"},
            runtime={"elapsed_s": 0.1},
            bootstrap_iterations=10,
        )
        expected = {
            "manifest.json",
            "runtime.json",
            "metrics.csv",
            "activation_error_by_code.csv",
            "predictions.csv",
            "events.csv",
            "intermediates.pt",
        }
        assert expected.issubset({path.name for path in output.iterdir()})
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["event_csv_coverage"]["full_raw_tensor"] == "intermediates.pt"
        with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            hardware_row = list(csv.DictReader(handle))[-1]
        assert "neuron_miss_rate_nominal_zero" in hardware_row
        assert "nonmiss_activation_mae_uint5" in hardware_row
        assert "oracle_miss_repair_accuracy" in hardware_row
        assert "torch_readout_accuracy" in hardware_row
        with (output / "activation_error_by_code.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            assert len(list(csv.DictReader(handle))) == 32
        with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            prediction_variants = {
                row["analysis_variant"] for row in csv.DictReader(handle)
            }
        assert prediction_variants == {
            "reference",
            "physical",
            "physical-oracle-miss-repair",
            "torch-readout",
            "torch-readout-oracle-miss-repair",
        }
        archive = torch.load(output / "intermediates.pt", weights_only=False)
        condition = archive["conditions"][evaluation.key]
        assert condition["nominal_hidden_uint5"].shape == hidden.shape
        assert condition["torch_readout_logits"].shape == logits.shape
        with (output / "events.csv").open(newline="", encoding="utf-8") as handle:
            event_rows = list(csv.DictReader(handle))
        assert len(event_rows) == 2 * 2 * 2 * 2
        assert "nominal_activation_uint5" in event_rows[0]


def verify_hardware_error_attribution() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#Hardware error attribution]]
    nominal = torch.tensor([[0, 5]], dtype=torch.int32)
    fired = torch.tensor(
        [[[[False], [True]]], [[[True], [True]]]]
    )
    first_spike = torch.where(
        fired,
        torch.full(fired.shape, 10.0e-6, dtype=torch.float64),
        torch.full(fired.shape, torch.nan, dtype=torch.float64),
    )
    all_miss = ~fired.any(dim=-1)
    result = ToyPoolResult(
        first_spike_s=first_spike,
        fired=fired,
        spike_count=fired.to(torch.int64),
        nominal_input_s=torch.tensor([[25.0e-6, 21.0e-6]], dtype=torch.float64),
        pooled_first_spike_s=torch.tensor(
            [[[torch.nan, 10.0e-6]], [[10.0e-6, 10.0e-6]]],
            dtype=torch.float64,
        ),
        decoded_uint5=torch.tensor([[[0, 7]], [[1, 4]]], dtype=torch.int32),
        all_miss=all_miss,
        physical_coordinates=torch.tensor([[0], [1]], dtype=torch.int64),
        pool_size=1,
        placement="local-pool",
        mapping="dedicated",
    )
    primary = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]]])
    physical_oracle = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]])
    torch_readout = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]])
    torch_oracle = torch.tensor([[[2.0, 0.0]], [[2.0, 0.0]]])
    evaluation = ToyConditionEvaluation(
        key="potential_M2_local-pool_dedicated",
        pool_size=2,
        pooling_domain="potential",
        pool_result=result,
        nominal_hidden_uint5=nominal,
        logits=primary,
        oracle_miss_repair_logits=physical_oracle,
        torch_readout_logits=torch_readout,
        torch_oracle_miss_repair_logits=torch_oracle,
        pwm_metadata={"first": {"avg": 2}, "output": {"backend": "hagen-mock"}},
    )
    labels = torch.tensor([0])
    row = summarize_toy_evaluations(
        labels,
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[2.0, 0.0]]),
        [evaluation],
        bootstrap_iterations=10,
    )[-1]
    assert row["hagen_avg"] == 2
    assert row["temporal_pool_size"] == 1
    assert row["neuron_miss_rate_nominal_zero"] == 0.5
    assert row["neuron_miss_rate_nominal_positive"] == 0.0
    assert row["all_miss_rate_nominal_zero"] == 0.5
    assert row["all_miss_rate_nominal_positive"] == 0.0
    assert abs(row["nonmiss_activation_mae_uint5"] - 4.0 / 3.0) < 1.0e-12
    assert abs(row["nonmiss_activation_bias_uint5"] - 2.0 / 3.0) < 1.0e-12
    assert row["oracle_miss_repair_accuracy"] == 0.5
    assert row["torch_readout_accuracy"] == 0.5
    assert row["torch_oracle_miss_repair_accuracy"] == 1.0
    by_code = {item["nominal_code"]: item for item in activation_error_by_code([evaluation])}
    assert by_code[0]["all_miss_rate"] == 0.5
    assert by_code[0]["activation_bias_uint5"] == 1.0
    assert by_code[5]["activation_mae_uint5"] == 1.5
    assert by_code[5]["activation_bias_uint5"] == 0.5

    class FixtureConverted:
        architecture = SimpleNamespace(hidden_features=2, output_features=2)

        @staticmethod
        def output_from_hidden(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = value.to(torch.int8)
            return value.to(torch.int32), logits

    (
        selected_logits,
        selected_oracle_logits,
        torch_logits,
        torch_oracle_logits,
        metadata,
    ) = _evaluate_readout_ablations(
        SimpleNamespace(),
        None,
        FixtureConverted(),  # type: ignore[arg-type]
        result,
        torch.tensor([[9, 5]], dtype=torch.int32),
    )
    torch.testing.assert_close(selected_logits, torch_logits)
    torch.testing.assert_close(selected_oracle_logits, torch_oracle_logits)
    assert selected_oracle_logits[0, 0, 0] == 9
    assert selected_oracle_logits[1, 0, 0] == 1
    assert metadata["oracle_repair_policy"] == (
        "replace-all-miss-position-with-ideal-hidden"
    )

    class FixtureHagen:
        config = SimpleNamespace(mode="mock")

        @staticmethod
        def output_layer(_converted: object, value: torch.Tensor) -> HagenResult:
            return HagenResult(value.to(torch.float32), {"backend": "hagen-mock"})

    physical, physical_oracle, _, _, physical_metadata = _evaluate_readout_ablations(
        SimpleNamespace(hagen_row_chunk_size=64),
        FixtureHagen(),  # type: ignore[arg-type]
        FixtureConverted(),  # type: ignore[arg-type]
        result,
        torch.tensor([[9, 5]], dtype=torch.int32),
    )
    assert physical[0, 0, 0] == 0
    assert physical_oracle[0, 0, 0] == 9
    assert physical_metadata["physical_oracle_miss_repair_reexecuted"] is True
    assert physical_metadata["row_segments"]["physical"] == [0, 2]


def verify_python311_and_notebook_contract() -> None:
    # @lat: [[hardware#Toy ANN2SNN Verification#EBRAINS launcher contract]]
    source_paths = [
        REPOSITORY_ROOT / "utils/hardware/brainscales2/toy.py",
        REPOSITORY_ROOT / "utils/hardware/brainscales2/toy_pooling.py",
        REPOSITORY_ROOT / "utils/hardware/brainscales2/hagen.py",
        REPOSITORY_ROOT / "scripts/evaluation/brainscales2_toy_hil.py",
    ]
    for path in source_paths:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 11))
    notebook = REPOSITORY_ROOT / "scripts/notebooks/ebrains_brainscales2_toy_hil.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    version = payload["metadata"]["language_info"]["version"]
    assert version == "3.11" or version.startswith("3.11.")
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "RUN_TRAIN = True" in source
    assert "RUN_HAGEN_PROBE = True" in source
    assert "RUN_HARDWARE_SMOKE = True" in source
    assert "RUN_YINYANG_FULL = True" in source
    assert "RUN_LOCAL_REPLAY = False" in source
    assert "RUN_MNIST_BENCHMARK = False" in source
    assert "brainscales2_toy_hil.py" in source
    assert "--max-test-samples" in source
    assert "save_nightly_calibration" in source
    assert "hagen_cocolist.pbin" in source
    assert "spiking_cocolist.pbin" in source
    assert "HAGEN_CALIBRATION_PATH is None" in source
    assert "SPIKING_CALIBRATION_PATH is None" in source
    assert "pipeline_status.json" in source
    assert "Using probe-selected HAGEN_HIDDEN_SHIFT" in source
    assert "formal stages require a passing same-run smoke gate" in source
    assert "SMOKE_MAX_MULTI_SPIKE_RATE" in source
    assert "SPIKING_INPUT_FAN_IN = 4" in source
    assert "'--input-fan-in', SPIKING_INPUT_FAN_IN" in source
    assert "RELU_BOUNDARY = 'implicit-lower-bound-host'" in source
    assert "'--relu-boundary', RELU_BOUNDARY" in source
    assert "TOY_ACTIVATION = 'relu'" in source
    assert "'--activation', TOY_ACTIVATION" in source
    assert "POOLING_DOMAIN = 'potential'" in source
    assert "'--pooling-domain', POOLING_DOMAIN" in source
    assert "Smoke did not use Hagen avg=M with one LIF" in source
    assert "activation_error_by_code.csv" in source
    assert "oracle_miss_repair_accuracy" in source
    assert "torch_readout_accuracy" in source
    assert "POOL_SAMPLE_CHUNK_SIZE = 64" in source
    assert "'--pool-sample-chunk-size', POOL_SAMPLE_CHUNK_SIZE" in source
    assert "POOL_REPLICA_SAMPLE_BUDGET = 128" in source
    assert "'--pool-replica-sample-budget', POOL_REPLICA_SAMPLE_BUDGET" in source
    assert "POOL_CALIBRATION_TRIAL_CHUNK_SIZE = 4" in source
    assert (
        "'--pool-calibration-trial-chunk-size', "
        "POOL_CALIBRATION_TRIAL_CHUNK_SIZE"
    ) in source
    assert "HAGEN_ROW_CHUNK_SIZE = 512" in source
    assert "CONDITION_WORKER_MAX_ATTEMPTS = 3" in source
    assert "CONDITION_WORKER_RETRY_BACKOFF_S = 20.0" in source
    assert "CONDITION_WORKER_IDLE_TIMEOUT_S = 180.0" in source
    assert "ARTIFACT_ROOT = None" in source
    assert "if ARTIFACT_ROOT is not None" in source
    assert "'--hagen-row-chunk-size', HAGEN_ROW_CHUNK_SIZE" in source
    assert "'--condition-worker-max-attempts', CONDITION_WORKER_MAX_ATTEMPTS" in source
    assert (
        "'--condition-worker-retry-backoff-s', "
        "CONDITION_WORKER_RETRY_BACKOFF_S"
    ) in source
    assert (
        "'--condition-worker-idle-timeout-s', "
        "CONDITION_WORKER_IDLE_TIMEOUT_S"
    ) in source
    assert source.index("'--phase', 'train'") < source.index(
        "setup_hardware_client()"
    )
    assert source.index("if RUN_HARDWARE_SMOKE:") < source.index(
        "if RUN_YINYANG_FULL:"
    )


def main() -> None:
    verify_deterministic_yin_yang_splits()
    verify_frozen_integer_conversion()
    verify_grouped_placement()
    verify_grouped_broadcast_fan_in()
    verify_mock_and_all_miss_policy()
    verify_mean_and_corrected_max_estimators()
    verify_chunked_pool_aggregation()
    verify_shared_split_timing_calibration()
    verify_m16_calibration_inference_batch_separation()
    verify_pool_size_aware_hardware_chunk_cap()
    verify_pool_chunk_process_isolation()
    verify_replay_split_and_reproducibility()
    verify_hagen_output_row_chunking()
    verify_hagen_host_tiling()
    verify_host_mediated_implicit_relu_boundary()
    verify_sigmoid_host_activation_adapter()
    verify_condition_process_isolation_contract()
    verify_transient_worker_retry()
    verify_metrics_and_artifact_schema()
    verify_hardware_error_attribution()
    verify_python311_and_notebook_contract()
    print("BrainScaleS-2 toy ANN2SNN verification passed")


if __name__ == "__main__":
    main()
