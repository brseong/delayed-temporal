"""Network-level TTFS pooling backends for converted toy classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol
import math

import torch

from utils.transforms.types import Potential, PotentialBounds

from .backend import (
    _configure_experiment_calibration,
    _find_raw_spikes,
    _fpga_time_scale_s,
    _legacy_experiment_observables,
    _logical_neuron_coordinates,
    _raw_events_to_tensors,
    calibration_sha256,
)
from .config import BrainScaleS2PoolConfig
from .encoding import encode_potential_for_brainscales2


NetworkPlacement = Literal["local-pool", "cross-quadrant"]
PoolMapping = Literal["dedicated", "time-multiplexed"]


@dataclass(frozen=True)
class ToyPoolConfig:
    """Shape, calibration, and synthetic-noise settings for hidden pooling."""

    pool_size: int
    logical_neurons: int
    placement: NetworkPlacement = "local-pool"
    mapping: PoolMapping = "dedicated"
    inference_trials: int = 8
    calibration_trials: int = 32
    seed: int = 0
    local_std_s: float = 0.8e-6
    shared_std_s: float = 0.25e-6
    static_std_s: float = 0.5e-6
    miss_probability: float = 0.01

    def __post_init__(self) -> None:
        if not 1 <= self.pool_size <= 128:
            raise ValueError("pool_size must lie in [1, 128]")
        if self.logical_neurons <= 0:
            raise ValueError("logical_neurons must be positive")
        if self.mapping == "dedicated" and self.logical_neurons * self.pool_size > 512:
            raise ValueError("dedicated mapping exceeds the 512-neuron chip")
        if self.placement not in ("local-pool", "cross-quadrant"):
            raise ValueError("unsupported network placement")
        if self.mapping not in ("dedicated", "time-multiplexed"):
            raise ValueError("unsupported pool mapping")
        if self.inference_trials <= 0 or self.calibration_trials < 2:
            raise ValueError("pool trial counts must be positive")
        if not 0.0 <= self.miss_probability <= 1.0:
            raise ValueError("miss_probability must lie in [0, 1]")


@dataclass(frozen=True)
class TimingCalibration:
    """Calibration-only global delay and persistent replica offsets."""

    response_delay_s: float
    neuron_offset_s: torch.Tensor
    calibration_trials: int


@dataclass(frozen=True)
class ToyPoolResult:
    """Raw and decoded hidden observations for one network condition."""

    first_spike_s: torch.Tensor
    fired: torch.Tensor
    spike_count: torch.Tensor
    nominal_input_s: torch.Tensor
    pooled_first_spike_s: torch.Tensor
    decoded_uint5: torch.Tensor
    all_miss: torch.Tensor
    physical_coordinates: torch.Tensor
    pool_size: int
    placement: NetworkPlacement
    mapping: PoolMapping
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = self.first_spike_s.shape
        if len(expected) != 4:
            raise ValueError("first_spike_s must have shape [trial, sample, logical, replica]")
        if self.fired.shape != expected or self.spike_count.shape != expected:
            raise ValueError("raw event tensors must have identical shapes")
        if expected[-1] != self.pool_size:
            raise ValueError("raw replica dimension does not match pool_size")
        if self.nominal_input_s.shape != expected[1:3]:
            raise ValueError("nominal_input_s must have shape [sample, logical]")
        if self.pooled_first_spike_s.shape != expected[:3]:
            raise ValueError("pooled_first_spike_s has an invalid shape")
        if self.decoded_uint5.shape != expected[:3] or self.all_miss.shape != expected[:3]:
            raise ValueError("decoded tensors have an invalid shape")
        if self.physical_coordinates.shape != expected[2:4]:
            raise ValueError("physical coordinate grid has an invalid shape")


def concatenate_toy_pool_results(
    results: list[ToyPoolResult],
) -> ToyPoolResult:
    """Concatenate sample chunks while preserving raw events and provenance."""
    if not results:
        raise ValueError("at least one pool result is required")
    reference = results[0]
    sample_chunks: list[dict[str, Any]] = []
    sample_start = 0
    for result in results:
        if (
            result.pool_size != reference.pool_size
            or result.placement != reference.placement
            or result.mapping != reference.mapping
            or result.first_spike_s.shape[0] != reference.first_spike_s.shape[0]
            or result.first_spike_s.shape[2:] != reference.first_spike_s.shape[2:]
            or not torch.equal(
                result.physical_coordinates, reference.physical_coordinates
            )
        ):
            raise ValueError("pool result chunks do not share one condition")
        sample_stop = sample_start + result.first_spike_s.shape[1]
        sample_chunks.append(
            {
                "sample_start": sample_start,
                "sample_stop": sample_stop,
                "metadata": result.metadata,
            }
        )
        sample_start = sample_stop
    metadata = dict(reference.metadata)
    metadata.pop("response_delay_s", None)
    metadata.update(
        {
            "chunked": len(results) > 1,
            "sample_chunk_count": len(results),
            "calibration_strategy": "per-sample-chunk",
            "sample_chunks": sample_chunks,
        }
    )
    return ToyPoolResult(
        first_spike_s=torch.cat([result.first_spike_s for result in results], dim=1),
        fired=torch.cat([result.fired for result in results], dim=1),
        spike_count=torch.cat([result.spike_count for result in results], dim=1),
        nominal_input_s=torch.cat(
            [result.nominal_input_s for result in results], dim=0
        ),
        pooled_first_spike_s=torch.cat(
            [result.pooled_first_spike_s for result in results], dim=1
        ),
        decoded_uint5=torch.cat(
            [result.decoded_uint5 for result in results], dim=1
        ),
        all_miss=torch.cat([result.all_miss for result in results], dim=1),
        physical_coordinates=reference.physical_coordinates,
        pool_size=reference.pool_size,
        placement=reference.placement,
        mapping=reference.mapping,
        metadata=metadata,
    )


class ToyTemporalPoolBackend(Protocol):
    """Common hidden-activation pooling boundary."""

    def run_uint5(
        self,
        hidden_uint5: torch.Tensor,
        config: ToyPoolConfig,
        spiking_config: BrainScaleS2PoolConfig,
    ) -> ToyPoolResult: ...


def resolve_grouped_physical_coordinates(
    logical_neurons: int,
    pool_size: int,
    placement: NetworkPlacement,
    mapping: PoolMapping,
) -> torch.Tensor:
    """Allocate a unique dedicated grid or a repeated time-multiplexed pool."""
    ToyPoolConfig(
        pool_size=pool_size,
        logical_neurons=logical_neurons,
        placement=placement,
        mapping=mapping,
        inference_trials=1,
        calibration_trials=2,
    )
    if mapping == "time-multiplexed":
        if placement == "local-pool":
            one_pool = torch.arange(pool_size, dtype=torch.long)
        else:
            quadrant_counters = [0, 0, 0, 0]
            coordinates: list[int] = []
            for replica in range(pool_size):
                quadrant = replica % 4
                coordinates.append(128 * quadrant + quadrant_counters[quadrant])
                quadrant_counters[quadrant] += 1
            one_pool = torch.tensor(coordinates, dtype=torch.long)
        return one_pool.reshape(1, -1).repeat(logical_neurons, 1)

    result = torch.empty((logical_neurons, pool_size), dtype=torch.long)
    if placement == "local-pool":
        quadrant_offsets = [0, 0, 0, 0]
        for logical in range(logical_neurons):
            quadrant = logical % 4
            start = quadrant_offsets[quadrant]
            stop = start + pool_size
            if stop > 128:
                raise ValueError("local-pool placement exceeds a quadrant capacity")
            result[logical] = torch.arange(
                128 * quadrant + start,
                128 * quadrant + stop,
                dtype=torch.long,
            )
            quadrant_offsets[quadrant] = stop
    else:
        quadrant_offsets = [0, 0, 0, 0]
        for logical in range(logical_neurons):
            for replica in range(pool_size):
                quadrant = replica % 4
                offset = quadrant_offsets[quadrant]
                if offset >= 128:
                    raise ValueError("cross-quadrant placement exceeds a quadrant capacity")
                result[logical, replica] = 128 * quadrant + offset
                quadrant_offsets[quadrant] += 1
    if torch.unique(result).numel() != result.numel():
        raise RuntimeError("dedicated placement produced duplicate physical neurons")
    return result


def _grouped_input_channel_slice(
    logical: int,
    config: ToyPoolConfig,
    input_fan_in: int,
) -> slice:
    """Return the simultaneous source lanes driving one logical pool."""
    if not 0 <= logical < config.logical_neurons:
        raise ValueError("logical input index is out of range")
    if input_fan_in <= 0:
        raise ValueError("input_fan_in must be positive")
    if config.mapping == "dedicated":
        start = logical * input_fan_in
        return slice(start, start + input_fan_in)
    return slice(0, input_fan_in)


def _configure_grouped_synapse_weights(
    weight: torch.Tensor,
    config: ToyPoolConfig,
    *,
    input_fan_in: int,
    synaptic_weight: float,
) -> None:
    """Connect every logical source lane only to its physical replica block."""
    expected_inputs = (
        config.logical_neurons * input_fan_in
        if config.mapping == "dedicated"
        else input_fan_in
    )
    expected_outputs = (
        config.logical_neurons * config.pool_size
        if config.mapping == "dedicated"
        else config.pool_size
    )
    if tuple(weight.shape) != (expected_outputs, expected_inputs):
        raise ValueError(
            "grouped synapse weight shape does not match mapping and fan-in"
        )
    weight.zero_()
    if config.mapping == "dedicated":
        for logical in range(config.logical_neurons):
            output_start = logical * config.pool_size
            input_lanes = _grouped_input_channel_slice(
                logical, config, input_fan_in
            )
            weight[
                output_start : output_start + config.pool_size,
                input_lanes,
            ] = synaptic_weight
    else:
        weight[:, :input_fan_in] = synaptic_weight


def _nanmean(value: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    finite = torch.isfinite(value)
    count = finite.sum(dim=dim)
    total = torch.where(finite, value, torch.zeros_like(value)).sum(dim=dim)
    return torch.where(
        count > 0,
        total / count.clamp_min(1),
        torch.full_like(total, torch.nan),
    )


def calibrate_timing(
    first_spike_s: torch.Tensor,
    nominal_input_s: torch.Tensor,
) -> TimingCalibration:
    """Estimate response delay and persistent offsets without evaluation events."""
    if first_spike_s.ndim != 4:
        raise ValueError("calibration spikes must have shape [trial, code, logical, replica]")
    if nominal_input_s.ndim != 1 or nominal_input_s.numel() != first_spike_s.shape[1]:
        raise ValueError("calibration code times do not match spike observations")
    residual = first_spike_s - nominal_input_s.reshape(1, -1, 1, 1)
    response_delay = float(_nanmean(residual, dim=(0, 1, 2, 3)))
    offsets = _nanmean(residual - response_delay, dim=(0, 1))
    offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
    offset_mean = float(offsets.mean())
    offsets = offsets - offset_mean
    return TimingCalibration(
        response_delay_s=response_delay + offset_mean,
        neuron_offset_s=offsets,
        calibration_trials=first_spike_s.shape[0],
    )


def _nominal_uint5_times(
    value: torch.Tensor,
    spiking_config: BrainScaleS2PoolConfig,
) -> torch.Tensor:
    potential = Potential(value.to(torch.float32), PotentialBounds(0.0, 31.0))
    encoding = encode_potential_for_brainscales2(
        potential,
        spiking_config,
        pool_size=1,
        routing="broadcast",
    )
    return encoding.injected_time_s.reshape(value.shape)


def decode_pool_observations(
    first_spike_s: torch.Tensor,
    nominal_input_s: torch.Tensor,
    calibration: TimingCalibration,
    coordinates: torch.Tensor,
    config: ToyPoolConfig,
    spiking_config: BrainScaleS2PoolConfig,
    *,
    spike_count: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToyPoolResult:
    """Apply calibrated mean pooling, inverse identity code, and all-miss zero."""
    offsets = calibration.neuron_offset_s
    if offsets.shape != coordinates.shape:
        raise ValueError("calibration offsets and physical coordinates must match")
    corrected = (
        first_spike_s
        - offsets.reshape(1, 1, *offsets.shape)
        - calibration.response_delay_s
    )
    pooled = _nanmean(corrected, dim=-1)
    all_miss = ~torch.isfinite(pooled)
    width = spiking_config.input_late_s - spiking_config.input_early_s
    decoded = (
        (spiking_config.input_late_s - pooled) / width * 31.0
    ).clamp(0.0, 31.0)
    decoded = torch.where(all_miss, torch.zeros_like(decoded), torch.round(decoded))
    fired = torch.isfinite(first_spike_s)
    if spike_count is None:
        spike_count = fired.to(torch.int64)
    return ToyPoolResult(
        first_spike_s=first_spike_s,
        fired=fired,
        spike_count=spike_count,
        nominal_input_s=nominal_input_s,
        pooled_first_spike_s=pooled,
        decoded_uint5=decoded.to(torch.int32),
        all_miss=all_miss,
        physical_coordinates=coordinates,
        pool_size=config.pool_size,
        placement=config.placement,
        mapping=config.mapping,
        metadata={
            "response_delay_s": calibration.response_delay_s,
            "calibration_trials": calibration.calibration_trials,
            **(metadata or {}),
        },
    )


def _synthetic_spikes(
    nominal: torch.Tensor,
    coordinates: torch.Tensor,
    config: ToyPoolConfig,
    *,
    trials: int,
    generator: torch.Generator,
    static_offset_s: torch.Tensor | None = None,
) -> torch.Tensor:
    logical, replicas = coordinates.shape
    if nominal.ndim == 1:
        nominal_grid = nominal.reshape(1, -1, 1, 1)
        samples = nominal.numel()
    elif nominal.ndim == 2:
        nominal_grid = nominal.reshape(1, nominal.shape[0], logical, 1)
        samples = nominal.shape[0]
    else:
        raise ValueError("nominal times must be code vector or [sample, logical]")
    if static_offset_s is None:
        static_by_coordinate = torch.randn(
            512, generator=generator, dtype=torch.float64
        )
        static = static_by_coordinate[coordinates] * config.static_std_s
        if config.mapping == "time-multiplexed":
            static = static[0:1].repeat(logical, 1)
    else:
        if static_offset_s.shape != coordinates.shape:
            raise ValueError("static_offset_s must match the physical coordinate grid")
        static = static_offset_s
    shared = torch.randn(
        (trials, samples, 1, 1), generator=generator, dtype=torch.float64
    ) * config.shared_std_s
    local = torch.randn(
        (trials, samples, logical, replicas),
        generator=generator,
        dtype=torch.float64,
    ) * config.local_std_s
    first = nominal_grid + 5.0e-6 + static.reshape(1, 1, logical, replicas) + shared + local
    miss = torch.rand(first.shape, generator=generator) < config.miss_probability
    return torch.where(miss, torch.full_like(first, torch.nan), first)


class MockToyPoolBackend:
    """Seeded network-level static/shared/local timing simulation."""

    def run_uint5(
        self,
        hidden_uint5: torch.Tensor,
        config: ToyPoolConfig,
        spiking_config: BrainScaleS2PoolConfig,
    ) -> ToyPoolResult:
        if hidden_uint5.shape[1] != config.logical_neurons:
            raise ValueError("hidden activation count does not match pool config")
        coordinates = resolve_grouped_physical_coordinates(
            config.logical_neurons,
            config.pool_size,
            config.placement,
            config.mapping,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        static_by_coordinate = torch.randn(
            512, generator=generator, dtype=torch.float64
        )
        static = static_by_coordinate[coordinates] * config.static_std_s
        if config.mapping == "time-multiplexed":
            static = static[0:1].repeat(config.logical_neurons, 1)
        code_values = torch.linspace(0.0, 31.0, 11)
        code_times = _nominal_uint5_times(code_values, spiking_config)
        calibration_first = _synthetic_spikes(
            code_times,
            coordinates,
            config,
            trials=config.calibration_trials,
            generator=generator,
            static_offset_s=static,
        )
        calibration = calibrate_timing(calibration_first, code_times)
        nominal = _nominal_uint5_times(hidden_uint5, spiking_config)
        first = _synthetic_spikes(
            nominal,
            coordinates,
            config,
            trials=config.inference_trials,
            generator=generator,
            static_offset_s=static,
        )
        return decode_pool_observations(
            first,
            nominal,
            calibration,
            coordinates,
            config,
            spiking_config,
            metadata={"backend": "mock", "seed": config.seed},
        )


class ReplayToyPoolBackend:
    """Sample held-out residuals from an accepted primitive hardware artifact."""

    def __init__(self, events_path: Path) -> None:
        self.events_path = events_path
        if not events_path.is_file():
            raise FileNotFoundError(events_path)

    def run_uint5(
        self,
        hidden_uint5: torch.Tensor,
        config: ToyPoolConfig,
        spiking_config: BrainScaleS2PoolConfig,
    ) -> ToyPoolResult:
        payload = torch.load(self.events_path, map_location="cpu", weights_only=False)
        primitive_placement = (
            "same-quadrant" if config.placement == "local-pool" else "cross-quadrant"
        )
        key = f"M{config.pool_size}_{primitive_placement}_broadcast"
        if key not in payload:
            raise KeyError(f"replay artifact does not contain {key}")
        source = payload[key]
        observed = source["first_spike_s"].to(torch.float64)
        code_times = source["nominal_input_s"].to(torch.float64)
        split = max(1, observed.shape[0] // 2)
        logical = config.logical_neurons
        calibration_source = observed[:split].unsqueeze(2).repeat(1, 1, logical, 1)
        calibration = calibrate_timing(calibration_source, code_times)
        coordinates = resolve_grouped_physical_coordinates(
            logical,
            config.pool_size,
            config.placement,
            config.mapping,
        )
        nominal = _nominal_uint5_times(hidden_uint5, spiking_config)
        nearest_code = (nominal.unsqueeze(-1) - code_times).abs().argmin(dim=-1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        evaluation = observed[split:]
        if evaluation.shape[0] == 0:
            raise ValueError("replay artifact has no held-out trials")
        trial_index = torch.randint(
            evaluation.shape[0],
            (config.inference_trials, *nearest_code.shape),
            generator=generator,
        )
        first = torch.empty(
            (config.inference_trials, *nearest_code.shape, config.pool_size),
            dtype=torch.float64,
        )
        source_offsets = calibration.neuron_offset_s[0]
        for trial in range(config.inference_trials):
            sampled = evaluation[trial_index[trial], nearest_code]
            noise = (
                sampled
                - code_times[nearest_code].unsqueeze(-1)
                - calibration.response_delay_s
                - source_offsets.reshape(1, 1, -1)
            )
            first[trial] = (
                nominal.unsqueeze(-1)
                + calibration.response_delay_s
                + source_offsets.reshape(1, 1, -1)
                + noise
            )
        replay_calibration = TimingCalibration(
            response_delay_s=calibration.response_delay_s,
            neuron_offset_s=source_offsets.reshape(1, -1).repeat(logical, 1),
            calibration_trials=split,
        )
        return decode_pool_observations(
            first,
            nominal,
            replay_calibration,
            coordinates,
            config,
            spiking_config,
            metadata={
                "backend": "replay",
                "events_path": str(self.events_path),
                "source_condition": key,
                "held_out_trials": evaluation.shape[0],
                "scope": "rough-model-only",
            },
        )


class GroupedHardwarePoolBackend:
    """Lazy hxtorch grouped-broadcast backend for dedicated or reused pools."""

    @staticmethod
    def dependencies_available() -> bool:
        try:
            import_module("hxtorch")
            import_module("hxtorch.spiking")
        except ImportError:
            return False
        return True

    def run_uint5(
        self,
        hidden_uint5: torch.Tensor,
        config: ToyPoolConfig,
        spiking_config: BrainScaleS2PoolConfig,
    ) -> ToyPoolResult:
        spiking_config.require_reproducible_calibration()
        if hidden_uint5.shape[1] != config.logical_neurons:
            raise ValueError("hidden activation count does not match pool config")
        try:
            hxtorch = import_module("hxtorch")
            hxsnn = import_module("hxtorch.spiking")
        except ImportError as error:
            raise RuntimeError("grouped pooling requires EBRAINS hxtorch") from error

        coordinates = resolve_grouped_physical_coordinates(
            config.logical_neurons,
            config.pool_size,
            config.placement,
            config.mapping,
        )
        unique_coordinates = (
            coordinates.reshape(-1)
            if config.mapping == "dedicated"
            else coordinates[0]
        )
        output_neurons = unique_coordinates.numel()
        input_channels = (
            config.logical_neurons * spiking_config.input_fan_in
            if config.mapping == "dedicated"
            else spiking_config.input_fan_in
        )
        code_values = torch.linspace(0.0, 31.0, 11)
        code_times = _nominal_uint5_times(code_values, spiking_config)
        nominal = _nominal_uint5_times(hidden_uint5, spiking_config)

        calibration_batches = config.calibration_trials * code_times.numel()
        if config.mapping == "dedicated":
            inference_entries = config.inference_trials * hidden_uint5.shape[0]
        else:
            inference_entries = (
                config.inference_trials * hidden_uint5.shape[0] * config.logical_neurons
            )
        total_batches = calibration_batches + inference_entries
        inputs = torch.zeros(
            (spiking_config.runtime_steps, total_batches, input_channels),
            dtype=torch.float32,
        )
        batch = 0
        for _ in range(config.calibration_trials):
            for time_s in code_times:
                step = int(round(float(time_s) / spiking_config.dt_s))
                inputs[step, batch, :] = 1.0
                batch += 1
        if config.mapping == "dedicated":
            for _ in range(config.inference_trials):
                for sample in range(hidden_uint5.shape[0]):
                    for logical in range(config.logical_neurons):
                        step = int(round(float(nominal[sample, logical]) / spiking_config.dt_s))
                        input_lanes = _grouped_input_channel_slice(
                            logical, config, spiking_config.input_fan_in
                        )
                        inputs[step, batch, input_lanes] = 1.0
                    batch += 1
        else:
            for _ in range(config.inference_trials):
                for sample in range(hidden_uint5.shape[0]):
                    for logical in range(config.logical_neurons):
                        step = int(round(float(nominal[sample, logical]) / spiking_config.dt_s))
                        input_lanes = _grouped_input_channel_slice(
                            logical, config, spiking_config.input_fan_in
                        )
                        inputs[step, batch, input_lanes] = 1.0
                        batch += 1
        if batch != total_batches:
            raise RuntimeError("grouped input construction lost batch entries")

        initialized = False
        try:
            hxtorch.init_hardware()
            initialized = True
            experiment = hxsnn.Experiment(dt=spiking_config.dt_s)
            experiment.inter_batch_entry_wait = int(
                round(spiking_config.inter_batch_wait_s / _fpga_time_scale_s())
            )
            calibration_loader = None
            if spiking_config.calibration_path is not None:
                calibration_loader = _configure_experiment_calibration(
                    experiment, spiking_config.calibration_path
                )
            synapse = hxsnn.Synapse(
                in_features=input_channels,
                out_features=output_neurons,
                experiment=experiment,
            )
            _configure_grouped_synapse_weights(
                synapse.weight.data,
                config,
                input_fan_in=spiking_config.input_fan_in,
                synaptic_weight=spiking_config.synaptic_weight,
            )
            lif = hxsnn.LIF(
                size=output_neurons,
                experiment=experiment,
                tau_mem=spiking_config.tau_mem_s,
                tau_syn=spiking_config.tau_syn_s,
                leak=spiking_config.leak,
                reset=spiking_config.reset,
                threshold=spiking_config.threshold,
                refractory_time=spiking_config.refractory_time_s,
                i_synin_gm=spiking_config.i_synin_gm,
                synapse_dac_bias=spiking_config.synapse_dac_bias,
                placement_constraint=_logical_neuron_coordinates(
                    tuple(int(value) for value in unique_coordinates.tolist())
                ),
                enable_spike_recording=True,
                enable_cadc_recording=False,
                enable_madc_recording=False,
            )
            observables = lif(synapse(hxsnn.LIFObservables(spikes=inputs)))
            run_output = hxsnn.run(experiment, spiking_config.runtime_steps)
            raw, raw_api = _find_raw_spikes(
                _legacy_experiment_observables(experiment, lif),
                lif,
                observables,
                run_output,
                experiment,
            )
            first, _, count = _raw_events_to_tensors(
                raw,
                batch_count=total_batches,
                pool_size=output_neurons,
                raw_time_scale_s=spiking_config.raw_time_scale_s,
                deadline_s=spiking_config.observation_deadline_s,
            )
            calibration_raw = first[:calibration_batches]
            if config.mapping == "dedicated":
                calibration_first = calibration_raw.reshape(
                    config.calibration_trials,
                    code_times.numel(),
                    config.logical_neurons,
                    config.pool_size,
                )
                inference_first = first[calibration_batches:].reshape(
                    config.inference_trials,
                    hidden_uint5.shape[0],
                    config.logical_neurons,
                    config.pool_size,
                )
                inference_count = count[calibration_batches:].reshape(inference_first.shape)
            else:
                base_calibration = calibration_raw.reshape(
                    config.calibration_trials,
                    code_times.numel(),
                    1,
                    config.pool_size,
                )
                calibration_first = base_calibration.repeat(
                    1, 1, config.logical_neurons, 1
                )
                inference_first = first[calibration_batches:].reshape(
                    config.inference_trials,
                    hidden_uint5.shape[0],
                    config.logical_neurons,
                    config.pool_size,
                )
                inference_count = count[calibration_batches:].reshape(inference_first.shape)
            timing = calibrate_timing(calibration_first, code_times)
            chip_identifier = None
            get_identifier = getattr(hxtorch, "get_unique_identifier", None)
            if callable(get_identifier):
                chip_identifier = [str(value) for value in get_identifier()]
            return decode_pool_observations(
                inference_first,
                nominal,
                timing,
                coordinates,
                config,
                spiking_config,
                spike_count=inference_count,
                metadata={
                    "backend": "hardware",
                    "hxtorch_version": getattr(hxtorch, "__version__", "unknown"),
                    "chip_identifier": chip_identifier,
                    "calibration_loader": calibration_loader,
                    "calibration_sha256": calibration_sha256(
                        spiking_config.calibration_path
                    ),
                    "input_fan_in": spiking_config.input_fan_in,
                    "raw_spike_api": raw_api,
                    "grouped_broadcast": True,
                },
            )
        finally:
            if initialized:
                hxtorch.release_hardware()
