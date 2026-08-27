"""Physical and mock execution backends for BrainScaleS-2 TTFS pooling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol
import math

import torch

from utils.transforms.types import Potential

from .config import (
    BrainScaleS2PoolConfig,
    CADCDiagnosticResult,
    PlacementMode,
    PoolRunResult,
    RoutingMode,
)
from .encoding import encode_potential_for_brainscales2


class PoolBackend(Protocol):
    """Common execution contract shared by the physical and mock backends."""

    def run(
        self,
        potential: Potential,
        config: BrainScaleS2PoolConfig,
        *,
        pool_size: int,
        placement: PlacementMode,
        routing: RoutingMode,
    ) -> PoolRunResult: ...


def resolve_physical_neuron_indices(
    pool_size: int,
    placement: PlacementMode,
) -> tuple[int, ...]:
    """Choose deterministic atomic-neuron indices for a placement ablation."""
    if pool_size <= 0 or pool_size > 128:
        raise ValueError("pool_size must lie in [1, 128]")
    if placement == "same-quadrant":
        coordinates = tuple(range(pool_size))
    elif placement == "cross-quadrant":
        quadrant_starts = (0, 128, 256, 384)
        coordinates = tuple(
            quadrant_starts[index % 4] + index // 4
            for index in range(pool_size)
        )
    else:
        raise ValueError("unsupported placement mode")
    if len(set(coordinates)) != pool_size or any(
        not 0 <= coordinate < 512 for coordinate in coordinates
    ):
        raise RuntimeError("placement resolver produced invalid neuron coordinates")
    return coordinates


def calibration_sha256(path: Path | None) -> str | None:
    """Hash a pinned calibration file for experiment provenance."""
    if path is None:
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MockPoolBackend:
    """Seeded synthetic backend with explicit static, shared, and local noise."""

    def run(
        self,
        potential: Potential,
        config: BrainScaleS2PoolConfig,
        *,
        pool_size: int,
        placement: PlacementMode,
        routing: RoutingMode,
    ) -> PoolRunResult:
        encoding = encode_potential_for_brainscales2(
            potential,
            config,
            pool_size=pool_size,
            routing=routing,
        )
        coordinates = resolve_physical_neuron_indices(pool_size, placement)

        condition_seed = (
            config.seed
            + 1009 * pool_size
            + (7919 if placement == "cross-quadrant" else 0)
            + (104729 if routing == "independent" else 0)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(condition_seed)

        trials = config.trials
        samples = encoding.sample_count
        shape = (trials, samples, pool_size)
        static_offset = torch.randn(
            (1, 1, pool_size), generator=generator, dtype=torch.float64
        ) * config.mock_static_std_s
        shared_scale = config.mock_shared_std_s
        if placement == "cross-quadrant":
            shared_scale *= 0.75
        if routing == "independent":
            shared_scale *= 0.5
        shared = torch.randn(
            (trials, samples, 1), generator=generator, dtype=torch.float64
        ) * shared_scale
        local = torch.randn(shape, generator=generator, dtype=torch.float64)
        local *= config.mock_local_std_s

        first_spike = (
            encoding.injected_time_s.reshape(1, samples, 1)
            + config.mock_response_delay_s
            + static_offset
            + shared
            + local
        )
        random_miss = (
            torch.rand(shape, generator=generator) < config.mock_miss_probability
        )
        fired = (~random_miss) & (first_spike >= 0.0) & (
            first_spike <= config.observation_deadline_s
        )
        spike_count = fired.to(torch.int64)
        first_spike = torch.where(
            fired,
            first_spike,
            torch.full_like(first_spike, torch.nan),
        )

        return PoolRunResult(
            first_spike_s=first_spike,
            fired=fired,
            spike_count=spike_count,
            nominal_input_s=encoding.injected_time_s,
            ideal_input_s=encoding.ideal_time_s,
            physical_coordinates=coordinates,
            pool_size=pool_size,
            placement=placement,
            routing=routing,
            original_input_shape=encoding.original_shape,
            metadata={
                "backend": "mock",
                "condition_seed": condition_seed,
                "clamped_values": int(encoding.clamp_mask.sum().item()),
            },
        )


def _walk_raw_candidates(root: Any, *, max_depth: int = 3) -> Iterable[Any]:
    """Traverse a small result-object surface looking for a raw spike handle."""
    seen: set[int] = set()
    frontier: list[tuple[Any, int]] = [(root, 0)]
    while frontier:
        value, depth = frontier.pop(0)
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        if depth >= max_depth:
            continue
        if isinstance(value, dict):
            frontier.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, (tuple, list)):
            frontier.extend((child, depth + 1) for child in value)
        else:
            for name in (
                "spikes",
                "hw_data",
                "hw_observables",
                "hardware_data",
                "observables",
                "data",
                "handle",
            ):
                if hasattr(value, name):
                    frontier.append((getattr(value, name), depth + 1))


def _find_raw_spikes(*roots: Any) -> tuple[Any, str]:
    for root in roots:
        for candidate in _walk_raw_candidates(root):
            to_raw = getattr(candidate, "to_raw", None)
            if callable(to_raw):
                return (
                    to_raw(),
                    f"{type(candidate).__module__}.{type(candidate).__name__}.to_raw",
                )
            get_data = getattr(candidate, "get_data", None)
            if callable(get_data) and "spike" in type(candidate).__name__.lower():
                return (
                    get_data(),
                    f"{type(candidate).__module__}.{type(candidate).__name__}.get_data",
                )
    raise RuntimeError(
        "the installed hxtorch release did not expose a raw SpikeHandle; "
        "dense-grid fallback is intentionally disabled for jitter measurements"
    )


def _legacy_experiment_observables(experiment: Any, lif: Any) -> Any | None:
    """Return raw observables retained by pre-13 EBRAINS hxtorch releases."""
    extractor = getattr(experiment, "_hw_data_extractor", None)
    descriptor = getattr(lif, "descriptor", None)
    get_observables = getattr(extractor, "get", None)
    if not callable(get_observables) or descriptor is None:
        return None
    return get_observables(descriptor)


def _fpga_time_scale_s() -> float:
    """Resolve the installed grenade FPGA tick duration in seconds."""
    errors: list[str] = []
    for module_name in ("pygrenade_vx", "pygrenade_vx_v3"):
        try:
            module = import_module(module_name)
            cycles_per_us = float(module.common.Time.fpga_clock_cycles_per_us)
            if math.isfinite(cycles_per_us) and cycles_per_us > 0.0:
                return 1.0e-6 / cycles_per_us
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            errors.append(f"{module_name}: {error}")
    raise RuntimeError(
        "cannot resolve the FPGA tick duration from the installed grenade API: "
        + "; ".join(errors)
    )


def _raw_events_to_tensors(
    raw: Any,
    *,
    batch_count: int,
    pool_size: int,
    raw_time_scale_s: float | None,
    deadline_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize supported hxtorch raw-event layouts into first-spike tensors."""
    event_rows: list[tuple[int, int, float]] = []
    if isinstance(raw, tuple) and len(raw) == 2:
        addresses = torch.as_tensor(raw[0]).detach().cpu()
        times = torch.as_tensor(raw[1]).detach().cpu().reshape(-1)
        if addresses.ndim == 1:
            if batch_count != 1:
                raise RuntimeError(
                    "raw spike addresses omit batch ids for a multi-batch experiment"
                )
            batch = torch.zeros_like(addresses, dtype=torch.long)
            neuron = addresses.to(torch.long)
        elif addresses.ndim == 2 and addresses.shape[1] >= 2:
            batch = addresses[:, 0].to(torch.long)
            neuron = addresses[:, -1].to(torch.long)
        else:
            raise RuntimeError("unsupported SpikeHandle.to_raw address layout")
        if batch.numel() != times.numel():
            raise RuntimeError(
                "raw spike addresses and timestamps have different lengths"
            )
        event_rows = [
            (int(batch_id), int(neuron_id), float(time))
            for batch_id, neuron_id, time in zip(
                batch.tolist(), neuron.tolist(), times.tolist()
            )
        ]
        integer_time = not torch.is_floating_point(times)
    elif isinstance(raw, (list, tuple)):
        integer_time = True
        for row in raw:
            if len(row) < 3:
                raise RuntimeError("raw spike tuple must contain time, batch, neuron")
            time, batch_id, neuron_id = row[:3]
            integer_time = integer_time and isinstance(time, int)
            event_rows.append((int(batch_id), int(neuron_id), float(time)))
    else:
        raise RuntimeError("unsupported raw spike representation")

    scale = raw_time_scale_s
    if scale is None:
        scale = _fpga_time_scale_s() if integer_time else 1.0

    first = torch.full((batch_count, pool_size), torch.nan, dtype=torch.float64)
    count = torch.zeros((batch_count, pool_size), dtype=torch.int64)
    for batch_id, neuron_id, raw_time in event_rows:
        if not 0 <= batch_id < batch_count or not 0 <= neuron_id < pool_size:
            raise RuntimeError(
                "raw event coordinate out of range: "
                f"batch={batch_id}, neuron={neuron_id}"
            )
        time_s = raw_time * scale
        if not math.isfinite(time_s) or time_s < 0.0:
            raise RuntimeError("raw spike timestamp must be finite and non-negative")
        count[batch_id, neuron_id] += 1
        previous = first[batch_id, neuron_id]
        if torch.isnan(previous) or time_s < float(previous):
            first[batch_id, neuron_id] = time_s

    fired = torch.isfinite(first) & (first <= deadline_s)
    first = torch.where(fired, first, torch.full_like(first, torch.nan))
    return first, fired, count


def _logical_neuron_coordinates(indices: tuple[int, ...]) -> list[Any]:
    """Construct single-compartment logical coordinates used by hxtorch."""
    halco = import_module("pyhalco_hicann_dls_vx_v3")
    compartment_map = halco.LogicalNeuronCompartments(
        {
            halco.CompartmentOnLogicalNeuron():
            [halco.AtomicNeuronOnLogicalNeuron()]
        }
    )
    coordinates: list[Any] = []
    for index in indices:
        atomic = halco.AtomicNeuronOnDLS(halco.common.Enum(index))
        coordinates.append(halco.LogicalNeuronOnDLS(compartment_map, atomic))
    return coordinates


class BrainScaleS2PoolBackend:
    """Lazy hxtorch backend that records only raw physical spike events."""

    @staticmethod
    def dependencies_available() -> bool:
        try:
            import_module("hxtorch")
            import_module("hxtorch.spiking")
        except ImportError:
            return False
        return True

    def diagnose_cadc(
        self,
        config: BrainScaleS2PoolConfig,
        *,
        pool_size: int = 4,
        placement: PlacementMode = "same-quadrant",
    ) -> CADCDiagnosticResult:
        """Record paired baseline and one-input PSP traces on fixed neurons."""
        config.require_reproducible_calibration()
        indices = resolve_physical_neuron_indices(pool_size, placement)
        stimulus_time_s = config.input_early_s
        stimulus_step = int(round(stimulus_time_s / config.dt_s))
        if not 1 <= stimulus_step < config.runtime_steps - 1:
            raise ValueError("diagnostic stimulus must leave pre/post CADC samples")

        try:
            hxtorch = import_module("hxtorch")
            hxsnn = import_module("hxtorch.spiking")
        except ImportError as error:
            raise RuntimeError(
                "BrainScaleS-2 CADC diagnosis requires the EBRAINS hxtorch environment"
            ) from error

        initialized = False
        try:
            hxtorch.init_hardware()
            initialized = True
            experiment = hxsnn.Experiment(dt=config.dt_s)
            experiment.inter_batch_entry_wait = int(
                round(config.inter_batch_wait_s / _fpga_time_scale_s())
            )
            if config.calibration_path is not None:
                calib_helper = import_module("hxtorch.core.utils").calib_helper
                experiment.calibration = calib_helper.fixture_calibration_from_file(
                    str(config.calibration_path)
                )

            synapse = hxsnn.Synapse(
                in_features=1,
                out_features=pool_size,
                experiment=experiment,
            )
            synapse.weight.data.fill_(config.synaptic_weight)
            lif = hxsnn.LIF(
                size=pool_size,
                experiment=experiment,
                tau_mem=config.tau_mem_s,
                tau_syn=config.tau_syn_s,
                leak=config.leak,
                reset=config.reset,
                threshold=config.threshold,
                refractory_time=config.refractory_time_s,
                i_synin_gm=config.i_synin_gm,
                synapse_dac_bias=config.synapse_dac_bias,
                placement_constraint=_logical_neuron_coordinates(indices),
                enable_spike_recording=True,
                enable_cadc_recording=True,
                cadc_time_shift=-1,
                enable_madc_recording=False,
            )

            # Each trial is a paired no-input/stimulated batch entry.
            batch_count = 2 * config.trials
            inputs = torch.zeros(
                (config.runtime_steps, batch_count, 1),
                dtype=torch.float32,
            )
            inputs[stimulus_step, 1::2, 0] = 1.0
            synapse_output = synapse(hxsnn.LIFObservables(spikes=inputs))
            observables = lif(synapse_output)
            hxsnn.run(experiment, config.runtime_steps)

            cadc = getattr(observables, "membrane_cadc", None)
            spikes = getattr(observables, "spikes", None)
            if not isinstance(cadc, torch.Tensor) or not isinstance(spikes, torch.Tensor):
                raise RuntimeError(
                    "installed hxtorch did not return dense CADC and spike observables"
                )
            batch_first = (batch_count, config.runtime_steps, pool_size)
            time_first = (config.runtime_steps, batch_count, pool_size)
            if tuple(spikes.shape) != tuple(cadc.shape):
                raise RuntimeError(
                    "CADC and spike diagnostic observables have different shapes: "
                    f"cadc={tuple(cadc.shape)}, spikes={tuple(spikes.shape)}"
                )
            if tuple(cadc.shape) == time_first:
                cadc = cadc.permute(1, 0, 2)
                spikes = spikes.permute(1, 0, 2)
            elif tuple(cadc.shape) != batch_first:
                raise RuntimeError(
                    "unexpected hxtorch diagnostic observable shape: "
                    f"cadc={tuple(cadc.shape)}, expected one of "
                    f"{batch_first} or {time_first}"
                )

            cadc = cadc.detach().cpu().to(torch.float64)
            spikes = spikes.detach().cpu().to(torch.float64)
            baseline_cadc = cadc[0::2].contiguous()
            stimulated_cadc = cadc[1::2].contiguous()
            baseline_spikes = spikes[0::2].contiguous()
            stimulated_spikes = spikes[1::2].contiguous()

            chip_identifier = None
            get_identifier = getattr(hxtorch, "get_unique_identifier", None)
            if callable(get_identifier):
                chip_identifier = [str(value) for value in get_identifier()]
            return CADCDiagnosticResult(
                baseline_cadc=baseline_cadc,
                stimulated_cadc=stimulated_cadc,
                baseline_spikes=baseline_spikes,
                stimulated_spikes=stimulated_spikes,
                time_s=torch.arange(cadc.shape[1], dtype=torch.float64) * config.dt_s,
                stimulus_time_s=stimulus_time_s,
                physical_coordinates=indices,
                metadata={
                    "backend": "hardware",
                    "hxtorch_version": getattr(hxtorch, "__version__", "unknown"),
                    "chip_identifier": chip_identifier,
                    "calibration_sha256": calibration_sha256(config.calibration_path),
                },
            )
        finally:
            if initialized:
                hxtorch.release_hardware()

    def run(
        self,
        potential: Potential,
        config: BrainScaleS2PoolConfig,
        *,
        pool_size: int,
        placement: PlacementMode,
        routing: RoutingMode,
    ) -> PoolRunResult:
        config.require_reproducible_calibration()
        encoding = encode_potential_for_brainscales2(
            potential,
            config,
            pool_size=pool_size,
            routing=routing,
        )
        indices = resolve_physical_neuron_indices(pool_size, placement)

        try:
            hxtorch = import_module("hxtorch")
            hxsnn = import_module("hxtorch.spiking")
        except ImportError as error:
            raise RuntimeError(
                "BrainScaleS-2 execution requires the EBRAINS hxtorch environment"
            ) from error

        initialized = False
        try:
            hxtorch.init_hardware()
            initialized = True
            experiment = hxsnn.Experiment(dt=config.dt_s)
            experiment.inter_batch_entry_wait = int(
                round(config.inter_batch_wait_s / _fpga_time_scale_s())
            )
            if config.calibration_path is not None:
                calib_helper = import_module("hxtorch.core.utils").calib_helper
                experiment.calibration = calib_helper.fixture_calibration_from_file(
                    str(config.calibration_path)
                )

            input_channels = 1 if routing == "broadcast" else pool_size
            synapse = hxsnn.Synapse(
                in_features=input_channels,
                out_features=pool_size,
                experiment=experiment,
            )
            synapse.weight.data.zero_()
            if routing == "broadcast":
                synapse.weight.data.fill_(config.synaptic_weight)
            else:
                diagonal = min(synapse.weight.data.shape)
                arange = torch.arange(diagonal)
                synapse.weight.data[arange, arange] = config.synaptic_weight

            lif = hxsnn.LIF(
                size=pool_size,
                experiment=experiment,
                tau_mem=config.tau_mem_s,
                tau_syn=config.tau_syn_s,
                leak=config.leak,
                reset=config.reset,
                threshold=config.threshold,
                refractory_time=config.refractory_time_s,
                i_synin_gm=config.i_synin_gm,
                synapse_dac_bias=config.synapse_dac_bias,
                placement_constraint=_logical_neuron_coordinates(indices),
                enable_spike_recording=True,
                enable_cadc_recording=False,
                enable_madc_recording=False,
            )

            # Batch ordering is [trial 0 samples..., trial 1 samples..., ...].
            inputs = encoding.dense_spikes.repeat(1, config.trials, 1)
            synapse_output = synapse(hxsnn.LIFObservables(spikes=inputs))
            observables = lif(synapse_output)
            run_output = hxsnn.run(experiment, config.runtime_steps)
            legacy_observables = _legacy_experiment_observables(experiment, lif)
            raw, raw_api = _find_raw_spikes(
                legacy_observables,
                lif,
                observables,
                run_output,
                experiment,
            )

            batch_count = config.trials * encoding.sample_count
            first, fired, count = _raw_events_to_tensors(
                raw,
                batch_count=batch_count,
                pool_size=pool_size,
                raw_time_scale_s=config.raw_time_scale_s,
                deadline_s=config.observation_deadline_s,
            )
            result_shape = (config.trials, encoding.sample_count, pool_size)
            chip_identifier = None
            get_identifier = getattr(hxtorch, "get_unique_identifier", None)
            if callable(get_identifier):
                chip_identifier = [str(value) for value in get_identifier()]

            return PoolRunResult(
                first_spike_s=first.reshape(result_shape),
                fired=fired.reshape(result_shape),
                spike_count=count.reshape(result_shape),
                nominal_input_s=encoding.injected_time_s,
                ideal_input_s=encoding.ideal_time_s,
                physical_coordinates=indices,
                pool_size=pool_size,
                placement=placement,
                routing=routing,
                original_input_shape=encoding.original_shape,
                metadata={
                    "backend": "hardware",
                    "hxtorch_version": getattr(hxtorch, "__version__", "unknown"),
                    "chip_identifier": chip_identifier,
                    "calibration_sha256": calibration_sha256(config.calibration_path),
                    "raw_spike_api": raw_api,
                    "clamped_values": int(encoding.clamp_mask.sum().item()),
                },
            )
        finally:
            if initialized:
                hxtorch.release_hardware()


def with_operating_point(
    config: BrainScaleS2PoolConfig,
    *,
    threshold: float,
    synaptic_weight: float,
    i_synin_gm: float,
) -> BrainScaleS2PoolConfig:
    """Return a validated immutable config for one calibration candidate."""
    return replace(
        config,
        threshold=threshold,
        synaptic_weight=synaptic_weight,
        i_synin_gm=i_synin_gm,
    )

