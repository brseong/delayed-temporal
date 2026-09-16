"""Lazy physical collectors for independent BrainScaleS-2 primitives.

There is intentionally no automatic replacement for missing low-level
constant-current control.  ``phi-np`` and ``phi-nl`` either use the installed
BrainScaleS-2 PyNN playback controls or fail with a capability error.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any
import math

import torch

from .backend import (
    _configure_experiment_calibration,
    _logical_neuron_coordinates,
    calibration_sha256,
)
from .hagen import HagenConfig, HagenPWMBackend
from .primitive_noise import (
    PrimitiveKind,
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    PrimitiveStage,
)


PYNN_BACKEND_MODULE = "pynn_brainscales.brainscales2"


class PrimitiveCapabilityError(RuntimeError):
    """Raised when the installed release cannot implement a physical primitive."""


def _module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def probe_primitive_capabilities() -> dict[str, Any]:
    """Inspect installed APIs without acquiring a hardware connection."""
    result: dict[str, Any] = {
        "hxtorch": False,
        "hxtorch_spiking": False,
        "hxtorch_perceptron": False,
        "brainscales2_pynn": False,
        "constant_current_playback": False,
        "missing": [],
    }
    try:
        hxtorch = import_module("hxtorch")
    except ImportError as error:
        result["missing"].append(f"hxtorch: {error}")
    else:
        result["hxtorch"] = True
        result["hxtorch_version"] = _module_version(hxtorch)
        try:
            import_module("hxtorch.spiking")
        except ImportError as error:
            result["missing"].append(f"hxtorch.spiking: {error}")
        else:
            result["hxtorch_spiking"] = True
        try:
            perceptron = import_module("hxtorch.perceptron")
        except ImportError as error:
            result["missing"].append(f"hxtorch.perceptron: {error}")
        else:
            result["hxtorch_perceptron"] = hasattr(perceptron, "nn")

    try:
        pynn = import_module(PYNN_BACKEND_MODULE)
    except ImportError as error:
        result["missing"].append(f"pyNN.brainscales2: {error}")
    else:
        result["brainscales2_pynn"] = True
        cell_namespace = getattr(pynn, "cells", None)
        cell_type = getattr(cell_namespace, "HXNeuron", None)
        run_command = getattr(pynn, "RunCommand", None)
        result["constant_current_playback"] = bool(
            cell_type is not None
            and run_command is not None
            and hasattr(run_command, "APPEND")
            and hasattr(run_command, "EXECUTE")
        )
        result["pynn_version"] = _module_version(pynn)
    result["phi-np"] = bool(result["constant_current_playback"])
    result["phi-nl"] = bool(result["constant_current_playback"])
    result["psi-int"] = bool(result["hxtorch_perceptron"])
    result["psi-ne"] = bool(result["hxtorch_spiking"])
    return result


def _chip_identifier(hxtorch: Any) -> list[str] | None:
    getter = getattr(hxtorch, "get_unique_identifier", None)
    if not callable(getter):
        return None
    return [str(item) for item in getter()]


def _codes(primitive: PrimitiveKind, quick: bool) -> torch.Tensor:
    if primitive == "phi-np":
        return (
            torch.tensor([0.0, 15.0, 30.0], dtype=torch.float64)
            if quick
            else torch.arange(32, dtype=torch.float64)
        )
    if primitive == "phi-nl":
        return (
            torch.tensor([1.0, 8.0, 31.0], dtype=torch.float64)
            if quick
            else torch.arange(1, 32, dtype=torch.float64)
        )
    raise ValueError(f"primitive does not use UInt5 state codes: {primitive}")


class PrimitiveHardwareBackend:
    """Collect raw per-device observations without pooling physical replicas."""

    def probe(self) -> dict[str, Any]:
        return probe_primitive_capabilities()

    def collect(
        self,
        primitive: PrimitiveKind,
        config: PrimitiveNoiseConfig,
        *,
        stage: PrimitiveStage = "transfer",
        quick: bool = False,
    ) -> PrimitiveObservation:
        if primitive == "psi-int":
            return self._collect_psi_int(config, quick=quick)
        if primitive == "psi-ne":
            return self._collect_psi_ne(config, quick=quick)
        if primitive in ("phi-np", "phi-nl"):
            return self._collect_pynn_transfer(
                primitive, config, stage=stage, quick=quick
            )
        raise ValueError(f"unsupported primitive: {primitive}")

    def _collect_psi_int(
        self,
        config: PrimitiveNoiseConfig,
        *,
        quick: bool,
    ) -> PrimitiveObservation:
        if config.hagen_calibration_path is None and not config.allow_environment_calibration:
            raise ValueError("psi-int hardware collection requires a Hagen calibration")
        durations = (
            torch.tensor([0.0, 15.0, 31.0], dtype=torch.float32)
            if quick
            else torch.arange(32, dtype=torch.float32)
        )
        drives = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float32)
        backend = HagenPWMBackend(
            HagenConfig(
                mode="hardware",
                calibration_path=config.hagen_calibration_path,
                allow_environment_calibration=config.allow_environment_calibration,
                wait_between_events=config.hagen_wait_between_events,
                num_sends=config.hagen_num_sends,
            )
        )
        by_drive: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        repeated = durations.repeat(config.repeats).reshape(-1, 1)
        with backend.hardware_session():
            for drive in drives:
                weights = torch.full(
                    (config.device_count, 1), float(drive), dtype=torch.float32
                )
                result = backend.direct_linear(repeated, weights, avg=1)
                expected = (config.repeats * durations.numel(), config.device_count)
                if tuple(result.value.shape) != expected:
                    raise RuntimeError(
                        f"unexpected Hagen primitive output {tuple(result.value.shape)}; "
                        f"expected {expected}"
                    )
                by_drive.append(
                    result.value.to(torch.float64).reshape(
                        config.repeats, durations.numel(), config.device_count
                    )
                )
                metadata.append(result.metadata)
        observed = torch.stack(by_drive, dim=2).reshape(
            config.repeats,
            durations.numel() * drives.numel(),
            config.device_count,
        )
        input_code = durations.to(torch.float64).repeat_interleave(drives.numel())
        auxiliary = drives.to(torch.float64).repeat(durations.numel())
        ideal = input_code * auxiliary
        delivered = torch.isfinite(observed)
        saturated = (observed <= -128.0) | (observed >= 127.0)
        return PrimitiveObservation(
            primitive="psi-int",
            stage="transfer",
            output_kind="pwm-code",
            input_code=input_code,
            auxiliary_input=auxiliary,
            ideal_variable=ideal,
            observed=torch.where(
                delivered, observed, torch.full_like(observed, torch.nan)
            ),
            delivered=delivered,
            spike_count=torch.zeros_like(observed, dtype=torch.int64),
            saturated=saturated,
            physical_coordinates=tuple(range(config.device_count)),
            fit_point_mask=torch.ones_like(input_code, dtype=torch.bool),
            metadata={
                "backend": "hagen-hardware",
                "avg": 1,
                "bias": False,
                "drive_codes": drives.tolist(),
                "per_drive": metadata,
                "chip_identifier": metadata[0].get("chip_identifier"),
                "calibration_sha256": calibration_sha256(
                    config.hagen_calibration_path
                ),
                "replicas_are_pooled": False,
                "coordinate_semantics": "hagen-output-index",
            },
        )

    def _collect_psi_ne(
        self,
        config: PrimitiveNoiseConfig,
        *,
        quick: bool,
    ) -> PrimitiveObservation:
        if config.spiking_calibration_path is None and not config.allow_environment_calibration:
            raise ValueError("psi-ne hardware collection requires a spiking calibration")
        try:
            hxtorch = import_module("hxtorch")
            hxsnn = import_module("hxtorch.spiking")
        except ImportError as error:
            raise PrimitiveCapabilityError(
                "psi-ne requires hxtorch.spiking in the EBRAINS environment"
            ) from error
        input_times = (
            torch.tensor(
                [config.input_early_s, 15.0e-6, config.input_late_s],
                dtype=torch.float64,
            )
            if quick
            else torch.linspace(
                config.input_early_s,
                config.input_late_s,
                21,
                dtype=torch.float64,
            )
        )
        runtime_steps = int(math.ceil(config.deadline_s / config.dt_s)) + 1
        observation_step = int(round(config.observation_time_s / config.dt_s))
        pair_count = config.repeats * input_times.numel()
        batch_count = 2 * pair_count
        inputs = torch.zeros((runtime_steps, batch_count, 1), dtype=torch.float32)
        for pair, input_time in enumerate(input_times.repeat(config.repeats)):
            input_step = int(round(float(input_time) / config.dt_s))
            inputs[input_step, 2 * pair + 1, 0] = 1.0

        initialized = False
        try:
            hxtorch.init_hardware()
            initialized = True
            experiment = hxsnn.Experiment(dt=config.dt_s)
            calibration_loader = None
            if config.spiking_calibration_path is not None:
                calibration_loader = _configure_experiment_calibration(
                    experiment, config.spiking_calibration_path
                )
            synapse = hxsnn.Synapse(
                in_features=1,
                out_features=config.device_count,
                experiment=experiment,
            )
            synapse.weight.data.fill_(config.exponential_input_weight)
            lif = hxsnn.LIF(
                size=config.device_count,
                experiment=experiment,
                tau_mem=config.tau_mem_s,
                tau_syn=config.tau_syn_s,
                leak=80,
                reset=80,
                threshold=125,
                placement_constraint=_logical_neuron_coordinates(
                    config.physical_coordinates
                ),
                enable_spike_recording=True,
                enable_cadc_recording=True,
                cadc_time_shift=-1,
                enable_madc_recording=False,
                threshold_enable=False,
            )
            synapse_output = synapse(hxsnn.LIFObservables(spikes=inputs))
            observables = lif(synapse_output)
            run_output = hxsnn.run(experiment, runtime_steps)
            cadc = getattr(observables, "membrane_cadc", None)
            spikes = getattr(observables, "spikes", None)
            if not isinstance(cadc, torch.Tensor) or not isinstance(spikes, torch.Tensor):
                raise PrimitiveCapabilityError(
                    "installed hxtorch did not expose dense CADC and spike observables"
                )
            time_first = (runtime_steps, batch_count, config.device_count)
            batch_first = (batch_count, runtime_steps, config.device_count)
            if tuple(cadc.shape) == time_first:
                cadc = cadc.permute(1, 0, 2)
                spikes = spikes.permute(1, 0, 2)
            elif tuple(cadc.shape) != batch_first:
                raise RuntimeError(
                    f"unexpected psi-ne CADC shape {tuple(cadc.shape)}"
                )
            cadc = cadc.detach().cpu().to(torch.float64)
            spikes = spikes.detach().cpu()
            baseline = cadc[0::2, observation_step, :].reshape(
                config.repeats, input_times.numel(), config.device_count
            )
            observed = cadc[1::2, observation_step, :].reshape(
                config.repeats, input_times.numel(), config.device_count
            )
            spike_count = spikes[1::2].sum(dim=1).to(torch.int64).reshape(
                config.repeats, input_times.numel(), config.device_count
            )
            delivered = torch.isfinite(observed)
            saturated = (observed <= -128.0) | (observed >= 127.0)
            return PrimitiveObservation(
                primitive="psi-ne",
                stage="transfer",
                output_kind="cadc-potential",
                input_code=input_times / config.dt_s,
                ideal_variable=input_times,
                observed=torch.where(
                    delivered, observed, torch.full_like(observed, torch.nan)
                ),
                delivered=delivered,
                spike_count=spike_count,
                saturated=saturated,
                baseline=baseline,
                physical_coordinates=config.physical_coordinates,
                fit_point_mask=torch.ones_like(input_times, dtype=torch.bool),
                metadata={
                    "backend": "hxtorch-spiking-hardware",
                    "hxtorch_version": _module_version(hxtorch),
                    "chip_identifier": _chip_identifier(hxtorch),
                    "calibration_loader": calibration_loader,
                    "calibration_sha256": calibration_sha256(
                        config.spiking_calibration_path
                    ),
                    "observation_step": observation_step,
                    "resolved_parameters": {
                        "tau_mem_s": config.tau_mem_s,
                        "tau_syn_s": config.tau_syn_s,
                        "synaptic_weight": config.exponential_input_weight,
                        "leak": 80,
                        "reset": 80,
                        "threshold": 125,
                    },
                    "requested_threshold_enable": False,
                    "threshold_validation": "reject-if-any-spike",
                    "raw_spike_handle_required": False,
                    "replicas_are_pooled": False,
                },
            )
        finally:
            if initialized:
                hxtorch.release_hardware()

    def _collect_pynn_transfer(
        self,
        primitive: PrimitiveKind,
        config: PrimitiveNoiseConfig,
        *,
        stage: PrimitiveStage,
        quick: bool,
    ) -> PrimitiveObservation:
        if config.spiking_calibration_path is None:
            raise ValueError("phi primitive collection requires a spiking calibration")
        capabilities = probe_primitive_capabilities()
        if not capabilities[primitive]:
            raise PrimitiveCapabilityError(
                f"{primitive} requires HXNeuron constant-current playback; "
                f"capabilities={capabilities}"
            )
        if primitive == "phi-np" and stage not in ("static", "dynamic"):
            raise ValueError("phi-np stage must be static or dynamic")
        if primitive == "phi-nl":
            stage = "transfer"
        codes = _codes(primitive, quick)
        observed = torch.full(
            (config.repeats, codes.numel(), config.device_count),
            torch.nan,
            dtype=torch.float64,
        )
        spike_count = torch.zeros_like(observed, dtype=torch.int64)
        precharge_cadc = (
            torch.full_like(observed, torch.nan)
            if stage == "dynamic" or primitive == "phi-nl"
            else None
        )
        per_code_metadata: list[dict[str, Any]] = []
        for point, code in enumerate(codes.tolist()):
            first, count, precharge, metadata = self._run_pynn_code(
                primitive,
                stage,
                int(code),
                config,
            )
            observed[:, point, :] = first
            spike_count[:, point, :] = count
            if precharge_cadc is not None:
                if precharge is None:
                    raise PrimitiveCapabilityError(
                        "dynamic precharge did not expose a CADC membrane trace"
                    )
                precharge_cadc[:, point, :] = precharge
            per_code_metadata.append(metadata)
        delivered = torch.isfinite(observed) & (observed <= config.deadline_s)
        observed = torch.where(
            delivered, observed, torch.full_like(observed, torch.nan)
        )
        if primitive == "phi-np":
            ideal = codes.to(torch.float64)
            fit_mask = codes != 31
        else:
            ideal = torch.log(codes.to(torch.float64))
            fit_mask = torch.ones_like(codes, dtype=torch.bool)
        return PrimitiveObservation(
            primitive=primitive,
            stage=stage,
            output_kind="spike-time-s",
            input_code=codes,
            ideal_variable=ideal,
            observed=observed,
            delivered=delivered,
            spike_count=spike_count,
            saturated=torch.zeros_like(delivered),
            precharge_cadc=precharge_cadc,
            physical_coordinates=config.physical_coordinates,
            fit_point_mask=fit_mask,
            metadata={
                "backend": "brainscales2-pynn-hardware",
                "pynn_version": capabilities.get("pynn_version"),
                "calibration_sha256": calibration_sha256(
                    config.spiking_calibration_path
                ),
                "constant_current_playback": primitive == "phi-np",
                "automatic_fallback": False,
                "per_code": per_code_metadata,
                "replicas_are_pooled": False,
            },
        )

    @staticmethod
    def _pynn_coordinates(config: PrimitiveNoiseConfig) -> list[int]:
        # The pynn_brainscales placement API consumes atomic neuron indices and
        # performs its own coordinate conversion.
        return list(config.physical_coordinates)

    @staticmethod
    def _hxneuron_type(pynn: Any) -> Any:
        for namespace_name in ("cells", "standardmodels"):
            namespace = getattr(pynn, namespace_name, None)
            candidate = getattr(namespace, "HXNeuron", None)
            if candidate is not None:
                return candidate
            cells = getattr(namespace, "cells", None)
            candidate = getattr(cells, "HXNeuron", None)
            if candidate is not None:
                return candidate
        raise PrimitiveCapabilityError("installed PyNN exposes no HXNeuron cell type")

    @staticmethod
    def _reset_code(code: int, config: PrimitiveNoiseConfig) -> int:
        if code == 31:
            return config.threshold_code
        fraction = code / 30.0
        return int(
            round(
                config.reset_code_minimum
                + fraction * (config.reset_code_maximum - config.reset_code_minimum)
            )
        )

    def _run_pynn_code(
        self,
        primitive: PrimitiveKind,
        stage: PrimitiveStage,
        code: int,
        config: PrimitiveNoiseConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        pynn = import_module(PYNN_BACKEND_MODULE)
        cell_type = self._hxneuron_type(pynn)
        chip = pynn.helper.chip_from_file(str(config.spiking_calibration_path))
        coordinates = self._pynn_coordinates(config)
        setup_complete = False
        window_us = config.deadline_s * 1.0e6
        reference_us = config.input_early_s * 1.0e6
        ramp_stop_us = config.input_late_s * 1.0e6
        precharge_us = max(0.5, reference_us - 2.0)
        try:
            pynn.setup(initial_config=chip, neuronPermutation=coordinates)
            setup_complete = True
            cell_parameters = {
                "leak_v_leak": (
                    self._reset_code(code, config)
                    if stage == "static"
                    else config.reset_code_minimum
                ),
                "leak_i_bias": config.leak_bias,
                "leak_enable_division": True,
                "threshold_enable": True,
                "threshold_v_threshold": config.threshold_code,
                "reset_v_reset": (
                    self._reset_code(code, config)
                    if stage == "static"
                    else config.reset_code_minimum
                ),
                "constant_current_enable": False,
                "constant_current_i_offset": config.constant_current_code,
                "reset_i_bias": 1022,
                "reset_enable_multiplication": True,
                "refractory_period_refractory_time": 255,
                "refractory_period_enable_pause": True,
            }
            population = pynn.Population(
                config.device_count,
                cell_type(**cell_parameters),
            )
            record_variables: list[str] = ["spikes"]
            uses_precharge = stage == "dynamic" or primitive == "phi-nl"
            if uses_precharge:
                record_variables.append("v")
            population.record(record_variables)

            if uses_precharge:
                precharge_times = [
                    trial * window_us + precharge_us
                    for trial in range(config.repeats)
                ]
                source = pynn.Population(
                    1,
                    pynn.SpikeSourceArray(spike_times=precharge_times),
                )
                weight = max(
                    1,
                    int(round(config.precharge_weight_maximum * code / 31.0)),
                )
                pynn.Projection(
                    source,
                    population,
                    pynn.AllToAllConnector(),
                    synapse_type=pynn.StaticSynapse(weight=weight),
                    receptor_type="excitatory",
                )
            else:
                weight = None

            if primitive == "phi-nl":
                exponential_times = [
                    trial * window_us + reference_us
                    for trial in range(config.repeats)
                ]
                exponential_source = pynn.Population(
                    1,
                    pynn.SpikeSourceArray(spike_times=exponential_times),
                )
                pynn.Projection(
                    exponential_source,
                    population,
                    pynn.AllToAllConnector(),
                    synapse_type=pynn.StaticSynapse(
                        weight=config.exponential_input_weight
                    ),
                    receptor_type="excitatory",
                )
                pynn.run(config.repeats * window_us)
            else:
                append = pynn.RunCommand.APPEND
                execute = pynn.RunCommand.EXECUTE
                for trial in range(config.repeats):
                    pynn.run(reference_us, append)
                    population.set(constant_current_enable=True)
                    pynn.run(ramp_stop_us - reference_us, append)
                    population.set(constant_current_enable=False)
                    command = execute if trial == config.repeats - 1 else append
                    pynn.run(window_us - ramp_stop_us, command)

            data = population.get_data(record_variables)
            segment = data.segments[-1]
            first, count = self._decode_pynn_spikes(
                segment.spiketrains,
                repeats=config.repeats,
                devices=config.device_count,
                window_us=window_us,
            )
            precharge = None
            if uses_precharge:
                precharge = self._decode_pynn_precharge(
                    segment,
                    repeats=config.repeats,
                    devices=config.device_count,
                    window_us=window_us,
                    sample_us=reference_us - 0.5,
                )
            identifier = pynn.helper.get_unique_identifier()
            chip_identifier = (
                [str(item) for item in identifier]
                if isinstance(identifier, (tuple, list))
                else [str(identifier)]
            )
            return first, count, precharge, {
                "input_code": code,
                "reset_code": cell_parameters["reset_v_reset"],
                "precharge_weight": weight,
                "resolved_cell_parameters": cell_parameters,
                "window_us": window_us,
                "reference_us": reference_us,
                "raw_time_unit": "PyNN-ms mapped to physical-us",
                "chip_identifier": chip_identifier,
            }
        finally:
            if setup_complete:
                pynn.end()

    @staticmethod
    def _decode_pynn_spikes(
        spike_trains: Any,
        *,
        repeats: int,
        devices: int,
        window_us: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(spike_trains) != devices:
            raise RuntimeError("PyNN spike-train count does not match physical devices")
        first = torch.full((repeats, devices), torch.nan, dtype=torch.float64)
        count = torch.zeros((repeats, devices), dtype=torch.int64)
        for device, train in enumerate(spike_trains):
            magnitude = getattr(train, "magnitude", train)
            for time_value in torch.as_tensor(magnitude, dtype=torch.float64).reshape(-1):
                time_us = float(time_value)
                trial = int(math.floor(time_us / window_us))
                if not 0 <= trial < repeats:
                    raise RuntimeError("PyNN spike lies outside the scheduled windows")
                physical_time_s = (time_us - trial * window_us) * 1.0e-6
                count[trial, device] += 1
                if torch.isnan(first[trial, device]) or physical_time_s < float(
                    first[trial, device]
                ):
                    first[trial, device] = physical_time_s
        return first, count

    @staticmethod
    def _decode_pynn_precharge(
        segment: Any,
        *,
        repeats: int,
        devices: int,
        window_us: float,
        sample_us: float,
    ) -> torch.Tensor:
        signals = list(getattr(segment, "irregularlysampledsignals", []))
        if not signals:
            signals = list(getattr(segment, "analogsignals", []))
        if not signals:
            raise PrimitiveCapabilityError("PyNN returned no CADC membrane signal")
        signal = signals[0]
        values = torch.as_tensor(
            getattr(signal, "magnitude", signal), dtype=torch.float64
        )
        if values.ndim != 2 or values.shape[1] != devices:
            raise RuntimeError(
                f"unexpected PyNN CADC signal shape {tuple(values.shape)}"
            )
        times = torch.as_tensor(
            getattr(getattr(signal, "times", None), "magnitude", []),
            dtype=torch.float64,
        )
        if times.numel() != values.shape[0]:
            raise RuntimeError("PyNN CADC signal has no usable time axis")
        selected = torch.empty((repeats, devices), dtype=torch.float64)
        for trial in range(repeats):
            target = trial * window_us + sample_us
            index = int(torch.argmin(torch.abs(times - target)))
            selected[trial] = values[index]
        return selected
