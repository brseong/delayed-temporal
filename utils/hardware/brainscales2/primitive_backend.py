"""Lazy physical collectors for independent BrainScaleS-2 primitives.

There is intentionally no automatic replacement for missing low-level
constant-current control.  ``phi-np`` and ``phi-nl`` either use the installed
BrainScaleS-2 PyNN playback controls or fail with a capability error.
"""

from __future__ import annotations

import copy
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any
import json
import math
import subprocess
import sys
import tempfile

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
        "causal_correlation_sensor": False,
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
        standardmodels = getattr(pynn, "standardmodels", None)
        synapses = getattr(standardmodels, "synapses", None)
        try:
            import_module("dlens_vx_v3")
        except ImportError as error:
            result["missing"].append(f"dlens_vx_v3: {error}")
        else:
            result["causal_correlation_sensor"] = bool(
                hasattr(pynn, "PlasticityRule")
                and hasattr(pynn, "Timer")
                and hasattr(pynn, "InjectedConfiguration")
                and hasattr(synapses, "PlasticSynapse")
            )
    result["phi-np"] = bool(result["constant_current_playback"])
    result["phi-nl"] = bool(result["constant_current_playback"])
    result["psi-int"] = bool(result["hxtorch_perceptron"])
    result["psi-ne"] = bool(result["hxtorch_spiking"])
    result["psi-ed"] = bool(result["causal_correlation_sensor"])
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
        if primitive == "psi-ed":
            return self._collect_psi_ed(config, quick=quick)
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
        with backend.hardware_session():
            for drive in drives:
                weights = torch.full(
                    (config.hagen_candidate_count, 1),
                    float(drive),
                    dtype=torch.float32,
                )
                drive_chunks: list[torch.Tensor] = []
                chunk_metadata: list[dict[str, Any]] = []
                for start in range(0, config.repeats, config.hagen_chunk_repeats):
                    stop = min(start + config.hagen_chunk_repeats, config.repeats)
                    chunk_repeats = stop - start
                    repeated = durations.repeat(chunk_repeats).reshape(-1, 1)
                    result = backend.direct_linear(repeated, weights, avg=1)
                    expected = (
                        chunk_repeats * durations.numel(),
                        config.hagen_candidate_count,
                    )
                    if tuple(result.value.shape) != expected:
                        raise RuntimeError(
                            "unexpected Hagen primitive output "
                            f"{tuple(result.value.shape)}; expected {expected}"
                        )
                    drive_chunks.append(
                        result.value.to(torch.float64).reshape(
                            chunk_repeats,
                            durations.numel(),
                            config.hagen_candidate_count,
                        )
                    )
                    chunk_metadata.append(
                        {
                            **result.metadata,
                            "trial_start": start,
                            "trial_stop": stop,
                        }
                    )
                by_drive.append(torch.cat(drive_chunks, dim=0))
                metadata.append(
                    {
                        "drive": float(drive),
                        "chunks": chunk_metadata,
                    }
                )
        candidates = torch.stack(by_drive, dim=2).reshape(
            config.repeats,
            durations.numel() * drives.numel(),
            config.hagen_candidate_count,
        )
        input_code = durations.to(torch.float64).repeat_interleave(drives.numel())
        auxiliary = drives.to(torch.float64).repeat(durations.numel())
        ideal = input_code * auxiliary
        candidate_saturated = (candidates <= -128.0) | (candidates >= 127.0)
        selected_indices, selection_scores = self._select_hagen_outputs(
            candidates,
            candidate_saturated,
            ideal,
            calibration_repeats=config.calibration_repeats,
            selected_count=config.device_count,
        )
        observed = candidates[:, :, selected_indices]
        delivered = torch.isfinite(observed)
        saturated = candidate_saturated[:, :, selected_indices]
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
            physical_coordinates=selected_indices,
            fit_point_mask=torch.ones_like(input_code, dtype=torch.bool),
            metadata={
                "backend": "hagen-hardware",
                "avg": 1,
                "bias": False,
                "drive_codes": drives.tolist(),
                "per_drive": metadata,
                "candidate_count": config.hagen_candidate_count,
                "chunk_repeats": config.hagen_chunk_repeats,
                "selected_output_indices": list(selected_indices),
                "candidate_selection_scores": selection_scores,
                "chip_identifier": metadata[0]["chunks"][0].get(
                    "chip_identifier"
                ),
                "calibration_sha256": calibration_sha256(
                    config.hagen_calibration_path
                ),
                "replicas_are_pooled": False,
                "coordinate_semantics": "hagen-output-index",
            },
        )

    @staticmethod
    def _select_hagen_outputs(
        observed: torch.Tensor,
        saturated: torch.Tensor,
        ideal_variable: torch.Tensor,
        *,
        calibration_repeats: int,
        selected_count: int,
    ) -> tuple[tuple[int, ...], list[dict[str, float | int | bool]]]:
        """Qualify Hagen output indices using calibration observations only."""
        if observed.ndim != 3 or saturated.shape != observed.shape:
            raise ValueError("Hagen candidate tensors must have matching 3D shapes")
        if ideal_variable.ndim != 1 or ideal_variable.numel() != observed.shape[1]:
            raise ValueError("Hagen candidate ideal variable has the wrong shape")
        if not 0 < calibration_repeats <= observed.shape[0]:
            raise ValueError("invalid Hagen calibration split")
        if not 0 < selected_count <= observed.shape[2]:
            raise ValueError("invalid Hagen selected output count")
        x = ideal_variable.to(torch.float64)
        design = torch.stack((torch.ones_like(x), x), dim=1).repeat(
            calibration_repeats, 1
        )
        rows: list[dict[str, float | int | bool]] = []
        for output_index in range(observed.shape[2]):
            values = observed[:calibration_repeats, :, output_index].to(torch.float64)
            finite = bool(torch.isfinite(values).all())
            has_saturation = bool(
                saturated[:calibration_repeats, :, output_index].any()
            )
            if finite and not has_saturation:
                fit = torch.linalg.lstsq(design, values.reshape(-1)).solution
                prediction = fit[0] + fit[1] * x
                span = float(prediction.max() - prediction.min())
                gain = float(fit[1])
                score = (
                    float((values - prediction).square().mean().sqrt()) / span
                    if gain > 0.0 and span > torch.finfo(torch.float64).eps
                    else float("inf")
                )
            else:
                gain = float("nan")
                score = float("inf")
            rows.append(
                {
                    "output_index": output_index,
                    "calibration_normalized_rmse": score,
                    "calibration_gain": gain,
                    "calibration_saturated": has_saturation,
                }
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                float(row["calibration_normalized_rmse"]),
                int(row["output_index"]),
            ),
        )
        if not math.isfinite(float(ranked[selected_count - 1]["calibration_normalized_rmse"])):
            raise RuntimeError("not enough finite unsaturated Hagen output candidates")
        selected = tuple(int(row["output_index"]) for row in ranked[:selected_count])
        return selected, rows

    def _collect_psi_ne(
        self,
        config: PrimitiveNoiseConfig,
        *,
        quick: bool,
    ) -> PrimitiveObservation:
        if config.spiking_calibration_path is None and not config.allow_environment_calibration:
            raise ValueError("psi-ne hardware collection requires a spiking calibration")
        capabilities = probe_primitive_capabilities()
        if not capabilities["psi-ne"]:
            raise PrimitiveCapabilityError(
                "psi-ne requires hxtorch.spiking in the EBRAINS environment"
            )
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

        calibration_loader = None
        baseline_chunks: list[torch.Tensor] = []
        observed_chunks: list[torch.Tensor] = []
        spike_count_chunks: list[torch.Tensor] = []
        saturated_chunks: list[torch.Tensor] = []
        chunk_metadata: list[dict[str, Any]] = []
        for start in range(0, config.repeats, config.psi_ne_chunk_repeats):
            stop = min(start + config.psi_ne_chunk_repeats, config.repeats)
            response = self._run_psi_ne_process(
                config,
                input_times=input_times,
                runtime_steps=runtime_steps,
                observation_step=observation_step,
                repeats=stop - start,
                trial_start=start,
            )
            chunk_calibration_loader = response["calibration_loader"]
            if calibration_loader is None:
                calibration_loader = chunk_calibration_loader
            elif calibration_loader != chunk_calibration_loader:
                raise RuntimeError("psi-ne calibration loader changed between chunks")
            baseline_chunks.append(response["baseline"])
            observed_chunks.append(response["observed"])
            spike_count_chunks.append(response["spike_count"])
            saturated_chunks.append(response["saturated"])
            chunk_metadata.append(
                {
                    **response["metadata"],
                    "trial_start": start,
                    "trial_stop": stop,
                    "batch_count": response["batch_count"],
                }
            )
            print(f"psi-ne acquisition trials={start}:{stop} complete", flush=True)
        identifiers = {
            json.dumps(item.get("chip_identifier"), sort_keys=True)
            for item in chunk_metadata
        }
        if len(identifiers) != 1:
            raise RuntimeError("psi-ne chip identifier changed between chunks")
        baseline = torch.cat(baseline_chunks, dim=0)
        observed = torch.cat(observed_chunks, dim=0)
        spike_count = torch.cat(spike_count_chunks, dim=0)
        saturated = torch.cat(saturated_chunks, dim=0)
        delivered = torch.isfinite(observed)
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
                "hxtorch_version": capabilities.get("hxtorch_version"),
                "chip_identifier": chunk_metadata[0].get("chip_identifier"),
                "calibration_loader": calibration_loader,
                "calibration_sha256": calibration_sha256(
                    config.spiking_calibration_path
                ),
                "observation_step": observation_step,
                "chunks": chunk_metadata,
                "resolved_parameters": {
                    "tau_mem_s": config.tau_mem_s,
                    "tau_syn_s": config.tau_syn_s,
                    "synaptic_weight": config.exponential_input_weight,
                    "input_fan_in": config.psi_ne_input_fan_in,
                    "leak": 80,
                    "reset": 80,
                    "threshold": 125,
                },
                "requested_threshold_enable": False,
                "threshold_validation": "reject-if-any-spike",
                "observation_value": "stimulated-minus-paired-quiet",
                "raw_spike_handle_required": False,
                "replicas_are_pooled": False,
            },
        )

    def _run_psi_ne_process(
        self,
        config: PrimitiveNoiseConfig,
        *,
        input_times: torch.Tensor,
        runtime_steps: int,
        observation_step: int,
        repeats: int,
        trial_start: int,
    ) -> dict[str, Any]:
        """Run one membrane-readout shard in a disposable child process."""
        fingerprint_payload = {
            "primitive": "psi-ne",
            "stage": "transfer",
            "trial_start": trial_start,
            "trial_stop": trial_start + repeats,
            "input_times": input_times.tolist(),
            "runtime_steps": runtime_steps,
            "observation_step": observation_step,
            "config": config.to_manifest_dict(),
        }
        fingerprint = sha256(
            json.dumps(
                fingerprint_payload,
                default=str,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_path: Path | None = None
        if config.pynn_worker_cache_dir is not None:
            config.pynn_worker_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = config.pynn_worker_cache_dir / (
                "psi-ne_transfer_trials"
                f"{trial_start}-{trial_start + repeats}_{fingerprint[:16]}.pt"
            )
            if cache_path.is_file():
                cached = torch.load(
                    cache_path, map_location="cpu", weights_only=False
                )
                if cached.get("fingerprint") != fingerprint:
                    raise RuntimeError(f"spiking worker cache mismatch: {cache_path}")
                response = dict(cached["response"])
                response["metadata"] = {
                    **response["metadata"],
                    "worker_cache_hit": True,
                }
                return response
        worker = Path(__file__).with_name("primitive_spiking_worker.py")
        with tempfile.TemporaryDirectory(prefix="bss2-spiking-worker-") as directory:
            root = Path(directory)
            request_path = root / "request.pt"
            response_path = root / "response.pt"
            torch.save(
                {
                    "config": config,
                    "input_times": input_times,
                    "runtime_steps": runtime_steps,
                    "observation_step": observation_step,
                    "repeats": repeats,
                },
                request_path,
            )
            try:
                completed = subprocess.run(
                    [sys.executable, str(worker), str(request_path), str(response_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=config.pynn_worker_timeout_s,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("spiking worker timed out") from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                raise RuntimeError(f"spiking worker failed: {detail}") from error
            if not response_path.is_file():
                raise RuntimeError("spiking worker did not write its response")
            response = torch.load(
                response_path, map_location="cpu", weights_only=False
            )
        response["metadata"] = {
            **response["metadata"],
            "worker_stdout": completed.stdout.strip(),
            "worker_cache_hit": False,
        }
        if cache_path is not None:
            temporary_cache = cache_path.with_suffix(".tmp")
            torch.save(
                {"fingerprint": fingerprint, "response": response},
                temporary_cache,
            )
            temporary_cache.replace(cache_path)
        return response

    def _run_psi_ne_chunk(
        self,
        hxsnn: Any,
        config: PrimitiveNoiseConfig,
        *,
        input_times: torch.Tensor,
        runtime_steps: int,
        observation_step: int,
        repeats: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str | None,
        int,
    ]:
        """Execute one bounded membrane-readout chunk on an active device."""
        inputs = self._psi_ne_input_events(
            input_times,
            repeats=repeats,
            runtime_steps=runtime_steps,
            dt_s=config.dt_s,
            fan_in=config.psi_ne_input_fan_in,
        )
        batch_count = inputs.shape[1]
        experiment = hxsnn.Experiment(dt=config.dt_s)
        calibration_loader = None
        if config.spiking_calibration_path is not None:
            calibration_loader = _configure_experiment_calibration(
                experiment, config.spiking_calibration_path
            )
        synapse = hxsnn.Synapse(
            in_features=config.psi_ne_input_fan_in,
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
        observables = lif(synapse(hxsnn.LIFObservables(spikes=inputs)))
        hxsnn.run(experiment, runtime_steps)
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
            raise RuntimeError(f"unexpected psi-ne CADC shape {tuple(cadc.shape)}")
        cadc = cadc.detach().cpu().to(torch.float64)
        spikes = spikes.detach().cpu()
        baseline = cadc[0::2, observation_step, :].reshape(
            repeats, input_times.numel(), config.device_count
        )
        stimulated = cadc[1::2, observation_step, :].reshape(
            repeats, input_times.numel(), config.device_count
        )
        observed = stimulated - baseline
        spike_count = spikes[1::2].sum(dim=1).to(torch.int64).reshape(
            repeats, input_times.numel(), config.device_count
        )
        saturated = (
            (baseline <= -128.0)
            | (baseline >= 127.0)
            | (stimulated <= -128.0)
            | (stimulated >= 127.0)
        )
        experiment.clear()
        return (
            baseline,
            observed,
            spike_count,
            saturated,
            calibration_loader,
            batch_count,
        )

    @staticmethod
    def _psi_ne_input_events(
        input_times: torch.Tensor,
        *,
        repeats: int,
        runtime_steps: int,
        dt_s: float,
        fan_in: int,
    ) -> torch.Tensor:
        """Build paired quiet/stimulated input batches for membrane readout."""
        if repeats <= 0 or runtime_steps <= 0 or dt_s <= 0 or fan_in <= 0:
            raise ValueError("psi-ne event dimensions and timing must be positive")
        pair_count = repeats * input_times.numel()
        inputs = torch.zeros(
            (runtime_steps, 2 * pair_count, fan_in), dtype=torch.float32
        )
        for pair, input_time in enumerate(input_times.repeat(repeats)):
            input_step = int(round(float(input_time) / dt_s))
            if not 0 <= input_step < runtime_steps:
                raise ValueError("psi-ne input time lies outside the runtime")
            inputs[input_step, 2 * pair + 1, :] = 1.0
        return inputs

    def _collect_psi_ed(
        self,
        config: PrimitiveNoiseConfig,
        *,
        quick: bool,
    ) -> PrimitiveObservation:
        """Measure the causal correlation sensor with paired quiet periods."""
        if config.correlation_calibration_path is None:
            raise ValueError(
                "psi-ed hardware collection requires a correlation calibration"
            )
        capabilities = probe_primitive_capabilities()
        if not capabilities["psi-ed"]:
            raise PrimitiveCapabilityError(
                "psi-ed requires the PyNN causal correlation sensor; "
                f"capabilities={capabilities}"
            )
        differences = (
            torch.tensor(
                [config.psi_ed_delta_min_s, 0.0, config.psi_ed_delta_max_s],
                dtype=torch.float64,
            )
            if quick
            else torch.linspace(
                config.psi_ed_delta_min_s,
                config.psi_ed_delta_max_s,
                config.psi_ed_delta_points,
                dtype=torch.float64,
            )
        )
        quiet_chunks: list[torch.Tensor] = []
        stimulated_chunks: list[torch.Tensor] = []
        first_chunks: list[torch.Tensor] = []
        count_chunks: list[torch.Tensor] = []
        chunk_metadata: list[dict[str, Any]] = []
        for start in range(0, config.repeats, config.psi_ed_chunk_repeats):
            stop = min(start + config.psi_ed_chunk_repeats, config.repeats)
            response = self._run_psi_ed_process(
                config,
                differences=differences,
                repeats=stop - start,
                trial_start=start,
            )
            expected = (
                stop - start,
                differences.numel(),
                config.device_count,
            )
            for name in ("quiet", "stimulated", "first", "count"):
                if tuple(response[name].shape) != expected:
                    raise RuntimeError(
                        f"unexpected psi-ed {name} shape "
                        f"{tuple(response[name].shape)}; expected {expected}"
                    )
            quiet_chunks.append(response["quiet"].to(torch.float64))
            stimulated_chunks.append(response["stimulated"].to(torch.float64))
            first_chunks.append(response["first"].to(torch.float64))
            count_chunks.append(response["count"].to(torch.int64))
            chunk_metadata.append(
                {
                    **response["metadata"],
                    "trial_start": start,
                    "trial_stop": stop,
                }
            )
            print(
                f"Correlation acquisition trials={start}:{stop} complete",
                flush=True,
            )
        quiet = torch.cat(quiet_chunks, dim=0)
        stimulated = torch.cat(stimulated_chunks, dim=0)
        first = torch.cat(first_chunks, dim=0)
        count = torch.cat(count_chunks, dim=0)
        (
            stored_trial_order,
            calibration_acquisition_indices,
            validation_acquisition_indices,
        ) = self._split_trial_order(
            config.repeats,
            config.calibration_repeats,
            seed=config.seed,
        )
        quiet = quiet[stored_trial_order]
        stimulated = stimulated[stored_trial_order]
        first = first[stored_trial_order]
        count = count[stored_trial_order]
        delivered = torch.isfinite(first) & (first <= config.deadline_s)
        observed = quiet - stimulated
        observed = torch.where(
            delivered, observed, torch.full_like(observed, torch.nan)
        )
        saturated = (
            (quiet <= 0.0)
            | (quiet >= 255.0)
            | (stimulated <= 0.0)
            | (stimulated >= 255.0)
        )
        identifiers = {
            json.dumps(item.get("chip_identifier"), sort_keys=True)
            for item in chunk_metadata
        }
        if len(identifiers) != 1:
            raise RuntimeError("psi-ed chip identifier changed between chunks")
        return PrimitiveObservation(
            primitive="psi-ed",
            stage="transfer",
            output_kind="cadc-potential",
            input_code=differences / config.dt_s,
            ideal_variable=differences,
            observed=observed,
            delivered=delivered,
            spike_count=count,
            saturated=saturated,
            physical_coordinates=config.physical_coordinates,
            fit_point_mask=torch.ones_like(differences, dtype=torch.bool),
            baseline=quiet,
            first_spike_time_s=first,
            metadata={
                "backend": "brainscales2-correlation-hardware",
                "pynn_version": capabilities.get("pynn_version"),
                "chip_identifier": chunk_metadata[0].get("chip_identifier"),
                "calibration_sha256": calibration_sha256(
                    config.correlation_calibration_path
                ),
                "chunks": chunk_metadata,
                "calibration_acquisition_indices": list(
                    calibration_acquisition_indices
                ),
                "validation_acquisition_indices": list(
                    validation_acquisition_indices
                ),
                "correlation_observation": "quiet-minus-stimulated-causal",
                "deadline_applies": True,
                "multiple_spike_handling": "first-spike-time",
                "replicas_are_pooled": False,
                "coordinate_semantics": "atomic-neuron-on-dls",
                "resolved_parameters": {
                    "separation_center_s": config.psi_ed_separation_center_s,
                    "post_time_s": config.psi_ed_post_time_s,
                    "readout_time_s": config.psi_ed_readout_time_s,
                    "deadline_s": config.deadline_s,
                    "trial_guard_s": config.psi_ed_trial_guard_s,
                    "trigger_fan_in": config.psi_ed_trigger_fan_in,
                    "trigger_weight": config.psi_ed_trigger_weight,
                    "plastic_weight": config.psi_ed_plastic_weight,
                },
            },
        )

    def _run_psi_ed_process(
        self,
        config: PrimitiveNoiseConfig,
        *,
        differences: torch.Tensor,
        repeats: int,
        trial_start: int,
    ) -> dict[str, Any]:
        """Run one bounded correlation-sensor shard in a child process."""
        fingerprint_payload = {
            "primitive": "psi-ed",
            "stage": "transfer",
            "trial_start": trial_start,
            "trial_stop": trial_start + repeats,
            "differences": differences.tolist(),
            "config": config.to_manifest_dict(),
        }
        fingerprint = sha256(
            json.dumps(
                fingerprint_payload,
                default=str,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_path: Path | None = None
        if config.pynn_worker_cache_dir is not None:
            config.pynn_worker_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = config.pynn_worker_cache_dir / (
                "psi-ed_transfer_trials"
                f"{trial_start}-{trial_start + repeats}_{fingerprint[:16]}.pt"
            )
            if cache_path.is_file():
                cached = torch.load(
                    cache_path, map_location="cpu", weights_only=False
                )
                if cached.get("fingerprint") != fingerprint:
                    raise RuntimeError(
                        f"correlation worker cache mismatch: {cache_path}"
                    )
                response = dict(cached["response"])
                response["metadata"] = {
                    **response["metadata"],
                    "worker_cache_hit": True,
                }
                return response
        worker = Path(__file__).with_name("primitive_correlation_worker.py")
        with tempfile.TemporaryDirectory(
            prefix="bss2-correlation-worker-"
        ) as directory:
            root = Path(directory)
            request_path = root / "request.pt"
            response_path = root / "response.pt"
            torch.save(
                {
                    "config": config,
                    "differences": differences,
                    "repeats": repeats,
                    "trial_start": trial_start,
                },
                request_path,
            )
            try:
                completed = subprocess.run(
                    [sys.executable, str(worker), str(request_path), str(response_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=config.pynn_worker_timeout_s,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("correlation worker timed out") from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                raise RuntimeError(
                    f"correlation worker failed: {detail}"
                ) from error
            if not response_path.is_file():
                raise RuntimeError("correlation worker did not write its response")
            response = torch.load(
                response_path, map_location="cpu", weights_only=False
            )
        response["metadata"] = {
            **response["metadata"],
            "worker_stdout": completed.stdout.strip(),
            "worker_cache_hit": False,
        }
        if cache_path is not None:
            temporary_cache = cache_path.with_suffix(".tmp")
            torch.save(
                {"fingerprint": fingerprint, "response": response},
                temporary_cache,
            )
            temporary_cache.replace(cache_path)
        return response

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
        (
            stored_trial_order,
            calibration_acquisition_indices,
            validation_acquisition_indices,
        ) = self._split_trial_order(
            config.repeats,
            config.calibration_repeats,
            seed=config.seed,
        )
        per_code_metadata: list[dict[str, Any]] = []
        for point, code in enumerate(codes.tolist()):
            first_chunks: list[torch.Tensor] = []
            count_chunks: list[torch.Tensor] = []
            precharge_chunks: list[torch.Tensor] = []
            chunk_metadata: list[dict[str, Any]] = []
            process_repeats = config.pynn_process_repeats or config.repeats
            for start in range(0, config.repeats, process_repeats):
                stop = min(start + process_repeats, config.repeats)
                if config.pynn_process_repeats is None:
                    first, count, precharge, metadata = self._run_pynn_code_chunks(
                        primitive,
                        stage,
                        int(code),
                        config,
                        repeats=stop - start,
                    )
                else:
                    first, count, precharge, metadata = (
                        self._run_pynn_code_process(
                            primitive,
                            stage,
                            int(code),
                            config,
                            repeats=stop - start,
                            trial_start=start,
                        )
                    )
                first_chunks.append(first)
                count_chunks.append(count)
                if precharge is not None:
                    precharge_chunks.append(precharge)
                chunk_metadata.append(
                    {
                        **metadata,
                        "trial_start": start,
                        "trial_stop": stop,
                    }
                )
                print(
                    f"PyNN acquisition code={int(code)} "
                    f"trials={start}:{stop} complete",
                    flush=True,
                )
            observed[:, point, :] = torch.cat(first_chunks, dim=0)[
                stored_trial_order
            ]
            spike_count[:, point, :] = torch.cat(count_chunks, dim=0)[
                stored_trial_order
            ]
            if precharge_cadc is not None:
                if not precharge_chunks:
                    raise PrimitiveCapabilityError(
                        "dynamic precharge did not expose a CADC membrane trace"
                    )
                precharge_cadc[:, point, :] = torch.cat(
                    precharge_chunks, dim=0
                )[stored_trial_order]
            per_code_metadata.append(
                {
                    "input_code": int(code),
                    "chunks": chunk_metadata,
                }
            )
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
                "chip_identifier": per_code_metadata[0]["chunks"][0].get(
                    "chip_identifier"
                ),
                "constant_current_playback": primitive == "phi-np",
                "chunk_repeats": config.pynn_chunk_repeats,
                "process_repeats": config.pynn_process_repeats,
                "calibration_acquisition_indices": list(
                    calibration_acquisition_indices
                ),
                "validation_acquisition_indices": list(
                    validation_acquisition_indices
                ),
                "automatic_fallback": False,
                "per_code": per_code_metadata,
                "replicas_are_pooled": False,
            },
        )

    def _run_pynn_code_chunks(
        self,
        primitive: PrimitiveKind,
        stage: PrimitiveStage,
        code: int,
        config: PrimitiveNoiseConfig,
        *,
        repeats: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Collect one process shard through bounded PyNN schedules."""
        first_chunks: list[torch.Tensor] = []
        count_chunks: list[torch.Tensor] = []
        precharge_chunks: list[torch.Tensor] = []
        metadata_chunks: list[dict[str, Any]] = []
        for start in range(0, repeats, config.pynn_chunk_repeats):
            stop = min(start + config.pynn_chunk_repeats, repeats)
            first, count, precharge, metadata = self._run_pynn_code(
                primitive,
                stage,
                code,
                config,
                repeats=stop - start,
            )
            first_chunks.append(first)
            count_chunks.append(count)
            if precharge is not None:
                precharge_chunks.append(precharge)
            metadata_chunks.append(
                {**metadata, "trial_start": start, "trial_stop": stop}
            )
        precharge_result = (
            torch.cat(precharge_chunks, dim=0) if precharge_chunks else None
        )
        return (
            torch.cat(first_chunks, dim=0),
            torch.cat(count_chunks, dim=0),
            precharge_result,
            {
                "chip_identifier": metadata_chunks[0].get("chip_identifier"),
                "chunks": metadata_chunks,
            },
        )

    def _run_pynn_code_process(
        self,
        primitive: PrimitiveKind,
        stage: PrimitiveStage,
        code: int,
        config: PrimitiveNoiseConfig,
        *,
        repeats: int,
        trial_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Isolate PyNN native allocations in a bounded child process."""
        fingerprint_payload = {
            "primitive": primitive,
            "stage": stage,
            "code": code,
            "trial_start": trial_start,
            "trial_stop": trial_start + repeats,
            "config": config.to_manifest_dict(),
        }
        fingerprint = sha256(
            json.dumps(
                fingerprint_payload,
                default=str,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_path: Path | None = None
        if config.pynn_worker_cache_dir is not None:
            config.pynn_worker_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = config.pynn_worker_cache_dir / (
                f"{primitive}_{stage}_code{code}_trials"
                f"{trial_start}-{trial_start + repeats}_{fingerprint[:16]}.pt"
            )
            if cache_path.is_file():
                cached = torch.load(
                    cache_path, map_location="cpu", weights_only=False
                )
                if cached.get("fingerprint") != fingerprint:
                    raise RuntimeError(f"PyNN worker cache mismatch: {cache_path}")
                metadata = dict(cached["metadata"])
                metadata["worker_cache_hit"] = True
                return (
                    cached["first"],
                    cached["count"],
                    cached["precharge"],
                    metadata,
                )
        worker = Path(__file__).with_name("primitive_pynn_worker.py")
        with tempfile.TemporaryDirectory(prefix="bss2-pynn-worker-") as directory:
            root = Path(directory)
            request_path = root / "request.pt"
            response_path = root / "response.pt"
            torch.save(
                {
                    "primitive": primitive,
                    "stage": stage,
                    "code": code,
                    "config": config,
                    "repeats": repeats,
                },
                request_path,
            )
            try:
                completed = subprocess.run(
                    [sys.executable, str(worker), str(request_path), str(response_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=config.pynn_worker_timeout_s,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"PyNN worker timed out for {primitive}/{stage}/code={code}"
                ) from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                raise RuntimeError(
                    f"PyNN worker failed for {primitive}/{stage}/code={code}: {detail}"
                ) from error
            if not response_path.is_file():
                raise RuntimeError("PyNN worker did not write its response")
            response = torch.load(
                response_path, map_location="cpu", weights_only=False
            )
        metadata = dict(response["metadata"])
        metadata["worker_stdout"] = completed.stdout.strip()
        metadata["worker_cache_hit"] = False
        if cache_path is not None:
            cache_payload = {
                "fingerprint": fingerprint,
                "first": response["first"],
                "count": response["count"],
                "precharge": response["precharge"],
                "metadata": metadata,
            }
            temporary_cache = cache_path.with_suffix(".tmp")
            torch.save(cache_payload, temporary_cache)
            temporary_cache.replace(cache_path)
        return (
            response["first"],
            response["count"],
            response["precharge"],
            metadata,
        )

    @staticmethod
    def _split_trial_order(
        repeats: int,
        calibration_repeats: int,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
        """Return a seeded held-out split ordered for calibration-first storage."""
        if not 0 < calibration_repeats < repeats:
            raise ValueError("trial split must leave calibration and validation data")
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(repeats, generator=generator)
        calibration = permutation[:calibration_repeats].sort().values
        validation = permutation[calibration_repeats:].sort().values
        stored = torch.cat((calibration, validation))
        return (
            stored,
            tuple(int(index) for index in calibration),
            tuple(int(index) for index in validation),
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
    def _pynn_component(pynn: Any, name: str, *namespaces: str) -> Any:
        candidate = getattr(pynn, name, None)
        if candidate is not None:
            return candidate
        for namespace_name in namespaces:
            namespace = getattr(pynn, namespace_name, None)
            candidate = getattr(namespace, name, None)
            if candidate is not None:
                return candidate
        raise PrimitiveCapabilityError(
            f"installed PyNN exposes no {name} component"
        )

    @staticmethod
    def _reset_code(
        code: int, config: PrimitiveNoiseConfig
    ) -> int | tuple[int, ...]:
        if config.reset_code_table is not None:
            return config.reset_code_table[code]
        if code == 31:
            return config.threshold_code
        fraction = code / 30.0
        return int(
            round(
                config.reset_code_minimum
                + fraction * (config.reset_code_maximum - config.reset_code_minimum)
            )
        )

    @staticmethod
    def _set_population_reset(
        population: Any, reset_code: int | tuple[int, ...]
    ) -> None:
        """Apply a scalar or per-device reset code through the PyNN API."""
        population.set(
            reset_v_reset=(
                list(reset_code) if isinstance(reset_code, tuple) else reset_code
            )
        )

    @staticmethod
    def _selected_refractory_parameters(
        settings: Any, coordinates: list[int]
    ) -> dict[str, list[int]]:
        """Select per-circuit counter settings produced by Calix."""
        fields = {
            "refractory_period_refractory_time": "refractory_counters",
            "refractory_period_reset_holdoff": "reset_holdoff",
            "refractory_period_input_clock": "input_clock",
        }
        selected: dict[str, list[int]] = {}
        for parameter, attribute in fields.items():
            values = getattr(settings, attribute)
            if len(values) != 512:
                raise RuntimeError(
                    f"Calix {attribute} does not cover all 512 circuits"
                )
            selected[parameter] = [int(values[index]) for index in coordinates]
        return selected

    @classmethod
    def _configure_first_spike_refractory(
        cls,
        chip: Any,
        config: PrimitiveNoiseConfig,
    ) -> tuple[dict[str, list[int]], dict[str, Any]]:
        """Keep each encoder circuit silent after its first recorded spike."""
        numpy = import_module("numpy")
        quantities = import_module("quantities")
        refractory_period = import_module("calix.spiking.refractory_period")
        # Acquisition retains one deadline of quiet time after the observation
        # interval.  Cover the complete repeated-trial window so that an early
        # first spike cannot be followed by another spike in the quiet interval.
        refractory_target_s = 2.0 * config.deadline_s
        targets = numpy.full(512, refractory_target_s) * quantities.s
        settings = refractory_period.calculate_settings(targets)
        settings.apply_to_chip(chip)
        parameters = cls._selected_refractory_parameters(
            settings, list(config.physical_coordinates)
        )
        metadata = {
            "target_s": refractory_target_s,
            "fast_clock": int(settings.fast_clock),
            "slow_clock": int(settings.slow_clock),
            "selected_parameters": parameters,
        }
        return parameters, metadata

    @staticmethod
    def _pynn_reset_injection(
        pynn: Any,
        chip: Any,
        *,
        trials: int,
        window_ms: float,
        release_ms: float,
    ) -> Any:
        """Schedule backend reset pulses without overwriting neuron routing."""
        vx = import_module("dlens_vx_v3")
        builder = vx.sta.AbsoluteTimePlaybackProgramBuilder()
        cycles_per_us = int(vx.hal.Timer.Value.fpga_clock_cycles_per_us)
        for trial in range(trials):
            base_us = trial * window_ms * 1.0e3
            for index, backend in enumerate(chip.neuron_block.backends):
                coordinate = vx.halco.CommonNeuronBackendConfigOnDLS(
                    vx.halco.common.Enum(index)
                )
                forced = copy.deepcopy(backend)
                forced.force_reset = True
                normal = copy.deepcopy(backend)
                normal.force_reset = False
                builder.write(
                    vx.hal.Timer.Value(round((base_us + 1.0) * cycles_per_us)),
                    coordinate,
                    forced,
                )
                builder.write(
                    vx.hal.Timer.Value(
                        round((base_us + release_ms * 1.0e3) * cycles_per_us)
                    ),
                    coordinate,
                    normal,
                )
        injection = pynn.InjectedConfiguration()
        injection.inside_realtime = builder
        return injection

    @staticmethod
    def _record_pynn_population(population: Any, *, membrane: bool) -> None:
        """Keep event recording separate from dense membrane recording."""
        population.record("spikes")
        if membrane:
            population.record("v", device="cadc")

    def _run_pynn_code(
        self,
        primitive: PrimitiveKind,
        stage: PrimitiveStage,
        code: int,
        config: PrimitiveNoiseConfig,
        *,
        repeats: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        resolved_repeats = config.repeats if repeats is None else repeats
        if resolved_repeats <= 0:
            raise ValueError("PyNN acquisition repeats must be positive")
        pynn = import_module(PYNN_BACKEND_MODULE)
        cell_type = self._hxneuron_type(pynn)
        spike_source_type = self._pynn_component(
            pynn, "SpikeSourceArray", "cells", "standardmodels"
        )
        static_synapse_type = self._pynn_component(
            pynn, "StaticSynapse", "synapses", "standardmodels"
        )
        chip = pynn.helper.chip_from_file(str(config.spiking_calibration_path))
        coordinates = self._pynn_coordinates(config)
        refractory_parameters, refractory_metadata = (
            self._configure_first_spike_refractory(chip, config)
        )
        setup_complete = False
        # A full-deadline quiet interval follows the observation deadline.  The
        # Calix refractory target can therefore suppress every later spike without
        # carrying state into the next repeated trial.
        window_ms = 2.0 * config.deadline_s * 1.0e3
        reference_ms = config.input_early_s * 1.0e3
        ramp_stop_ms = config.input_late_s * 1.0e3
        precharge_ms = max(0.0005, reference_ms - 0.002)
        total_trials = resolved_repeats + 1
        reset_release_ms = (
            0.002 if stage == "dynamic" or primitive == "phi-nl" else 0.004
        )
        injection = self._pynn_reset_injection(
            pynn,
            chip,
            trials=total_trials,
            window_ms=window_ms,
            release_ms=reset_release_ms,
        )
        try:
            pynn.setup(
                initial_config=chip,
                neuronPermutation=coordinates,
                injected_config=injection,
            )
            setup_complete = True
            resolved_reset_code = (
                self._reset_code(code, config)
                if stage == "static"
                else config.reset_code_minimum
            )
            scalar_reset_code = (
                resolved_reset_code[0]
                if isinstance(resolved_reset_code, tuple)
                else resolved_reset_code
            )
            cell_parameters = {
                "leak_v_leak": (
                    scalar_reset_code
                    if stage == "static"
                    else config.reset_code_minimum
                ),
                "leak_i_bias": config.leak_bias,
                "leak_enable_division": True,
                "threshold_enable": True,
                "threshold_v_threshold": config.threshold_code,
                "reset_v_reset": (
                    scalar_reset_code
                    if stage == "static"
                    else config.reset_code_minimum
                ),
                "constant_current_enable": False,
                "constant_current_i_offset": config.constant_current_code,
                "reset_i_bias": 1022,
                "reset_enable_multiplication": True,
                "refractory_period_enable_pause": True,
                **refractory_parameters,
            }
            population = pynn.Population(
                config.device_count,
                cell_type(**cell_parameters),
            )
            if isinstance(resolved_reset_code, tuple):
                population.set(leak_v_leak=list(resolved_reset_code))
                self._set_population_reset(population, resolved_reset_code)
            uses_precharge = stage == "dynamic" or primitive == "phi-nl"
            record_variables: list[str] = ["spikes"]
            if uses_precharge:
                record_variables.append("v")
            self._record_pynn_population(population, membrane=uses_precharge)

            if uses_precharge:
                precharge_times = [
                    trial * window_ms + precharge_ms
                    for trial in range(total_trials)
                ]
                source = pynn.Population(
                    config.precharge_input_fan_in,
                    spike_source_type(spike_times=precharge_times),
                )
                weight = max(
                    1,
                    int(round(config.precharge_weight_maximum * code / 31.0)),
                )
                pynn.Projection(
                    source,
                    population,
                    pynn.AllToAllConnector(),
                    synapse_type=static_synapse_type(weight=weight),
                    receptor_type="excitatory",
                )
            else:
                weight = None

            if primitive == "phi-nl":
                exponential_times = [
                    trial * window_ms + reference_ms
                    for trial in range(total_trials)
                ]
                exponential_source = pynn.Population(
                    config.exponential_input_fan_in,
                    spike_source_type(spike_times=exponential_times),
                )
                pynn.Projection(
                    exponential_source,
                    population,
                    pynn.AllToAllConnector(),
                    synapse_type=static_synapse_type(
                        weight=config.exponential_input_weight
                    ),
                    receptor_type="excitatory",
                )
                pynn.run(total_trials * window_ms)
            else:
                append = pynn.RunCommand.APPEND
                execute = pynn.RunCommand.EXECUTE
                for trial in range(total_trials):
                    # The code defines the initial membrane state, not the reset
                    # state after a spike. Restore it before each injected reset,
                    # then use the common minimum while the ramp is active. This
                    # preserves the first crossing and suppresses code-dependent
                    # repeated spikes during long timing windows.
                    if stage == "static" and trial > 0:
                        self._set_population_reset(population, resolved_reset_code)
                    pynn.run(reference_ms, append)
                    if stage == "static":
                        self._set_population_reset(
                            population, config.reset_code_minimum
                        )
                    population.set(constant_current_enable=True)
                    pynn.run(ramp_stop_ms - reference_ms, append)
                    population.set(constant_current_enable=False)
                    command = execute if trial == total_trials - 1 else append
                    pynn.run(window_ms - ramp_stop_ms, command)

            data = population.get_data(record_variables)
            segment = data.segments[-1]
            first, count = self._decode_pynn_spikes(
                segment.spiketrains,
                repeats=total_trials,
                devices=config.device_count,
                window_ms=window_ms,
            )
            precharge = None
            if uses_precharge:
                precharge_probe = self._decode_pynn_precharge(
                    segment,
                    devices=config.device_count,
                    sample_ms=reference_ms - 0.0005,
                )
            first = first[1:]
            count = count[1:]
            if uses_precharge:
                precharge = torch.full(
                    (resolved_repeats, config.device_count),
                    torch.nan,
                    dtype=torch.float64,
                )
                precharge[0] = precharge_probe
            identifier = pynn.helper.get_unique_identifier()
            chip_identifier = (
                [str(item) for item in identifier]
                if isinstance(identifier, (tuple, list))
                else [str(identifier)]
            )
            return first, count, precharge, {
                "input_code": code,
                "reset_code": resolved_reset_code,
                "precharge_weight": weight,
                "precharge_input_fan_in": (
                    config.precharge_input_fan_in if uses_precharge else None
                ),
                "resolved_cell_parameters": cell_parameters,
                "window_ms": window_ms,
                "observation_deadline_ms": config.deadline_s * 1.0e3,
                "reference_ms": reference_ms,
                "reset_release_ms": reset_release_ms,
                "first_spike_refractory": refractory_metadata,
                "static_reset_policy": (
                    "initial state code during reset; configured minimum during ramp"
                    if stage == "static"
                    else "configured minimum"
                ),
                "discarded_warmup_trials": 1,
                "precharge_cadc_acquisitions": 1 if uses_precharge else 0,
                "exponential_input_fan_in": (
                    config.exponential_input_fan_in
                    if primitive == "phi-nl"
                    else None
                ),
                "raw_time_unit": "PyNN-ms mapped to seconds",
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
        window_ms: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trains = list(spike_trains)
        if not trains:
            raise RuntimeError("PyNN returned no spike-train records")
        source_indices: list[int] = []
        for position, train in enumerate(trains):
            annotations = getattr(train, "annotations", {})
            source_index = annotations.get("source_index")
            if source_index is None:
                if len(trains) != devices:
                    raise RuntimeError(
                        "segmented PyNN spike trains do not expose source_index"
                    )
                source_index = position
            source_index = int(source_index)
            if not 0 <= source_index < devices:
                raise RuntimeError(
                    f"PyNN spike train has invalid source_index {source_index}"
                )
            source_indices.append(source_index)
        if set(source_indices) != set(range(devices)):
            raise RuntimeError("PyNN spike trains do not cover every physical device")
        first = torch.full((repeats, devices), torch.nan, dtype=torch.float64)
        count = torch.zeros((repeats, devices), dtype=torch.int64)
        for device, train in zip(source_indices, trains):
            magnitude = getattr(train, "magnitude", train)
            for time_value in torch.as_tensor(magnitude, dtype=torch.float64).reshape(-1):
                time_ms = float(time_value)
                trial = int(math.floor(time_ms / window_ms))
                if not 0 <= trial < repeats:
                    raise RuntimeError("PyNN spike lies outside the scheduled windows")
                physical_time_s = (time_ms - trial * window_ms) * 1.0e-3
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
        devices: int,
        sample_ms: float,
    ) -> torch.Tensor:
        signals = list(getattr(segment, "irregularlysampledsignals", []))
        if not signals:
            signals = list(getattr(segment, "analogsignals", []))
        if not signals:
            raise PrimitiveCapabilityError("PyNN returned no CADC membrane signal")
        by_device: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {
            device: [] for device in range(devices)
        }
        for signal in signals:
            values = torch.as_tensor(
                getattr(signal, "magnitude", signal), dtype=torch.float64
            )
            times = torch.as_tensor(
                getattr(getattr(signal, "times", None), "magnitude", []),
                dtype=torch.float64,
            )
            if values.ndim != 2 or times.numel() != values.shape[0]:
                raise RuntimeError("PyNN CADC signal has no usable time axis")
            source_ids = list(getattr(signal, "annotations", {}).get("source_ids", []))
            if len(source_ids) != values.shape[1]:
                raise RuntimeError("PyNN CADC signal does not identify its sources")
            for column, source_id in enumerate(source_ids):
                device = int(source_id)
                if not 0 <= device < devices:
                    raise RuntimeError(
                        f"PyNN CADC signal has invalid source id {device}"
                    )
                by_device[device].append((times, values[:, column]))

        selected = torch.full((devices,), torch.nan, dtype=torch.float64)
        for device, chunks in by_device.items():
            if not chunks:
                raise RuntimeError(
                    f"PyNN CADC signals do not cover physical device {device}"
                )
            times = torch.cat([chunk[0] for chunk in chunks])
            values = torch.cat([chunk[1] for chunk in chunks])
            index = int(torch.argmin(torch.abs(times - sample_ms)))
            selected[device] = values[index]
        return selected
