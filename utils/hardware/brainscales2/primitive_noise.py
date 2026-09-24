"""Independent marginal-noise characterization for five TTFS primitives.

The module deliberately analyzes one physical circuit at a time.  Device
replicas are used to expose fixed-pattern variation and are never averaged into
one logical output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import math

import torch

from .backend import calibration_sha256


PrimitiveKind = Literal["phi-np", "phi-nl", "psi-int", "psi-ne", "psi-ed"]
PrimitiveStage = Literal["static", "dynamic", "transfer"]
RESET_ASSERT_S = 1.0e-6
OutputKind = Literal["spike-time-s", "pwm-code", "cadc-potential"]

PRIMITIVES: tuple[PrimitiveKind, ...] = (
    "phi-np",
    "phi-nl",
    "psi-int",
    "psi-ne",
    "psi-ed",
)

_VALIDATED_16_CIRCUIT_COORDINATES: tuple[int, ...] = (
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


def default_primitive_coordinates(device_count: int = 16) -> tuple[int, ...]:
    """Return atomic-neuron coordinates without pooling physical outputs."""
    if device_count <= 0 or device_count > 512:
        raise ValueError("device_count must lie in [1, 512]")
    if device_count == len(_VALIDATED_16_CIRCUIT_COORDINATES):
        coordinates = _VALIDATED_16_CIRCUIT_COORDINATES
    else:
        coordinates = tuple(
            (index % 4) * 128 + index // 4 for index in range(device_count)
        )
    if len(set(coordinates)) != device_count or any(
        coordinate < 0 or coordinate >= 512 for coordinate in coordinates
    ):
        raise RuntimeError("primitive placement produced invalid coordinates")
    return coordinates


@dataclass(frozen=True)
class PrimitiveNoiseConfig:
    """Collection, fitting, validation, and physical operating-point settings."""

    repeats: int = 256
    calibration_repeats: int = 128
    device_count: int = 16
    seed: int = 0
    dt_s: float = 1.0e-6
    input_early_s: float = 5.0e-6
    input_late_s: float = 25.0e-6
    observation_time_s: float = 40.0e-6
    deadline_s: float = 60.0e-6
    spiking_calibration_path: Path | None = None
    hagen_calibration_path: Path | None = None
    correlation_calibration_path: Path | None = None
    allow_environment_calibration: bool = False
    physical_coordinates: tuple[int, ...] = field(
        default_factory=default_primitive_coordinates
    )
    normalized_rmse_limit: float = 0.05
    parameter_drift_limit: float = 0.10
    multiple_spike_rate_limit: float = 0.001
    deadline_miss_rate_limit: float = 0.01
    saturation_rate_limit: float = 0.01
    monotonicity_minimum: float = 0.95
    # Raw hardware controls.  They are recorded verbatim and are not treated as
    # calibrated physical units.
    reset_code_minimum: int = 300
    reset_code_maximum: int = 900
    reset_code_table: (
        tuple[int, ...] | tuple[tuple[int, ...], ...] | None
    ) = None
    threshold_code: int = 600
    leak_bias: int = 0
    constant_current_code: int = 1022
    reset_current_code: int = 1022
    reset_current_enable_multiplication: bool = True
    static_reset_release_s: float = 4.0e-6
    dynamic_reset_release_s: float = 2.0e-6
    membrane_capacitance_code: int | None = None
    threshold_comparator_bias_code: int | None = None
    excitatory_input_i_bias_tau_code: int | None = None
    excitatory_input_i_bias_gm_code: int | None = None
    synaptic_input_drop_bias_code: int | None = None
    precharge_weight_maximum: int = 63
    precharge_input_fan_in: int = 1
    record_precharge_cadc: bool = True
    exponential_input_weight: int = 63
    exponential_input_fan_in: int = 8
    phi_nl_lower_bound_minimum_code: float = -128.0
    phi_nl_lower_bound_grid_size: int = 2048
    pynn_chunk_repeats: int = 8
    pynn_process_repeats: int | None = 32
    pynn_worker_timeout_s: float = 300.0
    pynn_worker_max_attempts: int = 3
    pynn_worker_cache_dir: Path | None = None
    psi_ne_input_fan_in: int = 2
    psi_ne_chunk_repeats: int = 4
    psi_ed_delta_min_s: float = -10.0e-6
    psi_ed_delta_max_s: float = 10.0e-6
    psi_ed_delta_points: int = 11
    psi_ed_separation_center_s: float = 15.0e-6
    psi_ed_post_time_s: float = 30.0e-6
    psi_ed_readout_time_s: float = 58.0e-6
    psi_ed_trial_guard_s: float = 1.0e-3
    psi_ed_trigger_fan_in: int = 8
    psi_ed_trigger_weight: int = 63
    psi_ed_plastic_weight: int = 0
    psi_ed_chunk_repeats: int = 32
    tau_mem_s: float = 100.0e-6
    tau_syn_s: float = 1.0e-6
    hagen_wait_between_events: int = 5
    hagen_num_sends: int | None = 1024
    hagen_candidate_count: int = 128
    hagen_chunk_repeats: int = 16

    def __post_init__(self) -> None:
        if self.repeats < 2:
            raise ValueError("repeats must be at least two")
        if not 0 < self.calibration_repeats < self.repeats:
            raise ValueError("calibration_repeats must leave a held-out split")
        if self.device_count != len(self.physical_coordinates):
            raise ValueError("device_count does not match physical coordinates")
        if len(set(self.physical_coordinates)) != self.device_count:
            raise ValueError("physical coordinates contain duplicates")
        if any(not 0 <= item < 512 for item in self.physical_coordinates):
            raise ValueError("physical coordinate lies outside [0, 511]")
        if not 0 < self.input_early_s < self.input_late_s < self.deadline_s:
            raise ValueError("input window must lie before the deadline")
        if not self.input_late_s < self.observation_time_s < self.deadline_s:
            raise ValueError("observation time must follow the input window")
        if self.dt_s <= 0 or self.tau_mem_s <= 0 or self.tau_syn_s <= 0:
            raise ValueError("time constants and dt must be positive")
        for name, value in (
            ("reset code minimum", self.reset_code_minimum),
            ("reset code maximum", self.reset_code_maximum),
            ("threshold code", self.threshold_code),
            ("leak bias", self.leak_bias),
            ("constant current code", self.constant_current_code),
            ("reset current code", self.reset_current_code),
        ):
            if not 0 <= value <= 1022:
                raise ValueError(f"{name} must lie in [0, 1022]")
        if not isinstance(self.reset_current_enable_multiplication, bool):
            raise TypeError("reset current multiplication flag must be a bool")
        if not RESET_ASSERT_S < self.static_reset_release_s < self.input_early_s:
            raise ValueError(
                "static reset release must follow reset assertion and precede "
                "the ramp start"
            )
        precharge_time_s = max(0.5e-6, self.input_early_s - 2.0e-6)
        if (
            not RESET_ASSERT_S < self.dynamic_reset_release_s < precharge_time_s
            or math.isclose(
                self.dynamic_reset_release_s,
                precharge_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "dynamic reset release must follow reset assertion and precede "
                "the precharge event"
            )
        if self.membrane_capacitance_code is not None and not (
            0 <= self.membrane_capacitance_code <= 63
        ):
            raise ValueError("membrane capacitance code must lie in [0, 63]")
        if self.threshold_comparator_bias_code is not None and not (
            0 <= self.threshold_comparator_bias_code <= 1022
        ):
            raise ValueError(
                "threshold comparator bias code must lie in [0, 1022]"
            )
        for name, value in (
            (
                "excitatory synaptic input time constant bias code",
                self.excitatory_input_i_bias_tau_code,
            ),
            (
                "excitatory synaptic input gain bias code",
                self.excitatory_input_i_bias_gm_code,
            ),
            ("synaptic input drop bias code", self.synaptic_input_drop_bias_code),
        ):
            if value is not None and not 0 <= value <= 1022:
                raise ValueError(f"{name} must lie in [0, 1022]")
        if self.reset_code_minimum > self.reset_code_maximum:
            raise ValueError("reset code range must be increasing")
        for name, value in (
            ("precharge maximum weight", self.precharge_weight_maximum),
            ("exponential input weight", self.exponential_input_weight),
        ):
            if not 0 <= value <= 63:
                raise ValueError(f"{name} must lie in [0, 63]")
        if (
            self.precharge_input_fan_in <= 0
            or self.exponential_input_fan_in <= 0
            or self.psi_ne_input_fan_in <= 0
            or self.psi_ed_trigger_fan_in <= 0
        ):
            raise ValueError("input fan-in must be positive")
        if not 0 <= self.deadline_miss_rate_limit <= 1:
            raise ValueError("deadline miss-rate limit must lie in [0, 1]")
        if not self.psi_ed_delta_min_s < self.psi_ed_delta_max_s:
            raise ValueError("psi-ed delta range must be increasing")
        if self.psi_ed_delta_points < 3:
            raise ValueError("psi-ed requires at least three delta points")
        maximum_separation = (
            self.psi_ed_separation_center_s - self.psi_ed_delta_min_s
        )
        minimum_separation = (
            self.psi_ed_separation_center_s - self.psi_ed_delta_max_s
        )
        if not 0 < minimum_separation <= maximum_separation:
            raise ValueError("psi-ed transformed separations must be positive")
        if not maximum_separation < self.psi_ed_post_time_s:
            raise ValueError("psi-ed pre-event must follow the trial start")
        if not (
            self.psi_ed_post_time_s
            < self.psi_ed_readout_time_s
            < self.deadline_s
            < self.psi_ed_trial_guard_s
        ):
            raise ValueError(
                "psi-ed post, readout, deadline, and trial guard are misordered"
            )
        for weight in (self.psi_ed_trigger_weight, self.psi_ed_plastic_weight):
            if not 0 <= weight <= 63:
                raise ValueError("psi-ed synaptic weights must lie in [0, 63]")
        if self.psi_ed_chunk_repeats <= 0:
            raise ValueError("psi-ed chunk repeats must be positive")
        if self.phi_nl_lower_bound_minimum_code >= 1.0:
            raise ValueError("phi-nl lower-bound search must start below code one")
        if self.phi_nl_lower_bound_grid_size < 2:
            raise ValueError("phi-nl lower-bound grid requires at least two points")
        if self.pynn_chunk_repeats <= 0:
            raise ValueError("PyNN chunk repeats must be positive")
        if self.pynn_process_repeats is not None and self.pynn_process_repeats <= 0:
            raise ValueError("PyNN process repeats must be positive when specified")
        if self.pynn_worker_timeout_s <= 0:
            raise ValueError("PyNN worker timeout must be positive")
        if self.pynn_worker_max_attempts <= 0:
            raise ValueError("PyNN worker maximum attempts must be positive")
        if self.reset_code_table is not None:
            if len(self.reset_code_table) != 32:
                raise ValueError("reset code table must contain 32 entries")
            first = self.reset_code_table[0]
            if isinstance(first, int):
                if not all(isinstance(value, int) for value in self.reset_code_table):
                    raise ValueError("reset code table cannot mix scalar and vector entries")
                if any(not 0 <= value <= 1022 for value in self.reset_code_table):
                    raise ValueError("reset code table entries must lie in [0, 1022]")
            else:
                if not all(
                    isinstance(row, (tuple, list))
                    and len(row) == self.device_count
                    for row in self.reset_code_table
                ):
                    raise ValueError(
                        "each reset code table row must match device_count"
                    )
                if any(
                    not isinstance(value, int) or not 0 <= value <= 1022
                    for row in self.reset_code_table
                    for value in row
                ):
                    raise ValueError("reset code table entries must lie in [0, 1022]")
        if self.psi_ne_chunk_repeats <= 0:
            raise ValueError("psi-ne chunk repeats must be positive")
        if self.hagen_num_sends is not None and self.hagen_num_sends <= 0:
            raise ValueError("Hagen num_sends must be positive when specified")
        if not self.device_count <= self.hagen_candidate_count <= 512:
            raise ValueError(
                "Hagen candidate count must lie between device_count and 512"
            )
        if self.hagen_chunk_repeats <= 0:
            raise ValueError("Hagen chunk repeats must be positive")
        for rate in (
            self.normalized_rmse_limit,
            self.parameter_drift_limit,
            self.multiple_spike_rate_limit,
            self.saturation_rate_limit,
        ):
            if not 0 <= rate <= 1:
                raise ValueError("validation limits must lie in [0, 1]")
        for rate in (self.monotonicity_minimum,):
            if not 0 <= rate <= 1:
                raise ValueError("validation minima must lie in [0, 1]")
        for path in (
            self.spiking_calibration_path,
            self.hagen_calibration_path,
            self.correlation_calibration_path,
        ):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)

    @property
    def validation_repeats(self) -> int:
        return self.repeats - self.calibration_repeats

    def primitive_timing_dict(self) -> dict[str, Any]:
        """Return the declared time budget for one primitive invocation."""
        return {
            "definition": (
                "worst_case_primitive_latency_s is the observation deadline "
                "measured from the primitive start"
            ),
            "encoding_window_start_s": self.input_early_s,
            "encoding_window_end_s": self.input_late_s,
            "encoding_window_duration_s": (
                self.input_late_s - self.input_early_s
            ),
            "observation_time_s": self.observation_time_s,
            "deadline_s": self.deadline_s,
            "worst_case_primitive_latency_s": self.deadline_s,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hagen_output_coordinates"] = None
        payload["hagen_candidate_output_indices"] = list(
            range(self.hagen_candidate_count)
        )
        payload["spiking_calibration_path"] = (
            str(self.spiking_calibration_path)
            if self.spiking_calibration_path is not None
            else None
        )
        payload["hagen_calibration_path"] = (
            str(self.hagen_calibration_path)
            if self.hagen_calibration_path is not None
            else None
        )
        payload["correlation_calibration_path"] = (
            str(self.correlation_calibration_path)
            if self.correlation_calibration_path is not None
            else None
        )
        payload["spiking_calibration_sha256"] = calibration_sha256(
            self.spiking_calibration_path
        )
        payload["hagen_calibration_sha256"] = calibration_sha256(
            self.hagen_calibration_path
        )
        payload["correlation_calibration_sha256"] = calibration_sha256(
            self.correlation_calibration_path
        )
        return payload


@dataclass(frozen=True)
class PrimitiveObservation:
    """Raw repeated observations for one primitive and acquisition stage."""

    primitive: PrimitiveKind
    stage: PrimitiveStage
    output_kind: OutputKind
    input_code: torch.Tensor
    ideal_variable: torch.Tensor
    observed: torch.Tensor
    delivered: torch.Tensor
    spike_count: torch.Tensor
    saturated: torch.Tensor
    physical_coordinates: tuple[int, ...]
    fit_point_mask: torch.Tensor | None = None
    auxiliary_input: torch.Tensor | None = None
    baseline: torch.Tensor | None = None
    precharge_cadc: torch.Tensor | None = None
    first_spike_time_s: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {self.primitive}")
        if self.observed.ndim != 3:
            raise ValueError("observed must have shape [trial, point, device]")
        if self.delivered.shape != self.observed.shape:
            raise ValueError("delivered shape does not match observed")
        if self.spike_count.shape != self.observed.shape:
            raise ValueError("spike_count shape does not match observed")
        if self.saturated.shape != self.observed.shape:
            raise ValueError("saturated shape does not match observed")
        if self.input_code.ndim != 1 or self.ideal_variable.ndim != 1:
            raise ValueError("input and ideal variables must be one-dimensional")
        if self.input_code.numel() != self.observed.shape[1]:
            raise ValueError("input point count does not match observations")
        if self.ideal_variable.numel() != self.observed.shape[1]:
            raise ValueError("ideal point count does not match observations")
        if len(self.physical_coordinates) != self.observed.shape[2]:
            raise ValueError("coordinate count does not match device dimension")
        if self.fit_point_mask is not None and (
            self.fit_point_mask.ndim != 1
            or self.fit_point_mask.numel() != self.observed.shape[1]
        ):
            raise ValueError("fit_point_mask must match the point dimension")
        if self.auxiliary_input is not None and (
            self.auxiliary_input.ndim != 1
            or self.auxiliary_input.numel() != self.observed.shape[1]
        ):
            raise ValueError("auxiliary input must match the point dimension")
        if self.baseline is not None and self.baseline.shape != self.observed.shape:
            raise ValueError("baseline shape does not match observed")
        if (
            self.precharge_cadc is not None
            and self.precharge_cadc.shape != self.observed.shape
        ):
            raise ValueError("precharge_cadc shape does not match observed")
        if (
            self.first_spike_time_s is not None
            and self.first_spike_time_s.shape != self.observed.shape
        ):
            raise ValueError("first_spike_time_s shape does not match observed")
        finite = torch.isfinite(self.observed)
        if bool((finite & ~self.delivered).any()):
            raise ValueError("missed observations must remain non-finite")
        if bool((self.delivered & ~finite).any()):
            raise ValueError("delivered observations must be finite")

    @property
    def repeats(self) -> int:
        return int(self.observed.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.observed.shape[1])

    @property
    def device_count(self) -> int:
        return int(self.observed.shape[2])


@dataclass(frozen=True)
class PrimitiveValidation:
    """Calibration-only fit and held-out validation result for one observation."""

    primitive: PrimitiveKind
    stage: PrimitiveStage
    validated: bool
    diagnostic_only: bool
    calibration_parameters: tuple[dict[str, float], ...]
    validation_refit_parameters: tuple[dict[str, float], ...]
    device_statistics: tuple[dict[str, Any], ...]
    transfer_rows: tuple[dict[str, Any], ...]
    moment_rows: tuple[dict[str, Any], ...]
    noise: dict[str, Any]
    gates: dict[str, Any]


def _fit_linear(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    design = torch.stack((torch.ones_like(x), x), dim=1).to(torch.float64)
    solution = torch.linalg.lstsq(design, y.to(torch.float64)).solution
    return float(solution[0]), float(solution[1])


def _relative_change(reference: float, candidate: float) -> float:
    scale = max(abs(reference), torch.finfo(torch.float64).eps)
    return abs(candidate - reference) / scale


def _parameterize(
    primitive: PrimitiveKind,
    intercept: float,
    slope: float,
    *,
    tau_s: float | None = None,
    lower_bound_code: float | None = None,
) -> dict[str, float]:
    if primitive == "phi-np":
        return {"offset_s": intercept, "slope_s_per_code": slope}
    if primitive == "phi-nl":
        if lower_bound_code is None:
            raise ValueError("phi-nl fit requires a lower-bound code")
        return {
            "offset_s": intercept,
            "log_slope_s": slope,
            "tau_effective_s": -slope,
            "lower_bound_code": lower_bound_code,
        }
    if primitive == "psi-int":
        return {"offset_code": intercept, "gain": slope}
    if primitive == "psi-ne":
        if tau_s is None:
            raise ValueError("psi-ne fit requires tau")
        return {
            "baseline_cadc": intercept,
            "response_scale_cadc": slope,
            "tau_effective_s": tau_s,
        }
    if primitive == "psi-ed":
        if tau_s is None:
            raise ValueError("psi-ed fit requires tau")
        return {
            "baseline_code": intercept,
            "response_scale_code": slope,
            "tau_effective_s": tau_s,
        }
    raise ValueError(f"unsupported primitive: {primitive}")


def _fit_ne(
    input_time_s: torch.Tensor,
    observation: torch.Tensor,
    observation_time_s: float,
) -> tuple[dict[str, float], torch.Tensor]:
    tau_grid = torch.logspace(
        math.log10(0.25e-6), math.log10(250.0e-6), 640, dtype=torch.float64
    )
    best_error = float("inf")
    best_parameters: dict[str, float] | None = None
    best_prediction: torch.Tensor | None = None
    time = input_time_s.to(torch.float64)
    target = observation.to(torch.float64)
    for tau in tau_grid:
        regressor = torch.exp(-(observation_time_s - time) / tau)
        intercept, slope = _fit_linear(regressor, target)
        prediction = intercept + slope * regressor
        error = float(torch.mean((target - prediction).square()))
        if error < best_error:
            best_error = error
            best_parameters = _parameterize(
                "psi-ne", intercept, slope, tau_s=float(tau)
            )
            best_prediction = prediction
    assert best_parameters is not None and best_prediction is not None
    return best_parameters, best_prediction


def _fit_ed(
    time_difference_s: torch.Tensor,
    observation: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    """Fit an exponential-difference response on calibration samples."""
    tau_grid = torch.logspace(
        math.log10(0.25e-6), math.log10(250.0e-6), 640, dtype=torch.float64
    )
    best_error = float("inf")
    best_parameters: dict[str, float] | None = None
    best_prediction: torch.Tensor | None = None
    difference = time_difference_s.to(torch.float64)
    target = observation.to(torch.float64)
    for tau in tau_grid:
        regressor = torch.exp(difference / tau)
        intercept, slope = _fit_linear(regressor, target)
        prediction = intercept + slope * regressor
        error = float(torch.mean((target - prediction).square()))
        if error < best_error:
            best_error = error
            best_parameters = _parameterize(
                "psi-ed", intercept, slope, tau_s=float(tau)
            )
            best_prediction = prediction
    assert best_parameters is not None and best_prediction is not None
    return best_parameters, best_prediction


def _nl_device_samples(
    observation: PrimitiveObservation,
    values: torch.Tensor,
    valid: torch.Tensor,
    device: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect one circuit's usable fitted codes without imputing misses."""
    fit_points = (
        observation.fit_point_mask.to(torch.bool)
        if observation.fit_point_mask is not None
        else torch.ones(observation.point_count, dtype=torch.bool)
    )
    x_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    for point in torch.arange(observation.point_count)[fit_points]:
        mask = valid[:, point, device]
        if bool(mask.any()):
            count = int(mask.sum())
            x_chunks.append(observation.input_code[point].repeat(count))
            y_chunks.append(values[:, point, device][mask])
    if not x_chunks or sum(item.numel() for item in x_chunks) < 3:
        raise _InsufficientFitError("not enough valid samples to fit transfer")
    return (
        torch.cat(x_chunks).to(torch.float64),
        torch.cat(y_chunks).to(torch.float64),
    )


def _fit_nl_at_lower_bound(
    code: torch.Tensor,
    target: torch.Tensor,
    lower_bound_code: float,
) -> dict[str, float]:
    regressor = torch.log(code.to(torch.float64) - lower_bound_code)
    intercept, slope = _fit_linear(regressor, target.to(torch.float64))
    return _parameterize(
        "phi-nl",
        intercept,
        slope,
        lower_bound_code=lower_bound_code,
    )


def _fit_nl_shared(
    observation: PrimitiveObservation,
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    lower_bound_minimum_code: float,
    lower_bound_grid_size: int,
    lower_bound_code: float | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Fit one shared physical lower bound and per-circuit gain and offset."""
    samples = [
        _nl_device_samples(observation, values, valid, device)
        for device in range(observation.device_count)
    ]
    if lower_bound_code is None:
        minimum_input = min(float(code.min()) for code, _ in samples)
        candidates = torch.linspace(
            lower_bound_minimum_code,
            minimum_input - 1.0e-3,
            lower_bound_grid_size,
            dtype=torch.float64,
        )
        best_score = float("inf")
        best_lower_bound: float | None = None
        for candidate in candidates:
            candidate_value = float(candidate)
            score = 0.0
            for code, target in samples:
                parameters = _fit_nl_at_lower_bound(
                    code, target, candidate_value
                )
                prediction = parameters["offset_s"] + parameters[
                    "log_slope_s"
                ] * torch.log(code - candidate_value)
                span = max(
                    float(target.max() - target.min()),
                    torch.finfo(torch.float64).eps,
                )
                score += float(torch.mean((target - prediction).square())) / (
                    span * span
                )
            if score < best_score:
                best_score = score
                best_lower_bound = candidate_value
        assert best_lower_bound is not None
        lower_bound_code = best_lower_bound
    parameters = [
        _fit_nl_at_lower_bound(code, target, lower_bound_code)
        for code, target in samples
    ]
    return lower_bound_code, parameters


def _predict(
    primitive: PrimitiveKind,
    parameters: dict[str, float],
    ideal_variable: torch.Tensor,
    observation_time_s: float,
) -> torch.Tensor:
    x = ideal_variable.to(torch.float64)
    if primitive == "phi-np":
        return parameters["offset_s"] + parameters["slope_s_per_code"] * x
    if primitive == "phi-nl":
        code = torch.exp(x)
        return parameters["offset_s"] + parameters["log_slope_s"] * torch.log(
            code - parameters["lower_bound_code"]
        )
    if primitive == "psi-int":
        return parameters["offset_code"] + parameters["gain"] * x
    if primitive == "psi-ne":
        tau = parameters["tau_effective_s"]
        return parameters["baseline_cadc"] + parameters[
            "response_scale_cadc"
        ] * torch.exp(-(observation_time_s - x) / tau)
    if primitive == "psi-ed":
        tau = parameters["tau_effective_s"]
        return parameters["baseline_code"] + parameters[
            "response_scale_code"
        ] * torch.exp(x / tau)
    raise ValueError(f"unsupported primitive: {primitive}")


class _InsufficientFitError(ValueError):
    """Raised when an observation split cannot support its transfer fit."""


def _fit_device(
    observation: PrimitiveObservation,
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    observation_time_s: float,
) -> dict[str, float]:
    point = torch.arange(observation.point_count)
    fit_points = (
        observation.fit_point_mask.to(torch.bool)
        if observation.fit_point_mask is not None
        else torch.ones(observation.point_count, dtype=torch.bool)
    )
    x_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    for point_index in point[fit_points]:
        mask = valid[:, point_index]
        if bool(mask.any()):
            count = int(mask.sum())
            x_chunks.append(observation.ideal_variable[point_index].repeat(count))
            y_chunks.append(values[mask, point_index])
    if not x_chunks or sum(item.numel() for item in x_chunks) < 3:
        raise _InsufficientFitError("not enough valid samples to fit transfer")
    x = torch.cat(x_chunks).to(torch.float64)
    y = torch.cat(y_chunks).to(torch.float64)
    if observation.primitive == "psi-ne":
        parameters, _ = _fit_ne(x, y, observation_time_s)
        return parameters
    if observation.primitive == "psi-ed":
        parameters, _ = _fit_ed(x, y)
        return parameters
    if observation.primitive == "phi-nl":
        raise RuntimeError("phi-nl must use the shared lower-bound fitter")
    intercept, slope = _fit_linear(x, y)
    return _parameterize(observation.primitive, intercept, slope)


def _unavailable_parameters(primitive: PrimitiveKind) -> dict[str, float]:
    value = float("nan")
    return _parameterize(
        primitive,
        value,
        value,
        tau_s=value if primitive in ("psi-ne", "psi-ed") else None,
        lower_bound_code=value if primitive == "phi-nl" else None,
    )


def _fit_drift_key(primitive: PrimitiveKind) -> str:
    if primitive == "phi-np":
        return "slope_s_per_code"
    if primitive == "phi-nl":
        return "tau_effective_s"
    if primitive == "psi-int":
        return "gain"
    if primitive == "psi-ed":
        return "tau_effective_s"
    return "tau_effective_s"


def _monotonicity(values: torch.Tensor, expected_slope: float) -> float:
    if values.numel() < 2:
        return float("nan")
    differences = values[1:] - values[:-1]
    if expected_slope >= 0:
        return float((differences >= 0).to(torch.float64).mean())
    return float((differences <= 0).to(torch.float64).mean())


def _monotonicity_xy(
    x_values: list[float],
    y_values: list[float],
    expected_slope: float,
) -> float:
    """Measure monotonicity after sorting and averaging duplicate ideal inputs."""
    grouped: dict[float, list[float]] = {}
    for x_value, y_value in zip(x_values, y_values):
        grouped.setdefault(x_value, []).append(y_value)
    ordered = sorted(grouped)
    means = torch.tensor(
        [sum(grouped[value]) / len(grouped[value]) for value in ordered],
        dtype=torch.float64,
    )
    return _monotonicity(means, expected_slope)


def _rank_monotonicity(values: torch.Tensor, expected_slope: float) -> float:
    """Return Spearman rank agreement with the declared transfer direction."""
    if values.numel() < 2:
        return float("nan")
    expected = torch.arange(values.numel(), dtype=torch.float64)
    directed = values if expected_slope >= 0 else -values
    order = torch.argsort(directed, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64)
    expected = expected - expected.mean()
    ranks = ranks - ranks.mean()
    denominator = expected.square().sum().sqrt() * ranks.square().sum().sqrt()
    if float(denominator) == 0.0:
        return 0.0
    return float((expected * ranks).sum() / denominator)


def _rank_monotonicity_xy(
    x_values: list[float],
    y_values: list[float],
    expected_slope: float,
) -> float:
    """Measure global rank order after averaging duplicate ideal inputs."""
    grouped: dict[float, list[float]] = {}
    for x_value, y_value in zip(x_values, y_values):
        grouped.setdefault(x_value, []).append(y_value)
    ordered = sorted(grouped)
    means = torch.tensor(
        [sum(grouped[value]) / len(grouped[value]) for value in ordered],
        dtype=torch.float64,
    )
    return _rank_monotonicity(means, expected_slope)


def validate_primitive_observation(
    observation: PrimitiveObservation,
    config: PrimitiveNoiseConfig,
) -> PrimitiveValidation:
    """Fit only the calibration split and score its frozen transfer on validation."""
    if observation.repeats != config.repeats:
        raise ValueError("observation repeat count does not match configuration")
    if observation.device_count != config.device_count:
        raise ValueError("observation device count does not match configuration")
    if observation.primitive == "psi-int":
        expected_coordinates = tuple(
            observation.metadata.get(
                "selected_output_indices", range(config.device_count)
            )
        )
        if (
            len(expected_coordinates) != config.device_count
            or len(set(expected_coordinates)) != config.device_count
            or any(
                index < 0 or index >= config.hagen_candidate_count
                for index in expected_coordinates
            )
        ):
            raise ValueError("observation Hagen output selection is invalid")
    else:
        expected_coordinates = config.physical_coordinates
    if observation.physical_coordinates != expected_coordinates:
        raise ValueError("observation coordinates do not match configuration")

    calibration = slice(0, config.calibration_repeats)
    validation = slice(config.calibration_repeats, config.repeats)
    calibration_parameters: list[dict[str, float]] = []
    validation_parameters: list[dict[str, float]] = []
    device_statistics: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    temporal_sigmas: list[float] = []
    drift_values: list[float] = []
    nrmse_values: list[float] = []
    rank_monotonicity_values: list[float] = []
    adjacent_order_fractions: list[float] = []
    fit_availability: list[bool] = []
    fit_points = (
        observation.fit_point_mask.to(torch.bool)
        if observation.fit_point_mask is not None
        else torch.ones(observation.point_count, dtype=torch.bool)
    )
    usable_observation = (
        observation.delivered
        & ~observation.saturated
        & torch.isfinite(observation.observed)
    )
    nl_calibration_parameters: list[dict[str, float]] | None = None
    nl_validation_parameters: list[dict[str, float]] | None = None
    nl_shared_lower_bound_code: float | None = None
    if observation.primitive == "phi-nl":
        try:
            (
                nl_shared_lower_bound_code,
                nl_calibration_parameters,
            ) = _fit_nl_shared(
                observation,
                observation.observed[calibration].to(torch.float64),
                usable_observation[calibration],
                lower_bound_minimum_code=(
                    config.phi_nl_lower_bound_minimum_code
                ),
                lower_bound_grid_size=config.phi_nl_lower_bound_grid_size,
            )
            _, nl_validation_parameters = _fit_nl_shared(
                observation,
                observation.observed[validation].to(torch.float64),
                usable_observation[validation],
                lower_bound_minimum_code=(
                    config.phi_nl_lower_bound_minimum_code
                ),
                lower_bound_grid_size=config.phi_nl_lower_bound_grid_size,
                lower_bound_code=nl_shared_lower_bound_code,
            )
        except _InsufficientFitError:
            nl_calibration_parameters = None
            nl_validation_parameters = None
    for device in range(observation.device_count):
        values = observation.observed[:, :, device].to(torch.float64)
        usable = usable_observation[:, :, device]
        try:
            if observation.primitive == "phi-nl":
                if (
                    nl_calibration_parameters is None
                    or nl_validation_parameters is None
                ):
                    raise _InsufficientFitError(
                        "shared phi-nl transfer fit is unavailable"
                    )
                calibration_fit = nl_calibration_parameters[device]
                validation_fit = nl_validation_parameters[device]
            else:
                calibration_fit = _fit_device(
                    observation,
                    values[calibration],
                    usable[calibration],
                    observation_time_s=config.observation_time_s,
                )
                validation_fit = _fit_device(
                    observation,
                    values[validation],
                    usable[validation],
                    observation_time_s=config.observation_time_s,
                )
            fit_available = True
        except _InsufficientFitError:
            calibration_fit = _unavailable_parameters(observation.primitive)
            validation_fit = _unavailable_parameters(observation.primitive)
            fit_available = False
        calibration_parameters.append(calibration_fit)
        validation_parameters.append(validation_fit)
        prediction = _predict(
            observation.primitive,
            calibration_fit,
            observation.ideal_variable,
            config.observation_time_s,
        )
        calibration_residuals: list[torch.Tensor] = []
        validation_residuals: list[torch.Tensor] = []
        validation_mean_residuals: list[torch.Tensor] = []
        validation_means: list[float] = []
        validation_x: list[float] = []
        for point in range(observation.point_count):
            calibration_mask = usable[calibration, point]
            validation_mask = usable[validation, point]
            calibration_selected = values[calibration, point][calibration_mask]
            validation_selected = values[validation, point][validation_mask]
            if bool(calibration_mask.any()) and bool(fit_points[point]):
                calibration_residuals.append(
                    calibration_selected - prediction[point]
                )
            if bool(validation_mask.any()) and bool(fit_points[point]):
                residual = validation_selected - prediction[point]
                validation_residuals.append(residual)
                validation_mean = validation_selected.mean()
                validation_mean_residuals.append(
                    validation_mean - prediction[point]
                )
                validation_means.append(float(validation_mean))
                validation_x.append(float(observation.ideal_variable[point]))
            for split_name, trial_slice, selected in (
                ("calibration", calibration, calibration_selected),
                ("validation", validation, validation_selected),
            ):
                sample_count = int(selected.numel())
                moment_rows.append(
                    {
                        "primitive": observation.primitive,
                        "stage": observation.stage,
                        "split": split_name,
                        "device": device,
                        "physical_coordinate": (
                            observation.physical_coordinates[device]
                        ),
                        "input_code": float(observation.input_code[point]),
                        "ideal_variable": float(
                            observation.ideal_variable[point]
                        ),
                        "sample_count": sample_count,
                        "mean": (
                            float(selected.mean()) if sample_count else None
                        ),
                        "variance": (
                            float(selected.var(unbiased=True))
                            if sample_count > 1
                            else (0.0 if sample_count == 1 else None)
                        ),
                        "miss_rate": float(
                            (~observation.delivered[trial_slice, point, device])
                            .to(torch.float64)
                            .mean()
                        ),
                        "multiple_spike_rate": float(
                            (
                                observation.spike_count[
                                    trial_slice, point, device
                                ]
                                > 1
                            )
                            .to(torch.float64)
                            .mean()
                        ),
                        "saturation_rate": float(
                            observation.saturated[
                                trial_slice, point, device
                            ]
                            .to(torch.float64)
                            .mean()
                        ),
                    }
                )
            transfer_rows.append(
                {
                    "primitive": observation.primitive,
                    "stage": observation.stage,
                    "device": device,
                    "physical_coordinate": observation.physical_coordinates[device],
                    "input_code": float(observation.input_code[point]),
                    "ideal_variable": float(observation.ideal_variable[point]),
                    "fit_point": bool(fit_points[point]),
                    "endpoint_diagnostic": bool(
                        observation.primitive == "phi-np"
                        and float(observation.input_code[point]) == 31.0
                    ),
                    "calibration_prediction": float(prediction[point]),
                    "calibration_mean": (
                        float(calibration_selected.mean())
                        if bool(calibration_mask.any())
                        else None
                    ),
                    "calibration_variance": (
                        float(calibration_selected.var(unbiased=True))
                        if calibration_selected.numel() > 1
                        else (
                            0.0 if calibration_selected.numel() == 1 else None
                        )
                    ),
                    "validation_mean": (
                        float(validation_selected.mean())
                        if bool(validation_mask.any())
                        else None
                    ),
                    "validation_variance": (
                        float(validation_selected.var(unbiased=True))
                        if validation_selected.numel() > 1
                        else (0.0 if validation_selected.numel() == 1 else None)
                    ),
                    "calibration_delivery_rate": float(
                        observation.delivered[calibration, point, device]
                        .to(torch.float64)
                        .mean()
                    ),
                    "validation_delivery_rate": float(
                        observation.delivered[validation, point, device]
                        .to(torch.float64)
                        .mean()
                    ),
                    "validation_saturation_rate": float(
                        observation.saturated[validation, point, device]
                        .to(torch.float64)
                        .mean()
                    ),
                }
            )
        drift_key = _fit_drift_key(observation.primitive)
        if (
            fit_available
            and calibration_residuals
            and validation_residuals
            and validation_mean_residuals
        ):
            calibration_residual = torch.cat(calibration_residuals)
            validation_residual = torch.cat(validation_residuals)
            validation_mean_residual = torch.stack(validation_mean_residuals)
            temporal_sigma = float(calibration_residual.std(unbiased=True))
            span = max(
                float(prediction[fit_points].max() - prediction[fit_points].min()),
                torch.finfo(torch.float64).eps,
            )
            validation_nrmse = (
                float(validation_mean_residual.square().mean().sqrt()) / span
            )
            parameter_drift = _relative_change(
                calibration_fit[drift_key], validation_fit[drift_key]
            )
            adjacent_order_fraction = _monotonicity_xy(
                validation_x,
                validation_means,
                calibration_fit[drift_key]
                if observation.primitive != "phi-nl"
                else calibration_fit["log_slope_s"],
            )
            rank_monotonicity = _rank_monotonicity_xy(
                validation_x,
                validation_means,
                calibration_fit[drift_key]
                if observation.primitive != "phi-nl"
                else calibration_fit["log_slope_s"],
            )
        else:
            fit_available = False
            calibration_residual = torch.empty(0, dtype=torch.float64)
            validation_residual = torch.empty(0, dtype=torch.float64)
            temporal_sigma = float("nan")
            validation_nrmse = float("inf")
            parameter_drift = float("inf")
            adjacent_order_fraction = float("nan")
            rank_monotonicity = float("nan")
        fit_availability.append(fit_available)
        temporal_sigmas.append(temporal_sigma)
        drift_values.append(parameter_drift)
        nrmse_values.append(validation_nrmse)
        adjacent_order_fractions.append(adjacent_order_fraction)
        rank_monotonicity_values.append(rank_monotonicity)
        device_statistics.append(
            {
                "primitive": observation.primitive,
                "stage": observation.stage,
                "device": device,
                "physical_coordinate": observation.physical_coordinates[device],
                "fit_available": fit_available,
                "temporal_sigma": temporal_sigma,
                "validation_normalized_rmse": validation_nrmse,
                "parameter_drift": parameter_drift,
                "rank_monotonicity": rank_monotonicity,
                "adjacent_order_fraction": adjacent_order_fraction,
                "calibration_residual_count": int(calibration_residual.numel()),
                "validation_residual_count": int(validation_residual.numel()),
                **{f"calibration_{key}": value for key, value in calibration_fit.items()},
                **{f"validation_refit_{key}": value for key, value in validation_fit.items()},
            }
        )

    exactly_one = observation.delivered & (observation.spike_count == 1)
    if (
        observation.output_kind == "spike-time-s"
        or observation.first_spike_time_s is not None
    ):
        overall_exactly_one = float(exactly_one.to(torch.float64).mean())
        minimum_point_exactly_one = float(
            exactly_one.to(torch.float64).mean(dim=(0, 2)).min()
        )
        overall_delivery = float(
            observation.delivered.to(torch.float64).mean()
        )
        point_delivery = observation.delivered.to(torch.float64).mean(dim=(0, 2))
    else:
        overall_exactly_one = None
        minimum_point_exactly_one = None
        overall_delivery = float(observation.delivered.to(torch.float64).mean())
        point_delivery = observation.delivered.to(torch.float64).mean(dim=(0, 2))
    minimum_point_delivery = float(point_delivery.min())
    calibration_point_observed = usable_observation[calibration].any(dim=0)
    validation_point_observed = usable_observation[validation].any(dim=0)
    fitted_point_and_device = fit_points.reshape(-1, 1).expand(
        observation.point_count, observation.device_count
    )
    every_fit_point_observed = bool(
        calibration_point_observed[fitted_point_and_device].all()
        and validation_point_observed[fitted_point_and_device].all()
    )
    saturation_rate = float(observation.saturated.to(torch.float64).mean())
    multiple_spike_rate = float(
        (observation.spike_count > 1).to(torch.float64).mean()
    )
    maximum_nrmse = max(nrmse_values)
    maximum_drift = max(drift_values)
    minimum_rank_monotonicity = min(rank_monotonicity_values)
    minimum_adjacent_order_fraction = min(adjacent_order_fractions)
    gates = {
        "fit_available": all(fit_availability),
        "normalized_rmse": maximum_nrmse <= config.normalized_rmse_limit,
        "parameter_drift": maximum_drift <= config.parameter_drift_limit,
        "multiple_spike_rate": (
            multiple_spike_rate <= config.multiple_spike_rate_limit
        ),
        "saturation": saturation_rate <= config.saturation_rate_limit,
        "point_observed": every_fit_point_observed,
    }
    deadline_applies = bool(
        observation.output_kind == "spike-time-s"
        or observation.metadata.get("deadline_applies", False)
    )
    if deadline_applies:
        gates["deadline_miss_rate"] = (
            1.0 - overall_delivery <= config.deadline_miss_rate_limit
        )
    if observation.primitive == "phi-np":
        gates["monotonicity"] = (
            minimum_rank_monotonicity >= config.monotonicity_minimum
        )
    if observation.primitive == "psi-ne":
        gates["premature_spike"] = not bool((observation.spike_count > 0).any())
    precharge_rank_monotonicity_minimum: float | None = None
    if observation.precharge_cadc is not None:
        precharge = observation.precharge_cadc.to(torch.float64)
        precharge_means = torch.nanmean(precharge, dim=0)
        gates["precharge_recorded"] = bool(
            torch.isfinite(precharge).any(dim=0).all()
        )
        precharge_rank_monotonicity_minimum = min(
            _rank_monotonicity(precharge_means[:, device], 1.0)
            for device in range(observation.device_count)
        )
    validated = all(bool(value) for value in gates.values())

    parameter_keys = tuple(calibration_parameters[0])
    fixed_pattern: dict[str, Any] = {}
    for key in parameter_keys:
        values = torch.tensor(
            [parameters[key] for parameters in calibration_parameters],
            dtype=torch.float64,
        )
        values = values[torch.isfinite(values)]
        if not values.numel():
            values = torch.tensor([float("nan")], dtype=torch.float64)
        fixed_pattern[key] = {
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std(unbiased=values.numel() > 1)),
        }
    sigma_tensor = torch.tensor(temporal_sigmas, dtype=torch.float64)
    sigma_tensor = sigma_tensor[torch.isfinite(sigma_tensor)]
    if not sigma_tensor.numel():
        sigma_tensor = torch.tensor([float("nan")], dtype=torch.float64)
    noise = {
        "temporal_sigma": {
            "representative_median": float(sigma_tensor.median()),
            "minimum": float(sigma_tensor.min()),
            "maximum": float(sigma_tensor.max()),
            "unit": observation.output_kind,
        },
        "fixed_pattern_parameters": fixed_pattern,
        "overall_delivery_rate": overall_delivery,
        "minimum_point_delivery_rate": minimum_point_delivery,
        "miss_rate": 1.0 - overall_delivery,
        "multiple_spike_rate": multiple_spike_rate,
        "multiple_spike_rate_limit": config.multiple_spike_rate_limit,
        "multiple_spike_handling": "first-spike-time",
        "maximum_spike_count": int(observation.spike_count.max()),
        "saturation_rate": saturation_rate,
        "validation_normalized_rmse_maximum": maximum_nrmse,
        "parameter_drift_maximum": maximum_drift,
        "rank_monotonicity_minimum": minimum_rank_monotonicity,
        "adjacent_order_fraction_minimum": minimum_adjacent_order_fraction,
        "interpretation": "independent-marginal",
    }
    if deadline_applies:
        noise["deadline_miss_rate"] = 1.0 - overall_delivery
        noise["deadline_miss_rate_limit"] = config.deadline_miss_rate_limit
    if overall_exactly_one is not None:
        noise["exactly_one_spike_rate"] = overall_exactly_one
        noise["minimum_point_exactly_one_spike_rate"] = (
            minimum_point_exactly_one
        )
    if precharge_rank_monotonicity_minimum is not None:
        noise["precharge_rank_monotonicity_minimum"] = (
            precharge_rank_monotonicity_minimum
        )
    if observation.primitive == "phi-np":
        noise["monotonicity_assessment"] = {
            "acceptance_metric": "rank_monotonicity",
            "diagnostic_metric": "adjacent_order_fraction",
            "rationale": (
                "Temporal jitter can reverse neighboring measured means without "
                "breaking the global transfer direction."
            ),
        }
    if nl_shared_lower_bound_code is not None:
        noise["shared_lower_bound_code"] = nl_shared_lower_bound_code
    return PrimitiveValidation(
        primitive=observation.primitive,
        stage=observation.stage,
        validated=validated,
        diagnostic_only=observation.primitive == "phi-np" and observation.stage == "static",
        calibration_parameters=tuple(calibration_parameters),
        validation_refit_parameters=tuple(validation_parameters),
        device_statistics=tuple(device_statistics),
        transfer_rows=tuple(transfer_rows),
        moment_rows=tuple(moment_rows),
        noise=noise,
        gates=gates,
    )


class MockPrimitiveNoiseBackend:
    """Seeded synthetic collector used to verify fitting and artifact contracts."""

    def collect(
        self,
        primitive: PrimitiveKind,
        config: PrimitiveNoiseConfig,
        *,
        stage: PrimitiveStage = "transfer",
        quick: bool = False,
    ) -> PrimitiveObservation:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + PRIMITIVES.index(primitive) * 1009)
        devices = config.device_count
        trials = config.repeats
        device_offset = torch.randn(devices, generator=generator, dtype=torch.float64)

        if primitive == "phi-np":
            codes = (
                torch.tensor([0.0, 15.0, 30.0], dtype=torch.float64)
                if quick
                else torch.arange(32, dtype=torch.float64)
            )
            ideal = codes.clone()
            mean = 25.0e-6 - (20.0e-6 / 31.0) * codes
            if stage == "dynamic":
                mean = mean + 0.15e-6
            sigma = 0.18e-6
            output_kind: OutputKind = "spike-time-s"
            fit_mask = codes != 31
        elif primitive == "phi-nl":
            codes = (
                torch.tensor([1.0, 8.0, 31.0], dtype=torch.float64)
                if quick
                else torch.arange(1, 32, dtype=torch.float64)
            )
            ideal = torch.log(codes)
            mean = 25.0e-6 - 4.0e-6 * torch.log(codes + 8.0)
            sigma = 0.20e-6
            output_kind = "spike-time-s"
            fit_mask = torch.ones_like(codes, dtype=torch.bool)
        elif primitive == "psi-int":
            durations = (
                torch.tensor([0.0, 15.0, 31.0], dtype=torch.float64)
                if quick
                else torch.arange(32, dtype=torch.float64)
            )
            drives = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
            codes = durations.repeat_interleave(drives.numel())
            auxiliary = drives.repeat(durations.numel())
            ideal = codes * auxiliary
            mean = 0.25 + 0.98 * ideal
            sigma = 0.45
            output_kind = "pwm-code"
            fit_mask = torch.ones_like(codes, dtype=torch.bool)
        elif primitive == "psi-ne":
            input_times = (
                torch.tensor([5.0e-6, 15.0e-6, 25.0e-6], dtype=torch.float64)
                if quick
                else torch.linspace(
                    config.input_early_s,
                    config.input_late_s,
                    21,
                    dtype=torch.float64,
                )
            )
            codes = input_times / config.dt_s
            ideal = input_times
            mean = 5.0 + 30.0 * torch.exp(
                -(config.observation_time_s - input_times) / 10.0e-6
            )
            sigma = 0.02 if quick else 0.20
            output_kind = "cadc-potential"
            fit_mask = torch.ones_like(codes, dtype=torch.bool)
        elif primitive == "psi-ed":
            differences = (
                torch.tensor([-10.0e-6, 0.0, 10.0e-6], dtype=torch.float64)
                if quick
                else torch.linspace(
                    config.psi_ed_delta_min_s,
                    config.psi_ed_delta_max_s,
                    config.psi_ed_delta_points,
                    dtype=torch.float64,
                )
            )
            codes = differences / config.dt_s
            ideal = differences
            mean = 2.0 + 20.0 * torch.exp(differences / 10.0e-6)
            sigma = 0.35
            output_kind = "cadc-potential"
            fit_mask = torch.ones_like(codes, dtype=torch.bool)
        else:
            raise ValueError(f"unsupported primitive: {primitive}")

        point_count = int(codes.numel())
        fixed_scale = sigma * 1.5
        observed = mean[None, :, None] + fixed_scale * device_offset[None, None, :]
        observed = observed + sigma * torch.randn(
            trials, point_count, devices, generator=generator, dtype=torch.float64
        )
        delivered = torch.rand(
            trials, point_count, devices, generator=generator
        ) >= 0.001
        saturated = torch.zeros_like(delivered)
        if primitive == "psi-int":
            saturated = (observed <= -128) | (observed >= 127)
        elif primitive in ("psi-ne", "psi-ed"):
            saturated = (observed <= -128) | (observed >= 127)
        spike_count = torch.zeros_like(delivered, dtype=torch.int64)
        if output_kind == "spike-time-s" or primitive == "psi-ed":
            spike_count[delivered] = 1
        first_spike_time_s = None
        if primitive == "psi-ed":
            first_spike_time_s = config.psi_ed_post_time_s + 0.20e-6 * torch.randn(
                trials,
                point_count,
                devices,
                generator=generator,
                dtype=torch.float64,
            )
            first_spike_time_s[~delivered] = float("nan")
        observed = observed.clone()
        observed[~delivered] = float("nan")
        precharge_cadc = None
        if (
            primitive in ("phi-np", "phi-nl")
            and (stage == "dynamic" or primitive == "phi-nl")
            and config.record_precharge_cadc
        ):
            precharge_cadc = codes[None, :, None] * 2.0 + 0.05 * torch.randn(
                trials,
                point_count,
                devices,
                generator=generator,
                dtype=torch.float64,
            )
        auxiliary_input = auxiliary if primitive == "psi-int" else None
        metadata = {
            "backend": "mock",
            "seed": config.seed,
            "quick": quick,
            "replicas_are_pooled": False,
            "coordinate_semantics": (
                "hagen-output-index"
                if primitive == "psi-int"
                else "atomic-neuron-on-dls"
            ),
            "deadline_applies": primitive in ("phi-np", "phi-nl", "psi-ed"),
        }
        return PrimitiveObservation(
            primitive=primitive,
            stage=stage,
            output_kind=output_kind,
            input_code=codes,
            ideal_variable=ideal,
            observed=observed,
            delivered=delivered,
            spike_count=spike_count,
            saturated=saturated,
            physical_coordinates=(
                tuple(range(config.device_count))
                if primitive == "psi-int"
                else config.physical_coordinates
            ),
            fit_point_mask=fit_mask,
            auxiliary_input=auxiliary_input,
            precharge_cadc=precharge_cadc,
            first_spike_time_s=first_spike_time_s,
            metadata=metadata,
        )


def primitive_result_key(observation: PrimitiveObservation) -> str:
    return f"{observation.primitive}_{observation.stage}"
