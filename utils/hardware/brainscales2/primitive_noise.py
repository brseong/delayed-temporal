"""Independent marginal-noise characterization for four TTFS primitives.

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


PrimitiveKind = Literal["phi-np", "phi-nl", "psi-int", "psi-ne"]
PrimitiveStage = Literal["static", "dynamic", "transfer"]
OutputKind = Literal["spike-time-s", "pwm-code", "cadc-potential"]

PRIMITIVES: tuple[PrimitiveKind, ...] = (
    "phi-np",
    "phi-nl",
    "psi-int",
    "psi-ne",
)


def default_primitive_coordinates(device_count: int = 16) -> tuple[int, ...]:
    """Return quadrant-balanced atomic-neuron coordinates without pooling."""
    if device_count <= 0 or device_count > 512:
        raise ValueError("device_count must lie in [1, 512]")
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
    observation_time_s: float = 30.0e-6
    deadline_s: float = 60.0e-6
    spiking_calibration_path: Path | None = None
    hagen_calibration_path: Path | None = None
    allow_environment_calibration: bool = False
    physical_coordinates: tuple[int, ...] = field(
        default_factory=default_primitive_coordinates
    )
    normalized_rmse_limit: float = 0.05
    parameter_drift_limit: float = 0.10
    overall_delivery_minimum: float = 0.99
    point_delivery_minimum: float = 0.95
    saturation_rate_limit: float = 0.01
    monotonicity_minimum: float = 0.95
    # Raw hardware controls.  They are recorded verbatim and are not treated as
    # calibrated physical units.
    reset_code_minimum: int = 400
    reset_code_maximum: int = 560
    threshold_code: int = 600
    leak_bias: int = 0
    constant_current_code: int = 300
    precharge_weight_maximum: int = 63
    exponential_input_weight: int = 63
    tau_mem_s: float = 100.0e-6
    tau_syn_s: float = 1.0e-6
    hagen_wait_between_events: int = 5
    hagen_num_sends: int | None = None

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
        for rate in (
            self.normalized_rmse_limit,
            self.parameter_drift_limit,
            self.saturation_rate_limit,
        ):
            if not 0 <= rate <= 1:
                raise ValueError("validation limits must lie in [0, 1]")
        for rate in (
            self.overall_delivery_minimum,
            self.point_delivery_minimum,
            self.monotonicity_minimum,
        ):
            if not 0 <= rate <= 1:
                raise ValueError("validation minima must lie in [0, 1]")
        for path in (self.spiking_calibration_path, self.hagen_calibration_path):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)

    @property
    def validation_repeats(self) -> int:
        return self.repeats - self.calibration_repeats

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hagen_output_coordinates"] = list(range(self.device_count))
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
        payload["spiking_calibration_sha256"] = calibration_sha256(
            self.spiking_calibration_path
        )
        payload["hagen_calibration_sha256"] = calibration_sha256(
            self.hagen_calibration_path
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
) -> dict[str, float]:
    if primitive == "phi-np":
        return {"offset_s": intercept, "slope_s_per_code": slope}
    if primitive == "phi-nl":
        return {
            "offset_s": intercept,
            "log_slope_s": slope,
            "tau_effective_s": -slope,
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
        return parameters["offset_s"] + parameters["log_slope_s"] * x
    if primitive == "psi-int":
        return parameters["offset_code"] + parameters["gain"] * x
    if primitive == "psi-ne":
        tau = parameters["tau_effective_s"]
        return parameters["baseline_cadc"] + parameters[
            "response_scale_cadc"
        ] * torch.exp(-(observation_time_s - x) / tau)
    raise ValueError(f"unsupported primitive: {primitive}")


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
        raise ValueError("not enough valid samples to fit transfer")
    x = torch.cat(x_chunks).to(torch.float64)
    y = torch.cat(y_chunks).to(torch.float64)
    if observation.primitive == "psi-ne":
        parameters, _ = _fit_ne(x, y, observation_time_s)
        return parameters
    intercept, slope = _fit_linear(x, y)
    return _parameterize(observation.primitive, intercept, slope)


def _fit_drift_key(primitive: PrimitiveKind) -> str:
    if primitive == "phi-np":
        return "slope_s_per_code"
    if primitive == "phi-nl":
        return "tau_effective_s"
    if primitive == "psi-int":
        return "gain"
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


def validate_primitive_observation(
    observation: PrimitiveObservation,
    config: PrimitiveNoiseConfig,
) -> PrimitiveValidation:
    """Fit only the calibration split and score its frozen transfer on validation."""
    if observation.repeats != config.repeats:
        raise ValueError("observation repeat count does not match configuration")
    if observation.device_count != config.device_count:
        raise ValueError("observation device count does not match configuration")
    expected_coordinates = (
        tuple(range(config.device_count))
        if observation.primitive == "psi-int"
        else config.physical_coordinates
    )
    if observation.physical_coordinates != expected_coordinates:
        raise ValueError("observation coordinates do not match configuration")

    calibration = slice(0, config.calibration_repeats)
    validation = slice(config.calibration_repeats, config.repeats)
    calibration_parameters: list[dict[str, float]] = []
    validation_parameters: list[dict[str, float]] = []
    device_statistics: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    temporal_sigmas: list[float] = []
    drift_values: list[float] = []
    nrmse_values: list[float] = []
    monotonicity_values: list[float] = []

    for device in range(observation.device_count):
        values = observation.observed[:, :, device].to(torch.float64)
        fit_points = (
            observation.fit_point_mask.to(torch.bool)
            if observation.fit_point_mask is not None
            else torch.ones(observation.point_count, dtype=torch.bool)
        )
        usable = (
            observation.delivered[:, :, device]
            & ~observation.saturated[:, :, device]
            & torch.isfinite(values)
        )
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
        validation_means: list[float] = []
        validation_x: list[float] = []
        for point in range(observation.point_count):
            calibration_mask = usable[calibration, point]
            validation_mask = usable[validation, point]
            if bool(calibration_mask.any()) and bool(fit_points[point]):
                calibration_residuals.append(
                    values[calibration, point][calibration_mask] - prediction[point]
                )
            if bool(validation_mask.any()) and bool(fit_points[point]):
                residual = values[validation, point][validation_mask] - prediction[point]
                validation_residuals.append(residual)
                validation_means.append(
                    float(values[validation, point][validation_mask].mean())
                )
                validation_x.append(float(observation.ideal_variable[point]))
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
                        float(values[calibration, point][calibration_mask].mean())
                        if bool(calibration_mask.any())
                        else None
                    ),
                    "validation_mean": (
                        float(values[validation, point][validation_mask].mean())
                        if bool(validation_mask.any())
                        else None
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
        if not calibration_residuals or not validation_residuals:
            raise ValueError("device has no usable residuals")
        calibration_residual = torch.cat(calibration_residuals)
        validation_residual = torch.cat(validation_residuals)
        temporal_sigma = float(calibration_residual.std(unbiased=True))
        span = max(
            float(prediction[fit_points].max() - prediction[fit_points].min()),
            torch.finfo(torch.float64).eps,
        )
        validation_nrmse = float(validation_residual.square().mean().sqrt()) / span
        drift_key = _fit_drift_key(observation.primitive)
        parameter_drift = _relative_change(
            calibration_fit[drift_key], validation_fit[drift_key]
        )
        monotonicity = _monotonicity_xy(
            validation_x,
            validation_means,
            calibration_fit[drift_key]
            if observation.primitive != "phi-nl"
            else calibration_fit["log_slope_s"],
        )
        temporal_sigmas.append(temporal_sigma)
        drift_values.append(parameter_drift)
        nrmse_values.append(validation_nrmse)
        monotonicity_values.append(monotonicity)
        device_statistics.append(
            {
                "primitive": observation.primitive,
                "stage": observation.stage,
                "device": device,
                "physical_coordinate": observation.physical_coordinates[device],
                "temporal_sigma": temporal_sigma,
                "validation_normalized_rmse": validation_nrmse,
                "parameter_drift": parameter_drift,
                "monotonicity": monotonicity,
                "calibration_residual_count": int(calibration_residual.numel()),
                "validation_residual_count": int(validation_residual.numel()),
                **{f"calibration_{key}": value for key, value in calibration_fit.items()},
                **{f"validation_refit_{key}": value for key, value in validation_fit.items()},
            }
        )

    exactly_one = observation.delivered & (observation.spike_count == 1)
    if observation.output_kind == "spike-time-s":
        overall_delivery = float(exactly_one.to(torch.float64).mean())
        point_delivery = exactly_one.to(torch.float64).mean(dim=(0, 2))
    else:
        overall_delivery = float(observation.delivered.to(torch.float64).mean())
        point_delivery = observation.delivered.to(torch.float64).mean(dim=(0, 2))
    minimum_point_delivery = float(point_delivery.min())
    saturation_rate = float(observation.saturated.to(torch.float64).mean())
    maximum_nrmse = max(nrmse_values)
    maximum_drift = max(drift_values)
    minimum_monotonicity = min(monotonicity_values)
    gates = {
        "normalized_rmse": maximum_nrmse <= config.normalized_rmse_limit,
        "parameter_drift": maximum_drift <= config.parameter_drift_limit,
        "saturation": saturation_rate <= config.saturation_rate_limit,
        "monotonicity": minimum_monotonicity >= config.monotonicity_minimum,
        "overall_delivery": overall_delivery >= config.overall_delivery_minimum,
        "point_delivery": minimum_point_delivery >= config.point_delivery_minimum,
    }
    if observation.primitive == "psi-ne":
        gates["premature_spike"] = not bool((observation.spike_count > 0).any())
    if observation.precharge_cadc is not None:
        precharge = observation.precharge_cadc.to(torch.float64)
        precharge_means = torch.nanmean(precharge, dim=(0, 2))
        gates["precharge_recorded"] = bool(torch.isfinite(precharge).all())
        gates["precharge_monotonic"] = (
            _monotonicity(precharge_means, 1.0) >= config.monotonicity_minimum
        )
    validated = all(bool(value) for value in gates.values())

    parameter_keys = tuple(calibration_parameters[0])
    fixed_pattern: dict[str, Any] = {}
    for key in parameter_keys:
        values = torch.tensor(
            [parameters[key] for parameters in calibration_parameters],
            dtype=torch.float64,
        )
        fixed_pattern[key] = {
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std(unbiased=values.numel() > 1)),
        }
    sigma_tensor = torch.tensor(temporal_sigmas, dtype=torch.float64)
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
        "saturation_rate": saturation_rate,
        "validation_normalized_rmse_maximum": maximum_nrmse,
        "parameter_drift_maximum": maximum_drift,
        "monotonicity_minimum": minimum_monotonicity,
        "interpretation": "independent-marginal",
    }
    return PrimitiveValidation(
        primitive=observation.primitive,
        stage=observation.stage,
        validated=validated,
        diagnostic_only=observation.primitive == "phi-np" and observation.stage == "static",
        calibration_parameters=tuple(calibration_parameters),
        validation_refit_parameters=tuple(validation_parameters),
        device_statistics=tuple(device_statistics),
        transfer_rows=tuple(transfer_rows),
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
            mean = 25.0e-6 - 4.0e-6 * ideal
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
            sigma = 0.05 if quick else 0.20
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
        elif primitive == "psi-ne":
            saturated = (observed <= -128) | (observed >= 127)
        spike_count = torch.zeros_like(delivered, dtype=torch.int64)
        if output_kind == "spike-time-s":
            spike_count[delivered] = 1
        observed = observed.clone()
        observed[~delivered] = float("nan")
        precharge_cadc = None
        if primitive in ("phi-np", "phi-nl") and (
            stage == "dynamic" or primitive == "phi-nl"
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
            metadata=metadata,
        )


def primitive_result_key(observation: PrimitiveObservation) -> str:
    return f"{observation.primitive}_{observation.stage}"
