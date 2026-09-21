"""Operating-point search for measured BrainScaleS-2 encoder timing ratios.

The search keeps the primitive equations and physical output interpretation
unchanged.  It varies only physical operating-point controls, ranks candidates
with calibration repetitions, and reports held-out confirmation separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterable, Literal

import torch

from .primitive_noise import (
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    PrimitiveValidation,
)


SearchPrimitive = Literal["phi-np", "phi-nl"]


@dataclass(frozen=True)
class EncoderOperatingPointCandidate:
    """One operator-preserving physical configuration in the search grid."""

    candidate_id: str
    constant_current_code: int
    threshold_code: int
    ramp_stop_s: float
    precharge_input_fan_in: int
    precharge_weight_maximum: int
    exponential_input_fan_in: int
    exponential_input_weight: int

    def apply(self, base: PrimitiveNoiseConfig) -> PrimitiveNoiseConfig:
        observation_time_s = self.ramp_stop_s + (
            base.deadline_s - self.ramp_stop_s
        ) / 2.0
        return replace(
            base,
            constant_current_code=self.constant_current_code,
            threshold_code=self.threshold_code,
            input_late_s=self.ramp_stop_s,
            observation_time_s=observation_time_s,
            precharge_input_fan_in=self.precharge_input_fan_in,
            precharge_weight_maximum=self.precharge_weight_maximum,
            exponential_input_fan_in=self.exponential_input_fan_in,
            exponential_input_weight=self.exponential_input_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_fan_in_weight_pair(
    value: str, *, label: str
) -> tuple[int, int]:
    """Parse an explicit ``FAN_IN:WEIGHT`` physical input setting."""
    try:
        fan_in_text, weight_text = value.split(":", maxsplit=1)
        fan_in = int(fan_in_text)
        weight = int(weight_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} pair must use FAN_IN:WEIGHT") from error
    if fan_in <= 0 or not 1 <= weight <= 63:
        raise ValueError(
            f"{label} fan-in must be positive and weight must lie in [1, 63]"
        )
    return fan_in, weight


def parse_precharge_pair(value: str) -> tuple[int, int]:
    """Parse one precharge fan-in and maximum-weight pair."""
    return _parse_fan_in_weight_pair(value, label="precharge")


def parse_exponential_pair(value: str) -> tuple[int, int]:
    """Parse one exponential-current fan-in and weight pair."""
    return _parse_fan_in_weight_pair(value, label="exponential input")


def parse_current_stop_pair(value: str) -> tuple[int, float]:
    """Parse ``CURRENT_CODE:RAMP_STOP_US`` for coupled time scaling."""
    try:
        current_text, stop_text = value.split(":", maxsplit=1)
        current = int(current_text)
        stop_s = float(stop_text) * 1.0e-6
    except (TypeError, ValueError) as error:
        raise ValueError(
            "current-stop pair must use CURRENT_CODE:RAMP_STOP_US"
        ) from error
    if not 0 <= current <= 1022 or not math.isfinite(stop_s):
        raise ValueError("current-stop pair contains an invalid raw value")
    return current, stop_s


def build_encoder_operating_point_candidates(
    base: PrimitiveNoiseConfig,
    *,
    constant_current_codes: Iterable[int],
    threshold_codes: Iterable[int],
    ramp_stop_times_s: Iterable[float],
    precharge_pairs: Iterable[tuple[int, int]],
    exponential_pairs: Iterable[tuple[int, int]] | None = None,
    current_stop_pairs: Iterable[tuple[int, float]] | None = None,
) -> tuple[EncoderOperatingPointCandidate, ...]:
    """Create a deterministic grid of explicit physical controls."""
    thresholds = tuple(threshold_codes)
    resolved_precharge_pairs = tuple(precharge_pairs)
    resolved_exponential_pairs = (
        tuple(exponential_pairs)
        if exponential_pairs is not None
        else ((base.exponential_input_fan_in, base.exponential_input_weight),)
    )
    if current_stop_pairs is None:
        currents = tuple(constant_current_codes)
        ramp_stops = tuple(ramp_stop_times_s)
        current_stops = tuple(
            (current, ramp_stop)
            for current in currents
            for ramp_stop in ramp_stops
        )
    else:
        current_stops = tuple(current_stop_pairs)
    candidates: list[EncoderOperatingPointCandidate] = []
    seen: set[str] = set()
    for current, ramp_stop_s in current_stops:
        for threshold in thresholds:
            for fan_in, weight in resolved_precharge_pairs:
                for (
                    exponential_fan_in,
                    exponential_weight,
                ) in resolved_exponential_pairs:
                    if exponential_fan_in <= 0 or not 1 <= exponential_weight <= 63:
                        raise ValueError(
                            "exponential input fan-in must be positive and weight "
                            "must lie in [1, 63]"
                        )
                    if not 0 <= current <= 1022:
                        raise ValueError(
                            "constant-current code must lie in [0, 1022]"
                        )
                    if not 0 <= threshold <= 1022:
                        raise ValueError("threshold code must lie in [0, 1022]")
                    if not base.input_early_s < ramp_stop_s < base.deadline_s:
                        raise ValueError(
                            "ramp stop must lie between input start and deadline"
                        )
                    candidate_id = (
                        f"cc{current:04d}_th{threshold:04d}_"
                        f"stop{ramp_stop_s * 1.0e6:06.2f}us_"
                        f"fanin{fan_in:02d}_w{weight:02d}_"
                        f"nlfanin{exponential_fan_in:02d}_nlw{exponential_weight:02d}"
                    )
                    if candidate_id in seen:
                        continue
                    seen.add(candidate_id)
                    candidates.append(
                        EncoderOperatingPointCandidate(
                            candidate_id=candidate_id,
                            constant_current_code=current,
                            threshold_code=threshold,
                            ramp_stop_s=ramp_stop_s,
                            precharge_input_fan_in=fan_in,
                            precharge_weight_maximum=weight,
                            exponential_input_fan_in=exponential_fan_in,
                            exponential_input_weight=exponential_weight,
                        )
                    )
    if not candidates:
        raise ValueError("operating-point search grid is empty")
    return tuple(candidates)


def _lookup_observation(
    observations: Iterable[PrimitiveObservation], primitive: str, stage: str
) -> PrimitiveObservation:
    matches = [
        observation
        for observation in observations
        if observation.primitive == primitive and observation.stage == stage
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one observation for {primitive}/{stage}, found {len(matches)}"
        )
    return matches[0]


def _lookup_validation(
    validations: Iterable[PrimitiveValidation], primitive: str, stage: str
) -> PrimitiveValidation:
    matches = [
        validation
        for validation in validations
        if validation.primitive == primitive and validation.stage == stage
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one validation for {primitive}/{stage}, found {len(matches)}"
        )
    return matches[0]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = torch.tensor(
        [value for value in values if math.isfinite(value)], dtype=torch.float64
    )
    if not finite.numel():
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
            "q1": None,
            "q3": None,
        }
    return {
        "count": int(finite.numel()),
        "minimum": float(finite.min()),
        "median": float(torch.quantile(finite, 0.5)),
        "maximum": float(finite.max()),
        "q1": float(torch.quantile(finite, 0.25)),
        "q3": float(torch.quantile(finite, 0.75)),
    }


def _prediction(
    observation: PrimitiveObservation,
    parameters: dict[str, float],
) -> torch.Tensor:
    if observation.primitive == "phi-np":
        return parameters["offset_s"] + parameters[
            "slope_s_per_code"
        ] * observation.ideal_variable.to(torch.float64)
    if observation.primitive == "phi-nl":
        return parameters["offset_s"] + parameters[
            "log_slope_s"
        ] * torch.log(
            observation.input_code.to(torch.float64)
            - parameters["lower_bound_code"]
        )
    raise ValueError("timing-ratio search supports only phi-np and phi-nl")


def calibration_transfer_gate(
    observation: PrimitiveObservation,
    validation: PrimitiveValidation,
    config: PrimitiveNoiseConfig,
    *,
    screening: bool = False,
) -> dict[str, Any]:
    trial_slice = slice(0, config.calibration_repeats)
    usable = (
        observation.delivered[trial_slice]
        & ~observation.saturated[trial_slice]
        & torch.isfinite(observation.observed[trial_slice])
    )
    fit_points = (
        observation.fit_point_mask.to(torch.bool)
        if observation.fit_point_mask is not None
        else torch.ones(observation.point_count, dtype=torch.bool)
    )
    nrmse_values: list[float] = []
    slope_direction: list[bool] = []
    rank_monotonicity_values: list[float] = []
    enough_samples = True
    for device, parameters in enumerate(validation.calibration_parameters):
        try:
            predicted = _prediction(observation, parameters)
        except (KeyError, ValueError):
            enough_samples = False
            continue
        means: list[torch.Tensor] = []
        references: list[torch.Tensor] = []
        for point in range(observation.point_count):
            if not bool(fit_points[point]):
                continue
            selected = observation.observed[
                trial_slice, point, device
            ][usable[:, point, device]]
            if selected.numel() < 2:
                enough_samples = False
                continue
            means.append(selected.to(torch.float64).mean())
            references.append(predicted[point])
        if len(means) < 2:
            enough_samples = False
            continue
        mean_tensor = torch.stack(means)
        reference_tensor = torch.stack(references)
        span = float(reference_tensor.max() - reference_tensor.min())
        if not math.isfinite(span) or span <= torch.finfo(torch.float64).eps:
            enough_samples = False
            continue
        nrmse_values.append(
            float(torch.mean((mean_tensor - reference_tensor).square()).sqrt())
            / span
        )
        if observation.primitive == "phi-np":
            expected = torch.arange(mean_tensor.numel(), dtype=torch.float64)
            order = torch.argsort(-mean_tensor, stable=True)
            ranks = torch.empty(mean_tensor.numel(), dtype=torch.float64)
            ranks[order] = torch.arange(mean_tensor.numel(), dtype=torch.float64)
            expected -= expected.mean()
            ranks -= ranks.mean()
            denominator = (
                expected.square().sum().sqrt() * ranks.square().sum().sqrt()
            )
            rank_monotonicity_values.append(
                float((expected * ranks).sum() / denominator)
                if float(denominator) > 0.0
                else 0.0
            )
        slope = (
            parameters.get("slope_s_per_code")
            if observation.primitive == "phi-np"
            else parameters.get("log_slope_s")
        )
        slope_direction.append(slope is not None and math.isfinite(slope) and slope < 0)
    miss_rate = float((~observation.delivered[trial_slice]).to(torch.float64).mean())
    multiple_rate = float(
        (observation.spike_count[trial_slice] > 1).to(torch.float64).mean()
    )
    saturation_rate = float(
        observation.saturated[trial_slice].to(torch.float64).mean()
    )
    maximum_nrmse = max(nrmse_values, default=float("inf"))
    gates = {
        "enough_samples": enough_samples,
        "negative_transfer_slope": bool(slope_direction) and all(slope_direction),
        "normalized_rmse": maximum_nrmse <= config.normalized_rmse_limit,
        "deadline_miss_rate": miss_rate <= config.deadline_miss_rate_limit,
        "multiple_spike_rate": (
            multiple_rate <= config.multiple_spike_rate_limit
        ),
        "saturation_rate": saturation_rate <= config.saturation_rate_limit,
        "rank_monotonicity": (
            observation.primitive != "phi-np"
            or (
                bool(rank_monotonicity_values)
                and min(rank_monotonicity_values) >= config.monotonicity_minimum
            )
        ),
    }
    required_gate_names = (
        tuple(gates)
        if not screening
        else (
            "enough_samples",
            "negative_transfer_slope",
            "deadline_miss_rate",
            "multiple_spike_rate",
            "saturation_rate",
        )
    )
    return {
        "eligible": all(gates[name] for name in required_gate_names),
        "strict_eligible": all(gates.values()),
        "screening": screening,
        "required_gates": list(required_gate_names),
        "gates": gates,
        "maximum_normalized_rmse": maximum_nrmse,
        "miss_rate": miss_rate,
        "multiple_spike_rate": multiple_rate,
        "saturation_rate": saturation_rate,
        "minimum_rank_monotonicity": min(rank_monotonicity_values, default=None),
    }


def _conditional_timing_ratio(
    observation: PrimitiveObservation,
    *,
    trial_slice: slice,
    signal_spans_s: tuple[float, ...],
) -> dict[str, Any]:
    usable = (
        observation.delivered[trial_slice]
        & ~observation.saturated[trial_slice]
        & torch.isfinite(observation.observed[trial_slice])
    )
    fit_points = (
        observation.fit_point_mask.to(torch.bool)
        if observation.fit_point_mask is not None
        else torch.ones(observation.point_count, dtype=torch.bool)
    )
    rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    for device, span in enumerate(signal_spans_s):
        variances: list[float] = []
        complete = math.isfinite(span) and span > 0.0
        for point in range(observation.point_count):
            if not bool(fit_points[point]):
                continue
            selected = observation.observed[
                trial_slice, point, device
            ][usable[:, point, device]]
            if selected.numel() < 2:
                complete = False
                continue
            variances.append(float(selected.to(torch.float64).var(unbiased=True)))
        sigma_s = (
            math.sqrt(sum(variances) / len(variances))
            if complete and variances
            else float("nan")
        )
        ratio = sigma_s / span if math.isfinite(sigma_s) else float("nan")
        ratios.append(ratio)
        rows.append(
            {
                "device": device,
                "physical_coordinate": observation.physical_coordinates[device],
                "conditional_sigma_s": sigma_s,
                "signal_span_s": span,
                "r_t": ratio,
                "complete": complete and bool(variances),
            }
        )
    return {"summary": _summary(ratios), "devices": rows}


def score_encoder_operating_point(
    observations: list[PrimitiveObservation],
    validations: list[PrimitiveValidation],
    config: PrimitiveNoiseConfig,
    *,
    primitive: SearchPrimitive,
    screening: bool = False,
) -> dict[str, Any]:
    """Score one candidate without using held-out values for selection."""
    np_static_observation = _lookup_observation(observations, "phi-np", "static")
    np_static_validation = _lookup_validation(validations, "phi-np", "static")
    np_observation = _lookup_observation(observations, "phi-np", "dynamic")
    np_validation = _lookup_validation(validations, "phi-np", "dynamic")
    spans = tuple(
        abs(parameters.get("slope_s_per_code", float("nan"))) * 31.0
        for parameters in np_validation.calibration_parameters
    )
    measured = [
        ("phi-np", np_observation, np_validation),
    ]
    if primitive == "phi-nl":
        measured.append(
            (
                "phi-nl",
                _lookup_observation(observations, "phi-nl", "transfer"),
                _lookup_validation(validations, "phi-nl", "transfer"),
            )
        )
    primitive_scores: dict[str, Any] = {}
    static_calibration_gate = calibration_transfer_gate(
        np_static_observation,
        np_static_validation,
        config,
        screening=screening,
    )
    selection_gates: list[bool] = [bool(static_calibration_gate["eligible"])]
    held_out_gates: list[bool] = [np_static_validation.validated]
    for name, observation, validation in measured:
        transfer_gate = calibration_transfer_gate(
            observation, validation, config, screening=screening
        )
        calibration = _conditional_timing_ratio(
            observation,
            trial_slice=slice(0, config.calibration_repeats),
            signal_spans_s=spans,
        )
        held_out = _conditional_timing_ratio(
            observation,
            trial_slice=slice(config.calibration_repeats, config.repeats),
            signal_spans_s=spans,
        )
        selection_gates.append(bool(transfer_gate["eligible"]))
        held_out_gates.append(validation.validated)
        primitive_scores[name] = {
            "calibration": calibration,
            "held_out": held_out,
            "calibration_transfer": transfer_gate,
            "held_out_validated": validation.validated,
        }
    target_score = primitive_scores[primitive]
    selection_objective = target_score["calibration"]["summary"]["median"]
    validation_objective = target_score["held_out"]["summary"]["median"]
    selection_eligible = all(selection_gates) and selection_objective is not None
    held_out_validated = all(held_out_gates) and validation_objective is not None
    if not selection_eligible:
        selection_objective = None
    if not held_out_validated:
        validation_objective = None
    return {
        "schema_version": 1,
        "primitive": primitive,
        "screening": screening,
        "selection_eligible": selection_eligible,
        "held_out_validated": held_out_validated,
        "selection_objective_rt": selection_objective,
        "validation_objective_rt": validation_objective,
        "targets": {
            "hardware_feasibility_1e-3": (
                validation_objective is not None
                and validation_objective <= 1.0e-3
            ),
            "meaningful_recovery_1e-4": (
                validation_objective is not None
                and validation_objective <= 1.0e-4
            ),
            "near_clean_recovery_3e-5": (
                validation_objective is not None
                and validation_objective <= 3.0e-5
            ),
        },
        "np_static": {
            "calibration_transfer": static_calibration_gate,
            "held_out_validated": np_static_validation.validated,
        },
        "primitive_scores": primitive_scores,
    }
