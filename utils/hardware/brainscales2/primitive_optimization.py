"""Operating-point search for measured BrainScaleS-2 encoder timing ratios.

The search keeps the primitive equations and physical output interpretation
unchanged.  It varies only physical operating-point controls, ranks candidates
with calibration repetitions, and reports held-out confirmation separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
import math
from typing import Any, Iterable, Literal

import torch

from .primitive_noise import (
    PrimitiveNoiseConfig,
    PrimitiveObservation,
    PrimitiveValidation,
    RESET_ASSERT_S,
)


SearchPrimitive = Literal["phi-np", "phi-nl"]


@dataclass(frozen=True)
class EncoderOperatingPointCandidate:
    """One operator-preserving physical configuration in the search grid."""

    candidate_id: str
    constant_current_code: int
    threshold_code: int
    reset_current_code: int
    reset_current_enable_multiplication: bool
    static_reset_release_s: float
    dynamic_reset_release_s: float
    membrane_capacitance_code: int | None
    threshold_comparator_bias_code: int | None
    excitatory_input_i_bias_tau_code: int | None
    excitatory_input_i_bias_gm_code: int | None
    synaptic_input_drop_bias_code: int | None
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
            reset_current_code=self.reset_current_code,
            reset_current_enable_multiplication=(
                self.reset_current_enable_multiplication
            ),
            static_reset_release_s=self.static_reset_release_s,
            dynamic_reset_release_s=self.dynamic_reset_release_s,
            membrane_capacitance_code=self.membrane_capacitance_code,
            threshold_comparator_bias_code=self.threshold_comparator_bias_code,
            excitatory_input_i_bias_tau_code=(
                self.excitatory_input_i_bias_tau_code
            ),
            excitatory_input_i_bias_gm_code=(
                self.excitatory_input_i_bias_gm_code
            ),
            synaptic_input_drop_bias_code=self.synaptic_input_drop_bias_code,
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
    reset_current_codes: Iterable[int] | None = None,
    reset_current_multiplication_modes: Iterable[bool] | None = None,
    static_reset_release_times_s: Iterable[float] | None = None,
    dynamic_reset_release_times_s: Iterable[float] | None = None,
    membrane_capacitance_codes: Iterable[int | None] | None = None,
    threshold_comparator_bias_codes: Iterable[int | None] | None = None,
    excitatory_input_i_bias_tau_codes: Iterable[int | None] | None = None,
    excitatory_input_i_bias_gm_codes: Iterable[int | None] | None = None,
    synaptic_input_drop_bias_codes: Iterable[int | None] | None = None,
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
    capacitance_codes = (
        tuple(membrane_capacitance_codes)
        if membrane_capacitance_codes is not None
        else (base.membrane_capacitance_code,)
    )
    comparator_bias_codes = (
        tuple(threshold_comparator_bias_codes)
        if threshold_comparator_bias_codes is not None
        else (base.threshold_comparator_bias_code,)
    )
    synaptic_tau_codes = (
        tuple(excitatory_input_i_bias_tau_codes)
        if excitatory_input_i_bias_tau_codes is not None
        else (base.excitatory_input_i_bias_tau_code,)
    )
    synaptic_gain_codes = (
        tuple(excitatory_input_i_bias_gm_codes)
        if excitatory_input_i_bias_gm_codes is not None
        else (base.excitatory_input_i_bias_gm_code,)
    )
    synaptic_drop_codes = (
        tuple(synaptic_input_drop_bias_codes)
        if synaptic_input_drop_bias_codes is not None
        else (base.synaptic_input_drop_bias_code,)
    )
    reset_currents = (
        tuple(reset_current_codes)
        if reset_current_codes is not None
        else (base.reset_current_code,)
    )
    reset_multiplication_modes = (
        tuple(reset_current_multiplication_modes)
        if reset_current_multiplication_modes is not None
        else (base.reset_current_enable_multiplication,)
    )
    static_reset_releases = (
        tuple(static_reset_release_times_s)
        if static_reset_release_times_s is not None
        else (base.static_reset_release_s,)
    )
    dynamic_reset_releases = (
        tuple(dynamic_reset_release_times_s)
        if dynamic_reset_release_times_s is not None
        else (base.dynamic_reset_release_s,)
    )
    candidates: list[EncoderOperatingPointCandidate] = []
    seen: set[str] = set()
    combinations = product(
        current_stops,
        thresholds,
        capacitance_codes,
        comparator_bias_codes,
        synaptic_tau_codes,
        synaptic_gain_codes,
        synaptic_drop_codes,
        reset_currents,
        reset_multiplication_modes,
        static_reset_releases,
        dynamic_reset_releases,
        resolved_precharge_pairs,
        resolved_exponential_pairs,
    )
    for (
        (current, ramp_stop_s),
        threshold,
        capacitance_code,
        comparator_bias_code,
        synaptic_tau_code,
        synaptic_gain_code,
        synaptic_drop_code,
        reset_current,
        reset_multiplication,
        static_release_s,
        dynamic_release_s,
        (fan_in, weight),
        (exponential_fan_in, exponential_weight),
    ) in combinations:
        if not isinstance(reset_multiplication, bool):
            raise TypeError("reset current multiplication mode must be a bool")
        if not 0 <= reset_current <= 1022:
            raise ValueError("reset current code must lie in [0, 1022]")
        if not RESET_ASSERT_S < static_release_s < base.input_early_s:
            raise ValueError("static reset release must precede the ramp")
        precharge_time_s = max(0.5e-6, base.input_early_s - 2.0e-6)
        if (
            not RESET_ASSERT_S < dynamic_release_s < precharge_time_s
            or math.isclose(
                dynamic_release_s,
                precharge_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("dynamic reset release must precede precharge")
        candidate = _build_encoder_candidate(
            base=base,
            seen=seen,
            current=current,
            ramp_stop_s=ramp_stop_s,
            threshold=threshold,
            reset_current=reset_current,
            reset_multiplication=reset_multiplication,
            static_release_s=static_release_s,
            dynamic_release_s=dynamic_release_s,
            capacitance_code=capacitance_code,
            comparator_bias_code=comparator_bias_code,
            synaptic_tau_code=synaptic_tau_code,
            synaptic_gain_code=synaptic_gain_code,
            synaptic_drop_code=synaptic_drop_code,
            fan_in=fan_in,
            weight=weight,
            exponential_fan_in=exponential_fan_in,
            exponential_weight=exponential_weight,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("operating-point search grid is empty")
    return tuple(candidates)


def _build_encoder_candidate(
    *,
    base: PrimitiveNoiseConfig,
    seen: set[str],
    current: int,
    ramp_stop_s: float,
    threshold: int,
    reset_current: int,
    reset_multiplication: bool,
    static_release_s: float,
    dynamic_release_s: float,
    capacitance_code: int | None,
    comparator_bias_code: int | None,
    synaptic_tau_code: int | None,
    synaptic_gain_code: int | None,
    synaptic_drop_code: int | None,
    fan_in: int,
    weight: int,
    exponential_fan_in: int,
    exponential_weight: int,
) -> EncoderOperatingPointCandidate | None:
    """Validate and materialize one deterministic search-grid point."""
    if exponential_fan_in <= 0 or not 1 <= exponential_weight <= 63:
        raise ValueError(
            "exponential input fan-in must be positive and weight must lie in [1, 63]"
        )
    if not 0 <= current <= 1022:
        raise ValueError("constant-current code must lie in [0, 1022]")
    if not 0 <= threshold <= 1022:
        raise ValueError("threshold code must lie in [0, 1022]")
    if capacitance_code is not None and not 0 <= capacitance_code <= 63:
        raise ValueError("membrane capacitance code must lie in [0, 63]")
    if comparator_bias_code is not None and not 0 <= comparator_bias_code <= 1022:
        raise ValueError("threshold comparator bias code must lie in [0, 1022]")
    for label, code in (
        ("synaptic time constant bias", synaptic_tau_code),
        ("synaptic gain bias", synaptic_gain_code),
        ("synaptic input drop bias", synaptic_drop_code),
    ):
        if code is not None and not 0 <= code <= 1022:
            raise ValueError(f"{label} code must lie in [0, 1022]")
    if not base.input_early_s < ramp_stop_s < base.deadline_s:
        raise ValueError("ramp stop must lie between input start and deadline")
    capacitance_id = (
        "capkeep" if capacitance_code is None else f"cap{capacitance_code:02d}"
    )
    comparator_id = (
        "cmpkeep"
        if comparator_bias_code is None
        else f"cmp{comparator_bias_code:04d}"
    )
    synaptic_tau_id = (
        "syntaukeep" if synaptic_tau_code is None else f"syntau{synaptic_tau_code:04d}"
    )
    synaptic_gain_id = (
        "syngmkeep" if synaptic_gain_code is None else f"syngm{synaptic_gain_code:04d}"
    )
    synaptic_drop_id = (
        "syndropkeep" if synaptic_drop_code is None else f"syndrop{synaptic_drop_code:04d}"
    )
    static_release_us = static_release_s * 1.0e6
    dynamic_release_us = dynamic_release_s * 1.0e6
    candidate_id = (
        f"cc{current:04d}_th{threshold:04d}_"
        f"rsti{reset_current:04d}_"
        f"rstm{int(reset_multiplication)}_"
        f"rs{static_release_us:04.1f}us_"
        f"rd{dynamic_release_us:04.1f}us_"
        f"{capacitance_id}_{comparator_id}_"
        f"{synaptic_tau_id}_{synaptic_gain_id}_{synaptic_drop_id}_"
        f"stop{ramp_stop_s * 1.0e6:06.2f}us_"
        f"fanin{fan_in:02d}_w{weight:02d}_"
        f"nlfanin{exponential_fan_in:02d}_"
        f"nlw{exponential_weight:02d}"
    )
    if candidate_id in seen:
        return None
    seen.add(candidate_id)
    return EncoderOperatingPointCandidate(
        candidate_id=candidate_id,
        constant_current_code=current,
        threshold_code=threshold,
        reset_current_code=reset_current,
        reset_current_enable_multiplication=reset_multiplication,
        static_reset_release_s=static_release_s,
        dynamic_reset_release_s=dynamic_release_s,
        membrane_capacitance_code=capacitance_code,
        threshold_comparator_bias_code=comparator_bias_code,
        excitatory_input_i_bias_tau_code=synaptic_tau_code,
        excitatory_input_i_bias_gm_code=synaptic_gain_code,
        synaptic_input_drop_bias_code=synaptic_drop_code,
        ramp_stop_s=ramp_stop_s,
        precharge_input_fan_in=fan_in,
        precharge_weight_maximum=weight,
        exponential_input_fan_in=exponential_fan_in,
        exponential_input_weight=exponential_weight,
    )


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
    trial_slice: slice | None = None,
    device_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if trial_slice is None:
        trial_slice = slice(0, config.calibration_repeats)
    devices = (
        tuple(range(observation.device_count))
        if device_indices is None
        else device_indices
    )
    if not devices or any(
        device < 0 or device >= observation.device_count for device in devices
    ):
        raise ValueError("transfer gate device selection is invalid")
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
    for device in devices:
        parameters = validation.calibration_parameters[device]
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
    selected_delivery = observation.delivered[trial_slice, :, devices]
    selected_spike_count = observation.spike_count[trial_slice, :, devices]
    selected_saturation = observation.saturated[trial_slice, :, devices]
    miss_rate = float((~selected_delivery).to(torch.float64).mean())
    multiple_rate = float(
        (selected_spike_count > 1).to(torch.float64).mean()
    )
    saturation_rate = float(selected_saturation.to(torch.float64).mean())
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
        "device_indices": list(devices),
        "required_gates": list(required_gate_names),
        "gates": gates,
        "maximum_normalized_rmse": maximum_nrmse,
        "miss_rate": miss_rate,
        "multiple_spike_rate": multiple_rate,
        "saturation_rate": saturation_rate,
        "minimum_rank_monotonicity": min(rank_monotonicity_values, default=None),
    }


def _held_out_device_gate(
    observation: PrimitiveObservation,
    validation: PrimitiveValidation,
    config: PrimitiveNoiseConfig,
    *,
    device: int,
    screening: bool,
) -> dict[str, Any]:
    """Evaluate one frozen physical circuit without pooling other circuits."""
    result = calibration_transfer_gate(
        observation,
        validation,
        config,
        screening=screening,
        trial_slice=slice(config.calibration_repeats, config.repeats),
        device_indices=(device,),
    )
    if screening:
        return result
    statistics = validation.device_statistics[device]
    result["gates"]["fit_available"] = bool(statistics["fit_available"])
    result["gates"]["parameter_drift"] = (
        statistics["parameter_drift"] <= config.parameter_drift_limit
    )
    if observation.precharge_cadc is not None:
        fit_points = (
            observation.fit_point_mask.to(torch.bool)
            if observation.fit_point_mask is not None
            else torch.ones(observation.point_count, dtype=torch.bool)
        )
        recorded = torch.isfinite(
            observation.precharge_cadc[:, fit_points, device]
        ).any(dim=0)
        result["gates"]["precharge_recorded"] = bool(recorded.all())
    result["required_gates"] = list(result["gates"])
    result["eligible"] = all(result["gates"].values())
    result["strict_eligible"] = result["eligible"]
    return result


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


def _select_calibration_device(
    calibration: dict[str, Any],
    held_out: dict[str, Any],
    *,
    eligible_devices: set[int],
) -> dict[str, Any] | None:
    """Freeze the lowest-ratio physical circuit using calibration data only."""
    candidates = [
        row
        for row in calibration["devices"]
        if (
            row["device"] in eligible_devices
            and row["complete"]
            and math.isfinite(row["r_t"])
        )
    ]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda row: (row["r_t"], row["physical_coordinate"]),
    )
    held_out_by_device = {
        row["device"]: row for row in held_out["devices"]
    }
    confirmation = held_out_by_device[selected["device"]]
    return {
        "device": selected["device"],
        "physical_coordinate": selected["physical_coordinate"],
        "calibration_rt": selected["r_t"],
        "held_out_rt": (
            confirmation["r_t"]
            if confirmation["complete"]
            and math.isfinite(confirmation["r_t"])
            else None
        ),
    }


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
    measured = [("phi-np", np_observation, np_validation)]
    if primitive == "phi-nl":
        measured.append(
            (
                "phi-nl",
                _lookup_observation(observations, "phi-nl", "transfer"),
                _lookup_validation(validations, "phi-nl", "transfer"),
            )
        )
    stages = {
        "np_static": (np_static_observation, np_static_validation),
        **{
            name: (observation, validation)
            for name, observation, validation in measured
        },
    }
    stage_gates: dict[str, dict[str, Any]] = {}
    for stage_name, (observation, validation) in stages.items():
        stage_gates[stage_name] = {
            "aggregate_calibration": calibration_transfer_gate(
                observation,
                validation,
                config,
                screening=screening,
            ),
            "calibration_by_device": tuple(
                calibration_transfer_gate(
                    observation,
                    validation,
                    config,
                    screening=screening,
                    device_indices=(device,),
                )
                for device in range(config.device_count)
            ),
            "held_out_by_device": tuple(
                _held_out_device_gate(
                    observation,
                    validation,
                    config,
                    device=device,
                    screening=screening,
                )
                for device in range(config.device_count)
            ),
        }

    primitive_scores: dict[str, Any] = {}
    for name, observation, validation in measured:
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
        required_stages = (
            ("np_static", "phi-np")
            if name == "phi-np"
            else ("np_static", "phi-np", "phi-nl")
        )
        eligible_devices = {
            device
            for device in range(config.device_count)
            if all(
                stage_gates[stage_name]["calibration_by_device"][device][
                    "eligible"
                ]
                for stage_name in required_stages
            )
        }
        selected_device = _select_calibration_device(
            calibration,
            held_out,
            eligible_devices=eligible_devices,
        )
        selected_index = (
            selected_device["device"] if selected_device is not None else None
        )
        held_out_validated = bool(
            selected_device is not None
            and selected_device["held_out_rt"] is not None
            and all(
                stage_gates[stage_name]["held_out_by_device"][selected_index][
                    "eligible"
                ]
                for stage_name in required_stages
            )
        )
        selected_gates = (
            {
                stage_name: {
                    "calibration": stage_gates[stage_name][
                        "calibration_by_device"
                    ][selected_index],
                    "held_out": stage_gates[stage_name]["held_out_by_device"][
                        selected_index
                    ],
                }
                for stage_name in required_stages
            }
            if selected_index is not None
            else None
        )
        primitive_scores[name] = {
            "calibration": calibration,
            "held_out": held_out,
            "calibration_selected_device": selected_device,
            "eligible_device_count": len(eligible_devices),
            "selection_eligible": selected_device is not None,
            "held_out_validated": held_out_validated,
            "selected_device_gates": selected_gates,
            "calibration_transfer": stage_gates[name][
                "aggregate_calibration"
            ],
        }
    target_score = primitive_scores[primitive]
    selected_device = target_score["calibration_selected_device"]
    selection_objective = (
        selected_device["calibration_rt"]
        if selected_device is not None
        else None
    )
    validation_objective = (
        selected_device["held_out_rt"]
        if selected_device is not None
        else None
    )
    selection_eligible = bool(target_score["selection_eligible"])
    held_out_validated = bool(target_score["held_out_validated"])
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
        "calibration_selected_device": selected_device,
        "targets": {
            "hardware_feasibility_1e-3": (
                not screening
                and validation_objective is not None
                and validation_objective <= 1.0e-3
            ),
            "meaningful_recovery_1e-4": (
                not screening
                and validation_objective is not None
                and validation_objective <= 1.0e-4
            ),
            "near_clean_recovery_3e-5": (
                not screening
                and validation_objective is not None
                and validation_objective <= 3.0e-5
            ),
        },
        "np_static": {
            "calibration_transfer": stage_gates["np_static"][
                "aggregate_calibration"
            ],
            "held_out_validated": np_static_validation.validated,
        },
        "primitive_scores": primitive_scores,
    }
