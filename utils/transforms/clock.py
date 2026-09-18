"""Process-wide discrete clock semantics for TTFS operator evaluation.

The maintained Transformer path normally evaluates continuous spike times with
closed-form tensor arithmetic. Clock-driven evaluation instead advances encoder
observation, pulse-width accumulation, and exponential state through explicit
Python time-step loops. It mutates one state tensor per operator rather than
materializing a leading time axis, but it does not skip inactive clock steps or
replace repeated updates with a closed-form duration or power.

The configuration and counters are mutable process-wide state, matching the
existing timing-noise configuration.  One configured evaluation therefore runs
in one process and must not use ``DataParallel``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor

from .types import TimeBounds


@dataclass(frozen=True)
class ClockDrivenConfig:
    """Global clock configuration shared by every temporal primitive."""

    enabled: bool = False
    time_step: float = 0.0


class ClockDrivenCounts(TypedDict):
    """Aggregate encoder measurements for one stable call-site name."""

    events: int
    rounded_events: int
    absolute_error_sum: float
    absolute_error_max: float
    windows: int
    minimum_window_steps: int
    maximum_window_steps: int


class ClockUpdateCounts(TypedDict):
    """Executed sequential-loop work for one clock-driven state update kind."""

    calls: int
    time_steps: int
    element_updates: int


@dataclass
class _ClockAccumulator:
    """Device-local scalar accumulators reduced only when reporting finishes."""

    events: int = 0
    rounded_events: Tensor | None = None
    absolute_error_sum: Tensor | None = None
    absolute_error_max: Tensor | None = None
    windows: int = 0
    minimum_window_steps: int = 2**63 - 1
    maximum_window_steps: int = 0


_GLOBAL_CLOCK_CONFIG = ClockDrivenConfig()
_CLOCK_STATS: dict[str, _ClockAccumulator] = {}
_CLOCK_UPDATE_STATS: dict[str, ClockUpdateCounts] = {}


# @lat: [[clock-driven#Clock-Driven TTFS Evaluation#Execution Contract]]
def set_clock_driven(*, enabled: bool, time_step: float = 0.0) -> None:
    """Install one global clock and start a fresh measurement interval."""

    global _GLOBAL_CLOCK_CONFIG
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    normalized_step = float(time_step)
    if enabled:
        if not math.isfinite(normalized_step) or normalized_step <= 0.0:
            raise ValueError(
                "clock-driven time_step must be finite and strictly positive"
            )
    elif normalized_step != 0.0:
        raise ValueError("disabled clock-driven execution requires time_step=0")
    _GLOBAL_CLOCK_CONFIG = ClockDrivenConfig(
        enabled=enabled,
        time_step=normalized_step,
    )
    clear_clock_driven_stats()


def get_clock_driven() -> ClockDrivenConfig:
    """Return the active immutable clock configuration."""

    return _GLOBAL_CLOCK_CONFIG


def clear_clock_driven_stats() -> None:
    """Clear clock measurements without changing the active configuration."""

    _CLOCK_STATS.clear()
    _CLOCK_UPDATE_STATS.clear()


def get_clock_driven_stats() -> dict[str, ClockDrivenCounts]:
    """Return a detached snapshot of encoder clock measurements."""

    snapshot: dict[str, ClockDrivenCounts] = {}
    for site, accumulator in _CLOCK_STATS.items():
        snapshot[site] = {
            "events": accumulator.events,
            "rounded_events": int(
                accumulator.rounded_events.item()
                if accumulator.rounded_events is not None
                else 0
            ),
            "absolute_error_sum": float(
                accumulator.absolute_error_sum.item()
                if accumulator.absolute_error_sum is not None
                else 0.0
            ),
            "absolute_error_max": float(
                accumulator.absolute_error_max.item()
                if accumulator.absolute_error_max is not None
                else 0.0
            ),
            "windows": accumulator.windows,
            "minimum_window_steps": accumulator.minimum_window_steps,
            "maximum_window_steps": accumulator.maximum_window_steps,
        }
    return snapshot


def get_clock_update_stats() -> dict[str, ClockUpdateCounts]:
    """Return counters proving which explicit state-update loops executed."""

    return {kind: dict(counts) for kind, counts in _CLOCK_UPDATE_STATS.items()}


def record_clock_updates(kind: str, *, time_steps: int, elements: int) -> None:
    """Accumulate work performed by one explicit sequential update loop."""

    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("clock update kind must be a non-empty string")
    if time_steps < 0 or elements < 0:
        raise ValueError("clock update counters must be non-negative")
    counts = _CLOCK_UPDATE_STATS.setdefault(
        kind,
        {"calls": 0, "time_steps": 0, "element_updates": 0},
    )
    counts["calls"] += 1
    counts["time_steps"] += time_steps
    counts["element_updates"] += time_steps * elements


def _stats_for(site: str) -> _ClockAccumulator:
    if not isinstance(site, str):
        raise TypeError("clock-driven statistics site must be a string")
    if not site.strip():
        raise ValueError("clock-driven statistics site must not be empty")
    counts = _CLOCK_STATS.get(site)
    if counts is None:
        counts = _ClockAccumulator()
        _CLOCK_STATS[site] = counts
    return counts


def _scalar_ceil_steps(value: float, time_step: float) -> int:
    """Return the first clock index not earlier than a scalar time."""

    quotient = value / time_step
    nearest = round(quotient)
    tolerance = 8.0 * math.ulp(max(abs(quotient), 1.0))
    if abs(quotient - nearest) <= tolerance:
        return int(nearest)
    return math.ceil(quotient)


def quantize_encoder_output(
    time: Tensor,
    domain: TimeBounds,
    *,
    site: str,
) -> tuple[Tensor, TimeBounds]:
    """Place one encoder output and its deadline on the active global clock.

    Threshold crossings use causal rounding: an event is delivered on the first
    clock edge at or after its continuous crossing time.  The declared deadline is
    rounded by the same rule, so the lower endpoint remains deliverable inside the
    expanded closed interval.
    """

    config = get_clock_driven()
    if not config.enabled:
        return domain.clamp(time), domain
    if not torch.is_floating_point(time):
        raise TypeError("clock-driven encoder time must be floating-point")

    step = config.time_step
    minimum_step = _scalar_ceil_steps(float(domain.min), step)
    maximum_step = _scalar_ceil_steps(float(domain.max), step)
    if maximum_step < minimum_step:
        raise ValueError("clock-driven time bounds collapsed after quantization")
    quantized_domain = TimeBounds(minimum_step * step, maximum_step * step)

    # The continuous formula supplies the threshold-crossing target, but delivery
    # is observed only while advancing the global clock. This loop is deliberately
    # sequential: no ceil-based index shortcut is used in the production path.
    nominal = domain.clamp(time)
    step_tensor = time.new_tensor(step)
    quotient = nominal / step_tensor
    tolerance = 8.0 * torch.finfo(time.dtype).eps * torch.maximum(
        quotient.abs(), torch.ones_like(quotient)
    )
    observed = torch.zeros_like(nominal, dtype=torch.bool)
    quantized = torch.full_like(nominal, float(quantized_domain.max))
    loop_steps = maximum_step - minimum_step + 1
    for step_index in range(minimum_step, maximum_step + 1):
        crossing_visible = quotient <= quotient.new_tensor(step_index) + tolerance
        newly_observed = (~observed) & crossing_visible
        quantized = torch.where(
            newly_observed,
            quantized.new_tensor(step_index * step),
            quantized,
        )
        observed = observed | newly_observed
    if not bool(observed.all()):
        raise RuntimeError("clock-driven encoder did not observe every bounded event")
    record_clock_updates("encoder", time_steps=loop_steps, elements=time.numel())

    error = (quantized - time).abs()
    counts = _stats_for(site)
    rounded_events = (error > 0.0).sum().detach()
    absolute_error_sum = error.sum().detach()
    absolute_error_max = error.max().detach()
    counts.events += time.numel()
    counts.rounded_events = (
        rounded_events
        if counts.rounded_events is None
        else counts.rounded_events + rounded_events
    )
    counts.absolute_error_sum = (
        absolute_error_sum
        if counts.absolute_error_sum is None
        else counts.absolute_error_sum + absolute_error_sum
    )
    counts.absolute_error_max = (
        absolute_error_max
        if counts.absolute_error_max is None
        else torch.maximum(counts.absolute_error_max, absolute_error_max)
    )
    counts.windows += 1
    window_steps = maximum_step - minimum_step
    counts.minimum_window_steps = min(counts.minimum_window_steps, window_steps)
    counts.maximum_window_steps = max(counts.maximum_window_steps, window_steps)
    return quantized, quantized_domain


def causal_clock_time(value: Tensor | float) -> Tensor | float:
    """Round an event or deadline to the first non-earlier clock edge."""

    config = get_clock_driven()
    if not config.enabled:
        return value
    step = config.time_step
    if isinstance(value, Tensor):
        if not torch.is_floating_point(value):
            raise TypeError("clock time tensor must be floating-point")
        quotient = value / value.new_tensor(step)
        nearest = torch.round(quotient)
        tolerance = 8.0 * torch.finfo(value.dtype).eps * torch.maximum(
            quotient.abs(), torch.ones_like(quotient)
        )
        indices = torch.where(
            (quotient - nearest).abs() <= tolerance,
            nearest,
            torch.ceil(quotient),
        )
        return indices * value.new_tensor(step)
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("clock time must be finite")
    return _scalar_ceil_steps(normalized, step) * step


def clock_step_indices(value: Tensor) -> Tensor:
    """Convert a clock-aligned signed duration into integer-valued indices."""

    config = get_clock_driven()
    if not config.enabled:
        raise RuntimeError("clock_step_indices requires clock-driven execution")
    quotient = value / value.new_tensor(config.time_step)
    nearest = torch.round(quotient)
    # An explicit PWM accumulator adds the same floating-point step once per
    # edge, so its roundoff can grow quadratically in normalized step count.
    # Admit that bounded arithmetic drift but never a quarter-step displacement.
    magnitude = torch.maximum(quotient.abs(), torch.ones_like(quotient))
    accumulated_roundoff = (
        64.0 * torch.finfo(value.dtype).eps * magnitude.square()
    )
    tolerance = torch.minimum(
        accumulated_roundoff,
        quotient.new_full((), 0.25),
    )
    error = (quotient - nearest).abs()
    violations = error > tolerance
    if bool(violations.any()):
        flat_index = int(violations.reshape(-1).nonzero()[0].item())
        flat_value = value.reshape(-1)[flat_index].item()
        flat_quotient = quotient.reshape(-1)[flat_index].item()
        flat_error = error.reshape(-1)[flat_index].item()
        flat_tolerance = tolerance.reshape(-1)[flat_index].item()
        raise ValueError(
            "clock-driven duration is not aligned to the global time step: "
            f"value={flat_value!r}, quotient={flat_quotient!r}, "
            f"error={flat_error!r}, tolerance={flat_tolerance!r}"
        )
    return nearest


def clocked_difference(value: Tensor, reference: Tensor | float) -> Tensor:
    """Subtract two times on the clock grid without losing their integer steps."""

    config = get_clock_driven()
    if not config.enabled:
        return value - reference
    reference_tensor = (
        reference.to(dtype=value.dtype, device=value.device)
        if isinstance(reference, Tensor)
        else value.new_tensor(reference)
    )
    value_steps = clock_step_indices(value).to(dtype=torch.int64)
    reference_steps = clock_step_indices(reference_tensor).to(dtype=torch.int64)
    difference_steps = value_steps - reference_steps
    return difference_steps.to(dtype=value.dtype) * value.new_tensor(config.time_step)


def clocked_exponential(value: Tensor, *, tau: float) -> Tensor:
    """Evaluate repeated per-step exponential state updates for signed time."""

    config = get_clock_driven()
    if not config.enabled:
        return torch.exp(value / tau)
    steps = clock_step_indices(value).to(dtype=torch.int64)
    maximum_steps = int(steps.abs().max().item()) if steps.numel() else 0
    growth = value.new_tensor(math.exp(config.time_step / tau))
    decay = value.new_tensor(math.exp(-config.time_step / tau))
    state = torch.ones_like(value)
    for step_index in range(maximum_steps):
        state = torch.where(steps > step_index, state * growth, state)
        state = torch.where(steps < -step_index, state * decay, state)
    record_clock_updates(
        "exponential",
        time_steps=maximum_steps,
        elements=value.numel(),
    )
    return state
