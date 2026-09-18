import torch
from jaxtyping import Float, Int
import math
from math import log, exp, isfinite
from numbers import Real
from .clock import (
    causal_clock_time,
    clock_time_step,
    clocked_difference,
    clock_step_indices,
    get_clock_driven,
    record_clock_updates,
)
from .types import ClosedBounds, PotentialBounds, SpikeSample, TimeBounds, check_domain


def _clock_time_bounds(
    observation_deadline: float,
    *events: torch.Tensor | float | SpikeSample,
    time_bounds: TimeBounds | None,
) -> TimeBounds:
    """Resolve the one declared code window shared by a clocked readout."""

    if time_bounds is not None and not isinstance(time_bounds, TimeBounds):
        raise TypeError("time_bounds must be TimeBounds")
    sample_domains = {
        event.domain for event in events if isinstance(event, SpikeSample)
    }
    if len(sample_domains) > 1:
        raise ValueError("clocked readout events require one shared time window")
    if time_bounds is None and sample_domains:
        time_bounds = next(iter(sample_domains))
    if time_bounds is None:
        time_bounds = TimeBounds(0.0, observation_deadline)
    config = get_clock_driven()
    if config.enabled and config.time_steps_per_window and not math.isclose(
        float(time_bounds.max),
        observation_deadline,
        rel_tol=0.0,
        abs_tol=8.0 * math.ulp(max(abs(observation_deadline), 1.0)),
    ):
        raise ValueError("time window maximum must equal observation_deadline")
    return time_bounds


def signed_pulse_width_duration(
    t_A: torch.Tensor | float | SpikeSample,
    t_B: torch.Tensor | float | SpikeSample,
    *,
    observation_deadline: float,
    time_bounds: TimeBounds | None = None,
) -> torch.Tensor | float:
    """Return two causal event-to-deadline accumulators after recombination.

    Continuous execution uses the algebraic duration. Clock-driven execution
    advances both rail accumulators once per clock step and therefore retains an
    explicit sequential simulation even in optimized affine and attention kernels.
    """

    config = get_clock_driven()
    if not config.enabled:
        deadline = observation_deadline
        time_A = t_A.time if isinstance(t_A, SpikeSample) else t_A
        time_B = t_B.time if isinstance(t_B, SpikeSample) else t_B
        raw_duration_A = deadline - time_A
        raw_duration_B = deadline - time_B
        if isinstance(raw_duration_A, torch.Tensor):
            duration_A = raw_duration_A.clamp_min(0.0)
            if isinstance(t_A, SpikeSample):
                duration_A = torch.where(
                    t_A.fired, duration_A, torch.zeros_like(duration_A)
                )
        else:
            duration_A = max(raw_duration_A, 0.0)
        if isinstance(raw_duration_B, torch.Tensor):
            duration_B = raw_duration_B.clamp_min(0.0)
            if isinstance(t_B, SpikeSample):
                duration_B = torch.where(
                    t_B.fired, duration_B, torch.zeros_like(duration_B)
                )
        else:
            duration_B = max(raw_duration_B, 0.0)
        return duration_A - duration_B

    resolved_bounds = _clock_time_bounds(
        observation_deadline,
        t_A,
        t_B,
        time_bounds=time_bounds,
    )
    step = clock_time_step(resolved_bounds)
    origin = float(resolved_bounds.min) if config.time_steps_per_window else 0.0
    deadline = float(
        causal_clock_time(observation_deadline, time_bounds=resolved_bounds)
    )
    time_A = t_A.time if isinstance(t_A, SpikeSample) else t_A
    time_B = t_B.time if isinstance(t_B, SpikeSample) else t_B
    time_A = causal_clock_time(time_A, time_bounds=resolved_bounds)
    time_B = causal_clock_time(time_B, time_bounds=resolved_bounds)
    if not isinstance(time_A, torch.Tensor) and not isinstance(time_B, torch.Tensor):
        deadline_step = round((deadline - origin) / step)
        event_a_step = round((float(time_A) - origin) / step)
        event_b_step = round((float(time_B) - origin) / step)
        first_step = min(0, event_a_step, event_b_step)
        accumulator_a = 0.0
        accumulator_b = 0.0
        for step_index in range(first_step, deadline_step):
            if step_index >= event_a_step:
                accumulator_a += step
            if step_index >= event_b_step:
                accumulator_b += step
        loop_steps = max(deadline_step - first_step, 0)
        record_clock_updates("pwm", time_steps=loop_steps, elements=2)
        return (
            round(accumulator_a / step)
            - round(accumulator_b / step)
        ) * step

    reference = time_A if isinstance(time_A, torch.Tensor) else time_B
    if not isinstance(reference, torch.Tensor):
        raise TypeError("clock-driven PWM requires a tensor or scalar event")
    tensor_A = time_A if isinstance(time_A, torch.Tensor) else reference.new_tensor(time_A)
    tensor_B = time_B if isinstance(time_B, torch.Tensor) else reference.new_tensor(time_B)
    tensor_A, tensor_B = torch.broadcast_tensors(tensor_A, tensor_B)
    steps_A = clock_step_indices(
        tensor_A, time_step=step, origin=origin
    ).to(dtype=torch.int64)
    steps_B = clock_step_indices(
        tensor_B, time_step=step, origin=origin
    ).to(dtype=torch.int64)
    fired_A = (
        torch.broadcast_to(t_A.fired, tensor_A.shape)
        if isinstance(t_A, SpikeSample)
        else torch.ones_like(tensor_A, dtype=torch.bool)
    )
    fired_B = (
        torch.broadcast_to(t_B.fired, tensor_B.shape)
        if isinstance(t_B, SpikeSample)
        else torch.ones_like(tensor_B, dtype=torch.bool)
    )
    deadline_step = round((deadline - origin) / step)
    first_step = min(
        0,
        int(steps_A.min().item()) if steps_A.numel() else 0,
        int(steps_B.min().item()) if steps_B.numel() else 0,
    )
    accumulator_A = torch.zeros_like(tensor_A)
    accumulator_B = torch.zeros_like(tensor_B)
    step_value = tensor_A.new_tensor(step)
    for step_index in range(first_step, deadline_step):
        accumulator_A = accumulator_A + ((steps_A <= step_index) & fired_A).to(
            tensor_A.dtype
        ) * step_value
        accumulator_B = accumulator_B + ((steps_B <= step_index) & fired_B).to(
            tensor_B.dtype
        ) * step_value
    loop_steps = max(deadline_step - first_step, 0)
    record_clock_updates(
        "pwm",
        time_steps=loop_steps,
        elements=2 * tensor_A.numel(),
    )
    duration_bounds = TimeBounds(0.0, float(resolved_bounds.range))
    return clocked_difference(
        accumulator_A,
        accumulator_B,
        time_bounds=duration_bounds,
    )


def pulse_width_duration(
    t_event: torch.Tensor | float,
    *,
    observation_deadline: float,
    time_bounds: TimeBounds | None = None,
) -> torch.Tensor | float:
    """Return one event-to-deadline duration under the active execution mode."""

    config = get_clock_driven()
    if not config.enabled:
        raw_duration = observation_deadline - t_event
        return (
            raw_duration.clamp_min(0.0)
            if isinstance(raw_duration, torch.Tensor)
            else max(raw_duration, 0.0)
        )
    resolved_bounds = _clock_time_bounds(
        observation_deadline,
        t_event,
        time_bounds=time_bounds,
    )
    step = clock_time_step(resolved_bounds)
    origin = float(resolved_bounds.min) if config.time_steps_per_window else 0.0
    deadline = float(
        causal_clock_time(observation_deadline, time_bounds=resolved_bounds)
    )
    aligned_event = causal_clock_time(t_event, time_bounds=resolved_bounds)
    deadline_step = round((deadline - origin) / step)
    if isinstance(aligned_event, torch.Tensor):
        event_steps = clock_step_indices(
            aligned_event,
            time_step=step,
            origin=origin,
        ).to(dtype=torch.int64)
        first_step = min(
            0,
            int(event_steps.min().item()) if event_steps.numel() else 0,
        )
        accumulator = torch.zeros_like(aligned_event)
        step_value = accumulator.new_tensor(step)
        for step_index in range(first_step, deadline_step):
            accumulator = accumulator + (event_steps <= step_index).to(
                accumulator.dtype
            ) * step_value
        loop_steps = max(deadline_step - first_step, 0)
        record_clock_updates(
            "pwm",
            time_steps=loop_steps,
            elements=aligned_event.numel(),
        )
        return clocked_difference(
            accumulator,
            0.0,
            time_bounds=TimeBounds(0.0, float(resolved_bounds.range)),
        )

    event_step = round((float(aligned_event) - origin) / step)
    first_step = min(0, event_step)
    accumulator = 0.0
    for step_index in range(first_step, deadline_step):
        if step_index >= event_step:
            accumulator += step
    record_clock_updates(
        "pwm",
        time_steps=max(deadline_step - first_step, 0),
        elements=1,
    )
    return round(accumulator / step) * step


@check_domain
def unsigned_pulse_width_modulation_operator(
    t_event: torch.Tensor | float,
    domain_t_event: TimeBounds | float,
    V: torch.Tensor,
    domain_V: PotentialBounds,
    *,
    observation_deadline: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Integrate one drive from a delivered event to a fixed future deadline.

    This function is the unsigned physical PWM primitive. One event starts a causal
    rail and the configured observation deadline terminates it, so its duration is
    always non-negative and no event-order detector is required. Signed differences
    are formed by evaluating two such rails against the same deadline and subtracting
    them in :func:`signed_pulse_width_modulation_operator`.

    Args:
        t_event: Scalar or tensor-valued event time that starts this causal rail.
        domain_t_event: Declared event-time interval, or an exact scalar time.
        V: Tensor-valued constant drive integrated while the rail is active.
        domain_V: Declared potential bounds of the integration drive.
        observation_deadline: Fixed scalar time at or after the declared event
            interval maximum, shared by every rail in one signed PWM readout.

    Returns:
        The causal PWM readout ``V * (observation_deadline - t_event)`` and a
        static interval derived only from the declared event, deadline, and drive.

    Raises:
        TypeError: If ``observation_deadline`` is not a real scalar.
        ValueError: If the deadline is non-finite or precedes the declared event
            interval, which would require backward integration on this rail.

    Notes:
        This tensor-only primitive does not yet interpret ``SpikeSample`` delivery
        masks. Event-aware execution will set a missed rail to reset zero while
        reusing one sampled event pair across both rails of the signed composition.
    """
    # Validate the global reference before tensor arithmetic. A malformed deadline
    # must not silently turn a causal rail into a negative-duration integration.
    if isinstance(observation_deadline, bool) or not isinstance(
        observation_deadline,
        Real,
    ):
        raise TypeError("observation_deadline must be a real scalar")
    raw_deadline = float(observation_deadline)
    readout_bounds = _clock_time_bounds(
        raw_deadline,
        time_bounds=(
            domain_t_event if isinstance(domain_t_event, TimeBounds) else None
        ),
    )
    deadline = float(
        causal_clock_time(raw_deadline, time_bounds=readout_bounds)
    )
    if not isfinite(deadline):
        raise ValueError("observation_deadline must be finite")

    # Normalize an exact event time or a declared event interval to scalar endpoints.
    # These configuration values define the rail independently of the current batch.
    event_min = causal_clock_time(
        domain_t_event
        if isinstance(domain_t_event, (int, float))
        else domain_t_event.min,
        time_bounds=readout_bounds,
    )
    event_max = causal_clock_time(
        domain_t_event
        if isinstance(domain_t_event, (int, float))
        else domain_t_event.max,
        time_bounds=readout_bounds,
    )
    if deadline < float(event_max):
        raise ValueError(
            "observation_deadline must not precede the event-domain maximum"
        )

    # Every declared event occurs no later than the common deadline. Clamp tiny
    # negative roundoff only at zero; physical event ordering is not inspected or
    # used to select a branch anywhere in this unsigned primitive.
    duration = pulse_width_duration(
        t_event,
        observation_deadline=deadline,
        time_bounds=readout_bounds,
    )
    result = V * duration

    # Subtracting event endpoints from one fixed deadline reverses their order. Both
    # duration endpoints are non-negative because validation covered the full domain.
    duration_min = deadline - event_max
    duration_max = deadline - event_min

    # The temporal duration is unsigned but the integrated drive may have either
    # sign, so all drive-duration endpoint products remain necessary.
    output_candidates = (
        domain_V.min * duration_min,
        domain_V.min * duration_max,
        domain_V.max * duration_min,
        domain_V.max * duration_max,
    )
    result_domain = PotentialBounds(
        min(output_candidates),
        max(output_candidates),
    )

    # Return the physical rail value together with its configuration-derived range;
    # neither endpoint depends on extrema observed in this invocation.
    return result, result_domain


@check_domain
def signed_pulse_width_modulation_operator(
    t_A: torch.Tensor | float | SpikeSample,
    domain_t_A: TimeBounds | float,
    t_B: torch.Tensor | float | SpikeSample,
    domain_t_B: TimeBounds | float,
    V: torch.Tensor,
    domain_V: PotentialBounds,
    *,
    observation_deadline: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Recover a signed temporal difference with event-aware deadline readout.

    A physical realization can let each delivered event start an independent rail
    that remains active until one shared future deadline. The tensor implementation
    evaluates the algebraically cancelled expression directly when both inputs are
    ordinary delivered times. When either input is a ``SpikeSample``, it instead
    forms each causal duration explicitly so a missed event contributes reset zero.

    Args:
        t_A: First delivered time or event-aware sample.
        domain_t_A: Declared interval, or exact scalar time, for ``t_A``.
        t_B: Second delivered time or event-aware sample broadcastable with ``t_A``.
        domain_t_B: Declared interval, or exact scalar time, for ``t_B``.
        V: Tensor-valued drive shared by both causal integration rails.
        domain_V: Declared potential bounds of the shared drive.
        observation_deadline: Fixed scalar termination time shared by both rails
            and no earlier than either declared event-domain maximum.

    Returns:
        The recombined signed PWM readout and its static interval over the declared
        drive and signed temporal-difference endpoints.

    Raises:
        TypeError: If the deadline or an event-aware declared domain has an invalid
            type.
        ValueError: If the deadline is invalid, a sample disagrees with its declared
            domain, or a sample does not use the shared deadline as its
            ``TimeBounds.max``.

    Notes:
        This function never samples an event. Existing ``SpikeSample`` objects fan
        out into the two causal durations without consuming Gaussian RNG state.
        With one miss, the delivered event's rail remains visible at the deadline;
        with two misses, both reset contributions cancel to zero.
    """
    # Validate the observation reference before inspecting either event. The same
    # finite scalar is a configuration value for both rails, never a batch statistic.
    if isinstance(observation_deadline, bool) or not isinstance(
        observation_deadline,
        Real,
    ):
        raise TypeError("observation_deadline must be a real scalar")
    deadline = float(observation_deadline)
    if not isfinite(deadline):
        raise ValueError("observation_deadline must be finite")

    declared_time_bounds = [
        domain
        for domain in (domain_t_A, domain_t_B)
        if isinstance(domain, TimeBounds)
    ]
    if (
        get_clock_driven().enabled
        and get_clock_driven().time_steps_per_window
        and len(declared_time_bounds) == 2
        and declared_time_bounds[0] != declared_time_bounds[1]
    ):
        raise ValueError(
            "fixed steps per time window require both PWM events to share one window"
        )
    readout_bounds = _clock_time_bounds(
        deadline,
        t_A,
        t_B,
        time_bounds=(declared_time_bounds[0] if declared_time_bounds else None),
    )
    deadline = float(causal_clock_time(deadline, time_bounds=readout_bounds))

    # Normalize declared endpoints without reading extrema from event tensors. The
    # full declared intervals must precede the common physical observation time.
    a_min = causal_clock_time(
        domain_t_A if isinstance(domain_t_A, (int, float)) else domain_t_A.min,
        time_bounds=readout_bounds,
    )
    a_max = causal_clock_time(
        domain_t_A if isinstance(domain_t_A, (int, float)) else domain_t_A.max,
        time_bounds=readout_bounds,
    )
    b_min = causal_clock_time(
        domain_t_B if isinstance(domain_t_B, (int, float)) else domain_t_B.min,
        time_bounds=readout_bounds,
    )
    b_max = causal_clock_time(
        domain_t_B if isinstance(domain_t_B, (int, float)) else domain_t_B.max,
        time_bounds=readout_bounds,
    )
    if deadline < float(a_max) or deadline < float(b_max):
        raise ValueError(
            "observation_deadline must not precede either event-domain maximum"
        )

    # A SpikeSample carries the code window used during sampling. Its separately
    # declared domain must agree, and TimeBounds.max must be the shared deadline so
    # a finite miss carrier cannot silently redefine the observation time.
    for name, event, declared_domain in (
        ("t_A", t_A, domain_t_A),
        ("t_B", t_B, domain_t_B),
    ):
        if not isinstance(event, SpikeSample):
            continue
        if not isinstance(declared_domain, TimeBounds):
            raise TypeError(f"{name} SpikeSample requires a TimeBounds domain")
        if event.domain != declared_domain:
            raise ValueError(f"{name} SpikeSample domain must match its declared domain")
        if float(event.domain.max) != deadline:
            raise ValueError(
                f"{name} SpikeSample domain maximum must equal observation_deadline"
            )

    # Ordinary tensors already represent delivered events. Evaluate the cancelled
    # expression directly, avoiding deadline-sized intermediates and their redundant
    # subtraction in the common deterministic path.
    signed_duration = signed_pulse_width_duration(
        t_A,
        t_B,
        observation_deadline=deadline,
        time_bounds=readout_bounds,
    )
    result = V * signed_duration

    # Derive the ideal both-event range directly from the signed time difference.
    # Treating the two physical rails as independent intervals would lose their
    # shared deadline cancellation and produce an unnecessarily wider envelope.
    signed_duration_min = b_min - a_max
    signed_duration_max = b_max - a_min
    output_candidates = (
        domain_V.min * signed_duration_min,
        domain_V.min * signed_duration_max,
        domain_V.max * signed_duration_min,
        domain_V.max * signed_duration_max,
    )
    result_domain = PotentialBounds(
        min(output_candidates),
        max(output_candidates),
    )

    # Expose one signed potential and one immutable ideal domain downstream. Future
    # event-aware one-sided misses may produce raw rail values outside this ideal
    # both-event range; output saturation must record those before physical clamping.
    return result, result_domain
