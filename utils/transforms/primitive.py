import torch
from jaxtyping import Float, Int
from math import log, exp, isfinite
from numbers import Real
from .types import ClosedBounds, PotentialBounds, SpikeSample, TimeBounds, check_domain


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
    deadline = float(observation_deadline)
    if not isfinite(deadline):
        raise ValueError("observation_deadline must be finite")

    # Normalize an exact event time or a declared event interval to scalar endpoints.
    # These configuration values define the rail independently of the current batch.
    event_min = (
        domain_t_event
        if isinstance(domain_t_event, (int, float))
        else domain_t_event.min
    )
    event_max = (
        domain_t_event
        if isinstance(domain_t_event, (int, float))
        else domain_t_event.max
    )
    if deadline < float(event_max):
        raise ValueError(
            "observation_deadline must not precede the event-domain maximum"
        )

    # Every declared event occurs no later than the common deadline. Clamp tiny
    # negative roundoff only at zero; physical event ordering is not inspected or
    # used to select a branch anywhere in this unsigned primitive.
    raw_duration = deadline - t_event
    duration = (
        raw_duration.clamp_min(0.0)
        if isinstance(raw_duration, torch.Tensor)
        else max(raw_duration, 0.0)
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

    # Normalize declared endpoints without reading extrema from event tensors. The
    # full declared intervals must precede the common physical observation time.
    a_min = domain_t_A if isinstance(domain_t_A, (int, float)) else domain_t_A.min
    a_max = domain_t_A if isinstance(domain_t_A, (int, float)) else domain_t_A.max
    b_min = domain_t_B if isinstance(domain_t_B, (int, float)) else domain_t_B.min
    b_max = domain_t_B if isinstance(domain_t_B, (int, float)) else domain_t_B.max
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
    if not isinstance(t_A, SpikeSample) and not isinstance(t_B, SpikeSample):
        result = V * (t_B - t_A)
    else:
        # Event-aware execution exposes each causal rail. A delivered event supplies
        # its non-negative time-to-deadline duration; a missed event never opens that
        # rail and therefore leaves its contribution at reset zero.
        time_A = t_A.time if isinstance(t_A, SpikeSample) else t_A
        time_B = t_B.time if isinstance(t_B, SpikeSample) else t_B
        raw_duration_A = deadline - time_A
        raw_duration_B = deadline - time_B
        duration_A = (
            torch.where(
                t_A.fired,
                raw_duration_A.clamp_min(0.0),
                torch.zeros_like(raw_duration_A),
            )
            if isinstance(t_A, SpikeSample)
            else raw_duration_A.clamp_min(0.0)
            if isinstance(raw_duration_A, torch.Tensor)
            else max(raw_duration_A, 0.0)
        )
        duration_B = (
            torch.where(
                t_B.fired,
                raw_duration_B.clamp_min(0.0),
                torch.zeros_like(raw_duration_B),
            )
            if isinstance(t_B, SpikeSample)
            else raw_duration_B.clamp_min(0.0)
            if isinstance(raw_duration_B, torch.Tensor)
            else max(raw_duration_B, 0.0)
        )

        # Subtract the B rail from the A rail. Both delivered events recover the
        # cancelled expression, one-sided misses retain the other rail with its
        # proper sign, and two misses return reset zero.
        result = V * (duration_A - duration_B)

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
