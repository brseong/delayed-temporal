import torch
from jaxtyping import Float, Int
from math import exp, isclose, isfinite
from numbers import Real

from .noise import clamp_gaussian_output, get_gaussian_time_noise
from .potential_to_spike import neg_identity_transform
from .types import OpenBounds, PotentialBounds, SpikeSample, TimeBounds, check_domain
from .primitive import signed_pulse_width_modulation_operator

@check_domain
def exp_operator(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: TimeBounds,
    *,
    tau_m: float = 1.0,
    **_
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        PotentialBounds]:
    """Decode latency relative to a fixed deadline with exponential decay.

    The response is ``exp(-(domain.max-input_value)/tau_m)`` and therefore lies in
    ``[exp(-domain.range/tau_m), 1]`` for an in-domain carrier. The endpoint interval
    is evaluated in the input tensor's dtype before the payload so a physical time
    constant or code window that cannot retain a positive response is rejected.

    Args:
        input_value: Finite spike-time carrier contained in ``domain``.
        domain: Observation window whose maximum is the fixed readout deadline.
        tau_m: Positive finite membrane time constant controlling exponential decay.

    Raises:
        TypeError: If ``tau_m`` is not a real scalar.
        ValueError: If ``tau_m`` is invalid or the earliest decoded endpoint
            underflows to zero in ``input_value.dtype``.

    Returns:
        The exponentially decoded potential and its dtype-representable rails.
    """
    # Reject invalid physical scales before performing tensor arithmetic. Booleans
    # are excluded explicitly even though Python treats them as integer subclasses.
    if isinstance(tau_m, bool) or not isinstance(tau_m, Real):
        raise TypeError("tau_m must be a real scalar")
    tau_value = float(tau_m)
    if not isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau_m must be finite and positive")

    # Decode the earliest and deadline carriers in the payload dtype and device.
    # A very wide window or very small tau_m can underflow the earliest response to
    # zero even though Python's float would still conceal the target dtype limit.
    endpoint_exponents = input_value.new_tensor(
        [
            -float(domain.range) / tau_value,
            0.0,
        ]
    )
    decoded_endpoints = torch.exp(endpoint_exponents)

    # For an ordered finite window this exponent is never positive and the deadline
    # endpoint is exactly one, so overflow is impossible. Only reject earliest-time
    # underflow that would collapse a delivered response onto reset zero.
    if not bool(decoded_endpoints[0] > 0.0):
        raise ValueError(
            "earliest exponential decay response must remain strictly positive "
            "in the input tensor dtype"
        )

    # Evaluate the payload with the same deadline-relative exponent and return
    # concrete scalar rails for device-independent downstream interval arithmetic.
    response = torch.exp(
        -(float(domain.max) - input_value) / tau_value
    )
    return response, PotentialBounds(
        decoded_endpoints[0].item(),
        decoded_endpoints[1].item(),
    )

@check_domain
def normalized_exp_operator(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: TimeBounds,
    *,
    tau_m: float = 1.0,
    **_
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        PotentialBounds]:
    """Decode a time-domain value through a normalized exponential response.

    The operator computes ``exp(input_value / tau_m)`` and propagates the same
    monotonic transformation over the declared time interval. Endpoint decoding is
    performed in the input tensor's dtype and device before the payload so invalid
    time constants, overflow, or positive-domain underflow fail deterministically.

    Args:
        input_value: Finite time-domain tensor contained in ``domain``.
        domain: Declared interval transformed into output-potential bounds.
        tau_m: Positive finite membrane time constant that scales the exponent.

    Raises:
        TypeError: If ``tau_m`` is not a real scalar.
        ValueError: If ``tau_m`` is non-finite or non-positive, or if either decoded
            endpoint is not finite and strictly positive in ``input_value.dtype``.
    """
    # Reject booleans explicitly even though Python models them as integers. A time
    # constant is a physical real-valued scale, not a feature toggle or tensor field.
    if isinstance(tau_m, bool) or not isinstance(tau_m, Real):
        raise TypeError("tau_m must be a real scalar")
    tau_value = float(tau_m)
    if not isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau_m must be finite and positive")

    # Decode the declared endpoints in the payload's dtype and device first. This
    # detects float16/float32 overflow or underflow that Python's wider float could
    # otherwise hide while advertising unusable potential bounds.
    scaled_endpoints = input_value.new_tensor(
        [float(domain.min) / tau_value, float(domain.max) / tau_value]
    )
    decoded_endpoints = torch.exp(scaled_endpoints)

    # Logarithmic consumers require strictly positive finite rails. Reject zero from
    # exponential underflow as well as infinities from overflow before evaluating the
    # full tensor, whose in-domain values lie monotonically between these endpoints.
    if not bool(
        (
            torch.isfinite(decoded_endpoints)
            & (decoded_endpoints > 0.0)
        ).all()
    ):
        raise ValueError(
            "normalized exponential bounds must be finite and strictly positive "
            "in the input tensor dtype"
        )

    # Apply the same scaled exponential to every payload element and return concrete
    # scalar rails so downstream interval arithmetic remains device-independent.
    result = torch.exp(input_value / tau_value)
    return result, PotentialBounds(
        decoded_endpoints[0].item(),
        decoded_endpoints[1].item(),
    )


def _gaussian_exponential_difference_operator(
    t_A: torch.Tensor | SpikeSample,
    domain_t_A: TimeBounds,
    t_B: torch.Tensor | SpikeSample,
    domain_t_B: TimeBounds,
    tau_s: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate exponential difference through event-aware physical readout.

    ``t_A`` and ``t_B`` supply two causal event-to-deadline rails under a unit
    negative drive, producing the intermediate potential ``t_A - t_B`` when both
    arrive. Tensor inputs represent already delivered events, while ``SpikeSample``
    inputs retain independent delivery masks. The finite intermediate potential is
    then encoded again; if that internal event misses, its response remains at zero.

    Args:
        t_A: First time tensor or event-aware sample.
        domain_t_A: Declared time bounds for the first input.
        t_B: Second time tensor or event-aware sample.
        domain_t_B: Declared time bounds for the second input.
        tau_s: Positive finite scale dividing the decoded temporal difference.

    Returns:
        The clamped exponential response and its event-aware output rails.

    Raises:
        TypeError: If ``tau_s`` is not a real scalar.
        ValueError: If ``tau_s`` is invalid, the event deadlines differ, or decoded
            exponential endpoints are unrepresentable in the carrier dtype.
        RuntimeError: If internal event encoding does not return ``SpikeSample``.
    """
    # Validate before the internal encoder boundary so malformed calls cannot
    # consume the next sample from the run-wide Gaussian generator.
    if isinstance(tau_s, bool) or not isinstance(tau_s, Real):
        raise TypeError("tau_s must be a real scalar")
    tau_value = float(tau_s)
    if not isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau_s must be finite and positive")

    # Plain tensors are deterministic events that have already arrived. Wrap them
    # with all-true masks so the remaining physical readout uses one representation.
    if isinstance(t_A, SpikeSample):
        event_A = t_A
    else:
        time_A = domain_t_A.clamp(t_A)
        event_A = SpikeSample(
            time=time_A,
            domain=domain_t_A,
            fired=torch.ones_like(time_A, dtype=torch.bool),
        )
    if isinstance(t_B, SpikeSample):
        event_B = t_B
    else:
        time_B = domain_t_B.clamp(t_B)
        event_B = SpikeSample(
            time=time_B,
            domain=domain_t_B,
            fired=torch.ones_like(time_B, dtype=torch.bool),
        )

    # Both causal rails participate in one differential readout and therefore must
    # use the same fixed observation deadline before their miss masks are applied.
    if not isclose(
        float(event_A.domain.max),
        float(event_B.domain.max),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "event-aware exponential difference requires a shared observation deadline"
        )

    # The fixed -1 drive is the physical exponential-difference current. Reusing the
    # already sampled events in signed PWM gives -[(T-t_A)-(T-t_B)] = t_A-t_B when
    # both arrive, while each one-sided miss leaves the other causal rail visible.
    intermediate, intermediate_domain = signed_pulse_width_modulation_operator(
        event_A,
        domain_t_A,
        event_B,
        domain_t_B,
        event_A.time.new_tensor(-1.0),
        PotentialBounds(-1.0, -1.0),
        observation_deadline=float(event_A.domain.max),
    )

    # The ideal PWM rails remain determined by the declared input-time endpoints.
    # Clamp the noisy observation before it is re-encoded into the exponential stage.
    intermediate = intermediate_domain.clamp(
        intermediate,
        name="exponential_difference_p",
    )

    # Re-encoding is itself a physical spike operation and consumes the next sample
    # from the shared generator. Its miss mask controls the exponential reset state.
    internal_event = neg_identity_transform(
        intermediate,
        intermediate_domain,
        return_spike_sample=True,
        noise_site="exponential_difference.internal",
    )
    if not isinstance(internal_event, SpikeSample):
        raise RuntimeError(
            "Gaussian exponential-difference encoding must return SpikeSample"
        )

    # Shift the finite encoded carrier by the intermediate upper rail, matching the
    # deterministic normalized exponential composition without decoding infinities.
    internal_time_domain = TimeBounds(0.0, float(intermediate_domain.range))
    exponential_input_domain = PotentialBounds(
        internal_time_domain.min - intermediate_domain.max,
        internal_time_domain.max - intermediate_domain.max,
    )
    exponential_input = torch.clamp(
        internal_event.time - float(intermediate_domain.max),
        min=float(exponential_input_domain.min),
        max=float(exponential_input_domain.max),
    )

    # Apply the physical membrane scale to both declared endpoints before decoding
    # the carrier. This makes log-encoded differences tau_s*log(X/Y) return X/Y
    # instead of the incorrect power (X/Y)**tau_s.
    scaled_endpoints = exponential_input.new_tensor(
        [
            float(exponential_input_domain.min) / tau_value,
            float(exponential_input_domain.max) / tau_value,
        ]
    )
    decoded_endpoints = torch.exp(scaled_endpoints)

    # A delivered exponential must remain finite and strictly positive even though
    # the complete event-aware output domain later adds reset zero for internal misses.
    if not bool(
        (
            torch.isfinite(decoded_endpoints)
            & (decoded_endpoints > 0.0)
        ).all()
    ):
        raise ValueError(
            "Gaussian exponential-difference bounds must be finite and strictly "
            "positive in the carrier tensor dtype"
        )
    response = torch.exp(exponential_input / tau_value)

    # A missed internal event never initiates the exponential response. Extend the
    # physical lower rail to reset zero while retaining the ideal delivered maximum.
    response = torch.where(
        internal_event.fired,
        response,
        torch.zeros_like(response),
    )
    response_domain = PotentialBounds(
        0.0,
        decoded_endpoints[1].item(),
    )

    # Record raw saturation before applying the final output rail clamp used by all
    # downstream potential operators.
    return (
        clamp_gaussian_output(
            response,
            response_domain,
            site="exponential_difference.output",
            name="exponential_difference_result",
        ),
        response_domain,
    )


@check_domain
def exponential_difference_operator(
    t_A: torch.Tensor | SpikeSample,
    domain_t_A: TimeBounds,
    t_B: torch.Tensor | SpikeSample,
    domain_t_B: TimeBounds,
    tau_s: float = 1.0
) -> tuple[torch.Tensor, PotentialBounds]:
    """Decode a temporal difference into an exponential potential.

    The public operator dispatches event-aware execution to the private Gaussian
    helper and otherwise preserves the original tensor composition. The physical
    construction first integrates a unit negative drive to obtain ``t_A - t_B``,
    re-encodes that bounded potential, and applies normalized exponential decoding,
    yielding a response proportional to ``exp(t_B - t_A)``.

    Args:
        t_A: Opening time tensor or Gaussian ``SpikeSample``.
        domain_t_A: Declared bounds of the opening time.
        t_B: Closing time tensor or Gaussian ``SpikeSample``.
        domain_t_B: Declared bounds of the closing time.
        tau_s: Temporal scale forwarded to normalized exponential decoding.

    Returns:
        The exponential-difference response and its propagated potential bounds.

    Raises:
        RuntimeError: If event-aware inputs are supplied while Gaussian timing noise
            is disabled.
    """
    # Event-aware inputs require observation-time miss semantics before ordinary
    # tensor arithmetic, so keep all such behavior inside the private helper.
    if get_gaussian_time_noise().enabled:
        return _gaussian_exponential_difference_operator(
            t_A,
            domain_t_A,
            t_B,
            domain_t_B,
            tau_s,
        )

    # Never discard a delivery mask by treating its finite carrier time as an
    # ordinary tensor after the Gaussian path has been disabled.
    if isinstance(t_A, SpikeSample) or isinstance(t_B, SpikeSample):
        raise RuntimeError(
            "SpikeSample inputs require enabled Gaussian time noise"
        )

    # Both physical rails terminate at one observation time. Deterministic tensors
    # have no miss masks, but accepting mismatched deadlines here would make this
    # path describe a different circuit from the event-aware implementation.
    if not isclose(
        float(domain_t_A.max),
        float(domain_t_B.max),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "deterministic exponential difference requires a shared observation "
            "deadline"
        )

    # Integrate the fixed unit-negative drive on the two causal rails. The signed
    # wrapper evaluates their cancelled delivered-time expression directly:
    # -1 * [(T_obs-t_A) - (T_obs-t_B)] = t_A - t_B.
    p, domain_p = signed_pulse_width_modulation_operator(
        t_A,
        domain_t_A,
        t_B,
        domain_t_B,
        t_A.new_tensor(-1.0),
        PotentialBounds(-1.0, -1.0),
        observation_deadline=float(domain_t_A.max),
    )

    # Re-encode the bounded intermediate potential with the negative-identity map;
    # this is the deterministic counterpart of the helper's internal event stage.
    # s = theta - p, where theta = domain_p.max
    s, domain_s = neg_identity_transform(p, domain_p)

    # Shift out the encoder's fixed upper-bound offset before normalized decoding,
    # preserving the existing finite exponential input and its propagated rails.
    # scaling_factor = exp(-domain_p.max / tau_s) = exp(-theta / tau_s)
    # p' = exp(-(T - s) / tau_s)
    #    = exp(-(T - theta + p) / tau_s)
    #    = exp(-T / tau_s) * exp(theta / tau_s) * exp(-p / tau_s)
    # Subtracting theta before normalized decoding removes the fixed encoder offset.
    domain_s_scaled = PotentialBounds(
        domain_s.min - domain_p.max,
        domain_s.max - domain_p.max,
    )
    result, domain_result = normalized_exp_operator(
        s - domain_p.max,
        domain_s_scaled,
        tau_m=tau_s,
    )
    # result = exp(-p / tau_s) = exp((t_B - t_A) / tau_s)
    return result, domain_result

if __name__ == "__main__":
    # Test normalized_exp_operator
    tau_m = 2.0
    t = torch.tensor([0.0, 1.0, 2.0])
    dt = TimeBounds(0.0, 2.0)
    
    # expected: exp(-(2-t)/2) * exp(2/2) = exp(-1 + t/2) * e = exp(t/2)
    # wait, normalized_reciprocal_exp_operator:
    # res = exp(-(2-t)/2)
    # scaling = exp(2/2) = e^1
    # out = e^1 * exp(-1 + t/2) = exp(t/2)
    
    out, domain_out = normalized_exp_operator(t, dt, tau_m=tau_m)
    print(f"normalized_exp_operator output: {out}")
    print(f"Expected: {torch.exp(t/tau_m)}")
    
    # Test exponential_difference_operator
    t_A = torch.tensor([1.0, 2.0, 3.0])
    t_B = torch.tensor([0.5, 0.5, 0.5])
    dt_A = TimeBounds(0.0, 5.0)
    dt_B = TimeBounds(0.0, 5.0)
    
    # psi_ED(t_A, t_B) proportional to exp((t_B - t_A)/tau_s)
    out_ed, domain_ed = exponential_difference_operator(t_A, dt_A, t_B, dt_B, tau_s=1.0)
    print(f"exponential_difference_operator output: {out_ed}")
    print(f"Expected proportional to: {torch.exp((t_B - t_A)/1.0)}")
    
    ratio = out_ed / torch.exp((t_B - t_A)/1.0)
    print(f"Ratio: {ratio}")
