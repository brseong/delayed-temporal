import torch
from jaxtyping import Float, Int
from math import exp, isclose

from .noise import clamp_gaussian_output, get_gaussian_time_noise
from .potential_to_spike import neg_identity_transform
from .types import OpenBounds, PotentialBounds, SpikeSample, TimeBounds, check_domain
from .primitive import pulse_width_modulation_operator

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
    """Apply exponential negative transform to the input potentials to produce spike times.

    Args:
        input_value (Float[torch.Tensor, "*batch dims"]): Input spike times of the neurons.
        domain (TimeBounds): The range of possible values for the input spike times.
        tau_m (float, optional): The time constant for the exponential transform. Defaults to 1.0.

    Raises:
        NotImplementedError: wave approximation is not implemented yet.
    
    Returns:
        tuple[Float[torch.Tensor, "*batch dims"], PotentialBounds]: A tuple containing the transformed spike times and the potential bounds of the output.
        """
    return torch.exp(-(domain.max - input_value) / tau_m), PotentialBounds(exp(-(domain.max - domain.min) / tau_m), 1.0)

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
    """Apply exponential negative transform to the input potentials to produce spike times, and normalize the output to have a maximum of 1.

    Args:
        input_value (Float[torch.Tensor, "*batch dims"]): Input spike times of the neurons.
        domain (TimeBounds): The range of possible values for the input spike times.
        tau_m (float, optional): The time constant for the exponential transform. Defaults to 1.0.

    Raises:
        NotImplementedError: wave approximation is not implemented yet.
        """
    # # result = exp(-(domain.max - input_value) / tau_m)
    # # = exp(-domain.max / tau_m) * exp(input_value / tau_m)
    # # scaling_factor = exp(domain.max / tau_m)
    # result, domain_result = exp_operator(input_value, domain, tau_m=tau_m)
    # scaling_factor = exp(domain.max / tau_m)
    # # out = scaling_factor * result
    # # = exp(domain.max / tau_m) * exp(-domain.max / tau_m) * exp(input_value / tau_m)
    # # = exp(input_value / tau_m)
    # return scaling_factor * result, PotentialBounds(domain_result.min * scaling_factor, domain_result.max * scaling_factor)

    return input_value.exp(), PotentialBounds(exp(domain.min), exp(domain.max)) # To avoid numerical instability


def _gaussian_exponential_difference_operator(
    t_A: torch.Tensor | SpikeSample,
    domain_t_A: TimeBounds,
    t_B: torch.Tensor | SpikeSample,
    domain_t_B: TimeBounds,
    tau_s: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate exponential difference through event-aware physical readout.

    ``t_A`` is the opening event and ``t_B`` is the closing event for a unit
    negative drive, producing the intermediate potential ``t_A - t_B``. Tensor
    inputs represent already delivered events, while ``SpikeSample`` inputs retain
    their sampled delivery masks. The finite intermediate potential is then encoded
    again; if that internal exponential event misses, its response remains at reset
    value zero.

    Args:
        t_A: Opening time tensor or event-aware opening sample.
        domain_t_A: Declared time bounds for the opening input.
        t_B: Closing time tensor or event-aware closing sample.
        domain_t_B: Declared time bounds for the closing input.
        tau_s: Temporal scale retained for the public operator contract.

    Returns:
        The clamped exponential response and its event-aware output rails.

    Raises:
        ValueError: If the two event inputs do not share an observation deadline.
        RuntimeError: If internal event encoding does not return ``SpikeSample``.
    """
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

    # Both events participate in one physical integration interval and therefore
    # must be observed against the same fixed deadline before miss masks are applied.
    if not isclose(
        float(event_A.domain.max),
        float(event_B.domain.max),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "event-aware exponential difference requires a shared observation deadline"
        )

    # A closing miss leaves an opened trajectory integrating until the deadline.
    # An opening miss never starts it, leaving the intermediate potential at reset zero.
    deadline = event_A.time.new_tensor(float(event_A.domain.max))
    stop_time = torch.where(event_B.fired, event_B.time, deadline)
    intermediate = torch.where(
        event_A.fired,
        -(stop_time - event_A.time),
        torch.zeros_like(event_A.time),
    )

    # The ideal PWM rails remain determined by the declared input-time endpoints.
    # Clamp the noisy observation before it is re-encoded into the exponential stage.
    intermediate_domain = PotentialBounds(
        domain_t_A.min - domain_t_B.max,
        domain_t_A.max - domain_t_B.min,
    )
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
    response = torch.exp(exponential_input)

    # A missed internal event never initiates the exponential response. Extend the
    # physical lower rail to reset zero while retaining the ideal delivered maximum.
    response = torch.where(
        internal_event.fired,
        response,
        torch.zeros_like(response),
    )
    response_domain = PotentialBounds(
        0.0,
        exp(float(exponential_input_domain.max)),
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

    # The deterministic PWM stage integrates a unit negative drive, producing the
    # signed intermediate potential p = t_A - t_B with explicit interval bounds.
    # p = -1 * (t_B - t_A) = t_A - t_B
    V_ref = torch.full_like(t_A, fill_value=-1.0)
    p, domain_p = pulse_width_modulation_operator(
        t_A, domain_t_A, t_B, domain_t_B, V_ref, PotentialBounds(-1.0, -1.0)
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
