import torch
from jaxtyping import Float
from math import isnan, log, exp

from utils.transforms import exp_operator

from .noise import clamp_gaussian_output, get_gaussian_time_noise
from .types import PotentialBounds, SpikeSample, TimeBounds, check_domain
from .primitive import pulse_width_modulation_operator
from .potential_to_spike import neg_identity_transform, neg_log_transform
from .spike_to_potential import normalized_exp_operator, exponential_difference_operator


def _gaussian_multiplication_operator(
    V: torch.Tensor,
    domain_V: PotentialBounds,
    encoded_B: torch.Tensor,
    domain_B: PotentialBounds,
    theta: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate multiplication from sampled data and zero-reference events.

    This private implementation owns only the maintained Gaussian path. ``encoded_B``
    must already be clamped to the symmetric identity-encoder domain, and Gaussian
    timing noise must be enabled before entry. The public operator remains responsible
    for input validation, common preprocessing, and selecting this implementation.

    Args:
        V: Potential supplying the constant integration drive.
        domain_V: Declared bounds of the integration drive.
        encoded_B: Pre-clamped operand encoded into the opening event.
        domain_B: Symmetric ``[-theta, theta]`` identity-encoder domain.
        theta: Nominal zero-reference time and multiplication-factor rail.

    Returns:
        The observation-time physical readout clamped to its ideal product rails,
        together with those rails.

    Raises:
        RuntimeError: If an event-aware encoder does not return ``SpikeSample``.
    """
    # Each encoded B element owns an opening event. The returned fired mask, rather
    # than the finite deadline carrier stored in time, controls whether integration starts.
    data_event = neg_identity_transform(
        encoded_B,
        domain_B,
        return_spike_sample=True,
        noise_site="multiplication.data",
    )
    if not isinstance(data_event, SpikeSample):
        raise RuntimeError(
            "Gaussian multiplication encoding must return SpikeSample"
        )

    # Sample one scalar zero-reference event for the entire operator invocation. Its
    # scalar time and fired flag broadcast across every data event without resampling.
    reference_event = neg_identity_transform(
        encoded_B.new_zeros(()),
        domain_B,
        return_spike_sample=True,
        noise_site="multiplication.reference",
    )
    if not isinstance(reference_event, SpikeSample):
        raise RuntimeError(
            "Gaussian multiplication reference must return SpikeSample"
        )

    # A delivered reference closes the active trajectory at its sampled time. A
    # missing reference leaves it active until the inclusive observation deadline.
    deadline = data_event.time.new_tensor(float(data_event.domain.max))
    stop_time = torch.where(
        reference_event.fired,
        reference_event.time,
        deadline,
    )

    # A missing opening event leaves the potential at reset zero. For a delivered
    # opening, preserve the signed physical interval before applying the drive V.
    duration = torch.where(
        data_event.fired,
        stop_time - data_event.time,
        torch.zeros_like(data_event.time),
    )
    result = V * duration

    # Gaussian excursions do not expand the representable product rails. Derive the
    # original ideal envelope from all V-bound and factor-bound endpoint products.
    th_val = float(theta) if isinstance(theta, (int, float)) else float(theta.max())
    result_candidates = (
        domain_V.min * -th_val,
        domain_V.min * th_val,
        domain_V.max * -th_val,
        domain_V.max * th_val,
    )
    result_domain = PotentialBounds(
        min(result_candidates),
        max(result_candidates),
    )

    # Saturation statistics inspect the raw physical readout, then the normal bounded
    # operator contract clamps it before any downstream composition receives it.
    return (
        clamp_gaussian_output(
            result,
            result_domain,
            site="multiplication.output",
            name="multiplication_result",
        ),
        result_domain,
    )


@check_domain
def multiplication_operator(
    V: torch.Tensor, 
    domain_V: PotentialBounds,
    B: torch.Tensor, 
    domain_B: PotentialBounds,
    theta: float
) -> tuple[torch.Tensor, PotentialBounds]:
    """Multiply two potentials through affine TTFS and PWM integration.

    ``B`` is encoded as ``t_B = theta - B`` inside ``[0, 2 * theta]``
    and ``V`` is integrated from that opening event to the zero-reference event at
    nominal time ``theta``. This public entry point performs the common calibrated
    encoding setup, then dispatches either to the private Gaussian event readout or
    to the deterministic analytic PWM primitive.

    In Gaussian mode, a missing data event contributes reset value zero, while a
    missing reference event leaves integration active until the fixed observation
    deadline. Those event-specific details remain isolated in the private helper so
    both paths continue to share one public operator and one bounds contract.

    Args:
        V: Potential supplying the constant integration drive.
        domain_V: Declared bounds of the integration drive.
        B: Potential encoded into the opening spike time.
        domain_B: Declared caller bounds used by input-domain validation.
        theta: Symmetric encoder rail and nominal zero-reference time.

    Returns:
        The physically read multiplication result and its ideal output rails.
    """
    # Multiplication always uses the calibrated symmetric identity-code interval,
    # while the caller-supplied domain has already served input validation.
    encoder_domain_B = PotentialBounds(-theta, theta)

    # Clamp the encoded operand once before dispatch so deterministic and Gaussian
    # implementations receive the exact same nominal potential tensor.
    encoded_B = encoder_domain_B.clamp(B, name="multiplication_B")

    # Keep stochastic sampling and physical missing-event readout behind a private
    # implementation; downstream callers never select a Gaussian-specific API.
    if get_gaussian_time_noise().enabled:
        return _gaussian_multiplication_operator(
            V,
            domain_V,
            encoded_B,
            encoder_domain_B,
            theta,
        )

    # Noise-free execution retains the original analytic PWM path and scalar closing
    # time. Convert theta only for the primitive's scalar temporal-bound contract.
    th_val = float(theta) if isinstance(theta, (int, float)) else float(theta.max())
    t_B, domain_t_B = neg_identity_transform(encoded_B, encoder_domain_B)
    return pulse_width_modulation_operator(
        t_A=t_B, 
        domain_t_A=domain_t_B, 
        t_B=theta, 
        domain_t_B=th_val, 
        V=V, 
        domain_V=domain_V
    )

@check_domain
def scaled_dot_product_function(
    q: torch.Tensor, 
    domain_q: PotentialBounds,
    k: torch.Tensor, 
    domain_k: PotentialBounds,
    theta: float
) -> tuple[torch.Tensor, PotentialBounds]:
    """Scaled dot-product operator (f_SDP)"""
    d_k = q.shape[-1]
    M_val, M_bounds = multiplication_operator(q, domain_q, k, domain_k, theta)
    summed_M = torch.sum(M_val, dim=-1)
    
    # Bound multiplication by sum
    sum_min = M_bounds.min * d_k
    sum_max = M_bounds.max * d_k
    
    scale = -(1.0 / (d_k ** 0.5))
    if scale < 0:
        out_min = sum_max * scale
        out_max = sum_min * scale
    else:
        out_min = sum_min * scale
        out_max = sum_max * scale
        
    return scale * summed_M, PotentialBounds(out_min, out_max)


def _gaussian_exponential_function(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_m: float,
    normalized: bool,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate the exponential operator from one sampled input event.

    This private implementation owns the maintained Gaussian path. The input
    potential is encoded through the shared negative-identity boundary, and its
    delivered timestamp drives the same normalized or shifted exponential mapping
    as the deterministic operator. A missed input event never starts that response,
    so the observation-time potential remains at reset value zero.

    Args:
        input_value: Potential tensor to encode into an exponential timing response.
        domain: Declared potential bounds defining the identity-code time window.
        tau_m: Exponential membrane time constant used by the selected mapping.
        normalized: Select the normalized exponential composition when true.

    Returns:
        The finite observation-time response clamped to its Gaussian-path rails,
        together with those rails.

    Raises:
        RuntimeError: If the event-aware encoder does not return ``SpikeSample``.
    """
    # Request one event per input element from the common encoder boundary so its
    # sampled time and fired mask originate from the same Gaussian draw.
    event = neg_identity_transform(
        input_value,
        domain,
        return_spike_sample=True,
        noise_site="exponential.input",
    )
    if not isinstance(event, SpikeSample):
        raise RuntimeError(
            "Gaussian exponential encoding must return SpikeSample"
        )

    # The sampler already stores early arrivals at the window start and misses at
    # the finite deadline. Clamp defensively to the declared carrier interval before
    # applying an exponentially sensitive readout.
    delivered_time = torch.clamp(
        event.time,
        min=float(event.domain.min),
        max=float(event.domain.max),
    )

    # Preserve the deterministic normalized composition, including its fixed scale
    # removal, while extending the lower physical rail to the reset value zero.
    if normalized:
        scaling_factor = exp(-float(domain.max) / tau_m)
        response = scaling_factor * torch.exp(delivered_time)
        response_max = scaling_factor * exp(float(event.domain.max))
    else:
        # The unnormalized path centers the code window before applying its explicit
        # membrane time constant, matching the existing noise-free tensor mapping.
        shift_val = float(event.domain.range) / 2.0
        response = torch.exp((delivered_time - shift_val) / tau_m)
        response_max = exp(
            (float(event.domain.max) - shift_val) / tau_m
        )

    # A missed opening event leaves the exponential membrane at reset; its stored
    # deadline is only a finite carrier and must not be decoded as an arriving spike.
    response = torch.where(
        event.fired,
        response,
        torch.zeros_like(response),
    )
    response_domain = PotentialBounds(0.0, response_max)

    # Count any raw rail excursions before clamping the finite physical readout for
    # downstream operators. Exact endpoint values remain valid representations.
    return (
        clamp_gaussian_output(
            response,
            response_domain,
            site="exponential.output",
            name="exponential_result",
        ),
        response_domain,
    )


@check_domain
def exponential_function(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_m: float = 1.0,
    normalized: bool = True,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Apply the composed exponential-potential operator.

    The public entry point selects either the event-aware Gaussian implementation or
    the original deterministic composition of negative-identity encoding and
    exponential temporal decoding. Both modes retain the same input contract and
    ``normalized`` selection; only the Gaussian helper interprets delivery masks and
    exposes reset-valued missed events.

    Args:
        input_value: Bounded potential tensor to transform.
        domain: Declared input-potential interval.
        tau_m: Membrane time constant used by the exponential mapping.
        normalized: Select the scaled normalized composition when true, or the
            centered direct exponential when false.

    Returns:
        The transformed potential tensor and the bounds declared by the selected
        physical or deterministic path.
    """
    # Keep event sampling, reset behavior, and noisy-output statistics isolated in
    # the private implementation while preserving one public operator API.
    if get_gaussian_time_noise().enabled:
        return _gaussian_exponential_function(
            input_value,
            domain,
            tau_m=tau_m,
            normalized=normalized,
        )

    # The deterministic path first applies the negative-identity encoder, mapping
    # the potential interval onto its equally wide time-code interval.
    t_out, tb_out = neg_identity_transform(input_value, domain)

    # Normalized decoding uses the existing temporal primitive and removes its fixed
    # domain-dependent scale without altering the propagated endpoint bounds.
    if normalized:
        v_out, domain_v_out = normalized_exp_operator(t_out, tb_out, tau_m=tau_m)
        scaling_factor = exp(-domain.max / tau_m)
        return scaling_factor * v_out, PotentialBounds(
            domain_v_out.min * scaling_factor,
            domain_v_out.max * scaling_factor,
        )

    # The unnormalized form centers the finite code window before exponentiation;
    # derive its output interval from the same two temporal endpoints.
    shift_val = tb_out.range / 2
    return torch.exp((t_out - shift_val) / tau_m), PotentialBounds(
        exp((tb_out.min - shift_val) / tau_m),
        exp((tb_out.max - shift_val) / tau_m),
    )


def _gaussian_softmin_function(
    input_value: torch.Tensor,
    domain: PotentialBounds,
    *,
    tau_s: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate softmin through event-aware exponential and division operators.

    Gaussian exponential misses physically produce reset value zero, but the
    following negative-log encoder requires a strictly positive finite domain. This
    helper therefore uses the ideal unnormalized exponential minimum as the shared
    positive floor for both numerator values and their reduction. The ordinary
    bounds clamp then maps a reset zero onto that declared finite representational
    floor before event-aware division.

    Args:
        input_value: Bounded score tensor normalized along its final dimension.
        domain: Declared score interval used by the exponential encoder.
        tau_s: Shared exponential and logarithmic temporal scale.

    Returns:
        The finite event-aware softmin weights and their propagated ratio rails.

    Raises:
        TypeError: If ``input_value`` is not a floating-point tensor.
        ValueError: If the normalization dimension is empty or the computed positive
            floor is not representable for the input dtype.
    """
    # Validate structural requirements before invoking a stochastic sub-operator so
    # a rejected softmin call cannot consume or advance the configured generator.
    if not torch.is_floating_point(input_value):
        raise TypeError("softmin input must be a floating-point tensor")
    element_count = input_value.size(-1)
    if element_count == 0:
        raise ValueError("softmin requires a non-empty final dimension")

    # First produce one event-aware unnormalized exponential per score. A missed
    # exponential input remains exactly zero at this physical stage.
    exp_value, exp_domain = exponential_function(
        input_value,
        domain,
        tau_m=tau_s,
        normalized=False,
    )

    # Reduce over the same final dimension used by attention normalization. Keep the
    # dimension so the independently sampled denominator event broadcasts on decode.
    sum_exp_value = exp_value.sum(dim=-1, keepdim=True)

    # The Gaussian exponential domain includes reset zero, which cannot enter a log
    # encoder. Recover the ideal delivered minimum and keep it above dtype underflow.
    ideal_exp_min = exp(-float(domain.range) / (2.0 * tau_s))
    dtype_floor = float(torch.finfo(input_value.dtype).tiny)
    positive_floor = max(ideal_exp_min, dtype_floor)
    if not torch.isfinite(input_value.new_tensor(positive_floor)):
        raise ValueError("softmin positive floor must be finite and representable")

    # One shared domain must contain both each numerator and the reduced denominator.
    # Its upper rail grows with the reduction, while its lower rail stays at the
    # single-element floor so low or missed numerator values are representable.
    joint_domain = PotentialBounds(
        positive_floor,
        exp_domain.max * element_count,
    )

    # Event-aware division performs the actual floor clamp, samples synchronized log
    # events, applies miss-aware exponential difference, and returns finite weights.
    return division_function(
        X=exp_value,
        Y=sum_exp_value,
        joint_domain=joint_domain,
        tau_s=tau_s,
    )


@check_domain
def softmin_function(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Normalize scores with the composed softmin operator.

    The construction exponentiates negated-score timing responses, reduces them
    along the final dimension, and divides each response by that sum. Gaussian mode
    delegates to the private implementation that reconciles exponential reset zero
    with the finite positive log domain; deterministic mode retains the original
    three-stage tensor composition.

    Args:
        input_value: Bounded score tensor normalized along its final dimension.
        domain: Declared score interval.
        tau_s: Shared exponential and logarithmic temporal scale.

    Returns:
        The softmin weights and their propagated potential bounds.

    According to Lemma 4.3, the normalization is composed as
    ``w_softmin,ij ≈ f_DIV(s_ij, sum_k s_ik)`` after exponentiating scores.
    """
    # Keep the reset-to-positive-floor policy and event-aware sub-operator sequence
    # isolated from the deterministic composition behind one public API.
    if get_gaussian_time_noise().enabled:
        return _gaussian_softmin_function(
            input_value,
            domain,
            tau_s=tau_s,
        )

    # 1. Exponential potential transformation: exp_v = exp(-s_ij / tau_s)
    exp_v, exp_domain = exponential_function(
        input_value,
        domain,
        tau_m=tau_s,
        normalized=False,
    )

    # 2. Sum of exponentiated scores: sum_k exp(s_ik / tau_s)
    sumexp_v = exp_v.sum(dim=-1, keepdim=True)
    N = input_value.size(-1)

    # Propagate the original worst-case reduction bounds. The deterministic lower
    # rail is positive, so this path does not require the Gaussian helper's floor.
    # sumexp_domain = PotentialBounds(exp_domain.min, exp_domain.max * N)
    # Too high max bound causes numerical instability in division, so use the
    # original mathematically equivalent worst-case reduction bound.
    sumexp_domain = PotentialBounds(
        exp_domain.min * N,
        exp_domain.max * N,
    )

    # 3. Apply the Division Operator: f_DIV(exp_v, sumexp_v)
    return division_function(
        X=exp_v,
        Y=sumexp_v,
        joint_domain=sumexp_domain,
        tau_s=tau_s,
    )


def _gaussian_division_function(
    X: torch.Tensor,
    Y: torch.Tensor,
    joint_domain: PotentialBounds,
    tau_s: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate division from independently sampled logarithmic events.

    This private implementation assumes ``X`` and ``Y`` have already been clamped
    to the same strictly positive domain and satisfy ``X <= Y``. Encoding both
    operands against that shared domain makes their logarithmic offsets cancel. The
    resulting delivery masks are preserved through event-aware exponential
    difference, which owns the physical opening, closing, and internal-event miss
    behavior.

    Args:
        X: Prevalidated numerator tensor.
        Y: Prevalidated denominator tensor.
        joint_domain: Shared positive bounds used by both logarithmic encoders.
        tau_s: Common logarithmic time scale.

    Returns:
        The event-aware ratio readout and its propagated output rails.

    Raises:
        RuntimeError: If either decorated logarithmic encoder fails to return a
            ``SpikeSample`` while Gaussian timing noise is enabled.
    """
    # Sample the numerator event through the shared log encoder. Its fired mask is
    # the opening-event state consumed by exponential difference.
    numerator_event = neg_log_transform(
        X,
        joint_domain,
        tau_s=tau_s,
        return_spike_sample=True,
        noise_site="division.numerator",
    )

    # Draw the denominator independently from the same generator stream and domain;
    # using one domain is what cancels the two fixed logarithmic timing offsets.
    denominator_event = neg_log_transform(
        Y,
        joint_domain,
        tau_s=tau_s,
        return_spike_sample=True,
        noise_site="division.denominator",
    )

    # Fail at this boundary if a decorated encoder violates the event-aware contract
    # rather than allowing tuple unpacking to discard a delivery mask downstream.
    if not isinstance(numerator_event, SpikeSample) or not isinstance(
        denominator_event,
        SpikeSample,
    ):
        raise RuntimeError("Gaussian division encoders must return SpikeSample")

    # Forward both complete event records. The exponential-difference dispatcher
    # applies opening/closing miss physics, re-encodes the finite intermediate state,
    # and records its final output saturation without a division-specific fallback.
    return exponential_difference_operator(
        numerator_event,
        numerator_event.domain,
        denominator_event,
        denominator_event.domain,
        tau_s=tau_s,
    )


@check_domain
def division_function(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    joint_domain: PotentialBounds,
    tau_s: float
) -> tuple[torch.Tensor, PotentialBounds]:
    """Divide two positive potentials through log timing and exponential difference.

    Both operands are projected into one shared domain so the fixed offset of their
    negative-log encodings cancels. This public entry point performs common input
    preparation and ordering validation, then selects either the private Gaussian
    event implementation or the original deterministic tensor composition.

    Args:
        X: Numerator tensor.
        Y: Denominator tensor.
        joint_domain: Shared strictly positive domain for both operands.
        tau_s: Common logarithmic encoding and temporal decoding scale.

    Returns:
        The ratio-like exponential-difference response and its propagated bounds.

    Raises:
        AssertionError: If any clamped numerator exceeds its denominator.
    """
    # Apply the identical shared rails before either execution path. Besides handling
    # floating-point boundary drift, this preserves the synchronized log offset.
    X = joint_domain.clamp(X, name="division_X")
    Y = joint_domain.clamp(Y, name="division_Y")

    # The current operator contract restricts the represented ratio to X/Y <= 1;
    # reject invalid ordering before sampling so failed calls consume no RNG state.
    assert torch.all(X <= Y), (
        "For division to be valid, each element of X must be less than or equal "
        "to the corresponding element of Y."
    )

    # Dispatch only after all common preprocessing and validation. The private path
    # retains both delivery masks through its physical exponential-difference readout.
    if get_gaussian_time_noise().enabled:
        return _gaussian_division_function(X, Y, joint_domain, tau_s)

    # Both transforms must use the same domain to synchronize their fixed offsets.
    # t_X = -\tau_s * log(X/T) = -\tau_s * (log(X) - log(T))
    # t_Y = -\tau_s * log(Y/T) = -\tau_s * (log(Y) - log(T))
    t_X, tb_X = neg_log_transform(X, joint_domain, tau_s=tau_s)
    t_Y, tb_Y = neg_log_transform(Y, joint_domain, tau_s=tau_s)

    # Keep deterministic latencies inside their analytic interval before temporal
    # subtraction, preventing endpoint roundoff from expanding the ratio envelope.
    t_X = t_X.clamp(min=tb_X.min, max=tb_X.max)
    t_Y = t_Y.clamp(min=tb_Y.min, max=tb_Y.max)

    # f_DIV(X, Y) = exp((t_Y - t_X) / tau_s)
    # = exp(-t_X / tau_s) * exp(t_Y / tau_s)
    # = exp(log(X/T)) * exp(-log(Y/T)) = X/Y
    result, domain_result = exponential_difference_operator(
        t_X,
        tb_X,
        t_Y,
        tb_Y,
        tau_s=tau_s,
    )
    return result, domain_result

@check_domain
def gelu_approximation(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    theta: float = 400.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Approximate GELU activation using spiking operators (tanh form).

    Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """
    input_clamped = domain.clamp(input_value, name="gelu_x")

    # x^2 and x^3 via f_M
    x2, domain_x2 = multiplication_operator(input_clamped, domain, input_clamped, domain, theta)
    x3, domain_x3 = multiplication_operator(x2, domain_x2, input_clamped, domain, theta)

    # 0.044715 * x^3
    coeff = 0.044715
    coeff_tensor = input_value.new_tensor(coeff).expand_as(input_value)
    coeff_domain = PotentialBounds(coeff, coeff)
    x3_scaled, domain_x3_scaled = multiplication_operator(x3, domain_x3, coeff_tensor, coeff_domain, theta)

    # x + 0.044715 * x^3
    inner = input_clamped + x3_scaled
    inner_domain = PotentialBounds(domain.min + domain_x3_scaled.min, domain.max + domain_x3_scaled.max)

    # sqrt(2/pi) * inner
    scale_const = 0.7978845608028654
    scale_tensor = input_value.new_tensor(scale_const).expand_as(input_value)
    scale_domain = PotentialBounds(scale_const, scale_const)
    tanh_in, tanh_in_domain = multiplication_operator(inner, inner_domain, scale_tensor, scale_domain, theta)

    # tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    tanh_out, tanh_domain = tanh(tanh_in, tanh_in_domain, tau_s=tau_s, theta=theta)

    # 0.5 * (1 + tanh(...))
    one_plus = 1.0 + tanh_out
    one_plus_domain = PotentialBounds(1.0 + tanh_domain.min, 1.0 + tanh_domain.max)
    half = 0.5
    half_tensor = input_value.new_tensor(half).expand_as(input_value)
    half_domain = PotentialBounds(half, half)
    gate, gate_domain = multiplication_operator(one_plus, one_plus_domain, half_tensor, half_domain, theta)

    # x * gate
    gelu_approx, gelu_domain = multiplication_operator(input_clamped, domain, gate, gate_domain, theta)
    return gelu_approx, gelu_domain


@check_domain
def gelu_approximation_sigmoid(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    theta: float = 400.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Approximate GELU activation using spiking operators (sigmoid form)."""
    # Step 1: f_NP(1.702v)
    scale_const = 1.702
    scale_bound = PotentialBounds(scale_const, scale_const)
    scaled_input, _ = multiplication_operator(
        input_value,
        domain,
        input_value.new_tensor(scale_const).expand_as(input_value),
        scale_bound,
        theta,
    )
    scaled_domain = PotentialBounds(scale_const * domain.min, scale_const * domain.max)

    # Stability cap for exp: exp(20) is safe, exp(400) overflows.
    # Since exp(-1.702*v) is used for sigmoid, we only need to worry about v being very negative.
    _STABILITY_CAP = 80.0
    scaled_input_clamped = scaled_input.clamp(min=-_STABILITY_CAP, max=_STABILITY_CAP)
    scaled_domain_clamped = PotentialBounds(
        max(scaled_domain.min, -_STABILITY_CAP),
        min(scaled_domain.max, _STABILITY_CAP),
    )

    # Step 2: f_NE(f_NP(1.702v))
    # Note: exponential_function outputs C * exp(-1.702v)
    neg_exp_out, neg_exp_domain = exponential_function(scaled_input_clamped, scaled_domain_clamped, tau_m=tau_s)

    # Step 3: f_DIV(C, C + f_NE(f_NP(1.702v)))
    # This mathematically equals 1 / (1 + exp(-1.702v))
    div_out, div_domain = division_function(
        X=torch.full_like(neg_exp_out, 1.0),
        Y=1.0 + neg_exp_out,
        joint_domain=PotentialBounds(1.0, neg_exp_domain.max + 1.0),
        tau_s=tau_s,
    )

    # Step 4: f_M(v, div_out)
    gelu_approx, gelu_domain = multiplication_operator(
        domain.clamp(input_value, name="gelu_x"),
        domain,
        div_domain.clamp(div_out),
        div_domain,
        theta=theta,
    )

    return gelu_approx, gelu_domain


@check_domain
def tanh(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    theta: float = 400.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Approximate tanh activation using spiking operators.
    
    According to Lemma 4.4 (Derivation of tanh Approximation) in the paper:
    f_tanh(v) := 2 * f_Div(1,1+f_Exp (-2v)) - 1
    """
    # Step 1: f_NP(-2v)
    scale_const = 2.0
    scale_bound = PotentialBounds(scale_const, scale_const)
    scaled_input, _ = multiplication_operator(
        input_value, domain,
        input_value.new_tensor(scale_const).expand_as(input_value), scale_bound,
        theta)
    scaled_domain = PotentialBounds(scale_const * domain.min, scale_const * domain.max)
    
    # Stability cap for exp: exp(20) is safe, exp(400) overflows.
    # Since exp(-1.702*v) is used for sigmoid, we only need to worry about v being very negative.
    _STABILITY_CAP = 80.0
    scaled_input_clamped = scaled_input.clamp(min=-_STABILITY_CAP, max=_STABILITY_CAP)
    scaled_domain_clamped = PotentialBounds(max(scaled_domain.min, -_STABILITY_CAP), min(scaled_domain.max, _STABILITY_CAP))

    # Step 2: f_NE(f_NP(1.702v))
    # Note: exponential_function outputs C * exp(-1.702v)
    neg_exp_out, neg_exp_domain = exponential_function(scaled_input_clamped, scaled_domain_clamped, tau_m=tau_s)
    
    # Step 3: f_DIV(C, C + f_NE(f_NP(1.702v)))
    # This mathematically equals 1 / (1 + exp(-1.702v))
    div_out, div_domain = division_function(
        X=torch.full_like(neg_exp_out, 1.0), 
        Y=1.0 + neg_exp_out, 
        joint_domain=PotentialBounds(1.0, neg_exp_domain.max + 1.0), 
        tau_s=tau_s
    )
    
    return 2.0 * div_out - 1.0, PotentialBounds(2.0 * div_domain.min - 1.0, 2.0 * div_domain.max - 1.0)


def _gaussian_swiglu_function(
    u: torch.Tensor,
    domain_u: PotentialBounds,
    v: torch.Tensor,
    domain_v: PotentialBounds,
    *,
    beta: float,
    tau_s: float,
    theta: float,
) -> tuple[torch.Tensor, PotentialBounds]:
    """Evaluate SwiGLU through event-aware exponential, division, and products.

    This private implementation follows
    ``v * (u * f_DIV(1, 1 + psi_NE(phi_NP(beta * u))))``. The direct exponential
    input is sampled explicitly because its delivery mask must select between a
    decoded response and reset zero. Division and both multiplication stages then
    reuse their public operators, which dispatch to their own Gaussian physical
    readouts under the same process-wide configuration.

    Args:
        u: Input controlling both the sigmoid-like gate and gated value.
        domain_u: Declared bounds of ``u``.
        v: Second input multiplied with the gated ``u`` value.
        domain_v: Declared bounds of ``v``.
        beta: Scale applied to ``u`` before gate construction.
        tau_s: Temporal scale forwarded to division.
        theta: Symmetric identity-code rail used by multiplication stages.

    Returns:
        The finite event-aware SwiGLU output and its propagated product rails.

    Raises:
        RuntimeError: If direct event-aware encoding does not return ``SpikeSample``.
    """
    # Step 1: Scale u by beta and propagate the same affine endpoint bounds used by
    # the deterministic composition before applying its exponential stability cap.
    scaled_u = beta * u
    scaled_domain_u = PotentialBounds(
        beta * domain_u.min,
        beta * domain_u.max,
    )
    stability_cap = 20.0
    scaled_u_clamped = scaled_u.clamp(
        min=-stability_cap,
        max=stability_cap,
    )
    scaled_domain_u_clamped = PotentialBounds(
        max(scaled_domain_u.min, -stability_cap),
        min(scaled_domain_u.max, stability_cap),
    )

    # Step 2: Apply phi_NP(beta*u) at the shared encoder boundary. One sampled event
    # supplies both the finite carrier time and the delivery decision for psi_NE.
    exponential_event = neg_identity_transform(
        scaled_u_clamped,
        scaled_domain_u_clamped,
        return_spike_sample=True,
        noise_site="swiglu.exponential_input",
    )
    if not isinstance(exponential_event, SpikeSample):
        raise RuntimeError("Gaussian SwiGLU encoding must return SpikeSample")

    # Decode only delivered events. Early samples are already stored at the start,
    # and a missed event leaves this internal exponential response at reset zero.
    delivered_time = torch.clamp(
        exponential_event.time,
        min=float(exponential_event.domain.min),
        max=float(exponential_event.domain.max),
    )
    exp_out = torch.where(
        exponential_event.fired,
        torch.exp(delivered_time),
        torch.zeros_like(delivered_time),
    )
    exp_domain = PotentialBounds(
        0.0,
        exp(float(exponential_event.domain.max)),
    )
    exp_out = clamp_gaussian_output(
        exp_out,
        exp_domain,
        site="swiglu.exponential_output",
        name="swiglu_exponential_result",
    )

    # Step 3: f_DIV(1, 1 + psi_NE(phi_NP(beta*u))) constructs the sigmoid-like gate.
    # The reset-inclusive exponential rail makes one the valid positive lower bound.
    one_plus_exp = 1.0 + exp_out
    one_plus_exp_domain = PotentialBounds(
        1.0 + exp_domain.min,
        1.0 + exp_domain.max,
    )
    sigmoid_out, sigmoid_domain = division_function(
        X=torch.ones_like(one_plus_exp),
        Y=one_plus_exp,
        joint_domain=one_plus_exp_domain,
        tau_s=tau_s,
    )

    # Step 4: psi_M(u, sigmoid) forms the gated Swish value. Its event-aware
    # multiplication applies opening/reference miss physics and output rail logging.
    swish_out, swish_domain = multiplication_operator(
        u,
        domain_u,
        sigmoid_out,
        sigmoid_domain,
        theta=theta,
    )

    # Step 5: psi_M(v, swish) completes v*u*sigmoid using a second independently
    # sampled multiplication call while preserving the propagated potential bounds.
    return multiplication_operator(
        v,
        domain_v,
        swish_out,
        swish_domain,
        theta=theta,
    )


@check_domain
def swiglu_function(
    u: torch.Tensor,
    domain_u: PotentialBounds,
    v: torch.Tensor,
    domain_v: PotentialBounds,
    *,
    beta: float = 1.0,
    tau_s: float = 1.0,
    theta: float = 400.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """SwiGLU activation function using spiking operators.

    The public entry point selects the private event-aware Gaussian implementation
    or the original deterministic five-stage composition while preserving one API,
    one algebraic definition, and the same propagated output contract.
    
    According to Lemma 4.5 (SwiGLU Operator) in the paper:
    f_SwiGLU(u, v) := ψ_M(v, ψ_M(u, f_DIV(1, 1 + ψ_NE(φ_NP(β u)))))
    
    where:
    - ψ_M is multiplication_operator
    - φ_NP is neg_identity_transform (Negative Potential operator)
    - ψ_NE is normalized_exp_operator (Negative Exp-Temporal operator)
    - f_DIV is division_function
    
    Args:
        u: First input potential
        domain_u: Potential bounds for u
        v: Second input potential
        domain_v: Potential bounds for v
        beta: Scaling constant for sigmoid computation (default: 1.0)
        tau_s: Time constant for operators (default: 1.0)
        theta: Parameter for multiplication operator (default: 400.0)
    
    Returns:
        Tuple of (output, output_domain)
    """
    # Keep direct event decoding, miss handling, and nested noisy operators isolated
    # in the private implementation while callers retain this single public surface.
    if get_gaussian_time_noise().enabled:
        return _gaussian_swiglu_function(
            u,
            domain_u,
            v,
            domain_v,
            beta=beta,
            tau_s=tau_s,
            theta=theta,
        )

    # Step 1: Scale u by beta
    scaled_u = beta * u
    scaled_domain_u = PotentialBounds(beta * domain_u.min, beta * domain_u.max)
    
    # Stability cap for exp
    _STABILITY_CAP = 20.0
    scaled_u_clamped = scaled_u.clamp(min=-_STABILITY_CAP, max=_STABILITY_CAP)
    scaled_domain_u_clamped = PotentialBounds(
        max(scaled_domain_u.min, -_STABILITY_CAP),
        min(scaled_domain_u.max, _STABILITY_CAP)
    )
    
    # Step 2: Apply φ_NP (neg_identity_transform) then ψ_NE (normalized_exp_operator)
    t_betau, domain_t_betau = neg_identity_transform(scaled_u_clamped, scaled_domain_u_clamped)
    exp_out, exp_domain = normalized_exp_operator(t_betau, domain_t_betau, tau_m=tau_s)
    
    # Step 3: Compute sigmoid σ(β u) = f_DIV(1, 1 + ψ_NE(φ_NP(β u)))
    one_plus_exp = 1.0 + exp_out
    # The shared division domain must contain both the constant numerator 1 and the
    # denominator 1 + exp_out; using 1 + exp_domain.min would clamp the numerator.
    one_plus_exp_domain = PotentialBounds(1.0, 1.0 + exp_domain.max)
    
    sigmoid_out, sigmoid_domain = division_function(
        X=torch.ones_like(one_plus_exp),
        Y=one_plus_exp,
        joint_domain=one_plus_exp_domain,
        tau_s=tau_s
    )

    # Expand the internal gate rail to the physical reset value zero. This absorbs
    # endpoint roundoff before multiplication and matches the Gaussian gate contract.
    sigmoid_domain = PotentialBounds(0.0, sigmoid_domain.max)
    
    # Step 4: Compute Swish: ψ_M(u, σ(β u)) = u * σ(β u)
    swish_out, swish_domain = multiplication_operator(
        u, domain_u,
        sigmoid_out, sigmoid_domain,
        theta=theta
    )
    
    # Step 5: Final multiplication: ψ_M(v, swish_out) = v * u * σ(β u)
    final_out, final_domain = multiplication_operator(
        v, domain_v,
        swish_out, swish_domain,
        theta=theta
    )
    
    return final_out, final_domain

if __name__ == "__main__":
    # Test for exponential_function and division_function
    tau_s = 1.0
    domain = PotentialBounds(0.1, 10.0)
    
    # 1. Test Exponential Function proportionality to exp(-x)
    x = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
    exp_out, _ = exponential_function(x, domain, tau_m=tau_s)
    expected_exp = torch.exp(-x / tau_s)
    ratios = exp_out / expected_exp
    is_exp_valid = torch.allclose(ratios, ratios[0] * torch.ones_like(ratios))
    print(f"Exponential Function Proportional to exp(-x): {is_exp_valid}")

    # 2. Test Division Function accuracy (X/Y)
    X_val = torch.tensor([1.0, 2.0], dtype=torch.float32)
    Y_val = torch.tensor([2.0, 4.0], dtype=torch.float32)
    div_out, _ = division_function(
        X=X_val, 
        Y=Y_val,
        joint_domain=PotentialBounds(0.1, 15.0),
        tau_s=tau_s
    )
    expected_div = X_val / Y_val
    is_div_valid = torch.allclose(div_out, expected_div, atol=1e-5)
    print(f"Division Function Accurate (X/Y): {is_div_valid}")
    if not is_div_valid:
        print(f"Expected: {expected_div}, Got: {div_out}")

    # 3. Test Softmin Function
    softmin_out, _ = softmin_function(x.unsqueeze(0), domain, tau_s=tau_s)
    expected_softmin = torch.softmax(-x / tau_s, dim=-1)
    is_softmin_valid = torch.allclose(softmin_out, expected_softmin.unsqueeze(0), atol=1e-5)
    print(f"Softmin Function Accurate: {is_softmin_valid}")

    # 4. Test GELU Approximation
    import torch.nn.functional as F
    
    gelu_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    gelu_domain = PotentialBounds(-5.0, 5.0)
    gelu_out, _ = gelu_approximation(gelu_x, gelu_domain, tau_s=tau_s)
    expected_gelu = F.gelu(gelu_x)
    sqrt_2_over_pi = 0.7978845608028654
    expected_gelu_tanh = 0.5 * gelu_x * (1.0 + torch.tanh(sqrt_2_over_pi * (gelu_x + 0.044715 * gelu_x ** 3)))
    
    print(f"GELU Approx Output:   {gelu_out.tolist()}")
    print(f"Expected PyTorch GELU: {expected_gelu.tolist()}")
    print(f"Expected Tanh GELU:    {expected_gelu_tanh.tolist()}")
    
    # As it's an approximation using mathematical substitutions, allow slightly higher tolerance
    is_gelu_formula_valid = torch.allclose(gelu_out, expected_gelu_tanh, atol=2e-2)
    is_gelu_valid = torch.allclose(gelu_out, expected_gelu, atol=2e-2)
    print(f"GELU Formula Match: {is_gelu_formula_valid}")
    print(f"GELU Approximation Accurate: {is_gelu_valid}")
