import torch
from jaxtyping import Float, Int
from math import log, exp, isfinite
from numbers import Real
from .types import ClosedBounds, PotentialBounds, SpikeSample, TimeBounds, check_domain
from .noise import inject_spike_time_noise

"""
domain: The range of possible values for the input potentials.
    This is important for ensuring that the transformations are valid and do not produce out-of-range values.
image_min: The minimum value in the output range of the transformation.
    This can be used to synchronize output spike times with global clock times, ensuring that spikes occur at the correct times relative to the input potentials.
"""

@inject_spike_time_noise
@check_domain
def neg_linear_transform(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    window_length: float = 1.0,
    **_
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        TimeBounds]:
    """Encode a bounded potential with a negative-linear TTFS mapping.

    The lower potential rail maps to ``window_length`` and the upper rail maps to
    time zero. Both the potential interval and time window are materialized in the
    payload dtype so the returned carrier and its declared bounds cannot silently
    disagree because of target-dtype overflow, underflow, or endpoint collapse.

    Args:
        input_value: Floating-point potential tensor contained in ``domain``.
        domain: Strictly ordered finite potential rails for affine normalization.
        window_length: Strictly positive finite duration of the output time code.

    Raises:
        TypeError: If the payload is not floating-point or ``window_length`` is not
            a real scalar.
        ValueError: If the time window or potential interval is non-finite,
            non-positive, or not representable in ``input_value.dtype``.

    Returns:
        Negative-linear spike times and the fixed interval
        ``TimeBounds(0.0, window_length)``.
    """
    # Affine timing arithmetic must retain fractional values. An integer payload
    # would also make dtype-level window and endpoint checks truncate silently.
    if not torch.is_floating_point(input_value):
        raise TypeError("input_value must be a floating-point tensor")

    # Validate the public physical duration before converting it into the narrower
    # carrier dtype. Booleans are not meaningful time scales despite being integers.
    if isinstance(window_length, bool) or not isinstance(window_length, Real):
        raise TypeError("window_length must be a real scalar")
    window_value = float(window_length)
    if not isfinite(window_value) or window_value <= 0.0:
        raise ValueError("window_length must be finite and positive")

    # Materialize both potential rails in the payload dtype and compute their width
    # there. This catches finite Python endpoints that overflow or collapse after
    # conversion, as well as a subtraction whose result exceeds the dtype range.
    domain_endpoints = input_value.new_tensor(
        [float(domain.min), float(domain.max)]
    )
    domain_width = domain_endpoints[1] - domain_endpoints[0]
    if not bool(
        torch.isfinite(domain_endpoints).all()
        and torch.isfinite(domain_width)
        and domain_width > 0.0
    ):
        raise ValueError(
            "potential domain must have finite, strictly ordered, "
            "dtype-representable endpoints and width"
        )

    # A finite positive Python duration may still become zero or infinity in the
    # payload dtype. Reject that mismatch before declaring a finite TimeBounds rail.
    window_tensor = input_value.new_tensor(window_value)
    if not bool(torch.isfinite(window_tensor) and window_tensor > 0.0):
        raise ValueError(
            "window_length must remain finite and positive in the input tensor dtype"
        )

    # Normalize the bounded potential, reverse its direction, and contain endpoint
    # roundoff before scaling. The result is therefore finite and lies in the exact
    # declared observation window used by the Gaussian injection boundary.
    normalized_time = 1.0 - (
        (input_value - domain_endpoints[0]) / domain_width
    )
    spike_time = window_tensor * normalized_time.clamp(min=0.0, max=1.0)
    return spike_time, TimeBounds(0.0, window_value)

def neg_identity_transform(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    **kwargs,
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        TimeBounds] | SpikeSample:
    """Encode potentials with a negative-identity TTFS mapping.

    This convenience transform delegates to the decorated negative-linear encoder
    with a time window equal to the potential-domain width. All caller keywords are
    forwarded so event-aware requests reach the shared Gaussian injection boundary
    instead of being silently discarded by this wrapper.

    Args:
        input_value: Initial potentials to encode.
        domain: Valid potential interval and source of the identity-code duration.
        **kwargs: Encoder-control keywords such as ``return_spike_sample`` and
            ``noise_site`` forwarded to :func:`neg_linear_transform`.

    Returns:
        A deterministic ``(time, bounds)`` pair, or a ``SpikeSample`` when the
        event-aware Gaussian path is explicitly requested.
    """
    # The identity code maps the entire potential interval onto an equally long
    # time interval; callers cannot redefine this physical mapping through kwargs.
    forwarded_kwargs = {
        **kwargs,
        "window_length": domain.max - domain.min,
    }

    # Forward event-awareness and site attribution unchanged to the one decorated
    # encoder boundary that owns sampling, deadline classification, and statistics.
    return neg_linear_transform(input_value, domain, **forwarded_kwargs)

@inject_spike_time_noise
@check_domain
def neg_log_transform(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    **_
    ) -> tuple[Float[torch.Tensor, "*batch dims"], TimeBounds]:
    """Encode a strictly positive potential as negative-log spike latency.

    The mapping ``tau_s*log(domain.max/input_value)`` places the upper potential
    endpoint at time zero and the lower endpoint at the fixed observation deadline.
    Validation is performed before tensor arithmetic so invalid physical scales or
    non-positive logarithmic domains fail with stable exceptions instead of asserts.

    Args:
        input_value: Strictly positive potential tensor contained in ``domain``.
        domain: Positive potential rails defining the logarithmic code window.
        tau_s: Positive finite temporal scale applied to all logarithmic latencies.

    Returns:
        Encoded spike times and their fixed interval from zero to
        ``tau_s*log(domain.max/domain.min)``.

    Raises:
        TypeError: If ``tau_s`` is not a real scalar.
        ValueError: If ``tau_s`` is non-finite or non-positive, or if the declared
            logarithmic domain does not have a strictly positive lower endpoint.
    """
    # Reject booleans and invalid physical scales explicitly; unlike ``assert``, this
    # contract remains active under optimized Python execution.
    if isinstance(tau_s, bool) or not isinstance(tau_s, Real):
        raise TypeError("tau_s must be a real scalar")
    tau_value = float(tau_s)
    if not isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau_s must be finite and positive")

    # Negative-log encoding is defined only on a strictly positive potential domain.
    # The domain checker already enforces payload membership, so validating the lower
    # rail also guarantees every accepted input is positive.
    if float(domain.min) <= 0.0:
        raise ValueError(
            "negative-log potential domain must have a strictly positive minimum"
        )

    # Take logarithms independently so a wide but valid positive domain does not
    # overflow while forming either payload or endpoint ratios. Larger potentials
    # map to earlier events, while the endpoint logs fix the shared deadline.
    domain_max = input_value.new_tensor(float(domain.max))
    spike_time = tau_value * (
        torch.log(domain_max) - torch.log(input_value)
    )
    deadline = tau_value * (
        log(float(domain.max)) - log(float(domain.min))
    )
    return spike_time, TimeBounds(0.0, deadline)
