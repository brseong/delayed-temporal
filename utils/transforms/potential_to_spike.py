import torch
from jaxtyping import Float, Int
from math import log, exp
from .types import OpenBounds, PotentialBounds, SpikeSample, TimeBounds, check_domain
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
    """Apply negative-linear transform to the input potentials to produce spike times.

    Args:
        input_value (Float[torch.Tensor, "*batch dims"]): Initial potentials of the neurons.
        domain (PotentialBounds): The range of possible values for the input potentials. supremum represents the threshold potential.
        window_length (float, optional): The length of the time window for the output spike times. Defaults to 1.0.
        wave_approx (bool, optional): Whether to use a wave approximation for the transformation.
            If True, the transformation will produce spike times that approximate a waveform. Defaults to False.

    Raises:
        NotImplementedError: wave approximation is not implemented yet.

    Returns:
        tuple[Float[torch.Tensor, "*batch dims"], TimeBounds]: A tuple containing the transformed spike times and the time bounds of the output.
    """
    range = domain.max - domain.min
    return window_length * (1 - (input_value - domain.min) / range).clamp(min=0.0, max=1.0), TimeBounds(0.0, window_length)

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
    assert tau_s > 0, "Time constant tau_s must be positive."
    """Apply a negative logarithmic transformation to the input potentials to produce spike times.
    
    Args:
        input_value (Float[torch.Tensor, "*batch dims"]): Input potentials of the neurons.
        domain (PotentialBounds): The range of possible values for the input potentials.
        tau_s (float, optional): Time constant for the transformation. Defaults to 1.0.
    
    Returns:
        tuple[Float[torch.Tensor, "*batch dims"], TimeBounds]:
        A tuple containing the transformed spike times and the time bounds of the output.
    """
    assert domain.min > 0.0, "The minimum of the potential domain must be greater than 0 for the logarithmic transform to be valid."
    # As the potential decreases towards the minimum, the spike time increases towards maximum.
    # This is the maximum spike time corresponding to the minimum potential in the domain.
    return -tau_s * torch.log(input_value) + tau_s * log(domain.max), TimeBounds(0, -tau_s * log(domain.min) + tau_s * log(domain.max))
