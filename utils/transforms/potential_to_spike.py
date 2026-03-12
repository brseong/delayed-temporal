import torch
from jaxtyping import Float, Int
from math import log, exp
from .types import OpenBounds, PotentialBounds, TimeBounds, check_domain

"""
domain: The range of possible values for the input potentials.
    This is important for ensuring that the transformations are valid and do not produce out-of-range values.
image_min: The minimum value in the output range of the transformation.
    This can be used to synchronize output spike times with global clock times, ensuring that spikes occur at the correct times relative to the input potentials.
"""

@check_domain
def neg_linear_approximation(
    input_value: Float[torch.Tensor, "*batch times"],
    domain: PotentialBounds,
    **_) -> tuple[Float[torch.Tensor, "*batch dims"], TimeBounds]:
    raise NotImplementedError("Identity approximation is not implemented yet.")
    return input_value, TimeBounds(0.0, 1.0) # Not implemented yet, just return the input as is.

@check_domain
def neg_linear_transform(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    window_length: float = 1.0,
    wave_approx: bool = False,
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
    if not wave_approx:
        range = domain.max - domain.min
        return window_length * (1 - (input_value - domain.min) / range), TimeBounds(0.0, window_length)
    else:
        image:TimeBounds
        t_return, image = neg_linear_approximation(input_value, domain)
        raise NotImplementedError("Wave approximation is not implemented yet.")
        return t_return, image

@check_domain
def neg_identity_transform(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    wave_approx: bool = False,
    **_
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        TimeBounds]:
    """Apply negative-identity transform to the input potentials to produce spike times.

    Args:
        input_value (Float[torch.Tensor, "*batch dims"]): Initial potentials of the neurons.
        domain (PotentialBounds): The range of possible values for the input potentials.
        wave_approx (bool, optional): Whether to use a wave approximation for the transformation.
            If True, the transformation will produce spike times that approximate a waveform. Defaults to False.
    
    Returns:
        tuple[Float[torch.Tensor, "*batch dims"], TimeBounds]: A tuple containing the transformed spike times and the time bounds of the output.
    """
    return neg_linear_transform(input_value,
                                domain,
                                window_length=domain.max - domain.min,
                                wave_approx=wave_approx)

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
    return (-tau_s * torch.log(input_value) + tau_s * log(domain.max), TimeBounds(0, -tau_s * log(domain.min) + tau_s * log(domain.max)))            