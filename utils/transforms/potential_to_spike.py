import torch
from jaxtyping import Float, Int
from .types import Bounds, PotentialBounds, TimeBounds, check_domain

"""
domain: The range of possible values for the input potentials.
    This is important for ensuring that the transformations are valid and do not produce out-of-range values.
image_min: The minimum value in the output range of the transformation.
    This can be used to synchronize output spike times with global clock times, ensuring that spikes occur at the correct times relative to the input potentials.
"""

@check_domain
def neg_linear_approximation(
    initial_v: Float[torch.Tensor, "*batch times"],
    domain: PotentialBounds) -> tuple[Float[torch.Tensor, "*batch dims"], TimeBounds]:
    raise NotImplementedError("Identity approximation is not implemented yet.")
    return initial_v, TimeBounds(0.0, 1.0) # Not implemented yet, just return the input as is.

@check_domain
def neg_linear_transform(
    initial_v: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds, # Suppose [0, 1] for normalized potentials.
    window_length: float = 1.0,
    threshold: float = 1.0,
    wave_approx: bool = True
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        TimeBounds]:
    if not wave_approx:
        return window_length * (1 - initial_v / threshold), TimeBounds(0.0, window_length)
    else:
        image:TimeBounds
        t_return, image = neg_linear_approximation(initial_v, domain)
        raise NotImplementedError("Wave approximation is not implemented yet.")
        return t_return, image

@check_domain
def neg_identity_transform(
    initial_v: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds, # Suppose [0, 1] for normalized potentials.
    threshold: float = 1.0,
    wave_approx: bool = True
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        TimeBounds]:
    return neg_linear_transform(initial_v, domain, window_length=1.0, threshold=threshold, wave_approx=wave_approx)

@check_domain
def neg_log_transform(
    initial_v: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    t_in: float = 0.0,
    tau_s: float = 1.0,
    threshold: float = 1.0
    ) -> tuple[Float[torch.Tensor, "*batch dims"], TimeBounds]:
    # Convert tau_s to exp(tau_s) for efficiency, as it is used in the denominator of the log function.
    exp_tau_s = (initial_v.new_ones(1) * tau_s).exp()
    
    # t_in is the time of syncronization spike,
    return (t_in + torch.log(exp_tau_s + 1 - threshold + domain.max) - torch.log((exp_tau_s + 1 - threshold) + initial_v),
            TimeBounds(t_in, t_in - torch.log((exp_tau_s + 1 - threshold) + domain.max)))            