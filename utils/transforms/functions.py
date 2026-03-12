import torch
from jaxtyping import Float, Int
from math import log, exp

from .types import OpenBounds, PotentialBounds, TimeBounds, check_domain
from .potential_to_spike import neg_identity_transform, neg_log_transform
from .spike_to_potential import reciprocal_exp_transform

@check_domain
def softmin_p2p(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau: float = 1.0,
    **_
    ) -> tuple[
        Float[torch.Tensor, "*batch dims"],
        PotentialBounds]:
    """Apply softmin transform to the input potentials to produce spike times.
    Input potentials are transformed in two steps:
        1. Negative-identity transform: v -> 1-v, which produces spike times that are linearly related to the input potentials.
        2. Exponential negative transform: t -> exp(-(T-t)/tau),
            which equals to exp(-(T-1+v)/tau) = exp(-(T-1)/tau) * exp(-v/tau),
            where T is the time window length determined by the time domain of the negative-identity transform,
            and tau is the temperature parameter that controls the sharpness of the transformation.

    Args:
        v (Float[torch.Tensor, "*batch dims"]): Input potentials of the neurons.
        potential_domain (PotentialBounds): The range of possible values for the input potentials.
        tau (float, optional): The temperature parameter for the softmin transform. Defaults to 1.0.
            Equals to time constant in the exponential post-synaptic potential (PSP) function, which controls the sharpness of the transformation.
        """
    assert tau > 0, "Temperature parameter tau must be positive."
    
    # Voltage_domain (V_min, V_max)
    neg_t, time_domain = neg_identity_transform(input_value, domain) # t represents \theta-v.
    # Time_domain (0, V_max - V_min)
    exp_v, potential_domain = reciprocal_exp_transform(neg_t, time_domain, tau=tau) # v represents exp(-(T-t)/tau) = exp(-(T-\theta+v)/tau) = C * exp(-v/tau), where C is a constant determined by the time domain.
    # Potential_domain (0<exp((V_min - V_max)/tau), 1.0)      = (exp(-(V_max - V_min)/tau), exp(-0/tau))
    sumexp_v = exp_v.sum(dim=-1, keepdim=True)       # Potentials addition
    # Potential_domain (0<N*exp((V_min - V_max)/tau), N),     where N is the number of elements in the last dimension.
    N = input_value.size(-1)
    potential_domain = PotentialBounds(potential_domain.min, potential_domain.max * N) # To use the same current intensity
    neg_t = neg_log_transform(exp_v, potential_domain, tau=tau)[0] # This is equivalent to -log(numerator) 
    neg_logsumexp = neg_log_transform(sumexp_v, potential_domain, tau=tau)[0] # This is equivalent to -log(denominator)
    # Time_domain (0, (V_max - V_min)/tau + log(N))
    logsoftmin = -neg_t + neg_logsumexp # This is equivalent to log(denominator) - log(numerator) but using the spiking transformation instead of the logarithm function.
    attn_weight = logsoftmin.exp() # This is equivalent to softmax(attn_weight, dim=-1) but using the spiking transformation instead of the exponential function.
    return attn_weight, PotentialBounds(0.0, 1.0)