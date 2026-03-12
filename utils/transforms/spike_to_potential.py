import torch
from jaxtyping import Float, Int
from math import log, exp
from .types import OpenBounds, PotentialBounds, TimeBounds, check_domain

@check_domain
def reciprocal_exp_transform(
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
    return torch.exp(-(domain.max - input_value) / tau_m), PotentialBounds(exp(-domain.max / tau_m), exp(-domain.min / tau_m))
    