import torch
from jaxtyping import Float, Int
from math import log, exp

from .potential_to_spike import neg_identity_transform
from .types import OpenBounds, PotentialBounds, TimeBounds, check_domain
from .primitive import pulse_width_modulation_operator

@check_domain
def reciprocal_exp_operator(
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
    

@check_domain
def exponential_difference_operator(
    t_A: torch.Tensor,
    domain_t_A: TimeBounds,
    t_B: torch.Tensor,
    domain_t_B: TimeBounds,
    tau_s: float = 1.0
) -> tuple[torch.Tensor, PotentialBounds]:
    """Exponential Difference operator (\\psi_ED)

    ψ_ED(a, b) := exp(-ψ_PWM(a, b; V_ref) / τ_s)  where I(V_ref) = -1
    Step 1: ψ_PWM(t_A, t_B; -1) = t_A − t_B  (= a − b)
    Step 2: φ_NP → ψ_NE  ∝  exp(-(a-b)/τ_s) = exp((b-a)/τ_s)
    """
    V_ref = torch.full_like(t_A, fill_value=-1.0)
    p, domain_p = pulse_width_modulation_operator(
        t_A, domain_t_A, t_B, domain_t_B, V_ref, PotentialBounds(-1.0, -1.0)
    )
    # p = t_A - t_B;  φ_NP → ψ_NE gives ∝ exp(-p/τ) = exp((b-a)/τ)
    s, domain_s = neg_identity_transform(p, domain_p)
    return reciprocal_exp_operator(s, domain_s, tau_m=tau_s)

