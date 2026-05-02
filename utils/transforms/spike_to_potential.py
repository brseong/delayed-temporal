import torch
from jaxtyping import Float, Int
from math import exp

from .potential_to_spike import neg_identity_transform
from .types import OpenBounds, PotentialBounds, TimeBounds, check_domain
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
    # result = exp(-(domain.max - input_value) / tau_m)
    # = exp(-domain.max / tau_m) * exp(input_value / tau_m)
    # scaling_factor = exp(domain.max / tau_m)
    result, domain_result = exp_operator(input_value, domain, tau_m=tau_m)
    scaling_factor = exp(domain.max / tau_m)
    # out = scaling_factor * result
    # = exp(domain.max / tau_m) * exp(-domain.max / tau_m) * exp(input_value / tau_m)
    # = exp(input_value / tau_m)
    return scaling_factor * result, PotentialBounds(domain_result.min * scaling_factor, domain_result.max * scaling_factor)

@check_domain
def exponential_difference_operator(
    t_A: torch.Tensor,
    domain_t_A: TimeBounds,
    t_B: torch.Tensor,
    domain_t_B: TimeBounds,
    tau_s: float = 1.0
) -> tuple[torch.Tensor, PotentialBounds]:
    """Exponential Difference operator (\\psi_ED)

    ψ_ED(a, b) := exp(-ψ_PWM(a, b; V_ref) / τ_s)  where I(V_ref) = 1
    Step 1: ψ_PWM(t_A, t_B; 1) = t_A − t_B  (= a − b)
    Step 2: φ_NP → ψ_NE  ∝  exp(-(a-b)/τ_s) = exp((b-a)/τ_s)
    """
    # p = -1 * (T_B - T_A) = T_A - T_B
    V_ref = torch.full_like(t_A, fill_value=-1.0)
    p, domain_p = pulse_width_modulation_operator(
        t_A, domain_t_A, t_B, domain_t_B, V_ref, PotentialBounds(-1.0, -1.0)
    )
    # s = theta - p, theta = domain_p.max
    s, domain_s = neg_identity_transform(p, domain_p)
    scaling_factor = exp(-domain_p.max / tau_s)
    # p = exp(-(T-s)/tau_s) = exp(-(T-theta+p)/tau_s) = exp(-T/tau_s) * exp(theta/tau_s) * exp(-p/tau_s), T = domain_s.max, theta = domain_p.max
    # p_normalized = exp(domain_p.max / tau_s) * exp(-p / tau_s), removes exp(-T/tau_s), which is constant.
    p, domain_p = normalized_exp_operator(s, domain_s, tau_m=tau_s)
    # result = scaling_factor * p_normalized
    # = exp(-domain_p.max / tau_s) * exp(domain_p.max / tau_s) * exp(-p / tau_s)
    # = exp(-p / tau_s)
    # = exp((T_B - T_A) / tau_s)
    result = scaling_factor * p, PotentialBounds(domain_p.min * scaling_factor, domain_p.max * scaling_factor)
    return result

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
