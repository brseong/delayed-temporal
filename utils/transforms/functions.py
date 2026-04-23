import torch
from jaxtyping import Float
from math import log, exp

from .types import PotentialBounds, TimeBounds, check_domain
from .primitive import pulse_width_modulation_operator
from .potential_to_spike import neg_identity_transform, neg_log_transform
from .spike_to_potential import reciprocal_exp_operator, exponential_difference_operator


@check_domain
def multiplication_operator(
    A: torch.Tensor, 
    domain_A: PotentialBounds,
    B: torch.Tensor, 
    domain_B: PotentialBounds,
    theta: float | torch.Tensor
) -> tuple[torch.Tensor, PotentialBounds]:
    """Multiplication operator (psi_M) - Special case of psi_PWM"""
    t_B = theta - B  
    
    # t_B bounds are [theta - B_max, theta - B_min]
    th_val = float(theta) if isinstance(theta, (int, float)) else float(theta.max())
    t_B_min = th_val - domain_B.max
    t_B_max = th_val - domain_B.min
    domain_t_B_arg = TimeBounds(t_B_min, t_B_max)
    
    return pulse_width_modulation_operator(
        t_A=t_B, 
        domain_t_A=domain_t_B_arg, 
        t_B=theta, 
        domain_t_B=th_val, 
        V=A, 
        domain_V=domain_A
    )

@check_domain
def scaled_dot_product_function(
    q: torch.Tensor, 
    domain_q: PotentialBounds,
    k: torch.Tensor, 
    domain_k: PotentialBounds,
    theta: float | torch.Tensor
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

@check_domain
def exponential_function(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_m: float = 1.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Exponential Potential operator (phi_EP)
    Calculates exp(V/tau_m) by composing phi_NP and psi_NE.
    """
    # 1. phi_NP (Negative Potential operator): V -> t_out = theta - V
    t_out, tb_out = neg_identity_transform(input_value, domain)
    
    # 2. psi_NE (Negative Exp-Temporal operator): t_out -> exp(-(t-t_out)/tau_m)
    # This results in exp(V/tau_m) * constant
    return reciprocal_exp_operator(t_out, tb_out, tau_m=tau_m)

@check_domain
def softmin_function(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_s: float = 1.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Apply Softmax transform to the input potentials to produce normalized weights.
    
    According to Lemma 4.3 (Derivation of Softmax Normalization) in the paper:
    w_{softmax, ij} ≈ psi_DIV(s_ij, sum_k s_ik)
    """
    # 1. Exponential potential transformation: exp_v = exp(s_ij / tau_s)
    exp_v, exp_domain = exponential_function(input_value, domain, tau_m=tau_s)
    
    # 2. Sum of exponentiated scores: sum_k exp(s_ik / tau_s)
    sumexp_v = exp_v.sum(dim=-1, keepdim=True)
    N = input_value.size(-1)
    sumexp_domain = PotentialBounds(exp_domain.min, exp_domain.max * N)
    
    # 3. Apply the Division Operator: psi_DIV(exp_v, sumexp_v)
    return division_function(
        X=exp_v,
        Y=sumexp_v,
        joint_domain=sumexp_domain,
        tau_s=tau_s
    )

@check_domain
def division_function(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    joint_domain: PotentialBounds,
    tau_s: float
) -> tuple[torch.Tensor, PotentialBounds]:
    """Division Operator (\\psi_DIV) based on mathematical identity.
    Strictly formulated as composition of \\psi_ED and \\phi_NL according to paper.
    """
    assert torch.all(X <= Y), "For division to be valid, each element of X must be less than or equal to the corresponding element of Y."
    
    # Note: must use same domain for both X and Y to synchronize the transformation, as the division operator relies on the relative values of X and Y after transformation.
    t_X, tb_X = neg_log_transform(X, joint_domain, tau_s=tau_s)
    t_Y, tb_Y = neg_log_transform(Y, joint_domain, tau_s=tau_s)
    
    result, domain_result = exponential_difference_operator(t_X, tb_X, t_Y, tb_Y, tau_s=tau_s)
    correction = exp(tb_X.max / tau_s)
    return result * correction, PotentialBounds(domain_result.min * correction, domain_result.max * correction)

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
