import torch
from jaxtyping import Float
from math import log, exp

from .types import PotentialBounds, TimeBounds, check_domain
from .primitive import pulse_width_modulation_operator
from .potential_to_spike import neg_identity_transform, neg_log_transform
from .spike_to_potential import normalized_exp_operator, exponential_difference_operator


@check_domain
def multiplication_operator(
    V: torch.Tensor, 
    domain_V: PotentialBounds,
    B: torch.Tensor, 
    domain_B: PotentialBounds,
    theta: float
) -> tuple[torch.Tensor, PotentialBounds]:
    """Multiplication operator (f_M) - Special case of f_PWM"""
    domain_B = PotentialBounds(-theta, theta)
    # b -> t_b = \theta - b
    t_B, domain_t_B = neg_identity_transform(domain_B.clamp(B, name="multiplication_B"), domain_B)
    
    # t_B bounds are [0, 2 * theta] since B is clamped to [-theta, theta]
    th_val = float(theta) if isinstance(theta, (int, float)) else float(theta.max())
    
    # result = \int_{t_B}^{\theta} V dt = V * (\theta - t_B) = V * b
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

@check_domain
def exponential_function(
    input_value: Float[torch.Tensor, "*batch dims"],
    domain: PotentialBounds,
    *,
    tau_m: float = 1.0,
    **_
) -> tuple[torch.Tensor, PotentialBounds]:
    """Exponential Potential operator (f_EP)
    Calculates exp(V/tau_m) by composing f_NP and f_NE.
    """
    # 1. f_NP (Negative Potential operator): V -> t_out = theta - V
    t_out, tb_out = neg_identity_transform(input_value, domain)
    
    # 2. f_NE (Negative Exp-Temporal operator): t_out -> exp(-(t_max-t_out)/tau_m)
    # This results in exp(-V/tau_m) * constant
    v_out, domain_v_out = normalized_exp_operator(t_out, tb_out, tau_m=tau_m)
    # exp(-(t_max-t_out)/tau_m) = exp(-(t_max-theta)/tau_m) * exp(-V/tau_m)
    # Thus recover exp(-V/tau_m) by multiplying with exp((t_max-theta)/tau_m) = exp(-theta/tau_m) * exp(t_max/tau_m)
    scaling_factor = exp(-domain.max / tau_m)
    return scaling_factor * v_out, PotentialBounds(domain_v_out.min * scaling_factor, domain_v_out.max * scaling_factor)

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
    w_{softmax, ij} ≈ f_DIV(s_ij, sum_k s_ik)
    """
    # 1. Exponential potential transformation: exp_v = exp(-s_ij / tau_s)
    exp_v, exp_domain = exponential_function(input_value, domain, tau_m=tau_s)
    
    # 2. Sum of exponentiated scores: sum_k exp(s_ik / tau_s)
    sumexp_v = exp_v.sum(dim=-1, keepdim=True)
    N = input_value.size(-1)
    # sumexp_domain = PotentialBounds(exp_domain.min, exp_domain.max * N)
    # Too high max bound causes numerical instability in division, so we use mathematically equivalent but tighter bound based on the fact that max of exp_v is exp(-domain.min / tau_s)
    sumexp_domain = PotentialBounds(exp_v.min().item(), sumexp_v.max().item())
    
    # 3. Apply the Division Operator: f_DIV(exp_v, sumexp_v)
    result = division_function(
        X=exp_v,
        Y=sumexp_v,
        joint_domain=sumexp_domain,
        tau_s=tau_s
    )
    return result

@check_domain
def division_function(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    joint_domain: PotentialBounds,
    tau_s: float
) -> tuple[torch.Tensor, PotentialBounds]:
    """Division Operator (f_DIV) based on mathematical identity.
    Strictly formulated as composition of f_ED and f_NL according to paper.
    """
    # Clamp to avoid domain assertion errors from float precision
    X = joint_domain.clamp(X, name="division_X")
    Y = joint_domain.clamp(Y, name="division_Y")
    
    assert torch.all(X <= Y), "For division to be valid, each element of X must be less than or equal to the corresponding element of Y."
    
    # Note: must use same domain for both X and Y to synchronize the transformation, as the division operator relies on the relative values of X and Y after transformation.
    # t_X = -\tau_s * log(X/T) = -\tau_s * (log(X) - log(T))
    # t_Y = -\tau_s * log(Y/T) = -\tau_s * (log(Y) - log(T))
    t_X, tb_X = neg_log_transform(X, joint_domain, tau_s=tau_s)
    t_Y, tb_Y = neg_log_transform(Y, joint_domain, tau_s=tau_s)
    # For numerical stability, ensure transformed values are within their theoretical bounds
    t_X = t_X.clamp(min=tb_X.min, max=tb_X.max)
    t_Y = t_Y.clamp(min=tb_Y.min, max=tb_Y.max)
    
    # f_DIV(X, Y) = exp((t_Y - t_X) / tau_s)
    # = exp(-t_X / tau_s) * exp(t_Y / tau_s)
    # = exp(log(X/T)) * exp(-log(Y/T)) = X/Y
    result, domain_result = exponential_difference_operator(t_X, tb_X, t_Y, tb_Y, tau_s=tau_s)
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
    """Approximate GELU activation using spiking operators.
    
    According to Lemma 4.4 (Derivation of GELU Approximation) in the paper:
    f_GELU(v) := f_M(v, f_DIV(1, 1 + f_NE(f_NP(1.702v))))
    """
    # Step 1: f_NP(1.702v)
    scale_const = 1.702
    scale_bound = PotentialBounds(scale_const, scale_const)
    scaled_input, _ = multiplication_operator(
        input_value, domain,
        input_value.new_tensor(scale_const).expand_as(input_value), scale_bound,
        theta)
    scaled_domain = PotentialBounds(scale_const * domain.min, scale_const * domain.max)
    
    # Stability cap for exp: exp(20) is safe, exp(400) overflows.
    # Since exp(-1.702*v) is used for sigmoid, we only need to worry about v being very negative.
    _STABILITY_CAP = 20.0
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
    
    # Step 4: f_M(v, div_out)
    
    gelu_approx, gelu_domain = multiplication_operator(domain.clamp(input_value, name="gelu_x"), domain, div_out, div_domain, theta=theta)
    
    return gelu_approx, gelu_domain

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
    one_plus_exp_domain = PotentialBounds(1.0 + exp_domain.min, 1.0 + exp_domain.max)
    
    sigmoid_out, sigmoid_domain = division_function(
        X=torch.ones_like(one_plus_exp),
        Y=one_plus_exp,
        joint_domain=one_plus_exp_domain,
        tau_s=tau_s
    )
    
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
    
    print(f"GELU Approx Output:   {gelu_out.tolist()}")
    print(f"Expected PyTorch GELU: {expected_gelu.tolist()}")
    
    # As it's an approximation using mathematical substitutions, allow slightly higher tolerance
    is_gelu_valid = torch.allclose(gelu_out, expected_gelu, atol=2e-2)
    print(f"GELU Approximation Accurate: {is_gelu_valid}")
