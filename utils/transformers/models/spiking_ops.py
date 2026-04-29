"""Shared SNN operator classes used across spiking transformer models."""

import math
from typing import Optional

import torch
from torch import nn

from utils.transforms import neg_identity_transform
from utils.transforms.functions import multiplication_operator, division_function
from utils.transforms.potential_to_spike import neg_log_transform
from utils.transforms.primitive import pulse_width_modulation_operator
from utils.transforms.spike_to_potential import exponential_difference_operator
from utils.transforms.types import Potential, PotentialBounds, TimeBounds, _emit_spike_time_core


class SpikingLayerNorm(nn.Module):
    """LayerNorm via SNN operators (Lemma 4.4).

    Computes (x - mean) / std using ψ_M for variance and ψ_ED for division,
    with dual-rail encoding to handle signed activations.
    Output is rescaled to match pretrained ANN LayerNorm weights, so the residual
    1/sqrt(theta) factor from the spiking derivation is compensated explicitly.

    Each of the three SNN stages can be replaced with its standard-PyTorch equivalent
    for ablation analysis:
      use_spiking_mul    : ψ_M  for variance  vs  x²
      use_spiking_log    : φ_NL for encoding  vs  τ·log(hi/x)
      use_spiking_expdiff: ψ_ED for division  vs  exp((t_σ - t_x)/τ)
    """

    def __init__(self, normalized_shape, eps=1e-5, theta=200.0, tau_s=1.0,
                 use_spiking_mul=False, use_spiking_log=True, use_spiking_expdiff=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.theta = theta
        self.tau_s = tau_s
        self.use_spiking_mul = use_spiking_mul
        self.use_spiking_log = use_spiking_log
        self.use_spiking_expdiff = use_spiking_expdiff
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
    
    def forward(self, pot: Potential) -> Potential:
        x: torch.Tensor = pot.value

        if not self.use_spiking_mul and not self.use_spiking_log and not self.use_spiking_expdiff:
            out = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return Potential(out, PotentialBounds(out.min().item(), out.max().item()))

        eps = self.eps
        theta = self.theta
        tau_s = self.tau_s

        x_err = x - x.mean(dim=-1, keepdim=True)
        domain_err: PotentialBounds = PotentialBounds(eps, theta - eps)
        x_err_pos = domain_err.clamp(x_err)
        x_err_neg = domain_err.clamp(-x_err)

        if self.use_spiking_mul:
            M_pos, _ = multiplication_operator(x_err_pos, domain_err, x_err_pos, domain_err, theta)
            M_neg, _ = multiplication_operator(x_err_neg, domain_err, x_err_neg, domain_err, theta)
            var_x = (M_pos + M_neg).mean(dim=-1, keepdim=True)
        else:
            var_x = (x_err_pos.pow(2) + x_err_neg.pow(2)).mean(dim=-1, keepdim=True)

        var_x = var_x + eps
        domain_var: PotentialBounds = PotentialBounds(domain_err.min ** 2, domain_err.max ** 2)
        var_x = domain_var.clamp(var_x)

        T0 = tau_s * math.log(domain_err.max / domain_err.min)
        if self.use_spiking_log:
            # First, we need tau_s/2 for sigma, to get sqrt of variance.
            # Thus, to match the bias terms between sigma and x_err terms
            # in the exponential difference operator, we also need to use tau_s/2 for x_err:
            # tau_s/2 * log(hi^2) = tau_s * log(hi).
            # hi^2 is the upper bound of variance, and hi is the upper bound of x_err, so this ensures the same bias term of tau_s * log(hi) for both sigma and x_err in the expdiff operator.
            t_sigma, tb_sigma = neg_log_transform(var_x, domain_var, tau_s=tau_s/2)
            t_err_pos, tb_err = neg_log_transform(x_err_pos, domain_err, tau_s=tau_s)
            t_err_neg, _ = neg_log_transform(x_err_neg, domain_err, tau_s=tau_s)
        else:
            _hi_t = x.new_tensor(domain_err.max)
            _hi2_t = x.new_tensor(domain_err.max ** 2)
            t_sigma = (tau_s / 2.0) * torch.log(_hi2_t / var_x)
            t_err_pos = tau_s * torch.log(_hi_t / x_err_pos)
            t_err_neg = tau_s * torch.log(_hi_t / x_err_neg)
            tb_sigma = TimeBounds(0.0, T0)
            tb_err = TimeBounds(0.0, T0)

        if self.use_spiking_expdiff:
            y_pos, _ = exponential_difference_operator(t_err_pos, tb_err, t_sigma, tb_sigma, tau_s=tau_s)
            y_neg, _ = exponential_difference_operator(t_err_neg, tb_err, t_sigma, tb_sigma, tau_s=tau_s)
            result: torch.Tensor = y_pos - y_neg
            out = multiplication_operator(
                result,
                PotentialBounds(result.min().item(), result.max().item()),
                self.weight,
                PotentialBounds(self.weight.min().item(), self.weight.max().item()),
                theta)[0] + self.bias
        else:
            y_pos = torch.exp((t_sigma - t_err_pos) / tau_s)
            y_neg = torch.exp((t_sigma - t_err_neg) / tau_s)
            result = y_pos - y_neg
            out = self.weight * result + self.bias

        out_domain = PotentialBounds(out.min().item(), out.max().item())
        return Potential(out, out_domain)


class SpikingLinear(nn.Linear):
    """Linear layer via ψ_PWM operator. Numerically identical to nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 theta: float = 400.0, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.theta = theta

    def forward(self, input: Potential) -> Potential:
        x: torch.Tensor = input.value
        domain_x = PotentialBounds(-self.theta, self.theta)
        t_A, domain_t_A = neg_identity_transform(domain_x.clamp(x), domain_x)
        w_min, w_max = self.weight.min().item(), self.weight.max().item()
        if w_min == w_max:
            w_min, w_max = w_min - 1e-8, w_max + 1e-8
        domain_W: PotentialBounds = PotentialBounds(w_min, w_max)
        y_syn, domain_y_syn = pulse_width_modulation_operator(
            t_A.unsqueeze(-2), domain_t_A,
            self.theta, self.theta,
            self.weight, domain_W,
        )
        N = self.in_features
        domain_y: PotentialBounds = PotentialBounds(
            domain_y_syn.min * N,
            domain_y_syn.max * N,
        )
        y: torch.Tensor = y_syn.sum(dim=-1)
        if self.bias is not None:
            b_min, b_max = self.bias.min().item(), self.bias.max().item()
            domain_y = PotentialBounds(domain_y.min + b_min, domain_y.max + b_max)
            y = y + self.bias
        return Potential(y, domain_y)


def _apply_norm(norm: nn.Module, pot: Potential) -> Potential:
    if isinstance(norm, SpikingLayerNorm):
        return norm(pot)
    out = norm(pot.value)
    return Potential(out, PotentialBounds(out.min().item(), out.max().item()))


if __name__ == "__main__":
    torch.manual_seed(7)

    x = torch.tensor(
        [[0.25, -1.50, 2.00, 0.75],
         [1.25, 0.50, -0.75, 2.50]],
        dtype=torch.float32,
    )
    pot = Potential(x, PotentialBounds(x.min().item(), x.max().item()))

    normalized_shape = x.shape[-1]
    eps = 1e-5
    theta = 200.0
    tau_s = 1.0

    layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
    spiking_layer_norm = SpikingLayerNorm(
        normalized_shape,
        eps=eps,
        theta=theta,
        tau_s=tau_s,
        use_spiking_mul=False,
        use_spiking_log=True,
        use_spiking_expdiff=True,
    )

    with torch.no_grad():
        spiking_layer_norm.weight.copy_(layer_norm.weight)
        spiking_layer_norm.bias.copy_(layer_norm.bias)

    ln_out = layer_norm(x)
    spiking_out = spiking_layer_norm(pot)

    x_hat = x - x.mean(dim=-1, keepdim=True)
    variance = x_hat.pow(2).mean(dim=-1, keepdim=True)
    std = torch.sqrt(variance + eps)
    manual_ln = (x_hat / std) * layer_norm.weight + layer_norm.bias

    print("=== LayerNorm vs SpikingLayerNorm ===")
    print(f"input:\n{x}")
    print(f"mean:\n{x.mean(dim=-1, keepdim=True)}")
    print(f"centered x_hat:\n{x_hat}")
    print(f"variance:\n{variance}")
    print(f"std:\n{std}")
    print(f"nn.LayerNorm output:\n{ln_out}")
    print(f"manual LayerNorm output:\n{manual_ln}")
    print(f"SpikingLayerNorm output:\n{spiking_out.value}")
    print(f"SpikingLayerNorm domain: [{spiking_out.domain.min}, {spiking_out.domain.max}]")

    print("=== Comparison ===")
    print(f"max |nn - manual| = {(ln_out - manual_ln).abs().max().item():.6e}")
    print(f"max |nn - spiking| = {(ln_out - spiking_out.value).abs().max().item():.6e}")
    print(f"allclose(nn, manual) = {torch.allclose(ln_out, manual_ln, atol=1e-5, rtol=1e-5)}")
    print(f"allclose(nn, spiking) = {torch.allclose(ln_out, spiking_out.value, atol=1e-3, rtol=1e-3)}")

    # --- Explicit division_function test (sigmoid via joint-domain division) ---
    print('\n=== division_function sigmoid test ===')
    beta = 1.0
    u = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32)
    exp_v = torch.exp(-beta * u)
    joint_dom = PotentialBounds(1.0, 1.0 + float(exp_v.max().item()))
    div_out, div_dom = division_function(
        X=torch.ones_like(exp_v),
        Y=1.0 + exp_v,
        joint_domain=joint_dom,
        tau_s=tau_s
    )
    print('division_function output:', div_out)
    print('torch.sigmoid output:  ', torch.sigmoid(beta * u))
