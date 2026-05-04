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
        
        # Debug: check if x_err exceeds theta
        # max_val = x_err.abs().max().item()
        # if max_val > theta:
        #     print(f"[DEBUG] x_err max {max_val:.2f} exceeds theta {theta}")
            
        domain_err: PotentialBounds = PotentialBounds(eps, theta - eps)
        x_err_pos = domain_err.clamp(x_err, name="x_err_pos")
        x_err_neg = domain_err.clamp(-x_err, name="x_err_neg")

        if self.use_spiking_mul:
            M_pos, _ = multiplication_operator(x_err_pos, domain_err, x_err_pos, domain_err, theta)
            M_neg, _ = multiplication_operator(x_err_neg, domain_err, x_err_neg, domain_err, theta)
            var_x = (M_pos + M_neg).mean(dim=-1, keepdim=True)
        else:
            var_x = (x_err_pos.pow(2) + x_err_neg.pow(2)).mean(dim=-1, keepdim=True)

        var_x = var_x + eps
        domain_var: PotentialBounds = PotentialBounds(domain_err.min ** 2, domain_err.max ** 2)
        var_x = domain_var.clamp(var_x, name="var_x")

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
        t_A, domain_t_A = neg_identity_transform(domain_x.clamp(x, name="linear_x"), domain_x)
        w_min, w_max = self.weight.min().item(), self.weight.max().item()
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

class SpikingConv2d(nn.Conv2d):
    """2D convolution via ψ_PWM operator. Numerically identical to nn.Conv2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 theta: float = 400.0, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias, device=device, dtype=dtype)
        self.theta = theta

    def forward(self, input: Potential) -> Potential:
        x: torch.Tensor = input.value
        domain_x = PotentialBounds(-self.theta, self.theta)
        t_A, domain_t_A = neg_identity_transform(domain_x.clamp(x, name="conv2d_x"), domain_x)
        w_min, w_max = self.weight.min().item(), self.weight.max().item()
        domain_W: PotentialBounds = PotentialBounds(w_min, w_max)
        
        # Manual padding for t_A to handle spiking domain correctly (x=0 corresponds to t=theta)
        if self.padding[0] > 0 or self.padding[1] > 0:
            t_A = torch.nn.functional.pad(t_A, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), value=self.theta)
        
        # Unfold input to (B, C_in * kH * kW, L)
        t_A_unfolded = nn.functional.unfold(
            t_A, self.kernel_size, self.dilation, padding=0, stride=self.stride
        )
        B, _, L = t_A_unfolded.shape
        G = self.groups
        C_out = self.out_channels
        
        # Reshape for grouped convolution broadcasting
        # t_A_unfolded: (B, G, C_in//G * kH * kW, L)
        t_A_unfolded = t_A_unfolded.view(B, G, -1, L)
        
        # weight: (G, C_out//G, C_in//G * kH * kW)
        V = self.weight.view(G, C_out // G, -1)
        
        # Prepare for broadcasting:
        # V:  (1, G, C_out//G, C_in//G * kH * kW, 1)
        # dt: (B, G, 1,         C_in//G * kH * kW, L)
        y_syn, domain_y_syn = pulse_width_modulation_operator(
            t_A_unfolded.unsqueeze(2), domain_t_A,
            self.theta, self.theta,
            V.unsqueeze(0).unsqueeze(-1), domain_W,
        )
        
        # y_syn: (B, G, C_out//G, C_in//G * kH * kW, L)
        y = y_syn.sum(dim=3).view(B, C_out, L)
        
        # Determine output H and W
        H_in, W_in = x.shape[2:]
        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride
        dh, dw = self.dilation
        H_out = (H_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        W_out = (W_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        
        y = y.view(B, C_out, H_out, W_out)
        
        N = (self.in_channels // G) * kh * kw
        domain_y: PotentialBounds = PotentialBounds(
            domain_y_syn.min * N,
            domain_y_syn.max * N,
        )
        
        if self.bias is not None:
            b_min, b_max = self.bias.min().item(), self.bias.max().item()
            domain_y = PotentialBounds(domain_y.min + b_min, domain_y.max + b_max)
            y = y + self.bias.view(1, -1, 1, 1)
        return Potential(y, domain_y)


def _apply_norm(norm: nn.Module, pot: Potential) -> Potential:
    if isinstance(norm, SpikingLayerNorm):
        return norm(pot)
    out = norm(pot.value)
    return Potential(out, PotentialBounds(out.min().item(), out.max().item()))


if __name__ == "__main__":
    import torch
    from torch import nn

    torch.manual_seed(42)
    
    dim = 768
    theta = 400.0
    
    # Initialize layers
    ln = nn.LayerNorm(dim)
    sln = SpikingLayerNorm(dim, theta=theta)
    
    # Sync weights
    with torch.no_grad():
        sln.weight.copy_(ln.weight)
        sln.bias.copy_(ln.bias)
    
    max_diff = -1.0
    worst_std = -1
    max_x_err_at_worst = 0.0
    
    print(f"Testing standard deviations from 1 to 128 for dim={dim}, theta={theta}...")
    
    for std in range(1, 129):
        # Create input tensor with mean 0 and current std
        x = torch.randn(1, dim) * std
        # Create Potential object for SpikingLayerNorm
        pot = Potential(x, PotentialBounds(x.min().item(), x.max().item()))
        
        with torch.no_grad():
            x_err = x - x.mean(dim=-1, keepdim=True)
            max_x_err = x_err.abs().max().item()
            
            ln_out = ln(x)
            sln_out = sln(pot).value
            
            diff = (ln_out - sln_out).abs().max().item()
            
            if diff > max_diff:
                max_diff = diff
                worst_std = std
                max_x_err_at_worst = max_x_err
                
    print("\n=== Result ===")
    print(f"Standard deviation with maximum difference: {worst_std}")
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Max abs(x_err) at worst std: {max_x_err_at_worst:.2f} (theta={theta})")
