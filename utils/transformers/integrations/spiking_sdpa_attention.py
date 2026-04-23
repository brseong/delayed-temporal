import torch
import math
import wandb

from transformers.utils import is_torch_npu_available, is_torch_xpu_available, logging
from transformers.utils.import_utils import is_torch_greater_or_equal
from utils.transforms.functions import scaled_dot_product_function, softmin_function
from utils.transforms.primitive import pulse_width_modulation_operator
from utils.transforms.types import PotentialBounds, TimeBounds

logger = logging.get_logger(__name__)

# Softmin masking: large positive suppresses masked positions (exp(-87) ≈ float32 min)
_MASK_VAL = 87.0

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None

def spiking_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, tau_m=1.0, theta=10.0) -> torch.Tensor:

    L, S = query.size(-2), key.size(-2)

    if enable_gqa:
        raise NotImplementedError("GQA is not implemented yet.")

    # Softmin masking: large positive bias pushes masked positions to domain max → exp(-2θ/τ) ≈ 0
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), _MASK_VAL * tau_m)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), _MASK_VAL * tau_m)
        else:
            # HF additive mask: negative for masked → negate for softmin convention
            attn_bias = attn_bias - attn_mask.clamp(min=-_MASK_VAL * tau_m, max=0.0)

    # Fixed domain for q, k: clamp inputs to [-θ, θ] so ψ_M spike times t_B = θ - k ≥ 0
    domain_qk = PotentialBounds(-theta, theta)
    q_exp = domain_qk.clamp(query).unsqueeze(-2)   # (B,H,L,1,D)
    k_exp = domain_qk.clamp(key).unsqueeze(-3)     # (B,H,1,S,D)

    # f_SDP(q,k) = ψ_M sum ≈ -(1/√d_k)·dot(q,k), broadcasted to (B,H,L,S)
    attn_score, _ = scaled_dot_product_function(q_exp, domain_qk, k_exp, domain_qk, theta)

    # Clamp score to [-θ, θ]; declare domain with +0.1 guard so t_out = domain.max - score ≥ 0.1,
    # preventing the float32/float64 precision gap at the exp(-2θ) boundary.
    _guard = 0.1
    attn_score = PotentialBounds(-theta, theta).clamp(attn_score + attn_bias)
    score_bound = PotentialBounds(-theta - _guard, theta + _guard)

    # softmin(f_SDP, τ_m) = softmax(−f_SDP/τ_m) = softmax(dot(q,k)/(τ_m·√d_k))
    attn_weight, _ = softmin_function(attn_score, score_bound, tau_s=tau_m)

    if dropout_p > 0.0:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)

    # Value 인코딩: φ_NP — 막 전위 → 스파이크 시각
    value_clamped = PotentialBounds(-theta, theta).clamp(value)
    t_v = theta - value_clamped                               # (B, H, S, D)
    domain_tv = TimeBounds(0.0, 2.0 * theta)

    # Attention weight 도메인: softmin 출력 ∈ [0, 1]
    domain_w = PotentialBounds(0.0, 1.0)

    # ψ_PWM(t_v[j], θ; w[i,j]) = w[i,j] * (θ − t_v[j]) = w[i,j] * v[j]
    # 브로드캐스트: (B,H,1,S,D) × (B,H,L,S,1) → (B,H,L,S,D)
    t_v_exp = t_v.unsqueeze(-4)          # (B, H, 1, S, D)
    w_exp   = attn_weight.unsqueeze(-1)  # (B, H, L, S, 1)

    out_per_sv, _ = pulse_width_modulation_operator(
        t_v_exp, domain_tv,
        theta,   theta,
        w_exp,   domain_w,
    )  # → (B, H, L, S, D)

    # S 차원 적분: Σ_j w[i,j] * v[j]  → (B, H, L, D)
    return out_per_sv.sum(dim=-2)

def spiking_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # Note: L2Net을 훈련하기 위해 사용하던 불필요한 로깅 제거 및 dropout 처리 정규화
    dropout_prob = dropout if module.training else 0.0
    
    attn_output = spiking_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout_prob,
        scale=scaling,
        is_causal=is_causal,
        tau_m=kwargs.get("tau_m", 1.0),
        theta=kwargs.get("theta", 10.0),
        **sdpa_kwargs,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
