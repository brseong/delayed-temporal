import torch
import math
import wandb
from typing import cast

from transformers.utils.import_utils import is_torch_greater_or_equal
from transformers.utils import logging
from transformers.utils.import_utils import is_torch_npu_available, is_torch_xpu_available
from utils.transforms.functions import scaled_dot_product_function, softmin_function
from utils.transforms.potential_to_spike import neg_identity_transform
from utils.transforms.primitive import pulse_width_modulation_operator
from utils.transforms.types import PotentialBounds, TimeBounds

logger = logging.get_logger(__name__)

# Softmin masking: large positive suppresses masked positions (exp(-20) ≈ 2e-9)
# Use a value that stays within stable range for float32 exp but provides enough suppression.
_MASK_VAL = 20.0
# reciprocal_exp_operator의 실효 지수 범위는 2*cap; exp(-2*20) ≈ 2e-18
_SOFTMIN_CAP = 20.0

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
    is_causal: bool = False, enable_gqa=False, tau_m=1.0, theta=10.0,
    training=False) -> torch.Tensor:

    L, S = query.size(-2), key.size(-2)

    if enable_gqa:
        raise NotImplementedError("GQA is not implemented yet.")

    # Build a boolean mask of positions to suppress, then hard-overwrite scores at those positions.
    masked_pos = None
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        masked_pos = temp_mask.logical_not()
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # True means keep, False means masked.
            masked_from_attn = attn_mask.logical_not()
        else:
            # HF additive mask convention: masked positions are negative.
            masked_from_attn = attn_mask < 0
        masked_pos = masked_from_attn if masked_pos is None else (masked_pos | masked_from_attn)

    # Fixed domain for q, k: clamp inputs to [-θ, θ] so ψ_M spike times t_B = θ - k ≥ 0
    domain_qk = PotentialBounds(-theta, theta)
    q_exp = domain_qk.clamp(query, name="query").unsqueeze(-2)   # (B,H,L,1,D)
    k_exp = domain_qk.clamp(key, name="key").unsqueeze(-3)     # (B,H,1,S,D)

    # f_SDP(q,k) = ψ_M sum ≈ -(1/√d_k)·dot(q,k), broadcasted to (B,H,L,S)
    attn_score, _ = scaled_dot_product_function(q_exp, domain_qk, k_exp, domain_qk, theta)

    # # Debug: Compare scores with torch.matmul
    # head_dim = query.size(-1)
    # torch_logits = torch.matmul(domain_qk.clamp(query), domain_qk.clamp(key).transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
    # score_error = (attn_score + torch_logits).abs().max().item()
    # print(f"[DEBUG] Attn score vs -torch_logits max diff: {score_error:.6f}")

    # softmin chain의 실효 지수 범위는 2*cap이므로 float32 underflow 방지를 위해 cap.
    # exp(-2*_SOFTMIN_CAP) ≈ 5e-35 > float32_tiny; ±40 밖의 점수는 어차피 weight ≈ 0.
    softmin_cap = min(float(theta), _SOFTMIN_CAP)
    
    # Clamp the unmasked score range first.
    score_bound = PotentialBounds(-softmin_cap, softmin_cap)
    attn_score = score_bound.clamp(attn_score, name="attn_score")

    # Hard overwrite: force masked scores to a fixed suppressing value.
    mask_fill = _MASK_VAL * tau_m
    if masked_pos is not None:
        attn_score = torch.where(masked_pos, torch.full_like(attn_score, mask_fill), attn_score)

    # Ensure the declared domain contains both unclamped and overwritten values.
    score_bound_with_bias = PotentialBounds(score_bound.min, max(score_bound.max, float(mask_fill)))

    # softmin(f_SDP, τ_m) = softmax(dot(q,k)/(τ_m·√d_k))
    attn_weight, _ = softmin_function(attn_score, score_bound_with_bias, tau_s=tau_m)

    # # Debug: Compare weights with torch.softmax
    # torch_logits_clamped = torch_logits.clamp(-softmin_cap, softmin_cap)
    # # attn_bias is positive for masked tokens in softmin convention
    # torch_weights = torch.nn.functional.softmax((torch_logits_clamped - attn_bias) / tau_m, dim=-1)
    # weight_error = (attn_weight - torch_weights).abs().max().item()
    # print(f"[DEBUG] Attn weight vs torch.softmax max diff: {weight_error:.6f}")

    if dropout_p > 0.0:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)

    # Value 인코딩: φ_NP — 막 전위 → 스파이크 시각
    value_clamped = PotentialBounds(-theta, theta).clamp(value, name="value")
    t_v, domain_tv = neg_identity_transform(
        value_clamped,
        PotentialBounds(-theta, theta),
        training=bool(training),
    )

    # softmin 출력은 이론상 [0,1]이지만 float 오차로 미소 초과 가능 → clamp
    attn_weight = attn_weight.clamp(0.0, 1.0)
    domain_w = PotentialBounds(0.0, 1.0)

    # ψ_PWM(t_v[j], θ; w[i,j]) = w[i,j] * (θ − t_v[j]) = w[i,j] * v[j]
    # 브로드캐스트: (B,H,1,S,D) × (B,H,L,S,1) → (B,H,L,S,D)
    t_v = t_v.unsqueeze(-3)          # (B, H, 1, S, D)
    w   = attn_weight.unsqueeze(-1)  # (B, H, L, S, 1)

    out_per_sv, _ = pulse_width_modulation_operator(
        t_v, domain_tv,
        theta,   theta,
        w,   domain_w,
    )  # → (B, H, L, S, D)

    # S 차원 적분: Σ_j w[i,j] * v[j]  → (B, H, L, D)
    attn_output = out_per_sv.sum(dim=-2)

    # Debug: Compare output with torch @
    # torch_output = torch.matmul(torch_weights, value_clamped)
    # output_error = (attn_output - torch_output).abs().max().item()
    # print(f"[DEBUG] Final output vs torch.matmul max diff: {output_error:.6f}")

    return attn_output

def spiking_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
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
            n_rep = int(cast(int, module.num_key_value_groups))
            key = repeat_kv(key, n_rep)
            value = repeat_kv(value, n_rep)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    is_causal_flag = bool(is_causal) if is_causal is not None else bool(getattr(module, "is_causal", True))
    is_causal_flag = query.shape[2] > 1 and attention_mask is None and is_causal_flag

    is_tracing = getattr(torch.jit, "is_tracing", lambda: False)()
    if is_tracing and isinstance(is_causal_flag, torch.Tensor):
        is_causal_flag = is_causal_flag.item()

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
        is_causal=is_causal_flag,
        tau_m=kwargs.get("tau_m", 1.0),
        theta=kwargs.get("theta", 10.0),
        training=module.training,
        **sdpa_kwargs,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
