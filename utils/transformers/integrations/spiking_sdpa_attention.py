import torch
import math
import wandb
from typing import cast

from transformers.utils.import_utils import is_torch_greater_or_equal
from transformers.utils import logging
from transformers.utils.import_utils import is_torch_npu_available, is_torch_xpu_available
from utils.transforms.functions import scaled_dot_product_function, softmin_function
from utils.transforms.noise import clamp_gaussian_output, get_gaussian_time_noise
from utils.transforms.potential_to_spike import neg_identity_transform
from utils.transforms.primitive import pulse_width_modulation_operator
from utils.transforms.types import PotentialBounds, SpikeSample, TimeBounds

logger = logging.get_logger(__name__)

# reciprocal_exp_operator의 실효 지수 범위는 2*cap; exp(-2*20) ≈ 2e-18
_SOFTMIN_CAP = 80.0

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


def _gaussian_attention_value_readout(
    value_clamped: torch.Tensor,
    attn_weight: torch.Tensor,
    domain_v: PotentialBounds,
) -> torch.Tensor:
    """Read attention values from Gaussian opening and reference events.

    Each value element supplies an opening event, while one scalar zero-reference
    event is shared by the entire attention invocation. A missing value event
    contributes reset zero. A missing reference event leaves every delivered value
    trajectory active until the fixed identity-code deadline. The resulting
    durations are contracted with attention weights without materializing the full
    query-by-key-by-feature synapse tensor.

    Args:
        value_clamped: Value tensor already restricted to the symmetric TTFS rail.
        attn_weight: Attention weights whose source dimension matches the values.
        domain_v: Fixed symmetric value domain defining the identity-code window.

    Returns:
        The physical observation-time attention output clamped to its conservative
        ideal summed rail envelope.

    Raises:
        RuntimeError: If either event-aware encoder call fails to return a
            ``SpikeSample``.
        ValueError: If value and reference events do not share one deadline.
    """
    # Every value element independently opens a weighted integration trajectory.
    # Its finite deadline carrier is never interpreted as a delivered event without
    # consulting the accompanying fired mask.
    value_event = neg_identity_transform(
        value_clamped,
        domain_v,
        return_spike_sample=True,
        noise_site="attention.value",
    )
    if not isinstance(value_event, SpikeSample):
        raise RuntimeError(
            "Gaussian attention value encoding must return SpikeSample"
        )

    # The zero codeword is a single physical closing event shared across batch,
    # heads, queries, source positions, and value features for this operator call.
    reference_event = neg_identity_transform(
        value_clamped.new_zeros(()),
        domain_v,
        return_spike_sample=True,
        noise_site="attention.value_reference",
    )
    if not isinstance(reference_event, SpikeSample):
        raise RuntimeError(
            "Gaussian attention value reference must return SpikeSample"
        )

    # Opening and closing events participate in the same integration interval, so
    # accepting different deadlines would make their physical duration undefined.
    if not math.isclose(
        float(value_event.domain.max),
        float(reference_event.domain.max),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Gaussian attention value and reference events require a shared deadline"
        )

    # A delivered reference closes every active trajectory at its sampled time. If
    # it misses, those trajectories continue until the inclusive observation limit.
    deadline = value_event.time.new_tensor(float(value_event.domain.max))
    stop_time = torch.where(
        reference_event.fired,
        reference_event.time,
        deadline,
    )

    # A missing value event never opens its trajectory and therefore contributes
    # reset zero. Delivered values retain signed durations if event order reverses.
    duration = torch.where(
        value_event.fired,
        stop_time - value_event.time,
        torch.zeros_like(value_event.time),
    )

    # Preserve the deterministic [0, 1] weight contract before contraction. Matmul
    # directly reduces the source dimension and avoids an (L, S, D) temporary.
    bounded_weight = attn_weight.clamp(0.0, 1.0)
    attn_output = torch.matmul(bounded_weight, duration)

    # Independent source weights conservatively sum at most one theta-wide value
    # per source position. Record raw saturation before enforcing these ideal rails.
    source_length = int(value_clamped.size(-2))
    theta = max(abs(float(domain_v.min)), abs(float(domain_v.max)))
    output_domain = PotentialBounds(
        -theta * source_length,
        theta * source_length,
    )
    return clamp_gaussian_output(
        attn_output,
        output_domain,
        site="attention.value_output",
        name="attention_value_output",
    )

def spiking_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    enable_gqa: bool = False,
    tau_m: float = 1.0,
    theta: float = 10.0,
    training: bool = False,
) -> torch.Tensor:
    """Evaluate scaled dot-product attention with spiking compositions.

    Query and key tensors are combined through the spiking multiplication and
    softmin chain. The resulting weights then select one of two value readouts:
    deterministic execution retains the explicit PWM tensor composition, while
    Gaussian execution delegates opening/reference event handling to
    :func:`_gaussian_attention_value_readout`.

    Args:
        query: Query tensor shaped ``(batch, heads, target, features)``.
        key: Key tensor shaped ``(batch, heads, source, features)``.
        value: Value tensor shaped ``(batch, heads, source, value_features)``.
        attn_mask: Optional boolean keep mask or additive suppression mask.
        dropout_p: Dropout probability applied to the normalized weights.
        is_causal: Whether to suppress source positions after each target index.
        enable_gqa: Request grouped-query attention, which is not implemented here.
        tau_m: Temporal scale used by the softmin composition.
        theta: Symmetric potential rail used by affine TTFS encoders.
        training: Training-state flag forwarded to deterministic value encoding.

    Returns:
        Attention output shaped ``(batch, heads, target, value_features)``.

    Raises:
        NotImplementedError: If grouped-query attention is requested directly.
    """

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
    if masked_pos is not None:
        attn_score = torch.where(masked_pos, _SOFTMIN_CAP, attn_score)

    # softmin(f_SDP, τ_m) = softmax(dot(q,k)/(τ_m·√d_k))
    attn_weight, _ = softmin_function(attn_score, score_bound, tau_s=tau_m, domain_shift=softmin_cap)

    # # Debug: Compare weights with torch.softmax
    # torch_logits_clamped = torch_logits.clamp(-softmin_cap, softmin_cap)
    # # attn_bias is positive for masked tokens in softmin convention
    # torch_weights = torch.nn.functional.softmax((torch_logits_clamped - attn_bias) / tau_m, dim=-1)
    # weight_error = (attn_weight - torch_weights).abs().max().item()
    # print(f"[DEBUG] Attn weight vs torch.softmax max diff: {weight_error:.6f}")

    if dropout_p > 0.0:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)

    # softmin 출력은 이론상 [0,1]이지만 float 오차로 미소 초과 가능 → clamp
    attn_weight = attn_weight.clamp(0.0, 1.0)

    # Value 인코딩: φ_NP — 막 전위 → 스파이크 시각. Both paths share the
    # identical fixed rail and clamped value tensor before selecting their readout.
    domain_v = PotentialBounds(-theta, theta)
    value_clamped = domain_v.clamp(value, name="value")

    # Gaussian timing noise changes the physical opening/closing event interval.
    # Keep those masks and deadline rules inside the dedicated value-readout helper.
    if get_gaussian_time_noise().enabled:
        return _gaussian_attention_value_readout(
            value_clamped,
            attn_weight,
            domain_v,
        )

    # Noise-free execution preserves the original explicit PWM composition and its
    # training keyword, without constructing event-delivery metadata.
    t_v, domain_tv = neg_identity_transform(
        value_clamped,
        domain_v,
        training=bool(training),
    )

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
