import torch
import math
import wandb
from functools import cache, lru_cache
from typing import cast

from transformers.utils.import_utils import is_torch_greater_or_equal
from transformers.utils import logging
from transformers.utils.import_utils import is_torch_npu_available, is_torch_xpu_available
from utils.transforms.functions import scaled_dot_product_function, softmin_function
from utils.transforms.noise import clamp_gaussian_output, get_gaussian_time_noise
from utils.transforms.potential_to_spike import neg_identity_transform
from utils.transforms.types import PotentialBounds, SpikeSample, TimeBounds
from utils.transformers.calibration import (
    calibrated_potential,
    model_calibration_is_bound,
    validate_symmetric_encoder_bounds,
)

logger = logging.get_logger(__name__)

# Reserve two natural-log units between the worst-case normalized weight and the
# dtype's minimum normal value. This is a configuration constant, not data-derived.
_SOFTMIN_LOG_SAFETY_MARGIN = 2.0

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


@lru_cache(maxsize=None, typed=True)
def attention_output_bounds(
    theta: float,
    source_length_max: int,
) -> PotentialBounds:
    """Return the fixed value interval for attention output.

    With noise and dropout disabled, normalized nonnegative weights preserve the
    encoded value interval ``[-theta, theta]``. Both execution paths enforce that
    same interval on the output; noisy weights and value readouts need not preserve
    a weighted average. Source capacity remains validated but does not widen the
    output interval. Identical configuration pairs reuse one immutable object.

    Args:
        theta: Positive finite magnitude of the symmetric value rail.
        source_length_max: Positive configured maximum number of source positions.

    Returns:
        The symmetric fixed output interval ``[-theta, theta]``.

    Raises:
        TypeError: If ``source_length_max`` is not an integer.
        ValueError: If either input cannot define finite, non-empty output rails.
    """
    # Keep source-capacity validation even though only theta defines this interval.
    theta_value = float(theta)
    if not math.isfinite(theta_value) or theta_value <= 0.0:
        raise ValueError("attention theta must be finite and positive")
    if isinstance(source_length_max, bool) or not isinstance(
        source_length_max,
        int,
    ):
        raise TypeError("attention source_length_max must be an integer")
    if source_length_max <= 0:
        raise ValueError("attention source_length_max must be positive")

    return PotentialBounds(-theta_value, theta_value)


@cache
def attention_score_representability_bounds(
    theta: float,
    tau: float,
    source_length_max: int,
    dtype: torch.dtype,
    *,
    cap_by_theta: bool = True,
) -> PotentialBounds:
    """Return the largest symmetric softmin score rail representable by a dtype.

    A score rail ``[-c, c]`` can produce a smallest normalized exponential weight
    near ``exp(-2c / tau) / source_length_max``. Requiring that value to stay above
    the dtype's minimum normal number, with a fixed logarithmic safety margin, gives
    a configuration-derived ceiling for calibration and fallback execution.

    Args:
        theta: Positive legacy score cap, used only when ``cap_by_theta`` is true.
        tau: Positive temporal scale used by exponential normalization.
        source_length_max: Configured maximum denominator population.
        dtype: Floating payload dtype used by the softmin implementation.
        cap_by_theta: Retain the legacy global threshold ceiling when true. Explicit
            calibrated ViT bounds use only the numerical representability ceiling.

    Returns:
        A symmetric immutable range limited by numerical representability and,
        optionally, the legacy global threshold.

    Raises:
        TypeError: If capacity or dtype has an invalid type.
        ValueError: If configuration is non-finite, non-positive, or leaves no
            positive representable score radius.
    """
    # Validate all configuration scalars before evaluating logarithms. The maximum
    # source length is used instead of the current request so one attention layer
    # retains the same cap for short and full-capacity sequences.
    theta_value = float(theta)
    tau_value = float(tau)
    if not isinstance(cap_by_theta, bool):
        raise TypeError("attention score cap_by_theta must be boolean")
    if not math.isfinite(theta_value) or theta_value <= 0.0:
        raise ValueError("attention score theta must be finite and positive")
    if not math.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("attention score tau must be finite and positive")
    if isinstance(source_length_max, bool) or not isinstance(source_length_max, int):
        raise TypeError("attention score source_length_max must be an integer")
    if source_length_max <= 0:
        raise ValueError("attention score source_length_max must be positive")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("attention score dtype must be a floating torch.dtype")

    # Work in log space so the ceiling calculation itself never constructs the tiny
    # exponential it is designed to protect. The source-capacity term accounts for
    # the largest denominator sum before the final normalized exponential readout.
    dtype_floor = float(torch.finfo(dtype).tiny)
    log_budget = (
        -math.log(dtype_floor)
        - math.log(float(source_length_max))
        - _SOFTMIN_LOG_SAFETY_MARGIN
    )
    representable_radius = 0.5 * tau_value * log_budget
    radius = min(theta_value, representable_radius) if cap_by_theta else representable_radius

    # A non-positive budget means this dtype and sequence capacity cannot represent
    # even a nontrivial symmetric softmin rail under the current temporal scale.
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            "attention score configuration has no positive representable radius"
        )
    return PotentialBounds(-radius, radius)


def _gaussian_attention_value_readout(
    value_clamped: torch.Tensor,
    attn_weight: torch.Tensor,
    domain_v: PotentialBounds,
    output_domain: PotentialBounds,
) -> torch.Tensor:
    """Read attention values from Gaussian opening and reference events.

    Each value element and one scalar zero-reference event supply the two causal
    rails of signed PWM value integration. Each missed event independently leaves
    its own rail at reset. The resulting signed pulse widths are contracted with
    attention weights without materializing the full query-by-key-by-feature
    synapse tensor.

    Args:
        value_clamped: Value tensor already restricted to the symmetric TTFS rail.
        attn_weight: Attention weights whose source dimension matches the values.
        domain_v: Fixed symmetric value domain defining the identity-code window.
        output_domain: Fixed output interval equal to the encoded value interval.

    Returns:
        The physical observation-time attention output clamped to the fixed value
        interval after recording the raw output saturation counts.

    Raises:
        RuntimeError: If either event-aware encoder call fails to return a
            ``SpikeSample``.
        ValueError: If value and reference events do not share one deadline.
    """
    # Every value element independently supplies one weighted causal rail. Its finite
    # deadline carrier is never interpreted as a delivered event without consulting
    # the accompanying fired mask.
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

    # The zero codeword supplies one physical reference rail shared across batch,
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

    # Both rails participate in one differential readout, so accepting different
    # deadlines would make their common-reference subtraction undefined.
    if not math.isclose(
        float(value_event.domain.max),
        float(reference_event.domain.max),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Gaussian attention value and reference events require a shared deadline"
        )

    # Convert the two sampled events into causal pulse widths against their common
    # deadline. Each miss suppresses only its own rail, so a surviving value or
    # reference rail remains visible with the correct sign at observation time.
    deadline = value_event.time.new_tensor(float(value_event.domain.max))
    value_pulse_width = torch.where(
        value_event.fired,
        (deadline - value_event.time).clamp_min(0.0),
        torch.zeros_like(value_event.time),
    )
    reference_pulse_width = torch.where(
        reference_event.fired,
        (deadline - reference_event.time).clamp_min(0.0),
        torch.zeros_like(reference_event.time),
    )
    signed_pulse_width = value_pulse_width - reference_pulse_width

    # Preserve the deterministic [0, 1] drive contract before the optimized PWM-MAC.
    # Conceptually, every unmaterialized query/source/value-feature synapse is:
    #
    # pwm_qsd, _ = signed_pulse_width_modulation_operator(
    #     value_event_sd, value_event.domain,
    #     reference_event, reference_event.domain,
    #     attn_weight_qs, PotentialBounds(0.0, 1.0),
    #     observation_deadline=float(value_event.domain.max),
    # )
    # attn_output_qd = sum_s(pwm_qsd)
    #
    # Matmul evaluates that full source reduction without an (L, S, D) temporary.
    bounded_weight = attn_weight.clamp(0.0, 1.0)
    attn_output = torch.matmul(bounded_weight, signed_pulse_width)

    # Record raw saturation before enforcing the fixed value interval. Do not
    # renormalize noisy weights or change their preceding event computation.
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
    tau: float = 1.0,
    theta: float = 10.0,
    training: bool = False,
    source_length_max: int | None = None,
    score_calibration_module: torch.nn.Module | None = None,
    *,
    query_bounds: PotentialBounds | None = None,
    key_bounds: PotentialBounds | None = None,
    value_bounds: PotentialBounds | None = None,
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
        tau: Temporal scale used by the softmin composition.
        theta: Global threshold used for legacy input bounds when explicit bounds
            are omitted. Explicit bounds set their own fixed encoding intervals.
        training: Training-state flag forwarded to deterministic value encoding.
        source_length_max: Configured source-position maximum used for capacity
            validation and the representable score interval, not output widening.
        score_calibration_module: Optional bound attention module owning the frozen
            ``attention_score`` calibration site.
        query_bounds: Explicit symmetric query bounds. All three input bounds must
            be supplied together; omission preserves the legacy global interval.
        key_bounds: Explicit symmetric bounds defining the key encoding interval.
        value_bounds: Explicit symmetric bounds shared by value encoding and output.

    Returns:
        Attention output shaped ``(batch, heads, target, value_features)``.

    Raises:
        NotImplementedError: If grouped-query attention is requested directly.
        ValueError: If the configured source maximum is absent or shorter than the
            current key/value sequence.
    """

    L, S = query.size(-2), key.size(-2)

    if enable_gqa:
        raise NotImplementedError("GQA is not implemented yet.")

    # Capacity remains explicit for validation and the fixed score interval.
    # Output bounds never depend on the current tensor length.
    if source_length_max is None:
        raise ValueError("attention source_length_max must be configured")
    output_domain = attention_output_bounds(theta, source_length_max)
    supplied_bounds = (query_bounds, key_bounds, value_bounds)
    explicit_bounds = all(bounds is not None for bounds in supplied_bounds)
    if any(bounds is not None for bounds in supplied_bounds) and not explicit_bounds:
        raise ValueError("attention query, key, and value bounds must be supplied together")
    if explicit_bounds:
        domain_q = validate_symmetric_encoder_bounds(
            query_bounds, query.dtype, name="query",
        )
        domain_k = validate_symmetric_encoder_bounds(
            key_bounds, key.dtype, name="key",
        )
        domain_v = validate_symmetric_encoder_bounds(
            value_bounds, value.dtype, name="value",
        )
        output_domain = domain_v
        key_encoder_radius = float(domain_k.max)
        value_reference_time = float(domain_v.max)
    else:
        domain_q = domain_k = domain_v = PotentialBounds(-theta, theta)
        key_encoder_radius = value_reference_time = float(theta)
    if int(value.size(-2)) != S:
        raise ValueError("attention key and value source lengths must match")
    if S > source_length_max:
        raise ValueError(
            "attention source length exceeds configured source_length_max"
        )

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

    # The key interval also sets multiplication's encoder radius: passing global
    # theta here would silently clamp keys again after their calibrated input clamp.
    q_exp = domain_q.clamp(query, name="query").unsqueeze(-2)   # (B,H,L,1,D)
    k_exp = domain_k.clamp(key, name="key").unsqueeze(-3)     # (B,H,1,S,D)

    # f_SDP(q,k) = ψ_M sum ≈ -(1/√d_k)·dot(q,k), broadcasted to (B,H,L,S)
    attn_score, analytic_score_bound = scaled_dot_product_function(
        q_exp,
        domain_q,
        k_exp,
        domain_k,
        key_encoder_radius,
    )

    # # Debug: Compare scores with torch.matmul
    # head_dim = query.size(-1)
    # torch_logits = torch.matmul(domain_q.clamp(query), domain_k.clamp(key).transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
    # score_error = (attn_score + torch_logits).abs().max().item()
    # print(f"[DEBUG] Attn score vs -torch_logits max diff: {score_error:.6f}")

    # Derive the collection safety cap from dtype, temporal scale, and configured
    # source capacity. It is static for this module and prevents the downstream
    # normalized exponential from underflowing even before an artifact is installed.
    representability_ceiling = attention_score_representability_bounds(
        float(theta),
        float(tau),
        source_length_max,
        attn_score.dtype,
        cap_by_theta=not explicit_bounds,
    )

    # Intersect the numerical ceiling with the symmetric portion of the operator's
    # static analytic interval. Normally the dot-product envelope is much broader;
    # this intersection also handles small theta configurations without returning a
    # collection rail wider than the analytic bound that validated the raw score.
    execution_radius = min(
        float(representability_ceiling.max),
        float(analytic_score_bound.max),
        -float(analytic_score_bound.min),
    )
    if not math.isfinite(execution_radius) or execution_radius <= 0.0:
        raise ValueError("attention score execution radius must be finite and positive")
    representable_score_bound = PotentialBounds(
        -execution_radius,
        execution_radius,
    )

    # A bound attention module observes raw pre-clamp scores during collection while
    # executing on the representable ceiling. Frozen phases instead clamp directly to
    # the persisted layer quantile and verify it does not exceed that same ceiling.
    if score_calibration_module is not None and model_calibration_is_bound(
        score_calibration_module
    ):
        calibrated_score = calibrated_potential(
            score_calibration_module,
            "attention_score",
            attn_score,
            collection_bounds=analytic_score_bound,
            collection_execution_bounds=representable_score_bound,
        )
        attn_score = calibrated_score.value
        score_bound = calibrated_score.domain
    else:
        score_bound = representable_score_bound
        attn_score = score_bound.clamp(attn_score, name="attn_score")

    softmin_cap = float(score_bound.max)

    # Hard overwrite: force masked scores to a fixed suppressing value.
    if masked_pos is not None:
        attn_score = torch.where(masked_pos, softmin_cap, attn_score)

    # softmin(f_SDP, τ) = softmax(dot(q,k)/(τ·√d_k))
    attn_weight, _ = softmin_function(attn_score, score_bound, tau=tau)

    # # Debug: Compare weights with torch.softmax
    # torch_logits_clamped = torch_logits.clamp(-softmin_cap, softmin_cap)
    # # attn_bias is positive for masked tokens in softmin convention
    # torch_weights = torch.nn.functional.softmax((torch_logits_clamped - attn_bias) / tau, dim=-1)
    # weight_error = (attn_weight - torch_weights).abs().max().item()
    # print(f"[DEBUG] Attn weight vs torch.softmax max diff: {weight_error:.6f}")

    # Training-time attention dropout is retained only for Hugging Face API compatibility.
    # The paper and maintained evaluator make fixed-range claims only with dropout disabled.
    if dropout_p > 0.0:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)

    # softmin 출력은 이론상 [0,1]이지만 float 오차로 미소 초과 가능 → clamp
    attn_weight = attn_weight.clamp(0.0, 1.0)

    # Value 인코딩: φ_NP — 막 전위 → 스파이크 시각. Both paths share the
    # identical fixed rail and clamped value tensor before selecting their readout.
    value_clamped = domain_v.clamp(value, name="value")

    # Gaussian timing noise changes the physical opening/closing event interval.
    # Keep those masks and deadline rules inside the dedicated value-readout helper.
    if get_gaussian_time_noise().enabled:
        return _gaussian_attention_value_readout(
            value_clamped,
            attn_weight,
            domain_v,
            output_domain,
        )

    # Noise-free execution preserves temporal encoding and its training keyword,
    # then recovers the delivered signed pulse width without event metadata.
    value_time, _ = neg_identity_transform(
        value_clamped,
        domain_v,
        training=bool(training),
    )
    signed_pulse_width = value_reference_time - value_time

    # The optimized matrix multiplication evaluates the complete deterministic
    # PWM reduction. Conceptually every source/value-feature term is:
    #
    # pwm_qsd, _ = signed_pulse_width_modulation_operator(
    #     value_time_sd, value_time_domain,
    #     value_reference_time, value_reference_time,
    #     attn_weight_qs, PotentialBounds(0.0, 1.0),
    #     observation_deadline=float(domain_v.range),
    # )
    # attn_output_qd = sum_s(pwm_qsd)
    #
    # Matmul avoids materializing the full (B,H,L,S,D) synapse tensor.
    attn_output = torch.matmul(attn_weight, signed_pulse_width)

    # Noise-free and Gaussian execution share the same configured physical rail.
    # Clamping here keeps the returned tensor consistent with the domain that model
    # adapters will attach in the following integration step.
    return clamp_gaussian_output(
        attn_output,
        output_domain,
        site="attention.value_output",
        name="attention_value_output",
    )

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

    # Reject the removed attention-specific names instead of silently ignoring them
    # through this permissive Hugging Face wrapper. Attention exposes only `tau`.
    if "tau_m" in kwargs or "tau_s" in kwargs:
        raise TypeError("spiking attention accepts only the unified tau keyword")

    # Prefer an evaluator-supplied fixed maximum. Standard text models otherwise
    # use their configured position capacity, which remains request-independent.
    source_length_max = kwargs.get("source_length_max")
    config = getattr(module, "config", None)
    if source_length_max is None and config is not None:
        source_length_max = getattr(config, "max_position_embeddings", None)

    # ViT has no max-position field, so derive its fixed token capacity from the
    # configured image and patch geometry, including the class token.
    if source_length_max is None and config is not None:
        image_size = getattr(config, "image_size", None)
        patch_size = getattr(config, "patch_size", None)
        if image_size is not None and patch_size is not None:
            image_hw = (
                (image_size, image_size)
                if isinstance(image_size, int)
                else tuple(image_size)
            )
            patch_hw = (
                (patch_size, patch_size)
                if isinstance(patch_size, int)
                else tuple(patch_size)
            )
            source_length_max = (
                (int(image_hw[0]) // int(patch_hw[0]))
                * (int(image_hw[1]) // int(patch_hw[1]))
                + 1
            )
    
    attn_output = spiking_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout_prob,
        is_causal=is_causal_flag,
        tau=kwargs.get("tau", 1.0),
        theta=kwargs.get("theta", 10.0),
        training=module.training,
        source_length_max=source_length_max,
        score_calibration_module=module,
        query_bounds=kwargs.get("query_bounds"),
        key_bounds=kwargs.get("key_bounds"),
        value_bounds=kwargs.get("value_bounds"),
        **sdpa_kwargs,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
