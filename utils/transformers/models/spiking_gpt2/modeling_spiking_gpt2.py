# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import initialization as init
from transformers.activations import ACT2FN, get_activation
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.utils import (
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config

from utils.transforms import neg_identity_transform
from utils.transforms.functions import gelu_approximation
from utils.transforms.noise import clamp_gaussian_output, get_gaussian_time_noise
from utils.transforms.types import Potential, PotentialBounds, SpikeSample
from utils.transformers.calibration import (
    calibrated_potential,
    model_calibration_is_bound,
)
from utils.transformers.integrations.spiking_sdpa_attention import attention_output_bounds
from utils.transformers.models.spiking_ops import (
    SpikingLayerNorm,
    SpikingLinear,
    _apply_norm,
    _validate_pwm_input_domain,
)

logger = logging.get_logger(__name__)


class SpikingConv1D(Conv1D):
    def __init__(self, nf, nx, theta=400.0, **kwargs):
        super().__init__(nf, nx, **kwargs)
        self.theta = theta

    def freeze_parameter_bounds(
        self,
        input_domain: PotentialBounds,
        *,
        refresh: bool = False,
    ) -> PotentialBounds:
        """Memoize GPT-2's affine output domain for one fixed input interval.

        Hugging Face stores ``Conv1D`` weights as ``[in_features, out_features]``.
        For input interval ``[l, u]``, each weight contributes ``min(W_ij*l, W_ij*u)``
        or ``max(W_ij*l, W_ij*u)`` to the corresponding output endpoint. Bias then
        translates each feature interval before reduction to one immutable domain.

        Args:
            input_domain: Immutable analytic or calibrated input rail containing zero.
            refresh: Recompute after an intentional parameter update and discard
                entries for prior domains. The default rejects parameter mutation.

        Returns:
            The immutable module-wide transposed affine output domain.

        Raises:
            RuntimeError: If parameters changed after memoization without refresh,
                or changed while bounds were being calculated.
            ValueError: If the input domain or a derived output endpoint is invalid.

        Notes:
            Mutation checks use PyTorch parameter version counters. Standard
            ``torch.no_grad()`` in-place changes are detected; direct ``.data`` writes
            bypass that bookkeeping and are unsupported.
        """
        # Treat the upstream fixed rail as the encoder and cache identity. The scalar
        # zero reference must remain representable inside that same temporal window.
        lower_input, upper_input = _validate_pwm_input_domain(
            input_domain,
            operator_name="SpikingConv1D",
        )
        domain_key = (lower_input, upper_input)

        # Parameter versions define a cache generation; multiple calibrated domain
        # endpoints can be memoized without rescanning a stable projection matrix.
        versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
        )
        cached = self.__dict__.get("_frozen_parameter_bounds")

        # Reject an unapproved parameter transition. Explicit refresh begins a new
        # coherent generation and intentionally removes every older domain entry.
        memoized_domains: dict[tuple[float, float], PotentialBounds]
        if cached is not None:
            cached_versions, memoized_domains = cached
            if versions != cached_versions and not refresh:
                raise RuntimeError(
                    "SpikingConv1D parameters changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            if (
                versions == cached_versions
                and not refresh
                and domain_key in memoized_domains
            ):
                return memoized_domains[domain_key]
            if refresh:
                memoized_domains = {}
        else:
            memoized_domains = {}

        # Conv1D's first weight axis is fan-in. Float64 interval arithmetic over that
        # axis selects endpoints by weight sign without a symmetric-radius relaxation.
        weight = self.weight.detach().to(dtype=torch.float64)
        lower_terms = torch.minimum(weight * lower_input, weight * upper_input)
        upper_terms = torch.maximum(weight * lower_input, weight * upper_input)
        if self.bias is None:
            bias = torch.zeros(self.nf, dtype=torch.float64, device=weight.device)
        else:
            bias = self.bias.detach().to(dtype=torch.float64)
        lower = lower_terms.sum(dim=0) + bias
        upper = upper_terms.sum(dim=0) + bias
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("SpikingConv1D parameter-derived bounds must be finite")
        output_domain = PotentialBounds(lower.min().item(), upper.max().item())

        # Check for concurrent writes before publishing the scalar endpoints. This
        # guarantees the cache describes one coherent parameter version.
        final_versions = (
            self.weight._version,
            self.bias._version if self.bias is not None else None,
        )
        if final_versions != versions:
            raise RuntimeError(
                "SpikingConv1D parameters changed while bounds were being frozen"
            )

        # Keep derived metadata outside the state dict and publish a fresh mapping so
        # every earlier immutable domain remains valid for this parameter generation.
        memoized_domains = {**memoized_domains, domain_key: output_domain}
        self.__dict__["_frozen_parameter_bounds"] = (
            final_versions,
            memoized_domains,
        )
        return output_domain

    def _gaussian_forward(
        self,
        x: torch.Tensor,
        encoded_x: torch.Tensor,
        domain_x: PotentialBounds,
        output_domain: PotentialBounds,
    ) -> Potential:
        """Evaluate GPT-2's transposed affine projection from sampled events.

        Hugging Face ``Conv1D`` stores weights as ``[in_features, out_features]``.
        This method preserves that layout while replacing each encoded input with a
        signed pair of causal PWM pulse widths. One scalar zero-reference event is
        shared by the complete projection call, and each missed data or reference
        event independently leaves its own rail at reset until the common deadline.

        Args:
            x: Original input tensor used for metadata and scalar allocation.
            encoded_x: Input clamped to the fixed identity-code domain.
            domain_x: Fixed zero-containing analytic or calibrated input rails.
            output_domain: Frozen parameter-derived projection output rail.

        Returns:
            A bounded ``Potential`` with the original leading dimensions and ``nf``
            output features.

        Raises:
            RuntimeError: If either event-aware encoder call fails to return a
                ``SpikeSample``.
        """
        # Each final-dimension input element independently opens its projection
        # trajectory; the fired mask preserves misses across arbitrary leading shapes.
        data_event = neg_identity_transform(
            encoded_x,
            domain_x,
            return_spike_sample=True,
            noise_site="conv1d.data",
        )
        if not isinstance(data_event, SpikeSample):
            raise RuntimeError(
                "Gaussian SpikingConv1D encoding must return SpikeSample"
            )

        # One scalar zero codeword supplies the shared reference rail for the entire
        # GPT-2 projection call instead of being resampled per token or feature.
        reference_event = neg_identity_transform(
            x.new_zeros(()),
            domain_x,
            return_spike_sample=True,
            noise_site="conv1d.reference",
        )
        if not isinstance(reference_event, SpikeSample):
            raise RuntimeError(
                "Gaussian SpikingConv1D reference must return SpikeSample"
            )

        # Convert both sampled events into causal pulse widths against the shared
        # deadline. A one-sided miss leaves the surviving rail visible with its sign;
        # no event ordering, additional sampling, or per-token reference is introduced.
        deadline = data_event.time.new_tensor(float(data_event.domain.max))
        data_pulse_width = torch.where(
            data_event.fired,
            (deadline - data_event.time).clamp_min(0.0),
            torch.zeros_like(data_event.time),
        )
        reference_pulse_width = torch.where(
            reference_event.fired,
            (deadline - reference_event.time).clamp_min(0.0),
            torch.zeros_like(reference_event.time),
        )
        signed_pulse_width = data_pulse_width - reference_pulse_width

        # The optimized matrix kernel evaluates the complete transposed PWM-MAC.
        # Conceptually, each unmaterialized input/output synapse is equivalent to:
        #
        # pwm_ij, _ = signed_pulse_width_modulation_operator(
        #     data_event_i, data_event.domain,
        #     reference_event, reference_event.domain,
        #     self.weight[i, j], weight_domain,
        #     observation_deadline=float(data_event.domain.max),
        # )
        # y_j = sum_i(pwm_ij) + bias_j
        #
        # Flatten only leading dimensions for addmm; no synapse tensor is materialized.
        output_shape = signed_pulse_width.size()[:-1] + (self.nf,)
        flat_pulse_width = signed_pulse_width.reshape(
            -1,
            signed_pulse_width.size(-1),
        )
        if self.bias is None:
            y = torch.matmul(flat_pulse_width, self.weight)
        else:
            y = torch.addmm(self.bias, flat_pulse_width, self.weight)
        y = y.view(output_shape)

        # Record raw saturation against the frozen output-specific safety rail before
        # returning the bounded potential. Timing misses may change the observation-
        # time value but never mutate the declared projection domain.
        return Potential(
            clamp_gaussian_output(
                y,
                output_domain,
                site="conv1d.output",
                name="conv1d_y",
            ),
            output_domain,
        )

    def forward(self, input: Potential) -> Potential:
        """Apply GPT-2's transposed affine projection through PWM integration.

        Common input calibration and one frozen output-specific parameter rail
        precede dispatch to the event-aware or delivered-time PWM path. Both modes
        use the optimized transposed matrix contraction and preserve Hugging Face's
        ``[in_features, out_features]`` layout, leading dimensions, bias, and bounds.

        Args:
            input: GPT-2 activation tensor paired with upstream potential bounds.

        Returns:
            The projected activation paired with conservative ideal potential rails.
        """
        # The upstream Potential owns the fixed encoder rail. Validate the shared
        # zero reference before clamping or consulting parameter-derived bounds.
        x: torch.Tensor = input.value
        domain_x = input.domain
        _validate_pwm_input_domain(domain_x, operator_name="SpikingConv1D")
        encoded_x = domain_x.clamp(x, name="conv1d_x")

        # Freeze transposed-weight and bias bounds after checkpoint loading or static
        # perturbation. Later calls reuse the exact domain after mutation validation.
        output_domain = self.freeze_parameter_bounds(domain_x)

        # Isolate event sampling, shared-reference timing, addmm, and saturation
        # logging inside the private Gaussian implementation.
        if get_gaussian_time_noise().enabled:
            return self._gaussian_forward(
                x,
                encoded_x,
                domain_x,
                output_domain,
            )

        # Convert delivered identity-code times to signed zero-reference pulse widths.
        # This preserves explicit temporal encoding without broadcasting an extra
        # output-feature dimension or materializing individual synaptic products.
        data_time, _ = neg_identity_transform(encoded_x, domain_x)
        reference_time, _ = neg_identity_transform(x.new_zeros(()), domain_x)
        signed_pulse_width = reference_time - data_time

        # Flatten arbitrary leading dimensions for the optimized transposed PWM-MAC.
        # Conceptually each unmaterialized term is:
        # pwm_ij, _ = signed_pulse_width_modulation_operator(
        #     data_time_i, data_time_domain,
        #     reference_time, data_time_domain,
        #     self.weight[i, j], weight_domain,
        #     observation_deadline=float(data_time_domain.max),
        # )
        # with the input-feature dimension reduced by the matrix contraction.
        output_shape = signed_pulse_width.size()[:-1] + (self.nf,)
        flat_pulse_width = signed_pulse_width.reshape(
            -1,
            signed_pulse_width.size(-1),
        )
        if self.bias is None:
            y = torch.matmul(flat_pulse_width, self.weight)
        else:
            y = torch.addmm(self.bias, flat_pulse_width, self.weight)
        y = y.view(output_shape)

        # Addmm already applies learned bias, and the frozen domain already contains
        # its endpoint translation. No parameter scan occurs in this forward pass.
        return Potential(y, output_domain)


def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        if is_cross_attention:
            raise NotImplementedError("GPT2Attention does not support cross-attention in the spiking backend.")
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        _theta = getattr(config, "theta", 400.0)
        self.c_attn = SpikingConv1D(3 * self.embed_dim, self.embed_dim, theta=_theta)
        self.c_proj = SpikingConv1D(self.embed_dim, self.embed_dim, theta=_theta)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with maybe_autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Potential,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[Potential, torch.Tensor | None]:
        """Apply cache-aware GPT-2 attention with analytic domain propagation.

        The combined Q/K/V projection supplies the eager value envelope. Spiking
        attention replaces it with the memoized rail derived from ``theta`` and
        ``max_position_embeddings``. Projection and dropout then propagate that
        fixed envelope analytically, eliminating both runtime output-extrema ranges
        previously constructed inside this method.

        Args:
            hidden_states: Input tensor paired with its declared potential bounds.
            past_key_values: Optional cache carrying prior key and value tensors.
            cache_position: Positions used when appending to the cache.
            attention_mask: Optional causal or additive attention mask.
            output_attentions: Whether the selected backend should expose weights.
            **kwargs: Additional Hugging Face attention controls.

        Returns:
            The projected attention output with its fixed propagated domain and the
            optional attention weights returned by the selected backend.
        """
        # Keep the combined projection domain before splitting its tensor payload.
        # The single envelope conservatively contains the value slice used by eager
        # attention and remains valid for values read back from the same model cache.
        projected_qkv = self.c_attn(hidden_states)
        query_states, key_states, value_states = projected_qkv.value.split(
            self.split_size,
            dim=2,
        )
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

        # Eager and standard dense attention begin with the projected-value range.
        # Spiking attention replaces this below with its fixed physical output rail.
        using_eager = self.config._attn_implementation == "eager"
        using_spiking = self.config._attn_implementation == "spiking_sdpa"
        context_domain = projected_qkv.domain
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            if using_spiking:
                theta = float(getattr(self.config, "theta", 10.0))
                source_length_max = int(self.config.max_position_embeddings)
                kwargs["theta"] = theta
                kwargs["tau_m"] = getattr(self.config, "tau_s", 1.0)
                kwargs["source_length_max"] = source_length_max
                context_domain = attention_output_bounds(theta, source_length_max)

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        # Dense attention dropout independently removes normalized weights and scales
        # survivors by 1/(1-p). Include zero and both scaled endpoints. The spiking
        # backend already clamps its noisy or dropped readout to the fixed rail.
        if not using_spiking and self.training and self.attn_dropout.p > 0.0:
            if self.attn_dropout.p >= 1.0:
                context_domain = PotentialBounds(0.0, 0.0)
            else:
                attention_scale = 1.0 / (1.0 - self.attn_dropout.p)
                attention_candidates = (
                    0.0,
                    float(context_domain.min) * attention_scale,
                    float(context_domain.max) * attention_scale,
                )
                context_domain = PotentialBounds(
                    min(attention_candidates),
                    max(attention_candidates),
                )

        # Head merging changes layout only. The output projection consumes the fixed
        # attention range and returns its own interval-arithmetic affine envelope.
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output_pot = self.c_proj(Potential(attn_output, context_domain))
        out_val = self.resid_dropout(attn_output_pot.value)

        # Residual dropout is identity during evaluation. Training includes zero and
        # scales both affine endpoints without observing the realized dropout mask.
        output_domain = attn_output_pot.domain
        if self.training and self.resid_dropout.p > 0.0:
            if self.resid_dropout.p >= 1.0:
                output_domain = PotentialBounds(0.0, 0.0)
            else:
                residual_scale = 1.0 / (1.0 - self.resid_dropout.p)
                residual_candidates = (
                    0.0,
                    float(output_domain.min) * residual_scale,
                    float(output_domain.max) * residual_scale,
                )
                output_domain = PotentialBounds(
                    min(residual_candidates),
                    max(residual_candidates),
                )

        return Potential(out_val, output_domain), attn_weights


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        _theta = getattr(config, "theta", 400.0)
        # SpikingConv1D preserves the Hugging Face Conv1D parameter layout. Dense
        # ablation calls its inherited tensor forward directly, while both paths can
        # reuse the same transposed-weight interval cache.
        self.c_fc = SpikingConv1D(intermediate_size, embed_dim, theta=_theta)
        self.c_proj = SpikingConv1D(embed_dim, intermediate_size, theta=_theta)
        self._activation_name = str(config.activation_function)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Potential) -> Potential:
        """Apply GPT-2's feed-forward network with fixed analytic ranges.

        Dense and spiking projections share pretrained parameters and exact frozen
        affine intervals. ReLU and Tanh map endpoints directly; GELU-family and SiLU
        activations multiply their input by a gate in ``[0, 1]``. Evaluation dropout
        preserves the projection range, while training scaling is analytic.

        Args:
            hidden_states: Pre-normalized block activation with fixed bounds.

        Returns:
            Feed-forward output paired with a batch-independent range.

        Raises:
            ValueError: If the activation or dropout has no maintained fixed rule.
        """
        # Select only numerical projection execution. The dense branch remains the
        # inherited Hugging Face Conv1D operation and attaches the same frozen range.
        if self.use_spiking_mlp:
            projected = self.c_fc(hidden_states)
        else:
            projected = Potential(
                Conv1D.forward(self.c_fc, hidden_states.value),
                self.c_fc.freeze_parameter_bounds(hidden_states.domain),
            )

        # Every maintained GPT-2 activation has a standard envelope derived from the
        # affine endpoints. Unknown custom functions must provide an explicit rule
        # instead of restoring output-tensor extrema.
        activated_value = self.act(projected.value)
        if self._activation_name == "relu":
            activated_domain = PotentialBounds(
                max(0.0, float(projected.domain.min)),
                max(0.0, float(projected.domain.max)),
            )
        elif self._activation_name == "tanh":
            activated_domain = PotentialBounds(
                math.tanh(float(projected.domain.min)),
                math.tanh(float(projected.domain.max)),
            )
        elif self._activation_name in {
            "gelu",
            "gelu_fast",
            "gelu_new",
            "gelu_pytorch_tanh",
            "quick_gelu",
            "silu",
            "swish",
        }:
            activated_domain = PotentialBounds(
                min(float(projected.domain.min), 0.0),
                max(float(projected.domain.max), 0.0),
            )
        else:
            raise ValueError(
                "GPT-2 MLP activation requires a maintained analytic range rule"
            )
        activated = Potential(activated_value, activated_domain)

        # The output projection follows the same dense/spiking split as c_fc while
        # retaining one parameter-derived interval contract.
        if self.use_spiking_mlp:
            projected_output = self.c_proj(activated)
        else:
            projected_output = Potential(
                Conv1D.forward(self.c_proj, activated.value),
                self.c_proj.freeze_parameter_bounds(activated.domain),
            )

        # Dropout is identity in maintained evaluation. Include zero and inverse keep
        # scaling for completeness without measuring a realized training mask.
        x = self.dropout(projected_output.value)
        dropout_probability = float(self.dropout.p)
        if not math.isfinite(dropout_probability) or not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("GPT-2 MLP dropout probability must lie in [0, 1]")
        output_domain = projected_output.domain
        if self.training and dropout_probability > 0.0:
            if dropout_probability >= 1.0:
                output_domain = PotentialBounds(0.0, 0.0)
            else:
                scale = 1.0 / (1.0 - dropout_probability)
                candidates = (
                    0.0,
                    float(output_domain.min) * scale,
                    float(output_domain.max) * scale,
                )
                output_domain = PotentialBounds(
                    min(candidates),
                    max(candidates),
                )
        return Potential(x, output_domain)


class GPT2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        _theta = getattr(config, "theta", 10.0)
        _tau_s = getattr(config, "tau_s", 1.0)
        _use_spiking_ln = getattr(config, "use_spiking_layernorm", True)
        if _use_spiking_ln:
            _sln_kwargs = dict(
                theta=_theta, tau_s=_tau_s,
                use_spiking_mul=getattr(config, "spiking_ln_mul", True),
                use_spiking_log=getattr(config, "spiking_ln_log", True),
                use_spiking_expdiff=getattr(config, "spiking_ln_expdiff", True),
            )
            self.ln_1 = SpikingLayerNorm(hidden_size, eps=config.layer_norm_epsilon, **_sln_kwargs)
            self.ln_2 = SpikingLayerNorm(hidden_size, eps=config.layer_norm_epsilon, **_sln_kwargs)
            if config.add_cross_attention:
                self.ln_cross_attn = SpikingLayerNorm(hidden_size, eps=config.layer_norm_epsilon, **_sln_kwargs)
        else:
            self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            if config.add_cross_attention:
                self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Potential,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> Potential:
        """Apply one pre-norm GPT-2 block with optional frozen residual ranges.

        Without calibration, residual additions use exact interval endpoint sums.
        A bound collector observes raw self-attention and final MLP residuals on those
        analytic safety rails; frozen execution counts and clamps excursions against
        persisted per-block ranges so intervals do not grow recursively with depth.

        Args:
            hidden_states: Incoming residual stream and its fixed range.
            past_key_values: Optional autoregressive key/value cache.
            cache_position: Absolute positions represented in the cache update.
            attention_mask: Causal or additive attention mask.
            encoder_hidden_states: Unsupported cross-attention source input.
            encoder_attention_mask: Unsupported cross-attention source mask.
            use_cache: Whether attention updates and returns cache state.
            **kwargs: Existing attention backend arguments.

        Returns:
            Block output on an analytic or frozen calibrated residual range.
        """
        # Pre-norm attention retains the incoming residual value and interval. The
        # attention projection supplies an independent fixed range for endpoint sum.
        residual = hidden_states.value
        residual_domain = hidden_states.domain
        pot = _apply_norm(self.ln_1, hidden_states)
        attn_output, _ = self.attn(
            pot,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        # The self-attention residual is the first range-reset boundary in each block.
        # Collection observes the raw sum before any frozen clamp is applied.
        x = attn_output.value + residual
        attention_residual_bounds = PotentialBounds(
            float(residual_domain.min) + float(attn_output.domain.min),
            float(residual_domain.max) + float(attn_output.domain.max),
        )
        if model_calibration_is_bound(self):
            hidden_states = calibrated_potential(
                self,
                "attention_residual",
                x,
                collection_bounds=attention_residual_bounds,
            )
        else:
            hidden_states = Potential(x, attention_residual_bounds)

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states.value
            pot = _apply_norm(self.ln_cross_attn, hidden_states)
            cross_attn_output, _ = self.crossattention(
                pot,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            # Cross-attention is not constructed by the maintained GPT-2 backend.
            # Keep endpoint arithmetic explicit so this unreachable compatibility
            # branch cannot reintroduce live bounds if support is added later.
            x = residual + cross_attn_output.value
            hidden_states = Potential(
                x,
                PotentialBounds(
                    float(hidden_states.domain.min)
                    + float(cross_attn_output.domain.min),
                    float(hidden_states.domain.max)
                    + float(cross_attn_output.domain.max),
                ),
            )

        # The MLP residual is the second calibrated block boundary. Its analytic sum
        # remains the collection safety rail and unbound fallback.
        residual = hidden_states.value
        residual_domain = hidden_states.domain
        pot = _apply_norm(self.ln_2, hidden_states)
        feed_forward_hidden_states = self.mlp(pot)
        x = residual + feed_forward_hidden_states.value
        output_bounds = PotentialBounds(
            float(residual_domain.min)
            + float(feed_forward_hidden_states.domain.min),
            float(residual_domain.max)
            + float(feed_forward_hidden_states.domain.max),
        )
        if model_calibration_is_bound(self):
            hidden_states = calibrated_potential(
                self,
                "output",
                x,
                collection_bounds=output_bounds,
            )
        else:
            hidden_states = Potential(x, output_bounds)

        return hidden_states


# Copied from transformers.models.xlm.modeling_xlm.XLMSequenceSummary with XLM->GPT2
class GPT2SequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`GPT2Config`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else nn.Identity()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: torch.LongTensor | None = None
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


@auto_docstring
class GPT2PreTrainedModel(PreTrainedModel):
    config: GPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": GPT2Block,
        "attentions": OutputRecorder(GPT2Attention, layer_name=".attn", index=1),
        "cross_attentions": OutputRecorder(GPT2Attention, layer_name=".crossattention", index=1),
    }

    # No longer used as we directly use our masks instead
    _keys_to_ignore_on_load_unexpected = ["attn.bias", "crossattention.bias"]

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, (nn.LayerNorm, SpikingLayerNorm)):
            init.zeros_(module.bias)
            init.ones_(module.weight)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        if isinstance(module, PreTrainedModel):
            for name, p in module.named_parameters():
                if name == "c_proj.weight":
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    init.normal_(p, mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """
)
class GPT2DoubleHeadsModelOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss.
    mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
        Multiple choice classification loss.
    logits (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
        Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: torch.FloatTensor | None = None
    mc_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    mc_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        _theta = getattr(config, "theta", 10.0)
        _tau_s = getattr(config, "tau_s", 1.0)
        _use_spiking_ln = getattr(config, "use_spiking_layernorm", True)
        if _use_spiking_ln:
            _sln_kwargs = dict(
                theta=_theta, tau_s=_tau_s,
                use_spiking_mul=getattr(config, "spiking_ln_mul", True),
                use_spiking_log=getattr(config, "spiking_ln_log", True),
                use_spiking_expdiff=getattr(config, "spiking_ln_expdiff", True),
            )
            self.ln_f = SpikingLayerNorm(self.embed_dim, eps=config.layer_norm_epsilon, **_sln_kwargs)
        else:
            self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_embedding_bounds(
        self,
        *,
        refresh: bool = False,
    ) -> tuple[PotentialBounds, PotentialBounds]:
        """Freeze token and position embedding-table ranges for model entry.

        Token, optional token-type, and position values are table lookups. The two
        complete pretrained tables therefore define conservative input-independent
        ranges before the first GPT-2 block, including autoregressive cache positions.

        Args:
            refresh: Recompute after an intentional embedding parameter replacement
                or update.

        Returns:
            Frozen token-table and position-table potential ranges.

        Raises:
            RuntimeError: If parameters changed after freezing or during reduction.
            ValueError: If either table contains non-finite values.
        """
        # Include Parameter identity as well as PyTorch version so public embedding
        # replacement cannot reuse a cache from a different table at version zero.
        identity = (
            id(self.wte.weight),
            self.wte.weight._version,
            id(self.wpe.weight),
            self.wpe.weight._version,
        )
        cached = self.__dict__.get("_frozen_embedding_bounds")
        if cached is not None and not refresh:
            cached_identity, cached_bounds = cached
            if identity != cached_identity:
                raise RuntimeError(
                    "GPT-2 embedding parameters changed after bounds were frozen; "
                    "call freeze_embedding_bounds(refresh=True) before inference"
                )
            return cached_bounds

        # Reduce once in float64, keeping derived scalars out of the state dict and
        # avoiding inward rounding for low-precision checkpoint parameters.
        tables = (
            self.wte.weight.detach().to(dtype=torch.float64),
            self.wpe.weight.detach().to(dtype=torch.float64),
        )
        if not all(bool(torch.isfinite(table).all()) for table in tables):
            raise ValueError("GPT-2 embedding parameters must be finite")
        frozen_bounds = tuple(
            PotentialBounds(table.min().item(), table.max().item())
            for table in tables
        )

        # A concurrent update or replacement invalidates the complete reduction.
        final_identity = (
            id(self.wte.weight),
            self.wte.weight._version,
            id(self.wpe.weight),
            self.wpe.weight._version,
        )
        if final_identity != identity:
            raise RuntimeError(
                "GPT-2 embedding parameters changed while bounds were being frozen"
            )
        self.__dict__["_frozen_embedding_bounds"] = (
            final_identity,
            frozen_bounds,
        )
        return frozen_bounds

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
        # Public embedding replacement intentionally begins a new parameter regime.
        # Discard only derived metadata; the next setup/forward freezes the new table.
        self.__dict__.pop("_frozen_embedding_bounds", None)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: Potential | torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        # Resolve exactly one token source. Custom embeddings may carry a separately
        # established fixed range; plain tensors must fit the frozen token-table rail.
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif isinstance(inputs_embeds, Potential):
            input_shape = inputs_embeds.value.size()[:-1]
            batch_size = inputs_embeds.value.shape[0]
        elif isinstance(inputs_embeds, torch.Tensor):
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        # Freeze table endpoints only after checkpoint initialization and before any
        # block execution. Integer lookup is covered automatically; custom tensor
        # values are validated against their predeclared range without defining it.
        word_bounds, position_bounds = self.freeze_embedding_bounds()
        if inputs_embeds is None:
            token_embeddings = self.wte(input_ids)
            token_bounds = word_bounds
            validate_token_values = False
        elif isinstance(inputs_embeds, Potential):
            token_embeddings = inputs_embeds.value
            token_bounds = inputs_embeds.domain
            validate_token_values = True
        else:
            token_embeddings = inputs_embeds
            token_bounds = word_bounds
            validate_token_values = True
        if not token_embeddings.is_floating_point() or token_embeddings.is_complex():
            raise TypeError("inputs_embeds must be a real floating-point tensor")
        if token_embeddings.numel() == 0:
            raise ValueError("inputs_embeds must not be empty")
        if not isinstance(token_bounds, PotentialBounds):
            raise TypeError("inputs_embeds domain must be PotentialBounds")
        token_lower = float(token_bounds.min)
        token_upper = float(token_bounds.max)
        if (
            not math.isfinite(token_lower)
            or not math.isfinite(token_upper)
            or token_lower > token_upper
        ):
            raise ValueError("inputs_embeds fixed range must be finite and ordered")
        if validate_token_values:
            token_min, token_max = torch.aminmax(token_embeddings.detach())
            if (
                not bool(torch.isfinite(token_min) and torch.isfinite(token_max))
                or token_min.item() < token_lower
                or token_max.item() > token_upper
            ):
                raise ValueError("inputs_embeds escaped its declared fixed range")

        # Mask construction operates on the unwrapped tensor exactly as before. The
        # Potential metadata remains beside it for the later embedding-sum interval.
        inputs_embeds = token_embeddings

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = token_embeddings + position_embeds.to(token_embeddings.device)
        hidden_domain = PotentialBounds(
            token_lower + float(position_bounds.min),
            token_upper + float(position_bounds.max),
        )

        # Attention mask.
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        encoder_attention_mask = None
        if encoder_hidden_states is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
            hidden_domain = PotentialBounds(
                float(hidden_domain.min) + float(word_bounds.min),
                float(hidden_domain.max) + float(word_bounds.max),
            )

        # Embedding dropout is identity in evaluation. Its training envelope includes
        # zero and inverse keep-probability scaling without reading the sampled mask.
        hidden_states = self.drop(hidden_states)
        dropout_probability = float(self.drop.p)
        if not math.isfinite(dropout_probability) or not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("GPT-2 embedding dropout probability must lie in [0, 1]")
        if self.training and dropout_probability > 0.0:
            if dropout_probability >= 1.0:
                hidden_domain = PotentialBounds(0.0, 0.0)
            else:
                scale = 1.0 / (1.0 - dropout_probability)
                candidates = (
                    0.0,
                    float(hidden_domain.min) * scale,
                    float(hidden_domain.max) * scale,
                )
                hidden_domain = PotentialBounds(
                    min(candidates),
                    max(candidates),
                )

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        # Model entry is a signed calibration boundary. Collection observes the raw
        # embedding sum on its table-derived safety rail; frozen phases reset the
        # residual stream to the persisted range before the first block.
        if model_calibration_is_bound(self):
            pot = calibrated_potential(
                self,
                "input",
                hidden_states,
                collection_bounds=hidden_domain,
            )
        else:
            pot = Potential(hidden_states, hidden_domain)

        for i, block in enumerate(self.h):
            pot = block(
                pot,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
                **kwargs,
            )

        pot = _apply_norm(self.ln_f, pot)

        hidden_states = pot.value.view(output_shape)

        past_key_values = past_key_values if use_cache else None
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


@auto_docstring(
    custom_intro="""
        The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = GPT2SequenceSummary(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        mc_token_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        mc_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> GPT2DoubleHeadsModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        mc_token_ids (`torch.LongTensor` of shape `(batch_size, num_choices)`, *optional*, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) -
            1]`.
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to
            `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
        mc_labels (`torch.LongTensor` of shape `(batch_size)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, GPT2DoubleHeadsModel

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

        >>> # Add a [CLS] to the vocabulary (we should train it also!)
        >>> num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        >>> # Update the model embeddings with the new vocabulary size
        >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))

        >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        >>> encoded_choices = [tokenizer.encode(s) for s in choices]
        >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
        >>> lm_logits = outputs.logits
        >>> mc_logits = outputs.mc_logits
        ```"""
        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = transformer_outputs.last_hidden_state

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    [`GPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class GPT2ForTokenClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        start_positions: torch.LongTensor | None = None,
        end_positions: torch.LongTensor | None = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        outputs: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "GPT2DoubleHeadsModel",
    "GPT2ForQuestionAnswering",
    "GPT2ForSequenceClassification",
    "GPT2ForTokenClassification",
    "GPT2LMHeadModel",
    "GPT2Model",
    "GPT2PreTrainedModel",
]
