# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""TTFS operator-backed Llama model derived from Hugging Face modeling_llama."""

import math
from collections.abc import Callable

import torch
from torch import nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from utils.transforms.functions import clamp_swish_output, rmsnorm_function, swiglu_function
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import calibrated_potential, model_calibration_is_bound
from utils.transformers.models.spiking_llama.configuration_llama import LlamaConfig
from utils.transformers.models.spiking_ops import SpikingLinear


def _interval_product(
    left: PotentialBounds,
    right: PotentialBounds,
) -> PotentialBounds:
    products = (
        float(left.min) * float(right.min),
        float(left.min) * float(right.max),
        float(left.max) * float(right.min),
        float(left.max) * float(right.max),
    )
    return PotentialBounds(min(products), max(products))


class LlamaRMSNorm(nn.Module):
    """Apply the RMSNorm operator composition and pretrained fixed gains."""

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, *,
        tau_s: float = 1.0, clip_margin: float = 1e-8,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.tau_s = tau_s
        self.clip_margin = clip_margin

    def freeze_parameter_bounds(self, *, refresh: bool = False) -> PotentialBounds:
        identity = (
            id(self.weight),
            self.weight._version,
            self.weight.dtype,
            float(self.variance_epsilon),
            float(self.tau_s),
            float(self.clip_margin),
        )
        cached = self.__dict__.get("_frozen_output_bounds")
        if cached is not None and not refresh:
            cached_identity, output_bounds = cached
            if identity != cached_identity:
                raise RuntimeError(
                    "LlamaRMSNorm parameters changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            return output_bounds

        epsilon = float(self.variance_epsilon)
        for name, value in (
            ("epsilon", epsilon), ("tau_s", self.tau_s), ("clip_margin", self.clip_margin),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"LlamaRMSNorm {name} must be finite and positive")
        weight = self.weight.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(weight).all()):
            raise ValueError("LlamaRMSNorm weights must be finite")
        radius = math.sqrt(self.weight.numel()) * weight.abs().max().item()
        output_bounds = PotentialBounds(-radius, radius)

        final_identity = (
            id(self.weight),
            self.weight._version,
            self.weight.dtype,
            float(self.variance_epsilon),
            float(self.tau_s),
            float(self.clip_margin),
        )
        if final_identity != identity:
            raise RuntimeError("LlamaRMSNorm parameters changed while bounds were frozen")
        self.__dict__["_frozen_output_bounds"] = (final_identity, output_bounds)
        return output_bounds

    def forward(self, hidden_states: Potential) -> Potential:
        if not isinstance(hidden_states, Potential):
            raise TypeError("LlamaRMSNorm requires Potential input")
        if hidden_states.value.ndim == 0 or hidden_states.value.shape[-1] != self.weight.numel():
            raise ValueError("LlamaRMSNorm input feature dimension must match its weight")
        input_dtype = hidden_states.value.dtype
        output_dtype = torch.promote_types(input_dtype, self.weight.dtype)
        output_bounds = self.freeze_parameter_bounds().outward_rounded(output_dtype)
        value = hidden_states.value.to(torch.float32)
        value, _ = rmsnorm_function(
            value, hidden_states.domain, eps=self.variance_epsilon,
            tau_s=self.tau_s, clip_margin=self.clip_margin,
        )
        # Pretrained gamma is the fixed gain of the receiving signed contributions.
        value = self.weight * value.to(input_dtype)
        return Potential(output_bounds.clamp(value, name="llama_rmsnorm"), output_bounds)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaMLP(nn.Module):
    """Apply Llama's gated MLP through the maintained SwiGLU composition."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.use_spiking_mlp = bool(getattr(config, "use_spiking_mlp", True))
        self.tau_s = float(getattr(config, "tau_s", 1.0))
        if self.use_spiking_mlp and config.hidden_act != "silu":
            raise ValueError("operator-backed Llama MLP requires hidden_act='silu'")
        self.gate_proj = SpikingLinear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = SpikingLinear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = SpikingLinear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @staticmethod
    def _dense_projection(projection: SpikingLinear, value: Potential) -> Potential:
        projected = nn.functional.linear(value.value, projection.weight, projection.bias)
        return Potential(projected, projection.freeze_parameter_bounds(value.domain))

    def forward(self, hidden_states: Potential) -> Potential:
        if not isinstance(hidden_states, Potential):
            raise TypeError("LlamaMLP requires Potential input")
        if self.use_spiking_mlp:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
            if model_calibration_is_bound(self):
                gate = calibrated_potential(
                    self, "gate", gate.value, collection_bounds=gate.domain,
                )
                up = calibrated_potential(
                    self, "up", up.value, collection_bounds=up.domain,
                )
            activated_value, activated_bounds = swiglu_function(
                gate.value,
                gate.domain,
                up.value,
                up.domain,
                beta=1.0,
                tau_s=self.tau_s,
            )
            return self.down_proj(Potential(activated_value, activated_bounds))

        gate = self._dense_projection(self.gate_proj, hidden_states)
        up = self._dense_projection(self.up_proj, hidden_states)
        activated_gate = self.act_fn(gate.value)
        if self.config.hidden_act not in {"silu", "swish"}:
            raise ValueError("dense Llama MLP requires a maintained SiLU range rule")
        activated_gate, activated_gate_bounds = clamp_swish_output(
            activated_gate, gate.domain
        )
        product = Potential(
            activated_gate * up.value,
            _interval_product(activated_gate_bounds, up.domain),
        )
        return self._dense_projection(self.down_proj, product)


class LlamaAttention(nn.Module):
    """Cache-aware grouped-query attention with fixed projection ranges."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = True
        self.q_proj = SpikingLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = SpikingLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = SpikingLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = SpikingLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: Potential,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[Potential, torch.Tensor | None]:
        if not isinstance(hidden_states, Potential):
            raise TypeError("LlamaAttention requires Potential input")
        input_shape = hidden_states.value.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query_states = query.value.view(hidden_shape).transpose(1, 2)
        key_states = key.value.view(hidden_shape).transpose(1, 2)
        value_states = value.value.view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings are required for Llama attention")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        query_radius = math.sqrt(2.0) * max(
            abs(float(query.domain.min)), abs(float(query.domain.max))
        )
        key_radius = math.sqrt(2.0) * max(
            abs(float(key.domain.min)), abs(float(key.domain.max))
        )
        selected_bounds = (
            PotentialBounds(-query_radius, query_radius),
            PotentialBounds(-key_radius, key_radius),
            value.domain,
        )
        using_spiking = self.config._attn_implementation == "spiking_sdpa"
        if using_spiking and model_calibration_is_bound(self):
            value_radius = max(abs(float(value.domain.min)), abs(float(value.domain.max)))
            projected = tuple(
                calibrated_potential(self, name, tensor, collection_bounds=bounds)
                for name, tensor, bounds in zip(
                    ("query", "key", "value"),
                    (query_states, key_states, value_states),
                    (*selected_bounds[:2], PotentialBounds(-value_radius, value_radius)),
                )
            )
            query_states, key_states, value_states = (item.value for item in projected)
            selected_bounds = tuple(item.domain for item in projected)
        else:
            query_states = selected_bounds[0].clamp(query_states, name="llama_query")
            key_states = selected_bounds[1].clamp(key_states, name="llama_key")

        if past_key_values is not None:
            identity = tuple(
                (float(bounds.min), float(bounds.max)) for bounds in selected_bounds
            )
            cache_bounds = getattr(
                past_key_values, "_delayed_temporal_llama_bounds", {}
            )
            previous = cache_bounds.get(self.layer_idx)
            if past_key_values.get_seq_length(self.layer_idx) > 0 and previous != identity:
                raise ValueError("Llama cache does not match the selected attention bounds")
            cache_bounds[self.layer_idx] = identity
            past_key_values._delayed_temporal_llama_bounds = cache_bounds
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attention_kwargs = dict(kwargs)
        if using_spiking:
            attention_kwargs.update(
                tau=float(getattr(self.config, "tau_s", 1.0)),
                source_length_max=int(self.config.max_position_embeddings),
                query_bounds=selected_bounds[0],
                key_bounds=selected_bounds[1],
                value_bounds=selected_bounds[2],
            )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **attention_kwargs,
        )

        context_bounds = selected_bounds[2]
        if not using_spiking and self.training and self.attention_dropout > 0.0:
            if self.attention_dropout >= 1.0:
                context_bounds = PotentialBounds(0.0, 0.0)
            else:
                scale = 1.0 / (1.0 - self.attention_dropout)
                candidates = (
                    0.0,
                    float(context_bounds.min) * scale,
                    float(context_bounds.max) * scale,
                )
                context_bounds = PotentialBounds(min(candidates), max(candidates))

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(Potential(attn_output, context_bounds)), attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            tau_s=config.tau_s, clip_margin=config.rmsnorm_clip_margin,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            tau_s=config.tau_s, clip_margin=config.rmsnorm_clip_margin,
        )

    def forward(
        self,
        hidden_states: Potential,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> Potential:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attended, _ = self.self_attn(
            hidden_states=normalized,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attention_sum = residual.value + attended.value
        attention_bounds = PotentialBounds(
            float(residual.domain.min) + float(attended.domain.min),
            float(residual.domain.max) + float(attended.domain.max),
        )
        hidden_states = (
            calibrated_potential(
                self, "attention_residual", attention_sum,
                collection_bounds=attention_bounds,
            )
            if model_calibration_is_bound(self)
            else Potential(attention_sum, attention_bounds)
        )

        residual = hidden_states
        fed_forward = self.mlp(self.post_attention_layernorm(hidden_states))
        output_sum = residual.value + fed_forward.value
        output_bounds = PotentialBounds(
            float(residual.domain.min) + float(fed_forward.domain.min),
            float(residual.domain.max) + float(fed_forward.domain.max),
        )
        return (
            calibrated_potential(
                self, "output", output_sum, collection_bounds=output_bounds,
            )
            if model_calibration_is_bound(self)
            else Potential(output_sum, output_bounds)
        )


class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, LlamaRotaryEmbedding):
            rope_type = module.config.rope_parameters["rope_type"]
            rope_init_fn = (
                module.compute_default_rope_parameters
                if rope_type == "default"
                else ROPE_INIT_FUNCTIONS[rope_type]
            )
            inv_freq, attention_scaling = rope_init_fn(
                module.config, module.inv_freq.device
            )
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq.clone()
            module.attention_scaling = attention_scaling
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, LlamaRMSNorm):
            init.ones_(module.weight)


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            tau_s=config.tau_s, clip_margin=config.rmsnorm_clip_margin,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def freeze_embedding_bounds(self, *, refresh: bool = False) -> PotentialBounds:
        identity = (id(self.embed_tokens.weight), self.embed_tokens.weight._version)
        cached = self.__dict__.get("_frozen_embedding_bounds")
        if cached is not None and not refresh:
            cached_identity, bounds = cached
            if identity != cached_identity:
                raise RuntimeError(
                    "Llama embedding parameters changed after bounds were frozen; "
                    "call freeze_embedding_bounds(refresh=True) before inference"
                )
            return bounds
        table = self.embed_tokens.weight.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(table).all()):
            raise ValueError("Llama embedding parameters must be finite")
        bounds = PotentialBounds(
            min(0.0, table.min().item()), max(0.0, table.max().item())
        )
        final_identity = (id(self.embed_tokens.weight), self.embed_tokens.weight._version)
        if final_identity != identity:
            raise RuntimeError("Llama embedding parameters changed while bounds were frozen")
        self.__dict__["_frozen_embedding_bounds"] = (final_identity, bounds)
        return bounds

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.__dict__.pop("_frozen_embedding_bounds", None)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Potential | torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        return_potential: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple[BaseModelOutputWithPast, Potential]:
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        embedding_bounds = self.freeze_embedding_bounds()
        if inputs_embeds is None:
            embedded = self.embed_tokens(input_ids)
            potential = Potential(embedded, embedding_bounds)
        elif isinstance(inputs_embeds, Potential):
            potential = inputs_embeds
            embedded = potential.value
        else:
            embedded = inputs_embeds
            potential = Potential(embedded, embedding_bounds)

        if not embedded.is_floating_point() or embedded.is_complex():
            raise TypeError("inputs_embeds must be a real floating-point tensor")
        if embedded.numel() == 0 or not bool(torch.isfinite(embedded).all()):
            raise ValueError("inputs_embeds must be nonempty and finite")
        if not isinstance(potential.domain, PotentialBounds):
            raise TypeError("inputs_embeds domain must be PotentialBounds")
        lower = float(potential.domain.min)
        upper = float(potential.domain.max)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("inputs_embeds fixed range must be finite and ordered")
        value_min, value_max = torch.aminmax(embedded.detach())
        if value_min.item() < lower or value_max.item() > upper:
            raise ValueError("inputs_embeds escaped its declared fixed range")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(embedded.shape[1], device=embedded.device) + past_seen_tokens
            ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=embedded,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(embedded, position_ids=position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            potential = decoder_layer(
                potential,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        potential = self.norm(potential)
        output = BaseModelOutputWithPast(
            last_hidden_state=potential.value,
            past_key_values=past_key_values if use_cache else None,
        )
        if not isinstance(return_potential, bool):
            raise TypeError("return_potential must be a bool")
        return (output, potential) if return_potential else output


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = SpikingLinear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        if not isinstance(new_embeddings, SpikingLinear):
            raise TypeError("Llama output embeddings must use SpikingLinear")
        self.lm_head = new_embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs, final_potential = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_potential=True,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(
            Potential(
                final_potential.value[:, slice_indices, :], final_potential.domain
            )
        ).value
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaForCausalLM",
    "LlamaMLP",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaRMSNorm",
]
