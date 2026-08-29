# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch Spiking RoBERTa model."""

from collections.abc import Callable
import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import initialization as init
from transformers.activations import ACT2FN, GELUActivation
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, logging
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_roberta import RobertaConfig

from utils.transforms.functions import gelu_approximation, tanh
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.integrations.spiking_sdpa_attention import attention_output_bounds
from utils.transformers.models.spiking_ops import SpikingLayerNorm, SpikingLinear, _apply_norm

logger = logging.get_logger(__name__)


class RobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

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
            self.LayerNorm = SpikingLayerNorm(config.hidden_size, **_sln_kwargs)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def freeze_parameter_bounds(
        self,
        *,
        refresh: bool = False,
    ) -> tuple[PotentialBounds, PotentialBounds, PotentialBounds]:
        """Freeze global ranges of the three RoBERTa embedding tables.

        Integer lookup can select only values already stored in the word, token-type,
        and position tables. Their complete finite ranges therefore form conservative
        batch-independent intervals and are cached after checkpoint loading.

        Args:
            refresh: Recompute ranges after an intentional embedding-table update.

        Returns:
            Frozen word, token-type, and position table ranges.

        Raises:
            RuntimeError: If parameters changed after freezing or during reduction.
            ValueError: If any embedding parameter is non-finite.
        """
        # Version counters define the cache generation without adding derived tensors
        # to state_dict or changing pretrained checkpoint compatibility.
        identity = (
            self.word_embeddings.weight._version,
            self.token_type_embeddings.weight._version,
            self.position_embeddings.weight._version,
        )
        cached = self.__dict__.get("_frozen_embedding_bounds")
        if cached is not None and not refresh:
            cached_identity, cached_bounds = cached
            if identity != cached_identity:
                raise RuntimeError(
                    "RoBERTa embedding parameters changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            return cached_bounds

        # One float64 scan prevents low-precision endpoint rounding from narrowing a
        # physical range. All subsequent forwards reuse the immutable scalar bounds.
        tables = (
            self.word_embeddings.weight.detach().to(dtype=torch.float64),
            self.token_type_embeddings.weight.detach().to(dtype=torch.float64),
            self.position_embeddings.weight.detach().to(dtype=torch.float64),
        )
        if not all(bool(torch.isfinite(table).all()) for table in tables):
            raise ValueError("RoBERTa embedding parameters must be finite")
        frozen_bounds = tuple(
            PotentialBounds(table.min().item(), table.max().item())
            for table in tables
        )

        # Refuse a mixed-version result if parameters changed during the reductions.
        final_identity = (
            self.word_embeddings.weight._version,
            self.token_type_embeddings.weight._version,
            self.position_embeddings.weight._version,
        )
        if final_identity != identity:
            raise RuntimeError(
                "RoBERTa embedding parameters changed while bounds were being frozen"
            )
        self.__dict__["_frozen_embedding_bounds"] = (
            final_identity,
            frozen_bounds,
        )
        return frozen_bounds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[Potential | torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        *,
        return_potential: bool = False,
    ) -> Potential | torch.Tensor:
        """Construct RoBERTa embeddings with frozen table-derived ranges.

        Token IDs use the full word-table envelope. Custom embeddings may carry an
        explicit ``Potential`` range or reuse the word-table range only when their
        values fit it. Position and token-type intervals are added before LayerNorm,
        whose fixed output range is retained for the internal encoder.

        Args:
            input_ids: Token indices, mutually exclusive with ``inputs_embeds``.
            token_type_ids: Optional segment indices.
            position_ids: Optional RoBERTa absolute position indices.
            inputs_embeds: Custom token embeddings with explicit or compatible range.
            past_key_values_length: Offset used to construct default position IDs.
            return_potential: Return internal range metadata when true; the default
                preserves the public tensor API.

        Returns:
            Embedding tensor or internal ``Potential`` with normalized fixed bounds.

        Raises:
            TypeError: If custom inputs or the return flag have invalid types.
            ValueError: If token sources are ambiguous or escape their fixed range.
        """
        # Resolve exactly one token source before constructing position IDs. Potential
        # is unwrapped only for the existing RoBERTa indexing formulas.
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if not isinstance(return_potential, bool):
            raise TypeError("return_potential must be a bool")
        word_bounds, token_type_bounds, position_bounds = (
            self.freeze_parameter_bounds()
        )

        if input_ids is not None:
            token_embeddings = self.word_embeddings(input_ids)
            token_bounds = word_bounds
            input_shape = input_ids.size()
            validate_token_values = False
        elif isinstance(inputs_embeds, Potential):
            token_embeddings = inputs_embeds.value
            token_bounds = inputs_embeds.domain
            input_shape = token_embeddings.size()[:-1]
            validate_token_values = True
        else:
            if not isinstance(inputs_embeds, torch.Tensor):
                raise TypeError("inputs_embeds must be Potential or torch.Tensor")
            token_embeddings = inputs_embeds
            token_bounds = word_bounds
            input_shape = token_embeddings.size()[:-1]
            validate_token_values = True

        # Custom inputs are validated against predeclared endpoints; their observed
        # extrema never become metadata. Integer lookup needs no per-forward scan.
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

        # Default position indices preserve RoBERTa padding behavior. A custom
        # Potential contributes only values and bounds, never an alternate indexing
        # convention.
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    token_embeddings,
                    self.padding_idx,
                )

        batch_size, seq_length = input_shape

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # NOTE: We assume either pos ids to have bsz == 1 (broadcastable) or bsz == effective bsz (input_shape[0])
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Sum fixed intervals before applying LayerNorm. The normalized module owns
        # its configuration/parameter-derived range, so the raw sum is never measured.
        raw_embeddings = (
            token_embeddings + token_type_embeddings + position_embeddings
        )
        raw_domain = PotentialBounds(
            token_lower
            + float(token_type_bounds.min)
            + float(position_bounds.min),
            token_upper
            + float(token_type_bounds.max)
            + float(position_bounds.max),
        )
        normalized = _apply_norm(
            self.LayerNorm,
            Potential(raw_embeddings, raw_domain),
        )

        # Evaluation dropout is identity. The analytic training envelope includes
        # zero and inverse keep-probability scaling without observing a sampled mask.
        dropped = self.dropout(normalized.value)
        dropout_probability = float(self.dropout.p)
        if not math.isfinite(dropout_probability) or not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("RoBERTa embedding dropout probability must lie in [0, 1]")
        output_domain = normalized.domain
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
        result = Potential(dropped, output_domain)
        return result if return_potential else result.value

    @staticmethod
    def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    @staticmethod
    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class RobertaSelfAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        
        _theta = getattr(config, "theta", 400.0)
        self.query = SpikingLinear(config.hidden_size, self.all_head_size, theta=_theta)
        self.key = SpikingLinear(config.hidden_size, self.all_head_size, theta=_theta)
        self.value = SpikingLinear(config.hidden_size, self.all_head_size, theta=_theta)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(self, pot: Potential, attention_mask=None, **kwargs) -> tuple[Potential, torch.Tensor]:
        """Apply RoBERTa self-attention with fixed backend-specific bounds.

        The eager path remains a convex combination and retains the projected value
        range. The spiking path derives one immutable output rail from ``theta`` and
        the configured positional capacity, passes that capacity to the backend,
        and attaches the same memoized domain to the reshaped context.

        Args:
            pot: Hidden states paired with their upstream potential bounds.
            attention_mask: Optional boolean or additive suppression mask.
            **kwargs: Additional Hugging Face attention controls forwarded unchanged.

        Returns:
            The attention context paired with its declared domain and the optional
            attention weights returned by the selected backend.
        """
        # Preserve the Potential returned by each projection while reshaping only
        # the tensor payload into the multi-head attention layout.
        batch_size = pot.value.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size
        
        pot_k = self.key(pot)
        pot_v = self.value(pot)
        pot_q = self.query(pot)
        
        key_layer = pot_k.value.view(*new_shape).transpose(1, 2)
        value_layer = pot_v.value.view(*new_shape).transpose(1, 2)
        query_layer = pot_q.value.view(*new_shape).transpose(1, 2)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Start with caller controls so spiking configuration can be installed once
        # without duplicate keyword expansion at the dispatch boundary.
        attention_kwargs = dict(kwargs)
        context_domain = pot_v.domain
        if self.config._attn_implementation == "spiking_sdpa":
            theta = float(getattr(self.config, "theta", 10.0))
            source_length_max = int(self.config.max_position_embeddings)
            attention_kwargs["theta"] = theta
            attention_kwargs["tau"] = getattr(self.config, "tau_s", 1.0)
            attention_kwargs["source_length_max"] = source_length_max
            context_domain = attention_output_bounds(theta, source_length_max)

        # Eager training dropout scales surviving normalized weights by 1/(1-p).
        # Include zero plus both scaled value endpoints without reading its mask.
        elif self.training and self.dropout.p > 0.0:
            if self.dropout.p >= 1.0:
                context_domain = PotentialBounds(0.0, 0.0)
            else:
                dropout_scale = 1.0 / (1.0 - self.dropout.p)
                dropout_candidates = (
                    0.0,
                    float(context_domain.min) * dropout_scale,
                    float(context_domain.max) * dropout_scale,
                )
                context_domain = PotentialBounds(
                    min(dropout_candidates),
                    max(dropout_candidates),
                )

        # The spiking backend receives exactly the configuration pair used for the
        # returned range; eager execution receives the original caller kwargs.
        context_layer, attention_probs = attention_interface(
            self, query_layer, key_layer, value_layer, attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **attention_kwargs,
        )
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # Head merging does not alter the numerical envelope, so retain the selected
        # fixed domain rather than measuring the current attention output.
        return Potential(context_layer, context_domain), attention_probs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        # SpikingLinear is parameter- and checkpoint-compatible with nn.Linear. The
        # dense ablation still calls functional.linear directly, while both paths can
        # reuse one exact frozen affine output interval.
        self.dense = SpikingLinear(
            config.hidden_size,
            config.hidden_size,
            theta=getattr(config, "theta", 400.0),
        )
        
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
            self.LayerNorm = SpikingLayerNorm(config.hidden_size, **_sln_kwargs)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pot: Potential, pot_skip: Potential) -> Potential:
        """Project attention output and normalize its residual on fixed ranges.

        Spiking execution delegates value and range propagation to ``SpikingLinear``.
        The dense ablation evaluates the identical affine parameters with PyTorch but
        obtains metadata from the same frozen interval calculation. Residual addition
        then uses endpoint arithmetic before the configured LayerNorm.

        Args:
            pot: Attention context with a declared fixed range.
            pot_skip: Residual stream with its upstream fixed range.

        Returns:
            Post-normalization attention residual and its analytic range.
        """
        # Select only the numerical implementation. Parameter-derived bounds are
        # shared, so the ablation flag cannot reintroduce current-output extrema.
        if self.use_spiking_mlp:
            pot_dense = self.dense(pot)
        else:
            out = nn.functional.linear(pot.value, self.dense.weight, self.dense.bias)
            pot_dense = Potential(
                out,
                self.dense.freeze_parameter_bounds(pot.domain),
            )

        # Evaluation dropout is identity. The maintained evaluator never trains this
        # adapter, so endpoint addition remains exact for its supported execution.
        dropped = self.dropout(pot_dense.value)
        val = dropped + pot_skip.value
        domain = PotentialBounds(
            pot_dense.domain.min + pot_skip.domain.min,
            pot_dense.domain.max + pot_skip.domain.max,
        )
        return _apply_norm(self.LayerNorm, Potential(val, domain))


class RobertaAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None, is_cross_attention=False):
        super().__init__()
        self.self = RobertaSelfAttention(config, is_causal=is_causal, layer_idx=layer_idx)
        self.output = RobertaSelfOutput(config)

    def forward(self, pot: Potential, attention_mask=None, **kwargs) -> tuple[Potential, torch.Tensor]:
        pot_attn, attention_probs = self.self(pot, attention_mask, **kwargs)
        pot_out = self.output(pot_attn, pot)
        return pot_out, attention_probs


class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        self.dense = SpikingLinear(
            config.hidden_size,
            config.intermediate_size,
            theta=getattr(config, "theta", 400.0),
        )
            
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, pot: Potential) -> Potential:
        """Apply the RoBERTa feed-forward activation on a fixed affine range.

        Operator GELU propagates its composed interval. Dense GELU lies between its
        input and zero, while ReLU maps affine endpoints monotonically. The dense
        ablation retains functional PyTorch arithmetic and never measures its output
        to define a physical range.

        Args:
            pot: Normalized hidden activation and its fixed range.

        Returns:
            Activated intermediate tensor with analytic range metadata.

        Raises:
            ValueError: If the configured activation has no maintained range rule.
        """
        # The spiking projection owns both event execution and affine interval
        # propagation. Its GELU/ReLU branches mirror BERT's maintained activation set.
        if self.use_spiking_mlp:
            pot_z = self.dense(pot)
            if isinstance(self.intermediate_act_fn, GELUActivation):
                return Potential(*gelu_approximation(*pot_z))
            if isinstance(self.intermediate_act_fn, nn.ReLU):
                return Potential(
                    pot_z.value.relu(),
                    PotentialBounds(
                        max(0.0, float(pot_z.domain.min)),
                        max(0.0, float(pot_z.domain.max)),
                    ),
                )
            raise ValueError(
                "RoBERTa intermediate activation requires a maintained analytic range rule"
            )

        # Dense projection uses the same parameters and frozen output interval, then
        # applies only analytically supported activation envelopes.
        projected = nn.functional.linear(
            pot.value,
            self.dense.weight,
            self.dense.bias,
        )
        projected_domain = self.dense.freeze_parameter_bounds(pot.domain)
        out = self.intermediate_act_fn(projected)
        if isinstance(self.intermediate_act_fn, GELUActivation):
            output_domain = PotentialBounds(
                min(float(projected_domain.min), 0.0),
                max(float(projected_domain.max), 0.0),
            )
        elif isinstance(self.intermediate_act_fn, nn.ReLU):
            output_domain = PotentialBounds(
                max(0.0, float(projected_domain.min)),
                max(0.0, float(projected_domain.max)),
            )
        else:
            raise ValueError(
                "RoBERTa intermediate activation requires a maintained analytic range rule"
            )
        return Potential(out, output_domain)


class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        self.dense = SpikingLinear(
            config.intermediate_size,
            config.hidden_size,
            theta=getattr(config, "theta", 400.0),
        )
        
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
            self.LayerNorm = SpikingLayerNorm(config.hidden_size, **_sln_kwargs)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pot_inter: Potential, pot_skip: Potential) -> Potential:
        """Project the MLP output and normalize its residual on fixed ranges.

        Dense and spiking projection paths share pretrained parameters and the exact
        fixed affine interval. Residual endpoint addition and LayerNorm propagation
        then remain independent of the activation values produced by this call.

        Args:
            pot_inter: Activated feed-forward representation and fixed range.
            pot_skip: Attention residual carried around the MLP.

        Returns:
            Completed RoBERTa layer output with fixed normalized bounds.
        """
        # Keep dense numerical execution event-free while reusing the affine cache
        # that prevents a second metadata implementation from observing live output.
        if self.use_spiking_mlp:
            pot_dense = self.dense(pot_inter)
        else:
            out = nn.functional.linear(pot_inter.value, self.dense.weight, self.dense.bias)
            pot_dense = Potential(
                out,
                self.dense.freeze_parameter_bounds(pot_inter.domain),
            )

        # The evaluator runs in eval mode, making dropout identity before the fixed
        # residual interval is normalized into a depth-independent output envelope.
        dropped = self.dropout(pot_dense.value)
        val = dropped + pot_skip.value
        domain = PotentialBounds(
            pot_dense.domain.min + pot_skip.domain.min,
            pot_dense.domain.max + pot_skip.domain.max,
        )
        return _apply_norm(self.LayerNorm, Potential(val, domain))


class RobertaLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.attention = RobertaAttention(config, is_causal=config.is_decoder, layer_idx=layer_idx)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, pot: Potential, attention_mask=None, **kwargs) -> tuple[Potential, torch.Tensor]:
        pot_attn, attention_probs = self.attention(pot, attention_mask, **kwargs)
        pot_inter = self.intermediate(pot_attn)
        pot_layer = self.output(pot_inter, pot_attn)
        return pot_layer, attention_probs


class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._theta = float(getattr(config, "theta", 10.0))
        self.layer = nn.ModuleList([RobertaLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Potential | torch.Tensor,
        attention_mask=None,
        **kwargs,
    ) -> Potential:
        """Enter the RoBERTa stack with an upstream or configured fixed range.

        Internal execution receives the embedding LayerNorm ``Potential``. Direct
        tensor calls retain compatibility through one fixed ``[-theta, theta]`` rail
        and never derive endpoints from their batch contents.

        Args:
            hidden_states: Embedding output with optional fixed metadata.
            attention_mask: Broadcast attention suppression tensor.
            **kwargs: Existing attention backend arguments.

        Returns:
            Final encoder activation and propagated fixed range.
        """
        # Preserve upstream metadata when available. A standalone tensor is clamped
        # to the same configuration-derived rail for every invocation.
        if isinstance(hidden_states, Potential):
            pot = hidden_states
        elif isinstance(hidden_states, torch.Tensor):
            if not math.isfinite(self._theta) or self._theta <= 0.0:
                raise ValueError("RoBERTa encoder theta must be finite and positive")
            entry_bounds = PotentialBounds(-self._theta, self._theta)
            pot = Potential(
                entry_bounds.clamp(hidden_states, name="roberta_encoder_input"),
                entry_bounds,
            )
        else:
            raise TypeError("hidden_states must be Potential or torch.Tensor")

        # Each post-norm layer returns a new analytic Potential, so no later block
        # needs to reconstruct its input domain from a live tensor.
        for layer_module in self.layer:
            pot, _ = layer_module(pot, attention_mask, **kwargs)
        return pot


class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        if self.use_spiking_mlp:
            self.dense = SpikingLinear(config.hidden_size, config.hidden_size, theta=getattr(config, "theta", 400.0))
            self.tau_s = getattr(config, "tau_s", 1.0)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()

    def forward(self, hidden_states: Potential | torch.Tensor) -> torch.Tensor:
        """Pool RoBERTa's first token while preserving the encoder range.

        Slicing the token dimension cannot enlarge the declared scalar interval.
        Spiking projection therefore consumes the final encoder range directly;
        standalone tensor calls use the configured fixed threshold rail.

        Args:
            hidden_states: Final sequence activation with optional fixed bounds.

        Returns:
            Dense or spiking Tanh-pooled first-token tensor.
        """
        # Separate value selection from range fallback so the dense path remains
        # numerically identical and never clamps a tensor it does not encode.
        if isinstance(hidden_states, Potential):
            first_token_tensor = hidden_states.value[:, 0]
            first_token_domain = hidden_states.domain
        elif isinstance(hidden_states, torch.Tensor):
            first_token_tensor = hidden_states[:, 0]
            first_token_domain = None
        else:
            raise TypeError("hidden_states must be Potential or torch.Tensor")

        # Spiking pooling requires a zero-containing identity-code rail. The internal
        # path supplies one from the encoder; only direct tensor calls use theta.
        if self.use_spiking_mlp:
            if first_token_domain is None:
                theta = float(self.dense.theta)
                if not math.isfinite(theta) or theta <= 0.0:
                    raise ValueError("RoBERTa pooler theta must be finite and positive")
                first_token_domain = PotentialBounds(-theta, theta)
                first_token_tensor = first_token_domain.clamp(
                    first_token_tensor,
                    name="roberta_pooler_input",
                )
            pot_in = Potential(first_token_tensor, first_token_domain)
            pot_dense = self.dense(pot_in)
            pooled_output, _ = tanh(pot_dense.value, pot_dense.domain, tau_s=self.tau_s, theta=self.dense.theta)
            return pooled_output

        # Dense pooling contains no temporal encoder and uses the unchanged slice.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@auto_docstring
class RobertaPreTrainedModel(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, SpikingLayerNorm)):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, RobertaEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
            init.zeros_(module.token_type_ids)


@auto_docstring
class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[Potential | torch.Tensor] = None,
        return_potential: bool = False,
        **kwargs
    ):
        """Run RoBERTa and optionally retain its internal final Potential.

        The default return remains the Hugging Face model-output type. Local task
        wrappers opt into the additional final ``Potential`` so spiking heads consume
        the encoder's fixed range without reconstructing it from ``last_hidden_state``.

        Args:
            input_ids: Token indices.
            attention_mask: Optional keep/suppression mask.
            token_type_ids: Optional segment indices.
            position_ids: Optional absolute position indices.
            inputs_embeds: Optional custom embeddings with fixed range metadata.
            return_potential: Return ``(model_output, final_potential)`` internally.
            **kwargs: Existing attention backend arguments.

        Returns:
            Standard model output, or that output paired with its final Potential.
        """
        # Request metadata only on this private adapter path. Calling the embedding
        # module directly retains its established tensor return by default.
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            return_potential=True,
        )
        if not isinstance(embedding_output, Potential):
            raise RuntimeError("RoBERTa internal embeddings must return Potential")
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embedding_output.value.dtype).min
        else:
            extended_attention_mask = None

        pot = self.encoder(embedding_output, extended_attention_mask, **kwargs)
        sequence_output = pot.value
        pooled_output = self.pooler(pot) if self.pooler is not None else None

        # Public callers see the unchanged Hugging Face output. Only local spiking
        # task heads request the extra metadata tuple and unwrap it immediately.
        output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )
        return (output, pot) if return_potential else output


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        self.dense = SpikingLinear(
            config.hidden_size,
            config.hidden_size,
            theta=getattr(config, "theta", 400.0),
        )

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
            self.layer_norm = SpikingLayerNorm(config.hidden_size, **_sln_kwargs)
        else:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features: Potential | torch.Tensor, **kwargs):
        """Apply the language-model transform without losing encoder bounds.

        Local wrappers provide the final encoder ``Potential``. Direct tensor calls
        use the configured threshold rail for compatibility. Dense and spiking affine
        paths share one frozen parameter interval, GELU uses an analytic or composed
        range, and LayerNorm supplies its own fixed output envelope before decoding.

        Args:
            features: Final sequence representation with optional fixed bounds.
            **kwargs: Reserved Hugging Face head arguments.

        Returns:
            Vocabulary logits with the established tied decoder and bias.
        """
        # Preserve final encoder metadata. A standalone tensor lacks an upstream
        # contract, so clamp it to the same fixed theta rail used by spiking affine
        # encoding instead of measuring the current head input.
        if isinstance(features, Potential):
            pot = features
        elif isinstance(features, torch.Tensor):
            theta = float(self.dense.theta)
            if not math.isfinite(theta) or theta <= 0.0:
                raise ValueError("RoBERTa LM head theta must be finite and positive")
            input_domain = PotentialBounds(-theta, theta)
            pot = Potential(
                input_domain.clamp(features, name="roberta_lm_head_input"),
                input_domain,
            )
        else:
            raise TypeError("features must be Potential or torch.Tensor")

        # Dense ablation retains functional linear and GELU values, while range
        # metadata comes from fixed affine endpoints and the [0,1] GELU gate.
        if self.use_spiking_mlp:
            pot_z = self.dense(pot)
            pot_act = Potential(*gelu_approximation(*pot_z))
        else:
            x = nn.functional.linear(
                pot.value,
                self.dense.weight,
                self.dense.bias,
            )
            dense_domain = self.dense.freeze_parameter_bounds(pot.domain)
            x = nn.functional.gelu(x)
            pot_act = Potential(
                x,
                PotentialBounds(
                    min(float(dense_domain.min), 0.0),
                    max(float(dense_domain.max), 0.0),
                ),
            )

        # LayerNorm eliminates the potentially broad GELU envelope through its fixed
        # normalized range. The final decoder is dense and does not emit spike metadata.
        x = _apply_norm(self.layer_norm, pot_act).value
        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x


@auto_docstring
class RobertaForCausalLM(RobertaPreTrainedModel):
    _tied_weights_keys = {
        "lm_head.decoder.weight": "roberta.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs, sequence_potential = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_potential=True,
            **kwargs
        )
        logits = self.lm_head(sequence_potential)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )


@auto_docstring
class RobertaForMaskedLM(RobertaPreTrainedModel):
    _tied_weights_keys = {
        "lm_head.decoder.weight": "roberta.embeddings.word_embeddings.weight",
        "lm_head.decoder.bias": "lm_head.bias",
    }

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs, sequence_potential = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_potential=True,
            **kwargs
        )
        prediction_scores = self.lm_head(sequence_potential)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
        )


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        self.dense = SpikingLinear(
            config.hidden_size,
            config.hidden_size,
            theta=getattr(config, "theta", 400.0),
        )
        self.tau_s = getattr(config, "tau_s", 1.0)
            
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: Potential | torch.Tensor, **kwargs):
        """Classify RoBERTa's first token with inherited fixed bounds.

        The first-token slice retains the final encoder interval. Evaluation dropout
        is identity; dense and spiking projections share one frozen affine range, and
        Tanh is structurally bounded before the ordinary output classifier.

        Args:
            features: Final sequence representation with optional fixed bounds.
            **kwargs: Reserved Hugging Face classifier arguments.

        Returns:
            Dense task logits.
        """
        # Slice values and metadata together. Direct calls use a fixed theta fallback
        # because no caller-supplied domain is available.
        if isinstance(features, Potential):
            first_token = features.value[:, 0, :]
            first_token_domain = features.domain
        elif isinstance(features, torch.Tensor):
            theta = float(self.dense.theta)
            if not math.isfinite(theta) or theta <= 0.0:
                raise ValueError(
                    "RoBERTa classification head theta must be finite and positive"
                )
            first_token_domain = PotentialBounds(-theta, theta)
            first_token = first_token_domain.clamp(
                features[:, 0, :],
                name="roberta_classification_head_input",
            )
        else:
            raise TypeError("features must be Potential or torch.Tensor")

        # Evaluation dropout leaves both tensor and interval unchanged. This project
        # does not train converted models, so no sampled training mask enters bounds.
        x = self.dropout(first_token)
        pot_in = Potential(x, first_token_domain)

        # Select numerical execution without splitting range semantics. The dense
        # Tanh maps the frozen affine endpoints monotonically; the spiking Tanh owns
        # the same public structural output range through its composed operator.
        if self.use_spiking_mlp:
            pot_z = self.dense(pot_in)
            pot_tanh = Potential(
                *tanh(
                    pot_z.value,
                    pot_z.domain,
                    tau_s=self.tau_s,
                    theta=self.dense.theta,
                )
            )
        else:
            x = nn.functional.linear(
                pot_in.value,
                self.dense.weight,
                self.dense.bias,
            )
            dense_domain = self.dense.freeze_parameter_bounds(pot_in.domain)
            pot_tanh = Potential(
                torch.tanh(x),
                PotentialBounds(
                    math.tanh(float(dense_domain.min)),
                    math.tanh(float(dense_domain.max)),
                ),
            )

        # No later operator-backed layer consumes a Potential. The second dropout and
        # output projection therefore remain the existing dense task-head arithmetic.
        x = self.dropout(pot_tanh.value)
        x = self.out_proj(x)
        return x


@auto_docstring
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs, sequence_potential = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_potential=True,
            **kwargs
        )
        logits = self.classifier(sequence_potential)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(loss=loss, logits=logits)


@auto_docstring
class RobertaForMultipleChoice(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            **kwargs
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits)


@auto_docstring
class RobertaForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)


@auto_docstring
class RobertaForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )


__all__ = [
    "RobertaModel",
    "RobertaForCausalLM",
    "RobertaForMaskedLM",
    "RobertaForSequenceClassification",
    "RobertaForMultipleChoice",
    "RobertaForTokenClassification",
    "RobertaForQuestionAnswering",
    "RobertaPreTrainedModel",
]
