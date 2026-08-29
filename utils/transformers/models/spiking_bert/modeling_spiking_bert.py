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
"""PyTorch Spiking BERT model."""

from collections.abc import Callable
import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import initialization as init
import transformers
from transformers.activations import ACT2FN, GELUActivation
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, logging
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_bert import BertConfig

from utils.transforms.functions import gelu_approximation, tanh
from utils.transforms.types import Potential, PotentialBounds
from utils.transformers.calibration import (
    calibrated_potential,
    model_calibration_is_bound,
)
from utils.transformers.integrations.spiking_sdpa_attention import attention_output_bounds
from utils.transformers.models.spiking_ops import SpikingLayerNorm, SpikingLinear, _apply_norm

logger = logging.get_logger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
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

    def freeze_parameter_bounds(
        self,
        *,
        refresh: bool = False,
    ) -> tuple[PotentialBounds, PotentialBounds, PotentialBounds]:
        """Freeze global ranges of the three pretrained embedding tables.

        A token embedding, token-type embedding, and position embedding are selected
        by integer lookup, so the full finite parameter tables provide conservative
        input-independent intervals for every supported sequence. The ranges are
        cached after checkpoint loading and rejected if a standard parameter update
        changes any table without an explicit refresh.

        Args:
            refresh: Recompute ranges after an intentional table mutation.

        Returns:
            Frozen word, token-type, and position embedding ranges.

        Raises:
            RuntimeError: If a table changed after freezing or during recomputation.
            ValueError: If an embedding table contains a non-finite value.

        Notes:
            PyTorch version counters detect checkpoint loads and ordinary in-place
            updates, but cannot detect unsupported direct ``parameter.data`` writes.
        """
        # Version counters form the cache identity. Shape need not be stored
        # separately because supported parameter replacement or resize operations
        # also change the owning Parameter or its version before reuse.
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
                    "BERT embedding parameters changed after bounds were frozen; "
                    "call freeze_parameter_bounds(refresh=True) before inference"
                )
            return cached_bounds

        # Scan each complete table once in float64 so low-precision checkpoints do
        # not round a true endpoint inward while establishing the scalar envelope.
        tables = (
            self.word_embeddings.weight.detach().to(dtype=torch.float64),
            self.token_type_embeddings.weight.detach().to(dtype=torch.float64),
            self.position_embeddings.weight.detach().to(dtype=torch.float64),
        )
        if not all(bool(torch.isfinite(table).all()) for table in tables):
            raise ValueError("BERT embedding parameters must be finite")
        frozen_bounds = tuple(
            PotentialBounds(table.min().item(), table.max().item())
            for table in tables
        )

        # Recheck versions after reductions to prevent a concurrent update from
        # publishing endpoints assembled from different parameter revisions.
        final_identity = (
            self.word_embeddings.weight._version,
            self.token_type_embeddings.weight._version,
            self.position_embeddings.weight._version,
        )
        if final_identity != identity:
            raise RuntimeError(
                "BERT embedding parameters changed while bounds were being frozen"
            )

        # Derived bounds stay outside state_dict so pretrained checkpoint keys and
        # serialization remain unchanged. Repeated forwards reuse immutable objects.
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
        """Construct BERT embeddings with parameter-derived fixed bounds.

        Integer token lookup uses the frozen word-table range. A caller-provided
        embedding may carry its own :class:`Potential`; an ordinary tensor is
        accepted only when it fits the same frozen word-table envelope. Position and
        token-type ranges are then added analytically before LayerNorm and dropout.

        Args:
            input_ids: Token indices, mutually exclusive with ``inputs_embeds``.
            token_type_ids: Optional segment indices, defaulting to zero.
            position_ids: Optional absolute position indices.
            inputs_embeds: Precomputed token embeddings with an explicit or compatible
                fixed word-embedding range.
            past_key_values_length: Position offset retained for API compatibility.
            return_potential: Return internal fixed-range metadata when true; the
                default preserves the Hugging Face tensor API.

        Returns:
            Embedding tensor, or the same tensor paired with its fixed normalized
            range for the local spiking model.

        Raises:
            TypeError: If inputs or the return flag have invalid types.
            ValueError: If token sources are ambiguous, a custom embedding escapes
                its declared range, or dropout probability is invalid.
        """
        # Exactly one token source is required. This also ensures input shape and
        # device are derived from the tensor whose range enters the embedding sum.
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if not isinstance(return_potential, bool):
            raise TypeError("return_potential must be a bool")
        word_bounds, token_type_bounds, position_bounds = (
            self.freeze_parameter_bounds()
        )

        # Integer lookup is covered by the complete word-table interval. Custom
        # tensors either carry an explicit Potential range or must fit the same table
        # envelope; validation reads extrema but never uses them to create a bound.
        if input_ids is not None:
            input_shape = input_ids.size()
            token_embeddings = self.word_embeddings(input_ids)
            token_bounds = word_bounds
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

        # Position and token-type defaults are deterministic functions of the input
        # shape. Their lookup values remain inside the two frozen table envelopes.
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=token_embeddings.device,
            ).unsqueeze(0).expand(input_shape[0], -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=token_embeddings.device,
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Addition uses exact interval arithmetic over the three fixed scalar
        # envelopes. LayerNorm then supplies its own configuration/parameter-derived
        # output range without inspecting the normalized activation.
        raw_embeddings = (
            token_embeddings + token_type_embeddings + position_embeddings
        )
        raw_domain = PotentialBounds(
            float(token_bounds.min)
            + float(token_type_bounds.min)
            + float(position_bounds.min),
            float(token_bounds.max)
            + float(token_type_bounds.max)
            + float(position_bounds.max),
        )
        normalized = _apply_norm(
            self.LayerNorm,
            Potential(raw_embeddings, raw_domain),
        )

        # Evaluation dropout is identity. In training, include zero and the standard
        # inverse-keep-probability scale in the analytic range without observing the
        # sampled mask. This adapter is evaluated rather than trained, but retaining
        # the formula keeps its declared contract complete.
        dropped = self.dropout(normalized.value)
        dropout_probability = float(self.dropout.p)
        if not math.isfinite(dropout_probability) or not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("BERT embedding dropout probability must lie in [0, 1]")
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

        # Only the local BertModel requests Potential. Direct embedding-module users
        # retain the established tensor result unless they opt into metadata.
        result = Potential(dropped, output_domain)
        return result if return_potential else result.value


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * (query.size(-1) ** -0.5)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        _theta = getattr(config, "theta", 400.0)
        self.query = SpikingLinear(config.hidden_size, self.all_head_size, theta=_theta)
        self.key = SpikingLinear(config.hidden_size, self.all_head_size, theta=_theta)
        self.value = SpikingLinear(config.hidden_size, self.all_head_size, theta=_theta)

    def forward(self, pot: Potential, attention_mask=None) -> tuple[Potential, torch.Tensor]:
        """Apply BERT self-attention with a backend-consistent output domain.

        Eager attention preserves the projected-value domain because its normalized
        weights form a convex combination. The spiking backend uses the model's
        fixed positional capacity as ``S_max`` and shares one memoized output rail
        between physical clamping and the returned ``Potential`` metadata.

        Args:
            pot: Hidden states paired with their declared potential bounds.
            attention_mask: Optional BERT keep/suppression mask forwarded to the
                selected attention implementation.

        Returns:
            The reshaped attention context with its fixed domain and the optional
            attention weights returned by the selected backend.
        """
        # Project once into Q/K/V potentials, then expose the head dimension without
        # discarding the value projection's domain used by the eager backend.
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
            # Check if it exists in ALL_ATTENTION_FUNCTIONS (transformers 4.38+)
            if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Default to the convex-combination range. Only spiking attention replaces
        # it with a source-capacity rail that is independent of the current sequence.
        kwargs = {}
        context_domain = pot_v.domain
        if self.config._attn_implementation == "spiking_sdpa":
            theta = float(getattr(self.config, "theta", 10.0))
            source_length_max = int(self.config.max_position_embeddings)
            kwargs["theta"] = theta
            kwargs["tau"] = getattr(self.config, "tau_s", 1.0)
            kwargs["source_length_max"] = source_length_max
            context_domain = attention_output_bounds(theta, source_length_max)

        # Eager training dropout scales surviving normalized weights by 1/(1-p).
        # Include zero plus both scaled value endpoints without observing its mask.
        elif self.training and self.dropout_prob > 0.0:
            if self.dropout_prob >= 1.0:
                context_domain = PotentialBounds(0.0, 0.0)
            else:
                dropout_scale = 1.0 / (1.0 - self.dropout_prob)
                dropout_candidates = (
                    0.0,
                    float(context_domain.min) * dropout_scale,
                    float(context_domain.max) * dropout_scale,
                )
                context_domain = PotentialBounds(
                    min(dropout_candidates),
                    max(dropout_candidates),
                )

        # Backend clamping receives the same theta and S_max pair used above, so the
        # memoized helper resolves to the identical immutable domain object.
        context_layer, attention_probs = attention_interface(
            self, query_layer, key_layer, value_layer, attention_mask,
            dropout=0.0 if not self.training else self.dropout_prob,
            **kwargs,
        )
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # Merging attention heads changes only layout; keep the selected domain
        # without inspecting the context tensor's runtime extrema.
        return Potential(context_layer, context_domain), attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SpikingLinear(config.hidden_size, config.hidden_size, theta=getattr(config, "theta", 400.0))
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
        pot_dense = self.dense(pot)
        dropped = self.dropout(pot_dense.value)
        val = dropped + pot_skip.value
        domain = PotentialBounds(
            pot_dense.domain.min + pot_skip.domain.min,
            pot_dense.domain.max + pot_skip.domain.max,
        )
        return _apply_norm(self.LayerNorm, Potential(val, domain))


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, pot: Potential, attention_mask=None) -> tuple[Potential, torch.Tensor]:
        pot_attn, attention_probs = self.self(pot, attention_mask)
        pot_out = self.output(pot_attn, pot)
        return pot_out, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SpikingLinear(config.hidden_size, config.intermediate_size, theta=getattr(config, "theta", 400.0))
        self._use_spiking_mlp = getattr(config, "use_spiking_mlp", True)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, pot: Potential) -> Potential:
        """Apply the BERT feed-forward activation on a fixed analytic range.

        The spiking GELU composition already returns its propagated interval. ReLU
        maps the affine endpoints monotonically, while dense GELU remains between
        its input and zero because it multiplies the input by a gate in ``[0, 1]``.
        No branch may derive physical metadata from the current output tensor.

        Args:
            pot: Normalized hidden states paired with a fixed input range.

        Returns:
            The activated intermediate tensor and its input-derived fixed range.

        Raises:
            ValueError: If the configured activation has no maintained analytic rule.
        """
        # The affine adapter freezes an exact output interval for the declared input
        # range. All activation branches below consume only those fixed endpoints.
        pot_z = self.dense(pot)

        # The operator-backed GELU propagates the intervals of its multiplication,
        # Tanh, and addition stages directly, so no additional envelope is needed.
        if self._use_spiking_mlp:
            if isinstance(self.intermediate_act_fn, GELUActivation):
                return Potential(*gelu_approximation(*pot_z))
            if isinstance(self.intermediate_act_fn, nn.ReLU):
                # ReLU is monotone and clips negative inputs to the fixed zero rail.
                return Potential(
                    pot_z.value.relu(),
                    PotentialBounds(
                        max(0.0, float(pot_z.domain.min)),
                        max(0.0, float(pot_z.domain.max)),
                    ),
                )

        # Dense execution still needs the same physical range contract. Standard
        # GELU is x times a normal-CDF gate, so its output lies between x and zero;
        # dense ReLU uses the same monotone endpoint mapping as the spiking branch.
        out = self.intermediate_act_fn(pot_z.value)
        if isinstance(self.intermediate_act_fn, GELUActivation):
            output_domain = PotentialBounds(
                min(float(pot_z.domain.min), 0.0),
                max(float(pot_z.domain.max), 0.0),
            )
        elif isinstance(self.intermediate_act_fn, nn.ReLU):
            output_domain = PotentialBounds(
                max(0.0, float(pot_z.domain.min)),
                max(0.0, float(pot_z.domain.max)),
            )
        else:
            # An unknown activation needs an explicit mathematical or calibrated
            # envelope; silently observing this batch would restore the removed bug.
            raise ValueError(
                "BERT intermediate activation requires a maintained analytic range rule"
            )
        return Potential(out, output_domain)


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SpikingLinear(config.intermediate_size, config.hidden_size, theta=getattr(config, "theta", 400.0))
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
        pot_dense = self.dense(pot_inter)
        dropped = self.dropout(pot_dense.value)
        val = dropped + pot_skip.value
        domain = PotentialBounds(
            pot_dense.domain.min + pot_skip.domain.min,
            pot_dense.domain.max + pot_skip.domain.max,
        )
        return _apply_norm(self.LayerNorm, Potential(val, domain))


class BertLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, pot: Potential, attention_mask=None) -> tuple[Potential, torch.Tensor]:
        pot_attn, attention_probs = self.attention(pot, attention_mask)
        pot_inter = self.intermediate(pot_attn)
        pot_layer = self.output(pot_inter, pot_attn)
        return pot_layer, attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._theta = float(getattr(config, "theta", 10.0))
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Potential | torch.Tensor,
        attention_mask=None,
    ) -> Potential:
        """Enter the BERT stack through an upstream or fixed calibrated range.

        Normal model execution may carry the embedding LayerNorm range as a
        :class:`Potential`. A direct tensor call has no upstream metadata, so it uses
        the configured symmetric ``theta`` rail. An installed calibration binding
        observes or clamps the same encoder-entry tensor without consulting its live
        extrema.

        Args:
            hidden_states: Embedding output with optional declared potential bounds.
            attention_mask: Broadcast attention suppression tensor for every block.

        Returns:
            Final BERT encoder activation with fixed propagated bounds.

        Raises:
            ValueError: If the configured fallback threshold is not finite and
                positive.
        """
        # Preserve a range already established by the embedding LayerNorm. Standalone
        # tensor callers instead receive the same configuration-derived physical rail
        # for every batch, independent of its values or ordering.
        if isinstance(hidden_states, Potential):
            entry_value = hidden_states.value
            entry_bounds = hidden_states.domain
        else:
            if not isinstance(hidden_states, torch.Tensor):
                raise TypeError("hidden_states must be Potential or torch.Tensor")
            if not math.isfinite(self._theta) or self._theta <= 0.0:
                raise ValueError("BERT encoder theta must be finite and positive")
            entry_value = hidden_states
            entry_bounds = PotentialBounds(-self._theta, self._theta)

        # Collection observes raw values on the fixed upstream safety interval;
        # frozen phases attach their persisted range. Without a binding, only direct
        # tensor calls need clamping because a Potential is already synchronized.
        if model_calibration_is_bound(self):
            pot = calibrated_potential(
                self,
                "input",
                entry_value,
                collection_bounds=entry_bounds,
            )
        elif isinstance(hidden_states, Potential):
            pot = hidden_states
        else:
            pot = Potential(
                entry_bounds.clamp(entry_value, name="bert_encoder_input"),
                entry_bounds,
            )

        # Every layer consumes and returns Potential, so the entry range remains part
        # of the operator graph instead of being reconstructed from a later tensor.
        for layer_module in self.layer:
            pot, _ = layer_module(pot, attention_mask)
        return pot


class BertPooler(nn.Module):
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
        """Pool the first BERT token without rebuilding its potential range.

        The first-token slice is a view of the final encoder activation and therefore
        retains the encoder's declared range. Direct tensor calls use the configured
        fixed ``theta`` rail for compatibility, while the dense pooler remains an
        ordinary PyTorch path that does not encode a temporal event.

        Args:
            hidden_states: Final sequence activation with optional fixed bounds.

        Returns:
            Dense or spiking Tanh-pooled first-token representation.

        Raises:
            TypeError: If ``hidden_states`` is neither Potential nor Tensor.
        """
        # Slicing the token dimension cannot enlarge the declared scalar envelope.
        # Preserve the same PotentialBounds object when the encoder supplied one.
        if isinstance(hidden_states, Potential):
            first_token_tensor = hidden_states.value[:, 0]
            first_token_domain = hidden_states.domain
        elif isinstance(hidden_states, torch.Tensor):
            first_token_tensor = hidden_states[:, 0]
            first_token_domain = None
        else:
            raise TypeError("hidden_states must be Potential or torch.Tensor")

        # The spiking projection consumes the inherited range directly. Tanh owns
        # its structural [-1, 1] output rail, so only its tensor result crosses the
        # Hugging Face-compatible pooler API boundary.
        if self.use_spiking_mlp:
            if first_token_domain is None:
                theta = float(self.dense.theta)
                if not math.isfinite(theta) or theta <= 0.0:
                    raise ValueError("BERT pooler theta must be finite and positive")
                first_token_domain = PotentialBounds(-theta, theta)
                first_token_tensor = first_token_domain.clamp(
                    first_token_tensor,
                    name="bert_pooler_input",
                )
            first_token_potential = Potential(
                first_token_tensor,
                first_token_domain,
            )
            pot_dense = self.dense(first_token_potential)
            pooled_output, _ = tanh(pot_dense.value, pot_dense.domain, tau_s=self.tau_s, theta=self.dense.theta)
            return pooled_output

        # Dense execution intentionally ignores range metadata after selecting the
        # first-token tensor because neither Linear nor Tanh emits a spike here.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@auto_docstring
class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
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
        elif isinstance(module, BertEmbeddings):
            init.trunc_normal_(module.word_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            init.trunc_normal_(module.position_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            init.trunc_normal_(module.token_type_embeddings.weight, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            return_potential=True,
        )
        if not isinstance(embedding_output, Potential):
            raise RuntimeError("BERT internal embeddings must return Potential")
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embedding_output.value.dtype).min
        else:
            extended_attention_mask = None

        pot = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = pot.value
        # The pooler slices the first token from the final Potential so its spiking
        # projection consumes the encoder's fixed range without measuring the slice.
        pooled_output = self.pooler(pot) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

@auto_docstring
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

__all__ = ["SpikingLayerNorm", "SpikingLinear", "BertModel", "BertForSequenceClassification", "BertPreTrainedModel"]
