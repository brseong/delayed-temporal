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
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import initialization as init
from transformers.activations import ACT2FN
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

from utils.transforms.functions import gelu_approximation
from utils.transforms.types import Potential, PotentialBounds
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=inputs_embeds.device if inputs_embeds is not None else input_ids.device,
            ).unsqueeze(0).expand(input_shape[0], -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=inputs_embeds.device if inputs_embeds is not None else input_ids.device,
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        embeddings = _apply_norm(self.LayerNorm, Potential(embeddings, PotentialBounds(embeddings.min().item(), embeddings.max().item()))).value
        embeddings = self.dropout(embeddings)
        return embeddings


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

        kwargs = {}
        if self.config._attn_implementation == "spiking_sdpa":
            kwargs["theta"] = getattr(self.config, "theta", 10.0)
            kwargs["tau_m"] = getattr(self.config, "tau_s", 1.0)

        context_layer, attention_probs = attention_interface(
            self, query_layer, key_layer, value_layer, attention_mask,
            dropout=0.0 if not self.training else self.dropout_prob,
            **kwargs,
        )
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        return Potential(context_layer, pot_v.domain), attention_probs


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
        pot_z = self.dense(pot)
        if self._use_spiking_mlp:
            return Potential(*gelu_approximation(*pot_z))
        out = self.intermediate_act_fn(pot_z.value)
        return Potential(out, PotentialBounds(out.min().item(), out.max().item()))


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
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> Potential:
        pot = Potential(
            hidden_states,
            PotentialBounds(hidden_states.min().item(), hidden_states.max().item()),
        )
        for layer_module in self.layer:
            pot, _ = layer_module(pot, attention_mask)
        return pot


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
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
        )
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embedding_output.dtype).min
        else:
            extended_attention_mask = None

        pot = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = pot.value
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

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
