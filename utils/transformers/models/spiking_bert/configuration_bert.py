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
"""BERT model configuration"""

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BertConfig(PreTrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
        is_decoder=False,
        add_cross_attention=False,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=True,
        use_spiking_layernorm=True,
        spiking_ln_mul=True,
        spiking_ln_log=True,
        spiking_ln_expdiff=True,
        use_spiking_mlp=True,
        theta=10.0,
        tau_s=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        
        # Spiking parameters
        self.use_spiking_layernorm = use_spiking_layernorm
        self.spiking_ln_mul = spiking_ln_mul
        self.spiking_ln_log = spiking_ln_log
        self.spiking_ln_expdiff = spiking_ln_expdiff
        self.use_spiking_mlp = use_spiking_mlp
        self.theta = theta
        self.tau_s = tau_s

__all__ = ["BertConfig"]
