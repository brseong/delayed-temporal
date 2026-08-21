# Model Integration

The model layer adapts pretrained Hugging Face architectures so their Transformer internals can exchange bounded potentials and select spiking operator implementations.

## Shared State Carrier

Transformer blocks use `Potential(value, domain)` internally and unwrap the tensor at framework-facing boundaries.

Embeddings initialize the carrier, spiking projections and residuals propagate or combine its bounds, and task heads consume the final `.value`. Shared layers are defined by [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]], [[utils/transformers/models/spiking_ops.py#SpikingLinear]], and [[utils/transformers/models/spiking_ops.py#SpikingConv2d]].

This wrapper avoids changing pretrained parameter storage and Hugging Face output types. It also makes domain metadata explicit enough for operator composition and clamp analysis.

## Checkpoint Compatibility

Spiking projection classes subclass or mirror the dense modules they replace so pretrained state dictionaries load without a conversion-training stage.

The project reconstructs model families with familiar module names and parameter shapes, then invokes `from_pretrained` on the local class. Embedding layers, classifier or language-model heads, dropout, and output dataclasses remain conventional unless a model adapter explicitly replaces them.

Compatibility here means parameter-layout and API compatibility. It does not imply that every finite-domain spiking forward is numerically identical to the source ANN under clipping, approximate activations, or noise.

## Supported Model Families

The current adapters cover image classification, encoder text classification, and decoder language modeling.

### ViT

The ViT path is the most complete operator composition and robustness target.

[[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTModel]] retains standard patch and position embeddings, runs a `Potential` through a stack of spiking-aware blocks, applies configurable final normalization, and returns a Hugging Face-compatible output. [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTForImageClassification]] keeps the final classifier conventional.

When ViT selects the spiking attention backend, [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTSelfAttention#forward]] derives $S_{\max}$ from the configured patch grid plus the class token and attaches the memoized fixed attention-output range. Eager attention retains the projected-value range.

ViT blocks are pre-norm: normalization precedes attention and MLP, and both residual ranges are combined by interval addition. Its MLP can use the cubic spiking GELU approximation, the same formula evaluated directly, or the configured dense activation.

### BERT and RoBERTa

BERT and RoBERTa preserve their post-norm encoder structure while replacing projections, optional normalization, attention, and MLP behavior.

[[utils/transformers/models/spiking_bert/modeling_spiking_bert.py#BertModel]] and [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaModel]] initialize bounded potentials after embeddings and return standard sequence-classification outputs through their task wrappers.

Their self-attention adapters use `max_position_embeddings` as fixed $S_{\max}$ for the spiking output rail. Eager evaluation retains the projected-value range, while training dropout expands it analytically by $1/(1-p)$ and includes zero.

Both adapters support accuracy experiments on SST-2, AG News, and IMDB through parallel runners. Their model configs expose stage-level LayerNorm switches, attention backend selection, MLP selection, `theta`, and `tau_s`.

### GPT-2

GPT-2 adapts causal self-attention, cache-aware decoding, pre-norm blocks, and the Hugging Face `Conv1D` projection layout.

[[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Model]] wraps token-plus-position embeddings, propagates a bounded potential through causal blocks, and returns standard cache-aware outputs. [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2LMHeadModel]] retains the tied conventional language-model head.

[[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Attention#forward]] uses `max_position_embeddings` for the spiking attention rail, preserves the combined projection range for dense attention, and propagates attention and residual dropout ranges analytically without runtime extrema.

The adapter does not support cross-attention in its spiking `GPT2Attention`. Its current MLP uses spiking projections when enabled but evaluates the configured activation directly, so model-family claims must record which nonlinear path is actually operator-composed.

## Attention Backend Selection

Attention is registered as a Hugging Face backend named `spiking_sdpa` and selected through each model’s configuration.

Evaluation runners choose `spiking_sdpa` only for the spiking backend on non-CPU devices; otherwise they use eager tensor attention. Model self-attention classes pass `theta` and `tau_s` into [[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_sdpa_attention_forward]].

This backend boundary keeps Q/K/V projection ownership in each model while centralizing score normalization and value accumulation. It also allows attention to be disabled independently during ablations.

## Configuration and Ablations

Configuration flags make operator stages independently replaceable so conversion error can be localized instead of measured only end to end.

The shared controls are:

- `use_spiking_layernorm`
- `spiking_ln_mul`
- `spiking_ln_log`
- `spiking_ln_expdiff`
- `use_spiking_mlp`
- attention implementation selection
- `theta` and `tau_s`

ViT additionally distinguishes an operator-composed cubic GELU from a direct evaluation of the same tanh formula. Experiments must log the full flag set because several configurations can all be described informally as a “spiking model” while executing different arithmetic.

## Conventional Boundaries

The project deliberately keeps dataset preprocessing, embeddings, losses, output containers, and most task heads within standard PyTorch and Hugging Face conventions.

This narrows the research question to Transformer operator replacement and permits direct loading of pretrained models. Reported system costs must therefore state whether these conventional boundaries are included; an operator-only count is not an end-to-end hardware estimate.
