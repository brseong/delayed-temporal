# Model Integration

The model layer adapts pretrained Hugging Face architectures so their Transformer internals can exchange bounded potentials and select spiking operator implementations.

## Shared State Carrier

Transformer blocks use `Potential(value, domain)` internally and unwrap tensor values only after the final learned projection.

Embeddings initialize the carrier, spiking projections and residuals propagate or combine its bounds, and task heads consume the final `Potential`. Shared layers are defined by [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]], [[utils/transformers/models/spiking_ops.py#SpikingLinear]], and [[utils/transformers/models/spiking_ops.py#SpikingConv2d]].

This wrapper avoids changing pretrained parameter storage and Hugging Face output types. It also makes domain metadata explicit enough for operator composition and clamp analysis.

## Checkpoint Compatibility

Spiking projection classes subclass or mirror the dense modules they replace so pretrained state dictionaries load without a conversion-training stage.

The project reconstructs model families with familiar module names and parameter shapes, then invokes `from_pretrained` on the local class. Embedding lookup, dropout, losses, and output dataclasses retain their framework roles; every learned affine output projection in the maintained wrappers uses the TTFS linear composition.

Compatibility here means parameter-layout and API compatibility. It does not imply that every finite-domain spiking forward is numerically identical to the source ANN under clipping, approximate activations, or noise.

## Supported Model Families

The current adapters cover image classification, encoder text classification, and decoder language modeling.

### ViT

The ViT path is the most complete operator composition and robustness target.

[[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTModel]] retains patch and position embeddings, runs a `Potential` through a stack of spiking-aware blocks, applies configurable final normalization, and returns a Hugging Face-compatible output. [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTForImageClassification]] passes the class-token `Potential` to a final [[utils/transformers/models/spiking_ops.py#SpikingLinear]] classifier.

When ViT selects the spiking attention backend, [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTSelfAttention#forward]] derives $S_{\max}$ from the configured patch grid plus the class token and attaches the memoized fixed attention-output range. Eager attention retains the projected-value range.

ViT blocks are pre-norm: normalization precedes attention and MLP, and both residual ranges are combined by interval addition. Its MLP can use the cubic spiking GELU approximation, the same formula evaluated directly, or the configured dense activation.

### BERT and RoBERTa

BERT and RoBERTa preserve their post-norm encoder structure while replacing projections, optional normalization, attention, and MLP behavior.

[[utils/transformers/models/spiking_bert/modeling_spiking_bert.py#BertModel]] and [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaModel]] initialize bounded potentials after embeddings and return standard sequence-classification outputs through their task wrappers.

Their self-attention adapters use `max_position_embeddings` as fixed $S_{\max}$ for the spiking output rail. The eager path retains the projected-value range in evaluation and expands it analytically by $1/(1-p)$ during training. The spiking path is fixed for evaluation; nonzero training dropout remains outside the paper scope.

Both adapters support accuracy experiments on SST-2, AG News, and IMDB through parallel runners. Their model configs expose stage-level LayerNorm switches, attention backend selection, MLP selection, and `tau_s`. Legacy serialized `theta` keys are rejected.

### GPT-2

GPT-2 adapts causal self-attention, cache-aware decoding, pre-norm blocks, and the Hugging Face `Conv1D` projection layout.

[[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Model]] wraps token-plus-position embeddings, propagates a bounded potential through causal blocks, and returns standard cache-aware outputs. [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2LMHeadModel]] preserves the tied input/output weight while its final [[utils/transformers/models/spiking_ops.py#SpikingLinear]] consumes the encoder `Potential`.

[[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Attention#forward]] uses local Q/K/V projection bounds and `max_position_embeddings` for the score representability ceiling. Eager attention and residual dropout propagate analytic ranges without runtime extrema; nonzero spiking attention training dropout remains outside the paper scope.

The adapter does not support cross-attention in its spiking `GPT2Attention`. With spiking MLP enabled, the paper configuration's `gelu_new` activation uses [[utils/transforms/functions.py#gelu_approximation]]; dense MLP ablations and other configured activations retain direct evaluation with a distinct persisted identity.

### Llama

The Llama path preserves decoder language modeling, attention with shared key and value heads, rotary position embeddings, cached autoregressive generation, RMSNorm, and the pretrained gated feed forward structure.

[[utils/transformers/models/spiking_llama/modeling_spiking_llama.py#LlamaModel]] carries fixed potential ranges from the token embedding table through pre-norm decoder blocks. Every learned projection, including the language model head, uses [[utils/transformers/models/spiking_ops.py#SpikingLinear]] with the original parameter names and shapes.

[[utils/transformers/models/spiking_llama/modeling_spiking_llama.py#LlamaMLP]] sends the gate and up projections to the established [[utils/transforms/functions.py#swiglu_function]] composition, while the down projection consumes its propagated potential range. RMSNorm now uses [[rmsnorm#RMSNorm Operator Composition]] through [[utils/transforms/functions.py#rmsnorm_function]], with pretrained fixed gains and an analytic fixed output range.

[[utils/transformers/models/spiking_llama/modeling_spiking_llama.py#LlamaAttention]] propagates conservative query and key ranges through the rotary transformation, keeps value ranges synchronized with the cache, and can select the shared spiking attention backend. The shared backend expands key and value heads to the query head count before its temporal score composition. [[scripts/verification/verify_spiking_llama.py#main]] loads one state dictionary into the Hugging Face and local models, checks deterministic logits, validates the SwiGLU feed forward computation against the dense formula, and exercises cache growth.

[[utils/transformers/models/spiking_llama/calibration.py#llama_calibration_specs]] declares residual, gate/up, rotary query/key, value, and attention score sites. A frozen table from the training population replaces the corresponding analytic execution ranges at those sites; embedding lookup, RMSNorm output, and other operator ranges remain analytic. The same table is used for clean and timing noise evaluation.

## Attention Backend Selection

Attention is registered as a Hugging Face backend named `spiking_sdpa` and selected through each model’s configuration.

Evaluation runners choose `spiking_sdpa` only for the spiking backend on non-CPU devices; otherwise they use eager tensor attention. Model adapters derive one attention `tau` from model-wide `tau_s` and pass Q/K/V bounds plus fixed `source_length_max` to [[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_sdpa_attention_forward]], whose module binding supplies score calibration.

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
- `tau_s`

The four maintained configs reject `theta`; GPT-2 also rejects `attention_theta`. This fail-closed behavior prevents an old checkpoint or runner from silently restoring a global range.

ViT additionally distinguishes an operator-composed cubic GELU from a direct evaluation of the same tanh formula. Experiments must log the full flag set because several configurations can all be described informally as a “spiking model” while executing different arithmetic.

## Framework Boundaries

The project keeps dataset preprocessing, embedding lookup, losses, tensor indexing or rearrangement, and output containers within standard PyTorch and Hugging Face conventions.

Learned patch projection, affine Transformer operations, normalization, and every maintained output projection use TTFS compositions. Reported costs must still state which input construction, loss, and control operations are excluded; an operator count is not a measured end-to-end hardware cost.
