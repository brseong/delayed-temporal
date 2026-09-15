# Text Model Calibration

BERT, RoBERTa and GPT-2 collect fixed training ranges and consume the same ranges in attention, LayerNorm and nonlinear inputs. This contract covers collection, persistence and evaluation rather than statistics alone.

## Shared Range Binding

Text tables use their actual model family and `text_calibration_policy_version=1`. They retain output bound policy 3; ViT policy 2 remains a separate contract.

[[utils/transformers/calibration.py#calibration_uses_explicit_bounds]] enables selected ranges without treating text models as ViT. [[utils/transformers/calibration.py#bind_model_calibration]] validates all selected text ranges, execution dtype, LayerNorm epsilon and positive floor before publishing any bindings. Zero, asymmetric, nonfinite or numerically unrepresentable ranges fail with the site identity.

Selected Q/K/V ranges survive head reshaping and are passed together to attention. They govern multiplication encoding, the zero reference used to restore V and the resulting context bound. Attention score selection retains its dtype, time constant and maximum sequence numerical limit without an additional global threshold cap.

Each active LayerNorm observes signed centered inputs before clipping. Its selected upper endpoint governs magnitude, square, variance and logarithmic encoding with a shared time window. The fixed positive input floor and variance floor remain unchanged, as do the learned affine coefficients and derived output range. Checkpoint epsilon is forwarded to every normalization, including embedding and output head normalization when present.

[[scripts/verification/verify_text_calibration_binding.py#verify_text_layernorm_binding]] checks all eight normalization combinations for each family, selected ranges above and below the global threshold, parity with noise disabled and with zero standard deviation, and seeded finite outputs. Incompatible family, version, dtype, epsilon or floor fails atomically.

## Encoder Coverage

BERT and RoBERTa discover sites from active modules, including embedding normalization and nonlinear task heads. Their sequence classification configurations with twelve fully spiking blocks each contain 110 selected sites.

[[utils/transformers/models/text_calibration.py#text_calibration_specs]] registers independent Q/K/V, attention scores, both residual sums before normalization, the composed GELU input and active centered LayerNorm inputs. The first token pooler or classification head Tanh input is included. RoBERTa masked language prediction additionally includes the head GELU input and head normalization, giving 111 sites with twelve fully spiking blocks.

Softmax, Tanh, GELU and normalized outputs keep their fixed or input derived output bounds. Embedding lookup ranges remain parameter derived. Attention output projection is covered by the residual sum before the next normalization; it is not replaced with an unrelated independently selected range. Disabled temporal attention and fully dense normalization do not register unexecuted sites. Decoder or cross attention configurations are rejected rather than silently collecting incomplete tables.

The existing composed encoder GELU remains distinct from ViT's configured construction. Its model threshold and time constant are forwarded explicitly; this calibration change does not claim identical GELU implementations across model families.

BERT and RoBERTa cast the attention mask to the embedding dtype before constructing its additive values. With float64, subtracting an integer mask in float32 before multiplying by the float64 minimum could otherwise create nonfinite values at unmasked positions. This numerical correction also applies when temporal attention is disabled.

## GPT-2 Coverage

GPT-2 discovers independent Q/K/V, scores, two residuals, the first MLP affine output and active centered LayerNorm inputs. Twelve fully spiking blocks contain 109 selected sites, including the final normalization.

[[utils/transformers/models/spiking_gpt2/calibration.py#gpt2_calibration_specs]] separates the fused projection's three outputs before reshaping. The existing dense `gelu_new` activation remains dense, but its affine input still consumes the selected range. Token and position embeddings retain parameter derived ranges; output logits are not independently calibrated.

Cached attention tensors may only be reused with the same selected range identity. A changed range is rejected before combining cached and new tensors. Collection disables caching. Both float32 and float64 are supported, with actual execution dtype and checkpoint epsilon persisted in metadata.

Collection rejects a missing Q/K/V site or mismatched policy, family, dtype or configuration before forwarding any batch. Calibrated evaluation rejects nonfinite logits or loss before accumulation and rejects incomplete or nonfinite final metrics. Finite batch mean loss remains unchanged; historical uncalibrated evaluation retains its earlier behavior.

## Evaluator Lifecycle

Collection uses one seeded training subset in two deterministic passes. Evaluation only consumes a persisted table with matching source model, data, preprocessing, numerical settings and site identities.

[[utils/transformers/models/text_calibration.py#collect_text_calibration_table]] collects raw inputs with static analytic bounds in both passes and validates sample order and counts. Labels are excluded from collection forwards. Ranges are selected together after collection, using the existing minimum, maximum, 2,048-bin histogram and margin of five percent of the interval width on each side defaults. A short diagnostic table is not a complete 5k training calibration.

[[scripts/evaluation/text_calibration_runtime.py#run_text_calibration]] supplies the BERT/RoBERTa `collect`, `validate` and `inference` lifecycle. Collection exits before loading the held out split or reporting task accuracy. Existing tables are not overwritten. Loading rejects missing sites and mismatched family, version, checkpoint, configuration, dtype, preprocessing or collection controls. Historical files remain available for their original implementation but are not silently reused by the new evaluator.

BERT/RoBERTa print flushed cumulative correct/total and accuracy after each evaluation batch, followed by a prediction digest. GPT-2 prints flushed batch mean loss and its exponent, preserving the existing metric rather than relabeling it as token weighted corpus perplexity. `--no-tensorboard` suppresses TensorBoard files. W&B can remain disabled without suppressing local metrics. Calibration also prints pass and sample progress immediately.

## Validation Scope

Tests must establish executed site coverage, selected range consumption and strict persistence before reporting calibrated accuracy. Passing small checks does not establish full dataset performance or update the manuscript.

Focused model tests exercise both dtypes, attention and normalization ablations, collection before clipping, identical inputs in both passes, frozen replay, save/load identity and rejection of missing tables. The existing ViT, shared LayerNorm, attention, Gaussian and calibration checks protect unaffected paths. Diagnostics with actual checkpoints use idle local GPUs 4–7 with shared device locks and runtime on disk under artifacts. Existing experiments and their frozen sources remain unchanged.

[[scripts/verification/smoke_text_calibration.py#main]] performs only bounded local checks using cached assets. It records exact commands, source hashes, device admission, separate collection and evaluation logs and phase exit status. The working source may be uncommitted, but every relevant file must remain unchanged throughout the check; results explicitly prohibit manuscript reuse. It rejects occupied devices, existing output paths and memory filesystems, and passes the shared GPU lock to the evaluator process.

On 2026-09-15, cached BERT SST-2, RoBERTa SST-2 and GPT-2 WikiText checkpoints each completed collection on 16 training examples followed by frozen evaluation on two batches of two examples, with float64, theta 40 and sequence length 32. All 110, 110 and 109 respective sites had positive execution counts with no missing or duplicate site reports. Separate logs, tables, hashes and completion records are preserved under `artifacts/logs/text_calibration_smoke/20260915_policy1_bert`, `20260915_policy1_roberta` and `20260915_policy1_gpt2`. These checks establish execution only, not full task accuracy or adequate calibration sample size.
