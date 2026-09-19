# Text Model Calibration

BERT, RoBERTa and GPT-2 collect fixed training ranges and consume the same ranges in attention, LayerNorm and nonlinear inputs. This contract covers collection, persistence and evaluation rather than statistics alone.

## Shared Range Binding

Text tables use their actual model family and `text_calibration_policy_version=1`. They retain output bound policy 3; ViT policy 2 remains a separate contract.

[[utils/transformers/calibration.py#calibration_uses_explicit_bounds]] enables selected ranges without treating text models as ViT. [[utils/transformers/calibration.py#bind_model_calibration]] validates all selected text ranges, execution dtype, LayerNorm epsilon and positive floor before publishing any bindings. Zero, asymmetric, nonfinite or numerically unrepresentable ranges fail with the site identity.

Selected Q/K/V ranges survive head reshaping and are passed together to attention. They govern multiplication encoding, the zero reference used to restore V and the resulting context bound. Attention score selection retains its dtype, time constant and maximum sequence numerical limit without an additional global threshold cap.

Each active LayerNorm observes signed centered inputs before clipping. Its selected upper endpoint governs magnitude, square, variance and logarithmic encoding with a shared time window. The fixed positive input floor and variance floor remain unchanged, as do the learned affine coefficients and derived output range. Checkpoint epsilon is forwarded to every normalization, including embedding and output head normalization when present.

[[scripts/verification/verify_text_calibration_binding.py#verify_text_layernorm_binding]] checks all eight normalization combinations for each family, selected ranges above and below the global threshold, parity with noise disabled and with zero standard deviation, and seeded finite outputs. Incompatible family, version, dtype, epsilon or floor fails atomically.

## Encoder Coverage

BERT and RoBERTa discover sites from active modules, including embedding normalization and nonlinear task heads. Sequence classifiers contain 110 selected sites with twelve fully spiking blocks and 218 with twenty-four blocks.

[[utils/transformers/models/text_calibration.py#text_calibration_specs]] registers independent Q/K/V, attention scores, both residual sums before normalization, the composed GELU input and active centered LayerNorm inputs. The first token pooler or classification head Tanh input is included. RoBERTa masked language prediction additionally includes the head GELU input and head normalization, giving 111 sites with twelve fully spiking blocks.

Softmax, Tanh, GELU and normalized outputs keep their fixed or input derived output bounds. Embedding lookup ranges remain parameter derived. Attention output projection is covered by the residual sum before the next normalization; it is not replaced with an unrelated independently selected range. Disabled temporal attention and fully dense normalization do not register unexecuted sites. Decoder or cross attention configurations are rejected rather than silently collecting incomplete tables.

The spiking MLPs in ViT, BERT, RoBERTa, and GPT-2 now resolve the same canonical Power cubic in [[utils/transforms/functions.py#gelu_approximation]]. Their model threshold and time constant are forwarded explicitly.

BERT and RoBERTa cast the attention mask to the embedding dtype before constructing its additive values. With float64, subtracting an integer mask in float32 before multiplying by the float64 minimum could otherwise create nonfinite values at unmasked positions. This numerical correction also applies when temporal attention is disabled.

## GPT-2 Coverage

GPT-2 discovers independent Q/K/V, scores, two residuals, the first MLP affine output and active centered LayerNorm inputs. Twelve fully spiking blocks contain 109 selected sites, including the final normalization.

[[utils/transformers/models/spiking_gpt2/calibration.py#gpt2_calibration_specs]] separates the fused projection's three outputs before reshaping. The paper configuration's `gelu_new` activation consumes the selected affine input range and executes [[utils/transforms/functions.py#gelu_approximation]]. A dense MLP ablation remains direct. Token and position embeddings retain parameter derived ranges; output logits are not independently calibrated.

Cached attention tensors may only be reused with the same selected range identity. A changed range is rejected before combining cached and new tensors. Collection disables caching. Both float32 and float64 are supported, with actual execution dtype and checkpoint epsilon persisted in metadata.

Collection rejects a missing Q/K/V site or mismatched policy, family, dtype or configuration before forwarding any batch. Calibrated evaluation rejects nonfinite logits or loss before accumulation and rejects incomplete or nonfinite final metrics. Finite batch mean loss remains unchanged; historical uncalibrated evaluation retains its earlier behavior.

## Evaluator Lifecycle

Collection uses one seeded training subset in two deterministic passes. Evaluation only consumes a persisted table with matching source model, data, preprocessing, numerical settings and site identities.

[[utils/transformers/models/text_calibration.py#collect_text_calibration_table]] collects raw inputs with static analytic bounds in both passes and validates sample order and counts. Labels are excluded from collection forwards. Ranges are selected together after collection, using the existing minimum, maximum, 2,048-bin histogram and margin of five percent of the interval width on each side defaults. A short diagnostic table is not a complete 5k training calibration.

[[scripts/evaluation/text_calibration_runtime.py#run_text_calibration]] supplies the BERT/RoBERTa `collect`, `validate` and `inference` lifecycle. Collection exits before loading the held out split or reporting task accuracy. Existing tables are not overwritten. Loading rejects missing sites and mismatched family, version, checkpoint, configuration, dtype, preprocessing or collection controls. Historical files remain available for their original implementation but are not silently reused by the new evaluator.

BERT/RoBERTa print flushed cumulative correct/total and accuracy after each evaluation batch, followed by a prediction digest. GPT-2 prints batch loss during execution and records total negative log likelihood and valid-token count so the final token-weighted corpus perplexity is independent of batch partitioning. The historical mean-of-batch-loss perplexity remains a separately named compatibility metric. `--no-tensorboard` suppresses TensorBoard files. W&B can remain disabled without suppressing local metrics. Calibration also prints pass and sample progress immediately.

[[utils/transformers/tokenizer_identity.py#tokenizer_backend_sha256]] hashes the tokenizer structure without its mutable padding and truncation state for each request. Those request settings, maximum length, sides and special token identifiers remain explicit metadata and must still match. Vocabulary, model, normalizer and other structural changes remain part of the hash. [[scripts/verification/verify_text_tokenizer_identity.py#verify_request_state_independence]] checks cached and newly encoded input paths against the same identity, while altered preprocessing settings or tokenization structure are rejected.

The first 256-example GPT-2 comparison at source `014f428` stopped before SNN inference because the old backend hash depended on whether dataset tokenization used its cache. Its partial logs are preserved separately. The corrected implementation requires fresh collection and does not rewrite or relabel that failed attempt. [[calibration#Layer-wise Calibration#Frozen Execution#Runtime Record Lookup]] also removes repeated complete table validation from frozen activation execution without changing numerical results.

## Complete Comparison Execution

The completed comparison collected fresh ranges from 5,000 training examples per model and evaluated every example in each pinned held-out artifact.

[[scripts/evaluation/text_calibration_runtime.py#load_text_dataset_artifact]] loads a single saved Dataset only when its stored fingerprint and exact sample count match the immutable model manifest. It does not replace the requested artifact with a network download or another cache entry.

Tokenization disables on-disk `Dataset.map` cache creation and keeps transformed batches in memory. The runner compares the complete saved Dataset file set and hashes before and after every phase, so a newly created cache file invalidates the run instead of silently changing its data artifact.

GPT-2 progress validation extracts each JSON object after an optional progress-bar prefix. It still requires exactly one valid record per expected batch and rejects malformed, missing or duplicate records.

[[scripts/experiments/run_full_calibrated_text_comparison.py#main]] runs collection, ANN evaluation and SNN evaluation sequentially on one GPU while allowing different models to run in parallel. It preserves each attempt log, reuses only hash-validated completed phases, and writes flushed progress records throughout evaluation.

RoBERTa-L completed under the separate `roberta_large_theta40_calibrated_float64_bounds3_v1` tag with its own checkpoint and 218-site calibration table. ANN and SNN both record 841/872, or 96.4450%, on SST-2 validation. This diagnostic does not reuse the RoBERTa-B table or imply that its checkpoint matches SpikeZIP-TF.

[[scripts/analysis/summarize_full_calibrated_text_comparison.py#build]] authenticates every phase log and calibration table again before producing raw, summary, calibration-site and provenance artifacts. Partial model results may be inspected but cannot satisfy the complete campaign gate.

GPT-2 records token-weighted corpus perplexity as the primary complete-run metric and retains the historical mean of batch losses as a compatibility metric. Both values come from the same model forwards; padding positions are excluded from the token count.

The pinned WikiText-2 raw test revision contains 4,358 rows and 2,891 rows after excluding empty text. The complete campaign evaluates those 2,891 rows rather than padding the dataset to an assumed count.

The completed results are BERT 806/872 for both ANN and SNN, RoBERTa 824/872 for ANN and 823/872 for SNN, and GPT-2 token-weighted corpus perplexity 21.984180 for ANN and 21.984387 for SNN over 204,257 valid tokens. GPT-2 compatibility perplexity is 23.253149 for ANN and 23.253467 for SNN. The GPT-2 result used a direct `gelu_new` activation and is retained as partial conversion evidence rather than reused by the composed rerun.

## GPT-2 Composed GELU Rerun

The rerun replaces the direct GPT-2 block activation with the maintained composed GELU while preserving the checkpoint, data, calibration population and numerical settings.

[[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2MLP]] selects `composed_gelu_new_v1` only when the spiking MLP and `gelu_new` selected by the checkpoint are active. Calibration metadata persists this identity, so the earlier direct-activation table is incompatible even though both configurations use the same activation name and selected sites.

The dedicated tag is `gpt2_theta40_calibrated_float64_bounds3_composed_gelu_v1`. It collected 109 ranges from the fixed training 5,000 artifact and evaluated all 2,891 nonempty fixed WikiText-2 test texts. No completed phase was imported from the partial conversion tag.

The generated summary records corpus perplexity, computed from total negative log likelihood and valid token count, as 21.9841797962 for the ANN and 21.9843868967 for the converted model over 204,257 valid tokens. Compatibility perplexity is 23.2531494262 and 23.2534667831. These values equal the archived direct activation run at recorded precision, but the new artifact independently executes the composed activation and carries its own source and calibration hashes.

The comparison manifest may admit local GPUs 0 through 3 without changing the repository-wide default GPU set. This permission is confined to the versioned campaign and each evaluator retains one visible RTX A6000.

`scripts/experiments/ubai/full_calibrated_text_pair.sbatch` gives BERT and RoBERTa one GPU and four CPU cores each inside one two-GPU Slurm allocation. The portable environment and task scratch are expanded only below the node-local `/enroot` disk, and the exact job-owned directory is removed after both children terminate.

## Shared Power Cubic Rerun

The rerun reuses authenticated completed calibration tables while evaluating RoBERTa-B, RoBERTa-L, and GPT-2 with the same canonical Power cubic.

Tag `text_power_gelu_theta40_float64_reused_calibration_v1` runs only ANN and SNN phases at source `0e4329945b255666e86b3313d2f88c21dfe5deb1`; no collection phase exists. Each manifest records the original completed calibration source, manifest, result, collection log, and calibration hashes. The copied tables contain 110, 218, and 109 sites.

RoBERTa-B records 824/872 for ANN and 823/872 for SNN. RoBERTa-L records 841/872 for both. GPT-2 corpus perplexity is 21.9841797962 for ANN and 21.9843868967 for SNN over 204,257 valid tokens; compatibility perplexity is 23.2531494262 and 23.2534667831. These values match the preceding runs at recorded precision, so replacing the cubic built from two multiplication operators with the canonical Power path does not change these task metrics.

## Validation Scope

Tests establish executed site coverage, selected range consumption and strict persistence; only the completed campaign above establishes its held-out metrics.

Focused model tests exercise both dtypes, attention and normalization ablations, collection before clipping, identical inputs in both passes, frozen replay, save/load identity and rejection of missing tables. The existing ViT, shared LayerNorm, attention, Gaussian and calibration checks protect unaffected paths. Diagnostics with actual checkpoints use idle local GPUs 4–7 with shared device locks and runtime on disk under artifacts. Existing experiments and their frozen sources remain unchanged.

[[scripts/verification/smoke_text_calibration.py#main]] performs only bounded local checks using cached assets. It records exact commands, source hashes, device admission, separate collection and evaluation logs and phase exit status. The working source may be uncommitted, but every relevant file must remain unchanged throughout the check; results explicitly prohibit manuscript reuse. It rejects occupied devices, existing output paths and memory filesystems, and passes the shared GPU lock to the evaluator process.

On 2026-09-15, cached BERT SST-2, RoBERTa SST-2 and GPT-2 WikiText checkpoints each completed collection on 16 training examples followed by frozen evaluation on two batches of two examples, with float64, theta 40 and sequence length 32. All 110, 110 and 109 respective sites had positive execution counts with no missing or duplicate site reports. Separate logs, tables, hashes and completion records are preserved under `artifacts/logs/text_calibration_smoke/20260915_policy1_bert`, `20260915_policy1_roberta` and `20260915_policy1_gpt2`. These checks establish execution only, not full task accuracy or adequate calibration sample size.
