# Evaluation and Verification

The experiment layer compares pretrained dense and operator-backed models, records finite-domain diagnostics, and runs targeted robustness and accounting analyses.

## Entry Points

Evaluation runners under `scripts/evaluation` are model-family-specific because datasets, preprocessing, task heads, and metrics differ while backend selection follows a common pattern.

- [[scripts/evaluation/error_analysis_vit.py#evaluate_vit_model]] evaluates ViT image classification on CIFAR-10 or ImageNet-style datasets.
- [[scripts/evaluation/error_analysis_bert.py#evaluate_bert_model]] evaluates BERT sequence classification.
- [[scripts/evaluation/error_analysis_roberta.py#evaluate_roberta_model]] evaluates RoBERTa sequence classification.
- [[scripts/evaluation/error_analysis_gpt2.py#evaluate_gpt2_model]] evaluates GPT-2 causal language modeling.

Shell drivers under `scripts/experiments` supply experiment matrices and use `scripts/lib/gpu_pool.sh` to distribute independent runs. They assume locally available checkpoints, datasets, GPUs, and logging credentials as specified by each script.

## Backend Comparison

Every main runner selects either an upstream Hugging Face model or a local spiking model loaded from the same pretrained checkpoint.

The `hf` backend provides the dense reference. The `spiking` backend reconstructs the corresponding local adapter and records LayerNorm stages, attention selection, MLP mode, `theta`, and noise settings.

On GPU, spiking attention is registered through Hugging Face’s attention interface. On CPU or when attention is disabled, the model uses eager dense attention even if other components remain spiking, so the resolved attention implementation is part of the experiment identity.

## Metrics

Task metrics remain conventional so operator conversion can be compared directly with the source model.

ViT, BERT, and RoBERTa report classification accuracy. GPT-2 masks padding labels, averages causal language-model loss over evaluated batches, and reports perplexity.

Quick tests and `max_eval_batches` are smoke-test controls, not final evaluation protocols. Final comparisons should keep dataset split, preprocessing, batch limit, precision, checkpoint, and random seed fixed across backends.

## Diagnostics and Instrumentation

The runners collect internal evidence needed to interpret finite-domain and approximation failures rather than relying only on final task metrics.

Available diagnostics include:

- TensorBoard histograms for LayerNorm inputs and outputs.
- Optional 99.9th-percentile activation magnitude collection for choosing `theta`.
- Named underflow and overflow counts from [[utils/transforms/types.py#set_clamp_log_enabled]].
- Per-site Gaussian event misses and noisy-readout saturation.
- ViT alerts and histograms for centered LayerNorm activations and bounds.
- W&B logging for configuration, intermediate metrics, and final metrics.

Clamp logging uses global module-name state. Hooks must set and clear that state consistently, especially when model execution is parallelized.

Each spiking runner prints per-site Gaussian rates and logs them under `Gaussian/<site>/...` in W&B. Gaussian counters are process-wide mutable state and are reset whenever a new seeded replica is configured.

## Noise and Ablation Sweeps

Shell scripts under `scripts/experiments` run isolated sweeps for Gaussian spike-time noise, static mismatch, activation variants, and module-level conversion ablations.

`scripts/experiments/noise_analysis_vit.sh` and `noise_scan_vit.sh` sweep Gaussian timing scale for ViT. The same directory contains model-family timing and theta sweeps, static-bias experiments, quantile collection, and GELU comparisons.

The scripts pass a dimensionless `time_noise_std_frac`. Each evaluator converts it to one absolute standard deviation using $\sigma_t=r_t(2\theta)$, applies that value at every encoder boundary, and records both values with the seed.

## Gaussian Spike-Time Verification

The maintained Gaussian model requires a seeded decorator-level regression check independent of model datasets and checkpoints.

[[scripts/verification/verify_gaussian_time_noise.py#verify_broadcast_gaussian_time_inputs]] first locks the shared scalar/tensor broadcasting contract, including value alignment plus nominal dtype and device preservation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_time_input_validation]] rejects malformed domains, non-floating or non-finite times, negative scales, and nominal codewords outside the declared interval before sampling.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_sampler_rng_contract]] checks full seeded-stream replay, generator advance across consecutive calls, and exact RNG non-consumption when every standard deviation is zero.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_sampler_deadline_contract]] verifies that early events clamp to the start and fire, deadline equality fires, and only strict exceedance becomes a miss with a finite deadline carrier.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_deadline_probability]] compares the closed-form strict Gaussian tail with seeded empirical misses and checks exact zero-scale probabilities at and beyond the inclusive deadline.

[[scripts/verification/verify_gaussian_time_noise.py#verify_exponential_time_constant_scaling]] checks `tau={0.5,1,2}` across exponential decoding, division, softmin, SwiGLU, and LayerNorm, plus dtype endpoint rejection and RNG preservation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_encoder_boundary]] enters through the decorated identity encoder to check noise-off tuples, zero-noise event parity, forced misses, and exact per-site event counters.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_statistics_contract]] checks strict pre-clamp rail counters, repeated-site accumulation, detached snapshots, disabled instrumentation, and counter clearing without replacing replica RNG state.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_multiplication_operator]] checks deterministic and zero-noise parity, isolated opening and reference misses, observation-time integration, ideal rails, and seeded output saturation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_exponential_function]] checks deterministic and zero-noise values, early-event start clamping, input-miss reset, the zero-extended Gaussian rail, and nonsaturating finite readout statistics.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_exponential_difference_operator]] checks zero-noise parity, opening-reset and closing-deadline readouts, internal-event reset, extended rails, and per-stage statistics.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_division_function]] checks ratio parity and tracks numerator, denominator, and internal exponential misses through shared-domain log encoding to finite output rails.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_softmin_function]] checks dense and zero-noise normalization, numerator-safe shared log bounds, nested event counts, and finite rail-bounded readout when all external events miss.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_swiglu_function]] checks current-bias cancellation on an asymmetric domain, exact zero-noise event topology, and reset-valued finite output when every nested event misses.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_linear]] checks dense affine parity, one shared reference sample, opening-miss bias retention, and deadline integration after a reference miss.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_conv2d]] checks padded dense-convolution parity, one shared reference sample, opening-miss bias retention, and deadline integration over spatial inputs.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_conv1d]] checks GPT-2’s transposed affine layout, arbitrary leading dimensions, shared-reference sampling, bias-only opening misses, and deadline readout.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_layernorm]] checks the dense ablation’s event-free bypass, full-spiking zero-noise topology, and learned-bias output when every nested event misses.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_attention]] checks dense end-to-end attention parity, one shared value reference, opening-miss reset, and reference-miss deadline integration with fixed weights.

The regression check covers the sampled distribution and deadline behavior plus affine, multiplication, exponential, exponential-difference, division, LayerNorm, softmin, attention value integration, and per-site counters. Operator checks retain noise-off parity paths and force opening, closing/reference, and internal exp-temporal cases where applicable.

The verification intentionally enters through decorated encoders. It does not define or test a separate Gaussian multiplication API.

As each operator is migrated, its regression must force opening and closing/reference misses independently and verify the readout equations in [[noise#Observation-Time Potential Invariant]]. A test expecting an invalid output conflicts with the maintained model.

## Targeted Analysis Programs

The `analysis/` directory contains focused attribution experiments and figure generators for mechanisms that are difficult to isolate in end-to-end task runs.

Timing-noise analyses use the same run-wide Gaussian configuration as model evaluation. Focused attribution must be expressed as an explicit experimental program rather than by mutating the global generator around selected modules.

Generated figures and run logs are experiment artifacts rather than architecture sources. Reproducing a figure requires the checkpoint, dataset cache, environment, and command described by the corresponding analysis script.

## Symbolic Operation-Count Check

The paper’s spike-operation and energy formulas have a dedicated symbolic regression checker independent of model execution.

[[scripts/verification/verify_sop.py#main]] recomputes atomic operators, module costs, full ViT formulas, and published rounded values. It encodes fixed-scalar multiplication as free weight calibration through [[scripts/verification/verify_sop.py#free_scale]] instead of counting raw Python calls.

This verifies internal arithmetic consistency under the stated cost model. It does not validate the physical energy constant, system boundary, routing, memory, static power, or circuit feasibility.

## Verification Boundaries

The repository currently emphasizes end-to-end evaluations, inline operator smoke checks, focused analysis scripts, and the symbolic SOP checker rather than a unified automated unit-test suite.

Before treating a change as validated, select checks proportional to its layer:

- Domain or primitive changes need algebraic value and bound tests, including endpoints and signed event order.
- Composite functions need comparisons against their dense mathematical references across calibrated domains.
- Model changes need noise-free checkpoint fidelity plus task-level smoke evaluation.
- Noise changes need deterministic seed checks, distribution checks, injection-coverage checks, and repeated confidence intervals.
- Cost-model changes need `python scripts/verification/verify_sop.py` and explicit review of modeling assumptions.

`lat check` validates this documentation’s section identities and source references; it is not a substitute for numerical model tests.
