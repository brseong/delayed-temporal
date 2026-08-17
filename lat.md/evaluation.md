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

The ViT runner prints per-site Gaussian rates and logs them under `Gaussian/<site>/...` in W&B. Gaussian counters are also process-wide mutable state and are reset whenever a new seeded replica is configured.

## Noise and Ablation Sweeps

Shell scripts under `scripts/experiments` run isolated sweeps for jitter, hazard-style drop/insertion, static mismatch, activation variants, and module-level conversion ablations.

`scripts/experiments/noise_analysis_vit.sh` and `noise_scan_vit.sh` vary one ViT noise component at a time. The same directory contains legacy jitter and theta sweeps, static-bias experiments, quantile collection, and GELU comparisons.

The scripts are orchestration, not definitions of the noise model. Parameter semantics come from [[noise#Noise Model]], and comparisons must distinguish current potential-referred jitter from legacy time-referred runs.

## Gaussian Spike-Time Verification

The maintained Gaussian model has a seeded decorator-level regression check independent of model datasets and checkpoints.

The regression check covers the sampled distribution and deadline behavior plus affine, multiplication, exponential, exponential-difference, division, LayerNorm, softmin, attention value integration, and per-site counters. Operator checks retain noise-off parity paths and force opening, closing/reference, and internal exp-temporal cases where applicable.

The verification intentionally enters through decorated encoders. It does not define or test a separate Gaussian multiplication API.

As each operator is migrated, its regression must force opening and closing/reference misses independently and verify the readout equations in [[noise#Observation-Time Potential Invariant]]. A test expecting an invalid output conflicts with the maintained model.

## Targeted Analysis Programs

The `analysis/` directory contains focused attribution experiments and figure generators for mechanisms that are difficult to isolate in end-to-end task runs.

The analyses cover LayerNorm noise gain, module-level noise contribution, MLP decomposition, GELU internal stages, and beta-linked pooling experiments. They can temporarily scope global encoder noise to selected modules; the restrictions in [[noise#Injection Scope and Compounding]] apply.

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
- Noise changes need deterministic seed checks, distribution checks, injection-scope checks, and repeated confidence intervals.
- Cost-model changes need `python scripts/verification/verify_sop.py` and explicit review of modeling assumptions.

`lat check` validates this documentation’s section identities and source references; it is not a substitute for numerical model tests.
