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

The `hf` backend provides the dense reference. The `spiking` backend reconstructs the corresponding local adapter and records LayerNorm stages, attention selection, MLP mode, global `theta`, GPT-2's effective `attention_theta`, and noise settings.

On GPU, spiking attention is registered through Hugging Face’s attention interface. On CPU or when attention is disabled, the model uses eager dense attention even if other components remain spiking, so the resolved attention implementation is part of the experiment identity.

## Metrics

Task metrics retain each evaluator's established aggregation so operator conversion can be compared with a source model under the identical runner.

ViT, BERT, and RoBERTa report classification accuracy. GPT-2 masks padding labels and reports $\exp$ of the unweighted mean of per-batch causal-language-model losses. This is batching-consistent but is not token-weighted corpus perplexity; publication tables must either retain identical batching and disclose the aggregation or rerun all GPT-2 conditions with token-weighted NLL.

Quick tests and `max_eval_batches` are smoke-test controls, not final evaluation protocols. Final comparisons should keep dataset split, preprocessing, batch limit, precision, checkpoint, and random seed fixed across backends.

## ViT and GPT-2 Calibration Workflow

The optional ViT/GPT-2 layer-wise calibration workflow selects activation ranges from training data and freezes them before validation. Global theta selection is a separate workflow with its own accuracy checks.

`--calibration-mode collect` selects a fixed-size prefix of a seeded training-split permutation, replays it sequentially for min-max and fixed-bin histogram passes, writes one JSON artifact, and exits without loading validation metrics. Timing noise, mismatch, parameter perturbation, and `DataParallel` are rejected in this mode.

`--calibration-mode validate` and `--calibration-mode inference` reconstruct the same training-subset and model metadata, require an exact artifact match, bind only declared residual, ViT GELU-input, and spiking-attention score ranges, and report strict layer underflow and overflow after the run. GPT-2 metadata records effective `attention_theta` separately from global `theta`, so artifacts cannot cross those numerical contracts. Analytic model-entry ranges bypass calibration.

The maintained defaults select observed min/max (`0/1`) without tail truncation, then add 5% of the selected width per calibrated side. Interior quantiles remain explicit diagnostic overrides. Collection does not optimize endpoints against task accuracy; validation reports the effect of the frozen ranges.

`--calibration-mode none` loads no layer-wise table. It retains fixed configuration limits, intervals derived from weights, and analytic residual sums. This preserves static bounds but omits the measured layer limits intended to control range growth. The current ViT-B noise campaign uses this mode; see [[noise#Timing Noise Scale Sweep at Ratio 4]].

The artifact path is explicit through `--calibration-path`. ViT records image processing and geometry; GPT-2 records filtering of empty texts, tokenizer controls, padded sequence length, and dataset configuration. Both record the seeded training subset, checkpoint, TTFS constants, attention path, and supported ablation settings. The separate ViT cubic implementation and floor are not part of the current artifact identity; see [[calibration#Two-pass Collection#Deterministic Training Subset]].

## Diagnostics and Instrumentation

The runners collect internal evidence needed to interpret finite-domain and approximation failures rather than relying only on final task metrics.

Available diagnostics include:

- TensorBoard histograms for LayerNorm inputs and outputs.
- Optional 99.9th-percentile activation magnitude collection for choosing `theta`.
- Named underflow and overflow counts from [[utils/transforms/types.py#set_clamp_log_enabled]].
- Per-site Gaussian event misses and noisy-readout saturation.
- ViT alerts and histograms for centered LayerNorm activations and bounds.
- W&B logging for configuration, intermediate metrics, and final metrics.

Clamp logging uses global module-name state. Hooks set and restore nested names consistently, and the evaluator rejects named clamp reporting under `DataParallel`; use one process per GPU.

All four runners enable batch-aggregated named clamp reporting only with `--report-clamp-stats`. Nested hooks attribute each clamp to its encoder, attention, LayerNorm, affine, or convolution module, restore the outer name after each call, and print one run-wide count and rate per site.

The text-model runners accept `--cache-dir` so a documented local dataset cache can be selected independently of the checkpoint cache. TensorBoard logging is optional: when the package is absent, a no-op writer preserves evaluation behavior instead of preventing the run.

Each spiking runner prints per-site Gaussian rates and logs them under `Gaussian/<site>/...` in W&B. Gaussian counters are process-wide mutable state and are reset whenever a new seeded replica is configured.

## Fixed-Domain ViT-S Real-Data Audit

The fixed-domain audit measures one cached pretrained ViT-S checkpoint on the same 5,000-image ImageNet-1k validation subset and separates analytic rails, residual calibration, and Gaussian event effects.

The checkpoint is `/data/nas/vit_small_patch16_224.augreg_in21k_ft_in1k`; all runs use float32, batch size 32, $\theta=2000$, and all three spiking LayerNorm stages plus spiking attention and MLP. The dense Hugging Face reference scores 80.26%, while calibration-free spiking scores 80.54%.

| Condition | Accuracy | Interpretation |
|---|---:|---|
| Hugging Face dense | 80.26% | Same checkpoint, preprocessing, and validation subset |
| Analytic fixed rails | 80.54% | The $\sqrt d$ mixed LayerNorm rail removes the prior float32 timestamp-cancellation failure |
| Min/max + 5% calibration | 80.36% | 1,024 training images, 48 necessary sites, full 5,000-image validation |
| Retired tail-trim calibration | 59.26% | 1,024 training images, 2,048 bins, 0.001/0.999 quantiles, 5% margin |
| Retired calibration + Gaussian | 59.26% | $r_t=3.162\times10^{-10}$, $\sigma_t=1.2648\times10^{-6}$, seed 0 |

The analytic run reports zero excursions for the input embedding convolution, affine input rails, attention scores, attention value outputs, LayerNorm variance, and the new $\sqrt d$ normalized LayerNorm rail across all 5,000 images. Inactive LayerNorm dual rails clamped to `clip_margin` and the product primitive's structural reset rail are bookkeeping, not failures of those ideal output rails. The conventional classifier has no TTFS rail and is assessed by task accuracy.

The retired tail-trim artifact is the observed accuracy bottleneck. Its largest single rate is layer-10 attention-score overflow at 0.158845%; layer-0 output underflow is 0.114174%, and encoder-input underflow/overflow are 0.0577443%/0.0628810%. These individually small clamps compound to a 21.28-point loss relative to the analytic spiking run, so this artifact is diagnostic and must not be treated as the maintained accuracy baseline.

The replacement artifact uses only the 48 necessary residual, composed-GELU-input, and spiking-attention-score sites, selects observed min/max, and adds 5% per side. Full 5,000-image validation scores 80.36%, recovering 21.10 percentage points from tail trimming and remaining within 0.18 points of calibration-free spiking.

Only 408 of 41,204,520,000 calibrated values exceed their frozen rails, an aggregate rate of $9.90183\times10^{-9}$. The largest site is layer-1 attention-score overflow at 377 of 1,164,270,000 values ($3.23808\times10^{-7}$); the next is layer-0 attention-residual overflow at 23 of 378,240,000 values ($6.0808\times10^{-8}$).

At the precision-limited Gaussian setting, division-numerator misses are 8.67352%, multiplication-output underflow saturation is 0.901420%, LayerNorm positive/negative log misses are 0.583904%/0.600765%, and convolution data-event misses are 0.453567%. Division-output overflow is $9.46541\times10^{-6}$; affine output, attention value, exponential output, and normalized LayerNorm saturation are zero.

The identical retired-calibration clean and Gaussian accuracies do not establish continuous-noise robustness because this $\sigma_t$ is below float32 spacing at relevant deadlines. They establish that physical miss and saturation counters remain observable even when top-1 predictions do not change. The replacement full-run log is `artifacts/logs/fixed_domain_validation/vit_small_minmax_margin5_clean_5000.log`, and its frozen table is `artifacts/calibration/vit_small_fixed_domain_minmax_margin5.json`.

## Fixed-Domain Text-Model Real-Data Audit

The text-model audit compares cached pretrained checkpoints on complete held-out splits and distinguishes representative wrapper settings from a deliberately narrow-domain diagnostic.

All runs use float32, maximum length 128, no timing noise, and all three temporal LayerNorm stages when LayerNorm is enabled. BERT and RoBERTa use all 872 GLUE/SST-2 validation examples with batch size 32. GPT-2 uses all 181 nonempty WikiText-2 test batches with batch size 16. The representative wrapper thresholds are 1,000 for BERT, 2,000 for RoBERTa, and global 2,000 plus attention-local 100 for GPT-2.

| Model and checkpoint | Dense reference | Full spiking | Difference |
|---|---:|---:|---:|
| BERT, `textattack/bert-base-uncased-SST-2` | 92.43% | 92.20% | -0.23 percentage points |
| RoBERTa, `Bhumika/roberta-base-finetuned-sst2` | 94.50% | 94.04% | -0.46 percentage points |
| GPT-2, `neulab/gpt2-finetuned-wikitext103` | loss 3.1227, PPL 22.7076 | loss 3.1311, PPL 22.8991 | +0.1915 PPL |

BERT has no attention-score excursion and only 383 actual negative LayerNorm-magnitude overflows among 2,143,027,200 values, a $1.78719\times10^{-7}$ rate. RoBERTa records 65,129 score excursions among 2,057,306,112 values, a $3.16574\times10^{-5}$ rate, with no actual magnitude excursion. Roughly half of each LayerNorm log carrier is floored because only one signed rail is active per centered value; these carrier-floor counts and the positive division-numerator floor are structural bookkeeping rather than output-domain failures.

The initial single-threshold GPT-2 run at $\theta=2000$ records 47,764,010 attention-score excursions among 6,820,724,736 values, a 0.700278% pre-mask diagnostic rate. Score clamping is counted before the causal overwrite, so this population includes future positions and is only an upper bound on effective unmasked clipping. Its only actual LayerNorm magnitude excursion is 49,147 positive-rail overflows among 7,104,921,600 values, a $6.91732\times10^{-6}$ rate. The score rail is limited by float32 softmin representability, so observed-extrema calibration cannot widen it past that analytic ceiling.

### GPT-2 Path Attribution

The representative GPT-2 ablation uses the local all-dense-stage wrapper as its attribution baseline, separating wrapper fidelity from temporal-operator effects.

| Enabled temporal path at $\theta=2000$ | Loss | PPL | PPL change from local wrapper |
|---|---:|---:|---:|
| Local wrapper, no temporal path | 3.1227 | 22.7082 | 0 |
| LayerNorm only | 3.1307 | 22.8910 | +0.1828 |
| Attention only | 3.2008 | 24.5520 | +1.8438 |
| LayerNorm + MLP affine | 3.1307 | 22.8908 | +0.1826 |
| Attention + MLP affine | 3.2011 | 24.5584 | +1.8502 |
| LayerNorm + attention + MLP affine | 3.2202 | 25.0324 | +2.3242 |
| LayerNorm + attention + MLP affine, attention-local $\theta=100$ | 3.1311 | 22.8991 | +0.1909 |

With one global threshold, attention is the dominant isolated contribution, LayerNorm is smaller, and the spiking GPT-2 MLP affine path is negligible in both pairwise controls. GPT-2 keeps `gelu_new` as a dense activation inside the fixed-range MLP, so this audit does not claim a temporal GELU contribution. Narrowing only attention's code window to 100 recovers 2.1333 PPL while preserving LayerNorm's required 2,000-wide rail; the mixed result is within 0.0081 PPL of the LayerNorm-only control.

At a global $\theta=100$, any mixed LayerNorm path collapses to roughly PPL 24,000--25,000 because centered magnitudes exceed the narrow LayerNorm rail. A 1,024-text min/max-plus-5% residual and score artifact leaves residual excursions at zero but cannot repair that threshold error. The attention-local override avoids this conflict; calibration remains restricted to declared residual and score sites and does not replace operator threshold selection.

The classifier results and mixed-threshold GPT-2 result support low degradation for these three selected checkpoints. The fresh simultaneous GPT-2 baseline and conversion runs differ by 0.843% PPL. The improvement comes from an explicit operator-local numerical contract rather than quantile tail trimming: the wider global rail protects LayerNorm range, while the narrower attention window improves float32 temporal subtraction precision.

### GPT-2 Floating-Point Precision Control

The appendix control separates float32 timestamp resolution from execution-range clipping in the fully converted GPT-2 path.

[[scripts/evaluation/error_analysis_gpt2.py#evaluate_gpt2_model]] accepts an explicit float32 or float64 model dtype when calibration is disabled. Active calibration remains float32-only because artifact metadata currently locks that numerical contract.

The reproducible `scripts/experiments/precision_analysis_gpt2.sh` protocol evaluates the complete WikiText-2 test split at batch size 16 and length 128. Its float32 attention thresholds 50, 100, 200, 500, 1,000, and 2,000 all retain the same softmin execution score radius, 40.242257, while timestamp ULP grows from $3.8147\times10^{-6}$ to $1.2207\times10^{-4}$.

| Attention $\theta$ | dtype | Loss | PPL | Score excursion rate |
|---:|---:|---:|---:|---:|
| 50 | float32 | 3.1308 | 22.8913 | 0.640466% |
| 100 | float32 | 3.1311 | 22.8991 | 0.641252% |
| 200 | float32 | 3.1316 | 22.9102 | 0.637540% |
| 500 | float32 | 3.1375 | 23.0466 | 0.645425% |
| 1,000 | float32 | 3.1535 | 23.4172 | 0.655191% |
| 2,000 | float32 | 3.2202 | 25.0324 | 0.700278% |
| 2,000 | float64 | 3.1308 | 22.8928 | 0% |

All float32 points record zero query, key, value, normalized-weight, division-result, and attention-output excursions. PPL worsens even from $\theta=100$ to 200 while the same pre-mask score diagnostic decreases, so its count does not explain the trend. Reducing the attention window to 50 recovers 92.1% of the dense-reference PPL gap and 91.7% of excess NLL.

The float64 endpoint confirms recovery at the original time window but also widens the exponent-representable score rail to 350.772. It is therefore a corroborating numerical reference rather than a pure dtype intervention. [[scripts/analysis/summarize_gpt2_precision.py#parse_run]] rejects incomplete logs, mismatched dtypes or thresholds, loss/PPL inconsistencies, missing clamp counts, and any non-score attention payload excursion.

### Text-Model LayerNorm Execution Path

The audited BERT, RoBERTa, and GPT-2 configurations use the same explicit LayerNorm stage topology.

| Stage | Audited implementation |
|---|---|
| Centering | Tensor feature mean subtraction |
| Variance square | Temporal multiplication, `spiking_ln_mul=True` |
| Negative logarithm | Temporal log encoding, `spiking_ln_log=True` |
| Normalized dual-rail readout | Temporal exponential difference, `spiking_ln_expdiff=True` |
| Learned affine | Temporal product when exponential difference is active |

## ViT-B/16 Global Theta Selection

The deterministic selection workflow chooses the smallest accurate global threshold before any new ViT-B/16 robustness result can be treated as manuscript evidence.

[[scripts/evaluation/error_analysis_vit.py#load_evaluation_dataset]] accepts a self-contained Hugging Face dataset artifact through `--evaluation-dataset-path`. It preserves saved order and fingerprints, accumulates top-1 from local correct/total counts, writes a prediction SHA-256 digest, and can disable TensorBoard completely with `--no-tensorboard`.

The training candidates are $\theta\in\{10,20,40,80,160,320,640,1000,1400,2000,2800,4000\}$. If 10 remains within 0.005 of the best accuracy, the lower boundary is unresolved and must be halved again until the first lower candidate fails; unresolved boundaries cannot be confirmed. Every run uses the seed-0 shuffled ImageNet training subset of exactly 5,000 images, float64, batch size 32, analytic ranges, all maintained ViT spiking paths, and zero timing noise, mismatch, deadline margin, weight noise, and bias noise.

[[scripts/analysis/summarize_theta_selection.py#choose_theta]] selects the smallest candidate whose accuracy is within 0.005 of the best spiking candidate, not the dense ANN reference. A gain above 0.001 from 2,800 to 4,000 requires 5,600 and 8,000; a further gain above 0.001 from 5,600 to 8,000 marks the search range insufficient rather than approving an endpoint.

The completed selection used source `bc973317` with layer-wise calibration disabled. The corrected GELU campaign at source `648af9bb` reuses the selected theta of 40; it did not repeat the candidate search. Its clean validation result confirms accuracy at 40, not that 40 remains the smallest acceptable candidate under the new implementation. Neither this global search nor its replay selects the positive log lower endpoint. See [[noise#Timing Noise Scale Sweep at Ratio 4]] for the actual fixed input bounds.

Threshold selection is based on accuracy rather than a requirement of zero clipping. At $\theta=40$, clipping at the finite potential domain is an intended part of the selected bounded implementation; its frequency is reported as a diagnostic and does not by itself invalidate selection.

The selected training condition must replay with the identical correct count and prediction digest. The selected candidate and its immediate available neighbors are then evaluated on the first 5,000 examples of the self-contained validation artifact; selection fails if the chosen value is more than 0.005 below that local validation maximum. Full-validation approval additionally requires selected-spiking and dense-reference runs on all 50,000 validation images. `THETA_SKIP_FULL_VALIDATION=1` explicitly stops after successful 5k confirmation, records `full_validation=skipped`, and retains `confirmed` rather than `approved` status.

Clamp reports retain the maximum pre-clamp rail-saturation rate and its site as diagnostics. Inactive dual rails named `x_err_neg` or `x_err_pos` and multiplication's `multiplication_result` reset rail are counted separately from the other clamp diagnostics and cannot change the accuracy-based choice.

UBAI execution benchmarks RTX 3090, A10, RTX 6000 Ada, and RTX A6000 twice with five warm-up and twenty measured batches. [[scripts/analysis/select_ubai_gpu_family.py#choose_family]] rejects OOM, incomplete timing, wrong hardware identity, replica disagreement, and predictions differing from the RTX A6000 reference. It chooses the lowest median seconds per image; families within 5% use current free-GPU capacity as the tie-break.

The UBAI task contract uses one GPU, four CPUs, 64 GB RAM, no `DataParallel`, and immutable condition/data/checkpoint/commit/GPU-family manifests. The portable environment is stored once as a compressed archive and expanded into job-local temporary storage, avoiding a second persistent copy under the 100 GB home quota. [[scripts/experiments/ubai/build_theta_selection_manifest.py#main]] creates one row per array index, and completed logs are resumed only after their recorded identity matches that row. A lightweight dependency controller uses an explicit Python 3.11-or-newer bootstrap interpreter to submit selection, conditional upper-grid extension, confirmation, and optional full-validation stages only after the preceding reducer succeeds; model evaluation still uses the archived Python 3.12 environment. Reducer submission preserves the scheduler-visible absolute path spelling, including `/home1` aliases, and splits multi-manifest inputs before container-path translation.

[[scripts/verification/verify_theta_selection.py#main]] covers the 0.5-point boundary, tie behavior, upper-grid guard, replay and validation gates, offline dataset order and fingerprint replay, TensorBoard suppression, GPU-family selection, and one-GPU manifest contract.

## Noise and Ablation Sweeps

Shell scripts under `scripts/experiments` retain isolated sweep and ablation tooling, but their presence does not make every matrix active work.

Descriptions below record available or historical tooling, not an experiment queue. Active compute is limited by [[todo#Active Experiment Work]], and optional reruns are centralized in [[deferred-experiments]].

`scripts/experiments/noise_analysis_vit.sh` and `scripts/experiments/noise_scan_vit.sh` sweep Gaussian timing scale for ViT. The maintained fine scan targets ViT-B/16, preserves completed outputs, records its expected-run manifest, and resumes only incomplete tagged logs.

The runner accepts only physical GPUs 4 through 7, refuses externally occupied devices by default, and schedules at most one evaluator process per selected GPU, thereby avoiding `DataParallel` and cross-job contention. Its canonical float64 payload resolves the selected tiny timing scales; batch size remains 32 because each affine reference event is shared per layer call. The quick protocol uses the fixed first 5,000 validation images, while the full protocol contains three confirmation points per noise axis.

`REPLICA_SEEDS` defaults to 0, 1, and 2 for both Gaussian timing and frozen threshold mismatch. `TIME_NOISE_STD_FRACS` and `MISMATCH_THETA_STDS` may replace their grids, and an explicitly empty mismatch list supports Gaussian-only theta analysis without fabricating a second axis.

The scripts pass a dimensionless `time_noise_std_frac`. Each evaluator converts it to one absolute standard deviation using $\sigma_t=r_t(2\theta)$, applies that value at every encoder boundary, and records both values with the seed.

Sweep interpretation is also conditioned on [[noise#Numerical Precision and Endpoint Caveat]]. Logs and CSVs preserve checkpoint, split, sample count, theta, dtype, absolute timing scale, identity-deadline ULP, per-site ULP range, endpoint occupancy, misses, and saturation counts.

Both stochastic axes use independent dedicated generators while holding the model, validation subset, and loader seed fixed. Gaussian timing advances one stream across event encoders; static mismatch samples one frozen module-offset set per replica.

[[scripts/analysis/summarize_noise_scan.py#summarize_noise_scan]] rejects missing, failed, identity-mixed, or parameter-inconsistent logs before publishing raw and aggregate CSV files. Both noise axes use 95% Student-t intervals; Gaussian event, endpoint, and saturation rates pool raw denominators across sites and replicas.

`scripts/experiments/theta_jitter_analysis_vit.sh` can run Gaussian-only scans for $\theta\in\{40,400,2000\}$ using transition grids scaled by $2000/\theta$. The multi-theta scan is not scheduled; [[scripts/analysis/summarize_theta_noise_scan.py#summarize_theta_noise]] remains available to validate historical or explicitly promoted results.

`scripts/experiments/run_noise_campaign_vit.sh` is the legacy publication-campaign supervisor. It waits until at least one of GPUs 4--7 is idle for two consecutive 60-second samples, fixes that idle subset for one stage, and runs smoke, quick, full, and theta stages in order. A lock rejects duplicate supervisors; every stage remains resumable through its child manifest.

The supervisor keeps generated PDFs under `artifacts/` until all summaries, numerical-resolution checks, seeded verifications, and `lat check` pass. Only then does it install the main and appendix PDFs under `paper/neurips_2026/figures/` and rebuild the manuscript.

`scripts/experiments/deadline_margin_sweep_vit.sh` preserves the earlier adaptive margin diagnostic at $r_t=10^{-10}$. It is not part of the active fixed-ratio scale sweep.

[[scripts/verification/verify_theta_noise_summary.py#verify_theta_noise_summary]] checks the three-theta identity contract plus combined CSV and PDF/PNG rendering with dataset-independent fixtures.

[[scripts/verification/verify_noise_scan_summary.py#verify_noise_scan_summary]] validates seeded manifest constraints, evaluator-log parsing, pooled physical counts, both Student-t intervals, artifact rendering, and rejection of Gaussian logs without mechanism statistics. [[scripts/verification/verify_noise_scan_runner.py#verify_noise_scan_runner]] verifies the GPU allowlist and two-axis dry-run manifest without launching a dataset job.

### Per-Layer ViT GELU Attribution

The ViT GELU layer scan isolates where temporal activation errors become task-critical without changing the activation's mathematical formula.

[[scripts/evaluation/error_analysis_vit.py#configure_vit_exact_gelu_layers]] selects zero-based encoder blocks whose MLP GELU uses the maintained cubic-tanh formula in dense arithmetic. Both affine layers remain unchanged, and every unselected block retains the temporal composite.

`scripts/experiments/ablation_gelu_layers_vit.sh` selects exactly one block per condition and compares its noisy accuracy with the corresponding noise-off accuracy. It schedules one process per GPU, resumes complete logs, and permits seed and layer subsets through environment variables.

Recovery relative to the fully temporal noisy run estimates that block's timing-error contribution; it is not an architecture or activation-function comparison. The default seed-zero scan ranks all blocks before additional seeds are assigned to the most influential conditions.

The original float32 layer scan at $r_t=3.162\times10^{-10}$ is precision-limited: its absolute $\sigma_t=1.2648\times10^{-6}$ is below float32 spacing near the GELU log-division deadline. Its block ranking is exploratory and cannot support a mechanism claim; a new layer scan is deferred unless that claim is restored.

`scripts/experiments/diagnose_gaussian_endpoint_vit.sh` reruns the same 5,000-image condition in float64 using baseline, full Gaussian, block-10 GELU bypass, all-GELU bypass, and all-GELU-plus-LayerNorm-log bypass. These controls determine whether a layer ranking remains identifiable after continuous endpoint behavior is numerically resolved.

The float64 diagnostic places full Gaussian, block-10 bypass, and all-GELU bypass at classification floor, while bypassing both temporal GELU and LayerNorm log restores baseline accuracy. The prior block ranking is therefore not identifiable under resolved continuous endpoint sampling, so no further layer sweep is scheduled.

[[scripts/verification/verify_vit_gelu_layer_ablation.py#verify_vit_gelu_layer_ablation]] checks sparse selection, empty-selection behavior, invalid indices, duplicate rejection, and all-or-nothing failure when the expected local ViT topology is absent.

### GELU Cubic Construction Comparison

The deterministic ViT comparison changes only construction of the cubic term in the maintained tanh-based GELU approximation.

[[scripts/analysis/gelu_cubic_phi_nl_vit.py#phi_nl_psi_ed_cube]] splits the signed input into positive and negative magnitudes, applies $\phi_{\mathrm{NL}}$ with $3\tau_s$, and evaluates each encoded time against one domain upper endpoint with $\psi_{\mathrm{ED}}$ at $\tau_s$. The resulting normalized cubes receive the fixed magnitude gain before signed recombination.

The `multiplication` condition retains the production $x^2$ and $x^3$ chain. The `phi_nl_psi_ed` condition replaces only that chain; coefficient scaling, membrane superposition, the tanh gate, final multiplication by the input, propagated domains, checkpoint, dataset order, and all other model paths remain fixed.

The deterministic construction comparison keeps direct Gaussian timing error disabled so its accuracy result isolates the cubic implementation. The alternative construction also supports robustness experiments: both signed magnitude encoders, their shared domain endpoint reference, and the internal encoding inside $\psi_{\mathrm{ED}}$ receive the replica Gaussian timing draws.

The alternative limits each signed magnitude to `theta` before log encoding, matching the bounded log domain used by LayerNorm and accepting wider analytic upstream bounds. [[scripts/verification/verify_gelu_cubic_phi_nl.py#verify_phi_nl_psi_ed_cube]] checks signed cubic values, invariance across positive time constants, finite domain floor behavior, threshold limiting, float32 GELU agreement, propagated bounds, parity without noise, seeded replay, seed independence, and patch isolation in the local ViT adapter.

#### Observed ViT-S Result

On the fixed first 5,000 ImageNet-1k validation images, the alternative cubic construction produced six more correct predictions and did not reduce top-1 accuracy.

| Cubic implementation | Correct | Accuracy | Difference from multiplication |
|---|---:|---:|---:|
| `multiplication` | 4,020 / 5,000 | 80.40% | 0 |
| `phi_nl_psi_ed` | 4,026 / 5,000 | 80.52% | +0.12 percentage points |

Both runs used the same checkpoint, float32 precision, batch size 32, $\theta=2{,}000$, calibration from the observed minimum and maximum with a 5% margin on each side, validation order, and disabled Gaussian spike-time error, mismatch, weight noise, and bias noise. The dataset fingerprint was `260dc8e69ecaea24`; prediction digests differed, while the complete calibration underflow and overflow reports for every site were identical.

This single deterministic 5,000-image comparison establishes that the alternative does not cause an observed accuracy drop under this condition. It is not evidence of a general accuracy improvement. Full validation and additional checkpoints belong to [[deferred-experiments#Scale and Generality]] and are needed only if that broader claim is retained.

### GELU-Internal Operator Attribution

The GELU operator scan attributes task-level timing sensitivity among multiplication, exponential, and division without changing the production deadline or margin contract.

[[scripts/analysis/gelu_operator_ablation_vit.py#gelu_operator_ablation]] reproduces the maintained cubic-tanh composition while allowing selected GELU-local atomic operators to use their nominal, noise-free temporal carriers. Every unselected GELU operator and every non-GELU use of the same primitive remains on the run-wide Gaussian path.

The `multiplication` unit covers all six products in one GELU call, including fixed polynomial scaling and the final input-gate product. The prior half-gate product is absent because the affine tanh map cancels it exactly. The `exponential` unit is the $\exp(-2z)$ stage. The `division` unit includes both negative-log operand encoders and their internal exponential-difference stage because those events jointly implement one ratio.

The eight-condition matrix contains the fully noisy composition, three leave-one-operator-dense conditions, three only-one-operator-noisy conditions, and the all-dense control. Comparing both directions distinguishes an operator whose removal is sufficient for recovery from one whose isolated noise is sufficient for failure.

The dense helpers retain the noise-off temporal arithmetic order, including float32 carrier rounding, and preserve Gaussian-compatible downstream rails. Direct mathematical products or ratios are not used because they would also remove nominal time-code quantization and confound attribution.

The dense helpers apply the production analytic endpoint clamp after the cubic inner sum, while constrained division supplies the fixed $[0,1]$ gate interval. A selected operator therefore changes event delivery without changing fixed-domain containment.

Selected operators shadow-consume the same Gaussian draws in the same tensor/scalar order but do not apply or count those events. This common-random-number coupling keeps every later GELU and non-GELU event aligned across variants, reducing paired seed variance without representing shadow draws as physical activity.

`scripts/experiments/ablation_gelu_operators_vit.sh` runs one condition per process and GPU, holds model, 5,000-image subset, absolute timing scale, and seed fixed, and resumes only complete logs. [[scripts/analysis/gelu_operator_ablation_vit.py#install_gelu_operator_ablation]] patches only the local ViT GELU symbol, leaving production implementations and other model families unchanged.

This scan deliberately leaves endpoint placement and calibration unchanged. At the existing float32 transition point it is an implementation-level attribution conditioned on [[noise#Numerical Precision and Endpoint Caveat]], not a calibrated continuous-noise robustness result. Repeating the matrix belongs to [[deferred-experiments#Mechanism and Operator Ablations]] and is needed only for a retained mechanism claim.

[[scripts/verification/verify_gelu_operator_ablation.py#verify_gelu_operator_ablation]] checks all eight noise-off subsets for value parity, rejects unknown operator labels, and verifies that installation changes only the local ViT adapter symbol. [[scripts/verification/verify_gelu_operator_ablation.py#verify_gelu_operator_event_selection]] checks physical event topology and equal post-GELU generator state across all dense selections.

#### Observed ViT-S Result

At the existing float32 transition point, GELU division accounts for essentially the entire task-level loss attributed to the temporal GELU composition.

The run uses ViT-S, the first 5,000 ImageNet-1k validation images, `theta=2000`, batch size 32, $r_t=3.162\times10^{-10}$, and therefore $\sigma_t=1.2648\times10^{-6}$. Seed zero evaluates all eight operator combinations; the full, dense-division, only-division-noisy, and all-dense controls are repeated with timing seeds 1 and 2.

Across the three seeds, mean accuracy is 56.353% for fully noisy GELU, 56.360% when division alone remains noisy, 78.773% when division alone is dense, and 78.753% when all GELU-local operators are dense. Division removal recovers 22.420 percentage points, while division-only noise reproduces a 22.393-point loss.

In the complete seed-zero matrix, removing multiplication or exponential alone changes no classifications relative to fully noisy GELU. Leaving only multiplication noisy matches the all-dense accuracy, while leaving only exponential noisy differs from the dense-division control by no classifications. The remaining three-seed dense-division versus all-dense mean difference is only 0.020 percentage points.

The GELU-local division numerator contributes 18,155,520,000 events per 5,000-image run. It misses 23,254,914, 23,250,225, and 23,249,362 times across seeds 0, 1, and 2, respectively, for a mean miss rate of 0.12807%. A numerator opening miss resets the ratio to zero and directly closes the GELU gate, explaining why sparse misses erase activations and compound through the model.

These measurements predate the direct division gate and its six multiplication calls, so they remain historical and are not scheduled for repetition.

These measurements identify the division numerator deadline boundary—not continuous multiplication or exponential perturbation—as the dominant implementation-level GELU failure in this configuration. Because $\sigma_t$ is sub-ULP near relevant float32 deadlines, this statement remains historical rather than a current mechanism claim.

## Gaussian Spike-Time Verification

The maintained Gaussian model requires a seeded decorator-level regression check independent of model datasets and checkpoints.

### Closed-Domain Verification

Central domain construction and tensor membership checks must fail consistently before malformed rails enter any operator.

[[scripts/verification/verify_gaussian_time_noise.py#verify_closed_bounds_validation]] accepts inclusive singleton rails, rejects non-real, non-finite, and reversed endpoints for every bounds type, and confirms that `check_domain` raises explicit exceptions under optimized Python.

[[scripts/verification/verify_gaussian_time_noise.py#verify_immutable_memoized_bounds]] rejects mutation of potential and time endpoints and checks that equal attention configurations reuse one bounds object while distinct configurations remain separate.

[[scripts/verification/verify_gaussian_time_noise.py#verify_broadcast_gaussian_time_inputs]] first locks the shared scalar/tensor broadcasting contract, including value alignment plus nominal dtype and device preservation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_time_input_validation]] rejects wrong domain types, non-floating or non-finite times, negative scales, and nominal codewords outside the declared interval before sampling; malformed endpoint declarations are rejected earlier by the common bounds constructor.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_sampler_rng_contract]] checks full seeded-stream replay, generator advance across consecutive calls, and exact RNG non-consumption when every standard deviation is zero.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_sampler_deadline_contract]] verifies that early events clamp to the start and fire, deadline equality fires, and only strict exceedance becomes a miss with a finite deadline carrier.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_deadline_probability]] compares the closed-form strict Gaussian tail with seeded empirical misses and checks exact zero-scale probabilities at and beyond the inclusive deadline.

[[scripts/verification/verify_gaussian_time_noise.py#verify_exponential_time_constant_scaling]] checks `tau={0.5,1,2}` across log encoding, exponential decoding, division, softmin, SwiGLU, and LayerNorm, plus domain rejection and RNG preservation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_encoder_boundary]] enters through the decorated identity encoder to check noise-off tuples, zero-noise event parity, forced misses, and exact per-site event counters.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_statistics_contract]] checks strict pre-clamp rail counters, repeated-site accumulation, detached snapshots, disabled instrumentation, and counter clearing without replacing replica RNG state.

[[scripts/verification/verify_gaussian_time_noise.py#verify_static_mismatch_rng_contract]] checks dedicated-seed replay, seed independence, global-RNG preservation, frozen offsets across forwards, non-persistent buffers, and invalid-seed rejection.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_multiplication_operator]] checks deterministic and zero-noise parity, isolated opening and reference misses, observation-time integration, ideal rails, and seeded output saturation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_exponential_function]] checks deterministic and zero-noise values, early-event start clamping, input-miss reset, the zero-extended Gaussian rail, and nonsaturating finite readout statistics.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_exponential_difference_operator]] checks zero-noise parity, opening-reset and closing-deadline readouts, internal-event reset, extended rails, and per-stage statistics.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_tanh_function]] checks deterministic and zero-noise tanh parity on the common $[-1,1]$ domain, nested event topology, forced structural saturation, and finite final clamping.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_sigmoid_gelu_function]] checks the sigmoid approximation against $x\,\sigma(1.702x)$, reconstructs its output domain from the fixed $[0,1]$ gate, and forces gate saturation without widening the final product rails.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_softmin_function]] checks dense and zero-noise normalization on the common structural $[0,1]$ domain, numerator-safe shared log bounds, nested event counts, final saturation denominators, and finite rail-bounded readout when all external events miss. Forced excursion accounting may occur at division or at the final softmin clamp without changing the public contract.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_swiglu_function]] checks current-bias cancellation on an asymmetric domain, output rails reconstructed from a fixed $[0,1]$ gate, exact zero-noise gate counters, forced gate saturation, and reset-valued finite output when every nested event misses.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_division_function]] checks the common $[0,1]$ division domain, exact deterministic and zero-noise ratios, output saturation counters for both one-sided misses, internal reset zero, and preservation of unrestricted exponential difference for dual-rail LayerNorm.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_linear]] checks dense affine parity, one shared reference sample, symmetric one-sided signed-PWM readout, output-row absolute-sum rails, and post-freeze parameter/threshold mutation rejection.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_conv2d]] checks padded dense-convolution parity, one shared reference sample, symmetric one-sided signed-PWM readout, output-channel absolute-sum rails, and post-freeze mutation rejection.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_conv1d]] checks GPT-2’s transposed affine layout, arbitrary leading dimensions, shared-reference sampling, symmetric one-sided signed-PWM readout, output-column absolute-sum rails, and post-freeze mutation rejection.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_layernorm]] checks the dense ablation’s event-free bypass, full-spiking zero-noise topology, learned-bias output when every nested event misses, independent analytic domains for all eight ablation topologies, immutable cache reuse across noise modes, and parameter/configuration mutation rejection with explicit refresh.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_attention]] checks dense end-to-end attention parity, an in-domain hard mask below the global cap, a request-independent maximum-source output rail, one shared value reference, and symmetric one-sided signed-PWM integration with fixed weights.

The regression check covers the sampled distribution and deadline behavior plus affine, multiplication, exponential, exponential-difference, division, LayerNorm, softmin, attention value integration, and per-site counters. Operator checks retain noise-off parity paths and force opening, closing/reference, and internal exp-temporal cases where applicable.

The verification intentionally enters through decorated encoders. It does not define or test a separate Gaussian multiplication API.

Regression for every migrated operator must force opening and closing/reference misses independently and verify the readout equations in [[noise#Observation-Time Potential Invariant]]. A test expecting an invalid output conflicts with the maintained model.

## Targeted Analysis Programs

The `analysis/` directory contains focused attribution experiments and figure generators for mechanisms that are difficult to isolate in end-to-end task runs.

Timing-noise analyses use the same run-wide Gaussian configuration as model evaluation. Focused attribution must be expressed as an explicit experimental program rather than by mutating the global generator around selected modules.

Generated figures and run logs are experiment artifacts rather than architecture sources. Reproducing a figure requires the checkpoint, dataset cache, environment, and command described by the corresponding analysis script.

## Hidden Activation Bounds Inspection

The CPU diagnostic captures declared hidden activation bounds from the frozen evaluation source and checkpoint, without collecting activation extrema or changing calibration.

[[scripts/analysis/inspect_vit_hidden_bounds.py#main]] uses the same spiking attention and GELU configuration as the selected experiment. Two synthetic clean inputs and one input with Gaussian timing noise must produce identical module bounds. The CSV records exact endpoints; these are not measured dataset activation ranges. Source, evaluator, and checkpoint identity checks prevent inspecting a different implementation by accident. Both residual additions are checked against their input bounds in every layer.

## Calibrated ViT Noise Comparison

A separate experiment repeats the existing timing noise and deadline margin sweeps with frozen layer-wise calibration at threshold 40; previous uncalibrated results remain unchanged.

[[scripts/experiments/run_calibrated_noise_vit.py#execute]] uses [[scripts/analysis/evaluate_calibrated_vit.py#main]] to consume the same seed-0 training 5k artifact in both collection passes without reshuffling. Observed minimum and maximum values receive an additional 5% of interval width on each calibrated side. All 48 configured sites are required; validation 5k never selects ranges. The 65-condition union reuses only the dense reference and keeps three seeds per stochastic condition. This is a controlled comparison at threshold 40, not a repeated threshold selection.

Source, checkpoint, training artifact, GELU implementation, floor, wrapper, and frozen table identities are checked before reuse. A complete clean evaluation precedes the stochastic sweep; clean accuracy at or below 1% stops execution for inspection. Partial outputs show all completed replicas but compute confidence intervals only for complete three-seed cells. GPU capacity waits never terminate the campaign merely because a device is occupied. GPU devices 4–7 and persistent runtime storage remain mandatory.

## Calibrated Three Sweep Campaign

The current ViT-B/16 campaign selects a threshold using training 5k and compares nine validation points on each of the threshold, timing noise, and deadline margin axes with output bounds policy 3.

[[scripts/experiments/calibrated_three_sweeps.py#make_tasks]] fixes nine thresholds from 10 to 160 at equal logarithmic intervals. Each threshold receives a new 48-site calibration table from the seed-0 training 5k artifact: observed minimum and maximum, 5% additional interval width, and two collection passes. The same training 5k determines the smallest candidate within 25 correct predictions of the best candidate. Validation 5k is never used for calibration or selection.

[[scripts/experiments/calibrated_three_sweeps.py#select_theta]] refuses selection at the smallest candidate or a gain greater than five correct predictions between the two largest candidates. [[scripts/experiments/calibrated_three_sweeps.py#confirm_selection]] requires exact training correct count and prediction digest on the opposite execution environment, plus validation accuracy within 25 correct predictions of the best immediate neighbor. Passing evidence is `confirmed` on 5k, not approved on 50k.

The experiment uses the augreg2 ViT-B/16 checkpoint, float64, batch size 32, time constant 1, all three spiking LayerNorm stages, attention and MLP, the current GELU construction, and no mismatch or weight noise. Tracking and TensorBoard are disabled. The basic workload is 71 evaluations and nine calibration collections; four short environment comparisons are separate. [[scripts/verification/verify_calibrated_three_sweep_contract.py#main]] verifies the counts, exact grids, table identity, replay, boundary selection, and old metadata rejection.

## Calibrated Three Sweep Scheduling

One local controller owns all assignments and waits for complete seeds in order. Unstarted cluster work can move to local workers only after guarded cancellation and terminal confirmation.

[[scripts/experiments/run_calibrated_three_sweeps.py#Controller]] freezes the source commit, evaluator and runtime hashes, checkpoint, datasets, and per-threshold calibration hashes. Immutable task and phase manifests accompany a resumable assignment record. Only completed results whose raw logs and identities validate can be reused. Partial logs are preserved; a repeatedly failing task requires inspection instead of being classified as scientific accuracy collapse.

Theta collection is followed by matched clean and seed-0 noisy short evaluations on both environments. Training and validation evaluate all nine candidates, then the selected training condition is replayed on the opposite environment. The two noise sweeps share one condition and therefore require 17 evaluations per seed. All seed 0 conditions complete before seed 1; all seed 1 conditions complete before seed 2. Normal completion advances automatically. Missing results prevent advancing.

After seed 0, execution stops for a range decision if all nine timing noise points are within one percentage point of clean accuracy, or all are at most one percent accurate. Individual low or nonmonotonic results are retained. [[scripts/verification/verify_calibrated_three_sweep_runner.py#main]] tests seed completion, resource limits, duplicate prevention, resume behavior, and the range decision. The old threshold-40 campaign was stopped under the user's 2026-09-14 instruction; its complete and partial outputs remain separate.

## Calibrated Three Sweep Distribution

Local GPU devices 4–7 remain the default and share work with the UBAI RTX A6000 partitions. Temporary permission for devices 0–3 applies only to the remaining work of the current campaign; cluster temporary storage stays on verified disk.

The initial assignment is six theta candidates on UBAI and three locally, and eleven noise conditions on UBAI and six locally per seed. Each experiment uses one GPU and four CPU cores. Following the user's allocation change, new UBAI jobs pair two experiments and request two GPUs, eight CPU cores and 128 GiB. Account limits remain ten running jobs, twenty submitted jobs and twelve GPUs. Existing jobs and odd remaining conditions retain one GPU and 64 GiB. No DataParallel is used.

Local admission retains the user's existing limit: total device memory at most 1 GiB and GPU utilization at most 5%, including both endpoints. A foreign compute PID alone does not block a device. [[scripts/experiments/run_calibrated_three_sweeps.py#gpu_available]] is used both when finding available devices and immediately before launch under the device lock. The controller records actual memory, utilization and PIDs; its own assignments and locks still allow only one campaign worker per GPU. Missing telemetry prevents new local launches without stopping cluster tasks.

The user's 2026-09-15 permission requires the explicit `--temporary-local-gpus` option for this campaign and its existing frozen source. It does not change the default device list or other experiment runners. The controller records the active device list separately from the immutable experiment and clears temporary permission when the campaign completes. Without the option, a new controller uses devices 4–7. CPU slots retain the order 4, 5, 6, 7, 0, 1, 2, 3 so existing workers keep their original four cores while additional workers use separate cores.

A controller-only correction may use a separately committed clean checkout while evaluation, calibration, task manifests and active cluster jobs retain their original frozen source. [[scripts/experiments/run_calibrated_three_sweeps.py#controller_identity]] records the controller commit, content hash and admission policy separately and rejects changes to imported experiment or reporting helpers. This avoids restarting scientifically unchanged work merely to correct task allocation.

[[scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py#prepare]] creates a separate deployment using the portable Python 3.12.13 environment and Ubuntu 24.04 image. Heavy checksum and environment verification runs in a Slurm preparation job. Read-only mounts preserve identical source, dataset and checkpoint paths. Every compute task rechecks preparation and source identity. The imported Transformers source subtree and SpikingJelly package subtree have separate content hashes because editable packages are not contained in the main source commit or environment archive.

[[scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py#admit_runtime]] admits environment extraction and scratch space under a filesystem-checked `/enroot` path and rejects tmpfs/ramfs. It reserves concurrent disk usage, binds container temporary paths to this disk, and cleans only the exact owned runtime directory after child exit. Leftovers require terminal Slurm evidence; no age-only cleanup is allowed. [[scripts/verification/verify_calibrated_three_sweep_ubai.py#UBAISafetyTests]] covers allocation, paths, capacity reservation, checksum rejection, and termination cleanup.

## Calibrated Three Sweep Paired Jobs

Two independent experiments share a Slurm job and an extracted environment, while separate GPU devices, process state and result files preserve the existing evaluation contract.

[[scripts/experiments/run_calibrated_three_sweeps.py#quota_available]] counts jobs and allocated GPU devices separately and reserves capacity for pending jobs too. Six paired jobs can run twelve experiments concurrently; twenty submitted jobs do not grant forty concurrent GPU devices. Other account jobs reduce the available capacity. Local devices 4–7 remain a separate pool.

[[scripts/experiments/run_calibrated_three_sweeps.py#Controller#start_remote_pair]] groups only unsubmitted conditions in the current stage. A pair cannot cross seed stages, change a fixed execution environment, or repeat an existing assignment. An odd remaining condition may use an individual job. The immutable pair manifest records raw file hashes for experiment, deployment and task files, plus the controller commit and runtime file hashes. These raw hashes are distinct from the canonical JSON hashes in the assignment record. Evaluation source and task definitions remain frozen separately.

[[scripts/experiments/ubai/run_calibrated_three_sweep_pair.py#run_workers]] gives each evaluator exactly one distinct allocated GPU device and four separate CPU cores. [[scripts/experiments/ubai/run_calibrated_three_sweep_pair.py#admit_pair_runtime]] extracts the environment once on verified disk and reserves an additional scratch allowance for the second evaluator. Separate writable scratch directories prevent cache collisions. Cleanup waits for both child processes; no temporary environment is extracted onto a RAM disk.

One failed condition does not cancel its peer. After Slurm confirms termination, each result is validated independently and only missing conditions are retried. Existing individual jobs remain unchanged during controller replacement. [[scripts/verification/verify_calibrated_three_sweep_pair.py#PairTests]] verifies paired runtime identity, distinct device and CPU assignments, disk reservation and cleanup, and failure isolation. The scheduling verification also covers resource limits, shared job recovery and reuse of complete results.

## Calibrated Three Sweep Local Worker

A separately verified local wrapper changes only device admission and delegates evaluation to the existing frozen worker, preserving the source, conditions, completed results and reporting contract.

[[scripts/experiments/run_calibrated_three_sweep_local_task.py#main]] checks the clean controller commit and content hashes before loading the original worker from the frozen source. It replaces only the worker's device-admission function, strips the wrapper-specific temporary option, and delegates the original task arguments. The evaluator, task and experiment files are not rewritten. Default admission remains devices 4–7; devices 0–3 additionally require the current campaign tag, frozen source and explicit temporary option.

[[scripts/verification/verify_calibrated_three_sweep_local_task.py#main]] verifies default and temporary permission, single-device validation, wrong-campaign rejection, source and controller integrity, and unchanged evaluator delegation. The scheduling tests retain both legacy and new worker identification through exact experiment and task paths, along with separate CPU assignments for eight workers.

## Calibrated Three Sweep Queue Reassignment

When cluster allocation is delayed, available local devices can take over pending conditions without interrupting running cluster evaluations or executing one condition twice.

[[scripts/experiments/calibrated_three_sweep_rebalance.py#rebalance_pending]] considers current-stage jobs pending for at least 60 seconds, within available local capacity. It checks the exact job identifier, name, user and account, and keeps both members of a pair together. A persisted intent precedes placing a user hold. The controller then verifies that the job is still pending with zero priority and its user hold before cancellation. If it started in the meantime, the controller retains remote execution and releases only its own hold; outside holds are preserved.

[[scripts/experiments/calibrated_three_sweep_rebalance.py#poll_rebalance]] waits for terminal Slurm accounting and validates any available results before releasing either assignment. A cancelled job with an explicitly absent start time and zero elapsed time returns to local scheduling without consuming an evaluation retry. Completed results are reused independently. Missing accounting or transport failures leave assignments reserved. Returned conditions cannot be resubmitted remotely, and seed completion order is unchanged. The verification covers paired identity, cancellation races, restart, external holds, partial results and duplicate prevention.

## Calibrated Three Sweep Reporting

Live figures show completed evaluations only; immutable seed snapshots distinguish provisional one- and two-seed results from final three-seed Student-t confidence intervals.

[[scripts/analysis/summarize_calibrated_three_sweeps.py#summarize]] regenerates replica, cell and site CSVs plus progress and source evidence from verified raw logs. Training selection results have a separate CSV from validation plots. Accuracy has three panels: logarithmic threshold and timing noise axes, and a linear deadline margin/noise standard deviation ratio axis. Deadline-miss and pre-clamp rail-saturation rates use a separate figure with true zeros retained.

One seed is a single observation; two seeds give a provisional mean without a 95% interval. Three seeds give the mean and 95% Student-t interval with two degrees of freedom. Physical rates always pool raw numerators and denominators. The shared timing noise and margin condition is evaluated once per seed but appears in both corresponding panels. [[scripts/verification/verify_calibrated_three_sweep_summary.py#main]] checks incomplete evidence, duplicate seeds, pooled rates, confidence intervals and immutable snapshots. Results remain under the campaign artifact tag and are not automatically promoted to the paper.

## Symbolic Operation-Count Check

The paper’s spike-operation and energy formulas have a dedicated symbolic regression checker independent of model execution.

[[scripts/verification/verify_sop.py#main]] recomputes atomic operators, module costs, full ViT formulas, and published rounded values. It encodes fixed-scalar multiplication as free weight calibration through [[scripts/verification/verify_sop.py#free_scale]] instead of counting raw Python calls.

This verifies internal arithmetic consistency under the stated cost model. It does not validate the physical energy constant, system boundary, routing, memory, static power, or circuit feasibility.

## Manuscript Terminology and Notation Check

The manuscript workflow uses a machine-readable pattern lexicon to prevent unreviewed labels and mathematical notation from entering publication-facing artifacts.

`scripts/verification/terminology_lexicon.json` records each concept, its explanation, matching expressions, handling status, affected surface, optional preferred form, and the source supporting the decision. [[scripts/verification/check_terminology.py#main]] consumes this lexicon but does not replace the manuscript as the authority.

The approved notation for the lower bound potential is $V_{\mathrm{lb}}$; publication text must not use the italic form.

The approved primitive definitions use an adjustable weight in $\psi_{\mathrm{NE}}$ and let $\psi_{\mathrm{Int}}$ integrate its supplied current waveform directly, without repeating that weight outside the integral.

The GELU approximation coefficient is locally defined as $c=0.044715$ in the composition table; the longer subscripted form is not used.

The standalone composed tanh operator remains internal implementation support and is omitted from the manuscript composition table. The lowercase $\tanh$ denotes only the mathematical function in the activation mapping.

The checker can inspect files, directories, standard input, or only added lines in a unified diff. Matches marked `review`, `internal-only`, or `forbid` stop publication-facing changes. Heuristic candidate expressions are advisory unless strict candidate handling is requested.

The preflight is required even if automatic skill selection does not occur. It runs on proposed content before mutation and on the exact current task added lines afterward; exit codes 1 and 2 block downstream generation and completion reporting.

The automated pass provides locations and explanations; it cannot determine whether a previously unseen phrase or symbol is semantically justified. Every candidate requires comparison with the manuscript and blocks progress until it is rewritten with an established form or explicitly approved.

## Conference Manuscript Layout

Conference-specific manuscript material is isolated by venue and year so archived submissions and new templates can coexist without ambiguous relative paths.

The withdrawn NeurIPS snapshot, review notes, bibliography, and publication figures live under `paper/neurips_2026/`. The official ICLR 2027 LaTeX template and its original ZIP live under `paper/iclr_2027/`. The complete `paper/` tree remains intentionally untracked.

The ICLR entry point `paper/iclr_2027/iclr2027_conference.tex` retains the preamble, submission metadata, abstract, bibliography, and ordered inputs. Introduction, Related Work, Preliminaries, Methodology, and Appendix content live in matching `iclr2027_conference_<section>.tex` files, each declaring the entry point as its TeX root.

Experiment scripts that publish figures or rebuild the archived NeurIPS manuscript target its venue-specific directory. ICLR manuscript work uses the ICLR directory without overwriting the archived source.

## Verification Boundaries

The repository currently emphasizes end-to-end evaluations, inline operator smoke checks, focused analysis scripts, and the symbolic SOP checker rather than a unified automated unit-test suite.

Before treating a change as validated, select checks proportional to its layer:

- Domain or primitive changes need algebraic value and bound tests, including endpoints and signed event order.
- Composite functions need comparisons against their dense mathematical references across calibrated domains.
- Model changes need noise-free checkpoint fidelity plus task-level smoke evaluation.
- Noise changes need deterministic seed checks, distribution checks, injection-coverage checks, and repeated confidence intervals.
- Cost-model changes need `python scripts/verification/verify_sop.py` and explicit review of modeling assumptions.

`lat check` validates this documentation’s section identities and source references; it is not a substitute for numerical model tests.
