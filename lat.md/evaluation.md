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

## ViT and GPT-2 Calibration Workflow

The ViT and GPT-2 runners separate clean range collection from frozen validation and inference so validation examples never influence a physical range.

`--calibration-mode collect` selects a fixed-size prefix of a seeded training-split permutation, replays it sequentially for min-max and fixed-bin histogram passes, writes one JSON artifact, and exits without loading validation metrics. Timing noise, mismatch, parameter perturbation, and `DataParallel` are rejected in this mode.

`--calibration-mode validate` and `--calibration-mode inference` reconstruct the same training-subset and model metadata, require an exact artifact match, bind frozen model-entry and per-block residual ranges, and report strict layer underflow and overflow after the run. Robustness noise may be enabled only after the clean artifact identity is checked.

The artifact path is explicit through `--calibration-path`. ViT records image processing and geometry; GPT-2 records empty-text filtering, tokenizer controls, padded sequence length, and dataset configuration. Both record the seeded training subset, checkpoint, TTFS constants, attention path, and active ablations.

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

`scripts/experiments/noise_analysis_vit.sh` and `scripts/experiments/noise_scan_vit.sh` sweep Gaussian timing scale for ViT. The fine scan preserves already-completed outputs, schedules one process per GPU, records its expected-run manifest, and resumes only incomplete tagged logs.

The fine-scan defaults reproduce the canonical ViT-S run. `MODEL_ID` and `BATCH_SIZE` may select a follow-up architecture, while distinct `SCAN_TAG`, `SCAN_FIGURE_PREFIX`, and `SCAN_MODEL_LABEL` values keep that model's artifacts and plotted identity separate from ViT-S.

`TIME_NOISE_STD_FRACS` may replace the default Gaussian grid, and an explicitly empty `MISMATCH_THETA_STDS` omits that independent axis. Gaussian-only refinement manifests remain fully validated and render as one-panel figures rather than fabricating mismatch results.

The scripts pass a dimensionless `time_noise_std_frac`. Each evaluator converts it to one absolute standard deviation using $\sigma_t=r_t(2\theta)$, applies that value at every encoder boundary, and records both values with the seed.

Sweep interpretation is also conditioned on [[noise#Numerical Precision and Endpoint Caveat]]. A float32 result at sub-ULP $\sigma_t$ is a precision-limited implementation diagnostic, not a continuous-Gaussian robustness measurement; float64 reference runs expose endpoint behavior before a range is treated as canonical.

The maintained fine scan evaluates each Gaussian magnitude with timing seeds 0, 1, and 2 while holding the model, 5,000-image subset, and loader seed fixed. Baseline and static threshold mismatch remain single fixed-seed measurements.

[[scripts/analysis/summarize_noise_scan.py#summarize_noise_scan]] rejects missing, failed, or parameter-inconsistent logs before publishing raw and aggregate CSV files. Gaussian accuracy uses a 95% Student-t interval across timing seeds; event-miss and saturation rates pool raw denominators across sites and replicas.

[[scripts/verification/verify_noise_scan_summary.py#verify_noise_scan_summary]] validates manifest constraints, evaluator-log parsing, pooled physical counts, the Student-t interval, artifact rendering, and rejection of Gaussian logs without mechanism statistics using dataset-independent fixtures.

### Per-Layer ViT GELU Attribution

The ViT GELU layer scan isolates where temporal activation errors become task-critical without changing the activation's mathematical formula.

[[scripts/evaluation/error_analysis_vit.py#configure_vit_exact_gelu_layers]] selects zero-based encoder blocks whose MLP GELU uses the maintained cubic-tanh formula in dense arithmetic. Both affine layers remain unchanged, and every unselected block retains the temporal composite.

`scripts/experiments/ablation_gelu_layers_vit.sh` selects exactly one block per condition and compares its noisy accuracy with the corresponding noise-off accuracy. It schedules one process per GPU, resumes complete logs, and permits seed and layer subsets through environment variables.

Recovery relative to the fully temporal noisy run estimates that block's timing-error contribution; it is not an architecture or activation-function comparison. The default seed-zero scan ranks all blocks before additional seeds are assigned to the most influential conditions.

The original float32 layer scan at $r_t=3.162\times10^{-10}$ is precision-limited: its absolute $\sigma_t=1.2648\times10^{-6}$ is below float32 spacing near the GELU log-division deadline. Its block ranking is exploratory and must be repeated under a numerically resolved timing representation before supporting a mechanism claim.

`scripts/experiments/diagnose_gaussian_endpoint_vit.sh` reruns the same 5,000-image condition in float64 using baseline, full Gaussian, block-10 GELU bypass, all-GELU bypass, and all-GELU-plus-LayerNorm-log bypass. These controls determine whether a layer ranking remains identifiable after continuous endpoint behavior is numerically resolved.

The float64 diagnostic places full Gaussian, block-10 bypass, and all-GELU bypass at classification floor, while bypassing both temporal GELU and LayerNorm log restores baseline accuracy. The prior block ranking is therefore not identifiable under resolved continuous endpoint sampling; the endpoint-heavy encodings must be corrected before another layer sweep.

[[scripts/verification/verify_vit_gelu_layer_ablation.py#verify_vit_gelu_layer_ablation]] checks sparse selection, empty-selection behavior, invalid indices, duplicate rejection, and all-or-nothing failure when the expected local ViT topology is absent.

### GELU-Internal Operator Attribution

The GELU operator scan attributes task-level timing sensitivity among multiplication, exponential, and division without changing the production deadline or margin contract.

[[scripts/analysis/gelu_operator_ablation_vit.py#gelu_operator_ablation]] reproduces the maintained cubic-tanh composition while allowing selected GELU-local atomic operators to use their nominal, noise-free temporal carriers. Every unselected GELU operator and every non-GELU use of the same primitive remains on the run-wide Gaussian path.

The `multiplication` unit covers all seven products in one GELU call, including polynomial coefficients and the final input-gate product. The `exponential` unit is tanh's $\exp(-2z)$ stage. The `division` unit includes both negative-log operand encoders and their internal exponential-difference stage because those events jointly implement one ratio.

The eight-condition matrix contains the fully noisy composition, three leave-one-operator-dense conditions, three only-one-operator-noisy conditions, and the all-dense control. Comparing both directions distinguishes an operator whose removal is sufficient for recovery from one whose isolated noise is sufficient for failure.

The dense helpers retain the noise-off temporal arithmetic order, including float32 carrier rounding, and preserve Gaussian-compatible downstream rails. Direct mathematical products or ratios are not used because they would also remove nominal time-code quantization and confound attribution.

The dense helpers also apply the production analytic endpoint clamps after the cubic inner sum and the one-plus-tanh gate, so a selected operator changes event delivery without changing fixed-domain containment.

Selected operators shadow-consume the same Gaussian draws in the same tensor/scalar order but do not apply or count those events. This common-random-number coupling keeps every later GELU and non-GELU event aligned across variants, reducing paired seed variance without representing shadow draws as physical activity.

`scripts/experiments/ablation_gelu_operators_vit.sh` runs one condition per process and GPU, holds model, 5,000-image subset, absolute timing scale, and seed fixed, and resumes only complete logs. [[scripts/analysis/gelu_operator_ablation_vit.py#install_gelu_operator_ablation]] patches only the local ViT GELU symbol, leaving production implementations and other model families unchanged.

This scan deliberately leaves endpoint placement and calibration unchanged. At the existing float32 transition point it is an implementation-level attribution conditioned on [[noise#Numerical Precision and Endpoint Caveat]], not a calibrated continuous-noise robustness result; the matrix must be repeated after a separately reviewed margin calibration becomes canonical.

[[scripts/verification/verify_gelu_operator_ablation.py#verify_gelu_operator_ablation]] checks all eight noise-off subsets for value parity, rejects unknown operator labels, and verifies that installation changes only the local ViT adapter symbol. [[scripts/verification/verify_gelu_operator_ablation.py#verify_gelu_operator_event_selection]] checks physical event topology and equal post-GELU generator state across all dense selections.

#### Observed ViT-S Result

At the existing float32 transition point, GELU division accounts for essentially the entire task-level loss attributed to the temporal GELU composition.

The run uses ViT-S, the first 5,000 ImageNet-1k validation images, `theta=2000`, batch size 32, $r_t=3.162\times10^{-10}$, and therefore $\sigma_t=1.2648\times10^{-6}$. Seed zero evaluates all eight operator combinations; the full, dense-division, only-division-noisy, and all-dense controls are repeated with timing seeds 1 and 2.

Across the three seeds, mean accuracy is 56.353% for fully noisy GELU, 56.360% when division alone remains noisy, 78.773% when division alone is dense, and 78.753% when all GELU-local operators are dense. Division removal recovers 22.420 percentage points, while division-only noise reproduces a 22.393-point loss.

In the complete seed-zero matrix, removing multiplication or exponential alone changes no classifications relative to fully noisy GELU. Leaving only multiplication noisy matches the all-dense accuracy, while leaving only exponential noisy differs from the dense-division control by no classifications. The remaining three-seed dense-division versus all-dense mean difference is only 0.020 percentage points.

The GELU-local division numerator contributes 18,155,520,000 events per 5,000-image run. It misses 23,254,914, 23,250,225, and 23,249,362 times across seeds 0, 1, and 2, respectively, for a mean miss rate of 0.12807%. A numerator opening miss resets the ratio to zero, drives the tanh output to -1, and closes the GELU gate, explaining why sparse misses erase activations and compound through the model.

These measurements identify the division numerator deadline boundary—not continuous multiplication or exponential perturbation—as the dominant implementation-level GELU failure in this configuration. Because $\sigma_t$ is sub-ULP near relevant float32 deadlines, this mechanism statement remains conditional on the current endpoint representation and must be rechecked after margin calibration.

## Gaussian Spike-Time Verification

The maintained Gaussian model requires a seeded decorator-level regression check independent of model datasets and checkpoints.

[[scripts/verification/verify_gaussian_time_noise.py#verify_immutable_memoized_bounds]] rejects mutation of potential and time endpoints and checks that equal attention configurations reuse one bounds object while distinct configurations remain separate.

[[scripts/verification/verify_gaussian_time_noise.py#verify_broadcast_gaussian_time_inputs]] first locks the shared scalar/tensor broadcasting contract, including value alignment plus nominal dtype and device preservation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_time_input_validation]] rejects malformed domains, non-floating or non-finite times, negative scales, and nominal codewords outside the declared interval before sampling.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_sampler_rng_contract]] checks full seeded-stream replay, generator advance across consecutive calls, and exact RNG non-consumption when every standard deviation is zero.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_sampler_deadline_contract]] verifies that early events clamp to the start and fire, deadline equality fires, and only strict exceedance becomes a miss with a finite deadline carrier.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_deadline_probability]] compares the closed-form strict Gaussian tail with seeded empirical misses and checks exact zero-scale probabilities at and beyond the inclusive deadline.

[[scripts/verification/verify_gaussian_time_noise.py#verify_exponential_time_constant_scaling]] checks `tau={0.5,1,2}` across log encoding, exponential decoding, division, softmin, SwiGLU, and LayerNorm, plus domain rejection and RNG preservation.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_encoder_boundary]] enters through the decorated identity encoder to check noise-off tuples, zero-noise event parity, forced misses, and exact per-site event counters.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_statistics_contract]] checks strict pre-clamp rail counters, repeated-site accumulation, detached snapshots, disabled instrumentation, and counter clearing without replacing replica RNG state.

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
