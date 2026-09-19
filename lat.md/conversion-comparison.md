# ViT Conversion Comparison

The current conversion comparison uses frozen calibration, checkpoint-matched preprocessing, and exact held-out counts; earlier direct-resize and incomplete calibration results are historical evidence only.

## GELU Construction

The evaluated ViT path uses the time-constant cubic construction and treats fixed constants as synaptic coefficients rather than separately encoded operands.

Calibration collection and SNN evaluation call the same canonical [[utils/transforms/functions.py#gelu_approximation]]. The final product combines the input with the gate, while the cubic and gate constants do not create extra time encodings. The general multiplication operator is unchanged.

The GELU output lower bound is -0.170041. LayerNorm uses the positive logarithmic input floor $10^{-5}$, checkpoint epsilon $10^{-12}$ for the four ViT checkpoints, and output bounds policy 3. Results produced before this construction or before policy-2 calibration are not reused.

## Evaluation Contract

The comparison selects a separate global threshold for each ViT checkpoint using only its training seed-0 5k artifact. The shared grid is $\theta=5\,2^{i/2}$ for integer indices 0 through 10.

[[scripts/experiments/vit_comparison.py#select_model_theta]] chooses the smallest candidate within 25 correct predictions, or 0.5 percentage points, of the best training accuracy. Validation and test labels never enter selection; each evaluation population is evaluated once after the choice is frozen.

All runs use float64, time constant 1, all three spiking LayerNorm stages, spiking attention, and spiking MLP. Noise, deadline margin, static threshold mismatch, parameter perturbation, W&B, TensorBoard, extra training, and 50k ImageNet evaluation are disabled. Each deterministic ANN/SNN pair is evaluated once without a replica confidence interval.

Calibration uses the model's training seed-0 5k artifact, two deterministic passes, 2,048 histogram bins, observed minimum and maximum, and 5% of the interval width added on each side. Policy 2 requires 109 active sites for ViT-S/B and 217 for ViT-L, including Q/K/V outputs and centered LayerNorm inputs. Frozen execution rejects a different site set, epsilon, preprocessing, dtype, source, checkpoint, or data identity.

CIFAR-10 ViT-S uses test 10k. ImageNet ViT-S/B/L use the existing fixed validation 5k in its saved order. ANN and SNN within a row share the checkpoint, deterministic preprocessing, sample order, batch size, and evaluation population.

## Scheduling

The completed evaluations used one visible GPU per process, persistent artifact storage, and checked runtime paths outside `/tmp`.

Local GPU devices 4--7 remain the default. Campaign-specific temporary access to devices 0--3 has expired and does not alter that rule. UBAI jobs use Slurm compute nodes and place extracted environments and scratch data under checked `/enroot` disk paths, never a RAM filesystem. A result is reusable only after the source, evaluator, dataset, checkpoint, preprocessing, and calibration identities match its manifest.

The active campaign first assigned CIFAR-10 ViT-S and ImageNet ViT-L locally and ImageNet ViT-S/B to UBAI. Local admission initially exhausted three technical attempts when a device changed from idle to occupied between the controller check and the worker's second check; completed results were retained and only the failed attempt counters were reset.

UBAI preparation completed, but all three paired jobs stopped before evaluation because the prepared artifact file membership no longer matched. The controller did not bypass that identity check or accept a partial remote result. After Slurm reported the pair jobs terminal, ImageNet ViT-S/B ownership moved to local devices 6 and 7; devices 4 and 5 continued ViT-L and CIFAR-10 ViT-S. Each evaluator still receives one visible GPU and writes outside `/tmp`.

## Results

Results distinguish complete rows from the active training-only threshold-selection campaign and retained common-threshold evidence. An incomplete row cannot update the paper table.

### Current Training-Only Evidence

The active campaign is frozen as `conversion_comparison_training_selected_theta_float64_bounds3_v4` at source `bf861b55cf1811209e436f8bae98c80240dfa042`; selection uses no validation or test labels.

| Task and model | Selected $\theta$ | Selected training correct | Best training correct | Evaluation population | ANN correct | SNN correct | ANN top-1 | SNN top-1 | Difference |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CIFAR-10 ViT-S/16 | 7.0710678119 | 4,959/5,000 | 4,969/5,000 | test 10k | 9,848 | 9,836 | 98.48% | 98.36% | -0.12 pp |
| ImageNet-1k ViT-S/16 | 7.0710678119 | 4,377/5,000 | 4,389/5,000 | fixed validation 5k | 4,114 | 4,053 | 82.28% | 81.06% | -1.22 pp |
| ImageNet-1k ViT-B/16 | 20 | 4,590/5,000 | 4,590/5,000 | fixed validation 5k | 4,303 | 4,300 | 86.06% | 86.00% | -0.06 pp |

ViT-L remains incomplete. Four of eleven training candidates are complete; the best observed count is 4,608/5,000 at $\theta=14.1421356237$. This interim value does not select a threshold and is not a held-out accuracy result.

The three complete rows are accepted by the campaign result validator, but the aggregate bundle remains incomplete until ViT-L finishes. Therefore these rows are progress evidence rather than authorization to update the manuscript table.

### Retained Common-Threshold Evidence

The prior common-$\theta=40$ rows remain provenance only and are not replicas of the active campaign.

| Task and model | Evaluation population | ANN correct | SNN correct | ANN top-1 | SNN top-1 | Difference |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 ViT-S/16 | test 10k | 9,848 | 9,847 | 98.48% | 98.47% | -0.01 pp |
| ImageNet-1k ViT-S/16 | fixed validation 5k | 4,114 | 4,114 | 82.28% | 82.28% | 0.00 pp |
| ImageNet-1k ViT-B/16 | fixed validation 5k | 4,303 | 4,300 | 86.06% | 86.00% | -0.06 pp |
| ImageNet-1k ViT-L/16 | fixed validation 5k | 4,319 | 4,319 | 86.38% | 86.38% | 0.00 pp |

CIFAR comes from `conversion_comparison_theta40_calibrated_float64_bounds3_v3` at source `b3a50bab6805ae62e620ce5fc12e8e7f5eb2cb05`. The ImageNet rows come from `conversion_comparison_imagenet_timm_theta40_float64_v4` at source `0573a1fad36bf04d512a9eac543851f6769b232a`. The two bundles are combined only because the ImageNet correction does not change CIFAR data or preprocessing; their source identities remain explicit.

## Complete Seven-Model Campaign

The completed campaign also evaluates BERT, RoBERTa, and GPT-2 with fresh model-specific training 5k calibration and their complete held-out populations.

BERT evaluates all 872 SST-2 validation examples and records 806/872 for both ANN and SNN, or 92.4312%. RoBERTa evaluates all 872 examples and records 824/872 for ANN and 823/872 for SNN, or 94.4954% and 94.3807%. GPT-2 evaluates every 2,891 nonempty row in the pinned WikiText-2 raw test artifact, covering 204,257 valid tokens. Its primary token-weighted corpus perplexity is 21.984180 for ANN and 21.984387 for SNN; the retained mean-of-batch-loss compatibility values are 23.253149 and 23.253467.

These text results are complete evaluations of the campaign's held-out artifacts. They are not 50k ImageNet results and do not imply that calibration used each full training split; calibration remains the separate training seed-0 5k procedure in [[text-calibration#Complete Comparison Execution]].

## ImageNet Preprocessing Correction

ImageNet ViT calibration and evaluation use the deterministic evaluation transform of the original timm checkpoints; the older Hugging Face direct-resize results are superseded.

The fixed configuration is 224-pixel input, bicubic interpolation, center crop, crop fraction 0.9, and channel mean and standard deviation 0.5. `scripts/configs/vit_timm_preprocessing.json` has SHA-256 `0b01660e631ce21e41da0000eb79ab2580728d38e42fa1ebd5e14776ecd2443b`. The evaluator stores both this identity and the resolved transform metadata.

The v3 ImageNet rows used a bilinear direct resize produced by the converted Hugging Face processor. They must not be averaged, compared as a replicate, or substituted for the current v4 rows. The current fixed 5k ANN/SNN difference measures conversion fidelity on the same population; absolute accuracy must not be ranked against prior work's full-validation values as if protocols matched.

[[scripts/experiments/run_imagenet_timm_recheck.py#run_pipeline]] enforces fresh calibration, ANN evaluation, and SNN evaluation for each ImageNet model. It accepts only complete pipelines with the timm configuration hash before generating the summary.

## Cost and Paper Integration

Operation and energy values are generated from the actual S/B/L architecture and remain estimates under the declared TTFS implementation boundary.

The calculation includes patch and class tokens, attention output projection, final LayerNorm, attention heads, and classifier dimensions. Energy is total SOP multiplied by 0.9 pJ/SOP; it is not measured GPU or chip energy. The evaluated runtime classifier is dense, while the cost table assumes its stated TTFS counterpart. Detailed formulas and exclusions are in [[comparison-costs]].

Raw logs, calibration tables, summaries, provenance, generated LaTeX, and manuscript build evidence remain under versioned artifact paths. ICLR rows may be filled only from the verified current summaries and must state CIFAR test 10k versus ImageNet fixed validation 5k.

## Clean Accuracy Diagnosis

The severe clean-accuracy drop observed in the early 48-site threshold-40 campaign occurred with an incomplete range-transfer contract and incorrect ImageNet preprocessing; Gaussian timing noise was disabled in those runs.

Policy 2 adds Q/K/V and centered LayerNorm ranges, consumes them through attention and LayerNorm internals, and uses checkpoint epsilon. The timm correction restores the checkpoint evaluation transform. The old 64-image threshold diagnostic and direct-resize results remain useful provenance but are not current model evidence; see [[deprecated#과거 실험과 범위 감사#과거 ViT Conversion Comparison]].

## Verification

Verification separates numerical operator checks, calibration coverage, data identity, evaluation completeness, and generated-table arithmetic.

The evaluator rejects nonfinite logits before metric accumulation. Model checks cover GELU constants, LayerNorm boundaries, 109/217 site execution, frozen table identity, CIFAR order and labels, timm preprocessing, source mixing, and complete correct/total counts. Cost checks recompute SOP partial sums, energy units, and CSV-to-LaTeX equality. `lat check`, terminology checks, and the manuscript build remain required after manuscript-facing changes.
