# Quick Model Family Checks

Small diagnostic comparisons inspect performance before full evaluation. They remain separate from publication results and do not reuse a short calibration as a complete training calibration.

## Small ViT Comparison

The first check uses ViT-S ImageNet, 256 fixed training images for calibration and 256 fixed validation images for matched ANN/SNN evaluation. The current float64, theta 40 and ViT policy 2 implementation remain fixed.

[[scripts/experiments/quick_vit_check.py#main]] derives evaluator commands from the current comparison manifest, verifies frozen source and assets, and records the helper hash separately. It acquires one idle GPU from 4–7 using the shared lock and stores runtime on real disk under artifacts. All logs and phase exit status are preserved; no paper table is changed.

## Text Calibration Readiness

BERT, GPT-2 and RoBERTa require the complete [[text-calibration]] contract before calibrated quick comparisons. The earlier audit of missing ranges below records why legacy uncalibrated language runs were not submitted.

At that audit, BERT had no evaluator collection or lifecycle for loading a frozen calibration table, and no Q/K/V or LayerNorm internal calibration bindings. GPT-2 collected residual and attention score ranges but did not bind Q/K/V or LayerNorm internal ranges, and its calibrated evaluator restricted dtype to float32. The shared LayerNorm time window correction alone did not supply those bindings.

The proposed language checks with historical settings were not submitted. Their historical thresholds, BERT epsilon behavior and GPT-2 dense GELU are not silently relabeled as the new ViT configuration. [[scripts/experiments/quick_text_check.py#main]] is only a prepared logging wrapper; it neither implements calibration nor authorizes a run.

The new text policy provides collection, immutable persistence and consumption of selected Q/K/V, attention score, residual, nonlinear input and centered LayerNorm ranges, including executed nonlinear task heads. Each family rejects older tables through its new metadata contract; BERT now forwards checkpoint epsilon to every spiking LayerNorm. GPT-2's dense GELU remains separate from range calibration. New short checks require newly collected tables and retain their own logs.

## Calibrated Text Comparison

Short comparisons use a newly collected training subset and matched ANN/SNN evaluation examples. They inspect performance before full evaluation and cannot replace complete training calibration or manuscript results.

The committed runner uses 256 training examples selected with seed 0, two collection passes and the first 256 held out examples in fixed order. BERT and RoBERTa use SST-2 validation; GPT-2 uses nonempty WikiText-2 test rows with its existing mean of batch losses. All use float64, theta 40, batch size 8, sequence length 128, active temporal attention and LayerNorm, and each model's maintained activation construction. Noise, W&B and TensorBoard are disabled.

[[scripts/experiments/quick_calibrated_text_check.py#main]] requires a clean checkout at the supplied commit, validates the source throughout execution, and records collection, ANN and SNN commands and logs separately. It checks complete evaluation counts, finite metrics and the executed calibration site list before accepting a result. The final comparison records its dataset order and preprocessing identity. Missing or failed phases are preserved as failures rather than silently omitted.

Only idle local GPUs 4–7 are admitted using the shared device lock plus memory and utilization checks. Each process owns one device. Logs and runtime use the canonical main repository artifacts directory even when the source is a separate checkout, so all runners share the same device locks. Existing CIFAR execution is not interrupted. Pending model checks may use the next available permitted device; no external GPU process is terminated.

The ViT-S 256-example check can use a paired source path and commit override to repeat the comparison with the new frozen source. It preserves the prior asset identities and manifest hash while recording a derived experiment manifest; it never relabels an old result as coming from the new commit.
