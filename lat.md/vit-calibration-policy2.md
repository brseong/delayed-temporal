# ViT Calibration Policy 2 Implementation

ViT policy 2 is the maintained range-transfer contract used by the current conversion comparison and calibrated timing-noise campaign; the earlier rollout failures are historical records.

## Numerical Contract

Policy 2 transfers selected Q/K/V and centered LayerNorm ranges into the temporal operations that encode, combine, and restore those values.

Every Q/K/V output and centered LayerNorm input has an independent symmetric selected range. S/B models contain 109 active sites and L contains 217. Calibration observes unclipped values in two deterministic training passes, selects all ranges together, and adds 5% of the observed interval width on each side.

Attention consumes the Q/K ranges in its multiplication and score path, and uses the V range for encoding, its zero reference, restoration, and context bounds. LayerNorm uses one selected centered-input upper endpoint for magnitude, square, variance, logarithmic encoding, and their shared observation deadline. The learned affine coefficients and derived output bounds remain separate.

Checkpoint `layer_norm_eps` is forwarded to every LayerNorm. The current four ViT checkpoints use $10^{-12}$. The logarithmic input floor $10^{-5}$ and variance floor $10^{-10}$ remain fixed. Computing variance from the same bounded centered potential is intentional clipping, not a second untracked approximation.

## Persistence and Compatibility

Frozen evaluation accepts only a table whose model family, policy version, output bounds policy, source, checkpoint, dataset, preprocessing, dtype, epsilon, floor, site set, and collection controls match.

Policy-1 tables and the early 48/96-site ViT tables remain readable as historical artifacts but are rejected for policy-2 evaluation. Calibration is model and threshold specific; the nine-candidate ViT-B/16 selection therefore collected nine separate 109-site tables. A short-check table never substitutes for the complete training 5k collection.

## Shared Deadline Fix

LayerNorm computes one common observation window for all three logarithmic encodings before Gaussian sampling and event delivery checks.

This avoids one-ULP endpoint disagreement between positive and negative domains without relaxing the primitive deadline invariant. Direct logarithmic paths are constrained to the same declared window. CPU boundary checks and the completed full-model campaigns verify the maintained implementation; the earlier ViT-B short-check failure remains preserved in [[deprecated#과거 실험과 범위 감사#과거 ViT Conversion Comparison]].

## Completed Evidence

Current full-model evidence replaces the rollout-era short checks as the maintained status of policy 2.

The conversion comparison completed CIFAR-10 ViT-S test 10k and timm-preprocessed ImageNet ViT-S/B/L fixed validation 5k; exact results are in [[conversion-comparison#Superseded Results]]. The noise campaign completed nine threshold tables, selected $\theta=20$, and evaluated the two separate robustness axes plus the high-scale timing-noise extension; see [[noise#Superseded Calibrated Threshold and Noise Sweeps]].

The comparison table still fixes $\theta=40$, whereas the separate timing-noise campaign selected $\theta=20$. This is an experiment-level distinction, not a calibration-policy version change.

## Verification

Verification covers range collection, transfer, frozen replay, numerical boundaries, inactive configurations, and rejection of mixed identities.

`scripts/verification/verify_vit_comparison_runner.py#verify_policy2_and_preparation` checks 109/217 sites and rejection of old tables. Attention and LayerNorm checks exercise selected ranges above the global threshold, all normalization ablations, equality when noise is disabled, Gaussian execution with standard deviation zero, seeded finite output, and shared deadlines. Campaign reducers revalidate each table and complete result before aggregation.
