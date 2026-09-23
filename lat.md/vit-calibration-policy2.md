# ViT Calibration Policy 3 Implementation

ViT policy 3 is the maintained range-transfer contract used by the current conversion comparison and calibrated timing-noise campaign; the earlier rollout failures are historical records.

## Numerical Contract

Policy 3 transfers selected Q/K/V and centered LayerNorm ranges into the temporal operations that encode, combine, and restore those values, and requires the current TTFS output head.

Every Q/K/V output and centered LayerNorm input has an independent symmetric selected range. S/B models contain 109 active sites and L contains 217. Calibration observes unclipped values in two deterministic training passes, selects all ranges together, and adds 5% of the observed interval width on each side.

Attention consumes the Q/K ranges in its multiplication and score path, and uses the V range for encoding, its zero reference, restoration, and context bounds. LayerNorm uses one selected centered-input upper endpoint for magnitude, square, variance, logarithmic encoding, and their shared observation deadline. The learned affine coefficients and derived output bounds remain separate.

Checkpoint `layer_norm_eps` is forwarded to every LayerNorm. The current four ViT checkpoints use $10^{-12}$. The logarithmic input floor $10^{-5}$ and variance floor $10^{-10}$ remain fixed. Computing variance from the same bounded centered potential is intentional clipping, not a second untracked approximation.

## Persistence and Compatibility

Frozen evaluation accepts only a table whose model family, policy version, output bounds policy, source, checkpoint, dataset, preprocessing, dtype, epsilon, floor, site set, and collection controls match.

Earlier policy tables and the 48/96-site ViT tables remain readable as historical artifacts but are rejected for policy-3 evaluation. Current calibration is model and source specific and contains no shared range setting. A short-check table never substitutes for the complete training 5k collection.

## Shared Deadline Fix

LayerNorm computes one common observation window for all three logarithmic encodings before Gaussian sampling and event delivery checks.

This avoids one-ULP endpoint disagreement between positive and negative domains without relaxing the primitive deadline invariant. Direct logarithmic paths are constrained to the same declared window. CPU boundary checks and the completed full-model campaigns verify the maintained implementation; the earlier ViT-B short-check failure remains preserved in [[deprecated#과거 실험과 범위 감사#과거 ViT Conversion Comparison]].

## Completed Evidence

New full-model evidence must use policy 3; completed policy-2 results are preserved only as superseded evidence.

The earlier conversion and timing-noise campaigns are preserved in [[conversion-comparison#Superseded Results]] and [[noise#Superseded Calibrated Threshold and Noise Sweeps]]. They use the removed global-range contract and cannot supply current manuscript numbers.

## Verification

Verification covers range collection, transfer, frozen replay, numerical boundaries, inactive configurations, and rejection of mixed identities.

The maintained calibration checks cover 109/217 sites and rejection of old tables. Attention and LayerNorm checks exercise ranges wider than earlier shared envelopes, all normalization ablations, equality when noise is disabled, Gaussian execution with standard deviation zero, seeded finite output, and shared deadlines. Campaign reducers revalidate each table and complete result before aggregation.
