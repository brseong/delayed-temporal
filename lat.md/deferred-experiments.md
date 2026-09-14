# Deferred Experiments

This page records plausible extensions that are not active TODO items. Do not submit them unless the user explicitly promotes one or the stated manuscript claim makes it necessary.

## Promotion Rule

A deferred experiment becomes active only when it is the shortest available way to support a retained claim or resolve a failure that invalidates current evidence.

A surprising result, an available GPU, or a possible mechanism is not sufficient by itself. First narrow or remove the claim when that resolves the evidence gap without new compute.

## Timing Error Model Refinements

These refinements are unnecessary for the accuracy result on the fixed 5,000-image subset while the manuscript avoids claims about physical event counts or calibrated devices.

- Do not draw timing error for the inactive member of a signed pair; report its counts separately from delivered events.
- Test alternative placement and correlation of the shared reference event, including the start of the code interval.
- Calibrate variation in output time instead of defining $r_t$ per sampled event.
- Repeat timing-noise sweeps over multiple theta values or adaptive sweeps of the deadline margin.

Promote one only if the paper attributes a failure to an internal event population, interprets $r_t$ as a calibrated output distribution, or compares robustness across theta values.

## Additional Robustness Axes

These axes are separate studies rather than missing cells in the current timing-noise sweep.

- The prior 12 by 13 grid of timing noise and deadline margin.
- Frozen threshold mismatch.
- Weight and bias perturbation, temperature variation, time constant variation, and combined perturbations.
- Calibration of noise magnitude for each layer.

Promote an axis only if it becomes an explicit result or a hardware collaborator supplies a defensible calibrated model for it.

## Mechanism and Operator Ablations

These runs investigate causes beyond the accuracy transition needed for the current result.

- Repeat the GELU layer or operator attribution matrices under the final timing representation.
- Sweep LayerNorm shortcut, partial operator, epsilon, and time window configurations.
- Decompose ViT-B or ViT-L conversion loss across operators, clipping, calibration, precision, depth, width, and activation range.
- Repeat comparison baselines solely to establish a ranking.
- Run a pure dtype control with every other range held fixed.

Promote an ablation only if a retained causal or superiority claim cannot be removed or supported by existing evidence.

## Scale and Generality

These evaluations would broaden external validity but are not required for a result from one checkpoint and the fixed 5,000-image subset.

- Full 50,000-image ImageNet validation.
- Additional ViT checkpoints, ViT-L, or other model families.
- Tighter calibration for individual BERT and RoBERTa tasks.
- GPT-2 evaluation with loss weighted by the number of tokens and a full precision matrix.

Promote them only if the manuscript makes a general accuracy, scale, comparison across model families, or standard perplexity claim.

## Validation with Calibrated Hardware

These tasks require a concrete substrate and measurement contract that the current computational model does not provide.

- Correlated device noise models and calibrated parameter distributions.
- Sensitivity to static, leakage, bias, routing, synchronization, and memory energy.
- SPICE, FPGA, device, or silicon measurements.

Promote them only after the manuscript commits to a quantitative hardware claim and the required external evidence is available.

## Current Exclusions

No item on this page is scheduled, a publication blocker, or permission to consume compute.

The active experiment ledger is [[todo#Active Experiment Work]]. Historical scripts and artifacts remain preserved for provenance, but completion is not required unless an item is promoted under the rule above.
