# TODO

This file tracks concrete follow-up work that is intentionally deferred from the maintained architecture and operator implementation.

## Causal Signed PWM Migration

Signed temporal differences must subtract two causal event-to-deadline PWM rails so neither physical path integrates backward or requires event-order detection.

- [x] Add an unsigned PWM primitive that integrates one event to a fixed future observation deadline and derives bounds from declared endpoints.
- [x] Add a signed PWM wrapper that reuses one deadline and drive across both event rails so the deadline cancels on subtraction.
- [x] Extend the signed composition to shared `SpikeSample` inputs with symmetric one-sided-miss readout at the observation deadline, without resampling either event.
- [x] Migrate Gaussian multiplication to the signed wrapper while preserving its ideal product bounds and output saturation site.
- [x] Apply symmetric signed-PWM pulse widths in `SpikingLinear._gaussian_forward` while retaining `torch.nn.functional.linear` as the accelerated evaluation of the complete PWM-MAC.
- [x] Apply symmetric signed-PWM pulse widths in `SpikingConv2d._gaussian_forward` while retaining `torch.nn.functional.conv2d` as the accelerated grouped PWM-MAC.
- [x] Apply symmetric signed-PWM pulse widths in GPT-2 `SpikingConv1D._gaussian_forward` while retaining its transposed `torch.addmm` contraction.
- [x] Apply symmetric signed-PWM pulse widths to attention value integration while retaining its optimized matrix-multiplication kernel.
- [x] Migrate exponential difference to the signed wrapper with its physical fixed unit-negative drive and internal exponential reset stage unchanged.
- [x] Apply the same symmetric pulse-width equation to LayerNorm's direct exponential ablation without introducing the disabled internal event.
- [x] Migrate the noise-free `multiplication_operator` call site to signed PWM while retaining direct delivered-tensor evaluation.
- [x] Migrate deterministic exponential difference to signed PWM with the same shared-deadline requirement as its event-aware path.
- [x] Replace explicit deterministic affine and attention synapse tensors with optimized kernels that evaluate the same signed PWM reductions.
- [x] Remove the algebraic single-rail PWM implementation and public export after all maintained callers migrate.
- [ ] After the planned manuscript rewrite, update its definition, proof, and SOP accounting for two parallel causal integration paths without duplicate encoder spikes. This documentation-only task is intentionally deferred and does not block the maintained implementation.

## Static Bounds for All Operators

Every maintained operator must use bounds fixed before inference; a forward pass must never define its own physical rails from values it has already produced.

The completed source audit, formulas, model-family inventory, and all execution cases are documented in [[bounds-audit]].

### Why Static Bounds Are Required

Static bounds turn domains into predeclared physical and mathematical contracts instead of batch-specific observations.

- Physical TTFS rails and observation windows must be configured before an input is encoded. Deriving them from the completed output is an unavailable runtime oracle.
- The same activation must receive the same domain and encoding regardless of batch contents, ordering, batch size, device partitioning, or noise seed.
- A tensor's observed minimum and maximum describe only that batch; they do not conservatively bound future inputs and therefore cannot satisfy the `Potential` contract.
- Widening bounds around a noisy output hides physical underflow and overflow. Raw outputs must be compared with fixed rails before statistics are recorded and clamping is applied.
- Immutable bounds keep deterministic clipping error, operator approximation, Gaussian timing error, deadline misses, and output saturation independently measurable.

### Intended Runtime Contract

Calibration and interval arithmetic establish an immutable bound table before evaluation, after which forward execution may only consume, propagate, compare, and clamp against those bounds.

The objective is not to replace every runtime range with the widest possible analytic interval. Analytic propagation is used only where it stays meaningful and depth-independent; calibration fixes sites whose bounds are difficult to derive, data-dependent, or inflated by repeated residual interval addition.

This range reset is necessary even when every individual operation has a finite formula. For a layer with Lipschitz constant $L_i>1$, $\lVert\delta x_{i+1}\rVert\le L_i\lVert\delta x_i\rVert+\lVert e_i\rVert$, so propagated intervals and upstream clipping error can grow with depth. Fixed layer boundaries deliberately clamp that growth, and validation must report both layer clipping rates and final task accuracy.

The required order is: load or calibrate static input envelopes, derive conservative operator outputs, evaluate the raw tensor, record excursions against the fixed output rail, clamp, and pass the unchanged declared envelope downstream. Neither clean nor noisy execution may widen a bound.

Calibration runs with timing noise disabled and is identified by stable operator sites. A checkpoint change, static parameter perturbation, preprocessing change, model-family change, or ablation-path change invalidates the affected calibration and requires rebuilding it before evaluation.

Calibration uncertainty is an engineering tolerance rather than a reason to restore runtime extrema. Each site should store signed lower and upper bounds obtained from representative extrema or quantiles, enlarge them by a documented margin, and report calibration-set and evaluation-set clipping rates. The margin may cover moderate distribution variation, but it must not conceal a domain-propagation error, an invalid operator condition, an attention-mask value outside its declared range, or a Gaussian deadline-miss case. Those cases require analytic interval bounds or a direct implementation fix, and inference must never widen a calibrated bound after observing an activation.

Calibration measurement uses two deterministic collection passes before frozen validation. The first pass records signed min/max values; the second replays the same dataset into fixed-bin histograms whose edges come from the first pass. The histogram and margin determine the immutable range table, after which validation applies inference clamps and reports clipping without updating any range.

Calibration sites use one of three policies. Signed-symmetric sites calibrate both tails and enforce a zero-centered rail; lower-bounded and upper-bounded sites preserve their finite analytic endpoint and calibrate only the unbounded direction. Fully bounded operators retain analytic propagation and are not calibration sites.

Nonnegative domains use the globally fixed lower endpoint zero and calibrate only their upper endpoint. Strictly positive logarithmic domains keep a separately configured positive lower rail because a zero endpoint is invalid for logarithmic encoding.

### Acceptance Criteria

The migration is complete only when static-domain behavior is invariant under evaluation batching and all runtime extrema-derived domain construction has left maintained paths.

- [ ] Reordering identical samples, changing batch size, or partitioning a batch produces identical declared bounds at every operator site.
- [ ] Changing the Gaussian seed changes sampled events and outputs but never changes any declared potential or time bound.
- [ ] Every out-of-envelope value increments pre-clamp underflow or overflow statistics without mutating the envelope.
- [ ] Evaluation fails clearly when a required calibrated bound is absent or incompatible instead of silently measuring the current tensor.
- [ ] Pre-norm residual bounds come from fixed per-block calibration entries and therefore do not widen through recursive interval addition during inference.
- [ ] A final source audit and direct tests reject `PotentialBounds` or `TimeBounds` constructed from live forward-tensor extrema.

### Implementation Checklist

The implementation work covers every maintained transform and model adapter, not only LayerNorm or operators that directly emit spikes.

- [x] Audit every maintained `PotentialBounds` and `TimeBounds` construction, including model inputs, embeddings, residuals, normalization, activations, attention, projections, and task readouts; the remaining violations are listed in [[bounds-audit#전수 검색 결과]].
- [x] Correct multiplication bounds to use the encoded operand's declared clamped endpoints instead of multiplying every ideal result by the full `theta` rail.
- [x] Restrict ordered division output to the noise-independent $[0,1]$ range; count and clamp Gaussian excursions without restricting the unrestricted exponential-difference primitive used by dual-rail LayerNorm.
- [x] Permanently verify division noise-mode domain identity, zero-noise output statistics, numerator-miss in-range behavior, denominator-miss overflow clamping, internal reset zero, and unrestricted exponential difference for LayerNorm.
- [x] Return softmin weights on the structural $[0,1]$ domain and count Gaussian excursions before the final rail clamp.
- [x] Permanently verify softmin noise-mode domain identity, zero-noise saturation counts, forced-miss excursion accounting, and final $[0,1]$ clamping.
- [x] Return tanh on the structural $[-1,1]$ domain and count Gaussian excursions before the final activation clamp.
- [x] Permanently verify tanh deterministic/zero-noise parity, the common $[-1,1]$ domain, forced excursion accounting, and final clamping.
- [x] Return sigmoid-GELU and Gaussian/deterministic SwiGLU gates on the structural $[0,1]$ domain before downstream multiplication.
- [x] Permanently verify sigmoid-GELU and SwiGLU gate-derived output domains, zero-noise counters, forced gate excursion accounting, and finite clamping.
- [x] Replace global-extrema-times-fan-in bounds in all three affine adapters with exact output-specific interval arithmetic before applying calibration.
- [x] Define `CalibrationMode` with distinct `collect`, `validate`, and `inference` phases so command-line and persisted representations use the same stable values.
- [x] Define the common layer-wise calibration data types: immutable ranges, histograms, layer records, run metadata, and calibration tables, plus mutable min-max observer, histogram observer, and clipping-count state with fixed fields.
- [x] Add a batch-order-independent min-max observer update that records finite signed extrema and tensor-element counts without retaining tensors or autograd graphs.
- [x] Select two deterministic calibration collection passes: signed min/max first, then fixed-bin histograms over the same dataset before frozen validation.
- [x] Construct each second-pass histogram from populated first-pass extrema with an explicit bin count and collection device, zeroed `int64` counters, and no arbitrary widening of constant ranges.
- [x] Accumulate batch-order-independent fixed-bin counts with inclusive outer endpoints, explicit underflow and overflow tails, constant-range handling, and no hidden device transfer.
- [x] Finalize a completed histogram only when bins and tails exactly match the total, copying device counters into an immutable JSON-compatible integer tuple without mutating the observer.
- [x] Select signed lower and upper quantiles from the immutable histogram with outward bin-edge rounding, rejecting cutoffs that fall inside unrecorded tails and leaving margin expansion as a separate policy.
- [x] Expand symmetric ranges on both calibrated sides, but expand one-sided ranges only toward the calibrated endpoint so a finite analytic endpoint never moves; leave zero-width ranges unchanged rather than inventing an absolute epsilon.
- [x] Persist policy-specific optional quantiles, analytic endpoints, and margin separately in each immutable layer calibration record so its final range can be reproduced and audited.
- [x] Build immutable layer records only from identical deterministic passes with zero replay tails, and count strict runtime excursions before autograd-preserving clamp.
- [x] Canonicalize calibration tables by stable layer identity, require exact metadata compatibility, and provide strict versioned JSON save, load, and setup-time lookup.
- [x] Permanently verify observer invariants, quantile and margin selection, frozen clipping, schema rejection, tamper detection, and deterministic persistence round trips.
- [x] Separate two-pass collection from frozen validation and inference with explicit state, one-way phase transitions, missing-site failure, and immutable clipping-report snapshots.
- [x] Declare calibration targets and range policy per layer: symmetric signed rails calibrate both tails, one-sided analytic rails calibrate only the unbounded direction, and fully bounded operators bypass calibration.
- [x] Bind calibration state to stable model-module identities without checkpoint keys, use analytic safety rails during collection, and return persisted clamp rails as `PotentialBounds` during validation and inference.
- [x] Select a fixed-size prefix of a seeded training-split permutation for ViT calibration, replay the exact subset sequentially in both passes, and persist its split, seed, sample count, fingerprint, preprocessing, dtype, and model-path identity.
- [x] Add ViT collection, frozen-validation, and inference CLI modes with strict clean-collection constraints, exact metadata validation, missing-entry failure, and per-layer frozen clipping reports.
- [x] Replace all ViT live activation-derived ranges with preprocessing-derived input bounds, analytic activation intervals, a calibrated encoder-entry range, and two calibrated residual boundaries per block.
- [x] Replace BERT intermediate GELU and ReLU live output extrema with ranges derived from the fixed affine input interval.
- [x] Propagate the fixed BERT encoder range through first-token pooling and use a configuration-derived standalone encoder fallback without live extrema.
- [x] Freeze BERT word, token-type, and position table ranges, sum their intervals before embedding LayerNorm, and preserve the resulting `Potential` through the internal encoder API.
- [x] Remove all RoBERTa live bounds by freezing embedding and affine ranges, propagating `Potential` through the encoder and pooler, and carrying the final range into local LM and classification heads without changing public model outputs.
- [x] Remove all GPT-2 live bounds with frozen embedding and Conv1D intervals, analytic MLP activation ranges, residual endpoint addition, and optional model-entry plus two-per-block calibration bindings.
- [ ] Prefer operator-derived interval arithmetic whenever the input bounds and transformation provide a conservative static result.
- [ ] For paths without a practical analytic envelope, record per-site minima and maxima during a representative noise-free calibration run.
- [ ] Persist stable site identifiers together with the checkpoint, dataset split, preprocessing, model family, and active ablation configuration used for calibration.
- [ ] Add explicit collection, frozen-validation, and inference modes so a site cannot measure and clamp against a range created by the same forward invocation.
- [ ] Freeze learned-parameter bounds once after checkpoint loading or static perturbation instead of recomputing parameter extrema on every forward.
- [x] Add `SpikingLinear.freeze_parameter_bounds` with exact sign-aware rails per fixed input domain, immutable reuse, mutation rejection, and explicit refresh.
- [x] Allow `SpikingLinear._gaussian_forward` to use the frozen output rail for saturation accounting without rescanning parameters.
- [x] Connect `SpikingLinear.forward` so deterministic and Gaussian execution attach the same frozen output rail and deterministic execution performs no parameter extrema scan.
- [x] Remove the transitional `domain_W` argument and fallback from `SpikingLinear._gaussian_forward`, eliminating the remaining Gaussian weight scan.
- [x] Apply the same fixed-input-domain interval arithmetic, parameter mutation validation, and noise-independent metadata to grouped `SpikingConv2d`.
- [x] Apply the same fixed-input-domain interval arithmetic, parameter mutation validation, and noise-independent metadata to GPT-2 `SpikingConv1D`.
- [x] Make all three affine adapters consume upstream zero-containing fixed ranges, derive the zero-reference time from those ranges, and permanently verify asymmetric-domain parity and memoization.
- [x] Add `SpikingLayerNorm.freeze_parameter_bounds` for dense, direct exponential, and spiking exponential-difference envelopes with parameter/configuration mutation rejection.
- [x] Connect `SpikingLayerNorm._gaussian_forward` to frozen weight, bias, and final output domains before event sampling.
- [x] Connect deterministic `SpikingLayerNorm.forward` to the same frozen parameter and output contract.
- [x] Permanently verify all eight `SpikingLayerNorm` ablation domains, deterministic/zero-noise metadata identity, stale-cache rejection, and explicit refresh.
- [ ] Initialize every model-family entry potential bound from calibration rather than measuring the first or current batch; ViT is complete.
- [ ] Clamp every out-of-envelope value against its fixed bound and report underflow and overflow counts without widening that bound at runtime.
- [ ] Include LayerNorm's pre-affine normalized result in calibration, then derive its post-affine envelope by interval arithmetic from fixed scale and bias endpoints.
- [ ] Calibrate and clamp every pre-norm residual output per block, recording raw underflow and overflow before attaching the frozen output range; ViT is complete, and GPT-2 adapter sites are connected but its evaluator artifact lifecycle remains.
- [x] Connect both ViT pre-norm residual boundaries to optional explicit calibration bindings while retaining batch-independent analytic interval addition when calibration is absent.
- [ ] Calibrate attention score clamp sites per layer/head and attention value outputs per layer, with sequence-capacity metadata and separate Gaussian saturation validation.
- [ ] Keep spike-time windows configuration-derived: LayerNorm log windows remain fixed by `clip_margin`, `theta`, and `tau_s`, while affine identity encoding uses each declared zero-containing fixed interval.
- [x] Make declared potential and time bounds immutable so cached or propagated endpoints cannot be widened in place.
- [x] Keep masked attention scores inside the declared softmin range and clamp both Gaussian and noise-free value readouts to a rail derived from fixed $S_{\max}$ and $\theta$.
- [x] Attach that shared fixed attention-output range to `Potential` in the ViT, BERT, RoBERTa, and GPT-2 adapters instead of reusing the value range or measuring output extrema.
- [x] Remove live activation extrema from the Gaussian `SpikingLayerNorm` path by propagating operator intervals and using the finite-feature dense LayerNorm bound.
- [x] Remove live activation extrema from deterministic `SpikingLayerNorm.forward` with the same operator intervals and finite-feature dense bound.
- [x] Remove live output extrema from ordinary `nn.LayerNorm` calls in `_apply_norm` with the finite-feature bound and learned affine endpoint propagation.
- [ ] Verify bounds are identical across batch contents, ordering, and batch size, and add a final source audit that rejects runtime tensor-extrema domain construction in maintained paths.
