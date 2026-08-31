# TODO

This file tracks concrete follow-up work that is intentionally deferred from the maintained architecture and operator implementation.

## 2026-08-31 Session Handoff

This handoff records the fixed-domain completion state, GPT-2 precision evidence, unpublished workspace changes, and publication risks that the next session must not infer from the manuscript alone.

### Established State

The maintained implementation now uses static bounds throughout, limits calibration to necessary sites, retains observed min/max with a 5% margin, and uses operator-local GPT-2 attention timing scale.

- Commit `44ddb0b` contains the merged text-model accuracy work: global GPT-2 $\theta=2000$, attention-local $\theta=100$, corresponding metadata identity, validation, and representative wrapper defaults.
- Generated artifacts are deny-by-default. Only the reviewed ViT-S min/max-plus-5% table is whitelisted; alternate GPT-2 calibration and precision logs remain local outputs.
- The simultaneous GPT-2 dense and mixed-window runs are 22.7076 and 22.8991 under the current batch-mean evaluator, a relative PPL increase of 0.843%; see [[evaluation#Fixed-Domain Text-Model Real-Data Audit]].
- The fixed-score-rail sweep and float64 reference isolate the large shared-window attention degradation as predominantly numerical; see [[evaluation#Fixed-Domain Text-Model Real-Data Audit#GPT-2 Floating-Point Precision Control]].

### Delivery State

The reviewed precision-control tooling, appendix note, and knowledge-graph updates are delivered together on top of `44ddb0b`; generated result artifacts remain local.

- `main` contains `44ddb0b` plus the precision-control handoff commit. Their remote publication state must be checked explicitly before assuming they are pushed.
- The requested `/root/.codex/worktrees/a4c5/delayed-temporal` worktree is removed. Other detached and EBRAINS/toy worktrees remain registered and were outside this session's scope.
- The handoff commit adds GPT-2 dtype control, its float32-only calibration guard, verification, the precision sweep, the strict summarizer, and the updated evaluation graph.
- `paper/gpt2_fp_precision_appendix_results_ko.md` is force-tracked as an appendix-ready table and English draft. `artifacts/precision_gpt2/` remains ignored and contains the local raw logs and generated CSV/Markdown.
- The calibration verifier passes all 18 groups, the float64 full-model smoke and held-out run complete, Python and shell syntax pass, `git diff --check` passes, and `lat check` passes. Ruff was unavailable in the active environment.
- Importing custom Transformer families emits pre-existing auto-docstring diagnostics labeled `[ERROR]` for unregistered custom configs and undocumented parameters even though verification exits successfully. This noise should be cleaned or filtered so it cannot hide a real failure.

### Publication Risks

The evidence supports a limited finite-precision claim, but several protocol and manuscript discrepancies remain publication blockers until explicitly resolved.

- The current GPT-2 metric is $\exp$ of an unweighted mean of 181 per-batch losses, not token-weighted corpus perplexity. Every reported comparison uses identical batching, but a publication-facing PPL table should disclose this or be rerun with token-weighted NLL.
- The float64 $\theta=2000$ reference also widens the softmin execution score radius from 40.242257 to 350.772, so it corroborates but does not independently prove a pure dtype intervention. The fixed-radius float32 window sweep is the primary causal control.
- Attention-score excursion counts are recorded before causal-mask overwrite and include future positions. Their absolute rate is an upper-bound diagnostic; only like-for-like sweep comparisons are currently justified.
- `paper/neurips_2026.tex` still reports the old GPT-2 row 22.40 to 23.43 ($+1.03$) and presents one GPT-2 threshold, while the current representative run is 22.7076 to 22.8991 with global/attention thresholds 2,000/100.
- `paper/reviewer_technical_verification_notes_ko.md` still cites the earlier dense value 22.4057 and $+2.6267$ shared-window gap. The current simultaneous dense reference makes that gap $+2.3248$.
- “The entire conversion gap is caused by floating-point precision” is unsupported. The safe claim is that the additional degradation from sharing $\theta=2000$ with attention is predominantly a float32 timestamp-subtraction effect; roughly 0.81--0.84% relative PPL remains.

### Next Session

The next session should resolve metric policy and publication consistency before treating the appendix evidence as final.

1. Decide whether to preserve and disclose batch-mean aggregation or implement token-weighted NLL and rerun the full dense, wrapper, attention-window, and float64 matrix.
2. Reconcile the manuscript and reviewer notes with one canonical simultaneous protocol, including separate global and attention thresholds; manuscript rewriting remains intentionally deferred until authorized.
3. Decide whether the tracked appendix drafting note should be incorporated into the manuscript or retained as a separate internal record.
4. Push `44ddb0b` and the precision-control handoff commit after confirming the intended remote branch.
5. If a reviewer requires a pure dtype intervention, add a reviewed diagnostic that holds the softmin score radius fixed across float32 and float64 rather than over-interpreting the present float64 reference.

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

Calibration runs with timing noise disabled and is identified by stable operator sites. A checkpoint change, preprocessing change, model-family change, or ablation-path change invalidates the affected calibration and requires rebuilding it before evaluation. Static parameter perturbation and threshold mismatch are robustness axes applied only after clean-artifact compatibility succeeds; they do not become calibration-table identities. Parameter-derived affine safety rails are frozen or refreshed after any static parameter perturbation.

Calibration uncertainty is an engineering tolerance rather than a reason to restore runtime extrema. Each site should store signed lower and upper bounds obtained from representative extrema or quantiles, enlarge them by a documented margin, and report calibration-set and evaluation-set clipping rates. The margin may cover moderate distribution variation, but it must not conceal a domain-propagation error, an invalid operator condition, an attention-mask value outside its declared range, or a Gaussian deadline-miss case. Those cases require analytic interval bounds or a direct implementation fix, and inference must never widen a calibrated bound after observing an activation.

Calibration measurement uses two deterministic collection passes before frozen validation. The first pass records signed min/max values; the second replays the same dataset into fixed-bin histograms whose edges come from the first pass. The histogram and margin determine the immutable range table, after which validation applies inference clamps and reports clipping without updating any range.

Calibration sites use one of three policies. Signed-symmetric sites calibrate both tails and enforce a zero-centered rail; lower-bounded and upper-bounded sites preserve their finite analytic endpoint and calibrate only the unbounded direction. Fully bounded operators retain analytic propagation and are not calibration sites.

Nonnegative domains use the globally fixed lower endpoint zero and calibrate only their upper endpoint. Strictly positive logarithmic domains keep a separately configured positive lower rail because a zero endpoint is invalid for logarithmic encoding.

### Acceptance Criteria

The migration is complete only when static-domain behavior is invariant under evaluation batching and all runtime extrema-derived domain construction has left maintained paths.

- [x] Reordering identical samples, changing batch size, or partitioning a batch produces identical declared bounds in representative shared operators, while the AST audit excludes activation-extrema construction across all maintained sites.
- [x] Changing the Gaussian seed changes sampled events and outputs but never changes any declared potential or time bound.
- [x] Every calibrated or Gaussian out-of-envelope value increments pre-clamp underflow or overflow statistics without mutating the envelope.
- [x] Evaluation fails clearly when a required calibrated bound is absent or incompatible instead of silently measuring the current tensor.
- [x] Pre-norm ViT and GPT-2 residual bounds come from fixed per-block calibration entries and therefore do not widen through recursive interval addition during frozen inference.
- [x] A final AST source audit and direct tests reject `PotentialBounds` or `TimeBounds` constructed directly or through local aliases from live forward-tensor extrema.

### Follow-up After Bound Re-Audit

The merged calibration work closes every known live-extrema violation; the remaining work concerns central validation and empirical evaluation rather than runtime calibration.

- [x] Rename the inclusive base interval to `ClosedBounds`; clamping, membership, and deadline equality all include both endpoints.
- [x] Enforce finite, ordered bound endpoints centrally and replace `check_domain` assertions with explicit exceptions that remain active under optimized Python.
- [x] Keep nonzero spiking attention training dropout outside the paper scope; the compatibility branch is documented, and maintained fixed-range claims apply only to evaluation with dropout disabled.
- [x] Run the ViT-S/ImageNet-1k real-checkpoint audit and report per-site clipping, Gaussian saturation, deadline misses, and task accuracy for LayerNorm, attention, affine, embedding, and the conventional task head; see [[evaluation#Fixed-Domain ViT-S Real-Data Audit]].
- [x] Repeat the real-data fixed-domain audit for BERT, RoBERTa, and GPT-2; the classifier gaps are small, and under the simultaneous current protocol GPT-2's attention-local threshold reduces the single-threshold PPL gap from 2.3248 to 0.1915; see [[evaluation#Fixed-Domain Text-Model Real-Data Audit]].

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
- [x] Replace all ViT live activation-derived ranges with preprocessing-derived input bounds, an analytic encoder-entry rail, analytic activation intervals, and two calibrated residual boundaries per block.
- [x] Replace BERT intermediate GELU and ReLU live output extrema with ranges derived from the fixed affine input interval.
- [x] Propagate the fixed BERT encoder range through first-token pooling and use a configuration-derived standalone encoder fallback without live extrema.
- [x] Freeze BERT word, token-type, and position table ranges, sum their intervals before embedding LayerNorm, and preserve the resulting `Potential` through the internal encoder API.
- [x] Remove all RoBERTa live bounds by freezing embedding and affine ranges, propagating `Potential` through the encoder and pooler, and carrying the final range into local LM and classification heads without changing public model outputs.
- [x] Remove all GPT-2 live bounds with frozen embedding and Conv1D intervals, an analytic model-entry range, analytic MLP activation ranges, residual endpoint addition, and two-per-block calibration bindings.
- [x] Add GPT-2 collection, frozen-validation, and inference evaluator modes using filtered WikiText training subsets, fixed tokenizer/sequence metadata, sequential two-pass replay, and per-site clipping reports.
- [x] Prefer operator-derived interval arithmetic whenever the input bounds and transformation provide a conservative static result in the maintained execution path.
- [x] For paths without a practical tight analytic envelope, calibrate only ViT/GPT-2 pre-norm residual resets, ViT composed-GELU pre-activations, and spiking attention scores; analytic model entries bypass calibration.
- [x] Make maintained calibration retain observed min/max without tail truncation and add a 5% per-side margin; keep interior quantiles only as explicit diagnostic overrides.
- [x] Persist stable site identifiers together with the checkpoint, dataset split, preprocessing, model family, and active ablation configuration used for calibration.
- [x] Add explicit collection, frozen-validation, and inference modes so a site cannot measure and clamp against a range created by the same forward invocation.
- [x] Freeze learned-parameter and embedding-table bounds in versioned caches after checkpoint setup instead of recomputing parameter extrema on repeated forwards, including ordinary and spiking LayerNorm.
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
- [x] Initialize every model-family entry potential bound from frozen embedding/preprocessing intervals or explicit calibration rather than measuring the first or current batch.
- [x] Clamp every calibrated out-of-envelope value against its fixed bound and report underflow and overflow counts without widening that bound at runtime; Gaussian operator rails use their separate saturation counters.
- [x] Keep LayerNorm outside calibration because its pre-affine normalized result has the finite analytic envelope $|z_i|\leq\sqrt{d-1}$; derive and freeze the post-affine envelope from scale and bias endpoints.
- [x] Calibrate and clamp every ViT and GPT-2 pre-norm residual output per block, recording raw underflow and overflow before attaching the frozen output range.
- [x] Connect both ViT pre-norm residual boundaries to optional explicit calibration bindings while retaining batch-independent analytic interval addition when calibration is absent.
- [x] Calibrate one symmetric pre-clamp softmin score range per ViT and GPT-2 attention layer, constrain it by the dtype-, temporal-scale-, and sequence-capacity-derived representability ceiling, and retain separate Gaussian value-readout saturation validation.
- [x] Replace the attention-specific `tau_m` and `tau_s` names with one `tau`; model adapters derive it from their shared `tau_s` configuration, and the ViT-only `tau_m` field is removed.
- [x] Calibrate each operator-composed ViT GELU affine pre-activation per layer, retain an analytic final activation interval, and remove avoidable deterministic exponential intermediate overflow without changing the operator equation.
- [x] Keep spike-time windows configuration-derived: LayerNorm log windows remain fixed by `clip_margin`, `theta`, and `tau_s`, while affine identity encoding uses each declared zero-containing fixed interval.
- [x] Make declared potential and time bounds immutable so cached or propagated endpoints cannot be widened in place.
- [x] Keep masked attention scores inside the declared softmin range and clamp both Gaussian and noise-free value readouts to a rail derived from fixed $S_{\max}$ and $\theta$.
- [x] Attach that shared fixed attention-output range to `Potential` in the ViT, BERT, RoBERTa, and GPT-2 adapters instead of reusing the value range or measuring output extrema.
- [x] Remove live activation extrema from the Gaussian `SpikingLayerNorm` path by propagating operator intervals and using the finite-feature dense LayerNorm bound.
- [x] Remove live activation extrema from deterministic `SpikingLayerNorm.forward` with the same operator intervals and finite-feature dense bound.
- [x] Remove live output extrema from ordinary `nn.LayerNorm` calls in `_apply_norm` with the finite-feature bound and learned affine endpoint propagation.
- [x] Verify bounds are identical across batch contents, ordering, and batch size, and add a final source audit that rejects runtime tensor-extrema domain construction in maintained paths.
