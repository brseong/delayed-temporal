# TODO

This file tracks work required for the current manuscript and maintained implementation. Optional experiment ideas are kept separately in [[deferred-experiments]] without checkboxes.

## Active Experiment Work

Only the explicitly listed evaluations are authorized. Unchecked items elsewhere are manuscript or packaging work unless they define a separate approved run.

### GPT-2 Composed GELU Rerun

The user authorized a fresh calibrated WikiText-2 evaluation after replacing the direct GPT-2 block activation with the maintained composed GELU.

- [x] Connect `gelu_new` selected by the checkpoint to the composed operator path and distinguish it in calibration metadata.
- [x] Preserve the earlier direct activation logs and assign the rerun a separate tag.
- [x] Collect fresh ranges from the fixed training 5,000 artifact.
- [x] Evaluate the dense reference and converted model on all 2,891 nonempty fixed test texts.
- [x] Validate and summarize the new artifact before changing the ICLR GPT-2 row or prose.

### ViT-B Timing Robustness

The active scope is the fixed 5,000-image validation subset at $\theta=40$, a deadline margin/noise standard deviation ratio of 4, and timing seeds 0, 1, and 2.

- [x] Aggregate the completed `v3` pilot through its first transition bracket: $r_t=10^{-5}$ remains near clean accuracy while $r_t=10^{-4}$ is degraded. Larger completed scales add no evidence to the first-transition decision.
- [x] Implement constant synaptic scaling at `0eec3e2`, bind the adaptive evaluator and reporting tools to completed logs at `f48dfc3`, and correct GPU process tracking across execution namespaces at experiment source commit `67de065`.
- [ ] Use the implemented diagnostic in the new clean 5,000-image pass to record the count and rate of scalar values entering the GELU cubic term outside $[-\theta,\theta]$ before clamping. Report it as metadata for interpretation, not as an acceptance gate or authorization for further runs; replace the preliminary smoke value only after the clean 5,000-image pass completes.
- [ ] Run the corrected sweep one scale at a time, starting with $r_t=10^{-5}$ and seeds 0, 1, and 2. Report each complete scale and require explicit user approval before creating the next round; never submit a larger scale after collapse.
- [ ] Keep the pilot and accepted results from the changed source in separate CSV files and figures, and never aggregate them into one statistical series. State that 50,000-image validation was omitted and retain $\theta=2000$ artifacts as provenance only.

## Manuscript Revision Master Checklist

This is the canonical status ledger for manuscript-facing work; the former review checklist remains a provenance record, and duplicate actions are consolidated here.

Source details and reviewer labels are consolidated in [[deprecated#과거 리뷰]]. Update task status here first, then reflect completed claims in `paper/neurips_2026/neurips_2026.tex` and the submission checklist.

### Current ViT-B Noise and Time-Scale Update

This manuscript checklist records only reporting decisions required by the active 5k result; it does not authorize additional sweeps.

The completed theta selection below belongs to source `bc973317`. The current `648af9bb` GELU campaign reuses 40 without a new candidate search and disables layer-wise calibration; see [[noise#Timing Noise Scale Sweep at Ratio 4]].

- [x] Select the clean ViT-B/16 threshold on the fixed training 5k subset and close the lower boundary with $\theta\in\{10,20,40,80,\ldots,4000\}$.
- [x] Replay the selected $\theta=40$ condition and confirm it against neighboring $\theta\in\{20,40,80\}$ on the fixed validation 5k prefix.
- [ ] Report why theta 40 was the smallest acceptable candidate for the selection source, using the predefined 0.5 percentage-point tolerance from the best spiking candidate; distinguish the later GELU implementation and its reuse of 40.
- [ ] Describe the selected threshold as an intentional tradeoff between accuracy and input domain size, not a constraint requiring zero clipping; distinguish scalar GELU counts collected before clamping from image fractions and measured top-1 accuracy loss.
- [ ] Explain that the results at theta 20 and 10 close the lower candidate range for the original selection source, not a repeated search under the corrected GELU implementation.
- [ ] State that the current noise result omits layer-wise calibration and retains fixed log lower endpoints and the limit on the GELU exponential input; do not report it as validation of the full design for selecting layer-wise ranges.
- [ ] Explain how the dimensionless code interval maps to a declared physical duration and state that every time constant must be rescaled consistently; list timing resolution, timing error, leakage, synchronization, and realizable time constants as limitations.
- [ ] State that the physical timing-error standard deviation and margin follow from the declared mapping rather than treating $\theta$ as a hardware time constant.
- [ ] Report the timing-noise configuration, injection scope, seeds, confidence interval, and empirical miss statistics from the accepted logs.
- [ ] Mark the old $\theta=2000$ robustness evidence as superseded for manuscript support and label the current result as applying only to the fixed 5,000-image validation subset.

### P0 Claims, Novelty, and Structure

The central claim must describe the demonstrated fixed-form operator composition without implying unverified biological or hardware realization.

- [ ] Fix the one-sentence research question and contribution around composing Transformer operations from a small library of fixed continuous-time operator forms rather than claiming the first nonlinear or TTFS Transformer realization.
- [ ] Replace `standard primitives`, `biologically standard`, `hardware-native`, `directly hardware-compatible`, `readily implementable`, and comparable completion claims with evidence-bounded language.
- [ ] Describe TTFSFormer and other conversion baselines by verifiable construction differences; acknowledge their existing softmax, LayerNorm, GELU-family, and TTFS contributions.
- [ ] Replace subjective comparison columns such as `Bio primitives only` with function-specific kernels, fixed operator library, explicit dynamics, circuit validation, and device-assumption columns.
- [ ] Audit `lossless`, `near-lossless`, `exact`, and `highly accurate`; use constructive or approximate language unless a quantitative bound supports the stronger term.
- [ ] Describe the MatMul $O(3MKN)$ to $O(MKN)$ change as a constant-factor operation-count reduction, not an asymptotic complexity improvement.
- [ ] Fairly describe the latency and efficient-simulation benefits of small-$T$ discrete-time methods.
- [ ] Rebuild the paper in the order: problem and gap, model and assumptions, fixed operators, constructive composition, deterministic error, conversion evidence, perturbation sensitivity, and hardware limitations.
- [ ] Map each current paragraph, theorem, table, and figure to keep, shorten, move to appendix, delete, or rewrite before polishing the abstract and introduction.

### P0 Mathematical and Operator Audit

Every retained theorem and operator claim must match the implemented equations, domains, finite-window behavior, and failure conditions.

- [ ] Reclassify Theorem 1 as a quantitative theorem with conditions and bounds, a conditional theorem, or a constructive proposition.
- [ ] Tabulate each primitive and composition with input domain, clipping condition, scale, zero-error condition, and worst-case or empirical error.
- [ ] Derive an $L$-block error relation where defensible; otherwise state why no useful end-to-end bound is available and limit the claim to empirical evidence.
- [ ] Present all four signed multiplication sign cases and the causal source--sink routing used by the two parallel causal integration paths.
- [ ] Define $\psi_{\mathrm{Int}}$ as a time-window integration mechanism based on an NMDA plateau rather than equating it with a validated biological spike or completed circuit.
- [ ] Update the signed multiplication definition, proof, and SOP accounting for two parallel causal paths without duplicating encoder spikes.
- [ ] Recheck softmax stability, exponent scaling, normalization, lower and upper clipping, deadline behavior, and the meaning and calibration of $\alpha$ against the implementation.
- [ ] Re-derive the GELU relation involving $1/(1+\kappa e^{-\beta x})$, state the required $\kappa$ and time-constant conditions, and quantify finite-window error.
- [ ] Write LayerNorm's actual target as $\sqrt{v+\epsilon_{\mathrm{LN}}}$ and keep $\epsilon_{\mathrm{LN}}$ distinct from the encoder floor $\epsilon_{\mathrm{enc}}$.
- [ ] Unify the LayerNorm references as $H$ and $H^2$, including the finite-domain definition used in code.
- [ ] Resolve the residual $1/\sqrt{\theta}$ explanation against any statement that no residual scale remains.
- [ ] Add deterministic boundary tests for zero variance, centered values close to zero, symmetric inputs of both signs, and declared upper bounds in the LayerNorm configuration used for publication.
- [ ] Have a human audit every retained lemma, theorem, and appendix proof line by line and record the verifier and status.
- [ ] Move routine statements to the appendix and foreground the genuinely new composition and failure conditions.
- [ ] Disclose accurately whether LLM assistance affected proof generation, transformation, verification, or only prose editing.

### P0 Conversion Fidelity and Comparison Fairness

Performance tables must use comparable protocols and expose degradation rather than selecting only favorable model sizes.

- [ ] Recompute every ANN-to-SNN absolute result and delta from generated artifacts, including the current GPT-2 metric policy and the new ViT-B operating point.
- [ ] Show checkpoint, preprocessing, training method, split, precision, and evaluation differences for every literature comparison.
- [ ] Remove ranking and superiority language unless an existing result already matches the checkpoint and protocol exactly.
- [ ] Summarize ViT-S, ViT-B/L, and GPT-2 together so the smallest observed degradation is not presented as universally representative.
- [ ] Describe the current GPT-2 batch-mean metric precisely and do not call it token-weighted corpus perplexity.

### P0 Energy, Latency, and Hardware Claims

Energy and feasibility claims must expose their system boundary and remain proxies unless supported by hardware-level evidence.

- [ ] Cite the source and conditions for $E_{\mathrm{AC}}=0.9$ pJ, including process, voltage, circuit type, and whether it is measured or estimated.
- [ ] Label the present calculation as an idealized SOP-only estimate and list omitted memory, routing, fanout, synchronization, static current, comparator, calibration, conversion, mismatch, and integration-time costs.
- [ ] Define the target substrate and a consistent system boundary before comparing against ANN or neuromorphic baselines.
- [ ] Estimate per-operator and end-to-end latency, including the selected code window and the physical-time rescaling assumption.
- [ ] Do not infer energy superiority from operation-count equality; use a comparable ANN energy boundary or narrow the claim.
- [ ] Audit external energy numbers such as SpikeZIP's 100.8 mJ for boundary compatibility before making direct comparisons.
- [ ] Keep hardware-level superiority outside the claim because SPICE, FPGA, device-level, and silicon evidence are absent.

### P1 Robustness and Non-Ideality Evidence

Computational stress tests must be separated from calibrated device models and from one another.

- [ ] Document each timing-noise distribution, magnitude, injection site, seed, repetition count, and confidence interval.
- [ ] Treat additive jitter with deadline misses as the maintained computational model; do not describe it as a calibrated neuronal noise process.
- [ ] Report accuracy against empirical miss rate and the available counts by operator site as simulator diagnostics only; do not interpret them as a physical event population or energy estimate.
- [ ] State whether timing error is injected at encoder outputs or at every internal $\Phi/\Psi$ boundary, and describe the latter coverage explicitly.
- [ ] State that frozen threshold mismatch and every other uncertainty axis in [[deferred-experiments#Additional Robustness Axes]] are outside the current result.
- [ ] Package the clean baseline, checkpoint, evaluator, manifests, and current jitter evidence before requesting analog hardware collaboration.

### P1 Scalability and Exposition

The paper must explain the computation to an ML reader and bound extrapolation beyond evaluated models.

- [ ] Add one end-to-end potential-to-time-to-potential flow diagram and a small numerical example.
- [ ] Tabulate each primitive's input, output, units, time constants, threshold, scale, and proposed neuron or circuit interpretation.
- [ ] Explain one complete $\Psi$-after-$\Phi$ composition at the ANN-layer level.
- [ ] Emphasize causal masking by omitted synapses and the logarithmic LayerNorm construction without overstating biological realization.
- [ ] Estimate SOP, memory, fanout, communication, calibration, latency, and energy bottlenecks for ViT-H/14 or billion-parameter scale.
- [ ] Discuss how 3D and multimodal Transformers change sequence length, operators, routing, and temporal-window constraints.

### P2 Limitations and Reproducibility

The release must make unsupported scope and reproducibility boundaries explicit.

- [ ] List deployment obstacles: mismatch, calibration, leakage, routing, fanout, synchronization, interfaces, latency, and hardware-dependent timing.
- [ ] List conversion losses: accuracy, numeric stability, dynamic range, clipping, depth-wise accumulation, and hardware dependence.
- [ ] State unvalidated scope: silicon, SPICE/FPGA, large models, multimodal models, and calibrated analog non-idealities.
- [ ] Generate every SOP total and publication table through checked scripts.
- [ ] Publish per-model time constants, thresholds, windows, clipping ranges, calibration, and precision.
- [ ] Publish checkpoint identifiers, preprocessing, splits, seeds, repetitions, evaluator commands, and baseline provenance.
- [ ] Validate each configuration used for publication with operator fixtures and existing artifact identity checks; any new dataset evaluation belongs to [[deferred-experiments]].
- [ ] Record compute device, memory, per-run duration, total successful compute, and material failed or preliminary compute.
- [ ] Verify licenses and usage terms for datasets, libraries, checkpoints, and released derived artifacts.

### NeurIPS Submission Checklist Audit

Submission-form answers must be revalidated after the manuscript changes rather than inherited from the withdrawn version.

- [ ] Reconcile Claims, Limitations, Assumptions/Proofs, Reproducibility, and Experimental Details answers with the final section references.
- [ ] Update the statistical-significance answer: deterministic conversion results are singletons, while stochastic robustness results use three replicas and explicitly defined 95% Student-t intervals.
- [ ] Complete the currently unfinished compute-resources justification with hardware, memory, wall time, and aggregate compute.
- [ ] Recheck code/data access, anonymous-release URLs, exact commands, environment versions, and which experiments are omitted.
- [ ] Audit ethics, broader impacts, safeguards, licenses, new assets, human-subject and IRB `N/A` answers against the final released assets.
- [ ] Make the declaration of LLM use match the actual use in prose, code, derivation, and verification.

### Recommended Execution Order

Work should proceed by evidence dependency so prose never outruns mathematical or empirical support.

1. Complete the reviewer-issue ledger and classify every issue as valid, partly valid, misunderstanding, or requiring evidence.
2. Fix the central question, claim boundary, paper outline, and keep/move/delete map.
3. Complete the mathematical and operator audit before changing theorem language.
4. Isolate deterministic conversion errors and settle metric and baseline policy.
5. Finalize the $\theta$ and physical-time interpretation, then run only the approved robustness diagnostics.
6. Rebuild the energy and latency section using an explicit system boundary.
7. Prepare the collaborator package before adding calibrated device claims.
8. Rewrite the abstract and introduction last, then update the NeurIPS submission checklist.

### Final Release Gate

The next public version is ready only when claims, equations, generated evidence, and disclosure documents agree.

- [ ] Abstract, introduction, theorem statements, experiments, limitations, and conclusion use the same claim strength.
- [ ] Every major equation has an independent verification record.
- [ ] Every table and figure is reproducible from preserved generated artifacts.
- [ ] Every comparison declares whether checkpoints and protocols are directly comparable.
- [ ] Every energy table states its system boundary and included and excluded costs.
- [ ] The NeurIPS LLM-use answer matches the actual workflow.
- [ ] Limitations include hardware non-validation, approximation error, scaling limits, and analog non-ideality scope.
- [ ] The paper is understandable without reviewer responses or internal notes.
- [ ] Negative settings and failure conditions are reported alongside favorable results.

### Imported Completed Foundations

These completed items are retained here so the migrated checklist does not lose the legacy status record.

- [x] Separate LayerNorm's denominator regularizer from the finite-window encoder floor.
- [x] Separate actual dual-rail magnitude from floor-clamped logarithm inputs and preserve inactive-rail no-spike semantics.
- [x] Compute LayerNorm variance from actual magnitudes rather than floor-clamped rails.
- [x] Publish the tensor versus spiking execution topology for the LayerNorm stages and model configurations.
- [x] Decompose the current GPT-2 degradation across attention, LayerNorm, MLP affine, residual, and wrapper paths.


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
- The precision results and protocol are consolidated in [[deprecated#과거 실험과 범위 감사#GPT-2 정밀도 통제 실험]]; the former appendix note is preserved in [[deprecated#원본 복구]]. `artifacts/precision_gpt2/` remains ignored and contains the local raw logs and generated tables.
- The calibration verifier passes all 18 groups, the float64 full-model smoke and held-out run complete, Python and shell syntax pass, `git diff --check` passes, and `lat check` passes. Ruff was unavailable in the active environment.
- Importing custom Transformer families emits pre-existing auto-docstring diagnostics labeled `[ERROR]` for unregistered custom configs and undocumented parameters even though verification exits successfully. This noise should be cleaned or filtered so it cannot hide a real failure.

### Publication Risks

The evidence supports a limited finite-precision claim, but several protocol and manuscript discrepancies remain publication blockers until explicitly resolved.

- The current GPT-2 metric is $\exp$ of an unweighted mean of 181 per-batch losses, not token-weighted corpus perplexity. Every reported comparison uses identical batching, but a publication-facing PPL table should disclose this or be rerun with token-weighted NLL.
- The float64 $\theta=2000$ reference also widens the softmin execution score radius from 40.242257 to 350.772, so it corroborates but does not independently prove a pure dtype intervention. The fixed-radius float32 window sweep is the primary causal control.
- Attention-score excursion counts are recorded before causal-mask overwrite and include future positions. Their absolute rate is an upper-bound diagnostic; only like-for-like sweep comparisons are currently justified.
- `paper/neurips_2026/neurips_2026.tex` still reports the old GPT-2 row 22.40 to 23.43 ($+1.03$) and presents one GPT-2 threshold, while the current representative run is 22.7076 to 22.8991 with global/attention thresholds 2,000/100.
- Earlier GPT-2 reference values are retained in [[deprecated#과거 실험과 범위 감사#교차 모델 평가의 비교 한계]]; do not mix them with the simultaneous precision control.
- “The entire conversion gap is caused by floating-point precision” is unsupported. The safe claim is that the additional degradation from sharing $\theta=2000$ with attention is predominantly a float32 timestamp-subtraction effect; roughly 0.81--0.84% relative PPL remains.

### Next Session

The next session should resolve publication consistency without silently expanding the experiment matrix. Optional reruns are catalogued in [[deferred-experiments]].

1. Preserve and disclose the batch-mean aggregation; do not label it token-weighted corpus perplexity.
2. Reconcile the manuscript and reviewer notes with one canonical simultaneous protocol, including separate global and attention thresholds; manuscript rewriting remains intentionally deferred until authorized.
3. Decide whether the tracked appendix drafting note should be incorporated into the manuscript or retained as a separate internal record.
4. Push `44ddb0b` and the precision-control handoff commit after confirming the intended remote branch.
5. Do not use the present float64 reference as a pure dtype intervention; the optional control that holds the softmin score radius fixed is in [[deferred-experiments#Mechanism and Operator Ablations]].

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

The objective is not to replace every runtime range with the widest possible analytic interval. The three cases in [[domain#Domain Propagation]] retain practical structural bounds, use calibration at selected finite but widening boundaries, and restrict domains with no finite output or timing bound. The implementation also supports disabled calibration with conservative analytic propagation; this mode does not exercise the intended layer-wise limits.

This range reset is necessary even when every individual operation has a finite formula. For a layer with Lipschitz constant $L_i>1$, $\lVert\delta x_{i+1}\rVert\le L_i\lVert\delta x_i\rVert+\lVert e_i\rVert$, so propagated intervals and upstream clipping error can grow with depth. Fixed layer boundaries deliberately clamp that growth, and validation must report both layer clipping rates and final task accuracy.

The required order is: load or calibrate static input envelopes, derive conservative operator outputs, evaluate the raw tensor, record excursions against the fixed output rail, clamp, and pass the unchanged declared envelope downstream. Neither clean nor noisy execution may widen a bound.

Calibration runs with timing noise disabled and is identified by stable operator sites. A checkpoint change, preprocessing change, model-family change, or ablation-path change invalidates the affected calibration and requires rebuilding it before evaluation. Static parameter perturbation and threshold mismatch are robustness axes applied only after clean-artifact compatibility succeeds; they do not become calibration-table identities. Parameter-derived affine safety rails are frozen or refreshed after any static parameter perturbation.

Calibration uncertainty is an engineering tolerance rather than a reason to restore runtime extrema. Each site should store signed lower and upper bounds obtained from representative extrema or quantiles, enlarge them by a documented margin, and report calibration-set and evaluation-set clipping rates. The margin may cover moderate distribution variation, but it must not conceal a domain-propagation error, an invalid operator condition, an attention-mask value outside its declared range, or a Gaussian deadline-miss case. Those cases require analytic interval bounds or a direct implementation fix, and inference must never widen a calibrated bound after observing an activation.

Calibration measurement uses two deterministic collection passes before frozen validation. The first pass records signed min/max values; the second replays the same dataset into fixed-bin histograms whose edges come from the first pass. The histogram and margin determine the immutable range table, after which validation applies inference clamps and reports clipping without updating any range.

Selected calibration sites use signed-symmetric, lower-bounded, or upper-bounded endpoint policies. These are distinct from the three reasons for selecting a range in [[domain#Domain Propagation]]. Practical structural bounds remain analytic, but a finite interval that grows excessively can still require calibration.

For nonnegative sites selected for one-sided calibration, the lower endpoint stays at zero and only the upper endpoint is calibrated. Logarithmic domains keep a separately configured positive lower endpoint; current ViT/GPT-2 bindings do not select that endpoint from data.

### Acceptance Criteria

The migration is complete only when static-domain behavior is invariant under evaluation batching and all runtime extrema-derived domain construction has left maintained paths.

- [x] Reordering identical samples, changing batch size, or partitioning a batch produces identical declared bounds in representative shared operators, while the AST audit excludes activation-extrema construction across all maintained sites.
- [x] Changing the Gaussian seed changes sampled events and outputs but never changes any declared potential or time bound.
- [x] Every calibrated or Gaussian out-of-envelope value increments pre-clamp underflow or overflow statistics without mutating the envelope.
- [x] Evaluation fails clearly when a required calibrated bound is absent or incompatible instead of silently measuring the current tensor.
- [x] With frozen calibration enabled, selected ViT and GPT-2 residual boundaries use persisted ranges for each block instead of accumulating analytic interval sums; disabled calibration retains those sums.
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
- [x] Declare calibration targets and endpoint policy per layer: signed-symmetric sites calibrate both sides, one-sided sites preserve their known endpoint, and practical structural bounds remain analytic; finiteness alone does not exclude a site.
- [x] Bind calibration state to stable model-module identities without checkpoint keys, use analytic safety rails during collection, and return persisted clamp rails as `PotentialBounds` during validation and inference.
- [x] Select a fixed-size prefix of a seeded training-split permutation for ViT calibration, replay the exact subset sequentially in both passes, and persist its split, seed, sample count, fingerprint, preprocessing, dtype, and model-path identity.
- [x] Add ViT collection, frozen-validation, and inference CLI modes with strict clean-collection constraints, exact metadata validation, missing-entry failure, and per-layer frozen clipping reports.
- [x] Replace ViT bounds from live activation extrema with preprocessing and analytic intervals plus two optional residual calibration boundaries per block; disabled calibration retains fixed residual interval sums.
- [x] Replace BERT intermediate GELU and ReLU live output extrema with ranges derived from the fixed affine input interval.
- [x] Propagate the fixed BERT encoder range through first-token pooling and use a configuration-derived standalone encoder fallback without live extrema.
- [x] Freeze BERT word, token-type, and position table ranges, sum their intervals before embedding LayerNorm, and preserve the resulting `Potential` through the internal encoder API.
- [x] Remove all RoBERTa live bounds by freezing embedding and affine ranges, propagating `Potential` through the encoder and pooler, and carrying the final range into local LM and classification heads without changing public model outputs.
- [x] Remove all GPT-2 live bounds with frozen embedding and Conv1D intervals, an analytic model-entry range, analytic MLP activation ranges, residual endpoint addition, and two-per-block calibration bindings.
- [x] Add GPT-2 collection, frozen-validation, and inference evaluator modes using filtered WikiText training subsets, fixed tokenizer/sequence metadata, sequential two-pass replay, and per-site clipping reports.
- [x] Use operator interval arithmetic where it provides a practical bound; retain calibration at selected boundaries whose finite ranges become excessively wide.
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
- [x] Use the implemented LayerNorm normalization bounds that do not accumulate growth of input ranges across layers, then derive the output interval from scale and bias; a tighter output calibration remains a separate extension.
- [x] Support residual collection for each ViT/GPT-2 block and, when frozen calibration is enabled, count values outside the interval and clamp to the persisted interval.
- [x] Connect both ViT pre-norm residual boundaries to optional explicit calibration bindings while retaining batch-independent analytic interval addition when calibration is absent.
- [x] Support one measured symmetric score range per ViT/GPT-2 attention layer, subject to the analytic representability ceiling; without calibration, use the fixed score limit and retain separate Gaussian validation of value readout.
- [x] Replace the attention-specific `tau_m` and `tau_s` names with one `tau`; model adapters derive it from their shared `tau_s` configuration, and the ViT-only `tau_m` field is removed.
- [x] Support optional layer-wise calibration of affine inputs to composed GELU, retain an analytic final activation interval, and remove avoidable deterministic exponential overflow without changing the operator equation.
- [x] Keep spike-time windows configuration-derived: LayerNorm log windows remain fixed by `clip_margin`, `theta`, and `tau_s`, while affine identity encoding uses each declared zero-containing fixed interval.
- [x] Make declared potential and time bounds immutable so cached or propagated endpoints cannot be widened in place.
- [x] Keep masked attention scores inside the declared softmin range and clamp both Gaussian and noise-free value readouts to a rail derived from fixed $S_{\max}$ and $\theta$.
- [x] Attach that shared fixed attention-output range to `Potential` in the ViT, BERT, RoBERTa, and GPT-2 adapters instead of reusing the value range or measuring output extrema.
- [x] Remove live activation extrema from the Gaussian `SpikingLayerNorm` path by propagating operator intervals and using the finite-feature dense LayerNorm bound.
- [x] Remove live activation extrema from deterministic `SpikingLayerNorm.forward` with the same operator intervals and finite-feature dense bound.
- [x] Remove live output extrema from ordinary `nn.LayerNorm` calls in `_apply_norm` with the finite-feature bound and learned affine endpoint propagation.
- [x] Verify bounds are identical across batch contents, ordering, and batch size, and add a final source audit that rejects runtime tensor-extrema domain construction in maintained paths.


## LayerNorm Upper Endpoint Proposal

사용자 승인에 따라 상한에서만 `clip_margin`을 빼던 정의를 코드에 수정했다. 양의 하한과 기존 실험 기록은 유지한다. 아래 비교는 변경 전후를 기록하며, 추가 실험이나 원고 변경은 수행하지 않는다.

### Direct Changes

공유 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]]의 입력 범위 정의만 바꾼다. 아래에서 $m$은 현재 `clip_margin` 값이며 기본값은 계속 $10^{-5}$다.

| 변경 대상 | 변경 전 코드 | 적용 코드 |
| --- | --- | --- |
| 실제 양/음 magnitude 범위 | `PotentialBounds(0.0, theta - clip_margin)` | `PotentialBounds(0.0, theta)` |
| log 계산용 범위 | `PotentialBounds(clip_margin, theta - clip_margin)` | `PotentialBounds(clip_margin, theta)` |
| margin 유효성 검사 | `margin >= theta / 2.0`이면 거부 | `margin >= theta`이면 거부 |

앞의 두 범위는 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#_gaussian_forward]]와 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#forward]]에 각각 있어 총 네 곳이다. 검사도 constructor와 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#freeze_parameter_bounds]]의 두 곳에서 일치시킨다. Constructor의 변수명은 `normalized_margin`이며, 기존 finite/positive 검사는 유지한다.

문서 문자열, 주석, 오류 메시지의 "양 끝점을 안쪽으로 이동"과 `theta/2` 설명을 "양의 log 입력 하한" 및 `0 < clip_margin < theta`로 바꾼다. 호출부와 설정 파일의 `clip_margin` 이름은 이번 범위에서 바꾸지 않는다.

### Derived Bounds

분산 범위와 시간창은 log 입력 구간의 끝점에서 이미 계산되므로, 별도 상수를 추가하거나 파생 수식을 중복 수정하지 않는다.

| 파생값 | 변경 전 | 적용 후 |
| --- | --- | --- |
| 분산 인코딩 범위 | $[m^2,(\theta - m)^2]$ | $[m^2,\theta^2]$ |
| magnitude 인코딩 시간창 길이 | $\tau_s\log((\theta - m)/m)$ | $\tau_s\log(\theta/m)$ |

기존 `domain_var = PotentialBounds(domain_err.min ** 2, domain_err.max ** 2)`와 `T0 = tau_s * math.log(domain_err.max / domain_err.min)`는 그대로 둔다. 분산 인코더의 시간상수 $\tau_s/2$도 유지하면 제곱된 구간으로부터 같은 시간창과 log 기준값이 나온다. 직접 로그를 계산하는 ablation 역시 `domain_err.max`와 그 제곱을 사용하므로 새 상한을 자동으로 따른다.

### Unchanged Behavior

하한의 의미, signed 값의 처리와 LayerNorm의 최종 출력 제한은 이번 변경 대상이 아니다.

- `clip_margin=0`으로 바꾸지 않는다. 실제 magnitude에는 0을 허용하되 log 계산용 값에만 양의 하한을 적용한다.
- `positive_active`와 `negative_active`의 하한 판정, 비활성 경로의 출력 기여 제거, 분산 계산에 쓰는 0 magnitude를 유지한다.
- 분산에 더하는 `eps`, 시간상수, 가중치와 bias, normalized 및 최종 affine 출력 bound를 유지한다.
- [[utils/transforms/potential_to_spike.py#neg_log_transform]], [[utils/transforms/spike_to_potential.py#exponential_difference_operator]], 일반 곱셈의 정의는 바꾸지 않는다.
- GELU의 magnitude 하한, 전역 theta 선택 절차, calibration 수집 위치와 표본 선택은 바꾸지 않는다.
- 공통 LayerNorm 클래스 변경이므로 ViT만의 변경으로 설명하지 않는다. 이 클래스를 사용하는 다른 모델과 ablation에도 적용된다.

### Verification Changes

기존 검사의 하드코딩된 내부 범위를 갱신하고, 새로 허용되는 상한을 실제로 밟는 경계 검사를 보강한다. 테스트가 기존 상한보다 작은 값만 사용하면 변경을 검증하지 못한다.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_layernorm]]의 `PotentialBounds(0.0, 3.9)`를 `PotentialBounds(0.0, 4.0)`, `PotentialBounds(0.1, 3.9)`를 `PotentialBounds(0.1, 4.0)`로 바꾼다. 나머지 파생 variance와 deadline은 같은 계산식을 유지한다. Event 수와 miss 수 기대값을 새 결과에 맞춰 무작정 고치지 않는다.

- theta=4에서 평균이 0인 `[-4,-1,1,4]` 및 상한 초과 입력으로 magnitude/log 상한이 4인지 확인한다. 모든 원소가 같은 비율로 잘려 normalization에서 차이가 상쇄되는 입력만 사용하지 않는다.
- 0, 양의 하한 미만, 하한과 같은 입력에서 log 계산용 값과 비활성 경로를 구분한다. 상수 입력의 출력이 bias가 되는 기존 검사를 보존한다.
- Constructor와 bounds freeze가 theta=4에서 margin=2 또는 3을 허용하고, 0 이하, 4 이상, NaN/Inf는 거부하는지 확인한다.
- float32/float64, 세 LayerNorm ablation flag의 8개 조합, 노이즈를 끈 경로와 표준편차가 0인 Gaussian 경로의 일치를 검사한다. 노이즈가 있는 경로의 유한성, 고정 bounds, event/miss 처리는 별도로 검사한다.
- [[scripts/verification/verify_layernorm_affine_bounds.py#verify_paired_bounds_and_parity]]의 최종 출력 범위 기대값은 log 상한에서 나온 값이 아니므로 유지한다. [[scripts/verification/verify_layernorm_affine_bounds.py#verify_cache_and_single_feature]]의 캐시 검사를 보존하고 margin 변경 후 refresh 조건을 확인한다.

구현 후에는 위 두 검증 파일과 `verify_calibration.py`를 실행한다. 공통 연산자나 연산 수 정의는 바꾸지 않으므로 이번 계획만으로 새로운 연산 수 모델을 도입하지 않는다. 모델 전체 성능 및 노이즈 결과의 동일성은 이 단위 검사로 주장하지 않는다.

### Artifact And Manuscript Changes

같은 theta와 clip_margin 숫자라도 상한의 의미가 달라지므로 구버전 calibration 표를 새 구현에서 조용히 재사용하지 않도록 한다.

ViT/GPT-2 metadata의 [[utils/transforms/functions.py#OUTPUT_BOUNDS_VERSION]]을 2에서 3으로 올렸다. 파일 구조를 바꾸는 것이 아니므로 calibration의 `format_version`은 유지한다. 구버전 metadata 거부와 새 버전의 저장/복원 검증은 [[calibration#Layer-wise Calibration#Frozen Execution#LayerNorm Positive Input Range]]에서 관리한다.

진행 중인 별도 실행기는 source commit도 metadata에 넣지만, 기본 모델별 수집 경로 전체가 이를 보장하는 것은 아니다. 기존 고정 source, calibration 표, 결과 파일은 수정하지 않는다. 새 코드로 평가하기로 한 경우에만 별도 source와 결과 경로를 사용하고, 필요한 calibration 표를 다시 수집한다. 이 계획은 실행 중인 작업을 중단하거나 재시작하라는 지시가 아니다.

실제로 새 정의를 적용하고 그 설정으로 얻은 결과를 보고할 때만 ICLR 실험 문단의 상한을 바꾼다. 기존 source 648af9bb 결과를 설명하는 문장은 이전 상한을 유지한다. 구현 변경 시 [[domain#Signed Values and Dual Rails]], [[domain#Scale Parameters]], [[bounds-audit#Fixed Range의 수식 계약#Layer Normalization]] 및 관련 calibration 설명도 갱신하되 과거 결과의 정의를 소급 수정하지 않는다.

### Verification Already Performed

변경 가능성을 확인하기 위해 source를 고치지 않고 새 구간을 기본 연산자에 직접 전달한 작은 CPU 검사만 수행했다.

theta=40과 양의 하한 $10^{-5}$에서 float32/float64 모두 상한의 log 시각이 0이고, magnitude와 variance의 시간창이 일치하며, log 뒤 exponential difference가 기대한 나눗셈을 복원하고 제곱 연산이 상한을 처리하는 것을 확인했다. 이는 완성된 LayerNorm 클래스 변경, 모든 Gaussian 분기, 전체 모델 정확도 또는 기존 결과와의 동일성을 검증한 것이 아니다.

### Implementation Verification

2026-09-14 코드 변경 후 CPU 검증을 통과했다. 새 정확도 실험, 기존 작업 재시작, 원고 수정 또는 UBAI 배포는 수행하지 않았다.

- [x] 실제 magnitude와 log 입력의 상한 네 곳 및 유효성 검사 두 곳을 수정했다. 파생 분산·시간창 수식과 variance의 시간상수는 유지했다.
- [x] [[scripts/verification/verify_layernorm_upper_endpoint.py#verify_upper_endpoint_and_ablations]]의 3개 검증 그룹을 통과했다. 8개 ablation, float32/float64, 시간상수 1과 0.75, 노이즈 유무 및 경계 입력을 포함한다.
- [x] float32 직접 log 계산이 고정 시간창을 반올림 오차만큼 넘는 경우를 발견해 두 직접 log 분기에서 계산 시각을 기존 시간창 안으로 제한했다. 시간창을 넓히거나 기본 연산자 정의를 바꾸지 않았다.
- [x] 기존 Gaussian 전체 검증, LayerNorm affine 4개 그룹, calibration 18개 그룹, GELU 4개 그룹 및 calibrated ViT evaluator 검증을 통과했다.
- [x] ViT/GPT-2의 version 3 표 저장·복원과 적용, version 2 표의 validation/inference 적용 거부를 확인했다. 변경된 규칙과 검증을 [[calibration#Layer-wise Calibration#Frozen Execution#LayerNorm Positive Input Range]]에 연결했다.

전체 모델의 정확도 변화는 이 검사로 주장하지 않는다. 이후 새 구현으로 평가할 때에만 별도 calibration 수집과 결과 경로를 사용한다.

## Calibrated Three Sweep Execution

2026-09-14 승인된 최신 bound 실험은 theta 선택과 두 noise 축을 각각 9점으로 평가하고 seed 전체 완료 순서로 중간 결과를 남긴다.

- [x] 이전 threshold-40 실행기와 해당 evaluator만 중단했다. 완료·부분 로그는 삭제하거나 새 결과와 합치지 않았다.
- [x] LayerNorm 상한 변경을 `c9f4e40`으로 별도 커밋하고 UBAI의 clean checkout에 동기화했다.
- [x] [[evaluation#Calibrated Three Sweep Campaign]]에 71회 평가, 9회 calibration, training 선택과 validation 분리 및 경계 중단 규칙을 정의했다.
- [x] [[evaluation#Calibrated Three Sweep Scheduling]]의 seed 0 전체 → seed 1 전체 → seed 2 전체 순서와 완료 결과 재사용을 구현했다.
- [x] 실행기와 집계기를 `36615ab`으로 별도 커밋하고 양쪽 clean checkout을 같은 commit으로 고정했다. 기존 사용자 문서 변경은 포함하지 않았다.
- [x] 새 계약 4그룹, 실행 순서 5그룹, 집계 4그룹, UBAI 안전성 19개 검증 및 관련 연산자·calibration·문서 검사를 통과했다.
- [x] Slurm 준비 작업 `984373`에서 자산·의존성 해시와 Python 3.12.13을 확인했다. 첫 준비 작업 `984371`의 경로 연결 실패 로그는 보존했다.
- [ ] CPU 검증과 Slurm 자산 검증 후 양쪽의 짧은 clean/noisy prediction 일치를 확인한다.
- [ ] 9개 theta의 training/validation 및 반대 환경 replay를 검증해 선택을 확정한다. 범위 부족·불안정이면 noise 시작 전에 보고한다.
- [ ] 17조건씩 세 seed를 진행하고 첫째·둘째 중간 그림과 최종 그림을 보존한다. seed 0 범위 중단 규칙을 적용한다.

로컬은 GPU 4–7만, UBAI는 gpu4/gpu5만 사용하며 RAM 디스크에는 환경을 풀지 않는다. 50k와 추가 축은 진행하지 않는다. 기존 시간 추정은 약 38 GPU-hours이며 구현·검증·대기 시간을 포함하지 않는다.

UBAI의 새 clean checkout에는 읽기 전용 source를 mount하기 전에 내부 연결 지점인 `artifacts/assets/theta-selection-v1`, 해당 실험의 `artifacts/logs/noise_scan` 하위 디렉터리, `src/transformers`, `src/spikingjelly`를 빈 디렉터리로 준비해야 한다. 이 경로 준비와 실패·재시도 이력은 실험의 `deployment-notes.json`에 기록했다.

## ViT Calibration Policy 2 Shared Deadline

공통 시간창 전달은 구현하고 CPU 회귀 검증을 통과했다. 새 source의 실제 ViT-B 재검사는 아직 필요하며, 기존 실패 기록이나 calibration을 덮어쓰지 않는다.

- [x] LayerNorm의 세 log 인코딩에 동일하게 계산한 공통 시간창을 sampling 전에 적용했다. 이후 event의 domain만 바꾸는 처리는 하지 않는다.
- [x] 일반 primitive의 deadline 검사를 유지했다. 상한 40.007과 반대 방향 반올림 사례, 8개 ablation, noise-off·Gaussian 표준편차 0·seeded 경로의 회귀 검증을 추가했다.
- [ ] 경고는 전역 theta 초과와 선택된 범위의 실제 clipping을 구분한다. 수정은 새 source와 별도 실패 기록을 유지한 재검사로 검증한다.
