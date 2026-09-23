# TODO

This file tracks work required for the current manuscript and maintained implementation. Optional experiment ideas are kept separately in [[deferred-experiments]] without checkboxes. Fully completed work units are archived in [[deprecated#완료된 TODO 기록]].

## Active Experiment Work

Only the explicitly listed evaluations are authorized. Unchecked items elsewhere are manuscript or packaging work unless they define a separate approved run.

The completed GPT-2 composed GELU rerun and shared Power cubic text rerun are archived in [[deprecated#완료된 TODO 기록]]; do not restart them from that record.

### Local-Range Re-evaluation

The active campaign replaces every manuscript result produced with a global range setting, excluding the separate discrete-time simulation requested to remain unchanged.

- [x] Remove global `theta` and GPT-2 `attention_theta` from operators, model configs, evaluators, calibration metadata, and maintained experiment wrappers.
- [x] Reject legacy config keys and add a production-source verifier that blocks later reintroduction.
- [x] Define timing-noise standard deviation and deadline margin relative to each encoder's declared local time window.
- [ ] Recollect schema-2 calibration and rerun Table 3 ViT rows on their complete declared evaluation populations.
- [ ] Recollect schema-2 calibration and rerun Table 4 RoBERTa-B/L and GPT-2 rows on their complete declared evaluation populations.
- [ ] Recollect ViT-B calibration and rerun Figure 4 timing-noise and deadline-margin sweeps with three replicas.
- [ ] Update manuscript numbers and Figure 4 only from verified new artifacts; keep old global-range results as superseded provenance.

### Historical ViT-B Timing Robustness

This completed global-range campaign is historical after local-range migration and must not be restarted as current evidence.

- [x] Complete policy-2 calibration for all nine threshold candidates, select $\theta=20$ from training 5k, and confirm the replay and neighboring validation candidates.
- [x] Complete the nine-point timing-noise sweep at deadline margin/noise standard deviation ratio 4 and the nine-point ratio sweep at $r_t=10^{-5}$.
- [x] Complete the four-point high-scale extension through $r_t=10^{-3}$ and preserve its source and calibration identity with the base campaign.
- [x] Keep threshold-40 uncalibrated campaigns, the 12 by 13 joint grid, and $\theta=2000$ artifacts as separate provenance rather than statistical replicas.
- [ ] Decide whether the fixed-5k computational stress test belongs in the main text or appendix; do not describe it as full ImageNet-1k validation or calibrated hardware evidence.

### Historical ViT Training-Only Threshold Comparison

This completed selection campaign is historical after global-range removal and cannot update the manuscript table.

- [x] Freeze the eleven-candidate grid and select each threshold only from training seed-0 5k accuracy.
- [x] Preserve the failed UBAI identity checks and reassign only terminal, non-duplicated work to local GPU devices 4--7.
- [x] Complete and validate CIFAR-10 ViT-S, ImageNet ViT-S, and ImageNet ViT-B.
- [x] Complete ImageNet ViT-L, regenerate the four-row summary, archive provenance, and update the manuscript only from the complete bundle.

## Manuscript Revision Master Checklist

This is the canonical status ledger for manuscript-facing work; the former review checklist remains a provenance record, and duplicate actions are consolidated here.

Source details and reviewer labels are consolidated in [[deprecated#과거 리뷰]]. Update task status here first, then reflect completed claims in `paper/neurips_2026/neurips_2026.tex` and the submission checklist.

### Current ViT-B Noise and Time-Scale Update

This checklist covers reporting from the active local-range 5k campaign; removed global-range campaigns remain historical evidence only.

- [x] Remove the global range selection phase and define timing-noise scale relative to each encoder's declared local time window.
- [ ] Complete the fresh ViT-B calibration and deterministic ANN/SNN evaluation on the fixed validation 5k subset.
- [ ] Complete the nine-point timing-noise fraction and thirteen-point deadline-margin ratio sweeps with seeds 0, 1, and 2.
- [ ] State that the frozen calibration covers the complete active site set and remains unchanged during validation and noise evaluation.
- [ ] Explain how each dimensionless local code interval maps to a declared physical duration and state that all associated time constants must be rescaled consistently.
- [ ] Report the injection scope, seeds, confidence interval and empirical miss statistics from the accepted local-range logs.
- [ ] Mark every earlier shared-range and threshold-selection result as superseded for manuscript support.

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
- [x] Replace “quantifying the resulting clipping error” with “accounting for clipping introduced by the finite ranges”; do not add a separate layer-wise activation/logit error study solely to support clipping-error quantification.
- [ ] Map each current paragraph, theorem, table, and figure to keep, shorten, move to appendix, delete, or rewrite before polishing the abstract and introduction.

### P0 Mathematical and Operator Audit

Every retained theorem and operator claim must match the implemented equations, domains, finite-window behavior, and failure conditions.

- [ ] Reclassify Theorem 1 as a quantitative theorem with conditions and bounds, a conditional theorem, or a constructive proposition.
- [ ] Tabulate each primitive and composition with input domain, clipping condition, scale, and zero-error condition; keep quantitative claims within the reported ANN--SNN task fidelity rather than introducing a separate layer-wise clipping-error study.
- [ ] Derive an $L$-block error relation where defensible; otherwise state why no useful end-to-end bound is available and limit the claim to empirical evidence.
- [ ] Present all four signed multiplication sign cases and the causal source--sink routing used by the two parallel causal integration paths.
- [ ] Define $\psi_{\mathrm{Int}}$ as a time-window integration mechanism based on an NMDA plateau rather than equating it with a validated biological spike or completed circuit.
- [ ] Update the signed multiplication definition, proof, and SOP accounting for two parallel causal paths without duplicating encoder spikes.
- [ ] Add the complete Softmax composition and recheck stability, exponent scaling, normalization, lower and upper clipping, deadline behavior, and the meaning and calibration of $\alpha$ against the implementation.
- [ ] Re-derive the GELU relation involving $1/(1+\kappa e^{-\beta x})$, state the required $\kappa$ and time-constant conditions, and quantify finite-window error.
- [ ] Add the complete LayerNorm dual-rail derivation, state that it uses unrestricted $\psi_{\mathrm{ED}}$ rather than public $f_{\mathrm{Div}}$, and distinguish the variance stabilizer from the positive log input floor.
- [ ] Unify the LayerNorm references as $H$ and $H^2$, including the finite-domain definition used in code.
- [ ] Resolve any residual normalization-scale explanation against the current declared local-range equations.
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
- [ ] Report GPT-2 token-weighted corpus perplexity as the primary metric and name the mean-of-batch-loss value only as a compatibility metric.

### P0 Energy, Latency, and Hardware Claims

Energy and feasibility claims must expose their system boundary and remain proxies unless supported by hardware-level evidence.

- [ ] Cite the source and conditions for $E_{\mathrm{AC}}=0.9$ pJ, including process, voltage, circuit type, and whether it is measured or estimated.
- [ ] Label the present calculation as an idealized SOP-only estimate and list omitted memory, routing, fanout, synchronization, static current, comparator, calibration, conversion, mismatch, and integration-time costs.
- [ ] Define the target substrate and a consistent system boundary before comparing against ANN or neuromorphic baselines.
- [ ] Estimate per-operator and end-to-end latency, including the selected code window and the physical-time rescaling assumption.
- [ ] Do not infer energy superiority from operation-count equality; use a comparable ANN energy boundary or narrow the claim.
- [ ] Audit external energy numbers such as SpikeZIP's 100.8 mJ for boundary compatibility before making direct comparisons.
- [ ] Keep hardware-level superiority outside the claim because SPICE, FPGA, device-level, and silicon evidence are absent.
- [ ] EBRAINS 계정에서 BrainScaleS-2의 여섯 `INA219StatusOnBoard` 전원 레일을 읽을 수 있는지 시험하고, shunt calibration과 conversion setting을 확인한 뒤에만 calibrated measurement로 해석한다.
- [ ] 하나의 고정 mapping에서 ready 실행과 반복 active 실행을 교대로 측정하고, hardware throughput과 crossbar event count를 기록하여 total 및 dynamic energy, 95% interval과 power와 event rate 사이 기울기를 함께 보고한다.

### P1 Robustness and Non-Ideality Evidence

Computational stress tests must be separated from calibrated device models and from one another.

- [ ] Document each timing-noise distribution, magnitude, injection site, seed, repetition count, and confidence interval.
- [ ] Treat additive jitter with deadline misses as the maintained computational model; do not describe it as a calibrated neuronal noise process.
- [ ] Report accuracy against empirical miss rate and the available counts by operator site as simulator diagnostics only; do not interpret them as a physical event population or energy estimate.
- [ ] State whether timing error is injected at encoder outputs or at every internal $\Phi/\Psi$ boundary, and describe the latter coverage explicitly.
- [ ] State that frozen threshold mismatch and every other uncertainty axis in [[deferred-experiments#Additional Robustness Axes]] are outside the current result.
- [ ] Package the clean baseline, checkpoint, evaluator, manifests, and current jitter evidence before requesting analog hardware collaboration.
- [ ] Reorder Section 5 as noise-free conversion fidelity, primitive operator measurements on BrainScaleS-2, and model robustness under the measured distributions with one deadline margin sweep.
- [ ] Keep the Gaussian timing-noise sweep in the main text only if measured BrainScaleS-2 conditions are located on the same sweep; otherwise move it to the appendix as a computational stress test.

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
5. Finalize the local-window and physical-time interpretation, then run only the approved robustness diagnostics.
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
- [x] Migrate deterministic exponential difference to signed PWM with the same shared-deadline requirement as its event-driven path.
- [x] Replace explicit deterministic affine and attention synapse tensors with optimized kernels that evaluate the same signed PWM reductions.
- [x] Remove the algebraic single-rail PWM implementation and public export after all maintained callers migrate.
- [ ] After the planned manuscript rewrite, update its definition, proof, and SOP accounting for two parallel causal integration paths without duplicate encoder spikes. This documentation-only task is intentionally deferred and does not block the maintained implementation.

## Static Bounds for All Operators

Every maintained operator must use bounds fixed before inference; a forward pass must never define its own physical rails from values it has already produced.

The completed source audit, formulas, model-family inventory, and all execution cases are documented in [[bounds-audit]].

The acceptance criteria, bound re-audit follow-up, and 80-item implementation checklist are all complete and archived in [[deprecated#완료된 TODO 기록#Static Bounds 구현 체크리스트]]. The design rationale and runtime contract below remain current.

### Why Static Bounds Are Required

Static bounds turn domains into predeclared physical and mathematical contracts instead of batch-specific observations.

- Physical TTFS rails and observation windows must be configured before an input is encoded. Deriving them from the completed output is an unavailable runtime oracle.
- The same activation must receive the same domain and encoding regardless of batch contents, ordering, batch size, device partitioning, or noise seed.
- A tensor's observed minimum and maximum describe only that batch; they do not conservatively bound future inputs and therefore cannot satisfy the `Potential` contract.
- Widening bounds around a noisy output hides physical underflow and overflow. Raw outputs must be compared with fixed bounds before statistics are recorded and clamping is applied.
- Immutable bounds keep deterministic clipping error, operator approximation, Gaussian timing error, deadline misses, and output saturation independently measurable.

### Intended Runtime Contract

Calibration and interval arithmetic establish an immutable bound table before evaluation, after which forward execution may only consume, propagate, compare, and clamp against those bounds.

The objective is not to replace every runtime range with the widest possible analytic interval. The three cases in [[domain#Domain Propagation]] retain practical structural bounds, use calibration at selected finite but widening boundaries, and restrict domains with no finite output or timing bound. The implementation also supports disabled calibration with conservative analytic propagation; this mode does not exercise the intended layer-wise limits.

This range reset is necessary even when every individual operation has a finite formula. For a layer with Lipschitz constant $L_i>1$, $\lVert\delta x_{i+1}\rVert\le L_i\lVert\delta x_i\rVert+\lVert e_i\rVert$, so propagated intervals and upstream clipping error can grow with depth. Fixed layer boundaries deliberately clamp that growth, and validation must report both layer clipping rates and final task accuracy.

The required order is: load or calibrate static input envelopes, derive conservative operator outputs, evaluate the raw tensor, record excursions against the fixed output bound, clamp, and pass the unchanged declared envelope downstream. Neither clean nor noisy execution may widen a bound.

Calibration runs with timing noise disabled and is identified by stable operator sites. A checkpoint change, preprocessing change, model-family change, or ablation-path change invalidates the affected calibration and requires rebuilding it before evaluation. Static parameter perturbation and threshold mismatch are robustness axes applied only after clean-artifact compatibility succeeds; they do not become calibration-table identities. Parameter-derived affine safety bounds are frozen or refreshed after any static parameter perturbation.

Calibration uncertainty is an engineering tolerance rather than a reason to restore runtime extrema. Each site should store signed lower and upper bounds obtained from representative extrema or quantiles, enlarge them by a documented margin, and report calibration-set and evaluation-set clipping rates. The margin may cover moderate distribution variation, but it must not conceal a domain-propagation error, an invalid operator condition, an attention-mask value outside its declared range, or a Gaussian deadline-miss case. Those cases require analytic interval bounds or a direct implementation fix, and inference must never widen a calibrated bound after observing an activation.

Calibration measurement uses two deterministic collection passes before frozen validation. The first pass records signed min/max values; the second replays the same dataset into fixed-bin histograms whose edges come from the first pass. The histogram and margin determine the immutable range table, after which validation applies inference clamps and reports clipping without updating any range.

Selected calibration sites use signed-symmetric, lower-bounded, or upper-bounded endpoint policies. These are distinct from the three reasons for selecting a range in [[domain#Domain Propagation]]. Practical structural bounds remain analytic, but a finite interval that grows excessively can still require calibration.

For nonnegative sites selected for one-sided calibration, the lower endpoint stays at zero and only the upper endpoint is calibrated. Logarithmic domains keep a separately configured positive lower endpoint; current ViT/GPT-2 bindings do not select that endpoint from data.
