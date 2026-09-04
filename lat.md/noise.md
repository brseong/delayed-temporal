# Noise Model

The maintained noise model adds Gaussian error directly to TTFS spike times and derives deadline misses from the same sampled event.

## Configuration

Direct timing noise uses one process-wide configuration and a dedicated random generator seeded once per experiment replica.

The process-wide configuration stores the absolute time mean and standard deviation, seed, and generator, and validates them before installation.

The generator advances across forward calls. Reconfiguring it restarts a replica; an individual forward must not reseed it. Because this state is process-wide, the maintained path rejects `DataParallel` execution.

Evaluation entry points expose a dimensionless standard-deviation fraction $r_t$ and convert it once using the base identity-code window, $\sigma_t=r_t(2\theta)$. Every encoder in that run then receives the same absolute $\sigma_t$.

## Direct Gaussian Spike-Time Noise

One Gaussian timing sample jointly determines the delivered spike time and whether the event misses the observation deadline.

For nominal encoder output $t_0$, the model draws

$$
\tilde{t}=t_0+\mu_t+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,1).
$$

The event is delivered exactly when

$$
\tilde{t}\le T_{\mathrm{obs}}.
$$

Early events are stored at the operation start. A late event stores the deadline only as a finite carrier value and sets `fired=False`; consumers must inspect the mask instead of substituting that placeholder into ordinary spike-time arithmetic. The model also defines the matching analytic tail probability.

The same Gaussian draw determines both the stored event time and whether the event misses its deadline; no independent event-dropout channel is sampled.

## Fixed Observation Deadline

Each event-aware encoder call uses the nominal end of the code interval as its physical observation deadline.

The maintained model fixes

$$
T_{\mathrm{obs}}=T_{\mathrm{code}}.
$$

Any sampled event later than this shared endpoint is a deadline miss. The model does not extend the observation window beyond the nominal latest codeword.

The diagnostic deadline-margin sweep may add a non-negative receiver grace $m=k\sigma_t$ after $T_{\mathrm{code}}$. An event inside the grace interval is delivered but its timestamp saturates at the original upper code rail, so bounds and clean operator arithmetic do not change. This is a late-arrival tolerance diagnostic, not a calibrated hardware window.

## Numerical Precision and Endpoint Caveat

Timing-noise results are interpretable only when the sampling dtype resolves the requested perturbation and nominal codewords are assessed for endpoint placement.

[[utils/transforms/noise.py#_sample_gaussian_spike_time]] samples in the nominal time tensor's dtype. If $\sigma_t$ is smaller than that dtype's spacing near a codeword, rounding can map most nonzero draws back to the nominal value and make empirical misses disagree with the continuous Gaussian probability.

An event nominally at $T_{\mathrm{obs}}$ has miss probability $0.5$ under any zero-mean continuous Gaussian with $\sigma_t>0$, because every positive perturbation is late. Float32 can conceal this endpoint behavior when $\sigma_t$ is below one ULP; that concealment is numerical quantization, not physical robustness.

Experiments must therefore record the payload dtype, compare $\sigma_t$ with time-value spacing over every exercised code interval, and report whether nominal events occupy the deadline. Sub-ULP sweeps and endpoint-heavy encodings are diagnostic only until empirical sampling agrees with [[utils/transforms/noise.py#gaussian_deadline_miss_probability]].

Each instrumented site records nominal deadline occupancy and the smallest and largest deadline ULP encountered during evaluation. The ViT evaluator reports both this range and $\sigma_t/\mathrm{ULP}$ so precision-limited conditions remain identifiable in saved logs and CSV summaries.

## Observation-Time Potential Invariant

A spike miss never invalidates an operator output; every operator reads its physical potential at the observation deadline and passes that finite value onward.

The maintained rule is

$$
V_{\mathrm{out}}
=\operatorname{clamp}\!\left(V(T_{\mathrm{obs}})\right).
$$

`fired` selects the physical state evolution before readout. It is event metadata, not an output-validity flag. The simulator must not propagate an `invalid` result, abort the operator chain, or replace the readout with an arbitrary fallback.

For signed PWM with reset potential $V_{\mathrm{reset}}$, event times $t_A,t_B$, delivery indicators $f_A,f_B\in\{0,1\}$, and drive $I$, define the two causal pulse widths

$$
d_A=f_A(T_{\mathrm{obs}}-t_A),
\qquad
d_B=f_B(T_{\mathrm{obs}}-t_B).
$$

The differential observation-time state is

$$
V(T_{\mathrm{obs}})=V_{\mathrm{reset}}+I(d_A-d_B).
$$

Thus two delivered events give $V_{\mathrm{reset}}+I(t_B-t_A)$, an $A$-only event gives $V_{\mathrm{reset}}+I(T_{\mathrm{obs}}-t_A)$, a $B$-only event gives $V_{\mathrm{reset}}-I(T_{\mathrm{obs}}-t_B)$, and two misses leave reset. Every case remains a finite potential after rail clamping.

A single-event operator such as the internal exponential stage has only one causal rail; its event miss therefore leaves reset zero. LayerNorm's direct exponential ablation applies the same two-rail pulse-width equation but deliberately skips the disabled exponential-difference operator and its internal event.

## Encoder Injection Boundary

The existing potential-to-spike decorator is the only production injection point for the direct Gaussian model.

[[utils/transforms/noise.py#inject_spike_time_noise]] first calls the deterministic encoder and then samples timing noise when the consumer requests `return_spike_sample=True`. Both [[utils/transforms/potential_to_spike.py#neg_linear_transform]] and [[utils/transforms/potential_to_spike.py#neg_log_transform]] carry this decorator.

There is no separate Gaussian multiplication operator and no encoder-specific Gaussian helper. Event-aware consumers receive a time-and-delivery record, while noise-free callers preserve the deterministic `(time, bounds)` interface.

## Layer-Shared Reference Event

An affine layer treats its zero-reference timing signal as a physical spike shared by the whole layer call.

[[utils/transformers/models/spiking_ops.py#SpikingLinear#forward]] requests both the data event and one scalar zero-reference event through the same decorator. Data events receive independent timing samples; the scalar reference sample is broadcast across the layer operation.

If a data event is absent, its contribution remains at the reset value. If the reference event is absent, integration continues to the observation deadline. These are direct applications of [[noise#Observation-Time Potential Invariant]], not operator-specific fallback policies.

## Static Threshold Mismatch

Static threshold mismatch remains separate from trial-to-trial timing noise.

[[utils/transforms/noise.py#install_device_mismatch]] samples one frozen potential-offset proxy per supported spiking module from a dedicated seeded generator and installs it with forward pre-hooks. The offset remains fixed within a replica, equal seeds replay the complete draw, and installation does not consume the model's global RNG stream.

This proxy is not Stanojevic-style neuron-slope perturbation and should not be reported as calibrated device mismatch. Timing noise and static mismatch remain separate experiment axes and are not enabled in the same evaluation replica.

## Static Weight and Bias Perturbation

The ViT evaluator supports one-time perturbation of loaded synaptic parameters outside the shared timing-noise module.

[[scripts/evaluation/error_analysis_vit.py#apply_parameter_noise]] applies multiplicative Gaussian weight perturbation and additive bias perturbation before evaluation. This path is static parameter uncertainty, not dynamic spike-time noise.

## Injection Scope and Compounding

Event-aware noise applies only where a consumer can interpret a delivered-event mask.

The maintained production integration covers the three affine adapters, multiplication, exponential, division, exponential difference, spiking LayerNorm, softmin and activation compositions, and attention value integration. Every noisy production event originates at the decorated encoder boundary; tensor-only branches remain only as noise-off parity references.

Missing-event semantics are already fixed by [[noise#Observation-Time Potential Invariant]]. Extending coverage means implementing each operator's ordinary physical state trajectory up to $T_{\mathrm{obs}}$ and reading the resulting clamped potential; it does not require another validity policy discussion.

## Interpretation Limits

The implementation is a controlled computational robustness model rather than calibrated circuit validation.

It models independent Gaussian event-time errors, deadline misses, layer- or operation-shared references, and event-aware coverage of maintained operators. It does not model temporal correlation, routing faults, temperature dependence, or calibrated BrainScaleS-2 parameters.

Experiments must report $\mu_t$, $\sigma_t$, seed or repeats, injection coverage, and observed miss rate. Deterministic conversion accuracy and static parameter perturbation remain separate evidence axes.

## Coverage and Experimental Order

The event-aware migration is complete across the shared sampler and encoder boundary, composed operators, model adapters, evaluation entry points, and seeded verification.

Affine, multiplication, exponential, exponential-difference/division, activation, softmin, and attention value paths use decorated events and retain noise-off parity references. Verification exercises opening, closing/reference, and internal exp-temporal cases.

Any new ViT-B/16 noise campaign must consume an `approved` `selection.json` produced by [[evaluation#ViT-B/16 Global Theta Selection]] and use its `selected_theta` as the global operating point. A $	heta=2000$ artifact remains preserved but is marked superseded and excluded from manuscript support whenever the approved value differs; no replacement robustness figure is promoted before the new-theta sweep finishes.

The maintained manuscript protocol is:

1. run a short ViT-B/16 smoke evaluation with zero timing noise and confirm checkpoint-level parity;
2. sweep Gaussian timing noise and static threshold mismatch independently on a fixed 5,000-image subset with replica seeds 0, 1, and 2;
3. record per-site deadline occupancy, ULP ranges, misses, and output saturation before interpreting the transition;
4. confirm the clean baseline and three representative points per axis on the full ImageNet validation split;
5. keep every evaluator process on one physical GPU and report both axes with 95% Student-t confidence intervals.

Every stage keeps the noise-free tensor path as a parity reference. No stage may introduce `gaussian_multiplication_operator`, an operator-specific sampler, or invalid-result propagation.

## Sigma and Deadline-Margin Grid

The follow-up ViT-B/16 diagnostic separates timing-error scale from the receiver's late-arrival grace after the global threshold has been approved.

The experiment fixes the `approved` threshold from [[evaluation#ViT-B/16 Global Theta Selection]] and defines

$$
\sigma_t=r_t(2\theta^*),\qquad k=\frac{m}{\sigma_t},\qquad m=k\sigma_t.
$$

It evaluates the existing 12-point $r_t$ grid against $k\in\{0,0.5,1,1.5,2,2.5,3,4,5,6,8,10,12\}$ on the fixed first 5,000 validation images. Every stochastic cell uses seeds 0, 1, and 2; clean spiking and dense references are deterministic singletons. Static mismatch, calibration, learned-parameter noise, and a second theta axis remain disabled.

[[scripts/experiments/ubai/build_sigma_margin_manifest.py#main]] validates the approved theta evidence and produces 470 immutable conditions bound to the data, checkpoint, source commit, and selected GPU family. UBAI jobs use one GPU each with at most eight concurrent array tasks, while resume submission includes only logs that fail the complete identity check.

[[scripts/analysis/summarize_sigma_margin_sweep.py#build_frontier]] defines recovery at each $r_t$ as the smallest preregistered $k$ whose three-seed mean is within one percentage point of the clean spiking baseline. Failure to recover at $k=12$ is retained as `unrecovered`, and later nonmonotonic cells are reported rather than removed.

The result set contains replica-level, cell-level, and site-level CSV files; a provenance JSON; a recovery-frontier JSON; and a three-panel accuracy, confidence-width, and pooled-miss-rate figure. It remains under `artifacts/` because this protocol intentionally stops at 5,000 images and does not by itself authorize manuscript promotion.

[[scripts/verification/verify_sigma_margin_sweep.py#main]] checks the approval gate, canonical grid, physical scale identities, confidence intervals, pooled counts, frontier rule, one-GPU Slurm contract, and resumable pending manifest.

## Gaussian Noise Statistics

Maintained-noise experiments expose per-site event delivery and readout saturation counters so robustness results can be attributed to physical failure modes.

The statistics interface reports event count, deadline misses, nominal deadline events, deadline ULP range, output count, and lower/upper rail saturation for each named site. Reconfiguring the Gaussian generator starts a new replica and clears these counters; callers can also clear them explicitly.

Output saturation is counted from the raw physical readout before its required rail clamp. Both denominators must be reported: event rates use event count, while saturation rates use output count.
