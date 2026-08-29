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

## Numerical Precision and Endpoint Caveat

Timing-noise results are interpretable only when the sampling dtype resolves the requested perturbation and nominal codewords are assessed for endpoint placement.

[[utils/transforms/noise.py#_sample_gaussian_spike_time]] samples in the nominal time tensor's dtype. If $\sigma_t$ is smaller than that dtype's spacing near a codeword, rounding can map most nonzero draws back to the nominal value and make empirical misses disagree with the continuous Gaussian probability.

An event nominally at $T_{\mathrm{obs}}$ has miss probability $0.5$ under any zero-mean continuous Gaussian with $\sigma_t>0$, because every positive perturbation is late. Float32 can conceal this endpoint behavior when $\sigma_t$ is below one ULP; that concealment is numerical quantization, not physical robustness.

Experiments must therefore record the payload dtype, compare $\sigma_t$ with time-value spacing over every exercised code interval, and report whether nominal events occupy the deadline. Sub-ULP sweeps and endpoint-heavy encodings are diagnostic only until empirical sampling agrees with [[utils/transforms/noise.py#gaussian_deadline_miss_probability]].

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

[[utils/transforms/noise.py#install_device_mismatch]] samples one frozen potential-offset proxy per supported spiking module and installs it with forward pre-hooks. It is not Stanojevic-style neuron-slope perturbation and should not be reported as such.

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

The next experimental order is:

1. run a short ViT smoke evaluation with zero timing noise and confirm checkpoint-level parity;
2. sweep small $\sigma_t$ values while recording per-site miss rates;
3. inspect per-site output saturation before selecting the manuscript's operating range;
4. repeat selected settings across seeds before any hardware comparison.

Every stage keeps the noise-free tensor path as a parity reference. No stage may introduce `gaussian_multiplication_operator`, an operator-specific sampler, or invalid-result propagation.

## Gaussian Noise Statistics

Maintained-noise experiments expose per-site event delivery and readout saturation counters so robustness results can be attributed to physical failure modes.

The statistics interface reports event count, deadline misses, output count, and lower/upper rail saturation for each named site. Reconfiguring the Gaussian generator starts a new replica and clears these counters; callers can also clear them explicitly.

Output saturation is counted from the raw physical readout before its required rail clamp. Both denominators must be reported: event rates use event count, while saturation rates use output count.
