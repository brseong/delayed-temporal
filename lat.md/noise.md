# Noise Model

The maintained dynamic model adds Gaussian error directly to TTFS spike times and derives deadline misses from the same sampled event.

## Configuration

Direct timing noise uses one process-wide configuration and a dedicated random generator seeded once per experiment replica.

The process-wide configuration stores the absolute time mean and standard deviation, seed, and generator, and validates them before installation.

The generator advances across forward calls. Reconfiguring it restarts a replica; an individual forward must not reseed it. Because this state is process-wide, the maintained path rejects `DataParallel` execution.

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

Jitter and deadline miss are therefore not independent random channels. No escape-rate ODE, first-passage solver, or separately sampled dropout is part of this maintained model.

## Fixed Observation Deadline

Each event-aware encoder call uses the nominal end of the code interval as its physical observation deadline.

The maintained model fixes

$$
T_{\mathrm{obs}}=T_{\mathrm{code}}.
$$

Any sampled event later than this shared endpoint is a deadline miss. The model does not extend the observation window beyond the nominal latest codeword.

## Observation-Time Potential Invariant

A spike miss never invalidates an operator output; every operator reads its physical potential at the observation deadline and passes that finite value onward.

The maintained rule is

$$
V_{\mathrm{out}}
=\operatorname{clamp}\!\left(V(T_{\mathrm{obs}})\right).
$$

`fired` selects the physical state evolution before readout. It is event metadata, not an output-validity flag. The simulator must not propagate an `invalid` result, abort the operator chain, or replace the readout with an arbitrary fallback.

For event-gated integration with reset potential $V_{\mathrm{reset}}$, opening event $t_{\mathrm{open}}$, closing/reference event $t_{\mathrm{close}}$, and drive $I$, the fixed cases are:

$$
V(T_{\mathrm{obs}})=
\begin{cases}
V_{\mathrm{reset}},
& \text{opening spike misses},\\
V_{\mathrm{reset}}+I\left(T_{\mathrm{obs}}-t_{\mathrm{open}}\right),
& \text{opening arrives and closing spike misses},\\
V_{\mathrm{reset}}+I\left(t_{\mathrm{close}}-t_{\mathrm{open}}\right),
& \text{both spikes arrive}.
\end{cases}
$$

Thus an opening miss gives zero synaptic contribution relative to reset, whereas a closing/reference miss continues integration until maximum time. Both cases still produce a valid finite potential after physical rail clamping, so subsequent operators continue normally.

## Encoder Injection Boundary

The existing potential-to-spike decorator is the only production injection point for the direct Gaussian model.

[[utils/transforms/noise.py#inject_spike_time_noise]] first calls the deterministic encoder and then samples timing noise when the consumer requests `return_spike_sample=True`. Both [[utils/transforms/potential_to_spike.py#neg_linear_transform]] and [[utils/transforms/potential_to_spike.py#neg_log_transform]] carry this decorator.

There is no separate Gaussian multiplication operator and no encoder-specific Gaussian helper. Event-aware consumers receive a time-and-delivery record, while tensor-only callers preserve the historical `(time, bounds)` interface.

## Layer-Shared Reference Event

An affine layer treats its zero-reference timing signal as a physical spike shared by the whole layer call.

[[utils/transformers/models/spiking_ops.py#SpikingLinear#forward]] requests both the data event and one scalar zero-reference event through the same decorator. Data events receive independent timing samples; the scalar reference sample is broadcast across the layer operation.

If a data event is absent, its contribution remains at the reset value. If the reference event is absent, integration continues to the observation deadline. These are direct applications of [[noise#Observation-Time Potential Invariant]], not operator-specific fallback policies.

## Legacy Dynamic Compatibility

The pre-existing potential-referred jitter and independent drop/insertion implementation remains temporarily available only for reproducing older experiments.

[[utils/transforms/noise.py#NoiseConfig]] and [[utils/transforms/noise.py#set_spike_time_noise]] configure that historical path. Direct Gaussian noise and legacy dynamic noise are mutually exclusive, because combining them would perturb one encoder twice and destroy the single-sample interpretation.

[[utils/transforms/noise.py#_apply_escape_hazard]] is not evidence for the maintained Gaussian model. Its independently sampled drop/insertion events and the linked-beta helper should be treated as legacy stress tests.

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

Legacy module-scoping hooks continue to mutate the old global configuration and do not scope the maintained Gaussian generator.

## Interpretation Limits

The implementation is a controlled computational robustness model rather than calibrated circuit validation.

It models independent Gaussian event-time errors, deadline misses, layer- or operation-shared references, and event-aware coverage of maintained operators. It does not model temporal correlation, routing faults, temperature dependence, or calibrated BrainScaleS-2 parameters.

Experiments must report $\mu_t$, $\sigma_t$, seed or repeats, injection coverage, and observed miss rate. Deterministic conversion accuracy and static parameter perturbation remain separate evidence axes.

## Current Coverage and Resume Order

The planned event-aware migration is complete; work now resumes from model-level smoke evaluation and timing-noise calibration rather than another operator rewrite.

Affine, multiplication, exponential, exponential-difference/division, activation, softmin, and attention value paths use decorated events and retain noise-off parity references. Verification exercises opening, closing/reference, and internal exp-temporal cases.

The next experimental order is:

1. run a short ViT smoke evaluation with zero timing noise and confirm checkpoint-level parity;
2. sweep small $\sigma_t$ values while recording per-site miss rates;
3. inspect per-site output saturation before selecting the manuscript's operating range;
4. repeat selected settings across seeds before any hardware comparison.

Every stage keeps the noise-free tensor path as a parity reference. No stage may introduce `gaussian_multiplication_operator`, a direct encoder sampler, or invalid-result propagation.

## Gaussian Noise Statistics

Maintained-noise experiments expose per-site event delivery and readout saturation counters so robustness results can be attributed to physical failure modes.

The statistics interface reports event count, deadline misses, output count, and lower/upper rail saturation for each named site. Reconfiguring the Gaussian generator starts a new replica and clears these counters; callers can also clear them explicitly.

Output saturation is counted from the raw physical readout before its required rail clamp. Both denominators must be reported: event rates use event count, while saturation rates use output count.
