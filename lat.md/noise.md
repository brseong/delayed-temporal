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

The diagnostic deadline-margin sweep may allow events to arrive up to $m=k\sigma_t$ after $T_{\mathrm{code}}$, with $m\ge 0$. An event arriving during this additional interval is delivered, but its timestamp is clamped to the original upper endpoint of the encoding interval, so bounds and clean operator arithmetic do not change. This is a late-arrival tolerance diagnostic, not a calibrated hardware window.

## Comparison with Stanojevic et al.

Stanojevic et al.'s $\zeta$ and the maintained deadline margin both allocate temporal slack, but they target different scheduling failures; a generic margin is not a novel mechanism.

Stanojevic et al. set $t_{\max}^{(n)}=t_{\min}^{(n)}+(1+\zeta)X^{(n)}$, with $t_{\min}^{(n)}=t_{\max}^{(n-1)}$, using the maximum activation observed in training data to prevent the earliest output spike in layer $n$ from preceding all input spikes from layer $n-1$. This changes the nominal layer schedule and code interval. Their main construction can also force an inactive ReLU neuron to fire at $t_{\max}^{(n)}$. Their separately reported Gaussian timing perturbation experiment changes spike times, but the paper does not define a delivery mask at a receiver deadline or a grace rule after the nominal code window.

The maintained diagnostic instead keeps the nominal code interval and potential bounds fixed, samples additive timing error, and classifies delivery against $T_{\mathrm{code}}+m$. An event arriving within $m=k\sigma_t$ is delivered with its stored timestamp limited to $T_{\mathrm{code}}$. The margin targets events delayed by noise rather than output firing before the preceding layer completes. Therefore the manuscript must not claim temporal slack or the margin alone as novel; the narrower distinction is the explicit deadline miss model and downstream potential readout and evaluation for both delivered and missed events across composed Transformer operators.

Primary source: [published article](https://doi.org/10.1016/j.neunet.2023.09.011).

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

The authoritative ViT-B/16 evidence is the completed calibrated campaign `vit_base_calibrated_theta_rt_ratio_float64_bounds3_v2` at source `f7b74c1aef38502caccf532d1e58a7cf321833d6`. It uses calibration policy 2, output bounds policy 3, float64, batch size 32, and the fixed 5,000-image ImageNet-1k validation subset. The threshold is $\theta=20$, selected on the separate training 5k artifact and confirmed by replay and neighboring validation candidates.

The maintained order is threshold calibration and selection, deterministic dense and spiking references, the timing-noise sweep at deadline margin ratio 4, and the deadline margin ratio sweep at $r_t=10^{-5}$. The later high-scale extension changes only the timing-noise grid. Static threshold mismatch, parameter perturbation, 50k validation, W&B, and TensorBoard are excluded from this evidence.

Until timing error draws for inactive members of signed pairs are removed, site counts are simulator diagnostics and are not interpreted as physical event totals or energy estimates.

Every stage keeps the noise-free tensor path as a parity reference. No stage may introduce `gaussian_multiplication_operator`, an operator-specific sampler, or invalid-result propagation.

## Sigma and Deadline-Margin Grid

The earlier 12 by 13 joint grid and its 470-run manifest are historical tooling, not the current result or an active queue.

The completed campaign separates the two questions into one-dimensional sweeps after selecting $\theta$. A new joint grid, static threshold mismatch, or another uncertainty axis requires a separate protocol under [[deferred-experiments#Additional Robustness Axes]]. The former threshold-40 and uncalibrated designs are summarized in [[deprecated#과거 실험과 범위 감사#과거 ViT Timing Noise Campaigns]].

## Timing Noise Scale Sweep at Ratio 4

The completed sweep fixes the deadline margin/noise standard deviation ratio at 4 and varies $r_t$ with the selected $\theta=20$.

For the base campaign, $r_t=10^{-5}10^{i/8}$ for integer indices 0 through 8. Each point uses seeds 0, 1, and 2. The mean top-1 accuracies are 85.9200, 85.8133, 85.8133, 85.7667, 85.5467, 85.2600, 84.3933, 82.9667, and 79.8733 percent in increasing $r_t$ order. The clean spiking reference is 86.00 percent.

The completed high-scale extension keeps the same source, checkpoint, data, selected calibration table, ratio, and seeds. It adds $r_t\in\{1.778279\times10^{-4},3.162278\times10^{-4},5.623413\times10^{-4},10^{-3}\}$ with mean accuracies 63.44, 34.34, 8.48, and 0.6867 percent. Their 95% Student-$t$ intervals are 60.553--66.327, 30.874--37.806, 6.800--10.160, and 0.184--1.189 percent.

The combined display uses nine logarithmically spaced values $10^{-5}10^{i/4}$ over two decades, reusing the five exact matching base points and adding the four extension points. It does not pool source identities or interpolate unexecuted conditions. The extension is a diagnostic stress test on the fixed 5k subset and is not a full ImageNet-1k validation result.

## 마진과 노이즈의 비율 실험

선택된 $\theta=20$과 $r_t=10^{-5}$를 고정하고 deadline margin/noise standard deviation ratio만 바꾸어 정확도와 deadline miss를 측정한다.

비율은 0, 1, 2, 2.5, 3, 3.5, 4, 5, 6이고 각 조건은 seed 0, 1, 2로 반복한다. 증가하는 비율 순서의 평균 top-1 정확도는 77.3200, 85.3467, 85.9267, 85.9733, 85.9267, 85.9400, 85.9200, 85.9133, 85.9133 percent이다. 비율 2 이상에서는 clean spiking 기준 86.00 percent와 거의 같은 plateau를 보인다.

Deadline margin은 calibration의 5% 구간 여유와 다른 설정이며 frozen range를 다시 선택하지 않는다. 정확도와 deadline miss의 관계는 simulator robustness로 해석하고 calibrated hardware behavior로 해석하지 않는다.

## Calibrated Threshold and Noise Sweeps

The completed campaign selects $\theta$ with calibration and output bounds policy 3 before evaluating the two separate noise axes; threshold-40 uncalibrated results are historical evidence only.

The exact threshold candidates are $10\,2^{i/2}$ for integer indices 0 through 8. Each candidate receives a separate 109-site calibration table from the training 5k artifact. The smallest candidate within 0.5 percentage points of the best training accuracy is $\theta=20$: it records 4,590/5,000 correct, replays with the same count and prediction digest, and records 4,300/5,000 on validation. The dense validation reference is 4,303/5,000. All candidates from 20 through 160 record the same 86.00 percent validation accuracy; 10 and $10\sqrt{2}$ record 76.86 and 84.68 percent.

For $\theta=20$, $\sigma_t=2\theta r_t=40r_t$ and the deadline margin is the requested ratio multiplied by $\sigma_t$. [[evaluation#Calibrated Three Sweep Campaign]] defines the selection and identity contract, while [[evaluation#Calibrated Three Sweep Reporting]] defines final three-replica aggregation. The campaign completed nine calibration collections and 71 evaluations; 50k confirmation and automatic manuscript promotion were deliberately omitted.

## Gaussian Noise Statistics

Maintained experiments expose event delivery and readout saturation counters for each site so robustness results can be related to delivery and readout effects in the simulator.

The statistics interface reports event count, deadline misses, nominal deadline events, deadline ULP range, output count, and lower/upper rail saturation for each named site. Reconfiguring the Gaussian generator starts a new replica and clears these counters; callers can also clear them explicitly.

Output saturation is counted from the raw physical readout before its required rail clamp. Both denominators must be reported: event rates use event count, while saturation rates use output count.
