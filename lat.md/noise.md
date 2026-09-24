# Noise Model

The maintained noise model adds Gaussian error directly to TTFS spike times and derives deadline misses from the same sampled event.

## Configuration

Direct timing noise uses one process-wide configuration and a dedicated random generator seeded once per experiment replica.

The process-wide configuration stores a dimensionless time-window fraction, optional per-encoder-kind fractions, the absolute time mean, deadline-margin ratio, seed, and generator, and validates them before installation.

The generator advances across forward calls. Reconfiguring it restarts a replica; an individual forward must not reseed it. Because this state is process-wide, the maintained path rejects `DataParallel` execution.

Evaluation entry points expose a dimensionless standard-deviation fraction $r_t$. For an encoder with declared time-window length $T$, that invocation uses $\sigma_t=r_tT$; optional linear and logarithmic overrides follow the same local rule. There is no global absolute conversion scale.

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

Every delivered event retains its sampled timestamp, including an early value below the nominal interval or a late value accepted by an observation margin. A missed event stores the receiver deadline only as a finite carrier value and sets `fired=False`; consumers must inspect the mask instead of substituting that placeholder into temporal arithmetic. The model also defines the matching analytic tail probability.

The same Gaussian draw determines both the stored event time and whether the event misses its deadline; no independent event-dropout channel is sampled.

## Fixed Observation Deadline

Each encoder keeps one nominal code interval and one fixed receiver cutoff for its invocation.

Without a margin the cutoff is the nominal endpoint:

$$
T_{\mathrm{obs}}=T_{\mathrm{code}}.
$$

Any sampled event later than this receiver cutoff is a deadline miss. A configured margin adds a nonnegative waiting duration; it does not change the nominal encoding map.

The diagnostic deadline margin sweep may allow events to arrive up to $m=k\sigma_t$ after $T_{\mathrm{code}}$, with $m\ge 0$. It keeps the nominal encoding interval fixed but records the receiver cutoff separately as $T_{\mathrm{obs}}=T_{\mathrm{code}}+m$. An event arriving during this additional interval is delivered at its sampled timestamp. This is a late arrival tolerance diagnostic, not a calibrated hardware window.

The tensor paths, boundary cases, and manuscript comparison are recorded in [[noise-timestamp-audit]].

## Comparison with Stanojevic et al.

Stanojevic et al.'s $\zeta$ and the maintained deadline margin both allocate temporal slack, but they target different scheduling failures; a generic margin is not a novel mechanism.

Stanojevic et al. set $t_{\max}^{(n)}=t_{\min}^{(n)}+(1+\zeta)X^{(n)}$, with $t_{\min}^{(n)}=t_{\max}^{(n-1)}$, using the maximum activation observed in training data to prevent the earliest output spike in layer $n$ from preceding all input spikes from layer $n-1$. This changes the nominal layer schedule and code interval. Their main construction can also force an inactive ReLU neuron to fire at $t_{\max}^{(n)}$. Their separately reported Gaussian timing perturbation experiment changes spike times, but the paper does not define a delivery mask at a receiver deadline or a grace rule after the nominal code window.

The maintained diagnostic instead keeps the nominal code interval and potential bounds fixed, samples additive timing error, and classifies delivery against $T_{\mathrm{code}}+m$. An event arriving within $m=k\sigma_t$ retains that raw timestamp. Physical pulse-width modulation readout uses the extended receiver cutoff, whereas exponential decoding retains the nominal code mapping; bounded operator outputs still apply their declared potential limits. The margin targets events delayed by noise rather than output firing before the preceding layer completes. Therefore the manuscript must not claim temporal slack or the margin alone as novel; the narrower distinction is the explicit deadline miss model and downstream potential readout and evaluation for both delivered and missed events across composed Transformer operators.

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

`SpikeSample.domain` retains the nominal code interval, while `SpikeSample.observation_deadline` records the inclusive receiver cutoff. `fired` selects the physical state evolution before readout. It is event metadata, not an output validity flag. The simulator must not propagate an `invalid` result, abort the operator chain, or replace the readout with an arbitrary fallback.

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

There is no separate Gaussian multiplication operator or secondary sampling helper. Physical consumers receive a time-and-delivery record, while noise-free callers preserve the deterministic `(time, bounds)` interface. One configuration may select different standard deviations for the linear and logarithmic encodings while retaining one generator and one deadline rule.

Measured encoder overrides support a marginal timing noise sensitivity test. They do not turn independently measured primitive statistics into a calibrated BrainScaleS-2 system prediction, and they do not model cross-primitive correlation or state coupling.

## Layer-Shared Reference Event

An affine layer treats its zero-reference timing signal as a physical spike shared by the whole layer call.

[[utils/transformers/models/spiking_ops.py#SpikingLinear#forward]] requests both the data event and one scalar zero-reference event through the same decorator. Data events receive independent timing samples; the scalar reference sample is broadcast across the layer operation.

If a data event is absent, its contribution remains at the reset value. If the reference event is absent, integration continues to the observation deadline. These are direct applications of [[noise#Observation-Time Potential Invariant]], not operator-specific fallback policies.

## Static Range Mismatch

Static range mismatch remains separate from trial-to-trial timing noise.

[[utils/transforms/noise.py#install_range_mismatch]] samples one frozen normalized offset per supported spiking module from a dedicated seeded generator and scales it by the module's declared input range in a forward pre-hook. Equal seeds replay the complete draw without consuming the model's global RNG stream.

This proxy is not Stanojevic-style neuron-slope perturbation and should not be reported as calibrated device mismatch. Timing noise and static range mismatch remain separate experiment axes and are not enabled in the same evaluation replica.

## Static Weight and Bias Perturbation

The ViT evaluator supports one-time perturbation of loaded synaptic parameters outside the shared timing-noise module.

[[scripts/evaluation/error_analysis_vit.py#apply_parameter_noise]] applies multiplicative Gaussian weight perturbation and additive bias perturbation before evaluation. This path is static parameter uncertainty, not dynamic spike-time noise.

## Injection Scope and Compounding

Event-aware noise applies only where a consumer can interpret a delivered-event mask.

The maintained production integration covers the three affine adapters, multiplication, exponential, division, exponential difference, spiking LayerNorm, softmin and activation compositions, and attention value integration. Every noisy production event originates at the decorated encoder boundary; tensor-only branches remain only as noise-off parity references.

Missing-event semantics are already fixed by [[noise#Observation-Time Potential Invariant]]. Extending coverage means implementing each operator's ordinary physical state trajectory up to $T_{\mathrm{obs}}$ and reading the resulting clamped potential; it does not require another validity policy discussion.

### Explicit ViT Encoder Block Scope

The optional explicit scope restricts sampling and statistics to selected ViT encoder blocks without creating another noise model.

[[utils/transforms/noise.py#gaussian_time_noise_scope]] marks one block active or inactive. [[utils/transforms/noise.py#gaussian_time_noise_is_active]] preserves model-wide behavior by default, while explicit inactive regions consume no RNG state and create no Gaussian counters.

[[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTEncoder#forward]] applies the scope around each selected block. Embeddings, the final normalization, the classifier, and blocks outside the selected input-side prefix remain deterministic.

The runtime configuration stores $k$ before model construction so pretrained configuration loading cannot discard the selected prefix. Timing noise parameters and sampler state remain in the shared noise configuration.

## Interpretation Limits

The implementation is a controlled computational robustness model rather than calibrated circuit validation.

It models independent Gaussian event-time errors, deadline misses, layer- or operation-shared references, and event-aware coverage of maintained operators. It does not model temporal correlation, routing faults, temperature dependence, or calibrated BrainScaleS-2 parameters.

Experiments must report $\mu_t$, $\sigma_t$, seed or repeats, injection coverage, and observed miss rate. Deterministic conversion accuracy and static parameter perturbation remain separate evidence axes.

## Coverage and Experimental Order

The event-aware migration is complete across the shared sampler and encoder boundary, composed operators, model adapters, evaluation entry points, and seeded verification.

Affine, multiplication, exponential, exponential-difference/division, activation, softmin, and attention value paths use decorated events and retain noise-off parity references. Verification exercises opening, closing/reference, and internal exp-temporal cases.

The previous ViT-B/16 campaign used the removed global-range contract and is superseded for manuscript support. Its immutable artifacts remain provenance, but maintained evaluators reject its calibration schema and do not combine it with new replicas.

The maintained order is one schema-2 training calibration, deterministic dense and spiking references, a timing-noise fraction sweep at deadline-margin ratio 4, and a deadline-margin ratio sweep at one fixed fraction. Static range mismatch, parameter perturbation, 50k validation, W&B, and TensorBoard are excluded.

Until timing error draws for inactive members of signed pairs are removed, site counts are simulator diagnostics and are not interpreted as physical event totals.

Every stage keeps the noise-free tensor path as a parity reference. No stage may introduce `gaussian_multiplication_operator`, an operator-specific sampler, or invalid-result propagation.

## Superseded Sigma and Deadline-Margin Grid

The earlier 12 by 13 joint grid and its 470-run manifest are historical tooling, not the current result or an active queue.

The historical campaign separated the two questions into one-dimensional sweeps under its selected global range. A new joint grid, static range mismatch, or another uncertainty axis requires a separate protocol under [[deferred-experiments#Additional Robustness Axes]]. Historical designs are summarized in [[deprecated#과거 실험과 범위 감사#과거 ViT Timing Noise Campaigns]].

## Superseded Timing Noise Scale Sweep at Ratio 4

This section records historical results from the removed global-range contract; it is not an executable or manuscript-supporting protocol.

For the base campaign, $r_t=10^{-5}10^{i/8}$ for integer indices 0 through 8. Each point uses seeds 0, 1, and 2. The mean top-1 accuracies are 85.9200, 85.8133, 85.8133, 85.7667, 85.5467, 85.2600, 84.3933, 82.9667, and 79.8733 percent in increasing $r_t$ order. The clean spiking reference is 86.00 percent.

The completed high-scale extension keeps the same source, checkpoint, data, selected calibration table, ratio, and seeds. It adds $r_t\in\{1.778279\times10^{-4},3.162278\times10^{-4},5.623413\times10^{-4},10^{-3}\}$ with mean accuracies 63.44, 34.34, 8.48, and 0.6867 percent. Their 95% Student-$t$ intervals are 60.553--66.327, 30.874--37.806, 6.800--10.160, and 0.184--1.189 percent.

The combined display uses nine logarithmically spaced values $10^{-5}10^{i/4}$ over two decades, reusing the five exact matching base points and adding the four extension points. It does not pool source identities or interpolate unexecuted conditions. The extension is a diagnostic stress test on the fixed 5k subset and is not a full ImageNet-1k validation result.

## 폐기된 마진과 노이즈의 비율 실험

이 절은 제거된 전역 범위 계약에서 얻은 과거 결과를 보존하며, 새 원고 근거로 재사용하지 않는다.

비율은 0, 1, 2, 2.5, 3, 3.5, 4, 5, 6이고 각 조건은 seed 0, 1, 2로 반복한다. 증가하는 비율 순서의 평균 top-1 정확도는 77.3200, 85.3467, 85.9267, 85.9733, 85.9267, 85.9400, 85.9200, 85.9133, 85.9133 percent이다. 비율 2 이상에서는 clean spiking 기준 86.00 percent와 거의 같은 plateau를 보인다.

Deadline margin은 calibration의 5% 구간 여유와 다른 설정이며 frozen range를 다시 선택하지 않는다. 정확도와 deadline miss의 관계는 simulator robustness로 해석하고 calibrated hardware behavior로 해석하지 않는다.

## Superseded Calibrated Threshold and Noise Sweeps

The old campaign selected a global range before evaluating two noise axes. The maintained local-range contract has no corresponding selection step, and these numbers remain historical only.

The exact threshold candidates are $10\,2^{i/2}$ for integer indices 0 through 8. Each candidate receives a separate 109-site calibration table from the training 5k artifact. The smallest candidate within 0.5 percentage points of the best training accuracy is $\theta=20$: it records 4,590/5,000 correct, replays with the same count and prediction digest, and records 4,300/5,000 on validation. The dense validation reference is 4,303/5,000. All candidates from 20 through 160 record the same 86.00 percent validation accuracy; 10 and $10\sqrt{2}$ record 76.86 and 84.68 percent.

For $\theta=20$, $\sigma_t=2\theta r_t=40r_t$ and the deadline margin is the requested ratio multiplied by $\sigma_t$. [[evaluation#Historical Calibrated Three Sweep Campaign]] defines the selection and identity contract, while [[evaluation#Historical Calibrated Three Sweep Reporting]] defines final three-replica aggregation. The campaign completed nine calibration collections and 71 evaluations; 50k confirmation and automatic manuscript promotion were deliberately omitted.

## Local-Window Timing Noise Sweep

The replacement campaign varies a dimensionless timing-noise fraction at fixed deadline-margin ratio 4 after one frozen ViT-B calibration.

Each encoder uses $\sigma_t=r_tT$ for its own declared time-window length $T$. Nine logarithmically spaced fractions and seeds 0, 1, and 2 provide the accuracy curve and its 95% Student-$t$ interval. Exact points are fixed in the new manifest and are not inherited from the superseded campaign.

The exact fractions are $r_t=10^{-5}10^{i/8}$ for integer indices 0 through 8. The numerical grid matches the earlier display, but its maintained meaning is now a fraction of each encoder's own window rather than a fraction of one global range.

The completed mean top-1 accuracies are 86.0267, 86.0000, 85.9800, 86.0133, 85.9533, 85.8800, 85.8200, 85.6600, and 85.5133 percent in increasing fraction order. At $r_t=10^{-4}$, the 95% Student-$t$ interval is 84.9638--86.0628 percent. The clean spiking reference is 86.00 percent.

## Deadline-Margin Ratio Sweep

The replacement campaign fixes one local-window noise fraction and varies the nonnegative ratio between deadline margin and local timing-noise standard deviation.

For each encoder, $m=k\sigma_t$ uses that encoder's local $\sigma_t$. Calibration's 5% range margin is unrelated and remains frozen. Accuracy and deadline misses are simulator robustness diagnostics rather than calibrated hardware behavior.

The fixed fraction is $r_t=10^{-5}$ and the ratios are $k\in\{0,0.5,1,1.5,2,2.5,3,4,5,6,8,10,12\}$. The condition $r_t=10^{-5},k=4$ is shared with the timing-noise fraction sweep and is executed only once per seed.

The completed mean top-1 accuracy is 77.32 percent at $k=0$ and reaches 86.02 percent at $k=3$ and 86.0267 percent at $k=4$. The pooled deadline-miss rate falls from 10.1118 percent at $k=0$ to 0.000628 percent at $k=4$; no misses are observed at $k\in\{8,10,12\}$.

The verified bundle contains 63 replica runs and 21 unique cells at source `b1a6bf8f7baa89250201c9af96d05b6154249de5`. `noise_raw_runs.csv` and `noise_summary.csv` have SHA-256 values `5dc9666048d61483c84e8c1af145213803bc2188a6abac003ea4fe42a2542266` and `5af6ea532113ccabf2b187f117387576e4e67f9f2e8cd307d27d2c29a6b1ee10`; the promoted PDF has SHA-256 `9572d722fab9f374db6202bf98cb9b6a0e486bfb6ff8e4cdc527fb9522ee478d`.

## Gaussian Noise Statistics

Maintained experiments expose event delivery and readout saturation counters for each site so robustness results can be related to delivery and readout effects in the simulator.

The statistics interface reports event count, deadline misses, nominal deadline events, deadline ULP range, output count, and lower/upper rail saturation for each named site. Reconfiguring the Gaussian generator starts a new replica and clears these counters; callers can also clear them explicitly.

Reports retain only the fields named above. Derived resolution ratios are not part of this schema because the timing noise standard deviation is defined separately for each encoder.

Output saturation is counted from the raw physical readout before its required rail clamp. Both denominators must be reported: event rates use event count, while saturation rates use output count.
