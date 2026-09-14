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

A manuscript-supporting ViT-B/16 noise campaign would require an `approved` `selection.json` produced by [[evaluation#ViT-B/16 Global Theta Selection]]. The active evidence is instead bounded to the `confirmed` $\theta=40$ operating point on the fixed 5,000-image validation subset. Broader validation and independent uncertainty axes are catalogued in [[deferred-experiments]] and are not scheduled.

The active protocol is:

1. retain clean spiking and dense references and verify checkpoint parity;
2. aggregate the completed pilot only through its first transition bracket;
3. align the GELU constant synaptic scaling with the manuscript before any result is used for reporting;
4. in one clean pass, count scalar values entering the GELU cubic term outside $[-\theta,\theta]$ before clamping as a diagnostic rather than a stopping gate;
5. after the scaling change, recheck only the two bracket endpoints and, if the bracket persists, add one midpoint at $r_t=3.162\times10^{-5}$; and
6. keep the pilot and accepted results from the changed source separate while reporting accuracy, confidence intervals, and empirical miss statistics without describing the fixed subset as full validation.

Until timing error draws for inactive members of signed pairs are removed, site counts are simulator diagnostics and are not interpreted as physical event totals or energy estimates.

Every stage keeps the noise-free tensor path as a parity reference. No stage may introduce `gaussian_multiplication_operator`, an operator-specific sampler, or invalid-result propagation.

## Sigma and Deadline-Margin Grid

This section preserves the earlier ViT-B/16 grid design and its tooling contract as historical provenance; the grid is not active compute.

The experiment fixes the `confirmed` threshold \(\theta^*=40\) from [[evaluation#ViT-B/16 Global Theta Selection]] and defines

$$
\sigma_t=r_t(2\theta^*),\qquad k=\frac{m}{\sigma_t},\qquad m=k\sigma_t.
$$

It evaluates the existing 12-point $r_t$ grid against $k\in\{0,0.5,1,1.5,2,2.5,3,4,5,6,8,10,12\}$ on the fixed first 5,000 validation images. Every stochastic cell uses seeds 0, 1, and 2; clean spiking and dense references are deterministic singletons. Static mismatch, calibration, learned-parameter noise, and a second theta axis remain disabled.

[[scripts/experiments/ubai/build_sigma_margin_manifest.py#main]] validates `selection.json`, the theta raw CSV, the 5k confirmation manifest, and GPU selection. It requires lower candidates 10 and 20, validation neighbors 20, 40, and 80, replay count/digest equality, validation stability, and matching data, checkpoint, and source identities. Their SHA-256 values are embedded in each of the 470 immutable rows.

A six-run pilot records clean spiking, dense reference, three zero-margin noise scales, and the largest-scale `k=12` endpoint.

Cluster tasks disable external experiment tracking and treat the immutable manifest plus complete evaluator logs as the sole resume and aggregation contract. Existing tracking artifacts remain archived but are not required for completion.

For local execution, [[scripts/experiments/run_sigma_margin_local.py#main]] accepts only physical GPUs 4--7, keeps one evaluator per GPU, and dynamically assigns the next pending row to the first free worker. Per-run temporary data is created under an explicit persistent runtime root and removed after the evaluator exits.

An exploratory mixed-code campaign may reuse completed cluster logs and fill only missing rows with a separately committed local source after overlapping conditions differ by at most one percentage point. Such a mixed result is diagnostic evidence only: source identities stay separate, and it cannot satisfy the single-source manuscript contract.

A single-scale ratio diagnostic sets `time_noise_std_frac=1` with `theta=40`, giving an absolute timing-noise standard deviation of 80, and varies only the deadline margin/noise standard deviation ratio. The local runner accepts this complete noncanonical manifest only through an explicit opt-in flag.

[[scripts/analysis/summarize_single_scale_margin_ratio.py#main]] validates the completed 41-run contract and generates replica-level, cell-level, and site-level CSV files plus a two-panel diagnostic figure. The figure uses plain-language axis labels and remains in `artifacts/`.

The portable Python environment is unpacked under the local container data path on each compute node and removed when each task exits. Runtime extraction and temporary caches never use the compute node temporary directory, and stale experiment directories older than the task limit are removed before evaluation.

The full array is submitted only after the pilot passes and its projected total asset use remains at or below 60 GB.

[[scripts/analysis/summarize_sigma_margin_sweep.py#build_frontier]] defines recovery at each $r_t$ as the smallest preregistered $k$ whose three-seed mean is within one percentage point of the clean spiking baseline. Failure to recover at $k=12$ is retained as `unrecovered`, and later nonmonotonic cells are reported rather than removed.

The result set contains replica-level, cell-level, and site-level CSV files; a provenance JSON; a recovery-frontier JSON; and a three-panel accuracy, confidence-width, and pooled-miss-rate figure. It remains under `artifacts/` because this protocol intentionally stops at 5,000 images and does not by itself authorize manuscript promotion.

While the array is still running, [[scripts/analysis/preview_sigma_margin_sweep.py#collect_complete_cells]] may create a diagnostic snapshot from the two deterministic baselines and stochastic cells whose seeds 0, 1, and 2 all have valid logs. Missing or invalid replicas leave the corresponding cell blank instead of contributing a zero, and the preview does not estimate the final recovery boundary or authorize manuscript promotion.

[[scripts/verification/verify_sigma_margin_sweep.py#main]] checks the confirmed evidence gate, canonical grid, physical scale identities, confidence intervals, pooled counts, frontier rule, single-device Slurm contract, disabled tracking mode, pilot contract, and resumable pending manifest.

Reviving the 12 by 13 grid or adding another uncertainty axis follows [[deferred-experiments#Additional Robustness Axes]] rather than the active TODO.

## Timing Noise Scale Sweep at Ratio 4

After the ratio diagnostic, the transition search fixes the deadline margin/noise standard deviation ratio at 4 and varies only $r_t$ with $\theta=40$.

The completed `v3` pilot brackets its first transition between $r_t=10^{-5}$ and $r_t=10^{-4}$ across all three seeds. It predates the constant synaptic scaling correction and remains separate historical evidence; no `v3` log is reused or pooled with the corrected source.

The local run uses physical GPUs 4--7 with one evaluator per GPU, disables external experiment tracking, and may reuse a completed log only when the full manifest identity and run parameters match. This diagnostic locates the accuracy transition before a narrower refinement sweep.

The restarted scale sweep uses the alternative GELU cubic construction. Positive and negative magnitudes use $\phi_{\mathrm{NL}}$ with $3\tau_s$, share one domain endpoint reference event, and are decoded by $\psi_{\mathrm{ED}}$ with $\tau_s$. Every sampled encoder event follows the maintained observation deadline and configured margin.

The manifest source identity distinguishes this topology from the earlier cubic path constructed from multiplication; their logs are never combined.

The current `v5` campaign records evaluator source `648af9bb`; `67de065` is an earlier corrected revision. The adaptive control implementation originated at `f48dfc3`, and constant synaptic scaling at `0eec3e2`. Changes to execution control do not imply changes to the frozen evaluator.

The campaign reuses $\theta=40$ selected at source `bc973317` by [[evaluation#ViT-B/16 Global Theta Selection]]. Accuracy at 40 has been checked with the corrected GELU implementation, but minimality among candidates has not been verified again for that source. The selection accepts clipping and does not require every internal potential to lie within the threshold.

Layer-wise calibration is disabled through `--calibration-mode none` in [[scripts/experiments/run_sigma_margin_local.py#main]]. Residual sums and affine intervals are not replaced with measured layer-wise ranges. Configuration limits and analytic bounds remain fixed; disabling calibration never reintroduces bounds from current activation extrema. This satisfies the requirement for fixed bounds but does not exercise the layer-wise limits intended to control range growth in [[domain#Domain Propagation]].

The GELU cubic magnitude floor is fixed at `1e-5`; its upper endpoint is the smaller of theta and the largest absolute endpoint of its declared input interval. LayerNorm separately uses `clip_margin=1e-5` for its positive magnitude log inputs, with a distinct variance interval. The GELU gate's exponential input is limited to $[-80\tau_s,80\tau_s]$. These floors and the exponential limit are fixed settings, not results of layer-wise calibration or the theta search. They do not imply that every log input has the same endpoints or every potential lies in $[-40,40]$.

The `v5` configuration is recorded in `artifacts/logs/noise_scan/vit_base_rt_sweep_ratio4_theta40_gelu_synaptic_scaling_float64_v5/local-assets/experiment.json`. The later ratio sweep records the same evaluator source and settings in its `ratio-grid-13/experiment.json`. Bounds and clamp settings remain identical between clean and noisy conditions.

The corrected clean pass records the count and rate of scalar values entering the GELU cubic term outside $[-\theta,\theta]$ before clamping, without changing the existing computation. The clean smoke over five batches and 160 images counted 12,830,744 of 1,161,953,280 such values (1.104%): 12,826,654 were below the negative limit and 4,090 were above the positive limit. This denominator is scalar occurrences across sites, not images, and almost all counts lie on the negative side where the GELU gate is already saturated. The rate remains a diagnostic and is not a technical failure or stopping condition; it must not be assigned a causal top-1 accuracy loss without a matched comparison.

The corrected search starts with $r_t=10^{-5}$ and runs seeds 0, 1, and 2 over all 5,000 images. Each complete scale is reported before the user approves another round. A scale is recovered when its mean top-1 accuracy is within one percentage point of the new clean spiking result, collapsed when its mean is at most one percent and every replica is at most two percent, and degraded otherwise.

When adjacent results have the same classification, the next proposed scale moves by $\sqrt{10}$. Once a recovered lower endpoint and a degraded or collapsed upper endpoint exist, only one geometric midpoint is proposed. No larger scale is submitted after collapse, and reaching $10^{-6}$ or $10^{-3}$ without a bracket stops the search for review. The authoritative checklist is [[todo#Active Experiment Work#ViT-B Timing Robustness]].

## 적응형 실험 일괄 실행

2026-09-12 사용자 승인으로 각 단계의 추가 확인 없이 기존 적응형 탐색 규칙을 순서대로 실행한다.

2026-09-13 사용자 승인으로 기존 세 점 사이를 로그 등간격 9점으로 보강한다. 양 끝은 0.00001과 0.0001이며 기존 중간값은 그대로 재사용한다. 각 점의 seed는 0, 1, 2이고 기준 평가 두 개를 포함해 총 29회 중 11회를 재사용하며 18회를 추가한다. 기존 적응형 탐색과 별도 manifest에 기록하고 평가 소스·모델·데이터는 동일하게 유지한다.

각 스케일의 세 seed가 완료된 후 기존 집계와 판정을 실행하고 다음 권장값을 승인한다. 기존 범위, 붕괴 후 상한 제한, 중간값 한 번 평가 규칙을 유지한다. 기술 실패나 판정 불일치가 있으면 진행을 멈춘다. GPU 4–7과 검증된 로컬 모델 및 데이터를 사용한다. 승인 근거는 실행 디렉터리에 보존한다.

추가 프로세스 감지는 사용자 지시에 따라 기록만 하고 실행 중 평가를 중단하지 않는다. 새 평가를 시작할 때는 물리 장치 4–7 중 비어 있는 장치를 선택한다. 실행 중 메모리 부족, 비정상 종료, 결과 검증 실패는 기술 실패로 처리한다. 실행 제어만 별도 래퍼에서 변경하며 평가 소스와 모델·데이터 해시는 유지한다.

## 마진과 노이즈의 비율 실험

역치 40과 시간 노이즈 비율 0.00001을 고정하고 마진을 바꾸어 정확도와 마감시간 초과 비율을 측정한다.

이 실험은 [[noise#Timing Noise Scale Sweep at Ratio 4]]의 source 648af9bb, 층별 calibration 미사용, 기존 theta 선택값과 고정 입력 제한을 유지한다. 마진을 바꾸는 것은 activation bound를 다시 calibration하는 것이 아니다.

비율은 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12이며 각 조건은 seed 0, 1, 2로 반복한다. 기존 기준 평가와 비율 4 결과를 재사용하여 총 41회 중 36회를 추가한다. 비율 그래프는 로그 세로축으로 표시하고 0인 값은 표시하지 않았음을 명시한다. 실행은 기존 검증된 로컬 자산과 평가 소스를 사용한다.

사용자의 계속 진행 지시에 따라 비율 실험에서는 새 평가 시작 시 기존 점유 메모리가 1GiB 이하이고 사용률이 5% 이하인 장치도 허용한다. 실제 점유 정보는 로그에 보존하고 장치 4–7만 사용한다. 실행 중 추가 프로세스만으로는 평가를 종료하지 않는다.

조건 전환 시 점유 검사에 걸리면 해당 장치에서 대기 후 재검사하며 전체 실험을 실패 처리하지 않는다. 검증은 연속 조건 실행과 일시적인 사용률 상승 후 재개, 완료 결과 재사용을 포함한다.

## Calibrated Threshold and Noise Sweeps

The current campaign repeats threshold selection with calibration and output bounds policy 3 before evaluating separate timing noise and deadline margin sweeps; previous threshold-40 results remain historical evidence.

The approved tag is `vit_base_calibrated_theta_rt_ratio_float64_bounds3_v1`. [[evaluation#Calibrated Three Sweep Campaign]] defines training-only selection and opposite-environment replay. The exact threshold values are $10\,2^{i/2}$ for integer indices 0 through 8; timing noise values are $10^{-5}10^{i/8}$ over the same indices. These are not rounded before execution.

For the confirmed threshold, the absolute timing standard deviation is $\sigma_t=2\theta r_t$ and deadline margin is the requested ratio multiplied by $\sigma_t$. The timing noise sweep fixes ratio 4. The ratio sweep fixes $r_t=10^{-5}$ and uses 0, 1, 2, 2.5, 3, 3.5, 4, 5, and 6. Deadline margin is distinct from the additional 5% calibration interval width. Bounds stay frozen during each evaluation.

[[evaluation#Calibrated Three Sweep Scheduling]] enforces 17 distinct conditions for each of seeds 0, 1, and 2 and reports twice before the final aggregate. [[evaluation#Calibrated Three Sweep Reporting]] keeps temporary estimates separate from final confidence intervals. No static mismatch, weight noise, 50k evaluation, automatic range extension, or automatic manuscript promotion is part of this campaign.

## Gaussian Noise Statistics

Maintained experiments expose event delivery and readout saturation counters for each site so robustness results can be related to delivery and readout effects in the simulator.

The statistics interface reports event count, deadline misses, nominal deadline events, deadline ULP range, output count, and lower/upper rail saturation for each named site. Reconfiguring the Gaussian generator starts a new replica and clears these counters; callers can also clear them explicitly.

Output saturation is counted from the raw physical readout before its required rail clamp. Both denominators must be reported: event rates use event count, while saturation rates use output count.
