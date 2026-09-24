# BrainScaleS-2 Encoder Measurement and ViT-B Handoff

이 문서는 BrainScaleS-2의 $\phi_{\mathrm{NP}}$·$\phi_{\mathrm{NL}}$ 측정값을 ViT-B 강건성 평가에 연결할 때 남아 있는 문제와 필수 실험 계약을 다른 작업 채팅에 전달한다.

관련 주장 경계는 [[neurips-hardware#Primitive operator 측정에서 전체 모델 추정으로]], 시뮬레이터 경계는 [[noise#Interpretation Limits]], 원고 작업 순서는 [[todo#Manuscript Revision Master Checklist#P1 Robustness and Non-Ideality Evidence]]를 따른다.

## 문제 상황

현재 핵심 문제는 좋은 단일 회로의 timing noise를 모델 전체에 복제하면 chip-wide 성능을 낙관적으로 추정하고, 측정 단위와 시뮬레이터의 local-window $r_t$가 일치하지 않는다는 점이다.

### 회로 선택 편향

Calibration $r_t$가 가장 작은 좌표를 선택한 뒤 그 validation 분포를 모든 encoder 위치에 적용하면 best-circuit upper bound를 chip population의 대표값처럼 사용하는 오류가 생긴다.

전체 Transformer 배치에는 많은 물리 회로가 필요하다. 따라서 best coordinate는 달성 가능한 상한선으로만 보고하고, median coordinate와 회로 분포를 main model evaluation에 포함해야 한다. Static device mismatch를 제외한다는 limitation만으로 이 선택 편향은 해결되지 않는다.

### Encoder별 정규화 불일치

$\phi_{\mathrm{NL}}$의 absolute timing deviation을 $\phi_{\mathrm{NP}}$의 fitted span으로 나누면 logarithmic encoder의 local time window에 적용할 noise fraction이 되지 않는다.

각 encoder는 calibration에서 자신의 signal span을 고정해야 한다.

$$
\widehat T_{\mathrm{NP},i}
=|\widehat f_{\mathrm{NP},i}(31)-\widehat f_{\mathrm{NP},i}(0)|,
\qquad
\widehat T_{\mathrm{NL},i}
=|\widehat f_{\mathrm{NL},i}(31)-\widehat f_{\mathrm{NL},i}(1)|.
$$

$$
r^{(e)}_{t,i}=\frac{\sigma_{t,e,i}}{\widehat T_{e,i}},
\qquad e\in\{\mathrm{NP},\mathrm{NL}\}.
$$

기존 NP-normalized NL 값은 비교용 진단값으로 보존할 수 있지만, logarithmic encoder의 $r_t$로 모델에 주입하면 안 된다.

### Code 분포와 단일 noise scale

Code-uniform conditional variance는 회로 비교에는 유용하지만 실제 모델의 encoder 입력 분포를 반영하지 않는다.

회로 $i$, encoder $e$, code $q$의 within-code variance를 $s^2_{eiq}$라 하고 label-free model calibration에서 location $\ell$의 code frequency를 $\pi_{e\ell q}$라 하면 model-weighted scale은 다음과 같다.

$$
\sigma^{\mathrm{model}}_{t,e,i,\ell}
=\left(\sum_q \pi_{e\ell q}s^2_{eiq}\right)^{1/2}.
$$

단일 Gaussian을 유지한다면 이를 사용하고 결과를 “measured scale로 parameterize한 Gaussian sensitivity”라고 불러야 한다. Code-conditioned residual을 직접 sampling하지 않았다면 “empirical distribution injection”이라고 부르지 않는다.

### Deadline miss gate와 모델 성능

Hardware screen의 5% miss gate는 측정을 계속할 수 있는지 판정하는 permissive requirement이지 모델 정확도를 보장하는 기준이 아니다.

회로별 physical margin은 다음처럼 simulator의 standard-deviation 단위로 환산한다.

$$
m^{\mathrm{phys}}_{e,i}
=D-\max_q\widehat f_{e,i}(q),
\qquad
k^{\mathrm{phys}}_{e,i}
=\frac{m^{\mathrm{phys}}_{e,i}}{\sigma_{t,e,i}}.
$$

5--25 microsecond encoding window와 60 microsecond deadline의 nominal slack은 35 microseconds이다. 이 slack에서 Gaussian tail이 예측하는 miss보다 실제 miss가 크다면 non-firing을 같은 Gaussian deadline tail로 합치지 않는다. Code별 held-out miss rate를 별도로 주입하고 보고한다.

### 측정 범위와 모델 sweep 범위

측정된 validation $r_t$가 model sweep 밖에 있으면 hardware condition에서의 정확도를 주장할 수 없다.

현재 reference level $10^{-3}$을 유지하려면 current-schema ViT-B sweep이 최소 $10^{-3}$까지 포함되어야 한다. 실제 best, median, 90th-percentile validation $r_t$가 더 크면 sweep도 그 최대값 이상까지 확장한다. 각 measured point에는 primitive latency와 held-out miss rate를 함께 기록한다.

### 32-code grid와 mapping error

32개 potential code가 measurement grid인지 실제 inference quantizer인지 구분해야 하며, mapping error를 timing variation에 섞으면 안 된다.

Timing-only evaluation은 continuous potential을 유지한다. Deployment proxy는 먼저 32-level quantization을 적용한다. 따라서 clean, quantization-only, timing-only, quantization-plus-error를 분리한다. Mapping error는 code mean과 frozen fitted transfer의 결정론적 차이이며 code-indexed offset으로만 적용한다.

### 주장 범위

이 측정은 $\Phi$ encoder의 marginal error evidence이며 full Transformer chip execution이나 모든 $\Phi/\Psi$ primitive의 joint error model이 아니다.

검증되지 않은 $\Psi$ primitive는 ideal로 남았다고 명시한다. Best-circuit 결과만 있을 때는 “attainable upper bound”만 주장한다. Median 또는 circuit-population 결과가 없으면 chip-wide model robustness를 주장하지 않는다.

## 확정된 실험 계약

다음 선택은 결과를 본 뒤 바꾸지 않는 protocol 결정이다.

- Operating point는 eligible circuit들의 calibration median $r_t$로 선택하며 best coordinate 값으로 선택하지 않는다.
- Coordinate quantile은 calibration data로만 정하고 validation에서 다시 선택하지 않는다.
- $\phi_{\mathrm{NP}}$와 $\phi_{\mathrm{NL}}$은 각각 자신의 frozen signal span을 사용한다.
- Representative-code screen은 후보 선택용이며 reference level 도달을 확정하지 않는다.
- Formal result만 전체 code와 128 calibration + 128 validation repetitions를 사용한다.
- Calibration block 다음에 validation block을 순차 획득하고 acquisition order와 wall-clock timestamp를 보존한다.
- First-spike time만 statistic에 사용하며 multiple-spike rate는 별도로 보고하고 0.1% 이하를 요구한다.
- Fixed-latency와 longer-latency operating point를 같은 $r_t$ ranking에 섞지 않는다.
- Primitive latency, absolute $\sigma_t$, signal span, $r_t$, miss rate를 항상 함께 기록한다.
- BrainScaleS-2 artifact는 보호된 증거이므로 명시적 대상 지정 없이 삭제하지 않는다.

## 필수 Hardware 실험

Hardware 측정은 회로 population을 확인한 뒤 formal coordinate를 고정하는 순서로 수행한다.

### 전체 좌표 screen

두 encoder에 대해 512개 atomic-neuron coordinate를 모두 조사한다.

- 64-coordinate graph 여덟 개로 나눠 실행한다.
- $\phi_{\mathrm{NP}}$ codes는 $\{0,15,30\}$, $\phi_{\mathrm{NL}}$ codes는 $\{1,8,31\}$을 사용한다.
- Code당 32 repetitions를 calibration 16과 validation 16으로 나눈다.
- Operating point별 eligible count, calibration/validation $r_t$ distribution, miss rate, multiple-spike rate를 저장한다.
- Operating point ranking은 eligible coordinates의 calibration median으로 한다.

### Formal coordinate confirmation

고정된 operating point에서 calibration distribution의 best, median, 90th-percentile에 가장 가까운 coordinate를 선택한다.

- $\phi_{\mathrm{NP}}$: codes 0--30, code당 calibration 128 + validation 128.
- $\phi_{\mathrm{NL}}$: codes 1--31, code당 calibration 128 + validation 128.
- 각 coordinate의 transfer parameters, primitive-specific span, code별 variance와 miss, mapping offset, latency를 저장한다.
- Code 31 of $\phi_{\mathrm{NP}}$는 immediate-firing endpoint diagnostic으로만 보존한다.
- Best coordinate는 upper bound, median coordinate는 main measured condition, 90th percentile은 adverse condition으로 사용한다.

### Miss와 drift 진단

Formal acquisition은 Gaussian late arrival와 다른 missing-event 원인을 구분할 수 있는 증거를 남긴다.

- Code별 actual held-out miss rate를 저장한다.
- Fitted latest mean time, deadline, $m^{\mathrm{phys}}$, $k^{\mathrm{phys}}$를 저장한다.
- Acquisition 시작·종료 timestamp와 block order를 저장한다.
- 가능하면 동일 operating point에서 더 긴 diagnostic deadline을 사용해 late spike와 non-firing을 구분하되, 이 diagnostic을 fixed-latency result와 섞지 않는다.

## 필수 Local/GPU 실험

Model evaluation은 measured conditions를 같은 checkpoint, calibration, image population에서 직접 비교한다.

### Screening median model sweep

현재 model-scale sensitivity campaign은 best 또는 adverse coordinate를 사용하지 않고 validation screening median pair만 사용한다.

- $\phi_{\mathrm{NP}}$: coordinate 184, $r_t=0.010838111004060678$.
- $\phi_{\mathrm{NL}}$: coordinate 1, $r_t=0.024617376541590685$.
- 두 encoder는 항상 함께 활성화하고 공통 multiplier $\alpha\in\{0.003,0.01,0.03,0.1,0.3,1\}$로 scale한다.
- CCT-7, ViT-S/16, ViT-B/16의 ANN, deterministic converted baseline, noisy converted accuracy를 같은 model별 population에서 비교한다.
- Noise seed는 0--2, dtype은 float64, deadline margin은 $4\sigma$로 고정한다.
- Poseidon GPU 0은 4,096 MiB 이하의 기존 allocation을 허용하고 CCT를 담당한다. GPU 1은 ViT-S, GPU 2--3은 ViT-B를 담당한다.
- 추가 multiplier는 protocol identity를 바꾸지 않는 immutable request로 실행하며, 동일 checkpoint, calibration, data, source, hardware summary checksum을 가진 cell만 결합한다.

이 campaign은 [[evaluation#Evaluation and Verification#Screening Median Model Timing Noise Sweep]]에 정의된다. 결과는 measured scale로 parameterize한 Gaussian sensitivity이며 empirical residual sampling이나 full-chip inference가 아니다. NP screening median은 usable coordinate 네 개의 median이므로 chip-wide median으로 부르지 않는다.

### Model code histogram

Label-free model calibration population에서 encoder kind와 location별 normalized potential을 32개 code bin에 매핑한다.

- $\phi_{\mathrm{NP}}$와 $\phi_{\mathrm{NL}}$ histogram을 분리한다.
- Evaluation labels를 사용하지 않는다.
- Histogram과 source calibration identity를 artifact에 기록한다.
- Uniform and model-weighted timing scales를 모두 산출한다.

### ViT-B measured-condition evaluation

Current-schema ViT-B evaluation을 GPU 0과 GPU 1에서 독립 프로세스로 실행한다.

- GPU 0: $\phi_{\mathrm{NP}}$ only.
- GPU 1: $\phi_{\mathrm{NP}}+\phi_{\mathrm{NL}}$.
- 먼저 동일 fixed 500-image subset에서 smoke/selection evaluation을 수행한다.
- 최종 paper result는 고정된 5,000-image population과 기존 seed contract를 따른다.
- Clean, best, median, 90th-percentile measured conditions를 같은 evaluator로 비교한다.
- Sweep은 최소 $10^{-5}$부터 가장 큰 formal validation $r_t$ 이상까지 포함한다.
- 각 결과는 encoder별 $r_t$, primitive latency, observed model miss rate와 accuracy를 함께 기록한다.

### Quantization과 mapping ablation

측정 grid와 model error source를 분리하기 위해 같은 population에서 다음 조건을 평가한다.

1. clean continuous potential;
2. 32-level quantization only;
3. timing noise only;
4. quantization plus timing noise;
5. quantization, deterministic mapping offset, and timing noise.

Mapping error를 main-text claim에서 제거한다면 condition 5는 생략할 수 있다. 제거하지 않는다면 mapping offset은 calibration에서 고정하고 validation 또는 task label로 조정하지 않는다.

### Deadline margin evaluation

Observation margin은 measured median condition에서 한 번만 평가한다.

- Physical $k^{\mathrm{phys}}$와 model margin grid의 관계를 명시한다.
- Actual measured miss와 Gaussian late-arrival sensitivity를 분리한다.
- Accuracy, all-event miss rate, encoder-kind miss rate를 함께 보고한다.
- Margin이 miss를 줄여도 timing variation 자체를 줄이지 않는다는 점을 해석에 반영한다.

## Artifact 계약

다른 채팅은 숫자를 대화 출력에서 복사하지 말고 immutable artifact에서 읽어야 한다.

Hardware artifact에는 다음 항목이 필요하다.

- chip identifier, calibration checksum, Git revision;
- operating point와 physical coordinates;
- calibration/validation split, acquisition timestamps;
- per-code first-spike samples, miss and multiple-spike counts;
- encoder별 transfer parameters와 signal span;
- absolute $\sigma_t$, uniform and model-weighted $r_t$;
- encoding window, observation time, deadline, maximum primitive latency.

Model artifact에는 다음 항목이 필요하다.

- checkpoint, dataset population, calibration identity and hashes;
- NP/NL injection scale and source hardware artifact;
- quantization and mapping-error flags;
- seed, accuracy, confidence interval and miss statistics;
- measured-condition label: best, median, 90th percentile, or clean.

## 완료 판정

Hardware claim은 다음 증거가 모두 있을 때만 완료된 것으로 본다.

- 512-coordinate distributions for both encoders;
- formal best, median, and 90th-percentile validation artifacts;
- primitive-specific NP and NL normalization;
- code distribution or an explicit uniform-code limitation;
- measured conditions located on the current-schema ViT-B sweep;
- quantization separated from timing noise;
- physical margin converted to model $k$ and actual miss reported;
- Appendix and main text with no `Results pending` placeholder.

Formal median $r_t$에서 accuracy가 붕괴하면 결과를 실패로 숨기지 않는다. 이 경우 결론은 full-model BrainScaleS-2 robustness가 아니라 measured encoder feasibility와 required timing-noise gap이다.

## 다른 채팅의 시작 순서

새 작업 채팅은 live process를 재시작하기 전에 current state와 authoritative artifacts를 확인한다.

1. `hostname`으로 local GPU host와 EBRAINS 환경을 구분한다.
2. `/data/delayed-temporal`과 `/data/delayed-temporal-worktrees/ebrains-toy-ann2snn`의 branch와 dirty state를 확인한다.
3. EBRAINS 쪽 `/mnt/user/shared/AnalogAttention` 아래 watcher PID와 artifact status를 실제 process handle로 확인한다.
4. 완료된 worker artifact와 checksums를 확인한 뒤에만 resume 또는 rerun을 결정한다.
5. 기존 best-coordinate $r_t$를 main result로 승격하지 않고 512-coordinate aggregation과 median selection을 먼저 수행한다.
6. Formal hardware 결과가 정해진 뒤 GPU 0/1 ViT-B evaluation을 시작한다.

현재 ICLR Appendix protocol은 `paper/iclr_2027/iclr2027_conference_appendix.tex`의 `BrainScaleS-2 Measurement Protocol for Timing Noise`에 있다. Source 끝의 `EXPERIMENT REQUIRED` comments는 이 문서의 필수 실험과 일치해야 한다.
