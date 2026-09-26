# RMSNorm Operator Composition

RMSNorm uses the existing multiplication, negative-log encoding, and exponential-difference operators without centering. This node records the algebra, finite-domain contract, and Llama integration.

[[utils/transforms/functions.py#rmsnorm_function]] owns the composition. [[utils/transformers/models/spiking_llama/modeling_spiking_llama.py#LlamaRMSNorm]] adds pretrained fixed gains and preserves parameter names. It does not call a dense square-root normalization as a fallback.

## Operator Form

The denominator is encoded with half the numerator time constant and the squared upper reference. This aligns both offsets and implements division by the square root through the existing exponential-difference operator.

For this local derivation, let $x_i\in[a,b]$ for $i=1,\ldots,d$, and let $\mathrm{eps}>0$ be the normalization bias. Define $R=\max(|a|,|b|)$, $U=\sqrt{R^2+\mathrm{eps}}$, and a positive encoding floor $\ell=\min(\texttt{clip\_margin},\sqrt{\mathrm{eps}},U/2)$. These are fixed interval parameters, not activation-dependent square roots. Define the local intermediate $m$ and the two magnitudes by

$$
m=\frac1d\sum_{j=1}^{d}
\psi_{\mathrm{Int}}\!\left(
\phi_{\mathrm{NP}}(x_j),\phi_{\mathrm{NP}}(0);x_j
\right)+\mathrm{eps}
=\frac1d\sum_{j=1}^{d}x_j^2+\mathrm{eps},
\qquad x_i^\pm=\max(\pm x_i,0).
$$

The two negative-potential events use the same interval containing $[a,b]$ and zero. Signed integration uses the two nonnegative accumulators documented in [[operators#Operator System#Primitive PWM Integration#Differential Physical Realization]]. The average and positive bias are fixed linear operations.

In this derivation, write $\phi_{\mathrm{NL}}(v;\tau,\theta)=\tau\log(\theta/v)$ for the existing encoder with zero lower reference and positive clipping. The magnitude and denominator domains are $[\ell,U]$ and $[\ell^2,U^2]$, respectively, with one common deadline $T=\tau_s\log(U/\ell)$. For each magnitude at least $\ell$, the delivered-event identity is

$$
\begin{aligned}
&\psi_{\mathrm{ED}}\!\left(
\phi_{\mathrm{NL}}(x_i^\pm;\tau_s,U),
\phi_{\mathrm{NL}}(m;\tau_s/2,U^2);\tau_s
\right)\\
&\quad=\exp\!\left(
\frac{\frac{\tau_s}{2}\log(U^2/m)-\tau_s\log(U/x_i^\pm)}{\tau_s}
\right)
=\frac{x_i^\pm}{\sqrt m}.
\end{aligned}
$$

Both offsets equal $\tau_s\log U$; omitting the squared upper reference would introduce an unwanted constant gain. The complete output, with each inactive or below-floor magnitude contribution defined as zero, is

$$
y_i=\gamma_i\left[
\psi_{\mathrm{ED}}\!\left(
\phi_{\mathrm{NL}}(x_i^+;\tau_s,U),
\phi_{\mathrm{NL}}(m;\tau_s/2,U^2);\tau_s
\right)
-
\psi_{\mathrm{ED}}\!\left(
\phi_{\mathrm{NL}}(x_i^-;\tau_s,U),
\phi_{\mathrm{NL}}(m;\tau_s/2,U^2);\tau_s
\right)
\right].
$$

Thus delivered events, active magnitudes, and exact arithmetic recover $y_i=\gamma_i x_i/\sqrt{\frac1d\sum_jx_j^2+\mathrm{eps}}$. The learned $\gamma_i$ values are fixed gains of the receiving contributions. There is no dynamic square-root, reciprocal-square-root, or division operation outside the existing operator interfaces.

The unrestricted $\psi_{\mathrm{ED}}$ is essential: a normalized coordinate may exceed one. The public normalized division function enforces numerator no larger than denominator and would incorrectly restrict this computation. The ED realization and its negative current convention remain those in [[operators#Operator System#Composed Functions#Division]]; RMSNorm introduces no new primitive.

## Finite Domains and Events

The implementation separates epsilon from the positive encoding floor, reuses one denominator event across both signs, and explicitly clamps noisy outputs to fixed ranges.

Squared values are clamped to $[0,R^2]$ before averaging, including when multiplication events are perturbed. Since $\ell^2\le\mathrm{eps}$ and $m\le U^2$, the denominator interval does not replace the normalization bias with the encoding floor. Below-floor numerator magnitudes produce zero after decoding; they are not treated as positive floor-valued inputs in the final result.

The clean coordinate bound follows from $x_i^2\le\sum_jx_j^2$: normalized coordinates lie in $[-\sqrt d,\sqrt d]$. Timing perturbations need not preserve that inequality, so the returned normalized potential is explicitly clamped to this same structural range. The adapter declares the frozen range $[-\sqrt d\max_i|\gamma_i|,\sqrt d\max_i|\gamma_i|]$. Both the normalized result and the adapter output now enclose this analytic range with endpoints rounded outward in their returned dtype.

The denominator encoder runs once per feature vector; both sign paths consume the same sampled event and common deadline. An external event miss retains the other event's causal contribution; a missing internal ED encoding event leaves its response at reset zero. Nominally inactive numerator contributions remain zero. Repeated samples of the denominator for different signs would change the stochastic computation and are not used.

Invalid nonpositive or nonfinite epsilon, time constant, or floor is rejected before event sampling. The derived logarithmic intervals and exponential ratio endpoints must be representable in the working dtype. The all-zero input interval is supported with positive epsilon and produces exact zero output.

## Llama Integration

Every decoder RMSNorm and the final RMSNorm use the same maintained composition, while checkpoint parameter names and shapes remain unchanged.

[[utils/transformers/models/spiking_llama/configuration_llama.py#LlamaConfig]] persists the model time constant and the RMSNorm floor, whose default is $10^{-8}$. All normalization layers receive these settings. The adapter computes the composition after the Hugging Face float32 input conversion, casts normalized values back to the input dtype, and applies pretrained gains.

Frozen output bounds reject subsequent parameter or normalization-setting changes until explicitly refreshed. RoPE and selectable attention implementations are unchanged by this work; implementing RMSNorm does not certify every Llama path as a temporal operator.

## Verification

[[scripts/verification/verify_rmsnorm_operator.py#main]] checks algebra, finite domains, shared noisy events, clock execution, and pretrained normalization parameters on the CPU.

### Algebra and Finite Domains

The symbolic identity and numerical references cover signed, zero, constant, sparse, boundary, and tiny inputs, multiple time constants, positive epsilon values, and asymmetric declared input ranges.

The original algebra and finite-domain matrix now has maximum absolute error about $4.54\times10^{-10}$ in float64 and $1.20\times10^{-7}$ in float32, including its tiny-input stress cases. Float32 uses float64 operator intermediates and is checked with absolute and relative tolerance $3\times10^{-7}$; the float64 stress tolerance remains $10^{-9}$.

These are errors on specified samples, not uniform accuracy guarantees. The former permissive float32 stress tolerance has been removed. See [[rmsnorm#RMSNorm Operator Composition#Rigorous Audit#Numerical Precision and Bounds]] for the expanded regression contract and [[deprecated#RMSNorm 수치 수정 전 반례]] for earlier observations.

### Shared Events and Noise

Forced misses compare against an independent causal-duration calculation, check shared denominator object identity and deadlines, and preserve zero rows and fixed output bounds.

The checks include delivered external events, a missed denominator, either missed numerator sign, and a missed internal ED event. Repeated seeded Gaussian runs reproduce exactly, remain finite, and differ from the deterministic result.

### Clock and Invalid Inputs

Both physical-step and fixed-step-count clock configurations exercise the existing integration and exponential state updates. Invalid normalization settings must fail without consuming random state.

The clock check compares a signed vector to the continuous reference with an explicit discretization tolerance. This is a small execution check, not a convergence study or a device-level validation.

### Pretrained Llama Parameters

Signed and zero pretrained gains, zero rows, dtype handling, frozen-bound invalidation, and configuration serialization are checked independently of the full language model.

The test disables the dense reciprocal-square-root API while running the adapter and requires one call to the maintained operator composition. It compares float32, float16, and bfloat16 output against Hugging Face normalization. [[scripts/verification/verify_spiking_llama.py#main]] additionally checks tiny model logits, checkpoint loading, cache growth, generation, and the selectable spiking attention path.

## Rigorous Audit

2026-09-25 재검증은 이상적 대수 동치와 유한 정밀도 구현을 구분한다. 확인했던 수치 오차와 bounds 불일치는 후속 수정 및 회귀 검사로 처리했고, 마스크와 극단적인 시간 설정은 이번 수정 범위에서 제외했다.

### Algebra and Exactness Conditions

정확한 실수 연산에서 활성 분자의 로그 비율은 RMSNorm과 일치한다. 유한 인코딩 floor, 이벤트 누락, 시간 이산화 및 부동소수점 오차는 이 동치의 조건 밖이다.

앞 절의 국소 기호를 그대로 사용하면 다음을 순서대로 확인할 수 있다.

1. 동일한 기준의 두 $\phi_{\mathrm{NP}}$ 출력 차이는 $x_j$이다. 이를 $x_j$ 전류로 적분한 결과는 부호와 무관하게 $x_j^2$이다.
2. 따라서 $0\le x_j^2\le R^2$이고 $\mathrm{eps}\le m\le R^2+\mathrm{eps}=U^2$이다. 또한 $\ell^2\le\mathrm{eps}$이므로 분모의 clamp는 이상적 입력에서 활성화되지 않는다.
3. 두 로그 인코딩의 기준 오프셋은 모두 $\tau_s\log U$이고, 관측 마감 시간도 동일한 $T$이다. 두 시간의 차이를 $\tau_s$로 나누면 $\log(x_i^\pm/\sqrt m)$이다.
4. 활성 경로 하나만 복원하고 원래 부호를 적용하면 $x_i/\sqrt m$이다. $\gamma_i$는 고정 계수이며, 평균 제거 단계는 없다.
5. $m\ge x_i^2/d$이므로 이상적 출력의 절댓값은 $\sqrt d|\gamma_i|$ 이하이다. dtype 변환 후의 반환 범위는 이 실수 구간을 바깥쪽으로 반올림하여 따로 보존한다.

SymPy 검사는 로그 오프셋 상쇄를 확인한다. 실제 호출 경로도 제곱에 기존 multiplication, 분모와 두 부호에 기존 logarithmic encoder, 복원에 기존 ED를 사용한다. 직접적인 활성값 제곱근이나 역제곱근으로 우회하는 경로는 없다. 다만 부호 분리, 활성 마스크, clamp, 고정 합산과 gain은 이 인터페이스 밖에서 수행하는 명시적 보조 동작이다.

### Floor and Input Masks

양의 floor는 함수 자체를 바꾸며, 원래 입력에서 만든 활성 마스크는 로그 이벤트와 별개의 정보를 최종 출력에 전달한다.

정확한 연산과 모든 이벤트 전달을 가정해도 현재 함수의 결과는 다음과 같다. 여기서 $\widetilde y_i$는 이 문단에서만 쓰는 유한 floor 구현의 출력이다.

$$
\widetilde y_i=
\begin{cases}
\gamma_i x_i/\sqrt m,& |x_i|\ge\ell,\\
0,& |x_i|<\ell.
\end{cases}
\qquad
|\widetilde y_i-y_i|<
\frac{|\gamma_i|\ell}{\sqrt{\mathrm{eps}}}
\quad\text{if }0<|x_i|<\ell\text{ and }\gamma_i\ne0.
$$

기본 floor $10^{-8}$과 epsilon $10^{-6}$에서는 이 절댓값 오차 상한이 $10^{-5}|\gamma_i|$이다. 이는 floor만의 오차이며, 시간 차 계산이나 지수 디코딩의 수치 오차를 포함하지 않는다. gain이 0이면 오차도 0이다. 작은 비영 입력에서 출력이 0이 되는 반례를 별도로 검사한다. 기존 기준 함수의 floor 마스크를 포함한 비교만으로는 원래 dense RMSNorm과의 차이를 발견할 수 없다.

마스크가 전달하는 정보는 입력 $(\ell,0)$와 $(0,\ell)$로 확인된다. 두 입력은 같은 제곱 평균을 가지며, 각 양·음수 로그 인코더에 들어가는 floor 적용 후 값도 모두 $\ell$이다. 따라서 ED로 들어가는 이벤트 시간과 전달 여부는 같지만, 최종 출력은 서로 다른 좌표에 비영 값을 가진다. 이 차이는 positive_active, negative_active 및 마지막 torch.where가 만든다.

이는 마스크의 물리적 구현이 불가능하다는 주장이 아니다. 현재 코드가 해당 활성 정보를 이벤트 외부에 보존하고 이용하며, 그 제어 동작을 순수한 Phi/Psi 합성 또는 구체적인 회로로 구성했다는 근거는 없다는 뜻이다. 따라서 “완전히 operator 형식”은 현재로서는 기존 연산자와 명시적 부호·floor 제어를 이용한 수치 구현으로 한정해야 한다.

### Complete Event Readout

768개 강제 이벤트 조합을 독립적인 관측 시점 식과 비교했다. 제곱 단계의 이벤트부터 분모, 두 부호, 내부 ED 이벤트까지 모두 포함했다.

검사는 여섯 위치의 누락 여부 $2^6$개, 세 시간 상수, 두 관측 여유 설정, 두 시간 변동 설정을 조합한다. ED 내부 위치는 두 부호 모두에 같은 강제 누락 정책을 적용하되 이벤트는 각각 기록한다. 제곱 기준 이벤트는 스칼라 하나이며, 분모 로그 이벤트는 벡터마다 한 번만 생성하고 두 부호가 같은 객체를 재사용하는지 확인한다.

독립 기준은 각 실제 전달 이벤트의 마감 시간까지 남은 적분 시간을 계산한다. 제곱 전류의 차동 적분, 일반 곱셈 범위 및 비음수 제곱 범위의 clamp, epsilon 합산, ED 중간 전위와 내부 이벤트 응답을 차례로 대조한다. 최종 값만 비교하지 않고 분모 인코더 입력과 내부 ED 인코더 입력도 검사한다. 음의 시간 변동, 양의 시간 변동, 마감 이후 전달 여유와 누락이 포함된다. 이는 지정된 조합의 동작 검증이지 모든 Gaussian 표본이나 실물 회로의 증명은 아니다.

특히 ED의 두 외부 이벤트가 모두 누락되면 차동 적분 결과는 0이다. 이 값을 다시 인코딩한 내부 이벤트가 전달되면 ED 결과는 1이지 0이 아니다. 두 부호의 외부 이벤트와 분모가 모두 누락된 RMSNorm은 활성 좌표에서 원래 부호에 따른 $\pm1$을 낼 수 있다. 내부 ED 이벤트까지 누락되어야 그 경로의 응답이 0이 된다. 이 경우도 현재 primitive의 인과적 계약과 일치하며, 별도 반례 검사로 고정했다.

### Numerical Precision and Bounds

RMSNorm의 시간 및 적분 계산은 float64로 수행하고, 반환값과 bounds는 실제 출력 dtype에 맞춘다. 수정 전 수치 반례는 현재 엄격한 정확도·범위 회귀 검사로 대체되었다.

[[utils/transforms/functions.py#rmsnorm_function]]은 기존 multiplication, logarithmic encoder, ED 호출을 그대로 사용한다. 입력 전위와 인코딩 구간을 float64에서 함께 표현해 큰 시간 값과 적분 상태의 차를 보존한다. Gaussian과 clock 실행도 같은 경로를 사용하며 dense normalization으로 우회하지 않는다.

입력 dtype에서 반올림된 구간 끝점을 먼저 포함해야 한다. 예를 들어 float32 값 40.172는 Python의 40.172보다 약간 크므로, 값만 float64로 올리고 원래 구간을 그대로 두면 입력 검사가 실패한다. [[utils/transforms/types.py#PotentialBounds#outward_rounded]]는 원래 구간을 포함하는 가장 가까운 유한 dtype 끝점을 제공한다. 구간을 값에 맞춰 동적으로 추정하는 것이 아니라 고정된 끝점만 보수적으로 반올림한다.

정규화 결과는 실수의 구조적 범위에서 clamp한 뒤 호출자의 dtype으로 반환한다. 반환 bounds는 출력 dtype으로 바깥쪽 반올림한다. Llama adapter는 입력과 가중치의 승격 결과 dtype으로 최종 범위를 설정하고, 가중치 dtype 변경도 frozen parameter identity에 포함한다. 따라서 float16 출력 1.732421875나 bfloat16 출력 1.734375를 실제보다 작은 scalar 상한과 함께 반환하지 않는다.

기존 입력 $(-0.001,0.0005,0.00025,0)$, epsilon $10^{-6}$에서 범위 반경 1, 40.172, 10000의 최대 절대오차는 모두 약 $6.67\times10^{-9}$로 감소했다. 이는 같은 float32 입력값에 대한 float64 dense 기준과의 비교이다. epsilon $10^{-12}$ 및 $10^{-6}$, 세 범위, 세 입력 크기, 세 시간 상수의 추가 검사에서는 절대오차 $3\times10^{-7}$을 요구한다.

bounds 검사는 float16, bfloat16, float32, float64, 폭 3·7·17, 양·음·영 gain과 입력 부호를 포함한다. 최종 출력을 float64로 올려 원래 scalar 끝점과 비교하며, 끝점의 표현 가능성과 반복 반올림의 동일성도 확인한다. 서로 다른 입력·가중치 dtype과 frozen parameter의 dtype 변경도 검사한다.

이 변경은 수치 시뮬레이터의 정밀도를 개선한 것이며, 장치의 정밀도가 향상되었다는 주장이 아니다. float64 중간 텐서와 연산에는 추가 메모리·실행 비용이 있고 GPU 성능은 이번 CPU 검증에서 측정하지 않았다. 유한 정밀도의 모든 가능한 범위에 대한 균일한 오차 상한을 주장하지 않는다. 비활성 입력 마스크와 극단적인 시간 상수 설정은 사용자 지시에 따라 이번 수정 범위에서 제외했다.

### Shape and Model Coverage

추가 검사는 비연속 메모리 입력, 다양한 마지막 축 길이, 전체 모델의 모든 RMSNorm 호출을 포함한다. 실제 대형 체크포인트나 언어 모델 과제 성능은 이번 검증 범위가 아니다.

폭 1, 2, 17, 128, 4096의 양·음수 입력과 두 시간 상수에서 float64 최대 절대오차는 약 $1.07\times10^{-14}$였다. 입력 크기는 $[0.2,1]$로 제한했으므로 이 결과는 작은 입력에 대한 반례를 부정하지 않는다.

두 decoder layer와 최종 normalization을 가진 작은 모델의 다섯 RMSNorm 모두 지정한 시간 상수와 floor를 전달받는지 검사한다. torch.rsqrt를 차단한 실제 모델 forward에서도 연산자 함수가 다섯 번 호출되고 유한 logits를 내는지 확인한다. 기존 작은 HF 모델의 state dictionary, 저장·복원, logits와 cache 검증은 별도로 유지한다.

Clock 실행은 연속시간 함수와 별도의 근사다. 한 float64 입력 $(-0.9,0.2,0.7,0)$에 대해 창당 128, 512, 2048, 8192 step을 사용한 추가 진단에서 최대 절대오차는 각각 약 0.2233, 0.09986, 0.01292, 0.005883이었다. 이 관측은 해당 입력에서의 오차 감소를 보여줄 뿐, 일반적인 단조 수렴 증명이 아니다. Gaussian과 clock 동시 실행은 기존 인코더가 명시적으로 거부하므로 결합 모드를 지원한다고 주장하지 않는다.

후속 수치 수정과 검증 상태는 [[todo#TODO#Manuscript Revision Master Checklist#P0 Mathematical and Operator Audit]]에 반영했다. 수정 전 구현의 검증 결과는 [[deprecated#RMSNorm 수치 수정 전 반례]]에 보존한다.
