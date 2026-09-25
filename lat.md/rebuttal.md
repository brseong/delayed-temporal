# Reviewer-risk 항목 1–12: 판단과 rebuttal 초안

이 문서는 원고 검토 과정에서 제기된 12개 항목에 대해 현재 합의된 판단, 원고 반영 상태와 reviewer 답변 논리를 기록한다. 이산 step과 진행 중인 BrainScaleS-2 결과 자체는 이 목록의 판단 범위에서 제외한다.

## 상태 요약

이 표는 12개 reviewer-risk 항목의 현재 처리 상태를 한눈에 정리한다.

| 번호 | 쟁점 | 상태 |
| --- | --- | --- |
| 1 | `exact`의 범위 | 수정 완료 |
| 2 | 임의의 ANN function 일반화 | 수정 완료 |
| 3 | 모든 primitive의 hardware realization | 수정 필요 |
| 4 | accuracy path와 SOP path의 output head 불일치 | 지적 철회: 현재 경로 일치 확인 |
| 5 | prior method와 operation count 비교 | 지적 철회: 비교 단위 확인, 범위 제한 유지 |
| 6 | 모든 $\Delta$를 순수 conversion loss로 해석 | 수정 완료 |
| 7 | negative-log firing time의 조건 | 수정 완료 |
| 8 | double-exponential 근사 조건 | 수정 완료 |
| 9 | signed integer power 일반화 | 지적 철회: algebraic construction |
| 10 | 문헌에 대한 부정형·인과 주장 | 수정 완료 |
| 11 | single data spike를 전체 효율로 확대 해석 | 현 문장으로 충분, 수정 불필요 |
| 12 | `duality`의 수학적 의미 | 정의가 이미 존재, 수정 불필요 |

## 1. `exact`의 범위

이 항목은 exactness 주장의 수학적 범위와 finite-domain 근사의 경계를 구분한다.

### 최종 판단

Exactness는 declared domain 안의 ideal operator equation에 한정된다.

`exact`는 실제 회로 전체나 무제한 함수 근사를 뜻하지 않는다. Section 4에 제시한 operator equations가 ideal continuous-time system의 stated domains 안에서 성립한다는 뜻이다. Domain 선택은 clipping boundaries와 spike-time windows를 정하므로 finite-domain approximation accuracy에 영향을 준다. GELU의 대상도 exact GELU 자체가 아니라 원고에 명시한 cubic-tanh approximation이다.

### Rebuttal 답변

답변은 ideal equation의 exactness와 physical non-ideality를 명시적으로 분리한다.

> Our exactness statement concerns the operator equations on their declared domains under ideal continuous-time evaluation. It does not claim exact physical-circuit behavior when clipping, deadline misses, or analog non-idealities are active. For GELU, the constructed target is the explicitly stated cubic-tanh approximation.

### 원고 조치

Methodology를 `operator identities`가 아니라 `operator equations`로 표현하고, stated domains에서의 exactness와 domain-dependent finite approximation을 구분하도록 수정했다.

## 2. 임의의 ANN function 일반화

이 항목은 constructive claim을 열거된 함수와 평가된 Transformer operation으로 제한한다.

### 최종 판단

원고는 arbitrary ANN function에 대한 closure theorem을 주장하지 않는다. 구성 범위는 표에 나열한 arithmetic functions와 실제 평가한 Transformer operations이다.

### Rebuttal 답변

답변은 arbitrary-function closure를 주장하지 않는다고 명확히 한다.

> We do not claim that arbitrary ANN functions are closed under the proposed operators. Our constructive claim is limited to the listed arithmetic functions and the Transformer operations evaluated in this work.

### 원고 조치

Framework의 `An ANN function $F$` 표현을 제거하고 `the arithmetic functions listed below and the Transformer operations evaluated in this work`로 제한했다.

## 3. 모든 primitive의 hardware realization

이 항목은 mechanism-level evidence와 완전한 operator 또는 system realization을 구분한다.

### 최종 판단

현재 인용과 측정은 모든 primitive의 완전한 hardware realization을 입증하지 않는다.

이 항목은 아직 남아 있다. 인용 문헌은 관련 neuron dynamics, constant-current integration, pulse-width integration 또는 구성 요소 수준의 mechanism을 서로 다른 수준으로 뒷받침한다. 이것이 각 $\Phi/\Psi$ operator와 동일한 input-output mapping, 모든 parameter range 또는 전체 composition의 single-substrate execution을 입증하지는 않는다. 진행 중인 BrainScaleS-2 실험도 selected $\Phi$ encoders에 대한 근거이지 모든 primitive와 전체 Transformer composition의 증명은 아니다.

### Rebuttal 답변

답변은 문헌과 측정이 실제로 뒷받침하는 hardware evidence의 범위를 제한한다.

> The cited hardware literature supports the component mechanisms used to motivate the operator abstractions. We do not interpret those citations as a demonstration of every complete $\Phi/\Psi$ mapping or of the full composed Transformer on one substrate. Our BrainScaleS-2 measurements concern the selected encoding operators stated in the experiment.

### 원고 조치

Framework와 Conclusion의 `each primitive ... demonstrated` 및 `known analog realizations` 표현을 mechanism-level evidence, proposed operator abstraction과 composed execution으로 분리해 다시 제한해야 한다.

## 4. Accuracy path와 SOP path의 output head 불일치

이 항목은 accuracy evaluation과 SOP accounting의 output-head 경로가 일치하는지 검증한다.

### 최종 판단

구버전 Appendix 문장에 근거한 지적이며 현재는 성립하지 않는다. ViT classifier는 `SpikingLinear`이고 final LayerNorm의 `Potential`을 직접 받는다. BERT, RoBERTa와 GPT-2의 final head도 `Potential`을 직접 받는다. Accuracy evaluation과 SOP 산정은 같은 operator-backed output head를 사용한다.

다만 end-to-end를 모든 주변 처리가 spike-domain이라는 뜻으로 확대하지 않는다. Input preprocessing, token embedding lookup, shape-only operations, output tensor 반환, argmax와 loss는 평가 경계의 일반 연산이다. 정확한 범위는 모든 학습된 affine Transformer operation과 output head가 operator-backed라는 것이다.

### Rebuttal 답변

답변은 동일한 operator-backed classifier를 쓰는 범위와 평가 경계를 함께 밝힌다.

> Accuracy evaluation and SOP accounting use the same operator-backed classifier. The evaluated path consumes the final LayerNorm potential and applies the TTFS linear composition to produce logits. The claim covers learned affine Transformer operations and output heads, not preprocessing, embedding lookup, shape-only operations, or downstream metric computation.

### 원고 조치

Appendix는 evaluated classifier와 SOP classifier가 동일한 TTFS linear composition임을 명시한다. Dense classifier라는 구버전 지적은 폐기한다.

## 5. Prior method와 operation count 비교

이 항목은 prior method와의 SOP 비교가 가능한 단위와 해석 범위를 정한다.

### 최종 판단

TTFSFormer와 본 논문의 수치는 모두 inference당 synaptic operations이며, spike 또는 reference-event delivery에 receiving fan-out을 곱하는 방식으로 산정된다. 따라서 `operation counts comparable to prior conversion methods`는 방어 가능하다.

다만 동일한 physical circuit, 포함 항목이 완전히 같은 accounting protocol 또는 measured energy를 뜻하지 않는다. 본 논문은 Data SOP와 Global SOP의 포함 범위를 공개하고 routing, memory, control과 implementation-dependent cost를 제외한다고 명시한다.

### Rebuttal 답변

답변은 synaptic-operation 단위의 비교와 system-level cost 주장을 구분한다.

> Both totals are reported as synaptic operations per inference and count event delivery to receiving fan-out. We use the comparison at the operation-count level only; it is not a claim of identical circuits, identical system boundaries, or measured energy.

### 원고 조치

`comparable operation counts`는 유지한다. SOP를 physical energy나 complete hardware cost로 확대하는 표현은 사용하지 않는다.

## 6. 모든 $\Delta$를 순수 conversion loss로 해석

이 항목은 $\Delta$를 각 행 내부의 matched difference로 해석하는 범위를 정한다.

### 최종 판단

직접 변환과 추가 학습 절차가 포함된 pipeline은 $\Delta$의 의미가 다르다.

TTFSFormer처럼 source ANN과 converted SNN을 직접 수치 시뮬레이션으로 비교한 행은 within-row conversion fidelity로 해석할 수 있다. 반면 SpikeZIP처럼 quantization-aware training 등 추가 절차가 있는 prior method에서는 $\Delta$가 conversion만 분리한 값이 아니라 해당 논문이 보고한 full method pipeline의 net difference다.

### Rebuttal 답변

답변은 direct conversion과 complete method pipeline을 나눠 설명한다.

> We use $\Delta$ as a within-row matched difference. For direct ANN-to-SNN numerical conversions, this measures conversion fidelity. When a prior method includes additional procedures such as quantization-aware training, $\Delta$ denotes the net difference reported for that complete method pipeline rather than conversion alone.

### 원고 조치

Experiment의 conversion-fidelity 문단에 위 예외를 한 문장으로 추가했다.

## 7. Negative-log firing time의 조건

이 항목은 negative-log firing-time 예시가 성립하는 normalization condition을 기록한다.

### 최종 판단

SNN 독자를 대상으로 elementary threshold-crossing integration 전체를 본문에 넣을 필요는 없다. 다만 임의의 exponential current에서 식이 자동으로 나온다고 읽히지 않도록 parameter condition을 명시해야 한다.

현재 예시는 initial membrane potential $0<V\leq1$, firing threshold $\theta=1$, reference time $C$, $\tau_s=3$과 $w\tau_s=\theta$를 둔다. 이 조건에서 threshold crossing은 $t=C-3\log V$를 준다. Reset condition은 first-spike time 계산에 필요하지 않다.

### Rebuttal 답변

답변은 긴 유도 대신 threshold-crossing equation에 필요한 조건을 제시한다.

> The example uses an IF neuron initialized at $V$, a unit threshold, reference time $C$, and an exponentially decaying current whose weight satisfies $w\tau_s=\theta$. Under these stated conditions, the threshold-crossing equation gives $t=C-\tau_s\log V$, and the example sets $\tau_s=3$.

### 원고 조치

Framework에 위 normalization condition을 추가하되 긴 적분 유도는 추가하지 않았다.

## 8. Double-exponential 근사 조건

이 항목은 double-exponential response를 single exponential로 근사할 observation-time condition을 명시한다.

### 최종 판단

핵심 누락은 fast time constant가 충분히 지난 observation-time condition이었다. 식은 LIF의 spike-evoked component로 한정하고 $\propto$를 유지하므로 전체 coefficient와 기존 membrane baseline을 별도로 전개할 필요가 없다.

$\tau_s\ll\tau_m$이면 $t-t_{\mathrm{in}}\gg\tau_s$에서 time constant $\tau_m$의 single exponential에 접근한다. 반대로 $\tau_m\ll\tau_s$이면 $t-t_{\mathrm{in}}\gg\tau_m$에서 time constant $\tau_s$의 single exponential에 접근한다. 식에는 $\tau_m\neq\tau_s$도 명시한다.

### Rebuttal 답변

답변은 LIF response의 scope와 fast component가 감쇠한 뒤라는 조건을 함께 제시한다.

> Equation (2) describes the spike-evoked component of a LIF response for $\tau_m\neq\tau_s$. The single-exponential approximation additionally requires observation after the faster component has decayed: $t-t_{\mathrm{in}}\gg\tau_s$ when $\tau_s\ll\tau_m$, or $t-t_{\mathrm{in}}\gg\tau_m$ in the converse regime.

### 원고 조치

Preliminaries의 식과 $\propto$는 유지하고 LIF scope와 두 observation-time condition을 추가했다.

## 9. Signed integer power 일반화

이 항목은 signed two-branch construction의 일반 범위와 실제 평가된 cubic instance를 구분한다.

### 최종 판단

이 지적은 철회한다. Positive input에 대해 time-constant ratio가 power exponent를 정하고, signed input은 $V$와 $-V$ 두 branch를 사용한다. Even exponent에서는 같은 부호로, odd exponent에서는 반대 부호로 branch 출력을 결합하므로 모든 positive integer exponent에 대한 construction이 algebraically 성립한다.

평가 모델이 직접 사용하는 것은 GELU의 cubic case $p=3$이지만, 이는 evaluated instance의 범위이며 general construction을 각 exponent마다 별도로 실험해야 한다는 뜻은 아니다.

### Rebuttal 답변

답변은 positive integer exponent에 대한 construction과 cubic 평가 범위를 함께 밝힌다.

> The two-branch construction is defined for any positive integer exponent: the time-constant ratio sets the magnitude power, and branch signs are combined according to exponent parity. The evaluated GELU composition instantiates the cubic case $p=3$.

### 원고 조치

현재 Methodology가 positive integer power와 even/odd branch combination을 이미 설명하므로 수정하지 않는다.

## 10. 문헌에 대한 부정형·인과 주장

이 항목은 prior work에 대한 포괄적 부정형과 단일 인과 표현을 contribution-first 서술로 바꾼다.

### 최종 판단

Prior work가 무엇을 하지 않았다고 단정하기보다 본 연구가 추가로 다루는 범위를 제시한다. Analog Transformer deployment가 제한됐다는 포괄적 주장과 그 원인을 Softmax, LayerNorm, GELU에 귀속하는 문장도 제거한다.

### Rebuttal 답변

답변은 prior work의 부재보다 본 연구가 추가로 다루는 범위를 중심으로 서술한다.

> We revised the comparison to describe the constructions used by prior TTFS methods and then state our added scope: synaptic operation counts under an explicit mapping and sensitivity to timing and primitive-level mapping errors. For analog hardware, we limit the claim to the operator-design challenge of efficiently realizing operations such as Softmax, LayerNorm, and GELU.

### 원고 조치

Related Work와 Introduction에서 `does not analyze`, `primarily consider`와 deployment limitation의 단일 인과 표현을 contribution-first 문장으로 교체했다.

## 11. Single data spike와 전체 효율

이 항목은 one-data-spike representation과 전체 event count 또는 efficiency 주장을 구분한다.

### 최종 판단

현재 원고는 이미 표현 범위를 제한한다. Abstract와 Introduction은 single spike를 전체 network event count가 아니라 activation을 communicate하기 위한 data representation으로 설명한다. Related Work의 문장도 TTFS coding 정의다. 전체 spike count, latency 또는 energy 감소를 이 사실만으로 주장하지 않는다.

Appendix의 SOP model은 data spikes와 별도로 reference delivery, synchronization, internal encoders와 signed branches를 계산한다. Conclusion은 one spike per encoded value가 injected error model의 제한이며 severe noise의 unintended additional spikes를 제외한다고 명시한다.

### Rebuttal 답변

답변은 single-spike 문장을 data representation에만 한정하고 별도 SOP 항목을 명시한다.

> The single-spike statement concerns the data representation: one data spike communicates an encoded scalar. It is not used as the total event count or as a direct claim about latency or energy. Our SOP accounting separately includes synchronization, reference delivery, internal encodings, and signed branches.

### 원고 조치

현재의 `spikes used to communicate an activation` 표현이 이미 경계를 두므로 추가 수정하지 않는다.

## 12. `Duality`의 수학적 의미

이 항목은 duality를 inverse theorem이 아닌 potential--time representation pairing으로 해석한다.

### 최종 판단

Framework는 duality를 두 방향의 mapping family가 이루는 representation pair로 정의한다.

이 지적은 철회한다. Framework는 $\Phi$를 $\mathcal V$에서 $\mathcal T$로 가는 mapping family, $\Psi$를 $\mathcal T$에서 $\mathcal V$로 돌아오는 mapping family로 정의한 뒤, 이 pair of representations를 `potential--time dual representation`이라고 직접 명명한다. 즉 `dual`은 논문 안에서 정의한 representation pairing이며 inverse 또는 bijection theorem을 주장하지 않는다.

Clipping과 deadline miss가 individual mapping의 정보 보존을 깨뜨릴 수 있다는 사실은 finite implementation의 조건이며, 원고가 inverse 관계를 주장하지 않으므로 정의와 충돌하지 않는다.

### Rebuttal 답변

답변은 local definition을 제시하고 inverse 또는 bijection 주장이 아님을 분명히 한다.

> The term is defined locally in the paper: $\Phi$ denotes mappings from potential space to spike-time space, $\Psi$ denotes mappings back to potential space, and we call this pair the potential--time dual representation. We do not claim that individual $\Phi$ and $\Psi$ mappings are mutual inverses or bijections.

### 원고 조치

Framework에 이미 정의 문장이 있으므로 inverse나 bijection이 아니라는 disclaimer를 중복해 추가하지 않는다.

## 남은 조치

12개 항목 중 manuscript claim으로 아직 남은 것은 3번이다. Hardware evidence를 component mechanism, operator abstraction, selected primitive measurement와 full composed execution의 네 수준으로 분리해야 한다. 진행 중인 BSS2 결과가 완료되더라도 이 evidence boundary는 별도로 유지한다.
