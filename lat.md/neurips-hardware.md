# 하드웨어 검증의 주장 범위

수식 평가, 장비 기능의 문헌 근거와 실제 칩 실행을 구별하는 원칙을 유지한다. BrainScaleS-2의 당시 기능 조사와 실험 제안은 [[deprecated#과거 하드웨어 검토]]에 보관한다.

## 구성 요소와 전체 합성의 구분

구성 요소가 장비에 존재한다는 사실만으로 제안한 연산 전체의 연결이 검증되지는 않는다. 물리 파라미터와 연결 제약은 장비별로 확인해야 한다.

고정 가중치의 적분과 실행 중 다른 전위를 곱하는 동작은 다르다. 부호 처리, 상태 전달과 외부 계산 없이 연산을 연결하는 조건도 별도 근거가 필요하다. [[neurips-current]], [[neurips-mathematics#부호 있는 곱셈과 이벤트 도착 순서]]를 참조한다.

## Pooling의 독립 오차와 공유 오차

동일 논리 뉴런의 복제가 독립 변화를 줄이더라도 공유 편향은 남을 수 있다. 반복 수만으로 독립성을 가정하지 않는다.

뉴런 보정, trial별 변화, 공유 입력과 배치 위치를 구별한다. 복제로 늘어나는 뉴런·시냅스·연결·이벤트 비용도 보고해야 한다. [[neurips-current#노이즈와 비용의 해석 한계]]와 [[deferred-experiments]]에 연결한다.

## 작은 네트워크가 증명하는 범위

Attention 없는 네트워크의 하드웨어 결과는 해당 구성의 효과를 지지할 수 있지만 전체 Transformer 칩 구현의 증거는 아니다.

일반 모델, 이상적 변환, 단일 하드웨어 구현과 복제 구현을 같은 체크포인트·분할에서 구별한다. 출력 투표와 논리 뉴런 복제, 소프트웨어 노이즈 재생과 실제 칩 실행을 혼동하지 않는다. 보정에 평가 정답을 사용하지 않으며 불확실성과 자원 비용을 함께 보고한다. 실제 작업 상태는 [[todo#Manuscript Revision Master Checklist#P0 Energy, Latency, and Hardware Claims]]를 따른다.

## Primitive operator 측정에서 전체 모델 추정으로

Primitive operator의 chip 측정을 전체 변환 모델의 성능 추정에 사용할 때는 실제 chip 실행과 측정값을 사용하는 소프트웨어 평가를 구분해야 한다.

먼저 $\phi_{\mathrm{NL}}$, $\phi_{\mathrm{NP}}$, $\psi_{\mathrm{NE}}$, $\psi_{\mathrm{Int}}$ 각각에 대해 BrainScaleS-2의 회로 설정, 입력과 출력, 초기 상태, 시간 및 potential 단위, calibration 절차를 대응시켜야 한다. 외부 digital controller 또는 host가 spike time 생성, 상태 전달, 적분 종료나 인출을 대신하면 그 부분은 analog primitive의 chip 구현으로 세지 않는다. 특히 pulse-width integration을 측정한 회로 설정과 TTFS operator를 측정한 회로 설정을 동일한 실행에서 연결할 수 있는지는 별도로 입증해야 한다.

측정 결과를 primitive마다 하나의 Gaussian 표준편차로 축약하지 않는다. 입력 구간, weight, $\tau_m$, $\tau_s$, threshold, deadline, 물리 뉴런과 synapse 위치, calibration 및 반복 실험별로 평균 mapping error와 분포를 기록한다. static mismatch, 반복 실행 간 변화, weight quantization, readout saturation, routing 또는 spike loss, deadline miss를 분리한다. 같은 회로 자원이나 기준 event가 만드는 공유 오차도 독립 표본으로 바꾸지 않는다.

전체 모델 평가는 이 측정 분포를 변환 모델의 동일한 primitive 호출 위치에 적용하여 반복한다. 이 결과는 chip에서 전체 모델을 실행한 결과가 아니라 chip 측정값에 근거한 소프트웨어 성능 추정이다. 추정의 타당성은 측정에 사용하지 않은 입력, 두 개 이상의 primitive composition, 가능하면 chip에 배치할 수 있는 작은 변환 네트워크에서 예측 오차와 실제 오차를 비교하여 확인한다.

논문의 근거는 세 단계로 나눈다. Primitive의 mapping과 noise를 실제 chip에서 측정했다는 근거, 그 측정값으로 전체 변환 모델 성능을 추정했다는 근거, 실제로 배치한 네트워크를 chip에서 실행했다는 근거는 서로 대체하지 않는다. Primitive 측정만으로 전체 Transformer의 chip 구현, energy 또는 latency를 주장하지 않는다.

## 실험 섹션의 근거 순서

실험은 noise-free 변환 충실도, primitive operator의 chip 측정, 측정 분포를 적용한 모델 강건성 순으로 제시해야 각 결과가 다음 결과의 근거가 된다.

첫 실험은 ANN과 변환 SNN의 noise-free 성능, clipping, SOP를 보고하여 framework 자체의 변환 손실을 고정한다. 둘째 실험은 BrainScaleS-2에서 primitive별 mapping error, spike-time noise, static mismatch와 deadline miss를 측정하여 이후 모델 평가에 사용할 분포를 정한다. 셋째 실험은 이 측정 분포를 동일한 operator 위치에 적용하고 deadline margin을 변화시키며 task 성능과 miss rate를 보고한다.

임의의 Gaussian timing-noise sweep은 chip 측정과 별개의 주요 결과로 두지 않는다. 측정된 조건을 sweep 위에 표시하여 동작점의 위치와 그보다 큰 noise에서의 변화를 보여줄 때만 main text의 보조 분석으로 유지한다. 그렇지 않으면 appendix로 옮긴다. 측정 분포가 Gaussian과 맞지 않거나 입력 및 배치 위치에 의존하면, main result는 측정 분포를 직접 적용한 평가이고 Gaussian timing-noise sweep은 computational stress test로만 남긴다.

Deadline margin sweep은 측정 분포를 적용한 모델 평가에서 한 번만 수행한다. 같은 margin sweep을 임의의 Gaussian 조건과 측정 조건에서 각각 핵심 결과로 제시하면 두 실험의 역할이 중복된다.

새 ViT-B/16 fixed-5k 두 panel figure는 appendix의 simulated sensitivity analysis로 둔다. 첫 panel은 local-window timing-noise fraction, 둘째 panel은 deadline-margin ratio를 보고한다. 제거된 global-range selection panel은 재사용하지 않는다. Main text의 Figure 2는 실제 측정 분포가 생긴 뒤에만 채운다.

## 에너지 추정의 근거 단계

절대 energy는 SOP 총합에 고정 단가를 곱하지 않고, 하나의 target analog substrate와 계산 구간을 정한 뒤 실제 입력에 따른 회로 전력을 시간에 따라 적분한다.

Horowitz의 $0.9$ pJ는 45 nm, 0.9 V 조건의 32-bit floating-point addition에 대한 rough estimate이며 analog synaptic event 또는 이 논문의 SOP 측정값이 아니다. 따라서 기존 $E_{\mathrm{AC}}=0.9$ pJ는 Data SOP와 Global SOP의 상대적인 count를 실제 chip energy로 바꾸는 근거가 될 수 없다.

계산 구간은 입력이 준비된 시점부터 최종 출력이 준비된 시점까지로 고정한다. 한 번의 configuration과 calibration을 $M$회 inference가 공유한다면 전체 값은 다음처럼 정의한다.

$$
E_{\mathrm{Total}}=\int_0^T P_{\mathrm{run}}(t)\,dt+\frac{E_{\mathrm{configuration}}+E_{\mathrm{calibration}}}{M}.
$$

여기서 $T$는 실제 end-to-end latency이고, $P_{\mathrm{run}}$은 계산 중 활성화된 모든 chip power domain의 합이다. 준비 상태 전력을 $P_{\mathrm{ready}}$라고 할 때 계산 중 증가분만 보고하려면 다음 값을 별도로 제시한다.

$$
E_{\mathrm{dynamic}}=\int_0^T \left(P_{\mathrm{run}}(t)-P_{\mathrm{ready}}(t)\right)\,dt.
$$

$E_{\mathrm{dynamic}}$은 $E_{\mathrm{Total}}$을 대신하지 않는다. 두 값은 같은 workload와 같은 실행에서 얻고, 포함된 chip, board, host 및 input/output 범위를 명시한다.

Chip 전체 transient simulation이 아직 불가능하면 실제 model activity를 사용한 component 합산을 차선으로 둔다. 각 $\Phi/\Psi$ 호출의 증가분은 고정 상수가 아니라 input value, spike time, pulse width, synaptic weight, fan-out, route, threshold와 code window의 함수로 측정한다. 이 값에는 최소한 encoder와 decoder, synapse와 integrator, reset, reference spike와 synchronization 전달, comparator/WTA, memory, routing, control, 필요한 conversion과 input/output, static analog bias current, leakage 및 calibration을 포함한다. 준비 상태 전력과 각 component 증가분은 중복하여 더하지 않는다.

제작 전 값은 실제 회로와 mapping을 고정한 뒤 SPICE-level transient simulation으로 구한다. Layout 이후의 parasitic element를 포함하고, process, supply voltage와 temperature 조건, circuit mismatch, 실제 workload의 spike-time 분포와 fan-out을 변화시킨다. Analog primitive는 SPICE로, routing, memory와 control은 해당 구현의 switching activity로 평가한다. 정확도, deadline miss와 latency도 같은 조건에서 함께 보고한다.

Silicon이 있으면 chip 전체 power가 주 결과다. NeuroBench는 configured idle power와 active power를 함께 보고하고 그 차이와 execution time으로 dynamic energy per sample을 계산하며, 모든 active processing component와 측정 방법을 공개하도록 한다. BrainScaleS-2의 measured result도 analog core, plasticity processors, periphery와 communication links를 포함한 약 200 mW와 throughput을 결합해 2.4 microjoule per image를 보고한다. TTFS analog memory test chip 연구는 complete output layer의 242 microwatt, 196 ns와 4.74 pJ per inference per neuron을 함께 보고하고 amplifier와 comparator의 power breakdown을 제시한다. 이 값들은 범위가 다른 회로의 SOP 단가로 가져오지 않고 측정 설계의 예로만 사용한다.

근거는 세 단계로 표기한다.

1. Data SOP와 Global SOP: 명시된 mapping 아래의 algorithmic count이며 absolute energy가 아니다.
2. SPICE 또는 layout 이후 estimate: 고정한 target substrate에서 제작 전에 얻은 projection이다.
3. Measured silicon: 공개한 범위에서 실제 workload를 실행한 energy, latency와 accuracy다.

현재 논문에서는 main comparison table의 mJ 열을 제거하고 Data SOP와 Global SOP를 operation-count proxy로 유지하는 안이 가장 안전하다. Appendix에는 기존 수치를 남기더라도 $0.9$ pJ/SOP 가정의 idealized estimate라고 표시하고 hardware superiority의 근거로 사용하지 않는다. Analog superposition, positional encoding, WTA와 residual connection의 0 SOP도 physical energy가 0이라는 뜻이 아니라 현재 count 밖이라는 뜻으로 제한한다.

Absolute energy가 필요하면 먼저 target substrate, process, supply voltage, code window, supported $\Phi/\Psi$ mapping과 포함할 component를 고정한다. 그 뒤 실제 evaluation trace로 호출 수와 입력 분포를 모으고, 대표 primitive와 routing을 SPICE와 layout 이후 result로 calibration하여 위 식을 채운다. ANN baseline도 같은 task, accuracy, batch, precision, memory와 input/output 범위에서 측정하거나, 이 조건을 맞출 수 없으면 energy ranking을 만들지 않는다.

근거 문헌은 [Horowitz ISSCC 2014](https://doi.org/10.1109/ISSCC.2014.6757323), [NeuroBench](https://www.nature.com/articles/s41467-025-56739-4), [Accelergy](https://accelergy.mit.edu/paper.pdf), [BrainScaleS-2 measured result](https://pmc.ncbi.nlm.nih.gov/articles/PMC8794842/), [TTFS analog memory test chip](https://doi.org/10.1109/TVLSI.2024.3368849), [IBM analog chip](https://www.nature.com/articles/s41586-023-06337-5)이다. Accelergy의 action count와 calibrated component cost 결합은 계산 구조의 참고이며, digital accelerator에서 확인한 오차를 이 analog substrate에 그대로 적용하지 않는다.

### 선행 수식과 적용 범위

선행 연구는 전체 측정값을 event 수로 나눈 지표, component별 예측식과 circuit transient energy 식을 서로 다른 목적으로 사용한다.

[Ostrau et al.](https://doi.org/10.3389/fnins.2022.873935)은 idle system, idle neuron, source event, neuron spike, event transmission, synaptic event와 plasticity의 static 및 activity cost를 순서대로 측정해 합산한다. 논문은 하나의 compact equation보다 측정 절차와 항목을 정의하며, route 차이를 무시한다는 한계도 밝힌다.

[Darwin3](https://doi.org/10.1093/nsr/nwae102)는 측정 구간을 정규화한 power model을 식 (7)로 제시한다.

$$
P_{\mathrm{total}}=P_I+P_B+P_N n+P_S s.
$$

$P_I$는 application을 올리기 전 chip power, $P_B$는 node를 활성화한 baseline, $P_Nn$은 neuron update, $P_Ss$는 synaptic event 항이다. 이는 component별 static 및 activity cost를 분리하는 직접적인 선례지만, 1 ms update를 쓰는 digital chip 식이므로 continuous-time analog TTFS에 그대로 대입하지 않는다.

[Accelergy](https://accelergy.mit.edu/paper.pdf)의 방법은 다음 일반식으로 요약할 수 있다.

$$
E_{\mathrm{estimate}}=\sum_c\sum_a N_{c,a}\,\epsilon_{c,a}(\mathbf{u}).
$$

$N_{c,a}$는 workload에서 component $c$의 action $a$가 일어난 수이고, $\epsilon_{c,a}(\mathbf{u})$는 data pattern, multicast destination 수와 같은 runtime argument에 따른 action energy다. 논문은 고정 단가 model이 control, idle cycle 및 data에 따른 activity를 놓쳐 큰 오차를 낼 수 있음을 layout 이후 result와 비교한다. 이 구조는 우리의 실제 evaluation trace와 fan-out에 따른 cost 결합에 적합하지만, analog primitive의 $\epsilon$은 별도 SPICE 또는 measurement로 calibration해야 한다.

[Analog LIF circuit study](https://www.nature.com/articles/s44335-024-00013-1)는 transient simulation에서 회수 가능한 capacitor energy와 conduction 및 logic loss를 분리한다.

$$
E_{\mathrm{diss}}(t)=E_{\mathrm{supply}}-E_{\mathrm{cap}}=E_{\mathrm{cond}}+E_{\mathrm{logic}}.
$$

$$
\mathrm{ESOP}=\frac{E_{\mathrm{diss}}(t_{\mathrm{end}})}{N_{\mathrm{neur}}(N_{\mathrm{SPK}}+N_{\mathrm{CLK}})}.
$$

원문의 $N_{\mathrm{neur}}$는 neuron 수, $N_{\mathrm{SPK}}$는 neuron spike 수, $N_{\mathrm{CLK}}$는 clock spike 수이며 ESOP는 이 합성 event당 energy다.

이 논문은 static component를 따로 추출해 dynamic component와 분리한다. Analog primitive의 transient current를 적분하고 static energy를 code window에 따라 포함하는 방법의 선례지만, 해당 crossbar의 ESOP를 Transformer 전체 SOP 단가로 가져오는 근거는 아니다.

[NeuroBench](https://www.nature.com/articles/s41467-025-56739-4)의 system track은 서술상 다음 관계를 사용한다.

$$
E_{\mathrm{dynamic}}=(P_{\mathrm{active}}-P_{\mathrm{idle}})T.
$$

이는 measured workload의 dynamic energy per sample 정의다. [Senk et al.](https://doi.org/10.1088/2634-4386/ae379a)의 식 (2)는 전체 energy를 synaptic event 수로 나눈 보고 지표를 명시한다.

$$
E_{\mathrm{syn}}=\frac{\int_0^{T_{\mathrm{wall}}}P(t)\,dt}{T_{\mathrm{model}}\sum_\alpha N_\alpha K_{\mathrm{out},\alpha}\nu_\alpha}.
$$

이 두 식은 측정한 전체 energy를 보고하고 정규화하는 식이지, 미측정 analog chip energy를 SOP 수만으로 예측하는 식이 아니다.

이 논문에는 Ostrau와 Accelergy의 component 합산을 예측식으로, analog LIF circuit study의 transient integration을 primitive calibration으로, NeuroBench의 관계를 최종 chip 측정으로 사용하는 조합이 적합하다. 절대 energy가 없을 때는 SOP만 보고하고, target substrate가 정해진 뒤에만 component별 $\epsilon$과 $P_{\mathrm{ready}}T$를 채운다.

## BrainScaleS-2 전원 레일 측정 절차

공개된 BrainScaleS-2 software stack은 chip carrier의 여섯 전원 레일을 읽을 수 있으므로, 긴 반복 실행에서 준비 상태와 실제 workload의 power를 같은 configuration으로 비교할 수 있다.

[`haldls`의 `INA219Status`](https://github.com/electronicvisions/haldls/blob/35a398d0d5bfdfe54d6379696f0e43cb0ac7acf2/include/haldls/vx/i2c.h)는 bus voltage와 shunt voltage를 읽고 power로 변환한다. [`halco`](https://github.com/electronicvisions/halco/blob/39e621071808aeb1a283deedc36c7d1390eb274f/src/halco/hicann-dls/vx/xboard.cpp)는 `vdd12_digital`, `vdd25_digital`, `vdd12_analog`, `vdd25_analog`, `vdd12_madc`, `vdd12_pll` 좌표를 제공한다. [공개 hardware test](https://github.com/electronicvisions/haldls/blob/35a398d0d5bfdfe54d6379696f0e43cb0ac7acf2/tests/hw/stadls/vx/v3/hw/test-ina219.cpp)는 `PlaybackProgramBuilder`로 모든 레일을 읽고, [PyNN의 `InjectedReadout`](https://github.com/electronicvisions/pynn-brainscales/blob/b4069f4c00d21151b98b3a4edb3f5bfff6d3e28c/brainscales2/pynn_brainscales/brainscales2/__init__.py)은 실행 경계에서 hardware coordinate를 읽는다. [공개 저자 코드](https://github.com/fmi-basel/brainscales-2-surrogate-gradients/blob/master/src/py/strobe/backend.py)도 여섯 레일을 읽어 합산했다. 일반 EBRAINS 사용자의 carrier read 권한은 문서가 보장하지 않으므로 실제 계정에서 먼저 smoke test한다.

[`toUncalibratedPower()`](https://github.com/electronicvisions/haldls/blob/35a398d0d5bfdfe54d6379696f0e43cb0ac7acf2/src/haldls/vx/i2c.cpp)는 bus voltage와 shunt voltage의 곱을 expected shunt resistance 0.027 ohm으로 나눈다. Board별 shunt calibration과 measurement uncertainty를 확인하기 전에는 이 값을 calibrated measurement로 부르지 않는다. 포함 범위는 여섯 chip rail이며 host와 power conversion loss는 별도다.

하나의 inference보다 INA219 conversion이 길 수 있으므로 개별 inference를 직접 적분하지 않는다. Sensor의 conversion setting을 기록하고, 같은 input을 적어도 완전한 변환 구간의 열 배 이상 연속 실행하여 sustained workload를 만든다. 준비 상태 실행은 입력 event 없이 같은 configuration과 시간 동안 유지한다. 두 조건의 순서를 바꾸어 반복하고, 가능하면 실행 중 여러 sample을 읽어 rail별 평균과 95% interval을 구한다.

[공개 TTFS experiment 코드](https://github.com/JulianGoeltz/fastAndDeep/blob/51cff005767f19ec229cde943931c71578edf661/src/py/fastanddeep/fd_backend.py#L83-L88)는 bus와 shunt를 각각 12 bit, 8-sample averaging으로 설정한다. [INA219 datasheet](https://www.ti.com/lit/ds/symlink/ina219.pdf)에서 이 setting은 channel당 약 4.26 ms이므로 두 channel의 한 갱신은 약 8.52 ms다. 따라서 48 microsecond inference 하나를 분리하지 못하며, 이 setting에서는 최소 약 100 ms의 반복 구간을 사용한다.

$$
E_{\mathrm{Total}}=\frac{P_{\mathrm{run}}}{R},
\qquad
E_{\mathrm{dynamic}}=\frac{P_{\mathrm{run}}-P_{\mathrm{ready}}}{R}.
$$

여기서 $P_{\mathrm{run}}$은 여섯 전원 레일의 합이고, $R$은 hardware timestamp와 완료한 inference 수로 구한 sustained throughput이다. Job queue와 host wall time은 사용하지 않는다. Configuration과 calibration은 별도 구간에서 측정하고 공유 inference 수로 나눈다.

TTFS의 적은 spike 수가 dynamic energy에 미치는 영향은 configuration, 실행시간, route, fan-out, weight와 spike-time 분포를 고정한 채 입력 event rate만 여러 단계로 바꾸어 검정한다. [Crossbar event counter](https://github.com/electronicvisions/pynn-brainscales/blob/b4069f4c00d21151b98b3a4edb3f5bfff6d3e28c/brainscales2/pynn_brainscales/brainscales2/examples/crossbar_event_counter_read.py)로 실제 전달 event 수를 확인하고 power와 event rate의 관계를 적합한다. 그 기울기 $\Delta P/\Delta(\text{event rate})$는 event당 joule 단위를 갖는다. 기울기가 measurement noise와 구분되지 않으면 0으로 두지 않고 95% interval의 upper bound를 보고한다.

전체 Transformer가 한 chip에 배치되지 않으면 대표 $\Phi/\Psi$와 작은 composition에서 이 측정을 반복하고, evaluation trace의 실제 호출 수, 전달 spike 수, fan-out, pulse width, weight와 code window 분포에 결합한다. 현재 repository는 Data SOP와 Global SOP, generic event와 miss count까지만 제공하므로 module별 operator call, active spike, destination fan-out과 pulse-width histogram을 추가로 기록해야 한다. 이 결과는 BrainScaleS-2에서 전체 model을 실행한 측정값이 아니라 component 측정값에 근거한 estimate로 표기한다.

가산식은 측정에 사용하지 않은 두 개 이상의 primitive composition과 chip에 배치 가능한 작은 network에서 예측값과 실제 전원 레일 값을 비교해 검증한다. 검증 오차가 허용 범위를 벗어나면 event당 값을 전체 model에 적용하지 않는다.
