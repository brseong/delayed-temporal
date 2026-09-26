# 수학적 성립 조건과 검증 범위

연산자 합성의 수학적 조건과 실제 구현의 검증 범위를 구별한다. 버전별 오류·수정 상태와 당시 증명 편집안은 [[deprecated#과거 수학 검산]]에 보관한다.

## 구성 가능성과 정확도 보장의 구분

연산을 합성할 수 있다는 사실만으로 다층 모델의 정량적 오차 경계가 얻어지지는 않는다. 국소 오차, 깊이에 따른 증폭과 유효 입력 범위를 함께 확인해야 한다.

Residual의 항등 경로도 오차 전파에 들어간다. Calibration으로 선언 범위를 고정해도 엄밀한 오차 상수가 자동으로 정해지지 않는다. [[decisions]], [[calibration]], [[operators#Operator Validity Conditions]]와 함께 판단한다.

## 로그 적분 계수와 지수 부호

로그 인코딩의 전류 정규화와 적분 계수는 일치해야 한다. 최종 로그는 무차원 비율을 사용하며, 지수 부호는 인코딩과 관측 시각을 대입해 확인한다.

정규화되지 않은 지수 전류의 적분에는 시냅스 시간 상수가 곱해진다. 전류 정의를 바꾸면서 가중치 조건을 그대로 둘 수 없다. 현재 식은 [[domain#TTFS Encoding]]과 [[domain#Temporal-to-Potential Decoding]]을 따른다.

### 거듭제곱 합성의 지수 부호

같은 음의 로그 시각이라도 고정 관측 시각의 지수 응답에 넣는지, 기준 스파이크와의 시간차로 해석하는지에 따라 거듭제곱 지수의 부호가 달라진다.

양의 입력 $v$에 대해 원고의 $\phi_{\mathrm{NL}}(v)=C-\tau_s\log v$를 $\psi_{\mathrm{NE}}(t_{\text{in}})=w\exp(-(t-t_{\text{in}})/\tau_m)$에 대입하면, 고정된 관측 시각 $t$에서

$$
\psi_{\mathrm{NE}}(\phi_{\mathrm{NL}}(v))
=w\exp(-(t-C)/\tau_m)\,v^{-\tau_s/\tau_m}.
$$

따라서 시간상수와 관측 시각에 따른 고정 가중치로 앞의 배율을 제거해도 지수의 음수 부호는 남는다. 이 직접 합성을 양의 지수 mapping과 동일시할 수 없다.

현행 ICLR 부록의 Power Operator는 양의 입력 $v$, 기준값 $R>0$, 지수 $p>0$을 같은 logarithmic domain에서 인코딩한다. 두 spike가 유효한 time window 안에서 전달되는 noise-free 조건에서

$$
\psi_{\mathrm{ED}}\!\left(
\phi_{\mathrm{NL}}(v;p\tau_m),
\phi_{\mathrm{NL}}(R;p\tau_m);\tau_m\right)
=\left(\frac{v}{R}\right)^p.
$$

따라서 receiving synaptic gain $R^p$를 고정하면 $v^p$가 복원되고, Table의 normalized case는 $R=1$이다. 공통 관측 deadline과 양의 finite domain이 필요하며, log input floor나 clamp가 활성화되면 이 identity는 근사로 바뀐다. 일반적인 양의 실수 지수는 양의 입력에서 성립하며, 원고의 음수 입력에 대한 양의 정수 거듭제곱은 별도의 두 부호 경로와 출력 결합을 사용한다. 구현의 관련 구성은 [[operators#Composed Functions#Activations]], 인코딩과 디코딩의 범위 계약은 [[domain#TTFS Encoding]]과 [[domain#Temporal-to-Potential Decoding]]에 있다.

## 부호 있는 곱셈과 이벤트 도착 순서

시간차의 부호를 계산하는 대수와 실제 회로의 전류 공급·제거는 서로 다른 검증 대상이다. 이벤트 누락도 정상적인 마지막 도착과 구별해야 한다.

현재 적분과 전달 상태의 계약은 [[operators#Primitive PWM Integration]], [[operators#Missing-Event Readout]]에 있다. 전류에 이미 포함된 가중치를 다시 곱하지 않는 원칙은 [[neurips-current#적분과 고정 배율]]을 따른다. 수식 평가의 일치만으로 회로 제작을 검증했다고 주장하지 않는다.

## 활성함수의 보정 위치

지수 응답의 크기나 시간 상수가 비선형 함수 내부에 영향을 주면, 최종 출력 뒤의 배율로 일반적으로 상쇄할 수 없다.

지수 응답의 크기는 분모의 상수와 합하기 전에 맞추고 시간 상수는 지수 입력과 디코더의 단위를 함께 확인한다. Softmin의 온도 역할과 활성함수 보정은 구별한다. 현재 합성과 고정 배율은 [[operators#Composed Functions]]를 따르며, 물리적 응답에서 무시한 항의 근사 오차는 별도다.

### GELU 지수 출력의 고정 배율

GELU 분모에 들어가는 지수 출력의 고정 배율은 receiving $\Psi$ gain으로 상쇄할 수 있지만, 상수 $1$을 더하기 전에 적용해야 한다.

고정 관측 시각에서 negative-potential encoder와 exponential decoding을 합성하면

$$
z(v)=w\exp\!\left[-\frac{t_{\mathrm{obs}}-u}{\tau_m}\right]
\exp(-v/\tau_m)=\kappa\exp(-v/\tau_m).
$$

여기서 $u$는 해당 encoder가 선언한 local upper endpoint다. 전체 exponential readout weight를 $w=\exp((t_{\mathrm{obs}}-u)/\tau_m)$로 정하거나, 기존 응답에 $\kappa^{-1}\cdot_{\mkern-2mu\scriptscriptstyle\Psi}$를 적용하면 normalized output은 $\exp(-v/\tau_m)$가 된다. 이 $\cdot_{\mkern-2mu\scriptscriptstyle\Psi}$는 원고가 이미 정의한 fixed receiving synaptic gain이며 별도 $f_{\mathrm{Mul}}$이 아니다. 다만 GELU 식 안에서 지수 입력 계수에 붙은 기존 $\cdot_{\mkern-2mu\scriptscriptstyle\Psi}$와 역할이 다르다. 여기서는 지수 contribution 전체에 적용한 뒤 $1$을 더해야 하며, 완성된 gate나 GELU 출력 뒤의 gain으로 옮길 수 없다. $\tau_m\ne1$에서 목표가 $\exp(-x)$이면 exponential decoding 전에 입력을 $\tau_m x$로 맞춘다.

Composition table의 $f_{\mathrm{Exp}}$ 행에만 $w$나 $\kappa^{-1}\cdot_{\mkern-2mu\scriptscriptstyle\Psi}$를 노출하면 다른 operator mapping과 추상화 수준이 달라진다. 모든 $\Psi$ operator가 fixed receiving synaptic gain을 가진다는 공통 규칙을 먼저 두고, 이 gain은 입력과 무관한 multiplicative prefactor만 상쇄할 수 있다고 한정한다. Primitive table에서는 $\psi_{\mathrm{NE}}$ 행에만 있는 $w$를 제거하며, $\alpha\cdot_{\mkern-2mu\scriptscriptstyle\Psi}z$는 composition에서 함수상 fixed coefficient를 드러낼 때만 쓴다. $f_{\mathrm{Exp}}$에 필요한 gain 값과 GELU 분모에서의 적용 순서는 appendix derivation에 명시하는 편이 일관된다.

물리적 관측 시각을 $t_{\mathrm{obs}}$에서 $t_{\mathrm{obs}}+\Delta$로 옮긴다면 normalized decoding을 유지하기 위해 fixed readout gain을 $\exp(\Delta/\tau_m)$만큼 늘려야 한다. 현행 observation margin은 [[utils/transforms/noise.py#_sample_gaussian_spike_time]]에서 nominal code interval과 receiver cutoff를 분리하고, 허용된 늦은 timestamp를 그대로 보존한다. Physical pulse-width modulation은 늘어난 receiver cutoff를 사용하지만 [[utils/transforms/functions.py#exponential_function]]은 nominal code mapping의 prefactor를 유지한 채 raw timestamp를 디코딩한다. 원고에서는 event 허용 구간, pulse-width readout 시각, exponential decoding 기준을 구별해야 한다.

## Softmax 범위와 공통 배율

양의 점수와 그 합을 모두 표현할 수 있어야 한다. 상한을 낮추는 배율이 작은 점수를 하한 아래로 밀 수 있으므로 양쪽 조건을 함께 검사한다.

배율의 상쇄는 공통 배율, 동일한 분자·분모 조건과 표현 가능한 구간을 전제한다. 입력별 최댓값에 의존하는 식을 사전 고정된 값처럼 설명하지 않는다. 현재 구현은 [[operators#Spiking Attention]], 과거 정밀도 비교의 수치는 [[deprecated#과거 실험과 범위 감사#GPT-2 정밀도 통제 실험]]을 참조한다.

## 정량적 오차 주장과 완성된 조합

현재 서론은 finite ranges가 clipping을 도입한다는 사실을 설명하는 범위로 낮췄으며, Softmax와 LayerNorm의 완성된 합성은 구현과 일치하는 정의역 조건까지 보강해야 한다.

이 논문의 중심은 연산자 합성과 변환 후 task fidelity이므로, clipping error를 정량화하기 위한 별도의 층별 activation 또는 logit 오차 실험은 추가하지 않는다. 서론은 “accounting for clipping introduced by the finite ranges”로 한정하고, noise-free conversion fidelity의 ANN--SNN task metric 차이를 원고의 실증 범위로 삼는다. 기존 clipping count는 구현 진단으로만 유지하며 clipping error를 정량화한 증거로 주장하지 않는다. Softmax와 LayerNorm의 완성된 합성은 정량적 오차 실험과 별개인 구성 완결성 문제이므로 보강한다.

### Softmax composition

Softmax는 public $f_{\mathrm{Div}}$의 ordered positive domain을 만족하므로 그 $[0,1]$ contract를 그대로 사용한다.

$r_{ij}=f_{\mathrm{SDP}}(\mathbf q_i,\mathbf k_j)=-\langle\mathbf q_i,\mathbf k_j\rangle/\sqrt{d_k}$로 두면

$$
e_{ij}=f_{\mathrm{Exp}}(r_{ij};\tau)=\exp(-r_{ij}/\tau),\qquad
a_{ij}=f_{\mathrm{Div}}\!\left(e_{ij},\sum_k e_{ik}\right).
$$

$0<e_{ij}\le\sum_k e_{ik}$이므로 public division의 출력은 $[0,1]$이다. $\tau=1$이면 $a_{ij}$는 ordinary scaled dot-product Softmax weight다. Finite score range, exponential representability, masking과 denominator delivery 조건을 유도와 함께 명시한다.

### LayerNorm composition

LayerNorm은 public $f_{\mathrm{Div}}$가 아니라 양·음 두 경로의 unrestricted $\psi_{\mathrm{ED}}$ 차를 사용하므로 normalized output은 $[0,1]$에 제한되지 않는다.

$\hat x_i=x_i-\mu$, $x_i^+=\max(\hat x_i,0)$, $x_i^-=\max(-\hat x_i,0)$, $v=D^{-1}\sum_i[(x_i^+)^2+(x_i^-)^2]+\epsilon$로 두고, shared positive reference $H$에 대해

$$
t_i^\pm=\tau_s\log(H/x_i^\pm),\qquad
t_v=\frac{\tau_s}{2}\log(H^2/v)=\tau_s\log(H/\sqrt v).
$$

각 signed branch는

$$
\psi_{\mathrm{ED}}(t_i^\pm,t_v;\tau_s)=\frac{x_i^\pm}{\sqrt v},\qquad
z_i=\frac{x_i^+-x_i^-}{\sqrt v}=\frac{\hat x_i}{\sqrt v}
$$

를 만들고 $y_i=\gamma_i z_i+\beta_i$를 적용한다. 따라서 $[0,1]$은 public division과 Softmax weight의 contract이지 LayerNorm normalized output의 범위가 아니다. 실제 finite implementation은 variance stabilizer와 별개인 positive log input floor, inactive branch masking, fixed rails와 output clamping을 포함한다. 구현 경로는 [[operators#Spiking LayerNorm]]에 있다.

## LayerNorm의 이상적 식과 유한 범위 구현

표준편차를 로그 인코딩으로 구성하는 항등식과 유한 범위 구현의 정확도는 구분한다. 수치 안정화, 인코딩 하한, 비활성 부호 경로와 분산 계산의 역할을 따로 확인한다.

상수 입력, 작은 크기, 상한 초과와 단계별 우회를 검증해야 한다. 노이즈가 없더라도 유한 범위 근사는 남을 수 있다. 현재 계약은 [[operators#Spiking LayerNorm]]과 [[evaluation#Fixed-Domain Text-Model Real-Data Audit#Text-Model LayerNorm Execution Path]]에 두고 옛 결함 목록은 재등록하지 않는다.

## 증명 간소화의 보존 조건

단순한 치환을 표나 설명으로 줄이더라도 양수 범위, 공통 배율, 부호 처리와 실패 조건은 보존해야 한다.

비자명한 구성과 성립 조건을 중심으로 본문을 정리한다. 수학적 미해결 문제를 편집으로 숨기지 않는다. 현재 원고 작업 상태는 [[todo#Manuscript Revision Master Checklist#P0 Mathematical and Operator Audit]]를 따른다.

## SwiGLU 합성의 의미 검증

Llama에 연결한 SwiGLU는 이상적인 연산자 의미에서 올바른 합성이지만, 실제 실행에는 입력 범위, 지수 clamp, 자료형, 시간 잡음과 관측 조건이 붙는다. 2026-09-25 검증은 이 조건과 이전 설명의 생략을 구분한다.

### 입력과 연산자의 의미

각 좌표에서 $u$는 gate projection, $v$는 up projection의 출력이고 Llama는 $\beta=1$을 사용한다. SwiGLU 내부의 모든 곱은 원소별 곱이다.

[[utils/transformers/models/spiking_llama/modeling_spiking_llama.py#LlamaMLP#forward]]는 두 projection의 tensor와 선언 범위를 [[utils/transforms/functions.py#swiglu_function]]에 전달하고, 결과를 down projection에 넣는다. PyTorch weight 저장 규약에서는 행 벡터 입력에 weight의 전치를 곱한다. 이전 답변의 $W\mathbf{x}$는 열 벡터 표기일 때 맞는다.

$\phi_{\mathrm{NP}},\phi_{\mathrm{NL}}$은 potential에서 spike time으로, $\psi_{\mathrm{NE}}$는 spike time에서 potential로 간다. $\psi_{\mathrm{Int}}$는 두 시각과 적분 전류를 받아 potential을 만든다. 현재 ICLR 원고는 $\psi_{\mathrm{ED}}$도 primitive interface로 채택했다. 아래의 추가 전개는 그 지위를 바꾸는 정의가 아니라 시뮬레이터가 선택한 내부 구현이다.

### 지수의 관측 시각과 고정 gain

지수 출력의 음수 부호와 보정 gain은 인코더의 구간과 디코더의 관측 시각을 함께 대입하면 정해진다.

이 절에서만 $z$를 지수 인코더에 들어가는 값, $[a,b]$를 그 선언 범위, $T=b-a$를 국소 관측 시각으로 둔다. 연속 시간, 잡음 없는 실수 연산에서는

$$
\phi_{\mathrm{NP}}(z)=b-z,\qquad
\psi_{\mathrm{NE}}(T;\phi_{\mathrm{NP}}(z))
=\exp\!\left(-\frac{T-(b-z)}{\tau_s}\right)
=\exp(a/\tau_s)\exp(-z/\tau_s).
$$

따라서 receiving current에 고정 gain $\exp(-a/\tau_s)$를 적용하면

$$
E=\exp(-a/\tau_s)\,
\psi_{\mathrm{NE}}(T;\phi_{\mathrm{NP}}(z))
=\exp(-z/\tau_s).
$$

실제 [[utils/transforms/functions.py#swiglu_function]]은 먼저 $z=\beta\tau_s u$를 만들고 지수 입력을 $[-20\tau_s,20\tau_s]$로 제한한다. 제한이 작동하지 않을 때에만 $E=\exp(-\beta u)$다. 보정은 반드시 $1+E$를 만들기 전에 적용한다. 완성된 sigmoid 또는 SwiGLU 뒤에 gain을 옮기면 같은 함수가 되지 않는다.

여기서 $a$는 clamp helper가 반환한 실제 인코딩 범위의 하한이다. 구간이 한 점으로 붕괴하거나 자료형의 정밀도로 $1+E$를 구별하기 어려우면 helper가 기존 전체 제한 구간을 사용하므로, 항상 원래 입력 하한에 $\beta\tau_s$를 곱한 값과 같지는 않다. 관련 구현은 [[utils/transforms/functions.py#clamp_sigmoid_exponential_input]], [[utils/transforms/potential_to_spike.py#neg_identity_transform]], [[utils/transforms/spike_to_potential.py#exp_operator]]다.

### Division의 시간 인자와 기본 연산자 전개

분자와 분모의 로그 인코딩은 같은 양의 구간과 시간 상수를 사용해야 하며, 적분 전류의 부호가 division의 방향을 결정한다.

이 절에서 $H=1+\exp(-a/\tau_s)$, $T_0=\tau_s\log H$로 둔다. 분자 $1$과 분모 $1+E$는 동일한 $[1,H]$에서 인코딩한다. 이 양의 유효 구간의 하한 $1$은 로그 변환의 물리적 기준 $V_{\mathrm{lb}}$와 다르다. 현재 로그 변환은 $V_{\mathrm{lb}}=0$에 해당하므로 $X-1$의 로그를 취하지 않는다.

$$
t_X=\phi_{\mathrm{NL}}(1)=\tau_s\log H,\qquad
t_Y=\phi_{\mathrm{NL}}(1+E)=\tau_s\log\frac{H}{1+E}.
$$

따라서 $t_Y\le t_X$이고

$$
G=\psi_{\mathrm{ED}}(t_X,t_Y)
=\exp((t_Y-t_X)/\tau_s)=\frac{1}{1+E}.
$$

시뮬레이터의 내부 구현을 전개하면, $\psi_{\mathrm{Int}}(t_X,t_Y;c)=c(t_Y-t_X)$라는 적분 규약에서

$$
p=\psi_{\mathrm{Int}}(t_X,t_Y;-1)=t_X-t_Y,\qquad
s=\phi_{\mathrm{NP}}(p)=T_0-p
$$

로 두어야 한다. 여기서 내부 potential 구간은 $[-T_0,T_0]$, 새 인코딩 구간은 $[0,2T_0]$다. 따라서

$$
G=\exp(T_0/\tau_s)\,
\psi_{\mathrm{NE}}(2T_0;s)
=\exp(-p/\tau_s)=\frac{1}{1+E}.
$$

[[utils/transforms/spike_to_potential.py#exponential_difference_operator]]는 실제로 전류 $-1$을 사용한다. 내부의 정규화된 지수 함수는 시각에서 고정 offset을 뺀 뒤 지수를 평가하는 대수 구현이며, 위 식은 이를 음수가 아닌 국소 spike time과 고정 gain으로 다시 표현한 것이다.

NeurIPS 보관본 composition table의 $\psi_{\mathrm{Int}}(t_1,t_2;1)$는 그 적분 정의 및 출력 $\exp((t_2-t_1)/\tau_s)$와 부호가 맞지 않는다. 이는 현재 ICLR의 오류가 아니다. ICLR은 ED를 primitive interface로 두고 있으며, 이번 보완에서는 부록에 실제 구현의 $-1$ 전류, 국소 시간 구간과 고정 gain을 명시했다. NeurIPS 보관본은 수정하지 않는다.

### 두 곱셈과 인과적 해석

두 곱셈은 두 번째 피연산자를 시간으로 인코딩하고 첫 번째 피연산자를 적분 전류로 사용한다. 음수 결과에는 부호를 표현하는 두 적분 경로가 필요하다.

선언 구간을 공유하는 데이터 시각과 영점 기준을 사용하면

$$
S=\psi_{\mathrm{Int}}(\phi_{\mathrm{NP}}(G),\phi_{\mathrm{NP}}(0);u)=uG,\qquad
y=\psi_{\mathrm{Int}}(\phi_{\mathrm{NP}}(S),\phi_{\mathrm{NP}}(0);v)=vS.
$$

이 식의 $\phi_{\mathrm{NP}}$는 각 곱셈마다 해당 두 번째 피연산자의 국소 구간을 사용한다. 두 호출이 전역 threshold나 동일한 deadline을 공유한다는 의미가 아니다. 각 구간은 영점 기준을 포함하도록 필요한 만큼 확장된다.

$S<0$이면 데이터 시각이 영점 기준보다 늦다. 단일 양의 시간 구간에 대한 적분만으로는 이 부호가 나오지 않는다. 현재 [[utils/transforms/primitive.py#signed_pulse_width_modulation_operator]]는 [[operators#Primitive PWM Integration#Differential Physical Realization]]의 $Q^+$와 $Q^-$를 명시적으로 계산해 뺀다. 두 시각의 순서를 미리 판정하지 않으며, 미도착 event의 적분량은 각각 0이다.

두 accumulator의 전류는 모두 음수가 아니고, 부호는 마지막 차에서 복원한다. event 두 개와 전류의 양·음수 부분이 만드는 연결은 네 가지이지만 accumulator는 두 개다. 이는 추가 spike 네 개를 요구한다는 뜻도, 물리 회로 비용이 검증됐다는 뜻도 아니다.

적분 중 일정한 전류와 입력과 무관하게 고정된 synaptic gain은 구분한다. 두 곱셈의 전류는 각각 $u$, $v$에 비례하며 다음 적분 동안 유지되어야 한다. 반면 지수 보정 gain과 $\beta\tau_s$는 선언 범위 및 설정으로 정해지는 고정 계수다.

식의 potential과 time은 정규화된 좌표다. $\phi_{\mathrm{NP}}$의 단위 기울기와 $I(v)=v$는 선택한 변환 배율을 전제하며, 물리 단위인 전압과 시간을 그대로 동일시하지 않는다.

각 합성 단계의 시각은 국소 시각이다. 다음 단계 인코딩은 이전 potential이 준비된 뒤 시작해야 한다. Tensor 합성에는 이 순서가 있으나 물리 회로의 전체 시간 배치, potential 유지, 전압에 비례하는 적분 전류 생성까지 검증한 것은 아니다. 실제 생물학적 응답의 근사를 수학적 등호와 동일시하지 않는다.

### 유한 범위와 수치 반례

입력이 선언 범위 안에 있고 지수 clamp가 비활성이어도 유한 자료형에서는 반올림 오차가 남는다. 잡음 또는 이산 시간 실행에서는 이상적인 항등식이 추가로 달라진다.

연속 시간의 실수 연산에서, 출력 clamp를 적용하기 전 목표는
$vu\,\sigma(\min(20,\max(-20,\beta u)))$다. 실제 경로는 여기에 sigmoid의 $[0,1]$ clamp, 내부 Swish 범위 clamp, 두 곱셈의 출력 범위 clamp를 적용한다. 최종 SwiGLU에는 $v$의 부호와 범위가 함께 반영되므로 내부 Swish의 고정 하한을 그대로 붙일 수 없다.

Gaussian 실행은 스파이크 시각 및 전달 여부를 소비한다. 이벤트가 누락되면 지수 응답과 적분 경로가 reset 값을 사용하므로 위의 표준 sigmoid 등식이 유지되지 않는다. 또한 Gaussian 지수 디코더는 명목 code deadline을 기준으로 하고 적분은 수신 관측 시각을 쓰므로, 수신 허용 시간을 늘렸다고 모든 지수 응답의 관측 기준까지 같아지는 것은 아니다.

2026-09-25에 Python 3.12 환경에서 연속 시간, 잡음 비활성 조건으로 실제 함수를 검사했다. $u\in[-3,4]$, $v\in[-2,3]$를 각각 257점으로 함께 변화시키고 $\beta\in\{0,0.7,1,-1\}$, $\tau_s\in\{0.5,1,2\}$를 조합한 12개 경우에서 $vu\sigma(\beta u)$ 대비 최대 절대오차는 float64에서 $1.07\times10^{-14}$, float32에서 $3.81\times10^{-6}$였다. 모든 반환값은 선언 출력 범위 안에 있었다. 이는 해당 점들에 대한 검사이지 전역 오차 보장은 아니다.

$u=-40$, $v=1$, $\beta=\tau_s=1$, 입력 구간 $[-40,40]$에서는 float64 출력이 약 $-8.2446\times10^{-8}$이지만 표준 SwiGLU는 약 $-1.6993\times10^{-16}$이다. 지수 입력의 $-20$ clamp에 대한 실제 반례다. 같은 구간의 float32 실행에서는 $u=-20$의 출력이 $0$으로 반올림되었고 float16은 지수 endpoint underflow 때문에 거부되었다. bfloat16에서는 $u=-10$도 $0$이 되었다. 따라서 큰 Llama 체크포인트의 낮은 정밀도 실행을 작은 모델의 float32 검사로 보장할 수 없다.

$t_A=2,t_B=1,\tau_s=1$에서 실제 exponential difference는 $e^{-1}=0.367879\ldots$이며, 적분 전류를 $+1$로 바꾼 구성은 $e=2.718281\ldots$다. 별도의 비대칭 입력 세 점에서는 Gaussian 잡음 표준편차를 $0$으로 설정한 출력과 deterministic 출력이 동일했고 반환 범위도 같았다.

### 검증 결론

SwiGLU의 함수 합성 순서, 시간 인자 방향, 고정 지수 보정과 두 곱셈은 올바르다. 이전 설명은 적용 조건과 내부 합성을 생략하여 실제 코드보다 강한 동일성을 주장했다.

정확한 등호는 유효 선언 범위, 비활성 clamp, 공통 로그 기준, 전달된 이벤트, 연속 시간, 정확한 실수 연산과 명시된 current 정규화를 전제한다. 실제 구현은 그 조건하의 수치 근사이며, 전체 Llama의 변환 완료나 물리 회로 구현을 증명하지 않는다. 관련 후속 조치는 [[todo#Manuscript Revision Master Checklist#P0 Mathematical and Operator Audit]]에서 관리한다.

### 재현 가능한 검증

2026-09-25 보완은 기호 항등식, 독립적인 부호 조합, event 전달 여부 및 기존 합성 검사를 함께 사용한다. 유한 범위 조건은 ICLR의 기존 설명을 유지한다.

[[scripts/verification/verify_swiglu_operator_form.py#verify_symbolic_identities]]는 지수 보정, ED 부호, 공통 로그 기준과 두 accumulator의 차를 기호적으로 확인한다. [[scripts/verification/verify_swiglu_operator_form.py#verify_signed_accumulators]]는 전류의 세 부호, event의 모든 전달 조합과 순서를 별도의 순차 적분 기준으로 검사한다. [[scripts/verification/verify_swiglu_operator_form.py#verify_exponential_difference_form]]는 ED의 두 시간 순서와 세 시간 상수를 검사한다.

[[scripts/verification/verify_swiglu_operator_form.py#verify_swiglu_form]]은 $u$의 양수·음수·혼합 구간 세 개, $\beta\in\{-1,0,0.7,1\}$ 및 $\tau_s\in\{0.5,1,2\}$의 36개 조합에서 서로 독립적으로 $u$ 33점과 $v$ 5점을 검사한다. 자료형별 5,940개 값의 최대 절대오차는 float64 $1.07\times10^{-14}$, float32 $3.34\times10^{-6}$이고 모든 출력은 반환 범위 안에 있다.

기존 raw timestamp 검사 일곱 개와 Gaussian 시간 상수·곱셈·ED·division·SwiGLU 검사 다섯 개가 통과했다. 이산 시간의 encoder·구간·적분·지수·모델 kernel·비활성 모드 검사는 통과했지만 전체 `verify_clock_driven.py` 실행은 무관한 ViT 검사 fixture에 `evaluation_samples`가 없어 중단됐다. 이 실패는 전체 검사를 통과한 것으로 보고하지 않는다.

`verify_sop.py`의 24개 검사는 기존 operation-count 산술의 일관성만 확인하며, 두 accumulator의 물리 비용을 증명하지 않는다. 현재 ICLR 부록 `SwiGLU Composition`에 같은 operator form, 국소 단계 순서와 potential을 전류로 공급하는 가정을 반영했다.

작은 Llama의 HF 출력 비교, 저장 및 재로딩 검사도 통과했다. 별도 체크포인트의 저정밀도 성능 검증은 이번 연산자 의미 검증의 완료 조건이 아니다.
