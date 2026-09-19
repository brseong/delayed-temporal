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

## Softmax 범위와 공통 배율

양의 점수와 그 합을 모두 표현할 수 있어야 한다. 상한을 낮추는 배율이 작은 점수를 하한 아래로 밀 수 있으므로 양쪽 조건을 함께 검사한다.

배율의 상쇄는 공통 배율, 동일한 분자·분모 조건과 표현 가능한 구간을 전제한다. 입력별 최댓값에 의존하는 식을 사전 고정된 값처럼 설명하지 않는다. 현재 구현은 [[operators#Spiking Attention]], 과거 정밀도 비교의 수치는 [[deprecated#과거 실험과 범위 감사#GPT-2 정밀도 통제 실험]]을 참조한다.

## LayerNorm의 이상적 식과 유한 범위 구현

표준편차를 로그 인코딩으로 구성하는 항등식과 유한 범위 구현의 정확도는 구분한다. 수치 안정화, 인코딩 하한, 비활성 부호 경로와 분산 계산의 역할을 따로 확인한다.

상수 입력, 작은 크기, 상한 초과와 단계별 우회를 검증해야 한다. 노이즈가 없더라도 유한 범위 근사는 남을 수 있다. 현재 계약은 [[operators#Spiking LayerNorm]]과 [[evaluation#Fixed-Domain Text-Model Real-Data Audit#Text-Model LayerNorm Execution Path]]에 두고 옛 결함 목록은 재등록하지 않는다.

## 증명 간소화의 보존 조건

단순한 치환을 표나 설명으로 줄이더라도 양수 범위, 공통 배율, 부호 처리와 실패 조건은 보존해야 한다.

비자명한 구성과 성립 조건을 중심으로 본문을 정리한다. 수학적 미해결 문제를 편집으로 숨기지 않는다. 현재 원고 작업 상태는 [[todo#Manuscript Revision Master Checklist#P0 Mathematical and Operator Audit]]를 따른다.
