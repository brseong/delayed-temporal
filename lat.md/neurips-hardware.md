# 하드웨어 검증의 주장 범위

현재 ICLR 원고에 적용할 하드웨어 근거의 경계를 정한다. BrainScaleS-2의 당시 기능 조사와 실험 제안은 [[deprecated#과거 하드웨어 검토]]에, 전원 및 절대 에너지 검토는 [[deprecated#폐기한 전원 및 절대 에너지 검토]]에 보관한다.

## 구성 요소와 전체 합성의 구분

구성 요소가 장비에 존재한다는 사실만으로 제안한 연산 전체의 연결이 검증되지는 않는다. 물리 파라미터와 연결 제약은 장비별로 확인해야 한다.

고정 가중치의 적분과 실행 중 다른 전위를 곱하는 동작은 다르다. 부호 처리, 상태 전달과 외부 계산 없이 연산을 연결하는 조건도 별도 근거가 필요하다. [[neurips-current]], [[neurips-mathematics#부호 있는 곱셈과 이벤트 도착 순서]]를 참조한다.

## Pooling의 독립 오차와 공유 오차

동일 논리 뉴런의 복제가 독립 변화를 줄이더라도 공유 편향은 남을 수 있다. 반복 수만으로 독립성을 가정하지 않는다.

뉴런 보정, trial별 변화, 공유 입력과 배치 위치를 구별한다. 복제로 늘어나는 뉴런·시냅스·연결·이벤트 비용도 보고해야 한다. [[neurips-current#노이즈와 비용의 해석 한계]]와 [[deferred-experiments]]에 연결한다.

## 작은 네트워크가 증명하는 범위

Attention 없는 네트워크의 하드웨어 결과는 해당 구성의 효과를 지지할 수 있지만 전체 Transformer 칩 구현의 증거는 아니다.

일반 모델, 이상적 변환, 단일 하드웨어 구현과 복제 구현을 같은 체크포인트·분할에서 구별한다. 출력 투표와 논리 뉴런 복제, 소프트웨어 노이즈 재생과 실제 칩 실행을 혼동하지 않는다. 보정에 평가 정답을 사용하지 않으며 불확실성과 자원 비용을 함께 보고한다. 실제 작업 상태는 [[todo#Manuscript Revision Master Checklist#P0 SOP, Latency, and Hardware Claims]]를 따른다.

## Primitive operator 측정에서 전체 모델 추정으로

Primitive operator의 chip 측정을 전체 변환 모델의 성능 추정에 사용할 때는 실제 chip 실행과 측정값을 사용하는 소프트웨어 평가를 구분해야 한다.

먼저 $\phi_{\mathrm{NL}}$, $\phi_{\mathrm{NP}}$, $\psi_{\mathrm{NE}}$, $\psi_{\mathrm{Int}}$ 각각에 대해 BrainScaleS-2의 회로 설정, 입력과 출력, 초기 상태, 시간 및 potential 단위, calibration 절차를 대응시켜야 한다. 외부 digital controller 또는 host가 spike time 생성, 상태 전달, 적분 종료나 인출을 대신하면 그 부분은 analog primitive의 chip 구현으로 세지 않는다. 특히 pulse-width integration을 측정한 회로 설정과 TTFS operator를 측정한 회로 설정을 동일한 실행에서 연결할 수 있는지는 별도로 입증해야 한다.

측정 결과를 primitive마다 하나의 Gaussian 표준편차로 축약하지 않는다. 입력 구간, weight, $\tau_m$, $\tau_s$, threshold, deadline, 물리 뉴런과 synapse 위치, calibration 및 반복 실험별로 평균 mapping error와 분포를 기록한다. static mismatch, 반복 실행 간 변화, weight quantization, readout saturation, routing 또는 spike loss, deadline miss를 분리한다. 같은 회로 자원이나 기준 event가 만드는 공유 오차도 독립 표본으로 바꾸지 않는다.

전체 모델 평가는 이 측정 분포를 변환 모델의 동일한 primitive 호출 위치에 적용하여 반복한다. 이 결과는 chip에서 전체 모델을 실행한 결과가 아니라 chip 측정값에 근거한 소프트웨어 성능 추정이다. 추정의 타당성은 측정에 사용하지 않은 입력, 두 개 이상의 primitive composition, 가능하면 chip에 배치할 수 있는 작은 변환 네트워크에서 예측 오차와 실제 오차를 비교하여 확인한다.

논문의 근거는 세 단계로 나눈다. Primitive의 mapping과 noise를 실제 chip에서 측정했다는 근거, 그 측정값으로 전체 변환 모델 성능을 추정했다는 근거, 실제로 배치한 네트워크를 chip에서 실행했다는 근거는 서로 대체하지 않는다. Primitive 측정만으로 전체 Transformer의 chip 구현이나 latency를 주장하지 않는다.

## 실험 섹션의 근거 순서

실험은 noise-free 변환 충실도, primitive operator의 chip 측정, 측정 분포를 적용한 모델 강건성 순으로 제시해야 각 결과가 다음 결과의 근거가 된다.

첫 실험은 ANN과 변환 SNN의 noise-free 성능, clipping, SOP를 보고하여 framework 자체의 변환 손실을 고정한다. 둘째 실험은 BrainScaleS-2에서 primitive별 mapping error, spike-time noise, static mismatch와 deadline miss를 측정하여 이후 모델 평가에 사용할 분포를 정한다. 셋째 실험은 이 측정 분포를 동일한 operator 위치에 적용하고 deadline margin을 변화시키며 task 성능과 miss rate를 보고한다.

임의의 Gaussian timing-noise sweep은 chip 측정과 별개의 주요 결과로 두지 않는다. 측정된 조건을 sweep 위에 표시하여 동작점의 위치와 그보다 큰 noise에서의 변화를 보여줄 때만 main text의 보조 분석으로 유지한다. 그렇지 않으면 appendix로 옮긴다. 측정 분포가 Gaussian과 맞지 않거나 입력 및 배치 위치에 의존하면, main result는 측정 분포를 직접 적용한 평가이고 Gaussian timing-noise sweep은 computational stress test로만 남긴다.

Deadline margin sweep은 측정 분포를 적용한 모델 평가에서 한 번만 수행한다. 같은 margin sweep을 임의의 Gaussian 조건과 측정 조건에서 각각 핵심 결과로 제시하면 두 실험의 역할이 중복된다.

새 ViT-B/16 fixed-5k 두 panel figure는 appendix의 simulated sensitivity analysis로 둔다. 첫 panel은 local-window timing-noise fraction, 둘째 panel은 deadline-margin ratio를 보고한다. 제거된 global-range selection panel은 재사용하지 않는다. Main text의 Figure 2는 실제 측정 분포가 생긴 뒤에만 채운다.
