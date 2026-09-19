# 원고 작성과 비교 원칙

현재도 유효한 용어·근거·비교 원칙을 유지한다. 옛 원고의 줄 번호, 오류 개수, 인용 누락과 절별 편집 의견은 [[deprecated#과거 원고 검토]]에 보관한다.

## 용어 감사와 표기 원칙

약어·기호·단위·측정 위치·분모를 독자가 복원할 수 있어야 한다. 구현 문서나 출력에서 쓰는 표현을 승인 없이 논문 정의로 가져오지 않는다.

전위 스칼라, 활성 텐서와 attention value를 구별하고 원고의 승인된 표기를 먼저 대조한다. 정의가 없으면 설명을 풀어 쓰거나 승인을 받는다. [[evaluation#Manuscript Terminology and Notation Check]]에 따라 편집 전 후보와 편집 후 정확한 추가 diff를 검사한다.

## 선행연구와 검증 범위

연산자 형식의 구성, 물리적 해석, 텐서 평가와 실제 칩 검증은 별도 근거다. 비교하지 않은 회로 복잡성이나 생물학적 우월성을 단정하지 않는다.

새 원고에 사용할 선행연구 주장은 원문을 다시 확인한다. 기존 노이즈·시간창·마진 연구와 구별할 실험 조건을 명시하고, 오래된 비교표를 최신 문헌 조사로 제시하지 않는다. [[neurips-hardware]]가 하드웨어 근거의 경계를 정한다.

## 모델과 뉴런 정의의 적용 범위

평가한 모델 종류와 실제 교체한 모듈을 명시한다. 서로 다른 normalization 순서나 뉴런 방정식의 해를 같은 것으로 설명하지 않는다.

실제 activation·embedding·head의 구현 범위는 [[models]]와 [[operators]]를 따른다. LIF와 IF의 해, 초기조건, 전류 정규화와 시간 상수의 근사 조건을 구별한다. 텐서로 수식을 계산했다는 사실만으로 회로 동역학을 검증했다고 주장하지 않는다.

## 실험과 출판의 근거 조건

동일한 코드·체크포인트·데이터·집계 조건의 결과를 비교한다. 결정론적 근사와 노이즈, 일반 모델과 변환 모델의 차이를 구분하고 실패 조건도 보존한다.

노이즈 분포와 이벤트 전달 규칙은 [[noise]], 수치와 통계의 계약은 [[evaluation]]을 따른다. 다른 버전의 정확도나 배치 평균 loss를 조건 확인 없이 합산하지 않는다. 증명 간소화는 [[neurips-mathematics#증명 간소화의 보존 조건]], 완료 상태는 [[todo#Manuscript Revision Master Checklist]]를 따른다.

## NeurIPS 부록 재사용 판정

NeurIPS 부록은 현재 ICLR 정의와 실험 계약을 기준으로 선별하며, 과거 문장을 그대로 복사하지 않는다.

가장 우선할 자료는 primitive operator와 composition의 유도이다. 다만 현재의 $f_{\mathrm{Pow}}$, GELU, LayerNorm, finite potential range에 맞추어 다시 써야 하며, 제거되거나 변경된 구성과 exactness 주장은 이어받지 않는다.

Experimental Details and Hyperparameters는 checkpoint, dataset split, calibration subset, seed, histogram, 고정된 양의 log 입력 하한, noise replica를 보고하는 재현성 표로 재구성한다. 과거의 $\theta=2000$과 완료되지 않은 sweep 조건은 사용하지 않는다.

Simulated Timing Noise Sweeps에는 현재 부록 그림을 재현하는 grid, subset, seed, replica, aggregation을 덧붙일 수 있다. 이 절은 BrainScaleS-2 측정과 분리한다.

Natural Language Performance Details는 이미 본문 표에 있으므로 중복하지 않는다. 기존 모듈 단위 proof, theorem, 아키텍처별 energy estimate는 현재 정의와 충돌하거나 현행 SOP 부록으로 대체되었으므로 가져오지 않는다. Compute Resources는 최종 실행 환경을 다시 수집한 뒤에만 작성한다.

실제 이전에서는 현행 코드와 대수적으로 일치하는 primitive 및 composition 식만 다시 유도하고, 평가에 사용한 checkpoint와 data population을 현행 artifact로 교체했다. 옛 robustness protocol은 현재 그림과 일치하지 않아 사용하지 않았다.

현행 ImageNet-1k 및 CIFAR-10 ViT 성능 artifact는 논문 평가 runner가 런타임에 적용한 Power cubic을 사용하므로 이 차이 때문에 재실행할 필요가 없다. 기존 simulated timing noise 그림도 Power cubic은 사용했지만 고정 GELU 계수를 별도 composed multiplication으로 계산하여, 그 그림만 current receiving synaptic gain 구성과 일치하지 않는다. RoBERTa 평가는 유지되는 multiplication cubic을 사용한다.

현행 ViT-B/16 근거는 source `f7b74c1aef38502caccf532d1e58a7cf321833d6`의 policy-2 campaign이다. Training 5k에서 $\theta=20$을 선택하고 fixed validation 5k에서 threshold, timing noise, deadline margin ratio를 각각 아홉 점으로 평가했다. Timing-noise 상단 네 점을 같은 identity로 추가해 $r_t=10^{-3}$까지 확인했다. Source `648af9bb`의 threshold-40 그림은 역사 자료이며 새 결과와 합치지 않는다.
