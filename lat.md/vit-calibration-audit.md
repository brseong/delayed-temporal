# ViT Calibration Coverage Audit

이 문서는 2026-09-15의 초기 48/96-site ViT 비교를 감사한 역사 기록이다. 현재 policy-2 계약과 결과는 [[vit-calibration-policy2]]와 [[conversion-comparison]]을 따른다.

## Scope and Evidence

2026-09-15의 초기 조사는 당시 비교 실험 source와 calibration 및 평가 로그를 대상으로 했다. 이후 policy 2가 Q/K/V와 centered LayerNorm 범위를 추가했고, 현재 policy 3은 TTFS output head identity까지 요구하므로 아래의 누락 판정은 현재 구현 상태가 아니다.

태그는 `vit_conversion_comparison_theta40_calibrated_float64_bounds3_v1`, 평가 source는 `2c0fbd3f6fd1d043ffc34295f891d4bccaa5113a`이다. 고정 checkout은 `/data/delayed-temporal-worktrees/vit-conversion-comparison`이며, 조사 시점 main의 연산·ViT adapter·calibration 구현도 이 source와 같았다. float64, theta 40, 모든 spiking LayerNorm 단계·attention·MLP 활성화, 잡음과 deadline margin 비활성화 조건을 확인했다.

원시 근거는 `artifacts/logs/conversion_comparison/vit_conversion_comparison_theta40_calibrated_float64_bounds3_v1/`의 모델별 calibration, collection 결과, SNN 로그이다. 네 calibration 결과의 table·log SHA-256과 metadata, 전체 block별 site 이름을 검증했다. 완료된 CIFAR ViT-S·ImageNet ViT-S/B의 최종 count를 확인했으며, ImageNet ViT-L SNN의 최종 count는 조사 시점에 아직 없었다.

| 모델 | Encoder block | 실제 calibration site | 별도 내부 calibration이 없는 LayerNorm |
|---|---:|---:|---:|
| CIFAR-10 ViT-S | 12 | 48 | 25 |
| ImageNet ViT-S | 12 | 48 | 25 |
| ImageNet ViT-B | 12 | 48 | 25 |
| ImageNet ViT-L | 24 | 96 | 49 |

LayerNorm 수는 block당 두 개와 마지막 한 개를 포함한다. Calibration site 수가 맞는 것은 등록한 네 지점이 완전하다는 뜻이며, 모든 내부 범위를 수집했다는 뜻이 아니다.

## Bound Selection and Intended Clipping

추가로 선택할 범위와 그 범위에서 유도되는 중간 구간을 구별한다. 같은 제한된 전위로 제곱·분산과 log 연산을 수행하는 것은 사용자가 확인한 의도된 clipping이며 독립적인 구현 오류로 분류하지 않는다.

현재 실행되는 네 ViT의 hidden 경로에서 추가 범위 선택을 검토할 대상은 Q/K/V projection 출력과 각 LayerNorm의 평균을 뺀 magnitude 상한이다. Attention score는 이미 calibration 대상이므로 새 site 추가가 아니라 통계 구간을 다시 theta로 제한하는 규칙의 문제이다. 이것은 점검 범위의 정리이며 아직 새로운 calibration 정책을 구현한 것은 아니다.

| 대상 | 범위 선택 또는 전파 방식 |
|---|---|
| Q/K/V projection 출력 | 현재 독립 calibration 없이 theta로 제한한다. 각각 수집할지 공통 구간을 사용할지 정한 뒤 attention에 일관되게 전달해야 한다. |
| Attention score | 기존 calibration을 유지하되 추가 theta 상한의 선택 근거를 정해야 한다. 수치 표현에 필요한 별도 제한과 혼동하지 않는다. |
| 각 LayerNorm의 평균을 뺀 magnitude | 새로 층별 상한을 선택한다면 여기가 대상이다. Block당 두 곳뿐 아니라 마지막 LayerNorm도 포함하여 S/B는 25곳, L은 49곳이다. |
| LayerNorm 제곱·분산·log 입력·시간창 | 선택한 magnitude 상한, 고정된 양의 하한과 checkpoint eps로 구간을 유도한다. 각 중간 tensor에 독립 calibration을 중복 추가하지 않는다. |
| LayerNorm 평균·정규화 출력·affine 출력 | 평균은 입력 구간, 정규화 출력은 feature 수, affine 출력은 고정 파라미터로 bound를 계산한다. 내부 magnitude 선택과 구별한다. |
| Attention 지수·합·division·weight·가중합 | Score 범위와 token 수에서 중간 구간을 계산하고 weight에는 고정 출력 범위 $[0,1]$을 쓴다. 가중합은 선택한 V 구간에서 유도한다. |
| 두 residual 합과 첫 MLP Linear 출력 | 이미 calibration에 연결돼 있다. 앞선 projection의 해석적 전파와 함께 유지한다. |
| GELU 내부·출력 및 나머지 경로 | 앞선 입력 구간과 이미 정한 연산 제한·출력 bound를 사용한다. 이번 범위 목록에 추가 독립 수집 대상으로 넣지 않는다. |

평균을 뺀 값 $[-45,15,15,15]$을 theta 40에서 제한하면 분산이 675에서 568.75로 바뀌는 것은 그 표현 범위의 의도된 영향이다. 포화 전 값을 별도로 보존해 분산을 계산하도록 바꾸지 않는다. 남은 선택은 상한이 충분한지이며 이 예 자체를 추가 실험 의무나 프로토콜 위반으로 세지 않는다.

### Epsilon Correction and Preserved Results

사용자의 후속 요청에 따라 main의 ViT 생성자에서 checkpoint eps를 전달하도록 수정했다. Bound 선택이나 clipping 동작은 바꾸지 않았고 기존 비교 실험도 재실행하지 않았다.

두 block LayerNorm과 마지막 LayerNorm이 모두 `config.layer_norm_eps`를 받는다. 현재 네 checkpoint의 값은 `1e-12`이며, log 입력의 양의 하한 `1e-5`와는 다른 설정이다. Calibration metadata에도 eps를 기록하여 누락되거나 다른 값인 기존 table의 재사용을 거부한다. 상세 계약과 CPU 검증은 [[calibration#Layer-wise Calibration#Frozen Execution#ViT LayerNorm Epsilon]]에 기록했다.

아래 초기 감사의 eps 불일치는 고정 source `2c0fbd3f6fd1d043ffc34295f891d4bccaa5113a`와 그 완료 결과에 대한 설명이다. 새 main 수정으로 과거 정확도가 바뀌거나 개선됐다고 주장하지 않는다. Q/K/V 및 LayerNorm 내부 calibration 추가와 attention score 상한 변경은 아직 구현하지 않았다.

## ICLR Methodology Comparison

수집 조건과 추론 중 고정 적용은 일치하지만, 최종 범위 선택과 checkpoint 설정까지 원고·비교 조건과 같다고 볼 수는 없다. 아래 전수 판정은 추가 theta 제한과 LayerNorm 산술을 별도로 확인한다.

사용자가 명확히 한 첫 번째 경우는 함수 자체가 고정된 출력 범위를 가지는 경우이다. Softmax의 $[0,1]$처럼 입력 activation 구간이나 관측 분포를 바꾸어도 유지되는 범위를 뜻한다. 제한된 입력에서 유한한 구간이 계산된다는 사실이나 calibration 후 범위를 고정한다는 사실만으로 첫 번째 경우에 해당하지 않는다. GELU는 상한이 입력 구간에 의존하므로 첫 번째가 아니라 입력 범위를 이용하는 별도 경우이다. LayerNorm의 정규화 출력과 평균을 뺀 내부 magnitude도 구별하며, 전자의 고정 bound가 후자의 범위까지 고정해 주지는 않는다.

2026-09-15에 확인한 `paper/iclr_2027/iclr2027_conference_methodology.tex`의 79행은 세 유형 중 두 번째인 증폭되는 유한 구간에만 calibration을 적용한다고 명시한다. 세 번째인 log 인코딩에는 먼저 표현 가능한 입력 구간을 정하도록 하며, 그 구간을 반드시 층별 통계로 선택하라고 요구하지 않는다. 같은 원고의 `iclr2027_conference_experiment.tex` 18행은 LayerNorm의 양의 log 입력 상한을 theta로 고정한다고 명시한다.

따라서 LayerNorm 내부 calibration 연결이 없다는 코드 관측은 유효하지만, 이를 현재 ICLR 방법론의 필수 절차가 빠진 것으로 단정해서는 안 된다. 앞선 점검의 우선 보완 판단은 정확도 하락에 대한 진단이며 원고 준수 판정과 다르다. 고정 상한이 현재 체크포인트에 충분한지 평가하는 문제와, 그 상한을 데이터로 선택하도록 방법을 확장하는 문제도 별개이다.

### Matching Procedure

잡음 없는 변환 SNN에서 training subset으로 통계를 수집하고, 대칭 구간에 여유를 더한 뒤 추론 동안 고정하는 절차는 현재 구현과 일치한다.

현재 비교 실험은 training seed-0 5k에서 최솟값·최댓값과 histogram을 수집한다. 일반 대칭 정책은 관측 양 끝점의 최대 절댓값으로 대칭화한 뒤 전체 폭의 5%를 양쪽에 각각 더한다. Attention score에는 그 뒤 theta·수치 상한을 적용하는 예외가 있으므로 방법론 87행만으로 모든 최종 구간을 재현할 수 없다. 정확도를 보며 범위를 반복 최적화하지 않고 저장된 범위를 고정 적용하며 초과값을 clamp한다. 양의 log 하한은 별도 고정값이다.

네 calibration site는 두 residual 합, 첫 MLP Linear 출력인 GELU 입력, attention score이다. 이는 증폭되는 구간의 선택된 경계에 해당한다. GELU 입력을 calibration하는 것은 GELU 출력의 해석적 bound를 유지한다는 방법론과 충돌하지 않는다. 모든 Linear나 Q/K/V를 각각 독립적으로 수집하라는 요구도 79행에는 없다.

### Scope and Reporting Differences

원고의 일반 설명만으로 실제 수집 단위와 최신 비교 실험 설정을 정확히 알 수는 없다. 이 차이를 실험 구현 오류와 구별해 보고한다.

| 원고 서술 | 실제 구현과의 대조 |
|---|---|
| 방법론 87행의 각 뉴런 범위·분포 수집 | 개별 뉴런마다 별도 구간을 추정하지 않는다. 등록한 site의 tensor 전체 원소를 모아 하나의 범위와 histogram을 공유한다. 개별 뉴런별 추정으로 읽히지 않도록 수집 단위를 명시할 필요가 있다. |
| 방법론의 실제 ViT 수집 위치 | 89–90행에 block당 네 지점과 48개 구간 설명이 있지만 주석 처리되어 PDF에는 나오지 않는다. 실제 S/B 48개, L 96개 적용 범위가 본문에 명시되어 있지 않다. |
| 실험 절의 기본 1,024장 및 기존 ViT-B noise calibration 비활성 서술 | 기본값·기존 noise 실험 설명이지, 새 네 모델 비교의 training 5k·calibration 활성 조건을 기록한 것이 아니다. 새 결과를 반영할 때 해당 실험의 별도 조건을 명시해야 한다. |
| LayerNorm의 고정 log 입력 범위 | 현재 구현과 일치한다. 그 상한에서 clipping과 정확도 손실이 생길 수 있다는 사실이 곧 방법론 미준수를 의미하지는 않는다. |

공유 범위 수집은 [[utils/transforms/calibration.py#observe_calibration_activation]]와 [[utils/transforms/calibration.py#update_min_max_observer]]의 site별 scalar 통계에서 확인했다. 실제 실행의 training 5k와 min/max·5% 설정은 `scripts/experiments/vit_comparison.py#evaluator_command` 및 앞 절의 네 calibration artifact 검증에 근거한다.

이번 대조에서는 원고, 코드, 실험 조건을 수정하지 않았다. LayerNorm 내부 calibration을 새로 도입하려면 고정 상한 설정과 구별하여 방법·인코딩 범위·실험 계약을 함께 정해야 하며, 이 문서는 그 변경을 승인하지 않는다.

## Protocol Compliance Findings

현재 네 모델의 patch 입력부터 classifier까지 범위의 출처와 적용을 대조했다. 통계 수집 절차의 일치만으로 전체 프로토콜 준수를 선언하지 않으며, 직접 불일치·명시되지 않은 추가 제한·허용된 처리를 나눠 판정한다.

### Findings That Affect Interpretation

고정 실험 source에서 attention의 최종 범위에는 통계·여유 규칙 외의 제한이 적용됐고 LayerNorm은 checkpoint와 다른 eps를 사용했다. 의도된 magnitude clipping은 이 설정 불일치와 구별한다.

| 대상 | 현재 처리와 근거 | 판정 |
|---|---|---|
| Attention score의 최종 calibration 범위 | min/max 대칭화와 양쪽 5% 여유 뒤 반경을 다시 theta 40 이하로 제한한다. ImageNet ViT-B block 0은 약 140.7337에서 40으로 줄어든다. | 방법론 87행의 규칙만으로는 최종 범위를 재현할 수 없다. 본문에 없는 추가 범위 선택 규칙이다. |
| LayerNorm eps | 고정 실험 source는 checkpoint의 `layer_norm_eps=1e-12`를 전달하지 않아 기본 `eps=1e-5`를 사용했다. | Calibration 여부와 별개인 ANN/SNN 설정 불일치였다. 후속 main 수정은 위에 기록하며 과거 결과는 보존한다. |
| Q/K/V 범위 전달 | Linear가 계산한 Potential domain을 attention에 전달하지 않고 각각 `[-40,40]`으로 교체한다. 별도 Q/K/V calibration은 없다. | 증폭되는 경계에서 층별 통계 기반 범위 선택을 거치지 않는 추가 고정 제한이다. 함수 자체의 고정 출력 범위가 아니며 활성 방법론에 선택 근거가 없다. |
| LayerNorm magnitude·분산 | 평균을 뺀 magnitude를 40으로 제한하고 같은 제한된 값으로 제곱·분산과 log 연산을 수행한다. | 선택한 표현 범위에서 의도된 clipping이다. 독립적인 구현 오류나 프로토콜 위반이 아니며 동작을 유지한다. |

Q/K/V 제한은 S/B에서 모델당 36개, L에서 72개의 projection 경계에 있다. 현재 원고의 필요한 층이라는 표현은 독립적인 calibration site를 전부 열거하지 않으므로, 그 문장만으로 모든 Linear에 새 수집이 필수라고 결론내리지 않는다. 그러나 별도 통계나 해당 bound의 해석적 근거 없이 고정 theta를 적용하는 동작을 첫 번째 유형 또는 이미 적용된 calibration으로 설명하는 것은 맞지 않는다.

코드 근거는 [[utils/transforms/calibration.py#select_calibration_policy_range]], [[utils/transformers/integrations/spiking_sdpa_attention.py#attention_score_representability_bounds]], [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTSelfAttention#forward]], [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTLayer#__init__]] 및 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#forward]]이다.

### Statistical Ranges Replaced by Theta

Attention에서 추가 상한이 실제 적용된 층을 최종 calibration artifact로 확인했다. 수치 표현에 필요한 제한과 theta 선택을 구별한다.

| 모델 | Block 번호 | 관측 min/max와 5% 여유에서 얻는 대칭 반경 | 저장된 반경 |
|---|---:|---:|---:|
| ImageNet ViT-S | 1 | 49.8168 | 40 |
| ImageNet ViT-B | 0 | 140.7337 | 40 |
| ImageNet ViT-B | 1 | 49.9589 | 40 |
| ImageNet ViT-L | 0 | 106.4080 | 40 |

CIFAR ViT-S에는 이 상한에 도달한 calibration site가 없다. ViT-B block 0의 수집 extrema 자체가 약 $[-127.940,96.153]$이므로 저장 범위 $[-40,40]$는 모든 training 관측값을 포함하지 않는다. float64, 시간상수 1, token 수 197에서 수치 조건만으로 계산한 반경은 약 350.5566이다. 따라서 위 사례의 40 제한은 float64 때문에 불가피한 값이 아니다.

이 예외는 구현과 table에 기록돼 있지만, 방법론 89–90행의 관련 설명은 주석이라 PDF에 표시되지 않는다. 이전 수집 절차 일치 판정을 모든 최종 endpoint가 원고의 min/max·여유 규칙 그대로라는 의미로 사용하지 않는다.

### Boundary Checks Without Model Evaluation

추가 데이터 평가나 GPU 실행 없이 파라미터의 구간 산술과 작은 CPU 입력으로 경계 조건을 확인했다. 이 결과는 전체 정확도 또는 clipping 발생 빈도의 대체 근거가 아니다.

네 image processor의 pixel 구간은 $[-1,1]$이다. Patch projection 가중치의 부호별 구간 합에 class token과 위치 embedding을 포함하면 다음 전체 embedding 구간을 얻는다.

| 모델 | 파라미터와 pixel 구간에서 계산한 embedding 범위 | Encoder 입구 |
|---|---|---|
| CIFAR ViT-S | $[-37.6915,37.8348]$ | $[-40,40]$ |
| ImageNet ViT-S | $[-37.6828,37.8305]$ | $[-40,40]$ |
| ImageNet ViT-B | $[-18.7445,16.4177]$ | $[-40,40]$ |
| ImageNet ViT-L | $[-28.3493,16.3273]$ | $[-40,40]$ |

따라서 encoder 입구가 이전 domain을 버린다는 코드 사실은 있지만, 현재 네 checkpoint·전처리에서는 40이 충분한 보수적 구간임을 별도로 확인했다. 이 지점을 현재 실험의 calibration 누락 또는 정확도 손실 원인으로 지목하지 않는다. 구현에는 이 포함 관계를 자동 검증하는 연결이 없으므로 다른 checkpoint나 theta까지 일반화하지 않는다.

반면 각 모델 block 0의 Q/K/V에 현재 Linear 구간 산술을 적용하면 각각 CIFAR 약 ±582/±581/±423, ImageNet ViT-S 약 ±582/±580/±424, ViT-B 약 ±2476/±2488/±2106, ViT-L 약 ±1760/±2291/±1321이다. 이들은 보수적인 선언 bound이지 실제 관측 extrema가 아니다. 이를 버리고 40으로 교체하는 연결은 여전히 해석적 전달 또는 calibration과 별개의 정책이다. 더 넓은 선언 bound만으로 실제 데이터가 40을 넘었다고 주장하지 않는다.

LayerNorm의 작은 CPU 예에서는 입력 $[-30,30,30,30]$이 선언 구간 $[-30,30]$ 안에 있지만 평균을 빼면 $[-45,15,15,15]$가 된다. Theta 40의 구현은 첫 magnitude를 잘라 제곱·분산까지 바꾸며, 동일 eps의 표준 LayerNorm과 최대 약 0.054795 차이가 난다. 출력의 고정 bound나 입력 calibration만으로 내부 산술이 보존되지는 않는다.

별도로 $[-0.001,0.001,-0.001,0.001]$에서는 theta clipping 없이도 eps 기본값 차이로 출력 크기가 약 0.301511과 0.9999995로 갈린다. 전자는 고정 실험 source의 spiking eps, 후자는 checkpoint eps를 사용한 값이다. 의도된 clipping과 eps 불일치를 구별하는 예이며 실제 영상 정확도 손실의 기여율을 측정한 것은 아니다.

### Complete Path Verdicts

현재 실행되는 모든 연산을 검사하되, 정상적으로 다음 calibration 경계에 연결되는 중간 연산을 별도 누락으로 중복 집계하지 않는다.

| 실행 경로 | 판정 |
|---|---|
| Pixel 전처리·patch Conv·embedding | 전처리 고정 구간과 파라미터 구간 산술을 사용한다. 현재 encoder 40 구간의 포함 관계는 위에서 확인했다. |
| LayerNorm 정규화 출력·affine 출력 bound | Feature 수와 고정 파라미터에 따른 출력 bound는 정상이다. 내부 magnitude 제한과 eps 문제는 별도 항목이다. |
| Q/K/V Linear → attention | Linear 자체는 정상이며, attention 진입 시 출력 bound가 사라지는 지점이 추가 정책이다. |
| Score 곱·합·scale → calibration | 수집·적용 연결은 있으나 통계로 정한 endpoint를 다시 theta로 줄이는 예외가 있다. |
| 정규화 지수·합·division | 선택된 score 범위와 token 수로 유도한 구간을 사용한다. 새 독립 calibration 누락은 찾지 못했다. |
| Attention weight | 함수의 고정 출력 범위 $[0,1]$을 사용한다. |
| V weighted output | 제한된 V의 가중평균이므로 현재 $[-40,40]$은 타당한 파생 구간이다. V 입력 제한의 근거와는 별개이다. |
| Attention output projection → residual | Projection의 해석적 구간을 합산한 뒤 `attention_residual` calibration으로 실제 값과 domain을 함께 교체한다. |
| 첫 MLP Linear → 활성화 | Affine 출력에 `activation_input` calibration을 적용한다. |
| 활성화 내부·출력 | 이미 점검한 고정 연산 제한과 출력 범위를 사용한다. 시냅스 상수 처리 적용 여부를 이번 전체 준수 판정의 대체 근거로 삼지 않는다. |
| 두 번째 MLP Linear → residual | Affine 구간을 합산한 뒤 `output` calibration을 적용한다. |
| 마지막 LayerNorm → class token → classifier | 마지막 LayerNorm에도 같은 eps·magnitude 동작이 적용된다. Classifier는 계획에 공개된 일반 Linear 경로이며 hidden activation calibration을 수행하지 않는다. |
| Reshape·transpose·평가 dropout | 범위를 넓히지 않는 배열 변환 또는 항등 동작이다. 현재 mask·pooler·학습 dropout은 실행되지 않는다. |

LayerNorm의 여러 축 `normalized_shape`에서 마지막 축만 reduce하는 문제와 큰 affine scale이 일반 곱셈에서 theta로 제한되는 문제도 공통 클래스에서 확인했다. 그러나 현재 네 모델의 shape는 한 축이고 실제 scale 최대 절댓값은 모두 40 미만이므로, 이 둘을 현재 결과의 원인이나 이번 필수 재실험 사유로 추가하지 않는다.

### Collection and Publication Contract

데이터 분리와 수집·저장·고정 적용은 확인됐지만, 원고의 수집 단위와 두 pass의 의미는 더 정확히 구별해야 한다.

등록된 site별 tensor 전체에 하나의 min/max와 histogram을 사용하며 뉴런별 독립 구간은 아니다. 첫 pass와 두 번째 pass 모두 원래 해석적 구간 및 기존 고정 제한으로 실행하고, 최종 calibration 구간은 수집이 끝난 뒤 모든 site에 함께 적용한다. 두 번째 pass는 histogram 수집이지 최종 calibrated network의 분포 검증이 아니다. 원고가 순차 수집을 요구하지 않으므로 이 사실 자체를 추가 위반으로 세지는 않는다.

선택된 training 5k·seed 0의 순서를 두 pass에서 유지하고 validation 자료를 calibration에 사용하지 않는다. Frozen 실행은 metadata·site 집합·endpoint 산정식 및 table hash를 검사한다. 이 절차에서는 추가 불일치를 찾지 못했다. 고정된 table의 무결성이 내부의 추가 theta 제한이나 모든 모델의 정확도 보존을 증명하지는 않는다.

재점검 시점에는 ViT-L도 완료되어 SNN 4272/5000이었다. 네 모델 모두 encoder 입구와 Q/K/V 등 attention 하위 모듈의 named clipping은 0건이었다. ViT-L의 calibration clipping 3,256건은 모두 attention score였고, GELU 입력과 residual은 0건이었다. LayerNorm magnitude clipping과 별도의 log 하한 count는 구별해 보존한다.

이 절의 초기 작업은 프로토콜 감사와 CPU 경계 확인이었으며 코드·checkpoint·calibration artifact·완료 결과·ICLR 원고를 바꾸지 않았다. 이후 main의 eps 연결만 수정했고, 원고 완화나 자동 재실험은 하지 않았다.

## Registered Calibration Sites

각 block에는 attention score, attention 뒤 residual 합, GELU 입력, MLP 뒤 residual 합의 네 지점만 등록되고 실제 forward에서 적용된다.

| 실제 site 이름의 block 내부 부분 | 범위 정책 | 적용 위치 |
|---|---|---|
| `attention.attention/attention_score` | 대칭 calibration, 실행 가능한 고정 상한 이내 | Q/K 제한 후 계산한 score, 정규화 지수 연산 전 |
| `attention_residual` | 대칭 calibration | Attention output projection과 skip의 합 |
| `intermediate/activation_input` | 대칭 calibration | 첫 MLP Linear 출력, GELU 연산 전 |
| `output` | 대칭 calibration | 두 번째 MLP Linear 출력과 skip의 합 |

수집 조건은 training seed-0 5k, 두 번의 수집, 2,048-bin histogram, 최솟값·최댓값, 구간 폭의 5% 여유, bound 정책 3이다. [[utils/transformers/models/spiking_vit/calibration.py#vit_calibration_specs]]와 실제 네 table의 site 집합이 일치한다. LayerNorm·embedding·Q/K/V에는 별도 record가 없다.

## Complete Forward Coverage

별도 calibration record가 없더라도 upstream bound를 정상 소비하거나 수학적으로 정해진 bound를 사용하는 연산은 누락으로 분류하지 않는다. 데이터 의존 입력을 고정 theta로 다시 제한하는 위치는 별도로 표시한다.

| 순서·대상 | 현재 범위 결정과 실제 처리 | 분류 |
|---|---|---|
| 전처리한 pixel 입력 | Image processor의 rescaling·normalization 설정으로 고정 구간 계산 | 해석적 bound; 별도 calibration 없음 |
| Patch projection | Pixel 구간과 Conv 가중치·bias로 출력 구간 계산 | 해석적 전파; 별도 calibration 없음 |
| Class token·위치 embedding 합 | Tensor로 처리하며 patch projection의 출력 domain은 이후 전달되지 않음 | 별도 calibration 없음; 다음 encoder 입구에서 범위 재설정 |
| Encoder 입구 | Embedding 값을 `[-40,40]`으로 제한 | 고정 theta; calibration 연결 없음 |
| 각 block의 LayerNorm 두 곳 및 마지막 LayerNorm | Upstream calibration은 내부 magnitude 상한으로 사용하지 않음 | 내부 calibration 연결 없음; 아래 세부 항목 참조 |
| Q/K/V Linear | 입력 Potential과 고정 가중치·bias로 출력 bound 계산 | 해석적 전파; Linear 자체에서 theta로 다시 제한하지 않음 |
| Head 분할·병합 | Reshape와 transpose | 별도 calibration 불필요 |
| Attention의 Q/K 입력 | Projection 출력 domain을 쓰지 않고 각각 `[-40,40]`으로 제한 | 고정 theta; Q/K calibration 연결 없음 |
| Score 곱·합·scale | 제한된 Q/K의 구간에서 유도; 곱셈의 시간 인코딩 피연산자인 K도 theta 제한 | 파생 bound; 별도 독립 calibration 없음 |
| Attention score | 층별 calibration을 실제 적용하되 상한은 theta 및 수치 조건 이하 | Calibration 적용; 고정 상한도 유지 |
| Score의 지수·합·division | Calibrated score와 token 수로 양의 입력 범위·시간 구간 계산 | 파생 bound; 별도 calibration 불필요 |
| Attention weight | `[0,1]` | 구조적으로 고정; 별도 calibration 불필요 |
| V 인코딩 | Projection 출력 domain 대신 `[-40,40]`으로 제한 | 고정 theta; V calibration 연결 없음 |
| Attention weighted output | 제한된 V의 가중평균이므로 `[-40,40]` | 현재 V 제한에서 정당한 파생 bound |
| Attention output projection | 입력 bound와 고정 가중치·bias로 출력 bound 계산 | 해석적 전파 |
| Attention residual 합 | 두 입력 bound를 더한 후 층별 `attention_residual` 적용 | Calibration 적용 |
| 첫 MLP Linear | LayerNorm 출력 bound로부터 affine 출력 구간 계산 | 해석적 전파 |
| GELU 입력 | 첫 Linear 출력에 층별 `activation_input` 적용 | Calibration 적용 |
| GELU 세제곱 magnitude | Calibrated 입력의 최대 절댓값과 40 중 작은 값이 상한; log 하한 `1e-5` | 입력 calibration에 부분 의존하지만 theta 제한·하한은 고정 |
| GELU 상수 계수 및 합 | 고정 계수로 값과 bound를 함께 변환 | 해석적 전파; 상수를 별도 시간 인코딩하지 않음 |
| GELU 내부 지수 입력 | 입력 bound를 전달하되 `±80*tau_s`에서 제한 | 고정 cap; 별도 calibration 없음 |
| GELU gate | 지수 출력에서 division 입력 범위 유도; gate는 `[0,1]` | 구조적·파생 bound |
| GELU 최종 곱 및 출력 | 입력×gate 후 하한 `-0.170041`, 상한 `max(0, 입력 상한)` | 고정 하한·해석적 상한; 출력 calibration 불필요 |
| 두 번째 MLP Linear | GELU 출력 bound와 고정 가중치·bias로 출력 bound 계산 | 해석적 전파 |
| MLP residual 합 | 두 입력 bound를 더한 후 층별 `output` 적용 | Calibration 적용 |
| 마지막 LayerNorm 이후 class token 추출 | Tensor slice | 별도 calibration 불필요 |
| Classifier | 일반 `nn.Linear`; 현재 평가에서는 TTFS 변환 경로가 아님 | Hidden layer 밖; calibration 없음 |

모든 모델은 197개 token과 head dimension 64를 사용한다. Attention에는 이번 평가에서 mask가 없고 비인과 설정이며 dropout은 꺼져 있다. 별도 pooler, exact GELU, 다른 활성화 함수와 비활성 ablation 경로는 이번 비교의 실행 경로가 아니다.

연결 근거는 [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTPatchEmbeddings#forward]], [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTEncoder#forward]], [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTLayer#forward]]와 [[utils/transformers/models/spiking_ops.py#SpikingLinear#forward]]이다.

### LayerNorm Internal Coverage

정규화 출력의 해석적 bound와 평균을 뺀 입력의 데이터 의존 크기를 구별한다. 현재 LayerNorm은 후자를 수집·저장·복원하는 연결이 없다.

| 내부 대상 | 현재 처리 | Calibration 여부 |
|---|---|---|
| 평균을 뺀 양·음 magnitude | 각각 `[0,40]` 제한 | 없음 |
| 양의 log 입력 | `[1e-5,40]`, 비활성 부호 경로는 mask로 분리 | 없음; 고정 하한과 상한 |
| 분산과 제곱근 경로 | 제한된 magnitude로 분산을 계산하고 eps를 더한 뒤 `[1e-10,1600]`에 제한 | 입력 제한에서 파생; 별도 수집 없음 |
| Log 시간 구간 | 위 magnitude·분산 구간과 시간상수에서 계산 | 파생 bound |
| 정규화 값 | 현재 spiking 구성의 feature 수 기반 대칭 bound | 해석적 bound |
| 학습된 affine scale | 체크포인트 파라미터 구간을 고정; 일반 곱셈에서 시간 인코딩되는 scale은 theta 제한 | 파라미터 기반; activation calibration 대상과 구별 |
| 최종 affine 출력 | 정규화 bound와 실제 scale·bias를 함께 사용 | 해석적 bound |

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#forward]]는 입력 Potential을 받지만 내부 magnitude 상한은 `self.theta`로 정한다. 따라서 residual calibration 범위가 ±40보다 넓어도 내부 표현 범위가 넓어지지 않는다. 일반 [[utils/transforms/functions.py#multiplication_operator]]도 시간 인코딩 피연산자를 theta로 제한하므로, calibration JSON에 큰 상한만 추가해서는 실제 동작이 달라지지 않는다.

출력 bound를 다시 데이터로 수집할 필요는 없다. 내부 범위를 새로 선택한다면 clipping 전 magnitude를 관측하고, 선택한 동일 구간을 제곱·분산과 log 경로가 함께 사용하도록 한다. 현재의 동일한 제한 전위 사용 자체는 의도된 동작이다. 기존 결과의 eps 불일치와 이후 main 수정은 [[vit-calibration-audit#ViT Calibration Coverage Audit#Bound Selection and Intended Clipping#Epsilon Correction and Preserved Results]]에 구별해 기록했다.

### Attention and Entry Limits

Q/K/V와 encoder 입구에는 upstream bound를 고정 theta 구간으로 교체하는 지점이 있다. Score calibration만 켜서는 이 앞단 제한이 해소되지 않는다.

[[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTEncoder#forward]]는 embedding을 theta 구간으로 제한한다. Q/K/V Linear는 해석적 bound를 만들지만 attention 호출은 값을 전달하고 내부에서 각 입력 구간을 theta로 새로 정한다. Q는 곱셈의 integration drive임에도 attention에서 별도로 theta 제한을 받는다.

Attention score calibration은 Q/K 제한 후의 score를 수집한다. 현재 float64·197 token·시간상수 1의 수치 조건만으로 정한 대칭 반경은 약 350.56이지만 실행 상한은 theta 40에도 제한된다. 따라서 이 40을 float64의 불가피한 한계라고 설명해서는 안 된다.

반면 attention weight의 `[0,1]`과 현재 V 제한에서 도출한 weighted output 구간은 정당한 bound이다. V 표현 범위를 바꾸면 weighted output 범위도 같이 바꿔야 하지만, 이를 독립적으로 calibration해야 한다는 뜻은 아니다.

## GELU Synaptic Coefficients Are Active

합의한 시간상수 기반 세제곱과 고정 상수의 시냅스 계수 처리는 현재 calibration 수집과 SNN 평가 양쪽에 적용되어 있다.

`scripts/experiments/vit_comparison.py#evaluator_command`가 GELU wrapper와 `phi_nl_psi_ed`를 선택하며, [[scripts/analysis/gelu_cubic_phi_nl_vit.py#install_phi_nl_psi_ed_cube]]가 실제 ViT adapter의 GELU 함수를 교체한다. 실제 collection·SNN 명령과 로그, calibration metadata에서 동일 구현과 source·GELU hash를 확인했다.

| 항목 | 현재 구현 |
|---|---|
| 세제곱 | Log 인코딩의 시간상수를 세 배로 두고 exponential-difference를 적용; 고정 magnitude 상한의 세제곱을 receiving gain으로 사용 |
| 세제곱 항 계수 | `0.044715`를 고정 시냅스 계수로 처리 |
| Tanh 입력 계수 | `sqrt(2/pi)`를 고정 시냅스 계수로 처리 |
| 지수 입력 계수 | `2*tau_s`를 고정 시냅스 계수로 처리 |
| 최종 GELU 곱 | 입력과 gate라는 두 변수의 곱셈은 유지; 시간 인코딩 피연산자는 gate |

실제 함수는 [[scripts/analysis/gelu_cubic_phi_nl_vit.py#gelu_with_phi_nl_psi_ed_cube]]이며 공통 [[utils/transforms/functions.py#_constant_synaptic_scale]]는 값과 bound에 같은 상수를 적용한다. 상수용 별도 곱셈 피연산자나 잡음 주입 사건을 만들지 않는다. 이는 pretrained Linear weight를 재학습하거나 변경했다는 뜻이 아니다.

기본 GELU 함수에 남아 있는 일반 곱셈 두 번의 세제곱 구성은 이번 wrapper에서 호출하지 않는다. 시간상수 세제곱의 고정 magnitude 제한과 지수 입력 cap은 유지되며, 상수 계수 처리가 활성화됐다는 사실과 내부 상한 calibration 여부는 별개이다.

## Observed Clipping and Audit Limits

현재 완료 로그에서 실제로 확인한 clipping과 코드상 존재하지만 별도 count가 없는 제한을 구별한다. 원소 count를 이미지 실패율로 바꾸지 않는다.

| 완료 모델 | Encoder 입구 및 Q/K/V 등 attention 내부의 기록된 clipping | Attention score calibration clipping | GELU 입력 calibration clipping | 두 residual calibration clipping |
|---|---:|---:|---:|---:|
| CIFAR ViT-S | 0 | 0 | 0 | 0 |
| ImageNet ViT-S | 0 | 272 | 0 | 0 |
| ImageNet ViT-B | 0 | 158,845 | 4 | 0 |

각 완료 모델의 attention 하위 모듈에서 Q/K/V Linear와 output projection을 포함한 named clamp 항목은 228개이며 모두 0건이었다. 다른 named clamp의 상한 초과는 LayerNorm magnitude에서 관측됐다. 이는 의도된 clipping이 실제 발생했다는 근거이지 독립적인 구현 결함의 증거가 아니다. 모델별 정확도 손실이 clipping 원소 비율에 비례한다고도 해석하지 않는다.

GELU 세제곱 magnitude 상한과 지수 입력 cap은 직접 `Tensor.clamp`를 사용하므로 전용 초과 count가 없다. GELU 내부 named count도 block별이 아니라 `vit.encoder`에 합산된다. 따라서 모든 GELU 내부 clipping이 0이었다고 단정할 수 없다. Log 하한 count에는 비활성 부호 경로의 0과 `1e-5`보다 작은 양의 magnitude가 함께 포함되며, 잘린 이미지 비율이 아니다.

LayerNorm 내부 범위는 확인된 우선 보완 대상이다. Encoder 입구와 Q/K/V의 연결 없는 고정 제한은 그 보완 이후에도 빠뜨리지 말아야 할 점검 대상이지만, 현재 완료 로그에서 이 제한들이 정확도 하락을 일으켰다는 증거는 없다. GELU·attention weight·정규화 출력의 구조적 bound에 불필요한 calibration을 일괄 추가하지 않는다.

이번 작업은 조사와 문서 기록만 수행했다. 모델 구현·실험 source·진행 중 평가·완료 artifact·논문은 바꾸지 않았다.
