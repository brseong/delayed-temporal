# ViT Calibration Policy 2 Implementation

새 ViT calibration은 Linear 출력의 범위가 Attention까지 전달되도록 하고, LayerNorm 내부 범위를 층별로 선택한다. 구버전 결과를 보존하며 이번 실행은 짧은 검사와 후속 준비로 제한한다.

## Numerical Contract

기존 네 비교 모델의 theta 40, float64, 시간상수 1과 GELU 구성을 유지한다. 모든 spiking 단계가 켜진 구성에서 S/B는 109개, L은 217개의 범위를 수집한다.

각 Q/K/V Linear 출력과 각 LayerNorm에서 평균을 뺀 값을 따로 수집하며, 대칭 구간의 전체 폭에서 5%를 양쪽에 더한다. 최종 LayerNorm도 포함한다. 범위가 이미 정해지거나 앞선 구간에서 유도되는 Softmax 출력·분산·가중합에는 중복 calibration을 추가하지 않는다.

LayerNorm의 checkpoint eps는 현재 네 모델 모두 1e-12이다. 평균을 뺀 값의 절댓값에 대한 log 하한 1e-5와 분산 하한 1e-10은 유지한다. 같은 제한된 전위로 분산을 계산하는 것은 의도된 clipping이며 별도의 결함으로 분류하지 않는다. 내부 상한을 바꾸더라도 학습된 곱셈계수의 인코딩 범위까지 바꾸지 않는다.

범위 전달과 검증 계약은 [[calibration#Layer-wise Calibration#Frozen Execution#ViT Calibration Policy 2]], [[calibration#Layer-wise Calibration#Frozen Execution#ViT Attention Bound Transfer]], [[calibration#Layer-wise Calibration#Frozen Execution#ViT LayerNorm Internal Calibration]]을 따른다.

## Execution and Preservation

새 태그는 `vit_conversion_comparison_theta40_calibrated_float64_bounds3_v2`이며 기존 v1의 calibration·로그·고정 checkout과 결과를 새 실행에 재사용하지 않는다.

[[scripts/experiments/run_vit_comparison.py#initialize]]는 실제 모델 구조에서 site 목록을 확인하고 source·evaluator·의존성·checkpoint·데이터 hash와 함께 고정한다. [[scripts/experiments/vit_comparison.py#validate_table]]은 저장된 site 집합과 정책, eps 및 log 하한을 대조한다. 새 실행은 정책 2만 허용하고 구버전 결과 검증은 별도로 유지한다.

짧은 검사는 GPU 4–7 중 유휴 장치에서만 수행한다. GPU별 잠금과 한 프로세스 제한을 유지하며 0–3의 과거 임시 허가는 적용하지 않는다. Batch size 32, 16, 8 순서에서 GPU 메모리 부족일 때만 다음 크기를 시도한다. 각 시도는 training 두 batch의 두 번 수집과 별도 평가 두 batch이며, 기술 오류는 즉시 중단하고 로그를 보존한다.

W&B·TensorBoard·잡음은 끄고 임시 파일은 `/data/delayed-temporal/artifacts/runtime/<tag>`에 둔다. 완료된 자기 작업의 임시 경로만 정리한다. 전체 5k/test 10k 비교, noise sweep, UBAI 제출 및 논문 결과 갱신은 이번에 실행하지 않는다.

## Prepared Commands and Verification

준비 명령은 검증된 짧은 실행 결과와 후속 명령만 기록하며 전체 평가를 시작하지 않는다. 짧은 검사용 calibration은 전체 수집을 대신하지 않는다.

[[scripts/experiments/run_vit_comparison.py#prepare_execution]]은 각 모델의 검증된 batch size 또는 아직 미확인인 상태와 전체 수집이 필요한 후속 명령을 내용 hash별 JSON으로 보존한다. 준비 파일은 실행 승인이 아니며 실행 시 source·데이터·checkpoint·calibration identity와 GPU 점유를 다시 확인한다.

[[scripts/verification/verify_vit_comparison_runner.py#verify_policy2_and_preparation]]은 109/217개 site, 구버전 읽기와 새 실행 차단, eps·하한 불일치 거부 및 준비 과정에서 evaluator를 호출하지 않는 것을 확인한다. Attention·LayerNorm·calibration·Gaussian 및 기존 비교 실행기 검증과 용어 검사, `lat check`를 함께 실행한다.
