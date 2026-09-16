# ViT Conversion Comparison

CIFAR-10과 ImageNet-1k의 네 ViT 체크포인트를 최신 GELU 구성과 고정 calibration으로 평가하고, 정확도와 명시한 TTFS 구현의 연산 비용을 별도로 보고한다.

## GELU Construction

시간상수 기반 세제곱을 사용하는 GELU에서 고정 상수는 시냅스 계수로 처리하고, 입력과 gate 사이의 최종 곱셈만 유지한다.

기존 두 번의 일반 곱셈으로 세제곱을 계산하는 기본 함수와 비교 실험의 시간상수 기반 경로를 구분한다. 비교 실험은 `phi_nl_psi_ed` 경로를 명시적으로 설치하며, calibration과 SNN 평가 모두 동일 wrapper를 거친다. 세제곱 계수, tanh 입력 계수, 지수 입력 계수는 별도의 시간 인코딩을 만들지 않는다. 일반 multiplication operator는 변경하지 않는다.

GELU 출력 하한 -0.170041, LayerNorm의 양의 log 입력 [1e-5, theta], bound 정책 3은 유지한다. 상수 처리 변경 이전 결과와 calibration은 보존하되 새 실행에 재사용하지 않는다.

## Evaluation Contract

역치 40은 네 모델에 사전 고정하며, calibration은 training 5k에서만 수집하고 평가 결과로 역치를 조정하지 않는다.

공통 설정은 float64, 기본 시간상수 1, 모든 spiking LayerNorm 단계와 attention 및 MLP 활성화이다. 잡음, deadline margin, mismatch, W&B, TensorBoard, 50k 평가 및 추가 학습은 제외한다. 결정적인 단일 평가이므로 seed 반복이나 반복 간 신뢰구간을 추가하지 않는다.

CIFAR-10은 기존 캐시의 순서를 보존한 training seed-0 5k와 test 10k를 사용한다. ImageNet-1k는 기존 training seed-0 5k와 기존 fixed validation 5k를 그대로 사용한다. ANN/SNN은 동일 체크포인트, 전처리, 표본 순서와 batch size를 사용한다.

모델마다 batch size 32, 16, 8 순으로 짧은 calibration 수집과 SNN 평가를 검사한다. CUDA 메모리 부족만 다음 크기를 허용하며 다른 오류는 중단한다. 처음 통과한 크기를 해당 모델의 전체 수집과 두 평가에 고정한다. 짧은 calibration은 별도 파일로 격리하고 최종 표와 섞지 않는다.

최종 calibration은 두 번의 수집, 2,048개 histogram bin, 최솟값과 최댓값, 구간 폭의 5% 여유를 사용한다. 각 encoder block의 네 site를 실제 모델 깊이에 맞춰 검사하므로 ViT-L은 96개, ViT-S/B는 48개이다. 기본 실행량은 calibration 4회와 ANN/SNN 평가 8회이다.

## Scheduling

로컬 GPU 4–7과 UBAI의 RTX A6000을 사용하며, 각 evaluator는 GPU 한 장만 사용한다.

로컬은 ViT-L과 CIFAR ViT-S, UBAI는 ImageNet ViT-S/B가 기본 배정이다. 로컬의 GPU별 잠금과 메모리·사용률 점유 판정을 유지하고, 과거 GPU 0–3 임시 허가는 적용하지 않는다. UBAI는 한 job에 GPU 2장, CPU 8개, RAM 128 GiB를 요청하고 두 독립 실행에 나눈다.

UBAI에는 NAS가 직접 공유된다고 가정하지 않는다. 기존 환경과 데이터 및 ViT-B를 재사용하고 없는 ViT-S만 전송한다. 대용량 검증은 Slurm 준비 작업에서 수행한다. 환경은 실제 디스크인 /enroot의 작업별 경로에 한 번만 풀고, 두 실행의 scratch를 따로 예약한다. tmpfs와 ramfs를 거부하며 종료가 확인된 자기 작업 경로만 정리한다.

두 환경은 동일한 ViT-B 입력과 calibration으로 correct count 및 prediction digest가 일치해야 한다. 대기 중인 UBAI 작업을 로컬로 옮길 때에는 소유자와 작업 이름을 확인하고 취소와 종료 확인을 마친 뒤 배정을 바꾼다. 실행 중인 작업을 중복 제출하지 않는다.

### Temporary Additional GPUs

이번 비교 실험에서는 사용자 승인으로 GPU 1–3을 일시적으로 추가 허용한다. 기본 GPU 4–7 규칙과 기존 experiment.json은 바꾸지 않으며, GPU 0은 제외한다. 추가 GPU는 승인된 batch size의 미시작 ANN 평가에만 사용한다.

추가 실행은 같은 source commit, evaluator, 데이터와 task 정의를 사용한다. 이미 완료했거나 실행 중인 평가를 중복 제출하지 않는다. 해당 모델의 calibration이 종료에 가까워지면 추가 평가만 중단하고 기존 pipeline에 양보한다. 원래 진행 중인 calibration과 SNN은 중단하거나 표본을 나누지 않는다. 따라서 추가 GPU를 허용해도 전체 종료 시간을 결정하는 ViT-L calibration과 SNN의 시간은 크게 줄지 않을 수 있다.

## Results

원시 로그와 manifest가 근거이며, 완료된 ANN/SNN 한 쌍마다 중간 표를 갱신한다.

[[scripts/experiments/vit_comparison_controller.py#Controller]]는 중앙 배정과 프로세스 시작 시각을 기록하고, 완료된 원격 로그·calibration·결과가 같은 hash를 가질 때만 집계한다. 실행 중 복사된 불완전한 묶음은 다음 확인까지 보류한다. 로컬 프로세스와 자식의 메모리 사용량이 64 GiB를 넘으면 해당 작업을 중단한다.

태그는 `vit_conversion_comparison_theta40_calibrated_float64_bounds3_v1`이다. source commit, evaluator와 dependency, 데이터, 체크포인트, calibration의 hash를 기록한다. 로그와 정확한 correct/total을 다시 검사하고 source 또는 calibration이 섞인 결과와 부분 로그를 거부한다.

raw_runs.csv, summary.csv, sop_breakdown.csv, provenance 및 생성한 LaTeX 행은 해당 artifacts 경로에 보존한다. 비용은 실제 checkpoint 설정에서 계산하며 전체 TTFS classifier를 가정한 비용과 실제 dense classifier 평가를 구별한다. Energy는 0.9 pJ/SOP의 가정에 따른 추정치이며 GPU나 칩 측정값이 아니다. 네 쌍과 calibration 네 개가 모두 검증되기 전 논문 표를 완성된 결과로 승격하지 않는다.

### Log Preservation and ICLR Handoff

실험의 원시 로그와 재현에 필요한 기록은 임시 파일 및 논문과 분리해 보존한다. 실행 중 사본과 검증된 완료본을 구별하며, ICLR에는 완료본에서 확인한 결과만 반영한다.

원본은 `artifacts/logs/conversion_comparison/<tag>/`, 별도 사본은 `artifacts/archives/conversion_comparison/<tag>/`에 둔다. 로그, task와 result, calibration과 admission, experiment와 assignment, 이벤트 및 controller 기록, Slurm 로그, 실패·중단 기록을 함께 보존한다. 재시도나 임시 경로 정리가 원본 로그 또는 보존 사본을 삭제해서는 안 된다.

진행 중 CSV는 계속 갱신되므로 시점별 사본은 고유 경로에 저장하고 덮어쓰지 않는다. 각 사본에는 파일별 SHA-256과 진행 중이라는 표시를 남긴다. 이러한 사본은 전체 결과가 아니며 논문 반영 근거로 승인하지 않는다. GPU 잠금과 재생성 가능한 실행 도구는 사본에서 제외할 수 있다. UBAI에서 평가했다면 원격 실패·중단 로그도 별도로 가져온 뒤 보존 완전성을 확인한다.

완료 후 calibration 네 개와 ANN/SNN 평가 여덟 개의 원시 로그, task 및 calibration hash를 다시 검증한다. 검증된 완료본과 CSV, 비용 부분합, provenance, 생성한 LaTeX를 별도로 고정한 뒤 ICLR의 네 Ours 행을 갱신한다. 기존 원고와 적용 diff, 빌드 로그도 해당 보존 경로에 남긴다. 논문에는 필요한 표와 설명만 복사하며 원시 로그는 이동하거나 지우지 않는다.

현재 controller는 집계와 완료 상태까지만 기록하며 원고를 자동 수정하지 않는다. 논문 반영은 [[comparison-costs#Paper Integration Contract]]의 별도 검증 및 diff 검토 절차로 수행한다.

## Clean Accuracy Diagnosis

잡음이 없는 평가에도 연산 내부의 유한 범위 제한은 남는다. 2026-09-15 진단은 고정 역치 40에서 나타난 ViT-S 정확도 하락과 calibration 경계 clipping을 구별한다.

완료된 전체 평가에서 CIFAR-10 ViT-S는 ANN 98.48%, SNN 91.56%, ImageNet ViT-S는 80.26%, 60.14%, ViT-B는 85.50%, 85.14%였다. Gaussian timing noise, deadline margin, mismatch 및 가중치·bias 잡음은 모두 꺼져 있었다. 데이터 순서, 체크포인트, 전처리, 모델별 calibration 연결과 완료 로그 hash는 검증됐다.

### Calibration and Internal Limits

Calibration은 측정한 범위를 전달하지만 activation 값을 재조정하거나 LayerNorm 내부 역치를 늘리지는 않는다.

현재 ImageNet ViT-S의 마지막 block 출력 calibration은 약 ±234.71을 허용하는 반면, [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#forward]]는 평균을 뺀 양·음 magnitude를 각각 40으로 제한한 뒤 분산과 정규화 값을 계산한다. 따라서 calibration clipping이 거의 없어도 LayerNorm clipping과 결정적인 변환 오차가 생긴다. 수집 자체도 내부 제한이 적용되는 SNN으로 수행한다.

최종 calibration site에서 CIFAR-S는 clipping 0건, ImageNet-S는 attention score 상한 272건뿐이었다. 반면 내부 LayerNorm magnitude clipping은 존재한다. ViT-B에도 더 높은 개별 clipping 비율이 있으므로 단순 원소 비율만으로 모델별 정확도 손실 크기를 설명해서는 안 된다. 비활성 양·음 경로에 붙는 log 입력 하한 count는 손실된 영상 수가 아니다.

### LayerNorm Calibration Coverage

현재 calibration 적용 완료는 등록된 site에 한정되며, LayerNorm 내부 입력 범위까지 수집했다는 뜻이 아니다. 정규화 출력의 해석적 bound와 정규화 전 입력의 데이터 의존 범위를 구별해야 한다.

[[utils/transformers/models/spiking_vit/calibration.py#vit_calibration_specs]]는 residual 두 곳, GELU 입력, attention score를 block마다 등록한다. ViT-S/B의 48개 site와 ViT-L의 96개 site에는 LayerNorm 내부 입력이 없다. SpikingLayerNorm은 calibration record를 조회하지 않고 평균을 뺀 magnitude의 상한을 전역 theta로 정한다. 이는 기능을 켜는 옵션이 누락된 것이 아니라 수집·저장·적용 경로가 연결되지 않은 것이다.

정규화된 출력은 feature 수와 학습된 affine 계수로 정해지는 해석적 bound를 유지할 수 있다. 그러나 평균을 뺀 입력의 크기는 층과 체크포인트에 따라 증폭될 수 있으므로, [[domain#Domain Propagation]]의 모델 전체 범위 정책에서 별도 수집 또는 충분한 표현 범위 설정의 대상이다. 출력이 유계라는 사실만으로 이 내부 입력을 제외할 수 없다.

필요한 보완은 clipping 전의 내부 magnitude 상한을 training 데이터에서 정하고, 그 범위와 실제 인코딩·분산·log 계산의 범위를 일치시키는 것이다. 분산 상한은 입력 범위에서 유도하고 log 입력의 양의 하한은 별도로 유지한다. 현재 [[utils/transforms/functions.py#multiplication_operator]]는 호출자가 더 넓은 bound를 주어도 시간으로 인코딩하는 피연산자를 theta로 제한하므로, calibration 표에 상한만 추가해서는 해결되지 않는다.

이 절은 범위 누락을 기록하며 구현 방식이나 새 실행을 승인하지 않는다. 기존 48/96-site 결과는 실제 적용 범위를 명시해 보존하고, LayerNorm 내부 calibration을 완료한 결과로 표현하지 않는다.

전체 hidden layer 경로의 구분, LayerNorm 외의 고정 제한과 GELU 실행 연결은 [[vit-calibration-audit#ViT Calibration Coverage Audit]]에 기록한다.

### Isolated 64-Image Check

같은 첫 64장과 현재의 전체 training 5k calibration을 유지하고 LayerNorm의 역치만 2000으로 바꾸는 별도 진단을 수행했다. 본 실험 조건 또는 완료 결과를 수정한 것은 아니다.

| 모델 | ANN 정답 | 현재 SNN 정답 | LayerNorm만 역치 2000 |
|---|---:|---:|---:|
| ImageNet ViT-S | 52/64 | 39/64 | 51/64 |
| CIFAR-10 ViT-S | 64/64 | 59/64 | 61/64 |

다른 연산의 역치 40, dtype, GELU 구현, 기존 calibration 및 LayerNorm eps는 유지했다. 역치를 바꿀 때 LayerNorm의 파생 bound를 명시적으로 다시 계산하고 호출 후 복원했다. 이는 LayerNorm 내부 magnitude, 분산, log 범위와 affine 곱셈의 역치를 함께 바꾼 진단이며, magnitude clamp 하나만 바꾼 실험이라고 부르지 않는다.

ImageNet ViT-S의 큰 하락에서 LayerNorm의 작은 역치가 주된 요인이라는 증거이며, CIFAR에서는 일부 회복만 확인됐다. 64장 결과를 전체 5k 또는 test 10k 결과로 대체하거나 성능 복구 완료로 보고하지 않는다. 진단 코드와 여섯 실행 로그 및 hash는 `artifacts/diagnostics/conversion_comparison/clean-drop-20260915-zY5PAN/`에 보존하며 공식 집계와 분리한다.

### Historical Comparison and Remaining Uncertainty

예전 ViT-S의 높은 정확도는 현재와 다른 역치 및 구현 조건에서 얻었으므로 동일 조건의 회귀 결과로 해석하지 않는다.

`artifacts/logs/fixed_domain_validation/vit_small_minmax_margin5_clean_5000.log`는 역치 2000, float32, training 1024장 calibration에서 80.36%를 기록했다. `artifacts/logs/gelu_cubic_phi_nl/validation5000_phi_nl_psi_ed.log`의 80.52%도 역치 2000이었다. 동일 모델 식별자와 validation fingerprint는 확인했지만, 시대별 checkpoint hash 규약과 모든 source 상태가 일치한다고 주장하지 않는다.

더 오래된 `artifacts/wandb/wandb-theta-std.csv`의 잡음 0 행은 역치 50에서 60.42%, 200에서 79.46%이다. 이 파일은 source, dtype, calibration 증거가 부족하므로 작은 역치에서 성능이 낮았다는 보조 자료로만 사용한다. CIFAR의 옛 98.42%는 같은 조건의 원시 로그를 확인하지 못했다.

실제 네 모델의 calibrated GELU 입력 범위에서 현재 합성식과 의도한 tanh 근사의 CPU 오차는 최대 약 1.4e-13, 이전 상수 처리와의 차이는 약 1.1e-13이었다. 새 GELU 상수 처리만으로 큰 하락을 설명할 근거는 발견하지 못했다. 별도로 ViT adapter가 checkpoint의 LayerNorm eps 1e-12 대신 기본값 1e-5를 쓰는 차이가 있으나 2026-04-30부터 존재하며, 이번 진단에서는 바꾸거나 원인으로 확정하지 않았다.

## Verification

단위 검사와 실제 입력 검사를 분리하여 구성, 데이터 순서, 실행 조건과 결과의 일치를 검증한다.

평가에서는 [[scripts/evaluation/error_analysis_vit.py#require_finite_logits]]가 NaN 또는 무한대인 logit을 정확도 계산 전에 거부한다. 전처리 설정 파일의 SHA-256과 주요 의존성의 버전도 별도로 고정하며, 두 실행 환경의 버전이 다르면 평가를 시작하지 않는다.

새 검증은 GELU 고정 계수의 부호와 경계, 일반 곱셈 유지, CIFAR image/label 순서, 모델별 calibration site 수, frozen table, source 혼합 거부, 메모리 검사 재개, 로그 완전성, SOP 부분합과 에너지 단위 및 CSV와 LaTeX 일치를 포함한다. 기존 NeurIPS SOP 검증기는 변경하지 않는다. 실행 전 관련 검증과 용어 검사 및 lat check를 통과하고, 최종 논문 반영 후 ICLR 원고를 빌드한다.
