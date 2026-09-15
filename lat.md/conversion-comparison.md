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

## Verification

단위 검사와 실제 입력 검사를 분리하여 구성, 데이터 순서, 실행 조건과 결과의 일치를 검증한다.

평가에서는 [[scripts/evaluation/error_analysis_vit.py#require_finite_logits]]가 NaN 또는 무한대인 logit을 정확도 계산 전에 거부한다. 전처리 설정 파일의 SHA-256과 주요 의존성의 버전도 별도로 고정하며, 두 실행 환경의 버전이 다르면 평가를 시작하지 않는다.

새 검증은 GELU 고정 계수의 부호와 경계, 일반 곱셈 유지, CIFAR image/label 순서, 모델별 calibration site 수, frozen table, source 혼합 거부, 메모리 검사 재개, 로그 완전성, SOP 부분합과 에너지 단위 및 CSV와 LaTeX 일치를 포함한다. 기존 NeurIPS SOP 검증기는 변경하지 않는다. 실행 전 관련 검증과 용어 검사 및 lat check를 통과하고, 최종 논문 반영 후 ICLR 원고를 빌드한다.

## Complete Seven-Model Campaign

최신 비교 실행은 네 ViT와 BERT, RoBERTa, GPT-2를 하나의 source identity 아래에서 다시 평가하고, ICLR에는 검증된 ViT 네 행만 반영한다.

태그 `conversion_comparison_theta40_calibrated_float64_bounds3_v3`은 ViT policy 2와 text policy 1을 구별해 기록한다. CIFAR-10 test 10k, ImageNet fixed validation 5k, SST-2 validation 872개, 고정 dataset revision의 비어 있지 않은 WikiText-2 test 2,891개 전부를 사용하며 모델별 training seed-0 5k calibration을 새로 수집한다.

이번 태그에 한해 로컬 GPU 0--3도 명시적으로 허용하지만 전역 GPU 기본값은 바꾸지 않는다. 로컬과 UBAI 작업은 같은 최종 commit, checkpoint, dataset, preprocessing과 calibration identity를 검사하고, `/tmp`가 아닌 실제 디스크에 runtime을 둔다.

각 ViT evaluator의 flush된 calibration 또는 평가 batch record는 모델별 `status` 파일과 덮어쓰지 않는 progress snapshot으로 복제된다. 부분 record는 진행 확인용이며 완료 결과나 표 생성의 근거로 승인되지 않는다.
