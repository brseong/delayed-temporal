# Fixed Potential Range Audit

This audit records the maintained operator-local range contract; global-range derivations and campaign-specific numerical audits are historical provenance in [[deprecated]].

## 결론

모든 production `Potential`은 analytic propagation 또는 frozen calibration에서 정한 유한 구간을 운반하며, inference batch의 관측 최솟값·최댓값으로 구간을 다시 만들지 않는다.

- ViT, BERT, RoBERTa, GPT-2에 공통 potential range 설정은 없다.
- 네 config는 과거 `theta` key를 거부하고 GPT-2는 `attention_theta`도 거부한다.
- affine, residual, attention, LayerNorm, GELU는 입력과 parameter의 declared range 또는 인증된 calibration record를 소비한다.
- positive log encoding에는 고정된 양의 하한이 남지만, 이는 모델 전체가 공유하는 activation envelope가 아니다.
- timing noise의 표준편차는 각 encoder가 선언한 time-window 길이에 대한 fraction이다.

[[scripts/verification/verify_no_global_theta.py#verify_production_surface]]는 active operator, adapter, evaluator, experiment script의 argument, assignment, keyword와 CLI를 AST로 검사한다.

## 감사 범위와 판정 기준

감사는 production range 생성, 전달, clamp, calibration 저장·복원, evaluator interface와 maintained experiment wrapper를 포함한다.

하나의 경로가 유지 조건을 만족하려면 다음이 모두 성립해야 한다.

1. encoding 전에 유한하고 순서가 있는 range가 정해져야 한다.
2. 같은 frozen identity에서는 batch, 순서, noise seed와 무관하게 같은 range를 사용해야 한다.
3. raw output의 초과 count를 기록한 뒤 declared range로 clamp해야 한다.
4. config나 CLI가 제거된 shared range 값을 다시 주입할 수 없어야 한다.

## Fixed Range의 수식 계약

Range propagation은 fixed function range, input-dependent analytic range, frozen calibration range를 구별한다.

### Potential과 time window

Linear identity encoding은 declared potential range $[l,u]$에서 $t(v)=u-v$와 $T=u-l$을 사용한다.

따라서 symmetric range는 필수 조건이 아니다. Signed PWM은 $l\le0\le u$인 같은 range에서 zero reference $t(0)=u$를 사용하므로 $t(0)-t(v)=v$를 보존한다.

Positive logarithmic encoding은 $0<l\le u$와 time constant $\tau_s$에 대해 $t(v)=\tau_s\log(u/v)$를 사용한다. Lower floor와 upper endpoint는 그 호출의 local domain에 속한다.

### Interval arithmetic

Composed operator는 payload와 range를 함께 전달하며, output range는 fixed definition 또는 input endpoint의 interval arithmetic에서 유도한다.

[[utils/transforms/functions.py#multiplication_operator]]는 두 factor의 declared endpoint를 소비한다. [[utils/transforms/functions.py#scaled_dot_product_function]]은 Q/K bounds, head dimension과 representability ceiling을 사용한다. 어느 함수도 모델 config의 scalar envelope를 조회하지 않는다.

### Affine projection

Affine layer는 frozen input endpoints와 loaded parameter의 부호를 이용해 output interval을 계산한다.

[[utils/transformers/models/spiking_ops.py#SpikingLinear]]와 [[utils/transformers/models/spiking_ops.py#SpikingConv2d]]는 parameter mutation을 검사하고 memoized bound를 payload와 함께 전달한다. GPT-2 `Conv1D` adapter도 같은 원칙을 따른다.

### Layer Normalization

LayerNorm은 centered-input calibration radius를 내부 magnitude, square, variance와 logarithmic encoding에 일관되게 적용한다.

[[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]]은 checkpoint epsilon, positive log floor, frozen centered-input bounds와 learned affine parameter bounds를 분리한다. Variance는 동일하게 제한된 centered potential에서 계산되며, 최종 output bound는 affine endpoint arithmetic으로 유도한다.

### Activation

GELU와 SwiGLU는 fixed contraction 정보와 input-dependent upper endpoints를 구별한다.

[[utils/transforms/functions.py#gelu_approximation]]은 고정된 GELU lower endpoint와 declared input upper magnitude를 전달한다. Power cubic의 positive log floor는 local nonlinear domain의 수치 조건이며 shared activation setting이 아니다.

### Attention

Attention은 Q/K/V projection의 서로 다른 frozen bounds를 실제 encoding, score와 value restoration까지 전달한다.

[[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_sdpa_attention_forward]]은 세 bounds의 부분 전달을 거부한다. Score range는 Q/K interval, head dimension, dtype, time constant와 source capacity로 제한하고, value context range는 V bounds와 normalized-weight bounds에서 유도한다.

## 전수 검색 결과

유지 코드의 range owner는 `utils/transforms`, 네 model adapter, evaluator와 experiment wrapper이며 shared scalar setting은 없다.

AST 검사는 `utils/transforms/`, `utils/transformers/`, `scripts/evaluation/`, `scripts/experiments/`의 production Python을 전수한다. Configuration 파일에 남은 문자열은 old serialized key를 fail closed로 거부하기 위한 검사뿐이다. Historical artifact directory names are identity strings, not executable settings.

README와 canonical agent guide의 evaluator 예시는 removed option을 전달하지 않는다. 과거 threshold plot만 만들던 notebook은 maintained tree에서 제거했다.

## 모든 실행 경우의 처리

Noise-off, zero-standard-deviation, seeded Gaussian과 ablation 경로는 동일한 frozen local ranges를 공유한다.

Gaussian timing error는 event time만 바꾸며 output range를 재선택하지 않는다. Deadline margin은 local Gaussian standard deviation의 배수이고 calibration의 range margin과 독립이다. Dense ablation은 temporal event를 만들지 않지만 paired evaluation identity에는 같은 dataset와 checkpoint를 기록한다.

## Calibration 기록 형식

Current calibration schema binds source, model, data, preprocessing, dtype, numerical floors, site population and output-bound policy.

ViT policy 2 and text policy 1 reject missing or duplicate sites and mismatched metadata. Calibration format version 2 and output bounds version 4 distinguish the local-range contract from historical tables. A short smoke table cannot substitute for a complete training-5k collection.

## 검증 기준

Verification combines static source rejection, boundary values, frozen replay and complete-model evidence.

- legacy config keys and evaluator flags are rejected;
- multiplication, attention, LayerNorm and activation tests cover endpoints and wider-than-earlier local ranges;
- calibration tests cover two-pass counts, table persistence and execution-site completeness;
- result reducers reject incomplete, duplicated or identity-mixed logs.

## Manuscript와의 일치

현재 원고의 continuous-time 실험은 shared range 선택 없이 analytic·calibrated local ranges와 각 encoder의 time window를 보고해야 한다.

과거 threshold selection과 그 수치는 새 Table 3, Table 4 또는 Figure 4의 근거가 아니다. 별도로 보존한 discrete-time 실험은 당시 설정을 명시하는 historical experiment이며 이번 continuous-time 재실행과 합치지 않는다.

## 2026-09-14 Output Bound Audit

This heading preserves links from the archived audit; its former numerical conclusions apply only to the superseded source.

The active contract is the operator-local audit above. Historical Swish, GELU, attention and LayerNorm cases are retained in [[deprecated#과거 실험과 범위 감사#함수 출력 범위 전수 감사]] and must not be copied as current range formulas.

## 2026-09-14 Bound Corrections

The corrections below remain maintained behaviors, now expressed entirely through declared local bounds.

### Attention Output Bounds

Attention context bounds come from frozen V and normalized-weight bounds, with raw output counts recorded before clamping.

[[scripts/verification/verify_attention_output_bounds.py#verify_attention_output_bounds]] covers clean and perturbed normalized-weight cases without a shared activation envelope.

### LayerNorm Affine Bounds

LayerNorm affine output bounds use the normalized-input interval and learned scale and bias endpoints.

[[scripts/verification/verify_layernorm_affine_bounds.py#main]] covers positive and negative learned scales and rejects stale parameter identities.

### Swish Output Bounds

Swish distinguishes its fixed lower endpoint from the input-dependent positive upper endpoint.

[[scripts/verification/verify_swish_output_bounds.py#main]] checks both branches and preserves the distinction between internal Swish and final SwiGLU bounds.

### Activation Intermediate Bounds

Activation intermediates use the narrowest sound interval supported by their fixed function and declared input bounds.

[[scripts/verification/verify_activation_bound_corrections.py#main]] verifies GELU gate, square, cubic and final-output bounds without a global fallback.
