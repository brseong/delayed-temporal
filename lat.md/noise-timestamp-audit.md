# Raw Timestamp Audit

Gaussian timestamp clamp 제거를 인코더, 모든 소비 함수, 합성 연산, 원고 수식과 저장된 결과에 대해 검증한다. 이 기록은 실험 정확도 재평가가 아니라 호출 경로와 경계 동작의 감사다.

## Scope and Evidence

검사 대상은 2026-09-24의 branch `codex/raw-timestamp-observation-margin`, 구현 commit `ed42bfc`와 이 감사에서 추가한 검증이다.

- 제거 대상: Gaussian 잡음 주입 뒤 전달되는 timestamp를 nominal code interval에 맞추는 상한·하한 clamp.
- 유지 대상: calibration 및 analytic potential bounds, 양의 log 입력 하한, 비활성 부호의 기여 제거, Gaussian 주입 전 nominal encoder의 수치 오차 clamp.
- 코드 근거: `utils/transforms/`와 `utils/transformers/` 전체에서 인코더 호출, `SpikeSample` 생성, `time/fired/observation_deadline` 접근을 추적했다. 모델 구현 밖의 분석용 shadow draw는 반환 timestamp를 버리므로 실제 소비 경로가 아니다.
- 원고 근거: ICLR methodology의 Primitive-Level Error Model, appendix의 Temporal Readout and Basic Compositions 및 Simulated Timing Noise Sweeps. 별도 paper Git 저장소의 작업 중 원고를 읽었으며 이 감사에서 원고와 그림은 수정하지 않았다.

## Concrete Boundary Cases

사용자 예시에서 마지막 관측 시각 3과 API의 추가 대기 길이를 구별해 검사한다. `None`은 보고용 표기이고 실제 텐서에서는 유한 저장값과 `fired=False`로 표현한다.

| Nominal endpoint | 추가 대기 길이 | 관측 시각 | raw input | 관측 결과 |
|---|---|---|---|---|
| 2 | 1 | 3 | 2.5, 2.9, 3.1 | 2.5, 2.9, None |
| 2 | 3 | 5 | 2.5, 2.9, 3.1 | 2.5, 2.9, 3.1 |
| 2 | 1 | 3 | 정확히 3, 3 바로 위의 float64 값 | 3, None |
| 4 | 4 | 8 | -0.25, 4.25, 8, 8.25 | -0.25, 4.25, 8, None |

첫 행의 실제 `time` 텐서는 `[2.5, 2.9, 3.0]`, `fired`는 `[True, True, False]`다. 마지막 3은 관측된 spike가 아니라 저장용 값이다. 이 사례는 sampler뿐 아니라 두 production encoder의 decorator를 통해서도 검증한다. 누락된 입력의 결과 potential은 소비 연산의 reset/readout 규칙으로 계산하며 Python `None`을 모델 텐서에 전달하지 않는다.

[[scripts/verification/verify_raw_timestamp_paths.py#verify_requested_boundary_example]]가 위 경계와 두 API 의미를 검사한다.

## Shared Tensor Path

모든 실제 Gaussian 인코더는 같은 sampling owner를 사용한다. Nominal time과 noisy time의 범위 계약은 서로 다르다.

`Potential.value + Potential.domain → neg_identity_transform → neg_linear_transform → inject_spike_time_noise → _sample_gaussian_spike_time → SpikeSample → consumer → bounded Potential`.

Log 경로는 `neg_log_transform → inject_spike_time_noise → _sample_gaussian_spike_time`으로 합류한다.

1. [[utils/transforms/potential_to_spike.py#neg_identity_transform]]은 입력 bounds의 전체 폭을 window 길이로 삼아 decorated linear encoder로 전달한다.
2. [[utils/transforms/potential_to_spike.py#neg_linear_transform]]의 normalized time clamp와 [[utils/transforms/noise.py#inject_spike_time_noise]]의 `out_domain.clamp(output)`은 잡음을 더하기 전 nominal time만 제한한다. Log encoder의 공유 bounds 검증도 잡음 주입 전이다.
3. [[utils/transforms/noise.py#_sample_gaussian_spike_time]]은 하나의 raw sample로 전달 여부를 정한다. 전달된 time에는 clamp가 없다. 누락된 time만 관측 시각 저장값으로 치환한다.
4. [[utils/transforms/types.py#SpikeSample]]의 `domain`은 nominal code interval이며 `observation_deadline`은 margin을 더한 receiver cutoff다. Noisy time이 nominal domain 밖에 있어도 이 metadata는 바꾸지 않는다.
5. 같은 sample을 소비할 때 재샘플링하지 않는다. Signed integration에서 두 event는 같은 receiver cutoff를 사용한다.
6. 소비 결과는 `PotentialBounds`로 제한한다. 이것은 timestamp clamp를 다시 거는 경로가 아니다.

## Encoder Sites and Consumers

다음 21개 site 문자열이 Gaussian event를 생성한다. 반복 호출 수와 layer prefix는 달라도 timestamp를 소비하는 함수는 아래 경로로 귀결된다.

| Site | 입력 tensor와 호출 경로 | timestamp 소비 및 판정 |
|---|---|---|
| `multiplication.data`, `multiplication.reference` | `multiplication_operator → _gaussian_multiplication_operator → neg_identity_transform`; 두 번째 피연산자 tensor와 scalar zero reference | `signed_pulse_width_modulation_operator → signed_pulse_width_duration`; raw time, 실제 cutoff 사용 확인 |
| `linear.data`, `linear.reference` | `SpikingLinear.forward → _gaussian_forward`; 입력 `[..., in_features]`와 scalar reference | duration → `functional.linear`; raw readout 수치 확인 |
| `conv2d.data`, `conv2d.reference` | `SpikingConv2d.forward → _gaussian_forward`; 영상 tensor와 scalar reference | duration → `functional.conv2d`; padding은 기존 potential 0, raw readout 확인 |
| `conv1d.data`, `conv1d.reference` | GPT-2 `SpikingConv1D.forward → _gaussian_forward`; 입력 `[..., in_features]`와 scalar reference | 두 mask 적용 후 duration 차 → `addmm`; raw readout 확인 |
| `attention.value`, `attention.value_reference` | `spiking_scaled_dot_product_attention → _gaussian_attention_value_readout`; value `[..., S, D]`와 scalar reference | duration → attention weight와 matmul; raw readout 확인 |
| `exponential.input` | `exponential_function → _gaussian_exponential_function → neg_identity_transform` | 전달된 time 그대로 offset 제거 후 exp; 최종 potential clamp 직전 값 확인 |
| `division.numerator`, `division.denominator` | `division_function → _gaussian_division_function → neg_log_transform`; numerator tensor 및 broadcast denominator | complete sample을 `exponential_difference_operator`에 전달; raw event 확인 |
| `exponential_difference.internal` | `_gaussian_exponential_difference_operator → signed integration → intermediate potential clamp → neg_identity_transform` | raw internal time에서 intermediate 상한을 빼고 exp; timestamp clamp 없음 확인 |
| `layernorm.log_sigma`, `layernorm.log_positive`, `layernorm.log_negative` | `SpikingLayerNorm.forward → _gaussian_forward → neg_log_transform`; reduced variance와 signed magnitude tensor | enabled expdiff 경로 또는 direct exp ablation 경로. 공통 cutoff와 두 경로 수치 확인 |
| `gelu.cubic.log_positive`, `gelu.cubic.log_negative`, `gelu.cubic.log_reference` | `gelu_approximation → gelu_cubic_power_operator → neg_log_transform`; 두 magnitude tensor, shared scalar reference | 두 `exponential_difference_operator`로 전달; 참조 spike는 재사용. 원래 timestamp 유지 확인 |
| `swiglu.exponential_input` | `swiglu_function → _gaussian_swiglu_function → neg_identity_transform` | raw time을 fixed offset으로 exp 디코딩; potential clamp 직전 값 확인 |

[[utils/transforms/functions.py#_gaussian_multiplication_operator]], [[utils/transforms/functions.py#_gaussian_exponential_function]], [[utils/transforms/functions.py#_gaussian_swiglu_function]], [[utils/transforms/spike_to_potential.py#_gaussian_exponential_difference_operator]], [[utils/transforms/primitive.py#signed_pulse_width_duration]], [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#_gaussian_forward]], [[utils/transformers/models/spiking_ops.py#SpikingLinear#_gaussian_forward]], [[utils/transformers/models/spiking_ops.py#SpikingConv2d#_gaussian_forward]], [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#SpikingConv1D#_gaussian_forward]], [[utils/transformers/integrations/spiking_sdpa_attention.py#_gaussian_attention_value_readout]]에서 직접 확인했다.

직접 time tensor를 읽는 실제 산술 함수는 `signed_pulse_width_duration`, `_gaussian_exponential_function`, `_gaussian_exponential_difference_operator`, `_gaussian_swiglu_function`, `SpikingLayerNorm._gaussian_forward`, `SpikingConv1D._gaussian_forward`다. Linear·Conv2d·attention은 duration helper로 위임한다. Sampling wrapper의 `sample.time.numel()`은 통계용 접근이다.

## Composed and Model Paths

여러 단계의 potential 제한을 timestamp 제한과 혼동하지 않도록 상위 함수의 연결도 확인한다.

| 상위 연산 | 실제 연결 |
|---|---|
| Attention score | Q/K potential clamp → `scaled_dot_product_function → multiplication_operator` → 차원 합산 → score potential 제한 |
| Attention weight | `softmin_function → exponential_function → sum → division_function → exponential_difference_operator` → weight potential 범위 `[0,1]` |
| GELU | signed cubic의 세 log event → 두 expdiff → 부호 기여 결합 → constant synaptic scaling → `_tanh_sigmoid_gate → exponential_function → division_function` → 마지막 multiplication |
| Tanh / sigmoid GELU | exponential → division → 고정 affine 결합 또는 multiplication; 독립적인 timestamp consumer 없음 |
| SwiGLU | 직접 exponential event → division → 두 multiplication; 직접 exponential 소비만 별도 검사 |
| LayerNorm | centered potential → 두 제곱 multiplication 또는 direct square → variance → 세 log event 또는 direct log → 두 expdiff 또는 direct exp → affine |
| ViT | patch Conv2d, Q/K/V·MLP·classifier Linear, shared attention, LayerNorm, GELU가 위 owner들을 호출 |
| BERT / RoBERTa | Linear, shared attention, LayerNorm, GELU 및 pool/head activation이 같은 owner 사용 |
| GPT-2 | projection의 Conv1D 외에 shared attention·LayerNorm·GELU·output Linear 사용 |
| CCT | Conv2d tokenizer, ViT 계열 block, final LayerNorm, sequence pooling Linear → softmin → multiplication → sum, classifier Linear |

최종 classifier와 pooling에 도달하기 전 timestamp는 이미 potential로 디코딩된다. Reshape, residual addition, calibration도 potential과 bounds를 다루며 noisy timestamp를 따로 재제한하는 owner가 아니다.

잡음이 꺼져 있거나 scope 밖이면 deterministic 경로를 실행한다. 또한 multiplication의 입력 factor 구간이 정확히 0인 경우와 GELU cubic의 전체 magnitude 구간이 log 하한 이하인 경우에는 event를 만들지 않는 특수 경로가 있다. 이들은 이미 생성된 noisy timestamp를 clamp하는 분기가 아니다.

## Remaining Clamps

남아 있는 clamp는 각 입력의 의미를 따라 분류했다. 여기서 제거 완료라는 결론은 전달된 Gaussian timestamp에 한정한다.

| 위치 | 유지 이유 |
|---|---|
| Linear encoder의 `normalized_time.clamp(0,1)`, wrapper의 nominal clamp | Gaussian draw 이전 codeword 수치 오차 제한 |
| LayerNorm에서 spiking log가 꺼진 분기의 `t_sigma/t_err.clamp(0,T0)` | 직접 계산한 nominal log의 roundoff 처리. 해당 분기는 Gaussian log event를 생성하지 않음 |
| Expdiff의 plain tensor wrapping에서 `domain_t.clamp(t)` | 이미 전달된 deterministic tensor 경로. `SpikeSample` 분기에는 적용되지 않음 |
| `(observation_deadline-time).clamp_min(0)` | 음수 integration duration 방지. 실제 fired sample은 cutoff 이하여서 raw time을 nominal endpoint로 바꾸지 않음 |
| 세 exponential 소비자의 `where(fired, time, nominal_max)` | miss의 저장값을 exp에 넣어 불필요한 overflow를 만들지 않기 위한 inactive 계산값. fired time은 그대로 선택됨 |
| Expdiff의 intermediate potential clamp | 시간차를 전위로 변환한 뒤 다음 encoder 입력의 고정 범위를 적용 |
| `clamp_gaussian_output`, calibrated/analytic bounds, attention score cap, positive log floor, GELU gate/cubic bounds | 고정 potential 표현 구간. 원고의 finite domain 제한과 연결됨 |
| `clock.py`의 time clamp / 양자화 | 독립적인 이산시간 실행. Gaussian과 동시 사용은 encoder boundary에서 거부 |

예를 들어 늦은 exponential spike가 timestamp clamp 없이 디코딩되어도 결과 potential이 상한에서 제한될 수 있다. 따라서 최종 출력값만 비교하지 않고 potential clamp 직전 값을 검사했다.

## Receiving Equations

원고의 이상적 시간차 및 normalized exponential을 raw timestamp와 실제 receiver cutoff에 적용하여 수치 기준을 만든다.

Signed integration은 [[noise]]의 두 duration 차를 사용한다. 양쪽 spike가 전달되면 같은 observation time이 상쇄되어 두 raw timestamp의 차만 남는다. 한쪽 miss에서는 살아 있는 쪽의 실제 receiver cutoff까지의 duration이 남는다. 두 event가 모두 miss이면 signed potential은 reset으로 남는다. 이후 expdiff의 내부 재인코딩이 timing error 없이 성공하면 exponential 값은 reset potential의 지수이므로 1이다. 내부 event에도 잡음을 넣으면 그 값은 추가로 변한다. Expdiff 자체를 두 입력 miss만으로 0으로 만드는 검사는 잘못된 기준이다.

Exponential에서는 이 절에서만 코드의 fixed offset을 $b$, 실제 관측 시각을 $D$로 둔다. Normalized branch의 $b$는 encoder 입력 potential 상한, unnormalized branch의 $b$는 nominal time window 폭의 절반이다.

$$
w=\exp((D-b)/\tau_m),\qquad
w\exp(-(D-\tilde t)/\tau_m)=\exp((\tilde t-b)/\tau_m).
$$

오른쪽이 구현의 계산이다. 코드에서 `D`가 최종 exp 식에 보이지 않는 것은 왼쪽의 보정 gain과 상쇄된 결과다. 실제 하드웨어에서 readout을 늦추고도 gain을 그대로 두는 동작과는 구별해야 한다. ICLR appendix는 observation time에 따른 $w$를 이미 명시한다. `tau=0.5,1,2`에서 두 식의 값도 검사한다.

## Paper Comparison

Timestamp 수정과 원고 전체 및 기존 결과의 일치를 별도로 판정한다. 원고가 현재 branch의 새 실험 결과를 이미 보고한다고 간주하지 않는다.

| 원고 위치 | 확인 결과 |
|---|---|
| Methodology, Spike-time noise and deadline misses | $\tilde t=t+\epsilon$ 및 cutoff를 넘을 때만 miss라는 정의는 현재 sampler와 일치. Equality는 코드에서 delivered로 처리 |
| Methodology, Observation margin | 최종 codeword 뒤에 대기 구간을 둔다는 설명과 cutoff 사용 일치. API margin은 추가 길이 |
| Methodology, finite potential ranges 및 appendix constraints | potential clamp와 log floor를 유지하는 것과 일치 |
| Appendix, normalized exponential의 $w$ | 위의 관측 시각 보정을 포함하면 raw exponential 계산과 일치 |
| Appendix, $\psi_{\mathrm{ED}}$ 설명과 methodology의 ideal $\Psi$ | 설명 보완 필요. 원고는 직접 primitive와 선택 가능한 분해를 구분하지만 실행은 항상 분해 경로를 사용하고 내부 `exponential_difference.internal`의 $\phi_{\mathrm{NP}}$에도 noise를 넣는다. 외부 spike 둘만 perturb하고 이상적 exp 차를 계산하는 수치 모델과는 다름 |
| 시작 전 early timestamp | 현재 모델은 음수 local time도 그대로 전달. 원고의 additive Gaussian 식에는 맞지만 수신기가 언제부터 동작하는지에 대한 물리적 scheduling을 이 구현만으로 입증하지 못함 |
| Appendix, completed simulated sweep와 figure | 현재 branch의 결과로 사용할 수 없음. 아래 provenance를 확인 |

원고 비교는 위 noise 계약과 해당 합성 연산 범위에 대한 검사다. 논문 전체의 모든 이론·비용 모델까지 동일하다는 판정은 아니다.

## Existing Figure Provenance

현재 게재용 noise figure의 데이터 출처를 실제 파일과 비교했다. Timestamp 모델을 바꾼 뒤 옛 그림을 새 결과로 해석하면 안 된다.

- `artifacts/results/paper_end_to_end_local_range_poseidon_v1/summary.json`의 `noise_source_commit`은 `b1a6bf8f7baa89250201c9af96d05b6154249de5`; 63 runs, 21 cells.
- 그 commit의 sampler에는 `torch.clamp(raw_time, min=start, max=code_deadline)`이 실제로 존재한다.
- `paper/iclr_2027/figures/ViT-noise-eval.pdf`와 `artifacts/figures/ViT-noise-eval-end-to-end.pdf`는 생성 시각 metadata의 두 byte만 다르다. CreationDate를 제거한 양쪽 digest는 `ad8c8efa68b6dfdc0db2d0a5a8d885b61db08ee76ed81f512bf3d1e2ec5b0ac2`.
- `scripts/analysis/summarize_local_range_paper_campaign.py`가 위 결과 CSV와 figure를 함께 생성한다. 따라서 기존 figure는 timestamp clamp 제거 효과를 검증하지 않는다.
- 이 감사는 기존 실험·BSS2 기록·figure를 덮어쓰거나 재승격하지 않는다.

## Boundary and Consumer Tests

독립적인 경계값과 수치식으로 timestamp의 보존과 실제 소비를 검사한다. 최종 정확도나 유한 출력만으로 경로 일치를 판정하지 않는다.

[[scripts/verification/verify_raw_timestamp_paths.py#main]]은 CPU float64에서 일곱 검증 그룹을 실행한다.

1. 사용자 경계 예제, 두 production encoder, equality 및 바로 다음 표현 가능 값.
2. Seeded Gaussian draw의 raw time과 실제 sampler 결과 비교, early/late tail, signed integration의 네 delivery 조합.
3. Multiplication·Linear·Conv2d·GPT-2 Conv1D의 raw potential을 독립 duration 식 및 dense contraction과 비교.
4. Normalized/unnormalized exponential, 내부/외부 expdiff, SwiGLU의 potential clamp 직전 값 확인.
5. Attention value readout, frozen calibration을 적용한 LayerNorm 8개 조합. 직접 exp와 spiking expdiff 양쪽에서 raw duration 식 비교.
6. GELU power, sigmoid GELU, tanh, softmin을 통한 모든 중첩 log event의 raw time 보존 및 출력 bounds 확인.
7. 이전 timestamp clamp를 임시로 되살리는 negative control에서 경계 검증이 실제로 실패함을 확인.

기존 `verify_layernorm_calibrated_bounds.py` fixture는 폐기된 policy 2 때문에 검증 전에 실패했다. 실행기의 policy 제한은 유지하고 fixture가 현재 `VIT_CALIBRATION_POLICY_VERSION` 및 output head metadata를 사용하도록 수정해 두 LayerNorm 검증기를 실제 실행했다. Clean 및 Gaussian std=0 parity, 두 pass, frozen bounds, 8개 조합은 이 기존 검증기로 함께 확인한다.

전체 실행 로그는 `artifacts/verification/raw_timestamp_ed42bfc_20260924/`에 보존한다. `verify_raw_timestamp_paths.py`, `verify_gaussian_time_noise.py`, `verify_layernorm_shared_deadline.py`, `verify_layernorm_calibrated_bounds.py`, `verify_attention_output_bounds.py`, `verify_vit_block_scoped_noise.py`를 `/opt/conda/envs/dt/bin/python scripts/verification/<파일명>`으로 실행한다. 검증 중 출력되는 Transformers docstring 진단은 assertion 실패가 아니며 원시 로그에 그대로 보존했다.
