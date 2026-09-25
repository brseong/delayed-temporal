# ICLR 2027 원고 문단 흐름 검토

이 문서는 2026-09-23 현재 ICLR 2027 초안의 문단별 역할과 연결을 기록하고, 서술 흐름이 끊기는 지점을 우선순위에 따라 정리한다.

## 판정 기준

영문 본문의 빈 줄 기준 실질 문단을 전수 검토하고, 식·표·그림은 이를 도입하거나 해석하는 문단과 함께 판단한다.

판정은 `자연`, `약함`, `도약`을 사용한다. `약함`은 연결 의도는 보이지만 독자가 중간 관계를 복원해야 하는 경우이고, `도약`은 뒤의 주장에 필요한 정의·근거·결과가 앞에서 준비되지 않은 경우다. 주석 처리된 한국어 초안은 문단 수에서 제외하되 영문 본문과 주장 범위가 다르면 별도로 기록한다.

이 검토는 [[neurips-writing#ICLR 문장 정리 범위]]와 [[todo#Manuscript Revision Master Checklist]]의 주장 범위를 따른다. 수치와 인용의 사실 여부를 새로 검증하지 않고, 현재 원고 안에서 문단의 역할과 근거 배치만 확인한다.

## 통합 결론

인접 문단의 흐름은 대체로 자연스럽지만, 전체 논증은 measured primitive-level error 결과가 비어 있어 끝까지 닫히지 않는다.

현재 흐름은 다음과 같다.

`TTFS motivation → Transformer operation gap → potential--time dual representation → primitive operators → composed functions → finite potential ranges and calibration → conversion fidelity and SOP → discrete-time simulation → measured primitive-level errors → conclusion`

의도한 골격은 분명하다. 그러나 실제 원고에서는 `primitive composition → evaluated model에서 교체한 operation subset`과 `Methodology의 range 분류 → Experiment의 calibrated upper bounds`가 충분히 연결되지 않는다. 또한 완료된 simulated timing-noise sweep는 부록에만 있고, measured primitive-level error 결과는 pending인데 초록·서론·결론은 이를 완료된 평가처럼 서술한다.

따라서 현재 판정은 **국소 흐름은 대체로 자연스럽지만, 전체 흐름은 아직 자연스럽게 완결되지 않는다**이다.

## 본문 문단 지도

아래 표는 abstract부터 Conclusion까지 모든 실질 영문 문단을 순서대로 요약한다.

### Abstract, Introduction, Related Work, Preliminaries

앞부분은 문제와 해법을 빠르게 세우지만, 실제 conversion scope와 hardware evidence의 범위가 늦게 제한된다.

| ID | 파일:줄 | 문단 요약 | 앞 문단과의 관계 | 판정 |
|---|---|---|---|---|
| P01 | `iclr2027_conference.tex:77-80` | TTFS의 single spike 표현, Transformer conversion 문제, $\Phi/\Psi$ framework, conversion fidelity, BrainScaleS-2 error distribution 주입을 압축한다. | 전체 논증의 출발점이다. | 자연 |
| P02 | `intro.tex:5` | ANN-to-SNN conversion과 TTFS의 이점을 제시하고 attention, normalization, smooth nonlinearities를 문제로 세운다. | Abstract의 문제를 확장한다. | 자연 |
| P03 | `intro.tex:7` | target function별 dynamics 대신 두 representation 사이 fixed operators로 계산할 수 있는지 묻는다. | 일반 난점을 research question으로 좁힌다. | 자연 |
| P04 | `intro.tex:9` | membrane potential과 spike timing의 왕복을 reusable operators로 보고 potential--time dual representation을 제안한다. | 질문에 framework로 답한다. | 자연 |
| P05 | `intro.tex:11` | ideal composition의 finite range 문제를 calibration과 clipping으로 다루고 model-level conversion fidelity를 보고한다. | framework를 finite implementation과 결과로 연결한다. | 자연 |
| P06 | `intro.tex:13` | timing perturbation, deadline miss, observation margin, measured encoding error 주입을 소개한다. | finite range에서 timing and mapping errors로 평가 범위를 넓힌다. | 자연 |
| P07 | `intro.tex:15` | framework, potential calibration, measured primitive-level errors의 세 기여를 요약한다. | Introduction의 내용을 회수한다. | 자연 |
| P08 | `related_work.tex:7` | rate-coded와 temporally coded conversion을 구분하고 fixed operator composition의 위치를 잡는다. | 기여를 가장 가까운 conversion literature에 놓는다. | 자연 |
| P09 | `related_work.tex:11` | discrete-time Spiking Transformers와 continuous-time temporal coding을 대비한다. | conversion에서 직접 설계된 spiking models로 범위를 넓힌다. | 약함 |
| P10 | `related_work.tex:15` | analog neuromorphic hardware의 dynamics와 Transformer operations 사이 간극을 설명한다. | analog substrate 언급을 operator gap으로 구체화한다. | 자연 |
| P11 | `preliminaries.tex:5-16` | pre-norm과 post-norm blocks 및 두 ordering이 공유하는 attention, normalization, activation, linear, residual operations를 정의한다. | hardware가 직접 표현하지 못하는 target operations를 수식으로 명시한다. | 자연 |
| P12 | `preliminaries.tex:20-38` | LIF/IF dynamics, exponential synaptic current, double-exponential PSP를 도입한다. | target operations에서 neuron dynamics로 전환하지만 연결 설명이 없다. | 약함 |

앞부분의 흐름은 `single spike TTFS → Transformer operation gap → fixed operators라는 질문 → potential--time dual representation → finite ranges → timing and mapping errors → contribution summary → prior work → target operations → neuron dynamics`이다.

### Framework and Methodology

중간 부분은 cubic example에서 primitive composition과 finite ranges로 진행하지만, full Transformer conversion으로 이어지는 설명이 비어 있다.

| ID | 파일:줄 | 문단 요약 | 앞 문단과의 관계 | 판정 |
|---|---|---|---|---|
| P13 | `framework.tex:5-12` | Figure 1로 membrane-potential과 spike-time representation의 교대 및 primitive composition을 제시한다. | Preliminaries의 dynamics를 computation 관점으로 전환한다. | 약함 |
| P14 | `framework.tex:14-25` | negative-log spike time과 exponential response를 조합해 $V^3$을 얻는 예를 보인다. | Figure의 cubic panel을 식으로 구체화한다. | 약함 |
| P15 | `framework.tex:26-34` | $\Phi:\mathcal V\to\mathcal T$와 $\Psi:\mathcal T\to\mathcal V$를 정의한다. | cubic example에서 두 mapping class를 추출한다. | 자연 |
| P16 | `framework.tex:35-39` | time constant의 비로 $V^{\tau_s/\tau_m}$을 구성할 수 있음을 보인다. | cubic construction을 power family로 일반화한다. | 자연 |
| P17 | `framework.tex:41` | ANN function을 $\Phi/\Psi$ operator composition으로 실현한다고 선언한다. | power family에서 일반 ANN function으로 범위를 넓힌다. | 도약 |
| P18 | `framework.tex:43-61` | 다섯 primitive operator의 type, mapping, implementation을 표로 정의한다. | 추상적 composition을 fixed interface로 구체화한다. | 자연 |
| P19 | `framework.tex:64` | primitive computations의 prior analog hardware 사례를 요약한다. | 표의 implementation 근거를 보강한다. | 자연 |
| P20 | `framework.tex:66-90` | multiplication, division, exponential, power, GELU, scaled dot product의 compositions를 표로 제시한다. | primitive interface를 composed functions로 확장한다. | 자연 |
| P21 | `framework.tex:92-103` | fixed gains, signed power branches, Softmax와 LayerNorm building blocks, appendix derivation을 설명한다. | composition table을 해설하지만 full operations는 보이지 않는다. | 약함 |
| P22 | `methodology.tex:6` | ideal continuous-time composition과 finite implementation 사이 unbounded range 문제를 설명한다. | noise-free identities에서 finite constraints로 이동한다. | 자연 |
| P23 | `methodology.tex:9` | encoder input potential range를 제한해 spike-time window를 제한한다. | unbounded 문제에 직접 대응한다. | 자연 |
| P24 | `methodology.tex:12` | finite ranges가 bounds 밖 입력을 clip하므로 conversion fidelity 평가가 필요하다고 말한다. | 해법의 정보 손실을 제시한다. | 자연 |
| P25 | `methodology.tex:14-16` | potential ranges를 세 유형으로 나누고 calibration 적용 대상을 정한다. | fixed range 선택을 정책으로 체계화한다. | 자연 |
| P26 | `methodology.tex:18-20` | training subset의 layer-wise calibration, 5% 확장, inference 고정, clamping을 규정한다. | calibration 절차를 상세화한다. | 자연 |
| P27 | `methodology.tex:22-24` | signed integer power를 위해 $V$와 $-V$ branches 및 weight sign 결합을 설명한다. | calibration에서 power의 signed input 처리로 바뀐다. | 약함 |
| P28 | `methodology.tex:26-28` | spike-time perturbation과 primitive mapping error라는 두 error mechanism을 정의한다. | signed power에서 model-level error evaluation으로 이동한다. | 도약 |
| P29 | `methodology.tex:30-32` | Gaussian spike-time noise와 deadline을 넘긴 spike omission을 정의한다. | 첫 error mechanism을 구체화한다. | 자연 |
| P30 | `methodology.tex:35` | $\psi_{\mathrm{Int}}$에서 deadline 뒤로 이동한 $t_2$의 영향을 설명한다. | 영향이 operator에 따라 달라지는 첫 사례다. | 자연 |
| P31 | `methodology.tex:38` | $\psi_{\mathrm{NE}}$에서 deadline miss가 output을 zero로 만드는 대조 사례를 든다. | 앞 사례와 대조해 영향을 완성한다. | 자연 |
| P32 | `methodology.tex:40-42` | BrainScaleS-2에서 primitive output deviation을 characterize하고 model locations에 주입한다고 말한다. | analytic timing case에서 measured distribution으로 확장한다. | 자연 |
| P33 | `methodology.tex:44-46` | observation margin이 deadline misses와 task performance에 미치는 영향을 평가한다고 말한다. | measured mapping error 뒤에서 deadline miss mitigation으로 돌아간다. | 약함 |

중간 부분의 흐름은 `neuron dynamics → cubic example → $\Phi/\Psi$ definition → primitive operators → composed functions → finite potential ranges → layer-wise calibration → signed power → spike-time noise and deadline misses → measured primitive-level errors → observation margin`이다.

### Experiment and Conclusion

뒷부분은 설정과 noise-free conversion results까지는 이어지지만, measured primitive-level error 결과가 없어 Conclusion으로 연결되지 않는다.

| ID | 파일:줄 | 문단 요약 | 앞 문단과의 관계 | 판정 |
|---|---|---|---|---|
| P34 | `experiment.tex:6` | conversion fidelity와 operation cost, discrete-time simulation, measured primitive-level error robustness의 세 평가 축을 제시한다. | Methodology에서 Experiment roadmap으로 전환한다. | 약함 |
| P35 | `experiment.tex:11` | models, datasets, implementation과 Appendix의 details를 밝힌다. | roadmap을 평가 대상으로 구체화한다. | 자연 |
| P36 | `experiment.tex:14` | source ANN과 두 부류의 prior methods를 baselines로 정한다. | 평가 대상을 비교 구도로 확장한다. | 자연 |
| P37 | `experiment.tex:17` | continuous-time evaluation의 time constants, potential ranges, spike-time windows를 설명한다. | baseline 뒤에 conversion setting을 제시한다. | 자연 |
| P38 | `experiment.tex:21` | calibration population, range expansion, positive log input floor, exponential input limit를 명시한다. | data-dependent range setting을 상세화한다. | 자연 |
| P39 | `experiment.tex:25` | source ANN과 converted SNN의 within-row $\Delta$를 conversion fidelity로 사용한다. | setting에서 첫 evaluation criterion으로 전환한다. | 자연 |
| P40 | `experiment.tex:63` | vision table의 evaluation populations 차이를 다시 설명한다. | comparison rule을 표에 적용하지만 caption을 반복한다. | 약함 |
| P41 | `experiment.tex:66` | evaluated ViTs의 accuracy reduction이 최대 $0.06$ pp라고 해석한다. | population 설명에서 vision result로 나아간다. | 자연 |
| P42 | `experiment.tex:94` | language tasks에서 $\Delta$의 의미와 단위를 설명한다. | vision에서 language model evaluation으로 확장한다. | 자연 |
| P43 | `experiment.tex:97` | evaluated language models에서도 small conversion differences가 유지된다고 말한다. | table definition을 result interpretation으로 닫는다. | 자연 |
| P44 | `experiment.tex:101` | data spikes와 global reference spikes를 합산한 SOP와 Appendix derivation을 안내한다. | conversion fidelity에서 operation cost로 초점을 옮긴다. | 약함 |
| P45 | `experiment.tex:111` | 추가 training이나 recalibration 없이 temporal resolution만 이산화한다고 설명한다. | continuous-time result에서 discrete-time simulation으로 넘어간다. | 약함 |
| P46 | `experiment.tex:136` | BrainScaleS-2 measurements가 끝나면 task performance와 deadline-miss rates를 보고한다고 말한다. | discrete-time simulation에서 measured errors로 넘어가지만 결과가 없다. | 도약 |
| P47 | `conclusion.tex:5` | framework, fixed primitive composition, BrainScaleS-2 empirical distribution evaluation을 핵심 기여로 요약한다. | pending result를 완료된 evaluation처럼 전환한다. | 도약 |
| P48 | `conclusion.tex:8` | small fixed operator set과 comparable operation counts를 주장한다. | 방법 요약에서 operation cost 결론을 추린다. | 약함 |
| P49 | `conclusion.tex:11` | SOP scope, simulation-based model evaluation, severe noise, hardware training 부재를 limitations로 제시한다. | contribution에서 limitations로 이동하지만 완료된 error evaluation을 전제한다. | 약함 |

뒷부분의 흐름은 `setup → conversion fidelity → SOP → discrete-time simulation → measured primitive-level errors pending → completed evaluation을 전제한 Conclusion`이다.

## Appendix 문단 지도

Appendix의 식과 표 자체보다 이를 도입·해석하는 모든 영문 문단을 순서대로 요약한다.

| ID | 파일:줄 | 문단 요약 | 앞 문단과의 관계 | 판정 |
|---|---|---|---|---|
| A01 | `appendix.tex:11-14` | main conversion table의 SOP derivation과 operation-count scope를 선언한다. | P44에서 직접 진입한다. | 자연 |
| A02 | `appendix.tex:18-26` | Data SOP, Global SOP, 포함·제외 항목을 정의한다. | derivation의 counting convention을 세운다. | 자연 |
| A03 | `appendix.tex:28-33` | ViT dimensions와 각 image에 적용되는 조건을 정의한다. | counting convention에 notation을 붙인다. | 자연 |
| A04 | `appendix.tex:35-45` | affine projection의 Data SOP와 Global SOP를 유도한다. | notation을 첫 reusable count에 적용한다. | 자연 |
| A05 | `appendix.tex:49-57` | GELU의 signed cubic, division, fixed gains의 SOP 항목을 설명한다. | linear count에서 composed nonlinearity로 확장한다. | 자연 |
| A06 | `appendix.tex:78-88` | LayerNorm branches, variance events, learned scale의 SOP를 합산한다. | GELU와 병렬로 second composed operation을 센다. | 자연 |
| A07 | `appendix.tex:110-113` | attention SOP에 포함되는 projections, scores, Softmax, weighted sum을 밝힌다. | elementwise compositions에서 attention module로 이동한다. | 자연 |
| A08 | `appendix.tex:136-141` | attention table을 total equation으로 합친다. | component count를 attention total로 닫는다. | 자연 |
| A09 | `appendix.tex:145-154` | two projections와 GELU를 합쳐 MLP SOP를 구한다. | attention과 병렬인 block component를 유도한다. | 자연 |
| A10 | `appendix.tex:156-171` | LayerNorm, attention, MLP를 Transformer block SOP로 합친다. | component counts를 Transformer block으로 조립한다. | 자연 |
| A11 | `appendix.tex:172-177` | $H=4D$ 조건으로 block SOP를 특수화한다. | general block count를 evaluated configuration에 맞춘다. | 자연 |
| A12 | `appendix.tex:181-201` | stem, final LayerNorm, head를 더해 end-to-end ViT SOP를 정의한다. | block count를 full model count로 확장한다. | 자연 |
| A13 | `appendix.tex:203-206` | ViT-S/B/L dimensions와 class counts를 대입 조건으로 제시한다. | end-to-end equation을 numerical table에 연결한다. | 자연 |
| A14 | `appendix.tex:225-228` | evaluated classifier가 SOP model과 같은 TTFS linear composition을 사용하고 final LayerNorm의 `Potential`을 직접 소비한다고 명시한다. | numerical result와 evaluated path의 operation-count boundary를 일치시킨다. | 자연 |
| A15 | `appendix.tex:231-235` | operator identities를 noise-free finite declared domains로 제한한다. | SOP derivation에서 functional derivation으로 새로 시작한다. | 자연 |
| A16 | `appendix.tex:239-259` | $\phi_{\mathrm{NP}}$, $\phi_{\mathrm{NL}}$, positive lower endpoint와 time window를 정의한다. | primitive mappings의 domain을 세운다. | 자연 |
| A17 | `appendix.tex:263-267` | fixed-gain pulse-width integration을 정의한다. | potential-to-time mappings 뒤에 temporal readout을 둔다. | 자연 |
| A18 | `appendix.tex:281-290` | $\psi_{\mathrm{ED}}$ mapping과 alternative composition을 구분한다. | temporal readout interface를 확장한다. | 자연 |
| A19 | `appendix.tex:292-299` | oriented pulse width에서 multiplication identity를 유도한다. | primitives를 첫 arithmetic composition에 적용한다. | 자연 |
| A20 | `appendix.tex:300-309` | shared positive domain에서 division identity와 output range를 유도한다. | multiplication 다음 second arithmetic composition을 보인다. | 자연 |
| A21 | `appendix.tex:313-320` | normalized decoding으로 exponential mapping을 얻는다. | division 뒤에 GELU와 Softmax에 필요한 exponential을 완성한다. | 자연 |
| A22 | `appendix.tex:324-344` | power identity, reference scale, signed cubic, floor와 clamping 조건을 설명한다. | basic compositions를 GELU cubic으로 확장한다. | 자연 |
| A23 | `appendix.tex:348-361` | scaled dot product에 exponential과 division을 적용해 Softmax를 구성한다. | arithmetic identities를 Transformer operation에 적용한다. | 자연 |
| A24 | `appendix.tex:363-390` | signed branches와 exponential difference로 LayerNorm을 구성한다. | Softmax와 병렬인 second Transformer operation을 보인다. | 자연 |
| A25 | `appendix.tex:392-398` | identities의 finite bounds, positive domain, deadline, masking, clamping 조건을 모은다. | operator derivation의 validity conditions로 닫는다. | 자연 |
| A26 | `appendix.tex:403-405` | checkpoints와 evaluation/calibration populations 표를 소개한다. | functional derivation에서 experimental details로 전환한다. | 자연 |
| A27 | `appendix.tex:452-454` | conversion rows가 같은 tanh-based GELU와 power composition을 쓴다고 밝힌다. | model table에서 common operator setting으로 이동한다. | 자연 |
| A28 | `appendix.tex:456-466` | ViT precision, time constant, two-pass calibration, quantiles, limits를 상세히 적는다. | common setting을 reproducible calibration procedure로 구체화한다. | 자연 |
| A29 | `appendix.tex:468-470` | software와 GPU environment를 기록한다. | experimental details를 execution environment로 닫는다. | 자연 |
| A30 | `appendix.tex:475-489` | simulated timing-noise grid, observation margin, seeds, intervals, deadline-miss aggregation과 BrainScaleS-2 measurement가 아님을 밝힌다. | Methodology의 timing-noise model에 대응하지만 main text reference가 없다. | 도약 |

Appendix 자체는 대체로 `counting assumptions → components → block → full ViT`, `primitive domains → arithmetic compositions → Transformer operations → validity conditions`, `checkpoints → calibration → environment`, `simulated timing-noise protocol`의 네 흐름으로 자연스럽다. 문제는 마지막 simulated timing-noise evidence가 main text에서 호출되지 않는다는 점이다.

## 우선순위별 흐름 문제

이 목록은 문장 다듬기보다 논증 순서와 근거 범위를 먼저 고쳐야 하는 항목이다.

### P0. Measured primitive-level error의 완료 상태 불일치

Experiment는 `Results pending`이며 measurements가 끝난 뒤 결과를 보고한다고 적지만, Abstract, Introduction, Conclusion은 measured distributions를 model에 주입해 평가한 것으로 서술한다.

현재 evidence에 맞춰 완료형 주장을 제한하거나, 결과가 완성된 뒤 같은 상태를 Abstract부터 Conclusion까지 일치시켜야 한다. 이 문제가 남으면 논문의 마지막 evaluation axis가 비어 있고 Conclusion이 본문보다 앞선다.

### P0. Primitive composition에서 evaluated Transformer로 가는 다리 누락

Preliminaries는 attention, normalization, activation, linear, residual operations를 열거하지만 Framework는 GELU와 scaled dot product 뒤에 Softmax와 LayerNorm을 building blocks로만 언급한다. Methodology는 곧 finite ranges로 넘어가고 Experiment는 converted models를 평가한다.

Framework 끝에서 composed functions가 full Transformer operations에 어떻게 대응하는지와 evaluated models마다 실제로 교체한 operation subset을 밝혀야 한다. 현재 paper configuration의 GPT-2 `gelu_new`는 composed GELU를 사용하며, direct evaluation은 dense MLP ablations와 다른 configured activations에만 해당한다.

### P0. Hardware evidence 범위의 층위가 흔들림

원고는 prior work에 known hardware realizations가 있다는 주장, 본 연구가 BrainScaleS-2에서 selected $\Phi$ encoding operators를 측정했다는 주장, 그 distributions를 model locations에 주입했다는 주장을 구분해야 한다.

주석 처리된 한국어 Abstract는 영문보다 넓게 모든 primitive operators를 구현하고 측정한 것처럼 읽힌다. 한국어 문장을 다시 사용할 경우 영문의 selected encoding operators 범위와 맞춰야 하며, perplexity에 percentage point 단위를 붙이지 않아야 한다.

### P0. Range 분류와 calibration setup의 대응 불명확

Methodology는 calibration을 세 range 유형 중 둘째에만 적용한다고 하지만, Experiment는 LayerNorm log operation과 GELU cubic operation의 upper bound를 layer-wise calibration으로 정한다고 적는다.

두 upper bounds가 세 유형 중 어디에 속하는지, calibrated upper bound와 fixed positive lower bound가 어떤 역할을 나누는지 같은 위치에서 설명해야 한다.

### P1. Error evaluation의 순서가 되감김

Methodology는 deadline-miss examples 뒤에 measured distributions를 제시하고 다시 observation margin으로 돌아간다. Experiment는 discrete-time simulation 뒤에 바로 pending measured errors로 가며, 완료된 simulated timing-noise sweep를 호출하지 않는다.

더 자연스러운 순서는 `spike-time noise and deadline misses → observation margin → simulated timing-noise result → measured primitive-level distributions → model-level result`이다.

### P1. SOP와 discrete-time simulation의 결과 해석 부족

Table~3은 TTFSFormer와 ours 모두 synaptic operations per inference를 보고하므로 Conclusion의 `comparable operation counts`는 같은 counting unit과 보고된 수치로 지지된다. Discrete-time simulation은 동기 문단이 figure 뒤에 있고 실제 observation은 caption에만 있다.

Operation-cost prose에 두 방법이 같은 per-inference synaptic-operation unit을 사용한다는 해석 한 문장을 둘 수 있지만, `comparable` 자체를 제거할 필요는 없다. Discrete-time subsection에는 setup과 핵심 observation을 분리한 prose가 필요하다.

### P2. Related Work의 비교 축과 Appendix caveat 배치

ANN-to-SNN conversion paragraph는 TTFSFormer와 본 연구를 한 번 대비한 뒤 Wang et al.을 소개하고 다시 본 연구로 돌아간다. prior methods를 모두 소개한 뒤 한 번만 contrast하면 흐름이 선명하다.

Appendix는 evaluated classifier와 SOP model이 같은 TTFS linear composition을 사용한다고 명시하므로, 이전 dense-classifier caveat는 폐기한다.

## 과거 항목 재유입 감사

Git 이력은 명시적 재유입, 이전 결과의 미제거, 그리고 결과 파일을 바꾸지 않은 protocol relabeling을 구분한다.

| 판정 | 현재 원고 | Git 근거 | 조치 |
|---|---|---|---|
| 확실한 재유입 | `experiment.tex:40,46,52,57`의 네 ViT row | `12344e2`와 `0e9a9ea`가 common-threshold 수치를 model-specific threshold 결과로 교체했지만, `823a336`과 `03a2b4c`가 각각 이전 수치를 다시 넣었다. 이후 main repository의 `abc98b7`이 global-range contract 자체를 폐기했으므로 두 수치 묶음 모두 current evidence가 아니다. | 새 verified local-range artifacts가 나오기 전에는 current results로 사용할 수 없다. |
| stale 유지 | `experiment.tex:82,86,89`의 RoBERTa/GPT-2 row와 최대 $0.11$ pp 요약 | 이 NLP 수치는 ICLR 최초 commit `74debe6`부터 계속 남았으며 paper history에서 제거된 적이 없다. 그러나 `abc98b7` 이후 old global-range results가 되어 local-range Table 4 rerun으로 교체해야 한다. | 재유입으로 부르지 않고 미교체된 old result로 분류한다. Abstract와 Introduction의 요약도 함께 보류한다. |
| protocol relabeling | `appendix.tex:475-500`의 simulated timing-noise sweep와 `ViT-noise-eval.pdf` | PDF는 `12344e2`에서 마지막으로 바뀌었고, 당시 prose는 $\sigma_t=r_t(2\theta)$, $\theta=40$, no per-layer calibration이라고 명시했다. `03a2b4c`는 PDF를 바꾸지 않은 채 frozen layer-wise calibration과 $\sigma_t=r_tT$로 설명만 교체했다. | 새 local-range sweep가 완료되기 전에는 current local-window result로 부를 수 없다. |
| 완료 상태 불일치 | Abstract, Introduction, Methodology, Experiment roadmap, Conclusion의 BrainScaleS-2 measured-distribution evaluation | completion-form claim과 `Results pending`은 ICLR 최초 commit `74debe6`부터 함께 존재했다. `e4abe64`가 문장을 다듬었지만 pending panel을 유지했다. | commit-level 재유입은 아니며, 완료 전까지 완료형 주장을 미래형 또는 계획 범위로 제한한다. |
| 이번 검토에서 복구 | GPT-2 activation remains dense | `12344e2`에서 제거됐지만 stale Overleaf commit `823a336`이 10분 뒤 되살렸고 `e4abe64`가 현재 형태로 다시 썼다. | 현재 working tree에서 제거됐다. |
| 이번 검토에서 복구 | quantifying the resulting clipping error | `12344e2`에서 `accounting for clipping`으로 교정됐지만 `823a336`이 10분 뒤 옛 문구를 복구했다. | 현재 working tree에서 승인된 wording으로 복구됐다. |

### Git provenance

두 번의 혼합 지점이 확인된다.

1. `12344e2`는 두 문제 문구를 제거하고 ViT 표를 model-specific threshold 결과로 교체했다. 이어진 `823a336`은 Overleaf Git bridge가 만든 commit이며, PDF와 새 logic note를 제외한 ICLR text edits를 사실상 `74debe6` 상태로 되돌렸다. 첫 번째 재유입은 이 stale synchronization에서 발생했다.
2. Main repository의 `abc98b7`, `38c9ad7`, `dc09bdb`, `102493c`, `0a6c059`는 global range 제거, local-range rerun, summary authentication, old calibration retirement를 순서대로 기록한다. 그 뒤 paper commit `03a2b4c`는 setup prose를 local range로 바꾸면서 ViT 표는 common-threshold 수치로 되돌리고, unchanged noise PDF를 local-window protocol로 다시 설명했다. 현재 가장 큰 result/protocol 혼합은 이 commit에서 생겼다.

`e4abe64`는 원고 전체를 다듬으면서 GPT-2 dense 문구를 더 일반적인 문장 속에 유지하고 BrainScaleS-2 주장을 완료형으로 정리했다. 그러나 이 commit은 해당 항목을 제거한 뒤 복원한 지점은 아니다.

## 복구 및 제거 대상 판정

사라진 교정 중 현재 contract와 양립하는 항목만 복구하고, global-range 결과와 설정은 되살리지 않는다. Old result를 current setup에 맞춰 relabel한 부분은 제거한다.

### 즉시 복구 또는 제거

이 항목들은 새 실험을 기다리지 않고 현재 evidence boundary에 맞게 원고 범위를 바로잡는다.

1. Preliminaries에는 embedding과 output head의 replacement boundary를 별도로 단정하는 문장을 두지 않는다. 이전의 conventional boundary 문장과 이를 대체하려던 end-to-end inclusion 문장은 모두 제거한다.
2. Appendix의 old timing-noise figure와 completed-sweep prose를 제거한다. 현재 `ViT-noise-eval.pdf`는 removed global-range campaign의 artifact이므로 $\theta=40$이나 $\sigma_t=r_t(2\theta)$를 active manuscript에 복구하지 않는다. Original protocol은 provenance record에만 남기고, 새 local-range sweep가 검증된 뒤 새 figure와 설명을 넣는다.
3. Experiment의 deterministic/noise-free framing: `Noise-Free Conversion Fidelity and Operation Cost` 구분과 primitive-level error injection 전 결과라는 해석을 복구한다. 원고에 반영됐다.
4. Limitations의 omitted non-ideality scope: membrane-potential storage noise, temporal correlation, routing or spike loss, mismatch와 temperature dependence가 현재 error model 밖이라는 취지를 복구한다. 이전 문장을 그대로 붙이기보다 current limitation paragraph에 evidence boundary로 통합한다. 원고에 반영됐다.

### Verified rerun 뒤 복구

이 항목은 새 artifacts의 identity와 completeness가 검증된 뒤에만 current result 설명으로 복구한다.

Current local-range artifacts가 검증된 뒤 모든 reported conversion run의 float64, time constant, complete active-site calibration, frozen replay와 noise injection scope를 model family 전체에 대해 다시 명시한다. 현재 old table을 유지한 채 new protocol이라고 쓰지는 않는다.

### 이미 복구됨

두 문제 문구는 current working tree에서 이미 교정됐다.

`quantifying the resulting clipping error`는 `account for clipping` 범위로 교정됐고, GPT-2 dense activation 문장은 제거됐다.

### 복구 금지

이 항목들은 폐기된 contract나 더 이상 필요하지 않은 drafting marker이므로 복구하지 않는다.

Global or model-specific threshold selection, `12344e2`의 historical ViT 수치, current table의 common-threshold 수치, validation-based threshold confirmation, absolute energy estimate와 template-abstract TODO는 현재 원고에 되살리지 않는다.

현재 원고에 다시 들어오지 않은 제거 항목도 확인했다. Absolute energy column, fixed pJ/SOP conversion, hardware energy superiority, continuous-time global `theta`/`attention_theta`, model-specific cubic distinction, global-range selection panel, analog substrate에 Transformer가 전혀 없다는 absolute claim은 active manuscript에 없다. Discrete-time figure의 $\theta=20$은 별도로 유지하기로 한 clock-driven experiment이므로 global-range regression으로 세지 않는다.

`known hardware realizations`, `each primitive operator ... demonstrated`, `analog-compatible`, `exact in an ideal continuous-time system`, TTFSFormer의 complexity/noise에 대한 부정형 비교는 current checklist에서 아직 해결되지 않은 claim audit 대상이다. 이전에 완료된 제거가 다시 들어온 것으로 확정하지는 않았지만, publication-ready wording으로 승인된 것도 아니다.

## Event-driven 문장 예상 리뷰 질문

Discretization 문단의 event-driven 가능성은 이론적 구현 경로와 현재 실험의 검증 범위를 분리하지 않으면 hardware claim으로 읽힌다.

이 검토에서 event는 data spike, target neuron 전체에 multicast되는 layer-shared reference spike, 그리고 integration window를 종료해 readout을 발생시키는 scheduled observation deadline을 포함한다. Deadline의 event 여부가 아니라 생성, 분배, fan-out 비용, timing skew와 reset/readout 구현이 검증 대상이다.

### P0. 실행 의미와 직접 근거

이 질문들은 현재 문장만으로 답하기 어렵고, 답변에 따라 문장을 약화하거나 구현 설명을 보강해야 한다.

1. Data spike, multicast reference spike와 scheduled observation deadline으로 구성된 event set과 각 event의 역할을 원고 어디에서 정의하는가?
2. 모든 operator state를 매 step 갱신한 simulation이 spike arrival에서만 state를 갱신하는 event-driven 실행의 가능성을 어떤 직접 증거로 뒷받침하는가?
3. PWM integration, exponential response, division, Softmax, LayerNorm과 GELU 각각에서 event arrival 사이의 state evolution을 어떻게 처리하는가?
4. Expected spike가 오지 않을 때 scheduled deadline event가 reset state를 readout하는 경로와 그 비용을 어떻게 구현하고 계산하는가?
5. 이 문장은 구현된 execution mode를 설명하는가, 아니면 특정 hardware mapping 없이 가능한 설계 방향만 제시하는가?

### P1. 비용과 hardware 가정

이 질문들은 가능성 자체보다 event-driven 표현이 암시하는 효율과 배포 범위를 제한한다.

1. Discrete time에서 state를 매 step 갱신하지 않는다면 scheduled deadline, event queue, timestamp comparison과 지연된 state update 비용은 무엇인가?
2. 각 activation이 one spike를 내는 TTFS에서 실제 event sparsity는 얼마이며, rate coding 대비 spike 감소가 실제 latency나 energy 감소로 이어지는가?
3. Data spike의 fan-out 외에 global reference, synchronization과 deadline delivery를 SOP가 어디까지 포함하는가?
4. Signed or differential rails, local time windows, simultaneous events와 layer barrier를 기존 event-driven chip이 지원한다는 근거가 있는가?
5. 필요한 timestamp resolution과 local window당 step 수가 hardware timer와 routing capacity에 현실적인가?
6. Analog state가 event 사이에도 적분하거나 감쇠한다면 static current와 leakage를 제외하고 event-driven efficiency를 논할 수 있는가?
7. Coarse discretization에서 발생하는 accuracy loss와 event scheduling 절감 사이의 trade-off는 어디에서 평가되는가?

### 답변 가능한 evidence boundary

현재 evidence는 discrete-time numerical compatibility를 지지하지만 event-driven efficiency나 특정 chip deployment를 검증하지 않는다.

Causal event-to-deadline rails는 data spike, multicast reference spike와 scheduled deadline event로 연산을 구성할 이론적 경로를 제공한다. Event 사이에는 analog state가 적분하거나 감쇠하므로 discrete state update가 필요하지 않다. 그러나 maintained simulation은 모든 step에서 state를 갱신하고, cost model은 routing, event queue, deadline controller, reset, memory와 static analog cost를 포함하지 않는다. 따라서 문장을 유지한다면 “in principle”과 simulation scope를 함께 쓰고, 구현 또는 효율을 입증했다는 표현은 피해야 한다.

## 권장 전체 순서

현재 내용을 버리지 않고 연결만 다시 세우려면 다음 순서가 가장 자연스럽다.

1. TTFS motivation과 Transformer operation gap.
2. Evaluated Transformer operations, neuron dynamics, finite-domain assumptions.
3. Cubic example, $\Phi/\Psi$ definitions, primitive operators.
4. Multiplication, division, exponential, power에서 Softmax, LayerNorm, GELU, attention으로 이어지는 complete composition과 model별 replacement scope.
5. Finite potential ranges, spike-time windows, layer-wise calibration, clamping.
6. Noise-free conversion fidelity와 SOP scope.
7. Discrete-time simulation과 simulated timing-noise result.
8. Measured primitive-level errors, observation margin, model-level result.
9. Limitations: operation-count boundary, simulation scope, full-chip deployment의 미검증 범위.

이 순서는 `problem → assumptions → construction → converted model mapping → deterministic constraints → conversion evidence → timing sensitivity → measured errors → limitations`로 한 번만 앞으로 진행한다.

## 주장 강도 감사

이 감사는 현재 원고의 완료 상태, 수식 조건, 실제 평가 경계와 문장 강도를 대조해 reviewer가 직접 공격할 지점을 분류한다.

### P0 제출 차단 주장

이 항목들은 현재 원고 내부에서 반박되거나, 결과가 아직 없거나, 평가 대상과 비용 모델의 경계가 달라 제출 전에 반드시 정리해야 한다.

| 위치 | 강한 주장 | 예상 질문 | 허용 가능한 범위 |
| --- | --- | --- | --- |
| `iclr2027_conference.tex:80`; `intro.tex:13,15`; `methodology.tex:42`; `conclusion.tex:5,11` | BrainScaleS-2에서 측정한 empirical distributions를 변환 모델에 주입해 평가했다고 완료형으로 말한다. | `experiment.tex:113-136`이 `Results pending`인데 측정 설정, 분포와 task result는 어디에 있는가? | 측정과 model-level injection 결과가 검증되기 전에는 계획으로 두고, 완료 뒤에도 selected primitives의 측정을 simulation에 주입한 평가라고 한정한다. |
| `iclr2027_conference.tex:79`; `intro.tex:11`; `experiment.tex:40-97` | 현재 table 수치로 classification degradation의 최대값과 GPT-2 perplexity 변화를 확정한다. | local-range rerun과 metric verification이 진행 중인데 어떤 artifact가 headline numbers를 지지하는가? | Complete evaluation bundle이 검증될 때까지 수치는 provisional로 취급하고, 이후에도 evaluated checkpoints와 stated populations로 범위를 제한한다. |
| `iclr2027_conference.tex:79`; `framework.tex:52-64`; `conclusion.tex:8` | 모든 primitive가 `known hardware realizations`을 갖고 prior analog hardware에서 demonstrated되었다고 말한다. | 인용된 hardware가 같은 input-output mapping과 parameter range를 직접 구현했는가? 전체 composition을 한 substrate에서 연결했는가? | Cited mechanisms, proposed operator abstraction, full composition과 full-chip execution을 서로 다른 evidence level로 구분한다. |

### 이번 수정에서 해결

2026-09-24에는 blanket exactness와 arbitrary ANN function 일반화를 현재 원고에서 제거했다.

- Methodology는 operator equations가 stated domains의 ideal continuous-time system에서 exact하게 성립하고, domain choice가 clipping boundaries와 spike-time windows를 통해 finite-domain approximation accuracy에 영향을 준다고 제한한다.
- Framework는 arbitrary $F$ 대신 listed arithmetic functions와 evaluated Transformer operations만 composition claim의 대상으로 둔다.
- Framework의 negative-log example은 initial potential, unit threshold, reference time과 $w\tau_s=\theta$ 조건을 명시한다. SNN 독자를 전제로 elementary threshold-crossing integration은 생략한다.
- Preliminaries는 double-exponential 식을 LIF의 spike-evoked component로 한정하고, single-exponential approximation에 fast time constant가 지난 observation condition을 명시한다. $\propto$를 유지하므로 coefficient를 주장하지 않으며, component 표현이 기존 membrane state와 분리한다.
- Output-head verification은 ViT, BERT, RoBERTa와 GPT-2의 final head가 `Potential`을 직접 받고, ViT accuracy path와 SOP classifier가 같은 `SpikingLinear` composition을 사용함을 확인한다. 이전 dense-classifier 지적은 폐기한다.
- TTFSFormer와 ours의 cost totals는 모두 synaptic operations per inference이며 spike 또는 reference-event delivery의 receiving fan-out을 센다. `Comparable operation counts`는 유지하되 identical circuit 또는 measured energy comparison으로 확대하지 않는다.
- Conversion fidelity는 TTFSFormer처럼 source ANN과 converted SNN을 직접 비교한 행에는 그대로 사용한다. Quantization-aware training처럼 추가 절차가 있는 prior row의 $\Delta$는 full method pipeline이 보고한 net difference라고 명시한다.
- Related Work와 Introduction은 prior work가 하지 않았다는 부정형 또는 literature-wide 주장을 제거한다. 대신 TTFSFormer의 construction을 기술하고, 명시적 mapping의 synaptic operation counts와 timing 및 primitive-level mapping error 평가를 본 연구가 추가하는 범위로 제시한다.
- Analog hardware 문단은 Transformer deployment가 제한됐다는 포괄적 서술과 Softmax, LayerNorm, GELU를 단일 원인처럼 연결한 문장을 제거한다. 대신 이 연산들의 efficient realization을 operator-design challenge로 한정한다.
- Signed integer power는 time-constant ratio와 두 signed branch의 parity-dependent 결합으로 모든 positive integer exponent에 대해 algebraically 구성된다. Evaluated instance는 GELU의 $p=3$이지만 별도 범위 축소는 필요하지 않다.
- Single-spike 문장은 encoded activation의 communication cost로 이미 한정되어 있고 SOP가 reference, synchronization, internal encoding과 signed branches를 별도로 계산하므로 추가 수정하지 않는다.
- Potential--time dual representation은 $\Phi:\mathcal V\to\mathcal T$와 $\Psi:\mathcal T\to\mathcal V$ mapping families의 pair로 원고 안에서 직접 정의된다. Inverse나 bijection theorem을 주장하지 않으므로 추가 disclaimer를 넣지 않는다.

### P1 범위를 좁혀야 하는 주장

이 항목들은 방향 자체는 방어 가능하지만 현재 표현이 framework 전체, 전체 hardware 또는 전체 literature로 쉽게 확대 해석된다.

- Continuous spike-time evaluation은 algebraic tensor evaluation이며 transistor dynamics, routing, arbitration, leakage 또는 calibrated device parameters를 시뮬레이션한 것이 아니다.
- Discrete-time compatibility evidence는 ViT-B/16, fixed 500 ImageNet-1k images, $\theta=20$과 no timing noise 조건의 numerical case study이다.
- Robustness는 computational Gaussian timing-noise stress test 또는 selected primitive measurements를 simulation에 주입한 sensitivity를 뜻하며 full-chip robustness가 아니다.
- `Three types of potential ranges`와 `calibration only in the second case`는 보편적 taxonomy가 아니라 이 연구의 range-selection policy로 써야 한다.
- Primitive table의 equality에는 bounded domains, shared log domain, common observation deadline, delivered events와 compensating gains가 필요하다.
- Analog platform capability, exact primitive mapping, composed operator execution과 full Transformer deployment는 별도 근거 수준이다.

### 유지해야 할 방어 문장

현재 Appendix의 제한 문장은 삭제하기보다 main claims와 같은 강도로 끌어올려야 한다.

- `appendix.tex:11-14`: SOP는 stated TTFS mapping의 operation-count model이며 complete physical implementation이 아니다.
- `appendix.tex:231-235`: identities는 finite declared domains의 noise-free functional compositions이며 physical-circuit exactness가 아니다.
- `appendix.tex:392-398`: LayerNorm identity는 positive domains, common deadline, branch masking, floor와 clamping이 결과를 바꾸지 않을 때만 적용된다.
- `appendix.tex:475-500`: 이 sweep의 stated scope는 computational Gaussian model이며 BrainScaleS-2 measurement가 아니다. 별도의 artifact-validity 검증 없이 current local-range result로 승격하지 않는다.

### 권장 수정 순서

문장 polishing 전에 internal contradiction부터 닫아야 한다.

1. Pending BrainScaleS-2 claim을 결과 상태와 일치시킨다.
2. Rerun이 끝날 때까지 headline numbers를 보류하고 verified bundle에서만 복구한다.
3. 모든 learned affine Transformer operations와 output heads가 operator-backed임을 밝히되, preprocessing, embedding lookup, shape-only operations, tensor return, argmax와 loss는 이 주장 밖이라고 구분한다.
4. Exactness와 arbitrary ANN scope를 conditional operator identities로 제한한다.
5. Hardware realization을 evidence level에 맞추고, operation-count comparison은 synaptic operations per inference 범위로 유지한다.
6. Literature-wide negative claims와 discrete-time, robustness, event-count generalization을 case-specific wording으로 낮춘다.

## Figure 3 수정 상태

2026-09-24 사용자 첨부 목록 1–12번을 기준으로 결과와 무관한 서술을 수정했다. 실험 수치와 물리 측정 절차의 대조는 별도 완료 조건으로 남긴다.

이 수정 범위에서는 $\phi_{\mathrm{NP}}$, $\phi_{\mathrm{NL}}$를 operator로 표기한다. Encoder는 Transformer encoder block을 뜻할 때만 사용하며, 별도의 encoder 정의를 도입하지 않는다.

- 완료: 2, 3, 4, 5, 9, 10. 기여 및 문헌 비교의 측정 범위, Gaussian 설정, CCT 소개와 평가 설정, Appendix sweep의 역할을 반영했다.
- 부분 완료: 1, 6, 7, 8. Abstract와 Conclusion의 평가 범위, Figure 3 구성 및 protocol은 수정했으며 실제 곡선과 결과 해석은 대기한다.
- 부분 완료: 11. model-weighted scale의 사용 주장, quantization과 mapping injection 및 90th-percentile 실험 예고를 정리하고 실제 summary의 screening 모집단과 pair 선택을 반영했다. 물리 구현의 시간 설정, repetitions, acceptance rules는 원자료 대조가 남았다.
- 12는 이전 수치를 새 본문에 사용하지 않는 조건으로 유지한다. Noise 허용 수준이나 hardware와의 격차는 formal 결과 후 판단한다.

Figure 3(a)는 CCT-7, ViT-S/16, ViT-B/16의 공통 noise multiplier를 비교하고, (b)는 ViT-B의 첫 $K$개 encoder block에만 noise를 적용한다. 두 panel은 noise가 없는 동일 converted model 대비 top-1 변화만 표시하며 별도 baseline accuracy나 baseline curve는 추가하지 않는다. NP/NL은 독립 Gaussian scale을 사용하고 두 physical coordinate를 하나의 joint measurement로 해석하지 않는다.

CCT 출처는 [공식 저장소](https://github.com/SHI-Labs/Compact-Transformers)의 300-epoch CIFAR-10 CCT-7/3x1 checkpoint 설명과 citation을 확인했다. 후속 작업은 [[todo#Manuscript Revision Master Checklist#P1 Robustness and Non-Ideality Evidence]]에 기록한다.
