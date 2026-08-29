# Fixed Potential Range Audit

이 보고서는 inference 중 관측한 tensor extrema로 membrane-potential range를 다시 정하는 위치를 전수 조사하고, fixed range와 calibration으로 교체하는 방법을 정리한다.

## 결론

현재 maintained model adapter에는 live activation으로 `PotentialBounds`를 만드는 production 위치가 없다. GPT-2 residual은 analytic sum과 optional frozen calibration을 사용하며 evaluator artifact lifecycle 연결은 별도 후속 단계다.

Transform operator의 time window는 고정되어 있다. 최초 감사의 32곳 중 GPT-2 attention의 2곳, Gaussian 및 deterministic LayerNorm의 각 3곳, ordinary LayerNorm helper의 1곳은 fixed/analytic range로 교체되었다.

핵심 결과는 다음과 같다.

- `utils/transforms/`의 production operator는 입력 range, threshold $\theta$, time constant $\tau_s,\tau_m$, tensor shape에 대한 interval arithmetic으로 output range를 계산한다. `TimeBounds`를 live activation extrema로 만드는 위치는 없다.
- 최초 감사에서 shared model operator와 네 model family의 live activation extrema 위치는 각각 7, 4, 4, 9, 8곳이었다. Shared operator와 네 model adapter를 모두 전환한 현재 합계는 0곳이다.
- `SpikingLinear`, `SpikingConv2d`, `SpikingConv1D`는 upstream fixed input range의 양 끝점을 weight 부호별로 선택해 exact affine range를 memoize하고 이후 parameter mutation을 검증한다. `SpikingLayerNorm`의 Gaussian과 deterministic 경로도 ablation별 frozen weight, bias, output domain을 공유한다.
- ViT, BERT, RoBERTa, GPT-2 attention adapter는 이제 Gaussian value integration의 fixed range $[-S_{\max}\theta,S_{\max}\theta]$를 spiking backend와 공유한다.
- 기존 quantile 수집은 진단용으로 남아 있다. ViT layer-wise calibration은 별도 artifact에 signed extrema, fixed-bin histogram, quantile, margin, training-subset identity, layer clipping을 보존한다.
- ViT는 parameter range, preprocessing input, encoder entry, activation, attention output, residual range 전환을 완료했다. 남은 model family는 같은 순서로 전환해야 한다.

## 감사 범위와 판정 기준

감사는 maintained path의 모든 `PotentialBounds`와 `TimeBounds` 생성, `Potential` 생성, activation extrema, pretrained parameter extrema, attention range 전달, evaluator quantile 수집을 포함한다.

대상은 다음과 같다.

- `utils/transforms/`: $\phi_{\mathrm{NP}}$, $\phi_{\mathrm{NL}}$, $\psi_{\mathrm{Int}}$, $\psi_{\mathrm{ED}}$, $f_{\mathrm{Mul}}$, $f_{\mathrm{Div}}$, softmin, GELU, Tanh, SwiGLU
- `utils/transformers/`: shared affine operator, LayerNorm, attention, ViT, BERT, RoBERTa, GPT-2
- `scripts/evaluation/`: ViT, BERT, RoBERTa, GPT-2 quantile collection과 static parameter perturbation

다음 사용은 위반으로 판정했다.

$$
[V_{lb},V_{ub}]
=
[\min X_{\mathrm{current\ batch}},\max X_{\mathrm{current\ batch}}].
$$

현재 tensor의 값을 검사하는 `check_domain`, clipping 횟수를 기록하는 진단, error maximum을 출력하는 evaluator 코드는 range를 만들지 않으므로 위반 수에 포함하지 않았다. `__main__` 아래의 local fixture도 production 위치에서 제외했다.

## Fixed Range의 수식 계약

Fixed range는 calibration 또는 interval arithmetic으로 inference 전에 정해지고, forward는 그 range를 읽고 clipping할 수만 있어야 한다.

목표는 모든 위치에 가장 넓은 analytic interval을 강제하는 것이 아니다. 수식으로 tight하고 depth-independent하게 정해지는 operator는 interval arithmetic을 사용하고, nonlinear range가 어렵거나 residual 반복으로 interval이 누적 확대되는 위치는 module/site별 calibration으로 고정한다.

각 layer의 Lipschitz constant가 $L_i>1$이면 $\lVert\delta x_{i+1}\rVert\le L_i\lVert\delta x_i\rVert+\lVert e_i\rVert$에 따라 이전 clipping error와 propagated range가 함께 증폭될 수 있다. 따라서 calibration clamp는 unbounded output만 잘라내는 보조 기능이 아니라 depth 방향의 range를 다시 고정하는 경계이며, layer별 clipping rate와 최종 task accuracy를 함께 검증해야 한다.

### Potential과 time window

논문의 membrane-potential space $\mathcal V$와 spike-time space $\mathcal T$ 사이 변환은 fixed potential range $(V_{lb},\theta)$와 finite time window $[0,T]$를 전제로 한다.

$\phi_{\mathrm{NP}}$의 affine encoding은

$$
t=\theta-V,
\qquad
T=\theta-V_{lb},
$$

이고 symmetric range $[-\theta,\theta]$에서는 $T=2\theta$이다. $\phi_{\mathrm{NL}}$의 negative-log encoding은 positive range $[V_{\min},V_{\max}]$에 대해

$$
t=\tau_s\bigl(\log V_{\max}-\log V\bigr),
\qquad
T=\tau_s\bigl(\log V_{\max}-\log V_{\min}\bigr)
$$

이다. 두 time window는 모두 configuration과 input range에서 정해지며 current tensor의 extrema를 필요로 하지 않는다.

### Interval arithmetic

두 potential이 $X\in[l_x,u_x]$, $Y\in[l_y,u_y]$이면 residual addition과 multiplication range는 다음과 같이 정해진다.

$$
X+Y\in[l_x+l_y,u_x+u_y],
$$

$$
XY\in
\left[
\min\{l_xl_y,l_xu_y,u_xl_y,u_xu_y\},
\max\{l_xl_y,l_xu_y,u_xl_y,u_xu_y\}
\right].
$$

[[utils/transforms/primitive.py#signed_pulse_width_modulation_operator]]와 composed operator는 이 방식을 사용하므로 current activation extrema에 의존하지 않는다.

### Affine projection

Pretrained affine projection은 weight를 checkpoint loading과 static perturbation 뒤에 한 번 읽으면 output range를 정확한 interval arithmetic으로 고정할 수 있다.

입력 feature가 모두 $x_i\in[l_x,u_x]$이고 $w_{ji}^{+}=\max(w_{ji},0)$, $w_{ji}^{-}=\min(w_{ji},0)$이면

$$
l_j=\sum_i\left(w_{ji}^{+}l_x+w_{ji}^{-}u_x\right)+b_j,
$$

$$
u_j=\sum_i\left(w_{ji}^{+}u_x+w_{ji}^{-}l_x\right)+b_j.
$$

전체 scalar potential range는 $V_{lb}=\min_j l_j$, $V_{ub}=\max_j u_j$를 사용한다. Linear, Conv2d, GPT-2 Conv1D는 이 식을 feature 또는 kernel dimension에 적용하고, 같은 immutable input range에 대한 결과를 parameter version별로 memoize한다.

Signed identity-code PWM은 $l_x\le0\le u_x$를 요구한다. $t(x)=u_x-x$, $t(0)=u_x$이므로 deterministic pulse width는 $t(0)-t(x)=x$이고, asymmetric calibrated range도 별도의 `theta` 기반 range로 교체하지 않는다.

### Layer Normalization

LayerNorm은 internal positive range와 final affine output range를 분리해야 한다.

`clip_margin=m`, upper threshold가 $\theta$이면 dual-rail input range와 variance encoding range는 현재 composition에서

$$
x_k^{+},x_k^{-}\in[m,\theta-m],
\qquad
V_{\sigma^2}\in[m^2,(\theta-m)^2],
$$

이며 두 negative-log encoding은 같은 time window를 갖는다.

$$
T_0
=\tau_s\log\frac{\theta-m}{m}
=\frac{\tau_s}{2}\log\frac{(\theta-m)^2}{m^2}.
$$

Variance에 $\epsilon$을 더한 raw value가 upper bound를 넘으면 range를 넓히지 말고 clipping으로 기록해야 한다. 이는 finite-window LayerNorm approximation의 일부다.

$\psi_{\mathrm{ED}}$가 반환한 두 range를 $Y^{+}\in[l_+,u_+]$, $Y^{-}\in[l_-,u_-]$라 하면 signed result는

$$
Y^{+}-Y^{-}\in[l_+-u_-,u_+-l_-]
$$

로 계산할 수 있다. 두 $\psi_{\mathrm{ED}}$ 출력이 non-negative dual rail이면 $U=\max(u_+,u_-)$에 대한 완화된 대칭 range $[-U,U]$도 유효하다. 구현은 이 직관적인 대칭 range를 learned $\gamma$와 곱하고 $\beta$를 더해 observed result 없이 final output range를 정한다.

Dense LayerNorm 분기는 pre-affine normalized value $z_k$를 noise-free calibration으로 측정한 뒤, fixed $z_k\in[l_z,u_z]$와 pretrained $\gamma_k,\beta_k$에 대해

$$
l_k=\min(\gamma_kl_z,\gamma_ku_z)+\beta_k,
\qquad
u_k=\max(\gamma_kl_z,\gamma_ku_z)+\beta_k
$$

를 적용한다. Calibration을 사용하지 않는 보수적 대안은 normalized feature 수를 $d$라 할 때 $|z_k|\le\sqrt{d-1}$을 사용하는 것이다.

### Activation

Monotone activation은 input range에서 직접 fixed output range를 계산하고, GELU 또는 SiLU처럼 전체 구간에서 monotone하지 않은 activation은 operator composition 또는 calibration을 사용한다.

$$
\operatorname{ReLU}([l,u])=[\max(0,l),\max(0,u)],
$$

$$
\operatorname{Tanh}([l,u])
=[\operatorname{Tanh}(l),\operatorname{Tanh}(u)]
\subset[-1,1].
$$

논문의 tanh-based GELU approximation을 operator-composed path로 실행하면 각 $f_{\mathrm{Mul}}$, Tanh, addition의 range를 그대로 전달할 수 있다. Direct GELU, `gelu_new`, SiLU는 $x$와 $[0,1]$ gate의 곱이므로 fixed input interval $[l,u]$에서 보수적 output interval $[\min(l,0),\max(u,0)]$을 사용할 수 있다.

### Attention

Attention은 score, softmin weight, value integration의 range를 각각 고정해야 한다.

$q_d,k_d\in[-\theta,\theta]$이고 head dimension이 $D$이면 negated scaled dot product의 보수적 range는

$$
-\theta^2\sqrt D
\le
-\frac{1}{\sqrt D}\sum_{d=1}^{D}q_dk_d
\le
\theta^2\sqrt D.
$$

현재 score는 $c=\min(\theta,80)$으로 clipping하므로 softmin input range는 $[-c,c]$이다. Masked score도 같은 upper bound $c$를 사용해야 한다. 현재처럼 $80$을 대입하면 $\theta<80$에서 declared range를 벗어난다.

Softmin weight가 $w_{ij}\in[0,1]$이고 source length의 fixed maximum이 $S_{\max}$이면 Gaussian value integration을 포함하는 공통 output range는

$$
V_{\mathrm{attn}}
\in[-S_{\max}\theta,S_{\max}\theta].
$$

Noise-free evaluation에서 $\sum_jw_{ij}=1$이 보장되면 $[-\theta,\theta]$까지 줄일 수 있지만, Gaussian on/off에 따라 range를 바꾸지 않으려면 위의 공통 range를 사용해야 한다. Training dropout까지 지원하면 보수적으로

$$
V_{\mathrm{attn}}
\in
\left[-\frac{S_{\max}\theta}{1-p},
       \frac{S_{\max}\theta}{1-p}\right]
$$

를 사용한다.

## Bound 증폭 감사

Static이라는 사실만으로 range가 유용해지는 것은 아니며, 현재 일부 interval은 feature 수, source length, threshold, 또는 exponential window에 따라 실제 activation보다 훨씬 빠르게 증가한다.

### 즉시 수정해야 하는 operator contract

Calibration으로 가리기 전에 수식상 더 tight한 불변식을 적용해야 하는 operator가 있다.

`multiplication_operator`는 이전에 두 번째 operand의 declared range 대신 full encoder rail $[-\theta,\theta]$를 output interval에 사용했다. 따라서 $V\in[-\theta,\theta]$와 fixed coefficient $a=0.044715$의 곱도 intended range $[-|a|\theta,|a|\theta]$가 아니라 $[-\theta^2,\theta^2]$를 반환했다. $\theta=100$에서 실제 endpoint 약 $4.47$ 대신 $10^4$, $\theta=2000$에서 약 $89.43$ 대신 $4\times10^6$이었다.

현재 구현은 물리 encoder window $[-\theta,\theta]$를 유지하면서 caller의 factor endpoints를 그 window에 clamp한 뒤 ideal product range를 계산한다. Gaussian one-sided miss가 ideal product rail을 벗어나면 raw excursion을 saturation으로 기록하고 같은 ideal rail에 clamp한다. 이 수정은 tanh-based GELU의 $x^2$, $x^3$, fixed coefficient, fixed scale, gate multiplication에서 반복되던 불필요한 full-$\theta$ 증폭을 제거한다.

`division_function`은 positive joint domain $[a,b]$에서 generic time-difference interval을 사용하여

$$
\frac{X}{Y}\in\left[\frac{a}{b},\frac{b}{a}\right]
$$

를 반환한다. 그러나 함수가 이미 elementwise $X\le Y$를 요구하므로 deterministic ideal output은

$$
\frac{X}{Y}\in\left[\frac{a}{b},1\right]
$$

이다. Softmin은 이 구조를 더 강하게 사용하므로 ideal weight range를 $[0,1]$로 고정하고 Gaussian raw output이 이를 벗어나면 saturation으로 처리해야 한다.

Score range가 $[-c,c]$, source length가 $N$, temporal scale이 $\tau_s$이면 current softmin joint domain은

$$
a=e^{-c/\tau_s},
\qquad
b=Ne^{c/\tau_s},
$$

이고 generic division upper bound는

$$
\frac{b}{a}=Ne^{2c/\tau_s}
$$

까지 증가한다. $c=80$, $\tau_s=1$, $N=197$이면 약 $10^{71.78}$이고 $N=512$이면 약 $10^{72.20}$이다. 이는 float32 maximum 약 $10^{38.53}$을 넘는다. 실제로 float32 zero-score tensor, $N=197$, `PotentialBounds(-80, 80)`의 deterministic softmin direct check는 `normalized_exp_operator`에서 finite-bound `ValueError`를 발생시킨다.

Tanh와 sigmoid-like gate도 각각 $[-1,1]$과 $[0,1]$이라는 structural output range를 사용해야 한다. Generic division과 multiplication의 넓은 intermediate range를 그대로 반환하면 bounded activation이라는 수학적 성질을 잃고 downstream interval만 확대한다.

### Dimension에 따라 증가하는 reduction

Reduction bound는 mathematically finite해도 configured dimension과 함께 빠르게 커지므로 calibration 또는 더 tight한 구조적 식이 필요하다.

Query와 key가 $[-\theta,\theta]$이고 head dimension이 $D$이면 raw scaled dot product는

$$
\left|\frac{1}{\sqrt D}\sum_{d=1}^{D}q_dk_d\right|
\le \theta^2\sqrt D.
$$

$D=64$에서 endpoint는 $\theta=100$일 때 $8\times10^4$, $\theta=2000$일 때 $3.2\times10^7$이다. Attention은 이를 $c=\min(\theta,80)$으로 즉시 clamp하므로 raw dot-product domain은 이후 graph로 전파되지 않지만, score clamp 자체는 큰 approximation site다. Layer/head별 score calibration과 pre-clamp saturation rate가 필요하다.

Attention value integration은 current common range

$$
[-S_{\max}\theta,S_{\max}\theta]
$$

를 사용한다. 이는 $\theta=100$에서 ViT $S_{\max}=197$이면 $\pm19{,}700$, BERT/RoBERTa $S_{\max}=512$이면 $\pm51{,}200$이고, $\theta=2000$에서는 각각 $\pm394{,}000$과 $\pm1{,}024{,}000$이다. Noise-free weights가 합계 1이면 value의 convex combination이므로 $[-\theta,\theta]$가 exact하다. Gaussian weights는 개별적으로 $[0,1]$에 clamp되지만 합계 1을 보장하지 않으므로 별도 calibrated output rail과 saturation policy가 필요하다.

Affine와 convolution bound는 이전에 global weight extrema와 fan-in $F$를 사용하여 대략

$$
|Y|\le F\theta\max_i|W_i|+|b|
$$

로 증가한다. 같은 fixed parameters에서 더 tight한 analytic 식은 output별

$$
Y_j\in
\left[b_j-\theta\sum_i|W_{ji}|,
      b_j+\theta\sum_i|W_{ji}|\right]
$$

이며 convolution도 output channel별 kernel absolute sum을 사용할 수 있다. 세 maintained affine adapter는 현재 weight layout에 맞춰 이 식을 적용하고 결과를 freeze한다. 이 식도 pretrained activation distribution보다 넓을 수 있으므로 parameter-derived rail은 safety bound로 두고 layer output calibration을 별도로 적용한다. Patch projection과 MLP의 큰 fan-in이 우선 측정 대상이다.

### LayerNorm과 residual depth

Spiking LayerNorm의 positive rail이 $[m,\theta-m]$이면 log-window ratio는

$$
R=\frac{\theta-m}{m}.
$$

Exponential-difference rail은 대략 $[R^{-1},R]$이고 signed dual-rail relaxation은 $[-R,R]$이다. Learned-scale multiplication은 이제 actual gamma endpoints를 사용하므로 이전의 추가 full-$\theta$ factor는 제거되었다. 그러나 default $m=10^{-5}$에서 $\theta=100$이면 여전히 $R\approx10^7$, $\theta=2000$이면 $R\approx2\times10^8$이다. Zero input direct check에서도 actual output은 0이지만 declared range는 약 $\pm9.999999\times10^6$이었다.

Fully dense LayerNorm의 $|z_i|\le\sqrt{d-1}$ bound는 $d=768$에서 약 $27.7$로 depth-independent하지만 typical activation보다 넓을 수 있다. Spiking/mixed LayerNorm은 `clip_margin` ratio 때문에 훨씬 심각하므로 final module output을 우선 calibration하고, analytic range는 physical safety 또는 missing-calibration failure 진단에만 사용해야 한다.

Pre-norm residual은 block마다

$$
[l_{h,\ell+1},u_{h,\ell+1}]
=
[l_{h,\ell},u_{h,\ell}]
+[l_{F,\ell},u_{F,\ell}]
$$

를 반복하여 width가 depth에 따라 누적된다. ViT와 GPT-2는 post-add block output을 calibration site로 고정해야 하며, analytic residual sum은 collection 중 safety/diagnostic range로만 유지한다. BERT와 RoBERTa의 post-norm 경계는 LayerNorm이 depth-independent range를 다시 제공하지만 calibrated LayerNorm output을 쓰면 더 tight하게 유지할 수 있다.

### 우선순위 판정

Bound 증폭은 operator contract 오류와 calibration 대상이 섞여 있으므로 다음 순서로 처리해야 한다.

| 우선순위 | 위치 | 판정 | 처리 |
|---|---|---|---|
| Complete | multiplication | declared factor interval 대신 full-$\theta$ interval을 사용했음 | clamped factor endpoints로 ideal product rail 계산 완료 |
| Complete | softmin | normalized weight의 $[0,1]$ 불변식 대신 generic division range를 반환했음 | public $[0,1]$ rail 적용과 Gaussian saturation 기록 완료 |
| Complete | tanh | bounded activation 대신 transformed generic division range를 반환했음 | public $[-1,1]$ rail 적용과 Gaussian saturation 기록 완료 |
| Complete | sigmoid-like gates | GELU와 SwiGLU gate가 generic division range를 downstream product에 전달했음 | public $[0,1]$ gate와 Gaussian saturation 기록 완료 |
| Complete | division | 알려진 operand ordering을 generic exponential ratio가 버렸음 | constrained wrapper에 public $[0,1]$ rail과 Gaussian saturation 적용; LayerNorm의 unrestricted exponential difference는 유지 |
| Critical | spiking/mixed LayerNorm | `clip_margin` ratio만으로도 $10^7$–$10^8$ rail 생성 | module-output calibration과 clamp |
| High | scaled dot product | $\theta^2\sqrt D$ 뒤 score cap으로 큰 clipping 가능 | layer/head score calibration과 clamp rate 기록 |
| High | attention value integration | $S_{\max}\theta$가 sequence capacity와 함께 증가 | noise-free invariant를 참고해 공통 calibrated rail을 정하고 Gaussian saturation을 별도 검증 |
| High | affine, convolution, MLP projection | output별 parameter absolute-sum safety bound는 완료됐지만 activation distribution보다 넓을 수 있음 | module-output calibration |
| High | ViT/GPT-2 residual | interval width가 block depth와 함께 누적 | block별 post-add calibration과 clamp |
| Medium | dense LayerNorm, embeddings, task heads | finite하지만 model width 또는 parameter table extrema에 비해 넓음 | calibration으로 tighten하고 fixed metadata 전달 |

Constrained division은 $X\leq Y$를 public $[0,1]$ rail에 반영하고 Gaussian event-order inversion과 one-sided miss의 raw overflow를 clamp 전에 기록한다. Generic exponential-difference primitive에는 이 ordering을 적용하지 않으므로 LayerNorm의 두 positive rail은 1보다 큰 magnitude를 계속 표현할 수 있다.

## 전수 검색 결과

최초 source audit는 32곳을 확인했으며, attention adapter, 모든 shared LayerNorm 경로와 네 model adapter 전환 뒤 live activation extrema call site는 0곳이다.

| 구분 | live activation call site | 주요 원인 |
|---|---:|---|
| Shared operator | 0 | 모든 LayerNorm 경로가 analytic/operator range 사용 |
| ViT | 0 | preprocessing range, analytic activation range, calibrated encoder entry로 전환 완료 |
| BERT | 0 | frozen embedding-table interval과 upstream `Potential` 전달로 전환 완료 |
| RoBERTa | 0 | frozen table/affine interval과 internal `Potential` 전달로 전환 완료 |
| GPT-2 | 0 | frozen embedding/affine interval, analytic activation, calibrated residual boundary로 전환 완료 |
| 합계 | 0 | maintained forward가 activation extrema로 domain을 만들지 않음 |

### Transform operator

Maintained transform operator의 output range는 configuration, input range, endpoint transformation, reduction shape로 계산되며 activation extrema 위반이 없다.

| 위치 | 현재 range | 판정 |
|---|---|---|
| [[utils/transforms/potential_to_spike.py#neg_linear_transform]] | $[0,\text{window\_length}]$ | fixed |
| [[utils/transforms/potential_to_spike.py#neg_identity_transform]] | input potential span | fixed |
| [[utils/transforms/potential_to_spike.py#neg_log_transform]] | $[0,\tau_s\log(V_{\max}/V_{\min})]$ | fixed |
| [[utils/transforms/spike_to_potential.py#exp_operator]] | transformed time endpoints | fixed |
| [[utils/transforms/spike_to_potential.py#normalized_exp_operator]] | transformed time endpoints | fixed |
| [[utils/transforms/spike_to_potential.py#exponential_difference_operator]] | interval arithmetic과 exponential endpoints | fixed |
| [[utils/transforms/functions.py#multiplication_operator]] | potential/time endpoint products | fixed; tensor `theta` 지원은 제거하고 scalar $\theta$만 허용해야 함 |
| [[utils/transforms/functions.py#scaled_dot_product_function]] | product range와 head dimension | fixed |
| [[utils/transforms/functions.py#softmin_function]] | structural normalized-weight range $[0,1]$ | fixed; Gaussian excursion은 clamp 전 기록 |
| [[utils/transforms/functions.py#division_function]] | constrained public range $[0,1]$ | fixed; Gaussian overflow는 clamp 전 기록하고 unrestricted exponential difference는 별도 유지 |
| [[utils/transforms/functions.py#gelu_approximation]] | composed interval arithmetic | fixed |
| [[utils/transforms/functions.py#gelu_approximation_sigmoid]] | input range와 structural gate $[0,1]$의 product | fixed; Gaussian gate excursion은 clamp 전 기록 |
| [[utils/transforms/functions.py#tanh]] | structural activation range $[-1,1]$ | fixed; Gaussian excursion은 clamp 전 기록 |
| [[utils/transforms/functions.py#swiglu_function]] | input ranges와 structural gate $[0,1]$의 composed products | fixed; Gaussian gate excursion은 clamp 전 기록 |

`check_domain`의 tensor `min/max`는 declared range membership 검사이며 range 생성이 아니다. `functions.py`에서 scalar로 선언된 `theta`에 tensor가 들어오면 `theta.max()`를 사용하는 분기는 현재 model call에서 사용되지 않으며, static mismatch는 module input offset으로 이미 분리되어 있으므로 제거하는 편이 명확하다.

### Shared operator

Shared operator의 activation-derived call site는 모두 제거되었고 forward-time parameter extrema 계산만 남아 있다.

| 함수 | 수 | 현재 동작 | 교체 |
|---|---:|---|---|
| [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#_gaussian_forward]] | 0 | sampling 전에 ablation별 weight/bias/output domain을 freeze하고 동일 immutable output domain을 재사용함 | 완료 |
| [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#forward]] | 0 | Gaussian 경로와 동일한 ablation별 immutable weight/bias/output domain을 재사용함 | 완료 |
| [[utils/transformers/models/spiking_ops.py#_apply_norm]] | 0 | ordinary `nn.LayerNorm`에 $|z_i|\leq\sqrt{d-1}$와 $\gamma,\beta$ interval 적용 | 완료; calibration으로 더 좁힐 수 있음 |
| [[utils/transformers/models/spiking_ops.py#SpikingLinear#forward]] | 0 | deterministic/Gaussian output이 frozen absolute-sum rail을 공유하며 forward-time parameter scan 없음 | 완료 |
| [[utils/transformers/models/spiking_ops.py#SpikingConv2d#forward]] | 0 | grouped output-channel absolute-sum rail을 freeze하며 forward-time parameter scan 없음 | 완료 |

LayerNorm internal ranges $[m,\theta-m]$, $[m^2,(\theta-m)^2]$와 $T_0$는 configuration-derived다. Gaussian, deterministic, ordinary LayerNorm 경로는 모두 activation extrema 없이 analytic 또는 returned range를 보존한다.

### ViT

ViT의 activation-derived call site는 모두 제거되었다. Fully bounded activation은 analytic interval을 사용하고 encoder entry와 두 residual 경계는 frozen layer-wise calibration으로 depth별 range를 reset한다.

| 함수 | 수 | 현재 동작 | 교체 |
|---|---:|---|---|
| [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTPatchEmbeddings#forward]] | 0 | image processor metadata에서 channel normalization endpoint를 계산해 fixed Conv2d input range로 사용 | 완료 |
| [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTIntermediate#forward]] | 0 | ReLU/Tanh endpoint와 GELU/SiLU의 $[0,1]$ gate envelope를 fixed affine input range에서 계산 | 완료 |
| [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTEncoder#forward]] | 0 | collection은 fixed $[-\theta,\theta]$ safety rail, frozen phase는 persisted signed-symmetric entry range 사용 | 완료 |
| [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTLayer#forward]] | 0 | attention residual과 block output에 각각 frozen layer-wise range를 적용하고 strict excursion을 기록 | 완료 |

Image processor가 channel별 $x_c=(r_c-\mu_c)/\sigma_c$, $r_c\in[0,1]$을 사용하면 pixel range는 preprocessing metadata에서 직접 계산할 수 있다. Custom preprocessing 또는 `inputs_embeds`는 별도 calibration identity가 필요하다.

### BERT

BERT의 activation-derived call site는 모두 제거되었다. Embedding table range를 freeze해 합산하고, LayerNorm output range를 encoder와 first-token pooler까지 전달하며, intermediate activation은 fixed affine range에서 계산한다.

| 함수 | 수 | 현재 동작 | 교체 |
|---|---:|---|---|
| [[utils/transformers/models/spiking_bert/modeling_spiking_bert.py#BertEmbeddings#forward]] | 0 | 세 embedding table의 frozen interval을 합산하고 LayerNorm output `Potential`을 내부 encoder에 전달 | 완료; custom tensor는 word-table envelope 검증, explicit `Potential` 지원 |
| [[utils/transformers/models/spiking_bert/modeling_spiking_bert.py#BertIntermediate#forward]] | 0 | operator GELU range를 전달하고 dense GELU/ReLU는 fixed affine endpoint에서 analytic range 계산 | 완료 |
| [[utils/transformers/models/spiking_bert/modeling_spiking_bert.py#BertEncoder#forward]] | 0 | upstream `Potential` range 또는 fixed $[-\theta,\theta]$ standalone fallback을 사용하고 optional calibration entry를 지원 | 완료 |
| [[utils/transformers/models/spiking_bert/modeling_spiking_bert.py#BertPooler#forward]] | 0 | final encoder `Potential`을 first-token slice와 함께 전달 | 완료 |

ReLU branch는 현재 pre-activation range를 그대로 유지해 negative lower bound도 포함한다. 안전하지만 불필요하게 넓으므로 $[\max(0,l),\max(0,u)]$로 줄일 수 있다.

### RoBERTa

RoBERTa의 9개 activation-derived call site는 모두 제거되었다. Dense ablation은 functional PyTorch 값을 유지하면서 frozen affine interval을 사용하고, internal task wrapper는 final encoder `Potential`을 head까지 전달한다.

| 함수 | 수 | 현재 동작 | 교체 |
|---|---:|---|---|
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaEmbeddings#forward]] | 0 | frozen word, token-type, position table interval을 합산하고 normalized `Potential` 전달 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaSelfOutput#forward]] | 0 | dense/spiking projection이 같은 frozen affine interval 사용 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaIntermediate#forward]] | 0 | dense/spiking projection range와 GELU/ReLU analytic range 사용 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaOutput#forward]] | 0 | dense/spiking projection이 같은 frozen affine interval 사용 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaEncoder#forward]] | 0 | embedding `Potential` 또는 fixed standalone threshold rail 사용 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaPooler#forward]] | 0 | final encoder range를 first-token slice와 함께 전달 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaLMHead#forward]] | 0 | internal final `Potential`, frozen affine range, GELU range를 LayerNorm까지 전달 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaClassificationHead#forward]] | 0 | final encoder range를 first-token classifier path로 전달 | 완료 |

Public `RobertaModel` output은 Hugging Face tensor API를 유지한다. Local LM과 sequence-classification wrapper만 final `Potential`을 함께 요청해 operator-backed head에 전달하므로 외부 반환형과 내부 fixed range를 모두 보존한다.

### GPT-2

GPT-2의 최초 8개 activation-derived call site는 모두 제거되었다. MLP와 model entry는 parameter-derived interval을 사용하고 세 residual branch는 endpoint addition을 사용하며 maintained self-attention과 MLP residual에는 optional calibration reset을 연결했다.

| 함수 | 수 | 현재 동작 | 교체 |
|---|---:|---|---|
| [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Attention#forward]] | 0 | backend별 fixed range를 `c_proj`에 전달하고 projection/dropout range를 analytic propagation | 완료 |
| [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2MLP#forward]] | 0 | dense/spiking Conv1D frozen interval, analytic activation, dropout interval 사용 | 완료 |
| [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Block#forward]] | 0 | residual endpoint addition과 optional `attention_residual`/`output` frozen calibration 사용 | adapter 완료; evaluator lifecycle 연결 필요 |
| [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Model#forward]] | 0 | frozen token/position table interval과 optional model-entry calibration 사용 | adapter 완료; evaluator lifecycle 연결 필요 |
| [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#SpikingConv1D#forward]] | 0 | transposed output-column absolute-sum rail을 freeze하며 forward-time parameter scan 없음 | 완료 |

GPT-2 MLP activation은 dense/spiking projection 모두 `ACT2FN` 값을 유지한다. GELU-family와 SiLU는 $[\min(l,0),\max(u,0)]$, ReLU와 Tanh는 endpoint mapping을 사용한다. Cross-attention은 constructor에서 지원하지 않지만 남은 compatibility branch도 endpoint addition만 사용한다.

### Attention backend

Attention에는 live extrema 외에도 declared range와 실제 clamped output이 달라질 수 있는 두 문제가 있다.

| 위치 | 판정 | 필요한 변경 |
|---|---|---|
| [[utils/transformers/integrations/spiking_sdpa_attention.py#spiking_scaled_dot_product_attention]] | Q/K/V와 score range는 fixed; masked score도 declared upper score bound $c$ 사용 | 완료 |
| [[utils/transformers/integrations/spiking_sdpa_attention.py#_gaussian_attention_value_readout]] | [[utils/transformers/integrations/spiking_sdpa_attention.py#attention_output_bounds]]가 정한 $[-S_{\max}\theta,S_{\max}\theta]$를 사용하고 adapter가 memoized range를 재사용 | 완료 |
| [[utils/transformers/models/spiking_vit/modeling_spiking_vit.py#ViTSelfAttention#forward]] | spiking backend는 patch-grid $S_{\max}$의 memoized output range를 부착하고 eager backend는 `pot_v.domain` 유지 | 완료 |
| [[utils/transformers/models/spiking_bert/modeling_spiking_bert.py#BertSelfAttention#forward]] | `max_position_embeddings` 기반 memoized spiking output range 부착 | 완료 |
| [[utils/transformers/models/spiking_roberta/modeling_spiking_roberta.py#RobertaSelfAttention#forward]] | `max_position_embeddings` 기반 memoized spiking output range 부착 | 완료 |
| [[utils/transformers/models/spiking_gpt2/modeling_spiking_gpt2.py#GPT2Attention#forward]] | spiking/eager attention, projection, dropout range를 fixed/analytic propagation | 완료 |

Variable sequence length에서는 current $S$로 range를 바꾸면 같은 module의 range가 request마다 달라진다. BERT/RoBERTa evaluator의 `max_length`, ViT patch count, GPT-2 `max_position_embeddings`를 $S_{\max}$로 사용해야 한다. Cache-aware generation도 같은 $S_{\max}$를 유지해야 한다.

### Evaluation과 calibration

기존 네 evaluator의 quantile 수집은 진단용이며 fixed potential range calibration contract를 충족하지 않는다. ViT evaluator에는 이 경로와 분리된 layer-wise artifact lifecycle이 추가되었다.

| 위치 | 현재 동작 | 부족한 정보 |
|---|---|---|
| [[scripts/evaluation/error_analysis_vit.py#evaluate_vit_model]] | clean training subset을 두 번 순차 replay해 stable module/site별 immutable artifact를 저장하거나 frozen artifact를 검증·적용 | 완료; legacy absolute-quantile hook은 별도 진단 경로 |
| [[scripts/evaluation/error_analysis_bert.py#evaluate_bert_model]] | 같은 global maximum 방식 | dataset/task별 module range |
| [[scripts/evaluation/error_analysis_roberta.py#evaluate_roberta_model]] | 같은 global maximum 방식 | dataset/task별 module range |
| [[scripts/evaluation/error_analysis_gpt2.py#evaluate_gpt2_model]] | 같은 global maximum 방식 | sequence length, cache state, activation branch별 range |

Calibration은 Gaussian timing noise를 반드시 disable하고 `model.eval()`에서 수행해야 한다. ViT collection mode는 timing noise, mismatch, weight perturbation, bias perturbation, `DataParallel`을 거부한다. Frozen validation과 inference는 clean artifact를 검증한 뒤 robustness axis를 독립적으로 적용한다.

Calibration 측정은 deterministic한 두 collection pass로 분리하고 그 뒤 frozen validation을 수행한다. 첫 pass는 clamp 전 activation의 signed min/max를 기록하고, 두 번째 pass는 같은 dataset을 다시 실행하여 첫 pass의 endpoint로 고정한 bin에 histogram을 누적한다. Histogram과 margin으로 immutable range table을 만든 뒤 validation은 inference와 동일하게 excursion을 먼저 집계하고 clamp하며, 실행 중 range를 넓히지 않는다.

Pre-norm ViT와 GPT-2의 residual stream은 단순 interval addition을 block마다 반복하면 실제 activation보다 range가 계속 넓어진다. 이 위치는 post-add raw value를 block별 site로 측정하고 고정 range로 clamp해야 한다. BERT와 RoBERTa의 post-norm 출력은 LayerNorm의 fixed output range가 다음 block 경계를 다시 정하지만, calibration을 사용한다면 동일한 frozen-site 규칙을 따른다.

ViT의 [[scripts/evaluation/error_analysis_vit.py#apply_parameter_noise]]는 static weight/bias perturbation을 model loading 뒤에 적용한다. 따라서 affine parameter range는 `from_pretrained`, dtype/device conversion, static parameter perturbation이 끝난 뒤 고정해야 한다. Static threshold mismatch는 input potential을 shift하지만 encoder range $[-\theta,\theta]$는 유지하고 clipping 통계로 관측한다.

## 모든 실행 경우의 처리

Fixed range는 backend, noise, ablation, shape가 달라지는 모든 실행 경우에 대해 선택 규칙이 명시되어야 한다.

| 경우 | 처리 |
|---|---|
| Gaussian off/on 또는 seed 변경 | 같은 fixed potential range 사용; output range는 Gaussian miss를 포함하는 합집합 사용 |
| LayerNorm stage ablation | flag 조합별 calibration identity를 분리하거나 모든 조합을 포함하는 보수적 range 사용 |
| Spiking attention/eager attention | backend별 fixed range를 저장; 한 run에서는 선택한 backend의 range만 사용 |
| Evaluation dropout | `model.eval()`에서 identity이므로 input range 유지 |
| Training dropout | $1/(1-p)$ scaling과 zero를 포함한 range 사용; 지원하지 않으면 fixed-range inference에서 명시적으로 거부 |
| Variable sequence length | current length가 아니라 configured $S_{\max}$ 사용 |
| GPT-2 cache generation | `max_position_embeddings`까지 같은 range 유지; cache length로 range를 다시 만들지 않음 |
| Custom `inputs_embeds` | embedding-table 식을 사용할 수 없으므로 별도 calibrated input range 요구 |
| ViT custom preprocessing 또는 image size | preprocessing와 patch count별 calibration identity 분리 |
| float16, bfloat16, float32 | dtype별 calibration과 endpoint representability 검사; dtype 변경 시 range 재생성 |
| Static weight/bias perturbation | perturbation 뒤 parameter range와 derived affine range 재생성 |
| Static threshold mismatch | 같은 $\theta$ range를 유지하고 clipping 변화만 기록 |
| DataParallel | process-wide noise와 calibration state를 공유하지 않도록 거부하거나 replica별 immutable copy 사용 |
| Missing calibration entry | current tensor를 측정해 보완하지 말고 evaluation을 실패시킴 |

## Calibration 기록 형식

Calibration은 module별 signed lower/upper potential range를 보존하고, 같은 configuration에서 재사용할 수 있는 정보와 함께 저장해야 한다.

현재 공통 기반은 stable module/tensor identity로 record를 정렬하고 schema version과 전체 metadata를 strict JSON으로 저장한다. Load 시 unknown field, non-finite value, 중복 identity, metadata mismatch, record-derived range 변조를 거부하며 missing entry를 runtime tensor로 보완하지 않는다. 세부 불변식과 영구 검증은 [[calibration]]에 정의한다.

각 기록에는 최소한 다음 값이 필요하다.

- stable module name과 input/output 구분
- $V_{lb}$, $V_{ub}$, sample count
- signed extrema와 fixed-bin histogram
- 선택한 lower/upper quantile과 margin; validation 및 inference clipping rate는 immutable table과 분리된 run statistic으로 기록
- range policy와 optional analytic endpoint; symmetric signed range는 양쪽 tail을, one-sided range는 unbounded 방향만 calibration
- checkpoint와 model family
- dataset split과 preprocessing
- `theta`, `tau_s`, `tau_m`, `clip_margin`, dtype
- LayerNorm, attention, MLP ablation flags와 activation 이름
- sequence length 또는 image/patch shape의 fixed maximum
- static weight/bias perturbation과 static threshold mismatch 설정

Observed extrema를 쓰는 경우 calibration set 밖의 입력을 보장하지 못한다. Quantile을 쓰는 경우 clipping을 의도적으로 허용하므로 quantile과 inference clipping rate를 함께 보고해야 한다. 어떤 경우에도 inference output으로 $V_{lb},V_{ub}$를 갱신하면 안 된다.

## 권장 migration 순서

Dependency 순서대로 fixed range를 도입하면 각 단계에서 current tensor extrema를 하나의 원인과 함께 제거할 수 있다.

1. Immutable calibration entry와 histogram, min/max 및 fixed-bin collection accumulator, frozen lookup, pre-clamp excursion counter를 공통 기반으로 추가한다. Stable module binding과 collection/runtime `Potential` 경계까지 완료되었다.
2. 네 evaluator에 collection-only run, frozen validation, metadata validation, missing-entry failure를 추가한다. ViT는 완료되었고 나머지 세 evaluator가 남아 있다. Collection forward는 live extrema 대신 analytic safety rail을 전달해야 한다.
3. Pretrained parameter와 embedding table의 range를 checkpoint loading, dtype/device conversion, static perturbation 뒤 한 번 고정한다.
4. `SpikingLinear`, `SpikingConv2d`, `SpikingConv1D`가 forward에서 parameter extrema를 읽지 않고 frozen output-specific affine range를 사용하게 한다. 완료되었다.
5. `SpikingLayerNorm`이 $\psi_{\mathrm{ED}}$와 $f_{\mathrm{Mul}}$의 returned range를 전달하게 하고, dense branch에는 analytic 또는 calibrated pre-affine range를 사용한다. Analytic $|z_i|\leq\sqrt{d-1}$ 적용은 완료되었다.
6. Attention이 tensor와 fixed output range를 함께 전달하게 하고 mask suppression을 score upper bound와 일치시킨다.
7. ViT와 GPT-2의 pre-norm post-add residual을 block별 calibration range로 clamp하고, BERT/RoBERTa post-norm 경계에도 frozen site range를 연결한다. ViT의 두 block 경계와 evaluator lifecycle은 완료되었다.
8. ViT, BERT, RoBERTa, GPT-2 model entry를 embedding interval arithmetic 또는 calibration range로 교체한다. ViT encoder entry는 완료되었다.
9. Direct GELU, `gelu_new`, SiLU와 dense ablation path에 activation별 fixed range를 적용한다. ViT의 bounded activation 경로는 analytic range로 완료되었다.
10. Pooler와 task head가 final hidden-state range를 slice와 함께 전달하도록 한다.
11. Source audit와 batch-order, batch-size, Gaussian-seed invariance verification을 permanent verification에 추가한다.

## 검증 기준

Migration 완료는 numerical output뿐 아니라 declared potential range의 불변성으로 검증해야 한다.

필수 검증은 다음과 같다.

- 같은 sample을 다른 order와 batch size로 실행해 모든 module의 `PotentialBounds`가 동일함을 확인한다.
- 같은 input에서 Gaussian seed만 바꾸어 sampled time과 output은 달라도 `PotentialBounds`, `TimeBounds`가 동일함을 확인한다.
- Gaussian off/on 모두 같은 fixed range를 사용하고 miss와 saturation만 달라지는지 확인한다.
- Calibration range 밖의 raw output이 range를 넓히지 않고 underflow/overflow count를 증가시키는지 확인한다.
- Checkpoint, preprocessing, dtype, ablation, sequence maximum이 다르면 calibration load가 실패하는지 확인한다.
- `PotentialBounds(current.min(), current.max())`와 같은 source pattern이 maintained forward에 남지 않았는지 확인한다.
- Weight와 bias `.min()/.max()`가 maintained forward에 남지 않고 initialization/calibration 단계에만 존재하는지 확인한다.
- Masked attention에서 score가 declared range 안에 있고 BERT/RoBERTa/GPT-2 causal mask smoke test가 통과하는지 확인한다.
- `verify_gaussian_time_noise.py`의 operator parity, miss, saturation 검증과 `verify_sop.py`를 다시 실행한다.

## Manuscript와의 일치

논문의 fixed potential range, threshold, time window, clipping, per-layer calibration 표현은 migration 방향과 일치하지만 두 문장은 static configuration과 batch-derived range를 구분하도록 정리할 필요가 있다.

- “fixed potential range $(V_{lb},\theta)$ guarantees every encodable value fires within $[0,T]$”는 이 보고서의 contract와 일치한다.
- “$T$ is dynamically determined based on $\theta$”는 $\theta$가 inference 전에 고정된다는 뜻이면 위반이 아니다. 구현 설명에서는 $T=2\theta$ 또는 negative-log 식처럼 configuration-derived임을 명시해야 한다.
- LayerNorm의 “configured dynamically with $V_{lb}=0$”도 current batch가 아니라 operator configuration이라는 뜻으로 명확히 해야 한다.
- “matching $\theta$ to the actual activation range via per-layer calibration”은 model-entry와 activation calibration의 근거가 된다. Calibration split과 clipping rate를 함께 보고해야 한다.

## 최종 판정

Audit 자체는 완료되었지만 구현은 아직 fixed potential range contract를 만족하지 않는다.

Transform algebra, time window, attention, LayerNorm, affine, embedding, activation, residual의 maintained forward range는 모두 static하다. 남은 작업은 GPT-2 residual artifact lifecycle과 최종 source-audit 자동 검증을 evaluator에 연결하는 것이다.
