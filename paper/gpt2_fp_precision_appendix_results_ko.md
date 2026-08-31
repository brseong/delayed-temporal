# GPT-2 부동소수점 정밀도 통제 실험

이 노트는 GPT-2 단일 임계값 설정의 큰 성능 저하가 TTFS 연산 자체보다 float32 시간차 복원의 수치 정밀도에 주로 기인한다는 appendix 근거와 주장 범위를 정리한다.

## 결론

공유 임계값 $\theta=2000$에서 나타나는 **추가 attention 성능 저하**는 대부분 유한 정밀도 효과로 판단된다. 전체 변환과 dense 기준 사이의 남은 약 0.18--0.19 PPL까지 전부 부동소수점 오차라고 주장하면 안 된다.

- float32 공유 $\theta=2000$의 PPL은 25.0324이지만, attention 시간창만 50으로 줄이면 22.8913으로 회복된다.
- 이 두 점 사이에서 softmin 실행 score rail은 동일한 $[-40.242257,40.242257]$이고, Q/K/V와 attention 출력 등 비-score rail의 excursion은 모두 0이다.
- timestamp ULP는 $1.2207\times10^{-4}$에서 $3.8147\times10^{-6}$로 32배 작아진다.
- float64 공유 $\theta=2000$도 PPL 22.8928로 회복되어 작은 float32 시간창의 결과와 0.0015 PPL 이내에서 일치한다.
- dense HF 대비 대표 혼합 설정(global $\theta=2000$, attention $\theta=100$)의 증가는 0.1915 PPL, 즉 0.843%이다.

float32 $\theta=50$은 공유 $\theta=2000$의 excess NLL 중 91.7%, PPL gap 중 92.1%를 회복한다. 남은 loss 차이 0.0081과 PPL 차이 0.1837은 LayerNorm 등 변환 경로의 잔여 근사 효과를 포함하므로 별도로 남겨야 한다.

## 실험 프로토콜

모든 비교는 동일 코드와 캐시를 사용해 한 번에 재실행했다.

- checkpoint: `neulab/gpt2-finetuned-wikitext103`, cached revision `f042c5d9d998c564e49cddb98ddec90148e5aa43`
- dataset: WikiText-2 raw test, cached builder revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- 2,896개 nonempty text, batch size 16, 총 181 batch
- padded sequence length 128, padding label 제외
- loss: 기존 evaluator와 동일한 batch-mean causal LM loss, PPL $=\exp(\text{mean loss})$
- global $\theta=2000$, $\tau_s=1$, LayerNorm/attention/MLP temporal path 활성화
- calibration 없음, timing noise 없음
- PyTorch 2.11.0+cu128, Transformers 5.8.0.dev0, Datasets 4.8.5
- source commit before this experiment: `44ddb0bd4f84b72a4ac14a8017eac041242832c2`

재현 명령은 `scripts/experiments/precision_analysis_gpt2.sh`, 검증·집계 코드는 `scripts/analysis/summarize_gpt2_precision.py`이다. 원시 로그와 생성 표는 deny-by-default artifact인 `artifacts/precision_gpt2/` 아래에 둔다.

## 결과

| Condition | dtype | Attention $\theta$ | timestamp ULP | softmin score rail | Loss | PPL | $\Delta$PPL vs HF | score excursions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense HF | float32 | -- | -- | -- | 3.1227 | 22.7076 | 0 | -- |
| Local wrapper, temporal paths off | float32 | -- | -- | -- | 3.1227 | 22.7082 | +0.0006 | -- |
| Full conversion | float32 | 50 | $3.8147\times10^{-6}$ | $\pm40.2423$ | 3.1308 | 22.8913 | +0.1837 | 43,684,446 / 6,820,724,736 (0.640466%) |
| Full conversion, representative | float32 | 100 | $7.6294\times10^{-6}$ | $\pm40.2423$ | 3.1311 | 22.8991 | +0.1915 | 43,738,016 / 6,820,724,736 (0.641252%) |
| Full conversion | float32 | 200 | $1.5259\times10^{-5}$ | $\pm40.2423$ | 3.1316 | 22.9102 | +0.2026 | 43,484,854 / 6,820,724,736 (0.637540%) |
| Full conversion | float32 | 500 | $3.0518\times10^{-5}$ | $\pm40.2423$ | 3.1375 | 23.0466 | +0.3390 | 44,022,674 / 6,820,724,736 (0.645425%) |
| Full conversion | float32 | 1,000 | $6.1035\times10^{-5}$ | $\pm40.2423$ | 3.1535 | 23.4172 | +0.7096 | 44,688,789 / 6,820,724,736 (0.655191%) |
| Full conversion, shared threshold | float32 | 2,000 | $1.2207\times10^{-4}$ | $\pm40.2423$ | 3.2202 | 25.0324 | +2.3248 | 47,764,010 / 6,820,724,736 (0.700278%) |
| Full conversion, high-precision control | float64 | 2,000 | $2.2737\times10^{-13}$ | $\pm350.772$ | 3.1308 | 22.8928 | +0.1852 | 0 / 6,820,724,736 (0%) |

float32 sweep의 PPL은 attention $\theta$가 증가할 때 여섯 점 모두 단조 증가한다. 반면 score excursion rate는 $\theta=100$에서 0.641252%, $\theta=200$에서 0.637540%로 감소해도 PPL은 악화된다. 따라서 단순한 clamp 횟수 증가는 이 추세를 설명하지 못한다.

Score clamp 통계는 causal mask overwrite 전에 수집되므로 future position을 포함한다. 따라서 표의 절대 rate는 실제 unmasked clipping rate가 아니라 상한 진단치이며, 동일 mask와 denominator를 사용하는 sweep 내부 비교에만 사용한다.

각 sweep 점에서 query, key, value, normalized softmin weight, division result, attention value output을 합친 27,282,898,944개 값의 excursion은 0이다. score execution rail도 float32 여섯 점에서 완전히 동일하다. 이에 따라 sweep에서 체계적으로 변하는 핵심 수치 조건은 큰 timestamp에서 작은 차이를 복원할 때의 float32 ULP이다.

float64 조건은 시간차 정밀도뿐 아니라 softmin exponent의 표현 가능 rail도 $\pm350.772$로 넓힌다. 그러므로 이 점 하나만으로 순수 dtype 인과를 주장하지 않고, 고정 rail float32 sweep의 결론을 확인하는 보조 통제로 사용한다.

## Appendix 삽입용 영문 초안

### Floating-point precision control for GPT-2

To distinguish temporal quantization error from range clipping, we evaluated the fully converted GPT-2 while varying only the attention-local time window. All float32 settings from $\theta_{\mathrm{attn}}=50$ to 2,000 share the same representability-limited softmin score rail, $[-40.2423,40.2423]$. No query, key, value, normalized-weight, division-result, or attention-output excursion occurred in any condition. Nevertheless, perplexity increased monotonically from 22.8913 at $\theta_{\mathrm{attn}}=50$ to 25.0324 at $\theta_{\mathrm{attn}}=2{,}000$, as the float32 spacing at the timestamp endpoint increased by $32\times$. The pre-mask score-excursion diagnostic was not monotonic over the lower part of the sweep, ruling out its count as an explanation of the trend; because it includes positions later overwritten by the causal mask, its absolute rate is only an upper bound on effective unmasked clipping.

As an additional numerical reference, executing the shared-$\theta=2{,}000$ model in float64 reduced perplexity from 25.0324 to 22.8928. This control also widens the exponent-representable score rail and is therefore not treated as a pure dtype intervention; rather, it corroborates the fixed-rail float32 sweep. Relative to the simultaneously evaluated dense model (22.7076 PPL), the representative mixed-window float32 configuration ($\theta_{\mathrm{global}}=2{,}000$, $\theta_{\mathrm{attn}}=100$) incurs 0.1915 PPL, or 0.843%. These results indicate that the large additional degradation of the shared-window configuration is predominantly a finite-precision timestamp-subtraction effect, while a small residual conversion gap remains.

## 허용되는 본문 주장

권장 문구는 다음처럼 원인을 attention 추가 저하로 한정한다.

> The larger degradation observed when a single $\theta=2{,}000$ window is shared by all operators is predominantly a finite-precision artifact of timestamp subtraction. An attention-local window removes 92.1% of the PPL gap while preserving the same float32 softmin execution rail and incurring no non-score range excursions.

다음과 같은 절대적 문구는 결과가 지지하지 않는다.

> The entire conversion gap is caused by floating-point precision.

float64 및 작은 attention 창에서도 dense 대비 약 0.81%의 PPL 증가가 남기 때문이다. near-lossless는 대표 혼합 설정의 0.843% 상대 PPL 증가처럼 수치로 정의해 함께 제시하는 편이 안전하다.
