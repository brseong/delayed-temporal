# Deprecated 기록

NeurIPS 지식 통합에서 분리한 과거 리뷰·설계·구현 상태·실험 결과를 한곳에 보존한다. Deprecated는 현재 계약이 아니라는 표시이며, 모든 유도나 실험 결과가 틀렸다는 뜻은 아니다.

## 적용 범위와 현재 문서

Deprecated. 이 문서는 2026-09-14 최초 지식 통합본의 역사적 설명을 보관한다. 아래의 현재·미해결·완료라는 표현은 각 기록 당시를 뜻하며 지금의 작업 상태가 아니다.

현재 유효한 원칙은 [[neurips]], 동작은 [[domain]], [[operators]], [[noise]], [[calibration]], [[evaluation]]에서 확인한다. 과거의 추가 실험·삭제·수정 제안은 실행 지시가 아니며 [[todo]]나 [[deferred-experiments]]를 대체하지 않는다. 사용자 요청에 따라 한국어 리뷰와 용어 감사 문서만 남기고, 통합 완료한 나머지 원본 Markdown 11개는 삭제했다. 상세 원문은 아래 압축본에서 복구할 수 있다.

## 원본 복구

Deprecated. 삭제한 통합 원본 11개의 전체 내용은 압축본으로 보존한다. 지식그래프 요약에 없는 긴 유도나 원문 문안이 필요할 때만 복구하며, 현재 계약으로 다시 등록하지 않는다.

[삭제 전 원본 압축본](../artifacts/neurips-md-backup.tBDnUt/originals.tar.gz)에는 원래 상대 경로와 파일 내용이 들어 있다. 11개 파일을 각각 삭제 전 SHA-256과 대조했다. 압축본 SHA-256은 d0c89bdb950fb498f21893d4a5e614d5a9f27d188ac123597243372b9fc7dd89이다.

현재 남긴 파일은 [한국어 리뷰](../paper/neurips_2026/neurips_2026_review_ko.md)와 [용어 감사](../paper/neurips_2026/neurips_2026_terminology_audit_ko.md)다. JSON 용어사전의 규칙과 출처는 변경하지 않았다. 아래에서 원본이나 당시 문서를 언급하면 남긴 두 파일 또는 이 압축본을 뜻한다.

## 과거 리뷰

Deprecated. 세 리뷰에서 나온 주장·검증·재현성 문제를 연구 판단의 근거로 보존한다. 리뷰 평가는 당시 투고에 대한 의견이며, 현재 코드의 결함이나 완료 상태를 직접 증명하지 않는다.

### 원문과 번역의 관계

Deprecated. 보존한 [한국어 리뷰](../paper/neurips_2026/neurips_2026_review_ko.md)는 영문 리뷰의 번역이며 같은 Submission 21828의 세 리뷰다. 영문 원본은 압축본에 있고 독립적인 여섯 리뷰로 세지 않는다.

J1Ug와 R3u1은 평점 2, 확신도 4였고 SbkV는 평점 4, 확신도 3이었다. 평가 점수는 재실험 결과가 아니라 출처 식별을 위한 역사적 기록이다. 번역과 원문이 다르게 읽히면 압축본의 영문을 대조하고, 수학적 판정은 [[deprecated#과거 수학 검산]]의 검산 근거로 따로 확인한다.

### 세 리뷰의 논점

Deprecated. 리뷰마다 긍정한 기여와 요구한 증거가 다르다. 공통 비판을 묶더라도 각 리뷰어의 질문을 삭제하거나 같은 판정으로 처리하지 않는다.

- J1Ug: 연산자 합성, 인과 마스크와 LayerNorm의 제곱근 구성을 인정했다. 정량 오차 경계 부재, 부호 있는 곱셈, 활성함수 보정, softmax 범위, 큰 모델의 정확도 저하, 에너지 환산의 근거와 증명의 사람 검산을 문제 삼았다.
- R3u1: 여러 모델 평가와 전위·시간 변환의 통일된 표현을 인정했다. 기존 Transformer 변환 연구를 무시한 독창성 주장, 이산시간 방식의 불공정한 서술, 구성과 정리의 혼동, 비교 프로토콜 차이 및 이상화된 에너지 주장을 지적했다.
- SbkV: 합성 구성과 평가 범위를 긍정했다. 소자 수준 검증, 온도·가중치 변화, 더 큰 모델과 다른 입력 양식에서의 확장을 질문했다. 이것이 모든 확장 실험을 필수로 승인한다는 뜻은 아니다.

[[deprecated#과거 원고 검토]]은 선행연구와 주장 범위를, [[deprecated#과거 하드웨어 검토]]는 하드웨어 근거의 한계를, [[deprecated#과거 실험과 범위 감사]]는 후속 실험에서 구분해야 할 근거를 정리한다.

### 우선순위와 완료 판정

Deprecated. 원래 개선 체크리스트는 근거 문서이고, 유일한 상태 원장은 [[todo#Manuscript Revision Master Checklist]]다.

핵심 결론의 성립에 필요한 수학·비교·주장 수정이 우선이며, 실증 강화와 적용 범위 확장은 별도로 판단한다. 검산하지 않음, 검산 중, 판정 완료, 원고 반영 완료는 구분한다. 설명 부족으로 판정한 항목도 독자가 원고에서 조건을 복원할 수 없으면 원고 수정 대상이다.

권장 순서는 리뷰 전수 판정, 중심 주장과 구조 확정, 수학 검산, 결정론적 오차 분리, 비용 모델 정리, 필요한 공동 연구의 범위 합의다. 현재 실행 순서는 [[todo#Manuscript Revision Master Checklist#Recommended Execution Order]]를 따른다. 오래된 목록의 체크박스나 협업 제안을 새 실행 지시로 복원하지 않는다.

### 논문 공개의 근거 조건

Deprecated. 공개 전에는 주장 강도, 성립 조건, 실험 프로토콜과 수치의 출처가 일치해야 한다. 리뷰 답변을 읽지 않아도 원고 자체로 결과를 이해할 수 있어야 한다.

정확도 비교는 같은 체크포인트·데이터·평가 절차인지 표시한다. 결정론적 오차와 확률적 노이즈를 나누고, 긍정적 결과뿐 아니라 실패 조건도 남긴다. 에너지는 포함 비용과 제외 비용을 적으며, 증명 검산과 작성 보조 도구의 사용 범위도 사실대로 기록한다. 세부 완료 기준은 [[todo#Manuscript Revision Master Checklist#Final Release Gate]]를 참조한다.

## 과거 수학 검산

Deprecated. 기술 검산 노트의 1–7절과 14절을 연결한다. 앞부분의 당시 미해결 판정을 후속 수정 뒤의 현재 상태로 재사용하지 않는다.

### 구성 가능성과 정확도 보장의 구분

Deprecated. Transformer 연산을 조합할 수 있다는 설명은 전체 모델의 정량적 오차 경계와 다르다. 연산자별 오차와 깊이에 따른 증폭 상수가 없으면 무조건적인 정확성 정리로 해석하지 않는다.

원본 14절은 다음 층 오차가 이전 오차의 증폭과 해당 층의 추가 오차로 제한되는 재귀식을 정리한다. Residual의 항등 경로도 증폭 계수에 들어간다. Calibration으로 선언 범위를 고정해도 필요한 상수의 엄밀한 수치가 자동으로 얻어지지 않는다. [[decisions]], [[calibration]], [[deprecated#과거 리뷰]]와 함께 읽는다.

### 로그 적분 계수와 지수 부호

Deprecated. 원본 1–2절은 로그 인코더의 적분에서 빠진 시간 상수와 지수 연산 결론의 부호 오타를 수정한 기록이다. 후자는 합성 방법 자체가 틀렸다는 판정이 아니다.

정규화되지 않은 지수 전류의 적분에는 시냅스 시간 상수가 곱해진다. 입력 전류의 정규화 규약과 가중치 조건을 동시에 바꿔야 하며, 최종 로그의 인수는 무차원 비율이어야 한다. 고정 시각에서 음의 전위 인코딩을 지수 응답으로 읽으면 해당 구성의 지수 부호는 음수다. 현재 정의는 [[domain#TTFS Encoding]]과 [[domain#Temporal-to-Potential Decoding]]을 참조한다.

### 부호 있는 곱셈과 이벤트 도착 순서

Deprecated. 원본 3절의 대수적 검산과 14절의 후속 구현 검산은 서로 다른 증거다. 시간차의 부호를 계산하는 것과 실제 회로에서 전류를 공급·제거하는 것은 구분한다.

대수에서는 적분 구간의 순서와 입력 전류의 부호가 곱의 부호를 정한다. 물리적 구성에는 두 이벤트의 순서 판별, 적분 시작·종료와 양·음 전류 처리가 필요하다. 후속 검산은 두 이벤트가 각각 관측 종료까지 만드는 적분량의 차를 사용하고, 없는 이벤트의 기여를 초기값에 남기는 네 가지 전달 경우를 확인했다. 이는 회로 제작 검증이 아니다.

현재 계약은 [[operators#Primitive PWM Integration]]과 [[operators#Missing-Event Readout]]을 따른다. 가중치를 이미 포함하는 적분 전류에 같은 가중치를 다시 곱하지 않는 원칙은 [[deprecated#과거 전류 설계]]에 연결한다.

### 활성함수의 보정 위치

Deprecated. 원본 5절의 핵심은 지수 응답의 크기와 시간 상수가 sigmoid 내부에 영향을 준다는 점이다. 비선형 결과 뒤에 상수를 곱하는 것으로 내부 변화를 일반적으로 상쇄할 수 없다.

지수 응답의 크기는 분모에 상수를 더하기 전에 보정하고, 시간 상수의 영향은 지수 입력 또는 시간 변환의 기울기에서 맞춘다. 후속 14절은 활성함수 입력의 사전 배율과 디코더의 나눗셈이 맞는지 여러 시간 상수에서 검사한 기록이다. Softmin의 시간 상수는 별도의 온도 역할을 하므로 같은 주장으로 없애지 않는다.

실제 합성, GELU 고정 배율과 함수 종류는 [[operators#Composed Functions]]를 따른다. 이 보정은 물리적 지수 응답에서 무시한 빠른 감쇠항의 오차까지 제거한다는 뜻이 아니다.

### Softmax 범위와 공통 배율

Deprecated. 원본 6절은 양의 점수와 그 합이 모두 표현 가능한 범위에 들어야 한다는 조건을 다룬다. 상한만 낮추는 배율은 작은 점수를 하한 아래로 밀 수 있다.

공통 배율은 분자와 분모에서 상쇄되지만, 허용 구간의 존재는 점수의 최소·최대, 최대 길이와 양의 하한에 달린다. 입력에 따른 행 최댓값을 쓰는 식을 사전 고정된 배율처럼 설명하면 안 된다. 범위가 너무 넓으면 모든 점수를 동시에 보존할 수 없어 제한된 범위와 작은 값의 손실을 구분해 보고해야 한다.

후속 14절은 구현의 표현 가능 범위 제한을 원고의 물리적 공통 배율과 동일시하지 않는다. 현재 softmin·attention 동작은 [[operators#Spiking Attention]]을, 정밀도 통제의 근거는 [[deprecated#과거 실험과 범위 감사#GPT-2 정밀도 통제 실험]]을 참조한다.

### LayerNorm의 이상적 식과 유한 범위 구현

Deprecated. 원본 7절은 로그 인코딩의 절반 시간 상수와 제곱 기준값으로 표준편차를 구성하는 항등식과, 실제 구현의 하한·상한 처리를 분리한다.

초기 감사에는 수치 안정화 상수와 인코딩 하한의 혼용, 비활성 부호 경로의 작은 잔여값, 하한을 먼저 적용한 값의 제곱, 기준값 설명과 부분적인 일반 텐서 연산 문제가 있었다. 후속 14절은 두 하한의 분리, 실제 크기로 분산 계산, 비활성 경로의 출력 억제와 상수 입력 검사를 기록했다. 초기 문제 목록 전체를 현재 미수정으로 다시 적지 않는다.

현재도 표현 범위 밖의 값 제한, 아주 작은 양의 값 처리와 분산의 유한 양수 범위에 따른 근사는 구분해야 한다. 실행 단계별 우회 설정까지 공개해야 같은 LayerNorm 실험이라고 할 수 있다. [[operators#Spiking LayerNorm]]과 [[evaluation#Fixed-Domain Text-Model Real-Data Audit#Text-Model LayerNorm Execution Path]]가 현재 계약이다.

### 증명 간소화의 보존 조건

Deprecated. 증명 간소화 후보는 형식적 정리의 수를 줄이되 성립 조건과 실패 조건을 숨기지 않는 편집 원칙을 제안한다.

본문에는 로그 인코딩, LayerNorm의 비자명한 구성과 전체 블록의 구성 조건을 중심으로 남긴다. Attention, MLP, 나눗셈과 활성함수의 단순 치환은 합성 표나 짧은 설명으로 묶을 수 있다. 긴 유도는 원본과 부록에서 확인한다.

부호 있는 곱셈, softmax 배율, 활성함수 보정과 LayerNorm의 유한 범위 조건은 검산 전에 단순한 항등식이라는 이유로 삭제하면 안 된다. 원본의 형식적 결과 개수와 줄 번호는 당시 원고 기준이며, 현재 편집 완료 상태는 [[todo#Manuscript Revision Master Checklist#P0 Mathematical and Operator Audit]]에서 확인한다.

## 과거 전류 설계

Deprecated. 전류 진폭 검토의 핵심은 시간 파형, 시냅스 가중치, 보정과 실행 입력을 구분하는 것이다. 탐색적 기호를 현재 원고의 정의로 자동 채택하지 않는다.

### 고정된 연산자와 조절 가능한 크기

Deprecated. 고정된 연산자 집합이라는 주장은 모든 파라미터가 상수 하나라는 뜻이 아니다. 제한된 시간 파형을 재사용하는 것과 목표 함수마다 새로운 파형을 만드는 것을 구분한다.

원본 1–10절은 전류 정규화와 진폭을 명시하지 않으면 시간 상수, 노이즈와 에너지의 물리적 의미를 복원할 수 없다고 지적한다. 뉴런의 시간 상수·임계값, 변환을 맞추는 공통 보정, 사전 학습 가중치·편향, 실행 입력은 역할이 다르다. 보정이 칩·층·연산·뉴런 중 어디에 공유되는지도 조건이다.

원본의 여러 보정 기호는 설계 검토용이다. 현재 지식에서의 대응은 [[domain#Dual Operator Algebra]], [[domain#Scale Parameters]], [[deprecated#과거 수학 검산#활성함수의 보정 위치]]에 둔다.

### 스파이크와 벡터 전류의 구분

Deprecated. 원본 11–13절은 한 스파이크가 여러 수신 시냅스에 유발하는 전류를 벡터로 묶는다. 뉴런 자체가 벡터 값을 스파이크에 실어 전송한다는 뜻이 아니다.

하나의 스파이크 시각에 같은 시간 응답이 시작되고, 각 수신 시냅스의 가중치가 크기를 정한다. 여러 입력의 기여를 더하면 기존 가중치 행렬의 연산으로 연결된다. 전류의 벡터 표기는 기존 스칼라 식을 묶은 것이며 임의의 함수별 시간 응답을 새로 허용하지 않는다.

원본 12절은 명시적으로 탐색적 표기이고, 13절이 기존 원고 기호에 맞춘 후속 정리다. 추가 기호를 복제하지 않고 기존 전류, 시각, 가중치와 연산자 정의를 재사용한다. [[deprecated#과거 원고 검토#용어 감사와 표기 원칙]]과 함께 읽는다.

### 적분과 고정 배율

Deprecated. 적분에 공급되는 전류가 이미 시냅스 가중치를 포함하면 같은 가중치를 적분 밖에서 다시 곱하지 않는다. 고정 배율과 실행 시 두 입력을 곱하는 합성 연산은 구별한다.

시간에 대해 일정한 가중치 전류를 두 이벤트 사이에서 적분하면 그 가중치와 시간차의 곱이다. 입력의 부호와 도착 순서를 실제 회로로 구현하는 조건은 별도로 남는다. [[operators#Primitive PWM Integration]], [[operators#Spiking Linear and Convolution]], [[deprecated#과거 수학 검산#부호 있는 곱셈과 이벤트 도착 순서]]가 연결된 계약이다.

### 노이즈와 비용의 해석 한계

Deprecated. 독립적인 전류 변화와 공유된 크기 편향은 평균화 효과가 다르다. 같은 스파이크 수라도 전류 크기와 지속 시간이 다르면 실제 에너지가 같다고 결론낼 수 없다.

원본의 전류 기반 확률 모형은 후속 설계 제안이며 현재 Gaussian 시간 노이즈의 구현 정의가 아니다. 독립 변화는 pooling으로 줄어들 수 있지만 공유 편향은 남는다. [[deprecated#과거 노이즈 구현]]와 [[deprecated#과거 하드웨어 검토]]에서 이 구분을 유지한다.

SOP 환산은 스파이크 처리의 조건부 비용 추정이다. 전류 생성·유지, 누설, 전달, 보정과 복제 비용까지 포함한 시스템 비교를 대신하지 않는다. Stanojevic과의 비교도 ReLU 직접 구성, 연산 수와 실제 회로 비용을 나눠야 한다. [[evaluation#Symbolic Operation-Count Check]]는 내부 산술의 검증 범위를 정한다.

## 과거 시간창 설계

Deprecated. 파라미터 감사와 대화 정리는 범위와 관측 시간의 관계를 설명한다. 당시 제안과 현재 실행 정책은 구분한다.

### 범위와 관측 시각의 관계

Deprecated. 실제 입력이 사용하는 범위보다 인코딩 범위를 넓게 예약하면 마지막 실제 스파이크와 관측 종료 사이에 여유가 생긴다. 새로운 연산자를 추가해야 생기는 효과는 아니다.

로그 인코딩의 양의 하한을 낮추면 종료 시각이 늦어진다. 기존 범위 안의 같은 입력은 인코딩 시각이 그대로일 수 있으므로 그 입력의 늦은 도착을 더 허용한다. 하지만 새 하한 끝점까지 실제 입력으로 포함하면 그 끝점의 여유는 다시 0이다. 선형 인코딩에서도 실제 입력 최대 크기와 표현 가능한 크기의 차이가 같은 역할을 한다.

두 원본의 결론은 같은 실제 입력 범위에 대한 효과와 확장된 전체 범위의 최악 조건을 구별한다는 것이다. [[domain#TTFS Encoding]], [[domain#Finite-Window Semantics]], [[calibration]]에 연결한다. 원문의 부모 대화 전달문은 역사적 기록이며 새로운 실행 요청이 아니다.

### 마진이 바꾸는 것과 바꾸지 않는 것

Deprecated. 관측을 늦추면 늦은 스파이크를 받아들일 가능성은 커지지만, 주입한 Gaussian 시간 오차 자체의 표준편차를 줄인 것은 아니다.

전달된 스파이크만 고른 분포는 선택 효과로 달라질 수 있다. 따라서 원문의 표현을 모든 조건부 분산도 불변이라는 주장으로 확대하지 않는다. 입력 clipping 감소, deadline miss 감소, 전달된 시각의 오차와 긴 관측 시간의 물리적 비용은 별개의 효과다.

현재 [[noise#Fixed Observation Deadline]]은 명목 종료 시각을 기준으로 하며, 추가 마진 진단에서는 허용한 늦은 시각도 원래 인코딩 끝점에 제한한다. 이는 원문의 2026-08-16 정책 설명과 같다고 가정할 수 없다. 과거 계획으로 현재 실행을 해석하지 않는다.

### 공통 시간 이동의 조건

Deprecated. 두 이벤트의 시간차는 같은 양만큼 이동하면 보존되지만, 고정 시각에서 읽는 지수 응답은 일반적으로 달라진다. 상위 합성도 어떤 시간과 상수가 함께 움직이는지 확인해야 한다.

- 곱셈과 선형 적분: 입력과 기준 이벤트가 함께 이동하고 전류가 해당 시간 이동에 따라 바뀌지 않을 때 시간차가 보존된다.
- 나눗셈: 분자와 분모의 로그 인코딩 offset이 같아야 상쇄된다. 내부 재인코딩과 관측 시각까지 자동으로 불변인 것은 아니다.
- 지수 응답: 입력 시각만 이동하면 읽는 값의 크기가 바뀐다. 관측 시각도 같이 움직여 경과 시간이 보존될 때 구별해서 판단한다.
- Softmax와 softmin: 모든 분기에 같은 배율이 생기고 clipping·누락·수치 한계가 없을 때 정규화에서 상쇄될 수 있다.
- Sigmoid, tanh, GELU: 지수 분기에만 생긴 배율이 상수 항과의 합에서 남으므로 일반적인 불변성을 주장할 수 없다.

원본은 정확한 두 지수 응답과 늦은 시각의 단일 지수 근사도 구분한다. [[deprecated#과거 수학 검산#활성함수의 보정 위치]] 및 [[deprecated#과거 전류 설계]]가 관련 조건을 설명한다.

### 연산자와 파라미터 감사

Deprecated. 파라미터 감사의 1–9절은 당시 encoder·적분·지수·나눗셈·활성함수·모델 경로를 조사하고, 10–16절은 공통 시간 이동과 범위의 최종 해석을 정리한다.

영향을 받는 조건은 노이즈 평균·표준편차, 임계값, 시간 상수, 입력 범위, 최대 길이, 기준 이벤트 공유 범위와 연산 깊이다. 가중치 값이나 seed가 마진 때문에 직접 변경되는 것은 아니지만, 함께 바꾸면 원인 분리가 어려워진다.

원본에서 논의한 추가 관측 시간의 감쇠를 그대로 둘지 크기를 다시 보정할지는 서로 다른 실험 질문이다. LayerNorm 수치 안정화와 인코딩 하한도 분리해야 한다. 현재 그 분리의 구현 상태는 [[operators#Spiking LayerNorm]]에서 확인한다. 물리 감쇠·누설·에너지의 추가 실험은 이 문서 통합으로 승인되지 않으며 [[deferred-experiments]]를 따른다.

## 과거 노이즈 구현

Deprecated. 초기 노이즈 문제 목록, 주입 경로 감사와 기술 검산의 확률 모형 제안을 연결한다. 현재 실험 정의는 [[noise]]이며, 이력 속 계획이나 삭제 제안을 다시 실행하지 않는다.

### 초기 구현 문제와 후속 계약

Deprecated. 초기 문제 목록은 2026-08-13의 부분 구현과 남은 과제를 기록한다. 그 문서의 미해결 표시는 현재 구현 상태가 아니다.

핵심 요구는 encoder 이후 한 번의 시간 샘플링, 전달 여부의 명시적 보존, 평가 시작 시 한 번 초기화하는 난수 생성기, 시간 노이즈와 기존 노이즈 옵션의 구분이다. 기준 스파이크도 실제 이벤트로 취급하고 공유 범위를 정해야 한다. 정확도 외에 이벤트 수, 누락과 출력 범위 제한을 기록해야 원인을 나눌 수 있다.

당시의 표본별 기준 이벤트 제안과 현재 [[noise#Layer-Shared Reference Event]]의 호출 전체 공유 계약을 동일시하지 않는다. 난수 생성기와 통계가 프로세스 전역 상태이므로 단순 복제 병렬화의 재현성을 가정하지 않는다. 현재 설정과 검사는 [[noise#Configuration]]과 [[evaluation#Gaussian Spike-Time Verification]]에 연결한다.

### 주입 경로 감사의 분류

Deprecated. 주입 경로 감사는 sampler 직접 호출 여부만으로 노이즈 적용을 판정할 수 없음을 보여 준다. Encoder를 거쳐도 전달 상태를 잃으면 누락이 사라진다.

감사는 중복 Gaussian 전용 API, 시간 텐서만 받는 합성 연산, 모델의 미이관 경로, 직접 sampler 호출, 과거 실험 옵션, 의도적으로 남길 일반 연산 기준을 구분했다. 당시 선형·합성곱·attention·LayerNorm·지수·나눗셈 경로를 조사했으며, 독립 검증용 sampler 호출과 명시적 일반 모델 비교는 제거 대상으로 취급하지 않았다.

현재 계약은 [[noise#Encoder Injection Boundary]]에 있다. 검사 대상은 주입 횟수, 기준 이벤트의 생성 위치, 전달 상태의 소비, 노이즈를 끈 결과와 난수의 연속 진행이다. 과거 보고서의 함수 삭제 목록은 당시 제안이며 현재 존재 여부를 보장하지 않는다.

### 메모리와 전달 상태

Deprecated. 물리적 연결을 모두 거대한 중간 텐서로 만들 필요는 없다. 다만 메모리를 줄이는 등가 계산에서도 이벤트 누락과 출력의 관측 시각을 보존해야 한다.

초기 감사는 선형층의 배치·토큰·출력·입력 차원을 모두 펼치는 계산과 attention의 유사 중간 배열을 경고했다. 시간차에서 기여를 만든 뒤 행렬 연산이나 합성곱으로 합산하는 방법을 검토했으며, 합산 전 개별 시냅스의 제한과 합산 후 뉴런 출력 제한을 혼동하지 않도록 했다.

현재 대응은 [[operators#Spiking Linear and Convolution]], [[operators#Spiking Attention]], [[operators#Missing-Event Readout]]이다. 단지 유한한 최종 시각을 대입하는 것만으로 없는 이벤트를 표시하면 정상적인 마지막 스파이크와 구별되지 않는다.

### 확률 모형 교체의 범위

Deprecated. 기술 검산 노트의 8절과 10절은 전위 궤적으로부터 첫 스파이크를 샘플링하는 대안을 검토한다. 이는 현재 Gaussian 주입 분포의 단순 교체가 아니다.

인코더 입력 전위를 흔드는 실험, 출력 시각에 Gaussian 오차를 더하는 실험, 궤적 기반 첫 발화 모형은 서로 다른 가정이다. 교체하려면 인코더별 궤적과 단위, 관측 종료 전 발화하지 않는 상태, 후속 연산의 처리, 난수 재현성과 수치 비용을 함께 정해야 한다.

원본은 affine 경로의 제한된 pilot 구현·검증만 완료했다고 기록하며 production 경로는 유지했다고 명시한다. 전체 모델에 대안이 적용되었거나 기존 GELU·pooling 결과가 그대로 성립한다고 해석하지 않는다. 현재 [[noise#Direct Gaussian Spike-Time Noise]]는 하나의 시간 샘플로 도착 시각과 deadline miss를 함께 결정한다.

### 누락의 영향과 검증 범위

Deprecated. 기술 검산 9절의 취약도는 정적 구조 분석이지 전체 모델에서 확정한 경험적 순위가 아니다. 같은 누락 비율이라도 어디에 공유되는 이벤트인지에 따라 영향이 달라진다.

공유 분모, 기준 이벤트와 넓게 전달되는 attention·선형 입력은 여러 출력에 영향을 줄 수 있다. 긴 합성의 이벤트 수 증가도 누적 영향의 후보이며, 국소 함수의 기울기만으로 설명할 수 없다. 독립 노이즈의 평균화와 공유 오차는 [[deprecated#과거 하드웨어 검토]]에서 분리한다.

대안 검증에는 이론적 발화 분포·누락 확률과 샘플의 일치, 경계 조건, seed 재현, 유한한 출력, 연산별 전달 상태와 기존 결과의 회귀 검사가 필요하다. 추가 비교 실험은 [[deferred-experiments]]에 따라 별도 승인하며, 이 통합에서는 실험을 제출하지 않는다.

## 과거 하드웨어 검토

Deprecated. 기술 검산 노트의 12–13절은 BrainScaleS-2에서 검토한 기능과 필요한 근거를 구분한다. 문헌·설계 검토는 칩 실험 완료 기록이 아니다.

### 구성 요소와 전체 합성의 구분

Deprecated. 원본은 조절 가능한 뉴런 동역학과 입력 지속 시간에 따른 적분이 존재한다는 근거를, 모든 제안 연산의 자유로운 합성이 지원된다는 주장과 분리한다.

시간 상수·임계값·초기화·전류의 보정 범위는 유한하며, 임의의 미분방정식이나 임의의 발화 확률 법칙을 직접 설정할 수 있다고 가정하지 않는다. 고정 시냅스 가중치의 적분과 실행 중 다른 전위를 곱하는 연산도 다르다. 두 이벤트 사이의 부호 있는 적분, 아날로그 상태 전달과 외부 계산 없이 여러 동작을 연결하는 조건은 추가 검증 대상이다.

관련 물리적 해석은 [[deprecated#과거 전류 설계]], 수학적 구성과 회로 구현의 차이는 [[deprecated#과거 수학 검산#부호 있는 곱셈과 이벤트 도착 순서]]를 참조한다. 원문의 장비·API 설명은 당시 검토이며 새 실험 준비에는 현행 장비 규약을 다시 확인해야 한다.

### Pooling의 독립 오차와 공유 오차

Deprecated. 같은 논리 뉴런을 여러 물리 뉴런으로 구현하는 실험은 독립 변화의 감소와 공유 오차의 잔류를 구별해야 한다. 반복 수만 늘렸다고 독립성을 가정할 수 없다.

원본의 첫 목표는 동일한 입력과 동역학에서 첫 발화와 deadline miss를 반복 측정하는 제한된 검증이다. 뉴런별 보정 차이, trial별 변화, 공유 입력 경로와 배치 위치가 상관을 만든다. 같은 위치군과 분산 배치, 공유·분리 입력을 비교하는 제안은 이 상관을 확인하기 위한 것이며 현재 실행 승인으로 가져오지 않는다.

스파이크 수 감소가 아니라 복제를 사용하는 만큼 뉴런·시냅스·연결·이벤트 비용을 함께 계산해야 한다. [[deprecated#과거 전류 설계#노이즈와 비용의 해석 한계]]와 [[deferred-experiments]]에 연결한다.

### 작은 네트워크가 증명하는 범위

Deprecated. Attention 없는 작은 변환 네트워크의 하드웨어 결과는 여러 층에서 pooling 효과가 전달되는지 검증할 수 있다. 전체 Transformer의 칩 구현 증거는 아니다.

동일 체크포인트와 평가 분할에서 일반 모델, 이상적 변환 모델, 단일 하드웨어 구현과 복제한 하드웨어 구현을 구별한다. 출력 투표나 독립 모델 앙상블을 논리 뉴런 복제와 혼동하지 않는다. 적어도 내부 층에서 오차 전달을 관측하고, 보정과 선택에는 평가 정답을 사용하지 않는다.

여러 배치 위치와 반복에서 불확실성, 정확도와 자원 증가를 함께 보고해야 한다. 측정한 노이즈를 소프트웨어에서 재생한 결과를 전체 네트워크의 칩 실행으로 부르지 않는다. 요구 근거와 실제 완료 상태는 [[todo#Manuscript Revision Master Checklist#P0 Energy, Latency, and Hardware Claims]]에서 구분한다.

## 과거 실험과 범위 감사

Deprecated. 보존된 정밀도 결과와 기술 검산의 실측·분석을 출처별로 연결한다. 원본에 적힌 수치는 해당 코드·데이터의 결과이며 새 버전이나 다른 프로토콜로 합산하지 않는다.

### 과거 ViT Timing Noise Campaigns

Deprecated. Calibration 정책 2와 $\theta=20$을 적용하기 전의 threshold-40 및 uncalibrated timing-noise 실행은 현재 robustness 결과와 합치지 않는다.

Source `bc973317`의 초기 threshold 선택은 layer-wise calibration 없이 40을 선택했다. Source `648af9bb`의 `log-grid-9`와 `ratio-grid-13`, 그 이전 `v3`/`v5` 적응형 실행, 12 by 13 sigma-margin 계획, 48-site 65-condition 비교는 서로 다른 GELU·calibration·bound 계약을 사용한다. 완료 로그와 그림은 provenance로 보존하지만 현재 source `f7b74c1aef38502caccf532d1e58a7cf321833d6`의 109-site policy-2 결과를 보충하는 replica가 아니다.

`scripts/analysis/plot_iclr_timing_noise.py#main`은 당시 `log-grid-9`와 `ratio-grid-13`의 identity를 확인해 ICLR 부록용 2-panel 그림을 만들었다. 이 도구와 생성물은 source `648af9bb`의 역사 자료이며 현재 calibrated 그림 생성기로 승격하지 않는다.

현재 근거는 [[noise#Superseded Calibrated Threshold and Noise Sweeps]]의 training 5k 선택, fixed validation 5k의 두 one-dimensional sweep, 그리고 같은 identity의 high-scale timing-noise extension이다. 과거 $\theta=2000$, static threshold mismatch, 470-run joint grid와 자동 W&B 동기화 계획은 현재 표나 그림의 근거가 아니다.

### 과거 ViT Conversion Comparison

Deprecated. 2026-09-15의 48/96-site 비교와 64-image LayerNorm 진단은 policy-2 범위 전달과 timm preprocessing correction 이전 상태를 기록한다.

초기 v1 결과에서 CIFAR-10 ViT-S 91.56%, ImageNet ViT-S 60.14%처럼 낮은 clean SNN accuracy가 관측됐다. 당시 Q/K/V와 centered LayerNorm 입력이 calibration site가 아니었고 ImageNet에는 bilinear direct resize가 적용됐다. LayerNorm threshold만 2000으로 바꾼 64-image 진단은 원인 분해용이며 현재 full result가 아니다.

후속 policy 2는 ViT-S/B 109곳과 ViT-L 217곳을 수집하고 실제 attention·LayerNorm 연산에 전달한다. 현재 표의 ImageNet 행은 [[conversion-comparison#ImageNet Preprocessing Correction]]의 timm v4 결과이고, CIFAR 행은 완성된 v3 policy-2 결과다. Direct-resize ImageNet 값은 replica나 baseline으로 재사용하지 않는다.

### GPT-2 정밀도 통제 실험

Deprecated. 정밀도 통제 노트는 큰 공통 시간창에서 생긴 추가 attention 성능 저하가 주로 유한 정밀도와 관련됨을 보여 준다. 전체 변환 오차가 모두 사라졌다는 뜻은 아니다.

기록된 프로토콜은 neulab/gpt2-finetuned-wikitext103, WikiText-2 raw test의 비어 있지 않은 2,896개 텍스트, 배치 16, 길이 128과 181개 배치다. Padding 정답은 제외하고 배치별 평균 loss의 평균을 지수화한다. 토큰 수로 가중한 전체 loss와 혼용하지 않는다. Calibration과 timing noise는 끄고 공통 임계값은 2000으로 두었다.

| 기록된 조건 | PPL |
|---|---:|
| 같은 실험의 일반 모델 | 22.7076 |
| 로컬 모델의 시간 연산 비활성화 | 22.7082 |
| float32, attention 임계값 50 | 22.8913 |
| float32, attention 임계값 100 | 22.8991 |
| float32, attention 임계값 2000 | 25.0324 |
| float64, attention 임계값 2000 | 22.8928 |

Float32 여섯 점은 같은 점수 제한 구간을 사용했다. Attention 시간창을 줄이면 끝점 부근 표현 간격이 작아지고, 임계값 50에서 일반 모델과의 PPL 차이 92.1%를 회복했다. 점수 clipping 비율의 변화는 같은 방향으로 단조적이지 않았다.

Float64 비교는 정밀도뿐 아니라 지수의 표현 가능 범위도 바꾸므로 순수한 dtype 원인 검증으로 단독 사용하지 않는다. 작은 시간창과 float64에도 약 0.18–0.19 PPL 차이가 남는다. 원본의 점수 제한 통계는 인과 마스크 적용 전 값이므로 실제 전달되는 위치만의 비율이 아니다.

여섯 float32 점의 전체 수치·환경·캐시 revision은 원본에 보존한다. 현재 집계·검증 계약은 [[evaluation#Fixed-Domain Text-Model Real-Data Audit#GPT-2 Floating-Point Precision Control]]이며 원시 출력은 artifacts/precision_gpt2/에 대응한다.

### 교차 모델 평가의 비교 한계

Deprecated. 기술 검산 노트 14절은 BERT·RoBERTa·GPT-2의 대표 설정과 단계별 오차를 기록한다. 정밀도 통제 실험의 기준선과 그대로 합산하지 않는다.

기록상 BERT SST-2는 92.43%에서 92.20%, RoBERTa는 94.50%에서 94.04%였다. GPT-2의 초기 비교 기준은 PPL 22.4057, 로컬 기준은 22.7082, 전체 변환은 25.0324로 서로 다르다. 뒤의 정밀도 실험은 같은 재실행 안의 22.7076을 기준으로 삼는다.

당시 GPT-2에서는 attention의 영향이 가장 컸고 LayerNorm이 추가 오차를 만들었다. MLP의 affine 비교가 거의 중립적이어도 일반 텐서 경로의 gelu_new를 시간 GELU 검증으로 바꾸어 부를 수 없다. 전체 오차는 단계별 차이의 단순 합이 아니다. [[evaluation#Fixed-Domain Text-Model Real-Data Audit]]를 현재 실행 경로와 연결한다.

### 곱셈과 attention 대안의 조건부 비용

Deprecated. 기술 검산의 2026-09-11과 09-14 기록은 다른 곱셈 구성과 나눗셈 위치를 검토한다. 구현·정확도·물리 비용의 우월성을 확정한 결과가 아니다.

로그 시각의 차는 나눗셈을 만들므로 양·음 분리만으로 곱셈이 되지 않는다. 제곱의 차를 사용하는 대안도 제곱 자체의 구성 비용과 유한 범위를 고려해야 한다. Attention에서 value 합산 후 나누는 식은 이상적으로 같지만 중간 합과 분모가 함께 표현 가능해야 한다.

원문의 SOP 비교는 head별 value 차원보다 sequence length가 클 때 나눗셈 부분이 감소한다는 조건부 계산이다. 예시 ViT-B 크기에서는 나눗셈 부분 감소와 달리 비교한 attention 전체 절감률은 약 0.2292%였다. Projection 등 동일 비용과 추가 인코딩·부호 처리를 빠뜨리면 안 된다.

원본의 식과 수치 예시는 전체 head를 명시한 대안이며 기존 원고 비용식을 교체한 결과가 아니다. [[evaluation#Symbolic Operation-Count Check]]의 통과가 대안 회로의 실현 가능성을 검증하지 않는다. 검토만으로 새 실험을 [[todo]]에 추가하지 않는다.

### GELU 범위와 실행 소스의 구분

Deprecated. 기술 검산의 2026-09-14 GELU 계산은 고정 소스 648af9b와 calibration 표의 구간을 사용했다. 검증 데이터의 활성값을 새로 수집한 결과가 아니다.

해당 경로는 입력 구간과 0–1 gate의 관계를 쓰지 않는 일반 구간 곱으로 출력 bound를 전달했다. 시간 상수로 계산하는 세제곱 크기의 제한과 최종 곱셈에 들어가는 원래 입력 범위도 서로 다르다. 따라서 함수 자체의 더 좁은 출력 범위, 코드가 선언한 범위와 실제 관측값은 구분해야 한다.

후속 감사는 현재 작업 트리의 GELU 출력 제한 수정과 실행 중인 고정 소스를 분리한다. 새 정책이 예전 결과에 소급 적용되었다고 해석하지 않는다. 현재 정책과 calibration 무효화 조건은 [[operators#Composed Functions]], [[calibration]], [[bounds-audit#2026-09-14 Output Bound Audit]]을 따른다.

### 함수 출력 범위 전수 감사

Deprecated. 기술 검산의 마지막 감사는 값이 선언 범위 안에 있는지와 선언 범위가 불필요하게 넓은지를 나눠 조사한다. 범위가 좁아진다는 사실만으로 함수가 거리를 항상 줄인다고 주장하지 않는다.

직접 Swish의 하한, 내부 Swish와 최종 SwiGLU의 구분, 깨끗한 attention 평균의 범위, 지수 입력 구간 전체가 제한 밖일 때의 끝점 처리, 제곱과 LayerNorm affine의 보수성을 따로 분류했다. 일반적인 최종 SwiGLU에 내부 Swish의 하한을 그대로 적용할 수 없다.

노이즈가 있으면 attention weight 합이 1이라는 조건이 깨질 수 있으므로 깨끗한 평균의 범위를 공통 출력에 바로 적용하는 것은 동작 변경이다. 원본은 필수 경계 버그, 현재 경로의 개선과 사용하지 않는 경로의 선택적 정리를 구분하며 당시에는 추가 코드 수정을 수행하지 않았다고 기록한다. 현재 상태는 [[bounds-audit]]에서 확인한다.

## 과거 원고 검토

Deprecated. [용어 감사](../paper/neurips_2026/neurips_2026_terminology_audit_ko.md)와 후속 원고 2–3절 검토를 연결한다. 감사에 등장한 표현 자체가 승인된 논문 용어는 아니다.

### 용어 감사와 표기 원칙

Deprecated. 원본은 미정의, 늦은 정의, 의미 충돌과 구현 전용 표현을 구분했다. 약어·축·단위·측정 위치·분모가 독자에게 복원 가능해야 한다.

2026-09-07 감사의 21개 용어 문제, 6개 표기 문제와 13개 출력 표현 묶음은 당시 상태의 집계이지 현재 남은 문제 개수가 아니다. 같은 기호를 전위 스칼라, 활성 텐서와 attention value에 동시에 쓰지 않고 문맥과 모양을 구별한다. 기술 검산 11절의 표기 제안 역시 당시 보고서 범위였음을 보존한다.

사용하는 원고의 정의와 승인된 기호를 먼저 대조하고, 맞는 정의가 없으면 설명을 풀어 쓰거나 승인을 받아야 한다. 소스·CSV·로그에서 자주 쓰는 말이라고 그림·캡션에 자동 승격하지 않는다. [[evaluation#Manuscript Terminology and Notation Check]]는 편집 전 후보 검사와 편집 후 정확한 추가 diff 검사의 계약이다.

감사에서 발견한 임계값 선택 상태와 원고 실험 설정의 차이는 용어만 바꿔 해결할 수 없다. [[evaluation#Historical ViT-B/16 Global Range Selection]], [[deprecated#과거 실험과 범위 감사]]의 원본 근거와 함께 확인한다.

### 후속 원고의 선행연구와 예비 정의

Deprecated. 원본은 2026-09-07 검토와 09-08 재검토, 09-10 노이즈 비교를 포함한다. NeurIPS 디렉터리에 있어도 실제 검토 대상은 ICLR 2027 초안인 문서다.

핵심은 네 가지 연산자 형식, 물리적 동작의 해석, 텐서 수식 평가와 실제 칩 검증을 분리하는 것이다. 임의 함수별 파형을 정하는 방법과 고정된 연산자 형식을 재사용하는 방법을 비교할 수 있지만, 회로 복잡성이나 생물학적 우월성을 측정 없이 단정하지 않는다.

당시 검토는 TTFSFormer의 유한 시간 정밀도 평가를 인정하며, Wang 등의 함수별 근사와 시간상수 조절을 같은 방식으로 묘사하지 말 것을 지적했다. 새 문헌 검색 결과가 아니라 원문에 보존된 비교 판단이다. 외부 논문의 구체적 주장과 인용은 원본의 대조 자료에서 확인하고 출판 전 다시 검증한다.

### 모델과 뉴런 정의의 적용 범위

Deprecated. 평가 모델을 하나의 normalization 순서로 정의하거나, 서로 다른 뉴런 방정식의 해를 같은 것으로 쓰면 근거 범위를 벗어난다.

ViT·GPT-2의 normalization 위치와 BERT·RoBERTa의 위치를 구별한다. 실제 교체한 모듈과 남겨 둔 embedding·head·activation은 [[models]]에서 확인한다. 당시 GPT-2의 일반 gelu_new 경로를 모든 비선형의 시간 구현 근거로 삼지 않는다.

LIF의 두 지수 응답은 IF의 적분 해와 다르다. 초기조건, 전류 정규화, 서로 다른 시간 상수와 늦은 관측 시각의 근사 조건을 명시해야 한다. 시간 상수가 같을 때는 별도의 극한식이 필요하다. 텐서로 지수를 계산한 결과는 해당 회로 미분방정식을 시뮬레이션한 증거가 아니다.

지속적인 적분 창이 가능하다는 생리학적 근거만으로 두 이벤트의 순서 판별과 부호 있는 적분 전체가 검증되지는 않는다. [[deprecated#과거 수학 검산]]와 [[deprecated#과거 하드웨어 검토]]에 연결한다.

### 노이즈 선행연구와 기여의 경계

Deprecated. 원본은 TTFS에서 jitter, 삭제, 유한 시간창과 마진 자체가 처음인 것처럼 주장하지 말 것을 권고한다. 비교에서는 노이즈 원인과 관측 종료 이후 처리의 정의를 나눠야 한다.

독립적인 스파이크 삭제, Gaussian 시간 오차, 유한 정밀도, 고정 파라미터 변화와 입력 데이터 교란은 서로 다른 실험이다. 원문의 비교표는 Park, Stanojevic, Sakemi, TTFSFormer와 Otters 계열이 이미 다룬 범위를 각각 기록한다.

현재 연구에서 설명할 대상은 같은 시간 샘플이 도착 시각과 deadline miss를 결정하고, 이후 연산이 관측 시점의 전위를 받는 구성이다. 마진과 pooling의 효과도 분리해야 한다. 다만 원문의 비교 실험 제안 네 가지가 이번 통합으로 새 필수 작업이 되는 것은 아니며 [[deferred-experiments]]의 조건을 따른다.

### 편집과 출판의 경계

Deprecated. 오래된 검토의 줄 번호, 문헌 누락과 중복 인용 지적은 해당 초안의 기록이다. 원고 구조와 현재 상태를 확인하지 않고 기계적으로 수정하지 않는다.

Related Work는 변환·직접 학습·노이즈·하드웨어의 비교 질문을 분리하고, 예비 정의는 모델 종류·유한 시간창·뉴런 동역학·실제 평가 범위를 준비한다. 증명의 간소화는 [[deprecated#과거 수학 검산#증명 간소화의 보존 조건]]에 따른다. 본문 반영과 실험 필요성의 상태 관리는 [[todo#Manuscript Revision Master Checklist]]만 사용한다.

## 2026-09-15 Calibration 준수 초기 판정

Deprecated. 수집 절차의 일치만으로 전체 범위 선택도 일치한다고 넓게 읽힐 수 있었던 초기 판정을 보존한다. 현재 전수 판정은 [[vit-calibration-audit#ViT Calibration Coverage Audit#Protocol Compliance Findings]]이다.

초기 요약은 다음과 같았다.

> 현재 ICLR의 calibration 원칙과 수집 절차는 대체로 구현과 일치한다. LayerNorm 내부를 데이터로 수집하지 않는 사실, 고정 상한의 성능 문제, 원고의 설명 부족을 서로 구별한다.

초기 수집 절차 설명은 다음과 같았다.

> 방법론 87행의 절차에 따라 현재 비교 실험은 training seed-0 5k에서 최솟값·최댓값과 histogram을 수집한다. 관측 양 끝점의 최대 절댓값으로 대칭화한 뒤 전체 폭의 5%를 양쪽에 각각 더한다. 정확도를 보며 범위를 반복 최적화하지 않고 저장된 범위를 고정 적용하며 초과값을 clamp한다. 양의 log 하한은 별도 고정값이다.

재점검에서는 attention의 통계 구간을 theta로 다시 줄이는 실제 예외, Q/K/V domain 교체, LayerNorm의 eps 불일치와 magnitude 제한이 분산에 미치는 영향을 분리했다. Log 입력 범위를 고정할 수 있다는 해석은 유지하지만, 그것이 전체 구현의 프로토콜 준수를 증명하지는 않는다. 이 기록은 실험 변경을 승인하지 않는다.

LayerNorm의 magnitude 제한이 분산에 미치는 영향을 eps 불일치와 같은 구현 문제로 읽힐 수 있게 나열했던 분류도 정정한다. 사용자가 확인한 동일한 제한 전위의 제곱·log 구성에서는 의도된 clipping이다. 동작을 바꾸거나 독립 실험을 요구하지 않으며, 현재 범위 선택 목록은 [[vit-calibration-audit#ViT Calibration Coverage Audit#Bound Selection and Intended Clipping]]에 보존한다.

## 원본 문서 전수 대응

Deprecated. 최초 통합의 원본 13개 중 남긴 두 파일과 삭제·압축 보관한 11개를 구분한다. 파일 이름은 출처 식별자이며 삭제한 경로로 연결하지 않는다.

| 원본 | 상태 | 통합된 지식 |
|---|---|---|
| `neurips_2026_review.md` | 삭제·압축 보관 | [[deprecated#과거 리뷰#세 리뷰의 논점]] |
| [neurips_2026_review_ko.md](../paper/neurips_2026/neurips_2026_review_ko.md) | 유지 | [[deprecated#과거 리뷰#원문과 번역의 관계]] |
| `neurips_2026_review_checklist_ko.md` | 삭제·압축 보관 | [[deprecated#과거 리뷰#우선순위와 완료 판정]] |
| `reviewer_technical_verification_notes_ko.md` | 삭제·압축 보관 | [[deprecated#과거 수학 검산]], [[deprecated#과거 노이즈 구현]], [[deprecated#과거 하드웨어 검토]], [[deprecated#과거 실험과 범위 감사]] |
| `proof_simplification_candidates_ko.md` | 삭제·압축 보관 | [[deprecated#과거 수학 검산#증명 간소화의 보존 조건]] |
| `basis_current_amplitude_contract_ko.md` | 삭제·압축 보관 | [[deprecated#과거 전류 설계]] |
| `guard_operator_parameter_audit_ko.md` | 삭제·압축 보관 | [[deprecated#과거 시간창 설계#연산자와 파라미터 감사]] |
| `guard_bound_parent_handoff_ko.md` | 삭제·압축 보관 | [[deprecated#과거 시간창 설계#범위와 관측 시각의 관계]] |
| `noise_implementation_issues_ko.md` | 삭제·압축 보관 | [[deprecated#과거 노이즈 구현#초기 구현 문제와 후속 계약]] |
| `noise_decorator_bypass_audit_ko.md` | 삭제·압축 보관 | [[deprecated#과거 노이즈 구현#주입 경로 감사의 분류]] |
| `gpt2_fp_precision_appendix_results_ko.md` | 삭제·압축 보관 | [[deprecated#과거 실험과 범위 감사#GPT-2 정밀도 통제 실험]] |
| [neurips_2026_terminology_audit_ko.md](../paper/neurips_2026/neurips_2026_terminology_audit_ko.md) | 유지 | [[deprecated#과거 원고 검토#용어 감사와 표기 원칙]] |
| `iclr_sections_2_3_review_ko.md` | 삭제·압축 보관 | [[deprecated#과거 원고 검토#후속 원고의 선행연구와 예비 정의]] |

## 완료된 TODO 기록

Deprecated. [[todo]]에서 모든 항목이 완료된 작업 단위를 본문 그대로 옮겨 보존한다. 완료 당시의 기록이며 새 실행 지시가 아니다. 진행 중인 작업은 [[todo]]에만 남긴다.

각 기록의 상세 근거는 [[bounds-audit]], [[calibration]], [[evaluation]], [[conversion-comparison]], [[noise]]에 있다. 여기의 체크박스는 완료 상태의 사본이므로 다시 열지 않는다.

### GPT-2 Composed GELU Rerun

The user authorized a fresh calibrated WikiText-2 evaluation after replacing the direct GPT-2 block activation with the maintained composed GELU.

- [x] Connect `gelu_new` selected by the checkpoint to the composed operator path and distinguish it in calibration metadata.
- [x] Preserve the earlier direct-activation logs and assign the rerun a separate tag.
- [x] Collect fresh ranges from the fixed training 5,000 artifact.
- [x] Evaluate the dense reference and converted model on all 2,891 nonempty fixed test texts.
- [x] Validate and summarize the new artifact before changing the ICLR GPT-2 row or prose.

### Shared Power Cubic Text Rerun

The user authorized RoBERTa-B, RoBERTa-L, and GPT-2 evaluation with the same Power cubic used by the ViT rows, while explicitly reusing completed calibration artifacts.

- [x] Move the signed Power cubic into the canonical GELU implementation and keep repeated multiplication only as an analysis condition.
- [x] Add strict calibration-reuse evidence containing the original source, manifest, result, collection log, and calibration hashes.
- [x] Complete ANN and SNN evaluation for both RoBERTa checkpoints and GPT-2 without running a collection phase.
- [x] Validate the combined summary and remove the model-specific cubic distinction from the ICLR appendix.

### Imported Completed Foundations

These completed items are retained here so the migrated checklist does not lose the legacy status record.

- [x] Separate LayerNorm's denominator regularizer from the finite-window encoder floor.
- [x] Separate actual dual-rail magnitude from floor-clamped logarithm inputs and preserve inactive-rail no-spike semantics.
- [x] Compute LayerNorm variance from actual magnitudes rather than floor-clamped rails.
- [x] Publish the tensor versus spiking execution topology for the LayerNorm stages and model configurations.
- [x] Decompose the current GPT-2 degradation across attention, LayerNorm, MLP affine, residual, and wrapper paths.

### Static Bounds 구현 체크리스트

[[todo#Static Bounds for All Operators]]의 완료된 수용 기준, 후속 항목, 구현 체크리스트 80개를 보존한다. 해당 섹션의 설계 근거와 런타임 계약은 현행 문서이므로 [[todo]]에 그대로 남아 있다.

#### Acceptance Criteria

The migration is complete only when static-domain behavior is invariant under evaluation batching and all runtime extrema-derived domain construction has left maintained paths.

- [x] Reordering identical samples, changing batch size, or partitioning a batch produces identical declared bounds in representative shared operators, while the AST audit excludes activation-extrema construction across all maintained sites.
- [x] Changing the Gaussian seed changes sampled events and outputs but never changes any declared potential or time bound.
- [x] Every calibrated or Gaussian out-of-envelope value increments pre-clamp underflow or overflow statistics without mutating the envelope.
- [x] Evaluation fails clearly when a required calibrated bound is absent or incompatible instead of silently measuring the current tensor.
- [x] With frozen calibration enabled, selected ViT and GPT-2 residual boundaries use persisted ranges for each block instead of accumulating analytic interval sums; disabled calibration retains those sums.
- [x] A final AST source audit and direct tests reject `PotentialBounds` or `TimeBounds` constructed directly or through local aliases from live forward-tensor extrema.

#### Follow-up After Bound Re-Audit

The merged calibration work closes every known live-extrema violation; the remaining work concerns central validation and empirical evaluation rather than runtime calibration.

- [x] Rename the inclusive base interval to `ClosedBounds`; clamping, membership, and deadline equality all include both endpoints.
- [x] Enforce finite, ordered bound endpoints centrally and replace `check_domain` assertions with explicit exceptions that remain active under optimized Python.
- [x] Keep nonzero spiking attention training dropout outside the paper scope; the compatibility branch is documented, and maintained fixed-range claims apply only to evaluation with dropout disabled.
- [x] Run the ViT-S/ImageNet-1k real-checkpoint audit and report per-site clipping, Gaussian saturation, deadline misses, and task accuracy for LayerNorm, attention, affine, embedding, and the conventional task head; see [[evaluation#Fixed-Domain ViT-S Real-Data Audit]].
- [x] Repeat the real-data fixed-domain audit for BERT, RoBERTa, and GPT-2; the classifier gaps are small, and under the simultaneous current protocol GPT-2's attention-local threshold reduces the single-threshold PPL gap from 2.3248 to 0.1915; see [[evaluation#Fixed-Domain Text-Model Real-Data Audit]].

#### Implementation Checklist

The implementation work covers every maintained transform and model adapter, not only LayerNorm or operators that directly emit spikes.

- [x] Audit every maintained `PotentialBounds` and `TimeBounds` construction, including model inputs, embeddings, residuals, normalization, activations, attention, projections, and task readouts; the remaining violations are listed in [[bounds-audit#전수 검색 결과]].
- [x] Correct multiplication bounds to use the encoded operand's declared clamped endpoints instead of multiplying every ideal result by the full `theta` rail.
- [x] Restrict ordered division output to the noise-independent $[0,1]$ range; count and clamp Gaussian excursions without restricting the unrestricted exponential-difference primitive used by dual-rail LayerNorm.
- [x] Permanently verify division noise-mode domain identity, zero-noise output statistics, numerator-miss in-range behavior, denominator-miss overflow clamping, internal reset zero, and unrestricted exponential difference for LayerNorm.
- [x] Return softmin weights on the structural $[0,1]$ domain and count Gaussian excursions before the final rail clamp.
- [x] Permanently verify softmin noise-mode domain identity, zero-noise saturation counts, forced-miss excursion accounting, and final $[0,1]$ clamping.
- [x] Return tanh on the structural $[-1,1]$ domain and count Gaussian excursions before the final activation clamp.
- [x] Permanently verify tanh deterministic/zero-noise parity, the common $[-1,1]$ domain, forced excursion accounting, and final clamping.
- [x] Return sigmoid-GELU and Gaussian/deterministic SwiGLU gates on the structural $[0,1]$ domain before downstream multiplication.
- [x] Permanently verify sigmoid-GELU and SwiGLU gate-derived output domains, zero-noise counters, forced gate excursion accounting, and finite clamping.
- [x] Replace global-extrema-times-fan-in bounds in all three affine adapters with exact output-specific interval arithmetic before applying calibration.
- [x] Define `CalibrationMode` with distinct `collect`, `validate`, and `inference` phases so command-line and persisted representations use the same stable values.
- [x] Define the common layer-wise calibration data types: immutable ranges, histograms, layer records, run metadata, and calibration tables, plus mutable min-max observer, histogram observer, and clipping-count state with fixed fields.
- [x] Add a batch-order-independent min-max observer update that records finite signed extrema and tensor-element counts without retaining tensors or autograd graphs.
- [x] Select two deterministic calibration collection passes: signed min/max first, then fixed-bin histograms over the same dataset before frozen validation.
- [x] Construct each second-pass histogram from populated first-pass extrema with an explicit bin count and collection device, zeroed `int64` counters, and no arbitrary widening of constant ranges.
- [x] Accumulate batch-order-independent fixed-bin counts with inclusive outer endpoints, explicit underflow and overflow tails, constant-range handling, and no hidden device transfer.
- [x] Finalize a completed histogram only when bins and tails exactly match the total, copying device counters into an immutable JSON-compatible integer tuple without mutating the observer.
- [x] Select signed lower and upper quantiles from the immutable histogram with outward bin-edge rounding, rejecting cutoffs that fall inside unrecorded tails and leaving margin expansion as a separate policy.
- [x] Expand symmetric ranges on both calibrated sides, but expand one-sided ranges only toward the calibrated endpoint so a finite analytic endpoint never moves; leave zero-width ranges unchanged rather than inventing an absolute epsilon.
- [x] Persist policy-specific optional quantiles, analytic endpoints, and margin separately in each immutable layer calibration record so its final range can be reproduced and audited.
- [x] Build immutable layer records only from identical deterministic passes with zero replay tails, and count strict runtime excursions before autograd-preserving clamp.
- [x] Canonicalize calibration tables by stable layer identity, require exact metadata compatibility, and provide strict versioned JSON save, load, and setup-time lookup.
- [x] Permanently verify observer invariants, quantile and margin selection, frozen clipping, schema rejection, tamper detection, and deterministic persistence round trips.
- [x] Separate two-pass collection from frozen validation and inference with explicit state, one-way phase transitions, missing-site failure, and immutable clipping-report snapshots.
- [x] Declare calibration targets and endpoint policy per layer: signed-symmetric sites calibrate both sides, one-sided sites preserve their known endpoint, and practical structural bounds remain analytic; finiteness alone does not exclude a site.
- [x] Bind calibration state to stable model-module identities without checkpoint keys, use analytic safety bounds during collection, and return persisted clamp bounds as `PotentialBounds` during validation and inference.
- [x] Select a fixed-size prefix of a seeded training-split permutation for ViT calibration, replay the exact subset sequentially in both passes, and persist its split, seed, sample count, fingerprint, preprocessing, dtype, and model-path identity.
- [x] Add ViT collection, frozen-validation, and inference CLI modes with strict clean-collection constraints, exact metadata validation, missing-entry failure, and per-layer frozen clipping reports.
- [x] Replace ViT bounds from live activation extrema with preprocessing and analytic intervals plus two optional residual calibration boundaries per block; disabled calibration retains fixed residual interval sums.
- [x] Replace BERT intermediate GELU and ReLU live output extrema with ranges derived from the fixed affine input interval.
- [x] Propagate the fixed BERT encoder range through first-token pooling and use a configuration-derived standalone encoder fallback without live extrema.
- [x] Freeze BERT word, token-type, and position table ranges, sum their intervals before embedding LayerNorm, and preserve the resulting `Potential` through the internal encoder API.
- [x] Remove all RoBERTa live bounds by freezing embedding and affine ranges, propagating `Potential` through the encoder and pooler, and carrying the final range into local LM and classification heads without changing public model outputs.
- [x] Remove all GPT-2 live bounds with frozen embedding and Conv1D intervals, an analytic model-entry range, analytic MLP activation ranges, residual endpoint addition, and two-per-block calibration bindings.
- [x] Add GPT-2 collection, frozen-validation, and inference evaluator modes using filtered WikiText training subsets, fixed tokenizer/sequence metadata, sequential two-pass replay, and per-site clipping reports.
- [x] Use operator interval arithmetic where it provides a practical bound; retain calibration at selected boundaries whose finite ranges become excessively wide.
- [x] For paths without a practical tight analytic envelope, calibrate only ViT/GPT-2 pre-norm residual resets, ViT composed-GELU pre-activations, and spiking attention scores; analytic model entries bypass calibration.
- [x] Make maintained calibration retain observed min/max without tail truncation and add a 5% per-side margin; keep interior quantiles only as explicit diagnostic overrides.
- [x] Persist stable site identifiers together with the checkpoint, dataset split, preprocessing, model family, and active ablation configuration used for calibration.
- [x] Add explicit collection, frozen-validation, and inference modes so a site cannot measure and clamp against a range created by the same forward invocation.
- [x] Freeze learned-parameter and embedding-table bounds in versioned caches after checkpoint setup instead of recomputing parameter extrema on repeated forwards, including ordinary and spiking LayerNorm.
- [x] Add `SpikingLinear.freeze_parameter_bounds` with exact sign-aware rails per fixed input domain, immutable reuse, mutation rejection, and explicit refresh.
- [x] Allow `SpikingLinear._gaussian_forward` to use the frozen output bound for saturation accounting without rescanning parameters.
- [x] Connect `SpikingLinear.forward` so deterministic and Gaussian execution attach the same frozen output bound and deterministic execution performs no parameter extrema scan.
- [x] Remove the transitional `domain_W` argument and fallback from `SpikingLinear._gaussian_forward`, eliminating the remaining Gaussian weight scan.
- [x] Apply the same fixed-input-domain interval arithmetic, parameter mutation validation, and noise-independent metadata to grouped `SpikingConv2d`.
- [x] Apply the same fixed-input-domain interval arithmetic, parameter mutation validation, and noise-independent metadata to GPT-2 `SpikingConv1D`.
- [x] Make all three affine adapters consume upstream zero-containing fixed ranges, derive the zero-reference time from those ranges, and permanently verify asymmetric-domain parity and memoization.
- [x] Add `SpikingLayerNorm.freeze_parameter_bounds` for dense, direct exponential, and spiking exponential-difference envelopes with parameter/configuration mutation rejection.
- [x] Connect `SpikingLayerNorm._gaussian_forward` to frozen weight, bias, and final output domains before event sampling.
- [x] Connect deterministic `SpikingLayerNorm.forward` to the same frozen parameter and output contract.
- [x] Permanently verify all eight `SpikingLayerNorm` ablation domains, deterministic/zero-noise metadata identity, stale-cache rejection, and explicit refresh.
- [x] Initialize every model-family entry potential bound from frozen embedding/preprocessing intervals or explicit calibration rather than measuring the first or current batch.
- [x] Clamp every calibrated out-of-envelope value against its fixed bound and report underflow and overflow counts without widening that bound at runtime; Gaussian operator rails use their separate saturation counters.
- [x] Use the implemented LayerNorm normalization bounds that do not accumulate growth of input ranges across layers, then derive the output interval from scale and bias; a tighter output calibration remains a separate extension.
- [x] Support residual collection for each ViT/GPT-2 block and, when frozen calibration is enabled, count values outside the interval and clamp to the persisted interval.
- [x] Connect both ViT pre-norm residual boundaries to optional explicit calibration bindings while retaining batch-independent analytic interval addition when calibration is absent.
- [x] Support one measured symmetric score range per ViT/GPT-2 attention layer, subject to the analytic representability ceiling; without calibration, use the fixed score limit and retain separate Gaussian validation of value readout.
- [x] Replace the attention-specific `tau_m` and `tau_s` names with one `tau`; model adapters derive it from their shared `tau_s` configuration, and the ViT-only `tau_m` field is removed.
- [x] Support optional layer-wise calibration of affine inputs to composed GELU, retain an analytic final activation interval, and remove avoidable deterministic exponential overflow without changing the operator equation.
- [x] Keep spike-time windows configuration-derived: LayerNorm log windows remain fixed by `clip_margin`, `theta`, and `tau_s`, while affine identity encoding uses each declared zero-containing fixed interval.
- [x] Make declared potential and time bounds immutable so cached or propagated endpoints cannot be widened in place.
- [x] Keep masked attention scores inside the declared softmin range and clamp both Gaussian and noise-free value readouts to a rail derived from fixed $S_{\max}$ and $\theta$.
- [x] Attach that shared fixed attention-output range to `Potential` in the ViT, BERT, RoBERTa, and GPT-2 adapters instead of reusing the value range or measuring output extrema.
- [x] Remove live activation extrema from the Gaussian `SpikingLayerNorm` path by propagating operator intervals and using the finite-feature dense LayerNorm bound.
- [x] Remove live activation extrema from deterministic `SpikingLayerNorm.forward` with the same operator intervals and finite-feature dense bound.
- [x] Remove live output extrema from ordinary `nn.LayerNorm` calls in `_apply_norm` with the finite-feature bound and learned affine endpoint propagation.
- [x] Verify bounds are identical across batch contents, ordering, and batch size, and add a final source audit that rejects runtime tensor-extrema domain construction in maintained paths.

### Completed LayerNorm Upper Endpoint Change

사용자 승인에 따라 상한에서만 `clip_margin`을 빼던 정의를 코드에 수정했다. 양의 하한과 기존 실험 기록은 유지한다. 아래 비교는 변경 전후를 기록하며, 추가 실험이나 원고 변경은 수행하지 않는다.

#### Direct Changes

공유 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm]]의 입력 범위 정의를 아래처럼 바꿨다. $m$은 현재 `clip_margin` 값이며 기본값은 계속 $10^{-5}$다.

| 변경 대상 | 변경 전 코드 | 적용 코드 |
| --- | --- | --- |
| 실제 양/음 magnitude 범위 | `PotentialBounds(0.0, theta - clip_margin)` | `PotentialBounds(0.0, theta)` |
| log 계산용 범위 | `PotentialBounds(clip_margin, theta - clip_margin)` | `PotentialBounds(clip_margin, theta)` |
| margin 유효성 검사 | `margin >= theta / 2.0`이면 거부 | `margin >= theta`이면 거부 |

앞의 두 범위는 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#_gaussian_forward]]와 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#forward]]에 각각 있어 총 네 곳이다. 검사도 constructor와 [[utils/transformers/models/spiking_ops.py#SpikingLayerNorm#freeze_parameter_bounds]]의 두 곳에서 일치시킨다. Constructor의 변수명은 `normalized_margin`이며, 기존 finite/positive 검사는 유지한다.

문서 문자열, 주석, 오류 메시지의 "양 끝점을 안쪽으로 이동"과 `theta/2` 설명을 "양의 log 입력 하한" 및 `0 < clip_margin < theta`로 바꾼다. 호출부와 설정 파일의 `clip_margin` 이름은 이번 범위에서 바꾸지 않는다.

#### Derived Bounds

분산 범위와 시간창은 log 입력 구간의 끝점에서 이미 계산되므로, 별도 상수를 추가하거나 파생 수식을 중복 수정하지 않는다.

| 파생값 | 변경 전 | 적용 후 |
| --- | --- | --- |
| 분산 인코딩 범위 | $[m^2,(\theta - m)^2]$ | $[m^2,\theta^2]$ |
| magnitude 인코딩 시간창 길이 | $\tau_s\log((\theta - m)/m)$ | $\tau_s\log(\theta/m)$ |

기존 `domain_var = PotentialBounds(domain_err.min ** 2, domain_err.max ** 2)`와 `T0 = tau_s * math.log(domain_err.max / domain_err.min)`는 그대로 둔다. 분산 인코더의 시간상수 $\tau_s/2$도 유지하면 제곱된 구간으로부터 같은 시간창과 log 기준값이 나온다. 직접 로그를 계산하는 ablation 역시 `domain_err.max`와 그 제곱을 사용하므로 새 상한을 자동으로 따른다.

#### Unchanged Behavior

하한의 의미, signed 값의 처리와 LayerNorm의 최종 출력 제한은 이번 변경 대상이 아니다.

- `clip_margin=0`으로 바꾸지 않는다. 실제 magnitude에는 0을 허용하되 log 계산용 값에만 양의 하한을 적용한다.
- `positive_active`와 `negative_active`의 하한 판정, 비활성 경로의 출력 기여 제거, 분산 계산에 쓰는 0 magnitude를 유지한다.
- 분산에 더하는 `eps`, 시간상수, 가중치와 bias, normalized 및 최종 affine 출력 bound를 유지한다.
- [[utils/transforms/potential_to_spike.py#neg_log_transform]], [[utils/transforms/spike_to_potential.py#exponential_difference_operator]], 일반 곱셈의 정의는 바꾸지 않는다.
- GELU의 magnitude 하한, 전역 theta 선택 절차, calibration 수집 위치와 표본 선택은 바꾸지 않는다.
- 공통 LayerNorm 클래스 변경이므로 ViT만의 변경으로 설명하지 않는다. 이 클래스를 사용하는 다른 모델과 ablation에도 적용된다.

#### Verification Changes

기존 검사의 하드코딩된 내부 범위를 갱신하고, 새로 허용되는 상한을 실제로 밟는 경계 검사를 보강한다. 테스트가 기존 상한보다 작은 값만 사용하면 변경을 검증하지 못한다.

[[scripts/verification/verify_gaussian_time_noise.py#verify_gaussian_spiking_layernorm]]의 `PotentialBounds(0.0, 3.9)`를 `PotentialBounds(0.0, 4.0)`, `PotentialBounds(0.1, 3.9)`를 `PotentialBounds(0.1, 4.0)`로 바꾼다. 나머지 파생 variance와 deadline은 같은 계산식을 유지한다. Event 수와 miss 수 기대값을 새 결과에 맞춰 무작정 고치지 않는다.

- theta=4에서 평균이 0인 `[-4,-1,1,4]` 및 상한 초과 입력으로 magnitude/log 상한이 4인지 확인한다. 모든 원소가 같은 비율로 잘려 normalization에서 차이가 상쇄되는 입력만 사용하지 않는다.
- 0, 양의 하한 미만, 하한과 같은 입력에서 log 계산용 값과 비활성 경로를 구분한다. 상수 입력의 출력이 bias가 되는 기존 검사를 보존한다.
- Constructor와 bounds freeze가 theta=4에서 margin=2 또는 3을 허용하고, 0 이하, 4 이상, NaN/Inf는 거부하는지 확인한다.
- float32/float64, 세 LayerNorm ablation flag의 8개 조합, 노이즈를 끈 경로와 표준편차가 0인 Gaussian 경로의 일치를 검사한다. 노이즈가 있는 경로의 유한성, 고정 bounds, event/miss 처리는 별도로 검사한다.
- [[scripts/verification/verify_layernorm_affine_bounds.py#verify_paired_bounds_and_parity]]의 최종 출력 범위 기대값은 log 상한에서 나온 값이 아니므로 유지한다. [[scripts/verification/verify_layernorm_affine_bounds.py#verify_cache_and_single_feature]]의 캐시 검사를 보존하고 margin 변경 후 refresh 조건을 확인한다.

구현 후 위 두 검증 파일과 `verify_calibration.py`를 실행했다. 공통 연산자나 연산 수 정의는 바꾸지 않았으므로 새로운 연산 수 모델을 도입하지 않는다. 단위 검사만으로 모델 전체 성능을 주장하지 않으며, 후속 전체 결과는 [[conversion-comparison]]과 [[noise]]에서 따로 관리한다.

#### Artifact And Manuscript Changes

같은 theta와 clip_margin 숫자라도 상한의 의미가 달라지므로 구버전 calibration 표를 새 구현에서 조용히 재사용하지 않도록 한다.

ViT/GPT-2 metadata의 [[utils/transforms/functions.py#OUTPUT_BOUNDS_VERSION]]을 2에서 3으로 올렸다. 파일 구조를 바꾸는 것이 아니므로 calibration의 `format_version`은 유지한다. 구버전 metadata 거부와 새 버전의 저장/복원 검증은 [[calibration#Layer-wise Calibration#Frozen Execution#LayerNorm Positive Input Range]]에서 관리한다.

당시 별도 실행기는 source commit도 metadata에 넣었다. 기존 고정 source, calibration 표, 결과 파일은 수정하지 않았고, 후속 policy-2 및 timing-noise 캠페인은 별도 source와 결과 경로에서 calibration 표를 새로 수집했다.

새 정의를 적용한 결과만 현행 설명에 사용한다. 기존 source 648af9bb 결과는 이전 상한을 사용한 역사 기록으로 유지한다. 현재 정의는 [[domain#Signed Values and Dual Rails]], [[domain#Scale Parameters]], [[bounds-audit#Fixed Range의 수식 계약#Layer Normalization]] 및 calibration 문서와 일치시키며 과거 결과를 소급 수정하지 않는다.

#### Verification Already Performed

변경 가능성을 확인하기 위해 source를 고치지 않고 새 구간을 기본 연산자에 직접 전달한 작은 CPU 검사만 수행했다.

theta=40과 양의 하한 $10^{-5}$에서 float32/float64 모두 상한의 log 시각이 0이고, magnitude와 variance의 시간창이 일치하며, log 뒤 exponential difference가 기대한 나눗셈을 복원하고 제곱 연산이 상한을 처리하는 것을 확인했다. 이는 완성된 LayerNorm 클래스 변경, 모든 Gaussian 분기, 전체 모델 정확도 또는 기존 결과와의 동일성을 검증한 것이 아니다.

#### Implementation Verification

2026-09-14 코드 변경 직후 CPU 검증을 통과했다. 그 시점에는 전체 평가를 수행하지 않았지만, 이후 policy-2 비교와 calibrated timing-noise 캠페인이 새 정의로 완료됐다.

- [x] 실제 magnitude와 log 입력의 상한 네 곳 및 유효성 검사 두 곳을 수정했다. 파생 분산·시간창 수식과 variance의 시간상수는 유지했다.
- [x] [[scripts/verification/verify_layernorm_upper_endpoint.py#verify_upper_endpoint_and_ablations]]의 3개 검증 그룹을 통과했다. 8개 ablation, float32/float64, 시간상수 1과 0.75, 노이즈 유무 및 경계 입력을 포함한다.
- [x] float32 직접 log 계산이 고정 시간창을 반올림 오차만큼 넘는 경우를 발견해 두 직접 log 분기에서 계산 시각을 기존 시간창 안으로 제한했다. 시간창을 넓히거나 기본 연산자 정의를 바꾸지 않았다.
- [x] 기존 Gaussian 전체 검증, LayerNorm affine 4개 그룹, calibration 18개 그룹, GELU 4개 그룹 및 calibrated ViT evaluator 검증을 통과했다.
- [x] ViT/GPT-2의 version 3 표 저장·복원과 적용, version 2 표의 validation/inference 적용 거부를 확인했다. 변경된 규칙과 검증을 [[calibration#Layer-wise Calibration#Frozen Execution#LayerNorm Positive Input Range]]에 연결했다.

전체 모델의 정확도 변화는 이 검사로 주장하지 않는다. 이후 새 구현으로 평가할 때에만 별도 calibration 수집과 결과 경로를 사용한다.

### Calibrated Three Sweep Execution

2026-09-14 승인된 bound 실험은 source `f7b74c1aef38502caccf532d1e58a7cf321833d6`에서 완료되었고, threshold 선택과 두 noise 축의 최종 결과를 보존한다.

- [x] 이전 threshold-40 실행기와 해당 evaluator만 중단했다. 완료·부분 로그는 삭제하거나 새 결과와 합치지 않았다.
- [x] LayerNorm 상한 변경을 `c9f4e40`으로 별도 커밋하고 UBAI의 clean checkout에 동기화했다.
- [x] [[evaluation#Historical Calibrated Three Sweep Campaign]]에 71회 평가, 9회 calibration, training 선택과 validation 분리 및 경계 중단 규칙을 정의했다.
- [x] [[evaluation#Historical Calibrated Three Sweep Scheduling]]의 seed 0 전체 → seed 1 전체 → seed 2 전체 순서와 완료 결과 재사용을 구현했다.
- [x] 실행기와 집계기를 `36615ab`으로 별도 커밋하고 양쪽 clean checkout을 같은 commit으로 고정했다. 기존 사용자 문서 변경은 포함하지 않았다.
- [x] 새 계약 4그룹, 실행 순서 5그룹, 집계 4그룹, UBAI 안전성 19개 검증 및 관련 연산자·calibration·문서 검사를 통과했다.
- [x] Slurm 준비 작업 `984373`에서 자산·의존성 해시와 Python 3.12.13을 확인했다. 첫 준비 작업 `984371`의 경로 연결 실패 로그는 보존했다.
- [x] CPU 검증과 Slurm 자산 검증 후 양쪽의 짧은 clean/noisy prediction 일치를 확인했다.
- [x] 아홉 threshold의 training/validation과 반대 환경 replay를 검증해 $\theta=20$을 `confirmed` 상태로 확정했다.
- [x] 17조건씩 세 seed를 순서대로 완료하고 중간 snapshot과 최종 집계를 보존했다.
- [x] 같은 identity에서 timing-noise 상단 네 점을 추가해 $r_t=10^{-3}$까지 정확도 붕괴 구간을 확인했다.

완료된 본 캠페인은 50k validation과 추가 uncertainty axis를 포함하지 않는다. 로컬 GPU 0–3의 일시 허가는 종료되었으며 새 실험의 기본 허용 장치는 다시 GPU 4–7이다. UBAI runtime은 RAM disk가 아닌 `/enroot` 디스크를 사용한다.

UBAI의 새 clean checkout에는 읽기 전용 source를 mount하기 전에 내부 연결 지점인 `artifacts/assets/theta-selection-v1`, 해당 실험의 `artifacts/logs/noise_scan` 하위 디렉터리, `src/transformers`, `src/spikingjelly`를 빈 디렉터리로 준비해야 한다. 이 경로 준비와 실패·재시도 이력은 실험의 `deployment-notes.json`에 기록했다.

### ViT Calibration Policy 2 Shared Deadline

공통 시간창 전달은 구현과 CPU 회귀 검증 뒤 실제 ViT 정책-2 캠페인에서 사용됐다. 수정 전 실패 기록과 calibration은 역사 자료로만 보존한다.

- [x] LayerNorm의 세 log 인코딩에 동일하게 계산한 공통 시간창을 sampling 전에 적용했다. 이후 event의 domain만 바꾸는 처리는 하지 않는다.
- [x] 일반 primitive의 deadline 검사를 유지했다. 상한 40.007과 반대 방향 반올림 사례, 8개 ablation, noise-off·Gaussian 표준편차 0·seeded 경로의 회귀 검증을 추가했다.
- [x] 전역 threshold 초과와 선택된 범위의 실제 clipping을 구분하고, 새 source의 109-site ViT-B 수집과 평가로 실행 경로를 재검사했다.

### 2026-09-23 Global Range Campaign Retirement

The model-wide range parameter and its accuracy-based selection workflows were removed after local calibrated and analytic bounds became the complete execution contract.

Historical ViT threshold-selection, three-axis sweep, noise-scan, comparison-controller, GPT-2 precision-sweep, and UBAI deployment scripts were deleted from the maintained source tree. Their logs and artifacts remain immutable provenance. They cannot be resumed, aggregated with schema-2 results, or cited as current manuscript evidence.

The replacement contract rejects legacy `theta` and `attention_theta` configuration keys, uses calibration format version 2 and output bounds version 4, and applies timing-noise fractions per local encoder time window. The discrete-time simulation remains a separate retained experiment and is not part of the continuous-time rerun.

### 2026-08-31 Session Handoff

This historical handoff records the fixed-domain state and GPT-2 precision evidence as of 2026-08-31. Current complete-run metrics and calibration contracts supersede its pending-work language.

#### Established State

The maintained implementation now uses static bounds throughout, limits calibration to necessary sites, retains observed min/max with a 5% margin, and uses operator-local GPT-2 attention timing scale.

- Commit `44ddb0b` contains the merged text-model accuracy work: global GPT-2 $\theta=2000$, attention-local $\theta=100$, corresponding metadata identity, validation, and representative wrapper defaults.
- Generated artifacts are deny-by-default. Only the reviewed ViT-S min/max-plus-5% table is whitelisted; alternate GPT-2 calibration and precision logs remain local outputs.
- The simultaneous GPT-2 dense and mixed-window runs are 22.7076 and 22.8991 under the current batch-mean evaluator, a relative PPL increase of 0.843%; see [[evaluation#Fixed-Domain Text-Model Real-Data Audit]].
- The fixed-score-rail sweep and float64 reference isolate the large shared-window attention degradation as predominantly numerical; see [[evaluation#Fixed-Domain Text-Model Real-Data Audit#GPT-2 Floating-Point Precision Control]].

#### Delivery State

The reviewed precision-control tooling, appendix note, and knowledge-graph updates are delivered together on top of `44ddb0b`; generated result artifacts remain local.

- `main` contains `44ddb0b` plus the precision-control handoff commit. Their remote publication state must be checked explicitly before assuming they are pushed.
- The requested `/root/.codex/worktrees/a4c5/delayed-temporal` worktree is removed. Other detached and EBRAINS/toy worktrees remain registered and were outside this session's scope.
- The handoff commit adds GPT-2 dtype control, its float32-only calibration guard, verification, the precision sweep, the strict summarizer, and the updated evaluation graph.
- The precision results and protocol are consolidated in [[deprecated#과거 실험과 범위 감사#GPT-2 정밀도 통제 실험]]; the former appendix note is preserved in [[deprecated#원본 복구]]. `artifacts/precision_gpt2/` remains ignored and contains the local raw logs and generated tables.
- The calibration verifier passes all 18 groups, the float64 full-model smoke and held-out run complete, Python and shell syntax pass, `git diff --check` passes, and `lat check` passes. Ruff was unavailable in the active environment.
- Importing custom Transformer families emits pre-existing auto-docstring diagnostics labeled `[ERROR]` for unregistered custom configs and undocumented parameters even though verification exits successfully. This noise should be cleaned or filtered so it cannot hide a real failure.

#### Publication Risks

The evidence supports a limited finite-precision claim, but several protocol and manuscript discrepancies remain publication blockers until explicitly resolved.

- At this handoff, the GPT-2 metric was $\exp$ of an unweighted mean of 181 per-batch losses. The completed campaign now reports token-weighted corpus perplexity as primary and retains this aggregation only as a compatibility metric.
- The float64 $\theta=2000$ reference also widens the softmin execution score radius from 40.242257 to 350.772, so it corroborates but does not independently prove a pure dtype intervention. The fixed-radius float32 window sweep is the primary causal control.
- Attention-score clipping counts are recorded before causal-mask overwrite and include future positions. Their absolute rate is an upper-bound diagnostic; only like-for-like sweep comparisons are currently justified.
- `paper/neurips_2026/neurips_2026.tex` still reports the old GPT-2 row 22.40 to 23.43 ($+1.03$) and presents one GPT-2 threshold, while the current representative run is 22.7076 to 22.8991 with global/attention thresholds 2,000/100.
- Earlier GPT-2 reference values are retained in [[deprecated#과거 실험과 범위 감사#교차 모델 평가의 비교 한계]]; do not mix them with the simultaneous precision control.
- “The entire conversion gap is caused by floating-point precision” is unsupported. The safe claim is that the additional degradation from sharing $\theta=2000$ with attention is predominantly a float32 timestamp-subtraction effect; roughly 0.81--0.84% relative PPL remains.

#### Next Session

The next session should resolve publication consistency without silently expanding the experiment matrix. Optional reruns are catalogued in [[deferred-experiments]].

1. Historical action: preserve and disclose the batch-mean aggregation; the completed campaign now labels it only as a compatibility metric.
2. Reconcile the manuscript and reviewer notes with one canonical simultaneous protocol, including separate global and attention thresholds; manuscript rewriting remains intentionally deferred until authorized.
3. Decide whether the tracked appendix drafting note should be incorporated into the manuscript or retained as a separate internal record.
4. Push `44ddb0b` and the precision-control handoff commit after confirming the intended remote branch.
5. Do not use the present float64 reference as a pure dtype intervention; the optional control that holds the softmin score radius fixed is in [[deferred-experiments#Mechanism and Operator Ablations]].
