# ICLR 2027 제출 서식 점검

2026-09-26 공식 Author Guidelines·AI Policy와 공식 배포 템플릿을 현행 원고에 대조했다. 표 폭 초과는 이후 사용자 수정으로 해소됐고 필수 AI 사용 명시 누락은 남아 있어 전체 적합으로 판정하지 않았다. 표의 글자 크기는 과거 템플릿과 채택 논문을 대조해 별도로 재평가했다.

## 근거와 확인 범위

공식 웹페이지와 그 페이지가 직접 배포하는 파일을 기준으로 확인했다. 현재 로컬 원고를 점검했으며 OpenReview에 올라간 파일이나 별도 제출 자료의 상태는 확인하지 않았다.

- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines): 초기 제출 본문 9쪽, 이후 단계 10쪽, 참고문헌·부록·별도 명시문, 익명화 규정.
- [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors): 원고와 제출 양식에 AI 사용을 명시하는 의무.
- [공식 스타일 배포본](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip): 구체적 서식과 필수·권장 절. 같은 템플릿의 [공식 저장소](https://github.com/ICLR/Master-Template/blob/master/iclr2027/iclr2027_conference.tex)도 확인했다.

원고는 `paper/iclr_2027/iclr2027_conference.tex`와 실제로 포함되는 8개 본문·부록 파일, 부록의 생성된 Llama 표를 확인했다. 사용하지 않는 `iclr2027_conference_bak.tex`의 명시문은 제출 원고에 포함되지 않는다. 최신 점검 PDF는 `artifacts/runtime/iclr2027/iclr2027_conference.pdf`의 24쪽 배포본이며 SHA-256은 `52aeacda100a03c5b0b73b9317fb65922a306dd1f0f585b1bdaea8ee68f3e7f2`다. 원고는 수정하지 않았다.

## 필수 수정과 서식 복원

AI 사용 명시 누락은 현재 파일에서 직접 확인된다. 표 폭은 최신 PDF에서 다시 확인했고, 표의 작은 글씨는 명시적 허용 여부와 실제 사용 사례를 구분해 판단한다. 수동 간격 조정은 적용 범위와 출력 결과를 점검해야 한다.

### AI use statement 누락

공식 지침은 AI 사용 명시 절을 필수로 두며 분량 제한에서 제외한다. 현행 원고는 Conclusion 뒤에 곧바로 참고문헌을 출력하므로 이 절이 없다.

- 위치: [원고 조립부](/data/delayed-temporal/paper/iclr_2027/iclr2027_conference.tex:90). 배포 템플릿에는 참고문헌 앞에 번호 없는 절이 있고 길이는 최대 1쪽이다.
- 원고 문장 정리·수식 검토·실험 해석 등 이 대화에서 수행한 작업도 실제 사용 범위에 포함된다. 제출용 문구는 저자의 전체 사용 내역을 반영해야 한다. 저자가 모든 AI 보조 결과를 검증했다거나 특정 인원이 검토했다는 사실을 임의로 넣지 않는다.
- OpenReview 양식의 관련 응답도 논문의 설명과 맞춰야 한다. 이 양식은 이번 점검에서 열거나 수정하지 않았다.

### 부록 표의 본문 폭 재점검

공식 템플릿은 내용을 정해진 본문 사각형 안에 배치하도록 요구한다. 첫 점검에서는 두 표의 폭이 초과됐지만, 사용자 수정 뒤 06:09에 생성된 PDF와 컴파일 기록에서는 두 표 모두 본문 폭 안에 들어간다.

| 위치 | 수정 전 기록 | 최신 점검 |
|---|---|---|
| 17쪽 Table 7, LayerNorm 연산량 | 32.0307 pt 초과 | [현재 원고](/data/delayed-temporal/paper/iclr_2027/iclr2027_conference_appendix.tex:438)는 설명 열을 줄바꿈하며 표의 양쪽 끝이 본문 경계 안에 있다. |
| 22쪽 Table 10, Llama 결과 | 23.68684 pt 초과 | [포함된 표](/data/delayed-temporal/artifacts/llama/llama2-7b-hf/two-dataset-quick-20260925/appendix_llama_noise_table.tex:1)의 양쪽 끝이 본문 경계 안에 있다. |

수정 전 초과 폭은 최신 상태의 판정에 사용하지 않는다. 최신 컴파일 기록에는 `Overfull \hbox`가 없으며, PDF에서도 두 표의 텍스트가 본문 사각형 안에 있다. 표 글씨는 `\small`이고 크기 조정 상자로 감싸지 않았다.

### 표 글자 크기와 수동 간격 조정

공식 템플릿은 글꼴 크기와 서식 매개변수를 바꾸지 않도록 지시하며 참고문헌의 크기만 예외 가능성을 언급한다. 이 문구는 [2025](https://github.com/ICLR/Master-Template/blob/master/iclr2025/iclr2025_conference.tex)·[2026](https://github.com/ICLR/Master-Template/blob/master/iclr2026/iclr2026_conference.tex) 공식 템플릿에도 있다. 두 템플릿의 표 예시에는 `\small`이 없다.

- 공식 스타일에서 `\normalsize`는 10pt/11pt, `\small`은 9pt/10pt다. [본문 표 설정](/data/delayed-temporal/paper/iclr_2027/iclr2027_conference_experiment.tex:29), Framework의 두 표, 부록 표와 생성된 Llama 표에 `\small`이 있다. 다만 2025년 채택 논문 [AFlow](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5492ecbce4439401798dcd2c90be94cd-Abstract-Conference.html)의 [공개 LaTeX 소스](https://export.arxiv.org/src/2410.10762)는 `main.tex`에서 공식 스타일을 불러오고, `files/5-Experiment.tex`의 본문 표 두 개에 `\small`을 쓴다. 실제 사용 사례가 공식 예외 허가를 뜻하지는 않는다. 국소적인 표 글씨 변경은 스타일 파일 수정과 구분해야 하므로, 기존의 “모든 표를 10pt로 복원”을 필수 조치로 단정한 판정을 철회한다. 그림 안의 글씨에 대한 별도 최소 포인트 수는 확인하지 못했다.
- 추가 채택 사례로 [TabDiff 최종 논문](https://proceedings.iclr.cc/paper_files/paper/2025/file/5c882988ce5fac487974ee4f415b96a9-Paper-Conference.pdf)의 [공개 소스](https://export.arxiv.org/src/2410.20626)는 `iclr2025_conference.tex`에서 공식 스타일을 불러오고 `tables/mle.tex`의 본문 Table 3에 `\small`을 쓴다. [길이 편향 보정 논문](https://proceedings.iclr.cc/paper_files/paper/2025/file/5d50c76fdf75c24ece568fc84a7125fb-Paper-Conference.pdf)의 [공개 소스](https://export.arxiv.org/src/2409.17407)도 같은 스타일을 불러오고 본문 Table 2에 `\small`을 쓴다. 두 표의 제목과 내용은 각 공식 최종 논문에서 대조했다.
- [부록 설정](/data/delayed-temporal/paper/iclr_2027/iclr2027_conference_appendix.tex:103)은 `\textfloatsep`, `\intextsep`, `\abovecaptionskip`, `\belowcaptionskip`을 덮어쓴다. 이 설정은 뒤의 부록 전체에 적용된다. 공식 `.sty` 파일 자체는 같아도 본문에서의 매개변수 변경은 남아 있다.
- [Figure 2 배치](/data/delayed-temporal/paper/iclr_2027/iclr2027_conference_experiment.tex:105)의 `-9.5ex`와 `-2.5ex` 수동 여백도 확인했다. 음수 여백 명령이 무조건 금지된다는 별도 규정은 찾지 못했으나, 캡션 앞뒤 간격과 본문 경계를 실제 결과로 확인해야 한다. 8쪽에서는 캡션 분리나 본문 중첩은 보이지 않았다. 그림 폭 40%는 기존 사용자 요구다.

## 항목별 점검 결과

형식상 통과한 항목과 권장 사항을 구분한다. 일부 문제가 있다는 이유로 다른 항목의 적합 여부까지 부정하지 않는다.

| 항목 | 현재 원고와 PDF | 판정 |
|---|---|---|
| 초기 제출 본문 분량 | Conclusion은 9쪽, References도 9쪽에서 시작 | 9쪽 제한 충족 |
| 전체 PDF 분량 | 총 24쪽 | 참고문헌·부록 때문에 늘어난 분량은 문제 없음 |
| 참고문헌·부록 순서 | References 9–11쪽, Appendix A 12쪽부터 | 적합 |
| 공식 스타일·서지 파일 | `.sty`, `.bst`, `natbib.sty`, `fancyhdr.sty`, `math_commands.tex` 모두 공식 ZIP과 바이트 일치 | 원본 일치 |
| 용지 크기 | 612×792 PDF 포인트, US Letter | 적합 |
| 본문 영역 기본값 | 폭 5.5인치, 높이 9인치, 좌측 1.5인치 설정 유지 | 기본 설정 적합; 최신 PDF의 표 폭도 적합 |
| 본문 글꼴·행간 | Times 계열 대체 글꼴, 스타일 기본 10pt/11pt | 적합; 표의 국소적 `\small`은 명시적 예외가 없으나 확정 위반으로 판정하지 않음 |
| 제목·소제목·초록 | 기본 제목·절 매크로, 초록 한 문단과 기본 들여쓰기 | 적합 |
| 심사용 표시 | `\iclrfinalcopy` 비활성, 익명 저자 표시와 줄 번호 있음 | 적합 |
| 페이지 번호 | 전 페이지 표시 | 적합 |
| 익명성 | 표시된 저자·소속·감사 문구 없음; PDF Author 메타데이터 비어 있음 | 로컬 PDF 확인 범위 적합 |
| 표·그림 번호와 캡션 | Table 1–11, Figure 1–5 연속; 표 제목 위·그림 설명 아래 | 표시 형식 적합 |
| 인용·참고문헌 | `natbib`, 저자·연도 방식, 공식 `.bst`로 저자순 정렬 | 적합 |
| Ethics statement | 없음 | 권장 사항; 모든 논문의 필수 절은 아님 |
| Reproducibility statement | 없음 | 권장 사항; 재현 설명을 모은 절을 추가할 수 있음 |
| AI use statement | 활성 원고에 없음 | 필수 수정 |

## 추가 확인 사항과 판정 한계

서식 규정에 명시되지 않은 조건을 새 의무로 취급하지 않는다. 로컬 파일만으로 판단할 수 없는 제출 상태도 별도로 남긴다.

- 모든 PDF 글꼴은 내장되어 있다. 일부 그림에 Type 3 글꼴이 있으나 확인한 2027 공식 지침에는 Type 3 금지 조항이 없어 위반으로 판정하지 않았다.
- 그림은 색상을 사용해도 된다. Figure 3 등은 마커·선 모양도 함께 사용한다. Figure 2의 작은 눈금·범례는 인쇄 크기에서 가독성을 별도로 살피는 것이 좋다.
- 컴파일에는 표·그림 PDF 링크 목적지의 중복 경고가 있다. 보이는 번호는 연속이고 참조가 미정의된 경고는 없지만, 최종 제출본에서는 링크가 올바른 곳으로 이동하는지 확인해야 한다. 공식 서식 위반이라고 단정하지 않았다.
- 공식 Author Guidelines 하단의 일부 FAQ는 초기 제출과 이후 단계가 모두 10쪽인 듯한 문구를 남겨 두었다. 상단 Paper formatting과 최신 배포 템플릿은 초기 9쪽·이후 10쪽으로 일치하므로 초기 제출은 9쪽을 적용한다.
- 코드 ZIP, 외부 익명 저장소, 제출 양식, 저자 등록·중복 제출·심사 의무 등 계정·제출 상태는 이번 로컬 서식 점검의 확인 대상이 아니다. 따라서 전체 제출 적격성을 보증하는 결과로 해석하지 않는다.

## 후속 작업

필수 명시문을 정리한 뒤 수동 간격의 출력 결과와 본문 9쪽 제한을 다시 확인해야 한다. 이전에 폭을 넘던 두 표는 현재 해소됐다. 기존 표의 `\small`을 일괄 제거할 필요가 있다고 단정하지 않는다. 기존 문단 구조와 과학적 내용의 변경은 별도로 판단한다.

구체적인 작업은 [[todo#Manuscript Revision Master Checklist#ICLR Submission Requirements Audit]]에서 관리한다. 과학적 서술의 일관성은 [[iclr-appendix-consistency]], 부록 구성은 [[iclr-detail-review#부록 구조 정리]]를 참조한다.
