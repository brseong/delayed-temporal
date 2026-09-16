# NeurIPS에서 이어받은 지식

NeurIPS 검토에서 현재도 유효한 수학·해석·작성 원칙만 안내한다. 과거 리뷰, 버전별 결과, 변경 이력과 당시 구현 계획은 [[deprecated]] 한곳에서 확인한다.

## 현재 읽을 문서

현재 실행의 정의는 유지 문서가 우선이며, 아래 주제 문서는 그 정의를 해석하고 검증할 때 필요한 원칙을 설명한다.

- [[neurips-mathematics]]: 수학적 성립 조건과 검증 범위.
- [[neurips-current]]: 전류 파형, 가중치와 고정 배율.
- [[neurips-windows]]: 인코딩 범위, 마진과 시간 이동.
- [[neurips-hardware]]: 하드웨어 증거의 해석 한계.
- [[neurips-writing]]: 용어, 비교와 원고 작성 기준.

실제 동작은 [[domain]], [[operators]], [[noise]], [[calibration]], [[evaluation]], [[bounds-audit]]을 따른다. 상태는 [[todo#Manuscript Revision Master Checklist]]만 관리하고 선택적 실험은 [[deferred-experiments]]의 승격 조건을 따른다.

## 과거 기록을 확인할 때

옛 설정이나 결과를 찾을 때만 [[deprecated]]를 읽는다. 작성 시점의 검산·수정·실행 상태를 현재 코드에 그대로 적용하지 않는다.

원본 13개와의 전체 대응은 [[deprecated#원본 문서 전수 대응]]에 있다. 새로 대체된 설명도 deprecated 문서로 이동하고 현재 주제 문서에는 유효한 원칙과 현재 계약 링크만 남긴다.
