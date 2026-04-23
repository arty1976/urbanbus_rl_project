---
name: baseline_contract_and_experiment_A_link
description: "통합 baseline_contract.json을 생성하고, [A] 순수 MAPPO 실험의 experiment_A_contract.json과 연결합니다."
---

# baseline_contract_and_experiment_A_link

## 언제 이 skill을 써야 하는가
- `B0_historical`, `B1_noop`, `B2_rulebased` 기준선이 준비(일부 executed, 일부 prepared)된 상태에서 이를 하나로 묶어주는 통합 컨트랙트(baseline_contract.json)를 만들어야 할 때
- 이후 진행될 RL 학습 실험(여기서는 `[A] pure_mappo_baseline`)이 베이스라인과 어떤 제약(Fairness constraints)과 지표(Shared KPIs)를 공유하는지 명시하기 위해 experiment_A_contract.json을 만들 때

## 전제 조건
- B0 메타데이터가 생성되어 있음 (`completed` 상태)
- B1, B2 레이아웃이 준비됨 (`prepared_not_executed` 상태)

## 실행 순서

### 1. baseline_contract.json 생성
`artifacts/baseline_v1/baseline_contract.json` 파일을 작성합니다.
다음 내용이 포함되어야 합니다:
- **Baseline Status**:
  - `B0_historical`: `completed`
  - `B1_noop`: `prepared_not_executed`
  - `B2_rulebased`: `prepared_not_executed`
- **Shared KPIs (6종 고정)**:
  1. `cv_headway`
  2. `avg_wait_seconds`
  3. `bunching_rate`
  4. `on_time_rate`
  5. `intervention_rate`
  6. `energy_proxy`
- **Fairness Constraints**:
  - `same_initial_state`: `true`
  - `same_exogenous_events`: `true`
  - `same_eval_window`: `true`

### 2. experiment_A_contract.json 생성
위 Baseline과 연계된 실험 구조를 명시합니다.
`artifacts/experiment_A_v1/experiment_A_contract.json`을 작성하며, Baseline v1을 참조(Reference)하고 있음을 명시합니다.

### 3. PowerShell 기반 작성
직접 에디터를 열지 않고, `windows_powershell_patch_workflow` 원칙에 따라 JSON 문자열을 PowerShell Here-String(`@' ... '@`)으로 구성하고 `Set-Content`로 밀어넣거나, Python 스크립트를 작성하여 덤프합니다.

## 성공 판정
- `baseline_contract.json`과 `experiment_A_contract.json` 파일이 정상적인 JSON 구조로 생성되었음.
- 명시된 공유 KPI 및 공정성 제약 조건(Fairness constraints)이 빠짐없이 선언되었음.

## 실패 시 복구법 및 트러블슈팅 사례

### 1. JSON UTF-8 BOM 파싱 에러
- **증상**: PowerShell의 `Set-Content -Encoding UTF8`을 사용해 만든 JSON을 추후 파이썬의 `json.load()`로 읽을 때, 맨 앞의 BOM 문자(`\ufeff`) 때문에 `json.decoder.JSONDecodeError`가 발생함.
- **해결법**: 파이썬 스크립트에서 파일을 열 때 `encoding="utf-8-sig"`를 사용하여 BOM을 투명하게 제거하도록 수정합니다. (실제 수정 사례 반영)
  ```python
  with open("baseline_contract.json", "r", encoding="utf-8-sig") as f:
      data = json.load(f)
  ```

### 2. JSON 구조 불일치
- **증상**: 수작업으로 Here-String 내에 JSON을 짜다가 콤마(,) 누락 등 구문 에러 발생.
- **해결법**: 가급적 파이썬 딕셔너리를 작성하고 `json.dump(..., indent=4)`를 사용해 출력하는 스크립트 방식의 작성을 권장합니다.
