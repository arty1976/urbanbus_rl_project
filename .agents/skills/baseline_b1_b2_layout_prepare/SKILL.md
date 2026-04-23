---
name: baseline_b1_b2_layout_prepare
description: "B1 No-op baseline 및 B2 rule-based baseline의 실행 전 계약 구조(디렉터리, 매니페스트, 정책 설정)를 구성합니다."
---

# baseline_b1_b2_layout_prepare

## 언제 이 skill을 써야 하는가
- RL 환경의 Baseline 중 시뮬레이션 기반인 `[B1] No-op` 및 `[B2] Rule-based` 평가 레이아웃을 구성해야 할 때
- 시뮬레이터 어댑터(Simulator Adapter)가 아직 준비되지 않은 단계에서, "실행 전 준비 완료(prepared_not_executed)" 상태의 메타데이터와 디렉터리 뼈대만 우선 확보하고 싶을 때

## 전제 조건
- `B0_historical`이 완료된 상태이거나, 적어도 전체 베이스라인 평가 프레임워크 v1 디렉터리가 생성 가능한 상황
- 시뮬레이터 구동은 생략하므로 특별한 패키지 의존성은 없으나, 디렉터리와 JSON/Parquet 파일을 생성할 Python 스크립트를 작성해야 함

## 입력
- B1 및 B2 기준의 룰 기반 파라미터 (특히 B2 파라미터 고정값)

## B2 Rule Parameters 고정값
- `target_headway_seconds`: 600
- `low_headway_threshold_seconds`: 360
- `high_headway_threshold_seconds`: 900
- `max_hold_seconds`: 120
- `allow_skip`: true

## 실행 순서

### 1. 디렉터리 및 뼈대 파일 생성 Python 스크립트 작성
다음과 같은 폴더 구조를 생성하는 Python 스크립트를 PowerShell의 Patch Workflow(Here-String) 방식으로 생성하여 실행합니다.

**목표 구조 (B1):**
- `artifacts/baseline_v1/B1_noop/`
  - `scenario_index.parquet` (빈 파일 또는 스텁)
  - `policy_config.json`
  - `rollouts/seed_001/`
  - `rollouts/seed_002/`
  - `rollouts/seed_003/`

**목표 구조 (B2):**
- `artifacts/baseline_v1/B2_rulebased/`
  - `scenario_index.parquet` (빈 파일 또는 스텁)
  - `policy_config.json`
  - `rollouts/seed_001/`
  - `rollouts/seed_002/`
  - `rollouts/seed_003/`

### 2. 정책 설정 및 매니페스트 생성
`policy_config.json` 파일 내에 각 Baseline의 목적과 룰 파라미터(B2의 경우)를 기입합니다.
또한 롤아웃 디렉터리 내부에는 상태가 명시된 매니페스트를 남깁니다.

- **원칙**: 시뮬레이터 부재 시, 실제 롤아웃은 돌리지 않고 상태가 `"prepared_not_executed"`임을 명시하는 JSON 매니페스트만 남깁니다.

### 3. 스크립트 실행
PowerShell에서 생성한 Python 스크립트를 실행하여 레이아웃 생성을 완료합니다.

## 성공 판정
지정된 B1_noop, B2_rulebased 디렉터리에 `scenario_index.parquet`, `policy_config.json` 파일이 존재하고, `rollouts/seed_001`부터 `seed_003`까지 디렉터리가 생성되어 있으면 성공입니다.
특히 B2의 `policy_config.json`에 지정된 파라미터 5종이 정확히 들어가 있는지 확인해야 합니다.

## 실패 시 복구법 및 트러블슈팅 사례

### 1. 누락된 부모 디렉터리 에러
- **증상**: 중첩된 롤아웃 디렉터리(`rollouts/seed_001` 등)를 생성하려 할 때 에러 발생.
- **해결법**: Python 로직에서 `os.makedirs(path, exist_ok=True)` 사용을 강제하여 경로 전체 체인이 생성되도록 보장합니다.

### 2. 시뮬레이터 미연결 오류 방어
- **증상**: 롤아웃 코드를 무리하게 구현하다가 SUMO/CityEngine 등의 어댑터가 없어 무한 대기 혹은 충돌 발생.
- **해결법**: 이 단계에서는 명시적으로 "시뮬레이터를 띄우지 않고 뼈대만 만든다"는 원칙을 지키며 상태를 `prepared_not_executed`로 고정합니다.
