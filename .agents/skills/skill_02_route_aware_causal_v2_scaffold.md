# Skill 02 — Route-aware causal simulator v2 scaffold 구축

## 목적

`/getBs02`에서 확보한 route_id / direction_id / ordered_stop_sequence를 기반으로 route-aware causal simulator v2의 최소 scaffold를 만든다.

이 스킬은 실제 성능 평가용 simulator가 아니라, route-stop sequence 위에서 bus-like agent가 reset/step/rollout을 수행하고 raw_events / window_rollup을 만들 수 있는지 검증하는 데 사용한다.

## 언제 사용하나

- route-stop sequence가 확보된 뒤 simulator skeleton을 만들 때
- 버스 agent가 stop_order 위에서 이동하는 최소 모델이 필요한 때
- canonical KPI(Key Performance Indicator=핵심 성과 지표) aggregation으로 넘길 window_rollup이 필요한 때
- 실제 headway 관측값 없이 route-aware 구조만 먼저 검증할 때

## 입력

- causal simulator v2 contract JSON/MD
- `/getBs02` normalized route-stop sequence
- condition list: 예) A,A90,A80,A70
- seed list: 예) 1,2,3
- max_routes, max_steps 같은 smoke control parameter

## 절차

### 1. Contract 확인

아래 필드가 candidate 이상인지 확인한다.

```text
route_id
direction_id
ordered_stop_sequence
stop_id
```

아래 필드는 반드시 not_observed로 유지한다.

```text
actual_headway
actual_arrival_departure_time
actual_dwell
```

### 2. Route table 생성

필수 컬럼:

```text
route_id
direction_id
stop_order
stop_id
node_uid 또는 node_index 후보
```

정렬 기준:

```text
route_id, direction_id, stop_order
```

### 3. Minimal simulator 구현

최소 상태:

```text
agent_id
condition_id
seed
route_id
direction_id
current_stop_order
current_stop_id
step_index
action
```

최소 action:

```text
hold
advance
```

action 의미:

```text
hold    → 같은 stop_order 유지
advance → 다음 stop_order로 이동
```

### 4. Rollout trace 생성

trace row 예시:

```text
condition_id
seed
route_id
direction_id
agent_id
step_index
action
prev_stop_id
next_stop_id
prev_stop_order
next_stop_order
```

### 5. raw_events / window_rollup 작성

raw_events는 step-level event를 저장한다.
window_rollup은 canonical KPI aggregator가 읽을 수 있는 window 단위 요약을 저장한다.

필수 window_rollup 컬럼:

```text
condition_id
seed
window_id
state_ts
service_date
time_band
evaluation_horizon_minutes
headway_mean_seconds
headway_std_seconds
headway_sample_count
bunching_event_count
headway_event_count
wait_total_passenger_seconds
wait_passenger_count
ontime_event_count
schedulable_arrival_count
intervention_count
decision_step_count
energy_proxy_total
source_mode
```

### 6. source_mode 정규화

canonical aggregator가 non-causal 상태를 놓치지 않게 source_mode에는 `non_causal` 또는 `non-causal` 패턴을 포함한다.

좋은 예:

```text
route_aware_minimal_scaffold_step102_non_causal
```

나쁜 예:

```text
route_aware_minimal_scaffold_step102_noncausal
```

## 완료 기준

- route-aware simulator self-test PASS
- raw_events 생성
- window_rollup 생성
- parquet 또는 CSV fallback 생성
- canonical KPI official_rollup 통과
- causal_allowed=false 유지
- actual_* observed로 승격하지 않음

## 대표 출력

```text
[OK] scaffold_status: READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD
[OK] rollout_status: READY_FOR_STEP103_CANONICAL_KPI_SCAFFOLD
[OK] integration_status: READY_FOR_STEP104_PROJECT_LOG_RUNBOOK_UPDATE
[OK] causal_allowed: False
```

## 에이전트용 프롬프트

```text
Step 100 causal simulator v2 contract와 getBs02 route-stop sequence를 입력으로 받아 route-aware minimal simulator scaffold를 작성하라.
버스 agent는 ordered_stop_sequence 위에서 hold/advance action만 수행한다.
이 단계는 성능 평가가 아니라 구조 검증이다.
raw_events와 window_rollup을 생성하고 canonical KPI aggregator official_rollup에 연결하라.
source_mode는 non_causal로 정규화하고 causal_allowed=false를 유지하라.
actual_headway, actual_arrival_departure_time, actual_dwell은 not_observed로 유지하라.
```
