# Step 97 — Tensor DB Data Availability and Generability Audit

Status: **package-ready; actual DB inspection must be run locally**  
Scope: Causal simulator v2 설계 전 데이터 가용성/생성가능성 감사  
Claim status: **paper-level performance claim is prohibited**

## 1. Purpose

Step 97의 목적은 causal simulator v2를 설계하기 전에, 현재 tensor DB(Database=데이터베이스)와 관련 source relation에서 어떤 field를 실제로 사용할 수 있는지 분류하는 것이다.

이 단계는 simulator 구현이 아니다. 성능 평가도 아니다. 이 단계는 다음 질문에 답한다.

```text
v2 simulator required contract에 넣어도 되는 데이터는 무엇인가?
optional/proxy로만 둬야 하는 데이터는 무엇인가?
missing/assumed라서 Step 99 데이터 확보 계획으로 넘겨야 하는 데이터는 무엇인가?
```

## 2. Classification rules

모든 candidate field는 다음 다섯 등급 중 하나로 분류한다.

| Class | Meaning | Required contract allowed? | Paper-level claim permitted? |
|---|---|---:|---:|
| `observed` | 현재 DB 또는 tensor artifact에 실제 관측/적재된 raw 또는 normalized value가 존재 | yes | field별 제한 |
| `derived` | observed field만으로 명확한 공식에 의해 계산 가능 | yes | field별 제한 |
| `proxy` | 실제 관측값은 아니지만 기존 field로 간접 근사 가능 | no, optional/debug only | no |
| `assumed` | field가 없고 상수, 규칙, 시나리오 설정으로 만든 값 | no | no |
| `missing` | 현재 tensor DB와 확인 가능한 source relation에서 만들 수 없음 | no | no |

강제 원칙:

- 직접 관측된 데이터만 `observed`로 표시한다.
- 계산 가능한 것은 `derived`로 표시한다.
- 간접 근사치는 `proxy`로 표시한다.
- 가정으로 만든 것은 `assumed`로 표시한다.
- 없는 것은 `missing`으로 표시한다.
- `proxy`, `assumed`, `missing`은 `paper_level_claim_allowed=false`이다.
- `observed` 또는 `derived`만 Step 98 causal simulator v2 required contract 후보가 될 수 있다.

## 3. Candidate v2 simulator data requirements

Step 97은 45개 candidate field를 감사한다.

1. stop_id / node_uid
2. node_index
3. directed edge src/dst
4. edge distance
5. edge travel time
6. route_id
7. direction_id
8. ordered stop sequence
9. state_ts
10. next_state_ts
11. service_date
12. hour_of_day / time band
13. is_peak
14. boardings_recent
15. alightings_recent
16. waiting_passenger_cnt
17. next_boardings_recent
18. next_alightings_recent
19. next_waiting_passenger_cnt
20. passenger_arrival_rate
21. passenger_queue_count
22. passenger_wait_age_distribution
23. passenger_wait_p95_seconds
24. passenger_service_rate
25. bus_id
26. vehicle trajectory
27. bus arrival time at stop
28. bus departure time at stop
29. headway_seconds
30. dwell_time_seconds
31. vehicle_capacity
32. vehicle_load
33. left_behind_passengers
34. active_bus_count
35. baseline_bus_count
36. fleet_reduction_ratio
37. hold_seconds
38. skip_stop_action_legality
39. dispatch_action_legality
40. traffic_delay
41. signal_delay
42. weather/event effect
43. incident delay
44. energy_proxy
45. energy_proxy_per_passenger

The authoritative machine-readable version is `tensor_db_data_requirements_v2.json`.

## 4. Tensor DB/source relation inspection scope

The inspector checks these relation candidates using read-only SQL only.

| Relation | Role |
|---|---|
| `public.gatv2_node_master_active` | node identity and node index candidate source |
| `public.gatv2_edge_primary_active` | edge candidate source |
| `public.gatv2_edge_primary_active_mat` | materialized edge candidate source |
| `public.gatv2_snapshot_stop_features_train` | GATv2 training feature source |
| `public.gatv2_snapshot_stop_features_train_mat` | materialized GATv2 training feature source |
| `public.graph_state_timeslice` | graph state source |
| `public.rl_state_training_base` | RL transition source |
| `public.fact_stop_usage_hourly` | normalized hourly fact source |
| `public.stg_daegu_stop_usage_2023` | raw CSV staging source |

## 5. Availability matrix

The JSON file contains the baseline availability matrix. The actual local DB inspection can refine source relation existence and source column matching.

Main interpretation:

- `node_uid`, `node_index`, `state_ts`, `next_state_ts`, demand fields, and edge geometry/time fields are likely observed if their columns exist in the inspected relations.
- `passenger_arrival_rate`, `passenger_service_rate`, `fleet_reduction_ratio`, `energy_proxy`, and `energy_proxy_per_passenger` are only derived/proxy/assumed depending on source availability and experimental definitions.
- vehicle-level fields such as `bus_id`, `vehicle_trajectory`, actual arrival/departure time, actual headway, dwell time, vehicle load, and left-behind passengers are missing unless a separate source exists.

## 6. What can be used in v2 required contract

Only fields classified as `observed` or `derived` can be promoted into Step 98 required contract.

Examples of likely required candidates after source confirmation:

- stop/node identity
- node index
- directed edge src/dst
- edge distance
- edge travel time
- state timestamp
- next state timestamp
- service date
- time band / peak indicator
- boardings and alightings fields
- waiting passenger count fields

## 7. What can be used only as optional/proxy

Fields classified as `proxy` can be used for debugging, shaping, or exploratory reward terms only when they are clearly labeled.

Rules:

- `proxy_kpi=true`
- `observed_kpi=false`
- `paper_level_claim_allowed=false`
- `causal_performance_claim_allowed=false`

Likely proxy-only candidates include:

- passenger arrival rate inferred from counts
- passenger queue count approximated from waiting passenger count
- passenger wait p95 without real wait-age distribution
- headway proxy without bus arrival events
- dwell time proxy without stop arrival/departure events

## 8. What must remain missing/assumed

Missing or assumed fields must not be placed into v2 required contract.

Likely missing candidates:

- bus ID
- vehicle trajectory
- bus arrival time at stop
- bus departure time at stop
- actual headway seconds
- observed dwell time
- vehicle load
- left-behind passengers
- passenger wait age distribution
- traffic delay
- signal delay
- weather/event effect
- incident delay

## 9. Paper-claim guardrails

This Step does not support any paper-level performance claim.

Mandatory flags:

```text
performance_claim_allowed=false
paper_level_claim_allowed=false for proxy/assumed/missing
causal_performance_claim_allowed=false for proxy/assumed/missing
fleet_reduction_claim_allowed=false unless later validated with observed/derived fleet contract
```

Forbidden claims:

- no performance improvement claim
- no fleet reduction performance claim
- no causal performance claim
- no claim that proxy wait p95 is observed wait p95
- no claim that assumed energy terms are observed energy use

Operational guardrails:

- 2023 CSV 재적재 금지
- 기존 tensor DB 덮어쓰기 금지
- 2022 CSV는 현재 2023 tensor DB에 병합 금지
- UPDATE / INSERT / DELETE / TRUNCATE / CREATE / DROP / ALTER 금지

## 10. How Step 97 constrains Step 98

Step 98, causal simulator v2 contract, must use Step 97 output buckets as a gate.

Step 98 may use:

- `observed_or_derived_required_candidates` as required contract candidates
- `proxy_optional_candidates` as optional/debug candidates only

Step 98 must exclude:

- `assumed_or_missing_excluded_candidates`

## 11. How Step 97 leads to Step 99 missing data acquisition plan

All `missing` fields and all fields whose acquisition path is `current_original_db`, `external_data_required`, or `not_available` should be passed to Step 99.

Step 99 should split missing data into:

- obtainable from current original DB
- obtainable from existing non-tensor project data
- requires external data
- unavailable or out of scope

## 12. Local execution command for actual DB inspection

Codex should not run the actual DB inspection. Run it locally in PowerShell after confirming the files exist.

```powershell
$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
Set-Location $ProjectRoot

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$env:URBANBUS_DB_DSN = "postgresql://USER:PASSWORD@HOST:PORT/DBNAME"

& $py ".\05_training\adapters\inspect_tensor_db_available_fields.py" `
  --output-root ".\artifacts\tensor_db_data_availability_audit"
```

Expected outputs:

```text
artifacts/tensor_db_data_availability_audit/tensor_db_available_fields_report.json
artifacts/tensor_db_data_availability_audit/tensor_db_available_fields_report.md
```
