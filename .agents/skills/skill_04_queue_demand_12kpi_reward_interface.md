# Skill 04 — Queue/Demand 12-KPI reward-policy interface scaffold

## 목적

route-aware rollout에 최소 queue/demand proxy를 붙이고, 12-KPI preserved output을 만든 뒤, 임시 scaffold reward와 policy-facing reward interface까지 연결한다.

이 스킬은 실제 MAPPO(Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) 학습 reward를 확정하는 것이 아니라, 12-KPI → reward component → policy interface 경로가 작동하는지 검증하는 용도다.

## 언제 사용하나

- 기존 6-KPI를 12-KPI로 확장해야 할 때
- passenger_service_rate, passenger_wait_p95_seconds 같은 queue/demand proxy가 필요할 때
- reward component 배선을 검증하고 싶지만 최종 reward는 아직 확정하지 않았을 때
- policy runner가 읽을 수 있는 reward interface schema를 만들 때

## 입력

- Step 102 또는 이후의 raw_events / window_rollup
- canonical 6-KPI output
- queue/demand proxy columns
- reward wiring scaffold config

## 12-KPI 목록

기존 canonical 6-KPI:

```text
cv_headway
avg_wait_seconds
bunching_rate
on_time_rate
intervention_rate
energy_proxy
```

queue/demand 확장 6-KPI:

```text
passenger_demand_generated
passenger_served_count
passenger_service_rate
passenger_wait_p95_seconds
energy_proxy_per_passenger
fleet_reduction_ratio
```

## Step A — Minimal queue/demand scaffold

추가할 proxy:

```text
demand_arrival_count
passenger_queue_before
passenger_queue_after
passenger_boarded_count
passenger_service_rate
passenger_wait_p95_seconds
```

guard:

```json
{
  "queue_demand_observed": false,
  "queue_demand_proxy": true,
  "actual_passenger_wait_observed": false
}
```

## Step B — 12-KPI preserved output 생성

canonical aggregator가 6-KPI만 공식 집계하더라도 wrapper에서 queue/demand 확장 KPI를 다시 join하여 preserved output을 만든다.

권장 파일:

```text
canonical_eval/kpi_by_window_queue_demand_preserved.parquet
canonical_eval/kpi_by_window_queue_demand_preserved.csv
```

필수 확인:

```text
all_12_kpis = True
causal_allowed = False
preserved_rows > 0
```

## Step C — Scaffold reward wiring

임시 reward component:

```text
reward_service
reward_wait
reward_long_wait
reward_energy
reward_fleet
reward_total
```

중요:
이 reward는 최종 MAPPO 학습 reward가 아니다.

반드시 넣을 guard:

```json
{
  "reward_scaffold_only": true,
  "reward_weights_are_final": false,
  "reward_formula_finalized": false,
  "final_reward_design_claim_allowed": false,
  "train_with_this_reward_allowed": false
}
```

## Step D — Policy-facing reward interface

policy runner가 읽기 쉬운 형태로 변환한다.

권장 컬럼:

```text
condition_id
seed
window_id
state_ts
time_band
reward_total_scaffold
reward_vector_json
reward_scaffold_only
train_with_this_reward_allowed
final_reward_design_claim_allowed
causal_performance_claim_allowed
```

## 완료 기준

- minimal queue/demand scaffold PASS
- 12-KPI preserved output 생성
- all_12_kpis=True
- reward_by_window rows > 0
- policy_reward_interface rows > 0
- reward_scaffold_only=True
- train_with_this_reward_allowed=False
- final_reward_design_claim_allowed=False
- causal_allowed=False

## 해석 금지

이 스킬의 결과로 다음을 주장하면 안 된다.

```text
이 reward가 최종 reward 설계다.
이 reward로 실제 MAPPO 학습을 시작해도 된다.
reward_total로 A/A90/A80/A70 성능을 비교할 수 있다.
논문 성능표에 넣을 수 있다.
```

## 에이전트용 프롬프트

```text
Step 106~109 queue/demand 12-KPI reward-policy scaffold를 작성하라.
route-aware rollout에 demand_arrival_count, passenger_queue, passenger_boarded_count, passenger_service_rate, passenger_wait_p95_seconds proxy를 추가하라.
canonical KPI output과 queue/demand 확장 KPI를 결합해 12-KPI preserved kpi_by_window를 만들라.
그다음 12-KPI를 임시 scaffold reward component로 변환하고 policy-facing reward interface를 생성하라.
이 reward는 최종 MAPPO reward가 아니므로 reward_scaffold_only=true, train_with_this_reward_allowed=false, final_reward_design_claim_allowed=false를 반드시 유지하라.
```
