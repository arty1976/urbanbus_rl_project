# Step 102 — CausalSimulatorAdapter v2 Rollout Writer Smoke Path

## 0. 목적

Step 102의 목적은 Step 101에서 만든 `CausalSimulatorV2Adapter`를 실제 rollout writer smoke path에 연결하는 것이다.

Step 101은 adapter가 reset/step을 수행할 수 있는지 확인한 단계였다.  
Step 102는 그 step 결과를 실험 산출물 파일로 저장한다.

생성 산출물:

```text
raw_events.parquet
window_rollup.parquet
rollout_manifest.json
```

---

## 1. 이 단계의 의미

쉽게 말하면 다음과 같다.

```text
Step 101:
엔진이 켜지는지 확인

Step 102:
엔진이 움직인 기록을 실험 파일로 남김
```

`raw_events.parquet`는 step 단위 기록이다.

```text
condition_id
seed
window_id
step_idx
agent_id
action
reward
demand_pressure_proxy
signal_count_250m_scaled
nearest_signal_distance_scaled
signal_delay_risk_proxy
intersection_complexity_proxy
```

`window_rollup.parquet`는 canonical KPI aggregator가 읽을 수 있는 window 단위 요약이다.

---

## 2. 중요한 guardrail

이 단계도 성능 실험이 아니다.

반드시 유지:

```text
trained_model = false
performance_claim_allowed = false
causal_performance_claim_allowed = false
dynamic_signal_phase_claim_allowed = false
red_light_delay_claim_allowed = false
green_time_claim_allowed = false
cycle_length_claim_allowed = false
```

source mode도 아래처럼 smoke/nonperformance임을 명시한다.

```text
static_signal_causal_v2_scaffold_smoke_nonperformance_v1
```

---

## 3. 생성되는 window_rollup 필수 컬럼

canonical KPI aggregator의 official_rollup 입력과 맞추기 위해 다음 컬럼을 생성한다.

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

확장 KPI 후보도 함께 생성한다.

```text
passenger_demand_generated
passenger_served_count
passenger_service_rate
passenger_wait_p95_seconds
energy_proxy_per_passenger
active_bus_count
baseline_bus_count
fleet_reduction_ratio
```

---

## 4. 금지 사항

아래 값은 생성하지 않는다.

```text
red_light_delay_seconds
green_time_seconds
cycle_length_seconds
phase_sequence
signal_offset_seconds
real_time_signal_state
queue_discharge_rate
```

---

## 5. 다음 단계

Step 103에서는 Step 102의 `window_rollup.parquet`를 `canonical_kpi_aggregator.py`에 연결하는 smoke test를 만든다.
