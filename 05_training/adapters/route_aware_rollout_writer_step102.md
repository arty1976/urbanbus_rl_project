# Step 102 — Route-aware Rollout Writer Scaffold

## Purpose

Step 102 connects the Step 101 route-aware minimal simulator scaffold to the
project's rollout artifact shape. It writes `raw_events` and `window_rollup`
files that can be consumed by the canonical KPI (Key Performance Indicator=핵심 성과 지표)
path in the next step.

This step is still a scaffold. It does **not** validate real bus operation
performance.

## Inputs

Default inputs:

```text
artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/causal_simulator_v2_contract.json
artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/getbs02_route_stop_sequence_normalized.csv
```

The script reads only existing CSV/JSON artifacts. It performs no DB (Database=데이터베이스)
write, no tensor DB overwrite, and no additional API (Application Programming Interface=응용 프로그램 인터페이스)
call.

## Outputs

Default output folder:

```text
artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/
```

Files:

```text
raw_events.csv
raw_events.parquet
window_rollup.csv
window_rollup.parquet
route_aware_rollout_writer_manifest.json
route_aware_rollout_writer_report.md
```

The direct `window_rollup.parquet` name is intentional. It allows the existing
canonical KPI aggregator to discover the output folder in the following step.

## Generated Columns

`window_rollup` includes the official rollup columns used by the current
canonical aggregator:

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

It also includes the 12-KPI schema columns:

```text
cv_headway
avg_wait_seconds
bunching_rate
on_time_rate
intervention_rate
energy_proxy
passenger_demand_generated
passenger_served_count
passenger_service_rate
passenger_wait_p95_seconds
energy_proxy_per_passenger
fleet_reduction_ratio
```

## Claim Boundary

The generated headway values are route-stop gap proxies converted to seconds to
fit the existing rollout schema. They are not actual observed headway.

These fields remain blocked:

```text
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## Commands

```powershell
python -m py_compile `
  05_training/adapters/route_aware_rollout_writer_step102.py `
  05_training/adapters/test_route_aware_rollout_writer_step102.py

python 05_training/adapters/test_route_aware_rollout_writer_step102.py

python 05_training/adapters/route_aware_rollout_writer_step102.py
```

Optional A-family scaffold matrix:

```powershell
python 05_training/adapters/route_aware_rollout_writer_step102.py `
  --conditions A,A90,A80,A70 `
  --seeds 1,2,3 `
  --max-routes 2 `
  --baseline-bus-count 3 `
  --max-steps 8
```

## Next Step

Step 103 should feed `window_rollup.parquet` into the canonical KPI aggregator
and verify that the route-aware scaffold output can pass the official rollup path
while retaining all claim guards.
