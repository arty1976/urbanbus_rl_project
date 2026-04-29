# Step 106 — Minimal Queue/Demand Scaffold

## Purpose

Step 106 adds a deterministic minimal queue/demand proxy layer on top of the
Step 102 route-aware rollout scaffold.

It introduces the first scaffold version of:

- `demand_arrival_count`
- `passenger_queue_before`
- `service_capacity_available`
- `passenger_boarded_count`
- `passenger_queue_after`
- `passenger_left_behind_count`
- `passenger_service_rate`
- `passenger_wait_p95_seconds`
- `avg_queue_depth`
- `max_queue_depth`
- `unmet_demand_rate`

## Inputs

Default inputs:

```text
artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/raw_events.parquet
artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/window_rollup.parquet
```

CSV fallback is supported if parquet files are unavailable.

## Outputs

Default output directory:

```text
artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/
```

Generated files:

```text
raw_events_queue_demand.csv
raw_events_queue_demand.parquet
window_rollup_queue_demand.csv
window_rollup_queue_demand.parquet
queue_demand_readiness_step106.csv
minimal_queue_demand_manifest.json
minimal_queue_demand_report.md
```

## Safety boundary

This is still a scaffold.  It does **not** observe real passenger arrivals,
real individual waiting times, real vehicle load, actual dwell time, or actual
arrival/departure timestamps.

The following guards must stay fixed:

```text
queue_demand_observed = false
queue_demand_proxy = true
actual_passenger_wait_observed = false
actual_headway_observed = false
actual_arrival_departure_time_observed = false
actual_dwell_observed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## How to run

```powershell
python -m py_compile `
  05_training/adapters/minimal_queue_demand_step106.py `
  05_training/adapters/test_minimal_queue_demand_step106.py

python 05_training/adapters/test_minimal_queue_demand_step106.py

python 05_training/adapters/minimal_queue_demand_step106.py
```

Expected success signal:

```text
[OK] Step 106 minimal queue/demand scaffold completed
[OK] audit_status   : PASS
[OK] scaffold_status: READY_FOR_STEP107_QUEUE_DEMAND_CANONICAL_KPI
```

## Next step

Step 107 should pass `window_rollup_queue_demand.parquet` through the canonical
KPI (Key Performance Indicator=핵심 성과 지표) aggregator and verify that the
queue/demand proxy columns are preserved while causal claims remain blocked.
