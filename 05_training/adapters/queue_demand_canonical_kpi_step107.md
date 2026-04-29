# Step 107 — Queue/Demand Canonical KPI Integration

## Purpose

Step 107 connects the Step 106 minimal queue/demand scaffold to the existing
canonical KPI (Key Performance Indicator=핵심 성과 지표) aggregation path.

The goal is not to make a performance claim.  The goal is to verify that queue
and demand proxy fields can survive the canonical evaluation boundary.

## Inputs

Default input root:

```text
artifacts/daegu_bis_api_audit/minimal_queue_demand_step106
```

Required input:

```text
window_rollup_queue_demand.parquet
```

This file is produced by Step 106 and contains:

```text
passenger_demand_generated
passenger_served_count
passenger_service_rate
passenger_wait_p95_seconds
energy_proxy_per_passenger
fleet_reduction_ratio
```

## Outputs

Default output root:

```text
artifacts/daegu_bis_api_audit/queue_demand_canonical_kpi_step107
```

Main outputs:

```text
step107_canonical_aggregator_contract.json
canonical_input_staging/window_rollup.parquet
canonical_eval/kpi_by_window.parquet
canonical_eval/kpi_by_seed.parquet
canonical_eval/kpi_by_time_band.parquet
canonical_eval/kpi_overall.json
canonical_eval/kpi_by_window_queue_demand_preserved.parquet
queue_demand_kpi_extension_summary_step107.json
queue_demand_canonical_readiness_step107.csv
queue_demand_canonical_kpi_step107_manifest.json
queue_demand_canonical_kpi_step107_report.md
```

## Why a preserved output is needed

The existing canonical KPI aggregator computes the stable six shared KPIs:

```text
cv_headway
avg_wait_seconds
bunching_rate
on_time_rate
intervention_rate
energy_proxy
```

Depending on the local version of the aggregator, optional queue/demand columns
may not be preserved in `kpi_by_window.parquet`.  Step 107 therefore creates:

```text
canonical_eval/kpi_by_window_queue_demand_preserved.parquet
```

This file joins Step 106 queue/demand proxy columns back onto canonical
`kpi_by_window` using:

```text
condition_id + seed + window_id
```

## Claim guard

Step 107 must keep:

```text
queue_demand_observed = false
queue_demand_proxy = true
actual_passenger_wait_observed = false
actual_headway_observed = false
actual_arrival_departure_time_observed = false
actual_dwell_observed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
canonical causal_comparison_allowed = false
```

## Run

```powershell
python -m py_compile `
  05_training/adapters/queue_demand_canonical_kpi_step107.py `
  05_training/adapters/test_queue_demand_canonical_kpi_step107.py

python 05_training/adapters/test_queue_demand_canonical_kpi_step107.py

python 05_training/adapters/queue_demand_canonical_kpi_step107.py
```

## Expected success signal

```text
[OK] Step 107 queue/demand canonical KPI integration completed
[OK] audit_status      : PASS
[OK] integration_status: READY_FOR_STEP108_QUEUE_DEMAND_REWARD_WIRING
[OK] all_12_kpis       : True
[OK] causal_allowed    : False
```

## Interpretation

Step 107 proves that queue/demand proxy rollups can pass through canonical KPI
aggregation and remain available for downstream reward wiring tests.

It does not prove real passenger demand, real passenger wait, actual headway,
actual arrival/departure time, actual dwell time, or causal policy performance.
