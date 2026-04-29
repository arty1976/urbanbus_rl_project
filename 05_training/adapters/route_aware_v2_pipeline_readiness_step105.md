# Step 105 — Route-aware v2 Pipeline Readiness Gate

## Purpose

Step 105 is a readiness gate for the route-aware causal simulator v2 scaffold.

It checks whether the outputs from Step 100 through Step 103 form a coherent
pipeline:

```text
Step 100 causal simulator v2 contract
→ Step 101 route-aware minimal simulator scaffold
→ Step 102 route-aware rollout writer scaffold
→ Step 103 canonical KPI aggregation scaffold
→ Step 104 project log/runbook documentation
```

This is not a performance evaluation step.

## What this step validates

1. Step 100 contract exists and is ready for route-aware minimal scaffold.
2. `actual_headway`, `actual_arrival_departure_time`, and `actual_dwell` remain `not_observed`.
3. `paper_level_claim_allowed=false`.
4. `causal_performance_claim_allowed=false`.
5. Step 101 manifest exists and produced positive smoke trace rows.
6. Step 102 raw event and window rollup files exist.
7. Step 102 has positive `raw_events` and `window_rollup` row counts.
8. Step 103 canonical KPI outputs exist.
9. Step 103 keeps `causal_allowed=false`.
10. canonical `kpi_overall.json` keeps `causal_comparison_allowed=false`.

## Why this step is needed

After Step 103, the route-aware scaffold can already pass through the canonical KPI
path. That is useful, but it also creates a risk: the output looks like an official
KPI result even though it is still scaffold-level and non-causal.

Step 105 prevents that confusion by enforcing the claim boundary in one place.

## Required input artifacts

Default paths:

```text
artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/causal_simulator_v2_contract.json
artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/route_aware_minimal_simulator_manifest.json
artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/route_aware_rollout_writer_manifest.json
artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/raw_events.parquet
artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/window_rollup.parquet
artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103/route_aware_canonical_kpi_step103_manifest.json
artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103/canonical_eval/kpi_by_window.parquet
artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103/canonical_eval/kpi_by_seed.parquet
artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103/canonical_eval/kpi_by_time_band.parquet
artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103/canonical_eval/kpi_overall.json
```

Step 104 documentation outputs are checked as warnings by default and as hard errors
when `--require-step104` is supplied.

## Outputs

```text
artifacts/daegu_bis_api_audit/route_aware_v2_pipeline_readiness_step105/
  route_aware_v2_pipeline_readiness_step105.json
  route_aware_v2_pipeline_readiness_step105.md
  route_aware_v2_pipeline_readiness_matrix_step105.csv
```

## Usage

```powershell
python -m py_compile `
  05_training/adapters/validate_route_aware_v2_pipeline_step105.py `
  05_training/adapters/test_validate_route_aware_v2_pipeline_step105.py

python 05_training/adapters/test_validate_route_aware_v2_pipeline_step105.py

python 05_training/adapters/validate_route_aware_v2_pipeline_step105.py
```

If Step 104 was already executed and should be mandatory:

```powershell
python 05_training/adapters/validate_route_aware_v2_pipeline_step105.py --require-step104
```

## Expected success signal

```text
[OK] Step 105 route-aware v2 pipeline readiness gate completed
[OK] audit_status    : PASS
[OK] readiness_status: READY_FOR_STEP106_MINIMAL_QUEUE_DEMAND_SCAFFOLD
```

## Claim boundary

Step 105 must keep the following fixed:

```text
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
canonical_causal_comparison_allowed = false
```

## Next step

Recommended next step:

```text
Step 106 — minimal queue/demand scaffold on route-aware simulator
```

Step 106 should introduce a minimal passenger queue and demand proxy layer on top
of the route-aware movement scaffold, while still preserving non-claim metadata.
