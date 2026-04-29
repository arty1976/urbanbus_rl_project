# Step 103 — Route-aware Rollout Canonical KPI Integration Scaffold

## Purpose

Step 103 connects the Step 102 route-aware rollout writer output to the existing canonical KPI (Key Performance Indicator=핵심 성과 지표) aggregation path.

This step answers one narrow question:

> Can `window_rollup.parquet` produced by the route-aware minimal scaffold pass the existing `canonical_kpi_aggregator.py --mode official_rollup` path?

It does **not** claim route-aware causal performance.

## Inputs

Default input root:

```text
artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102
```

Required file:

```text
window_rollup.parquet
```

Default canonical aggregator:

```text
05_training/evaluation/canonical_kpi_aggregator.py
```

## Outputs

Default output root:

```text
artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103
```

Main outputs:

```text
step103_canonical_aggregator_contract.json
canonical_input_staging/window_rollup.parquet
canonical_eval/official_rollup_input_validation.json
canonical_eval/kpi_by_window.parquet
canonical_eval/kpi_by_seed.parquet
canonical_eval/kpi_by_time_band.parquet
canonical_eval/kpi_overall.json
canonical_eval/aggregation_manifest.json
canonical_output_readiness_step103.csv
route_aware_canonical_kpi_step103_manifest.json
route_aware_canonical_kpi_step103_report.md
```

## Source mode guard

Step 102 may produce a `source_mode` containing `noncausal` as one glued token.
The current canonical aggregator detects `non_causal` and `non-causal`, but may not detect `noncausal`.
Therefore Step 103 creates a staged copy of `window_rollup.parquet` and normalizes:

```text
noncausal -> non_causal
```

This is a guard-preserving normalization. It prevents route-aware scaffold outputs from being treated as causal-comparison-ready.

## Claim guards

The following must remain false:

```text
db_write_performed = false
tensor_db_overwrite_performed = false
additional_api_calls_performed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
actual_headway_observed = false
actual_arrival_departure_time_observed = false
actual_dwell_observed = false
```

## What Step 103 proves

Step 103 proves:

```text
Step 102 route-aware window_rollup.parquet
-> canonical KPI official_rollup input validation
-> kpi_by_window.parquet
-> kpi_by_seed.parquet
-> kpi_by_time_band.parquet
-> kpi_overall.json
```

## What Step 103 does not prove

Step 103 does not prove:

```text
actual headway observation
actual arrival/departure observation
actual dwell observation
realistic passenger queue behavior
MAPPO policy performance
causal performance improvement
paper-level performance claim
```

## Recommended execution

```powershell
python -m py_compile `
  05_training/adapters/route_aware_canonical_kpi_step103.py `
  05_training/adapters/test_route_aware_canonical_kpi_step103.py

python 05_training/adapters/test_route_aware_canonical_kpi_step103.py

python 05_training/adapters/route_aware_canonical_kpi_step103.py
```

Optional clean run:

```powershell
python 05_training/adapters/route_aware_canonical_kpi_step103.py --clean-output
```

## Next step

Step 104 should update the project log/runbook and record the completed path:

```text
Step 100 contract
-> Step 101 route-aware simulator scaffold
-> Step 102 route-aware rollout writer scaffold
-> Step 103 canonical KPI integration scaffold
```
