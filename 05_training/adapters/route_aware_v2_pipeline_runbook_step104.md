# Step 104 — Route-aware causal simulator v2 runbook

## Purpose

This runbook freezes the current Step 99-D through Step 103 route-aware v2 pipeline status.
It is a continuation guide, not a performance report.

The current pipeline proves that BIS API artifacts can be connected into a route-aware simulator scaffold and then passed through canonical KPI aggregation.
It does **not** prove real-world dispatch performance.

## Completed gates

| Step | Status | Output meaning |
|---|---:|---|
| Step 99-D | PASS | getRealtime02 ETA rows and ETA-based headway candidates collected as candidates |
| Step 99-E | PASS | getBs02, getPos02, getRealtime02 artifacts cross-match on route/direction/stop keys |
| Step 99-F | PASS | Step 97/98 classification updates consolidated |
| Step 100 | PASS | causal simulator v2 contract updated from classification patch |
| Step 101 | PASS | route-aware minimal simulator scaffold created |
| Step 102 | PASS | raw_events/window_rollup writer scaffold created |
| Step 103 | PASS | route-aware window_rollup passed through canonical KPI official_rollup |

## Canonical guardrails

The following fields must remain blocked until real observed event data is available:

```text
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

`eta_based_headway` is a candidate/calibration signal only. It must not be upgraded to actual headway.

## Re-run commands

Run from project root:

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"
```

### Step 99-E integration audit

```powershell
python -m py_compile `
  05_training/adapters/analyze_bis_api_integration_step99e.py `
  05_training/adapters/test_bis_api_integration_step99e.py

python 05_training/adapters/test_bis_api_integration_step99e.py
python 05_training/adapters/analyze_bis_api_integration_step99e.py
```

Expected:

```text
getPos02 coverage = 1.0
getRealtime02 ETA coverage = 1.0
headway candidate coverage = 1.0
```

### Step 99-F classification consolidation

```powershell
python -m py_compile `
  05_training/adapters/consolidate_bis_api_classification_step99f.py `
  05_training/adapters/test_bis_api_classification_step99f.py

python 05_training/adapters/test_bis_api_classification_step99f.py
python 05_training/adapters/consolidate_bis_api_classification_step99f.py
```

Expected:

```text
audit_status = PASS
actual_headway = not_observed
causal_performance_claim_allowed = false
```

### Step 100 contract update

```powershell
python -m py_compile `
  05_training/adapters/build_causal_simulator_v2_contract_step100.py `
  05_training/adapters/test_causal_simulator_v2_contract_step100.py

python 05_training/adapters/test_causal_simulator_v2_contract_step100.py
python 05_training/adapters/build_causal_simulator_v2_contract_step100.py
```

Expected:

```text
contract_status = READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD
```

### Step 101 route-aware minimal simulator scaffold

```powershell
python -m py_compile `
  05_training/adapters/route_aware_minimal_simulator_step101.py `
  05_training/adapters/test_route_aware_minimal_simulator_step101.py

python 05_training/adapters/test_route_aware_minimal_simulator_step101.py
python 05_training/adapters/route_aware_minimal_simulator_step101.py
```

Expected:

```text
scaffold_status = READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD
trace_rows = 24
```

### Step 102 route-aware rollout writer scaffold

```powershell
python -m py_compile `
  05_training/adapters/route_aware_rollout_writer_step102.py `
  05_training/adapters/test_route_aware_rollout_writer_step102.py

python 05_training/adapters/test_route_aware_rollout_writer_step102.py
python 05_training/adapters/route_aware_rollout_writer_step102.py `
  --conditions A,A90,A80,A70 `
  --seeds 1,2,3 `
  --max-routes 2 `
  --baseline-bus-count 3 `
  --max-steps 8
```

Expected:

```text
raw_events = 480
window_rollup = 24
parquet_ready = True
```

### Step 103 canonical KPI integration

```powershell
python -m py_compile `
  05_training/adapters/route_aware_canonical_kpi_step103.py `
  05_training/adapters/test_route_aware_canonical_kpi_step103.py

python 05_training/adapters/test_route_aware_canonical_kpi_step103.py
python 05_training/adapters/route_aware_canonical_kpi_step103.py
```

Expected:

```text
kpi_by_window = 24
kpi_by_seed = 12
kpi_by_time_band = 12
causal_allowed = False
```

## Commit sequence

Recommended commits:

```powershell
git add `
  .\05_training\adapters\route_aware_canonical_kpi_step103.py `
  .\05_training\adapters\test_route_aware_canonical_kpi_step103.py `
  .\05_training\adapters\route_aware_canonical_kpi_step103.md

git commit -m "Connect route-aware rollout to canonical KPI aggregation"
```

Then Step 104:

```powershell
git add `
  .\05_training\adapters\update_project_log_step104.py `
  .\05_training\adapters\test_project_log_update_step104.py `
  .\05_training\adapters\route_aware_v2_pipeline_runbook_step104.md `
  .\05_training\adapters\project_log_runbook_update_step104.md `
  .\project_log.md

git commit -m "Document route-aware causal simulator v2 pipeline progress"
```

Artifacts under `artifacts/` should normally stay uncommitted because they are reproducible local outputs.

## Next recommended steps

1. Step 105 — route-aware v2 realism gap closure plan.
2. Step 106 — passenger queue and demand proxy v2 design.
3. Step 107 — getRealtime02 ETA calibration hook design.
4. Step 108 — formal simulator adapter interface for route-aware v2.
5. Step 109 — 12-KPI contract alignment check for route-aware v2.
