# Step 104 — Project log and route-aware v2 runbook update

This step records the completed Step 99-D through Step 103 route-aware causal simulator v2 scaffold progress.

It updates `project_log.md` idempotently and writes a continuation runbook.

## Files

- `update_project_log_step104.py`
- `test_project_log_update_step104.py`
- `route_aware_v2_pipeline_runbook_step104.md`
- `project_log_runbook_update_step104.md`

## Usage

```powershell
python -m py_compile `
  05_training/adapters/update_project_log_step104.py `
  05_training/adapters/test_project_log_update_step104.py

python 05_training/adapters/test_project_log_update_step104.py
python 05_training/adapters/update_project_log_step104.py
```

## Non-claim guard

The update explicitly preserves:

```text
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```
