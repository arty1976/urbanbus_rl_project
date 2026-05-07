# Step 167 — Actual Route-Aware Rollout Evidence Runbook

Project: `urbanbus_rl_project`  
Scope: Zero-Loss Pickup patent evidence pipeline  
Status: `RUNBOOK_READY_STILL_NONCLAIM`

## 0. Purpose

Step 167 fixes the execution procedure for converting an actual route-aware rollout artifact into a Zero-Loss Pickup patent evidence report.

This is not a training step and not a performance-claim release step. It does not declare that the project has already produced actual operational results.

The runbook defines how to run the chain:

```text
raw_events.parquet / window_rollup.parquet
→ Step 162 route-aware rollout adapter
→ Step 161 pickup attempt / ETA counterfactual writer
→ Step 163 real GATv2 attention extractor
→ Step 165 attempt-specific route/path attention filter
→ Step 160 zero-loss pickup evidence reporter
→ Step 166 patent evidence report v2
```

## 1. Non-claim guard

The following flags must remain locked unless a later explicit release step changes them:

```text
actual_operational_claim_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
train_allowed = false
actual_results_claimed = false
```

## 2. Required inputs for actual-like run

Minimum required actual route-aware rollout inputs:

```text
raw_events.parquet or raw_events.csv
window_rollup.parquet or window_rollup.csv  [recommended]
route_stop_sequence.csv or route_stop_sequence.parquet
GATv2 snapshot/checkpoint source for attention extraction
run metadata: condition_id, seed, route/corridor scope, simulator version
```

## 3. Required Step files

The project must already contain Step 160 through Step 166 files.

```text
05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py
05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py
05_training/patent_evidence/route_aware_rollout_adapter_step162.py
05_training/patent_evidence/gatv2_real_attention_extractor_step163.py
05_training/patent_evidence/attempt_route_path_attention_filter_step165.py
05_training/patent_evidence/patent_evidence_report_v2_step166.py
```

Step 164 may be used for connector smoke, but Step 165 is preferred for the final v2 report because it performs attempt-specific route/path filtering.

## 4. Execution stages

### Stage A — Adapt route-aware rollout

Convert actual route-aware raw events into Step 161-compatible normalized events.

```powershell
python 05_training/patent_evidence/route_aware_rollout_adapter_step162.py `
  --mode from-route-aware-rollout `
  --raw-events <RAW_EVENTS_PATH> `
  --window-rollup <WINDOW_ROLLUP_PATH> `
  --output-root artifacts/patent_evidence/actual_route_aware_rollout_adapter_step167 `
  --file-format csv
```

### Stage B — Generate pickup attempt and ETA counterfactual files

```powershell
python 05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py `
  --mode from-normalized-rollout `
  --normalized-events artifacts/patent_evidence/actual_route_aware_rollout_adapter_step167/normalized_route_aware_rollout_events.csv `
  --output-root artifacts/patent_evidence/actual_pickup_attempt_events_step167 `
  --file-format csv
```

### Stage C — Extract real GATv2 attention

```powershell
python 05_training/patent_evidence/gatv2_real_attention_extractor_step163.py `
  --mode from-gatv2-snapshot `
  --output-root artifacts/patent_evidence/actual_gatv2_attention_step167 `
  --require-real-gatv2conv
```

For the initial actual-like run, `--mode sample` may be used only as a smoke substitute. Such a run must remain non-claim.

### Stage D — Attempt-specific route/path attention filtering

```powershell
python 05_training/patent_evidence/attempt_route_path_attention_filter_step165.py `
  --pickup-attempt-events artifacts/patent_evidence/actual_pickup_attempt_events_step167/pickup_attempt_events.csv `
  --eta-counterfactual artifacts/patent_evidence/actual_pickup_attempt_events_step167/eta_counterfactual.csv `
  --real-attention artifacts/patent_evidence/actual_gatv2_attention_step167/gatv2_attention.csv `
  --route-stop-sequence <ROUTE_STOP_SEQUENCE_PATH> `
  --output-root artifacts/patent_evidence/actual_attempt_path_attention_step167 `
  --file-format csv
```

### Stage E — Generate Step 160 evidence bundle

```powershell
python 05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py `
  --pickup-attempt-events artifacts/patent_evidence/actual_attempt_path_attention_step167/pickup_attempt_events.csv `
  --eta-counterfactual artifacts/patent_evidence/actual_attempt_path_attention_step167/eta_counterfactual.csv `
  --gatv2-attention artifacts/patent_evidence/actual_attempt_path_attention_step167/gatv2_attention.csv `
  --run-manifest artifacts/patent_evidence/actual_attempt_path_attention_step167/run_manifest.json `
  --output-root artifacts/patent_evidence/actual_zero_loss_evidence_step167
```

### Stage F — Generate Step 166 patent report v2

```powershell
python 05_training/patent_evidence/patent_evidence_report_v2_step166.py `
  --zero-loss-evidence-root artifacts/patent_evidence/actual_zero_loss_evidence_step167 `
  --attempt-path-attention-root artifacts/patent_evidence/actual_attempt_path_attention_step167 `
  --output-root artifacts/patent_evidence/actual_patent_evidence_report_v2_step167
```

## 5. Evidence interpretation

For actual-like run outputs, the report should answer:

```text
total_pickup_attempts
zero_loss_success_count
zero_loss_success_rate
filtered_attention_row_count
exact_edge_overlap_count
node_overlap_count
fallback_attempt_count
attention_mass_by_path_segment
```

## 6. Allowed and prohibited claims

Allowed after this Step 167 runbook self-test:

```text
The execution procedure for producing Zero-Loss Pickup evidence from route-aware rollout artifacts is fixed and validator-protected.
```

Still prohibited:

```text
Actual Daegu-wide Zero-Loss success rate claim
Paper-level performance claim
Causal performance claim
Deployment claim
```

## 7. Promotion criteria for a later actual evidence run

A later step may promote actual evidence only if all of these are true:

```text
actual route-aware rollout artifact exists
actual raw_events path is recorded
Step 162 adapter validation PASS
Step 161 pickup/ETA validation PASS
Step 163 real GATv2 attention extraction PASS
Step 165 attempt-specific path filtering PASS
Step 160 evidence bundle validation PASS
Step 166 report v2 validation PASS
claim flags remain false until explicit release
```
