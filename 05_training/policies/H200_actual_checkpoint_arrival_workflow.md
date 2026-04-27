# Step 62 — H200 actual checkpoint arrival workflow

## 1. Purpose

This step defines the workflow that starts when a real H200-trained MAPPO checkpoint arrives.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) actual execution must not begin directly from a checkpoint file. The checkpoint must first pass the Step 58 actual preflight, then generate a Step 59 READY_TO_EXECUTE plan, and then pass the Step 60 dry-run guard.

Step 62 connects those gates into one arrival workflow.

---

## 2. Workflow contract

Input:

```text
checkpoint_path
output_root
```

Output:

```text
h200_actual_checkpoint_preflight_report.json
a_family_mappo_actual_execution_plan.json
a_family_mappo_actual_matrix_runner_dry_run_report.json
h200_actual_checkpoint_arrival_workflow_report.json
```

The workflow stops at dry-run validation. It does not execute the actual 12-run matrix.

---

## 3. Required order

```text
1. checkpoint file existence check
2. Step 58 actual preflight
3. preflight report PASS check
4. checkpoint_path consistency check
5. Step 59 execution plan generation
6. plan status READY_TO_EXECUTE check
7. Step 60 matrix runner dry-run
8. dry-run report DRY_RUN_VALIDATED check
9. write arrival workflow report
```

---

## 4. Stop conditions

The workflow must stop if any item below is true.

```text
checkpoint file does not exist
Step 58 preflight command fails
preflight report is missing
preflight report status is BLOCKED
preflight report checkpoint_path differs from input checkpoint_path
Step 59 plan command fails
plan JSON is missing
plan status is not READY_TO_EXECUTE
Step 60 dry-run command fails
dry-run report is missing
dry-run report status is not DRY_RUN_VALIDATED
dry-run report executed = true
```

---

## 5. Success condition

The only success state is:

```text
arrival_workflow_status = READY_FOR_MANUAL_EXECUTE
preflight_status = PASS
plan_status = READY_TO_EXECUTE
dry_run_status = DRY_RUN_VALIDATED
```

This means the actual checkpoint is ready for a human operator to execute the matrix through the Step 61 runbook.

It does not mean causal performance has been proven.

---

## 6. Claim boundary

HistoricalReplayAdapter or any historical/replay/noncausal adapter may validate the actual policy execution path, but it cannot support causal performance claims.

Correct phrase:

```text
A-family actual MAPPO checkpoint arrival workflow passed up to dry-run validation.
```

Incorrect phrase:

```text
A-family actual MAPPO causally outperformed baselines.
```

---

## 7. Default command template

The workflow expects the prior scripts to support the following command patterns.

```powershell
# Step 58
python 05_training/policies/h200_actual_checkpoint_preflight_checklist.py `
  --checkpoint-path <checkpoint_path> `
  --mode actual `
  --report-path <preflight_report>

# Step 59
python 05_training/policies/a_family_mappo_actual_execution_plan.py `
  --preflight-report <preflight_report> `
  --checkpoint-path <checkpoint_path> `
  --output-plan <plan_json>

# Step 60 dry-run
python 05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py `
  --plan-json <plan_json> `
  --report-path <dry_run_report> `
  --dry-run
```

---

## 8. Operator interpretation

If Step 62 passes, proceed to Step 61 runbook execution review.

If Step 62 blocks, do not patch around the block. Fix the checkpoint, report, plan, or command contract first.
