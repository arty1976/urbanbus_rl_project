# Step 61 — H200 actual MAPPO final runbook

## 1. Purpose

This runbook defines the final execution procedure for running the A-family actual MAPPO matrix after a real H200-trained checkpoint exists.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) actual execution is only allowed after the Step 58 preflight report passes, the Step 59 execution plan is READY_TO_EXECUTE, and the Step 60 matrix runner dry-run validates the command order and guard conditions.

This runbook does not create a trained checkpoint. It defines the final gate before using a real H200 checkpoint in the A/A90/A80/A70 matrix.

---

## 2. Scope

### Included

- Actual checkpoint preflight review
- A-family execution plan generation
- Matrix runner dry-run validation
- Actual execution command order
- Output verification
- Claim boundary rules

### Not included

- H200 training implementation
- GPU cluster scheduler configuration
- Causal simulator implementation
- Qwen training or Qwen inference integration

---

## 3. Required prior steps

The following steps must already be complete.

| Step | Requirement |
|---|---|
| Step 58 | H200 actual checkpoint preflight checklist exists and self-test passes |
| Step 59 | A-family mappo_actual execution plan exists and self-test passes |
| Step 60 | A-family mappo_actual matrix runner guard exists and self-test passes |

Required files:

```text
05_training/policies/h200_actual_checkpoint_preflight_checklist.py
05_training/policies/a_family_mappo_actual_execution_plan.py
05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py
05_training/policies/validate_window_rollup_policy_metadata.py
05_training/evaluation/canonical_kpi_aggregator.py
```

---

## 4. Non-negotiable actual checkpoint gates

A checkpoint may be used for mappo_actual only if all gates below pass.

```text
checkpoint file exists
trained_model = true
performance_claim_allowed = true
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
k_dist_kwh_per_m = 0.0012
k_acc_kwh_per_event = 0.1800
k_idle_kwh_per_sec = 0.0080
validate_mappo_checkpoint.py --mode actual PASS
```

If any item fails, the result must remain BLOCKED and must not be promoted to mappo_actual.

---

## 5. Claim boundary

### Actual policy claim

Allowed only when:

```text
policy_source_mode = mappo_actual
checkpoint_validation_mode = actual
trained_model = true
performance_claim_allowed = true
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
```

### Causal performance claim

Not allowed when using HistoricalReplayAdapter or any historical/replay/noncausal adapter.

Historical/replay adapter may run the actual policy path, but the result is still non-causal. Therefore, it can support contract validation and execution path verification, but it cannot support causal performance claims.

The phrase "A-family actual execution completed" does not mean "causal superiority proven."

---

## 6. Required A-family matrix

The Step 59 plan must contain exactly the A-family conditions and seeds below.

```text
conditions = A, A90, A80, A70
seeds = 1, 2, 3
run_count = 12
commands_per_run = 4
command_count = 48
```

Each run must execute commands in this exact order.

```text
1. rollout_command
2. metadata_validation_command
3. canonical_command
4. post_canonical_validation_command
```

This order must not be changed. The policy metadata validator must run before canonical aggregation, and post-canonical validation must run after canonical aggregation.

---

## 7. Dry-run first

Before real execution, the Step 60 runner must pass in dry-run mode.

Expected dry-run report:

```text
runner_status = DRY_RUN_VALIDATED
dry_run = true
executed = false
run_count = 12
command_count = 48
actual_claim_guard.all_passed = true
causal_claim_guard.all_passed = true
```

Dry-run validates the plan, command order, expected outputs, actual claim guard, and causal claim guard. It must not execute the real H200 matrix.

---

## 8. Real execution mode

Real execution may be run only after all items below are true.

```text
git working tree is clean or intentionally documented
Step 58 preflight report PASS
Step 59 plan READY_TO_EXECUTE
Step 60 dry-run report DRY_RUN_VALIDATED
checkpoint path matches preflight checkpoint_path
no BLOCKED run exists in the plan
no B0/B1/B2 condition exists in the plan
no Qwen intervention exists in A-family actual plan
```

Real execution must use the Step 60 runner with `--execute`.

---

## 9. Canonical output expectations

Each run must eventually produce or validate:

```text
window_rollup.parquet
canonical_eval/kpi_by_window.parquet
canonical_eval/kpi_by_seed.parquet
canonical_eval/kpi_by_time_band.parquet
canonical_eval/kpi_overall.json
canonical_eval/aggregation_manifest.json
```

For actual MAPPO outputs, policy metadata must remain visible through canonical KPI aggregation.

Required metadata fields include:

```text
policy_source
policy_source_mode
checkpoint_path
checkpoint_validation_mode
checkpoint_validator_ran
checkpoint_loaded
trained_model
performance_claim_allowed
mock_action_used
placeholder_fallback_used
qwen_train
qwen_inference
qwen_trigger_rate
reward_version
energy_proxy_model_version
actual_policy_claim_ready
causal_policy_claim_ready
```

---

## 10. Expected final reports

The execution folder should contain:

```text
h200_actual_checkpoint_preflight_report.json
a_family_mappo_actual_execution_plan.json
a_family_mappo_actual_matrix_runner_dry_run_report.json
a_family_mappo_actual_matrix_runner_execute_report.json
```

For historical/replay adapter runs, the final report must explicitly state:

```text
actual policy path execution allowed = true
causal performance claim allowed = false
```

---

## 11. Stop conditions

Immediately stop if any of the following occurs.

```text
preflight report is BLOCKED
plan status is not READY_TO_EXECUTE
run_count is not 12
command_count is not 48
checkpoint_path mismatch
policy_source_mode is not mappo_actual
checkpoint_validation_mode is not actual
trained_model is not true
performance_claim_allowed is not true
qwen_trigger_rate is not 0.0
mock_action_used is true
placeholder_fallback_used is true
condition_id outside A/A90/A80/A70
causal claim requested on HistoricalReplayAdapter
metadata validation fails before canonical aggregation
post-canonical metadata validation fails
```

---

## 12. Operator checklist

Before pressing execute:

```text
[ ] I have the real H200 checkpoint path.
[ ] The checkpoint passed Step 58 actual preflight.
[ ] The Step 59 plan was generated from that same checkpoint path.
[ ] The Step 60 dry-run report passed.
[ ] I confirmed run_count = 12.
[ ] I confirmed command_count = 48.
[ ] I confirmed A/A90/A80/A70 only.
[ ] I confirmed seeds 1,2,3.
[ ] I confirmed Qwen is disabled.
[ ] I understand historical/replay results are not causal performance claims.
```

---

## 13. Recommended PowerShell sequence

The exact paths below are templates. Replace checkpoint and report paths with the real H200 output paths.

```powershell
$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
Set-Location $ProjectRoot

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$CheckpointPath = "D:\h200_outputs\best_mappo.pt"
$PreflightReport = ".\artifacts\experiment_A_v1\h200_actual_checkpoint_preflight_report.json"
$PlanJson = ".\artifacts\experiment_A_v1\a_family_mappo_actual_execution_plan.json"
$DryRunReport = ".\artifacts\experiment_A_v1\a_family_mappo_actual_matrix_runner_dry_run_report.json"
$ExecuteReport = ".\artifacts\experiment_A_v1\a_family_mappo_actual_matrix_runner_execute_report.json"

# Step 58
& $py ".\05_training\policies\h200_actual_checkpoint_preflight_checklist.py" `
  --checkpoint-path $CheckpointPath `
  --mode actual `
  --report-path $PreflightReport

if ($LASTEXITCODE -ne 0) {
    throw "[STOP] Step 58 actual preflight failed"
}

# Step 59
& $py ".\05_training\policies\a_family_mappo_actual_execution_plan.py" `
  --preflight-report $PreflightReport `
  --checkpoint-path $CheckpointPath `
  --output-plan $PlanJson

if ($LASTEXITCODE -ne 0) {
    throw "[STOP] Step 59 actual execution plan failed"
}

# Step 60 dry-run
& $py ".\05_training\policies\run_a_family_mappo_actual_matrix_from_plan.py" `
  --plan-json $PlanJson `
  --report-path $DryRunReport `
  --dry-run

if ($LASTEXITCODE -ne 0) {
    throw "[STOP] Step 60 dry-run failed"
}

# Step 60 execute
& $py ".\05_training\policies\run_a_family_mappo_actual_matrix_from_plan.py" `
  --plan-json $PlanJson `
  --report-path $ExecuteReport `
  --execute

if ($LASTEXITCODE -ne 0) {
    throw "[STOP] Step 60 execute failed"
}
```

---

## 14. Final interpretation rule

If the matrix executes under HistoricalReplayAdapter, write the result as:

```text
A-family actual MAPPO checkpoint execution path validated under non-causal replay.
```

Do not write:

```text
A-family actual MAPPO causally outperformed baselines.
```

Causal comparison requires the future Phase 2 causal simulator adapter.
