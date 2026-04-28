# Phase 2 Toy Causal Train/Eval Runbook

## Purpose

This runbook freezes the current Phase 2 toy causal training and evaluation smoke pipeline.

It covers Steps 86 through 91:

```text
Step 86: 12-KPI MAPPO reward contract
Step 87: CausalSimulatorAdapter reward_v1 wiring
Step 88: Toy causal MAPPO training smoke scaffold
Step 89: Toy MAPPO smoke checkpoint validator
Step 91: Toy MAPPO smoke checkpoint evaluator
```

This runbook is not a performance report.

It is a structural reproducibility guide.

## Non-Claim Boundary

All artifacts produced by this pipeline must retain the following flags:

```text
trained_model = false
performance_claim_allowed = false
smoke_training_only = true       # training smoke artifacts
smoke_evaluation_only = true     # evaluation smoke artifacts
```

The current pipeline must not be used to claim:

```text
actual MAPPO policy convergence
Daegu-wide operational improvement
paper-level performance result
H200 actual training result
```

The only allowed claim is:

```text
The Phase 2 toy causal train/eval smoke pipeline is structurally wired and reproducible.
```

## 12-KPI Schema

The Phase 2 12-KPI schema is:

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

These KPIs are propagated through:

```text
CausalSimulatorAdapter.step().info.reward_metrics
window_rollup.parquet
canonical KPI aggregation
inspection report
training manifest
checkpoint validator
evaluation manifest
```

## Step 86 ??Reward Contract

Files:

```text
05_training/rewards/mappo_reward_v1.py
05_training/rewards/reward_config_v1.yaml
05_training/rewards/README_reward_contract.md
05_training/rewards/test_mappo_reward_v1.py
```

Validation:

```powershell
python 05_training/rewards/test_mappo_reward_v1.py
```

Core rules:

```text
service quality is primary
p95 wait is penalized strongly
energy_proxy_per_passenger is used instead of raw energy alone
fleet_reduction_ratio is a secondary bonus
fleet bonus is disabled when passenger_service_rate is below the floor
```

## Step 87 ??Adapter Reward Wiring

Files:

```text
05_training/adapters/causal_simulator_adapter.py
05_training/adapters/test_causal_simulator_reward_v1_integration.py
```

Validation:

```powershell
python 05_training/adapters/test_causal_simulator_reward_v1_integration.py
```

Expected StepResult.info keys:

```text
reward_version
reward_claim_boundary
reward_total
reward_components
reward_debug
reward_metrics
```

Expected reward behavior:

```text
team-level reward_total is broadcast to all agents
same seed and same action sequence gives deterministic reward_total
all-hold policy is worse than dispatch-dominant policy in smoke test
```

## Step 88 ??Toy MAPPO Training Smoke

Files:

```text
05_training/train_toy_causal_mappo_smoke.py
05_training/adapters/test_toy_causal_mappo_training_smoke.py
05_training/adapters/toy_causal_mappo_training_smoke.md
```

Command:

```powershell
python 05_training/adapters/test_toy_causal_mappo_training_smoke.py
```

Expected outputs:

```text
artifacts/phase2_toy_causal_mappo_smoke_selftest/
  training_manifest.json
  training_trace.csv
  checkpoints/toy_mappo_smoke_checkpoint.pt
```

Expected flags:

```text
trained_model = false
performance_claim_allowed = false
smoke_training_only = true
```

## Step 89 ??Smoke Checkpoint Validator

Files:

```text
05_training/policies/validate_toy_mappo_smoke_checkpoint.py
05_training/policies/test_toy_mappo_smoke_checkpoint_validator.py
05_training/policies/toy_mappo_smoke_checkpoint_contract.md
```

Command:

```powershell
python 05_training/policies/test_toy_mappo_smoke_checkpoint_validator.py
```

Validator confirms:

```text
checkpoint has required keys
manifest/checkpoint metadata are consistent
12-KPI reward_metric_keys exist
loss/reward/trace values are finite
performance_claim_allowed=true is rejected
```

## Step 91 ??Smoke Checkpoint Evaluation

Files:

```text
05_training/evaluation/evaluate_toy_mappo_smoke_checkpoint.py
05_training/evaluation/test_evaluate_toy_mappo_smoke_checkpoint.py
05_training/evaluation/toy_mappo_smoke_checkpoint_evaluation.md
```

Command:

```powershell
python 05_training/evaluation/test_evaluate_toy_mappo_smoke_checkpoint.py
```

Expected outputs:

```text
artifacts/phase2_toy_mappo_smoke_checkpoint_eval_selftest/
  evaluation_manifest.json
  rollouts/
  canonical_eval/
  inspection/
```

Expected flags:

```text
trained_model = false
performance_claim_allowed = false
smoke_evaluation_only = true
```

## Full Local Smoke Validation Sequence

Run from project root:

```powershell
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py .\05_training\rewards\test_mappo_reward_v1.py
& $py .\05_training\adapters\test_causal_simulator_reward_v1_integration.py
& $py .\05_training\adapters\test_toy_causal_mappo_training_smoke.py
& $py .\05_training\policies\test_toy_mappo_smoke_checkpoint_validator.py
& $py .\05_training\evaluation\test_evaluate_toy_mappo_smoke_checkpoint.py
```

## Current Gate Status

The current gate is passed when all of the following are true:

```text
reward_v1 self-test PASS
adapter reward integration PASS
toy MAPPO training smoke PASS
checkpoint validator PASS
smoke checkpoint evaluator PASS
all artifacts keep non-claim flags
```

## Recommended Next Step

Step 93 should extend the smoke evaluator from single condition `A` to a toy A-family matrix:

```text
A / A90 / A80 / A70
```

The next matrix must still keep:

```text
performance_claim_allowed = false
smoke_evaluation_only = true
```

It can compare smoke outputs structurally, but it must not be interpreted as paper-level performance.

## Step 93 ??Toy MAPPO Smoke Matrix Evaluation

Files:

```text
05_training/evaluation/evaluate_toy_mappo_smoke_matrix.py
05_training/evaluation/test_evaluate_toy_mappo_smoke_matrix.py
05_training/evaluation/toy_mappo_smoke_matrix_evaluation.md
```

Command:

```powershell
python 05_training/evaluation/test_evaluate_toy_mappo_smoke_matrix.py
```

Expected outputs:

```text
artifacts/phase2_toy_mappo_smoke_matrix_eval_selftest/
  matrix_evaluation_manifest.json
  matrix_summary.csv
  matrix_training/
  conditions/
    A/
    A90/
    A80/
    A70/
  matrix_canonical_eval/
  matrix_inspection/
```

Expected flags:

```text
trained_model = false
performance_claim_allowed = false
smoke_evaluation_only = true
matrix_smoke_evaluation_only = true
```

Expected matrix validation:

```text
conditions = A/A90/A80/A70
kpi_by_window_rows = 12
condition_summary rows = 4
condition_vs_A_delta rows = 36
12-KPI schema validated
```

Important implementation detail:

```text
Condition-level Step 91 evaluation uses --skip-inspector.
The final matrix-level combined canonical output runs the A-family inspector.
```

This avoids trying to run A-family delta inspection on a single non-baseline condition such as A90.

## Updated Full Local Smoke Validation Sequence

Run from project root:

```powershell
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py .\05_training\rewards\test_mappo_reward_v1.py
& $py .\05_training\adapters\test_causal_simulator_reward_v1_integration.py
& $py .\05_training\adapters\test_toy_causal_mappo_training_smoke.py
& $py .\05_training\policies\test_toy_mappo_smoke_checkpoint_validator.py
& $py .\05_training\evaluation\test_evaluate_toy_mappo_smoke_checkpoint.py
& $py .\05_training\evaluation\test_evaluate_toy_mappo_smoke_matrix.py
```

## Updated Current Gate Status

The current Phase 2 toy causal smoke gate is passed when all of the following are true:

```text
reward_v1 self-test PASS
adapter reward integration PASS
toy MAPPO training smoke PASS
checkpoint validator PASS
single-condition smoke checkpoint evaluator PASS
A-family smoke matrix evaluator PASS
all artifacts keep non-claim flags
```

## Updated Recommended Next Step

Step 95 should produce one of the following:

```text
Phase 2 toy causal smoke pipeline final status report
```

or

```text
Toy causal simulator realism gap analysis
```

The final status report is recommended first, because Steps 77 through 93 now form a complete structural smoke pipeline.

