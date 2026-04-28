# Step 89 ??Toy MAPPO Smoke Checkpoint Contract Validator

## Purpose

Step 89 validates the checkpoint produced by Step 88.

The validator ensures that the artifact is clearly marked as a smoke artifact, not a trained policy and not a paper-level performance result.

## Required Files

```text
training_manifest.json
training_trace.csv
checkpoints/toy_mappo_smoke_checkpoint.pt
```

## Core Checks

The validator checks:

```text
trained_model = false
performance_claim_allowed = false
smoke_training_only = true
causal_comparison_allowed = true
reward_version = mappo_reward_v1
reward_claim_boundary = toy_causal_training_reward_contract_not_paper_performance_claim
```

It also verifies:

```text
12-KPI reward_metric_keys
actor_obs_dim > 0
critic_obs_dim > 0
action_dim = 3
model_state_dict exists
optimizer_state_dict exists
loss and reward summaries are finite
training_trace.csv has finite reward/loss rows
```

## Claim Boundary

Passing this validator does not mean that the checkpoint is a trained MAPPO model.

It only means the Step 88 toy causal smoke checkpoint is structurally valid and safely labeled.

## Usage

```powershell
python 05_training/policies/validate_toy_mappo_smoke_checkpoint.py `
  --checkpoint artifacts/phase2_toy_causal_mappo_smoke_selftest/checkpoints/toy_mappo_smoke_checkpoint.pt `
  --manifest artifacts/phase2_toy_causal_mappo_smoke_selftest/training_manifest.json `
  --trace artifacts/phase2_toy_causal_mappo_smoke_selftest/training_trace.csv
```
