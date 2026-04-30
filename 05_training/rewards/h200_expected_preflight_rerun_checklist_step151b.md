# Step 151-B - H200 expected preflight rerun checklist, still locked

## Purpose

Step 151-B prepares the checklist for rerunning Step 149 on the real H200 server.

This step does not execute the H200 rerun.
This step does not release actual reward ablation execution.

## Required input

- Step 151-A explicit operator release manifest draft:
  artifacts/rewards/explicit_operator_release_manifest_draft_step151a/explicit_operator_release_manifest_draft_step151a_manifest.json

## Required Step 151-A state

- release_manifest_status = RELEASE_MANIFEST_DRAFT_STILL_LOCKED
- decision = DRAFT_ONLY_NOT_RELEASED
- hard_failures = 0
- actual_execution_allowed = false
- actual_execution_released = false
- train_allowed = false
- h200_expected_preflight_required = true
- h200_expected_preflight_required_args = --expect-h200 --min-gpu-count 1

## Required H200 command

On the real H200 server, rerun Step 149 with:

```bash
python 05_training/rewards/h200_environment_preflight_result_manifest_step149.py \
  --project-root /workspace/urbanbus_rl_project \
  --expect-h200 \
  --min-gpu-count 1 \
  --output-root artifacts/rewards/h200_environment_preflight_result_manifest_step149_h200_actual
```

## Critical lock

Step 151-B must keep all execution/research claim gates locked:

- actual_execution_allowed = false
- actual_execution_released = false
- train_allowed = false
- actual_results = false
- winner_selected = false
- trainable_reward_promoted = false
- paper_level_claim_allowed = false
- causal_performance_claim_allowed = false
- h200_expected_preflight_rerun_executed = false
- h200_expected_preflight_passed = false

## Decision

Step 151-B can produce:

```text
H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED
```

It cannot produce:

```text
actual_execution_allowed = true
actual_execution_released = true
train_allowed = true
h200_expected_preflight_passed = true
```
