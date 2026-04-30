# Step 151-A - Explicit operator release manifest draft, still locked

## Purpose

Step 151-A creates an explicit operator release manifest draft after Step 150.

This is not an actual execution release.

## Required input

- Step 150 actual reward ablation operator release checklist manifest:
  artifacts/rewards/actual_reward_ablation_operator_release_checklist_step150/actual_reward_ablation_operator_release_checklist_step150_manifest.json

## Required Step 150 state

- checklist_status = READY_FOR_STEP151_EXPLICIT_OPERATOR_RELEASE_MANIFEST_DRAFT
- decision = CHECKLIST_ONLY_NOT_RELEASED
- hard_failures = 0
- actual_execution_allowed = false
- train_allowed = false

## Step 151-A output

- explicit_operator_release_manifest_draft_step151a_manifest.json
- explicit_operator_release_manifest_draft_step151a_summary.md

## Critical lock

Step 151-A must keep all execution/research claim gates locked:

- actual_execution_allowed = false
- actual_execution_released = false
- train_allowed = false
- actual_results = false
- winner_selected = false
- trainable_reward_promoted = false
- paper_level_claim_allowed = false
- causal_performance_claim_allowed = false

## H200 requirement

Before actual execution, Step 149 must be rerun on the real H200 server with:

```text
--expect-h200 --min-gpu-count 1
```

## Decision

Step 151-A can produce:

```text
RELEASE_MANIFEST_DRAFT_STILL_LOCKED
```

It cannot produce:

```text
actual_execution_allowed = true
actual_execution_released = true
train_allowed = true
```
