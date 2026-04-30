# Step 150 - Actual reward ablation operator release checklist

## Purpose

Step 150 creates the final human-operator checklist before opening the actual A-family reward ablation run.

This step is **not** an execution release.

## Scope

Step 150 checks that the following upstream gates are available and still guarded:

1. Step 143 A-family 72-run matrix:
   - `A`, `A90`, `A80`, `A70`
   - `R0`, `R1`, `R2`, `R3`, `R4`, `R5`
   - seeds `1`, `2`, `3`
   - total `72` runs

2. Step 141 baseline reference:
   - `B0R`
   - `B1`
   - `B2`

3. Step 144 H200 execution package boundary manifest

4. Step 145 H200 transfer package export manifest

5. Step 146 H200 transfer package integrity verifier

6. Step 147 H200 receive-side transfer runbook

7. Step 148 H200 receive-side preflight / operator handoff gate

8. Step 149 H200 environment preflight result manifest

## Critical guard

Step 150 must keep the following values false:

- `actual_execution_allowed`
- `actual_execution_released`
- `actual_results`
- `winner_selected`
- `trainable_reward_promoted`
- `train_allowed`
- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`

## H200 note

The current Step 149 result may be local record-only.

Before actual H200 execution, Step 149 must be rerun on the real H200 server with:

```text
--expect-h200 --min-gpu-count 1
```

## Decision

Step 150 can produce:

```text
READY_FOR_STEP151_EXPLICIT_OPERATOR_RELEASE_MANIFEST_DRAFT
```

It cannot produce:

```text
actual_execution_allowed = true
```
