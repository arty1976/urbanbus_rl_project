# Step 124 ??Reward Ablation Actual Result Schema Bridge

## Purpose

Step 124 defines the bridge between no-op/template reward ablation outputs and future actual reward ablation result ingestion.

This step does not run reward ablation, does not ingest actual results, and does not select a winner.

## Current Status

- `bridge_status = ACTUAL_RESULT_SCHEMA_BRIDGE_NOT_ACTUAL_RESULTS`
- `actual_results = false`
- `winner_selected = false`
- `train_with_this_reward_allowed = false`
- `actual_training_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Expected Matrix

The future actual result table must cover:

```text
R0/R1/R2/R3/R4/R5
횞 A/A90/A80/A70
횞 seeds 1,2,3
= 72 rows
```

## Required Actual Result Meaning

A row may become an actual result only when all of the following are true:

1. The reward candidate was actually run.
2. The checkpoint is a real trained checkpoint.
3. The checkpoint path and SHA256 are recorded.
4. Canonical KPI outputs exist.
5. Hard-constraint results are recorded.
6. The result is not from no-op, template, mock, smoke-only, or dry-run output.

## Important Guard

This bridge only prepares the shape of future actual result files.

It must not promote any reward candidate and must not allow training.

## Next Step

Step 125 should define winner selection criteria without selecting a winner.
