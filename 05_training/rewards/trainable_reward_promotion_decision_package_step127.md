# Step 127 ??Trainable Reward Promotion Decision Package

## Purpose

Step 127 packages the reward-design gates into a formal decision record.

This step does not promote a reward candidate.

## Current Decision

```text
trainable_reward_promoted = false
selected_candidate_id = null
train_with_this_reward_allowed = false
```

## Reason

No actual reward ablation results have been ingested.

The current pipeline has produced templates, no-op guards, dry-run plans, execution manifests, and actual-result preflight gates. These are necessary safeguards, but they are not evidence that any reward candidate is better.

## Promotion Requires

A future promotion decision requires:

1. actual ablation result bundle
2. all 72 R0~R5 횞 A/A90/A80/A70 횞 seed rows
3. actual trained checkpoint evidence
4. canonical KPI outputs
5. hard-constraint verification
6. Step 125 selection criteria pass
7. no template/no-op/dry-run/smoke rows
8. separate paper-level and causal-claim gates

## Guard

Even after this package exists, training remains blocked until a later decision package has actual evidence.
