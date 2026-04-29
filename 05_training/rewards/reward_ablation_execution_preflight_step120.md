# Step 120 ??Reward Ablation Execution Preflight

## Purpose

Step 120 checks the Step 119 reward ablation dry-run plan before any execution path is allowed.

This step is still not an execution step. It does not run MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻??, does not train a policy, does not evaluate a winner, and does not promote a reward.

## Expected Plan Shape

The preflight expects:

```text
R0~R5 reward candidates 6
횞 A/A90/A80/A70 conditions 4
횞 seeds 1,2,3
= 72 planned dry-run rows
```

## Required Guards

All rows must preserve:

- `execute_allowed = false`
- `actual_training_allowed = false`
- `train_with_this_reward_allowed = false`
- `actual_results = false`
- `winner_selected = false`
- `best_reward_claim_allowed = false`

## Command Safety

Every planned command must include:

- `--dry-run-plan`

No planned command may include:

- `--execute`
- `--train`
- `--promote`
- `--select-winner`
- `--allow-actual-results`

## Interpretation

A PASS means the dry-run plan is complete and still non-executable.

A PASS does not mean the reward ablation has run. It only means the plan is safe enough to be reviewed by the next gate.
