# Step 119 ??Reward Ablation Runner Dry-Run Plan

## Purpose

Step 119 creates a dry-run-only run plan for reward ablation candidates.

It does not run MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻?? training, does not generate actual reward ablation results, and does not select a best reward.

## Planned Grid

The dry-run plan covers:

- reward candidates: `R0`, `R1`, `R2`, `R3`, `R4`, `R5`
- conditions: `A`, `A90`, `A80`, `A70`
- seeds: `1`, `2`, `3`

Total planned run rows:

```text
6 candidates 횞 4 conditions 횞 3 seeds = 72 planned rows
```

## Execution Guards

The following must remain false:

- `execute_allowed`
- `actual_training_allowed`
- `train_with_this_reward_allowed`
- `actual_results`
- `winner_selected`
- `best_reward_claim_allowed`
- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`

## Command Rule

Every planned command must be review-only or dry-run-only.

Required:

```text
--dry-run-plan
```

Forbidden:

```text
--execute
--train
--promote
--select-winner
```

## Meaning

Step 119 prepares the runner grid so that later steps can decide how to execute reward candidate comparison safely. It is still not a training approval step.

## Next Step

Recommended next step:

```text
Step120 reward ablation runner guard / execution preflight
```
