# Step 121 ??Reward Ablation Execution Manifest

## Purpose

Step 121 converts the Step 119 reward ablation dry-run plan into a stable execution manifest.

This is still not execution. It is the final packaging layer before any sandbox/no-op runner guard.

## Expected Matrix

- Reward candidates: `R0`, `R1`, `R2`, `R3`, `R4`, `R5`
- Conditions: `A`, `A90`, `A80`, `A70`
- Seeds: `1`, `2`, `3`
- Expected rows: `6 횞 4 횞 3 = 72`

## Manifest Fields

Each row must include:

- `execution_manifest_id`
- `execution_order`
- `run_id`
- `run_hash`
- `candidate_id`
- `condition_id`
- `seed`
- `planned_output_root`
- `planned_command_text`
- `command_sha256`
- `manifest_only`
- `execute_allowed`
- `actual_training_allowed`
- `train_with_this_reward_allowed`
- `actual_results`
- `winner_selected`

## Execution Guards

The manifest must preserve:

- `manifest_only = true`
- `execute_allowed = false`
- `actual_training_allowed = false`
- `train_with_this_reward_allowed = false`
- `actual_results = false`
- `winner_selected = false`
- `best_reward_claim_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Forbidden Tokens

Commands must not include:

- `--execute`
- `--train`
- `--promote`
- `--select-winner`
- `--allow-training`
- `--actual-results`
- `--winner-selected`

Commands must include:

- `--dry-run-plan`

## Interpretation

Step 121 produces a reproducible command manifest. It does not run reward ablation, does not select a winner, and does not allow MAPPO training.
