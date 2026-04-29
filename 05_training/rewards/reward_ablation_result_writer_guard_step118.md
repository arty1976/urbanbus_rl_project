# Step 118 ??Reward Ablation Result Writer / Runner Guard

## Purpose

Step 118 creates the result-file writer and guard for future reward ablation evaluation.

It does not run MAPPO training.
It does not evaluate R0-R5.
It does not select a winner.
It does not promote a trainable reward.

## Inputs

Required upstream artifacts:

- `05_training/rewards/reward_ablation_matrix_step115.json`
- `05_training/rewards/reward_ablation_result_schema_step117.json`

## Outputs

The writer creates template-only result files:

1. `reward_ablation_results_by_candidate_seed_condition_template.csv`
2. `reward_ablation_results_by_candidate_template.csv`
3. `reward_ablation_hard_constraint_violations_template.csv`
4. `reward_ablation_selection_summary_template.json`
5. `reward_ablation_result_writer_guard_manifest.json`

## Candidate Coverage

The result template must include all six reward candidates:

- R0
- R1
- R2
- R3
- R4
- R5

## Condition and Seed Coverage

The detail result template is shaped as:

```text
6 candidates x 4 conditions x 3 seeds = 72 rows
```

A-family conditions:

- A
- A90
- A80
- A70

Seeds:

- 1
- 2
- 3

## Required Guard Status

The following must remain false:

- `actual_results`
- `winner_selected`
- `best_claim_allowed`
- `train_with_this_reward_allowed`
- `trainable_reward_promoted`

## Why This Step Exists

Step 117 defined the result schema.
Step 118 creates a concrete writer/template and validates that future result files have the required shape.

This prevents later reward ablation results from being written in inconsistent formats.

## Next Step

Step 119 should define a dry-run runner plan for producing actual ablation result files later.

Actual training and reward promotion remain blocked.
