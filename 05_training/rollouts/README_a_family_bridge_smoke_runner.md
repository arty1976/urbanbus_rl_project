# A-family Bridge Smoke Runner v1

## Purpose

This runner verifies the Step 25 A-family rollout bridge before modifying run_causal_rollout.py.

It creates one extended rollout row for each A-family condition:

- A
- A90
- A80
- A70

## What it checks

- Same passenger_demand_generated across A/A90/A80/A70
- Correct fleet reduction ratio
- qwen_trigger_rate equal to 0.0
- policy_source and source_mode attached
- reward fields attached
- rollout_schema_v1 validation passes

## Output files

Default output directory:

    artifacts/step26_a_family_bridge_smoke

Generated files:

- a_family_rows.jsonl
- a_family_rows.csv
- summary.json
- manifest.json
- a_family_rows.parquet, if --write-parquet succeeds

## Usage

Placeholder smoke:

    python .\05_training\rollouts\run_a_family_bridge_smoke_v1.py --self-test --write-parquet

Actual MAPPO boundary smoke, without requiring real checkpoint existence:

    python .\05_training\rollouts\run_a_family_bridge_smoke_v1.py --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --write-parquet

Actual MAPPO strict checkpoint existence check:

    python .\05_training\rollouts\run_a_family_bridge_smoke_v1.py --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --require-existing-checkpoint
