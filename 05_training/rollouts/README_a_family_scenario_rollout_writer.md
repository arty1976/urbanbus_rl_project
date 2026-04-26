# A-family Scenario Rollout Writer v1

## Purpose

This writer converts a scenario_index.parquet file into A/A90/A80/A70 extended window_rollup files.

It is a sidecar integration step before modifying run_causal_rollout.py.

## What it creates

Default output directory:

    artifacts/step27_a_family_scenario_rollout

Combined files:

- combined_a_family_window_rollup.csv
- combined_a_family_window_rollup.jsonl
- combined_a_family_window_rollup.parquet, if --write-parquet succeeds
- summary.json
- manifest.json

Condition and seed layout:

    artifacts/step27_a_family_scenario_rollout/A/rollouts/seed_001/window_rollup.parquet
    artifacts/step27_a_family_scenario_rollout/A90/rollouts/seed_001/window_rollup.parquet
    artifacts/step27_a_family_scenario_rollout/A80/rollouts/seed_001/window_rollup.parquet
    artifacts/step27_a_family_scenario_rollout/A70/rollouts/seed_001/window_rollup.parquet

## Rules

- A/A90/A80/A70 use the same passenger_demand_generated for the same window.
- A90/A80/A70 change active_bus_count only.
- qwen_trigger_rate remains 0.0.
- placeholder mode is not valid for paper-level performance claims.
- mappo mode requires checkpoint_path.

## Usage

Self-test using internal sample scenarios:

    python .\05_training\rollouts\run_a_family_scenario_rollout_writer_v1.py --self-test --write-parquet

Scenario-index smoke:

    python .\05_training\rollouts\run_a_family_scenario_rollout_writer_v1.py --scenario-index .\artifacts\baseline_v1\B1_noop\scenario_index.parquet --limit 8 --write-parquet

Actual MAPPO boundary mode without checking checkpoint existence:

    python .\05_training\rollouts\run_a_family_scenario_rollout_writer_v1.py --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --limit 8 --write-parquet
