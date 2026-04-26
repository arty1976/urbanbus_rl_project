# Root A-family Rollout Bridge Entrypoint v1

## Purpose

This file provides a root-level entrypoint before modifying run_causal_rollout.py.

It delegates to:

    05_training/rollouts/run_a_family_scenario_rollout_writer_v1.py

## Why this exists

Step 27 proved that the A-family scenario rollout writer can create valid A/A90/A80/A70 window_rollup outputs.

Step 28 moves that workflow to the 05_training root level so it can later be connected into run_causal_rollout.py.

## Conditions

- A
- A90
- A80
- A70

## Modes

Placeholder mode:

    python .\05_training\run_causal_rollout_a_family_bridge_v1.py --self-test --write-parquet

Scenario-index placeholder mode:

    python .\05_training\run_causal_rollout_a_family_bridge_v1.py --scenario-index .\artifacts\baseline_v1\B1_noop\scenario_index.parquet --limit 8 --write-parquet

Actual MAPPO boundary mode without checkpoint existence check:

    python .\05_training\run_causal_rollout_a_family_bridge_v1.py --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --limit 8 --write-parquet

Actual MAPPO strict mode:

    python .\05_training\run_causal_rollout_a_family_bridge_v1.py --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --require-existing-checkpoint

## Important

This does not implement neural MAPPO inference yet.

It verifies the contract boundary:

- reward contract
- rollout schema
- policy registry
- policy inference boundary
- A-family bridge
- scenario-index writer
