# Step 29: run_causal_rollout.py A-family Bridge Dispatch

## Purpose

This patch adds an explicit bridge dispatch option to:

    05_training/run_causal_rollout.py

The original run_causal_rollout.py behavior remains unchanged unless this option is passed:

    --a-family-bridge

## Usage

Placeholder bridge smoke:

    python .\05_training\run_causal_rollout.py --a-family-bridge --self-test --write-parquet

Scenario-index bridge smoke:

    python .\05_training\run_causal_rollout.py --a-family-bridge --scenario-index .\artifacts\baseline_v1\B1_noop\scenario_index.parquet --limit 8 --write-parquet

Actual MAPPO boundary mode without checking checkpoint existence:

    python .\05_training\run_causal_rollout.py --a-family-bridge --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --limit 8 --write-parquet

Strict actual MAPPO checkpoint existence check:

    python .\05_training\run_causal_rollout.py --a-family-bridge --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --require-existing-checkpoint

## Important

This is still a bridge dispatch, not neural MAPPO inference.

It delegates to:

    05_training/run_causal_rollout_a_family_bridge_v1.py

A/A90/A80/A70 remain Qwen-disabled in this phase.
