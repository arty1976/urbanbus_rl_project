# Fake Checkpoint MAPPO Boundary Smoke v1

## Purpose

This Step 34 smoke validates the actual MAPPO boundary without a trained MAPPO model.

It creates a fake checkpoint file and runs:

    run_causal_rollout.py --a-family-bridge --policy-kind mappo --require-existing-checkpoint

## What this proves

- checkpoint existence validation works
- mappo boundary mode can pass when checkpoint path exists
- policy_source is mappo_policy
- source_mode is causal_*_mappo_policy_v1
- placeholder_policy is not mixed in
- qwen_trigger_rate remains 0.0
- canonical KPI aggregation accepts the outputs

## What this does not prove

- It does not prove MAPPO performance.
- It does not run neural network inference.
- It does not create paper-ready results.

The fake checkpoint is only a boundary validation artifact.

## Default command

    python ./05_training/evaluation/run_fake_checkpoint_mappo_boundary_smoke_v1.py --write-parquet --smoke --self-test
