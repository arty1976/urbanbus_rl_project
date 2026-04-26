# A-family Policy Interface Scenario Writer v1

## Purpose

This Step 37 writer applies MAPPO policy interface actions to scenario-index based A-family rollout rows.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.

## Flow

    scenario_index.parquet
    -> seed
    -> A/A90/A80/A70
    -> build base rollout row
    -> derive policy observation
    -> mappo_policy_interface_v1 action
    -> apply action fields to window_rollup row
    -> rollout_schema_v1 validation

## Action fields added

- action_version
- dispatch_delta
- hold_seconds
- skip_stop_flag
- target_headway_ratio
- policy_debug
- policy_action_debug
- policy_interface_version
- performance_claim_allowed

## Important

This does not run a trained neural MAPPO model.

The current action is a conservative mock action from `mappo_policy_interface_v1`.

`performance_claim_allowed` remains false.

## Usage

Self-test:

    python ./05_training/rollouts/run_a_family_policy_interface_scenario_writer_v1.py --self-test --write-parquet

Real scenario-index smoke:

    python ./05_training/rollouts/run_a_family_policy_interface_scenario_writer_v1.py --scenario-index ./artifacts/baseline_v1/B1_noop/scenario_index.parquet --limit 4 --seeds 1 --write-parquet

Use existing checkpoint metadata:

    python ./05_training/rollouts/run_a_family_policy_interface_scenario_writer_v1.py --use-existing-checkpoint --checkpoint-path ./path/to/checkpoint.metadata.json
