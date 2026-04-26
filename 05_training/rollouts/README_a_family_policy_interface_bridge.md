# A-family Policy Interface Bridge v1

## Purpose

This Step 36 bridge connects the Step 35 MAPPO policy interface action dict to the existing A-family rollout row.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.

## Flow

    A-family base rollout row
    -> derive observation fields
    -> mappo_policy_interface_v1.run_policy_interface_once
    -> action dict
    -> apply action to rollout row
    -> validate rollout_schema_v1

## What this proves

- policy_source is mappo_policy
- source_mode is causal_*_mappo_policy_v1
- qwen_trigger_rate is 0.0
- action fields are attached:
  - action_version
  - dispatch_delta
  - hold_seconds
  - skip_stop_flag
  - target_headway_ratio
  - policy_debug
- placeholder/stub/smoke markers are rejected

## What this does not prove

This does not run a trained neural MAPPO model.

The current action is the conservative mock action from mappo_policy_interface_v1.
It validates the action contract only.

## Example

    python ./05_training/rollouts/a_family_policy_interface_bridge_v1.py --self-test --condition-id A80
