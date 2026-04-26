# Policy Interface Canonical KPI Smoke v1

## Purpose

This Step 38 smoke verifies that policy-interface rollout rows can be consumed by the canonical KPI aggregator.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.

## Flow

    Step 37 policy-interface scenario writer
    -> A/A90/A80/A70 window_rollup rows
    -> action fields attached
    -> canonical_kpi_aggregator.py official_rollup
    -> condition-level canonical_eval outputs

## Required action fields

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

This is not a performance claim.

Step 38 still uses conservative mock MAPPO actions from the policy interface contract.

The purpose is to verify canonical KPI path compatibility before real neural MAPPO inference is connected.

## Usage

    python ./05_training/evaluation/run_policy_interface_canonical_smoke_v1.py --write-parquet --smoke --self-test
