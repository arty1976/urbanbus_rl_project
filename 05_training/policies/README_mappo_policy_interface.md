# MAPPO Policy Interface v1

## Purpose

This document defines the future actual MAPPO inference interface.

MAPPO means Multi-Agent Proximal Policy Optimization.
GAT means Graph Attention Network.
KPI means Key Performance Indicator.

Step 35 does not implement neural network inference.
It defines the contract that a future trained checkpoint must satisfy.

## Inference Flow

    checkpoint metadata
    -> observation dict
    -> MAPPO policy inference
    -> action dict
    -> causal simulator adapter
    -> window_rollup
    -> canonical KPI

## Required Checkpoint Metadata

- artifact_type
- interface_version
- policy_kind
- trained_model
- action_space_version
- observation_space_version

Expected values:

- artifact_type: urbanbus_mappo_checkpoint
- interface_version: mappo_policy_interface_v1
- policy_kind: mappo
- trained_model: true
- action_space_version: bus_control_action_v1
- observation_space_version: urbanbus_observation_v1

## Required Observation Fields

- condition_id
- window_id
- state_ts
- time_band
- seed
- baseline_bus_count
- active_bus_count
- passenger_demand_generated
- headway_mean_seconds
- headway_std_seconds
- bunching_rate
- on_time_rate
- avg_wait_seconds
- passenger_wait_p95_seconds
- energy_proxy_per_passenger

## Required Action Fields

- action_version
- condition_id
- policy_source
- source_mode
- qwen_trigger_rate
- dispatch_delta
- hold_seconds
- skip_stop_flag
- target_headway_ratio
- active_bus_count
- policy_debug

## Current Status

The current implementation includes a conservative mock action only to validate the interface shape.

It is not a learned MAPPO action.
It is not valid for performance claims.

The future implementation should replace `conservative_mock_action` with actual neural MAPPO inference.
