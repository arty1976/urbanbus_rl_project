# Rollout Schema Contract v1

## Purpose

This document defines the extended window_rollup schema for UrbanBus MAPPO rollouts.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.

Step 22 does not replace the actual policy yet.
Step 22 fixes the schema boundary that future A/A90/A80/A70 actual MAPPO rollout writers must follow.

## Main Rule

B0R, B1, B2, A, A90, A80, and A70 must use the same passenger_demand_generated for the same window.

A90, A80, and A70 reduce active_bus_count only.
They must not regenerate easier demand.

## Core Columns

- condition_id
- seed
- window_id
- state_ts
- service_date
- time_band
- evaluation_horizon_minutes

## Official Rollup Raw Columns

- headway_mean_seconds
- headway_std_seconds
- headway_sample_count
- bunching_event_count
- headway_event_count
- wait_total_passenger_seconds
- wait_passenger_count
- ontime_event_count
- schedulable_arrival_count
- intervention_count
- decision_step_count
- energy_proxy_total

## Extended Passenger Columns

- passenger_demand_generated
- passenger_served_count
- passenger_service_rate
- passenger_wait_p95_seconds
- long_wait_passenger_count

## Extended Energy and Fleet Columns

- energy_proxy
- energy_proxy_per_passenger
- active_bus_count
- baseline_bus_count
- fleet_reduction_ratio

## Policy Provenance Columns

- policy_source
- policy_checkpoint_path
- source_mode
- qwen_trigger_rate
- effective_replay_step_minutes

## Reward Columns

- reward_total
- reward_service
- reward_avg_wait
- reward_long_wait
- reward_on_time
- reward_bunching
- reward_energy
- reward_fleet
- reward_constraint

## Source Mode Rule

Placeholder or non-causal outputs must not be used for paper-level performance claims.

Examples that are not causal claims:

    stub_A_pure_mappo_smoke
    replay_A_pure_mappo_noncausal
    placeholder_A_smoke

Examples that can be causal claims after Phase 2 causal simulator is implemented:

    causal_A_mappo_policy_v1
    causal_A90_mappo_policy_v1
    causal_A80_mappo_policy_v1
    causal_A70_mappo_policy_v1

## Qwen Rule

A, A90, A80, and A70 must have qwen_trigger_rate equal to 0.0 in the current phase.

Qwen intervention will be tested later in separate B/C/D-style conditions, not in A_pure_mappo.
