# MAPPO Reward Contract v1

## Purpose

This document defines the reward contract for UrbanBus MAPPO.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.

The reward must not simply reduce energy use. The policy must first preserve passenger service quality, and only then improve energy and fleet efficiency.

## Input KPI Fields

The reward function expects or recommends the following metrics:

- condition_id
- avg_wait_seconds
- passenger_wait_p95_seconds
- passenger_service_rate
- energy_proxy
- energy_proxy_per_passenger
- on_time_rate
- bunching_rate
- fleet_reduction_ratio
- qwen_trigger_rate

## Reward Components

reward_total is the sum of:

- reward_service
- reward_avg_wait
- reward_long_wait
- reward_on_time
- reward_bunching
- reward_energy
- reward_fleet
- reward_constraint

## Design Rules

### 1. Passenger service quality comes first

If the system reduces buses but fails to serve passengers, the policy is invalid.

Therefore, passenger_service_rate has a strong constraint penalty.

### 2. Average wait and long-wait tail risk are separated

avg_wait_seconds measures average passenger experience.

passenger_wait_p95_seconds measures tail-risk, meaning a small group of passengers may be waiting too long.

A policy with good average wait but poor p95 wait should not be accepted.

### 3. Do not optimize energy_proxy alone

Raw energy_proxy can improve by simply reducing service.

The reward therefore uses energy_proxy_per_passenger as the main energy-efficiency term.

### 4. fleet_reduction_ratio is only a bonus

A90, A80, and A70 reduce active_bus_count.

The reward must not let fleet reduction hide bad passenger service.

### 5. A_pure_mappo must not use Qwen

For condition_id A, qwen_trigger_rate must be exactly 0.0.

If qwen_trigger_rate is nonzero in A, the reward function marks a constraint violation.

## Placeholder vs Actual Policy Boundary

Placeholder results and actual MAPPO inference results must be separated.

Recommended source_mode names:

    stub_A_pure_mappo_smoke
    replay_A_pure_mappo_noncausal
    causal_A_mappo_policy_v1
    causal_A90_mappo_policy_v1
    causal_A80_mappo_policy_v1
    causal_A70_mappo_policy_v1

Only causal_* source_mode results should be used for paper-level performance claims.

## Files

- mappo_reward_v1.py
- reward_config_v1.yaml
- test_mappo_reward_v1.py
