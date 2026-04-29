# Step 114 ??Hard Constraint Review

## Purpose

Step 114 reviews hard constraint candidates for the final MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻?? reward.

This is not a training approval step.

The goal is to document the service-quality guardrails that must block shortcut behavior before any reward is promoted to a trainable reward.

## Current Status

- `review_status = REVIEWED_NOT_TRAINABLE`
- `train_with_this_reward_allowed = false`
- `hard_constraints_finalized = false`
- `hard_constraint_numeric_values_locked = false`
- `reward_weights_locked = false`
- `reward_formula_finalized = false`

## Baseline Position

Step 113 locked the normalization baseline identity to:

```text
B1_noop
```

But Step 113 did not lock numeric baseline values.

Therefore, Step 114 can review threshold rules such as:

```text
avg_wait_seconds <= B1_noop_avg_wait_seconds * 1.10
```

but the actual numeric threshold is still not locked until the empirical baseline table is generated and approved.

## Core Principle

Hard constraints exist to prevent reward hacking.

The reward must not learn these shortcuts:

1. Reduce active buses and strand passengers.
2. Lower energy by suppressing service.
3. Improve average wait while harming long-wait passengers.
4. Treat ETA-based headway as actual observed headway.
5. Treat queue/demand proxy as directly observed passenger queue.

## Reviewed Constraint Candidates

### 1. `service_rate_floor`

```text
passenger_service_rate >= 0.95
```

This prevents a policy from saving energy or reducing fleet while failing to serve passengers.

### 2. `avg_wait_regression_cap`

```text
avg_wait_seconds <= B1_noop_avg_wait_seconds * 1.10
```

This blocks policies that make average waiting time more than 10 percent worse than the no-op baseline candidate.

### 3. `p95_wait_regression_cap`

```text
passenger_wait_p95_seconds <= B1_noop_p95_wait_seconds * 1.15
```

This protects passengers who experience long waits. It prevents average-only optimization.

### 4. `energy_service_coupling_guard`

Energy improvement can only count if service quality constraints pass.

This blocks the shortcut where energy appears better because the policy served fewer passengers.

### 5. `fleet_reduction_service_guard`

Fleet reduction bonus can only count if service quality constraints pass.

This keeps fleet reduction as a secondary bonus, not the main objective.

### 6. `qwen_off_for_A_family`

```text
qwen_trigger_rate == 0.0
```

A/A90/A80/A70 pure MAPPO experiments must not include Qwen intervention.

### 7. `observability_overclaim_guard`

The following remain blocked:

- `actual_headway = not_observed`
- `actual_arrival_departure_time = not_observed`
- `actual_dwell = not_observed`
- `actual_passenger_wait_observed = false`
- `queue_demand_observed = false`
- `queue_demand_proxy = true`

### 8. `finite_reward_guard`

All reward components and `reward_total` must be finite.

NaN or infinite values must hard-fail.

## Still Not Allowed

- Do not train MAPPO with these constraints yet.
- Do not claim these thresholds are empirically final.
- Do not report scaffold reward outputs as paper-level performance.
- Do not treat ETA-based headway as actual observed headway.
- Do not treat queue/demand proxy as directly observed queue demand.

## Next Gates

Before training is allowed:

1. Step 115 ??reward ablation matrix
2. Step 116 ??trainable reward promotion gate
3. empirical baseline numeric value table
4. constraint stress test against shortcut policies
