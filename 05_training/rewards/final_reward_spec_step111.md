# Step 111 ??Final Reward Specification Draft

## Purpose

Step 111 drafts the final reward specification for MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻??.

This is not a training approval step. It only defines a candidate reward direction while preserving all Step 110 guards.

## Current Status

- `spec_status = DRAFT_NOT_TRAINABLE`
- `train_with_this_reward_allowed = false`
- `reward_formula_finalized = false`
- `reward_weights_locked = false`
- `normalization_locked = false`
- `hard_constraints_locked = false`
- `final_reward_design_claim_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Weight Sign Convention

All reward weights are stored as positive magnitudes.

Good terms use:

```text
+ weight * score
```

Bad terms use:

```text
- weight * penalty
```

This avoids sign confusion such as `-(-2.0) * penalty`.

## Core Principle

The reward must optimize service quality first.

Energy and fleet reduction are secondary objectives. They must never become shortcuts that reduce service while making energy appear better.

## Canonical 12-KPI Set

KPI (Key Performance Indicator=?듭떖 ?깃낵 吏?? list:

1. `cv_headway`
2. `avg_wait_seconds`
3. `bunching_rate`
4. `on_time_rate`
5. `intervention_rate`
6. `energy_proxy`
7. `passenger_demand_generated`
8. `passenger_served_count`
9. `passenger_service_rate`
10. `passenger_wait_p95_seconds`
11. `energy_proxy_per_passenger`
12. `fleet_reduction_ratio`

## Draft Reward Formula

```text
reward_t =
  + w_service * service_rate_score
  - w_avg_wait * avg_wait_penalty
  - w_long_wait * long_wait_penalty
  + w_ontime * on_time_score
  - w_bunching * bunching_penalty
  - w_headway_cv * headway_cv_penalty
  - w_energy_per_passenger * energy_per_passenger_penalty
  + w_fleet_reduction * fleet_reduction_bonus
  - w_intervention * intervention_penalty
  - w_constraint * constraint_violation_penalty
```

This formula is a draft only.

## Draft Weight Candidates

These values are not locked.

| Term | Draft Weight | Role |
|---|---:|---|
| `w_service` | `3.0` | Primary service quality |
| `w_avg_wait` | `2.0` | Average wait penalty |
| `w_long_wait` | `3.0` | Long-wait fairness |
| `w_ontime` | `1.0` | Schedule reliability |
| `w_bunching` | `1.5` | Anti-bunching |
| `w_headway_cv` | `1.0` | Headway regularity |
| `w_energy_per_passenger` | `1.0` | Secondary energy efficiency |
| `w_fleet_reduction` | `0.5` | Secondary fleet reduction bonus |
| `w_intervention` | `0.25` | Over-control penalty |
| `w_constraint` | `10.0` | Hard-constraint violation |

## Direct Reward Candidate Terms

The reward may directly use:

- `passenger_service_rate`
- `avg_wait_seconds`
- `passenger_wait_p95_seconds`
- `on_time_rate`
- `bunching_rate`
- `cv_headway`
- `energy_proxy_per_passenger`
- `fleet_reduction_ratio`
- `intervention_rate`

## Evaluation or Context Only Terms

These are preserved, but should not directly become reward terms:

- `passenger_demand_generated`
- `passenger_served_count`
- `energy_proxy`

Reason:

Raw total energy can be reduced by reducing service. Therefore, the reward should prefer `energy_proxy_per_passenger`.

Raw passenger demand and served count are required for auditing and denominator checks, but `passenger_service_rate` is safer as a direct reward term.

## Draft Normalization

Not locked:

```text
avg_wait_penalty =
  max(0, avg_wait_seconds / baseline_avg_wait_seconds - 1)

long_wait_penalty =
  max(0, passenger_wait_p95_seconds / baseline_p95_wait_seconds - 1)

energy_per_passenger_penalty =
  energy_proxy_per_passenger / baseline_energy_proxy_per_passenger

service_rate_score =
  passenger_service_rate

fleet_reduction_bonus =
  fleet_reduction_ratio
```

Candidate training reward baseline:

```text
B1_noop
```

Candidate reporting baselines:

```text
B0_historical
B1_noop
B2_rulebased
```

## Draft Hard Constraints

Not locked:

1. `passenger_service_rate >= 0.95`
2. `avg_wait_seconds <= baseline_avg_wait_seconds * 1.10`
3. `passenger_wait_p95_seconds <= baseline_p95_wait_seconds * 1.15`
4. `qwen_trigger_rate == 0.0` for A-family experiments
5. No overclaiming of actual headway, arrival/departure time, dwell, or passenger wait observation

## Observability Guard

Still blocked:

- `actual_headway = not_observed`
- `actual_arrival_departure_time = not_observed`
- `actual_dwell = not_observed`
- `actual_passenger_wait_observed = false`
- `queue_demand_observed = false`
- `queue_demand_proxy = true`

## Required Next Gates

Before training is allowed:

1. Step 112 ??reward candidate protocol
2. Step 113 ??reward normalization baseline lock
3. Step 114 ??hard constraint review
4. Step 115 ??reward ablation matrix
5. Step 116 ??trainable reward promotion gate
