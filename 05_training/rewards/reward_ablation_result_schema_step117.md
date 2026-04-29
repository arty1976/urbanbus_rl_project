# Step 117 ??Reward Ablation Result Schema

## Purpose

Step 117 defines the result schema for future comparison of R0-R5 reward candidates.

This step does **not** run reward ablation.  
This step does **not** select the best reward.  
This step does **not** allow MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻?? training.

## Current Status

- `schema_status = RESULT_SCHEMA_DRAFT_NOT_ACTUAL_RESULT`
- `actual_ablation_results_available = false`
- `reward_winner_selected = false`
- `best_reward_claim_allowed = false`
- `train_with_this_reward_allowed = false`
- `reward_promotion_allowed = false`
- `performance_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Candidate Set

The schema expects six candidate reward configurations:

| Candidate | Meaning |
|---|---|
| `R0` | service-only safety baseline candidate |
| `R1` | balanced default candidate |
| `R2` | wait-heavy candidate |
| `R3` | long-wait fairness-heavy candidate |
| `R4` | energy-light candidate |
| `R5` | fleet-reduction cautious candidate |

## Canonical 12-KPI Columns

KPI (Key Performance Indicator=?듭떖 ?깃낵 吏?? columns expected in future result tables:

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

## Main Result Table

Expected future file:

```text
reward_ablation_result_by_candidate_seed_condition.parquet
```

Required meaning:

```text
one row = one reward candidate 횞 one condition 횞 one seed
```

This table must include:

- candidate identity
- condition identity
- seed
- adapter/source mode
- scenario/window counts
- 12-KPI means
- service quality summary score
- secondary efficiency summary score
- hard constraint violation count
- claim guard columns

## Overall Candidate Table

Expected future file:

```text
reward_ablation_result_overall_by_candidate.parquet
```

Required meaning:

```text
one row = one reward candidate aggregated across valid condition-seed pairs
```

This table can support later review, but it cannot select a winner by itself.

## Hard Constraint Ledger

Expected future file:

```text
hard_constraint_violations_by_candidate.parquet
```

This ledger is mandatory because a candidate with good average KPI values can still be unsafe if it violates service constraints.

Examples:

- service rate floor violation
- average wait regression cap violation
- p95 wait regression cap violation
- Qwen trigger violation in A-family experiment
- observed/proxy overclaim violation

## Draft Ranking Policy

The draft ranking order is:

1. lower hard constraint violation count
2. higher service-quality score
3. lower p95 wait
4. lower energy per passenger

This ranking policy is not locked and cannot select a winner yet.

## Promotion Rule

A reward candidate cannot be promoted unless:

1. all R0-R5 candidates are evaluated on the same scenarios, seeds, conditions, and windows;
2. the canonical 12-KPI outputs are complete;
3. hard constraint violations are fully logged;
4. no candidate is promoted from smoke-only or scaffold-only outputs;
5. service quality dominates energy/fleet efficiency in the decision.

## Prohibited Now

- Do not select a best reward candidate.
- Do not train MAPPO with any candidate based only on this schema.
- Do not claim reward performance.
- Do not claim causal improvement.
- Do not include this schema as an empirical result table.

## Next Step

Recommended next step:

```text
Step118 reward ablation result writer or runner guard
```
