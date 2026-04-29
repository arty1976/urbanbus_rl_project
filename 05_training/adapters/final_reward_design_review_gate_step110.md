# Step 110 — Final Reward Design Review Gate

## Purpose

Step 110 prevents the Step 108/109 scaffold reward from being mistaken for the final MAPPO training reward.

The current reward path proves that 12 KPI values can be transformed into reward-like components and exposed to a policy-facing interface. It does not prove that the reward is final, trainable, causal, or paper-level.

## Current Required Status

The gate must preserve the following status:

- `reward_scaffold_only = true`
- `reward_weights_are_final = false`
- `reward_formula_finalized = false`
- `train_with_this_reward_allowed = false`
- `final_reward_design_claim_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`
- `canonical_causal_comparison_allowed = false`

## Observability Guard

The following fields must remain blocked:

- `actual_headway = not_observed`
- `actual_arrival_departure_time = not_observed`
- `actual_dwell = not_observed`
- `actual_passenger_wait_observed = false`
- `queue_demand_observed = false`
- `queue_demand_proxy = true`

## Reward Design Principle

The final reward must follow this priority:

1. Service quality first.
2. Energy and fleet reduction second.
3. Energy proxy must not reward service collapse.
4. Fleet reduction must remain a secondary bonus.
5. Long-wait penalty must protect passengers who are harmed by average-only optimization.

## Terms Requiring Final Review

Before training is allowed, these must be locked:

- `passenger_service_rate`
- `avg_wait_seconds`
- `passenger_wait_p95_seconds`
- `energy_proxy_per_passenger`
- `fleet_reduction_ratio`
- baseline normalization reference
- hard constraints
- reward weights
- reward ablation protocol

## Promotion Conditions

`train_with_this_reward_allowed` can become true only after all of the following are documented:

1. Reward weights are reviewed and locked.
2. Normalization baseline is selected and locked.
3. Hard constraints are reviewed and locked.
4. Reward ablation protocol exists.
5. Actual observation/proxy status is explicitly preserved.
6. The reward is tested against shortcut behavior such as reducing buses while stranding passengers.
7. The policy path proves no fallback to placeholder or scaffold result.

## Prohibited Until Next Gate

- Do not train MAPPO with Step 108/109 scaffold reward.
- Do not use `reward_total` as a performance metric.
- Do not claim A/A90/A80/A70 superiority.
- Do not include scaffold reward outputs in a paper-level performance table.
