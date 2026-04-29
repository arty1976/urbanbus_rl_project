# Step 116 ??Trainable Reward Promotion Gate

## Purpose

Step 116 defines the gate that must be passed before any reward candidate from Step 111-115 can be promoted to a trainable MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻?? reward.

This step does **not** promote a reward yet.

## Current Decision

- `trainable_reward_promoted = false`
- `train_with_this_reward_allowed = false`
- `selected_candidate_id = null`
- `selected_candidate_claim_allowed = false`

Reason:

No actual reward ablation result exists yet. Step 115 only defined the R0-R5 candidate matrix.

## Required Prior Artifacts

The gate requires these prior artifacts:

1. `final_reward_spec_step111.json`
2. `reward_candidate_protocol_step112.json`
3. `reward_normalization_baseline_lock_step113.json`
4. `hard_constraint_review_step114.json`
5. `reward_ablation_matrix_step115.json`

## Non-Promotable Conditions

A reward cannot be promoted until all of the following become true:

- `actual_ablation_results_exist`
- `selected_candidate_has_best_evidence`
- `selected_candidate_passes_service_floor`
- `selected_candidate_passes_wait_regression_caps`
- `selected_candidate_passes_energy_shortcut_guard`
- `selected_candidate_passes_qwen_off_guard`
- `selected_candidate_passes_observability_guard`
- `human_review_completed`

## Locked Guards

The following guards remain locked:

- `normalization_baseline = B1_noop`
- `numeric_baseline_values_locked = false`
- `hard_constraints_finalized = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`
- `canonical_causal_comparison_allowed = false`
- `actual_headway = not_observed`
- `actual_arrival_departure_time = not_observed`
- `actual_dwell = not_observed`
- `actual_passenger_wait_observed = false`
- `queue_demand_observed = false`
- `queue_demand_proxy = true`

## Service Quality First Rule

A candidate reward must preserve the following:

- Service quality weights dominate energy/fleet weights.
- Long-wait penalty is at least as strong as average-wait penalty.
- Fleet reduction remains a secondary bonus.
- Total `energy_proxy` is not a direct reward term.
- `passenger_service_rate` remains a direct reward candidate.
- `passenger_wait_p95_seconds` remains a direct reward candidate.

## Next Step

Recommended next step:

`Step117 reward ablation result schema and collector`

Step 117 should define how future ablation results are recorded before any candidate can be promoted.
