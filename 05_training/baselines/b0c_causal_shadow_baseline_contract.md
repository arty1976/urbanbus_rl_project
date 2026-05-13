# B0C Causal Shadow Baseline Contract

## 1. Purpose

B0C is a synthetic causal-shadow baseline for Phase 2 simulator-backed evaluation.

It must not replace B0R historical replay. B0R remains the historical reference baseline, while B0C is used to compare policies inside the same causal simulator environment.

## 2. Baseline identity

| Field | Value |
|---|---|
| condition_id | B0C |
| baseline_id | B0C_causal_shadow_v1 |
| baseline_family | causal_shadow |
| parent_reference | B0R_historical_12kpi_compat |
| policy_source | historical_shadow_policy |
| source_mode_target | causal_B0C_historical_shadow_v1 |
| qwen_train | false |
| qwen_inference | false |
| qwen_trigger_rate | 0.0 |

## 3. Policy definition

B0C uses a no-learned-intervention, full-fleet, historical-shadow policy.

| Parameter | Value |
|---|---|
| active_bus_ratio | 1.0 |
| fleet_reduction_ratio | 0.0 |
| learned_policy | false |
| rule_based_policy | false |
| qwen_assist | false |
| action_mode | maintain_no_learned_intervention |

## 4. Relation to other baselines

| Baseline | Meaning | Claim scope |
|---|---|---|
| B0R_historical_12kpi_compat | Historical replay compatibility baseline | Historical reference, not strict causal evidence |
| B0C_causal_shadow_v1 | Synthetic causal simulator shadow baseline | Simulator-internal comparison baseline after validation |
| B1_noop | No-op replay baseline | Non-causal replay baseline |
| B2_rulebased_calibrated | Calibrated rule-based replay baseline | Non-causal replay baseline |

## 5. 12-KPI schema

B0C must emit all 12 KPI columns.

1. cv_headway
2. avg_wait_seconds
3. bunching_rate
4. on_time_rate
5. intervention_rate
6. energy_proxy
7. passenger_demand_generated
8. passenger_served_count
9. passenger_service_rate
10. passenger_wait_p95_seconds
11. energy_proxy_per_passenger
12. fleet_reduction_ratio

## 6. Required output artifacts

- raw_events.parquet
- window_rollup.parquet
- run_manifest.json
- status.json
- canonical_eval/kpi_by_window.parquet
- canonical_eval/kpi_by_seed.parquet
- canonical_eval/kpi_by_time_band.parquet
- canonical_eval/kpi_overall.json
- canonical_eval/aggregation_manifest.json

## 7. Claim guard

B0C contract creation alone does not allow causal or paper-level claims.

| Guard | Value |
|---|---|
| causal_simulator_target | true |
| causal_comparison_allowed | false until simulator validation gate passes |
| paper_level_claim_allowed | false |
| actual_results | false |
| winner_selected | false |
| trainable_reward_promoted | false |

## 8. Success gates

1. B0C-1 contract validation PASS
2. B0C-2 rollout writer PASS
3. B0C-3 canonical 12-KPI aggregation PASS
4. B0C-4 B0R vs B0C calibration sanity check PASS
5. B0C-5 simulator validation tolerance report PASS

## 9. Non-claim statement

B0C is not observed historical operation. It is a simulator-generated causal-shadow baseline. Any comparison against A-family policies must explicitly state that it is simulator-internal unless separately validated against historical observations.
