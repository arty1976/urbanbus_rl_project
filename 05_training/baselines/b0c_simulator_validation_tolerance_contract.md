# B0C Simulator Validation Tolerance Contract

## 1. Purpose

This contract defines the tolerance rules required before B0C_causal_shadow_v1 can be treated as a simulator-validation candidate.

B0C is not observed historical operation. B0C is a synthetic causal-shadow baseline generated inside the Phase 2 simulator path.

Contract creation does not allow causal comparison claims.

## 2. Scope

| Field | Value |
|---|---|
| baseline_id | B0C_causal_shadow_v1 |
| parent_reference | B0R_historical_12kpi_compat |
| comparison_references | B0R, B1_noop, B2_rulebased_calibrated |
| validation_stage | B0C-5 tolerance contract only |
| next_stage | B0C-6 simulator validation report |

## 3. Required guards

| Guard | Required value |
|---|---|
| causal_comparison_allowed | false |
| paper_level_claim_allowed | false |
| actual_results | false |
| winner_selected | false |
| trainable_reward_promoted | false |
| qwen_trigger_rate | 0.0 |

## 4. Required structural checks

| Check | Rule |
|---|---|
| condition_id | B0C only |
| seed set | 1, 2, 3 |
| window rows | 19,710 |
| windows per seed | 6,570 |
| time bands | peak, offpeak, night |
| 12-KPI schema | all 12 KPI columns present |

## 5. KPI tolerance rules

| KPI | Tolerance rule |
|---|---|
| avg_wait_seconds | Should be within B1/B2 scale band with 30 percent margin |
| cv_headway | Should be within B1/B2 scale band with 40 percent margin |
| bunching_rate | Must be between 0.0 and 0.25 |
| on_time_rate | Must be between 0.60 and 0.95 |
| intervention_rate | Must be exactly 0.0 for B0C |
| passenger_service_rate | Must be between 0.90 and 1.00 |
| passenger_wait_p95_seconds | Must be greater than avg_wait_seconds and not excessively large |
| energy_proxy_per_passenger | Must be finite and greater than 0 |
| fleet_reduction_ratio | Must be exactly 0.0 for B0C |

## 6. Observability restrictions

The following must not be promoted to observed actual values at this stage:

- actual_headway
- actual_arrival_departure_time
- actual_dwell
- passenger-level observed waiting time distribution

These may remain proxy/candidate fields only until separate observed data validation exists.

## 7. Validation outcomes

| Outcome | Meaning |
|---|---|
| PASS | Structural checks and tolerance checks pass, but claim guards remain false |
| PASS_WITH_WARNINGS | Usable as validation candidate, but calibration gaps remain |
| FAIL | Missing files, guard violation, schema failure, or major tolerance violation |

## 8. Promotion policy

Even if B0C-6 returns PASS, causal_comparison_allowed must not be changed to true automatically.

Promotion requires a later explicit release gate, documented operator approval, and separate simulator validation manifest.

## 9. Non-claim statement

B0C validation checks are simulator sanity checks. They are not proof of real-world operational improvement.
