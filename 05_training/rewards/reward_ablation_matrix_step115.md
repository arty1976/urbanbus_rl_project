# Step 115 ??Reward Ablation Matrix

## Purpose

Step 115 defines a reward ablation matrix for MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻??.

This step does not choose the final reward and does not allow training. It only defines safe candidate reward weight sets that can later be compared under the same baseline, seeds, windows, and hard constraints.

## Current Status

- `matrix_status = DRAFT_NOT_TRAINABLE`
- `train_with_any_candidate_allowed = false`
- `best_reward_claim_allowed = false`
- `reward_formula_finalized = false`
- `reward_weights_locked = false`
- `hard_constraints_finalized = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Shared Baseline

The normalization baseline identity is locked to:

```text
B1_noop
```

However, numerical baseline values are not locked in Step 115.

## Shared Hard Constraints

All candidates must share the same hard constraints:

1. `passenger_service_rate >= 0.95`
2. `avg_wait_seconds <= baseline_avg_wait_seconds * 1.10`
3. `passenger_wait_p95_seconds <= baseline_p95_wait_seconds * 1.15`
4. `qwen_trigger_rate == 0.0`
5. No overclaiming of actual headway, arrival/departure time, dwell, or passenger wait observation

## Weight Sign Convention

All weights are stored as positive magnitudes.

Good terms use:

```text
+ weight * score
```

Bad terms use:

```text
- weight * penalty
```

## Candidate Matrix

| Candidate | Intent | Main Change |
|---|---|---|
| `R0_SERVICE_SAFETY` | Service-only safety baseline | Energy/fleet pressure disabled |
| `R1_BALANCED_DEFAULT` | Balanced default | Step 111 default |
| `R2_WAIT_HEAVY` | Average-wait heavy | Stronger average wait penalty |
| `R3_LONG_WAIT_FAIRNESS` | Long-wait fairness heavy | Stronger p95 wait penalty |
| `R4_ENERGY_LIGHT` | Energy-aware but service-first | Slightly stronger energy-per-passenger penalty |
| `R5_FLEET_CAUTIOUS` | Fleet-reduction cautious | Modest fleet bonus, still service-first |

## Candidate Weights

| Candidate | w_service | w_avg_wait | w_long_wait | w_ontime | w_bunching | w_headway_cv | w_energy_pp | w_fleet | w_intervention | w_constraint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0_SERVICE_SAFETY | 4.0 | 2.0 | 3.0 | 0.5 | 1.0 | 0.5 | 0.0 | 0.0 | 0.25 | 12.0 |
| R1_BALANCED_DEFAULT | 3.0 | 2.0 | 3.0 | 1.0 | 1.5 | 1.0 | 1.0 | 0.5 | 0.25 | 10.0 |
| R2_WAIT_HEAVY | 3.0 | 3.0 | 3.5 | 0.8 | 1.5 | 1.0 | 0.75 | 0.25 | 0.25 | 12.0 |
| R3_LONG_WAIT_FAIRNESS | 3.2 | 2.0 | 4.0 | 0.8 | 1.5 | 1.0 | 0.75 | 0.25 | 0.25 | 12.0 |
| R4_ENERGY_LIGHT | 3.0 | 2.0 | 3.0 | 1.0 | 1.5 | 1.0 | 1.5 | 0.25 | 0.25 | 12.0 |
| R5_FLEET_CAUTIOUS | 3.5 | 2.0 | 3.5 | 1.0 | 1.5 | 1.0 | 1.0 | 0.75 | 0.3 | 12.0 |

## Safety Rules

Every candidate must satisfy:

- service-quality weights dominate energy/fleet weights
- `w_long_wait >= w_avg_wait`
- fleet bonus does not exceed energy-per-passenger penalty
- all candidates share the same hard constraints
- no candidate directly rewards total `energy_proxy`
- no candidate is selected as final in Step 115

## Interpretation

Step 115 produces a safe comparison matrix. It does not prove that any candidate is better. Candidate selection must wait until later gates and must not rely on smoke-only output.
