# Step 113 ??Reward Normalization Baseline Lock

## Purpose

Step 113 locks the baseline reference used for reward candidate normalization.

This step does not finalize reward weights, hard constraints, numerical constants, or training approval.

## Locked Reference

The candidate reward normalization baseline is:

```text
B1_noop
```

Reason:

- B1_noop represents no policy intervention.
- It is the cleanest anchor for asking whether MAPPO improves beyond doing nothing.
- B0_historical remains important for reporting but may have missing or legacy KPI coverage.
- B2_rulebased is a comparator policy, not the primary reward normalization anchor.

## Reporting Baselines

Reporting should still compare against:

```text
B0_historical
B1_noop
B2_rulebased
```

## Still Not Trainable

The following remain false:

- `train_with_this_reward_allowed`
- `final_reward_design_claim_allowed`
- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`
- `reward_formula_finalized`
- `reward_weights_locked`
- `normalization_numeric_values_locked`
- `hard_constraints_locked`

## Normalization Rules

The main normalization rules are:

```text
avg_wait_penalty =
  max(0, avg_wait_seconds / baseline_B1_noop_avg_wait_seconds - 1)

long_wait_penalty =
  max(0, passenger_wait_p95_seconds / baseline_B1_noop_passenger_wait_p95_seconds - 1)

energy_per_passenger_penalty =
  energy_proxy_per_passenger / max(baseline_B1_noop_energy_proxy_per_passenger, epsilon)

service_rate_score =
  clip(passenger_service_rate, 0.0, 1.0)

fleet_reduction_bonus =
  clip(fleet_reduction_ratio, 0.0, 1.0)
```

## Leakage Guard

Final numerical baseline constants must not be computed from held-out test windows.
Use calibration/train-reference baseline artifacts, then report held-out test performance separately.

## Proxy Guard

The following remain not observed:

- `actual_headway`
- `actual_arrival_departure_time`
- `actual_dwell`
- `actual_passenger_wait_observed`

Any queue, wait, ETA, or headway candidate must retain proxy/candidate labels.

## Forbidden Direct Normalization Terms

These are not direct reward normalization terms:

- `passenger_demand_generated`
- `passenger_served_count`
- `energy_proxy`

They remain audit/context fields. Direct reward should use service rate and energy per passenger instead.

## Next Gates

1. Step 114 ??hard constraint review
2. Step 115 ??reward ablation matrix
3. Step 116 ??trainable reward promotion gate
