# Step 112 — Reward Candidate Protocol

## Purpose

Step 112 defines a controlled candidate protocol for reward weights derived from Step 111.

This step does not select a final reward and does not allow MAPPO training.

## Current status

- `protocol_status = DRAFT_NOT_TRAINABLE`
- `candidate_selected = false`
- `reward_weights_locked = false`
- `normalization_locked = false`
- `hard_constraints_locked = false`
- `train_with_candidate_reward_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Why this step is needed

Step 111 created one draft reward formula and one balanced draft value set. That is not enough to justify a final training reward.

Step 112 therefore turns the single draft into a small candidate family:

1. `R0_BALANCED_STEP111`
2. `R1_SERVICE_STRICT`
3. `R2_WAIT_FAIRNESS_HEAVY`
4. `R3_ENERGY_LIGHT`
5. `R4_FLEET_BONUS_BOUNDARY`

These candidates let us later test whether the reward is too weak on service, too weak on long-wait fairness, too aggressive on energy, or too permissive about fleet reduction.

## Shared rule

All weights are positive magnitudes.

Good terms are added:

```text
+ weight * score
```

Bad terms are subtracted:

```text
- weight * penalty
```

## Candidate summaries

### R0_BALANCED_STEP111

This is the Step 111 balanced reference candidate.

It keeps the main service-first structure:

- service rate strong
- p95 wait at least as important as average wait
- energy/fleet secondary

### R1_SERVICE_STRICT

This candidate increases service protection.

It is designed to block this shortcut:

```text
reduce buses -> reduce energy -> leave passengers unserved
```

### R2_WAIT_FAIRNESS_HEAVY

This candidate increases average-wait and p95-wait penalties.

It is designed to block this shortcut:

```text
average wait looks acceptable -> some passengers wait far too long
```

### R3_ENERGY_LIGHT

This candidate reduces energy pressure.

It asks:

```text
Can the policy still improve service quality when energy pressure is light?
```

### R4_FLEET_BONUS_BOUNDARY

This candidate tests the upper boundary for fleet reduction bonus.

It still keeps fleet reduction smaller than energy-per-passenger and much smaller than service-quality terms.

## Candidate rejection rules

Reject a candidate if:

1. Any weight is non-positive.
2. Service-quality weight sum does not dominate energy/fleet weight sum.
3. `w_long_wait < w_avg_wait`.
4. `w_fleet_reduction > w_energy_per_passenger`.
5. Total `energy_proxy` is directly rewarded.
6. `train_with_candidate_reward_allowed=true` appears in Step 112.
7. Paper-level or causal performance claim is allowed.

## Next steps

Step 112 only defines the protocol.

Before training, the following gates are still required:

1. Step 113 — reward normalization baseline lock
2. Step 114 — hard constraint review
3. Step 115 — reward ablation matrix
4. Step 116 — trainable reward promotion gate
