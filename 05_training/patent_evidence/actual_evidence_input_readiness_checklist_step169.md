# Step 169 — Actual Evidence Input Readiness Checklist

Project: `urbanbus_rl_project`  
Status: Non-claim readiness gate  
Purpose: verify that the inputs required to run the Zero-Loss Pickup Threshold evidence chain are present before executing an actual route-aware rollout evidence report.

## Scope

Step 169 does **not** run training, does **not** select a winner, and does **not** permit paper-level or causal-performance claims.

It checks whether the following inputs are available and structurally compatible:

1. `raw_events` from a route-aware rollout.
2. `window_rollup` from the same run.
3. `route_stop_sequence` for route/path matching.
4. `run_manifest` documenting source run identity and non-claim flags.
5. GATv2 attention extraction readiness, either by providing a checkpoint path or confirming the Step 163 extractor is present.
6. Required Step 160/162/163/165/166 pipeline scripts.

## Expected downstream chain

```text
raw_events / window_rollup / route_stop_sequence / run_manifest
→ Step 162 route-aware rollout adapter
→ Step 161 pickup-attempt and ETA counterfactual writer
→ Step 163 real GATv2 attention extractor
→ Step 165 attempt-specific route/path attention filter
→ Step 160 Zero-Loss evidence reporter
→ Step 166 patent evidence report v2
```

## Required raw event columns

The readiness gate expects route-aware rollout rows to contain at least:

```text
attempt_id
state_ts
condition_id
seed
route_id
direction_id
vehicle_id
current_stop_id
pickup_stop_id
dropoff_stop_id
existing_passenger_id
new_passenger_id
eta_without_new_pickup_sec
eta_with_new_pickup_sec
```

The exact names are intentionally aligned with Step 160/162/165 contracts so that ETA counterfactual values are not duplicated in multiple files after normalization.

## Required route-stop sequence columns

```text
route_id
direction_id
stop_id
stop_order
```

Optional but useful:

```text
node_id
node_index
node_uid
```

## Guard flags

The manifest must keep the following flags locked:

```text
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
actual_operational_claim_allowed = false
train_allowed = false
winner_selected = false
```

Even if all inputs pass this checklist, the correct claim is only:

```text
Actual-like evidence input files are ready to be processed by the Zero-Loss evidence pipeline.
```

The incorrect claim is:

```text
The real-world Zero-Loss Pickup success rate has been proven.
```

## Modes

### `template`

Creates a blocked checklist manifest without input validation. This is useful for documenting the expected inputs before actual files exist.

### `sample`

Creates small synthetic sample inputs and validates the readiness checker end-to-end. This is a smoke test only.

### `check`

Validates user-supplied paths for actual-like execution. This mode can mark `ready_for_actual_like_execution=true` if all inputs are present and structurally valid, but claim flags remain locked.
