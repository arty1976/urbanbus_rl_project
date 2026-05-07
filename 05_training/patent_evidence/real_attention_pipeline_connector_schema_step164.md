# Step 164 — Real GATv2 Attention Pipeline Connector Schema

## Purpose

Step 164 connects Step 163 real `GATv2Conv(return_attention_weights=True)` attention outputs to the Step 160 Zero-Loss Pickup Evidence Reporter input contract.

This step is a connector, not a performance experiment.

## Scope

Step 164 does the following:

1. Read pickup attempt events.
2. Read ETA counterfactual rows.
3. Read Step 163 `gatv2_attention` rows.
4. Verify that the attention source is real GATv2 attention when `--require-real-attention` is enabled.
5. Re-emit Step 160 compatible inputs:
   - `pickup_attempt_events.csv`
   - `eta_counterfactual.csv`
   - `gatv2_attention.csv`
   - `run_manifest.json`
6. Write `real_attention_pipeline_connector_manifest.json`.
7. Preserve non-claim guards.

## Non-goals

Step 164 does not yet perform attempt-specific route/path edge filtering. That is Step 165.

Therefore Step 164 may still use the attempt-level attention rows produced by Step 163, including top-k mapping. It only verifies and connects real attention source compatibility.

## Required input columns

### pickup attempt events

- `attempt_id`
- `state_ts`
- `condition_id`
- `seed`
- `vehicle_id`
- `existing_passenger_id`
- `new_passenger_id`
- `route_id`
- `direction_id`
- `pickup_stop_id`
- `dropoff_stop_id`
- `decision`

### ETA counterfactual

- `attempt_id`
- `existing_passenger_id`
- `eta_without_new_pickup_sec`
- `eta_with_new_pickup_sec`

### Step 163 attention

- `attempt_id`
- `state_ts`
- `layer_id`
- `head_id`
- `src_node`
- `dst_node`
- `attention_weight`

Recommended metadata columns:

- `attention_source`
- `real_gatv2conv_attention_extracted`
- `path_segment`
- `edge_id`
- `distance_m`
- `time_sec`
- `generalized_cost`

## Connector output status

Successful output uses:

```text
bundle_status = REAL_GATV2_ATTENTION_CONNECTED_TO_ZERO_LOSS_EVIDENCE_NONCLAIM
```

## Guard flags

The following must remain locked:

```text
simulation_evidence_only = true
actual_operational_claim_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
trained_model_claim_allowed = false
train_allowed = false
```

## Interpretation

A PASS means:

```text
The Zero-Loss Pickup evidence pipeline can consume real GATv2 attention rows.
```

It does not mean:

```text
The real Daegu full-scale experiment proved a Zero-Loss Pickup performance claim.
```
