# Step 165 — Attempt-Specific Route/Path Attention Filter

## Purpose

Step 165 upgrades Step 164's real-attention connector from snapshot-level top-k projection to attempt-specific route/path filtering.

The goal is to use real GATv2 attention rows, but attach only the rows that are relevant to each pickup attempt's route context:

```text
current_stop_id -> pickup_stop_id -> dropoff_stop_id
```

This is a patent-evidence pipeline step, not a performance-claim step.

## Inputs

### Required in from-files mode

1. `pickup_attempt_events.csv` or `.parquet`
2. `real_attention.csv` or `.parquet`
3. `route_stop_sequence.csv` or `.parquet`

### Optional

1. `eta_counterfactual.csv` or `.parquet`
2. `run_manifest.json`

If `eta_counterfactual` is absent but the pickup attempt file contains direct ETA columns, Step 165 creates it from those columns.

## Required attempt columns

```text
attempt_id
state_ts
condition_id
seed
vehicle_id
existing_passenger_id
new_passenger_id
route_id
direction_id
current_stop_id
pickup_stop_id
dropoff_stop_id
decision
```

`current_stop_id` is required for path filtering. If a future raw rollout lacks it, Step 162 should recover it from route-aware state before Step 165.

## Required route-stop sequence columns

```text
route_id
direction_id
stop_id
stop_order
node_index
```

The route sequence is used to map stops to graph node indices and then build path edge candidates.

## Required real attention columns

```text
state_ts
layer_id
head_id
src_node
dst_node
attention_weight
```

The real attention file should come from Step 163 or Step 164 and should also carry:

```text
attention_source = gatv2conv_return_attention_weights
real_gatv2conv_attention_extracted = true
```

## Filtering rule

For each pickup attempt:

1. Locate `current_stop_id`, `pickup_stop_id`, and `dropoff_stop_id` on the route sequence.
2. Build directed route edges for:
   - `candidate_pickup_path`: current -> pickup
   - `candidate_dropoff_path`: pickup -> dropoff
   - `existing_passenger_path`: current -> dropoff
3. Match GATv2 attention rows whose `(src_node, dst_node)` edge is in one of those path edge sets.
4. If no exact edge match exists, fall back to node-overlap rows using the same route segment node set.
5. If still empty, use snapshot top-k as a last-resort non-claim fallback and mark the attempt accordingly.

## Outputs

```text
pickup_attempt_events.csv
eta_counterfactual.csv
gatv2_attention.csv
attempt_route_path_filter_report.csv
attention_path_mass_by_attempt.csv
run_manifest.json
attempt_route_path_attention_filter_manifest.json
```

The `gatv2_attention.csv` output is Step 160-compatible.

## Non-claim guards

Step 165 must keep:

```text
simulation_evidence_only = true
actual_operational_claim_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
train_allowed = false
```

Step 165 proves that route/path-specific filtering works. It does not prove real-world zero-loss performance.
