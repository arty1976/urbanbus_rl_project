# Step 162 — Route-Aware Rollout Adapter Schema

## Purpose

Step 162 converts actual or scaffold route-aware simulator rollout artifacts into the normalized event table consumed by Step 161.

```text
route-aware simulator raw_events / rollout output
→ Step 162 normalized_route_aware_rollout_events
→ Step 161 pickup_attempt_events + ETA counterfactual + GATv2 attention input
→ Step 160 Zero-Loss Pickup Evidence Report
```

## Non-claim boundary

This adapter is an evidence plumbing layer only.

```text
simulation_evidence_only = true
actual_operational_claim_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
train_allowed = false
```

The adapter may consume real simulator outputs, but unless the source rollout was produced by a separately released, validated actual run, Step 162 output remains simulation/proxy evidence. It must not be described as real-world observed zero-loss performance.

## Primary input

Supported table formats:

```text
.csv
.parquet
.json
.jsonl
```

Expected source artifact:

```text
raw_events.parquet / raw_events.csv / route_aware_rollout_events.*
```

Optional source artifact:

```text
window_rollup.parquet / window_rollup.csv
```

Optional manifest:

```text
run_manifest.json / rollout manifest / training manifest
```

## Required normalized output columns

`normalized_route_aware_rollout_events.csv` or `.parquet` must contain:

| Column | Meaning |
|---|---|
| `state_ts` | simulator state timestamp |
| `condition_id` | A/A90/A80/A70 or experiment condition |
| `seed` | random seed |
| `window_id` | evaluation window id |
| `time_band` | peak/offpeak/night or unknown |
| `vehicle_id` | vehicle or bus-agent id |
| `agent_id` | agent id, when available |
| `route_id` | route identifier |
| `direction_id` | route direction identifier |
| `current_stop_id` | stop at or near the pickup decision point |
| `next_stop_id` | next stop, if available |
| `pickup_stop_id` | candidate pickup stop id |
| `dropoff_stop_id` | candidate dropoff stop id |
| `policy_action` | action label passed to Step 161 |
| `decision` | accepted or rejected |
| `existing_passenger_id` | existing passenger affected by counterfactual ETA |
| `new_passenger_id` | new candidate passenger/request id |
| `eta_without_new_pickup_sec` | existing passenger ETA without candidate pickup |
| `eta_with_new_pickup_sec` | existing passenger ETA with candidate pickup |
| `route_sequence_index` | route sequence index candidate |
| `source_row_index` | source row index in raw event file |
| `source_table` | source table label |
| `actual_operational_observation` | always false in Step 162 scaffold |
| `simulation_evidence_only` | always true |

## Alias mapping

The adapter accepts multiple aliases for the same field. Examples:

```text
vehicle_id     <- vehicle_id, bus_id, bus_id_or_vehicle_no, vhcNo2, agent_id
current_stop_id<- current_stop_id, stop_id, bsId, current_stop, node_uid
route_id       <- route_id, routeId, line_id
seed           <- seed, training_seed, eval_seed
policy_action  <- policy_action, action, action_name, dispatch_action
```

A `source_column_mapping.json` file is written so the final evidence bundle can disclose which source columns were used.

## ETA handling

Priority order:

```text
1. Use explicit eta_without_new_pickup_sec / eta_with_new_pickup_sec when present.
2. Use alias columns such as eta_baseline_sec / eta_after_pickup_sec when present.
3. Use eta_delta_existing_passenger_sec with a deterministic baseline ETA when present.
4. Generate a deterministic proxy ETA for scaffold/integration smoke only.
```

If proxy ETA is used, `eta_proxy_used = true` and claim flags remain locked.

## Output files

```text
normalized_route_aware_rollout_events.csv
source_column_mapping.json
data_quality_report.json
route_aware_rollout_adapter_manifest.json
```

## Compatibility target

The normalized table is designed to be passed into Step 161 as:

```powershell
python 05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py `
  --mode from-rollout `
  --rollout-events <normalized_route_aware_rollout_events.csv> `
  --source-manifest <route_aware_rollout_adapter_manifest.json> `
  --output-root <step161_output_root>
```
