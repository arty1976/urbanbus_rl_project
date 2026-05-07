# Step 161 — Route-Aware Pickup Attempt Event Writer Schema

Project: `urbanbus_rl_project`  
Status: scaffold / non-claim evidence input generator  
Consumes: route-aware simulator rollout artifacts or deterministic sample rows  
Produces: Step 160-compatible zero-loss pickup evidence inputs

---

## 1. Purpose

Step 160 created the Zero-Loss Pickup Evidence Reporter. That reporter expects three evidence input tables:

1. `pickup_attempt_events`
2. `eta_counterfactual`
3. `gatv2_attention`

Step 161 creates the preceding writer that can generate those three input tables from route-aware simulator rollout rows or from a deterministic sample mode. The output is intended for patent evidence scaffolding only.

This step does **not** assert actual field operation performance.

```text
simulation_evidence_only = true
actual_operational_claim_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
train_allowed = false
```

---

## 2. Output files

The writer creates a bundle root containing:

```text
pickup_attempt_events.{csv|parquet}
eta_counterfactual.{csv|parquet}
gatv2_attention.{csv|parquet}
run_manifest.json
route_aware_pickup_attempt_event_writer_manifest.json
```

The first four files are direct inputs to Step 160.

---

## 3. `pickup_attempt_events` required columns

| Column | Meaning |
|---|---|
| `attempt_id` | Stable pickup-attempt identifier |
| `state_ts` | Simulator state timestamp |
| `condition_id` | A/A90/A80/A70 or related condition |
| `seed` | Experiment seed |
| `vehicle_id` | Vehicle or agent identifier |
| `existing_passenger_id` | Already-boarded passenger identifier |
| `new_passenger_id` | Candidate new passenger identifier |
| `route_id` | Route identifier |
| `direction_id` | Direction identifier |
| `pickup_stop_id` | Candidate pickup stop |
| `dropoff_stop_id` | Candidate drop-off stop |
| `decision` | `accepted` or `rejected` |

Recommended optional columns:

```text
window_id
time_band
pickup_stop_order
dropoff_stop_order
candidate_request_time_sec
policy_action
policy_score
reason_code
```

---

## 4. `eta_counterfactual` required columns

| Column | Meaning |
|---|---|
| `attempt_id` | Stable pickup-attempt identifier |
| `existing_passenger_id` | Already-boarded passenger identifier |
| `eta_without_new_pickup_sec` | Existing passenger ETA without candidate pickup |
| `eta_with_new_pickup_sec` | Existing passenger ETA with candidate pickup |

Derived by Step 160:

```text
delta_eta_existing_passenger_sec =
  eta_with_new_pickup_sec - eta_without_new_pickup_sec

zero_loss_success =
  delta_eta_existing_passenger_sec <= zero_loss_epsilon_sec
```

In Step 161 scaffold mode, ETA values are deterministic proxy values. They are not actual observed arrival times.

---

## 5. `gatv2_attention` required columns

| Column | Meaning |
|---|---|
| `attempt_id` | Stable pickup-attempt identifier |
| `state_ts` | Simulator state timestamp |
| `layer_id` | GATv2 layer index |
| `head_id` | Attention head index |
| `src_node` | Source node |
| `dst_node` | Destination node |
| `attention_weight` | Attention weight |

Recommended optional columns:

```text
edge_id
route_id
direction_id
distance_m
time_sec
generalized_cost
path_segment
```

`path_segment` should be one of:

```text
existing_passenger_path
candidate_pickup_path
candidate_dropoff_path
unrelated
unknown
```

---

## 6. Input modes

### 6.1 `sample`

Creates deterministic toy pickup-attempt rows. This is used for self-test and schema validation.

### 6.2 `from-rollout`

Reads a simulator event table. The input may be CSV or Parquet. If the expected columns are missing, the writer fills safe deterministic defaults and marks the evidence as proxy/scaffold.

Expected useful input columns:

```text
state_ts
condition_id
seed
vehicle_id / agent_id
route_id
direction_id
current_stop_id
next_stop_id
window_id
time_band
policy_action
decision
```

---

## 7. Claim boundary

The Step 161 output is only an evidence-input bundle for Step 160. It does not prove actual operational zero-loss performance.

Allowed wording:

```text
The route-aware simulator produced pickup-attempt evidence rows that can be used by the Zero-Loss Pickup Evidence Reporter.
```

Prohibited wording:

```text
Actual Daegu operations achieved zero-loss shared pickup.
The policy is deployable.
The result is paper-level performance evidence.
```
