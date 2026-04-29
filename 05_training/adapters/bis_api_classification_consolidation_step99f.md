# Step 99-F — Step 97/98 classification update consolidation

## Purpose

Step 99-F consolidates the Step 99-A~E BIS (Bus Information System=버스정보시스템) API (Application Programming Interface=응용 프로그램 인터페이스) audit results into one official classification update package for Step 97 and Step 98.

This step answers one question:

```text
Which fields can now be used by the causal simulator v2 contract, and under what claim boundary?
```

It does **not** create a simulator, does **not** update the tensor DB (Database=데이터베이스), and does **not** make a performance claim.

## Safety contract

- DB write: forbidden
- tensor DB overwrite: forbidden
- additional API calls: forbidden
- paper-level claim: forbidden
- causal performance claim: forbidden
- allowed input: existing Step 99-E JSON/CSV artifacts only

## Default inputs

```text
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/classification_update_step99e.json
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/bis_api_integration_report.json
```

Optional trace input:

```text
artifacts/tensor_db_data_availability_audit/tensor_db_available_fields_report.json
```

The optional Step 97 report is used only for traceability. Step 99-F uses its own conservative baseline classification map so it remains stable even if the Step 97 report schema changes.

## Outputs

```text
artifacts/daegu_bis_api_audit/bis_api_classification_step99f/classification_update_consolidated_step99f.json
artifacts/daegu_bis_api_audit/bis_api_classification_step99f/classification_update_consolidated_step99f.md
artifacts/daegu_bis_api_audit/bis_api_classification_step99f/field_classification_matrix_step99f.csv
artifacts/daegu_bis_api_audit/bis_api_classification_step99f/step98_causal_simulator_v2_contract_patch.json
```

## Main classification changes

### Upgraded fields

| Field | Final classification |
|---|---|
| `route_id` | `observed_candidate_full_collection_234_of_238_routes` |
| `direction_id` | `observed_candidate_full_collection_234_of_238_routes_cross_confirmed` |
| `ordered_stop_sequence` | `observed_candidate_full_collection_234_of_238_routes` |
| `bus_id_or_vehicle_no` | `repeated_sample_observed_candidate_from_getPos02_vhcNo2` |
| `live_position_xy` | `repeated_sample_observed_candidate_from_getPos02_xPos_yPos` |
| `current_route_sequence` | `repeated_sample_observed_candidate_from_getPos02_seq` |
| `current_stop_id` | `repeated_sample_observed_candidate_from_getPos02_bsId` |
| `vehicle_trajectory` | `trajectory_candidate_from_repeated_getPos02_sampling` |
| `getRealtime02_eta` | `observed_candidate_cross_matched_to_getBs02` |
| `eta_based_headway` | `candidate_from_getRealtime02_cross_matched_to_getBs02` |

### Fields that must not be upgraded

| Field | Final classification | Reason |
|---|---|---|
| `actual_headway` | `not_observed` | ETA rows are not actual arrival/departure events. |
| `actual_arrival_departure_time` | `not_observed` | No stop-level actual event timestamp source exists. |
| `actual_dwell` | `not_observed` | No observed dwell duration source exists. |
| `passenger_wait_age_distribution` | `missing` | Needed for true p95 passenger wait. |
| `vehicle_load` | `missing` | Needed for crowding/capacity validation. |
| `left_behind_passengers` | `missing` | Needed for denied-boarding validation. |

## Step 98 contract impact

Step 99-F prepares a patch file for causal simulator v2 contract drafting.

Required route-aware v2 fields:

```text
route_id
direction_id
ordered_stop_sequence
node_uid
node_index
edge_index
edge_distance_m
edge_time_sec
boardings_recent
alightings_recent
waiting_passenger_cnt
hour_sin_cos
is_peak
```

Optional API candidate fields:

```text
bus_id_or_vehicle_no
live_position_xy
current_route_sequence
current_stop_id
vehicle_trajectory
getRealtime02_eta
eta_based_headway
```

Important rule:

```text
eta_based_headway may be used only as a candidate/proxy field.
actual_headway must remain not_observed.
```

## Run commands

From project root:

```powershell
python -m py_compile `
  05_training/adapters/consolidate_bis_api_classification_step99f.py `
  05_training/adapters/test_bis_api_classification_step99f.py

python 05_training/adapters/test_bis_api_classification_step99f.py

python 05_training/adapters/consolidate_bis_api_classification_step99f.py
```

If the optional Step 97 report file is not present, run:

```powershell
python 05_training/adapters/consolidate_bis_api_classification_step99f.py --no-step97-report
```

## Expected result

If Step 99-E was successful, the expected console result is:

```text
[OK] Step 99-F classification consolidation completed
[OK] audit_status: PASS
[OK] getPos02 coverage: 1.0
[OK] getRealtime02 ETA coverage: 1.0
[OK] headway candidate coverage: 1.0
```

## Prohibited claims

Do not claim:

```text
actual_headway is observed
actual_arrival_departure_time is observed
actual_dwell is observed
ETA-based headway is paper-level actual headway evidence
Step 99-F proves causal simulator performance
Step 99-F updates the tensor DB
```

## Files created by this package

```text
05_training/adapters/consolidate_bis_api_classification_step99f.py
05_training/adapters/test_bis_api_classification_step99f.py
05_training/adapters/bis_api_classification_consolidation_step99f.md
```
