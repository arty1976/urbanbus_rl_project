# Step 100 — causal simulator v2 contract update

## Purpose

Step 100 turns the Step 99-F consolidated classification patch into a formal causal simulator v2 contract.

It answers this question:

```text
What can the route-aware causal simulator v2 consume as observed/candidate/proxy inputs, and what must remain blocked?
```

This step does not implement the simulator adapter. It only fixes the contract for the next implementation step.

## Safety contract

- DB (Database=데이터베이스) write: forbidden
- tensor DB (Database=데이터베이스) overwrite: forbidden
- additional API (Application Programming Interface=응용 프로그램 인터페이스) calls: forbidden
- simulator rollout execution: forbidden
- paper-level claim: forbidden
- causal performance claim: forbidden

## Default inputs

```text
artifacts/daegu_bis_api_audit/bis_api_classification_step99f/step98_causal_simulator_v2_contract_patch.json
artifacts/daegu_bis_api_audit/bis_api_classification_step99f/classification_update_consolidated_step99f.json
```

## Outputs

```text
artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/causal_simulator_v2_contract.json
artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/causal_simulator_v2_contract.md
artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/causal_simulator_v2_field_readiness_matrix.csv
artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/causal_simulator_v2_contract_manifest.json
```

## Contract scope

Step 100 defines a **route-aware minimal causal simulator v2** contract.

Required route-aware inputs:

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

Optional API candidate inputs:

```text
bus_id_or_vehicle_no
live_position_xy
current_route_sequence
current_stop_id
vehicle_trajectory
getRealtime02_eta
eta_based_headway
```

Blocked actual-observed inputs:

```text
actual_headway
actual_arrival_departure_time
actual_dwell
vehicle_load
left_behind_passengers
```

## Critical rule

`eta_based_headway` may be used only as a candidate/calibration signal.

It must not be renamed or treated as `actual_headway`.

## KPI contract

The contract fixes the 12-KPI (Key Performance Indicator=핵심 성과 지표) schema:

```text
cv_headway
avg_wait_seconds
bunching_rate
on_time_rate
intervention_rate
energy_proxy
passenger_demand_generated
passenger_served_count
passenger_service_rate
passenger_wait_p95_seconds
energy_proxy_per_passenger
fleet_reduction_ratio
```

Some of these are simulator-generated or proxy metrics. They are not paper-level actual operating measurements at this step.

## Run commands

From the project root:

```powershell
python -m py_compile `
  05_training/adapters/build_causal_simulator_v2_contract_step100.py `
  05_training/adapters/test_causal_simulator_v2_contract_step100.py

python 05_training/adapters/test_causal_simulator_v2_contract_step100.py

python 05_training/adapters/build_causal_simulator_v2_contract_step100.py
```

## Expected result

```text
[OK] Step 100 causal simulator v2 contract update completed
[OK] contract_status: READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD
```

If the script detects an upstream mismatch that it can safely guard against, the status can be:

```text
READY_WITH_WARNINGS
```

That is acceptable only if the warning is understood and documented.

## Prohibited claims

Do not claim:

```text
actual_headway is observed
actual_arrival_departure_time is observed
actual_dwell is observed
ETA-based headway is actual headway
Step 100 proves causal simulator performance
Step 100 produces paper-level results
```

## Files created by this package

```text
05_training/adapters/build_causal_simulator_v2_contract_step100.py
05_training/adapters/test_causal_simulator_v2_contract_step100.py
05_training/adapters/causal_simulator_v2_contract_step100.md
```
