# Step 101 — Route-aware Minimal Simulator Scaffold

## Purpose

Step 101 converts the Step 100 causal simulator v2 contract into the first executable route-aware simulator scaffold.

The scaffold reads the existing `/getBs02` route-stop sequence artifact and verifies that bus-like agents can move monotonically along a `route_id + direction_id + ordered_stop_sequence` path.

## Inputs

Default inputs:

```text
artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/causal_simulator_v2_contract.json
artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/getbs02_route_stop_sequence_normalized.csv
```

## Outputs

Default outputs:

```text
artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/route_aware_minimal_simulator_manifest.json
artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/route_aware_minimal_simulator_report.md
artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/route_sequence_readiness_step101.csv
artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/smoke_rollout_trace_step101.csv
```

## Safety boundary

Step 101 does not perform DB (Database=데이터베이스) writes, tensor DB (Database=데이터베이스) overwrites, or additional API (Application Programming Interface=응용 프로그램 인터페이스) calls.

The scaffold is not a causal performance simulator. It does not observe:

```text
actual_headway
actual_arrival_departure_time
actual_dwell
```

Therefore the following guards remain fixed:

```text
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## What this step proves

Step 101 proves only that:

```text
route_id + direction_id + ordered_stop_sequence
→ route-aware agent position
→ monotonic scaffold movement
→ smoke rollout trace
```

can be executed from existing artifacts.

## What this step does not prove

Step 101 does not prove:

```text
real bus dispatch quality
actual headway improvement
actual passenger waiting-time improvement
actual dwell-time realism
MAPPO policy performance
paper-level causal performance
```

## Usage

```powershell
python -m py_compile `
  05_training/adapters/route_aware_minimal_simulator_step101.py `
  05_training/adapters/test_route_aware_minimal_simulator_step101.py

python 05_training/adapters/test_route_aware_minimal_simulator_step101.py

python 05_training/adapters/route_aware_minimal_simulator_step101.py
```

Optional route-specific smoke run:

```powershell
python 05_training/adapters/route_aware_minimal_simulator_step101.py `
  --route-id 1234567890 `
  --direction-id 0 `
  --num-agents 3 `
  --max-steps 8
```

## Next step

Step 102 should build a route-aware rollout writer scaffold that converts Step 101 simulator transitions into canonical `raw_events` and `window_rollup` artifacts without making causal performance claims.
