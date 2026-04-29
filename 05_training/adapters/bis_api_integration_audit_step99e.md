# Step 99-E — BIS API data source integration audit

## Purpose

This step links the already-created BIS (Bus Information System=버스정보시스템) API (Application Programming Interface=응용 프로그램 인터페이스) artifacts from `/getBs02`, `/getPos02`, and `/getRealtime02`.

The goal is not to collect new data. The goal is to confirm whether the three artifact families use compatible `route_id`, `direction_id`, `stop_id`, and stop-sequence semantics.

## Safety contract

This step must remain read-only.

- DB (Database=데이터베이스) write: forbidden
- tensor DB (Database=데이터베이스) overwrite: forbidden
- additional API calls: forbidden
- allowed input: existing artifact CSV/JSON files only
- `paper_level_claim_allowed=false`
- `causal_performance_claim_allowed=false`

## Input files

Default inputs are:

```text
artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/getbs02_route_stop_sequence_normalized.csv
artifacts/daegu_bis_api_audit/getpos02_repeated_sampling/20260429_090603/getpos02_trajectory_candidate_timeseries.csv
artifacts/daegu_bis_api_audit/getpos02_repeated_sampling/20260429_090603/getpos02_trajectory_candidate_summary.csv
artifacts/daegu_bis_api_audit/getrealtime02_eta_sampling_arrlist_fix/getrealtime02_eta_normalized.csv
artifacts/daegu_bis_api_audit/getrealtime02_eta_sampling_arrlist_fix/getrealtime02_headway_candidate.csv
```

## Output files

The analyzer writes:

```text
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/bis_api_integration_report.json
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/bis_api_integration_report.md
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/getpos02_to_getbs02_match.csv
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/getrealtime02_to_getbs02_match.csv
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/headway_candidate_to_getbs02_match.csv
artifacts/daegu_bis_api_audit/bis_api_integration_step99e/classification_update_step99e.json
```

## What the script checks

### 1. `/getBs02` base table

The analyzer builds a canonical route-stop sequence table with:

```text
route_id
direction_id
stop_id
getbs02_stop_order
route_no
stop_name
```

`route_no` and `stop_name` are preserved when present.

### 2. `/getPos02` to `/getBs02` match

Join key:

```text
getPos02.route_id
getPos02.direction_id
getPos02.current_stop_id = getBs02.stop_id
```

The output `getpos02_to_getbs02_match.csv` includes:

```text
route_stop_match
missing_route_stop_match
getbs02_stop_order
current_stop_order
order_delta
exact_order_match
```

### 3. `/getRealtime02` ETA to `/getBs02` match

Join key:

```text
route_id
direction_id
stop_id
```

The output is `getrealtime02_to_getbs02_match.csv`.

### 4. `/getRealtime02` headway candidate to `/getBs02` match

Join key:

```text
route_id
direction_id
stop_id
```

The output is `headway_candidate_to_getbs02_match.csv`.

### 5. Route/direction consistency

The report records:

```text
getBs02 route-direction count
getPos02 route-direction count
getRealtime02 route-direction count
intersection across all three
unmatched getPos02 route-direction count
unmatched getRealtime02 route-direction count
```

## Classification update

Step 99-E writes the following classification updates:

| Field | Classification |
|---|---|
| `route_id` | `observed_candidate_full_collection_234_of_238_routes` |
| `direction_id` | `observed_candidate_full_collection_234_of_238_routes_cross_confirmed` |
| `ordered_stop_sequence` | `observed_candidate_full_collection_234_of_238_routes` |
| `bus_id_or_vehicle_no` | `repeated_sample_observed_candidate_from_getPos02_vhcNo2` |
| `live_position_xy` | `repeated_sample_observed_candidate_from_getPos02_xPos_yPos` |
| `current_route_sequence` | `repeated_sample_observed_candidate_from_getPos02_seq` |
| `current_stop_id` | `repeated_sample_observed_candidate_from_getPos02_bsId` |
| `vehicle_trajectory` | `trajectory_candidate_from_repeated_getPos02_sampling` |
| `getRealtime02_eta` | `observed_candidate` |
| `eta_based_headway` | `candidate_from_getRealtime02` |
| `actual_headway` | `not_observed` |
| `actual_arrival_departure_time` | `not_observed` |
| `actual_dwell` | `not_observed` |
| `passenger_wait_age_distribution` | `missing` |
| `vehicle_load` | `missing` |
| `left_behind_passengers` | `missing` |

## Important interpretation guard

Even if `eta_based_headway` is available, it is still only a candidate from `/getRealtime02`.

Do not upgrade the following fields to observed in this step:

```text
actual_headway
actual_arrival_departure_time
actual_dwell
```

Reason: `/getRealtime02` ETA (Estimated Time of Arrival=도착예정시간) rows and `/getPos02` live snapshots are not the same as actual stop arrival/departure event logs.

## Run commands

From project root:

```powershell
python -m py_compile `
  05_training/adapters/analyze_bis_api_integration_step99e.py `
  05_training/adapters/test_bis_api_integration_step99e.py

python 05_training/adapters/test_bis_api_integration_step99e.py

python 05_training/adapters/analyze_bis_api_integration_step99e.py
```

## Optional threshold overrides

The default match threshold is 80%.

```powershell
python 05_training/adapters/analyze_bis_api_integration_step99e.py `
  --min-match-coverage 0.80 `
  --min-exact-order-coverage 0.80
```

If route-stop coverage is high but exact order coverage is low, the script does not automatically reject the integration. It records:

```text
sequence_semantic_review_required
```

This means the route-stop ID system is compatible, but `/getPos02 seq` and `/getBs02 ordered_stop_sequence` may use different sequence semantics and need manual review.

## Expected status interpretation

| Status | Meaning |
|---|---|
| `PASS` | All non-empty match coverages meet thresholds and no sequence warning was raised. |
| `PASS_WITH_REVIEW_REQUIRED` | Some match coverage or order semantics require review, but at least some integration evidence exists. |
| `REVIEW_REQUIRED` | A non-empty input source has zero route-stop matches to `/getBs02`. |

## Files created by this package

```text
05_training/adapters/analyze_bis_api_integration_step99e.py
05_training/adapters/test_bis_api_integration_step99e.py
05_training/adapters/bis_api_integration_audit_step99e.md
```
