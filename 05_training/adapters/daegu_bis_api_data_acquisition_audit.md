# Step 99 — Daegu BIS API Data Acquisition Audit

## Position in the research pipeline

Step 99 is inserted after Step 97 Tensor DB data availability and generability audit and before the Causal Simulator v2 contract. The reason is simple: route-aware simulation cannot be promoted into the v2 required contract until route-stop sequence data is proven to be obtainable and matchable to the current graph node population.

## Step 99-A scope

**Step 99-A — /getBs02 route-stop sequence acquisition feasibility audit** checks whether the Daegu BIS API can recover:

- `route_id`
- `direction_id` or an equivalent field such as `move_dir_code` / direction label
- `ordered_stop_sequence`

This step is a feasibility audit only. It is not a data ingestion job and does not modify the DB.

## Why /getBs02 comes first

The existing graph skeleton was likely built mainly from route-link sequence data. That was enough to reconstruct STOP_TO_STOP edges, but the Causal Simulator v2 needs route-specific stop order. A route catalog CSV can provide route IDs, route numbers, route types, direction labels, and origin/destination stops, but it does not provide ordered stop sequence. Therefore `/getBs02` must be tested before route-aware simulator fields are promoted from missing to observed.

## Guardrails

- DB write is forbidden.
- Existing tensor DB overwrite is forbidden.
- 2023 boarding/alighting CSV reload is forbidden.
- API full collection is forbidden.
- Only sample route IDs, default and hard cap: 3 routes.
- API key must not be hardcoded.
- API key can be supplied through `--service-key`, `DAEGU_BIS_SERVICE_KEY`, or `DATAGO_SERVICE_KEY`.
- If no API key exists, the script must skip actual API calls and pass `--self-test` using an embedded schema sample.
- `paper_level_claim_allowed=false`.
- `causal_performance_claim_allowed=false`.
- `fleet_reduction_claim_allowed=false`.

## Files

- `05_training/adapters/inspect_getbs02_route_stop_sequence.py`
- `05_training/adapters/test_getbs02_route_stop_sequence_audit.py`
- `05_training/adapters/daegu_bis_api_requirements_step99.json`

## Artifacts

- `artifacts/daegu_bis_api_audit/getbs02_sample_response.json`
- `artifacts/daegu_bis_api_audit/getbs02_route_stop_sequence_audit_report.json`
- `artifacts/daegu_bis_api_audit/getbs02_route_stop_sequence_audit_report.md`
- `artifacts/daegu_bis_api_audit/getbs02_sample_response_route_<route_id>.raw`

## Field promotion criteria

| Field | Promotion candidate | Rule |
|---|---|---|
| `route_id` | observed candidate | Response includes a route ID-like field and can be linked to route catalog or DB. |
| `direction_id` | observed or partial observed candidate | Response includes direction-equivalent field such as `move_dir_code`, `direction`, or `방면`. |
| `ordered_stop_sequence` | observed candidate | Response includes both stop ID and stop order/sequence, and graph node coverage is acceptable. |

If stop ID exists but no sequence/order exists, `ordered_stop_sequence` stays missing. If the API returns a stop list with no documented or detectable order guarantee, `ordered_stop_sequence` also stays missing.

## Example commands

Self-test without API key:

```powershell
python .\05_training\adapters\inspect_getbs02_route_stop_sequence.py --self-test
python .\05_training\adapters\test_getbs02_route_stop_sequence_audit.py
```

Actual sample call, only after confirming route IDs and service key:

```powershell
$env:DAEGU_BIS_SERVICE_KEY = "YOUR_KEY"
python .\05_training\adapters\inspect_getbs02_route_stop_sequence.py `
  --route-id "SAMPLE_ROUTE_ID_1" `
  --route-id "SAMPLE_ROUTE_ID_2" `
  --max-routes 2 `
  --db-dsn $env:URBANBUS_DB_DSN
```

If the Swagger page shows a different endpoint URL or route parameter name, override them explicitly:

```powershell
python .\05_training\adapters\inspect_getbs02_route_stop_sequence.py `
  --base-url "https://apis.data.go.kr/.../getBs02" `
  --route-param "routeId" `
  --route-id "SAMPLE_ROUTE_ID"
```

## Step 99-A output interpretation

- If `ordered_stop_sequence_possible_any=true` and graph node coverage is high, prepare a Step 97 classification update document.
- If route and stop fields are present but sequence is missing, keep `ordered_stop_sequence=missing`.
- If coverage is low, keep route-aware simulator fields optional until ID mapping is resolved.
- Do not claim causal performance or Daegu-wide fleet reduction from this audit.
