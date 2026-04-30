# Step 99-B — /getBasic02 master snapshot audit

Purpose: inspect `/getBasic02` as a master/basic snapshot source before Step 98 causal simulator v2 contract.

Guardrails:
- No DB write
- No tensor DB overwrite
- No CSV reload
- API key is read from `DAEGU_BIS_SERVICE_KEY` or `DATAGO_SERVICE_KEY`
- If no API key is present, the script runs embedded self-test/schema-only mode
- This is not a paper-level or causal-performance claim

## Files

```text
05_training/adapters/inspect_getbasic02_master_snapshot.py
05_training/adapters/run_step99b_getbasic02_master_audit.ps1
05_training/adapters/test_getbasic02_master_snapshot_audit.py
README_step99b_getbasic02_master_audit.md
```

## Self-test

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"
python .\05_training\adapters\test_getbasic02_master_snapshot_audit.py
```

## Actual sample audit

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"
$env:DAEGU_BIS_SERVICE_KEY = "YOUR_DECODING_KEY"
$env:URBANBUS_DB_DSN = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/urbanbus"

powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99b_getbasic02_master_audit.ps1
```

If Swagger shows optional response parameters, pass them as extra params:

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99b_getbasic02_master_audit.ps1 `
  -ExtraParam "resultType=json" `
  -ExtraParam "pageNo=1" `
  -ExtraParam "numOfRows=1000"
```

## Outputs

```text
artifacts/daegu_bis_api_audit/getbasic02_master_snapshot_audit/
  getbasic02_master_snapshot.raw
  getbasic02_master_snapshot_normalized.csv
  getbasic02_master_snapshot_audit_report.json
  getbasic02_master_snapshot_audit_report.md
```

## Expected interpretation

- If route_id/route_no/route_type/direction fields are present, they can become `observed_candidate` master fields.
- If the response is provider/auth/no-items, keep Step 99-B as `deferred` and do not claim the fields are missing.
- Step 99-A already confirmed route-stop sequence from `/getBs02`; Step 99-B is for master/basic catalog enrichment.
