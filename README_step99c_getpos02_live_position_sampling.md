# Step 99-C /getPos02 Live Bus Position Sampling Audit

This bundle adds a safe sampling-only audit for Daegu BIS `/getPos02`.

## Purpose

- Determine whether live vehicle-position data can provide `bus_id`, `route_id`, `direction_id`, and current vehicle location fields.
- Capture raw responses and normalized preview rows only.
- Do not write to DB.
- Do not overwrite tensor DB.
- Do not make performance or causal claims.

## Files

```text
05_training/adapters/inspect_getpos02_live_position_sampling.py
05_training/adapters/run_step99c_getpos02_live_position_sampling.ps1
05_training/adapters/test_getpos02_live_position_sampling.py
05_training/adapters/getbs02_getbasic02_classification_update_step99.md
```

## Self-test

```powershell
python .\05_training\adapters\test_getpos02_live_position_sampling.py
```

## Sample API call during operating hours

```powershell
$env:DAEGU_BIS_SERVICE_KEY = "YOUR_DECODING_KEY"

powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99c_getpos02_live_position_sampling.ps1 `
  -RouteId "1000005000","1000001000","3000655000" `
  -MaxRoutes 3
```

If Swagger shows extra parameters, pass them as `-ExtraParam`:

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99c_getpos02_live_position_sampling.ps1 `
  -RouteId "1000005000" `
  -MaxRoutes 1 `
  -ExtraParam "resultType=json" `
  -ExtraParam "pageNo=1" `
  -ExtraParam "numOfRows=500"
```

If Swagger shows a different base URL:

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99c_getpos02_live_position_sampling.ps1 `
  -BaseUrl "https://apis.data.go.kr/6270000/dbmsapi02/getPos02" `
  -RouteId "1000005000" `
  -MaxRoutes 1
```

## Expected interpretation

- `SUCCESS_WITH_ROWS`: live vehicle rows were returned and normalized.
- `NO_ITEMS`: API call succeeded, but no vehicle rows were present.
- `PROVIDER_ERROR`: provider returned error text such as `Unexpected errors`.
- `AUTH_ERROR`: authentication/key error markers detected.
- `SCHEMA_UNRECOGNIZED`: rows exist but parser cannot recognize vehicle/position fields.

For off-hours, `NO_ITEMS` should not be treated as a data-source failure.
