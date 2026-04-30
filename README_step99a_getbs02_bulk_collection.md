# Step 99-A Full `/getBs02` Route-Stop Sequence Collection

This bundle follows the previous `bus_route_link_bulk_collect_orchestrator` skill rules:

- Use Decoding `serviceKey` only.
- Extract `route_id` dynamically from `public.stg_daegu_routes`.
- Preserve raw response snapshots.
- Write a manifest with separated `collect_status` and `load_status`.
- Do not write to DB.
- Do not overwrite tensor DB.
- Do not reload CSV.

## Install

Extract this bundle at the project root:

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"
Expand-Archive -Path "$env:USERPROFILE\Downloads\step99a_getbs02_bulk_collection_bundle.zip" -DestinationPath . -Force
```

## Required environment

```powershell
$env:DAEGU_BIS_SERVICE_KEY = "YOUR_DECODING_KEY"
$env:URBANBUS_DB_DSN = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/urbanbus"
```

## Smoke bulk test, 5 routes

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99a_getbs02_bulk_collect.ps1 -MaxRoutes 5 -SleepSec 0.25
```

## Full source-route collection

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99a_getbs02_bulk_collect.ps1 -MaxRoutes 0 -SleepSec 0.25
```

## Outputs

Under:

```text
artifacts/daegu_bis_api_audit/getbs02_bulk_collect/<run_id>/
```

Files:

- `raw/getbs02_<route_id>.json`
- `manifest.csv`
- `getbs02_route_stop_sequence_normalized.csv`
- `route_summary.csv`
- `getbs02_full_collection_report.json`
- `getbs02_full_collection_report.md`

## After run

```powershell
Get-ChildItem .\artifacts\daegu_bis_api_audit\getbs02_bulk_collect | Sort-Object LastWriteTime -Descending | Select-Object -First 1 FullName
```

Then inspect the latest report:

```powershell
Get-Content "<latest_run>\getbs02_full_collection_report.md" -Encoding UTF8
Import-Csv "<latest_run>\manifest.csv" | Group-Object collect_status | Select-Object Name,Count
```
