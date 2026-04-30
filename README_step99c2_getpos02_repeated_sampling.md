# Step 99-C2 — /getPos02 Repeated Sampling Plan

This bundle documents Step 99-C-Final and provides a bounded repeated sampling runner for /getPos02.

## Files

```text
05_training/adapters/getpos02_live_position_classification_update_step99c.md
05_training/adapters/run_step99c2_getpos02_repeated_sampling.ps1
README_step99c2_getpos02_repeated_sampling.md
```

## Guardrails

- No DB writes.
- No Tensor DB overwrite.
- No CSV reload.
- No bulk streaming.
- Hard cap: max 5 routes, max 12 samples, max 60 API calls per run.
- Paper-level claims are forbidden.
- Causal performance claims are forbidden.

## Recommended short run

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"
$env:DAEGU_BIS_SERVICE_KEY = "YOUR_DECODING_KEY"

powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99c2_getpos02_repeated_sampling.ps1 `
  -RouteId "1000005000","1000001000","3000655000" `
  -MaxRoutes 3 `
  -Samples 3 `
  -IntervalSec 300
```

## Recommended 30-minute run

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99c2_getpos02_repeated_sampling.ps1 `
  -RouteId "1000005000","1000001000","3000655000" `
  -MaxRoutes 3 `
  -Samples 6 `
  -IntervalSec 300
```

This performs 18 API calls: 3 routes x 6 samples.

## Output

```text
artifacts/daegu_bis_api_audit/getpos02_repeated_sampling/<run_id>/
  step99c2_repeated_sampling_manifest.csv
  step99c2_repeated_sampling_report.md
  sample_001/
  sample_002/
  ...
```

Each sample folder contains the normal Step 99-C report and raw response files.
