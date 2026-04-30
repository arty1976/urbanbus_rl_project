param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [string]$InputRoot = ".\artifacts\causal_simulator_v2_rollout_smoke",
    [string]$OutputRoot = ".\artifacts\causal_simulator_v2_canonical_kpi_smoke"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py ".\05_training\run_causal_simulator_v2_canonical_kpi_smoke.py" `
  --project-root "." `
  --input-root $InputRoot `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 103 actual canonical KPI smoke failed"
}

Write-Host "[OK] Step 103 actual canonical KPI smoke complete."
Write-Host "[OK] output_root = $OutputRoot"
