param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [string]$SignalCsv = ".\data\source\daegu_signal\대구광역시_신호등_20251231.csv",
    [string]$OutputDir = ".\artifacts\signal_features_v2_preflight"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (-not (Test-Path $SignalCsv)) {
    throw "[STOP] Signal CSV not found: $SignalCsv"
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py ".\05_training\adapters\run_daegu_signal_csv_preflight_v2.py" `
  --signal-csv $SignalCsv `
  --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99 actual Daegu signal CSV preflight failed"
}

Write-Host "[OK] Step 99 actual Daegu signal CSV preflight complete."
Write-Host "[OK] output dir = $OutputDir"
