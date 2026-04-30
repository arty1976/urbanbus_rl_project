param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [string]$SignalCsv = ".\data\source\daegu_signal\대구광역시_신호등_20251231.csv",
    [string]$OutputDir = ".\artifacts\signal_features_v2",
    [string]$DbUrl = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (-not (Test-Path $SignalCsv)) {
    throw "[STOP] Signal CSV not found: $SignalCsv"
}

if ([string]::IsNullOrWhiteSpace($DbUrl)) {
    $DbUrl = $env:URBANBUS_DB_DSN
}

if ([string]::IsNullOrWhiteSpace($DbUrl)) {
    throw "[STOP] DbUrl is empty. Set URBANBUS_DB_DSN or pass -DbUrl."
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py ".\05_training\adapters\patch_signal_feature_builder_for_daegu_csv_v2.py"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99 builder patch failed"
}

& $py ".\05_training\adapters\build_signal_features_for_causal_simulator_v2.py" `
  --signal-csv $SignalCsv `
  --db-url $DbUrl `
  --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99 DB-based signal feature build failed"
}

Write-Host "[OK] Step 99 DB-based signal feature build complete."
Write-Host "[OK] output dir = $OutputDir"
