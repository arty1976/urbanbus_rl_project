param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [switch]$RunSelfTest,
    [switch]$RunActualCsvPreflight
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$AdaptersDir = ".\05_training\adapters"
New-Item -ItemType Directory -Force -Path $AdaptersDir | Out-Null

$ProjectFilesDir = Join-Path $PSScriptRoot "project_files"
$SourceCsvInPackage = Join-Path $PSScriptRoot "data\대구광역시_신호등_20251231.csv"
$ProjectSignalDir = ".\data\source\daegu_signal"
$ProjectSignalCsv = Join-Path $ProjectSignalDir "대구광역시_신호등_20251231.csv"

if (-not (Test-Path $ProjectFilesDir)) {
    throw "[STOP] project_files directory not found in package: $ProjectFilesDir"
}

Copy-Item -Path (Join-Path $ProjectFilesDir "daegu_signal_csv_preflight_v2.md") -Destination (Join-Path $AdaptersDir "daegu_signal_csv_preflight_v2.md") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "run_daegu_signal_csv_preflight_v2.py") -Destination (Join-Path $AdaptersDir "run_daegu_signal_csv_preflight_v2.py") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "patch_signal_feature_builder_for_daegu_csv_v2.py") -Destination (Join-Path $AdaptersDir "patch_signal_feature_builder_for_daegu_csv_v2.py") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "test_daegu_signal_csv_preflight_v2.py") -Destination (Join-Path $AdaptersDir "test_daegu_signal_csv_preflight_v2.py") -Force

New-Item -ItemType Directory -Force -Path $ProjectSignalDir | Out-Null
if (Test-Path $SourceCsvInPackage) {
    Copy-Item -Path $SourceCsvInPackage -Destination $ProjectSignalCsv -Force
    Write-Host "[OK] copied signal CSV to: $ProjectSignalCsv"
} else {
    Write-Host "[WARN] package CSV not found: $SourceCsvInPackage"
    Write-Host "[WARN] place your CSV at: $ProjectSignalCsv"
}

Write-Host "[OK] Step 99 files copied:"
Write-Host " - .\05_training\adapters\daegu_signal_csv_preflight_v2.md"
Write-Host " - .\05_training\adapters\run_daegu_signal_csv_preflight_v2.py"
Write-Host " - .\05_training\adapters\patch_signal_feature_builder_for_daegu_csv_v2.py"
Write-Host " - .\05_training\adapters\test_daegu_signal_csv_preflight_v2.py"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

& $py ".\05_training\adapters\patch_signal_feature_builder_for_daegu_csv_v2.py"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99 builder patch failed"
}

if ($RunSelfTest) {
    & $py -m py_compile `
      ".\05_training\adapters\run_daegu_signal_csv_preflight_v2.py" `
      ".\05_training\adapters\patch_signal_feature_builder_for_daegu_csv_v2.py" `
      ".\05_training\adapters\test_daegu_signal_csv_preflight_v2.py" `
      ".\05_training\adapters\build_signal_features_for_causal_simulator_v2.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 99 py_compile failed"
    }

    & $py ".\05_training\adapters\test_daegu_signal_csv_preflight_v2.py"
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 99 self-test failed"
    }

    Write-Host "[DONE] Step 99 Daegu signal CSV preflight self-test complete."
}

if ($RunActualCsvPreflight -or $RunSelfTest) {
    if (Test-Path $ProjectSignalCsv) {
        & $py ".\05_training\adapters\run_daegu_signal_csv_preflight_v2.py" `
          --signal-csv $ProjectSignalCsv `
          --output-dir ".\artifacts\signal_features_v2_preflight"

        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 99 actual Daegu signal CSV preflight failed"
        }

        Write-Host "[DONE] Step 99 actual Daegu signal CSV preflight complete."
    } else {
        Write-Host "[WARN] actual signal CSV not found, skipped actual preflight: $ProjectSignalCsv"
    }
}
