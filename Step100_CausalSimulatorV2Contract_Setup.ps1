param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [switch]$RunSelfTest,
    [switch]$RunActualIntegration
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$AdaptersDir = ".\05_training\adapters"
New-Item -ItemType Directory -Force -Path $AdaptersDir | Out-Null

$ProjectFilesDir = Join-Path $PSScriptRoot "project_files"
if (-not (Test-Path $ProjectFilesDir)) {
    throw "[STOP] project_files directory not found: $ProjectFilesDir"
}

Copy-Item -Path (Join-Path $ProjectFilesDir "causal_simulator_v2_input_contract.md") -Destination (Join-Path $AdaptersDir "causal_simulator_v2_input_contract.md") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "tensor_signal_availability_integration_v2.md") -Destination (Join-Path $AdaptersDir "tensor_signal_availability_integration_v2.md") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "build_causal_simulator_v2_contract_from_signal_features.py") -Destination (Join-Path $AdaptersDir "build_causal_simulator_v2_contract_from_signal_features.py") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "test_causal_simulator_v2_contract_from_signal_features.py") -Destination (Join-Path $AdaptersDir "test_causal_simulator_v2_contract_from_signal_features.py") -Force

Write-Host "[OK] Step 100 files copied:"
Write-Host " - .\05_training\adapters\causal_simulator_v2_input_contract.md"
Write-Host " - .\05_training\adapters\tensor_signal_availability_integration_v2.md"
Write-Host " - .\05_training\adapters\build_causal_simulator_v2_contract_from_signal_features.py"
Write-Host " - .\05_training\adapters\test_causal_simulator_v2_contract_from_signal_features.py"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

if ($RunSelfTest) {
    & $py -m py_compile `
      ".\05_training\adapters\build_causal_simulator_v2_contract_from_signal_features.py" `
      ".\05_training\adapters\test_causal_simulator_v2_contract_from_signal_features.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 100 py_compile failed"
    }

    & $py ".\05_training\adapters\test_causal_simulator_v2_contract_from_signal_features.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 100 self-test failed"
    }

    Write-Host "[DONE] Step 100 causal simulator v2 contract self-test complete."
}

if ($RunActualIntegration -or $RunSelfTest) {
    if (Test-Path ".\artifacts\signal_features_v2\node_signal_features.parquet") {
        & $py ".\05_training\adapters\build_causal_simulator_v2_contract_from_signal_features.py" `
          --signal-features-dir ".\artifacts\signal_features_v2" `
          --preflight-dir ".\artifacts\signal_features_v2_preflight" `
          --output-dir ".\artifacts\causal_simulator_v2_contract"

        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 100 actual integration failed"
        }

        Write-Host "[DONE] Step 100 actual causal simulator v2 contract integration complete."
    } else {
        Write-Host "[WARN] actual signal feature artifacts not found; skipped actual integration."
        Write-Host "[WARN] expected: .\artifacts\signal_features_v2\node_signal_features.parquet"
    }
}
