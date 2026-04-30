param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [switch]$RunSelfTest,
    [switch]$RunActualSmoke
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
    throw "[STOP] project_files directory not found: $ProjectFilesDir. Run this from the unzipped package folder."
}

Copy-Item -Path (Join-Path $ProjectFilesDir "causal_simulator_v2_adapter_contract.md") -Destination (Join-Path $AdaptersDir "causal_simulator_v2_adapter_contract.md") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "causal_simulator_v2_adapter.py") -Destination (Join-Path $AdaptersDir "causal_simulator_v2_adapter.py") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "test_causal_simulator_v2_adapter.py") -Destination (Join-Path $AdaptersDir "test_causal_simulator_v2_adapter.py") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "run_causal_simulator_v2_adapter_smoke.py") -Destination ".\05_training\run_causal_simulator_v2_adapter_smoke.py" -Force

Write-Host "[OK] Step 101 files copied"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

if ($RunSelfTest) {
    & $py -m py_compile `
      ".\05_training\adapters\causal_simulator_v2_adapter.py" `
      ".\05_training\adapters\test_causal_simulator_v2_adapter.py" `
      ".\05_training\run_causal_simulator_v2_adapter_smoke.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 101 py_compile failed"
    }

    Push-Location ".\05_training\adapters"
    try {
        & $py ".\test_causal_simulator_v2_adapter.py"
    } finally {
        Pop-Location
    }

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 101 self-test failed"
    }

    Write-Host "[DONE] Step 101 CausalSimulatorAdapter v2 scaffold self-test complete."
}

if ($RunActualSmoke -or $RunSelfTest) {
    if (
        (Test-Path ".\artifacts\causal_simulator_v2_contract\causal_simulator_v2_input_contract.json") -and
        (Test-Path ".\artifacts\signal_features_v2\node_signal_features.parquet") -and
        (Test-Path ".\artifacts\signal_features_v2\edge_signal_features.parquet")
    ) {
        & $py ".\05_training\run_causal_simulator_v2_adapter_smoke.py" `
          --project-root "." `
          --num-agents 8 `
          --episode-steps 4 `
          --seed 101 `
          --output-json ".\artifacts\causal_simulator_v2_adapter_smoke\status.json"

        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 101 actual artifact smoke failed"
        }

        Write-Host "[DONE] Step 101 actual artifact adapter smoke complete."
    } else {
        Write-Host "[WARN] Step 99/100 artifacts not found; skipped actual adapter smoke."
    }
}
