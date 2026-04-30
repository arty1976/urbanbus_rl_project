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

$ProjectFilesDir = Join-Path $PSScriptRoot "project_files"
if (-not (Test-Path $ProjectFilesDir)) {
    throw "[STOP] project_files directory not found: $ProjectFilesDir. Run this from the unzipped package folder."
}

$AdaptersDir = ".\05_training\adapters"
New-Item -ItemType Directory -Force -Path $AdaptersDir | Out-Null

Copy-Item -Path (Join-Path $ProjectFilesDir "causal_simulator_v2_rollout_writer_contract.md") -Destination (Join-Path $AdaptersDir "causal_simulator_v2_rollout_writer_contract.md") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "test_causal_simulator_v2_rollout_smoke.py") -Destination (Join-Path $AdaptersDir "test_causal_simulator_v2_rollout_smoke.py") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "run_causal_simulator_v2_rollout_smoke.py") -Destination ".\05_training\run_causal_simulator_v2_rollout_smoke.py" -Force

Write-Host "[OK] Step 102 files copied:"
Write-Host " - .\05_training\adapters\causal_simulator_v2_rollout_writer_contract.md"
Write-Host " - .\05_training\adapters\test_causal_simulator_v2_rollout_smoke.py"
Write-Host " - .\05_training\run_causal_simulator_v2_rollout_smoke.py"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

if ($RunSelfTest) {
    & $py -m py_compile `
      ".\05_training\run_causal_simulator_v2_rollout_smoke.py" `
      ".\05_training\adapters\test_causal_simulator_v2_rollout_smoke.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 102 py_compile failed"
    }

    Push-Location ".\05_training\adapters"
    try {
        & $py ".\test_causal_simulator_v2_rollout_smoke.py"
    } finally {
        Pop-Location
    }

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 102 self-test failed"
    }

    Write-Host "[DONE] Step 102 rollout writer self-test complete."
}

if ($RunActualSmoke -or $RunSelfTest) {
    if (
        (Test-Path ".\artifacts\causal_simulator_v2_contract\causal_simulator_v2_input_contract.json") -and
        (Test-Path ".\artifacts\signal_features_v2\node_signal_features.parquet") -and
        (Test-Path ".\artifacts\signal_features_v2\edge_signal_features.parquet")
    ) {
        & $py ".\05_training\run_causal_simulator_v2_rollout_smoke.py" `
          --project-root "." `
          --conditions "C2_STATIC_SIGNAL" `
          --seeds "101" `
          --num-agents 8 `
          --episode-steps 4 `
          --baseline-bus-count 8 `
          --output-root ".\artifacts\causal_simulator_v2_rollout_smoke"

        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 102 actual rollout smoke failed"
        }

        Write-Host "[DONE] Step 102 actual rollout smoke complete."
    } else {
        Write-Host "[WARN] Step 99/100/101 artifacts not found; skipped actual rollout smoke."
    }
}
