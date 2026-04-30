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

Copy-Item -Path (Join-Path $ProjectFilesDir "causal_simulator_v2_canonical_kpi_smoke_contract.md") -Destination (Join-Path $AdaptersDir "causal_simulator_v2_canonical_kpi_smoke_contract.md") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "test_causal_simulator_v2_canonical_kpi_smoke.py") -Destination (Join-Path $AdaptersDir "test_causal_simulator_v2_canonical_kpi_smoke.py") -Force
Copy-Item -Path (Join-Path $ProjectFilesDir "run_causal_simulator_v2_canonical_kpi_smoke.py") -Destination ".\05_training\run_causal_simulator_v2_canonical_kpi_smoke.py" -Force

Write-Host "[OK] Step 103 files copied:"
Write-Host " - .\05_training\adapters\causal_simulator_v2_canonical_kpi_smoke_contract.md"
Write-Host " - .\05_training\adapters\test_causal_simulator_v2_canonical_kpi_smoke.py"
Write-Host " - .\05_training\run_causal_simulator_v2_canonical_kpi_smoke.py"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

if ($RunSelfTest) {
    & $py -m py_compile `
      ".\05_training\run_causal_simulator_v2_canonical_kpi_smoke.py" `
      ".\05_training\adapters\test_causal_simulator_v2_canonical_kpi_smoke.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 103 py_compile failed"
    }

    Push-Location ".\05_training\adapters"
    try {
        & $py ".\test_causal_simulator_v2_canonical_kpi_smoke.py"
    } finally {
        Pop-Location
    }

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 103 self-test failed"
    }

    Write-Host "[DONE] Step 103 canonical KPI smoke self-test complete."
}

if ($RunActualSmoke -or $RunSelfTest) {
    if (Test-Path ".\artifacts\causal_simulator_v2_rollout_smoke") {
        & $py ".\05_training\run_causal_simulator_v2_canonical_kpi_smoke.py" `
          --project-root "." `
          --input-root ".\artifacts\causal_simulator_v2_rollout_smoke" `
          --output-root ".\artifacts\causal_simulator_v2_canonical_kpi_smoke"

        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 103 actual canonical KPI smoke failed"
        }

        Write-Host "[DONE] Step 103 actual canonical KPI smoke complete."
    } else {
        Write-Host "[WARN] Step 102 rollout artifact not found; skipped actual canonical KPI smoke."
    }
}
