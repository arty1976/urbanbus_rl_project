$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$ObsDir = Join-Path $ProjectRoot "05_training\observability"
if (-not (Test-Path $ObsDir)) {
    throw "[STOP] observability folder not found: $ObsDir"
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $Py = "python"
}

$OutputRoot = Join-Path $ProjectRoot "artifacts\observability\h200_monitoring_dashboard_contract_step152"

Write-Host "[RUN] Step 152 generator"
& $Py ".\05_training\observability\h200_experiment_observability_logger_step152.py" `
    --project-root $ProjectRoot `
    --output-root $OutputRoot `
    --run-id "step152_selftest_run" `
    --condition-id "A" `
    --reward-id "R0" `
    --seed 1 `
    --write-sample-events

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 152 generator failed"
}

$Manifest = Join-Path $OutputRoot "h200_monitoring_dashboard_contract_step152_manifest.json"

Write-Host "[RUN] Step 152 validator"
& $Py ".\05_training\observability\validate_h200_monitoring_dashboard_contract_step152.py" `
    --manifest $Manifest

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 152 validator failed"
}

Write-Host "[RUN] Step 152 self-test"
& $Py ".\05_training\observability\test_h200_monitoring_dashboard_contract_step152.py"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 152 self-test failed"
}

Write-Host "[OK] Step 152 H200 monitoring dashboard contract completed"
Write-Host "[OK] manifest: $Manifest"
Write-Host "[OK] status  : OBSERVABILITY_CONTRACT_READY_MONITORING_ONLY_STILL_LOCKED"
Write-Host "[OK] control : NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
