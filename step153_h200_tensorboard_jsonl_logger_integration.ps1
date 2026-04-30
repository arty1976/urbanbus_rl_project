$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$Py = "python"
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
}

$Generator = ".\05_training\observability\h200_tensorboard_jsonl_logger_step153.py"
$Validator = ".\05_training\observability\validate_h200_tensorboard_jsonl_logger_step153.py"
$SelfTest = ".\05_training\observability\test_h200_tensorboard_jsonl_logger_step153.py"
$OutputRoot = ".\artifacts\observability\h200_tensorboard_jsonl_logger_step153"
$Manifest = Join-Path $OutputRoot "h200_tensorboard_jsonl_logger_step153_manifest.json"

foreach ($Path in @($Generator, $Validator, $SelfTest)) {
    if (-not (Test-Path $Path)) {
        throw "[STOP] missing required file: $Path"
    }
}

Write-Host "[RUN] Step 153 logger scaffold generator"
& $Py $Generator `
    --project-root $ProjectRoot `
    --output-root $OutputRoot `
    --run-id "step153_sample_A_R0_seed001" `
    --condition-id "A" `
    --reward-id "R0" `
    --seed 1 `
    --dry-run-sample
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 153 generator failed"
}

Write-Host "[RUN] Step 153 validator"
& $Py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 153 validator failed"
}

Write-Host "[RUN] Step 153 self-test"
& $Py $SelfTest --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 153 self-test failed"
}

Write-Host "[OK] Step 153 H200 TensorBoard + JSONL logger integration completed"
Write-Host "[OK] manifest: $Manifest"
Write-Host "[OK] status  : LOGGER_INTEGRATION_READY_MONITORING_ONLY_STILL_LOCKED"
Write-Host "[OK] control : NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY"
