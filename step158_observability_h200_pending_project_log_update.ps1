$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$Py = "python"
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
}

Write-Host "[RUN] Step 158 project log update generator"
& $Py ".\05_training\observability\observability_h200_pending_project_log_update_step158.py" `
    --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 158 generator failed"
}

$Manifest = ".\artifacts\observability\observability_h200_pending_project_log_update_step158\observability_h200_pending_project_log_update_step158_manifest.json"

Write-Host "[RUN] Step 158 validator"
& $Py ".\05_training\observability\validate_observability_h200_pending_project_log_update_step158.py" `
    --project-root $ProjectRoot `
    --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 158 validator failed"
}

Write-Host "[RUN] Step 158 self-test"
& $Py ".\05_training\observability\test_observability_h200_pending_project_log_update_step158.py" `
    --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 158 self-test failed"
}

Write-Host "[OK] Step 158 observability + H200 pending project log update completed"
Write-Host "[OK] manifest: $Manifest"
