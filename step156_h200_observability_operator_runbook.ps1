$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$ObsDir = Join-Path $ProjectRoot "05_training\observability"
if (-not (Test-Path $ObsDir)) {
    New-Item -ItemType Directory -Force -Path $ObsDir | Out-Null
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $Py = "python"
}

Write-Host "[RUN] Step 156 operator runbook generator"
& $Py ".\05_training\observability\h200_observability_operator_runbook_step156.py" `
    --project-root $ProjectRoot `
    --output-root ".\artifacts\observability\h200_observability_operator_runbook_step156"
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 156 generator failed" }

$Manifest = ".\artifacts\observability\h200_observability_operator_runbook_step156\h200_observability_operator_runbook_step156_manifest.json"

Write-Host "[RUN] Step 156 validator"
& $Py ".\05_training\observability\validate_h200_observability_operator_runbook_step156.py" `
    --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 156 validator failed" }

Write-Host "[RUN] Step 156 self-test"
& $Py ".\05_training\observability\test_h200_observability_operator_runbook_step156.py"
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 156 self-test failed" }

Write-Host "[OK] Step 156 H200 observability operator runbook completed"
Write-Host "[OK] manifest: $Manifest"
