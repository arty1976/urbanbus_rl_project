$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$Py = "python"
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
}

$Generator = ".\05_training\observability\h200_observability_dashboard_index_step155.py"
$Validator = ".\05_training\observability\validate_h200_observability_dashboard_index_step155.py"
$SelfTest = ".\05_training\observability\test_h200_observability_dashboard_index_step155.py"
$OutputRoot = ".\artifacts\observability\h200_observability_dashboard_index_step155"
$Manifest = Join-Path $OutputRoot "h200_observability_dashboard_index_step155_manifest.json"

Write-Host "[RUN] Step 155 dashboard index generator"
& $Py $Generator `
  --project-root $ProjectRoot `
  --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 155 generator failed"
}

Write-Host "[RUN] Step 155 dashboard index validator"
& $Py $Validator `
  --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 155 validator failed"
}

Write-Host "[RUN] Step 155 self-test"
& $Py $SelfTest `
  --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 155 self-test failed"
}

Write-Host "[OK] Step 155 H200 observability dashboard index/status page completed"
Write-Host "[OK] manifest: $Manifest"
