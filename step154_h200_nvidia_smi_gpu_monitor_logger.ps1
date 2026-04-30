$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$Py = "python"
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = ".\05_training\.venv\Scripts\python.exe"
}

$Generator = ".\05_training\observability\h200_nvidia_smi_gpu_monitor_step154.py"
$Validator = ".\05_training\observability\validate_h200_nvidia_smi_gpu_monitor_step154.py"
$SelfTest = ".\05_training\observability\test_h200_nvidia_smi_gpu_monitor_step154.py"
$OutRoot = ".\artifacts\observability\h200_nvidia_smi_gpu_monitor_step154"

if (-not (Test-Path $Generator)) {
    throw "[FAIL] Step 154 generator not found: $Generator"
}
if (-not (Test-Path $Validator)) {
    throw "[FAIL] Step 154 validator not found: $Validator"
}
if (-not (Test-Path $SelfTest)) {
    throw "[FAIL] Step 154 self-test not found: $SelfTest"
}

Write-Host "[RUN] Step 154 nvidia-smi GPU monitor scaffold generator"
& $Py $Generator --project-root $ProjectRoot --output-root $OutRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 154 generator failed"
}

$Manifest = Join-Path $OutRoot "h200_nvidia_smi_gpu_monitor_step154_manifest.json"
if (-not (Test-Path $Manifest)) {
    throw "[FAIL] Step 154 manifest was not created: $Manifest"
}

Write-Host "[RUN] Step 154 validator"
& $Py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 154 validator failed"
}

Write-Host "[RUN] Step 154 self-test"
& $Py $SelfTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 154 self-test failed"
}

Write-Host "[OK] Step 154 H200 nvidia-smi GPU monitor logger scaffold completed"
Write-Host "[OK] manifest: $Manifest"

$ManifestJson = Get-Content $Manifest -Raw | ConvertFrom-Json
Write-Host "[OK] status  : $($ManifestJson.monitor_status)"
Write-Host "[OK] mode    : $($ManifestJson.monitor_mode)"
Write-Host "[OK] control : $($ManifestJson.control_policy)"
Write-Host "[OK] nvidia  : $($ManifestJson.nvidia_smi_status)"
