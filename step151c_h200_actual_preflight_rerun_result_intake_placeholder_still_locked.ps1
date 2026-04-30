$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$TestPath = ".\05_training\rewards\test_h200_actual_preflight_rerun_result_intake_placeholder_step151c.py"
$GenPath = ".\05_training\rewards\h200_actual_preflight_rerun_result_intake_placeholder_step151c.py"
$ValPath = ".\05_training\rewards\validate_h200_actual_preflight_rerun_result_intake_placeholder_step151c.py"

& $py $TestPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-C self-test failed"
}

$Step151BManifest = ".\artifacts\rewards\h200_expected_preflight_rerun_checklist_step151b\h200_expected_preflight_rerun_checklist_step151b_manifest.json"
if (-not (Test-Path $Step151BManifest)) {
    throw "[STOP] Step 151-B manifest not found: $Step151BManifest"
}

$H200Step149Manifest = ".\artifacts\rewards\h200_environment_preflight_result_manifest_step149_h200_actual\h200_environment_preflight_result_manifest_step149.json"

& $py $GenPath `
  --project-root "." `
  --step151b-manifest $Step151BManifest `
  --h200-step149-manifest $H200Step149Manifest `
  --min-gpu-count 1 `
  --output-root ".\artifacts\rewards\h200_actual_preflight_rerun_result_intake_placeholder_step151c"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-C intake placeholder generation failed"
}

$Step151CManifest = ".\artifacts\rewards\h200_actual_preflight_rerun_result_intake_placeholder_step151c\h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json"

& $py $ValPath --manifest $Step151CManifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-C manifest validation failed"
}

Write-Host "[DONE] Step 151-C H200 actual preflight rerun result intake placeholder complete."
Write-Host "[OK] manifest: $Step151CManifest"
