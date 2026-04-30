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

$TestPath = ".\05_training\rewards\test_h200_expected_preflight_rerun_checklist_step151b.py"
$GenPath = ".\05_training\rewards\h200_expected_preflight_rerun_checklist_step151b.py"
$ValPath = ".\05_training\rewards\validate_h200_expected_preflight_rerun_checklist_step151b.py"

& $py $TestPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-B self-test failed"
}

$Step151AManifest = ".\artifacts\rewards\explicit_operator_release_manifest_draft_step151a\explicit_operator_release_manifest_draft_step151a_manifest.json"
if (-not (Test-Path $Step151AManifest)) {
    throw "[STOP] Step 151-A manifest not found: $Step151AManifest"
}

$Step149LocalManifest = ".\artifacts\rewards\h200_environment_preflight_result_manifest_step149\h200_environment_preflight_result_manifest_step149.json"
if (-not (Test-Path $Step149LocalManifest)) {
    Write-Host "[WARN] Step 149 local manifest not found: $Step149LocalManifest"
    Write-Host "[WARN] Step 151-B can continue, but traceability warning will be recorded."
}

& $py $GenPath `
  --project-root "." `
  --step151a-manifest $Step151AManifest `
  --step149-local-manifest $Step149LocalManifest `
  --output-root ".\artifacts\rewards\h200_expected_preflight_rerun_checklist_step151b"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-B checklist generation failed"
}

$Step151BManifest = ".\artifacts\rewards\h200_expected_preflight_rerun_checklist_step151b\h200_expected_preflight_rerun_checklist_step151b_manifest.json"

& $py $ValPath --manifest $Step151BManifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-B manifest validation failed"
}

Write-Host "[DONE] Step 151-B H200 expected preflight rerun checklist complete."
Write-Host "[OK] manifest: $Step151BManifest"
