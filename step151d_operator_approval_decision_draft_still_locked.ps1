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

$TestPath = ".\05_training\rewards\test_operator_approval_decision_draft_step151d.py"
$GenPath = ".\05_training\rewards\operator_approval_decision_draft_step151d.py"
$ValPath = ".\05_training\rewards\validate_operator_approval_decision_draft_step151d.py"

& $py $TestPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-D self-test failed"
}

$Step151CManifest = ".\artifacts\rewards\h200_actual_preflight_rerun_result_intake_placeholder_step151c\h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json"
if (-not (Test-Path $Step151CManifest)) {
    throw "[STOP] Step 151-C manifest not found: $Step151CManifest"
}

& $py $GenPath `
  --project-root "." `
  --step151c-manifest $Step151CManifest `
  --output-root ".\artifacts\rewards\operator_approval_decision_draft_step151d"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-D operator approval decision draft generation failed"
}

$Step151DManifest = ".\artifacts\rewards\operator_approval_decision_draft_step151d\operator_approval_decision_draft_step151d_manifest.json"

& $py $ValPath --manifest $Step151DManifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-D manifest validation failed"
}

Write-Host "[DONE] Step 151-D operator approval decision draft complete."
Write-Host "[OK] manifest: $Step151DManifest"
