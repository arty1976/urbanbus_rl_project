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

$TestPath = ".\05_training\rewards\test_explicit_operator_release_manifest_draft_step151a.py"
$GenPath = ".\05_training\rewards\explicit_operator_release_manifest_draft_step151a.py"
$ValPath = ".\05_training\rewards\validate_explicit_operator_release_manifest_draft_step151a.py"

& $py $TestPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-A self-test failed"
}

$Step150Manifest = ".\artifacts\rewards\actual_reward_ablation_operator_release_checklist_step150\actual_reward_ablation_operator_release_checklist_step150_manifest.json"
if (-not (Test-Path $Step150Manifest)) {
    throw "[STOP] Step 150 manifest not found: $Step150Manifest"
}

& $py $GenPath `
  --project-root "." `
  --step150-manifest $Step150Manifest `
  --output-root ".\artifacts\rewards\explicit_operator_release_manifest_draft_step151a"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-A manifest draft generation failed"
}

$Step151Manifest = ".\artifacts\rewards\explicit_operator_release_manifest_draft_step151a\explicit_operator_release_manifest_draft_step151a_manifest.json"

& $py $ValPath --manifest $Step151Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 151-A manifest validation failed"
}

Write-Host "[DONE] Step 151-A explicit operator release manifest draft complete."
Write-Host "[OK] manifest: $Step151Manifest"
