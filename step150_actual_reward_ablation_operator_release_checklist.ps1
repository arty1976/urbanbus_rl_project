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

$TestPath = ".\05_training\rewards\test_actual_reward_ablation_operator_release_checklist_step150.py"
$PyPath = ".\05_training\rewards\actual_reward_ablation_operator_release_checklist_step150.py"
$ValPath = ".\05_training\rewards\validate_actual_reward_ablation_operator_release_checklist_step150.py"
$OutRoot = ".\artifacts\rewards\actual_reward_ablation_operator_release_checklist_step150"

if (-not (Test-Path $TestPath)) {
    throw "[STOP] Test file not found: $TestPath"
}
if (-not (Test-Path $PyPath)) {
    throw "[STOP] Generator file not found: $PyPath"
}
if (-not (Test-Path $ValPath)) {
    throw "[STOP] Validator file not found: $ValPath"
}

& $py $TestPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 150 self-test failed"
}

& $py $PyPath `
  --project-root "." `
  --output-root $OutRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 150 checklist generation failed"
}

$Manifest = "$OutRoot\actual_reward_ablation_operator_release_checklist_step150_manifest.json"

& $py $ValPath --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 150 checklist manifest validation failed"
}

Write-Host "[DONE] Step 150 actual reward ablation operator release checklist complete."
Write-Host "[OK] manifest: $Manifest"
