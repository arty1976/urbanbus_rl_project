$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$PyMain = ".\05_training\rewards\generate_concrete_reward_ablation_release_manifest_draft_step140.py"
$PyValidate = ".\05_training\rewards\validate_concrete_reward_ablation_release_manifest_draft_step140.py"
$PyTest = ".\05_training\rewards\test_concrete_reward_ablation_release_manifest_draft_step140.py"

if (-not (Test-Path $PyMain)) {
    throw "[STOP] Step 140 main python file not found: $PyMain. Run the Step 140 setup script once first."
}
if (-not (Test-Path $PyValidate)) {
    throw "[STOP] Step 140 validator python file not found: $PyValidate"
}
if (-not (Test-Path $PyTest)) {
    throw "[STOP] Step 140 test python file not found: $PyTest"
}

# Patch wrapper script if it exists.
$Wrapper = ".\step140_concrete_reward_ablation_release_manifest_draft.ps1"
if (Test-Path $Wrapper) {
    $Text = Get-Content $Wrapper -Raw

    # Remove empty-string native command arguments that PowerShell may not pass
    # reliably to Python argparse.
    $Text = $Text -replace '\s+--requested-by\s+""\s+`', ''
    $Text = $Text -replace '\s+--reviewed-by\s+""\s+`', ''
    $Text = $Text -replace '\s+--approved-by\s+""\s+`', ''
    $Text = $Text -replace '\s+--operator-approval-phrase\s+""\s+`', ''

    Set-Content -Path $Wrapper -Value $Text -Encoding UTF8
    Write-Host "[OK] Step 140 wrapper patched to omit empty argparse values"
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutputRoot = "artifacts/rewards/concrete_reward_ablation_release_manifest_draft_step140"
$Manifest = Join-Path $ProjectRoot "$OutputRoot\concrete_reward_ablation_release_manifest_draft_step140_manifest.json"
$Draft = Join-Path $ProjectRoot "$OutputRoot\concrete_actual_reward_ablation_release_manifest_draft_step140.json"

& $py $PyMain `
  --project-root "." `
  --output-root $OutputRoot `
  --result-output-root "artifacts/rewards/actual_reward_ablation_results"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 140 concrete release manifest draft generator failed after patch"
}

& $py $PyValidate --manifest $Manifest --draft $Draft
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 140 manifest/draft validation failed after patch"
}

& $py $PyTest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 140 self-test failed after patch"
}

Write-Host "[DONE] Step 140 empty-arg patch and rerun PASS."
