$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$Py = ".\05_training\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

$Update = ".\05_training\patent_evidence\zero_loss_patent_evidence_project_log_update_step168.py"
$Validate = ".\05_training\patent_evidence\validate_zero_loss_patent_evidence_project_log_update_step168.py"
$Test = ".\05_training\patent_evidence\test_zero_loss_patent_evidence_project_log_update_step168.py"
$OutputRoot = ".\artifacts\patent_evidence\zero_loss_patent_evidence_project_log_update_step168"
$Manifest = Join-Path $OutputRoot "project_log_update_step168_manifest.json"

Write-Host "[INFO] Step 168 Zero-Loss patent evidence project log update should be copied into:"
Write-Host "       $ProjectRoot\05_training\patent_evidence"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\zero_loss_patent_evidence_project_log_update_step168.md"
Write-Host "  05_training\patent_evidence\zero_loss_patent_evidence_project_log_update_step168.py"
Write-Host "  05_training\patent_evidence\validate_zero_loss_patent_evidence_project_log_update_step168.py"
Write-Host "  05_training\patent_evidence\test_zero_loss_patent_evidence_project_log_update_step168.py"
Write-Host ""

foreach ($Path in @($Update, $Validate, $Test)) {
    if (-not (Test-Path $Path)) {
        throw "[STOP] Missing expected file: $Path"
    }
}

& $Py $Update --project-root . --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 168 project log update failed"
}

& $Py $Validate --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 168 project log update validation failed"
}

& $Py $Test
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 168 unittest failed"
}

Write-Host "[OK] Step 168 Zero-Loss patent evidence project log update self-test PASS"
Write-Host "[OK] artifact: $OutputRoot"
Write-Host "[DONE] Step 168 complete. Project log updated; non-claim guards remain locked."
