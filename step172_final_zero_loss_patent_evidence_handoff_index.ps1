$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$Py = Join-Path $ProjectRoot "05_training\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

$ScriptDir = Join-Path $ProjectRoot "05_training\patent_evidence"
$Generator = Join-Path $ScriptDir "final_zero_loss_patent_evidence_handoff_index_step172.py"
$Validator = Join-Path $ScriptDir "validate_final_zero_loss_patent_evidence_handoff_index_step172.py"
$Test = Join-Path $ScriptDir "test_final_zero_loss_patent_evidence_handoff_index_step172.py"
$OutRoot = "artifacts\patent_evidence\final_zero_loss_patent_evidence_handoff_index_step172_selftest"
$Manifest = Join-Path $ProjectRoot (Join-Path $OutRoot "final_zero_loss_patent_evidence_handoff_index_step172.json")

Write-Host "[INFO] Step 172 Final Zero-Loss Patent Evidence Handoff Index should be copied into:"
Write-Host "       $ScriptDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\final_zero_loss_patent_evidence_handoff_index_step172.md"
Write-Host "  05_training\patent_evidence\final_zero_loss_patent_evidence_handoff_index_step172.py"
Write-Host "  05_training\patent_evidence\validate_final_zero_loss_patent_evidence_handoff_index_step172.py"
Write-Host "  05_training\patent_evidence\test_final_zero_loss_patent_evidence_handoff_index_step172.py"

if (-not (Test-Path $Generator)) {
    throw "[STOP] Missing generator: $Generator"
}
if (-not (Test-Path $Validator)) {
    throw "[STOP] Missing validator: $Validator"
}
if (-not (Test-Path $Test)) {
    throw "[STOP] Missing test: $Test"
}

& $Py $Generator --project-root $ProjectRoot --output-root $OutRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 172 handoff index generation failed"
}

& $Py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 172 handoff index validation failed"
}

& $Py $Test
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 172 handoff index unittest failed"
}

Write-Host "[OK] Step 172 Final Zero-Loss Patent Evidence Handoff Index self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
Write-Host "[DONE] Step 172 complete. Handoff index is ready; non-claim guards remain locked."
