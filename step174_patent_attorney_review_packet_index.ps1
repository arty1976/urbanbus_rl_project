$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Py = ".\05_training\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

$Dir = Join-Path $ProjectRoot "05_training\patent_evidence"
Write-Host "[INFO] Step 174 Patent Attorney Review Packet Index should be copied into:"
Write-Host "       $Dir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\patent_attorney_review_packet_index_step174.md"
Write-Host "  05_training\patent_evidence\patent_attorney_review_packet_index_step174.py"
Write-Host "  05_training\patent_evidence\validate_patent_attorney_review_packet_index_step174.py"
Write-Host "  05_training\patent_evidence\test_patent_attorney_review_packet_index_step174.py"
Write-Host ""

$Script = ".\05_training\patent_evidence\patent_attorney_review_packet_index_step174.py"
$Validator = ".\05_training\patent_evidence\validate_patent_attorney_review_packet_index_step174.py"
$Test = ".\05_training\patent_evidence\test_patent_attorney_review_packet_index_step174.py"
$OutRoot = ".\artifacts\patent_evidence\patent_attorney_review_packet_index_step174_selftest"

& $Py $Script --project-root . --output-root $OutRoot --mode selftest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 174 packet generation failed" }

$Manifest = Join-Path $OutRoot "patent_attorney_review_manifest_step174.json"
& $Py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 174 validation failed" }

& $Py $Test
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 174 unittest failed" }

Write-Host "[OK] Step 174 Patent Attorney Review Packet Index self-test PASS"
Write-Host "[OK] artifact: $OutRoot"
Write-Host "[DONE] Step 174 complete. Attorney review packet index generated; legal and claim guards remain locked."
