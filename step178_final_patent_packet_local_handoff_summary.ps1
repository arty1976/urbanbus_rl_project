$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $ProjectRoot "05_training"))) {
    $ProjectRoot = Get-Location
}
Set-Location $ProjectRoot

$PatentDir = Join-Path $ProjectRoot "05_training\patent_evidence"
$Script = Join-Path $PatentDir "final_patent_packet_local_handoff_summary_step178.py"
$Validator = Join-Path $PatentDir "validate_final_patent_packet_local_handoff_summary_step178.py"
$Test = Join-Path $PatentDir "test_final_patent_packet_local_handoff_summary_step178.py"

Write-Host "[INFO] Step 178 Final Patent Packet Local Handoff Summary should be copied into:"
Write-Host "       $PatentDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\final_patent_packet_local_handoff_summary_step178.md"
Write-Host "  05_training\patent_evidence\final_patent_packet_local_handoff_summary_step178.py"
Write-Host "  05_training\patent_evidence\validate_final_patent_packet_local_handoff_summary_step178.py"
Write-Host "  05_training\patent_evidence\test_final_patent_packet_local_handoff_summary_step178.py"
Write-Host ""

if (-not (Test-Path $Script)) { throw "[FAIL] missing script: $Script" }
if (-not (Test-Path $Validator)) { throw "[FAIL] missing validator: $Validator" }
if (-not (Test-Path $Test)) { throw "[FAIL] missing test: $Test" }

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $Py = "python"
}

$OutRoot = "artifacts\patent_evidence\final_patent_packet_local_handoff_summary_step178_selftest"
$Manifest = Join-Path $OutRoot "final_patent_packet_local_handoff_manifest_step178.json"

& $Py $Script --project-root $ProjectRoot --output-root $OutRoot
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 178 summary generation failed" }

& $Py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 178 summary validation failed" }

& $Py $Test
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 178 unittest failed" }

Write-Host "[OK] Step 178 Final Patent Packet Local Handoff Summary self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
Write-Host "[DONE] Step 178 complete. Local patent packet preparation is summarized; ZIP/legal/claim guards remain locked."
