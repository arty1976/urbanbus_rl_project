$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$PatentEvidenceDir = Join-Path $ProjectRoot "05_training\patent_evidence"
$Script = Join-Path $PatentEvidenceDir "patent_packet_export_checklist_step175.py"
$Validator = Join-Path $PatentEvidenceDir "validate_patent_packet_export_checklist_step175.py"
$Test = Join-Path $PatentEvidenceDir "test_patent_packet_export_checklist_step175.py"
$OutRoot = "artifacts\patent_evidence\patent_packet_export_checklist_step175_selftest"
$Manifest = Join-Path $OutRoot "patent_packet_export_manifest_step175.json"

Write-Host "[INFO] Step 175 Patent Packet Export Checklist should be copied into:"
Write-Host "       $PatentEvidenceDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\patent_packet_export_checklist_step175.md"
Write-Host "  05_training\patent_evidence\patent_packet_export_checklist_step175.py"
Write-Host "  05_training\patent_evidence\validate_patent_packet_export_checklist_step175.py"
Write-Host "  05_training\patent_evidence\test_patent_packet_export_checklist_step175.py"
Write-Host ""

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

if (-not (Test-Path $Script)) { throw "[FAIL] missing script: $Script" }
if (-not (Test-Path $Validator)) { throw "[FAIL] missing validator: $Validator" }
if (-not (Test-Path $Test)) { throw "[FAIL] missing test: $Test" }

& $py $Script --mode sample --output-root $OutRoot
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 175 checklist generation failed" }

& $py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 175 checklist validation failed" }

Push-Location $PatentEvidenceDir
try {
    & $py (Split-Path $Test -Leaf)
    if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 175 unittest failed" }
} finally {
    Pop-Location
}

Write-Host "[OK] Step 175 Patent Packet Export Checklist self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
Write-Host "[DONE] Step 175 complete. Export checklist generated; no export ZIP created; legal guards remain locked."
