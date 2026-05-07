$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$PatentDir = Join-Path $ProjectRoot "05_training\patent_evidence"
Write-Host "[INFO] Step 176 Attorney Packet Dry-Run Exporter should be copied into:"
Write-Host "       $PatentDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\attorney_packet_dry_run_exporter_step176.md"
Write-Host "  05_training\patent_evidence\attorney_packet_dry_run_exporter_step176.py"
Write-Host "  05_training\patent_evidence\validate_attorney_packet_dry_run_exporter_step176.py"
Write-Host "  05_training\patent_evidence\test_attorney_packet_dry_run_exporter_step176.py"
Write-Host ""

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$Exporter = ".\05_training\patent_evidence\attorney_packet_dry_run_exporter_step176.py"
$Validator = ".\05_training\patent_evidence\validate_attorney_packet_dry_run_exporter_step176.py"
$TestFile = ".\05_training\patent_evidence\test_attorney_packet_dry_run_exporter_step176.py"
$OutRoot = ".\artifacts\patent_evidence\attorney_packet_dry_run_exporter_step176_selftest"

& $py $Exporter --project-root . --output-root $OutRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 176 dry-run exporter failed"
}

$Manifest = Join-Path $OutRoot "attorney_packet_dry_run_manifest_step176.json"
& $py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 176 validation failed"
}

& $py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 176 unittest failed"
}

Write-Host "[OK] Step 176 Attorney Packet Dry-Run Exporter self-test PASS"
Write-Host "[OK] artifact: $OutRoot"
Write-Host "[DONE] Step 176 complete. Dry-run manifest created; no ZIP was created."
