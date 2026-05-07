$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$PatentDir = Join-Path $ProjectRoot "05_training\patent_evidence"
Write-Host "[INFO] Step 177 Attorney Packet ZIP Exporter Still Locked should be copied into:"
Write-Host "       $PatentDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\attorney_packet_zip_exporter_still_locked_step177.md"
Write-Host "  05_training\patent_evidence\attorney_packet_zip_exporter_still_locked_step177.py"
Write-Host "  05_training\patent_evidence\validate_attorney_packet_zip_exporter_still_locked_step177.py"
Write-Host "  05_training\patent_evidence\test_attorney_packet_zip_exporter_still_locked_step177.py"
Write-Host ""

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$Exporter = ".\05_training\patent_evidence\attorney_packet_zip_exporter_still_locked_step177.py"
$Validator = ".\05_training\patent_evidence\validate_attorney_packet_zip_exporter_still_locked_step177.py"
$TestFile = ".\05_training\patent_evidence\test_attorney_packet_zip_exporter_still_locked_step177.py"
$OutRoot = ".\artifacts\patent_evidence\attorney_packet_zip_exporter_still_locked_step177_selftest"
$Step176Script = ".\05_training\patent_evidence\attorney_packet_dry_run_exporter_step176.py"
$Step176OutRoot = ".\artifacts\patent_evidence\attorney_packet_dry_run_exporter_step177_input"

if (Test-Path $Step176Script) {
    Write-Host "[INFO] Step 176 dry-run exporter detected. Generating fresh Step 176 input manifest."
    & $py $Step176Script --project-root . --output-root $Step176OutRoot
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 176 input dry-run generation failed"
    }
    $DryRunManifest = Join-Path $Step176OutRoot "attorney_packet_dry_run_manifest_step176.json"
    & $py $Exporter --project-root . --dry-run-manifest $DryRunManifest --output-root $OutRoot
} else {
    Write-Host "[INFO] Step 176 dry-run exporter not detected. Running Step 177 sample mode."
    & $py $Exporter --mode sample --project-root . --output-root $OutRoot
}

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 177 locked ZIP exporter failed"
}

$Manifest = Join-Path $OutRoot "attorney_packet_zip_exporter_still_locked_manifest_step177.json"
& $py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 177 validation failed"
}

& $py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 177 unittest failed"
}

Write-Host "[OK] Step 177 Attorney Packet ZIP Exporter Still Locked self-test PASS"
Write-Host "[OK] artifact: $OutRoot"
Write-Host "[DONE] Step 177 complete. ZIP creation remains locked; no ZIP was created."
