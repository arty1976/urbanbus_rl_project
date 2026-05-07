$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$EvidenceDir = Join-Path $ProjectRoot "05_training\patent_evidence"
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null

Write-Host "[INFO] Step 160 Zero-Loss Pickup Evidence Reporter scaffold should be copied into:"
Write-Host "       $EvidenceDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\zero_loss_pickup_evidence_schema_step160.md"
Write-Host "  05_training\patent_evidence\zero_loss_pickup_evidence_reporter_step160.py"
Write-Host "  05_training\patent_evidence\validate_zero_loss_pickup_evidence_bundle_step160.py"
Write-Host "  05_training\patent_evidence\test_zero_loss_pickup_evidence_reporter_step160.py"
Write-Host ""

$Py = "python"
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
}

$OutputRoot = ".\artifacts\patent_evidence\zero_loss_pickup_evidence_step160_selftest"
& $Py ".\05_training\patent_evidence\zero_loss_pickup_evidence_reporter_step160.py" `
    --output-root $OutputRoot `
    --make-sample-inputs
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 160 reporter sample run failed"
}

$Manifest = Join-Path $OutputRoot "zero_loss_evidence_manifest.json"
& $Py ".\05_training\patent_evidence\validate_zero_loss_pickup_evidence_bundle_step160.py" `
    --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 160 validator failed"
}

& $Py ".\05_training\patent_evidence\test_zero_loss_pickup_evidence_reporter_step160.py"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 160 unittest failed"
}

Write-Host "[OK] Step 160 Zero-Loss Pickup Evidence Reporter scaffold self-test PASS"
Write-Host "[OK] artifact: $OutputRoot"
Write-Host "[DONE] Step 160 complete. Non-claim guards remain locked."
