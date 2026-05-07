$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$ScriptDir = Join-Path $ProjectRoot "05_training\patent_evidence"
$Generator = Join-Path $ScriptDir "zero_loss_patent_claim_drafting_packet_step173.py"
$Validator = Join-Path $ScriptDir "validate_zero_loss_patent_claim_drafting_packet_step173.py"
$TestScript = Join-Path $ScriptDir "test_zero_loss_patent_claim_drafting_packet_step173.py"
$OutRoot = "artifacts\patent_evidence\zero_loss_patent_claim_drafting_packet_step173_selftest"
$Manifest = Join-Path $OutRoot "zero_loss_claim_drafting_manifest_step173.json"

Write-Host "[INFO] Step 173 Zero-Loss Patent Claim Drafting Packet should be copied into:"
Write-Host "       $ScriptDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\zero_loss_patent_claim_drafting_packet_step173.md"
Write-Host "  05_training\patent_evidence\zero_loss_patent_claim_drafting_packet_step173.py"
Write-Host "  05_training\patent_evidence\validate_zero_loss_patent_claim_drafting_packet_step173.py"
Write-Host "  05_training\patent_evidence\test_zero_loss_patent_claim_drafting_packet_step173.py"
Write-Host ""

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

if (-not (Test-Path $Generator)) { throw "[FAIL] missing generator: $Generator" }
if (-not (Test-Path $Validator)) { throw "[FAIL] missing validator: $Validator" }
if (-not (Test-Path $TestScript)) { throw "[FAIL] missing test script: $TestScript" }

& $py $Generator --output-root $OutRoot
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 173 packet generation failed" }

& $py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 173 packet validation failed" }

& $py $TestScript
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 173 unittest failed" }

Write-Host "[OK] Step 173 Zero-Loss Patent Claim Drafting Packet self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
Write-Host "[DONE] Step 173 complete. Technical drafting packet generated; legal/claim guards remain locked."
