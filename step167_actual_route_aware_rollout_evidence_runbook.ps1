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

$Root = ".\05_training\patent_evidence"
$Runbook = Join-Path $Root "actual_route_aware_rollout_evidence_runbook_step167.py"
$Validator = Join-Path $Root "validate_actual_route_aware_rollout_evidence_runbook_step167.py"
$Test = Join-Path $Root "test_actual_route_aware_rollout_evidence_runbook_step167.py"
$OutRoot = ".\artifacts\patent_evidence\actual_route_aware_rollout_evidence_runbook_step167_selftest"
$Manifest = Join-Path $OutRoot "actual_route_aware_rollout_evidence_runbook_manifest_step167.json"

Write-Host "[INFO] Step 167 Actual Route-Aware Rollout Evidence Runbook should be copied into:"
Write-Host "       $ProjectRoot\05_training\patent_evidence"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\actual_route_aware_rollout_evidence_runbook_step167.md"
Write-Host "  05_training\patent_evidence\actual_route_aware_rollout_evidence_runbook_step167.py"
Write-Host "  05_training\patent_evidence\validate_actual_route_aware_rollout_evidence_runbook_step167.py"
Write-Host "  05_training\patent_evidence\test_actual_route_aware_rollout_evidence_runbook_step167.py"
Write-Host ""

foreach ($Required in @($Runbook, $Validator, $Test)) {
    if (-not (Test-Path $Required)) {
        throw "[STOP] Required file not found: $Required"
    }
}

& $Py $Runbook --project-root . --output-root $OutRoot
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 167 runbook generation failed"
}

& $Py $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 167 runbook validation failed"
}

& $Py $Test
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 167 unittest failed"
}

Write-Host "[OK] Step 167 Actual Route-Aware Rollout Evidence Runbook self-test PASS"
Write-Host "[OK] artifact: $OutRoot"
Write-Host "[DONE] Step 167 complete. Actual evidence procedure is fixed; non-claim guards remain locked."
