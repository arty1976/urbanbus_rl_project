$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$TargetDir = Join-Path $ProjectRoot "05_training\patent_evidence"

Write-Host "[INFO] Step 171 Actual Evidence Execution Release Checklist should be copied into:"
Write-Host "       $TargetDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\actual_evidence_execution_release_checklist_step171.md"
Write-Host "  05_training\patent_evidence\actual_evidence_execution_release_checklist_step171.py"
Write-Host "  05_training\patent_evidence\validate_actual_evidence_execution_release_checklist_step171.py"
Write-Host "  05_training\patent_evidence\test_actual_evidence_execution_release_checklist_step171.py"
Write-Host ""

$py = "python"
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
}

$Script = ".\05_training\patent_evidence\actual_evidence_execution_release_checklist_step171.py"
$Validator = ".\05_training\patent_evidence\validate_actual_evidence_execution_release_checklist_step171.py"
$Test = ".\05_training\patent_evidence\test_actual_evidence_execution_release_checklist_step171.py"

foreach ($f in @($Script, $Validator, $Test)) {
    if (-not (Test-Path $f)) {
        throw "[FAIL] Required file missing: $f"
    }
}

$OutRoot = ".\artifacts\patent_evidence\actual_evidence_execution_release_checklist_step171_selftest"

& $py $Script `
  --mode sample `
  --output-root $OutRoot `
  --operator-approval-granted false `
  --release-manifest-committed false

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 171 release checklist generation failed"
}

$Manifest = Join-Path $OutRoot "actual_evidence_execution_release_checklist_manifest.json"

& $py $Validator --manifest $Manifest --expect-locked
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 171 release checklist validation failed"
}

& $py $Test
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 171 unittest failed"
}

Write-Host "[OK] Step 171 Actual Evidence Execution Release Checklist self-test PASS"
Write-Host "[OK] artifact: $OutRoot"
Write-Host "[DONE] Step 171 complete. Actual evidence execution remains locked until explicit approval."
