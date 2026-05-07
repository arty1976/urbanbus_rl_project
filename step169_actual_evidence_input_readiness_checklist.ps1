$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$Py = "python"
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
}

$BaseDir = "05_training\patent_evidence"
$Runner = Join-Path $BaseDir "actual_evidence_input_readiness_checklist_step169.py"
$Validator = Join-Path $BaseDir "validate_actual_evidence_input_readiness_checklist_step169.py"
$TestFile = Join-Path $BaseDir "test_actual_evidence_input_readiness_checklist_step169.py"
$OutRoot = "artifacts\patent_evidence\actual_evidence_input_readiness_checklist_step169_selftest"

Write-Host "[INFO] Step 169 Actual Evidence Input Readiness Checklist should be copied into:"
Write-Host "       $ProjectRoot\05_training\patent_evidence"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\actual_evidence_input_readiness_checklist_step169.md"
Write-Host "  05_training\patent_evidence\actual_evidence_input_readiness_checklist_step169.py"
Write-Host "  05_training\patent_evidence\validate_actual_evidence_input_readiness_checklist_step169.py"
Write-Host "  05_training\patent_evidence\test_actual_evidence_input_readiness_checklist_step169.py"
Write-Host ""

foreach ($Required in @($Runner, $Validator, $TestFile)) {
    if (-not (Test-Path $Required)) {
        throw "[STOP] Required Step 169 file not found: $Required"
    }
}

if (Test-Path $OutRoot) {
    Remove-Item -Recurse -Force $OutRoot
}

& $Py $Runner `
  --project-root $ProjectRoot `
  --mode sample `
  --output-root $OutRoot `
  --require-pipeline-scripts

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 169 readiness checklist generation failed"
}

$Manifest = Join-Path $OutRoot "actual_evidence_input_readiness_checklist_manifest.json"
& $Py $Validator --manifest $Manifest --require-ready
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 169 readiness checklist validation failed"
}

& $Py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 169 unittest failed"
}

Write-Host "[OK] Step 169 Actual Evidence Input Readiness Checklist self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
Write-Host "[DONE] Step 169 complete. Input readiness can be checked; claim guards remain locked."
