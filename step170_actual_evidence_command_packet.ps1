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
$Runner = Join-Path $BaseDir "actual_evidence_command_packet_step170.py"
$Validator = Join-Path $BaseDir "validate_actual_evidence_command_packet_step170.py"
$TestFile = Join-Path $BaseDir "test_actual_evidence_command_packet_step170.py"
$Step169Runner = Join-Path $BaseDir "actual_evidence_input_readiness_checklist_step169.py"
$Step169Validator = Join-Path $BaseDir "validate_actual_evidence_input_readiness_checklist_step169.py"

$OutRoot169 = "artifacts\patent_evidence\actual_evidence_input_readiness_checklist_step170_input_selftest"
$OutRoot170 = "artifacts\patent_evidence\actual_evidence_command_packet_step170_selftest"

Write-Host "[INFO] Step 170 Actual Evidence Command Packet should be copied into:"
Write-Host "       $ProjectRoot\05_training\patent_evidence"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\actual_evidence_command_packet_step170.md"
Write-Host "  05_training\patent_evidence\actual_evidence_command_packet_step170.py"
Write-Host "  05_training\patent_evidence\validate_actual_evidence_command_packet_step170.py"
Write-Host "  05_training\patent_evidence\test_actual_evidence_command_packet_step170.py"
Write-Host ""

foreach ($Required in @($Runner, $Validator, $TestFile)) {
    if (-not (Test-Path $Required)) {
        throw "[STOP] Required Step 170 file not found: $Required"
    }
}

if (Test-Path $OutRoot170) {
    Remove-Item -Recurse -Force $OutRoot170
}

if ((Test-Path $Step169Runner) -and (Test-Path $Step169Validator)) {
    Write-Host "[INFO] Step 169 files detected. Generating Step 169 sample readiness manifest for Step 170."
    if (Test-Path $OutRoot169) {
        Remove-Item -Recurse -Force $OutRoot169
    }
    & $Py $Step169Runner `
      --project-root $ProjectRoot `
      --mode sample `
      --output-root $OutRoot169 `
      --require-pipeline-scripts
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 169 sample readiness generation failed"
    }
    $ReadinessManifest = Join-Path $OutRoot169 "actual_evidence_input_readiness_checklist_manifest.json"
    & $Py $Step169Validator --manifest $ReadinessManifest --require-ready
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 169 readiness validation failed"
    }

    & $Py $Runner `
      --project-root $ProjectRoot `
      --mode from-readiness `
      --readiness-manifest $ReadinessManifest `
      --output-root $OutRoot170 `
      --run-id step170_selftest_from_step169 `
      --attention-mode sample_real_attention_smoke
} else {
    Write-Host "[WARN] Step 169 files not detected. Falling back to Step 170 internal sample mode."
    & $Py $Runner `
      --project-root $ProjectRoot `
      --mode sample `
      --output-root $OutRoot170 `
      --run-id step170_selftest_sample `
      --attention-mode sample_real_attention_smoke
}

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 170 command packet generation failed"
}

$Manifest = Join-Path $OutRoot170 "actual_evidence_command_packet_manifest_step170.json"
& $Py $Validator --manifest $Manifest --require-pass --require-dry-run
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 170 command packet validation failed"
}

& $Py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 170 unittest failed"
}

Write-Host "[OK] Step 170 Actual Evidence Command Packet self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot170"
Write-Host "[DONE] Step 170 complete. Command packet is dry-run only; non-claim guards remain locked."
