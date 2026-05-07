$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$TargetDir = Join-Path $ProjectRoot "05_training\patent_evidence"
Write-Host "[INFO] Step 161 Route-Aware Pickup Attempt Event Writer should be copied into:"
Write-Host "       $TargetDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\route_aware_pickup_attempt_event_schema_step161.md"
Write-Host "  05_training\patent_evidence\route_aware_pickup_attempt_event_writer_step161.py"
Write-Host "  05_training\patent_evidence\validate_route_aware_pickup_attempt_events_step161.py"
Write-Host "  05_training\patent_evidence\test_route_aware_pickup_attempt_event_writer_step161.py"
Write-Host ""

$Py = "python"
$VenvPy = Join-Path $ProjectRoot "05_training\.venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    $Py = $VenvPy
}

$Writer = Join-Path $TargetDir "route_aware_pickup_attempt_event_writer_step161.py"
$Validator = Join-Path $TargetDir "validate_route_aware_pickup_attempt_events_step161.py"
$TestFile = Join-Path $TargetDir "test_route_aware_pickup_attempt_event_writer_step161.py"
$Step160Reporter = Join-Path $TargetDir "zero_loss_pickup_evidence_reporter_step160.py"
$Step160Validator = Join-Path $TargetDir "validate_zero_loss_pickup_evidence_bundle_step160.py"

foreach ($RequiredFile in @($Writer, $Validator, $TestFile)) {
    if (-not (Test-Path $RequiredFile)) {
        throw "[FAIL] Required Step 161 file missing: $RequiredFile"
    }
}

$OutRoot = "artifacts\patent_evidence\route_aware_pickup_attempt_events_step161_selftest"
$Step160OutRoot = "artifacts\patent_evidence\zero_loss_pickup_evidence_step161_integration_selftest"

& $Py $Writer `
  --mode sample `
  --output-root $OutRoot `
  --condition-id A `
  --seed 1 `
  --file-format csv `
  --zero-loss-epsilon-sec 0
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 161 writer sample run failed"
}

$WriterManifest = Join-Path $OutRoot "route_aware_pickup_attempt_event_writer_manifest.json"
& $Py $Validator --manifest $WriterManifest --require-step160-ready
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 161 validator failed"
}

if ((Test-Path $Step160Reporter) -and (Test-Path $Step160Validator)) {
    Write-Host "[INFO] Step 160 files detected. Running Step 161 -> Step 160 integration smoke."
    & $Py $Step160Reporter `
      --pickup-attempt-events (Join-Path $OutRoot "pickup_attempt_events.csv") `
      --eta-counterfactual (Join-Path $OutRoot "eta_counterfactual.csv") `
      --attention-weights (Join-Path $OutRoot "gatv2_attention.csv") `
      --run-manifest (Join-Path $OutRoot "run_manifest.json") `
      --output-root $Step160OutRoot `
      --zero-loss-epsilon-sec 0
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 161 -> Step 160 reporter integration failed"
    }

    & $Py $Step160Validator --manifest (Join-Path $Step160OutRoot "zero_loss_evidence_manifest.json")
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 161 -> Step 160 bundle validation failed"
    }
} else {
    Write-Host "[WARN] Step 160 reporter/validator not found. Skipping integration smoke."
}

& $Py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 161 unittest failed"
}

Write-Host "[OK] Step 161 Route-Aware Pickup Attempt Event Writer self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
Write-Host "[OK] integration_artifact: .\$Step160OutRoot"
Write-Host "[DONE] Step 161 complete. Non-claim guards remain locked."
