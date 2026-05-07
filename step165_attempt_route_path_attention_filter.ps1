$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$TargetDir = Join-Path $ProjectRoot "05_training\patent_evidence"

Write-Host "[INFO] Step 165 Attempt-Specific Route/Path Attention Filter should be copied into:"
Write-Host "       $TargetDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\attempt_route_path_attention_filter_schema_step165.md"
Write-Host "  05_training\patent_evidence\attempt_route_path_attention_filter_step165.py"
Write-Host "  05_training\patent_evidence\validate_attempt_route_path_attention_filter_step165.py"
Write-Host "  05_training\patent_evidence\test_attempt_route_path_attention_filter_step165.py"
Write-Host ""

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$Filter = Join-Path $TargetDir "attempt_route_path_attention_filter_step165.py"
$Validator = Join-Path $TargetDir "validate_attempt_route_path_attention_filter_step165.py"
$Test = Join-Path $TargetDir "test_attempt_route_path_attention_filter_step165.py"

foreach ($p in @($Filter, $Validator, $Test)) {
    if (-not (Test-Path $p)) {
        throw "[STOP] Expected Step 165 file missing: $p"
    }
}

$OutRoot = "artifacts\patent_evidence\attempt_route_path_attention_filter_step165_selftest"
& $py $Filter `
  --mode sample `
  --output-root $OutRoot `
  --top-k-per-attempt 6
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 165 attempt-route-path attention filter sample run failed"
}

$Manifest = Join-Path $OutRoot "attempt_route_path_attention_filter_manifest.json"
& $py $Validator `
  --manifest $Manifest `
  --require-real-attention `
  --require-no-fallback
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 165 validator failed"
}

$Step160Reporter = Join-Path $TargetDir "zero_loss_pickup_evidence_reporter_step160.py"
$Step160Validator = Join-Path $TargetDir "validate_zero_loss_pickup_evidence_bundle_step160.py"
if ((Test-Path $Step160Reporter) -and (Test-Path $Step160Validator)) {
    Write-Host "[INFO] Step 160 files detected. Running Step 165 -> Step 160 evidence report integration smoke."
    $Step160Out = "artifacts\patent_evidence\zero_loss_pickup_evidence_step165_integration_selftest"
    & $py $Step160Reporter `
      --pickup-attempt-events (Join-Path $OutRoot "pickup_attempt_events.csv") `
      --eta-counterfactual (Join-Path $OutRoot "eta_counterfactual.csv") `
      --attention-weights (Join-Path $OutRoot "gatv2_attention.csv") `
      --run-manifest (Join-Path $OutRoot "run_manifest.json") `
      --output-root $Step160Out
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 165 -> Step 160 reporter integration failed"
    }

    & $py $Step160Validator `
      --manifest (Join-Path $Step160Out "zero_loss_evidence_manifest.json")
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 165 -> Step 160 validator integration failed"
    }
}

Push-Location $TargetDir
try {
    & $py $Test
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 165 unittest failed"
    }
} finally {
    Pop-Location
}

Write-Host "[OK] Step 165 Attempt-Specific Route/Path Attention Filter self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
if (Test-Path "artifacts\patent_evidence\zero_loss_pickup_evidence_step165_integration_selftest") {
    Write-Host "[OK] step160_integration_artifact: .\artifacts\patent_evidence\zero_loss_pickup_evidence_step165_integration_selftest"
}
Write-Host "[DONE] Step 165 complete. Attempt-specific real attention filtering is connected; non-claim guards remain locked."
