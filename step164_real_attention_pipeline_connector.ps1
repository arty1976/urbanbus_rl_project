$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $ProjectRoot "05_training"))) {
    $ProjectRoot = Get-Location
}
Set-Location $ProjectRoot

$PatentDir = Join-Path $ProjectRoot "05_training\patent_evidence"
$Connector = Join-Path $PatentDir "real_attention_pipeline_connector_step164.py"
$Validator = Join-Path $PatentDir "validate_real_attention_pipeline_connector_step164.py"
$TestFile = Join-Path $PatentDir "test_real_attention_pipeline_connector_step164.py"
$Step161Writer = Join-Path $PatentDir "route_aware_pickup_attempt_event_writer_step161.py"
$Step163Extractor = Join-Path $PatentDir "gatv2_real_attention_extractor_step163.py"
$Step160Reporter = Join-Path $PatentDir "zero_loss_pickup_evidence_reporter_step160.py"
$Step160Validator = Join-Path $PatentDir "validate_zero_loss_pickup_evidence_bundle_step160.py"

Write-Host "[INFO] Step 164 Real Attention Pipeline Connector should be copied into:"
Write-Host "       $PatentDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\real_attention_pipeline_connector_schema_step164.md"
Write-Host "  05_training\patent_evidence\real_attention_pipeline_connector_step164.py"
Write-Host "  05_training\patent_evidence\validate_real_attention_pipeline_connector_step164.py"
Write-Host "  05_training\patent_evidence\test_real_attention_pipeline_connector_step164.py"
Write-Host ""

$Required = @($Connector, $Validator, $TestFile)
foreach ($p in $Required) {
    if (-not (Test-Path $p)) {
        throw "[FAIL] Required Step 164 file missing: $p"
    }
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} elseif (Test-Path ".\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$OutRoot = "artifacts\patent_evidence\real_attention_pipeline_connector_step164_selftest"
$Step161Out = "artifacts\patent_evidence\route_aware_pickup_attempt_events_step164_input"
$Step163Out = "artifacts\patent_evidence\gatv2_real_attention_extractor_step164_input"
$Step160Out = "artifacts\patent_evidence\zero_loss_pickup_evidence_step164_integration_selftest"

if ((Test-Path $Step161Writer) -and (Test-Path $Step163Extractor)) {
    Write-Host "[INFO] Step 161 and Step 163 files detected. Running real-attention connector integration path."

    if (Test-Path $Step161Out) { Remove-Item -Recurse -Force $Step161Out }
    if (Test-Path $Step163Out) { Remove-Item -Recurse -Force $Step163Out }
    if (Test-Path $OutRoot) { Remove-Item -Recurse -Force $OutRoot }

    & $py $Step161Writer `
      --mode sample `
      --output-root $Step161Out `
      --condition-id A `
      --seed 1 `
      --zero-loss-epsilon-sec 0 `
      --file-format csv `
      --max-attempts 5
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 161 input generation for Step 164 failed"
    }

    & $py $Step163Extractor `
      --mode sample `
      --output-root $Step163Out `
      --condition-id A `
      --seed 1 `
      --file-format csv `
      --top-k-per-attempt 8
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 163 real attention extraction for Step 164 failed"
    }

    & $py $Connector `
      --pickup-attempt-events (Join-Path $Step161Out "pickup_attempt_events.csv") `
      --eta-counterfactual (Join-Path $Step161Out "eta_counterfactual.csv") `
      --step161-run-manifest (Join-Path $Step161Out "run_manifest.json") `
      --real-snapshot-attention (Join-Path $Step163Out "gatv2_real_attention_edges.csv") `
      --real-attention-manifest (Join-Path $Step163Out "gatv2_real_attention_extractor_manifest.json") `
      --output-root $OutRoot `
      --file-format csv `
      --top-k-per-attempt 12 `
      --require-real-attention
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 164 connector integration path failed"
    }
} else {
    Write-Host "[WARN] Step 161 or Step 163 files are missing. Running Step 164 standalone sample mode."
    if (Test-Path $OutRoot) { Remove-Item -Recurse -Force $OutRoot }
    & $py $Connector `
      --make-sample-inputs `
      --output-root $OutRoot `
      --file-format csv `
      --top-k-per-attempt 12 `
      --require-real-attention
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 164 connector standalone sample path failed"
    }
}

$Step164Manifest = Join-Path $OutRoot "real_attention_pipeline_connector_manifest.json"
& $py $Validator --manifest $Step164Manifest --require-real-attention
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 164 connector validator failed"
}

if ((Test-Path $Step160Reporter) -and (Test-Path $Step160Validator)) {
    Write-Host "[INFO] Step 160 files detected. Running Step 164 -> Step 160 evidence report integration smoke."
    if (Test-Path $Step160Out) { Remove-Item -Recurse -Force $Step160Out }
    & $py $Step160Reporter `
      --pickup-attempt-events (Join-Path $OutRoot "pickup_attempt_events.csv") `
      --eta-counterfactual (Join-Path $OutRoot "eta_counterfactual.csv") `
      --attention-weights (Join-Path $OutRoot "gatv2_attention.csv") `
      --run-manifest (Join-Path $OutRoot "run_manifest.json") `
      --output-root $Step160Out `
      --zero-loss-epsilon-sec 0 `
      --top-k-attention 12
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 164 -> Step 160 reporter integration failed"
    }
    & $py $Step160Validator --manifest (Join-Path $Step160Out "zero_loss_evidence_manifest.json")
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 164 -> Step 160 evidence bundle validation failed"
    }
} else {
    Write-Host "[WARN] Step 160 reporter/validator missing. Skipping Step 160 integration smoke."
}

& $py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 164 unittest failed"
}

Write-Host "[OK] Step 164 Real Attention Pipeline Connector self-test PASS"
Write-Host "[OK] artifact: .\$OutRoot"
if (Test-Path $Step160Out) {
    Write-Host "[OK] step160_integration_artifact: .\$Step160Out"
}
Write-Host "[DONE] Step 164 complete. Real attention is connected; non-claim guards remain locked."
