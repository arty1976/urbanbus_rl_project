$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$EvidenceDir = Join-Path $ProjectRoot "05_training\patent_evidence"
Write-Host "[INFO] Step 162 Route-Aware Rollout Adapter should be copied into:"
Write-Host "       $EvidenceDir"
Write-Host ""
Write-Host "[INFO] Expected files:"
Write-Host "  05_training\patent_evidence\route_aware_rollout_adapter_schema_step162.md"
Write-Host "  05_training\patent_evidence\route_aware_rollout_adapter_step162.py"
Write-Host "  05_training\patent_evidence\validate_route_aware_rollout_adapter_step162.py"
Write-Host "  05_training\patent_evidence\test_route_aware_rollout_adapter_step162.py"
Write-Host ""

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$Adapter = ".\05_training\patent_evidence\route_aware_rollout_adapter_step162.py"
$Validator = ".\05_training\patent_evidence\validate_route_aware_rollout_adapter_step162.py"
$TestFile = ".\05_training\patent_evidence\test_route_aware_rollout_adapter_step162.py"

foreach ($p in @($Adapter, $Validator, $TestFile)) {
    if (-not (Test-Path $p)) {
        throw "[FAIL] Missing expected Step 162 file: $p"
    }
}

$OutRoot = ".\artifacts\patent_evidence\route_aware_rollout_adapter_step162_selftest"
$Step161OutRoot = ".\artifacts\patent_evidence\route_aware_pickup_attempt_events_step162_integration_selftest"
$Step160OutRoot = ".\artifacts\patent_evidence\zero_loss_pickup_evidence_step162_integration_selftest"

if (Test-Path $OutRoot) { Remove-Item -Recurse -Force $OutRoot }
if (Test-Path $Step161OutRoot) { Remove-Item -Recurse -Force $Step161OutRoot }
if (Test-Path $Step160OutRoot) { Remove-Item -Recurse -Force $Step160OutRoot }

& $py $Adapter `
  --mode sample `
  --output-root $OutRoot `
  --condition-id A `
  --seed 1 `
  --file-format csv `
  --max-events 20
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 162 adapter sample run failed"
}

$Step162Manifest = Join-Path $OutRoot "route_aware_rollout_adapter_manifest.json"
& $py $Validator --manifest $Step162Manifest
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 162 adapter validation failed"
}

$Step161Writer = ".\05_training\patent_evidence\route_aware_pickup_attempt_event_writer_step161.py"
$Step161Validator = ".\05_training\patent_evidence\validate_route_aware_pickup_attempt_events_step161.py"
$Step160Reporter = ".\05_training\patent_evidence\zero_loss_pickup_evidence_reporter_step160.py"
$Step160Validator = ".\05_training\patent_evidence\validate_zero_loss_pickup_evidence_bundle_step160.py"

if ((Test-Path $Step161Writer) -and (Test-Path $Step161Validator)) {
    Write-Host "[INFO] Step 161 files detected. Running Step 162 -> Step 161 integration smoke."
    $Step162Obj = Get-Content $Step162Manifest -Raw | ConvertFrom-Json
    $NormalizedEvents = $Step162Obj.output_files.normalized_route_aware_rollout_events

    & $py $Step161Writer `
      --mode from-rollout `
      --rollout-events $NormalizedEvents `
      --source-manifest $Step162Manifest `
      --output-root $Step161OutRoot `
      --file-format csv `
      --zero-loss-epsilon-sec 0 `
      --max-attempts 100
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 162 -> Step 161 integration writer failed"
    }

    $Step161Manifest = Join-Path $Step161OutRoot "route_aware_pickup_attempt_event_writer_manifest.json"
    & $py $Step161Validator --manifest $Step161Manifest --require-step160-ready
    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 162 -> Step 161 integration validation failed"
    }

    if ((Test-Path $Step160Reporter) -and (Test-Path $Step160Validator)) {
        Write-Host "[INFO] Step 160 files detected. Running Step 162 -> Step 161 -> Step 160 integration smoke."
        $Step161Obj = Get-Content $Step161Manifest -Raw | ConvertFrom-Json
        $PickupEvents = $Step161Obj.output_files.pickup_attempt_events
        $EtaCounterfactual = $Step161Obj.output_files.eta_counterfactual
        $AttentionWeights = $Step161Obj.output_files.gatv2_attention
        $RunManifest = $Step161Obj.output_files.run_manifest

        & $py $Step160Reporter `
          --pickup-attempt-events $PickupEvents `
          --eta-counterfactual $EtaCounterfactual `
          --attention-weights $AttentionWeights `
          --run-manifest $RunManifest `
          --output-root $Step160OutRoot `
          --zero-loss-epsilon-sec 0 `
          --top-k-attention 10
        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 162 -> Step 161 -> Step 160 integration reporter failed"
        }

        $Step160Manifest = Join-Path $Step160OutRoot "zero_loss_evidence_manifest.json"
        & $py $Step160Validator --manifest $Step160Manifest
        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 162 -> Step 161 -> Step 160 integration validation failed"
        }
    } else {
        Write-Host "[WARN] Step 160 files not detected. Skipping Step 160 integration smoke."
    }
} else {
    Write-Host "[WARN] Step 161 files not detected. Skipping Step 161/160 integration smoke."
}

& $py $TestFile
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 162 unittest failed"
}

Write-Host "[OK] Step 162 Route-Aware Rollout Adapter self-test PASS"
Write-Host "[OK] artifact: $OutRoot"
if (Test-Path $Step161OutRoot) {
    Write-Host "[OK] step161_integration_artifact: $Step161OutRoot"
}
if (Test-Path $Step160OutRoot) {
    Write-Host "[OK] step160_integration_artifact: $Step160OutRoot"
}
Write-Host "[DONE] Step 162 complete. Non-claim guards remain locked."
