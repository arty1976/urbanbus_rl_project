$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$Generator = Join-Path $ProjectRoot "05_training\observability\h200_observability_handoff_packet_index_step157.py"
$Validator = Join-Path $ProjectRoot "05_training\observability\validate_h200_observability_handoff_packet_index_step157.py"
$SelfTest = Join-Path $ProjectRoot "05_training\observability\test_h200_observability_handoff_packet_index_step157.py"
$OutputRoot = Join-Path $ProjectRoot "artifacts\observability\h200_observability_handoff_packet_index_step157"
$Manifest = Join-Path $OutputRoot "h200_observability_handoff_packet_index_step157_manifest.json"

if (-not (Test-Path $Generator)) { throw "[STOP] generator not found: $Generator" }
if (-not (Test-Path $Validator)) { throw "[STOP] validator not found: $Validator" }
if (-not (Test-Path $SelfTest)) { throw "[STOP] self-test not found: $SelfTest" }

Write-Host "[RUN] Step 157 observability handoff packet index generator"
python $Generator --project-root $ProjectRoot --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 157 generator failed" }

Write-Host "[RUN] Step 157 validator"
python $Validator --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 157 validator failed" }

Write-Host "[RUN] Step 157 self-test"
python $SelfTest
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 157 self-test failed" }

Write-Host "[OK] Step 157 H200 observability handoff packet index completed"
Write-Host "[OK] manifest: .\artifacts\observability\h200_observability_handoff_packet_index_step157\h200_observability_handoff_packet_index_step157_manifest.json"
Write-Host "[OK] status  : H200_OBSERVABILITY_HANDOFF_PACKET_INDEX_READY_MONITORING_ONLY_STILL_LOCKED"
Write-Host "[OK] decision: OBSERVABILITY_PACKET_ONLY_NOT_RELEASED"
