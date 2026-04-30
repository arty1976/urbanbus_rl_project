$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$RewardDir = ".\05_training\rewards"
$OutputRoot = ".\artifacts\rewards\h200_receive_side_preflight_gate_step148"

& $py "$RewardDir\test_h200_receive_side_preflight_gate_step148.py"
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 148 self-test failed"
}

& $py "$RewardDir\h200_receive_side_preflight_gate_step148.py" `
  --project-root "." `
  --output-root $OutputRoot `
  --h200-project-root "/workspace/urbanbus_rl_project"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 148 preflight gate failed"
}

$Manifest = Join-Path $OutputRoot "h200_receive_side_preflight_gate_step148_manifest.json"

& $py "$RewardDir\validate_h200_receive_side_preflight_gate_step148.py" `
  --manifest $Manifest

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 148 manifest validation failed"
}

Write-Host "[DONE] Step 148 H200 receive-side preflight gate complete."
