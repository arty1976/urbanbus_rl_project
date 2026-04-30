param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [int]$NumAgents = 8,
    [int]$EpisodeSteps = 4,
    [int]$Seed = 101,
    [string]$OutputJson = ".\artifacts\causal_simulator_v2_adapter_smoke\status.json"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py ".\05_training\run_causal_simulator_v2_adapter_smoke.py" `
  --project-root "." `
  --num-agents $NumAgents `
  --episode-steps $EpisodeSteps `
  --seed $Seed `
  --output-json $OutputJson

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 101 actual adapter smoke failed"
}

Write-Host "[OK] Step 101 actual adapter smoke complete."
Write-Host "[OK] output_json = $OutputJson"
