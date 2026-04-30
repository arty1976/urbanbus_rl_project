param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [string]$Conditions = "C2_STATIC_SIGNAL",
    [string]$Seeds = "101",
    [int]$NumAgents = 8,
    [int]$EpisodeSteps = 4,
    [int]$BaselineBusCount = 8,
    [string]$OutputRoot = ".\artifacts\causal_simulator_v2_rollout_smoke"
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

& $py ".\05_training\run_causal_simulator_v2_rollout_smoke.py" `
  --project-root "." `
  --conditions $Conditions `
  --seeds $Seeds `
  --num-agents $NumAgents `
  --episode-steps $EpisodeSteps `
  --baseline-bus-count $BaselineBusCount `
  --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 102 actual rollout smoke failed"
}

Write-Host "[OK] Step 102 actual rollout smoke complete."
Write-Host "[OK] output_root = $OutputRoot"
