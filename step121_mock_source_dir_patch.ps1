$ErrorActionPreference = "Stop"

Set-Location "C:\Users\ryujo\urbanbus_rl_project"

$TestPath = ".\05_training\rewards\test_validate_reward_ablation_execution_manifest_step121.py"

if (-not (Test-Path $TestPath)) {
    throw "[STOP] Step 121 test file not found: $TestPath"
}

$text = Get-Content $TestPath -Raw -Encoding UTF8

$old = @'
def write_mock_step119_plan(source_root: Path) -> Path:
    csv_path = source_root / "reward_ablation_runner_dry_run_plan.csv"
'@

$new = @'
def write_mock_step119_plan(source_root: Path) -> Path:
    source_root.mkdir(parents=True, exist_ok=True)
    csv_path = source_root / "reward_ablation_runner_dry_run_plan.csv"
'@

if ($text -like "*source_root.mkdir(parents=True, exist_ok=True)*") {
    Write-Host "[OK] Step 121 mock source mkdir patch already present."
} else {
    if (-not $text.Contains($old)) {
        throw "[STOP] Target block not found in Step 121 test file. Do not rerun original Step 121 ps1."
    }
    $text = $text.Replace($old, $new)
    Set-Content -Path $TestPath -Value $text -Encoding UTF8
    Write-Host "[OK] patched Step 121 self-test mock source directory creation"
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 121 reward ablation execution manifest self-test failed after mock source mkdir patch"
}

Write-Host "[DONE] Step 121 mock source mkdir patch and self-test PASS"
