$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[INFO] project_root: $ProjectRoot"

$AnalysisDir = Join-Path $ProjectRoot "05_training\analysis"
if (-not (Test-Path $AnalysisDir)) {
    throw "[STOP] analysis folder not found: $AnalysisDir"
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $Py = "python"
}

$OutputRoot = Join-Path $ProjectRoot "artifacts\analysis\kpi_comparison_analyzer_step159"

Write-Host "[RUN] Step 159 generator (sample-mode)"
& $Py ".\05_training\analysis\kpi_comparison_analyzer_step159.py" `
    --project-root $ProjectRoot `
    --output-root $OutputRoot `
    --baseline-condition "B0" `
    --sample-mode

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 159 generator failed"
}

$Manifest = Join-Path $OutputRoot "kpi_comparison_manifest_step159.json"

Write-Host "[RUN] Step 159 validator"
& $Py ".\05_training\analysis\validate_kpi_comparison_analyzer_step159.py" `
    --manifest $Manifest

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 159 validator failed"
}

Write-Host "[RUN] Step 159 self-test"
& $Py ".\05_training\analysis\test_kpi_comparison_analyzer_step159.py"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 159 self-test failed"
}

Write-Host "[OK] Step 159 KPI comparison analyzer scaffold completed"
Write-Host "[OK] manifest: $Manifest"
Write-Host "[OK] status  : KPI_COMPARISON_ANALYZER_READY_SCAFFOLD_STILL_LOCKED"
Write-Host "[OK] decision: SCAFFOLD_ONLY_NOT_ACTUAL_COMPARISON"
Write-Host "[OK] next gate: WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT"
