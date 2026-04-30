param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [switch]$RunBuilderAfterPatch,
    [string]$SignalCsv = ".\data\source\daegu_signal\대구광역시_신호등_20251231.csv",
    [string]$OutputDir = ".\artifacts\signal_features_v2",
    [string]$DbUrl = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$BuilderPath = ".\05_training\adapters\build_signal_features_for_causal_simulator_v2.py"

if (-not (Test-Path $BuilderPath)) {
    throw "[STOP] builder not found: $BuilderPath"
}

$text = Get-Content -Path $BuilderPath -Raw -Encoding UTF8

# Current DB view uses node_idx, while the Step 98 builder expected node_index.
# Keep downstream output column as node_index by aliasing n.node_idx AS node_index.
$text2 = $text `
    -replace "n\.node_index,", "n.node_idx AS node_index," `
    -replace "ORDER BY n\.node_index", "ORDER BY n.node_idx"

if ($text2 -eq $text) {
    Write-Host "[WARN] No node_index text replacement was applied. The file may already be patched."
} else {
    $text2 | Set-Content -Path $BuilderPath -Encoding UTF8
    Write-Host "[OK] Patched node_index reference to node_idx AS node_index"
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

& $py -m py_compile $BuilderPath
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] patched builder py_compile failed"
}

Write-Host "[OK] Step 99 node_idx simple patch complete"

if ($RunBuilderAfterPatch) {
    if ([string]::IsNullOrWhiteSpace($DbUrl)) {
        $DbUrl = $env:URBANBUS_DB_DSN
    }
    if ([string]::IsNullOrWhiteSpace($DbUrl)) {
        throw "[STOP] DbUrl is empty. Set URBANBUS_DB_DSN or pass -DbUrl."
    }
    if (-not (Test-Path $SignalCsv)) {
        throw "[STOP] Signal CSV not found: $SignalCsv"
    }

    & $py ".\05_training\adapters\build_signal_features_for_causal_simulator_v2.py" `
      --signal-csv $SignalCsv `
      --db-url $DbUrl `
      --output-dir $OutputDir

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 99 DB-based signal feature build failed after node_idx patch"
    }

    Write-Host "[DONE] Step 99 DB-based signal feature build complete after node_idx patch."
}
