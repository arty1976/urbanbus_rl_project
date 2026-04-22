# =============================================================================
# run_materialize.ps1
# Purpose : One-shot runner for materialize_gatv2_training_table.sql.
#           Expect ~2h runtime. Logs full psql output.
# Usage   : .\run_materialize.ps1
# Prereq  : psql.exe on PATH or $env:PSQL_EXE set.
# Author  : urbanbus_rl_project
# Date    : 2026-04-19
# Notes   : ASCII-only to avoid PowerShell 5.1 CP949 parse errors.
# =============================================================================

$ErrorActionPreference = "Continue"

try {
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $PSNativeCommandUseErrorActionPreference = $false
    }
} catch { }

$PGHOST     = $env:PGHOST     ; if (-not $PGHOST)     { $PGHOST     = "localhost" }
$PGPORT     = $env:PGPORT     ; if (-not $PGPORT)     { $PGPORT     = "5432" }
$PGUSER     = $env:PGUSER     ; if (-not $PGUSER)     { $PGUSER     = "postgres" }
$PGDATABASE = $env:PGDATABASE ; if (-not $PGDATABASE) { $PGDATABASE = "urbanbus" }

function Find-Psql {
    if ($env:PSQL_EXE -and (Test-Path $env:PSQL_EXE)) { return $env:PSQL_EXE }
    if ($env:PGBIN) {
        $candidate = Join-Path $env:PGBIN "psql.exe"
        if (Test-Path $candidate) { return $candidate }
    }
    $onPath = Get-Command psql.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $baseDirs = @("$env:ProgramFiles\PostgreSQL", "${env:ProgramFiles(x86)}\PostgreSQL")
    foreach ($base in $baseDirs) {
        if (-not (Test-Path $base)) { continue }
        $versions = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending
        foreach ($v in $versions) {
            $candidate = Join-Path $v.FullName "bin\psql.exe"
            if (Test-Path $candidate) { return $candidate }
        }
    }
    return $null
}

$PSQL = Find-Psql
if (-not $PSQL) {
    Write-Host "[FAIL] psql.exe not found." -ForegroundColor Red
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir    = Join-Path $ScriptDir "logs_materialize_$Timestamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$sqlPath = Join-Path $ScriptDir "materialize_gatv2_training_table.sql"
$logPath = Join-Path $LogDir "materialize.log"

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " GATv2 training table MATERIALIZATION" -ForegroundColor Cyan
Write-Host "   psql     = $PSQL"
Write-Host "   db       = $PGDATABASE"
Write-Host "   sql      = $sqlPath"
Write-Host "   log      = $logPath"
Write-Host "   expect   = ~2h runtime, 20.29M rows copied" -ForegroundColor Yellow
Write-Host "=============================================================" -ForegroundColor Cyan
& $PSQL --version

if (-not (Test-Path $sqlPath)) {
    Write-Host "[FAIL] SQL file not found: $sqlPath" -ForegroundColor Red
    exit 1
}

$stepStart = Get-Date
& $PSQL `
    -h $PGHOST `
    -p $PGPORT `
    -U $PGUSER `
    -d $PGDATABASE `
    -v ON_ERROR_STOP=1 `
    -f $sqlPath `
    *>&1 | Tee-Object -FilePath $logPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] materialization failed. log: $logPath" -ForegroundColor Red
    exit $LASTEXITCODE
}

$stepDur = (Get-Date) - $stepStart
$stepSecs = [math]::Round($stepDur.TotalSeconds, 1)
Write-Host ""
Write-Host "[DONE] materialize  (${stepSecs}s)" -ForegroundColor Green
Write-Host " log: $logPath" -ForegroundColor Gray

# Summary grep
Write-Host ""
Write-Host " >>> Materialization summary <<<" -ForegroundColor Magenta
$patterns = "materialized_rows|mat_rows|view_total_rows|mat_snapshots|view_snapshots|parity_check|MATERIALIZATION"
Get-Content $logPath | Select-String -Pattern $patterns | ForEach-Object {
    Write-Host "   $_" -ForegroundColor Gray
}
