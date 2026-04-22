# =============================================================================
# run_gatv2_views.ps1
# Purpose : Build 5 GATv2 training views + run readiness checks.
#             - create_gatv2_training_views.sql
# Usage   : .\run_gatv2_views.ps1
# Prereq  : PostgreSQL client (psql.exe). On PATH, or set $env:PSQL_EXE.
# Author  : urbanbus_rl_project
# Date    : 2026-04-19
# Notes   : This file is intentionally ASCII-only to avoid Windows PowerShell
#           5.1 CP949 vs UTF-8 encoding issues. Do not edit with a tool that
#           re-encodes to UTF-8 without BOM and inserts non-ASCII characters.
# =============================================================================

# psql RAISE NOTICE / \echo goes to stderr. Under 'Stop' policy, PowerShell
# treats that as NativeCommandError. Force 'Continue' and judge success by
# $LASTEXITCODE only.
$ErrorActionPreference = "Continue"

try {
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $PSNativeCommandUseErrorActionPreference = $false
    }
} catch { }

# ---- Connection env --------------------------------------------------------
$PGHOST     = $env:PGHOST     ; if (-not $PGHOST)     { $PGHOST     = "localhost" }
$PGPORT     = $env:PGPORT     ; if (-not $PGPORT)     { $PGPORT     = "5432" }
$PGUSER     = $env:PGUSER     ; if (-not $PGUSER)     { $PGUSER     = "postgres" }
$PGDATABASE = $env:PGDATABASE ; if (-not $PGDATABASE) { $PGDATABASE = "urbanbus" }

# ---- psql.exe locator ------------------------------------------------------
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
    Write-Host "       Fix: set env var PSQL_EXE to the full path, e.g." -ForegroundColor Gray
    Write-Host '              $env:PSQL_EXE = "C:\Program Files\PostgreSQL\16\bin\psql.exe"' -ForegroundColor Gray
    Write-Host "       then re-run .\run_gatv2_views.ps1" -ForegroundColor Gray
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir    = Join-Path $ScriptDir "logs_$Timestamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " GATv2 training views build + readiness" -ForegroundColor Cyan
Write-Host "   psql     = $PSQL"
Write-Host "   host     = $PGHOST`:$PGPORT"
Write-Host "   db       = $PGDATABASE"
Write-Host "   user     = $PGUSER"
Write-Host "   log dir  = $LogDir"
Write-Host "=============================================================" -ForegroundColor Cyan
& $PSQL --version

$Steps = @(
    @{ Name = "create_views"; File = "create_gatv2_training_views.sql" }
)

$overallStart = Get-Date
foreach ($step in $Steps) {
    $sqlPath = Join-Path $ScriptDir $step.File
    $logPath = Join-Path $LogDir ("{0}.log" -f $step.Name)

    if (-not (Test-Path $sqlPath)) {
        Write-Host "[FAIL] SQL file not found: $sqlPath" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "[RUN ] $($step.Name) -> $($step.File)" -ForegroundColor Yellow
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
        Write-Host "[FAIL] step '$($step.Name)' failed. log: $logPath" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    $stepDur = (Get-Date) - $stepStart
    $stepSecs = [math]::Round($stepDur.TotalSeconds, 1)
    Write-Host "[DONE] $($step.Name)  (${stepSecs}s)  log: $logPath" -ForegroundColor Green
}

$overallDur = (Get-Date) - $overallStart
$overallSecs = [math]::Round($overallDur.TotalSeconds, 1)
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " All steps OK (${overallSecs}s total)" -ForegroundColor Green
Write-Host " Logs: $LogDir" -ForegroundColor Gray
Write-Host "=============================================================" -ForegroundColor Cyan

# Readiness summary: grep the key metric lines out of the log.
$readinessLog = Join-Path $LogDir "create_views.log"
if (Test-Path $readinessLog) {
    Write-Host ""
    Write-Host " >>> Readiness summary (key lines) <<<" -ForegroundColor Magenta
    $patterns = "num_nodes|num_edges|orphan|self_loop|overnight_leak|terminal_leak|distinct_snapshots|total_training_rows|nodes_without_any_training_row|negative_target_cnt|min_state_ts|max_state_ts|avg_rows_per_snapshot|rank1_cnt|index_gap_check_zero_ok"
    Get-Content $readinessLog | Select-String -Pattern $patterns | ForEach-Object {
        Write-Host "   $_" -ForegroundColor Gray
    }
}
