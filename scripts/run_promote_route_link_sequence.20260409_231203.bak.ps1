<#
.SYNOPSIS
    Guard-enabled promotion pipeline for route_link_sequence.
.DESCRIPTION
    1. Ensures target table exists.
    2. Runs scoped pre-validation and aborts on failure.
    3. Runs promotion (scoped inline SQL or full promote script).
    4. Runs scoped post-validation and aborts on failure.
    5. Emits final scoped validation report.
#>
[CmdletBinding()]
param(
    [string]$DbName = "urbanbus",
    [string]$DbUser = "postgres",
    [string]$DbHost = "localhost",
    [int]$DbPort = 5432,
    [string]$PsqlPath = "C:\Program Files\PostgreSQL\18\bin\psql.exe",
    [string]$RouteId,
    [string]$SnapshotFile,
    [string]$CreateSqlPath,
    [string]$PromoteSqlPath,
    [string]$ValidateSqlPath,
    [string]$LogPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { (Get-Location).Path }

if ([string]::IsNullOrWhiteSpace($CreateSqlPath)) {
    $CreateSqlPath = Join-Path $ScriptDir 'create_route_link_sequence.sql'
}
if ([string]::IsNullOrWhiteSpace($PromoteSqlPath)) {
    $PromoteSqlPath = Join-Path $ScriptDir 'promote_route_link_sequence.sql'
}
if ([string]::IsNullOrWhiteSpace($ValidateSqlPath)) {
    $ValidateSqlPath = Join-Path $ScriptDir 'validate_route_link_sequence.sql'
}
if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $LogPath = Join-Path $ScriptDir ("run_promote_route_link_sequence_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
}

function ConvertTo-SqlLiteral {
    param([AllowNull()][string]$Value)
    if ($null -eq $Value) { return 'NULL' }
    return "'" + $Value.Replace("'", "''") + "'"
}

function Get-ScopedStagingFilter {
    $parts = @('1=1')
    if (-not [string]::IsNullOrWhiteSpace($RouteId)) {
        $parts += "s.route_id_raw = $(ConvertTo-SqlLiteral $RouteId)"
    }
    if (-not [string]::IsNullOrWhiteSpace($SnapshotFile)) {
        $parts += "s.snapshot_file = $(ConvertTo-SqlLiteral $SnapshotFile)"
    }
    return ($parts -join ' AND ')
}

function Get-PsqlVariableMap {
    $routeValue = if ([string]::IsNullOrWhiteSpace($RouteId)) { '' } else { $RouteId }
    $snapshotValue = if ([string]::IsNullOrWhiteSpace($SnapshotFile)) { '' } else { $SnapshotFile }

    return @{
        route_id_raw = $routeValue
        snapshot_file = $snapshotValue
    }
}

function Run-Psql {
    param(
        [string]$FilePath,
        [string]$Command,
        [hashtable]$Variables,
        [switch]$Quiet
    )

    if ([string]::IsNullOrWhiteSpace($FilePath) -and [string]::IsNullOrWhiteSpace($Command)) {
        throw 'Run-Psql requires either -FilePath or -Command.'
    }

    $args = @('-X', '-h', $DbHost, '-p', $DbPort, '-U', $DbUser, '-d', $DbName, '-v', 'ON_ERROR_STOP=1')

    if ($Variables) {
        foreach ($key in ($Variables.Keys | Sort-Object)) {
            $args += @('-v', "${key}=$($Variables[$key])")
        }
    }

    if ($Quiet) {
        $args += @('-t', '-A')
    }

    if (-not [string]::IsNullOrWhiteSpace($FilePath)) {
        $args += @('-f', $FilePath)
    }
    else {
        $args += @('-c', $Command)
    }

    $raw = & $PsqlPath @args 2>&1
    $exitCode = $LASTEXITCODE
    $output = ($raw | Out-String).Trim()

    if ($output) {
        Write-Host $output
    }

    if ($exitCode -ne 0) {
        throw "psql failed (exit=$exitCode)."
    }

    return $output
}

function Invoke-ScopedPromotion {
    $scopeFilter = Get-ScopedStagingFilter

    $sql = @"
INSERT INTO public.route_link_sequence (
    route_id,
    move_dir_code,
    link_seq,
    link_id,
    st_node_id,
    ed_node_id,
    gis_dist,
    snapshot_file
)
SELECT
    s.route_id_raw::integer AS route_id,
    s.move_dir_raw::integer AS move_dir_code,
    s.link_seq_raw::integer AS link_seq,
    CASE
        WHEN btrim(COALESCE(s.link_id_raw::text, '')) ~ '^-?\\d+$' THEN s.link_id_raw::bigint
        ELSE NULL
    END AS link_id,
    CASE
        WHEN btrim(COALESCE(s.st_node_raw::text, '')) ~ '^-?\\d+$' THEN s.st_node_raw::bigint
        ELSE NULL
    END AS st_node_id,
    CASE
        WHEN btrim(COALESCE(s.ed_node_raw::text, '')) ~ '^-?\\d+$' THEN s.ed_node_raw::bigint
        ELSE NULL
    END AS ed_node_id,
    CASE
        WHEN btrim(COALESCE(s.gis_dist_raw::text, '')) ~ '^-?\\d+(\\.\\d+)?$' THEN s.gis_dist_raw::numeric
        ELSE NULL
    END AS gis_dist,
    s.snapshot_file
FROM public.stg_daegu_route_links_api s
WHERE $scopeFilter
  AND btrim(COALESCE(s.route_id_raw, '')) ~ '^\\d+$'
  AND s.route_id_raw::integer > 0
  AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\\d+$'
  AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\\d+$'
  AND s.link_seq_raw::integer > 0
ON CONFLICT (route_id, move_dir_code, link_seq)
DO UPDATE
SET
    link_id = EXCLUDED.link_id,
    st_node_id = EXCLUDED.st_node_id,
    ed_node_id = EXCLUDED.ed_node_id,
    gis_dist = EXCLUDED.gis_dist,
    snapshot_file = EXCLUDED.snapshot_file;
"@

    Run-Psql -Command $sql | Out-Null
}

try {
    if (-not (Test-Path -LiteralPath $PsqlPath)) {
        throw "psql.exe not found: $PsqlPath"
    }
    if (-not (Test-Path -LiteralPath $CreateSqlPath)) {
        throw "Create SQL not found: $CreateSqlPath"
    }
    if (-not (Test-Path -LiteralPath $ValidateSqlPath)) {
        throw "Validate SQL not found: $ValidateSqlPath"
    }
    if ([string]::IsNullOrWhiteSpace($RouteId) -and [string]::IsNullOrWhiteSpace($SnapshotFile) -and -not (Test-Path -LiteralPath $PromoteSqlPath)) {
        throw "Promote SQL not found: $PromoteSqlPath"
    }

    Start-Transcript -Path $LogPath -Force | Out-Null

    $scopeVars = Get-PsqlVariableMap

    Write-Host "--- [STAGE 0] Scope ---" -ForegroundColor Cyan
    Write-Host ("route_id_raw = {0}" -f ($(if ([string]::IsNullOrWhiteSpace($RouteId)) { '<ALL>' } else { $RouteId })))
    Write-Host ("snapshot_file = {0}" -f ($(if ([string]::IsNullOrWhiteSpace($SnapshotFile)) { '<ALL>' } else { $SnapshotFile })))
    Write-Host ("log = {0}" -f $LogPath)

    Write-Host "--- [STAGE 1] Ensuring Target Table ---" -ForegroundColor Cyan
    Run-Psql -FilePath $CreateSqlPath | Out-Null

    Write-Host "--- [STAGE 2] PRE-VALIDATION ---" -ForegroundColor Cyan
    $preVars = @{} + $scopeVars
    $preVars['phase'] = 'pre'
    Run-Psql -FilePath $ValidateSqlPath -Variables $preVars | Out-Null
    Write-Host "[PASS] Pre-validation completed." -ForegroundColor Green

    Write-Host "--- [STAGE 3] PROMOTION ---" -ForegroundColor Cyan
    if (-not [string]::IsNullOrWhiteSpace($RouteId) -or -not [string]::IsNullOrWhiteSpace($SnapshotFile)) {
        Invoke-ScopedPromotion
    }
    else {
        Run-Psql -FilePath $PromoteSqlPath | Out-Null
    }
    Write-Host "[PASS] Promotion completed." -ForegroundColor Green

    Write-Host "--- [STAGE 4] POST-VALIDATION ---" -ForegroundColor Cyan
    $postVars = @{} + $scopeVars
    $postVars['phase'] = 'post'
    Run-Psql -FilePath $ValidateSqlPath -Variables $postVars | Out-Null
    Write-Host "[PASS] Post-validation completed." -ForegroundColor Green

    Write-Host "--- [STAGE 5] FINAL REPORT ---" -ForegroundColor Cyan
    $allVars = @{} + $scopeVars
    $allVars['phase'] = 'all'
    Run-Psql -FilePath $ValidateSqlPath -Variables $allVars | Out-Null

    Write-Host "[SUCCESS] Promotion pipeline finished successfully." -ForegroundColor Green
    Write-Host ("[INFO] Transcript saved to: {0}" -f $LogPath) -ForegroundColor DarkGreen
    exit 0
}
catch {
    Write-Error ("Error in promotion pipeline: {0}" -f $_.Exception.Message)
    Write-Host ("[FAIL] See transcript for details: {0}" -f $LogPath) -ForegroundColor Red
    exit 1
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
}
