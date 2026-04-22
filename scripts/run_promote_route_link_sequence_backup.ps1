<#
.SYNOPSIS
    Guarded pipeline to promote bus route links with Pre/Post validation gates.
.DESCRIPTION
    1. Ensures target table exists.
    2. Runs Pre-validation (Staging audit). Aborts if fails.
    3. Runs Promotion (Staging -> Refined).
    4. Runs Post-validation (Refined audit). Aborts if fails.
#>
param(
    [string]$DbName = "urbanbus",
    [string]$DbUser = "postgres",
    [string]$DbHost = "localhost",
    [int]$DbPort = 5432,
    [string]$PsqlPath = "C:\Program Files\PostgreSQL\18\bin\psql.exe",
    [string]$RouteId,
    [string]$SnapshotFile
)

$ErrorActionPreference = "Stop"

# Helper to run psql and get raw output or execute file
function Run-Psql {
    param([string]$FilePath, [string]$Command, [bool]$Quiet = $false)
    $baseArgs = @("-h", $DbHost, "-p", $DbPort, "-U", $DbUser, "-d", $DbName, "-v", "ON_ERROR_STOP=1")
    if ($Quiet) { $baseArgs += @("-t", "-A") } # Tuples only, unaligned for getting counts

    if ($null -ne $Command) {
        return (& $PsqlPath $baseArgs -c $Command).Trim()
    } else {
        & $PsqlPath $baseArgs -f $FilePath
    }
}

# Scope Filter Builder
function Get-FilterSql {
    param([string]$Prefix = "")
    $filters = "WHERE 1=1 "
    if ($null -ne $RouteId) { $filters += "AND $($Prefix)route_id_raw = '$RouteId' " }
    if ($null -ne $SnapshotFile) { $filters += "AND snapshot_file = '$SnapshotFile' " }
    return $filters
}

try {
    Write-Host "--- [STAGE 1] Ensuring Target Table ---" -ForegroundColor Cyan
    Run-Psql -FilePath "d:\urbanbus_rl_project\scripts\create_route_link_sequence.sql"

    # --- [STAGE 2] PRE-VALIDATION ---
    Write-Host "--- [STAGE 2] PRE-VALIDATION (Staging Audit) ---" -ForegroundColor Cyan
    $filter = Get-FilterSql
    
    $unkCnt = Run-Psql -Quiet -Command "SELECT count(*) FROM public.stg_daegu_route_links_api $filter AND (route_id_raw IS NULL OR route_id_raw IN ('', 'UNKNOWN'))"
    $badSeqCnt = Run-Psql -Quiet -Command "SELECT count(*) FROM public.stg_daegu_route_links_api $filter AND link_seq_raw !~ '^\d+$'"
    $dupStgCnt = Run-Psql -Quiet -Command "SELECT count(*) FROM (SELECT route_id_raw, move_dir_raw, link_seq_raw FROM public.stg_daegu_route_links_api $filter GROUP BY route_id_raw, move_dir_raw, link_seq_raw HAVING count(*) > 1) AS t"

    if ($unkCnt -gt 0 -or $badSeqCnt -gt 0 -or $dupStgCnt -gt 0) {
        Write-Host "[FAIL] Pre-validation failed! (UNKNOWN: $unkCnt, BadSeq: $badSeqCnt, Dup: $dupStgCnt)" -ForegroundColor Red
        Write-Host "Please check 'scripts/validate_route_link_sequence.sql' for details." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[PASS] Staging data looks clean." -ForegroundColor Green

    # --- [STAGE 3] PROMOTION ---
    Write-Host "--- [STAGE 3] PROMOTION (Staging -> Refined) ---" -ForegroundColor Cyan
    if ($null -ne $RouteId -or $null -ne $SnapshotFile) {
        $promoteSql = "INSERT INTO public.route_link_sequence (route_id, move_dir_code, link_seq, link_id, st_node_id, ed_node_id, gis_dist, snapshot_file) " +
                      "SELECT route_id_raw, move_dir_raw, link_seq_raw::int, link_id_raw, st_node_raw, ed_node_raw, " +
                      "CASE WHEN gis_dist_raw ~ '^\d+(\.\d+)?$' THEN gis_dist_raw::numeric ELSE NULL END, snapshot_file " +
                      "FROM public.stg_daegu_route_links_api $(Get-FilterSql) " +
                      "AND link_seq_raw ~ '^\d+$' AND route_id_raw NOT IN ('', 'UNKNOWN') " +
                      "ON CONFLICT (route_id, move_dir_code, link_seq) DO UPDATE SET " +
                      "link_id = EXCLUDED.link_id, st_node_id = EXCLUDED.st_node_id, ed_node_id = EXCLUDED.ed_node_id, " +
                      "gis_dist = EXCLUDED.gis_dist, snapshot_file = EXCLUDED.snapshot_file;"
        Run-Psql -Command $promoteSql
    } else {
        Run-Psql -FilePath "d:\urbanbus_rl_project\scripts\promote_route_link_sequence.sql"
    }

    # --- [STAGE 4] POST-VALIDATION ---
    Write-Host "--- [STAGE 4] POST-VALIDATION (Refined Audit) ---" -ForegroundColor Cyan
    # Scope for analytical table doesn't have _raw suffix
    $postFilter = "WHERE 1=1 "
    if ($null -ne $RouteId) { $postFilter += "AND route_id = '$RouteId' " }
    if ($null -ne $SnapshotFile) { $postFilter += "AND snapshot_file = '$SnapshotFile' " }

    $dupPromCnt = Run-Psql -Quiet -Command "SELECT count(*) FROM (SELECT route_id, move_dir_code, link_seq FROM public.route_link_sequence $postFilter GROUP BY route_id, move_dir_code, link_seq HAVING count(*) > 1) AS t"
    $contCnt = Run-Psql -Quiet -Command "SELECT count(*) FROM (SELECT route_id, move_dir_code FROM public.route_link_sequence $postFilter GROUP BY route_id, move_dir_code HAVING (max(link_seq) - min(link_seq) + 1) != count(*)) AS t"

    if ($dupPromCnt -gt 0 -or $contCnt -gt 0) {
        Write-Host "[CRITICAL] Promotion completed BUT Validation FAILED!" -ForegroundColor Red
        Write-Host "Integrity issues found: PromotedDup: $dupPromCnt, ContinuityBreak: $contCnt" -ForegroundColor DarkRed
        exit 1
    }

    Write-Host "[SUCCESS] Promotion and Validation completed successfully." -ForegroundColor Green
    # Output final summary report
    Run-Psql -FilePath "d:\urbanbus_rl_project\scripts\validate_route_link_sequence.sql"

} catch {
    Write-Error "Error in promotion pipeline: $($_.Exception.Message)"
    exit 1
}
