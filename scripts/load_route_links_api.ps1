<#
.SYNOPSIS
    Loads getLink02 API snapshot JSON into PostgreSQL stg_daegu_route_links_api.

.DESCRIPTION
    Reads JSON files under data/raw/daegu/api_snapshots/, flattens the response
    array into a temporary CSV, and loads it via psql \copy into
    stg_daegu_route_links_api.

    route_id_raw is injected exclusively from the -RouteId parameter (request metadata).
    API response body routeId fields and filename-based inference are NOT used.

    Execution order:
      1. Parameter validation (RouteId is mandatory)
      2. Snapshot file existence check
      3. DDL execution (CREATE TABLE IF NOT EXISTS)
      4. JSON load and basic validation
      5. Duplicate load check
      6. Temporary CSV generation
      7. \copy load
      8. Row count log and cleanup

.PARAMETER RouteId
    [REQUIRED] Route identifier. Injected into staging as request metadata.
    Script exits immediately if missing or whitespace-only.
    Example: "1000" or "daegu-route-1"

.PARAMETER SnapshotFile
    Absolute path to the JSON snapshot file to process.
    If omitted, the latest file in api_snapshots/ is selected automatically.

.PARAMETER DbName
    Target PostgreSQL database name (default: urbanbus)

.PARAMETER DbUser
    PostgreSQL user (default: postgres)

.PARAMETER DbHost
    PostgreSQL host (default: localhost)

.PARAMETER DbPort
    PostgreSQL port (default: 5432)

.EXAMPLE
    # Normal run
    .\load_route_links_api.ps1 -RouteId "1000" -SnapshotFile "D:\...\daegu_bus_api_20260409_120000.json"

.EXAMPLE
    # Missing RouteId -> immediate error
    .\load_route_links_api.ps1 -SnapshotFile "D:\...\daegu_bus_api_20260409_120000.json"
    # ERROR: RouteId parameter is required. route_id_raw must come from request metadata, not response body or filename.
#>
param (
    [Parameter(Mandatory = $true)]
    [string]$RouteId,
    [string]$SnapshotFile = "",
    [string]$DbName       = "urbanbus",
    [string]$DbUser       = "postgres",
    [string]$DbHost       = "localhost",
    [int]   $DbPort       = 5432
)

$ErrorActionPreference = "Stop"

# Set client encoding to UTF8 to prevent psql UHC (CP949) character conversion errors
$env:PGCLIENTENCODING = "UTF8"

# -- [1/8] Parameter validation -----------------------------------------------
# [Parameter(Mandatory=$true)] is the first line of defense (PowerShell level).
# The check below is a second guard against whitespace-only strings.
if ([string]::IsNullOrWhiteSpace($RouteId)) {
    throw "RouteId parameter is required. route_id_raw must come from request metadata, not response body or filename."
}

# -- Path setup ---------------------------------------------------------------
$ScriptDir    = if (-not [string]::IsNullOrEmpty($PSScriptRoot)) {
    $PSScriptRoot
} elseif (-not [string]::IsNullOrEmpty($MyInvocation.MyCommand.Path)) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    (Get-Location).Path
}
$WorkspaceDir = Split-Path -Parent $ScriptDir
$SnapshotDir  = Join-Path $WorkspaceDir "data\raw\daegu\api_snapshots"
$TempDir      = Join-Path $WorkspaceDir "data\staging\daegu\tmp"
$CreateSql    = Join-Path $WorkspaceDir "scripts\create_stg_daegu_route_links_api.sql"

# -- psql presence check ------------------------------------------------------
if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
    throw "psql not found. Install PostgreSQL client tools or add psql to PATH."
}

# -- [2/8] Snapshot file existence check --------------------------------------
if ([string]::IsNullOrWhiteSpace($SnapshotFile)) {
    $Latest = Get-ChildItem -Path $SnapshotDir -Filter "daegu_bus_api_*.json" |
              Sort-Object LastWriteTime -Descending |
              Select-Object -First 1
    if (-not $Latest) {
        throw "No snapshot files found in: $SnapshotDir"
    }
    $SnapshotFile = $Latest.FullName
    Write-Host "[INFO] Auto-selected latest snapshot: $SnapshotFile"
}

if (-not (Test-Path $SnapshotFile)) {
    throw "Snapshot file not found: $SnapshotFile"
}

$SnapshotFileName = Split-Path -Leaf $SnapshotFile
Write-Host "[INFO] RouteId (request metadata): $RouteId"
Write-Host "[INFO] SnapshotFile: $SnapshotFileName"

# -- [3/8] DDL execution (CREATE TABLE IF NOT EXISTS) -------------------------
# Run before duplicate check to guarantee the table exists.
Write-Host "[1/5] Ensuring staging table exists (CREATE TABLE IF NOT EXISTS)..."
& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 -f $CreateSql
if ($LASTEXITCODE -ne 0) { throw "DDL execution failed: $CreateSql" }

# -- [4/8] JSON load and basic validation -------------------------------------
Write-Host "[2/5] Loading and validating JSON structure..."
$Json = Get-Content -Path $SnapshotFile -Raw -Encoding UTF8 | ConvertFrom-Json

# TODO: Adjust the path below to match the actual API response structure.
# Actual API response structure: $Json.body.items
$Items = $Json.body.items
if (-not $Items) {
    throw "No items found in JSON response. Check the JSON structure and update the Items path in this script."
}

# -- [5/8] Duplicate load check -----------------------------------------------
# Table is guaranteed to exist at this point (DDL already ran).
Write-Host "[3/5] Checking for duplicate snapshot..."
# Escape single quotes in filename to prevent SQL syntax errors.
$SnapshotFileNameSql = $SnapshotFileName -replace "'", "''"
$DupCount = & psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -A `
    -c "SELECT COUNT(*) FROM stg_daegu_route_links_api WHERE snapshot_file = '$SnapshotFileNameSql';"
if ($LASTEXITCODE -ne 0) { throw "Duplicate check query failed." }

if ([int]($DupCount.Trim()) -gt 0) {
    Write-Host "[SKIP] Snapshot already loaded: $SnapshotFileName"
    exit 0
}

# -- [6/8] Temporary CSV generation -------------------------------------------
Write-Host "[4/5] Converting to CSV..."
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

$TempCsv   = Join-Path $TempDir ("route_links_" + [System.IO.Path]::GetFileNameWithoutExtension($SnapshotFileName) + ".csv")
$CsvHeader = "route_id_raw,link_id_raw,st_node_raw,ed_node_raw,gis_dist_raw,move_dir_raw,link_seq_raw,snapshot_file"
$Lines     = @($CsvHeader)

foreach ($Item in $Items) {
    # route_id_raw: always from $RouteId (request metadata).
    # API response routeId field and filename-based inference are NOT used.
    $Row = @(
        ($RouteId             -replace '"','""'),
        ($Item.linkId         -replace '"','""'),
        ($Item.stNode         -replace '"','""'),
        ($Item.edNode         -replace '"','""'),
        ($Item.gisDist        -replace '"','""'),
        ($Item.moveDir        -replace '"','""'),
        ($Item.linkSeq        -replace '"','""'),
        ($SnapshotFileName    -replace '"','""')
    ) | ForEach-Object { '"' + $_ + '"' }
    $Lines += $Row -join ","
}

[System.IO.File]::WriteAllLines($TempCsv, $Lines, [System.Text.Encoding]::UTF8)
$RowCount = $Lines.Count - 1
Write-Host "[INFO] Rows converted: $RowCount"

# -- [7/8] \copy load ---------------------------------------------------------
Write-Host "[5/5] Loading into PostgreSQL..."
$CsvPsqlPath = $TempCsv -replace '\\', '/'
& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 `
    -c "\copy stg_daegu_route_links_api(route_id_raw, link_id_raw, st_node_raw, ed_node_raw, gis_dist_raw, move_dir_raw, link_seq_raw, snapshot_file) FROM '$CsvPsqlPath' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
if ($LASTEXITCODE -ne 0) { throw "\copy load failed." }

# -- [8/8] Log and cleanup ----------------------------------------------------
Remove-Item -Path $TempCsv -Force
Write-Host "[INFO] Load complete. route_id_raw=$RouteId | rows=$RowCount | snapshot=$SnapshotFileName"
Write-Host "Done."
