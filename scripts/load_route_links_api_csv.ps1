<#
.SYNOPSIS
    Loads getLink02 API snapshot CSV into PostgreSQL stg_daegu_route_links_api.

.DESCRIPTION
    Reads a CSV file (e.g., response_1775701914078.csv), transforms it into
    the staging format, and loads it via psql \copy.

    Execution order:
      1. Parameter validation (RouteId and CsvFile are mandatory)
      2. DDL execution (Ensures table exists)
      3. Duplicate check (Based on filename)
      4. Temporary CSV generation (Maps columns and injects RouteId)
      5. \copy load
      6. Cleanup

.PARAMETER RouteId
    [REQUIRED] Route identifier (request metadata).
    Example: "1000"

.PARAMETER CsvFile
    [REQUIRED] Path to the source CSV file.
    Example: "D:\urbanbus_rl_project\data\raw\daegu\api_snapshots\response_1775701914078.csv"

.PARAMETER DbName
    Database name (default: urbanbus)

.PARAMETER DbUser
    Database user (default: postgres)

.PARAMETER DbHost
    Database host (default: localhost)

.PARAMETER DbPort
    Database port (default: 5432)

.PARAMETER PsqlPath
    Optional path to psql.exe. If not provided, the script looks for 'psql' in PATH.

.EXAMPLE
    .\scripts\load_route_links_api_csv.ps1 -RouteId "1000" -CsvFile "D:\urbanbus_rl_project\data\raw\daegu\api_snapshots\response_1775701914078.csv"
#>
param (
    [Parameter(Mandatory = $true)]
    [string]$RouteId,

    [Parameter(Mandatory = $true)]
    [string]$CsvFile,

    [string]$DbName   = "urbanbus",
    [string]$DbUser   = "postgres",
    [string]$DbHost   = "localhost",
    [int]   $DbPort   = 5432,
    [string]$PsqlPath = ""
)

$ErrorActionPreference = "Stop"

# -- [1/6] Parameter validation -----------------------------------------------
if ([string]::IsNullOrWhiteSpace($RouteId)) {
    throw "RouteId parameter is required."
}
if (-not (Test-Path $CsvFile)) {
    throw "CsvFile not found: $CsvFile"
}

# psql check
$PsqlCmd = if ([string]::IsNullOrWhiteSpace($PsqlPath)) { "psql" } else { $PsqlPath }
if (-not (Get-Command $PsqlCmd -ErrorAction SilentlyContinue)) {
    throw "psql not found. Provide -PsqlPath or add psql to PATH."
}

# Path setup
$ScriptDir    = if (-not [string]::IsNullOrEmpty($PSScriptRoot)) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$WorkspaceDir = Split-Path -Parent $ScriptDir
$TempDir      = Join-Path $WorkspaceDir "data\staging\daegu\tmp"
$CreateSql    = Join-Path $WorkspaceDir "scripts\create_stg_daegu_route_links_api.sql"
$CsvFileName  = Split-Path -Leaf $CsvFile

Write-Host "[INFO] RouteId: $RouteId"
Write-Host "[INFO] CsvFile: $CsvFileName"

# -- [2/6] DDL execution (Conditional) ----------------------------------------
Write-Host "[1/5] Checking if staging table exists..."
$TableExists = & $PsqlCmd -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -A `
    -c "SELECT count(*) FROM pg_tables WHERE schemaname = 'public' AND tablename = 'stg_daegu_route_links_api';"
if ($LASTEXITCODE -ne 0) { throw "Failed to check table existence." }

if ([int]($TableExists.Trim()) -gt 0) {
    Write-Host "[INFO] Staging table already exists. Skipping DDL."
} else {
    Write-Host "[INFO] Staging table not found. Executing DDL..."
    & $PsqlCmd -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 -f $CreateSql
    if ($LASTEXITCODE -ne 0) { throw "DDL execution failed." }
}

# -- [3/6] Duplicate check ----------------------------------------------------
Write-Host "[2/5] Checking for duplicate snapshot..."
$FileNameSql = $CsvFileName -replace "'", "''"
$DupCount = & $PsqlCmd -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -A `
    -c "SELECT COUNT(*) FROM stg_daegu_route_links_api WHERE snapshot_file = '$FileNameSql';"
if ($LASTEXITCODE -ne 0) { throw "Duplicate check query failed." }

if ([int]($DupCount.Trim()) -gt 0) {
    Write-Host "[SKIP] Snapshot already loaded: $CsvFileName"
    exit 0
}

# -- [4/6] Temporary CSV generation (Transformation) --------------------------
Write-Host "[3/5] Transforming CSV data..."
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

$TempCsv = Join-Path $TempDir ("route_links_csv_" + [System.IO.Path]::GetFileNameWithoutExtension($CsvFileName) + ".csv")

# Load source CSV
# Expected headers: linkId, stNode, edNode, gisDist, moveDir, linkSeq
$Data = Import-Csv -Path $CsvFile -Encoding UTF8

$CsvHeader = "route_id_raw,link_id_raw,st_node_raw,ed_node_raw,gis_dist_raw,move_dir_raw,link_seq_raw,snapshot_file"
$Lines = @($CsvHeader)

foreach ($Row in $Data) {
    $OutRow = @(
        ($RouteId     -replace '"','""'), # route_id_raw
        ($Row.linkId  -replace '"','""'), # link_id_raw
        ($Row.stNode  -replace '"','""'), # st_node_raw
        ($Row.edNode  -replace '"','""'), # ed_node_raw
        ($Row.gisDist -replace '"','""'), # gis_dist_raw
        ($Row.moveDir -replace '"','""'), # move_dir_raw
        ($Row.linkSeq -replace '"','""'), # link_seq_raw
        ($CsvFileName -replace '"','""')  # snapshot_file
    ) | ForEach-Object { '"' + $_ + '"' }
    $Lines += $OutRow -join ","
}

[System.IO.File]::WriteAllLines($TempCsv, $Lines, [System.Text.Encoding]::UTF8)
$RowCount = $Lines.Count - 1
Write-Host "[INFO] Rows transformed: $RowCount"

# -- [5/6] \copy load ---------------------------------------------------------
Write-Host "[4/5] Loading into PostgreSQL..."
$CsvPsqlPath = $TempCsv -replace '\\', '/'
& $PsqlCmd -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 `
    -c "\copy stg_daegu_route_links_api(route_id_raw, link_id_raw, st_node_raw, ed_node_raw, gis_dist_raw, move_dir_raw, link_seq_raw, snapshot_file) FROM '$CsvPsqlPath' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
if ($LASTEXITCODE -ne 0) { throw "\copy load failed." }

# -- [6/6] Cleanup ------------------------------------------------------------
Remove-Item -Path $TempCsv -Force
Write-Host "[5/5] Load complete. RouteId=$RouteId | rows=$RowCount | file=$CsvFileName"
Write-Host "Done."
