<#
.SYNOPSIS
    Generic Bus Route Link Loader for Antigravity Skill: bus_route_link_ingest_guard.
    Handles JSON/CSV ingestion with metadata enforcement.

.DESCRIPTION
    1. Detects file format (JSON/CSV).
    2. Ensures staging table exists (Region-agnostic naming).
    3. Injects RouteId from metadata only.
    4. Performs psql \copy for high performance.
    5. Clean up temporary files.

.PARAMETER Region
    Target region name (e.g., "seoul"). Used for table naming: stg_{Region}_route_links_api.

.PARAMETER RouteId
    The identifier to assign to all rows in this snapshot. (MANDATORY metadata)

.PARAMETER InputFile
    Path to the JSON or CSV snapshot.

.PARAMETER DbName
    PostgreSQL Database (default: urbanbus)

.PARAMETER JsonItemPath
    Dot notation path to the items array in JSON. (default: response.body.items.item)
#>
param (
    [Parameter(Mandatory = $true)]
    [string]$Region,

    [Parameter(Mandatory = $true)]
    [string]$RouteId,

    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [string]$DbName   = "urbanbus",
    [string]$DbUser   = "postgres",
    [string]$DbHost   = "localhost",
    [int]   $DbPort   = 5432,
    [string]$PsqlPath = "psql",
    [string]$JsonItemPath = "response.body.items.item",
    [string]$TargetTable = ""
)

$ErrorActionPreference = "Stop"

# -- [1] Validation -----------------------------------------------------------
if (-not (Test-Path $InputFile)) { throw "InputFile not found: $InputFile" }
if (-not (Get-Command $PsqlPath -ErrorAction SilentlyContinue)) { throw "psql not found." }

$FileName = Split-Path -Leaf $InputFile
$Extension = [System.IO.Path]::GetExtension($InputFile).ToLower()

# Set TargetTable if not provided
if ([string]::IsNullOrWhiteSpace($TargetTable)) {
    $TargetTable = "public.stg_${Region}_route_links_api"
}

Write-Host "[INFO] Region: $Region | Target Table: $TargetTable"
Write-Host "[INFO] RouteId: $RouteId | Source File: $FileName"

# -- [2] Table & DDL Check ----------------------------------------------------
Write-Host "[1/5] Checking table existence..."

# Parse schema and table name (e.g., practice.stg_links -> schema=practice, table=stg_links)
$SchemaPart = "public"
$TablePart  = $TargetTable
if ($TargetTable.Contains(".")) {
    $Parts = $TargetTable.Split(".")
    $SchemaPart = $Parts[0]
    $TablePart  = $Parts[1]
}

$TableExists = & $PsqlPath -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -A `
    -c "SELECT count(*) FROM pg_tables WHERE schemaname = '$SchemaPart' AND tablename = '$TablePart';"

if ([int]($TableExists.Trim()) -eq 0) {
    Write-Host "[WARN] Table $TargetTable not found. Please provide a DDL script or run create script first."
    # Note: In a real skill, we might have a generic DDL template here.
    # For now, we assume the user has a script or we throw.
    throw "Target table $TargetTable does not exist. Create it first using the regional DDL script."
} else {
    Write-Host "[INFO] Staging table exists. Skipping DDL."
}

# -- [3] Duplicate Check ------------------------------------------------------
$FileSql = $FileName -replace "'", "''"
$DupCount = & $PsqlPath -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -A `
    -c "SELECT count(*) FROM $TargetTable WHERE snapshot_file = '$FileSql';"

if ([int]($DupCount.Trim()) -gt 0) {
    Write-Host "[SKIP] Snapshot '$FileName' is already loaded for region '$Region'."
    exit 0
}

# -- [4] Transformation -------------------------------------------------------
Write-Host "[2/5] Transforming $Extension data..."
$TempDir = Join-Path $HOME "AppData\Local\Temp\antigravity_ingest"
if (-not (Test-Path $TempDir)) { New-Item -ItemType Directory -Path $TempDir -Force | Out-Null }
$TempCsv = Join-Path $TempDir ("ingest_" + [Guid]::NewGuid().ToString() + ".csv")

$CsvHeader = "route_id_raw,link_id_raw,st_node_raw,ed_node_raw,gis_dist_raw,move_dir_raw,link_seq_raw,snapshot_file"
$Lines = @($CsvHeader)

try {
    if ($Extension -eq ".json") {
        $Json = Get-Content -Path $InputFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $Items = $Json
        foreach ($segment in $JsonItemPath.Split('.')) { $Items = $Items.$segment }
        
        if (-not $Items) { throw "JSON items not found at path: $JsonItemPath" }

        foreach ($Item in $Items) {
            $Row = @(
                ($RouteId      -replace '"','""'),
                ($Item.linkId  -replace '"','""'),
                ($Item.stNode  -replace '"','""'),
                ($Item.edNode  -replace '"','""'),
                ($Item.gisDist -replace '"','""'),
                ($Item.moveDir -replace '"','""'),
                ($Item.linkSeq -replace '"','""'),
                ($FileName     -replace '"','""')
            ) | ForEach-Object { '"' + $_ + '"' }
            $Lines += $Row -join ","
        }
    } 
    elseif ($Extension -eq ".csv") {
        $Data = Import-Csv -Path $InputFile -Encoding UTF8
        foreach ($Row in $Data) {
            # Map CSV columns properly
            $CsvRow = @(
                ($RouteId      -replace '"','""'),
                ($Row.linkId   -replace '"','""'),
                ($Row.stNode   -replace '"','""'),
                ($Row.edNode   -replace '"','""'),
                ($Row.gisDist  -replace '"','""'),
                ($Row.moveDir  -replace '"','""'),
                ($Row.linkSeq  -replace '"','""'),
                ($FileName     -replace '"','""')
            ) | ForEach-Object { '"' + $_ + '"' }
            $Lines += $CsvRow -join ","
        }
    }
    else {
        throw "Unsupported file extension: $Extension"
    }

    [System.IO.File]::WriteAllLines($TempCsv, $Lines, [System.Text.Encoding]::UTF8)
    $RowCount = $Lines.Count - 1
    Write-Host "[INFO] Transformation complete. Rows: $RowCount"

    # -- [5] Load -------------------------------------------------------------
    Write-Host "[3/5] Loading into PostgreSQL..."
    $CsvPsqlPath = $TempCsv -replace '\\', '/'
    & $PsqlPath -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 `
        -c "\copy $TargetTable(route_id_raw, link_id_raw, st_node_raw, ed_node_raw, gis_dist_raw, move_dir_raw, link_seq_raw, snapshot_file) FROM '$CsvPsqlPath' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
    
    if ($LASTEXITCODE -ne 0) { throw "psql \copy failed." }

    Write-Host "[INFO] Load successful."

    # -- [6] Verify -----------------------------------------------------------
    Write-Host "[4/5] Running verification..."
    $VerifySql = "SELECT count(*) FROM $TargetTable WHERE route_id_raw = '$RouteId' AND snapshot_file = '$FileSql';"
    $LoadedCount = & $PsqlPath -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -A -c $VerifySql
    Write-Host "[VERIFY] Rows in DB for current metadata: $LoadedCount"

    $UnknownCount = & $PsqlPath -h $DbHost -p $DbPort -U $DbUser -d $DbName -t -A `
        -c "SELECT count(*) FROM $TargetTable WHERE route_id_raw = 'UNKNOWN';"
    Write-Host "[VERIFY] Rows with 'UNKNOWN' route_id: $UnknownCount"

} finally {
    if (Test-Path $TempCsv) { Remove-Item -Path $TempCsv -Force }
}

Write-Host "[5/5] All steps completed for $Region."
