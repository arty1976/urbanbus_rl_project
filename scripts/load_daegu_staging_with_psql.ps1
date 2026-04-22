param(
    [string]$ProjectRoot = "D:\urbanbus_rl_project",
    [string]$DbName = "urbanbus",
    [string]$DbUser = "postgres",
    [string]$DbHost = "localhost",
    [int]$DbPort = 5432
)

$ErrorActionPreference = "Stop"

$SqlFile        = Join-Path $PSScriptRoot "create_daegu_staging_tables.sql"
$StopsStaging   = Join-Path $ProjectRoot "data\staging\daegu\daegu_stops_geo_2025-09-03.csv"
$RoutesStaging  = Join-Path $ProjectRoot "data\staging\daegu\daegu_routes_2025-12-02.csv"
$ShapeRegistry  = Join-Path $ProjectRoot "data\staging\daegu\daegu_shape_registry_2025-09-03.csv"

if (-not (Get-Command psql -ErrorAction SilentlyContinue)) {
    throw "psql command not found. Install PostgreSQL client tools or add psql to PATH."
}

Write-Host "[1/2] Creating staging tables ..."
& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -f $SqlFile
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create staging tables."
}

Write-Host "[2/2] Loading staging CSV files with \\copy ..."
$copySql = @"
\\copy stg_daegu_stops_geo(stop_name_ko, stop_name_en, sido, gugun, dong, lon, lat, pass_route_count, pass_routes, source_file) FROM '$StopsStaging' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
\\copy stg_daegu_routes(route_type, route_id, route_no, route_dir, origin_stop_id, origin_stop_name, dest_stop_id, dest_stop_name, source_file) FROM '$RoutesStaging' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
\\copy stg_daegu_shape_registry(layer_name, file_group_date, shp_path, dbf_path, prj_path, shx_path, source_zip) FROM '$ShapeRegistry' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
"@

$tempSql = Join-Path $env:TEMP "load_daegu_staging_$(Get-Date -Format yyyyMMdd_HHmmss).sql"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($tempSql, $copySql, $utf8NoBom)

& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -f $tempSql
if ($LASTEXITCODE -ne 0) {
    throw "Failed to load staging CSV files."
}

Write-Host "Done."
