param(
    [string]$DbName = "urbanbus",
    [string]$DbUser = "postgres",
    [string]$DbHost = "localhost",
    [int]$DbPort = 5432
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\urbanbus_rl_project"
$Psql = Get-Command psql -ErrorAction SilentlyContinue
if (-not $Psql) {
    throw "psql command not found. Install PostgreSQL client tools or add psql to PATH."
}

$CreateSql = Join-Path $ProjectRoot "scripts\create_daegu_staging_tables.sql"
$StopsCsv  = Join-Path $ProjectRoot "data\staging\daegu\daegu_stops_geo_2025-09-03.csv"
$RoutesCsv = Join-Path $ProjectRoot "data\staging\daegu\daegu_routes_2025-12-02.csv"
$ShapesCsv = Join-Path $ProjectRoot "data\staging\daegu\daegu_shape_registry_2025-09-03.csv"

Write-Host "[1/2] Creating staging tables ..."
& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 -f $CreateSql
if ($LASTEXITCODE -ne 0) { throw "Failed to create staging tables." }

Write-Host "[2/2] Loading staging CSV files with \copy ..."
& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 `
  -c "\copy stg_daegu_stops_geo(stop_name_ko, stop_name_en, sido, gugun, dong, lon, lat, pass_route_count, pass_routes) FROM '$($StopsCsv -replace '\\','/')' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
if ($LASTEXITCODE -ne 0) { throw "Failed to load stg_daegu_stops_geo." }

& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 `
  -c "\copy stg_daegu_routes(route_type, route_id, route_no, route_dir, origin_stop_id, origin_stop_name, dest_stop_id, dest_stop_name) FROM '$($RoutesCsv -replace '\\','/')' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
if ($LASTEXITCODE -ne 0) { throw "Failed to load stg_daegu_routes." }

& psql -h $DbHost -p $DbPort -U $DbUser -d $DbName -v ON_ERROR_STOP=1 `
  -c "\copy stg_daegu_shape_registry(layer_name, file_group_date, shp_path, dbf_path, prj_path, shx_path, source_zip) FROM '$($ShapesCsv -replace '\\','/')' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')"
if ($LASTEXITCODE -ne 0) { throw "Failed to load stg_daegu_shape_registry." }

Write-Host "Done."
