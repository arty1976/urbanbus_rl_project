param(
    [string]$ProjectRoot = "D:\urbanbus_rl_project"
)

$ErrorActionPreference = "Stop"

function Export-Utf8NoBomCsv {
    param(
        [Parameter(Mandatory=$true)] $InputObject,
        [Parameter(Mandatory=$true)] [string]$Path
    )

    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    $csvLines = $InputObject | ConvertTo-Csv -NoTypeInformation
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($Path, $csvLines, $utf8NoBom)
}

$StopsRaw        = Join-Path $ProjectRoot "data\raw\daegu\stops\daegu_bus_stops_2025-09-03.csv"
$RoutesRaw       = Join-Path $ProjectRoot "data\raw\daegu\routes\daegu_bus_routes_2025-12-02.csv"
$ShapesZip       = Join-Path $ProjectRoot "data\raw\daegu\shapes\zip\대구광역시_버스 노선 공간정보_20250903.zip"
$ShapesExtracted = Join-Path $ProjectRoot "data\raw\daegu\shapes\extracted\2025-09-03"
$StagingDir      = Join-Path $ProjectRoot "data\staging\daegu"

$StopsStaging    = Join-Path $StagingDir "daegu_stops_geo_2025-09-03.csv"
$RoutesStaging   = Join-Path $StagingDir "daegu_routes_2025-12-02.csv"
$ShapeRegistry   = Join-Path $StagingDir "daegu_shape_registry_2025-09-03.csv"

New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

if (-not (Test-Path $StopsRaw)) {
    throw "Stops raw file not found: $StopsRaw"
}
if (-not (Test-Path $RoutesRaw)) {
    throw "Routes raw file not found: $RoutesRaw"
}

Write-Host "[1/3] Transforming stops CSV ..."
$stops = Import-Csv -Path $StopsRaw -Encoding UTF8
$stopsOut = $stops | Select-Object `
    @{Name='stop_name_ko';Expression={$_.정류소명}},
    @{Name='stop_name_en';Expression={$_.영문명}},
    @{Name='sido';Expression={$_.시도}},
    @{Name='gugun';Expression={$_.구군}},
    @{Name='dong';Expression={$_.동}},
    @{Name='lon';Expression={$_.경도}},
    @{Name='lat';Expression={$_.위도}},
    @{Name='pass_route_count';Expression={$_.경유노선수}},
    @{Name='pass_routes';Expression={$_.경유노선}},
    @{Name='source_file';Expression={Split-Path $StopsRaw -Leaf}}
Export-Utf8NoBomCsv -InputObject $stopsOut -Path $StopsStaging
Write-Host "  -> Created: $StopsStaging ($($stopsOut.Count) rows)"

Write-Host "[2/3] Transforming routes CSV ..."
$routes = Import-Csv -Path $RoutesRaw -Encoding Default
$routesOut = $routes | Select-Object `
    @{Name='route_type';Expression={$_.노선유형}},
    @{Name='route_id';Expression={$_.노선ID}},
    @{Name='route_no';Expression={$_.노선번호}},
    @{Name='route_dir';Expression={$_.노선방면}},
    @{Name='origin_stop_id';Expression={$_.기점정류소ID}},
    @{Name='origin_stop_name';Expression={$_.기점정류소명}},
    @{Name='dest_stop_id';Expression={$_.종점정류소ID}},
    @{Name='dest_stop_name';Expression={$_.종점정류소명}},
    @{Name='source_file';Expression={Split-Path $RoutesRaw -Leaf}}
Export-Utf8NoBomCsv -InputObject $routesOut -Path $RoutesStaging
Write-Host "  -> Created: $RoutesStaging ($($routesOut.Count) rows)"

Write-Host "[3/3] Building shape registry CSV ..."
$shapeRows = @()
foreach ($layer in @('bs','link','node')) {
    $shapeRows += [pscustomobject]@{
        layer_name     = $layer
        file_group_date= '2025-09-03'
        shp_path       = Join-Path $ShapesExtracted ("{0}_20250903.shp" -f $layer)
        dbf_path       = Join-Path $ShapesExtracted ("{0}_20250903.dbf" -f $layer)
        prj_path       = Join-Path $ShapesExtracted ("{0}_20250903.prj" -f $layer)
        shx_path       = Join-Path $ShapesExtracted ("{0}_20250903.shx" -f $layer)
        source_zip     = $ShapesZip
    }
}
Export-Utf8NoBomCsv -InputObject $shapeRows -Path $ShapeRegistry
Write-Host "  -> Created: $ShapeRegistry ($($shapeRows.Count) rows)"

Write-Host "Done."
