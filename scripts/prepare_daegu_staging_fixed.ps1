$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\urbanbus_rl_project"

$StopsIn  = Join-Path $ProjectRoot "data\raw\daegu\stops\daegu_bus_stops_2025-09-03.csv"
$RoutesIn = Join-Path $ProjectRoot "data\raw\daegu\routes\daegu_bus_routes_2025-12-02.csv"

$StopsOut  = Join-Path $ProjectRoot "data\staging\daegu\daegu_stops_geo_2025-09-03.csv"
$RoutesOut = Join-Path $ProjectRoot "data\staging\daegu\daegu_routes_2025-12-02.csv"
$ShapesOut = Join-Path $ProjectRoot "data\staging\daegu\daegu_shape_registry_2025-09-03.csv"

$ShapesRoot = Join-Path $ProjectRoot "data\raw\daegu\shapes\extracted\2025-09-03"
$ShapesZip  = Join-Path $ProjectRoot "data\raw\daegu\shapes\zip\대구광역시_버스 노선 공간정보_20250903.zip"

New-Item -ItemType Directory -Force -Path (Split-Path $StopsOut), (Split-Path $RoutesOut) | Out-Null

# Stops: source file is UTF-8 with Korean headers.
# Read the file with original headers, then remap by column position to avoid Korean identifier issues in the script.
$stopRows = Get-Content -Path $StopsIn -Encoding UTF8 | ConvertFrom-Csv
$stopOutRows = foreach ($r in $stopRows) {
    $vals = @($r.PSObject.Properties | ForEach-Object { $_.Value })
    [pscustomobject]@{
        stop_name_ko     = $vals[0]
        stop_name_en     = $vals[1]
        sido             = $vals[2]
        gugun            = $vals[3]
        dong             = $vals[4]
        lon              = $vals[5]
        lat              = $vals[6]
        pass_route_count = $vals[7]
        pass_routes      = $vals[8]
    }
}
$stopOutRows | Export-Csv -Path $StopsOut -NoTypeInformation -Encoding UTF8

# Routes: source file is CP949/EUC-KR on Windows. Use Default code page.
$routeRows = Get-Content -Path $RoutesIn -Encoding Default | ConvertFrom-Csv
$routeOutRows = foreach ($r in $routeRows) {
    $vals = @($r.PSObject.Properties | ForEach-Object { $_.Value })
    [pscustomobject]@{
        route_type       = $vals[0]
        route_id         = $vals[1]
        route_no         = $vals[2]
        route_dir        = $vals[3]
        origin_stop_id   = $vals[4]
        origin_stop_name = $vals[5]
        dest_stop_id     = $vals[6]
        dest_stop_name   = $vals[7]
    }
}
$routeOutRows | Export-Csv -Path $RoutesOut -NoTypeInformation -Encoding UTF8

# Shape registry
$shapeRows = @(
    [pscustomobject]@{
        layer_name      = "bs"
        file_group_date = "2025-09-03"
        shp_path        = (Join-Path $ShapesRoot "bs_20250903.shp")
        dbf_path        = (Join-Path $ShapesRoot "bs_20250903.dbf")
        prj_path        = (Join-Path $ShapesRoot "bs_20250903.prj")
        shx_path        = (Join-Path $ShapesRoot "bs_20250903.shx")
        source_zip      = $ShapesZip
    }
    [pscustomobject]@{
        layer_name      = "link"
        file_group_date = "2025-09-03"
        shp_path        = (Join-Path $ShapesRoot "link_20250903.shp")
        dbf_path        = (Join-Path $ShapesRoot "link_20250903.dbf")
        prj_path        = (Join-Path $ShapesRoot "link_20250903.prj")
        shx_path        = (Join-Path $ShapesRoot "link_20250903.shx")
        source_zip      = $ShapesZip
    }
    [pscustomobject]@{
        layer_name      = "node"
        file_group_date = "2025-09-03"
        shp_path        = (Join-Path $ShapesRoot "node_20250903.shp")
        dbf_path        = (Join-Path $ShapesRoot "node_20250903.dbf")
        prj_path        = (Join-Path $ShapesRoot "node_20250903.prj")
        shx_path        = (Join-Path $ShapesRoot "node_20250903.shx")
        source_zip      = $ShapesZip
    }
)
$shapeRows | Export-Csv -Path $ShapesOut -NoTypeInformation -Encoding UTF8

Write-Host "Created:"
Write-Host " - $StopsOut"
Write-Host " - $RoutesOut"
Write-Host " - $ShapesOut"
