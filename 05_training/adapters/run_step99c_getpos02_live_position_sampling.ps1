param(
    [string]$BaseUrl = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02",
    [string]$RouteParam = "routeId",
    [string[]]$RouteId = @(),
    [int]$MaxRoutes = 3,
    [string]$OutputDir = ".\artifacts\daegu_bis_api_audit\getpos02_live_position_sampling",
    [string[]]$ExtraParam = @(),
    [string]$ServiceKey = ""
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Step 99-C /getPos02 live position sampling audit"
Write-Host "[INFO] BaseUrl    : $BaseUrl"
Write-Host "[INFO] RouteParam : $RouteParam"
Write-Host "[INFO] RouteId    : $($RouteId -join ',')"
Write-Host "[INFO] MaxRoutes  : $MaxRoutes"
Write-Host "[INFO] OutputDir  : $OutputDir"
Write-Host "[INFO] ExtraParam : $($ExtraParam -join ',')"

if (-not $ServiceKey -and $env:DAEGU_BIS_SERVICE_KEY) {
    $ServiceKey = $env:DAEGU_BIS_SERVICE_KEY
}
if (-not $ServiceKey -and $env:DATAGO_SERVICE_KEY) {
    $ServiceKey = $env:DATAGO_SERVICE_KEY
}

if ($ServiceKey) {
    Write-Host "[INFO] ServiceKey : <REDACTED>"
} else {
    Write-Host "[WARN] ServiceKey : <MISSING> -- schema/self-test mode only"
}

$script = ".\05_training\adapters\inspect_getpos02_live_position_sampling.py"
if (-not (Test-Path $script)) {
    throw "[STOP] audit script not found: $script"
}

$argsList = @(
    $script,
    "--base-url", $BaseUrl,
    "--route-param", $RouteParam,
    "--max-routes", "$MaxRoutes",
    "--output-dir", $OutputDir
)

foreach ($rid in $RouteId) {
    $argsList += @("--route-id", $rid)
}
foreach ($ep in $ExtraParam) {
    $argsList += @("--extra-param", $ep)
}
if ($ServiceKey) {
    $argsList += @("--service-key", $ServiceKey)
}

python @argsList

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99-C /getPos02 live position sampling audit failed"
}

Write-Host "[OK] Step 99-C /getPos02 live position sampling audit completed"
Write-Host "[OK] Report JSON: $OutputDir\getpos02_live_position_sampling_report.json"
Write-Host "[OK] Report MD  : $OutputDir\getpos02_live_position_sampling_report.md"
