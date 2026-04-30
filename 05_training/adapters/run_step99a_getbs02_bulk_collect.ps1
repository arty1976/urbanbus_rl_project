param(
    [string]$DbDsn = $env:URBANBUS_DB_DSN,
    [string]$ServiceKey = $env:DAEGU_BIS_SERVICE_KEY,
    [string]$BaseUrl = "https://apis.data.go.kr/6270000/dbmsapi02/getBs02",
    [string]$RouteParam = "routeId",
    [string]$OutRoot = ".\artifacts\daegu_bis_api_audit\getbs02_bulk_collect",
    [int]$MaxRoutes = 0,
    [double]$SleepSec = 0.25,
    [string]$RunId = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$script = ".\05_training\adapters\collect_getbs02_route_stop_sequence_bulk.py"
if (-not (Test-Path $script)) {
    throw "[STOP] bulk collector script not found: $script"
}
if (-not $ServiceKey) {
    throw "[STOP] service key missing. Set DAEGU_BIS_SERVICE_KEY."
}
if (-not $DbDsn) {
    throw "[STOP] DB DSN missing. Set URBANBUS_DB_DSN."
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$argsList = @(
    $script,
    "--db-dsn", $DbDsn,
    "--service-key", $ServiceKey,
    "--base-url", $BaseUrl,
    "--route-param", $RouteParam,
    "--out-root", $OutRoot,
    "--sleep-sec", ([string]$SleepSec),
    "--extra-param", "resultType=json"
)

if ($MaxRoutes -gt 0) {
    $argsList += @("--max-routes", ([string]$MaxRoutes))
}
if ($RunId) {
    $argsList += @("--run-id", $RunId)
}
if ($Resume) {
    $argsList += "--resume"
}

Write-Host "[INFO] Step 99-A full /getBs02 bulk collection"
Write-Host "[INFO] BaseUrl   : $BaseUrl"
Write-Host "[INFO] MaxRoutes : $MaxRoutes (0 means all source routes)"
Write-Host "[INFO] OutRoot   : $OutRoot"
Write-Host "[INFO] SleepSec  : $SleepSec"
Write-Host "[INFO] ServiceKey: <REDACTED>"
Write-Host "[INFO] DB write  : forbidden/not performed"

& $py @argsList
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99-A full /getBs02 bulk collection failed"
}

Write-Host "[OK] Step 99-A full /getBs02 bulk collection completed"
