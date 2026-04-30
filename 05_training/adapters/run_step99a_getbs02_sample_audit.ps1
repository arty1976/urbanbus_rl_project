param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [string]$BaseUrl = "https://apis.data.go.kr/6270000/dbmsapi02/getBs02",
    [string]$RouteParam = "routeId",
    [string[]]$RouteId = @(),
    [int]$MaxRoutes = 1,
    [string]$ServiceKey = $env:DAEGU_BIS_SERVICE_KEY,
    [string]$DbDsn = $env:URBANBUS_DB_DSN,
    [string[]]$ExtraParam = @(),
    [string]$OutputDir = ".\artifacts\daegu_bis_api_audit"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] ProjectRoot not found: $ProjectRoot"
}
Set-Location $ProjectRoot

if (-not $ServiceKey) {
    throw "[STOP] DAEGU_BIS_SERVICE_KEY is empty. Use the public data portal general authentication key (Decoding)."
}

if ($MaxRoutes -gt 3) {
    throw "[STOP] MaxRoutes must be <= 3 for Step 99-A feasibility audit."
}
if ($RouteId.Count -lt 1) {
    throw "[STOP] At least one -RouteId is required. Use a sample route_id from the route catalog CSV."
}
if ($RouteId.Count -gt 3) {
    throw "[STOP] RouteId count must be <= 3."
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$script = ".\05_training\adapters\inspect_getbs02_route_stop_sequence.py"
if (-not (Test-Path $script)) {
    throw "[STOP] audit script not found: $script"
}

$argsList = @(
    $script,
    "--base-url", $BaseUrl,
    "--route-param", $RouteParam,
    "--max-routes", "$MaxRoutes",
    "--service-key", $ServiceKey,
    "--output-dir", $OutputDir
)

foreach ($rid in $RouteId) {
    $argsList += @("--route-id", $rid)
}
foreach ($p in $ExtraParam) {
    $argsList += @("--param", $p)
}
if ($DbDsn) {
    $argsList += @("--db-dsn", $DbDsn)
}

Write-Host "[INFO] Step 99-A /getBs02 sample feasibility audit"
Write-Host "[INFO] BaseUrl    : $BaseUrl"
Write-Host "[INFO] RouteParam : $RouteParam"
Write-Host "[INFO] RouteId    : $($RouteId -join ', ')"
Write-Host "[INFO] MaxRoutes  : $MaxRoutes"
Write-Host "[INFO] OutputDir  : $OutputDir"
Write-Host "[INFO] ServiceKey : <REDACTED>"

& $py @argsList
if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99-A /getBs02 sample audit failed"
}

Write-Host "[OK] Step 99-A /getBs02 sample audit completed"
Write-Host "[OK] Report JSON: $OutputDir\getbs02_route_stop_sequence_audit_report.json"
Write-Host "[OK] Report MD  : $OutputDir\getbs02_route_stop_sequence_audit_report.md"
