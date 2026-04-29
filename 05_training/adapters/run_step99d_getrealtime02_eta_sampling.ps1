param(
    [string]$BaseUrl = "https://apis.data.go.kr/6270000/dbmsapi02/getRealtime02",
    [string]$StopParam = "bsId",
    [string[]]$StopId = @(),
    [int]$MaxStops = 10,
    [string]$OutputDir = ".\artifacts\daegu_bis_api_audit\getrealtime02_eta_sampling",
    [string[]]$ExtraParam = @(),
    [string]$ServiceKey = "",
    [int]$Timeout = 30,
    [double]$SleepSec = 0.0,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"

$ScriptPath = ".\05_training\adapters\inspect_getrealtime02_eta_sampling.py"
if (-not (Test-Path $ScriptPath)) {
    throw "[STOP] audit script not found: $ScriptPath"
}

if (-not $ServiceKey) {
    if ($env:DAEGU_BIS_SERVICE_KEY) {
        $ServiceKey = $env:DAEGU_BIS_SERVICE_KEY
    } elseif ($env:DATAGO_SERVICE_KEY) {
        $ServiceKey = $env:DATAGO_SERVICE_KEY
    }
}

$StopIdExpanded = @()
foreach ($sidItem in @($StopId)) {
    foreach ($part in ([string]$sidItem).Split(",")) {
        $clean = $part.Trim().Trim('"').Trim("'")
        if ($clean) {
            $StopIdExpanded += $clean
        }
    }
}
$StopId = @($StopIdExpanded | Select-Object -Unique)

if (-not $SelfTest -and (-not $StopId -or $StopId.Count -eq 0)) {
    throw "[STOP] at least one -StopId is required unless -SelfTest is used."
}

if ($MaxStops -gt 0 -and $StopId.Count -gt $MaxStops) {
    $StopId = @($StopId[0..($MaxStops - 1)])
}

if (-not $SelfTest -and -not $ServiceKey) {
    throw "[STOP] service key not found. Set DAEGU_BIS_SERVICE_KEY or DATAGO_SERVICE_KEY, or pass -ServiceKey."
}

Write-Host "[INFO] Step 99-D /getRealtime02 ETA sampling audit"
Write-Host "[INFO] BaseUrl   : $BaseUrl"
Write-Host "[INFO] StopParam : $StopParam"
Write-Host "[INFO] StopId    : $($StopId -join ',')"
Write-Host "[INFO] MaxStops  : $MaxStops"
Write-Host "[INFO] OutputDir : $OutputDir"
Write-Host "[INFO] ExtraParam: $($ExtraParam -join ',')"
if ($ServiceKey) {
    Write-Host "[INFO] ServiceKey: <REDACTED>"
} else {
    Write-Host "[INFO] ServiceKey: <NONE>"
}

$ArgsList = @(
    $ScriptPath,
    "--base-url", $BaseUrl,
    "--stop-param", $StopParam,
    "--max-stops", "$MaxStops",
    "--output-dir", $OutputDir,
    "--timeout", "$Timeout",
    "--sleep-sec", "$SleepSec"
)

foreach ($sid in @($StopId)) {
    $ArgsList += @("--stop-id", "$sid")
}

foreach ($ep in @($ExtraParam)) {
    if ($ep) {
        $ArgsList += @("--extra-param", "$ep")
    }
}

if ($ServiceKey) {
    $ArgsList += @("--service-key", $ServiceKey)
}

if ($SelfTest) {
    $ArgsList += "--self-test"
}

python @ArgsList

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99-D /getRealtime02 ETA sampling audit failed"
}

Write-Host "[OK] Step 99-D /getRealtime02 ETA sampling audit completed"
Write-Host "[OK] Report JSON: $OutputDir\getrealtime02_eta_sampling_report.json"
Write-Host "[OK] Report MD  : $OutputDir\getrealtime02_eta_sampling_report.md"
