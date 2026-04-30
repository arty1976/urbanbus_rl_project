param(
    [string]$BaseUrl = "https://apis.data.go.kr/6270000/dbmsapi02/getBasic02",
    [string]$OutputDir = ".\artifacts\daegu_bis_api_audit\getbasic02_master_snapshot_audit",
    [string[]]$ExtraParam = @(),
    [switch]$SelfTest,
    [int]$TimeoutSec = 30
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Step 99-B /getBasic02 master snapshot audit"
Write-Host "[INFO] BaseUrl   : $BaseUrl"
Write-Host "[INFO] OutputDir : $OutputDir"
Write-Host "[INFO] ExtraParam: $($ExtraParam -join ',')"

$script = ".\05_training\adapters\inspect_getbasic02_master_snapshot.py"
if (-not (Test-Path $script)) {
    throw "[STOP] audit script not found: $script"
}

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

$argsList = @(
    $script,
    "--base-url", $BaseUrl,
    "--output-dir", $OutputDir,
    "--timeout", "$TimeoutSec"
)

foreach ($p in $ExtraParam) {
    $argsList += @("--extra-param", $p)
}

if ($SelfTest) {
    $argsList += "--self-test"
}

if ($env:URBANBUS_DB_DSN) {
    $argsList += @("--db-dsn", $env:URBANBUS_DB_DSN)
}

if ($env:DAEGU_BIS_SERVICE_KEY -or $env:DATAGO_SERVICE_KEY) {
    Write-Host "[INFO] ServiceKey: <REDACTED>"
} elseif (-not $SelfTest) {
    Write-Host "[WARN] No API key found. Script will use embedded self-test sample."
}

& $py @argsList

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 99-B /getBasic02 master snapshot audit failed"
}

Write-Host "[OK] Step 99-B /getBasic02 master snapshot audit completed"
Write-Host "[OK] Report JSON: $OutputDir\getbasic02_master_snapshot_audit_report.json"
Write-Host "[OK] Report MD  : $OutputDir\getbasic02_master_snapshot_audit_report.md"
