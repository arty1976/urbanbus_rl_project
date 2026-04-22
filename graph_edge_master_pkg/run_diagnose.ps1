# =============================================================================
# run_diagnose.ps1
# Purpose : 임의의 진단 SQL 파일을 실행하고 결과를 타임스탬프 로그로 저장.
#           ON_ERROR_STOP=0 으로 섹션 하나가 실패해도 전체 진단은 끝까지 진행.
# Usage   : .\run_diagnose.ps1                              # 기본 D_diagnose_stop_id_mapping.sql
#           .\run_diagnose.ps1 -SqlFile D2_probe_mapping_tables.sql
#           .\run_diagnose.ps1 -SqlFile D3_xxx.sql -LogName D3_probe
# =============================================================================

param(
    [string]$SqlFile = "D_diagnose_stop_id_mapping.sql",
    [string]$LogName = ""
)

$ErrorActionPreference = "Continue"
try {
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $PSNativeCommandUseErrorActionPreference = $false
    }
} catch { }

$PGHOST     = $env:PGHOST     ; if (-not $PGHOST)     { $PGHOST     = "localhost" }
$PGPORT     = $env:PGPORT     ; if (-not $PGPORT)     { $PGPORT     = "5432" }
$PGUSER     = $env:PGUSER     ; if (-not $PGUSER)     { $PGUSER     = "postgres" }
$PGDATABASE = $env:PGDATABASE ; if (-not $PGDATABASE) { $PGDATABASE = "urbanbus" }

function Find-Psql {
    if ($env:PSQL_EXE -and (Test-Path $env:PSQL_EXE)) { return $env:PSQL_EXE }
    $onPath = Get-Command psql.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    foreach ($base in @("$env:ProgramFiles\PostgreSQL", "${env:ProgramFiles(x86)}\PostgreSQL")) {
        if (-not (Test-Path $base)) { continue }
        $versions = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending
        foreach ($v in $versions) {
            $c = Join-Path $v.FullName "bin\psql.exe"
            if (Test-Path $c) { return $c }
        }
    }
    return $null
}

$PSQL = Find-Psql
if (-not $PSQL) {
    Write-Host "[FAIL] psql.exe 를 찾을 수 없습니다." -ForegroundColor Red
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir    = Join-Path $ScriptDir "logs_$Timestamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$SqlPath = Join-Path $ScriptDir $SqlFile
if (-not (Test-Path $SqlPath)) {
    Write-Host "[FAIL] SQL 파일이 없습니다: $SqlPath" -ForegroundColor Red
    exit 1
}
if (-not $LogName) {
    $LogName = [System.IO.Path]::GetFileNameWithoutExtension($SqlFile)
}
$LogPath = Join-Path $LogDir ("{0}.log" -f $LogName)

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " stop_id mapping diagnosis" -ForegroundColor Cyan
Write-Host "   psql     = $PSQL" -ForegroundColor Gray
Write-Host "   db       = $PGDATABASE" -ForegroundColor Gray
Write-Host "   sql      = $SqlPath" -ForegroundColor Gray
Write-Host "   log      = $LogPath" -ForegroundColor Gray
Write-Host "=============================================================" -ForegroundColor Cyan

& $PSQL `
    -h $PGHOST `
    -p $PGPORT `
    -U $PGUSER `
    -d $PGDATABASE `
    -v ON_ERROR_STOP=0 `
    -f $SqlPath `
    *>&1 | Tee-Object -FilePath $LogPath

Write-Host ""
Write-Host "[DONE] 진단 완료. 전체 로그: $LogPath" -ForegroundColor Green
Write-Host "       이 파일 내용을 Claude 에 붙여 주세요." -ForegroundColor Gray
