# =============================================================================
# run_graph_edge_master.ps1
# Purpose : 01 → 02 → 03 SQL 을 순차 실행하고 각 단계 로그를 타임스탬프 파일로 저장
# Usage   : .\run_graph_edge_master.ps1
# Prereq  : PostgreSQL 클라이언트(psql.exe) 설치
#           → psql 이 PATH 에 있거나, 아래 $PSQL_CANDIDATES 중 하나에 존재하면 자동 탐지
#           → 수동 지정: $env:PSQL_EXE = "C:\Program Files\PostgreSQL\16\bin\psql.exe"
# Author  : urbanbus_rl_project
# Date    : 2026-04-19
# =============================================================================

# psql 의 RAISE NOTICE / \echo 는 stderr 로 섞여 나와 PowerShell (5.1 및 7.x 공통)
# 에서 "Stop" 정책이면 NativeCommandError 를 일으켜 스크립트를 중단시킴.
# 따라서 전체 스크립트는 "Continue" 로 두고, 실패 판정은 오직 $LASTEXITCODE 로 한다.
$ErrorActionPreference = "Continue"

# PS 7.3+ 추가 가드 (변수 없으면 무시)
try {
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $PSNativeCommandUseErrorActionPreference = $false
    }
} catch { }

# ---- 사용자 환경에 맞춰 수정 ------------------------------------------------
$PGHOST     = $env:PGHOST     ; if (-not $PGHOST)     { $PGHOST     = "localhost" }
$PGPORT     = $env:PGPORT     ; if (-not $PGPORT)     { $PGPORT     = "5432" }
$PGUSER     = $env:PGUSER     ; if (-not $PGUSER)     { $PGUSER     = "postgres" }
$PGDATABASE = $env:PGDATABASE ; if (-not $PGDATABASE) { $PGDATABASE = "urbanbus" }
# PGPASSWORD 는 환경변수로 설정하거나 %APPDATA%\postgresql\pgpass.conf 사용 권장
# ----------------------------------------------------------------------------

# ---- psql.exe 탐지 ---------------------------------------------------------
function Find-Psql {
    # 1) 명시적 환경변수 우선
    if ($env:PSQL_EXE -and (Test-Path $env:PSQL_EXE)) {
        return $env:PSQL_EXE
    }

    # 2) $env:PGBIN 으로 지정된 경우
    if ($env:PGBIN) {
        $candidate = Join-Path $env:PGBIN "psql.exe"
        if (Test-Path $candidate) { return $candidate }
    }

    # 3) PATH 에 등록된 경우
    $onPath = Get-Command psql.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    # 4) 기본 설치 경로 탐색 (버전 17 → 13 역순)
    $baseDirs = @(
        "$env:ProgramFiles\PostgreSQL",
        "${env:ProgramFiles(x86)}\PostgreSQL"
    )
    foreach ($base in $baseDirs) {
        if (-not (Test-Path $base)) { continue }
        $versions = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending
        foreach ($v in $versions) {
            $candidate = Join-Path $v.FullName "bin\psql.exe"
            if (Test-Path $candidate) { return $candidate }
        }
    }

    return $null
}

$PSQL = Find-Psql
if (-not $PSQL) {
    Write-Host "[FAIL] psql.exe 를 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "      해결 방법 (택 1):" -ForegroundColor Yellow
    Write-Host "       (A) PostgreSQL 설치 시 'Command Line Tools' 옵션 체크 후 PATH 추가" -ForegroundColor Gray
    Write-Host "       (B) 본 세션에서만:" -ForegroundColor Gray
    Write-Host '           $env:PSQL_EXE = "C:\Program Files\PostgreSQL\<버전>\bin\psql.exe"' -ForegroundColor Gray
    Write-Host "           (그 후 .\run_graph_edge_master.ps1 재실행)" -ForegroundColor Gray
    Write-Host "       (C) 영구 PATH 추가 (관리자 PowerShell):" -ForegroundColor Gray
    Write-Host '           [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\PostgreSQL\<버전>\bin", "Machine")' -ForegroundColor Gray
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir    = Join-Path $ScriptDir "logs_$Timestamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host " graph_edge_master pipeline" -ForegroundColor Cyan
Write-Host "   psql     = $PSQL" -ForegroundColor Gray
Write-Host "   host     = $PGHOST`:$PGPORT" -ForegroundColor Gray
Write-Host "   db       = $PGDATABASE" -ForegroundColor Gray
Write-Host "   user     = $PGUSER" -ForegroundColor Gray
Write-Host "   log dir  = $LogDir" -ForegroundColor Gray
Write-Host "=============================================================" -ForegroundColor Cyan

# psql 버전 출력 (연결 전 검증)
& $PSQL --version

$Steps = @(
    @{ Name = "01_alter";      File = "01_alter_graph_edge_master.sql" },
    @{ Name = "02_load";       File = "02_load_graph_edge_master_from_route_link_sequence.sql" },
    @{ Name = "03_readiness";  File = "03_graph_edge_master_readiness.sql" }
)

$overallStart = Get-Date

foreach ($step in $Steps) {
    $sqlPath = Join-Path $ScriptDir $step.File
    $logPath = Join-Path $LogDir ("{0}.log" -f $step.Name)

    if (-not (Test-Path $sqlPath)) {
        Write-Host "[FAIL] SQL 파일이 없습니다: $sqlPath" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "[RUN ] $($step.Name) -> $($step.File)" -ForegroundColor Yellow
    $stepStart = Get-Date

    & $PSQL `
        -h $PGHOST `
        -p $PGPORT `
        -U $PGUSER `
        -d $PGDATABASE `
        -v ON_ERROR_STOP=1 `
        -f $sqlPath `
        *>&1 | Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $($step.Name) 실패. 로그: $logPath" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    $stepDur = (Get-Date) - $stepStart
    Write-Host ("[DONE] {0}  ({1:F1}s)  log: {2}" -f $step.Name, $stepDur.TotalSeconds, $logPath) -ForegroundColor Green
}

$overallDur = (Get-Date) - $overallStart
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host (" All steps OK ({0:F1}s total)" -f $overallDur.TotalSeconds) -ForegroundColor Green
Write-Host " Logs: $LogDir" -ForegroundColor Gray
Write-Host "=============================================================" -ForegroundColor Cyan

# readiness 핵심 결과만 콘솔로 한 번 더 요약 출력
$readinessLog = Join-Path $LogDir "03_readiness.log"
if (Test-Path $readinessLog) {
    Write-Host ""
    Write-Host " >>> Readiness 핵심 결과 (03_readiness.log 요약) <<<" -ForegroundColor Magenta
    Get-Content $readinessLog | Select-String -Pattern "stop_to_stop_total|primary_edges|self_loop_cnt|orphan_src|orphan_dst|null_distance|zero_distance|coverage_pct" | ForEach-Object {
        Write-Host "   $_" -ForegroundColor Gray
    }
}
