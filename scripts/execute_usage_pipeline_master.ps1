<#
.SYNOPSIS
    Daegu Bus Ingestion Pipeline Master Execution Script
.DESCRIPTION
    12개의 단계를 순차적으로 실행하고 결과를 집계하여 보고서 형식으로 출력합니다.
#>

$DbName = "urbanbus"
$User = "postgres"
$LogFile = "d:\urbanbus_rl_project\usage_pipeline_execution.log"

# psql 경로 확인
$PsqlPath = "psql"
if (!(Get-Command $PsqlPath -ErrorAction SilentlyContinue)) {
    $CommonPaths = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($CommonPaths) { $PsqlPath = $CommonPaths[0].FullName }
}

# 0. 비밀번호 세션 설정 (인증 반복 방지)
$InputPass = Read-Host "PostgreSQL 비밀번호를 입력하세요 (최초 1회)"
$env:PGPASSWORD = $InputPass

function Run-SqlFile($FilePath) {
    Write-Host "[SQL] Executing: $(Split-Path $FilePath -Leaf)..." -ForegroundColor Cyan
    $Output = & $PsqlPath -X -U $User -d $DbName -f $FilePath 2>&1
    $FilteredOutput = $Output | Where-Object { $_ -notmatch 'relation already exists|already exists, skipping|릴레이션.*이미 있습니다' }
    $FilteredOutput | Tee-Object -FilePath $LogFile -Append
}

function Run-Query($Query) {
    & $PsqlPath -X -U $User -d $DbName -t -c $Query
}

Clear-Host
Write-Host "====================================================" -ForegroundColor White
Write-Host "  Daegu Bus Usage Pipeline Master Execution" -ForegroundColor White
Write-Host "====================================================" -ForegroundColor White

# 1. DDL Steps (1-5)
Write-Host "`n[PHASE 1] Setup DDLs (SKIPPED TO PRESERVE DATA)..." -ForegroundColor Yellow
# Run-SqlFile "d:\urbanbus_rl_project\02_ingest_jobs\create_stg_daegu_stop_usage_2023.sql"
# Run-SqlFile "d:\urbanbus_rl_project\02_ingest_jobs\create_stg_daegu_stop_usage_2025_monthly.sql"
# Run-SqlFile "d:\urbanbus_rl_project\02_ingest_jobs\create_err_daegu_stop_usage_mapping_failed.sql"
# Run-SqlFile "d:\urbanbus_rl_project\02_ingest_jobs\create_fact_stop_usage_hourly.sql"
# Run-SqlFile "d:\urbanbus_rl_project\02_ingest_jobs\create_fact_stop_usage_hourly_profile_2025.sql"

# 2. Ingestion Steps (6-7)
Write-Host "`n[PHASE 2] Loading Raw Data..." -ForegroundColor Yellow
& "d:\urbanbus_rl_project\02_ingest_jobs\load_daegu_stop_usage_2023.ps1"
& "d:\urbanbus_rl_project\02_ingest_jobs\load_daegu_stop_usage_2025_monthly.ps1"

# 3. Readiness Check (8)
Write-Host "`n[PHASE 3] Staging Readiness Check..." -ForegroundColor Yellow
Run-SqlFile "d:\urbanbus_rl_project\03_validation_queries\stg_usage_readiness_check.sql"

$Count2023 = (& $PsqlPath -X -U $User -d $DbName -t -c "SELECT COUNT(*) FROM public.stg_daegu_stop_usage_2023;") -join ""
$Count2025 = (& $PsqlPath -X -U $User -d $DbName -t -c "SELECT COUNT(*) FROM public.stg_daegu_stop_usage_2025_monthly;") -join ""

if ([int]$Count2023.Trim() -eq 0 ) {
    Write-Host "`n[ERROR] Pipeline aborted: Staging tables are empty. (2023: $Count2023)" -ForegroundColor Red
    Write-Host "Please fix the ingestion logic before executing downstream tasks." -ForegroundColor Red
    exit 1
}

# 4. Promotion (9-10)
Write-Host "`n[PHASE 4] Promotion to Fact..." -ForegroundColor Yellow
Run-SqlFile "d:\urbanbus_rl_project\02_ingest_jobs\promote_fact_stop_usage_hourly_from_2023_file.sql"
# Run-SqlFile "d:\urbanbus_rl_project\02_ingest_jobs\promote_fact_stop_usage_profile_2025.sql"

# 5. Reconciliation Check
Write-Host "`n[PHASE 5] Validation & Reconciliation..." -ForegroundColor Yellow
Run-SqlFile "d:\urbanbus_rl_project\03_validation_queries\reconcile_usage_sums_2023.sql"

# 5. Reconciliation (11-12)
Write-Host "`n[PHASE 5] Final Reconciliation..." -ForegroundColor Yellow
Run-SqlFile "d:\urbanbus_rl_project\03_validation_queries\reconcile_usage_sums_2023.sql"
Run-SqlFile "d:\urbanbus_rl_project\03_validation_queries\reconcile_usage_sums_2025_monthly.sql"

Write-Host "`n====================================================" -ForegroundColor White
Write-Host "  Pipeline Execution Completed. Please see results above." -ForegroundColor White
Write-Host "====================================================" -ForegroundColor White
