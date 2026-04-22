<#
.SYNOPSIS
    2023년 대구 버스 정류소별 시간대별 이용량 CSV -> Staging 적재 (Clean Version)
.DESCRIPTION
    - 한글 경로 인식 문제를 해결하기 위해 동적 경로 탐색 사용
    - 구문 오류 방지를 위해 전체 스크립트 재작성
#>

# 1. 경로 설정
$ParentDir = "d:\urbanbus_rl_project\data\raw\daegu\traffic_cards\stop_hourly_usage_2023"
$DbName = "urbanbus"
$BatchId = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "[INFO] 2023 Usage Data Ingestion Starting (Clean Version)..." -ForegroundColor Cyan

# 동적 경로 탐색 (한글 폴더명 매칭 보완)
# 명시적인 한글 문자열 대신 인덱스나 와일드카드 조합 사용 시도
$SubDir = Get-ChildItem -Path $ParentDir -Directory | Where-Object { $_.Name -like "*정류소별*" -and ($_.Name -like "*2023*" -or $_.Name -like "*(1)*") } | Select-Object -First 1

if ($null -eq $SubDir) {
    Write-Host "[WARN] Specific subfolder not found by name pattern. Searching for large CSV directly..." -ForegroundColor Yellow
    $SourceFileObj = Get-ChildItem -Path $ParentDir -Filter "*.csv" -Recurse | Where-Object { $_.Length -gt 200MB } | Select-Object -First 1
} else {
    Write-Host "[INFO] Found Subdirectory: $($SubDir.Name)" -ForegroundColor Green
    $SourceFileObj = Get-ChildItem -Path $SubDir.FullName -Filter "*.csv" | Select-Object -First 1
}

if ($null -eq $SourceFileObj) {
    throw "[ERROR] 2023 Usage CSV file not found in $ParentDir"
}

$SourceFile = $SourceFileObj.FullName
Write-Host "[INFO] Original Source File Path: $SourceFile" -ForegroundColor Green

# 1-a. 인코딩 변환 (CP949 -> UTF8)
$StagingDir = "d:\urbanbus_rl_project\data\staging\daegu\tmp"
if (-not (Test-Path $StagingDir)) { New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null }
$AsciiFile = Join-Path $StagingDir "usage_2023.csv"

Write-Host "[INFO] Copying UTF-8 CSV without encoding conversion to ASCII path: $AsciiFile..." -ForegroundColor Yellow
Copy-Item -Path $SourceFile -Destination $AsciiFile -Force

# 2. psql 경로 찾기
$PsqlPath = "psql"
if (!(Get-Command $PsqlPath -ErrorAction SilentlyContinue)) {
    $CommonPaths = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($CommonPaths) { $PsqlPath = $CommonPaths[0].FullName }
    else { throw "[ERROR] psql.exe not found." }
}

# 3. CSV 로드 명령어 구성
$AsciiCsvPath = "d:/urbanbus_rl_project/data/staging/daegu/tmp/usage_2023.csv"

$CurrentCountRaw = & $PsqlPath -X -U postgres -d $DbName -t -c "SELECT COUNT(*) FROM public.stg_daegu_stop_usage_2023;"
$CurrentCountStr = ($CurrentCountRaw -join "").Trim()
$CurrentCount = 0
if (-not [string]::IsNullOrWhiteSpace($CurrentCountStr)) { $CurrentCount = [int]$CurrentCountStr }

$TempSqlFile = Join-Path $env:TEMP "load_2023_merged.sql"

$SqlContent = ""
if ($CurrentCount -lt 2100000) {
    Write-Host "[INFO] Current row count ($CurrentCount) is low. Running COPY command." -ForegroundColor Yellow
    $SqlContent += "\copy public.stg_daegu_stop_usage_2023 (service_date_raw, stop_name_raw, stop_id_raw, mobile_id_raw, admin_area_raw, usage_type_raw, h05_raw, h06_raw, h07_raw, h08_raw, h09_raw, h10_raw, h11_raw, h12_raw, h13_raw, h14_raw, h15_raw, h16_raw, h17_raw, h18_raw, h19_raw, h20_raw, h21_raw, h22_raw, h23_raw, row_sum_raw) FROM '$AsciiCsvPath' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');`n"
} else {
    Write-Host "[INFO] Rows already loaded ($CurrentCount). Skipping COPY to preserve existing data." -ForegroundColor Green
}

$SqlContent += "UPDATE public.stg_daegu_stop_usage_2023 SET batch_id = '$BatchId', source_file = 'usage_2023.csv';"

[System.IO.File]::WriteAllText($TempSqlFile, $SqlContent, [System.Text.Encoding]::UTF8)

$env:PGCLIENTENCODING = "UTF8"

Write-Host "  -> Executing DML commands from temporary SQL file..." -ForegroundColor Gray
& $PsqlPath -X -U postgres -d $DbName -v ON_ERROR_STOP=1 -f $TempSqlFile 2>&1 | Tee-Object -FilePath $LogFile -Append
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Detected database error (e.g., deadlock). This usually means an old hanging psql process is using the table. Trying to proceed anyway..." -ForegroundColor Yellow
}

if (Test-Path $TempSqlFile) { Remove-Item $TempSqlFile -ErrorAction SilentlyContinue }

Write-Host "[INFO] Loading data into staging..." -ForegroundColor Green
# env:PGPASSWORD는 마스터 스크립트에서 상속됨

try {
    # 1-d. staging row count > 0 확인
    $CountCheckQuery = "SELECT COUNT(*) FROM public.stg_daegu_stop_usage_2023 WHERE batch_id = '$BatchId';"
    $StgCountRaw = & $PsqlPath -X -U postgres -d $DbName -t -c $CountCheckQuery
    $StgCountStr = ($StgCountRaw -join "").Trim()
    
    if ([int]$StgCountStr -gt 0) {
        Write-Host "[SUCCESS] Data loaded successfully with BatchId: $BatchId (Rows: $StgCountStr)" -ForegroundColor Green
    } else {
        throw "0 rows loaded into staging table."
    }
}
catch {
    Write-Host "[ERROR] Ingestion failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    if (Test-Path $TempSqlFile) { Remove-Item $TempSqlFile }
}
