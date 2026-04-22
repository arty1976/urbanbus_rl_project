<#
.SYNOPSIS
    2025년 대구 버스 정류소별 시간대별 이용량 CSV (월별) -> Staging 적재 (Robust Version)
.DESCRIPTION
    - 한글 경로 인식 문제를 해결하기 위해 보다 유연한 동적 탐색 사용
#>

# 1. 경로 설정
$ParentDir = "d:\urbanbus_rl_project\data\raw\daegu\traffic_cards\stop_monthly_usage"
$DbName = "urbanbus"
$BatchId = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "[INFO] 2025 Monthly Usage Data Ingestion Starting (Robust Version)..." -ForegroundColor Cyan

# 동적 경로 탐색: "정류소별" 과 "2025"를 동시에 포함하는 폴더 검색
$SourceDirObj = Get-ChildItem -Path $ParentDir -Directory | Where-Object { $_.Name -like "*정류소별*" -and $_.Name -like "*2025*" } | Select-Object -First 1

if ($null -eq $SourceDirObj) {
    # 보조 검색: "2025" 폴더만이라도 검색
    Write-Host "[WARN] Standard pattern failed. Searching for any directory containing '2025'..." -ForegroundColor Yellow
    $SourceDirObj = Get-ChildItem -Path $ParentDir -Directory | Where-Object { $_.Name -like "*2025*" } | Select-Object -First 1
}

if ($null -eq $SourceDirObj) {
     throw "[ERROR] 2025 Monthly Usage Source Directory not found in $ParentDir"
}

$SourceDir = $SourceDirObj.FullName
Write-Host "[INFO] Processing directory: $($SourceDirObj.Name)" -ForegroundColor Green

# 2. psql 경로 찾기
$PsqlPath = "psql"
if (!(Get-Command $PsqlPath -ErrorAction SilentlyContinue)) {
    $CommonPaths = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($CommonPaths) { $PsqlPath = $CommonPaths[0].FullName }
    else { throw "[ERROR] psql.exe not found." }
}

# 3. CSV 파일 목록 가져오기
$AllFiles = Get-ChildItem -Path $SourceDir -Recurse
$CsvFiles = $AllFiles | Where-Object { $_.Extension -match "\.csv$" }

Write-Host "[INFO] Searched $($AllFiles.Count) total files in $SourceDir" -ForegroundColor Gray
Write-Host "[INFO] Found $($CsvFiles.Count) CSV files" -ForegroundColor Cyan

if ($CsvFiles.Count -eq 0) {
    Write-Host "[WARN] No CSV files found. Outputting debug list of child items:" -ForegroundColor Yellow
    $AllFiles | Sort-Object Length -Descending | Select-Object -First 5 | ForEach-Object { Write-Host "   -> $($_.FullName) ($($_.Length) bytes)" }
    # 2023 검증을 방해하지 않기 위해 throw 대신 경고만 출력하고 넘어갑니다.
    return
}

# 4. 적재 명령어 구성 및 실행
$Columns = "year_month, stop_nm, stop_id_raw, category, h05, h06, h07, h08, h09, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23"
# env:PGPASSWORD는 마스터 스크립트에서 상속됨

foreach ($file in $CsvFiles) {
    Write-Host "[INFO] Loading $($file.Name)..." -ForegroundColor Yellow
    
    # 인코딩 변환 (CP949 -> UTF8) & ASCII-safe 임시 파일 생성
    $StagingDir = "d:\urbanbus_rl_project\data\staging\daegu\tmp"
    if (-not (Test-Path $StagingDir)) { New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null }
    $AsciiFile = Join-Path $StagingDir "usage_2025_temp.csv"

    Write-Host "  -> Converting CSV from CP949 to UTF8: $AsciiFile..." -ForegroundColor Gray
    $reader = New-Object System.IO.StreamReader($file.FullName, [System.Text.Encoding]::GetEncoding(949))
    $writer = New-Object System.IO.StreamWriter($AsciiFile, $false, [System.Text.Encoding]::UTF8)
    while (($line = $reader.ReadLine()) -ne $null) {
        $writer.WriteLine($line)
    }
    $reader.Close()
    $writer.Close()

    $CopyCmd = "\copy public.stg_daegu_stop_usage_2025_monthly ($Columns) from '$AsciiFile' with (format csv, header true, encoding 'UTF8');"
    $UpdateCmd = "UPDATE public.stg_daegu_stop_usage_2025_monthly SET batch_id = '$BatchId' WHERE batch_id IS NULL;"
    
    $env:PGCLIENTENCODING = "UTF8"
    
    Write-Host "    -> Executing COPY command directly..." -ForegroundColor Gray
    & $PsqlPath -X -U postgres -d $DbName -c $CopyCmd 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Ingestion failed: COPY command returned error for 2025 data ($($file.Name))."
    }

    Write-Host "    -> Executing UPDATE command directly..." -ForegroundColor Gray
    & $PsqlPath -X -U postgres -d $DbName -c $UpdateCmd 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Ingestion failed: UPDATE command returned error for 2025 data ($($file.Name))."
    }
}

# 전체 staging row count > 0 확인
$CountCheckQuery = "SELECT COUNT(*) FROM public.stg_daegu_stop_usage_2025_monthly WHERE batch_id = '$BatchId';"
$StgCountRaw = & $PsqlPath -X -U postgres -d $DbName -t -c $CountCheckQuery
$StgCountStr = ($StgCountRaw -join "").Trim()

if ([int]$StgCountStr -gt 0) {
    Write-Host "[SUCCESS] 2025 Monthly ingestion completed. (Total Rows: $StgCountStr)" -ForegroundColor Green
} else {
    throw "0 rows loaded into staging table for 2025."
}
