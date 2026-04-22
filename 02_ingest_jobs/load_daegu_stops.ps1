<#
.SYNOPSIS
    대구 정류소 위치정보 CSV 파일을 읽어 저장소 구조(raw_csv)에 복사 및 적재합니다.

.DESCRIPTION
    지정된 원본 CSV 파일을 읽은 뒤, 현재 스크립트가 위치한 경로의 하위 폴더(data/raw_csv)가 없으면 생성하고, 
    해당 위치로 UTF-8 포맷으로 정리된 복사본을 타임스탬프 파일명과 함께 저장하는 골격입니다.

.PARAMETER SourceCsvPath
    읽어들일 원본 정류소 CSV 파일의 경로입니다.
#>
param (
    [Parameter(Mandatory=$true, HelpMessage="원본 CSV 파일 경로를 입력하세요.")]
    [string]$SourceCsvPath
)

# 오류 발생 시 즉시 실행 중단
$ErrorActionPreference = "Stop"

# 현재 위치 및 데이터 적재 폴더 정의
$ScriptDir = $PSScriptRoot
if ([string]::IsNullOrEmpty($ScriptDir)) { $ScriptDir = Get-Location }
$DestDir = Join-Path -Path $ScriptDir -ChildPath "data\raw_csv"

try {
    Write-Host "[INFO] 대구 정류소 CSV 수집 스크립트 시작: load_daegu_stops.ps1"

    # 대상 폴더 생성 (현재 폴더 안 하위 폴더)
    if (-not (Test-Path -Path $DestDir)) {
        Write-Host "[INFO] 데이터 적재 경로 '$DestDir' 를 생성합니다."
        New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    }

    if (-not (Test-Path -Path $SourceCsvPath)) {
        throw "입력한 원본 파일 경로를 찾을 수 없습니다: $SourceCsvPath"
    }

    # 복사본 파일명 (타임스탬프 기준)
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $DestFile = Join-Path -Path $DestDir -ChildPath "daegu_stops_$Timestamp.csv"

    Write-Host "[INFO] CSV 데이터 로드 및 적재 중..."
    
    # 원본 파일 인코딩(UTF-8)을 유지하며 새 경로로 저장합니다.
    # 추가적인 데이터 정제나 필터링 로직은 하위 파이프라인에서 별도로 처리함을 권장합니다.
    Get-Content -Path $SourceCsvPath -Encoding UTF8 | Set-Content -Path $DestFile -Encoding UTF8
    
    Write-Host "[INFO] 정류소 데이터 적재 완료. 저장 경로: $DestFile"
}
catch {
    Write-Host "[ERROR] 정류소 데이터 처리 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
