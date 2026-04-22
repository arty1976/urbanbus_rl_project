<#
.SYNOPSIS
    대구버스정보시스템 Open API를 호출하고 결과를 로컬 JSON 파일로 저장합니다.

.DESCRIPTION
    API 키를 인자로 받거나 환경변수에서 읽어 대상 정보를 조회한 뒤,
    응답 원본을 workspace 내 data/raw/daegu/api_snapshots/ 하위 폴더에
    날짜/시간 파일명으로 저장합니다.
    재시도 횟수와 대기 시간을 파라미터로 지정할 수 있습니다.

.PARAMETER ApiKey
    Open API에 사용할 인증 키입니다.
    지정하지 않으면 DAEGU_BUS_API_KEY 환경변수를 사용합니다.

.PARAMETER MaxRetries
    API 호출 실패 시 최대 재시도 횟수 (기본값: 2)

.PARAMETER RetryDelaySeconds
    재시도 대기 시간(초) (기본값: 3)
#>
param (
    [string]$ApiKey            = "",
    [int]   $MaxRetries        = 2,
    [int]   $RetryDelaySeconds = 3
)

# 예기치 않은 오류 발생 시 스크립트를 즉지 중단
$ErrorActionPreference = "Stop"

# 경로 설정: 이 스크립트는 <workspace>/02_ingest_jobs/ 에 위치
# $PSScriptRoot 가 null 인 실행 컨텍스트에서도 안전하게 경로를 확보한다
$ScriptDir = if (-not [string]::IsNullOrEmpty($PSScriptRoot)) {
    $PSScriptRoot                                        # 일반 실행 시 항상 채워짐
} elseif (-not [string]::IsNullOrEmpty($MyInvocation.MyCommand.Path)) {
    Split-Path -Parent $MyInvocation.MyCommand.Path      # dot-source 등 대체 경로
} else {
    (Get-Location).Path                                  # 최후 fallback (스크립트 루트와 다를 수 있음)
}
$WorkspaceDir = Split-Path -Parent $ScriptDir
$DestDir      = Join-Path -Path $WorkspaceDir -ChildPath "data\raw\daegu\api_snapshots"

try {
    Write-Host "[INFO] 대구버스 Open API 데이터 수집 시작: fetch_bus_api.ps1"

    # API 키 검증 (가장 먼저 파라미터 확인 후, 없으면 환경변수)
    if ([string]::IsNullOrWhiteSpace($ApiKey)) {
        $ApiKey = [System.Environment]::GetEnvironmentVariable("DAEGU_BUS_API_KEY")
        if ([string]::IsNullOrWhiteSpace($ApiKey)) {
            throw "API 키가 제공되지 않았습니다. -ApiKey 옵션으로 넘기거나 시스템의 DAEGU_BUS_API_KEY 환경변수를 설정하세요."
        }
    }

    # JSON 데이터 적재 경로 생성
    if (-not (Test-Path -Path $DestDir)) {
        Write-Host "[INFO] 데이터 적재 경로 '$DestDir' 를 생성합니다."
        New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    }

    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $DestFile = Join-Path -Path $DestDir -ChildPath "daegu_bus_api_$Timestamp.json"

    # 베이스 엔드포인트 (공공데이터포털 대구버스 API)
    $ApiBaseUrl  = "https://apis.data.go.kr/6270000/dbmsapi02"

    # 세부 기능 경로: -ApiPath 인자로 외부 주입 (예: /getBusRouteList)
    # 지정하지 않으면 베이스 URL 단독 호출
    $ApiEndpoint = if ([string]::IsNullOrWhiteSpace($ApiPath)) {
        $ApiBaseUrl
    } else {
        $ApiBaseUrl.TrimEnd('/') + '/' + $ApiPath.TrimStart('/')
    }

    # 필수 쿼리 파라미터 + 외부 추가 파라미터 조합
    # resultType=json 은 항상 포함; ExtraParams 는 "&key=value" 형태로 추가
    $QueryParams = "?serviceKey=$ApiKey&resultType=json$ExtraParams"
    $RequestUrl  = $ApiEndpoint + $QueryParams

    Write-Host "[INFO] API 요청 엔드포인트: $ApiEndpoint"
    Write-Host "[INFO] 추가 파라미터: $(if ($ExtraParams) { $ExtraParams } else { '(없음)' })"

    # 재시도 루프 ($MaxRetries, $RetryDelaySeconds 파라미터 실제 사용)
    $Attempt  = 0
    $Response = $null
    while ($Attempt -le $MaxRetries) {
        try {
            $Response = Invoke-RestMethod -Uri $RequestUrl -Method Get -ErrorAction Stop
            break   # 성공 시 루프 탈출
        }
        catch {
            $Attempt++
            if ($Attempt -gt $MaxRetries) { throw }   # 재시도 초과 → 상위 catch 전파
            Write-Host "[WARN] API 호출 실패 (시도 $Attempt / $MaxRetries). ${RetryDelaySeconds}초 후 재시도..."
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }

    # 응답 객체를 원본 보존 목적으로 JSON 포맷 변환
    Write-Host "[INFO] 데이터 변환 및 로컬 적재 진행 중..."
    
    # UTF-8 파일 저장을 위한 IO.File 처리
    $ResponseJsonString = ConvertTo-Json -InputObject $Response -Depth 10
    [System.IO.File]::WriteAllText($DestFile, $ResponseJsonString, [System.Text.Encoding]::UTF8)

    Write-Host "[INFO] API 적재가 완료되었습니다. 저장 경로: $DestFile"
}
catch {
    Write-Host "[ERROR] API 호출 및 데이터 저장 실패: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
