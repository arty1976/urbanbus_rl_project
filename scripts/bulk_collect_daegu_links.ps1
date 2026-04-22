<#
.SYNOPSIS
    대구시 전체 노선의 링크 정보를 일괄 수집하고, 원본 스냅샷 저장 후 로더 스크립트로 DB (Database=데이터베이스) 적재까지 수행합니다.

.DESCRIPTION
    - route_id 목록은 public.stg_daegu_routes 에서 추출합니다.
    - 원본 응답은 data/raw/daegu/api_snapshots/route_links/getLink02/ 아래에 저장합니다.
    - 수집 성공/실패, 적재 성공/실패는 manifest.csv 에 분리 기록합니다.
    - endpointBaseUrl, operationPath, routeIdParamName 등은 config/secrets.json 에서 읽습니다.
    - serviceKey 는 반드시 일반 인증키(Decoding=복호화된 일반 인증키)를 사용해야 합니다.
#>

param (
    [string]$ConfigPath = "configs/secrets.json",
    [string]$DbName = "urbanbus",
    [string]$DbUser = "postgres",
    [string]$PsqlPath = "C:\Program Files\PostgreSQL\18\bin\psql.exe",
    [int]$MaxRetries = 2,
    [int]$RetryDelaySeconds = 2,
    [int]$SleepMilliseconds = 500,
    [int]$MaxRoutes = 0
)

$ErrorActionPreference = "Stop"

function ConvertTo-QueryString {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Params
    )

    return (($Params.GetEnumerator() | ForEach-Object {
                "{0}={1}" -f [System.Uri]::EscapeDataString([string]$_.Key),
                [System.Uri]::EscapeDataString([string]$_.Value)
            }) -join "&")
}

function Add-ManifestRow {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$RouteIdRaw,
        [Parameter(Mandatory = $true)][string]$SnapshotFile,
        [Parameter(Mandatory = $true)][string]$CollectStatus,
        [Parameter(Mandatory = $true)][string]$LoadStatus,
        [string]$HttpStatus = "",
        [string]$ErrorMessage = ""
    )

    $collectedAt = (Get-Date).ToString("s")
    $safeError = $ErrorMessage -replace '"', "'"
    $line = '"{0}","{1}","{2}","{3}","{4}","{5}","{6}"' -f `
        $RouteIdRaw, $collectedAt, $SnapshotFile, $CollectStatus, $LoadStatus, $HttpStatus, $safeError

    Add-Content -Path $ManifestPath -Value $line -Encoding UTF8
}

# ------------------------------------------------------------
# 0. workspace / 경로 계산
# ------------------------------------------------------------
$WorkspaceDir = Split-Path -Parent $PSScriptRoot

if (-not [System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath = Join-Path $WorkspaceDir $ConfigPath
}

$SnapshotDir = Join-Path $WorkspaceDir "data\raw\daegu\api_snapshots\route_links\getLink02"
$ManifestPath = Join-Path $SnapshotDir "manifest.csv"
$LoaderScript = Join-Path $PSScriptRoot "load_route_links_api.ps1"

New-Item -ItemType Directory -Force -Path $SnapshotDir | Out-Null

if (-not (Test-Path $ManifestPath)) {
    'route_id_raw,collected_at,snapshot_file,collect_status,load_status,http_status,error_message' |
    Set-Content -Path $ManifestPath -Encoding UTF8
}

# ------------------------------------------------------------
# 1. 설정 및 서비스키 로드
# ------------------------------------------------------------
if (-not (Test-Path $ConfigPath)) {
    throw "설정 파일이 없습니다: $ConfigPath"
}

$Config = Get-Content $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json

$ServiceKey = $Config.serviceKey
$EndpointBaseUrl = $Config.endpointBaseUrl
$OperationPath = $Config.operationPath
$HttpMethod = if ($Config.httpMethod) { $Config.httpMethod } else { "GET" }
$RouteIdParamName = $Config.routeIdParamName

if ([string]::IsNullOrWhiteSpace($ServiceKey)) {
    throw "serviceKey 가 비어 있습니다. config/secrets.json 을 확인하세요."
}
if ([string]::IsNullOrWhiteSpace($EndpointBaseUrl)) {
    throw "endpointBaseUrl 이 비어 있습니다. config/secrets.json 을 확인하세요."
}
if ([string]::IsNullOrWhiteSpace($OperationPath)) {
    throw "operationPath 가 비어 있습니다. config/secrets.json 을 확인하세요."
}
if ([string]::IsNullOrWhiteSpace($RouteIdParamName)) {
    throw "routeIdParamName 이 비어 있습니다. config/secrets.json 을 확인하세요."
}
$BaseUrl = $EndpointBaseUrl.TrimEnd('/')
$OpPath = $OperationPath.TrimStart('/')
$RequestUrl = "{0}/{1}" -f $BaseUrl, $OpPath

if (-not (Test-Path $PsqlPath)) {
    throw "psql 실행파일이 없습니다: $PsqlPath"
}

# 하위 로더 스크립트에서 psql을 바로 사용할 수 있도록 PATH 임시 추가
$PsqlDir = Split-Path -Parent $PsqlPath
if ($env:PATH -notlike "*$PsqlDir*") {
    $env:PATH = "$PsqlDir;" + $env:PATH
}
if (-not (Test-Path $LoaderScript)) {
    throw "로더 스크립트가 없습니다: $LoaderScript"
}

Write-Host "[1/4] DB에서 노선 목록 추출 중..."

# ------------------------------------------------------------
# 2. DB에서 수집 대상 노선(route_id) 추출
# ------------------------------------------------------------
$RouteListRaw = & $PsqlPath -h localhost -U $DbUser -d $DbName -t -A `
    -c "SELECT DISTINCT route_id::text FROM public.stg_daegu_routes WHERE route_id IS NOT NULL ORDER BY 1;"

if ($LASTEXITCODE -ne 0) {
    throw "route_id 목록 추출에 실패했습니다."
}

$RouteList = @(
    $RouteListRaw |
    ForEach-Object { $_.ToString().Trim() } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

if ($RouteList.Count -eq 0) {
    throw "수집할 노선 ID가 없습니다. public.stg_daegu_routes 를 확인하세요."
}

if ($MaxRoutes -gt 0) {
    $RouteList = $RouteList | Select-Object -First $MaxRoutes
}

Write-Host "[INFO] 총 $($RouteList.Count)개의 노선을 수집합니다."

if ($RouteList.Count -gt 950) {
    Write-Warning "일일 트래픽 한도(1000)에 근접할 수 있습니다. 배치 실행을 권장합니다."
}

# ------------------------------------------------------------
# 3. 루프 시작
# ------------------------------------------------------------
Write-Host "[2/4] 일괄 수집 및 적재 시작..."

$Count = 0

foreach ($Rid in $RouteList) {
    $Count++
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $FileName = "daegu_link_{0}_{1}.json" -f $Rid, $Timestamp
    $FilePath = Join-Path $SnapshotDir $FileName

    Write-Host ("[{0}/{1}] 노선 {2} 처리 중..." -f $Count, $RouteList.Count, $Rid)

    $QueryParams = @{
        serviceKey        = $ServiceKey
        $RouteIdParamName = $Rid
    }

    # responseType 계열은 설정이 있을 때만 추가
    if ($Config.responseTypeParamName -and $Config.responseTypeValue) {
        $QueryParams[$Config.responseTypeParamName] = $Config.responseTypeValue
    }

    # cityCode 등 선택 파라미터는 additionalQueryParams 에 있을 때만 추가
    if ($Config.additionalQueryParams) {
        foreach ($prop in $Config.additionalQueryParams.PSObject.Properties) {
            if (-not [string]::IsNullOrWhiteSpace([string]$prop.Value)) {
                $QueryParams[$prop.Name] = [string]$prop.Value
            }
        }
    }

    $ApiUrl = "{0}?{1}" -f $RequestUrl, (ConvertTo-QueryString -Params $QueryParams)

    $CollectSucceeded = $false
    $LoadSucceeded = $false
    $HttpStatus = ""
    $LastError = ""

    # -------------------------------
    # 3-1. API 호출 및 원본 저장
    # -------------------------------
    for ($try = 0; $try -le $MaxRetries; $try++) {
        try {
            $Response = Invoke-WebRequest -Uri $ApiUrl -Method $HttpMethod -TimeoutSec 30 -OutFile $FilePath -PassThru -UseBasicParsing
            $HttpStatus = [string]$Response.StatusCode
            $CollectSucceeded = $true
            break
        }
        catch {
            $LastError = $_.Exception.Message
            if ($try -lt $MaxRetries) {
                Start-Sleep -Seconds $RetryDelaySeconds
            }
        }
    }

    if (-not $CollectSucceeded) {
        Write-Warning ("   -> [수집 실패] route_id={0} / {1}" -f $Rid, $LastError)
        Add-ManifestRow -ManifestPath $ManifestPath `
            -RouteIdRaw $Rid `
            -SnapshotFile $FileName `
            -CollectStatus "FAILED" `
            -LoadStatus "NOT_RUN" `
            -HttpStatus $HttpStatus `
            -ErrorMessage $LastError

        Start-Sleep -Milliseconds $SleepMilliseconds
        continue
    }

    # -------------------------------
    # 3-2. 로더 스크립트 호출
    # -------------------------------
    try {
        & $LoaderScript -RouteId $Rid -SnapshotFile $FilePath -DbName $DbName -DbUser $DbUser
        if ($LASTEXITCODE -ne 0) {
            throw "load_route_links_api.ps1 가 비정상 종료했습니다."
        }

        $LoadSucceeded = $true
        Write-Host "   -> 수집 및 적재 완료."
    }
    catch {
        $LastError = $_.Exception.Message
        Write-Warning ("   -> [적재 실패] route_id={0} / {1}" -f $Rid, $LastError)
    }

    Add-ManifestRow -ManifestPath $ManifestPath `
        -RouteIdRaw $Rid `
        -SnapshotFile $FileName `
        -CollectStatus "SUCCESS" `
        -LoadStatus $(if ($LoadSucceeded) { "SUCCESS" } else { "FAILED" }) `
        -HttpStatus $HttpStatus `
        -ErrorMessage $LastError

    Start-Sleep -Milliseconds $SleepMilliseconds
}

# ------------------------------------------------------------
# 4. 종료 메시지
# ------------------------------------------------------------
Write-Host "[3/4] 전체 수집/적재 루프 종료"
Write-Host "[INFO] 원본 스냅샷 위치: $SnapshotDir"
Write-Host "[INFO] 수집 이력(manifest) 위치: $ManifestPath"
Write-Host "[4/4] 다음 단계: COUNT(DISTINCT route_id_raw) 확인 후 승격 스크립트 실행"