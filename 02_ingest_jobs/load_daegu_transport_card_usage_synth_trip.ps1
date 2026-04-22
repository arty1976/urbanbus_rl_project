<#
.SYNOPSIS
    대구 교통카드 이용 합성 데이터 API -> Staging 테이블 적재 스크립트 (Trip-level)
.DESCRIPTION
    - 대구 전용 엔드포인트(getDaeguTransportationCardUsageSyntheticData) 사용
    - OPR_YMD, RTE_ID, USERS_TYPE_CD 필수 파라미터 대응
    - 에러 발생 시 HTTP 상세 정보(Status, URL, Body) 출력
#>

# 1. 설정 및 보안 정보 로드
$ConfigPath = "D:\urbanbus_rl_project\configs\secrets.json"
if (-not (Test-Path $ConfigPath)) {
    throw "[ERROR] Config file not found: $ConfigPath"
}
$Secrets = Get-Content $ConfigPath | ConvertFrom-Json
$ServiceKey = $Secrets.serviceKey

# 2. 엔드포인트 및 파라미터 설정
$ApiBaseUrl = "https://apis.data.go.kr/1613000/RegionalTransportationCardUsageSyntheticData"
$ApiPath    = "getDaeguTransportationCardUsageSyntheticData"

$BatchId = Get-Date -Format "yyyyMMdd_HHmmss"
$PageNo = 1
$NumOfRows = 1000

# 필터 값 (웹 조회 페이지 성공 조합 반영)
$OprYmd = "20260228"
$RteId = "1"
$UsersTypeCd = "01"
$RideCtpvCd = "27"

# 3. Request URL 생성
$RequestUrl = "$ApiBaseUrl/$ApiPath" +
    "?serviceKey=$ServiceKey" +
    "&pageNo=$PageNo" +
    "&numOfRows=$NumOfRows" +
    "&dataType=JSON" +
    "&OPR_YMD=$OprYmd" +
    "&RTE_ID=$RteId" +
    "&USERS_TYPE_CD=$UsersTypeCd" +
    "&ride_ctpv_cd=$RideCtpvCd"

Write-Host "[INFO] Fetching synthetic card data (Daegu Specific)..." -ForegroundColor Cyan

# 필수 파라미터 빈값 검사 (사용자 요청 사항)
if ([string]::IsNullOrWhiteSpace($OprYmd)) { throw "OprYmd parameter is required." }
if ([string]::IsNullOrWhiteSpace($RteId)) { throw "RteId parameter is required." }
if ([string]::IsNullOrWhiteSpace($UsersTypeCd)) { throw "UsersTypeCd parameter is required." }

Write-Host "[INFO] RequestUrl: $RequestUrl"

# 4. API 호출 및 에러 핸들링 (사용자 요청 기준 코드)
try {
    $resp = Invoke-RestMethod -Uri $RequestUrl -Method Get -ErrorAction Stop
}
catch {
    $status = "Unknown"
    $body = "No Body Content"

    if ($_.Exception.Response) {
        try {
            $status = [int]$_.Exception.Response.StatusCode
        } catch {
            $status = "$($_.Exception.Response.StatusCode)"
        }

        try {
            $stream = $_.Exception.Response.GetResponseStream()
            if ($stream) {
                $reader = New-Object System.IO.StreamReader($stream)
                $body = $reader.ReadToEnd()
                $reader.Close()
            }
        } catch {
            $body = "Could not read response body"
        }
    }

    Write-Host "[ERROR] API Call Failed!" -ForegroundColor Red
    Write-Host "[ERROR] HTTP Status: $status" -ForegroundColor Red
    Write-Host "[ERROR] URL: $RequestUrl" -ForegroundColor Gray
    Write-Host "[ERROR] Response Body: $body" -ForegroundColor Yellow
    exit 1
}

# 5. 응답 분석
$Header = $resp.response.header
$BodyContent = $resp.response.body

if ($null -eq $Header -or $Header.resultCode -ne "00") {
    $ErrMsg = "Unknown Error"
    if ($null -ne $Header.resultMsg) { $ErrMsg = $Header.resultMsg }
    Write-Error "[ERROR] API Logical Error: $ErrMsg"
    exit 1
}

$Items = $BodyContent.items.item
if ($null -eq $Items) {
    Write-Host "[WARN] No data found for the given parameters." -ForegroundColor Yellow
    exit 0
}

# 1건인 경우에도 배열로 처리
$ItemArray = @($Items)
Write-Host "[INFO] Processing $($ItemArray.Count) records..." -ForegroundColor Green

# 6. CSV 및 SQL 생성 준비
$TempFile = "D:\urbanbus_rl_project\scratch\stg_load_$BatchId.csv"
$CsvData = New-Object System.Collections.Generic.List[PSObject]

foreach ($item in $ItemArray) {
    # record_hash 생성 (SHA256)
    $RawKey = "" + $item.OPR_YMD + $item.VR_CARD_NO + $item.RIDE_DT + $item.RIDE_STTN_ID + $item.GOFF_DT + $item.GOFF_STTN_ID
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($RawKey)
    $Stream = New-Object System.IO.MemoryStream(,$Bytes)
    $Hash = (Get-FileHash -InputStream $Stream -Algorithm SHA256).Hash
    $Stream.Dispose()
    
    $obj = New-Object PSObject
    $obj | Add-Member NoteProperty "record_hash" $Hash
    $obj | Add-Member NoteProperty "batch_id" $BatchId
    $obj | Add-Member NoteProperty "source_ref" "api_page_$PageNo"
    $obj | Add-Member NoteProperty "raw_payload_json" ($item | ConvertTo-Json -Compress)
    $obj | Add-Member NoteProperty "opr_ymd" $item.OPR_YMD
    $obj | Add-Member NoteProperty "ride_dt" $item.RIDE_DT
    $obj | Add-Member NoteProperty "goff_dt" $item.GOFF_DT
    $obj | Add-Member NoteProperty "rte_id" $item.RTE_ID
    $obj | Add-Member NoteProperty "ride_sttn_id" $item.RIDE_STTN_ID
    $obj | Add-Member NoteProperty "goff_sttn_id" $item.GOFF_STTN_ID
    $obj | Add-Member NoteProperty "utztn_nope" $item.UTZTN_NOPE
    $obj | Add-Member NoteProperty "msv_intrpl_yn" $item.MSV_INTRPL_YN
    $obj | Add-Member NoteProperty "vr_card_no" $item.VR_CARD_NO
    $obj | Add-Member NoteProperty "card_se_cd" $item.CARD_SE_CD
    $obj | Add-Member NoteProperty "trnf_cnt" $item.TRNF_CNT
    $obj | Add-Member NoteProperty "users_type_cd" $item.USERS_TYPE_CD
    $obj | Add-Member NoteProperty "utztn_dstnc" $item.UTZTN_DSTNC
    $obj | Add-Member NoteProperty "brdg_hr" $item.BRDG_HR
    $obj | Add-Member NoteProperty "ride_ctpv_cd" $item.RIDE_CTPV_CD
    $obj | Add-Member NoteProperty "goff_ctpv_cd" $item.GOFF_CTPV_CD
    $obj | Add-Member NoteProperty "clcln_bzmn_id" $item.CLCLN_BZMN_ID
    $obj | Add-Member NoteProperty "clcln_bzmn_trfc_mns_cd" $item.CLCLN_BZMN_TRFC_MNS_CD
    
    $CsvData.Add($obj)
}

# CSV 파일 저장
$CsvData | Export-Csv -Path $TempFile -NoTypeInformation -Encoding UTF8

# psql용 COPY 명령어 생성
$ColList = "record_hash, batch_id, source_ref, raw_payload_json, opr_ymd, ride_dt, goff_dt, rte_id, ride_sttn_id, goff_sttn_id, utztn_nope, msv_intrpl_yn, vr_card_no, card_se_cd, trnf_cnt, users_type_cd, utztn_dstnc, brdg_hr, ride_ctpv_cd, goff_ctpv_cd, clcln_bzmn_id, clcln_bzmn_trfc_mns_cd"
$PsqlCmd = "\copy public.stg_daegu_transport_card_usage_synth_trip ($ColList) from '$TempFile' with (format csv, header true, encoding 'UTF8');"

$SqlFile = "D:\urbanbus_rl_project\scratch\stg_load_$BatchId.sql"
$PsqlCmd | Out-File -FilePath $SqlFile -Encoding UTF8
Write-Host "[INFO] SQL file created: $SqlFile" -ForegroundColor Cyan
Write-Host "[TODO] Execution: psql -U postgres -d urbanbus -f $SqlFile" -ForegroundColor Yellow
