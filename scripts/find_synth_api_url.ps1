param(
    [string]$ProjectRoot = "D:\urbanbus_rl_project",
    [string]$EndpointKeyword = "getDaeguTransportationCardUsageSyntheticData"
)

$ErrorActionPreference = "SilentlyContinue"

Write-Host "=== 1) 프로젝트 파일 내 API 키워드 검색 ===" -ForegroundColor Cyan
$patterns = @(
    $EndpointKeyword,
    "TransportationCardUsageSyntheticData",
    "swagger",
    "openapi",
    "v3/api-docs",
    "localhost",
    "127.0.0.1",
    ":8080",
    "http://",
    "https://"
)

$files = Get-ChildItem -Path $ProjectRoot -Recurse -File |
Where-Object {
    $_.Extension -in @(".ps1", ".json", ".yaml", ".yml", ".env", ".txt", ".md", ".js", ".ts", ".py", ".java", ".properties", ".xml")
}

foreach ($pattern in $patterns) {
    Write-Host "`n--- pattern: $pattern ---" -ForegroundColor Yellow
    Select-String -Path $files.FullName -Pattern $pattern -SimpleMatch |
    Select-Object Path, LineNumber, Line |
    Format-Table -AutoSize
}

Write-Host "`n=== 2) 현재 LISTEN 중인 로컬 포트 확인 ===" -ForegroundColor Cyan
$listenPorts = Get-NetTCPConnection -State Listen |
Sort-Object LocalPort |
Select-Object LocalAddress, LocalPort, OwningProcess

$procMap = @{}
Get-Process | ForEach-Object { $procMap[$_.Id] = $_.ProcessName }

$listenPorts | ForEach-Object {
    [PSCustomObject]@{
        LocalAddress  = $_.LocalAddress
        LocalPort     = $_.LocalPort
        OwningProcess = $_.OwningProcess
        ProcessName   = $procMap[$_.OwningProcess]
    }
} | Format-Table -AutoSize

Write-Host "`n=== 3) 로컬 포트 후보 자동 탐색 ===" -ForegroundColor Cyan
$commonPorts = @(
    8080, 8000, 8001, 3000, 5000, 5001, 7000, 7001, 9000, 9001, 8888
)

$actualPorts = (Get-NetTCPConnection -State Listen | Select-Object -ExpandProperty LocalPort -Unique)
$portsToTest = ($commonPorts + $actualPorts) | Sort-Object -Unique

$pathsToTest = @(
    "/",
    "/swagger",
    "/swagger-ui",
    "/swagger-ui.html",
    "/v3/api-docs",
    "/openapi.json",
    "/api-docs",
    "/$EndpointKeyword",
    "/api/$EndpointKeyword"
)

$results = New-Object System.Collections.Generic.List[object]

foreach ($port in $portsToTest) {
    foreach ($path in $pathsToTest) {
        $url = "http://127.0.0.1:$port$path"
        try {
            $resp = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec 3 -UseBasicParsing
            $bodyPreview = ""
            if ($resp.Content) {
                $bodyPreview = $resp.Content.Substring(0, [Math]::Min(300, $resp.Content.Length))
            }

            $results.Add([PSCustomObject]@{
                    Url         = $url
                    StatusCode  = $resp.StatusCode
                    ContentType = $resp.Headers["Content-Type"]
                    BodyPreview = $bodyPreview
                })
        }
        catch {
            # 404도 서버가 살아 있다는 단서가 될 수 있으므로 일부 기록
            $msg = $_.Exception.Message
            if ($msg -match "404" -or $msg -match "405" -or $msg -match "400") {
                $results.Add([PSCustomObject]@{
                        Url         = $url
                        StatusCode  = "HTTP error"
                        ContentType = ""
                        BodyPreview = $msg
                    })
            }
        }
    }
}

if ($results.Count -gt 0) {
    Write-Host "`n=== 4) 응답 있는 URL 후보 ===" -ForegroundColor Green
    $results | Sort-Object Url | Format-Table -AutoSize
}
else {
    Write-Host "`n응답이 있는 URL 후보를 찾지 못했습니다." -ForegroundColor Red
}

Write-Host "`n=== 5) Swagger/OpenAPI 문서 후보 우선 출력 ===" -ForegroundColor Cyan
$results |
Where-Object {
    $_.Url -match "swagger" -or
    $_.Url -match "openapi" -or
    $_.Url -match "api-docs" -or
    $_.BodyPreview -match "openapi" -or
    $_.BodyPreview -match "swagger"
} |
Format-Table -AutoSize

Write-Host "`n=== 6) 엔드포인트 키워드 후보 우선 출력 ===" -ForegroundColor Cyan
$results |
Where-Object {
    $_.Url -match [regex]::Escape($EndpointKeyword) -or
    $_.BodyPreview -match [regex]::Escape($EndpointKeyword)
} |
Format-Table -AutoSize

Write-Host "`n=== 완료 ===" -ForegroundColor Green
Write-Host "1) 위 결과에서 Swagger/OpenAPI 문서 URL이 보이면 먼저 여세요."
Write-Host "2) 그 다음 getDaeguTransportationCardUsageSyntheticData 실제 경로를 확인하세요."
Write-Host "3) 찾은 URL로 perPage=1 호출 후 샘플 JSON(JavaScript Object Notation=자바스크립트 객체 표기법) 응답을 확인하세요."