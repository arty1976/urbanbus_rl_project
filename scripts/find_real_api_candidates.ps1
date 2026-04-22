param(
    [string]$ProjectRoot = "D:\urbanbus_rl_project"
)

$ErrorActionPreference = "SilentlyContinue"

Write-Host "=== 1) 프로젝트 파일에서 실제 URL/키 후보 찾기 ===" -ForegroundColor Cyan

$targetFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File |
Where-Object {
    $_.Extension -in @(".ps1", ".json", ".yaml", ".yml", ".env", ".txt", ".md", ".js", ".ts", ".py", ".java", ".properties", ".xml")
}

$patterns = @(
    "https://",
    "http://",
    "serviceKey",
    "apiKey",
    "baseUrl",
    "base_url",
    "endpoint",
    "openapi",
    "swagger",
    "data.go.kr",
    "TransportationCard",
    "SyntheticData",
    "DaeguTransportationCard",
    "businfo",
    "smartcard"
)

foreach ($pattern in $patterns) {
    Write-Host "`n--- pattern: $pattern ---" -ForegroundColor Yellow
    Select-String -Path $targetFiles.FullName -Pattern $pattern -SimpleMatch |
    Select-Object Path, LineNumber, Line |
    Format-Table -Wrap -AutoSize
}

Write-Host "`n=== 2) 특히 볼 파일 우선 출력 ===" -ForegroundColor Cyan
$priorityFiles = @(
    "$ProjectRoot\configs\secrets.json",
    "$ProjectRoot\02_ingest_jobs\fetch_bus_api.ps1",
    "$ProjectRoot\scripts\load_route_links_api.ps1",
    "$ProjectRoot\scripts\load_route_links_api_csv.ps1",
    "$ProjectRoot\scripts\load_route_links_generic.ps1",
    "$ProjectRoot\01_data_contracts\route_link_api_profile.md"
)

foreach ($file in $priorityFiles) {
    if (Test-Path $file) {
        Write-Host "`n>>> $file" -ForegroundColor Green
        Get-Content $file | Select-Object -First 120
    }
}

Write-Host "`n=== 3) 로컬 LISTEN 포트 + 프로세스명 확인 ===" -ForegroundColor Cyan
Get-NetTCPConnection -State Listen |
Sort-Object LocalPort |
ForEach-Object {
    $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
    [PSCustomObject]@{
        LocalAddress = $_.LocalAddress
        LocalPort    = $_.LocalPort
        PID          = $_.OwningProcess
        ProcessName  = if ($proc) { $proc.ProcessName } else { "<unknown>" }
        Path         = if ($proc) { $proc.Path } else { "" }
    }
} | Format-Table -Wrap -AutoSize

Write-Host "`n=== 4) localhost 후보 포트에 Swagger/OpenAPI 경로 탐색 ===" -ForegroundColor Cyan
$localPorts = Get-NetTCPConnection -State Listen |
Where-Object { $_.LocalAddress -in @("127.0.0.1", "::1", "0.0.0.0", "::") } |
Select-Object -ExpandProperty LocalPort -Unique |
Sort-Object

$paths = @(
    "/",
    "/swagger",
    "/swagger-ui",
    "/swagger-ui.html",
    "/v3/api-docs",
    "/openapi.json",
    "/api-docs",
    "/actuator",
    "/health"
)

$hits = @()

foreach ($port in $localPorts) {
    foreach ($path in $paths) {
        $url = "http://127.0.0.1:$port$path"
        try {
            $resp = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing
            $body = ""
            if ($resp.Content) {
                $body = $resp.Content.Substring(0, [Math]::Min(250, $resp.Content.Length))
            }

            $hits += [PSCustomObject]@{
                Url         = $url
                StatusCode  = $resp.StatusCode
                ContentType = $resp.Headers["Content-Type"]
                BodyPreview = $body
            }
        }
        catch {
            $msg = $_.Exception.Message
            if ($msg -match "404|400|405") {
                $hits += [PSCustomObject]@{
                    Url         = $url
                    StatusCode  = "HTTP error"
                    ContentType = ""
                    BodyPreview = $msg
                }
            }
        }
    }
}

if ($hits.Count -gt 0) {
    Write-Host "`n=== URL 후보 ===" -ForegroundColor Green
    $hits | Sort-Object Url | Format-Table -Wrap -AutoSize
}
else {
    Write-Host "응답 있는 localhost 후보를 찾지 못했습니다." -ForegroundColor Red
}

Write-Host "`n=== 5) 결론 가이드 ===" -ForegroundColor Cyan
Write-Host "A. configs\secrets.json / fetch_bus_api.ps1 에 실제 https API 주소가 있으면 그쪽이 우선입니다."
Write-Host "B. localhost 후보가 없으면 로컬 API 서버는 현재 실행 중이 아닐 가능성이 큽니다."
Write-Host "C. Swagger/OpenAPI URL이 잡히면 그 문서에서 실제 엔드포인트를 확인하세요."