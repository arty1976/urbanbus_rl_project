# Step 99-A /getBs02 샘플 수집 실행 런북

목적: Daegu BIS `/getBs02` 노선별 경유 정류장 정보 조회 API의 샘플 응답을 1~3개 route_id에 대해서만 수집하고, `route_id`, `direction_id`, `ordered_stop_sequence` 확보 가능성을 판정한다.

## Swagger GW 방식 적용 원칙

1. Swagger UI에서 `/getBs02` 항목을 연다.
2. `OpenAPI 실행 준비`를 누른다.
3. `serviceKey`에는 공공데이터포털의 **일반 인증키(Decoding)** 를 입력한다.
4. Swagger에 표시되는 필수 요청 파라미터만 먼저 입력한다.
5. 호출 후 Swagger 하단의 curl 또는 Request URL을 확인한다.
6. 실제 endpoint URL, route parameter 이름이 기본값과 다르면 `--BaseUrl`, `--RouteParam`으로 덮어쓴다.

## PowerShell 실행 예시

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"

# 중요: 일반 인증키(Decoding)를 사용한다. 채팅창이나 git에는 저장하지 않는다.
$env:DAEGU_BIS_SERVICE_KEY = "여기에_일반_인증키_Decoding"

# 먼저 1개 노선만 샘플 호출
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99a_getbs02_sample_audit.ps1 `
  -RouteId "샘플_노선ID_1" `
  -MaxRoutes 1
```

Swagger에서 확인한 route parameter 이름이 `routeId`가 아니면 다음처럼 바꾼다.

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99a_getbs02_sample_audit.ps1 `
  -RouteId "샘플_노선ID_1" `
  -RouteParam "routeId" `
  -MaxRoutes 1
```

Swagger의 endpoint가 기본값과 다르면 다음처럼 바꾼다.

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99a_getbs02_sample_audit.ps1 `
  -BaseUrl "https://apis.data.go.kr/6270000/dbmsapi01/getBs02" `
  -RouteParam "routeId" `
  -RouteId "샘플_노선ID_1" `
  -MaxRoutes 1
```

pageNo / numOfRows 같은 추가 파라미터가 Swagger에 표시되면 다음처럼 넣는다.

```powershell
powershell -ExecutionPolicy Bypass -File .\05_training\adapters\run_step99a_getbs02_sample_audit.ps1 `
  -RouteId "샘플_노선ID_1" `
  -MaxRoutes 1 `
  -ExtraParam "pageNo=1" `
  -ExtraParam "numOfRows=500"
```

DB read-only coverage도 같이 확인하려면 `URBANBUS_DB_DSN`을 설정한다.

```powershell
$env:URBANBUS_DB_DSN = "postgresql://postgres:비밀번호@localhost:5432/urbanbus"
```

## 결과 확인

```powershell
Get-Content .\artifacts\daegu_bis_api_audit\getbs02_route_stop_sequence_audit_report.md -Encoding UTF8
```

판정 기준:

- `has_route_id=true`이고 기존 노선 카탈로그와 연결 가능하면 `route_id=observed 후보`
- `has_direction_id=true`이면 `direction_id=observed 또는 partial_observed 후보`
- `has_stop_id=true` + `has_sequence=true`이면 `ordered_stop_sequence=observed 후보`
- `has_stop_id=true`이나 `has_sequence=false`이면 ordered sequence는 아직 missing 유지
- graph node coverage가 낮으면 Causal simulator v2 required contract로 바로 승격하지 않는다.

## 금지 사항

- API 전수 수집 금지
- DB write 금지
- tensor DB overwrite 금지
- 2023 CSV 재적재 금지
- API key 하드코딩 및 git commit 금지
