# 02_ingest_jobs

대구광역시 버스 위치정보 및 정류소 데이터를 수집하고 로컬에 적재하기 위한 스크립트 공간입니다.

## 풀더 구조 (데이터 적재용)
이 폴더 내부 구동용 스크립트에 의해 원본 저장 목적으로 아래 구조가 사용(또는 자동 생성)됩니다.
- `data/raw_csv/`: 대구 정류소 위치정보 등 외부 CSV 파일의 원본 복사본 보관 폴더
- `data/raw_json/`: 대구버스정보시스템 Open API 응답(로 데이터 JSON) 보관 폴더

## 스크립트 설명 및 실행 순서

### 1. `load_daegu_stops.ps1`
- **목적**: 기존에 존재하는 대구 정류소 위치정보 CSV 파일을 읽어서, `data/raw_csv` 하위 폴더에 정리된 복사본 파일로 저장합니다.
- **실행 예시**: 
  ```powershell
  .\load_daegu_stops.ps1 -SourceCsvPath "C:\temp\source_stops.csv"
  ```
- **파라미터**:
  - `-SourceCsvPath` (필수): 원본 CSV 파일이 위치한 절대 또는 상대 경로입니다.

### 2. `fetch_bus_api.ps1`
- **목적**: 대구버스정보시스템 Open API를 호출하여 버스 위치/노선 등 원본 JSON 응답 데이터를 수신한 뒤 `data/raw_json` 하위 폴더에 타임스탬프(YYYYMMDD_HHmmss) 파일명으로 저장합니다.
- **실행 예시**: 
  ```powershell
  .\fetch_bus_api.ps1 -ApiKey "YOUR_API_KEY_HERE"
  ```
- **파라미터**:
  - `-ApiKey` (선택): Open API 호출용 서비스 인증키입니다. 파라미터가 없으면 시스템 환경변수 `DAEGU_BUS_API_KEY` 값을 읽어서 사용합니다.

### 3. `route_link_sequence` 승격 실행 (Promotion)
- **운영 기준본 (Operational Standard)**: [promote_route_link_sequence_dedup.sql](./promote_route_link_sequence_dedup.sql)
  - 중복 데이터(dedup) 처리 로직과 예외 노선 필터링이 포함된 승인된 버전입니다.
- **레거시 (Legacy/Audit)**: `promote_route_link_sequence.sql` (최초 승격 로직 보존용)
- **실행 절차**: [route_link_promotion_execution_runbook.md](./route_link_promotion_execution_runbook.md) 및 [운영 체크리스트](../01_data_contracts/route_link_operational_checklist.md) 참조
- **주요 내용**: 
  - `vw_stg_daegu_route_links_api_dedup` 뷰를 기반으로 무결성이 검증된 데이터만 분석 레이어로 승격합니다.

## 개발/운영 전 주의사항
- 스크립트 파일 내부에 사용자가 직접 실제 API URL 주소 등으로 수정해야 하는 영역들에 `TODO` 주석이 작성되어 있습니다.
- 모든 스크립트는 `UTF-8` 인코딩 기준으로 출력 및 동작하며, 예기치 않은 오류 시 에러 로그를 출력하고 즉시 실행을 중단합니다.
