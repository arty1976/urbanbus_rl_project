# fact_stop_usage_hourly 데이터 계약 (Data Contract)

본 문서는 정류장별 시간대별 이용량 통계를 저장하는 표준 팩트 테이블의 정의를 기술합니다.

## 1. 개요
- **목적**: 노드 수준의 수요 분석 및 그래프 상태 시계열(`graph_state_timeslice`) 생성을 위한 기초 데이터 제공.
- **해상도**: 시간(1시간) - 공간(정류장).

## 2. 테이블 스키마

| 컬럼명 | 데이터 타입 | 필수 여부 | 설명 |
| :--- | :--- | :--- | :--- |
| **service_date** | DATE | 필수 (PK) | 운행 일자 |
| **service_hour** | INT | 필수 (PK) | 시간대 (0-23) |
| **stop_id** | VARCHAR(50) | 필수 (PK) | 정류장 식별자 (`dim_stop.stop_id` 외래키) |
| **boardings** | INT | 필수 | 승차 인원 총합 |
| **alightings** | INT | 필수 | 하차 인원 총합 |
| **source_system** | VARCHAR(100) | 필수 | 원천 시스템 (예: `daegu_transport_card_synth_api`) |
| **data_tier** | VARCHAR(20) | 필수 | 데이터 등급 (`synthetic`, `observed`) |
| **is_observed** | BOOLEAN | 필수 | 실측 여부 (`true`: 실측, `false`: 합성/가공) |
| **source_ref** | VARCHAR(255) | 선택 | 원천 파일명 또는 API 배치 ID |
| **created_at** | TIMESTAMP | 필수 | 레코드 생성 시각 |

## 3. 집계 및 표준화 규칙 (합성 데이터 기준)
- **승차수(boardings)**: `RIDE_DT`를 기준으로 시각을 추출하여 해당 시간대(`service_hour`)에 `UTZTN_NOPE`를 합산.
- **하차수(alightings)**: `GOFF_DT`를 기준으로 시각을 추출하여 해당 시간대(`service_hour`)에 `UTZTN_NOPE`를 합산.
- **중복 처리**: `(service_date, service_hour, stop_id)` 기준으로 `UPSERT` (Overwrite) 수행.

## 4. 데이터 정합성 지표
- `boardings` / `alightings` 합계는 원천 Staging의 `UTZTN_NOPE` 합계와 일치해야 함.
- `stop_id`는 반드시 `dim_stop`에 존재해야 함 (미매핑 데이터는 별도로 관리).
