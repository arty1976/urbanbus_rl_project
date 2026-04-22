# 교통카드 데이터 파이프라인 키 및 단위 정의

본 문서는 동적 운영 대중버스 데이터 파이프라인에서 사용되는 식별자, 해상도, 규약 등을 정의합니다.

## 1. 식별자 (Keys)
* **Primary Key (PK)**: 각 테이블의 레코드를 고유하게 식별하는 유일 키 (예: `stop_id`, `route_id`, `link_id`)
* **Alternate Key (AK)**: PK 외에 외부 데이터와의 결합 및 통계를 위해 사용될 수 있는 고유한 대체 키 (예: 국가 표준 정류장 ID, 외부 연계 시스템의 `ext_station_id` 등)
* **service_date (영업일자)**
  * Definition: 대중교통 운영 및 배차의 기준이 되는 일자
  * Data Type: DATE
  * Temporal Granularity: 1일 (Daily)
  * Spatial Granularity: N/A
  * Null Semantics: NOT NULL
* **time_bucket (시간대 집계단위)**
  * Definition: 통행량 및 운영 데이터를 집계하기 위한 시간 구간
  * Data Type: INTEGER (0~23) 또는 TIME
  * Temporal Granularity: 1시간 (Hourly) 등
  * Spatial Granularity: N/A
  * Null Semantics: NOT NULL

## 2. 데이터 해상도 (Granularity)
* **Temporal Granularity (시간적 해상도)**
  * `fact_stop_usage_hourly`: 1시간 단위 (Hourly)로 집계
  * `fact_od_daily`: 1일 단위 (Daily)로 집계
* **Spatial Granularity (공간적 해상도)**
  * 정류장 (Stop): 가장 낮은 수준의 점(Point) 단위 공간 기준 (`dim_stop` 기준)
  * 노선 (Route): 여러 정류장을 포함하는 선형 집합 (`dim_route` 기준)
  * 노선 링크 (Route Link): 정류장과 정류장 사이의 물리적 구간 (`gis_route_link` 기준)

## 3. 결측치 처리 (Null Semantics)
* 승객 수나 횟수 등 측정값(Measure)에서 관측되지 않은 데이터는 단순 `0`이 아닌 `NULL`로 처리하여 관측 불가 상태와 실제 0회 이용을 구분합니다.
* 외래 키(Foreign Key)에 매핑되지 않는 미확인 참조의 경우, 알 수 없는 상태를 나타내는 더미(Dummy) 레코드 ID(예: `-1` 또는 `UNKNOWN_STOP`)를 생성하여 참조 스키마 무결성을 유지하고 실제 브릿지/팩트 테이블에 `NULL` 값을 넣는 것을 지양합니다.

## 4. 조인 규칙 (Join Rules - Critical)
* **주의사항**: **"stop_name으로 조인하지 말 것"**
* 정류장 이름(`stop_name`)은 중복될 수 있고 시간이 지나면서 변경될 수 있습니다. 테이블 간의 조인은 반드시 고유 식별자인 `stop_id` 또는 보장된 `Alternate Key`만을 사용해야 합니다.
