---
name: dim_stop_promotion_guard
description: Safeguards the promotion of bus stop source data (Shapefile bs_YYYYMMDD, staging stops, or CSV) into the dim_stop dimension table, enforcing a wide schema that preserves alt-key columns (bis_id, bs_id, mobile_no, legacy_stop_id, node_uid) for future downstream mapping. Use this whenever creating or rebuilding dim_stop for any city's bus network — Daegu, Seoul, Busan, Daejeon, Gwangju, or new regions — even if the user only says "dim_stop 만들어" or "stop 테이블 정리해". Triggers on dim_stop creation, stop dimension table design, bus stop master table work, or when user mentions that a later stage (edge build, mapping, RL feature) cannot find a key. Enforces SRID, city-bounds, and the critical "no stop_name join ever" rule.
---

# Purpose

이 스킬은 Shapefile (`bs_YYYYMMDD`), staging 스톱 테이블, 혹은 CSV 원천으로부터
`dim_stop` 차원 테이블을 생성·재생성할 때, **추후 매핑/엣지/RL 피처 단계에서
겪을 문제를 사전에 차단할 수 있는 "넓은 스키마"** 를 강제합니다.

### 왜 이 가드가 필요한가 (one-line)
얕은 `dim_stop` (stop_id + geom 만 있는 3-4 컬럼 테이블) 은 데이터 준비 시점엔
가볍고 예뻐 보이지만, 몇 주~몇 달 뒤 **coverage gap 진단 (예: `route_link_sequence` ↔
`dim_stop` 22.84%) 단계에서 alt-key 매핑이 원천적으로 불가능** 해진다.
**Daegu 2026-04-19 사례**: dim_stop 이 7개 컬럼뿐이어서 D1 진단 Hypothesis A
(alt-key 매핑) 를 시도조차 못하고 곧바로 기각. 원천 Shapefile 에는 분명 있던
`bis_id`, `bs_id` 등의 보조 표기를 적재 시점에 버렸기 때문. 이 스킬은 그 실수를
되풀이하지 않게 막는다.

---

## Core Rules

1. **Mandatory columns (필수 9)** — 하나라도 빠지면 promotion 거절:
   - `stop_id` varchar(50) PRIMARY KEY
   - `stop_name` text (NOT NULL)
   - `longitude` numeric(12,8) (NOT NULL)
   - `latitude`  numeric(12,8) (NOT NULL)
   - `geom_5187` geometry(Point,5187) — 미터 단위 CRS, `ST_Distance` 전용
   - `geom_4326` geometry(Point,4326) — 경위도, 지도 표시 전용
   - `node_uid` text UNIQUE — `'STOP:' || stop_id` 자동 계산
   - `created_at` / `updated_at` timestamptz DEFAULT now()
2. **Recommended alt-key columns (권장 5)** — 원천에 있으면 **반드시 보존**:
   - `bis_id` text — BIS (Bus Information System) 내부 ID
   - `bs_id` text — Shapefile bs_YYYYMMDD 원표기
   - `mobile_no` text — 모바일 안내용 정류장번호 (승객 가시번호)
   - `legacy_stop_id` text — 이전 시스템 ID (마이그레이션 추적용)
   - `external_code` text — 지자체 공개 API 응답의 외부 코드
   이들 중 **원천에 존재하는 것은 하나라도 버리지 말 것**. 당장 쓸 일 없어 보여도
   3개월 뒤 Hypothesis A 진단에서 생명줄이 된다.
3. **Name join 금지**: `stop_name` 은 중복이 흔하다 (Daegu 케이스: 250건 중복).
   어떠한 분석·매핑에서도 `JOIN ... ON stop_name = stop_name` 을 쓰지 않는다.
   오직 `stop_id` / `node_uid` / alt-key 만이 조인 키.
4. **SRID 고정**: `geom_5187` 은 EPSG:5187 (한국 중부 원점 TM) 로 제약. 다른
   도시가 다른 좌표계를 쓰면 컬럼명과 SRID 를 페어로 교체 (`geom_5186` 등).
   **임시 `ST_Transform` 을 view 에 숨기지 말 것** — 거리 계산용 CRS 는 영구
   컬럼으로 물리적 보존.
5. **Geometry 생성**: `ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)` 으로
   4326 을 먼저 만들고, `ST_Transform(..., 5187)` 으로 5187 을 파생. 두 컬럼
   모두 원천 데이터에서 직접 읽지 말고 경위도에서 계산.
6. **Upsert**: `ON CONFLICT (stop_id) DO UPDATE SET (모든 변경가능 컬럼) +
   updated_at = now()`. 절대 `DELETE + INSERT` 하지 말 것 (외래키 위반 유발).
7. **Stale 처리**: 신규 Shapefile 에서 사라진 stop 은 `DELETE` 아니라
   `is_active = false` + `deactivated_at = now()` 로 soft-delete.

---

## Pre-validation (promotion 직전 검사)

원천 (Shapefile/CSV/staging) 에서 dim_stop 로 승격하기 전:

1. **Row count sanity**: 원천 행수와 예상 stop 규모 (도시 단위: Daegu ≈ 5700,
   Seoul ≈ 13000, Busan ≈ 7500). 10x 넘게 벗어나면 **원천 잘못됨** 의심.
2. **PK 후보 null/중복**: `stop_id` 후보 컬럼의 NULL 0 건, 중복 0 건 보장.
   Daegu 케이스처럼 `stop_id` 가 있는데 `UNKNOWN` 채워진 행이 있을 수 있으니
   `WHERE stop_id IS NULL OR stop_id IN ('UNKNOWN', '', '-')` 로 확인.
3. **좌표 sanity**: 도시 권역 bounding box 확인. Daegu 는 대략
   `longitude BETWEEN 128.4 AND 129.0 AND latitude BETWEEN 35.6 AND 36.1`.
   권역 이탈 정류장은 Shapefile 오염 가능성.
4. **Geometry validity**: `ST_IsValid(geom)`, `ST_IsEmpty(geom)` 확인.
5. **원천 alt-key 컬럼 검색**: 원천 테이블 information_schema 를 스캔하여
   `%id%`, `%no%`, `%code%` 패턴 컬럼 목록을 뽑아 사용자에게 "이 중 alt-key
   로 보존할 컬럼" 을 확인받는다. 모르면 **다 보존하는 쪽이 안전**.
6. **stop_name 중복 스캔**: 중복 수를 사전 보고. 0 이면 이상적이지만, 한국
   지자체 데이터에서 수십~수백 건 중복은 정상. 사용자가 "join by name 절대
   금지" 원칙을 확인했는지 로깅.

---

## Post-validation (promotion 직후 검사)

`dim_stop_readiness.sql` (또는 유사) 을 실행하고 다음 값을 보고:

| ID | 체크 | 기대값 |
|---|---|---|
| D1 | `dim_stop_rows` | > 0, 도시 규모 범위 안 |
| D2 | `null_stop_id`, `null_stop_name`, `null_longitude`, `null_latitude` | **0** 전부 |
| D3 | `duplicate_stop_id_cnt` | **0** |
| D4 | `invalid_geom_5187_cnt`, `empty_geom_5187_cnt` | **0, 0** |
| D5 | `outside_city_bounds_cnt` (도시별 bbox) | 0 이상적, 소량(<5) 허용 |
| D6 | `duplicate_stop_name_cnt` | 참고용, 0 이상적이지만 한국은 수십~수백 정상 |
| D7 | `alt_key_coverage` — 각 보조키 (bis_id, bs_id, mobile_no…) 별 채워진 비율 | 원천에 있었다면 > 95% |
| D8 | `node_uid_pattern_cnt` — `node_uid LIKE 'STOP:%'` 개수 vs 전체 | 100% |

---

## 판독 기준 (Readiness Guide)

- **PASS (APPROVED)**: D2/D3/D4 전부 0, D5 < 5, D7 (alt-key 를 적어도 1개)
  ≥ 95%.
- **WARN (HOLD)**: D6 (이름중복) 이 총 행수의 20% 넘거나, D5 가 5~20 건.
  운영자 리뷰 후 승격 허용.
- **FAIL (BLOCKED)**: D2/D3/D4 중 단 1건이라도 0 이 아니거나, D7 (alt-key 를
  적어도 1개) < 50%. 즉시 승격 중단, 원천 재점검.

---

## Failure Modes (흔한 함정)

### F1. "stop_id 만 있으면 된다" 의 덫
당장 PK 로 동작해 보여서 bis_id/bs_id 등을 버리고 싶어진다. 6개월 뒤
coverage gap 진단에서 **Hypothesis A (alt-key) 를 시도할 수 없게 되어** 전체
진단 절차가 막힌다. Daegu 2026-04-19 사례 반복 금지.

### F2. stop_name 을 natural key 로 오인
중복이 워낙 많아 실제 Daegu 는 250건. 한 번만 name-join 을 허용하면 downstream
에서 승객 수요가 2배 이상 부풀려 기록되는 사고가 난다. **절대 금지**.

### F3. geom SRID 누락
`ST_SetSRID` 를 생략하고 `ST_MakePoint(lon, lat)` 로만 geom 컬럼을 채우면
SRID=0 이 되어 이후 `ST_Distance` 결과가 무의미한 도(degree) 단위로 계산됨.
`ST_SetSRID(ST_MakePoint(lon, lat), 4326)` 을 반드시 명시.

### F4. 4326 → 5187 직접 변환만 하고 4326 을 안 남김
지도 시각화 (leaflet / folium / QGIS) 에서 매번 역변환을 해야 해서 느려짐.
**두 컬럼 모두 보존**.

### F5. CSV 인코딩 (UTF-8 BOM / CP949) 혼재
Windows 원천 CSV 는 CP949 (EUC-KR) 인 경우가 많다. `\copy ... with (format
csv, encoding 'UHC')` 로 명시. BOM 이 붙어 있으면 첫 컬럼명이 `\ufeffstop_id`
로 들어간다 — 승격 전 BOM 제거.

---

## Other-City Adaptation Checklist

1. **도시 bounding box** 확인. Daegu `[128.4~129.0, 35.6~36.1]`, Seoul
   `[126.7~127.3, 37.4~37.7]`, Busan `[128.9~129.3, 34.9~35.3]` 등. 도시별
   bbox 를 pre-validation 쿼리에 대입.
2. **예상 규모**: Daegu ~5700, Seoul ~13000, Busan ~7500, Daejeon ~2500.
   규모가 맞지 않으면 Shapefile 범위 자체를 의심.
3. **원천 alt-key 네이밍 다름**: Seoul 은 `ars_id`, 부산은 `stop_no` 등
   도시별 차이. information_schema 스캔 후 사용자 confirm.
4. **이름 중복률**: 한국 도시 대부분 5~10% 수준. 20% 넘으면 staging 데이터
   품질 이슈.
5. **SRID 차이**: Daegu / 영남권은 EPSG:5187 (중부), 영동권은 EPSG:5186,
   제주는 EPSG:5188. Shapefile `.prj` 파일 먼저 확인.

---

## 선행 스킬 / 후행 스킬 관계

- **선행**: `shape-inspection-skill` (bs_YYYYMMDD 레이어 검수),
  `data-contract-audit-skill` (stop 컬럼 semantic 확인).
- **후행**: `spatial_graph_mapping_guard` (dim_stop ↔ link 매핑),
  `stop_to_stop_edge_leg_aggregator` (dim_stop 좌표로 edge 거리 계산),
  `db_coverage_gap_diagnoser` (dim_stop 과 다른 테이블 간 coverage gap 진단).

요컨대 이 스킬은 **전체 파이프라인의 발목** 을 잡는 위치에 있다. 얕게 만들면
downstream 전체가 6개월 뒤에 아프다.

---

## 호출 프롬프트 예시

- "대전시 정류장 Shapefile 을 `dim_stop` 로 승격할 건데
  `dim_stop_promotion_guard` 스킬의 Mandatory 9 + Recommended 5 컬럼 규약을
  지켜서 DDL 과 적재 스크립트 짜줘."
- "서울 dim_stop 에 Hypothesis A 진단이 안 먹혀. `dim_stop_promotion_guard`
  의 Recommended alt-key 컬럼이 얼마나 채워져 있는지 D7 쿼리로 먼저 확인해줘."
- "부산 dim_stop 기존 테이블을 스킬 규약에 맞게 ALTER 로 확장해줘. 원본은
  건드리지 말고 alt-key 컬럼만 NULL 허용으로 추가 후 backfill 스크립트 초안."
- "광주 정류장 데이터 승격 전에 `dim_stop_promotion_guard` 의 Pre-validation
  6 단계를 돌려서 PASS/WARN/FAIL 판정해줘."
