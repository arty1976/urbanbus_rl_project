# Dimension 승격 계획 (dim_stop, dim_route)

본 문서는 `shape_layer_profile.md` 결과와 `stg_daegu_routes` 원천 데이터를 기반으로 `dim_stop` 및 `dim_route` 디멘전 테이블 생성 및 검증을 위한 SQL 초안과 분석 정보를 정의합니다. (참고: `bridge_route_stop`는 시퀀스 정보 부족으로 현재 승격 보류 상태입니다.)

> **전제 조건 (Prerequisites)**: 
> * 본 계획의 SQL을 온전히 실행하기 위해서는 데이터베이스 내에 공간 데이터 처리를 위한 **PostGIS 익스텐션(extension)이 활성화**되어 있어야 합니다.
> * 공간 레이어 기반 테이블 추출을 위해, `bs_20250903` 원천 파일이 실제 DB 통신이 가능하도록 **사전에 테이블 또는 뷰(View)의 형태로 적재되어 있다는 점을 전제**합니다.

## 1. dim_stop 승격 계획

`bs_20250903` 레이어 기반 정류점 마스터 테이블 승격 계획입니다.

### 1) 제안 컬럼
- `stop_id`: 고유 식별자 (원본 `bs_id`)
- `stop_name`: 정류장 이름 (원본 `bs_nm`)
- `latitude`: 위도 (원본 `ngis_y_pos` 또는 변환된 좌표계 Y축)
- `longitude`: 경도 (원본 `ngis_x_pos` 또는 변환된 좌표계 X축)
- `geom`: 공간 지오메트리 데이터 (원본 `geometry`)

### 2) PK (Primary Key)
- `stop_id` 
  - **Fact**: 공간 탐색 결과 `bs_id`는 결측치(Null) 및 중복이 0건이므로 PK로 완벽하게 적합합니다.

### 3) Null 위험
- `stop_id`: Null 위험 0% (**Fact**)
- `stop_name`: Null 위험 매우 낮음 (**Fact**: 분석 샘플상 필수 파라미터로 처리됨)
- `latitude`, `longitude`, `geom`: 변환 실패 및 공간 객체 누락 외에는 Null 위험 낮음 (**Assumption**)

### 4) Duplicate 위험
- `stop_id`: 중복 위험 0% (**Fact**)
- `stop_name`: 중복 위험 매우 높음 (**Fact**: 총 5,705건 중 이름 중복 334건 확인. 절대 키나 조인 용도로 사용 불가)

### 5) 생성 SQL 초안
```sql
CREATE TABLE dim_stop (
    stop_id VARCHAR(50) PRIMARY KEY,
    stop_name VARCHAR(100) NOT NULL,
    latitude NUMERIC(10, 7),
    longitude NUMERIC(10, 7),
    geom GEOMETRY(Point, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO dim_stop (stop_id, stop_name, latitude, longitude, geom)
SELECT 
    bs_id AS stop_id,
    bs_nm AS stop_name,
    ngis_y_pos AS latitude,   -- (Assumption: 필요 시 위경도 변환 로직 추가)
    ngis_x_pos AS longitude,  -- (Assumption: 필요 시 위경도 변환 로직 추가)
    geometry AS geom          -- (Assumption: 대상 구조의 지오메트리 SRID가 4326에 준하게 될 것으로 가정)
FROM bs_20250903;
```

### 6) 검증 SQL 초안
```sql
-- 목적: dim_stop 대상 필수 키 무결성 검증, 이름 중복 경고, 공간 속성 누락 확인
SELECT 'stop_id_null' AS check_type, COUNT(*) AS cnt FROM dim_stop WHERE stop_id IS NULL
UNION ALL
SELECT 'stop_id_duplicate' AS check_type, COUNT(*) AS cnt FROM (SELECT stop_id FROM dim_stop GROUP BY stop_id HAVING COUNT(*) > 1) d
UNION ALL
-- (Fact: bs_nm 중복은 334건 알려져 있으므로 심각한 오류가 아닌 경고용 지표 관리)
SELECT 'stop_name_duplicate_warning' AS check_type, COUNT(*) AS cnt FROM (SELECT stop_name FROM dim_stop GROUP BY stop_name HAVING COUNT(*) > 1) d
UNION ALL
SELECT 'geom_invalid' AS check_type, COUNT(*) AS cnt FROM dim_stop WHERE geom IS NULL OR NOT ST_IsValid(geom);
```

---

## 2. dim_route 승격 계획

`stg_daegu_routes` 원천 데이터를 기반으로 한 노선 마스터 테이블 승격 계획입니다.

### 1) 제안 컬럼
- `route_id`: 노선 고유 식별자 (원본 `route_id`)
- `route_no`: 노선 번호 (원본 `route_no`)
- `route_type`: 노선 유형 (원본 `route_type`)
- `route_dir`: 운행 방향 (원본 `route_dir`)
- `origin_stop_id`: 기점 정류장 ID (원본 `origin_stop_id`)
- `origin_stop_name`: 기점 정류장 이름 (원본 `origin_stop_name`)
- `dest_stop_id`: 종점 정류장 ID (원본 `dest_stop_id`)
- `dest_stop_name`: 종점 정류장 이름 (원본 `dest_stop_name`)

### 2) PK (Primary Key)
- `route_id`
  - **Assumption**: `stg_daegu_routes` 내에서 단일 가상/고정 노선 (상/하행, 본선/지선 포함)을 유일하게 식별할 수 있는 수준의 ID일 것으로 추론 및 권장합니다.

### 3) Null 위험
- `route_id`, `route_no`, `route_type`: 핵심 식별/속성 정보이므로 Null 가능성 낮음 (**Assumption**)
- `route_dir`, `origin_stop_id`, `dest_stop_id`: 순환선이거나 원점 회귀노선의 경우 기종점 정보가 없거나 불분명할 수 있어 다소간의 Null 위험 존재 (**Assumption**)

### 4) Duplicate 위험
- `route_id`: 설계상 고유 식별자여야 하므로 중복 위험 낮음 (**Assumption**)
- `route_no`: 노선 번호는 같지만 상/하행에 따라 분리되거나 특수 지선인 경우 등 동일한 `route_no`를 쓰는 여러 논리적 노선이 존재할 수 있어 중복 위험 큼 (**Assumption**)

### 5) 생성 SQL 초안
```sql
CREATE TABLE dim_route (
    route_id VARCHAR(50) PRIMARY KEY,
    route_no VARCHAR(50) NOT NULL,
    route_type VARCHAR(50),
    route_dir VARCHAR(20),
    origin_stop_id VARCHAR(50),
    origin_stop_name VARCHAR(100),
    dest_stop_id VARCHAR(50),
    dest_stop_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO dim_route (
    route_id, route_no, route_type, route_dir, 
    origin_stop_id, origin_stop_name, 
    dest_stop_id, dest_stop_name
)
SELECT 
    route_id, 
    route_no, 
    route_type, 
    route_dir, 
    origin_stop_id, 
    origin_stop_name, 
    dest_stop_id, 
    dest_stop_name
FROM stg_daegu_routes;
```

### 6) 검증 SQL 초안
```sql
-- 목적: stg_daegu_routes 원천 데이터의 route_id 고유성(PK 무결성) 사전 검증
SELECT 'stg_route_id_duplicate_precheck' AS check_type, COUNT(*) AS cnt 
FROM (SELECT route_id FROM stg_daegu_routes GROUP BY route_id HAVING COUNT(*) > 1) d;

-- 목적: dim_route 식별자 무결성(PK), 노선 방면 속성 이상 탐지, 기/종점 정류장 외래키(orphan) 여부 점검
SELECT 'route_id_null' AS check_type, COUNT(*) AS cnt FROM dim_route WHERE route_id IS NULL
UNION ALL
SELECT 'route_id_duplicate' AS check_type, COUNT(*) AS cnt FROM (SELECT route_id FROM dim_route GROUP BY route_id HAVING COUNT(*) > 1) d
UNION ALL
SELECT 'route_no_or_dir_missing' AS check_type, COUNT(*) AS cnt FROM dim_route WHERE route_no IS NULL OR route_dir IS NULL
UNION ALL
-- (Assumption: 기종점 정류장이 dim_stop에 잘 연계되는지 확인. ID값이 자체적으로 NULL인 경우는 고아(Orphan)로 집계하지 않음)
SELECT 'origin_stop_orphan' AS check_type, COUNT(*) AS cnt 
FROM dim_route r LEFT JOIN dim_stop s ON r.origin_stop_id = s.stop_id 
WHERE r.origin_stop_id IS NOT NULL AND s.stop_id IS NULL
UNION ALL
SELECT 'dest_stop_orphan' AS check_type, COUNT(*) AS cnt 
FROM dim_route r LEFT JOIN dim_stop s ON r.dest_stop_id = s.stop_id 
WHERE r.dest_stop_id IS NOT NULL AND s.stop_id IS NULL;
```
