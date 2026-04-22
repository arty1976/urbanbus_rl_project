-- ==============================================================================
-- 프로젝트: 대중버스 동적 운영(Dynamic Operation) 연구 기반 데이터 파이프라인
-- 파일 설명: 교통카드 데이터 파이프라인 기초 셋팅을 위한 PostgreSQL 스키마 초안
-- ==============================================================================

-- 1. Dimension Tables

-- 정류점(Stop) 디멘전
-- 고정 노선 정류장뿐만 아니라 동적 셔틀의 가상 정류장(Virtual Stop) 정보도 포함 가능.
CREATE TABLE dim_stop (
    stop_id VARCHAR(50) PRIMARY KEY,
    ext_station_id VARCHAR(50),      -- 외부 시스템의 대체 식별자 (Alternate Key)
    stop_name VARCHAR(100) NOT NULL, -- 조인 금지 컬럼 (화면 표시용)
    latitude NUMERIC(10, 7) NOT NULL,
    longitude NUMERIC(10, 7) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 노선(Route) 디멘전
-- 고정(Fixed) 및 동적(Dynamic) 운영 노선을 모두 포괄.
CREATE TABLE dim_route (
    route_id VARCHAR(50) PRIMARY KEY,
    route_name VARCHAR(100) NOT NULL,
    operation_type VARCHAR(20) NOT NULL, -- 예: 'FIXED', 'DYNAMIC_ON_DEMAND' 등
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- 2. Bridge & Spatial Tables

-- 노선-정류장 브릿지 (Bridge Route Stop)
-- 동적 운영 환경에서는 특정 노선의 정차 정류장이 가변적일 수 있으므로 다대다(N:M) 브릿지로 구성.
CREATE TABLE bridge_route_stop (
    route_id VARCHAR(50) NOT NULL REFERENCES dim_route(route_id),
    stop_id VARCHAR(50) NOT NULL REFERENCES dim_stop(stop_id),
    stop_sequence INTEGER NOT NULL,
    is_dynamic_waypoint BOOLEAN DEFAULT FALSE, -- 동적으로 추가/변경된 경유지 여부
    valid_from TIMESTAMP NOT NULL,
    valid_to TIMESTAMP,
    PRIMARY KEY (route_id, stop_id, stop_sequence, valid_from)
);

-- 노선 링크 공간 정보 (GIS Route Link)
-- 라우팅 및 시뮬레이션을 위한 정류장 간 물리 이동 링크 (PostGIS 활용 전제).
CREATE TABLE gis_route_link (
    link_id VARCHAR(50) PRIMARY KEY,
    route_id VARCHAR(50) NOT NULL REFERENCES dim_route(route_id),
    from_stop_id VARCHAR(50) NOT NULL REFERENCES dim_stop(stop_id),
    to_stop_id VARCHAR(50) NOT NULL REFERENCES dim_stop(stop_id),
    distance_meters NUMERIC(10, 2),
    geom GEOMETRY(LINESTRING, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- 3. Fact Tables

-- 정류장 시간당 이용량 팩트 테이블 (Fact Stop Usage Hourly)
-- (시간적 해상도: 1시간 단위 / 공간적 해상도: 정류장)
CREATE TABLE fact_stop_usage_hourly (
    service_date DATE NOT NULL,
    time_bucket INTEGER NOT NULL CHECK (time_bucket >= 0 AND time_bucket <= 23),
    stop_id VARCHAR(50) NOT NULL REFERENCES dim_stop(stop_id),
    board_count INTEGER,     -- 승차량 (결측치는 NULL)
    alight_count INTEGER,    -- 하차량
    transfer_count INTEGER,  -- 환승량
    PRIMARY KEY (service_date, time_bucket, stop_id)
);

-- OD 일별 통행량 팩트 테이블 (Fact OD Daily)
-- (시간적 해상도: 1일 단위 / 공간적 해상도: 출발지-도착지 정류장 쌍)
CREATE TABLE fact_od_daily (
    service_date DATE NOT NULL,
    origin_stop_id VARCHAR(50) NOT NULL REFERENCES dim_stop(stop_id),
    destination_stop_id VARCHAR(50) NOT NULL REFERENCES dim_stop(stop_id),
    passenger_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (service_date, origin_stop_id, destination_stop_id)
);
