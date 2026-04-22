-- ============================================================
-- create_fact_stop_usage_hourly.sql
-- 목적: 정류장별 시간대별 이용량 표준 팩트 테이블 (Source-Agnostic)
-- ============================================================

DROP TABLE IF EXISTS public.fact_stop_usage_hourly CASCADE;

CREATE TABLE public.fact_stop_usage_hourly (
    service_date    DATE NOT NULL,
    service_hour    INTEGER NOT NULL CHECK (service_hour >= 0 AND service_hour <= 23),
    stop_id         VARCHAR(50) NOT NULL,
    boardings       INTEGER DEFAULT 0,  -- 승차량
    alightings      INTEGER DEFAULT 0,  -- 하차량
    source_system   VARCHAR(100) NOT NULL, -- 예: 'daegu_transport_card_synth_api'
    data_tier       VARCHAR(20) NOT NULL,  -- 'synthetic', 'observed'
    is_observed     BOOLEAN NOT NULL,      -- 실측 여부
    source_ref      VARCHAR(255),          -- 원천 참조 (batch_id 등)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (service_date, service_hour, stop_id, source_system) -- 원천별 병행 적재 가능하도록 구성
);

COMMENT ON TABLE public.fact_stop_usage_hourly IS '정류장 시간당 이용량 표준 팩트 테이블. (시간적 해상도: 1시간 / 공간적 해상도: 정류장)';
