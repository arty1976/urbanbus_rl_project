-- ============================================================
-- create_fact_stop_usage_hourly_profile_2025.sql
-- 목적: 2025년 월별 데이터를 기반으로 한 정류소 시간대별 이용량 프로파일
-- ============================================================

DROP TABLE IF EXISTS public.fact_stop_usage_hourly_profile_2025 CASCADE;

CREATE TABLE public.fact_stop_usage_hourly_profile_2025 (
    year_month      VARCHAR(6) NOT NULL,
    service_hour    INTEGER NOT NULL CHECK (service_hour >= 0 AND service_hour <= 23),
    stop_id         VARCHAR(50) NOT NULL,
    boardings       INTEGER DEFAULT 0,
    alightings      INTEGER DEFAULT 0,
    source_system   VARCHAR(100) DEFAULT 'daegu_traffic_card_file_2025_monthly',
    batch_id        VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (year_month, service_hour, stop_id)
);

COMMENT ON TABLE public.fact_stop_usage_hourly_profile_2025 IS '2025년 대구 버스 정류소 시간대별 월평균(또는 월합계)이용량 프로파일';
