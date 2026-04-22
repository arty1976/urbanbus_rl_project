-- ============================================================
-- create_stg_daegu_stop_usage_2025_monthly.sql
-- 목적: 2025년 정류소별 시간대별 승하차인원 로우 데이터 적재 (Monthly)
-- 성격: 보조 프로파일 계층용
-- ============================================================

DROP TABLE IF EXISTS public.stg_daegu_stop_usage_2025_monthly CASCADE;

CREATE TABLE public.stg_daegu_stop_usage_2025_monthly (
    year_month      VARCHAR(6),      -- 년월 (YYYYMM)
    stop_nm         VARCHAR(200),
    stop_id_raw     VARCHAR(50),
    category        VARCHAR(50),     -- 구분 (승차/하차)
    h05             INTEGER,
    h06             INTEGER,
    h07             INTEGER,
    h08             INTEGER,
    h09             INTEGER,
    h10             INTEGER,
    h11             INTEGER,
    h12             INTEGER,
    h13             INTEGER,
    h14             INTEGER,
    h15             INTEGER,
    h16             INTEGER,
    h17             INTEGER,
    h18             INTEGER,
    h19             INTEGER,
    h20             INTEGER,
    h21             INTEGER,
    h22             INTEGER,
    h23             INTEGER,
    batch_id        VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stg_daegu_stop_usage_2025_month ON public.stg_daegu_stop_usage_2025_monthly (year_month);

COMMENT ON TABLE public.stg_daegu_stop_usage_2025_monthly IS '2025년 대구 버스 정류소별 시간대별 이용량 스테이징 레벨 (월별)';
