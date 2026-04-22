-- ============================================================
-- create_err_daegu_stop_usage_mapping_failed.sql
-- 목적: dim_stop 매핑에 실패했거나 '승차/하차' 외 부적합 데이터 기록
-- ============================================================

DROP TABLE IF EXISTS public.err_daegu_stop_usage_mapping_failed CASCADE;

CREATE TABLE public.err_daegu_stop_usage_mapping_failed (
    source_table    VARCHAR(100),   -- 원천 테이블 (stg_daegu_stop_usage_2023 등)
    svc_date_or_month VARCHAR(10),  -- 날짜 또는 년월
    stop_nm         VARCHAR(200),
    stop_id_raw     VARCHAR(50),
    mobile_id_raw   VARCHAR(50),
    category        VARCHAR(50),
    mapping_method  VARCHAR(20),    -- 'unmatched'
    error_reason    VARCHAR(200),   -- 'stop_not_found', 'invalid_category' 등
    batch_id        VARCHAR(50),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_err_usage_mapping_failed_batch ON public.err_daegu_stop_usage_mapping_failed (batch_id);

COMMENT ON TABLE public.err_daegu_stop_usage_mapping_failed IS '정류소 매핑 실패 또는 데이터 유효성 위반 로그';
