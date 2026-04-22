-- ============================================================
-- create_stg_daegu_stop_usage_2023.sql
-- 목적: 2023년 정류소별 시간대별 승하차인원 로우 데이터 적재 (Daily)
-- 원천 파일: 정류소별시간대별승하차인원조회_20230101_20231231.csv
-- ============================================================

DROP TABLE IF EXISTS public.stg_daegu_stop_usage_2023 CASCADE;

CREATE TABLE public.stg_daegu_stop_usage_2023 (
    service_date_raw  TEXT,
    stop_name_raw     TEXT,
    stop_id_raw       TEXT,
    mobile_id_raw     TEXT,
    admin_area_raw    TEXT,
    usage_type_raw    TEXT,
    h05_raw           TEXT,
    h06_raw           TEXT,
    h07_raw           TEXT,
    h08_raw           TEXT,
    h09_raw           TEXT,
    h10_raw           TEXT,
    h11_raw           TEXT,
    h12_raw           TEXT,
    h13_raw           TEXT,
    h14_raw           TEXT,
    h15_raw           TEXT,
    h16_raw           TEXT,
    h17_raw           TEXT,
    h18_raw           TEXT,
    h19_raw           TEXT,
    h20_raw           TEXT,
    h21_raw           TEXT,
    h22_raw           TEXT,
    h23_raw           TEXT,
    row_sum_raw       TEXT,
    batch_id          TEXT,
    source_file       TEXT,
    loaded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stg_daegu_stop_usage_2023_date ON public.stg_daegu_stop_usage_2023 (service_date_raw);
CREATE INDEX IF NOT EXISTS idx_stg_daegu_stop_usage_2023_stop_id ON public.stg_daegu_stop_usage_2023 (stop_id_raw);

COMMENT ON TABLE public.stg_daegu_stop_usage_2023 IS '2023년 대구 버스 정류소별 시간대별 이용량 스테이징 레벨 (Raw TEXT)';
