-- ============================================================
-- check_fact_data_presence.sql
-- 목적: fact_stop_usage_hourly 테이블에 데이터가 적재되어 있는지 확인
-- ============================================================

SELECT count(*) as row_count 
FROM public.fact_stop_usage_hourly;
