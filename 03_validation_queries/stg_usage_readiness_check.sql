-- ============================================================
-- stg_usage_readiness_check.sql
-- 목적: 적재된 staging 데이터의 기초 무결성 검증 (Promotion 전 실행)
-- ============================================================

WITH latest_batch_2023 AS (
    SELECT MAX(batch_id) as bid FROM public.stg_daegu_stop_usage_2023
),
latest_batch_2025 AS (
    SELECT MAX(batch_id) as bid FROM public.stg_daegu_stop_usage_2025_monthly
)
-- [1] 2023 데이터 기초 검증
    'stg_2023' AS layer,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE service_date_raw IS NULL OR TRIM(service_date_raw) = '') AS null_dates,
    COUNT(*) FILTER (WHERE stop_id_raw IS NULL OR TRIM(stop_id_raw) = '') AS null_stop_ids,
    COUNT(*) FILTER (WHERE TRIM(usage_type_raw) NOT IN ('승차', '하차')) AS invalid_categories,
    SUM(CAST(NULLIF(REGEXP_REPLACE(TRIM(row_sum_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT)) AS total_sum_check
FROM public.stg_daegu_stop_usage_2023
WHERE batch_id = (SELECT bid FROM latest_batch_2023)

UNION ALL

-- [2] 2025 데이터 기초 검증
SELECT 
    'stg_2025_monthly' AS layer,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE year_month IS NULL) AS null_dates,
    COUNT(*) FILTER (WHERE stop_id_raw IS NULL) AS null_stop_ids,
    COUNT(*) FILTER (WHERE category NOT IN ('승차', '하차')) AS invalid_categories,
    NULL AS total_sum_check -- 2025는 합계 컬럼이 없음
FROM public.stg_daegu_stop_usage_2025_monthly
WHERE batch_id = (SELECT bid FROM latest_batch_2025);
