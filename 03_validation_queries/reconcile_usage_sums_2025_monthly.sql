-- ============================================================
-- reconcile_usage_sums_2025_monthly.sql
-- 목적: 2025년 월별 데이터의 프로파일 승격 정합성 검증
-- ============================================================

WITH latest_batch AS (
    SELECT MAX(batch_id) as bid FROM public.stg_daegu_stop_usage_2025_monthly
),
-- [1] 원천 행 수 및 기초 합계 (Staging)
stg_stats AS (
    SELECT 
        COUNT(*) AS total_stg_rows,
        SUM(h05+h06+h07+h08+h09+h10+h11+h12+h13+h14+h15+h16+h17+h18+h19+h20+h21+h22+h23) AS total_stg_sum
    FROM public.stg_daegu_stop_usage_2025_monthly
    WHERE batch_id = (SELECT bid FROM latest_batch)
),
-- [2] Fact 합계
fact_stats AS (
    SELECT 
        COUNT(*) AS total_fact_rows,
        SUM(boardings + alightings) AS total_fact_sum,
        COUNT(DISTINCT stop_id) AS distinct_stops
    FROM public.fact_stop_usage_hourly_profile_2025
    WHERE batch_id = (SELECT bid FROM latest_batch)
)
SELECT 
    (SELECT bid FROM latest_batch) AS batch_id,
    ss.total_stg_rows,
    fs.total_fact_rows AS promoted_fact_rows,
    ss.total_stg_sum AS staging_hourly_sum,
    fs.total_fact_sum AS promoted_fact_sum,
    CASE 
        WHEN ss.total_stg_sum = fs.total_fact_sum THEN 'PASS'
        ELSE 'FAIL (Sum Mismatch)'
    END AS sum_reconciliation,
    ROUND((fs.total_fact_sum::NUMERIC / NULLIF(ss.total_stg_sum, 0)) * 100, 2) AS promotion_success_rate_percent
FROM stg_stats ss, fact_stats fs;
