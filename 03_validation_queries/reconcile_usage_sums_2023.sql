-- ============================================================
-- reconcile_usage_sums_2023.sql
-- 목적: 2023년 데이터의 원천 합계 vs 가공 합계 정합성 검증
-- ============================================================

WITH latest_batch AS (
    SELECT MAX(batch_id) as bid FROM public.stg_daegu_stop_usage_2023
),
-- [1] 원천 합계 (stg.row_sum)
stg_totals AS (
    SELECT 
        SUM(CAST(NULLIF(REGEXP_REPLACE(TRIM(row_sum_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT)) AS total_stg_sum,
        COUNT(*) AS total_stg_rows
    FROM public.stg_daegu_stop_usage_2023
    WHERE batch_id = (SELECT bid FROM latest_batch)
),
-- [2] Unpivoted 합계 (h05~h23 합산)
h_totals AS (
    SELECT 
        SUM(
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h05_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h06_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h07_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h08_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h09_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h10_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h11_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h12_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h13_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h14_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h15_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h16_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h17_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h18_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h19_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h20_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h21_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h22_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT) +
            CAST(NULLIF(REGEXP_REPLACE(TRIM(h23_raw), '[^0-9\-]', '', 'g'), '') AS BIGINT)
        ) AS total_h_sum
    FROM public.stg_daegu_stop_usage_2023
    WHERE batch_id = (SELECT bid FROM latest_batch)
),
-- [3] Fact 합계 (승격된 결과)
fact_totals AS (
    SELECT 
        SUM(boardings + alightings) AS total_fact_sum,
        COUNT(DISTINCT stop_id) AS distinct_stops_in_fact
    FROM public.fact_stop_usage_hourly
    WHERE source_ref LIKE 'batch_id:' || (SELECT bid FROM latest_batch)
),
-- [4] Exception 합계 (실패한 결과)
err_totals AS (
    SELECT 
        COUNT(*) AS exception_rows
    FROM public.err_daegu_stop_usage_mapping_failed
    WHERE batch_id = (SELECT bid FROM latest_batch)
    AND source_table = 'stg_daegu_stop_usage_2023'
)
SELECT 
    (SELECT bid FROM latest_batch) AS batch_id,
    st.total_stg_rows,
    st.total_stg_sum AS stg_row_sum_column,
    ht.total_h_sum AS stg_hourly_columns_sum,
    ft.total_fact_sum AS promoted_fact_sum,
    et.exception_rows,
    CASE 
        WHEN st.total_stg_sum = ht.total_h_sum THEN 'PASS'
        ELSE 'FAIL (STG Check)'
    END AS stg_internal_reconciliation,
    CASE 
        WHEN COALESCE(ft.total_fact_sum, 0) <= st.total_stg_sum THEN 'PASS (Fact <= Source)'
        ELSE 'FAIL (Fact Overflow)'
    END AS promote_reconciliation,
    ROUND((ft.total_fact_sum::NUMERIC / NULLIF(st.total_stg_sum, 0)) * 100, 2) AS promotion_success_rate_percent
FROM stg_totals st, h_totals ht, fact_totals ft, err_totals et;
