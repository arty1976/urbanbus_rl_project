-- ============================================================
-- promote_fact_stop_usage_profile_2025.sql
-- 목적: 2025년 월별 스테이징 데이터를 가공하여 프로파일 테이블로 승격
-- ============================================================

DROP TABLE IF EXISTS tmp_classified_2025;
CREATE TEMP TABLE tmp_classified_2025 AS
WITH latest_batch AS (
    SELECT MAX(batch_id) as bid FROM public.stg_daegu_stop_usage_2025_monthly
),
-- [STEP 1] Unpivoting
unpivoted AS (
    SELECT 
        s.year_month,
        s.stop_nm,
        s.stop_id_raw,
        s.category,
        s.batch_id,
        h.hour_num,
        h.amount
    FROM public.stg_daegu_stop_usage_2025_monthly s
    CROSS JOIN LATERAL (
        VALUES 
            (5, s.h05), (6, s.h06), (7, s.h07), (8, s.h08), (9, s.h09),
            (10, s.h10), (11, s.h11), (12, s.h12), (13, s.h13), (14, s.h14),
            (15, s.h15), (16, s.h16), (17, s.h17), (18, s.h18), (19, s.h19),
            (20, s.h20), (21, s.h21), (22, s.h22), (23, s.h23)
    ) AS h(hour_num, amount)
    WHERE s.batch_id = (SELECT bid FROM latest_batch)
),
-- [STEP 2] Direct Mapping Only (based on 2025 schema)
mapped AS (
    SELECT 
        u.*,
        ds.stop_id AS final_stop_id,
        CASE 
            WHEN ds.stop_id IS NOT NULL THEN 'direct'
            ELSE 'unmatched'
        END AS mapping_method
    FROM unpivoted u
    LEFT JOIN public.dim_stop ds ON u.stop_id_raw = ds.stop_id
),
-- [STEP 3] Classification
classified AS (
    SELECT 
        m.*,
        CASE 
            WHEN m.category NOT IN ('승차', '하차') THEN 'invalid_category'
            WHEN m.final_stop_id IS NULL THEN 'stop_not_found'
            ELSE 'valid'
        END AS processing_status
    FROM mapped m
)
SELECT * FROM classified;

-- [STEP 4] Promotion
INSERT INTO public.fact_stop_usage_hourly_profile_2025 (
    year_month,
    service_hour,
    stop_id,
    boardings,
    alightings,
    batch_id
)
SELECT 
    year_month,
    hour_num,
    final_stop_id,
    SUM(CASE WHEN category = '승차' THEN amount ELSE 0 END) AS boardings,
    SUM(CASE WHEN category = '하차' THEN amount ELSE 0 END) AS alightings,
    batch_id
FROM tmp_classified_2025
WHERE processing_status = 'valid'
GROUP BY year_month, hour_num, final_stop_id, batch_id
ON CONFLICT (year_month, service_hour, stop_id) 
DO UPDATE SET 
    boardings = EXCLUDED.boardings,
    alightings = EXCLUDED.alightings,
    batch_id = EXCLUDED.batch_id,
    created_at = CURRENT_TIMESTAMP;

-- [STEP 5] Log Exceptions
INSERT INTO public.err_daegu_stop_usage_mapping_failed (
    source_table,
    svc_date_or_month,
    stop_nm,
    stop_id_raw,
    category,
    mapping_method,
    error_reason,
    batch_id
)
SELECT DISTINCT
    'stg_daegu_stop_usage_2025_monthly',
    year_month,
    stop_nm,
    stop_id_raw,
    category,
    mapping_method,
    processing_status,
    batch_id
FROM tmp_classified_2025
WHERE processing_status != 'valid';

ANALYZE public.fact_stop_usage_hourly_profile_2025;
DROP TABLE IF EXISTS tmp_classified_2025;
