-- ============================================================
-- promote_fact_stop_usage_hourly_from_2023_file.sql
-- 목적: 2023년 스테이징 데이터를 가공하여 분석용 팩트 테이블로 승격
-- 로직: Unpivoting + Multi-tier Mapping + Exception Logging
-- ============================================================

-- [STEP 0] 배치 식별용 변수 설정 (실행 시 외부에서 부여하거나 최신 batch_id 사용)
-- 여기서는 가장 최근의 batch_id를 대상으로 함

DROP TABLE IF EXISTS tmp_classified_2023;
CREATE TEMP TABLE tmp_classified_2023 AS
WITH latest_batch AS (
    SELECT MAX(batch_id) as bid FROM public.stg_daegu_stop_usage_2023
),
-- [STEP 1] Unpivoting (Wide to Long)
unpivoted AS (
    SELECT 
        CASE 
            WHEN TRIM(s.service_date_raw) ~ '^[0-9]{8}$' THEN TO_DATE(TRIM(s.service_date_raw), 'YYYYMMDD')
            ELSE CAST(NULLIF(TRIM(s.service_date_raw), '') AS DATE)
        END AS svc_date,
        TRIM(s.stop_name_raw) AS stop_nm,
        TRIM(s.stop_id_raw) AS stop_id_raw,
        TRIM(s.mobile_id_raw) AS mobile_id_raw,
        TRIM(s.usage_type_raw) AS category,
        CAST(NULLIF(REGEXP_REPLACE(TRIM(s.row_sum_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER) AS row_sum,
        s.batch_id,
        h.hour_num,
        h.amount
    FROM public.stg_daegu_stop_usage_2023 s
    CROSS JOIN LATERAL (
        VALUES 
            (5, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h05_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (6, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h06_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (7, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h07_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (8, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h08_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (9, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h09_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (10, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h10_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (11, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h11_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (12, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h12_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (13, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h13_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (14, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h14_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (15, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h15_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (16, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h16_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (17, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h17_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (18, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h18_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (19, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h19_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (20, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h20_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (21, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h21_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (22, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h22_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER)),
            (23, CAST(NULLIF(REGEXP_REPLACE(TRIM(s.h23_raw), '[^0-9\-]', '', 'g'), '') AS INTEGER))
    ) AS h(hour_num, amount)
    WHERE s.batch_id = (SELECT bid FROM latest_batch)
),
-- [STEP 2] Multi-tier Mapping Logic
mapped AS (
    SELECT 
        u.*,
        ds_direct.stop_id AS stop_id_direct,
        ds_fallback.stop_id AS stop_id_fallback,
        CASE 
            WHEN ds_direct.stop_id IS NOT NULL THEN 'direct'
            WHEN ds_fallback.stop_id IS NOT NULL THEN 'fallback'
            ELSE 'unmatched'
        END AS mapping_method,
        CASE 
            WHEN ds_direct.stop_id IS NOT NULL THEN ds_direct.stop_id
            WHEN ds_fallback.stop_id IS NOT NULL THEN ds_fallback.stop_id
            ELSE NULL
        END AS final_stop_id
    FROM unpivoted u
    LEFT JOIN public.dim_stop ds_direct ON u.stop_id_raw = ds_direct.stop_id
    LEFT JOIN public.dim_stop ds_fallback ON u.mobile_id_raw = ds_fallback.stop_id
),
-- [STEP 3] Valid/Invalid Classification
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

-- [STEP 4] Promotion to Fact
INSERT INTO public.fact_stop_usage_hourly (
    service_date,
    service_hour,
    stop_id,
    boardings,
    alightings,
    source_system,
    data_tier,
    is_observed,
    source_ref
)
SELECT 
    svc_date,
    hour_num,
    final_stop_id,
    SUM(CASE WHEN category = '승차' THEN amount ELSE 0 END) AS boardings,
    SUM(CASE WHEN category = '하차' THEN amount ELSE 0 END) AS alightings,
    'daegu_traffic_card_file_2023' AS source_system,
    'observed' AS data_tier,
    TRUE AS is_observed,
    'batch_id:' || batch_id AS source_ref
FROM tmp_classified_2023
WHERE processing_status = 'valid'
GROUP BY svc_date, hour_num, final_stop_id, batch_id
ON CONFLICT (service_date, service_hour, stop_id, source_system) 
DO UPDATE SET 
    boardings = EXCLUDED.boardings,
    alightings = EXCLUDED.alightings,
    source_ref = EXCLUDED.source_ref,
    created_at = CURRENT_TIMESTAMP;

-- [STEP 5] Log Exceptions
INSERT INTO public.err_daegu_stop_usage_mapping_failed (
    source_table,
    svc_date_or_month,
    stop_nm,
    stop_id_raw,
    mobile_id_raw,
    category,
    mapping_method,
    error_reason,
    batch_id
)
SELECT DISTINCT
    'stg_daegu_stop_usage_2023',
    svc_date::TEXT,
    stop_nm,
    stop_id_raw,
    mobile_id_raw,
    category,
    mapping_method,
    processing_status,
    batch_id
FROM tmp_classified_2023
WHERE processing_status != 'valid';

ANALYZE public.fact_stop_usage_hourly;
DROP TABLE IF EXISTS tmp_classified_2023;
