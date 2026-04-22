-- ============================================================
-- load_fact_stop_usage_hourly_from_synth_trip.sql
-- 목적: Staging(Trip-level) 데이터를 시각별 정류장 이용량(Fact)으로 집계 및 적재
-- ============================================================

WITH ride_agg AS (
    -- 1. 승차 데이터 집계 (RIDE_DT 기준)
    SELECT 
        TO_DATE(LEFT(ride_dt, 8), 'YYYYMMDD') as service_date,
        CAST(SUBSTRING(ride_dt, 9, 2) AS INTEGER) as service_hour,
        ride_sttn_id as raw_stop_id,
        SUM(utztn_nope) as boardings
    FROM public.stg_daegu_transport_card_usage_synth_trip
    GROUP BY 1, 2, 3
),
alight_agg AS (
    -- 2. 하차 데이터 집계 (GOFF_DT 기준)
    SELECT 
        TO_DATE(LEFT(goff_dt, 8), 'YYYYMMDD') as service_date,
        CAST(SUBSTRING(goff_dt, 9, 2) AS INTEGER) as service_hour,
        goff_sttn_id as raw_stop_id,
        SUM(utztn_nope) as alightings
    FROM public.stg_daegu_transport_card_usage_synth_trip
    GROUP BY 1, 2, 3
),
combined_agg AS (
    -- 3. 승하차 데이터 결합
    SELECT 
        COALESCE(r.service_date, a.service_date) as service_date,
        COALESCE(r.service_hour, a.service_hour) as service_hour,
        COALESCE(r.raw_stop_id, a.raw_stop_id) as raw_stop_id,
        COALESCE(r.boardings, 0) as boardings,
        COALESCE(a.alightings, 0) as alightings
    FROM ride_agg r
    FULL OUTER JOIN alight_agg a 
        ON r.service_date = a.service_date 
        AND r.service_hour = a.service_hour 
        AND r.raw_stop_id = a.raw_stop_id
)
-- 4. Fact 테이블 적재 (Mapping 수행)
INSERT INTO public.fact_stop_usage_hourly (
    service_date,
    service_hour,
    stop_id,
    boardings,
    alightings,
    source_system,
    data_tier,
    is_observed,
    source_ref,
    created_at
)
SELECT 
    c.service_date,
    c.service_hour,
    s.stop_id, -- dim_stop과 매핑된 ID (매핑 실패 시 NULL 대응 로직 필요시 추가)
    c.boardings,
    c.alightings,
    'daegu_transport_card_synth_api' as source_system,
    'synthetic' as data_tier,
    false as is_observed,
    'batch_aggregated' as source_ref,
    CURRENT_TIMESTAMP
FROM combined_agg c
INNER JOIN public.dim_stop s ON c.raw_stop_id = s.stop_id -- 직접 매핑 시도
ON CONFLICT (service_date, service_hour, stop_id, source_system) 
DO UPDATE SET 
    boardings = EXCLUDED.boardings,
    alightings = EXCLUDED.alightings,
    source_ref = EXCLUDED.source_ref,
    created_at = CURRENT_TIMESTAMP;

-- 매핑 실패 건수 확인용 로그 (출력은 안되지만 쿼리 실행 후 확인 가능)
-- SELECT raw_stop_id FROM combined_agg WHERE raw_stop_id NOT IN (SELECT stop_id FROM dim_stop);
