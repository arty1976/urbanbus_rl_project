-- ============================================================
-- fact_stop_usage_hourly_from_synth_trip_readiness.sql
-- 목적: Fact(Hourly) 데이터의 집계 정합성 및 매핑 품질 검증
-- ============================================================

-- 1. 기본 통계
SELECT 
    '[METRIC] Total Fact Rows' as metric,
    COUNT(*)::TEXT as value
FROM public.fact_stop_usage_hourly
WHERE source_system = 'daegu_transport_card_synth_api'

UNION ALL

SELECT 
    '[METRIC] Total Boardings' as metric,
    SUM(boardings)::TEXT
FROM public.fact_stop_usage_hourly
WHERE source_system = 'daegu_transport_card_synth_api'

UNION ALL

-- 2. Staging vs Fact 동기화 검증 (UTZTN_NOPE 합계 비교)
-- Fact에는 dim_stop 매핑된 건만 들어가므로, 
-- (Staging 총합 - Fact 총합) = 매핑 실패로 인한 누락분이어야 함.
SELECT 
    '[VAL] Staging UTZTN_NOPE vs Fact Boardings Diff' as metric,
    ((SELECT SUM(utztn_nope) FROM public.stg_daegu_transport_card_usage_synth_trip) - 
     (SELECT SUM(boardings) FROM public.fact_stop_usage_hourly WHERE source_system = 'daegu_transport_card_synth_api'))::TEXT

UNION ALL

-- 3. 매핑 실패 리포트
SELECT 
    '[VAL] Unmapped Stop IDs Count' as metric,
    COUNT(DISTINCT raw_stop_id)::TEXT
FROM (
    SELECT ride_sttn_id as raw_stop_id FROM public.stg_daegu_transport_card_usage_synth_trip
    UNION
    SELECT goff_sttn_id FROM public.stg_daegu_transport_card_usage_synth_trip
) t
WHERE NOT EXISTS (SELECT 1 FROM public.dim_stop s WHERE s.stop_id = t.raw_stop_id);

-- 4. 미매핑 상세 목록 (사용자가 별도 쿼리 실행 시 참고)
/*
SELECT DISTINCT t.raw_stop_id, COUNT(*) as trip_count
FROM (
    SELECT ride_sttn_id as raw_stop_id FROM public.stg_daegu_transport_card_usage_synth_trip
    UNION ALL
    SELECT goff_sttn_id FROM public.stg_daegu_transport_card_usage_synth_trip
) t
WHERE NOT EXISTS (SELECT 1 FROM public.dim_stop s WHERE s.stop_id = t.raw_stop_id)
GROUP BY 1 ORDER BY 2 DESC;
*/
