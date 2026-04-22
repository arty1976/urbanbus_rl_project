-- ============================================================
-- daegu_transport_card_synth_trip_readiness.sql
-- 목적: Staging(Trip-level) 데이터의 적재 상태 및 기본 품질 검증
-- ============================================================

SELECT 
    '[METRIC] Total Staging Rows' as metric,
    COUNT(*)::TEXT as value
FROM public.stg_daegu_transport_card_usage_synth_trip

UNION ALL

SELECT 
    '[METRIC] Distinct Operation Days' as metric,
    COUNT(DISTINCT opr_ymd)::TEXT
FROM public.stg_daegu_transport_card_usage_synth_trip

UNION ALL

SELECT 
    '[METRIC] Interpolated Data Ratio (%)' as metric,
    ROUND(100.0 * COUNT(*) FILTER (WHERE msv_intrpl_yn = 'Y') / NULLIF(COUNT(*), 0), 2)::TEXT
FROM public.stg_daegu_transport_card_usage_synth_trip

UNION ALL

SELECT 
    '[METRIC] Total UTZTN_NOPE Sum' as metric,
    SUM(utztn_nope)::TEXT
FROM public.stg_daegu_transport_card_usage_synth_trip

UNION ALL

SELECT 
    '[METRIC] Invalid Ride/Goff Datetime Rows' as metric,
    COUNT(*)::TEXT
FROM public.stg_daegu_transport_card_usage_synth_trip
WHERE ride_dt IS NULL OR LENGTH(ride_dt) != 14 OR goff_dt IS NULL OR LENGTH(goff_dt) != 14;
