-- ============================================================
-- graph_state_timeslice_readiness.sql
-- 목적: 통합 그래프 상태 테이블 적재 정합성 및 가용성 검증
-- 작성일: 2026-04-10
-- 수정일: 2026-04-14 (복구 확인용, ZeroDivide 방어 처리, 2023 전체 1시간 버킷 타겟)
-- ============================================================

-- 1. 전체 건수 및 노드 유형별 분포
SELECT 
    node_type,
    count(*) as row_cnt,
    count(distinct state_ts) as distinct_time_buckets,
    min(state_ts) as min_ts,
    max(state_ts) as max_ts
FROM public.graph_state_timeslice
GROUP BY node_type;

-- 2. 시간축 밀도 (일별 버킷 확인)
WITH date_buckets AS (
    SELECT 
        state_ts::date as service_date,
        count(distinct state_ts) as buckets_per_day
    FROM public.graph_state_timeslice
    GROUP BY state_ts::date
)
SELECT 
    buckets_per_day,
    count(*) as active_days
FROM date_buckets
GROUP BY buckets_per_day;

-- 3. STOP 노드 핵심 지표 결측치/0 비율 점검
SELECT 
    count(*) as total_stop_rows,
    count(*) filter (where boardings_recent is null) as null_boardings,
    count(*) filter (where boardings_recent = 0) as zero_boardings,
    CASE 
        WHEN count(*) = 0 THEN 0.00
        ELSE round(count(*) filter (where boardings_recent = 0)::numeric / count(*) * 100, 2)
    END as zero_rate_pct
FROM public.graph_state_timeslice
WHERE node_type = 'STOP';

-- 4. 고아 노드 점검 (Node Master에 없는 UID)
SELECT 
    count(*) as orphan_cnt
FROM public.graph_state_timeslice s
LEFT JOIN public.graph_node_master n ON s.node_uid = n.node_uid
WHERE n.node_uid IS NULL;

-- 5. 수치 범위 이상치 점검 (음수 등)
SELECT 
    'negative_boardings' as check_type, count(*) as cnt from public.graph_state_timeslice where boardings_recent < 0
UNION ALL
SELECT 
    'invalid_hour' as check_type, count(*) as cnt from public.graph_state_timeslice where hour_of_day < 0 or hour_of_day > 23;

-- 6. 전체 종합 판정 (결과 요약)
SELECT
    CASE
        WHEN count(*) = 0 THEN 'BLOCKED (No Rows)'
        WHEN count(*) filter (where boardings_recent is null) > 0 THEN 'HOLD (Null Boardings Found)'
        WHEN count(distinct state_ts) < 24 THEN 'HOLD (Low Bucket count)'
        ELSE 'APPROVED'
    END as final_decision,
    CASE WHEN count(*) > 0 THEN 'YES' ELSE 'NO' END as gat_rl_input_ready
FROM public.graph_state_timeslice
WHERE node_type = 'STOP';
