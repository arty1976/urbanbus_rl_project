-- ============================================================
-- load_graph_state_timeslice_initial.sql
-- 목적: 2023년 관측 전체 영업일 기준 시간단위 상태 데이터 고속 적재 (Full Reload)
-- 기준: 2100만 건 대량 적재를 위해 트랜잭션 내에서 인덱스 일시 해제 후 재적재 방식 적용 (선택지 B)
-- 작성일: 2026-04-10
-- 수정일: 2026-04-14 (대량 데이터 I/O 병목 해소, 트랜잭션 안전성 확보)
-- ============================================================
-- 주의사항: 
-- 1. 이 스크립트는 graph_state_timeslice 의 전량 재적재(full reload)용입니다.
--    운영 중 다른 node_type(예: LINK)이 추가되면 TRUNCATE 대신 범위 삭제 전략으로 재검토해야 합니다.
-- 2. waiting_passenger_cnt 는 실제 대기열(queue)이 아니라 승하차 차이를 이용한 프록시(proxy) 근사값입니다.
-- ============================================================

BEGIN;

-- 1. 누락된 STOP 노드를 dim_stop에서 graph_node_master로 보충
-- TRUNCATE 금지, is_active 제외, LEFT JOIN으로 누락분만 안전하게 추가
INSERT INTO public.graph_node_master (node_uid, node_type, node_id, node_name)
SELECT
    'STOP:' || d.stop_id,
    'STOP',
    d.stop_id,
    d.stop_name
FROM public.dim_stop d
LEFT JOIN public.graph_node_master g
  ON g.node_uid = 'STOP:' || d.stop_id
WHERE g.node_uid IS NULL;

-- 2. 고속 적재를 위한 테이블 초기화 및 인덱스/제약조건 일시 제거 (단일 트랜잭션 내부)
TRUNCATE TABLE public.graph_state_timeslice;

ALTER TABLE public.graph_state_timeslice DROP CONSTRAINT IF EXISTS graph_state_timeslice_pkey;
DROP INDEX IF EXISTS idx_graph_state_ts_type;
DROP INDEX IF EXISTS idx_graph_state_node_uid;

-- 3. 데이터 대량 삽입 (약 2160만 건)
-- hour_of_day, day_of_week 등 명시적 매핑 및 2023 Observed Only 필터 적용
INSERT INTO public.graph_state_timeslice (
    state_ts,
    node_uid,
    node_type,
    boardings_recent,
    alightings_recent,
    waiting_passenger_cnt,
    link_speed_kmh,
    hour_of_day,
    day_of_week,
    is_peak,
    created_at,
    updated_at
)
SELECT
    (f.service_date::timestamp + (f.service_hour * interval '1 hour')) AT TIME ZONE 'Asia/Seoul' AS state_ts,
    n.node_uid AS node_uid,
    'STOP' AS node_type,
    COALESCE(f.boardings, 0) AS boardings_recent,
    COALESCE(f.alightings, 0) AS alightings_recent,
    GREATEST(COALESCE(f.boardings, 0) - COALESCE(f.alightings, 0), 0) AS waiting_passenger_cnt,
    NULL AS link_speed_kmh,
    f.service_hour AS hour_of_day,
    EXTRACT(DOW FROM f.service_date) AS day_of_week,
    CASE WHEN f.service_hour IN (7, 8, 17, 18) THEN TRUE ELSE FALSE END AS is_peak,
    NOW() AS created_at,
    NOW() AS updated_at
FROM public.fact_stop_usage_hourly f
INNER JOIN public.graph_node_master n
  ON n.node_id = f.stop_id AND n.node_type = 'STOP'
WHERE f.service_date >= DATE '2023-01-01'
  AND f.service_date < DATE '2024-01-01'
  AND f.is_observed = true
  AND f.data_tier = 'observed';

-- 4. 인덱스 및 제약조건 재생성 (단일 트랜잭션 내부)
-- 대량 데이터 적재 후 일괄 생성하여 I/O 비용 최소화
ALTER TABLE public.graph_state_timeslice ADD PRIMARY KEY (state_ts, node_uid);
CREATE INDEX idx_graph_state_ts_type ON public.graph_state_timeslice (state_ts, node_type);
CREATE INDEX idx_graph_state_node_uid ON public.graph_state_timeslice (node_uid);

COMMIT;
