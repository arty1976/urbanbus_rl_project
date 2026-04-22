SET client_encoding = 'UTF8';
-- =============================================================================
-- File   : 04_model_inputs/create_gatv2_training_views.sql
-- Author : urbanbus_rl_project
-- Date   : 2026-04-19
-- Stage  : Graph-ready transition learning  (pre-RL, post-edge-build)
--
-- Purpose
--   GATv2 학습 파이프라인 (PyTorch Geometric) 이 읽을 5 개 DB 뷰를
--   한 파일 안에서 idempotent 하게 생성 / 재정의한다.
--
-- 업스트림 승인 상태 (2026-04-19 기준)
--   - graph_state_timeslice             = APPROVED
--   - rl_state_training_base            = APPROVED
--   - rl_stop_transition_features       = APPROVED
--   - graph_edge_master STOP_TO_STOP v2 = leg-aggregation, coverage 99.91%
--
-- Scope
--   GATv2 정적 그래프 스켈레톤 + node temporal feature snapshot 결합.
--   ※ dynamic traffic (link_travel_time_sec 등) 은 현 단계 범위 **아님**.
--   ※ RL 정책 학습이 아닌 transition (t -> t+1) 예측용 dataset 구성 단계.
--
-- 설계 원칙
--   1) 노드 모집단:  primary + active STOP_TO_STOP 엣지에 실제 참여하는 STOP 노드만.
--                    => 고립/terminal-only stop 은 그래프에서 제외되어 isolated node
--                       문제를 PyG 측에서 만들지 않는다.
--   2) 엣지 모집단:  edge_type='STOP_TO_STOP' AND is_primary_edge AND is_active
--                    => overlapping 노선 다중성은 edge_rank=1 (primary) 로 대표됨.
--   3) 노드 인덱스:  node_uid 알파벳 오름차순 row_number() - 1 (0-based, dense).
--                    뷰 재빌드 시 node_uid set 이 바뀌지 않는 한 인덱스는 고정.
--   4) 스냅샷 단위:  state_ts (동일 state_ts 를 공유하는 모든 노드 = 한 그래프 스냅샷).
--   5) 학습 제외:    is_overnight_gap=1 OR is_terminal_transition=1  전부 컷.
--   6) 대상 변환:    target 은 log1p 변환도 함께 노출 (MSE on log space 권장).
--   7) edge_attr:   distance_m / time_sec + km/min 변환 + log1p 변환.
--   8) 참조 무결성: 모든 뷰는 CASCADE 로 재생성되지 않음 — 다른 객체가 참조하면
--                   재실행 전에 drop 순서 고려. 처음 실행이면 문제 없음.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- (1) public.gatv2_node_master_active
--     active + primary STOP_TO_STOP 엣지에 실제 참여하는 STOP 노드.
--     node_index = node_uid 알파벳 정렬 row_number() - 1  (PyG x 텐서 행 순서).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gatv2_node_master_active AS
WITH participating AS (
    SELECT DISTINCT src_node_uid AS node_uid
      FROM public.graph_edge_master
     WHERE edge_type = 'STOP_TO_STOP'
       AND is_primary_edge = true
       AND is_active = true
    UNION
    SELECT DISTINCT dst_node_uid AS node_uid
      FROM public.graph_edge_master
     WHERE edge_type = 'STOP_TO_STOP'
       AND is_primary_edge = true
       AND is_active = true
),
indexed AS (
    SELECT
        node_uid,
        CASE WHEN node_uid LIKE 'STOP:%'
             THEN substring(node_uid FROM 6)
             ELSE NULL
        END AS stop_id_text,
        (row_number() OVER (ORDER BY node_uid) - 1)::integer AS node_index
    FROM participating
)
SELECT
    i.node_uid,
    i.stop_id_text,
    i.node_index,
    count(*) OVER ()::integer AS num_nodes
FROM indexed i;

COMMENT ON VIEW public.gatv2_node_master_active IS
'GATv2 입력 노드 마스터. primary+active STOP_TO_STOP 엣지 참여 STOP 노드만. node_index 는 0-based dense (node_uid 오름차순).';

-- -----------------------------------------------------------------------------
-- (2) public.gatv2_edge_primary_active
--     edge_index / edge_attr 용. src/dst 양쪽 모두 node_master_active 에 있어야 함.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gatv2_edge_primary_active AS
SELECT
    e.edge_uid,
    e.src_node_uid,
    e.dst_node_uid,
    n_src.node_index AS src_node_index,
    n_dst.node_index AS dst_node_index,
    e.route_id,
    e.move_dir_code,
    e.distance_m,
    e.time_sec,
    e.edge_rank,
    -- edge_attr 학습 친화 변환
    (e.distance_m / 1000.0)::double precision           AS distance_km,
    (e.time_sec   / 60.0  )::double precision           AS time_min,
    ln(1.0 + e.distance_m)::double precision            AS distance_m_log,
    ln(1.0 + e.time_sec  )::double precision            AS time_sec_log
FROM public.graph_edge_master e
JOIN public.gatv2_node_master_active n_src
       ON n_src.node_uid = e.src_node_uid
JOIN public.gatv2_node_master_active n_dst
       ON n_dst.node_uid = e.dst_node_uid
WHERE e.edge_type = 'STOP_TO_STOP'
  AND e.is_primary_edge = true
  AND e.is_active = true;

COMMENT ON VIEW public.gatv2_edge_primary_active IS
'GATv2 edge_index/edge_attr. primary + active STOP_TO_STOP directed edges. (src_node_index, dst_node_index) 바로 PyG edge_index 로 pivot 가능.';

-- -----------------------------------------------------------------------------
-- (3) public.gatv2_snapshot_stop_features_train
--     학습용 (s, s') 전이 쌍. node_index 포함 (PyG batching 용).
--     overnight / terminal 전이는 제외.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gatv2_snapshot_stop_features_train AS
SELECT
    -- keys
    t.state_ts,
    t.next_state_ts,
    t.node_uid,
    n.node_index,
    -- service metadata (batching / split 용)
    t.service_date,
    t.next_service_date,
    t.service_bucket_seq,
    t.next_service_bucket_seq,
    t.buckets_per_day_actual,
    -- 현재 상태 raw
    t.boardings_recent,
    t.alightings_recent,
    t.waiting_passenger_cnt,
    t.hour_of_day,
    t.day_of_week,
    t.is_peak,
    -- 현재 상태 변환 (GATv2 x 벡터 성분)
    t.boardings_recent_log,
    t.alightings_recent_log,
    t.waiting_passenger_cnt_log,
    t.hour_sin,
    t.hour_cos,
    t.dow_sin,
    t.dow_cos,
    t.node_type_stop,
    -- 전이 타이밍
    t.delta_t_min_raw,
    t.delta_t_hr_raw,
    t.delta_t_hr_effective,
    t.is_cross_calendar_day,
    t.is_overnight_gap,
    t.is_terminal_transition,
    -- target (t+1) raw
    t.next_boardings_recent,
    t.next_alightings_recent,
    t.next_waiting_passenger_cnt,
    -- target (t+1) log1p (학습 대상으로 추천 — MSE on log space)
    ln(1 + t.next_boardings_recent)::double precision         AS next_boardings_recent_log,
    ln(1 + t.next_alightings_recent)::double precision        AS next_alightings_recent_log,
    ln(1 + t.next_waiting_passenger_cnt)::double precision    AS next_waiting_passenger_cnt_log
FROM public.rl_stop_transition_features t
JOIN public.gatv2_node_master_active n
       ON n.node_uid = t.node_uid
WHERE t.node_type = 'STOP'
  AND t.is_overnight_gap = 0
  AND t.is_terminal_transition = 0;

COMMENT ON VIEW public.gatv2_snapshot_stop_features_train IS
'GATv2 학습 dataset rows. state_ts = 한 snapshot, node_index = 해당 snapshot 안의 노드 순서. overnight/terminal 전이 제외.';

-- -----------------------------------------------------------------------------
-- (4) public.gatv2_snapshot_summary
--     데이터셋 크기 / 시간범위 / 스냅샷당 rows 분포.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gatv2_snapshot_summary AS
WITH by_snap AS (
    SELECT state_ts, COUNT(*)::integer AS rows_per_snapshot
      FROM public.gatv2_snapshot_stop_features_train
     GROUP BY state_ts
)
SELECT
    COUNT(*)::bigint                           AS distinct_snapshots,
    MIN(state_ts)                              AS min_state_ts,
    MAX(state_ts)                              AS max_state_ts,
    SUM(rows_per_snapshot)::bigint             AS total_training_rows,
    MIN(rows_per_snapshot)::integer            AS min_rows_per_snapshot,
    AVG(rows_per_snapshot)::numeric(12,2)      AS avg_rows_per_snapshot,
    MAX(rows_per_snapshot)::integer            AS max_rows_per_snapshot
FROM by_snap;

COMMENT ON VIEW public.gatv2_snapshot_summary IS
'GATv2 학습 dataset 전체 요약. min/max state_ts = 학습 가능 시간범위, total_training_rows = PyG Data 객체 row 수.';

-- -----------------------------------------------------------------------------
-- (5) public.gatv2_edge_summary
--     그래프 스켈레톤 통계 (num_nodes/edges/ route coverage / 거리·시간 분포).
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gatv2_edge_summary AS
SELECT
    (SELECT COUNT(*) FROM public.gatv2_node_master_active)::integer     AS num_nodes,
    COUNT(*)::bigint                                                    AS num_edges,
    COUNT(DISTINCT src_node_uid)::integer                               AS distinct_src_nodes,
    COUNT(DISTINCT dst_node_uid)::integer                               AS distinct_dst_nodes,
    COUNT(DISTINCT (route_id, move_dir_code))::integer                  AS distinct_route_dirs,
    MIN(distance_m)::numeric(12,3)                                      AS min_distance_m,
    MAX(distance_m)::numeric(12,3)                                      AS max_distance_m,
    AVG(distance_m)::numeric(12,3)                                      AS avg_distance_m,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY distance_m)::numeric(12,3) AS p50_distance_m,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY distance_m)::numeric(12,3) AS p95_distance_m,
    MIN(time_sec)::numeric(12,3)                                        AS min_time_sec,
    MAX(time_sec)::numeric(12,3)                                        AS max_time_sec,
    AVG(time_sec)::numeric(12,3)                                        AS avg_time_sec
FROM public.gatv2_edge_primary_active;

COMMENT ON VIEW public.gatv2_edge_summary IS
'GATv2 static graph skeleton 통계. PyG Data 객체의 num_nodes / edge_index.size(1) 와 반드시 일치해야 함.';

COMMIT;

-- =============================================================================
--  Readiness checks  (SELECT-only)
-- =============================================================================
\echo ''
\echo '================================================================='
\echo ' [V1] gatv2_node_master_active  — num_nodes + index 연속성'
\echo '================================================================='
SELECT
    max(num_nodes)                                AS num_nodes,
    count(*)                                      AS view_row_count,
    min(node_index)                               AS min_idx,
    max(node_index)                               AS max_idx,
    count(*) - (max(node_index) + 1)              AS index_gap_check_zero_ok
FROM public.gatv2_node_master_active;

\echo ''
\echo '================================================================='
\echo ' [V2] gatv2_edge_primary_active  — orphan / 자기루프 / rank 분포'
\echo '================================================================='
SELECT
    count(*)                                                         AS num_edges,
    count(DISTINCT src_node_uid)                                     AS distinct_src,
    count(DISTINCT dst_node_uid)                                     AS distinct_dst,
    count(*) FILTER (WHERE src_node_index IS NULL
                      OR dst_node_index IS NULL)                     AS orphan_index_cnt,
    count(*) FILTER (WHERE src_node_uid = dst_node_uid)              AS self_loop_cnt,
    count(*) FILTER (WHERE edge_rank = 1)                            AS rank1_cnt,
    count(*) FILTER (WHERE edge_rank IS NULL OR edge_rank <> 1)      AS non_rank1_cnt
FROM public.gatv2_edge_primary_active;

\echo ''
\echo '================================================================='
\echo ' [V3] gatv2_edge_summary'
\echo '================================================================='
SELECT * FROM public.gatv2_edge_summary;

\echo ''
\echo '================================================================='
\echo ' [V4] gatv2_snapshot_stop_features_train — 누수 / orphan 점검'
\echo '================================================================='
SELECT
    count(*)                                         AS total_training_rows,
    count(*) FILTER (WHERE node_index IS NULL)       AS orphan_node_index_cnt,
    count(*) FILTER (WHERE is_overnight_gap = 1)     AS overnight_leak_cnt,
    count(*) FILTER (WHERE is_terminal_transition=1) AS terminal_leak_cnt,
    count(*) FILTER (WHERE node_type_stop <> 1)      AS non_stop_leak_cnt
FROM public.gatv2_snapshot_stop_features_train;

\echo ''
\echo '================================================================='
\echo ' [V5] gatv2_snapshot_summary'
\echo '================================================================='
SELECT * FROM public.gatv2_snapshot_summary;

\echo ''
\echo '================================================================='
\echo ' [V6] per-node 학습 row density (graph coverage)'
\echo '================================================================='
WITH per_node AS (
    SELECT node_uid, count(*) AS row_cnt
      FROM public.gatv2_snapshot_stop_features_train
     GROUP BY node_uid
)
SELECT
    count(*)                                       AS nodes_with_training_rows,
    min(row_cnt)                                   AS min_rows_per_node,
    avg(row_cnt)::numeric(12,2)                    AS avg_rows_per_node,
    max(row_cnt)                                   AS max_rows_per_node
FROM per_node;

\echo ''
\echo '================================================================='
\echo ' [V7] nodes in graph but WITHOUT any training row  (expected: few/0)'
\echo '================================================================='
SELECT count(*) AS nodes_without_any_training_row
  FROM public.gatv2_node_master_active n
 WHERE NOT EXISTS (
     SELECT 1 FROM public.gatv2_snapshot_stop_features_train t
      WHERE t.node_uid = n.node_uid
 );

\echo ''
\echo '================================================================='
\echo ' [V8] target value sanity — 음수/NaN 없어야 함'
\echo '================================================================='
SELECT
    count(*) FILTER (WHERE next_boardings_recent < 0
                       OR next_alightings_recent < 0
                       OR next_waiting_passenger_cnt < 0)            AS negative_target_cnt,
    min(next_boardings_recent)                                       AS min_next_boardings,
    max(next_boardings_recent)                                       AS max_next_boardings,
    avg(next_boardings_recent)::numeric(12,3)                        AS avg_next_boardings,
    max(next_waiting_passenger_cnt)                                  AS max_next_waiting
FROM public.gatv2_snapshot_stop_features_train;

\echo ''
\echo '================================================================='
\echo ' GATv2 VIEWS READINESS COMPLETE'
\echo '================================================================='
