SET client_encoding = 'UTF8';
-- =============================================================================
-- File   : 04_model_inputs/materialize_gatv2_training_table.sql
-- Purpose: Convert public.gatv2_snapshot_stop_features_train (VIEW, slow)
--          into a materialized TABLE with indexes so PyG DataLoader can
--          iterate fast during training.
-- Date   : 2026-04-19
-- Notes  :
--   - Snapshot view 재전개 비용이 크므로 CREATE TABLE AS SELECT 로 1회 고정.
--   - 한번 돌고 나면 이후 조회는 (state_ts, node_index) 인덱스로 즉답.
--   - 학습 파이프라인은 테이블이 있으면 테이블을, 없으면 뷰를 자동 사용.
--     (build_gatv2_dataset.py 내부 분기)
--   - 소스 data 갱신 시: 테이블 drop 후 이 스크립트 재실행 (idempotent via
--     DROP IF EXISTS).
-- =============================================================================

\timing on

\echo ''
\echo '================================================================='
\echo ' [MAT-0] drop existing materialized table (if any)'
\echo '================================================================='
DROP TABLE IF EXISTS public.gatv2_snapshot_stop_features_train_mat CASCADE;

\echo ''
\echo '================================================================='
\echo ' [MAT-1] CREATE TABLE AS SELECT  (expect ~2h, 20.29M rows)'
\echo '================================================================='
CREATE TABLE public.gatv2_snapshot_stop_features_train_mat AS
SELECT * FROM public.gatv2_snapshot_stop_features_train;

\echo ''
\echo '================================================================='
\echo ' [MAT-2] row count sanity (expect 20,289,600)'
\echo '================================================================='
SELECT count(*) AS materialized_rows
FROM public.gatv2_snapshot_stop_features_train_mat;

\echo ''
\echo '================================================================='
\echo ' [MAT-3] indexes for training-time access patterns'
\echo '================================================================='
-- (state_ts, node_index) : 스냅샷 단위 읽기의 기본 키. PyG snapshot iteration.
CREATE INDEX IF NOT EXISTS idx_gatv2_mat_ts_node
    ON public.gatv2_snapshot_stop_features_train_mat (state_ts, node_index);

-- (service_date) : 학습/검증/테스트 split 시 date-range 필터에 사용.
CREATE INDEX IF NOT EXISTS idx_gatv2_mat_service_date
    ON public.gatv2_snapshot_stop_features_train_mat (service_date);

-- (node_index) : per-node 분석 쿼리용 보조 인덱스.
CREATE INDEX IF NOT EXISTS idx_gatv2_mat_node_index
    ON public.gatv2_snapshot_stop_features_train_mat (node_index);

\echo ''
\echo '================================================================='
\echo ' [MAT-4] VACUUM ANALYZE'
\echo '================================================================='
VACUUM ANALYZE public.gatv2_snapshot_stop_features_train_mat;

\echo ''
\echo '================================================================='
\echo ' [MAT-5] final sanity — view <-> table row/snapshot parity'
\echo '================================================================='
SELECT
    (SELECT count(*) FROM public.gatv2_snapshot_stop_features_train_mat)       AS mat_rows,
    (SELECT total_training_rows FROM public.gatv2_snapshot_summary)            AS view_total_rows,
    (SELECT count(DISTINCT state_ts)
       FROM public.gatv2_snapshot_stop_features_train_mat)                     AS mat_snapshots,
    (SELECT distinct_snapshots FROM public.gatv2_snapshot_summary)             AS view_snapshots,
    CASE
      WHEN (SELECT count(*) FROM public.gatv2_snapshot_stop_features_train_mat)
         = (SELECT total_training_rows FROM public.gatv2_snapshot_summary)
      THEN 'OK'
      ELSE 'MISMATCH'
    END AS parity_check;

\echo ''
\echo '================================================================='
\echo ' MATERIALIZATION COMPLETE'
\echo '================================================================='
