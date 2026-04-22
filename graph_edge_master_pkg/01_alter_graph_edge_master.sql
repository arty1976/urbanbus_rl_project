-- =============================================================================
-- 01_alter_graph_edge_master.sql
-- Purpose : 기존 graph_edge_master 에 누락된 is_primary_edge / edge_rank 컬럼 추가
--           + STOP_TO_STOP 전용 보조 인덱스 생성
-- Scope   : idempotent. 기존 테이블/데이터/제약 건드리지 않음 (ADD COLUMN IF NOT EXISTS)
-- Safety  : DROP 없음, 다른 edge_type 행 보존
-- Author  : urbanbus_rl_project
-- Date    : 2026-04-19
-- =============================================================================

begin;

alter table if exists public.graph_edge_master
    add column if not exists is_primary_edge boolean not null default false;

alter table if exists public.graph_edge_master
    add column if not exists edge_rank smallint null;

-- (src, dst) 쌍에서 primary edge 를 한 개만 찾기 쉽게
create index if not exists idx_graph_edge_master_primary_pair
    on public.graph_edge_master (src_node_uid, dst_node_uid)
    where is_primary_edge = true;

-- edge_type 별 필터 가속
create index if not exists idx_graph_edge_master_edge_type
    on public.graph_edge_master (edge_type);

-- route 별 조회 가속 (STOP_TO_STOP 로드/검증에서 자주 사용)
create index if not exists idx_graph_edge_master_route_dir
    on public.graph_edge_master (route_id, move_dir_code)
    where route_id is not null;

commit;

-- 확인
\d public.graph_edge_master
