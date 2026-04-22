-- ============================================================
-- create_route_link_graph_edge_view_and_load.sql
-- 목적: route_link_sequence 기반 링크 간 순차 연결 뷰 생성 및 graph_edge_master 적재
-- 전제: public.route_link_sequence 운영 기준 테이블 존재
-- 기준 키: (route_id, move_dir_code, link_seq)
-- 작성일: 2026-04-10
-- ============================================================

create or replace view public.route_link_graph_edge_vw as
with base as (
    select
        route_id,
        move_dir_code,
        link_seq,
        link_id,
        lead(link_id) over (
            partition by route_id, move_dir_code
            order by link_seq
        ) as next_link_id,
        lead(link_seq) over (
            partition by route_id, move_dir_code
            order by link_seq
        ) as next_link_seq
    from public.route_link_sequence
)
select
    route_id,
    move_dir_code,
    link_id as src_link_id,
    next_link_id as dst_link_id,
    link_seq as src_link_seq,
    next_link_seq as dst_link_seq
from base
where next_link_id is not null;

comment on view public.route_link_graph_edge_vw is
'route_link_sequence에서 동일 route_id + move_dir_code 내부의 연속 링크 연결을 생성하는 뷰';

-- ------------------------------------------------------------
-- graph_node_master 에 LINK 노드 적재
-- 주의: 실제 링크 마스터 원천 테이블이 확정되면 좌표 컬럼을 맞춰 수정 필요
-- 현재 버전은 route_link_sequence에 등장하는 link_id를 기준으로 최소 노드만 적재
-- ------------------------------------------------------------
insert into public.graph_node_master (
    node_uid,
    node_type,
    node_id,
    node_name,
    route_id,
    move_dir_code,
    is_active
)
select distinct
    'LINK:' || rls.link_id as node_uid,
    'LINK' as node_type,
    rls.link_id as node_id,
    null as node_name,
    null as route_id,
    null as move_dir_code,
    true as is_active
from public.route_link_sequence rls
on conflict (node_uid) do nothing;

-- ------------------------------------------------------------
-- graph_edge_master 에 LINK_TO_LINK 간선 적재
-- generalized_cost는 현 단계에서는 time_sec 우선, 없으면 distance_m 사용 여지를 남김
-- ------------------------------------------------------------
insert into public.graph_edge_master (
    edge_uid,
    src_node_uid,
    dst_node_uid,
    src_node_type,
    src_node_id,
    dst_node_type,
    dst_node_id,
    edge_type,
    route_id,
    move_dir_code,
    link_id,
    link_seq,
    distance_m,
    time_sec,
    generalized_cost,
    is_bidirectional,
    is_active
)
select
    'LINK_TO_LINK:' || coalesce(route_id, 'NA') || ':' || coalesce(move_dir_code::text, 'NA') || ':' || src_link_seq::text || ':' || src_link_id || '->' || dst_link_id as edge_uid,
    'LINK:' || src_link_id as src_node_uid,
    'LINK:' || dst_link_id as dst_node_uid,
    'LINK' as src_node_type,
    src_link_id as src_node_id,
    'LINK' as dst_node_type,
    dst_link_id as dst_node_id,
    'LINK_TO_LINK' as edge_type,
    route_id,
    move_dir_code,
    src_link_id as link_id,
    src_link_seq as link_seq,
    null::numeric(12,3) as distance_m,
    null::numeric(12,3) as time_sec,
    null::numeric(12,3) as generalized_cost,
    false as is_bidirectional,
    true as is_active
from public.route_link_graph_edge_vw
on conflict (edge_uid) do nothing;
