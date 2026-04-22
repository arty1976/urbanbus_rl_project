-- ============================================================
-- graph_master_readiness.sql
-- 목적: 통합 그래프 마스터 1차 readiness 점검
-- 범위: route_link_sequence, route_link_graph_edge_vw, graph_node_master,
--       graph_edge_master, stop_link_mapping_master
-- 작성일: 2026-04-10
-- ============================================================

-- ------------------------------------------------------------
-- 1) route_id + move_dir_code 단위 연결 간선 미생성 여부
-- 기대: 결과 0행
-- ------------------------------------------------------------
select
    route_id,
    move_dir_code,
    count(*) as edge_cnt
from public.route_link_graph_edge_vw
group by route_id, move_dir_code
having count(*) = 0;

-- ------------------------------------------------------------
-- 2) link_seq 역전 / 비증가 여부
-- 기대: 결과 0행
-- ------------------------------------------------------------
with seq_chk as (
    select
        route_id,
        move_dir_code,
        link_seq,
        lag(link_seq) over (
            partition by route_id, move_dir_code
            order by link_seq
        ) as prev_seq
    from public.route_link_sequence
)
select *
from seq_chk
where prev_seq is not null
  and link_seq <= prev_seq;

-- ------------------------------------------------------------
-- 3) graph_edge_master 의 고아 노드 여부
-- 기대: 결과 0행
-- ------------------------------------------------------------
select
    e.edge_uid,
    e.src_node_uid,
    e.dst_node_uid
from public.graph_edge_master e
left join public.graph_node_master ns
    on e.src_node_uid = ns.node_uid
left join public.graph_node_master nd
    on e.dst_node_uid = nd.node_uid
where ns.node_uid is null
   or nd.node_uid is null;

-- ------------------------------------------------------------
-- 4) 정류장-링크 기본 매핑 누락 정류장
-- 기대: 초기 단계에서는 일부 발생 가능. 후속 보정 대상 목록으로 사용
-- ------------------------------------------------------------
select
    s.stop_id,
    s.stop_name
from public.dim_stop s
left join public.stop_link_mapping_master m
    on s.stop_id = m.stop_id
   and m.is_active = true
   and m.is_primary_mapping = true
where m.stop_id is null
order by s.stop_id;

-- ------------------------------------------------------------
-- 5) 활성 간선 out-degree 상위 노드
-- 기대: 비정상 허브 노드 탐지용
-- ------------------------------------------------------------
select
    src_node_uid,
    count(*) as out_degree
from public.graph_edge_master
where is_active = true
group by src_node_uid
order by out_degree desc, src_node_uid
limit 50;

-- ------------------------------------------------------------
-- 6) LINK 노드 수와 LINK_TO_LINK 간선 수 요약
-- 기대: 운영 규모 파악용
-- ------------------------------------------------------------
select
    count(*) filter (where node_type = 'LINK') as link_node_cnt,
    count(*) filter (where node_type = 'STOP') as stop_node_cnt
from public.graph_node_master;

select
    count(*) as link_to_link_edge_cnt
from public.graph_edge_master
where edge_type = 'LINK_TO_LINK'
  and is_active = true;
