-- ============================================================
-- load_graph_master_phase_2.sql
-- 목적: 통합 그래프 마스터 2단계 - STOP 노드 적재 및 정류장-링크 공간 매핑
-- 기준: EPSG:5187 직접 사용, 150m/500m 2단계 반경 전략
-- 작성일: 2026-04-10
-- ============================================================

/*
 [PREREQUISITE]
 본 스크립트 실행 전 public.prepare_source_spatial_columns.sql 을 실행하여
 dim_stop 및 gis_route_link 에 geom_5187 필드가 존재하고 백필되었는지 확인하십시오.
*/


-- ------------------------------------------------------------
-- 1. 물리 링크 대표 뷰 생성 (link_master_candidate_vw)
-- 목적: 동일 link_id를 가진 여러 노선 링크를 단일 물리 링크로 통합
-- ------------------------------------------------------------
create or replace view public.link_master_candidate_vw as
select distinct on (link_id)
    link_id,
    geom as geom_4326,
    geom_5187
from public.gis_route_link
order by link_id, created_at desc;

comment on view public.link_master_candidate_vw is 
'공간 매핑 및 그래프 구축을 위한 물리 링크 식별자 기준 대표 행 뷰';


-- ------------------------------------------------------------
-- 2. STOP 노드 적재 (graph_node_master)
-- ------------------------------------------------------------
insert into public.graph_node_master (
    node_uid,
    node_type,
    node_id,
    node_name,
    lon,
    lat,
    geom_4326,
    geom_5187,
    is_active
)
select
    'STOP:' || s.stop_id as node_uid,
    'STOP' as node_type,
    s.stop_id as node_id,
    s.stop_name as node_name,
    s.longitude as lon,
    s.latitude as lat,
    s.geom as geom_4326,
    s.geom_5187,
    s.is_active
from public.dim_stop s
on conflict (node_uid) do update
set node_name = excluded.node_name,
    lon = excluded.lon,
    lat = excluded.lat,
    geom_4326 = excluded.geom_4326,
    geom_5187 = excluded.geom_5187,
    updated_at = now();


-- ------------------------------------------------------------
-- 3. 정류장-링크 1차 공간 매핑 (stop_link_mapping_master)
-- 전략: 150m (Primary) -> 500m (Fallback) 최근접 매핑
-- ------------------------------------------------------------
with mapping_candidates as (
    -- 500m fallback 반경 내 모든 후보 탐색 (직접 geom_5187 GIST 인덱스 활용)
    select
        s.stop_id,
        l.link_id,
        st_distance(s.geom_5187, l.geom_5187) as dist_m
    from public.dim_stop s
    cross join public.link_master_candidate_vw l
    where st_dwithin(s.geom_5187, l.geom_5187, 500)
),
prioritized_mapping as (
    select
        stop_id,
        link_id,
        dist_m,
        row_number() over (
            partition by stop_id 
            order by 
                case when dist_m <= 150 then 1 else 2 end,  -- 150m 내 우선
                dist_m asc,                                -- 가까운 순
                link_id                                    -- 결정론적 순서
        ) as rn
    from mapping_candidates
)
insert into public.stop_link_mapping_master (
    stop_id,
    link_id,
    mapping_type,
    distance_to_link_m,
    confidence_score,
    is_primary_mapping,
    is_active
)
select
    stop_id,
    link_id,
    'SNAP_NEAREST' as mapping_type,
    dist_m as distance_to_link_m,
    case 
        when dist_m <= 15  then 1.00
        when dist_m <= 30  then 0.95
        when dist_m <= 50  then 0.85
        when dist_m <= 100 then 0.70
        else 0.50
    end as confidence_score,
    true as is_primary_mapping,
    true as is_active
from prioritized_mapping
where rn = 1
on conflict (stop_id, link_id, coalesce(route_id, ''), coalesce(move_dir_code, -1)) 
do update set
    distance_to_link_m = excluded.distance_to_link_m,
    confidence_score = excluded.confidence_score,
    updated_at = now();


-- ------------------------------------------------------------
-- 4. STOP_TO_LINK / LINK_TO_STOP 간선 적재 (graph_edge_master)
-- ------------------------------------------------------------

-- STOP_TO_LINK
insert into public.graph_edge_master (
    edge_uid,
    src_node_uid,
    dst_node_uid,
    src_node_type,
    src_node_id,
    dst_node_type,
    dst_node_id,
    edge_type,
    distance_m,
    generalized_cost,
    is_bidirectional,
    is_active
)
select
    'STOP_TO_LINK:' || stop_id || '->' || link_id as edge_uid,
    'STOP:' || stop_id as src_node_uid,
    'LINK:' || link_id as dst_node_uid,
    'STOP' as src_node_type,
    stop_id as src_node_id,
    'LINK' as dst_node_type,
    link_id as dst_node_id,
    'STOP_TO_LINK' as edge_type,
    distance_to_link_m as distance_m,
    distance_to_link_m as generalized_cost,
    false as is_bidirectional,
    true as is_active
from public.stop_link_mapping_master
where is_primary_mapping = true 
  and is_active = true
on conflict (edge_uid) do nothing;

-- LINK_TO_STOP
insert into public.graph_edge_master (
    edge_uid,
    src_node_uid,
    dst_node_uid,
    src_node_type,
    src_node_id,
    dst_node_type,
    dst_node_id,
    edge_type,
    distance_m,
    generalized_cost,
    is_bidirectional,
    is_active
)
select
    'LINK_TO_STOP:' || link_id || '->' || stop_id as edge_uid,
    'LINK:' || link_id as src_node_uid,
    'STOP:' || stop_id as dst_node_uid,
    'LINK' as src_node_type,
    link_id as src_node_id,
    'STOP' as dst_node_type,
    stop_id as dst_node_id,
    'LINK_TO_STOP' as edge_type,
    distance_to_link_m as distance_m,
    distance_to_link_m as generalized_cost,
    false as is_bidirectional,
    true as is_active
from public.stop_link_mapping_master
where is_primary_mapping = true 
  and is_active = true
on conflict (edge_uid) do nothing;
