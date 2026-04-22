-- ============================================================
-- graph_mapping_quality_check.sql
-- 목적: 통합 그래프 마스터 2단계 매핑 품질 및 정합성 검증
-- 작성일: 2026-04-10
-- ============================================================

-- ------------------------------------------------------------
-- 1) 정류장 매핑 통계 (total / mapped / unmapped)
-- ------------------------------------------------------------
with stats as (
    select
        count(s.stop_id) as total_stops,
        count(m.stop_id) as mapped_stops,
        count(s.stop_id) - count(m.stop_id) as unmapped_stops
    from public.dim_stop s
    left join public.stop_link_mapping_master m
        on s.stop_id = m.stop_id
        and m.is_primary_mapping = true
        and m.is_active = true
)
select
    *,
    round(mapped_stops::numeric / total_stops * 100, 2) as mapping_rate_pct
from stats;


-- ------------------------------------------------------------
-- 2) 중복 Primary 매핑 점검
-- 기대: 결과 0행 (Primary는 정류장당 1개여야 함)
-- ------------------------------------------------------------
select
    stop_id,
    count(*) as primary_cnt
from public.stop_link_mapping_master
where is_primary_mapping = true
  and is_active = true
group by stop_id
having count(*) > 1;


-- ------------------------------------------------------------
-- 3) 과거리 매핑 경고 (Extreme Distance Warning)
-- ------------------------------------------------------------
-- > 100m: 일반적인 도시 정류장-링크 간격 초과 (Warning)
-- > 250m: 심각한 오매핑 또는 데이터 누락 의심 (Review 대상)
select
    case 
        when distance_to_link_m > 250 then 'CRITICAL_REVIEW_REQUIRED'
        when distance_to_link_m > 100 then 'DISTANCE_WARNING'
    end as alert_level,
    count(*) as count
from public.stop_link_mapping_master
where is_primary_mapping = true
  and is_active = true
  and distance_to_link_m > 100
group by 1;


-- ------------------------------------------------------------
-- 4) 노드-간선 참조 무결성 (Node-Edge Integrity)
-- 기대: 결과 0행
-- ------------------------------------------------------------
select
    e.edge_uid,
    e.edge_type,
    e.src_node_uid,
    e.dst_node_uid
from public.graph_edge_master e
left join public.graph_node_master ns on e.src_node_uid = ns.node_uid
left join public.graph_node_master nd on e.dst_node_uid = nd.node_uid
where e.edge_type in ('STOP_TO_LINK', 'LINK_TO_STOP')
  and (ns.node_uid is null or nd.node_uid is null);


-- ------------------------------------------------------------
-- 5) 매핑 거리 분포 통계 (Distance Distribution Summary)
-- ------------------------------------------------------------
select
    round(avg(distance_to_link_m), 2) as avg_dist_m,
    percentile_cont(0.5) within group (order by distance_to_link_m) as median_dist_m,
    percentile_cont(0.95) within group (order by distance_to_link_m) as p95_dist_m,
    round(max(distance_to_link_m), 2) as max_dist_m
from public.stop_link_mapping_master
where is_primary_mapping = true
  and is_active = true;


-- ------------------------------------------------------------
-- 6) 생성된 서비스 간선 수 요약
-- ------------------------------------------------------------
select
    edge_type,
    count(*) as edge_cnt
from public.graph_edge_master
where edge_type in ('STOP_TO_LINK', 'LINK_TO_STOP')
  and is_active = true
group by edge_type;


-- ------------------------------------------------------------
-- [수동 검토용] 상위 과거리 매핑 정류장 목록 (TOP 20)
-- ------------------------------------------------------------
select
    s.stop_id,
    s.stop_name,
    m.link_id,
    round(m.distance_to_link_m, 2) as dist_m,
    m.confidence_score
from public.dim_stop s
join public.stop_link_mapping_master m on s.stop_id = m.stop_id
where m.is_primary_mapping = true
  and m.is_active = true
order by m.distance_to_link_m desc
limit 20;


-- ------------------------------------------------------------
-- [수동 검토용] 주요 랜드마크 정류장 매핑 확인 예시
-- ------------------------------------------------------------
select
    s.stop_id,
    s.stop_name,
    m.link_id,
    round(m.distance_to_link_m, 2) as dist_m,
    case 
        when m.distance_to_link_m < 20 then 'EXCELLENT'
        when m.distance_to_link_m < 50 then 'GOOD'
        else 'NEED_CHECK'
    end as status
from public.dim_stop s
join public.stop_link_mapping_master m on s.stop_id = m.stop_id
where s.stop_name like '%대구역%' or s.stop_name like '%동대구역%'
  and m.is_primary_mapping = true;
