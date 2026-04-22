-- =============================================================================
-- 03_graph_edge_master_readiness.sql
-- Purpose : STOP_TO_STOP edge 로드 후 readiness 검증 (GATv2 static graph 용)
-- 통과 기준 (pass criteria):
--   [R1] stop_to_stop_total > 0
--   [R2] primary_edges > 0 AND primary_edges <= stop_to_stop_total
--   [R3] self_loop_cnt = 0
--   [R4] orphan_src = 0 AND orphan_dst = 0
--   [R5] duplicate_primary_cnt = 0
--   [R6] null_distance = 0 AND null_time = 0 AND zero_distance = 0
--   [R7] min_dist > 0 AND max_dist < 20000 (20 km 이상이면 이상치 의심)
-- =============================================================================

\echo '================================================================='
\echo ' [R0] Overall Counts'
\echo '================================================================='
select
    count(*) filter (where edge_type = 'STOP_TO_STOP')                              as stop_to_stop_total,
    count(*) filter (where edge_type = 'STOP_TO_STOP' and is_primary_edge)          as primary_edges,
    count(*) filter (where edge_type = 'STOP_TO_STOP' and not is_primary_edge)      as non_primary_edges,
    count(*) filter (where edge_type = 'STOP_TO_STOP' and is_active = false)        as inactive_edges,
    count(*)                                                                        as total_rows_all_types
from public.graph_edge_master;

\echo ''
\echo '================================================================='
\echo ' [R3] Self-loop Check  (expect 0)'
\echo '================================================================='
select count(*) as self_loop_cnt
from public.graph_edge_master
where edge_type = 'STOP_TO_STOP'
  and src_node_uid = dst_node_uid;

\echo ''
\echo '================================================================='
\echo ' [R4] Orphan Check  (expect 0, 0)'
\echo '================================================================='
select
    (select count(*)
       from public.graph_edge_master e
       left join public.dim_stop ds on ds.stop_id = e.src_node_id
      where e.edge_type = 'STOP_TO_STOP' and ds.stop_id is null) as orphan_src,
    (select count(*)
       from public.graph_edge_master e
       left join public.dim_stop ds on ds.stop_id = e.dst_node_id
      where e.edge_type = 'STOP_TO_STOP' and ds.stop_id is null) as orphan_dst;

\echo ''
\echo '================================================================='
\echo ' [R5] Duplicate Primary Edge Check  (expect 0 rows)'
\echo '================================================================='
select src_node_uid, dst_node_uid, count(*) as primary_cnt
from public.graph_edge_master
where edge_type = 'STOP_TO_STOP' and is_primary_edge = true
group by src_node_uid, dst_node_uid
having count(*) > 1
order by primary_cnt desc
limit 20;

\echo ''
\echo '================================================================='
\echo ' [R6] NULL / Zero Distance & Time Check  (expect 0, 0, 0)'
\echo '================================================================='
select
    count(*) filter (where distance_m is null)  as null_distance,
    count(*) filter (where time_sec   is null)  as null_time,
    count(*) filter (where distance_m = 0)      as zero_distance
from public.graph_edge_master
where edge_type = 'STOP_TO_STOP';

\echo ''
\echo '================================================================='
\echo ' [R7] Distance Distribution (sanity: 대부분 50m ~ 2000m)'
\echo '================================================================='
select
    min(distance_m)::numeric(12,3)                                       as min_dist_m,
    max(distance_m)::numeric(12,3)                                       as max_dist_m,
    avg(distance_m)::numeric(12,3)                                       as avg_dist_m,
    percentile_cont(0.50) within group (order by distance_m)::numeric(12,3) as p50_dist_m,
    percentile_cont(0.95) within group (order by distance_m)::numeric(12,3) as p95_dist_m,
    count(*) filter (where distance_m > 5000)                            as over_5km,
    count(*) filter (where distance_m > 20000)                           as over_20km
from public.graph_edge_master
where edge_type = 'STOP_TO_STOP';

\echo ''
\echo '================================================================='
\echo ' [R8] Time Distribution (base_travel_time_min 환산용 참고)'
\echo '================================================================='
select
    min(time_sec)::numeric(12,3)                                          as min_time_sec,
    max(time_sec)::numeric(12,3)                                          as max_time_sec,
    avg(time_sec)::numeric(12,3)                                          as avg_time_sec,
    avg(time_sec / 60.0)::numeric(12,3)                                   as avg_time_min,
    percentile_cont(0.50) within group (order by time_sec)::numeric(12,3) as p50_time_sec
from public.graph_edge_master
where edge_type = 'STOP_TO_STOP';

\echo ''
\echo '================================================================='
\echo ' [R9] Per-route Edge Distribution (top 20 routes)'
\echo '================================================================='
select
    route_id,
    move_dir_code,
    count(*)                                              as edges,
    count(*) filter (where is_primary_edge)               as primary_edges,
    min(distance_m)::numeric(12,3)                        as min_d,
    max(distance_m)::numeric(12,3)                        as max_d,
    avg(distance_m)::numeric(12,3)                        as avg_d
from public.graph_edge_master
where edge_type = 'STOP_TO_STOP' and is_active = true
group by route_id, move_dir_code
order by edges desc
limit 20;

\echo ''
\echo '================================================================='
\echo ' [R10] Source Coverage — leg-aggregation 기준'
\echo '  (v2 이후: route_link_sequence 의 link row 가 아니라, leg-aggregation'
\echo '           으로 도출되는 "연속 두 stop 사이" leg 수를 분모로 사용)'
\echo '================================================================='
with rls_tagged as (
    select
        r.route_id,
        r.move_dir_code::integer as move_dir_code,
        r.link_seq,
        r.st_node_id,
        r.ed_node_id,
        (d1.stop_id is not null) as st_is_stop,
        (d2.stop_id is not null) as ed_is_stop
    from public.route_link_sequence r
    left join public.dim_stop d1 on d1.stop_id = r.st_node_id
    left join public.dim_stop d2 on d2.stop_id = r.ed_node_id
    where r.st_node_id is not null
      and r.ed_node_id is not null
      and r.st_node_id <> r.ed_node_id
),
rls_legged as (
    select *,
        sum(case when st_is_stop then 1 else 0 end) over (
            partition by route_id, move_dir_code
            order by link_seq
            rows unbounded preceding
        ) as leg_id
    from rls_tagged
),
leg_aggr as (
    select
        route_id, move_dir_code, leg_id,
        (array_agg(st_node_id order by link_seq))[1]                as src_stop_id,
        (array_agg(ed_node_id order by link_seq)
           filter (where ed_is_stop))[1]                            as dst_stop_id
    from rls_legged
    group by route_id, move_dir_code, leg_id
),
source_legs as (
    select count(*) as leg_cnt_total,
           count(*) filter (where src_stop_id is not null
                              and dst_stop_id is not null
                              and src_stop_id <> dst_stop_id) as leg_cnt_valid,
           count(distinct (route_id, move_dir_code, src_stop_id, dst_stop_id))
               filter (where src_stop_id is not null
                         and dst_stop_id is not null
                         and src_stop_id <> dst_stop_id)     as leg_cnt_distinct
      from leg_aggr
),
edge_pairs as (
    select count(*) as edge_cnt
      from public.graph_edge_master
     where edge_type = 'STOP_TO_STOP' and is_active = true
)
select
    s.leg_cnt_total                            as source_legs_total,
    s.leg_cnt_valid                            as source_legs_valid,
    s.leg_cnt_distinct                         as source_legs_distinct,
    e.edge_cnt                                 as loaded_edges,
    (s.leg_cnt_distinct - e.edge_cnt)          as missing,
    round((e.edge_cnt::numeric / nullif(s.leg_cnt_distinct,0)) * 100, 2)
                                               as coverage_pct
from source_legs s, edge_pairs e;

\echo ''
\echo '================================================================='
\echo ' [R11] route_link_sequence 의 non-stop node_id 샘플'
\echo '  (v2 이후: 이 값들은 매칭 실패가 아니라 의도된 intermediate node.'
\echo '           dim_stop 에는 없어야 정상 — 대부분 도로/교차로 node)'
\echo '================================================================='
select distinct r.st_node_id as non_stop_node_id, 'st_node_id' as role
from public.route_link_sequence r
left join public.dim_stop ds on ds.stop_id = r.st_node_id
where ds.stop_id is null and r.st_node_id is not null
union all
select distinct r.ed_node_id as non_stop_node_id, 'ed_node_id' as role
from public.route_link_sequence r
left join public.dim_stop ds on ds.stop_id = r.ed_node_id
where ds.stop_id is null and r.ed_node_id is not null
limit 30;

\echo ''
\echo '================================================================='
\echo ' [R12] edge_uid 샘플 (패턴 확인)'
\echo '================================================================='
select edge_uid, src_node_uid, dst_node_uid, distance_m, time_sec,
       is_primary_edge, edge_rank, route_id, move_dir_code
from public.graph_edge_master
where edge_type = 'STOP_TO_STOP'
order by edge_uid
limit 10;

\echo ''
\echo '================================================================='
\echo ' READINESS SUMMARY — 위 쿼리 결과를 복사해 project_log.md 에 붙여주세요'
\echo '================================================================='
