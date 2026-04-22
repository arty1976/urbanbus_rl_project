-- =============================================================================
-- 02_load_graph_edge_master_from_route_link_sequence.sql
-- Purpose : route_link_sequence + dim_stop 으로부터 STOP_TO_STOP edge 생성/갱신
-- Scope   : edge_type = 'STOP_TO_STOP' ONLY. 다른 edge_type 행은 건드리지 않음.
--
-- Version : 2 (2026-04-19, leg-aggregation 방식으로 재설계)
-- -----------------------------------------------------------------------------
-- v1 → v2 변경 사유 (D1~D3 진단 결과):
--   route_link_sequence.st_node_id / ed_node_id 는 "link 의 양 끝 노드" 이며
--   정류장(7xxxxxxxxx) + 도로/교차로 노드(15.../30007.../36.../73611... 등)
--   혼합 시퀀스이다. v1 은 두 끝이 모두 stop 인 link 만 적재 → coverage
--   = 22.84%. v2 는 route/dir 별로 link_seq 를 순회해 "연속된 두 stop 사이"
--   (leg) 의 모든 link 를 하나의 STOP_TO_STOP 엣지로 접는다.
-- -----------------------------------------------------------------------------
--
-- Leg-aggregation 알고리즘:
--   1. 각 link 에 (st_is_stop, ed_is_stop) flag 태깅
--   2. st_is_stop=TRUE 를 만날 때마다 새 leg_id 증가 (window sum)
--      → 한 leg 는 "한 stop 출발 ~ 다음 stop 도착 직전" 까지의 모든 link
--   3. leg 단위 집계:
--        src_stop_id = leg 첫 link 의 st_node_id         (반드시 stop)
--        dst_stop_id = leg 내 ed_is_stop=T 인 첫 link 의 ed_node_id
--        cum_gis_dist = seg_start_seq ~ seg_end_seq 의 gis_dist 합
--   4. distance_m = coalesce(nullif(cum_gis_dist,0), ST_Distance(geom_5187))
--      time_sec  = distance_m / 1000.0 / 20.0 * 3600.0  (20 km/h 가정)
--   5. 중간 intermediate node 는 graph_edge_master 에 저장하지 않음
--      (stop-to-stop directed edge only 원칙 유지)
--
-- 원칙 (user-approved 2026-04-19):
--   - GATv2 정적 엣지, 방향성 (is_bidirectional=false)
--   - 자기루프 / orphan 제거
--   - 동일 (src_stop, dst_stop) 에 다중 노선이 관여하면 distance_m ASC 로
--     edge_rank 부여, rank=1 → is_primary_edge=true
--   - edge_uid 포맷: STOP_TO_STOP:<route>:<dir>:<src>-><dst>  (v1 과 동일)
--   - Idempotent: ON CONFLICT (edge_uid) DO UPDATE
--   - Stale: 소스에서 사라진 엣지는 is_active=false (DELETE 아님)
-- =============================================================================

begin;

-- -----------------------------------------------------------------------------
-- 0) 로드 전 카운트 스냅샷
-- -----------------------------------------------------------------------------
do $$
declare
    v_before int;
begin
    select count(*) into v_before
    from public.graph_edge_master where edge_type = 'STOP_TO_STOP';
    raise notice 'STOP_TO_STOP rows BEFORE load: %', v_before;
end $$;

-- -----------------------------------------------------------------------------
-- STAGE 1) 각 link 에 stop flag 태깅
-- -----------------------------------------------------------------------------
drop table if exists tmp_rls_tagged;
create temp table tmp_rls_tagged on commit drop as
select
    r.route_id,
    r.move_dir_code::integer            as move_dir_code,
    r.link_seq,
    r.link_id,
    r.st_node_id,
    r.ed_node_id,
    r.gis_dist,
    (d1.stop_id is not null)            as st_is_stop,
    (d2.stop_id is not null)            as ed_is_stop
from public.route_link_sequence r
left join public.dim_stop d1 on d1.stop_id = r.st_node_id
left join public.dim_stop d2 on d2.stop_id = r.ed_node_id
where r.st_node_id is not null
  and r.ed_node_id is not null
  and r.st_node_id <> r.ed_node_id;           -- link 수준 self-loop 제거

create index on tmp_rls_tagged (route_id, move_dir_code, link_seq);

-- -----------------------------------------------------------------------------
-- STAGE 2) leg_id 부여 — st_is_stop=TRUE 마다 증가
-- -----------------------------------------------------------------------------
drop table if exists tmp_rls_legged;
create temp table tmp_rls_legged on commit drop as
select *,
    sum(case when st_is_stop then 1 else 0 end) over (
        partition by route_id, move_dir_code
        order by link_seq
        rows unbounded preceding
    ) as leg_id
from tmp_rls_tagged;

create index on tmp_rls_legged (route_id, move_dir_code, leg_id);

-- -----------------------------------------------------------------------------
-- STAGE 3) leg 단위 집계 — src/dst stop, 누적거리, 대표 link
-- -----------------------------------------------------------------------------
drop table if exists tmp_leg_edges;
create temp table tmp_leg_edges on commit drop as
with leg_aggr as (
    select
        route_id, move_dir_code, leg_id,
        min(link_seq)                                              as seg_start_seq,
        min(link_seq) filter (where ed_is_stop)                    as seg_end_seq,
        (array_agg(st_node_id order by link_seq))[1]               as src_stop_id,
        (array_agg(ed_node_id order by link_seq)
           filter (where ed_is_stop))[1]                           as dst_stop_id,
        (array_agg(link_id order by link_seq))[1]                  as first_link_id
    from tmp_rls_legged
    group by route_id, move_dir_code, leg_id
),
leg_dist as (
    select l.route_id, l.move_dir_code, l.leg_id,
           sum(l.gis_dist)                                         as cum_gis_dist,
           count(*)                                                as link_steps
    from tmp_rls_legged l
    join leg_aggr a
         on  a.route_id      = l.route_id
         and a.move_dir_code = l.move_dir_code
         and a.leg_id        = l.leg_id
    where a.seg_end_seq is not null
      and l.link_seq between a.seg_start_seq and a.seg_end_seq
    group by l.route_id, l.move_dir_code, l.leg_id
)
select
    a.route_id, a.move_dir_code, a.leg_id,
    a.seg_start_seq, a.seg_end_seq,
    a.src_stop_id, a.dst_stop_id,
    a.first_link_id,
    coalesce(d.cum_gis_dist, 0)::numeric  as cum_gis_dist,
    coalesce(d.link_steps, 0)             as link_steps
from leg_aggr a
left join leg_dist d
       on  d.route_id      = a.route_id
       and d.move_dir_code = a.move_dir_code
       and d.leg_id        = a.leg_id
where a.src_stop_id is not null
  and a.dst_stop_id is not null
  and a.seg_end_seq is not null
  and a.src_stop_id <> a.dst_stop_id;

create index on tmp_leg_edges (src_stop_id, dst_stop_id);
create index on tmp_leg_edges (route_id, move_dir_code, leg_id);

do $$
declare
    v_legs int;
    v_avg_steps numeric;
begin
    select count(*), avg(link_steps)::numeric(6,2)
      into v_legs, v_avg_steps from tmp_leg_edges;
    raise notice 'leg edges aggregated : % (avg link_steps per leg: %)',
                 v_legs, v_avg_steps;
end $$;

-- -----------------------------------------------------------------------------
-- STAGE 4) dim_stop 좌표 join, distance_m / time_sec 계산, edge_uid 생성
-- -----------------------------------------------------------------------------
drop table if exists tmp_edge_src;
create temp table tmp_edge_src on commit drop as
select
    ('STOP_TO_STOP:' || le.route_id
       || ':' || le.move_dir_code::text
       || ':' || le.src_stop_id || '->' || le.dst_stop_id)        as edge_uid,
    ('STOP:' || le.src_stop_id)                                    as src_node_uid,
    ('STOP:' || le.dst_stop_id)                                    as dst_node_uid,
    'STOP'                                                         as src_node_type,
    le.src_stop_id                                                 as src_node_id,
    'STOP'                                                         as dst_node_type,
    le.dst_stop_id                                                 as dst_node_id,
    'STOP_TO_STOP'                                                 as edge_type,
    le.route_id,
    le.move_dir_code,
    le.first_link_id                                               as link_id,
    le.seg_start_seq                                               as link_seq,
    le.src_stop_id                                                 as stop_id,
    le.leg_id::integer                                             as stop_seq,
    -- distance_m: 누적 링크 거리 우선, 0/NULL 이면 stop 좌표 직선거리 fallback
    coalesce(
        nullif(le.cum_gis_dist, 0),
        ST_Distance(ds1.geom_5187, ds2.geom_5187)::numeric
    )::numeric(12,3)                                               as distance_m,
    -- time_sec: 평균 20 km/h 가정
    ((coalesce(
        nullif(le.cum_gis_dist, 0),
        ST_Distance(ds1.geom_5187, ds2.geom_5187)::numeric
    ) / 1000.0) / 20.0 * 3600.0)::numeric(12,3)                    as time_sec
from tmp_leg_edges le
inner join public.dim_stop ds1 on ds1.stop_id = le.src_stop_id
inner join public.dim_stop ds2 on ds2.stop_id = le.dst_stop_id
where ds1.geom_5187 is not null
  and ds2.geom_5187 is not null;

create index on tmp_edge_src (edge_uid);
create index on tmp_edge_src (src_node_uid, dst_node_uid);

-- -----------------------------------------------------------------------------
-- STAGE 5) edge_uid 단위 dedup
--     같은 (route, dir, src_stop, dst_stop) 이 회차/순환 노선에서 여러 leg 로
--     반복될 수 있으므로, 가장 이른 leg 1개만 유지.
-- -----------------------------------------------------------------------------
drop table if exists tmp_edge_dedup;
create temp table tmp_edge_dedup on commit drop as
with ranked_uid as (
    select *,
        row_number() over (
            partition by edge_uid
            order by link_seq asc, link_id asc
        ) as uid_rn
    from tmp_edge_src
)
select
    edge_uid,
    src_node_uid, dst_node_uid,
    src_node_type, src_node_id,
    dst_node_type, dst_node_id,
    edge_type,
    route_id, move_dir_code,
    link_id, link_seq,
    stop_id, stop_seq,
    distance_m, time_sec
from ranked_uid
where uid_rn = 1;

create index on tmp_edge_dedup (src_node_uid, dst_node_uid);

do $$
declare
    v_src int; v_dedup int;
begin
    select count(*) into v_src   from tmp_edge_src;
    select count(*) into v_dedup from tmp_edge_dedup;
    raise notice 'tmp_edge_src rows    : %', v_src;
    raise notice 'tmp_edge_dedup rows  : % (collapsed % duplicates)',
                 v_dedup, (v_src - v_dedup);
end $$;

-- -----------------------------------------------------------------------------
-- STAGE 6) edge_rank / is_primary_edge 부여
--     동일 (src_stop, dst_stop) 에서 여러 노선이 관여하면 distance_m ASC 로 rank
-- -----------------------------------------------------------------------------
drop table if exists tmp_edge_ranked;
create temp table tmp_edge_ranked on commit drop as
select
    e.*,
    row_number() over (
        partition by src_node_uid, dst_node_uid
        order by distance_m asc, route_id asc, move_dir_code asc, link_seq asc
    )::smallint as edge_rank
from tmp_edge_dedup e;

-- -----------------------------------------------------------------------------
-- STAGE 7) UPSERT into graph_edge_master
-- -----------------------------------------------------------------------------
insert into public.graph_edge_master (
    edge_uid,
    src_node_uid, dst_node_uid,
    src_node_type, src_node_id,
    dst_node_type, dst_node_id,
    edge_type,
    route_id, move_dir_code,
    link_id, link_seq,
    stop_id, stop_seq,
    distance_m, time_sec,
    is_bidirectional, is_active,
    is_primary_edge, edge_rank,
    created_at, updated_at
)
select
    edge_uid,
    src_node_uid, dst_node_uid,
    src_node_type, src_node_id,
    dst_node_type, dst_node_id,
    edge_type,
    route_id, move_dir_code,
    link_id, link_seq,
    stop_id, stop_seq,
    distance_m, time_sec,
    false            as is_bidirectional,
    true             as is_active,
    (edge_rank = 1)  as is_primary_edge,
    edge_rank,
    now(), now()
from tmp_edge_ranked
on conflict (edge_uid) do update set
    src_node_uid    = excluded.src_node_uid,
    dst_node_uid    = excluded.dst_node_uid,
    src_node_type   = excluded.src_node_type,
    src_node_id     = excluded.src_node_id,
    dst_node_type   = excluded.dst_node_type,
    dst_node_id     = excluded.dst_node_id,
    edge_type       = excluded.edge_type,
    route_id        = excluded.route_id,
    move_dir_code   = excluded.move_dir_code,
    link_id         = excluded.link_id,
    link_seq        = excluded.link_seq,
    stop_id         = excluded.stop_id,
    stop_seq        = excluded.stop_seq,
    distance_m      = excluded.distance_m,
    time_sec        = excluded.time_sec,
    is_primary_edge = excluded.is_primary_edge,
    edge_rank       = excluded.edge_rank,
    is_active       = true,
    updated_at      = now();

-- -----------------------------------------------------------------------------
-- STAGE 8) stale 처리 — 소스에 없어진 엣지는 is_active=false
-- -----------------------------------------------------------------------------
update public.graph_edge_master g
   set is_active = false,
       updated_at = now()
 where g.edge_type = 'STOP_TO_STOP'
   and not exists (
        select 1 from tmp_edge_ranked t where t.edge_uid = g.edge_uid
   )
   and g.is_active = true;

-- -----------------------------------------------------------------------------
-- STAGE 9) 로드 후 카운트
-- -----------------------------------------------------------------------------
do $$
declare
    v_after int;
    v_primary int;
    v_inactive int;
begin
    select count(*) into v_after
      from public.graph_edge_master
     where edge_type = 'STOP_TO_STOP';
    select count(*) into v_primary
      from public.graph_edge_master
     where edge_type = 'STOP_TO_STOP' and is_primary_edge = true;
    select count(*) into v_inactive
      from public.graph_edge_master
     where edge_type = 'STOP_TO_STOP' and is_active = false;
    raise notice 'STOP_TO_STOP rows AFTER  load : %', v_after;
    raise notice 'STOP_TO_STOP primary edges    : %', v_primary;
    raise notice 'STOP_TO_STOP inactive (stale) : %', v_inactive;
end $$;

commit;
