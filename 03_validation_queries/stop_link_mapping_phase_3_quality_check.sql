-- ============================================================================
-- stop_link_mapping_phase_3_quality_check.sql
-- 목적:
--   Integrated Graph Master Phase 3 결과를 검증한다.
--
-- 검증 범위:
--   1) route-aware 매핑 커버리지
--   2) duplicate primary(중복 주매핑) 여부
--   3) 기존 SNAP_NEAREST 대비 보정 건수
--   4) 반대 방향 링크 의심 건수
--   5) sequence 정합성 이상치
--   6) 거리 분포 비교
--   7) 품질 등급(APPROVED / HOLD / MANUAL_REVIEW) 분포
--   8) route-aware 서비스 간선 무결성
--   9) 수동 검토용 주요 정류장 조회
--
-- 작성일: 2026-04-10
-- ============================================================================

-- --------------------------------------------------------------------------
-- 0) stop-route-direction 원천 자동 탐지
-- --------------------------------------------------------------------------
drop table if exists _q3_stop_route_source;

do $$
declare
    v_source text;
begin
    if to_regclass('public.stop_route_sequence') is not null then
        v_source := 'public.stop_route_sequence';
    elsif to_regclass('public.route_stop_sequence') is not null then
        v_source := 'public.route_stop_sequence';
    elsif to_regclass('public.stg_daegu_stop_route_sequence') is not null then
        v_source := 'public.stg_daegu_stop_route_sequence';
    elsif to_regclass('public.vw_stop_route_sequence') is not null then
        v_source := 'public.vw_stop_route_sequence';
    else
        raise exception 'No stop-route-direction sequence source found. Expected one of: public.stop_route_sequence, public.route_stop_sequence, public.stg_daegu_stop_route_sequence, public.vw_stop_route_sequence';
    end if;

    execute format($fmt$
        create temp table _q3_stop_route_source as
        select distinct
            cast(stop_id as text) as stop_id,
            cast(route_id as text) as route_id,
            cast(move_dir_code as integer) as move_dir_code,
            cast(stop_seq as integer) as stop_seq
        from %s
        where stop_id is not null
          and route_id is not null
          and move_dir_code is not null
          and stop_seq is not null
    $fmt$, v_source);

    raise notice 'Phase 3 validation stop-route source detected: %', v_source;
end $$;

-- --------------------------------------------------------------------------
-- 1) 정규화 준비
-- --------------------------------------------------------------------------
drop table if exists _q3_stop_route_norm;
create temp table _q3_stop_route_norm as
select
    s.*,
    max(s.stop_seq) over (partition by s.route_id, s.move_dir_code) as max_stop_seq,
    case
        when max(s.stop_seq) over (partition by s.route_id, s.move_dir_code) > 1
            then ((s.stop_seq - 1)::numeric /
                 nullif((max(s.stop_seq) over (partition by s.route_id, s.move_dir_code) - 1)::numeric, 0))::numeric(12,6)
        else 0::numeric(12,6)
    end as stop_seq_norm
from _q3_stop_route_source s;

create index on _q3_stop_route_norm (route_id, move_dir_code, stop_id);

drop table if exists _q3_route_link_norm;
create temp table _q3_route_link_norm as
select
    r.*,
    max(r.link_seq) over (partition by r.route_id, r.move_dir_code) as max_link_seq,
    case
        when max(r.link_seq) over (partition by r.route_id, r.move_dir_code) > 1
            then ((r.link_seq - 1)::numeric /
                 nullif((max(r.link_seq) over (partition by r.route_id, r.move_dir_code) - 1)::numeric, 0))::numeric(12,6)
        else 0::numeric(12,6)
    end as link_seq_norm
from (
    select distinct
        cast(route_id as text) as route_id,
        cast(move_dir_code as integer) as move_dir_code,
        cast(link_seq as integer) as link_seq,
        cast(link_id as text) as link_id
    from public.route_link_sequence
    where route_id is not null
      and move_dir_code is not null
      and link_seq is not null
      and link_id is not null
) r;

create index on _q3_route_link_norm (route_id, move_dir_code, link_id);

drop table if exists _q3_snap;
create temp table _q3_snap as
select
    m.stop_id,
    m.link_id,
    m.distance_to_link_m,
    m.confidence_score
from public.stop_link_mapping_master m
where m.mapping_type = 'SNAP_NEAREST'
  and m.is_active = true
  and m.is_primary_mapping = true
  and m.route_id is null
  and m.move_dir_code is null;

create index on _q3_snap (stop_id);

drop table if exists _q3_route_inferred;
create temp table _q3_route_inferred as
select
    m.stop_id,
    m.link_id,
    m.route_id,
    m.move_dir_code,
    m.distance_to_link_m,
    m.confidence_score,
    m.is_primary_mapping,
    m.is_active,
    m.created_at,
    m.updated_at
from public.stop_link_mapping_master m
where m.mapping_type = 'ROUTE_INFERRED'
  and m.is_active = true
  and m.is_primary_mapping = true
  and m.route_id is not null
  and m.move_dir_code is not null;

create index on _q3_route_inferred (stop_id, route_id, move_dir_code);

-- --------------------------------------------------------------------------
-- 2) route-aware 커버리지
-- --------------------------------------------------------------------------
select
    'coverage' as check_group,
    count(*)::bigint as total_stop_route_dir,
    count(ri.stop_id)::bigint as mapped_stop_route_dir,
    (count(*) - count(ri.stop_id))::bigint as unmapped_stop_route_dir,
    round((count(ri.stop_id)::numeric / nullif(count(*)::numeric, 0)) * 100.0, 2) as mapped_pct
from _q3_stop_route_norm s
left join _q3_route_inferred ri
  on ri.stop_id = s.stop_id
 and ri.route_id = s.route_id
 and ri.move_dir_code = s.move_dir_code;

-- --------------------------------------------------------------------------
-- 3) duplicate primary 점검
-- --------------------------------------------------------------------------
select
    stop_id,
    route_id,
    move_dir_code,
    count(*) as primary_cnt
from _q3_route_inferred
group by stop_id, route_id, move_dir_code
having count(*) > 1
order by primary_cnt desc, stop_id, route_id, move_dir_code;

-- --------------------------------------------------------------------------
-- 4) route_id + move_dir_code + link_id 일치율
-- --------------------------------------------------------------------------
select
    'route_dir_link_match_rate' as check_name,
    count(*)::bigint as total_route_inferred,
    count(*) filter (
        where exists (
            select 1
            from public.route_link_sequence r
            where cast(r.route_id as text) = ri.route_id
              and cast(r.move_dir_code as integer) = ri.move_dir_code
              and cast(r.link_id as text) = ri.link_id
        )
    )::bigint as exact_route_dir_link_match,
    round(
        (
            count(*) filter (
                where exists (
                    select 1
                    from public.route_link_sequence r
                    where cast(r.route_id as text) = ri.route_id
                      and cast(r.move_dir_code as integer) = ri.move_dir_code
                      and cast(r.link_id as text) = ri.link_id
                )
            )::numeric / nullif(count(*)::numeric, 0)
        ) * 100.0,
        2
    ) as exact_match_pct
from _q3_route_inferred ri;

-- --------------------------------------------------------------------------
-- 5) 기존 SNAP_NEAREST 대비 보정 건수
-- --------------------------------------------------------------------------
select
    'correction_vs_snap' as check_name,
    count(*)::bigint as total_route_inferred,
    count(*) filter (where s.link_id = ri.link_id)::bigint as same_as_snap_cnt,
    count(*) filter (where s.link_id is not null and s.link_id <> ri.link_id)::bigint as corrected_cnt,
    count(*) filter (where s.link_id is null)::bigint as no_snap_baseline_cnt
from _q3_route_inferred ri
left join _q3_snap s
  on s.stop_id = ri.stop_id;

-- --------------------------------------------------------------------------
-- 6) 반대 방향 링크 의심 건수
--    정의:
--      generic SNAP 링크가 동일 route_id에 존재하되,
--      selected move_dir_code가 아닌 다른 방향에서만 발견되는 경우를 의심 대상으로 본다.
-- --------------------------------------------------------------------------
with suspected as (
    select
        ri.stop_id,
        ri.route_id,
        ri.move_dir_code as selected_move_dir_code,
        s.link_id as snap_link_id
    from _q3_route_inferred ri
    join _q3_snap s
      on s.stop_id = ri.stop_id
    where s.link_id <> ri.link_id
      and exists (
            select 1
            from public.route_link_sequence r_opp
            where cast(r_opp.route_id as text) = ri.route_id
              and cast(r_opp.link_id as text) = s.link_id
              and cast(r_opp.move_dir_code as integer) <> ri.move_dir_code
      )
      and not exists (
            select 1
            from public.route_link_sequence r_same
            where cast(r_same.route_id as text) = ri.route_id
              and cast(r_same.link_id as text) = s.link_id
              and cast(r_same.move_dir_code as integer) = ri.move_dir_code
      )
)
select
    'opposite_direction_suspected' as check_name,
    count(*)::bigint as suspected_cnt
from suspected;

select *
from (
    with suspected as (
        select
            ri.stop_id,
            ri.route_id,
            ri.move_dir_code as selected_move_dir_code,
            s.link_id as snap_link_id,
            ri.link_id as selected_link_id
        from _q3_route_inferred ri
        join _q3_snap s
          on s.stop_id = ri.stop_id
        where s.link_id <> ri.link_id
          and exists (
                select 1
                from public.route_link_sequence r_opp
                where cast(r_opp.route_id as text) = ri.route_id
                  and cast(r_opp.link_id as text) = s.link_id
                  and cast(r_opp.move_dir_code as integer) <> ri.move_dir_code
          )
          and not exists (
                select 1
                from public.route_link_sequence r_same
                where cast(r_same.route_id as text) = ri.route_id
                  and cast(r_same.link_id as text) = s.link_id
                  and cast(r_same.move_dir_code as integer) = ri.move_dir_code
          )
    )
    select * from suspected
) q
order by route_id, stop_id
limit 100;

-- --------------------------------------------------------------------------
-- 7) sequence 정합성 계산 및 이상치
-- --------------------------------------------------------------------------
drop table if exists _q3_seq_alignment;
create temp table _q3_seq_alignment as
select
    ri.stop_id,
    ri.route_id,
    ri.move_dir_code,
    sr.stop_seq,
    sr.stop_seq_norm,
    rl.link_id,
    rl.link_seq,
    rl.link_seq_norm,
    ri.distance_to_link_m,
    round(abs(sr.stop_seq_norm - rl.link_seq_norm)::numeric, 6) as seq_alignment_gap,
    case
        when ri.distance_to_link_m <= 100 and abs(sr.stop_seq_norm - rl.link_seq_norm) <= 0.20 then 'APPROVED'
        when ri.distance_to_link_m <= 250 and abs(sr.stop_seq_norm - rl.link_seq_norm) <= 0.35 then 'HOLD'
        else 'MANUAL_REVIEW'
    end as quality_grade
from _q3_route_inferred ri
join _q3_stop_route_norm sr
  on sr.stop_id = ri.stop_id
 and sr.route_id = ri.route_id
 and sr.move_dir_code = ri.move_dir_code
join _q3_route_link_norm rl
  on rl.route_id = ri.route_id
 and rl.move_dir_code = ri.move_dir_code
 and rl.link_id = ri.link_id;

create index on _q3_seq_alignment (quality_grade);

select
    'quality_grade_distribution' as check_name,
    quality_grade,
    count(*)::bigint as cnt
from _q3_seq_alignment
group by quality_grade
order by quality_grade;

select
    'sequence_outlier_summary' as check_name,
    count(*) filter (where seq_alignment_gap > 0.20)::bigint as gap_over_020,
    count(*) filter (where seq_alignment_gap > 0.35)::bigint as gap_over_035,
    max(seq_alignment_gap) as max_gap
from _q3_seq_alignment;

select
    stop_id,
    route_id,
    move_dir_code,
    stop_seq,
    link_id,
    link_seq,
    distance_to_link_m,
    seq_alignment_gap,
    quality_grade
from _q3_seq_alignment
where seq_alignment_gap > 0.35
order by seq_alignment_gap desc, distance_to_link_m desc
limit 100;

-- --------------------------------------------------------------------------
-- 8) 거리 분포 비교: SNAP_NEAREST vs ROUTE_INFERRED
-- --------------------------------------------------------------------------
with dist_union as (
    select 'SNAP_NEAREST'::text as mapping_type, distance_to_link_m::numeric as dist_m
    from _q3_snap
    union all
    select 'ROUTE_INFERRED'::text as mapping_type, distance_to_link_m::numeric as dist_m
    from _q3_route_inferred
)
select
    mapping_type,
    count(*)::bigint as row_cnt,
    round(avg(dist_m), 3) as avg_distance_m,
    percentile_cont(0.50) within group (order by dist_m) as median_distance_m,
    percentile_cont(0.95) within group (order by dist_m) as p95_distance_m,
    max(dist_m) as max_distance_m,
    count(*) filter (where dist_m > 100)::bigint as cnt_over_100m,
    count(*) filter (where dist_m > 250)::bigint as cnt_over_250m
from dist_union
group by mapping_type
order by mapping_type;

-- --------------------------------------------------------------------------
-- 9) route-aware 간선 무결성
-- --------------------------------------------------------------------------
select
    'route_aware_edge_counts' as check_name,
    count(*) filter (where edge_type = 'STOP_TO_LINK')::bigint as stop_to_link_cnt,
    count(*) filter (where edge_type = 'LINK_TO_STOP')::bigint as link_to_stop_cnt
from public.graph_edge_master
where is_active = true
  and route_id is not null
  and move_dir_code is not null
  and edge_type in ('STOP_TO_LINK', 'LINK_TO_STOP');

select
    e.edge_uid,
    e.src_node_uid,
    e.dst_node_uid,
    e.edge_type,
    e.route_id,
    e.move_dir_code,
    e.link_id,
    e.stop_id
from public.graph_edge_master e
left join public.graph_node_master ns
  on ns.node_uid = e.src_node_uid
left join public.graph_node_master nd
  on nd.node_uid = e.dst_node_uid
where e.is_active = true
  and e.route_id is not null
  and e.move_dir_code is not null
  and e.edge_type in ('STOP_TO_LINK', 'LINK_TO_STOP')
  and (ns.node_uid is null or nd.node_uid is null)
order by e.edge_uid
limit 100;

-- --------------------------------------------------------------------------
-- 10) 수동 검토용 상위 과거리 매핑
-- --------------------------------------------------------------------------
select
    ri.stop_id,
    ds.stop_name,
    ri.route_id,
    ri.move_dir_code,
    ri.link_id,
    ri.distance_to_link_m,
    sa.seq_alignment_gap,
    sa.quality_grade
from _q3_route_inferred ri
left join public.dim_stop ds
  on cast(ds.stop_id as text) = ri.stop_id
left join _q3_seq_alignment sa
  on sa.stop_id = ri.stop_id
 and sa.route_id = ri.route_id
 and sa.move_dir_code = ri.move_dir_code
 and sa.link_id = ri.link_id
where ri.distance_to_link_m > 100
order by ri.distance_to_link_m desc, sa.seq_alignment_gap desc nulls last
limit 100;

-- --------------------------------------------------------------------------
-- 11) 주요 정류장 수동 점검 도우미
--     검토 기준:
--       1) 정류장-링크 거리가 상식적인지
--       2) 반대편 차로 링크에 잘못 붙지 않았는지
--       3) 환승센터 내부 보행공간이 아니라 실제 버스 통행 링크인지
-- --------------------------------------------------------------------------
select
    ri.stop_id,
    ds.stop_name,
    ri.route_id,
    ri.move_dir_code,
    ri.link_id,
    ri.distance_to_link_m,
    sa.link_seq,
    sa.seq_alignment_gap,
    sa.quality_grade
from _q3_route_inferred ri
join public.dim_stop ds
  on cast(ds.stop_id as text) = ri.stop_id
left join _q3_seq_alignment sa
  on sa.stop_id = ri.stop_id
 and sa.route_id = ri.route_id
 and sa.move_dir_code = ri.move_dir_code
 and sa.link_id = ri.link_id
where ds.stop_name like any (array[
    '%대구역%',
    '%동대구역%',
    '%서부정류장%',
    '%반월당%'
])
order by ds.stop_name, ri.route_id, ri.move_dir_code;
