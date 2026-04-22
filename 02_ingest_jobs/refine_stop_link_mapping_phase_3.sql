-- ============================================================================
-- refine_stop_link_mapping_phase_3.sql
-- 목적:
--   Integrated Graph Master Phase 3로서
--   정류장-링크 1차 공간 매핑(SNAP_NEAREST)을
--   route-aware(노선 인지형) 기준으로 2차 정밀 보정한다.
--
-- 핵심 원칙:
--   1) 기존 SNAP_NEAREST 결과를 삭제하지 않는다.
--   2) route_id + move_dir_code + sequence(순서) 정합성을 반영한다.
--   3) stop_link_mapping_master 에 ROUTE_INFERRED 행을 누적한다.
--   4) graph_edge_master 에 route-aware 서비스 간선을 추가한다.
--
-- 전제:
--   - public.dim_stop
--   - public.link_master_candidate_vw
--   - public.route_link_sequence
--   - public.stop_link_mapping_master
--   - public.graph_node_master
--   - public.graph_edge_master
--
-- 참고:
--   stop-route-direction 순서 원천은 아래 후보 중 자동 탐지한다.
--   - public.stop_route_sequence
--   - public.route_stop_sequence
--   - public.stg_daegu_stop_route_sequence
--   - public.vw_stop_route_sequence
--
-- 작성일: 2026-04-10
-- ============================================================================

begin;

-- --------------------------------------------------------------------------
-- 0) 필수 객체 점검
-- --------------------------------------------------------------------------
do $$
begin
    if to_regclass('public.dim_stop') is null then
        raise exception 'Required relation missing: public.dim_stop';
    end if;

    if to_regclass('public.link_master_candidate_vw') is null then
        raise exception 'Required relation missing: public.link_master_candidate_vw';
    end if;

    if to_regclass('public.route_link_sequence') is null then
        raise exception 'Required relation missing: public.route_link_sequence';
    end if;

    if to_regclass('public.stop_link_mapping_master') is null then
        raise exception 'Required relation missing: public.stop_link_mapping_master';
    end if;

    if to_regclass('public.graph_node_master') is null then
        raise exception 'Required relation missing: public.graph_node_master';
    end if;

    if to_regclass('public.graph_edge_master') is null then
        raise exception 'Required relation missing: public.graph_edge_master';
    end if;
end $$;

-- --------------------------------------------------------------------------
-- 1) stop-route-direction 순서 원천 자동 탐지
-- --------------------------------------------------------------------------
drop table if exists _phase3_stop_route_source;

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
        create temp table _phase3_stop_route_source as
        select
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

    raise notice 'Phase 3 stop-route source detected: %', v_source;
end $$;

-- --------------------------------------------------------------------------
-- 2) 정규화된 stop / route-link sequence 준비
-- --------------------------------------------------------------------------
drop table if exists _phase3_stop_route_norm;
create temp table _phase3_stop_route_norm as
with base as (
    select distinct
        stop_id,
        route_id,
        move_dir_code,
        stop_seq
    from _phase3_stop_route_source
)
select
    b.stop_id,
    b.route_id,
    b.move_dir_code,
    b.stop_seq,
    max(b.stop_seq) over (partition by b.route_id, b.move_dir_code) as max_stop_seq,
    case
        when max(b.stop_seq) over (partition by b.route_id, b.move_dir_code) > 1
            then ((b.stop_seq - 1)::numeric /
                 nullif((max(b.stop_seq) over (partition by b.route_id, b.move_dir_code) - 1)::numeric, 0))::numeric(12,6)
        else 0::numeric(12,6)
    end as stop_seq_norm
from base b;

create index on _phase3_stop_route_norm (route_id, move_dir_code, stop_id);
create index on _phase3_stop_route_norm (route_id, move_dir_code, stop_seq);

drop table if exists _phase3_route_link_norm;
create temp table _phase3_route_link_norm as
with base as (
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
)
select
    b.route_id,
    b.move_dir_code,
    b.link_seq,
    b.link_id,
    max(b.link_seq) over (partition by b.route_id, b.move_dir_code) as max_link_seq,
    case
        when max(b.link_seq) over (partition by b.route_id, b.move_dir_code) > 1
            then ((b.link_seq - 1)::numeric /
                 nullif((max(b.link_seq) over (partition by b.route_id, b.move_dir_code) - 1)::numeric, 0))::numeric(12,6)
        else 0::numeric(12,6)
    end as link_seq_norm
from base b;

create index on _phase3_route_link_norm (route_id, move_dir_code, link_id);
create index on _phase3_route_link_norm (route_id, move_dir_code, link_seq);

-- --------------------------------------------------------------------------
-- 3) Phase 2의 SNAP_NEAREST 기준 1차 primary 매핑 준비
-- --------------------------------------------------------------------------
drop table if exists _phase3_current_snap;
create temp table _phase3_current_snap as
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

create index on _phase3_current_snap (stop_id);

-- --------------------------------------------------------------------------
-- 4) route-aware 후보 생성
--    반경: 150m 우선 + 500m fallback(대체 보완 경로)
-- --------------------------------------------------------------------------
drop table if exists _phase3_candidate_raw;
create temp table _phase3_candidate_raw as
select
    s.stop_id,
    s.route_id,
    s.move_dir_code,
    s.stop_seq,
    s.stop_seq_norm,

    r.link_id,
    r.link_seq,
    r.link_seq_norm,

    snap.link_id as snap_link_id,
    snap.distance_to_link_m as snap_distance_to_link_m,

    round(ST_Distance(ds.geom_5187, l.geom_5187)::numeric, 3) as distance_to_link_m,
    case
        when ST_DWithin(ds.geom_5187, l.geom_5187, 150.0) then 1
        else 2
    end as radius_tier,
    round(abs(s.stop_seq_norm - r.link_seq_norm)::numeric, 6) as seq_alignment_gap,
    case when snap.link_id is not null and snap.link_id = r.link_id then true else false end as same_as_snap
from _phase3_stop_route_norm s
join public.dim_stop ds
  on cast(ds.stop_id as text) = s.stop_id
 and ds.geom_5187 is not null
join _phase3_route_link_norm r
  on r.route_id = s.route_id
 and r.move_dir_code = s.move_dir_code
join public.link_master_candidate_vw l
  on cast(l.link_id as text) = r.link_id
 and l.geom_5187 is not null
left join _phase3_current_snap snap
  on snap.stop_id = s.stop_id
where ST_DWithin(ds.geom_5187, l.geom_5187, 500.0);

create index on _phase3_candidate_raw (stop_id, route_id, move_dir_code);
create index on _phase3_candidate_raw (route_id, move_dir_code, link_id);

-- --------------------------------------------------------------------------
-- 5) 후보 순위화
--    우선순위:
--      1) 150m 우선
--      2) 기존 SNAP_NEAREST 링크와 동일하면 우선
--      3) sequence 정합성 우선
--      4) 거리 우선
-- --------------------------------------------------------------------------
drop table if exists _phase3_candidate_ranked;
create temp table _phase3_candidate_ranked as
select
    c.*,
    row_number() over (
        partition by c.stop_id, c.route_id, c.move_dir_code
        order by
            c.radius_tier asc,
            case when c.same_as_snap then 0 else 1 end asc,
            c.seq_alignment_gap asc,
            c.distance_to_link_m asc,
            c.link_seq asc,
            c.link_id asc
    ) as rn
from _phase3_candidate_raw c;

create index on _phase3_candidate_ranked (stop_id, route_id, move_dir_code, rn);

-- --------------------------------------------------------------------------
-- 6) route-aware 최종 선택 및 품질 등급 산정
--    주의:
--      품질 등급은 검증 및 운영 판정용이며,
--      stop_link_mapping_master 에는 ROUTE_INFERRED만 적재한다.
-- --------------------------------------------------------------------------
drop table if exists _phase3_selected_final;
create temp table _phase3_selected_final as
select
    c.stop_id,
    c.route_id,
    c.move_dir_code,
    c.stop_seq,
    c.stop_seq_norm,
    c.link_id,
    c.link_seq,
    c.link_seq_norm,
    c.snap_link_id,
    c.snap_distance_to_link_m,
    c.distance_to_link_m,
    c.radius_tier,
    c.seq_alignment_gap,
    c.same_as_snap,
    case
        when c.radius_tier = 1
         and c.distance_to_link_m <= 100
         and c.seq_alignment_gap <= 0.20
            then 'APPROVED'
        when c.distance_to_link_m <= 250
         and c.seq_alignment_gap <= 0.35
            then 'HOLD'
        else 'MANUAL_REVIEW'
    end as quality_grade,
    case
        when c.distance_to_link_m <= 15 then 1.000000
        when c.distance_to_link_m <= 30 then 0.950000
        when c.distance_to_link_m <= 50 then 0.850000
        when c.distance_to_link_m <= 100 then 0.700000
        else 0.500000
    end::numeric(10,6) as confidence_score
from _phase3_candidate_ranked c
where c.rn = 1;

create index on _phase3_selected_final (stop_id, route_id, move_dir_code);
create index on _phase3_selected_final (quality_grade);

-- --------------------------------------------------------------------------
-- 7) 현재 단계에서 승격 가능한 대상 선별
--    - APPROVED / HOLD 는 ROUTE_INFERRED로 승격 가능
--    - MANUAL_REVIEW 는 검증 리포트에서만 관리
-- --------------------------------------------------------------------------
drop table if exists _phase3_promotable;
create temp table _phase3_promotable as
select *
from _phase3_selected_final
where quality_grade in ('APPROVED', 'HOLD');

create index on _phase3_promotable (stop_id, route_id, move_dir_code);

-- --------------------------------------------------------------------------
-- 8) 기존 route-aware primary 정리
--    동일 stop_id + route_id + move_dir_code 에서 새 link_id가 있으면,
--    기존 route-aware primary 는 비활성화한다.
-- --------------------------------------------------------------------------
update public.stop_link_mapping_master m
set
    is_primary_mapping = false,
    is_active = false,
    updated_at = now()
from _phase3_promotable p
where m.stop_id = p.stop_id
  and coalesce(m.route_id, '') = p.route_id
  and coalesce(m.move_dir_code, -1) = p.move_dir_code
  and m.mapping_type = 'ROUTE_INFERRED'
  and m.is_active = true
  and m.is_primary_mapping = true
  and m.link_id <> p.link_id;

-- --------------------------------------------------------------------------
-- 9) 동일 natural key(자연키) 갱신
-- --------------------------------------------------------------------------
update public.stop_link_mapping_master m
set
    mapping_type = 'ROUTE_INFERRED',
    distance_to_link_m = p.distance_to_link_m,
    confidence_score = p.confidence_score,
    route_id = p.route_id,
    move_dir_code = p.move_dir_code,
    is_primary_mapping = true,
    is_active = true,
    updated_at = now()
from _phase3_promotable p
where m.stop_id = p.stop_id
  and m.link_id = p.link_id
  and coalesce(m.route_id, '') = p.route_id
  and coalesce(m.move_dir_code, -1) = p.move_dir_code;

-- --------------------------------------------------------------------------
-- 10) 없는 route-aware 행 신규 적재
-- --------------------------------------------------------------------------
insert into public.stop_link_mapping_master (
    stop_id,
    link_id,
    mapping_type,
    distance_to_link_m,
    projection_ratio,
    confidence_score,
    route_id,
    move_dir_code,
    is_primary_mapping,
    is_active
)
select
    p.stop_id,
    p.link_id,
    'ROUTE_INFERRED' as mapping_type,
    p.distance_to_link_m,
    null as projection_ratio,
    p.confidence_score,
    p.route_id,
    p.move_dir_code,
    true as is_primary_mapping,
    true as is_active
from _phase3_promotable p
where not exists (
    select 1
    from public.stop_link_mapping_master m
    where m.stop_id = p.stop_id
      and m.link_id = p.link_id
      and coalesce(m.route_id, '') = p.route_id
      and coalesce(m.move_dir_code, -1) = p.move_dir_code
);

-- --------------------------------------------------------------------------
-- 11) route-aware 간선 정리
--     기존 route-aware 서비스 간선 중 현재 선택과 다른 link는 비활성화
-- --------------------------------------------------------------------------
update public.graph_edge_master e
set
    is_active = false,
    updated_at = now()
from _phase3_promotable p
where e.edge_type = 'STOP_TO_LINK'
  and coalesce(e.route_id, '') = p.route_id
  and coalesce(e.move_dir_code, -1) = p.move_dir_code
  and e.stop_id = p.stop_id
  and e.is_active = true
  and e.link_id <> p.link_id;

update public.graph_edge_master e
set
    is_active = false,
    updated_at = now()
from _phase3_promotable p
where e.edge_type = 'LINK_TO_STOP'
  and coalesce(e.route_id, '') = p.route_id
  and coalesce(e.move_dir_code, -1) = p.move_dir_code
  and e.stop_id = p.stop_id
  and e.is_active = true
  and e.link_id <> p.link_id;

-- --------------------------------------------------------------------------
-- 12) route-aware STOP_TO_LINK 간선 upsert(갱신 또는 삽입)
-- --------------------------------------------------------------------------
update public.graph_edge_master e
set
    src_node_uid = 'STOP:' || p.stop_id,
    dst_node_uid = 'LINK:' || p.link_id,
    src_node_type = 'STOP',
    src_node_id = p.stop_id,
    dst_node_type = 'LINK',
    dst_node_id = p.link_id,
    edge_type = 'STOP_TO_LINK',
    route_id = p.route_id,
    move_dir_code = p.move_dir_code,
    link_id = p.link_id,
    stop_id = p.stop_id,
    distance_m = p.distance_to_link_m,
    generalized_cost = p.distance_to_link_m,
    is_bidirectional = false,
    is_active = true,
    updated_at = now()
from _phase3_promotable p
where e.edge_uid =
      'STOP_TO_LINK:R' || p.route_id || ':D' || p.move_dir_code::text || ':' || p.stop_id || '->' || p.link_id;

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
    stop_id,
    stop_seq,
    distance_m,
    time_sec,
    generalized_cost,
    is_bidirectional,
    is_active
)
select
    'STOP_TO_LINK:R' || p.route_id || ':D' || p.move_dir_code::text || ':' || p.stop_id || '->' || p.link_id as edge_uid,
    'STOP:' || p.stop_id as src_node_uid,
    'LINK:' || p.link_id as dst_node_uid,
    'STOP' as src_node_type,
    p.stop_id as src_node_id,
    'LINK' as dst_node_type,
    p.link_id as dst_node_id,
    'STOP_TO_LINK' as edge_type,
    p.route_id,
    p.move_dir_code,
    p.link_id,
    p.link_seq,
    p.stop_id,
    p.stop_seq,
    p.distance_to_link_m,
    null as time_sec,
    p.distance_to_link_m as generalized_cost,
    false as is_bidirectional,
    true as is_active
from _phase3_promotable p
join public.graph_node_master ns
  on ns.node_uid = 'STOP:' || p.stop_id
join public.graph_node_master nl
  on nl.node_uid = 'LINK:' || p.link_id
where not exists (
    select 1
    from public.graph_edge_master e
    where e.edge_uid =
          'STOP_TO_LINK:R' || p.route_id || ':D' || p.move_dir_code::text || ':' || p.stop_id || '->' || p.link_id
);

-- --------------------------------------------------------------------------
-- 13) route-aware LINK_TO_STOP 간선 upsert(갱신 또는 삽입)
-- --------------------------------------------------------------------------
update public.graph_edge_master e
set
    src_node_uid = 'LINK:' || p.link_id,
    dst_node_uid = 'STOP:' || p.stop_id,
    src_node_type = 'LINK',
    src_node_id = p.link_id,
    dst_node_type = 'STOP',
    dst_node_id = p.stop_id,
    edge_type = 'LINK_TO_STOP',
    route_id = p.route_id,
    move_dir_code = p.move_dir_code,
    link_id = p.link_id,
    stop_id = p.stop_id,
    distance_m = p.distance_to_link_m,
    generalized_cost = p.distance_to_link_m,
    is_bidirectional = false,
    is_active = true,
    updated_at = now()
from _phase3_promotable p
where e.edge_uid =
      'LINK_TO_STOP:R' || p.route_id || ':D' || p.move_dir_code::text || ':' || p.link_id || '->' || p.stop_id;

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
    stop_id,
    stop_seq,
    distance_m,
    time_sec,
    generalized_cost,
    is_bidirectional,
    is_active
)
select
    'LINK_TO_STOP:R' || p.route_id || ':D' || p.move_dir_code::text || ':' || p.link_id || '->' || p.stop_id as edge_uid,
    'LINK:' || p.link_id as src_node_uid,
    'STOP:' || p.stop_id as dst_node_uid,
    'LINK' as src_node_type,
    p.link_id as src_node_id,
    'STOP' as dst_node_type,
    p.stop_id as dst_node_id,
    'LINK_TO_STOP' as edge_type,
    p.route_id,
    p.move_dir_code,
    p.link_id,
    p.link_seq,
    p.stop_id,
    p.stop_seq,
    p.distance_to_link_m,
    null as time_sec,
    p.distance_to_link_m as generalized_cost,
    false as is_bidirectional,
    true as is_active
from _phase3_promotable p
join public.graph_node_master nl
  on nl.node_uid = 'LINK:' || p.link_id
join public.graph_node_master ns
  on ns.node_uid = 'STOP:' || p.stop_id
where not exists (
    select 1
    from public.graph_edge_master e
    where e.edge_uid =
          'LINK_TO_STOP:R' || p.route_id || ':D' || p.move_dir_code::text || ':' || p.link_id || '->' || p.stop_id
);

-- --------------------------------------------------------------------------
-- 14) 실행 요약 출력용 결과 셋
-- --------------------------------------------------------------------------
select
    'phase3_selected_final' as metric_group,
    quality_grade as metric_name,
    count(*)::bigint as metric_value
from _phase3_selected_final
group by quality_grade
order by quality_grade;

select
    'phase3_promoted_route_inferred' as metric_group,
    count(*)::bigint as promoted_count
from _phase3_promotable;

commit;
