-- =============================================================================
-- D3_link_vs_stop_hypothesis.sql
-- Purpose : Stage 1~2 결과로 "가설 D — route_link_sequence 는 stop +
--           intermediate link node 혼합 시퀀스" 가 유력해짐. 이 가설을
--           확정/기각하기 위한 진단.
-- 조회 전용 (DDL/DML 없음).
-- =============================================================================

\pset pager off
\timing off

-- tmp_missing_stop 재구성
drop table if exists tmp_missing_stop;
create temp table tmp_missing_stop as
with st_miss as (
    select distinct r.st_node_id as missing_id
      from public.route_link_sequence r
      left join public.dim_stop d on d.stop_id = r.st_node_id
     where r.st_node_id is not null and d.stop_id is null
),
ed_miss as (
    select distinct r.ed_node_id as missing_id
      from public.route_link_sequence r
      left join public.dim_stop d on d.stop_id = r.ed_node_id
     where r.ed_node_id is not null and d.stop_id is null
)
select distinct missing_id from (
    select missing_id from st_miss union all
    select missing_id from ed_miss
) u;
create index on tmp_missing_stop (missing_id);

\echo ''
\echo '============================================================'
\echo ' T1. missing_id 접두사(2자리/4자리) 분포'
\echo '============================================================'
select substring(missing_id from 1 for 4) as prefix4,
       substring(missing_id from 1 for 2) as prefix2,
       count(*)                             as distinct_cnt
  from tmp_missing_stop
 group by 1, 2
 order by distinct_cnt desc
 limit 30;

\echo ''
\echo '============================================================'
\echo ' T2. dim_stop.stop_id 접두사(2자리/4자리) 분포 (비교용)'
\echo '============================================================'
select substring(stop_id from 1 for 4) as prefix4,
       substring(stop_id from 1 for 2) as prefix2,
       count(*)                         as cnt
  from public.dim_stop
 group by 1, 2
 order by cnt desc
 limit 30;

\echo ''
\echo '============================================================'
\echo ' T3. graph_node_master 의 node_type 분포'
\echo '     (STOP 외 LINK / NODE 타입이 있으면 혼합 가설을 직접 검증)'
\echo '============================================================'
select node_type,
       count(*)                                                                as cnt,
       count(*) filter (where node_id in (select missing_id from tmp_missing_stop)) as missing_match
  from public.graph_node_master
 group by node_type
 order by cnt desc;

\echo ''
\echo '============================================================'
\echo ' T4. missing_id 가 bs_20250903 / graph_node_master / err_* 어디에도'
\echo '     없는지 교차 확인'
\echo '============================================================'
with m as (select missing_id from tmp_missing_stop)
select
    (select count(*) from m)                                                      as missing_total,
    (select count(*) from m join public.bs_20250903 b      on b.bs_id     = m.missing_id) as in_bs_bsid,
    (select count(*) from m join public.bs_20250903 b      on b.node_id   = m.missing_id) as in_bs_nodeid,
    (select count(*) from m join public.graph_node_master g on g.node_id  = m.missing_id) as in_gnm_nodeid,
    (select count(*) from m join public.graph_node_master g on g.node_uid = m.missing_id) as in_gnm_uid,
    (select count(*) from m join public.err_daegu_stop_usage_mapping_failed e on e.stop_id_raw = m.missing_id) as in_err_stopraw;

\echo ''
\echo '============================================================'
\echo ' T5. 핫스팟 노선 7361109008 dir=1 의 전체 link_seq 덤프'
\echo '     (S = stop, - = non-stop. 교차 패턴이 보이면 가설 D 확정)'
\echo '============================================================'
with seq as (
    select r.link_seq, r.link_id, r.st_node_id, r.ed_node_id,
           case when exists (select 1 from public.dim_stop d where d.stop_id = r.st_node_id) then 'S' else '-' end as st_tag,
           case when exists (select 1 from public.dim_stop d where d.stop_id = r.ed_node_id) then 'S' else '-' end as ed_tag
      from public.route_link_sequence r
     where r.route_id = '7361109008' and r.move_dir_code = '1'
     order by r.link_seq
)
select link_seq,
       st_tag || '→' || ed_tag                  as pattern,
       st_node_id,
       ed_node_id,
       link_id
  from seq
 order by link_seq
 limit 60;

\echo ''
\echo '============================================================'
\echo ' T6. 같은 노선의 pattern 요약 — S→S / S→- / -→S / -→-'
\echo '============================================================'
with seq as (
    select
        case when exists (select 1 from public.dim_stop d where d.stop_id = r.st_node_id) then 'S' else '-' end ||
        '→' ||
        case when exists (select 1 from public.dim_stop d where d.stop_id = r.ed_node_id) then 'S' else '-' end as pattern
      from public.route_link_sequence r
     where r.route_id = '7361109008' and r.move_dir_code = '1'
)
select pattern, count(*) as cnt from seq group by pattern order by cnt desc;

\echo ''
\echo '============================================================'
\echo ' T7. 전체 route_link_sequence 의 S→S / S→- / -→S / -→- 분포'
\echo '============================================================'
with seq as (
    select
        case when exists (select 1 from public.dim_stop d where d.stop_id = r.st_node_id) then 'S' else '-' end ||
        '→' ||
        case when exists (select 1 from public.dim_stop d where d.stop_id = r.ed_node_id) then 'S' else '-' end as pattern
      from public.route_link_sequence r
)
select pattern, count(*) as cnt,
       round(count(*)::numeric * 100.0 / sum(count(*)) over (), 2) as pct
  from seq
 group by pattern
 order by cnt desc;

\echo ''
\echo '============================================================'
\echo ' T8. 노선당 link_seq 수 / 매칭 stop 수 / links_per_stop'
\echo '     (links_per_stop >> 1 이면 intermediate node 다수 존재)'
\echo '============================================================'
with route_stats as (
    select
        r.route_id,
        r.move_dir_code,
        count(*)                                                                 as link_cnt,
        count(distinct case when exists (select 1 from public.dim_stop d where d.stop_id = r.st_node_id)
                            then r.st_node_id end)                               as matched_src_stops,
        count(distinct case when exists (select 1 from public.dim_stop d where d.stop_id = r.ed_node_id)
                            then r.ed_node_id end)                               as matched_dst_stops
    from public.route_link_sequence r
    group by r.route_id, r.move_dir_code
)
select route_id, move_dir_code, link_cnt,
       matched_src_stops, matched_dst_stops,
       round(link_cnt::numeric / nullif(greatest(matched_src_stops, matched_dst_stops), 0), 2)
                                                                             as links_per_stop
  from route_stats
 order by links_per_stop desc nulls last
 limit 30;

\echo ''
\echo '============================================================'
\echo ' T9. 전체 요약 — links_per_stop 분포'
\echo '============================================================'
with route_stats as (
    select r.route_id, r.move_dir_code,
           count(*) as link_cnt,
           count(distinct case when exists (select 1 from public.dim_stop d where d.stop_id = r.st_node_id)
                then r.st_node_id end) as matched_src
      from public.route_link_sequence r
     group by r.route_id, r.move_dir_code
)
select
    count(*)                                                as routes_dir_total,
    count(*) filter (where matched_src = 0)                 as routes_with_no_matched_stop,
    avg(link_cnt::numeric / nullif(matched_src, 0))::numeric(10,2) as avg_links_per_stop,
    min(link_cnt::numeric / nullif(matched_src, 0))::numeric(10,2) as min_links_per_stop,
    max(link_cnt::numeric / nullif(matched_src, 0))::numeric(10,2) as max_links_per_stop
  from route_stats;

\echo ''
\echo '============================================================'
\echo ' T10. missing_id 와 연결된 link_id 의 id 체계(길이/숫자 여부)'
\echo '============================================================'
select length(r.link_id)                                                      as link_id_len,
       (r.link_id ~ '^[0-9]+$')                                               as is_numeric,
       count(distinct r.link_id)                                              as cnt
  from public.route_link_sequence r
 where r.st_node_id in (select missing_id from tmp_missing_stop)
    or r.ed_node_id in (select missing_id from tmp_missing_stop)
 group by 1, 2
 order by 1;

\echo ''
\echo '============================================================'
\echo ' T11. route_link_sequence.link_id 전체 길이 분포 (비교)'
\echo '============================================================'
select length(link_id) as len, count(distinct link_id) as cnt
  from public.route_link_sequence
 group by 1
 order by 1;

\echo ''
\echo '============================================================'
\echo ' T12. 또 다른 핫스팟 노선 3000707000 dir=1 의 link_seq 샘플 40'
\echo '============================================================'
with seq as (
    select r.link_seq, r.st_node_id, r.ed_node_id,
           case when exists (select 1 from public.dim_stop d where d.stop_id = r.st_node_id) then 'S' else '-' end as st_tag,
           case when exists (select 1 from public.dim_stop d where d.stop_id = r.ed_node_id) then 'S' else '-' end as ed_tag
      from public.route_link_sequence r
     where r.route_id = '3000707000' and r.move_dir_code = '1'
     order by r.link_seq
)
select link_seq, st_tag || '→' || ed_tag as pattern, st_node_id, ed_node_id
  from seq
 limit 40;

\echo ''
\echo '============================================================'
\echo ' END OF D3 — 결과 회신 부탁'
\echo '============================================================'
