-- =============================================================================
-- D_diagnose_stop_id_mapping.sql
-- Purpose : route_link_sequence 와 dim_stop 사이 stop_id 매핑이 22.84% 밖에
--           안 되는 원인 규명. 본 파일은 **조회 전용** — 데이터를 수정하지 않음.
-- Usage   : psql -h localhost -U postgres -d urbanbus \
--               -v ON_ERROR_STOP=0 \
--               -f D_diagnose_stop_id_mapping.sql \
--             | Tee-Object D_diagnose_stop_id_mapping.log
--           (ON_ERROR_STOP=0 로 둬서 단일 섹션 실패가 전체를 중단시키지 않도록)
-- Author  : urbanbus_rl_project
-- Date    : 2026-04-19
-- =============================================================================

\pset pager off
\timing off
\echo ''
\echo '###############################################################'
\echo '# D1. dim_stop 컬럼 리스트 (매칭 후보 키 탐색용)'
\echo '###############################################################'
select ordinal_position as pos,
       column_name,
       data_type,
       character_maximum_length as maxlen,
       is_nullable
  from information_schema.columns
 where table_schema = 'public' and table_name = 'dim_stop'
 order by ordinal_position;

\echo ''
\echo '###############################################################'
\echo '# D2. route_link_sequence 컬럼 리스트 (참조용)'
\echo '###############################################################'
select ordinal_position as pos,
       column_name,
       data_type,
       character_maximum_length as maxlen,
       is_nullable
  from information_schema.columns
 where table_schema = 'public' and table_name = 'route_link_sequence'
 order by ordinal_position;

\echo ''
\echo '###############################################################'
\echo '# D3. stop / node 관련 public 테이블 전체 목록'
\echo '#     (staging / source / dim / fact 가능한 테이블을 넓게 캐치)'
\echo '###############################################################'
select table_name,
       (select count(*) from information_schema.columns c
         where c.table_schema = 'public' and c.table_name = t.table_name) as col_cnt
  from information_schema.tables t
 where table_schema = 'public'
   and table_type   = 'BASE TABLE'
   and (table_name ilike '%stop%'
        or table_name ilike '%node%'
        or table_name ilike '%bis%'
        or table_name ilike '%station%'
        or table_name ilike '%bs\_%'  escape '\')
 order by table_name;

\echo ''
\echo '###############################################################'
\echo '# D4. route_link_sequence 내 st_node_id / ed_node_id 기본 분포'
\echo '###############################################################'
select
    count(*)                                                        as rows_total,
    count(*) filter (where st_node_id is null)                      as null_st,
    count(*) filter (where ed_node_id is null)                      as null_ed,
    count(*) filter (where st_node_id = ed_node_id)                 as self_loop,
    count(distinct st_node_id)                                      as distinct_st,
    count(distinct ed_node_id)                                      as distinct_ed
  from public.route_link_sequence;

\echo ''
\echo '###############################################################'
\echo '# D5. missing_stop_id 전체 추출 (st/ed 를 UNION, DISTINCT)'
\echo '#     결과 tmp_missing_stop 을 뒤 쿼리에서 재사용'
\echo '###############################################################'
drop table if exists tmp_missing_stop;
create temp table tmp_missing_stop as
with st_miss as (
    select distinct r.st_node_id as missing_id, 'st'::text as role
      from public.route_link_sequence r
      left join public.dim_stop d on d.stop_id = r.st_node_id
     where r.st_node_id is not null
       and d.stop_id is null
),
ed_miss as (
    select distinct r.ed_node_id as missing_id, 'ed'::text as role
      from public.route_link_sequence r
      left join public.dim_stop d on d.stop_id = r.ed_node_id
     where r.ed_node_id is not null
       and d.stop_id is null
)
select missing_id, role from st_miss
union all
select missing_id, role from ed_miss;

select
    count(*)                              as missing_rows_st_plus_ed,
    count(distinct missing_id)            as missing_distinct_ids,
    count(distinct missing_id) filter (where role='st') as missing_st_only_distinct,
    count(distinct missing_id) filter (where role='ed') as missing_ed_only_distinct
  from tmp_missing_stop;

\echo ''
\echo '###############################################################'
\echo '# D6. missing_id 길이 / 문자유형 분포'
\echo '###############################################################'
select
    length(missing_id)                                            as len,
    (missing_id ~ '^[0-9]+$')                                     as is_numeric,
    (missing_id ~ '^[A-Za-z]')                                    as starts_with_alpha,
    count(distinct missing_id)                                    as distinct_cnt
  from tmp_missing_stop
 group by 1, 2, 3
 order by len, is_numeric desc;

\echo ''
\echo '###############################################################'
\echo '# D7. dim_stop.stop_id 길이 / 문자유형 분포 (비교용)'
\echo '###############################################################'
select
    length(stop_id)                                               as len,
    (stop_id ~ '^[0-9]+$')                                        as is_numeric,
    (stop_id ~ '^[A-Za-z]')                                       as starts_with_alpha,
    count(*)                                                      as cnt
  from public.dim_stop
 group by 1, 2, 3
 order by len, is_numeric desc;

\echo ''
\echo '###############################################################'
\echo '# D8. 매칭 성공/실패 이분 — st_node_id 길이 × 매칭 여부 교차표'
\echo '###############################################################'
select
    length(r.st_node_id)                                          as len_st,
    count(*) filter (where d.stop_id is not null)                 as matched,
    count(*) filter (where d.stop_id is null)                     as unmatched
  from public.route_link_sequence r
  left join public.dim_stop d on d.stop_id = r.st_node_id
 where r.st_node_id is not null
 group by 1
 order by 1;

\echo ''
\echo '###############################################################'
\echo '# D9. missing_id 샘플 30개 (포맷 유추용)'
\echo '###############################################################'
select missing_id, role, length(missing_id) as len
  from tmp_missing_stop
 order by length(missing_id), missing_id
 limit 30;

\echo ''
\echo '###############################################################'
\echo '# D10. dim_stop.stop_id 샘플 30개 (포맷 유추용)'
\echo '###############################################################'
select stop_id, length(stop_id) as len
  from public.dim_stop
 order by length(stop_id), stop_id
 limit 30;

\echo ''
\echo '###############################################################'
\echo '# D11. (선택) dim_stop 에 node_uid / node_id / bis_id / bs_id / nid'
\echo '#       류 컬럼이 있다면 missing_id 와의 매칭률 탐색.'
\echo '#       존재하지 않는 컬럼은 ERROR 로 뜨지만 ON_ERROR_STOP=0 이라 건너뜀.'
\echo '###############################################################'

\echo '--- D11-a: missing_id vs dim_stop.node_uid (있을 때만) ---'
select
    count(*)                                                      as joined_cnt,
    count(*) filter (where ds.node_uid like 'STOP:%')             as uid_prefixed
  from tmp_missing_stop m
  join public.dim_stop ds on ds.node_uid = m.missing_id;

\echo '--- D11-b: missing_id vs dim_stop.node_uid (STOP: prefix 제거 가정) ---'
select
    count(*)                                                      as joined_cnt
  from tmp_missing_stop m
  join public.dim_stop ds on ds.node_uid = 'STOP:' || m.missing_id;

\echo '--- D11-c: missing_id vs dim_stop.bis_id ---'
select count(*) as joined_cnt
  from tmp_missing_stop m
  join public.dim_stop ds on ds.bis_id = m.missing_id;

\echo '--- D11-d: missing_id vs dim_stop.bs_id ---'
select count(*) as joined_cnt
  from tmp_missing_stop m
  join public.dim_stop ds on ds.bs_id = m.missing_id;

\echo '--- D11-e: missing_id vs dim_stop.nid ---'
select count(*) as joined_cnt
  from tmp_missing_stop m
  join public.dim_stop ds on ds.nid = m.missing_id;

\echo '--- D11-f: missing_id vs dim_stop.node_id ---'
select count(*) as joined_cnt
  from tmp_missing_stop m
  join public.dim_stop ds on ds.node_id::text = m.missing_id;

\echo '--- D11-g: missing_id vs dim_stop.mobile_no / moving_no (가끔 쓰임) ---'
select count(*) as joined_cnt_mobile
  from tmp_missing_stop m
  join public.dim_stop ds on ds.mobile_no::text = m.missing_id;

\echo ''
\echo '###############################################################'
\echo '# D12. (선택) 전형적 staging 테이블과 missing_id 의 접점 탐색'
\echo '#       존재하지 않는 테이블은 건너뜀.'
\echo '###############################################################'

\echo '--- D12-a: public.stg_bs_info ---'
select count(*) as matched_in_stg_bs_info
  from public.stg_bs_info s
  join tmp_missing_stop m on s.bs_id = m.missing_id;

\echo '--- D12-b: public.stg_bus_stop ---'
select count(*) as matched_in_stg_bus_stop
  from public.stg_bus_stop s
  join tmp_missing_stop m on s.stop_id = m.missing_id;

\echo '--- D12-c: public.bis_node ---'
select count(*) as matched_in_bis_node
  from public.bis_node s
  join tmp_missing_stop m on s.node_id = m.missing_id;

\echo '--- D12-d: public.bis_stop ---'
select count(*) as matched_in_bis_stop
  from public.bis_stop s
  join tmp_missing_stop m on s.stop_id = m.missing_id;

\echo '--- D12-e: public.raw_route_link_sequence ---'
select count(*) as src_has_raw_rls
  from public.raw_route_link_sequence
 limit 1;

\echo ''
\echo '###############################################################'
\echo '# D13. 매칭 성공 노선 vs 실패 노선 분포 (route 단위 핫스팟 찾기)'
\echo '###############################################################'
with per_route as (
    select r.route_id,
           r.move_dir_code,
           count(*)                                            as link_cnt,
           count(*) filter (
               where exists (select 1 from public.dim_stop d
                             where d.stop_id = r.st_node_id)
                 and exists (select 1 from public.dim_stop d
                             where d.stop_id = r.ed_node_id)
           ) as both_matched
    from public.route_link_sequence r
    group by r.route_id, r.move_dir_code
)
select route_id, move_dir_code,
       link_cnt,
       both_matched,
       (link_cnt - both_matched)                               as unmatched_links,
       round(both_matched::numeric / nullif(link_cnt,0) * 100, 2) as match_pct
  from per_route
 order by unmatched_links desc, link_cnt desc
 limit 30;

\echo ''
\echo '###############################################################'
\echo '# D14. dim_stop 전체 건수 vs 실제 route_link_sequence 에 등장하는'
\echo '#       distinct stop 수 비교 (coverage 상한 추정)'
\echo '###############################################################'
select
    (select count(*) from public.dim_stop)                        as dim_stop_rows,
    (select count(distinct x) from (
         select st_node_id as x from public.route_link_sequence
         union
         select ed_node_id     from public.route_link_sequence
     ) u where x is not null)                                     as rls_distinct_stops;

\echo ''
\echo '###############################################################'
\echo '# D15. missing_id 중 매칭 성공한 dim_stop 과 숫자 prefix 가 겹치는지'
\echo '#       (예: 702341 vs 7023410000) 접두사 4자리 교차'
\echo '###############################################################'
with miss as (
    select distinct missing_id from tmp_missing_stop
     where missing_id ~ '^[0-9]+$'
),
ds as (
    select distinct stop_id from public.dim_stop
     where stop_id ~ '^[0-9]+$'
)
select
    count(*) filter (where exists (
        select 1 from ds
         where substring(ds.stop_id from 1 for 4) = substring(m.missing_id from 1 for 4)
    )) as missing_with_prefix4_match,
    count(*) as missing_numeric_total
  from miss m;

\echo ''
\echo '###############################################################'
\echo '# END OF DIAGNOSIS — 결과를 캡처해 회신해 주세요.'
\echo '###############################################################'
