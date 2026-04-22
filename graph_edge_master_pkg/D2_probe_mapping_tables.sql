-- =============================================================================
-- D2_probe_mapping_tables.sql
-- Purpose : Stage 1 진단(D_diagnose_stop_id_mapping.sql) 결과로 가설 B(브리지/
--           staging 매핑) 가 유력해짐. D3 에서 발견된 5개 후보 테이블에 대해
--           컬럼별로 missing_id 매칭률을 동적으로 계산해 매핑 키를 식별한다.
-- 후보 테이블:
--   1) stop_link_mapping_master (13 cols)  ← 1순위
--   2) stg_daegu_stops_geo (11)
--   3) bs_20250903 (16)
--   4) graph_node_master (13)
--   5) err_daegu_stop_usage_mapping_failed (10)
-- 본 파일은 조회 전용 (DDL 없음, DML 없음).
-- =============================================================================

\pset pager off
\timing off

-- -----------------------------------------------------------------------------
-- 0) tmp_missing_stop 재구성 (이전 세션 종료로 사라졌으므로 다시 만든다)
-- -----------------------------------------------------------------------------
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
    select missing_id from st_miss
    union all
    select missing_id from ed_miss
) u;

create index on tmp_missing_stop (missing_id);

select count(*) as missing_distinct_ids from tmp_missing_stop;

-- -----------------------------------------------------------------------------
-- 1) 후보 테이블별 컬럼 목록 (수동 검토용)
-- -----------------------------------------------------------------------------
\echo ''
\echo '============================================================'
\echo ' P1. stop_link_mapping_master columns'
\echo '============================================================'
select ordinal_position, column_name, data_type, character_maximum_length as maxlen
  from information_schema.columns
 where table_schema='public' and table_name='stop_link_mapping_master'
 order by ordinal_position;

\echo ''
\echo '============================================================'
\echo ' P2. stg_daegu_stops_geo columns'
\echo '============================================================'
select ordinal_position, column_name, data_type, character_maximum_length as maxlen
  from information_schema.columns
 where table_schema='public' and table_name='stg_daegu_stops_geo'
 order by ordinal_position;

\echo ''
\echo '============================================================'
\echo ' P3. bs_20250903 columns'
\echo '============================================================'
select ordinal_position, column_name, data_type, character_maximum_length as maxlen
  from information_schema.columns
 where table_schema='public' and table_name='bs_20250903'
 order by ordinal_position;

\echo ''
\echo '============================================================'
\echo ' P4. graph_node_master columns'
\echo '============================================================'
select ordinal_position, column_name, data_type, character_maximum_length as maxlen
  from information_schema.columns
 where table_schema='public' and table_name='graph_node_master'
 order by ordinal_position;

\echo ''
\echo '============================================================'
\echo ' P5. err_daegu_stop_usage_mapping_failed columns'
\echo '============================================================'
select ordinal_position, column_name, data_type, character_maximum_length as maxlen
  from information_schema.columns
 where table_schema='public' and table_name='err_daegu_stop_usage_mapping_failed'
 order by ordinal_position;

-- -----------------------------------------------------------------------------
-- 2) 후보 테이블별 첫 5행 샘플
-- -----------------------------------------------------------------------------
\echo ''
\echo '============================================================'
\echo ' S1. stop_link_mapping_master sample (5 rows)'
\echo '============================================================'
select * from public.stop_link_mapping_master limit 5;

\echo ''
\echo '============================================================'
\echo ' S2. stg_daegu_stops_geo sample (5 rows)'
\echo '============================================================'
select * from public.stg_daegu_stops_geo limit 5;

\echo ''
\echo '============================================================'
\echo ' S3. bs_20250903 sample (5 rows)'
\echo '============================================================'
select * from public.bs_20250903 limit 5;

\echo ''
\echo '============================================================'
\echo ' S4. graph_node_master sample (5 rows)'
\echo '============================================================'
select * from public.graph_node_master limit 5;

\echo ''
\echo '============================================================'
\echo ' S5. err_daegu_stop_usage_mapping_failed sample (5 rows)'
\echo '============================================================'
select * from public.err_daegu_stop_usage_mapping_failed limit 5;

-- -----------------------------------------------------------------------------
-- 3) 동적 매칭 — 후보 테이블 X 모든 (text/varchar/numeric/integer/bigint) 컬럼
--    에 대해 missing_id (text, len=10, numeric) 와의 매칭 건수를 NOTICE 로 출력
-- -----------------------------------------------------------------------------
\echo ''
\echo '============================================================'
\echo ' M0. dynamic match probe — column x missing_id (NOTICE 로 출력)'
\echo '     기준값: missing_distinct_ids 와 비교'
\echo '============================================================'

do $$
declare
    v_tables text[] := array[
        'stop_link_mapping_master',
        'stg_daegu_stops_geo',
        'bs_20250903',
        'graph_node_master',
        'err_daegu_stop_usage_mapping_failed'
    ];
    v_table  text;
    rec      record;
    v_cnt    bigint;
    v_sql    text;
    v_total  bigint;
begin
    select count(*) into v_total from tmp_missing_stop;
    raise notice '----------------------------------------';
    raise notice 'BASELINE missing_distinct_ids = %', v_total;
    raise notice '----------------------------------------';

    foreach v_table in array v_tables loop
        -- 테이블 존재 확인
        if not exists (
            select 1 from information_schema.tables
             where table_schema='public' and table_name=v_table
        ) then
            raise notice '[skip] table % not present', v_table;
            continue;
        end if;

        raise notice '';
        raise notice '== TABLE: % ==', v_table;

        for rec in
            select column_name, data_type
              from information_schema.columns
             where table_schema='public'
               and table_name=v_table
               and (data_type in ('text','character varying','character','varchar')
                    or data_type in ('integer','bigint','numeric','smallint'))
             order by ordinal_position
        loop
            v_sql := format(
              'select count(*) from public.%I t join tmp_missing_stop m on t.%I::text = m.missing_id',
              v_table, rec.column_name
            );
            begin
                execute v_sql into v_cnt;
            exception when others then
                v_cnt := -1;
            end;
            if v_cnt is null then v_cnt := 0; end if;
            if v_cnt > 0 then
                raise notice '   %  [%]  => % matches  (%.2f %% of missing)',
                    rpad(rec.column_name, 30), rec.data_type, v_cnt,
                    (v_cnt::numeric * 100.0 / nullif(v_total,0));
            else
                raise notice '   %  [%]  => % matches', rpad(rec.column_name, 30), rec.data_type, v_cnt;
            end if;
        end loop;
    end loop;

    raise notice '';
    raise notice '== probe done ==';
end $$;

-- -----------------------------------------------------------------------------
-- 4) BONUS — 후보 테이블 행수 비교
-- -----------------------------------------------------------------------------
\echo ''
\echo '============================================================'
\echo ' B1. row counts of candidate tables'
\echo '============================================================'
select 'stop_link_mapping_master' as t, (select count(*) from public.stop_link_mapping_master) as rows
union all select 'stg_daegu_stops_geo',                  (select count(*) from public.stg_daegu_stops_geo)
union all select 'bs_20250903',                          (select count(*) from public.bs_20250903)
union all select 'graph_node_master',                    (select count(*) from public.graph_node_master)
union all select 'err_daegu_stop_usage_mapping_failed',  (select count(*) from public.err_daegu_stop_usage_mapping_failed);

\echo ''
\echo '============================================================'
\echo ' END OF PROBE — D2 결과 캡처해 회신'
\echo '============================================================'
