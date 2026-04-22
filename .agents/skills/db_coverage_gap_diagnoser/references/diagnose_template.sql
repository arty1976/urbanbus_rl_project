-- =============================================================================
-- diagnose_template.sql  (db_coverage_gap_diagnoser skill bundle)
--
-- Purpose : readiness coverage_pct 가 기대치 대비 크게 낮을 때,
--           A(alt-key) / B(bridge) / C(upstream) / D(semantic) 4가설을
--           D1 → D2 → D3 → D4 순서로 체계적으로 기각/확정해 나가는
--           범용 진단 쿼리 풀세트.
--
-- Usage   : 도시/유스케이스별로 `-- @CHANGE_FOR_USE_CASE` 표시된 라인만
--           치환하고 실행.
--              psql -h <host> -U <user> -d <db> -v ON_ERROR_STOP=0 \
--                    -f diagnose_template.sql
--
-- IMPORTANT : 이 스크립트는 **SELECT / DO $$ 전용**. INSERT/UPDATE/DELETE
--             를 포함하지 않는다. 진단과 수정은 분리.
--
-- 출력 해석 기준 (판정은 Skill SKILL.md 의 Stage D1~D4 참고):
--   D1  실패 측 id 의 길이/prefix 분포 + dim 컬럼 전수 매칭 스캔 → 가설 A
--   D2  staging/bridge 후보 테이블 × 컬럼 dynamic probe → 가설 B
--   D3  핫스팟 1노선 semantic pattern tagging + 전체 분포 → 가설 D
--   D4  증거 요약 블록 (사람이 읽고 평가)
-- =============================================================================

\pset pager off
\timing off

-- @CHANGE_FOR_USE_CASE  스키마
\set schema_name 'public'

-- @CHANGE_FOR_USE_CASE  원천 테이블 & 컬럼 (edge source)
--   예: route_link_sequence(route_id, move_dir_code, link_seq, st_node_id, ed_node_id)
\set src_table              'route_link_sequence'
\set src_partition_col_a    'route_id'
\set src_partition_col_b    'move_dir_code'
\set src_seq_col            'link_seq'
\set src_st_col             'st_node_id'
\set src_ed_col             'ed_node_id'

-- @CHANGE_FOR_USE_CASE  타겟 dim 테이블 & PK
\set dim_table 'dim_stop'
\set dim_pk    'stop_id'

\echo '================================================================='
\echo ' D1. Key Pattern Analysis  (가설 A: Alt-Key Mismatch 검증)'
\echo '================================================================='

\echo ''
\echo '-- D1.1  missing id 모수 / distinct / 길이 / prefix 분포'
with missing as (
    select r.st_node_id as missing_id from public.route_link_sequence r     -- @CHANGE_FOR_USE_CASE
      left join public.dim_stop d on d.stop_id = r.st_node_id               -- @CHANGE_FOR_USE_CASE
      where r.st_node_id is not null and d.stop_id is null
    union all
    select r.ed_node_id as missing_id from public.route_link_sequence r     -- @CHANGE_FOR_USE_CASE
      left join public.dim_stop d on d.stop_id = r.ed_node_id               -- @CHANGE_FOR_USE_CASE
      where r.ed_node_id is not null and d.stop_id is null
)
select
    count(*)                                 as missing_total,
    count(distinct missing_id)               as distinct_missing
from missing;

\echo ''
\echo '-- D1.2  missing id 의 길이별 분포 (대부분 하나의 길이에 몰리면 코드체계 힌트)'
with missing as (
    select r.st_node_id::text as missing_id from public.route_link_sequence r   -- @CHANGE_FOR_USE_CASE
     left join public.dim_stop d on d.stop_id = r.st_node_id                    -- @CHANGE_FOR_USE_CASE
     where r.st_node_id is not null and d.stop_id is null
    union all
    select r.ed_node_id::text from public.route_link_sequence r                 -- @CHANGE_FOR_USE_CASE
     left join public.dim_stop d on d.stop_id = r.ed_node_id                    -- @CHANGE_FOR_USE_CASE
     where r.ed_node_id is not null and d.stop_id is null
)
select length(missing_id) as id_len, count(*) as cnt
from missing group by 1 order by 2 desc;

\echo ''
\echo '-- D1.3  missing id prefix 분포 (첫 5자 기준)'
with missing as (
    select r.st_node_id::text as missing_id from public.route_link_sequence r   -- @CHANGE_FOR_USE_CASE
     left join public.dim_stop d on d.stop_id = r.st_node_id                    -- @CHANGE_FOR_USE_CASE
     where r.st_node_id is not null and d.stop_id is null
    union all
    select r.ed_node_id::text from public.route_link_sequence r                 -- @CHANGE_FOR_USE_CASE
     left join public.dim_stop d on d.stop_id = r.ed_node_id                    -- @CHANGE_FOR_USE_CASE
     where r.ed_node_id is not null and d.stop_id is null
)
select substring(missing_id from 1 for 5) as id_prefix, count(*) as cnt
from missing group by 1 order by 2 desc limit 30;

\echo ''
\echo '-- D1.4  dim 테이블 전 text/numeric 컬럼에 missing_id 매칭 스캔'
\echo '--       매칭 수 > 0 컬럼 있으면 가설 A 확정, 전부 0 이면 A 기각'
DO $do$
DECLARE
    r record;
    hit_count bigint;
    sample_sql text;
BEGIN
    -- @CHANGE_FOR_USE_CASE  (스키마명 / dim 테이블명)
    FOR r IN
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'dim_stop'
          AND data_type IN ('text','character varying','bigint','integer','numeric','character')
    LOOP
        -- @CHANGE_FOR_USE_CASE  (원천 테이블 / dim 테이블명)
        sample_sql := format($q$
            WITH missing AS (
              SELECT st_node_id::text AS mid FROM public.route_link_sequence r
                LEFT JOIN public.dim_stop d ON d.stop_id = r.st_node_id
                WHERE r.st_node_id IS NOT NULL AND d.stop_id IS NULL
              UNION ALL
              SELECT ed_node_id::text FROM public.route_link_sequence r
                LEFT JOIN public.dim_stop d ON d.stop_id = r.ed_node_id
                WHERE r.ed_node_id IS NOT NULL AND d.stop_id IS NULL
            )
            SELECT count(*) FROM public.dim_stop t
            WHERE t.%I::text IN (SELECT mid FROM missing)
        $q$, r.column_name);
        EXECUTE sample_sql INTO hit_count;
        IF hit_count > 0 THEN
            RAISE NOTICE '  [D1.4 HIT]  dim_stop.% (type=%): % matches',
                r.column_name, r.data_type, hit_count;
        END IF;
    END LOOP;
    RAISE NOTICE '  [D1.4 done]  (이 블록에서 HIT 라인이 하나도 없으면 가설 A 기각)';
END
$do$;

\echo ''
\echo '-- D1.5  missing id 샘플 30개'
with missing as (
    select distinct r.st_node_id::text as mid from public.route_link_sequence r  -- @CHANGE_FOR_USE_CASE
     left join public.dim_stop d on d.stop_id = r.st_node_id                     -- @CHANGE_FOR_USE_CASE
     where r.st_node_id is not null and d.stop_id is null
)
select mid from missing order by mid limit 30;

\echo ''
\echo '================================================================='
\echo ' D2. Dynamic Bridge/Staging Probe  (가설 B: Bridge Exists 검증)'
\echo '================================================================='
\echo ''
\echo '-- 후보 테이블 패턴:'
\echo '--   %mapping% / %bridge% / %xref% / %lookup% / stg_% / staging_% / raw_% / err_%'
\echo '-- 각 후보 테이블 × 각 text/numeric 컬럼에 대해 missing_id 매칭 수 측정'
\echo ''
DO $do$
DECLARE
    r record;
    hit_count bigint;
    probe_sql text;
BEGIN
    -- @CHANGE_FOR_USE_CASE  (스키마명 + missing 정의)
    FOR r IN
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND (table_name ~* '(mapping|bridge|xref|lookup)'
            OR table_name ~* '^(stg_|staging_|raw_)'
            OR table_name ~* '^err_')
          AND data_type IN ('text','character varying','bigint','integer','numeric','character')
          AND table_name NOT IN ('dim_stop')   -- @CHANGE_FOR_USE_CASE  (이미 D1 에서 본 테이블)
    LOOP
        probe_sql := format($q$
            WITH missing AS (
              SELECT st_node_id::text AS mid FROM public.route_link_sequence r
                LEFT JOIN public.dim_stop d ON d.stop_id = r.st_node_id
                WHERE r.st_node_id IS NOT NULL AND d.stop_id IS NULL
              UNION ALL
              SELECT ed_node_id::text FROM public.route_link_sequence r
                LEFT JOIN public.dim_stop d ON d.stop_id = r.ed_node_id
                WHERE r.ed_node_id IS NOT NULL AND d.stop_id IS NULL
            )
            SELECT count(*) FROM %I.%I t
            WHERE t.%I::text IN (SELECT mid FROM missing)
        $q$, r.table_schema, r.table_name, r.column_name);

        BEGIN
            EXECUTE probe_sql INTO hit_count;
        EXCEPTION WHEN others THEN
            hit_count := -1;   -- 캐스팅 실패 등은 skip
        END;

        IF hit_count > 0 THEN
            RAISE NOTICE '  [D2 HIT]  %.%.% (type=%): % matches',
                r.table_schema, r.table_name, r.column_name, r.data_type, hit_count;
        END IF;
    END LOOP;
    RAISE NOTICE '  [D2 done]  (HIT 라인이 하나도 없으면 가설 B 기각)';
END
$do$;

\echo ''
\echo '================================================================='
\echo ' D3. Hotspot Semantic Pattern Tagging  (가설 D: Semantic Mismatch)'
\echo '================================================================='

\echo ''
\echo '-- D3.1  핫스팟 (route, direction) 1건 선정 — link 수 상위'
with hot as (
    select route_id, move_dir_code,
           count(*) as links
    from public.route_link_sequence                   -- @CHANGE_FOR_USE_CASE
    group by 1,2
    order by links desc limit 1
)
select * from hot;

\echo ''
\echo '-- D3.2  핫스팟 시퀀스의 st/ed 각각 stop 인지 태깅한 뒤 패턴 분포'
\echo '--       S->S 만 95%+ 이면 D 기각 / 혼합 섞이면 D 확정'
with hot as (
    select route_id, move_dir_code
    from public.route_link_sequence                   -- @CHANGE_FOR_USE_CASE
    group by 1,2
    order by count(*) desc limit 1
),
tagged as (
    select r.link_seq,
           r.st_node_id, r.ed_node_id,
           case when d1.stop_id is not null then 'S' else '-' end as st_tag,
           case when d2.stop_id is not null then 'S' else '-' end as ed_tag
    from public.route_link_sequence r                 -- @CHANGE_FOR_USE_CASE
    join hot h on h.route_id = r.route_id
              and h.move_dir_code = r.move_dir_code
    left join public.dim_stop d1 on d1.stop_id = r.st_node_id   -- @CHANGE_FOR_USE_CASE
    left join public.dim_stop d2 on d2.stop_id = r.ed_node_id   -- @CHANGE_FOR_USE_CASE
)
select st_tag || '->' || ed_tag as pattern, count(*) as cnt,
       round(count(*) * 100.0 / sum(count(*)) over (), 2) as pct
from tagged group by 1 order by 2 desc;

\echo ''
\echo '-- D3.3  전체 코퍼스 패턴 분포 (핫스팟이 아닌 전체 기준)'
\echo '--       핫스팟과 전체 분포가 비슷하면 가설 D 일반화 OK'
with tagged as (
    select case when d1.stop_id is not null then 'S' else '-' end as st_tag,
           case when d2.stop_id is not null then 'S' else '-' end as ed_tag
    from public.route_link_sequence r                 -- @CHANGE_FOR_USE_CASE
    left join public.dim_stop d1 on d1.stop_id = r.st_node_id   -- @CHANGE_FOR_USE_CASE
    left join public.dim_stop d2 on d2.stop_id = r.ed_node_id   -- @CHANGE_FOR_USE_CASE
    where r.st_node_id is not null and r.ed_node_id is not null
)
select st_tag || '->' || ed_tag as pattern, count(*) as cnt,
       round(count(*) * 100.0 / sum(count(*)) over (), 2) as pct
from tagged group by 1 order by 2 desc;

\echo ''
\echo '-- D3.4  "한 stop 당 평균 link 수" — D 확정 시 2배 이상이 정상'
with per_route as (
    select route_id, move_dir_code,
           count(*) filter (where d1.stop_id is not null)::numeric as stop_starts,
           count(*)::numeric                                       as total_links
    from public.route_link_sequence r                 -- @CHANGE_FOR_USE_CASE
    left join public.dim_stop d1 on d1.stop_id = r.st_node_id   -- @CHANGE_FOR_USE_CASE
    group by 1,2
    having count(*) filter (where d1.stop_id is not null) > 0
)
select
    avg(total_links / stop_starts)::numeric(10,3) as avg_links_per_stop,
    max(total_links / stop_starts)::numeric(10,3) as max_links_per_stop,
    count(*)                                       as routes_observed
from per_route;

\echo ''
\echo '================================================================='
\echo ' D4. Evidence Aggregation  (결론 노트용)'
\echo '================================================================='
\echo '-- 아래 결과를 읽고 evidence_{YYYYMMDD}.md 에 다음 표를 채워 기록:'
\echo ''
\echo '   | 가설 | 진단 방법            | 결과 요약                  | 판정 |'
\echo '   |------|--------------------|-------------------------|------|'
\echo '   | A    | D1.1~D1.4          | (missing 수, dim 전수 스캔) | ?    |'
\echo '   | B    | D2                 | (후보 테이블 × 컬럼 매칭)    | ?    |'
\echo '   | D    | D3.1~D3.4          | (패턴 분포, links/stop)     | ?    |'
\echo '   | C    | A/B/D 모두 기각이면  | (ETL/raw row count 재조회)  | ?    |'
\echo ''
\echo '-- 확정된 가설에 따라 다음 스킬로 이관:'
\echo '--   A 확정 → dim_stop_promotion_guard (alt-key 승격)'
\echo '--   B 확정 → bridge-aware 조인 재작성'
\echo '--   C 확정 → ETL runbook 재실행'
\echo '--   D 확정 → stop_to_stop_edge_leg_aggregator (leg-aggregation)'
\echo ''
\echo '================================================================='
\echo ' DIAGNOSE COMPLETE  — 위 로그 전체를 diagnose/evidence_*.md 에 붙여둘 것'
\echo '================================================================='
