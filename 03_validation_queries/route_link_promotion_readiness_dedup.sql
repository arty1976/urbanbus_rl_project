-- [route_link_sequence ?밴꺽 以鍮?/ ?밴꺽 ??寃利?
-- 紐⑹쟻:
-- 1) public.stg_daegu_routes 湲곗? ?먯쿇 ?몄꽑 238媛?以??덉쇅 4媛쒕? ?쒖쇅??234媛쒓? 寃利???곸씤吏 ?뺤씤
-- 2) legacy sample route_id = '1000' ??寃利???곸뿉 ?욎씠吏 ?딆븯?붿? ?뺤씤
-- 3) public.vw_stg_daegu_route_links_api_dedup ? public.route_link_sequence 媛???嫄댁닔/?곗냽??臾닿껐?깆쓣 ?먭?
--
-- ?ъ슜 ?먯튃:
-- - 寃곌낵媛 0?됱씠硫??뺤긽??荑쇰━媛 ?遺遺꾩엯?덈떎.
-- - summary 荑쇰━???곹깭 ?뺤씤?⑹씠誘濡???긽 1???댁긽 諛섑솚?⑸땲??
-- - 蹂??뚯씪? "?덉쇅 4媛??쒖쇅 + legacy sample 1000 ?쒖쇅" 洹쒖튃???댁옣?⑸땲??

------------------------------------------------------------
-- 0. Summary
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text, 'no_items_in_json_response'::text),
        ('4040006021'::text, 'no_items_in_json_response'::text),
        ('4040007001'::text, 'no_items_in_json_response'::text),
        ('4040007009'::text, 'no_items_in_json_response'::text)
    ) v(route_id, reason)
),
source_routes AS (
    SELECT DISTINCT route_id::text AS route_id
    FROM public.stg_daegu_routes
    WHERE route_id IS NOT NULL
),
eligible_source_routes AS (
    SELECT s.route_id
    FROM source_routes s
    LEFT JOIN excluded_routes e
      ON e.route_id = s.route_id
    WHERE e.route_id IS NULL
),
stg_effective_routes AS (
    SELECT DISTINCT s.route_id_raw::text AS route_id
    FROM public.vw_stg_daegu_route_links_api_dedup s
    WHERE btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
      AND s.route_id_raw::bigint > 0
      AND s.route_id_raw::text <> '1000'
      AND s.route_id_raw::text NOT IN (SELECT route_id FROM excluded_routes)
),
tgt_all_routes AS (
    SELECT DISTINCT route_id::text AS route_id
    FROM public.route_link_sequence
),
tgt_effective_routes AS (
    SELECT DISTINCT route_id::text AS route_id
    FROM public.route_link_sequence
    WHERE route_id::text <> '1000'
      AND route_id::text NOT IN (SELECT route_id FROM excluded_routes)
)
SELECT
    'route_link_promotion_scope_summary' AS section,
    (SELECT count(*) FROM source_routes) AS source_routes_total,
    (SELECT count(*) FROM excluded_routes) AS excluded_route_count,
    (SELECT count(*) FROM eligible_source_routes) AS eligible_source_routes,
    (SELECT count(*) FROM stg_effective_routes) AS staging_effective_distinct_routes,
    (SELECT count(*) FROM tgt_all_routes) AS promoted_distinct_routes_all,
    (SELECT count(*) FROM tgt_effective_routes) AS promoted_effective_distinct_routes,
    CASE
        WHEN (SELECT count(*) FROM eligible_source_routes) = 234 THEN 'PASS'
        ELSE 'FAIL'
    END AS eligible_route_count_status;

------------------------------------------------------------
-- 1. ?먯쿇 湲곗? ?덉뼱????234媛??몄꽑 以?staging ???녿뒗 ?몄꽑
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
),
eligible_source_routes AS (
    SELECT DISTINCT route_id::text AS route_id
    FROM public.stg_daegu_routes
    WHERE route_id IS NOT NULL
      AND route_id::text NOT IN (SELECT route_id FROM excluded_routes)
),
stg_effective_routes AS (
    SELECT DISTINCT route_id_raw::text AS route_id
    FROM public.vw_stg_daegu_route_links_api_dedup
    WHERE btrim(COALESCE(route_id_raw, '')) ~ '^\d+$'
      AND route_id_raw::bigint > 0
      AND route_id_raw::text <> '1000'
      AND route_id_raw::text NOT IN (SELECT route_id FROM excluded_routes)
)
SELECT
    'missing_route_in_staging' AS check_name,
    s.route_id
FROM eligible_source_routes s
LEFT JOIN stg_effective_routes t
  ON t.route_id = s.route_id
WHERE t.route_id IS NULL
ORDER BY s.route_id;

------------------------------------------------------------
-- 2. staging ???ㅼ뼱?붿?留??먯쿇 湲곗? ??곸씠 ?꾨땶 route_id
--    (?? legacy sample 1000, source 誘몄〈??route_id)
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
),
eligible_source_routes AS (
    SELECT DISTINCT route_id::text AS route_id
    FROM public.stg_daegu_routes
    WHERE route_id IS NOT NULL
      AND route_id::text NOT IN (SELECT route_id FROM excluded_routes)
),
stg_all_routes AS (
    SELECT DISTINCT route_id_raw::text AS route_id
    FROM public.vw_stg_daegu_route_links_api_dedup
    WHERE btrim(COALESCE(route_id_raw, '')) ~ '^\d+$'
      AND route_id_raw::bigint > 0
)
SELECT
    'unexpected_route_in_staging' AS check_name,
    s.route_id,
    CASE
        WHEN s.route_id = '1000' THEN 'legacy_sample_route'
        WHEN s.route_id IN (SELECT route_id FROM excluded_routes) THEN 'excluded_exception_route'
        ELSE 'not_found_in_source_routes'
    END AS reason
FROM stg_all_routes s
LEFT JOIN eligible_source_routes e
  ON e.route_id = s.route_id
WHERE e.route_id IS NULL
ORDER BY s.route_id;

------------------------------------------------------------
-- 3. staging ?먯뿰??以묐났
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
),
stg_valid AS (
    SELECT
        route_id_raw::text AS route_id,
        move_dir_raw::integer::text AS move_dir_code,
        link_seq_raw::integer AS link_seq
    FROM public.vw_stg_daegu_route_links_api_dedup
    WHERE btrim(COALESCE(route_id_raw, '')) ~ '^\d+$'
      AND route_id_raw::bigint > 0
      AND route_id_raw::text <> '1000'
      AND route_id_raw::text NOT IN (SELECT route_id FROM excluded_routes)
      AND btrim(COALESCE(move_dir_raw, '')) ~ '^-?\d+$'
      AND btrim(COALESCE(link_seq_raw, '')) ~ '^\d+$'
      AND link_seq_raw::integer > 0
)
SELECT
    'staging_dup_natural_key' AS check_name,
    route_id,
    move_dir_code,
    link_seq,
    count(*) AS dup_count
FROM stg_valid
GROUP BY route_id, move_dir_code, link_seq
HAVING count(*) > 1
ORDER BY route_id, move_dir_code, link_seq;

------------------------------------------------------------
-- 4. promoted ?뚯씠釉붿뿉 ?덉쇅 4媛??먮뒗 legacy sample 1000 ???⑥븘 ?덈뒗吏 ?뺤씤
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
)
SELECT
    'unexpected_route_in_promoted' AS check_name,
    route_id,
    move_dir_code,
    min(link_seq) AS min_link_seq,
    max(link_seq) AS max_link_seq,
    count(*) AS row_count,
    CASE
        WHEN route_id = '1000' THEN 'legacy_sample_route'
        ELSE 'excluded_exception_route'
    END AS reason
FROM public.route_link_sequence
WHERE route_id = '1000'
   OR route_id IN (SELECT route_id FROM excluded_routes)
GROUP BY route_id, move_dir_code
ORDER BY route_id, move_dir_code;

------------------------------------------------------------
-- 5. promoted ?먯뿰??以묐났
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
)
SELECT
    'promoted_dup_natural_key' AS check_name,
    route_id,
    move_dir_code,
    link_seq,
    count(*) AS dup_count
FROM public.route_link_sequence
WHERE route_id <> '1000'
  AND route_id NOT IN (SELECT route_id FROM excluded_routes)
GROUP BY route_id, move_dir_code, link_seq
HAVING count(*) > 1
ORDER BY route_id, move_dir_code, link_seq;

------------------------------------------------------------
-- 6. staging distinct key ??vs promoted ??遺덉씪移?------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
),
stg_distinct_keys AS (
    SELECT DISTINCT
        route_id_raw::text AS route_id,
        move_dir_raw::integer::text AS move_dir_code,
        link_seq_raw::integer AS link_seq
    FROM public.vw_stg_daegu_route_links_api_dedup
    WHERE btrim(COALESCE(route_id_raw, '')) ~ '^\d+$'
      AND route_id_raw::bigint > 0
      AND route_id_raw::text <> '1000'
      AND route_id_raw::text NOT IN (SELECT route_id FROM excluded_routes)
      AND btrim(COALESCE(move_dir_raw, '')) ~ '^-?\d+$'
      AND btrim(COALESCE(link_seq_raw, '')) ~ '^\d+$'
      AND link_seq_raw::integer > 0
),
stg_counts AS (
    SELECT route_id, move_dir_code, count(*) AS staging_cnt
    FROM stg_distinct_keys
    GROUP BY route_id, move_dir_code
),
tgt_counts AS (
    SELECT route_id, move_dir_code, count(*) AS promoted_cnt
    FROM public.route_link_sequence
    WHERE route_id <> '1000'
      AND route_id NOT IN (SELECT route_id FROM excluded_routes)
    GROUP BY route_id, move_dir_code
)
SELECT
    'count_mismatch_by_route_dir' AS check_name,
    COALESCE(s.route_id, t.route_id) AS route_id,
    COALESCE(s.move_dir_code, t.move_dir_code) AS move_dir_code,
    COALESCE(s.staging_cnt, 0) AS staging_distinct_cnt,
    COALESCE(t.promoted_cnt, 0) AS promoted_cnt,
    COALESCE(t.promoted_cnt, 0) - COALESCE(s.staging_cnt, 0) AS diff
FROM stg_counts s
FULL OUTER JOIN tgt_counts t
  ON t.route_id = s.route_id
 AND t.move_dir_code = s.move_dir_code
WHERE COALESCE(s.staging_cnt, 0) <> COALESCE(t.promoted_cnt, 0)
ORDER BY COALESCE(s.route_id, t.route_id), COALESCE(s.move_dir_code, t.move_dir_code);

------------------------------------------------------------
-- 7. key ?⑥쐞 ?꾨씫/珥덇낵 寃異?------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
),
stg_distinct_keys AS (
    SELECT DISTINCT
        route_id_raw::text AS route_id,
        move_dir_raw::integer::text AS move_dir_code,
        link_seq_raw::integer AS link_seq
    FROM public.vw_stg_daegu_route_links_api_dedup
    WHERE btrim(COALESCE(route_id_raw, '')) ~ '^\d+$'
      AND route_id_raw::bigint > 0
      AND route_id_raw::text <> '1000'
      AND route_id_raw::text NOT IN (SELECT route_id FROM excluded_routes)
      AND btrim(COALESCE(move_dir_raw, '')) ~ '^-?\d+$'
      AND btrim(COALESCE(link_seq_raw, '')) ~ '^\d+$'
      AND link_seq_raw::integer > 0
),
tgt_distinct_keys AS (
    SELECT DISTINCT route_id, move_dir_code, link_seq
    FROM public.route_link_sequence
    WHERE route_id <> '1000'
      AND route_id NOT IN (SELECT route_id FROM excluded_routes)
)
SELECT
    'key_gap_between_staging_and_promoted' AS check_name,
    COALESCE(s.route_id, t.route_id) AS route_id,
    COALESCE(s.move_dir_code, t.move_dir_code) AS move_dir_code,
    COALESCE(s.link_seq, t.link_seq) AS link_seq,
    CASE
        WHEN s.route_id IS NOT NULL AND t.route_id IS NULL THEN 'missing_in_promoted'
        WHEN s.route_id IS NULL AND t.route_id IS NOT NULL THEN 'extra_in_promoted'
    END AS issue_type
FROM stg_distinct_keys s
FULL OUTER JOIN tgt_distinct_keys t
  ON t.route_id = s.route_id
 AND t.move_dir_code = s.move_dir_code
 AND t.link_seq = s.link_seq
WHERE s.route_id IS NULL
   OR t.route_id IS NULL
ORDER BY COALESCE(s.route_id, t.route_id), COALESCE(s.move_dir_code, t.move_dir_code), COALESCE(s.link_seq, t.link_seq);

------------------------------------------------------------
-- 8. promoted ?곗냽???⑥젅
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
)
SELECT
    'promoted_continuity_break' AS check_name,
    route_id,
    move_dir_code,
    min(link_seq) AS seq_min,
    max(link_seq) AS seq_max,
    count(*) AS actual_cnt,
    max(link_seq) - min(link_seq) + 1 AS expected_cnt
FROM public.route_link_sequence
WHERE route_id <> '1000'
  AND route_id NOT IN (SELECT route_id FROM excluded_routes)
GROUP BY route_id, move_dir_code
HAVING (max(link_seq) - min(link_seq) + 1) <> count(*)
ORDER BY route_id, move_dir_code;

------------------------------------------------------------
-- 9. 媛숈? ?먯뿰?ㅼ뿉??link_id 異⑸룎
------------------------------------------------------------
WITH excluded_routes AS (
    SELECT *
    FROM (VALUES
        ('4040006020'::text),
        ('4040006021'::text),
        ('4040007001'::text),
        ('4040007009'::text)
    ) v(route_id)
),
stg_valid AS (
    SELECT
        route_id_raw::text AS route_id,
        move_dir_raw::integer::text AS move_dir_code,
        link_seq_raw::integer AS link_seq,
        link_id_raw::text AS staging_link_id
    FROM public.vw_stg_daegu_route_links_api_dedup
    WHERE btrim(COALESCE(route_id_raw, '')) ~ '^\d+$'
      AND route_id_raw::bigint > 0
      AND route_id_raw::text <> '1000'
      AND route_id_raw::text NOT IN (SELECT route_id FROM excluded_routes)
      AND btrim(COALESCE(move_dir_raw, '')) ~ '^-?\d+$'
      AND btrim(COALESCE(link_seq_raw, '')) ~ '^\d+$'
      AND link_seq_raw::integer > 0
      AND btrim(COALESCE(link_id_raw, '')) <> ''
)
SELECT
    'link_id_collision_between_staging_and_promoted' AS check_name,
    s.route_id,
    s.move_dir_code,
    s.link_seq,
    s.staging_link_id,
    t.link_id AS promoted_link_id
FROM stg_valid s
JOIN public.route_link_sequence t
  ON t.route_id = s.route_id
 AND t.move_dir_code = s.move_dir_code
 AND t.link_seq = s.link_seq
WHERE t.route_id <> '1000'
  AND t.route_id NOT IN (SELECT route_id FROM excluded_routes)
  AND btrim(COALESCE(t.link_id, '')) <> ''
  AND s.staging_link_id <> t.link_id
ORDER BY s.route_id, s.move_dir_code, s.link_seq;
