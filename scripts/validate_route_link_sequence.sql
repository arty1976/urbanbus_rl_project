-- validate_route_link_sequence.sql
-- Guard-capable validation for public.stg_daegu_route_links_api and public.route_link_sequence
-- Optional psql variables:
--   route_id_raw   : scope to one route_id_raw in staging / one route_id in target (text compare)
--   snapshot_file  : scope to one snapshot_file
--   phase          : pre | post | all   (default: all)

\if :{?route_id_raw}
\else
\set route_id_raw
\endif

\if :{?snapshot_file}
\else
\set snapshot_file
\endif

\if :{?phase}
\else
\set phase all
\endif

-----------------------------------------------------------
-- [Scope Summary]
-----------------------------------------------------------
WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter,
        lower(:'phase') AS phase_filter
),
stg_scope AS (
    SELECT s.*
    FROM public.stg_daegu_route_links_api s
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR s.route_id_raw = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR s.snapshot_file = p.snapshot_file_filter)
),
tgt_scope AS (
    SELECT t.*
    FROM public.route_link_sequence t
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR t.route_id::text = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR t.snapshot_file = p.snapshot_file_filter)
)
SELECT
    'scope' AS section,
    (SELECT phase_filter FROM scope_params) AS phase,
    COALESCE((SELECT route_id_raw_filter FROM scope_params), '<ALL>') AS route_id_raw,
    COALESCE((SELECT snapshot_file_filter FROM scope_params), '<ALL>') AS snapshot_file,
    (SELECT count(*)::text FROM stg_scope) AS staging_rows,
    (SELECT count(*)::text FROM tgt_scope) AS promoted_rows;

-----------------------------------------------------------
-- [PRE-VALIDATION] Target: Staging Layer (Quality Check)
-----------------------------------------------------------
WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
stg_scope AS (
    SELECT s.*
    FROM public.stg_daegu_route_links_api s
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR s.route_id_raw = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR s.snapshot_file = p.snapshot_file_filter)
),
stg_normalized AS (
    SELECT
        s.*,
        btrim(COALESCE(s.route_id_raw, '')) AS route_id_raw_trim,
        btrim(COALESCE(s.move_dir_raw, '')) AS move_dir_raw_trim,
        btrim(COALESCE(s.link_seq_raw, '')) AS link_seq_raw_trim,
        CASE
            WHEN btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$' THEN s.route_id_raw::integer
            ELSE NULL
        END AS route_id_num,
        CASE
            WHEN btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$' THEN s.move_dir_raw::integer
            ELSE NULL
        END AS move_dir_code_num,
        CASE
            WHEN btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$' THEN s.link_seq_raw::integer
            ELSE NULL
        END AS link_seq_num
    FROM stg_scope s
)
SELECT
    'pre_unknown_route' AS check_name,
    route_id_raw,
    snapshot_file,
    count(*) AS issue_count
FROM stg_normalized
WHERE route_id_raw_trim = ''
   OR upper(route_id_raw_trim) = 'UNKNOWN'
   OR route_id_num IS NULL
   OR route_id_num = 0
GROUP BY route_id_raw, snapshot_file
ORDER BY snapshot_file, route_id_raw;

WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
stg_scope AS (
    SELECT s.*
    FROM public.stg_daegu_route_links_api s
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR s.route_id_raw = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR s.snapshot_file = p.snapshot_file_filter)
),
stg_normalized AS (
    SELECT
        s.*,
        btrim(COALESCE(s.move_dir_raw, '')) AS move_dir_raw_trim,
        CASE
            WHEN btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$' THEN s.move_dir_raw::integer
            ELSE NULL
        END AS move_dir_code_num
    FROM stg_scope s
)
SELECT
    'pre_bad_move_dir' AS check_name,
    route_id_raw,
    move_dir_raw,
    link_seq_raw,
    snapshot_file
FROM stg_normalized
WHERE move_dir_raw_trim = ''
   OR move_dir_code_num IS NULL
ORDER BY snapshot_file, route_id_raw, move_dir_raw, link_seq_raw;

WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
stg_scope AS (
    SELECT s.*
    FROM public.stg_daegu_route_links_api s
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR s.route_id_raw = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR s.snapshot_file = p.snapshot_file_filter)
),
stg_normalized AS (
    SELECT
        s.*,
        btrim(COALESCE(s.link_seq_raw, '')) AS link_seq_raw_trim,
        CASE
            WHEN btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$' THEN s.link_seq_raw::integer
            ELSE NULL
        END AS link_seq_num
    FROM stg_scope s
)
SELECT
    'pre_bad_link_seq' AS check_name,
    route_id_raw,
    move_dir_raw,
    link_seq_raw,
    snapshot_file
FROM stg_normalized
WHERE link_seq_raw_trim = ''
   OR link_seq_num IS NULL
   OR link_seq_num <= 0
ORDER BY snapshot_file, route_id_raw, move_dir_raw, link_seq_raw;

WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
stg_scope AS (
    SELECT s.*
    FROM public.stg_daegu_route_links_api s
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR s.route_id_raw = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR s.snapshot_file = p.snapshot_file_filter)
),
stg_normalized AS (
    SELECT
        s.*,
        CASE
            WHEN btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$' THEN s.route_id_raw::integer
            ELSE NULL
        END AS route_id_num,
        CASE
            WHEN btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$' THEN s.move_dir_raw::integer
            ELSE NULL
        END AS move_dir_code_num,
        CASE
            WHEN btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$' THEN s.link_seq_raw::integer
            ELSE NULL
        END AS link_seq_num
    FROM stg_scope s
)
SELECT
    'pre_dup_natural_key' AS check_name,
    route_id_num AS route_id,
    move_dir_code_num AS move_dir_code,
    link_seq_num AS link_seq,
    count(*) AS dup_count
FROM stg_normalized
WHERE route_id_num IS NOT NULL
  AND route_id_num > 0
  AND move_dir_code_num IS NOT NULL
  AND link_seq_num IS NOT NULL
  AND link_seq_num > 0
GROUP BY route_id_num, move_dir_code_num, link_seq_num
HAVING count(*) > 1
ORDER BY route_id_num, move_dir_code_num, link_seq_num;

-----------------------------------------------------------
-- [POST-VALIDATION] Target: Analytical Layer (Integrity Check)
-----------------------------------------------------------
WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
tgt_scope AS (
    SELECT t.*
    FROM public.route_link_sequence t
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR t.route_id::text = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR t.snapshot_file = p.snapshot_file_filter)
)
SELECT
    'post_dup_promoted' AS check_name,
    route_id,
    move_dir_code,
    link_seq,
    count(*) AS dup_count
FROM tgt_scope
GROUP BY route_id, move_dir_code, link_seq
HAVING count(*) > 1
ORDER BY route_id, move_dir_code, link_seq;

WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
tgt_scope AS (
    SELECT t.*
    FROM public.route_link_sequence t
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR t.route_id::text = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR t.snapshot_file = p.snapshot_file_filter)
)
SELECT
    'post_continuity_break' AS check_name,
    route_id,
    move_dir_code,
    min(link_seq) AS seq_min,
    max(link_seq) AS seq_max,
    count(*) AS actual_cnt,
    max(link_seq) - min(link_seq) + 1 AS expected_cnt
FROM tgt_scope
GROUP BY route_id, move_dir_code
HAVING (max(link_seq) - min(link_seq) + 1) <> count(*)
ORDER BY route_id, move_dir_code;

WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
stg_scope AS (
    SELECT s.*
    FROM public.stg_daegu_route_links_api s
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR s.route_id_raw = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR s.snapshot_file = p.snapshot_file_filter)
),
stg_normalized AS (
    SELECT
        s.*,
        CASE
            WHEN btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$' THEN s.route_id_raw::integer
            ELSE NULL
        END AS route_id_num,
        CASE
            WHEN btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$' THEN s.move_dir_raw::integer
            ELSE NULL
        END AS move_dir_code_num,
        CASE
            WHEN btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$' THEN s.link_seq_raw::integer
            ELSE NULL
        END AS link_seq_num,
        CASE
            WHEN btrim(COALESCE(s.link_id_raw::text, '')) ~ '^-?\d+$' THEN s.link_id_raw::bigint
            ELSE NULL
        END AS link_id_num
    FROM stg_scope s
),
stg_valid AS (
    SELECT *
    FROM stg_normalized
    WHERE route_id_num IS NOT NULL
      AND route_id_num > 0
      AND move_dir_code_num IS NOT NULL
      AND link_seq_num IS NOT NULL
      AND link_seq_num > 0
),
tgt_scope AS (
    SELECT t.*
    FROM public.route_link_sequence t
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR t.route_id::text = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR t.snapshot_file = p.snapshot_file_filter)
)
SELECT
    'post_key_collision' AS check_name,
    v.route_id_num AS route_id,
    v.move_dir_code_num AS move_dir_code,
    v.link_seq_num AS link_seq,
    v.link_id_num AS staging_link_id,
    t.link_id AS promoted_link_id,
    v.snapshot_file
FROM stg_valid v
JOIN tgt_scope t
  ON t.route_id = v.route_id_num::text
 AND t.move_dir_code = v.move_dir_code_num::text
 AND t.link_seq = v.link_seq_num
WHERE v.link_id_num IS NOT NULL
  AND t.link_id IS NOT NULL
  AND v.link_id_num::text <> t.link_id::text
ORDER BY v.route_id_num, v.move_dir_code_num, v.link_seq_num;

WITH scope_params AS (
    SELECT
        NULLIF(:'route_id_raw', '') AS route_id_raw_filter,
        NULLIF(:'snapshot_file', '') AS snapshot_file_filter
),
stg_scope AS (
    SELECT s.*
    FROM public.stg_daegu_route_links_api s
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR s.route_id_raw = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR s.snapshot_file = p.snapshot_file_filter)
),
stg_distinct_keys AS (
    SELECT DISTINCT
        s.route_id_raw::integer::text AS route_id,
        s.move_dir_raw::integer::text AS move_dir_code,
        s.link_seq_raw::integer AS link_seq
    FROM stg_scope s
    WHERE btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
      AND s.route_id_raw::integer > 0
      AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$'
      AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$'
      AND s.link_seq_raw::integer > 0
),
stg_counts AS (
    SELECT route_id, move_dir_code, count(*) AS stg_cnt
    FROM stg_distinct_keys
    GROUP BY route_id, move_dir_code
),
tgt_scope AS (
    SELECT t.*
    FROM public.route_link_sequence t
    CROSS JOIN scope_params p
    WHERE (p.route_id_raw_filter IS NULL OR t.route_id::text = p.route_id_raw_filter)
      AND (p.snapshot_file_filter IS NULL OR t.snapshot_file = p.snapshot_file_filter)
),
tgt_counts AS (
    SELECT route_id, move_dir_code, count(*) AS tgt_cnt
    FROM tgt_scope
    GROUP BY route_id, move_dir_code
)
SELECT
    'post_count_mismatch' AS check_name,
    COALESCE(s.route_id, t.route_id) AS route_id,
    COALESCE(s.move_dir_code, t.move_dir_code) AS move_dir_code,
    COALESCE(s.stg_cnt, 0) AS staging_distinct_cnt,
    COALESCE(t.tgt_cnt, 0) AS promoted_cnt,
    COALESCE(t.tgt_cnt, 0) - COALESCE(s.stg_cnt, 0) AS diff
FROM stg_counts s
FULL OUTER JOIN tgt_counts t
  ON t.route_id = s.route_id
 AND t.move_dir_code = s.move_dir_code
WHERE COALESCE(s.stg_cnt, 0) <> COALESCE(t.tgt_cnt, 0)
ORDER BY COALESCE(s.route_id, t.route_id), COALESCE(s.move_dir_code, t.move_dir_code);

-----------------------------------------------------------
-- [Summary]
-----------------------------------------------------------
WITH scope_params AS (
    SELECT lower(:'phase') AS phase_filter
),
issue_summary AS (
    SELECT 'pre_unknown_route' AS check_name, count(*)::bigint AS issue_count
    FROM public.stg_daegu_route_links_api s
    WHERE (NULLIF(:'route_id_raw', '') IS NULL OR s.route_id_raw = NULLIF(:'route_id_raw', ''))
      AND (NULLIF(:'snapshot_file', '') IS NULL OR s.snapshot_file = NULLIF(:'snapshot_file', ''))
      AND (
            btrim(COALESCE(s.route_id_raw, '')) = ''
         OR upper(btrim(COALESCE(s.route_id_raw, ''))) = 'UNKNOWN'
         OR btrim(COALESCE(s.route_id_raw, '')) !~ '^\d+$'
         OR CASE
                WHEN btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$' THEN s.route_id_raw::integer = 0
                ELSE false
            END
      )

    UNION ALL

    SELECT 'pre_bad_move_dir', count(*)::bigint
    FROM public.stg_daegu_route_links_api s
    WHERE (NULLIF(:'route_id_raw', '') IS NULL OR s.route_id_raw = NULLIF(:'route_id_raw', ''))
      AND (NULLIF(:'snapshot_file', '') IS NULL OR s.snapshot_file = NULLIF(:'snapshot_file', ''))
      AND (
            btrim(COALESCE(s.move_dir_raw, '')) = ''
         OR btrim(COALESCE(s.move_dir_raw, '')) !~ '^-?\d+$'
      )

    UNION ALL

    SELECT 'pre_bad_link_seq', count(*)::bigint
    FROM public.stg_daegu_route_links_api s
    WHERE (NULLIF(:'route_id_raw', '') IS NULL OR s.route_id_raw = NULLIF(:'route_id_raw', ''))
      AND (NULLIF(:'snapshot_file', '') IS NULL OR s.snapshot_file = NULLIF(:'snapshot_file', ''))
      AND (
            btrim(COALESCE(s.link_seq_raw, '')) = ''
         OR btrim(COALESCE(s.link_seq_raw, '')) !~ '^\d+$'
         OR CASE
                WHEN btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$' THEN s.link_seq_raw::integer <= 0
                ELSE false
            END
      )

    UNION ALL

    SELECT 'pre_dup_natural_key', count(*)::bigint
    FROM (
        SELECT
            s.route_id_raw::integer AS route_id,
            s.move_dir_raw::integer::text AS move_dir_code,
            s.link_seq_raw::integer AS link_seq
        FROM public.stg_daegu_route_links_api s
        WHERE (NULLIF(:'route_id_raw', '') IS NULL OR s.route_id_raw = NULLIF(:'route_id_raw', ''))
          AND (NULLIF(:'snapshot_file', '') IS NULL OR s.snapshot_file = NULLIF(:'snapshot_file', ''))
          AND btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
          AND s.route_id_raw::integer > 0
          AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$'
          AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$'
          AND s.link_seq_raw::integer > 0
        GROUP BY s.route_id_raw::integer, s.move_dir_raw::integer, s.link_seq_raw::integer
        HAVING count(*) > 1
    ) d

    UNION ALL

    SELECT 'post_dup_promoted', count(*)::bigint
    FROM (
        SELECT t.route_id, t.move_dir_code, t.link_seq
        FROM public.route_link_sequence t
        WHERE (NULLIF(:'route_id_raw', '') IS NULL OR t.route_id::text = NULLIF(:'route_id_raw', ''))
          AND (NULLIF(:'snapshot_file', '') IS NULL OR t.snapshot_file = NULLIF(:'snapshot_file', ''))
        GROUP BY t.route_id, t.move_dir_code, t.link_seq
        HAVING count(*) > 1
    ) d

    UNION ALL

    SELECT 'post_continuity_break', count(*)::bigint
    FROM (
        SELECT t.route_id, t.move_dir_code
        FROM public.route_link_sequence t
        WHERE (NULLIF(:'route_id_raw', '') IS NULL OR t.route_id::text = NULLIF(:'route_id_raw', ''))
          AND (NULLIF(:'snapshot_file', '') IS NULL OR t.snapshot_file = NULLIF(:'snapshot_file', ''))
        GROUP BY t.route_id, t.move_dir_code
        HAVING (max(t.link_seq) - min(t.link_seq) + 1) <> count(*)
    ) d

    UNION ALL

    SELECT 'post_key_collision', count(*)::bigint
    FROM (
        WITH stg_valid AS (
            SELECT
                s.route_id_raw::integer::text AS route_id,
                s.move_dir_raw::integer::text AS move_dir_code,
                s.link_seq_raw::integer AS link_seq,
                s.link_id_raw::bigint AS link_id
            FROM public.stg_daegu_route_links_api s
            WHERE (NULLIF(:'route_id_raw', '') IS NULL OR s.route_id_raw = NULLIF(:'route_id_raw', ''))
              AND (NULLIF(:'snapshot_file', '') IS NULL OR s.snapshot_file = NULLIF(:'snapshot_file', ''))
              AND btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
              AND s.route_id_raw::integer > 0
              AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$'
              AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$'
              AND s.link_seq_raw::integer > 0
              AND btrim(COALESCE(s.link_id_raw::text, '')) ~ '^-?\d+$'
        )
        SELECT 1
        FROM stg_valid s
        JOIN public.route_link_sequence t
          ON t.route_id = s.route_id
         AND t.move_dir_code = s.move_dir_code
         AND t.link_seq = s.link_seq
        WHERE (NULLIF(:'route_id_raw', '') IS NULL OR t.route_id::text = NULLIF(:'route_id_raw', ''))
          AND (NULLIF(:'snapshot_file', '') IS NULL OR t.snapshot_file = NULLIF(:'snapshot_file', ''))
          AND t.link_id IS NOT NULL
          AND s.link_id::text <> t.link_id::text
    ) d

    UNION ALL

    SELECT 'post_count_mismatch', count(*)::bigint
    FROM (
        WITH stg_distinct_keys AS (
            SELECT DISTINCT
                s.route_id_raw::integer::text AS route_id,
                s.move_dir_raw::integer::text AS move_dir_code,
                s.link_seq_raw::integer AS link_seq
            FROM public.stg_daegu_route_links_api s
            WHERE (NULLIF(:'route_id_raw', '') IS NULL OR s.route_id_raw = NULLIF(:'route_id_raw', ''))
              AND (NULLIF(:'snapshot_file', '') IS NULL OR s.snapshot_file = NULLIF(:'snapshot_file', ''))
              AND btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
              AND s.route_id_raw::integer > 0
              AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$'
              AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$'
              AND s.link_seq_raw::integer > 0
        ),
        stg_counts AS (
            SELECT route_id, move_dir_code, count(*) AS stg_cnt
            FROM stg_distinct_keys
            GROUP BY route_id, move_dir_code
        ),
        tgt_counts AS (
            SELECT t.route_id, t.move_dir_code, count(*) AS tgt_cnt
            FROM public.route_link_sequence t
            WHERE (NULLIF(:'route_id_raw', '') IS NULL OR t.route_id::text = NULLIF(:'route_id_raw', ''))
              AND (NULLIF(:'snapshot_file', '') IS NULL OR t.snapshot_file = NULLIF(:'snapshot_file', ''))
            GROUP BY t.route_id, t.move_dir_code
        )
        SELECT 1
        FROM stg_counts s
        FULL OUTER JOIN tgt_counts t
          ON t.route_id = s.route_id
         AND t.move_dir_code = s.move_dir_code
        WHERE COALESCE(s.stg_cnt, 0) <> COALESCE(t.tgt_cnt, 0)
    ) d
)
SELECT
    check_name,
    issue_count,
    CASE
        WHEN issue_count = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM issue_summary
CROSS JOIN scope_params p
WHERE (
        p.phase_filter = 'all'
     OR (p.phase_filter = 'pre'  AND check_name LIKE 'pre_%')
     OR (p.phase_filter = 'post' AND check_name LIKE 'post_%')
      )
ORDER BY check_name;

SELECT set_config('urbanbus_rl.phase', lower(COALESCE(NULLIF(:'phase', ''), 'all')), true);
SELECT set_config('urbanbus_rl.route_id_raw', COALESCE(NULLIF(:'route_id_raw', ''), ''), true);
SELECT set_config('urbanbus_rl.snapshot_file', COALESCE(NULLIF(:'snapshot_file', ''), ''), true);

DO $$
DECLARE
    v_phase text := current_setting('urbanbus_rl.phase', true);
    v_route_id_raw text := NULLIF(current_setting('urbanbus_rl.route_id_raw', true), '');
    v_snapshot_file text := NULLIF(current_setting('urbanbus_rl.snapshot_file', true), '');
    v_issue_count bigint;
BEGIN
    WITH issue_summary AS (
        SELECT 'pre_unknown_route' AS check_name, count(*)::bigint AS issue_count
        FROM public.stg_daegu_route_links_api s
        WHERE (v_route_id_raw IS NULL OR s.route_id_raw = v_route_id_raw)
          AND (v_snapshot_file IS NULL OR s.snapshot_file = v_snapshot_file)
          AND (
                btrim(COALESCE(s.route_id_raw, '')) = ''
             OR upper(btrim(COALESCE(s.route_id_raw, ''))) = 'UNKNOWN'
             OR btrim(COALESCE(s.route_id_raw, '')) !~ '^\d+$'
             OR CASE
                    WHEN btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$' THEN s.route_id_raw::integer = 0
                    ELSE false
                END
          )

        UNION ALL

        SELECT 'pre_bad_move_dir', count(*)::bigint
        FROM public.stg_daegu_route_links_api s
        WHERE (v_route_id_raw IS NULL OR s.route_id_raw = v_route_id_raw)
          AND (v_snapshot_file IS NULL OR s.snapshot_file = v_snapshot_file)
          AND (
                btrim(COALESCE(s.move_dir_raw, '')) = ''
             OR btrim(COALESCE(s.move_dir_raw, '')) !~ '^-?\d+$'
          )

        UNION ALL

        SELECT 'pre_bad_link_seq', count(*)::bigint
        FROM public.stg_daegu_route_links_api s
        WHERE (v_route_id_raw IS NULL OR s.route_id_raw = v_route_id_raw)
          AND (v_snapshot_file IS NULL OR s.snapshot_file = v_snapshot_file)
          AND (
                btrim(COALESCE(s.link_seq_raw, '')) = ''
             OR btrim(COALESCE(s.link_seq_raw, '')) !~ '^\d+$'
             OR CASE
                    WHEN btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$' THEN s.link_seq_raw::integer <= 0
                    ELSE false
                END
          )

        UNION ALL

        SELECT 'pre_dup_natural_key', count(*)::bigint
        FROM (
            SELECT
                s.route_id_raw::integer::text AS route_id,
                s.move_dir_raw::integer::text AS move_dir_code,
                s.link_seq_raw::integer AS link_seq
            FROM public.stg_daegu_route_links_api s
            WHERE (v_route_id_raw IS NULL OR s.route_id_raw = v_route_id_raw)
              AND (v_snapshot_file IS NULL OR s.snapshot_file = v_snapshot_file)
              AND btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
              AND s.route_id_raw::integer > 0
              AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$'
              AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$'
              AND s.link_seq_raw::integer > 0
            GROUP BY s.route_id_raw::integer, s.move_dir_raw::integer, s.link_seq_raw::integer
            HAVING count(*) > 1
        ) d

        UNION ALL

        SELECT 'post_dup_promoted', count(*)::bigint
        FROM (
            SELECT t.route_id, t.move_dir_code, t.link_seq
            FROM public.route_link_sequence t
            WHERE (v_route_id_raw IS NULL OR t.route_id::text = v_route_id_raw)
              AND (v_snapshot_file IS NULL OR t.snapshot_file = v_snapshot_file)
            GROUP BY t.route_id, t.move_dir_code, t.link_seq
            HAVING count(*) > 1
        ) d

        UNION ALL

        SELECT 'post_continuity_break', count(*)::bigint
        FROM (
            SELECT t.route_id, t.move_dir_code
            FROM public.route_link_sequence t
            WHERE (v_route_id_raw IS NULL OR t.route_id::text = v_route_id_raw)
              AND (v_snapshot_file IS NULL OR t.snapshot_file = v_snapshot_file)
            GROUP BY t.route_id, t.move_dir_code
            HAVING (max(t.link_seq) - min(t.link_seq) + 1) <> count(*)
        ) d

        UNION ALL

        SELECT 'post_key_collision', count(*)::bigint
        FROM (
            WITH stg_valid AS (
                SELECT
                    s.route_id_raw::integer::text AS route_id,
                    s.move_dir_raw::integer::text AS move_dir_code,
                    s.link_seq_raw::integer AS link_seq,
                    s.link_id_raw::bigint AS link_id
                FROM public.stg_daegu_route_links_api s
                WHERE (v_route_id_raw IS NULL OR s.route_id_raw = v_route_id_raw)
                  AND (v_snapshot_file IS NULL OR s.snapshot_file = v_snapshot_file)
                  AND btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
                  AND s.route_id_raw::integer > 0
                  AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$'
                  AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$'
                  AND s.link_seq_raw::integer > 0
                  AND btrim(COALESCE(s.link_id_raw::text, '')) ~ '^-?\d+$'
            )
            SELECT 1
            FROM stg_valid s
            JOIN public.route_link_sequence t
              ON t.route_id = s.route_id
             AND t.move_dir_code = s.move_dir_code
             AND t.link_seq = s.link_seq
            WHERE (v_route_id_raw IS NULL OR t.route_id::text = v_route_id_raw)
              AND (v_snapshot_file IS NULL OR t.snapshot_file = v_snapshot_file)
              AND t.link_id IS NOT NULL
              AND s.link_id::text <> t.link_id::text
        ) d

        UNION ALL

        SELECT 'post_count_mismatch', count(*)::bigint
        FROM (
            WITH stg_distinct_keys AS (
                SELECT DISTINCT
                    s.route_id_raw::integer::text AS route_id,
                    s.move_dir_raw::integer::text AS move_dir_code,
                    s.link_seq_raw::integer AS link_seq
                FROM public.stg_daegu_route_links_api s
                WHERE (v_route_id_raw IS NULL OR s.route_id_raw = v_route_id_raw)
                  AND (v_snapshot_file IS NULL OR s.snapshot_file = v_snapshot_file)
                  AND btrim(COALESCE(s.route_id_raw, '')) ~ '^\d+$'
                  AND s.route_id_raw::integer > 0
                  AND btrim(COALESCE(s.move_dir_raw, '')) ~ '^-?\d+$'
                  AND btrim(COALESCE(s.link_seq_raw, '')) ~ '^\d+$'
                  AND s.link_seq_raw::integer > 0
            ),
            stg_counts AS (
                SELECT route_id, move_dir_code, count(*) AS stg_cnt
                FROM stg_distinct_keys
                GROUP BY route_id, move_dir_code
            ),
            tgt_counts AS (
                SELECT t.route_id, t.move_dir_code, count(*) AS tgt_cnt
                FROM public.route_link_sequence t
                WHERE (v_route_id_raw IS NULL OR t.route_id::text = v_route_id_raw)
                  AND (v_snapshot_file IS NULL OR t.snapshot_file = v_snapshot_file)
                GROUP BY t.route_id, t.move_dir_code
            )
            SELECT 1
            FROM stg_counts s
            FULL OUTER JOIN tgt_counts t
              ON t.route_id = s.route_id
             AND t.move_dir_code = s.move_dir_code
            WHERE COALESCE(s.stg_cnt, 0) <> COALESCE(t.tgt_cnt, 0)
        ) d
    )
    SELECT COALESCE(sum(issue_count), 0)
      INTO v_issue_count
    FROM issue_summary
    WHERE (
            v_phase = 'all'
         OR (v_phase = 'pre'  AND check_name LIKE 'pre_%')
         OR (v_phase = 'post' AND check_name LIKE 'post_%')
          );

    IF v_issue_count > 0 THEN
        RAISE EXCEPTION 'validate_route_link_sequence failed. phase=%, issue_count=%', v_phase, v_issue_count;
    END IF;
END
$$;
