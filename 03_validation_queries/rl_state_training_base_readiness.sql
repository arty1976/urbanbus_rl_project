-- ============================================================
-- rl_state_training_base_readiness.sql
-- Purpose: Validate RL training base table readiness
-- Note: next_state_ts means next observed timestamp, not strict +1 hour
-- Updated: 2026-04-15
-- ============================================================

WITH base AS (
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT node_uid) AS distinct_node_uid_cnt
    FROM public.rl_state_training_base
),
time_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE state_ts >= next_state_ts) AS time_inversion_cnt
    FROM public.rl_state_training_base
),
null_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE next_state_ts IS NULL) AS null_next_state_ts_cnt,
        COUNT(*) FILTER (WHERE next_boardings_recent IS NULL) AS null_next_boardings_cnt,
        COUNT(*) FILTER (WHERE next_alightings_recent IS NULL) AS null_next_alightings_cnt,
        COUNT(*) FILTER (WHERE next_waiting_passenger_cnt IS NULL) AS null_next_waiting_cnt
    FROM public.rl_state_training_base
),
negative_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE boardings_recent < 0) AS negative_boardings_cnt,
        COUNT(*) FILTER (WHERE alightings_recent < 0) AS negative_alightings_cnt,
        COUNT(*) FILTER (WHERE waiting_passenger_cnt < 0) AS negative_waiting_cnt,
        COUNT(*) FILTER (WHERE next_boardings_recent < 0) AS negative_next_boardings_cnt,
        COUNT(*) FILTER (WHERE next_alightings_recent < 0) AS negative_next_alightings_cnt,
        COUNT(*) FILTER (WHERE next_waiting_passenger_cnt < 0) AS negative_next_waiting_cnt
    FROM public.rl_state_training_base
),
hour_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE hour_of_day < 0 OR hour_of_day > 23) AS invalid_hour_cnt
    FROM public.rl_state_training_base
)

SELECT
    b.total_rows,
    b.distinct_node_uid_cnt,
    t.time_inversion_cnt,
    n.null_next_state_ts_cnt,
    n.null_next_boardings_cnt,
    n.null_next_alightings_cnt,
    n.null_next_waiting_cnt,
    g.negative_boardings_cnt,
    g.negative_alightings_cnt,
    g.negative_waiting_cnt,
    g.negative_next_boardings_cnt,
    g.negative_next_alightings_cnt,
    g.negative_next_waiting_cnt,
    h.invalid_hour_cnt
FROM base b
CROSS JOIN time_checks t
CROSS JOIN null_checks n
CROSS JOIN negative_checks g
CROSS JOIN hour_checks h;

WITH base AS (
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT node_uid) AS distinct_node_uid_cnt
    FROM public.rl_state_training_base
),
time_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE state_ts >= next_state_ts) AS time_inversion_cnt
    FROM public.rl_state_training_base
),
null_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE next_state_ts IS NULL) AS null_next_state_ts_cnt,
        COUNT(*) FILTER (WHERE next_boardings_recent IS NULL) AS null_next_boardings_cnt,
        COUNT(*) FILTER (WHERE next_alightings_recent IS NULL) AS null_next_alightings_cnt,
        COUNT(*) FILTER (WHERE next_waiting_passenger_cnt IS NULL) AS null_next_waiting_cnt
    FROM public.rl_state_training_base
),
negative_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE boardings_recent < 0) AS negative_boardings_cnt,
        COUNT(*) FILTER (WHERE alightings_recent < 0) AS negative_alightings_cnt,
        COUNT(*) FILTER (WHERE waiting_passenger_cnt < 0) AS negative_waiting_cnt,
        COUNT(*) FILTER (WHERE next_boardings_recent < 0) AS negative_next_boardings_cnt,
        COUNT(*) FILTER (WHERE next_alightings_recent < 0) AS negative_next_alightings_cnt,
        COUNT(*) FILTER (WHERE next_waiting_passenger_cnt < 0) AS negative_next_waiting_cnt
    FROM public.rl_state_training_base
),
hour_checks AS (
    SELECT
        COUNT(*) FILTER (WHERE hour_of_day < 0 OR hour_of_day > 23) AS invalid_hour_cnt
    FROM public.rl_state_training_base
)
SELECT
    CASE
        WHEN b.total_rows = 0 THEN 'BLOCKED'
        WHEN t.time_inversion_cnt > 0 THEN 'HOLD'
        WHEN n.null_next_state_ts_cnt > 0 THEN 'HOLD'
        WHEN n.null_next_boardings_cnt > 0 THEN 'HOLD'
        WHEN n.null_next_alightings_cnt > 0 THEN 'HOLD'
        WHEN n.null_next_waiting_cnt > 0 THEN 'HOLD'
        WHEN g.negative_boardings_cnt > 0 THEN 'HOLD'
        WHEN g.negative_alightings_cnt > 0 THEN 'HOLD'
        WHEN g.negative_waiting_cnt > 0 THEN 'HOLD'
        WHEN g.negative_next_boardings_cnt > 0 THEN 'HOLD'
        WHEN g.negative_next_alightings_cnt > 0 THEN 'HOLD'
        WHEN g.negative_next_waiting_cnt > 0 THEN 'HOLD'
        WHEN h.invalid_hour_cnt > 0 THEN 'HOLD'
        ELSE 'APPROVED'
    END AS final_decision,
    CASE
        WHEN b.total_rows = 0 THEN 'NO'
        WHEN t.time_inversion_cnt > 0 THEN 'NO'
        WHEN n.null_next_state_ts_cnt > 0 THEN 'NO'
        WHEN n.null_next_boardings_cnt > 0 THEN 'NO'
        WHEN n.null_next_alightings_cnt > 0 THEN 'NO'
        WHEN n.null_next_waiting_cnt > 0 THEN 'NO'
        WHEN g.negative_boardings_cnt > 0 THEN 'NO'
        WHEN g.negative_alightings_cnt > 0 THEN 'NO'
        WHEN g.negative_waiting_cnt > 0 THEN 'NO'
        WHEN g.negative_next_boardings_cnt > 0 THEN 'NO'
        WHEN g.negative_next_alightings_cnt > 0 THEN 'NO'
        WHEN g.negative_next_waiting_cnt > 0 THEN 'NO'
        WHEN h.invalid_hour_cnt > 0 THEN 'NO'
        ELSE 'YES'
    END AS rl_training_base_ready
FROM base b
CROSS JOIN time_checks t
CROSS JOIN null_checks n
CROSS JOIN negative_checks g
CROSS JOIN hour_checks h;
