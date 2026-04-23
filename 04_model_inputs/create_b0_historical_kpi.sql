-- ============================================================
-- File: create_b0_historical_kpi.sql
-- Purpose:
--   Build Phase-1 S0 artifact table for B0 historical baseline.
--   Contract-first version:
--     - fixes the 6 KPI columns
--     - fills only metrics that are defensible with current data
--     - keeps unsupported KPIs as NULL until richer logs land
-- ============================================================

SET search_path TO public;

DROP TABLE IF EXISTS public.baseline_b0_historical_kpi_by_window;
DROP TABLE IF EXISTS public.baseline_b0_historical_kpi_metadata;

CREATE TABLE public.baseline_b0_historical_kpi_by_window AS
WITH base AS (
    SELECT
        r.state_ts,
        r.state_ts::date AS service_date,
        EXTRACT(HOUR FROM r.state_ts)::int AS hour_of_day,
        CASE
            WHEN EXTRACT(HOUR FROM r.state_ts) BETWEEN 7 AND 9
              OR EXTRACT(HOUR FROM r.state_ts) BETWEEN 17 AND 19
                THEN 'peak'
            WHEN EXTRACT(HOUR FROM r.state_ts) >= 20
              OR EXTRACT(HOUR FROM r.state_ts) < 6
                THEN 'night'
            ELSE 'offpeak'
        END AS time_band,
        r.node_uid,
        COALESCE(r.waiting_passenger_cnt, 0)::double precision AS waiting_passenger_cnt,
        COALESCE(r.next_boardings_recent, 0)::double precision AS next_boardings_recent,
        COALESCE(r.next_alightings_recent, 0)::double precision AS next_alightings_recent,
        COALESCE(r.delta_t_hr_effective, 0)::double precision * 3600.0 AS delta_t_sec
    FROM public.rl_stop_transition_features r
    WHERE COALESCE(r.is_overnight_gap, 0) = 0
            AND COALESCE(r.is_terminal_transition, 0) = 0
),
node_proxy AS (
    SELECT
        state_ts,
        service_date,
        hour_of_day,
        time_band,
        node_uid,
        waiting_passenger_cnt,
        next_boardings_recent,
        next_alightings_recent,
        delta_t_sec,
        CASE
            WHEN delta_t_sec IS NULL OR delta_t_sec <= 0 THEN NULL
            WHEN waiting_passenger_cnt <= 0 THEN 0.0
            WHEN next_boardings_recent <= 0 THEN LEAST(delta_t_sec, 3600.0)
            ELSE LEAST((waiting_passenger_cnt * delta_t_sec) / NULLIF(next_boardings_recent, 0), 14400.0)
        END AS est_wait_seconds_proxy
    FROM base
),
window_agg AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY state_ts) AS window_id,
        state_ts,
        service_date,
        time_band,

        NULL::double precision AS cv_headway,
        AVG(est_wait_seconds_proxy) FILTER (
            WHERE est_wait_seconds_proxy IS NOT NULL
        )::double precision AS avg_wait_seconds,
        NULL::double precision AS bunching_rate,
        NULL::double precision AS on_time_rate,
        0.0::double precision AS intervention_rate,
        NULL::double precision AS energy_proxy,

        COUNT(*)::bigint AS n_nodes,
        COUNT(est_wait_seconds_proxy)::bigint AS n_wait_obs,
        SUM(waiting_passenger_cnt)::double precision AS waiting_sum,
        SUM(next_boardings_recent)::double precision AS next_boardings_sum,
        SUM(next_alightings_recent)::double precision AS next_alightings_sum,
        AVG(delta_t_sec)::double precision AS avg_delta_t_sec
    FROM node_proxy
    GROUP BY state_ts, service_date, time_band
)
SELECT *
FROM window_agg
ORDER BY state_ts;

ALTER TABLE public.baseline_b0_historical_kpi_by_window
    ADD PRIMARY KEY (window_id);

CREATE INDEX idx_baseline_b0_historical_kpi_state_ts
    ON public.baseline_b0_historical_kpi_by_window (state_ts);

CREATE INDEX idx_baseline_b0_historical_kpi_service_date
    ON public.baseline_b0_historical_kpi_by_window (service_date);

COMMENT ON TABLE public.baseline_b0_historical_kpi_by_window IS
'Phase-1 S0 B0 historical baseline KPI table. Contract-first version with 6 fixed KPI columns. Unsupported metrics remain NULL until vehicle-arrival / schedule / intervention / energy logs are available.';

COMMENT ON COLUMN public.baseline_b0_historical_kpi_by_window.cv_headway IS
'Reserved contract column. NULL in v0 because actual vehicle arrival/headway events are not yet available.';

COMMENT ON COLUMN public.baseline_b0_historical_kpi_by_window.avg_wait_seconds IS
'Proxy metric computed from waiting_passenger_cnt, next_boardings_recent, delta_t_hr_effective.';

COMMENT ON COLUMN public.baseline_b0_historical_kpi_by_window.bunching_rate IS
'Reserved contract column. NULL in v0 because actual headway events are not yet available.';

COMMENT ON COLUMN public.baseline_b0_historical_kpi_by_window.on_time_rate IS
'Reserved contract column. NULL in v0 because scheduled-vs-actual arrival data are not yet available.';

COMMENT ON COLUMN public.baseline_b0_historical_kpi_by_window.intervention_rate IS
'Algorithmic intervention rate. Set to 0.0 for B0 historical baseline.';

COMMENT ON COLUMN public.baseline_b0_historical_kpi_by_window.energy_proxy IS
'Reserved contract column. NULL in v0 because idle/acceleration trajectory logs are not yet available.';

CREATE TABLE public.baseline_b0_historical_kpi_metadata AS
SELECT jsonb_build_object(
    'baseline_id', 'B0_historical',
    'artifact_version', 'baseline_v1',
    'kpi_semantics_version', 'b0_contract_first_v0',
    'source_table', 'public.rl_stop_transition_features',
    'row_count', (SELECT COUNT(*) FROM public.baseline_b0_historical_kpi_by_window),
    'supported_metrics', jsonb_build_array(
        'avg_wait_seconds',
        'intervention_rate'
    ),
    'reserved_null_metrics', jsonb_build_array(
        'cv_headway',
        'bunching_rate',
        'on_time_rate',
        'energy_proxy'
    ),
    'time_bands', jsonb_build_array('peak', 'offpeak', 'night'),
    'notes', jsonb_build_array(
        'avg_wait_seconds is a proxy, not a direct observed passenger wait.',
        'intervention_rate=0.0 means no algorithmic controller was applied in B0.',
        'Remaining KPI columns are fixed for contract compatibility and must be backfilled later when richer operational logs arrive.'
    ),
    'created_at', NOW()
) AS metadata;

SELECT
    COUNT(*) AS row_count,
    MIN(state_ts) AS min_state_ts,
    MAX(state_ts) AS max_state_ts,
    COUNT(*) FILTER (WHERE avg_wait_seconds IS NOT NULL) AS wait_filled_rows,
    COUNT(*) FILTER (WHERE cv_headway IS NULL) AS cv_headway_null_rows,
    COUNT(*) FILTER (WHERE intervention_rate = 0.0) AS zero_intervention_rows
FROM public.baseline_b0_historical_kpi_by_window;
