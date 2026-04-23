SELECT
    COUNT(*) AS row_count,
    MIN(state_ts) AS min_state_ts,
    MAX(state_ts) AS max_state_ts,
    COUNT(*) FILTER (WHERE avg_wait_seconds IS NOT NULL) AS wait_filled_rows,
    COUNT(*) FILTER (WHERE cv_headway IS NULL) AS cv_headway_null_rows,
    COUNT(*) FILTER (WHERE intervention_rate = 0.0) AS zero_intervention_rows
FROM public.baseline_b0_historical_kpi_by_window;
