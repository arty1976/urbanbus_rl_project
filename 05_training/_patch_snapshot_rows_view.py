from sqlalchemy import create_engine, text

DBURL = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus"
e = create_engine(DBURL)

with e.begin() as conn:
    conn.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_matviews
                WHERE schemaname = 'public'
                  AND matviewname = 'gatv2_snapshot_stop_features_train_mat'
            ) AND NOT EXISTS (
                SELECT 1
                FROM pg_matviews
                WHERE schemaname = 'public'
                  AND matviewname = 'gatv2_snapshot_stop_features_train_mat_base'
            ) THEN
                EXECUTE 'ALTER MATERIALIZED VIEW public.gatv2_snapshot_stop_features_train_mat RENAME TO gatv2_snapshot_stop_features_train_mat_base';
            END IF;
        END
        $$;
    """))

    conn.execute(text("DROP VIEW IF EXISTS public.gatv2_snapshot_stop_features_train_mat"))

    conn.execute(text("""
        CREATE VIEW public.gatv2_snapshot_stop_features_train_mat AS
        SELECT
            s.snapshot_id,
            r.state_ts,
            r.next_state_ts,
            r.node_uid,
            r.node_index AS node_idx,
            r.boardings_recent_log,
            r.alightings_recent_log,
            r.waiting_passenger_cnt_log,
            r.hour_sin,
            r.hour_cos,
            r.dow_sin,
            r.dow_cos,
            r.is_peak,
            r.delta_t_hr_effective,
            r.next_boardings_recent,
            r.next_alightings_recent,
            r.next_waiting_passenger_cnt
        FROM public.gatv2_snapshot_stop_features_train_mat_base r
        JOIN public.gatv2_snapshot_summary s
          ON r.state_ts = s.state_ts
    """))

print("OK: created compatibility view public.gatv2_snapshot_stop_features_train_mat")
