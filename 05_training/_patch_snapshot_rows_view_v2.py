from sqlalchemy import create_engine, text

DBURL = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus"
SCHEMA = "public"
OBJ = "gatv2_snapshot_stop_features_train_mat"
BASE = "gatv2_snapshot_stop_features_train_mat_base"
SUMMARY = "gatv2_snapshot_summary"

e = create_engine(DBURL)

with e.begin() as conn:
    relkind = conn.execute(text("""
        select c.relkind
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = :schema
          and c.relname = :name
    """), {"schema": SCHEMA, "name": OBJ}).scalar()

    base_relkind = conn.execute(text("""
        select c.relkind
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = :schema
          and c.relname = :name
    """), {"schema": SCHEMA, "name": BASE}).scalar()

    print("current relkind =", relkind, "base relkind =", base_relkind)

    if relkind and not base_relkind:
        if relkind == "r":
            conn.execute(text(f"ALTER TABLE {SCHEMA}.{OBJ} RENAME TO {BASE}"))
        elif relkind == "v":
            conn.execute(text(f"ALTER VIEW {SCHEMA}.{OBJ} RENAME TO {BASE}"))
        elif relkind == "m":
            conn.execute(text(f"ALTER MATERIALIZED VIEW {SCHEMA}.{OBJ} RENAME TO {BASE}"))
        else:
            raise SystemExit(f"Unsupported relkind for {OBJ}: {relkind}")

    relkind = conn.execute(text("""
        select c.relkind
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = :schema
          and c.relname = :name
    """), {"schema": SCHEMA, "name": OBJ}).scalar()

    if relkind == "r":
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{OBJ}"))
    elif relkind == "v":
        conn.execute(text(f"DROP VIEW IF EXISTS {SCHEMA}.{OBJ}"))
    elif relkind == "m":
        conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {SCHEMA}.{OBJ}"))

    conn.execute(text(f"""
        CREATE VIEW {SCHEMA}.{OBJ} AS
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
        FROM {SCHEMA}.{BASE} r
        JOIN {SCHEMA}.{SUMMARY} s
          ON r.state_ts = s.state_ts
    """))

print("OK: created compatibility view public.gatv2_snapshot_stop_features_train_mat")
