from sqlalchemy import create_engine, inspect, text

DBURL = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus"
SCHEMA = "public"
ROWS_OBJ = "gatv2_snapshot_stop_features_train_mat"
SUMMARY_OBJ = "gatv2_snapshot_summary"
SUMMARY_BASE = "gatv2_snapshot_summary_base"

e = create_engine(DBURL)
insp = inspect(e)

row_cols = [c["name"] for c in insp.get_columns(ROWS_OBJ, schema=SCHEMA)]
print("snapshot_rows columns =", row_cols)

ts_candidates = ["state_ts", "snapshot_ts", "timeslice_ts", "current_ts", "ts"]
ts_col = next((c for c in ts_candidates if c in row_cols), None)

if not ts_col:
    raise SystemExit(f"No compatible timestamp column found in {ROWS_OBJ}. columns={row_cols}")

with e.begin() as conn:
    relkind = conn.execute(text("""
        select c.relkind
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = :schema
          and c.relname = :name
    """), {"schema": SCHEMA, "name": SUMMARY_OBJ}).scalar()

    base_relkind = conn.execute(text("""
        select c.relkind
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = :schema
          and c.relname = :name
    """), {"schema": SCHEMA, "name": SUMMARY_BASE}).scalar()

    if relkind and not base_relkind:
        if relkind == "v":
            conn.execute(text(f"ALTER VIEW {SCHEMA}.{SUMMARY_OBJ} RENAME TO {SUMMARY_BASE}"))
        elif relkind == "m":
            conn.execute(text(f"ALTER MATERIALIZED VIEW {SCHEMA}.{SUMMARY_OBJ} RENAME TO {SUMMARY_BASE}"))

    relkind = conn.execute(text("""
        select c.relkind
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = :schema
          and c.relname = :name
    """), {"schema": SCHEMA, "name": SUMMARY_OBJ}).scalar()

    if relkind == "v":
        conn.execute(text(f"DROP VIEW IF EXISTS {SCHEMA}.{SUMMARY_OBJ}"))
    elif relkind == "m":
        conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {SCHEMA}.{SUMMARY_OBJ}"))

    conn.execute(text(f"""
        CREATE VIEW {SCHEMA}.{SUMMARY_OBJ} AS
        WITH grouped AS (
            SELECT
                {ts_col} AS state_ts,
                COUNT(*)::integer AS node_cnt
            FROM {SCHEMA}.{ROWS_OBJ}
            GROUP BY {ts_col}
        )
        SELECT
            DENSE_RANK() OVER (ORDER BY state_ts)::integer AS snapshot_id,
            state_ts,
            LEAD(state_ts) OVER (ORDER BY state_ts) AS next_state_ts,
            node_cnt
        FROM grouped
        ORDER BY state_ts
    """))

print("OK: created compatibility view public.gatv2_snapshot_summary")
