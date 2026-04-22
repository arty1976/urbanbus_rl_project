param(
    [string]$DbUrl = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus"
)

$py = @'
from sqlalchemy import create_engine, inspect, text
import json
import sys

DBURL = sys.argv[1]
e = create_engine(DBURL)

queries = {
    "indexes_snapshot_rows_base": """
        select schemaname, tablename, indexname, indexdef
        from pg_indexes
        where schemaname='public'
          and tablename in (
            'gatv2_snapshot_stop_features_train_mat_base',
            'gatv2_snapshot_stop_features_train_mat',
            'gatv2_snapshot_summary',
            'gatv2_edge_primary_active',
            'gatv2_node_master_active'
          )
        order by tablename, indexname
    """,
    "count_distinct_state_ts_base": """
        select count(distinct state_ts) as distinct_state_ts
        from public.gatv2_snapshot_stop_features_train_mat_base
    """,
    "snapshot_summary_preview": """
        select snapshot_id, state_ts, next_state_ts, node_cnt
        from public.gatv2_snapshot_summary
        order by snapshot_id
        limit 10
    """,
    "snapshot_rows_count_per_snapshot": """
        select snapshot_id, count(*) as row_cnt
        from public.gatv2_snapshot_stop_features_train_mat
        group by snapshot_id
        order by snapshot_id
        limit 10
    """,
    "explain_summary": """
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
        select snapshot_id, state_ts, next_state_ts, node_cnt
        from public.gatv2_snapshot_summary
        order by snapshot_id
    """,
    "explain_rows_snapshot1": """
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
        select
            snapshot_id,
            state_ts,
            next_state_ts,
            node_uid,
            node_idx,
            boardings_recent_log,
            alightings_recent_log,
            waiting_passenger_cnt_log,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            is_peak,
            delta_t_hr_effective,
            next_boardings_recent,
            next_alightings_recent,
            next_waiting_passenger_cnt
        from public.gatv2_snapshot_stop_features_train_mat
        where snapshot_id = 1
        order by node_idx
    """,
}

with e.connect() as conn:
    for name, q in queries.items():
        print("=" * 88)
        print(name)
        print("-" * 88)
        rows = conn.execute(text(q)).fetchall()
        for r in rows:
            print(r)
        if not rows:
            print("(no rows)")
        print()
'@

$tmp = Join-Path $env:TEMP "diagnose_gatv2_db_perf.py"
Set-Content -Path $tmp -Value $py -Encoding UTF8

& "C:\Users\ryujo\AppData\Local\Programs\Python\Python312\python.exe" $tmp $DbUrl
