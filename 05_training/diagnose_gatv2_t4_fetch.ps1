param(
    [string]$DbUrl = 'postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus',
    [string]$Schema = 'public',
    [string]$RowsObject = 'gatv2_snapshot_stop_features_train_mat',
    [int]$SnapshotId = 1
)

$ErrorActionPreference = 'Stop'

$py = @"
from sqlalchemy import create_engine, text

DBURL = r'''$DbUrl'''
SCHEMA = r'''$Schema'''
ROWS_OBJ = r'''$RowsObject'''
SNAPSHOT_ID = $SnapshotId

e = create_engine(DBURL)

queries = {
    'rows_count': f"""
        select count(*)
        from {SCHEMA}.{ROWS_OBJ}
        where snapshot_id = :snapshot_id
    """,
    'rows_preview': f"""
        select snapshot_id, state_ts, node_uid, node_idx
        from {SCHEMA}.{ROWS_OBJ}
        where snapshot_id = :snapshot_id
        order by node_idx
        limit 5
    """,
    'explain_rows_fetch': f"""
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT)
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
        from {SCHEMA}.{ROWS_OBJ}
        where snapshot_id = :snapshot_id
        order by node_idx
    """
}

with e.connect() as conn:
    for name, q in queries.items():
        print('=' * 88)
        print(name)
        print('-' * 88)
        rows = conn.execute(text(q), {'snapshot_id': SNAPSHOT_ID}).fetchall()
        for r in rows:
            print(r)
        print()
"@

$scriptPath = Join-Path $env:TEMP 'diagnose_gatv2_t4_fetch_tmp.py'
Set-Content -Path $scriptPath -Value $py -Encoding UTF8
try {
    & 'C:\Users\ryujo\AppData\Local\Programs\Python\Python312\python.exe' $scriptPath
}
finally {
    Remove-Item $scriptPath -ErrorAction SilentlyContinue
}
