param(
    [string]$DbUrl = 'postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus',
    [string]$Schema = 'public',
    [string]$RowsObject = 'gatv2_snapshot_stop_features_train_mat',
    [string]$RowsBackup = 'gatv2_snapshot_stop_features_train_mat_view_backup',
    [string]$RowsBaseObject = 'gatv2_snapshot_stop_features_train_mat_base',
    [string]$SummaryObject = 'gatv2_snapshot_summary',
    [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'

Write-Host ('=' * 88)
Write-Host 'materialize_gatv2_snapshot_rows'
Write-Host "  db_url              = $DbUrl"
Write-Host "  schema              = $Schema"
Write-Host "  rows_object         = $RowsObject"
Write-Host "  rows_backup         = $RowsBackup"
Write-Host "  rows_base_object    = $RowsBaseObject"
Write-Host "  summary_object      = $SummaryObject"
Write-Host "  rebuild             = $($Rebuild.IsPresent)"
Write-Host ('=' * 88)

$py = @"
from sqlalchemy import create_engine, text

DBURL = r'''$DbUrl'''
SCHEMA = r'''$Schema'''
ROWS_OBJ = r'''$RowsObject'''
ROWS_BACKUP = r'''$RowsBackup'''
ROWS_BASE = r'''$RowsBaseObject'''
SUMMARY_OBJ = r'''$SummaryObject'''
REBUILD = $([bool]$Rebuild.IsPresent)

e = create_engine(DBURL)

with e.begin() as conn:
    def relkind(name: str):
        return conn.execute(text("""
            select c.relkind
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = :schema and c.relname = :name
        """), {"schema": SCHEMA, "name": name}).scalar()

    rows_base_relkind = relkind(ROWS_BASE)
    rows_relkind = relkind(ROWS_OBJ)
    rows_backup_relkind = relkind(ROWS_BACKUP)
    print(f"rows_base relkind      = {rows_base_relkind}")
    print(f"rows relkind           = {rows_relkind}")
    print(f"rows_backup relkind    = {rows_backup_relkind}")

    if rows_base_relkind is None:
        raise SystemExit(f"Base object {SCHEMA}.{ROWS_BASE} not found.")

    if REBUILD:
        if rows_relkind == 'm':
            conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {SCHEMA}.{ROWS_OBJ}"))
            print(f"[OK] dropped existing materialized view {ROWS_OBJ}")
            rows_relkind = None
        elif rows_relkind == 'v':
            conn.execute(text(f"DROP VIEW IF EXISTS {SCHEMA}.{ROWS_OBJ}"))
            print(f"[OK] dropped existing view {ROWS_OBJ}")
            rows_relkind = None

    if rows_relkind == 'v' and rows_backup_relkind is None:
        conn.execute(text(f"ALTER VIEW {SCHEMA}.{ROWS_OBJ} RENAME TO {ROWS_BACKUP}"))
        print(f"[OK] renamed existing compatibility view -> {ROWS_BACKUP}")
        rows_relkind = None

    if rows_relkind == 'm':
        conn.execute(text(f"REFRESH MATERIALIZED VIEW {SCHEMA}.{ROWS_OBJ}"))
        print(f"[OK] refreshed materialized view {ROWS_OBJ}")
    elif rows_relkind is None:
        conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {SCHEMA}.{ROWS_OBJ}"))
        conn.execute(text(f"DROP VIEW IF EXISTS {SCHEMA}.{ROWS_OBJ}"))
        conn.execute(text(f"""
            CREATE MATERIALIZED VIEW {SCHEMA}.{ROWS_OBJ} AS
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
            FROM {SCHEMA}.{ROWS_BASE} r
            JOIN {SCHEMA}.{SUMMARY_OBJ} s
              ON r.state_ts = s.state_ts
        """))
        print(f"[OK] created materialized view {ROWS_OBJ}")
    else:
        raise SystemExit(f"Unsupported current relkind for {ROWS_OBJ}: {rows_relkind}")

    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{ROWS_OBJ}_snapshot_id ON {SCHEMA}.{ROWS_OBJ} (snapshot_id)"))
    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{ROWS_OBJ}_snapshot_node ON {SCHEMA}.{ROWS_OBJ} (snapshot_id, node_idx)"))
    conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{ROWS_OBJ}_state_ts ON {SCHEMA}.{ROWS_OBJ} (state_ts)"))
    conn.execute(text(f"ANALYZE {SCHEMA}.{ROWS_OBJ}"))
    print("[OK] ensured indexes + ANALYZE")

with e.connect() as conn:
    stats = conn.execute(text(f"""
        select
            count(*)::bigint as row_count,
            count(distinct snapshot_id)::bigint as snapshot_count,
            min(snapshot_id)::integer as min_snapshot_id,
            max(snapshot_id)::integer as max_snapshot_id,
            min(state_ts) as min_state_ts,
            max(state_ts) as max_state_ts
        from {SCHEMA}.{ROWS_OBJ}
    """)).mappings().one()
    print('-' * 88)
    print('snapshot_rows stats')
    for k, v in stats.items():
        print(f"  {k:18s} = {v}")
    print('-' * 88)
    print('preview')
    rows = conn.execute(text(f"""
        select snapshot_id, state_ts, node_uid, node_idx
        from {SCHEMA}.{ROWS_OBJ}
        order by snapshot_id, node_idx
        limit 5
    """)).fetchall()
    for r in rows:
        print(f"  {r}")
"@

$scriptPath = Join-Path $env:TEMP 'materialize_gatv2_snapshot_rows_tmp.py'
Set-Content -Path $scriptPath -Value $py -Encoding UTF8
try {
    & 'C:\Users\ryujo\AppData\Local\Programs\Python\Python312\python.exe' $scriptPath
}
finally {
    Remove-Item $scriptPath -ErrorAction SilentlyContinue
}

Write-Host ('=' * 88)
Write-Host 'DONE'
