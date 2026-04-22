param(
    [string]$DbUrl = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus",
    [string]$PythonExe = "C:\Users\ryujo\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$Schema = "public",
    [string]$SummaryObject = "gatv2_snapshot_summary",
    [string]$SummaryBackupObject = "gatv2_snapshot_summary_view_backup",
    [string]$RowsBaseObject = "gatv2_snapshot_stop_features_train_mat_base",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$pyFile = Join-Path $env:TEMP "materialize_gatv2_snapshot_summary.py"

$pyCode = @'
from sqlalchemy import create_engine, text
import argparse
import sys


def qscalar(conn, sql, params=None):
    return conn.execute(text(sql), params or {}).scalar()


def get_relkind(conn, schema, name):
    return qscalar(
        conn,
        """
        select c.relkind
        from pg_class c
        join pg_namespace n on n.oid = c.relnamespace
        where n.nspname = :schema
          and c.relname = :name
        """,
        {"schema": schema, "name": name},
    )


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


parser = argparse.ArgumentParser()
parser.add_argument("--db-url", required=True)
parser.add_argument("--schema", required=True)
parser.add_argument("--summary-object", required=True)
parser.add_argument("--summary-backup-object", required=True)
parser.add_argument("--rows-base-object", required=True)
parser.add_argument("--rebuild", action="store_true")
args = parser.parse_args()

engine = create_engine(args.db_url)
qschema = quote_ident(args.schema)
qsummary = quote_ident(args.summary_object)
qbackup = quote_ident(args.summary_backup_object)
qrows = quote_ident(args.rows_base_object)

print("=" * 88)
print("materialize_gatv2_snapshot_summary")
print(f"  db_url              = {args.db_url}")
print(f"  schema              = {args.schema}")
print(f"  summary_object      = {args.summary_object}")
print(f"  summary_backup      = {args.summary_backup_object}")
print(f"  rows_base_object    = {args.rows_base_object}")
print(f"  rebuild             = {args.rebuild}")
print("=" * 88)

with engine.begin() as conn:
    rows_kind = get_relkind(conn, args.schema, args.rows_base_object)
    summary_kind = get_relkind(conn, args.schema, args.summary_object)
    backup_kind = get_relkind(conn, args.schema, args.summary_backup_object)

    print(f"rows_base relkind     = {rows_kind}")
    print(f"summary relkind       = {summary_kind}")
    print(f"backup relkind        = {backup_kind}")

    if rows_kind not in ("r", "m", "v"):
        raise SystemExit(
            f"Required source object {args.schema}.{args.rows_base_object} not found or unsupported (relkind={rows_kind})."
        )

    if summary_kind == "v":
        if backup_kind is None:
            conn.execute(text(f"ALTER VIEW {qschema}.{qsummary} RENAME TO {qbackup}"))
            print(f"[OK] renamed existing compatibility view -> {args.summary_backup_object}")
        else:
            conn.execute(text(f"DROP VIEW IF EXISTS {qschema}.{qsummary}"))
            print(f"[OK] dropped current view {args.summary_object} because backup already exists")
        summary_kind = None

    elif summary_kind == "m" and args.rebuild:
        conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {qschema}.{qsummary}"))
        print(f"[OK] dropped existing materialized view {args.summary_object} for rebuild")
        summary_kind = None

    elif summary_kind == "r":
        raise SystemExit(
            f"Unexpected table at {args.schema}.{args.summary_object}. Rename or remove it first."
        )

    if summary_kind is None:
        conn.execute(
            text(
                f"""
                CREATE MATERIALIZED VIEW {qschema}.{qsummary} AS
                WITH grouped AS (
                    SELECT
                        state_ts,
                        COUNT(*)::integer AS node_cnt
                    FROM {qschema}.{qrows}
                    GROUP BY state_ts
                )
                SELECT
                    DENSE_RANK() OVER (ORDER BY state_ts)::integer AS snapshot_id,
                    state_ts,
                    LEAD(state_ts) OVER (ORDER BY state_ts) AS next_state_ts,
                    node_cnt
                FROM grouped
                ORDER BY state_ts
                """
            )
        )
        print(f"[OK] created materialized view {args.summary_object}")
    else:
        conn.execute(text(f"REFRESH MATERIALIZED VIEW {qschema}.{qsummary}"))
        print(f"[OK] refreshed materialized view {args.summary_object}")

    conn.execute(
        text(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_gatv2_snapshot_summary_snapshot_id ON {qschema}.{qsummary} (snapshot_id)"
        )
    )
    conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS idx_gatv2_snapshot_summary_state_ts ON {qschema}.{qsummary} (state_ts)"
        )
    )
    conn.execute(text(f"ANALYZE {qschema}.{qsummary}"))
    print("[OK] ensured indexes + ANALYZE")

with engine.connect() as conn:
    stats = conn.execute(
        text(
            f"""
            select
                count(*)::integer as snapshot_count,
                min(snapshot_id)::integer as min_snapshot_id,
                max(snapshot_id)::integer as max_snapshot_id,
                min(state_ts) as min_state_ts,
                max(state_ts) as max_state_ts,
                min(node_cnt)::integer as min_node_cnt,
                avg(node_cnt)::numeric(18,2) as avg_node_cnt,
                max(node_cnt)::integer as max_node_cnt
            from {qschema}.{qsummary}
            """
        )
    ).mappings().one()

    print("-" * 88)
    print("snapshot_summary stats")
    for k, v in stats.items():
        print(f"  {k:18} = {v}")

    preview = conn.execute(
        text(
            f"""
            select snapshot_id, state_ts, next_state_ts, node_cnt
            from {qschema}.{qsummary}
            order by snapshot_id
            limit 5
            """
        )
    ).fetchall()
    print("-" * 88)
    print("preview")
    for row in preview:
        print(f"  {row}")

print("=" * 88)
print("DONE")
'@

Set-Content -Path $pyFile -Value $pyCode -Encoding UTF8

try {
    $argsList = @(
        $pyFile,
        "--db-url", $DbUrl,
        "--schema", $Schema,
        "--summary-object", $SummaryObject,
        "--summary-backup-object", $SummaryBackupObject,
        "--rows-base-object", $RowsBaseObject
    )
    if ($Rebuild) {
        $argsList += "--rebuild"
    }

    & $PythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Python materialization helper failed with exit code $LASTEXITCODE"
    }
}
finally {
    if (Test-Path $pyFile) {
        Remove-Item $pyFile -Force -ErrorAction SilentlyContinue
    }
}
