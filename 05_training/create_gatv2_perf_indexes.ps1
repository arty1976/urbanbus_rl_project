param(
    [string]$DbUrl = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus"
)

$py = @'
from sqlalchemy import create_engine, text
import sys

DBURL = sys.argv[1]
e = create_engine(DBURL)

stmts = [
    "CREATE INDEX IF NOT EXISTS idx_gatv2_rows_base_state_ts ON public.gatv2_snapshot_stop_features_train_mat_base (state_ts)",
    "CREATE INDEX IF NOT EXISTS idx_gatv2_rows_base_state_ts_node_index ON public.gatv2_snapshot_stop_features_train_mat_base (state_ts, node_index)",
]

with e.begin() as conn:
    for s in stmts:
        print(s)
        conn.execute(text(s))

print('OK: index statements executed')
'@

$tmp = Join-Path $env:TEMP "create_gatv2_perf_indexes.py"
Set-Content -Path $tmp -Value $py -Encoding UTF8

& "C:\Users\ryujo\AppData\Local\Programs\Python\Python312\python.exe" $tmp $DbUrl
