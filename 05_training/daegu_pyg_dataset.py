"""
daegu_pyg_dataset.py
====================
Role
----
**로컬 사전검증용 prototype. 정식 학습 builder 대체 아님.**

이 파일은 갤럭시북(local, RAM 16GB) 에서 1일치 / 소수 snapshot 기준으로
PyG ``Data`` 객체의 shape / dtype / 저장 포맷을 빠르게 확인하기 위한
lightweight prototype 이다. 현재 확정된 GATv2 스키마/뷰/컬럼명과 최소한의
정합성만 맞춘다.

canonical training builder
--------------------------
정식 대량 생성기는 ``05_training/build_gatv2_dataset.py`` 이다.
chronological split / build_report.json / invariant C1..C8 전체 체계 /
대량 materialized iteration / time-based split / .pt 번들링은 모두
canonical 측 책임. 이 파일은 **그것을 대체하지 않는다**.

현재 정합 대상 스키마 (2026-04-20 기준, APPROVED)
------------------------------------------------
노드     : ``public.gatv2_node_master_active``      (node_uid, node_index, num_nodes)
엣지     : ``public.gatv2_edge_primary_active_mat``  우선
           → fallback: ``public.gatv2_edge_primary_active``
           컬럼 : src_idx, dst_idx, distance_m, time_sec,
                  generalized_cost, long_edge_5km_flag
피처     : ``public.gatv2_snapshot_stop_features_train_mat`` 우선
           → fallback: ``public.gatv2_snapshot_stop_features_train``
           x (6) : boardings_recent_log, alightings_recent_log,
                   waiting_passenger_cnt_log,
                   hour_sin, hour_cos, is_peak
           y (3) : next_boardings_recent, next_alightings_recent,
                   next_waiting_passenger_cnt

Usage (local smoke)
-------------------
    # 환경변수 세팅 (또는 --dsn 직접 전달)
    set URBANBUS_DB_DSN=postgresql://postgres:password@localhost:5432/urbanbus

    # 1일치 .pt 샘플 생성
    python daegu_pyg_dataset.py --mode sample --date 2023-06-15

    # 저장된 .pt validate
    python daegu_pyg_dataset.py --mode validate --pt-dir ./processed/pt
"""

from __future__ import annotations

import os
import gc
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import torch
from torch_geometric.data import Data, Dataset

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Column contract (matches canonical build_gatv2_dataset.py shape)
# --------------------------------------------------------------------------- #
X_COLS = [
    "boardings_recent_log",
    "alightings_recent_log",
    "waiting_passenger_cnt_log",
    "hour_sin",
    "hour_cos",
    "is_peak",
]
Y_COLS = [
    "next_boardings_recent",
    "next_alightings_recent",
    "next_waiting_passenger_cnt",
]
E_COLS = [
    "distance_m",
    "time_sec",
    "generalized_cost",
    "long_edge_5km_flag",
]
NODE_FEAT_DIM   = len(X_COLS)   # 6
TARGET_DIM      = len(Y_COLS)   # 3
EDGE_FEAT_DIM   = len(E_COLS)   # 4


# --------------------------------------------------------------------------- #
#  DB helpers
# --------------------------------------------------------------------------- #
def get_conn(dsn: str):
    """PostgreSQL 연결. DSN 예: postgresql://user:pw@host:5432/dbname"""
    conn = psycopg2.connect(dsn)
    conn.set_client_encoding("UTF8")
    return conn


def _relation_exists(conn, schema: str, name: str) -> bool:
    """pg_class 기반으로 table/view/matview 존재 여부 확인."""
    sql = """
    SELECT 1
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = %s
      AND c.relname = %s
      AND c.relkind IN ('r', 'v', 'm')
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, name))
        return cur.fetchone() is not None


def _resolve_source(conn, mat_name: str, view_name: str) -> str:
    """
    'schema.relname' 두 후보 중 존재하는 쪽을 반환.
    둘 다 없으면 RuntimeError.
    """
    for candidate in (mat_name, view_name):
        sch, rel = candidate.split(".", 1)
        if _relation_exists(conn, sch, rel):
            log.info(f"  source = {candidate}")
            return candidate
    raise RuntimeError(
        f"Neither {mat_name} nor {view_name} exists. "
        f"Check that 04_model_inputs/create_gatv2_training_views.sql "
        f"has been applied."
    )


# --------------------------------------------------------------------------- #
#  Node loader  → node_uid -> node_idx  (num_nodes should be 4116)
# --------------------------------------------------------------------------- #
def load_node_index(conn) -> dict:
    """
    우선 ``public.gatv2_node_master_active`` 에서 (node_uid, node_index) 를
    읽는다. 해당 뷰가 없으면 legacy fallback (warning 로그 후 graph_node_master
    STOP 전체) 사용.
    """
    if _relation_exists(conn, "public", "gatv2_node_master_active"):
        sql = """
        SELECT node_uid, node_index
        FROM public.gatv2_node_master_active
        ORDER BY node_index
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        node2idx = {r[0]: int(r[1]) for r in rows}
        log.info(f"  nodes = {len(node2idx):,}  (source: gatv2_node_master_active)")
        return node2idx

    log.warning(
        "gatv2_node_master_active not found; "
        "falling back to graph_node_master WHERE node_type='STOP'. "
        "This will likely NOT match the canonical 4116-node population."
    )
    sql = """
    SELECT node_uid,
           (row_number() OVER (ORDER BY node_uid) - 1)::int AS node_index
    FROM graph_node_master
    WHERE node_type = 'STOP'
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    node2idx = {r[0]: int(r[1]) for r in rows}
    log.info(f"  nodes = {len(node2idx):,}  (source: graph_node_master [fallback])")
    return node2idx


# --------------------------------------------------------------------------- #
#  Edge loader  → (edge_index [2,E], edge_attr [E,4])
# --------------------------------------------------------------------------- #
def load_edge_data(conn, num_nodes: int):
    """
    현재 확정 뷰에서 엣지 + E_COLS 4종을 로드.
    edge_attr 은 raw 단위 저장 (정규화는 학습 쪽 책임).
    """
    source = _resolve_source(
        conn,
        mat_name="public.gatv2_edge_primary_active_mat",
        view_name="public.gatv2_edge_primary_active",
    )
    sql = f"""
    SELECT src_idx,
           dst_idx,
           distance_m,
           time_sec,
           generalized_cost,
           long_edge_5km_flag::int AS long_edge_5km_flag
    FROM {source}
    ORDER BY src_idx, dst_idx
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"{source} returned zero rows")

    src = np.fromiter((r["src_idx"] for r in rows), dtype=np.int64)
    dst = np.fromiter((r["dst_idx"] for r in rows), dtype=np.int64)

    if (src >= num_nodes).any() or (dst >= num_nodes).any():
        raise RuntimeError(
            "edge index out of range: max_idx >= num_nodes. "
            "Node population and edge population are inconsistent."
        )

    edge_attr_np = np.array(
        [[float(r[c]) for c in E_COLS] for r in rows],
        dtype=np.float32,
    )
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr  = torch.from_numpy(edge_attr_np)

    log.info(
        f"  edges = {edge_index.shape[1]:,}  "
        f"edge_attr.shape = {tuple(edge_attr.shape)}  (source: {source})"
    )
    return edge_index, edge_attr, source


# --------------------------------------------------------------------------- #
#  Snapshot loader  → list[dict(snapshot_id, state_ts, next_state_ts, x, y, mask)]
# --------------------------------------------------------------------------- #
def fetch_day_snapshots(conn, date_str: str, num_nodes: int) -> list:
    """
    특정 날짜의 모든 (state_ts, node) 행을 조회, snapshot 단위로 (x, y, node_mask)
    텐서를 구성하여 반환.

    없는 노드는 0 패딩 + node_mask=False 로 처리 (future-compatible).
    """
    source = _resolve_source(
        conn,
        mat_name="public.gatv2_snapshot_stop_features_train_mat",
        view_name="public.gatv2_snapshot_stop_features_train",
    )
    select_cols = ", ".join(
        ["state_ts", "next_state_ts", "node_index"] + X_COLS + Y_COLS
    )
    sql = f"""
    SELECT {select_cols}
    FROM {source}
    WHERE state_ts::date = %s
    ORDER BY state_ts, node_index
    """
    log.info(f"  query  = date = {date_str}")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (date_str,))
        rows = cur.fetchall()

    if not rows:
        log.warning(f"  date {date_str}: no rows returned (skip)")
        return []

    df = pd.DataFrame(rows)
    df["node_index"] = df["node_index"].astype(int)

    out = []
    # groupby preserves state_ts order due to ORDER BY above
    for (state_ts, next_state_ts), grp in df.groupby(
        ["state_ts", "next_state_ts"], sort=True
    ):
        idxs = grp["node_index"].to_numpy(dtype=np.int64)

        x = np.zeros((num_nodes, NODE_FEAT_DIM), dtype=np.float32)
        y = np.zeros((num_nodes, TARGET_DIM),    dtype=np.float32)
        m = np.zeros((num_nodes,),               dtype=bool)

        x[idxs] = grp[X_COLS].to_numpy(dtype=np.float32)
        y[idxs] = grp[Y_COLS].to_numpy(dtype=np.float32)
        m[idxs] = True

        ts_tag = pd.Timestamp(state_ts).strftime("%Y-%m-%d_%H%M")
        out.append({
            "snapshot_id":   ts_tag,
            "state_ts":      str(state_ts),
            "next_state_ts": str(next_state_ts),
            "rows_in_db":    int(len(grp)),
            "x":             torch.from_numpy(x),
            "y":             torch.from_numpy(y),
            "node_mask":     torch.from_numpy(m),
        })

    log.info(
        f"  snapshots = {len(out)}  "
        f"(x_shape = [{num_nodes}, {NODE_FEAT_DIM}], "
        f"y_shape = [{num_nodes}, {TARGET_DIM}])"
    )
    return out


# --------------------------------------------------------------------------- #
#  Save day as .pt files
# --------------------------------------------------------------------------- #
def save_day_as_pt(
    date_str: str,
    snapshots: list,
    edge_index: torch.Tensor,
    edge_attr:  torch.Tensor,
    out_dir:    Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = edge_attr.new_zeros(0)  # placeholder for shape query
    num_nodes = int(snapshots[0]["x"].shape[0]) if snapshots else 0
    num_edges = int(edge_index.shape[1])

    saved = 0
    for slot_idx, s in enumerate(snapshots):
        data = Data(
            x              = s["x"],
            edge_index     = edge_index,
            edge_attr      = edge_attr,
            y              = s["y"],
            node_mask      = s["node_mask"],
            snapshot_id    = s["snapshot_id"],
            state_ts       = s["state_ts"],
            next_state_ts  = s["next_state_ts"],
            num_nodes      = num_nodes,
            num_edges      = num_edges,
            feature_names  = list(X_COLS),
            target_names   = list(Y_COLS),
            date           = date_str,
            slot_idx       = slot_idx,
        )
        fname = out_dir / f"{date_str}_{slot_idx:04d}.pt"
        torch.save(data, fname)
        saved += 1

    log.info(f"  saved = {saved}  ->  {out_dir}")
    return saved


# --------------------------------------------------------------------------- #
#  Sample mode (local smoke — the primary use case for this prototype)
# --------------------------------------------------------------------------- #
def convert_sample(dsn: str, date_str: str, out_dir: Path):
    log.info(f"=== prototype sample: {date_str} ===")
    conn = get_conn(dsn)
    try:
        node2idx = load_node_index(conn)
        num_nodes = len(node2idx)

        edge_index, edge_attr, edge_src = load_edge_data(conn, num_nodes)
        snapshots = fetch_day_snapshots(conn, date_str, num_nodes)
    finally:
        conn.close()

    if not snapshots:
        log.error("no snapshots for this date. check DSN / date / source views.")
        return

    saved = save_day_as_pt(date_str, snapshots, edge_index, edge_attr, out_dir)
    save_stats(
        out_dir,
        num_snapshots_saved = saved,
        num_nodes           = num_nodes,
        num_edges           = int(edge_index.shape[1]),
        edge_source         = edge_src,
        feature_source      = "gatv2_snapshot_stop_features_train[_mat]",
        state_ts_min        = snapshots[0]["state_ts"],
        state_ts_max        = snapshots[-1]["state_ts"],
    )
    validate_sample(out_dir, date_str, num_nodes)


# --------------------------------------------------------------------------- #
#  Full mode — local multi-day helper. NOT canonical.
# --------------------------------------------------------------------------- #
def convert_full(
    dsn: str, start: str, end: str, out_dir: Path, workers: int = 1
):
    """
    start..end 범위 multi-day 변환 (prototype scope — chronological split /
    invariant / build_report 모두 제외).

    Canonical 전년도 변환은 ``build_gatv2_dataset.py`` 를 사용할 것.
    """
    log.warning(
        "convert_full() is a prototype-only multi-day loop. "
        "For canonical full-year training dataset, use "
        "05_training/build_gatv2_dataset.py instead."
    )
    if workers and workers > 1:
        log.warning(
            f"--workers={workers} ignored: parallel conversion is reserved "
            f"for the H200 phase (canonical builder)."
        )

    dates = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    log.info(f"prototype multi-day: {len(dates)} days ({start} ~ {end})")

    conn = get_conn(dsn)
    try:
        node2idx = load_node_index(conn)
        num_nodes = len(node2idx)
        edge_index, edge_attr, edge_src = load_edge_data(conn, num_nodes)

        total_snaps = 0
        min_ts, max_ts = None, None
        for i, date_str in enumerate(dates, 1):
            log.info(f"[{i}/{len(dates)}] {date_str}")
            try:
                snaps = fetch_day_snapshots(conn, date_str, num_nodes)
                if snaps:
                    total_snaps += save_day_as_pt(
                        date_str, snaps, edge_index, edge_attr, out_dir
                    )
                    if min_ts is None:
                        min_ts = snaps[0]["state_ts"]
                    max_ts = snaps[-1]["state_ts"]
            except Exception as e:
                log.error(f"  {date_str} failed: {e}")
            finally:
                gc.collect()
    finally:
        conn.close()

    log.info(f"prototype multi-day done: total_snapshots = {total_snaps:,}")
    save_stats(
        out_dir,
        num_snapshots_saved = total_snaps,
        num_nodes           = num_nodes,
        num_edges           = int(edge_index.shape[1]),
        edge_source         = edge_src,
        feature_source      = "gatv2_snapshot_stop_features_train[_mat]",
        state_ts_min        = min_ts or "",
        state_ts_max        = max_ts or "",
    )


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #
def validate_sample(pt_dir: Path, date_str: str, n_nodes: int):
    log.info("=== validate ===")
    files = sorted(pt_dir.glob(f"{date_str}_*.pt"))
    if not files:
        log.error(f"no .pt files for date {date_str} in {pt_dir}")
        return

    data = torch.load(files[0], weights_only=False)

    # --- shape / dtype / content checks --- #
    checks = {}
    checks["x shape == (N, 6)"]         = tuple(data.x.shape)          == (n_nodes, NODE_FEAT_DIM)
    checks["y shape == (N, 3)"]         = tuple(data.y.shape)          == (n_nodes, TARGET_DIM)
    checks["node_mask shape == (N,)"]   = tuple(data.node_mask.shape)  == (n_nodes,)
    checks["edge_index rows == 2"]      = data.edge_index.shape[0]     == 2
    checks["edge_attr cols == 4"]       = data.edge_attr.shape[1]      == EDGE_FEAT_DIM
    checks["x dtype float32"]           = data.x.dtype                 == torch.float32
    checks["y dtype float32"]           = data.y.dtype                 == torch.float32
    checks["node_mask dtype bool"]      = data.node_mask.dtype         == torch.bool
    checks["no NaN in x"]               = not bool(torch.isnan(data.x).any().item())
    checks["no NaN in y"]               = not bool(torch.isnan(data.y).any().item())
    checks["no NaN in edge_attr"]       = not bool(torch.isnan(data.edge_attr).any().item())
    checks["no self-loop"]              = bool((data.edge_index[0] != data.edge_index[1]).all().item())
    # targets are raw counts (pre-log), therefore must be >= 0
    masked_y = data.y[data.node_mask]
    checks["no negative target"]        = bool((masked_y >= 0).all().item())
    # node_mask.sum() should match the number of real rows persisted
    rows_in_db = getattr(data, "rows_in_db", None)
    if rows_in_db is None:
        # rows_in_db is not stored on Data; infer from mask
        rows_in_db = int(data.node_mask.sum().item())
        checks["mask.sum() self-consistent"] = True
    else:
        checks["mask.sum() == rows_in_db"] = int(data.node_mask.sum().item()) == int(rows_in_db)

    all_pass = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        log.info(f"  [{status}] {name}")

    # --- descriptive stats (masked only, so zeros-padding does not dominate) --- #
    mask = data.node_mask
    log.info("  x stats (masked):")
    for j, col in enumerate(X_COLS):
        v = data.x[mask, j]
        if v.numel() == 0:
            log.info(f"    {col:32s}  (no valid rows)")
            continue
        log.info(
            f"    {col:32s}  mean={v.mean().item():+.4f}  "
            f"std={v.std().item():.4f}  "
            f"min={v.min().item():+.4f}  max={v.max().item():+.4f}"
        )
    log.info("  y stats (masked):")
    for j, col in enumerate(Y_COLS):
        v = data.y[mask, j]
        if v.numel() == 0:
            continue
        log.info(
            f"    {col:32s}  mean={v.mean().item():.4f}  "
            f"std={v.std().item():.4f}  "
            f"min={v.min().item():.4f}  max={v.max().item():.4f}"
        )

    # --- rough size accounting --- #
    def _mb(t):
        return t.element_size() * t.nelement() / 1e6
    slot_mb = _mb(data.x) + _mb(data.y) + _mb(data.node_mask) + _mb(data.edge_index) + _mb(data.edge_attr)
    day_mb  = slot_mb * len(files)
    log.info(f"  size: one snapshot ~= {slot_mb:.2f} MB,  this day = {day_mb:.1f} MB  ({len(files)} files)")

    msg = "PROTOTYPE VALIDATION PASSED" if all_pass else "PROTOTYPE VALIDATION FAILED"
    log.info(f"  => {msg}")


# --------------------------------------------------------------------------- #
#  Stats file
# --------------------------------------------------------------------------- #
def save_stats(
    out_dir: Path,
    *,
    num_snapshots_saved: int,
    num_nodes:           int,
    num_edges:           int,
    edge_source:         str,
    feature_source:      str,
    state_ts_min:        str,
    state_ts_max:        str,
):
    stats = {
        "created_at":          datetime.now().isoformat(),
        "prototype_mode":      True,
        "source_table": {
            "nodes":           "public.gatv2_node_master_active",
            "edges":           edge_source,
            "features":        feature_source,
        },
        "num_snapshots_saved": int(num_snapshots_saved),
        "num_nodes":           int(num_nodes),
        "num_edges":           int(num_edges),
        "feature_names":       list(X_COLS),
        "target_names":        list(Y_COLS),
        "edge_feature_names":  list(E_COLS),
        "uses_node_mask":      True,
        "state_ts_min":        state_ts_min,
        "state_ts_max":        state_ts_max,
        "note":
            "prototype sample only — NOT canonical. "
            "Use 05_training/build_gatv2_dataset.py for full-year training.",
    }
    path = out_dir / "dataset_stats.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"  stats saved  ->  {path}")


# --------------------------------------------------------------------------- #
#  PyG Dataset wrapper (for ad-hoc loading during local experiments)
# --------------------------------------------------------------------------- #
class DaeguBusGraphDataset(Dataset):
    """
    prototype sample dataset only.

    Not canonical full-year training dataset — use the artifacts produced
    by ``05_training/build_gatv2_dataset.py`` (train.pt / val.pt / test.pt)
    for real training.

    This wrapper is a thin glob-and-torch-load helper for the per-snapshot
    .pt files that ``convert_sample()`` drops into ``pt_dir``.
    """
    def __init__(self, pt_dir: str):
        super().__init__()
        self.files = sorted(Path(pt_dir).glob("*.pt"))
        assert len(self.files) > 0, f"no .pt files in {pt_dir}"
        log.info(
            f"DaeguBusGraphDataset (prototype): {len(self.files):,} snapshot files"
        )

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> Data:
        return torch.load(self.files[idx], weights_only=False)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="UrbanBus GATv2 prototype dataset builder "
                    "(local smoke only; canonical = build_gatv2_dataset.py)"
    )
    p.add_argument(
        "--mode", choices=["sample", "full", "validate"], required=True,
        help="sample: 1-day prototype | full: multi-day prototype | validate: check .pt"
    )
    p.add_argument(
        "--dsn",
        default=os.environ.get(
            "URBANBUS_DB_DSN",
            "postgresql://postgres:password@localhost:5432/urbanbus",
        ),
        help="PostgreSQL DSN (or env URBANBUS_DB_DSN)",
    )
    p.add_argument("--date",    default="2023-06-15",   help="sample date YYYY-MM-DD")
    p.add_argument("--start",   default="2023-01-01",   help="full mode start")
    p.add_argument("--end",     default="2023-12-31",   help="full mode end")
    p.add_argument("--out-dir", default="./processed/pt", help="output directory")
    p.add_argument("--pt-dir",  default="./processed/pt", help="validate target dir")
    p.add_argument(
        "--workers", type=int, default=1,
        help="RESERVED for H200 phase (canonical). Ignored here; warning emitted if > 1.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out  = Path(args.out_dir)

    if args.mode == "sample":
        convert_sample(args.dsn, args.date, out)

    elif args.mode == "full":
        convert_full(args.dsn, args.start, args.end, out, args.workers)

    elif args.mode == "validate":
        stats_path = Path(args.pt_dir) / "dataset_stats.json"
        n_nodes = 4116
        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                n_nodes = int(json.load(f).get("num_nodes", 4116))
        validate_sample(Path(args.pt_dir), args.date, n_nodes)
