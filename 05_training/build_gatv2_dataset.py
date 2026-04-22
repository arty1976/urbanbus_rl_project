from __future__ import annotations
import numpy as np

"""
build_gatv2_dataset.py
======================
Canonical snapshot-level GATv2 dataset builder for the urbanbus_rl_project.

Purpose
-------
- Build PyTorch Geometric Data objects for stop-level transition prediction.
- Prefer materialized sources when available.
- Keep the full graph (including nodes without any training rows) and use
  node_mask to restrict supervised loss to nodes that have labels.
- Support both smoke-test builds and full builds.

This script is the canonical dataset builder.
It is distinct from lightweight local prototypes such as daegu_pyg_dataset.py.

Expected upstream objects
-------------------------
Preferred:
- public.gatv2_edge_primary_active_mat
- public.gatv2_snapshot_stop_features_train_mat

Fallback:
- public.gatv2_edge_primary_active
- public.gatv2_snapshot_stop_features_train

Always used:
- public.gatv2_node_master_active
- public.gatv2_snapshot_summary

Feature contract
----------------
x columns (9):
1. boardings_recent_log
2. alightings_recent_log
3. waiting_passenger_cnt_log
4. hour_sin
5. hour_cos
6. dow_sin
7. dow_cos
8. is_peak
9. delta_t_hr_effective

y columns (3):
1. next_boardings_recent
2. next_alightings_recent
3. next_waiting_passenger_cnt

edge_attr columns (4):
1. distance_m
2. time_sec
3. generalized_cost
4. long_edge_5km_flag
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from torch_geometric.data import Data

try:
    import torch_geometric
except Exception:  # pragma: no cover - only used for logging
    torch_geometric = None

LOGGER = logging.getLogger("build_gatv2_dataset")

X_COLS = [
    "boardings_recent_log",
    "alightings_recent_log",
    "waiting_passenger_cnt_log",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_peak",
    "delta_t_hr_effective",
]
Y_COLS = [
    "next_boardings_recent",
    "next_alightings_recent",
    "next_waiting_passenger_cnt",
]
EDGE_COLS = [
    "distance_m",
    "time_sec",
    "generalized_cost",
    "long_edge_5km_flag",
]


@dataclass
class BuildConfig:
    db_url: str
    out_dir: str
    train_end: date
    val_end: date
    test_end: date
    use_materialized: bool = True
    max_snapshots: Optional[int] = None
    dry_run: bool = False
    save_format: str = "pt"
    log_dir: Optional[str] = None


class PerfTracker:
    def __init__(self) -> None:
        self.values: Dict[str, List[float]] = defaultdict(list)

    def add(self, phase: str, sec: float) -> None:
        self.values[phase].append(sec)

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for phase, vals in self.values.items():
            if not vals:
                continue
            result[phase] = {
                "count": len(vals),
                "total_sec": round(sum(vals), 3),
                "avg_sec": round(mean(vals), 3),
                "max_sec": round(max(vals), 3),
                "min_sec": round(min(vals), 3),
            }
        return result


@contextmanager
def timed_phase(
    tracker: PerfTracker,
    phase: str,
    *,
    snapshot_id: Optional[int] = None,
    state_ts: Optional[str] = None,
    rows: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        sec = time.perf_counter() - start
        tracker.add(phase, sec)
        payload = {"phase": phase, "snapshot_id": snapshot_id, "state_ts": state_ts, "rows": rows, "sec": round(sec, 3)}
        if extra:
            payload.update(extra)
        msg = " ".join(f"{k}={v}" for k, v in payload.items() if v is not None)
        LOGGER.info("[perf] %s", msg)


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build canonical GATv2 snapshot dataset.")
    parser.add_argument(
        "--db-url",
        default=os.environ.get("URBANBUS_DB_URL") or os.environ.get("URBANBUS_DB_DSN") or "postgresql+psycopg2://postgres:password@localhost:5432/urbanbus",
        help="SQLAlchemy DB URL. Defaults to URBANBUS_DB_URL / URBANBUS_DB_DSN / localhost urbanbus.",
    )
    parser.add_argument("--out-dir", default=str(Path.cwd() / ".." / "data" / "gatv2_dataset"), help="Output directory for split datasets.")
    parser.add_argument("--max-snapshots", type=int, default=None, help="Optional snapshot cap for smoke testing.")
    parser.add_argument("--dry-run", action="store_true", help="Build in memory only. Skip .pt writes.")
    parser.add_argument("--use-materialized", dest="use_materialized", action="store_true", help="Prefer materialized tables if present.")
    parser.add_argument("--no-use-materialized", dest="use_materialized", action="store_false", help="Force view fallback.")
    parser.set_defaults(use_materialized=True)
    parser.add_argument("--log-dir", default=None, help="Optional log directory. Defaults to logs_build_<timestamp>.")
    args = parser.parse_args()

    return BuildConfig(
        db_url=args.db_url,
        out_dir=args.out_dir,
        train_end=date(2023, 10, 31),
        val_end=date(2023, 11, 30),
        test_end=date(2023, 12, 31),
        use_materialized=args.use_materialized,
        max_snapshots=args.max_snapshots,
        dry_run=args.dry_run,
        log_dir=args.log_dir,
    )


def create_db_engine(db_url: str) -> Engine:
    return create_engine(db_url, future=True)


def table_exists(engine: Engine, fq_name: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(text("select to_regclass(:name)"), {"name": fq_name}).scalar()
    return result is not None


def resolve_source_tables(engine: Engine, use_materialized: bool) -> Dict[str, str]:
    sources = {
        "node_master": "public.gatv2_node_master_active",
        "snapshot_summary": "public.gatv2_snapshot_summary",
        "edge": "public.gatv2_edge_primary_active",
        "snapshot_rows": "public.gatv2_snapshot_stop_features_train",
    }
    if use_materialized and table_exists(engine, "public.gatv2_edge_primary_active_mat"):
        sources["edge"] = "public.gatv2_edge_primary_active_mat"
    if use_materialized and table_exists(engine, "public.gatv2_snapshot_stop_features_train_mat"):
        sources["snapshot_rows"] = "public.gatv2_snapshot_stop_features_train_mat"
    return sources


def load_node_master(engine: Engine, source_table: str) -> pd.DataFrame:
    query = text(f"""
        select node_idx, node_uid
        from {source_table}
        order by node_idx
    """)
    df = pd.read_sql_query(query, engine)
    df["node_idx"] = df["node_idx"].astype(int)
    return df


def load_edge_master(engine: Engine, source_table: str) -> pd.DataFrame:
    query = text(f"""
        select
            src_idx,
            dst_idx,
            distance_m,
            time_sec,
            generalized_cost,
            long_edge_5km_flag,
            route_id,
            move_dir_code
        from {source_table}
        order by src_idx, dst_idx
    """)
    df = pd.read_sql_query(query, engine)
    df["src_idx"] = df["src_idx"].astype(int)
    df["dst_idx"] = df["dst_idx"].astype(int)
    return df


def load_snapshot_summary(engine: Engine, source_table: str, max_snapshots: Optional[int]) -> pd.DataFrame:
    query = text(f"""
        select snapshot_id, state_ts, next_state_ts, node_cnt
        from {source_table}
        order by snapshot_id
    """)
    df = pd.read_sql_query(query, engine)
    df["snapshot_id"] = df["snapshot_id"].astype(int)
    df["state_ts"] = pd.to_datetime(df["state_ts"], utc=False)
    df["next_state_ts"] = pd.to_datetime(df["next_state_ts"], utc=False)
    if max_snapshots is not None:
        df = df.head(max_snapshots).copy()
    return df


def load_snapshot_rows(engine: Engine, source_table: str, snapshot_id: int) -> pd.DataFrame:
    query = text(f"""
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
        from {source_table}
        where snapshot_id = :snapshot_id
        order by node_idx
    """)
    return pd.read_sql_query(query, engine, params={"snapshot_id": snapshot_id})


def build_static_graph_tensors(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor(edge_df[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long)
    edge_attr = torch.tensor(edge_df[EDGE_COLS].to_numpy(), dtype=torch.float32)
    return edge_index, edge_attr


def classify_split(ts: pd.Timestamp, cfg: BuildConfig) -> str:
    d = ts.date()
    if d <= cfg.train_end:
        return "train"
    if d <= cfg.val_end:
        return "val"
    if d <= cfg.test_end:
        return "test"
    raise ValueError(f"state_ts {ts} outside configured split range")


def build_snapshot_tensors(snap_rows: pd.DataFrame, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
    x = torch.zeros((num_nodes, len(X_COLS)), dtype=torch.float32)
    y = torch.zeros((num_nodes, len(Y_COLS)), dtype=torch.float32)
    node_mask = torch.zeros((num_nodes,), dtype=torch.bool)

    if snap_rows.empty:
        return x, y, node_mask, [], []

    node_indices = snap_rows["node_idx"].astype(int).to_numpy()
    x_vals = snap_rows[X_COLS].to_numpy(dtype="float32")
    y_vals = snap_rows[Y_COLS].to_numpy(dtype="float32")

    node_indices_t = torch.tensor(node_indices.tolist(), dtype=torch.long)
    x[node_indices_t] = torch.tensor(x_vals.tolist(), dtype=torch.float32)
    y[node_indices_t] = torch.tensor(y_vals.tolist(), dtype=torch.float32)
    node_mask[node_indices_t] = True

    node_uid_list = snap_rows["node_uid"].astype(str).tolist()
    node_idx_list = snap_rows["node_idx"].astype(int).tolist()
    return x, y, node_mask, node_uid_list, node_idx_list


def run_invariant_checks(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    node_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    snapshot_rows: int,
) -> None:
    # C1: node_idx range represented through x/y/mask size
    assert x.shape[0] == num_nodes, f"C1 failed: x rows {x.shape[0]} != num_nodes {num_nodes}"

    # C2: edge_index range
    assert int(edge_index.min()) >= 0, "C2 failed: negative edge index"
    assert int(edge_index.max()) < num_nodes, f"C2 failed: edge_index max {int(edge_index.max())} >= {num_nodes}"

    # C3: self-loop 없음
    assert bool((edge_index[0] != edge_index[1]).all()), "C3 failed: self-loop detected"

    # C4: x shape
    assert x.shape == (num_nodes, len(X_COLS)), f"C4 failed: x.shape={tuple(x.shape)}"

    # C5: y shape
    assert y.shape == (num_nodes, len(Y_COLS)), f"C5 failed: y.shape={tuple(y.shape)}"

    # C6: node_mask shape
    assert node_mask.shape == (num_nodes,), f"C6 failed: node_mask.shape={tuple(node_mask.shape)}"

    # C7: row count == mask sum
    assert int(node_mask.sum().item()) == int(snapshot_rows), (
        f"C7 failed: node_mask.sum={int(node_mask.sum().item())} != snapshot_rows={snapshot_rows}"
    )

    # C8: negative target 없음
    assert bool((y[node_mask] >= 0).all()), "C8 failed: negative target detected"

    assert not torch.isnan(x).any().item(), "Invariant failed: NaN in x"
    assert not torch.isnan(y).any().item(), "Invariant failed: NaN in y"
    assert not torch.isnan(edge_attr).any().item(), "Invariant failed: NaN in edge_attr"


def save_snapshot_artifact(data: Data, out_dir: Path, split_name: str, snapshot_id: int) -> Path:
    target_dir = out_dir / split_name
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"snapshot_{snapshot_id:05d}.pt"
    torch.save(data, out_path)
    return out_path


def build_report(
    cfg: BuildConfig,
    sources: Dict[str, str],
    perf: PerfTracker,
    env: Dict[str, Any],
    num_nodes: int,
    num_edges: int,
    snapshot_df: pd.DataFrame,
    split_counts: Dict[str, int],
    avg_rows_per_snapshot: float,
    processed_snapshots: int,
) -> Dict[str, Any]:
    return {
        "env": env,
        "config": {
            **asdict(cfg),
            "train_end": cfg.train_end.isoformat(),
            "val_end": cfg.val_end.isoformat(),
            "test_end": cfg.test_end.isoformat(),
        },
        "sources": sources,
        "graph": {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        },
        "snapshots": {
            "total_snapshots": int(len(snapshot_df)),
            "processed_snapshots": int(processed_snapshots),
            "min_state_ts": snapshot_df["state_ts"].min().isoformat() if not snapshot_df.empty else None,
            "max_state_ts": snapshot_df["state_ts"].max().isoformat() if not snapshot_df.empty else None,
            "avg_rows_per_snapshot": round(avg_rows_per_snapshot, 3),
            "split_counts": split_counts,
        },
        "timing": perf.summary(),
        "materialized_preferred": cfg.use_materialized,
    }


def main() -> int:
    cfg = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.log_dir) if cfg.log_dir else Path.cwd() / f"logs_build_{timestamp}"
    configure_logging(log_dir / "build.log")
    perf = PerfTracker()

    env = {
        "torch": torch.__version__,
        "pyg": getattr(torch_geometric, "__version__", "unknown"),
        "pandas": pd.__version__,
    }

    with timed_phase(perf, "T0_env_boot"):
        LOGGER.info("[env ] torch %s / pyg %s / pandas %s", env["torch"], env["pyg"], env["pandas"])
        LOGGER.info("[cfg ] %s", cfg)

    with timed_phase(perf, "T1_db_connect"):
        engine = create_db_engine(cfg.db_url)
        with engine.connect() as conn:
            conn.execute(text("select 1"))

    sources = resolve_source_tables(engine, cfg.use_materialized)
    LOGGER.info("[db ] sources: %s", sources)

    with timed_phase(perf, "T2_load_graph_static"):
        node_df = load_node_master(engine, sources["node_master"])
        edge_df = load_edge_master(engine, sources["edge"])
        edge_index, edge_attr = build_static_graph_tensors(node_df, edge_df)
        num_nodes = len(node_df)
        num_edges = edge_index.shape[1]
        LOGGER.info("[graph] num_nodes=%s  num_edges=%s", num_nodes, num_edges)

    with timed_phase(perf, "T3_load_snapshot_summary"):
        snapshot_df = load_snapshot_summary(engine, sources["snapshot_summary"], cfg.max_snapshots)
        total_training_rows = None
        with engine.connect() as conn:
            summary_result = conn.execute(text(f"select count(*) from {sources['snapshot_rows']}"))
            total_training_rows = int(summary_result.scalar() or 0)
        LOGGER.info(
            "[db ] snapshot_summary: %s",
            {
                "distinct_snapshots": len(snapshot_df),
                "total_training_rows": total_training_rows,
                "min_state_ts": snapshot_df["state_ts"].min() if not snapshot_df.empty else None,
                "max_state_ts": snapshot_df["state_ts"].max() if not snapshot_df.empty else None,
            },
        )
        LOGGER.info("[db ] edge_summary   : %s", {"num_nodes": num_nodes, "num_edges": num_edges})

    out_dir = Path(cfg.out_dir)
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    row_counts: List[int] = []
    processed_snapshots = 0

    for i, snap in enumerate(snapshot_df.itertuples(index=False), start=1):
        snapshot_id = int(snap.snapshot_id)
        state_ts = pd.Timestamp(snap.state_ts)
        next_state_ts = pd.Timestamp(snap.next_state_ts)
        split_name = classify_split(state_ts, cfg)

        with timed_phase(perf, "T4_fetch_snapshot_rows", snapshot_id=snapshot_id, state_ts=str(state_ts)):
            snap_rows = load_snapshot_rows(engine, sources["snapshot_rows"], snapshot_id)

        with timed_phase(
            perf,
            "T5_build_dense_tensors",
            snapshot_id=snapshot_id,
            state_ts=str(state_ts),
            rows=len(snap_rows),
        ):
            x, y, node_mask, node_uid_list, node_idx_list = build_snapshot_tensors(snap_rows, num_nodes)

        with timed_phase(
            perf,
            "T5A_invariant_checks",
            snapshot_id=snapshot_id,
            state_ts=str(state_ts),
            rows=len(snap_rows),
        ):
            run_invariant_checks(
                x=x,
                y=y,
                node_mask=node_mask,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                snapshot_rows=len(snap_rows),
            )

        with timed_phase(
            perf,
            "T6_wrap_pyg_data",
            snapshot_id=snapshot_id,
            state_ts=str(state_ts),
            rows=len(snap_rows),
        ):
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                node_mask=node_mask,
            )
            data.snapshot_id = snapshot_id
            data.state_ts = state_ts.isoformat()
            data.next_state_ts = next_state_ts.isoformat()
            data.split = split_name
            data.num_active_nodes = int(node_mask.sum().item())
            data.node_uid_list = node_uid_list
            data.node_idx_list = node_idx_list

        if not cfg.dry_run:
            with timed_phase(
                perf,
                "T7_save_artifact",
                snapshot_id=snapshot_id,
                state_ts=str(state_ts),
                rows=len(snap_rows),
            ):
                save_snapshot_artifact(data, out_dir, split_name, snapshot_id)

        processed_snapshots += 1
        split_counts[split_name] += 1
        row_counts.append(len(snap_rows))
        LOGGER.info(
            "[snap %s/%s] ts=%s rows=%s mask=%s split=%s",
            i,
            len(snapshot_df),
            state_ts.isoformat(),
            len(snap_rows),
            int(node_mask.sum().item()),
            split_name,
        )

    avg_rows_per_snapshot = float(sum(row_counts) / len(row_counts)) if row_counts else 0.0
    report = build_report(
        cfg=cfg,
        sources=sources,
        perf=perf,
        env=env,
        num_nodes=num_nodes,
        num_edges=num_edges,
        snapshot_df=snapshot_df,
        split_counts=split_counts,
        avg_rows_per_snapshot=avg_rows_per_snapshot,
        processed_snapshots=processed_snapshots,
    )

    if not cfg.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "build_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        LOGGER.info("[done] build_report.json -> %s", report_path)

    LOGGER.info("[DONE] dataset build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



