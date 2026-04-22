"""
daegu_pyg_dataset.py
====================
대구 버스 그래프 상태 -> PyTorch Geometric 텐서 변환 파이프라인 (Local Prototype)

[ 중요 - 역할 명시 ]
- 본 파이프라인은 정식 canonical builder(정식 학습용 생성기)가 아닙니다.
- 본 스크립트는 로컬 머신(갤럭시북)에서 1일치 또는 소수 스냅샷을 돌려보며
  shape 검증과 기본적인 .pt 샘플 생성 동작 및 sanity check를 수행하기 위한
  lightweight prototype(경량 프로토타입)입니다.
- 정식 전체 1년치 학습 파이프라인과 full validation은 `build_gatv2_dataset.py`를 사용합니다.

[ 정식 GATv2 입력 명세와의 정합성 ]
- 입력 피처: `gatv2_snapshot_stop_features_train_mat` (또는 view) 참조
- 엣지 소스: `gatv2_edge_primary_active_mat` (또는 view) 참조
- 노드 인덱스: `gatv2_node_master_active` 참조

[ 실행 예시 ]
  # 로컬 샘플 검증 (1일치)
  python daegu_pyg_dataset.py --mode sample --date 2023-06-15

  # 임시 전체 검증
  python daegu_pyg_dataset.py --mode full --start 2023-01-01 --end 2023-12-31

  # 변환 결과 검증
  python daegu_pyg_dataset.py --mode validate --pt-dir ./processed/pt/
"""

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

# -------------------------------------------------
# 로깅 설정
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -------------------------------------------------
# 상수
# -------------------------------------------------
# GATv2 계약에 맞춘 현재 선택 피처
GROUP_A_COLS = [
    "boardings_recent_log",
    "alightings_recent_log",
    "waiting_passenger_cnt_log",
    "hour_sin",
    "hour_cos",
    "is_peak",
]
TARGET_COLS = [
    "next_boardings_recent",
    "next_alightings_recent",
    "next_waiting_passenger_cnt",
]

NODE_FEAT_DIM = len(GROUP_A_COLS)  # 6
EDGE_FEAT_COLS = ["distance_m", "time_sec", "generalized_cost", "long_edge_5km_flag"]

# 프로토타입에서 사용되는 정규화 처리 (로그 피처는 이미 처리되었으므로 단순 클리핑 또는 유지 수행)
# (이미 log 처리되었거나 sin/cos, is_peak 등이라 캡핑을 크게 두거나 1.0으로 둠)
NORM_CAPS = {
    "boardings_recent_log": 10.0,
    "alightings_recent_log": 10.0,
    "waiting_passenger_cnt_log": 10.0,
    "hour_sin": 1.0,
    "hour_cos": 1.0,
    "is_peak": 1.0,
}

# -------------------------------------------------
# DB 연결
# -------------------------------------------------
def get_conn(dsn: str):
    """PostgreSQL 연결 반환."""
    conn = psycopg2.connect(dsn)
    conn.set_client_encoding("UTF8")
    return conn

# -------------------------------------------------
# 노드 마스터 (node_index 맵) 로드
# -------------------------------------------------
def load_node_index(conn) -> dict:
    """
    승인된 active graph 노드 기준으로 정수 인덱스 매핑 반환 (0-based)
    소스: gatv2_node_master_active
    """
    sql_primary = """
    SELECT node_uid, node_index
    FROM public.gatv2_node_master_active
    ORDER BY node_index
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql_primary)
            rows = cur.fetchall()
        node2idx = {row[0]: int(row[1]) for row in rows}
        log.info(f"노드 인덱스 로드 (gatv2_node_master_active): {len(node2idx):,}개 정류장")
        return node2idx
    except Exception as e:
        log.warning(f"gatv2_node_master_active 조회 실패 ({e}). Fallback 적용 중...")
        conn.rollback()
        # Fallback to general table if primary logic fails
        sql_fallback = """
        SELECT node_uid
        FROM graph_node_master
        WHERE node_type = 'STOP'
        ORDER BY node_uid
        """
        with conn.cursor() as cur:
            cur.execute(sql_fallback)
            rows = cur.fetchall()
        node2idx = {row[0]: i for i, row in enumerate(rows)}
        log.info(f"노드 인덱스 로드 (graph_node_master fallback): {len(node2idx):,}개 정류장")
        return node2idx

# -------------------------------------------------
# 엣지 데이터 로드
# -------------------------------------------------
def load_edge_data(conn) -> tuple:
    """
    gatv2_edge_primary_active(_mat) 조회하여 edge_index, edge_attr 생성
    """
    sql_mat = "SELECT src_idx, dst_idx, distance_m, time_sec, generalized_cost, long_edge_5km_flag FROM public.gatv2_edge_primary_active_mat ORDER BY src_idx, dst_idx"
    sql_view = "SELECT src_idx, dst_idx, distance_m, time_sec, generalized_cost, long_edge_5km_flag FROM public.gatv2_edge_primary_active ORDER BY src_idx, dst_idx"
    
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_mat)
            rows = cur.fetchall()
        log.info(f"엣지 데이터 로드 완료 (gatv2_edge_primary_active_mat)")
    except Exception as e:
        log.warning(f"mat 엣지 테이블 조회 실패 ({e}). 뷰로 fallback...")
        conn.rollback()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_view)
            rows = cur.fetchall()
        log.info(f"엣지 데이터 로드 완료 (gatv2_edge_primary_active view)")

    src_list, dst_list = [], []
    feats = []
    
    for r in rows:
        src_list.append(r["src_idx"])
        dst_list.append(r["dst_idx"])
        feats.append([
            float(r["distance_m"]),
            float(r["time_sec"]),
            float(r["generalized_cost"]),
            float(r["long_edge_5km_flag"])
        ])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(feats, dtype=torch.float)

    log.info(f"엣지 텐서 빌드: edge_index={tuple(edge_index.shape)}, edge_attr={tuple(edge_attr.shape)}")
    return edge_index, edge_attr

# -------------------------------------------------
# 하루치 타임슬라이스 피처 조회
# -------------------------------------------------
def fetch_day_features(conn, date_str: str, N: int) -> tuple:
    """
    특정 날짜의 모든 스냅샷별 node feautres 로드 -> 텐서 리스트 반환
    """
    sql_mat = f"""
    SELECT
        state_ts, next_state_ts, node_index,
        COALESCE(boardings_recent_log, 0.0) AS boardings_recent_log,
        COALESCE(alightings_recent_log, 0.0) AS alightings_recent_log,
        COALESCE(waiting_passenger_cnt_log, 0.0) AS waiting_passenger_cnt_log,
        COALESCE(hour_sin, 0.0) AS hour_sin,
        COALESCE(hour_cos, 0.0) AS hour_cos,
        COALESCE(is_peak, 0) AS is_peak,
        COALESCE(next_boardings_recent, 0) AS next_boardings_recent,
        COALESCE(next_alightings_recent, 0) AS next_alightings_recent,
        COALESCE(next_waiting_passenger_cnt, 0) AS next_waiting_passenger_cnt
    FROM public.gatv2_snapshot_stop_features_train_mat
    WHERE state_ts::date = %s
    ORDER BY state_ts, node_index
    """
    
    sql_view = sql_mat.replace("gatv2_snapshot_stop_features_train_mat", "gatv2_snapshot_stop_features_train")
    
    source_table = "gatv2_snapshot_stop_features_train_mat"
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_mat, (date_str,))
            rows = cur.fetchall()
    except Exception as e:
        log.warning(f"mat feature 테이블 조회 실패 ({e}). 뷰로 fallback...")
        conn.rollback()
        source_table = "gatv2_snapshot_stop_features_train"
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_view, (date_str,))
            rows = cur.fetchall()
            
    if not rows:
        log.warning(f"  {date_str}: 데이터 없음 (skip)")
        return [], source_table

    df = pd.DataFrame(rows)
    df["node_idx"] = df["node_index"].astype(int)

    results = []
    # state_ts 기준으로 그룹바잉
    for ts, grp in df.groupby("state_ts", sort=True):
        x = torch.zeros((N, NODE_FEAT_DIM), dtype=torch.float)
        y = torch.zeros((N, 3), dtype=torch.float)
        node_mask = torch.zeros((N,), dtype=torch.bool)
        
        node_idxs = grp["node_idx"].values
        feat_vals = grp[GROUP_A_COLS].values.astype(np.float32)
        target_vals = grp[TARGET_COLS].values.astype(np.float32)
        
        x[node_idxs] = torch.from_numpy(feat_vals)
        y[node_idxs] = torch.from_numpy(target_vals)
        node_mask[node_idxs] = True
        
        # 첫 target 행에서 next_state_ts를 취함 (전체 노드가 동일해야함)
        next_ts_val = grp["next_state_ts"].iloc[0] if "next_state_ts" in grp else None

        results.append((str(ts), str(next_ts_val), x, y, node_mask))

    log.info(f"  {date_str}: {len(results)}개 스냅샷 변환 완료 (N={N})")
    return results, source_table

# -------------------------------------------------
# PyG Data 생성 및 .pt 저장
# -------------------------------------------------
def save_day_as_pt(
    date_str: str,
    snapshots: list,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    out_dir: Path,
    num_nodes: int
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for snapshot_id, (state_ts, next_state_ts, x, y, node_mask) in enumerate(snapshots):
        data = Data(
            x             = x,
            edge_index    = edge_index,
            edge_attr     = edge_attr,
            y             = y,
            node_mask     = node_mask,
            snapshot_id   = snapshot_id,
            state_ts      = state_ts,
            next_state_ts = next_state_ts,
            date          = date_str,
            num_nodes     = num_nodes,
            num_edges     = edge_index.shape[1],
            feature_names = GROUP_A_COLS,
            target_names  = TARGET_COLS
        )
        fname = out_dir / f"{date_str}_{snapshot_id:04d}.pt"
        torch.save(data, fname)
        saved += 1

    log.info(f"  저장: {saved}개 파일 -> {out_dir}")
    return saved

# -------------------------------------------------
# 샘플 변환 (로컬 프로토타입용)
# -------------------------------------------------
def convert_sample(dsn: str, date_str: str, out_dir: Path):
    log.info(f"=== 샘플 변환 시작 (로컬 프로토타입 모드) - {date_str} ===")
    conn = get_conn(dsn)

    node2idx = load_node_index(conn)
    N = len(node2idx)
    edge_index, edge_attr = load_edge_data(conn)
    snapshots, source_table = fetch_day_features(conn, date_str, N)
    conn.close()

    if not snapshots:
        log.error("데이터 없음. DB 연결과 날짜를 확인하세요.")
        return

    saved_cnt = save_day_as_pt(date_str, snapshots, edge_index, edge_attr, out_dir, N)
    save_stats(out_dir, saved_cnt, N, edge_index.shape[1], source_table)
    validate_sample(out_dir, date_str, N)

# -------------------------------------------------
# 전체 변환 루트
# -------------------------------------------------
def convert_full(dsn: str, start: str, end: str, out_dir: Path, workers: int = 1):
    dates = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    log.info(f"전체 변환(프로토타입 모드): {len(dates)}일치 ({start} ~ {end})")
    
    if workers > 1:
        log.warning("WARNING: 'workers' 옵션은 H200 단계용 예약(reserved for H200 phase)이므로 "
                    "본 프로토타입 스크립트에서는 단일 스레드로만 실행됩니다.")

    conn = get_conn(dsn)
    node2idx = load_node_index(conn)
    N = len(node2idx)
    edge_index, edge_attr = load_edge_data(conn)

    total_slots = 0
    used_source = "Unknown"
    for i, date_str in enumerate(dates, 1):
        log.info(f"[{i}/{len(dates)}] {date_str}")
        try:
            snapshots, source_table = fetch_day_features(conn, date_str, N)
            used_source = source_table
            if snapshots:
                total_slots += save_day_as_pt(
                    date_str, snapshots, edge_index, edge_attr, out_dir, N
                )
        except Exception as e:
            log.error(f"  {date_str} 실패: {e}")
        finally:
            gc.collect()

    conn.close()
    log.info(f"\n변환 완료: 총 {total_slots:,}개 슬롯")
    save_stats(out_dir, total_slots, N, edge_index.shape[1], used_source)

# -------------------------------------------------
# 검증
# -------------------------------------------------
def validate_sample(pt_dir: Path, date_str: str, n_nodes: int):
    log.info("\n=== 검증 (Sanity Check) ===")
    files = sorted(pt_dir.glob(f"{date_str}_*.pt"))
    if not files:
        log.error("검증할 pt 파일이 없습니다.")
        return

    data = torch.load(files[0], weights_only=False)

    checks = {
        "x shape [N, 6]": tuple(data.x.shape) == (n_nodes, NODE_FEAT_DIM),
        "y shape [N, 3]": tuple(data.y.shape) == (n_nodes, len(TARGET_COLS)),
        "node_mask shape [N]": tuple(data.node_mask.shape) == (n_nodes,),
        "edge_index rows == 2": data.edge_index.shape[0] == 2,
        "edge_attr cols == 4": data.edge_attr.shape[1] == len(EDGE_FEAT_COLS),
        "node_mask_sum <= active_rows": data.node_mask.sum().item() <= n_nodes, # active node count
        "negative target 없음": not (data.y < 0).any().item(),
        "x NaN 없음": not data.x.isnan().any().item(),
        "y NaN 없음": not data.y.isnan().any().item(),
        "edge_attr NaN 없음": not data.edge_attr.isnan().any().item(),
        "자기루프 없음": bool((data.edge_index[0] != data.edge_index[1]).all()),
    }

    all_pass = True
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        log.info(f"  [{status}] {name}")

    msg = "Sanity Check PASS" if all_pass else "일부 검증 실패 -- 확인 필요"
    log.info(f"\n  => {msg}")

# -------------------------------------------------
# 통계 저장
# -------------------------------------------------
def save_stats(out_dir: Path, total_slots: int, n_nodes: int, n_edges: int, source_table: str):
    stats = {
        "created_at": datetime.now().isoformat(),
        "prototype_mode": True,
        "source_table": source_table,
        "num_snapshots_saved": total_slots,
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "feat_dim": NODE_FEAT_DIM,
        "feature_names": GROUP_A_COLS,
        "target_names": TARGET_COLS,
        "uses_node_mask": True,
    }
    path = out_dir / "dataset_stats.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"통계 저장: {path}")

# -------------------------------------------------
# PyG Dataset 래퍼
# -------------------------------------------------
class DaeguBusGraphDataset(Dataset):
    """
    저장된 로컬 프로토타입 .pt 파일을 PyG Dataset으로 래핑.
    *주의* : 이 클래스는 prototype sample dataset only 이며 
             not canonical full-year training dataset 입니다. 
             정식 학습용은 H200 모델 서버의 canonical loader를 사용하세요.
    """
    def __init__(self, pt_dir: str):
        super().__init__()
        self.files = sorted(Path(pt_dir).glob("*.pt"))
        if len(self.files) == 0:
            log.warning(f"pt 파일이 없습니다. 경로 확인 필요: {pt_dir}")
        else:
            log.info(f"DaeguBusGraphDataset (Prototype): {len(self.files):,}개 슬롯 로드")

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> Data:
        return torch.load(self.files[idx], weights_only=False)

# -------------------------------------------------
# CLI
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="대구 버스 그래프 텐서 변환 (Local Prototype)")
    p.add_argument(
        "--mode", choices=["sample", "full", "validate"],
        required=True,
        help="sample: 1일치 검증 | full: 전체 변환 | validate: pt 검증"
    )
    p.add_argument(
        "--dsn",
        default=os.environ.get(
            "URBANBUS_DB_DSN",
            "postgresql://postgres:password@localhost:5432/urbanbus"
        ),
        help="PostgreSQL DSN (또는 환경변수 URBANBUS_DB_DSN 설정)"
    )
    p.add_argument("--date",    default="2023-06-15",   help="샘플 날짜 (YYYY-MM-DD)")
    p.add_argument("--start",   default="2023-01-01",   help="전체 변환 시작일")
    p.add_argument("--end",     default="2023-12-31",   help="전체 변환 종료일")
    p.add_argument("--out-dir", default="./processed/pt", help="출력 디렉터리")
    p.add_argument("--pt-dir",  default="./processed/pt", help="검증할 pt 디렉터리")
    p.add_argument("--workers", type=int, default=1,    help="병렬 프로세스 수 (reserved for H200 phase - 본 스크립트에선 무시됨)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = Path(args.out_dir)

    if args.mode == "sample":
        convert_sample(args.dsn, args.date, out)
    elif args.mode == "full":
        convert_full(args.dsn, args.start, args.end, out, args.workers)
    elif args.mode == "validate":
        stats_path = Path(args.pt_dir) / "dataset_stats.json"
        n_nodes = 4116
        if stats_path.exists():
            with open(stats_path) as f:
                n_nodes = json.load(f).get("num_nodes", 4116)
        validate_sample(Path(args.pt_dir), args.date, n_nodes)
