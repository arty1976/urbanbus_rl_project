param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [switch]$RunSelfTest
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$AdaptersDir = ".\05_training\adapters"
New-Item -ItemType Directory -Force -Path $AdaptersDir | Out-Null

# ---------------------------------------------------------------------
# Step 98 file 1:
# 05_training/adapters/signal_feature_builder_contract_v2.md
# ---------------------------------------------------------------------
$ContractMd = @'
# Step 98 — Signal Feature Builder for Causal Simulator v2

## 0. 목적

Step 98의 목적은 Step 97에서 분류한 사용 가능 데이터 중 **정적 신호 인프라 feature**만을 causal simulator v2 입력 후보로 변환하는 것이다.

이 단계는 신호등 CSV(Comma-Separated Values=쉼표 구분값)를 읽어 다음 산출물을 생성한다.

- `node_signal_features.parquet`
- `edge_signal_features.parquet`
- `tensor_signal_feature_contract_v2.json`
- `signal_feature_quality_report.json`

---

## 1. 중요한 금지선

신호등 CSV는 동적 신호 현시 자료가 아니다.

따라서 아래 값은 생성하지 않는다.

| 금지 feature | 이유 |
|---|---|
| `red_light_delay_seconds` | 신호 phase, 차량 arrival trajectory 없음 |
| `green_time_seconds` | 현시 시간 자료 없음 |
| `cycle_length_seconds` | 신호 주기 자료 없음 |
| `signal_offset_seconds` | 연동 offset 자료 없음 |
| `real_time_signal_state` | 실시간 신호 상태 자료 없음 |
| `queue_discharge_rate` | 차로별 queue/flow 자료 없음 |

허용되는 것은 아래와 같은 정적 인프라 feature뿐이다.

| 허용 feature | 설명 |
|---|---|
| `signal_count_250m` | 정류장 주변 250m 내 신호 수 |
| `nearest_signal_distance_m` | 가장 가까운 신호까지 거리 |
| `pedestrian_signal_count_250m` | 주변 보행 신호 수 |
| `blink_signal_ratio_250m` | 주변 점멸 신호 비율 |
| `controlled_signal_ratio_250m` | 주변 제어 신호 비율 |
| `edge_signal_count` | edge 양끝 또는 edge 주변 신호 수 proxy |
| `edge_signal_density_per_km` | edge 길이 대비 신호 밀도 |

---

## 2. 입력 계약

### 2.1 필수 입력

| 인자 | 설명 |
|---|---|
| `--signal-csv` | 대구 신호등 CSV 경로 |
| `--output-dir` | 산출물 저장 디렉터리 |

### 2.2 node/edge source 입력 방식

둘 중 하나를 사용한다.

#### 방식 A: Parquet 입력

| 인자 | 설명 |
|---|---|
| `--node-source` | `node_uid`, `node_index`, 좌표 컬럼이 있는 parquet |
| `--edge-source` | `src_idx`, `dst_idx`, `distance_m`이 있는 parquet |

#### 방식 B: DB 입력

| 인자 | 설명 |
|---|---|
| `--db-url` | PostgreSQL DSN(Data Source Name=데이터 원본 이름) |
| `--node-source-db` | 기본값 `public.gatv2_node_master_active` |
| `--edge-source-db` | 기본값 `public.gatv2_edge_primary_active` |

DB 방식은 `gatv2_node_master_active.node_uid = 'STOP:' || dim_stop.stop_id` 조인을 시도하고, `dim_stop.geom_5187`에서 node 좌표를 읽는 것을 기본 가정으로 둔다.

---

## 3. 출력 계약

### 3.1 node_signal_features.parquet

필수 컬럼:

```text
node_uid
node_index
signal_count_100m
signal_count_250m
signal_count_500m
nearest_signal_distance_m
pedestrian_signal_count_250m
blink_signal_ratio_250m
controlled_signal_ratio_250m
signal_delay_risk_proxy
intersection_complexity_proxy
signal_feature_quality_flag
```

### 3.2 edge_signal_features.parquet

필수 컬럼:

```text
src_idx
dst_idx
distance_m
edge_signal_count
edge_signal_density_per_km
edge_nearest_signal_distance_m
edge_control_complexity_proxy
edge_signal_feature_quality_flag
```

### 3.3 tensor_signal_feature_contract_v2.json

필수 내용:

```text
source_csv_path
source_csv_hash
coordinate_mode
join_method
feature_definitions
unavailable_dynamic_signal_fields
claim_guardrails
```

### 3.4 signal_feature_quality_report.json

필수 내용:

```text
signal_csv_row_count
node_row_count
edge_row_count
node_feature_row_count
edge_feature_row_count
coordinate_mode
step_98_ready
warnings
```

---

## 4. proxy feature 원칙

`signal_delay_risk_proxy`, `intersection_complexity_proxy`, `edge_control_complexity_proxy`는 실제 측정값이 아니다.

이 feature들은 다음 의미로만 사용한다.

- 신호 인프라가 많은 지역일수록 운행 복잡도가 높을 수 있다는 약한 사전정보
- policy observation context
- simulator 환경 난이도 proxy

논문 또는 보고서에서 이것을 실제 신호 지연 시간으로 주장하면 안 된다.

---

## 5. Step 98 성공 판정

Step 98 self-test는 다음을 확인한다.

1. builder script가 문법 오류 없이 compile되는가
2. synthetic node/edge/signal CSV에서 4개 산출물이 생성되는가
3. node output에 필수 컬럼이 있는가
4. edge output에 필수 컬럼이 있는가
5. 금지 feature가 output에 섞이지 않았는가
6. claim guardrail이 `performance_claim_allowed=false`로 유지되는가

---
'@

$ContractPath = Join-Path $AdaptersDir "signal_feature_builder_contract_v2.md"
$ContractMd | Set-Content -Path $ContractPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 98 file 2:
# 05_training/adapters/build_signal_features_for_causal_simulator_v2.py
# ---------------------------------------------------------------------
$BuilderPy = @'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

LAT_CANDIDATES = ["lat", "latitude", "위도", "y_wgs84", "wgs84_y", "위도값"]
LON_CANDIDATES = ["lon", "lng", "longitude", "경도", "x_wgs84", "wgs84_x", "경도값"]
X_CANDIDATES = ["x", "x_coord", "coord_x", "좌표x", "x좌표", "tm_x", "epsg5187_x", "x_5187"]
Y_CANDIDATES = ["y", "y_coord", "coord_y", "좌표y", "y좌표", "tm_y", "epsg5187_y", "y_5187"]
ID_CANDIDATES = ["signal_id", "id", "신호등id", "신호등_id", "시설물관리번호", "관리번호"]
TYPE_CANDIDATES = ["signal_type", "type", "신호등종류", "종류", "시설구분", "신호기종류"]
NAME_CANDIDATES = ["signal_name", "name", "신호등명", "시설명", "교차로명", "설치위치"]

NODE_X_CANDIDATES = ["x", "x_coord", "coord_x", "node_x", "x_5187", "epsg5187_x", "lon", "lng", "longitude", "경도"]
NODE_Y_CANDIDATES = ["y", "y_coord", "coord_y", "node_y", "y_5187", "epsg5187_y", "lat", "latitude", "위도"]

FORBIDDEN_DYNAMIC_SIGNAL_FIELDS = [
    "red_light_delay_seconds",
    "green_time_seconds",
    "cycle_length_seconds",
    "phase_sequence",
    "signal_offset_seconds",
    "real_time_signal_state",
    "queue_discharge_rate",
]


def normalize_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "_")


def find_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {normalize_col(c): c for c in cols}
    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_any_encoding(path: Path) -> Tuple[pd.DataFrame, str]:
    last_error: Optional[Exception] = None
    for enc in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"failed to read CSV with known encodings: {path}, last_error={last_error}")


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def approx_wgs84_to_meter_xy(lon: pd.Series, lat: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Lightweight local projection for neighborhood distance.
    This is not a legal survey transform. It is sufficient for Step 98 feature
    smoke/contract building when precise EPSG transform libraries are unavailable.
    """
    lon_num = to_numeric_series(lon)
    lat_num = to_numeric_series(lat)
    lat0 = float(lat_num.dropna().mean()) if lat_num.notna().any() else 35.87
    lon0 = float(lon_num.dropna().mean()) if lon_num.notna().any() else 128.60

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    x = (lon_num - lon0) * meters_per_deg_lon
    y = (lat_num - lat0) * meters_per_deg_lat
    return x, y


def infer_signal_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "id_col": find_first(cols, ID_CANDIDATES),
        "type_col": find_first(cols, TYPE_CANDIDATES),
        "name_col": find_first(cols, NAME_CANDIDATES),
        "lat_col": find_first(cols, LAT_CANDIDATES),
        "lon_col": find_first(cols, LON_CANDIDATES),
        "x_col": find_first(cols, X_CANDIDATES),
        "y_col": find_first(cols, Y_CANDIDATES),
    }


def add_signal_flags(df: pd.DataFrame, inferred: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    type_col = inferred.get("type_col")
    name_col = inferred.get("name_col")

    text = pd.Series([""] * len(out), index=out.index, dtype="object")
    if type_col and type_col in out.columns:
        text = text + " " + out[type_col].astype(str)
    if name_col and name_col in out.columns:
        text = text + " " + out[name_col].astype(str)

    lowered = text.str.lower()

    out["_pedestrian_signal_flag"] = (
        lowered.str.contains("보행", regex=False)
        | lowered.str.contains("pedestrian", regex=False)
        | lowered.str.contains("횡단", regex=False)
    ).astype(int)

    out["_blink_signal_flag"] = (
        lowered.str.contains("점멸", regex=False)
        | lowered.str.contains("blink", regex=False)
        | lowered.str.contains("flashing", regex=False)
    ).astype(int)

    out["_controlled_signal_flag"] = (
        lowered.str.contains("제어", regex=False)
        | lowered.str.contains("control", regex=False)
        | lowered.str.contains("controlled", regex=False)
        | lowered.str.contains("신호", regex=False)
    ).astype(int)

    return out


def prepare_signal_xy(signal_csv: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw, encoding = read_csv_any_encoding(signal_csv)
    inferred = infer_signal_columns(raw)

    coordinate_mode = "unknown"
    sig = raw.copy()

    if inferred.get("lat_col") and inferred.get("lon_col"):
        coordinate_mode = "wgs84_lat_lon"
        x, y = approx_wgs84_to_meter_xy(sig[inferred["lon_col"]], sig[inferred["lat_col"]])
        sig["_x_m"] = x
        sig["_y_m"] = y
    elif inferred.get("x_col") and inferred.get("y_col"):
        coordinate_mode = "projected_x_y"
        sig["_x_m"] = to_numeric_series(sig[inferred["x_col"]])
        sig["_y_m"] = to_numeric_series(sig[inferred["y_col"]])
    else:
        raise RuntimeError(
            "Signal CSV coordinate columns were not inferred. "
            "Expected lat/lon or x/y style columns."
        )

    sig = add_signal_flags(sig, inferred)
    before = len(sig)
    sig = sig.dropna(subset=["_x_m", "_y_m"]).copy()

    if sig.empty:
        raise RuntimeError("No signal rows remain after coordinate parsing")

    report = {
        "signal_csv_encoding": encoding,
        "signal_csv_row_count": int(before),
        "signal_rows_with_valid_coordinates": int(len(sig)),
        "signal_columns": list(raw.columns),
        "inferred_columns": inferred,
        "coordinate_mode": coordinate_mode,
    }
    return sig, report


def read_parquet_source(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"parquet source not found: {p}")
    return pd.read_parquet(p)


def load_nodes_from_db(db_url: str, node_source_db: str) -> pd.DataFrame:
    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError(f"psycopg2 is required for DB mode: {exc}") from exc

    sql = f"""
    SELECT
        n.node_uid,
        n.node_index,
        ST_X(d.geom_5187)::double precision AS x_5187,
        ST_Y(d.geom_5187)::double precision AS y_5187
    FROM {node_source_db} n
    JOIN public.dim_stop d
      ON n.node_uid = 'STOP:' || d.stop_id::text
    WHERE d.geom_5187 IS NOT NULL
    ORDER BY n.node_index
    """

    conn = psycopg2.connect(db_url)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def load_edges_from_db(db_url: str, edge_source_db: str) -> pd.DataFrame:
    try:
        import psycopg2
    except Exception as exc:
        raise RuntimeError(f"psycopg2 is required for DB mode: {exc}") from exc

    sql = f"""
    SELECT
        src_idx,
        dst_idx,
        distance_m
    FROM {edge_source_db}
    ORDER BY src_idx, dst_idx
    """

    conn = psycopg2.connect(db_url)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def prepare_node_xy(nodes: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required = ["node_uid", "node_index"]
    missing = [c for c in required if c not in nodes.columns]
    if missing:
        raise RuntimeError(f"node source missing required columns: {missing}")

    cols = list(nodes.columns)
    x_col = find_first(cols, NODE_X_CANDIDATES)
    y_col = find_first(cols, NODE_Y_CANDIDATES)

    if not x_col or not y_col:
        raise RuntimeError(
            "node source needs coordinate columns. "
            "Expected x/y, x_5187/y_5187, lon/lat, or longitude/latitude."
        )

    out = nodes.copy()
    # If the detected columns look like lon/lat, approximate-project them.
    nx = normalize_col(x_col)
    ny = normalize_col(y_col)
    if nx in {"lon", "lng", "longitude", "경도"} or ny in {"lat", "latitude", "위도"}:
        x, y = approx_wgs84_to_meter_xy(out[x_col], out[y_col])
        coordinate_mode = "node_wgs84_lat_lon"
        out["_x_m"] = x
        out["_y_m"] = y
    else:
        coordinate_mode = "node_projected_x_y"
        out["_x_m"] = to_numeric_series(out[x_col])
        out["_y_m"] = to_numeric_series(out[y_col])

    before = len(out)
    out = out.dropna(subset=["_x_m", "_y_m"]).copy()
    if out.empty:
        raise RuntimeError("No node rows remain after coordinate parsing")

    return out, {
        "node_row_count": int(before),
        "node_rows_with_valid_coordinates": int(len(out)),
        "node_coordinate_columns": {"x": x_col, "y": y_col},
        "node_coordinate_mode": coordinate_mode,
    }


def validate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    required = ["src_idx", "dst_idx"]
    missing = [c for c in required if c not in edges.columns]
    if missing:
        raise RuntimeError(f"edge source missing required columns: {missing}")

    out = edges.copy()
    if "distance_m" not in out.columns:
        out["distance_m"] = np.nan
    out["src_idx"] = pd.to_numeric(out["src_idx"], errors="raise").astype(int)
    out["dst_idx"] = pd.to_numeric(out["dst_idx"], errors="raise").astype(int)
    out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce")
    return out


def distance_matrix_to_signals(nodes_xy: np.ndarray, sig_xy: np.ndarray) -> np.ndarray:
    # Step 98 is designed for modest CSV sizes and smoke validation.
    # If signal count becomes very large, Step 99 can replace this with KD-tree.
    diff = nodes_xy[:, None, :] - sig_xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def build_node_signal_features(nodes: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    node_xy = nodes[["_x_m", "_y_m"]].to_numpy(dtype=float)
    sig_xy = signals[["_x_m", "_y_m"]].to_numpy(dtype=float)

    dist = distance_matrix_to_signals(node_xy, sig_xy)

    within_100 = dist <= 100.0
    within_250 = dist <= 250.0
    within_500 = dist <= 500.0

    nearest = dist.min(axis=1)

    ped = signals["_pedestrian_signal_flag"].to_numpy(dtype=float)
    blink = signals["_blink_signal_flag"].to_numpy(dtype=float)
    controlled = signals["_controlled_signal_flag"].to_numpy(dtype=float)

    count_250 = within_250.sum(axis=1).astype(float)

    ped_count_250 = (within_250 * ped[None, :]).sum(axis=1)
    blink_count_250 = (within_250 * blink[None, :]).sum(axis=1)
    controlled_count_250 = (within_250 * controlled[None, :]).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        blink_ratio_250 = np.where(count_250 > 0, blink_count_250 / count_250, 0.0)
        controlled_ratio_250 = np.where(count_250 > 0, controlled_count_250 / count_250, 0.0)

    out = pd.DataFrame({
        "node_uid": nodes["node_uid"].astype(str).values,
        "node_index": pd.to_numeric(nodes["node_index"], errors="raise").astype(int).values,
        "signal_count_100m": within_100.sum(axis=1).astype(int),
        "signal_count_250m": count_250.astype(int),
        "signal_count_500m": within_500.sum(axis=1).astype(int),
        "nearest_signal_distance_m": nearest.astype(float),
        "pedestrian_signal_count_250m": ped_count_250.astype(int),
        "blink_signal_ratio_250m": blink_ratio_250.astype(float),
        "controlled_signal_ratio_250m": controlled_ratio_250.astype(float),
    })

    out["signal_delay_risk_proxy"] = (
        np.log1p(out["signal_count_250m"].astype(float))
        + np.minimum(out["nearest_signal_distance_m"].astype(float), 500.0).rsub(500.0) / 500.0
    ).astype(float)

    out["intersection_complexity_proxy"] = (
        np.log1p(out["signal_count_500m"].astype(float))
        + out["controlled_signal_ratio_250m"].astype(float)
    ).astype(float)

    out["signal_feature_quality_flag"] = np.where(
        np.isfinite(out["nearest_signal_distance_m"].astype(float)),
        "ok",
        "missing_distance",
    )

    return out.sort_values("node_index").reset_index(drop=True)


def build_edge_signal_features(edges: pd.DataFrame, node_features: pd.DataFrame) -> pd.DataFrame:
    nf = node_features.set_index("node_index", drop=False)

    rows: List[Dict[str, Any]] = []
    for r in edges.itertuples(index=False):
        src = int(getattr(r, "src_idx"))
        dst = int(getattr(r, "dst_idx"))
        dist_m = float(getattr(r, "distance_m")) if pd.notna(getattr(r, "distance_m")) else np.nan

        if src not in nf.index or dst not in nf.index:
            rows.append({
                "src_idx": src,
                "dst_idx": dst,
                "distance_m": dist_m,
                "edge_signal_count": np.nan,
                "edge_signal_density_per_km": np.nan,
                "edge_nearest_signal_distance_m": np.nan,
                "edge_control_complexity_proxy": np.nan,
                "edge_signal_feature_quality_flag": "missing_node_signal_feature",
            })
            continue

        src_row = nf.loc[src]
        dst_row = nf.loc[dst]

        edge_signal_count = float(src_row["signal_count_250m"] + dst_row["signal_count_250m"])
        nearest = float(min(src_row["nearest_signal_distance_m"], dst_row["nearest_signal_distance_m"]))

        if np.isfinite(dist_m) and dist_m > 0:
            density = edge_signal_count / (dist_m / 1000.0)
        else:
            density = np.nan

        complexity = float(
            0.5 * src_row["intersection_complexity_proxy"]
            + 0.5 * dst_row["intersection_complexity_proxy"]
        )

        rows.append({
            "src_idx": src,
            "dst_idx": dst,
            "distance_m": dist_m,
            "edge_signal_count": edge_signal_count,
            "edge_signal_density_per_km": density,
            "edge_nearest_signal_distance_m": nearest,
            "edge_control_complexity_proxy": complexity,
            "edge_signal_feature_quality_flag": "ok",
        })

    return pd.DataFrame(rows)


def build_feature_definitions() -> Dict[str, str]:
    return {
        "signal_count_100m": "Number of static signal records within 100m of node.",
        "signal_count_250m": "Number of static signal records within 250m of node.",
        "signal_count_500m": "Number of static signal records within 500m of node.",
        "nearest_signal_distance_m": "Distance to nearest static signal record in meters.",
        "pedestrian_signal_count_250m": "Number of likely pedestrian signal records within 250m.",
        "blink_signal_ratio_250m": "Ratio of likely blink/flashing signal records within 250m.",
        "controlled_signal_ratio_250m": "Ratio of likely controlled signal records within 250m.",
        "signal_delay_risk_proxy": "Proxy only. Static signal density and nearness risk indicator; not measured red-light delay.",
        "intersection_complexity_proxy": "Proxy only. Static signal infrastructure complexity indicator.",
        "edge_signal_count": "Proxy edge signal count from source/destination node neighborhoods.",
        "edge_signal_density_per_km": "Proxy edge signal density using edge length.",
        "edge_nearest_signal_distance_m": "Minimum nearest signal distance among edge endpoints.",
        "edge_control_complexity_proxy": "Proxy only. Average endpoint static control complexity."
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_no_forbidden_columns(df: pd.DataFrame, name: str) -> None:
    forbidden = [c for c in FORBIDDEN_DYNAMIC_SIGNAL_FIELDS if c in df.columns]
    if forbidden:
        raise RuntimeError(f"{name} contains forbidden dynamic signal fields: {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-source", default="", help="Parquet path with node_uid,node_index and coordinates")
    parser.add_argument("--edge-source", default="", help="Parquet path with src_idx,dst_idx,distance_m")
    parser.add_argument("--db-url", default="", help="PostgreSQL DSN for DB mode")
    parser.add_argument("--node-source-db", default="public.gatv2_node_master_active")
    parser.add_argument("--edge-source-db", default="public.gatv2_edge_primary_active")
    args = parser.parse_args()

    signal_csv = Path(args.signal_csv)
    if not signal_csv.exists():
        raise SystemExit(f"signal CSV not found: {signal_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signals, signal_report = prepare_signal_xy(signal_csv)

    if args.node_source:
        nodes_raw = read_parquet_source(args.node_source)
        node_source_mode = "parquet"
    elif args.db_url:
        nodes_raw = load_nodes_from_db(args.db_url, args.node_source_db)
        node_source_mode = "db"
    else:
        raise SystemExit("Either --node-source or --db-url is required")

    if args.edge_source:
        edges_raw = read_parquet_source(args.edge_source)
        edge_source_mode = "parquet"
    elif args.db_url:
        edges_raw = load_edges_from_db(args.db_url, args.edge_source_db)
        edge_source_mode = "db"
    else:
        raise SystemExit("Either --edge-source or --db-url is required")

    nodes, node_report = prepare_node_xy(nodes_raw)
    edges = validate_edges(edges_raw)

    node_features = build_node_signal_features(nodes, signals)
    edge_features = build_edge_signal_features(edges, node_features)

    assert_no_forbidden_columns(node_features, "node_signal_features")
    assert_no_forbidden_columns(edge_features, "edge_signal_features")

    node_path = output_dir / "node_signal_features.parquet"
    edge_path = output_dir / "edge_signal_features.parquet"
    contract_path = output_dir / "tensor_signal_feature_contract_v2.json"
    quality_path = output_dir / "signal_feature_quality_report.json"

    node_features.to_parquet(node_path, index=False)
    edge_features.to_parquet(edge_path, index=False)

    warnings: List[str] = []
    if signal_report["signal_rows_with_valid_coordinates"] < signal_report["signal_csv_row_count"]:
        warnings.append("Some signal rows were dropped because coordinates were missing or non-numeric.")

    missing_node_signal_features = int(
        (edge_features["edge_signal_feature_quality_flag"] != "ok").sum()
    )
    if missing_node_signal_features:
        warnings.append(f"{missing_node_signal_features} edges could not find endpoint node signal features.")

    contract = {
        "artifact_version": "tensor_signal_feature_contract_v2",
        "step": 98,
        "source_csv_path": str(signal_csv),
        "source_csv_hash": sha256_file(signal_csv),
        "node_source_mode": node_source_mode,
        "edge_source_mode": edge_source_mode,
        "coordinate_mode": {
            "signal": signal_report["coordinate_mode"],
            "node": node_report["node_coordinate_mode"],
        },
        "join_method": "endpoint_neighborhood_static_signal_features_v1",
        "feature_definitions": build_feature_definitions(),
        "unavailable_dynamic_signal_fields": FORBIDDEN_DYNAMIC_SIGNAL_FIELDS,
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
        },
        "output_files": {
            "node_signal_features": str(node_path),
            "edge_signal_features": str(edge_path),
            "signal_feature_quality_report": str(quality_path),
            "tensor_signal_feature_contract": str(contract_path),
        }
    }

    quality = {
        "artifact_version": "signal_feature_quality_report_v1",
        "step": 98,
        "signal_csv_row_count": signal_report["signal_csv_row_count"],
        "signal_rows_with_valid_coordinates": signal_report["signal_rows_with_valid_coordinates"],
        "node_row_count": node_report["node_row_count"],
        "node_rows_with_valid_coordinates": node_report["node_rows_with_valid_coordinates"],
        "edge_row_count": int(len(edges)),
        "node_feature_row_count": int(len(node_features)),
        "edge_feature_row_count": int(len(edge_features)),
        "coordinate_mode": contract["coordinate_mode"],
        "step_98_ready": True,
        "warnings": warnings,
        "forbidden_dynamic_signal_fields_absent": True,
    }

    write_json(contract_path, contract)
    write_json(quality_path, quality)

    print("[OK] Step 98 signal feature builder complete")
    print(f"[OK] node_signal_features : {node_path}")
    print(f"[OK] edge_signal_features : {edge_path}")
    print(f"[OK] contract             : {contract_path}")
    print(f"[OK] quality_report       : {quality_path}")
    print(f"[OK] node_rows            : {len(node_features)}")
    print(f"[OK] edge_rows            : {len(edge_features)}")
    if warnings:
        print("[WARN] quality warnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
'@

$BuilderPath = Join-Path $AdaptersDir "build_signal_features_for_causal_simulator_v2.py"
$BuilderPy | Set-Content -Path $BuilderPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 98 file 3:
# 05_training/adapters/test_build_signal_features_for_causal_simulator_v2.py
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = PROJECT_ROOT / "05_training" / "adapters"
BUILDER = ADAPTERS_DIR / "build_signal_features_for_causal_simulator_v2.py"
CONTRACT_MD = ADAPTERS_DIR / "signal_feature_builder_contract_v2.md"

FORBIDDEN = {
    "red_light_delay_seconds",
    "green_time_seconds",
    "cycle_length_seconds",
    "phase_sequence",
    "signal_offset_seconds",
    "real_time_signal_state",
    "queue_discharge_rate",
}


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_fixture(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)

    signal_csv = root / "toy_signal.csv"
    signal_df = pd.DataFrame([
        {"signal_id": "S1", "signal_type": "차량신호", "latitude": 35.8700, "longitude": 128.6000},
        {"signal_id": "S2", "signal_type": "보행신호", "latitude": 35.8705, "longitude": 128.6005},
        {"signal_id": "S3", "signal_type": "점멸신호", "latitude": 35.8750, "longitude": 128.6050},
    ])
    signal_df.to_csv(signal_csv, index=False, encoding="utf-8-sig")

    node_path = root / "toy_nodes.parquet"
    node_df = pd.DataFrame([
        {"node_uid": "STOP:1", "node_index": 0, "latitude": 35.8700, "longitude": 128.6000},
        {"node_uid": "STOP:2", "node_index": 1, "latitude": 35.8706, "longitude": 128.6006},
        {"node_uid": "STOP:3", "node_index": 2, "latitude": 35.8800, "longitude": 128.6100},
    ])
    node_df.to_parquet(node_path, index=False)

    edge_path = root / "toy_edges.parquet"
    edge_df = pd.DataFrame([
        {"src_idx": 0, "dst_idx": 1, "distance_m": 120.0},
        {"src_idx": 1, "dst_idx": 2, "distance_m": 1400.0},
    ])
    edge_df.to_parquet(edge_path, index=False)

    return {
        "signal_csv": signal_csv,
        "node_path": node_path,
        "edge_path": edge_path,
        "output_dir": root / "out",
    }


def test_files_exist() -> None:
    assert_true(BUILDER.exists(), f"missing: {BUILDER}")
    assert_true(CONTRACT_MD.exists(), f"missing: {CONTRACT_MD}")


def test_contract_guardrails() -> None:
    text = CONTRACT_MD.read_text(encoding="utf-8-sig")
    for phrase in [
        "red_light_delay_seconds",
        "green_time_seconds",
        "cycle_length_seconds",
        "정적 신호 인프라 feature",
        "performance_claim_allowed=false",
    ]:
        assert_true(phrase in text, f"missing contract phrase: {phrase}")


def test_builder_smoke() -> None:
    fixture_root = PROJECT_ROOT / "artifacts" / "step98_selftest"
    fx = make_fixture(fixture_root)

    cmd = [
        sys.executable,
        str(BUILDER),
        "--signal-csv",
        str(fx["signal_csv"]),
        "--node-source",
        str(fx["node_path"]),
        "--edge-source",
        str(fx["edge_path"]),
        "--output-dir",
        str(fx["output_dir"]),
    ]

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("Step 98 builder smoke failed")

    node_out = fx["output_dir"] / "node_signal_features.parquet"
    edge_out = fx["output_dir"] / "edge_signal_features.parquet"
    contract_out = fx["output_dir"] / "tensor_signal_feature_contract_v2.json"
    quality_out = fx["output_dir"] / "signal_feature_quality_report.json"

    for p in [node_out, edge_out, contract_out, quality_out]:
        assert_true(p.exists(), f"missing output: {p}")

    node_df = pd.read_parquet(node_out)
    edge_df = pd.read_parquet(edge_out)

    required_node_cols = [
        "node_uid",
        "node_index",
        "signal_count_250m",
        "nearest_signal_distance_m",
        "pedestrian_signal_count_250m",
        "blink_signal_ratio_250m",
        "controlled_signal_ratio_250m",
        "signal_delay_risk_proxy",
        "intersection_complexity_proxy",
        "signal_feature_quality_flag",
    ]
    for c in required_node_cols:
        assert_true(c in node_df.columns, f"node output missing column: {c}")

    required_edge_cols = [
        "src_idx",
        "dst_idx",
        "distance_m",
        "edge_signal_count",
        "edge_signal_density_per_km",
        "edge_nearest_signal_distance_m",
        "edge_control_complexity_proxy",
        "edge_signal_feature_quality_flag",
    ]
    for c in required_edge_cols:
        assert_true(c in edge_df.columns, f"edge output missing column: {c}")

    for forbidden in FORBIDDEN:
        assert_true(forbidden not in node_df.columns, f"forbidden node column present: {forbidden}")
        assert_true(forbidden not in edge_df.columns, f"forbidden edge column present: {forbidden}")

    contract = json.loads(contract_out.read_text(encoding="utf-8"))
    assert_true(contract["artifact_version"] == "tensor_signal_feature_contract_v2", "bad contract version")
    assert_true(contract["claim_guardrails"]["performance_claim_allowed"] is False, "performance claim guardrail broken")
    assert_true(contract["claim_guardrails"]["red_light_delay_claim_allowed"] is False, "red-light delay claim guardrail broken")

    quality = json.loads(quality_out.read_text(encoding="utf-8"))
    assert_true(quality["step_98_ready"] is True, "step_98_ready must be true")
    assert_true(quality["forbidden_dynamic_signal_fields_absent"] is True, "forbidden fields flag must be true")


def main() -> None:
    test_files_exist()
    test_contract_guardrails()
    test_builder_smoke()
    print("[OK] Step 98 signal feature builder self-test PASS")


if __name__ == "__main__":
    main()
'@

$TestPath = Join-Path $AdaptersDir "test_build_signal_features_for_causal_simulator_v2.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

Write-Host "[OK] Step 98 files created:"
Write-Host " - $ContractPath"
Write-Host " - $BuilderPath"
Write-Host " - $TestPath"

if ($RunSelfTest) {
    if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
        $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
    } else {
        $py = "python"
    }

    Write-Host "[INFO] python = $py"

    & $py -m py_compile `
      ".\05_training\adapters\build_signal_features_for_causal_simulator_v2.py" `
      ".\05_training\adapters\test_build_signal_features_for_causal_simulator_v2.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 98 py_compile failed"
    }

    & $py ".\05_training\adapters\test_build_signal_features_for_causal_simulator_v2.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 98 self-test failed"
    }

    Write-Host "[DONE] Step 98 signal feature builder complete."
}
