from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

LAT_CANDIDATES = [
    "lat", "latitude", "위도", "y_wgs84", "wgs84_y", "위도값"
]
LON_CANDIDATES = [
    "lon", "lng", "longitude", "경도", "x_wgs84", "wgs84_x", "경도값"
]
X_CANDIDATES = [
    "x", "x_coord", "coord_x", "좌표x", "x좌표", "tm_x", "epsg5187_x", "x_5187"
]
Y_CANDIDATES = [
    "y", "y_coord", "coord_y", "좌표y", "y좌표", "tm_y", "epsg5187_y", "y_5187"
]
ID_CANDIDATES = [
    "signal_id", "id", "신호등id", "신호등_id", "시설물관리번호", "관리번호"
]
TYPE_CANDIDATES = [
    "signal_type", "type", "신호등종류", "종류", "시설구분", "신호기종류"
]
NAME_CANDIDATES = [
    "signal_name", "name", "신호등명", "시설명", "교차로명", "설치위치"
]


TENSOR_RELATIONS = [
    ("public", "gatv2_node_master_active"),
    ("public", "gatv2_edge_primary_active"),
    ("public", "gatv2_edge_primary_active_mat"),
    ("public", "gatv2_snapshot_stop_features_train"),
    ("public", "gatv2_snapshot_stop_features_train_mat"),
    ("public", "gatv2_snapshot_summary"),
    ("public", "gatv2_edge_summary"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "_")


def find_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {normalize_col(c): c for c in cols}
    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized:
            return normalized[key]
    return None


def read_csv_header_and_sample(path: Path, sample_rows: int = 20) -> Tuple[str, List[str], List[Dict[str, str]], int]:
    last_error: Optional[Exception] = None

    for enc in CSV_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample_text = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample_text)
                except Exception:
                    dialect = csv.excel

                reader = csv.DictReader(f, dialect=dialect)
                cols = list(reader.fieldnames or [])
                rows: List[Dict[str, str]] = []
                total = 0
                for row in reader:
                    total += 1
                    if len(rows) < sample_rows:
                        rows.append(dict(row))
                return enc, cols, rows, total
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"failed to read CSV with known encodings: {path}, last_error={last_error}")


def inspect_signal_csv(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "status": "not_provided",
            "note": "No signal CSV path was supplied. Document/contract self-test can still pass."
        }

    if not path.exists():
        return {
            "provided": True,
            "path": str(path),
            "status": "missing",
            "error": "signal CSV file does not exist"
        }

    enc, cols, rows, row_count = read_csv_header_and_sample(path)
    lat_col = find_first(cols, LAT_CANDIDATES)
    lon_col = find_first(cols, LON_CANDIDATES)
    x_col = find_first(cols, X_CANDIDATES)
    y_col = find_first(cols, Y_CANDIDATES)
    id_col = find_first(cols, ID_CANDIDATES)
    type_col = find_first(cols, TYPE_CANDIDATES)
    name_col = find_first(cols, NAME_CANDIDATES)

    coordinate_mode = "unknown"
    if lat_col and lon_col:
        coordinate_mode = "wgs84_lat_lon"
    elif x_col and y_col:
        coordinate_mode = "projected_x_y"

    inferred = {
        "id_col": id_col,
        "type_col": type_col,
        "name_col": name_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "x_col": x_col,
        "y_col": y_col,
        "coordinate_mode": coordinate_mode
    }

    direct_features = []
    if id_col:
        direct_features.append("signal_id")
    if type_col:
        direct_features.append("signal_type")
    if name_col:
        direct_features.append("signal_name")
    if lat_col and lon_col:
        direct_features.extend(["latitude", "longitude"])
    if x_col and y_col:
        direct_features.extend(["x_coord", "y_coord"])

    possible_flag_features = []
    joined_cols = " ".join(cols)
    if any(k in joined_cols for k in ["보행", "pedestrian"]):
        possible_flag_features.append("pedestrian_signal_flag_candidate")
    if any(k in joined_cols for k in ["점멸", "blink"]):
        possible_flag_features.append("blink_signal_flag_candidate")
    if any(k in joined_cols for k in ["제어", "control", "controlled"]):
        possible_flag_features.append("controlled_signal_flag_candidate")

    return {
        "provided": True,
        "path": str(path),
        "status": "ok",
        "sha256": sha256_file(path),
        "encoding": enc,
        "row_count": int(row_count),
        "column_count": int(len(cols)),
        "columns": cols,
        "inferred_columns": inferred,
        "direct_signal_features_available": direct_features,
        "possible_flag_features": possible_flag_features,
        "sample_rows": rows[:3],
        "admission": {
            "static_signal_infrastructure_allowed": coordinate_mode != "unknown",
            "dynamic_signal_phase_allowed": False,
            "red_light_delay_allowed": False,
            "green_time_allowed": False
        },
        "warnings": [] if coordinate_mode != "unknown" else [
            "No coordinate columns were confidently inferred. Step 98 spatial join requires manual column mapping."
        ]
    }


def build_db_url_from_env() -> Optional[str]:
    if os.environ.get("URBANBUS_DB_DSN"):
        return os.environ["URBANBUS_DB_DSN"]

    host = os.environ.get("PGHOST")
    db = os.environ.get("PGDATABASE")
    user = os.environ.get("PGUSER")
    pw = os.environ.get("PGPASSWORD")
    port = os.environ.get("PGPORT", "5432")
    if host and db and user:
        if pw:
            return f"postgresql://{user}:{pw}@{host}:{port}/{db}"
        return f"postgresql://{user}@{host}:{port}/{db}"
    return None


def inspect_tensor_db(db_url: Optional[str]) -> Dict[str, Any]:
    if not db_url:
        return {
            "provided": False,
            "status": "not_provided",
            "note": "No DB URL supplied. Set --db-url or URBANBUS_DB_DSN to inspect tensor DB fields."
        }

    try:
        import psycopg2
        import psycopg2.extras
    except Exception as exc:
        return {
            "provided": True,
            "status": "dependency_missing",
            "error": f"psycopg2 not importable: {exc}"
        }

    out: Dict[str, Any] = {
        "provided": True,
        "status": "ok",
        "relations": {}
    }

    try:
        conn = psycopg2.connect(db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                for schema, rel in TENSOR_RELATIONS:
                    cur.execute(
                        """
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = %s
                          AND table_name = %s
                        LIMIT 1
                        """,
                        (schema, rel),
                    )
                    exists = cur.fetchone() is not None

                    payload: Dict[str, Any] = {"exists": bool(exists)}
                    if exists:
                        cur.execute(
                            """
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_schema = %s
                              AND table_name = %s
                            ORDER BY ordinal_position
                            """,
                            (schema, rel),
                        )
                        cols = cur.fetchall()
                        payload["columns"] = [
                            {"name": r["column_name"], "data_type": r["data_type"]}
                            for r in cols
                        ]

                        try:
                            cur.execute(f'SELECT COUNT(*) AS n FROM "{schema}"."{rel}"')
                            payload["row_count"] = int(cur.fetchone()["n"])
                        except Exception as exc:
                            payload["row_count_error"] = str(exc)

                    out["relations"][f"{schema}.{rel}"] = payload
        finally:
            conn.close()
    except Exception as exc:
        return {
            "provided": True,
            "status": "connection_failed",
            "error": str(exc)
        }

    return out


def classify_availability(signal_report: Dict[str, Any], tensor_report: Dict[str, Any]) -> Dict[str, Any]:
    tensor_direct = [
        "node_uid",
        "node_index",
        "src_idx",
        "dst_idx",
        "distance_m",
        "time_sec",
        "generalized_cost",
        "long_edge_5km_flag",
        "state_ts",
        "next_state_ts",
        "boardings_recent_log",
        "alightings_recent_log",
        "waiting_passenger_cnt_log",
        "hour_sin",
        "hour_cos",
        "is_peak",
        "next_boardings_recent",
        "next_alightings_recent",
        "next_waiting_passenger_cnt"
    ]

    signal_direct = signal_report.get("direct_signal_features_available", [])

    derived_possible = []
    coord_ok = (
        signal_report.get("status") == "ok"
        and signal_report.get("inferred_columns", {}).get("coordinate_mode") != "unknown"
    )
    if coord_ok:
        derived_possible = [
            "signal_count_100m",
            "signal_count_250m",
            "signal_count_500m",
            "signal_density_per_km",
            "nearest_signal_distance_m",
            "pedestrian_signal_count_250m",
            "blink_signal_ratio_250m",
            "controlled_signal_ratio_250m",
            "edge_signal_count",
            "edge_signal_density_per_km",
            "edge_nearest_signal_distance_m"
        ]

    proxy_only = [
        "signal_delay_risk_proxy",
        "intersection_complexity_proxy",
        "edge_control_complexity_proxy",
        "stop_access_friction_proxy"
    ]

    unavailable = [
        "red_light_delay_seconds",
        "green_time_seconds",
        "cycle_length_seconds",
        "phase_sequence",
        "signal_offset_seconds",
        "real_time_signal_state",
        "queue_discharge_rate",
        "lane_level_turning_movement",
        "actual_vehicle_trajectory",
        "incident_event_stream",
        "weather_event_stream"
    ]

    return {
        "tensor_db_direct": tensor_direct,
        "signal_csv_direct": signal_direct,
        "tensor_db_plus_signal_csv_derived": derived_possible,
        "proxy_only": proxy_only,
        "currently_unavailable": unavailable,
        "causal_simulator_v2_admission": {
            "allowed_now": [
                "tensor_db_direct",
                "signal_csv_direct_static_fields",
                "derived_static_signal_infrastructure_features",
                "explicit_proxy_features"
            ],
            "not_allowed_now": [
                "dynamic_signal_phase",
                "red_light_delay",
                "green_time",
                "cycle_length",
                "signal_offset",
                "real_time_signal_state"
            ]
        },
        "step_98_ready": bool(coord_ok),
        "step_98_blockers": [] if coord_ok else [
            "Signal CSV coordinate columns were not inferred. Provide explicit column mapping before spatial feature build."
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-csv", default="", help="Path to Daegu signal CSV")
    parser.add_argument("--db-url", default="", help="PostgreSQL DSN. If omitted, URBANBUS_DB_DSN or PG* env vars are used.")
    parser.add_argument("--output-json", default="artifacts/step97_tensor_signal_data_availability_audit_report.json")
    args = parser.parse_args()

    signal_path = Path(args.signal_csv) if args.signal_csv else None
    db_url = args.db_url or build_db_url_from_env()

    signal_report = inspect_signal_csv(signal_path)
    tensor_report = inspect_tensor_db(db_url)
    classification = classify_availability(signal_report, tensor_report)

    report = {
        "artifact_version": "tensor_signal_data_availability_audit_report_v1",
        "step": 97,
        "signal_csv_report": signal_report,
        "tensor_db_report": tensor_report,
        "availability_classification": classification,
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "smoke_only": True
        }
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Step 97 tensor + signal availability audit report written")
    print(f"[OK] output_json : {out_path}")
    print(f"[OK] signal_csv  : {signal_report.get('status')}")
    print(f"[OK] tensor_db   : {tensor_report.get('status')}")
    print(f"[OK] step98_ready: {classification.get('step_98_ready')}")
    if classification.get("step_98_blockers"):
        print("[WARN] Step 98 blockers:")
        for b in classification["step_98_blockers"]:
            print(f"  - {b}")


if __name__ == "__main__":
    main()
