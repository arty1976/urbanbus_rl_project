from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
REQUIRED_COORD_COLS = ["위도", "경도"]

STATIC_DIRECT_COLUMNS = [
    "시도명", "시군구명", "소재지도로명주소", "소재지지번주소",
    "위도", "경도", "신호등관리번호", "신호등구분", "신호등색종류",
    "신호등화방식", "신호제어방식", "신호시간결정방식",
    "점멸등운영여부", "보행자작동신호기유무", "잔여시간표시기유무",
    "시각장애인용음향신호기유무", "데이터기준일자",
]

QUARANTINE_DYNAMIC_LOOKING_COLUMNS = [
    "신호등화순서", "신호등화시간", "점멸등운영시작시각", "점멸등운영종료시각",
]

FORBIDDEN_DYNAMIC_SIGNAL_FIELDS = [
    "red_light_delay_seconds", "green_time_seconds", "cycle_length_seconds",
    "phase_sequence", "signal_offset_seconds", "real_time_signal_state",
    "queue_discharge_rate",
]


def read_csv_any_encoding(path: Path) -> Tuple[pd.DataFrame, str]:
    last_error: Optional[Exception] = None
    for enc in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df, enc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read CSV with known encodings: {path}, last_error={last_error}")


def normalize_yes_no(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.isin(["Y", "YES", "TRUE", "1", "O", "○", "있음", "유", "운영", "작동"])


def top_values(df: pd.DataFrame, col: str, n: int = 10) -> List[Dict[str, Any]]:
    if col not in df.columns:
        return []
    vc = df[col].value_counts(dropna=False).head(n)
    return [{"value": None if pd.isna(k) else str(k), "count": int(v)} for k, v in vc.items()]


def build_report(csv_path: Path) -> Dict[str, Any]:
    df, encoding = read_csv_any_encoding(csv_path)
    columns = list(df.columns)
    missing_coord = [c for c in REQUIRED_COORD_COLS if c not in columns]

    lat = pd.to_numeric(df["위도"], errors="coerce") if "위도" in df.columns else pd.Series([], dtype=float)
    lon = pd.to_numeric(df["경도"], errors="coerce") if "경도" in df.columns else pd.Series([], dtype=float)

    coord_valid = (
        lat.notna() & lon.notna() & lat.between(33.0, 39.5) & lon.between(124.0, 132.0)
    ) if not missing_coord else pd.Series([False] * len(df))

    static_available = [c for c in STATIC_DIRECT_COLUMNS if c in columns]
    quarantined_available = [c for c in QUARANTINE_DYNAMIC_LOOKING_COLUMNS if c in columns]

    blink_yes = int(normalize_yes_no(df["점멸등운영여부"]).sum()) if "점멸등운영여부" in df.columns else None
    pedestrian_yes = int(normalize_yes_no(df["보행자작동신호기유무"]).sum()) if "보행자작동신호기유무" in df.columns else None
    remaining_timer_yes = int(normalize_yes_no(df["잔여시간표시기유무"]).sum()) if "잔여시간표시기유무" in df.columns else None
    acoustic_yes = int(normalize_yes_no(df["시각장애인용음향신호기유무"]).sum()) if "시각장애인용음향신호기유무" in df.columns else None

    coordinate_summary = {
        "missing_coordinate_columns": missing_coord,
        "valid_coordinate_rows": int(coord_valid.sum()) if len(df) else 0,
        "invalid_coordinate_rows": int((~coord_valid).sum()) if len(df) else 0,
        "lat_min": float(lat[coord_valid].min()) if bool(coord_valid.any()) else None,
        "lat_max": float(lat[coord_valid].max()) if bool(coord_valid.any()) else None,
        "lon_min": float(lon[coord_valid].min()) if bool(coord_valid.any()) else None,
        "lon_max": float(lon[coord_valid].max()) if bool(coord_valid.any()) else None,
    }

    duplicate_signal_id_count = None
    if "신호등관리번호" in df.columns:
        duplicate_signal_id_count = int(df["신호등관리번호"].duplicated(keep=False).sum())

    feature_admission = {
        "allowed_now": [
            "signal_count_100m", "signal_count_250m", "signal_count_500m",
            "nearest_signal_distance_m", "pedestrian_signal_count_250m",
            "blink_signal_ratio_250m", "controlled_signal_ratio_250m",
            "signal_delay_risk_proxy", "intersection_complexity_proxy",
            "edge_signal_count", "edge_signal_density_per_km",
            "edge_control_complexity_proxy",
        ],
        "not_allowed_now": FORBIDDEN_DYNAMIC_SIGNAL_FIELDS,
    }

    warnings: List[str] = []
    if missing_coord:
        warnings.append(f"Missing coordinate columns: {missing_coord}")
    if quarantined_available:
        warnings.append("Dynamic-looking signal timing columns exist, but they are quarantined as static metadata only.")
    if duplicate_signal_id_count:
        warnings.append("신호등관리번호 has duplicate values. This may be normal if each physical signal head is a row; do not assume unique intersection ID.")

    return {
        "artifact_version": "daegu_signal_csv_preflight_report_v2",
        "step": 99,
        "source_csv_path": str(csv_path),
        "csv_encoding": encoding,
        "row_count": int(len(df)),
        "column_count": int(len(columns)),
        "columns": columns,
        "static_direct_columns_available": static_available,
        "coordinate_summary": coordinate_summary,
        "signal_id_column": "신호등관리번호" if "신호등관리번호" in columns else None,
        "duplicate_signal_id_count": duplicate_signal_id_count,
        "flag_column_summary": {
            "blink_yes_count_from_점멸등운영여부": blink_yes,
            "pedestrian_yes_count_from_보행자작동신호기유무": pedestrian_yes,
            "remaining_timer_yes_count_from_잔여시간표시기유무": remaining_timer_yes,
            "acoustic_yes_count_from_시각장애인용음향신호기유무": acoustic_yes,
        },
        "top_value_samples": {
            "시군구명": top_values(df, "시군구명"),
            "신호등구분": top_values(df, "신호등구분"),
            "신호제어방식": top_values(df, "신호제어방식"),
            "신호시간결정방식": top_values(df, "신호시간결정방식"),
            "점멸등운영여부": top_values(df, "점멸등운영여부"),
            "보행자작동신호기유무": top_values(df, "보행자작동신호기유무"),
            "데이터기준일자": top_values(df, "데이터기준일자"),
        },
        "dynamic_quarantine": {
            "columns_present": quarantined_available,
            "decision": "quarantine_as_static_metadata_only",
            "not_allowed_interpretations": [
                "red-light delay", "green time", "cycle length",
                "phase sequence", "signal offset", "real-time signal state",
            ],
        },
        "feature_admission": feature_admission,
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
            "green_time_claim_allowed": False,
            "cycle_length_claim_allowed": False,
        },
        "step_99_ready": bool((not missing_coord) and coordinate_summary["valid_coordinate_rows"] > 0),
        "warnings": warnings,
    }


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    coord = report["coordinate_summary"]
    lines = [
        "# Daegu Signal CSV Preflight Report",
        "",
        f"- artifact_version: `{report['artifact_version']}`",
        f"- step: `{report['step']}`",
        f"- source_csv_path: `{report['source_csv_path']}`",
        f"- csv_encoding: `{report['csv_encoding']}`",
        f"- row_count: `{report['row_count']}`",
        f"- column_count: `{report['column_count']}`",
        f"- step_99_ready: `{report['step_99_ready']}`",
        "",
        "## Coordinate Summary",
        "",
        f"- valid_coordinate_rows: `{coord['valid_coordinate_rows']}`",
        f"- invalid_coordinate_rows: `{coord['invalid_coordinate_rows']}`",
        f"- lat_range: `{coord['lat_min']} ~ {coord['lat_max']}`",
        f"- lon_range: `{coord['lon_min']} ~ {coord['lon_max']}`",
        "",
        "## Dynamic-Looking Columns Quarantined",
        "",
    ]
    for c in report["dynamic_quarantine"]["columns_present"]:
        lines.append(f"- `{c}`")
    lines.extend([
        "",
        "These columns must not be interpreted as measured red-light delay, green time, cycle length, phase sequence, signal offset, or real-time signal state.",
        "",
        "## Allowed Feature Families",
        "",
    ])
    for c in report["feature_admission"]["allowed_now"]:
        lines.append(f"- `{c}`")
    lines.extend(["", "## Not Allowed Now", ""])
    for c in report["feature_admission"]["not_allowed_now"]:
        lines.append(f"- `{c}`")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for w in report["warnings"]:
            lines.append(f"- {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-csv", required=True)
    parser.add_argument("--output-dir", default="artifacts/signal_features_v2_preflight")
    args = parser.parse_args()

    signal_csv = Path(args.signal_csv)
    if not signal_csv.exists():
        raise SystemExit(f"signal CSV not found: {signal_csv}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(signal_csv)
    json_path = out_dir / "daegu_signal_csv_preflight_report.json"
    md_path = out_dir / "daegu_signal_csv_preflight_report.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)

    print("[OK] Step 99 Daegu signal CSV preflight complete")
    print(f"[OK] json_report : {json_path}")
    print(f"[OK] md_report   : {md_path}")
    print(f"[OK] row_count   : {report['row_count']}")
    print(f"[OK] coord_valid : {report['coordinate_summary']['valid_coordinate_rows']}")
    print(f"[OK] ready       : {report['step_99_ready']}")

    if report["warnings"]:
        print("[WARN] warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")

    if not report["step_99_ready"]:
        raise SystemExit("[FAIL] Step 99 preflight found blockers")


if __name__ == "__main__":
    main()
