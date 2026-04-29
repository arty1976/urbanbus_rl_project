#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 99-E — BIS API data source integration audit.

This script links already-created Daegu BIS API artifact CSV files from:
- /getBs02 route-stop sequence collection
- /getPos02 repeated live-position sampling
- /getRealtime02 ETA sampling

Important safety contract:
- No DB write.
- No tensor DB overwrite.
- No additional API call.
- Reads artifact CSV/JSON files only.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ARTIFACT_VERSION = "bis_api_integration_step99e_v1"

DEFAULT_GETBS02_PATH = Path(
    "artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/"
    "getbs02_route_stop_sequence_normalized.csv"
)
DEFAULT_GETPOS02_TIMESERIES_PATH = Path(
    "artifacts/daegu_bis_api_audit/getpos02_repeated_sampling/20260429_090603/"
    "getpos02_trajectory_candidate_timeseries.csv"
)
DEFAULT_GETPOS02_SUMMARY_PATH = Path(
    "artifacts/daegu_bis_api_audit/getpos02_repeated_sampling/20260429_090603/"
    "getpos02_trajectory_candidate_summary.csv"
)
DEFAULT_GETREALTIME02_ETA_PATH = Path(
    "artifacts/daegu_bis_api_audit/getrealtime02_eta_sampling_arrlist_fix/"
    "getrealtime02_eta_normalized.csv"
)
DEFAULT_GETREALTIME02_HEADWAY_PATH = Path(
    "artifacts/daegu_bis_api_audit/getrealtime02_eta_sampling_arrlist_fix/"
    "getrealtime02_headway_candidate.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/daegu_bis_api_audit/bis_api_integration_step99e"
)


COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "route_id": (
        "route_id",
        "routeId",
        "routeid",
        "routeID",
        "ROUTE_ID",
        "bs_route_id",
    ),
    "direction_id": (
        "direction_id",
        "direction",
        "dir",
        "dir_id",
        "route_dir",
        "route_direction",
        "moveDir",
        "move_dir",
        "movedir",
        "updown",
    ),
    "stop_id": (
        "stop_id",
        "bs_id",
        "bsId",
        "bsid",
        "bus_stop_id",
        "busStopId",
        "node_id",
        "station_id",
    ),
    "stop_order": (
        "stop_order",
        "ordered_stop_sequence",
        "ordered_sequence",
        "sequence",
        "seq",
        "bs_seq",
        "bsSeq",
        "bs_ord",
        "bsOrd",
        "stop_seq",
        "stop_order_no",
        "current_stop_order",
        "current_route_sequence",
    ),
    "route_no": (
        "route_no",
        "routeNo",
        "route_nm",
        "routeNm",
        "route_name",
        "routeName",
        "line_no",
    ),
    "stop_name": (
        "stop_name",
        "bs_name",
        "bsNm",
        "bsnm",
        "bus_stop_name",
        "stopNm",
        "station_name",
    ),
    "sample_index": (
        "sample_index",
        "sample_idx",
        "sampling_index",
        "sample_no",
        "idx",
    ),
    "bus_id_candidate": (
        "bus_id_candidate",
        "bus_id_or_vehicle_no",
        "vehicle_id",
        "vehicle_no",
        "bus_no",
        "busNo",
        "vhcNo",
        "vhcNo2",
        "vhc_no",
        "vhc_no2",
    ),
    "current_stop_id": (
        "current_stop_id",
        "bsId",
        "bs_id",
        "stop_id",
        "current_bs_id",
        "current_station_id",
    ),
    "current_stop_order": (
        "current_stop_order",
        "current_route_sequence",
        "seq",
        "bsSeq",
        "bs_seq",
        "stop_order",
    ),
    "x_pos": (
        "x_pos",
        "xPos",
        "xpos",
        "gps_x",
        "lon",
        "lng",
        "longitude",
    ),
    "y_pos": (
        "y_pos",
        "yPos",
        "ypos",
        "gps_y",
        "lat",
        "latitude",
    ),
    "event_time_raw": (
        "event_time_raw",
        "event_time",
        "arTime",
        "ar_time",
        "timestamp",
        "created_at",
        "sample_time",
    ),
}


CLASSIFICATION_UPDATE: Dict[str, str] = {
    "route_id": "observed_candidate_full_collection_234_of_238_routes",
    "direction_id": "observed_candidate_full_collection_234_of_238_routes_cross_confirmed",
    "ordered_stop_sequence": "observed_candidate_full_collection_234_of_238_routes",
    "bus_id_or_vehicle_no": "repeated_sample_observed_candidate_from_getPos02_vhcNo2",
    "live_position_xy": "repeated_sample_observed_candidate_from_getPos02_xPos_yPos",
    "current_route_sequence": "repeated_sample_observed_candidate_from_getPos02_seq",
    "current_stop_id": "repeated_sample_observed_candidate_from_getPos02_bsId",
    "vehicle_trajectory": "trajectory_candidate_from_repeated_getPos02_sampling",
    "getRealtime02_eta": "observed_candidate",
    "eta_based_headway": "candidate_from_getRealtime02",
    "actual_headway": "not_observed",
    "actual_arrival_departure_time": "not_observed",
    "actual_dwell": "not_observed",
    "passenger_wait_age_distribution": "missing",
    "vehicle_load": "missing",
    "left_behind_passengers": "missing",
}


@dataclass(frozen=True)
class AuditPaths:
    getbs02_path: Path
    getpos02_timeseries_path: Path
    getpos02_summary_path: Path
    getrealtime02_eta_path: Path
    getrealtime02_headway_path: Path
    output_root: Path


def _casefold_column_map(columns: Iterable[str]) -> Dict[str, str]:
    """Return a case-insensitive lookup from normalized column name to actual name."""
    out: Dict[str, str] = {}
    for col in columns:
        key = re.sub(r"[\s_\-]+", "", str(col)).casefold()
        out.setdefault(key, str(col))
    return out


def find_column(
    df: pd.DataFrame,
    aliases: Sequence[str],
    *,
    required: bool,
    logical_name: str,
) -> Optional[str]:
    """Find a column by explicit or normalized/case-insensitive aliases."""
    direct = {str(c): str(c) for c in df.columns}
    for alias in aliases:
        if alias in direct:
            return direct[alias]

    normalized = _casefold_column_map(df.columns)
    for alias in aliases:
        key = re.sub(r"[\s_\-]+", "", str(alias)).casefold()
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Required column not found for {logical_name!r}. "
            f"aliases={list(aliases)}, available={list(df.columns)}"
        )
    return None


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    """Read a CSV with common encodings used by Korean/Windows workflows."""
    if not path.exists():
        raise FileNotFoundError(f"required input CSV not found: {path}")

    errors: List[str] = []
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            errors.append(f"{enc}: {e}")
        except Exception as e:
            # Keep trying only on encoding-like issues; re-raise structural problems.
            errors.append(f"{enc}: {type(e).__name__}: {e}")

    raise RuntimeError(f"failed to read CSV with known encodings: {path}; errors={errors}")


def normalize_key_value(value: Any) -> Optional[str]:
    """Normalize IDs so 123, '123', and '123.0' join consistently."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def normalize_key_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_key_value).astype("string")


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def optional_standard_column(
    df: pd.DataFrame,
    aliases: Sequence[str],
    logical_name: str,
    default: Any = pd.NA,
) -> pd.Series:
    col = find_column(df, aliases, required=False, logical_name=logical_name)
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]


def standardize_getbs02(raw: pd.DataFrame) -> pd.DataFrame:
    route_col = find_column(raw, COLUMN_ALIASES["route_id"], required=True, logical_name="route_id")
    direction_col = find_column(raw, COLUMN_ALIASES["direction_id"], required=True, logical_name="direction_id")
    stop_col = find_column(raw, COLUMN_ALIASES["stop_id"], required=True, logical_name="stop_id")
    order_col = find_column(raw, COLUMN_ALIASES["stop_order"], required=True, logical_name="stop_order")

    out = pd.DataFrame({
        "route_id": normalize_key_series(raw[route_col]),
        "direction_id": normalize_key_series(raw[direction_col]),
        "stop_id": normalize_key_series(raw[stop_col]),
        "getbs02_stop_order": numeric_series(raw[order_col]),
        "route_no": optional_standard_column(raw, COLUMN_ALIASES["route_no"], "route_no"),
        "stop_name": optional_standard_column(raw, COLUMN_ALIASES["stop_name"], "stop_name"),
    })

    out = out.dropna(subset=["route_id", "direction_id", "stop_id"]).copy()
    out["route_direction_key"] = out["route_id"].astype(str) + "|" + out["direction_id"].astype(str)

    sort_cols = ["route_id", "direction_id", "getbs02_stop_order", "stop_id"]
    out = out.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return out


def make_getbs02_join_base(getbs02: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create one row per route/direction/stop to avoid accidental row multiplication."""
    dup_mask = getbs02.duplicated(["route_id", "direction_id", "stop_id"], keep=False)
    duplicate_count = int(dup_mask.sum())

    join_base = (
        getbs02.sort_values(["route_id", "direction_id", "stop_id", "getbs02_stop_order"], na_position="last")
        .drop_duplicates(["route_id", "direction_id", "stop_id"], keep="first")
        .reset_index(drop=True)
    )

    summary = {
        "raw_route_stop_rows": int(len(getbs02)),
        "join_base_rows": int(len(join_base)),
        "duplicate_route_direction_stop_rows": duplicate_count,
        "unique_route_direction_count": int(
            getbs02[["route_id", "direction_id"]].drop_duplicates().shape[0]
        ),
        "unique_route_count": int(getbs02["route_id"].nunique(dropna=True)),
        "unique_stop_count": int(getbs02["stop_id"].nunique(dropna=True)),
    }
    return join_base, summary


def standardize_getpos02(raw: pd.DataFrame) -> pd.DataFrame:
    route_col = find_column(raw, COLUMN_ALIASES["route_id"], required=True, logical_name="getPos02.route_id")
    direction_col = find_column(raw, COLUMN_ALIASES["direction_id"], required=True, logical_name="getPos02.direction_id")
    stop_col = find_column(raw, COLUMN_ALIASES["current_stop_id"], required=True, logical_name="getPos02.current_stop_id")
    order_col = find_column(raw, COLUMN_ALIASES["current_stop_order"], required=False, logical_name="getPos02.current_stop_order")

    out = pd.DataFrame({
        "sample_index": optional_standard_column(raw, COLUMN_ALIASES["sample_index"], "sample_index"),
        "route_id": normalize_key_series(raw[route_col]),
        "direction_id": normalize_key_series(raw[direction_col]),
        "bus_id_candidate": optional_standard_column(raw, COLUMN_ALIASES["bus_id_candidate"], "bus_id_candidate"),
        "current_stop_id": normalize_key_series(raw[stop_col]),
        "current_stop_order": numeric_series(raw[order_col]) if order_col else pd.Series([pd.NA] * len(raw), index=raw.index),
        "x_pos": optional_standard_column(raw, COLUMN_ALIASES["x_pos"], "x_pos"),
        "y_pos": optional_standard_column(raw, COLUMN_ALIASES["y_pos"], "y_pos"),
        "event_time_raw": optional_standard_column(raw, COLUMN_ALIASES["event_time_raw"], "event_time_raw"),
    })
    out = out.dropna(subset=["route_id", "direction_id", "current_stop_id"]).copy()
    out["route_direction_key"] = out["route_id"].astype(str) + "|" + out["direction_id"].astype(str)
    return out.reset_index(drop=True)


def standardize_route_stop_csv(
    raw: pd.DataFrame,
    *,
    source_prefix: str,
    require_stop_id: bool = True,
) -> pd.DataFrame:
    route_col = find_column(raw, COLUMN_ALIASES["route_id"], required=True, logical_name=f"{source_prefix}.route_id")
    direction_col = find_column(raw, COLUMN_ALIASES["direction_id"], required=True, logical_name=f"{source_prefix}.direction_id")
    stop_col = find_column(raw, COLUMN_ALIASES["stop_id"], required=require_stop_id, logical_name=f"{source_prefix}.stop_id")

    out = pd.DataFrame({
        "route_id": normalize_key_series(raw[route_col]),
        "direction_id": normalize_key_series(raw[direction_col]),
        "stop_id": normalize_key_series(raw[stop_col]) if stop_col else pd.Series([pd.NA] * len(raw), index=raw.index, dtype="string"),
    })

    # Preserve all non-conflicting raw columns for debugging/audit traceability.
    for col in raw.columns:
        target = str(col)
        if target not in out.columns:
            out[f"{source_prefix}_{target}"] = raw[col]

    out = out.dropna(subset=["route_id", "direction_id", "stop_id"]).copy()
    out["route_direction_key"] = out["route_id"].astype(str) + "|" + out["direction_id"].astype(str)
    return out.reset_index(drop=True)


def join_getpos02_to_getbs02(getpos02: pd.DataFrame, getbs02_join: pd.DataFrame) -> pd.DataFrame:
    right = getbs02_join[[
        "route_id",
        "direction_id",
        "stop_id",
        "getbs02_stop_order",
        "route_no",
        "stop_name",
    ]].copy()

    merged = getpos02.merge(
        right,
        how="left",
        left_on=["route_id", "direction_id", "current_stop_id"],
        right_on=["route_id", "direction_id", "stop_id"],
        indicator=True,
    )

    merged["route_stop_match"] = merged["_merge"].eq("both")
    merged["order_delta"] = numeric_series(merged["current_stop_order"]) - numeric_series(merged["getbs02_stop_order"])
    merged["exact_order_match"] = (
        merged["route_stop_match"]
        & merged["current_stop_order"].notna()
        & merged["getbs02_stop_order"].notna()
        & (merged["order_delta"] == 0)
    )
    merged["missing_route_stop_match"] = ~merged["route_stop_match"]

    return merged.drop(columns=["_merge"])


def join_route_stop_to_getbs02(df: pd.DataFrame, getbs02_join: pd.DataFrame) -> pd.DataFrame:
    right = getbs02_join[[
        "route_id",
        "direction_id",
        "stop_id",
        "getbs02_stop_order",
        "route_no",
        "stop_name",
    ]].copy()

    merged = df.merge(
        right,
        how="left",
        on=["route_id", "direction_id", "stop_id"],
        indicator=True,
    )
    merged["route_stop_match"] = merged["_merge"].eq("both")
    merged["missing_route_stop_match"] = ~merged["route_stop_match"]
    return merged.drop(columns=["_merge"])


def summarize_match(df: pd.DataFrame, *, match_col: str = "route_stop_match") -> Dict[str, Any]:
    total = int(len(df))
    matched = int(df[match_col].fillna(False).sum()) if total else 0
    return {
        "row_count": total,
        "matched_rows": matched,
        "missing_match_rows": int(total - matched),
        "match_coverage": float(matched / total) if total else None,
    }


def summarize_getpos02_order_match(df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(df))
    exact = int(df["exact_order_match"].fillna(False).sum()) if total else 0
    nonzero_delta = int((df["order_delta"].notna() & (df["order_delta"] != 0)).sum()) if total else 0
    delta_known = int(df["order_delta"].notna().sum()) if total else 0
    missing_match = int(df["missing_route_stop_match"].fillna(False).sum()) if total else 0

    delta_desc: Dict[str, Any] = {}
    if delta_known:
        s = numeric_series(df.loc[df["order_delta"].notna(), "order_delta"])
        delta_desc = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }

    return {
        "row_count": total,
        "exact_order_match_count": exact,
        "exact_order_match_coverage": float(exact / total) if total else None,
        "nonzero_order_delta_count": nonzero_delta,
        "known_order_delta_count": delta_known,
        "missing_route_stop_match_count": missing_match,
        "order_delta_summary": delta_desc,
    }


def route_direction_set(df: pd.DataFrame) -> set[Tuple[str, str]]:
    if df.empty:
        return set()
    return {
        (str(r.route_id), str(r.direction_id))
        for r in df[["route_id", "direction_id"]].dropna().drop_duplicates().itertuples(index=False)
    }


def summarize_route_direction_consistency(
    getbs02: pd.DataFrame,
    getpos02: pd.DataFrame,
    getrealtime02: pd.DataFrame,
) -> Dict[str, Any]:
    bs = route_direction_set(getbs02)
    pos = route_direction_set(getpos02)
    realtime = route_direction_set(getrealtime02)

    all_three = bs & pos & realtime
    pos_unmatched = pos - bs
    realtime_unmatched = realtime - bs

    return {
        "getbs02_route_direction_count": int(len(bs)),
        "getpos02_route_direction_count": int(len(pos)),
        "getrealtime02_route_direction_count": int(len(realtime)),
        "intersection_all_three_count": int(len(all_three)),
        "getpos02_route_direction_unmatched_to_getbs02_count": int(len(pos_unmatched)),
        "getrealtime02_route_direction_unmatched_to_getbs02_count": int(len(realtime_unmatched)),
        "getpos02_route_direction_unmatched_to_getbs02_sample": sorted(list(pos_unmatched))[:20],
        "getrealtime02_route_direction_unmatched_to_getbs02_sample": sorted(list(realtime_unmatched))[:20],
    }


def coverage_pass_or_review(
    summaries: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    min_match_coverage: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    warnings: List[Dict[str, Any]] = []
    hard_zero = False

    for name, summary in summaries:
        cov = summary.get("match_coverage")
        rows = int(summary.get("row_count", 0) or 0)
        if rows == 0:
            warnings.append({
                "code": "empty_input",
                "source": name,
                "message": f"{name} has zero rows; coverage not evaluated.",
            })
            continue
        if cov is None:
            continue
        if cov <= 0:
            hard_zero = True
            warnings.append({
                "code": "zero_match_coverage",
                "source": name,
                "match_coverage": cov,
                "message": f"{name} did not match any getBs02 route-stop rows.",
            })
        elif cov < min_match_coverage:
            warnings.append({
                "code": "low_match_coverage",
                "source": name,
                "match_coverage": cov,
                "min_match_coverage": min_match_coverage,
                "message": f"{name} match coverage is below threshold.",
            })

    if hard_zero:
        return "REVIEW_REQUIRED", warnings
    if warnings:
        return "PASS_WITH_REVIEW_REQUIRED", warnings
    return "PASS", warnings


def sequence_semantic_warning(order_summary: Dict[str, Any], min_exact_order_coverage: float) -> Optional[Dict[str, Any]]:
    coverage = order_summary.get("exact_order_match_coverage")
    if coverage is None:
        return None
    if coverage < min_exact_order_coverage:
        return {
            "code": "sequence_semantic_review_required",
            "exact_order_match_coverage": coverage,
            "min_exact_order_coverage": min_exact_order_coverage,
            "message": (
                "getPos02 current_stop_order does not exactly match getBs02 stop_order "
                "at the expected threshold. Keep route-stop matching, but review whether "
                "getPos02 seq and getBs02 ordered_stop_sequence use the same sequence semantics."
            ),
        }
    return None


def build_classification_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "step": "99-E",
        "classification_update": CLASSIFICATION_UPDATE,
        "claim_guards": {
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "db_write_performed": False,
            "tensor_db_overwrite_performed": False,
            "additional_api_calls_performed": False,
        },
        "source_evidence": {
            "getBs02": {
                "status": "PASS_WITH_KNOWN_EXCEPTIONS",
                "attempted_routes": 238,
                "success_routes": 234,
                "auth_error_routes": 4,
                "normalized_rows_expected_from_step99a": 20508,
            },
            "getPos02": {
                "total_timeseries_rows_expected_from_step99c3": 1452,
                "vehicle_group_count": 290,
                "repeated_vehicle_count": 271,
                "trajectory_candidate_count": 248,
            },
            "getRealtime02": {
                "normalized_eta_rows_expected_from_step99d": 58,
                "eta_based_headway_candidates_expected_from_step99d": 23,
            },
        },
        "integration_summary": {
            "audit_status": report.get("audit_status"),
            "getpos02_match": report.get("getpos02_to_getbs02", {}),
            "getrealtime02_eta_match": report.get("getrealtime02_to_getbs02", {}),
            "headway_candidate_match": report.get("headway_candidate_to_getbs02", {}),
            "route_direction_consistency": report.get("route_direction_consistency", {}),
        },
        "non_claim_notes": [
            "actual_headway remains not_observed; ETA-based headway is only a candidate.",
            "actual_arrival_departure_time remains not_observed.",
            "actual_dwell remains not_observed.",
            "vehicle_trajectory is only a candidate from repeated getPos02 live snapshots.",
        ],
    }


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    def pct(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"

    pos = report["getpos02_to_getbs02"]
    eta = report["getrealtime02_to_getbs02"]
    headway = report["headway_candidate_to_getbs02"]
    order = report["getpos02_order_match"]
    rd = report["route_direction_consistency"]

    lines = [
        "# Step 99-E — BIS API data source integration audit",
        "",
        "## Safety contract",
        "",
        "- DB(Database=데이터베이스) write: **false**",
        "- tensor DB(Database=데이터베이스) overwrite: **false**",
        "- Additional API(Application Programming Interface=응용 프로그램 인터페이스) calls: **false**",
        "- paper_level_claim_allowed: **false**",
        "- causal_performance_claim_allowed: **false**",
        "",
        "## Audit status",
        "",
        f"- `audit_status`: `{report['audit_status']}`",
        "",
        "## Match coverage",
        "",
        "| Source | Rows | Matched | Missing | Coverage |",
        "|---|---:|---:|---:|---:|",
        f"| getPos02 → getBs02 | {pos['row_count']} | {pos['matched_rows']} | {pos['missing_match_rows']} | {pct(pos['match_coverage'])} |",
        f"| getRealtime02 ETA → getBs02 | {eta['row_count']} | {eta['matched_rows']} | {eta['missing_match_rows']} | {pct(eta['match_coverage'])} |",
        f"| getRealtime02 headway candidate → getBs02 | {headway['row_count']} | {headway['matched_rows']} | {headway['missing_match_rows']} | {pct(headway['match_coverage'])} |",
        "",
        "## getPos02 order consistency",
        "",
        f"- exact_order_match_count: `{order['exact_order_match_count']}`",
        f"- exact_order_match_coverage: `{pct(order['exact_order_match_coverage'])}`",
        f"- nonzero_order_delta_count: `{order['nonzero_order_delta_count']}`",
        f"- missing_route_stop_match_count: `{order['missing_route_stop_match_count']}`",
        "",
        "## Route/direction consistency",
        "",
        f"- getBs02 route-direction count: `{rd['getbs02_route_direction_count']}`",
        f"- getPos02 route-direction count: `{rd['getpos02_route_direction_count']}`",
        f"- getRealtime02 route-direction count: `{rd['getrealtime02_route_direction_count']}`",
        f"- intersection across all three: `{rd['intersection_all_three_count']}`",
        f"- unmatched getPos02 route-direction count: `{rd['getpos02_route_direction_unmatched_to_getbs02_count']}`",
        f"- unmatched getRealtime02 route-direction count: `{rd['getrealtime02_route_direction_unmatched_to_getbs02_count']}`",
        "",
        "## Classification update",
        "",
        "| Field | Step 99-E classification |",
        "|---|---|",
    ]

    for key, value in CLASSIFICATION_UPDATE.items():
        lines.append(f"| `{key}` | `{value}` |")

    lines.extend([
        "",
        "## Warnings",
        "",
    ])
    warnings = report.get("warnings", [])
    if warnings:
        for w in warnings:
            lines.append(f"- `{w.get('code')}`: {w.get('message')} ({w})")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(paths: AuditPaths, *, min_match_coverage: float = 0.80, min_exact_order_coverage: float = 0.80) -> Dict[str, Any]:
    paths.output_root.mkdir(parents=True, exist_ok=True)

    getbs02_raw = read_csv_any_encoding(paths.getbs02_path)
    getpos02_raw = read_csv_any_encoding(paths.getpos02_timeseries_path)
    # Summary is read to verify the artifact exists and remains part of the audit trace.
    getpos02_summary_raw = read_csv_any_encoding(paths.getpos02_summary_path)
    realtime_eta_raw = read_csv_any_encoding(paths.getrealtime02_eta_path)
    headway_raw = read_csv_any_encoding(paths.getrealtime02_headway_path)

    getbs02 = standardize_getbs02(getbs02_raw)
    getbs02_join, getbs02_summary = make_getbs02_join_base(getbs02)

    getpos02 = standardize_getpos02(getpos02_raw)
    realtime_eta = standardize_route_stop_csv(realtime_eta_raw, source_prefix="eta")
    headway = standardize_route_stop_csv(headway_raw, source_prefix="headway")

    getpos_match = join_getpos02_to_getbs02(getpos02, getbs02_join)
    eta_match = join_route_stop_to_getbs02(realtime_eta, getbs02_join)
    headway_match = join_route_stop_to_getbs02(headway, getbs02_join)

    getpos_summary = summarize_match(getpos_match)
    eta_summary = summarize_match(eta_match)
    headway_summary = summarize_match(headway_match)
    order_summary = summarize_getpos02_order_match(getpos_match)
    route_direction_summary = summarize_route_direction_consistency(getbs02, getpos02, realtime_eta)

    audit_status, warnings = coverage_pass_or_review(
        [
            ("getpos02_to_getbs02", getpos_summary),
            ("getrealtime02_to_getbs02", eta_summary),
            ("headway_candidate_to_getbs02", headway_summary),
        ],
        min_match_coverage=min_match_coverage,
    )

    seq_warning = sequence_semantic_warning(order_summary, min_exact_order_coverage)
    if seq_warning:
        warnings.append(seq_warning)
        if audit_status == "PASS":
            audit_status = "PASS_WITH_REVIEW_REQUIRED"

    report: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": "99-E",
        "audit_status": audit_status,
        "input_paths": {
            "getbs02": str(paths.getbs02_path),
            "getpos02_timeseries": str(paths.getpos02_timeseries_path),
            "getpos02_summary": str(paths.getpos02_summary_path),
            "getrealtime02_eta": str(paths.getrealtime02_eta_path),
            "getrealtime02_headway": str(paths.getrealtime02_headway_path),
        },
        "output_root": str(paths.output_root),
        "claim_guards": {
            "db_write_performed": False,
            "tensor_db_overwrite_performed": False,
            "additional_api_calls_performed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
        "source_row_counts": {
            "getbs02_raw": int(len(getbs02_raw)),
            "getbs02_standardized": int(len(getbs02)),
            "getpos02_timeseries_raw": int(len(getpos02_raw)),
            "getpos02_timeseries_standardized": int(len(getpos02)),
            "getpos02_summary_rows": int(len(getpos02_summary_raw)),
            "getrealtime02_eta_raw": int(len(realtime_eta_raw)),
            "getrealtime02_eta_standardized": int(len(realtime_eta)),
            "getrealtime02_headway_raw": int(len(headway_raw)),
            "getrealtime02_headway_standardized": int(len(headway)),
        },
        "getbs02_base": getbs02_summary,
        "getpos02_to_getbs02": getpos_summary,
        "getpos02_order_match": order_summary,
        "getrealtime02_to_getbs02": eta_summary,
        "headway_candidate_to_getbs02": headway_summary,
        "route_direction_consistency": route_direction_summary,
        "warnings": warnings,
        "classification_update": CLASSIFICATION_UPDATE,
    }

    getpos_match_path = paths.output_root / "getpos02_to_getbs02_match.csv"
    eta_match_path = paths.output_root / "getrealtime02_to_getbs02_match.csv"
    headway_match_path = paths.output_root / "headway_candidate_to_getbs02_match.csv"
    report_json_path = paths.output_root / "bis_api_integration_report.json"
    report_md_path = paths.output_root / "bis_api_integration_report.md"
    classification_path = paths.output_root / "classification_update_step99e.json"

    getpos_match.to_csv(getpos_match_path, index=False, encoding="utf-8-sig")
    eta_match.to_csv(eta_match_path, index=False, encoding="utf-8-sig")
    headway_match.to_csv(headway_match_path, index=False, encoding="utf-8-sig")

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_markdown_report(report_md_path, report)

    classification_payload = build_classification_payload(report)
    with open(classification_path, "w", encoding="utf-8") as f:
        json.dump(classification_payload, f, ensure_ascii=False, indent=2)

    report["output_files"] = {
        "bis_api_integration_report_json": str(report_json_path),
        "bis_api_integration_report_md": str(report_md_path),
        "getpos02_to_getbs02_match": str(getpos_match_path),
        "getrealtime02_to_getbs02_match": str(eta_match_path),
        "headway_candidate_to_getbs02_match": str(headway_match_path),
        "classification_update_step99e": str(classification_path),
    }

    # Rewrite report JSON including output file manifest.
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 99-E BIS API integration audit over existing artifact CSV files only."
    )
    parser.add_argument("--project-root", default=".", help="Project root. Default: current directory.")
    parser.add_argument("--getbs02", default=str(DEFAULT_GETBS02_PATH))
    parser.add_argument("--getpos02-timeseries", default=str(DEFAULT_GETPOS02_TIMESERIES_PATH))
    parser.add_argument("--getpos02-summary", default=str(DEFAULT_GETPOS02_SUMMARY_PATH))
    parser.add_argument("--getrealtime02-eta", default=str(DEFAULT_GETREALTIME02_ETA_PATH))
    parser.add_argument("--getrealtime02-headway", default=str(DEFAULT_GETREALTIME02_HEADWAY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--min-match-coverage", type=float, default=0.80)
    parser.add_argument("--min-exact-order-coverage", type=float, default=0.80)
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    paths = AuditPaths(
        getbs02_path=resolve_path(project_root, args.getbs02),
        getpos02_timeseries_path=resolve_path(project_root, args.getpos02_timeseries),
        getpos02_summary_path=resolve_path(project_root, args.getpos02_summary),
        getrealtime02_eta_path=resolve_path(project_root, args.getrealtime02_eta),
        getrealtime02_headway_path=resolve_path(project_root, args.getrealtime02_headway),
        output_root=resolve_path(project_root, args.output_root),
    )

    report = run_audit(
        paths,
        min_match_coverage=float(args.min_match_coverage),
        min_exact_order_coverage=float(args.min_exact_order_coverage),
    )

    print("[OK] Step 99-E BIS API integration audit completed")
    print(f"[OK] audit_status: {report['audit_status']}")
    print(f"[OK] output_root : {paths.output_root}")
    print(
        "[OK] getPos02 coverage: "
        f"{report['getpos02_to_getbs02']['match_coverage']}"
    )
    print(
        "[OK] getRealtime02 ETA coverage: "
        f"{report['getrealtime02_to_getbs02']['match_coverage']}"
    )
    print(
        "[OK] headway candidate coverage: "
        f"{report['headway_candidate_to_getbs02']['match_coverage']}"
    )


if __name__ == "__main__":
    main()
