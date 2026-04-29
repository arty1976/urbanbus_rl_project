from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ARTIFACT_VERSION = "daegu_bis_api_audit_step99c3_getpos02_trajectory_candidate_v1"


ALIASES = {
    "route_id": {"routeId", "route_id", "ROUTE_ID"},
    "route_no": {"routeNo", "route_no", "routeNm", "routeName"},
    "direction_id": {"moveDir", "move_dir", "direction_id", "dir", "dirCode"},
    "event_time_raw": {"arTime", "arrTime", "arrivalTime", "eventTime", "tm", "time"},
    "current_stop_order": {"seq", "sequence", "stopSeq", "stop_order", "bsSeq", "ord"},
    "current_stop_id": {"bsId", "bs_id", "stopId", "stop_id", "nodeId"},
    "x_pos": {"xPos", "xpos", "x", "longitude", "lon", "lng"},
    "y_pos": {"yPos", "ypos", "y", "latitude", "lat"},
    "bus_id_candidate": {
        "vhcNo2", "vhcno2", "VHCNO2", "VHC_NO2", "vhc_no2",
        "vhcNo", "vhcno", "VHCNO", "VHC_NO", "vhc_no",
        "busId", "bus_id", "vehicleNo", "vehicle_no", "vehicleId",
        "vehNo", "carNo", "busNo", "plateNo", "vhNo",
    },
    "bus_tcd2": {"busTCd2", "bus_tcd2", "busTypeCd2"},
    "bus_tcd3": {"busTCd3", "bus_tcd3", "busTypeCd3"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_key(v: Any) -> str:
    return str(v).strip().replace("_", "").lower()


def get_first(row: Dict[str, Any], field: str) -> Any:
    aliases = ALIASES[field]
    for k, v in row.items():
        if any(norm_key(k) == norm_key(a) for a in aliases):
            return v
    return None


def to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def as_str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def haversine_m(x1: Optional[float], y1: Optional[float], x2: Optional[float], y2: Optional[float]) -> Optional[float]:
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    lon1 = math.radians(float(x1))
    lat1 = math.radians(float(y1))
    lon2 = math.radians(float(x2))
    lat2 = math.radians(float(y2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return 6371000.0 * c


def extract_sample_index(sample_dir: Path) -> int:
    m = re.search(r"sample_(\d+)", sample_dir.name)
    if not m:
        raise ValueError(f"cannot parse sample index from {sample_dir}")
    return int(m.group(1))


def load_manifest(run_dir: Path) -> Dict[int, Dict[str, str]]:
    manifest = run_dir / "step99c2_repeated_sampling_manifest.csv"
    out: Dict[int, Dict[str, str]] = {}

    if not manifest.exists():
        return out

    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row.get("sample_index", "0"))
            except Exception:
                continue
            out[idx] = row

    return out


def extract_items_from_json(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if not isinstance(obj, dict):
        return []

    # Provider error response.
    header = obj.get("header")
    if isinstance(header, dict) and header.get("success") is False:
        return []

    paths = [
        ["body", "items"],
        ["response", "body", "items"],
        ["items"],
        ["body", "item"],
        ["response", "body", "item"],
    ]

    for path in paths:
        cur: Any = obj
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break

        if not ok:
            continue

        if isinstance(cur, list):
            return [x for x in cur if isinstance(x, dict)]
        if isinstance(cur, dict):
            # Some providers wrap items as {"item": [...]}.
            for key in ("item", "items"):
                if key in cur:
                    val = cur[key]
                    if isinstance(val, list):
                        return [x for x in val if isinstance(x, dict)]
                    if isinstance(val, dict):
                        return [val]
            return [cur]

    rows: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if any(norm_key(k) in {"routeid", "vhcno2", "xpos", "ypos", "bsid"} for k in x.keys()):
                rows.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return rows


def extract_items_from_csv_text(raw_text: str) -> List[Dict[str, Any]]:
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return []

    header = lines[0]
    if "," not in header:
        return []

    if "routeId" not in header and "vhcNo2" not in header and "xPos" not in header:
        return []

    reader = csv.DictReader(lines)
    return [dict(r) for r in reader]


def parse_raw_file(path: Path) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    raw_text = path.read_text(encoding="utf-8-sig", errors="replace").strip()

    if not raw_text:
        return "empty", [], None

    try:
        if raw_text.startswith("{") or raw_text.startswith("["):
            obj = json.loads(raw_text)
            return "json", extract_items_from_json(obj), None

        if raw_text.startswith("<"):
            return "xml", [], "xml response is not parsed by Step 99-C3 analyzer"

        if "," in raw_text.splitlines()[0]:
            return "csv", extract_items_from_csv_text(raw_text), None

        return "unknown", [], "unknown raw response format"
    except Exception as e:
        return "error", [], str(e)


def normalize_raw_row(
    row: Dict[str, Any],
    sample_index: int,
    sample_tag: str,
    manifest_row: Dict[str, str],
    raw_path: Path,
) -> Optional[Dict[str, Any]]:
    bus_id = as_str_or_none(get_first(row, "bus_id_candidate"))
    route_id = as_str_or_none(get_first(row, "route_id"))
    direction_id = as_str_or_none(get_first(row, "direction_id"))
    x_pos = to_float(get_first(row, "x_pos"))
    y_pos = to_float(get_first(row, "y_pos"))
    stop_order = to_int(get_first(row, "current_stop_order"))

    if not route_id:
        return None

    # Need at least vehicle id or location/sequence evidence.
    if not bus_id and x_pos is None and y_pos is None and stop_order is None:
        return None

    return {
        "sample_index": sample_index,
        "sample_tag": sample_tag,
        "sample_started_at": manifest_row.get("started_at", ""),
        "sample_ended_at": manifest_row.get("ended_at", ""),
        "route_id": route_id,
        "route_no": as_str_or_none(get_first(row, "route_no")),
        "direction_id": direction_id,
        "bus_id_candidate": bus_id,
        "current_stop_order": stop_order,
        "current_stop_id": as_str_or_none(get_first(row, "current_stop_id")),
        "x_pos": x_pos,
        "y_pos": y_pos,
        "event_time_raw": as_str_or_none(get_first(row, "event_time_raw")),
        "bus_tcd2": as_str_or_none(get_first(row, "bus_tcd2")),
        "bus_tcd3": as_str_or_none(get_first(row, "bus_tcd3")),
        "raw_path": str(raw_path),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stable_value_set(values: Iterable[Any]) -> List[str]:
    out = []
    seen = set()
    for v in values:
        if v is None or v == "":
            continue
        s = str(v)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def make_summary(rows: List[Dict[str, Any]], xy_threshold_m: float) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        route_id = str(r.get("route_id") or "")
        direction_id = str(r.get("direction_id") or "")
        bus_id = str(r.get("bus_id_candidate") or "")
        if not route_id or not bus_id:
            continue
        grouped[(route_id, direction_id, bus_id)].append(r)

    summaries: List[Dict[str, Any]] = []

    for (route_id, direction_id, bus_id), group in grouped.items():
        g = sorted(group, key=lambda x: (int(x["sample_index"]), str(x.get("event_time_raw") or "")))
        sample_values = [int(x["sample_index"]) for x in g]
        unique_samples = sorted(set(sample_values))

        first = g[0]
        last = g[-1]

        stop_orders = [x.get("current_stop_order") for x in g if x.get("current_stop_order") is not None]
        stop_ids = stable_value_set(x.get("current_stop_id") for x in g)

        min_stop_order = min(stop_orders) if stop_orders else None
        max_stop_order = max(stop_orders) if stop_orders else None
        first_stop_order = first.get("current_stop_order")
        last_stop_order = last.get("current_stop_order")
        stop_order_delta = None
        if first_stop_order is not None and last_stop_order is not None:
            stop_order_delta = int(last_stop_order) - int(first_stop_order)

        first_last_distance_m = haversine_m(first.get("x_pos"), first.get("y_pos"), last.get("x_pos"), last.get("y_pos"))

        incremental_distance_m = 0.0
        incremental_valid = False
        for a, b in zip(g[:-1], g[1:]):
            d = haversine_m(a.get("x_pos"), a.get("y_pos"), b.get("x_pos"), b.get("y_pos"))
            if d is not None:
                incremental_distance_m += d
                incremental_valid = True

        unique_xy = stable_value_set(
            f"{x.get('x_pos')}|{x.get('y_pos')}"
            for x in g
            if x.get("x_pos") is not None and x.get("y_pos") is not None
        )

        stop_order_changed = len(set(str(x) for x in stop_orders)) > 1
        stop_id_changed = len(stop_ids) > 1
        xy_changed = (first_last_distance_m is not None and first_last_distance_m >= xy_threshold_m) or len(unique_xy) > 1

        trajectory_candidate = (
            len(unique_samples) >= 2
            and (stop_order_changed or stop_id_changed or xy_changed)
        )

        summaries.append({
            "route_id": route_id,
            "direction_id": direction_id,
            "bus_id_candidate": bus_id,
            "observation_count": len(g),
            "sample_count": len(unique_samples),
            "first_sample_index": min(unique_samples) if unique_samples else None,
            "last_sample_index": max(unique_samples) if unique_samples else None,
            "observed_sample_span": (max(unique_samples) - min(unique_samples) + 1) if unique_samples else 0,
            "first_event_time_raw": first.get("event_time_raw"),
            "last_event_time_raw": last.get("event_time_raw"),
            "unique_stop_order_count": len(set(str(x) for x in stop_orders)),
            "unique_stop_id_count": len(stop_ids),
            "min_stop_order": min_stop_order,
            "max_stop_order": max_stop_order,
            "first_stop_order": first_stop_order,
            "last_stop_order": last_stop_order,
            "stop_order_delta": stop_order_delta,
            "first_stop_id": first.get("current_stop_id"),
            "last_stop_id": last.get("current_stop_id"),
            "first_x_pos": first.get("x_pos"),
            "first_y_pos": first.get("y_pos"),
            "last_x_pos": last.get("x_pos"),
            "last_y_pos": last.get("y_pos"),
            "first_last_distance_m": round(first_last_distance_m, 3) if first_last_distance_m is not None else None,
            "incremental_distance_m": round(incremental_distance_m, 3) if incremental_valid else None,
            "xy_changed": bool(xy_changed),
            "stop_order_changed": bool(stop_order_changed),
            "stop_id_changed": bool(stop_id_changed),
            "trajectory_candidate": bool(trajectory_candidate),
            "classification_update_candidate": (
                "trajectory_candidate_from_repeated_getPos02_sampling"
                if trajectory_candidate
                else "repeated_vehicle_observed_without_movement_change"
            ),
        })

    summaries.sort(
        key=lambda x: (
            bool(x["trajectory_candidate"]),
            int(x["sample_count"]),
            int(x["unique_stop_order_count"]),
            abs(int(x["stop_order_delta"] or 0)),
        ),
        reverse=True,
    )
    return summaries


def write_reports(
    run_dir: Path,
    output_dir: Path,
    timeseries: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    parse_stats: Dict[str, Any],
    xy_threshold_m: float,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = output_dir / "getpos02_trajectory_candidate_timeseries.csv"
    summary_path = output_dir / "getpos02_trajectory_candidate_summary.csv"
    report_json_path = output_dir / "getpos02_trajectory_candidate_report.json"
    report_md_path = output_dir / "getpos02_trajectory_candidate_report.md"

    timeseries_fields = [
        "sample_index",
        "sample_tag",
        "sample_started_at",
        "sample_ended_at",
        "route_id",
        "route_no",
        "direction_id",
        "bus_id_candidate",
        "current_stop_order",
        "current_stop_id",
        "x_pos",
        "y_pos",
        "event_time_raw",
        "bus_tcd2",
        "bus_tcd3",
        "raw_path",
    ]

    summary_fields = [
        "route_id",
        "direction_id",
        "bus_id_candidate",
        "observation_count",
        "sample_count",
        "first_sample_index",
        "last_sample_index",
        "observed_sample_span",
        "first_event_time_raw",
        "last_event_time_raw",
        "unique_stop_order_count",
        "unique_stop_id_count",
        "min_stop_order",
        "max_stop_order",
        "first_stop_order",
        "last_stop_order",
        "stop_order_delta",
        "first_stop_id",
        "last_stop_id",
        "first_x_pos",
        "first_y_pos",
        "last_x_pos",
        "last_y_pos",
        "first_last_distance_m",
        "incremental_distance_m",
        "xy_changed",
        "stop_order_changed",
        "stop_id_changed",
        "trajectory_candidate",
        "classification_update_candidate",
    ]

    write_csv(timeseries_path, timeseries, timeseries_fields)
    write_csv(summary_path, summary_rows, summary_fields)

    repeated_vehicle_count = sum(1 for x in summary_rows if int(x["sample_count"]) >= 2)
    trajectory_candidate_count = sum(1 for x in summary_rows if bool(x["trajectory_candidate"]))

    route_count = len(set(x["route_id"] for x in timeseries if x.get("route_id")))
    vehicle_group_count = len(summary_rows)

    top_candidates = [x for x in summary_rows if bool(x["trajectory_candidate"])][:20]

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "xy_change_threshold_m": xy_threshold_m,
        "total_timeseries_rows": len(timeseries),
        "route_count_with_rows": route_count,
        "vehicle_group_count": vehicle_group_count,
        "repeated_vehicle_count": repeated_vehicle_count,
        "trajectory_candidate_count": trajectory_candidate_count,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "vehicle_trajectory_claim_allowed": False,
        "classification_update": {
            "bus_id_or_vehicle_no": "repeated_sample_observed_candidate",
            "live_position_xy": "repeated_sample_observed_candidate",
            "current_route_sequence": "repeated_sample_observed_candidate",
            "current_stop_id": "repeated_sample_observed_candidate",
            "vehicle_trajectory": (
                "trajectory_candidate_from_repeated_getPos02_sampling"
                if trajectory_candidate_count > 0
                else "trajectory_candidate_not_confirmed"
            ),
            "actual_headway": "not_observed",
            "actual_dwell": "not_observed",
            "actual_arrival_departure_time": "not_observed",
        },
        "parse_stats": parse_stats,
        "outputs": {
            "timeseries_csv": str(timeseries_path),
            "summary_csv": str(summary_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
        },
        "top_trajectory_candidates": top_candidates,
    }

    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Step 99-C3 - getPos02 Trajectory Candidate Analysis")
    lines.append("")
    lines.append(f"- artifact_version: {ARTIFACT_VERSION}")
    lines.append(f"- created_at_utc: {report['created_at_utc']}")
    lines.append(f"- run_dir: {run_dir}")
    lines.append(f"- total_timeseries_rows: {len(timeseries)}")
    lines.append(f"- route_count_with_rows: {route_count}")
    lines.append(f"- vehicle_group_count: {vehicle_group_count}")
    lines.append(f"- repeated_vehicle_count: {repeated_vehicle_count}")
    lines.append(f"- trajectory_candidate_count: {trajectory_candidate_count}")
    lines.append(f"- paper_level_claim_allowed: False")
    lines.append(f"- causal_performance_claim_allowed: False")
    lines.append(f"- vehicle_trajectory_claim_allowed: False")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This analysis parses the full raw getPos02 repeated-sampling responses, not the normalized_preview field. A trajectory candidate is counted only when the same route_id, direction_id, and vehicle identifier candidate appears in at least two samples and shows a stop-order, stop-id, or x/y position change.")
    lines.append("")
    lines.append("The result can upgrade vehicle_trajectory only to trajectory_candidate_from_repeated_getPos02_sampling. It does not justify actual vehicle trajectory, actual headway, dwell time, or arrival/departure claims.")
    lines.append("")
    lines.append("## Classification update candidates")
    lines.append("")
    for k, v in report["classification_update"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Top trajectory candidates")
    lines.append("")
    lines.append("| route_id | direction_id | bus_id | sample_count | first_seq | last_seq | delta_seq | distance_m |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for x in top_candidates[:20]:
        lines.append(
            f"| {x['route_id']} | {x['direction_id']} | {x['bus_id_candidate']} | "
            f"{x['sample_count']} | {x['first_stop_order']} | {x['last_stop_order']} | "
            f"{x['stop_order_delta']} | {x['first_last_distance_m']} |"
        )
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- DB write: forbidden and not performed")
    lines.append("- tensor DB overwrite: forbidden and not performed")
    lines.append("- paper-level claim: forbidden")
    lines.append("- causal performance claim: forbidden")
    lines.append("- actual headway, dwell, and arrival/departure claims: forbidden")

    report_md_path.write_text("\n".join(lines), encoding="utf-8")

    return report


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"[FAIL] run_dir not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    manifest = load_manifest(run_dir)

    raw_files = sorted(run_dir.glob("sample_*/getpos02_sample_response_route_*.raw"))
    if not raw_files:
        raise SystemExit(f"[FAIL] no raw files found under {run_dir}/sample_*")

    timeseries: List[Dict[str, Any]] = []
    parse_stats: Dict[str, Any] = {
        "raw_file_count": len(raw_files),
        "format_counts": {},
        "parse_error_count": 0,
        "empty_item_file_count": 0,
    }

    for raw_path in raw_files:
        sample_dir = raw_path.parent
        sample_index = extract_sample_index(sample_dir)
        sample_tag = sample_dir.name
        manifest_row = manifest.get(sample_index, {})

        fmt, items, parse_error = parse_raw_file(raw_path)
        parse_stats["format_counts"][fmt] = parse_stats["format_counts"].get(fmt, 0) + 1

        if parse_error:
            parse_stats["parse_error_count"] += 1
        if not items:
            parse_stats["empty_item_file_count"] += 1

        for item in items:
            normalized = normalize_raw_row(
                item,
                sample_index=sample_index,
                sample_tag=sample_tag,
                manifest_row=manifest_row,
                raw_path=raw_path,
            )
            if normalized:
                timeseries.append(normalized)

    summary_rows = make_summary(timeseries, xy_threshold_m=float(args.xy_change_threshold_m))

    report = write_reports(
        run_dir=run_dir,
        output_dir=output_dir,
        timeseries=timeseries,
        summary_rows=summary_rows,
        parse_stats=parse_stats,
        xy_threshold_m=float(args.xy_change_threshold_m),
    )

    print("[OK] Step 99-C3 getPos02 trajectory candidate analysis complete")
    print(f"[OK] run_dir                    : {run_dir}")
    print(f"[OK] raw_file_count             : {parse_stats['raw_file_count']}")
    print(f"[OK] total_timeseries_rows      : {report['total_timeseries_rows']}")
    print(f"[OK] repeated_vehicle_count     : {report['repeated_vehicle_count']}")
    print(f"[OK] trajectory_candidate_count : {report['trajectory_candidate_count']}")
    print(f"[OK] timeseries_csv             : {report['outputs']['timeseries_csv']}")
    print(f"[OK] summary_csv                : {report['outputs']['summary_csv']}")
    print(f"[OK] report_md                  : {report['outputs']['report_md']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--xy-change-threshold-m", type=float, default=5.0)
    args = parser.parse_args()

    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
