from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence
import xml.etree.ElementTree as ET


DEFAULT_BASE_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
ARTIFACT_VERSION = "daegu_bis_api_audit_step99c_getpos02_v1"


ROUTE_ID_ALIASES = {"routeId", "route_id", "ROUTE_ID", "routeid"}
DIRECTION_ALIASES = {"moveDir", "movedir", "move_dir", "MOVE_DIR", "direction_id", "dir", "dirCode"}
BUS_ID_ALIASES = {"busId", "bus_id", "BUS_ID", "vhcNo", "vhcno", "vhcNo2", "vhcno2", "VHCNO2", "vhc_no", "vhc_no2", "VHC_NO", "VHC_NO2", "vehicleNo", "vehicle_no", "vehicleId", "vehicle_id", "vehNo", "veh_no", "carNo", "car_no", "busNo", "bus_no", "plateNo", "plate_no", "vhNo", "vh_no"}
ROUTE_NO_ALIASES = {"routeNo", "route_no", "routeNm", "routeName"}
STOP_ID_ALIASES = {"bsId", "bs_id", "stopId", "stop_id", "nodeId"}
STOP_ORDER_ALIASES = {"seq", "sequence", "stopSeq", "stop_order", "bsSeq", "ord"}
X_ALIASES = {"xPos", "xpos", "x", "longitude", "lon", "lng"}
Y_ALIASES = {"yPos", "ypos", "y", "latitude", "lat"}
TIME_ALIASES = {"arTime", "arrTime", "arrivalTime", "eventTime", "tm", "time"}


SELF_TEST_RAW = {
    "header": {
        "success": True,
        "resultCode": "0000",
        "resultMsg": "성공"
    },
    "body": {
        "items": [
            {
                "routeId": "1000005000",
                "moveDir": "1",
                "arTime": "233921",
                "seq": 26,
                "bsId": "7031007900",
                "xPos": 128.55891885,
                "yPos": 35.8712853,
                "vhcNo": "1136",
                "routeNo": "급행5",
                "busTCd2": "N",
                "busTCd3": "N"
            }
        ]
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_first(row: Dict[str, Any], aliases: set[str]) -> Any:
    for k in row.keys():
        if k in aliases:
            return row.get(k)
    lower = {str(k).lower(): k for k in row.keys()}
    for a in aliases:
        key = lower.get(str(a).lower())
        if key is not None:
            return row.get(key)
    return None


def as_int_or_none(v: Any) -> int | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return int(float(str(v).strip()))
    except Exception:
        return None


def as_float_or_none(v: Any) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(str(v).strip())
    except Exception:
        return None


def detect_provider_status(raw_text: str) -> str:
    s = (raw_text or "").strip()
    low = s.lower()

    if not s:
        return "empty_response"

    if any(x in low for x in ["unexpected errors", "service error", "application error", "internal server error"]):
        return "provider_error"

    if any(x in low for x in ["servicekey", "service key", "인증키", "등록되지 않은", "invalid", "unauthorized"]):
        return "auth_error"

    return "ok_or_unclassified"


def build_url(base_url: str, service_key: str, route_id: str, route_param: str, extra_params: Sequence[str]) -> str:
    params: Dict[str, str] = {
        "serviceKey": service_key,
        route_param: str(route_id),
    }

    for item in extra_params or []:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"extra param must be KEY=VALUE: {item}")
        k, v = item.split("=", 1)
        params[k] = v

    return base_url + "?" + urllib.parse.urlencode(params, safe="%")


def fetch_api(base_url: str, service_key: str, route_id: str, route_param: str, extra_params: Sequence[str], timeout: int) -> Dict[str, Any]:
    url = build_url(base_url, service_key, route_id, route_param, extra_params)
    redacted_url = url.replace(urllib.parse.quote(service_key, safe=""), "<SERVICE_KEY>").replace(service_key, "<SERVICE_KEY>")
    req = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-step99c-audit/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = int(getattr(resp, "status", 200) or 200)
            raw_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            encoding = resp.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as e:
        status_code = int(e.code)
        raw_bytes = e.read() or b""
        content_type = e.headers.get("Content-Type", "") if e.headers else ""
        encoding = e.headers.get_content_charset() if e.headers else "utf-8"
    except urllib.error.URLError as e:
        status_code = None
        raw_bytes = str(e).encode("utf-8", errors="replace")
        content_type = "text/plain"
        encoding = "utf-8"

    raw_text = raw_bytes.decode(encoding or "utf-8", errors="replace")
    return {
        "route_id": route_id,
        "request_url_redacted": redacted_url,
        "http_status": status_code,
        "content_type": content_type,
        "raw_text": raw_text,
        "provider_status": detect_provider_status(raw_text),
    }


def infer_format(raw_text: str) -> str:
    s = (raw_text or "").lstrip()
    if s.startswith("{") or s.startswith("["):
        return "json"
    if s.startswith("<"):
        return "xml"
    return "unknown"


def extract_rows_from_json(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if not isinstance(obj, dict):
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
        if ok:
            if isinstance(cur, list):
                return [x for x in cur if isinstance(x, dict)]
            if isinstance(cur, dict):
                return [cur]

    rows: List[Dict[str, Any]] = []
    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if any(k in x for k in ["routeId", "vhcNo", "xPos", "yPos", "bsId"]):
                rows.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return rows


def extract_rows_from_xml(raw_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(raw_text)
    rows: List[Dict[str, Any]] = []
    for item in root.findall(".//item"):
        row = {child.tag: child.text for child in list(item)}
        if row:
            rows.append(row)
    return rows


def parse_raw(raw_text: str) -> tuple[str, List[Dict[str, Any]], str | None]:
    fmt = infer_format(raw_text)
    try:
        if fmt == "json":
            return fmt, extract_rows_from_json(json.loads(raw_text)), None
        if fmt == "xml":
            return fmt, extract_rows_from_xml(raw_text), None
        return fmt, [], "unknown response format"
    except Exception as e:
        return fmt, [], str(e)


def normalize_row(request_route_id: str, row: Dict[str, Any]) -> Dict[str, Any] | None:
    route_id = get_first(row, ROUTE_ID_ALIASES) or request_route_id
    direction_id = get_first(row, DIRECTION_ALIASES)
    bus_id = get_first(row, BUS_ID_ALIASES)
    stop_id = get_first(row, STOP_ID_ALIASES)
    stop_order = as_int_or_none(get_first(row, STOP_ORDER_ALIASES))
    x_pos = as_float_or_none(get_first(row, X_ALIASES))
    y_pos = as_float_or_none(get_first(row, Y_ALIASES))
    event_time = get_first(row, TIME_ALIASES)
    route_no = get_first(row, ROUTE_NO_ALIASES)

    # Live position row must have at least vehicle or position/sequence evidence.
    if bus_id is None and x_pos is None and y_pos is None and stop_order is None:
        return None

    return {
        "route_id": str(route_id) if route_id is not None else None,
        "route_no": str(route_no) if route_no is not None else None,
        "direction_id": str(direction_id) if direction_id is not None else None,
        "bus_id_candidate": str(bus_id) if bus_id is not None else None,
        "current_stop_id": str(stop_id) if stop_id is not None else None,
        "current_stop_order": stop_order,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "event_time_raw": str(event_time) if event_time is not None else None,
        "raw_keys": sorted([str(k) for k in row.keys()]),
    }


def analyze_route(raw: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    route_id = raw["route_id"]
    raw_path = output_dir / f"getpos02_sample_response_route_{route_id}.raw"
    raw_path.write_text(raw["raw_text"], encoding="utf-8")

    fmt, rows, parse_error = parse_raw(raw["raw_text"])
    normalized = []
    for row in rows:
        nr = normalize_row(route_id, row)
        if nr is not None:
            normalized.append(nr)

    schema_hits = {
        "route_id": sorted({k for row in rows for k in row.keys() if k in ROUTE_ID_ALIASES}),
        "direction_id": sorted({k for row in rows for k in row.keys() if k in DIRECTION_ALIASES}),
        "bus_id": sorted({k for row in rows for k in row.keys() if k in BUS_ID_ALIASES}),
        "route_no": sorted({k for row in rows for k in row.keys() if k in ROUTE_NO_ALIASES}),
        "stop_id": sorted({k for row in rows for k in row.keys() if k in STOP_ID_ALIASES}),
        "stop_order": sorted({k for row in rows for k in row.keys() if k in STOP_ORDER_ALIASES}),
        "x_pos": sorted({k for row in rows for k in row.keys() if k in X_ALIASES}),
        "y_pos": sorted({k for row in rows for k in row.keys() if k in Y_ALIASES}),
        "event_time": sorted({k for row in rows for k in row.keys() if k in TIME_ALIASES}),
    }

    return {
        "route_id": route_id,
        "request_url_redacted": raw.get("request_url_redacted"),
        "http_status": raw.get("http_status"),
        "provider_status": raw.get("provider_status"),
        "content_type": raw.get("content_type"),
        "raw_path": str(raw_path),
        "format": fmt,
        "parse_error": parse_error,
        "candidate_row_count": len(rows),
        "normalized_row_count": len(normalized),
        "schema_hits": schema_hits,
        "has_route_id": bool(schema_hits["route_id"]) or bool(route_id),
        "has_direction_id": bool(schema_hits["direction_id"]),
        "has_bus_id": bool(schema_hits["bus_id"]) or any(x.get("bus_id_candidate") for x in normalized),
        "has_position_xy": bool(schema_hits["x_pos"]) and bool(schema_hits["y_pos"]),
        "has_stop_sequence": bool(schema_hits["stop_order"]),
        "has_current_stop_id": bool(schema_hits["stop_id"]),
        "normalized_preview": normalized[:10],
    }


def write_reports(report: Dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "getpos02_live_position_sampling_report.json"
    md_path = output_dir / "getpos02_live_position_sampling_report.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Step 99-C — /getPos02 Live Bus Position Sampling Audit")
    lines.append("")
    lines.append(f"- artifact_version: `{report['artifact_version']}`")
    lines.append(f"- created_at_utc: `{report['created_at_utc']}`")
    lines.append(f"- mode: `{report['mode']}`")
    lines.append(f"- audit_status: `{report['audit_status']}`")
    lines.append(f"- api_called: `{report['api_call_summary']['api_called']}`")
    lines.append(f"- routes_requested: `{report['api_call_summary']['routes_requested']}`")
    lines.append(f"- paper_level_claim_allowed: `{report['paper_level_claim_allowed']}`")
    lines.append(f"- causal_performance_claim_allowed: `{report['causal_performance_claim_allowed']}`")
    lines.append("")
    lines.append("## API response summary")
    lines.append("")
    lines.append(f"- response_format_counts: `{report['response_format_counts']}`")
    lines.append(f"- provider_status_counts: `{report['provider_status_counts']}`")
    lines.append(f"- total_candidate_rows: `{report['schema_summary']['total_candidate_rows']}`")
    lines.append(f"- total_normalized_rows: `{report['schema_summary']['total_normalized_rows']}`")
    lines.append(f"- live_position_possible_any: `{report['schema_summary']['live_position_possible_any']}`")
    lines.append("")
    lines.append("## Field classification update candidates")
    lines.append("")
    lines.append("| Field | Candidate status | Reason |")
    lines.append("|---|---|---|")
    for row in report["field_classification_update_candidates"]:
        lines.append(f"| `{row['field']}` | `{row['candidate_status']}` | {row['reason']} |")
    lines.append("")
    lines.append("## Route reports")
    lines.append("")
    lines.append("| route_id | provider_status | candidate_rows | normalized_rows | bus_id | xy | stop_seq |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for rr in report["route_reports"]:
        lines.append(
            f"| `{rr['route_id']}` | `{rr['provider_status']}` | "
            f"{rr['candidate_row_count']} | {rr['normalized_row_count']} | "
            f"{rr['has_bus_id']} | {rr['has_position_xy']} | {rr['has_stop_sequence']} |"
        )
    lines.append("")
    lines.append("## Guardrail status")
    lines.append("")
    lines.append("- DB write: forbidden and not performed")
    lines.append("- tensor DB overwrite: forbidden and not performed")
    lines.append("- collection scope: live sampling audit only")
    lines.append("- paper-level claim: forbidden")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    service_key = args.service_key or os.getenv("DAEGU_BIS_SERVICE_KEY") or os.getenv("DATAGO_SERVICE_KEY")

    route_ids = list(args.route_id or [])
    if args.max_routes and args.max_routes > 0:
        route_ids = route_ids[: args.max_routes]

    raw_records: List[Dict[str, Any]] = []
    mode = "self_test_or_schema_only"

    if service_key and route_ids:
        mode = "sample_api_call"
        for rid in route_ids:
            raw_records.append(fetch_api(args.base_url, service_key, str(rid), args.route_param, args.extra_param, args.timeout))
    else:
        raw_records = [{
            "route_id": "SELF_TEST_ROUTE",
            "request_url_redacted": None,
            "http_status": 200,
            "provider_status": "ok_or_unclassified",
            "content_type": "application/json",
            "raw_text": json.dumps(SELF_TEST_RAW, ensure_ascii=False),
        }]

    route_reports = [analyze_route(raw, output_dir) for raw in raw_records]
    total_candidate = sum(int(r["candidate_row_count"]) for r in route_reports)
    total_normalized = sum(int(r["normalized_row_count"]) for r in route_reports)

    provider_counts: Dict[str, int] = {}
    format_counts: Dict[str, int] = {}
    for rr in route_reports:
        provider_counts[rr["provider_status"]] = provider_counts.get(rr["provider_status"], 0) + 1
        format_counts[rr["format"]] = format_counts.get(rr["format"], 0) + 1

    live_possible = any(
        rr["has_bus_id"] and rr["has_position_xy"] for rr in route_reports
    )

    def status(field: str, ok: bool, reason_ok: str, reason_no: str) -> Dict[str, str]:
        return {
            "field": field,
            "candidate_status": "observed_candidate" if ok else "missing_or_unparsed",
            "reason": reason_ok if ok else reason_no,
        }

    has_bus = any(rr["has_bus_id"] for rr in route_reports)
    has_xy = any(rr["has_position_xy"] for rr in route_reports)
    has_seq = any(rr["has_stop_sequence"] for rr in route_reports)
    has_stop = any(rr["has_current_stop_id"] for rr in route_reports)
    has_dir = any(rr["has_direction_id"] for rr in route_reports)
    has_time = any(bool(rr["schema_hits"]["event_time"]) for rr in route_reports)

    audit_status = "PASS" if total_normalized > 0 and live_possible else "REVIEW_REQUIRED"

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "step": "Step 99-C",
        "title": "/getPos02 live bus position sampling audit",
        "mode": mode,
        "audit_status": audit_status,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "api_call_summary": {
            "api_called": bool(service_key and route_ids),
            "base_url": args.base_url,
            "route_param": args.route_param,
            "routes_requested": route_ids,
            "service_key_source": "arg/env" if service_key else None,
            "daily_limit_guard": "sample-only live sampling; no bulk collection",
        },
        "response_format_counts": format_counts,
        "provider_status_counts": provider_counts,
        "schema_summary": {
            "total_candidate_rows": total_candidate,
            "total_normalized_rows": total_normalized,
            "live_position_possible_any": live_possible,
        },
        "field_classification_update_candidates": [
            status("bus_id_or_vehicle_no", has_bus, "vehicle/bus id-like field detected, e.g. vhcNo", "no vehicle id-like field detected"),
            status("live_position_xy", has_xy, "xPos/yPos-like fields detected", "no x/y position fields detected"),
            status("current_route_sequence", has_seq, "seq-like field detected", "no sequence-like field detected"),
            status("current_stop_id", has_stop, "bsId/stop_id-like field detected", "no current stop id-like field detected"),
            status("direction_id", has_dir, "moveDir-like field detected", "no direction field detected"),
            status("live_event_time_raw", has_time, "arTime/time-like field detected", "no event time-like field detected"),
        ],
        "route_reports": route_reports,
        "guardrails": {
            "db_write_forbidden": True,
            "tensor_db_overwrite_forbidden": True,
            "bulk_api_collection_forbidden": True,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }

    write_reports(report, output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--route-param", default="routeId")
    parser.add_argument("--route-id", action="append", default=[])
    parser.add_argument("--max-routes", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts/daegu_bis_api_audit/getpos02_live_position_sampling")
    parser.add_argument("--service-key", default="")
    parser.add_argument("--extra-param", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    report = build_report(args)

    output_dir = Path(args.output_dir)
    print("[OK] Step 99-C /getPos02 live position sampling audit complete")
    print(f"[OK] mode       : {report['mode']}")
    print(f"[OK] status     : {report['audit_status']}")
    print(f"[OK] api_called : {report['api_call_summary']['api_called']}")
    print(f"[OK] normalized : {report['schema_summary']['total_normalized_rows']}")
    print(f"[OK] json report: {output_dir / 'getpos02_live_position_sampling_report.json'}")
    print(f"[OK] md report  : {output_dir / 'getpos02_live_position_sampling_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
