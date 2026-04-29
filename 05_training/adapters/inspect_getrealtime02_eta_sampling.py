from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ARTIFACT_VERSION = "daegu_bis_api_audit_step99d_getrealtime02_eta_sampling_v1"
DEFAULT_BASE_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getRealtime02"
DEFAULT_OUTPUT_DIR = "artifacts/daegu_bis_api_audit/getrealtime02_eta_sampling"

ALIASES = {
    "stop_id": ["bsId", "bs_id", "stopId", "stop_id", "stationId", "station_id", "nodeId", "node_id"],
    "route_id": ["routeId", "route_id", "routeID", "ROUTE_ID"],
    "route_no": ["routeNo", "route_no", "routeNm", "routeName", "route_name", "lineNo", "lineName"],
    "direction_id": ["moveDir", "move_dir", "direction", "direction_id", "dir", "dirCode", "way"],
    "eta_time_candidate": [
        "arrState", "arrivalState", "arrival_state", "arrTime", "arrivalTime", "arrival_time", "predictTime", "predictionTime",
        "remainTime", "remain_time", "expArrTime", "expectedArrivalTime", "eta", "etaTime",
        "arriveTime", "leftTime", "time"
    ],
    "eta_seconds_candidate": [
        "arrTime", "arrTimeSec", "arrivalTimeSec", "predictTimeSec", "predictSec", "remainSec",
        "remainTimeSec", "etaSec", "etaSeconds", "leftTimeSec", "seconds", "remainSecond"
    ],
    "remaining_stop_count_candidate": [
        "bsGap", "prevBsGap", "remainStop", "remain_stop", "leftStation", "left_station", "stationCount",
        "station_count", "restStop", "restStation", "arrPrevStationCnt", "leftStationCnt"
    ],
    "bus_id_candidate": [
        "vhcNo2", "vhcno2", "vhc_no2", "vhcNo", "vhcno", "vhc_no", "busNo", "bus_no",
        "vehicleNo", "vehicle_no", "vehicleId", "vehicle_id", "busId", "bus_id", "carNo"
    ],
    "route_sequence_candidate": ["seq", "sequence", "stopSeq", "stop_seq", "stationSeq", "station_seq", "bsSeq"],
    "bus_tcd2": ["busTCd2", "bus_tcd2", "busTypeCd2"],
    "bus_tcd3": ["busTCd3", "bus_tcd3", "busTypeCd3"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_key(value: Any) -> str:
    return str(value).strip().replace("_", "").replace("-", "").lower()


def first_value(row: Dict[str, Any], canonical: str) -> Any:
    if not isinstance(row, dict):
        return None
    normalized = {norm_key(k): v for k, v in row.items()}
    for alias in ALIASES[canonical]:
        key = norm_key(alias)
        if key in normalized:
            return normalized[key]
    return None


def clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def to_int(value: Any) -> Optional[int]:
    f = to_float(value)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def parse_eta_seconds_from_text(value: Any) -> Tuple[Optional[int], bool]:
    """Return (seconds, parsed_from_text). Conservative parser for common ETA text."""
    if value is None:
        return None, False
    text = str(value).strip()
    if not text:
        return None, False

    # Direct numeric means seconds only when the provider already uses a seconds-like field.
    # For eta_time_candidate we only parse explicit time-unit strings.
    h = re.search(r"(\d+)\s*(?:hour|hours|hr|hrs|시간|시)", text, re.IGNORECASE)
    m = re.search(r"(\d+)\s*(?:min|mins|minute|minutes|분)", text, re.IGNORECASE)
    s = re.search(r"(\d+)\s*(?:sec|secs|second|seconds|초)", text, re.IGNORECASE)
    if h or m or s:
        total = 0
        if h:
            total += int(h.group(1)) * 3600
        if m:
            total += int(m.group(1)) * 60
        if s:
            total += int(s.group(1))
        return total, True

    # HH:MM:SS or MM:SS duration-like values.
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
        hh, mm, ss = [int(x) for x in text.split(":")]
        return hh * 3600 + mm * 60 + ss, True
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        mm, ss = [int(x) for x in text.split(":")]
        return mm * 60 + ss, True

    return None, False


def normalize_stop_ids(raw_values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in raw_values:
        if raw is None:
            continue
        for part in str(raw).split(","):
            clean = part.strip().strip('"').strip("'")
            if clean and clean not in seen:
                out.append(clean)
                seen.add(clean)
    return out


def parse_extra_params(values: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values or []:
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"[FAIL] extra param must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"[FAIL] extra param key is empty: {item}")
        out[key] = value.strip()
    return out


def build_url(base_url: str, service_key: str, stop_param: str, stop_id: str, extra_params: Dict[str, str]) -> str:
    params: List[Tuple[str, str]] = [("serviceKey", service_key), (stop_param, stop_id)]
    for key, value in extra_params.items():
        params.append((key, value))
    encoded_parts = []
    for key, value in params:
        safe = "%" if key == "serviceKey" else ""
        encoded_parts.append(
            f"{urllib.parse.quote(str(key), safe='')}={urllib.parse.quote(str(value), safe=safe)}"
        )
    separator = "&" if "?" in base_url else "?"
    return base_url + separator + "&".join(encoded_parts)


def fetch_api(base_url: str, service_key: str, stop_param: str, stop_id: str, extra_params: Dict[str, str], timeout: int = 30) -> Tuple[int, str, str]:
    url = build_url(base_url, service_key, stop_param, stop_id, extra_params)
    req = urllib.request.Request(url, headers={"accept": "*/*", "User-Agent": "urbanbus-step99d-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            body = resp.read().decode("utf-8", errors="replace")
            return status, body, url
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        return int(e.code), body, url
    except Exception as e:
        return 0, str(e), url


def flatten_xml_element(elem: ET.Element) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for child in list(elem):
        if len(list(child)) == 0:
            tag = child.tag.split("}")[-1]
            row[tag] = (child.text or "").strip()
    return row


def parse_xml_items(raw_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(raw_text)
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag in {"item", "items", "row"}:
            row = flatten_xml_element(elem)
            if row:
                rows.append(row)
    if rows:
        return rows
    # Fallback: collect leaf parent nodes containing route/ETA-like keys.
    for elem in root.iter():
        row = flatten_xml_element(elem)
        if row and any(norm_key(k) in {"routeid", "routeno", "arrtime", "predicttime", "remainsec", "vhcno2"} for k in row):
            rows.append(row)
    return rows


def parse_json_items(obj: Any) -> List[Dict[str, Any]]:
    """Parse getRealtime02 JSON.

    The Daegu getRealtime02 response can be nested as:
      body.items[] -> route bucket -> arrList[] -> ETA rows

    This function flattens each arrList row while preserving parent route fields.
    """
    rows: List[Dict[str, Any]] = []

    def eta_like(d: Dict[str, Any]) -> bool:
        keys = {norm_key(k) for k in d.keys()}
        return bool(
            keys
            & {
                "routeid",
                "routeno",
                "movedir",
                "arrtime",
                "arrstate",
                "predicttime",
                "remainsec",
                "vhcno2",
                "bsgap",
            }
        )

    def add_arr_list_rows(d: Dict[str, Any]) -> bool:
        added = False
        arr = None
        for key in ("arrList", "arrivalList", "arrival_list", "arrivals"):
            if key in d:
                arr = d.get(key)
                break

        if isinstance(arr, list):
            parent = {k: v for k, v in d.items() if k not in {"arrList", "arrivalList", "arrival_list", "arrivals"}}
            for child in arr:
                if isinstance(child, dict):
                    merged = dict(parent)
                    merged.update(child)
                    rows.append(merged)
                    added = True
            return added

        if isinstance(arr, dict):
            parent = {k: v for k, v in d.items() if k not in {"arrList", "arrivalList", "arrival_list", "arrivals"}}
            merged = dict(parent)
            merged.update(arr)
            rows.append(merged)
            return True

        return False

    def path_get(root: Any, path: List[str]) -> Any:
        cur = root
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                if not add_arr_list_rows(x) and eta_like(x):
                    rows.append(x)
        return rows

    if not isinstance(obj, dict):
        return []

    candidate_paths = [
        ["body", "items"],
        ["body", "item"],
        ["response", "body", "items"],
        ["response", "body", "item"],
        ["items"],
        ["item"],
        ["data"],
        ["result"],
    ]

    for p in candidate_paths:
        cur = path_get(obj, p)
        if cur is None:
            continue

        if isinstance(cur, list):
            before = len(rows)
            for x in cur:
                if isinstance(x, dict):
                    if not add_arr_list_rows(x) and eta_like(x):
                        rows.append(x)
            if len(rows) > before:
                return rows

        if isinstance(cur, dict):
            before = len(rows)
            if add_arr_list_rows(cur):
                return rows
            for sub in ("items", "item", "list", "rows"):
                val = cur.get(sub)
                if isinstance(val, list):
                    for x in val:
                        if isinstance(x, dict):
                            if not add_arr_list_rows(x) and eta_like(x):
                                rows.append(x)
                elif isinstance(val, dict):
                    if not add_arr_list_rows(val) and eta_like(val):
                        rows.append(val)
            if len(rows) > before:
                return rows
            if eta_like(cur):
                return [cur]

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if add_arr_list_rows(x):
                return
            if eta_like(x):
                rows.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return rows


def parse_csv_items(raw_text: str) -> List[Dict[str, Any]]:
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if not lines or "," not in lines[0]:
        return []
    try:
        reader = csv.DictReader(lines)
        return [dict(row) for row in reader]
    except Exception:
        return []


def infer_format_and_items(raw_text: str) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    text = (raw_text or "").strip()
    if not text:
        return "empty", [], None

    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
        except Exception as e:
            return "json_error", [], {"parse_error": str(e)}
        return "json", parse_json_items(obj), obj if isinstance(obj, dict) else None

    if text.startswith("<"):
        return "xml", parse_xml_items(text), None

    if "," in text.splitlines()[0]:
        return "csv", parse_csv_items(text), None

    return "unknown", [], None


def classify_provider_status(http_status: int, raw_text: str, meta_obj: Optional[Dict[str, Any]]) -> str:
    text = (raw_text or "").lower()
    if http_status == 0:
        return "request_error"
    if http_status >= 400:
        return "http_error"
    header = meta_obj.get("header") if isinstance(meta_obj, dict) else None
    if isinstance(header, dict):
        success = header.get("success")
        result_code = str(header.get("resultCode", "")).strip()
        result_msg = str(header.get("resultMsg", "")).strip()
        if success is False or result_code in {"9003", "30", "99"}:
            if "인증" in result_msg or "auth" in result_msg.lower() or result_code == "9003":
                return "auth_or_parameter_error"
            return "provider_error"
        if success is True or result_code in {"0000", "0"}:
            return "ok_or_unclassified"
    if "auth" in text or "인증" in text or "servicekey" in text and "error" in text:
        return "auth_or_parameter_error"
    if "error" in text or "오류" in text:
        return "provider_error"
    return "ok_or_unclassified"


def normalize_item(item: Dict[str, Any], requested_stop_id: str, raw_path: Path) -> Optional[Dict[str, Any]]:
    stop_id = clean_str(first_value(item, "stop_id")) or requested_stop_id
    route_id = clean_str(first_value(item, "route_id"))
    route_no = clean_str(first_value(item, "route_no"))
    direction_id = clean_str(first_value(item, "direction_id"))

    eta_seconds = to_int(first_value(item, "eta_seconds_candidate"))
    eta_time = clean_str(first_value(item, "eta_time_candidate"))
    parsed_from_text = False
    if eta_seconds is None and eta_time:
        eta_seconds, parsed_from_text = parse_eta_seconds_from_text(eta_time)

    row = {
        "stop_id": stop_id,
        "requested_stop_id": requested_stop_id,
        "route_id": route_id,
        "route_no": route_no,
        "direction_id": direction_id,
        "eta_time_candidate": eta_time,
        "eta_seconds_candidate": eta_seconds,
        "eta_seconds_parsed_from_text": parsed_from_text,
        "remaining_stop_count_candidate": to_int(first_value(item, "remaining_stop_count_candidate")),
        "bus_id_candidate": clean_str(first_value(item, "bus_id_candidate")),
        "route_sequence_candidate": to_int(first_value(item, "route_sequence_candidate")),
        "bus_tcd2": clean_str(first_value(item, "bus_tcd2")),
        "bus_tcd3": clean_str(first_value(item, "bus_tcd3")),
        "raw_path": str(raw_path),
    }

    if not any([row["route_id"], row["route_no"], row["eta_time_candidate"], row["eta_seconds_candidate"], row["bus_id_candidate"]]):
        return None
    return row


def safe_filename_part(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_headway_candidates(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    parsing_not_available = 0
    for row in rows:
        key = (str(row.get("stop_id") or ""), str(row.get("route_id") or row.get("route_no") or ""), str(row.get("direction_id") or ""))
        if not key[0] or not key[1]:
            continue
        if row.get("eta_seconds_candidate") is None:
            if row.get("eta_time_candidate"):
                parsing_not_available += 1
            continue
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    for (stop_id, route_key, direction_id), g in groups.items():
        if len(g) < 2:
            continue
        sorted_g = sorted(g, key=lambda x: int(x["eta_seconds_candidate"]))
        for idx in range(len(sorted_g) - 1):
            a = sorted_g[idx]
            b = sorted_g[idx + 1]
            delta = int(b["eta_seconds_candidate"]) - int(a["eta_seconds_candidate"])
            out.append({
                "stop_id": stop_id,
                "route_id": a.get("route_id") or b.get("route_id"),
                "route_no": a.get("route_no") or b.get("route_no"),
                "direction_id": direction_id,
                "first_eta_seconds": a.get("eta_seconds_candidate"),
                "second_eta_seconds": b.get("eta_seconds_candidate"),
                "eta_headway_seconds_candidate": delta,
                "first_bus_id_candidate": a.get("bus_id_candidate"),
                "second_bus_id_candidate": b.get("bus_id_candidate"),
                "candidate_type": "eta_based_headway_candidate",
            })
    return out, parsing_not_available


def make_self_test_payload() -> Dict[str, Any]:
    return {
        "header": {"success": True, "resultCode": "0000", "resultMsg": "성공"},
        "body": {
            "items": [
                {
                    "bsId": "S_TEST",
                    "routeId": "R_TEST",
                    "routeNo": "T1",
                    "moveDir": "1",
                    "predictSec": 120,
                    "remainStop": 2,
                    "vhcNo2": "BUS_A",
                    "seq": 10,
                    "busTCd2": "N",
                    "busTCd3": "N",
                },
                {
                    "bsId": "S_TEST",
                    "routeId": "R_TEST",
                    "routeNo": "T1",
                    "moveDir": "1",
                    "predictSec": 420,
                    "remainStop": 5,
                    "vhcNo2": "BUS_B",
                    "seq": 13,
                    "busTCd2": "N",
                    "busTCd3": "N",
                },
            ]
        },
    }


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stop_ids = normalize_stop_ids(args.stop_id or [])
    if args.self_test and not stop_ids:
        stop_ids = ["S_TEST"]
    if args.max_stops and len(stop_ids) > args.max_stops:
        stop_ids = stop_ids[: args.max_stops]
    if not stop_ids:
        raise SystemExit("[FAIL] at least one --stop-id is required unless --self-test is used")

    extra_params = parse_extra_params(args.extra_param or [])
    service_key = args.service_key or os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY") or ""
    api_called = False
    if not service_key and not args.self_test:
        print("[WARN] service key not found; switching to self-test schema mode")
        args.self_test = True

    normalized_rows: List[Dict[str, Any]] = []
    stop_reports: List[Dict[str, Any]] = []
    response_format_counts: Dict[str, int] = defaultdict(int)
    provider_status_counts: Dict[str, int] = defaultdict(int)

    for stop_id in stop_ids:
        raw_path = output_dir / f"getrealtime02_sample_response_stop_{safe_filename_part(stop_id)}.raw"
        request_url = ""
        http_status = None
        if args.self_test:
            raw_text = json.dumps(make_self_test_payload(), ensure_ascii=False)
            http_status = 200
            request_url = "self-test://getRealtime02"
        else:
            api_called = True
            http_status, raw_text, request_url = fetch_api(
                args.base_url,
                service_key,
                args.stop_param,
                stop_id,
                extra_params,
                timeout=int(args.timeout),
            )
            if args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))

        raw_path.write_text(raw_text, encoding="utf-8")
        fmt, items, meta_obj = infer_format_and_items(raw_text)
        provider_status = classify_provider_status(int(http_status or 0), raw_text, meta_obj)
        response_format_counts[fmt] += 1
        provider_status_counts[provider_status] += 1

        rows_for_stop = []
        for item in items:
            norm = normalize_item(item, requested_stop_id=stop_id, raw_path=raw_path)
            if norm:
                rows_for_stop.append(norm)
        normalized_rows.extend(rows_for_stop)

        stop_reports.append({
            "stop_id": stop_id,
            "http_status": http_status,
            "provider_status": provider_status,
            "response_format": fmt,
            "candidate_row_count": len(items),
            "normalized_row_count": len(rows_for_stop),
            "raw_path": str(raw_path),
            "request_url_redacted": request_url.replace(service_key, "<REDACTED>") if service_key else request_url,
        })

    headway_rows, parsing_not_available_count = build_headway_candidates(normalized_rows)

    normalized_fields = [
        "stop_id", "requested_stop_id", "route_id", "route_no", "direction_id",
        "eta_time_candidate", "eta_seconds_candidate", "eta_seconds_parsed_from_text",
        "remaining_stop_count_candidate", "bus_id_candidate", "route_sequence_candidate",
        "bus_tcd2", "bus_tcd3", "raw_path",
    ]
    headway_fields = [
        "stop_id", "route_id", "route_no", "direction_id", "first_eta_seconds", "second_eta_seconds",
        "eta_headway_seconds_candidate", "first_bus_id_candidate", "second_bus_id_candidate", "candidate_type",
    ]

    normalized_path = output_dir / "getrealtime02_eta_normalized.csv"
    headway_path = output_dir / "getrealtime02_headway_candidate.csv"
    report_json_path = output_dir / "getrealtime02_eta_sampling_report.json"
    report_md_path = output_dir / "getrealtime02_eta_sampling_report.md"

    write_csv(normalized_path, normalized_rows, normalized_fields)
    write_csv(headway_path, headway_rows, headway_fields)

    eta_rows_exist = len(normalized_rows) > 0
    headway_exists = len(headway_rows) > 0
    provider_error_count = sum(v for k, v in provider_status_counts.items() if k not in {"ok_or_unclassified"})

    if eta_rows_exist:
        audit_status = "PASS"
    elif provider_error_count > 0:
        audit_status = "REVIEW_REQUIRED"
    else:
        audit_status = "NO_ITEMS_OR_NO_ARRIVAL_INFO"

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "mode": "self_test" if args.self_test else "sample_api_call",
        "audit_status": audit_status,
        "api_called": api_called,
        "base_url": args.base_url,
        "stop_param": args.stop_param,
        "stops_requested": stop_ids,
        "response_format_counts": dict(response_format_counts),
        "provider_status_counts": dict(provider_status_counts),
        "total_candidate_rows": int(sum(x["candidate_row_count"] for x in stop_reports)),
        "total_normalized_rows": int(len(normalized_rows)),
        "headway_candidate_count": int(len(headway_rows)),
        "eta_time_parsing_not_available_count": int(parsing_not_available_count),
        "classification_update_candidates": {
            "getRealtime02_eta": "observed_candidate" if eta_rows_exist else "not_confirmed",
            "eta_based_headway": "candidate_from_getRealtime02" if headway_exists else "not_confirmed",
            "actual_headway": "not_observed",
            "actual_arrival_departure_time": "not_observed",
            "actual_dwell": "not_observed",
        },
        "guardrails": {
            "db_write_performed": False,
            "tensor_db_overwrite_performed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "actual_headway_observed": False,
            "actual_arrival_departure_time_observed": False,
            "actual_dwell_observed": False,
        },
        "stop_reports": stop_reports,
        "outputs": {
            "normalized_csv": str(normalized_path),
            "headway_candidate_csv": str(headway_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
        },
    }

    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Step 99-D - getRealtime02 ETA Sampling Audit")
    lines.append("")
    lines.append(f"- artifact_version: {ARTIFACT_VERSION}")
    lines.append(f"- created_at_utc: {report['created_at_utc']}")
    lines.append(f"- mode: {report['mode']}")
    lines.append(f"- audit_status: {audit_status}")
    lines.append(f"- api_called: {api_called}")
    lines.append(f"- base_url: {args.base_url}")
    lines.append(f"- stop_param: {args.stop_param}")
    lines.append(f"- stops_requested: {stop_ids}")
    lines.append(f"- total_candidate_rows: {report['total_candidate_rows']}")
    lines.append(f"- total_normalized_rows: {len(normalized_rows)}")
    lines.append(f"- headway_candidate_count: {len(headway_rows)}")
    lines.append(f"- eta_time_parsing_not_available_count: {parsing_not_available_count}")
    lines.append(f"- paper_level_claim_allowed: False")
    lines.append(f"- causal_performance_claim_allowed: False")
    lines.append("")
    lines.append("## Classification update candidates")
    lines.append("")
    for key, value in report["classification_update_candidates"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Stop reports")
    lines.append("")
    lines.append("| stop_id | provider_status | format | candidate_rows | normalized_rows |")
    lines.append("|---|---|---|---:|---:|")
    for item in stop_reports:
        lines.append(
            f"| {item['stop_id']} | {item['provider_status']} | {item['response_format']} | "
            f"{item['candidate_row_count']} | {item['normalized_row_count']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("ETA rows can support getRealtime02_eta observed_candidate status. Multiple ETA rows for the same stop_id, route_id, and direction_id can support eta_based_headway candidate status. This is not an actual observed headway, actual arrival/departure, or dwell-time claim.")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- DB write: forbidden and not performed")
    lines.append("- tensor DB overwrite: forbidden and not performed")
    lines.append("- actual_headway: not observed")
    lines.append("- actual_arrival_departure_time: not observed")
    lines.append("- actual_dwell: not observed")

    report_md_path.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Step 99-D /getRealtime02 ETA sampling audit complete")
    print(f"[OK] mode       : {report['mode']}")
    print(f"[OK] status     : {audit_status}")
    print(f"[OK] api_called : {api_called}")
    print(f"[OK] normalized : {len(normalized_rows)}")
    print(f"[OK] headway    : {len(headway_rows)}")
    print(f"[OK] json report: {report_json_path}")
    print(f"[OK] md report  : {report_md_path}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 99-D /getRealtime02 ETA sampling audit")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--stop-param", default="bsId")
    parser.add_argument("--stop-id", action="append", default=[])
    parser.add_argument("--max-stops", type=int, default=10)
    parser.add_argument("--service-key", default="")
    parser.add_argument("--extra-param", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_audit(args)
    if args.self_test:
        if report["total_normalized_rows"] < 2:
            raise SystemExit("[FAIL] self-test expected at least two normalized ETA rows")
        if report["headway_candidate_count"] < 1:
            raise SystemExit("[FAIL] self-test expected at least one headway candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
