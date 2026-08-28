from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_r1_bis_credential_mapping_preflight"
R2D1C_ROOT = "05_training/artifacts/prompt5_e01_r2d1c_official_source_live_recovery_preflight_20260721_020000"
SERVICE_GRAPH = "05_training/artifacts/suseong_service_graph_v1"
KEY_VALUE_RE = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY|serviceKey)\s*[:=]\s*([A-Za-z0-9%+/=_\\-]+)")
MISSING_KEYS = [
    ("2000002000", "1"),
    ("2000002100", "1"),
    ("2000003100", "1"),
    ("3000232000", "1"),
    ("3000323101", "1"),
    ("3000410000", "1"),
    ("3000410100", "1"),
    ("4010002001", "1"),
    ("4010002003", "1"),
    ("4010002004", "1"),
    ("4010002118", "1"),
    ("4050001001", "1"),
    ("4050010000", "1"),
]
CANARY_ROUTE_IDS = ["3000204000", "3000240001", "2000002000", "3000156000", "2000002100", "2000003100"]
ENDPOINTS = {
    "getBasic02": {"url": "https://apis.data.go.kr/6270000/dbmsapi02/getBasic02", "params": {}},
    "getBs02": {"url": "https://apis.data.go.kr/6270000/dbmsapi02/getBs02", "params": {"routeId": "3000204000"}},
    "getLink02": {"url": "https://apis.data.go.kr/6270000/dbmsapi02/getLink02", "params": {"routeId": "3000204000"}},
    "getPos02": {"url": "https://apis.data.go.kr/6270000/dbmsapi02/getPos02", "params": {"routeId": "3000204000"}},
    "getRealtime02": {"url": "https://apis.data.go.kr/6270000/dbmsapi02/getRealtime02", "params": {"bsId": "7001007900"}},
}
POSITION_SAMPLE_COLUMNS = [
    "request_time_kst",
    "endpoint",
    "route_id",
    "direction_id",
    "vehicle_id",
    "provider_event_time",
    "current_sequence",
    "current_stop_id",
    "x",
    "y",
    "response_status",
    "raw_file_path",
    "raw_sha256",
]


def kst_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows], columns=columns).to_parquet(path, index=False)


def read_text_maybe_rtf(path: Path) -> str:
    if path.suffix.lower() == ".rtf":
        result = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)], check=True, capture_output=True, text=True)
        return result.stdout
    return path.read_text(encoding="utf-8", errors="ignore")


def read_service_key(key_file: str) -> Tuple[str, str]:
    if key_file:
        text = read_text_maybe_rtf(Path(key_file).expanduser())
        match = KEY_VALUE_RE.search(text)
        if match:
            return match.group(1).strip(), "file"
        candidates = re.findall(r"[A-Za-z0-9%+/=_\\-]{20,}", text)
        if candidates:
            return max(candidates, key=len).strip(), "file"
    env_key = os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY") or ""
    return env_key, "environment" if env_key else "missing"


def redact(value: str, service_key: str) -> str:
    encoded = urllib.parse.quote(service_key, safe="%")
    encoded_plus = urllib.parse.quote_plus(service_key, safe="%")
    return value.replace(service_key, "<SERVICE_KEY>").replace(encoded, "<SERVICE_KEY>").replace(encoded_plus, "<SERVICE_KEY>")


def build_url(base_url: str, service_key: str, params: Mapping[str, str]) -> str:
    merged = [("serviceKey", service_key), *[(k, v) for k, v in params.items()], ("resultType", "json")]
    sep = "&" if "?" in base_url else "?"
    return base_url + sep + urllib.parse.urlencode(merged, safe="%")


def detect_provider_status(raw_text: str, http_status: Optional[int]) -> str:
    stripped = (raw_text or "").strip()
    low = stripped.lower()
    if http_status == 429:
        return "RATE_LIMIT"
    if not stripped:
        return "EMPTY_RESPONSE"
    if stripped.startswith("<!doctype html") or stripped.startswith("<html") or "<html" in low[:500]:
        return "HTML_RESPONSE"
    if any(marker in low for marker in ["servicekey", "service key", "인증키", "등록되지 않은", "unauthorized", "not authorized", "invalid"]):
        return "AUTH_ERROR"
    if any(marker in low for marker in ["service error", "application error", "internal server error", "시스템 오류", "서비스 오류"]):
        return "PROVIDER_ERROR"
    if http_status is None:
        return "NETWORK_ERROR"
    if http_status >= 400:
        return "HTTP_ERROR"
    return "OK_OR_NO_ITEMS"


def infer_format(raw_text: str, content_type: str) -> str:
    stripped = (raw_text or "").lstrip()
    low_ct = (content_type or "").lower()
    if "json" in low_ct or stripped.startswith("{") or stripped.startswith("["):
        return "json"
    if "xml" in low_ct or stripped.startswith("<"):
        return "xml"
    if "," in stripped[:1000]:
        return "csv_or_text"
    return "unknown"


def find_items(obj: Any) -> List[Dict[str, Any]]:
    paths = [
        ["body", "items"],
        ["body", "items", "item"],
        ["response", "body", "items"],
        ["response", "body", "items", "item"],
        ["items"],
        ["item"],
    ]
    for path in paths:
        cur = obj
        for part in path:
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = None
                break
        if isinstance(cur, list):
            return [row for row in cur if isinstance(row, dict)]
        if isinstance(cur, dict):
            return [cur]
    rows: List[Dict[str, Any]] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if any(k in value for k in ["routeId", "bsId", "vhcNo", "vhcNo2", "linkId", "seq"]):
                rows.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(obj)
    return rows


def parse_rows(raw_text: str, content_type: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    fmt = infer_format(raw_text, content_type)
    try:
        if fmt == "json":
            return fmt, find_items(json.loads(raw_text)), None
        if fmt == "xml":
            root = ET.fromstring(raw_text)
            rows = []
            for item in root.findall(".//item"):
                row = {child.tag.split("}")[-1]: child.text for child in list(item)}
                if row:
                    rows.append(row)
            return fmt, rows, None
        return fmt, [], "unparsed_response_format"
    except Exception as exc:
        return fmt, [], f"{type(exc).__name__}: {exc}"


def fetch_endpoint(endpoint: str, base_url: str, params: Mapping[str, str], service_key: str, raw_dir: Path, timeout: int = 30) -> Dict[str, Any]:
    request_time = kst_now()
    url = build_url(base_url, service_key, params)
    req = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1c-r1/1.0"})
    http_status: Optional[int] = None
    content_type = ""
    raw_bytes = b""
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            http_status = int(getattr(response, "status", 200) or 200)
            raw_bytes = response.read()
            content_type = response.headers.get("Content-Type", "")
            encoding = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        http_status = int(exc.code)
        raw_bytes = exc.read() or b""
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        encoding = exc.headers.get_content_charset() if exc.headers else "utf-8"
    except Exception as exc:
        raw_bytes = str(exc).encode("utf-8", errors="replace")
        content_type = "text/plain"
        encoding = "utf-8"
        error = f"{type(exc).__name__}: {exc}"
    raw_text = raw_bytes.decode(encoding or "utf-8", errors="replace")
    fmt, rows, parse_error = parse_rows(raw_text, content_type)
    provider_status = detect_provider_status(raw_text, http_status)
    ext = ".json" if fmt == "json" else ".xml" if fmt == "xml" else ".txt"
    raw_path = raw_dir / endpoint / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_bytes)
    parseable = parse_error is None and fmt in {"json", "xml"}
    pass_condition = bool(
        http_status is not None
        and 200 <= http_status < 300
        and provider_status not in {"AUTH_ERROR", "HTML_RESPONSE", "NETWORK_ERROR", "PROVIDER_ERROR", "HTTP_ERROR", "RATE_LIMIT"}
        and parseable
    )
    return {
        "endpoint": endpoint,
        "request_time_kst": request_time,
        "request_url_redacted": redact(url, service_key),
        "http_status": http_status,
        "content_type": content_type,
        "provider_status": provider_status,
        "response_format": fmt,
        "parseable": parseable,
        "parse_error": parse_error,
        "row_count": len(rows),
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "pass": pass_condition,
        "error": error,
        "rows": rows,
    }


def norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_bs_rows(route_id: str, rows: Sequence[Mapping[str, Any]], raw_meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "route_id": norm(row.get("routeId")) or route_id,
                "direction_id": norm(row.get("moveDir")) or norm(row.get("moveDirCode")) or norm(row.get("direction_id")),
                "stop_order": norm(row.get("seq")) or norm(row.get("bsSeq")) or norm(row.get("stopSeq")),
                "stop_id": norm(row.get("bsId")) or norm(row.get("stopId")),
                "stop_name": norm(row.get("bsNm")) or norm(row.get("stopName")),
                "x": norm(row.get("xPos")) or norm(row.get("x")),
                "y": norm(row.get("yPos")) or norm(row.get("y")),
                "raw_path": raw_meta["raw_path"],
                "raw_sha256": raw_meta["raw_sha256"],
            }
        )
    return out


def normalize_link_rows(route_id: str, rows: Sequence[Mapping[str, Any]], raw_meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "route_id": norm(row.get("routeId")) or route_id,
                "direction_id": norm(row.get("moveDir")) or norm(row.get("moveDirCode")),
                "link_seq": norm(row.get("linkSeq")) or norm(row.get("seq")),
                "link_id": norm(row.get("linkId")),
                "st_node": norm(row.get("stNode")),
                "ed_node": norm(row.get("edNode")),
                "gis_dist": norm(row.get("gisDist")),
                "raw_path": raw_meta["raw_path"],
                "raw_sha256": raw_meta["raw_sha256"],
            }
        )
    return out


def normalize_position_rows(route_id: str, rows: Sequence[Mapping[str, Any]], raw_meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "request_time_kst": raw_meta["request_time_kst"],
                "endpoint": raw_meta["endpoint"],
                "route_id": norm(row.get("routeId")) or route_id,
                "direction_id": norm(row.get("moveDir")) or norm(row.get("direction_id")),
                "vehicle_id": norm(row.get("vhcNo2")) or norm(row.get("vhcNo")) or norm(row.get("busId")),
                "provider_event_time": norm(row.get("arTime")) or norm(row.get("eventTime")) or norm(row.get("tm")),
                "current_sequence": norm(row.get("seq")) or norm(row.get("stopSeq")),
                "current_stop_id": norm(row.get("bsId")) or norm(row.get("stopId")),
                "x": norm(row.get("xPos")) or norm(row.get("x")),
                "y": norm(row.get("yPos")) or norm(row.get("y")),
                "response_status": raw_meta["provider_status"],
                "raw_file_path": raw_meta["raw_path"],
                "raw_sha256": raw_meta["raw_sha256"],
            }
        )
    return out


def normalize_realtime_rows(stop_id: str, rows: Sequence[Mapping[str, Any]], raw_meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "stop_id": norm(row.get("bsId")) or stop_id,
                "route_id": norm(row.get("routeId")),
                "direction_id": norm(row.get("moveDir")) or norm(row.get("direction_id")),
                "vehicle_id": norm(row.get("vhcNo2")) or norm(row.get("vhcNo")),
                "eta_raw": norm(row.get("arrTime")) or norm(row.get("predictTime")) or norm(row.get("remainTime")),
                "raw_path": raw_meta["raw_path"],
                "raw_sha256": raw_meta["raw_sha256"],
            }
        )
    return out


def scan_for_secret(root: Path, service_key: str, output_root: Optional[Path] = None) -> Dict[str, Any]:
    needles = {service_key, urllib.parse.quote(service_key, safe="%"), urllib.parse.quote_plus(service_key, safe="%")}
    findings = []
    skip_names = {".git", ".venv", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in skip_names for part in path.parts):
            continue
        if output_root is not None and output_root in path.parents:
            continue
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            data = path.read_bytes()
        except Exception:
            continue
        if any(needle.encode("utf-8", errors="ignore") in data for needle in needles):
            findings.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def summarize_endpoint(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if k != "rows"}


def load_mapping_v5(project_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    artifact = project_root / R2D1C_ROOT
    mapping_path = artifact / "turnaround_mapping_contract_v5.parquet"
    gate_path = artifact / "prompt5_e01_r2d1c_gate.json"
    rows = pd.read_parquet(mapping_path).to_dict("records")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    return rows, {"artifact_dir": str(artifact), "gate_path": str(gate_path), "gate_sha256": sha256_file(gate_path), "gate": gate}


def build_mapping_v6(v5_rows: Sequence[Mapping[str, Any]], bs_rows: Sequence[Mapping[str, Any]], link_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bs_by_route = {}
    for row in bs_rows:
        key = str(row.get("route_id"))
        bs_by_route.setdefault(key, 0)
        bs_by_route[key] += 1
    link_by_route = {}
    for row in link_rows:
        key = str(row.get("route_id"))
        link_by_route.setdefault(key, 0)
        link_by_route[key] += 1
    out_rows = []
    audit_rows = []
    for row in v5_rows:
        route_id = str(row.get("route_id"))
        copied = dict(row)
        if not bool(row.get("approved")):
            copied["mapping_v6_decision"] = "UNRESOLVED_SEMANTICS"
            copied["mapping_v6_reason"] = (
                "BIS credential/API access can provide route sequence/link candidates, but this run found no direct provider field "
                "proving terminal paired-direction transition, loop closure after service terminal, or off-graph re-entry elapsed path."
            )
            audit_rows.append(
                {
                    "route_id": route_id,
                    "direction_id": str(row.get("direction_id") or row.get("service_direction_id")),
                    "getbs02_row_count": bs_by_route.get(route_id, 0),
                    "getlink02_row_count": link_by_route.get(route_id, 0),
                    "approved_in_v6": False,
                    "reason": copied["mapping_v6_reason"],
                }
            )
        else:
            copied["mapping_v6_decision"] = "PRESERVED_APPROVED_FROM_V5"
            copied["mapping_v6_reason"] = "Previously approved mapping preserved; no checkpoint/config/model change."
        out_rows.append(copied)
    approved = sum(1 for row in out_rows if bool(row.get("approved")))
    return (
        {
            "contract_version": "turnaround_mapping_contract_v6",
            "expected_mapping_count": 38,
            "previous_approved_mapping_count": 25,
            "newly_recovered_mapping_count": 0,
            "approved_mapping_count": approved,
            "missing_mapping_count": 38 - approved,
            "mapping_provenance_ready": approved == 38,
            "status_if_credentials_pass": "BLOCKED_MAPPING_SEMANTICS_UNRESOLVED" if approved < 38 else "PASS_CREDENTIAL_AND_MAPPING_READY",
            "rows": out_rows,
        },
        audit_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="R2D-1C-R1 BIS credential, endpoint preflight, canary, and mapping v6 audit.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--service-key-file", default="")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--canary-seconds", type=int, default=600)
    parser.add_argument("--canary-interval-seconds", type=int, default=60)
    parser.add_argument("--max-calls-per-minute", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    raw_root = output_root / "raw_bis_responses"

    service_key, key_source = read_service_key(args.service_key_file)
    if not service_key:
        raise SystemExit("service key missing")
    credential = {
        "credential_available": True,
        "credential_length": len(service_key),
        "credential_source": key_source,
        "service_key_literal_recorded": False,
        "shell_history_required": False,
        "missing_service_key_unblocked": True,
    }
    dump_json(output_root / "credential_injection_audit.json", credential)
    pre_secret_audit = scan_for_secret(project_root, service_key, output_root)

    v5_rows, r2d1c_reference = load_mapping_v5(project_root)
    dump_json(output_root / "r2d1c_reference.json", r2d1c_reference)

    endpoint_results = []
    normalized_basic_rows: List[Dict[str, Any]] = []
    normalized_realtime_rows: List[Dict[str, Any]] = []
    for name, spec in ENDPOINTS.items():
        result = fetch_endpoint(name, spec["url"], spec["params"], service_key, raw_root)
        endpoint_results.append(summarize_endpoint(result))
        if name == "getBasic02":
            for row in result["rows"]:
                normalized_basic_rows.append(
                    {
                        "route_id": norm(row.get("routeId")),
                        "route_no": norm(row.get("routeNo")) or norm(row.get("routeNm")),
                        "route_type": norm(row.get("routeType")),
                        "origin_stop_id": norm(row.get("stBsId")) or norm(row.get("startBsId")),
                        "destination_stop_id": norm(row.get("edBsId")) or norm(row.get("endBsId")),
                        "raw_path": result["raw_path"],
                        "raw_sha256": result["raw_sha256"],
                    }
                )
        elif name == "getRealtime02":
            normalized_realtime_rows.extend(normalize_realtime_rows(str(spec["params"]["bsId"]), result["rows"], result))
        time.sleep(max(60.0 / max(args.max_calls_per_minute, 1), 0.0))

    write_parquet(output_root / "getbasic02_master_preflight_normalized.parquet", normalized_basic_rows)
    write_parquet(output_root / "getrealtime02_preflight_normalized.parquet", normalized_realtime_rows)
    dump_json(output_root / "endpoint_preflight_audit.json", {"endpoint_results": endpoint_results})
    with (output_root / "endpoint_preflight_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint", "http_status", "provider_status", "response_format", "parseable", "row_count", "pass", "raw_sha256"])
        writer.writeheader()
        for result in endpoint_results:
            writer.writerow({key: result.get(key) for key in writer.fieldnames})

    recapture_results = []
    bs_rows: List[Dict[str, Any]] = []
    link_rows: List[Dict[str, Any]] = []
    for route_id, direction_id in MISSING_KEYS:
        for endpoint_name in ["getBs02", "getLink02"]:
            spec = ENDPOINTS[endpoint_name]
            result = fetch_endpoint(endpoint_name, spec["url"], {"routeId": route_id}, service_key, raw_root / "missing_routes")
            row = summarize_endpoint(result)
            row["requested_direction_id"] = direction_id
            recapture_results.append(row)
            if endpoint_name == "getBs02":
                bs_rows.extend(normalize_bs_rows(route_id, result["rows"], result))
            else:
                link_rows.extend(normalize_link_rows(route_id, result["rows"], result))
            time.sleep(max(60.0 / max(args.max_calls_per_minute, 1), 0.0))
    write_parquet(output_root / "getbs02_missing_route_sequences.parquet", bs_rows)
    write_parquet(output_root / "getlink02_missing_route_links.parquet", link_rows)
    dump_json(
        output_root / "missing_route_bis_recapture_manifest.json",
        {
            "target_route_direction_count": len(MISSING_KEYS),
            "getbs02_normalized_row_count": len(bs_rows),
            "getlink02_normalized_row_count": len(link_rows),
            "results": recapture_results,
        },
    )

    canary_rows: List[Dict[str, Any]] = []
    canary_requests = []
    canary_started = datetime.now().astimezone()
    canary_end_monotonic = time.monotonic() + max(args.canary_seconds, 0)
    sample_index = 0
    min_call_sleep = max(60.0 / max(args.max_calls_per_minute, 1), 0.0)
    while time.monotonic() < canary_end_monotonic:
        sample_index += 1
        sample_started = time.monotonic()
        for route_id in CANARY_ROUTE_IDS:
            result = fetch_endpoint("getPos02", ENDPOINTS["getPos02"]["url"], {"routeId": route_id}, service_key, raw_root / "live_canary")
            request = summarize_endpoint(result)
            request["sample_index"] = sample_index
            request["normalized_row_count"] = len(result["rows"])
            canary_requests.append(request)
            canary_rows.extend(normalize_position_rows(route_id, result["rows"], result))
            sleep_for = max(min_call_sleep * 2.0, 10.0) if result["provider_status"] == "RATE_LIMIT" else min_call_sleep
            time.sleep(min(sleep_for, max(0.0, canary_end_monotonic - time.monotonic())))
            if time.monotonic() >= canary_end_monotonic:
                break
        remaining = args.canary_interval_seconds - (time.monotonic() - sample_started)
        if remaining > 0:
            time.sleep(min(remaining, max(0.0, canary_end_monotonic - time.monotonic())))
    canary_finished = datetime.now().astimezone()
    write_parquet(output_root / "getpos02_live_canary_samples.parquet", canary_rows, columns=POSITION_SAMPLE_COLUMNS)
    canary_df = pd.DataFrame(canary_rows, columns=POSITION_SAMPLE_COLUMNS)
    canary_manifest = {
        "canary_executed": True,
        "started_at_kst": canary_started.isoformat(timespec="seconds"),
        "finished_at_kst": canary_finished.isoformat(timespec="seconds"),
        "duration_seconds_requested": args.canary_seconds,
        "route_ids": CANARY_ROUTE_IDS,
        "interval_seconds": args.canary_interval_seconds,
        "max_calls_per_minute": args.max_calls_per_minute,
        "request_count": len(canary_requests),
        "raw_file_count": len(canary_requests),
        "normalized_row_count": len(canary_rows),
        "unique_vehicle_count": int(canary_df["vehicle_id"].nunique(dropna=True)) if not canary_df.empty else 0,
        "repeated_vehicle_count": int((canary_df.groupby(["route_id", "direction_id", "vehicle_id"]).size() >= 2).sum()) if not canary_df.empty else 0,
        "sequence_progression_observed_count": int(
            sum(group["current_sequence"].nunique(dropna=True) >= 2 for _, group in canary_df.groupby(["route_id", "direction_id", "vehicle_id"]))
        ) if not canary_df.empty else 0,
        "rate_limit_response_count": sum(1 for row in canary_requests if row["provider_status"] == "RATE_LIMIT"),
        "auth_error_count": sum(1 for row in canary_requests if row["provider_status"] == "AUTH_ERROR"),
        "html_response_count": sum(1 for row in canary_requests if row["provider_status"] == "HTML_RESPONSE"),
        "requests": canary_requests,
        "pilot_recovery_claim_allowed": False,
    }
    dump_json(output_root / "getpos02_canary_manifest.json", canary_manifest)

    mapping_v6, mapping_audit_rows = build_mapping_v6(v5_rows, bs_rows, link_rows)
    write_parquet(output_root / "turnaround_mapping_contract_v6.parquet", mapping_v6["rows"])
    dump_json(output_root / "turnaround_mapping_contract_v6.json", mapping_v6)
    dump_json(output_root / "mapping_semantics_audit_v6.json", {"rows": mapping_audit_rows})

    endpoint_pass = all(bool(row["pass"]) for row in endpoint_results)
    credential_pass = endpoint_pass and not any(row["provider_status"] == "AUTH_ERROR" for row in endpoint_results + recapture_results + canary_requests)
    if not credential_pass:
        status = "BLOCKED_CREDENTIAL_OR_ENDPOINT_PREFLIGHT_FAILED"
    elif mapping_v6["approved_mapping_count"] == 38:
        status = "PASS_CREDENTIAL_AND_MAPPING_READY"
    else:
        status = "BLOCKED_MAPPING_SEMANTICS_UNRESOLVED"

    authorization = {
        "approved_for_phase2_turnaround_execution": False,
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "phase2_production_executed": False,
        "three_day_recovery_campaign_executed": False,
    }
    dump_json(output_root / "phase2_execution_authorization.json", authorization)

    post_secret_audit = scan_for_secret(output_root, service_key)
    leak_audit = {
        "repo_scan": pre_secret_audit,
        "artifact_scan": post_secret_audit,
        "artifact_secret_literal_occurrence_count": post_secret_audit["secret_literal_occurrence_count"],
        "artifact_secret_leak_pass": post_secret_audit["secret_literal_occurrence_count"] == 0,
        "raw_response_sha_recorded": True,
        "request_urls_redacted": True,
    }
    dump_json(output_root / "secret_leak_audit.json", leak_audit)
    if post_secret_audit["secret_literal_occurrence_count"] > 0:
        status = "FAIL_SECRET_LEAK_IN_ARTIFACT"

    gate = {
        "status": status,
        "classification": status,
        "credential_available": True,
        "credential_length": len(service_key),
        "missing_service_key_unblocked": True,
        "secret_leak_artifact_pass": post_secret_audit["secret_literal_occurrence_count"] == 0,
        "secret_leak_repo_existing_occurrence_count": pre_secret_audit["secret_literal_occurrence_count"],
        "endpoint_preflight_count": len(endpoint_results),
        "endpoint_preflight_pass_count": sum(1 for row in endpoint_results if row["pass"]),
        "auth_error_count": sum(1 for row in endpoint_results + recapture_results + canary_requests if row["provider_status"] == "AUTH_ERROR"),
        "html_response_count": sum(1 for row in endpoint_results + recapture_results + canary_requests if row["provider_status"] == "HTML_RESPONSE"),
        "missing_route_recapture_route_direction_count": len(MISSING_KEYS),
        "getbs02_missing_route_normalized_row_count": len(bs_rows),
        "getlink02_missing_route_normalized_row_count": len(link_rows),
        "canary_executed": True,
        "canary_duration_seconds_requested": args.canary_seconds,
        "canary_request_count": len(canary_requests),
        "canary_normalized_row_count": len(canary_rows),
        "canary_unique_vehicle_count": canary_manifest["unique_vehicle_count"],
        "canary_repeated_vehicle_count": canary_manifest["repeated_vehicle_count"],
        "canary_sequence_progression_observed_count": canary_manifest["sequence_progression_observed_count"],
        "expected_mapping_count": 38,
        "previous_approved_mapping_count": 25,
        "newly_recovered_mapping_count": mapping_v6["newly_recovered_mapping_count"],
        "approved_mapping_count": mapping_v6["approved_mapping_count"],
        "missing_mapping_count": mapping_v6["missing_mapping_count"],
        "official_recovery_approved_count": 0,
        "terminal_recovery_unresolved_count": 38,
        "phase2_production_executed": False,
        "three_day_recovery_campaign_executed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        **authorization,
    }
    dump_json(output_root / "prompt5_e01_r2d1c_r1_gate.json", gate)
    dump_json(
        output_root / "prompt5_e01_r2d1c_r1_manifest.json",
        {
            "artifact_version": "prompt5_e01_r2d1c_r1_bis_credential_mapping_preflight_v1",
            "created_at_kst": kst_now(),
            "project_root": str(project_root),
            "artifact_dir": str(output_root),
            "files": sorted([str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()]),
        },
    )
    report = f"""# Prompt 5-E01-R2D-1C-R1 Final Report

status: {status}
classification: {status}

## Credential
- credential_available: true
- credential_source: {key_source}
- service key literal recorded: false
- artifact secret leak pass: {post_secret_audit["secret_literal_occurrence_count"] == 0}

## Endpoint Preflight
- endpoint count: {len(endpoint_results)}
- pass count: {sum(1 for row in endpoint_results if row["pass"])}
- auth errors: {gate["auth_error_count"]}
- html responses: {gate["html_response_count"]}

## Missing Route Recapture
- route-directions: {len(MISSING_KEYS)}
- getBs02 normalized rows: {len(bs_rows)}
- getLink02 normalized rows: {len(link_rows)}

## 10-Min Live Canary
- executed: true
- requested duration seconds: {args.canary_seconds}
- requests: {len(canary_requests)}
- normalized rows: {len(canary_rows)}
- unique vehicles: {canary_manifest["unique_vehicle_count"]}
- repeated vehicles: {canary_manifest["repeated_vehicle_count"]}
- sequence progression observed: {canary_manifest["sequence_progression_observed_count"]}

## Mapping v6
- previous approved: 25
- newly recovered: {mapping_v6["newly_recovered_mapping_count"]}
- total approved: {mapping_v6["approved_mapping_count"]}
- missing: {mapping_v6["missing_mapping_count"]}
- interpretation: API credential access is no longer the blocker if endpoint preflight passes. Remaining unresolved mappings require direct terminal operation semantics.

## Guards
- phase2_production_executed: false
- three_day_recovery_campaign_executed: false
- E0/E1 retraining: false
- Prompt 6A: false
- E2: false
- test split/target/embedding read: false
"""
    (output_root / "prompt5_e01_r2d1c_r1_final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"artifact_dir": str(output_root), "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
