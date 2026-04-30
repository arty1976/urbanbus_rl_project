from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore

ARTIFACT_VERSION = "daegu_bis_api_audit_step99b_getbasic02_v1"
DEFAULT_BASE_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getBasic02"
DEFAULT_OUTPUT_DIR = Path("artifacts/daegu_bis_api_audit/getbasic02_master_snapshot_audit")

ROUTE_ID_ALIASES = {
    "routeid", "route_id", "routeId", "ROUTE_ID", "노선ID", "노선아이디",
    "routeIdRaw", "route_id_raw",
}
ROUTE_NO_ALIASES = {
    "routeno", "route_no", "routeNo", "ROUTE_NO", "노선번호", "노선명", "routeNm", "routeName",
}
ROUTE_TYPE_ALIASES = {
    "route_type", "routeType", "routeTp", "노선유형", "노선종류", "type", "routeKind",
}
DIRECTION_ALIASES = {
    "directionid", "direction_id", "directionId", "moveDir", "movedir", "move_dir",
    "moveDirCode", "move_dir_code", "MOVE_DIR_CODE", "dir", "dirCode", "방면", "노선방면",
    "direction", "updown", "upDown",
}
ORIGIN_STOP_ID_ALIASES = {
    "origin_stop_id", "originStopId", "startStopId", "stBsId", "startBsId",
    "기점정류소ID", "기점정류장ID", "기점ID",
}
ORIGIN_STOP_NAME_ALIASES = {
    "origin_stop_name", "originStopName", "startStopName", "stBsNm", "startBsNm",
    "기점정류소명", "기점정류장명", "기점명",
}
DEST_STOP_ID_ALIASES = {
    "destination_stop_id", "destinationStopId", "endStopId", "edBsId", "endBsId",
    "종점정류소ID", "종점정류장ID", "종점ID",
}
DEST_STOP_NAME_ALIASES = {
    "destination_stop_name", "destinationStopName", "endStopName", "edBsNm", "endBsNm",
    "종점정류소명", "종점정류장명", "종점명",
}

SELF_TEST_RAW = {
    "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
    "body": {
        "items": {
            "item": [
                {
                    "routeId": "1000005000",
                    "routeNo": "급행5",
                    "routeType": "급행",
                    "moveDir": "1",
                    "stBsId": "7041054300",
                    "stBsNm": "신흥버스",
                    "edBsId": "7121021100",
                    "edBsNm": "대구대(정문1)",
                },
                {
                    "routeId": "3000655000",
                    "routeNo": "655",
                    "routeType": "간선",
                    "moveDir": "0",
                    "stBsId": "7001000100",
                    "stBsNm": "기점정류소",
                    "edBsId": "7001000200",
                    "edBsNm": "종점정류소",
                },
            ]
        }
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def detect_provider_status(raw_text: str) -> str:
    s = (raw_text or "").strip()
    low = s.lower()
    if not s:
        return "empty_response"
    provider_markers = [
        "unexpected errors",
        "service error",
        "application error",
        "internal server error",
        "provider error",
        "시스템 오류",
        "서비스 오류",
    ]
    auth_markers = [
        "servicekey",
        "service key",
        "인증키",
        "등록되지 않은",
        "invalid",
        "unauthorized",
        "not authorized",
        "auth",
        "인증",
    ]
    no_item_markers = ["no items", "no data", "데이터 없음", "조회된 데이터가 없습니다"]
    if any(m in low for m in provider_markers):
        return "provider_error"
    if any(m in low for m in auth_markers):
        return "auth_error"
    if any(m in low for m in no_item_markers):
        return "no_items"
    return "ok_or_unclassified"


def build_url(base_url: str, service_key: str, extra_params: Sequence[str]) -> str:
    params: List[Tuple[str, str]] = [("serviceKey", service_key)]
    for p in extra_params:
        if not p:
            continue
        if "=" not in p:
            raise ValueError(f"extra param must be key=value: {p}")
        k, v = p.split("=", 1)
        params.append((k, v))
    query = urllib.parse.urlencode(params, doseq=True, safe="%")
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{query}"


def fetch_api(base_url: str, service_key: str, extra_params: Sequence[str], timeout: int) -> Dict[str, Any]:
    url = build_url(base_url, service_key, extra_params)
    encoded_key = urllib.parse.quote(service_key, safe="")
    redacted_url = url.replace(encoded_key, "<SERVICE_KEY>").replace(service_key, "<SERVICE_KEY>")
    req = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-step99b-getbasic02-audit/1.0"})

    status_code = None
    error_reason = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = int(getattr(resp, "status", 200) or 200)
            raw_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            encoding = resp.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as e:
        status_code = int(e.code)
        error_reason = str(e.reason)
        raw_bytes = e.read() or b""
        content_type = e.headers.get("Content-Type", "") if e.headers else ""
        encoding = e.headers.get_content_charset() if e.headers else None
        encoding = encoding or "utf-8"
    except urllib.error.URLError as e:
        status_code = None
        error_reason = str(e.reason)
        raw_bytes = str(e).encode("utf-8", errors="replace")
        content_type = "text/plain"
        encoding = "utf-8"

    try:
        raw_text = raw_bytes.decode(encoding, errors="replace")
    except Exception:
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    return {
        "request_url_redacted": redacted_url,
        "http_status": status_code,
        "http_error_reason": error_reason,
        "api_call_ok": bool(status_code is not None and 200 <= status_code < 300),
        "content_type": content_type,
        "raw_text": raw_text,
        "provider_status": detect_provider_status(raw_text),
    }


def infer_response_format(raw_text: str, content_type: str = "") -> str:
    s = (raw_text or "").strip()
    ct = (content_type or "").lower()
    if "json" in ct or s.startswith("{") or s.startswith("["):
        return "json"
    if "xml" in ct or s.startswith("<"):
        return "xml"
    if "," in s and "\n" in s[:1000]:
        return "csv_or_text"
    return "unknown"


def normalize_key(k: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣_]", "", str(k)).strip()


def is_alias(k: str, aliases: set[str]) -> bool:
    nk = normalize_key(k)
    lk = nk.lower()
    return nk in aliases or lk in {a.lower() for a in aliases}


def first_value(row: Dict[str, Any], aliases: set[str]) -> Optional[Any]:
    for k, v in row.items():
        if is_alias(k, aliases):
            return v
    return None


def hit_keys(row: Dict[str, Any], aliases: set[str]) -> List[str]:
    return [str(k) for k in row.keys() if is_alias(str(k), aliases)]


def flatten_json_candidates(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                # Prefer lists that look like table rows.
                row_like = [i for i in x if isinstance(i, dict) and len(i) >= 2]
                if row_like:
                    out.extend(row_like)
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(obj)
    # Deduplicate by JSON representation while preserving order.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in out:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def parse_xml_rows(raw_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(raw_text)
    except Exception:
        return rows
    for item in root.findall(".//item"):
        row: Dict[str, Any] = {}
        for child in list(item):
            row[child.tag] = child.text
        if row:
            rows.append(row)
    return rows


def extract_rows(raw_text: str, fmt: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        if fmt == "json":
            obj = json.loads(raw_text)
            return flatten_json_candidates(obj), None
        if fmt == "xml":
            return parse_xml_rows(raw_text), None
        return [], None
    except Exception as e:
        return [], str(e)


def normalize_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], Counter]:
    normalized: List[Dict[str, Any]] = []
    schema_hits: Dict[str, List[str]] = defaultdict(list)
    key_counter: Counter = Counter()

    for row in rows:
        for k in row.keys():
            key_counter[str(k)] += 1

        hits = {
            "route_id": hit_keys(row, ROUTE_ID_ALIASES),
            "route_no": hit_keys(row, ROUTE_NO_ALIASES),
            "route_type": hit_keys(row, ROUTE_TYPE_ALIASES),
            "direction_id": hit_keys(row, DIRECTION_ALIASES),
            "origin_stop_id": hit_keys(row, ORIGIN_STOP_ID_ALIASES),
            "origin_stop_name": hit_keys(row, ORIGIN_STOP_NAME_ALIASES),
            "destination_stop_id": hit_keys(row, DEST_STOP_ID_ALIASES),
            "destination_stop_name": hit_keys(row, DEST_STOP_NAME_ALIASES),
        }
        for field, ks in hits.items():
            for k in ks:
                if k not in schema_hits[field]:
                    schema_hits[field].append(k)

        route_id = first_value(row, ROUTE_ID_ALIASES)
        route_no = first_value(row, ROUTE_NO_ALIASES)
        route_type = first_value(row, ROUTE_TYPE_ALIASES)
        direction_id = first_value(row, DIRECTION_ALIASES)
        origin_stop_id = first_value(row, ORIGIN_STOP_ID_ALIASES)
        origin_stop_name = first_value(row, ORIGIN_STOP_NAME_ALIASES)
        destination_stop_id = first_value(row, DEST_STOP_ID_ALIASES)
        destination_stop_name = first_value(row, DEST_STOP_NAME_ALIASES)

        if any(v is not None for v in [route_id, route_no, route_type, direction_id, origin_stop_id, destination_stop_id]):
            normalized.append({
                "route_id": str(route_id).strip() if route_id is not None else None,
                "route_no": str(route_no).strip() if route_no is not None else None,
                "route_type": str(route_type).strip() if route_type is not None else None,
                "direction_id": str(direction_id).strip() if direction_id is not None else None,
                "origin_stop_id": str(origin_stop_id).strip() if origin_stop_id is not None else None,
                "origin_stop_name": str(origin_stop_name).strip() if origin_stop_name is not None else None,
                "destination_stop_id": str(destination_stop_id).strip() if destination_stop_id is not None else None,
                "destination_stop_name": str(destination_stop_name).strip() if destination_stop_name is not None else None,
                "raw_keys": sorted([str(k) for k in row.keys()]),
            })

    return normalized, dict(schema_hits), key_counter


def load_db_route_ids(dsn: str) -> Tuple[set[str], str]:
    if psycopg2 is None:
        return set(), "psycopg2 is not installed/importable"
    try:
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                # Prefer stg_daegu_routes because this is the source catalog used by Step 99-A.
                cur.execute("""
                    SELECT route_id::text
                    FROM public.stg_daegu_routes
                    WHERE route_id IS NOT NULL
                """)
                ids = {str(r[0]) for r in cur.fetchall()}
                return ids, ""
        finally:
            conn.close()
    except Exception as e:
        return set(), str(e)


def compute_db_coverage(normalized: List[Dict[str, Any]], dsn: Optional[str]) -> Dict[str, Any]:
    route_ids = {str(r["route_id"]) for r in normalized if r.get("route_id")}
    if not dsn:
        return {
            "db_coverage_enabled": False,
            "route_id_count_in_api": len(route_ids),
            "db_route_id_count": None,
            "matched_route_id_count": None,
            "route_id_coverage_pct": None,
            "reason_error": "URBANBUS_DB_DSN not provided",
        }
    db_ids, err = load_db_route_ids(dsn)
    if err:
        return {
            "db_coverage_enabled": False,
            "route_id_count_in_api": len(route_ids),
            "db_route_id_count": None,
            "matched_route_id_count": None,
            "route_id_coverage_pct": None,
            "reason_error": err,
        }
    matched = route_ids & db_ids
    pct = (len(matched) / len(route_ids) * 100.0) if route_ids else None
    return {
        "db_coverage_enabled": True,
        "route_id_count_in_api": len(route_ids),
        "db_route_id_count": len(db_ids),
        "matched_route_id_count": len(matched),
        "route_id_coverage_pct": round(pct, 4) if pct is not None else None,
        "api_route_ids_not_in_db_sample": sorted(route_ids - db_ids)[:20],
        "db_route_ids_not_in_api_sample": sorted(db_ids - route_ids)[:20],
        "reason_error": "",
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["raw_keys"] = "|".join(out.get("raw_keys") or [])
            writer.writerow(out)


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    service_key = args.service_key or os.getenv("DAEGU_BIS_SERVICE_KEY") or os.getenv("DATAGO_SERVICE_KEY") or ""
    api_called = False
    skip_reason = None

    if args.self_test or not service_key:
        raw_text = json.dumps(SELF_TEST_RAW, ensure_ascii=False)
        raw_record = {
            "request_url_redacted": "self_test_embedded_sample",
            "http_status": None,
            "http_error_reason": "",
            "api_call_ok": False,
            "content_type": "application/json",
            "raw_text": raw_text,
            "provider_status": "self_test",
        }
        mode = "self_test_or_schema_only"
        skip_reason = None if args.self_test else "service key missing; used embedded self-test sample"
    else:
        raw_record = fetch_api(args.base_url, service_key, args.extra_param, args.timeout)
        api_called = True
        mode = "sample_api_call"
        time.sleep(max(0.0, float(args.sleep_sec)))

    raw_path = out_dir / "getbasic02_master_snapshot.raw"
    write_text(raw_path, raw_record["raw_text"])

    fmt = infer_response_format(raw_record["raw_text"], raw_record.get("content_type", ""))
    rows, parse_error = extract_rows(raw_record["raw_text"], fmt)
    normalized, schema_hits, key_counter = normalize_rows(rows)

    normalized_path = out_dir / "getbasic02_master_snapshot_normalized.csv"
    write_csv(normalized_path, normalized)

    route_ids = {r.get("route_id") for r in normalized if r.get("route_id")}
    direction_ids = {r.get("direction_id") for r in normalized if r.get("direction_id")}

    classification = {
        "route_id": {
            "candidate_status": "observed_candidate" if route_ids else "missing_or_unparsed",
            "reason": "route_id-like field detected in /getBasic02 response" if route_ids else "no route_id-like field detected",
        },
        "route_no": {
            "candidate_status": "observed_candidate" if any(r.get("route_no") for r in normalized) else "missing_or_unparsed",
            "reason": "route number/name-like field detected" if any(r.get("route_no") for r in normalized) else "no route number/name-like field detected",
        },
        "route_type": {
            "candidate_status": "observed_candidate" if any(r.get("route_type") for r in normalized) else "missing_or_unparsed",
            "reason": "route type-like field detected" if any(r.get("route_type") for r in normalized) else "no route type-like field detected",
        },
        "direction_id": {
            "candidate_status": "observed_candidate" if direction_ids else "missing_or_unparsed",
            "reason": "direction/move_dir-like field detected" if direction_ids else "no direction-equivalent field detected",
        },
        "origin_destination_stop": {
            "candidate_status": "observed_candidate" if any(r.get("origin_stop_id") or r.get("destination_stop_id") for r in normalized) else "missing_or_unparsed",
            "reason": "origin/destination stop fields detected" if any(r.get("origin_stop_id") or r.get("destination_stop_id") for r in normalized) else "no origin/destination stop fields detected",
        },
    }

    db_coverage = compute_db_coverage(normalized, args.db_dsn or os.getenv("URBANBUS_DB_DSN"))

    audit_status = "PASS"
    if raw_record.get("provider_status") in {"provider_error", "auth_error", "empty_response"}:
        audit_status = str(raw_record.get("provider_status")).upper()
    elif parse_error:
        audit_status = "PARSE_ERROR"
    elif not normalized:
        audit_status = "NO_NORMALIZED_ROWS"

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": now_utc(),
        "step": "Step 99-B",
        "title": "/getBasic02 master snapshot feasibility audit",
        "mode": mode,
        "audit_status": audit_status,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "fleet_reduction_claim_allowed": False,
        "guardrails": {
            "db_write_forbidden": True,
            "tensor_db_overwrite_forbidden": True,
            "csv_reload_forbidden": True,
            "bulk_api_collection_forbidden": True,
            "raw_snapshot_only": True,
        },
        "api_call_summary": {
            "api_called": api_called,
            "base_url": args.base_url,
            "service_key_source": "arg/env" if service_key else "missing",
            "skip_reason": skip_reason,
            "request_url_redacted": raw_record.get("request_url_redacted"),
            "http_status": raw_record.get("http_status"),
            "http_error_reason": raw_record.get("http_error_reason"),
            "api_call_ok": raw_record.get("api_call_ok"),
            "provider_status": raw_record.get("provider_status"),
            "content_type": raw_record.get("content_type"),
            "extra_params": list(args.extra_param),
        },
        "response_format": fmt,
        "parse_error": parse_error,
        "candidate_row_count": len(rows),
        "total_normalized_rows": len(normalized),
        "unique_route_id_count": len(route_ids),
        "unique_direction_id_count": len(direction_ids),
        "schema_hits": schema_hits,
        "raw_key_frequency_top50": key_counter.most_common(50),
        "classification_update_candidates": classification,
        "db_coverage": db_coverage,
        "output_files": {
            "raw": str(raw_path),
            "normalized_csv": str(normalized_path),
            "report_json": str(out_dir / "getbasic02_master_snapshot_audit_report.json"),
            "report_md": str(out_dir / "getbasic02_master_snapshot_audit_report.md"),
        },
        "normalized_preview": normalized[:20],
    }
    return report


def render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Step 99-B — /getBasic02 Master Snapshot Feasibility Audit")
    lines.append("")
    for k in ["artifact_version", "created_at_utc", "mode", "audit_status"]:
        lines.append(f"- {k}: `{report.get(k)}`")
    lines.append(f"- api_called: `{report['api_call_summary'].get('api_called')}`")
    lines.append(f"- paper_level_claim_allowed: `{report.get('paper_level_claim_allowed')}`")
    lines.append(f"- causal_performance_claim_allowed: `{report.get('causal_performance_claim_allowed')}`")
    lines.append("")
    lines.append("## API response summary")
    lines.append("")
    lines.append(f"- base_url: `{report['api_call_summary'].get('base_url')}`")
    lines.append(f"- provider_status: `{report['api_call_summary'].get('provider_status')}`")
    lines.append(f"- response_format: `{report.get('response_format')}`")
    lines.append(f"- candidate_row_count: `{report.get('candidate_row_count')}`")
    lines.append(f"- total_normalized_rows: `{report.get('total_normalized_rows')}`")
    lines.append(f"- unique_route_id_count: `{report.get('unique_route_id_count')}`")
    lines.append(f"- unique_direction_id_count: `{report.get('unique_direction_id_count')}`")
    lines.append("")
    lines.append("## Field classification update candidates")
    lines.append("")
    lines.append("| Field | Candidate status | Reason |")
    lines.append("|---|---|---|")
    for field, payload in report["classification_update_candidates"].items():
        lines.append(f"| `{field}` | `{payload.get('candidate_status')}` | {payload.get('reason')} |")
    lines.append("")
    lines.append("## DB route_id coverage")
    lines.append("")
    db = report.get("db_coverage", {})
    for k in ["db_coverage_enabled", "route_id_count_in_api", "db_route_id_count", "matched_route_id_count", "route_id_coverage_pct", "reason_error"]:
        lines.append(f"- {k}: `{db.get(k)}`")
    lines.append("")
    lines.append("## Guardrail status")
    lines.append("")
    lines.append("- DB write: forbidden and not performed")
    lines.append("- tensor DB overwrite: forbidden and not performed")
    lines.append("- CSV reload: forbidden and not performed")
    lines.append("- collection scope: master snapshot audit only")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 99-B /getBasic02 master snapshot feasibility audit")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--service-key", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--db-dsn", default="")
    p.add_argument("--extra-param", action="append", default=[])
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out_dir = Path(args.output_dir)
    json_path = out_dir / "getbasic02_master_snapshot_audit_report.json"
    md_path = out_dir / "getbasic02_master_snapshot_audit_report.md"
    dump_json(json_path, report)
    write_text(md_path, render_md(report))

    print("[OK] Step 99-B /getBasic02 master snapshot audit complete")
    print(f"[OK] mode       : {report['mode']}")
    print(f"[OK] status     : {report['audit_status']}")
    print(f"[OK] api_called : {report['api_call_summary']['api_called']}")
    print(f"[OK] normalized : {report['total_normalized_rows']}")
    print(f"[OK] json report: {json_path}")
    print(f"[OK] md report  : {md_path}")

    # Do not hard fail for provider/no-items because this is an audit step.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
