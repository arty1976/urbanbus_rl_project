#!/usr/bin/env python3
"""
Step 99-A — /getBs02 route-stop sequence acquisition feasibility audit.

This script is intentionally read-only:
- no DB writes
- no tensor DB overwrite
- no bulk API collection
- default max routes is capped at 3
- API calls are skipped unless a service key and route id are provided

Supported modes:
  python inspect_getbs02_route_stop_sequence.py --self-test
  python inspect_getbs02_route_stop_sequence.py --route-id 3000001000 --service-key %DAEGU_BIS_SERVICE_KEY%
  python inspect_getbs02_route_stop_sequence.py --route-id 3000001000 --db-dsn %URBANBUS_DB_DSN%

Notes:
- The default endpoint is a conservative placeholder based on the public
  data.go.kr Daegu BIS API naming pattern. If the Swagger/detail page gives a
  different URL, pass --base-url explicitly.
- The parser is schema-flexible and detects common Korean/English aliases.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import psycopg2
except Exception:
    psycopg2 = None


ARTIFACT_VERSION = "daegu_bis_api_audit_step99a_v1"
DEFAULT_BASE_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getBs02"
DEFAULT_OUTPUT_DIR = Path("artifacts/daegu_bis_api_audit")
MAX_ROUTES_HARD_CAP = 3

ROUTE_ID_ALIASES = {
    "routeid", "route_id", "routeId", "ROUTE_ID", "노선ID", "노선id", "노선아이디",
    "routeno", "routeNo", "ROUTE_NO", "노선번호",
}
DIRECTION_ALIASES = {
    "directionid", "direction_id", "directionId",
    "moveDir", "movedir", "MOVE_DIR", "MOVEDIR",
    "move_dir", "move_dir_code", "moveDirCode", "MOVE_DIR_CODE",
    "dir", "dirCode", "방면", "노선방면", "direction", "updown", "upDown",
}
STOP_ID_ALIASES = {
    "stopid", "stop_id", "stopId", "bsid", "bs_id", "bsId", "BUSSTOP_ID",
    "정류소ID", "정류장ID", "정류소아이디", "정류장아이디", "node_id", "nodeId",
}
STOP_NAME_ALIASES = {
    "stopname", "stop_name", "stopName", "bsname", "bs_name", "bsNm", "bsName",
    "정류소명", "정류장명", "정류소명칭", "정류장명칭",
}
SEQUENCE_ALIASES = {
    "seq", "sequence", "order", "ord", "stopseq", "stopSeq", "stop_sequence",
    "stopOrder", "bsseq", "bsSeq", "bs_order", "정류소순번", "정류장순번", "순번", "정렬순서",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def normalize_key(key: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(key)).lower()


def alias_lookup(row: Dict[str, Any], aliases: Iterable[str]) -> Tuple[Optional[str], Optional[Any]]:
    normalized_aliases = {normalize_key(a) for a in aliases}
    for k, v in row.items():
        if normalize_key(k) in normalized_aliases:
            return str(k), v
    return None, None


def coerce_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return None
    s = str(value).strip()
    return s if s else None


def coerce_int(value: Any) -> Optional[int]:
    s = coerce_scalar(value)
    if not s:
        return None
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def xml_to_obj(elem: ET.Element) -> Any:
    children = list(elem)
    if not children:
        return (elem.text or "").strip()

    grouped: Dict[str, List[Any]] = {}
    for child in children:
        tag = child.tag.split("}", 1)[-1]
        grouped.setdefault(tag, []).append(xml_to_obj(child))

    obj: Dict[str, Any] = {}
    for tag, values in grouped.items():
        obj[tag] = values[0] if len(values) == 1 else values
    return obj


def parse_raw_response(raw_text: str) -> Dict[str, Any]:
    stripped = raw_text.strip()
    if not stripped:
        return {"format": "empty", "parsed": None, "parse_error": None}

    try:
        return {"format": "json", "parsed": json.loads(stripped), "parse_error": None}
    except Exception as e_json:
        try:
            root = ET.fromstring(stripped)
            return {"format": "xml", "parsed": {root.tag.split('}', 1)[-1]: xml_to_obj(root)}, "parse_error": None}
        except Exception as e_xml:
            return {
                "format": "unknown",
                "parsed": None,
                "parse_error": f"json_error={e_json}; xml_error={e_xml}",
            }


def walk_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk_dicts(x)


def score_candidate_row(row: Dict[str, Any]) -> int:
    score = 0
    for aliases in (ROUTE_ID_ALIASES, DIRECTION_ALIASES, STOP_ID_ALIASES, STOP_NAME_ALIASES, SEQUENCE_ALIASES):
        k, v = alias_lookup(row, aliases)
        if k is not None and coerce_scalar(v) is not None:
            score += 1
    return score


def extract_candidate_rows(parsed: Any) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for d in walk_dicts(parsed):
        score = score_candidate_row(d)
        if score >= 2:
            marker = tuple(sorted((str(k), str(v)) for k, v in d.items() if not isinstance(v, (dict, list))))
            if marker not in seen:
                seen.add(marker)
                rows.append(d)
    rows.sort(key=score_candidate_row, reverse=True)
    return rows


def normalize_rows(route_id_requested: Optional[str], parsed: Any) -> Dict[str, Any]:
    candidate_rows = extract_candidate_rows(parsed)
    normalized: List[Dict[str, Any]] = []
    schema_hits: Dict[str, List[str]] = {
        "route_id": [],
        "direction_id": [],
        "stop_id": [],
        "stop_name": [],
        "stop_order": [],
    }

    for row in candidate_rows:
        route_key, route_val = alias_lookup(row, ROUTE_ID_ALIASES)
        dir_key, dir_val = alias_lookup(row, DIRECTION_ALIASES)
        stop_key, stop_val = alias_lookup(row, STOP_ID_ALIASES)
        name_key, name_val = alias_lookup(row, STOP_NAME_ALIASES)
        seq_key, seq_val = alias_lookup(row, SEQUENCE_ALIASES)

        route_id = coerce_scalar(route_val) or route_id_requested
        direction_id = coerce_scalar(dir_val)
        stop_id = coerce_scalar(stop_val)
        stop_name = coerce_scalar(name_val)
        stop_order = coerce_int(seq_val)

        for label, key in [
            ("route_id", route_key),
            ("direction_id", dir_key),
            ("stop_id", stop_key),
            ("stop_name", name_key),
            ("stop_order", seq_key),
        ]:
            if key and key not in schema_hits[label]:
                schema_hits[label].append(key)

        if stop_id or stop_name:
            normalized.append({
                "route_id": route_id,
                "direction_id": direction_id,
                "stop_id": stop_id,
                "stop_name": stop_name,
                "stop_order": stop_order,
                "raw_keys": sorted([str(k) for k in row.keys()]),
            })

    has_route_id = any(r.get("route_id") for r in normalized)
    has_direction_id = any(r.get("direction_id") for r in normalized)
    has_stop_id = any(r.get("stop_id") for r in normalized)
    has_sequence = any(r.get("stop_order") is not None for r in normalized)

    if has_sequence:
        normalized.sort(
            key=lambda r: (
                str(r.get("direction_id") or ""),
                10**9 if r.get("stop_order") is None else int(r["stop_order"]),
                str(r.get("stop_id") or ""),
            )
        )

    return {
        "candidate_row_count": len(candidate_rows),
        "normalized_row_count": len(normalized),
        "schema_hits": schema_hits,
        "has_route_id": has_route_id,
        "has_direction_id": has_direction_id,
        "has_stop_id": has_stop_id,
        "has_sequence": has_sequence,
        "ordered_stop_sequence_possible": bool(has_stop_id and has_sequence),
        "normalized_preview": normalized[:50],
    }


def build_url(base_url: str, service_key: str, route_id: str, route_param: str, extra_params: Sequence[str]) -> str:
    params: Dict[str, str] = {
        "serviceKey": service_key,
        route_param: route_id,
    }
    for p in extra_params:
        if "=" not in p:
            raise ValueError(f"--param must be key=value, got: {p}")
        k, v = p.split("=", 1)
        params[k] = v
    sep = "&" if "?" in base_url else "?"
    return base_url + sep + urllib.parse.urlencode(params, doseq=True)


def fetch_api(base_url: str, service_key: str, route_id: str, route_param: str, extra_params: Sequence[str], timeout: int) -> Dict[str, Any]:
    url = build_url(base_url, service_key, route_id, route_param, extra_params)
    redacted_url = url.replace(urllib.parse.quote(service_key, safe=""), "<SERVICE_KEY>").replace(service_key, "<SERVICE_KEY>")
    req = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-step99a-audit/1.0"})

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
        "route_id": route_id,
        "request_url_redacted": redacted_url,
        "http_status": status_code,
        "http_error_reason": error_reason,
        "api_call_ok": bool(status_code is not None and 200 <= status_code < 300),
        "content_type": content_type,
        "raw_text": raw_text,
    }


SELF_TEST_RAW = json.dumps({
    "body": {
        "items": [
            {"routeId": "R_TEST", "moveDirCode": "0", "bsId": "STOP_A", "bsNm": "Alpha", "bsSeq": "1"},
            {"routeId": "R_TEST", "moveDirCode": "0", "bsId": "STOP_B", "bsNm": "Beta", "bsSeq": "2"},
            {"routeId": "R_TEST", "moveDirCode": "1", "bsId": "STOP_C", "bsNm": "Gamma", "bsSeq": "1"},
        ]
    }
}, ensure_ascii=False)


def sample_route_ids_from_args(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        for part in str(v).split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out


def quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"unsafe identifier: {name}")
    return '"' + name.replace('"', '""') + '"'


def db_connect(dsn: str):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed; DB coverage skipped")
    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def relation_exists(conn: Any, schema: str, relation: str) -> bool:
    sql = """
    SELECT 1
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = %s
      AND c.relname = %s
      AND c.relkind IN ('r', 'v', 'm')
    LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, relation))
        return cur.fetchone() is not None


def get_columns(conn: Any, schema: str, relation: str) -> List[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, relation))
        return [r[0] for r in cur.fetchall()]


def pick_stop_id_columns(columns: Sequence[str]) -> List[str]:
    normalized = {normalize_key(c): c for c in columns}
    picks: List[str] = []
    for alias in ["stop_id", "stopid", "bs_id", "bsid", "node_uid"]:
        key = normalize_key(alias)
        if key in normalized and normalized[key] not in picks:
            picks.append(normalized[key])
    return picks


def compute_db_coverage(dsn: Optional[str], normalized_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stop_ids = sorted({str(r.get("stop_id")) for r in normalized_rows if r.get("stop_id")})
    if not dsn:
        return {"enabled": False, "reason": "no --db-dsn or URBANBUS_DB_DSN provided"}
    if not stop_ids:
        return {"enabled": False, "reason": "no stop_id values in normalized preview"}

    out = {
        "enabled": True,
        "read_only": True,
        "stop_id_count": len(stop_ids),
        "relations": {},
        "best_match_relation": None,
        "best_coverage_pct": None,
    }
    try:
        conn = db_connect(dsn)
    except Exception as e:
        out["enabled"] = False
        out["reason"] = f"DB connection failed: {e}"
        return out

    try:
        for schema, relation in [
            ("public", "gatv2_node_master_active"),
            ("public", "graph_node_master"),
            ("public", "dim_stop"),
        ]:
            rel_key = f"{schema}.{relation}"
            if not relation_exists(conn, schema, relation):
                out["relations"][rel_key] = {"exists": False}
                continue
            cols = get_columns(conn, schema, relation)
            stop_cols = pick_stop_id_columns(cols)
            rel_info: Dict[str, Any] = {"exists": True, "columns_checked": stop_cols, "coverage_by_column": {}}
            for col in stop_cols:
                q_schema = quote_ident(schema)
                q_relation = quote_ident(relation)
                q_col = quote_ident(col)
                if normalize_key(col) == normalize_key("node_uid"):
                    values = [f"STOP:{x}" for x in stop_ids]
                else:
                    values = stop_ids
                sql = f"SELECT COUNT(DISTINCT {q_col}::text) FROM {q_schema}.{q_relation} WHERE {q_col}::text = ANY(%s)"
                with conn.cursor() as cur:
                    cur.execute(sql, (values,))
                    matched = int(cur.fetchone()[0] or 0)
                coverage = matched / max(1, len(stop_ids)) * 100.0
                rel_info["coverage_by_column"][col] = {"matched": matched, "coverage_pct": round(coverage, 4)}
                if out["best_coverage_pct"] is None or coverage > float(out["best_coverage_pct"]):
                    out["best_coverage_pct"] = round(coverage, 4)
                    out["best_match_relation"] = f"{rel_key}.{col}"
            out["relations"][rel_key] = rel_info
    except Exception as e:
        out["error"] = str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def classify_fields(summary: Dict[str, Any], db_coverage: Dict[str, Any]) -> Dict[str, Any]:
    has_route_id = bool(summary.get("has_route_id"))
    has_direction_id = bool(summary.get("has_direction_id"))
    has_stop_id = bool(summary.get("has_stop_id"))
    has_sequence = bool(summary.get("has_sequence"))
    ordered_possible = bool(summary.get("ordered_stop_sequence_possible"))
    coverage = db_coverage.get("best_coverage_pct") if db_coverage.get("enabled") else None

    def with_reason(status: str, reason: str) -> Dict[str, str]:
        return {"candidate_status": status, "reason": reason}

    route_id = with_reason(
        "observed_candidate" if has_route_id else "missing_or_request_only",
        "route_id-like field found in API response" if has_route_id else "route_id not found inside response; may only be request parameter",
    )
    direction_id = with_reason(
        "observed_candidate" if has_direction_id else "missing",
        "direction/move_dir/방면-like field found" if has_direction_id else "no direction-equivalent field detected",
    )
    if has_direction_id and not has_route_id:
        direction_id["candidate_status"] = "partial_observed_candidate"

    if ordered_possible:
        seq_status = "observed_candidate"
        seq_reason = "stop_id and stop_order/sequence-like field both detected"
    elif has_stop_id and not has_sequence:
        seq_status = "missing"
        seq_reason = "stop_id detected but no order/sequence field detected; ordered_stop_sequence cannot be derived safely"
    else:
        seq_status = "missing"
        seq_reason = "stop_id and sequence fields not both detected"

    if coverage is not None and coverage < 80.0:
        seq_reason += f"; graph node coverage is low ({coverage:.2f}%), so do not promote to v2 required contract yet"

    return {
        "route_id": route_id,
        "direction_id": direction_id,
        "ordered_stop_sequence": with_reason(seq_status, seq_reason),
    }


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    fc = report.get("field_classification_candidates", {})
    api = report.get("api_call_summary", {})
    db = report.get("db_coverage", {})
    lines = [
        "# Step 99-A — /getBs02 Route-Stop Sequence Acquisition Feasibility Audit",
        "",
        f"- artifact_version: `{report.get('artifact_version')}`",
        f"- created_at_utc: `{report.get('created_at_utc')}`",
        f"- mode: `{report.get('mode')}`",
        f"- api_called: `{api.get('api_called')}`",
        f"- routes_requested: `{api.get('routes_requested')}`",
        f"- paper_level_claim_allowed: `{report.get('paper_level_claim_allowed')}`",
        f"- causal_performance_claim_allowed: `{report.get('causal_performance_claim_allowed')}`",
        f"- fleet_reduction_claim_allowed: `{report.get('fleet_reduction_claim_allowed')}`",
        "",
        "## Field classification update candidates",
        "",
        "| Field | Candidate status | Reason |",
        "|---|---|---|",
    ]
    for k in ["route_id", "direction_id", "ordered_stop_sequence"]:
        item = fc.get(k, {})
        lines.append(f"| `{k}` | `{item.get('candidate_status')}` | {item.get('reason', '')} |")
    lines.extend([
        "",
        "## API schema summary",
        "",
        f"- response_format_counts: `{report.get('response_format_counts')}`",
        f"- total_normalized_rows: `{report.get('total_normalized_rows')}`",
        f"- ordered_stop_sequence_possible_any: `{report.get('ordered_stop_sequence_possible_any')}`",
        "",
        "## Graph node coverage",
        "",
        f"- db_coverage_enabled: `{db.get('enabled')}`",
        f"- best_match_relation: `{db.get('best_match_relation')}`",
        f"- best_coverage_pct: `{db.get('best_coverage_pct')}`",
        f"- reason/error: `{db.get('reason') or db.get('error') or ''}`",
        "",
        "## Guardrail status",
        "",
        "- DB write: forbidden and not performed",
        "- tensor DB overwrite: forbidden and not performed",
        "- API bulk collection: forbidden and not performed",
        "- max route hard cap: 3",
        "- actual collection scope: sample-only feasibility audit",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_route_ids = sample_route_ids_from_args(args.route_id or [])
    max_routes = min(max(1, int(args.max_routes)), MAX_ROUTES_HARD_CAP)
    requested_route_ids = requested_route_ids[:max_routes]

    service_key = args.service_key or os.getenv("DAEGU_BIS_SERVICE_KEY") or os.getenv("DATAGO_SERVICE_KEY")
    db_dsn = args.db_dsn or os.getenv("URBANBUS_DB_DSN")

    use_self_test_payload = bool(args.self_test) or not service_key or not requested_route_ids
    raw_records: List[Dict[str, Any]] = []

    if use_self_test_payload:
        raw_records.append({
            "route_id": "R_TEST",
            "request_url_redacted": None,
            "content_type": "application/json; self-test",
            "raw_text": SELF_TEST_RAW,
        })
    else:
        for i, route_id in enumerate(requested_route_ids, start=1):
            if i > 1 and args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
            raw_records.append(fetch_api(
                base_url=args.base_url,
                service_key=service_key,
                route_id=route_id,
                route_param=args.route_param,
                extra_params=args.param or [],
                timeout=int(args.timeout),
            ))

    per_route = []
    all_normalized: List[Dict[str, Any]] = []
    format_counts: Dict[str, int] = {}

    sample_payload = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "raw_response_storage_note": "raw_text preserves the raw JSON/XML body exactly as decoded from HTTP bytes",
        "records": [],
    }

    for idx, record in enumerate(raw_records, start=1):
        raw_text = record["raw_text"]
        raw_path = output_dir / f"getbs02_sample_response_route_{record.get('route_id') or idx}.raw"
        raw_path.write_text(raw_text, encoding="utf-8")

        parsed_info = parse_raw_response(raw_text)
        fmt = parsed_info["format"]
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        normalized_summary = normalize_rows(record.get("route_id"), parsed_info["parsed"])
        all_normalized.extend(normalized_summary["normalized_preview"])

        per_route.append({
            "route_id": record.get("route_id"),
            "request_url_redacted": record.get("request_url_redacted"),
            "content_type": record.get("content_type"),
            "raw_path": str(raw_path),
            "format": fmt,
            "parse_error": parsed_info.get("parse_error"),
            **normalized_summary,
        })
        sample_payload["records"].append({
            "route_id": record.get("route_id"),
            "request_url_redacted": record.get("request_url_redacted"),
            "content_type": record.get("content_type"),
            "raw_path": str(raw_path),
            "raw_text": raw_text,
            "parsed_format": fmt,
            "parsed": parsed_info["parsed"],
        })

    db_coverage = compute_db_coverage(db_dsn, all_normalized)

    merged_summary = {
        "has_route_id": any(x.get("has_route_id") for x in per_route),
        "has_direction_id": any(x.get("has_direction_id") for x in per_route),
        "has_stop_id": any(x.get("has_stop_id") for x in per_route),
        "has_sequence": any(x.get("has_sequence") for x in per_route),
        "ordered_stop_sequence_possible": any(x.get("ordered_stop_sequence_possible") for x in per_route),
    }

    report = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now_iso(),
        "step": "Step 99-A",
        "title": "/getBs02 route-stop sequence acquisition feasibility audit",
        "mode": "self_test_or_schema_only" if use_self_test_payload else "sample_api_call",
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "fleet_reduction_claim_allowed": False,
        "guardrails": {
            "db_write_forbidden": True,
            "tensor_db_overwrite_forbidden": True,
            "csv_reload_forbidden": True,
            "bulk_api_collection_forbidden": True,
            "max_routes_hard_cap": MAX_ROUTES_HARD_CAP,
        },
        "api_call_summary": {
            "api_called": not use_self_test_payload,
            "base_url": args.base_url,
            "base_url_default_may_require_confirmation": args.base_url == DEFAULT_BASE_URL,
            "route_param": args.route_param,
            "routes_requested": requested_route_ids,
            "service_key_source": "arg/env" if service_key else None,
            "skip_reason": None if not use_self_test_payload else (
                "self-test requested" if args.self_test else "missing service key or route id"
            ),
            "daily_limit_guard": "sample-only; max_routes<=3; no bulk collection",
        },
        "response_format_counts": format_counts,
        "route_reports": per_route,
        "total_normalized_rows": len(all_normalized),
        "ordered_stop_sequence_possible_any": merged_summary["ordered_stop_sequence_possible"],
        "db_coverage": db_coverage,
        "field_classification_candidates": classify_fields(merged_summary, db_coverage),
        "next_step_recommendation": (
            "If actual API sample confirms stop_id+sequence and graph coverage is acceptable, "
            "write Step 97 classification update before making route-aware fields required in Step 98."
        ),
    }

    dump_json(output_dir / "getbs02_sample_response.json", sample_payload)
    dump_json(output_dir / "getbs02_route_stop_sequence_audit_report.json", report)
    write_markdown_report(output_dir / "getbs02_route_stop_sequence_audit_report.md", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 99-A /getBs02 route-stop sequence audit")
    p.add_argument("--self-test", action="store_true", help="run without API key using embedded schema sample")
    p.add_argument("--service-key", default="", help="data.go.kr service key; otherwise DAEGU_BIS_SERVICE_KEY or DATAGO_SERVICE_KEY")
    p.add_argument("--route-id", action="append", default=[], help="sample route id; may be repeated or comma-separated")
    p.add_argument("--max-routes", type=int, default=3, help="sample route cap; hard-capped at 3")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="/getBs02 endpoint URL; override from Swagger if needed")
    p.add_argument("--route-param", default="routeId", help="route id query parameter name")
    p.add_argument("--param", action="append", default=[], help="extra query parameter key=value; may be repeated")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--sleep-seconds", type=float, default=0.2)
    p.add_argument("--db-dsn", default="", help="optional read-only coverage check; otherwise URBANBUS_DB_DSN")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_routes) > MAX_ROUTES_HARD_CAP:
        print(f"[WARN] --max-routes={args.max_routes} reduced to hard cap {MAX_ROUTES_HARD_CAP}")
    report = run_audit(args)
    out_dir = Path(args.output_dir)
    print("[OK] Step 99-A /getBs02 audit complete")
    print(f"[OK] mode       : {report['mode']}")
    print(f"[OK] api_called : {report['api_call_summary']['api_called']}")
    print(f"[OK] normalized : {report['total_normalized_rows']}")
    print(f"[OK] sequence   : {report['ordered_stop_sequence_possible_any']}")
    print(f"[OK] json report: {out_dir / 'getbs02_route_stop_sequence_audit_report.json'}")
    print(f"[OK] md report  : {out_dir / 'getbs02_route_stop_sequence_audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
