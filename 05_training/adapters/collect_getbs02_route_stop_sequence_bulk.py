from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import psycopg2
    import psycopg2.extras
except Exception:  # pragma: no cover
    psycopg2 = None

ARTIFACT_VERSION = "daegu_bis_getbs02_bulk_collect_step99a_v1"
DEFAULT_BASE_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getBs02"
DEFAULT_OUT_ROOT = "artifacts/daegu_bis_api_audit/getbs02_bulk_collect"
MAX_DAILY_LIMIT_GUARD = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_extra_params(values: Iterable[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for v in values or []:
        if not v:
            continue
        if "=" not in v:
            raise SystemExit(f"--extra-param must be key=value, got: {v}")
        k, val = v.split("=", 1)
        out.append((k.strip(), val.strip()))
    return out


def build_url(base_url: str, service_key: str, route_id: str, route_param: str, extra_params: List[Tuple[str, str]]) -> str:
    params = [("serviceKey", service_key), (route_param, route_id), *extra_params]
    return base_url + "?" + urllib.parse.urlencode(params)


def redact_url(url: str, service_key: str) -> str:
    return url.replace(urllib.parse.quote_plus(service_key), "<SERVICE_KEY>").replace(service_key, "<SERVICE_KEY>")


def detect_error(raw_text: str, http_status: Optional[int]) -> str:
    s = (raw_text or "").strip()
    low = s.lower()
    if http_status is not None and http_status >= 500:
        return "PROVIDER_ERROR"
    if not s:
        return "EMPTY_RESPONSE"
    if any(x in low for x in ["unexpected errors", "service error", "application error", "internal server error", "시스템 오류", "서비스 오류"]):
        return "PROVIDER_ERROR"
    if any(x in low for x in ["servicekey", "service key", "인증키", "등록되지 않은", "unauthorized", "not authorized", "invalid"]):
        return "AUTH_ERROR"
    return ""


def fetch_one(base_url: str, service_key: str, route_id: str, route_param: str, extra_params: List[Tuple[str, str]], timeout: int) -> Dict[str, Any]:
    url = build_url(base_url, service_key, route_id, route_param, extra_params)
    req = urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-step99a-getbs02-bulk/1.0"})
    status_code: Optional[int] = None
    reason = ""
    raw_bytes = b""
    content_type = ""
    encoding = "utf-8"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = int(getattr(resp, "status", 200) or 200)
            raw_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            encoding = resp.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as e:
        status_code = int(e.code)
        reason = str(e.reason)
        raw_bytes = e.read() or b""
        content_type = e.headers.get("Content-Type", "") if e.headers else ""
        encoding = e.headers.get_content_charset() if e.headers else None
        encoding = encoding or "utf-8"
    except urllib.error.URLError as e:
        reason = str(e.reason)
        raw_bytes = str(e).encode("utf-8", errors="replace")
        content_type = "text/plain"
        encoding = "utf-8"

    raw_text = raw_bytes.decode(encoding, errors="replace")
    return {
        "route_id": route_id,
        "request_url_redacted": redact_url(url, service_key),
        "http_status": status_code,
        "http_error_reason": reason,
        "content_type": content_type,
        "raw_text": raw_text,
        "provider_status": detect_error(raw_text, status_code),
        "api_call_ok": bool(status_code is not None and 200 <= status_code < 300),
    }


def json_loads_maybe(raw_text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(raw_text), None
    except Exception as e:
        return None, str(e)


def get_path(obj: Any, path: List[str]) -> Any:
    cur = obj
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def find_items(obj: Any) -> List[Dict[str, Any]]:
    candidates = [
        ["body", "items"],
        ["response", "body", "items"],
        ["response", "body", "items", "item"],
        ["body", "items", "item"],
        ["items"],
    ]
    for path in candidates:
        v = get_path(obj, path)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            item = v.get("item")
            if isinstance(item, list):
                return [x for x in item if isinstance(x, dict)]
            if isinstance(item, dict):
                return [item]
    return []


def get_header_status(obj: Any) -> Dict[str, Any]:
    header = None
    if isinstance(obj, dict):
        header = obj.get("header") or get_path(obj, ["response", "header"])
    return header if isinstance(header, dict) else {}


def to_int_maybe(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        s = str(v).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def norm_str(v: Any) -> str:
    return "" if v is None else str(v).strip()


def normalize_items(route_id: str, route_no: str, route_type: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        stop_id = norm_str(item.get("bsId") or item.get("bs_id") or item.get("stopId") or item.get("stop_id"))
        stop_name = norm_str(item.get("bsNm") or item.get("bs_name") or item.get("stopName") or item.get("stop_name"))
        move_dir = norm_str(item.get("moveDir") or item.get("move_dir") or item.get("moveDirCode") or item.get("direction_id"))
        seq = to_int_maybe(item.get("seq") or item.get("stopSeq") or item.get("bsSeq") or item.get("order"))
        x_pos = norm_str(item.get("xPos") or item.get("x_pos") or item.get("lon") or item.get("longitude"))
        y_pos = norm_str(item.get("yPos") or item.get("y_pos") or item.get("lat") or item.get("latitude"))
        rows.append({
            "route_id": route_id,
            "route_no": route_no,
            "route_type": route_type,
            "direction_id": move_dir,
            "stop_order": seq if seq is not None else "",
            "stop_id": stop_id,
            "stop_name": stop_name,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "raw_keys": ",".join(sorted(item.keys())),
        })
    return rows


def get_conn(dsn: str):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed or importable")
    return psycopg2.connect(dsn)


def load_source_routes(dsn: str, max_routes: int = 0) -> List[Dict[str, str]]:
    sql = """
    SELECT DISTINCT route_id::text AS route_id,
           COALESCE(route_no::text, '') AS route_no,
           COALESCE(route_type::text, '') AS route_type
    FROM public.stg_daegu_routes
    WHERE route_id IS NOT NULL
    ORDER BY route_id::text
    """
    if max_routes and max_routes > 0:
        sql += f" LIMIT {int(max_routes)}"
    with get_conn(dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


def relation_exists(conn, schema: str, rel: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname=%s AND c.relname=%s AND c.relkind IN ('r','v','m') LIMIT 1
        """, (schema, rel))
        return cur.fetchone() is not None


def load_stop_sets(dsn: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    with get_conn(dsn) as conn:
        with conn.cursor() as cur:
            if relation_exists(conn, "public", "gatv2_node_master_active"):
                cur.execute("SELECT node_uid::text FROM public.gatv2_node_master_active")
                vals = {str(x[0]).replace("STOP:", "") for x in cur.fetchall()}
                out["public.gatv2_node_master_active.node_uid"] = vals
            if relation_exists(conn, "public", "graph_node_master"):
                cur.execute("SELECT node_uid::text FROM public.graph_node_master WHERE node_type='STOP'")
                vals = {str(x[0]).replace("STOP:", "") for x in cur.fetchall()}
                out["public.graph_node_master.node_uid"] = vals
            if relation_exists(conn, "public", "dim_stop"):
                # Try common stop id column names.
                cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema='public' AND table_name='dim_stop'
                ORDER BY ordinal_position
                """)
                cols = [r[0] for r in cur.fetchall()]
                for c in ["stop_id", "bs_id", "bsId", "node_id"]:
                    if c in cols:
                        cur.execute(f"SELECT {c}::text FROM public.dim_stop")
                        out[f"public.dim_stop.{c}"] = {str(x[0]) for x in cur.fetchall()}
                        break
    return out


def compute_coverage(rows: List[Dict[str, Any]], stop_sets: Dict[str, set]) -> Dict[str, Any]:
    stop_ids = sorted({str(r.get("stop_id") or "").strip() for r in rows if str(r.get("stop_id") or "").strip()})
    if not stop_ids:
        return {"enabled": bool(stop_sets), "reason": "no stop_id values", "by_relation": {}, "best_match_relation": None, "best_coverage_pct": None}
    by_rel: Dict[str, Any] = {}
    for rel, values in stop_sets.items():
        matched = sum(1 for s in stop_ids if s in values or f"STOP:{s}" in values)
        pct = round(100.0 * matched / len(stop_ids), 4)
        by_rel[rel] = {"unique_stop_ids": len(stop_ids), "matched_stop_ids": matched, "coverage_pct": pct}
    best_rel = None
    best_pct = None
    for rel, d in by_rel.items():
        if best_pct is None or d["coverage_pct"] > best_pct:
            best_rel = rel
            best_pct = d["coverage_pct"]
    return {"enabled": bool(stop_sets), "reason": "", "by_relation": by_rel, "best_match_relation": best_rel, "best_coverage_pct": best_pct}


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Step 99-A Full /getBs02 Route-Stop Sequence Collection Report")
    lines.append("")
    lines.append(f"- artifact_version: `{summary['artifact_version']}`")
    lines.append(f"- created_at_utc: `{summary['created_at_utc']}`")
    lines.append(f"- source_route_count: `{summary['source_route_count']}`")
    lines.append(f"- attempted_route_count: `{summary['attempted_route_count']}`")
    lines.append(f"- total_normalized_rows: `{summary['total_normalized_rows']}`")
    lines.append(f"- ordered_stop_sequence_possible_any: `{summary['ordered_stop_sequence_possible_any']}`")
    lines.append("")
    lines.append("## Status counts")
    lines.append("")
    lines.append("| collect_status | count |")
    lines.append("|---|---:|")
    for k, v in summary["collect_status_counts"].items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("")
    lines.append("## Field classification candidates")
    lines.append("")
    lines.append("| Field | Candidate status | Reason |")
    lines.append("|---|---|---|")
    lines.append("| `route_id` | `observed_full_collection_candidate` | request routeId propagated to every normalized row |")
    lines.append("| `direction_id` | `observed_full_collection_candidate` | moveDir parsed as direction_id when present |")
    lines.append("| `ordered_stop_sequence` | `observed_full_collection_candidate` | bsId + seq parsed into stop_id + stop_order |")
    lines.append("")
    cov = summary.get("graph_node_coverage", {})
    lines.append("## Graph node coverage")
    lines.append("")
    lines.append(f"- enabled: `{cov.get('enabled')}`")
    lines.append(f"- best_match_relation: `{cov.get('best_match_relation')}`")
    lines.append(f"- best_coverage_pct: `{cov.get('best_coverage_pct')}`")
    if cov.get("reason"):
        lines.append(f"- reason: `{cov.get('reason')}`")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for k, v in summary.get("output_files", {}).items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- DB write: forbidden and not performed")
    lines.append("- tensor DB overwrite: forbidden and not performed")
    lines.append("- 2023 CSV reload: forbidden and not performed")
    lines.append("- raw response snapshots: preserved")
    lines.append("- manifest collect_status and load_status: separated")
    lines.append("- load_status: `NOT_LOADED_STEP99A_AUDIT_ONLY`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Step 99-A full /getBs02 route-stop sequence collector, audit-only no DB writes")
    p.add_argument("--db-dsn", default=os.environ.get("URBANBUS_DB_DSN", ""))
    p.add_argument("--service-key", default=os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY") or "")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--route-param", default="routeId")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--run-id", default="")
    p.add_argument("--max-routes", type=int, default=0, help="0 means all source routes")
    p.add_argument("--sleep-sec", type=float, default=0.25)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--extra-param", action="append", default=[])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-over-1000", action="store_true")
    args = p.parse_args(argv)

    if not args.service_key:
        raise SystemExit("[STOP] service key missing. Set DAEGU_BIS_SERVICE_KEY or pass --service-key")
    if not args.db_dsn:
        raise SystemExit("[STOP] DB DSN missing. Set URBANBUS_DB_DSN or pass --db-dsn")

    extra_params = parse_extra_params(args.extra_param)
    if not any(k == "resultType" for k, _ in extra_params):
        # The endpoint worked without resultType in Swagger, but JSON is the full-collection contract.
        extra_params.append(("resultType", "json"))

    run_id = args.run_id or timestamp_tag()
    out_root = Path(args.out_root) / run_id
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    routes = load_source_routes(args.db_dsn, max_routes=args.max_routes)
    source_route_count = len(routes)
    if source_route_count == 0:
        raise SystemExit("[STOP] no source routes found in public.stg_daegu_routes")
    if source_route_count > MAX_DAILY_LIMIT_GUARD and not args.allow_over_1000:
        raise SystemExit(f"[STOP] route count {source_route_count} exceeds daily guard {MAX_DAILY_LIMIT_GUARD}; pass --allow-over-1000 only if quota permits")

    print("[INFO] Step 99-A full /getBs02 route-stop sequence collection")
    print(f"[INFO] base_url           : {args.base_url}")
    print(f"[INFO] source routes      : {source_route_count}")
    print(f"[INFO] output root        : {out_root}")
    print(f"[INFO] sleep_sec          : {args.sleep_sec}")
    print(f"[INFO] extra params       : {extra_params}")
    print("[INFO] DB write           : forbidden/not performed")

    manifest_rows: List[Dict[str, Any]] = []
    norm_rows: List[Dict[str, Any]] = []
    started = time.time()

    for i, r in enumerate(routes, 1):
        route_id = str(r["route_id"])
        route_no = str(r.get("route_no", ""))
        route_type = str(r.get("route_type", ""))
        raw_path = raw_dir / f"getbs02_{route_id}.json"
        call_started = utc_now()

        if args.resume and raw_path.exists():
            raw_text = read_text_any(raw_path)
            fetch = {
                "route_id": route_id,
                "request_url_redacted": "<resume_existing_raw>",
                "http_status": None,
                "http_error_reason": "",
                "content_type": "resume_existing_raw",
                "raw_text": raw_text,
                "provider_status": detect_error(raw_text, None),
                "api_call_ok": True,
            }
            api_called = False
        else:
            fetch = fetch_one(args.base_url, args.service_key, route_id, args.route_param, extra_params, args.timeout)
            raw_path.write_text(fetch["raw_text"], encoding="utf-8")
            api_called = True

        obj, parse_error = json_loads_maybe(fetch["raw_text"])
        items: List[Dict[str, Any]] = []
        header_status: Dict[str, Any] = {}
        if obj is not None:
            items = find_items(obj)
            header_status = get_header_status(obj)

        normalized = normalize_items(route_id, route_no, route_type, items)
        norm_rows.extend(normalized)

        has_seq = any(str(x.get("stop_order", "")).strip() for x in normalized)
        has_stop = any(str(x.get("stop_id", "")).strip() for x in normalized)
        has_dir = any(str(x.get("direction_id", "")).strip() for x in normalized)

        if fetch.get("provider_status"):
            collect_status = fetch["provider_status"]
        elif parse_error:
            collect_status = "PARSE_ERROR"
        elif not items:
            collect_status = "NO_ITEMS"
        elif has_stop and has_seq:
            collect_status = "SUCCESS"
        else:
            collect_status = "SCHEMA_INCOMPLETE"

        manifest_row = {
            "run_id": run_id,
            "route_id": route_id,
            "route_no": route_no,
            "route_type": route_type,
            "route_index": i,
            "source_route_count": source_route_count,
            "api_called": str(api_called).lower(),
            "http_status": fetch.get("http_status") if fetch.get("http_status") is not None else "",
            "content_type": fetch.get("content_type", ""),
            "header_success": header_status.get("success", ""),
            "header_result_code": header_status.get("resultCode", ""),
            "header_result_msg": header_status.get("resultMsg", ""),
            "candidate_row_count": len(items),
            "normalized_row_count": len(normalized),
            "has_stop_id": str(has_stop).lower(),
            "has_direction_id": str(has_dir).lower(),
            "has_sequence": str(has_seq).lower(),
            "ordered_stop_sequence_possible": str(bool(has_stop and has_seq)).lower(),
            "collect_status": collect_status,
            "load_status": "NOT_LOADED_STEP99A_AUDIT_ONLY",
            "parse_error": parse_error or "",
            "http_error_reason": fetch.get("http_error_reason", ""),
            "raw_path": str(raw_path),
            "request_url_redacted": fetch.get("request_url_redacted", ""),
            "called_at_utc": call_started,
        }
        manifest_rows.append(manifest_row)
        print(f"[{i:03d}/{source_route_count:03d}] route_id={route_id} route_no={route_no} status={collect_status} rows={len(normalized)}")

        if api_called and args.sleep_sec > 0 and i < source_route_count:
            time.sleep(args.sleep_sec)

    manifest_path = out_root / "manifest.csv"
    normalized_path = out_root / "getbs02_route_stop_sequence_normalized.csv"
    route_summary_path = out_root / "route_summary.csv"
    report_json_path = out_root / "getbs02_full_collection_report.json"
    report_md_path = out_root / "getbs02_full_collection_report.md"

    write_csv(manifest_path, manifest_rows, [
        "run_id", "route_id", "route_no", "route_type", "route_index", "source_route_count",
        "api_called", "http_status", "content_type", "header_success", "header_result_code", "header_result_msg",
        "candidate_row_count", "normalized_row_count", "has_stop_id", "has_direction_id", "has_sequence",
        "ordered_stop_sequence_possible", "collect_status", "load_status", "parse_error", "http_error_reason",
        "raw_path", "request_url_redacted", "called_at_utc",
    ])
    write_csv(normalized_path, norm_rows, [
        "route_id", "route_no", "route_type", "direction_id", "stop_order", "stop_id", "stop_name", "x_pos", "y_pos", "raw_keys",
    ])

    route_summary: List[Dict[str, Any]] = []
    by_route: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in norm_rows:
        by_route[str(row["route_id"])].append(row)
    for r in routes:
        rid = str(r["route_id"])
        rows = by_route.get(rid, [])
        dirs = sorted({str(x.get("direction_id") or "") for x in rows if str(x.get("direction_id") or "")})
        orders = [to_int_maybe(x.get("stop_order")) for x in rows]
        orders_i = [x for x in orders if x is not None]
        route_summary.append({
            "route_id": rid,
            "route_no": r.get("route_no", ""),
            "route_type": r.get("route_type", ""),
            "normalized_row_count": len(rows),
            "unique_stop_count": len({str(x.get("stop_id") or "") for x in rows if str(x.get("stop_id") or "")}),
            "direction_values": ";".join(dirs),
            "min_stop_order": min(orders_i) if orders_i else "",
            "max_stop_order": max(orders_i) if orders_i else "",
            "ordered_stop_sequence_possible": str(bool(rows and orders_i)).lower(),
        })
    write_csv(route_summary_path, route_summary, [
        "route_id", "route_no", "route_type", "normalized_row_count", "unique_stop_count", "direction_values", "min_stop_order", "max_stop_order", "ordered_stop_sequence_possible",
    ])

    try:
        stop_sets = load_stop_sets(args.db_dsn)
        coverage = compute_coverage(norm_rows, stop_sets)
    except Exception as e:
        coverage = {"enabled": False, "reason": str(e), "by_relation": {}, "best_match_relation": None, "best_coverage_pct": None}

    status_counts = dict(Counter([x["collect_status"] for x in manifest_rows]))
    elapsed = round(time.time() - started, 3)
    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "base_url": args.base_url,
        "route_param": args.route_param,
        "source_route_count": source_route_count,
        "attempted_route_count": len(manifest_rows),
        "collect_status_counts": status_counts,
        "total_normalized_rows": len(norm_rows),
        "ordered_stop_sequence_possible_any": any(x.get("ordered_stop_sequence_possible") == "true" for x in manifest_rows),
        "routes_with_ordered_sequence": sum(1 for x in manifest_rows if x.get("ordered_stop_sequence_possible") == "true"),
        "graph_node_coverage": coverage,
        "guardrails": {
            "db_write_forbidden": True,
            "db_write_performed": False,
            "tensor_db_overwrite_forbidden": True,
            "tensor_db_overwrite_performed": False,
            "csv_reload_forbidden": True,
            "bulk_collection_requested_by_user": True,
            "daily_limit_guard": MAX_DAILY_LIMIT_GUARD,
            "source_routes_under_daily_limit": source_route_count <= MAX_DAILY_LIMIT_GUARD,
        },
        "output_files": {
            "manifest_csv": str(manifest_path),
            "normalized_csv": str(normalized_path),
            "route_summary_csv": str(route_summary_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
            "raw_dir": str(raw_dir),
        },
        "elapsed_seconds": elapsed,
    }
    write_json(report_json_path, summary)
    write_md(report_md_path, summary)

    print("[OK] Step 99-A full /getBs02 collection complete")
    print(f"[OK] attempted routes : {len(manifest_rows)}")
    print(f"[OK] status counts    : {status_counts}")
    print(f"[OK] normalized rows  : {len(norm_rows)}")
    print(f"[OK] manifest         : {manifest_path}")
    print(f"[OK] normalized csv   : {normalized_path}")
    print(f"[OK] report md        : {report_md_path}")
    if status_counts.get("AUTH_ERROR", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
