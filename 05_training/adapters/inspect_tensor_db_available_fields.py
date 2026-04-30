from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

RELATION_CANDIDATES = [
    "public.gatv2_node_master_active",
    "public.gatv2_edge_primary_active",
    "public.gatv2_edge_primary_active_mat",
    "public.gatv2_snapshot_stop_features_train",
    "public.gatv2_snapshot_stop_features_train_mat",
    "public.graph_state_timeslice",
    "public.rl_state_training_base",
    "public.fact_stop_usage_hourly",
    "public.stg_daegu_stop_usage_2023",
]

TIME_COLUMN_CANDIDATES = [
    "state_ts",
    "next_state_ts",
    "service_date",
    "usage_date",
    "date",
    "base_date",
]

DDL_DML_PATTERN = re.compile(
    r"\b(UPDATE|INSERT|DELETE|TRUNCATE|CREATE|DROP|ALTER|MERGE|GRANT|REVOKE|VACUUM|ANALYZE)\b",
    re.IGNORECASE,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def requirements_path() -> Path:
    return Path(__file__).resolve().with_name("tensor_db_data_requirements_v2.json")


def json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_requirements(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or requirements_path()
    if not p.exists():
        raise FileNotFoundError(f"requirements JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def split_relation(label: str) -> Tuple[str, str]:
    if "." not in label:
        raise ValueError(f"relation must be schema.name: {label}")
    schema, name = label.split(".", 1)
    return schema, name


def assert_read_only_query(query: str) -> None:
    if DDL_DML_PATTERN.search(query):
        raise RuntimeError(f"non-read-only SQL rejected: {query}")


def query_all(conn: Any, query: Any, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    import psycopg2.extras

    text = query.as_string(conn) if hasattr(query, "as_string") else str(query)
    assert_read_only_query(text)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def query_one(conn: Any, query: Any, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
    rows = query_all(conn, query, params)
    return rows[0] if rows else None


def relation_exists(conn: Any, relation: str) -> Dict[str, Any]:
    schema, name = split_relation(relation)
    sql = """
    SELECT c.relkind::text AS relkind
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = %s
      AND c.relname = %s
      AND c.relkind IN ('r', 'p', 'v', 'm')
    LIMIT 1
    """
    row = query_one(conn, sql, (schema, name))
    return {"exists": row is not None, "relkind": row["relkind"] if row else None}


def relation_columns(conn: Any, relation: str) -> List[Dict[str, Any]]:
    schema, name = split_relation(relation)
    sql = """
    SELECT ordinal_position, column_name, data_type, udt_name, is_nullable
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position
    """
    return query_all(conn, sql, (schema, name))


def inspect_relation(conn: Any, relation: str, include_row_counts: bool = True) -> Dict[str, Any]:
    from psycopg2 import sql

    schema, name = split_relation(relation)
    exists_info = relation_exists(conn, relation)
    result: Dict[str, Any] = {
        "relation": relation,
        "exists": exists_info["exists"],
        "relkind": exists_info["relkind"],
        "columns": [],
        "row_count": None,
        "time_ranges": [],
        "inspection_note": "read-only metadata inspection",
    }
    if not exists_info["exists"]:
        return result

    columns = relation_columns(conn, relation)
    result["columns"] = columns
    column_names = [str(c["column_name"]) for c in columns]

    if not include_row_counts:
        result["inspection_note"] = "metadata only; row counts and ranges skipped"
        return result

    table_ref = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(name))
    count_query = sql.SQL("SELECT COUNT(*)::bigint AS row_count FROM {}").format(table_ref)
    row = query_one(conn, count_query)
    result["row_count"] = int(row["row_count"]) if row else None

    for col in TIME_COLUMN_CANDIDATES:
        if col not in column_names:
            continue
        range_query = sql.SQL(
            "SELECT "
            "COUNT(DISTINCT {col})::bigint AS distinct_count, "
            "MIN({col})::text AS min_value, "
            "MAX({col})::text AS max_value "
            "FROM {table}"
        ).format(col=sql.Identifier(col), table=table_ref)
        stats = query_one(conn, range_query)
        result["time_ranges"].append({"column": col, **(stats or {})})

    return result


def normalize_columns(relations: Iterable[Dict[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for rel in relations:
        cols = {str(c.get("column_name", "")).lower() for c in rel.get("columns", [])}
        out[str(rel.get("relation"))] = cols
    return out


def evaluate_candidate_matches(requirements: Dict[str, Any], relation_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relation_cols = normalize_columns(relation_reports)
    existing_relations = {str(r["relation"]) for r in relation_reports if r.get("exists")}

    evaluations: List[Dict[str, Any]] = []
    for field in requirements.get("candidate_fields", []):
        source_relations = [str(x) for x in field.get("source_relation", [])]
        source_columns = [str(x) for x in field.get("source_columns", [])]
        matched: List[Dict[str, Any]] = []
        for rel in source_relations:
            if rel not in existing_relations:
                continue
            cols = relation_cols.get(rel, set())
            hit_cols = [c for c in source_columns if c.lower() in cols]
            if hit_cols:
                matched.append({"relation": rel, "columns": hit_cols})

        effective_class = field.get("availability_class")
        source_match_found = bool(matched)
        if effective_class in {"observed", "derived", "proxy"} and not source_match_found and source_relations:
            # Keep design-time class, but expose actual source mismatch.
            match_status = "declared_source_not_found_or_columns_missing"
        elif effective_class in {"missing", "assumed"}:
            match_status = "not_expected_from_tensor_source"
        else:
            match_status = "matched" if source_match_found else "not_matched"

        evaluations.append({
            "field_name": field.get("field_name"),
            "declared_availability_class": effective_class,
            "source_match_found": source_match_found,
            "matched_sources": matched,
            "match_status": match_status,
            "is_allowed_in_v2_required_contract": bool(field.get("is_allowed_in_v2_required_contract")) and effective_class in {"observed", "derived"},
            "is_allowed_in_reward": bool(field.get("is_allowed_in_reward")),
            "is_allowed_in_kpi": bool(field.get("is_allowed_in_kpi")),
            "proxy_kpi": bool(field.get("proxy_kpi")),
            "observed_kpi": bool(field.get("observed_kpi")),
            "paper_level_claim_allowed": bool(field.get("paper_level_claim_allowed")) and effective_class in {"observed", "derived"},
            "causal_performance_claim_allowed": bool(field.get("causal_performance_claim_allowed")) and effective_class in {"observed", "derived"},
            "missing_reason": field.get("missing_reason", ""),
            "required_source_data": field.get("required_source_data", ""),
            "acquisition_path": field.get("acquisition_path", ""),
            "notes": field.get("notes", ""),
        })
    return evaluations


def build_handoff_buckets(field_evaluations: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    buckets = {
        "observed_or_derived_required_candidates": [],
        "proxy_optional_candidates": [],
        "assumed_or_missing_excluded_candidates": [],
        "missing_data_for_step99": [],
    }
    for f in field_evaluations:
        name = str(f["field_name"])
        cls = str(f["declared_availability_class"])
        if cls in {"observed", "derived"} and f.get("is_allowed_in_v2_required_contract"):
            buckets["observed_or_derived_required_candidates"].append(name)
        elif cls == "proxy":
            buckets["proxy_optional_candidates"].append(name)
        elif cls in {"assumed", "missing"}:
            buckets["assumed_or_missing_excluded_candidates"].append(name)
        if cls == "missing" or f.get("acquisition_path") in {"current_original_db", "external_data_required", "not_available"}:
            buckets["missing_data_for_step99"].append(name)
    return buckets


def build_report(requirements: Dict[str, Any], relation_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    field_evaluations = evaluate_candidate_matches(requirements, relation_reports)
    return {
        "artifact_version": "tensor_db_available_fields_report_step97_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_session": True,
        "requirements_artifact_version": requirements.get("artifact_version"),
        "guardrails": {
            "performance_claim_allowed": False,
            "paper_level_claim_allowed_for_proxy_assumed_missing": False,
            "causal_performance_claim_allowed_for_proxy_assumed_missing": False,
            "fleet_reduction_claim_allowed": False,
            "no_2023_csv_reimport": True,
            "no_tensor_db_overwrite": True,
            "no_2022_merge_into_2023_tensor_db": True,
        },
        "relation_reports": relation_reports,
        "field_evaluations": field_evaluations,
        "step98_handoff": build_handoff_buckets(field_evaluations),
        "final_note": "Step 97 is an audit gate only. It does not make performance claims.",
    }


def generate_markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Step 97 — Tensor DB Available Fields Report")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report.get('generated_at_utc')}`")
    lines.append("- read_only_session: `true`")
    lines.append("- performance_claim_allowed: `false`")
    lines.append("- causal_performance_claim_allowed_for_proxy_assumed_missing: `false`")
    lines.append("")
    lines.append("## Relation reports")
    lines.append("")
    lines.append("| Relation | Exists | Relkind | Row count | Column count | Time ranges |")
    lines.append("|---|---:|---|---:|---:|---|")
    for rel in report.get("relation_reports", []):
        ranges = []
        for t in rel.get("time_ranges", []):
            ranges.append(f"{t.get('column')}: {t.get('min_value')} ~ {t.get('max_value')}")
        lines.append(
            f"| `{rel.get('relation')}` | {rel.get('exists')} | {rel.get('relkind')} | "
            f"{rel.get('row_count')} | {len(rel.get('columns', []))} | {'; '.join(ranges)} |"
        )
    lines.append("")
    lines.append("## Field evaluations")
    lines.append("")
    lines.append("| Field | Class | Source matched | Required candidate | Proxy KPI | Paper claim | Step 99? |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    missing_set = set(report.get("step98_handoff", {}).get("missing_data_for_step99", []))
    for f in report.get("field_evaluations", []):
        name = f.get("field_name")
        lines.append(
            f"| `{name}` | {f.get('declared_availability_class')} | {f.get('source_match_found')} | "
            f"{f.get('is_allowed_in_v2_required_contract')} | {f.get('proxy_kpi')} | "
            f"{f.get('paper_level_claim_allowed')} | {name in missing_set} |"
        )
    lines.append("")
    lines.append("## Step 98 handoff")
    lines.append("")
    for key, values in report.get("step98_handoff", {}).items():
        lines.append(f"### {key}")
        lines.append("")
        for value in values:
            lines.append(f"- `{value}`")
        lines.append("")
    lines.append("## Guardrail")
    lines.append("")
    lines.append("This report is for data availability and generability audit only. It does not support performance, fleet-reduction, or causal performance claims.")
    return "\n".join(lines).rstrip() + "\n"


def connect_readonly(db_url: str) -> Any:
    import psycopg2

    conn = psycopg2.connect(db_url)
    conn.set_session(readonly=True, autocommit=False)
    return conn


def run_actual_inspection(db_url: str, output_root: Path, include_row_counts: bool = True) -> Dict[str, Any]:
    requirements = load_requirements()
    conn = connect_readonly(db_url)
    try:
        relation_reports = [inspect_relation(conn, rel, include_row_counts=include_row_counts) for rel in RELATION_CANDIDATES]
        report = build_report(requirements, relation_reports)
        output_root.mkdir(parents=True, exist_ok=True)
        dump_json(output_root / "tensor_db_available_fields_report.json", report)
        (output_root / "tensor_db_available_fields_report.md").write_text(generate_markdown_report(report), encoding="utf-8")
        conn.rollback()
        return report
    finally:
        try:
            conn.rollback()
        finally:
            conn.close()


def run_self_test(output_root: Optional[Path] = None) -> Dict[str, Any]:
    requirements = load_requirements()
    fake_relations = [
        {
            "relation": "public.gatv2_node_master_active",
            "exists": True,
            "relkind": "v",
            "columns": [
                {"column_name": "node_uid", "data_type": "text"},
                {"column_name": "node_index", "data_type": "integer"},
            ],
            "row_count": 4116,
            "time_ranges": [],
        },
        {
            "relation": "public.gatv2_edge_primary_active_mat",
            "exists": True,
            "relkind": "r",
            "columns": [
                {"column_name": "src_idx", "data_type": "integer"},
                {"column_name": "dst_idx", "data_type": "integer"},
                {"column_name": "distance_m", "data_type": "numeric"},
                {"column_name": "time_sec", "data_type": "numeric"},
            ],
            "row_count": 5484,
            "time_ranges": [],
        },
        {
            "relation": "public.gatv2_snapshot_stop_features_train_mat",
            "exists": True,
            "relkind": "r",
            "columns": [
                {"column_name": "state_ts", "data_type": "timestamp with time zone"},
                {"column_name": "next_state_ts", "data_type": "timestamp with time zone"},
                {"column_name": "node_index", "data_type": "integer"},
                {"column_name": "boardings_recent_log", "data_type": "numeric"},
                {"column_name": "alightings_recent_log", "data_type": "numeric"},
                {"column_name": "waiting_passenger_cnt_log", "data_type": "numeric"},
                {"column_name": "next_boardings_recent", "data_type": "numeric"},
                {"column_name": "next_alightings_recent", "data_type": "numeric"},
                {"column_name": "next_waiting_passenger_cnt", "data_type": "numeric"},
                {"column_name": "is_peak", "data_type": "boolean"},
            ],
            "row_count": 20289600,
            "time_ranges": [
                {"column": "state_ts", "min_value": "2023-01-01 05:00:00+09", "max_value": "2023-12-31 22:00:00+09"}
            ],
        },
    ]
    report = build_report(requirements, fake_relations)

    # Guardrail checks in self-test.
    for field in report["field_evaluations"]:
        cls = field["declared_availability_class"]
        if cls in {"proxy", "assumed", "missing"}:
            assert field["paper_level_claim_allowed"] is False
            assert field["causal_performance_claim_allowed"] is False
        if field["is_allowed_in_v2_required_contract"]:
            assert cls in {"observed", "derived"}
        if cls == "proxy":
            assert field["proxy_kpi"] is True or field["is_allowed_in_kpi"] is False
            assert field["observed_kpi"] is False

    target = output_root or Path(tempfile.mkdtemp(prefix="step97_tensor_audit_selftest_"))
    target.mkdir(parents=True, exist_ok=True)
    dump_json(target / "tensor_db_available_fields_report.json", report)
    (target / "tensor_db_available_fields_report.md").write_text(generate_markdown_report(report), encoding="utf-8")
    print("[OK] Step 97 tensor DB available fields inspector self-test PASS")
    print(f"[OK] self-test json: {target / 'tensor_db_available_fields_report.json'}")
    print(f"[OK] self-test md  : {target / 'tensor_db_available_fields_report.md'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default="", help="PostgreSQL DSN. If omitted, URBANBUS_DB_DSN is used.")
    parser.add_argument("--output-root", default="artifacts/tensor_db_data_availability_audit")
    parser.add_argument("--self-test", action="store_true", help="Run without DB access and create a fake report.")
    parser.add_argument("--metadata-only", action="store_true", help="Skip COUNT/MIN/MAX full-scan style checks.")
    args = parser.parse_args()

    output_root = Path(args.output_root)

    if args.self_test:
        run_self_test(output_root=output_root)
        return

    db_url = args.db_url.strip() or os.environ.get("URBANBUS_DB_DSN", "").strip()
    if not db_url:
        raise SystemExit("[STOP] DB URL missing. Set URBANBUS_DB_DSN or pass --db-url.")

    report = run_actual_inspection(db_url, output_root=output_root, include_row_counts=not args.metadata_only)
    print("[OK] Step 97 actual read-only inspection complete")
    print(f"[OK] fields evaluated: {len(report.get('field_evaluations', []))}")
    print(f"[OK] json report: {output_root / 'tensor_db_available_fields_report.json'}")
    print(f"[OK] md report  : {output_root / 'tensor_db_available_fields_report.md'}")
    print("[DONE] Step 97 tensor DB data availability audit complete")


if __name__ == "__main__":
    main()
