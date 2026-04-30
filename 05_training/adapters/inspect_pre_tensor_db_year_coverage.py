from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2 import sql


PRIORITY_RELATIONS = [
    "public.graph_state_timeslice",
    "public.rl_state_training_base",
    "public.rl_stop_transition_features",
    "public.gatv2_snapshot_stop_features_train",
    "public.gatv2_snapshot_summary",
    "public.gatv2_node_master_active",
    "public.gatv2_edge_primary_active",
    "public.fact_stop_usage_hourly",
    "public.stop_hourly_ridership",
    "public.stop_hourly_usage",
]

DATE_COLUMN_PRIORITY = [
    "state_ts",
    "service_date",
    "operation_date",
    "snapshot_ts",
    "snapshot_time",
    "event_ts",
    "created_at",
    "일자",
    "date",
    "day",
    "base_date",
    "min_state_ts",
    "max_state_ts",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def split_relation(relation: str) -> Tuple[str, str]:
    if "." not in relation:
        return "public", relation
    schema, table = relation.split(".", 1)
    return schema.strip('"'), table.strip('"')


def relation_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.tables
              WHERE table_schema = %s
                AND table_name = %s
            )
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def get_columns(conn, schema: str, table: str) -> List[Dict[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        return [
            {"column_name": str(r[0]), "data_type": str(r[1])}
            for r in cur.fetchall()
        ]


def choose_date_column(columns: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    exact = {c["column_name"]: c for c in columns}
    lower = {c["column_name"].lower(): c for c in columns}

    for name in DATE_COLUMN_PRIORITY:
        if name in exact:
            return exact[name]
        if name.lower() in lower:
            return lower[name.lower()]

    for c in columns:
        name = c["column_name"].lower()
        dtype = c["data_type"].lower()
        if (
            "date" in name
            or "time" in name
            or name.endswith("_ts")
            or "일자" in c["column_name"]
            or "timestamp" in dtype
            or dtype == "date"
        ):
            return c

    return None


def classify_year_counts(year_counts: Dict[str, int]) -> str:
    years = {int(y) for y, cnt in year_counts.items() if int(cnt) > 0}

    if not years:
        return "NO_ROWS_WITH_DATE"
    if years == {2023}:
        return "YEAR_2023_ONLY"
    if years == {2022}:
        return "YEAR_2022_ONLY"
    if 2023 in years and len(years) > 1:
        return "MIXED_WITH_2023"
    return "NON_2023_ONLY"


def make_date_expr(column: Dict[str, str]) -> sql.SQL:
    col = sql.Identifier(column["column_name"])
    dtype = column["data_type"].lower()

    if dtype in {"date", "timestamp without time zone", "timestamp with time zone"}:
        return sql.SQL("{}::timestamptz").format(col)

    return sql.SQL("NULLIF({}::text, '')::timestamptz").format(col)


def audit_relation(conn, relation: str) -> Dict[str, Any]:
    schema, table = split_relation(relation)

    result: Dict[str, Any] = {
        "relation": relation,
        "exists": False,
        "audited": False,
        "date_column": None,
        "row_count": None,
        "min_ts": None,
        "max_ts": None,
        "year_counts": {},
        "verdict": "NOT_FOUND",
        "error": None,
    }

    if not relation_exists(conn, schema, table):
        return result

    result["exists"] = True

    columns = get_columns(conn, schema, table)
    result["columns"] = columns

    date_col = choose_date_column(columns)
    if date_col is None:
        result["verdict"] = "NO_DATE_COLUMN"
        return result

    result["date_column"] = date_col
    expr = make_date_expr(date_col)

    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                      COUNT(*)::bigint AS row_count,
                      MIN({expr}) AS min_ts,
                      MAX({expr}) AS max_ts
                    FROM {schema}.{table}
                    WHERE {expr} IS NOT NULL
                    """
                ).format(
                    expr=expr,
                    schema=sql.Identifier(schema),
                    table=sql.Identifier(table),
                )
            )
            row_count, min_ts, max_ts = cur.fetchone()

            cur.execute(
                sql.SQL(
                    """
                    SELECT
                      EXTRACT(YEAR FROM {expr})::int AS year,
                      COUNT(*)::bigint AS row_count
                    FROM {schema}.{table}
                    WHERE {expr} IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1
                    """
                ).format(
                    expr=expr,
                    schema=sql.Identifier(schema),
                    table=sql.Identifier(table),
                )
            )
            year_counts = {
                str(int(year)): int(count)
                for year, count in cur.fetchall()
                if year is not None
            }

        result["audited"] = True
        result["row_count"] = int(row_count or 0)
        result["min_ts"] = min_ts.isoformat() if min_ts else None
        result["max_ts"] = max_ts.isoformat() if max_ts else None
        result["year_counts"] = year_counts
        result["verdict"] = classify_year_counts(year_counts)
        return result

    except Exception as exc:
        result["error"] = str(exc)
        result["verdict"] = "AUDIT_ERROR"
        return result


def discover_date_relations(conn, schema: str = "public") -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT table_schema, table_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND (
                   lower(column_name) IN (
                     'state_ts',
                     'service_date',
                     'operation_date',
                     'snapshot_ts',
                     'snapshot_time',
                     'event_ts',
                     'created_at',
                     'date',
                     'day',
                     'base_date',
                     'min_state_ts',
                     'max_state_ts'
                   )
                   OR column_name LIKE '%%일자%%'
                   OR lower(column_name) LIKE '%%date%%'
                   OR lower(column_name) LIKE '%%time%%'
                   OR lower(column_name) LIKE '%%_ts'
              )
            ORDER BY table_schema, table_name
            """,
            (schema,),
        )
        return [f"{r[0]}.{r[1]}" for r in cur.fetchall()]


def overall_verdict(priority_results: List[Dict[str, Any]], discovered_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    priority_audited = [
        r for r in priority_results
        if r.get("audited")
        and r.get("verdict") not in {"NOT_FOUND", "NO_DATE_COLUMN", "AUDIT_ERROR"}
    ]

    all_audited = [
        r for r in priority_results + discovered_results
        if r.get("audited")
        and r.get("verdict") not in {"NOT_FOUND", "NO_DATE_COLUMN", "AUDIT_ERROR"}
    ]

    if not all_audited:
        return {
            "status": "UNKNOWN_NO_AUDITED_DATE_RELATIONS",
            "can_merge_2023_ridership_now": False,
            "reason": "날짜 컬럼이 있는 auditable relation을 확인하지 못했습니다.",
        }

    if priority_audited and all(r["verdict"] == "YEAR_2023_ONLY" for r in priority_audited):
        return {
            "status": "CORE_PRIORITY_RELATIONS_2023_ONLY",
            "can_merge_2023_ridership_now": True,
            "reason": "핵심 tensor-source relation들이 2023년 자료로 확인되었습니다.",
            "priority_relations_audited": [r["relation"] for r in priority_audited],
        }

    mixed = [r for r in priority_audited if r["verdict"] == "MIXED_WITH_2023"]
    if mixed:
        return {
            "status": "MIXED_YEAR_PRIORITY_RELATIONS",
            "can_merge_2023_ridership_now": False,
            "reason": "핵심 relation 중 2023과 다른 연도가 섞인 relation이 있습니다. 2023-only view를 먼저 만들어야 합니다.",
            "mixed_relations": [r["relation"] for r in mixed],
        }

    only_2023 = [r for r in all_audited if r["verdict"] == "YEAR_2023_ONLY"]
    non_2023 = [r for r in all_audited if r["verdict"] in {"YEAR_2022_ONLY", "NON_2023_ONLY"}]

    if only_2023 and non_2023:
        return {
            "status": "DATABASE_HAS_2023_AND_NON_2023_RELATIONS",
            "can_merge_2023_ridership_now": True,
            "reason": "DB 안에 2023 relation과 non-2023 relation이 함께 있습니다. 현재 tensor DB와 합칠 때는 2023-only relation만 사용해야 합니다.",
            "year_2023_relations": [r["relation"] for r in only_2023],
            "non_2023_relations": [r["relation"] for r in non_2023],
        }

    if only_2023:
        return {
            "status": "AUDITED_RELATIONS_2023_ONLY",
            "can_merge_2023_ridership_now": True,
            "reason": "확인된 날짜 기반 relation들이 2023년 자료입니다.",
        }

    return {
        "status": "NO_2023_ONLY_SOURCE_CONFIRMED",
        "can_merge_2023_ridership_now": False,
        "reason": "2023-only tensor source relation을 확인하지 못했습니다.",
    }


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Pre-tensor DB Year Coverage Audit",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- overall_status: `{report['overall_verdict']['status']}`",
        f"- can_merge_2023_ridership_now: `{report['overall_verdict']['can_merge_2023_ridership_now']}`",
        f"- reason: {report['overall_verdict']['reason']}",
        "",
        "## Priority relation audit",
        "",
        "| relation | exists | date_column | row_count | min_ts | max_ts | year_counts | verdict |",
        "|---|---:|---|---:|---|---|---|---|",
    ]

    for r in report["priority_relation_audits"]:
        dc = r.get("date_column") or {}
        dc_name = dc.get("column_name", "") if isinstance(dc, dict) else ""
        lines.append(
            f"| {r['relation']} | {r.get('exists')} | {dc_name} | {r.get('row_count')} | {r.get('min_ts')} | {r.get('max_ts')} | `{json.dumps(r.get('year_counts', {}), ensure_ascii=False)}` | {r.get('verdict')} |"
        )

    lines.extend([
        "",
        "## Recommendation",
        "",
    ])

    if report["overall_verdict"]["can_merge_2023_ridership_now"]:
        lines.append("현재 tensor DB와 합칠 자료는 2023년 승하차 CSV만 사용하는 방향이 타당합니다.")
        lines.append("")
        lines.append("2022년 자료는 지금 합치지 말고 future cross-year validation 또는 multiyear dataset 후보로 보관하세요.")
    else:
        lines.append("아직 2023 승하차 CSV를 바로 합치지 마세요.")
        lines.append("")
        lines.append("먼저 2023-only source view를 만들거나, 어떤 relation이 현재 tensor DB의 원천인지 확정해야 합니다.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(db_url: str, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(db_url)
    try:
        priority_results = [audit_relation(conn, rel) for rel in PRIORITY_RELATIONS]

        discovered_relations = discover_date_relations(conn)
        discovered_results = [
            audit_relation(conn, rel)
            for rel in discovered_relations
            if rel not in PRIORITY_RELATIONS
        ]
    finally:
        conn.close()

    report = {
        "artifact_version": "pre_tensor_db_year_coverage_audit_v1",
        "created_at_utc": now_utc_iso(),
        "scope": "read_only_database_year_coverage_check_before_2023_ridership_merge",
        "priority_relations": PRIORITY_RELATIONS,
        "priority_relation_audits": priority_results,
        "discovered_date_relations": discovered_relations,
        "discovered_relation_audits": discovered_results,
        "overall_verdict": overall_verdict(priority_results, discovered_results),
        "merge_policy": {
            "current_tensor_db_alignment": "use 2023 ridership only if current DB source is confirmed as 2023",
            "year_2022_policy": "archive for future cross-year validation; do not merge into current 2023 tensor DB",
            "if_mixed_years": "create explicit 2023-only DB views before merge",
        },
        "claim_guardrails": {
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }

    json_path = output_dir / "pre_tensor_db_year_coverage_report.json"
    md_path = output_dir / "pre_tensor_db_year_coverage_report.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--output-dir", default="artifacts/pre_tensor_db_year_coverage")
    args = parser.parse_args()

    report = run_audit(
        db_url=args.db_url,
        output_dir=Path(args.output_dir),
    )

    print("[OK] pre-tensor DB year coverage audit complete")
    print(f"[OK] output_dir : {args.output_dir}")
    print(f"[OK] status     : {report['overall_verdict']['status']}")
    print(f"[OK] can_merge_2023_ridership_now: {report['overall_verdict']['can_merge_2023_ridership_now']}")
    print(f"[OK] reason     : {report['overall_verdict']['reason']}")

    print("")
    print("[SUMMARY] Priority relations:")
    for r in report["priority_relation_audits"]:
        print(
            f" - {r['relation']}: exists={r.get('exists')} "
            f"date_col={(r.get('date_column') or {}).get('column_name') if isinstance(r.get('date_column'), dict) else None} "
            f"min={r.get('min_ts')} max={r.get('max_ts')} "
            f"years={r.get('year_counts')} verdict={r.get('verdict')}"
        )


if __name__ == "__main__":
    main()
