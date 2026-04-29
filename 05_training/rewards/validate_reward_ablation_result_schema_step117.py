from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]

EXPECTED_12_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]

EXPECTED_TABLES = [
    "reward_ablation_result_by_candidate_seed_condition",
    "reward_ablation_result_overall_by_candidate",
    "hard_constraint_violations_by_candidate",
    "reward_ablation_result_manifest",
]

REQUIRED_GUARD_FALSE = [
    "actual_ablation_results_available",
    "reward_winner_selected",
    "best_reward_claim_allowed",
    "train_with_this_reward_allowed",
    "reward_promotion_allowed",
    "performance_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "canonical_causal_comparison_allowed",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def add_check(
    checks: List[Dict[str, Any]],
    check_id: str,
    expected: Any,
    actual: Any,
    description: str,
    severity: str = "blocker",
) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "severity": severity,
        "blocking": severity == "blocker",
    })


def validate_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(
        checks,
        "artifact_version",
        "reward_ablation_result_schema_step117_v1",
        schema.get("artifact_version"),
        "Artifact version must match Step 117 schema.",
    )
    add_check(
        checks,
        "step",
        117,
        schema.get("step"),
        "Step number must be 117.",
    )
    add_check(
        checks,
        "schema_status",
        "RESULT_SCHEMA_DRAFT_NOT_ACTUAL_RESULT",
        schema.get("schema_status"),
        "Step 117 must define schema only, not actual results.",
    )

    guards = schema.get("claim_guards", {})
    for key in REQUIRED_GUARD_FALSE:
        add_check(
            checks,
            f"guard_{key}",
            False,
            guards.get(key),
            f"{key} must remain false.",
        )

    candidates = schema.get("candidate_ids_expected", [])
    add_check(
        checks,
        "candidate_ids_exact",
        EXPECTED_CANDIDATES,
        candidates,
        "Expected reward candidates must be R0-R5 in order.",
    )

    kpis = schema.get("canonical_12_kpis", [])
    add_check(
        checks,
        "canonical_12_kpis_exact",
        EXPECTED_12_KPIS,
        kpis,
        "Canonical 12-KPI list must match expected order and content.",
    )

    tables = schema.get("result_tables", [])
    table_names = [t.get("table_name") for t in tables]
    add_check(
        checks,
        "result_tables_exact",
        EXPECTED_TABLES,
        table_names,
        "Expected result tables must match Step 117 contract.",
    )

    table_map = {t.get("table_name"): t for t in tables}

    main_table = table_map.get("reward_ablation_result_by_candidate_seed_condition", {})
    main_cols = main_table.get("required_columns", [])
    for col in [
        "candidate_id",
        "condition_id",
        "seed",
        "window_count",
        "service_quality_score_mean",
        "secondary_efficiency_score_mean",
        "hard_constraint_violation_count",
        "train_with_this_reward_allowed",
        "best_reward_claim_allowed",
        "causal_performance_claim_allowed",
    ]:
        add_check(
            checks,
            f"main_table_col_{col}",
            True,
            col in main_cols,
            f"Main result table must include {col}.",
        )

    for kpi in EXPECTED_12_KPIS:
        col = f"{kpi}_mean"
        add_check(
            checks,
            f"main_table_kpi_col_{col}",
            True,
            col in main_cols,
            f"Main result table must include {col}.",
        )

    overall_table = table_map.get("reward_ablation_result_overall_by_candidate", {})
    overall_cols = overall_table.get("required_columns", [])
    for col in [
        "eligible_for_promotion",
        "promotion_block_reason",
        "candidate_rank_placeholder",
        "reward_winner_selected",
        "best_reward_claim_allowed",
        "train_with_this_reward_allowed",
    ]:
        add_check(
            checks,
            f"overall_table_col_{col}",
            True,
            col in overall_cols,
            f"Overall candidate table must include {col}.",
        )

    violation_table = table_map.get("hard_constraint_violations_by_candidate", {})
    violation_cols = violation_table.get("required_columns", [])
    for col in [
        "constraint_id",
        "metric",
        "threshold_rule",
        "observed_value",
        "baseline_value",
        "violation_margin",
        "violation_count",
        "blocking",
    ]:
        add_check(
            checks,
            f"violation_table_col_{col}",
            True,
            col in violation_cols,
            f"Hard constraint violation ledger must include {col}.",
        )

    ranking = schema.get("ranking_policy_draft_not_locked", {})
    add_check(
        checks,
        "ranking_not_locked",
        "DRAFT_NOT_LOCKED",
        ranking.get("ranking_status"),
        "Ranking policy must remain draft and not locked.",
    )
    add_check(
        checks,
        "winner_selection_not_allowed",
        False,
        ranking.get("winner_selection_allowed_now"),
        "Winner selection must not be allowed in Step 117.",
    )

    requirements = "\n".join(str(x) for x in schema.get("promotion_requirements_after_results", []))
    for phrase in [
        "same scenarios",
        "canonical 12-KPI",
        "Hard constraint violation ledger",
        "blocking hard constraint violations",
        "service-quality score",
        "smoke-only",
        "train_with_this_reward_allowed",
    ]:
        add_check(
            checks,
            f"promotion_requirement_mentions_{phrase.replace(' ', '_').replace('-', '_')}",
            True,
            phrase in requirements,
            f"Promotion requirements must mention {phrase}.",
        )

    prohibited = "\n".join(str(x) for x in schema.get("prohibited_now", []))
    for phrase in [
        "Do not select a best reward candidate",
        "Do not train MAPPO",
        "Do not claim reward performance",
        "Do not claim causal improvement",
    ]:
        add_check(
            checks,
            f"prohibited_mentions_{phrase.split()[2].lower()}",
            True,
            phrase in prohibited,
            f"Prohibited list must include: {phrase}.",
        )

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_ablation_result_schema_step117_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_REWARD_ABLATION_RESULT_SCHEMA_NOT_EVALUATED"
            if audit_status == "PASS"
            else "FAIL_REWARD_ABLATION_RESULT_SCHEMA_GUARD_VIOLATION"
        ),
        "next_status": (
            "READY_FOR_STEP118_REWARD_ABLATION_RESULT_WRITER_OR_RUNNER_GUARD"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP117_SCHEMA"
        ),
        "candidate_count": len(candidates),
        "table_count": len(tables),
        "actual_results_available": False,
        "reward_winner_selected": False,
        "train_with_this_reward_allowed": False,
        "best_reward_claim_allowed": False,
        "failure_count": len(failures),
        "check_count": len(checks),
        "checks": checks,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="05_training/rewards/reward_ablation_result_schema_step117.json")
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_ablation_result_schema_step117_validation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    schema_path = resolve_path(project_root, args.schema)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not schema_path.exists():
        raise SystemExit(f"schema not found: {schema_path}")

    schema = load_json_any_encoding(schema_path)
    report = validate_schema(schema)
    report["schema_path"] = str(schema_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_ablation_result_schema_step117_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 117 reward ablation result schema validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] candidate_count: {report['candidate_count']}")
    print(f"[OK] table_count    : {report['table_count']}")
    print(f"[OK] actual_results : {report['actual_results_available']}")
    print(f"[OK] winner_selected: {report['reward_winner_selected']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
