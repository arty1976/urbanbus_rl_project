from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


ARTIFACT_VERSION = "final_reward_design_review_gate_step110_v1"

REQUIRED_GUARDS = {
    "actual_headway": "not_observed",
    "actual_arrival_departure_time": "not_observed",
    "actual_dwell": "not_observed",
    "actual_passenger_wait_observed": False,
    "queue_demand_observed": False,
    "queue_demand_proxy": True,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "canonical_causal_comparison_allowed": False,
    "reward_scaffold_only": True,
    "reward_weights_are_final": False,
    "reward_formula_finalized": False,
    "train_with_this_reward_allowed": False,
    "final_reward_design_claim_allowed": False,
}

REQUIRED_PRINCIPLES = {
    "service_quality_first": True,
    "energy_fleet_secondary": True,
    "energy_proxy_not_primary_objective": True,
    "fleet_reduction_bonus_only": True,
}

REQUIRED_UNLOCKED_ITEMS = {
    "reward_weights_locked": False,
    "normalization_locked": False,
    "hard_constraints_locked": False,
    "baseline_reference_locked": False,
    "train_reward_promotion_approved": False,
}

PROMOTION_REQUIREMENT_KEYWORDS = [
    "reward_weights_locked",
    "normalization_locked",
    "hard_constraints_locked",
    "baseline_reference_locked",
    "actual_observation_upgrade",
    "reward_ablation_protocol",
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


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, full_key))
        else:
            out[full_key] = value
    return out


def find_value(payload: Dict[str, Any], key: str) -> Tuple[bool, Any, str]:
    flat = flatten_dict(payload)

    if key in payload:
        return True, payload[key], key

    if key in flat:
        return True, flat[key], key

    suffix = "." + key
    matches = [(k, v) for k, v in flat.items() if k.endswith(suffix)]
    if matches:
        return True, matches[0][1], matches[0][0]

    return False, None, ""


def add_check(
    checks: List[Dict[str, Any]],
    check_id: str,
    description: str,
    expected: Any,
    actual: Any,
    actual_path: str,
    severity: str = "blocker",
) -> None:
    passed = actual == expected
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "actual_path": actual_path,
        "passed": bool(passed),
        "severity": severity,
        "blocking": severity == "blocker",
    })


def default_review_input() -> Dict[str, Any]:
    return {
        "artifact_version": "final_reward_design_review_input_step110_default_v1",
        "source_steps": ["Step108", "Step109"],
        "review_purpose": (
            "Confirm that Step 108/109 reward remains scaffold-only and cannot be "
            "used as final MAPPO training reward before design review completion."
        ),
        "observability": {
            "actual_headway": "not_observed",
            "actual_arrival_departure_time": "not_observed",
            "actual_dwell": "not_observed",
            "actual_passenger_wait_observed": False,
            "queue_demand_observed": False,
            "queue_demand_proxy": True,
        },
        "claim_guards": {
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "canonical_causal_comparison_allowed": False,
        },
        "reward_status": {
            "reward_scaffold_only": True,
            "reward_weights_are_final": False,
            "reward_formula_finalized": False,
            "train_with_this_reward_allowed": False,
            "final_reward_design_claim_allowed": False,
        },
        "design_principles": {
            "service_quality_first": True,
            "energy_fleet_secondary": True,
            "energy_proxy_not_primary_objective": True,
            "fleet_reduction_bonus_only": True,
        },
        "unlocked_items": {
            "reward_weights_locked": False,
            "normalization_locked": False,
            "hard_constraints_locked": False,
            "baseline_reference_locked": False,
            "train_reward_promotion_approved": False,
        },
        "reward_terms_to_review": [
            {
                "term": "passenger_service_rate",
                "role": "primary_service_quality",
                "unit": "ratio",
                "current_status": "proxy_scaffold",
                "must_be_locked_before_training": True,
            },
            {
                "term": "avg_wait_seconds",
                "role": "service_quality_penalty",
                "unit": "seconds",
                "current_status": "proxy_or_aggregated",
                "must_be_locked_before_training": True,
            },
            {
                "term": "passenger_wait_p95_seconds",
                "role": "long_wait_fairness_penalty",
                "unit": "seconds",
                "current_status": "proxy_scaffold",
                "must_be_locked_before_training": True,
            },
            {
                "term": "energy_proxy_per_passenger",
                "role": "secondary_efficiency_penalty",
                "unit": "kwh_equivalent_per_passenger",
                "current_status": "proxy_scaffold",
                "must_be_locked_before_training": True,
            },
            {
                "term": "fleet_reduction_ratio",
                "role": "secondary_bonus_only",
                "unit": "ratio",
                "current_status": "experimental_condition_derived",
                "must_be_locked_before_training": True,
            },
        ],
        "promotion_requirements": [
            "reward_weights_locked must become true after documented review",
            "normalization_locked must become true after baseline reference selection",
            "hard_constraints_locked must become true before any actual training",
            "baseline_reference_locked must become true before reward promotion",
            "actual_observation_upgrade must be documented for headway/wait/dwell fields",
            "reward_ablation_protocol must compare candidate weights before finalization",
        ],
        "prohibited_until_next_gate": [
            "Do not train MAPPO with Step 108/109 scaffold reward.",
            "Do not use reward_total as a performance metric.",
            "Do not claim A/A90/A80/A70 policy superiority.",
            "Do not include scaffold reward results in a paper-level performance table.",
        ],
    }


def validate_review_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    for key, expected in REQUIRED_GUARDS.items():
        found, actual, actual_path = find_value(payload, key)
        add_check(
            checks=checks,
            check_id=f"guard_{key}",
            description=f"Required guard must remain {expected!r}.",
            expected=expected,
            actual=actual if found else "__MISSING__",
            actual_path=actual_path if found else "__MISSING__",
        )

    for key, expected in REQUIRED_PRINCIPLES.items():
        found, actual, actual_path = find_value(payload, key)
        add_check(
            checks=checks,
            check_id=f"principle_{key}",
            description=f"Required reward design principle must be {expected!r}.",
            expected=expected,
            actual=actual if found else "__MISSING__",
            actual_path=actual_path if found else "__MISSING__",
        )

    for key, expected in REQUIRED_UNLOCKED_ITEMS.items():
        found, actual, actual_path = find_value(payload, key)
        add_check(
            checks=checks,
            check_id=f"unlocked_{key}",
            description=f"Final reward promotion item must remain unlocked: {key}.",
            expected=expected,
            actual=actual if found else "__MISSING__",
            actual_path=actual_path if found else "__MISSING__",
        )

    promotion_requirements = payload.get("promotion_requirements", [])
    promotion_text = "\n".join(str(x) for x in promotion_requirements)

    for keyword in PROMOTION_REQUIREMENT_KEYWORDS:
        add_check(
            checks=checks,
            check_id=f"promotion_requirement_{keyword}",
            description=f"Promotion requirement must mention {keyword}.",
            expected=True,
            actual=keyword in promotion_text,
            actual_path="promotion_requirements",
        )

    blocker_failures = [
        c for c in checks
        if c["blocking"] and not c["passed"]
    ]

    audit_status = "PASS" if not blocker_failures else "FAIL"

    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_SCAFFOLD_REWARD_BLOCKED_FROM_TRAINING"
            if audit_status == "PASS"
            else "FAIL_REWARD_CLAIM_OR_TRAINING_GUARD_VIOLATION"
        ),
        "next_status": (
            "READY_FOR_STEP111_FINAL_REWARD_SPEC_OR_REWARD_CANDIDATE_PROTOCOL"
            if audit_status == "PASS"
            else "BLOCKED_FIX_REWARD_REVIEW_GUARDS"
        ),
        "final_reward_status": "NOT_FINAL_SCAFFOLD_REWARD_ONLY",
        "train_with_this_reward_allowed": False,
        "final_reward_design_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "service_quality_first": True,
        "energy_fleet_secondary": True,
        "check_count": len(checks),
        "failure_count": len(blocker_failures),
        "checks": checks,
        "promotion_requirements": payload.get("promotion_requirements", []),
        "prohibited_until_next_gate": payload.get("prohibited_until_next_gate", []),
        "review_input_snapshot": payload,
    }


def write_checks_csv(path: Path, checks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "check_id",
        "description",
        "expected",
        "actual",
        "actual_path",
        "passed",
        "severity",
        "blocking",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in checks:
            out = dict(row)
            out["expected"] = json.dumps(out["expected"], ensure_ascii=False)
            out["actual"] = json.dumps(out["actual"], ensure_ascii=False)
            writer.writerow(out)


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    failed = [c for c in report["checks"] if not c["passed"]]
    lines: List[str] = []
    lines.append("# Step 110 Final Reward Design Review Gate")
    lines.append("")
    lines.append(f"- artifact_version: `{report['artifact_version']}`")
    lines.append(f"- audit_status: `{report['audit_status']}`")
    lines.append(f"- gate_status: `{report['gate_status']}`")
    lines.append(f"- next_status: `{report['next_status']}`")
    lines.append(f"- final_reward_status: `{report['final_reward_status']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This gate passes only when the current reward remains explicitly "
        "classified as a scaffold reward that is blocked from actual MAPPO training."
    )
    lines.append("")
    lines.append("## Required Position")
    lines.append("")
    lines.append("- Service quality is the primary objective.")
    lines.append("- Energy and fleet reduction are secondary objectives.")
    lines.append("- Fleet reduction is a bonus term, not a standalone optimization target.")
    lines.append("- Energy proxy must not become a shortcut that reduces service quality.")
    lines.append("")
    lines.append("## Current Guard Status")
    lines.append("")
    lines.append(f"- train_with_this_reward_allowed: `{report['train_with_this_reward_allowed']}`")
    lines.append(f"- final_reward_design_claim_allowed: `{report['final_reward_design_claim_allowed']}`")
    lines.append(f"- causal_performance_claim_allowed: `{report['causal_performance_claim_allowed']}`")
    lines.append(f"- paper_level_claim_allowed: `{report['paper_level_claim_allowed']}`")
    lines.append("")
    lines.append("## Promotion Requirements")
    lines.append("")
    for item in report.get("promotion_requirements", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Prohibited Until Next Gate")
    lines.append("")
    for item in report.get("prohibited_until_next_gate", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Failed Checks")
    lines.append("")
    if failed:
        for c in failed:
            lines.append(
                f"- `{c['check_id']}` expected `{c['expected']}` "
                f"but got `{c['actual']}` at `{c['actual_path']}`"
            )
    else:
        lines.append("- None")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument(
        "--output-root",
        default="artifacts/daegu_bis_api_audit/final_reward_design_review_gate_step110",
    )
    parser.add_argument("--write-default-input", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = resolve_path(project_root, args.input)
        payload = load_json_any_encoding(input_path)
        source_input = str(input_path)
    else:
        payload = default_review_input()
        source_input = "default_review_input"

    if args.write_default_input:
        dump_json(output_root / "review_input_default_step110.json", default_review_input())

    report = validate_review_payload(payload)
    report["source_input"] = source_input
    report["output_root"] = str(output_root)

    report_path = output_root / "final_reward_design_review_gate_step110_report.json"
    checks_csv_path = output_root / "final_reward_design_review_gate_step110_checks.csv"
    md_path = output_root / "final_reward_design_review_gate_step110_summary.md"

    dump_json(report_path, report)
    write_checks_csv(checks_csv_path, report["checks"])
    write_markdown(md_path, report)

    print("[OK] Step 110 final reward design review gate completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] output_root   : {output_root}")
    print(f"[OK] report_json   : {report_path}")
    print(f"[OK] checks_csv    : {checks_csv_path}")
    print(f"[OK] summary_md    : {md_path}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] final_claim   : {report['final_reward_design_claim_allowed']}")
    print(f"[OK] causal_allowed: {report['causal_performance_claim_allowed']}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
