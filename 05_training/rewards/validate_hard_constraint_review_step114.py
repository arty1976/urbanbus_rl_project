from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_CONSTRAINT_IDS = [
    "service_rate_floor",
    "avg_wait_regression_cap",
    "p95_wait_regression_cap",
    "energy_service_coupling_guard",
    "fleet_reduction_service_guard",
    "qwen_off_for_A_family",
    "observability_overclaim_guard",
    "finite_reward_guard",
]

REQUIRED_FALSE_GUARDS = [
    "train_with_this_reward_allowed",
    "final_reward_design_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "canonical_causal_comparison_allowed",
    "hard_constraints_finalized",
    "hard_constraint_numeric_values_locked",
    "reward_weights_locked",
    "reward_formula_finalized",
]

REQUIRED_TRUE_PRINCIPLES = [
    "service_quality_floor_required",
    "long_wait_fairness_required",
    "energy_saving_by_service_suppression_blocked",
    "fleet_reduction_bonus_must_not_override_service",
    "qwen_off_for_A_family_required",
    "observability_overclaim_blocked",
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


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str, severity: str = "blocker") -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "severity": severity,
        "blocking": severity == "blocker",
    })


def by_id(constraints: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(c.get("constraint_id")): c for c in constraints}


def validate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "artifact_version", "hard_constraint_review_step114_v1", spec.get("artifact_version"), "Artifact version must match Step 114 spec.")
    add_check(checks, "review_status", "REVIEWED_NOT_TRAINABLE", spec.get("review_status"), "Step 114 must remain reviewed but not trainable.")

    baseline = spec.get("baseline_reference", {})
    add_check(checks, "baseline_id", "B1_noop", baseline.get("normalization_baseline_id"), "Normalization baseline ID must remain B1_noop.")
    add_check(checks, "baseline_identity_locked", True, baseline.get("normalization_baseline_locked"), "Baseline identity must be locked.")
    add_check(checks, "numeric_baseline_values_locked_false", False, baseline.get("numeric_baseline_values_locked"), "Numeric baseline values must not be locked in Step 114.")

    guards = spec.get("claim_guards", {})
    for key in REQUIRED_FALSE_GUARDS:
        add_check(checks, f"claim_guard_{key}", False, guards.get(key), f"{key} must remain false.")

    obs = spec.get("observability_guards", {})
    add_check(checks, "actual_headway_guard", "not_observed", obs.get("actual_headway"), "Actual headway must remain not_observed.")
    add_check(checks, "actual_arrival_departure_guard", "not_observed", obs.get("actual_arrival_departure_time"), "Actual arrival/departure time must remain not_observed.")
    add_check(checks, "actual_dwell_guard", "not_observed", obs.get("actual_dwell"), "Actual dwell must remain not_observed.")
    add_check(checks, "actual_passenger_wait_guard", False, obs.get("actual_passenger_wait_observed"), "Actual passenger wait must remain unobserved.")
    add_check(checks, "queue_demand_observed_guard", False, obs.get("queue_demand_observed"), "Queue demand observed must remain false.")
    add_check(checks, "queue_demand_proxy_guard", True, obs.get("queue_demand_proxy"), "Queue demand proxy must remain true.")
    add_check(checks, "eta_not_actual_headway_guard", False, obs.get("eta_based_headway_is_actual_headway"), "ETA-based headway must not be treated as actual headway.")

    principles = spec.get("hard_constraint_design_principles", {})
    for key in REQUIRED_TRUE_PRINCIPLES:
        add_check(checks, f"principle_{key}", True, principles.get(key), f"{key} must be true.")

    constraints = spec.get("constraints", [])
    ids = by_id(constraints)
    add_check(checks, "constraint_count_minimum", True, len(constraints) >= len(REQUIRED_CONSTRAINT_IDS), "All required hard constraints must be present.")

    listed_ids = spec.get("required_constraint_ids", [])
    add_check(checks, "required_constraint_ids_exact", REQUIRED_CONSTRAINT_IDS, listed_ids, "Required constraint ID list must match expected order.")

    for cid in REQUIRED_CONSTRAINT_IDS:
        add_check(checks, f"constraint_exists_{cid}", True, cid in ids, f"Required constraint missing: {cid}")
        if cid in ids:
            c = ids[cid]
            add_check(checks, f"constraint_unlocked_{cid}", False, c.get("locked"), f"{cid} must remain unlocked in Step 114.")
            add_check(checks, f"constraint_has_metric_{cid}", True, bool(c.get("metric")), f"{cid} must define metric.")
            add_check(checks, f"constraint_has_rule_{cid}", True, bool(c.get("draft_rule")), f"{cid} must define draft_rule.")
            add_check(checks, f"constraint_has_reason_{cid}", True, bool(c.get("reason")), f"{cid} must define reason.")

    if "service_rate_floor" in ids:
        add_check(checks, "service_rate_floor_threshold", 0.95, ids["service_rate_floor"].get("draft_threshold_value"), "Service rate floor draft threshold must be 0.95.")

    if "avg_wait_regression_cap" in ids:
        add_check(checks, "avg_wait_cap_multiplier", 1.10, ids["avg_wait_regression_cap"].get("draft_threshold_multiplier"), "Average wait draft cap multiplier must be 1.10.")
        add_check(checks, "avg_wait_cap_baseline", "B1_noop", ids["avg_wait_regression_cap"].get("baseline_dependency"), "Average wait cap must depend on B1_noop.")

    if "p95_wait_regression_cap" in ids:
        add_check(checks, "p95_wait_cap_multiplier", 1.15, ids["p95_wait_regression_cap"].get("draft_threshold_multiplier"), "P95 wait draft cap multiplier must be 1.15.")
        add_check(checks, "p95_wait_cap_baseline", "B1_noop", ids["p95_wait_regression_cap"].get("baseline_dependency"), "P95 wait cap must depend on B1_noop.")

    if "qwen_off_for_A_family" in ids:
        add_check(checks, "qwen_threshold_zero", 0.0, ids["qwen_off_for_A_family"].get("draft_threshold_value"), "Qwen trigger threshold must be exactly zero.")
        add_check(checks, "qwen_violation_hard_fail", "hard_fail_candidate", ids["qwen_off_for_A_family"].get("violation_action"), "Qwen violation should hard-fail candidate.")

    still_not_allowed = "\n".join(str(x) for x in spec.get("still_not_allowed", []))
    for marker in ["Do not train MAPPO", "Do not claim", "ETA-based headway", "queue/demand proxy"]:
        add_check(checks, f"still_not_allowed_{marker.lower().replace(' ', '_').replace('/', '_')}", True, marker in still_not_allowed, f"Missing still-not-allowed marker: {marker}")

    next_gates = "\n".join(str(x) for x in spec.get("required_next_gates_before_training", []))
    for marker in ["Step115", "Step116", "empirical baseline numeric value table", "constraint stress test"]:
        add_check(checks, f"next_gate_{marker.lower().replace(' ', '_')}", True, marker in next_gates, f"Missing next gate marker: {marker}")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_hard_constraint_review_step114_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "review_status": spec.get("review_status"),
        "gate_status": "PASS_HARD_CONSTRAINT_REVIEW_NOT_TRAINABLE" if audit_status == "PASS" else "FAIL_HARD_CONSTRAINT_REVIEW_GUARD_VIOLATION",
        "next_status": "READY_FOR_STEP115_REWARD_ABLATION_MATRIX" if audit_status == "PASS" else "BLOCKED_FIX_STEP114_HARD_CONSTRAINT_REVIEW",
        "baseline": baseline.get("normalization_baseline_id"),
        "train_with_this_reward_allowed": False,
        "hard_constraints_finalized": False,
        "numeric_values_locked": False,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="05_training/rewards/hard_constraint_review_step114.json")
    parser.add_argument("--output-root", default="artifacts/rewards/hard_constraint_review_step114_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.spec)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not spec_path.exists():
        raise SystemExit(f"spec not found: {spec_path}")

    spec = load_json_any_encoding(spec_path)
    report = validate_spec(spec)
    report["spec_path"] = str(spec_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "hard_constraint_review_step114_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 114 hard constraint review validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] baseline      : {report['baseline']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] finalized     : {report['hard_constraints_finalized']}")
    print(f"[OK] numeric_locked: {report['numeric_values_locked']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
