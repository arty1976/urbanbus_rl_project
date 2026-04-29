from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_TERMS = [
    "passenger_service_rate",
    "avg_wait_seconds",
    "passenger_wait_p95_seconds",
    "on_time_rate",
    "bunching_rate",
    "cv_headway",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
    "intervention_rate",
]

FORBIDDEN_DIRECT_TERMS = [
    "passenger_demand_generated",
    "passenger_served_count",
    "energy_proxy",
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


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "severity": "blocker",
        "blocking": True,
    })


def validate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(
        checks,
        "artifact_version",
        "reward_normalization_baseline_lock_step113_v1",
        spec.get("artifact_version"),
        "Artifact version must match Step 113.",
    )

    add_check(
        checks,
        "lock_status",
        "BASELINE_REFERENCE_LOCKED_FOR_CANDIDATE_EVALUATION_NOT_TRAINABLE",
        spec.get("lock_status"),
        "Step 113 must lock only the baseline reference and remain not trainable.",
    )

    guards = spec.get("claim_guards", {})
    add_check(
        checks,
        "baseline_reference_locked_for_candidate_evaluation",
        True,
        guards.get("baseline_reference_locked_for_candidate_evaluation"),
        "Baseline reference must be locked for candidate evaluation.",
    )

    for key in [
        "reward_formula_finalized",
        "reward_weights_locked",
        "normalization_numeric_values_locked",
        "hard_constraints_locked",
        "train_with_this_reward_allowed",
        "final_reward_design_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "canonical_causal_comparison_allowed",
    ]:
        add_check(checks, f"claim_guard_{key}", False, guards.get(key), f"{key} must remain false.")

    obs = spec.get("observability_guards", {})
    add_check(checks, "actual_headway", "not_observed", obs.get("actual_headway"), "Actual headway must remain not_observed.")
    add_check(checks, "actual_arrival_departure_time", "not_observed", obs.get("actual_arrival_departure_time"), "Actual arrival/departure must remain not_observed.")
    add_check(checks, "actual_dwell", "not_observed", obs.get("actual_dwell"), "Actual dwell must remain not_observed.")
    add_check(checks, "actual_passenger_wait_observed", False, obs.get("actual_passenger_wait_observed"), "Actual passenger wait must remain false.")
    add_check(checks, "queue_demand_observed", False, obs.get("queue_demand_observed"), "Queue demand observed must remain false.")
    add_check(checks, "queue_demand_proxy", True, obs.get("queue_demand_proxy"), "Queue demand proxy must remain true.")

    lock = spec.get("baseline_reference_lock", {})
    add_check(checks, "training_baseline", "B1_noop", lock.get("training_reward_normalization_baseline"), "Training reward normalization baseline must be B1_noop.")
    add_check(checks, "numeric_baseline_values_locked", False, lock.get("numeric_baseline_values_locked"), "Numeric baseline values are not locked in Step 113.")

    reporting = lock.get("reporting_comparison_baselines", [])
    for b in ["B0_historical", "B1_noop", "B2_rulebased"]:
        add_check(checks, f"reporting_baseline_{b}", True, b in reporting, f"{b} must be a reporting baseline.")

    policy = spec.get("normalization_policy", {})
    add_check(checks, "weight_sign_convention", "positive_magnitude_with_formula_sign", policy.get("weight_sign_convention"), "Weight sign convention must remain positive magnitude.")
    add_check(checks, "scale_policy", "baseline_ratio_or_bounded_score", policy.get("scale_policy"), "Scale policy must be baseline ratio or bounded score.")

    terms = spec.get("normalization_terms", [])
    by_term = {t.get("term"): t for t in terms}
    for term in EXPECTED_TERMS:
        add_check(checks, f"normalization_term_{term}", True, term in by_term, f"{term} must have a normalization rule.")
        if term in by_term:
            add_check(checks, f"normalization_rule_present_{term}", True, bool(by_term[term].get("normalization_rule")), f"{term} must have non-empty normalization_rule.")
            add_check(checks, f"observation_status_present_{term}", True, bool(by_term[term].get("observation_status")), f"{term} must preserve observation/proxy status.")

    forbidden = spec.get("forbidden_direct_normalization_terms", [])
    for term in FORBIDDEN_DIRECT_TERMS:
        add_check(checks, f"forbidden_direct_{term}", True, term in forbidden, f"{term} must be forbidden as a direct normalization term.")
        add_check(checks, f"forbidden_not_in_normalization_terms_{term}", False, term in by_term, f"{term} must not be in direct normalization terms.")

    artifacts = spec.get("required_baseline_artifacts_before_numeric_lock", [])
    for marker in [
        "B1_noop canonical 12-KPI kpi_by_window.parquet",
        "B1_noop canonical 12-KPI kpi_by_seed.parquet",
        "B1_noop canonical 12-KPI kpi_overall.json",
        "B1_noop normalization_constants_step113_or_later.json",
    ]:
        add_check(checks, f"required_artifact_{str(marker).lower().replace(' ', '_').replace('/', '_').replace('.', '_').replace('-', '_')}", True, marker in artifacts, f"Required artifact missing: {marker}")

    next_gates = spec.get("next_required_gates", [])
    for gate in [
        "Step114 hard constraint review",
        "Step115 reward ablation matrix",
        "Step116 trainable reward promotion gate",
    ]:
        add_check(checks, f"next_gate_{gate.split()[0].lower()}", True, gate in next_gates, f"{gate} must be listed.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_normalization_baseline_lock_step113_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_BASELINE_REFERENCE_LOCKED_NOT_TRAINABLE"
            if audit_status == "PASS"
            else "FAIL_BASELINE_LOCK_GUARD_VIOLATION"
        ),
        "next_status": (
            "READY_FOR_STEP114_HARD_CONSTRAINT_REVIEW"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP113_BASELINE_LOCK"
        ),
        "training_reward_normalization_baseline": lock.get("training_reward_normalization_baseline"),
        "train_with_this_reward_allowed": False,
        "numeric_baseline_values_locked": False,
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
    parser.add_argument("--spec", default="05_training/rewards/reward_normalization_baseline_lock_step113.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_normalization_baseline_lock_step113_validation")
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

    report_path = output_root / "reward_normalization_baseline_lock_step113_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 113 reward normalization baseline lock validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] baseline      : {report['training_reward_normalization_baseline']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] numeric_locked: {report['numeric_baseline_values_locked']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
