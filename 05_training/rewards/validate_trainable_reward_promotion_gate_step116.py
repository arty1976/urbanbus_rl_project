from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_REQUIRED_ARTIFACTS = [
    "05_training/rewards/final_reward_spec_step111.json",
    "05_training/rewards/reward_candidate_protocol_step112.json",
    "05_training/rewards/reward_normalization_baseline_lock_step113.json",
    "05_training/rewards/hard_constraint_review_step114.json",
    "05_training/rewards/reward_ablation_matrix_step115.json",
]

EXPECTED_CANDIDATE_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]


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


def resolve_project_path(project_root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return project_root / p


def validate_gate(gate: Dict[str, Any], project_root: Path, check_artifacts: bool) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "artifact_version", "trainable_reward_promotion_gate_step116_v1", gate.get("artifact_version"), "Artifact version must match Step 116.")
    add_check(checks, "step", 116, int(gate.get("step", -1)), "Step must be 116.")
    add_check(checks, "gate_status", "DRAFT_GATE_NOT_PROMOTED", gate.get("gate_status"), "Step 116 must remain a draft non-promotion gate.")
    add_check(checks, "trainable_reward_promoted_false", False, gate.get("trainable_reward_promoted"), "Reward must not be promoted yet.")
    add_check(checks, "train_allowed_false", False, gate.get("train_with_this_reward_allowed"), "Training must remain disabled.")
    add_check(checks, "selected_candidate_null", None, gate.get("selected_candidate_id"), "No selected candidate is allowed yet.")
    add_check(checks, "selected_claim_false", False, gate.get("selected_candidate_claim_allowed"), "Best candidate claim must remain false.")

    artifacts = gate.get("required_prior_artifacts", [])
    add_check(checks, "required_artifact_list_exact", EXPECTED_REQUIRED_ARTIFACTS, artifacts, "Prior artifact list must match Step 111-115.")

    if check_artifacts:
        for rel in EXPECTED_REQUIRED_ARTIFACTS:
            exists = resolve_project_path(project_root, rel).exists()
            safe_id = rel.replace("/", "_").replace(".", "_").replace("-", "_")
            add_check(checks, f"artifact_exists_{safe_id}", True, exists, f"Required artifact must exist: {rel}")

    requirements = gate.get("promotion_requirements", {})
    must_be_true_now = [
        "final_reward_spec_exists",
        "candidate_protocol_exists",
        "normalization_baseline_locked",
        "hard_constraint_review_exists",
        "reward_ablation_matrix_exists",
    ]
    for key in must_be_true_now:
        add_check(checks, f"prior_requirement_{key}", True, requirements.get(key), f"{key} must be true.")

    must_remain_false = [
        "actual_ablation_results_exist",
        "selected_candidate_has_best_evidence",
        "selected_candidate_passes_service_floor",
        "selected_candidate_passes_wait_regression_caps",
        "selected_candidate_passes_energy_shortcut_guard",
        "selected_candidate_passes_qwen_off_guard",
        "selected_candidate_passes_observability_guard",
        "human_review_completed",
    ]
    for key in must_remain_false:
        add_check(checks, f"not_ready_{key}", False, requirements.get(key), f"{key} must remain false until actual evidence exists.")

    non_promotable = gate.get("non_promotable_until_all_true", [])
    add_check(checks, "non_promotable_list_exact", must_remain_false, non_promotable, "Non-promotable list must match missing evidence requirements.")

    matrix = gate.get("candidate_matrix_expected", {})
    add_check(checks, "candidate_count", 6, int(matrix.get("candidate_count", -1)), "Candidate count must be 6.")
    add_check(checks, "candidate_ids", EXPECTED_CANDIDATE_IDS, matrix.get("candidate_ids", []), "Candidate IDs must be R0-R5.")
    add_check(checks, "best_claim_false", False, matrix.get("best_candidate_claim_allowed"), "No best candidate claim allowed yet.")

    guards = gate.get("locked_guards", {})
    add_check(checks, "weight_sign_convention", "positive_magnitude_with_formula_sign", guards.get("weight_sign_convention"), "Weight sign convention must remain positive magnitude.")
    add_check(checks, "normalization_baseline", "B1_noop", guards.get("normalization_baseline"), "Normalization baseline must be B1_noop.")
    add_check(checks, "numeric_baseline_values_locked_false", False, guards.get("numeric_baseline_values_locked"), "Numeric baseline values must not be locked yet.")
    add_check(checks, "hard_constraints_finalized_false", False, guards.get("hard_constraints_finalized"), "Hard constraints must not be finalized yet.")
    add_check(checks, "paper_claim_false", False, guards.get("paper_level_claim_allowed"), "Paper-level claim must remain false.")
    add_check(checks, "causal_claim_false", False, guards.get("causal_performance_claim_allowed"), "Causal claim must remain false.")
    add_check(checks, "canonical_causal_false", False, guards.get("canonical_causal_comparison_allowed"), "Canonical causal comparison must remain false.")
    add_check(checks, "actual_headway_not_observed", "not_observed", guards.get("actual_headway"), "Actual headway must remain not_observed.")
    add_check(checks, "actual_arrival_departure_not_observed", "not_observed", guards.get("actual_arrival_departure_time"), "Actual arrival/departure must remain not_observed.")
    add_check(checks, "actual_dwell_not_observed", "not_observed", guards.get("actual_dwell"), "Actual dwell must remain not_observed.")
    add_check(checks, "actual_passenger_wait_false", False, guards.get("actual_passenger_wait_observed"), "Actual passenger wait must remain false.")
    add_check(checks, "queue_demand_observed_false", False, guards.get("queue_demand_observed"), "Queue demand observed must remain false.")
    add_check(checks, "queue_demand_proxy_true", True, guards.get("queue_demand_proxy"), "Queue demand proxy must remain true.")

    service = gate.get("service_quality_first_requirements", {})
    for key in [
        "service_weight_must_dominate_energy_fleet",
        "long_wait_penalty_at_least_avg_wait_penalty",
        "fleet_reduction_bonus_must_remain_secondary",
        "energy_proxy_total_must_not_be_direct_reward",
        "passenger_service_rate_must_be_direct_reward_candidate",
        "passenger_wait_p95_seconds_must_be_direct_reward_candidate",
    ]:
        add_check(checks, f"service_quality_{key}", True, service.get(key), f"{key} must be true.")

    decision = gate.get("promotion_decision", {})
    add_check(checks, "decision_not_promoted", "NOT_PROMOTED", decision.get("decision"), "Decision must be NOT_PROMOTED.")
    add_check(checks, "decision_trainable_false", False, decision.get("trainable_reward_promoted"), "Decision must not promote reward.")
    add_check(checks, "decision_train_allowed_false", False, decision.get("train_with_this_reward_allowed"), "Decision must not allow training.")
    add_check(checks, "decision_next_step", "Step117 reward ablation result schema and collector", decision.get("next_step"), "Next step must be Step117 result schema/collector.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_trainable_reward_promotion_gate_step116_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": "PASS_TRAINABLE_REWARD_PROMOTION_GATE_NOT_PROMOTED" if audit_status == "PASS" else "FAIL_TRAINABLE_REWARD_PROMOTION_GATE",
        "next_status": "READY_FOR_STEP117_REWARD_ABLATION_RESULT_SCHEMA" if audit_status == "PASS" else "BLOCKED_FIX_STEP116_GATE",
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "selected_candidate_claim_allowed": False,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", default="05_training/rewards/trainable_reward_promotion_gate_step116.json")
    parser.add_argument("--output-root", default="artifacts/rewards/trainable_reward_promotion_gate_step116_validation")
    parser.add_argument("--check-artifacts", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    gate_path = resolve_project_path(project_root, args.gate)
    output_root = resolve_project_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not gate_path.exists():
        raise SystemExit(f"gate not found: {gate_path}")

    gate = load_json_any_encoding(gate_path)
    report = validate_gate(gate, project_root=project_root, check_artifacts=bool(args.check_artifacts))
    report["gate_path"] = str(gate_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "trainable_reward_promotion_gate_step116_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 116 trainable reward promotion gate validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] promoted      : {report['trainable_reward_promoted']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] best_claim    : {report['selected_candidate_claim_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
