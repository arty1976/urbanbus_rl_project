from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_CANDIDATE_IDS = [
    "R0_SERVICE_SAFETY",
    "R1_BALANCED_DEFAULT",
    "R2_WAIT_HEAVY",
    "R3_LONG_WAIT_FAIRNESS",
    "R4_ENERGY_LIGHT",
    "R5_FLEET_CAUTIOUS",
]

EXPECTED_WEIGHT_KEYS = [
    "w_service",
    "w_avg_wait",
    "w_long_wait",
    "w_ontime",
    "w_bunching",
    "w_headway_cv",
    "w_energy_per_passenger",
    "w_fleet_reduction",
    "w_intervention",
    "w_constraint",
]

REQUIRED_CONSTRAINT_IDS = [
    "service_rate_floor",
    "avg_wait_regression_cap",
    "p95_wait_regression_cap",
    "qwen_off_for_A_family",
    "no_actual_observation_overclaim",
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


def weight_service_sum(weights: Dict[str, Any]) -> float:
    return float(weights.get("w_service", 0.0)) + float(weights.get("w_avg_wait", 0.0)) + float(weights.get("w_long_wait", 0.0)) + float(weights.get("w_ontime", 0.0)) + float(weights.get("w_bunching", 0.0)) + float(weights.get("w_headway_cv", 0.0))


def weight_secondary_sum(weights: Dict[str, Any]) -> float:
    return float(weights.get("w_energy_per_passenger", 0.0)) + float(weights.get("w_fleet_reduction", 0.0))


def validate_matrix(spec: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(
        checks,
        "artifact_version",
        "reward_ablation_matrix_step115_v1",
        spec.get("artifact_version"),
        "Artifact version must match Step 115.",
    )
    add_check(
        checks,
        "matrix_status",
        "DRAFT_NOT_TRAINABLE",
        spec.get("matrix_status"),
        "Step 115 matrix must remain draft and not trainable.",
    )
    add_check(
        checks,
        "weight_sign_convention",
        "positive_magnitude_with_formula_sign",
        spec.get("weight_sign_convention"),
        "Weights must use positive magnitude convention.",
    )

    guards = spec.get("claim_guards", {})
    for key in [
        "train_with_any_candidate_allowed",
        "best_reward_claim_allowed",
        "reward_formula_finalized",
        "reward_weights_locked",
        "normalization_numeric_values_locked",
        "hard_constraints_finalized",
        "final_reward_design_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "canonical_causal_comparison_allowed",
    ]:
        add_check(
            checks,
            f"claim_guard_{key}",
            False,
            guards.get(key),
            f"{key} must remain false.",
        )

    baseline = spec.get("baseline_reference", {})
    add_check(checks, "baseline_id", "B1_noop", baseline.get("normalization_baseline_id"), "Normalization baseline must be B1_noop.")
    add_check(checks, "baseline_identity_locked", True, baseline.get("baseline_reference_locked"), "Baseline identity must be locked.")
    add_check(checks, "numeric_baseline_values_unlocked", False, baseline.get("numeric_baseline_values_locked"), "Numeric baseline values must remain unlocked.")

    rules = spec.get("common_design_rules", {})
    for key in [
        "service_quality_first",
        "energy_fleet_secondary",
        "energy_proxy_not_primary_objective",
        "fleet_reduction_bonus_only",
        "all_candidates_share_hard_constraints",
        "all_candidates_share_baseline",
        "no_candidate_may_reward_service_collapse",
    ]:
        add_check(checks, f"design_rule_{key}", True, rules.get(key), f"{key} must be true.")

    constraints = spec.get("common_hard_constraints", [])
    constraint_ids = {str(c.get("constraint_id")) for c in constraints}
    for cid in REQUIRED_CONSTRAINT_IDS:
        add_check(checks, f"constraint_{cid}", True, cid in constraint_ids, f"Required common hard constraint missing: {cid}")

    candidates = spec.get("candidate_weight_sets", [])
    candidate_ids = [str(c.get("candidate_id")) for c in candidates]
    add_check(checks, "candidate_ids_exact", EXPECTED_CANDIDATE_IDS, candidate_ids, "Candidate list must match expected order.")

    candidate_count = len(candidates)
    add_check(checks, "candidate_count", 6, candidate_count, "Exactly six candidate sets are expected.")

    per_candidate_summary: List[Dict[str, Any]] = []

    for candidate in candidates:
        cid = str(candidate.get("candidate_id"))
        weights = candidate.get("weights", {})

        for key in EXPECTED_WEIGHT_KEYS:
            add_check(checks, f"{cid}_weight_exists_{key}", True, key in weights, f"{cid} must define {key}.")
            if key in weights:
                value = float(weights[key])
                add_check(checks, f"{cid}_weight_non_negative_{key}", True, value >= 0.0, f"{cid}.{key} must be non-negative.")

        add_check(checks, f"{cid}_service_positive", True, float(weights.get("w_service", 0.0)) > 0.0, f"{cid}.w_service must be positive.")
        add_check(checks, f"{cid}_constraint_positive", True, float(weights.get("w_constraint", 0.0)) > 0.0, f"{cid}.w_constraint must be positive.")

        service_sum = weight_service_sum(weights)
        secondary_sum = weight_secondary_sum(weights)

        add_check(
            checks,
            f"{cid}_service_dominates_secondary",
            True,
            service_sum > secondary_sum,
            f"{cid}: service-quality weights must dominate energy/fleet weights.",
        )
        add_check(
            checks,
            f"{cid}_long_wait_ge_avg_wait",
            True,
            float(weights.get("w_long_wait", 0.0)) >= float(weights.get("w_avg_wait", 0.0)),
            f"{cid}: long-wait fairness penalty must be at least average-wait penalty.",
        )
        add_check(
            checks,
            f"{cid}_fleet_not_above_energy",
            True,
            float(weights.get("w_fleet_reduction", 0.0)) <= float(weights.get("w_energy_per_passenger", 0.0)),
            f"{cid}: fleet bonus must not exceed energy-per-passenger penalty.",
        )

        per_candidate_summary.append({
            "candidate_id": cid,
            "service_weight_sum": service_sum,
            "secondary_weight_sum": secondary_sum,
            "service_to_secondary_ratio": None if secondary_sum == 0 else service_sum / secondary_sum,
            "weights": weights,
        })

    selection = spec.get("selection_rules_not_active_yet", {})
    add_check(checks, "selection_status", "NOT_ALLOWED_IN_STEP115", selection.get("selection_status"), "Candidate selection must not be active in Step 115.")

    add_check(checks, "required_next_gate", "Step116 trainable reward promotion gate", spec.get("required_next_gate"), "Next gate must be Step116.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_ablation_matrix_step115_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "matrix_status": spec.get("matrix_status"),
        "gate_status": (
            "PASS_REWARD_ABLATION_MATRIX_DRAFT_NOT_TRAINABLE"
            if audit_status == "PASS"
            else "FAIL_REWARD_ABLATION_MATRIX_GUARD_VIOLATION"
        ),
        "next_status": (
            "READY_FOR_STEP116_TRAINABLE_REWARD_PROMOTION_GATE"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP115_MATRIX"
        ),
        "candidate_count": candidate_count,
        "candidate_ids": candidate_ids,
        "train_with_any_candidate_allowed": False,
        "best_reward_claim_allowed": False,
        "reward_weights_locked": False,
        "hard_constraints_finalized": False,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
        "per_candidate_summary": per_candidate_summary,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="05_training/rewards/reward_ablation_matrix_step115.json")
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_ablation_matrix_step115_validation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    matrix_path = resolve_path(project_root, args.matrix)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not matrix_path.exists():
        raise SystemExit(f"matrix not found: {matrix_path}")

    spec = load_json_any_encoding(matrix_path)
    report = validate_matrix(spec)
    report["matrix_path"] = str(matrix_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_ablation_matrix_step115_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 115 reward ablation matrix validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] candidate_count: {report['candidate_count']}")
    print(f"[OK] train_allowed : {report['train_with_any_candidate_allowed']}")
    print(f"[OK] best_claim    : {report['best_reward_claim_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
