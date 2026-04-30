from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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

REQUIRED_DIRECT_REWARD_TERMS = [
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

EVALUATION_ONLY_TERMS = [
    "passenger_demand_generated",
    "passenger_served_count",
    "energy_proxy",
]

EXPECTED_WEIGHTS = [
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


def validate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(
        checks,
        "artifact_version",
        "final_reward_spec_step111_v1",
        spec.get("artifact_version"),
        "Artifact version must match Step 111 spec.",
    )

    add_check(
        checks,
        "spec_status",
        "DRAFT_NOT_TRAINABLE",
        spec.get("spec_status"),
        "Step 111 spec must remain draft and not trainable.",
    )

    add_check(
        checks,
        "weight_sign_convention",
        "positive_magnitude_with_formula_sign",
        spec.get("weight_sign_convention"),
        "Weights must be positive magnitudes and formula signs must determine reward direction.",
    )

    guards = spec.get("claim_guards", {})
    required_false_guards = [
        "reward_formula_finalized",
        "reward_weights_locked",
        "normalization_locked",
        "hard_constraints_locked",
        "baseline_reference_locked",
        "train_with_this_reward_allowed",
        "final_reward_design_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "canonical_causal_comparison_allowed",
    ]

    for key in required_false_guards:
        add_check(
            checks,
            f"claim_guard_{key}",
            False,
            guards.get(key),
            f"{key} must remain false in Step 111.",
        )

    obs = spec.get("observability_guards", {})
    add_check(checks, "actual_headway_guard", "not_observed", obs.get("actual_headway"), "Actual headway must remain not_observed.")
    add_check(checks, "actual_arrival_departure_guard", "not_observed", obs.get("actual_arrival_departure_time"), "Actual arrival/departure time must remain not_observed.")
    add_check(checks, "actual_dwell_guard", "not_observed", obs.get("actual_dwell"), "Actual dwell must remain not_observed.")
    add_check(checks, "actual_passenger_wait_guard", False, obs.get("actual_passenger_wait_observed"), "Actual passenger wait must remain unobserved.")
    add_check(checks, "queue_demand_observed_guard", False, obs.get("queue_demand_observed"), "Queue demand observed flag must remain false.")
    add_check(checks, "queue_demand_proxy_guard", True, obs.get("queue_demand_proxy"), "Queue demand proxy flag must remain true.")

    principles = spec.get("design_principles", {})
    required_true_principles = [
        "service_quality_first",
        "energy_fleet_secondary",
        "energy_proxy_not_primary_objective",
        "fleet_reduction_bonus_only",
        "no_energy_saving_by_stranding_passengers",
        "long_wait_fairness_required",
    ]

    for key in required_true_principles:
        add_check(
            checks,
            f"principle_{key}",
            True,
            principles.get(key),
            f"{key} must be true.",
        )

    kpis = spec.get("canonical_12_kpis", [])
    add_check(
        checks,
        "canonical_12_kpis_exact",
        EXPECTED_12_KPIS,
        kpis,
        "Canonical 12-KPI list must match expected order and content.",
    )

    cls = spec.get("reward_term_classification", {})
    direct_terms = cls.get("direct_reward_candidate_terms", [])
    eval_only_terms = cls.get("evaluation_or_context_only_terms", [])

    for term in REQUIRED_DIRECT_REWARD_TERMS:
        add_check(
            checks,
            f"direct_term_{term}",
            True,
            term in direct_terms,
            f"{term} must be listed as a direct reward candidate term.",
        )

    for term in EVALUATION_ONLY_TERMS:
        add_check(
            checks,
            f"eval_only_term_{term}",
            True,
            term in eval_only_terms,
            f"{term} must be preserved as evaluation/context only.",
        )
        add_check(
            checks,
            f"eval_only_not_direct_{term}",
            False,
            term in direct_terms,
            f"{term} must not be a direct reward term.",
        )

    formula = spec.get("reward_formula_draft", {})
    formula_text = str(formula.get("formula_text", ""))
    add_check(checks, "formula_finalized_false", False, formula.get("formula_finalized"), "Formula must not be finalized.")
    add_check(checks, "formula_weights_locked_false", False, formula.get("weights_locked"), "Weights must not be locked.")
    add_check(checks, "formula_normalization_locked_false", False, formula.get("normalization_locked"), "Normalization must not be locked.")
    add_check(checks, "formula_hard_constraints_locked_false", False, formula.get("hard_constraints_locked"), "Hard constraints must not be locked.")
    add_check(checks, "formula_uses_positive_service", True, "+ w_service" in formula_text, "Formula must positively reward service.")
    add_check(checks, "formula_penalizes_avg_wait", True, "- w_avg_wait" in formula_text, "Formula must penalize average wait.")
    add_check(checks, "formula_penalizes_long_wait", True, "- w_long_wait" in formula_text, "Formula must penalize long wait.")
    add_check(checks, "formula_penalizes_energy_per_passenger", True, "- w_energy_per_passenger" in formula_text, "Formula must penalize energy per passenger.")
    add_check(checks, "formula_only_bonus_fleet_reduction", True, "+ w_fleet_reduction" in formula_text, "Fleet reduction must be a bonus term.")

    weights = spec.get("draft_weight_candidates_not_locked", {})

    for key in EXPECTED_WEIGHTS:
        add_check(
            checks,
            f"weight_exists_{key}",
            True,
            key in weights,
            f"{key} must exist.",
        )
        if key in weights:
            add_check(
                checks,
                f"weight_positive_{key}",
                True,
                float(weights[key]) > 0.0,
                f"{key} must be positive magnitude.",
            )

    service_abs = (
        abs(float(weights.get("w_service", 0.0)))
        + abs(float(weights.get("w_avg_wait", 0.0)))
        + abs(float(weights.get("w_long_wait", 0.0)))
        + abs(float(weights.get("w_ontime", 0.0)))
        + abs(float(weights.get("w_bunching", 0.0)))
        + abs(float(weights.get("w_headway_cv", 0.0)))
    )
    secondary_abs = (
        abs(float(weights.get("w_energy_per_passenger", 0.0)))
        + abs(float(weights.get("w_fleet_reduction", 0.0)))
    )

    add_check(
        checks,
        "service_weight_dominates_secondary",
        True,
        service_abs > secondary_abs,
        "Service-quality draft weights must dominate energy/fleet secondary weights.",
    )

    add_check(
        checks,
        "long_wait_weight_at_least_avg_wait",
        True,
        abs(float(weights.get("w_long_wait", 0.0))) >= abs(float(weights.get("w_avg_wait", 0.0))),
        "Long-wait p95 penalty should be at least as strong as average-wait penalty.",
    )

    add_check(
        checks,
        "fleet_weight_less_than_energy_weight",
        True,
        float(weights.get("w_fleet_reduction", 0.0)) < float(weights.get("w_energy_per_passenger", 0.0)),
        "Fleet reduction must remain a smaller bonus than the energy-per-passenger penalty.",
    )

    normalization = spec.get("normalization_draft_not_locked", {})
    add_check(checks, "normalization_locked_false", False, normalization.get("locked"), "Normalization draft must not be locked.")
    add_check(checks, "training_baseline_candidate", "B1_noop", normalization.get("baseline_reference_candidate_for_training_reward"), "Training reward baseline candidate should be B1_noop.")

    hard_constraints = spec.get("hard_constraints_draft_not_locked", [])
    constraint_ids = {c.get("constraint_id") for c in hard_constraints}
    for cid in [
        "service_rate_floor",
        "avg_wait_regression_cap",
        "p95_wait_regression_cap",
        "qwen_off_for_A_family",
        "no_actual_observation_overclaim",
    ]:
        add_check(
            checks,
            f"hard_constraint_{cid}",
            True,
            cid in constraint_ids,
            f"Draft hard constraint {cid} must exist.",
        )

    next_gates = spec.get("required_next_gates_before_training", [])
    for gate in [
        "Step112 reward candidate protocol",
        "Step113 reward normalization baseline lock",
        "Step114 hard constraint review",
        "Step115 reward ablation matrix",
        "Step116 trainable reward promotion gate",
    ]:
        add_check(
            checks,
            f"next_gate_{gate.split()[0].lower()}",
            True,
            gate in next_gates,
            f"{gate} must be listed before training promotion.",
        )

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_final_reward_spec_step111_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "spec_status": spec.get("spec_status"),
        "gate_status": (
            "PASS_FINAL_REWARD_SPEC_DRAFT_NOT_TRAINABLE"
            if audit_status == "PASS"
            else "FAIL_FINAL_REWARD_SPEC_GUARD_VIOLATION"
        ),
        "next_status": (
            "READY_FOR_STEP112_REWARD_CANDIDATE_PROTOCOL"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP111_SPEC"
        ),
        "train_with_this_reward_allowed": False,
        "final_reward_design_claim_allowed": False,
        "reward_formula_finalized": False,
        "reward_weights_locked": False,
        "normalization_locked": False,
        "hard_constraints_locked": False,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
        "service_weight_abs_sum": service_abs,
        "secondary_weight_abs_sum": secondary_abs,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="05_training/rewards/final_reward_spec_step111.json")
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/final_reward_spec_step111_validation",
    )
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

    report_path = output_root / "final_reward_spec_step111_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 111 final reward spec validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] spec_path     : {spec_path}")
    print(f"[OK] report_json   : {report_path}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] formula_final : {report['reward_formula_finalized']}")
    print(f"[OK] weights_locked: {report['reward_weights_locked']}")
    print(f"[OK] failure_count : {report['failure_count']}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
