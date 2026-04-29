from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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

REQUIRED_FALSE_GUARDS = [
    "reward_candidate_protocol_finalized",
    "candidate_selected",
    "reward_weights_locked",
    "normalization_locked",
    "hard_constraints_locked",
    "baseline_reference_locked",
    "train_with_candidate_reward_allowed",
    "train_with_this_reward_allowed",
    "final_reward_design_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "canonical_causal_comparison_allowed",
]

REQUIRED_REJECTION_TEXT = [
    "non-positive",
    "service-quality weight sum",
    "w_long_wait",
    "w_fleet_reduction",
    "energy_proxy",
    "train_with_candidate_reward_allowed",
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


def service_weight_sum(weights: Dict[str, Any]) -> float:
    return (
        abs(float(weights.get("w_service", 0.0)))
        + abs(float(weights.get("w_avg_wait", 0.0)))
        + abs(float(weights.get("w_long_wait", 0.0)))
        + abs(float(weights.get("w_ontime", 0.0)))
        + abs(float(weights.get("w_bunching", 0.0)))
        + abs(float(weights.get("w_headway_cv", 0.0)))
    )


def secondary_weight_sum(weights: Dict[str, Any]) -> float:
    return (
        abs(float(weights.get("w_energy_per_passenger", 0.0)))
        + abs(float(weights.get("w_fleet_reduction", 0.0)))
    )


def validate_candidate(candidate: Dict[str, Any], checks: List[Dict[str, Any]]) -> None:
    cid = str(candidate.get("candidate_id", "__MISSING__"))
    weights = candidate.get("weights", {})

    add_check(checks, f"candidate_{cid}_selection_status", "not_selected", candidate.get("selection_status"), "Candidate must not be selected in Step 112.")
    add_check(checks, f"candidate_{cid}_trainable", False, candidate.get("trainable"), "Candidate must not be trainable in Step 112.")
    add_check(checks, f"candidate_{cid}_performance_claim_allowed", False, candidate.get("performance_claim_allowed"), "Candidate must not allow performance claims.")

    for key in EXPECTED_WEIGHTS:
        add_check(checks, f"candidate_{cid}_weight_exists_{key}", True, key in weights, f"{key} must exist in candidate weights.")
        if key in weights:
            add_check(checks, f"candidate_{cid}_weight_positive_{key}", True, float(weights[key]) > 0.0, f"{key} must be positive magnitude.")

    s_sum = service_weight_sum(weights)
    sec_sum = secondary_weight_sum(weights)
    add_check(checks, f"candidate_{cid}_service_dominates_secondary", True, s_sum > sec_sum, "Service-quality weights must dominate energy/fleet weights.")
    add_check(checks, f"candidate_{cid}_long_wait_ge_avg_wait", True, float(weights.get("w_long_wait", 0.0)) >= float(weights.get("w_avg_wait", 0.0)), "Long-wait penalty must be at least average-wait penalty.")
    add_check(checks, f"candidate_{cid}_fleet_le_energy", True, float(weights.get("w_fleet_reduction", 0.0)) <= float(weights.get("w_energy_per_passenger", 0.0)), "Fleet bonus must not exceed energy-per-passenger penalty.")
    add_check(checks, f"candidate_{cid}_constraint_weight_floor", True, float(weights.get("w_constraint", 0.0)) >= 10.0, "Constraint penalty must remain strong enough.")


def validate_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "artifact_version", "reward_candidate_protocol_step112_v1", protocol.get("artifact_version"), "Artifact version must match Step 112 protocol.")
    add_check(checks, "protocol_status", "DRAFT_NOT_TRAINABLE", protocol.get("protocol_status"), "Protocol must remain draft and not trainable.")
    add_check(checks, "weight_sign_convention", "positive_magnitude_with_formula_sign", protocol.get("weight_sign_convention"), "Weight sign convention must match Step 111.")

    guards = protocol.get("claim_guards", {})
    for key in REQUIRED_FALSE_GUARDS:
        add_check(checks, f"claim_guard_{key}", False, guards.get(key), f"{key} must remain false.")

    selection = protocol.get("selection_policy", {})
    add_check(checks, "selection_status", "not_selected", selection.get("selection_status"), "No candidate may be selected in Step 112.")
    add_check(checks, "selection_allowed_in_step112", False, selection.get("selection_allowed_in_step112"), "Selection must not be allowed in Step 112.")
    add_check(checks, "same_windows_required", True, selection.get("same_windows_required"), "Ablation must require same windows.")
    add_check(checks, "same_seeds_required", True, selection.get("same_seeds_required"), "Ablation must require same seeds.")
    add_check(checks, "qwen_trigger_rate_required", 0.0, float(selection.get("qwen_trigger_rate_required", -1.0)), "A-family reward candidate protocol requires qwen_trigger_rate=0.0.")

    common = protocol.get("candidate_common_constraints", {})
    for key in [
        "service_quality_first",
        "energy_fleet_secondary",
        "energy_proxy_not_primary_objective",
        "fleet_reduction_bonus_only",
        "no_energy_saving_by_stranding_passengers",
        "long_wait_fairness_required",
    ]:
        add_check(checks, f"common_{key}", True, common.get(key), f"{key} must be true.")
    add_check(checks, "common_trainable", False, common.get("trainable"), "Common trainable flag must be false.")
    add_check(checks, "common_performance_claim_allowed", False, common.get("performance_claim_allowed"), "Common performance claim flag must be false.")

    normalization = protocol.get("normalization_policy_draft_not_locked", {})
    add_check(checks, "normalization_candidate_training_baseline", "B1_noop", normalization.get("candidate_training_baseline"), "Training baseline candidate must remain B1_noop until Step 113.")
    add_check(checks, "normalization_baseline_reference_locked", False, normalization.get("baseline_reference_locked"), "Baseline reference must not be locked in Step 112.")
    add_check(checks, "normalization_locked", False, normalization.get("normalization_locked"), "Normalization must not be locked in Step 112.")

    candidates = protocol.get("candidate_matrix", [])
    add_check(checks, "candidate_count_at_least_5", True, len(candidates) >= 5, "At least five reward candidates are expected.")
    candidate_ids = [c.get("candidate_id") for c in candidates]
    add_check(checks, "candidate_ids_unique", True, len(candidate_ids) == len(set(candidate_ids)), "Candidate IDs must be unique.")

    for candidate in candidates:
        validate_candidate(candidate, checks)

    rejection_text = "\n".join(str(x) for x in protocol.get("candidate_rejection_rules", []))
    for phrase in REQUIRED_REJECTION_TEXT:
        add_check(checks, f"rejection_rule_mentions_{phrase}", True, phrase in rejection_text, f"Rejection rules must mention {phrase}.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_candidate_protocol_step112_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": "PASS_REWARD_CANDIDATE_PROTOCOL_DRAFT_NOT_TRAINABLE" if audit_status == "PASS" else "FAIL_REWARD_CANDIDATE_PROTOCOL_GUARD_VIOLATION",
        "next_status": "READY_FOR_STEP113_NORMALIZATION_BASELINE_LOCK" if audit_status == "PASS" else "BLOCKED_FIX_STEP112_PROTOCOL",
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "train_with_candidate_reward_allowed": False,
        "candidate_selected": False,
        "reward_weights_locked": False,
        "normalization_locked": False,
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
    parser.add_argument("--protocol", default="05_training/rewards/reward_candidate_protocol_step112.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_candidate_protocol_step112_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    protocol_path = resolve_path(project_root, args.protocol)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not protocol_path.exists():
        raise SystemExit(f"protocol not found: {protocol_path}")

    protocol = load_json_any_encoding(protocol_path)
    report = validate_protocol(protocol)
    report["protocol_path"] = str(protocol_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_candidate_protocol_step112_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 112 reward candidate protocol validation completed")
    print(f"[OK] audit_status      : {report['audit_status']}")
    print(f"[OK] gate_status       : {report['gate_status']}")
    print(f"[OK] next_status       : {report['next_status']}")
    print(f"[OK] candidate_count   : {report['candidate_count']}")
    print(f"[OK] report_json       : {report_path}")
    print(f"[OK] train_allowed     : {report['train_with_candidate_reward_allowed']}")
    print(f"[OK] candidate_selected: {report['candidate_selected']}")
    print(f"[OK] failure_count     : {report['failure_count']}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
