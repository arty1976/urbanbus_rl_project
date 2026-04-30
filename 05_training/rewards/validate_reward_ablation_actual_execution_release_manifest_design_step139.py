from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_ablation_actual_execution_release_manifest_design_step139_v1"
FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "actual_executed",
    "actual_results",
    "reward_result_written",
    "winner_selected",
    "trainable_reward_promoted",
    "train_with_this_reward_allowed",
    "actual_training_allowed",
    "final_reward_design_claim_allowed",
    "best_reward_claim_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_json: {path}")


def validate_release_template(template: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if template.get("release_manifest_version") != "actual_reward_ablation_release_manifest_v1":
        errors.append("template_version_mismatch")
    matrix = template.get("requested_matrix", {})
    if matrix.get("conditions") != ["A"]:
        errors.append("template_matrix_conditions_must_be_A_only")
    if matrix.get("reward_ids") != ["R0", "R1", "R2", "R3", "R4", "R5"]:
        errors.append("template_matrix_reward_ids_mismatch")
    if matrix.get("seeds") != [1, 2, 3]:
        errors.append("template_matrix_seeds_mismatch")
    if int(matrix.get("planned_run_count", 0)) != 18:
        errors.append("template_matrix_planned_run_count_must_be_18")
    decision = template.get("release_decision", {})
    if bool(decision.get("actual_execution_allowed", False)):
        errors.append("template_must_not_allow_actual_execution")
    if bool(decision.get("actual_execution_released", False)):
        errors.append("template_must_not_release_actual_execution")
    if bool(decision.get("release_ready", False)):
        errors.append("template_release_ready_must_be_false")
    approval_gates = template.get("approval_gates", {})
    if bool(approval_gates.get("operator_explicit_approval", False)):
        errors.append("operator_explicit_approval_must_be_false_in_template")
    non_claim_guards = template.get("non_claim_guards", {})
    for key, value in non_claim_guards.items():
        if bool(value):
            errors.append(f"template_non_claim_guard_must_be_false: {key}")
    phrase = template.get("required_operator_approval_phrase", "")
    if "I APPROVE ACTUAL REWARD ABLATION EXECUTION" not in phrase:
        errors.append("required_operator_approval_phrase_missing")
    return errors


def validate_payload(payload: Dict[str, Any], template: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 139:
        errors.append("step_must_be_139")
    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")
    design_status = payload.get("design_status")
    valid_statuses = {
        "RELEASE_MANIFEST_SCHEMA_DESIGNED_ACTUAL_STILL_LOCKED",
        "RELEASE_MANIFEST_SCHEMA_DESIGN_BLOCKED",
    }
    if design_status not in valid_statuses:
        errors.append("invalid_design_status")
    if require_pass and design_status != "RELEASE_MANIFEST_SCHEMA_DESIGNED_ACTUAL_STILL_LOCKED":
        errors.append("design_status_not_ready")
    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")
    errors.extend(validate_release_template(template))
    return {"validation_status": "PASS" if not errors else "FAIL", "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    template = load_json(Path(args.template))
    result = validate_payload(payload, template, require_pass=not args.allow_blocked)
    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)
    print("[OK] Step 139 release manifest design validation PASS")


if __name__ == "__main__":
    main()
