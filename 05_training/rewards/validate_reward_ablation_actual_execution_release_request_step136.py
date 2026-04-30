from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_ablation_actual_execution_release_request_step136_v1"

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


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")

    if payload.get("step") != 136:
        errors.append("step_must_be_136")

    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")

    request_status = payload.get("release_request_status")
    valid_statuses = {
        "REQUEST_PACKAGE_CREATED_PENDING_OPERATOR_APPROVAL",
        "REQUEST_PACKAGE_BLOCKED",
    }
    if request_status not in valid_statuses:
        errors.append("invalid_release_request_status")

    if require_pass and request_status != "REQUEST_PACKAGE_CREATED_PENDING_OPERATOR_APPROVAL":
        errors.append("release_request_status_not_pending_operator_approval")

    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")

    matrix = payload.get("requested_matrix", {})
    if not isinstance(matrix, dict):
        errors.append("requested_matrix_missing_or_invalid")
    else:
        if matrix.get("conditions") != ["A"]:
            errors.append("requested_matrix_conditions_must_be_A_only")
        if matrix.get("reward_ids") != ["R0", "R1", "R2", "R3", "R4", "R5"]:
            errors.append("requested_matrix_reward_ids_mismatch")
        if matrix.get("seeds") != [1, 2, 3]:
            errors.append("requested_matrix_seeds_mismatch")
        if int(matrix.get("planned_run_count", 0)) != 18:
            errors.append("requested_matrix_planned_run_count_must_be_18")

    checklist = payload.get("operator_checklist", [])
    if not isinstance(checklist, list) or not checklist:
        errors.append("operator_checklist_missing_or_invalid")
    else:
        for item in checklist:
            if not isinstance(item, dict):
                errors.append("operator_checklist_item_invalid")
                continue
            if bool(item.get("confirmed", False)):
                errors.append(f"operator_checklist_must_not_be_confirmed_yet: {item.get('item_id')}")

    if not bool(payload.get("release_manifest_required_next", False)):
        errors.append("release_manifest_required_next_must_be_true")

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 136 release request package manifest validation PASS")


if __name__ == "__main__":
    main()
