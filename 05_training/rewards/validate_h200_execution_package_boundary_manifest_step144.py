from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "h200_execution_package_boundary_manifest_step144_v1"
CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

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


def recursive_forbidden_true(obj: Any, prefix: str = "") -> List[str]:
    errors: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and bool(value):
                errors.append(f"forbidden_true: {path}")
            if isinstance(value, (dict, list)):
                errors.extend(recursive_forbidden_true(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            if isinstance(value, (dict, list)):
                errors.extend(recursive_forbidden_true(value, path))
    return errors


def validate_boundary(boundary: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if boundary.get("package_boundary_version") != "h200_execution_package_boundary_v1":
        errors.append("boundary_version_mismatch")
    if boundary.get("created_from_step") != 144:
        errors.append("boundary_created_from_step_must_be_144")

    matrix = boundary.get("matrix_source", {})
    if int(matrix.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("boundary_matrix_planned_run_count_must_be_72")
    if matrix.get("conditions") != CONDITIONS:
        errors.append("boundary_matrix_conditions_mismatch")
    if matrix.get("reward_ids") != REWARD_IDS:
        errors.append("boundary_matrix_reward_ids_mismatch")
    if matrix.get("seeds") != SEEDS:
        errors.append("boundary_matrix_seeds_mismatch")

    decision = boundary.get("execution_boundary", {})
    if bool(decision.get("actual_execution_allowed", False)):
        errors.append("boundary_actual_execution_allowed_true")
    if bool(decision.get("actual_execution_released", False)):
        errors.append("boundary_actual_execution_released_true")
    if bool(decision.get("release_ready", False)):
        errors.append("boundary_release_ready_true")

    guards = boundary.get("non_claim_guards", {})
    if not isinstance(guards, dict) or not guards:
        errors.append("boundary_non_claim_guards_missing")
    else:
        for key, value in guards.items():
            if bool(value):
                errors.append(f"boundary_non_claim_guard_true: {key}")

    return errors


def validate_payload(payload: Dict[str, Any], boundary: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 144:
        errors.append("step_must_be_144")

    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")

    if payload.get("package_status") not in {
        "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED",
        "H200_EXECUTION_PACKAGE_BOUNDARY_BLOCKED",
    }:
        errors.append("invalid_package_status")
    if require_pass and payload.get("package_status") != "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED":
        errors.append("package_status_not_ready_locked")

    if int(payload.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("payload_planned_run_count_must_be_72")
    if payload.get("conditions") != CONDITIONS:
        errors.append("payload_conditions_mismatch")
    if payload.get("reward_ids") != REWARD_IDS:
        errors.append("payload_reward_ids_mismatch")
    if payload.get("seeds") != SEEDS:
        errors.append("payload_seeds_mismatch")

    errors.extend(recursive_forbidden_true(payload))
    errors.extend(validate_boundary(boundary))

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    boundary = load_json(Path(args.boundary))
    result = validate_payload(payload, boundary, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 144 H200 execution package boundary manifest validation PASS")


if __name__ == "__main__":
    main()
