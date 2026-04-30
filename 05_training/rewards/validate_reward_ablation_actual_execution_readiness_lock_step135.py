from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_ablation_actual_execution_readiness_lock_step135_v1"

FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed",
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

    if payload.get("step") != 135:
        errors.append("step_must_be_135")

    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")

    lock_status = payload.get("readiness_lock_status")
    valid_lock_status = {"LOCKED_READY_FOR_EXPLICIT_RELEASE", "LOCKED_BLOCKED"}
    if lock_status not in valid_lock_status:
        errors.append("invalid_readiness_lock_status")

    if require_pass and lock_status != "LOCKED_READY_FOR_EXPLICIT_RELEASE":
        errors.append("readiness_lock_status_not_ready_for_release")

    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")

    release_requirements = payload.get("release_requirements", [])
    if not isinstance(release_requirements, list) or not release_requirements:
        errors.append("release_requirements_missing_or_invalid")
    else:
        for req in release_requirements:
            if not isinstance(req, dict):
                errors.append("release_requirement_item_invalid")
                continue
            if bool(req.get("satisfied", False)):
                errors.append(f"release_requirement_must_not_be_satisfied_yet: {req.get('requirement_id')}")

    upstream_checks = payload.get("upstream_checks", [])
    if not isinstance(upstream_checks, list) or len(upstream_checks) < 3:
        errors.append("upstream_checks_missing_or_incomplete")

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

    print("[OK] Step 135 actual execution readiness lock manifest validation PASS")


if __name__ == "__main__":
    main()
