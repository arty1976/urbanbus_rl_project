from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "actual_reward_ablation_runner_implementation_review_step137_v1"

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

    if payload.get("step") != 137:
        errors.append("step_must_be_137")

    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")

    review_status = payload.get("review_status")
    valid_statuses = {
        "RUNNER_GUARD_REVIEW_PASS_ACTUAL_STILL_LOCKED",
        "RUNNER_GUARD_REVIEW_BLOCKED",
    }
    if review_status not in valid_statuses:
        errors.append("invalid_review_status")

    if require_pass and review_status != "RUNNER_GUARD_REVIEW_PASS_ACTUAL_STILL_LOCKED":
        errors.append("review_status_not_guard_pass")

    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")

    static_review = payload.get("static_review", {})
    if not isinstance(static_review, dict):
        errors.append("static_review_missing_or_invalid")
    elif require_pass and static_review.get("violations"):
        errors.append("static_review_has_violations")

    dynamic_review = payload.get("dynamic_review", {})
    if not isinstance(dynamic_review, dict):
        errors.append("dynamic_review_missing_or_invalid")
    elif require_pass and dynamic_review.get("violations"):
        errors.append("dynamic_review_has_violations")

    actual_attempt = dynamic_review.get("actual_attempt", {}) if isinstance(dynamic_review, dict) else {}
    if require_pass:
        if int(actual_attempt.get("returncode", 0)) == 0:
            errors.append("actual_attempt_must_return_nonzero")
        if actual_attempt.get("audit_status") != "BLOCKED":
            errors.append("actual_attempt_audit_status_must_be_blocked")

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

    print("[OK] Step 137 runner implementation review manifest validation PASS")


if __name__ == "__main__":
    main()
