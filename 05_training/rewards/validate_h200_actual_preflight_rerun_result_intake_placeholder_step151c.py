from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_STATUSES = {
    "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED",
    "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED",
}
EXPECTED_DECISION = "INTAKE_PLACEHOLDER_ONLY_EXECUTION_STILL_LOCKED"


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def validate(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    status = manifest.get("intake_status")
    require(manifest.get("step") == "151-C", "step must be 151-C", errors)
    require(status in ALLOWED_STATUSES, f"intake_status must be one of {sorted(ALLOWED_STATUSES)}", errors)
    require(manifest.get("decision") == EXPECTED_DECISION, f"decision must be {EXPECTED_DECISION}", errors)
    require(int(manifest.get("hard_failures", -1)) == 0, "hard_failures must be 0", errors)

    false_fields = [
        "actual_execution_allowed",
        "actual_execution_released",
        "train_allowed",
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "operator_approval_recorded",
    ]

    for field in false_fields:
        require(manifest.get(field) is False, f"{field} must be false", errors)

    require(manifest.get("h200_preflight_result_intake_placeholder_ready") is True, "h200_preflight_result_intake_placeholder_ready must be true", errors)
    require(manifest.get("actual_release_requires_separate_manifest") is True, "actual_release_requires_separate_manifest must be true", errors)

    if status == "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED":
        require(manifest.get("h200_expected_preflight_manifest_present") is False, "waiting status requires h200 manifest present=false", errors)
        require(manifest.get("h200_expected_preflight_passed") is False, "waiting status requires h200 passed=false", errors)

    if status == "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED":
        require(manifest.get("h200_expected_preflight_manifest_present") is True, "validated status requires h200 manifest present=true", errors)
        require(manifest.get("h200_expected_preflight_passed") is True, "validated status requires h200 passed=true", errors)

    required_before = set(manifest.get("required_before_actual_release", []))
    required_items = {
        "h200_expected_preflight_manifest_present_true",
        "h200_expected_preflight_passed_true",
        "operator_approval_recorded_true",
        "actual_execution_release_manifest_separate_from_intake_placeholder",
        "output_root_empty_or_archived",
        "git_commit_pinned",
    }
    missing = sorted(required_items - required_before)
    require(not missing, f"missing required_before_actual_release items: {missing}", errors)

    checks = manifest.get("checks", [])
    require(isinstance(checks, list) and len(checks) >= 10, "checks must contain at least 10 entries", errors)
    for check in checks:
        if check.get("severity") == "hard":
            require(check.get("passed") is True, f"hard check failed: {check.get('check_id')}", errors)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    path = Path(args.manifest)
    manifest = load_json(path)
    errors = validate(manifest)

    if errors:
        print("[FAIL] Step 151-C manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 151-C manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
