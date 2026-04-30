from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_STATUS = "OPERATOR_APPROVAL_DECISION_DRAFT_STILL_LOCKED"
EXPECTED_DECISION = "APPROVAL_DRAFT_ONLY_NOT_APPROVED_NOT_RELEASED"


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

    require(manifest.get("step") == "151-D", "step must be 151-D", errors)
    require(manifest.get("approval_decision_status") == EXPECTED_STATUS, f"approval_decision_status must be {EXPECTED_STATUS}", errors)
    require(manifest.get("decision") == EXPECTED_DECISION, f"decision must be {EXPECTED_DECISION}", errors)
    require(int(manifest.get("hard_failures", -1)) == 0, "hard_failures must be 0", errors)

    false_fields = [
        "operator_approval_recorded",
        "operator_approval_granted",
        "operator_approval_decision_final",
        "actual_execution_allowed",
        "actual_execution_released",
        "train_allowed",
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]

    for field in false_fields:
        require(manifest.get(field) is False, f"{field} must be false", errors)

    require(manifest.get("actual_release_requires_separate_manifest") is True, "actual_release_requires_separate_manifest must be true", errors)

    required_before = set(manifest.get("required_before_operator_approval_record", []))
    required_items = {
        "h200_expected_preflight_passed_true",
        "real_h200_step149_manifest_archived",
        "operator_identity_recorded",
        "approval_timestamp_recorded",
        "approved_git_commit_recorded",
        "output_root_empty_or_archived",
        "separate_actual_execution_release_manifest",
    }
    missing = sorted(required_items - required_before)
    require(not missing, f"missing required_before_operator_approval_record items: {missing}", errors)

    checks = manifest.get("checks", [])
    require(isinstance(checks, list) and len(checks) >= 10, "checks must contain at least 10 entries", errors)
    for check in checks:
        if check.get("severity") == "hard":
            require(check.get("passed") is True, f"hard check failed: {check.get('check_id')}", errors)

    scope = manifest.get("approval_scope", {})
    require(scope.get("target_run_count") == 72, "approval_scope.target_run_count must be 72", errors)
    require(scope.get("approval_type") == "draft_only_still_locked", "approval_scope.approval_type mismatch", errors)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    path = Path(args.manifest)
    manifest = load_json(path)
    errors = validate(manifest)

    if errors:
        print("[FAIL] Step 151-D manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 151-D manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
