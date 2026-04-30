from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_STATUS = "RELEASE_MANIFEST_DRAFT_STILL_LOCKED"
EXPECTED_DECISION = "DRAFT_ONLY_NOT_RELEASED"
EXPECTED_NEXT_GATE = "READY_FOR_STEP151B_OPERATOR_APPROVAL_DRAFT_OR_H200_EXPECTED_PREFLIGHT_RERUN"


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

    require(manifest.get("step") == "151-A", "step must be 151-A", errors)
    require(
        manifest.get("release_manifest_status") == EXPECTED_STATUS,
        f"release_manifest_status must be {EXPECTED_STATUS}",
        errors,
    )
    require(
        manifest.get("decision") == EXPECTED_DECISION,
        f"decision must be {EXPECTED_DECISION}",
        errors,
    )
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
        "h200_expected_preflight_passed",
    ]

    for field in false_fields:
        require(manifest.get(field) is False, f"{field} must be false", errors)

    require(manifest.get("h200_expected_preflight_required") is True, "h200_expected_preflight_required must be true", errors)
    require(
        manifest.get("h200_expected_preflight_required_args") == "--expect-h200 --min-gpu-count 1",
        "h200_expected_preflight_required_args mismatch",
        errors,
    )
    require(manifest.get("next_gate") == EXPECTED_NEXT_GATE, f"next_gate must be {EXPECTED_NEXT_GATE}", errors)

    required_before = set(manifest.get("required_before_actual_release", []))
    required_items = {
        "operator_approval_recorded_true",
        "real_h200_step149_expect_h200_pass",
        "cuda_available_true_on_h200",
        "gpu_count_at_least_1_on_h200",
        "git_commit_pinned",
        "worktree_clean_or_dirty_diff_recorded",
        "output_root_empty_or_archived",
        "actual_execution_release_manifest_separate_from_draft",
    }
    missing = sorted(required_items - required_before)
    require(not missing, f"missing required_before_actual_release items: {missing}", errors)

    checks = manifest.get("checks", [])
    require(isinstance(checks, list) and len(checks) >= 10, "checks must contain at least 10 entries", errors)
    for check in checks:
        if check.get("severity") == "hard":
            require(check.get("passed") is True, f"hard check failed: {check.get('check_id')}", errors)

    scope = manifest.get("release_scope", {})
    require(scope.get("target_run_count") == 72, "release_scope.target_run_count must be 72", errors)
    require(scope.get("release_type") == "draft_only_still_locked", "release_scope.release_type mismatch", errors)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    path = Path(args.manifest)
    manifest = load_json(path)
    errors = validate(manifest)

    if errors:
        print("[FAIL] Step 151-A manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 151-A manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
