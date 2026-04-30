from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_STATUS = "H200_HANDOFF_PACKET_INDEX_READY_STILL_LOCKED"
EXPECTED_DECISION = "INDEX_ONLY_HANDOFF_NOT_RELEASED"


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

    require(manifest.get("step") == "151-E", "step must be 151-E", errors)
    require(manifest.get("handoff_index_status") == EXPECTED_STATUS, f"handoff_index_status must be {EXPECTED_STATUS}", errors)
    require(manifest.get("decision") == EXPECTED_DECISION, f"decision must be {EXPECTED_DECISION}", errors)
    require(int(manifest.get("hard_failures", -1)) == 0, "hard_failures must be 0", errors)

    false_fields = [
        "operator_approval_recorded",
        "operator_approval_granted",
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

    require(manifest.get("handoff_packet_index_only") is True, "handoff_packet_index_only must be true", errors)
    require(manifest.get("actual_release_requires_separate_manifest") is True, "actual_release_requires_separate_manifest must be true", errors)

    manifests = manifest.get("handoff_packet_manifests", [])
    require(isinstance(manifests, list) and len(manifests) == 4, "handoff_packet_manifests must contain 4 entries", errors)

    command = str(manifest.get("h200_step149_command", ""))
    require("--expect-h200" in command, "h200_step149_command must include --expect-h200", errors)
    require("--min-gpu-count 1" in command, "h200_step149_command must include --min-gpu-count 1", errors)

    required_before = set(manifest.get("required_before_actual_release", []))
    required_items = {
        "real_h200_step149_expect_h200_pass",
        "h200_expected_preflight_manifest_present_true",
        "h200_expected_preflight_passed_true",
        "operator_approval_recorded_true",
        "operator_approval_granted_true",
        "separate_actual_execution_release_manifest",
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
        print("[FAIL] Step 151-E manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 151-E manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
