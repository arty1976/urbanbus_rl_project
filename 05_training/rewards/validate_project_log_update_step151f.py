from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ALLOWED_STATUSES = {"PROJECT_LOG_UPDATED", "PROJECT_LOG_ALREADY_PRESENT"}
EXPECTED_DECISION = "PROJECT_LOG_ONLY_EXECUTION_STILL_LOCKED"
LOG_MARKER = "Step 151-F project log update: Step 150~151-E H200 handoff guard chain"


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")


def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read text: {path}")


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def validate(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    require(manifest.get("step") == "151-F", "step must be 151-F", errors)
    require(manifest.get("log_update_status") in ALLOWED_STATUSES, f"log_update_status must be one of {sorted(ALLOWED_STATUSES)}", errors)
    require(manifest.get("decision") == EXPECTED_DECISION, f"decision must be {EXPECTED_DECISION}", errors)
    require(int(manifest.get("hard_failures", -1)) == 0, "hard_failures must be 0", errors)
    require(manifest.get("project_log_contains_marker") is True, "project_log_contains_marker must be true", errors)
    require(manifest.get("documentation_only") is True, "documentation_only must be true", errors)
    require(manifest.get("actual_release_requires_separate_manifest") is True, "actual_release_requires_separate_manifest must be true", errors)

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
        "operator_approval_granted",
    ]

    for field in false_fields:
        require(manifest.get(field) is False, f"{field} must be false", errors)

    project_log = Path(str(manifest.get("project_log", "")))
    require(project_log.exists(), f"project_log does not exist: {project_log}", errors)
    if project_log.exists():
        text = read_text_any(project_log)
        require(LOG_MARKER in text, "project_log must contain Step 151-F marker", errors)
        require("actual_execution_allowed = false" in text, "project_log must record actual_execution_allowed=false", errors)
        require("--expect-h200 --min-gpu-count 1" in text, "project_log must record H200 Step 149 rerun args", errors)

    checks = manifest.get("checks", [])
    require(isinstance(checks, list) and len(checks) >= 12, "checks must contain at least 12 entries", errors)
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
        print("[FAIL] Step 151-F manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 151-F manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
