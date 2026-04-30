from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_STATUS = "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED"
EXPECTED_DECISION = "CHECKLIST_ONLY_H200_RERUN_NOT_EXECUTED"
EXPECTED_NEXT_GATE = "READY_FOR_REAL_H200_STEP149_EXPECT_H200_RERUN"
EXPECTED_H200_ARGS = "--expect-h200 --min-gpu-count 1"



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

    require(manifest.get("step") == "151-B", "step must be 151-B", errors)
    require(
        manifest.get("checklist_status") == EXPECTED_STATUS,
        f"checklist_status must be {EXPECTED_STATUS}",
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
        "h200_expected_preflight_rerun_executed",
        "h200_expected_preflight_passed",
    ]

    for field in false_fields:
        require(manifest.get(field) is False, f"{field} must be false", errors)

    require(manifest.get("h200_expected_preflight_required") is True, "h200_expected_preflight_required must be true", errors)
    require(
        manifest.get("h200_expected_preflight_required_args") == EXPECTED_H200_ARGS,
        "h200_expected_preflight_required_args mismatch",
        errors,
    )
    require(manifest.get("next_gate") == EXPECTED_NEXT_GATE, f"next_gate must be {EXPECTED_NEXT_GATE}", errors)

    command = str(manifest.get("h200_rerun_command", ""))
    required_command_parts = [
        "h200_environment_preflight_result_manifest_step149.py",
        "--project-root /workspace/urbanbus_rl_project",
        "--expect-h200",
        "--min-gpu-count 1",
        "h200_environment_preflight_result_manifest_step149_h200_actual",
    ]
    for part in required_command_parts:
        require(part in command, f"h200_rerun_command missing: {part}", errors)

    pass_conditions = manifest.get("required_h200_step149_pass_conditions", {})
    require(pass_conditions.get("audit_status") == "PASS", "required H200 audit_status must be PASS", errors)
    require(pass_conditions.get("expect_h200") is True, "required H200 expect_h200 must be true", errors)
    require(pass_conditions.get("cuda_available") is True, "required H200 cuda_available must be true", errors)
    require(pass_conditions.get("gpu_count_minimum") == 1, "required H200 gpu_count_minimum must be 1", errors)
    require(pass_conditions.get("hard_failures") == 0, "required H200 hard_failures must be 0", errors)
    require(pass_conditions.get("actual_execution_allowed") is False, "required H200 actual_execution_allowed must remain false", errors)
    require(pass_conditions.get("train_allowed") is False, "required H200 train_allowed must remain false", errors)

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
        print("[FAIL] Step 151-B manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 151-B manifest validation PASS")
    print(f"[OK] manifest: {path}")



if __name__ == "__main__":
    main()
