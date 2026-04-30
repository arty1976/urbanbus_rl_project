from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "concrete_release_manifest_draft_validator_gate_step142_v1"

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


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 142:
        errors.append("step_must_be_142")

    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")

    valid_gate_statuses = {
        "A_RELEASE_DRAFT_VALIDATED_AND_B_GROUP_BASELINES_READY_ACTUAL_STILL_LOCKED",
        "RELEASE_DRAFT_VALIDATOR_GATE_BLOCKED",
    }
    if payload.get("gate_status") not in valid_gate_statuses:
        errors.append("invalid_gate_status")

    if require_pass and payload.get("gate_status") != "A_RELEASE_DRAFT_VALIDATED_AND_B_GROUP_BASELINES_READY_ACTUAL_STILL_LOCKED":
        errors.append("gate_status_not_ready_locked")

    if require_pass and not bool(payload.get("a_release_draft_validated", False)):
        errors.append("a_release_draft_not_validated")

    if require_pass and not bool(payload.get("b_group_baseline_ready", False)):
        errors.append("b_group_baseline_not_ready")

    errors.extend(recursive_forbidden_true(payload))

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

    print("[OK] Step 142 concrete release manifest draft validator gate validation PASS")


if __name__ == "__main__":
    main()
