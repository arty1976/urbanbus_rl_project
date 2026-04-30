from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_ablation_command_runner_integration_step134_v1"
FORBIDDEN_TRUE_KEYS = [
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

    if payload.get("step") != 134:
        errors.append("step_must_be_134")

    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")

    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        errors.append("summary_missing_or_invalid")
    else:
        if int(summary.get("planned_command_count", 0)) != 18:
            errors.append("planned_command_count_must_be_18")
        if require_pass and int(summary.get("step133_invocation_count", 0)) != 18:
            errors.append("step133_invocation_count_must_be_18")
        if require_pass and int(summary.get("step133_guard_pass_count", 0)) != 18:
            errors.append("step133_guard_pass_count_must_be_18")
        if require_pass and int(summary.get("step133_guard_fail_count", 0)) != 0:
            errors.append("step133_guard_fail_count_must_be_0")

    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")

    planned = payload.get("planned_commands", [])
    if not isinstance(planned, list):
        errors.append("planned_commands_missing_or_invalid")
    elif len(planned) != 18:
        errors.append("planned_commands_len_must_be_18")

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

    print("[OK] Step 134 command-runner integration manifest validation PASS")


if __name__ == "__main__":
    main()
