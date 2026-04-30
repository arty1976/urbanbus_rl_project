from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


FORBIDDEN_TRUE_GUARDS = [
    "actual_results",
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

    if payload.get("step") != 132:
        errors.append("step_must_be_132")

    if payload.get("artifact_version") != "reward_ablation_command_dry_run_executor_step132_v1":
        errors.append("artifact_version_mismatch")

    audit_status = payload.get("audit_status")
    if audit_status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and audit_status != "PASS":
        errors.append("audit_status_not_pass")

    if require_pass and payload.get("decision", {}).get("command_dry_run_ready") is not True:
        errors.append("command_dry_run_ready_false_on_pass")

    if payload.get("decision", {}).get("actual_execution_performed") is not False:
        errors.append("actual_execution_must_be_false")

    if payload.get("planned_command_count") != len(payload.get("planned_commands", [])):
        errors.append("planned_command_count_mismatch")

    if require_pass and int(payload.get("planned_command_count", 0)) <= 0:
        errors.append("planned_command_count_must_be_positive")

    for cmd in payload.get("planned_commands", []):
        if cmd.get("execution_status") != "NOT_EXECUTED_DRY_RUN_ONLY":
            errors.append(f"planned_command_not_dry_run_only: {cmd.get('run_id')}")
        if bool(cmd.get("actual_result_written", True)):
            errors.append(f"actual_result_written_true: {cmd.get('run_id')}")
        if bool(cmd.get("winner_selection_allowed", True)):
            errors.append(f"winner_selection_allowed_true: {cmd.get('run_id')}")
        if bool(cmd.get("training_allowed", True)):
            errors.append(f"training_allowed_true: {cmd.get('run_id')}")

    guards = payload.get("non_claim_guards", {})
    for key in FORBIDDEN_TRUE_GUARDS:
        if bool(guards.get(key, False)):
            errors.append(f"forbidden_true_non_claim_guard: {key}")

    return {"validation_status": "PASS" if not errors else "FAIL", "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for e in result["errors"]:
            print(f"[FAIL] {e}")
        raise SystemExit(2)

    print("[OK] Step 132 reward ablation command dry-run manifest validation PASS")


if __name__ == "__main__":
    main()
