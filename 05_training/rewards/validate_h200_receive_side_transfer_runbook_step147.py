from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ARTIFACT_VERSION = "h200_receive_side_transfer_runbook_step147_v1"
CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
FORBIDDEN_TRUE_KEYS = [
    "actual_execution_allowed", "actual_execution_released", "actual_executed",
    "actual_results", "reward_result_written", "winner_selected",
    "trainable_reward_promoted", "train_with_this_reward_allowed",
    "actual_training_allowed", "final_reward_design_claim_allowed",
    "best_reward_claim_allowed", "paper_level_claim_allowed",
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
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if k in FORBIDDEN_TRUE_KEYS and bool(v):
                errors.append(f"forbidden_true: {path}")
            if isinstance(v, (dict, list)):
                errors.extend(recursive_forbidden_true(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                errors.extend(recursive_forbidden_true(v, f"{prefix}[{i}]"))
    return errors


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 147:
        errors.append("step_must_be_147")
    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")
    if payload.get("receive_plan_status") not in {
        "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_READY_ACTUAL_STILL_LOCKED",
        "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_BLOCKED",
    }:
        errors.append("invalid_receive_plan_status")
    if require_pass and payload.get("receive_plan_status") != "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_READY_ACTUAL_STILL_LOCKED":
        errors.append("receive_plan_status_not_ready_locked")
    if int(payload.get("planned_run_count", -1)) != 72:
        errors.append("planned_run_count_must_be_72")
    if payload.get("conditions") != CONDITIONS:
        errors.append("conditions_mismatch")
    if payload.get("reward_ids") != REWARD_IDS:
        errors.append("reward_ids_mismatch")
    if payload.get("seeds") != SEEDS:
        errors.append("seeds_mismatch")
    if int(payload.get("transfer_file_count", -1)) != 10:
        errors.append("transfer_file_count_must_be_10")
    if not str(payload.get("pinned_git_commit", "")):
        errors.append("pinned_git_commit_missing")
    plan = payload.get("receive_command_plan", {})
    for key in ["local_before_transfer", "h200_git_receive", "h200_metadata_receive_note", "h200_integrity_verify", "blocked_execution_reminder"]:
        if key not in plan or not isinstance(plan.get(key), list) or not plan.get(key):
            errors.append(f"command_plan_missing_or_empty: {key}")
    metadata = payload.get("verification_metadata_files", [])
    if not isinstance(metadata, list) or len(metadata) != 2:
        errors.append("verification_metadata_files_must_have_2_entries")
    errors.extend(recursive_forbidden_true(payload))
    return {"validation_status": "PASS" if not errors else "FAIL", "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--allow-blocked", action="store_true")
    args = ap.parse_args()
    payload = load_json(Path(args.manifest))
    result = validate_payload(payload, require_pass=not args.allow_blocked)
    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for e in result["errors"]:
            print(f"[FAIL] {e}")
        raise SystemExit(2)
    print("[OK] Step 147 H200 receive-side transfer runbook validation PASS")


if __name__ == "__main__":
    main()
