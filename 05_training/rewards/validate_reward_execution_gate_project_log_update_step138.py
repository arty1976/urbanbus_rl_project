from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_execution_gate_project_log_update_step138_v1"
SECTION_START = "<!-- STEP138_REWARD_EXECUTION_GATE_UPDATE_START -->"
SECTION_END = "<!-- STEP138_REWARD_EXECUTION_GATE_UPDATE_END -->"

REQUIRED_LOG_TOKENS = [
    "Step 138 - Reward execution gate project log update",
    "Step 131: actual reward ablation execution environment preflight",
    "Step 132: reward ablation command dry-run executor",
    "Step 133: actual reward ablation runner guard",
    "Step 134: command-to-guarded-runner integration",
    "Step 135: actual execution readiness lock",
    "Step 136: actual execution release request package",
    "Step 137: actual runner implementation review gate",
    "actual_execution_allowed = false",
    "actual_results = false",
    "winner_selected = false",
    "train_with_this_reward_allowed = false",
    "paper_level_claim_allowed = false",
    "causal_performance_claim_allowed = false",
]


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


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"failed_to_read_text: {path}")


def validate_payload(payload: Dict[str, Any], project_log_text: str, require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")

    if payload.get("step") != 138:
        errors.append("step_must_be_138")

    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")

    if require_pass and not bool(payload.get("project_log_updated", False)):
        errors.append("project_log_updated_false")

    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")

    if require_pass:
        if SECTION_START not in project_log_text or SECTION_END not in project_log_text:
            errors.append("step138_project_log_section_markers_missing")
        for token in REQUIRED_LOG_TOKENS:
            if token not in project_log_text:
                errors.append(f"project_log_required_token_missing: {token}")

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-log", default="project_log.md")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    project_log_text = read_text(Path(args.project_log))
    result = validate_payload(payload, project_log_text, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 138 reward execution gate project log update validation PASS")


if __name__ == "__main__":
    main()
