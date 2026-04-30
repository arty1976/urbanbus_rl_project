from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "a_family_72run_release_matrix_extension_draft_step143_v1"
CONDITIONS = ["A", "A90", "A80", "A70"]
REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

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


def validate_draft(draft: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if draft.get("release_matrix_version") != "a_family_72run_release_matrix_v1":
        errors.append("draft_release_matrix_version_mismatch")
    if draft.get("created_from_step") != 143:
        errors.append("draft_created_from_step_must_be_143")
    if draft.get("conditions") != CONDITIONS:
        errors.append(f"draft_conditions_mismatch: {draft.get('conditions')}")
    if draft.get("reward_ids") != REWARD_IDS:
        errors.append(f"draft_reward_ids_mismatch: {draft.get('reward_ids')}")
    if draft.get("seeds") != SEEDS:
        errors.append(f"draft_seeds_mismatch: {draft.get('seeds')}")
    if int(draft.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("draft_planned_run_count_must_be_72")

    matrix = draft.get("run_matrix", [])
    if len(matrix) != EXPECTED_RUN_COUNT:
        errors.append(f"draft_run_matrix_count_mismatch: {len(matrix)}")

    run_ids = [str(r.get("run_id")) for r in matrix]
    if len(set(run_ids)) != len(run_ids):
        errors.append("draft_run_id_duplicate")

    condition_counts = Counter(str(r.get("condition_id")) for r in matrix)
    reward_counts = Counter(str(r.get("reward_id")) for r in matrix)
    seed_counts = Counter(int(r.get("seed")) for r in matrix) if matrix else Counter()

    for condition in CONDITIONS:
        if condition_counts[condition] != 18:
            errors.append(f"condition_count_mismatch: {condition}={condition_counts[condition]}")
    for reward_id in REWARD_IDS:
        if reward_counts[reward_id] != 12:
            errors.append(f"reward_count_mismatch: {reward_id}={reward_counts[reward_id]}")
    for seed in SEEDS:
        if seed_counts[seed] != 24:
            errors.append(f"seed_count_mismatch: {seed}={seed_counts[seed]}")

    release_decision = draft.get("release_decision", {})
    if bool(release_decision.get("actual_execution_allowed", False)):
        errors.append("draft_actual_execution_allowed_true")
    if bool(release_decision.get("actual_execution_released", False)):
        errors.append("draft_actual_execution_released_true")
    if bool(release_decision.get("release_ready", False)):
        errors.append("draft_release_ready_true")

    non_claim_guards = draft.get("non_claim_guards", {})
    if not isinstance(non_claim_guards, dict) or not non_claim_guards:
        errors.append("draft_non_claim_guards_missing")
    else:
        for key, value in non_claim_guards.items():
            if bool(value):
                errors.append(f"draft_non_claim_guard_true: {key}")

    errors.extend([f"draft.{e}" for e in recursive_forbidden_true(draft)])
    return errors


def validate_payload(payload: Dict[str, Any], draft: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 143:
        errors.append("step_must_be_143")
    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")

    if payload.get("matrix_status") not in {
        "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_READY_ACTUAL_STILL_LOCKED",
        "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_BLOCKED",
    }:
        errors.append("invalid_matrix_status")

    if require_pass and payload.get("matrix_status") != "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_READY_ACTUAL_STILL_LOCKED":
        errors.append("matrix_status_not_ready_locked")

    if int(payload.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("payload_planned_run_count_must_be_72")
    if payload.get("conditions") != CONDITIONS:
        errors.append("payload_conditions_mismatch")
    if payload.get("reward_ids") != REWARD_IDS:
        errors.append("payload_reward_ids_mismatch")
    if payload.get("seeds") != SEEDS:
        errors.append("payload_seeds_mismatch")

    errors.extend(recursive_forbidden_true(payload))
    errors.extend(validate_draft(draft))

    return {
        "validation_status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    payload = load_json(Path(args.manifest))
    draft = load_json(Path(args.draft))
    result = validate_payload(payload, draft, require_pass=not args.allow_blocked)

    print(f"[OK] validation_status: {result['validation_status']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 143 A-family 72-run release matrix extension draft validation PASS")


if __name__ == "__main__":
    main()
