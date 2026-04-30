from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "actual_reward_ablation_runner_guard_step133_v1"
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

    if payload.get("step") != 133:
        errors.append("step_must_be_133")

    status = payload.get("audit_status")
    if status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and status != "PASS":
        errors.append("audit_status_not_pass")

    candidate = payload.get("candidate", {})
    if not isinstance(candidate, dict):
        errors.append("candidate_missing_or_invalid")
    else:
        if candidate.get("reward_id") not in ("R0", "R1", "R2", "R3", "R4", "R5"):
            errors.append("invalid_reward_id")
        if candidate.get("condition") != "A":
            errors.append("condition_must_be_A")
        if int(candidate.get("seed", 0)) <= 0:
            errors.append("seed_must_be_positive")

    for key in FORBIDDEN_TRUE_KEYS:
        if bool(payload.get(key, False)):
            errors.append(f"forbidden_true_key: {key}")

    guards = payload.get("non_claim_guards", {})
    if not isinstance(guards, dict):
        errors.append("non_claim_guards_missing_or_invalid")
    else:
        for key, value in guards.items():
            if bool(value):
                errors.append(f"forbidden_true_non_claim_guard: {key}")

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
        for err in result["errors"]:
            print(f"[FAIL] {err}")
        raise SystemExit(2)

    print("[OK] Step 133 actual reward ablation runner guard manifest validation PASS")


if __name__ == "__main__":
    main()
