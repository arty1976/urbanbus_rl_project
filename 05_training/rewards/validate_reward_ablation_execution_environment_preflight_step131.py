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

    if payload.get("step") != 131:
        errors.append("step_must_be_131")

    if payload.get("artifact_version") != "reward_ablation_execution_environment_preflight_step131_v1":
        errors.append("artifact_version_mismatch")

    audit_status = payload.get("audit_status")
    if audit_status not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")

    if require_pass and audit_status != "PASS":
        errors.append("audit_status_not_pass")

    decision = payload.get("decision", {})
    if require_pass and decision.get("actual_reward_ablation_execution_allowed") is not True:
        errors.append("execution_ready_false_on_pass")

    guards = payload.get("non_claim_guards", {})
    for key in FORBIDDEN_TRUE_GUARDS:
        if bool(guards.get(key, False)):
            errors.append(f"forbidden_true_non_claim_guard: {key}")

    if not isinstance(payload.get("step_markers"), list):
        errors.append("step_markers_missing_or_invalid")

    if not isinstance(payload.get("blocking_reasons"), list):
        errors.append("blocking_reasons_missing_or_invalid")

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

    print("[OK] Step 131 reward ablation environment preflight manifest validation PASS")


if __name__ == "__main__":
    main()
