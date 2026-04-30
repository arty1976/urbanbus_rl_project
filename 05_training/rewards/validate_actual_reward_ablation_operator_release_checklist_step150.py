from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_STATUS = "READY_FOR_STEP151_EXPLICIT_OPERATOR_RELEASE_MANIFEST_DRAFT"
EXPECTED_DECISION = "CHECKLIST_ONLY_NOT_RELEASED"


def read_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def validate(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    require(payload.get("step") == 150, "step must be 150", errors)
    require(payload.get("checklist_status") == EXPECTED_STATUS, f"checklist_status must be {EXPECTED_STATUS}", errors)
    require(payload.get("decision") == EXPECTED_DECISION, f"decision must be {EXPECTED_DECISION}", errors)
    require(int(payload.get("hard_failures", -1)) == 0, "hard_failures must be 0", errors)

    false_keys = [
        "actual_execution_allowed",
        "actual_execution_released",
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
        "train_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "operator_release_manifest_created",
    ]
    for key in false_keys:
        require(payload.get(key) is False, f"{key} must be false", errors)

    require(payload.get("checklist_only") is True, "checklist_only must be true", errors)
    require(payload.get("h200_step149_actual_server_rerun_required") is True, "H200 Step 149 rerun must be required", errors)
    require(
        payload.get("h200_step149_actual_server_rerun_args") == "--expect-h200 --min-gpu-count 1",
        "H200 rerun args must be --expect-h200 --min-gpu-count 1",
        errors,
    )

    required_matrix = payload.get("required_matrix", {})
    require(required_matrix.get("expected_run_count") == 72, "expected_run_count must be 72", errors)
    require(required_matrix.get("conditions") == ["A", "A90", "A80", "A70"], "conditions mismatch", errors)
    require(required_matrix.get("reward_candidates") == ["R0", "R1", "R2", "R3", "R4", "R5"], "reward candidates mismatch", errors)
    require(required_matrix.get("seeds") == [1, 2, 3], "seeds mismatch", errors)

    required_baselines = set(payload.get("required_baselines", []))
    require(required_baselines == {"B0R", "B1", "B2"}, "required_baselines must be B0R/B1/B2", errors)

    checks = payload.get("checks", [])
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
    payload = read_json(path)
    errors = validate(payload)

    if errors:
        print("[FAIL] Step 150 checklist manifest validation failed")
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 150 checklist manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
