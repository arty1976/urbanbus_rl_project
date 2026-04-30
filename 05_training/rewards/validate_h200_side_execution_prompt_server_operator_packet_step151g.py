from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_STATUS = "H200_SIDE_EXECUTION_PROMPT_PACKET_READY_STILL_LOCKED"
EXPECTED_DECISION = "SERVER_OPERATOR_PACKET_ONLY_NOT_RELEASED"


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
    require(manifest.get("step") == "151-G", "step must be 151-G", errors)
    require(manifest.get("packet_status") == EXPECTED_STATUS, f"packet_status must be {EXPECTED_STATUS}", errors)
    require(manifest.get("decision") == EXPECTED_DECISION, f"decision must be {EXPECTED_DECISION}", errors)
    require(int(manifest.get("hard_failures", -1)) == 0, "hard_failures must be 0", errors)
    for field in ["operator_approval_recorded", "operator_approval_granted", "actual_execution_allowed", "actual_execution_released", "train_allowed", "actual_results", "winner_selected", "trainable_reward_promoted", "paper_level_claim_allowed", "causal_performance_claim_allowed"]:
        require(manifest.get(field) is False, f"{field} must be false", errors)
    require(manifest.get("server_operator_packet_only") is True, "server_operator_packet_only must be true", errors)
    require(manifest.get("actual_release_requires_separate_manifest") is True, "actual_release_requires_separate_manifest must be true", errors)
    command = str(manifest.get("h200_command", ""))
    require("--expect-h200" in command, "h200_command must include --expect-h200", errors)
    require("--min-gpu-count 1" in command, "h200_command must include --min-gpu-count 1", errors)
    output_files = manifest.get("output_files", {})
    for key in ("manifest", "summary", "operator_prompt", "h200_command_script"):
        require(bool(output_files.get(key)), f"output_files.{key} must be recorded", errors)
    checks = manifest.get("checks", [])
    require(isinstance(checks, list) and len(checks) >= 5, "checks must contain at least 5 entries", errors)
    for check in checks:
        if check.get("severity") == "hard":
            require(check.get("passed") is True, f"hard check failed: {check.get('check_id')}", errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    path = Path(args.manifest)
    errors = validate(load_json(path))
    if errors:
        print("[FAIL] Step 151-G manifest validation failed")
        for e in errors:
            print(f"[FAIL] {e}")
        raise SystemExit(2)
    print("[OK] Step 151-G manifest validation PASS")
    print(f"[OK] manifest: {path}")


if __name__ == "__main__":
    main()
