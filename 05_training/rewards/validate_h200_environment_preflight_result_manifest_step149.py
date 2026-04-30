from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FALSE_LOCKS = [
    "actual_execution_allowed",
    "actual_execution_released",
    "train_allowed",
    "actual_results",
    "winner_selected",
    "trainable_reward_promoted",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def validate_manifest(path: Path, allow_local: bool) -> None:
    payload = load_json(path)

    if payload.get("artifact_version") != "h200_environment_preflight_result_manifest_step149_v1":
        raise RuntimeError("artifact_version mismatch")

    if int(payload.get("step_id", -1)) != 149:
        raise RuntimeError("step_id must be 149")

    if payload.get("audit_status") != "PASS":
        raise RuntimeError(f"audit_status must be PASS: {payload.get('audit_status')}")

    status = payload.get("environment_status")
    valid_status = {
        "H200_ENVIRONMENT_PREFLIGHT_RECORDED_READY_ACTUAL_STILL_LOCKED",
        "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED",
    }
    if status not in valid_status:
        raise RuntimeError(f"invalid environment_status: {status}")

    if not allow_local and status != "H200_ENVIRONMENT_PREFLIGHT_RECORDED_READY_ACTUAL_STILL_LOCKED":
        raise RuntimeError("H200-ready status required but local status was recorded")

    if payload.get("hard_failures"):
        raise RuntimeError(f"hard_failures found: {payload.get('hard_failures')}")

    locks = payload.get("execution_locks", {})
    for key in REQUIRED_FALSE_LOCKS:
        if locks.get(key) is not False:
            raise RuntimeError(f"execution lock must remain false: {key}")

    if not payload.get("python", {}).get("version"):
        raise RuntimeError("python version missing")

    if "torch" not in payload:
        raise RuntimeError("torch section missing")

    if "disk" not in payload:
        raise RuntimeError("disk section missing")

    print("[OK] Step 149 manifest validation PASS")
    print(f"[OK] manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-local", action="store_true")
    args = parser.parse_args()
    validate_manifest(Path(args.manifest), allow_local=bool(args.allow_local))


if __name__ == "__main__":
    main()
