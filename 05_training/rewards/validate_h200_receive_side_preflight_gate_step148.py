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


def validate_manifest(path: Path) -> None:
    payload = load_json(path)

    if payload.get("artifact_version") != "h200_receive_side_preflight_gate_step148_v1":
        raise RuntimeError("artifact_version mismatch")

    if int(payload.get("step_id", -1)) != 148:
        raise RuntimeError("step_id must be 148")

    if payload.get("gate_status") != "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW":
        raise RuntimeError(f"gate_status is not ready: {payload.get('gate_status')}")

    if payload.get("hard_failures"):
        raise RuntimeError(f"hard_failures found: {payload.get('hard_failures')}")

    locks = payload.get("execution_locks", {})
    for key in REQUIRED_FALSE_LOCKS:
        if locks.get(key) is not False:
            raise RuntimeError(f"execution lock must remain false: {key}")

    h200_root = str(payload.get("h200_project_root", ""))
    if "\\" in h200_root or not h200_root.startswith("/"):
        raise RuntimeError("h200_project_root must be an absolute Linux-style path")

    inventory = payload.get("file_inventory", [])
    if not inventory:
        raise RuntimeError("file_inventory is empty")

    missing = [x["relative_path"] for x in inventory if not x.get("exists")]
    if missing:
        raise RuntimeError(f"missing inventory files: {missing}")

    print("[OK] Step 148 manifest validation PASS")
    print(f"[OK] manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    validate_manifest(Path(args.manifest))


if __name__ == "__main__":
    main()
