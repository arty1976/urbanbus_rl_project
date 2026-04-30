from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "h200_transfer_package_export_manifest_step145_v1"
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_REWARD_IDS = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = 72

FORBIDDEN_TRANSFER_PREFIXES = [
    "artifacts/",
    "1000005000/",
    "project_files/",
]
FORBIDDEN_TRANSFER_SUFFIXES = [
    ".latest.json",
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


def validate_transfer_entries(entries: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    if not entries:
        errors.append("transfer_entries_empty")

    seen: set[str] = set()
    for entry in entries:
        rel = str(entry.get("relative_path", "")).replace("\\", "/")
        if not rel:
            errors.append("transfer_entry_empty_relative_path")
            continue

        if rel in seen:
            errors.append(f"duplicate_transfer_entry: {rel}")
        seen.add(rel)

        for prefix in FORBIDDEN_TRANSFER_PREFIXES:
            if rel.startswith(prefix):
                errors.append(f"{rel}: forbidden_transfer_prefix_{prefix}")
        for suffix in FORBIDDEN_TRANSFER_SUFFIXES:
            if rel.endswith(suffix):
                errors.append(f"{rel}: forbidden_transfer_suffix_{suffix}")

        if not bool(entry.get("exists", False)):
            errors.append(f"{rel}: transfer_entry_missing")
        if int(entry.get("size_bytes", 0)) <= 0:
            errors.append(f"{rel}: transfer_entry_empty_file")
        sha = str(entry.get("sha256", ""))
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
            errors.append(f"{rel}: invalid_sha256")

    return errors


def validate_payload(payload: Dict[str, Any], require_pass: bool = True) -> Dict[str, Any]:
    errors: List[str] = []

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        errors.append("artifact_version_mismatch")
    if payload.get("step") != 145:
        errors.append("step_must_be_145")

    if payload.get("audit_status") not in ("PASS", "BLOCKED"):
        errors.append("invalid_audit_status")
    if require_pass and payload.get("audit_status") != "PASS":
        errors.append("audit_status_not_pass")

    if payload.get("export_status") not in {
        "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED",
        "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_BLOCKED",
    }:
        errors.append("invalid_export_status")
    if require_pass and payload.get("export_status") != "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED":
        errors.append("export_status_not_ready_locked")

    if int(payload.get("planned_run_count", -1)) != EXPECTED_RUN_COUNT:
        errors.append("planned_run_count_must_be_72")
    if payload.get("conditions") != EXPECTED_CONDITIONS:
        errors.append("conditions_mismatch")
    if payload.get("reward_ids") != EXPECTED_REWARD_IDS:
        errors.append("reward_ids_mismatch")
    if payload.get("seeds") != EXPECTED_SEEDS:
        errors.append("seeds_mismatch")

    entries = payload.get("transfer_entries", [])
    if int(payload.get("transfer_file_count", -1)) != len(entries):
        errors.append("transfer_file_count_mismatch")

    errors.extend(validate_transfer_entries(entries))
    errors.extend(recursive_forbidden_true(payload))

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
        for error in result["errors"]:
            print(f"[FAIL] {error}")
        raise SystemExit(2)

    print("[OK] Step 145 H200 transfer package export manifest validation PASS")


if __name__ == "__main__":
    main()
