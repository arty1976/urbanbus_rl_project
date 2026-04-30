from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def unique_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    materializer = project_root / "05_training" / "rewards" / "trainable_reward_promotion_decision_package_step127.py"
    validator = project_root / "05_training" / "rewards" / "validate_trainable_reward_promotion_decision_package_step127.py"
    spec = project_root / "05_training" / "rewards" / "trainable_reward_promotion_decision_package_step127.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"trainable_reward_promotion_decision_package_step127_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"trainable_reward_promotion_decision_package_step127_validation_{suffix}"

    run_cmd = [
        sys.executable,
        str(materializer),
        "--spec",
        str(spec),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(run_cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 127 materializer failed")

    manifest_path = output_root / "trainable_reward_promotion_decision_package_manifest_step127.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 127 manifest missing")

    val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(validation_root),
    ]
    val = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr)
        raise SystemExit("[FAIL] Step 127 validator failed")

    report = load_json(validation_root / "trainable_reward_promotion_decision_package_step127_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_PROMOTION_DECISION_PACKAGE_NOT_PROMOTED",
        "next_status": "READY_FOR_STEP128_REWARD_PIPELINE_DOCUMENTATION_OR_ACTUAL_ABLATION_DATA_WAIT",
        "trainable_reward_promoted": False,
        "selected_candidate_id": None,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test: promoted=true must fail.
    bad_manifest = load_json(manifest_path)
    bad_manifest["trainable_reward_promoted"] = True
    bad_manifest["source_spec_snapshot"]["guard_flags"]["trainable_reward_promoted"] = True
    bad_manifest["source_spec_snapshot"]["decision"]["trainable_reward_promoted"] = True
    bad_path = output_root / "bad_promoted_manifest_step127.json"
    dump_json(bad_path, bad_manifest)

    bad_root = project_root / "artifacts" / "rewards" / f"trainable_reward_promotion_decision_package_step127_bad_validation_{suffix}"
    bad_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(bad_path),
        "--output-root",
        str(bad_root),
    ]
    bad = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad.returncode == 0:
        print(bad.stdout)
        raise SystemExit("[FAIL] bad promoted manifest unexpectedly passed")

    bad_report = load_json(bad_root / "trainable_reward_promotion_decision_package_step127_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "manifest_guard_trainable_reward_promoted" not in failed_ids:
        raise SystemExit("[FAIL] promoted guard did not fail")

    print("[OK] Step 127 trainable reward promotion decision package self-test PASS")
    print("[DONE] Step 127 trainable reward promotion decision package complete.")


if __name__ == "__main__":
    main()
