from __future__ import annotations

import csv
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


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unique_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    materializer = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_schema_bridge_step124.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_actual_result_schema_bridge_step124.py"
    spec = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_schema_bridge_step124.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_schema_bridge_step124_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_schema_bridge_step124_validation_{suffix}"

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
        raise SystemExit("[FAIL] Step 124 materializer failed")

    manifest_path = output_root / "reward_ablation_actual_result_schema_bridge_manifest_step124.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 124 manifest missing")

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
        raise SystemExit("[FAIL] Step 124 validator failed")

    report = load_json(validation_root / "reward_ablation_actual_result_schema_bridge_step124_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_ACTUAL_RESULT_SCHEMA_BRIDGE_TEMPLATE_ONLY",
        "next_status": "READY_FOR_STEP125_REWARD_ABLATION_SELECTION_CRITERIA_GATE",
        "detail_rows": 72,
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test: setting one row to actual_result=true must fail.
    manifest = load_json(manifest_path)
    detail_path = Path(manifest["output_files"]["actual_detail_template_csv"])
    with open(detail_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    rows[0]["execution_status"] = "COMPLETED_ACTUAL"
    rows[0]["actual_result"] = "True"
    rows[0]["trained_model"] = "True"
    write_csv(detail_path, rows, fieldnames)

    bad_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_schema_bridge_step124_bad_validation_{suffix}"
    bad_val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(bad_root),
    ]
    bad_val = subprocess.run(bad_val_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_val.returncode == 0:
        print(bad_val.stdout)
        raise SystemExit("[FAIL] bad actual_result row unexpectedly passed")

    bad_report = load_json(bad_root / "reward_ablation_actual_result_schema_bridge_step124_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "no_actual_result_rows" not in failed_ids:
        raise SystemExit("[FAIL] actual_result guard did not fail")

    print("[OK] Step 124 reward ablation actual result schema bridge self-test PASS")
    print("[DONE] Step 124 reward ablation actual result schema bridge complete.")


if __name__ == "__main__":
    main()
