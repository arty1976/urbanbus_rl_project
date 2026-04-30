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
    materializer = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_ingestion_preflight_step126.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_actual_result_ingestion_preflight_step126.py"
    spec = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_ingestion_preflight_step126.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_ingestion_preflight_step126_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_ingestion_preflight_step126_validation_{suffix}"

    run_cmd = [sys.executable, str(materializer), "--spec", str(spec), "--output-root", str(output_root)]
    result = subprocess.run(run_cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 126 materializer failed")

    manifest_path = output_root / "reward_ablation_actual_result_ingestion_preflight_manifest_step126.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 126 manifest missing")

    val_cmd = [sys.executable, str(validator), "--manifest", str(manifest_path), "--output-root", str(validation_root)]
    val = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr)
        raise SystemExit("[FAIL] Step 126 validator failed")

    report = load_json(validation_root / "reward_ablation_actual_result_ingestion_preflight_step126_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_ACTUAL_RESULT_INGESTION_PREFLIGHT_DEFINED_NOT_INGESTED",
        "next_status": "READY_FOR_STEP127_TRAINABLE_REWARD_PROMOTION_DECISION_PACKAGE",
        "expected_actual_result_rows": 72,
        "ingestion_allowed_now": False,
        "actual_results_ingested": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    bad_manifest = load_json(manifest_path)
    bad_manifest["winner_selected"] = True
    bad_path = output_root / "bad_winner_selected_preflight_manifest_step126.json"
    dump_json(bad_path, bad_manifest)

    bad_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_ingestion_preflight_step126_bad_validation_{suffix}"
    bad_cmd = [sys.executable, str(validator), "--manifest", str(bad_path), "--output-root", str(bad_root)]
    bad = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad.returncode == 0:
        print(bad.stdout)
        raise SystemExit("[FAIL] bad winner_selected manifest unexpectedly passed")

    bad_report = load_json(bad_root / "reward_ablation_actual_result_ingestion_preflight_step126_validation_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "manifest_guard_winner_selected" not in failed_ids:
        raise SystemExit("[FAIL] winner_selected guard did not fail")

    print("[OK] Step 126 reward ablation actual result ingestion preflight self-test PASS")
    print("[DONE] Step 126 reward ablation actual result ingestion preflight complete.")


if __name__ == "__main__":
    main()
