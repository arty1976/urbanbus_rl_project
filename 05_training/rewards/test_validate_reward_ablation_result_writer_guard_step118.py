from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
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


def read_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows):
    if not rows:
        raise RuntimeError("cannot write empty csv rows")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer_obj = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_obj.writeheader()
        writer_obj.writerows(rows)


def run_checked(cmd, cwd: Path, label: str, expect_success: bool = True):
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if expect_success and result.returncode != 0:
        raise SystemExit(f"[FAIL] {label} unexpectedly failed")
    if (not expect_success) and result.returncode == 0:
        raise SystemExit(f"[FAIL] {label} unexpectedly passed")
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    writer = project_root / "05_training" / "rewards" / "reward_ablation_result_writer_guard_step118.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_result_writer_guard_step118.py"

    run_id = f"pid{os.getpid()}_{int(time.time() * 1000)}"
    base_root = project_root / "artifacts" / "rewards"
    output_root = base_root / f"reward_ablation_result_writer_guard_step118_selftest_{run_id}"
    report_root = base_root / f"reward_ablation_result_writer_guard_step118_selftest_validation_{run_id}"
    bad_root = base_root / f"reward_ablation_result_writer_guard_step118_bad_case_{run_id}"
    bad_report_root = base_root / f"reward_ablation_result_writer_guard_step118_bad_case_validation_{run_id}"

    writer_cmd = [
        sys.executable,
        str(writer),
        "--output-root",
        str(output_root),
    ]
    run_checked(writer_cmd, project_root, "Step 118 writer")

    validator_cmd = [
        sys.executable,
        str(validator),
        "--output-root",
        str(output_root),
        "--report-root",
        str(report_root),
    ]
    run_checked(validator_cmd, project_root, "Step 118 validator")

    report_path = report_root / "reward_ablation_result_writer_guard_step118_validation_report.json"
    report = load_json(report_path)

    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_RESULT_WRITER_GUARD_TEMPLATE_ONLY",
        "next_status": "READY_FOR_STEP119_REWARD_ABLATION_RUNNER_DRY_RUN_PLAN",
        "candidate_count": 6,
        "detail_row_count": 72,
        "summary_row_count": 6,
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "best_claim_allowed": False,
        "trainable_reward_promoted": False,
        "failure_count": 0,
    }

    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    manifest = load_json(output_root / "reward_ablation_result_writer_guard_manifest.json")
    detail_path = output_root / "reward_ablation_results_by_candidate_seed_condition_template.csv"
    summary_path = output_root / "reward_ablation_results_by_candidate_template.csv"
    violations_path = output_root / "reward_ablation_hard_constraint_violations_template.csv"
    selection_path = output_root / "reward_ablation_selection_summary_template.json"

    detail_rows = read_csv_rows(detail_path)
    summary_rows = read_csv_rows(summary_path)
    violations = read_csv_rows(violations_path)
    selection = load_json(selection_path)

    if manifest["actual_results"] is not False:
        raise SystemExit("[FAIL] manifest actual_results must be false")
    if len(detail_rows) != 72:
        raise SystemExit("[FAIL] detail template must have 72 rows")
    if len(summary_rows) != 6:
        raise SystemExit("[FAIL] summary template must have 6 rows")
    if len(violations) != 6:
        raise SystemExit("[FAIL] violation template must have 6 rows")
    if selection["winner_selected"] is not False:
        raise SystemExit("[FAIL] selection winner_selected must be false")
    if selection["train_with_this_reward_allowed"] is not False:
        raise SystemExit("[FAIL] selection train flag must be false")

    # Negative test without deleting/copying locked directories.
    # Create a fresh bad folder, copy only files needed by the validator,
    # then intentionally corrupt the detail template.
    bad_root.mkdir(parents=True, exist_ok=False)
    import shutil
    for src in [
        output_root / "reward_ablation_result_writer_guard_manifest.json",
        detail_path,
        summary_path,
        violations_path,
        selection_path,
    ]:
        shutil.copy2(src, bad_root / src.name)

    bad_detail = bad_root / "reward_ablation_results_by_candidate_seed_condition_template.csv"
    rows = read_csv_rows(bad_detail)
    rows[0]["train_with_this_reward_allowed"] = "True"
    write_csv_rows(bad_detail, rows)

    # Repoint copied manifest to bad_root so the validator reads the corrupted file.
    bad_manifest_path = bad_root / "reward_ablation_result_writer_guard_manifest.json"
    bad_manifest = load_json(bad_manifest_path)
    bad_manifest["output_root"] = str(bad_root)
    bad_manifest["output_files"]["detail_template_csv"] = str(bad_detail)
    bad_manifest["output_files"]["summary_template_csv"] = str(bad_root / "reward_ablation_results_by_candidate_template.csv")
    bad_manifest["output_files"]["hard_constraint_violations_template_csv"] = str(bad_root / "reward_ablation_hard_constraint_violations_template.csv")
    bad_manifest["output_files"]["selection_summary_template_json"] = str(bad_root / "reward_ablation_selection_summary_template.json")
    dump_json(bad_manifest_path, bad_manifest)

    bad_cmd = [
        sys.executable,
        str(validator),
        "--output-root",
        str(bad_root),
        "--report-root",
        str(bad_report_root),
    ]
    run_checked(bad_cmd, project_root, "corrupted Step 118 template", expect_success=False)

    bad_report = load_json(bad_report_root / "reward_ablation_result_writer_guard_step118_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "detail_train_allowed_all_false" not in failed_ids:
        raise SystemExit("[FAIL] train_allowed corruption was not detected")

    print("[OK] Step 118 reward ablation result writer guard self-test PASS")
    print("[DONE] Step 118 reward ablation result writer guard complete.")


if __name__ == "__main__":
    main()
