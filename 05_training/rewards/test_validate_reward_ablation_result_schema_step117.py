from __future__ import annotations

import json
import subprocess
import sys
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


def run_cmd(cmd, cwd: Path):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_result_schema_step117.py"
    schema = project_root / "05_training" / "rewards" / "reward_ablation_result_schema_step117.json"
    output_root = project_root / "artifacts" / "rewards" / "reward_ablation_result_schema_step117_selftest"

    cmd = [
        sys.executable,
        str(validator),
        "--schema",
        str(schema),
        "--output-root",
        str(output_root),
    ]

    result = run_cmd(cmd, cwd=project_root)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 117 valid schema unexpectedly failed")

    report_path = output_root / "reward_ablation_result_schema_step117_validation_report.json"
    if not report_path.exists():
        raise SystemExit("[FAIL] validation report missing")

    report = load_json(report_path)
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_RESULT_SCHEMA_NOT_EVALUATED",
        "next_status": "READY_FOR_STEP118_REWARD_ABLATION_RESULT_WRITER_OR_RUNNER_GUARD",
        "candidate_count": 6,
        "table_count": 4,
        "actual_results_available": False,
        "reward_winner_selected": False,
        "train_with_this_reward_allowed": False,
        "best_reward_claim_allowed": False,
        "failure_count": 0,
    }

    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test 1: candidate set must include R0-R5 exactly.
    bad = load_json(schema)
    bad["candidate_ids_expected"] = ["R0", "R1", "R2", "R3", "R4"]
    bad_path = output_root / "bad_missing_candidate_schema.json"
    dump_json(bad_path, bad)

    bad_cmd = [
        sys.executable,
        str(validator),
        "--schema",
        str(bad_path),
        "--output-root",
        str(output_root / "bad_missing_candidate_case"),
    ]
    bad_result = run_cmd(bad_cmd, cwd=project_root)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad missing candidate schema unexpectedly passed")

    bad_report = load_json(output_root / "bad_missing_candidate_case" / "reward_ablation_result_schema_step117_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "candidate_ids_exact" not in failed_ids:
        raise SystemExit("[FAIL] candidate_ids_exact guard did not fail")

    # Negative test 2: actual results must not be marked available in Step 117.
    bad2 = load_json(schema)
    bad2["claim_guards"]["actual_ablation_results_available"] = True
    bad2_path = output_root / "bad_actual_results_available_schema.json"
    dump_json(bad2_path, bad2)

    bad2_cmd = [
        sys.executable,
        str(validator),
        "--schema",
        str(bad2_path),
        "--output-root",
        str(output_root / "bad_actual_results_available_case"),
    ]
    bad2_result = run_cmd(bad2_cmd, cwd=project_root)
    if bad2_result.returncode == 0:
        print(bad2_result.stdout)
        raise SystemExit("[FAIL] bad actual-results schema unexpectedly passed")

    bad2_report = load_json(output_root / "bad_actual_results_available_case" / "reward_ablation_result_schema_step117_validation_report.json")
    failed_ids2 = {
        c["check_id"]
        for c in bad2_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "guard_actual_ablation_results_available" not in failed_ids2:
        raise SystemExit("[FAIL] actual_ablation_results_available guard did not fail")

    # Negative test 3: main table must include all 12 KPI mean columns.
    bad3 = load_json(schema)
    for table in bad3["result_tables"]:
        if table["table_name"] == "reward_ablation_result_by_candidate_seed_condition":
            table["required_columns"].remove("passenger_wait_p95_seconds_mean")
            break
    bad3_path = output_root / "bad_missing_kpi_column_schema.json"
    dump_json(bad3_path, bad3)

    bad3_cmd = [
        sys.executable,
        str(validator),
        "--schema",
        str(bad3_path),
        "--output-root",
        str(output_root / "bad_missing_kpi_column_case"),
    ]
    bad3_result = run_cmd(bad3_cmd, cwd=project_root)
    if bad3_result.returncode == 0:
        print(bad3_result.stdout)
        raise SystemExit("[FAIL] bad missing KPI column schema unexpectedly passed")

    bad3_report = load_json(output_root / "bad_missing_kpi_column_case" / "reward_ablation_result_schema_step117_validation_report.json")
    failed_ids3 = {
        c["check_id"]
        for c in bad3_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "main_table_kpi_col_passenger_wait_p95_seconds_mean" not in failed_ids3:
        raise SystemExit("[FAIL] missing KPI column guard did not fail")

    print("[OK] Step 117 reward ablation result schema self-test PASS")
    print("[DONE] Step 117 reward ablation result schema complete.")


if __name__ == "__main__":
    main()
