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


def run_validator(project_root: Path, validator: Path, spec: Path, output_root: Path):
    cmd = [
        sys.executable,
        str(validator),
        "--spec",
        str(spec),
        "--output-root",
        str(output_root),
    ]
    return subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)


def failed_check_ids(report):
    return {c["check_id"] for c in report.get("checks", []) if not c.get("passed", False)}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "05_training" / "rewards" / "validate_hard_constraint_review_step114.py"
    spec = project_root / "05_training" / "rewards" / "hard_constraint_review_step114.json"
    output_root = project_root / "artifacts" / "rewards" / "hard_constraint_review_step114_selftest"

    result = run_validator(project_root, validator, spec, output_root)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 114 valid spec unexpectedly failed")

    report_path = output_root / "hard_constraint_review_step114_validation_report.json"
    if not report_path.exists():
        raise SystemExit("[FAIL] validation report missing")

    report = load_json(report_path)
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_HARD_CONSTRAINT_REVIEW_NOT_TRAINABLE",
        "next_status": "READY_FOR_STEP115_REWARD_ABLATION_MATRIX",
        "baseline": "B1_noop",
        "train_with_this_reward_allowed": False,
        "hard_constraints_finalized": False,
        "numeric_values_locked": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test 1: train flag must fail.
    bad = load_json(spec)
    bad["claim_guards"]["train_with_this_reward_allowed"] = True
    bad_path = output_root / "bad_train_allowed_step114.json"
    dump_json(bad_path, bad)
    bad_result = run_validator(project_root, validator, bad_path, output_root / "bad_train_allowed_case")
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad train_allowed spec unexpectedly passed")
    bad_report = load_json(output_root / "bad_train_allowed_case" / "hard_constraint_review_step114_validation_report.json")
    if "claim_guard_train_with_this_reward_allowed" not in failed_check_ids(bad_report):
        raise SystemExit("[FAIL] train flag guard did not fail")

    # Negative test 2: missing service constraint must fail.
    bad2 = load_json(spec)
    bad2["constraints"] = [c for c in bad2["constraints"] if c.get("constraint_id") != "service_rate_floor"]
    bad2_path = output_root / "bad_missing_service_constraint_step114.json"
    dump_json(bad2_path, bad2)
    bad2_result = run_validator(project_root, validator, bad2_path, output_root / "bad_missing_service_constraint_case")
    if bad2_result.returncode == 0:
        print(bad2_result.stdout)
        raise SystemExit("[FAIL] bad missing service constraint spec unexpectedly passed")
    bad2_report = load_json(output_root / "bad_missing_service_constraint_case" / "hard_constraint_review_step114_validation_report.json")
    if "constraint_exists_service_rate_floor" not in failed_check_ids(bad2_report):
        raise SystemExit("[FAIL] missing service constraint guard did not fail")

    # Negative test 3: treating ETA headway as actual headway must fail.
    bad3 = load_json(spec)
    bad3["observability_guards"]["eta_based_headway_is_actual_headway"] = True
    bad3_path = output_root / "bad_eta_actual_headway_step114.json"
    dump_json(bad3_path, bad3)
    bad3_result = run_validator(project_root, validator, bad3_path, output_root / "bad_eta_actual_headway_case")
    if bad3_result.returncode == 0:
        print(bad3_result.stdout)
        raise SystemExit("[FAIL] bad ETA actual headway spec unexpectedly passed")
    bad3_report = load_json(output_root / "bad_eta_actual_headway_case" / "hard_constraint_review_step114_validation_report.json")
    if "eta_not_actual_headway_guard" not in failed_check_ids(bad3_report):
        raise SystemExit("[FAIL] ETA actual headway guard did not fail")

    print("[OK] Step 114 hard constraint review self-test PASS")
    print("[DONE] Step 114 hard constraint review complete.")


if __name__ == "__main__":
    main()
