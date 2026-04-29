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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "05_training" / "rewards" / "validate_reward_normalization_baseline_lock_step113.py"
    spec = project_root / "05_training" / "rewards" / "reward_normalization_baseline_lock_step113.json"
    output_root = project_root / "artifacts" / "rewards" / "reward_normalization_baseline_lock_step113_selftest"

    result = run_validator(project_root, validator, spec, output_root)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 113 valid spec unexpectedly failed")

    report_path = output_root / "reward_normalization_baseline_lock_step113_validation_report.json"
    report = load_json(report_path)

    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_BASELINE_REFERENCE_LOCKED_NOT_TRAINABLE",
        "next_status": "READY_FOR_STEP114_HARD_CONSTRAINT_REVIEW",
        "training_reward_normalization_baseline": "B1_noop",
        "train_with_this_reward_allowed": False,
        "numeric_baseline_values_locked": False,
        "failure_count": 0,
    }

    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test 1: training baseline must be B1_noop.
    bad = load_json(spec)
    bad["baseline_reference_lock"]["training_reward_normalization_baseline"] = "B2_rulebased"
    bad_path = output_root / "bad_baseline_spec.json"
    dump_json(bad_path, bad)
    bad_result = run_validator(project_root, validator, bad_path, output_root / "bad_baseline_case")
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad baseline spec unexpectedly passed")
    bad_report = load_json(output_root / "bad_baseline_case" / "reward_normalization_baseline_lock_step113_validation_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "training_baseline" not in failed_ids:
        raise SystemExit("[FAIL] training_baseline guard did not fail")

    # Negative test 2: train flag must remain false.
    bad2 = load_json(spec)
    bad2["claim_guards"]["train_with_this_reward_allowed"] = True
    bad2_path = output_root / "bad_train_allowed_spec.json"
    dump_json(bad2_path, bad2)
    bad2_result = run_validator(project_root, validator, bad2_path, output_root / "bad_train_allowed_case")
    if bad2_result.returncode == 0:
        print(bad2_result.stdout)
        raise SystemExit("[FAIL] bad train allowed spec unexpectedly passed")
    bad2_report = load_json(output_root / "bad_train_allowed_case" / "reward_normalization_baseline_lock_step113_validation_report.json")
    failed_ids2 = {c["check_id"] for c in bad2_report.get("checks", []) if not c.get("passed", False)}
    if "claim_guard_train_with_this_reward_allowed" not in failed_ids2:
        raise SystemExit("[FAIL] train_with_this_reward_allowed guard did not fail")

    # Negative test 3: energy_proxy must remain forbidden as direct normalization term.
    bad3 = load_json(spec)
    bad3["normalization_terms"].append({
        "term": "energy_proxy",
        "direction": "lower_is_better",
        "reward_component": "raw_energy_penalty",
        "normalization_rule": "energy_proxy / baseline_energy_proxy",
        "baseline_source": "B1_noop",
        "observation_status": "forbidden_direct_test"
    })
    bad3_path = output_root / "bad_energy_proxy_direct_spec.json"
    dump_json(bad3_path, bad3)
    bad3_result = run_validator(project_root, validator, bad3_path, output_root / "bad_energy_proxy_direct_case")
    if bad3_result.returncode == 0:
        print(bad3_result.stdout)
        raise SystemExit("[FAIL] bad direct energy_proxy spec unexpectedly passed")
    bad3_report = load_json(output_root / "bad_energy_proxy_direct_case" / "reward_normalization_baseline_lock_step113_validation_report.json")
    failed_ids3 = {c["check_id"] for c in bad3_report.get("checks", []) if not c.get("passed", False)}
    if "forbidden_not_in_normalization_terms_energy_proxy" not in failed_ids3:
        raise SystemExit("[FAIL] direct energy_proxy guard did not fail")

    print("[OK] Step 113 reward normalization baseline lock self-test PASS")
    print("[DONE] Step 113 reward normalization baseline lock complete.")


if __name__ == "__main__":
    main()
