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


def run_validator(project_root: Path, validator: Path, protocol: Path, output_root: Path):
    cmd = [
        sys.executable,
        str(validator),
        "--protocol",
        str(protocol),
        "--output-root",
        str(output_root),
    ]
    return subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "05_training" / "rewards" / "validate_reward_candidate_protocol_step112.py"
    protocol = project_root / "05_training" / "rewards" / "reward_candidate_protocol_step112.json"
    output_root = project_root / "artifacts" / "rewards" / "reward_candidate_protocol_step112_selftest"

    result = run_validator(project_root, validator, protocol, output_root)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 112 valid protocol unexpectedly failed")

    report_path = output_root / "reward_candidate_protocol_step112_validation_report.json"
    report = load_json(report_path)

    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_CANDIDATE_PROTOCOL_DRAFT_NOT_TRAINABLE",
        "next_status": "READY_FOR_STEP113_NORMALIZATION_BASELINE_LOCK",
        "candidate_count": 5,
        "train_with_candidate_reward_allowed": False,
        "candidate_selected": False,
        "reward_weights_locked": False,
        "normalization_locked": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test 1: training allowed must fail.
    bad = load_json(protocol)
    bad["claim_guards"]["train_with_candidate_reward_allowed"] = True
    bad_path = output_root / "bad_train_allowed_protocol.json"
    dump_json(bad_path, bad)
    bad_result = run_validator(project_root, validator, bad_path, output_root / "bad_train_allowed_case")
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad train-allowed protocol unexpectedly passed")
    bad_report = load_json(output_root / "bad_train_allowed_case" / "reward_candidate_protocol_step112_validation_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "claim_guard_train_with_candidate_reward_allowed" not in failed_ids:
        raise SystemExit("[FAIL] train_with_candidate_reward_allowed guard did not fail")

    # Negative test 2: fleet bonus larger than energy penalty must fail.
    bad2 = load_json(protocol)
    bad2["candidate_matrix"][0]["weights"]["w_fleet_reduction"] = 2.0
    bad2_path = output_root / "bad_fleet_weight_protocol.json"
    dump_json(bad2_path, bad2)
    bad2_result = run_validator(project_root, validator, bad2_path, output_root / "bad_fleet_case")
    if bad2_result.returncode == 0:
        print(bad2_result.stdout)
        raise SystemExit("[FAIL] bad fleet-weight protocol unexpectedly passed")
    bad2_report = load_json(output_root / "bad_fleet_case" / "reward_candidate_protocol_step112_validation_report.json")
    failed_ids2 = {c["check_id"] for c in bad2_report.get("checks", []) if not c.get("passed", False)}
    if "candidate_R0_BALANCED_STEP111_fleet_le_energy" not in failed_ids2:
        raise SystemExit("[FAIL] fleet weight guard did not fail")

    # Negative test 3: candidate count too small must fail.
    bad3 = load_json(protocol)
    bad3["candidate_matrix"] = bad3["candidate_matrix"][:3]
    bad3_path = output_root / "bad_candidate_count_protocol.json"
    dump_json(bad3_path, bad3)
    bad3_result = run_validator(project_root, validator, bad3_path, output_root / "bad_count_case")
    if bad3_result.returncode == 0:
        print(bad3_result.stdout)
        raise SystemExit("[FAIL] bad candidate-count protocol unexpectedly passed")
    bad3_report = load_json(output_root / "bad_count_case" / "reward_candidate_protocol_step112_validation_report.json")
    failed_ids3 = {c["check_id"] for c in bad3_report.get("checks", []) if not c.get("passed", False)}
    if "candidate_count_at_least_5" not in failed_ids3:
        raise SystemExit("[FAIL] candidate count guard did not fail")

    print("[OK] Step 112 reward candidate protocol self-test PASS")
    print("[DONE] Step 112 reward candidate protocol complete.")


if __name__ == "__main__":
    main()
