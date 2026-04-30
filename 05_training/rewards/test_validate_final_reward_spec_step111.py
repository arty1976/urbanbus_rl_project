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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "05_training" / "rewards" / "validate_final_reward_spec_step111.py"
    spec = project_root / "05_training" / "rewards" / "final_reward_spec_step111.json"

    output_root = project_root / "artifacts" / "rewards" / "final_reward_spec_step111_selftest"

    cmd = [
        sys.executable,
        str(validator),
        "--spec",
        str(spec),
        "--output-root",
        str(output_root),
    ]

    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 111 valid spec unexpectedly failed")

    report_path = output_root / "final_reward_spec_step111_validation_report.json"
    if not report_path.exists():
        raise SystemExit("[FAIL] validation report missing")

    report = load_json(report_path)

    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_FINAL_REWARD_SPEC_DRAFT_NOT_TRAINABLE",
        "next_status": "READY_FOR_STEP112_REWARD_CANDIDATE_PROTOCOL",
        "train_with_this_reward_allowed": False,
        "final_reward_design_claim_allowed": False,
        "reward_formula_finalized": False,
        "reward_weights_locked": False,
        "normalization_locked": False,
        "hard_constraints_locked": False,
        "failure_count": 0,
    }

    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    if not report["service_weight_abs_sum"] > report["secondary_weight_abs_sum"]:
        raise SystemExit("[FAIL] service weights must dominate secondary weights")

    # Negative test 1: train flag must not be allowed.
    bad = load_json(spec)
    bad["claim_guards"]["train_with_this_reward_allowed"] = True
    bad_path = output_root / "bad_train_allowed_spec.json"
    dump_json(bad_path, bad)

    bad_cmd = [
        sys.executable,
        str(validator),
        "--spec",
        str(bad_path),
        "--output-root",
        str(output_root / "bad_train_allowed_case"),
    ]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad train_allowed spec unexpectedly passed")

    bad_report = load_json(output_root / "bad_train_allowed_case" / "final_reward_spec_step111_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "claim_guard_train_with_this_reward_allowed" not in failed_ids:
        raise SystemExit("[FAIL] train_with_this_reward_allowed guard did not fail")

    # Negative test 2: energy_proxy must not become a direct reward term.
    bad2 = load_json(spec)
    bad2["reward_term_classification"]["direct_reward_candidate_terms"].append("energy_proxy")
    bad2_path = output_root / "bad_energy_direct_spec.json"
    dump_json(bad2_path, bad2)

    bad2_cmd = [
        sys.executable,
        str(validator),
        "--spec",
        str(bad2_path),
        "--output-root",
        str(output_root / "bad_energy_direct_case"),
    ]
    bad2_result = subprocess.run(bad2_cmd, cwd=project_root, text=True, capture_output=True)
    if bad2_result.returncode == 0:
        print(bad2_result.stdout)
        raise SystemExit("[FAIL] bad energy direct spec unexpectedly passed")

    bad2_report = load_json(output_root / "bad_energy_direct_case" / "final_reward_spec_step111_validation_report.json")
    failed_ids2 = {
        c["check_id"]
        for c in bad2_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "eval_only_not_direct_energy_proxy" not in failed_ids2:
        raise SystemExit("[FAIL] energy_proxy direct reward guard did not fail")

    # Negative test 3: negative weights must fail.
    bad3 = load_json(spec)
    bad3["draft_weight_candidates_not_locked"]["w_avg_wait"] = -2.0
    bad3_path = output_root / "bad_negative_weight_spec.json"
    dump_json(bad3_path, bad3)

    bad3_cmd = [
        sys.executable,
        str(validator),
        "--spec",
        str(bad3_path),
        "--output-root",
        str(output_root / "bad_negative_weight_case"),
    ]
    bad3_result = subprocess.run(bad3_cmd, cwd=project_root, text=True, capture_output=True)
    if bad3_result.returncode == 0:
        print(bad3_result.stdout)
        raise SystemExit("[FAIL] bad negative weight spec unexpectedly passed")

    bad3_report = load_json(output_root / "bad_negative_weight_case" / "final_reward_spec_step111_validation_report.json")
    failed_ids3 = {
        c["check_id"]
        for c in bad3_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "weight_positive_w_avg_wait" not in failed_ids3:
        raise SystemExit("[FAIL] negative weight guard did not fail")

    print("[OK] Step 111 final reward spec self-test PASS")
    print("[DONE] Step 111 final reward specification draft complete.")


if __name__ == "__main__":
    main()

