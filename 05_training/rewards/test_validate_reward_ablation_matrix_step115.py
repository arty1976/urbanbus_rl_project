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


def run_validator(project_root: Path, validator: Path, matrix: Path, output_root: Path):
    cmd = [
        sys.executable,
        str(validator),
        "--matrix",
        str(matrix),
        "--output-root",
        str(output_root),
    ]
    return subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)


def failed_check_ids(report_path: Path):
    report = load_json(report_path)
    return {
        c["check_id"]
        for c in report.get("checks", [])
        if not c.get("passed", False)
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_matrix_step115.py"
    matrix = project_root / "05_training" / "rewards" / "reward_ablation_matrix_step115.json"

    output_root = project_root / "artifacts" / "rewards" / "reward_ablation_matrix_step115_selftest"

    result = run_validator(project_root, validator, matrix, output_root)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 115 valid matrix unexpectedly failed")

    report_path = output_root / "reward_ablation_matrix_step115_validation_report.json"
    if not report_path.exists():
        raise SystemExit("[FAIL] validation report missing")

    report = load_json(report_path)

    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_MATRIX_DRAFT_NOT_TRAINABLE",
        "next_status": "READY_FOR_STEP116_TRAINABLE_REWARD_PROMOTION_GATE",
        "candidate_count": 6,
        "train_with_any_candidate_allowed": False,
        "best_reward_claim_allowed": False,
        "reward_weights_locked": False,
        "hard_constraints_finalized": False,
        "failure_count": 0,
    }

    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    spec = load_json(matrix)

    # Negative test 1: training flag must remain false.
    bad = load_json(matrix)
    bad["claim_guards"]["train_with_any_candidate_allowed"] = True
    bad_path = output_root / "bad_train_allowed_matrix.json"
    dump_json(bad_path, bad)
    bad_out = output_root / "bad_train_allowed_case"
    bad_result = run_validator(project_root, validator, bad_path, bad_out)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad train_allowed matrix unexpectedly passed")
    ids = failed_check_ids(bad_out / "reward_ablation_matrix_step115_validation_report.json")
    if "claim_guard_train_with_any_candidate_allowed" not in ids:
        raise SystemExit("[FAIL] train guard did not fail")

    # Negative test 2: energy/fleet cannot dominate service.
    bad2 = load_json(matrix)
    bad2["candidate_weight_sets"][1]["weights"]["w_energy_per_passenger"] = 20.0
    bad2_path = output_root / "bad_energy_dominates_matrix.json"
    dump_json(bad2_path, bad2)
    bad2_out = output_root / "bad_energy_dominates_case"
    bad2_result = run_validator(project_root, validator, bad2_path, bad2_out)
    if bad2_result.returncode == 0:
        print(bad2_result.stdout)
        raise SystemExit("[FAIL] bad energy-dominates matrix unexpectedly passed")
    ids2 = failed_check_ids(bad2_out / "reward_ablation_matrix_step115_validation_report.json")
    if "R1_BALANCED_DEFAULT_service_dominates_secondary" not in ids2:
        raise SystemExit("[FAIL] energy dominance guard did not fail")

    # Negative test 3: negative weights must fail.
    bad3 = load_json(matrix)
    bad3["candidate_weight_sets"][2]["weights"]["w_avg_wait"] = -1.0
    bad3_path = output_root / "bad_negative_weight_matrix.json"
    dump_json(bad3_path, bad3)
    bad3_out = output_root / "bad_negative_weight_case"
    bad3_result = run_validator(project_root, validator, bad3_path, bad3_out)
    if bad3_result.returncode == 0:
        print(bad3_result.stdout)
        raise SystemExit("[FAIL] bad negative-weight matrix unexpectedly passed")
    ids3 = failed_check_ids(bad3_out / "reward_ablation_matrix_step115_validation_report.json")
    if "R2_WAIT_HEAVY_weight_non_negative_w_avg_wait" not in ids3:
        raise SystemExit("[FAIL] negative weight guard did not fail")

    # Negative test 4: final selection must not be active.
    bad4 = load_json(matrix)
    bad4["selection_rules_not_active_yet"]["selection_status"] = "SELECTED_R1"
    bad4_path = output_root / "bad_selection_active_matrix.json"
    dump_json(bad4_path, bad4)
    bad4_out = output_root / "bad_selection_active_case"
    bad4_result = run_validator(project_root, validator, bad4_path, bad4_out)
    if bad4_result.returncode == 0:
        print(bad4_result.stdout)
        raise SystemExit("[FAIL] bad selection-active matrix unexpectedly passed")
    ids4 = failed_check_ids(bad4_out / "reward_ablation_matrix_step115_validation_report.json")
    if "selection_status" not in ids4:
        raise SystemExit("[FAIL] selection-status guard did not fail")

    print("[OK] Step 115 reward ablation matrix self-test PASS")
    print("[DONE] Step 115 reward ablation matrix complete.")


if __name__ == "__main__":
    main()
