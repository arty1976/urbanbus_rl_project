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
    validator = project_root / "05_training" / "rewards" / "validate_trainable_reward_promotion_gate_step116.py"
    gate = project_root / "05_training" / "rewards" / "trainable_reward_promotion_gate_step116.json"
    output_root = project_root / "artifacts" / "rewards" / "trainable_reward_promotion_gate_step116_selftest"

    cmd = [
        sys.executable,
        str(validator),
        "--gate",
        str(gate),
        "--output-root",
        str(output_root),
        "--check-artifacts",
    ]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 116 valid promotion gate unexpectedly failed")

    report_path = output_root / "trainable_reward_promotion_gate_step116_validation_report.json"
    report = load_json(report_path)

    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_TRAINABLE_REWARD_PROMOTION_GATE_NOT_PROMOTED",
        "next_status": "READY_FOR_STEP117_REWARD_ABLATION_RESULT_SCHEMA",
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "selected_candidate_claim_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test 1: promotion must not be allowed without actual ablation results.
    bad = load_json(gate)
    bad["trainable_reward_promoted"] = True
    bad["train_with_this_reward_allowed"] = True
    bad["selected_candidate_id"] = "R1"
    bad_path = output_root / "bad_promoted_gate.json"
    dump_json(bad_path, bad)

    bad_cmd = [
        sys.executable,
        str(validator),
        "--gate",
        str(bad_path),
        "--output-root",
        str(output_root / "bad_promoted_case"),
    ]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad promoted gate unexpectedly passed")

    bad_report = load_json(output_root / "bad_promoted_case" / "trainable_reward_promotion_gate_step116_validation_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "trainable_reward_promoted_false" not in failed_ids:
        raise SystemExit("[FAIL] promotion guard did not fail")
    if "train_allowed_false" not in failed_ids:
        raise SystemExit("[FAIL] train_allowed guard did not fail")

    # Negative test 2: actual_ablation_results_exist cannot be true in this gate.
    bad2 = load_json(gate)
    bad2["promotion_requirements"]["actual_ablation_results_exist"] = True
    bad2_path = output_root / "bad_actual_results_gate.json"
    dump_json(bad2_path, bad2)

    bad2_cmd = [
        sys.executable,
        str(validator),
        "--gate",
        str(bad2_path),
        "--output-root",
        str(output_root / "bad_actual_results_case"),
    ]
    bad2_result = subprocess.run(bad2_cmd, cwd=project_root, text=True, capture_output=True)
    if bad2_result.returncode == 0:
        print(bad2_result.stdout)
        raise SystemExit("[FAIL] bad actual-results gate unexpectedly passed")

    bad2_report = load_json(output_root / "bad_actual_results_case" / "trainable_reward_promotion_gate_step116_validation_report.json")
    failed_ids2 = {c["check_id"] for c in bad2_report.get("checks", []) if not c.get("passed", False)}
    if "not_ready_actual_ablation_results_exist" not in failed_ids2:
        raise SystemExit("[FAIL] actual_ablation_results_exist guard did not fail")

    print("[OK] Step 116 trainable reward promotion gate self-test PASS")
    print("[DONE] Step 116 trainable reward promotion gate complete.")


if __name__ == "__main__":
    main()
