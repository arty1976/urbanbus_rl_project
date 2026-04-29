from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    script = Path(__file__).resolve().parent / "final_reward_design_review_gate_step110.py"
    project_root = Path(__file__).resolve().parents[2]

    output_root = (
        project_root
        / "artifacts"
        / "daegu_bis_api_audit"
        / "final_reward_design_review_gate_step110_selftest"
    )

    cmd = [
        sys.executable,
        str(script),
        "--output-root",
        str(output_root),
        "--write-default-input",
    ]

    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 110 default gate run failed")

    report_path = output_root / "final_reward_design_review_gate_step110_report.json"
    checks_csv = output_root / "final_reward_design_review_gate_step110_checks.csv"
    summary_md = output_root / "final_reward_design_review_gate_step110_summary.md"

    if not report_path.exists():
        raise SystemExit("[FAIL] report json missing")
    if not checks_csv.exists():
        raise SystemExit("[FAIL] checks csv missing")
    if not summary_md.exists():
        raise SystemExit("[FAIL] summary md missing")

    report = load_json(report_path)

    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_SCAFFOLD_REWARD_BLOCKED_FROM_TRAINING",
        "final_reward_status": "NOT_FINAL_SCAFFOLD_REWARD_ONLY",
        "train_with_this_reward_allowed": False,
        "final_reward_design_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "service_quality_first": True,
        "energy_fleet_secondary": True,
    }

    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    if int(report.get("failure_count", -1)) != 0:
        raise SystemExit("[FAIL] failure_count must be 0")

    if int(report.get("check_count", 0)) < 20:
        raise SystemExit("[FAIL] check_count too small")

    # Negative test: if training is enabled, the gate must fail.
    bad_input = report["review_input_snapshot"]
    bad_input["reward_status"]["train_with_this_reward_allowed"] = True

    bad_path = output_root / "bad_review_input_train_allowed_true.json"
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump(bad_input, f, ensure_ascii=False, indent=2)

    bad_output = output_root / "bad_case"
    bad_cmd = [
        sys.executable,
        str(script),
        "--input",
        str(bad_path),
        "--output-root",
        str(bad_output),
    ]

    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad input unexpectedly passed")

    bad_report = load_json(bad_output / "final_reward_design_review_gate_step110_report.json")
    if bad_report.get("audit_status") != "FAIL":
        raise SystemExit("[FAIL] bad report audit_status must be FAIL")

    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "guard_train_with_this_reward_allowed" not in failed_ids:
        raise SystemExit("[FAIL] train_allowed guard did not fail as expected")

    print("[OK] Step 110 final reward design review gate self-test PASS")
    print("[DONE] Step 110 final reward design review gate complete.")


if __name__ == "__main__":
    main()
