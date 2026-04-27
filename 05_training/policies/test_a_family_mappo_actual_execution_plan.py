from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_cmd(cmd, expect_success: bool):
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(" ".join(str(x) for x in cmd))
    print(proc.stdout)

    if expect_success and proc.returncode != 0:
        raise RuntimeError(f"expected success but failed: returncode={proc.returncode}")

    if (not expect_success) and proc.returncode == 0:
        raise RuntimeError("expected failure but command succeeded")

    return proc


def make_preflight_report(
    path: Path,
    *,
    checkpoint_path: str,
    status: str = "PASS",
    actual_ready: bool = True,
    actual_validator_passed: bool = True,
    blockers=None,
):
    payload = {
        "artifact_version": "h200_actual_checkpoint_preflight_v1",
        "status": status,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": True,
        "actual_checkpoint_ready": bool(actual_ready),
        "actual_policy_claim_ready_candidate": bool(actual_ready),
        "causal_policy_claim_ready_candidate": False,
        "historical_or_replay_adapter": True,
        "actual_validator": {
            "passed": bool(actual_validator_passed),
            "returncode": 0 if actual_validator_passed else 1,
            "report_path": str(path.parent / "actual_checkpoint_validation_report.json"),
        },
        "blockers": blockers or [],
        "warnings": [
            "simulator adapter is historical/replay; actual checkpoint may run, but causal performance claim is not allowed"
        ],
    }
    dump_json(path, payload)


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    planner = training_dir / "policies" / "a_family_mappo_actual_execution_plan.py"
    plan_doc = training_dir / "policies" / "a_family_mappo_actual_execution_plan.md"

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "a_family_mappo_actual_execution_plan_selftest"
    reports_dir = out_dir / "preflight_reports"
    plans_dir = out_dir / "plans"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    if not plan_doc.exists():
        raise RuntimeError(f"plan document missing: {plan_doc}")

    doc_text = plan_doc.read_text(encoding="utf-8-sig")
    for phrase in [
        "policy_source_mode = mappo_actual",
        "checkpoint_validation_mode = actual",
        "actual_checkpoint_ready = true",
        "causal performance claim is not allowed",
    ]:
        if phrase not in doc_text:
            raise RuntimeError(f"plan document missing phrase: {phrase}")

    checkpoint_path = str(out_dir / "fake_h200_best.pt")
    valid_preflight = reports_dir / "valid_actual_preflight_report.json"
    make_preflight_report(
        valid_preflight,
        checkpoint_path=checkpoint_path,
        status="PASS",
        actual_ready=True,
        actual_validator_passed=True,
    )

    plan_path = plans_dir / "valid_actual_execution_plan.json"
    run_cmd(
        [
            sys.executable,
            str(planner),
            "--checkpoint-path",
            checkpoint_path,
            "--preflight-report",
            str(valid_preflight),
            "--output-root",
            str(plans_dir / "valid_plan_output"),
            "--conditions",
            "A,A90,A80,A70",
            "--seeds",
            "1,2,3",
            "--device",
            "cpu",
            "--json-output",
            str(plan_path),
        ],
        expect_success=True,
    )

    plan = read_json(plan_path)
    if plan.get("status") != "READY_TO_EXECUTE":
        raise RuntimeError("valid plan must be READY_TO_EXECUTE")
    if int(plan.get("run_count", 0)) != 12:
        raise RuntimeError("A/A90/A80/A70 x seeds 1,2,3 must produce 12 runs")
    if plan.get("policy_source_mode") != "mappo_actual":
        raise RuntimeError("policy_source_mode must be mappo_actual")
    if plan.get("checkpoint_validation_mode") != "actual":
        raise RuntimeError("checkpoint_validation_mode must be actual")
    if plan.get("actual_policy_claim_candidate") is not True:
        raise RuntimeError("actual_policy_claim_candidate must be true")
    if plan.get("causal_claim_allowed_by_adapter") is not False:
        raise RuntimeError("historical adapter must not allow causal claim")
    if not plan.get("warnings"):
        raise RuntimeError("historical adapter warning must be present")

    first_cmd = plan["run_matrix"][0]["rollout_command"]
    joined = " ".join(first_cmd)
    for token in [
        "--policy-source-mode mappo_actual",
        "--checkpoint-validation-mode actual",
        "--checkpoint-path",
    ]:
        if token not in joined:
            raise RuntimeError(f"rollout command missing token: {token}")

    blocked_preflight = reports_dir / "blocked_preflight_report.json"
    make_preflight_report(
        blocked_preflight,
        checkpoint_path=checkpoint_path,
        status="BLOCKED",
        actual_ready=False,
        actual_validator_passed=False,
        blockers=["trained_model mismatch: expected True, got False"],
    )

    run_cmd(
        [
            sys.executable,
            str(planner),
            "--checkpoint-path",
            checkpoint_path,
            "--preflight-report",
            str(blocked_preflight),
            "--output-root",
            str(plans_dir / "expected_fail_blocked_preflight"),
            "--conditions",
            "A,A90",
            "--seeds",
            "1",
        ],
        expect_success=False,
    )

    mismatch_preflight = reports_dir / "mismatch_preflight_report.json"
    make_preflight_report(
        mismatch_preflight,
        checkpoint_path=str(out_dir / "other_best.pt"),
        status="PASS",
        actual_ready=True,
        actual_validator_passed=True,
    )

    run_cmd(
        [
            sys.executable,
            str(planner),
            "--checkpoint-path",
            checkpoint_path,
            "--preflight-report",
            str(mismatch_preflight),
            "--output-root",
            str(plans_dir / "expected_fail_checkpoint_mismatch"),
            "--conditions",
            "A",
            "--seeds",
            "1",
        ],
        expect_success=False,
    )

    run_cmd(
        [
            sys.executable,
            str(planner),
            "--checkpoint-path",
            checkpoint_path,
            "--preflight-report",
            str(valid_preflight),
            "--output-root",
            str(plans_dir / "expected_fail_bad_condition"),
            "--conditions",
            "A,B2",
            "--seeds",
            "1",
        ],
        expect_success=False,
    )

    run_cmd(
        [
            sys.executable,
            str(planner),
            "--checkpoint-path",
            checkpoint_path,
            "--preflight-report",
            str(valid_preflight),
            "--output-root",
            str(plans_dir / "expected_fail_historical_causal"),
            "--conditions",
            "A",
            "--seeds",
            "1",
            "--require-causal-claim",
        ],
        expect_success=False,
    )

    final_report = {
        "status": "PASS",
        "step": 59,
        "planner": str(planner),
        "plan_doc": str(plan_doc),
        "valid_plan": str(plan_path),
        "valid_run_count": int(plan.get("run_count")),
        "blocked_cases": [
            "blocked preflight report",
            "checkpoint path mismatch",
            "non A-family condition B2",
            "require causal claim with historical adapter",
        ],
        "note": (
            "Step 59 validates guarded A-family mappo_actual execution planning. "
            "It does not execute the actual matrix and does not claim performance."
        ),
    }

    out_report = out_dir / "step59_a_family_mappo_actual_execution_plan_selftest_report.json"
    dump_json(out_report, final_report)

    print("[OK] Step 59 A-family mappo_actual execution plan self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
