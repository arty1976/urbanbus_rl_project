from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "05_training" / "policies" / "run_h200_actual_checkpoint_arrival_workflow.py"
DOC = ROOT / "05_training" / "policies" / "H200_actual_checkpoint_arrival_workflow.md"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_script(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def fake_preflight_script(status: str = "PASS") -> str:
    return f'''
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--report-path", required=True)
args = parser.parse_args()

payload = {{
    "artifact_version": "fake_preflight_v1",
    "preflight_status": "{status}",
    "status": "{status}",
    "mode": args.mode,
    "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
    "trained_model": True,
    "performance_claim_allowed": True,
    "qwen_train": False,
    "qwen_inference": False,
    "qwen_trigger_rate": 0.0,
    "reward_version": "mappo_reward_v1",
    "energy_proxy_model_version": "daegu_energy_proxy_v1",
    "k_dist_kwh_per_m": 0.0012,
    "k_acc_kwh_per_event": 0.1800,
    "k_idle_kwh_per_sec": 0.0080,
}}

Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
with open(args.report_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("[OK] fake preflight wrote report")
'''


def fake_preflight_mismatch_script() -> str:
    return '''
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--report-path", required=True)
args = parser.parse_args()

payload = {
    "artifact_version": "fake_preflight_v1",
    "preflight_status": "PASS",
    "status": "PASS",
    "checkpoint_path": str((Path(args.checkpoint_path).parent / "different_checkpoint.pt").resolve()),
}

Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
with open(args.report_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("[OK] fake preflight mismatch wrote report")
'''


def fake_plan_script(status: str = "READY_TO_EXECUTE") -> str:
    return f'''
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--preflight-report", required=True)
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--output-plan", required=True)
args = parser.parse_args()

payload = {{
    "artifact_version": "fake_plan_v1",
    "status": "{status}",
    "plan_status": "{status}",
    "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
    "runs": [],
}}

Path(args.output_plan).parent.mkdir(parents=True, exist_ok=True)
with open(args.output_plan, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("[OK] fake plan wrote plan")
'''


def fake_runner_script(status: str = "DRY_RUN_VALIDATED", executed: bool = False) -> str:
    executed_literal = "True" if executed else "False"
    return f'''
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--plan-json", required=True)
parser.add_argument("--report-path", required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

payload = {{
    "artifact_version": "fake_step60_runner_v1",
    "runner_status": "{status}",
    "status": "{status}",
    "dry_run": bool(args.dry_run),
    "executed": {executed_literal},
    "run_count": 12,
    "command_count": 48,
    "actual_claim_guard": {{"all_passed": True}},
    "causal_claim_guard": {{"all_passed": True}},
}}

Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
with open(args.report_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("[OK] fake runner wrote dry-run report")
'''


def run_workflow(
    tmp: Path,
    checkpoint: Path,
    preflight: Path,
    plan: Path,
    runner: Path,
    output_name: str,
    expect_ok: bool,
) -> subprocess.CompletedProcess:
    output_root = tmp / output_name

    cmd = [
        sys.executable,
        str(WORKFLOW),
        "--checkpoint-path",
        str(checkpoint),
        "--output-root",
        str(output_root),
        "--python-exe",
        sys.executable,
        "--preflight-script",
        str(preflight),
        "--plan-script",
        str(plan),
        "--runner-script",
        str(runner),
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)

    if expect_ok and completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError(f"expected PASS but got returncode={completed.returncode}")

    if (not expect_ok) and completed.returncode == 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("expected BLOCKED but workflow passed")

    return completed


def make_fake_environment(tmp: Path) -> dict:
    checkpoint = tmp / "best_mappo.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes for Step 62 self-test")

    preflight = tmp / "fake_preflight.py"
    plan = tmp / "fake_plan.py"
    runner = tmp / "fake_runner.py"

    write_script(preflight, fake_preflight_script("PASS"))
    write_script(plan, fake_plan_script("READY_TO_EXECUTE"))
    write_script(runner, fake_runner_script("DRY_RUN_VALIDATED", executed=False))

    return {
        "checkpoint": checkpoint,
        "preflight": preflight,
        "plan": plan,
        "runner": runner,
    }


def test_success_path(tmp: Path) -> None:
    env = make_fake_environment(tmp)

    completed = run_workflow(
        tmp=tmp,
        checkpoint=env["checkpoint"],
        preflight=env["preflight"],
        plan=env["plan"],
        runner=env["runner"],
        output_name="success",
        expect_ok=True,
    )

    assert "[OK] Step 62" in completed.stdout

    report = read_json(tmp / "success" / "h200_actual_checkpoint_arrival_workflow_report.json")
    assert report["arrival_workflow_status"] == "READY_FOR_MANUAL_EXECUTE"
    assert report["preflight_status"] == "PASS"
    assert report["plan_status"] == "READY_TO_EXECUTE"
    assert report["dry_run_status"] == "DRY_RUN_VALIDATED"
    assert report["dry_run"] is True
    assert report["executed"] is False
    assert report["run_count"] == 12
    assert report["command_count"] == 48
    assert report["claim_boundary"]["causal_performance_claim_allowed"] is False


def test_missing_checkpoint_blocks(tmp: Path) -> None:
    env = make_fake_environment(tmp)
    missing = tmp / "missing_best_mappo.pt"

    completed = run_workflow(
        tmp=tmp,
        checkpoint=missing,
        preflight=env["preflight"],
        plan=env["plan"],
        runner=env["runner"],
        output_name="missing_checkpoint",
        expect_ok=False,
    )

    assert "checkpoint file does not exist" in completed.stderr

    report = read_json(tmp / "missing_checkpoint" / "h200_actual_checkpoint_arrival_workflow_report.json")
    assert report["arrival_workflow_status"] == "BLOCKED"


def test_blocked_preflight_blocks(tmp: Path) -> None:
    env = make_fake_environment(tmp)
    write_script(env["preflight"], fake_preflight_script("BLOCKED"))

    completed = run_workflow(
        tmp=tmp,
        checkpoint=env["checkpoint"],
        preflight=env["preflight"],
        plan=env["plan"],
        runner=env["runner"],
        output_name="blocked_preflight",
        expect_ok=False,
    )

    assert "preflight report is not PASS" in completed.stderr


def test_checkpoint_mismatch_blocks(tmp: Path) -> None:
    env = make_fake_environment(tmp)
    write_script(env["preflight"], fake_preflight_mismatch_script())

    completed = run_workflow(
        tmp=tmp,
        checkpoint=env["checkpoint"],
        preflight=env["preflight"],
        plan=env["plan"],
        runner=env["runner"],
        output_name="mismatch",
        expect_ok=False,
    )

    assert "checkpoint_path mismatch" in completed.stderr


def test_plan_not_ready_blocks(tmp: Path) -> None:
    env = make_fake_environment(tmp)
    write_script(env["plan"], fake_plan_script("BLOCKED_PREFLIGHT"))

    completed = run_workflow(
        tmp=tmp,
        checkpoint=env["checkpoint"],
        preflight=env["preflight"],
        plan=env["plan"],
        runner=env["runner"],
        output_name="plan_blocked",
        expect_ok=False,
    )

    assert "execution plan is not READY_TO_EXECUTE" in completed.stderr


def test_bad_dry_run_blocks(tmp: Path) -> None:
    env = make_fake_environment(tmp)
    write_script(env["runner"], fake_runner_script("DRY_RUN_VALIDATED", executed=True))

    completed = run_workflow(
        tmp=tmp,
        checkpoint=env["checkpoint"],
        preflight=env["preflight"],
        plan=env["plan"],
        runner=env["runner"],
        output_name="bad_dry_run",
        expect_ok=False,
    )

    assert "dry-run report is not DRY_RUN_VALIDATED" in completed.stderr


def test_doc_contains_required_contract() -> None:
    if not DOC.exists():
        raise AssertionError(f"doc not found: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")
    required = [
        "Step 62",
        "checkpoint_path",
        "h200_actual_checkpoint_preflight_report.json",
        "a_family_mappo_actual_execution_plan.json",
        "a_family_mappo_actual_matrix_runner_dry_run_report.json",
        "READY_FOR_MANUAL_EXECUTE",
        "READY_TO_EXECUTE",
        "DRY_RUN_VALIDATED",
        "HistoricalReplayAdapter",
        "It does not mean causal performance has been proven.",
    ]

    missing = [x for x in required if x not in text]
    if missing:
        raise AssertionError(f"doc missing required phrases: {missing}")


def main() -> None:
    if not WORKFLOW.exists():
        raise AssertionError(f"workflow runner not found: {WORKFLOW}")

    test_doc_contains_required_contract()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        test_success_path(tmp)
        test_missing_checkpoint_blocks(tmp)
        test_blocked_preflight_blocks(tmp)
        test_checkpoint_mismatch_blocks(tmp)
        test_plan_not_ready_blocks(tmp)
        test_bad_dry_run_blocks(tmp)

    print("[OK] Step 62 H200 actual checkpoint arrival workflow self-test PASS")


if __name__ == "__main__":
    main()
