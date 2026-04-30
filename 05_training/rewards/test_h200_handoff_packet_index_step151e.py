from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd, cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )

    if expect_ok and cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command failed: {cmd}")

    if (not expect_ok) and cp.returncode == 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command unexpectedly passed: {cmd}")

    return cp


def step151a() -> dict:
    return {
        "step": "151-A",
        "release_manifest_status": "RELEASE_MANIFEST_DRAFT_STILL_LOCKED",
        "decision": "DRAFT_ONLY_NOT_RELEASED",
        "hard_failures": 0,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def step151b() -> dict:
    return {
        "step": "151-B",
        "checklist_status": "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED",
        "decision": "CHECKLIST_ONLY_H200_RERUN_NOT_EXECUTED",
        "hard_failures": 0,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def step151c(waiting: bool = True) -> dict:
    return {
        "step": "151-C",
        "intake_status": (
            "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED"
            if waiting
            else "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED"
        ),
        "decision": "INTAKE_PLACEHOLDER_ONLY_EXECUTION_STILL_LOCKED",
        "hard_failures": 0,
        "h200_expected_preflight_manifest_present": not waiting,
        "h200_expected_preflight_passed": not waiting,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def step151d() -> dict:
    return {
        "step": "151-D",
        "approval_decision_status": "OPERATOR_APPROVAL_DECISION_DRAFT_STILL_LOCKED",
        "decision": "APPROVAL_DRAFT_ONLY_NOT_APPROVED_NOT_RELEASED",
        "hard_failures": 0,
        "operator_approval_recorded": False,
        "operator_approval_granted": False,
        "operator_approval_decision_final": False,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    gen = reward_dir / "h200_handoff_packet_index_step151e.py"
    val = reward_dir / "validate_h200_handoff_packet_index_step151e.py"

    selftest_root = project_root / "artifacts" / "rewards" / "h200_handoff_packet_index_step151e_selftest"
    input_root = selftest_root / "input"

    a = input_root / "step151a.json"
    b = input_root / "step151b.json"
    c_waiting = input_root / "step151c_waiting.json"
    c_validated = input_root / "step151c_validated.json"
    d = input_root / "step151d.json"
    d_bad = input_root / "step151d_bad.json"

    dump_json(a, step151a())
    dump_json(b, step151b())
    dump_json(c_waiting, step151c(waiting=True))
    dump_json(c_validated, step151c(waiting=False))
    dump_json(d, step151d())

    bad_d = step151d()
    bad_d["actual_execution_allowed"] = True
    dump_json(d_bad, bad_d)

    waiting_output = selftest_root / "waiting_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151a-manifest",
            str(a),
            "--step151b-manifest",
            str(b),
            "--step151c-manifest",
            str(c_waiting),
            "--step151d-manifest",
            str(d),
            "--output-root",
            str(waiting_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )
    waiting_manifest = waiting_output / "h200_handoff_packet_index_step151e_manifest.json"
    run([sys.executable, str(val), "--manifest", str(waiting_manifest)], cwd=project_root, expect_ok=True)

    validated_output = selftest_root / "validated_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151a-manifest",
            str(a),
            "--step151b-manifest",
            str(b),
            "--step151c-manifest",
            str(c_validated),
            "--step151d-manifest",
            str(d),
            "--output-root",
            str(validated_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )
    validated_manifest = validated_output / "h200_handoff_packet_index_step151e_manifest.json"
    run([sys.executable, str(val), "--manifest", str(validated_manifest)], cwd=project_root, expect_ok=True)

    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151a-manifest",
            str(a),
            "--step151b-manifest",
            str(b),
            "--step151c-manifest",
            str(c_waiting),
            "--step151d-manifest",
            str(d_bad),
            "--output-root",
            str(selftest_root / "bad_output"),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    print("[OK] Step 151-E H200 handoff packet index self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
