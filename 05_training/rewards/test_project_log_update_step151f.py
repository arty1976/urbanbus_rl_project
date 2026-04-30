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


def base_locks() -> dict:
    return {
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def step150() -> dict:
    p = base_locks()
    p.update({
        "step": 150,
        "checklist_status": "READY_FOR_STEP151_EXPLICIT_OPERATOR_RELEASE_MANIFEST_DRAFT",
        "hard_failures": 0,
    })
    return p


def step151a() -> dict:
    p = base_locks()
    p.update({
        "step": "151-A",
        "release_manifest_status": "RELEASE_MANIFEST_DRAFT_STILL_LOCKED",
        "hard_failures": 0,
    })
    return p


def step151b() -> dict:
    p = base_locks()
    p.update({
        "step": "151-B",
        "checklist_status": "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED",
        "hard_failures": 0,
    })
    return p


def step151c(waiting: bool = True) -> dict:
    p = base_locks()
    p.update({
        "step": "151-C",
        "intake_status": "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED" if waiting else "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED",
        "hard_failures": 0,
        "h200_expected_preflight_manifest_present": not waiting,
        "h200_expected_preflight_passed": not waiting,
    })
    return p


def step151d() -> dict:
    p = base_locks()
    p.update({
        "step": "151-D",
        "approval_decision_status": "OPERATOR_APPROVAL_DECISION_DRAFT_STILL_LOCKED",
        "hard_failures": 0,
        "operator_approval_recorded": False,
        "operator_approval_granted": False,
    })
    return p


def step151e() -> dict:
    p = base_locks()
    p.update({
        "step": "151-E",
        "handoff_index_status": "H200_HANDOFF_PACKET_INDEX_READY_STILL_LOCKED",
        "hard_failures": 0,
        "operator_approval_recorded": False,
        "operator_approval_granted": False,
    })
    return p


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    gen = reward_dir / "project_log_update_step151f.py"
    val = reward_dir / "validate_project_log_update_step151f.py"

    selftest_root = project_root / "artifacts" / "rewards" / "project_log_update_step151f_selftest"
    input_root = selftest_root / "input"

    m150 = input_root / "step150.json"
    m151a = input_root / "step151a.json"
    m151b = input_root / "step151b.json"
    m151c = input_root / "step151c.json"
    m151d = input_root / "step151d.json"
    m151e = input_root / "step151e.json"
    m151e_bad = input_root / "step151e_bad.json"

    dump_json(m150, step150())
    dump_json(m151a, step151a())
    dump_json(m151b, step151b())
    dump_json(m151c, step151c(waiting=True))
    dump_json(m151d, step151d())
    dump_json(m151e, step151e())

    bad = step151e()
    bad["actual_execution_allowed"] = True
    dump_json(m151e_bad, bad)

    project_log = selftest_root / "project_log.md"
    project_log.parent.mkdir(parents=True, exist_ok=True)
    project_log.write_text("# Project Log\n\nInitial test log.\n", encoding="utf-8")

    output = selftest_root / "output"
    common_args = [
        sys.executable,
        str(gen),
        "--project-root",
        str(project_root),
        "--project-log",
        str(project_log),
        "--step150-manifest",
        str(m150),
        "--step151a-manifest",
        str(m151a),
        "--step151b-manifest",
        str(m151b),
        "--step151c-manifest",
        str(m151c),
        "--step151d-manifest",
        str(m151d),
    ]

    run(common_args + ["--step151e-manifest", str(m151e), "--output-root", str(output)], cwd=project_root, expect_ok=True)
    manifest = output / "project_log_update_step151f_manifest.json"
    run([sys.executable, str(val), "--manifest", str(manifest)], cwd=project_root, expect_ok=True)

    # Idempotency: second run should not duplicate the marker.
    run(common_args + ["--step151e-manifest", str(m151e), "--output-root", str(selftest_root / "output_second")], cwd=project_root, expect_ok=True)
    text = project_log.read_text(encoding="utf-8")
    if text.count("Step 151-F project log update: Step 150~151-E H200 handoff guard chain") != 1:
        raise RuntimeError("Step 151-F marker duplicated")

    run(common_args + ["--step151e-manifest", str(m151e_bad), "--output-root", str(selftest_root / "bad_output")], cwd=project_root, expect_ok=False)

    print("[OK] Step 151-F project log update self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
