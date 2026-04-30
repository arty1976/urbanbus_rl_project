from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd, cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if expect_ok and cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command failed: {cmd}")
    if (not expect_ok) and cp.returncode == 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command unexpectedly passed: {cmd}")
    return cp


def good_151e() -> dict:
    return {
        "artifact_version": "h200_handoff_packet_index_step151e_v1",
        "step": "151-E",
        "handoff_index_status": "H200_HANDOFF_PACKET_INDEX_READY_STILL_LOCKED",
        "decision": "INDEX_ONLY_HANDOFF_NOT_RELEASED",
        "hard_failures": 0,
        "operator_approval_recorded": False,
        "operator_approval_granted": False,
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
    gen = reward_dir / "h200_side_execution_prompt_server_operator_packet_step151g.py"
    val = reward_dir / "validate_h200_side_execution_prompt_server_operator_packet_step151g.py"
    selftest_root = project_root / "artifacts" / "rewards" / "h200_side_execution_prompt_server_operator_packet_step151g_selftest"
    input_root = selftest_root / "input"
    good = input_root / "step151e_good.json"
    bad = input_root / "step151e_bad.json"
    dump_json(good, good_151e())
    bad_payload = good_151e()
    bad_payload["actual_execution_allowed"] = True
    dump_json(bad, bad_payload)
    positive_output = selftest_root / "positive_output"
    run([sys.executable, str(gen), "--project-root", str(project_root), "--step151e-manifest", str(good), "--output-root", str(positive_output)], cwd=project_root, expect_ok=True)
    manifest = positive_output / "h200_side_execution_prompt_server_operator_packet_step151g_manifest.json"
    run([sys.executable, str(val), "--manifest", str(manifest)], cwd=project_root, expect_ok=True)
    run([sys.executable, str(gen), "--project-root", str(project_root), "--step151e-manifest", str(bad), "--output-root", str(selftest_root / "negative_output")], cwd=project_root, expect_ok=False)
    print("[OK] Step 151-G H200-side execution prompt server operator packet self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
