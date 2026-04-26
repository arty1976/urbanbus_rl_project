"""
Self-test for Step 33 friendly MAPPO checkpoint STOP message.

Run:
    python ./05_training/test_step33_friendly_checkpoint_stop.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def test_friendly_stop_message_for_missing_checkpoint() -> None:
    cmd = [
        PY,
        "05_training/run_causal_rollout.py",
        "--a-family-bridge",
        "--policy-kind",
        "mappo",
        "--checkpoint-path",
        "artifacts/experiment_A_v1/checkpoints/best.pt",
        "--require-existing-checkpoint",
        "--write-parquet",
    ]

    result = run_cmd(cmd)
    combined = result.stdout + result.stderr

    assert result.returncode != 0
    assert "[STOP] MAPPO checkpoint does not exist" in combined
    assert "This is expected before actual MAPPO training." in combined
    assert "No placeholder fallback was used." in combined
    assert "Traceback" not in combined


def main() -> None:
    test_friendly_stop_message_for_missing_checkpoint()
    print("[PASS] test_friendly_stop_message_for_missing_checkpoint")
    print("[OK] Step 33 friendly checkpoint STOP self-test passed")


if __name__ == "__main__":
    main()
