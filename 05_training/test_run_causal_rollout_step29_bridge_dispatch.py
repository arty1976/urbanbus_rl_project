"""
Self-test for Step 29 run_causal_rollout.py bridge dispatch.

Run:
    python .\05_training\test_run_causal_rollout_step29_bridge_dispatch.py
"""

from __future__ import annotations

import json
import shutil
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


def test_bridge_dispatch_self_test() -> None:
    out = ROOT / "artifacts" / "step29_bridge_dispatch_test"

    if out.exists():
        shutil.rmtree(out)

    cmd = [
        PY,
        "05_training/run_causal_rollout.py",
        "--a-family-bridge",
        "--self-test",
        "--output-root",
        str(out),
        "--limit",
        "2",
        "--seeds",
        "1",
        "--write-parquet",
    ]

    result = run_cmd(cmd)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise AssertionError("bridge dispatch self-test command failed")

    entry_manifest = out / "entry_manifest.json"
    summary = out / "summary.json"

    assert entry_manifest.exists()
    assert summary.exists()

    payload = json.loads(entry_manifest.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_bridge_dispatch_mappo_requires_checkpoint_path() -> None:
    out = ROOT / "artifacts" / "step29_bridge_dispatch_test_fail"

    if out.exists():
        shutil.rmtree(out)

    cmd = [
        PY,
        "05_training/run_causal_rollout.py",
        "--a-family-bridge",
        "--self-test",
        "--output-root",
        str(out),
        "--policy-kind",
        "mappo",
        "--checkpoint-path",
        "",
    ]

    result = run_cmd(cmd)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "checkpoint" in combined.lower()


def main() -> None:
    tests = [
        test_bridge_dispatch_self_test,
        test_bridge_dispatch_mappo_requires_checkpoint_path,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] Step 29 run_causal_rollout bridge dispatch self-test passed")


if __name__ == "__main__":
    main()
