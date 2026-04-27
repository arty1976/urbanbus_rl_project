from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "05_training" / "policies" / "run_h200_training_local_validation_bundle.py"
DOC = ROOT / "05_training" / "policies" / "H200_training_local_validation_bundle.md"


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_bundle(report_path: Path, *args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--project-root",
        str(ROOT),
        "--report-path",
        str(report_path),
        "--python-exe",
        sys.executable,
        *args,
    ]

    completed = subprocess.run(cmd, text=True, capture_output=True)

    if expect_ok and completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError(f"expected PASS but got returncode={completed.returncode}")

    if (not expect_ok) and completed.returncode == 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("expected BLOCKED but command passed")

    return completed


def test_doc_contains_contract() -> None:
    if not DOC.exists():
        raise AssertionError(f"doc not found: {DOC}")

    text = DOC.read_text(encoding="utf-8-sig")
    required = [
        "Step 70",
        "H200 training local validation bundle",
        "Step 58",
        "Step 59",
        "Step 60",
        "Step 61",
        "Step 62",
        "Step 65",
        "Step 66",
        "Step 67",
        "Step 68",
        "Step 69",
        "DRY_RUN_VALIDATED",
        "EXECUTION_VALIDATED",
        "Causal performance claims require Phase 2 causal simulator evaluation.",
    ]
    missing = [x for x in required if x not in text]
    if missing:
        raise AssertionError(f"doc missing required phrases: {missing}")


def test_dry_run_bundle_passes(tmp: Path) -> None:
    report_path = tmp / "step70_dry_run_report.json"
    completed = run_bundle(report_path, "--dry-run", expect_ok=True)

    assert "[OK] Step 70" in completed.stdout
    report = read_json(report_path)

    assert report["artifact_version"] == "h200_training_local_validation_bundle_v1_step70"
    assert report["bundle_status"] == "DRY_RUN_VALIDATED"
    assert report["dry_run"] is True
    assert report["executed"] is False
    assert report["required_file_check"]["passed"] is True
    assert report["command_count"] >= 10
    assert report["claim_boundary"]["trained_h200_checkpoint_exists"] is False
    assert report["claim_boundary"]["causal_performance_claim_allowed"] is False


def test_requires_single_mode(tmp: Path) -> None:
    report_path = tmp / "bad_mode_report.json"
    completed = run_bundle(report_path, expect_ok=False)
    assert "Choose exactly one mode" in completed.stderr


def main() -> None:
    if not RUNNER.exists():
        raise AssertionError(f"runner not found: {RUNNER}")

    test_doc_contains_contract()

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_dry_run_bundle_passes(tmp)
        test_requires_single_mode(tmp)

    print("[OK] Step 70 H200 training local validation bundle self-test PASS")


if __name__ == "__main__":
    main()
