from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_cmd(cmd: list[str], cwd: Path, expect_success: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)

    if expect_success and completed.returncode != 0:
        print("[STDOUT]")
        print(completed.stdout)
        print("[STDERR]")
        print(completed.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")

    if not expect_success and completed.returncode == 0:
        print("[STDOUT]")
        print(completed.stdout)
        print("[STDERR]")
        print(completed.stderr)
        raise RuntimeError(f"command unexpectedly succeeded: {' '.join(cmd)}")

    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    return completed


def test_smoke_checkpoint_validator() -> None:
    py = Path(sys.executable)

    output_root = ROOT / "artifacts" / "phase2_toy_causal_mappo_checkpoint_validator_selftest"
    if output_root.exists():
        shutil.rmtree(output_root)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "train_toy_causal_mappo_smoke.py"),
            "--output-root",
            str(output_root),
            "--condition-id",
            "A",
            "--seed",
            "89",
            "--num-agents",
            "8",
            "--steps",
            "4",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--clean",
        ],
        cwd=ROOT,
    )

    checkpoint_path = output_root / "checkpoints" / "toy_mappo_smoke_checkpoint.pt"
    manifest_path = output_root / "training_manifest.json"
    trace_path = output_root / "training_trace.csv"
    report_path = output_root / "checkpoint_validation_report.json"

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "policies" / "validate_toy_mappo_smoke_checkpoint.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--manifest",
            str(manifest_path),
            "--trace",
            str(trace_path),
            "--output-json",
            str(report_path),
        ],
        cwd=ROOT,
    )

    assert_true(report_path.exists(), "checkpoint validation report missing")
    report = load_json(report_path)

    assert_true(report["valid"] is True, "report valid flag must be true")
    assert_true(report["artifact_version"] == "toy_mappo_smoke_checkpoint_validation_v1_step89", "report artifact version mismatch")
    assert_true(report["reward_version"] == "mappo_reward_v1", "reward version mismatch")
    assert_true(report["trained_model"] is False, "trained_model must be false")
    assert_true(report["performance_claim_allowed"] is False, "performance claim must be false")
    assert_true(report["smoke_training_only"] is True, "smoke_training_only must be true")
    assert_true(report["causal_comparison_allowed"] is True, "causal flag must be true")
    assert_true(report["total_transitions"] == 32, "total_transitions should be 32")
    assert_true(set(report["reward_metric_keys"]) == {
        "cv_headway",
        "avg_wait_seconds",
        "bunching_rate",
        "on_time_rate",
        "intervention_rate",
        "energy_proxy",
        "passenger_demand_generated",
        "passenger_served_count",
        "passenger_service_rate",
        "passenger_wait_p95_seconds",
        "energy_proxy_per_passenger",
        "fleet_reduction_ratio",
    }, "12-KPI key set mismatch")
    assert_true(report["trace_summary"]["row_count"] > 0, "trace row count must be positive")
    assert_true("not a paper-level performance claim" in report["note"], "report note must preserve claim boundary")

    # Negative smoke: tamper manifest performance_claim_allowed to true and ensure validator fails.
    bad_manifest_path = output_root / "training_manifest_bad_claim.json"
    bad_manifest = load_json(manifest_path)
    bad_manifest["performance_claim_allowed"] = True
    with open(bad_manifest_path, "w", encoding="utf-8") as f:
        json.dump(bad_manifest, f, ensure_ascii=False, indent=2)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "policies" / "validate_toy_mappo_smoke_checkpoint.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--manifest",
            str(bad_manifest_path),
            "--trace",
            str(trace_path),
        ],
        cwd=ROOT,
        expect_success=False,
    )

    print("[OK] Step 89 toy MAPPO smoke checkpoint validator self-test PASS")


def main() -> None:
    test_smoke_checkpoint_validator()


if __name__ == "__main__":
    main()
