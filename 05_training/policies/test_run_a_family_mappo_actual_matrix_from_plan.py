from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


RUNNER = Path(__file__).resolve().parent / "run_a_family_mappo_actual_matrix_from_plan.py"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def base_actual_guard() -> dict:
    return {
        "policy_source_mode": "mappo_actual",
        "checkpoint_validation_mode": "actual",
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
    }


def base_causal_guard() -> dict:
    return {
        "require_causal_claim": False,
        "causal_performance_claim_allowed": False,
        "adapter_claim_scope": "actual_policy_path_only_noncausal_replay",
    }


def make_run(condition_id: str, seed: int) -> dict:
    root = f"artifacts/experiment_A_v1/mappo_actual/{condition_id}/seed_{seed:03d}"
    return {
        "condition_id": condition_id,
        "seed": seed,
        "policy_source_mode": "mappo_actual",
        "checkpoint_validation_mode": "actual",
        "simulator_adapter": "adapters.historical_replay_adapter.HistoricalReplayAdapter",
        "rollout_command": f"python 05_training/policies/fake_rollout.py --condition {condition_id} --seed {seed}",
        "metadata_validation_command": f"python 05_training/policies/validate_window_rollup_policy_metadata.py --input {root}/window_rollup.parquet --require-actual-ready",
        "canonical_command": f"python 05_training/evaluation/canonical_kpi_aggregator.py --mode official_rollup --input-root {root}",
        "post_canonical_validation_command": f"python 05_training/policies/validate_window_rollup_policy_metadata.py --input {root}/canonical_eval/kpi_by_window.parquet --require-actual-ready",
        "expected_outputs": [
            f"{root}/window_rollup.parquet",
            f"{root}/canonical_eval/kpi_by_window.parquet",
            f"{root}/canonical_eval/kpi_by_seed.parquet",
            f"{root}/canonical_eval/kpi_by_time_band.parquet",
            f"{root}/canonical_eval/kpi_overall.json",
        ],
        "actual_claim_guard": base_actual_guard(),
        "causal_claim_guard": base_causal_guard(),
    }


def make_plan(status: str = "READY_TO_EXECUTE") -> dict:
    return {
        "artifact_version": "a_family_mappo_actual_execution_plan_v1_step59",
        "status": status,
        "policy_source_mode": "mappo_actual",
        "checkpoint_validation_mode": "actual",
        "simulator_adapter": "adapters.historical_replay_adapter.HistoricalReplayAdapter",
        "actual_claim_guard": base_actual_guard(),
        "causal_claim_guard": base_causal_guard(),
        "runs": [
            make_run("A", 1),
            make_run("A90", 1),
        ],
    }


def run_runner(plan_path: Path, report_path: Path, expect_ok: bool) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--plan-json",
        str(plan_path),
        "--report-path",
        str(report_path),
        "--dry-run",
    ]
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    if expect_ok and completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError(f"expected PASS but got returncode={completed.returncode}")

    if (not expect_ok) and completed.returncode == 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("expected BLOCKED but command passed")

    return completed


def test_ready_plan_dry_run_passes(tmp: Path) -> None:
    plan_path = tmp / "ready_plan.json"
    report_path = tmp / "ready_report.json"
    write_json(plan_path, make_plan())

    completed = run_runner(plan_path, report_path, expect_ok=True)
    assert "[OK] Step 60" in completed.stdout

    report = read_json(report_path)
    assert report["runner_status"] == "DRY_RUN_VALIDATED"
    assert report["dry_run"] is True
    assert report["executed"] is False
    assert report["run_count"] == 2
    assert report["command_count"] == 8
    assert report["actual_claim_guard"]["all_passed"] is True
    assert report["causal_claim_guard"]["all_passed"] is True

    expected_order = [
        "rollout_command",
        "metadata_validation_command",
        "canonical_command",
        "post_canonical_validation_command",
    ]
    assert report["command_order_contract"] == expected_order

    first_run_commands = [
        c["command_key"]
        for c in report["commands"]
        if c["run_index"] == 0
    ]
    assert first_run_commands == expected_order

    assert len(report["expected_outputs"]) >= 5


def test_blocked_plan_is_rejected(tmp: Path) -> None:
    plan = make_plan(status="BLOCKED_PREFLIGHT")
    plan_path = tmp / "blocked_plan.json"
    report_path = tmp / "blocked_report.json"
    write_json(plan_path, plan)

    completed = run_runner(plan_path, report_path, expect_ok=False)
    assert "not READY_TO_EXECUTE" in completed.stderr

    report = read_json(report_path)
    assert report["runner_status"] == "BLOCKED"


def test_non_a_family_condition_is_rejected(tmp: Path) -> None:
    plan = make_plan()
    plan["runs"][0]["condition_id"] = "B2"

    plan_path = tmp / "bad_condition_plan.json"
    report_path = tmp / "bad_condition_report.json"
    write_json(plan_path, plan)

    completed = run_runner(plan_path, report_path, expect_ok=False)
    assert "non A-family condition" in completed.stderr


def test_missing_command_is_rejected(tmp: Path) -> None:
    plan = make_plan()
    del plan["runs"][0]["canonical_command"]

    plan_path = tmp / "missing_command_plan.json"
    report_path = tmp / "missing_command_report.json"
    write_json(plan_path, plan)

    completed = run_runner(plan_path, report_path, expect_ok=False)
    assert "missing required command" in completed.stderr


def test_causal_claim_on_historical_replay_is_rejected(tmp: Path) -> None:
    plan = make_plan()
    plan["causal_claim_guard"]["require_causal_claim"] = True

    plan_path = tmp / "bad_causal_plan.json"
    report_path = tmp / "bad_causal_report.json"
    write_json(plan_path, plan)

    completed = run_runner(plan_path, report_path, expect_ok=False)
    assert "cannot require causal claim" in completed.stderr


def test_actual_claim_guard_rejects_smoke_mode(tmp: Path) -> None:
    plan = make_plan()
    plan["runs"][0]["actual_claim_guard"]["policy_source_mode"] = "mappo_smoke"

    plan_path = tmp / "bad_actual_plan.json"
    report_path = tmp / "bad_actual_report.json"
    write_json(plan_path, plan)

    completed = run_runner(plan_path, report_path, expect_ok=False)
    assert "policy_source_mode must be mappo_actual" in completed.stderr


def main() -> None:
    if not RUNNER.exists():
        raise AssertionError(f"runner not found: {RUNNER}")

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        test_ready_plan_dry_run_passes(tmp)
        test_blocked_plan_is_rejected(tmp)
        test_non_a_family_condition_is_rejected(tmp)
        test_missing_command_is_rejected(tmp)
        test_causal_claim_on_historical_replay_is_rejected(tmp)
        test_actual_claim_guard_rejects_smoke_mode(tmp)

    print("[OK] Step 60 mappo_actual matrix runner self-test PASS")


if __name__ == "__main__":
    main()
