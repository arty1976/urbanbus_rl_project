from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


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


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    import pandas as pd
    import torch
    from policies.mappo_checkpoint_builder import save_mappo_checkpoint
    from policies.mappo_neural_inference_adapter import ActorCriticMLP

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "window_rollup_policy_metadata_smoke"
    ckpt_dir = out_dir / "checkpoints"
    run_root = out_dir / "runs"
    report_dir = out_dir / "reports"
    invalid_dir = out_dir / "invalid_inputs"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    smoke_ckpt = ckpt_dir / "step54_smoke_checkpoint.pt"
    save_mappo_checkpoint(
        smoke_ckpt,
        model_state_dict=model.state_dict(),
        training_seed=54,
        git_commit="STEP54_SELFTEST_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "not_for_performance_claims": True,
        },
    )

    rollout_runner = training_dir / "run_a_family_policy_rollout_smoke.py"
    validator = training_dir / "policies" / "validate_window_rollup_policy_metadata.py"

    run_cmd(
        [
            sys.executable,
            str(rollout_runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mappo_smoke",
            "--checkpoint-path",
            str(smoke_ckpt),
            "--checkpoint-validation-mode",
            "smoke",
            "--seed",
            "54",
            "--device",
            "cpu",
            "--output-root",
            str(run_root),
        ],
        expect_success=True,
    )

    mappo_rollup = run_root / "A_mappo_smoke_seed_054" / "window_rollup.parquet"
    if not mappo_rollup.exists():
        raise RuntimeError(f"mappo window_rollup not found: {mappo_rollup}")

    mappo_report = report_dir / "mappo_smoke_validation_report.json"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--input",
            str(mappo_rollup),
            "--json-output",
            str(mappo_report),
        ],
        expect_success=True,
    )

    mappo_payload = read_json(mappo_report)
    if not mappo_payload.get("valid", False):
        raise RuntimeError("mappo smoke validation report must be valid=true")
    if mappo_payload.get("mappo_row_count") != 1:
        raise RuntimeError("mappo_row_count must be 1")
    if mappo_payload.get("actual_policy_claim_ready_count") != 0:
        raise RuntimeError("mappo smoke must not be actual claim-ready")

    run_cmd(
        [
            sys.executable,
            str(rollout_runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mock_smoke",
            "--allow-mock",
            "--seed",
            "54",
            "--output-root",
            str(run_root),
        ],
        expect_success=True,
    )

    mock_rollup = run_root / "A_mock_smoke_seed_054" / "window_rollup.parquet"
    if not mock_rollup.exists():
        raise RuntimeError(f"mock window_rollup not found: {mock_rollup}")

    mock_report = report_dir / "mock_allow_validation_report.json"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--input",
            str(mock_rollup),
            "--allow-mock",
            "--json-output",
            str(mock_report),
        ],
        expect_success=True,
    )

    mock_no_allow_report = report_dir / "mock_no_allow_expected_fail_report.json"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--input",
            str(mock_rollup),
            "--json-output",
            str(mock_no_allow_report),
        ],
        expect_success=False,
    )

    actual_required_report = report_dir / "mappo_require_actual_expected_fail_report.json"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--input",
            str(mappo_rollup),
            "--require-actual-ready",
            "--json-output",
            str(actual_required_report),
        ],
        expect_success=False,
    )

    df = pd.read_parquet(mappo_rollup)

    invalid_qwen = df.copy()
    invalid_qwen["qwen_trigger_rate"] = 0.5
    invalid_qwen_path = invalid_dir / "invalid_qwen_window_rollup.parquet"
    invalid_qwen.to_parquet(invalid_qwen_path, index=False)

    invalid_qwen_report = report_dir / "invalid_qwen_expected_fail_report.json"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--input",
            str(invalid_qwen_path),
            "--json-output",
            str(invalid_qwen_report),
        ],
        expect_success=False,
    )

    invalid_missing = df.drop(columns=["policy_source"])
    invalid_missing_path = invalid_dir / "invalid_missing_policy_source.parquet"
    invalid_missing.to_parquet(invalid_missing_path, index=False)

    invalid_missing_report = report_dir / "invalid_missing_policy_source_expected_fail_report.json"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--input",
            str(invalid_missing_path),
            "--json-output",
            str(invalid_missing_report),
        ],
        expect_success=False,
    )

    invalid_mock_flag = df.copy()
    invalid_mock_flag["mock_action_used"] = True
    invalid_mock_flag_path = invalid_dir / "invalid_mock_action_flag.parquet"
    invalid_mock_flag.to_parquet(invalid_mock_flag_path, index=False)

    invalid_mock_flag_report = report_dir / "invalid_mock_action_flag_expected_fail_report.json"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--input",
            str(invalid_mock_flag_path),
            "--json-output",
            str(invalid_mock_flag_report),
        ],
        expect_success=False,
    )

    final_report = {
        "status": "PASS",
        "step": 54,
        "validator": str(validator),
        "mappo_rollup": str(mappo_rollup),
        "mock_rollup": str(mock_rollup),
        "reports": {
            "mappo_report": str(mappo_report),
            "mock_report": str(mock_report),
            "mock_no_allow_report": str(mock_no_allow_report),
            "actual_required_report": str(actual_required_report),
            "invalid_qwen_report": str(invalid_qwen_report),
            "invalid_missing_report": str(invalid_missing_report),
            "invalid_mock_flag_report": str(invalid_mock_flag_report),
        },
        "note": (
            "Step 54 validates policy metadata columns in window_rollup.parquet. "
            "Smoke artifacts are not for performance claims."
        ),
    }

    out_report = out_dir / "step54_window_rollup_policy_metadata_selftest_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("[OK] Step 54 window_rollup policy metadata self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
