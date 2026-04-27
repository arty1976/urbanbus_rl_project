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


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def make_contract(path: Path) -> None:
    payload = {
        "artifact_version": "experiment_A_contract_v1_for_step55_smoke",
        "condition_id": "A",
        "qwen_train": False,
        "qwen_inference": False,
        "evaluation_horizon_minutes": 30,
        "shared_kpis": [
            "cv_headway",
            "avg_wait_seconds",
            "bunching_rate",
            "on_time_rate",
            "intervention_rate",
            "energy_proxy",
        ],
        "time_bands": ["peak", "offpeak", "night"],
        "fairness_constraints": {
            "same_initial_state": True,
            "same_exogenous_events": True,
            "same_eval_window": True,
        },
    }
    dump_json(path, payload)


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    import pandas as pd
    from policies.mappo_checkpoint_builder import save_mappo_checkpoint
    from policies.mappo_neural_inference_adapter import ActorCriticMLP

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "canonical_policy_metadata_smoke"
    ckpt_dir = out_dir / "checkpoints"
    rollout_root = out_dir / "rollouts"
    canonical_root = out_dir / "canonical_eval"
    report_dir = out_dir / "reports"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    contract_path = out_dir / "experiment_A_contract_step55.json"
    make_contract(contract_path)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    smoke_ckpt = ckpt_dir / "step55_smoke_checkpoint.pt"
    save_mappo_checkpoint(
        smoke_ckpt,
        model_state_dict=model.state_dict(),
        training_seed=55,
        git_commit="STEP55_SELFTEST_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "not_for_performance_claims": True,
        },
    )

    rollout_runner = training_dir / "run_a_family_policy_rollout_smoke.py"
    aggregator = training_dir / "evaluation" / "canonical_kpi_aggregator.py"
    metadata_validator = training_dir / "policies" / "validate_window_rollup_policy_metadata.py"

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
            "55",
            "--device",
            "cpu",
            "--output-root",
            str(rollout_root),
        ],
        expect_success=True,
    )

    rollout_run_dir = rollout_root / "A_mappo_smoke_seed_055"
    rollout_window = rollout_run_dir / "window_rollup.parquet"

    if not rollout_window.exists():
        raise RuntimeError(f"rollout window_rollup not found: {rollout_window}")

    pre_report = report_dir / "pre_canonical_window_rollup_metadata_validation.json"
    run_cmd(
        [
            sys.executable,
            str(metadata_validator),
            "--input",
            str(rollout_window),
            "--json-output",
            str(pre_report),
        ],
        expect_success=True,
    )

    run_cmd(
        [
            sys.executable,
            str(aggregator),
            "--mode",
            "official_rollup",
            "--contract",
            str(contract_path),
            "--input-root",
            str(rollout_run_dir),
            "--output-root",
            str(canonical_root),
            "--smoke",
        ],
        expect_success=True,
    )

    canonical_window = canonical_root / "kpi_by_window.parquet"
    if not canonical_window.exists():
        raise RuntimeError(f"canonical kpi_by_window not found: {canonical_window}")

    post_report = report_dir / "post_canonical_kpi_by_window_metadata_validation.json"
    run_cmd(
        [
            sys.executable,
            str(metadata_validator),
            "--input",
            str(canonical_window),
            "--json-output",
            str(post_report),
        ],
        expect_success=True,
    )

    df = pd.read_parquet(canonical_window)

    required_preserved = [
        "policy_metadata_version",
        "policy_source",
        "policy_action_source_version",
        "policy_action_source_mode",
        "checkpoint_path",
        "checkpoint_validation_mode",
        "checkpoint_validator_ran",
        "checkpoint_loaded",
        "trained_model",
        "performance_claim_allowed",
        "placeholder_fallback_used",
        "mock_action_used",
        "qwen_train",
        "qwen_inference",
        "reward_version",
        "energy_proxy_model_version",
        "k_dist_kwh_per_m",
        "k_acc_kwh_per_event",
        "k_idle_kwh_per_sec",
        "actual_policy_claim_ready",
        "causal_policy_claim_ready",
    ]

    missing = [c for c in required_preserved if c not in df.columns]
    if missing:
        raise RuntimeError(f"canonical kpi_by_window missing preserved policy metadata columns: {missing}")

    row = df.iloc[0].to_dict()

    checks = {
        "policy_source": "mappo_policy",
        "policy_action_source_version": "mappo_policy_action_source_v1",
        "checkpoint_validation_mode": "smoke",
        "reward_version": "mappo_reward_v1",
        "energy_proxy_model_version": "daegu_energy_proxy_v1",
    }

    for key, expected in checks.items():
        actual = row.get(key)
        if actual != expected:
            raise RuntimeError(f"{key} mismatch after canonical aggregation: expected {expected}, got {actual}")

    bool_checks = {
        "checkpoint_validator_ran": True,
        "checkpoint_loaded": True,
        "trained_model": False,
        "performance_claim_allowed": False,
        "placeholder_fallback_used": False,
        "mock_action_used": False,
        "qwen_train": False,
        "qwen_inference": False,
        "actual_policy_claim_ready": False,
        "causal_policy_claim_ready": False,
    }

    for key, expected in bool_checks.items():
        actual = bool(row.get(key))
        if actual is not expected:
            raise RuntimeError(f"{key} mismatch after canonical aggregation: expected {expected}, got {actual}")

    if float(row.get("qwen_trigger_rate", 1.0)) != 0.0:
        raise RuntimeError("qwen_trigger_rate must remain 0.0 after canonical aggregation")

    if str(row.get("source_mode", "")) != "noncausal_A_mappo_policy_smoke_v1":
        raise RuntimeError(f"source_mode was not preserved as expected: {row.get('source_mode')}")

    manifest = read_json(canonical_root / "aggregation_manifest.json")
    if manifest.get("causal_comparison_allowed") is not False:
        raise RuntimeError("smoke/noncausal canonical output must have causal_comparison_allowed=false")

    final_report = {
        "status": "PASS",
        "step": 55,
        "rollout_window": str(rollout_window),
        "canonical_window": str(canonical_window),
        "contract": str(contract_path),
        "pre_validation_report": str(pre_report),
        "post_validation_report": str(post_report),
        "canonical_manifest": str(canonical_root / "aggregation_manifest.json"),
        "preserved_columns": required_preserved,
        "source_mode": str(row.get("source_mode")),
        "policy_source": str(row.get("policy_source")),
        "actual_policy_claim_ready": bool(row.get("actual_policy_claim_ready")),
        "causal_policy_claim_ready": bool(row.get("causal_policy_claim_ready")),
        "causal_comparison_allowed": manifest.get("causal_comparison_allowed"),
        "note": (
            "Step 55 validates that canonical_kpi_aggregator.py preserves policy metadata "
            "from window_rollup.parquet into kpi_by_window.parquet. Smoke artifacts are not "
            "for performance claims."
        ),
    }

    out_report = out_dir / "step55_canonical_policy_metadata_smoke_report.json"
    dump_json(out_report, final_report)

    print("[OK] Step 55 canonical policy metadata preservation self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
