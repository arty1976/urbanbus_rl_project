from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def assert_raises(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        print(f"[OK] expected failure: {name}: {exc}")
        return
    raise AssertionError(f"{name}: expected failure but none was raised")


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    import torch
    from simulator_adapter_interface import load_adapter_class
    from policies.mappo_checkpoint_builder import save_mappo_checkpoint
    from policies.mappo_neural_inference_adapter import ActorCriticMLP
    from policies.mappo_policy_action_source import build_mappo_policy_action_source
    from policies.policy_source_metadata import (
        build_rollout_policy_metadata,
        policy_metadata_columns,
        validate_policy_metadata_records,
        validate_rollout_policy_metadata,
    )

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "policy_metadata_propagation_smoke"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    smoke_ckpt = ckpt_dir / "step49_smoke_checkpoint.pt"
    save_mappo_checkpoint(
        smoke_ckpt,
        model_state_dict=model.state_dict(),
        training_seed=49,
        git_commit="STEP49_SELFTEST_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "not_for_performance_claims": True,
        },
    )

    adapter_cls = load_adapter_class("adapters.historical_replay_adapter.HistoricalReplayAdapter")
    env = adapter_cls(
        {
            "condition_id": "A",
            "qwen_trigger_rate": 0.0,
        }
    )

    try:
        obs = env.reset(
            seed=49,
            scenario_config={
                "window_id": "step49_policy_metadata_propagation_smoke",
                "state_ts": "2024-01-01T08:00:00Z",
                "time_band": "offpeak",
                "snapshot_path": None,
                "effective_replay_step_minutes": 60,
            },
        )

        validation_report = out_dir / "step49_action_source_checkpoint_validation_report.json"

        source = build_mappo_policy_action_source(
            str(smoke_ckpt),
            validation_mode="smoke",
            device="cpu",
            deterministic=True,
            require_checkpoint=True,
            hidden_dim=128,
            seed=49,
            validation_report_path=str(validation_report),
        )

        action_result = source.select_actions(obs)

        rollout_metadata = build_rollout_policy_metadata(
            action_result.metadata,
            condition_id="A",
            causal_simulator=False,
        )

        cols = policy_metadata_columns()
        missing_cols = [c for c in cols if c not in rollout_metadata]
        if missing_cols:
            raise AssertionError(f"rollout metadata missing columns: {missing_cols}")

        result = validate_rollout_policy_metadata(rollout_metadata)

        if not result.valid:
            raise AssertionError(f"smoke rollout metadata should be structurally valid: {result.errors}")

        if rollout_metadata["policy_source"] != "mappo_policy":
            raise AssertionError("policy_source must be mappo_policy")

        if rollout_metadata["mock_action_used"] is not False:
            raise AssertionError("mock_action_used must be false")

        if rollout_metadata["placeholder_fallback_used"] is not False:
            raise AssertionError("placeholder_fallback_used must be false")

        if rollout_metadata["checkpoint_loaded"] is not True:
            raise AssertionError("checkpoint_loaded must be true")

        if rollout_metadata["checkpoint_validator_ran"] is not True:
            raise AssertionError("checkpoint_validator_ran must be true")

        if rollout_metadata["qwen_train"] is not False:
            raise AssertionError("qwen_train must be false")

        if rollout_metadata["qwen_inference"] is not False:
            raise AssertionError("qwen_inference must be false")

        if float(rollout_metadata["qwen_trigger_rate"]) != 0.0:
            raise AssertionError("qwen_trigger_rate must be 0.0")

        if rollout_metadata["energy_proxy_model_version"] != "daegu_energy_proxy_v1":
            raise AssertionError("energy_proxy_model_version mismatch")

        if rollout_metadata["actual_policy_claim_ready"] is not False:
            raise AssertionError("smoke checkpoint must not be actual_policy_claim_ready")

        if rollout_metadata["causal_policy_claim_ready"] is not False:
            raise AssertionError("noncausal smoke must not be causal_policy_claim_ready")

        strict_actual_result = validate_rollout_policy_metadata(
            rollout_metadata,
            require_actual_ready=True,
        )

        if strict_actual_result.valid:
            raise AssertionError("smoke metadata must fail require_actual_ready")

        bad_mock = dict(rollout_metadata)
        bad_mock["mock_action_used"] = True
        bad_mock_result = validate_rollout_policy_metadata(bad_mock)
        if bad_mock_result.valid:
            raise AssertionError("mock_action_used=true must fail")

        bad_placeholder = dict(rollout_metadata)
        bad_placeholder["placeholder_fallback_used"] = True
        bad_placeholder_result = validate_rollout_policy_metadata(bad_placeholder)
        if bad_placeholder_result.valid:
            raise AssertionError("placeholder_fallback_used=true must fail")

        bad_qwen = dict(rollout_metadata)
        bad_qwen["qwen_trigger_rate"] = 0.5
        bad_qwen_result = validate_rollout_policy_metadata(bad_qwen)
        if bad_qwen_result.valid:
            raise AssertionError("qwen_trigger_rate != 0 must fail")

        bad_energy = dict(rollout_metadata)
        bad_energy["energy_proxy_model_version"] = "wrong_energy_proxy"
        bad_energy_result = validate_rollout_policy_metadata(bad_energy)
        if bad_energy_result.valid:
            raise AssertionError("wrong energy proxy model must fail")

        causal_bad = dict(rollout_metadata)
        causal_bad["source_mode"] = "causal_A_mappo_policy_v1"
        causal_bad_result = validate_rollout_policy_metadata(causal_bad)
        if causal_bad_result.valid:
            raise AssertionError("causal source_mode without causal readiness must fail")

        batch_summary = validate_policy_metadata_records(
            [rollout_metadata, bad_mock, bad_qwen],
        )

        if batch_summary["valid"]:
            raise AssertionError("batch summary with bad rows must be invalid")

        out_payload = {
            "status": "PASS",
            "step": 49,
            "policy_metadata_version": rollout_metadata["policy_metadata_version"],
            "policy_metadata_columns": cols,
            "rollout_metadata": rollout_metadata,
            "strict_actual_errors": strict_actual_result.errors,
            "bad_mock_errors": bad_mock_result.errors,
            "bad_qwen_errors": bad_qwen_result.errors,
            "bad_energy_errors": bad_energy_result.errors,
            "causal_bad_errors": causal_bad_result.errors,
            "batch_summary": batch_summary,
            "action_count": len(action_result.actions),
            "note": (
                "This validates policy source metadata propagation readiness. "
                "Smoke metadata is structurally valid but not actual/casual claim-ready."
            ),
        }

        out_report = out_dir / "step49_policy_metadata_propagation_report.json"
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)

        print("[OK] Step 49 policy source metadata propagation self-test PASS")
        print(f"[OK] report: {out_report}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
