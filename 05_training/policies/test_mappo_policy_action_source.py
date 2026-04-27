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


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_raises(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        print(f"[OK] expected failure: {name}: {exc}")
        return
    raise AssertionError(f"{name}: expected exception but none was raised")


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    import torch
    from simulator_adapter_interface import load_adapter_class
    from policies.mappo_checkpoint_builder import save_mappo_checkpoint
    from policies.mappo_neural_inference_adapter import ActorCriticMLP
    from policies.mappo_policy_action_source import build_mappo_policy_action_source

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "policy_action_source_smoke"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    smoke_ckpt = ckpt_dir / "step48_smoke_checkpoint.pt"
    save_mappo_checkpoint(
        smoke_ckpt,
        model_state_dict=model.state_dict(),
        training_seed=48,
        git_commit="STEP48_SELFTEST_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "not_for_performance_claims": True,
        },
    )

    qwen_ckpt = ckpt_dir / "step48_invalid_qwen_checkpoint.pt"
    payload = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    payload["qwen_train"] = True
    torch.save(payload, qwen_ckpt)

    adapter_cls = load_adapter_class("adapters.historical_replay_adapter.HistoricalReplayAdapter")
    env = adapter_cls(
        {
            "condition_id": "A",
            "qwen_trigger_rate": 0.0,
        }
    )

    try:
        obs = env.reset(
            seed=48,
            scenario_config={
                "window_id": "step48_policy_action_source_smoke",
                "state_ts": "2024-01-01T08:00:00Z",
                "time_band": "offpeak",
                "snapshot_path": None,
                "effective_replay_step_minutes": 60,
            },
        )

        validation_report = out_dir / "step48_action_source_checkpoint_validation_report.json"

        source = build_mappo_policy_action_source(
            str(smoke_ckpt),
            validation_mode="smoke",
            device="cpu",
            deterministic=True,
            require_checkpoint=True,
            hidden_dim=128,
            seed=48,
            validation_report_path=str(validation_report),
        )

        result = source.select_actions(obs)

        if not result.actions:
            raise AssertionError("actions must not be empty")

        metadata = result.metadata

        if metadata.get("policy_source") != "mappo_policy":
            raise AssertionError("policy_source must be mappo_policy")

        if metadata.get("policy_action_source_mode") != "mappo_policy_smoke_v1":
            raise AssertionError("policy_action_source_mode mismatch")

        if metadata.get("checkpoint_validator_ran") is not True:
            raise AssertionError("checkpoint_validator_ran must be true")

        if metadata.get("checkpoint_loaded") is not True:
            raise AssertionError("checkpoint_loaded must be true")

        if metadata.get("placeholder_fallback_used") is not False:
            raise AssertionError("placeholder_fallback_used must be false")

        if metadata.get("mock_action_used") is not False:
            raise AssertionError("mock_action_used must be false")

        if metadata.get("qwen_train") is not False:
            raise AssertionError("qwen_train must be false")

        if metadata.get("qwen_inference") is not False:
            raise AssertionError("qwen_inference must be false")

        if float(metadata.get("qwen_trigger_rate", 0.0)) != 0.0:
            raise AssertionError("qwen_trigger_rate must be 0.0")

        if metadata.get("energy_proxy_model_version") != "daegu_energy_proxy_v1":
            raise AssertionError("energy_proxy_model_version mismatch")

        if not validation_report.exists():
            raise AssertionError("validation_report was not written")

        assert_raises(
            "missing checkpoint must stop",
            lambda: build_mappo_policy_action_source(
                "",
                validation_mode="smoke",
                device="cpu",
                require_checkpoint=True,
            ),
        )

        assert_raises(
            "smoke checkpoint must fail actual mode",
            lambda: build_mappo_policy_action_source(
                str(smoke_ckpt),
                validation_mode="actual",
                device="cpu",
                require_checkpoint=True,
            ).select_actions(obs),
        )

        assert_raises(
            "qwen-invalid checkpoint must stop",
            lambda: build_mappo_policy_action_source(
                str(qwen_ckpt),
                validation_mode="smoke",
                device="cpu",
                require_checkpoint=True,
            ).select_actions(obs),
        )

        out_payload = {
            "status": "PASS",
            "step": 48,
            "policy_source": metadata.get("policy_source"),
            "policy_action_source_version": metadata.get("policy_action_source_version"),
            "policy_action_source_mode": metadata.get("policy_action_source_mode"),
            "checkpoint_path": str(smoke_ckpt),
            "validation_report": str(validation_report),
            "checkpoint_loaded": metadata.get("checkpoint_loaded"),
            "checkpoint_validator_ran": metadata.get("checkpoint_validator_ran"),
            "trained_model": metadata.get("trained_model"),
            "performance_claim_allowed": metadata.get("performance_claim_allowed"),
            "qwen_trigger_rate": metadata.get("qwen_trigger_rate"),
            "energy_proxy_model_version": metadata.get("energy_proxy_model_version"),
            "action_count": metadata.get("action_count"),
            "actions_preview": metadata.get("actions_preview"),
            "note": (
                "This validates the MAPPO policy action source bridge only. "
                "Smoke checkpoint is not for performance claims."
            ),
        }

        out_report = out_dir / "step48_policy_action_source_smoke_report.json"
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)

        print("[OK] Step 48 MAPPO policy action source self-test PASS")
        print(f"[OK] report: {out_report}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
