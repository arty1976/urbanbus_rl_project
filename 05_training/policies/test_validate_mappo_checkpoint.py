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


def build_base_checkpoint(model_state_dict):
    return {
        "artifact_version": "mappo_policy_checkpoint_v1",
        "contract_version": "mappo_checkpoint_contract_v1",
        "condition_id": "A",
        "qwen_train": False,
        "qwen_inference": False,
        "qwen_trigger_rate": 0.0,
        "policy_architecture": "ActorCriticMLP",
        "actor_obs_dim": 16,
        "critic_obs_dim": 64,
        "action_dim": 2,
        "hidden_dim": 128,
        "shared_policy": True,
        "ctde_enabled": True,
        "model_state_dict": model_state_dict,
        "training_seed": 1,
        "git_commit": "TEST_FAKE_NOT_FOR_PERFORMANCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trained_model": False,
        "performance_claim_allowed": False,
        "reward_version": "mappo_reward_v1",
        "rollout_schema_version": "rollout_schema_v1",
        "policy_interface_version": "mappo_policy_interface_v1",
        "action_space_version": "bus_control_action_v1",
        "observation_space_version": "urbanbus_observation_v1",
        "energy_proxy_model_version": "daegu_energy_proxy_v1",
        "energy_proxy_unit": "kwh_equivalent",
        "k_dist_kwh_per_m": 0.0012,
        "k_acc_kwh_per_event": 0.1800,
        "k_idle_kwh_per_sec": 0.0080,
        "boundary_test_marker": "fake_checkpoint_boundary_test_not_for_performance",
    }


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


def main() -> None:
    training_dir = append_training_dir_to_path()

    import torch
    from policies.mappo_neural_inference_adapter import ActorCriticMLP

    out_dir = training_dir.parent / "artifacts" / "experiment_A_v1" / "checkpoint_contract_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    valid_smoke_ckpt = build_base_checkpoint(model.state_dict())
    valid_smoke_path = out_dir / "step41_fake_valid_smoke_checkpoint.pt"
    torch.save(valid_smoke_ckpt, valid_smoke_path)

    invalid_qwen_ckpt = dict(valid_smoke_ckpt)
    invalid_qwen_ckpt["qwen_train"] = True
    invalid_qwen_path = out_dir / "step41_invalid_qwen_checkpoint.pt"
    torch.save(invalid_qwen_ckpt, invalid_qwen_path)

    invalid_energy_ckpt = dict(valid_smoke_ckpt)
    invalid_energy_ckpt["k_dist_kwh_per_m"] = 0.9999
    invalid_energy_path = out_dir / "step41_invalid_energy_checkpoint.pt"
    torch.save(invalid_energy_ckpt, invalid_energy_path)

    validator = training_dir / "policies" / "validate_mappo_checkpoint.py"

    smoke_report = out_dir / "valid_smoke_report.json"
    actual_fail_report = out_dir / "actual_mode_expected_fail_report.json"
    invalid_qwen_report = out_dir / "invalid_qwen_report.json"
    invalid_energy_report = out_dir / "invalid_energy_report.json"

    run_cmd(
        [
            sys.executable,
            str(validator),
            "--checkpoint",
            str(valid_smoke_path),
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--json-output",
            str(smoke_report),
        ],
        expect_success=True,
    )

    run_cmd(
        [
            sys.executable,
            str(validator),
            "--checkpoint",
            str(valid_smoke_path),
            "--mode",
            "actual",
            "--device",
            "cpu",
            "--json-output",
            str(actual_fail_report),
        ],
        expect_success=False,
    )

    run_cmd(
        [
            sys.executable,
            str(validator),
            "--checkpoint",
            str(invalid_qwen_path),
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--json-output",
            str(invalid_qwen_report),
        ],
        expect_success=False,
    )

    run_cmd(
        [
            sys.executable,
            str(validator),
            "--checkpoint",
            str(invalid_energy_path),
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--json-output",
            str(invalid_energy_report),
        ],
        expect_success=False,
    )

    with open(smoke_report, "r", encoding="utf-8") as f:
        smoke_payload = json.load(f)

    if not smoke_payload.get("valid", False):
        raise RuntimeError("valid smoke checkpoint report did not mark valid=true")

    if smoke_payload.get("performance_claim_allowed", True):
        raise RuntimeError("smoke checkpoint must not allow performance claims")

    print("[OK] Step 41 checkpoint validator self-test PASS")
    print(f"[OK] out_dir: {out_dir}")


if __name__ == "__main__":
    main()
