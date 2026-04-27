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

    import torch
    from policies.mappo_checkpoint_builder import (
        build_mappo_checkpoint_payload,
        checkpoint_contract_metadata,
        save_mappo_checkpoint,
        write_checkpoint_contract_preview_json,
    )
    from policies.mappo_neural_inference_adapter import ActorCriticMLP
    from mappo_runner import MAPPOExperimentRunner, RunnerConfig

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "checkpoint_builder_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = checkpoint_contract_metadata(
        training_seed=123,
        git_commit="TEST_COMMIT_FOR_STEP45",
        trained_model=False,
        performance_claim_allowed=False,
    )

    required = [
        "artifact_version",
        "contract_version",
        "condition_id",
        "qwen_train",
        "qwen_inference",
        "qwen_trigger_rate",
        "reward_version",
        "energy_proxy_model_version",
        "energy_proxy_unit",
        "k_dist_kwh_per_m",
        "k_acc_kwh_per_event",
        "k_idle_kwh_per_sec",
    ]

    missing = [k for k in required if k not in metadata]
    if missing:
        raise RuntimeError(f"metadata missing required keys: {missing}")

    if metadata["condition_id"] != "A":
        raise RuntimeError("condition_id must be A")

    if metadata["qwen_train"] or metadata["qwen_inference"] or metadata["qwen_trigger_rate"] != 0.0:
        raise RuntimeError("Qwen must be disabled for Experiment A")

    if metadata["energy_proxy_model_version"] != "daegu_energy_proxy_v1":
        raise RuntimeError("energy_proxy_model_version mismatch")

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    payload = build_mappo_checkpoint_payload(
        model_state_dict=model.state_dict(),
        training_seed=123,
        git_commit="TEST_COMMIT_FOR_STEP45",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
    )

    if "model_state_dict" not in payload:
        raise RuntimeError("payload missing model_state_dict")

    ckpt_path = out_dir / "step45_builder_smoke_checkpoint.pt"
    save_mappo_checkpoint(
        ckpt_path,
        model_state_dict=model.state_dict(),
        training_seed=123,
        git_commit="TEST_COMMIT_FOR_STEP45",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "smoke_test": True,
            "not_for_performance_claims": True,
        },
    )

    validator = training_dir / "policies" / "validate_mappo_checkpoint.py"
    report_path = out_dir / "step45_builder_smoke_validation_report.json"

    run_cmd(
        [
            sys.executable,
            str(validator),
            "--checkpoint",
            str(ckpt_path),
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--json-output",
            str(report_path),
        ],
        expect_success=True,
    )

    report = read_json(report_path)
    if not report.get("valid", False):
        raise RuntimeError("validator report must be valid=true in smoke mode")
    if not report.get("state_dict_load_ok", False):
        raise RuntimeError("state_dict_load_ok must be true")

    preview_path = out_dir / "checkpoint_contract_preview.json"
    preview = write_checkpoint_contract_preview_json(
        preview_path,
        project_root=project_root,
        seed=123,
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "smoke_test": True,
            "source": "test_mappo_checkpoint_builder",
        },
    )

    if preview["is_torch_checkpoint"]:
        raise RuntimeError("preview must not be a torch checkpoint")

    if preview["has_model_state_dict"]:
        raise RuntimeError("preview must not have model_state_dict")

    if preview["actual_checkpoint_allowed"]:
        raise RuntimeError("preview must not allow actual checkpoint use")

    # Direct runner smoke: do not call run_experiment_A_stub.py CLI here.
    # The CLI signature is intentionally small. We test the patched runner method directly.
    run_root = out_dir / "runner_contract_preview"
    run_dir = run_root / "seed_123"
    run_dir.mkdir(parents=True, exist_ok=True)

    runner_config = RunnerConfig(
        root_dir=str(project_root),
        experiment_contract_path=str(project_root / "artifacts" / "experiment_A_v1" / "experiment_A_contract.json"),
        run_root_dir=str(run_root),
        simulator_adapter_path="adapters.historical_replay_adapter.HistoricalReplayAdapter",
    )

    runner = MAPPOExperimentRunner(runner_config)
    runner.write_checkpoint_stub(run_dir, seed=123)

    checkpoint_stub_path = run_dir / "checkpoint_stub.json"
    runner_preview_path = run_dir / "checkpoint_contract_preview.json"

    if not checkpoint_stub_path.exists():
        raise RuntimeError("runner did not write checkpoint_stub.json")

    if not runner_preview_path.exists():
        raise RuntimeError("runner did not write checkpoint_contract_preview.json")

    runner_stub = read_json(checkpoint_stub_path)
    runner_preview = read_json(runner_preview_path)

    if runner_stub.get("trained_model") is not False:
        raise RuntimeError("runner checkpoint_stub must have trained_model=false")

    if runner_stub.get("performance_claim_allowed") is not False:
        raise RuntimeError("runner checkpoint_stub must have performance_claim_allowed=false")

    required_metadata = runner_preview.get("required_checkpoint_metadata", {})
    if required_metadata.get("contract_version") != "mappo_checkpoint_contract_v1":
        raise RuntimeError("runner preview contract_version mismatch")

    if required_metadata.get("energy_proxy_model_version") != "daegu_energy_proxy_v1":
        raise RuntimeError("runner preview energy_proxy_model_version mismatch")

    if required_metadata.get("qwen_train") is not False:
        raise RuntimeError("runner preview qwen_train must be false")

    if required_metadata.get("qwen_inference") is not False:
        raise RuntimeError("runner preview qwen_inference must be false")

    final_report = {
        "status": "PASS",
        "step": 45,
        "checkpoint_builder": "policies.mappo_checkpoint_builder",
        "smoke_checkpoint_path": str(ckpt_path),
        "validator_report": str(report_path),
        "preview_path": str(preview_path),
        "runner_checkpoint_stub": str(checkpoint_stub_path),
        "runner_checkpoint_contract_preview": str(runner_preview_path),
        "note": (
            "Step 45 validates checkpoint payload building and runner checkpoint "
            "contract preview. This is not an actual trained MAPPO checkpoint."
        ),
    }

    out_report = out_dir / "step45_checkpoint_builder_smoke_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("[OK] Step 45 MAPPO checkpoint builder self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
