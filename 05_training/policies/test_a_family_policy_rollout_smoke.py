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
    from policies.mappo_checkpoint_builder import save_mappo_checkpoint
    from policies.mappo_neural_inference_adapter import ActorCriticMLP

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "a_family_policy_rollout_smoke_selftest"
    ckpt_dir = out_dir / "checkpoints"
    run_root = out_dir / "runs"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    smoke_ckpt = ckpt_dir / "step52_smoke_checkpoint.pt"
    save_mappo_checkpoint(
        smoke_ckpt,
        model_state_dict=model.state_dict(),
        training_seed=52,
        git_commit="STEP52_SELFTEST_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "not_for_performance_claims": True,
        },
    )

    qwen_ckpt = ckpt_dir / "step52_invalid_qwen_checkpoint.pt"
    payload = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    payload["qwen_train"] = True
    torch.save(payload, qwen_ckpt)

    runner = training_dir / "run_a_family_policy_rollout_smoke.py"

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mock_smoke",
            "--allow-mock",
            "--seed",
            "52",
            "--output-root",
            str(run_root),
        ],
        expect_success=True,
    )

    mock_status = read_json(run_root / "A_mock_smoke_seed_052" / "status.json")
    if mock_status["policy_source"] != "mock_policy":
        raise RuntimeError("mock_smoke must write policy_source=mock_policy")
    if mock_status["mock_action_used"] is not True:
        raise RuntimeError("mock_smoke must set mock_action_used=true")
    if mock_status["actual_policy_claim_ready"] is not False:
        raise RuntimeError("mock_smoke must not be actual claim ready")

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mappo_smoke",
            "--checkpoint-path",
            str(smoke_ckpt),
            "--checkpoint-validation-mode",
            "smoke",
            "--seed",
            "52",
            "--device",
            "cpu",
            "--output-root",
            str(run_root),
        ],
        expect_success=True,
    )

    mappo_status = read_json(run_root / "A_mappo_smoke_seed_052" / "status.json")
    if mappo_status["policy_source"] != "mappo_policy":
        raise RuntimeError("mappo_smoke must write policy_source=mappo_policy")
    if mappo_status["checkpoint_loaded"] is not True:
        raise RuntimeError("mappo_smoke must load checkpoint")
    if mappo_status["checkpoint_validator_ran"] is not True:
        raise RuntimeError("mappo_smoke must run checkpoint validator")
    if mappo_status["mock_action_used"] is not False:
        raise RuntimeError("mappo_smoke must set mock_action_used=false")
    if mappo_status["placeholder_fallback_used"] is not False:
        raise RuntimeError("mappo_smoke must set placeholder_fallback_used=false")
    if mappo_status["actual_policy_claim_ready"] is not False:
        raise RuntimeError("mappo_smoke must not be actual claim ready")

    mappo_rollup = run_root / "A_mappo_smoke_seed_052" / "window_rollup.parquet"
    if not mappo_rollup.exists():
        raise RuntimeError("mappo_smoke window_rollup.parquet was not written")

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--condition-id",
            "A90",
            "--policy-source-mode",
            "mappo_smoke",
            "--checkpoint-path",
            str(smoke_ckpt),
            "--checkpoint-validation-mode",
            "smoke",
            "--seed",
            "52",
            "--device",
            "cpu",
            "--output-root",
            str(run_root),
        ],
        expect_success=True,
    )

    a90_status = read_json(run_root / "A90_mappo_smoke_seed_052" / "status.json")
    if a90_status["condition_id"] != "A90":
        raise RuntimeError("A90 status condition_id mismatch")
    if not str(a90_status["source_mode"]).startswith("noncausal_A90_mappo_policy_smoke"):
        raise RuntimeError("A90 source_mode mismatch")

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mock_smoke",
            "--seed",
            "52",
            "--output-root",
            str(run_root),
        ],
        expect_success=False,
    )

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mappo_smoke",
            "--checkpoint-validation-mode",
            "smoke",
            "--seed",
            "52",
            "--output-root",
            str(run_root),
        ],
        expect_success=False,
    )

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mappo_actual",
            "--checkpoint-path",
            str(smoke_ckpt),
            "--checkpoint-validation-mode",
            "actual",
            "--seed",
            "52",
            "--device",
            "cpu",
            "--output-root",
            str(run_root),
        ],
        expect_success=False,
    )

    run_cmd(
        [
            sys.executable,
            str(runner),
            "--condition-id",
            "A",
            "--policy-source-mode",
            "mappo_smoke",
            "--checkpoint-path",
            str(qwen_ckpt),
            "--checkpoint-validation-mode",
            "smoke",
            "--seed",
            "52",
            "--device",
            "cpu",
            "--output-root",
            str(run_root),
        ],
        expect_success=False,
    )

    report = {
        "status": "PASS",
        "step": 52,
        "runner": str(runner),
        "run_root": str(run_root),
        "smoke_checkpoint": str(smoke_ckpt),
        "qwen_invalid_checkpoint": str(qwen_ckpt),
        "validated_paths": {
            "mock_status": str(run_root / "A_mock_smoke_seed_052" / "status.json"),
            "mappo_status": str(run_root / "A_mappo_smoke_seed_052" / "status.json"),
            "a90_status": str(run_root / "A90_mappo_smoke_seed_052" / "status.json"),
            "mappo_rollup": str(mappo_rollup),
        },
        "note": (
            "Step 52 validates policy-source selectable A-family rollout smoke writer. "
            "Smoke artifacts are not for performance claims."
        ),
    }

    out_report = out_dir / "step52_a_family_policy_rollout_smoke_selftest_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Step 52 A-family policy rollout smoke self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
