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

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "checkpoint_preflight_selftest"
    ckpt_dir = out_dir / "checkpoints"
    smoke_root = out_dir / "smoke_preflight_pass"
    actual_fail_root = out_dir / "actual_preflight_expected_fail"
    qwen_fail_root = out_dir / "qwen_preflight_expected_fail"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    smoke_ckpt = ckpt_dir / "step46_smoke_checkpoint.pt"
    save_mappo_checkpoint(
        smoke_ckpt,
        model_state_dict=model.state_dict(),
        training_seed=46,
        git_commit="STEP46_SELFTEST_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "not_for_performance_claims": True,
        },
    )

    # Build a Qwen-invalid checkpoint by editing the saved smoke payload.
    qwen_ckpt = ckpt_dir / "step46_invalid_qwen_checkpoint.pt"
    payload = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    payload["qwen_train"] = True
    torch.save(payload, qwen_ckpt)

    preflight = training_dir / "policies" / "preflight_mappo_checkpoint.py"

    run_cmd(
        [
            sys.executable,
            str(preflight),
            "--checkpoint",
            str(smoke_ckpt),
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--seed",
            "46",
            "--output-root",
            str(smoke_root),
        ],
        expect_success=True,
    )

    smoke_manifest = smoke_root / "smoke_seed_046" / "preflight_manifest.json"
    if not smoke_manifest.exists():
        raise RuntimeError(f"smoke preflight manifest not found: {smoke_manifest}")

    smoke_payload = read_json(smoke_manifest)
    if not smoke_payload.get("valid", False):
        raise RuntimeError("smoke preflight manifest must have valid=true")
    if smoke_payload.get("actual_performance_claim_allowed", True):
        raise RuntimeError("smoke preflight must not allow actual performance claims")

    run_cmd(
        [
            sys.executable,
            str(preflight),
            "--checkpoint",
            str(smoke_ckpt),
            "--mode",
            "actual",
            "--device",
            "cpu",
            "--seed",
            "46",
            "--output-root",
            str(actual_fail_root),
        ],
        expect_success=False,
    )

    actual_manifest = actual_fail_root / "actual_seed_046" / "preflight_manifest.json"
    if not actual_manifest.exists():
        raise RuntimeError(f"actual expected-fail manifest not found: {actual_manifest}")

    actual_payload = read_json(actual_manifest)
    if actual_payload.get("valid", True):
        raise RuntimeError("actual preflight for smoke checkpoint must fail")

    run_cmd(
        [
            sys.executable,
            str(preflight),
            "--checkpoint",
            str(qwen_ckpt),
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--seed",
            "46",
            "--output-root",
            str(qwen_fail_root),
        ],
        expect_success=False,
    )

    qwen_manifest = qwen_fail_root / "smoke_seed_046" / "preflight_manifest.json"
    if not qwen_manifest.exists():
        raise RuntimeError(f"qwen expected-fail manifest not found: {qwen_manifest}")

    qwen_payload = read_json(qwen_manifest)
    if qwen_payload.get("valid", True):
        raise RuntimeError("qwen-invalid checkpoint preflight must fail")

    report = {
        "status": "PASS",
        "step": 46,
        "preflight_script": str(preflight),
        "smoke_checkpoint": str(smoke_ckpt),
        "qwen_invalid_checkpoint": str(qwen_ckpt),
        "smoke_manifest": str(smoke_manifest),
        "actual_expected_fail_manifest": str(actual_manifest),
        "qwen_expected_fail_manifest": str(qwen_manifest),
        "note": (
            "Step 46 self-test verifies smoke-mode preflight pass and expected failures. "
            "Actual-mode PASS is intentionally not tested with synthetic checkpoints; "
            "it requires a real H200 trained checkpoint."
        ),
    }

    out_report = out_dir / "step46_preflight_selftest_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Step 46 checkpoint preflight self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
