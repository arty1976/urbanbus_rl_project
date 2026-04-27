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
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    import torch
    from policies.mappo_checkpoint_builder import save_mappo_checkpoint
    from policies.mappo_neural_inference_adapter import ActorCriticMLP
    from policies.h200_actual_checkpoint_preflight_checklist import (
        evaluate_h200_actual_preflight,
    )

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "h200_actual_checkpoint_preflight_selftest"
    ckpt_dir = out_dir / "checkpoints"
    reports_dir = out_dir / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    checklist_path = training_dir / "policies" / "H200_actual_checkpoint_preflight_checklist.md"
    preflight_script = training_dir / "policies" / "h200_actual_checkpoint_preflight_checklist.py"

    if not checklist_path.exists():
        raise RuntimeError(f"checklist doc missing: {checklist_path}")

    text = checklist_path.read_text(encoding="utf-8-sig")
    required_phrases = [
        "trained_model = true",
        "performance_claim_allowed = true",
        "qwen_trigger_rate = 0.0",
        "energy_proxy_model_version = daegu_energy_proxy_v1",
        "HistoricalReplayAdapter remains non-causal",
    ]

    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    if missing_phrases:
        raise RuntimeError(f"checklist missing phrases: {missing_phrases}")

    missing_report = reports_dir / "missing_checkpoint_expected_blocked.json"
    run_cmd(
        [
            sys.executable,
            str(preflight_script),
            "--checkpoint-path",
            str(ckpt_dir / "does_not_exist.pt"),
            "--device",
            "cpu",
            "--json-output",
            str(missing_report),
        ],
        expect_success=False,
    )

    missing_payload = read_json(missing_report)
    if missing_payload.get("status") != "BLOCKED":
        raise RuntimeError("missing checkpoint must be BLOCKED")

    model = ActorCriticMLP(
        actor_obs_dim=16,
        critic_obs_dim=64,
        action_dim=2,
        hidden_dim=128,
    )

    smoke_ckpt = ckpt_dir / "step58_smoke_not_actual_checkpoint.pt"
    save_mappo_checkpoint(
        smoke_ckpt,
        model_state_dict=model.state_dict(),
        training_seed=58,
        git_commit="STEP58_SELFTEST_COMMIT",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        trained_model=False,
        performance_claim_allowed=False,
        extra_metadata={
            "selftest": True,
            "not_for_performance_claims": True,
        },
    )

    smoke_report = reports_dir / "smoke_checkpoint_actual_preflight_expected_blocked.json"
    run_cmd(
        [
            sys.executable,
            str(preflight_script),
            "--checkpoint-path",
            str(smoke_ckpt),
            "--device",
            "cpu",
            "--run-validator",
            "--json-output",
            str(smoke_report),
            "--output-dir",
            str(reports_dir / "validator_outputs"),
        ],
        expect_success=False,
    )

    smoke_payload = read_json(smoke_report)
    blockers = "\n".join(smoke_payload.get("blockers", []))

    if smoke_payload.get("status") != "BLOCKED":
        raise RuntimeError("smoke checkpoint must be BLOCKED in actual preflight")

    if "trained_model" not in blockers:
        raise RuntimeError("smoke checkpoint blocker must mention trained_model")

    if "performance_claim_allowed" not in blockers:
        raise RuntimeError("smoke checkpoint blocker must mention performance_claim_allowed")

    qwen_ckpt = ckpt_dir / "step58_qwen_invalid_checkpoint.pt"
    payload = torch.load(smoke_ckpt, map_location="cpu", weights_only=False)
    payload["qwen_train"] = True
    torch.save(payload, qwen_ckpt)

    qwen_report = reports_dir / "qwen_checkpoint_actual_preflight_expected_blocked.json"
    qwen_payload = evaluate_h200_actual_preflight(
        checkpoint_path=str(qwen_ckpt),
        device="cpu",
        run_validator=False,
        output_dir=str(reports_dir / "qwen_plan_only"),
        simulator_adapter="adapters.historical_replay_adapter.HistoricalReplayAdapter",
        require_causal_claim=False,
    )

    with open(qwen_report, "w", encoding="utf-8") as f:
        json.dump(qwen_payload, f, ensure_ascii=False, indent=2)

    qwen_blockers = "\n".join(qwen_payload.get("blockers", []))
    if "qwen_train" not in qwen_blockers:
        raise RuntimeError("qwen invalid checkpoint must block on qwen_train")

    causal_report = reports_dir / "historical_replay_causal_claim_expected_blocked.json"
    causal_payload = evaluate_h200_actual_preflight(
        checkpoint_path=str(smoke_ckpt),
        device="cpu",
        run_validator=False,
        output_dir=str(reports_dir / "causal_plan_only"),
        simulator_adapter="adapters.historical_replay_adapter.HistoricalReplayAdapter",
        require_causal_claim=True,
    )

    with open(causal_report, "w", encoding="utf-8") as f:
        json.dump(causal_payload, f, ensure_ascii=False, indent=2)

    causal_blockers = "\n".join(causal_payload.get("blockers", []))
    if "require_causal_claim=true" not in causal_blockers:
        raise RuntimeError("historical replay must block require_causal_claim")

    final_report = {
        "status": "PASS",
        "step": 58,
        "checklist": str(checklist_path),
        "preflight_script": str(preflight_script),
        "reports": {
            "missing_checkpoint": str(missing_report),
            "smoke_checkpoint": str(smoke_report),
            "qwen_checkpoint": str(qwen_report),
            "historical_replay_causal_claim": str(causal_report),
        },
        "note": (
            "Step 58 validates the H200 actual checkpoint preflight checklist. "
            "Smoke checkpoints are correctly blocked from actual claims."
        ),
    }

    out_report = out_dir / "step58_h200_actual_checkpoint_preflight_selftest_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print("[OK] Step 58 H200 actual checkpoint preflight checklist self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
