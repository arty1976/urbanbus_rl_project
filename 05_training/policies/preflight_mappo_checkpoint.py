from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight MAPPO checkpoint before strict neural inference"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to MAPPO checkpoint .pt")
    parser.add_argument(
        "--mode",
        choices=["actual", "smoke"],
        default="actual",
        help="actual for H200 trained checkpoint, smoke for boundary validation only",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--contract", default="", help="Optional experiment_A_contract.json path")
    parser.add_argument(
        "--simulator-adapter",
        default="adapters.historical_replay_adapter.HistoricalReplayAdapter",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--expected-actor-obs-dim", type=int, default=16)
    parser.add_argument("--expected-critic-obs-dim", type=int, default=64)
    parser.add_argument("--expected-action-dim", type=int, default=2)
    parser.add_argument("--expected-hidden-dim", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    checkpoint_path = Path(args.checkpoint)
    mode = str(args.mode).strip().lower()

    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / "artifacts" / "experiment_A_v1" / "checkpoint_preflight"
    )

    run_tag = f"{mode}_seed_{int(args.seed):03d}"
    run_dir = output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    validation_report_path = run_dir / "preflight_checkpoint_validation_report.json"
    neural_output_root = run_dir / "neural_strict_load"
    neural_runner_report = neural_output_root / f"seed_{int(args.seed):03d}" / "status.json"
    preflight_manifest_path = run_dir / "preflight_manifest.json"

    manifest: Dict[str, Any] = {
        "artifact_version": "mappo_checkpoint_preflight_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "mode": mode,
        "device": str(args.device),
        "seed": int(args.seed),
        "valid": False,
        "stage": "starting",
        "validation_report": str(validation_report_path),
        "neural_output_root": str(neural_output_root),
        "neural_runner_status": str(neural_runner_report),
        "preflight_manifest": str(preflight_manifest_path),
        "actual_performance_claim_allowed": False,
        "notes": [
            "actual mode is for H200 trained checkpoints only",
            "smoke mode is for boundary validation only",
            "historical replay smoke does not provide causal performance evidence",
        ],
    }

    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        from policies.validate_mappo_checkpoint import validate_checkpoint_path

        manifest["stage"] = "validating_checkpoint_contract"
        dump_json(preflight_manifest_path, manifest)

        report = validate_checkpoint_path(
            checkpoint_path,
            mode=mode,
            device=str(args.device),
            expected_actor_obs_dim=int(args.expected_actor_obs_dim),
            expected_critic_obs_dim=int(args.expected_critic_obs_dim),
            expected_action_dim=int(args.expected_action_dim),
            expected_hidden_dim=int(args.expected_hidden_dim),
        )

        validation_payload = asdict(report)
        dump_json(validation_report_path, validation_payload)

        if not report.valid:
            raise RuntimeError(
                f"checkpoint contract validation failed: {report.errors}"
            )

        manifest["checkpoint_contract_valid"] = True
        manifest["checkpoint_validation"] = validation_payload
        manifest["trained_model"] = bool(report.trained_model)
        manifest["performance_claim_allowed"] = bool(report.performance_claim_allowed)

        if mode == "actual":
            if not bool(report.trained_model):
                raise RuntimeError("actual preflight requires trained_model=true")
            if not bool(report.performance_claim_allowed):
                raise RuntimeError("actual preflight requires performance_claim_allowed=true")
            manifest["actual_performance_claim_allowed"] = True
        else:
            manifest["actual_performance_claim_allowed"] = False

        manifest["stage"] = "strict_neural_load_smoke"
        dump_json(preflight_manifest_path, manifest)

        neural_runner = training_dir / "run_experiment_A_neural_inference_smoke.py"
        if not neural_runner.exists():
            raise FileNotFoundError(f"neural runner not found: {neural_runner}")

        cmd = [
            sys.executable,
            str(neural_runner),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
            "--checkpoint-path",
            str(checkpoint_path),
            "--require-checkpoint",
            "--checkpoint-validation-mode",
            mode,
            "--checkpoint-validation-report",
            str(run_dir / "neural_runner_checkpoint_validation_report.json"),
            "--output-root",
            str(neural_output_root),
            "--simulator-adapter",
            str(args.simulator_adapter),
        ]

        if args.contract:
            cmd.extend(["--contract", str(args.contract)])

        proc = run_cmd(cmd)

        manifest["neural_runner_command"] = cmd
        manifest["neural_runner_returncode"] = int(proc.returncode)
        manifest["neural_runner_stdout"] = proc.stdout

        if proc.returncode != 0:
            raise RuntimeError(
                "strict neural load runner failed. "
                f"returncode={proc.returncode}. stdout={proc.stdout}"
            )

        if not neural_runner_report.exists():
            raise RuntimeError(
                f"strict neural load status was not written: {neural_runner_report}"
            )

        neural_status = read_json(neural_runner_report)

        if not bool(neural_status.get("checkpoint_loaded", False)):
            raise RuntimeError("strict neural load did not load checkpoint")

        if not bool(neural_status.get("checkpoint_validator_ran", False)):
            raise RuntimeError("strict neural load did not run checkpoint validator")

        if neural_status.get("qwen_trigger_rate") != 0.0:
            raise RuntimeError(
                f"qwen_trigger_rate must be 0.0, got {neural_status.get('qwen_trigger_rate')}"
            )

        manifest["neural_status"] = neural_status
        manifest["strict_neural_load_ok"] = True
        manifest["qwen_disabled_confirmed"] = True
        manifest["valid"] = True
        manifest["stage"] = "completed"

        dump_json(preflight_manifest_path, manifest)

        print("[OK] MAPPO checkpoint preflight PASS")
        print(f"[OK] mode                    : {mode}")
        print(f"[OK] checkpoint              : {checkpoint_path}")
        print(f"[OK] validation_report       : {validation_report_path}")
        print(f"[OK] neural_runner_status    : {neural_runner_report}")
        print(f"[OK] preflight_manifest      : {preflight_manifest_path}")
        print(f"[OK] trained_model           : {manifest.get('trained_model')}")
        print(f"[OK] performance_claim_allowed: {manifest.get('performance_claim_allowed')}")
        print(f"[OK] actual_claim_allowed    : {manifest.get('actual_performance_claim_allowed')}")
        return 0

    except Exception as exc:
        manifest["valid"] = False
        manifest["stage"] = "failed"
        manifest["error"] = str(exc)
        dump_json(preflight_manifest_path, manifest)

        print("[FAIL] MAPPO checkpoint preflight FAIL")
        print(f"[FAIL] mode               : {mode}")
        print(f"[FAIL] checkpoint         : {checkpoint_path}")
        print(f"[FAIL] preflight_manifest : {preflight_manifest_path}")
        print(f"[FAIL] error              : {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
