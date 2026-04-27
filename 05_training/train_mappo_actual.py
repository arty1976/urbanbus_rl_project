from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_CONDITION_ID = "A"
EXPECTED_REWARD_VERSION = "mappo_reward_v1"
EXPECTED_ENERGY_PROXY_MODEL_VERSION = "daegu_energy_proxy_v1"
EXPECTED_K_DIST = 0.0012
EXPECTED_K_ACC = 0.1800
EXPECTED_K_IDLE = 0.0080


class TrainingContractError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"expected boolean string, got: {value}")


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise TrainingContractError(f"failed to read JSON: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_git_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=15,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass

    return "UNKNOWN_GIT_COMMIT"


def normalize_command(argv: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in argv)


def validate_contract(args: argparse.Namespace) -> Dict[str, Any]:
    errors: List[str] = []

    condition_id = str(args.condition_id).strip().upper()
    if condition_id != EXPECTED_CONDITION_ID:
        errors.append(f"condition_id must be {EXPECTED_CONDITION_ID}, got={condition_id}")

    if bool(args.qwen_train):
        errors.append("qwen_train must be false for condition A")

    if bool(args.qwen_inference):
        errors.append("qwen_inference must be false for condition A")

    if float(args.qwen_trigger_rate) != 0.0:
        errors.append(f"qwen_trigger_rate must be 0.0, got={args.qwen_trigger_rate}")

    if str(args.reward_version).strip() != EXPECTED_REWARD_VERSION:
        errors.append(
            f"reward_version must be {EXPECTED_REWARD_VERSION}, got={args.reward_version}"
        )

    if str(args.energy_proxy_model_version).strip() != EXPECTED_ENERGY_PROXY_MODEL_VERSION:
        errors.append(
            "energy_proxy_model_version must be "
            f"{EXPECTED_ENERGY_PROXY_MODEL_VERSION}, got={args.energy_proxy_model_version}"
        )

    if abs(float(args.k_dist_kwh_per_m) - EXPECTED_K_DIST) > 1e-12:
        errors.append(f"k_dist_kwh_per_m mismatch, got={args.k_dist_kwh_per_m}")

    if abs(float(args.k_acc_kwh_per_event) - EXPECTED_K_ACC) > 1e-12:
        errors.append(f"k_acc_kwh_per_event mismatch, got={args.k_acc_kwh_per_event}")

    if abs(float(args.k_idle_kwh_per_sec) - EXPECTED_K_IDLE) > 1e-12:
        errors.append(f"k_idle_kwh_per_sec mismatch, got={args.k_idle_kwh_per_sec}")

    seeds = [int(x) for x in args.seeds]
    if sorted(seeds) != [1, 2, 3]:
        errors.append(f"seeds must be exactly 1,2,3, got={seeds}")

    experiment_contract = Path(args.experiment_contract)
    baseline_contract = Path(args.baseline_contract)

    if not experiment_contract.exists():
        errors.append(f"experiment_contract not found: {experiment_contract}")

    if not baseline_contract.exists():
        errors.append(f"baseline_contract not found: {baseline_contract}")

    experiment_payload: Dict[str, Any] = {}
    baseline_payload: Dict[str, Any] = {}

    if experiment_contract.exists():
        experiment_payload = load_json(experiment_contract)
        exp_condition = str(experiment_payload.get("condition_id", "")).strip().upper()
        if exp_condition and exp_condition != EXPECTED_CONDITION_ID:
            errors.append(f"experiment_contract condition_id must be A, got={exp_condition}")

        if bool(experiment_payload.get("qwen_train", False)):
            errors.append("experiment_contract qwen_train must be false")

        if bool(experiment_payload.get("qwen_inference", False)):
            errors.append("experiment_contract qwen_inference must be false")

    if baseline_contract.exists():
        baseline_payload = load_json(baseline_contract)
        horizon = int(baseline_payload.get("evaluation_horizon_minutes", 30))
        if horizon != 30:
            errors.append(f"baseline evaluation_horizon_minutes must be 30, got={horizon}")

        fairness = baseline_payload.get("fairness_constraints", {})
        for key in ("same_initial_state", "same_exogenous_events", "same_eval_window"):
            if key in fairness and not bool(fairness.get(key)):
                errors.append(f"baseline fairness_constraints.{key} must be true")

    if errors:
        raise TrainingContractError("; ".join(errors))

    return {
        "condition_id": condition_id,
        "condition_name": "pure_mappo_baseline",
        "seeds": seeds,
        "device": str(args.device),
        "experiment_contract": str(experiment_contract),
        "baseline_contract": str(baseline_contract),
        "qwen_train": bool(args.qwen_train),
        "qwen_inference": bool(args.qwen_inference),
        "qwen_trigger_rate": float(args.qwen_trigger_rate),
        "reward_version": str(args.reward_version),
        "energy_proxy_model_version": str(args.energy_proxy_model_version),
        "k_dist_kwh_per_m": float(args.k_dist_kwh_per_m),
        "k_acc_kwh_per_event": float(args.k_acc_kwh_per_event),
        "k_idle_kwh_per_sec": float(args.k_idle_kwh_per_sec),
        "dry_run_contract": bool(args.dry_run_contract),
        "experiment_contract_summary": {
            "condition_id": experiment_payload.get("condition_id"),
            "qwen_train": experiment_payload.get("qwen_train"),
            "qwen_inference": experiment_payload.get("qwen_inference"),
        },
        "baseline_contract_summary": {
            "evaluation_horizon_minutes": baseline_payload.get("evaluation_horizon_minutes"),
            "seeds": baseline_payload.get("seeds"),
            "time_bands": baseline_payload.get("time_bands"),
            "shared_kpis": baseline_payload.get("shared_kpis"),
        },
    }


def write_contract_scaffold(
    args: argparse.Namespace,
    contract: Dict[str, Any],
    project_root: Path,
    original_argv: List[str],
) -> Dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    git_commit = run_git_commit(project_root)

    training_config = {
        "artifact_version": "mappo_actual_training_config_v1_step69",
        "created_at_utc": utc_now(),
        **contract,
        "training_status": "dry_run_contract_only" if args.dry_run_contract else "not_started",
        "trained_model": False,
        "performance_claim_allowed": False,
        "note": (
            "Step 69 writes the CLI and output-folder contract scaffold only. "
            "It does not perform actual MAPPO training."
        ),
    }

    dump_json(output_root / "training_config.json", training_config)

    (output_root / "training_command.txt").write_text(
        normalize_command([sys.executable, *original_argv]) + "\n",
        encoding="utf-8",
    )

    (output_root / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")

    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "train_stdout.log").write_text(
        "[DRY-RUN-CONTRACT] actual MAPPO training not executed in Step 69\n",
        encoding="utf-8",
    )
    (logs_dir / "train_stderr.log").write_text("", encoding="utf-8")
    (logs_dir / "train_metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 0,
                "event": "dry_run_contract_only",
                "trained_model": False,
                "performance_claim_allowed": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manifest = {
        "artifact_version": "mappo_actual_checkpoint_manifest_v1_step69",
        "created_at_utc": utc_now(),
        "condition_id": contract["condition_id"],
        "checkpoint_status": "not_trained",
        "best_checkpoint_path": str(checkpoints_dir / "best_mappo.pt"),
        "last_checkpoint_path": str(checkpoints_dir / "last_mappo.pt"),
        "best_checkpoint_exists": False,
        "last_checkpoint_exists": False,
        "trained_model": False,
        "performance_claim_allowed": False,
        "qwen_train": contract["qwen_train"],
        "qwen_inference": contract["qwen_inference"],
        "qwen_trigger_rate": contract["qwen_trigger_rate"],
        "reward_version": contract["reward_version"],
        "energy_proxy_model_version": contract["energy_proxy_model_version"],
        "k_dist_kwh_per_m": contract["k_dist_kwh_per_m"],
        "k_acc_kwh_per_event": contract["k_acc_kwh_per_event"],
        "k_idle_kwh_per_sec": contract["k_idle_kwh_per_sec"],
        "note": "No checkpoint is written in Step 69 dry-run contract mode.",
    }
    dump_json(checkpoints_dir / "checkpoint_manifest.json", checkpoint_manifest)

    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        validation_dir / "train_mappo_actual_contract_report.json",
        {
            "artifact_version": "train_mappo_actual_contract_report_v1_step69",
            "created_at_utc": utc_now(),
            "contract_status": "PASS",
            "dry_run_contract": bool(args.dry_run_contract),
            "trained_model": False,
            "performance_claim_allowed": False,
            "ready_for_actual_training_implementation": True,
            "ready_for_step58_actual_preflight": False,
        },
    )

    (output_root / "README_run_summary.md").write_text(
        "# MAPPO actual training scaffold summary\n\n"
        "This folder was created by `05_training/train_mappo_actual.py` in "
        "`--dry-run-contract` mode.\n\n"
        "- best_mappo.pt was not created.\n"
        "- trained_model = false\n"
        "- performance_claim_allowed = false\n"
        "- qwen_train = false\n"
        "- qwen_inference = false\n"
        "- qwen_trigger_rate = 0.0\n"
        "- reward_version = mappo_reward_v1\n"
        "- energy_proxy_model_version = daegu_energy_proxy_v1\n\n"
        "This scaffold validates the command and folder contract only. "
        "It does not prove causal performance and cannot enter Step 58 actual preflight.\n",
        encoding="utf-8",
    )

    return {
        "output_root": str(output_root),
        "training_config": str(output_root / "training_config.json"),
        "training_command": str(output_root / "training_command.txt"),
        "git_commit": str(output_root / "git_commit.txt"),
        "checkpoint_manifest": str(checkpoints_dir / "checkpoint_manifest.json"),
        "contract_report": str(validation_dir / "train_mappo_actual_contract_report.json"),
        "run_summary": str(output_root / "README_run_summary.md"),
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actual MAPPO H200 training entrypoint scaffold."
    )

    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--experiment-contract", required=True)
    parser.add_argument("--baseline-contract", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", required=True)

    parser.add_argument("--qwen-train", type=str_to_bool, required=True)
    parser.add_argument("--qwen-inference", type=str_to_bool, required=True)
    parser.add_argument("--qwen-trigger-rate", type=float, required=True)

    parser.add_argument("--reward-version", required=True)
    parser.add_argument("--energy-proxy-model-version", required=True)
    parser.add_argument("--k-dist-kwh-per-m", type=float, required=True)
    parser.add_argument("--k-acc-kwh-per-event", type=float, required=True)
    parser.add_argument("--k-idle-kwh-per-sec", type=float, required=True)

    parser.add_argument(
        "--dry-run-contract",
        action="store_true",
        help="Validate CLI contract and write scaffold output without training.",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    original_argv = list(sys.argv if argv is None else [sys.argv[0], *argv])
    parse_argv = sys.argv[1:] if argv is None else argv

    try:
        args = parse_args(parse_argv)
        project_root = Path(__file__).resolve().parents[1]

        contract = validate_contract(args)

        if not args.dry_run_contract:
            raise TrainingContractError(
                "actual MAPPO training is not implemented in Step 69. "
                "Use --dry-run-contract to validate the CLI and output-folder contract."
            )

        outputs = write_contract_scaffold(
            args=args,
            contract=contract,
            project_root=project_root,
            original_argv=original_argv,
        )

        print("[OK] Step 69 train_mappo_actual.py contract validation PASS")
        print("[OK] dry_run_contract : true")
        print(f"[OK] output_root      : {outputs['output_root']}")
        print(f"[OK] training_config  : {outputs['training_config']}")
        print(f"[OK] checkpoint_manifest: {outputs['checkpoint_manifest']}")
        print("[OK] trained_model    : false")
        print("[OK] performance_claim_allowed: false")
        return 0

    except TrainingContractError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
