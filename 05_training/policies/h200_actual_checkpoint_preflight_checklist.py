from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CHECKLIST_VERSION = "h200_actual_checkpoint_preflight_v1"

EXPECTED_CONSTANTS = {
    "k_dist_kwh_per_m": 0.0012,
    "k_acc_kwh_per_event": 0.1800,
    "k_idle_kwh_per_sec": 0.0080,
}

EXPECTED_METADATA = {
    "condition_id": "A",
    "qwen_train": False,
    "qwen_inference": False,
    "qwen_trigger_rate": 0.0,
    "reward_version": "mappo_reward_v1",
    "rollout_schema_version": "rollout_schema_v1",
    "policy_interface_version": "mappo_policy_interface_v1",
    "action_space_version": "bus_control_action_v1",
    "observation_space_version": "urbanbus_observation_v1",
    "energy_proxy_model_version": "daegu_energy_proxy_v1",
    "energy_proxy_unit": "kwh_equivalent",
    "trained_model": True,
    "performance_claim_allowed": True,
}

ACTUAL_ACCEPTANCE_RULES = [
    "checkpoint file exists",
    "validate_mappo_checkpoint.py --mode actual PASS",
    "trained_model=true",
    "performance_claim_allowed=true",
    "qwen_train=false",
    "qwen_inference=false",
    "qwen_trigger_rate=0.0",
    "reward_version=mappo_reward_v1",
    "energy_proxy_model_version=daegu_energy_proxy_v1",
    "K_DIST/K_ACC/K_IDLE constants match",
    "model_state_dict strict-loads into NeuralMAPPOInferenceAdapter",
]


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
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def float_equal(a: Any, b: Any, tol: float = 1.0e-12) -> bool:
    return abs(float_value(a) - float_value(b)) <= tol


def find_validator_script(training_dir: Path) -> Path:
    candidates = [
        training_dir / "policies" / "validate_mappo_checkpoint.py",
        training_dir / "policies" / "preflight_mappo_checkpoint.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "checkpoint validator script not found. Expected one of: "
        + ", ".join(str(x) for x in candidates)
    )


def inspect_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    append_training_dir_to_path()

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {
            "loaded": True,
            "payload_type": str(type(payload)),
            "metadata": {},
            "error": "checkpoint payload is not a dict",
        }

    metadata: Dict[str, Any] = {}
    for key in [
        "artifact_version",
        "condition_id",
        "qwen_train",
        "qwen_inference",
        "qwen_trigger_rate",
        "reward_version",
        "rollout_schema_version",
        "policy_interface_version",
        "action_space_version",
        "observation_space_version",
        "energy_proxy_model_version",
        "energy_proxy_unit",
        "k_dist_kwh_per_m",
        "k_acc_kwh_per_event",
        "k_idle_kwh_per_sec",
        "trained_model",
        "performance_claim_allowed",
        "actor_obs_dim",
        "critic_obs_dim",
        "action_dim",
        "hidden_dim",
        "shared_policy",
        "ctde_enabled",
        "training_seed",
        "git_commit",
        "created_at_utc",
    ]:
        if key in payload:
            metadata[key] = payload.get(key)

    metadata["has_model_state_dict"] = "model_state_dict" in payload

    return {
        "loaded": True,
        "payload_type": "dict",
        "metadata": metadata,
    }


def validate_metadata_snapshot(metadata: Dict[str, Any]) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []

    for key, expected in EXPECTED_METADATA.items():
        actual = metadata.get(key)

        if isinstance(expected, bool):
            if bool_value(actual) is not expected:
                blockers.append(f"{key} mismatch: expected {expected}, got {actual}")

        elif isinstance(expected, float):
            if not float_equal(actual, expected):
                blockers.append(f"{key} mismatch: expected {expected}, got {actual}")

        else:
            if actual != expected:
                blockers.append(f"{key} mismatch: expected {expected!r}, got {actual!r}")

    for key, expected in EXPECTED_CONSTANTS.items():
        actual = metadata.get(key)
        if not float_equal(actual, expected):
            blockers.append(f"{key} mismatch: expected {expected}, got {actual}")

    if bool_value(metadata.get("has_model_state_dict", False)) is not True:
        blockers.append("model_state_dict missing")

    if bool_value(metadata.get("shared_policy", True)) is not True:
        blockers.append("shared_policy must be true")

    if bool_value(metadata.get("ctde_enabled", True)) is not True:
        blockers.append("ctde_enabled must be true")

    for key in ["actor_obs_dim", "critic_obs_dim", "action_dim", "hidden_dim"]:
        if key not in metadata:
            blockers.append(f"{key} missing")
        elif int(metadata.get(key, 0)) <= 0:
            blockers.append(f"{key} must be positive")

    if not metadata.get("git_commit"):
        warnings.append("git_commit missing or empty")

    if not metadata.get("created_at_utc"):
        warnings.append("created_at_utc missing or empty")

    return {
        "metadata_valid_for_actual": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings,
    }


def run_checkpoint_validator(
    *,
    validator_script: Path,
    checkpoint_path: Path,
    device: str,
    report_path: Path,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(validator_script),
        "--checkpoint",
        str(checkpoint_path),
        "--mode",
        "actual",
        "--device",
        str(device),
        "--json-output",
        str(report_path),
    ]

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    report_payload: Optional[Dict[str, Any]] = None
    if report_path.exists():
        try:
            report_payload = read_json(report_path)
        except Exception as exc:
            report_payload = {"read_error": str(exc)}

    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "report_path": str(report_path),
        "report": report_payload,
        "passed": proc.returncode == 0,
    }


def evaluate_h200_actual_preflight(
    *,
    checkpoint_path: str,
    device: str = "cpu",
    run_validator: bool = False,
    output_dir: Optional[str] = None,
    simulator_adapter: str = "adapters.historical_replay_adapter.HistoricalReplayAdapter",
    require_causal_claim: bool = False,
) -> Dict[str, Any]:
    training_dir = append_training_dir_to_path()
    ckpt = Path(checkpoint_path) if checkpoint_path else Path("")

    out_dir = (
        Path(output_dir)
        if output_dir
        else training_dir.parent / "artifacts" / "experiment_A_v1" / "h200_actual_checkpoint_preflight"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    blockers: List[str] = []
    warnings: List[str] = []

    payload: Dict[str, Any] = {
        "artifact_version": CHECKLIST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt) if checkpoint_path else "",
        "device": str(device),
        "run_validator": bool(run_validator),
        "simulator_adapter": str(simulator_adapter),
        "require_causal_claim": bool(require_causal_claim),
        "acceptance_rules": ACTUAL_ACCEPTANCE_RULES,
        "expected_metadata": EXPECTED_METADATA,
        "expected_constants": EXPECTED_CONSTANTS,
    }

    if not checkpoint_path:
        blockers.append("checkpoint_path is required")
        payload["checkpoint_exists"] = False
    elif not ckpt.exists():
        blockers.append(f"checkpoint file does not exist: {ckpt}")
        payload["checkpoint_exists"] = False
    else:
        payload["checkpoint_exists"] = True
        try:
            inspection = inspect_checkpoint_metadata(ckpt)
            payload["checkpoint_inspection"] = inspection
            metadata = inspection.get("metadata", {})
            metadata_validation = validate_metadata_snapshot(metadata)
            payload["metadata_validation"] = metadata_validation
            blockers.extend(metadata_validation["blockers"])
            warnings.extend(metadata_validation["warnings"])
        except Exception as exc:
            blockers.append(f"checkpoint metadata inspection failed: {exc}")
            payload["checkpoint_inspection_error"] = str(exc)

    try:
        validator_script = find_validator_script(training_dir)
        payload["validator_script"] = str(validator_script)
    except Exception as exc:
        validator_script = None
        blockers.append(str(exc))
        payload["validator_script_error"] = str(exc)

    if run_validator and payload.get("checkpoint_exists") and validator_script is not None:
        validator_report_path = out_dir / "actual_checkpoint_validation_report.json"
        validation = run_checkpoint_validator(
            validator_script=validator_script,
            checkpoint_path=ckpt,
            device=str(device),
            report_path=validator_report_path,
        )
        payload["actual_validator"] = validation
        if not validation.get("passed", False):
            blockers.append("validate_mappo_checkpoint.py --mode actual failed")
    elif not run_validator:
        warnings.append("actual checkpoint validator was not executed; plan-only preflight")

    adapter_lower = str(simulator_adapter).lower()
    historical_or_replay = "historical" in adapter_lower or "replay" in adapter_lower
    payload["historical_or_replay_adapter"] = bool(historical_or_replay)

    if historical_or_replay:
        warnings.append(
            "simulator adapter is historical/replay; actual checkpoint may run, but causal performance claim is not allowed"
        )

    if require_causal_claim and historical_or_replay:
        blockers.append("require_causal_claim=true but simulator adapter is historical/replay")

    actual_checkpoint_ready = (
        len(blockers) == 0
        and payload.get("checkpoint_exists") is True
        and (not run_validator or payload.get("actual_validator", {}).get("passed") is True)
    )

    causal_claim_ready = (
        bool(actual_checkpoint_ready)
        and not historical_or_replay
        and bool(require_causal_claim)
    )

    payload["actual_checkpoint_ready"] = bool(actual_checkpoint_ready)
    payload["actual_policy_claim_ready_candidate"] = bool(actual_checkpoint_ready)
    payload["causal_policy_claim_ready_candidate"] = bool(causal_claim_ready)
    payload["blockers"] = blockers
    payload["warnings"] = warnings
    payload["status"] = "PASS" if actual_checkpoint_ready else "BLOCKED"

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H200 actual MAPPO checkpoint preflight checklist"
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-validator", action="store_true")
    parser.add_argument(
        "--simulator-adapter",
        default="adapters.historical_replay_adapter.HistoricalReplayAdapter",
    )
    parser.add_argument("--require-causal-claim", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    payload = evaluate_h200_actual_preflight(
        checkpoint_path=str(args.checkpoint_path),
        device=str(args.device),
        run_validator=bool(args.run_validator),
        output_dir=str(args.output_dir or ""),
        simulator_adapter=str(args.simulator_adapter),
        require_causal_claim=bool(args.require_causal_claim),
    )

    if args.json_output:
        dump_json(Path(args.json_output), payload)

    if payload["status"] == "PASS":
        print("[OK] H200 actual checkpoint preflight PASS")
        print(f"[OK] checkpoint: {payload['checkpoint_path']}")
        print(f"[OK] actual_checkpoint_ready: {payload['actual_checkpoint_ready']}")
        print(f"[OK] causal_policy_claim_ready_candidate: {payload['causal_policy_claim_ready_candidate']}")
        for warning in payload["warnings"]:
            print(f"[WARN] {warning}")
        return 0

    print("[BLOCKED] H200 actual checkpoint preflight did not pass")
    print(f"[BLOCKED] checkpoint: {payload['checkpoint_path']}")
    for blocker in payload["blockers"]:
        print(f"[BLOCKED] {blocker}")
    for warning in payload["warnings"]:
        print(f"[WARN] {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
