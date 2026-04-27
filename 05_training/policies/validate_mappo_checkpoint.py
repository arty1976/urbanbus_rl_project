from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXPECTED = {
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
}

REQUIRED_KEYS = [
    "artifact_version",
    "contract_version",
    "condition_id",
    "qwen_train",
    "qwen_inference",
    "qwen_trigger_rate",
    "policy_architecture",
    "actor_obs_dim",
    "critic_obs_dim",
    "action_dim",
    "hidden_dim",
    "shared_policy",
    "ctde_enabled",
    "model_state_dict",
    "training_seed",
    "git_commit",
    "created_at_utc",
    "trained_model",
    "performance_claim_allowed",
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
]

FORBIDDEN_ACTUAL_MARKER_WORDS = [
    "fake",
    "mock",
    "stub",
    "smoke",
    "placeholder",
    "scaffold",
    "boundary_test",
]


class CheckpointValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    checkpoint_path: str
    mode: str
    valid: bool
    checkpoint_loaded: bool
    state_dict_load_ok: bool
    trained_model: bool
    performance_claim_allowed: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


def _append_training_dir_to_path() -> None:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))


def _load_torch():
    try:
        import torch
    except Exception as exc:
        raise CheckpointValidationError(f"torch import failed: {exc}") from exc
    return torch


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _is_close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    except Exception:
        return False


def _strip_common_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in ("module.", "policy.", "actor_critic."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = value
    return out


def _require_keys(checkpoint: Dict[str, Any]) -> List[str]:
    return [key for key in REQUIRED_KEYS if key not in checkpoint]


def _validate_exact_values(
    checkpoint: Dict[str, Any],
    expected_actor_obs_dim: int,
    expected_critic_obs_dim: int,
    expected_action_dim: int,
    expected_hidden_dim: int,
) -> List[str]:
    errors: List[str] = []
    expected = dict(EXPECTED)
    expected["actor_obs_dim"] = int(expected_actor_obs_dim)
    expected["critic_obs_dim"] = int(expected_critic_obs_dim)
    expected["action_dim"] = int(expected_action_dim)
    expected["hidden_dim"] = int(expected_hidden_dim)

    for key, expected_value in expected.items():
        if key not in checkpoint:
            continue

        actual_value = checkpoint.get(key)

        if isinstance(expected_value, float):
            if not _is_close(actual_value, expected_value):
                errors.append(
                    f"{key} mismatch: expected {expected_value}, got {actual_value}"
                )
        else:
            if actual_value != expected_value:
                errors.append(
                    f"{key} mismatch: expected {expected_value!r}, got {actual_value!r}"
                )

    return errors


def _validate_basic_types(checkpoint: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if "model_state_dict" in checkpoint and not isinstance(checkpoint["model_state_dict"], dict):
        errors.append("model_state_dict must be a dict")

    for key in ("training_seed", "actor_obs_dim", "critic_obs_dim", "action_dim", "hidden_dim"):
        if key in checkpoint and not isinstance(checkpoint[key], int):
            errors.append(f"{key} must be int")

    for key in ("git_commit", "created_at_utc"):
        if key in checkpoint:
            value = checkpoint[key]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{key} must be a non-empty string")

    for key in (
        "qwen_train",
        "qwen_inference",
        "shared_policy",
        "ctde_enabled",
        "trained_model",
        "performance_claim_allowed",
    ):
        if key in checkpoint and not isinstance(checkpoint[key], bool):
            errors.append(f"{key} must be bool")

    return errors


def _json_safe_marker_text(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()

    if isinstance(value, (int, float, bool)) or value is None:
        return str(value).lower()

    if isinstance(value, (list, tuple)):
        return " ".join(_json_safe_marker_text(x) for x in value).lower()

    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if str(k) == "model_state_dict":
                continue
            if hasattr(v, "shape"):
                continue
            parts.append(str(k))
            parts.append(_json_safe_marker_text(v))
        return " ".join(parts).lower()

    if hasattr(value, "shape"):
        return ""

    return str(value).lower()


def _actual_mode_marker_errors(checkpoint: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    for key, value in checkpoint.items():
        if str(key) == "model_state_dict":
            continue

        text = _json_safe_marker_text(value)
        for word in FORBIDDEN_ACTUAL_MARKER_WORDS:
            if word in text:
                errors.append(
                    f"actual mode forbids marker word {word!r} found in key {key!r}"
                )

    return errors


def _load_state_dict_into_model(
    checkpoint: Dict[str, Any],
    device: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    errors: List[str] = []
    summary: Dict[str, Any] = {
        "missing_keys": [],
        "unexpected_keys": [],
        "parameter_tensor_count": None,
        "parameter_scalar_count": None,
    }

    _append_training_dir_to_path()

    try:
        import torch
        from policies.mappo_neural_inference_adapter import ActorCriticMLP
    except Exception as exc:
        return False, summary, [f"failed to import neural MAPPO model: {exc}"]

    try:
        requested = str(device or "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"

        model = ActorCriticMLP(
            actor_obs_dim=int(checkpoint["actor_obs_dim"]),
            critic_obs_dim=int(checkpoint["critic_obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(requested)

        raw_state_dict = checkpoint["model_state_dict"]
        state_dict = _strip_common_prefixes(raw_state_dict)

        load_result = model.load_state_dict(state_dict, strict=False)

        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)

        summary["missing_keys"] = missing
        summary["unexpected_keys"] = unexpected
        summary["parameter_tensor_count"] = int(len(state_dict))
        summary["parameter_scalar_count"] = int(
            sum(int(v.numel()) for v in state_dict.values() if hasattr(v, "numel"))
        )

        if missing:
            errors.append(f"state_dict missing keys: {missing}")
        if unexpected:
            errors.append(f"state_dict unexpected keys: {unexpected}")

        return len(errors) == 0, summary, errors
    except Exception as exc:
        return False, summary, [f"state_dict load failed: {exc}"]


def validate_checkpoint_dict(
    checkpoint: Dict[str, Any],
    *,
    checkpoint_path: str,
    mode: str,
    device: str = "cpu",
    expected_actor_obs_dim: int = 16,
    expected_critic_obs_dim: int = 64,
    expected_action_dim: int = 2,
    expected_hidden_dim: int = 128,
) -> ValidationReport:
    mode = str(mode).strip().lower()
    if mode not in {"actual", "smoke"}:
        raise ValueError(f"mode must be actual or smoke, got {mode}")

    errors: List[str] = []
    warnings: List[str] = []

    missing = _require_keys(checkpoint)
    if missing:
        errors.append(f"missing required keys: {missing}")

    errors.extend(_validate_basic_types(checkpoint))
    errors.extend(
        _validate_exact_values(
            checkpoint,
            expected_actor_obs_dim=expected_actor_obs_dim,
            expected_critic_obs_dim=expected_critic_obs_dim,
            expected_action_dim=expected_action_dim,
            expected_hidden_dim=expected_hidden_dim,
        )
    )

    trained_model = bool(checkpoint.get("trained_model", False))
    performance_claim_allowed = bool(checkpoint.get("performance_claim_allowed", False))

    if mode == "actual":
        if not trained_model:
            errors.append("actual mode requires trained_model=true")
        if not performance_claim_allowed:
            errors.append("actual mode requires performance_claim_allowed=true")
        errors.extend(_actual_mode_marker_errors(checkpoint))
    else:
        if performance_claim_allowed:
            errors.append("smoke mode requires performance_claim_allowed=false")
        if not trained_model:
            warnings.append(
                "trained_model=false: valid only for boundary/smoke validation, not performance claims"
            )

    state_dict_load_ok = False
    state_dict_summary: Dict[str, Any] = {}

    if "model_state_dict" in checkpoint and isinstance(checkpoint.get("model_state_dict"), dict):
        state_dict_load_ok, state_dict_summary, state_dict_errors = _load_state_dict_into_model(
            checkpoint,
            device=device,
        )
        errors.extend(state_dict_errors)
    else:
        state_dict_summary = {
            "missing_keys": [],
            "unexpected_keys": [],
            "parameter_tensor_count": None,
            "parameter_scalar_count": None,
        }

    summary = {
        "artifact_version": checkpoint.get("artifact_version"),
        "contract_version": checkpoint.get("contract_version"),
        "condition_id": checkpoint.get("condition_id"),
        "qwen_train": checkpoint.get("qwen_train"),
        "qwen_inference": checkpoint.get("qwen_inference"),
        "qwen_trigger_rate": checkpoint.get("qwen_trigger_rate"),
        "policy_architecture": checkpoint.get("policy_architecture"),
        "actor_obs_dim": checkpoint.get("actor_obs_dim"),
        "critic_obs_dim": checkpoint.get("critic_obs_dim"),
        "action_dim": checkpoint.get("action_dim"),
        "hidden_dim": checkpoint.get("hidden_dim"),
        "shared_policy": checkpoint.get("shared_policy"),
        "ctde_enabled": checkpoint.get("ctde_enabled"),
        "reward_version": checkpoint.get("reward_version"),
        "energy_proxy_model_version": checkpoint.get("energy_proxy_model_version"),
        "energy_proxy_constants": {
            "k_dist_kwh_per_m": checkpoint.get("k_dist_kwh_per_m"),
            "k_acc_kwh_per_event": checkpoint.get("k_acc_kwh_per_event"),
            "k_idle_kwh_per_sec": checkpoint.get("k_idle_kwh_per_sec"),
        },
        "state_dict_summary": state_dict_summary,
    }

    valid = len(errors) == 0

    return ValidationReport(
        checkpoint_path=str(checkpoint_path),
        mode=mode,
        valid=valid,
        checkpoint_loaded=True,
        state_dict_load_ok=bool(state_dict_load_ok),
        trained_model=trained_model,
        performance_claim_allowed=performance_claim_allowed,
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def validate_checkpoint_path(
    path: Path,
    *,
    mode: str,
    device: str = "cpu",
    expected_actor_obs_dim: int = 16,
    expected_critic_obs_dim: int = 64,
    expected_action_dim: int = 2,
    expected_hidden_dim: int = 128,
) -> ValidationReport:
    path = Path(path)
    if not path.exists():
        raise CheckpointValidationError(f"checkpoint not found: {path}")

    torch = _load_torch()

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise CheckpointValidationError(f"torch.load failed: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise CheckpointValidationError(
            f"checkpoint must be dict, got {type(checkpoint).__name__}"
        )

    return validate_checkpoint_dict(
        checkpoint,
        checkpoint_path=str(path),
        mode=mode,
        device=device,
        expected_actor_obs_dim=expected_actor_obs_dim,
        expected_critic_obs_dim=expected_critic_obs_dim,
        expected_action_dim=expected_action_dim,
        expected_hidden_dim=expected_hidden_dim,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MAPPO checkpoint against mappo_checkpoint_contract_v1"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["actual", "smoke"], default="actual")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-actor-obs-dim", type=int, default=16)
    parser.add_argument("--expected-critic-obs-dim", type=int, default=64)
    parser.add_argument("--expected-action-dim", type=int, default=2)
    parser.add_argument("--expected-hidden-dim", type=int, default=128)
    parser.add_argument("--json-output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        report = validate_checkpoint_path(
            Path(args.checkpoint),
            mode=args.mode,
            device=args.device,
            expected_actor_obs_dim=args.expected_actor_obs_dim,
            expected_critic_obs_dim=args.expected_critic_obs_dim,
            expected_action_dim=args.expected_action_dim,
            expected_hidden_dim=args.expected_hidden_dim,
        )
    except Exception as exc:
        payload = {
            "checkpoint_path": str(args.checkpoint),
            "mode": str(args.mode),
            "valid": False,
            "checkpoint_loaded": False,
            "state_dict_load_ok": False,
            "trained_model": False,
            "performance_claim_allowed": False,
            "errors": [str(exc)],
            "warnings": [],
            "summary": {},
        }

        if args.json_output:
            _dump_json(Path(args.json_output), payload)

        print("[FAIL] checkpoint validation failed before full report")
        print(f"[FAIL] {exc}")
        return 1

    payload = asdict(report)

    if args.json_output:
        _dump_json(Path(args.json_output), payload)

    if report.valid:
        print("[OK] MAPPO checkpoint contract validation PASS")
        print(f"[OK] mode                    : {report.mode}")
        print(f"[OK] checkpoint              : {report.checkpoint_path}")
        print(f"[OK] state_dict_load_ok       : {report.state_dict_load_ok}")
        print(f"[OK] trained_model            : {report.trained_model}")
        print(f"[OK] performance_claim_allowed: {report.performance_claim_allowed}")
        if report.warnings:
            for warning in report.warnings:
                print(f"[WARN] {warning}")
        return 0

    print("[FAIL] MAPPO checkpoint contract validation FAIL")
    print(f"[FAIL] mode      : {report.mode}")
    print(f"[FAIL] checkpoint: {report.checkpoint_path}")
    for err in report.errors:
        print(f"[FAIL] {err}")
    for warning in report.warnings:
        print(f"[WARN] {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
