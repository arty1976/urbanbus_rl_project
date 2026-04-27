from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ARTIFACT_VERSION = "mappo_policy_checkpoint_v1"
CONTRACT_VERSION = "mappo_checkpoint_contract_v1"

DEFAULT_POLICY_ARCHITECTURE = "ActorCriticMLP"
DEFAULT_ACTOR_OBS_DIM = 16
DEFAULT_CRITIC_OBS_DIM = 64
DEFAULT_ACTION_DIM = 2
DEFAULT_HIDDEN_DIM = 128

REWARD_VERSION = "mappo_reward_v1"
ROLLOUT_SCHEMA_VERSION = "rollout_schema_v1"
POLICY_INTERFACE_VERSION = "mappo_policy_interface_v1"
ACTION_SPACE_VERSION = "bus_control_action_v1"
OBSERVATION_SPACE_VERSION = "urbanbus_observation_v1"

ENERGY_PROXY_MODEL_VERSION = "daegu_energy_proxy_v1"
ENERGY_PROXY_UNIT = "kwh_equivalent"
K_DIST_KWH_PER_M = 0.0012
K_ACC_KWH_PER_EVENT = 0.1800
K_IDLE_KWH_PER_SEC = 0.0080


class MAPPOCheckpointBuilderError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_git_commit(project_root: Optional[Path] = None) -> str:
    root = Path(project_root or Path.cwd())
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        value = proc.stdout.strip()
        return value if value else "UNKNOWN_GIT_COMMIT"
    except Exception:
        return "UNKNOWN_GIT_COMMIT"


def checkpoint_contract_metadata(
    *,
    condition_id: str = "A",
    qwen_train: bool = False,
    qwen_inference: bool = False,
    qwen_trigger_rate: float = 0.0,
    policy_architecture: str = DEFAULT_POLICY_ARCHITECTURE,
    actor_obs_dim: int = DEFAULT_ACTOR_OBS_DIM,
    critic_obs_dim: int = DEFAULT_CRITIC_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    shared_policy: bool = True,
    ctde_enabled: bool = True,
    training_seed: int = 0,
    git_commit: str = "",
    created_at_utc: str = "",
    trained_model: bool = False,
    performance_claim_allowed: bool = False,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build metadata fields required by mappo_checkpoint_contract_v1.

    This does not include model_state_dict.
    """

    condition_id = str(condition_id).strip().upper()

    if condition_id != "A":
        raise MAPPOCheckpointBuilderError(
            f"Step 45 builder currently supports Experiment A only, got {condition_id}"
        )

    if bool(qwen_train) or bool(qwen_inference) or float(qwen_trigger_rate) != 0.0:
        raise MAPPOCheckpointBuilderError(
            "Experiment A checkpoint metadata must keep Qwen disabled and qwen_trigger_rate=0.0"
        )

    if bool(performance_claim_allowed) and not bool(trained_model):
        raise MAPPOCheckpointBuilderError(
            "performance_claim_allowed=true requires trained_model=true"
        )

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "condition_id": condition_id,
        "qwen_train": bool(qwen_train),
        "qwen_inference": bool(qwen_inference),
        "qwen_trigger_rate": float(qwen_trigger_rate),
        "policy_architecture": str(policy_architecture),
        "actor_obs_dim": int(actor_obs_dim),
        "critic_obs_dim": int(critic_obs_dim),
        "action_dim": int(action_dim),
        "hidden_dim": int(hidden_dim),
        "shared_policy": bool(shared_policy),
        "ctde_enabled": bool(ctde_enabled),
        "training_seed": int(training_seed),
        "git_commit": str(git_commit or "UNKNOWN_GIT_COMMIT"),
        "created_at_utc": str(created_at_utc or utc_now_iso()),
        "trained_model": bool(trained_model),
        "performance_claim_allowed": bool(performance_claim_allowed),
        "reward_version": REWARD_VERSION,
        "rollout_schema_version": ROLLOUT_SCHEMA_VERSION,
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "observation_space_version": OBSERVATION_SPACE_VERSION,
        "energy_proxy_model_version": ENERGY_PROXY_MODEL_VERSION,
        "energy_proxy_unit": ENERGY_PROXY_UNIT,
        "k_dist_kwh_per_m": K_DIST_KWH_PER_M,
        "k_acc_kwh_per_event": K_ACC_KWH_PER_EVENT,
        "k_idle_kwh_per_sec": K_IDLE_KWH_PER_SEC,
    }

    if extra_metadata:
        safe_extra = dict(extra_metadata)
        forbidden = set(metadata.keys()) | {"model_state_dict"}
        overlap = sorted(set(safe_extra.keys()) & forbidden)
        if overlap:
            raise MAPPOCheckpointBuilderError(
                f"extra_metadata must not override contract keys: {overlap}"
            )
        metadata["extra_metadata"] = safe_extra

    return metadata


def build_mappo_checkpoint_payload(
    *,
    model_state_dict: Mapping[str, Any],
    condition_id: str = "A",
    qwen_train: bool = False,
    qwen_inference: bool = False,
    qwen_trigger_rate: float = 0.0,
    policy_architecture: str = DEFAULT_POLICY_ARCHITECTURE,
    actor_obs_dim: int = DEFAULT_ACTOR_OBS_DIM,
    critic_obs_dim: int = DEFAULT_CRITIC_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    shared_policy: bool = True,
    ctde_enabled: bool = True,
    training_seed: int = 0,
    git_commit: str = "",
    created_at_utc: str = "",
    trained_model: bool = False,
    performance_claim_allowed: bool = False,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a complete torch-saveable checkpoint payload.

    Actual H200 training should call this with:
    - trained_model=True
    - performance_claim_allowed=True only after official training/evaluation readiness
    """

    if not isinstance(model_state_dict, Mapping):
        raise MAPPOCheckpointBuilderError("model_state_dict must be a mapping/dict")

    metadata = checkpoint_contract_metadata(
        condition_id=condition_id,
        qwen_train=qwen_train,
        qwen_inference=qwen_inference,
        qwen_trigger_rate=qwen_trigger_rate,
        policy_architecture=policy_architecture,
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        shared_policy=shared_policy,
        ctde_enabled=ctde_enabled,
        training_seed=training_seed,
        git_commit=git_commit,
        created_at_utc=created_at_utc,
        trained_model=trained_model,
        performance_claim_allowed=performance_claim_allowed,
        extra_metadata=extra_metadata,
    )

    payload = dict(metadata)
    payload["model_state_dict"] = dict(model_state_dict)
    return payload


def save_mappo_checkpoint(
    path: Path,
    *,
    model_state_dict: Mapping[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Save checkpoint payload with torch.save and return the payload.
    """
    try:
        import torch
    except Exception as exc:
        raise MAPPOCheckpointBuilderError(f"torch import failed: {exc}") from exc

    payload = build_mappo_checkpoint_payload(
        model_state_dict=model_state_dict,
        **kwargs,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def write_checkpoint_contract_preview_json(
    path: Path,
    *,
    project_root: Optional[Path] = None,
    seed: int = 0,
    trained_model: bool = False,
    performance_claim_allowed: bool = False,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write metadata-only preview for runner smoke outputs.

    This is not a valid actual checkpoint because it intentionally has no
    model_state_dict. It exists so every runner seed directory records the
    exact future checkpoint contract that actual training must satisfy.
    """

    metadata = checkpoint_contract_metadata(
        training_seed=int(seed),
        git_commit=resolve_git_commit(project_root),
        trained_model=bool(trained_model),
        performance_claim_allowed=bool(performance_claim_allowed),
        extra_metadata=extra_metadata,
    )

    payload = {
        "artifact_version": "mappo_checkpoint_contract_preview_v1",
        "is_torch_checkpoint": False,
        "has_model_state_dict": False,
        "actual_checkpoint_allowed": False,
        "note": (
            "Metadata-only checkpoint contract preview. "
            "Actual MAPPO checkpoints must include model_state_dict and pass validate_mappo_checkpoint.py."
        ),
        "required_checkpoint_metadata": metadata,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload
