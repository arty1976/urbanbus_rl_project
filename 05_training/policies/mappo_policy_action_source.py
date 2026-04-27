from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class MAPPOPolicyActionSourceConfig:
    checkpoint_path: str
    validation_mode: str = "actual"
    device: str = "cpu"
    deterministic: bool = True
    require_checkpoint: bool = True
    hidden_dim: int = 128
    seed: int = 0
    validation_report_path: str = ""


@dataclass
class MAPPOPolicyActionResult:
    actions: Dict[int, int]
    metadata: Dict[str, Any]


class MAPPOPolicyActionSourceError(RuntimeError):
    pass


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _shape_last_dim(value: Any, default: int) -> int:
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            return int(len(value[0]))
        except Exception:
            return int(default)
    return int(shape[-1])


class MAPPOPolicyActionSource:
    """
    Strict neural MAPPO policy action source.

    This object is the bridge that A/A90/A80/A70 rollout writers should call
    when they want actual MAPPO neural actions instead of mock/stub actions.

    Safety rules:
    - no checkpoint -> STOP when require_checkpoint=True
    - validator fail -> STOP
    - no placeholder fallback
    - Qwen must remain disabled for Experiment A-family pure MAPPO path
    """

    def __init__(self, config: MAPPOPolicyActionSourceConfig) -> None:
        self.config = config
        self.validation_payload: Optional[Dict[str, Any]] = None
        self.adapter = None

        mode = str(config.validation_mode).strip().lower()
        if mode not in {"actual", "smoke"}:
            raise MAPPOPolicyActionSourceError(
                f"validation_mode must be actual or smoke, got {config.validation_mode}"
            )

        checkpoint_path = str(config.checkpoint_path or "").strip()
        if bool(config.require_checkpoint) and not checkpoint_path:
            raise MAPPOPolicyActionSourceError(
                "checkpoint_path is required for MAPPOPolicyActionSource; no fallback is allowed"
            )

        if checkpoint_path and not Path(checkpoint_path).exists():
            raise MAPPOPolicyActionSourceError(f"checkpoint not found: {checkpoint_path}")

    @classmethod
    def from_config(
        cls,
        checkpoint_path: str,
        *,
        validation_mode: str = "actual",
        device: str = "cpu",
        deterministic: bool = True,
        require_checkpoint: bool = True,
        hidden_dim: int = 128,
        seed: int = 0,
        validation_report_path: str = "",
    ) -> "MAPPOPolicyActionSource":
        return cls(
            MAPPOPolicyActionSourceConfig(
                checkpoint_path=str(checkpoint_path or ""),
                validation_mode=str(validation_mode),
                device=str(device),
                deterministic=bool(deterministic),
                require_checkpoint=bool(require_checkpoint),
                hidden_dim=int(hidden_dim),
                seed=int(seed),
                validation_report_path=str(validation_report_path or ""),
            )
        )

    def _validate_checkpoint_for_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        from policies.validate_mappo_checkpoint import validate_checkpoint_path

        checkpoint_path = str(self.config.checkpoint_path or "").strip()
        if not checkpoint_path:
            if self.config.require_checkpoint:
                raise MAPPOPolicyActionSourceError(
                    "checkpoint_path is empty and require_checkpoint=True"
                )
            raise MAPPOPolicyActionSourceError(
                "checkpoint_path is empty. MAPPOPolicyActionSource never falls back to placeholder."
            )

        actor_obs_dim = _shape_last_dim(obs["actor_obs"], default=16)
        critic_obs_dim = _shape_last_dim(obs["critic_obs"], default=64)
        action_dim = _shape_last_dim(obs["action_mask"], default=2)

        report = validate_checkpoint_path(
            Path(checkpoint_path),
            mode=str(self.config.validation_mode),
            device=str(self.config.device),
            expected_actor_obs_dim=int(actor_obs_dim),
            expected_critic_obs_dim=int(critic_obs_dim),
            expected_action_dim=int(action_dim),
            expected_hidden_dim=int(self.config.hidden_dim),
        )

        payload = asdict(report)

        if self.config.validation_report_path:
            _dump_json(Path(self.config.validation_report_path), payload)

        if not report.valid:
            raise MAPPOPolicyActionSourceError(
                f"checkpoint validation failed: {report.errors}"
            )

        if payload.get("summary", {}).get("qwen_train") is not False:
            raise MAPPOPolicyActionSourceError("qwen_train must be false")

        if payload.get("summary", {}).get("qwen_inference") is not False:
            raise MAPPOPolicyActionSourceError("qwen_inference must be false")

        if float(payload.get("summary", {}).get("qwen_trigger_rate", 0.0)) != 0.0:
            raise MAPPOPolicyActionSourceError("qwen_trigger_rate must be 0.0")

        self.validation_payload = payload
        return payload

    def _ensure_adapter(self, obs: Dict[str, Any]) -> None:
        if self.adapter is not None:
            return

        from policies.mappo_neural_inference_adapter import NeuralMAPPOInferenceAdapter

        validation_payload = self._validate_checkpoint_for_obs(obs)

        self.adapter = NeuralMAPPOInferenceAdapter.from_obs(
            obs,
            checkpoint_path=str(self.config.checkpoint_path),
            device=str(self.config.device),
            deterministic=bool(self.config.deterministic),
            require_checkpoint=True,
            hidden_dim=int(self.config.hidden_dim),
            seed=int(self.config.seed),
        )

        checkpoint_status = getattr(self.adapter, "checkpoint_status", {})
        if not bool(checkpoint_status.get("checkpoint_loaded", False)):
            raise MAPPOPolicyActionSourceError(
                f"checkpoint validator passed but neural adapter did not load checkpoint: {checkpoint_status}"
            )

        self.validation_payload = validation_payload

    def select_actions(self, obs: Dict[str, Any]) -> MAPPOPolicyActionResult:
        self._ensure_adapter(obs)

        actions, policy_info = self.adapter.select_actions(obs)

        validation = self.validation_payload or {}
        validation_summary = validation.get("summary", {})

        metadata = {
            "policy_source": "mappo_policy",
            "policy_action_source_version": "mappo_policy_action_source_v1",
            "policy_action_source_mode": (
                "mappo_policy_actual_v1"
                if str(self.config.validation_mode).lower() == "actual"
                else "mappo_policy_smoke_v1"
            ),
            "checkpoint_path": str(self.config.checkpoint_path),
            "checkpoint_validation_mode": str(self.config.validation_mode),
            "checkpoint_validator_ran": True,
            "checkpoint_loaded": bool(
                policy_info.get("checkpoint_status", {}).get("checkpoint_loaded", False)
            ),
            "trained_model": bool(validation.get("trained_model", False)),
            "performance_claim_allowed": bool(
                validation.get("performance_claim_allowed", False)
            ),
            "placeholder_fallback_used": False,
            "mock_action_used": False,
            "qwen_train": validation_summary.get("qwen_train"),
            "qwen_inference": validation_summary.get("qwen_inference"),
            "qwen_trigger_rate": validation_summary.get("qwen_trigger_rate"),
            "reward_version": validation_summary.get("reward_version"),
            "energy_proxy_model_version": validation_summary.get("energy_proxy_model_version"),
            "energy_proxy_constants": validation_summary.get("energy_proxy_constants"),
            "action_count": int(len(actions)),
            "actions_preview": [
                {"agent_id": int(agent_id), "action": int(action)}
                for agent_id, action in list(actions.items())[:20]
            ],
            "policy_info": policy_info,
            "note": (
                "This is an action-source bridge for rollout writers. "
                "Causal performance claims still require a causal simulator source_mode "
                "and an actual H200 trained checkpoint."
            ),
        }

        return MAPPOPolicyActionResult(actions=actions, metadata=metadata)


def build_mappo_policy_action_source(
    checkpoint_path: str,
    *,
    validation_mode: str = "actual",
    device: str = "cpu",
    deterministic: bool = True,
    require_checkpoint: bool = True,
    hidden_dim: int = 128,
    seed: int = 0,
    validation_report_path: str = "",
) -> MAPPOPolicyActionSource:
    return MAPPOPolicyActionSource.from_config(
        checkpoint_path=checkpoint_path,
        validation_mode=validation_mode,
        device=device,
        deterministic=deterministic,
        require_checkpoint=require_checkpoint,
        hidden_dim=hidden_dim,
        seed=seed,
        validation_report_path=validation_report_path,
    )
