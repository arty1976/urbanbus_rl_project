from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except Exception as exc:
    torch = None
    nn = None
    Categorical = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class MAPPOInferenceConfig:
    """Configuration for actual neural MAPPO inference wiring."""

    checkpoint_path: str = ""
    device: str = "cpu"
    actor_obs_dim: int = 16
    critic_obs_dim: int = 64
    action_dim: int = 2
    hidden_dim: int = 128
    deterministic: bool = True
    require_checkpoint: bool = False
    seed: int = 0


class ActorCriticMLP(nn.Module):
    """Minimal actor-critic network used as the MAPPO policy inference shell."""

    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(actor_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        actor_obs: "torch.Tensor",
        critic_obs: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        logits = self.actor(actor_obs)

        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)

        if critic_obs.size(0) == 1 and actor_obs.size(0) > 1:
            critic_in = critic_obs.expand(actor_obs.size(0), -1)
        elif critic_obs.size(0) != actor_obs.size(0):
            critic_in = critic_obs[:1].expand(actor_obs.size(0), -1)
        else:
            critic_in = critic_obs

        values = self.critic(critic_in).squeeze(-1)
        return logits, values


class NeuralMAPPOInferenceAdapter:
    """
    Actual neural MAPPO inference adapter shell.

    This class is intentionally inference-only. It owns no optimizer and performs
    no training update. When checkpoint_path is empty and require_checkpoint=False,
    it runs with deterministic random-initialized weights for Step 40 smoke tests.
    """

    def __init__(self, config: MAPPOInferenceConfig) -> None:
        if torch is None:
            raise RuntimeError(
                f"torch import failed; neural inference adapter cannot run: {_TORCH_IMPORT_ERROR}"
            )

        self.config = config
        self.device = self._resolve_device(config.device)
        torch.manual_seed(int(config.seed))

        self.model = ActorCriticMLP(
            actor_obs_dim=int(config.actor_obs_dim),
            critic_obs_dim=int(config.critic_obs_dim),
            action_dim=int(config.action_dim),
            hidden_dim=int(config.hidden_dim),
        ).to(self.device)
        self.model.eval()

        self.checkpoint_status: Dict[str, Any] = {
            "checkpoint_path": str(config.checkpoint_path or ""),
            "checkpoint_loaded": False,
            "status": "random_initialized_smoke_only",
            "missing_keys": [],
            "unexpected_keys": [],
        }

        self._maybe_load_checkpoint()

    @classmethod
    def from_obs(
        cls,
        obs: Dict[str, Any],
        *,
        checkpoint_path: str = "",
        device: str = "cpu",
        deterministic: bool = True,
        require_checkpoint: bool = False,
        hidden_dim: int = 128,
        seed: int = 0,
    ) -> "NeuralMAPPOInferenceAdapter":
        actor_obs = np.asarray(obs["actor_obs"])
        critic_obs = np.asarray(obs["critic_obs"])
        action_mask = np.asarray(
            obs.get("action_mask", np.ones((actor_obs.shape[0], 2), dtype=bool))
        )

        cfg = MAPPOInferenceConfig(
            checkpoint_path=str(checkpoint_path or ""),
            device=str(device),
            actor_obs_dim=int(actor_obs.shape[-1]),
            critic_obs_dim=int(critic_obs.shape[-1]),
            action_dim=int(action_mask.shape[-1]),
            hidden_dim=int(hidden_dim),
            deterministic=bool(deterministic),
            require_checkpoint=bool(require_checkpoint),
            seed=int(seed),
        )
        return cls(cfg)

    def _resolve_device(self, requested: str) -> "torch.device":
        requested = str(requested or "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _extract_state_dict(self, checkpoint: Any) -> Dict[str, Any]:
        if not isinstance(checkpoint, dict):
            raise RuntimeError("checkpoint must be a dict or state_dict-like object")

        for key in (
            "model_state_dict",
            "policy_state_dict",
            "actor_critic_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint

        raise RuntimeError(
            "checkpoint does not contain a recognized state_dict key "
            "(model_state_dict, policy_state_dict, actor_critic_state_dict, state_dict, model)"
        )

    def _strip_common_prefixes(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in state_dict.items():
            new_key = str(key)
            for prefix in ("module.", "policy.", "actor_critic."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            out[new_key] = value
        return out

    def _maybe_load_checkpoint(self) -> None:
        path_text = str(self.config.checkpoint_path or "").strip()
        if not path_text:
            if self.config.require_checkpoint:
                raise FileNotFoundError("checkpoint_path is required but empty")
            return

        path = Path(path_text)
        if not path.exists():
            if self.config.require_checkpoint:
                raise FileNotFoundError(f"checkpoint not found: {path}")
            self.checkpoint_status["status"] = "checkpoint_missing_random_initialized_smoke_only"
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = self._strip_common_prefixes(self._extract_state_dict(checkpoint))
        load_result = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.checkpoint_status.update(
            {
                "checkpoint_path": str(path),
                "checkpoint_loaded": True,
                "status": "checkpoint_loaded_inference_ready",
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }
        )

    def _tensor(self, value: Any, dtype: Any) -> "torch.Tensor":
        arr = np.asarray(value)
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def select_actions(self, obs: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[str, Any]]:
        actor_obs = self._tensor(obs["actor_obs"], torch.float32)
        critic_obs = self._tensor(obs["critic_obs"], torch.float32)

        num_agents = int(actor_obs.shape[0])
        agent_ids = list(obs.get("agent_ids", list(range(num_agents))))
        if len(agent_ids) != num_agents:
            raise ValueError(
                f"agent_ids length mismatch: {len(agent_ids)} vs actor_obs agents {num_agents}"
            )

        action_mask_np = np.asarray(
            obs.get(
                "action_mask",
                np.ones((num_agents, self.config.action_dim), dtype=bool),
            )
        ).astype(bool)

        if action_mask_np.shape != (num_agents, int(self.config.action_dim)):
            raise ValueError(
                f"action_mask shape mismatch: got {action_mask_np.shape}, "
                f"expected {(num_agents, int(self.config.action_dim))}"
            )

        action_mask = torch.as_tensor(action_mask_np, dtype=torch.bool, device=self.device)

        active_mask_np = np.asarray(
            obs.get("active_bus_mask", np.ones((num_agents,), dtype=bool))
        ).astype(bool)

        if active_mask_np.shape != (num_agents,):
            raise ValueError(
                f"active_bus_mask shape mismatch: got {active_mask_np.shape}, expected {(num_agents,)}"
            )

        active_mask = torch.as_tensor(active_mask_np, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits, values = self.model(actor_obs, critic_obs)
            masked_logits = logits.masked_fill(~action_mask, -1.0e9)

            if bool(self.config.deterministic):
                actions_tensor = torch.argmax(masked_logits, dim=-1)
                dist = Categorical(logits=masked_logits)
                log_probs_tensor = dist.log_prob(actions_tensor)
            else:
                dist = Categorical(logits=masked_logits)
                actions_tensor = dist.sample()
                log_probs_tensor = dist.log_prob(actions_tensor)

            actions_tensor = torch.where(
                active_mask,
                actions_tensor,
                torch.zeros_like(actions_tensor),
            )
            log_probs_tensor = torch.where(
                active_mask,
                log_probs_tensor,
                torch.zeros_like(log_probs_tensor),
            )

        actions_list = [int(x) for x in actions_tensor.detach().cpu().tolist()]
        actions = {
            int(agent_id): int(action)
            for agent_id, action in zip(agent_ids, actions_list)
        }

        info = {
            "adapter": "NeuralMAPPOInferenceAdapter",
            "config": asdict(self.config),
            "device_resolved": str(self.device),
            "checkpoint_status": self.checkpoint_status,
            "num_agents": num_agents,
            "active_agent_count": int(active_mask_np.sum()),
            "action_dim": int(self.config.action_dim),
            "deterministic": bool(self.config.deterministic),
            "qwen_trigger_rate": 0.0,
            "actions_preview": actions_list[:10],
            "log_probs_preview": [
                float(x) for x in log_probs_tensor.detach().cpu().tolist()[:10]
            ],
            "values_preview": [
                float(x) for x in values.detach().cpu().tolist()[:10]
            ],
            "logits_preview": [
                [float(v) for v in row]
                for row in masked_logits.detach().cpu().tolist()[:3]
            ],
            "note": (
                "Step 40 adapter wiring. Random-initialized output is valid only for smoke "
                "validation unless checkpoint_loaded=true."
            ),
        }

        return actions, info
