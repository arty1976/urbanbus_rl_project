"""BT8-F1 frozen replicate and PPO partition authority.

This module contains accounting and validation only.  It creates no model,
optimizer, rollout, checkpoint, or authorization grant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


CONTRACT_ID = "LS3_BT8_R4A_F1_SEED_PPO_PARTITION_V1"
REPLICATES = (
    {"replicate_id": "F1_R1", "environment_seed": 20260822, "actor_init_seed": 20260824,
     "critic_init_seed": 20260826, "train_windows": 6, "trajectories": 6,
     "decisions": 24, "ppo_update_cycles": 3},
    {"replicate_id": "F1_R2", "environment_seed": 20260823, "actor_init_seed": 20260825,
     "critic_init_seed": 20260827, "train_windows": 6, "trajectories": 6,
     "decisions": 24, "ppo_update_cycles": 3},
)
PPO_PARTITION = {
    "contract_id": CONTRACT_ID, "trajectory_length": 4, "samples_per_replicate": 24,
    "gae_scope": "per trajectory/window only; no cross-trajectory or cross-seed recurrence",
    "advantage_normalization_scope": "each replicate's 24 train samples only",
    "full_batch_size": 24, "ppo_epochs": 3, "minibatch_size": 24,
    "updates_per_replicate": 3, "total_update_cycles": 6,
    "actor_optimizer_steps": 6, "critic_optimizer_steps": 6, "raw_optimizer_step_calls": 12,
    "review_optimizer_inclusion": False, "optimizer_state_reuse": False,
}


class F1ExecutionContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise F1ExecutionContractError(code, detail)


def validate_contract(replicates: Sequence[Mapping[str, int | str]] = REPLICATES,
                      ppo: Mapping[str, object] = PPO_PARTITION) -> None:
    require(len(replicates) == 2, "F1_REPLICATE_COUNT_INVALID")
    env, actor, critic, ids = set(), set(), set(), set()
    for row in replicates:
        env.add(int(row["environment_seed"])); actor.add(int(row["actor_init_seed"])); critic.add(int(row["critic_init_seed"])); ids.add(str(row["replicate_id"]))
        require(int(row["train_windows"]) == int(row["trajectories"]) == 6, "F1_REPLICATE_TRAJECTORY_SCOPE_INVALID")
        require(int(row["decisions"]) == 24 and int(row["ppo_update_cycles"]) == 3, "F1_REPLICATE_UPDATE_SCOPE_INVALID")
    require(env == {20260822, 20260823} and actor == {20260824, 20260825} and critic == {20260826, 20260827} and len(ids) == 2,
            "F1_SEED_LINEAGE_INVALID")
    require(int(ppo.get("trajectory_length", -1)) == 4 and int(ppo.get("samples_per_replicate", -1)) == 24,
            "F1_PPO_SAMPLE_PARTITION_INVALID")
    require(int(ppo.get("full_batch_size", -1)) == int(ppo.get("minibatch_size", -1)) == 24 and int(ppo.get("ppo_epochs", -1)) == 3,
            "F1_PPO_FULL_BATCH_EPOCH_INVALID")
    require(int(ppo.get("updates_per_replicate", -1)) == 3 and int(ppo.get("total_update_cycles", -1)) == 6,
            "F1_PPO_UPDATE_ACCOUNTING_INVALID")
    require(ppo.get("review_optimizer_inclusion") is False and ppo.get("optimizer_state_reuse") is False,
            "F1_REVIEW_OR_OPTIMIZER_LEAKAGE")


@dataclass
class ReplicateBudget:
    replicate_id: str
    allowed_updates: int = 3
    updates: int = 0
    train_samples: int = 0
    review_samples: int = 0

    def add_train_samples(self, count: int) -> None:
        self.train_samples += int(count)
        require(self.train_samples <= 24, "F1_REPLICATE_TRAIN_SAMPLE_OVERRUN")

    def add_review_sample(self) -> None:
        self.review_samples += 1
        raise F1ExecutionContractError("REVIEW_ROW_ENTERED_OPTIMIZER_PATH")

    def update(self) -> None:
        self.updates += 1
        require(self.updates <= self.allowed_updates, "F1_REPLICATE_UPDATE_BUDGET_EXCEEDED")

    def finalize(self) -> None:
        require(self.train_samples == 24, "F1_REPLICATE_TRAIN_SAMPLE_COUNT_INVALID")
        require(self.updates == self.allowed_updates, "F1_REPLICATE_UPDATE_COUNT_INVALID")
        require(self.review_samples == 0, "REVIEW_ROW_ENTERED_OPTIMIZER_PATH")


def normalize_advantages_for_replicate(values: Iterable[float]) -> list[float]:
    raw = [float(value) for value in values]
    require(len(raw) == 24, "F1_ADVANTAGE_SCOPE_MUST_BE_24")
    require(all(math.isfinite(value) for value in raw), "F1_ADVANTAGE_NONFINITE")
    mean = sum(raw) / len(raw)
    variance = sum((value - mean) ** 2 for value in raw) / len(raw)
    scale = max(variance ** 0.5, 1e-8)
    return [(value - mean) / scale for value in raw]
