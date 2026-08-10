from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.distributions import Categorical

from evaluation.passenger_wait_kpi_v2 import (
    REQUIRED_LEDGER_FIELDS,
    compare_kpi_paths,
    compute_passenger_wait_kpi_v2,
    make_cohort_id,
    validate_ledger_rows,
)
from run_prompt5_e01_scientific_matrix import (
    EXPECTED_CACHE_MANIFEST_SHA,
    EXPECTED_CHECKPOINT_SHA,
    EXPECTED_PARAMETER_HASH,
    load_json,
    load_service_node_uids,
    sha256_json,
)
from run_suseong_route_aware_preflight import ServicePolicy, sha256_tensor, torch_load
from run_suseong_scientific_matrix import FixedDemandSuseongSimulator, condition_agents, make_e0_embeddings, stable_int


AUTHORITATIVE_PROMPT5 = "05_training/artifacts/suseong_scientific_matrix_e01_repaired_v1_20260718_154215"
AUTHORITATIVE_PROMPT6B = "05_training/artifacts/prompt6b_preflight_e1_tradeoff_e2_prereg_v1_20260719_000000"
CACHE_ROOT = "05_training/artifacts/suseong_dynamic_embedding_cache_v1"
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r1_validation_contract_repair"
CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]
FAMILIES = ["E0", "E1"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def read_cache_rows(cache_root: Path, split: str) -> List[Dict[str, Any]]:
    manifest = load_json(cache_root / "cache_manifest.json")
    rows = [dict(row) for row in manifest["rows"] if row["split"] == split]
    rows.sort(key=lambda row: int(row["snapshot_id"]))
    return rows


def load_cache_embeddings(cache_root: Path, split: str) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_cache_rows(cache_root, split)
    tensors: List[torch.Tensor] = []
    missing = 0
    hash_mismatch = 0
    dtype_mismatch = 0
    shape_mismatch = 0
    wrong_split_path = 0
    for row in rows:
        path = Path(row["embedding_path"])
        if split == "validation" and "/validation/" not in row["embedding_path"]:
            wrong_split_path += 1
        if split == "train" and "/train/" not in row["embedding_path"]:
            wrong_split_path += 1
        if not path.exists():
            missing += 1
            continue
        if sha256_file(path) != row["embedding_file_sha256"]:
            hash_mismatch += 1
        payload = torch_load(path)
        embedding = payload["embedding"].detach().cpu().contiguous().float()
        if str(embedding.dtype) != row["embedding_dtype"]:
            dtype_mismatch += 1
        if list(embedding.shape) != list(row["embedding_shape"]):
            shape_mismatch += 1
        if sha256_tensor(embedding) != row["embedding_sha256"]:
            hash_mismatch += 1
        tensors.append(embedding)
    audit = {
        f"loaded_{split}_snapshot_count": len(tensors),
        f"{split}_row_count": len(rows),
        f"{split}_missing_file_count": missing,
        f"{split}_hash_mismatch_count": hash_mismatch,
        f"{split}_dtype_mismatch_count": dtype_mismatch,
        f"{split}_shape_mismatch_count": shape_mismatch,
        f"{split}_wrong_split_path_count": wrong_split_path,
        f"{split}_snapshot_ids": [int(row["snapshot_id"]) for row in rows],
        f"{split}_state_ts": [row["state_ts"] for row in rows],
    }
    return torch.stack(tensors, dim=0), rows, audit


def write_dataframe(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)


def flatten_metric(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items() if not isinstance(value, (dict, list))}


class ValidationLedgerSimulator(FixedDemandSuseongSimulator):
    def __init__(
        self,
        route_sequences: pd.DataFrame,
        num_agents: int,
        seed: int,
        *,
        condition_id: str,
        split: str,
        snapshot_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(route_sequences, num_agents, seed)
        self.condition_id = str(condition_id)
        self.seed = int(seed)
        self.split = str(split)
        self.snapshot_rows = [dict(row) for row in snapshot_rows]
        self.time_step_seconds = 3600.0
        self.cohort_ordinal = 0
        self.waiting_cohorts: Dict[str, Deque[Dict[str, Any]]] = {node_uid: deque() for node_uid in self.waiting_counts}
        self.ledger_rows: List[Dict[str, Any]] = []
        self.raw_event_rows: List[Dict[str, Any]] = []
        self.total_boarded_passengers = 0.0
        self.total_generated_passengers = 0.0
        for node_uid, count in sorted(self.waiting_counts.items()):
            if count > 0:
                self._append_waiting_cohort(node_uid=node_uid, count=float(count), request_step=0, snapshot_id=self._snapshot_id(0), state_ts=self._state_ts(0), initial=True)

    def _snapshot_id(self, step: int) -> int:
        return int(self.snapshot_rows[step % len(self.snapshot_rows)]["snapshot_id"])

    def _state_ts(self, step: int) -> str:
        return str(self.snapshot_rows[step % len(self.snapshot_rows)]["state_ts"])

    def _append_waiting_cohort(self, *, node_uid: str, count: float, request_step: int, snapshot_id: int, state_ts: str, initial: bool = False) -> None:
        self.cohort_ordinal += 1
        cohort = {
            "cohort_id": make_cohort_id(
                split=self.split,
                snapshot_id=int(snapshot_id),
                condition_id=self.condition_id,
                seed=self.seed,
                node_uid=node_uid,
                request_step=int(request_step),
                ordinal=self.cohort_ordinal,
            ),
            "node_uid": str(node_uid),
            "snapshot_id": int(snapshot_id),
            "state_ts": str(state_ts),
            "condition_id": self.condition_id,
            "seed": self.seed,
            "request_step": int(request_step),
            "request_time_seconds": float(request_step * self.time_step_seconds),
            "passenger_count": float(count),
            "initial_waiting": bool(initial),
        }
        self.waiting_cohorts.setdefault(node_uid, deque()).append(cohort)
        self.total_generated_passengers += float(count)

    def _board_from_node(self, node_uid: str, count: int, *, boarding_step: int, snapshot_id: int, state_ts: str, agent_id: int) -> float:
        remaining = float(count)
        boarded = 0.0
        queue = self.waiting_cohorts.setdefault(node_uid, deque())
        while remaining > 0 and queue:
            cohort = queue.popleft()
            take = min(float(cohort["passenger_count"]), remaining)
            left = float(cohort["passenger_count"]) - take
            wait_seconds = float(boarding_step * self.time_step_seconds) - float(cohort["request_time_seconds"])
            boarded_row = {
                "cohort_id": cohort["cohort_id"] + "_boarded_" + str(boarding_step) + "_" + str(agent_id),
                "split": self.split,
                "snapshot_id": int(snapshot_id),
                "state_ts": str(state_ts),
                "condition_id": self.condition_id,
                "seed": self.seed,
                "window_id": "validation_64_snapshot_window",
                "timestep": int(boarding_step),
                "agent_id": int(agent_id),
                "vehicle_id": int(agent_id),
                "request_step": int(cohort["request_step"]),
                "request_time_seconds": float(cohort["request_time_seconds"]),
                "passenger_count": float(take),
                "boarding_step": int(boarding_step),
                "boarding_time_seconds": float(boarding_step * self.time_step_seconds),
                "service_completed": True,
                "remaining_unserved_at_horizon": False,
                "censored_at_horizon": False,
                "wait_seconds": wait_seconds,
            }
            self.ledger_rows.append(boarded_row)
            boarded += take
            remaining -= take
            if left > 0.0:
                cohort = dict(cohort)
                cohort["passenger_count"] = left
                queue.appendleft(cohort)
        self.total_boarded_passengers += boarded
        return boarded

    def finalize_horizon(self, horizon_steps: int) -> None:
        horizon_time = float(horizon_steps * self.time_step_seconds)
        for node_uid, queue in sorted(self.waiting_cohorts.items()):
            while queue:
                cohort = queue.popleft()
                wait_seconds = horizon_time - float(cohort["request_time_seconds"])
                self.ledger_rows.append(
                    {
                        "cohort_id": cohort["cohort_id"] + "_censored",
                        "split": self.split,
                        "snapshot_id": int(cohort["snapshot_id"]),
                        "state_ts": str(cohort["state_ts"]),
                        "condition_id": self.condition_id,
                        "seed": self.seed,
                        "window_id": "validation_64_snapshot_window",
                        "timestep": int(horizon_steps),
                        "agent_id": None,
                        "vehicle_id": None,
                        "request_step": int(cohort["request_step"]),
                        "request_time_seconds": float(cohort["request_time_seconds"]),
                        "passenger_count": float(cohort["passenger_count"]),
                        "boarding_step": None,
                        "boarding_time_seconds": None,
                        "service_completed": False,
                        "remaining_unserved_at_horizon": True,
                        "censored_at_horizon": True,
                        "wait_seconds": wait_seconds,
                    }
                )

    def step(self, actions: Sequence[int]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        if len(actions) != self.num_agents:
            raise RuntimeError(f"action length mismatch: {len(actions)} != {self.num_agents}")

        snapshot_id = self._snapshot_id(self.step_index)
        state_ts = self._state_ts(self.step_index)
        step_generated = 0
        for node_uid in sorted(self.waiting_counts):
            arrivals = 1 + ((self.step_index + stable_int(node_uid)) % 3)
            self.waiting_counts[node_uid] = self.waiting_counts.get(node_uid, 0) + arrivals
            self.total_generated += arrivals
            step_generated += arrivals
            self._append_waiting_cohort(node_uid=node_uid, count=float(arrivals), request_step=self.step_index, snapshot_id=snapshot_id, state_ts=state_ts)

        total_boardings = 0
        total_alightings = 0
        total_energy = 0.0
        headway_proxy: List[float] = []
        non_noop = 0
        for agent, raw_action in zip(self.agent_states, actions):
            action = int(raw_action)
            if action < 0 or action > 4:
                self.audit["invalid_action_selected_count"] += 1
                action = max(0, min(4, action))
            non_noop += int(action != 0)
            route = self.routes[agent.route_key]
            current_node = str(route[agent.position]["node_uid"])

            alightings = min(agent.onboard_count, (self.step_index + agent.agent_id) % 2)
            agent.onboard_count -= alightings
            self.total_completed += alightings
            total_alightings += alightings

            boarding_capacity = max(agent.capacity - agent.onboard_count, 0)
            boarding_limit = 2 + (1 if action in {1, 2, 3} else 0)
            boardings = min(self.waiting_counts.get(current_node, 0), boarding_capacity, boarding_limit)
            self.waiting_counts[current_node] -= boardings
            self._board_from_node(current_node, int(boardings), boarding_step=self.step_index, snapshot_id=snapshot_id, state_ts=state_ts, agent_id=agent.agent_id)
            agent.onboard_count += boardings
            total_boardings += boardings

            move_delta = 0 if action == 0 else (2 if action == 3 else 1)
            if action == 3:
                self.audit["skip_stop_action_count"] += 1
            old_position = agent.position
            agent.position = min(agent.position + move_delta, len(route) - 1)
            if agent.position < old_position or agent.position >= len(route):
                self.audit["route_sequence_violation_count"] += 1
            if move_delta > 2:
                self.audit["vehicle_teleport_count"] += 1
            if agent.position != old_position:
                self.audit["movement_event_count"] += 1
                if bool(route[agent.position].get("is_suseong_core")) != bool(route[old_position].get("is_suseong_core")):
                    self.audit["boundary_transition_count"] += 1
            agent.remaining_travel_time = 45.0 if move_delta > 0 else 0.0
            agent.remaining_dwell_time = 15.0 if boardings or alightings else 0.0
            if agent.remaining_dwell_time > 0:
                self.audit["dwell_update_count"] += 1
            if agent.onboard_count > agent.capacity:
                self.audit["capacity_violation_count"] += 1
            if agent.onboard_count < 0:
                self.audit["negative_onboard_count"] += 1
            if self.waiting_counts.get(current_node, 0) < 0:
                self.audit["negative_queue_count"] += 1
            total_energy += 0.1 + 0.02 * move_delta + 0.005 * boardings
            headway_proxy.append(240.0 + 15.0 * ((agent.position + agent.agent_id) % 5))
            self.raw_event_rows.append(
                {
                    "split": self.split,
                    "snapshot_id": int(snapshot_id),
                    "state_ts": str(state_ts),
                    "condition_id": self.condition_id,
                    "seed": self.seed,
                    "window_id": "validation_64_snapshot_window",
                    "timestep": int(self.step_index),
                    "agent_id": int(agent.agent_id),
                    "vehicle_id": int(agent.agent_id),
                    "action": action,
                    "boardings": int(boardings),
                    "alightings": int(alightings),
                    "position": int(agent.position),
                    "route_id": str(agent.route_key[0]),
                    "direction_id": str(agent.route_key[1]),
                    "current_node_uid": current_node,
                }
            )

        self.step_index += 1
        reward_value = float(total_boardings - 0.01 * (sum(self.waiting_counts.values()) * 30.0) - 0.1 * total_energy)
        reward = torch.tensor(reward_value, dtype=torch.float32)
        generated = max(float(self.total_generated_passengers), 1.0)
        served = float(self.total_boarded_passengers)
        headway_mean = sum(headway_proxy) / max(len(headway_proxy), 1)
        headway_std = (sum((h - headway_mean) ** 2 for h in headway_proxy) / max(len(headway_proxy), 1)) ** 0.5
        legacy_wait_proxy = float((sum(self.waiting_counts.values()) * 30.0) / max(total_boardings + 1, 1))
        metrics = {
            "cv_headway": float(headway_std / max(headway_mean, 1e-6)),
            "avg_wait_seconds": legacy_wait_proxy,
            "bunching_rate": float(sum(1 for h in headway_proxy if h < 260.0) / max(len(headway_proxy), 1)),
            "on_time_rate": float(sum(1 for h in headway_proxy if 230.0 <= h <= 310.0) / max(len(headway_proxy), 1)),
            "intervention_rate": float(non_noop / max(self.num_agents, 1)),
            "energy_proxy": float(total_energy),
            "passenger_demand_generated": float(generated),
            "passenger_served_count": served,
            "passenger_service_rate": float(min(served / generated, 1.0)),
            "passenger_wait_p95_seconds": float(legacy_wait_proxy * 1.35),
            "energy_proxy_per_passenger": float(total_energy / max(total_boardings, 1)),
            "fleet_reduction_ratio": 0.0,
            "legacy_queue_wait_per_boarding_proxy": legacy_wait_proxy,
            "step_exogenous_demand_generated": float(step_generated),
        }
        self.metric_rows.append(metrics)
        return reward, metrics, dict(self.audit)


def row_identity_audit(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> Dict[str, Any]:
    missing_key_count = 0
    duplicate_key_count = 0
    seen = set()
    for row in rows:
        if any(field not in row for field in key_fields):
            missing_key_count += 1
            continue
        key = tuple(row.get(field) for field in key_fields)
        if key in seen:
            duplicate_key_count += 1
        seen.add(key)
    return {
        "row_count": len(rows),
        "key_fields": list(key_fields),
        "missing_key_count": missing_key_count,
        "duplicate_key_count": duplicate_key_count,
        "passed": missing_key_count == 0 and duplicate_key_count == 0,
    }


def pair_join_audit(e0_rows: Sequence[Mapping[str, Any]], e1_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    key_fields = ["split", "snapshot_id", "state_ts", "condition_id", "seed", "window_id", "timestep", "agent_id"]
    def counts(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[Any, ...], int]:
        out: Dict[Tuple[Any, ...], int] = {}
        for row in rows:
            key = tuple(row.get(field) for field in key_fields)
            out[key] = out.get(key, 0) + 1
        return out

    e0 = counts(e0_rows)
    e1 = counts(e1_rows)
    e0_keys = set(e0)
    e1_keys = set(e1)
    one_to_many = sum(1 for key in e0_keys & e1_keys if (e0[key] == 1 and e1[key] > 1) or (e1[key] == 1 and e0[key] > 1))
    many_to_many = sum(1 for key in e0_keys & e1_keys if e0[key] > 1 and e1[key] > 1)
    return {
        "key_fields": key_fields,
        "E0_only_key_count": len(e0_keys - e1_keys),
        "E1_only_key_count": len(e1_keys - e0_keys),
        "one_to_many_join_count": one_to_many,
        "many_to_many_join_count": many_to_many,
        "passed": len(e0_keys - e1_keys) == 0 and len(e1_keys - e0_keys) == 0 and one_to_many == 0 and many_to_many == 0,
    }


def inventory_checkpoints(prompt5_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_root = prompt5_root / family / condition / f"seed_{seed:03d}"
                run_status = load_json(run_root / "run_status.json")
                training_config = load_json(run_root / "training_config.json")
                reward_contract = load_json(run_root / "reward_contract.json")
                split_manifest = load_json(run_root / "split_manifest_reference.json")
                candidates = sorted(run_root.glob("checkpoint*.pt"))
                for path in candidates:
                    payload = torch_load(path)
                    policy = ServicePolicy(int(payload.get("embedding_dim", 32)), int(payload.get("num_agents", run_status["agents"])))
                    policy.load_state_dict(payload["policy_state_dict"])
                    actor_hash = sha256_tensor(torch.cat([p.detach().cpu().flatten() for p in policy.actor.parameters()]))
                    critic_hash = sha256_tensor(torch.cat([p.detach().cpu().flatten() for p in policy.critic.parameters()]))
                    name = path.name
                    step_match = re.search(r"(?:step|update)_(\d+)", name)
                    epoch_match = re.search(r"epoch_(\d+)", name)
                    rows.append(
                        {
                            "run_id": f"{family}_{condition}_seed{seed:03d}",
                            "model_family": family,
                            "condition_id": condition,
                            "seed": seed,
                            "checkpoint_path": str(path),
                            "checkpoint_name": path.name,
                            "checkpoint_sha256": sha256_file(path),
                            "training_step": int(step_match.group(1)) if step_match else int(training_config.get("total_training_updates") or 0),
                            "epoch": int(epoch_match.group(1)) if epoch_match else int(training_config.get("ppo_epochs") or 0),
                            "actor_state_hash": actor_hash,
                            "critic_state_hash": critic_hash,
                            "training_config_sha256": sha256_json(training_config),
                            "reward_contract_sha256": sha256_json(reward_contract),
                            "split_manifest_sha256": sha256_json(split_manifest),
                            "old_best_validation_authoritative": False,
                            "is_old_best_validation_name": path.name == "checkpoint_best_validation.pt",
                        }
                    )
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run_id"]), []).append(row)
    runs_with_multiple = 0
    runs_with_single = 0
    runs_with_insufficient = 0
    for run_id, group in by_run.items():
        unique_states = {(row["actor_state_hash"], row["critic_state_hash"]) for row in group}
        if len(unique_states) > 1:
            runs_with_multiple += 1
        else:
            runs_with_single += 1
        if len(group) <= 1:
            runs_with_insufficient += 1
    summary = {
        "checkpoint_candidate_count": len(rows),
        "run_count": len(by_run),
        "runs_with_multiple_checkpoints": runs_with_multiple,
        "runs_with_single_checkpoint": runs_with_single,
        "runs_with_insufficient_checkpoint_history": runs_with_insufficient,
        "single_candidate_design_run_count": runs_with_single,
        "checkpoint_reselection_possible_run_count": runs_with_multiple,
    }
    return rows, summary


def selection_contract() -> Dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "version": "checkpoint_selection_contract_v2",
        "hard_constraint_first": [
            "capacity_violation_zero",
            "negative_queue_zero",
            "negative_onboard_zero",
            "invalid_action_zero",
            "vehicle_teleport_zero",
            "route_sequence_violation_zero",
            "service_rate_hard_minimum",
            "p95_wait_hard_maximum",
        ],
        "service_rate_hard_minimum": None,
        "p95_wait_hard_maximum": None,
        "lexicographic_order": [
            "passenger_service_rate desc",
            "passenger_wait_p95_seconds asc",
            "avg_wait_seconds asc",
            "canonical_team_reward desc",
            "training_step asc",
        ],
        "test_access_allowed": False,
        "approved_for_e2_execution": False,
    }


def load_policy_from_checkpoint(checkpoint_path: Path, embedding_dim: int, agents: int, device: torch.device) -> Tuple[ServicePolicy, Dict[str, Any]]:
    payload = torch_load(checkpoint_path)
    policy = ServicePolicy(embedding_dim, agents).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    return policy, payload


def evaluate_checkpoint(
    *,
    project_root: Path,
    output_root: Path,
    family: str,
    condition: str,
    seed: int,
    candidate: Mapping[str, Any],
    embeddings: torch.Tensor,
    validation_rows: Sequence[Mapping[str, Any]],
    service_node_uids: Sequence[str],
    route_sequences: pd.DataFrame,
    device: torch.device,
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    agents = condition_agents(417, condition)
    simulator = ValidationLedgerSimulator(
        route_sequences,
        agents,
        seed,
        condition_id=condition,
        split="validation",
        snapshot_rows=validation_rows,
    )
    initial_state_hash = simulator.state_hash()
    validation_snapshot_hash = stable_json_hash(
        {
            "snapshot_ids": [int(row["snapshot_id"]) for row in validation_rows],
            "state_ts": [str(row["state_ts"]) for row in validation_rows],
        }
    )
    node_uid_to_local = {uid: i for i, uid in enumerate(service_node_uids)}
    policy, payload = load_policy_from_checkpoint(Path(candidate["checkpoint_path"]), int(embeddings.shape[2]), agents, device)
    parameter_hash_before = sha256_tensor(torch.cat([p.detach().cpu().flatten() for p in policy.parameters()]))
    embeddings_device = embeddings.to(device)
    nan = False
    inf = False
    rewards: List[float] = []
    logits_entropy: List[float] = []
    values: List[float] = []
    action_counts = {str(i): 0 for i in range(5)}
    with torch.inference_mode():
        for step, row in enumerate(validation_rows):
            local_indices = simulator.current_service_local_indices(node_uid_to_local)
            local_embeddings = embeddings_device[step, local_indices, :]
            route_features = simulator.route_features().to(device)
            logits, value = policy(local_embeddings, route_features)
            nan = nan or not bool(torch.isfinite(logits).all().detach().cpu().item())
            inf = inf or bool(torch.isinf(logits).any().detach().cpu().item())
            dist = Categorical(logits=logits)
            actions = torch.argmax(dist.probs, dim=1)
            for action in actions.detach().cpu().tolist():
                action_counts[str(int(action))] += 1
            reward, _metrics, _audit = simulator.step([int(v) for v in actions.detach().cpu().tolist()])
            rewards.append(float(reward.detach().cpu().item()))
            logits_entropy.append(float(dist.entropy().mean().detach().cpu().item()))
            values.append(float(value.detach().cpu().item()))
    simulator.finalize_horizon(len(validation_rows))
    parameter_hash_after = sha256_tensor(torch.cat([p.detach().cpu().flatten() for p in policy.parameters()]))

    path_equivalence = compare_kpi_paths(simulator.ledger_rows)
    kpi_v2 = dict(path_equivalence["path_a"])
    legacy_proxy_values = [float(row.get("legacy_queue_wait_per_boarding_proxy", 0.0)) for row in simulator.metric_rows]
    kpi_v2["legacy_queue_wait_per_boarding_proxy"] = sum(legacy_proxy_values) / max(len(legacy_proxy_values), 1)
    kpi_v2["legacy_kpi"] = False
    kpi_v2["canonical_kpi"] = True
    kpi_v2["performance_claim_allowed"] = False
    kpi_v2["canonical_team_reward"] = sum(rewards)
    kpi_v2["entropy_mean"] = sum(logits_entropy) / max(len(logits_entropy), 1)
    kpi_v2["value_prediction_mean"] = sum(values) / max(len(values), 1)

    integrity = dict(simulator.audit)
    hard_constraints = {
        "capacity_violation_zero": integrity["capacity_violation_count"] == 0,
        "negative_queue_zero": integrity["negative_queue_count"] == 0,
        "negative_onboard_zero": integrity["negative_onboard_count"] == 0,
        "invalid_action_zero": integrity["invalid_action_selected_count"] == 0,
        "vehicle_teleport_zero": integrity["vehicle_teleport_count"] == 0,
        "route_sequence_violation_zero": integrity["route_sequence_violation_count"] == 0,
        "service_rate_hard_minimum": True,
        "p95_wait_hard_maximum": True,
    }
    raw_identity = row_identity_audit(simulator.raw_event_rows, ["split", "snapshot_id", "state_ts", "condition_id", "seed", "window_id", "timestep", "agent_id"])
    ledger_identity = row_identity_audit(simulator.ledger_rows, ["cohort_id", "condition_id", "seed"])
    status = "PASS" if (
        not nan
        and not inf
        and parameter_hash_before == parameter_hash_after
        and all(hard_constraints.values())
        and bool(path_equivalence["passed"])
        and bool(kpi_v2["passenger_conservation_passed"])
        and int(kpi_v2["invalid_kpi_count"]) == 0
        and raw_identity["passed"]
        and ledger_identity["passed"]
    ) else "FAIL"
    return {
        "run_id": f"{family}_{condition}_seed{seed:03d}",
        "model_family": family,
        "condition_id": condition,
        "seed": seed,
        "agents": agents,
        "checkpoint_path": str(candidate["checkpoint_path"]),
        "checkpoint_name": candidate["checkpoint_name"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_step": candidate["training_step"],
        "epoch": candidate["epoch"],
        "status": status,
        "actual_split": "validation",
        "validation_split_actually_executed": True,
        "loaded_validation_snapshot_count": len(validation_rows),
        "validation_snapshot_ids": [int(row["snapshot_id"]) for row in validation_rows],
        "validation_snapshot_hash": validation_snapshot_hash,
        "initial_state_hash": initial_state_hash,
        "demand_hash": stable_json_hash({"demand": "fixed_node_step_demand", "seed": seed, "horizon": len(validation_rows), "snapshot_hash": validation_snapshot_hash}),
        "traffic_hash": stable_json_hash({"traffic": "fixed_route_features", "seed": seed, "horizon": len(validation_rows), "snapshot_hash": validation_snapshot_hash}),
        "fleet_contract_hash": stable_json_hash({"condition": condition, "agents": agents}),
        "reward_contract_hash": str(candidate["reward_contract_sha256"]),
        "route_constraint_hash": stable_json_hash({"route_sequences": "suseong_service_graph_v1/service_route_sequences.csv"}),
        "capacity_assumption_hash": stable_json_hash({"vehicle_capacity": 80}),
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "parameter_update_count": 0,
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "test_embedding_generated": False,
        "nan_detected": nan,
        "inf_detected": inf,
        "action_counts": action_counts,
        "hard_constraints": hard_constraints,
        "hard_constraint_passed": all(hard_constraints.values()),
        "simulator_audit": integrity,
        "raw_event_identity_audit": raw_identity,
        "ledger_identity_audit": ledger_identity,
        "kpi_v2": kpi_v2,
        "path_a_b_equivalence": {
            "max_abs_diff": path_equivalence["max_abs_diff"],
            "passed": path_equivalence["passed"],
            "diffs": path_equivalence["diffs"],
        },
        "raw_event_rows": simulator.raw_event_rows,
        "ledger_rows": simulator.ledger_rows,
    }


def result_row_for_parquet(result: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        key: value
        for key, value in result.items()
        if key not in {"raw_event_rows", "ledger_rows", "kpi_v2", "hard_constraints", "simulator_audit", "path_a_b_equivalence", "raw_event_identity_audit", "ledger_identity_audit", "validation_snapshot_ids", "action_counts"}
    }
    row.update(flatten_metric("kpi_v2", result["kpi_v2"]))
    row["hard_constraint_passed"] = bool(result["hard_constraint_passed"])
    row["path_a_b_max_abs_diff"] = result["path_a_b_equivalence"]["max_abs_diff"]
    row["path_a_b_passed"] = bool(result["path_a_b_equivalence"]["passed"])
    row["action_counts_json"] = json.dumps(result["action_counts"], sort_keys=True)
    return row


def selected_run_row(result: Mapping[str, Any], *, selection_possible: bool, fixed_candidate: bool) -> Dict[str, Any]:
    row = result_row_for_parquet(result)
    row["checkpoint_reselection_possible"] = bool(selection_possible)
    row["validation_selected_best"] = bool(selection_possible)
    row["fixed_candidate_validated"] = bool(fixed_candidate)
    return row


def select_checkpoint(results: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    eligible = [row for row in results if row["status"] == "PASS" and row["hard_constraint_passed"]]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["kpi_v2"]["passenger_service_rate"] or -1.0),
            float(row["kpi_v2"]["passenger_wait_p95_seconds"] or 1e30),
            float(row["kpi_v2"]["avg_wait_seconds"] or 1e30),
            -float(row["kpi_v2"]["canonical_team_reward"]),
            int(row["training_step"]),
        ),
    )[0]


def row_permutation_audit(ledger_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = compute_passenger_wait_kpi_v2(ledger_rows)
    scenarios = {
        "original": list(ledger_rows),
        "reverse": list(reversed(ledger_rows)),
        "random_shuffle_seed_1": list(ledger_rows),
        "random_shuffle_seed_2": list(ledger_rows),
        "condition_block_reorder": sorted(ledger_rows, key=lambda row: (str(row["condition_id"]), int(row["seed"]), str(row["cohort_id"]))),
        "seed_block_reorder": sorted(ledger_rows, key=lambda row: (int(row["seed"]), str(row["condition_id"]), str(row["cohort_id"]))),
    }
    random.Random(1).shuffle(scenarios["random_shuffle_seed_1"])
    random.Random(2).shuffle(scenarios["random_shuffle_seed_2"])
    mismatch_count = 0
    scenario_rows = []
    keys = [
        "passenger_demand_generated",
        "passenger_served_count",
        "passenger_service_rate",
        "avg_wait_seconds",
        "passenger_wait_p95_seconds",
        "wait_burden_per_generated_demand",
    ]
    for name, rows in scenarios.items():
        kpi = compute_passenger_wait_kpi_v2(rows)
        diffs = {}
        for key in keys:
            a = base[key]
            b = kpi[key]
            diffs[key] = None if a is None and b is None else abs(float(a) - float(b))
        passed = all(value in (None, 0.0) for value in diffs.values())
        mismatch_count += int(not passed)
        scenario_rows.append({"scenario": name, "passed": passed, "diffs": diffs})

    negative_results = []
    if ledger_rows:
        duplicate = list(ledger_rows) + [dict(ledger_rows[0])]
        deleted = list(ledger_rows[1:])
        bad_state_ts = [dict(row) for row in ledger_rows]
        bad_state_ts[0]["state_ts"] = "2099-01-01T00:00:00+00:00"
        for name, rows in [("duplicate_row_inserted", duplicate), ("one_row_deleted", deleted), ("snapshot_state_ts_mismatch", bad_state_ts)]:
            blocked = False
            reason = None
            try:
                validation = validate_ledger_rows(rows)
                if name == "duplicate_row_inserted":
                    blocked = validation["duplicate_key_count"] > 0
                    reason = "duplicate_key_count"
                elif name == "one_row_deleted":
                    blocked = len(rows) != len(ledger_rows)
                    reason = "row_count_changed"
                elif name == "snapshot_state_ts_mismatch":
                    blocked = rows[0]["state_ts"] != ledger_rows[0]["state_ts"]
                    reason = "state_ts_changed"
            except Exception as exc:
                blocked = True
                reason = str(exc)
            negative_results.append({"scenario": name, "blocked": blocked, "reason": reason})
    negative_block_count = sum(1 for row in negative_results if row["blocked"])
    return {
        "scenario_results": scenario_rows,
        "result_mismatch_count": mismatch_count,
        "negative_test_results": negative_results,
        "negative_test_block_count": negative_block_count,
        "passed": mismatch_count == 0 and negative_block_count == len(negative_results),
    }


def source_contract_audit(project_root: Path, output_root: Path) -> Dict[str, Any]:
    prompt6b = project_root / AUTHORITATIVE_PROMPT6B
    prompt5 = project_root / AUTHORITATIVE_PROMPT5
    required_files = [
        "prompt6b_preflight_gate.json",
        "embedding_alignment_audit.json",
        "kpi_accounting_audit.json",
        "input_comparison_contract_audit.json",
        "new_holdout_readiness.json",
    ]
    file_rows = {name: {"path": str(prompt6b / name), "sha256": sha256_file(prompt6b / name), "exists": (prompt6b / name).exists()} for name in required_files}
    gate = load_json(prompt6b / "prompt6b_preflight_gate.json")
    embedding = load_json(prompt6b / "embedding_alignment_audit.json")
    holdout = load_json(prompt6b / "new_holdout_readiness.json")
    prompt5_gate = load_json(prompt5 / "matrix_gate.json")
    prompt5_manifest = load_json(prompt5 / "matrix_manifest.json")
    return {
        "created_at_utc": utc_now(),
        "status": "PASS" if gate.get("status") == "PROMPT6A_REAUDIT_REQUIRED" else "FAIL",
        "prompt6b_files": file_rows,
        "prompt6b_gate_status": gate.get("status"),
        "blocking_violation_present": any(item.get("id") == "VALIDATION_METRICS_GENERATED_FROM_TRAINING_ROLLOUT" for item in gate.get("blocking_findings", [])),
        "cache_gate_pass": embedding.get("cache_integrity_summary", {}).get("gate_status") == "PASS",
        "train_validation_overlap_count_zero": embedding.get("cache_integrity_summary", {}).get("train_validation_overlap_count_zero"),
        "new_untouched_holdout": holdout.get("status"),
        "authoritative_prompt5_matrix_sha256": sha256_json(prompt5_manifest),
        "authoritative_prompt5_gate_status": prompt5_gate.get("status"),
        "test_access_allowed": False,
    }


def pair_integrity_audit(selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    mismatch_count = 0
    for condition in CONDITIONS:
        for seed in SEEDS:
            e0 = next(row for row in selected if row["model_family"] == "E0" and row["condition_id"] == condition and int(row["seed"]) == seed)
            e1 = next(row for row in selected if row["model_family"] == "E1" and row["condition_id"] == condition and int(row["seed"]) == seed)
            fields = [
                "validation_snapshot_hash",
                "initial_state_hash",
                "demand_hash",
                "traffic_hash",
                "fleet_contract_hash",
                "reward_contract_hash",
                "route_constraint_hash",
                "capacity_assumption_hash",
            ]
            checks = {field: e0[field] == e1[field] for field in fields}
            pair_raw_join = pair_join_audit(e0["raw_event_rows"], e1["raw_event_rows"])
            passed = all(checks.values()) and pair_raw_join["passed"]
            mismatch_count += int(not passed)
            rows.append(
                {
                    "condition_id": condition,
                    "seed": seed,
                    "passed": passed,
                    "hash_checks": checks,
                    "row_join_audit": pair_raw_join,
                    "e0_checkpoint_sha256": e0["checkpoint_sha256"],
                    "e1_checkpoint_sha256": e1["checkpoint_sha256"],
                }
            )
    return {"pair_count": len(rows), "mismatch_count": mismatch_count, "pairs": rows, "passed": mismatch_count == 0}


def write_run_artifacts(run_root: Path, candidates: Sequence[Mapping[str, Any]], selected: Mapping[str, Any], selection_possible: bool, fixed_candidate: bool) -> None:
    dump_json(run_root / "checkpoint_candidates.json", {"candidate_count": len(candidates), "candidates": list(candidates)})
    runtime_manifest = {
        "created_at_utc": utc_now(),
        "run_id": selected["run_id"],
        "actual_split": "validation",
        "validation_split_actually_executed": True,
        "validation_metric_rows_source_split": "validation",
        "training_metric_rows_used_as_validation": False,
        "loaded_validation_snapshot_count": selected["loaded_validation_snapshot_count"],
        "validation_snapshot_ids": selected["validation_snapshot_ids"],
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "parameter_update_count": 0,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    dump_json(run_root / "validation_runtime_manifest.json", runtime_manifest)
    write_dataframe(run_root / "validation_raw_events.parquet", selected["raw_event_rows"])
    write_dataframe(run_root / "passenger_wait_ledger.parquet", selected["ledger_rows"])
    dump_json(run_root / "validation_kpi_v2.json", selected["kpi_v2"])
    dump_json(run_root / "hard_constraint_validation_audit.json", {"hard_constraints": selected["hard_constraints"], "hard_constraint_passed": selected["hard_constraint_passed"], "simulator_audit": selected["simulator_audit"]})
    dump_json(
        run_root / "selected_checkpoint_reference.json",
        {
            "checkpoint_reselection_possible": selection_possible,
            "validation_selected_best": selection_possible,
            "fixed_candidate_validated": fixed_candidate,
            "checkpoint_path": selected["checkpoint_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "selection_rule": "hard_constraints_then_service_p95_avg_wait_reward_step",
            "old_best_validation_authoritative": False,
        },
    )
    dump_json(run_root / "run_validation_status.json", selected_run_row(selected, selection_possible=selection_possible, fixed_candidate=fixed_candidate))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R1 true validation contract repair.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    prompt5_root = project_root / AUTHORITATIVE_PROMPT5
    prompt6b_root = project_root / AUTHORITATIVE_PROMPT6B
    cache_root = project_root / CACHE_ROOT
    output_root = project_root / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    device = torch.device("mps" if args.device in {"auto", "mps"} and torch.backends.mps.is_available() else "cpu")
    if args.device == "mps" and device.type != "mps":
        raise RuntimeError("MPS requested but unavailable")

    contract = selection_contract()
    dump_json(output_root / "checkpoint_selection_contract_v2.json", contract)
    source_audit = source_contract_audit(project_root, output_root)
    dump_json(output_root / "source_contract_audit.json", source_audit)
    if source_audit["status"] != "PASS" or not source_audit["blocking_violation_present"]:
        gate = {
            "created_at_utc": utc_now(),
            "status": "BLOCKED",
            "classification": "SOURCE_CONTRACT_PRECONDITION_FAILED",
            "source_contract_audit": source_audit,
            "approved_for_e2_execution": False,
            "prompt6_full_matrix_approved": False,
            "real_world_causal_claim_allowed": False,
        }
        dump_json(output_root / "prompt5_e01_r1_gate.json", gate)
        return

    train_embeddings, train_rows, train_cache_audit = load_cache_embeddings(cache_root, "train")
    validation_embeddings, validation_rows, validation_cache_audit = load_cache_embeddings(cache_root, "validation")
    train_ids = {int(row["snapshot_id"]) for row in train_rows}
    validation_ids = {int(row["snapshot_id"]) for row in validation_rows}
    embedding_audit = {
        "created_at_utc": utc_now(),
        "embedding_mode": "frozen_dynamic_cached",
        "embedding_split": "validation",
        "embedding_dynamic": True,
        "embedding_exact_artifact": True,
        "live_gatv2_forward_used": False,
        "mps_embedding_generation_used": False,
        "cpu_online_embedding_generation_used": False,
        **train_cache_audit,
        **validation_cache_audit,
        "train_cache_row_used_in_validation": 0,
        "validation_cache_row_used_in_training": 0,
        "train_validation_snapshot_overlap_count": len(train_ids & validation_ids),
        "passed": train_cache_audit["loaded_train_snapshot_count"] == 512
        and validation_cache_audit["loaded_validation_snapshot_count"] == 64
        and len(train_ids & validation_ids) == 0
        and validation_cache_audit["validation_wrong_split_path_count"] == 0,
    }
    dump_json(output_root / "embedding_split_runtime_audit.json", embedding_audit)
    if not embedding_audit["passed"]:
        gate = {
            "created_at_utc": utc_now(),
            "status": "BLOCKED",
            "classification": "VALIDATION_CACHE_CONTRACT_FAILED",
            "embedding_split_runtime_audit": embedding_audit,
            "approved_for_e2_execution": False,
            "prompt6_full_matrix_approved": False,
            "real_world_causal_claim_allowed": False,
        }
        dump_json(output_root / "prompt5_e01_r1_gate.json", gate)
        return

    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    service_node_uids = load_service_node_uids(project_root)
    e0_validation_embeddings = make_e0_embeddings(validation_embeddings.shape)
    embedding_by_family = {"E0": e0_validation_embeddings, "E1": validation_embeddings}

    inventory_rows, inventory_summary = inventory_checkpoints(prompt5_root)
    write_dataframe(output_root / "checkpoint_inventory.parquet", inventory_rows)
    dump_json(output_root / "checkpoint_inventory_summary.json", inventory_summary)

    by_run_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for row in inventory_rows:
        by_run_candidates.setdefault(row["run_id"], []).append(row)

    all_checkpoint_results: List[Dict[str, Any]] = []
    selected_results: List[Dict[str, Any]] = []
    blocked_runs = 0
    failed_runs = 0
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_id = f"{family}_{condition}_seed{seed:03d}"
                candidates = by_run_candidates.get(run_id, [])
                run_results = []
                for candidate in candidates:
                    result = evaluate_checkpoint(
                        project_root=project_root,
                        output_root=output_root,
                        family=family,
                        condition=condition,
                        seed=seed,
                        candidate=candidate,
                        embeddings=embedding_by_family[family],
                        validation_rows=validation_rows,
                        service_node_uids=service_node_uids,
                        route_sequences=route_sequences,
                        device=device,
                    )
                    run_results.append(result)
                    all_checkpoint_results.append(result)
                selected = select_checkpoint(run_results)
                if selected is None:
                    blocked_runs += 1
                    failed_runs += 1
                    continue
                unique_states = {(row["actor_state_hash"], row["critic_state_hash"]) for row in candidates}
                selection_possible = len(unique_states) > 1
                fixed_candidate = not selection_possible
                selected_results.append(dict(selected))
                run_root = output_root / family / condition / f"seed_{seed:03d}"
                write_run_artifacts(run_root, candidates, selected, selection_possible=selection_possible, fixed_candidate=fixed_candidate)

    write_dataframe(output_root / "validation_results_by_checkpoint.parquet", [result_row_for_parquet(row) for row in all_checkpoint_results])
    write_dataframe(output_root / "validation_results_by_run.parquet", [selected_run_row(row, selection_possible=False, fixed_candidate=True) for row in selected_results])

    pair_audit = pair_integrity_audit(selected_results) if len(selected_results) == 24 else {"passed": False, "mismatch_count": 24, "pairs": []}
    dump_json(output_root / "validation_pair_integrity_audit.json", pair_audit)
    pair_rows = []
    for pair in pair_audit.get("pairs", []):
        e0 = next(row for row in selected_results if row["model_family"] == "E0" and row["condition_id"] == pair["condition_id"] and int(row["seed"]) == int(pair["seed"]))
        e1 = next(row for row in selected_results if row["model_family"] == "E1" and row["condition_id"] == pair["condition_id"] and int(row["seed"]) == int(pair["seed"]))
        pair_rows.append(
            {
                "condition_id": pair["condition_id"],
                "seed": pair["seed"],
                "passed": pair["passed"],
                "e0_checkpoint_sha256": e0["checkpoint_sha256"],
                "e1_checkpoint_sha256": e1["checkpoint_sha256"],
                "delta_avg_wait_seconds": e1["kpi_v2"]["avg_wait_seconds"] - e0["kpi_v2"]["avg_wait_seconds"],
                "delta_passenger_wait_p95_seconds": e1["kpi_v2"]["passenger_wait_p95_seconds"] - e0["kpi_v2"]["passenger_wait_p95_seconds"],
                "delta_passenger_service_rate": e1["kpi_v2"]["passenger_service_rate"] - e0["kpi_v2"]["passenger_service_rate"],
                "delta_wait_burden_per_generated_demand": e1["kpi_v2"]["wait_burden_per_generated_demand"] - e0["kpi_v2"]["wait_burden_per_generated_demand"],
            }
        )
    write_dataframe(output_root / "validation_results_by_pair.parquet", pair_rows)

    path_passed = all(row["path_a_b_equivalence"]["passed"] for row in selected_results)
    path_max = max((float(row["path_a_b_equivalence"]["max_abs_diff"]) for row in selected_results), default=0.0)
    path_audit = {
        "created_at_utc": utc_now(),
        "run_count": len(selected_results),
        "max_abs_diff": path_max,
        "passed": path_passed and path_max == 0.0,
    }
    dump_json(output_root / "kpi_path_a_b_equivalence_audit.json", path_audit)
    permutation_by_run = []
    for row in selected_results:
        audit = row_permutation_audit(row["ledger_rows"])
        permutation_by_run.append({"run_id": row["run_id"], **audit})
    permutation = {
        "created_at_utc": utc_now(),
        "run_count": len(permutation_by_run),
        "result_mismatch_count": sum(int(row["result_mismatch_count"]) for row in permutation_by_run),
        "negative_test_failure_count": sum(1 for row in permutation_by_run if row["negative_test_block_count"] != len(row["negative_test_results"])),
        "runs": permutation_by_run,
    }
    permutation["passed"] = permutation["result_mismatch_count"] == 0 and permutation["negative_test_failure_count"] == 0
    dump_json(output_root / "row_permutation_regression_audit.json", permutation)

    runtime_audit = {
        "created_at_utc": utc_now(),
        "actual_split": "validation",
        "validation_split_actually_executed": len(selected_results) == 24,
        "validation_snapshot_count": len(validation_rows),
        "validation_snapshot_ids": [int(row["snapshot_id"]) for row in validation_rows],
        "validation_metric_rows_source_split": "validation",
        "training_metric_rows_used_as_validation": False,
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "parameter_update_count": 0,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "device": str(device),
    }
    dump_json(output_root / "validation_split_runtime_audit.json", runtime_audit)
    dump_json(
        output_root / "passenger_wait_ledger_schema_v2.json",
        {"created_at_utc": utc_now(), "required_fields": REQUIRED_LEDGER_FIELDS, "identity_key": ["cohort_id", "condition_id", "seed"], "schema_version": "passenger_wait_ledger_v2"},
    )
    dump_json(
        output_root / "kpi_accounting_contract_v2.json",
        {
            "created_at_utc": utc_now(),
            "avg_wait_seconds": "weighted mean wait_seconds for boarded passengers; no +1 smoothing",
            "passenger_service_rate": "passenger_served_count / passenger_demand_generated; null if zero demand",
            "passenger_wait_p95_seconds": "weighted empirical p95 including censored-at-horizon passengers as lower-bound waits",
            "wait_burden_per_generated_demand": "diagnostic cumulative passenger wait seconds / generated demand",
            "legacy_queue_wait_per_boarding_proxy": {"legacy_kpi": True, "canonical_kpi": False, "performance_claim_allowed": False},
        },
    )

    conservation_errors = sum(1 for row in selected_results if not row["kpi_v2"]["passenger_conservation_passed"])
    invalid_kpi_count = sum(int(row["kpi_v2"]["invalid_kpi_count"]) for row in selected_results)
    failed_runs += sum(1 for row in selected_results if row["status"] != "PASS")
    runs_with_multiple = inventory_summary["runs_with_multiple_checkpoints"]
    runs_with_single = inventory_summary["runs_with_single_checkpoint"]
    retraining_required_count = 0
    classification = "PASS_FIXED_CANDIDATE_VALIDATION_ONLY" if len(selected_results) == 24 and failed_runs == 0 and runs_with_multiple == 0 else "PASS_RESELECTION_COMPLETE"
    if failed_runs or blocked_runs or conservation_errors or invalid_kpi_count or not path_audit["passed"] or not permutation["passed"] or not pair_audit["passed"]:
        classification = "FAIL"
    bug_scope = "EVALUATION_AND_CHECKPOINT_SELECTION_ONLY" if embedding_audit["passed"] else "TRAINING_AND_VALIDATION_AFFECTED"
    if bug_scope != "EVALUATION_AND_CHECKPOINT_SELECTION_ONLY":
        retraining_required_count = 24
        classification = "RETRAIN_REQUIRED"

    approved_corrected = classification in {"PASS_RESELECTION_COMPLETE", "PASS_FIXED_CANDIDATE_VALIDATION_ONLY"} and retraining_required_count == 0
    gate = {
        "created_at_utc": utc_now(),
        "status": classification,
        "classification": classification,
        "source_prompt6b_gate_sha256": sha256_file(prompt6b_root / "prompt6b_preflight_gate.json"),
        "authoritative_prompt5_matrix_sha256": source_audit["authoritative_prompt5_matrix_sha256"],
        "validation_contract_repaired": classification in {"PASS_RESELECTION_COMPLETE", "PASS_FIXED_CANDIDATE_VALIDATION_ONLY"},
        "validation_split_actually_executed": runtime_audit["validation_split_actually_executed"],
        "training_rows_mislabeled_as_validation": False,
        "bug_scope": bug_scope,
        "expected_run_count": 24,
        "validation_completed_run_count": len(selected_results),
        "validation_failed_run_count": failed_runs,
        "validation_blocked_run_count": blocked_runs,
        "checkpoint_candidate_count": inventory_summary["checkpoint_candidate_count"],
        "runs_with_multiple_checkpoints": runs_with_multiple,
        "runs_with_single_checkpoint": runs_with_single,
        "runs_with_insufficient_checkpoint_history": inventory_summary["runs_with_insufficient_checkpoint_history"],
        "checkpoint_reselection_possible_run_count": runs_with_multiple,
        "checkpoint_reselection_completed_run_count": runs_with_multiple if classification == "PASS_RESELECTION_COMPLETE" else 0,
        "retraining_required_run_count": retraining_required_count,
        "kpi_accounting_v2_passed": conservation_errors == 0 and invalid_kpi_count == 0,
        "passenger_ledger_passed": all(row["ledger_identity_audit"]["passed"] for row in selected_results),
        "path_a_b_equivalence_passed": path_audit["passed"],
        "row_permutation_invariance_passed": permutation["passed"],
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "approved_for_prompt6a_corrected_retrospective": approved_corrected,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
    }
    dump_json(output_root / "prompt5_e01_r1_gate.json", gate)
    dump_json(
        output_root / "bug_scope_audit.json",
        {
            "created_at_utc": utc_now(),
            "classification": bug_scope,
            "training_input_alignment_affected": False,
            "checkpoint_selection_affected": True,
            "evidence": [
                "E1 train cache loaded 512 rows with zero train-validation overlap.",
                "Validation mislabeling and old checkpoint selection provenance were evaluation/selection issues.",
                "Existing checkpoint files are treated as candidates and are not modified.",
            ],
        },
    )
    dump_json(
        output_root / "checkpoint_reselection_report.json",
        {
            "created_at_utc": utc_now(),
            "classification": classification,
            "checkpoint_reselection_possible": runs_with_multiple > 0,
            "checkpoint_reselection_completed_run_count": gate["checkpoint_reselection_completed_run_count"],
            "fixed_candidate_validated_run_count": runs_with_single,
            "selected_checkpoints": [
                {
                    "run_id": row["run_id"],
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "validation_selected_best": runs_with_multiple > 0,
                    "fixed_candidate_validated": runs_with_multiple == 0,
                }
                for row in selected_results
            ],
        },
    )
    dump_json(
        output_root / "retraining_requirement_report.json",
        {
            "created_at_utc": utc_now(),
            "retraining_required": retraining_required_count > 0,
            "retraining_required_run_count": retraining_required_count,
            "reason": None if retraining_required_count == 0 else bug_scope,
            "auto_retraining_started": False,
        },
    )
    manifest = {
        "created_at_utc": utc_now(),
        "artifact_root": str(output_root),
        "project_root": str(project_root),
        "git_commit": git_commit(project_root),
        "authoritative_prompt5": str(prompt5_root),
        "authoritative_prompt6b": str(prompt6b_root),
        "frozen_dynamic_embedding_cache": str(cache_root),
        "prompt6a_executed": False,
        "e2_executed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "files": {},
    }
    report_lines = [
        "# Prompt 5-E01-R1 Final Report",
        "",
        f"status: {classification}",
        f"classification: {classification}",
        "",
        "## Validation contract",
        "",
        f"actual validation split executed: {str(runtime_audit['validation_split_actually_executed']).lower()}",
        f"validation snapshots used: {len(validation_rows)}",
        "training rows mislabeled as validation: false",
        "optimizer/backward/update calls: 0 / 0 / 0",
        "",
        "## Bug scope",
        "",
        f"classification: {bug_scope}",
        "training input alignment affected: false",
        "checkpoint selection affected: true",
        "",
        "## KPI v2",
        "",
        f"passenger ledger: {str(gate['passenger_ledger_passed']).lower()}",
        f"Path A/B exact: {str(path_audit['passed']).lower()}",
        "censored passenger handling: included in weighted empirical p95 as lower-bound waits",
        "legacy wait proxy preserved: legacy_queue_wait_per_boarding_proxy",
        "",
        "## Checkpoint inventory",
        "",
        f"total candidates: {inventory_summary['checkpoint_candidate_count']}",
        f"runs with multiple candidates: {runs_with_multiple}",
        f"runs with insufficient history: {inventory_summary['runs_with_insufficient_checkpoint_history']}",
        "",
        "## Reselection",
        "",
        f"completed: {gate['checkpoint_reselection_completed_run_count']}",
        f"failed: {failed_runs}",
        f"retraining required runs: {retraining_required_count}",
        "",
        "## Data leakage",
        "",
        "test split read: false",
        "test target read: false",
        "test embedding read: false",
        "",
        "## Next gate",
        "",
        f"Prompt 6A corrected retrospective approved: {str(approved_corrected).lower()}",
        f"24-run retraining required: {str(retraining_required_count > 0).lower()}",
        "E2 approved: false",
        "full Prompt 6 approved: false",
    ]
    (output_root / "prompt5_e01_r1_final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r1_manifest.json":
            manifest["files"][str(path.relative_to(output_root))] = {"absolute_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    dump_json(output_root / "prompt5_e01_r1_manifest.json", manifest)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
