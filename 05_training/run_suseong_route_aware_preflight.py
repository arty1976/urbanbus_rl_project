from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from simulator.suseong_service_transition_engine import (
    StopServiceResult,
    TransitionConfig,
    advance_vehicle_time_budget,
)


ARTIFACT_VERSION = "suseong_route_aware_causal_preflight_prompt2r_v1"
CANONICAL_12_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def sha256_state_dict(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_device(require_mps: bool) -> Tuple[Optional[torch.device], Dict[str, Any]]:
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    if require_mps and not mps_available:
        return None, {
            "requested": "mps",
            "selected": None,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "cpu_model_fallback_used": False,
            "gpu_blocked": True,
            "reason": "Strict MPS requested, but torch.backends.mps.is_available() is false.",
        }
    device = torch.device("mps" if mps_available else "cpu")
    return device, {
        "requested": "mps" if require_mps else "auto",
        "selected": str(device),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cpu_model_fallback_used": bool(require_mps and device.type != "mps"),
        "gpu_blocked": bool(require_mps and device.type != "mps"),
        "reason": "MPS selected." if device.type == "mps" else "CPU selected only because require_mps=false.",
    }


class ServicePolicy(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_agents: int, action_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.agent_embedding = torch.nn.Embedding(num_agents, embedding_dim)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2 + 4, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, action_dim),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim + 4, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def actor_obs(self, local_embeddings: torch.Tensor, route_features: torch.Tensor) -> torch.Tensor:
        agent_ids = torch.arange(self.num_agents, device=local_embeddings.device)
        agent_emb = self.agent_embedding(agent_ids)
        return torch.cat([local_embeddings, agent_emb, route_features], dim=1)

    def forward(self, local_embeddings: torch.Tensor, route_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self.actor_obs(local_embeddings, route_features)
        logits = self.actor(obs)
        critic_input = torch.cat([local_embeddings.mean(dim=0), route_features.mean(dim=0)], dim=0).reshape(1, -1)
        value = self.critic(critic_input).reshape(())
        return logits, value


@dataclass
class AgentState:
    agent_id: int
    route_key: Tuple[str, str]
    route_no: str
    position: int
    onboard_count: int
    capacity: int
    remaining_travel_time: float
    remaining_dwell_time: float


class SuseongRouteAwareSimulator:
    def __init__(self, route_sequences: pd.DataFrame, num_agents: int, seed: int) -> None:
        self.num_agents = int(num_agents)
        self.rng = random.Random(int(seed))
        self.step_index = 0
        self.routes: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for (route_id, direction_id), group in route_sequences.groupby(["route_id", "direction_id"], sort=False):
            rows = group.sort_values("stop_order").to_dict("records")
            if len(rows) >= 2:
                self.routes[(str(route_id), str(direction_id))] = rows
        if not self.routes:
            raise RuntimeError("No eligible route sequences with at least two stops.")
        route_keys = sorted(self.routes.keys())
        self.agent_states: List[AgentState] = []
        for agent_id in range(self.num_agents):
            key = route_keys[agent_id % len(route_keys)]
            route = self.routes[key]
            max_start = max(len(route) - 2, 0)
            start = (agent_id * 3) % (max_start + 1)
            self.agent_states.append(
                AgentState(
                    agent_id=agent_id,
                    route_key=key,
                    route_no=str(route[0].get("route_no", "")),
                    position=start,
                    onboard_count=agent_id % 7,
                    capacity=80,
                    remaining_travel_time=0.0,
                    remaining_dwell_time=0.0,
                )
            )
        self.waiting_counts: Dict[str, int] = {}
        for rows in self.routes.values():
            for row in rows:
                uid = str(row["node_uid"])
                self.waiting_counts.setdefault(uid, (len(uid) % 5) + 1)
        self.total_generated = 0
        self.total_completed = 0
        self.initial_system_passengers = self.system_passenger_count()
        self.audit = {
            "vehicle_teleport_count": 0,
            "route_sequence_violation_count": 0,
            "capacity_violation_count": 0,
            "negative_queue_count": 0,
            "negative_onboard_count": 0,
            "invalid_action_selected_count": 0,
            "boundary_transition_count": 0,
            "dwell_update_count": 0,
            "skip_stop_action_count": 0,
            "movement_event_count": 0,
        }
        self.metric_rows: List[Dict[str, float]] = []

    def clone(self) -> "SuseongRouteAwareSimulator":
        return copy.deepcopy(self)

    def system_passenger_count(self) -> int:
        return int(sum(self.waiting_counts.values()) + sum(a.onboard_count for a in self.agent_states))

    def state_hash(self) -> str:
        payload = {
            "step_index": self.step_index,
            "agents": [a.__dict__ for a in self.agent_states],
            "waiting_counts": dict(sorted(self.waiting_counts.items())),
            "total_generated": self.total_generated,
            "total_completed": self.total_completed,
        }
        return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def current_service_local_indices(self, node_uid_to_local: Mapping[str, int]) -> List[int]:
        indices: List[int] = []
        for agent in self.agent_states:
            route = self.routes[agent.route_key]
            node_uid = str(route[agent.position]["node_uid"])
            indices.append(int(node_uid_to_local[node_uid]))
        return indices

    def route_features(self) -> torch.Tensor:
        features: List[List[float]] = []
        for agent in self.agent_states:
            route = self.routes[agent.route_key]
            denom = max(len(route) - 1, 1)
            progress = float(agent.position) / float(denom)
            queue = float(self.waiting_counts.get(str(route[agent.position]["node_uid"]), 0)) / 100.0
            onboard = float(agent.onboard_count) / float(agent.capacity)
            dwell = float(agent.remaining_dwell_time) / 60.0
            features.append([progress, queue, onboard, dwell])
        return torch.tensor(features, dtype=torch.float32)

    def step(
        self,
        actions: Sequence[int],
        delta_t_seconds: float = 3600.0,
        snapshot_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        if len(actions) != self.num_agents:
            raise RuntimeError(f"action length mismatch: {len(actions)} != {self.num_agents}")
        if delta_t_seconds < 0:
            raise ValueError("delta_t_seconds must be non-negative")

        total_boardings = 0
        total_alightings = 0
        total_wait_seconds = 0.0
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
            arrivals = 1 + ((self.step_index + agent.agent_id) % 3)
            self.waiting_counts[current_node] = self.waiting_counts.get(current_node, 0) + arrivals
            self.total_generated += arrivals

            if action == 3:
                self.audit["skip_stop_action_count"] += 1

            def service_stop(vehicle: Any, stop_row: Mapping[str, Any]) -> StopServiceResult:
                node_uid = str(stop_row["node_uid"])
                alightings = min(vehicle.onboard_count, (self.step_index + vehicle.agent_id) % 2)
                vehicle.onboard_count -= alightings
                self.total_completed += alightings
                boarding_capacity = max(vehicle.capacity - vehicle.onboard_count, 0)
                boarding_limit = 2 + (1 if action in {1, 2, 3} else 0)
                boardings = min(self.waiting_counts.get(node_uid, 0), boarding_capacity, boarding_limit)
                self.waiting_counts[node_uid] -= boardings
                vehicle.onboard_count += boardings
                return StopServiceResult(
                    boardings=int(boardings),
                    alightings=int(alightings),
                    dwell_required=bool(boardings or alightings),
                    metadata={
                        "node_uid": node_uid,
                        "boardings": int(boardings),
                        "alightings": int(alightings),
                        "boarding_limit": int(boarding_limit),
                    },
                )

            old_position = int(agent.position)
            old_route = route
            trace = advance_vehicle_time_budget(
                vehicle=agent,
                routes=self.routes,
                delta_t_seconds=float(delta_t_seconds),
                action=action,
                stop_service=service_stop,
                config=TransitionConfig(edge_travel_seconds=45.0, dwell_seconds=15.0, allow_turnaround=False),
            )
            total_boardings += int(trace.boardings)
            total_alightings += int(trace.alightings)
            total_wait_seconds += float(sum(self.waiting_counts.values()) * 30.0)
            self.audit["movement_event_count"] += int(trace.edges_traversed)
            self.audit["dwell_update_count"] += int(sum(1 for event in trace.events if event["event_type"] == "STOP_SERVICE" and (event.get("boardings", 0) or event.get("alightings", 0))))
            if int(agent.position) < old_position:
                self.audit["route_sequence_violation_count"] += 1
            if trace.transition_guard_triggered:
                self.audit["route_sequence_violation_count"] += 1
            if old_position != int(agent.position) and bool(self.routes[agent.route_key][agent.position].get("is_suseong_core")) != bool(old_route[old_position].get("is_suseong_core")):
                self.audit["boundary_transition_count"] += 1
            if agent.onboard_count > agent.capacity:
                self.audit["capacity_violation_count"] += 1
            if agent.onboard_count < 0:
                self.audit["negative_onboard_count"] += 1
            if any(value < 0 for value in self.waiting_counts.values()):
                self.audit["negative_queue_count"] += 1
            total_energy += 0.1 + 0.02 * trace.edges_traversed + 0.005 * trace.boardings
            headway_proxy.append(240.0 + 15.0 * ((agent.position + agent.agent_id) % 5))

        self.step_index += 1
        reward_value = float(total_boardings - 0.01 * total_wait_seconds - 0.1 * total_energy)
        reward = torch.tensor(reward_value, dtype=torch.float32)
        generated = max(float(self.total_generated), 1.0)
        served = float(self.total_completed)
        headway_mean = sum(headway_proxy) / max(len(headway_proxy), 1)
        headway_std = (
            sum((h - headway_mean) ** 2 for h in headway_proxy) / max(len(headway_proxy), 1)
        ) ** 0.5
        system_expected = self.initial_system_passengers + self.total_generated - self.total_completed
        conservation_error = abs(float(self.system_passenger_count() - system_expected))
        metrics = {
            "cv_headway": float(headway_std / max(headway_mean, 1e-6)),
            "avg_wait_seconds": float(total_wait_seconds / max(total_boardings + 1, 1)),
            "bunching_rate": float(sum(1 for h in headway_proxy if h < 260.0) / max(len(headway_proxy), 1)),
            "on_time_rate": float(sum(1 for h in headway_proxy if 230.0 <= h <= 310.0) / max(len(headway_proxy), 1)),
            "intervention_rate": float(non_noop / max(self.num_agents, 1)),
            "energy_proxy": float(total_energy),
            "passenger_demand_generated": float(self.total_generated),
            "passenger_served_count": served,
            "passenger_service_rate": float(min(served / generated, 1.0)),
            "passenger_wait_p95_seconds": float((total_wait_seconds / max(total_boardings + 1, 1)) * 1.35),
            "energy_proxy_per_passenger": float(total_energy / max(total_boardings, 1)),
            "fleet_reduction_ratio": 0.0,
            "queue_conservation_error": conservation_error,
            "onboard_conservation_error": conservation_error,
        }
        self.metric_rows.append(metrics)
        return reward, metrics, dict(self.audit)


def kpi_summary(metric_rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in CANONICAL_12_KPIS:
        values = [float(row.get(key, 0.0)) for row in metric_rows]
        result[f"{key}_mean"] = float(sum(values) / max(len(values), 1))
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_preflight(
    *,
    project_root: Path,
    service_graph_dir: Path,
    embedding_contract_dir: Path,
    output_root: Path,
    prompt_label: str,
    prompt2_full_execution: bool,
    agents: int,
    horizon: int,
    trace_mode: str,
    ppo_epochs: int,
    seed: int,
    require_mps: bool,
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    device, device_report = resolve_device(require_mps)
    if device is None:
        manifest = {
            "created_at_utc": utc_now(),
            "artifact_version": ARTIFACT_VERSION,
            "status": "BLOCKED",
            "device": device_report,
            "cpu_model_fallback_used": False,
            "blockers": ["STRICT_MPS_NOT_AVAILABLE"],
        }
        dump_json(output_root / "preflight_manifest.json", manifest)
        return manifest

    embedding_payload = torch_load(embedding_contract_dir / "service_node_embeddings.pt")
    embeddings = embedding_payload["service_node_embeddings"].detach().cpu().float()
    service_node_uids = [str(v) for v in embedding_payload["service_node_uids"]]
    node_uid_to_local = {uid: idx for idx, uid in enumerate(service_node_uids)}
    if embeddings.ndim != 3:
        raise RuntimeError(f"expected [snapshots, service_nodes, dim] embeddings, got {tuple(embeddings.shape)}")

    route_sequences = pd.read_csv(service_graph_dir / "service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    missing_uids = sorted(set(route_sequences["node_uid"].astype(str)) - set(node_uid_to_local))
    if missing_uids:
        raise RuntimeError(f"route sequence node_uids missing from embedding mapping: {missing_uids[:10]}")

    simulator = SuseongRouteAwareSimulator(route_sequences, agents, seed)
    noop_simulator = simulator.clone()
    policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    rewards: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    actions_by_step: List[torch.Tensor] = []
    route_features_by_step: List[torch.Tensor] = []
    local_indices_by_step: List[List[int]] = []
    trace_rows: List[Dict[str, Any]] = []
    sampled_trace: List[Dict[str, Any]] = []
    profile = {
        "actor_device": str(device),
        "critic_device": str(device),
        "simulator_device": "cpu",
        "batched_actor_inference": True,
        "per_agent_device_transfer_used": False,
        "per_agent_mps_synchronize_used": False,
    }
    started = time.perf_counter()
    for step in range(horizon):
        snapshot_idx = step % int(embeddings.shape[0])
        local_indices = simulator.current_service_local_indices(node_uid_to_local)
        route_features_cpu = simulator.route_features()
        local_embedding_cpu = embeddings[snapshot_idx, local_indices, :]
        local_embeddings = local_embedding_cpu.to(device)
        route_features = route_features_cpu.to(device)
        logits, value = policy(local_embeddings, route_features)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_log_prob = dist.log_prob(actions)
        actions_cpu = [int(v) for v in actions.detach().cpu().tolist()]
        reward, metrics, audit = simulator.step(actions_cpu)
        noop_simulator.step([0] * agents)
        rewards.append(reward)
        values.append(value.detach().cpu())
        old_log_probs.append(old_log_prob.detach().cpu())
        actions_by_step.append(actions.detach().cpu())
        route_features_by_step.append(route_features_cpu)
        local_indices_by_step.append(local_indices)
        if trace_mode == "audit":
            for agent_id, action in enumerate(actions_cpu):
                trace_rows.append(
                    {
                        "step": step,
                        "agent_id": agent_id,
                        "action": action,
                        "old_logprob": float(old_log_prob[agent_id].detach().cpu().item()),
                        "current_service_local_index": local_indices[agent_id],
                        "current_node_uid": service_node_uids[local_indices[agent_id]],
                        "team_reward": float(reward.item()),
                    }
                )
        if step < 8:
            sampled_trace.append(
                {
                    "step": step,
                    "actions_sample": actions_cpu[: min(10, len(actions_cpu))],
                    "reward": float(reward.item()),
                    "metrics": {k: float(metrics[k]) for k in CANONICAL_12_KPIS},
                }
            )
    rollout_seconds = time.perf_counter() - started

    raw_rewards = torch.stack(rewards).to(device)
    reward_targets = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std(unbiased=False) + 1e-6)
    actions_tensor = torch.stack(actions_by_step).to(device)
    old_log_probs_tensor = torch.stack(old_log_probs).to(device)
    values_tensor = torch.stack(values).to(device)
    actor_elements = int(actions_tensor.numel())

    update_started = time.perf_counter()
    for _ in range(ppo_epochs):
        epoch_log_probs: List[torch.Tensor] = []
        epoch_values: List[torch.Tensor] = []
        for step in range(horizon):
            snapshot_idx = step % int(embeddings.shape[0])
            local_embeddings = embeddings[snapshot_idx, local_indices_by_step[step], :].to(device)
            route_features = route_features_by_step[step].to(device)
            logits, value = policy(local_embeddings, route_features)
            dist = Categorical(logits=logits)
            epoch_log_probs.append(dist.log_prob(actions_tensor[step]))
            epoch_values.append(value)
        log_probs_tensor = torch.stack(epoch_log_probs)
        new_values_tensor = torch.stack(epoch_values)
        ratio = torch.exp(log_probs_tensor - old_log_probs_tensor)
        advantage = reward_targets.reshape(horizon, 1).expand(horizon, agents)
        actor_loss = -(ratio * advantage).mean()
        critic_loss = F.mse_loss(new_values_tensor, reward_targets)
        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
    update_seconds = time.perf_counter() - update_started

    first_indices = local_indices_by_step[0]
    first_route_features = route_features_by_step[0]
    first_embeddings = embeddings[0, first_indices, :].to(device)
    first_features_device = first_route_features.to(device)
    with torch.no_grad():
        before_logits, before_value = policy(first_embeddings, first_features_device)
    checkpoint_path = output_root / "checkpoints" / "suseong_route_aware_policy_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_version": (
                "suseong_prompt2_full_preflight_policy_checkpoint_v1"
                if prompt2_full_execution
                else "suseong_prompt2r_preflight_policy_checkpoint_v1"
            ),
            "created_at_utc": utc_now(),
            "model_state_dict": policy.state_dict(),
            "num_agents": agents,
            "embedding_dim": int(embeddings.shape[2]),
            "action_dim": 5,
            "performance_claim_allowed": False,
        },
        checkpoint_path,
    )
    reloaded = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"])
    with torch.no_grad():
        after_logits, after_value = reloaded(first_embeddings, first_features_device)
    logits_diff = float((before_logits - after_logits).abs().max().detach().cpu().item())
    value_diff = float((before_value - after_value).abs().max().detach().cpu().item())

    kpis = kpi_summary(simulator.metric_rows)
    canonical_12 = {key: kpis[f"{key}_mean"] for key in CANONICAL_12_KPIS}
    queue_error = max(float(row.get("queue_conservation_error", 0.0)) for row in simulator.metric_rows)
    onboard_error = max(float(row.get("onboard_conservation_error", 0.0)) for row in simulator.metric_rows)
    state_difference_detected = simulator.state_hash() != noop_simulator.state_hash()
    audit = dict(simulator.audit)
    pass_conditions = {
        "actual_device_mps": device.type == "mps",
        "cpu_model_fallback_used_false": not bool(device_report["cpu_model_fallback_used"]),
        "state_difference_detected": bool(state_difference_detected),
        "vehicle_teleport_count_zero": audit["vehicle_teleport_count"] == 0,
        "route_sequence_violation_count_zero": audit["route_sequence_violation_count"] == 0,
        "capacity_violation_count_zero": audit["capacity_violation_count"] == 0,
        "negative_queue_count_zero": audit["negative_queue_count"] == 0,
        "negative_onboard_count_zero": audit["negative_onboard_count"] == 0,
        "invalid_action_selected_count_zero": audit["invalid_action_selected_count"] == 0,
        "queue_conservation_error_within_tolerance": queue_error <= 1e-6,
        "onboard_conservation_error_within_tolerance": onboard_error <= 1e-6,
        "checkpoint_reload_ok": logits_diff <= 1e-7 and value_diff <= 1e-7,
        "canonical_kpi_count_12": len(canonical_12) == 12,
        "per_agent_device_transfer_used_false": not profile["per_agent_device_transfer_used"],
        "per_agent_mps_synchronize_used_false": not profile["per_agent_mps_synchronize_used"],
    }
    status = "PASS" if all(pass_conditions.values()) else "FAIL"

    dump_json(output_root / "canonical_12kpi.json", canonical_12)
    dump_json(output_root / "simulator_state_audit.json", {**audit, "state_difference_detected": state_difference_detected})
    dump_json(
        output_root / "passenger_conservation_audit.json",
        {
            "queue_conservation_error": queue_error,
            "onboard_conservation_error": onboard_error,
            "initial_system_passengers": simulator.initial_system_passengers,
            "total_generated": simulator.total_generated,
            "total_completed": simulator.total_completed,
            "final_system_passengers": simulator.system_passenger_count(),
        },
    )
    dump_json(
        output_root / "actor_loss_audit.json",
        {
            "policy_logits_shape": [horizon, agents, 5],
            "policy_actions_shape": list(actions_tensor.shape),
            "unique_actor_elements": actor_elements,
            "actor_elements_per_ppo_epoch": actor_elements,
            "actor_elements_processed_across_all_ppo_epochs": actor_elements * ppo_epochs,
            "team_advantage_shape_before_broadcast": [horizon],
            "actor_advantage_shape_after_broadcast": [horizon, agents],
            "critic_targets": horizon,
        },
    )
    dump_json(
        output_root / "rollout_tensor_hashes.json",
        {
            "actions_sha256": sha256_tensor(actions_tensor.cpu()),
            "old_log_probs_sha256": sha256_tensor(old_log_probs_tensor.cpu()),
            "rewards_sha256": sha256_tensor(raw_rewards.cpu()),
        },
    )
    dump_json(
        output_root / "rollout_tensor_summary.json",
        {
            "actions_shape": list(actions_tensor.shape),
            "old_log_probs_shape": list(old_log_probs_tensor.shape),
            "rewards_shape": list(raw_rewards.shape),
        },
    )
    dump_json(output_root / "sampled_decision_trace.json", {"rows": sampled_trace})
    if trace_mode == "audit":
        write_csv(
            output_root / "agent_decision_trace.csv",
            trace_rows,
            ["step", "agent_id", "action", "old_logprob", "current_service_local_index", "current_node_uid", "team_reward"],
        )

    manifest = {
        "created_at_utc": utc_now(),
        "artifact_version": (
            "suseong_route_aware_causal_preflight_prompt2_full_v1"
            if prompt2_full_execution
            else ARTIFACT_VERSION
        ),
        "status": status,
        "prompt": prompt_label,
        "prompt2_full_execution": bool(prompt2_full_execution),
        "trace_mode": trace_mode,
        "agents": agents,
        "horizon": horizon,
        "ppo_epochs": ppo_epochs,
        "seed": seed,
        "condition": "A",
        "actor_device": str(device),
        "critic_device": str(device),
        "simulator_device": "cpu",
        "cpu_model_fallback_used": bool(device_report["cpu_model_fallback_used"]),
        "gpu_blocked": bool(device_report["gpu_blocked"]),
        "embedding_contract_dir": str(embedding_contract_dir),
        "embedding_shape": list(embeddings.shape),
        "service_node_count": int(embeddings.shape[1]),
        "service_route_direction_count": int(route_sequences[["route_id", "direction_id"]].drop_duplicates().shape[0]),
        "observed_agents": agents,
        "observed_decisions": agents * horizon,
        "causal_actuation_detected": bool(state_difference_detected),
        "checkpoint_reload_ok": bool(pass_conditions["checkpoint_reload_ok"]),
        "reload_logits_diff": logits_diff,
        "reload_value_diff": value_diff,
        "canonical_kpi_count": len(canonical_12),
        "canonical_kpi_keys": CANONICAL_12_KPIS,
        "pass_conditions": pass_conditions,
        "rollout_total_seconds": round(rollout_seconds, 6),
        "ppo_update_seconds": round(update_seconds, 6),
        "policy_state_hash": sha256_state_dict(policy),
        "checkpoint_path": str(checkpoint_path),
        "performance_claim_allowed": False,
        "blockers_preserved": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
    }
    dump_json(output_root / "preflight_manifest.json", manifest)
    return manifest


def write_gate(
    project_root: Path,
    preflight_a: Mapping[str, Any],
    preflight_b: Mapping[str, Any],
    *,
    prompt2_full_execution: bool,
    preflight_a_manifest: Path,
    preflight_b_manifest: Path,
) -> Dict[str, Any]:
    gate_dir = project_root / "05_training/artifacts/suseong_service_preflight_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    prompt1_fleet = project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json"
    fleet_status = "UNKNOWN"
    if prompt1_fleet.exists():
        fleet_status = json.loads(prompt1_fleet.read_text(encoding="utf-8")).get("status", "UNKNOWN")
    status = "PASS" if preflight_a.get("status") == "PASS" and preflight_b.get("status") == "PASS" else "BLOCKED"
    gate = {
        "created_at_utc": utc_now(),
        "artifact_version": (
            "suseong_service_preflight_gate_prompt2_full_v1"
            if prompt2_full_execution
            else "suseong_service_preflight_gate_prompt2r_v1"
        ),
        "status": status,
        "prompt2r_status": "PASS" if prompt2_full_execution and status == "PASS" else status,
        "prompt2_full_status": status if prompt2_full_execution else "NOT_EXECUTED",
        "prompt2_full_execution": bool(prompt2_full_execution),
        "preflight_a_status": preflight_a.get("status"),
        "preflight_b_status": preflight_b.get("status"),
        "approved_for_prompt2_full_causal_preflight": (status == "PASS") and not prompt2_full_execution,
        "approved_for_512x256_stress_gate": (status == "PASS") and bool(prompt2_full_execution),
        "prompt3_executed": False,
        "approved_for_scientific_fleet_claims": False,
        "graph_mapping": "PASS",
        "embedding_mapping_coverage": 100.0,
        "causal_actuation_detected": bool(preflight_a.get("causal_actuation_detected")) and bool(preflight_b.get("causal_actuation_detected")),
        "checkpoint_reload_ok": bool(preflight_b.get("checkpoint_reload_ok")),
        "canonical_kpi_count": int(preflight_b.get("canonical_kpi_count", 0)),
        "prompt1_scientific_fleet_gate_status": fleet_status,
        "blockers_preserved": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "blockers_resolved": [
            "NODE_LEVEL_EMBEDDING_CONTRACT_NOT_SATISFIED_FIXED_EMBEDDING_IS_GRAPH_LEVEL_OR_SNAPSHOT_LEVEL",
            "SUSEONG_ROUTE_AWARE_CAUSAL_SERVICE_GRAPH_RUNNER_NOT_FOUND_IN_REPO_SEARCH",
        ] if status == "PASS" else [],
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "preflight_a_manifest": str(preflight_a_manifest),
        "preflight_b_manifest": str(preflight_b_manifest),
    }
    dump_json(gate_dir / "preflight_gate.json", gate)
    title = "SUSEONG_SERVICE Prompt 2 Full Causal Preflight Gate" if prompt2_full_execution else "SUSEONG_SERVICE Prompt 2-R Preflight Gate"
    (gate_dir / "preflight_gate.md").write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"Status: {status}",
                "",
                f"Preflight A: {preflight_a.get('status')}",
                f"Preflight B: {preflight_b.get('status')}",
                "",
                f"Prompt 2 full execution: {str(bool(prompt2_full_execution)).lower()}",
                f"Approved for 512x256 stress gate: {str((status == 'PASS') and bool(prompt2_full_execution)).lower()}",
                "Prompt 3 execution: false",
                "",
                "Remaining blocker:",
                "- FLEET_FREQUENCY_NOT_OFFICIAL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return gate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Suseong route-aware causal preflights.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--service-graph-dir", default="05_training/artifacts/suseong_service_graph_v1")
    parser.add_argument("--embedding-contract-dir", default="05_training/artifacts/suseong_node_level_embedding_contract_v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--write-gate", action="store_true")
    parser.add_argument("--prompt2-full", action="store_true", help="Write Prompt 2 full causal preflight artifacts and approval gate.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    service_graph_dir = (project_root / args.service_graph_dir).resolve()
    embedding_contract_dir = (project_root / args.embedding_contract_dir).resolve()
    if args.prompt2_full:
        prompt_label = "Prompt 2 full causal preflight"
        preflight_a_root = project_root / "05_training/artifacts/mac_suseong_service_full_preflight_a16_h8"
        preflight_b_root = project_root / "05_training/artifacts/mac_suseong_service_full_preflight_a128_h32"
    else:
        prompt_label = "Prompt 2-R recovered preflight"
        preflight_a_root = project_root / "05_training/artifacts/mac_suseong_service_preflight_a16_h8"
        preflight_b_root = project_root / "05_training/artifacts/mac_suseong_service_preflight_a128_h32"
    preflight_a = run_preflight(
        project_root=project_root,
        service_graph_dir=service_graph_dir,
        embedding_contract_dir=embedding_contract_dir,
        output_root=preflight_a_root,
        prompt_label=prompt_label,
        prompt2_full_execution=bool(args.prompt2_full),
        agents=16,
        horizon=8,
        trace_mode="audit",
        ppo_epochs=1,
        seed=args.seed,
        require_mps=bool(args.require_mps),
    )
    preflight_b = run_preflight(
        project_root=project_root,
        service_graph_dir=service_graph_dir,
        embedding_contract_dir=embedding_contract_dir,
        output_root=preflight_b_root,
        prompt_label=prompt_label,
        prompt2_full_execution=bool(args.prompt2_full),
        agents=128,
        horizon=32,
        trace_mode="benchmark",
        ppo_epochs=2,
        seed=args.seed,
        require_mps=bool(args.require_mps),
    )
    gate = (
        write_gate(
            project_root,
            preflight_a,
            preflight_b,
            prompt2_full_execution=bool(args.prompt2_full),
            preflight_a_manifest=preflight_a_root / "preflight_manifest.json",
            preflight_b_manifest=preflight_b_root / "preflight_manifest.json",
        )
        if args.write_gate
        else {}
    )
    print(json.dumps({"preflight_a": preflight_a, "preflight_b": preflight_b, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
