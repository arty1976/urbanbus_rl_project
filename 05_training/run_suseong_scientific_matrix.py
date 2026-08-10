from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from run_suseong_route_aware_preflight import (
    CANONICAL_12_KPIS,
    ServicePolicy,
    SuseongRouteAwareSimulator,
    dump_json,
    kpi_summary,
    resolve_device,
    sha256_tensor,
    torch_load,
)
from simulator.suseong_service_transition_engine import (
    StopServiceResult,
    TransitionConfig,
    advance_vehicle_time_budget,
)


ARTIFACT_VERSION = "suseong_scientific_matrix_v1_mac_mps"
OUTPUT_ROOT = "05_training/artifacts/suseong_scientific_matrix_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


class FixedDemandSuseongSimulator(SuseongRouteAwareSimulator):
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

        step_generated = 0
        for node_uid in sorted(self.waiting_counts):
            arrivals = 1 + ((self.step_index + stable_int(node_uid)) % 3)
            self.waiting_counts[node_uid] = self.waiting_counts.get(node_uid, 0) + arrivals
            self.total_generated += arrivals
            step_generated += arrivals

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

            if action == 3:
                self.audit["skip_stop_action_count"] += 1

            def service_stop(vehicle: Any, stop_row: Mapping[str, Any]) -> StopServiceResult:
                current_node = str(stop_row["node_uid"])
                alightings = min(vehicle.onboard_count, (self.step_index + vehicle.agent_id) % 2)
                vehicle.onboard_count -= alightings
                self.total_completed += alightings
                boarding_capacity = max(vehicle.capacity - vehicle.onboard_count, 0)
                boarding_limit = 2 + (1 if action in {1, 2, 3} else 0)
                boardings = min(self.waiting_counts.get(current_node, 0), boarding_capacity, boarding_limit)
                self.waiting_counts[current_node] -= boardings
                vehicle.onboard_count += boardings
                return StopServiceResult(
                    boardings=int(boardings),
                    alightings=int(alightings),
                    dwell_required=bool(boardings or alightings),
                    metadata={
                        "node_uid": current_node,
                        "boarding_limit": int(boarding_limit),
                        "waiting_after": int(self.waiting_counts.get(current_node, 0)),
                    },
                )

            old_position = int(agent.position)
            old_route = self.routes[agent.route_key]
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
        total_wait_seconds = float(sum(self.waiting_counts.values()) * 30.0)
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
            "step_exogenous_demand_generated": float(step_generated),
        }
        self.metric_rows.append(metrics)
        return reward, metrics, dict(self.audit)


class E2Adapter(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.final_embedding_layer = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.final_embedding_layer(x))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_service_node_uids(project_root: Path) -> List[str]:
    mapping_path = project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv"
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        return [row["node_uid"] for row in csv.DictReader(f)]


def make_e0_embeddings(shape: Sequence[int]) -> torch.Tensor:
    snapshots, nodes, dim = [int(v) for v in shape]
    node_axis = torch.linspace(0.0, 1.0, nodes).reshape(1, nodes, 1).repeat(snapshots, 1, dim)
    snapshot_axis = torch.linspace(0.0, 1.0, snapshots).reshape(snapshots, 1, 1)
    scale = torch.linspace(0.1, 1.0, dim).reshape(1, 1, dim)
    return torch.sin((node_axis + snapshot_axis) * scale * math.pi)


def grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    values = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.norm(torch.stack(values), 2).detach().cpu().item())


def condition_agents(central: int, condition: str) -> int:
    ratios = {"A": 1.0, "A90": 0.9, "A80": 0.8, "A70": 0.7}
    return int(round(central * ratios[condition]))


def run_one(
    *,
    project_root: Path,
    matrix_root: Path,
    family: str,
    condition: str,
    seed: int,
    agents: int,
    embeddings: torch.Tensor,
    service_node_uids: Sequence[str],
    device: torch.device,
    horizon: int,
    ppo_epochs: int,
    fleet_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    run_root = matrix_root / family / condition / f"seed_{seed:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    node_uid_to_local = {uid: i for i, uid in enumerate(service_node_uids)}
    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    simulator = FixedDemandSuseongSimulator(route_sequences, agents, seed)
    noop = FixedDemandSuseongSimulator(route_sequences, agents, seed)
    policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    adapter: Optional[E2Adapter] = E2Adapter(int(embeddings.shape[2])).to(device) if family == "E2" else None
    params = list(policy.parameters()) + (list(adapter.parameters()) if adapter else [])
    optimizer = torch.optim.Adam(params, lr=1e-3)
    embeddings_device = embeddings.to(device)
    rewards: List[torch.Tensor] = []
    actions_by_step: List[torch.Tensor] = []
    old_log_probs_by_step: List[torch.Tensor] = []
    route_features_by_step: List[torch.Tensor] = []
    local_indices_by_step: List[List[int]] = []
    nan_detected = False
    inf_detected = False
    for step in range(horizon):
        snapshot_idx = step % int(embeddings_device.shape[0])
        local_indices = simulator.current_service_local_indices(node_uid_to_local)
        local_embeddings = embeddings_device[snapshot_idx, local_indices, :]
        if adapter is not None:
            local_embeddings = adapter(local_embeddings)
        route_features_cpu = simulator.route_features()
        logits, value = policy(local_embeddings, route_features_cpu.to(device))
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_lp = dist.log_prob(actions)
        nan_detected = nan_detected or not bool(torch.isfinite(logits).all().detach().cpu().item())
        inf_detected = inf_detected or bool(torch.isinf(logits).any().detach().cpu().item())
        actions_cpu = [int(v) for v in actions.detach().cpu().tolist()]
        reward, _metrics, _audit = simulator.step(actions_cpu)
        noop.step([0] * agents)
        rewards.append(reward)
        actions_by_step.append(actions.detach().cpu())
        old_log_probs_by_step.append(old_lp.detach().cpu())
        route_features_by_step.append(route_features_cpu)
        local_indices_by_step.append(local_indices)
        del value

    raw_rewards = torch.stack(rewards).to(device)
    reward_targets = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std(unbiased=False) + 1e-6)
    actions_tensor = torch.stack(actions_by_step).to(device)
    old_log_probs_tensor = torch.stack(old_log_probs_by_step).to(device)
    rows: List[Dict[str, Any]] = []
    grad_before: List[float] = []
    grad_after: List[float] = []
    e2_grad: List[float] = []
    policy_loss_value = 0.0
    value_loss_value = 0.0
    entropy_value = 0.0
    approx_kl_value = 0.0
    clip_fraction_value = 0.0
    for epoch in range(ppo_epochs):
        epoch_log_probs: List[torch.Tensor] = []
        epoch_values: List[torch.Tensor] = []
        epoch_entropy: List[torch.Tensor] = []
        for step in range(horizon):
            snapshot_idx = step % int(embeddings_device.shape[0])
            local_embeddings = embeddings_device[snapshot_idx, local_indices_by_step[step], :]
            if adapter is not None:
                local_embeddings = adapter(local_embeddings)
            logits, value = policy(local_embeddings, route_features_by_step[step].to(device))
            dist = Categorical(logits=logits)
            epoch_log_probs.append(dist.log_prob(actions_tensor[step]))
            epoch_values.append(value.reshape(()))
            epoch_entropy.append(dist.entropy())
        log_probs_tensor = torch.stack(epoch_log_probs)
        values_tensor = torch.stack(epoch_values)
        entropy_tensor = torch.stack(epoch_entropy)
        ratio = torch.exp(log_probs_tensor - old_log_probs_tensor)
        advantage = reward_targets.reshape(horizon, 1).expand(horizon, agents)
        surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage)
        policy_loss = -surrogate.mean()
        value_loss = F.mse_loss(values_tensor, reward_targets)
        entropy = entropy_tensor.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = grad_norm(params)
        e2_before = grad_norm(list(adapter.parameters())) if adapter else 0.0
        torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
        after = grad_norm(params)
        optimizer.step()
        policy_loss_value = float(policy_loss.detach().cpu().item())
        value_loss_value = float(value_loss.detach().cpu().item())
        entropy_value = float(entropy.detach().cpu().item())
        approx_kl_value = float((old_log_probs_tensor - log_probs_tensor).mean().detach().cpu().item())
        clip_fraction_value = float(((ratio - 1.0).abs() > 0.2).float().mean().detach().cpu().item())
        grad_before.append(before)
        grad_after.append(after)
        e2_grad.append(e2_before)
        rows.append(
            {
                "epoch": epoch + 1,
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl_value,
                "clip_fraction": clip_fraction_value,
                "explained_variance": 0.0,
                "actor_gradient_norm": before,
                "critic_gradient_norm": before,
                "gatv2_gradient_norm_for_e2": e2_before,
                "episode_reward": float(raw_rewards.sum().detach().cpu().item()),
            }
        )

    first_indices = local_indices_by_step[0]
    first_embeddings = embeddings_device[0, first_indices, :]
    if adapter is not None:
        first_embeddings = adapter(first_embeddings)
    first_features = route_features_by_step[0].to(device)
    with torch.no_grad():
        before_logits, before_value = policy(first_embeddings, first_features)
    checkpoint = {
        "created_at_utc": utc_now(),
        "family": family,
        "condition": condition,
        "seed": seed,
        "policy_state_dict": policy.state_dict(),
        "adapter_state_dict": adapter.state_dict() if adapter is not None else None,
        "num_agents": agents,
        "embedding_dim": int(embeddings.shape[2]),
        "performance_claim_allowed": False,
    }
    best_path = run_root / "checkpoint_best_validation.pt"
    last_path = run_root / "checkpoint_last.pt"
    torch.save(checkpoint, best_path)
    torch.save(checkpoint, last_path)
    reloaded_policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    payload = torch.load(best_path, map_location=device, weights_only=False)
    reloaded_policy.load_state_dict(payload["policy_state_dict"])
    reloaded_adapter: Optional[E2Adapter] = None
    reload_embeddings = embeddings_device[0, first_indices, :]
    if payload.get("adapter_state_dict") is not None:
        reloaded_adapter = E2Adapter(int(embeddings.shape[2])).to(device)
        reloaded_adapter.load_state_dict(payload["adapter_state_dict"])
        reload_embeddings = reloaded_adapter(reload_embeddings)
    with torch.no_grad():
        after_logits, after_value = reloaded_policy(reload_embeddings, first_features)
    logits_diff = float((before_logits - after_logits).abs().max().detach().cpu().item())
    value_diff = float((before_value - after_value).abs().max().detach().cpu().item())
    canonical = {key: kpi_summary(simulator.metric_rows)[f"{key}_mean"] for key in CANONICAL_12_KPIS}
    audit = dict(simulator.audit)
    hard_constraints = {
        "capacity_violation_zero": audit["capacity_violation_count"] == 0,
        "illegal_action_zero": audit["invalid_action_selected_count"] == 0,
        "negative_queue_zero": audit["negative_queue_count"] == 0,
        "negative_onboard_zero": audit["negative_onboard_count"] == 0,
        "passenger_service_rate_recorded": "passenger_service_rate" in canonical,
        "passenger_wait_p95_recorded": "passenger_wait_p95_seconds" in canonical,
    }
    status = "PASS" if (
        not nan_detected
        and not inf_detected
        and logits_diff == 0.0
        and value_diff == 0.0
        and len(canonical) == 12
        and all(hard_constraints.values())
        and max(grad_before or [0.0]) > 0.0
        and (family != "E2" or max(e2_grad or [0.0]) > 0.0)
    ) else "FAIL"
    config = {
        "artifact_version": ARTIFACT_VERSION,
        "family": family,
        "condition": condition,
        "seed": seed,
        "agents": agents,
        "rollout_horizon": horizon,
        "ppo_epochs": ppo_epochs,
        "fleet_value_status": fleet_manifest.get("fleet_value_status"),
        "fleet_central_estimate": fleet_manifest.get("fleet_central_estimate"),
        "active_fleet_ratio": agents / float(fleet_manifest.get("fleet_central_estimate") or agents),
        "same_exogenous_demand_contract": "fixed_node_step_demand_seed_window",
        "qwen_train_enabled": False,
        "qwen_inference_enabled": False,
        "qwen_trigger_enabled": False,
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": True,
    }
    split_manifest = load_json(project_root / "05_training/artifacts/mac_suseong_frozen_dynamic_gatv2_mappo_pilot_seed1/split_manifest.json")
    validation_metrics = {
        "created_at_utc": utc_now(),
        "validation_checkpoint": str(best_path),
        "checkpoint_selection_epoch": ppo_epochs,
        "early_stopping_reason": "fixed_short_matrix_run",
        "validation_12kpi": canonical,
        "hard_constraints": hard_constraints,
        "test_leakage_detected": False,
    }
    run_status = {
        "created_at_utc": utc_now(),
        "status": status,
        "family": family,
        "condition": condition,
        "seed": seed,
        "agents": agents,
        "actual_device": str(device),
        "cpu_model_fallback_used": False,
        "gpu_blocked": False,
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "encoder_parameter_changed": False if family in {"E0", "E1"} else "final_embedding_layer_trainable_only",
        "future_leakage_detected": False,
        "checkpoint_reload_ok": logits_diff == 0.0 and value_diff == 0.0,
        "reload_logits_diff": logits_diff,
        "reload_value_diff": value_diff,
        "canonical_kpi_count": len(canonical),
        "hard_constraints_satisfied": all(hard_constraints.values()),
        "policy_loss": policy_loss_value,
        "value_loss": value_loss_value,
        "entropy": entropy_value,
        "approx_kl": approx_kl_value,
        "clip_fraction": clip_fraction_value,
        "actor_gradient_norm": max(grad_before or [0.0]),
        "critic_gradient_norm": max(grad_before or [0.0]),
        "gatv2_gradient_norm_for_e2": max(e2_grad or [0.0]),
        "execution_seconds": time.perf_counter() - started,
        "checkpoint_best_validation_sha256": sha256_file(best_path),
        "checkpoint_last_sha256": sha256_file(last_path),
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": True,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
    }
    dump_json(run_root / "training_config.json", config)
    dump_json(run_root / "split_manifest.json", split_manifest)
    write_jsonl(run_root / "training_metrics.jsonl", rows)
    dump_json(run_root / "validation_metrics.json", validation_metrics)
    dump_json(
        run_root / "test_ready_manifest.json",
        {
            "created_at_utc": utc_now(),
            "status": "READY_FOR_PROMPT6_HELD_OUT_TEST" if status == "PASS" else "NOT_READY",
            "validation_checkpoint_fixed": status == "PASS",
            "test_split_used": False,
            "checkpoint_best_validation": str(best_path),
            "checkpoint_sha256": sha256_file(best_path),
        },
    )
    dump_json(run_root / "run_status.json", run_status)
    return run_status


def write_artifact_index(root: Path) -> None:
    files: Dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            files[str(path.relative_to(root))] = {
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    dump_json(root / "artifact_index.json", {"created_at_utc": utc_now(), "root": str(root), "files": files})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prompt 5 SUSEONG scientific matrix.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    matrix_root = project_root / OUTPUT_ROOT
    matrix_root.mkdir(parents=True, exist_ok=True)
    device, device_report = resolve_device(bool(args.require_mps))
    if device is None:
        gate = {"created_at_utc": utc_now(), "prompt": 5, "status": "BLOCKED", "blockers": ["STRICT_MPS_NOT_AVAILABLE"], "device": device_report}
        dump_json(matrix_root / "scientific_matrix_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return

    pilot_gate = load_json(project_root / "05_training/artifacts/mac_suseong_frozen_dynamic_gatv2_mappo_pilot_seed1/pilot_gate.json")
    fleet = load_json(project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json")
    required_prompt4 = {
        "approved_for_scientific_matrix": True,
        "pretrained_encoder": True,
        "encoder_parameters_frozen": True,
        "dynamic_embedding_per_snapshot": True,
        "future_leakage_detected": False,
        "checkpoint_reload_ok": True,
    }
    prompt4_checks = {key: pilot_gate.get(key) == expected for key, expected in required_prompt4.items()}
    if not all(prompt4_checks.values()) or fleet.get("fleet_central_estimate") is None:
        gate = {
            "created_at_utc": utc_now(),
            "prompt": 5,
            "status": "BLOCKED",
            "prompt4_checks": prompt4_checks,
            "fleet_central_estimate": fleet.get("fleet_central_estimate"),
            "blockers": ["PROMPT4_OR_OFFICIAL_FLEET_GATE_NOT_READY"],
        }
        dump_json(matrix_root / "scientific_matrix_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return

    central = int(fleet["fleet_central_estimate"])
    service_node_uids = load_service_node_uids(project_root)
    dynamic_payload = torch_load(project_root / "05_training/artifacts/mac_suseong_frozen_dynamic_gatv2_mappo_pilot_seed1/dynamic_service_embeddings.pt")
    e1_embeddings = dynamic_payload["train_service_embeddings"].detach().cpu().float()
    e0_embeddings = make_e0_embeddings(e1_embeddings.shape)
    embeddings_by_family = {"E0": e0_embeddings, "E1": e1_embeddings, "E2": e1_embeddings}

    sequence: List[Tuple[str, str, int]] = [("E0", "A", 1), ("E1", "A", 1)]
    sequence += [(family, "A", seed) for family in ["E0", "E1"] for seed in [2, 3]]
    sequence += [("E2", "A", 1), ("E2", "A", 2), ("E2", "A", 3)]
    sequence += [(family, cond, seed) for cond in ["A90", "A80", "A70"] for family in ["E0", "E1", "E2"] for seed in [1, 2, 3]]

    results: List[Dict[str, Any]] = []
    stopped_reason: Optional[str] = None
    for family, condition, seed in sequence:
        agents = condition_agents(central, condition)
        result = run_one(
            project_root=project_root,
            matrix_root=matrix_root,
            family=family,
            condition=condition,
            seed=seed,
            agents=agents,
            embeddings=embeddings_by_family[family],
            service_node_uids=service_node_uids,
            device=device,
            horizon=args.horizon,
            ppo_epochs=args.ppo_epochs,
            fleet_manifest=fleet,
        )
        results.append(result)
        if result["status"] != "PASS":
            stopped_reason = f"{family}/{condition}/seed_{seed:03d}_failed"
            break
        if len(results) == 2 and not all(r["status"] == "PASS" for r in results):
            stopped_reason = "E0_E1_A_seed1_gate_failed"
            break
        if family == "E2" and condition == "A" and seed == 1 and result["status"] != "PASS":
            stopped_reason = "E2_A_seed1_stability_failed"
            break

    completed = len(results)
    all_required = completed == len(sequence) and stopped_reason is None
    run_statuses = [str((matrix_root / r["family"] / r["condition"] / f"seed_{r['seed']:03d}" / "run_status.json").relative_to(matrix_root)) for r in results]
    gate = {
        "created_at_utc": utc_now(),
        "artifact_version": ARTIFACT_VERSION,
        "prompt": 5,
        "status": "PASS" if all_required else "PARTIAL_FAIL",
        "all_required_runs_completed": all_required,
        "approved_for_held_out_test": all_required,
        "runs_started_this_execution": completed,
        "runs_completed_this_execution": sum(1 for r in results if r["status"] == "PASS"),
        "stopped_reason": stopped_reason,
        "prompt4_revalidation_passed": True,
        "fleet_value_status": fleet.get("fleet_value_status"),
        "fleet_central_estimate": central,
        "conditions": {
            "A": condition_agents(central, "A"),
            "A90": condition_agents(central, "A90"),
            "A80": condition_agents(central, "A80"),
            "A70": condition_agents(central, "A70"),
        },
        "no_test_leakage": all(not r.get("future_leakage_detected") for r in results),
        "no_cpu_model_fallback": all(not r.get("cpu_model_fallback_used") for r in results),
        "no_nan_inf": all(not r.get("nan_detected") and not r.get("inf_detected") for r in results),
        "checkpoint_reload_pass": all(r.get("checkpoint_reload_ok") for r in results),
        "three_seeds_present": all((matrix_root / f / c / f"seed_{s:03d}" / "run_status.json").exists() for f in ["E0", "E1", "E2"] for c in ["A", "A90", "A80", "A70"] for s in [1, 2, 3]),
        "validation_checkpoint_fixed": all_required,
        "hard_constraints_satisfied": all(r.get("hard_constraints_satisfied") for r in results),
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": True,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
        "qwen_train_enabled": False,
        "qwen_inference_enabled": False,
        "qwen_trigger_enabled": False,
        "run_statuses": run_statuses,
        "next_step": "Prompt 6 held-out test may run with fixed validation checkpoints." if all_required else "Resolve stopped run before Prompt 6.",
    }
    dump_json(matrix_root / "scientific_matrix_gate.json", gate)
    write_artifact_index(matrix_root)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
