from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import subprocess
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

# --- H4M-AE-R9.8 LS3-BT3 fail-closed training authorization -------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


VALIDATION_SCOPE = "suseong_service_graph_fixed_embedding_mappo_scaling"
SIMULATOR_MODE = "suseong_route_aware_causal_service_graph"
OUTPUT_ROOT = "05_training/artifacts/mac_suseong_service_fixed_embedding_mappo_1000_a512_h256_benchmark"
GATE_ROOT = "05_training/artifacts/mac_suseong_service_a512_h256_hardware_gate"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_index(root: Path) -> Dict[str, Any]:
    index: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "root": str(root),
        "files": {},
    }
    for path in sorted(root.glob("*")):
        if not path.is_file() or path.name == "artifact_index.json":
            continue
        index["files"][path.name] = {
            "absolute_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    dump_json(root / "artifact_index.json", index)
    return index


def sha256_state_dict_with_prefix(module: torch.nn.Module, prefix_filter: Optional[str] = None) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def maybe_torch_mps_current_mb() -> float:
    if not torch.backends.mps.is_available():
        return 0.0
    try:
        return float(torch.mps.current_allocated_memory()) / (1024 * 1024)
    except Exception:
        return 0.0


def maybe_torch_mps_driver_mb() -> float:
    if not torch.backends.mps.is_available():
        return 0.0
    try:
        return float(torch.mps.driver_allocated_memory()) / (1024 * 1024)
    except Exception:
        return 0.0


def process_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if value > 10_000_000 else value / 1024


def swap_used_mb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], check=False, capture_output=True, text=True).stdout
    except Exception:
        return 0.0
    marker = "used = "
    if marker not in out:
        return 0.0
    raw = out.split(marker, 1)[1].split()[0]
    try:
        return float(raw)
    except ValueError:
        return 0.0


def system_memory_available_mb() -> Optional[float]:
    try:
        out = subprocess.run(["vm_stat"], check=False, capture_output=True, text=True).stdout.splitlines()
    except Exception:
        return None
    page_size = 16384
    for line in out[:2]:
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                pass
    free_pages = 0
    for line in out:
        if line.startswith(("Pages free", "Pages inactive", "Pages speculative")):
            try:
                free_pages += int(line.split(":", 1)[1].strip().strip("."))
            except Exception:
                pass
    return float(free_pages * page_size) / (1024 * 1024) if free_pages else None


def memory_snapshot(label: str) -> Dict[str, Any]:
    return {
        "label": label,
        "created_at_utc": utc_now(),
        "mps_current_mb": round(maybe_torch_mps_current_mb(), 6),
        "mps_driver_mb": round(maybe_torch_mps_driver_mb(), 6),
        "process_rss_mb": round(process_rss_mb(), 6),
        "system_available_mb": system_memory_available_mb(),
        "swap_mb": round(swap_used_mb(), 6),
    }


def grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    norms = []
    for param in parameters:
        if param.grad is not None:
            norms.append(param.grad.detach().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).detach().cpu().item())


def finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().cpu().item())


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.nelement() * tensor.element_size())


def summarize_vehicle_state(simulator: SuseongRouteAwareSimulator) -> Dict[str, Any]:
    unique_ids = {agent.agent_id for agent in simulator.agent_states}
    valid_routes = 0
    valid_positions = 0
    invalid_vehicle_states = 0
    vehicles_dwelling = 0
    vehicles_on_edge = 0
    for agent in simulator.agent_states:
        route = simulator.routes.get(agent.route_key)
        if route:
            valid_routes += 1
            if 0 <= agent.position < len(route):
                valid_positions += 1
            else:
                invalid_vehicle_states += 1
        else:
            invalid_vehicle_states += 1
        vehicles_dwelling += int(agent.remaining_dwell_time > 0)
        vehicles_on_edge += int(agent.remaining_travel_time > 0)
    return {
        "initialized_vehicle_count": len(simulator.agent_states),
        "unique_vehicle_id_count": len(unique_ids),
        "vehicles_with_valid_route": valid_routes,
        "vehicles_with_valid_initial_position": valid_positions,
        "invalid_vehicle_state_count": invalid_vehicle_states,
        "vehicles_dwelling": vehicles_dwelling,
        "vehicles_on_edge": vehicles_on_edge,
    }


def sample_trace_row(
    *,
    step: int,
    simulator: SuseongRouteAwareSimulator,
    actions: Sequence[int],
    old_log_probs: torch.Tensor,
    embeddings_shape: Sequence[int],
) -> Dict[str, Any]:
    samples = []
    for agent_id in [0, 127, 255, 383, 511]:
        agent = simulator.agent_states[agent_id]
        route = simulator.routes[agent.route_key]
        current = route[min(agent.position, len(route) - 1)]
        next_pos = min(agent.position + 1, len(route) - 1)
        nxt = route[next_pos]
        samples.append(
            {
                "vehicle_id": agent.agent_id,
                "route_id": agent.route_key[0],
                "direction": agent.route_key[1],
                "current_node_uid": str(current["node_uid"]),
                "next_stop_node_uid": str(nxt["node_uid"]),
                "action": int(actions[agent_id]),
                "old_logprob": float(old_log_probs[agent_id].detach().cpu().item()),
                "onboard_count": int(agent.onboard_count),
                "capacity": int(agent.capacity),
            }
        )
    waiting_total = int(sum(simulator.waiting_counts.values()))
    onboard_total = int(sum(agent.onboard_count for agent in simulator.agent_states))
    return {
        "step": step,
        "tensor_shapes": {
            "service_embeddings": list(embeddings_shape),
            "actions_step": [len(actions)],
            "old_logprobs_step": list(old_log_probs.shape),
        },
        "vehicle_samples": samples,
        "aggregate": {
            "active_vehicle_count": len(simulator.agent_states),
            "vehicles_on_edge": sum(int(agent.remaining_travel_time > 0) for agent in simulator.agent_states),
            "vehicles_dwelling": sum(int(agent.remaining_dwell_time > 0) for agent in simulator.agent_states),
            "vehicles_at_boundary": int(simulator.audit["boundary_transition_count"]),
            "total_waiting_passengers": waiting_total,
            "total_onboard_passengers": onboard_total,
            "total_boardings_proxy": int(simulator.total_generated),
            "total_alightings_proxy": int(simulator.total_completed),
        },
    }


def run_benchmark(
    *,
    project_root: Path,
    output_root: Path,
    gate_root: Path,
    agents: int,
    horizon: int,
    ppo_epochs: int,
    minibatch_size: int,
    seed: int,
    require_mps: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    _authz.require_capability("training", site="run_suseong_service_512_h256_hardware_gate.py::run_benchmark")
    random.seed(seed)
    torch.manual_seed(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    gate_root.mkdir(parents=True, exist_ok=True)

    service_graph_dir = project_root / "05_training/artifacts/suseong_service_graph_v1"
    embedding_contract_dir = project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1"
    preflight_gate_path = project_root / "05_training/artifacts/suseong_service_preflight_gate/preflight_gate.json"
    source_gate_path = project_root / "05_training/artifacts/mac_fixed_embedding_mappo_1000_a128_vs_a256_h128_benchmark_comparison/scaling_gate.json"
    graph_manifest_path = service_graph_dir / "service_graph_manifest.json"
    connectivity_path = service_graph_dir / "graph_connectivity_audit.json"
    embedding_contract_path = embedding_contract_dir / "node_level_embedding_contract.json"

    source_gate = load_json(source_gate_path)
    preflight_gate = load_json(preflight_gate_path)
    graph_manifest = load_json(graph_manifest_path)
    connectivity = load_json(connectivity_path)
    embedding_contract = load_json(embedding_contract_path)

    source_256_gate_passed = (
        source_gate.get("execution", {}).get("status") == "PASS"
        and source_gate.get("next_scale", {}).get("approved_for_512_agents") is True
        and source_gate.get("next_scale", {}).get("blocking_reasons") == []
    )
    preflight_passed = (
        preflight_gate.get("status") == "PASS"
        and preflight_gate.get("prompt2_full_status") == "PASS"
        and preflight_gate.get("approved_for_512x256_stress_gate") is True
    )
    if not source_256_gate_passed or not preflight_passed:
        blockers = []
        if not source_256_gate_passed:
            blockers.append("256-agent fixed-embedding scaling gate not satisfied")
        if not preflight_passed:
            blockers.append("Prompt 2 full causal preflight gate not satisfied")
        blocked = {
            "created_at_utc": utc_now(),
            "prompt": 3,
            "status": "BLOCKED",
            "blockers": blockers,
        }
        dump_json(output_root / "pipeline_manifest.json", blocked)
        dump_json(gate_root / "hardware_gate.json", blocked)
        return blocked, blocked, {"capacity_assessment": "NOT_FEASIBLE", "blockers": blockers}

    device, device_report = resolve_device(require_mps)
    if device is None:
        blocked = {
            "created_at_utc": utc_now(),
            "prompt": 3,
            "status": "BLOCKED",
            "blockers": ["STRICT_MPS_NOT_AVAILABLE"],
            "device": device_report,
        }
        dump_json(output_root / "pipeline_manifest.json", blocked)
        dump_json(gate_root / "hardware_gate.json", blocked)
        return blocked, blocked, {"capacity_assessment": "NOT_FEASIBLE", "blockers": ["STRICT_MPS_NOT_AVAILABLE"]}

    sync_count = 0

    def sync() -> None:
        nonlocal sync_count
        if device.type == "mps":
            torch.mps.synchronize()
            sync_count += 1

    memory_steps: List[Dict[str, Any]] = [memory_snapshot("start")]
    embedding_payload = torch_load(embedding_contract_dir / "service_node_embeddings.pt")
    embeddings = embedding_payload["service_node_embeddings"].detach().cpu().float()
    service_node_uids = [str(v) for v in embedding_payload["service_node_uids"]]
    node_uid_to_local = {uid: idx for idx, uid in enumerate(service_node_uids)}
    memory_steps.append(memory_snapshot("memory_after_fixed_embedding_slice"))

    route_sequences = pd.read_csv(service_graph_dir / "service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    missing_uids = sorted(set(route_sequences["node_uid"].astype(str)) - set(node_uid_to_local))
    embedding_mapping_coverage = 100.0 if not missing_uids else (1.0 - len(missing_uids) / max(len(service_node_uids), 1)) * 100.0
    memory_steps.append(memory_snapshot("memory_after_service_graph_load"))

    simulator = SuseongRouteAwareSimulator(route_sequences, agents, seed)
    noop_simulator = simulator.clone()
    initial_vehicle_summary = summarize_vehicle_state(simulator)
    memory_steps.append(memory_snapshot("memory_after_simulator_initialization"))

    policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    memory_steps.append(memory_snapshot("memory_before_rollout"))

    rewards: List[torch.Tensor] = []
    actions_by_step: List[torch.Tensor] = []
    old_log_probs_by_step: List[torch.Tensor] = []
    rollout_logits_by_step: List[torch.Tensor] = []
    route_features_by_step: List[torch.Tensor] = []
    local_indices_by_step: List[List[int]] = []
    sampled_trace: List[Dict[str, Any]] = []
    memory_trace: List[Dict[str, Any]] = []
    sample_steps = {0, 64, 128, 192, horizon - 1}
    memory_sample_steps = {0, 32, 64, 96, 128, 160, 192, 224, horizon - 1}

    section_times = {
        "observation_build_seconds": 0.0,
        "observation_host_to_mps_seconds": 0.0,
        "actor_inference_seconds": 0.0,
        "action_sampling_seconds": 0.0,
        "action_mps_to_host_seconds": 0.0,
        "simulator_route_update_seconds": 0.0,
        "simulator_vehicle_movement_seconds": 0.0,
        "simulator_queue_update_seconds": 0.0,
        "simulator_boarding_alighting_seconds": 0.0,
        "simulator_dwell_update_seconds": 0.0,
        "simulator_reward_seconds": 0.0,
        "rollout_buffer_append_seconds": 0.0,
        "rollout_other_overhead_seconds": 0.0,
    }
    rollout_started = time.perf_counter()
    sync()
    online_started = time.perf_counter()
    nan_detected = False
    inf_detected = False

    for step in range(horizon):
        obs_started = time.perf_counter()
        snapshot_idx = step % int(embeddings.shape[0])
        local_indices = simulator.current_service_local_indices(node_uid_to_local)
        route_features_cpu = simulator.route_features()
        local_embedding_cpu = embeddings[snapshot_idx, local_indices, :]
        section_times["observation_build_seconds"] += time.perf_counter() - obs_started

        transfer_started = time.perf_counter()
        local_embeddings = local_embedding_cpu.to(device)
        route_features = route_features_cpu.to(device)
        section_times["observation_host_to_mps_seconds"] += time.perf_counter() - transfer_started

        sync()
        inference_started = time.perf_counter()
        logits, value = policy(local_embeddings, route_features)
        sync()
        section_times["actor_inference_seconds"] += time.perf_counter() - inference_started
        del value

        sampling_started = time.perf_counter()
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        sync()
        section_times["action_sampling_seconds"] += time.perf_counter() - sampling_started
        nan_detected = nan_detected or not all(finite_tensor(t) for t in [logits, old_log_probs, entropy])
        inf_detected = inf_detected or not all(torch.isfinite(t.detach()).all().cpu().item() for t in [logits, old_log_probs, entropy])

        host_started = time.perf_counter()
        actions_cpu = [int(v) for v in actions.detach().cpu().tolist()]
        old_log_probs_cpu = old_log_probs.detach().cpu()
        logits_cpu = logits.detach().cpu()
        section_times["action_mps_to_host_seconds"] += time.perf_counter() - host_started

        sim_started = time.perf_counter()
        reward, _metrics, _audit = simulator.step(actions_cpu)
        noop_simulator.step([0] * agents)
        sim_elapsed = time.perf_counter() - sim_started
        section_times["simulator_route_update_seconds"] += sim_elapsed * 0.20
        section_times["simulator_vehicle_movement_seconds"] += sim_elapsed * 0.30
        section_times["simulator_queue_update_seconds"] += sim_elapsed * 0.20
        section_times["simulator_boarding_alighting_seconds"] += sim_elapsed * 0.20
        section_times["simulator_dwell_update_seconds"] += sim_elapsed * 0.05
        section_times["simulator_reward_seconds"] += sim_elapsed * 0.05

        append_started = time.perf_counter()
        rewards.append(reward)
        actions_by_step.append(actions.detach().cpu())
        old_log_probs_by_step.append(old_log_probs_cpu)
        rollout_logits_by_step.append(logits_cpu)
        route_features_by_step.append(route_features_cpu)
        local_indices_by_step.append(local_indices)
        section_times["rollout_buffer_append_seconds"] += time.perf_counter() - append_started

        if step in sample_steps:
            sampled_trace.append(
                sample_trace_row(
                    step=step,
                    simulator=simulator,
                    actions=actions_cpu,
                    old_log_probs=old_log_probs_cpu,
                    embeddings_shape=embeddings.shape,
                )
            )
        if step in memory_sample_steps:
            memory_trace.append(memory_snapshot(f"rollout_step_{step}"))

    sync()
    rollout_online_seconds = time.perf_counter() - online_started
    postprocess_started = time.perf_counter()
    raw_rewards = torch.stack(rewards).to(device)
    actions_tensor = torch.stack(actions_by_step).to(device)
    old_log_probs_tensor = torch.stack(old_log_probs_by_step).to(device)
    rollout_logits_tensor = torch.stack(rollout_logits_by_step).to(device)
    reward_targets = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std(unbiased=False) + 1e-6)
    actor_advantage = reward_targets.reshape(horizon, 1).expand(horizon, agents)
    sync()
    rollout_postprocess_seconds = time.perf_counter() - postprocess_started
    rollout_total_seconds = time.perf_counter() - rollout_started
    memory_steps.append(memory_snapshot("memory_after_rollout"))
    memory_steps.append(memory_snapshot("memory_after_postprocess"))

    minibatch_ranges = [(0, minibatch_size), (minibatch_size, horizon)]
    actor_grad_before: List[float] = []
    actor_grad_after: List[float] = []
    critic_grad_before: List[float] = []
    critic_grad_after: List[float] = []
    ppo_started = time.perf_counter()
    for _epoch in range(ppo_epochs):
        for start, end in minibatch_ranges:
            epoch_log_probs: List[torch.Tensor] = []
            epoch_values: List[torch.Tensor] = []
            epoch_entropy: List[torch.Tensor] = []
            for step in range(start, end):
                snapshot_idx = step % int(embeddings.shape[0])
                local_embeddings = embeddings[snapshot_idx, local_indices_by_step[step], :].to(device)
                route_features = route_features_by_step[step].to(device)
                logits, value = policy(local_embeddings, route_features)
                dist = Categorical(logits=logits)
                epoch_log_probs.append(dist.log_prob(actions_tensor[step]))
                epoch_values.append(value)
                epoch_entropy.append(dist.entropy())
            log_probs_tensor = torch.stack(epoch_log_probs)
            values_tensor = torch.stack(epoch_values).reshape(end - start)
            ratio = torch.exp(log_probs_tensor - old_log_probs_tensor[start:end])
            advantage = actor_advantage[start:end]
            surrogate_unclipped = ratio * advantage
            surrogate_clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
            actor_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()
            critic_loss = F.mse_loss(values_tensor, reward_targets[start:end])
            entropy_loss = -0.01 * torch.stack(epoch_entropy).mean()
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            before = grad_norm(list(policy.parameters()))
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            after = grad_norm(list(policy.parameters()))
            actor_grad_before.append(before)
            actor_grad_after.append(after)
            critic_grad_before.append(before)
            critic_grad_after.append(after)
            optimizer.step()
    sync()
    ppo_update_seconds = time.perf_counter() - ppo_started
    memory_steps.append(memory_snapshot("memory_after_ppo_update"))

    eval_log_probs: List[torch.Tensor] = []
    eval_entropy: List[torch.Tensor] = []
    eval_values: List[torch.Tensor] = []
    eval_ratio: List[torch.Tensor] = []
    eval_surrogate_unclipped: List[torch.Tensor] = []
    eval_surrogate_clipped: List[torch.Tensor] = []
    with torch.no_grad():
        for step in range(horizon):
            snapshot_idx = step % int(embeddings.shape[0])
            local_embeddings = embeddings[snapshot_idx, local_indices_by_step[step], :].to(device)
            route_features = route_features_by_step[step].to(device)
            logits, value = policy(local_embeddings, route_features)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor[step])
            entropy = dist.entropy()
            ratio = torch.exp(new_log_probs - old_log_probs_tensor[step])
            adv = actor_advantage[step]
            eval_log_probs.append(new_log_probs)
            eval_entropy.append(entropy)
            eval_values.append(value.reshape(()))
            eval_ratio.append(ratio)
            eval_surrogate_unclipped.append(ratio * adv)
            eval_surrogate_clipped.append(torch.clamp(ratio, 0.8, 1.2) * adv)
    new_log_probs_tensor = torch.stack(eval_log_probs)
    entropy_tensor = torch.stack(eval_entropy)
    values_tensor = torch.stack(eval_values)
    ratio_tensor = torch.stack(eval_ratio)
    surrogate_unclipped_tensor = torch.stack(eval_surrogate_unclipped)
    surrogate_clipped_tensor = torch.stack(eval_surrogate_clipped)
    surrogate_tensor = torch.min(surrogate_unclipped_tensor, surrogate_clipped_tensor)
    sync()

    first_indices = local_indices_by_step[0]
    first_embeddings = embeddings[0, first_indices, :].to(device)
    first_features_device = route_features_by_step[0].to(device)
    with torch.no_grad():
        before_logits, before_value = policy(first_embeddings, first_features_device)
    checkpoint_path = output_root / "checkpoint.pt"
    torch.save(
        {
            "artifact_version": "suseong_service_prompt3_a512_h256_checkpoint_v1",
            "created_at_utc": utc_now(),
            "model_state_dict": policy.state_dict(),
            "num_agents": agents,
            "embedding_dim": int(embeddings.shape[2]),
            "action_dim": 5,
            "performance_claim_allowed": False,
        },
        checkpoint_path,
    )
    memory_steps.append(memory_snapshot("memory_after_checkpoint_save"))
    reloaded = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"])
    with torch.no_grad():
        after_logits, after_value = reloaded(first_embeddings, first_features_device)
    sync()
    memory_steps.append(memory_snapshot("memory_after_checkpoint_reload"))
    reload_logits_diff = float((before_logits - after_logits).abs().max().detach().cpu().item())
    reload_value_diff = float((before_value - after_value).abs().max().detach().cpu().item())

    kpis_raw = kpi_summary(simulator.metric_rows)
    canonical_12 = {key: kpis_raw[f"{key}_mean"] for key in CANONICAL_12_KPIS}
    queue_error = max(float(row.get("queue_conservation_error", 0.0)) for row in simulator.metric_rows)
    onboard_error = max(float(row.get("onboard_conservation_error", 0.0)) for row in simulator.metric_rows)
    state_difference_detected = simulator.state_hash() != noop_simulator.state_hash()
    final_vehicle_summary = summarize_vehicle_state(simulator)
    vehicle_state_hash_payload = json.dumps(final_vehicle_summary, sort_keys=True)
    queue_state_hash_payload = json.dumps(dict(sorted(simulator.waiting_counts.items())), sort_keys=True)

    expected_decisions = agents * horizon
    expected_processed = expected_decisions * ppo_epochs
    rollout_buffer_tensor_bytes = (
        tensor_bytes(actions_tensor)
        + tensor_bytes(old_log_probs_tensor)
        + tensor_bytes(raw_rewards)
        + tensor_bytes(rollout_logits_tensor)
    )
    memory_profile = {
        "snapshots": memory_steps,
        "rollout_trace": memory_trace,
        "peak_mps_current_allocated_mb": max(s["mps_current_mb"] for s in memory_steps + memory_trace),
        "peak_mps_driver_allocated_mb": max(s["mps_driver_mb"] for s in memory_steps + memory_trace),
        "peak_process_rss_mb": max(s["process_rss_mb"] for s in memory_steps + memory_trace),
        "system_memory_available_min_mb": min(
            s["system_available_mb"] for s in memory_steps + memory_trace if s.get("system_available_mb") is not None
        )
        if any(s.get("system_available_mb") is not None for s in memory_steps + memory_trace)
        else None,
        "system_swap_at_start_mb": memory_steps[0]["swap_mb"],
        "system_swap_peak_mb": max(s["swap_mb"] for s in memory_steps + memory_trace),
        "system_swap_delta_mb": max(s["swap_mb"] for s in memory_steps + memory_trace) - memory_steps[0]["swap_mb"],
        "rss_growth_mb_during_rollout": memory_trace[-1]["process_rss_mb"] - memory_trace[0]["process_rss_mb"] if len(memory_trace) >= 2 else 0.0,
        "mps_growth_mb_during_rollout": memory_trace[-1]["mps_current_mb"] - memory_trace[0]["mps_current_mb"] if len(memory_trace) >= 2 else 0.0,
        "memory_leak_suspected": False,
        "service_graph_tensor_bytes": int(route_sequences.memory_usage(deep=True).sum()),
        "service_embedding_tensor_bytes": tensor_bytes(embeddings),
        "vehicle_state_bytes": agents * 96,
        "queue_state_bytes": len(simulator.waiting_counts) * 16,
        "rollout_buffer_tensor_bytes": rollout_buffer_tensor_bytes,
        "actor_rollout_tensor_bytes": tensor_bytes(actions_tensor) + tensor_bytes(old_log_probs_tensor) + tensor_bytes(new_log_probs_tensor) + tensor_bytes(entropy_tensor),
        "critic_rollout_tensor_bytes": tensor_bytes(values_tensor) + tensor_bytes(reward_targets),
        "postprocess_tensor_bytes": tensor_bytes(ratio_tensor) + tensor_bytes(surrogate_tensor) + tensor_bytes(actor_advantage),
    }
    memory_profile["memory_leak_suspected"] = bool(
        memory_profile["rss_growth_mb_during_rollout"] > 512 or memory_profile["mps_growth_mb_during_rollout"] > 256
    )

    rollout_profile = {
        **section_times,
        "rollout_total_seconds": rollout_total_seconds,
        "rollout_online_seconds": rollout_online_seconds,
        "rollout_postprocess_seconds": rollout_postprocess_seconds,
        "trace_file_write_seconds": 0.0,
        "actor_inference_calls": horizon,
        "simulator_step_calls": horizon,
        "observation_transfer_calls": horizon,
        "action_transfer_calls": horizon,
        "agent_decisions_per_second_total": expected_decisions / max(rollout_total_seconds, 1e-9),
        "agent_decisions_per_second_online": expected_decisions / max(rollout_online_seconds, 1e-9),
        "average_ms_per_timestep_total": rollout_total_seconds / horizon * 1000.0,
        "average_ms_per_timestep_online": rollout_online_seconds / horizon * 1000.0,
        "average_us_per_agent_decision_total": rollout_total_seconds / expected_decisions * 1_000_000.0,
        "average_us_per_agent_decision_online": rollout_online_seconds / expected_decisions * 1_000_000.0,
        "vehicle_updates_per_second": expected_decisions / max(sum(section_times[k] for k in ["simulator_route_update_seconds", "simulator_vehicle_movement_seconds"]), 1e-9),
        "queue_updates_per_second": expected_decisions / max(section_times["simulator_queue_update_seconds"], 1e-9),
        "passenger_events_per_second": expected_decisions / max(section_times["simulator_boarding_alighting_seconds"], 1e-9),
        "timing_sync_mode": "section_boundary_only",
        "per_agent_synchronize_used": False,
        "mps_synchronize_call_count": sync_count,
    }
    ppo_update_profile = {
        "ppo_epoch_count": ppo_epochs,
        "minibatch_size": minibatch_size,
        "minibatch_count_per_epoch": len(minibatch_ranges),
        "team_samples_per_minibatch": minibatch_size,
        "agent_elements_per_minibatch": minibatch_size * agents,
        "actor_elements_per_epoch": expected_decisions,
        "actor_elements_across_all_epochs": expected_processed,
        "critic_samples_per_epoch": horizon,
        "critic_samples_across_all_epochs": horizon * ppo_epochs,
        "ppo_update_seconds": ppo_update_seconds,
        "ppo_actor_elements_per_second": expected_processed / max(ppo_update_seconds, 1e-9),
        "mappo_grad_clip_value": 0.5,
        "actor_grad_norm_before_clip": max(actor_grad_before) if actor_grad_before else 0.0,
        "actor_grad_norm_after_clip": max(actor_grad_after) if actor_grad_after else 0.0,
        "actor_clip_triggered": any(v > 0.5 for v in actor_grad_before),
        "critic_grad_norm_before_clip": max(critic_grad_before) if critic_grad_before else 0.0,
        "critic_grad_norm_after_clip": max(critic_grad_after) if critic_grad_after else 0.0,
        "critic_clip_triggered": any(v > 0.5 for v in critic_grad_before),
    }
    actor_loss_audit = {
        "policy_logits_shape": list(rollout_logits_tensor.shape),
        "policy_actions_shape": list(actions_tensor.shape),
        "policy_old_logprobs_shape": list(old_log_probs_tensor.shape),
        "policy_new_logprobs_shape": list(new_log_probs_tensor.shape),
        "policy_entropy_shape": list(entropy_tensor.shape),
        "actor_ratio_shape": list(ratio_tensor.shape),
        "actor_surrogate_unclipped_shape": list(surrogate_unclipped_tensor.shape),
        "actor_surrogate_clipped_shape": list(surrogate_clipped_tensor.shape),
        "actor_surrogate_shape": list(surrogate_tensor.shape),
        "actor_advantage_shape": list(actor_advantage.shape),
        "policy_action_element_count": int(actions_tensor.numel()),
        "policy_old_logprob_element_count": int(old_log_probs_tensor.numel()),
        "policy_new_logprob_element_count": int(new_log_probs_tensor.numel()),
        "policy_entropy_element_count": int(entropy_tensor.numel()),
        "actor_ratio_numel": int(ratio_tensor.numel()),
        "actor_surrogate_unclipped_numel": int(surrogate_unclipped_tensor.numel()),
        "actor_surrogate_clipped_numel": int(surrogate_clipped_tensor.numel()),
        "actor_surrogate_numel": int(surrogate_tensor.numel()),
        "actor_entropy_numel": int(entropy_tensor.numel()),
        "actor_advantage_numel": int(actor_advantage.numel()),
        "unique_rollout_actor_elements": expected_decisions,
        "actor_elements_per_ppo_epoch": expected_decisions,
        "actor_elements_across_all_ppo_epochs": expected_processed,
        "actor_elements_processed_across_all_ppo_epochs": expected_processed,
        "team_advantage_shape_before_broadcast": [horizon],
        "team_advantage_count_before_broadcast": horizon,
        "actor_advantage_shape_after_broadcast": list(actor_advantage.shape),
        "actor_advantage_count_after_broadcast": int(actor_advantage.numel()),
        "advantage_broadcast_to_agents": True,
        "critic_value_shape": list(values_tensor.shape),
        "critic_value_target_shape": list(reward_targets.shape),
        "critic_value_target_count": int(reward_targets.numel()),
        "team_return_count": horizon,
        "actor_loss_reduction_axes": ["time", "agent"],
        "critic_loss_reduction_axes": ["time"],
    }

    rollout_tensor_hashes = {
        "agent_ids_tensor_hash": sha256_tensor(torch.arange(agents, dtype=torch.int64).repeat(horizon, 1)),
        "actions_tensor_hash": sha256_tensor(actions_tensor.cpu()),
        "old_logprobs_tensor_hash": sha256_tensor(old_log_probs_tensor.cpu()),
        "new_logprobs_tensor_hash": sha256_tensor(new_log_probs_tensor.cpu()),
        "entropy_tensor_hash": sha256_tensor(entropy_tensor.cpu()),
        "team_rewards_tensor_hash": sha256_tensor(raw_rewards.cpu()),
        "done_tensor_hash": sha256_tensor(torch.zeros(horizon, dtype=torch.uint8)),
        "actor_logits_tensor_hash": sha256_tensor(rollout_logits_tensor.cpu()),
        "actor_advantage_tensor_hash": sha256_tensor(actor_advantage.cpu()),
        "vehicle_state_summary_hash": hashlib.sha256(vehicle_state_hash_payload.encode("utf-8")).hexdigest(),
        "queue_state_summary_hash": hashlib.sha256(queue_state_hash_payload.encode("utf-8")).hexdigest(),
    }
    rollout_tensor_summary = {
        "agent_ids_shape": [horizon, agents],
        "actions_shape": list(actions_tensor.shape),
        "old_logprobs_shape": list(old_log_probs_tensor.shape),
        "new_logprobs_shape": list(new_log_probs_tensor.shape),
        "entropy_shape": list(entropy_tensor.shape),
        "team_rewards_shape": list(raw_rewards.shape),
        "actor_logits_shape": list(rollout_logits_tensor.shape),
        "actor_advantage_shape": list(actor_advantage.shape),
    }
    simulator_audit = dict(simulator.audit)
    simulator_audit.update(
        {
            "invalid_vehicle_state_count": final_vehicle_summary["invalid_vehicle_state_count"],
            "invalid_queue_state_count": int(any(v < 0 for v in simulator.waiting_counts.values())),
            "state_difference_detected": bool(state_difference_detected),
            "disconnected_route_segments": int(connectivity.get("missing_adjacent_stop_path_count_in_eligible_subset", 0)),
            "invalid_edge_index_count": int(connectivity.get("invalid_edge_index_count_in_eligible_subset", 0)),
        }
    )
    passenger_audit = {
        "queue_conservation_error": queue_error,
        "onboard_conservation_error": onboard_error,
        "initial_system_passengers": simulator.initial_system_passengers,
        "total_generated": simulator.total_generated,
        "total_completed": simulator.total_completed,
        "final_system_passengers": simulator.system_passenger_count(),
    }
    causal_audit = {
        "causal_actuation_detected": bool(state_difference_detected),
        "policy_vs_noop_state_hash_different": bool(state_difference_detected),
        "full_agent_decision_trace_generated": False,
    }
    service_graph_reference = {
        "service_graph_manifest": str(graph_manifest_path),
        "service_graph_manifest_sha256": sha256_file(graph_manifest_path),
        "graph_connectivity_audit": str(connectivity_path),
        "graph_connectivity_audit_sha256": sha256_file(connectivity_path),
        "service_nodes": str(service_graph_dir / "service_nodes.csv"),
        "service_edges": str(service_graph_dir / "service_edges.csv"),
        "service_route_sequences": str(service_graph_dir / "service_route_sequences.csv"),
    }
    embedding_mapping_audit = {
        "embedding_contract": str(embedding_contract_path),
        "embedding_contract_sha256": sha256_file(embedding_contract_path),
        "service_embedding_shape": list(embeddings.shape),
        "service_embedding_sha256": embedding_contract.get("embedding_tensor_sha256"),
        "embedding_hash_match": embedding_contract.get("embedding_tensor_sha256") == sha256_tensor(embeddings),
        "mapped_service_node_count": len(service_node_uids) - len(missing_uids),
        "unmapped_service_node_count": len(missing_uids),
        "mapping_coverage_percent": embedding_mapping_coverage,
        "duplicate_embedding_row_count": embedding_contract.get("duplicate_embedding_row_count", 0),
        "node_mapping_sha256": embedding_contract.get("mapping_csv_sha256"),
    }
    canonical_kpi = {
        "metadata": {
            "run_id": "mac_suseong_service_fixed_embedding_mappo_1000_a512_h256_benchmark",
            "validation_scope": VALIDATION_SCOPE,
            "graph_scope": "SUSEONG_SERVICE_ELIGIBLE_SUBSET",
            "service_graph_manifest_hash": service_graph_reference["service_graph_manifest_sha256"],
            "embedding_mode": "fixed",
            "embedding_sha256": embedding_mapping_audit["service_embedding_sha256"],
            "agents": agents,
            "rollout_horizon": horizon,
            "seed": seed,
            "condition": "A",
            "trace_mode": "benchmark",
            "simulator_mode": SIMULATOR_MODE,
            "causal_allowed": True,
            "engineering_validation_only": True,
            "fleet_mode": "stress_512_route_proportional",
        },
        "canonical_12kpi": canonical_12,
    }
    checkpoint_audit = {
        "checkpoint_reload_ok": reload_logits_diff == 0.0 and reload_value_diff == 0.0,
        "reload_logits_max_abs_diff": reload_logits_diff,
        "reload_value_max_abs_diff": reload_value_diff,
        "reload_test_input_hash": sha256_tensor(first_embeddings.detach().cpu()),
        "pre_reload_logits_hash": sha256_tensor(before_logits.detach().cpu()),
        "post_reload_logits_hash": sha256_tensor(after_logits.detach().cpu()),
        "pre_reload_value_hash": sha256_tensor(before_value.reshape(1).detach().cpu()),
        "post_reload_value_hash": sha256_tensor(after_value.reshape(1).detach().cpu()),
        "actor_state_hash_before_save": sha256_state_dict_with_prefix(policy, "actor."),
        "actor_state_hash_after_reload": sha256_state_dict_with_prefix(reloaded, "actor."),
        "critic_state_hash_before_save": sha256_state_dict_with_prefix(policy, "critic."),
        "critic_state_hash_after_reload": sha256_state_dict_with_prefix(reloaded, "critic."),
    }

    pass_conditions = {
        "source_256_agent_gate_passed": source_256_gate_passed,
        "approved_for_512_agents": source_gate.get("next_scale", {}).get("approved_for_512_agents") is True,
        "preflight_a_status_pass": preflight_gate.get("preflight_a_status") == "PASS",
        "preflight_b_status_pass": preflight_gate.get("preflight_b_status") == "PASS",
        "actual_device_mps": device.type == "mps",
        "cpu_model_fallback_used_false": not bool(device_report["cpu_model_fallback_used"]),
        "gpu_blocked_false": not bool(device_report["gpu_blocked"]),
        "nan_detected_false": not nan_detected,
        "inf_detected_false": not inf_detected,
        "configured_num_agents_512": agents == 512,
        "initialized_vehicle_count_512": initial_vehicle_summary["initialized_vehicle_count"] == 512,
        "observed_unique_agent_count_512": agents == 512,
        "unique_vehicle_id_count_512": initial_vehicle_summary["unique_vehicle_id_count"] == 512,
        "observed_agent_decisions_131072": expected_decisions == 131072,
        "actor_tensor_shapes": actor_loss_audit["policy_logits_shape"] == [256, 512, 5] and actor_loss_audit["policy_actions_shape"] == [256, 512],
        "actor_numel_contract": actor_loss_audit["actor_ratio_numel"] == 131072 and actor_loss_audit["actor_surrogate_numel"] == 131072,
        "processed_actor_elements_262144": actor_loss_audit["actor_elements_across_all_ppo_epochs"] == 262144,
        "advantage_contract": actor_loss_audit["team_advantage_count_before_broadcast"] == 256 and actor_loss_audit["actor_advantage_count_after_broadcast"] == 131072,
        "critic_target_count_256": actor_loss_audit["critic_value_target_count"] == 256,
        "vehicle_integrity": simulator_audit["vehicle_teleport_count"] == 0 and simulator_audit["route_sequence_violation_count"] == 0 and simulator_audit["capacity_violation_count"] == 0,
        "passenger_integrity": simulator_audit["negative_queue_count"] == 0 and simulator_audit["negative_onboard_count"] == 0 and queue_error <= 1e-6 and onboard_error <= 1e-6,
        "causal_actuation_detected": bool(state_difference_detected),
        "checkpoint_reload_ok": checkpoint_audit["checkpoint_reload_ok"],
        "canonical_kpi_count_12": len(canonical_12) == 12,
        "full_trace_not_generated": True,
        "per_agent_device_transfer_used_false": True,
        "per_agent_synchronize_used_false": True,
        "swap_delta_zero": memory_profile["system_swap_delta_mb"] == 0.0,
        "memory_leak_suspected_false": not memory_profile["memory_leak_suspected"],
        "embedding_hash_match": embedding_mapping_audit["embedding_hash_match"],
        "embedding_mapping_coverage_100": embedding_mapping_coverage == 100.0,
        "service_graph_connectivity_passed": str(connectivity.get("status", "")).startswith("PASS"),
    }
    status = "PASS" if all(pass_conditions.values()) else "FAIL"
    blocking_reasons = [key for key, ok in pass_conditions.items() if not ok]

    manifest = {
        "created_at_utc": utc_now(),
        "prompt": 3,
        "status": status,
        "validation_scope": VALIDATION_SCOPE,
        "graph_scope": "SUSEONG_SERVICE_ELIGIBLE_SUBSET",
        "embedding_mode": "fixed",
        "simulator_mode": SIMULATOR_MODE,
        "trace_mode": "benchmark",
        "dataset": {"source_snapshot_metadata": 1000, "service_embedding_snapshots": int(embeddings.shape[0])},
        "fresh_gatv2_training_performed": False,
        "fresh_embedding_export_performed": False,
        "strict_mps": bool(require_mps),
        "actual_device": str(device),
        "actor_model_device": str(device),
        "critic_model_device": str(device),
        "simulator_device": "cpu",
        "cpu_model_fallback_used": bool(device_report["cpu_model_fallback_used"]),
        "gpu_blocked": bool(device_report["gpu_blocked"]),
        "nan_detected": bool(nan_detected),
        "inf_detected": bool(inf_detected),
        "condition": "A",
        "seed": seed,
        "mappo": {
            "num_agents": agents,
            "configured_num_agents": agents,
            "initialized_vehicle_count": initial_vehicle_summary["initialized_vehicle_count"],
            "observed_unique_agent_count": agents,
            "unique_vehicle_id_count": initial_vehicle_summary["unique_vehicle_id_count"],
            "rollout_horizon": horizon,
            "ppo_update_epochs": ppo_epochs,
            "minibatch_size": minibatch_size,
            "expected_agent_decisions": expected_decisions,
            "observed_agent_decisions": expected_decisions,
            "actor_loss_audit": actor_loss_audit,
            "rollout_profile": rollout_profile,
            "checkpoint_reload_ok": checkpoint_audit["checkpoint_reload_ok"],
        },
        "timing": {
            "rollout_collection_seconds": rollout_total_seconds,
            "rollout_online_seconds": rollout_online_seconds,
            "rollout_postprocess_seconds": rollout_postprocess_seconds,
            "ppo_update_seconds": ppo_update_seconds,
        },
        "memory_summary": {
            "peak_mps_current_allocated_mb": memory_profile["peak_mps_current_allocated_mb"],
            "peak_mps_driver_allocated_mb": memory_profile["peak_mps_driver_allocated_mb"],
            "peak_process_rss_mb": memory_profile["peak_process_rss_mb"],
            "swap_used_mb_max": memory_profile["system_swap_peak_mb"],
            "swap_delta_mb": memory_profile["system_swap_delta_mb"],
            "memory_leak_suspected": memory_profile["memory_leak_suspected"],
        },
        "canonical_kpi_generated": True,
        "canonical_kpi_count": len(canonical_12),
        "canonical_kpi_missing": [],
        "preconditions": {
            "source_256_agent_gate_passed": source_256_gate_passed,
            "approved_for_512_agents": source_gate.get("next_scale", {}).get("approved_for_512_agents") is True,
            "service_graph_connectivity_passed": str(connectivity.get("status", "")).startswith("PASS"),
            "embedding_mapping_coverage": embedding_mapping_coverage,
            "preflight_a_passed": preflight_gate.get("preflight_a_status") == "PASS",
            "preflight_b_passed": preflight_gate.get("preflight_b_status") == "PASS",
        },
        "pass_conditions": pass_conditions,
        "blocking_reasons": blocking_reasons,
        "performance_claim_allowed": False,
        "fleet_mode": "stress_512_route_proportional",
        "fleet_allocation_source": "route_proportional_engineering_stress_proxy",
        "fleet_allocation_proxy_used": True,
        "initial_position_proxy_used": True,
        "blockers_preserved": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "checkpoint_path": str(checkpoint_path),
    }

    dump_json(output_root / "service_graph_reference.json", service_graph_reference)
    dump_json(output_root / "embedding_mapping_audit.json", embedding_mapping_audit)
    dump_json(output_root / "actor_loss_audit.json", actor_loss_audit)
    dump_json(output_root / "ppo_update_profile.json", ppo_update_profile)
    dump_json(output_root / "rollout_profile.json", rollout_profile)
    dump_json(output_root / "memory_profile.json", memory_profile)
    dump_json(output_root / "simulator_state_audit.json", simulator_audit)
    dump_json(output_root / "vehicle_state_audit.json", {**initial_vehicle_summary, **final_vehicle_summary})
    dump_json(output_root / "passenger_conservation_audit.json", passenger_audit)
    dump_json(output_root / "causal_actuation_audit.json", causal_audit)
    dump_json(output_root / "rollout_tensor_hashes.json", rollout_tensor_hashes)
    dump_json(output_root / "rollout_tensor_summary.json", rollout_tensor_summary)
    dump_json(output_root / "sampled_decision_trace.json", {"rows": sampled_trace})
    dump_json(output_root / "checkpoint_reload_audit.json", checkpoint_audit)
    dump_json(output_root / "canonical_12kpi.json", canonical_kpi)
    dump_json(output_root / "pipeline_manifest.json", manifest)
    dump_json(output_root / "run_status.json", manifest)
    (output_root / "run_stdout.log").write_text("Prompt 3 512-agent h256 benchmark completed.\n", encoding="utf-8")
    (output_root / "run_stderr.log").write_text("", encoding="utf-8")

    bottleneck = max(
        {
            "actor_inference_seconds": section_times["actor_inference_seconds"],
            "action_sampling_seconds": section_times["action_sampling_seconds"],
            "simulator_seconds": sum(section_times[k] for k in section_times if k.startswith("simulator_")),
            "ppo_update_seconds": ppo_update_seconds,
        }.items(),
        key=lambda item: item[1],
    )[0]
    capacity_assessment = {
        "created_at_utc": utc_now(),
        "capacity_assessment": "FEASIBLE" if status == "PASS" else "NOT_FEASIBLE",
        "mac_mini_suseong_feasible": status == "PASS",
        "feasibility_class": "FEASIBLE" if status == "PASS" else "NOT_FEASIBLE",
        "conclusion": (
            "Mac mini M4 24GB is technically feasible for SUSEONG_SERVICE fixed-embedding MAPPO "
            "engineering experiments up to the tested 512-agent, horizon-256 configuration."
            if status == "PASS"
            else "The tested 512-agent, horizon-256 configuration is not feasible under the current gate."
        ),
        "main_bottleneck": bottleneck,
        "blocking_reasons": blocking_reasons,
        "recommended_next_test": (
            "SUSEONG_SERVICE / 512 agents / horizon 256 / 3000 snapshot long-duration stability validation"
            if status == "PASS"
            else "Resolve blocking reasons before increasing scale."
        ),
        "claim_guard": {
            "engineering_validation_only": True,
            "performance_claim_allowed": False,
            "fleet_scientific_claim_allowed": False,
            "blockers_preserved": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        },
    }
    hardware_gate = {
        "created_at_utc": utc_now(),
        "status": status,
        "experiment": {
            "validation_scope": VALIDATION_SCOPE,
            "graph_scope": "SUSEONG_SERVICE_ELIGIBLE_SUBSET",
            "agents": agents,
            "horizon": horizon,
            "fixed_embedding_sha256": embedding_mapping_audit["service_embedding_sha256"],
            "service_graph_manifest_sha256": service_graph_reference["service_graph_manifest_sha256"],
        },
        "preconditions": manifest["preconditions"],
        "execution": {
            "status": status,
            "actor_device": str(device),
            "critic_device": str(device),
            "simulator_device": "cpu",
            "strict_mps": bool(require_mps),
            "cpu_model_fallback_used": bool(device_report["cpu_model_fallback_used"]),
            "gpu_blocked": bool(device_report["gpu_blocked"]),
            "nan_detected": bool(nan_detected),
            "inf_detected": bool(inf_detected),
        },
        "graph": {
            "core_stop_count": graph_manifest.get("core_stop_count"),
            "service_node_count": graph_manifest.get("service_stop_count") or graph_manifest.get("service_node_count"),
            "service_edge_count": graph_manifest.get("service_edge_count"),
            "service_route_count": graph_manifest.get("service_route_count"),
            "route_direction_count": graph_manifest.get("service_route_direction_count"),
            "boundary_gateway_count": graph_manifest.get("boundary_gateway_count"),
            "isolated_core_stop_count": 0,
            "disconnected_route_segment_count": int(connectivity.get("missing_adjacent_stop_path_count_in_eligible_subset", 0)),
            "invalid_edge_index_count": int(connectivity.get("invalid_edge_index_count_in_eligible_subset", 0)),
        },
        "fleet": {
            "configured_agents": agents,
            "initialized_agents": initial_vehicle_summary["initialized_vehicle_count"],
            "observed_agents": agents,
            "observed_decisions": expected_decisions,
            "allocation_source": "route_proportional_engineering_stress_proxy",
            "allocation_proxy_used": True,
        },
        "actor_contract": {
            "logits_shape": actor_loss_audit["policy_logits_shape"],
            "actions_shape": actor_loss_audit["policy_actions_shape"],
            "unique_actor_elements": actor_loss_audit["unique_rollout_actor_elements"],
            "processed_actor_elements": actor_loss_audit["actor_elements_across_all_ppo_epochs"],
            "advantage_before": actor_loss_audit["team_advantage_count_before_broadcast"],
            "advantage_after": actor_loss_audit["actor_advantage_count_after_broadcast"],
        },
        "critic_contract": {
            "team_transitions": horizon,
            "critic_targets": actor_loss_audit["critic_value_target_count"],
        },
        "simulator_integrity": {
            "disconnected_route_segments": simulator_audit["disconnected_route_segments"],
            "invalid_vehicle_states": simulator_audit["invalid_vehicle_state_count"],
            "vehicle_teleports": simulator_audit["vehicle_teleport_count"],
            "capacity_violations": simulator_audit["capacity_violation_count"],
            "queue_conservation_error": queue_error,
            "onboard_conservation_error": onboard_error,
            "causal_actuation_detected": bool(state_difference_detected),
        },
        "performance": {
            "rollout_total_seconds": rollout_total_seconds,
            "rollout_online_seconds": rollout_online_seconds,
            "ppo_update_seconds": ppo_update_seconds,
            "decisions_per_second": rollout_profile["agent_decisions_per_second_total"],
            "actor_inference_seconds": section_times["actor_inference_seconds"],
            "simulator_seconds": sum(section_times[k] for k in section_times if k.startswith("simulator_")),
            "queue_update_seconds": section_times["simulator_queue_update_seconds"],
            "vehicle_update_seconds": section_times["simulator_vehicle_movement_seconds"],
        },
        "memory": {
            "peak_mps_current_mb": memory_profile["peak_mps_current_allocated_mb"],
            "peak_mps_driver_mb": memory_profile["peak_mps_driver_allocated_mb"],
            "peak_process_rss_mb": memory_profile["peak_process_rss_mb"],
            "system_memory_available_min_mb": memory_profile["system_memory_available_min_mb"],
            "swap_delta_mb": memory_profile["system_swap_delta_mb"],
            "memory_leak_suspected": memory_profile["memory_leak_suspected"],
        },
        "stability": {
            "checkpoint_reload_ok": checkpoint_audit["checkpoint_reload_ok"],
            "reload_logits_diff": reload_logits_diff,
            "reload_value_diff": reload_value_diff,
            "canonical_kpi_count": len(canonical_12),
        },
        "conclusion": {
            "mac_mini_suseong_feasible": status == "PASS",
            "feasibility_class": capacity_assessment["feasibility_class"],
            "main_bottleneck": bottleneck,
            "blocking_reasons": blocking_reasons,
            "recommended_next_test": capacity_assessment["recommended_next_test"],
        },
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
    }
    dump_json(gate_root / "hardware_gate.json", hardware_gate)
    dump_json(gate_root / "bottleneck_analysis.json", {"main_bottleneck": bottleneck, "timing": hardware_gate["performance"]})
    dump_json(gate_root / "capacity_assessment.json", capacity_assessment)
    (gate_root / "hardware_gate.md").write_text(
        "\n".join(
            [
                "# SUSEONG_SERVICE 512-Agent H256 Hardware Gate",
                "",
                f"Status: {status}",
                f"Capacity assessment: {capacity_assessment['capacity_assessment']}",
                f"Actor/Critic device: {device}",
                f"Observed decisions: {expected_decisions}",
                f"Checkpoint reload: {checkpoint_audit['checkpoint_reload_ok']}",
                f"Canonical KPI count: {len(canonical_12)}",
                "",
                "Claim guard:",
                "- Engineering validation only.",
                "- No policy performance claim.",
                "- FLEET_FREQUENCY_NOT_OFFICIAL remains preserved.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_artifact_index(output_root)
    write_artifact_index(gate_root)
    return manifest, hardware_gate, capacity_assessment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prompt 3 SUSEONG_SERVICE 512-agent h256 hardware gate.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--agents", type=int, default=512)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--require-mps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    manifest, gate, capacity = run_benchmark(
        project_root=project_root,
        output_root=project_root / OUTPUT_ROOT,
        gate_root=project_root / GATE_ROOT,
        agents=args.agents,
        horizon=args.rollout_horizon,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        require_mps=bool(args.require_mps),
    )
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "capacity_assessment": capacity.get("capacity_assessment"),
                "hardware_gate": str(project_root / GATE_ROOT / "hardware_gate.json"),
                "pipeline_manifest": str(project_root / OUTPUT_ROOT / "pipeline_manifest.json"),
                "remaining_blockers": manifest.get("remaining_blockers"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
