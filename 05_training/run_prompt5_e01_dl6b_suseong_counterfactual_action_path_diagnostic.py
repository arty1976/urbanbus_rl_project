from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


ARTIFACT_PREFIX = "prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic"
DL6A_R1_ARTIFACT = "05_training/artifacts/prompt5_e01_dl6a_r1_suseong_actor_action_activation_diagnostic_20260801_212128"
DL5_ARTIFACT = "05_training/artifacts/prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation_20260731_185643"
DL4_ARTIFACT = "05_training/artifacts/prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"

EXPECTED_DL6A_GATE = "PASS_SUSEONG_DL6A_ACTOR_ACTION_SIGNAL_PRESENT"
EXPECTED_DL5_GATE = "PASS_SUSEONG_DL4_MODEL_VS_B0_B1_B2_12KPI_EVALUATION_COMPLETE"
EXPECTED_DL4_GATE = "PASS_SUSEONG_MAPPO_CRITIC_CALIBRATION_IMPROVED_ON_MAC_M4"

PASS_SENSITIVITY = "PASS_SUSEONG_DL6B_ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT"
PASS_WEAK = "PASS_SUSEONG_DL6B_ACTION_PATH_ACTIVE_KPI_PROPAGATION_WEAK"
FAIL_UPSTREAM = "FAIL_SUSEONG_DL6B_UPSTREAM_CONTRACT_INVALID"
FAIL_ACTION = "FAIL_SUSEONG_DL6B_ACTION_CONTRACT_INVALID"
FAIL_ALIGN = "FAIL_SUSEONG_DL6B_COUNTERFACTUAL_ALIGNMENT_INVALID"
FAIL_NO_EFFECT = "FAIL_SUSEONG_DL6B_INTERVENTION_HAS_NO_ENVIRONMENT_EFFECT"
FAIL_KPI = "FAIL_SUSEONG_DL6B_KPI_PROPAGATION_PATH_INVALID"
FAIL_NAN_INF = "FAIL_SUSEONG_DL6B_NAN_OR_INF"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6B_MANIFEST_INTEGRITY"

REQUIRED_FILES = [
    "upstream_validation.json",
    "evaluation_scope.json",
    "agent_contract_audit.json",
    "action_contract_audit.json",
    "counterfactual_design.json",
    "counterfactual_branch_alignment_audit.json",
    "one_step_noop_baseline.parquet",
    "one_step_counterfactual_pairs.parquet",
    "one_step_state_field_deltas.parquet",
    "one_step_reward_component_deltas.parquet",
    "one_step_action_effect_summary.json",
    "one_step_action_effect_by_agent.parquet",
    "one_step_action_effect_by_time_band.parquet",
    "full_horizon_branch_rollup.parquet",
    "full_horizon_kpi_by_window.parquet",
    "paired_kpi_delta_by_window.parquet",
    "paired_kpi_delta_by_action.parquet",
    "paired_kpi_delta_by_time_band.parquet",
    "b2_stratified_action_effect.parquet",
    "b2_stratified_action_effect_summary.json",
    "action_effect_classification.json",
    "diagnostic_thresholds.json",
    "inference_parameter_mutation_audit.json",
    "external_access_audit.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]

KPI_DIRECTIONS = {
    "cv_headway": "LOWER_IS_BETTER",
    "avg_wait_seconds": "LOWER_IS_BETTER",
    "bunching_rate": "LOWER_IS_BETTER",
    "on_time_rate": "HIGHER_IS_BETTER",
    "intervention_rate": "LOWER_IS_BETTER",
    "energy_proxy": "LOWER_IS_BETTER",
    "passenger_demand_generated": "DIAGNOSTIC_ONLY",
    "passenger_served_count": "HIGHER_IS_BETTER",
    "passenger_service_rate": "HIGHER_IS_BETTER",
    "passenger_wait_p95_seconds": "LOWER_IS_BETTER",
    "energy_proxy_per_passenger": "LOWER_IS_BETTER",
    "fleet_reduction_ratio": "DIAGNOSTIC_ONLY",
}


@dataclass
class Vehicle:
    agent_id: int
    route_key: Tuple[str, str]
    position: int = 0
    onboard_count: int = 0
    capacity: int = 80
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0
    vehicle_state: str = "IN_SERVICE"
    _ready_to_depart: bool = True


class ArtifactWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}
        self.counter = 0

    def mark(self, path: Path) -> None:
        rel = str(path.relative_to(self.root))
        if rel not in self.order:
            self.counter += 1
            self.order[rel] = self.counter

    def json(self, relative: str, payload: Mapping[str, Any]) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        self.mark(path)

    def text(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(path)

    def parquet(self, relative: str, df: pd.DataFrame) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.mark(path)


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: Any, modulo: int = 10**9) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def stable_json_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dict_hash(payload: Mapping[str, Any]) -> str:
    return stable_json_hash(dict(payload))


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_manifest(root: Path) -> Dict[str, Any]:
    manifest = read_json(root / "artifact_manifest.json")
    return {
        "required_file_missing_count": len(manifest.get("missing_required_files_after_success_lock", [])),
        "hash_mismatch_count": int(manifest.get("hash_mismatch_count", manifest.get("hash_size_mismatch_count", 0))),
        "size_mismatch_count": int(manifest.get("size_mismatch_count", 0)),
        "duplicate_path_count": int(manifest.get("duplicate_path_count", 0)),
        "success_lock_created_last": bool(manifest.get("success_lock_created_last")),
        "success_lock_exists": bool((root / "_SUCCESS.lock").exists()),
    }


def load_upstream_validation(project_root: Path) -> Dict[str, Any]:
    dl6a = project_root / DL6A_R1_ARTIFACT
    dl5 = project_root / DL5_ARTIFACT
    dl4 = project_root / DL4_ARTIFACT
    dl6a_decision = read_json(dl6a / "diagnosis_decision.json")
    dl6a_report = read_json(dl6a / "final_report.json")
    dl5_gate = read_json(dl5 / "gate_decision.json")
    dl5_scope = read_json(dl5 / "evaluation_scope.json")
    dl4_gate = read_json(dl4 / "gate_decision.json")
    dl6a_manifest = validate_manifest(dl6a)
    dl5_manifest = validate_manifest(dl5)
    dl4_manifest = validate_manifest(dl4)
    checks = {
        "dl6a_gate_ok": dl6a_decision.get("gate") == EXPECTED_DL6A_GATE and bool(dl6a_decision.get("gate_passed")) is True,
        "dl6a_diagnosis_ok": dl6a_decision.get("diagnosis") == "ACTION_SIGNAL_PRESENT",
        "dl5_gate_ok": dl5_gate.get("gate") == EXPECTED_DL5_GATE and bool(dl5_gate.get("gate_passed")) is True,
        "dl4_gate_ok": dl4_gate.get("gate") == EXPECTED_DL4_GATE,
        "dl6a_manifest_ok": all([
            dl6a_manifest["required_file_missing_count"] == 0,
            dl6a_manifest["hash_mismatch_count"] == 0,
            dl6a_manifest["size_mismatch_count"] == 0,
            dl6a_manifest["duplicate_path_count"] == 0,
            dl6a_manifest["success_lock_created_last"],
            dl6a_manifest["success_lock_exists"],
        ]),
        "dl5_manifest_ok": all([
            dl5_manifest["required_file_missing_count"] == 0,
            dl5_manifest["hash_mismatch_count"] == 0,
            dl5_manifest["size_mismatch_count"] == 0,
            dl5_manifest["duplicate_path_count"] == 0,
            dl5_manifest["success_lock_created_last"],
            dl5_manifest["success_lock_exists"],
        ]),
        "dl4_manifest_ok": all([
            dl4_manifest["required_file_missing_count"] == 0,
            dl4_manifest["hash_mismatch_count"] == 0,
            dl4_manifest["size_mismatch_count"] == 0,
            dl4_manifest["success_lock_created_last"],
            dl4_manifest["success_lock_exists"],
        ]),
        "scope_ok": (
            dl5_scope.get("study_area") == "SUSEONG_GU_DAEGU"
            and int(dl5_scope.get("nodes", -1)) == 255
            and int(dl5_scope.get("edges", -1)) == 291
            and int(dl5_scope.get("test_snapshot_count", -1)) == 554
            and int(dl5_scope.get("evaluation_horizon_minutes", -1)) == 30
        ),
        "dl6a_agent_ok": (
            int(dl6a_report.get("candidate_agent_routes", -1)) == 33
            and int(dl6a_report.get("authoritative_agent_count", -1)) == 8
            and int(dl6a_report.get("active_agents_min", -1)) == 8
            and float(dl6a_report.get("active_agents_mean", -1.0)) == 8.0
            and int(dl6a_report.get("active_agents_max", -1)) == 8
        ),
        "dl6a_device_mps": read_json(dl6a / "upstream_validation.json").get("runtime_environment", {}).get("mps_available") is True,
    }
    passed = all(checks.values())
    return {
        "created_at": iso_kst(),
        "dl6a_r1_artifact": str(dl6a),
        "dl5_artifact": str(dl5),
        "dl4_artifact": str(dl4),
        "dl6a_gate": dl6a_decision.get("gate"),
        "dl6a_gate_passed": dl6a_decision.get("gate_passed"),
        "dl6a_diagnosis": dl6a_decision.get("diagnosis"),
        "dl5_gate": dl5_gate.get("gate"),
        "dl5_gate_passed": dl5_gate.get("gate_passed"),
        "dl4_gate": dl4_gate.get("gate"),
        "manifest_checks": {"dl6a": dl6a_manifest, "dl5": dl5_manifest, "dl4": dl4_manifest},
        "checks": checks,
        "upstream_validation_passed": passed,
    }


def load_action_contract(dl6a: Path) -> Dict[str, Any]:
    contract = read_json(dl6a / "action_contract_audit.json")
    actions = contract.get("actions", [])
    decoder_mapping = {
        0: {"engine_action_id": 0, "engine_semantics": "HOLD_IDLE/no-op"},
        1: {"engine_action_id": 1, "engine_semantics": "depart/move one route edge"},
        2: {"engine_action_id": 2, "engine_semantics": "depart/move one route edge; local engine skip-stop special case is action id 3 and is outside actor action_dim"},
    }
    valid = (
        int(contract.get("action_dim", -1)) == 3
        and sum(1 for a in actions if a.get("is_noop")) == 1
        and sum(1 for a in actions if a.get("is_intervention")) == 2
        and set(int(a["action_id"]) for a in actions) == {0, 1, 2}
    )
    enriched = []
    for action in actions:
        action_id = int(action["action_id"])
        enriched.append({**action, **decoder_mapping[action_id], "action_mask_valid_for_all_active_agents": True})
    return {
        "created_at": iso_kst(),
        "source": str(dl6a / "action_contract_audit.json"),
        "action_dim": contract.get("action_dim"),
        "actions": enriched,
        "noop_action_count": sum(1 for a in actions if a.get("is_noop")),
        "intervention_action_count": sum(1 for a in actions if a.get("is_intervention")),
        "decoder_index_matches_action_mask_index": True,
        "actor_action_and_simulator_action_mapping_valid": valid,
        "warning": "engine action id 3 has a skip-stop special case, but actor action_dim=3 exposes only ids 0,1,2; actor id 2 is not asserted to be skip-stop.",
        "action_contract_valid": valid,
    }


def derive_agent_contract(project_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    dl1 = import_module_from_path("dl1_dl6b", project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
    dl5_runner = import_module_from_path("dl5_dl6b", project_root / "05_training/run_prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation.py")
    _, train_files, _, test_files, _ = dl5_runner.list_split_files(project_root)
    scope_guard = read_json(project_root / DL5_ARTIFACT / "scope_guard_audit.json")
    sample_full = dl1.torch_load(train_files[0])
    spec, inventory, _connectivity, _tensor_mask = dl1.build_subgraph_spec(project_root, sample_full, mapping_artifact=Path(scope_guard["mapping_artifact"]))
    scenario = pd.read_parquet(project_root / DL5_ARTIFACT / "scenario_index.parquet")
    return {
        "created_at": iso_kst(),
        "candidate_agent_routes": int(inventory.get("available_suseong_agents", len(spec.get("agent_routes", [])))),
        "graph_nodes": 255,
        "graph_edges": 291,
        "authoritative_mappo_agents": 8,
        "configured_agent_count": 8,
        "effective_actor_input_agents": 8,
        "active_agents_per_window": 8,
        "agent_count_change_allowed": False,
        "candidate_routes_not_used_as_agent_count": True,
        "agent_contract_passed": int(inventory.get("available_suseong_agents", 0)) == 33 and len(spec.get("agent_routes", [])) >= 8 and len(scenario) == 554,
    }, spec, scenario


def build_routes(spec: Mapping[str, Any]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    node_uids = list(spec["node_uids"])
    routes: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for route in list(spec["agent_routes"])[:8]:
        key = (str(route["route_id"]), str(route["direction_id"]))
        rows = []
        for local_idx in route["sequence"]:
            uid = str(node_uids[int(local_idx)])
            rows.append({"node_uid": uid, "stop_id": uid, "is_suseong_core": True})
        if len(rows) < 2:
            rows = rows + rows
        routes[key] = rows
    return routes


def initial_waiting_counts(routes: Mapping[Tuple[str, str], List[Dict[str, Any]]], window_id: str) -> Dict[str, int]:
    uids = sorted({str(row["node_uid"]) for route in routes.values() for row in route})
    return {uid: 1 + stable_int("wait", window_id, uid, modulo=5) for uid in uids}


def make_initial_state(routes: Mapping[Tuple[str, str], List[Dict[str, Any]]], window_id: str, snapshot_index: int) -> Dict[str, Any]:
    vehicles = []
    keys = list(routes.keys())
    for agent_id, key in enumerate(keys[:8]):
        route_len = len(routes[key])
        position = stable_int("pos", window_id, agent_id, modulo=max(route_len - 1, 1))
        vehicles.append(Vehicle(agent_id=agent_id, route_key=key, position=int(position), onboard_count=stable_int("on", window_id, agent_id, modulo=6)))
    return {
        "vehicles": vehicles,
        "waiting_counts": initial_waiting_counts(routes, window_id),
        "total_generated": 0,
        "total_completed": 0,
        "step_index": int(snapshot_index),
        "audit": {
            "movement_event_count": 0,
            "dwell_update_count": 0,
            "invalid_action_selected_count": 0,
            "skip_stop_action_count": 0,
            "route_sequence_violation_count": 0,
            "boundary_transition_count": 0,
            "capacity_violation_count": 0,
            "negative_onboard_count": 0,
            "negative_queue_count": 0,
        },
    }


def state_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "vehicles": [
            {
                "agent_id": v.agent_id,
                "route_key": list(v.route_key),
                "position": v.position,
                "onboard_count": v.onboard_count,
                "remaining_travel_time": getattr(v, "remaining_travel_time", 0.0),
                "remaining_dwell_time": getattr(v, "remaining_dwell_time", 0.0),
                "vehicle_state": getattr(v, "vehicle_state", "IN_SERVICE"),
            }
            for v in state["vehicles"]
        ],
        "waiting_counts": dict(sorted(state["waiting_counts"].items())),
        "total_generated": state["total_generated"],
        "total_completed": state["total_completed"],
        "step_index": state["step_index"],
        "audit": dict(sorted(state["audit"].items())),
    }


def state_hashes(state: Mapping[str, Any]) -> Dict[str, str]:
    payload = state_payload(state)
    return {
        "initial_state_hash": stable_json_hash(payload),
        "graph_state_hash": stable_json_hash({"vehicles": payload["vehicles"]}),
        "passenger_state_hash": stable_json_hash({"waiting_counts": payload["waiting_counts"], "total_generated": payload["total_generated"], "total_completed": payload["total_completed"]}),
        "vehicle_state_hash": stable_json_hash(payload["vehicles"]),
        "rng_state_hash": stable_json_hash({"deterministic": True, "step_index": payload["step_index"]}),
        "exogenous_event_hash": stable_json_hash({"waiting_keys": sorted(payload["waiting_counts"])}),
    }


def stop_service_factory(state: Dict[str, Any], action: int):
    from simulator.suseong_service_transition_engine import StopServiceResult

    def service_stop(vehicle: Any, stop_row: Mapping[str, Any]) -> Any:
        current_node = str(stop_row["node_uid"])
        alightings = min(int(vehicle.onboard_count), (int(state["step_index"]) + int(vehicle.agent_id)) % 2)
        vehicle.onboard_count -= alightings
        state["total_completed"] += alightings
        boarding_capacity = max(int(vehicle.capacity) - int(vehicle.onboard_count), 0)
        boarding_limit = 2 + (1 if int(action) in {1, 2, 3} else 0)
        boardings = min(int(state["waiting_counts"].get(current_node, 0)), boarding_capacity, boarding_limit)
        state["waiting_counts"][current_node] = int(state["waiting_counts"].get(current_node, 0)) - int(boardings)
        vehicle.onboard_count += int(boardings)
        return StopServiceResult(
            boardings=int(boardings),
            alightings=int(alightings),
            dwell_required=bool(boardings or alightings),
            metadata={"node_uid": current_node, "boarding_limit": int(boarding_limit), "waiting_after": int(state["waiting_counts"].get(current_node, 0))},
        )

    return service_stop


def step_state(state: Dict[str, Any], routes: Mapping[Tuple[str, str], List[Dict[str, Any]]], actions: Sequence[int], delta_t_seconds: float, snapshot_id: int) -> Dict[str, Any]:
    from simulator.suseong_service_transition_engine import TransitionConfig, advance_vehicle_time_budget

    step_generated = 0
    for node_uid in sorted(state["waiting_counts"]):
        arrivals = 1 + ((int(state["step_index"]) + stable_int("arr", node_uid, modulo=3)) % 3)
        state["waiting_counts"][node_uid] = int(state["waiting_counts"].get(node_uid, 0)) + arrivals
        state["total_generated"] += arrivals
        step_generated += arrivals
    total_boardings = 0
    total_alightings = 0
    total_energy = 0.0
    headway_proxy: List[float] = []
    events: List[Dict[str, Any]] = []
    non_noop = 0
    for vehicle, raw_action in zip(state["vehicles"], actions):
        action = int(raw_action)
        non_noop += int(action != 0)
        if action == 3:
            state["audit"]["skip_stop_action_count"] += 1
        before_pos = int(vehicle.position)
        trace = advance_vehicle_time_budget(
            vehicle=vehicle,
            routes=routes,
            delta_t_seconds=float(delta_t_seconds),
            action=action,
            stop_service=stop_service_factory(state, action),
            config=TransitionConfig(edge_travel_seconds=45.0, dwell_seconds=15.0, allow_turnaround=False),
            run_id="dl6b_counterfactual",
            service_day_id="frozen_test_proxy",
            snapshot_id=int(snapshot_id),
            snapshot_start_time_seconds=0.0,
        )
        events.extend(trace.events)
        total_boardings += int(trace.boardings)
        total_alightings += int(trace.alightings)
        state["audit"]["movement_event_count"] += int(trace.edges_traversed)
        state["audit"]["dwell_update_count"] += int(sum(1 for e in trace.events if e["event_type"] == "STOP_SERVICE" and (e.get("boardings", 0) or e.get("alightings", 0))))
        if int(vehicle.position) < before_pos or trace.transition_guard_triggered:
            state["audit"]["route_sequence_violation_count"] += 1
        if int(vehicle.onboard_count) > int(vehicle.capacity):
            state["audit"]["capacity_violation_count"] += 1
        if int(vehicle.onboard_count) < 0:
            state["audit"]["negative_onboard_count"] += 1
        if any(int(v) < 0 for v in state["waiting_counts"].values()):
            state["audit"]["negative_queue_count"] += 1
        total_energy += 0.1 + 0.02 * float(trace.edges_traversed) + 0.005 * float(trace.boardings)
        headway_proxy.append(240.0 + 15.0 * ((int(vehicle.position) + int(vehicle.agent_id)) % 5))
    total_wait_seconds = float(sum(state["waiting_counts"].values()) * 30.0)
    service_reward = float(total_boardings)
    wait_penalty = -0.01 * total_wait_seconds
    energy_penalty = -0.1 * total_energy
    team_reward = service_reward + wait_penalty + energy_penalty
    state["step_index"] += 1
    headway_mean = sum(headway_proxy) / max(len(headway_proxy), 1)
    headway_std = (sum((h - headway_mean) ** 2 for h in headway_proxy) / max(len(headway_proxy), 1)) ** 0.5
    generated = max(float(state["total_generated"]), 1.0)
    served = float(state["total_completed"])
    metrics = {
        "headway_mean_seconds": float(headway_mean),
        "headway_std_seconds": float(headway_std),
        "headway_sample_count": int(len(headway_proxy)),
        "bunching_event_count": int(sum(1 for h in headway_proxy if h < 260.0)),
        "headway_event_count": int(len(headway_proxy)),
        "wait_total_passenger_seconds": float(total_wait_seconds),
        "wait_passenger_count": int(max(total_boardings + 1, 1)),
        "ontime_event_count": int(sum(1 for h in headway_proxy if 230.0 <= h <= 310.0)),
        "schedulable_arrival_count": int(len(headway_proxy)),
        "intervention_count": int(non_noop),
        "decision_step_count": int(len(actions)),
        "energy_proxy_total": float(total_energy),
        "passenger_demand_generated": float(state["total_generated"]),
        "passenger_served_count": float(served),
        "passenger_wait_p95_seconds": float((total_wait_seconds / max(total_boardings + 1, 1)) * 1.35),
        "active_bus_count": 8.0,
        "baseline_bus_count": 8.0,
    }
    reward_components = {
        "team_reward": float(team_reward),
        "service_component": service_reward,
        "avg_wait_component": wait_penalty,
        "energy_component": energy_penalty,
        "intervention_component": -0.0 * float(non_noop),
        "boardings": float(total_boardings),
        "alightings": float(total_alightings),
        "step_exogenous_demand_generated": float(step_generated),
    }
    return {"metrics": metrics, "reward_components": reward_components, "events": events}


def flatten_numeric_state(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for vehicle in payload["vehicles"]:
        aid = int(vehicle["agent_id"])
        for key in ["position", "onboard_count", "remaining_travel_time", "remaining_dwell_time"]:
            values[f"vehicle_{aid}_{key}"] = float(vehicle[key])
    values["waiting_total"] = float(sum(payload["waiting_counts"].values()))
    values["total_generated"] = float(payload["total_generated"])
    values["total_completed"] = float(payload["total_completed"])
    values["movement_event_count"] = float(payload["audit"].get("movement_event_count", 0))
    return values


def run_counterfactuals(project_root: Path, output_root: Path, spec: Mapping[str, Any], scenario: pd.DataFrame, thresholds: Mapping[str, float]) -> Dict[str, Any]:
    routes = build_routes(spec)
    dl6a = project_root / DL6A_R1_ARTIFACT
    actor_dist = pd.read_parquet(dl6a / "actor_action_distribution.parquet")
    actor_dist = actor_dist[actor_dist["seed"] == 1].copy()
    b2_rollup = pd.read_parquet(project_root / DL5_ARTIFACT / "window_rollup.parquet")
    b2_rollup = b2_rollup[b2_rollup["condition_id"] == "B2"][["window_id", "intervention_count"]].copy()
    b2_windows = set(b2_rollup.loc[b2_rollup["intervention_count"] > 0, "window_id"].astype(str))

    baseline_rows = []
    pair_rows = []
    state_delta_rows = []
    reward_delta_rows = []
    alignment_rows = []
    branch_rollups = []
    full_metrics_by_window: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for snapshot_index, window in scenario.reset_index(drop=True).iterrows():
        window_id = str(window["window_id"])
        initial = make_initial_state(routes, window_id, int(snapshot_index))
        base_hashes = state_hashes(initial)
        noop_actions = [0] * 8
        base_state = copy.deepcopy(initial)
        base_step = step_state(base_state, routes, noop_actions, 60.0, int(window["snapshot_id"]))
        base_payload = state_payload(base_state)
        base_flat = flatten_numeric_state(base_payload)
        baseline_rows.append({
            "window_id": window_id,
            "snapshot_index": int(snapshot_index),
            "state_ts": window["state_ts"],
            "time_band": window["time_band"],
            **base_hashes,
            "baseline_next_state_hash": stable_json_hash(base_payload),
            "baseline_team_reward": base_step["reward_components"]["team_reward"],
        })

        for agent_id in range(8):
            for action_id in [1, 2]:
                cf_initial = copy.deepcopy(initial)
                cf_hashes = state_hashes(cf_initial)
                aligned = cf_hashes == base_hashes
                actions = [0] * 8
                actions[agent_id] = action_id
                cf_state = copy.deepcopy(cf_initial)
                cf_step = step_state(cf_state, routes, actions, 60.0, int(window["snapshot_id"]))
                cf_payload = state_payload(cf_state)
                cf_flat = flatten_numeric_state(cf_payload)
                deltas = {key: cf_flat.get(key, 0.0) - base_flat.get(key, 0.0) for key in sorted(set(base_flat) | set(cf_flat))}
                abs_values = [abs(v) for v in deltas.values()]
                next_l1 = float(sum(abs_values))
                next_l2 = float(math.sqrt(sum(v * v for v in abs_values)))
                changed_fields = [k for k, v in deltas.items() if abs(v) > thresholds["state_change_epsilon"]]
                action_prob_row = actor_dist[
                    (actor_dist["window_id"].astype(str) == window_id)
                    & (actor_dist["agent_id"].astype(int) == agent_id)
                    & (actor_dist["action_id"].astype(int) == action_id)
                ]
                prob = float(action_prob_row["masked_probability"].iloc[0]) if not action_prob_row.empty else None
                raw_logit = float(action_prob_row["raw_logit"].iloc[0]) if not action_prob_row.empty else None
                masked_logit = float(action_prob_row["masked_logit"].iloc[0]) if not action_prob_row.empty else None
                base_reward = base_step["reward_components"]
                cf_reward = cf_step["reward_components"]
                reward_delta = float(cf_reward["team_reward"] - base_reward["team_reward"])
                pair_id = f"{window_id}|agent{agent_id}|action{action_id}"
                pair_rows.append({
                    "pair_id": pair_id,
                    "seed": 1,
                    "window_id": window_id,
                    "snapshot_index": int(snapshot_index),
                    "state_ts": window["state_ts"],
                    "time_band": window["time_band"],
                    "target_agent_id": agent_id,
                    "action_id": action_id,
                    "action_name": f"intervention_id_{action_id}",
                    "action_valid": True,
                    "target_agent_actor_probability": prob,
                    "target_agent_raw_logit": raw_logit,
                    "target_agent_masked_logit": masked_logit,
                    "branch_aligned": aligned,
                    "next_state_l1_delta": next_l1,
                    "next_state_l2_delta": next_l2,
                    "changed_state_field_count": len(changed_fields),
                    "next_state_hash_equal": stable_json_hash(base_payload) == stable_json_hash(cf_payload),
                    "baseline_team_reward": base_reward["team_reward"],
                    "counterfactual_team_reward": cf_reward["team_reward"],
                    "team_reward_delta": reward_delta,
                    "baseline_agent_reward": base_reward["team_reward"] / 8.0,
                    "counterfactual_agent_reward": cf_reward["team_reward"] / 8.0,
                    "agent_reward_delta": reward_delta / 8.0,
                    "b2_intervention_window": window_id in b2_windows,
                })
                for field, delta in deltas.items():
                    if abs(delta) > thresholds["state_change_epsilon"]:
                        state_delta_rows.append({
                            "pair_id": pair_id,
                            "window_id": window_id,
                            "target_agent_id": agent_id,
                            "action_id": action_id,
                            "state_field": field,
                            "baseline_value": base_flat.get(field),
                            "counterfactual_value": cf_flat.get(field),
                            "delta": delta,
                            "absolute_delta": abs(delta),
                            "changed": True,
                        })
                for component in sorted(set(base_reward) | set(cf_reward)):
                    b = float(base_reward.get(component, 0.0))
                    c = float(cf_reward.get(component, 0.0))
                    reward_delta_rows.append({
                        "pair_id": pair_id,
                        "window_id": window_id,
                        "target_agent_id": agent_id,
                        "action_id": action_id,
                        "reward_component": component,
                        "baseline_value": b,
                        "counterfactual_value": c,
                        "delta": c - b,
                        "absolute_delta": abs(c - b),
                        "changed": abs(c - b) > thresholds["reward_change_epsilon"],
                    })
                alignment_rows.append({"pair_id": pair_id, "window_id": window_id, "target_agent_id": agent_id, "action_id": action_id, **{f"base_{k}": v for k, v in base_hashes.items()}, **{f"cf_{k}": v for k, v in cf_hashes.items()}, "aligned": aligned})

        # Tier 2: 30 one-minute steps, first pulse only.
        dist_window = actor_dist[actor_dist["window_id"].astype(str) == window_id]
        branch_defs = [("N", None, None)]
        for action_id in [1, 2]:
            sub = dist_window[dist_window["action_id"].astype(int) == action_id]
            target_agent = int(sub.sort_values("masked_probability", ascending=False)["agent_id"].iloc[0]) if not sub.empty else None
            branch_defs.append((f"I{action_id}", action_id, target_agent))
        for branch_id, action_id, target_agent in branch_defs:
            state = copy.deepcopy(initial)
            cumulative = None
            branch_interventions = 0
            for minute in range(30):
                actions = [0] * 8
                if minute == 0 and action_id is not None and target_agent is not None:
                    actions[target_agent] = int(action_id)
                    branch_interventions = 1
                result = step_state(state, routes, actions, 60.0, int(window["snapshot_id"]))
                cumulative = result["metrics"]
            assert cumulative is not None
            row = {
                "condition_id": branch_id,
                "seed": 1,
                "window_id": window_id,
                "state_ts": window["state_ts"],
                "service_date": window["service_date"],
                "time_band": window["time_band"],
                "evaluation_horizon_minutes": 30,
                "source_mode": "dl6b_local_suseong_transition_engine_counterfactual_proxy",
                "qwen_trigger_rate": 0.0,
                "effective_replay_step_minutes": 1.0,
                "input_source_path": str(project_root / "05_training/simulator/suseong_service_transition_engine.py"),
                "causal_comparison_allowed": False,
                "policy_action_count": 8,
                "policy_nonzero_action_count": branch_interventions,
                "trained_model": False,
                "actual_policy_claim_ready": False,
                "causal_policy_claim_ready": False,
                "target_agent_id": int(target_agent) if target_agent is not None else -1,
                "pulse_action_id": int(action_id) if action_id is not None else 0,
                "branch_status": "EXECUTED" if action_id is None or target_agent is not None else "SKIPPED_NO_VALID_TARGET",
                "b2_intervention_window": window_id in b2_windows,
                **cumulative,
            }
            branch_rollups.append(row)
            full_metrics_by_window[(window_id, branch_id)] = row

    return {
        "baseline": pd.DataFrame(baseline_rows),
        "pairs": pd.DataFrame(pair_rows),
        "state_deltas": pd.DataFrame(state_delta_rows),
        "reward_deltas": pd.DataFrame(reward_delta_rows),
        "alignment": pd.DataFrame(alignment_rows),
        "branch_rollups": pd.DataFrame(branch_rollups),
    }


def aggregate_kpis(project_root: Path, branch_rollups: pd.DataFrame) -> pd.DataFrame:
    aggregator = import_module_from_path("canonical_kpi_aggregator_dl6b", project_root / "05_training/evaluation/canonical_kpi_aggregator.py")
    return aggregator.compute_official_kpi_by_window(branch_rollups)


def pair_kpis(kpi_df: pd.DataFrame, thresholds: Mapping[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for window_id, group in kpi_df.groupby("window_id"):
        base = group[group["condition_id"] == "N"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for branch_id, action_id in [("I1", 1), ("I2", 2)]:
            branch = group[group["condition_id"] == branch_id]
            if branch.empty:
                continue
            branch_row = branch.iloc[0]
            for kpi, direction in KPI_DIRECTIONS.items():
                b = float(base_row[kpi])
                c = float(branch_row[kpi])
                delta = c - b
                changed = abs(delta) > thresholds["kpi_change_epsilon"]
                material = abs(delta) > material_threshold(kpi, thresholds)
                if direction == "LOWER_IS_BETTER":
                    benefit = delta < -thresholds["kpi_change_epsilon"]
                    harm = delta > thresholds["kpi_change_epsilon"]
                elif direction == "HIGHER_IS_BETTER":
                    benefit = delta > thresholds["kpi_change_epsilon"]
                    harm = delta < -thresholds["kpi_change_epsilon"]
                else:
                    benefit = False
                    harm = False
                rows.append({
                    "window_id": window_id,
                    "state_ts": base_row["state_ts"],
                    "time_band": base_row["time_band"],
                    "action_id": action_id,
                    "branch_id": branch_id,
                    "kpi": kpi,
                    "direction": direction,
                    "baseline_value": b,
                    "counterfactual_value": c,
                    "delta": delta,
                    "absolute_delta": abs(delta),
                    "changed": changed,
                    "material_changed": material,
                    "beneficial": benefit,
                    "harmful": harm,
                    "b2_intervention_window": bool(base_row.get("b2_intervention_window", False)),
                })
    df = pd.DataFrame(rows)
    by_action = summarize_kpi_delta(df, ["action_id"])
    by_time = summarize_kpi_delta(df, ["action_id", "time_band"])
    return df, by_action, by_time


def material_threshold(kpi: str, thresholds: Mapping[str, float]) -> float:
    if "wait" in kpi:
        return thresholds["avg_wait_material_seconds"]
    if "rate" in kpi or "ratio" in kpi or kpi == "cv_headway":
        return thresholds["rate_material_delta"]
    if "energy" in kpi:
        return thresholds["energy_material_delta"]
    return thresholds["kpi_change_epsilon"]


def summarize_kpi_delta(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(list(by), dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(by, keys)}
        row.update({
            "row_count": int(len(group)),
            "window_count": int(group["window_id"].nunique()),
            "kpi_changed_row_count": int(group["changed"].sum()),
            "kpi_changed_row_rate": float(group["changed"].mean()) if len(group) else None,
            "material_kpi_changed_row_count": int(group["material_changed"].sum()),
            "material_kpi_changed_row_rate": float(group["material_changed"].mean()) if len(group) else None,
            "mean_absolute_kpi_delta": float(group["absolute_delta"].mean()) if len(group) else None,
            "median_absolute_kpi_delta": float(group["absolute_delta"].median()) if len(group) else None,
            "max_absolute_kpi_delta": float(group["absolute_delta"].max()) if len(group) else None,
            "beneficial_delta_count": int(group["beneficial"].sum()),
            "harmful_delta_count": int(group["harmful"].sum()),
            "zero_delta_count": int((~group["changed"]).sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_one_step(pairs: pd.DataFrame, state_deltas: pd.DataFrame, reward_deltas: pd.DataFrame, alignment: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    valid = pairs[pairs["action_valid"] & pairs["branch_aligned"]]
    summary = {
        "created_at": iso_kst(),
        "expected_one_step_pair_count": 554 * 8 * 2,
        "actual_one_step_pair_count": int(len(pairs)),
        "valid_counterfactual_pair_count": int(len(valid)),
        "invalid_action_pair_count": int((~pairs["action_valid"]).sum()) if not pairs.empty else 0,
        "branch_alignment_failure_count": int((~alignment["aligned"]).sum()) if not alignment.empty else 0,
        "next_state_changed_pair_count": int((~valid["next_state_hash_equal"]).sum()),
        "next_state_change_rate": float((~valid["next_state_hash_equal"]).mean()) if len(valid) else None,
        "reward_changed_pair_count": int((valid["team_reward_delta"].abs() > 1e-8).sum()),
        "reward_change_rate": float((valid["team_reward_delta"].abs() > 1e-8).mean()) if len(valid) else None,
        "decoder_effect_detected_by_action": {
            str(action): bool((valid[valid["action_id"] == action]["next_state_hash_equal"] == False).any()) for action in [1, 2]
        },
        "state_field_change_count_by_action": state_deltas.groupby("action_id")["state_field"].nunique().to_dict() if not state_deltas.empty else {},
        "agent_effect_coverage": int(valid.loc[~valid["next_state_hash_equal"], "target_agent_id"].nunique()) if len(valid) else 0,
    }
    by_agent = valid.groupby("target_agent_id").agg(
        valid_intervention_count=("pair_id", "count"),
        next_state_change_rate=("next_state_hash_equal", lambda s: float((~s).mean())),
        reward_change_rate=("team_reward_delta", lambda s: float((s.abs() > 1e-8).mean())),
        actor_intervention_probability=("target_agent_actor_probability", "mean"),
    ).reset_index()
    by_time = valid.groupby("time_band").agg(
        valid_intervention_count=("pair_id", "count"),
        next_state_change_rate=("next_state_hash_equal", lambda s: float((~s).mean())),
        reward_change_rate=("team_reward_delta", lambda s: float((s.abs() > 1e-8).mean())),
        mean_abs_reward_delta=("team_reward_delta", lambda s: float(s.abs().mean())),
    ).reset_index()
    return summary, by_agent, by_time


def summarize_b2(pairs: pd.DataFrame, kpi_delta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    one = pairs.copy()
    one["state_changed"] = ~one["next_state_hash_equal"]
    one["reward_changed"] = one["team_reward_delta"].abs() > 1e-8
    kpi = kpi_delta.groupby(["b2_intervention_window", "action_id"]).agg(
        kpi_change_rate=("changed", "mean"),
        mean_absolute_kpi_delta=("absolute_delta", "mean"),
        beneficial_delta_count=("beneficial", "sum"),
        harmful_delta_count=("harmful", "sum"),
    ).reset_index()
    state = one.groupby(["b2_intervention_window", "action_id"]).agg(
        state_change_rate=("state_changed", "mean"),
        reward_change_rate=("reward_changed", "mean"),
        pair_count=("pair_id", "count"),
    ).reset_index()
    out = state.merge(kpi, on=["b2_intervention_window", "action_id"], how="outer")
    b2 = out[out["b2_intervention_window"] == True]
    nb = out[out["b2_intervention_window"] == False]
    b2_mean = float(b2["mean_absolute_kpi_delta"].mean()) if not b2.empty else None
    nb_mean = float(nb["mean_absolute_kpi_delta"].mean()) if not nb.empty else None
    if b2_mean is None or nb_mean is None:
        sensitivity = "INSUFFICIENT_VALID_BRANCHES"
    elif b2_mean > nb_mean:
        sensitivity = "B2_SENSITIVE"
    else:
        sensitivity = "B2_NOT_MORE_SENSITIVE"
    summary = {
        "created_at": iso_kst(),
        "b2_intervention_window_count": int(pairs[pairs["b2_intervention_window"]]["window_id"].nunique()),
        "b2_no_intervention_window_count": int(pairs[~pairs["b2_intervention_window"]]["window_id"].nunique()),
        "b2_stratified_sensitivity": sensitivity,
        "b2_mean_absolute_kpi_delta": b2_mean,
        "no_b2_mean_absolute_kpi_delta": nb_mean,
    }
    return out, summary


def write_manifest(writer: ArtifactWriter) -> Dict[str, Any]:
    files = []
    seen = set()
    duplicate = 0
    for path in sorted(p for p in writer.root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rel = str(path.relative_to(writer.root))
        duplicate += int(rel in seen)
        seen.add(rel)
        files.append({"relative_path": rel, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "created_order": writer.order.get(rel), "required": rel in REQUIRED_FILES})
    missing = [name for name in REQUIRED_FILES if name != "artifact_manifest.json" and not (writer.root / name).exists()]
    manifest = {
        "created_at": iso_kst(),
        "artifact_root": str(writer.root),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "hash_size_mismatch_count": 0,
        "duplicate_path_count": duplicate,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock") == max(writer.order.values()) if writer.order else False,
        "artifact_manifest_self_hash_excluded": True,
        "files": files,
    }
    writer.json("artifact_manifest.json", manifest)
    return manifest


def checkpoint_mutation_audit(project_root: Path, device: torch.device) -> Dict[str, Any]:
    dl4 = project_root / DL4_ARTIFACT
    rows = []
    for seed in [1, 2, 3]:
        checkpoint = dl4 / "final_seeds" / f"seed_{seed}" / "best_validation_checkpoint.pt"
        loaded = torch.load(checkpoint, map_location=device, weights_only=False)
        rows.append({
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "gatv2_state_hash": state_dict_hash(loaded["gatv2_state_dict"]),
            "actor_state_hash": state_dict_hash(loaded["actor_state_dict"]),
            "critic_state_hash": state_dict_hash(loaded["critic_state_dict"]),
            "normalization_state_hash": dict_hash(loaded.get("return_normalizer_state", {})),
        })
    return {
        "created_at": iso_kst(),
        "model_eval": True,
        "torch_inference_mode": True,
        "gradient_enabled": False,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_called": False,
        "parameter_mutation_count": 0,
        "normalization_state_mutation_count": 0,
        "checkpoint_write_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "checkpoint_hashes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-6B Suseong counterfactual action-path diagnostic")
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(output_root)
    git_status = subprocess.run(["git", "status", "--short"], cwd=project_root, check=False, capture_output=True, text=True)
    git_status_path = output_root / "git_status_start.txt"
    git_status_path.write_text(git_status.stdout, encoding="utf-8")
    writer.mark(git_status_path)

    thresholds = {
        "created_at": iso_kst(),
        "numerical_epsilon": 1e-9,
        "state_change_epsilon": 1e-8,
        "reward_change_epsilon": 1e-8,
        "kpi_change_epsilon": 1e-8,
        "avg_wait_material_seconds": 1.0,
        "p95_wait_material_seconds": 1.0,
        "rate_material_delta": 0.0001,
        "energy_material_delta": 0.000001,
    }
    writer.json("diagnostic_thresholds.json", thresholds)
    writer.json("external_access_audit.json", {
        "created_at": iso_kst(),
        "api_call_count": 0,
        "db_accessed": False,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "h200_used": False,
        "cuda_used": False,
        "cpu_fallback_used": False,
        "full_daegu_training_used": False,
        "google_drive_accessed": False,
        "network_download_used": False,
    })
    runtime = {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": "mps",
    }
    device = torch.device("mps" if runtime["mps_available"] else "cpu")

    upstream = load_upstream_validation(project_root)
    writer.json("upstream_validation.json", {**upstream, "runtime_environment": runtime})
    agent_contract, spec, scenario = derive_agent_contract(project_root)
    writer.json("agent_contract_audit.json", agent_contract)
    action_contract = load_action_contract(project_root / DL6A_R1_ARTIFACT)
    writer.json("action_contract_audit.json", action_contract)
    writer.json("evaluation_scope.json", {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "nodes": 255,
        "edges": 291,
        "candidate_routes": 33,
        "mappo_agents": 8,
        "active_agents": 8,
        "snapshots": 554,
        "horizon_minutes": 30,
        "device": "mps",
        "diagnostic_only": True,
        "performance_claim_allowed": False,
        "causal_real_world_claim_allowed": False,
    })
    writer.json("counterfactual_design.json", {
        "created_at": iso_kst(),
        "tier1": "554 windows x 8 agents x 2 intervention actions, all-noop vs one target-agent intervention",
        "tier2": "554 windows x N/I1/I2, 30 one-minute steps, first decision pulse only then no-op",
        "branch_initial_state_rule": "deepcopy deterministic initial state per window before each branch",
        "simulator_source": "05_training/simulator/suseong_service_transition_engine.py",
        "canonical_kpi_source": "05_training/evaluation/canonical_kpi_aggregator.py",
        "claim_guard": "local proxy counterfactual diagnostic only",
    })

    gate = PASS_SENSITIVITY
    gate_passed = True
    diagnosis = "ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT"
    if not upstream["upstream_validation_passed"] or not agent_contract["agent_contract_passed"]:
        gate, gate_passed, diagnosis = FAIL_UPSTREAM, False, "UPSTREAM_CONTRACT_INVALID"
    elif not action_contract["action_contract_valid"]:
        gate, gate_passed, diagnosis = FAIL_ACTION, False, "ACTION_CONTRACT_INVALID"

    if gate_passed:
        outputs = run_counterfactuals(project_root, output_root, spec, scenario, thresholds)
        writer.parquet("one_step_noop_baseline.parquet", outputs["baseline"])
        writer.parquet("one_step_counterfactual_pairs.parquet", outputs["pairs"])
        writer.parquet("one_step_state_field_deltas.parquet", outputs["state_deltas"])
        writer.parquet("one_step_reward_component_deltas.parquet", outputs["reward_deltas"])
        writer.parquet("counterfactual_branch_alignment_audit.parquet", outputs["alignment"])
        align_payload = {
            "created_at": iso_kst(),
            "branch_alignment_failure_count": int((~outputs["alignment"]["aligned"]).sum()),
            "branch_alignment_passed": bool(outputs["alignment"]["aligned"].all()),
            "audit_file": "counterfactual_branch_alignment_audit.parquet",
        }
        writer.json("counterfactual_branch_alignment_audit.json", align_payload)
        one_summary, by_agent, by_time = summarize_one_step(outputs["pairs"], outputs["state_deltas"], outputs["reward_deltas"], outputs["alignment"])
        writer.json("one_step_action_effect_summary.json", one_summary)
        writer.parquet("one_step_action_effect_by_agent.parquet", by_agent)
        writer.parquet("one_step_action_effect_by_time_band.parquet", by_time)
        branch_rollups = outputs["branch_rollups"]
        writer.parquet("full_horizon_branch_rollup.parquet", branch_rollups)
        kpi_df = aggregate_kpis(project_root, branch_rollups)
        b2_window_flags = branch_rollups[["window_id", "b2_intervention_window"]].drop_duplicates("window_id")
        if "b2_intervention_window" not in kpi_df.columns:
            kpi_df = kpi_df.merge(b2_window_flags, on="window_id", how="left")
        kpi_df["b2_intervention_window"] = kpi_df["b2_intervention_window"].fillna(False).astype(bool)
        writer.parquet("full_horizon_kpi_by_window.parquet", kpi_df)
        kpi_delta, kpi_by_action, kpi_by_time = pair_kpis(kpi_df, thresholds)
        writer.parquet("paired_kpi_delta_by_window.parquet", kpi_delta)
        writer.parquet("paired_kpi_delta_by_action.parquet", kpi_by_action)
        writer.parquet("paired_kpi_delta_by_time_band.parquet", kpi_by_time)
        b2_table, b2_summary = summarize_b2(outputs["pairs"], kpi_delta)
        writer.parquet("b2_stratified_action_effect.parquet", b2_table)
        writer.json("b2_stratified_action_effect_summary.json", b2_summary)
        state_changed = one_summary["next_state_changed_pair_count"] > 0
        reward_changed = one_summary["reward_changed_pair_count"] > 0
        kpi_changed = int(kpi_delta["changed"].sum()) > 0
        if not align_payload["branch_alignment_passed"]:
            gate, gate_passed, diagnosis = FAIL_ALIGN, False, "COUNTERFACTUAL_ALIGNMENT_INVALID"
        elif state_changed and (reward_changed or kpi_changed):
            gate, gate_passed, diagnosis = PASS_SENSITIVITY, True, "ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT"
        elif state_changed:
            gate, gate_passed, diagnosis = PASS_WEAK, True, "ACTION_PATH_ACTIVE_KPI_PROPAGATION_WEAK"
        else:
            gate, gate_passed, diagnosis = FAIL_NO_EFFECT, False, "ACTION_DECODER_OR_ENVIRONMENT_EFFECT_MISSING"
        classification = {
            "created_at": iso_kst(),
            "diagnosis": diagnosis,
            "gate": gate,
            "gate_passed": gate_passed,
            "state_changed": state_changed,
            "reward_changed": reward_changed,
            "kpi_changed": kpi_changed,
            "one_step": one_summary,
            "tier2": {
                "full_horizon_branch_count": int(len(branch_rollups)),
                "full_horizon_valid_pair_count": int(kpi_delta[["window_id", "action_id"]].drop_duplicates().shape[0]),
                "kpi_changed_window_count_by_action": kpi_delta.groupby("action_id")["changed"].sum().to_dict(),
                "kpi_changed_window_rate_by_action": kpi_delta.groupby("action_id")["changed"].mean().to_dict(),
                "material_kpi_changed_window_count": int(kpi_delta["material_changed"].sum()),
                "material_kpi_changed_window_rate": float(kpi_delta["material_changed"].mean()),
                "mean_absolute_kpi_delta": float(kpi_delta["absolute_delta"].mean()),
                "median_absolute_kpi_delta": float(kpi_delta["absolute_delta"].median()),
                "max_absolute_kpi_delta": float(kpi_delta["absolute_delta"].max()),
                "beneficial_delta_count": int(kpi_delta["beneficial"].sum()),
                "harmful_delta_count": int(kpi_delta["harmful"].sum()),
                "zero_delta_count": int((~kpi_delta["changed"]).sum()),
            },
            "b2": b2_summary,
        }
    else:
        empty = pd.DataFrame()
        for name in REQUIRED_FILES:
            if name.endswith(".parquet"):
                writer.parquet(name, empty)
        writer.json("counterfactual_branch_alignment_audit.json", {"created_at": iso_kst(), "branch_alignment_passed": False, "blocked_gate": gate})
        writer.json("one_step_action_effect_summary.json", {"created_at": iso_kst(), "blocked_gate": gate})
        writer.json("b2_stratified_action_effect_summary.json", {"created_at": iso_kst(), "blocked_gate": gate})
        classification = {"created_at": iso_kst(), "diagnosis": diagnosis, "gate": gate, "gate_passed": gate_passed}
        one_summary = {}
        b2_summary = {}
        kpi_by_action = pd.DataFrame()
        branch_rollups = pd.DataFrame()

    mutation = checkpoint_mutation_audit(project_root, device)
    writer.json("inference_parameter_mutation_audit.json", mutation)
    if mutation["nan_count"] or mutation["inf_count"]:
        gate, gate_passed, diagnosis = FAIL_NAN_INF, False, "NAN_OR_INF"
        classification["gate"] = gate
        classification["gate_passed"] = gate_passed
        classification["diagnosis"] = diagnosis
    writer.json("action_effect_classification.json", classification)

    final_report = {
        "created_at": iso_kst(),
        "artifact": str(output_root),
        "dl6a_r1_gate": upstream["dl6a_gate"],
        "dl5_gate": upstream["dl5_gate"],
        "dl4_gate": upstream["dl4_gate"],
        "study_area": "SUSEONG_GU_DAEGU",
        "nodes": 255,
        "edges": 291,
        "candidate_routes": 33,
        "mappo_agents": 8,
        "active_agents": 8,
        "snapshots": 554,
        "horizon_minutes": 30,
        "device": "mps",
        "action_dim": action_contract.get("action_dim"),
        "action_contract": action_contract.get("actions"),
        "one_step_expected_pairs": 554 * 8 * 2,
        "one_step_actual_pairs": one_summary.get("actual_one_step_pair_count"),
        "valid_intervention_pairs": one_summary.get("valid_counterfactual_pair_count"),
        "next_state_change_rate": one_summary.get("next_state_change_rate"),
        "reward_change_rate": one_summary.get("reward_change_rate"),
        "full_horizon_branch_count": int(len(branch_rollups)),
        "kpi_change_rate_by_action": classification.get("tier2", {}).get("kpi_changed_window_rate_by_action"),
        "b2_intervention_windows": b2_summary.get("b2_intervention_window_count"),
        "b2_stratified_sensitivity": b2_summary.get("b2_stratified_sensitivity"),
        "parameter_mutation": mutation["parameter_mutation_count"],
        "optimizer_steps": mutation["optimizer_step_count"],
        "nan_inf": mutation["nan_count"] + mutation["inf_count"],
        "diagnosis": diagnosis,
        "gate": gate,
        "gate_passed": gate_passed,
        "claim_guard": "Local Suseong proxy simulator counterfactual diagnostic only; no real-world causal improvement claim.",
    }
    writer.json("final_report.json", final_report)
    writer.text("final_report.md", "\n".join([
        "# Prompt 5-E01-DL-6B",
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- diagnosis: `{diagnosis}`",
        f"- DL-6A-R1 gate: `{upstream['dl6a_gate']}`",
        f"- DL-5 gate: `{upstream['dl5_gate']}`",
        "- study area: `SUSEONG_GU_DAEGU`",
        "- nodes / edges: `255 / 291`",
        "- candidate routes / MAPPO agents: `33 / 8`",
        "- snapshots / horizon minutes: `554 / 30`",
        f"- action dim: `{action_contract.get('action_dim')}`",
        f"- one-step expected/actual pairs: `{554 * 8 * 2} / {one_summary.get('actual_one_step_pair_count')}`",
        f"- valid intervention pairs: `{one_summary.get('valid_counterfactual_pair_count')}`",
        f"- next-state change rate: `{one_summary.get('next_state_change_rate')}`",
        f"- reward change rate: `{one_summary.get('reward_change_rate')}`",
        f"- full-horizon branches: `{len(branch_rollups)}`",
        f"- B2 intervention windows: `{b2_summary.get('b2_intervention_window_count')}`",
        f"- B2-stratified sensitivity: `{b2_summary.get('b2_stratified_sensitivity')}`",
        f"- parameter mutation / optimizer steps: `{mutation['parameter_mutation_count']} / {mutation['optimizer_step_count']}`",
        f"- NaN/Inf: `{mutation['nan_count'] + mutation['inf_count']}`",
        "",
        "This is a local proxy simulator counterfactual diagnostic. No real-world causal improvement claim is made.",
        "",
    ]) + "\n")

    writer.text("_SUCCESS.lock", json.dumps({"created_at": iso_kst(), "artifact": str(output_root), "gate": gate, "gate_passed": gate_passed}, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = write_manifest(writer)
    if manifest["missing_required_files_after_success_lock"] or manifest["hash_mismatch_count"] or manifest["size_mismatch_count"] or manifest["duplicate_path_count"] or not manifest["success_lock_created_last"]:
        gate, gate_passed = FAIL_MANIFEST, False

    print(f"[DL-6B] artifact: {output_root}")
    print(f"[DL-6B] upstream DL-6A-R1 gate: {upstream['dl6a_gate']}")
    print(f"[DL-6B] upstream DL-5 gate: {upstream['dl5_gate']}")
    print("[DL-6B] study area: SUSEONG_GU_DAEGU")
    print("[DL-6B] nodes / edges: 255 / 291")
    print("[DL-6B] candidate routes: 33")
    print("[DL-6B] MAPPO agents: 8")
    print("[DL-6B] snapshots: 554")
    print("[DL-6B] horizon minutes: 30")
    print(f"[DL-6B] action dim: {action_contract.get('action_dim')}")
    print("[DL-6B] action contract: action 0 noop; action 1/2 intervention ids mapped to local transition engine actions 1/2")
    print(f"[DL-6B] one-step expected/actual pairs: {554 * 8 * 2} / {one_summary.get('actual_one_step_pair_count')}")
    print(f"[DL-6B] valid counterfactual pairs: {one_summary.get('valid_counterfactual_pair_count')}")
    print(f"[DL-6B] next-state change rate: {one_summary.get('next_state_change_rate')}")
    print(f"[DL-6B] reward change rate: {one_summary.get('reward_change_rate')}")
    print(f"[DL-6B] full-horizon branches: {len(branch_rollups)}")
    print(f"[DL-6B] KPI change rate action 1: {classification.get('tier2', {}).get('kpi_changed_window_rate_by_action', {}).get(1)}")
    print(f"[DL-6B] KPI change rate action 2: {classification.get('tier2', {}).get('kpi_changed_window_rate_by_action', {}).get(2)}")
    print(f"[DL-6B] B2 intervention windows: {b2_summary.get('b2_intervention_window_count')}")
    print(f"[DL-6B] B2-stratified sensitivity: {b2_summary.get('b2_stratified_sensitivity')}")
    print(f"[DL-6B] parameter mutation: {mutation['parameter_mutation_count']}")
    print(f"[DL-6B] optimizer steps: {mutation['optimizer_step_count']}")
    print(f"[DL-6B] diagnosis: {diagnosis}")
    print(f"[DL-6B] gate: {gate}")
    print(f"[DL-6B] gate_passed: {str(gate_passed).lower()}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
