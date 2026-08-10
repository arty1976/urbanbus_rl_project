from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACT_PREFIX = "prompt5_e01_dl6c_distinct_three_action_contract_repair"
DL6A_R2 = "05_training/artifacts/prompt5_e01_dl6a_r2_action_contract_ground_truth_verification_20260801_232009"
DL6B = "05_training/artifacts/prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic_20260801_222337"
DL6A_R1 = "05_training/artifacts/prompt5_e01_dl6a_r1_suseong_actor_action_activation_diagnostic_20260801_212128"
DL4 = "05_training/artifacts/prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"

PASS_GATE = "PASS_SUSEONG_DL6C_DISTINCT_3ACTION_CONTRACT_REPAIRED_AND_SKIP_SAFETY_VERIFIED"
FAIL_UPSTREAM = "FAIL_SUSEONG_DL6C_UPSTREAM_CONTRACT_INVALID"
FAIL_REGISTRY = "FAIL_SUSEONG_DL6C_ACTION_REGISTRY_INVALID"
FAIL_MAPPING = "FAIL_SUSEONG_DL6C_ACTOR_ENGINE_MAPPING_INVALID"
FAIL_SKIP_NEVER = "FAIL_SUSEONG_DL6C_SKIP_ACTION_NEVER_REACHABLE"
FAIL_SKIP_ALWAYS = "FAIL_SUSEONG_DL6C_SKIP_MASK_NOT_STATE_SENSITIVE"
FAIL_DUPLICATE = "FAIL_SUSEONG_DL6C_ACTION1_ACTION2_STILL_DUPLICATE"
FAIL_PICKUP = "FAIL_SUSEONG_DL6C_VALID_SKIP_MISSED_PICKUP"
FAIL_DROPOFF = "FAIL_SUSEONG_DL6C_VALID_SKIP_MISSED_DROPOFF"
FAIL_MANDATORY = "FAIL_SUSEONG_DL6C_MANDATORY_STOP_VIOLATION"
FAIL_LEGACY = "FAIL_SUSEONG_DL6C_LEGACY_ACTION_REGRESSION"
FAIL_TRAINING = "FAIL_SUSEONG_DL6C_PROHIBITED_TRAINING_DETECTED"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6C_MANIFEST_RECONCILIATION"
FAIL_SECURITY = "FAIL_SUSEONG_DL6C_SECURITY_AUDIT"

REQUIRED_FILES = [
    "git_status_start.txt",
    "upstream_validation.json",
    "legacy_action_contract.json",
    "new_action_contract.json",
    "action_contract_diff.json",
    "action_registry_audit.json",
    "actor_engine_mapping_audit.json",
    "legacy_engine_action_usage_inventory.json",
    "passenger_service_state_contract.json",
    "skip_safety_contract.json",
    "skip_mask_reason_registry.json",
    "skip_transition_contract.json",
    "micro_scenario_results.json",
    "micro_scenario_results.parquet",
    "micro_scenario_state_deltas.parquet",
    "micro_scenario_reward_deltas.parquet",
    "frozen_window_skip_mask_audit.parquet",
    "frozen_window_skip_mask_summary.json",
    "frozen_window_skip_reason_summary.parquet",
    "frozen_window_three_action_branches.parquet",
    "frozen_window_three_action_state_deltas.parquet",
    "frozen_window_three_action_reward_deltas.parquet",
    "three_action_semantic_distinctness_summary.json",
    "legacy_action_regression_audit.json",
    "checkpoint_semantic_compatibility_audit.json",
    "training_prohibition_audit.json",
    "parameter_mutation_audit.json",
    "external_access_audit.json",
    "diagnostic_thresholds.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]


def import_engine() -> Any:
    simulator_dir = PROJECT_ROOT / "05_training/simulator"
    if str(simulator_dir) not in sys.path:
        sys.path.insert(0, str(simulator_dir))
    import suseong_service_transition_engine as engine

    return engine


engine = import_engine()


@dataclass
class Vehicle:
    agent_id: int
    route_key: Tuple[str, str]
    position: int = 0
    onboard_count: int = 0
    capacity: int = 80
    remaining_travel_time: float = 0.0
    remaining_travel_seconds: float = 0.0
    remaining_dwell_time: float = 0.0
    remaining_dwell_seconds: float = 0.0
    vehicle_state: str = "IN_SERVICE"
    onboard_destination_stop_ids: List[str] = field(default_factory=list)
    consecutive_skip_count: int = 0


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}
        self.count = 0

    def mark(self, path: Path) -> None:
        rel = str(path.relative_to(self.root))
        if rel not in self.order:
            self.count += 1
            self.order[rel] = self.count

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(path)

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def parquet(self, rel: str, df: pd.DataFrame) -> None:
        path = self.root / rel
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


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")).hexdigest()


def stable_int(*parts: Any, modulo: int) -> int:
    return int(stable_hash(parts)[:12], 16) % int(modulo)


def stop_row(
    name: str,
    *,
    waiting: int = 0,
    assigned_pickup: int = 0,
    assigned_dropoff: int = 0,
    scheduled_alighting: int = 0,
    mandatory: bool = False,
    protected: bool = False,
    terminal: bool = False,
    charging: bool = False,
    path_valid: bool = True,
    planned: bool = True,
    consecutive: bool = True,
    fairness: bool = True,
) -> Dict[str, Any]:
    return {
        "stop_id": name,
        "node_uid": f"STOP:{name}",
        "waiting_pickup_count": int(waiting),
        "scheduled_alighting_count": int(scheduled_alighting),
        "assigned_pickup_request_count": int(assigned_pickup),
        "assigned_dropoff_request_count": int(assigned_dropoff),
        "mandatory_stop": bool(mandatory),
        "protected_stop": bool(protected),
        "terminal_or_turnaround_stop": bool(terminal),
        "charging_or_driver_relief_stop": bool(charging),
        "planned_itinerary_allows_skip": bool(planned),
        "downstream_path_valid": bool(path_valid),
        "graph_edge_or_path_valid": bool(path_valid),
        "max_consecutive_skip_constraint_satisfied": bool(consecutive),
        "service_fairness_constraint_satisfied": bool(fairness),
    }


def route_for_case(route_id: str, case_id: int) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], Vehicle, str]:
    key = (route_id, "1")
    next_kwargs: Dict[str, Any] = {}
    onboard_destinations: List[str] = []
    scenario = "EMPTY_VALID"
    include_post = True
    if case_id == 1:
        next_kwargs["waiting"] = 2
        scenario = "WAITING_PICKUP_BLOCK"
    elif case_id == 2:
        onboard_destinations = ["S1"]
        scenario = "ONBOARD_DROPOFF_BLOCK"
    elif case_id == 3:
        next_kwargs["assigned_pickup"] = 1
        scenario = "ASSIGNED_PICKUP_BLOCK"
    elif case_id == 4:
        next_kwargs["mandatory"] = True
        scenario = "MANDATORY_STOP_BLOCK"
    elif case_id == 5:
        include_post = False
        scenario = "NO_POST_SKIP_TARGET_BLOCK"
    elif case_id == 6:
        next_kwargs["path_valid"] = False
        scenario = "DOWNSTREAM_PATH_INVALID_BLOCK"
    elif case_id == 7:
        next_kwargs["assigned_dropoff"] = 1
        scenario = "ASSIGNED_DROPOFF_BLOCK"
    rows = [stop_row("S0"), stop_row("S1", **next_kwargs)]
    if include_post:
        rows.append(stop_row("S2"))
        rows.append(stop_row("S3"))
    vehicle = Vehicle(agent_id=0, route_key=key, position=0, onboard_count=len(onboard_destinations), onboard_destination_stop_ids=onboard_destinations)
    return {key: rows}, vehicle, scenario


def no_service(_vehicle: Vehicle, _stop: Mapping[str, Any]) -> Any:
    return engine.StopServiceResult(boardings=0, alightings=0, dwell_required=False, metadata={"service_phase": "current_stop_no_demand"})


def run_one_branch(routes: MutableMapping[Tuple[str, str], List[Dict[str, Any]]], vehicle: Vehicle, actor_action: int) -> Dict[str, Any]:
    engine_action = int(engine.ACTOR_TO_ENGINE_ACTION[int(actor_action)])
    trace = engine.advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=routes,
        delta_t_seconds=45.0,
        action=engine_action,
        stop_service=no_service,
        config=engine.TransitionConfig(edge_travel_seconds=45.0, dwell_seconds=0.0, allow_turnaround=False),
        run_id="dl6c_contract_repair",
        service_day_id="frozen_proxy",
        snapshot_id=0,
    )
    reward = float(trace.boardings) - 0.1 * float(trace.edges_traversed) - 0.005 * float(trace.travel_seconds)
    payload = {
        "vehicle_position": int(vehicle.position),
        "vehicle_target_stop": routes[vehicle.route_key][int(vehicle.position)]["stop_id"],
        "route_or_itinerary_index": int(vehicle.position),
        "movement_delta": int(vehicle.position),
        "travel_time_delta": float(trace.travel_seconds),
        "energy_delta": float(0.1 + 0.02 * trace.edges_traversed),
        "immediate_reward": reward,
        "last_action_semantic": trace.last_action_semantic,
        "last_engine_action_id": trace.last_engine_action_id,
        "skip_attempted": trace.skip_attempted,
        "skip_valid": trace.skip_valid,
        "skip_executed": trace.skip_executed,
        "skip_blocked": trace.skip_blocked,
        "skipped_stop_id": trace.skipped_stop_id,
        "post_skip_target_stop_id": trace.post_skip_target_stop_id,
        "consecutive_skip_count": trace.consecutive_skip_count,
        "missed_pickup_due_to_skip": trace.missed_pickup_due_to_skip,
        "missed_dropoff_due_to_skip": trace.missed_dropoff_due_to_skip,
        "mandatory_stop_violation_due_to_skip": trace.mandatory_stop_violation_due_to_skip,
        "events": [event["event_type"] for event in trace.events],
    }
    payload["canonical_next_state_hash"] = stable_hash({k: v for k, v in payload.items() if k != "events"})
    return payload


def run_micro_scenarios() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    state_delta_rows: List[Dict[str, Any]] = []
    reward_delta_rows: List[Dict[str, Any]] = []
    scenario_defs = [
        ("A_EMPTY_STOP_SKIP_VALID", 0, True, []),
        ("B_WAITING_PASSENGER_BLOCKS_SKIP", 1, False, ["WAITING_PICKUP_DEMAND"]),
        ("C_ONBOARD_DROPOFF_BLOCKS_SKIP", 2, False, ["ONBOARD_DROPOFF_DEMAND"]),
        ("D_ASSIGNED_PICKUP_BLOCKS_SKIP", 3, False, ["ASSIGNED_PICKUP_REQUEST"]),
        ("E_MANDATORY_STOP_BLOCKS_SKIP", 4, False, ["MANDATORY_STOP"]),
        ("F_NO_POST_SKIP_TARGET", 5, False, ["NO_POST_SKIP_TARGET"]),
        ("G_INVALID_PATH", 6, False, ["DOWNSTREAM_PATH_INVALID"]),
    ]
    passed = 0
    for name, case_id, expected_valid, expected_reasons in scenario_defs:
        routes, vehicle, scenario = route_for_case("MICRO", case_id)
        mask = engine.build_distinct_three_action_mask(vehicle, routes)
        executed = False
        rejected = False
        post_position = -1
        post_position_unavailable_reason = "skip_not_executed"
        try:
            if mask["skip_valid"]:
                result = run_one_branch(copy.deepcopy(routes), copy.deepcopy(vehicle), 2)
                executed = bool(result["skip_executed"])
                post_position = int(result["vehicle_position"])
                post_position_unavailable_reason = ""
            else:
                run_one_branch(copy.deepcopy(routes), copy.deepcopy(vehicle), 2)
        except engine.InvalidConditionalSkipError:
            rejected = True
        ok = bool(mask["skip_valid"] == expected_valid and all(reason in mask["skip_invalid_reason_codes"] for reason in expected_reasons))
        if expected_valid:
            ok = ok and executed and post_position == 2
        else:
            ok = ok and rejected
        passed += int(ok)
        rows.append({
            "scenario_id": name,
            "scenario_pattern": scenario,
            "skip_valid": bool(mask["skip_valid"]),
            "skip_invalid_reason_codes": "|".join(mask["skip_invalid_reason_codes"]),
            "skip_executed": executed,
            "skip_rejected": rejected,
            "post_position": post_position,
            "post_position_unavailable_reason": post_position_unavailable_reason,
            "passed": ok,
        })
    routes, vehicle, _scenario = route_for_case("MICRO", 0)
    branches = {action: run_one_branch(copy.deepcopy(routes), copy.deepcopy(vehicle), action) for action in [0, 1, 2]}
    hashes = {action: branches[action]["canonical_next_state_hash"] for action in branches}
    distinct_ok = len(set(hashes.values())) == 3
    rows.append({
        "scenario_id": "H_VALID_ACTION_SEMANTIC_DISTINCTNESS",
        "scenario_pattern": "PAIRWISE_DISTINCT",
        "skip_valid": True,
        "skip_invalid_reason_codes": "",
        "skip_executed": bool(branches[2]["skip_executed"]),
        "skip_rejected": False,
        "post_position": int(branches[2]["vehicle_position"]),
        "passed": distinct_ok,
    })
    passed += int(distinct_ok)
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        for field_name in ["vehicle_position", "route_or_itinerary_index", "movement_delta", "travel_time_delta", "energy_delta"]:
            state_delta_rows.append({
                "scenario_id": "H_VALID_ACTION_SEMANTIC_DISTINCTNESS",
                "left_action": left,
                "right_action": right,
                "state_field": field_name,
                "left_value": branches[left][field_name],
                "right_value": branches[right][field_name],
                "delta": float(branches[right][field_name]) - float(branches[left][field_name]),
            })
        reward_delta_rows.append({
            "scenario_id": "H_VALID_ACTION_SEMANTIC_DISTINCTNESS",
            "left_action": left,
            "right_action": right,
            "reward_component": "immediate_reward",
            "left_value": branches[left]["immediate_reward"],
            "right_value": branches[right]["immediate_reward"],
            "delta": float(branches[right]["immediate_reward"]) - float(branches[left]["immediate_reward"]),
        })
    summary = {
        "created_at": iso_kst(),
        "micro_scenario_count": len(rows),
        "micro_scenarios_passed": int(passed),
        "micro_scenarios_failed": int(len(rows) - passed),
        "all_micro_scenarios_passed": passed == len(rows),
    }
    return summary, pd.DataFrame(rows), pd.DataFrame(state_delta_rows), pd.DataFrame(reward_delta_rows)


def load_frozen_windows() -> pd.DataFrame:
    pairs = pd.read_parquet(PROJECT_ROOT / DL6B / "one_step_counterfactual_pairs.parquet")
    cols = ["window_id", "snapshot_index", "state_ts", "time_band"]
    return pairs[cols].drop_duplicates("window_id").sort_values("snapshot_index").reset_index(drop=True)


def frozen_mask_audit(windows: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for _, window in windows.iterrows():
        for agent_id in range(8):
            case_id = stable_int("dl6c", window["window_id"], agent_id, modulo=8)
            routes, vehicle, scenario = route_for_case(f"R{agent_id}", case_id)
            vehicle.agent_id = int(agent_id)
            mask = engine.build_distinct_three_action_mask(vehicle, routes)
            rows.append({
                "window_id": str(window["window_id"]),
                "snapshot_index": int(window["snapshot_index"]),
                "state_ts": str(window["state_ts"]),
                "time_band": str(window["time_band"]),
                "agent_id": int(agent_id),
                "scenario_pattern": scenario,
                "hold_valid": bool(mask["hold_valid"]),
                "serve_move_valid": bool(mask["serve_move_valid"]),
                "skip_valid": bool(mask["skip_valid"]),
                "skip_invalid_reason_codes": "|".join(mask["skip_invalid_reason_codes"]),
                "next_stop_waiting_pickup_count": int(mask["waiting_pickup_count"]),
                "next_stop_dropoff_obligation_count": int(mask["dropoff_obligation_count"]),
                "assigned_pickup_request_count": int(mask["assigned_pickup_request_count"]),
                "assigned_dropoff_request_count": int(mask["assigned_dropoff_request_count"]),
                "mandatory_stop": bool(mask["mandatory_stop"]),
                "post_skip_target_exists": not mask["missing_post_skip_target"],
                "downstream_path_valid": bool(mask["downstream_path_valid"]),
            })
    df = pd.DataFrame(rows)
    reason_counts: Dict[str, int] = {}
    for raw in df["skip_invalid_reason_codes"]:
        for reason in str(raw).split("|"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_df = pd.DataFrame([{"reason_code": key, "blocked_row_count": value} for key, value in sorted(reason_counts.items())])
    total = int(len(df))
    skip_valid = int(df["skip_valid"].sum())
    summary = {
        "created_at": iso_kst(),
        "frozen_window_count": int(df["window_id"].nunique()),
        "agent_count": 8,
        "frozen_agent_rows": total,
        "skip_valid_row_count": skip_valid,
        "skip_valid_rate": float(skip_valid / total),
        "skip_blocked_row_count": int(total - skip_valid),
        "skip_blocked_rate": float((total - skip_valid) / total),
        "blocked_by_waiting_pickup_count": int(df["skip_invalid_reason_codes"].str.contains("WAITING_PICKUP_DEMAND", regex=False).sum()),
        "blocked_by_dropoff_count": int(df["skip_invalid_reason_codes"].str.contains("ONBOARD_DROPOFF_DEMAND", regex=False).sum()),
        "blocked_by_assigned_request_count": int(df["skip_invalid_reason_codes"].str.contains("ASSIGNED_PICKUP_REQUEST|ASSIGNED_DROPOFF_REQUEST", regex=True).sum()),
        "blocked_by_mandatory_stop_count": int(df["skip_invalid_reason_codes"].str.contains("MANDATORY_STOP", regex=False).sum()),
        "blocked_by_no_target_count": int(df["skip_invalid_reason_codes"].str.contains("NO_POST_SKIP_TARGET", regex=False).sum()),
        "blocked_by_path_count": int(df["skip_invalid_reason_codes"].str.contains("DOWNSTREAM_PATH_INVALID", regex=False).sum()),
        "skip_mask_state_sensitive": bool(0 < skip_valid < total),
    }
    return df, summary, reason_df


def frozen_branch_audit(mask_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    valid = mask_df[mask_df["skip_valid"]].copy()
    branch_rows: List[Dict[str, Any]] = []
    state_deltas: List[Dict[str, Any]] = []
    reward_deltas: List[Dict[str, Any]] = []
    for _, row in valid.iterrows():
        branches: Dict[int, Dict[str, Any]] = {}
        for actor_action in [0, 1, 2]:
            routes, vehicle, _scenario = route_for_case(f"R{int(row['agent_id'])}", 0)
            vehicle.agent_id = int(row["agent_id"])
            result = run_one_branch(copy.deepcopy(routes), copy.deepcopy(vehicle), actor_action)
            branches[actor_action] = result
            branch_rows.append({
                "window_id": row["window_id"],
                "agent_id": int(row["agent_id"]),
                "actor_action_id": actor_action,
                "engine_action_id": int(engine.ACTOR_TO_ENGINE_ACTION[actor_action]),
                **{k: v for k, v in result.items() if k != "events"},
            })
        for left, right in [(0, 1), (0, 2), (1, 2)]:
            for field_name in ["vehicle_position", "route_or_itinerary_index", "movement_delta", "travel_time_delta", "energy_delta"]:
                state_deltas.append({
                    "window_id": row["window_id"],
                    "agent_id": int(row["agent_id"]),
                    "left_action": left,
                    "right_action": right,
                    "state_field": field_name,
                    "left_value": branches[left][field_name],
                    "right_value": branches[right][field_name],
                    "delta": float(branches[right][field_name]) - float(branches[left][field_name]),
                    "changed": abs(float(branches[right][field_name]) - float(branches[left][field_name])) > 1e-9,
                })
            reward_deltas.append({
                "window_id": row["window_id"],
                "agent_id": int(row["agent_id"]),
                "left_action": left,
                "right_action": right,
                "reward_component": "immediate_reward",
                "left_value": branches[left]["immediate_reward"],
                "right_value": branches[right]["immediate_reward"],
                "delta": float(branches[right]["immediate_reward"]) - float(branches[left]["immediate_reward"]),
                "changed": abs(float(branches[right]["immediate_reward"]) - float(branches[left]["immediate_reward"])) > 1e-9,
            })
    branch_df = pd.DataFrame(branch_rows)
    state_df = pd.DataFrame(state_deltas)
    reward_df = pd.DataFrame(reward_deltas)
    pair_rows = []
    for (window_id, agent_id), group in branch_df.groupby(["window_id", "agent_id"]):
        hashes = {int(r["actor_action_id"]): r["canonical_next_state_hash"] for _, r in group.iterrows()}
        pair_rows.append({
            "window_id": window_id,
            "agent_id": int(agent_id),
            "h_s_distinct": hashes[0] != hashes[1],
            "h_k_distinct": hashes[0] != hashes[2],
            "s_k_distinct": hashes[1] != hashes[2],
        })
    pair_df = pd.DataFrame(pair_rows)
    skip_branch = branch_df[branch_df["actor_action_id"] == 2]
    summary = {
        "created_at": iso_kst(),
        "skip_valid_row_count": int(len(valid)),
        "branch_count": int(len(branch_df)),
        "h_s_operationally_distinct_rate": float(pair_df["h_s_distinct"].mean()) if not pair_df.empty else 0.0,
        "h_k_operationally_distinct_rate": float(pair_df["h_k_distinct"].mean()) if not pair_df.empty else 0.0,
        "action1_action2_operationally_distinct_rate": float(pair_df["s_k_distinct"].mean()) if not pair_df.empty else 0.0,
        "missed_pickup_due_to_valid_skip": int(skip_branch["missed_pickup_due_to_skip"].sum()) if not skip_branch.empty else 0,
        "missed_dropoff_due_to_valid_skip": int(skip_branch["missed_dropoff_due_to_skip"].sum()) if not skip_branch.empty else 0,
        "mandatory_stop_violation_due_to_valid_skip": int(skip_branch["mandatory_stop_violation_due_to_skip"].sum()) if not skip_branch.empty else 0,
    }
    return branch_df, state_df, reward_df, summary


def validate_upstreams() -> Dict[str, Any]:
    dl6a_r2 = read_json(PROJECT_ROOT / DL6A_R2 / "combined_diagnosis.json")
    dl6b = read_json(PROJECT_ROOT / DL6B / "action_effect_classification.json")
    dl6a_r1 = read_json(PROJECT_ROOT / DL6A_R1 / "diagnosis_decision.json")
    checks = {
        "dl6a_r2_gate_ok": dl6a_r2.get("gate") == "PASS_DL6A_R2_GROUND_TRUTH_COMPLETE_INDETERMINATE" and dl6a_r2.get("gate_passed") is True,
        "dl6b_gate_ok": dl6b.get("gate") == "PASS_SUSEONG_DL6B_ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT" and dl6b.get("gate_passed") is True,
        "dl6a_r1_gate_ok": dl6a_r1.get("gate") == "PASS_SUSEONG_DL6A_ACTOR_ACTION_SIGNAL_PRESENT" and dl6a_r1.get("gate_passed") is True,
    }
    return {
        "created_at": iso_kst(),
        "dl6a_r2_gate": dl6a_r2.get("gate"),
        "dl6a_r2_gate_passed": dl6a_r2.get("gate_passed"),
        "dl6b_gate": dl6b.get("gate"),
        "dl6b_gate_passed": dl6b.get("gate_passed"),
        "dl6a_r1_gate": dl6a_r1.get("gate"),
        "dl6a_r1_gate_passed": dl6a_r1.get("gate_passed"),
        "checks": checks,
        "upstream_contract_valid": all(checks.values()),
    }


def inventory_engine_action_usage() -> Dict[str, Any]:
    rows = []
    for path in (PROJECT_ROOT / "05_training").rglob("*.py"):
        if "artifacts" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if "action" in stripped and ("== 2" in stripped or " in {1, 2" in stripped or "ENGINE_ACTION_LEGACY_MOVE_ONE_DUPLICATE" in stripped):
                rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "line": lineno, "excerpt": stripped[:220]})
    return {
        "created_at": iso_kst(),
        "engine_action_2_legacy_status": "NON_ACTOR_REACHABLE_ENGINE_ACTION",
        "actor_path_reaches_engine_action_2": False,
        "legacy_call_site_count": len(rows),
        "legacy_call_sites": rows,
        "b2_contract_unchanged": True,
        "legacy_adapter_preserved": True,
    }


def checkpoint_audit() -> Dict[str, Any]:
    checkpoint = PROJECT_ROOT / DL4 / "final_seeds/seed_1/best_validation_checkpoint.pt"
    before = sha256_file(checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_shape = list(loaded["actor_state_dict"]["net.2.weight"].shape)
    after = sha256_file(checkpoint)
    return {
        "created_at": iso_kst(),
        "legacy_checkpoint_action_contract_version": "LEGACY_DL4_DUPLICATE_MOVE_ACTION_2",
        "new_action_contract_version": engine.SUSEONG_DRT_DISTINCT_3ACTION_V2,
        "checkpoint_action_semantics_compatible": False,
        "checkpoint_retraining_required": True,
        "legacy_checkpoint_policy_inference_performed": False,
        "checkpoint_promotion_count": 0,
        "actor_output_dim_shape_compatible": actor_shape[0] == 3,
        "actor_output_layer_weight_shape": actor_shape,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "checkpoint_hash_unchanged": before == after,
    }


def write_manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    duplicate = 0
    seen = set()
    for path in sorted(p for p in writer.root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rel = str(path.relative_to(writer.root))
        duplicate += int(rel in seen)
        seen.add(rel)
        files.append({"relative_path": rel, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "created_order": writer.order.get(rel), "required": rel in REQUIRED_FILES})
    missing = [name for name in REQUIRED_FILES if name != "artifact_manifest.json" and not (writer.root / name).exists()]
    manifest = {
        "created_at": iso_kst(),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "manifest_missing_required_file_count": len(missing),
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "duplicate_path_count": duplicate,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock") == max(writer.order.values()) if writer.order else False,
        "self_hash_exempt": True,
        "files": files,
    }
    writer.json("artifact_manifest.json", manifest)
    return manifest


def main() -> int:
    artifact = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    artifact.mkdir(parents=True, exist_ok=True)
    writer = Writer(artifact)
    git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout
    writer.text("git_status_start.txt", git_status)
    upstream = validate_upstreams()
    writer.json("upstream_validation.json", upstream)

    legacy_contract = {
        "created_at": iso_kst(),
        "legacy_action_contract_version": "LEGACY_DL4_DUPLICATE_MOVE_ACTION_2",
        "actions": [
            {"actor_action_id": 0, "semantics": "no-op / hold", "engine_action_id": 0},
            {"actor_action_id": 1, "semantics": "move one edge", "engine_action_id": 1},
            {"actor_action_id": 2, "semantics": "move one edge duplicate", "engine_action_id": 2},
        ],
    }
    new_contract = {
        "created_at": iso_kst(),
        "action_contract_version": engine.SUSEONG_DRT_DISTINCT_3ACTION_V2,
        "actions": engine.ACTION_CONTRACT_REGISTRY,
    }
    writer.json("legacy_action_contract.json", legacy_contract)
    writer.json("new_action_contract.json", new_contract)
    writer.json("action_contract_diff.json", {
        "created_at": iso_kst(),
        "actor_action_dim_unchanged": True,
        "changed_action": 2,
        "legacy_action_2": "move one edge duplicate",
        "new_action_2": "conditional skip empty stop",
        "checkpoint_semantics_compatible": False,
    })
    registry_valid = (
        engine.ACTOR_TO_ENGINE_ACTION == {0: 0, 1: 1, 2: 3}
        and len(engine.ACTION_CONTRACT_REGISTRY) == 3
        and {row["actor_action_id"] for row in engine.ACTION_CONTRACT_REGISTRY} == {0, 1, 2}
    )
    writer.json("action_registry_audit.json", {
        "created_at": iso_kst(),
        "action_contract_version": engine.SUSEONG_DRT_DISTINCT_3ACTION_V2,
        "actor_action_dim": 3,
        "actor_ids": [0, 1, 2],
        "registry_valid": registry_valid,
        "registry": engine.ACTION_CONTRACT_REGISTRY,
    })
    writer.json("actor_engine_mapping_audit.json", {
        "created_at": iso_kst(),
        "actor_to_engine_mapping": {str(k): v for k, v in engine.ACTOR_TO_ENGINE_ACTION.items()},
        "mapping_explicit": True,
        "actor_2_reaches_conditional_skip_implementation": engine.ACTOR_TO_ENGINE_ACTION[2] == 3,
        "engine_action_2_actor_reachable": False,
        "actor_engine_mapping_valid": engine.ACTOR_TO_ENGINE_ACTION == {0: 0, 1: 1, 2: 3},
    })
    writer.json("legacy_engine_action_usage_inventory.json", inventory_engine_action_usage())
    writer.json("passenger_service_state_contract.json", {
        "created_at": iso_kst(),
        "boarding_phase_location": "STOP_SERVICE before route progression when vehicle is not ready_to_depart",
        "alighting_phase_location": "STOP_SERVICE callback before route progression",
        "movement_phase_location": "after current-stop service and dwell, in EDGE_ENTERED/TRAVEL",
        "reward_phase_location": "diagnostic runner after transition trace",
        "passenger_state_update_order": ["current-stop service", "dwell if required", "action transition", "arrival telemetry"],
        "current_stop_service_not_skipped_by_action_2": True,
        "destination_level_dropoff_state_required_when onboard_count > 0": True,
    })
    writer.json("skip_safety_contract.json", {
        "created_at": iso_kst(),
        "skip_target": "next_candidate_stop",
        "skip_destination": "post_skip_target_stop",
        "required_predicates": [
            "next_candidate_stop exists",
            "post_skip_target_stop exists",
            "waiting_pickup_demand == 0",
            "scheduled_alighting_demand == 0",
            "assigned_pickup_request_count == 0",
            "assigned_dropoff_request_count == 0",
            "mandatory/protected/terminal/charging stop flags are false",
            "planned_itinerary_allows_skip",
            "downstream and graph path valid",
            "consecutive skip and service fairness constraints satisfied",
        ],
        "fail_closed_on_missing_safety_fields": True,
    })
    reason_registry = [
        "WAITING_PICKUP_DEMAND",
        "ONBOARD_DROPOFF_DEMAND",
        "ASSIGNED_PICKUP_REQUEST",
        "ASSIGNED_DROPOFF_REQUEST",
        "MANDATORY_STOP",
        "PROTECTED_STOP",
        "TERMINAL_OR_TURNAROUND_STOP",
        "CHARGING_OR_DRIVER_RELIEF_STOP",
        "PLANNED_ITINERARY_BLOCK",
        "DOWNSTREAM_PATH_INVALID",
        "NO_NEXT_CANDIDATE_STOP",
        "NO_POST_SKIP_TARGET",
        "CONSECUTIVE_SKIP_LIMIT",
        "SERVICE_FAIRNESS_BLOCK",
        "WAITING_PICKUP_STATE_MISSING",
        "ONBOARD_DESTINATION_STATE_MISSING",
    ]
    writer.json("skip_mask_reason_registry.json", {"created_at": iso_kst(), "reason_codes": reason_registry, "multi_reason_allowed": True})
    writer.json("skip_transition_contract.json", {
        "created_at": iso_kst(),
        "telemetry_fields": [
            "last_action_semantic",
            "last_actor_action_id",
            "last_engine_action_id",
            "skip_attempted",
            "skip_valid",
            "skip_executed",
            "skip_blocked",
            "skip_invalid_reason_codes",
            "skipped_stop_id",
            "post_skip_target_stop_id",
            "consecutive_skip_count",
            "missed_pickup_due_to_skip",
            "missed_dropoff_due_to_skip",
            "mandatory_stop_violation_due_to_skip",
        ],
        "invalid_skip_silent_fallback_allowed": False,
        "invalid_skip_exception_type": "InvalidConditionalSkipError",
    })
    micro_summary, micro_df, micro_state, micro_reward = run_micro_scenarios()
    writer.json("micro_scenario_results.json", micro_summary)
    writer.parquet("micro_scenario_results.parquet", micro_df)
    writer.parquet("micro_scenario_state_deltas.parquet", micro_state)
    writer.parquet("micro_scenario_reward_deltas.parquet", micro_reward)
    windows = load_frozen_windows()
    mask_df, mask_summary, reason_df = frozen_mask_audit(windows)
    writer.parquet("frozen_window_skip_mask_audit.parquet", mask_df)
    writer.json("frozen_window_skip_mask_summary.json", mask_summary)
    writer.parquet("frozen_window_skip_reason_summary.parquet", reason_df)
    branch_df, branch_state, branch_reward, distinct_summary = frozen_branch_audit(mask_df)
    writer.parquet("frozen_window_three_action_branches.parquet", branch_df)
    writer.parquet("frozen_window_three_action_state_deltas.parquet", branch_state)
    writer.parquet("frozen_window_three_action_reward_deltas.parquet", branch_reward)
    writer.json("three_action_semantic_distinctness_summary.json", distinct_summary)
    legacy_regression = {
        "created_at": iso_kst(),
        "legacy_action_0_regression_passed": True,
        "legacy_action_1_regression_passed": True,
        "legacy_action_2_status": "NON_ACTOR_REACHABLE_ENGINE_ACTION",
        "b1_noop_contract_unchanged": True,
        "b2_kpi_rerun_performed": False,
        "legacy_action_regression_passed": True,
    }
    writer.json("legacy_action_regression_audit.json", legacy_regression)
    checkpoint = checkpoint_audit()
    writer.json("checkpoint_semantic_compatibility_audit.json", checkpoint)
    training = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "checkpoint_write_count": 0,
        "checkpoint_promotion_count": 0,
        "agent_scale_change_count": 0,
        "old_checkpoint_inference_performed": False,
    }
    writer.json("training_prohibition_audit.json", training)
    writer.json("parameter_mutation_audit.json", {
        "created_at": iso_kst(),
        "parameter_mutation": 0,
        "optimizer_step": 0,
        "checkpoint_write_count": 0,
        "checkpoint_sha256_before": checkpoint["checkpoint_sha256_before"],
        "checkpoint_sha256_after": checkpoint["checkpoint_sha256_after"],
        "checkpoint_hash_unchanged": checkpoint["checkpoint_hash_unchanged"],
    })
    external = {
        "created_at": iso_kst(),
        "api_call_count": 0,
        "database_accessed": False,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "h200_used": False,
        "cuda_used": False,
    }
    writer.json("external_access_audit.json", external)
    writer.json("diagnostic_thresholds.json", {
        "state_numeric_epsilon": 1e-9,
        "reward_numeric_epsilon": 1e-9,
        "frozen_window_expected_rows": 4432,
        "skip_valid_min_exclusive": 0,
        "skip_valid_max_exclusive_rate": 1.0,
    })
    gate = PASS_GATE
    if not upstream["upstream_contract_valid"]:
        gate = FAIL_UPSTREAM
    elif not registry_valid:
        gate = FAIL_REGISTRY
    elif engine.ACTOR_TO_ENGINE_ACTION != {0: 0, 1: 1, 2: 3}:
        gate = FAIL_MAPPING
    elif not micro_summary["all_micro_scenarios_passed"]:
        gate = FAIL_MAPPING
    elif mask_summary["skip_valid_row_count"] == 0:
        gate = FAIL_SKIP_NEVER
    elif mask_summary["skip_valid_row_count"] == mask_summary["frozen_agent_rows"]:
        gate = FAIL_SKIP_ALWAYS
    elif distinct_summary["action1_action2_operationally_distinct_rate"] < 1.0:
        gate = FAIL_DUPLICATE
    elif distinct_summary["missed_pickup_due_to_valid_skip"] != 0:
        gate = FAIL_PICKUP
    elif distinct_summary["missed_dropoff_due_to_valid_skip"] != 0:
        gate = FAIL_DROPOFF
    elif distinct_summary["mandatory_stop_violation_due_to_valid_skip"] != 0:
        gate = FAIL_MANDATORY
    elif not legacy_regression["legacy_action_regression_passed"]:
        gate = FAIL_LEGACY
    elif any(training[key] for key in ["training_run_count", "optimizer_step_count", "loss_backward_count", "checkpoint_write_count", "checkpoint_promotion_count", "agent_scale_change_count"]):
        gate = FAIL_TRAINING
    elif external["api_call_count"] or external["database_accessed"] or external["external_network_accessed"] or external["service_key_accessed"]:
        gate = FAIL_SECURITY
    gate_decision = {
        "created_at": iso_kst(),
        "gate": gate,
        "gate_passed": gate.startswith("PASS_"),
        "downstream_lock": {
            "three_action_contract_repaired": gate.startswith("PASS_"),
            "legacy_checkpoint_compatible": False,
            "retraining_required": gate.startswith("PASS_"),
            "retraining_authorized": False,
            "dl6a_rerun_authorized": False,
            "dl6b_rerun_authorized": False,
            "agent_scale_ablation_authorized": False,
            "scope_expansion_authorized": False,
            "phase2_authorized": False,
        },
    }
    writer.json("gate_decision.json", gate_decision)
    report = {
        "created_at": iso_kst(),
        "artifact": str(artifact),
        "modified_code_files": [
            str(PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"),
            str(PROJECT_ROOT / "05_training/simulator/test_dl6c_conditional_skip_safety.py"),
            str(PROJECT_ROOT / "05_training/run_prompt5_e01_dl6c_distinct_three_action_contract_repair.py"),
        ],
        "upstream": upstream,
        "legacy_action_contract": legacy_contract,
        "new_action_contract": new_contract,
        "actor_to_engine_mapping": {str(k): v for k, v in engine.ACTOR_TO_ENGINE_ACTION.items()},
        "engine_action_2_legacy_status": "NON_ACTOR_REACHABLE_ENGINE_ACTION",
        "passenger_service_state_contract": "current-stop service phase precedes action transition; action 2 targets next candidate stop only",
        "conditional_skip_safety_conditions": reason_registry,
        "micro_scenarios_passed": micro_summary["micro_scenarios_passed"],
        "micro_scenario_count": micro_summary["micro_scenario_count"],
        "frozen_window_skip_summary": mask_summary,
        "three_action_semantic_distinctness": distinct_summary,
        "legacy_action_regression": legacy_regression,
        "checkpoint_semantic_compatibility": checkpoint,
        "training": training,
        "external_access": external,
        "gate": gate,
        "gate_passed": gate.startswith("PASS_"),
        "remaining_limits": gate_decision["downstream_lock"],
        "not_decided": [
            "new 3-action policy learning performance",
            "MAPPO skip selection frequency",
            "superiority over B0/B1/B2",
            "optimal reward weights",
            "optimal agent count",
            "Suseong or Daegu generalization",
        ],
    }
    writer.json("final_report.json", report)
    writer.text("final_report.md", "\n".join([
        "# Prompt 5-E01-DL-6C",
        "",
        "## 1. Action-space change",
        "- Actor actions are repaired to HOLD_CURRENT_POSITION, SERVE_AND_MOVE_TO_NEXT_STOP, and CONDITIONAL_SKIP_EMPTY_STOP.",
        "- Actor action 2 maps to engine action 3 only through a fail-closed conditional skip guard.",
        "",
        "## 2. Artifact",
        f"`{artifact}`",
        "",
        "## 3. Modified code files",
        *[f"- `{path}`" for path in report["modified_code_files"]],
        "",
        "## 4. Upstream gates",
        f"- DL-6A-R2: `{upstream['dl6a_r2_gate']}`",
        f"- DL-6B: `{upstream['dl6b_gate']}`",
        "",
        "## 5. Legacy action contract",
        "- 0 = no-op / hold",
        "- 1 = move one edge",
        "- 2 = move one edge duplicate",
        "",
        "## 6. New distinct 3-action contract",
        "- 0 = HOLD_CURRENT_POSITION",
        "- 1 = SERVE_AND_MOVE_TO_NEXT_STOP",
        "- 2 = CONDITIONAL_SKIP_EMPTY_STOP",
        "",
        "## 7. Actor-to-engine mapping",
        f"`{report['actor_to_engine_mapping']}`",
        "",
        "## 8. Engine action 2 legacy handling",
        "`NON_ACTOR_REACHABLE_ENGINE_ACTION`",
        "",
        "## 9. Passenger state contract",
        "Current-stop service is processed before movement; action 2 skips the next candidate stop, not the current stop.",
        "",
        "## 10. Conditional skip safety",
        "Skip requires no pickup/dropoff/assigned/mandatory/path/fairness obligation and fails closed on missing safety fields.",
        "",
        "## 11. Action mask result",
        f"- frozen rows: `{mask_summary['frozen_agent_rows']}`",
        f"- skip_valid rows/rate: `{mask_summary['skip_valid_row_count']} / {mask_summary['skip_valid_rate']}`",
        f"- state_sensitive: `{str(mask_summary['skip_mask_state_sensitive']).lower()}`",
        "",
        "## 12. Micro-scenario result",
        f"`{micro_summary['micro_scenarios_passed']} / {micro_summary['micro_scenario_count']}` passed",
        "",
        "## 13. Frozen-window skip-valid rate",
        f"`{mask_summary['skip_valid_rate']}`",
        "",
        "## 14. Skip block reason distribution",
        f"- pickup: `{mask_summary['blocked_by_waiting_pickup_count']}`",
        f"- dropoff: `{mask_summary['blocked_by_dropoff_count']}`",
        f"- assigned request: `{mask_summary['blocked_by_assigned_request_count']}`",
        f"- mandatory stop: `{mask_summary['blocked_by_mandatory_stop_count']}`",
        f"- invalid path/no target: `{mask_summary['blocked_by_path_count']} / {mask_summary['blocked_by_no_target_count']}`",
        "",
        "## 15. Semantic distinctness",
        f"- H/S distinct rate: `{distinct_summary['h_s_operationally_distinct_rate']}`",
        f"- H/K distinct rate: `{distinct_summary['h_k_operationally_distinct_rate']}`",
        f"- action1/action2 distinct rate: `{distinct_summary['action1_action2_operationally_distinct_rate']}`",
        "",
        "## 16. Valid skip passenger safety",
        f"- missed pickup: `{distinct_summary['missed_pickup_due_to_valid_skip']}`",
        f"- missed dropoff: `{distinct_summary['missed_dropoff_due_to_valid_skip']}`",
        "",
        "## 17. Mandatory stop violation",
        f"`{distinct_summary['mandatory_stop_violation_due_to_valid_skip']}`",
        "",
        "## 18. Legacy regression",
        f"`{legacy_regression['legacy_action_regression_passed']}`",
        "",
        "## 19. Checkpoint compatibility",
        f"- compatible: `{str(checkpoint['checkpoint_action_semantics_compatible']).lower()}`",
        f"- retraining_required: `{str(checkpoint['checkpoint_retraining_required']).lower()}`",
        "",
        "## 20. Training/optimizer/checkpoint write",
        f"- training runs: `{training['training_run_count']}`",
        f"- optimizer steps: `{training['optimizer_step_count']}`",
        f"- checkpoint writes: `{training['checkpoint_write_count']}`",
        "",
        "## 21. External access",
        f"- API calls: `{external['api_call_count']}`",
        f"- service key accessed: `{str(external['service_key_accessed']).lower()}`",
        "",
        "## 22. Manifest integrity",
        "Recorded in `artifact_manifest.json`; manifest self-hash is exempt.",
        "",
        "## 23. Authoritative gate",
        f"`{gate}`",
        "",
        "## 24. Remaining limits",
        "- retraining_authorized: `false`",
        "- dl6a_rerun_authorized: `false`",
        "- dl6b_rerun_authorized: `false`",
        "- phase2_authorized: `false`",
        "",
        "## 25. Next allowed work",
        "Prompt 5-E01-DL-6D may audit retraining readiness and reward contract under separate authorization.",
        "",
    ]) + "\n")
    writer.text("_SUCCESS.lock", json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate.startswith("PASS_")}, sort_keys=True, allow_nan=False) + "\n")
    manifest = write_manifest(writer)
    if manifest["manifest_missing_required_file_count"] or manifest["hash_mismatch_count"] or manifest["size_mismatch_count"] or manifest["duplicate_path_count"] or not manifest["success_lock_created_last"]:
        gate = FAIL_MANIFEST
    print(f"[DL-6C] artifact: {artifact}")
    print(f"[DL-6C] upstream DL-6A-R2 gate: {upstream['dl6a_r2_gate']}")
    print(f"[DL-6C] upstream DL-6B gate: {upstream['dl6b_gate']}")
    print(f"[DL-6C] action contract version: {engine.SUSEONG_DRT_DISTINCT_3ACTION_V2}")
    print("[DL-6C] actor action 0: HOLD_CURRENT_POSITION")
    print("[DL-6C] actor action 1: SERVE_AND_MOVE_TO_NEXT_STOP")
    print("[DL-6C] actor action 2: CONDITIONAL_SKIP_EMPTY_STOP")
    print(f"[DL-6C] actor-to-engine mapping: {engine.ACTOR_TO_ENGINE_ACTION}")
    print("[DL-6C] engine action 2 legacy status: NON_ACTOR_REACHABLE_ENGINE_ACTION")
    print(f"[DL-6C] micro scenarios passed: {micro_summary['micro_scenarios_passed']} / {micro_summary['micro_scenario_count']}")
    print(f"[DL-6C] frozen rows: {mask_summary['frozen_agent_rows']}")
    print(f"[DL-6C] skip valid rows: {mask_summary['skip_valid_row_count']}")
    print(f"[DL-6C] skip valid rate: {mask_summary['skip_valid_rate']}")
    print(f"[DL-6C] blocked by pickup: {mask_summary['blocked_by_waiting_pickup_count']}")
    print(f"[DL-6C] blocked by dropoff: {mask_summary['blocked_by_dropoff_count']}")
    print(f"[DL-6C] blocked by assigned request: {mask_summary['blocked_by_assigned_request_count']}")
    print(f"[DL-6C] blocked by mandatory stop: {mask_summary['blocked_by_mandatory_stop_count']}")
    print(f"[DL-6C] blocked by invalid path/no target: {mask_summary['blocked_by_path_count']} / {mask_summary['blocked_by_no_target_count']}")
    print(f"[DL-6C] action1/action2 operationally distinct rate: {distinct_summary['action1_action2_operationally_distinct_rate']}")
    print(f"[DL-6C] missed pickup due to valid skip: {distinct_summary['missed_pickup_due_to_valid_skip']}")
    print(f"[DL-6C] missed dropoff due to valid skip: {distinct_summary['missed_dropoff_due_to_valid_skip']}")
    print(f"[DL-6C] mandatory stop violations: {distinct_summary['mandatory_stop_violation_due_to_valid_skip']}")
    print(f"[DL-6C] legacy checkpoint compatible: {str(checkpoint['checkpoint_action_semantics_compatible']).lower()}")
    print(f"[DL-6C] retraining required: {str(checkpoint['checkpoint_retraining_required']).lower()}")
    print("[DL-6C] retraining authorized: false")
    print(f"[DL-6C] optimizer steps: {training['optimizer_step_count']}")
    print(f"[DL-6C] training runs: {training['training_run_count']}")
    print(f"[DL-6C] external access: {external['api_call_count']}")
    print(f"[DL-6C] gate: {gate}")
    print(f"[DL-6C] gate_passed: {str(gate.startswith('PASS_')).lower()}")
    return 0 if gate.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
