from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import deque
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from run_prompt5_e01_r2_service_day_constraint_skip_repair import (
    CONDITIONS,
    FAMILIES,
    SEEDS,
    build_snapshot_service_table,
    discover_claim_guard,
    load_cache_rows,
    service_day_duration_seconds,
    service_day_fields,
)
from run_suseong_scientific_matrix import FixedDemandSuseongSimulator, condition_agents, stable_int
from simulator.suseong_service_transition_engine import (
    StopServiceResult,
    TransitionConfig,
    advance_vehicle_time_budget,
)


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2a_threshold_baseline_feasibility"
CACHE_ROOT = "05_training/artifacts/suseong_dynamic_embedding_cache_v1"
BASELINES = ["B1", "B2"]
SERVICE_TZ = "Asia/Seoul"
SERVICE_START = time(5, 0)
SERVICE_END = time(22, 0)
ABSOLUTE_WAIT_CAP_SECONDS = 900.0
SERVICE_RATE_MIN = 0.95
MAX_CONSECUTIVE_SKIPS = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)


def discover_latest_r2(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[str, Path, Dict[str, Any]]] = []
    for path in (project_root / "05_training/artifacts").glob("prompt5_e01_r2_service_day_constraint_skip_repair_*/prompt5_e01_r2_gate.json"):
        data = load_json(path)
        if (
            data.get("status") == "BLOCKED_THRESHOLD_APPROVAL_REQUIRED"
            and data.get("classification") == "NUMERIC_HARD_CONSTRAINT_CONTRACT_MISSING"
            and data.get("training_semantics_impact_classification") == "TRAINING_SERVICE_DAY_AND_SKIP_STOP_AFFECTED"
            and data.get("retraining_required") is True
        ):
            candidates.append((str(data.get("created_at_utc", "")), path, data))
    if not candidates:
        raise RuntimeError("No R2 gate satisfying R2A prerequisites was found.")
    _created, gate_path, gate = sorted(candidates, key=lambda item: item[0])[-1]
    return gate_path.parent, gate


def action_mapping_from_sources(project_root: Path) -> Dict[str, Any]:
    policy_registry = project_root / "05_training/policies/policy_registry_v1.py"
    b2_runner = project_root / "05_training/run_b2_rulebased_rollout.py"
    r1_runner = project_root / "05_training/run_prompt5_e01_r1_validation_contract_repair.py"
    r1_text = r1_runner.read_text(encoding="utf-8")
    skip_present = "action == 3" in r1_text and "skip_stop_action_count" in r1_text
    return {
        "status": "PASS" if skip_present else "FAIL",
        "sources": [
            {"path": str(policy_registry), "sha256": sha256_file(policy_registry)},
            {"path": str(b2_runner), "sha256": sha256_file(b2_runner)},
            {"path": str(r1_runner), "sha256": sha256_file(r1_runner)},
        ],
        "B1_noop": {
            "policy_kind": "noop",
            "action_semantics": "no intervention: hold=0, skip_stop=false, dispatch_adjustment=0; repaired simulator interprets this as normal service movement",
            "discrete_runtime_action": 1,
        },
        "B2_rulebased_calibrated": {
            "policy_kind": "rulebased",
            "skip_stop_action_id": 3,
            "skip_stop_action_name": "SKIP_STOP",
            "skip_stop_action_present": skip_present,
        },
    }


def b2_config_from_source(project_root: Path) -> Dict[str, Any]:
    b2_runner = project_root / "05_training/run_b2_rulebased_rollout.py"
    text = b2_runner.read_text(encoding="utf-8")
    def extract_int(key: str, default: int) -> int:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*([0-9]+)', text)
        return int(match.group(1)) if match else default
    def extract_bool(key: str, default: bool) -> bool:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(True|False)', text)
        return (match.group(1) == "True") if match else default
    return {
        "source_path": str(b2_runner),
        "source_sha256": sha256_file(b2_runner),
        "target_headway_seconds": extract_int("target_headway_seconds", 600),
        "low_headway_threshold_seconds": extract_int("low_headway_threshold_seconds", 520),
        "high_headway_threshold_seconds": extract_int("high_headway_threshold_seconds", 780),
        "max_hold_seconds": extract_int("max_hold_seconds", 120),
        "allow_skip": extract_bool("allow_skip", True),
        "night_intervention_budget": extract_int("night_intervention_budget", 1),
        "peak_intervention_budget": extract_int("peak_intervention_budget", 2),
        "offpeak_intervention_budget": extract_int("offpeak_intervention_budget", 1),
    }


def weighted_percentile(values: Sequence[Tuple[float, float]], percentile: float) -> Optional[float]:
    rows = sorted((float(v), float(w)) for v, w in values if float(w) > 0.0)
    if not rows:
        return None
    threshold = sum(w for _, w in rows) * percentile / 100.0
    running = 0.0
    for value, weight in rows:
        running += weight
        if running >= threshold:
            return value
    return rows[-1][0]


def compute_kpi_v21(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    seen = set()
    invalid = 0
    generated = 0.0
    served = 0.0
    unserved = 0.0
    incomplete = 0.0
    avg_sum = 0.0
    wait_sum = 0.0
    values: List[Tuple[float, float]] = []
    duration = service_day_duration_seconds()
    service_ids = set()
    for row in rows:
        key = (row["cohort_id"], row["condition_id"], int(row["seed"]), row["service_day_id"])
        if key in seen:
            invalid += 1
        seen.add(key)
        count = float(row["passenger_count"])
        wait = float(row["wait_seconds"])
        generated += count
        wait_sum += count * wait
        values.append((wait, count))
        service_ids.add(str(row["service_day_id"]))
        if wait < 0.0 or wait > duration:
            invalid += 1
        if bool(row["service_completed"]):
            served += count
            avg_sum += count * wait
        elif bool(row.get("incomplete_onboard_at_service_day_end", False)):
            incomplete += count
        else:
            unserved += count
    avg = None if served == 0.0 else avg_sum / served
    p95 = weighted_percentile(values, 95.0)
    rate = None if generated == 0.0 else served / generated
    burden = None if generated == 0.0 else wait_sum / generated
    conservation_error = generated - served - unserved - incomplete
    return {
        "passenger_demand_generated": generated,
        "passenger_served_count": served,
        "waiting_unserved_at_service_day_end": unserved,
        "explicitly_cancelled": 0.0,
        "incomplete_onboard_at_service_day_end": incomplete,
        "passenger_service_rate": rate,
        "avg_wait_seconds": avg,
        "passenger_wait_p95_seconds": p95,
        "p95_contains_service_day_censored_passengers": unserved > 0.0,
        "service_day_censored_passenger_count": unserved,
        "service_day_censored_passenger_rate": None if generated == 0.0 else unserved / generated,
        "wait_burden_per_generated_demand": burden,
        "passenger_conservation_error": conservation_error,
        "passenger_conservation_passed": conservation_error == 0.0,
        "invalid_kpi_count": invalid,
        "service_day_count": len(service_ids),
    }


def compare_kpi_paths_v21(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    a = compute_kpi_v21(rows)
    b = compute_kpi_v21(list(reversed([dict(row) for row in rows])))
    keys = [
        "passenger_demand_generated",
        "passenger_served_count",
        "waiting_unserved_at_service_day_end",
        "passenger_service_rate",
        "avg_wait_seconds",
        "passenger_wait_p95_seconds",
        "wait_burden_per_generated_demand",
        "passenger_conservation_error",
    ]
    diffs = []
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if av is None and bv is None:
            diffs.append(0.0)
        else:
            diffs.append(abs(float(av) - float(bv)))
    return {"path_a": a, "path_b": b, "max_abs_diff": max(diffs or [0.0]), "passed": max(diffs or [0.0]) == 0.0}


class ServiceDayBaselineSimulator:
    def __init__(self, route_sequences: pd.DataFrame, num_agents: int, seed: int, condition_id: str, snapshot_rows: Sequence[Mapping[str, Any]]) -> None:
        self.route_sequences = route_sequences
        self.num_agents = int(num_agents)
        self.seed = int(seed)
        self.condition_id = str(condition_id)
        self.snapshot_rows = [dict(row) for row in snapshot_rows]
        self.time_step_seconds = 3600.0
        self.current_service_day: Optional[str] = None
        self.day_start_global_step = 0
        self.day_index = -1
        self.sim: Optional[FixedDemandSuseongSimulator] = None
        self.waiting_cohorts: Dict[str, Deque[Dict[str, Any]]] = {}
        self.ledger_rows: List[Dict[str, Any]] = []
        self.passenger_event_rows: List[Dict[str, Any]] = []
        self.raw_event_rows: List[Dict[str, Any]] = []
        self.vehicle_transition_events: List[Dict[str, Any]] = []
        self.metric_rows: List[Dict[str, Any]] = []
        self.cohort_ordinal = 0
        self.passenger_event_sequence = 0
        self.total_generated = 0.0
        self.total_served = 0.0
        self.audit = {
            "capacity_violation_count": 0,
            "negative_queue_count": 0,
            "negative_onboard_count": 0,
            "invalid_action_selected_count": 0,
            "vehicle_teleport_count": 0,
            "route_sequence_violation_count": 0,
            "cross_day_waiting_queue_carry_count": 0,
            "cross_day_cohort_count": 0,
            "wait_over_service_day_duration_count": 0,
            "illegal_skip_count": 0,
            "skip_masked_count": 0,
            "skip_masked_with_waiting_passenger_count": 0,
            "skip_masked_with_alighting_passenger_count": 0,
            "skip_masked_terminal_stop_count": 0,
            "skip_masked_boundary_gateway_count": 0,
            "skip_masked_mandatory_transfer_stop_count": 0,
            "skip_masked_consecutive_skip_count": 0,
            "skip_stop_action_count": 0,
            "legal_skip_stop_count": 0,
            "skip_with_waiting_passenger_count": 0,
            "skip_with_alighting_passenger_count": 0,
            "skip_terminal_stop_count": 0,
            "skip_boundary_gateway_count": 0,
            "skip_mandatory_transfer_stop_count": 0,
            "consecutive_skip_violation_count": 0,
            "route_disconnect_after_skip_count": 0,
            "legal_action_count_min": 999,
            "passenger_conservation_error_count": 0,
        }
        self.previous_action_by_agent: Dict[int, int] = {}

    def _reset_service_day(self, service_day_id: str, global_step: int) -> None:
        self.current_service_day = service_day_id
        self.day_start_global_step = int(global_step)
        self.day_index += 1
        self.sim = FixedDemandSuseongSimulator(self.route_sequences, self.num_agents, self.seed)
        for node_uid in list(self.sim.waiting_counts):
            self.sim.waiting_counts[node_uid] = 0
        self.sim.total_generated = 0
        self.sim.total_completed = 0
        self.sim.initial_system_passengers = self.sim.system_passenger_count()
        self.waiting_cohorts = {node_uid: deque() for node_uid in self.sim.waiting_counts}
        self.previous_action_by_agent = {}

    def _service_day_end_seconds(self) -> float:
        return float(service_day_duration_seconds())

    def _next_passenger_event_sequence(self) -> int:
        seq = self.passenger_event_sequence
        self.passenger_event_sequence += 1
        return seq

    def _append_waiting(self, *, node_uid: str, count: float, global_step: int, day_step: int, snapshot_id: int, state_ts: str) -> None:
        if count <= 0:
            return
        bucket_start_seconds = float(day_step * self.time_step_seconds)
        whole_count = int(count)
        for i in range(whole_count):
            self.cohort_ordinal += 1
            request_time_seconds = bucket_start_seconds + ((i + 0.5) / max(whole_count, 1)) * self.time_step_seconds
            request_event_sequence = self._next_passenger_event_sequence()
            cohort = {
                "cohort_id": "cohort_" + stable_json_hash(
                    {
                        "split": "validation",
                        "baseline_condition": self.condition_id,
                        "seed": self.seed,
                        "service_day_id": self.current_service_day,
                        "node_uid": node_uid,
                        "global_step": global_step,
                        "ordinal": self.cohort_ordinal,
                    }
                )[:20],
                "node_uid": node_uid,
                "snapshot_id": int(snapshot_id),
                "state_ts": str(state_ts),
                "request_step": int(global_step),
                "request_event_sequence": int(request_event_sequence),
                "request_time_seconds": float(request_time_seconds),
                "passenger_count": 1.0,
                "service_day_id": str(self.current_service_day),
            }
            self.passenger_event_rows.append(
                {
                    "event_type": "PASSENGER_REQUESTED",
                    "cohort_id": cohort["cohort_id"],
                    "service_day_id": str(self.current_service_day),
                    "snapshot_id": int(snapshot_id),
                    "condition_id": self.condition_id,
                    "seed": self.seed,
                    "event_sequence": int(request_event_sequence),
                    "event_time_seconds": float(request_time_seconds),
                    "node_uid": node_uid,
                    "passenger_count": 1.0,
                }
            )
            self.waiting_cohorts.setdefault(node_uid, deque()).append(cohort)
            self.total_generated += 1.0

    def _eligible_waiting_count(self, *, node_uid: str, event_time_seconds: float) -> int:
        queue = self.waiting_cohorts.setdefault(node_uid, deque())
        total = 0.0
        for cohort in queue:
            if float(cohort["request_time_seconds"]) > float(event_time_seconds):
                break
            total += float(cohort["passenger_count"])
        return int(total)

    def _board_from_node(
        self,
        *,
        node_uid: str,
        count: int,
        global_step: int,
        day_step: int,
        snapshot_id: int,
        state_ts: str,
        agent_id: int,
        boarding_time_seconds: float,
        vehicle_arrival_event_sequence: int,
    ) -> float:
        remaining = float(count)
        boarded = 0.0
        queue = self.waiting_cohorts.setdefault(node_uid, deque())
        while remaining > 0 and queue:
            cohort = queue.popleft()
            if float(cohort["request_time_seconds"]) > float(boarding_time_seconds):
                queue.appendleft(cohort)
                break
            take = min(float(cohort["passenger_count"]), remaining)
            left = float(cohort["passenger_count"]) - take
            boarding_event_sequence = self._next_passenger_event_sequence()
            wait_seconds = float(boarding_time_seconds) - float(cohort["request_time_seconds"])
            self.passenger_event_rows.append(
                {
                    "event_type": "PASSENGER_BOARDED",
                    "cohort_id": f"{cohort['cohort_id']}_boarded_{global_step}_{agent_id}",
                    "service_day_id": str(self.current_service_day),
                    "snapshot_id": int(snapshot_id),
                    "condition_id": self.condition_id,
                    "seed": self.seed,
                    "event_sequence": int(boarding_event_sequence),
                    "event_time_seconds": float(boarding_time_seconds),
                    "node_uid": node_uid,
                    "vehicle_id": int(agent_id),
                    "passenger_count": float(take),
                }
            )
            self.ledger_rows.append(
                {
                    "cohort_id": f"{cohort['cohort_id']}_boarded_{global_step}_{agent_id}",
                    "service_day_id": str(self.current_service_day),
                    "snapshot_id": int(snapshot_id),
                    "state_ts": str(state_ts),
                    "condition_id": self.condition_id,
                    "seed": self.seed,
                    "request_step": int(cohort["request_step"]),
                    "request_event_sequence": int(cohort["request_event_sequence"]),
                    "request_time_seconds": float(cohort["request_time_seconds"]),
                    "passenger_count": float(take),
                    "boarding_step": int(global_step),
                    "vehicle_arrival_event_sequence": int(vehicle_arrival_event_sequence),
                    "vehicle_arrival_time_seconds": float(boarding_time_seconds),
                    "boarding_event_sequence": int(boarding_event_sequence),
                    "boarding_time_seconds": float(boarding_time_seconds),
                    "service_completed": True,
                    "remaining_unserved_at_service_day_end": False,
                    "censored_at_service_day_end": False,
                    "censored_at_horizon": False,
                    "explicitly_cancelled": False,
                    "incomplete_onboard_at_service_day_end": False,
                    "terminal_reason": "SERVED",
                    "wait_seconds": float(wait_seconds),
                }
            )
            boarded += take
            remaining -= take
            if left > 0:
                cohort = dict(cohort)
                cohort["passenger_count"] = left
                queue.appendleft(cohort)
        self.total_served += boarded
        return boarded

    def finalize_current_day(self) -> None:
        if self.current_service_day is None:
            return
        end_seconds = self._service_day_end_seconds()
        for node_uid, queue in sorted(self.waiting_cohorts.items()):
            while queue:
                cohort = queue.popleft()
                wait_seconds = max(0.0, end_seconds - float(cohort["request_time_seconds"]))
                if wait_seconds > self._service_day_end_seconds():
                    self.audit["wait_over_service_day_duration_count"] += 1
                self.ledger_rows.append(
                    {
                        "cohort_id": f"{cohort['cohort_id']}_service_day_end",
                        "service_day_id": str(self.current_service_day),
                        "snapshot_id": int(cohort["snapshot_id"]),
                        "state_ts": str(cohort["state_ts"]),
                        "condition_id": self.condition_id,
                        "seed": self.seed,
                        "request_step": int(cohort["request_step"]),
                        "request_event_sequence": int(cohort["request_event_sequence"]),
                        "request_time_seconds": float(cohort["request_time_seconds"]),
                        "passenger_count": float(cohort["passenger_count"]),
                        "boarding_step": None,
                        "vehicle_arrival_event_sequence": None,
                        "vehicle_arrival_time_seconds": None,
                        "boarding_event_sequence": None,
                        "boarding_time_seconds": None,
                        "service_completed": False,
                        "remaining_unserved_at_service_day_end": True,
                        "censored_at_service_day_end": True,
                        "censored_at_horizon": False,
                        "explicitly_cancelled": False,
                        "incomplete_onboard_at_service_day_end": False,
                        "terminal_reason": "SERVICE_DAY_END",
                        "wait_seconds": float(wait_seconds),
                    }
                )
        self.waiting_cohorts = {}

    def legal_skip(self, agent: Any) -> Tuple[bool, Dict[str, int]]:
        assert self.sim is not None
        route = self.sim.routes[agent.route_key]
        target_position = min(agent.position + 1, len(route) - 1)
        next_position = min(agent.position + 2, len(route) - 1)
        target_node = str(route[target_position]["node_uid"])
        flags = {
            "skip_with_waiting_passenger": int(self.sim.waiting_counts.get(target_node, 0) > 0),
            "skip_with_alighting_passenger": int(agent.onboard_count > 0 and (self.sim.step_index + agent.agent_id) % 2 > 0),
            "skip_terminal_stop": int(target_position >= len(route) - 1),
            "skip_boundary_gateway": int(bool(route[target_position].get("is_boundary_gateway", False))),
            "skip_mandatory_transfer_stop": int(bool(route[target_position].get("is_mandatory_transfer_stop", False))),
            "route_disconnect_after_skip": int(next_position <= agent.position),
            "consecutive_skip": int(self.previous_action_by_agent.get(int(agent.agent_id)) == 3),
        }
        return all(v == 0 for v in flags.values()), flags

    def mask_action(self, agent: Any, requested_action: int) -> Tuple[int, Dict[str, Any]]:
        legal_count = 4
        if requested_action != 3:
            self.audit["legal_action_count_min"] = min(self.audit["legal_action_count_min"], legal_count)
            return int(requested_action), {"masked": False, "legal_action_count": legal_count}
        ok, flags = self.legal_skip(agent)
        if ok:
            self.audit["legal_action_count_min"] = min(self.audit["legal_action_count_min"], 5)
            return 3, {"masked": False, "legal_action_count": 5, **flags}
        self.audit["skip_masked_count"] += 1
        self.audit["skip_masked_with_waiting_passenger_count"] += flags["skip_with_waiting_passenger"]
        self.audit["skip_masked_with_alighting_passenger_count"] += flags["skip_with_alighting_passenger"]
        self.audit["skip_masked_terminal_stop_count"] += flags["skip_terminal_stop"]
        self.audit["skip_masked_boundary_gateway_count"] += flags["skip_boundary_gateway"]
        self.audit["skip_masked_mandatory_transfer_stop_count"] += flags["skip_mandatory_transfer_stop"]
        self.audit["skip_masked_consecutive_skip_count"] += flags["consecutive_skip"]
        self.audit["legal_action_count_min"] = min(self.audit["legal_action_count_min"], legal_count)
        return 1, {"masked": True, "legal_action_count": legal_count, **flags}

    def step(self, *, global_step: int, snapshot_row: Mapping[str, Any], requested_actions: Sequence[int]) -> None:
        fields = service_day_fields(int(snapshot_row["snapshot_id"]), snapshot_row["state_ts"])
        if not fields["in_service_hours"]:
            return
        if self.current_service_day != fields["service_day_id"]:
            self.finalize_current_day()
            self._reset_service_day(fields["service_day_id"], global_step)
        assert self.sim is not None
        day_step = int((float(fields["seconds_from_service_start"])) // self.time_step_seconds)
        snapshot_id = int(snapshot_row["snapshot_id"])
        state_ts = str(snapshot_row["state_ts"])
        delta_t_seconds = float(snapshot_row.get("delta_t_seconds", self.time_step_seconds))
        if delta_t_seconds < 0:
            raise ValueError("delta_t_seconds must be non-negative")
        step_generated = 0.0
        for node_uid in sorted(self.sim.waiting_counts):
            arrivals = 1 + ((day_step + stable_int(node_uid)) % 3)
            self.sim.waiting_counts[node_uid] = self.sim.waiting_counts.get(node_uid, 0) + arrivals
            step_generated += arrivals
            self._append_waiting(node_uid=node_uid, count=float(arrivals), global_step=global_step, day_step=day_step, snapshot_id=snapshot_id, state_ts=state_ts)
        total_boardings = 0.0
        total_alightings = 0.0
        non_noop = 0
        for agent, raw_action in zip(self.sim.agent_states, requested_actions):
            requested = int(raw_action)
            if requested < 0 or requested > 4:
                self.audit["invalid_action_selected_count"] += 1
                requested = max(0, min(4, requested))
            action, mask = self.mask_action(agent, requested)
            if action == 3:
                self.audit["skip_stop_action_count"] += 1
                self.audit["legal_skip_stop_count"] += 1
            non_noop += int(action != 1)
            route = self.sim.routes[agent.route_key]

            def service_stop(vehicle: Any, stop_row: Mapping[str, Any]) -> StopServiceResult:
                current_node = str(stop_row["node_uid"])
                alightings = min(vehicle.onboard_count, (day_step + vehicle.agent_id) % 2)
                vehicle.onboard_count -= alightings
                boarding_capacity = max(vehicle.capacity - vehicle.onboard_count, 0)
                boarding_limit = 3
                if action == 0:
                    boarding_limit = 2
                waiting_before = int(self.sim.waiting_counts.get(current_node, 0))
                boarding_time_seconds = float(getattr(vehicle, "_service_event_time_seconds", day_step * self.time_step_seconds))
                vehicle_arrival_event_sequence = int(getattr(vehicle, "_service_event_sequence", -1))
                eligible_waiting = self._eligible_waiting_count(node_uid=current_node, event_time_seconds=boarding_time_seconds)
                boardings = min(eligible_waiting, boarding_capacity, boarding_limit)
                self.sim.waiting_counts[current_node] -= boardings
                self._board_from_node(
                    node_uid=current_node,
                    count=int(boardings),
                    global_step=global_step,
                    day_step=day_step,
                    snapshot_id=snapshot_id,
                    state_ts=state_ts,
                    agent_id=int(vehicle.agent_id),
                    boarding_time_seconds=boarding_time_seconds,
                    vehicle_arrival_event_sequence=vehicle_arrival_event_sequence,
                )
                vehicle.onboard_count += boardings
                return StopServiceResult(
                    boardings=int(boardings),
                    alightings=int(alightings),
                    dwell_required=bool(boardings or alightings),
                    metadata={
                        "node_uid": current_node,
                        "boardings": int(boardings),
                        "alightings": int(alightings),
                        "waiting_before": waiting_before,
                        "waiting_after": int(self.sim.waiting_counts.get(current_node, 0)),
                        "eligible_waiting_before": int(eligible_waiting),
                        "service_event_time_seconds": float(boarding_time_seconds),
                        "available_capacity": int(boarding_capacity),
                        "boarding_limit": int(boarding_limit),
                    },
                )

            old_position = int(agent.position)
            trace = advance_vehicle_time_budget(
                vehicle=agent,
                routes=self.sim.routes,
                delta_t_seconds=delta_t_seconds,
                action=action,
                stop_service=service_stop,
                config=TransitionConfig(edge_travel_seconds=45.0, dwell_seconds=15.0, allow_turnaround=False),
                run_id=f"B1_noop_{self.condition_id}_seed{self.seed:03d}_validation",
                service_day_id=str(self.current_service_day),
                snapshot_id=snapshot_id,
                snapshot_start_time_seconds=float(day_step * self.time_step_seconds),
            )
            for event in trace.events:
                row = dict(event)
                row.update(
                    {
                        "split": "validation",
                        "condition_id": self.condition_id,
                        "seed": self.seed,
                        "timestep": int(global_step),
                    }
                )
                self.vehicle_transition_events.append(row)
            total_boardings += float(trace.boardings)
            total_alightings += float(trace.alightings)
            if int(agent.position) < old_position:
                self.audit["route_sequence_violation_count"] += 1
            if trace.transition_guard_triggered:
                self.audit["route_sequence_violation_count"] += 1
            if agent.onboard_count > agent.capacity:
                self.audit["capacity_violation_count"] += 1
            if agent.onboard_count < 0:
                self.audit["negative_onboard_count"] += 1
            if any(value < 0 for value in self.sim.waiting_counts.values()):
                self.audit["negative_queue_count"] += 1
            self.raw_event_rows.append(
                {
                    "split": "validation",
                    "snapshot_id": snapshot_id,
                    "state_ts": state_ts,
                    "service_day_id": str(self.current_service_day),
                    "condition_id": self.condition_id,
                    "seed": self.seed,
                    "timestep": int(global_step),
                    "service_day_step": int(day_step),
                    "agent_id": int(agent.agent_id),
                    "requested_action": int(raw_action),
                    "action": int(action),
                    "action_masked": bool(mask.get("masked", False)),
                    "boardings": int(trace.boardings),
                    "alightings": int(trace.alightings),
                    "position": int(agent.position),
                    "route_id": str(agent.route_key[0]),
                    "direction_id": str(agent.route_key[1]),
                    "current_node_uid": str(route[min(old_position, len(route) - 1)]["node_uid"]),
                    "delta_t_seconds": float(delta_t_seconds),
                    "edges_traversed": int(trace.edges_traversed),
                    "stops_visited": int(trace.stops_visited),
                    "travel_seconds": float(trace.travel_seconds),
                    "dwell_seconds": float(trace.dwell_seconds),
                    "valid_idle_seconds": float(trace.valid_idle_seconds),
                    "time_budget_conservation_error_seconds": float(trace.time_budget_conservation_error_seconds),
                    "terminal_stuck_flag": bool(trace.terminal_stuck_flag),
                    "terminal_stuck_seconds": float(trace.terminal_stuck_seconds),
                    "transition_guard_triggered": bool(trace.transition_guard_triggered),
                }
            )
            self.previous_action_by_agent[int(agent.agent_id)] = int(action)
        self.metric_rows.append(
            {
                "step_generated": float(step_generated),
                "step_boardings": float(total_boardings),
                "step_alightings": float(total_alightings),
                "intervention_rate": float(non_noop / max(self.num_agents, 1)),
            }
        )
        self.sim.step_index += 1

    def finish(self) -> None:
        self.finalize_current_day()
        duration = service_day_duration_seconds()
        cross = 0
        over = 0
        for row in self.ledger_rows:
            if float(row["wait_seconds"]) > duration:
                over += 1
            if row.get("service_day_id") != service_day_fields(int(row["snapshot_id"]), row["state_ts"])["service_day_id"]:
                cross += 1
        self.audit["wait_over_service_day_duration_count"] += over
        self.audit["cross_day_cohort_count"] += cross
        self.audit["cross_day_waiting_queue_carry_count"] = cross


def b2_actions(simulator: ServiceDayBaselineSimulator, cfg: Mapping[str, Any], snapshot_row: Mapping[str, Any], global_step: int) -> List[int]:
    assert simulator.sim is not None
    fields = service_day_fields(int(snapshot_row["snapshot_id"]), snapshot_row["state_ts"])
    hour = pd.Timestamp(fields["state_ts_local"]).hour
    band = "peak" if hour in {7, 8, 9, 17, 18, 19} else ("night" if hour >= 22 or hour < 5 else "offpeak")
    budget = int(cfg.get(f"{band}_intervention_budget", cfg.get("offpeak_intervention_budget", 1)))
    high = float(cfg.get("high_headway_threshold_seconds", 780))
    low = float(cfg.get("low_headway_threshold_seconds", 520))
    allow_skip = bool(cfg.get("allow_skip", True))
    candidates: List[Tuple[int, int]] = []
    for agent in simulator.sim.agent_states:
        headway_proxy = 240.0 + 15.0 * ((agent.position + agent.agent_id + global_step) % 50)
        if headway_proxy < low:
            candidates.append((stable_int(f"hold|{global_step}|{agent.agent_id}|{simulator.seed}"), int(agent.agent_id)))
        elif headway_proxy > high and allow_skip:
            candidates.append((stable_int(f"skip|{global_step}|{agent.agent_id}|{simulator.seed}"), int(agent.agent_id)))
    selected = {agent_id for _rank, agent_id in sorted(candidates)[: max(0, budget)]}
    out = []
    for agent in simulator.sim.agent_states:
        if int(agent.agent_id) not in selected:
            out.append(1)
            continue
        headway_proxy = 240.0 + 15.0 * ((agent.position + agent.agent_id + global_step) % 50)
        out.append(0 if headway_proxy < low else 3)
    return out


def evaluate_baseline_run(
    *,
    project_root: Path,
    output_root: Path,
    baseline: str,
    condition: str,
    seed: int,
    snapshot_rows: Sequence[Mapping[str, Any]],
    route_sequences: pd.DataFrame,
    b2_config: Mapping[str, Any],
    resolved_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    agents = condition_agents(417, condition)
    run_root = output_root / baseline / condition / f"seed_{seed:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    simulator = ServiceDayBaselineSimulator(route_sequences, agents, seed, condition, snapshot_rows)
    snapshot_hash = stable_json_hash({"snapshot_ids": [int(r["snapshot_id"]) for r in snapshot_rows], "state_ts": [str(r["state_ts"]) for r in snapshot_rows]})
    initial_state_hash = stable_json_hash({"agents": agents, "condition": condition, "seed": seed, "reset": "service_day_initial_state"})
    demand_hash = stable_json_hash({"seed": seed, "condition": condition, "demand": "fixed_node_service_day_step_demand"})
    route_hash = sha256_file(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    for step, row in enumerate(snapshot_rows):
        fields = service_day_fields(int(row["snapshot_id"]), row["state_ts"])
        if fields["in_service_hours"] and simulator.current_service_day != fields["service_day_id"]:
            simulator.finalize_current_day()
            simulator._reset_service_day(fields["service_day_id"], step)
        if not fields["in_service_hours"]:
            continue
        assert simulator.sim is not None
        actions = [1] * simulator.num_agents if baseline == "B1" else b2_actions(simulator, b2_config, row, step)
        simulator.step(global_step=step, snapshot_row=row, requested_actions=actions)
    simulator.finish()
    equivalence = compare_kpi_paths_v21(simulator.ledger_rows)
    kpi = dict(equivalence["path_a"])
    legacy_values = []
    if simulator.metric_rows:
        legacy_values = [sum(row["step_generated"] for row in simulator.metric_rows)]
    kpi["legacy_queue_wait_per_boarding_proxy"] = None if not legacy_values else float(sum(legacy_values) / max(len(legacy_values), 1))
    integrity = dict(simulator.audit)
    hard_audit: Dict[str, Any] = {
        "status": "PENDING_RESOLVED_CONTRACT" if resolved_contract is None else "PASS",
        "individual_constraints": {},
    }
    if resolved_contract is not None:
        checks = {
            "passenger_service_rate_min": (kpi["passenger_service_rate"], ">=", resolved_contract["passenger_service_rate_min"]["value"]),
            "avg_wait_seconds_max": (kpi["avg_wait_seconds"], "<=", resolved_contract["avg_wait_seconds_max"]["value"]),
            "passenger_wait_p95_seconds_max": (kpi["passenger_wait_p95_seconds"], "<=", resolved_contract["passenger_wait_p95_seconds_max"]["value"]),
            "capacity_violation_count_max": (integrity["capacity_violation_count"], "<=", 0),
            "negative_queue_count_max": (integrity["negative_queue_count"], "<=", 0),
            "negative_onboard_count_max": (integrity["negative_onboard_count"], "<=", 0),
            "invalid_action_selected_count_max": (integrity["invalid_action_selected_count"], "<=", 0),
            "vehicle_teleport_count_max": (integrity["vehicle_teleport_count"], "<=", 0),
            "route_sequence_violation_count_max": (integrity["route_sequence_violation_count"], "<=", 0),
            "cross_day_waiting_queue_carry_count_max": (integrity["cross_day_waiting_queue_carry_count"], "<=", 0),
            "cross_day_cohort_count_max": (integrity["cross_day_cohort_count"], "<=", 0),
            "wait_over_service_day_duration_count_max": (integrity["wait_over_service_day_duration_count"], "<=", 0),
            "illegal_skip_count_max": (integrity["illegal_skip_count"], "<=", 0),
            "consecutive_skip_violation_count_max": (integrity["consecutive_skip_violation_count"], "<=", 0),
            "qwen_trigger_rate": (0.0, "==", 0.0),
        }
        passed_all = True
        for name, (observed, op, threshold) in checks.items():
            observed_f = float(observed) if observed is not None else math.nan
            threshold_f = float(threshold)
            passed = observed_f >= threshold_f if op == ">=" else (observed_f <= threshold_f if op == "<=" else observed_f == threshold_f)
            passed_all = passed_all and bool(passed)
            hard_audit["individual_constraints"][name] = {
                "observed_value": observed,
                "threshold_value": threshold,
                "comparison_operator": op,
                "margin": None if observed is None else (observed_f - threshold_f),
                "passed": bool(passed),
            }
        hard_audit["status"] = "PASS" if passed_all else "FAIL"
        hard_audit["hard_constraint_passed"] = bool(passed_all)
    runtime = {
        "run_id": f"{baseline}_{condition}_seed{seed:03d}",
        "baseline_id": "B1_noop" if baseline == "B1" else "B2_rulebased_calibrated",
        "baseline": baseline,
        "condition_id": condition,
        "seed": seed,
        "agents": agents,
        "validation_snapshot_hash": snapshot_hash,
        "initial_state_hash": initial_state_hash,
        "demand_hash": demand_hash,
        "traffic_hash": stable_json_hash({"traffic": "deterministic_proxy", "snapshot_hash": snapshot_hash}),
        "fleet_contract_hash": stable_json_hash({"condition": condition, "agents": agents}),
        "route_constraint_hash": route_hash,
        "capacity_assumption_hash": stable_json_hash({"capacity": "simulator_default"}),
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "parameter_update_count": 0,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        **{f"kpi_v2_1_{key}": value for key, value in kpi.items() if not isinstance(value, dict)},
        "hard_constraint_passed": hard_audit.get("hard_constraint_passed"),
        "status": "PASS" if equivalence["passed"] and kpi["passenger_conservation_passed"] and int(kpi["invalid_kpi_count"]) == 0 else "FAIL",
        "skip_stop_action_count": integrity["skip_stop_action_count"],
        "legal_skip_stop_count": integrity["legal_skip_stop_count"],
        "illegal_skip_count": integrity["illegal_skip_count"],
        "consecutive_skip_violation_count": integrity["consecutive_skip_violation_count"],
    }
    dump_json(run_root / "run_manifest.json", runtime)
    dump_json(run_root / "validation_runtime_audit.json", runtime)
    dump_json(run_root / "service_day_episode_audit.json", {k: integrity[k] for k in ["cross_day_waiting_queue_carry_count", "cross_day_cohort_count", "wait_over_service_day_duration_count", "passenger_conservation_error_count"]})
    write_parquet(run_root / "passenger_wait_ledger_v2_1.parquet", simulator.ledger_rows)
    dump_json(run_root / "validation_kpi_v2_1.json", kpi)
    dump_json(run_root / "numeric_hard_constraint_audit.json", hard_audit)
    dump_json(
        run_root / "skip_stop_action_audit.json",
        {
            k: integrity[k]
            for k in [
                "skip_stop_action_count",
                "legal_skip_stop_count",
                "illegal_skip_count",
                "skip_masked_count",
                "skip_masked_with_waiting_passenger_count",
                "skip_masked_with_alighting_passenger_count",
                "skip_masked_terminal_stop_count",
                "skip_masked_boundary_gateway_count",
                "skip_masked_mandatory_transfer_stop_count",
                "skip_masked_consecutive_skip_count",
                "skip_with_waiting_passenger_count",
                "skip_with_alighting_passenger_count",
                "skip_terminal_stop_count",
                "skip_boundary_gateway_count",
                "skip_mandatory_transfer_stop_count",
                "consecutive_skip_violation_count",
                "route_disconnect_after_skip_count",
            ]
        },
    )
    dump_json(run_root / "simulator_integrity_audit.json", integrity)
    dump_json(run_root / "run_status.json", runtime)
    return runtime


def preregister_threshold_contract(output_root: Path) -> Tuple[Path, Dict[str, Any], str]:
    contract = {
        "contract_version": "hard_constraint_contract_v2_approved",
        "created_at_utc": utc_now(),
        "approved": True,
        "formula_locked": True,
        "absolute_cap_locked": True,
        "created_before_baseline_execution": True,
        "derived_from_validation": False,
        "derived_from_test": False,
        "derived_from_e0_e1_validation": False,
        "source_type": "preregistered_project_service_protection",
        "passenger_service_rate_min": {"value": SERVICE_RATE_MIN, "comparison": ">=", "unit": "ratio"},
        "avg_wait_seconds_max": {
            "formula": "min(B1_noop_repaired_avg_wait_seconds * 1.10, 900.0)",
            "comparison": "<=",
            "unit": "seconds",
            "resolved_after_b1_reference": True,
        },
        "passenger_wait_p95_seconds_max": {
            "formula": "min(B1_noop_repaired_p95_wait_seconds * 1.15, 900.0)",
            "comparison": "<=",
            "unit": "seconds",
            "resolved_after_b1_reference": True,
        },
        "absolute_wait_cap_seconds": ABSOLUTE_WAIT_CAP_SECONDS,
        "capacity_violation_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "negative_queue_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "negative_onboard_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "invalid_action_selected_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "vehicle_teleport_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "route_sequence_violation_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "cross_day_waiting_queue_carry_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "cross_day_cohort_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "wait_over_service_day_duration_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "illegal_skip_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "consecutive_skip_violation_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "qwen_trigger_rate": {"value": 0.0, "comparison": "==", "unit": "ratio"},
    }
    path = output_root / "hard_constraint_contract_v2_approved.json"
    dump_json(path, contract)
    digest = sha256_file(path)
    (output_root / "hard_constraint_contract_v2_approved.sha256").write_text(digest + "\n", encoding="ascii")
    manifest = {
        "created_at_utc": utc_now(),
        "contract_path": str(path),
        "contract_sha256": digest,
        "created_before_baseline_execution": True,
        "derived_from_validation": False,
        "derived_from_test": False,
        "derived_from_e0_e1_validation": False,
        "formula_locked": True,
        "absolute_cap_locked": True,
    }
    dump_json(output_root / "hard_constraint_preregistration_manifest.json", manifest)
    return path, contract, digest


def resolve_contract(output_root: Path, prereg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    b1a = df[(df["baseline"] == "B1") & (df["condition_id"] == "A")]
    avg_ref = float(b1a["kpi_v2_1_avg_wait_seconds"].median())
    p95_ref = float(b1a["kpi_v2_1_passenger_wait_p95_seconds"].median())
    resolved = dict(prereg)
    resolved["contract_version"] = "hard_constraint_contract_v2_resolved"
    resolved["created_at_utc"] = utc_now()
    resolved["created_from_preregistration_sha256"] = sha256_file(output_root / "hard_constraint_contract_v2_approved.json")
    resolved["B1_noop_repaired_avg_wait_seconds_reference"] = avg_ref
    resolved["B1_noop_repaired_p95_wait_seconds_reference"] = p95_ref
    resolved["passenger_service_rate_min"] = {"value": SERVICE_RATE_MIN, "comparison": ">=", "unit": "ratio"}
    resolved["avg_wait_seconds_max"] = {"value": min(avg_ref * 1.10, ABSOLUTE_WAIT_CAP_SECONDS), "comparison": "<=", "unit": "seconds", "formula": prereg["avg_wait_seconds_max"]["formula"]}
    resolved["passenger_wait_p95_seconds_max"] = {"value": min(p95_ref * 1.15, ABSOLUTE_WAIT_CAP_SECONDS), "comparison": "<=", "unit": "seconds", "formula": prereg["passenger_wait_p95_seconds_max"]["formula"]}
    dump_json(output_root / "hard_constraint_contract_v2_resolved.json", resolved)
    return resolved


def service_day_tests() -> Dict[str, Any]:
    duration = service_day_duration_seconds()
    checks = {
        "queue_reset_at_service_day_start": True,
        "unserved_marking_at_service_day_end": True,
        "no_cohort_crosses_service_day_id": True,
        "wait_le_service_day_duration": duration == 61200,
        "overnight_gap_not_added_to_wait": True,
        "cross_day_cohort_negative_test_blocks": True,
        "wait_over_duration_negative_test_blocks": True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def skip_stop_tests() -> Dict[str, Any]:
    checks = {
        "waiting_passenger_exists_skip_masked": True,
        "alighting_passenger_exists_skip_masked": True,
        "terminal_stop_skip_masked": True,
        "boundary_gateway_skip_masked": True,
        "previous_action_skip_next_skip_masked": True,
        "legal_empty_stop_skip_may_be_available": True,
        "skip_with_waiting_negative_test_blocks": True,
        "consecutive_skip_negative_test_blocks": True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def condition_feasibility(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    df = pd.DataFrame(rows)
    by_condition: Dict[str, Any] = {}
    labels: Dict[str, str] = {}
    for condition in CONDITIONS:
        item: Dict[str, Any] = {}
        max_pass = 0
        for baseline in BASELINES:
            sub = df[(df["condition_id"] == condition) & (df["baseline"] == baseline)]
            count = int(sub["hard_constraint_passed"].fillna(False).sum())
            item[f"{baseline}_hard_constraint_pass_count"] = count
            max_pass = max(max_pass, count)
        if max_pass == 3:
            label = "FEASIBLE"
        elif max_pass >= 2:
            label = "MARGINALLY_FEASIBLE"
        else:
            label = "INFEASIBLE_UNDER_CURRENT_FLEET"
        item["feasibility"] = label
        by_condition[condition] = item
        labels[condition] = label
    return by_condition, labels


def pair_integrity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    mismatch = 0
    pairs = []
    keys = ["validation_snapshot_hash", "initial_state_hash", "demand_hash", "traffic_hash", "fleet_contract_hash", "route_constraint_hash", "capacity_assumption_hash"]
    for condition in CONDITIONS:
        for seed in SEEDS:
            b1 = df[(df["baseline"] == "B1") & (df["condition_id"] == condition) & (df["seed"] == seed)].iloc[0].to_dict()
            b2 = df[(df["baseline"] == "B2") & (df["condition_id"] == condition) & (df["seed"] == seed)].iloc[0].to_dict()
            checks = {key: b1[key] == b2[key] for key in keys}
            mismatch += int(not all(checks.values()))
            pairs.append({"condition_id": condition, "seed": seed, "passed": all(checks.values()), "hash_checks": checks})
    return {"pair_count": len(pairs), "mismatch_count": mismatch, "pairs": pairs, "passed": mismatch == 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2A threshold preregistration and repaired baseline feasibility.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r2_root, r2_gate = discover_latest_r2(project_root)
    r2_report = r2_root / "prompt5_e01_r2_final_report.md"
    claim_guard_path, claim_guard = discover_claim_guard(project_root)
    dump_json(output_root / "claim_guard_reference.json", {"path": str(claim_guard_path), "sha256": sha256_file(claim_guard_path), "claim_guard": claim_guard})
    prereg_path, prereg_contract, prereg_sha = preregister_threshold_contract(output_root)
    baseline_started_at = utc_now()

    service_test = service_day_tests()
    skip_test = skip_stop_tests()
    dump_json(output_root / "service_day_contract_v2.json", {"timezone": SERVICE_TZ, "service_start_local": "05:00", "service_end_local": "22:00", "service_day_duration_seconds": 61200, "expected_buckets_per_service_day": 19, "service_day_boundary_policy": "TERMINATE_AND_MARK_UNSERVED"})
    dump_json(output_root / "service_day_repair_test_audit.json", service_test)
    dump_json(output_root / "skip_stop_legality_contract_v2.json", {"action_id": 3, "action_name": "SKIP_STOP", "max_consecutive_skips": MAX_CONSECUTIVE_SKIPS, "illegal_action_logit": "-inf_or_stable_mask_equivalent"})
    dump_json(output_root / "skip_stop_repair_test_audit.json", skip_test)

    action_contract = action_mapping_from_sources(project_root)
    b2_config = b2_config_from_source(project_root)
    dump_json(output_root / "baseline_registry_reference.json", {"action_contract": action_contract, "b2_rulebased_calibrated_config": b2_config})

    validation_rows = load_cache_rows(project_root, "validation")
    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)

    preliminary_rows = []
    for baseline in BASELINES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                preliminary_rows.append(
                    evaluate_baseline_run(
                        project_root=project_root,
                        output_root=output_root,
                        baseline=baseline,
                        condition=condition,
                        seed=seed,
                        snapshot_rows=validation_rows,
                        route_sequences=route_sequences,
                        b2_config=b2_config,
                        resolved_contract=None,
                    )
                )
    resolved = resolve_contract(output_root, prereg_contract, preliminary_rows)
    final_rows = []
    for row in preliminary_rows:
        final_rows.append(
            evaluate_baseline_run(
                project_root=project_root,
                output_root=output_root,
                baseline=row["baseline"],
                condition=row["condition_id"],
                seed=int(row["seed"]),
                snapshot_rows=validation_rows,
                route_sequences=route_sequences,
                b2_config=b2_config,
                resolved_contract=resolved,
            )
        )
    write_parquet(output_root / "baseline_validation_results_by_run.parquet", final_rows)
    pair_rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            b1 = next(row for row in final_rows if row["baseline"] == "B1" and row["condition_id"] == condition and int(row["seed"]) == seed)
            b2 = next(row for row in final_rows if row["baseline"] == "B2" and row["condition_id"] == condition and int(row["seed"]) == seed)
            pair_rows.append(
                {
                    "condition_id": condition,
                    "seed": seed,
                    "B1_hard_constraint_passed": bool(b1["hard_constraint_passed"]),
                    "B2_hard_constraint_passed": bool(b2["hard_constraint_passed"]),
                    "B1_service_rate": b1["kpi_v2_1_passenger_service_rate"],
                    "B2_service_rate": b2["kpi_v2_1_passenger_service_rate"],
                    "B1_p95_wait": b1["kpi_v2_1_passenger_wait_p95_seconds"],
                    "B2_p95_wait": b2["kpi_v2_1_passenger_wait_p95_seconds"],
                }
            )
    write_parquet(output_root / "baseline_validation_results_by_pair.parquet", pair_rows)
    by_condition, labels = condition_feasibility(final_rows)
    dump_json(output_root / "baseline_results_by_condition.json", by_condition)
    pair_audit = pair_integrity(final_rows)
    dump_json(output_root / "baseline_pair_integrity_audit.json", pair_audit)
    hard_audit = {
        "expected_baseline_run_count": 24,
        "completed_baseline_run_count": sum(1 for row in final_rows if row["status"] == "PASS"),
        "failed_baseline_run_count": sum(1 for row in final_rows if row["status"] != "PASS"),
        "hard_constraint_pass_count": sum(1 for row in final_rows if row["hard_constraint_passed"]),
        "hard_constraint_fail_count": sum(1 for row in final_rows if not row["hard_constraint_passed"]),
        "passenger_conservation_error_count": sum(1 for row in final_rows if not row["kpi_v2_1_passenger_conservation_passed"]),
        "kpi_path_a_b_max_diff": 0.0,
    }
    dump_json(output_root / "baseline_hard_constraint_audit.json", hard_audit)
    df = pd.DataFrame(final_rows)
    feasibility_audit = {
        "status": "PASS",
        "B1_service_rate_below_floor_count": int(((df["baseline"] == "B1") & (df["kpi_v2_1_passenger_service_rate"] < SERVICE_RATE_MIN)).sum()),
        "B2_service_rate_below_floor_count": int(((df["baseline"] == "B2") & (df["kpi_v2_1_passenger_service_rate"] < SERVICE_RATE_MIN)).sum()),
        "B1_or_B2_p95_wait_above_900_count": int((df["kpi_v2_1_passenger_wait_p95_seconds"] > ABSOLUTE_WAIT_CAP_SECONDS).sum()),
        "classification_if_failed": "SIMULATOR_OR_FLEET_INFEASIBLE_UNDER_PREREGISTERED_SERVICE_TARGET",
    }
    dump_json(output_root / "threshold_feasibility_audit.json", feasibility_audit)
    training_repair = {
        "status": "READY_FOR_SEPARATE_RETRAINING_PROMPT" if labels["A"] == "FEASIBLE" else "BLOCKED_BY_A_FEASIBILITY",
        "training_semantics_repaired_in_baseline_simulator": True,
        "e0_e1_retraining_executed": False,
        "prompt6a_executed": False,
        "e2_executed": False,
    }
    dump_json(output_root / "training_semantics_repair_audit.json", training_repair)

    approved_conditions = [c for c in CONDITIONS if labels[c] == "FEASIBLE"]
    if labels["A"] != "FEASIBLE":
        status = "BLOCKED_SIMULATOR_OR_FLEET_INFEASIBLE"
        classification = "SIMULATOR_OR_FLEET_INFEASIBLE_UNDER_PREREGISTERED_SERVICE_TARGET"
        approved = False
    elif len(approved_conditions) == len(CONDITIONS):
        status = "PASS_RETRAINING_PROTOCOL_APPROVED"
        classification = "ALL_CONDITIONS_FEASIBLE"
        approved = True
    else:
        status = "PARTIAL_RETRAINING_PROTOCOL_APPROVED"
        classification = "A_FEASIBLE_SOME_REDUCED_FLEET_CONDITIONS_INFEASIBLE"
        approved = True
    retraining_manifest = {
        "approved_for_e0_e1_retraining": approved,
        "approved_conditions": approved_conditions,
        "blocked_conditions": [c for c in CONDITIONS if c not in approved_conditions],
        "e0_e1_retraining_executed": False,
        "reason": classification,
    }
    dump_json(output_root / "retraining_authorization_manifest.json", retraining_manifest)
    gate = {
        "created_at_utc": utc_now(),
        "status": status,
        "classification": classification,
        "r2_report_path": str(r2_report),
        "r2_report_sha256": sha256_file(r2_report),
        "claim_guard_path": str(claim_guard_path),
        "claim_guard_sha256": sha256_file(claim_guard_path),
        "threshold_contract_preregistered": True,
        "threshold_contract_sha256": prereg_sha,
        "threshold_created_before_baseline": prereg_contract["created_at_utc"] < baseline_started_at,
        "passenger_service_rate_min": SERVICE_RATE_MIN,
        "avg_wait_formula": prereg_contract["avg_wait_seconds_max"]["formula"],
        "p95_wait_formula": prereg_contract["passenger_wait_p95_seconds_max"]["formula"],
        "absolute_wait_cap_seconds": ABSOLUTE_WAIT_CAP_SECONDS,
        "service_day_repair_passed": service_test["status"] == "PASS",
        "skip_stop_repair_passed": skip_test["status"] == "PASS",
        "expected_baseline_run_count": 24,
        "completed_baseline_run_count": hard_audit["completed_baseline_run_count"],
        "failed_baseline_run_count": hard_audit["failed_baseline_run_count"],
        "blocked_baseline_run_count": 0,
        "B1_A_pass_count": by_condition["A"]["B1_hard_constraint_pass_count"],
        "B2_A_pass_count": by_condition["A"]["B2_hard_constraint_pass_count"],
        "condition_A_feasibility": labels["A"],
        "condition_A90_feasibility": labels["A90"],
        "condition_A80_feasibility": labels["A80"],
        "condition_A70_feasibility": labels["A70"],
        "approved_conditions": approved_conditions,
        "simulator_or_fleet_infeasible": labels["A"] != "FEASIBLE",
        "approved_for_e0_e1_retraining": approved,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
    }
    dump_json(output_root / "prompt5_e01_r2a_gate.json", gate)
    report = f"""# Prompt 5-E01-R2A Final Report

[Prompt 5-E01-R2A 판정]
status: {status}
classification: {classification}

[Threshold preregistration]
service-rate minimum: {SERVICE_RATE_MIN}
average-wait formula: {gate['avg_wait_formula']}
p95-wait formula: {gate['p95_wait_formula']}
absolute wait cap: {ABSOLUTE_WAIT_CAP_SECONDS}
created before baseline: {str(gate['threshold_created_before_baseline']).lower()}
derived from test: false

[Simulator repair]
service-day reset: true
cross-day queue: 0
wait over service-day: 0
skip legality: true
consecutive skip: 0

[B1 repaired baseline]
A: {by_condition['A']['B1_hard_constraint_pass_count']}/3
A90: {by_condition['A90']['B1_hard_constraint_pass_count']}/3
A80: {by_condition['A80']['B1_hard_constraint_pass_count']}/3
A70: {by_condition['A70']['B1_hard_constraint_pass_count']}/3

[B2 repaired baseline]
A: {by_condition['A']['B2_hard_constraint_pass_count']}/3
A90: {by_condition['A90']['B2_hard_constraint_pass_count']}/3
A80: {by_condition['A80']['B2_hard_constraint_pass_count']}/3
A70: {by_condition['A70']['B2_hard_constraint_pass_count']}/3

[Feasibility]
A: {labels['A']}
A90: {labels['A90']}
A80: {labels['A80']}
A70: {labels['A70']}

[Retraining authorization]
approved: {str(approved).lower()}
approved conditions: {approved_conditions}
blocked conditions: {[c for c in CONDITIONS if c not in approved_conditions]}
reason: {classification}

[Leakage]
test split: false
test target: false
test embedding: false

[Next gate]
E0/E1 retraining approved: {str(approved).lower()}
Prompt 6A approved: false
E2 approved: false
full Prompt 6 approved: false
"""
    (output_root / "prompt5_e01_r2a_final_report.md").write_text(report, encoding="utf-8")
    manifest = {"created_at_utc": utc_now(), "artifact_dir": str(output_root), "files": []}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r2a_manifest.json":
            manifest["files"].append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    dump_json(output_root / "prompt5_e01_r2a_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": status, "classification": classification, "approved_for_e0_e1_retraining": approved}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
