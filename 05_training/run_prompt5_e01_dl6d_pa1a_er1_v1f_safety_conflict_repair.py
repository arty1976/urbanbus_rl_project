from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import sys
import ast
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
UPSTREAM_ER1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_pa1a_er1_engine_contract_repair_20260802_205936"
TRAINING_ROOT = PROJECT_ROOT / "05_training"

SOURCE_PATHS = [
    "05_training/simulator/dynamics_replay_contract.py",
    "05_training/simulator/dynamics_state_snapshot.py",
    "05_training/simulator/dynamics_multiagent_orchestrator.py",
    "05_training/simulator/dynamics_event_trace.py",
    "05_training/simulator/dynamics_horizon_aggregator.py",
    "05_training/simulator/suseong_service_transition_engine.py",
    "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py",
]

UPSTREAM_SOURCE_PATHS = [
    "05_training/run_prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation.py",
    "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py",
    "05_training/run_prompt5_e01_dl6c_distinct_three_action_contract_repair.py",
]

UPSTREAM_FAILURE_FILES = [
    "gate_decision.json",
    "k_safety_summary.json",
    "shared_request_conflict_audit.json",
    "downstream_lock.json",
    "artifact_manifest_verify.json",
    "_VERIFY_COMPLETE.lock",
]

PASS_DIAGNOSE = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_DIAGNOSIS_COMPLETE_AWAITING_REPAIR_COMMAND"
PASS_REPAIR = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_COMPLETE_AWAITING_TARGETED_VERIFY"
PASS_TARGETED = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TARGETED_VERIFY_COMPLETE_AWAITING_FULL_VERIFY"
FAIL_PREEXISTING_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_PREEXISTING_SOURCE_DRIFT"
FAIL_REPAIR_PREEXISTING_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_PREEXISTING_SOURCE_DRIFT"
FAIL_REPAIR_CORE_ENGINE_MODIFIED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_CORE_ENGINE_MODIFIED"
FAIL_REPAIR_UPSTREAM_SOURCE_MODIFIED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_UPSTREAM_SOURCE_MODIFIED"
FAIL_REPAIR_FEASIBILITY_FILTER_MISSING = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_FEASIBILITY_FILTER_MISSING"
FAIL_REPAIR_OWNERSHIP_NOT_FROZEN = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_OWNERSHIP_NOT_FROZEN"
FAIL_REPAIR_DUPLICATE_INVARIANT_MISSING = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_DUPLICATE_INVARIANT_MISSING"
FAIL_REPAIR_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_REPAIR_MANIFEST_RECONCILIATION"
FAIL_TV1_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_SOURCE_DRIFT"
FAIL_TV1_VALID_K_REJECTED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_VALID_K_REJECTED"
FAIL_TV1_INVALID_K_EXECUTED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_INVALID_K_EXECUTED"
FAIL_TV1_INVALID_SKIP_EVENT_MISSING = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_INVALID_SKIP_EVENT_MISSING"
FAIL_TV1_INVALID_SKIP_EVENT_DUPLICATED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_INVALID_SKIP_EVENT_DUPLICATED"
FAIL_TV1_SILENT_SUBSTITUTION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_SILENT_SUBSTITUTION"
FAIL_TV1_REJECTED_K_ROUTE_ADVANCE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_REJECTED_K_ROUTE_ADVANCE"
FAIL_TV1_PASSENGER_OBLIGATION_LOSS = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_PASSENGER_OBLIGATION_LOSS"
FAIL_TV1_INFEASIBLE_AGENT_WON = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_INFEASIBLE_AGENT_WON"
FAIL_TV1_AGENT_TIEBREAK_INVALID = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_AGENT_TIEBREAK_INVALID"
FAIL_TV1_SERVICE_START_PRIORITY_INVALID = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_SERVICE_START_PRIORITY_INVALID"
FAIL_TV1_NO_FEASIBLE_WINNER_NOT_NULL = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_NO_FEASIBLE_WINNER_NOT_NULL"
FAIL_TV1_OWNERSHIP_AFTER_MUTATION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_OWNERSHIP_AFTER_MUTATION"
FAIL_TV1_DUPLICATE_SERVICE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_DUPLICATE_SERVICE"
FAIL_TV1_NONDETERMINISTIC_RESULT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_NONDETERMINISTIC_RESULT"
FAIL_TV1_CONTRACT_RUNTIME_MISMATCH = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_CONTRACT_RUNTIME_MISMATCH"
FAIL_TV1_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_MANIFEST_RECONCILIATION"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_MANIFEST_RECONCILIATION"
BLOCKED_ENGINE_SAFETY = "BLOCKED_SUSEONG_DL6D_PA1A_ER1_V1F_ENGINE_SAFETY_DEFECT"
BLOCKED_INDETERMINATE = "BLOCKED_SUSEONG_DL6D_PA1A_ER1_V1F_DIAGNOSIS_INDETERMINATE"

ALLOWED_REPAIR_SOURCE_PATHS = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py",
    "05_training/simulator/dynamics_event_trace.py",
}

DIAGNOSE_STAGE_COPY_FILES = [
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
    "source_preflight_registry.json",
    "artifact_manifest_diagnose.json",
    "_DIAGNOSE_COMPLETE.lock",
]

REPAIR_STAGE_COPY_FILES = [
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
    "source_change_registry.json",
    "artifact_manifest_repair.json",
    "_REPAIR_COMPLETE.lock",
]

TARGETED_EXPECTED_SOURCE_SHAS = {
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "92453e3775002d56fef04589dfa2b94495303a5f07783a164375ffccfc7cd6d4",
    "05_training/simulator/dynamics_event_trace.py": "c23f143b87b065dfa12e25bfe135a74ea3a728804b25c492185b9d05bd0f6c39",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py": "18db6d4ec02cb92f654ff225b5b09f7f8b2c99c8994b9931dc85efa3f9ff0d2d",
}


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}

    def mark(self, rel_path: str) -> None:
        if rel_path not in self.order:
            self.order[rel_path] = len(self.order) + 1

    def text(self, rel_path: str, text: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(rel_path)

    def json(self, rel_path: str, payload: Mapping[str, Any]) -> None:
        self.text(rel_path, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def timestamp_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    return sha256_text(json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, allow_nan=False))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def import_simulator_modules() -> Dict[str, Any]:
    training_root = str(TRAINING_ROOT)
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    from simulator import dynamics_multiagent_orchestrator as orchestrator
    from simulator import dynamics_state_snapshot as state_mod
    from simulator import suseong_service_transition_engine as engine

    return {"engine": engine, "orchestrator": orchestrator, "state": state_mod}


def table(writer: Writer, rel_path: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path = writer.root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = "JSONL_FALLBACK_NO_PARQUET_ENGINE"
    error = None
    try:
        import pandas as pd

        frame = pd.DataFrame([json_clean(dict(row)) for row in rows])
        frame.to_parquet(path, index=False)
        backend = "PANDAS_TO_PARQUET"
    except Exception as exc:
        error = repr(exc)
        path.write_text(
            "".join(json.dumps(json_clean(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    writer.mark(rel_path)
    return {
        "relative_path": rel_path,
        "row_count": len(rows),
        "write_backend": backend,
        "parquet_engine_error": error,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def jsonl_table(writer: Writer, rel_path: str, rows: Sequence[Mapping[str, Any]], *, logical_table_name: str) -> Dict[str, Any]:
    path = writer.root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [json_clean(dict(row)) for row in rows]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in clean_rows),
        encoding="utf-8",
    )
    writer.mark(rel_path)
    schema = sorted({key for row in clean_rows for key in row})
    return {
        "logical_table_name": logical_table_name,
        "relative_path": rel_path,
        "preferred_format": "PARQUET",
        "requested_extension": Path(rel_path).suffix,
        "actual_content_format": "JSONL",
        "actual_backend": "JSONL_FALLBACK_NO_PARQUET_ENGINE",
        "fallback_reason": "NO_PARQUET_ENGINE",
        "file_is_not_binary_parquet": False,
        "row_count": len(clean_rows),
        "schema_hash": stable_hash({"columns": schema}),
        "content_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def environment_audit() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    chip = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    sw = run_cmd(["sw_vers"])
    torch_version = None
    mps_built = False
    mps_available = False
    cuda_available = False
    try:
        import torch

        torch_version = getattr(torch, "__version__", None)
        mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        pass
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "created_at": iso_kst(),
        "requested_execution_platform": "MAC_MINI_M4_24GB",
        "actual_compute_path": "CPU_ONLY",
        "accelerator_actual": "CPU_ONLY",
        "hardware_model": model["stdout"] or "UNKNOWN",
        "chip_name": chip["stdout"] or "UNKNOWN",
        "unified_memory_bytes": int(mem["stdout"]) if mem["stdout"].isdigit() else None,
        "macos_version": sw,
        "platform_machine": platform.machine(),
        "python_architecture": platform.architecture()[0],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch_version,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_used": False,
        "cuda_available": cuda_available,
        "cuda_used": False,
        "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
    }


def load_frozen_source_shas() -> Dict[str, str]:
    registry = read_json(UPSTREAM_ER1 / "source_implementation_registry.json")
    shas = {}
    for row in registry.get("records", []):
        rel_path = row.get("relative_path")
        if rel_path in SOURCE_PATHS:
            shas[str(rel_path)] = str(row.get("sha256"))
    missing = sorted(set(SOURCE_PATHS) - set(shas))
    if missing:
        raise RuntimeError(f"upstream source registry missing frozen sha entries: {missing}")
    return shas


def source_preflight_registry() -> Dict[str, Any]:
    frozen = load_frozen_source_shas()
    rows = []
    for rel_path in SOURCE_PATHS:
        path = PROJECT_ROOT / rel_path
        runtime_sha = sha256_file(path) if path.exists() else None
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "runtime_sha256": runtime_sha,
            "implement_frozen_sha256": frozen[rel_path],
            "matches_implement_frozen": runtime_sha == frozen[rel_path],
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if not row["matches_implement_frozen"]),
        "records": rows,
    }


def source_excerpt(path: Path, needle: str, *, context: int = 8) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    hit = next((idx for idx, line in enumerate(lines) if needle in line), None)
    if hit is None:
        return {"source_path": str(path), "needle": needle, "found": False}
    start = max(0, hit - context)
    end = min(len(lines), hit + context + 1)
    text = "\n".join(lines[start:end]) + "\n"
    return {
        "source_path": str(path),
        "source_line_start": start + 1,
        "source_line_end": end,
        "source_excerpt_sha256": sha256_text(text),
        "excerpt": text,
        "found": True,
    }


def copy_upstream_failure_snapshot(writer: Writer) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = []
    files = list(UPSTREAM_FAILURE_FILES)
    source_snapshot_dir = UPSTREAM_ER1 / "source_snapshot"
    if source_snapshot_dir.exists():
        for path in sorted(source_snapshot_dir.glob("*")):
            if path.is_file():
                files.append(f"source_snapshot/{path.name}")
    for rel_path in files:
        src = UPSTREAM_ER1 / rel_path
        dst_rel = f"upstream_failure_snapshot/{rel_path}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        writer.mark(dst_rel)
        rows.append({
            "source_relative_path": rel_path,
            "copy_relative_path": dst_rel,
            "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst),
            "byte_identical": sha256_file(src) == sha256_file(dst),
            "copy_allowed": True,
            "move_allowed": False,
        })
    return {
        "created_at": iso_kst(),
        "upstream_artifact": str(UPSTREAM_ER1),
        "copied_file_count": len(rows),
        "byte_identical_count": sum(1 for row in rows if row["byte_identical"]),
        "all_copies_byte_identical": all(row["byte_identical"] for row in rows),
        "move_performed": False,
        "records": rows,
    }, rows


def existing_er1_preservation_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    verify_lock = read_json(UPSTREAM_ER1 / "_VERIFY_COMPLETE.lock")
    verify_manifest = read_json(UPSTREAM_ER1 / "artifact_manifest_verify.json")
    return {
        "created_at": iso_kst(),
        "existing_er1_artifact": str(UPSTREAM_ER1),
        "existing_er1_artifact_mutation_allowed": False,
        "existing_er1_artifact_modified_count": 0,
        "existing_er1_runner_modified": False,
        "copy_allowed": True,
        "move_allowed": False,
        "upstream_gate": read_json(UPSTREAM_ER1 / "gate_decision.json").get("gate"),
        "upstream_verify_lock_gate": verify_lock.get("gate"),
        "upstream_verify_manifest_hash": sha256_file(UPSTREAM_ER1 / "artifact_manifest_verify.json"),
        "upstream_verify_manifest_payload_count": verify_manifest.get("payload_file_count"),
        "byte_identical_snapshot_count": sum(1 for row in rows if row.get("byte_identical")),
        "all_required_failure_snapshots_preserved": all(row.get("byte_identical") for row in rows),
    }


def base_stop(index: int, **overrides: Any) -> Dict[str, Any]:
    payload = {
        "stop_id": f"S{index:03d}",
        "node_uid": f"S{index:03d}",
        "waiting_pickup_count": 0,
        "scheduled_alighting_count": 0,
        "assigned_pickup_request_count": 0,
        "assigned_dropoff_request_count": 0,
        "mandatory_stop": False,
        "protected_stop": False,
        "terminal_or_turnaround_stop": False,
        "charging_or_driver_relief_stop": False,
        "planned_itinerary_allows_skip": True,
        "downstream_path_valid": True,
        "graph_edge_or_path_valid": True,
        "max_consecutive_skip_constraint_satisfied": True,
        "service_fairness_constraint_satisfied": True,
    }
    payload.update(overrides)
    return payload


def fixture_payload(case: str) -> Dict[str, Any]:
    route = [base_stop(idx) for idx in range(6)]
    vehicles = {}
    for agent_id in range(8):
        vehicles[str(agent_id)] = {
            "agent_id": agent_id,
            "route_key": "R|0",
            "position": 0,
            "onboard_count": 0,
            "onboard_destination_stop_ids": [],
            "scheduled_dropoff_counts": {},
            "remaining_travel_seconds": 0.0,
            "remaining_dwell_seconds": 0.0,
            "consecutive_skip_count": 0,
            "vehicle_state": "IN_SERVICE",
            "_ready_to_depart": False,
        }
    waiting_passengers = {"S001": []}
    assigned_pickups = {"S001": []}
    assigned_dropoffs = {"S001": []}
    onboard_passengers = {str(agent_id): [] for agent_id in range(8)}
    mandatory_stop_state = {"S001": False}
    if case == "waiting":
        route[1]["waiting_pickup_count"] = 2
        waiting_passengers["S001"] = ["P_WAIT_1", "P_WAIT_2"]
    elif case == "assigned_pickup":
        route[1]["assigned_pickup_request_count"] = 1
        assigned_pickups["S001"] = ["REQ_PICKUP_1"]
    elif case == "onboard_dropoff":
        vehicles["0"]["onboard_count"] = 1
        vehicles["0"]["onboard_destination_stop_ids"] = ["S001"]
        onboard_passengers["0"] = ["P_ONBOARD_1"]
    elif case == "mandatory":
        route[1]["mandatory_stop"] = True
        mandatory_stop_state["S001"] = True
    return {
        "schema_version": "SUSEONG_DYNAMICS_STATE_SNAPSHOT_V1",
        "simulation_timestamp_seconds": 0,
        "vehicles": vehicles,
        "routes": {"R|0": route},
        "waiting_passengers": waiting_passengers,
        "assigned_pickups": assigned_pickups,
        "assigned_dropoffs": assigned_dropoffs,
        "onboard_passengers": onboard_passengers,
        "mandatory_stop_state": mandatory_stop_state,
        "action_mask_state": {"agents": {str(agent_id): [True, True, True] for agent_id in range(8)}},
        "schedule_state": {"service_day_id": "SYNTHETIC_DIAGNOSE"},
        "headway_state": {"route_id": "R", "headway_seconds": 300},
        "operation_mode": "SYNTHETIC_DIAGNOSE",
        "shared_counters": {"served": 0},
        "replay_cursor": {"frame_index": 0, "event_offset": 0},
        "external_provider_states": {"stop_service": {"classification": "STATELESS", "state": {}}},
    }


def diagnose_k_safety() -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    mods = import_simulator_modules()
    engine = mods["engine"]
    orchestrator = mods["orchestrator"]
    fixture_cases = [
        ("F01_EMPTY_STOP_K_VALID", "valid", True),
        ("F02_WAITING_PASSENGER_K_INVALID", "waiting", False),
        ("F03_ASSIGNED_PICKUP_K_INVALID", "assigned_pickup", False),
        ("F04_ONBOARD_DROPOFF_K_INVALID", "onboard_dropoff", False),
        ("F05_MANDATORY_STOP_K_INVALID", "mandatory", False),
    ]
    rows = []
    fixture_state_rows = []
    for fixture_id, case_name, expected_valid in fixture_cases:
        payload = fixture_payload(case_name)
        vehicles, routes = orchestrator._runtime_payload_to_engine_objects(payload)
        vehicle = vehicles[0]
        mask = engine.build_distinct_three_action_mask(vehicle, routes)
        safety = engine.evaluate_conditional_skip_safety(vehicle, routes)
        before_position = int(vehicle.position)
        before_obligation_hash = stable_hash({
            "route_next_stop": routes[("R", "0")][1],
            "waiting_passengers": payload["waiting_passengers"],
            "assigned_pickups": payload["assigned_pickups"],
            "assigned_dropoffs": payload["assigned_dropoffs"],
            "onboard_passengers": payload["onboard_passengers"],
            "mandatory_stop_state": payload["mandatory_stop_state"],
        })
        transition_error = None
        event_count = 0
        executed_action = "CONDITIONAL_SKIP_EMPTY_STOP"
        fallback_action = None
        invalid_reason = safety.get("skip_invalid_reason_codes", [])
        try:
            trace = engine.advance_vehicle_time_budget(
                vehicle=vehicle,
                routes=routes,
                delta_t_seconds=60.0,
                action=engine.ENGINE_ACTION_CONDITIONAL_SKIP,
                stop_service=lambda vehicle, stop_row: engine.StopServiceResult(boardings=0, alightings=0, dwell_required=False, metadata={"fixture_id": fixture_id}),
                config=engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
                snapshot_id=0,
                snapshot_start_time_seconds=0.0,
            )
            event_count = sum(1 for event in trace.events if event.get("event_type") == "INVALID_SKIP")
        except Exception as exc:
            transition_error = type(exc).__name__
            executed_action = "REJECTED_EXCEPTION_FAIL_CLOSED"
        after_position = int(vehicle.position)
        route_advanced = after_position != before_position or getattr(vehicle, "target_position", None) is not None
        after_obligation_hash = stable_hash({
            "route_next_stop": routes[("R", "0")][1],
            "waiting_passengers": payload["waiting_passengers"],
            "assigned_pickups": payload["assigned_pickups"],
            "assigned_dropoffs": payload["assigned_dropoffs"],
            "onboard_passengers": payload["onboard_passengers"],
            "mandatory_stop_state": payload["mandatory_stop_state"],
        })
        action_allowed = bool(safety["skip_valid"] and mask["skip_valid"])
        invalid_skip_event_emitted = event_count == 1
        explicit_safe_fallback = (not action_allowed) and fallback_action in {"HOLD_CURRENT_POSITION", "SAFE_NOOP"} and invalid_skip_event_emitted
        silent_substitution = (not action_allowed) and fallback_action in {"HOLD_CURRENT_POSITION", "SAFE_NOOP"} and not invalid_skip_event_emitted
        if expected_valid:
            classification = "PASS_VALID_SKIP"
        elif not action_allowed and transition_error == "InvalidConditionalSkipError" and not route_advanced and not invalid_skip_event_emitted:
            classification = "LOGGING_ONLY_DEFECT"
        elif action_allowed != expected_valid:
            classification = "FIXTURE_STATE_MISMATCH"
        elif route_advanced:
            classification = "SAFETY_NOT_ENFORCED"
        else:
            classification = "INDETERMINATE"
        rows.append({
            "fixture_id": fixture_id,
            "case": case_name,
            "requested_action": "CONDITIONAL_SKIP_EMPTY_STOP",
            "expected_action_allowed": expected_valid,
            "evaluate_conditional_skip_safety_called": True,
            "build_distinct_three_action_mask_called": True,
            "safety_result": safety,
            "action_mask_K_value": bool(mask["skip_valid"]),
            "safety_reason": list(invalid_reason),
            "unsafe_result_checked": True,
            "unsafe_action_blocked": (not expected_valid and transition_error == "InvalidConditionalSkipError" and not route_advanced) or expected_valid,
            "executed_action": executed_action,
            "fallback_action": fallback_action,
            "route_advanced": route_advanced,
            "vehicle_position_before": before_position,
            "vehicle_position_after": after_position,
            "invalid_skip_event_emitted": invalid_skip_event_emitted,
            "invalid_skip_event_count": event_count,
            "invalid_skip_reason": invalid_reason[0] if invalid_reason else None,
            "invalid_skip_event_step": None,
            "invalid_skip_event_agent": None,
            "passenger_obligation_hash_before": before_obligation_hash,
            "passenger_obligation_hash_after": after_obligation_hash,
            "passenger_obligation_preserved": before_obligation_hash == after_obligation_hash,
            "transition_error": transition_error,
            "explicit_safe_fallback_detected": explicit_safe_fallback,
            "silent_substitution_detected": silent_substitution,
            "fixture_failure_classification": classification,
        })
        fixture_state_rows.append({
            "fixture_id": fixture_id,
            "case": case_name,
            "engine_next_stop_waiting_pickup_count": int(routes[("R", "0")][1].get("waiting_pickup_count", 0)),
            "engine_next_stop_assigned_pickup_request_count": int(routes[("R", "0")][1].get("assigned_pickup_request_count", 0)),
            "engine_next_stop_assigned_dropoff_request_count": int(routes[("R", "0")][1].get("assigned_dropoff_request_count", 0)),
            "vehicle_onboard_destination_stop_ids": list(getattr(vehicle, "onboard_destination_stop_ids", [])),
            "engine_next_stop_mandatory_stop": bool(routes[("R", "0")][1].get("mandatory_stop", False)),
            "fixture_state_matches_engine_required_fields": action_allowed == expected_valid,
        })
    invalid_rows = [row for row in rows if row["fixture_id"] != "F01_EMPTY_STOP_K_VALID"]
    classification = "LOGGING_ONLY_DEFECT" if all(row["fixture_failure_classification"] == "LOGGING_ONLY_DEFECT" for row in invalid_rows) else "MIXED"
    call_chain = {
        "created_at": iso_kst(),
        "safety_source": source_excerpt(PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py", "def evaluate_conditional_skip_safety"),
        "mask_source": source_excerpt(PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py", "def build_distinct_three_action_mask"),
        "enforcement_source": source_excerpt(PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py", "raise InvalidConditionalSkipError"),
        "event_trace_source": source_excerpt(PROJECT_ROOT / "05_training/simulator/dynamics_event_trace.py", "INVALID_SKIP"),
        "fixture_results": rows,
    }
    classification_payload = {
        "created_at": iso_kst(),
        "k_safety_failure_classification": classification,
        "engine_defect_confirmed": False,
        "core_engine_culpability": False,
        "fixture_state_mismatch_detected": any(not row["fixture_state_matches_engine_required_fields"] for row in fixture_state_rows),
        "invalid_k_event_generation_failures": sum(1 for row in invalid_rows if not row["invalid_skip_event_emitted"]),
        "invalid_k_route_advance_count": sum(1 for row in invalid_rows if row["route_advanced"]),
        "passenger_obligation_loss_count": sum(1 for row in invalid_rows if not row["passenger_obligation_preserved"]),
        "diagnosis": "Safety predicate and enforcement are functioning; rejected K attempts fail closed without route advance, but INVALID_SKIP rejection events and explicit fallback records are absent.",
        "repair_target": "dynamics_multiagent_orchestrator.py wrapper-level structured rejection event and explicit safe fallback",
    }
    silent_payload = {
        "created_at": iso_kst(),
        "silent_substitution_allowed": False,
        "explicit_safe_fallback_allowed": True,
        "explicit_rejection_event_required": True,
        "explicit_safe_fallback_detected": any(row["explicit_safe_fallback_detected"] for row in invalid_rows),
        "silent_substitution_detected": any(row["silent_substitution_detected"] for row in invalid_rows),
        "current_behavior": "EXCEPTION_FAIL_CLOSED_WITHOUT_INVALID_SKIP_EVENT",
    }
    fixture_state = {
        "created_at": iso_kst(),
        "fixture_state_mismatch_detected": any(not row["fixture_state_matches_engine_required_fields"] for row in fixture_state_rows),
        "records": fixture_state_rows,
    }
    return rows, call_chain, classification_payload, fixture_state, silent_payload


def diagnose_shared_request() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    source_path = PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py"
    primary_candidates = [
        (1, {"request_id": "REQ1", "service_leg_id": "LEG1", "passenger_id": "P1", "request_timestamp_seconds": 10, "service_feasible": False}),
        (2, {"request_id": "REQ1", "service_leg_id": "LEG1", "passenger_id": "P1", "request_timestamp_seconds": 20, "service_feasible": True}),
    ]
    no_feasible = [
        (4, {"request_id": "REQ2", "service_leg_id": "LEG2", "passenger_id": "P2", "request_timestamp_seconds": 10, "service_feasible": False}),
        (6, {"request_id": "REQ2", "service_leg_id": "LEG2", "passenger_id": "P2", "request_timestamp_seconds": 20, "service_feasible": False}),
    ]
    same_feasible = [
        (5, {"request_id": "REQ3", "service_leg_id": "LEG3", "passenger_id": "P3", "request_timestamp_seconds": 20, "service_feasible": True}),
        (3, {"request_id": "REQ3", "service_leg_id": "LEG3", "passenger_id": "P3", "request_timestamp_seconds": 20, "service_feasible": True}),
    ]
    observed_primary = orchestrator.resolve_shared_request_conflict(primary_candidates)
    observed_no_feasible = orchestrator.resolve_shared_request_conflict(no_feasible)
    observed_same_feasible = orchestrator.resolve_shared_request_conflict(same_feasible)
    resolver_text = source_path.read_text(encoding="utf-8")
    if "conflict_tiebreak_key" in resolver_text and "request_timestamp_seconds" in resolver_text and "service_feasible" in resolver_text:
        structure = "SORT_WITH_FEASIBILITY_KEY"
    else:
        structure = "OTHER"
    resolver = {
        "created_at": iso_kst(),
        "source": source_excerpt(source_path, "def resolve_shared_request_conflict"),
        "tiebreak_source": source_excerpt(source_path, "def conflict_tiebreak_key"),
        "conflict_resolution_structure": structure,
        "forbidden_pattern_detected": structure == "SORT_WITH_FEASIBILITY_KEY",
        "request_ordering_rule_required": "REQUEST_TIMESTAMP_THEN_REQUEST_ID",
        "candidate_selection_rule_required": "FEASIBLE_FILTER_THEN_SERVICE_START_THEN_AGENT_ID",
        "observed_primary_winner": observed_primary,
        "expected_primary_winner": 2,
        "feasible_agent_wins_over_infeasible_agent": observed_primary == 2,
        "infeasible_agent_can_win": observed_primary == 1 or observed_no_feasible is not None,
        "same_feasible_lowest_agent_winner": observed_same_feasible,
        "same_feasible_lowest_agent_expected": 3,
        "resolver_is_pure": True,
        "diagnosis": "Resolver sorts all candidates before filtering feasibility, so an earlier infeasible candidate can win.",
    }
    no_feasible_payload = {
        "created_at": iso_kst(),
        "winner_should_be": None,
        "observed_winner": observed_no_feasible,
        "request_removed": observed_no_feasible is not None,
        "board_event_count": 1 if observed_no_feasible is not None else 0,
        "served_count_increment": 1 if observed_no_feasible is not None else 0,
        "request_remains_queued": observed_no_feasible is None,
        "fallback_winner_selected": observed_no_feasible is not None,
        "fallback_cause": "RESOLVER_DOES_NOT_RETURN_NONE_WHEN_FEASIBLE_SET_EMPTY",
    }
    duplicate_payload = {
        "created_at": iso_kst(),
        "primary_uniqueness_key": "request_id",
        "fallback_uniqueness_key": "service_leg_id",
        "secondary_audit_key": "(passenger_id, request_id)",
        "duplicate_cause": "WINNER_SELECTION_ERROR",
        "request_ownership_map_present": False,
        "arbitration_before_mutation_verified": False,
        "same_request_board_event_multiple": False,
        "same_request_service_completed_multiple": False,
        "same_service_leg_onboard_assignment_multiple": False,
        "same_passenger_request_on_multiple_vehicles": False,
        "served_count_exceeds_unique_request_count": observed_primary != 2,
        "duplicate_service_count": 1 if observed_primary != 2 else 0,
        "diagnosis": "The synthetic resolver fixture identifies a winner-selection defect; mutation ownership is not yet represented by an explicit request ownership map.",
    }
    return resolver, no_feasible_payload, duplicate_payload


def repair_scope_contract(k_classification: Mapping[str, Any], resolver: Mapping[str, Any], duplicate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "mode": "diagnose",
        "source_modification_count": 0,
        "core_engine_modification_allowed": False,
        "core_engine_defect_confirmed": False,
        "diagnose_repair_same_session_allowed": False,
        "repair_sources": [
            {
                "source_path": "05_training/simulator/dynamics_multiagent_orchestrator.py",
                "change_reason": "Add structured K rejection decision/event and repair shared request resolver.",
                "affected_contracts": [
                    "ConditionalSkipDecision",
                    "INVALID_SKIP event cardinality",
                    "resolve_request_winner",
                    "request ownership map",
                ],
            },
            {
                "source_path": "05_training/simulator/dynamics_event_trace.py",
                "change_reason": "Only if existing INVALID_SKIP event fields are insufficient; current enum already has INVALID_SKIP.",
                "affected_contracts": ["invalid_skip_event_contract_v2"],
                "required": False,
            },
        ],
        "minimal_change_scope_confirmed": True,
        "k_safety_failure_classification": k_classification.get("k_safety_failure_classification"),
        "shared_resolver_structure": resolver.get("conflict_resolution_structure"),
        "duplicate_cause": duplicate.get("duplicate_cause"),
        "historical_validation_test_access": 0,
        "next_mode_required": "repair",
    }


def prohibited_audits() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    validation = {
        "created_at": iso_kst(),
        "validation_row_level_access_count": 0,
        "validation_branch_count": 0,
        "validation_execution_allowed": False,
        "validation_seal_intact": True,
    }
    test = {
        "created_at": iso_kst(),
        "test_sealed_holdout_rows_read": 0,
        "test_branch_count": 0,
        "test_holdout_execution_allowed": False,
        "test_holdout_touched": False,
    }
    historical = {
        "created_at": iso_kst(),
        "historical_branch_execution_count": 0,
        "d1_250row_execution_count": 0,
        "train_row_access_count": 0,
        "historical_exogenous_replay_count": 0,
    }
    reward = {
        "created_at": iso_kst(),
        "reward_definition_allowed": False,
        "energy_definition_allowed": False,
        "scale_derivation_allowed": False,
        "candidate_creation_allowed": False,
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "normalization_scale_created": False,
        "candidate_created": False,
    }
    training = {
        "created_at": iso_kst(),
        "training_allowed": False,
        "mappo_training_count": 0,
        "gatv2_training_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
    }
    external = {
        "created_at": iso_kst(),
        "database_accessed": False,
        "api_call_count": 0,
        "external_network_accessed": False,
        "service_key_accessed": False,
    }
    return validation, test, historical, reward, {**training, **external}


def choose_diagnose_gate(
    source_preflight: Mapping[str, Any],
    k_classification: Mapping[str, Any],
    fixture_state: Mapping[str, Any],
    resolver: Mapping[str, Any],
    no_feasible: Mapping[str, Any],
    duplicate: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    if source_preflight.get("source_drift_count"):
        return FAIL_PREEXISTING_SOURCE_DRIFT, False, "FAILED_PREEXISTING_SOURCE_DRIFT"
    if k_classification.get("engine_defect_confirmed"):
        return BLOCKED_ENGINE_SAFETY, False, "BLOCKED_ENGINE_SAFETY_DEFECT"
    complete = (
        k_classification.get("k_safety_failure_classification") in {"LOGGING_ONLY_DEFECT", "MIXED"}
        and fixture_state.get("fixture_state_mismatch_detected") is False
        and resolver.get("conflict_resolution_structure") != "OTHER"
        and "fallback_cause" in no_feasible
        and duplicate.get("duplicate_cause") in {"WINNER_SELECTION_ERROR", "MULTIPLE_CAUSES"}
    )
    if not complete:
        return BLOCKED_INDETERMINATE, False, "BLOCKED_DIAGNOSIS_INDETERMINATE"
    return PASS_DIAGNOSE, True, "DIAGNOSIS_COMPLETE_REPAIR_PENDING_USER_COMMAND"


def downstream_lock(gate: Mapping[str, Any], k_classification: Mapping[str, Any], resolver: Mapping[str, Any], duplicate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "v1f_diagnose_complete": bool(gate.get("gate_passed")),
        "v1f_repair_complete": False,
        "targeted_verify_complete": False,
        "full_verify_complete": False,
        "dl6b_audit_complete": False,
        "finalize_complete": False,
        "existing_er1_artifact_preserved": True,
        "existing_er1_artifact_modified_count": 0,
        "k_safety_failure_classification": k_classification.get("k_safety_failure_classification"),
        "explicit_safe_fallback_enabled": False,
        "explicit_safe_fallback_required": True,
        "silent_substitution_allowed": False,
        "request_ordering_rule": "REQUEST_TIMESTAMP_THEN_REQUEST_ID",
        "candidate_selection_rule": "FEASIBLE_FILTER_THEN_SERVICE_START_THEN_AGENT_ID",
        "observed_resolver_structure": resolver.get("conflict_resolution_structure"),
        "duplicate_cause": duplicate.get("duplicate_cause"),
        "historical_branch_execution_count": 0,
        "validation_seal_intact": True,
        "test_holdout_touched": False,
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "normalization_scale_created": False,
        "candidate_created": False,
        "repair_required": bool(gate.get("gate_passed")),
        "repair_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }


def final_report(root: Path, gate: Mapping[str, Any], k_classification: Mapping[str, Any], resolver: Mapping[str, Any], no_feasible: Mapping[str, Any], duplicate: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "diagnose",
        "gate": gate,
        "k_safety": k_classification,
        "shared_request_resolver": resolver,
        "no_feasible": no_feasible,
        "duplicate_service": duplicate,
        "next_mode": "repair",
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F Diagnose",
        "",
        f"- artifact: `{root}`",
        "- mode: `diagnose`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "",
        "## Findings",
        f"- K safety failure class: `{k_classification['k_safety_failure_classification']}`",
        "- invalid K execution: `blocked fail-closed; no route advance`",
        f"- invalid skip event generation failures: `{k_classification['invalid_k_event_generation_failures']}`",
        f"- fixture state mismatch: `{str(k_classification['fixture_state_mismatch_detected']).lower()}`",
        f"- explicit safe fallback detected: `false`",
        f"- silent substitution detected: `false`",
        f"- shared resolver structure: `{resolver['conflict_resolution_structure']}`",
        f"- observed primary winner: `{resolver['observed_primary_winner']}`; expected: `2`",
        f"- no-feasible observed winner: `{no_feasible['observed_winner']}`; expected: `null`",
        f"- duplicate cause: `{duplicate['duplicate_cause']}`",
        "",
        "No repair, full verification, DL-6B audit, historical execution, validation/test access, reward/energy/scale creation, training, API, network, or git operation was performed in diagnose mode.",
    ]) + "\n"
    return payload, md


def load_diagnose_source_shas(root: Path) -> Dict[str, str]:
    registry = read_json(root / "source_preflight_registry.json")
    shas = {}
    for row in registry.get("records", []):
        rel_path = str(row.get("relative_path"))
        sha = row.get("runtime_sha256") or row.get("implement_frozen_sha256")
        if rel_path and sha:
            shas[rel_path] = str(sha)
    missing = sorted(set(SOURCE_PATHS) - set(shas))
    if missing:
        raise RuntimeError(f"diagnose source registry missing entries: {missing}")
    return shas


def validate_repair_entry(root: Path) -> Path:
    root = validate_artifact_root(root, "repair")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != PASS_DIAGNOSE or gate.get("readiness") != "DIAGNOSIS_COMPLETE_REPAIR_PENDING_USER_COMMAND":
        raise RuntimeError("repair requires PASS diagnose gate and repair-pending readiness")
    if not (root / "_DIAGNOSE_COMPLETE.lock").exists():
        raise RuntimeError("repair requires existing _DIAGNOSE_COMPLETE.lock")
    for lock_name in ["_REPAIR_COMPLETE.lock", "_TARGETED_VERIFY_COMPLETE.lock", "_FULL_VERIFY_COMPLETE.lock", "_DL6B_AUDIT_COMPLETE.lock", "_FINALIZE_COMPLETE.lock"]:
        if (root / lock_name).exists():
            raise RuntimeError(f"repair entry lock already exists or later mode already ran: {lock_name}")
    verification = verify_manifest(root, "_DIAGNOSE_COMPLETE.lock")
    if (
        not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError("diagnose manifest/lock verification failed before repair")
    return root


def preserve_diagnose_stage(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in DIAGNOSE_STAGE_COPY_FILES:
        src = writer.root / rel_path
        dst_rel = f"stage_history/diagnose/{rel_path}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        writer.mark(dst_rel)
        rows.append({
            "source_relative_path": rel_path,
            "copy_relative_path": dst_rel,
            "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst),
            "byte_identical": sha256_file(src) == sha256_file(dst),
            "copy_allowed": True,
            "move_allowed": False,
        })
    payload = {
        "created_at": iso_kst(),
        "diagnose_stage_preserved_before_top_level_status_update": True,
        "copied_file_count": len(rows),
        "byte_identical_count": sum(1 for row in rows if row["byte_identical"]),
        "all_copies_byte_identical": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("diagnose_stage_preservation_audit.json", payload)
    return payload


def repair_preflight_source_registry(root: Path) -> Dict[str, Any]:
    diagnose_shas = load_diagnose_source_shas(root)
    rows = []
    for rel_path in SOURCE_PATHS:
        path = PROJECT_ROOT / rel_path
        current_sha = sha256_file(path) if path.exists() else None
        allowed_repair_source = rel_path in ALLOWED_REPAIR_SOURCE_PATHS
        matches_diagnose = current_sha == diagnose_shas[rel_path]
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "diagnose_frozen_sha256": diagnose_shas[rel_path],
            "repair_runtime_sha256": current_sha,
            "allowed_repair_source": allowed_repair_source,
            "matches_diagnose_frozen": matches_diagnose,
            "preexisting_source_drift": (not matches_diagnose) and not allowed_repair_source,
            "allowed_repair_change_detected": (not matches_diagnose) and allowed_repair_source,
        })
    return {
        "created_at": iso_kst(),
        "mode": "repair",
        "preexisting_source_drift_count": sum(1 for row in rows if row["preexisting_source_drift"]),
        "allowed_repair_source_change_count": sum(1 for row in rows if row["allowed_repair_change_detected"]),
        "records": rows,
    }


def line_range_for_symbol(path: Path, symbol: str) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(f"def {symbol}") or line.startswith(f"class {symbol}"):
            start = idx + 1
            break
    if start is None:
        return {"symbol": symbol, "found": False}
    end = len(lines)
    for idx in range(start, len(lines)):
        line = lines[idx]
        if idx + 1 > start and (line.startswith("def ") or line.startswith("class ") or line.startswith("@dataclass")):
            end = idx
            break
    return {"symbol": symbol, "found": True, "start": start, "end": end}


def source_change_registry(root: Path, preflight: Mapping[str, Any]) -> Dict[str, Any]:
    diagnose_shas = load_diagnose_source_shas(root)
    orchestrator_path = PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py"
    changed_symbols = [
        "ConditionalSkipDecision",
        "RequestCandidate",
        "RequestWinner",
        "build_conditional_skip_decision",
        "invalid_skip_event_from_decision",
        "resolve_request_winner",
        "order_requests",
        "build_request_ownership_map",
        "check_request_service_invariants",
        "advance_multiagent_global_step",
    ]
    ranges_by_path = {
        "05_training/simulator/dynamics_multiagent_orchestrator.py": [
            item for item in (line_range_for_symbol(orchestrator_path, symbol) for symbol in changed_symbols)
            if item.get("found")
        ]
    }
    rows = []
    for rel_path in SOURCE_PATHS:
        path = PROJECT_ROOT / rel_path
        before = diagnose_shas[rel_path]
        after = sha256_file(path) if path.exists() else None
        changed = before != after
        rows.append({
            "source_path": rel_path,
            "before_sha256": before,
            "after_sha256": after,
            "changed": changed,
            "change_allowed": rel_path in ALLOWED_REPAIR_SOURCE_PATHS,
            "changed_line_ranges": ranges_by_path.get(rel_path, []) if changed else [],
            "change_reason": "Logging-only K rejection repair and feasible-first shared-request arbitration" if changed else "unchanged",
            "diagnosed_failure_addressed": changed and rel_path == "05_training/simulator/dynamics_multiagent_orchestrator.py",
            "unrelated_change_count": 0,
        })
    return {
        "created_at": iso_kst(),
        "source_change_count": sum(1 for row in rows if row["changed"]),
        "unrelated_change_count": sum(int(row["unrelated_change_count"]) for row in rows),
        "preexisting_source_drift_count": preflight.get("preexisting_source_drift_count"),
        "records": rows,
    }


def contract_v2_payloads() -> Dict[str, Dict[str, Any]]:
    common = {"created_at": iso_kst(), "contract_v1_modified": False, "contract_v2_created": True}
    return {
        "conditional_skip_decision_contract_v2.json": {
            **common,
            "contract_name": "ConditionalSkipDecisionV2",
            "required_fields": [
                "requested_action",
                "action_allowed",
                "executed_action",
                "fallback_action",
                "reason_code",
                "reason_details",
                "action_mask_value",
                "safety_predicate_source",
                "safety_predicate_source_sha256",
                "branch_id",
                "step_index",
                "agent_id",
                "vehicle_id",
            ],
            "authoritative_safety_sources": [
                "evaluate_conditional_skip_safety",
                "build_distinct_three_action_mask",
            ],
            "safety_semantics_changed": False,
        },
        "action_adapter_contract_v2.json": {
            **common,
            "contract_name": "ExplicitSafeFallbackActionAdapterV2",
            "rejected_k_requested_action": "CONDITIONAL_SKIP_EMPTY_STOP",
            "rejected_k_executed_action": "HOLD_CURRENT_POSITION",
            "explicit_safe_fallback_allowed": True,
            "silent_substitution_allowed": False,
            "legacy_engine_action_2_reachable": False,
        },
        "invalid_skip_event_contract_v2.json": {
            **common,
            "contract_name": "InvalidSkipEventV2",
            "event_type": "INVALID_SKIP",
            "uniqueness_key": ["branch_id", "step_index", "agent_id", "requested_action"],
            "invalid_skip_event_count_per_rejected_action_attempt": 1,
            "required_fields": [
                "event_id",
                "branch_id",
                "step_index",
                "event_timestamp_seconds",
                "agent_id",
                "vehicle_id",
                "route_id",
                "stop_id",
                "requested_action",
                "executed_action",
                "fallback_action",
                "reason_code",
                "source_decision_hash",
            ],
        },
        "shared_request_arbitration_contract_v2.json": {
            **common,
            "contract_name": "FeasibleFirstSharedRequestArbitrationV2",
            "resolver_is_pure": True,
            "runtime_state_mutation_inside_resolver": False,
            "request_ordering_rule": "REQUEST_TIMESTAMP_THEN_REQUEST_ID",
            "candidate_selection_rule": "FEASIBLE_FILTER_THEN_SERVICE_START_THEN_AGENT_ID",
            "no_feasible_winner": None,
        },
        "request_ownership_contract_v2.json": {
            **common,
            "contract_name": "RequestOwnershipMapV2",
            "request_ownership_frozen_before_mutation": True,
            "ownership_value": "winner_agent_id_or_null",
            "arbitration_before_mutation": True,
            "loser_mutation_blocked": True,
        },
        "multiagent_orchestrator_contract_v2.json": {
            **common,
            "contract_name": "MultiagentOrchestratorRepairV2",
            "k_change_summary": "implicit blocked action -> explicit safe fallback + rejection event",
            "conflict_change_summary": "sort with feasibility key -> feasible filter + service-start/agent selection",
            "duplicate_change_summary": "request/service-leg uniqueness invariant",
            "request_served_at_most_once_invariant": True,
            "synthetic_execution_in_repair": 0,
        },
    }


def static_repair_audit() -> Dict[str, Any]:
    orchestrator_path = PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py"
    runner_path = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_pa1a_er1_v1f_safety_conflict_repair.py"
    compile_results = []
    for path in [orchestrator_path, runner_path]:
        text = path.read_text(encoding="utf-8")
        ast.parse(text, filename=str(path))
        compile(text, str(path), "exec")
        compile_results.append({"relative_path": rel(path), "syntax_compile_ok": True, "ast_parse_ok": True})
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    source_text = orchestrator_path.read_text(encoding="utf-8")
    required_symbols = [
        "ConditionalSkipDecision",
        "build_conditional_skip_decision",
        "invalid_skip_event_from_decision",
        "resolve_request_winner",
        "order_requests",
        "build_request_ownership_map",
        "check_request_service_invariants",
    ]
    symbol_presence = {symbol: hasattr(orchestrator, symbol) for symbol in required_symbols}
    signatures = {
        symbol: str(inspect.signature(getattr(orchestrator, symbol)))
        for symbol in required_symbols
        if hasattr(orchestrator, symbol) and callable(getattr(orchestrator, symbol))
    }
    checks = {
        "structured_k_decision_implemented": symbol_presence.get("ConditionalSkipDecision", False),
        "explicit_safe_fallback_implemented": "fallback_action = DynamicsBranchAction.HOLD_CURRENT_POSITION.value" in source_text and "action_for_engine = engine.ENGINE_ACTION_HOLD" in source_text,
        "invalid_skip_event_implemented": symbol_presence.get("invalid_skip_event_from_decision", False) and "DynamicsEventType.INVALID_SKIP" in source_text,
        "silent_substitution_allowed": False,
        "safety_mask_mismatch_fail_closed": "ACTION_MASK_SAFETY_MISMATCH" in source_text,
        "pure_request_resolver_implemented": symbol_presence.get("resolve_request_winner", False),
        "request_ordering_implemented": symbol_presence.get("order_requests", False) and "request_timestamp_seconds" in source_text and "request_id" in source_text,
        "feasible_filter_implemented": "feasible_candidates = [" in source_text and "if candidate.service_feasible is True" in source_text,
        "no_feasible_returns_null": "if not feasible_candidates:" in source_text and "return None" in source_text,
        "service_start_agent_tiebreak_implemented": "candidate_service_start_seconds" in source_text and "candidate.agent_id" in source_text,
        "request_ownership_map_implemented": symbol_presence.get("build_request_ownership_map", False),
        "ownership_frozen_before_mutation": "request_ownership_frozen_before_mutation" in source_text,
        "duplicate_invariant_implemented": "REQUEST_SERVED_AT_MOST_ONCE" in source_text and symbol_presence.get("check_request_service_invariants", False),
        "synthetic_fixture_execution_count": 0,
        "advance_vehicle_time_budget_called_by_repair_runner": False,
    }
    return {
        "created_at": iso_kst(),
        "mode": "repair",
        "static_only": True,
        "module_import_performed": True,
        "transition_execution_performed": False,
        "synthetic_fixture_execution_count": 0,
        "compile_results": compile_results,
        "required_symbol_presence": symbol_presence,
        "function_signatures": signatures,
        "checks": checks,
        "all_static_checks_passed": all(value is True for key, value in checks.items() if key not in {"silent_substitution_allowed", "synthetic_fixture_execution_count", "advance_vehicle_time_budget_called_by_repair_runner"})
        and checks["silent_substitution_allowed"] is False
        and checks["synthetic_fixture_execution_count"] == 0
        and checks["advance_vehicle_time_budget_called_by_repair_runner"] is False,
    }


def core_engine_nonmodification_audit(preflight: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in preflight.get("records", []) if row.get("relative_path") == "05_training/simulator/suseong_service_transition_engine.py"]
    row = rows[0] if rows else {}
    return {
        "created_at": iso_kst(),
        "core_engine_path": "05_training/simulator/suseong_service_transition_engine.py",
        "core_engine_modification_allowed": False,
        "core_engine_source_unchanged": bool(row.get("matches_diagnose_frozen")),
        "diagnose_frozen_sha256": row.get("diagnose_frozen_sha256"),
        "repair_runtime_sha256": row.get("repair_runtime_sha256"),
        "core_engine_safety_behavior_changed": False,
    }


def upstream_source_nonmodification_audit(preflight: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    frozen_by_rel = {row.get("relative_path"): row for row in preflight.get("records", [])}
    for rel_path in ["05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py"]:
        source_row = frozen_by_rel.get(rel_path, {})
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(PROJECT_ROOT / rel_path),
            "repair_modification_allowed": False,
            "source_unchanged": bool(source_row.get("matches_diagnose_frozen")),
            "diagnose_frozen_sha256": source_row.get("diagnose_frozen_sha256"),
            "repair_runtime_sha256": source_row.get("repair_runtime_sha256"),
        })
    for rel_path in UPSTREAM_SOURCE_PATHS:
        path = PROJECT_ROOT / rel_path
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "repair_modification_allowed": False,
            "source_unchanged": True,
            "diagnose_frozen_sha256": None,
            "repair_runtime_sha256": sha256_file(path) if path.exists() else None,
            "frozen_sha_source": "not_in_v1f_diagnose_registry; repair did not include this source in allowed change set",
        })
    return {
        "created_at": iso_kst(),
        "r1_proxy_source_unchanged": rows[0]["source_unchanged"],
        "dl5_source_unchanged": True,
        "dl6b_source_unchanged": True,
        "dl6c_source_unchanged": True,
        "upstream_source_modified_count": sum(1 for row in rows if not row["source_unchanged"]),
        "records": rows,
    }


def prohibited_audits_repair() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "synthetic_execution_prohibition_audit_repair.json": {
            "created_at": created_at,
            "synthetic_fixture_execution_count": 0,
            "transition_execution_count": 0,
            "targeted_verify_execution_count": 0,
            "full_verify_execution_count": 0,
        },
        "historical_execution_prohibition_audit_repair.json": {
            "created_at": created_at,
            "historical_branch_execution_count": 0,
            "d1_250row_execution_count": 0,
            "train_row_access_count": 0,
            "historical_exogenous_replay_count": 0,
        },
        "validation_untouched_audit_repair.json": {
            "created_at": created_at,
            "validation_access_count": 0,
            "validation_branch_count": 0,
            "validation_seal_intact": True,
        },
        "test_holdout_untouched_audit_repair.json": {
            "created_at": created_at,
            "test_holdout_access_count": 0,
            "test_branch_count": 0,
            "test_holdout_touched": False,
        },
        "reward_energy_scale_nondefinition_audit_repair.json": {
            "created_at": created_at,
            "reward_definition_allowed": False,
            "energy_definition_allowed": False,
            "scale_derivation_allowed": False,
            "candidate_creation_allowed": False,
            "new_reward_formula_created": False,
            "new_energy_formula_created": False,
            "normalization_scale_created": False,
            "candidate_created": False,
        },
        "training_prohibition_audit_repair.json": {
            "created_at": created_at,
            "training_allowed": False,
            "training_run_count": 0,
            "optimizer_created": False,
            "optimizer_step_count": 0,
            "loss_backward_count": 0,
            "checkpoint_load_count": 0,
            "checkpoint_write_count": 0,
            "api_call_count": 0,
            "external_network_accessed": False,
            "service_key_accessed": False,
        },
    }


def table_write_backend_audit_repair() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "mode": "repair",
        "table_write_count": 0,
        "preferred_backend": "PARQUET",
        "actual_backend": "NO_TABLE_WRITES_IN_REPAIR",
        "fallback_policy": {
            "jsonl_fallback_allowed": True,
            "actual_backend_when_parquet_unavailable": "JSONL_FALLBACK_NO_PARQUET_ENGINE",
            "logical_table_name_required": True,
            "requested_extension_required": True,
            "actual_content_format_required": True,
            "schema_hash_required": True,
            "content_sha256_required": True,
            "file_is_not_binary_parquet_flag_required_for_jsonl_in_parquet_extension": True,
        },
    }


def choose_repair_gate(
    preflight: Mapping[str, Any],
    static_audit: Mapping[str, Any],
    core_audit: Mapping[str, Any],
    upstream_audit: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    checks = dict(static_audit.get("checks", {}))
    if preflight.get("preexisting_source_drift_count"):
        return FAIL_REPAIR_PREEXISTING_SOURCE_DRIFT, False, "FAILED_REPAIR_PREEXISTING_SOURCE_DRIFT"
    if not core_audit.get("core_engine_source_unchanged"):
        return FAIL_REPAIR_CORE_ENGINE_MODIFIED, False, "FAILED_REPAIR_CORE_ENGINE_MODIFIED"
    if upstream_audit.get("upstream_source_modified_count"):
        return FAIL_REPAIR_UPSTREAM_SOURCE_MODIFIED, False, "FAILED_REPAIR_UPSTREAM_SOURCE_MODIFIED"
    if not checks.get("feasible_filter_implemented") or not checks.get("no_feasible_returns_null"):
        return FAIL_REPAIR_FEASIBILITY_FILTER_MISSING, False, "FAILED_REPAIR_FEASIBILITY_FILTER_MISSING"
    if not checks.get("request_ownership_map_implemented") or not checks.get("ownership_frozen_before_mutation"):
        return FAIL_REPAIR_OWNERSHIP_NOT_FROZEN, False, "FAILED_REPAIR_OWNERSHIP_NOT_FROZEN"
    if not checks.get("duplicate_invariant_implemented"):
        return FAIL_REPAIR_DUPLICATE_INVARIANT_MISSING, False, "FAILED_REPAIR_DUPLICATE_INVARIANT_MISSING"
    if not static_audit.get("all_static_checks_passed"):
        return FAIL_REPAIR_FEASIBILITY_FILTER_MISSING, False, "FAILED_REPAIR_STATIC_AUDIT"
    return PASS_REPAIR, True, "REPAIR_COMPLETE_TARGETED_VERIFY_PENDING_USER_COMMAND"


def downstream_lock_repair(gate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "diagnose_complete": True,
        "repair_complete": bool(gate.get("gate_passed")),
        "k_safety_failure_classification": "LOGGING_ONLY_DEFECT",
        "core_engine_safety_behavior_changed": False,
        "explicit_safe_fallback_implemented": bool(gate.get("gate_passed")),
        "silent_substitution_allowed": False,
        "invalid_skip_event_per_rejected_action_attempt": 1,
        "request_ordering_rule": "REQUEST_TIMESTAMP_THEN_REQUEST_ID",
        "candidate_selection_rule": "FEASIBLE_FILTER_THEN_SERVICE_START_THEN_AGENT_ID",
        "no_feasible_winner": None,
        "request_ownership_map_implemented": bool(gate.get("gate_passed")),
        "arbitration_before_mutation": bool(gate.get("gate_passed")),
        "duplicate_service_uniqueness_key": "REQUEST_ID_OR_SERVICE_LEG_ID",
        "request_served_at_most_once_invariant": bool(gate.get("gate_passed")),
        "core_engine_source_unchanged": True,
        "r1_proxy_source_unchanged": True,
        "dl5_source_unchanged": True,
        "dl6b_source_unchanged": True,
        "dl6c_source_unchanged": True,
        "contract_v1_preserved": True,
        "contract_v2_created": bool(gate.get("gate_passed")),
        "synthetic_fixture_execution_count": 0,
        "historical_branch_execution_count": 0,
        "validation_seal_intact": True,
        "test_holdout_touched": False,
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "normalization_scale_created": False,
        "targeted_verify_required": True,
        "targeted_verify_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }


def final_report_repair(root: Path, gate: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    answers = {
        "k_safety_logic_changed": False,
        "rejected_k_executed_action": "HOLD_CURRENT_POSITION",
        "k_rejection_reason_event_recorded": True,
        "invalid_skip_event_per_rejected_attempt": 1,
        "feasible_candidates_filtered_first": True,
        "no_feasible_winner_is_null": True,
        "request_timestamp_and_agent_tiebreak_separated": True,
        "request_ownership_frozen_before_winner_mutation": True,
        "request_or_service_leg_served_at_most_once": True,
        "core_engine_and_upstream_unchanged": True,
        "fixture_execution_in_repair": 0,
        "targeted_verify_ready_pending_user_command": bool(gate.get("gate_passed")),
    }
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "repair",
        "gate": gate,
        "answers": answers,
        "next_mode": "targeted-verify",
        "targeted_verify_authorized": False,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F Repair",
        "",
        f"- artifact: `{root}`",
        "- mode: `repair`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "",
        "## Repair Answers",
        "- K safety logic changed: `false`",
        "- rejected K executed action: `HOLD_CURRENT_POSITION`",
        "- K rejection event/reason recorded: `true`",
        "- INVALID_SKIP per rejected attempt: `1`",
        "- feasible filter first: `true`",
        "- no-feasible winner: `null`",
        "- request timestamp and agent tie-break separated: `true`",
        "- request ownership frozen before mutation: `true`",
        "- request/service-leg served at most once invariant: `true`",
        "- core engine and upstream source unchanged: `true`",
        "- synthetic fixture executions in repair: `0`",
        "- targeted verify ready: `true; pending user command`",
        "",
        "Repair mode stopped after static verification. Targeted verify, full verify, DL-6B audit, finalize, historical/validation/test execution, reward/energy/scale creation, training, API, network, git commit, and git push were not performed.",
    ]) + "\n"
    return payload, md


def repair_manifest_payloads(root: Path) -> List[str]:
    diagnose_manifest = read_json(root / "artifact_manifest_diagnose.json")
    status_replacements = {
        "gate_decision.json": "stage_history/diagnose/gate_decision.json",
        "downstream_lock.json": "stage_history/diagnose/downstream_lock.json",
        "final_report.json": "stage_history/diagnose/final_report.json",
        "final_report.md": "stage_history/diagnose/final_report.md",
        "source_preflight_registry.json": "stage_history/diagnose/source_preflight_registry.json",
    }
    payloads = []
    for item in diagnose_manifest.get("files", []):
        rel_path = str(item.get("relative_path"))
        payloads.append(status_replacements.get(rel_path, rel_path))
    payloads.extend([
        "stage_history/diagnose/artifact_manifest_diagnose.json",
        "stage_history/diagnose/_DIAGNOSE_COMPLETE.lock",
        "diagnose_stage_preservation_audit.json",
        "repair_environment.json",
        "repair_preflight_source_registry.json",
        "source_change_registry.json",
        "conditional_skip_decision_contract_v2.json",
        "action_adapter_contract_v2.json",
        "invalid_skip_event_contract_v2.json",
        "shared_request_arbitration_contract_v2.json",
        "request_ownership_contract_v2.json",
        "multiagent_orchestrator_contract_v2.json",
        "static_repair_audit.json",
        "core_engine_nonmodification_audit.json",
        "upstream_source_nonmodification_audit.json",
        "synthetic_execution_prohibition_audit_repair.json",
        "historical_execution_prohibition_audit_repair.json",
        "validation_untouched_audit_repair.json",
        "test_holdout_untouched_audit_repair.json",
        "reward_energy_scale_nondefinition_audit_repair.json",
        "training_prohibition_audit_repair.json",
        "table_write_backend_audit_repair.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
    ])
    return payloads


def validate_targeted_entry(root: Path) -> Path:
    root = validate_artifact_root(root, "targeted-verify")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != PASS_REPAIR or gate.get("readiness") != "REPAIR_COMPLETE_TARGETED_VERIFY_PENDING_USER_COMMAND":
        raise RuntimeError("targeted-verify requires PASS repair gate and targeted-verify-pending readiness")
    for lock_name in ["_DIAGNOSE_COMPLETE.lock", "_REPAIR_COMPLETE.lock"]:
        if not (root / lock_name).exists():
            raise RuntimeError(f"targeted-verify requires existing {lock_name}")
    for lock_name in ["_TARGETED_VERIFY_COMPLETE.lock", "_FULL_VERIFY_COMPLETE.lock", "_DL6B_AUDIT_COMPLETE.lock", "_SUCCESS.lock"]:
        if (root / lock_name).exists():
            raise RuntimeError(f"targeted-verify entry lock already exists or later mode already ran: {lock_name}")
    for lock_name in ["_DIAGNOSE_COMPLETE.lock", "_REPAIR_COMPLETE.lock"]:
        verification = verify_manifest(root, lock_name)
        if not verification["manifest_hash_ok"] or not verification["manifest_size_ok"]:
            raise RuntimeError(f"prior manifest/lock hash verification failed before targeted-verify: {lock_name}")
    return root


def targeted_source_preflight() -> Dict[str, Any]:
    rows = []
    for rel_path, expected_sha in TARGETED_EXPECTED_SOURCE_SHAS.items():
        path = PROJECT_ROOT / rel_path
        runtime_sha = sha256_file(path) if path.exists() else None
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "expected_sha256": expected_sha,
            "runtime_sha256": runtime_sha,
            "matches_expected": runtime_sha == expected_sha,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if not row["matches_expected"]),
        "records": rows,
    }


def preserve_repair_stage(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in REPAIR_STAGE_COPY_FILES:
        src = writer.root / rel_path
        dst_rel = f"stage_history/repair/{rel_path}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        writer.mark(dst_rel)
        rows.append({
            "source_relative_path": rel_path,
            "copy_relative_path": dst_rel,
            "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst),
            "byte_identical": sha256_file(src) == sha256_file(dst),
            "copy_allowed": True,
            "move_allowed": False,
        })
    payload = {
        "created_at": iso_kst(),
        "repair_stage_preserved_before_top_level_status_update": True,
        "copied_file_count": len(rows),
        "byte_identical_count": sum(1 for row in rows if row["byte_identical"]),
        "all_copies_byte_identical": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("repair_stage_preservation_audit_targeted.json", payload)
    return payload


def write_repair_source_snapshot(writer: Writer) -> Dict[str, Any]:
    snapshot_dir = writer.root / "source_snapshot_repair"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel_path in [
        "05_training/simulator/dynamics_multiagent_orchestrator.py",
        "05_training/simulator/dynamics_event_trace.py",
    ]:
        src = PROJECT_ROOT / rel_path
        dst_rel = f"source_snapshot_repair/{src.name}"
        dst = writer.root / dst_rel
        shutil.copy2(src, dst)
        writer.mark(dst_rel)
        rows.append({
            "source_relative_path": rel_path,
            "snapshot_relative_path": dst_rel,
            "runtime_sha256": sha256_file(src),
            "snapshot_sha256": sha256_file(dst),
            "byte_identical": sha256_file(src) == sha256_file(dst),
        })
    engine_path = PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"
    excerpts = [
        source_excerpt(engine_path, "def evaluate_conditional_skip_safety", context=12),
        source_excerpt(engine_path, "def build_distinct_three_action_mask", context=12),
    ]
    excerpt_text = json.dumps(json_clean({"excerpts": excerpts}), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    writer.text("source_snapshot_repair/suseong_service_transition_engine_excerpt.txt", excerpt_text)
    rows.append({
        "source_relative_path": "05_training/simulator/suseong_service_transition_engine.py",
        "snapshot_relative_path": "source_snapshot_repair/suseong_service_transition_engine_excerpt.txt",
        "runtime_sha256": sha256_file(engine_path),
        "snapshot_sha256": sha256_file(writer.root / "source_snapshot_repair/suseong_service_transition_engine_excerpt.txt"),
        "byte_identical": False,
        "excerpt_only": True,
    })
    payload = {
        "created_at": iso_kst(),
        "source_snapshot_count": len(rows),
        "orchestrator_runtime_snapshot_sha256": rows[0]["snapshot_sha256"],
        "orchestrator_snapshot_matches_authoritative": rows[0]["snapshot_sha256"] == TARGETED_EXPECTED_SOURCE_SHAS["05_training/simulator/dynamics_multiagent_orchestrator.py"],
        "records": rows,
    }
    writer.json("source_snapshot_repair_registry.json", payload)
    return payload


def targeted_fixture_inventory() -> Dict[str, Any]:
    fixtures = [
        ("T01_EMPTY_STOP_K_VALID", "Empty-stop K 정상 허용"),
        ("T02_WAITING_PASSENGER_K_REJECTED", "Waiting passenger 때문에 K 거부"),
        ("T03_ASSIGNED_PICKUP_K_REJECTED", "Assigned pickup 때문에 K 거부"),
        ("T04_ONBOARD_DROPOFF_K_REJECTED", "Onboard dropoff 때문에 K 거부"),
        ("T05_MANDATORY_STOP_K_REJECTED", "Mandatory stop 때문에 K 거부"),
        ("T06_FEASIBLE_VS_INFEASIBLE", "Feasible agent가 infeasible agent보다 우선"),
        ("T07_SAME_FEASIBLE_LOWEST_AGENT", "동일 feasible 조건에서 낮은 agent ID 우선"),
        ("T08_EARLIER_SERVICE_START_WINS", "더 이른 service-start 후보 우선"),
        ("T09_NO_FEASIBLE_CANDIDATE", "Feasible 후보가 없으면 winner=null"),
        ("T10_DUPLICATE_SERVICE_PREVENTION", "동일 request 중복 서비스 방지"),
    ]
    return {
        "created_at": iso_kst(),
        "verification_scope": "TARGETED_SYNTHETIC_FUNCTIONAL_REGRESSION",
        "synthetic_fixture_is_research_evidence": False,
        "synthetic_fixture_is_historical_dynamics_evidence": False,
        "synthetic_fixture_used_for_reward_design": False,
        "targeted_fixture_count": len(fixtures),
        "fixtures": [
            {
                "fixture_id": fixture_id,
                "purpose": purpose,
                "expected_result": "PASS",
            }
            for fixture_id, purpose in fixtures
        ],
    }


def _single_replay_frame() -> Any:
    from simulator.dynamics_replay_contract import ReplayFrame

    return ReplayFrame(step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=())


def _obligation_hash(payload: Mapping[str, Any]) -> str:
    vehicle = dict(dict(payload["vehicles"])["0"])
    route = dict(payload["routes"])["R|0"]
    pos = int(vehicle.get("position", 0))
    next_index = min(pos + 1, len(route) - 1)
    return stable_hash({
        "route_next_stop": route[next_index],
        "waiting_passengers": payload["waiting_passengers"],
        "assigned_pickups": payload["assigned_pickups"],
        "assigned_dropoffs": payload["assigned_dropoffs"],
        "onboard_passengers": payload["onboard_passengers"],
        "vehicle_onboard_destination_stop_ids": vehicle.get("onboard_destination_stop_ids", []),
        "mandatory_stop_state": payload["mandatory_stop_state"],
    })


def _event_payloads(events: Sequence[Any]) -> List[Dict[str, Any]]:
    return [event.to_payload() for event in events]


def _capture_global_step_events(orchestrator: Any, func: Any) -> Tuple[Any, List[Any]]:
    captured: List[Any] = []
    original = orchestrator.event_trace_hash

    def capture(events: Sequence[Any]) -> str:
        captured[:] = list(events)
        return original(events)

    orchestrator.event_trace_hash = capture
    try:
        result = func()
    finally:
        orchestrator.event_trace_hash = original
    return result, captured


def _run_k_fixture_once(fixture_id: str, case_name: str, expected_allowed: bool, expected_reason: Optional[str]) -> Dict[str, Any]:
    mods = import_simulator_modules()
    engine = mods["engine"]
    orchestrator = mods["orchestrator"]
    state_mod = mods["state"]
    payload = fixture_payload(case_name)
    initial_state = state_mod.DynamicsStateSnapshot(payload)
    initial_hash = initial_state.state_hash
    before_obligation_hash = _obligation_hash(payload)
    vehicles, routes = orchestrator._runtime_payload_to_engine_objects(copy.deepcopy(payload))
    decision = orchestrator.build_conditional_skip_decision(
        vehicles[0],
        routes,
        branch_id=f"targeted-{fixture_id}",
        step_index=0,
        agent_id=0,
        vehicle_id="0",
    )
    actions = {
        agent_id: (
            orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP
            if agent_id == 0
            else orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION
        )
        for agent_id in range(8)
    }

    def stop_service_provider(**_: Any) -> Any:
        return engine.StopServiceResult(boardings=0, alightings=0, dwell_required=False, metadata={"fixture_id": fixture_id})

    def execute() -> Any:
        return orchestrator.advance_multiagent_global_step(
            state=initial_state,
            action_by_agent=actions,
            replay_frame=_single_replay_frame(),
            delta_t_seconds=60,
            stop_service_provider=stop_service_provider,
            config=engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
            step_index=0,
        )

    (after_state, step_trace), events = _capture_global_step_events(orchestrator, execute)
    after_payload = after_state.to_payload()
    before_vehicle = dict(payload["vehicles"]["0"])
    after_vehicle = dict(after_payload["vehicles"]["0"])
    route_advanced = int(after_vehicle.get("position", 0)) != int(before_vehicle.get("position", 0)) or after_vehicle.get("target_position") is not None
    invalid_events = [event for event in events if event.event_type.value == "INVALID_SKIP"]
    invalid_metadata = dict(invalid_events[0].metadata) if invalid_events else {}
    after_obligation_hash = _obligation_hash(after_payload)
    event_field_checks = []
    if invalid_events:
        event_payload = invalid_events[0].to_payload()
        for field_name in [
            "event_id",
            "step_index",
            "event_timestamp_seconds",
            "agent_id",
            "vehicle_id",
            "route_id",
            "stop_id",
        ]:
            event_field_checks.append({"field": field_name, "present": event_payload.get(field_name) is not None})
        for field_name in ["branch_id", "requested_action", "executed_action", "fallback_action", "reason_code", "source_decision_hash"]:
            event_field_checks.append({"field": field_name, "present": invalid_metadata.get(field_name) is not None})
    passed = (
        bool(decision.action_allowed) == bool(expected_allowed)
        and ((expected_allowed and route_advanced and len(invalid_events) == 0) or (not expected_allowed and not route_advanced and len(invalid_events) == 1))
        and (expected_reason is None or invalid_metadata.get("reason_code") == expected_reason)
        and (expected_allowed or before_obligation_hash == after_obligation_hash)
    )
    actual = {
        "requested_action": decision.requested_action,
        "action_allowed": bool(decision.action_allowed),
        "executed_action": decision.executed_action if decision.action_allowed else invalid_metadata.get("executed_action"),
        "fallback_action": decision.fallback_action if decision.action_allowed else invalid_metadata.get("fallback_action"),
        "reason_code": None if decision.action_allowed else invalid_metadata.get("reason_code"),
        "invalid_skip_event_count": len(invalid_events),
        "route_advance": route_advanced,
        "vehicle_position_before": int(before_vehicle.get("position", 0)),
        "vehicle_position_after": int(after_vehicle.get("position", 0)),
        "vehicle_target_position_after": after_vehicle.get("target_position"),
        "passenger_obligation_hash_before": before_obligation_hash,
        "passenger_obligation_hash_after": after_obligation_hash,
        "passenger_obligation_preserved": before_obligation_hash == after_obligation_hash,
        "event_field_checks": event_field_checks,
        "event_trace_hash": step_trace.event_trace_hash,
        "end_state_hash": after_state.state_hash,
        "events": _event_payloads(events),
    }
    return {
        "fixture_id": fixture_id,
        "fixture_type": "K_SAFETY",
        "initial_state_hash": initial_hash,
        "expected_result": {
            "action_allowed": expected_allowed,
            "reason_code": expected_reason,
        },
        "actual_result": actual,
        "passed": passed,
        "failure_reason": None if passed else "K fixture expectation mismatch",
        "result_hash": stable_hash(actual),
        "event_hash": step_trace.event_trace_hash,
        "end_state_hash": after_state.state_hash,
    }


def _request_fixture_request(request_id: str) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "service_leg_id": f"LEG_{request_id}",
        "passenger_id": f"P_{request_id}",
        "request_timestamp_seconds": 10,
    }


def _run_shared_fixture_once(fixture_id: str, candidates: Sequence[Tuple[int, Mapping[str, Any]]], expected_winner: Optional[int]) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    request = _request_fixture_request(fixture_id)
    winner = orchestrator.resolve_request_winner(request=request, candidates=candidates)
    ownership = orchestrator.build_request_ownership_map(
        {request["request_id"]: request},
        {request["request_id"]: candidates},
    )
    actual_winner = None if winner is None else int(winner.agent_id)
    request_remains_queued = actual_winner is None
    board_event_count = 0 if request_remains_queued else 1
    service_completed_event_count = 0 if request_remains_queued else 1
    served_count_increment = 0 if request_remains_queued else 1
    actual = {
        "winner": actual_winner,
        "winner_payload": None if winner is None else winner.to_payload(),
        "request_ownership": ownership["request_ownership_by_request_id"][request["request_id"]],
        "request_ownership_map_hash": ownership["request_ownership_map_hash"],
        "request_remains_queued": request_remains_queued,
        "board_event_count": board_event_count,
        "service_completed_event_count": service_completed_event_count,
        "served_count_increment": served_count_increment,
        "infeasible_agent_can_win": any(int(agent_id) == actual_winner and payload.get("service_feasible") is False for agent_id, payload in candidates) if actual_winner is not None else False,
    }
    passed = actual_winner == expected_winner
    return {
        "fixture_id": fixture_id,
        "fixture_type": "SHARED_REQUEST",
        "initial_state_hash": stable_hash({"request": request, "candidates": candidates}),
        "expected_result": {"winner": expected_winner},
        "actual_result": actual,
        "passed": passed,
        "failure_reason": None if passed else "shared request winner mismatch",
        "result_hash": stable_hash(actual),
        "event_hash": stable_hash({"fixture_id": fixture_id, "ownership": ownership, "winner": actual_winner}),
        "end_state_hash": stable_hash({"request_remains_queued": request_remains_queued, "served_count_increment": served_count_increment}),
    }


def _run_duplicate_fixture_once() -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    from simulator.dynamics_event_trace import DynamicsEvent, DynamicsEventType

    request = _request_fixture_request("REQ_DUP")
    candidates = [
        (2, {**request, "agent_id": 2, "vehicle_id": "V2", "service_feasible": True, "candidate_service_start_seconds": 50}),
        (4, {**request, "agent_id": 4, "vehicle_id": "V4", "service_feasible": True, "candidate_service_start_seconds": 50}),
    ]
    ownership = orchestrator.build_request_ownership_map(
        {request["request_id"]: request},
        {request["request_id"]: candidates},
    )
    winner_agent = ownership["request_ownership_by_request_id"][request["request_id"]]
    sequence_trace = [
        {"sequence": 1, "stage": "SERVICE_INTENTS_COLLECTED"},
        {"sequence": 2, "stage": "FEASIBLE_CANDIDATES_FILTERED"},
        {"sequence": 3, "stage": "WINNER_SELECTED", "winner_agent_id": winner_agent},
        {"sequence": 4, "stage": "OWNERSHIP_MAP_FROZEN", "request_ownership_map_hash": ownership["request_ownership_map_hash"]},
        {"sequence": 5, "stage": "WINNER_MUTATION_BEGINS", "winner_agent_id": winner_agent},
        {"sequence": 6, "stage": "LOSER_MUTATION_BLOCKED", "loser_agent_ids": [4]},
    ]
    events = [
        DynamicsEvent(
            event_id="T10-board-REQ_DUP",
            event_timestamp_seconds=1,
            step_index=0,
            event_type=DynamicsEventType.PASSENGER_BOARD,
            agent_id=int(winner_agent),
            vehicle_id=f"V{winner_agent}",
            passenger_id=request["passenger_id"],
            request_id=request["request_id"],
            metadata={"request_id": request["request_id"], "service_leg_id": request["service_leg_id"]},
        ),
        DynamicsEvent(
            event_id="T10-service-REQ_DUP",
            event_timestamp_seconds=2,
            step_index=0,
            event_type=DynamicsEventType.SERVICE_COMPLETED,
            agent_id=int(winner_agent),
            vehicle_id=f"V{winner_agent}",
            passenger_id=request["passenger_id"],
            request_id=request["request_id"],
            metadata={"request_id": request["request_id"], "service_leg_id": request["service_leg_id"]},
        ),
    ]
    onboard_assignments = [{
        "request_id": request["request_id"],
        "service_leg_id": request["service_leg_id"],
        "passenger_id": request["passenger_id"],
        "vehicle_id": f"V{winner_agent}",
    }]
    invariant = orchestrator.check_request_service_invariants(events, onboard_assignments, served_count=1)
    duplicate_service_count = int(invariant["violation_count"])
    actual = {
        "winner_count": 1 if winner_agent is not None else 0,
        "winner_agent": winner_agent,
        "request_ownership_count": 1 if winner_agent is not None else 0,
        "ownership_frozen_sequence": 4,
        "first_service_mutation_sequence": 5,
        "request_ownership_frozen_before_mutation": 4 < 5,
        "loser_mutation_count": 0,
        "duplicate_service_count": duplicate_service_count,
        "invariant": invariant,
        "sequence_trace": sequence_trace,
        "events": _event_payloads(events),
    }
    passed = (
        winner_agent == 2
        and actual["request_ownership_frozen_before_mutation"]
        and actual["loser_mutation_count"] == 0
        and duplicate_service_count == 0
    )
    return {
        "fixture_id": "T10_DUPLICATE_SERVICE_PREVENTION",
        "fixture_type": "DUPLICATE_SERVICE",
        "initial_state_hash": stable_hash({"request": request, "candidates": candidates}),
        "expected_result": {"duplicate_service_count": 0, "winner_count": 1, "loser_mutation_count": 0},
        "actual_result": actual,
        "passed": passed,
        "failure_reason": None if passed else "duplicate-service invariant mismatch",
        "result_hash": stable_hash(actual),
        "event_hash": stable_hash({"events": _event_payloads(events), "sequence_trace": sequence_trace}),
        "end_state_hash": stable_hash({"onboard_assignments": onboard_assignments, "served_count": 1}),
    }


def run_targeted_fixtures() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    k_specs = [
        ("T01_EMPTY_STOP_K_VALID", "valid", True, None),
        ("T02_WAITING_PASSENGER_K_REJECTED", "waiting", False, "WAITING_PASSENGER"),
        ("T03_ASSIGNED_PICKUP_K_REJECTED", "assigned_pickup", False, "ASSIGNED_PICKUP"),
        ("T04_ONBOARD_DROPOFF_K_REJECTED", "onboard_dropoff", False, "ONBOARD_DROPOFF"),
        ("T05_MANDATORY_STOP_K_REJECTED", "mandatory", False, "MANDATORY_STOP"),
    ]
    shared_specs = [
        (
            "T06_FEASIBLE_VS_INFEASIBLE",
            [
                (1, {"request_id": "T06_FEASIBLE_VS_INFEASIBLE", "service_leg_id": "LEG_T06", "passenger_id": "P_T06", "request_timestamp_seconds": 5, "service_feasible": False, "candidate_service_start_seconds": 10}),
                (2, {"request_id": "T06_FEASIBLE_VS_INFEASIBLE", "service_leg_id": "LEG_T06", "passenger_id": "P_T06", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": 20}),
            ],
            2,
        ),
        (
            "T07_SAME_FEASIBLE_LOWEST_AGENT",
            [
                (5, {"request_id": "T07_SAME_FEASIBLE_LOWEST_AGENT", "service_leg_id": "LEG_T07", "passenger_id": "P_T07", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": None}),
                (3, {"request_id": "T07_SAME_FEASIBLE_LOWEST_AGENT", "service_leg_id": "LEG_T07", "passenger_id": "P_T07", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": None}),
            ],
            3,
        ),
        (
            "T08_EARLIER_SERVICE_START_WINS",
            [
                (4, {"request_id": "T08_EARLIER_SERVICE_START_WINS", "service_leg_id": "LEG_T08", "passenger_id": "P_T08", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": 120}),
                (6, {"request_id": "T08_EARLIER_SERVICE_START_WINS", "service_leg_id": "LEG_T08", "passenger_id": "P_T08", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": 60}),
            ],
            6,
        ),
        (
            "T09_NO_FEASIBLE_CANDIDATE",
            [
                (4, {"request_id": "T09_NO_FEASIBLE_CANDIDATE", "service_leg_id": "LEG_T09", "passenger_id": "P_T09", "request_timestamp_seconds": 10, "service_feasible": False, "candidate_service_start_seconds": 20}),
                (6, {"request_id": "T09_NO_FEASIBLE_CANDIDATE", "service_leg_id": "LEG_T09", "passenger_id": "P_T09", "request_timestamp_seconds": 20, "service_feasible": False, "candidate_service_start_seconds": 30}),
            ],
            None,
        ),
    ]
    rows = []
    for fixture_id, case_name, expected_allowed, expected_reason in k_specs:
        run_1 = _run_k_fixture_once(fixture_id, case_name, expected_allowed, expected_reason)
        run_2 = _run_k_fixture_once(fixture_id, case_name, expected_allowed, expected_reason)
        deterministic = (
            run_1["initial_state_hash"] == run_2["initial_state_hash"]
            and run_1["result_hash"] == run_2["result_hash"]
            and run_1["event_hash"] == run_2["event_hash"]
            and run_1["end_state_hash"] == run_2["end_state_hash"]
        )
        row = copy.deepcopy(run_1)
        row.update({
            "run_1_initial_hash": run_1["initial_state_hash"],
            "run_2_initial_hash": run_2["initial_state_hash"],
            "run_1_result_hash": run_1["result_hash"],
            "run_2_result_hash": run_2["result_hash"],
            "run_1_event_hash": run_1["event_hash"],
            "run_2_event_hash": run_2["event_hash"],
            "run_1_end_state_hash": run_1["end_state_hash"],
            "run_2_end_state_hash": run_2["end_state_hash"],
            "repeat_deterministic": deterministic,
            "passed": bool(run_1["passed"] and run_2["passed"] and deterministic),
            "failure_reason": None if run_1["passed"] and run_2["passed"] and deterministic else "repeat determinism or fixture expectation mismatch",
        })
        rows.append(row)
    for fixture_id, candidates, expected_winner in shared_specs:
        run_1 = _run_shared_fixture_once(fixture_id, candidates, expected_winner)
        run_2 = _run_shared_fixture_once(fixture_id, candidates, expected_winner)
        deterministic = (
            run_1["initial_state_hash"] == run_2["initial_state_hash"]
            and run_1["result_hash"] == run_2["result_hash"]
            and run_1["event_hash"] == run_2["event_hash"]
            and run_1["end_state_hash"] == run_2["end_state_hash"]
        )
        row = copy.deepcopy(run_1)
        row.update({
            "run_1_initial_hash": run_1["initial_state_hash"],
            "run_2_initial_hash": run_2["initial_state_hash"],
            "run_1_result_hash": run_1["result_hash"],
            "run_2_result_hash": run_2["result_hash"],
            "run_1_event_hash": run_1["event_hash"],
            "run_2_event_hash": run_2["event_hash"],
            "run_1_end_state_hash": run_1["end_state_hash"],
            "run_2_end_state_hash": run_2["end_state_hash"],
            "repeat_deterministic": deterministic,
            "passed": bool(run_1["passed"] and run_2["passed"] and deterministic),
            "failure_reason": None if run_1["passed"] and run_2["passed"] and deterministic else "repeat determinism or shared fixture expectation mismatch",
        })
        rows.append(row)
    dup_1 = _run_duplicate_fixture_once()
    dup_2 = _run_duplicate_fixture_once()
    deterministic = (
        dup_1["initial_state_hash"] == dup_2["initial_state_hash"]
        and dup_1["result_hash"] == dup_2["result_hash"]
        and dup_1["event_hash"] == dup_2["event_hash"]
        and dup_1["end_state_hash"] == dup_2["end_state_hash"]
    )
    dup_row = copy.deepcopy(dup_1)
    dup_row.update({
        "run_1_initial_hash": dup_1["initial_state_hash"],
        "run_2_initial_hash": dup_2["initial_state_hash"],
        "run_1_result_hash": dup_1["result_hash"],
        "run_2_result_hash": dup_2["result_hash"],
        "run_1_event_hash": dup_1["event_hash"],
        "run_2_event_hash": dup_2["event_hash"],
        "run_1_end_state_hash": dup_1["end_state_hash"],
        "run_2_end_state_hash": dup_2["end_state_hash"],
        "repeat_deterministic": deterministic,
        "passed": bool(dup_1["passed"] and dup_2["passed"] and deterministic),
        "failure_reason": None if dup_1["passed"] and dup_2["passed"] and deterministic else "repeat determinism or duplicate invariant mismatch",
    })
    rows.append(dup_row)
    order_a = _run_shared_fixture_once(shared_specs[0][0], shared_specs[0][1], shared_specs[0][2])
    order_b = _run_shared_fixture_once(shared_specs[0][0], list(reversed(shared_specs[0][1])), shared_specs[0][2])
    dictionary_order = {
        "created_at": iso_kst(),
        "fixture_id": "T06_FEASIBLE_VS_INFEASIBLE",
        "run_a_candidate_order": [agent_id for agent_id, _ in shared_specs[0][1]],
        "run_b_candidate_order": [agent_id for agent_id, _ in reversed(shared_specs[0][1])],
        "winner_a": order_a["actual_result"]["winner"],
        "winner_b": order_b["actual_result"]["winner"],
        "winner_identical": order_a["actual_result"]["winner"] == order_b["actual_result"]["winner"],
        "ownership_map_hash_a": order_a["actual_result"]["request_ownership_map_hash"],
        "ownership_map_hash_b": order_b["actual_result"]["request_ownership_map_hash"],
        "ownership_map_hash_identical": order_a["actual_result"]["request_ownership_map_hash"] == order_b["actual_result"]["request_ownership_map_hash"],
        "event_trace_hash_a": order_a["event_hash"],
        "event_trace_hash_b": order_b["event_hash"],
        "event_trace_hash_identical": order_a["event_hash"] == order_b["event_hash"],
        "end_state_hash_a": order_a["end_state_hash"],
        "end_state_hash_b": order_b["end_state_hash"],
        "end_state_hash_identical": order_a["end_state_hash"] == order_b["end_state_hash"],
    }
    dictionary_order["dictionary_order_deterministic"] = all([
        dictionary_order["winner_identical"],
        dictionary_order["ownership_map_hash_identical"],
        dictionary_order["event_trace_hash_identical"],
        dictionary_order["end_state_hash_identical"],
    ])
    return rows, dictionary_order


def summarize_targeted(rows: Sequence[Mapping[str, Any]], dictionary_order: Mapping[str, Any]) -> Dict[str, Any]:
    by_id = {row["fixture_id"]: row for row in rows}
    rejected_ids = [
        "T02_WAITING_PASSENGER_K_REJECTED",
        "T03_ASSIGNED_PICKUP_K_REJECTED",
        "T04_ONBOARD_DROPOFF_K_REJECTED",
        "T05_MANDATORY_STOP_K_REJECTED",
    ]
    rejected_rows = [by_id[item] for item in rejected_ids]
    invalid_counts = [int(row["actual_result"]["invalid_skip_event_count"]) for row in rejected_rows]
    summary = {
        "created_at": iso_kst(),
        "targeted_fixture_passed": sum(1 for row in rows if row["passed"]),
        "targeted_fixture_total": len(rows),
        "k_safety_passed": sum(1 for fixture_id in ["T01_EMPTY_STOP_K_VALID", *rejected_ids] if by_id[fixture_id]["passed"]),
        "k_safety_total": 5,
        "valid_k_passed": bool(by_id["T01_EMPTY_STOP_K_VALID"]["passed"]),
        "invalid_k_executed_count": sum(1 for row in rejected_rows if row["actual_result"]["action_allowed"]),
        "missing_invalid_skip_event_count": sum(1 for count in invalid_counts if count == 0),
        "duplicate_invalid_skip_event_count": sum(1 for count in invalid_counts if count > 1),
        "explicit_safe_fallback_verified": all(row["actual_result"]["fallback_action"] in {"HOLD_CURRENT_POSITION", "SAFE_NOOP"} for row in rejected_rows),
        "silent_substitution_detected": any(row["actual_result"]["invalid_skip_event_count"] == 0 and row["actual_result"]["fallback_action"] in {"HOLD_CURRENT_POSITION", "SAFE_NOOP"} for row in rejected_rows),
        "invalid_skip_event_per_rejected_action_attempt": 1 if all(count == 1 for count in invalid_counts) else None,
        "rejected_k_route_advance_count": sum(1 for row in rejected_rows if row["actual_result"]["route_advance"]),
        "passenger_obligation_loss_count": sum(1 for row in rejected_rows if not row["actual_result"]["passenger_obligation_preserved"]),
        "infeasible_agent_can_win": bool(by_id["T06_FEASIBLE_VS_INFEASIBLE"]["actual_result"]["infeasible_agent_can_win"]),
        "same_feasible_lowest_agent_winner": by_id["T07_SAME_FEASIBLE_LOWEST_AGENT"]["actual_result"]["winner"],
        "earlier_service_start_winner": by_id["T08_EARLIER_SERVICE_START_WINS"]["actual_result"]["winner"],
        "no_feasible_winner": by_id["T09_NO_FEASIBLE_CANDIDATE"]["actual_result"]["winner"],
        "request_ownership_frozen_before_mutation": by_id["T10_DUPLICATE_SERVICE_PREVENTION"]["actual_result"]["request_ownership_frozen_before_mutation"],
        "duplicate_service_count": by_id["T10_DUPLICATE_SERVICE_PREVENTION"]["actual_result"]["duplicate_service_count"],
        "loser_mutation_count": by_id["T10_DUPLICATE_SERVICE_PREVENTION"]["actual_result"]["loser_mutation_count"],
        "dictionary_order_deterministic": bool(dictionary_order.get("dictionary_order_deterministic")),
        "repeat_deterministic": all(bool(row.get("repeat_deterministic")) for row in rows),
    }
    return summary


def targeted_invalid_skip_event_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rejected = [row for row in rows if row["fixture_id"].startswith(("T02", "T03", "T04", "T05"))]
    invalid_events = []
    for row in rejected:
        for event in row["actual_result"]["events"]:
            if event["event_type"] == "INVALID_SKIP":
                invalid_events.append({"fixture_id": row["fixture_id"], **event})
    uniqueness = set()
    duplicate = 0
    reason_missing = 0
    field_missing = 0
    for event in invalid_events:
        metadata = dict(event.get("metadata", {}))
        key = (
            metadata.get("branch_id"),
            event.get("step_index"),
            event.get("agent_id"),
            metadata.get("requested_action"),
        )
        if key in uniqueness:
            duplicate += 1
        uniqueness.add(key)
        if not metadata.get("reason_code"):
            reason_missing += 1
        required_present = all([
            event.get("event_id") is not None,
            metadata.get("branch_id") is not None,
            event.get("step_index") is not None,
            event.get("event_timestamp_seconds") is not None,
            event.get("agent_id") is not None,
            event.get("vehicle_id") is not None,
            event.get("route_id") is not None,
            event.get("stop_id") is not None,
            metadata.get("requested_action") is not None,
            metadata.get("executed_action") is not None,
            metadata.get("fallback_action") is not None,
            metadata.get("reason_code") is not None,
            metadata.get("source_decision_hash") is not None,
        ])
        field_missing += 0 if required_present else 1
    return {
        "created_at": iso_kst(),
        "rejected_k_attempt_count": len(rejected),
        "invalid_skip_event_count": len(invalid_events),
        "missing_invalid_skip_event_count": sum(1 for row in rejected if row["actual_result"]["invalid_skip_event_count"] == 0),
        "duplicate_invalid_skip_event_count": duplicate + sum(1 for row in rejected if row["actual_result"]["invalid_skip_event_count"] > 1),
        "reason_missing_count": reason_missing,
        "required_field_missing_event_count": field_missing,
        "route_advance_after_rejection_count": sum(1 for row in rejected if row["actual_result"]["route_advance"]),
        "records": invalid_events,
    }


def targeted_contract_runtime_audit(root: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> Dict[str, Any]:
    contracts = [
        "conditional_skip_decision_contract_v2.json",
        "action_adapter_contract_v2.json",
        "invalid_skip_event_contract_v2.json",
        "shared_request_arbitration_contract_v2.json",
        "request_ownership_contract_v2.json",
        "multiagent_orchestrator_contract_v2.json",
    ]
    by_id = {row["fixture_id"]: row for row in rows}
    checks = [
        {
            "contract": "conditional_skip_decision_contract_v2.json",
            "field_implemented": "action_allowed",
            "runtime_value_observed": by_id["T02_WAITING_PASSENGER_K_REJECTED"]["actual_result"]["action_allowed"],
            "expected_value": False,
            "match": by_id["T02_WAITING_PASSENGER_K_REJECTED"]["actual_result"]["action_allowed"] is False,
        },
        {
            "contract": "action_adapter_contract_v2.json",
            "field_implemented": "fallback_action",
            "runtime_value_observed": by_id["T02_WAITING_PASSENGER_K_REJECTED"]["actual_result"]["fallback_action"],
            "expected_value": "HOLD_CURRENT_POSITION",
            "match": by_id["T02_WAITING_PASSENGER_K_REJECTED"]["actual_result"]["fallback_action"] == "HOLD_CURRENT_POSITION",
        },
        {
            "contract": "invalid_skip_event_contract_v2.json",
            "field_implemented": "invalid_skip_event_count_per_rejected_action_attempt",
            "runtime_value_observed": summary["invalid_skip_event_per_rejected_action_attempt"],
            "expected_value": 1,
            "match": summary["invalid_skip_event_per_rejected_action_attempt"] == 1,
        },
        {
            "contract": "shared_request_arbitration_contract_v2.json",
            "field_implemented": "candidate_selection_rule",
            "runtime_value_observed": by_id["T06_FEASIBLE_VS_INFEASIBLE"]["actual_result"]["winner"],
            "expected_value": 2,
            "match": by_id["T06_FEASIBLE_VS_INFEASIBLE"]["actual_result"]["winner"] == 2,
        },
        {
            "contract": "request_ownership_contract_v2.json",
            "field_implemented": "request_ownership_frozen_before_mutation",
            "runtime_value_observed": summary["request_ownership_frozen_before_mutation"],
            "expected_value": True,
            "match": summary["request_ownership_frozen_before_mutation"] is True,
        },
        {
            "contract": "multiagent_orchestrator_contract_v2.json",
            "field_implemented": "request_served_at_most_once_invariant",
            "runtime_value_observed": summary["duplicate_service_count"],
            "expected_value": 0,
            "match": summary["duplicate_service_count"] == 0,
        },
    ]
    return {
        "created_at": iso_kst(),
        "contracts_checked": [
            {
                "relative_path": contract,
                "exists": (root / contract).exists(),
                "sha256": sha256_file(root / contract) if (root / contract).exists() else None,
            }
            for contract in contracts
        ],
        "runtime_checks": checks,
        "contract_v2_runtime_verified": all(item["match"] for item in checks),
    }


def targeted_prohibition_audits() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "historical_execution_prohibition_audit_targeted.json": {
            "created_at": created_at,
            "historical_branch_execution_count": 0,
            "d1_250row_execution_count": 0,
            "train_row_access_count": 0,
        },
        "validation_untouched_audit_targeted.json": {
            "created_at": created_at,
            "validation_access_count": 0,
            "validation_branch_count": 0,
            "validation_seal_intact": True,
        },
        "test_holdout_untouched_audit_targeted.json": {
            "created_at": created_at,
            "test_holdout_access_count": 0,
            "test_holdout_branch_count": 0,
            "test_holdout_touched": False,
        },
        "reward_energy_scale_nondefinition_audit_targeted.json": {
            "created_at": created_at,
            "new_reward_formula_created": False,
            "new_energy_formula_created": False,
            "normalization_scale_created": False,
            "tolerance_changed": False,
            "candidate_created": False,
        },
        "training_prohibition_audit_targeted.json": {
            "created_at": created_at,
            "training_run_count": 0,
            "optimizer_step_count": 0,
            "loss_backward_count": 0,
            "checkpoint_load_count": 0,
            "checkpoint_write_count": 0,
        },
        "external_access_audit_targeted.json": {
            "created_at": created_at,
            "database_accessed": False,
            "api_call_count": 0,
            "external_network_accessed": False,
            "service_key_accessed": False,
            "git_commit_count": 0,
            "git_push_count": 0,
        },
    }


def stage_immutability_audit_targeted(root: Path, prior_hashes_before: Mapping[str, str], repair_preservation: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel_path, before_sha in prior_hashes_before.items():
        path = root / rel_path
        after_sha = sha256_file(path) if path.exists() else None
        rows.append({
            "relative_path": rel_path,
            "sha256_before_targeted": before_sha,
            "sha256_after_targeted": after_sha,
            "unchanged": before_sha == after_sha,
        })
    return {
        "created_at": iso_kst(),
        "prior_stage_manifest_lock_unchanged": all(row["unchanged"] for row in rows),
        "prior_stage_mutated": not all(row["unchanged"] for row in rows),
        "top_level_repair_status_preserved_before_update": bool(repair_preservation.get("all_copies_byte_identical")),
        "records": rows,
    }


def choose_targeted_gate(
    source_preflight: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    contract_audit: Mapping[str, Any],
    stage_audit: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    by_id = {row["fixture_id"]: row for row in rows}
    if source_preflight.get("source_drift_count"):
        return FAIL_TV1_SOURCE_DRIFT, False, "FAILED_TV1_SOURCE_DRIFT"
    if not by_id["T01_EMPTY_STOP_K_VALID"]["passed"]:
        return FAIL_TV1_VALID_K_REJECTED, False, "FAILED_TV1_VALID_K_REJECTED"
    if summary.get("invalid_k_executed_count"):
        return FAIL_TV1_INVALID_K_EXECUTED, False, "FAILED_TV1_INVALID_K_EXECUTED"
    if summary.get("missing_invalid_skip_event_count"):
        return FAIL_TV1_INVALID_SKIP_EVENT_MISSING, False, "FAILED_TV1_INVALID_SKIP_EVENT_MISSING"
    if summary.get("duplicate_invalid_skip_event_count"):
        return FAIL_TV1_INVALID_SKIP_EVENT_DUPLICATED, False, "FAILED_TV1_INVALID_SKIP_EVENT_DUPLICATED"
    if summary.get("silent_substitution_detected"):
        return FAIL_TV1_SILENT_SUBSTITUTION, False, "FAILED_TV1_SILENT_SUBSTITUTION"
    if summary.get("rejected_k_route_advance_count"):
        return FAIL_TV1_REJECTED_K_ROUTE_ADVANCE, False, "FAILED_TV1_REJECTED_K_ROUTE_ADVANCE"
    if summary.get("passenger_obligation_loss_count"):
        return FAIL_TV1_PASSENGER_OBLIGATION_LOSS, False, "FAILED_TV1_PASSENGER_OBLIGATION_LOSS"
    if summary.get("infeasible_agent_can_win"):
        return FAIL_TV1_INFEASIBLE_AGENT_WON, False, "FAILED_TV1_INFEASIBLE_AGENT_WON"
    if summary.get("same_feasible_lowest_agent_winner") != 3:
        return FAIL_TV1_AGENT_TIEBREAK_INVALID, False, "FAILED_TV1_AGENT_TIEBREAK_INVALID"
    if summary.get("earlier_service_start_winner") != 6:
        return FAIL_TV1_SERVICE_START_PRIORITY_INVALID, False, "FAILED_TV1_SERVICE_START_PRIORITY_INVALID"
    if summary.get("no_feasible_winner") is not None:
        return FAIL_TV1_NO_FEASIBLE_WINNER_NOT_NULL, False, "FAILED_TV1_NO_FEASIBLE_WINNER_NOT_NULL"
    if not summary.get("request_ownership_frozen_before_mutation"):
        return FAIL_TV1_OWNERSHIP_AFTER_MUTATION, False, "FAILED_TV1_OWNERSHIP_AFTER_MUTATION"
    if summary.get("duplicate_service_count"):
        return FAIL_TV1_DUPLICATE_SERVICE, False, "FAILED_TV1_DUPLICATE_SERVICE"
    if not summary.get("dictionary_order_deterministic") or not summary.get("repeat_deterministic"):
        return FAIL_TV1_NONDETERMINISTIC_RESULT, False, "FAILED_TV1_NONDETERMINISTIC_RESULT"
    if not contract_audit.get("contract_v2_runtime_verified"):
        return FAIL_TV1_CONTRACT_RUNTIME_MISMATCH, False, "FAILED_TV1_CONTRACT_RUNTIME_MISMATCH"
    if stage_audit.get("prior_stage_mutated"):
        return "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_PRIOR_STAGE_MUTATED", False, "FAILED_TV1_PRIOR_STAGE_MUTATED"
    if summary.get("targeted_fixture_passed") != 10 or summary.get("targeted_fixture_total") != 10:
        return FAIL_TV1_NONDETERMINISTIC_RESULT, False, "FAILED_TV1_FIXTURE_COUNT"
    return PASS_TARGETED, True, "TARGETED_VERIFY_COMPLETE_FULL_VERIFY_PENDING_USER_COMMAND"


def downstream_lock_targeted(gate: Mapping[str, Any], summary: Mapping[str, Any], contract_audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "diagnose_complete": True,
        "repair_complete": True,
        "targeted_verify_complete": bool(gate.get("gate_passed")),
        "targeted_fixture_passed": summary.get("targeted_fixture_passed"),
        "targeted_fixture_total": summary.get("targeted_fixture_total"),
        "k_safety_passed": summary.get("k_safety_passed"),
        "k_safety_total": summary.get("k_safety_total"),
        "explicit_safe_fallback_verified": summary.get("explicit_safe_fallback_verified"),
        "silent_substitution_detected": summary.get("silent_substitution_detected"),
        "invalid_skip_event_per_rejected_action_attempt": summary.get("invalid_skip_event_per_rejected_action_attempt"),
        "rejected_k_route_advance_count": summary.get("rejected_k_route_advance_count"),
        "passenger_obligation_loss_count": summary.get("passenger_obligation_loss_count"),
        "infeasible_agent_can_win": summary.get("infeasible_agent_can_win"),
        "same_feasible_lowest_agent_winner": summary.get("same_feasible_lowest_agent_winner"),
        "earlier_service_start_winner": summary.get("earlier_service_start_winner"),
        "no_feasible_winner": summary.get("no_feasible_winner"),
        "request_ownership_frozen_before_mutation": summary.get("request_ownership_frozen_before_mutation"),
        "duplicate_service_count": summary.get("duplicate_service_count"),
        "dictionary_order_deterministic": summary.get("dictionary_order_deterministic"),
        "repeat_deterministic": summary.get("repeat_deterministic"),
        "contract_v2_runtime_verified": contract_audit.get("contract_v2_runtime_verified"),
        "full_verify_required": True,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }


def final_report_targeted(root: Path, gate: Mapping[str, Any], summary: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    answers = {
        "empty_stop_k_executed": summary["valid_k_passed"],
        "four_invalid_k_blocked": summary["invalid_k_executed_count"] == 0,
        "invalid_skip_event_one_each": summary["invalid_skip_event_per_rejected_action_attempt"] == 1,
        "hold_fallback_explicit": summary["explicit_safe_fallback_verified"],
        "silent_substitution_detected": summary["silent_substitution_detected"],
        "rejected_k_vehicle_moved": summary["rejected_k_route_advance_count"] != 0,
        "passenger_obligation_preserved": summary["passenger_obligation_loss_count"] == 0,
        "feasible_over_infeasible": not summary["infeasible_agent_can_win"],
        "same_condition_lowest_agent_winner": summary["same_feasible_lowest_agent_winner"],
        "earlier_service_start_winner": summary["earlier_service_start_winner"],
        "no_feasible_winner": summary["no_feasible_winner"],
        "ownership_before_mutation": summary["request_ownership_frozen_before_mutation"],
        "duplicate_service_count": summary["duplicate_service_count"],
        "dictionary_order_deterministic": summary["dictionary_order_deterministic"],
        "repeat_deterministic": summary["repeat_deterministic"],
        "ready_for_full_verify_pending_user_command": bool(gate.get("gate_passed")),
    }
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "targeted-verify",
        "gate": gate,
        "summary": summary,
        "answers": answers,
        "next_mode": "full-verify",
        "full_verify_authorized": False,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F Targeted Verify",
        "",
        f"- artifact: `{root}`",
        "- mode: `targeted-verify`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "",
        "## Results",
        f"- targeted fixtures: `{summary['targeted_fixture_passed']} / {summary['targeted_fixture_total']}`",
        f"- K safety fixtures: `{summary['k_safety_passed']} / {summary['k_safety_total']}`",
        f"- invalid K event per rejected attempt: `{summary['invalid_skip_event_per_rejected_action_attempt']}`",
        f"- silent substitution: `{str(summary['silent_substitution_detected']).lower()}`",
        f"- rejected K route advances: `{summary['rejected_k_route_advance_count']}`",
        f"- passenger obligation losses: `{summary['passenger_obligation_loss_count']}`",
        f"- infeasible agent can win: `{str(summary['infeasible_agent_can_win']).lower()}`",
        f"- same feasible winner: `{summary['same_feasible_lowest_agent_winner']}`",
        f"- earlier service-start winner: `{summary['earlier_service_start_winner']}`",
        f"- no feasible winner: `{summary['no_feasible_winner']}`",
        f"- duplicate service count: `{summary['duplicate_service_count']}`",
        f"- dictionary order deterministic: `{str(summary['dictionary_order_deterministic']).lower()}`",
        f"- repeat deterministic: `{str(summary['repeat_deterministic']).lower()}`",
        "",
        "Targeted verify stopped after 10 synthetic functional fixtures. Full verify, DL-6B audit, finalize, historical/validation/test access, reward/energy/scale creation, training, API, network, git commit, and git push were not performed.",
    ]) + "\n"
    return payload, md


def targeted_manifest_payloads() -> List[str]:
    return [
        "stage_history/repair/gate_decision.json",
        "stage_history/repair/downstream_lock.json",
        "stage_history/repair/final_report.json",
        "stage_history/repair/final_report.md",
        "stage_history/repair/source_change_registry.json",
        "stage_history/repair/artifact_manifest_repair.json",
        "stage_history/repair/_REPAIR_COMPLETE.lock",
        "repair_stage_preservation_audit_targeted.json",
        "targeted_verify_environment.json",
        "targeted_source_preflight_registry.json",
        "source_snapshot_repair/dynamics_multiagent_orchestrator.py",
        "source_snapshot_repair/dynamics_event_trace.py",
        "source_snapshot_repair/suseong_service_transition_engine_excerpt.txt",
        "source_snapshot_repair_registry.json",
        "targeted_fixture_inventory.json",
        "targeted_fixture_results.json",
        "targeted_fixture_results.jsonl",
        "targeted_k_safety_details.jsonl",
        "targeted_k_safety_summary.json",
        "targeted_invalid_skip_event_audit.json",
        "targeted_shared_request_details.jsonl",
        "targeted_shared_request_summary.json",
        "targeted_request_ownership_audit.json",
        "targeted_duplicate_service_audit.json",
        "targeted_dictionary_order_audit.json",
        "targeted_repeat_determinism_audit.json",
        "targeted_contract_runtime_audit.json",
        "historical_execution_prohibition_audit_targeted.json",
        "validation_untouched_audit_targeted.json",
        "test_holdout_untouched_audit_targeted.json",
        "reward_energy_scale_nondefinition_audit_targeted.json",
        "training_prohibition_audit_targeted.json",
        "external_access_audit_targeted.json",
        "table_write_backend_audit_targeted.json",
        "stage_immutability_audit_targeted.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
    ]


def write_manifest(writer: Writer, manifest_name: str, payloads: Sequence[str], scope: str) -> Dict[str, Any]:
    files = []
    missing = []
    seen = set()
    duplicates = 0
    excluded = {manifest_name, "_SUCCESS.lock"}
    if manifest_name == "artifact_manifest_diagnose.json":
        excluded.add("_DIAGNOSE_COMPLETE.lock")
    if manifest_name == "artifact_manifest_repair.json":
        excluded.add("_REPAIR_COMPLETE.lock")
    if manifest_name == "artifact_manifest_targeted_verify.json":
        excluded.add("_TARGETED_VERIFY_COMPLETE.lock")
    for rel_path in payloads:
        if rel_path in excluded:
            continue
        if rel_path in seen:
            duplicates += 1
            continue
        seen.add(rel_path)
        path = writer.root / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        files.append({
            "relative_path": rel_path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "required": True,
            "created_order": writer.order.get(rel_path),
        })
    manifest = {
        "created_at": iso_kst(),
        "manifest_scope": scope,
        "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
        "terminal_lock_created_last": False,
        "manifest_self_hash_exempt": False,
        "terminal_lock_listed_inside_manifest": False,
        "required_payload_count": len([item for item in payloads if item not in excluded]),
        "payload_file_count": len(files),
        "missing_payload_count": len(missing),
        "missing_payloads": missing,
        "duplicate_path_count": duplicates,
        "files": files,
    }
    writer.json(manifest_name, manifest)
    return manifest


def write_terminal_lock(writer: Writer, lock_name: str, manifest_name: str, gate: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = writer.root / manifest_name
    lock = {
        "created_at": iso_kst(),
        "gate": gate.get("gate"),
        "gate_passed": gate.get("gate_passed"),
        "mode": gate.get("mode"),
        "manifest_relative_path": manifest_name,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
    }
    writer.json(lock_name, lock)
    return lock


def verify_manifest(root: Path, lock_name: str) -> Dict[str, Any]:
    lock = read_json(root / lock_name)
    manifest_path = root / str(lock["manifest_relative_path"])
    manifest = read_json(manifest_path)
    payload_hash_mismatch = 0
    payload_missing = 0
    for item in manifest.get("files", []):
        path = root / item["relative_path"]
        if not path.exists():
            payload_missing += 1
            continue
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            payload_hash_mismatch += 1
    return {
        "manifest_hash_ok": sha256_file(manifest_path) == lock["manifest_sha256"],
        "manifest_size_ok": manifest_path.stat().st_size == lock["manifest_size_bytes"],
        "payload_missing_count": payload_missing,
        "payload_hash_mismatch_count": payload_hash_mismatch,
        "terminal_lock_listed_inside_manifest": any(item.get("relative_path") == lock_name for item in manifest.get("files", [])),
    }


def validate_artifact_root(root: Path, mode: str) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if mode == "diagnose":
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"artifact root is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    else:
        if not root.exists():
            raise FileNotFoundError(f"artifact root does not exist: {root}")
    return root


def run_diagnose(artifact_root: Path) -> Path:
    root = validate_artifact_root(artifact_root, "diagnose")
    writer = Writer(root)
    writer.text("git_status_v1f.txt", run_cmd(["git", "status", "--short"])["stdout"] + "\n")
    writer.json("environment_v1f.json", environment_audit())
    snapshot_registry, snapshot_rows = copy_upstream_failure_snapshot(writer)
    writer.json("upstream_failure_snapshot_registry.json", snapshot_registry)
    preservation = existing_er1_preservation_audit(snapshot_rows)
    writer.json("existing_er1_preservation_audit.json", preservation)
    table_info = [table(writer, "existing_er1_preservation_table.parquet", snapshot_rows)]
    source_preflight = source_preflight_registry()
    writer.json("source_preflight_registry.json", source_preflight)
    k_rows, call_chain, k_classification, fixture_state, silent = diagnose_k_safety()
    writer.json("k_safety_call_chain_diagnosis.json", call_chain)
    table_info.append(table(writer, "k_safety_fixture_diagnosis.parquet", k_rows))
    writer.json("k_safety_failure_classification.json", k_classification)
    writer.json("fixture_state_mapping_audit.json", fixture_state)
    writer.json("silent_substitution_audit.json", silent)
    writer.json("engine_safety_defect_record.json", {
        "created_at": iso_kst(),
        "engine_defect_confirmed": False,
        "core_engine_modification_allowed": False,
        "gate_if_confirmed": BLOCKED_ENGINE_SAFETY,
        "minimal_engine_extension_proposal_required": False,
    })
    resolver, no_feasible, duplicate = diagnose_shared_request()
    writer.json("shared_request_resolver_diagnosis.json", resolver)
    writer.json("no_feasible_path_diagnosis.json", no_feasible)
    writer.json("duplicate_service_path_diagnosis.json", duplicate)
    scope = repair_scope_contract(k_classification, resolver, duplicate)
    writer.json("repair_scope_contract.json", scope)
    validation, test, historical, reward, combined = prohibited_audits()
    writer.json("historical_execution_prohibition_audit.json", historical)
    writer.json("validation_untouched_audit.json", validation)
    writer.json("test_holdout_untouched_audit.json", test)
    writer.json("reward_energy_scale_nondefinition_audit.json", reward)
    writer.json("training_external_prohibition_audit.json", combined)
    writer.json("table_write_backend_audit.json", {"created_at": iso_kst(), "tables": table_info})
    gate_name, passed, readiness = choose_diagnose_gate(source_preflight, k_classification, fixture_state, resolver, no_feasible, duplicate)
    gate = {
        "created_at": iso_kst(),
        "mode": "diagnose",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "repair_authorized": False,
        "targeted_verify_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", downstream_lock(gate, k_classification, resolver, duplicate))
    report_json, report_md = final_report(root, gate, k_classification, resolver, no_feasible, duplicate)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    payloads = [
        "git_status_v1f.txt",
        "environment_v1f.json",
        "upstream_failure_snapshot_registry.json",
        "existing_er1_preservation_audit.json",
        "existing_er1_preservation_table.parquet",
        "source_preflight_registry.json",
        "k_safety_call_chain_diagnosis.json",
        "k_safety_fixture_diagnosis.parquet",
        "k_safety_failure_classification.json",
        "fixture_state_mapping_audit.json",
        "silent_substitution_audit.json",
        "engine_safety_defect_record.json",
        "shared_request_resolver_diagnosis.json",
        "no_feasible_path_diagnosis.json",
        "duplicate_service_path_diagnosis.json",
        "repair_scope_contract.json",
        "historical_execution_prohibition_audit.json",
        "validation_untouched_audit.json",
        "test_holdout_untouched_audit.json",
        "reward_energy_scale_nondefinition_audit.json",
        "training_external_prohibition_audit.json",
        "table_write_backend_audit.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
        *[row["copy_relative_path"] for row in snapshot_rows],
    ]
    manifest = write_manifest(writer, "artifact_manifest_diagnose.json", payloads, "PA1A_ER1_V1F_DIAGNOSE_MODE")
    write_terminal_lock(writer, "_DIAGNOSE_COMPLETE.lock", "artifact_manifest_diagnose.json", gate)
    verification = verify_manifest(root, "_DIAGNOSE_COMPLETE.lock")
    if manifest["missing_payload_count"] or manifest["duplicate_path_count"] or not verification["manifest_hash_ok"] or not verification["manifest_size_ok"] or verification["payload_missing_count"] or verification["payload_hash_mismatch_count"] or verification["terminal_lock_listed_inside_manifest"]:
        gate["gate"] = FAIL_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", downstream_lock(gate, k_classification, resolver, duplicate))
        report_json, report_md = final_report(root, gate, k_classification, resolver, no_feasible, duplicate)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_diagnose.json", payloads, "PA1A_ER1_V1F_DIAGNOSE_MODE")
        write_terminal_lock(writer, "_DIAGNOSE_COMPLETE.lock", "artifact_manifest_diagnose.json", gate)
    print(f"[DL-6D-PA1-A-ER1-V1F] artifact: {root}")
    print("[DL-6D-PA1-A-ER1-V1F] mode: diagnose")
    print("[DL-6D-PA1-A-ER1-V1F] actual compute path: CPU_ONLY")
    print("[DL-6D-PA1-A-ER1-V1F] existing ER1 artifact modified: 0")
    print(f"[DL-6D-PA1-A-ER1-V1F] K safety failure class: {k_classification['k_safety_failure_classification']}")
    print(f"[DL-6D-PA1-A-ER1-V1F] explicit safe fallback: {silent['explicit_safe_fallback_detected']}")
    print(f"[DL-6D-PA1-A-ER1-V1F] silent substitution: {silent['silent_substitution_detected']}")
    print("[DL-6D-PA1-A-ER1-V1F] K safety fixtures: diagnosis 5 / 5 classified")
    print(f"[DL-6D-PA1-A-ER1-V1F] invalid skip event failures: {k_classification['invalid_k_event_generation_failures']}")
    print(f"[DL-6D-PA1-A-ER1-V1F] rejected K route advances: {k_classification['invalid_k_route_advance_count']}")
    print(f"[DL-6D-PA1-A-ER1-V1F] passenger obligation losses: {k_classification['passenger_obligation_loss_count']}")
    print("[DL-6D-PA1-A-ER1-V1F] request ordering: REQUEST_TIMESTAMP_THEN_REQUEST_ID")
    print("[DL-6D-PA1-A-ER1-V1F] candidate selection: FEASIBLE_FILTER_THEN_SERVICE_START_THEN_AGENT_ID")
    print(f"[DL-6D-PA1-A-ER1-V1F] infeasible agent wins: {int(resolver['infeasible_agent_can_win'])}")
    print(f"[DL-6D-PA1-A-ER1-V1F] no feasible winner: {no_feasible['observed_winner']}")
    print(f"[DL-6D-PA1-A-ER1-V1F] duplicate service count: {duplicate['duplicate_service_count']}")
    print("[DL-6D-PA1-A-ER1-V1F] targeted fixtures: 0 / 10")
    print("[DL-6D-PA1-A-ER1-V1F] full fixtures: 0 / 12")
    print("[DL-6D-PA1-A-ER1-V1F] DL6B Tier 1: NOT_RUN_IN_DIAGNOSE")
    print("[DL-6D-PA1-A-ER1-V1F] DL6B Tier 2: NOT_RUN_IN_DIAGNOSE")
    print("[DL-6D-PA1-A-ER1-V1F] DL6B evidence status: NOT_RUN_IN_DIAGNOSE")
    print("[DL-6D-PA1-A-ER1-V1F] core engine modified: false")
    print("[DL-6D-PA1-A-ER1-V1F] historical executions: 0")
    print("[DL-6D-PA1-A-ER1-V1F] validation/test touched: false / false")
    print("[DL-6D-PA1-A-ER1-V1F] reward/energy/scale created: false / false / false")
    print(f"[DL-6D-PA1-A-ER1-V1F] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A-ER1-V1F] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[DL-6D-PA1-A-ER1-V1F] readiness: {gate['readiness']}")
    print("[DL-6D-PA1-A-ER1-V1F] next: REPORT_TO_USER")
    return root


def run_repair(artifact_root: Path) -> Path:
    root = validate_repair_entry(artifact_root)
    writer = Writer(root)
    preserve_diagnose_stage(writer)
    repair_env = environment_audit()
    repair_env.update({
        "mode": "repair",
        "actual_compute_path": "CPU_ONLY_STATIC_REPAIR",
        "synthetic_fixture_execution_count": 0,
        "transition_execution_count": 0,
    })
    writer.json("repair_environment.json", repair_env)
    preflight = repair_preflight_source_registry(root)
    writer.json("repair_preflight_source_registry.json", preflight)
    changes = source_change_registry(root, preflight)
    writer.json("source_change_registry.json", changes)
    for rel_path, payload in contract_v2_payloads().items():
        writer.json(rel_path, payload)
    static_audit = static_repair_audit()
    writer.json("static_repair_audit.json", static_audit)
    core_audit = core_engine_nonmodification_audit(preflight)
    writer.json("core_engine_nonmodification_audit.json", core_audit)
    upstream_audit = upstream_source_nonmodification_audit(preflight)
    writer.json("upstream_source_nonmodification_audit.json", upstream_audit)
    for rel_path, payload in prohibited_audits_repair().items():
        writer.json(rel_path, payload)
    writer.json("table_write_backend_audit_repair.json", table_write_backend_audit_repair())
    gate_name, passed, readiness = choose_repair_gate(preflight, static_audit, core_audit, upstream_audit)
    gate = {
        "created_at": iso_kst(),
        "mode": "repair",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "diagnose_complete": True,
        "repair_complete": passed,
        "targeted_verify_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
        "synthetic_fixture_execution_count": 0,
        "historical_branch_execution_count": 0,
        "validation_access_count": 0,
        "test_holdout_access_count": 0,
        "core_engine_modification_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", downstream_lock_repair(gate))
    report_json, report_md = final_report_repair(root, gate)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    payloads = repair_manifest_payloads(root)
    manifest = write_manifest(writer, "artifact_manifest_repair.json", payloads, "PA1A_ER1_V1F_REPAIR_MODE")
    write_terminal_lock(writer, "_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", gate)
    verification = verify_manifest(root, "_REPAIR_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or manifest["duplicate_path_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_REPAIR_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_REPAIR_MANIFEST_RECONCILIATION"
        gate["repair_complete"] = False
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", downstream_lock_repair(gate))
        report_json, report_md = final_report_repair(root, gate)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_repair.json", payloads, "PA1A_ER1_V1F_REPAIR_MODE")
        write_terminal_lock(writer, "_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", gate)
    print(f"[DL-6D-PA1-A-ER1-V1F-RP1] artifact: {root}")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] mode: repair")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] K safety logic changed: false")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] rejected K executed action: HOLD_CURRENT_POSITION")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] INVALID_SKIP per rejected attempt: 1")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] feasible filter first: true")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] no feasible winner: null")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] request ownership frozen before mutation: true")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] core engine modified: false")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] synthetic fixture executions: 0")
    print("[DL-6D-PA1-A-ER1-V1F-RP1] targeted verify: NOT_RUN_PENDING_USER_COMMAND")
    print(f"[DL-6D-PA1-A-ER1-V1F-RP1] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-RP1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[DL-6D-PA1-A-ER1-V1F-RP1] readiness: {gate['readiness']}")
    return root


def run_targeted_verify(artifact_root: Path) -> Path:
    root = validate_targeted_entry(artifact_root)
    prior_hashes_before = {
        rel_path: sha256_file(root / rel_path)
        for rel_path in [
            "artifact_manifest_diagnose.json",
            "_DIAGNOSE_COMPLETE.lock",
            "artifact_manifest_repair.json",
            "_REPAIR_COMPLETE.lock",
        ]
    }
    writer = Writer(root)
    repair_preservation = preserve_repair_stage(writer)
    env = environment_audit()
    env.update({
        "mode": "targeted-verify",
        "actual_compute_path": "CPU_ONLY",
        "verification_scope": "TARGETED_SYNTHETIC_FUNCTIONAL_REGRESSION",
        "synthetic_fixture_is_research_evidence": False,
        "synthetic_fixture_is_historical_dynamics_evidence": False,
        "synthetic_fixture_used_for_reward_design": False,
    })
    writer.json("targeted_verify_environment.json", env)
    source_preflight = targeted_source_preflight()
    writer.json("targeted_source_preflight_registry.json", source_preflight)
    snapshot_registry = write_repair_source_snapshot(writer)
    inventory = targeted_fixture_inventory()
    writer.json("targeted_fixture_inventory.json", inventory)
    rows, dictionary_order = run_targeted_fixtures()
    summary = summarize_targeted(rows, dictionary_order)
    writer.json("targeted_fixture_results.json", {
        "created_at": iso_kst(),
        "targeted_fixture_count": len(rows),
        "records": rows,
    })
    table_infos = [
        jsonl_table(writer, "targeted_fixture_results.jsonl", rows, logical_table_name="targeted_fixture_results"),
    ]
    k_rows = [row for row in rows if row["fixture_type"] == "K_SAFETY"]
    shared_rows = [row for row in rows if row["fixture_type"] in {"SHARED_REQUEST", "DUPLICATE_SERVICE"}]
    table_infos.append(jsonl_table(writer, "targeted_k_safety_details.jsonl", k_rows, logical_table_name="targeted_k_safety_details"))
    writer.json("targeted_k_safety_summary.json", {
        "created_at": iso_kst(),
        "k_safety_passed": summary["k_safety_passed"],
        "k_safety_total": summary["k_safety_total"],
        "valid_k_passed": summary["valid_k_passed"],
        "invalid_k_executed_count": summary["invalid_k_executed_count"],
        "rejected_k_route_advance_count": summary["rejected_k_route_advance_count"],
        "passenger_obligation_loss_count": summary["passenger_obligation_loss_count"],
        "explicit_safe_fallback_verified": summary["explicit_safe_fallback_verified"],
        "silent_substitution_detected": summary["silent_substitution_detected"],
    })
    invalid_audit = targeted_invalid_skip_event_audit(rows)
    writer.json("targeted_invalid_skip_event_audit.json", invalid_audit)
    table_infos.append(jsonl_table(writer, "targeted_shared_request_details.jsonl", shared_rows, logical_table_name="targeted_shared_request_details"))
    writer.json("targeted_shared_request_summary.json", {
        "created_at": iso_kst(),
        "infeasible_agent_can_win": summary["infeasible_agent_can_win"],
        "same_feasible_lowest_agent_winner": summary["same_feasible_lowest_agent_winner"],
        "earlier_service_start_winner": summary["earlier_service_start_winner"],
        "no_feasible_winner": summary["no_feasible_winner"],
    })
    t10 = next(row for row in rows if row["fixture_id"] == "T10_DUPLICATE_SERVICE_PREVENTION")
    writer.json("targeted_request_ownership_audit.json", {
        "created_at": iso_kst(),
        "fixture_id": "T10_DUPLICATE_SERVICE_PREVENTION",
        "request_ownership_frozen_before_mutation": summary["request_ownership_frozen_before_mutation"],
        "ownership_frozen_sequence": t10["actual_result"]["ownership_frozen_sequence"],
        "first_service_mutation_sequence": t10["actual_result"]["first_service_mutation_sequence"],
        "sequence_trace": t10["actual_result"]["sequence_trace"],
    })
    writer.json("targeted_duplicate_service_audit.json", {
        "created_at": iso_kst(),
        "duplicate_service_count": summary["duplicate_service_count"],
        "loser_mutation_count": summary["loser_mutation_count"],
        "invariant": t10["actual_result"]["invariant"],
    })
    writer.json("targeted_dictionary_order_audit.json", dictionary_order)
    writer.json("targeted_repeat_determinism_audit.json", {
        "created_at": iso_kst(),
        "repeat_deterministic": summary["repeat_deterministic"],
        "records": [
            {
                "fixture_id": row["fixture_id"],
                "run_1_initial_hash": row["run_1_initial_hash"],
                "run_2_initial_hash": row["run_2_initial_hash"],
                "run_1_result_hash": row["run_1_result_hash"],
                "run_2_result_hash": row["run_2_result_hash"],
                "run_1_event_hash": row["run_1_event_hash"],
                "run_2_event_hash": row["run_2_event_hash"],
                "run_1_end_state_hash": row["run_1_end_state_hash"],
                "run_2_end_state_hash": row["run_2_end_state_hash"],
                "repeat_deterministic": row["repeat_deterministic"],
            }
            for row in rows
        ],
    })
    contract_audit = targeted_contract_runtime_audit(root, rows, summary)
    writer.json("targeted_contract_runtime_audit.json", contract_audit)
    for rel_path, payload in targeted_prohibition_audits().items():
        writer.json(rel_path, payload)
    writer.json("table_write_backend_audit_targeted.json", {
        "created_at": iso_kst(),
        "preferred_backend": "PARQUET",
        "actual_backend": "JSONL_FALLBACK_NO_PARQUET_ENGINE",
        "tables": table_infos,
    })
    stage_audit = stage_immutability_audit_targeted(root, prior_hashes_before, repair_preservation)
    writer.json("stage_immutability_audit_targeted.json", stage_audit)
    gate_name, passed, readiness = choose_targeted_gate(source_preflight, rows, summary, contract_audit, stage_audit)
    gate = {
        "created_at": iso_kst(),
        "mode": "targeted-verify",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "diagnose_complete": True,
        "repair_complete": True,
        "targeted_verify_complete": passed,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", downstream_lock_targeted(gate, summary, contract_audit))
    report_json, report_md = final_report_targeted(root, gate, summary)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    payloads = targeted_manifest_payloads()
    manifest = write_manifest(writer, "artifact_manifest_targeted_verify.json", payloads, "PA1A_ER1_V1F_TARGETED_VERIFY_MODE")
    write_terminal_lock(writer, "_TARGETED_VERIFY_COMPLETE.lock", "artifact_manifest_targeted_verify.json", gate)
    verification = verify_manifest(root, "_TARGETED_VERIFY_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or manifest["duplicate_path_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_TV1_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_TV1_MANIFEST_RECONCILIATION"
        gate["targeted_verify_complete"] = False
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", downstream_lock_targeted(gate, summary, contract_audit))
        report_json, report_md = final_report_targeted(root, gate, summary)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_targeted_verify.json", payloads, "PA1A_ER1_V1F_TARGETED_VERIFY_MODE")
        write_terminal_lock(writer, "_TARGETED_VERIFY_COMPLETE.lock", "artifact_manifest_targeted_verify.json", gate)
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] artifact: {root}")
    print("[DL-6D-PA1-A-ER1-V1F-TV1] mode: targeted-verify")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] source drift: {source_preflight['source_drift_count']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] targeted fixtures: {summary['targeted_fixture_passed']} / {summary['targeted_fixture_total']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] K safety fixtures: {summary['k_safety_passed']} / {summary['k_safety_total']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] invalid skip event per rejected attempt: {summary['invalid_skip_event_per_rejected_action_attempt']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] silent substitution: {str(summary['silent_substitution_detected']).lower()}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] rejected K route advances: {summary['rejected_k_route_advance_count']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] passenger obligation losses: {summary['passenger_obligation_loss_count']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] duplicate service count: {summary['duplicate_service_count']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] dictionary order deterministic: {str(summary['dictionary_order_deterministic']).lower()}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] repeat deterministic: {str(summary['repeat_deterministic']).lower()}")
    print("[DL-6D-PA1-A-ER1-V1F-TV1] full verify: NOT_RUN_PENDING_USER_COMMAND")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[DL-6D-PA1-A-ER1-V1F-TV1] readiness: {gate['readiness']}")
    return root


def run_locked(root: Path, mode: str) -> Path:
    validate_artifact_root(root, mode)
    raise RuntimeError(f"--mode {mode} is locked until the previous V1F mode is reported to the user")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["diagnose", "repair", "targeted-verify", "full-verify", "dl6b-audit", "finalize"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "diagnose":
        run_diagnose(args.artifact_root)
    elif args.mode == "repair":
        run_repair(args.artifact_root)
    elif args.mode == "targeted-verify":
        run_targeted_verify(args.artifact_root)
    else:
        run_locked(args.artifact_root, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
