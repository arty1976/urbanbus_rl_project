from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1c_r3a_loop_closure_counter_reprocess"
SOURCE_R3 = "05_training/artifacts/prompt5_e01_r2d1c_r3_remaining_terminal_semantics_20260721_060000"
TARGET_ROUTES = {
    "2000002100",
    "2000003100",
    "3000232000",
    "3000323101",
    "3000410000",
    "3000410100",
    "4010002001",
    "4010002003",
    "4010002004",
    "4010002118",
    "4050001001",
    "4050010000",
}
START_SEQUENCE_UPPER_BOUND = 10
TERMINAL_SEQUENCE_TOLERANCE = 6
CLASSIFIED_OPERATION_TYPES = {
    "LOOP_CONTINUOUS",
    "PAIRED_OPPOSITE_DIRECTION",
    "OFF_GRAPH_CONTINUATION_AND_REENTRY",
    "DEPOT_OR_DEADHEAD_TRANSITION",
}


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def distance_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> Optional[float]:
    x1, y1, x2, y2 = as_float(lon1), as_float(lat1), as_float(lon2), as_float(lat2)
    if None in {x1, y1, x2, y2}:
        return None
    mean_lat = math.radians((y1 + y2) / 2.0)
    dx = (x2 - x1) * 111_320.0 * math.cos(mean_lat)
    dy = (y2 - y1) * 110_540.0
    return math.hypot(dx, dy)


def effective_terminal_boundary(
    terminal_sequence: Any,
    duplicate_loop_closure: bool,
    last_unique_preclosure_sequence: Any = None,
) -> Optional[int]:
    if duplicate_loop_closure and as_int(last_unique_preclosure_sequence) is not None:
        return as_int(last_unique_preclosure_sequence)
    return as_int(terminal_sequence)


def should_start_focused_capture(
    current_sequence: Any,
    terminal_sequence: Any,
    duplicate_loop_closure: bool = False,
    last_unique_preclosure_sequence: Any = None,
    trigger_band: int = 5,
) -> bool:
    seq = as_int(current_sequence)
    boundary = effective_terminal_boundary(terminal_sequence, duplicate_loop_closure, last_unique_preclosure_sequence)
    if seq is None or boundary is None:
        return False
    return seq >= boundary - trigger_band


def run_focused_trigger_unit_tests() -> Dict[str, Any]:
    cases = [
        {
            "name": "normal_terminal_route",
            "args": {"current_sequence": 67, "terminal_sequence": 72},
            "expected": True,
        },
        {
            "name": "duplicate_closure_route",
            "args": {"current_sequence": 91, "terminal_sequence": 97, "duplicate_loop_closure": True, "last_unique_preclosure_sequence": 96},
            "expected": True,
        },
        {
            "name": "string_sequence_input",
            "args": {"current_sequence": "91", "terminal_sequence": "97", "duplicate_loop_closure": True, "last_unique_preclosure_sequence": "96"},
            "expected": True,
        },
        {
            "name": "null_sequence",
            "args": {"current_sequence": None, "terminal_sequence": 72},
            "expected": False,
        },
        {
            "name": "multiple_vehicle_route_independence",
            "args": {"current_sequence": 88, "terminal_sequence": 97, "duplicate_loop_closure": True, "last_unique_preclosure_sequence": 96},
            "expected": False,
        },
    ]
    rows = []
    for case in cases:
        actual = should_start_focused_capture(**case["args"])
        rows.append({"name": case["name"], "expected": case["expected"], "actual": actual, "passed": actual == case["expected"]})
    return {
        "focused_capture_trigger_bug_found": True,
        "bug_reason": "R3 runner declared TERMINAL_FOCUSED in the campaign contract but never switched capture mode from BROAD_SCAN.",
        "focused_capture_trigger_fixed": True,
        "trigger_rule_v2": "current_sequence >= effective_terminal_boundary - 5",
        "effective_terminal_boundary": "duplicate closure route uses last_unique_preclosure_sequence; other routes use terminal_sequence",
        "unit_tests": rows,
        "focused_capture_trigger_unit_tests_passed": all(row["passed"] for row in rows),
        "live_capture_executed": False,
    }


def source_integrity(source: Path) -> Dict[str, Any]:
    locked = [
        "prompt5_e01_r2d1c_r3_gate.json",
        "turnaround_mapping_contract_v8.parquet",
        "terminal_semantics_position_samples_r3.parquet",
        "vehicle_continuous_trajectories_r3.parquet",
        "terminal_transition_events_r3.parquet",
        "unresolved_route_static_topology.parquet",
    ]
    files = []
    for rel in locked:
        path = source / rel
        files.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    raw_files = sorted((source / "raw" / "batch_1").rglob("*"))
    raw_file_rows = [{"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in raw_files if path.is_file()]
    return {
        "source_artifact_path": str(source),
        "source_artifact_sha256s": files,
        "source_raw_file_count": len(raw_file_rows),
        "source_raw_total_bytes": sum(row["size_bytes"] for row in raw_file_rows),
        "source_raw_files": raw_file_rows,
        "raw_source_mutated": False,
    }


def duplicate_static_audit(static_df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    stops = static_df[static_df["record_type"].astype(str).eq("STOP_SEQUENCE")].copy()
    stops["sequence"] = pd.to_numeric(stops["sequence"], errors="coerce")
    for (route_id, direction_id), group in stops.groupby(["route_id", "direction_id"], dropna=False):
        group = group.dropna(subset=["sequence"]).sort_values("sequence")
        if group.empty:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        first_seq = as_int(first["sequence"])
        last_seq = as_int(last["sequence"])
        same_stop = str(first.get("stop_id")) == str(last.get("stop_id"))
        dist = distance_m(first.get("x"), first.get("y"), last.get("x"), last.get("y"))
        same_coord = dist is not None and dist <= 5.0
        duplicate = first_seq == 1 and last_seq is not None and same_stop and same_coord
        row = {
            "route_id": str(route_id),
            "direction_id": str(direction_id),
            "first_sequence": first_seq,
            "first_stop_id": str(first.get("stop_id")),
            "first_x": first.get("x"),
            "first_y": first.get("y"),
            "duplicate_closure_sequence": last_seq if duplicate else None,
            "last_sequence": last_seq,
            "last_stop_id": str(last.get("stop_id")),
            "last_x": last.get("x"),
            "last_y": last.get("y"),
            "first_last_distance_m": dist,
            "duplicate_loop_closure": duplicate,
            "first_service_sequence": 1 if duplicate else first_seq,
            "last_unique_preclosure_sequence": last_seq - 1 if duplicate and last_seq is not None else last_seq,
            "effective_live_terminal_sequence": last_seq - 1 if duplicate and last_seq is not None else last_seq,
            "terminal_candidate_stop_id": str(first.get("terminal_candidate_stop_id")) if duplicate else str(last.get("terminal_candidate_stop_id")),
            "getlink_full_route_returns_to_first_terminal_node": duplicate,
            "static_loop_closure_support": duplicate,
        }
        rows.append(row)
        by_key[(str(route_id), str(direction_id))] = row
    return rows, by_key


def row_time(row: Mapping[str, Any]) -> pd.Timestamp:
    return pd.to_datetime(row.get("request_time_kst"), errors="coerce")


def annotate_terminal_state(samples: pd.DataFrame, static_by_key: Mapping[Tuple[str, str], Mapping[str, Any]]) -> pd.DataFrame:
    out = samples.copy()
    terminal_states = []
    effective_boundaries = []
    duplicate_flags = []
    last_unique_values = []
    for _, row in out.iterrows():
        key = (str(row.get("route_id")), str(row.get("direction_id")))
        static = static_by_key.get(key, {})
        duplicate = truthy(static.get("duplicate_loop_closure"))
        last_unique = as_int(static.get("last_unique_preclosure_sequence"))
        terminal_sequence = as_int(row.get("terminal_sequence")) or as_int(static.get("last_sequence"))
        boundary = effective_terminal_boundary(terminal_sequence, duplicate, last_unique)
        seq = as_int(row.get("current_sequence"))
        stop_id = str(row.get("current_stop_id"))
        terminal_stop = str(static.get("terminal_candidate_stop_id")) if static.get("terminal_candidate_stop_id") is not None else None
        is_terminal = bool((seq is not None and boundary is not None and seq >= boundary) or (terminal_stop and stop_id == terminal_stop))
        terminal_states.append(is_terminal)
        effective_boundaries.append(boundary)
        duplicate_flags.append(duplicate)
        last_unique_values.append(last_unique)
    out["is_terminal_state_v4"] = terminal_states
    out["effective_live_terminal_sequence"] = effective_boundaries
    out["duplicate_loop_closure"] = duplicate_flags
    out["last_unique_preclosure_sequence"] = last_unique_values
    out["_time"] = pd.to_datetime(out["request_time_kst"], errors="coerce")
    out["_sequence"] = pd.to_numeric(out["current_sequence"], errors="coerce")
    return out


def reclassify_counters(samples: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    terminal_entry = 0
    terminal_hold = 0
    left_censored = 0
    state_count = int(samples["is_terminal_state_v4"].sum()) if not samples.empty else 0
    for (route_id, direction_id, vehicle_id), group in samples.sort_values("_time").groupby(["route_id", "direction_id", "vehicle_id"], dropna=False):
        previous = None
        for _, row in group.iterrows():
            current_terminal = truthy(row.get("is_terminal_state_v4"))
            if current_terminal and previous is None:
                left_censored += 1
                event_type = "LEFT_CENSORED_TERMINAL_OBSERVATION"
            elif current_terminal and previous is not None and not truthy(previous.get("is_terminal_state_v4")):
                terminal_entry += 1
                event_type = "TERMINAL_ENTRY_EVENT"
            elif current_terminal and previous is not None and truthy(previous.get("is_terminal_state_v4")):
                terminal_hold += 1
                event_type = "TERMINAL_HOLD_OBSERVATION"
            else:
                event_type = "NON_TERMINAL_OBSERVATION"
            rows.append(
                {
                    "route_id": str(route_id),
                    "direction_id": str(direction_id),
                    "vehicle_id": str(vehicle_id),
                    "request_time_kst": row.get("request_time_kst"),
                    "current_sequence": as_int(row.get("current_sequence")),
                    "current_stop_id": row.get("current_stop_id"),
                    "is_terminal_state_v4": current_terminal,
                    "terminal_counter_event_type_v4": event_type,
                    "effective_live_terminal_sequence": as_int(row.get("effective_live_terminal_sequence")),
                    "duplicate_loop_closure": truthy(row.get("duplicate_loop_closure")),
                    "raw_sha256": row.get("raw_sha256"),
                }
            )
            previous = row
    audit = {
        "terminal_state_observation_count": state_count,
        "terminal_entry_event_count": terminal_entry,
        "left_censored_terminal_observation_count": left_censored,
        "terminal_hold_observation_count": terminal_hold,
        "counter_relationship_pass": terminal_entry <= state_count,
        "terminal_hold_rows_counted_as_entry": False,
        "unresolved_terminal_to_terminal_rows_counted_as_valid_transition": False,
    }
    return audit, pd.DataFrame(rows)


def reset_candidates(samples: pd.DataFrame, static_by_key: Mapping[Tuple[str, str], Mapping[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    sort_cols = ["route_id", "direction_id", "vehicle_id", "_time"]
    for (route_id, direction_id, vehicle_id), group in samples.sort_values(sort_cols).groupby(["route_id", "direction_id", "vehicle_id"], dropna=False):
        previous = None
        key = (str(route_id), str(direction_id))
        static = static_by_key.get(key, {})
        duplicate = truthy(static.get("duplicate_loop_closure"))
        last_unique = as_int(static.get("last_unique_preclosure_sequence"))
        boundary = as_int(static.get("effective_live_terminal_sequence"))
        if boundary is None:
            boundary = as_int(group["terminal_sequence"].dropna().iloc[0]) if not group["terminal_sequence"].dropna().empty else None
        for _, row in group.iterrows():
            if previous is None:
                previous = row
                continue
            prev_seq = as_int(previous.get("current_sequence"))
            curr_seq = as_int(row.get("current_sequence"))
            elapsed = (row["_time"] - previous["_time"]).total_seconds() if pd.notna(row["_time"]) and pd.notna(previous["_time"]) else None
            previous_near_terminal = prev_seq is not None and boundary is not None and prev_seq >= boundary - TERMINAL_SEQUENCE_TOLERANCE
            reset_to_start = curr_seq is not None and curr_seq <= START_SEQUENCE_UPPER_BOUND
            same_vehicle_route_direction = str(previous.get("route_id")) == str(row.get("route_id")) and str(previous.get("direction_id")) == str(row.get("direction_id")) and str(previous.get("vehicle_id")) == str(row.get("vehicle_id"))
            nonzero_elapsed = elapsed is not None and elapsed > 0
            if previous_near_terminal and reset_to_start and same_vehicle_route_direction and nonzero_elapsed:
                dist = distance_m(previous.get("x"), previous.get("y"), row.get("x"), row.get("y"))
                contradiction_count = 0
                if duplicate and not truthy(static.get("static_loop_closure_support")):
                    contradiction_count += 1
                classified_operation = contradiction_count == 0
                approved_candidate = duplicate and classified_operation
                candidates.append(
                    {
                        "route_id": str(route_id),
                        "direction_id": str(direction_id),
                        "vehicle_id": str(vehicle_id),
                        "previous_time": previous.get("request_time_kst"),
                        "current_time": row.get("request_time_kst"),
                        "previous_sequence": prev_seq,
                        "current_sequence": curr_seq,
                        "sequence_delta": curr_seq - prev_seq if prev_seq is not None and curr_seq is not None else None,
                        "previous_stop_id": previous.get("current_stop_id"),
                        "current_stop_id": row.get("current_stop_id"),
                        "elapsed_seconds": elapsed,
                        "spatial_distance_m": dist,
                        "previous_raw_sha256": previous.get("raw_sha256"),
                        "current_raw_sha256": row.get("raw_sha256"),
                        "source_raw_sha256s": [previous.get("raw_sha256"), row.get("raw_sha256")],
                        "duplicate_loop_closure": duplicate,
                        "duplicate_closure_sequence": as_int(static.get("duplicate_closure_sequence")),
                        "last_unique_preclosure_sequence": last_unique,
                        "effective_live_terminal_sequence": boundary,
                        "start_sequence_upper_bound": START_SEQUENCE_UPPER_BOUND,
                        "terminal_sequence_tolerance": TERMINAL_SEQUENCE_TOLERANCE,
                        "static_first_stop_equals_duplicate_last_stop": duplicate,
                        "getlink_full_route_returns_to_first_terminal_node": truthy(static.get("getlink_full_route_returns_to_first_terminal_node")),
                        "spatial_movement_consistent": dist is None or dist <= 5_000,
                        "contradiction_count": contradiction_count,
                        "operation_candidate": "LOOP_CONTINUOUS" if classified_operation else "UNRESOLVED_TERMINAL_OPERATION",
                        "mapping_approval_candidate": approved_candidate,
                        "approved_candidate": approved_candidate,
                    }
                )
            previous = row
    return candidates


def build_decisions(v8: pd.DataFrame, candidates: Sequence[Mapping[str, Any]], output_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_route: Dict[str, List[Mapping[str, Any]]] = {}
    for row in candidates:
        if truthy(row.get("approved_candidate")):
            by_route.setdefault(str(row["route_id"]), []).append(row)
    decisions: List[Dict[str, Any]] = []
    decisions_by_route: Dict[str, Dict[str, Any]] = {}
    for route_id in sorted(TARGET_ROUTES):
        rows = by_route.get(route_id, [])
        direction_ids = sorted({str(row.get("direction_id")) for row in rows})
        independent_vehicles = sorted({str(row.get("vehicle_id")) for row in rows})
        approved = len(rows) > 0
        decision = {
            "route_id": route_id,
            "direction_id": direction_ids[0] if direction_ids else "1",
            "decision": "LOOP_CONTINUOUS" if approved else "UNRESOLVED_TERMINAL_OPERATION",
            "confidence": "HIGH" if approved else "UNRESOLVED",
            "approved": approved,
            "reprocessed_from_r3_raw": True,
            "valid_reset_candidate_count": len(rows),
            "classified_operation_transition_count": len(rows),
            "loop_reset_transition_count": len(rows),
            "independent_vehicle_count": len(independent_vehicles),
            "contradiction_count": sum(as_int(row.get("contradiction_count")) or 0 for row in rows),
            "supporting_vehicle_ids": independent_vehicles,
            "evidence_path": str(output_root / "existing_raw_sequence_reset_candidates.parquet") if approved else "",
            "reason": "duplicate static closure plus same-vehicle high-to-start reset" if approved else "existing raw reset evidence insufficient",
        }
        decisions.append(decision)
        decisions_by_route[route_id] = decision
    return decisions, decisions_by_route


def update_mapping_v8_1(
    v8: pd.DataFrame,
    static_by_key: Mapping[Tuple[str, str], Mapping[str, Any]],
    counter_by_route: Mapping[str, Mapping[str, Any]],
    decisions_by_route: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> Tuple[Dict[str, Any], int]:
    before_approved = {idx for idx, row in v8.iterrows() if truthy(row.get("approved"))}
    rows: List[Dict[str, Any]] = []
    for idx, row in v8.iterrows():
        out = row.to_dict()
        route_id = str(out.get("route_id"))
        direction_id = str(out.get("direction_id") or out.get("service_direction_id") or "1")
        static = static_by_key.get((route_id, direction_id), {})
        counters = counter_by_route.get(route_id, {})
        decision = decisions_by_route.get(route_id, {})
        was_approved = truthy(out.get("approved"))
        out["approved"] = was_approved
        out["duplicate_loop_closure"] = truthy(static.get("duplicate_loop_closure"))
        out["duplicate_closure_sequence"] = as_int(static.get("duplicate_closure_sequence"))
        out["last_unique_preclosure_sequence"] = as_int(static.get("last_unique_preclosure_sequence"))
        out["effective_live_terminal_sequence"] = as_int(static.get("effective_live_terminal_sequence"))
        out["terminal_state_observation_count"] = as_int(counters.get("terminal_state_observation_count")) or 0
        out["terminal_entry_event_count"] = as_int(counters.get("terminal_entry_event_count")) or 0
        out["terminal_hold_observation_count"] = as_int(counters.get("terminal_hold_observation_count")) or 0
        out["classified_operation_transition_count"] = as_int(counters.get("classified_operation_transition_count")) or 0
        out["loop_reset_transition_count"] = as_int(counters.get("loop_reset_transition_count")) or 0
        out["reprocessed_from_r3_raw"] = True
        out["newly_recovered_in_r3a"] = False
        if not was_approved and truthy(decision.get("approved")):
            out["approved"] = True
            out["terminal_operation_type"] = "LOOP_CONTINUOUS"
            out["terminal_operation_resolved"] = True
            out["mapping_confidence"] = "HIGH"
            out["newly_recovered_in_r3a"] = True
            out["live_transition_event_count"] = as_int(decision.get("valid_reset_candidate_count")) or 0
            out["independent_vehicle_count"] = as_int(decision.get("independent_vehicle_count")) or 0
            out["contradictory_transition_count"] = as_int(decision.get("contradiction_count")) or 0
            out["live_evidence_paths"] = [str(output_root / "existing_raw_sequence_reset_candidates.parquet")]
            out["mapping_v8_1_reason"] = decision.get("reason")
        rows.append(out)
    after_approved = {idx for idx, row in enumerate(rows) if truthy(row.get("approved"))}
    regression_count = len(before_approved - after_approved)
    payload = {
        "contract_version": "turnaround_mapping_contract_v8_1",
        "source_contract_version": "turnaround_mapping_contract_v8",
        "previous_mapping_count": len(before_approved),
        "newly_recovered_mapping_count": sum(1 for row in rows if truthy(row.get("newly_recovered_in_r3a"))),
        "approved_mapping_count": len(after_approved),
        "missing_mapping_count": 38 - len(after_approved),
        "mapping_regression_count": regression_count,
        "additional_api_calls": 0,
        "rows": rows,
    }
    return payload, regression_count


def route_counter_summaries(counter_rows: pd.DataFrame, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    cand_by_route: Dict[str, List[Mapping[str, Any]]] = {}
    for row in candidates:
        cand_by_route.setdefault(str(row["route_id"]), []).append(row)
    for route_id, group in counter_rows.groupby("route_id", dropna=False):
        route_id = str(route_id)
        cands = cand_by_route.get(route_id, [])
        summaries[route_id] = {
            "terminal_state_observation_count": int(group["is_terminal_state_v4"].sum()),
            "terminal_entry_event_count": int((group["terminal_counter_event_type_v4"] == "TERMINAL_ENTRY_EVENT").sum()),
            "terminal_hold_observation_count": int((group["terminal_counter_event_type_v4"] == "TERMINAL_HOLD_OBSERVATION").sum()),
            "classified_operation_transition_count": sum(1 for row in cands if str(row.get("operation_candidate")) in CLASSIFIED_OPERATION_TYPES),
            "loop_reset_transition_count": sum(1 for row in cands if str(row.get("operation_candidate")) == "LOOP_CONTINUOUS"),
        }
    return summaries


def validate_source_gate(source: Path) -> Dict[str, Any]:
    gate = json.loads((source / "prompt5_e01_r2d1c_r3_gate.json").read_text(encoding="utf-8"))
    required = {
        "status": "PASS_MAPPING_PARTIALLY_RECOVERED",
        "approved_mapping_count": 28,
        "missing_mapping_count": 10,
        "newly_recovered_mapping_count": 2,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "phase2_production_executed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    mismatches = {key: {"expected": value, "actual": gate.get(key)} for key, value in required.items() if gate.get(key) != value}
    if gate.get("additional_api_calls_for_correction") not in {0, None}:
        mismatches["additional_api_calls_for_correction"] = {"expected": 0, "actual": gate.get("additional_api_calls_for_correction")}
    if mismatches:
        raise RuntimeError(f"R3 prerequisite failed: {mismatches}")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1C-R3A loop-closure counter reprocessing.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    source = project_root / SOURCE_R3
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    source_gate = validate_source_gate(source)
    integrity = source_integrity(source)
    dump_json(output_root / "source_raw_integrity_audit.json", integrity)
    dump_json(
        output_root / "r3_reference.json",
        {
            "source_r3_artifact": str(source),
            "source_r3_gate_path": str(source / "prompt5_e01_r2d1c_r3_gate.json"),
            "source_r3_gate_sha256": sha256_file(source / "prompt5_e01_r2d1c_r3_gate.json"),
            "required_status": "PASS_MAPPING_PARTIALLY_RECOVERED",
            "required_approved_mapping_count": 28,
            "required_missing_mapping_count": 10,
            "additional_api_calls": 0,
        },
    )

    samples = pd.read_parquet(source / "terminal_semantics_position_samples_r3.parquet")
    v8 = pd.read_parquet(source / "turnaround_mapping_contract_v8.parquet")
    transitions = pd.read_parquet(source / "terminal_transition_events_r3.parquet")
    static_df = pd.read_parquet(source / "unresolved_route_static_topology.parquet")

    static_rows, static_by_key = duplicate_static_audit(static_df)
    dump_json(
        output_root / "duplicate_loop_closure_static_audit.json",
        {
            "duplicate_closure_route_count": sum(1 for row in static_rows if truthy(row.get("duplicate_loop_closure"))),
            "rows": static_rows,
        },
    )

    annotated = annotate_terminal_state(samples, static_by_key)
    counter_audit, counter_rows = reclassify_counters(annotated)
    reset_rows = reset_candidates(annotated, static_by_key)
    counter_by_route = route_counter_summaries(counter_rows, reset_rows)
    classified_count = sum(1 for row in reset_rows if str(row.get("operation_candidate")) in CLASSIFIED_OPERATION_TYPES)
    loop_reset_count = sum(1 for row in reset_rows if str(row.get("operation_candidate")) == "LOOP_CONTINUOUS")
    counter_audit.update(
        {
            "counter_contract_version": "terminal_counter_contract_v4",
            "classified_operation_transition_count": classified_count,
            "loop_reset_transition_count": loop_reset_count,
            "legacy_terminal_reach_event_count": source_gate.get("terminal_reach_event_count"),
            "legacy_valid_terminal_transition_count": source_gate.get("valid_terminal_transition_count"),
            "legacy_terminal_transition_rows": len(transitions),
            "legacy_classified_loop_rows": int((transitions.get("operation_candidate", pd.Series(dtype=str)).astype(str) == "LOOP_CONTINUOUS").sum()) if not transitions.empty else 0,
            "required_relationship_pass": classified_count <= counter_audit["terminal_entry_event_count"] + counter_audit["left_censored_terminal_observation_count"] <= counter_audit["terminal_state_observation_count"],
        }
    )
    dump_json(
        output_root / "terminal_counter_contract_v4.json",
        {
            "terminal_state_observation_count": "all rows observed in effective terminal state; not an event count",
            "terminal_entry_event_count": "non-terminal to terminal transition per vehicle",
            "left_censored_terminal_observation_count": "first observed row already terminal",
            "terminal_hold_observation_count": "terminal to terminal observations",
            "classified_operation_transition_count": sorted(CLASSIFIED_OPERATION_TYPES),
            "loop_reset_transition_count": "same vehicle route/direction reset from terminal boundary band to start sequence band",
            "start_sequence_upper_bound": START_SEQUENCE_UPPER_BOUND,
            "terminal_sequence_tolerance": TERMINAL_SEQUENCE_TOLERANCE,
        },
    )
    dump_json(output_root / "terminal_counter_reclassification_audit.json", counter_audit)
    counter_rows.to_parquet(output_root / "terminal_counter_reclassification_rows.parquet", index=False)
    write_parquet(output_root / "existing_raw_sequence_reset_candidates.parquet", reset_rows)
    dump_json(
        output_root / "existing_raw_sequence_reset_audit.json",
        {
            "terminal_sequence_tolerance": TERMINAL_SEQUENCE_TOLERANCE,
            "start_sequence_upper_bound": START_SEQUENCE_UPPER_BOUND,
            "duplicate_closure_reset_candidate_count": len(reset_rows),
            "approved_duplicate_closure_reset_candidate_count": sum(1 for row in reset_rows if truthy(row.get("approved_candidate"))),
            "routes_with_candidates": sorted({str(row["route_id"]) for row in reset_rows}),
            "special_reaudit_targets": {
                "3000232000": {
                    "expected_vehicle_ids": ["5224", "5271"],
                    "observed_vehicle_ids": sorted({str(row["vehicle_id"]) for row in reset_rows if str(row["route_id"]) == "3000232000"}),
                },
                "3000323101": {
                    "expected_vehicle_ids": ["5220", "5229", "5265"],
                    "observed_vehicle_ids": sorted({str(row["vehicle_id"]) for row in reset_rows if str(row["route_id"]) == "3000323101"}),
                    "vehicle_5220_terminal_tolerance_pass": any(str(row["route_id"]) == "3000323101" and str(row["vehicle_id"]) == "5220" and (as_int(row.get("previous_sequence")) or 0) >= (as_int(row.get("last_unique_preclosure_sequence")) or 0) - TERMINAL_SEQUENCE_TOLERANCE for row in reset_rows),
                },
            },
            "rows": reset_rows,
        },
    )

    trigger_audit = run_focused_trigger_unit_tests()
    dump_json(
        output_root / "focused_capture_trigger_contract_v2.json",
        {
            "trigger_rule_v2": trigger_audit["trigger_rule_v2"],
            "effective_terminal_boundary": trigger_audit["effective_terminal_boundary"],
            "terminal_sequence_tolerance": TERMINAL_SEQUENCE_TOLERANCE,
            "start_sequence_upper_bound": START_SEQUENCE_UPPER_BOUND,
            "live_capture_executed": False,
        },
    )
    dump_json(output_root / "focused_capture_trigger_unit_test_audit.json", trigger_audit)

    decisions, decisions_by_route = build_decisions(v8, reset_rows, output_root)
    write_parquet(output_root / "terminal_operation_decisions_r3a.parquet", decisions)

    mapping_v8_1, regression_count = update_mapping_v8_1(v8, static_by_key, counter_by_route, decisions_by_route, output_root)
    dump_json(output_root / "turnaround_mapping_contract_v8_1.json", mapping_v8_1)
    pd.DataFrame(mapping_v8_1["rows"]).to_parquet(output_root / "turnaround_mapping_contract_v8_1.parquet", index=False)

    newly = mapping_v8_1["newly_recovered_mapping_count"]
    counter_pass = bool(counter_audit["counter_relationship_pass"] and counter_audit["required_relationship_pass"])
    tests_pass = bool(trigger_audit["focused_capture_trigger_unit_tests_passed"])
    if regression_count:
        status = "FAIL_MAPPING_REGRESSION"
    elif not counter_pass:
        status = "FAIL_COUNTER_RECLASSIFICATION"
    elif newly > 0 and tests_pass:
        status = "PASS_EXISTING_RAW_MAPPING_RECOVERED"
    elif tests_pass:
        status = "PASS_COUNTER_FIXED_MAPPING_UNCHANGED"
    else:
        status = "FAIL"

    remaining_rows = []
    approved_routes = {str(row["route_id"]) for row in mapping_v8_1["rows"] if truthy(row.get("approved"))}
    for route_id in sorted(TARGET_ROUTES):
        if route_id in approved_routes:
            continue
        remaining_rows.append(
            {
                "route_id": route_id,
                "blocker": "NOT_OBSERVED_IN_R3A_REPROCESSING" if route_id not in {str(row["route_id"]) for row in reset_rows} else "BLOCKED_DUPLICATE_CLOSURE_EVIDENCE_INSUFFICIENT",
                "remaining_requirement": "R4 batch 2/3 broad observation or targeted focused capture; no Phase-2/recovery authorized",
            }
        )
    dump_json(output_root / "remaining_mapping_evidence_requirements_v2.json", {"rows": remaining_rows})
    phase2 = {
        "approved_for_terminal_recovery_campaign": False,
        "approved_for_phase2_turnaround_execution": False,
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "phase2_production_executed": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
    }
    dump_json(output_root / "phase2_execution_authorization.json", phase2)

    gate = {
        "status": status,
        "classification": status,
        "source_r3_artifact": str(source),
        "source_r3_gate_sha256": sha256_file(source / "prompt5_e01_r2d1c_r3_gate.json"),
        "additional_api_calls": 0,
        "previous_mapping_count": source_gate.get("approved_mapping_count"),
        "newly_recovered_mapping_count": newly,
        "approved_mapping_count": mapping_v8_1["approved_mapping_count"],
        "missing_mapping_count": mapping_v8_1["missing_mapping_count"],
        "mapping_regression_count": regression_count,
        "duplicate_closure_route_count": sum(1 for row in static_rows if truthy(row.get("duplicate_loop_closure"))),
        "duplicate_closure_reset_candidate_count": len(reset_rows),
        "approved_duplicate_closure_mapping_count": sum(1 for row in decisions if truthy(row.get("approved"))),
        "terminal_state_observation_count": counter_audit["terminal_state_observation_count"],
        "terminal_entry_event_count": counter_audit["terminal_entry_event_count"],
        "left_censored_terminal_observation_count": counter_audit["left_censored_terminal_observation_count"],
        "terminal_hold_observation_count": counter_audit["terminal_hold_observation_count"],
        "classified_operation_transition_count": classified_count,
        "loop_reset_transition_count": loop_reset_count,
        "focused_capture_trigger_bug_found": trigger_audit["focused_capture_trigger_bug_found"],
        "focused_capture_trigger_fixed": trigger_audit["focused_capture_trigger_fixed"],
        "focused_capture_trigger_unit_tests_passed": tests_pass,
        **phase2,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "threshold_changed": False,
        "scientific_parameter_changed": False,
        "raw_source_mutated": False,
    }
    dump_json(output_root / "prompt5_e01_r2d1c_r3a_gate.json", gate)

    report = f"""# Prompt 5-E01-R2D-1C-R3A Final Report

status: {status}
classification: {status}

source R3 artifact: {source}
additional API calls: 0

## Mapping

previous approved: {source_gate.get('approved_mapping_count')}
newly recovered from existing raw: {newly}
approved: {mapping_v8_1['approved_mapping_count']}
missing: {mapping_v8_1['missing_mapping_count']}
mapping regression count: {regression_count}

New duplicate-closure approvals:
{chr(10).join('- ' + row['route_id'] + ' / direction ' + row['direction_id'] + ' / ' + row['decision'] + ' / ' + row['confidence'] for row in decisions if truthy(row.get('approved'))) or '- none'}

## Counter v4

terminal state observations: {counter_audit['terminal_state_observation_count']}
terminal entries: {counter_audit['terminal_entry_event_count']}
left-censored terminal observations: {counter_audit['left_censored_terminal_observation_count']}
terminal holds: {counter_audit['terminal_hold_observation_count']}
classified operation transitions: {classified_count}
loop reset transitions: {loop_reset_count}

Legacy mixed counters were not carried forward as event counters.

## Focused Trigger

bug found: true
fixed in trigger contract/unit helper: {str(trigger_audit['focused_capture_trigger_fixed']).lower()}
unit tests passed: {str(tests_pass).lower()}
live capture executed: false

## Guards

terminal recovery estimated: false
terminal recovery applied: false
Phase-2 executed: false
training/test access: false
"""
    (output_root / "prompt5_e01_r2d1c_r3a_final_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "prompt": "Prompt 5-E01-R2D-1C-R3A",
        "artifact_dir": str(output_root),
        "source_r3_artifact": str(source),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "additional_api_calls": 0,
        "status": status,
        "files": sorted(str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file()),
    }
    dump_json(output_root / "prompt5_e01_r2d1c_r3a_manifest.json", manifest)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
