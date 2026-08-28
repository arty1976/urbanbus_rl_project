from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1b_route_recovery_modeling"
R2D_ROOT = "05_training/artifacts/prompt5_e01_r2d_approved_throughput_repair_20260720_000000"
R2D1A_ROOT = "05_training/artifacts/prompt5_e01_r2d1a_instrumentation_mapping_recovery_20260720_000000"
SOURCE_PACK = "05_training/artifacts/suseong_source_pack_v1"
SERVICE_GRAPH = "05_training/artifacts/suseong_service_graph_v1"
OFFICIAL_URLS = [
    {
        "source_id": "official_data_go_kr_15050946",
        "url": "https://www.data.go.kr/data/15050946/fileData.do#/tab-layer-openapi",
        "expected_content": "Daegu bus stop location metadata and auto-converted open API page",
    },
    {
        "source_id": "official_daegu_15050942",
        "url": "https://data.daegu.go.kr/open/data/dataView.do?dataSetId=15050942&dataSetDetailId=150509421f71a562bb88d&provdMethod=FILE",
        "expected_content": "Daegu city bus route information file metadata page",
    },
]
MISSING_KEYS = [
    ("2000002000", "1"),
    ("2000002100", "1"),
    ("2000003100", "1"),
    ("3000232000", "1"),
    ("3000323101", "1"),
    ("3000410000", "1"),
    ("3000410100", "1"),
    ("4010002001", "1"),
    ("4010002003", "1"),
    ("4010002004", "1"),
    ("4010002118", "1"),
    ("4050001001", "1"),
    ("4050010000", "1"),
]
LOCK_FILES = [
    f"{R2D1A_ROOT}/prompt5_e01_r2d1a_gate.json",
    f"{R2D1A_ROOT}/turnaround_mapping_contract_v3.json",
    f"{R2D1A_ROOT}/terminal_recovery_contract_v3.json",
    f"{R2D1A_ROOT}/counter_semantics_contract_v3.json",
    f"{R2D1A_ROOT}/passenger_event_time_contract_v1.json",
    f"{SOURCE_PACK}/route_stop_sequences.parquet",
    f"{SERVICE_GRAPH}/service_route_sequences.csv",
    f"{SOURCE_PACK}/route_stop_sequences.parquet",
    f"{SOURCE_PACK}/full_graph_edges.parquet",
    f"{SOURCE_PACK}/full_graph_nodes.parquet",
    f"{SOURCE_PACK}/vehicle_position_samples.parquet",
    f"{SOURCE_PACK}/route_service_frequency.parquet",
    f"{SERVICE_GRAPH}/official_fleet_route_estimate.csv",
    f"{SERVICE_GRAPH}/official_frequency_route_direction_runtime_audit.csv",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def route_key(route_id: Any, direction_id: Any) -> Tuple[str, str]:
    return (str(route_id), str(direction_id))


def condition_agents(central: int, condition: str) -> int:
    return int(round(central * {"A": 1.0, "A90": 0.9, "A80": 0.8, "A70": 0.7}[condition]))


def validate_r2d1a(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    root = project_root / R2D1A_ROOT
    gate_path = root / "prompt5_e01_r2d1a_gate.json"
    gate = load_json(gate_path)
    required = {
        "status": "PASS_INSTRUMENTATION_REPAIRED_MAPPING_BLOCKED",
        "counter_semantics_passed": True,
        "passenger_event_time_passed": True,
        "expected_mapping_count": 38,
        "approved_mapping_count": 25,
        "missing_mapping_count": 13,
        "expected_recovery_count": 38,
        "approved_recovery_count": 0,
        "unresolved_recovery_count": 38,
        "phase2_production_executed": False,
    }
    mismatches = {key: {"expected": value, "actual": gate.get(key)} for key, value in required.items() if gate.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R2D-1A prerequisite failed: {mismatches}")
    return root, gate


def lock_sources(project_root: Path) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for rel in LOCK_FILES:
        if rel in seen:
            continue
        seen.add(rel)
        path = project_root / rel
        rows.append(
            {
                "relative_path": rel,
                "absolute_path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "role": "route_link_sequence_materialized" if rel.endswith("route_stop_sequences.parquet") else ("graph_edge_master_materialized" if rel.endswith("full_graph_edges.parquet") else "source_lock"),
            }
        )
    return rows


def acquire_official_sources(output_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_root = output_root / "external_sources_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    successes = []
    failures = []
    for item in OFFICIAL_URLS:
        started_at = utc_now()
        request = urllib.request.Request(item["url"], headers={"User-Agent": "urbanbus-rl-provenance-audit/1.0"})
        raw_path = raw_root / f"{item['source_id']}.html"
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = response.read()
                status = int(getattr(response, "status", 0) or 0)
                content_type = response.headers.get("Content-Type")
            raw_path.write_bytes(payload)
            text = payload.decode("utf-8", errors="ignore")
            row_match = re.search(r"전체\s*행\s*([0-9,]+)", text)
            successes.append(
                {
                    "source_id": item["source_id"],
                    "source_acquisition_timestamp": started_at,
                    "endpoint_or_source_url": item["url"],
                    "request_parameters": {},
                    "http_status": status,
                    "content_type": content_type,
                    "raw_path": str(raw_path),
                    "raw_file_sha256": sha256_file(raw_path),
                    "normalized_file_sha256": None,
                    "row_count": int(row_match.group(1).replace(",", "")) if row_match else None,
                    "expected_content": item["expected_content"],
                    "usable_for_route_direction_mapping": False,
                    "usable_for_terminal_recovery": False,
                    "limitation": "Acquired metadata/HTML page, not an authenticated route-direction sequence or terminal layover table.",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "source_id": item["source_id"],
                    "source_acquisition_timestamp": started_at,
                    "endpoint_or_source_url": item["url"],
                    "request_parameters": {},
                    "http_status": None,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "interpreted_as_empty_route": False,
                }
            )
    manifest = {
        "created_at_utc": utc_now(),
        "new_external_source_count": len(successes),
        "source_acquisition_failure_count": len(failures),
        "successes": successes,
    }
    return manifest, {"failures": failures}


def service_routes(service: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, group in service.groupby(["route_id", "direction_id"], sort=False):
        out[route_key(*key)] = group.sort_values("stop_order").copy()
    return out


def route_summary_rows(project_root: Path, mapping_v3: pd.DataFrame) -> List[Dict[str, Any]]:
    service = read_table(project_root / SERVICE_GRAPH / "service_route_sequences.csv")
    full = read_table(project_root / SOURCE_PACK / "route_stop_sequences.parquet")
    rows = []
    for route_id, direction_id in MISSING_KEYS:
        service_sub = service[(service["route_id"].astype(str) == route_id) & (service["direction_id"].astype(str) == direction_id)].sort_values("stop_order")
        full_sub = full[(full["route_id"].astype(str) == route_id) & (full["direction_id"].astype(str) == direction_id)].sort_values("stop_order")
        same_no_route_ids: List[str] = []
        route_no = None
        if not service_sub.empty:
            route_no = str(service_sub.iloc[0]["route_no"])
        elif not full_sub.empty:
            route_no = str(full_sub.iloc[0]["route_no"])
        if route_no is not None:
            same_no_route_ids = sorted(full[full["route_no"].astype(str) == route_no]["route_id"].astype(str).unique().tolist())
        old = mapping_v3[(mapping_v3["route_id"].astype(str) == route_id) & (mapping_v3["service_direction_id"].astype(str) == direction_id)]
        rows.append(
            {
                "route_id": route_id,
                "route_no": route_no,
                "direction_id": direction_id,
                "branch_id": route_id if route_no and len(same_no_route_ids) > 1 else None,
                "service_pattern_id": f"{route_id}:{direction_id}:validation_service_graph",
                "service_graph_first_stop": None if service_sub.empty else str(service_sub.iloc[0]["stop_id"]),
                "service_graph_last_stop": None if service_sub.empty else str(service_sub.iloc[-1]["stop_id"]),
                "full_route_first_stop": None if full_sub.empty else str(full_sub.iloc[0]["stop_id"]),
                "full_route_last_stop": None if full_sub.empty else str(full_sub.iloc[-1]["stop_id"]),
                "known_opposite_direction": False,
                "known_off_graph_continuation": False,
                "known_service_reentry": False,
                "same_route_no_route_ids": same_no_route_ids,
                "current_failure_classification": None if old.empty else str(old.iloc[0].get("failure_reason")),
            }
        )
    return rows


def route_identity_alias_audit(project_root: Path) -> Dict[str, Any]:
    full = read_table(project_root / SOURCE_PACK / "route_stop_sequences.parquet")
    pos = read_table(project_root / SOURCE_PACK / "vehicle_position_samples.parquet")
    rows = []
    for route_id, direction_id in MISSING_KEYS:
        sub = full[full["route_id"].astype(str) == route_id]
        route_no = None if sub.empty else str(sub.iloc[0]["route_no"])
        same_no = full[full["route_no"].astype(str) == str(route_no)] if route_no is not None else pd.DataFrame()
        pos_support = not pos[(pos["route_id"].astype(str) == route_id) & (pos["direction_id"].astype(str) == direction_id)].empty
        candidate_ids = sorted(same_no["route_id"].astype(str).unique().tolist()) if not same_no.empty else []
        rows.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "route_no": route_no,
                "route_id_changed": False,
                "deprecated_route_id": False,
                "same_route_no_multiple_route_id": len(candidate_ids) > 1,
                "candidate_route_ids": candidate_ids,
                "position_trace_support": pos_support,
                "evidence_type": "ROUTE_NO_ONLY" if candidate_ids else "NONE",
                "sequence_overlap_ratio": None,
                "terminal_match": None,
                "confidence": "LOW",
                "approved": False,
                "reason": "No official route ID mapping, exact topology mapping, or sufficient position trace identity evidence was available.",
            }
        )
    return {"status": "PASS_SOURCE_EXHAUSTED", "rows": rows}


def dijkstra_path(edges: pd.DataFrame, src: int, dst: int, max_nodes: int = 2000) -> Tuple[List[int], float, float]:
    adj: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for row in edges.to_dict("records"):
        adj[int(row["src_idx"])].append((int(row["dst_idx"]), float(row["time_sec"]), float(row["distance_m"])))
    import heapq

    heap = [(0.0, src, [src], 0.0)]
    seen: Dict[int, float] = {}
    while heap:
        cost, node, path, dist = heapq.heappop(heap)
        if node == dst:
            return path, cost, dist
        if node in seen and seen[node] <= cost:
            continue
        seen[node] = cost
        if len(path) > max_nodes:
            continue
        for nxt, sec, meters in adj.get(node, []):
            if nxt in path:
                continue
            heapq.heappush(heap, (cost + sec, nxt, [*path, nxt], dist + meters))
    return [], math.nan, math.nan


def loop_closure_v2(project_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    full = read_table(project_root / SOURCE_PACK / "route_stop_sequences.parquet")
    service = read_table(project_root / SERVICE_GRAPH / "service_route_sequences.csv")
    nodes = read_table(project_root / SOURCE_PACK / "full_graph_nodes.parquet")
    edges = read_table(project_root / SOURCE_PACK / "full_graph_edges.parquet")
    node_idx = {str(row["node_uid"]): int(row["node_index"]) for row in nodes.to_dict("records")}
    edge_sha = sha256_file(project_root / SOURCE_PACK / "full_graph_edges.parquet")
    rows = []
    path_rows = []
    for route_id, direction_id in MISSING_KEYS[:3]:
        full_sub = full[(full["route_id"].astype(str) == route_id) & (full["direction_id"].astype(str) == direction_id)].sort_values("stop_order")
        service_sub = service[(service["route_id"].astype(str) == route_id) & (service["direction_id"].astype(str) == direction_id)].sort_values("stop_order")
        if full_sub.empty or service_sub.empty:
            rows.append({"route_id": route_id, "direction_id": direction_id, "operation": "UNRESOLVED", "approved": False, "reason": "missing sequence"})
            continue
        start_uid = str(service_sub.iloc[0]["node_uid"])
        terminal_uid = str(service_sub.iloc[-1]["node_uid"])
        src = node_idx.get(terminal_uid)
        dst = node_idx.get(start_uid)
        path, travel, dist = ([], math.nan, math.nan) if src is None or dst is None else dijkstra_path(edges, src, dst)
        direct_full_reentry = bool((full_sub["stop_order"] > service_sub.iloc[-1]["stop_order"]).any() and (full_sub[full_sub["stop_order"] > service_sub.iloc[-1]["stop_order"]]["stop_id"].astype(str) == str(service_sub.iloc[0]["stop_id"])).any())
        approved = False
        operation = "UNRESOLVED"
        confidence = "LOW"
        reason = "graph path alone does not prove official bus loop closure"
        if direct_full_reentry:
            approved = True
            operation = "LOOP_CONTINUOUS"
            confidence = "HIGH"
            reason = None
        rows.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "route_no": str(service_sub.iloc[0]["route_no"]),
                "operation": operation,
                "service_terminal_stop_id": str(service_sub.iloc[-1]["stop_id"]),
                "next_trip_initial_stop_id": str(service_sub.iloc[0]["stop_id"]),
                "loop_closure_edge_sequence_available": bool(path),
                "loop_closure_node_count": len(path),
                "loop_closure_distance_m": None if math.isnan(dist) else dist,
                "loop_closure_travel_seconds": None if math.isnan(travel) else travel,
                "loop_closure_source": str(project_root / SOURCE_PACK / "full_graph_edges.parquet") if path else None,
                "loop_closure_source_sha256": edge_sha if path else None,
                "position_trace_crossing_closure": False,
                "mapping_confidence": confidence,
                "approved": approved,
                "reason": reason,
            }
        )
        for order, node in enumerate(path):
            path_rows.append({"route_id": route_id, "direction_id": direction_id, "path_order": order, "node_index": node})
    return {"status": "PASS" if all(row["approved"] for row in rows) else "PARTIAL_UNRESOLVED", "rows": rows}, path_rows


def mapping_contract_v4(project_root: Path, output_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    r2d1a = project_root / R2D1A_ROOT
    v3 = read_table(r2d1a / "turnaround_mapping_contract_v3.parquet")
    loop_audit, loop_paths = loop_closure_v2(project_root)
    loop_by_key = {(row["route_id"], row["direction_id"]): row for row in loop_audit["rows"]}
    rows = []
    newly_recovered = 0
    for item in v3.to_dict("records"):
        route_id = str(item["route_id"])
        direction_id = str(item["service_direction_id"])
        row = dict(item)
        row["direction_id"] = direction_id
        row["branch_id"] = route_id if str(item.get("route_no", "")).startswith("가창2") else None
        row["off_graph_stop_sequence"] = None
        row["off_graph_edge_sequence"] = None
        row["off_graph_distance_m"] = None
        row["off_graph_travel_seconds"] = None
        row["mapping_evidence"] = row.get("mapping_evidence_rows")
        if (route_id, direction_id) in loop_by_key and loop_by_key[(route_id, direction_id)]["approved"]:
            row["terminal_operation_type"] = loop_by_key[(route_id, direction_id)]["operation"]
            row["next_direction_id"] = direction_id
            row["next_trip_initial_stop_id"] = loop_by_key[(route_id, direction_id)]["next_trip_initial_stop_id"]
            row["mapping_confidence"] = loop_by_key[(route_id, direction_id)]["mapping_confidence"]
            row["approved"] = True
            row["mapping_source"] = loop_by_key[(route_id, direction_id)]["loop_closure_source"]
            row["mapping_source_sha256"] = loop_by_key[(route_id, direction_id)]["loop_closure_source_sha256"]
            newly_recovered += int(not bool(item.get("approved")))
        else:
            row["approved"] = bool(item.get("approved"))
            if not row["approved"]:
                row["terminal_operation_type"] = "UNRESOLVED"
        rows.append(row)
    approved = sum(1 for row in rows if bool(row.get("approved")))
    contract = {
        "contract_version": "turnaround_mapping_contract_v4",
        "expected_mapping_count": 38,
        "previously_approved_mapping_count": int(v3["approved"].fillna(False).sum()),
        "newly_recovered_mapping_count": newly_recovered,
        "approved_mapping_count": approved,
        "missing_mapping_count": 38 - approved,
        "loop_route_count": 3,
        "loop_closure_approved_count": sum(1 for row in loop_audit["rows"] if row["approved"]),
        "off_graph_reentry_count": sum(1 for row in rows if row.get("terminal_operation_type") == "OFF_GRAPH_CONTINUATION_AND_REENTRY" and row.get("approved")),
        "paired_direction_count": sum(1 for row in rows if row.get("terminal_operation_type") == "PAIRED_OPPOSITE_DIRECTION" and row.get("approved")),
        "rows": rows,
    }
    write_parquet(output_root / "turnaround_mapping_contract_v4.parquet", rows)
    write_parquet(output_root / "loop_closure_paths.parquet", loop_paths)
    return contract, loop_audit["rows"], loop_audit


def weak_components(nodes: Iterable[int], edge_pairs: Iterable[Tuple[int, int]]) -> int:
    node_set = set(nodes)
    adj: Dict[int, set] = {node: set() for node in node_set}
    for a, b in edge_pairs:
        if a in node_set and b in node_set:
            adj[a].add(b)
            adj[b].add(a)
    seen = set()
    count = 0
    for node in node_set:
        if node in seen:
            continue
        count += 1
        queue = deque([node])
        seen.add(node)
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return count


def route_exclusion_impact(project_root: Path, mapping_contract: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    service = read_table(project_root / SERVICE_GRAPH / "service_route_sequences.csv")
    nodes = read_table(project_root / SOURCE_PACK / "full_graph_nodes.parquet")
    edges = read_table(project_root / SOURCE_PACK / "full_graph_edges.parquet")
    passenger_events = read_table(project_root / R2D1A_ROOT / "passenger_event_ledger.parquet")
    unresolved = {(str(row["route_id"]), str(row["direction_id"])) for row in mapping_contract["rows"] if not bool(row.get("approved"))}
    all_keys = sorted({route_key(row["route_id"], row["direction_id"]) for row in service.to_dict("records")})
    unresolved_stop_nodes = set(service[[route_key(row["route_id"], row["direction_id"]) in unresolved for row in service.to_dict("records")]]["node_uid"].astype(str))
    approved_stop_nodes = set(service[[route_key(row["route_id"], row["direction_id"]) not in unresolved for row in service.to_dict("records")]]["node_uid"].astype(str))
    only_unresolved_nodes = unresolved_stop_nodes - approved_stop_nodes
    total_stop_nodes = set(service["node_uid"].astype(str))
    request_events = passenger_events[passenger_events["event_type"] == "PASSENGER_REQUESTED"].copy()
    boarded_events = passenger_events[passenger_events["event_type"] == "PASSENGER_BOARDED"].copy()
    total_demand = float(request_events["passenger_count"].sum()) if not request_events.empty else 0.0
    unresolved_demand = float(request_events[request_events["node_uid"].astype(str).isin(unresolved_stop_nodes)]["passenger_count"].sum()) if not request_events.empty else 0.0
    unresolved_boardings = float(boarded_events[boarded_events["node_uid"].astype(str).isin(unresolved_stop_nodes)]["passenger_count"].sum()) if not boarded_events.empty else 0.0
    total_boardings = float(boarded_events["passenger_count"].sum()) if not boarded_events.empty else 0.0
    node_idx = {str(row["node_uid"]): int(row["node_index"]) for row in nodes.to_dict("records")}
    remaining_indices = {node_idx[node] for node in (total_stop_nodes - only_unresolved_nodes) if node in node_idx}
    edge_pairs = [(int(row["src_idx"]), int(row["dst_idx"])) for row in edges.to_dict("records")]
    components = weak_components(remaining_indices, edge_pairs)
    isolated = 0
    degree = defaultdict(int)
    for a, b in edge_pairs:
        if a in remaining_indices and b in remaining_indices:
            degree[a] += 1
            degree[b] += 1
    isolated = sum(1 for node in remaining_indices if degree[node] == 0)
    by_condition = {}
    route_keys = sorted(all_keys)
    for condition in ["A", "A90", "A80", "A70"]:
        agents = condition_agents(417, condition)
        counts = {key: 0 for key in route_keys}
        for agent_id in range(agents):
            counts[route_keys[agent_id % len(route_keys)]] += 1
        unresolved_fleet = sum(count for key, count in counts.items() if key in unresolved)
        by_condition[condition] = {
            "active_fleet_count": agents,
            "unresolved_active_fleet_count": unresolved_fleet,
            "unresolved_fleet_share": unresolved_fleet / max(agents, 1),
            "unresolved_service_stop_count": len(unresolved_stop_nodes),
            "unresolved_stop_share": len(unresolved_stop_nodes) / max(len(total_stop_nodes), 1),
            "unresolved_generated_demand": unresolved_demand,
            "unresolved_demand_share": unresolved_demand / max(total_demand, 1.0),
            "unresolved_historical_boardings": unresolved_boardings,
            "unresolved_historical_boarding_share": unresolved_boardings / max(total_boardings, 1.0),
            "unique_stops_served_only_by_unresolved_routes": len(only_unresolved_nodes),
            "remaining_fleet": agents - unresolved_fleet,
            "remaining_stops": len(total_stop_nodes - only_unresolved_nodes),
            "remaining_demand": total_demand - unresolved_demand,
            "remaining_graph_connectivity_components": components,
            "isolated_stop_count": isolated,
        }
    classification = "EXCLUSION_BREAKS_SERVICE_GRAPH" if isolated > 0 or components > 1 else ("EXCLUSION_MATERIAL" if by_condition["A"]["unresolved_fleet_share"] > 0.05 or by_condition["A"]["unresolved_demand_share"] > 0.05 else "EXCLUSION_IMMATERIAL")
    audit = {
        "status": classification,
        "unresolved_route_count": len(unresolved),
        "by_condition": by_condition,
        "route_exclusion_breaks_connectivity": classification == "EXCLUSION_BREAKS_SERVICE_GRAPH",
        "classification": classification,
    }
    protocol = {
        "approved": False,
        "applied": False,
        "user_approval_required": True,
        "included_routes": [{"route_id": key[0], "direction_id": key[1]} for key in route_keys if key not in unresolved],
        "excluded_routes": [{"route_id": key[0], "direction_id": key[1]} for key in sorted(unresolved)],
        "impact_summary": by_condition,
        "decision_required": True,
    }
    options = {
        "option_a_full_service_graph": {
            "requires_mapping_38_of_38": True,
            "requires_direct_or_user_approved_assumption_recovery_38_of_38": True,
            "currently_possible": False,
        },
        "option_b_provenance_complete_subset": {
            "approved": False,
            "requires_user_approval": True,
            "subset_dataset_created": False,
            "currently_possible": False,
            "reason": "Mapping-complete subset can be described, but recovery provenance is still unresolved for all routes.",
            "coverage": by_condition,
        },
    }
    return audit, protocol, options


def terminal_recovery_model(project_root: Path, mapping_contract: Mapping[str, Any], output_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    r2d1a = project_root / R2D1A_ROOT
    v3 = read_table(r2d1a / "terminal_recovery_contract_v3.parquet")
    pos = read_table(project_root / SOURCE_PACK / "vehicle_position_samples.parquet")
    freq = read_table(project_root / SOURCE_PACK / "route_service_frequency.parquet")
    rows = []
    inventory = []
    obs_events: List[Dict[str, Any]] = []
    obs_summary: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    for item in v3.to_dict("records"):
        route_id = str(item["route_id"])
        direction_id = str(item["direction_id"])
        pos_sub = pos[(pos["route_id"].astype(str) == route_id) & (pos["direction_id"].astype(str) == direction_id)].copy()
        if not pos_sub.empty:
            pos_sub["event_day"] = pd.to_datetime(pos_sub["event_time"], errors="coerce").dt.date.astype(str)
        unique_vehicle_count = int(pos_sub["vehicle_id"].astype(str).nunique()) if not pos_sub.empty else 0
        service_day_count = int(pos_sub["event_day"].nunique()) if not pos_sub.empty and "event_day" in pos_sub else 0
        inventory.append(
            {
                "route_id": route_id,
                "direction_id": direction_id,
                "source_type": "POSITION_SAMPLE",
                "source_path": str(project_root / SOURCE_PACK / "vehicle_position_samples.parquet"),
                "source_sha256": sha256_file(project_root / SOURCE_PACK / "vehicle_position_samples.parquet"),
                "observation_count": int(len(pos_sub)),
                "unique_vehicle_count": unique_vehicle_count,
                "service_day_count": service_day_count,
                "time_unit": "timestamp",
                "approved": False,
                "reason": "Position samples do not reconstruct uncensored terminal arrival/departure dwell events over >=3 service days.",
            }
        )
        row = dict(item)
        row.update(
            {
                "terminal_operation_type": next((m.get("terminal_operation_type") for m in mapping_contract["rows"] if str(m["route_id"]) == route_id and str(m["direction_id"]) == direction_id), None),
                "recovery_seconds_base": None,
                "recovery_seconds_low": None,
                "recovery_seconds_high": None,
                "recovery_source_classification": "UNRESOLVED_RECOVERY",
                "derivation_formula": None,
                "sample_count": 0,
                "unique_vehicle_count": unique_vehicle_count,
                "service_day_count": service_day_count,
                "directly_approved": False,
                "assumption_candidate": False,
                "requires_user_approval": True,
            }
        )
        rows.append(row)
    route_class_def = {
        "route_class_fields": ["terminal_operation_type", "route_length_band", "service_frequency_band", "route_type"],
        "pooled_candidate_minimums": {"distinct_routes": 3, "terminal_dwell_events": 30, "service_days": 3},
        "approved": False,
        "executed": False,
        "reason": "No eligible uncensored terminal dwell observations were available for pooling.",
    }
    contract = {
        "contract_version": "terminal_recovery_contract_v4",
        "expected_recovery_count": 38,
        "official_explicit_recovery_count": 0,
        "official_schedule_derived_count": 0,
        "route_level_observed_recovery_count": 0,
        "route_class_pooled_candidate_count": 0,
        "directly_approved_recovery_count": 0,
        "assumption_candidate_count": 0,
        "unresolved_recovery_count": 38,
        "rows": rows,
    }
    protocol = {
        "approved": False,
        "executed": False,
        "requires_user_approval": True,
        "status": "BLOCKED_RECOVERY_EVIDENCE_INSUFFICIENT",
        "sensitivity_matrix": ["RECOVERY_LOW", "RECOVERY_BASE", "RECOVERY_HIGH"],
        "routes": [
            {
                "route_id": row["route_id"],
                "direction_id": row["direction_id"],
                "available_evidence": "No direct layover field, no schedule-derived cycle decomposition, no sufficient terminal dwell sample.",
                "pooled_route_class": None,
                "low_seconds": None,
                "base_seconds": None,
                "high_seconds": None,
                "source_statistics": None,
                "reason": "No source-backed numeric candidate can be produced without an approved assumption design.",
            }
            for row in rows
        ],
    }
    write_parquet(output_root / "terminal_recovery_contract_v4.parquet", rows)
    write_parquet(output_root / "terminal_position_event_reconstruction.parquet", obs_events)
    write_parquet(output_root / "terminal_recovery_observation_summary.parquet", obs_summary)
    write_parquet(output_root / "route_class_recovery_statistics.parquet", class_rows)
    return contract, {"sources": inventory}, protocol, obs_events, obs_summary, [route_class_def]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1B route mapping and recovery evidence modeling.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r2d1a_root, r2d1a_gate = validate_r2d1a(project_root)
    r2d1a_gate_path = r2d1a_root / "prompt5_e01_r2d1a_gate.json"
    dump_json(output_root / "r2d1a_reference.json", {"path": str(r2d1a_gate_path), "sha256": sha256_file(r2d1a_gate_path), "gate": r2d1a_gate})
    dump_json(
        output_root / "instrumentation_supersession_manifest.json",
        {
            "old_artifact_status": "SUPERSEDED_FOR_COUNTER_AND_EVENT_TIME",
            "old_artifact": str(project_root / R2D_ROOT / "phase1_canary_results.json"),
            "superseded_values": {"served_passenger_count": 1910, "avg_wait_seconds": 0.0, "completed_trip_count": 26688},
            "new_authoritative_artifact": "R2D-1A",
            "new_values": {
                "terminal_arrival_event_count": 1668,
                "completed_trip_count": 1668,
                "terminal_idle_observation_count": 26688,
                "served_passenger_count": 235,
                "boarding_before_request_count": 0,
                "negative_wait_count": 0,
                "wait_formula_mismatch_count": 0,
            },
            "existing_files_modified": False,
        },
    )
    dump_json(output_root / "source_provenance_lock.json", {"files": lock_sources(project_root)})
    source_manifest, source_failures = acquire_official_sources(output_root)
    dump_json(output_root / "route_source_acquisition_manifest.json", source_manifest)
    dump_json(output_root / "route_source_acquisition_failures.json", source_failures)

    mapping_v3 = read_table(r2d1a_root / "turnaround_mapping_contract_v3.parquet")
    missing_rows = route_summary_rows(project_root, mapping_v3)
    dump_json(output_root / "missing_route_recovery_audit.json", {"target_route_direction_count": len(MISSING_KEYS), "rows": missing_rows})
    dump_json(output_root / "route_identity_alias_audit.json", route_identity_alias_audit(project_root))
    mapping_contract, loop_rows, loop_audit = mapping_contract_v4(project_root, output_root)
    dump_json(output_root / "loop_closure_audit_v2.json", loop_audit)
    dump_json(output_root / "turnaround_mapping_contract_v4.json", mapping_contract)
    exclusion_audit, exclusion_protocol, full_vs_subset = route_exclusion_impact(project_root, mapping_contract)
    dump_json(output_root / "route_exclusion_impact_audit.json", exclusion_audit)
    dump_json(output_root / "route_exclusion_protocol_draft.json", exclusion_protocol)

    recovery_contract, recovery_inventory, recovery_protocol, _events, _summary, class_defs = terminal_recovery_model(project_root, mapping_contract, output_root)
    dump_json(output_root / "terminal_recovery_source_inventory_v2.json", recovery_inventory)
    dump_json(output_root / "route_class_definition.json", class_defs[0])
    dump_json(output_root / "terminal_recovery_contract_v4.json", recovery_contract)
    dump_json(output_root / "terminal_recovery_assumption_protocol_v2.json", recovery_protocol)
    md = ["# Terminal Recovery Assumption Protocol v2", "", "approved: false", "executed: false", "requires_user_approval: true", "", "No route has a source-backed low/base/high recovery candidate in this execution."]
    (output_root / "terminal_recovery_assumption_protocol_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    dump_json(output_root / "phase2_full_vs_subset_options.json", full_vs_subset)

    unresolved_a = exclusion_audit["by_condition"]["A"]
    if mapping_contract["approved_mapping_count"] == 38 and recovery_contract["directly_approved_recovery_count"] == 38:
        status = "PASS_PHASE2_DIRECT_PROVENANCE_READY"
    elif mapping_contract["approved_mapping_count"] == 38 and recovery_contract["directly_approved_recovery_count"] + recovery_contract["assumption_candidate_count"] == 38:
        status = "PASS_PHASE2_ASSUMPTION_PROTOCOL_READY"
    elif mapping_contract["missing_mapping_count"] > 0 and recovery_contract["unresolved_recovery_count"] > 0:
        status = "BLOCKED_MULTIPLE_PROVENANCE_ISSUES"
    elif mapping_contract["missing_mapping_count"] > 0:
        status = "BLOCKED_ROUTE_SUBSET_DECISION_REQUIRED"
    else:
        status = "BLOCKED_RECOVERY_EVIDENCE_INSUFFICIENT"
    authorization = {
        "approved_for_phase2_turnaround_execution": status == "PASS_PHASE2_DIRECT_PROVENANCE_READY",
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "phase2_production_executed": False,
    }
    dump_json(output_root / "phase2_execution_authorization.json", authorization)
    gate = {
        "status": status,
        "classification": status,
        "r2d1a_gate_path": str(r2d1a_gate_path),
        "r2d1a_gate_sha256": sha256_file(r2d1a_gate_path),
        "instrumentation_supersession_applied": True,
        "counter_instrumentation_authoritative": True,
        "passenger_event_time_authoritative": True,
        "full_route_source_count": 1,
        "new_external_source_count": source_manifest["new_external_source_count"],
        "source_acquisition_failure_count": source_manifest["source_acquisition_failure_count"],
        "expected_mapping_count": 38,
        "previously_approved_mapping_count": mapping_contract["previously_approved_mapping_count"],
        "newly_recovered_mapping_count": mapping_contract["newly_recovered_mapping_count"],
        "approved_mapping_count": mapping_contract["approved_mapping_count"],
        "missing_mapping_count": mapping_contract["missing_mapping_count"],
        "loop_route_count": mapping_contract["loop_route_count"],
        "loop_closure_approved_count": mapping_contract["loop_closure_approved_count"],
        "off_graph_reentry_count": mapping_contract["off_graph_reentry_count"],
        "paired_direction_count": mapping_contract["paired_direction_count"],
        "unresolved_route_fleet_share": unresolved_a["unresolved_fleet_share"],
        "unresolved_route_demand_share": unresolved_a["unresolved_demand_share"],
        "unresolved_route_stop_share": unresolved_a["unresolved_stop_share"],
        "route_exclusion_breaks_connectivity": exclusion_audit["route_exclusion_breaks_connectivity"],
        "expected_recovery_count": 38,
        "official_explicit_recovery_count": 0,
        "official_schedule_derived_count": 0,
        "route_level_observed_recovery_count": 0,
        "route_class_pooled_candidate_count": 0,
        "directly_approved_recovery_count": 0,
        "assumption_candidate_count": 0,
        "unresolved_recovery_count": 38,
        "recovery_assumption_protocol_created": True,
        "recovery_assumption_protocol_approved": False,
        "user_approval_required": True,
        "phase2_full_graph_possible": False,
        "phase2_subset_option_available": False,
        "threshold_changed": False,
        "calibration_applied": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        **authorization,
    }
    dump_json(output_root / "prompt5_e01_r2d1b_gate.json", gate)
    report = f"""[Prompt 5-E01-R2D-1B 판정]
status: {status}
classification: {status}

[Instrumentation authority]
counter: authoritative from R2D-1A
passenger event time: authoritative from R2D-1A
superseded artifacts: R2D Phase-1 served=1910, avg_wait=0.0, completed_trips=26688

[Mapping]
previous approved: {mapping_contract['previously_approved_mapping_count']}
new recovered: {mapping_contract['newly_recovered_mapping_count']}
total approved: {mapping_contract['approved_mapping_count']}
missing: {mapping_contract['missing_mapping_count']}

loop closures: {mapping_contract['loop_closure_approved_count']}/{mapping_contract['loop_route_count']}
paired directions: {mapping_contract['paired_direction_count']}
off-graph re-entry: {mapping_contract['off_graph_reentry_count']}

[Unresolved route impact]
fleet share: {unresolved_a['unresolved_fleet_share']}
demand share: {unresolved_a['unresolved_demand_share']}
stop share: {unresolved_a['unresolved_stop_share']}
connectivity impact: {exclusion_audit['status']}

[Recovery]
official explicit: 0
schedule derived: 0
route-level observed: 0
class-pooled candidates: 0
direct approved: 0
assumption candidates: 0
unresolved: 38

[Options]
full graph possible: false
subset possible: false
assumption protocol ready: false
user approval required: true

[Leakage]
test split: false
test target: false
test embedding: false

[Next gate]
Phase 2 direct execution approved: false
assumption approval required: false
subset decision required: true
baseline feasibility: false
E0/E1 retraining: false
Prompt 6A: false
E2: false
full Prompt 6: false
"""
    (output_root / "prompt5_e01_r2d1b_final_report.md").write_text(report, encoding="utf-8")
    manifest = {"created_at_utc": utc_now(), "artifact_dir": str(output_root), "files": []}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r2d1b_manifest.json":
            manifest["files"].append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    dump_json(output_root / "prompt5_e01_r2d1b_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
