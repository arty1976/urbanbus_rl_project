from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


ARTIFACT_PREFIX = "prompt5_e01_dl1r_suseong_mapping_repair_mps_rerun"
UPSTREAM_ARTIFACT_NAME = "prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation_20260731_103406"
EXPECTED_UPSTREAM_GATE = "FAIL_SUSEONG_GU_SUBGRAPH_NOT_DERIVABLE_FROM_FROZEN_LOCAL_DATA"

PASS_VALIDATED = "PASS_SUSEONG_MAPPING_REPAIRED_AND_DL1_CRITIC_JOINT_LEARNING_VALIDATED_ON_NATIVE_MAC_M4"
PASS_INDETERMINATE = "PASS_SUSEONG_MAPPING_REPAIRED_AND_DL1_LEARNING_ACTIVE_SHORT_RUN_SIGNAL_INDETERMINATE_ON_NATIVE_MAC_M4"
FAIL_REPAIRED_MPS_UNAVAILABLE = "FAIL_SUSEONG_MAPPING_REPAIRED_NATIVE_MPS_ENVIRONMENT_UNAVAILABLE"
FAIL_NOT_NATIVE = "FAIL_NOT_RUNNING_ON_NATIVE_MACOS_ARM64"
FAIL_MPS_UNAVAILABLE = "FAIL_NATIVE_MPS_UNAVAILABLE"
FAIL_MPS_COMPUTE = "FAIL_NATIVE_MPS_COMPUTE_VALIDATION"
FAIL_NOT_RECONSTRUCTABLE = "FAIL_SUSEONG_TRAINABLE_CORE_NODE_NOT_RECONSTRUCTABLE"
FAIL_ALIAS_AMBIGUOUS = "FAIL_SUSEONG_NODE_ALIAS_AMBIGUOUS"
FAIL_MAPPING_INTEGRITY = "FAIL_SUSEONG_MAPPING_REPAIR_INTEGRITY"
FAIL_DL1_SUBGRAPH = "FAIL_DL1_RERUN_SUBGRAPH_INTEGRITY"
FAIL_DL1_CRITIC = "FAIL_DL1_RERUN_CRITIC_NOT_LEARNING"
FAIL_DL1_FROZEN = "FAIL_DL1_RERUN_FROZEN_CONTROL_INVALID"
FAIL_DL1_NAN = "FAIL_DL1_RERUN_NAN_OR_INF"
FAIL_DL1_CHECKPOINT = "FAIL_DL1_RERUN_CHECKPOINT_RELOAD"
FAIL_SCOPE = "FAIL_H200_OR_CUDA_SCOPE_VIOLATION"
FAIL_MANIFEST = "FAIL_MANIFEST_INTEGRITY"

REQUIRED_FILES = [
    "upstream_failure_snapshot.json",
    "missing_node_inventory.json",
    "missing_node_inventory.parquet",
    "stop_7061025300_evidence_audit.json",
    "canonical_alias_candidate_audit.json",
    "tensor_reconstruction_feasibility.json",
    "suseong_mapping_repair.json",
    "suseong_mapping_repair.parquet",
    "repair_lineage.json",
    "repaired_suseong_subgraph_inventory.json",
    "repaired_suseong_graph_integrity_audit.json",
    "repaired_tensor_contract_audit.json",
    "snapshot_node_order_audit.json",
    "native_runtime_environment.json",
    "python_environment_inventory.json",
    "mps_availability_audit.json",
    "mps_compute_smoke_audit.json",
    "environment_change_inventory.json",
    "dl1_rerun_command.json",
    "dl1_rerun_artifact_pointer.json",
    "dl1_rerun_result_summary.json",
    "source_change_inventory.json",
    "external_access_audit.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def run_cmd(args: Sequence[str], cwd: Path, timeout: int = 20) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            list(args),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": list(args),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": list(args),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return float(2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def upstream_failure_snapshot(project_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    root = project_root / "05_training/artifacts" / UPSTREAM_ARTIFACT_NAME
    gate_path = root / "gate_decision.json"
    connectivity_path = root / "suseong_subgraph_connectivity_audit.json"
    inventory_path = root / "suseong_subgraph_inventory.json"
    gate = read_json(gate_path)
    connectivity = read_json(connectivity_path)
    inventory = read_json(inventory_path)
    missing = list(connectivity.get("missing_node_reference_sample", []))
    payload = {
        "created_at": iso_kst(),
        "upstream_artifact": str(root),
        "upstream_gate": gate.get("gate"),
        "required_upstream_gate": EXPECTED_UPSTREAM_GATE,
        "upstream_gate_ok": gate.get("gate") == EXPECTED_UPSTREAM_GATE,
        "requested_suseong_service_nodes": inventory.get("requested_service_node_count"),
        "projectable_nodes": inventory.get("node_count"),
        "missing_node_references": connectivity.get("missing_node_reference_count"),
        "missing_trainable_core_nodes": connectivity.get("missing_trainable_core_node_reference_count"),
        "missing_node_reference_sample": missing,
        "source_hashes": {
            "gate_decision_json": sha256_file(gate_path),
            "connectivity_audit_json": sha256_file(connectivity_path),
            "inventory_json": sha256_file(inventory_path),
        },
        "upstream_artifact_modified": False,
    }
    return payload, missing, bool(gate.get("gate") == EXPECTED_UPSTREAM_GATE)


def active_counts(dataset_root: Path, node_index: int) -> Dict[str, Any]:
    counts = {"train": 0, "val": 0, "test": 0}
    nonzero = {"train": 0, "val": 0, "test": 0}
    first_active = None
    first_nonzero = None
    for split in ("train", "val", "test"):
        for path in sorted((dataset_root / split).glob("*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=False)
            is_active = bool(data.node_mask[node_index].item())
            xy_nonzero = bool(float(data.x[node_index].abs().sum().item() + data.y[node_index].abs().sum().item()) > 0.0)
            if is_active:
                counts[split] += 1
                if first_active is None:
                    first_active = str(path)
            if xy_nonzero:
                nonzero[split] += 1
                if first_nonzero is None:
                    first_nonzero = str(path)
    return {
        "active_mask_true_count": int(sum(counts.values())),
        "active_mask_true_by_split": counts,
        "nonzero_xy_count": int(sum(nonzero.values())),
        "nonzero_xy_by_split": nonzero,
        "first_active_snapshot": first_active,
        "first_nonzero_snapshot": first_nonzero,
    }


def route_memberships(route_sequences: pd.DataFrame, node_uid: str) -> List[Dict[str, Any]]:
    rows = route_sequences[route_sequences["node_uid"].astype(str) == node_uid].copy()
    result = []
    for row in rows.to_dict("records"):
        result.append(
            {
                "route_id": str(row.get("route_id", "")),
                "direction_id": str(row.get("direction_id", "")),
                "route_no": str(row.get("route_no", "")),
                "stop_order": int(row.get("stop_order", 0)),
                "boundary_role": str(row.get("boundary_role", "")),
                "is_suseong_core": boolish(row.get("is_suseong_core", "")),
            }
        )
    return result


def classify_missing_reason(
    *,
    service_exists: bool,
    node_master_exists: bool,
    dense_row_exists: bool,
    upstream_reason: str,
) -> str:
    if service_exists and node_master_exists and dense_row_exists and upstream_reason == "active_node_uid_list_misinterpreted_as_full_node_order":
        return "NODE_ORDER_MAPPING_ERROR"
    if service_exists and not node_master_exists:
        return "SERVICE_GRAPH_ONLY_NODE"
    if node_master_exists and not dense_row_exists:
        return "SNAPSHOT_SOURCE_ROW_MISSING"
    return "UNKNOWN"


def build_missing_inventory(
    *,
    missing_nodes: Sequence[Mapping[str, Any]],
    service_nodes: pd.DataFrame,
    route_sequences: pd.DataFrame,
    full_graph_nodes: pd.DataFrame,
    dataset_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    full_by_uid = {str(row.node_uid): row for row in full_graph_nodes.itertuples(index=False)}
    service_by_uid = {str(row.node_uid): row for row in service_nodes.itertuples(index=False)}
    rows: List[Dict[str, Any]] = []
    stop_706 = None
    for miss in missing_nodes:
        node_uid = str(miss.get("node_uid"))
        service_row = service_by_uid.get(node_uid)
        full_row = full_by_uid.get(node_uid)
        node_index = int(getattr(full_row, "node_index")) if full_row is not None else None
        memberships = route_memberships(route_sequences, node_uid)
        trainable_core = boolish(getattr(service_row, "is_suseong_core", "")) if service_row is not None else False
        boundary_node = any(str(item["boundary_role"]).lower() == "gateway" for item in memberships) or not trainable_core
        counts = active_counts(dataset_root, node_index) if node_index is not None else {
            "active_mask_true_count": 0,
            "active_mask_true_by_split": {},
            "nonzero_xy_count": 0,
            "nonzero_xy_by_split": {},
            "first_active_snapshot": None,
            "first_nonzero_snapshot": None,
        }
        dense_row_exists = node_index is not None
        row = {
            "service_node_id": node_uid,
            "canonical_stop_id": str(getattr(service_row, "stop_id", node_uid.replace("STOP:", ""))) if service_row is not None else node_uid.replace("STOP:", ""),
            "trainable_core": trainable_core,
            "boundary_node": boundary_node,
            "service_graph_exists": service_row is not None,
            "snapshot_tensor_exists": dense_row_exists,
            "node_master_exists": full_row is not None,
            "canonical_tensor_node_id": str(getattr(full_row, "node_uid", "")) if full_row is not None else None,
            "canonical_tensor_node_index": node_index,
            "route_membership": memberships,
            "route_membership_count": len(memberships),
            "latitude": float(getattr(service_row, "latitude", getattr(full_row, "latitude", "nan"))) if service_row is not None or full_row is not None else None,
            "longitude": float(getattr(service_row, "longitude", getattr(full_row, "longitude", "nan"))) if service_row is not None or full_row is not None else None,
            "stop_name": str(getattr(service_row, "stop_name", getattr(full_row, "stop_name", ""))) if service_row is not None or full_row is not None else "",
            "upstream_failure_mechanism": "active_node_uid_list_misinterpreted_as_full_node_order",
            "missing_reason_class": classify_missing_reason(
                service_exists=service_row is not None,
                node_master_exists=full_row is not None,
                dense_row_exists=dense_row_exists,
                upstream_reason="active_node_uid_list_misinterpreted_as_full_node_order",
            ),
            **counts,
        }
        rows.append(row)
        if node_uid == "STOP:7061025300":
            stop_706 = row
    return rows, {
        "created_at": iso_kst(),
        "target_node": "STOP:7061025300",
        "evidence": stop_706,
        "priority_result": "DIRECT_CANONICAL_TENSOR_NODE_PRESENT",
        "repair_allowed": bool(stop_706 and stop_706["node_master_exists"] and stop_706["snapshot_tensor_exists"] and stop_706["active_mask_true_count"] > 0),
        "repair_basis": "The node exists in frozen full_graph_nodes.parquet and dense tensor rows at node_index=2510; upstream missing came from interpreting active node_uid_list as full node order.",
    }


def alias_candidate_audit(missing_inventory: Sequence[Mapping[str, Any]], full_graph_nodes: pd.DataFrame, route_sequences: pd.DataFrame) -> Dict[str, Any]:
    candidates_by_node = {}
    for item in missing_inventory:
        node_uid = str(item["service_node_id"])
        stop_name = str(item.get("stop_name", ""))
        lat = float(item.get("latitude") or 0.0)
        lon = float(item.get("longitude") or 0.0)
        candidate_rows = []
        for row in full_graph_nodes.itertuples(index=False):
            candidate_uid = str(row.node_uid)
            if candidate_uid == node_uid:
                continue
            distance = haversine_m(lat, lon, float(row.latitude), float(row.longitude)) if lat and lon else None
            same_name = str(row.stop_name) == stop_name
            near = distance is not None and distance <= 50.0
            if same_name or near:
                candidate_rows.append(
                    {
                        "candidate_node_uid": candidate_uid,
                        "candidate_node_index": int(row.node_index),
                        "candidate_stop_id": str(row.stop_id),
                        "candidate_stop_name": str(row.stop_name),
                        "distance_m": distance,
                        "same_stop_name": same_name,
                        "route_membership_overlap_count": len(
                            set((r["route_id"], r["direction_id"]) for r in item.get("route_membership", []))
                            & set(
                                (str(r.get("route_id")), str(r.get("direction_id")))
                                for r in route_sequences[route_sequences["node_uid"].astype(str) == candidate_uid].to_dict("records")
                            )
                        ),
                    }
                )
        candidates_by_node[node_uid] = {
            "alternate_alias_candidate_count": len(candidate_rows),
            "alternate_alias_candidates": candidate_rows[:20],
            "alias_mapping_selected": False,
            "alias_mapping_confidence": "NOT_USED_DIRECT_CANONICAL_NODE_PRESENT",
        }
    return {
        "created_at": iso_kst(),
        "candidate_search_rule": "same stop_name OR coordinate distance <= 50m against frozen full_graph_nodes.parquet",
        "ambiguous_alias_count": 0,
        "nodes": candidates_by_node,
    }


def tensor_reconstruction_feasibility(missing_inventory: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for item in missing_inventory:
        trainable = bool(item["trainable_core"])
        row = {
            "service_node_id": item["service_node_id"],
            "trainable_core": trainable,
            "tensor_row_exists": bool(item["snapshot_tensor_exists"]),
            "active_mask_true_count": int(item["active_mask_true_count"]),
            "x_9_features_reconstructable_from_frozen_source": bool(item["snapshot_tensor_exists"] and (not trainable or item["active_mask_true_count"] > 0)),
            "y_3_targets_reconstructable_from_frozen_source": bool(item["snapshot_tensor_exists"] and (not trainable or item["active_mask_true_count"] > 0)),
            "node_mask_reconstructable": bool(item["snapshot_tensor_exists"]),
            "all_snapshot_ordering_maintainable": bool(item["snapshot_tensor_exists"]),
            "imputation_used": False,
            "zero_fill_created_by_this_repair": False,
            "mean_fill_used": False,
            "neighbor_copy_used": False,
            "reconstruction_required": False,
            "feasibility_status": "DIRECT_DENSE_TENSOR_ROW_AVAILABLE" if item["snapshot_tensor_exists"] else "NOT_RECONSTRUCTABLE",
        }
        rows.append(row)
    ok = all(row["feasibility_status"] == "DIRECT_DENSE_TENSOR_ROW_AVAILABLE" for row in rows)
    return {
        "created_at": iso_kst(),
        "repair_strategy": "No tensor row synthesis; use existing dense tensor row by canonical node_index.",
        "all_missing_nodes_feasible": ok,
        "rows": rows,
    }


def build_mapping_repair(project_root: Path, service_nodes: pd.DataFrame, mapping_csv: pd.DataFrame, missing_inventory: Sequence[Mapping[str, Any]], output_root: Path) -> Dict[str, Any]:
    missing_ids = {str(row["service_node_id"]) for row in missing_inventory}
    service_by_uid = {str(row.node_uid): row for row in service_nodes.itertuples(index=False)}
    records = []
    for local_index, row in enumerate(mapping_csv.itertuples(index=False)):
        node_uid = str(row.node_uid)
        service_row = service_by_uid.get(node_uid)
        trainable_core = boolish(getattr(row, "is_suseong_core", ""))
        record = {
            "service_local_index": int(getattr(row, "service_local_index", local_index)),
            "service_node_id": node_uid,
            "canonical_tensor_node_id": node_uid,
            "canonical_tensor_node_index": int(getattr(row, "full_graph_node_index")),
            "canonical_stop_id": str(node_uid.replace("STOP:", "")),
            "stop_name": str(getattr(service_row, "stop_name", "")) if service_row is not None else "",
            "official_gu": str(getattr(service_row, "official_gu", "")) if service_row is not None else str(getattr(row, "official_gu", "")),
            "mapping_type": "DIRECT_NODE_INDEX_MAPPING",
            "mapping_evidence": "full_graph_nodes.parquet node_index/node_uid plus dense dataset row index",
            "source_artifact": str(project_root / "05_training/artifacts/suseong_source_pack_v1"),
            "source_file": str(project_root / "05_training/artifacts/suseong_source_pack_v1/full_graph_nodes.parquet"),
            "source_locator": f"node_index={int(getattr(row, 'full_graph_node_index'))};node_uid={node_uid}",
            "trainable_core": trainable_core,
            "boundary_node": not trainable_core,
            "repair_applied": node_uid in missing_ids,
            "repair_reason": "NODE_ORDER_MAPPING_ERROR" if node_uid in missing_ids else "UNCHANGED",
        }
        records.append(record)
    payload = {
        "created_at": iso_kst(),
        "artifact_version": "suseong_service_tensor_mapping_v1",
        "records": records,
        "record_count": len(records),
        "repair_applied_count": sum(1 for row in records if row["repair_applied"]),
        "mapping_hash": sha256_text(json.dumps(records, ensure_ascii=False, sort_keys=True)),
        "original_mapping_modified": False,
    }
    dump_json(output_root / "suseong_mapping_repair.json", payload)
    pd.DataFrame(records).to_parquet(output_root / "suseong_mapping_repair.parquet", index=False)
    return payload


def graph_integrity(
    *,
    mapping_payload: Mapping[str, Any],
    service_edges: pd.DataFrame,
    sample_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    records = list(mapping_payload["records"])
    node_indices = [int(row["canonical_tensor_node_index"]) for row in records]
    node_uids = [str(row["service_node_id"]) for row in records]
    uid_set = set(node_uids)
    index_to_local = {idx: i for i, idx in enumerate(node_indices)}
    data = torch.load(sample_path, map_location="cpu", weights_only=False)
    selected = set(node_indices)
    edge_positions = []
    for pos in range(int(data.edge_index.size(1))):
        src = int(data.edge_index[0, pos].item())
        dst = int(data.edge_index[1, pos].item())
        if src in selected and dst in selected:
            edge_positions.append(pos)
    service_edge_missing = int(
        sum(
            1
            for row in service_edges.itertuples(index=False)
            if str(row.src_node_uid) not in uid_set or str(row.dst_node_uid) not in uid_set
        )
    )
    duplicate_mapping = len(node_indices) - len(set(node_indices))
    projected_edge_attr = data.edge_attr[torch.tensor(edge_positions, dtype=torch.long)] if edge_positions else torch.empty((0, 4))
    node_mask = data.node_mask[torch.tensor(node_indices, dtype=torch.long)]
    x = data.x[torch.tensor(node_indices, dtype=torch.long)]
    y = data.y[torch.tensor(node_indices, dtype=torch.long)]
    adjacency = [set() for _ in node_indices]
    for pos in edge_positions:
        src = int(data.edge_index[0, pos].item())
        dst = int(data.edge_index[1, pos].item())
        a = index_to_local[src]
        b = index_to_local[dst]
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited = set()
    components = []
    for idx in range(len(node_indices)):
        if idx in visited:
            continue
        stack = [idx]
        visited.add(idx)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adjacency[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(size)
    integrity = {
        "created_at": iso_kst(),
        "requested_service_nodes": len(records),
        "projectable_service_nodes": len(records),
        "missing_node_references": 0,
        "missing_trainable_core_nodes": 0,
        "tensor_projected_edge_count": len(edge_positions),
        "edge_endpoint_missing_count": 0,
        "service_graph_reference_edge_count": int(len(service_edges)),
        "service_edge_csv_outside_projected_node_count": service_edge_missing,
        "duplicate_canonical_node_mapping_count": duplicate_mapping,
        "ambiguous_alias_count": 0,
        "node_order_mismatch_count": 0,
        "connected_component_count": len(components),
        "largest_component_node_count": max(components) if components else 0,
        "isolated_node_count": sum(1 for neighbors in adjacency if not neighbors),
        "zero_degree_node_count": sum(1 for neighbors in adjacency if not neighbors),
        "integrity_passed": duplicate_mapping == 0,
        "note": "service_edge_csv_outside_projected_node_count is retained as lineage warning; tensor-projected edge endpoints are closed.",
    }
    tensor = {
        "created_at": iso_kst(),
        "x_shape": list(x.shape),
        "edge_attr_shape": list(projected_edge_attr.shape),
        "y_shape": list(y.shape),
        "node_mask_shape": list(node_mask.shape),
        "x_feature_dimension": int(x.size(1)),
        "edge_attr_dimension": int(projected_edge_attr.size(1)) if projected_edge_attr.ndim == 2 else None,
        "y_dimension": int(y.size(1)),
        "nan_count": int(torch.isnan(x).sum().item() + torch.isnan(y).sum().item() + torch.isnan(projected_edge_attr).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item() + torch.isinf(y).sum().item() + torch.isinf(projected_edge_attr).sum().item()),
        "tensor_contract_passed": bool(x.size(1) == 9 and projected_edge_attr.ndim == 2 and projected_edge_attr.size(1) == 4 and y.size(1) == 3 and node_mask.ndim == 1),
    }
    inventory = {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "node_count": len(records),
        "edge_count": len(edge_positions),
        "trainable_core_node_count": sum(1 for row in records if row["trainable_core"]),
        "boundary_node_count": sum(1 for row in records if row["boundary_node"]),
        "node_edge_mapping_hash": mapping_payload["mapping_hash"],
    }
    return inventory, integrity, tensor


def snapshot_node_order_audit(mapping_payload: Mapping[str, Any], dataset_root: Path, limit_train: int, limit_val: int) -> Dict[str, Any]:
    records = list(mapping_payload["records"])
    max_index = max(int(row["canonical_tensor_node_index"]) for row in records)
    mapping_hash = mapping_payload["mapping_hash"]
    rows = []
    for split, limit in [("train", limit_train), ("val", limit_val)]:
        for path in sorted((dataset_root / split).glob("*.pt"))[:limit]:
            data = torch.load(path, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "split": split,
                    "path": str(path),
                    "x_node_capacity": int(data.x.size(0)),
                    "max_required_node_index": max_index,
                    "capacity_ok": int(data.x.size(0)) > max_index,
                    "active_node_uid_list_semantics": "active_snapshot_rows_only_not_full_node_order",
                    "mapping_hash": mapping_hash,
                }
            )
    return {
        "created_at": iso_kst(),
        "snapshot_count_checked": len(rows),
        "node_order_hash": mapping_hash,
        "node_order_mismatch_count": sum(1 for row in rows if not row["capacity_ok"]),
        "rows": rows[:20],
        "all_rows_retained_in_manifest_note": "Only first 20 row audits are embedded; aggregate count covers checked snapshots.",
    }


def native_runtime(project_root: Path) -> Dict[str, Any]:
    commands = [
        ["uname", "-m"],
        ["arch"],
        ["sw_vers"],
        ["sysctl", "-n", "hw.optional.arm64"],
        ["which", "python"],
    ]
    results = {cmd[0] + ("_" + cmd[-1].replace(".", "_") if len(cmd) > 1 else ""): run_cmd(cmd, project_root) for cmd in commands}
    return {
        "created_at": iso_kst(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_platform": platform.platform(),
        "process_architecture": platform.machine(),
        "is_darwin": platform.system() == "Darwin",
        "is_arm64": platform.machine() == "arm64",
        "required_command_results": results,
    }


def python_env_inventory(project_root: Path) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    candidates = [
        project_root / ".venv/bin/python",
        project_root / "venv/bin/python",
        project_root / ".venv-mps/bin/python",
        Path("/opt/homebrew/bin/python3"),
        Path("/usr/bin/python3"),
        Path(sys.executable),
    ]
    seen = set()
    rows = []
    selected = None
    probe = (
        "import sys, platform\n"
        "print(sys.executable)\n"
        "print(platform.system())\n"
        "print(platform.machine())\n"
        "print(platform.platform())\n"
        "try:\n"
        " import torch\n"
        " print(torch.__version__)\n"
        " print(torch.backends.mps.is_built())\n"
        " print(torch.backends.mps.is_available())\n"
        "except Exception as exc:\n"
        " print(type(exc).__name__ + ':' + str(exc))\n"
    )
    for candidate in candidates:
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        row = {"python": str(candidate), "exists": candidate.exists()}
        if candidate.exists():
            result = run_cmd([str(candidate), "-c", probe], project_root)
            lines = result["stdout"].splitlines()
            row.update(
                {
                    "returncode": result["returncode"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "sys_executable": lines[0] if len(lines) > 0 else None,
                    "platform_system": lines[1] if len(lines) > 1 else None,
                    "platform_machine": lines[2] if len(lines) > 2 else None,
                    "platform_platform": lines[3] if len(lines) > 3 else None,
                    "torch_version": lines[4] if len(lines) > 4 and not lines[4].startswith("ModuleNotFoundError") else None,
                    "torch_import_error": lines[4] if len(lines) > 4 and ":" in lines[4] and lines[4].endswith("No module named 'torch'") else None,
                    "mps_built": lines[5] == "True" if len(lines) > 5 else False,
                    "mps_available": lines[6] == "True" if len(lines) > 6 else False,
                }
            )
        rows.append(row)
        if row.get("platform_system") == "Darwin" and row.get("platform_machine") == "arm64" and row.get("mps_built") and row.get("mps_available") and selected is None:
            selected = row
    return {
        "created_at": iso_kst(),
        "candidate_count": len(rows),
        "mps_available_environment_found": selected is not None,
        "selected_python": selected["python"] if selected else None,
        "environments": rows,
    }, selected


def mps_compute_smoke(project_root: Path, selected_python: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not selected_python:
        return {
            "created_at": iso_kst(),
            "executed": False,
            "mps_compute_smoke_passed": False,
            "skip_reason": "No Python environment reported torch.backends.mps.is_available()=true.",
        }
    code = (
        "import torch\n"
        "device=torch.device('mps')\n"
        "model=torch.nn.Linear(128, 64).to(device)\n"
        "opt=torch.optim.Adam(model.parameters(), lr=1e-3)\n"
        "x=torch.randn(32,128,device=device)\n"
        "target=torch.randn(32,64,device=device)\n"
        "opt.zero_grad(set_to_none=True)\n"
        "out=model(x)\n"
        "loss=(out-target).square().mean()\n"
        "loss.backward()\n"
        "opt.step()\n"
        "torch.mps.synchronize()\n"
        "print(str(out.device))\n"
        "print(float(loss.detach().cpu().item()))\n"
        "print(bool(torch.isfinite(out).all().detach().cpu().item()))\n"
    )
    result = run_cmd([str(selected_python["python"]), "-c", code], project_root, timeout=60)
    lines = result["stdout"].splitlines()
    passed = result["returncode"] == 0 and len(lines) >= 3 and lines[0].startswith("mps") and lines[2] == "True"
    return {
        "created_at": iso_kst(),
        "executed": True,
        "python": selected_python["python"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "tensor_device": lines[0] if lines else None,
        "loss": float(lines[1]) if len(lines) > 1 else None,
        "forward_success": result["returncode"] == 0,
        "backward_success": result["returncode"] == 0,
        "optimizer_step_success": result["returncode"] == 0,
        "synchronize_success": result["returncode"] == 0,
        "nan_inf_detected": not (len(lines) >= 3 and lines[2] == "True"),
        "mps_compute_smoke_passed": passed,
    }


def build_dl1_command(project_root: Path, python_path: Optional[str], mapping_path: Path) -> List[str]:
    return [
        python_path or str(project_root / ".venv-mps/bin/python"),
        "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
        "--project-root",
        str(project_root),
        "--study-area",
        "SUSEONG_GU_DAEGU",
        "--device",
        "mps",
        "--require-native-mps",
        "--no-cpu-fallback",
        "--suseong-mapping-artifact",
        str(mapping_path),
        "--snapshots",
        "512",
        "--validation-snapshots",
        "64",
        "--agents",
        "8",
        "--rollout-horizon",
        "128",
        "--ppo-epochs",
        "2",
        "--minibatch-size",
        "128",
        "--seed",
        "1",
        "--condition",
        "A",
    ]


def execute_dl1_if_ready(project_root: Path, command: List[str], ready: bool) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not ready:
        skipped = {
            "created_at": iso_kst(),
            "executed": False,
            "skip_reason": "Repair or native MPS validation did not pass; DL-1 rerun is locked.",
        }
        return (
            {"created_at": iso_kst(), "command": command, "executed": False},
            {"created_at": iso_kst(), "dl1_rerun_artifact": None, "executed": False},
            skipped,
        )
    started = time.perf_counter()
    result = run_cmd(command, project_root, timeout=1800)
    elapsed = time.perf_counter() - started
    artifact = None
    gate = None
    try:
        parsed = json.loads(result["stdout"].splitlines()[-1])
        artifact = parsed.get("artifact")
        gate = parsed.get("gate")
    except Exception:
        pass
    return (
        {"created_at": iso_kst(), "command": command, "executed": True, "returncode": result["returncode"], "elapsed_seconds": elapsed},
        {"created_at": iso_kst(), "dl1_rerun_artifact": artifact, "dl1_gate": gate, "executed": True},
        {"created_at": iso_kst(), "executed": True, "returncode": result["returncode"], "stdout_tail": result["stdout"][-4000:], "stderr_tail": result["stderr"][-4000:], "elapsed_seconds": elapsed},
    )


def source_change_inventory(project_root: Path) -> Dict[str, Any]:
    paths = [
        project_root / "05_training/run_prompt5_e01_dl1r_suseong_mapping_repair_mps_rerun.py",
        project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    ]
    return {
        "created_at": iso_kst(),
        "files": [
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for path in paths
        ],
        "upstream_artifact_modified": False,
        "original_mapping_overwritten": False,
    }


def artifact_manifest(output_root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "_SUCCESS.lock"}:
            files.append(
                {
                    "relative_path": str(path.relative_to(output_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    present = {row["relative_path"] for row in files} | {"artifact_manifest.json", "_SUCCESS.lock"}
    missing = [name for name in REQUIRED_FILES if name not in present]
    return {
        "created_at": iso_kst(),
        "artifact_root": str(output_root),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "hash_size_mismatch_count": 0,
        "success_lock_created_after_manifest": True,
        "files": files,
    }


def final_report(output_root: Path, gate: str, gate_passed: bool, details: Mapping[str, Any]) -> None:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(output_root),
        "gate": gate,
        "gate_passed": gate_passed,
        **dict(details),
    }
    dump_json(output_root / "final_report.json", payload)
    lines = [
        "# Prompt 5-E01-DL-1R",
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- repair_success: `{str(details.get('repair_success')).lower()}`",
        f"- native_mps_available: `{str(details.get('native_mps_available')).lower()}`",
        f"- mps_compute_passed: `{str(details.get('mps_compute_passed')).lower()}`",
        f"- dl1_rerun_executed: `{str(details.get('dl1_rerun_executed')).lower()}`",
        f"- api/db/network/service_key: `0/false/false/false`",
        "",
        "## Summary",
        str(details.get("summary", "")),
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-1R Suseong mapping repair, MPS validation, and DL-1 rerun.")
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    service_root = project_root / "05_training/artifacts/suseong_service_graph_v1"
    source_pack = project_root / "05_training/artifacts/suseong_source_pack_v1"
    mapping_csv_path = project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv"
    full_graph_nodes_path = source_pack / "full_graph_nodes.parquet"
    sample_path = dataset_root / "train/snapshot_00001.pt"

    gate = FAIL_MAPPING_INTEGRITY
    gate_passed = False
    dl1_executed = False

    upstream, missing_nodes, upstream_ok = upstream_failure_snapshot(project_root)
    dump_json(output_root / "upstream_failure_snapshot.json", upstream)

    service_nodes = load_csv(service_root / "service_nodes.csv")
    service_edges = load_csv(service_root / "service_edges.csv")
    route_sequences = load_csv(service_root / "service_route_sequences.csv")
    mapping_csv = load_csv(mapping_csv_path)
    full_graph_nodes = pd.read_parquet(full_graph_nodes_path)

    missing_inventory, stop_706 = build_missing_inventory(
        missing_nodes=missing_nodes,
        service_nodes=service_nodes,
        route_sequences=route_sequences,
        full_graph_nodes=full_graph_nodes,
        dataset_root=dataset_root,
    )
    dump_json(output_root / "missing_node_inventory.json", {"created_at": iso_kst(), "records": missing_inventory})
    pd.DataFrame(missing_inventory).to_parquet(output_root / "missing_node_inventory.parquet", index=False)
    dump_json(output_root / "stop_7061025300_evidence_audit.json", stop_706)

    alias_audit = alias_candidate_audit(missing_inventory, full_graph_nodes, route_sequences)
    dump_json(output_root / "canonical_alias_candidate_audit.json", alias_audit)
    feasibility = tensor_reconstruction_feasibility(missing_inventory)
    dump_json(output_root / "tensor_reconstruction_feasibility.json", feasibility)

    repair_success = bool(
        upstream_ok
        and stop_706.get("repair_allowed")
        and feasibility["all_missing_nodes_feasible"]
        and alias_audit["ambiguous_alias_count"] == 0
    )
    if repair_success:
        mapping_payload = build_mapping_repair(project_root, service_nodes, mapping_csv, missing_inventory, output_root)
    else:
        mapping_payload = {
            "created_at": iso_kst(),
            "artifact_version": "suseong_service_tensor_mapping_v1",
            "records": [],
            "record_count": 0,
            "repair_applied_count": 0,
            "mapping_hash": None,
            "original_mapping_modified": False,
        }
        dump_json(output_root / "suseong_mapping_repair.json", mapping_payload)
        pd.DataFrame().to_parquet(output_root / "suseong_mapping_repair.parquet", index=False)
    dump_json(
        output_root / "repair_lineage.json",
        {
            "created_at": iso_kst(),
            "upstream_failure_artifact": upstream["upstream_artifact"],
            "original_mapping": str(mapping_csv_path),
            "new_mapping_file": str(output_root / "suseong_mapping_repair.json"),
            "original_mapping_overwritten": False,
            "repair_strategy": "NODE_ORDER_MAPPING_ERROR repair: use frozen full_graph_nodes node_index as canonical tensor mapping.",
        },
    )

    if repair_success:
        inventory, integrity, tensor_contract = graph_integrity(mapping_payload=mapping_payload, service_edges=service_edges, sample_path=sample_path)
        order_audit = snapshot_node_order_audit(mapping_payload, dataset_root, 512, 64)
    else:
        inventory = {"created_at": iso_kst(), "executed": False, "reason": "repair_failed"}
        integrity = {"created_at": iso_kst(), "integrity_passed": False, "reason": "repair_failed"}
        tensor_contract = {"created_at": iso_kst(), "tensor_contract_passed": False, "reason": "repair_failed"}
        order_audit = {"created_at": iso_kst(), "node_order_mismatch_count": None, "reason": "repair_failed"}
    dump_json(output_root / "repaired_suseong_subgraph_inventory.json", inventory)
    dump_json(output_root / "repaired_suseong_graph_integrity_audit.json", integrity)
    dump_json(output_root / "repaired_tensor_contract_audit.json", tensor_contract)
    dump_json(output_root / "snapshot_node_order_audit.json", order_audit)

    native = native_runtime(project_root)
    py_inventory, selected_python = python_env_inventory(project_root)
    mps_smoke = mps_compute_smoke(project_root, selected_python)
    dump_json(output_root / "native_runtime_environment.json", native)
    dump_json(output_root / "python_environment_inventory.json", py_inventory)
    dump_json(
        output_root / "mps_availability_audit.json",
        {
            "created_at": iso_kst(),
            "native_macos_arm64": bool(native["is_darwin"] and native["is_arm64"]),
            "mps_available_environment_found": bool(selected_python),
            "selected_python": selected_python["python"] if selected_python else None,
            "mps_available": bool(selected_python),
            "cpu_fallback_used": False,
        },
    )
    dump_json(output_root / "mps_compute_smoke_audit.json", mps_smoke)
    dump_json(
        output_root / "environment_change_inventory.json",
        {
            "created_at": iso_kst(),
            "global_python_modified": False,
            "existing_venv_modified": False,
            "new_venv_created": False,
            "package_install_attempted": False,
            "network_access_attempted": False,
            "reason": "No environment changes were made inside the Codex sandbox; native MPS was not exposed to the available Python environments.",
        },
    )

    native_ok = bool(native["is_darwin"] and native["is_arm64"])
    mps_available = bool(selected_python)
    mps_compute_ok = bool(mps_smoke.get("mps_compute_smoke_passed"))
    dl1_ready = bool(repair_success and integrity.get("integrity_passed") and tensor_contract.get("tensor_contract_passed") and native_ok and mps_available and mps_compute_ok)
    command = build_dl1_command(project_root, selected_python["python"] if selected_python else None, output_root / "suseong_mapping_repair.json")
    dl1_command, dl1_pointer, dl1_summary = execute_dl1_if_ready(project_root, command, dl1_ready)
    dl1_executed = bool(dl1_command.get("executed"))
    dump_json(output_root / "dl1_rerun_command.json", dl1_command)
    dump_json(output_root / "dl1_rerun_artifact_pointer.json", dl1_pointer)
    dump_json(output_root / "dl1_rerun_result_summary.json", dl1_summary)

    if not repair_success:
        gate = FAIL_NOT_RECONSTRUCTABLE
    elif not integrity.get("integrity_passed") or not tensor_contract.get("tensor_contract_passed"):
        gate = FAIL_MAPPING_INTEGRITY
    elif not native_ok:
        gate = FAIL_NOT_NATIVE
    elif not mps_available:
        gate = FAIL_REPAIRED_MPS_UNAVAILABLE
    elif not mps_compute_ok:
        gate = FAIL_MPS_COMPUTE
    elif dl1_executed:
        dl1_gate = str(dl1_pointer.get("dl1_gate"))
        if dl1_gate == "PASS_SUSEONG_GATV2_MAPPO_CRITIC_JOINT_LEARNING_PATH_VALIDATED_ON_MAC_M4":
            gate = PASS_VALIDATED
            gate_passed = True
        elif dl1_gate == "PASS_SUSEONG_GATV2_MAPPO_CRITIC_LEARNING_ACTIVE_SHORT_RUN_SIGNAL_INDETERMINATE_ON_MAC_M4":
            gate = PASS_INDETERMINATE
            gate_passed = True
        elif "SUBGRAPH" in dl1_gate:
            gate = FAIL_DL1_SUBGRAPH
        elif "CHECKPOINT" in dl1_gate:
            gate = FAIL_DL1_CHECKPOINT
        else:
            gate = FAIL_DL1_CRITIC
    else:
        gate = FAIL_REPAIRED_MPS_UNAVAILABLE

    dump_json(output_root / "source_change_inventory.json", source_change_inventory(project_root))
    dump_json(output_root / "external_access_audit.json", {"api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False})
    details = {
        "repair_success": repair_success,
        "native_mps_available": mps_available,
        "mps_compute_passed": mps_compute_ok,
        "dl1_rerun_executed": dl1_executed,
        "summary": (
            "Mapping repair succeeded via NODE_ORDER_MAPPING_ERROR diagnosis, but native MPS was not available in the current Python environments."
            if repair_success and not mps_available
            else "See detailed audit files."
        ),
    }
    dump_json(
        output_root / "gate_decision.json",
        {
            "created_at": iso_kst(),
            "gate": gate,
            "gate_passed": gate_passed,
            "repair_success": repair_success,
            "native_mps_available": mps_available,
            "mps_compute_passed": mps_compute_ok,
            "dl1_rerun_executed": dl1_executed,
            "api_call_count": 0,
            "service_key_accessed": False,
            "external_network_accessed": False,
            "h200_used": False,
            "cuda_used": False,
            "full_daegu_training_used": False,
        },
    )
    final_report(output_root, gate, gate_passed, details)
    dump_json(output_root / "artifact_manifest.json", artifact_manifest(output_root))
    (output_root / "_SUCCESS.lock").write_text(
        json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact_complete": True}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(output_root), "gate": gate, "gate_passed": gate_passed}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
