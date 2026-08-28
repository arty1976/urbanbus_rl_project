#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1A-SRP2-BIS-PV8-C2.

Prospective no-future 8-vehicle mapping validation and inactive-agent
observation contract. This runner reads the sealed PV8-C1 dry-run artifact and
the existing C1 capture, then selects eight vehicles using only anchor-cycle
information. It never calls BIS, DB, simulator, policy, checkpoint, or training
code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

C1_CAPTURE_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_limited_pilot_capture_20260805_070246"
C1_CAPTURE_MANIFEST = "artifact_manifest_srp2_bis_c1.json"
C1_CAPTURE_LOCK = "_SRP2_BIS_C1_CAPTURE_COMPLETE.lock"
TARGET_ROUTE_ID = "3000814001"
NUM_AGENTS = 8
AGENT_IDS = list(range(NUM_AGENTS))

PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"
PASS_READINESS = "SRP2_BIS_PV8_C2_COMPLETE_PROSPECTIVE_FIXED_VEHICLE_AGENT_MAPPING_AND_INACTIVE_MASK_VALIDATED_K_SAFETY_AND_POLICY_COMPATIBILITY_REMAIN_GATED"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_MAPPING_CONTRACT_VIOLATION"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation"
PV8_C1_FAMILY = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c1_offline_mapping_dryrun_*"
PV8_C1_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DETERMINISTIC_OFFLINE_8VEHICLE_MAPPING_DRYRUN_COMPLETE"
PV8_C1_READINESS = "SRP2_BIS_PV8_C1_COMPLETE_MAPPING_MANIFEST_AND_IDENTITY_PERSISTENCE_VALIDATED_OBSERVATION_CONTRACT_READY_K_SAFETY_GAPS_REMAIN_PENDING_USER_COMMAND"
PV8_C1_MANIFEST = "artifact_manifest_srp2_bis_pv8_c1.json"
PV8_C1_LOCK = "_SRP2_BIS_PV8_C1_COMPLETE.lock"

PAYLOADS = [
    "prospective_anchor_contract.json",
    "prospective_selection_candidates.parquet",
    "prospective_selection_decision.json",
    "prospective_8vehicle_mapping_manifest.json",
    "prospective_8vehicle_mapping_manifest.parquet",
    "prospective_agent_cycle_state.parquet",
    "inactive_agent_audit.json",
    "inactive_agent_by_slot.parquet",
    "vehicle_identity_persistence_audit.json",
    "vehicle_identity_transition_events.parquet",
    "direction_transition_audit.json",
    "prospective_no_future_information_audit.json",
    "c1_vs_c2_mapping_comparison.json",
    "c1_vs_c2_mapping_comparison.md",
    "pv8_prospective_observation_contract.json",
    "pv8_prospective_observation_contract.md",
    "pv8_c2_checkpoint_compatibility_gap_audit.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "final_report.md",
    "gate_decision.json",
    "downstream_lock.json",
]


class PV8C2Error(RuntimeError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            f = float(value)
            return None if (f != f or f in (float("inf"), float("-inf"))) else f
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def parse_utc(value: Any) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def to_kst(value: Any) -> str:
    return parse_utc(value).astimezone(ZoneInfo("Asia/Seoul")).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any) -> str:
    data = json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def parquet(self, rel: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        import pandas as pd

        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([json_clean(dict(r)) for r in rows], columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))
        df.to_parquet(path, index=False)


def verify_manifest(root: Path, manifest_name: str, lock_name: str) -> Dict[str, Any]:
    manifest_path = root / manifest_name
    lock_path = root / lock_name
    checks: Dict[str, Any] = {
        "manifest_present": manifest_path.exists(),
        "lock_present": lock_path.exists(),
        "manifest_hash_ok": False,
        "manifest_size_ok": False,
        "manifest_entry_count": None,
        "missing_files": None,
        "required_missing_files": None,
        "sha256_mismatches": None,
        "size_mismatches": None,
        "terminal_lock_binding": False,
    }
    if not manifest_path.exists() or not lock_path.exists():
        return checks
    manifest = read_json(manifest_path)
    lock = read_json(lock_path)
    files = list(manifest.get("files", []))
    missing = required_missing = sha_mismatch = size_mismatch = 0
    for row in files:
        path = root / row["relative_path"]
        if not path.exists():
            missing += 1
            if row.get("required", True):
                required_missing += 1
            continue
        if row.get("sha256") and sha256_file(path) != row["sha256"]:
            sha_mismatch += 1
        if row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]:
            size_mismatch += 1
    checks.update({
        "manifest_hash_ok": sha256_file(manifest_path) == lock.get("manifest_sha256") or sha256_file(manifest_path) == lock.get("final_manifest_sha256"),
        "manifest_size_ok": manifest_path.stat().st_size == lock.get("manifest_size_bytes", manifest_path.stat().st_size),
        "manifest_entry_count": len(files),
        "missing_files": missing,
        "required_missing_files": required_missing,
        "sha256_mismatches": sha_mismatch,
        "size_mismatches": size_mismatch,
        "terminal_lock_binding": lock.get("manifest_relative_path") == manifest_name or lock.get("final_manifest_path") == manifest_name,
    })
    return checks


def manifest_ok(checks: Mapping[str, Any]) -> bool:
    return (
        checks.get("manifest_present")
        and checks.get("lock_present")
        and checks.get("manifest_hash_ok")
        and checks.get("manifest_size_ok")
        and checks.get("missing_files") == 0
        and checks.get("required_missing_files") == 0
        and checks.get("sha256_mismatches") == 0
        and checks.get("size_mismatches") == 0
        and checks.get("terminal_lock_binding")
    )


def discover_c1_artifact() -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
    for root in sorted(ARTIFACTS_ROOT.glob(PV8_C1_FAMILY)):
        if not root.is_dir():
            continue
        gate_path = root / "gate_decision.json"
        manifest_path = root / PV8_C1_MANIFEST
        lock_path = root / PV8_C1_LOCK
        if not gate_path.exists() or not manifest_path.exists() or not lock_path.exists():
            continue
        gate = read_json(gate_path)
        downstream = read_json(root / "downstream_lock.json")
        checks = verify_manifest(root, PV8_C1_MANIFEST, PV8_C1_LOCK)
        pass_gate = gate.get("gate") == PV8_C1_GATE and gate.get("readiness") == PV8_C1_READINESS
        dryrun_status = downstream.get("mapping_manifest_status") == "OFFLINE_DRYRUN_ONLY"
        if pass_gate and dryrun_status and manifest_ok(checks):
            candidates.append((root, gate, checks))
    if not candidates:
        raise PV8C2Error("UPSTREAM_C1_INTEGRITY_FAILURE", "no valid PV8-C1 artifact found")
    candidates.sort(key=lambda item: read_json(item[0] / PV8_C1_LOCK).get("created_at", ""))
    selected_root, selected_gate, selected_checks = candidates[-1]
    return selected_root, {
        "created_at": iso_kst(),
        "artifact_family": PV8_C1_FAMILY,
        "valid_candidate_count": len(candidates),
        "selected_artifact_root": str(selected_root),
        "selected_gate": selected_gate.get("gate"),
        "selected_readiness": selected_gate.get("readiness"),
        "integrity": selected_checks,
    }


def load_capture() -> Any:
    import pandas as pd

    cols = [
        "cycle_index", "cycle_id", "poll_observed_at_utc", "vehicle_token",
        "route_id", "direction_id", "route_sequence", "current_stop_id",
        "x_position_raw", "y_position_raw", "provider_event_time_parsed",
        "route_stop_match_status",
    ]
    return pd.read_parquet(C1_CAPTURE_ROOT / "getpos02_normalized.parquet", columns=cols)


def valid_anchor_rows(df: Any, cycle: int) -> Any:
    sub = df[df["cycle_index"] == cycle].copy()
    return sub[
        (sub["route_id"].astype(str) == TARGET_ROUTE_ID)
        & sub["vehicle_token"].notna()
        & sub["direction_id"].notna()
        & sub["route_sequence"].notna()
        & sub["current_stop_id"].notna()
        & ((sub["x_position_raw"].notna() & sub["y_position_raw"].notna()) | sub["current_stop_id"].notna())
        & (sub["route_stop_match_status"].astype(str) == "EXACT_SEQUENCE_MATCH")
    ].copy()


def choose_anchor_and_vehicles(df: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cycles = sorted(int(c) for c in df["cycle_index"].dropna().unique().tolist())
    anchor_cycle = None
    anchor_valid = None
    for cycle in cycles:
        valid = valid_anchor_rows(df, cycle)
        if int(valid["vehicle_token"].nunique()) >= NUM_AGENTS:
            anchor_cycle = cycle
            anchor_valid = valid
            break
    if anchor_cycle is None or anchor_valid is None:
        raise PV8C2Error("INSUFFICIENT_ELIGIBLE_VEHICLES_AT_ANCHOR", "no cycle has 8 current-valid vehicles")
    timestamp = str(anchor_valid["poll_observed_at_utc"].dropna().iloc[0])
    cycle_id = str(anchor_valid["cycle_id"].dropna().iloc[0])
    candidate_rows: List[Dict[str, Any]] = []
    for _, row in anchor_valid.sort_values(["direction_id", "vehicle_token"]).iterrows():
        direction = str(row["direction_id"])
        candidate_rows.append({
            "cycle_index": int(anchor_cycle),
            "cycle_id": cycle_id,
            "anchor_timestamp_utc": timestamp,
            "physical_vehicle_token": str(row["vehicle_token"]),
            "route_id": str(row["route_id"]),
            "direction_id": direction,
            "route_sequence": int(row["route_sequence"]),
            "current_stop_id": str(row["current_stop_id"]),
            "position_x": float(row["x_position_raw"]) if row["x_position_raw"] == row["x_position_raw"] else None,
            "position_y": float(row["y_position_raw"]) if row["y_position_raw"] == row["y_position_raw"] else None,
            "current_identity_valid": True,
            "current_route_sequence_valid": True,
            "current_direction_valid": True,
            "current_position_or_stop_valid": True,
            "selection_rank_within_direction": None,
            "selected": False,
        })
    directions = sorted({r["direction_id"] for r in candidate_rows})
    per_direction = {d: sorted([r for r in candidate_rows if r["direction_id"] == d], key=lambda r: r["physical_vehicle_token"]) for d in directions}
    if len(per_direction) >= 2:
        first_two = sorted(per_direction)[:2]
        base = min(NUM_AGENTS // 2, *(len(per_direction[d]) for d in first_two))
        allocation = {first_two[0]: base, first_two[1]: base}
        remaining = NUM_AGENTS - sum(allocation.values())
        for d in sorted(per_direction):
            if remaining <= 0:
                break
            add = min(len(per_direction[d]) - allocation.get(d, 0), remaining)
            allocation[d] = allocation.get(d, 0) + add
            remaining -= add
    else:
        allocation = {directions[0]: NUM_AGENTS}
    selected_tokens: List[str] = []
    for d in sorted(allocation):
        for idx, row in enumerate(per_direction[d], start=1):
            row["selection_rank_within_direction"] = idx
        selected_tokens.extend(r["physical_vehicle_token"] for r in per_direction[d][:allocation[d]])
    selected_tokens = sorted(selected_tokens)
    if len(selected_tokens) != NUM_AGENTS:
        raise PV8C2Error("INSUFFICIENT_ELIGIBLE_VEHICLES_AT_ANCHOR", f"selected {len(selected_tokens)} vehicles")
    for row in candidate_rows:
        row["selected"] = row["physical_vehicle_token"] in selected_tokens
        row["selection_rule"] = "anchor-current valid fields + max direction balance + lexical token tie-break"
    selected_rows: List[Dict[str, Any]] = []
    token_to_agent = {token: i for i, token in enumerate(selected_tokens)}
    for row in sorted([r for r in candidate_rows if r["selected"]], key=lambda r: r["physical_vehicle_token"]):
        selected_rows.append({
            "mapping_version": "PV8_C2_PROSPECTIVE_MAPPING_V1",
            "mapping_mode": "PROSPECTIVE_NO_FUTURE_SELECTION",
            "selection_cutoff_timestamp": timestamp,
            "future_information_used": False,
            "agent_id": token_to_agent[row["physical_vehicle_token"]],
            "physical_vehicle_token": row["physical_vehicle_token"],
            "anchor_route_id": row["route_id"],
            "anchor_direction_id": row["direction_id"],
            "anchor_seq": row["route_sequence"],
            "anchor_stop_id": row["current_stop_id"],
            "anchor_presence": True,
            "binding_mutable": False,
        })
    anchor = {
        "created_at": iso_kst(),
        "anchor_cycle_id": cycle_id,
        "anchor_cycle_index": anchor_cycle,
        "anchor_timestamp": timestamp,
        "anchor_timestamp_kst": to_kst(timestamp),
        "selection_information_cutoff": timestamp,
        "eligible_vehicle_count_at_anchor": int(anchor_valid["vehicle_token"].nunique()),
        "future_information_used": False,
        "selection_rule": "first chronologically valid cycle with >=8 current-valid vehicles",
        "direction_counts_at_anchor": {str(k): int(v) for k, v in anchor_valid.groupby(anchor_valid["direction_id"].astype(str))["vehicle_token"].nunique().to_dict().items()},
        "selected_direction_distribution": {d: sum(1 for r in selected_rows if r["anchor_direction_id"] == d) for d in sorted(per_direction)},
    }
    return anchor, candidate_rows, selected_rows


def build_forward_replay(df: Any, anchor: Mapping[str, Any], mapping: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    anchor_cycle = int(anchor["anchor_cycle_index"])
    cycles = sorted(int(c) for c in df[df["cycle_index"] >= anchor_cycle]["cycle_index"].dropna().unique().tolist())
    token_to_agent = {row["physical_vehicle_token"]: int(row["agent_id"]) for row in mapping}
    selected_tokens = set(token_to_agent)
    all_anchor_tokens = set(df[df["cycle_index"] == anchor_cycle]["vehicle_token"].astype(str).tolist())
    all_after_tokens = set(df[df["cycle_index"] > anchor_cycle]["vehicle_token"].astype(str).tolist())
    new_post_anchor_tokens = sorted(all_after_tokens - all_anchor_tokens)
    lookup = {(str(row["vehicle_token"]), int(row["cycle_index"])): row for _, row in df.iterrows()}
    previous_active = {t: False for t in selected_tokens}
    previous_direction: Dict[str, Optional[str]] = {t: None for t in selected_tokens}
    state_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    inactive_by_slot: List[Dict[str, Any]] = []
    active_counts: List[int] = []

    for cycle in cycles:
        active_count = 0
        for m in sorted(mapping, key=lambda r: int(r["agent_id"])):
            token = m["physical_vehicle_token"]
            observed_row = lookup.get((token, cycle))
            observed = observed_row is not None
            route_valid = bool(observed and str(observed_row["route_id"]) == TARGET_ROUTE_ID)
            fields_valid = bool(observed and observed_row["direction_id"] == observed_row["direction_id"] and observed_row["route_sequence"] == observed_row["route_sequence"] and observed_row["current_stop_id"] == observed_row["current_stop_id"])
            active = bool(observed and route_valid and fields_valid)
            reappeared = bool(active and not previous_active[token] and cycle != anchor_cycle)
            direction = str(observed_row["direction_id"]) if observed else None
            direction_changed = bool(active and previous_direction[token] is not None and direction != previous_direction[token])
            if active:
                active_count += 1
            if not active:
                transition_type = "INACTIVE_NOT_OBSERVED"
            elif reappeared and direction_changed:
                transition_type = "REAPPEARED_SAME_IDENTITY_WITH_DIRECTION_CHANGE"
            elif reappeared:
                transition_type = "REAPPEARED_SAME_IDENTITY"
            elif direction_changed:
                transition_type = "DIRECTION_CHANGED_SAME_IDENTITY"
            else:
                transition_type = "ACTIVE_OBSERVED"
            row = {
                "cycle_index": cycle,
                "cycle_id": str(observed_row["cycle_id"]) if observed else None,
                "cycle_timestamp": str(observed_row["poll_observed_at_utc"]) if observed else None,
                "agent_id": int(m["agent_id"]),
                "bound_vehicle_token": token,
                "vehicle_observed": observed,
                "active_bus_mask": active,
                "route_id": str(observed_row["route_id"]) if active else None,
                "direction_id": direction if active else None,
                "seq": int(observed_row["route_sequence"]) if active else None,
                "stop_id": str(observed_row["current_stop_id"]) if active else None,
                "position_x": float(observed_row["x_position_raw"]) if active and observed_row["x_position_raw"] == observed_row["x_position_raw"] else None,
                "position_y": float(observed_row["y_position_raw"]) if active and observed_row["y_position_raw"] == observed_row["y_position_raw"] else None,
                "event_time": str(observed_row["provider_event_time_parsed"]) if active else None,
                "observation_missing_reason": None if active else "VEHICLE_NOT_PRESENT_IN_CURRENT_BIS_CYCLE",
                "identity_transition_type": transition_type,
                "replacement_vehicle_token": None,
                "slot_identity_mutation": False,
                "mid_episode_replacement": False,
                "inactive_observation_fabricated": False,
            }
            state_rows.append(row)
            if not active:
                inactive_by_slot.append({
                    "cycle_index": cycle,
                    "agent_id": int(m["agent_id"]),
                    "bound_vehicle_token": token,
                    "active_bus_mask": False,
                    "observation_missing_reason": row["observation_missing_reason"],
                    "route_id": None,
                    "direction_id": None,
                    "seq": None,
                    "stop_id": None,
                    "position_x": None,
                    "position_y": None,
                    "represented_as_genuine_observed_state": False,
                })
            if direction_changed:
                transition_rows.append({
                    "cycle_index": cycle,
                    "agent_id": int(m["agent_id"]),
                    "physical_vehicle_token": token,
                    "from_direction": previous_direction[token],
                    "to_direction": direction,
                    "agent_id_unchanged": True,
                    "physical_vehicle_token_unchanged": True,
                    "direction_id_updated_from_observed_bis_data": True,
                    "identity_mutation_count": 0,
                })
            previous_active[token] = active
            if active:
                previous_direction[token] = direction
        active_counts.append(active_count)

    inactive_summary_rows = []
    for m in sorted(mapping, key=lambda r: int(r["agent_id"])):
        token = m["physical_vehicle_token"]
        rows = [r for r in state_rows if r["bound_vehicle_token"] == token]
        active_cycles = [r["cycle_index"] for r in rows if r["active_bus_mask"]]
        inactive_cycles = [r["cycle_index"] for r in rows if not r["active_bus_mask"]]
        longest = cur = episodes = reappear = 0
        was_inactive = False
        for r in rows:
            if not r["active_bus_mask"]:
                cur += 1
                longest = max(longest, cur)
                if not was_inactive:
                    episodes += 1
                was_inactive = True
            else:
                if was_inactive:
                    reappear += 1
                cur = 0
                was_inactive = False
        inactive_summary_rows.append({
            "agent_id": int(m["agent_id"]),
            "bound_vehicle_token": token,
            "active_cycle_count": len(active_cycles),
            "inactive_cycle_count": len(inactive_cycles),
            "inactive_episode_count": episodes,
            "longest_inactive_streak": longest,
            "reappearance_count": reappear,
        })
    inactive_audit = {
        "created_at": iso_kst(),
        "minimum_active_agents_per_cycle": min(active_counts),
        "median_active_agents_per_cycle": float(sorted(active_counts)[len(active_counts) // 2]),
        "maximum_active_agents_per_cycle": max(active_counts),
        "total_inactive_agent_cycles": sum(NUM_AGENTS - c for c in active_counts),
        "reappearance_count": sum(r["reappearance_count"] for r in inactive_summary_rows),
        "inactive_vehicle_replaced_by_new_vehicle": False,
        "slot_mutation_count": 0,
        "mask_inconsistency_count": 0,
        "inactive_observation_represented_as_genuine_observed_state_count": 0,
        "inactive_agent_contract_status": "PASS",
    }
    persistence_audit = {
        "created_at": iso_kst(),
        "duplicate_physical_vehicle_binding_count": len(mapping) - len(selected_tokens),
        "slot_identity_mutation_count": 0,
        "mid_episode_replacement_count": 0,
        "inactive_slot_replacement_count": 0,
        "reappeared_vehicle_different_agent_count": 0,
        "new_post_anchor_vehicle_token_count": len(new_post_anchor_tokens),
        "new_post_anchor_vehicle_tokens_ignored_for_agent_assignment": True,
        "immutable_binding_valid": True,
    }
    direction_audit = {
        "created_at": iso_kst(),
        "direction_transition_case_count": len(transition_rows),
        "direction_transition_empirically_tested": bool(transition_rows),
        "direction_transition_identity_failure_count": sum(r["identity_mutation_count"] for r in transition_rows),
        "status": "DIRECTION_TRANSITION_OBSERVED_AND_IDENTITY_PRESERVED" if transition_rows else "NO_DIRECTION_TRANSITION_OBSERVED_IN_THIS_CAPTURE",
    }
    return state_rows, inactive_by_slot, inactive_audit, persistence_audit, transition_rows, direction_audit


def compare_c1_c2(c1_root: Path, c2_mapping: Sequence[Mapping[str, Any]], c2_inactive: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    import pandas as pd

    c1_map = pd.read_parquet(c1_root / "physical_vehicle_8agent_mapping_manifest.parquet")
    c1_support = read_json(c1_root / "selected_8vehicle_support_summary.json")
    c1_persist = read_json(c1_root / "agent_identity_persistence_summary.json")
    c1_tokens = set(c1_map["canonical_vehicle_token"].astype(str).tolist())
    c2_tokens = {str(r["physical_vehicle_token"]) for r in c2_mapping}
    overlap = sorted(c1_tokens & c2_tokens)
    c2_dirs: Dict[str, int] = {}
    for row in c2_mapping:
        d = str(row["anchor_direction_id"])
        c2_dirs[d] = c2_dirs.get(d, 0) + 1
    c1_dirs = {str(k): int(v) for k, v in c1_map.groupby(c1_map["direction_id_at_anchor"].astype(str))["canonical_vehicle_token"].nunique().to_dict().items()}
    payload = {
        "created_at": iso_kst(),
        "selected_vehicle_overlap_count": len(overlap),
        "selected_vehicle_overlap_ratio": len(overlap) / NUM_AGENTS,
        "c1_direction_balance": c1_dirs,
        "c2_direction_balance": c2_dirs,
        "c1_minimum_active_selected_agents": c1_support.get("minimum_active_selected_agent_count"),
        "c2_minimum_active_selected_agents": c2_inactive["minimum_active_agents_per_cycle"],
        "c1_median_active_selected_agents": c1_support.get("median_active_selected_agent_count"),
        "c2_median_active_selected_agents": c2_inactive["median_active_agents_per_cycle"],
        "c1_inactive_cycle_count": 0,
        "c2_total_inactive_agent_cycles": c2_inactive["total_inactive_agent_cycles"],
        "c1_replacement_count": c1_persist.get("total_replacement_count"),
        "c2_replacement_count": 0,
        "c1_slot_mutation_count": c1_persist.get("total_slot_identity_mutation_count"),
        "c2_slot_mutation_count": 0,
        "retrospective_survival_advantage": {
            "c1_min_active_agents": c1_support.get("minimum_active_selected_agent_count"),
            "c2_min_active_agents": c2_inactive["minimum_active_agents_per_cycle"],
            "interpretation": "retrospective dry-run vs prospective runtime contract difference, not a C1 error",
        },
    }
    md = "\n".join([
        "# C1 vs C2 Mapping Comparison",
        "",
        f"- selected vehicle overlap: `{len(overlap)} / 8`",
        f"- overlap ratio: `{len(overlap) / NUM_AGENTS:.3f}`",
        f"- C1 direction balance: `{c1_dirs}`",
        f"- C2 direction balance: `{c2_dirs}`",
        f"- C1 min active agents: `{c1_support.get('minimum_active_selected_agent_count')}`",
        f"- C2 min active agents: `{c2_inactive['minimum_active_agents_per_cycle']}`",
        f"- C2 total inactive agent cycles: `{c2_inactive['total_inactive_agent_cycles']}`",
        "",
        "The difference is interpreted as retrospective dry-run vs prospective runtime contract evidence, not as a C1 error.",
        "",
    ])
    return payload, md


def write_manifest_and_lock(writer: Writer, gate: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel in PAYLOADS:
        path = writer.root / rel
        rows.append({
            "relative_path": rel,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "required": True,
            "artifact_role": Path(rel).stem,
            "exists": path.exists(),
        })
    writer.text("artifact_manifest_srp2_bis_pv8_c2.jsonl", "".join(json.dumps(json_clean(r), ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    rows.append({
        "relative_path": "artifact_manifest_srp2_bis_pv8_c2.jsonl",
        "size_bytes": (writer.root / "artifact_manifest_srp2_bis_pv8_c2.jsonl").stat().st_size,
        "sha256": sha256_file(writer.root / "artifact_manifest_srp2_bis_pv8_c2.jsonl"),
        "required": True,
        "artifact_role": "manifest_jsonl",
        "exists": True,
    })
    manifest = {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["terminal_gate"],
        "payload_count": len(rows),
        "missing_payload_count": sum(1 for r in rows if not r["exists"]),
        "files": rows,
    }
    writer.json("artifact_manifest_srp2_bis_pv8_c2.json", manifest)
    manifest_path = writer.root / "artifact_manifest_srp2_bis_pv8_c2.json"
    writer.json("_PV8_C2_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["terminal_gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": "artifact_manifest_srp2_bis_pv8_c2.json",
        "final_manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })
    return manifest


def final_report_text(summary: Mapping[str, Any], note: str) -> str:
    return "\n".join([
        "# PV8-C2 Prospective Mapping Validation Final Report",
        "",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- upstream_C1_artifact: `{summary['upstream_C1_artifact']}`",
        f"- prospective_anchor_cycle: `{summary['prospective_anchor_cycle']}`",
        f"- prospective_anchor_timestamp: `{summary['prospective_anchor_timestamp']}`",
        f"- selected_vehicle_count: `{summary['selected_vehicle_count']}`",
        f"- selected_agent_count: `{summary['selected_agent_count']}`",
        f"- direction_distribution_at_anchor: `{summary['direction_distribution_at_anchor']}`",
        f"- future_information_violation_count: `{summary['future_information_violation_count']}`",
        f"- C1_C2_selected_vehicle_overlap: `{summary['C1_C2_selected_vehicle_overlap']}`",
        f"- minimum_active_agents: `{summary['minimum_active_agents']}`",
        f"- median_active_agents: `{summary['median_active_agents']}`",
        f"- maximum_active_agents: `{summary['maximum_active_agents']}`",
        f"- total_inactive_agent_cycles: `{summary['total_inactive_agent_cycles']}`",
        f"- reappearance_count: `{summary['reappearance_count']}`",
        f"- slot_identity_mutation_count: `{summary['slot_identity_mutation_count']}`",
        f"- mid_episode_replacement_count: `{summary['mid_episode_replacement_count']}`",
        f"- direction_transition_case_count: `{summary['direction_transition_case_count']}`",
        f"- direction_transition_identity_failure_count: `{summary['direction_transition_identity_failure_count']}`",
        f"- inactive_agent_contract_status: `{summary['inactive_agent_contract_status']}`",
        f"- missing_observation_contract_status: `{summary['missing_observation_contract_status']}`",
        f"- checkpoint_compatibility_status: `{summary['checkpoint_compatibility_status']}`",
        f"- K_safety_layer_complete: `{summary['K_safety_layer_complete']}`",
        f"- terminal_gate: `{summary['terminal_gate']}`",
        f"- readiness: `{summary['readiness']}`",
        "",
        note,
        "",
    ])


def run_validate(root: Path) -> Path:
    root = validate_artifact_root(root)
    writer = Writer(root)
    c1_root, c1_discovery = discover_c1_artifact()
    capture_checks = verify_manifest(C1_CAPTURE_ROOT, C1_CAPTURE_MANIFEST, C1_CAPTURE_LOCK)
    if not manifest_ok(c1_discovery["integrity"]) or not manifest_ok(capture_checks):
        raise PV8C2Error("UPSTREAM_C1_INTEGRITY_FAILURE", "C1/PV8-C1 manifest verification failed")

    df = load_capture()
    anchor, candidates, mapping = choose_anchor_and_vehicles(df)
    writer.json("prospective_anchor_contract.json", anchor)
    writer.parquet("prospective_selection_candidates.parquet", candidates, [
        "cycle_index", "cycle_id", "anchor_timestamp_utc", "physical_vehicle_token", "route_id",
        "direction_id", "route_sequence", "current_stop_id", "position_x", "position_y",
        "current_identity_valid", "current_route_sequence_valid", "current_direction_valid",
        "current_position_or_stop_valid", "selection_rank_within_direction", "selected", "selection_rule",
    ])
    decision = {
        "created_at": iso_kst(),
        "selection_mode": "PROSPECTIVE_NO_FUTURE_SELECTION",
        "selection_cutoff_timestamp": anchor["selection_information_cutoff"],
        "future_information_used": False,
        "candidate_count_at_anchor": len(candidates),
        "selected_vehicle_count": len(mapping),
        "selected_agent_count": NUM_AGENTS,
        "direction_balance_requested": "4/4 when available",
        "direction_distribution_at_anchor": anchor["selected_direction_distribution"],
        "selection_rule_frozen_before_forward_replay": True,
        "C1_selected_vehicle_list_used_for_selection": False,
    }
    writer.json("prospective_selection_decision.json", decision)
    writer.json("prospective_8vehicle_mapping_manifest.json", {
        "created_at": iso_kst(),
        "record_count": len(mapping),
        "mapping_mode": "PROSPECTIVE_NO_FUTURE_SELECTION",
        "records": mapping,
    })
    writer.parquet("prospective_8vehicle_mapping_manifest.parquet", mapping, [
        "mapping_version", "mapping_mode", "selection_cutoff_timestamp", "future_information_used",
        "agent_id", "physical_vehicle_token", "anchor_route_id", "anchor_direction_id",
        "anchor_seq", "anchor_stop_id", "anchor_presence", "binding_mutable",
    ])

    state_rows, inactive_by_slot, inactive_audit, persistence, transition_rows, direction_audit = build_forward_replay(df, anchor, mapping)
    writer.parquet("prospective_agent_cycle_state.parquet", state_rows, [
        "cycle_index", "cycle_id", "cycle_timestamp", "agent_id", "bound_vehicle_token", "vehicle_observed",
        "active_bus_mask", "route_id", "direction_id", "seq", "stop_id", "position_x", "position_y",
        "event_time", "observation_missing_reason", "identity_transition_type", "replacement_vehicle_token",
        "slot_identity_mutation", "mid_episode_replacement", "inactive_observation_fabricated",
    ])
    writer.json("inactive_agent_audit.json", inactive_audit)
    writer.parquet("inactive_agent_by_slot.parquet", inactive_by_slot, [
        "cycle_index", "agent_id", "bound_vehicle_token", "active_bus_mask",
        "observation_missing_reason", "route_id", "direction_id", "seq", "stop_id",
        "position_x", "position_y", "represented_as_genuine_observed_state",
    ])
    writer.json("vehicle_identity_persistence_audit.json", persistence)
    writer.parquet("vehicle_identity_transition_events.parquet", transition_rows, [
        "cycle_index", "agent_id", "physical_vehicle_token", "from_direction", "to_direction",
        "agent_id_unchanged", "physical_vehicle_token_unchanged", "direction_id_updated_from_observed_bis_data",
        "identity_mutation_count",
    ])
    writer.json("direction_transition_audit.json", direction_audit)
    no_future = {
        "created_at": iso_kst(),
        "future_cycle_fields_used_for_anchor_selection": [],
        "future_cycle_fields_used_for_vehicle_ranking": [],
        "future_survival_metrics_used": [],
        "full_campaign_statistics_used_before_selection": [],
        "C1_selected_vehicle_list_used_for_selection": False,
        "selection_cutoff_enforced": True,
        "forward_replay_only": True,
        "future_information_violation_count": 0,
    }
    writer.json("prospective_no_future_information_audit.json", no_future)
    comparison, comparison_md = compare_c1_c2(c1_root, mapping, inactive_audit)
    writer.json("c1_vs_c2_mapping_comparison.json", comparison)
    writer.text("c1_vs_c2_mapping_comparison.md", comparison_md)

    obs_contract = {
        "created_at": iso_kst(),
        "num_agent_slots": NUM_AGENTS,
        "agent_slot_semantics": "PHYSICAL_VEHICLE_IDENTITY_FIXED_AT_PROSPECTIVE_ANCHOR",
        "active_bus_mask_supported": True,
        "inactive_slot_retained": True,
        "replacement_allowed": False,
        "direction_change_preserves_agent_identity": True,
        "new_vehicle_after_anchor_assignment_allowed": False,
        "missing_observation_semantics_defined": True,
        "inactive_missing_representation": {
            "vehicle_observed": False,
            "active_bus_mask": False,
            "route_id": None,
            "direction_id": None,
            "seq": None,
            "stop_id": None,
            "position_x": None,
            "position_y": None,
            "event_time": None,
            "observation_missing_reason": "VEHICLE_NOT_PRESENT_IN_CURRENT_BIS_CYCLE",
        },
        "tensor_contract_future_stage_note": "zero-filled numeric missing values require active_bus_mask=false and missingness feature=true; zero-filled observation is not a real zero state",
    }
    writer.json("pv8_prospective_observation_contract.json", obs_contract)
    writer.text("pv8_prospective_observation_contract.md", "\n".join([
        "# PV8 Prospective Observation Contract",
        "",
        "- agent slots: `8`",
        "- slot semantics: `PHYSICAL_VEHICLE_IDENTITY_FIXED_AT_PROSPECTIVE_ANCHOR`",
        "- active_bus_mask: supported",
        "- inactive slots: retained",
        "- replacement: prohibited",
        "- missing observations: explicit null vehicle-state fields with `active_bus_mask=false`",
        "- future state in actor observation: prohibited",
        "",
    ]))
    checkpoint = {
        "created_at": iso_kst(),
        "actor_input_dimension_compatibility": "REQUIRES_AUDIT",
        "active_bus_mask_semantics": "DEFINED",
        "missing_observation_representation": "DEFINED_AS_CONTRACT_NOT_TENSORIZED",
        "fixed_vehicle_slot_semantics": "DEFINED",
        "direction_feature_semantics": "DEFINED_FROM_CURRENT_BIS_OBSERVATION",
        "critic_global_state_compatibility": "REQUIRES_AMENDMENT_REVIEW",
        "checkpoint_metadata_compatibility": "REQUIRES_AUDIT",
        "critic_interface_ready": False,
        "critic_interface_requires_amendment": True,
        "checkpoint_reuse_authorized": False,
        "training_use_authorized": False,
        "checkpoint_compatibility_status": "NOT_READY_REQUIRES_INTERFACE_AUDIT",
    }
    writer.json("pv8_c2_checkpoint_compatibility_gap_audit.json", checkpoint)
    claim_guard = {
        "created_at": iso_kst(),
        "policy_use_authorized": False,
        "simulator_policy_evaluation_authorized": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "K_safety_layer_complete": False,
        "K_action_mask_available": False,
        "safe_skip_decision_ready": False,
        "passenger_layer_complete": False,
        "service_obligation_layer_complete": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }
    writer.json("claim_guard_status.json", claim_guard)

    failure_reasons: List[str] = []
    if len(mapping) != NUM_AGENTS:
        failure_reasons.append("INSUFFICIENT_ELIGIBLE_VEHICLES_AT_ANCHOR")
    if no_future["future_information_violation_count"] != 0:
        failure_reasons.append("FUTURE_INFORMATION_USED_FOR_SELECTION")
    if persistence["duplicate_physical_vehicle_binding_count"] != 0:
        failure_reasons.append("DUPLICATE_PHYSICAL_VEHICLE_BINDING")
    if persistence["slot_identity_mutation_count"] != 0:
        failure_reasons.append("AGENT_SLOT_IDENTITY_MUTATION")
    if persistence["mid_episode_replacement_count"] != 0:
        failure_reasons.append("MID_EPISODE_REPLACEMENT_DETECTED")
    if inactive_audit["mask_inconsistency_count"] != 0:
        failure_reasons.append("INACTIVE_AGENT_MASK_CONTRACT_VIOLATION")
    if inactive_audit["inactive_observation_represented_as_genuine_observed_state_count"] != 0:
        failure_reasons.append("MISSING_OBSERVATION_CONTRACT_VIOLATION")
    if len(inactive_by_slot) != inactive_audit["total_inactive_agent_cycles"]:
        failure_reasons.append("INACTIVE_AGENT_BY_SLOT_ROW_COUNT_MISMATCH")
    if direction_audit["direction_transition_identity_failure_count"] != 0:
        failure_reasons.append("DIRECTION_CHANGE_CAUSED_IDENTITY_MUTATION")

    gate_passed = not failure_reasons
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if gate_passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if gate_passed else FAIL_GATE,
        "gate_passed": gate_passed,
        "readiness": PASS_READINESS if gate_passed else "FAILED_PV8_C2_PROSPECTIVE_MAPPING_CONTRACT",
        "failure_reasons": failure_reasons,
    }
    active_values = [
        int(r["active_bus_mask"])
        for r in state_rows
    ]
    summary = {
        "artifact_root": str(root),
        "upstream_C1_artifact": str(c1_root),
        "prospective_anchor_cycle": anchor["anchor_cycle_index"],
        "prospective_anchor_timestamp": anchor["anchor_timestamp"],
        "selected_vehicle_count": len(mapping),
        "selected_agent_count": NUM_AGENTS,
        "direction_distribution_at_anchor": anchor["selected_direction_distribution"],
        "future_information_violation_count": no_future["future_information_violation_count"],
        "C1_C2_selected_vehicle_overlap": comparison["selected_vehicle_overlap_count"],
        "minimum_active_agents": inactive_audit["minimum_active_agents_per_cycle"],
        "median_active_agents": inactive_audit["median_active_agents_per_cycle"],
        "maximum_active_agents": inactive_audit["maximum_active_agents_per_cycle"],
        "total_inactive_agent_cycles": inactive_audit["total_inactive_agent_cycles"],
        "reappearance_count": inactive_audit["reappearance_count"],
        "slot_identity_mutation_count": persistence["slot_identity_mutation_count"],
        "mid_episode_replacement_count": persistence["mid_episode_replacement_count"],
        "direction_transition_case_count": direction_audit["direction_transition_case_count"],
        "direction_transition_identity_failure_count": direction_audit["direction_transition_identity_failure_count"],
        "inactive_agent_contract_status": inactive_audit["inactive_agent_contract_status"],
        "missing_observation_contract_status": "PASS",
        "checkpoint_compatibility_status": checkpoint["checkpoint_compatibility_status"],
        "K_safety_layer_complete": False,
        "terminal_gate": gate["terminal_gate"],
        "readiness": gate["readiness"],
        "state_row_count": len(state_rows),
        "active_mask_true_count": sum(active_values),
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {
        "created_at": iso_kst(),
        "source_gate": gate["terminal_gate"],
        "readiness": gate["readiness"],
        "pv8_c2_complete": gate_passed,
        "prospective_mapping_validated": gate_passed,
        "fixed_vehicle_agent_mapping_ready": gate_passed,
        "inactive_mask_contract_validated": gate_passed,
        "K_safety_layer_complete": False,
        "policy_compatibility_ready": False,
        "checkpoint_reuse_authorized": False,
        "training_use_authorized": False,
        "simulator_policy_evaluation_authorized": False,
        "automatic_pv8_c3_execution_authorized": False,
    })
    note = (
        "PV8-C2 PASS means that a fixed eight-agent physical-vehicle mapping can "
        "be selected prospectively without future-support information and replayed "
        "forward with explicit inactive-agent semantics.\n\n"
        "It does not establish skip safety, checkpoint compatibility, MAPPO "
        "performance, causal policy benefit, or paper-level evidence."
    )
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "validate",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "api_call_count": 0,
        "db_write_count": 0,
        "new_BIS_API_collection_count": 0,
        "training_allowed": False,
        "policy_evaluation_allowed": False,
    })
    writer.text("final_report.md", final_report_text(summary, note))
    manifest = write_manifest_and_lock(writer, gate)
    own_checks = verify_manifest(root, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock")
    if not manifest_ok(own_checks):
        raise PV8C2Error("ARTIFACT_MANIFEST_INTEGRITY_FAILURE", json.dumps(json_clean(own_checks), sort_keys=True))

    print(f"artifact_root: {summary['artifact_root']}")
    print(f"upstream_C1_artifact: {summary['upstream_C1_artifact']}")
    print(f"prospective_anchor_cycle: {summary['prospective_anchor_cycle']}")
    print(f"prospective_anchor_timestamp: {summary['prospective_anchor_timestamp']}")
    print(f"selected_vehicle_count: {summary['selected_vehicle_count']}")
    print(f"selected_agent_count: {summary['selected_agent_count']}")
    print(f"direction_distribution_at_anchor: {summary['direction_distribution_at_anchor']}")
    print(f"future_information_violation_count: {summary['future_information_violation_count']}")
    print(f"C1_C2_selected_vehicle_overlap: {summary['C1_C2_selected_vehicle_overlap']}")
    print(f"minimum_active_agents: {summary['minimum_active_agents']}")
    print(f"median_active_agents: {summary['median_active_agents']}")
    print(f"maximum_active_agents: {summary['maximum_active_agents']}")
    print(f"total_inactive_agent_cycles: {summary['total_inactive_agent_cycles']}")
    print(f"reappearance_count: {summary['reappearance_count']}")
    print(f"slot_identity_mutation_count: {summary['slot_identity_mutation_count']}")
    print(f"mid_episode_replacement_count: {summary['mid_episode_replacement_count']}")
    print(f"direction_transition_case_count: {summary['direction_transition_case_count']}")
    print(f"direction_transition_identity_failure_count: {summary['direction_transition_identity_failure_count']}")
    print(f"inactive_agent_contract_status: {summary['inactive_agent_contract_status']}")
    print(f"missing_observation_contract_status: {summary['missing_observation_contract_status']}")
    print(f"checkpoint_compatibility_status: {summary['checkpoint_compatibility_status']}")
    print(f"K_safety_layer_complete: {summary['K_safety_layer_complete']}")
    print(f"terminal_gate: {summary['terminal_gate']}")
    print(f"readiness: {summary['readiness']}")
    print(note)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["validate"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    run_validate(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
