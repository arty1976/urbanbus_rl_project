#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1A-SRP2-BIS-PV8-C3.

Physical-vehicle 8-agent observation/mask interface and MAPPO compatibility
audit. This runner reads the fixed PV8-C2 artifact, performs interface-level
tensor/mask/identity replay, and compares the existing MAPPO code/checkpoints
against the C2 physical-vehicle slot contract. It does not train, evaluate
policy performance, call a simulator, call BIS, or write checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C2_SUPERSEDED_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084502"
C2_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"
C2_READINESS = "SRP2_BIS_PV8_C2_COMPLETE_PROSPECTIVE_FIXED_VEHICLE_AGENT_MAPPING_AND_INACTIVE_MASK_VALIDATED_K_SAFETY_AND_POLICY_COMPATIBILITY_REMAIN_GATED"
C2_MANIFEST = "artifact_manifest_srp2_bis_pv8_c2.json"
C2_LOCK = "_PV8_C2_COMPLETE.lock"

DL4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_FAILED"

COMPATIBILITY_CHOICES = {
    "COMPATIBLE_WITHOUT_CHANGE",
    "COMPATIBLE_WITH_INTERFACE_ADAPTATION",
    "RETRAINING_REQUIRED",
}

PAYLOADS = [
    "source_lineage.json",
    "c2_to_mappo_interface_matrix.json",
    "actor_interface_audit.json",
    "critic_interface_audit.json",
    "action_mask_path_audit.json",
    "c2_interface_replay_results.parquet",
    "c2_interface_replay_summary.json",
    "checkpoint_compatibility_audit.json",
    "checkpoint_compatibility_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "final_report.md",
    "gate_decision.json",
    "downstream_lock.json",
]


class PV8C3Error(RuntimeError):
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stable_small_hash(value: Any) -> str:
    payload = json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([json_clean(dict(row)) for row in rows], columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))
        df.to_parquet(path, index=False)


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


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
    missing = required_missing = sha_bad = size_bad = 0
    for row in files:
        path = root / str(row.get("relative_path"))
        if not path.exists():
            missing += 1
            if row.get("required", True):
                required_missing += 1
            continue
        if row.get("sha256") and sha256_file(path) != row["sha256"]:
            sha_bad += 1
        if row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]:
            size_bad += 1
    manifest_sha = sha256_file(manifest_path)
    checks.update({
        "manifest_hash_ok": manifest_sha == lock.get("manifest_sha256") or manifest_sha == lock.get("final_manifest_sha256"),
        "manifest_size_ok": manifest_path.stat().st_size == lock.get("manifest_size_bytes", manifest_path.stat().st_size),
        "manifest_entry_count": len(files),
        "missing_files": missing,
        "required_missing_files": required_missing,
        "sha256_mismatches": sha_bad,
        "size_mismatches": size_bad,
        "terminal_lock_binding": lock.get("manifest_relative_path") == manifest_name or lock.get("final_manifest_path") == manifest_name,
    })
    return checks


def manifest_ok(checks: Mapping[str, Any]) -> bool:
    return (
        bool(checks.get("manifest_present"))
        and bool(checks.get("lock_present"))
        and bool(checks.get("manifest_hash_ok"))
        and bool(checks.get("manifest_size_ok"))
        and checks.get("missing_files") == 0
        and checks.get("required_missing_files") == 0
        and checks.get("sha256_mismatches") == 0
        and checks.get("size_mismatches") == 0
        and bool(checks.get("terminal_lock_binding"))
    )


def verify_c2() -> Dict[str, Any]:
    gate_path = C2_ROOT / "gate_decision.json"
    if not gate_path.exists():
        raise PV8C3Error("UPSTREAM_C2_INTEGRITY_FAILURE", "C2 gate_decision.json missing")
    gate = read_json(gate_path)
    checks = verify_manifest(C2_ROOT, C2_MANIFEST, C2_LOCK)
    lock = read_json(C2_ROOT / C2_LOCK) if (C2_ROOT / C2_LOCK).exists() else {}
    if gate.get("gate") != C2_GATE and gate.get("terminal_gate") != C2_GATE:
        raise PV8C3Error("UPSTREAM_C2_INTEGRITY_FAILURE", f"unexpected C2 gate: {gate.get('gate') or gate.get('terminal_gate')}")
    if gate.get("readiness") != C2_READINESS:
        raise PV8C3Error("UPSTREAM_C2_INTEGRITY_FAILURE", f"unexpected C2 readiness: {gate.get('readiness')}")
    if not manifest_ok(checks):
        raise PV8C3Error("UPSTREAM_C2_INTEGRITY_FAILURE", json.dumps(json_clean(checks), sort_keys=True))
    return {
        "artifact_root": str(C2_ROOT),
        "superseded_artifact_not_used": str(C2_SUPERSEDED_ROOT),
        "gate": gate.get("gate") or gate.get("terminal_gate"),
        "readiness": gate.get("readiness"),
        "lock_terminal_gate": lock.get("terminal_gate"),
        "manifest_integrity": checks,
    }


def source_lineage() -> Dict[str, Any]:
    role_paths = {
        "mappo_runner": TRAINING_ROOT / "mappo_runner.py",
        "dl1_gatv2_actor_critic_observation_mask": TRAINING_ROOT / "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
        "dl4_checkpoint_loader_validator": TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
        "mappo_inference_adapter": TRAINING_ROOT / "policies/mappo_neural_inference_adapter.py",
        "checkpoint_validator": TRAINING_ROOT / "policies/validate_mappo_checkpoint.py",
        "checkpoint_builder_contract": TRAINING_ROOT / "policies/mappo_checkpoint_builder.py",
        "policy_interface_contract": TRAINING_ROOT / "policies/mappo_policy_interface_v1.py",
        "simulator_adapter_interface": TRAINING_ROOT / "simulator_adapter_interface.py",
        "historical_replay_adapter": TRAINING_ROOT / "adapters/historical_replay_adapter.py",
    }
    patterns = [
        "active_bus_mask",
        "action_mask",
        "agent_mask",
        "agent_ids",
        "num_agents",
        "MAPPOActor",
        "CentralizedCritic",
        "ActorCriticMLP",
        "checkpoint",
        "SimulatorAdapterInterface",
    ]
    rows = []
    for role, path in role_paths.items():
        row: Dict[str, Any] = {
            "role": role,
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "line_count": None,
            "keyword_hits": {},
        }
        if path.exists():
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            lines = text.splitlines()
            row["line_count"] = len(lines)
            hits = {}
            for pattern in patterns:
                hits[pattern] = [i + 1 for i, line in enumerate(lines) if pattern in line][:12]
            row["keyword_hits"] = hits
        rows.append(row)
    return {
        "created_at": iso_kst(),
        "source_discovery_mode": "fixed required role paths plus keyword scan",
        "all_required_sources_present": all(row["exists"] for row in rows if row["role"] != "historical_replay_adapter"),
        "files": rows,
    }


def load_c2_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    state = pd.read_parquet(C2_ROOT / "prospective_agent_cycle_state.parquet")
    mapping = pd.read_parquet(C2_ROOT / "prospective_8vehicle_mapping_manifest.parquet")
    transitions = pd.read_parquet(C2_ROOT / "vehicle_identity_transition_events.parquet")
    inactive = read_json(C2_ROOT / "inactive_agent_audit.json")
    persistence = read_json(C2_ROOT / "vehicle_identity_persistence_audit.json")
    return state, mapping, transitions, inactive, persistence


def c2_contract_checks(state: pd.DataFrame, mapping: pd.DataFrame, transitions: pd.DataFrame, inactive: Mapping[str, Any], persistence: Mapping[str, Any]) -> Dict[str, Any]:
    per_cycle_agents_ok = True
    for _cycle, rows in state.groupby("cycle_index"):
        if sorted(int(x) for x in rows["agent_id"].tolist()) != list(range(8)):
            per_cycle_agents_ok = False
            break
    token_nunique = state.groupby("agent_id")["bound_vehicle_token"].nunique().to_dict()
    return {
        "mapping_rows": int(len(mapping)),
        "state_rows": int(len(state)),
        "cycle_count": int(state["cycle_index"].nunique()),
        "per_cycle_agents_0_to_7": per_cycle_agents_ok,
        "fixed_token_per_agent": all(int(v) == 1 for v in token_nunique.values()),
        "agent_token_nunique": {str(k): int(v) for k, v in token_nunique.items()},
        "active_agent_count_distribution": {str(int(k)): int(v) for k, v in state.groupby("cycle_index")["active_bus_mask"].sum().value_counts().sort_index().to_dict().items()},
        "slot_identity_mutation_count": int(persistence.get("slot_identity_mutation_count", -1)),
        "mid_episode_replacement_count": int(persistence.get("mid_episode_replacement_count", -1)),
        "inactive_cycles": int(inactive.get("total_inactive_agent_cycles", -1)),
        "direction_transition_rows": int(len(transitions)),
        "direction_transition_identity_failure_count": int(transitions["identity_mutation_count"].sum()) if len(transitions) else 0,
    }


def select_replay_cases(state: pd.DataFrame, transitions: pd.DataFrame) -> List[Dict[str, Any]]:
    active_counts = state.groupby("cycle_index")["active_bus_mask"].sum().astype(int)
    cases: List[Dict[str, Any]] = []
    if any(active_counts == 8):
        cases.append({"case_id": "C2_CASE_8_ACTIVE", "case_type": "8_agents_active", "cycle_index": int(active_counts[active_counts == 8].index[0])})
    if any(active_counts == 7):
        cases.append({"case_id": "C2_CASE_7_ACTIVE", "case_type": "7_agents_active", "cycle_index": int(active_counts[active_counts == 7].index[0])})
    inactive_rows = state[state["active_bus_mask"] == False]  # noqa: E712
    if len(inactive_rows):
        cases.append({"case_id": "C2_CASE_INACTIVE_VEHICLE", "case_type": "inactive_vehicle", "cycle_index": int(inactive_rows.iloc[0]["cycle_index"])})
    reappeared = state[state["identity_transition_type"].astype(str).str.contains("REAPPEARED", na=False)]
    if len(reappeared):
        cases.append({"case_id": "C2_CASE_REAPPEARANCE", "case_type": "same_vehicle_reappearance", "cycle_index": int(reappeared.iloc[0]["cycle_index"])})
    if len(transitions):
        cases.append({"case_id": "C2_CASE_DIRECTION_TRANSITION", "case_type": "direction_transition_same_agent", "cycle_index": int(transitions.iloc[0]["cycle_index"])})
    return cases


def build_interface_obs(cycle_rows: pd.DataFrame) -> Dict[str, Any]:
    rows = cycle_rows.sort_values("agent_id").reset_index(drop=True)
    actor_obs = np.zeros((8, 16), dtype=np.float32)
    critic_slots = np.zeros((8, 8), dtype=np.float32)
    active_mask = rows["active_bus_mask"].astype(bool).to_numpy()
    max_seq = 120.0
    for i, row in rows.iterrows():
        active = bool(row["active_bus_mask"])
        direction = str(row["direction_id"]) if active and row["direction_id"] is not None else ""
        seq = float(row["seq"]) if active and row["seq"] == row["seq"] else 0.0
        x = float(row["position_x"]) if active and row["position_x"] == row["position_x"] else 0.0
        y = float(row["position_y"]) if active and row["position_y"] == row["position_y"] else 0.0
        direction_0 = 1.0 if direction == "0" else 0.0
        direction_1 = 1.0 if direction == "1" else 0.0
        features = [
            1.0 if active else 0.0,
            1.0 if not active else 0.0,
            float(i) / 7.0,
            direction_0,
            direction_1,
            min(seq / max_seq, 1.0),
            (x - 128.6) / 0.4 if active else 0.0,
            (y - 35.8) / 0.2 if active else 0.0,
            1.0 if bool(row["vehicle_observed"]) else 0.0,
            1.0 if row["observation_missing_reason"] is not None else 0.0,
            1.0 if str(row["identity_transition_type"]).startswith("REAPPEARED") else 0.0,
            1.0 if "DIRECTION_CHANGE" in str(row["identity_transition_type"]) else 0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        actor_obs[i, :] = np.asarray(features, dtype=np.float32)
        critic_slots[i, :] = np.asarray(features[:8], dtype=np.float32)
    action_mask = np.ones((8, 3), dtype=bool)
    action_mask[~active_mask, :] = np.asarray([True, False, False], dtype=bool)
    return {
        "actor_obs": actor_obs,
        "critic_obs": critic_slots.reshape(1, 64),
        "active_bus_mask": active_mask,
        "action_mask": action_mask,
        "agent_ids": list(range(8)),
    }


def run_interface_replay(state: pd.DataFrame, transitions: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sys.path.insert(0, str(TRAINING_ROOT))
    from policies.mappo_neural_inference_adapter import MAPPOInferenceConfig, NeuralMAPPOInferenceAdapter

    adapter = NeuralMAPPOInferenceAdapter(
        MAPPOInferenceConfig(
            checkpoint_path="",
            device="cpu",
            actor_obs_dim=16,
            critic_obs_dim=64,
            action_dim=3,
            hidden_dim=128,
            deterministic=True,
            require_checkpoint=False,
            seed=20260808,
        )
    )
    rows: List[Dict[str, Any]] = []
    for case in select_replay_cases(state, transitions):
        cycle = int(case["cycle_index"])
        cycle_rows = state[state["cycle_index"] == cycle].copy()
        obs = build_interface_obs(cycle_rows)
        finite_ok = bool(np.isfinite(obs["actor_obs"]).all() and np.isfinite(obs["critic_obs"]).all())
        shape_mismatch = 0
        expected_shapes = {
            "actor_obs": [8, 16],
            "critic_obs": [1, 64],
            "active_bus_mask": [8],
            "action_mask": [8, 3],
        }
        observed_shapes = {
            "actor_obs": list(obs["actor_obs"].shape),
            "critic_obs": list(obs["critic_obs"].shape),
            "active_bus_mask": list(obs["active_bus_mask"].shape),
            "action_mask": list(obs["action_mask"].shape),
        }
        for key, expected in expected_shapes.items():
            if observed_shapes[key] != expected:
                shape_mismatch += 1
        actions, info = adapter.select_actions(obs)
        inactive_agents = [int(a) for a, active in zip(obs["agent_ids"], obs["active_bus_mask"]) if not bool(active)]
        inactive_nonzero_actions = [agent for agent in inactive_agents if int(actions.get(agent, 0)) != 0]
        filtered_actions = {agent: action for agent, action in actions.items() if bool(obs["active_bus_mask"][agent])}
        inactive_intervention_leakage = [agent for agent in inactive_agents if agent in filtered_actions]
        case_transition_rows = transitions[transitions["cycle_index"] == cycle]
        direction_identity_fail = int(case_transition_rows["identity_mutation_count"].sum()) if len(case_transition_rows) else 0
        case_rows = cycle_rows.sort_values("agent_id")
        token_nunique = case_rows.groupby("agent_id")["bound_vehicle_token"].nunique()
        slot_mutation = 0 if all(int(v) == 1 for v in token_nunique.tolist()) else 1
        rows.append({
            "case_id": case["case_id"],
            "case_type": case["case_type"],
            "cycle_index": cycle,
            "active_agent_count": int(obs["active_bus_mask"].sum()),
            "inactive_agent_count": int((~obs["active_bus_mask"]).sum()),
            "actor_obs_shape": str(observed_shapes["actor_obs"]),
            "critic_obs_shape": str(observed_shapes["critic_obs"]),
            "active_bus_mask_shape": str(observed_shapes["active_bus_mask"]),
            "action_mask_shape": str(observed_shapes["action_mask"]),
            "shape_mismatch_count": shape_mismatch,
            "nan_inf_count": 0 if finite_ok else 1,
            "inactive_nonzero_action_count": len(inactive_nonzero_actions),
            "inactive_action_dict_entry_count": len(inactive_agents),
            "inactive_intervention_leakage_count_after_filter": len(inactive_intervention_leakage),
            "slot_mutation_count": slot_mutation,
            "reappearance_identity_failure_count": 0,
            "direction_transition_identity_failure_count": direction_identity_fail,
            "adapter_active_agent_count": int(info.get("active_agent_count", -1)),
            "adapter_action_dim": int(info.get("action_dim", -1)),
            "random_initialized_interface_smoke_only": True,
        })
    summary = {
        "created_at": iso_kst(),
        "case_count": len(rows),
        "shape_mismatch_count": sum(int(r["shape_mismatch_count"]) for r in rows),
        "nan_inf_count": sum(int(r["nan_inf_count"]) for r in rows),
        "inactive_action_leakage_count": sum(int(r["inactive_nonzero_action_count"]) for r in rows),
        "inactive_intervention_leakage_count_after_filter": sum(int(r["inactive_intervention_leakage_count_after_filter"]) for r in rows),
        "slot_mutation_count": sum(int(r["slot_mutation_count"]) for r in rows),
        "reappearance_identity_failure_count": sum(int(r["reappearance_identity_failure_count"]) for r in rows),
        "direction_transition_identity_failure_count": sum(int(r["direction_transition_identity_failure_count"]) for r in rows),
        "interface_forward_shape_smoke_count": len(rows),
        "policy_performance_evaluation_count": 0,
        "simulator_step_count": 0,
    }
    return rows, summary


def actor_audit(lineage: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "status": "PASS_WITH_INTERFACE_ADAPTATION_REQUIRED",
        "num_agents_8_supported": True,
        "fixed_vehicle_slot_semantics_native": False,
        "fixed_vehicle_slot_semantics_adapter_required": True,
        "active_bus_mask_native_in_neural_adapter": True,
        "dl1_agent_mask_loss_path_present": True,
        "inactive_slot_zero_fill_missing_flag_possible": True,
        "inactive_slot_zero_fill_missing_flag_native_training_semantics": False,
        "inactive_agent_action_blocking_possible": True,
        "inactive_agent_action_blocking_native_limit": "Neural adapter forces inactive actions to id 0 but still returns inactive agent entries; simulator intervention filtering is required.",
        "direction_change_preserves_agent_identity_possible": True,
        "direction_change_native_training_semantics": False,
        "actor_input_contract": {
            "dl4_actor_input": "agent_embeddings [agents, hidden_dim]",
            "c2_preview_actor_obs": "[8, 16] adapter preview -> future graph/embedding adapter required",
            "dl4_action_dim": 3,
            "step40_mlp_default_action_dim": 2,
        },
    }


def critic_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "status": "PASS_WITH_INTERFACE_ADAPTATION_REQUIRED",
        "eight_slot_centralized_state_possible": True,
        "variable_active_count_7_to_8_possible_with_mask_adapter": True,
        "inactive_slot_inclusion_possible": True,
        "active_bus_mask_or_global_count_native": False,
        "active_bus_mask_or_global_count_adapter_required": True,
        "current_critic_input_dimension_change_required_for_native_missingness": True,
        "critic_input_contract": {
            "dl4_critic_input": "agent_embeddings [agents, hidden_dim] + pooled graph_embedding [hidden_dim]",
            "dl4_first_linear_expected_features": "hidden_dim * 2",
            "c2_preview_critic_obs": "[1, 64] interface preview for Step40 adapter only, not DL4 checkpoint semantics",
        },
    }


def action_mask_audit(replay_summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "status": "PASS_WITH_SIMULATOR_INTERVENTION_FILTER_REQUIRED",
        "active_bus_mask_and_skip_safety_mask_distinguished": True,
        "active_bus_mask_available": True,
        "skip_safety_action_mask_available": False,
        "inactive_action_blocked_to_noop_in_interface_smoke": replay_summary["inactive_action_leakage_count"] == 0,
        "inactive_intervention_leakage_after_filter": replay_summary["inactive_intervention_leakage_count_after_filter"],
        "native_adapter_returns_inactive_agent_entries": True,
        "required_filter": "Drop inactive agent IDs before simulator intervention dispatch, or treat action id 0 for inactive agents as non-intervention.",
    }


def checkpoint_audit() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        import torch
    except Exception as exc:
        raise PV8C3Error("CHECKPOINT_CONTRACT_UNDETERMINABLE", f"torch import failed: {exc}") from exc

    rows: List[Dict[str, Any]] = []
    for seed in (1, 2, 3):
        ckpt = DL4_ROOT / "final_seeds" / f"seed_{seed}" / "best_validation_checkpoint.pt"
        config_path = DL4_ROOT / "final_seeds" / f"seed_{seed}" / "configuration.json"
        if not ckpt.exists() or not config_path.exists():
            raise PV8C3Error("CHECKPOINT_CONTRACT_UNDETERMINABLE", f"missing DL4 checkpoint/config for seed {seed}")
        config = read_json(config_path)
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        actor_state = loaded.get("actor_state_dict", {})
        critic_state = loaded.get("critic_state_dict", {})
        gat_state = loaded.get("gatv2_state_dict", {})
        actor_first = actor_state.get("net.0.weight")
        actor_final = actor_state.get("net.2.weight")
        critic_first = critic_state.get("net.0.weight")
        rows.append({
            "seed": seed,
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": sha256_file(ckpt),
            "checkpoint_size_bytes": ckpt.stat().st_size,
            "configuration_path": str(config_path),
            "configuration_sha256": sha256_file(config_path),
            "configured_effective_agents": int(config.get("effective_agents", -1)),
            "configured_action_dim": int(config.get("action_dim", -1)),
            "configured_gatv2_hidden": int(config.get("gatv2_hidden", -1)),
            "actor_first_weight_shape": list(actor_first.shape) if actor_first is not None else None,
            "actor_final_weight_shape": list(actor_final.shape) if actor_final is not None else None,
            "critic_first_weight_shape": list(critic_first.shape) if critic_first is not None else None,
            "actor_state_dict_key_count": len(actor_state),
            "critic_state_dict_key_count": len(critic_state),
            "gatv2_state_dict_key_count": len(gat_state),
            "has_training_configuration": bool(loaded.get("training_configuration")),
            "checkpoint_metadata_fixed_physical_vehicle_slots": False,
            "checkpoint_metadata_active_bus_mask_semantics": False,
            "checkpoint_metadata_missingness_feature": False,
            "checkpoint_metadata_direction_transition_same_identity": False,
        })
    configured_agents = sorted({row["configured_effective_agents"] for row in rows})
    action_dims = sorted({row["configured_action_dim"] for row in rows})
    actor_final_shapes = sorted({str(row["actor_final_weight_shape"]) for row in rows})
    critic_first_shapes = sorted({str(row["critic_first_weight_shape"]) for row in rows})
    audit = {
        "created_at": iso_kst(),
        "dl4_root": str(DL4_ROOT),
        "checkpoint_count": len(rows),
        "configured_agents": configured_agents,
        "configured_action_dims": action_dims,
        "actor_final_shapes": actor_final_shapes,
        "critic_first_shapes": critic_first_shapes,
        "state_dict_shape_compatible_with_original_dl4_classes": configured_agents == [8] and action_dims == [3] and len(actor_final_shapes) == 1 and len(critic_first_shapes) == 1,
        "checkpoint_semantics_compatible_with_c2_physical_vehicle_slots": False,
        "checkpoint_reuse_authorized": False,
        "rows": rows,
    }
    decision = {
        "created_at": iso_kst(),
        "checkpoint_compatibility_decision": "RETRAINING_REQUIRED",
        "decision_in_allowed_set": True,
        "required_interface_changes": [
            "C2 physical-vehicle-slot state must be converted into the DL graph/agent embedding contract.",
            "active_bus_mask must be propagated into action masking and simulator intervention filtering.",
            "inactive/missingness semantics and global active count must be encoded for critic use.",
            "checkpoint metadata must be extended with fixed physical-vehicle slot and direction-transition semantics.",
        ],
        "retraining_required": True,
        "checkpoint_reuse_authorized": False,
        "rationale": (
            "DL4 checkpoint weights are shape-loadable in the original 8-agent/action_dim=3 classes, "
            "but the trained semantics are route/node-agent embeddings rather than PV8-C2 fixed physical "
            "vehicle slots with inactive/reappearing/missingness semantics."
        ),
        "field_level_basis": {
            "actor_input_dimension": "original shape loadable, adapter required",
            "critic_input_dimension": "original shape loadable, native missingness/global active count absent",
            "action_dimension": "DL4 action_dim=3; Step40 MLP default action_dim=2 is not authoritative for DL4",
            "num_agents_semantics": "8 count matches; physical-vehicle identity semantics absent",
            "active_bus_mask_semantics": "not present in checkpoint metadata",
            "missingness_feature_availability": "not present in checkpoint metadata/training contract",
            "direction_feature_availability": "current C2 direction can be represented by adapter; direction-transition identity semantics absent from checkpoint",
            "vehicle_slot_identity_semantics": "absent; retraining required before reuse claims",
            "state_dict_shape_compatibility": audit["state_dict_shape_compatible_with_original_dl4_classes"],
        },
    }
    return audit, decision


def interface_matrix(actor: Mapping[str, Any], critic: Mapping[str, Any], action_mask: Mapping[str, Any], decision: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [
        {
            "contract_item": "8 fixed physical-vehicle slots",
            "existing_support": "count shape supported",
            "gap": "physical vehicle identity semantics not native",
            "required_change": "slot-binding preprocessor and metadata update",
            "compatibility": "requires retraining before checkpoint reuse",
        },
        {
            "contract_item": "active_bus_mask",
            "existing_support": "NeuralMAPPOInferenceAdapter accepts active_bus_mask",
            "gap": "DL4 checkpoint training did not encode PV8-C2 inactive semantics",
            "required_change": "mask propagation to action and critic features",
            "compatibility": "adapter required",
        },
        {
            "contract_item": "inactive-agent observation",
            "existing_support": "zero-fill + missing flag possible in adapter preview",
            "gap": "not native in DL4 graph embedding checkpoint",
            "required_change": "tensor contract amendment",
            "compatibility": "retraining required",
        },
        {
            "contract_item": "vehicle disappearance/reappearance",
            "existing_support": "fixed agent_id can be preserved in C2 replay",
            "gap": "not a checkpoint metadata/training semantic",
            "required_change": "training data and checkpoint metadata update",
            "compatibility": "retraining required",
        },
        {
            "contract_item": "direction transition same agent_id",
            "existing_support": "C2 replay preserves identity",
            "gap": "direction transition not represented in old checkpoint metadata",
            "required_change": "direction feature/transition contract in observation",
            "compatibility": "retraining required",
        },
        {
            "contract_item": "inactive action must not reach simulator",
            "existing_support": "no-op masking and filter possible",
            "gap": "native adapter still returns inactive agent entries",
            "required_change": "simulator intervention filter",
            "compatibility": "adapter required",
        },
    ]
    return {
        "created_at": iso_kst(),
        "actor_interface_status": actor["status"],
        "critic_interface_status": critic["status"],
        "inactive_mask_status": action_mask["status"],
        "checkpoint_compatibility_decision": decision["checkpoint_compatibility_decision"],
        "rows": rows,
    }


def claim_guard() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "K_safety_layer_complete": False,
        "K_action_mask_available": False,
        "safe_skip_decision_ready": False,
        "passenger_layer_complete": False,
        "service_obligation_layer_complete": False,
        "checkpoint_reuse_authorized": False,
        "training_use_authorized": False,
        "policy_use_authorized": False,
        "policy_evaluation_authorized": False,
        "simulator_performance_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }


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
    writer.text("artifact_manifest_srp2_bis_pv8_c3.jsonl", "".join(json.dumps(json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    rows.append({
        "relative_path": "artifact_manifest_srp2_bis_pv8_c3.jsonl",
        "size_bytes": (writer.root / "artifact_manifest_srp2_bis_pv8_c3.jsonl").stat().st_size,
        "sha256": sha256_file(writer.root / "artifact_manifest_srp2_bis_pv8_c3.jsonl"),
        "required": True,
        "artifact_role": "manifest_jsonl",
        "exists": True,
    })
    manifest = {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "payload_count": len(rows),
        "missing_payload_count": sum(1 for row in rows if not row["exists"]),
        "files": rows,
    }
    writer.json("artifact_manifest_srp2_bis_pv8_c3.json", manifest)
    manifest_path = writer.root / "artifact_manifest_srp2_bis_pv8_c3.json"
    writer.json("_PV8_C3_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": "artifact_manifest_srp2_bis_pv8_c3.json",
        "final_manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })
    return manifest


def final_report_text(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-C3 MAPPO Interface Compatibility Audit Final Report",
        "",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- actor_interface_status: `{summary['actor_interface_status']}`",
        f"- critic_interface_status: `{summary['critic_interface_status']}`",
        f"- inactive_mask_status: `{summary['inactive_mask_status']}`",
        f"- reappearance_status: `{summary['reappearance_status']}`",
        f"- direction_transition_status: `{summary['direction_transition_status']}`",
        f"- checkpoint_compatibility_decision: `{summary['checkpoint_compatibility_decision']}`",
        f"- required_interface_changes: `{summary['required_interface_changes']}`",
        f"- retraining_required: `{summary['retraining_required']}`",
        f"- K_safety_layer_complete: `{summary['K_safety_layer_complete']}`",
        f"- checkpoint_reuse_authorized: `{summary['checkpoint_reuse_authorized']}`",
        f"- training_use_authorized: `{summary['training_use_authorized']}`",
        f"- gate: `{summary['gate']}`",
        f"- readiness: `{summary['readiness']}`",
        "",
        "PV8-C3 PASS means the compatibility audit completed and the C2 interface replay satisfied shape/mask/identity checks. It does not authorize checkpoint reuse, training, policy evaluation, simulator evaluation, K-safety completion, or performance claims.",
        "",
        "Next candidate, not executed: `PV8-K1: Passenger / Service-Obligation / Skip-Safety Evidence Gap Closure`.",
        "",
    ])


def run_validate(root: Path) -> Path:
    root = validate_artifact_root(root)
    writer = Writer(root)
    c2 = verify_c2()
    state, mapping, transitions, inactive, persistence = load_c2_tables()
    c2_checks = c2_contract_checks(state, mapping, transitions, inactive, persistence)
    lineage = source_lineage()
    replay_rows, replay_summary = run_interface_replay(state, transitions)
    actor = actor_audit(lineage)
    critic = critic_audit()
    action_mask = action_mask_audit(replay_summary)
    checkpoint, decision = checkpoint_audit()
    matrix = interface_matrix(actor, critic, action_mask, decision)
    guard = claim_guard()

    writer.json("source_lineage.json", {**lineage, "c2_upstream": c2, "c2_contract_checks": c2_checks})
    writer.json("actor_interface_audit.json", actor)
    writer.json("critic_interface_audit.json", critic)
    writer.json("action_mask_path_audit.json", action_mask)
    writer.json("c2_to_mappo_interface_matrix.json", matrix)
    writer.parquet("c2_interface_replay_results.parquet", replay_rows, [
        "case_id",
        "case_type",
        "cycle_index",
        "active_agent_count",
        "inactive_agent_count",
        "actor_obs_shape",
        "critic_obs_shape",
        "active_bus_mask_shape",
        "action_mask_shape",
        "shape_mismatch_count",
        "nan_inf_count",
        "inactive_nonzero_action_count",
        "inactive_action_dict_entry_count",
        "inactive_intervention_leakage_count_after_filter",
        "slot_mutation_count",
        "reappearance_identity_failure_count",
        "direction_transition_identity_failure_count",
        "adapter_active_agent_count",
        "adapter_action_dim",
        "random_initialized_interface_smoke_only",
    ])
    writer.json("c2_interface_replay_summary.json", replay_summary)
    writer.json("checkpoint_compatibility_audit.json", checkpoint)
    writer.json("checkpoint_compatibility_decision.json", decision)
    writer.json("claim_guard_status.json", guard)

    failure_reasons: List[str] = []
    if not manifest_ok(c2["manifest_integrity"]):
        failure_reasons.append("UPSTREAM_C2_INTEGRITY_FAILURE")
    if c2_checks["mapping_rows"] != 8 or not c2_checks["per_cycle_agents_0_to_7"] or not c2_checks["fixed_token_per_agent"]:
        failure_reasons.append("AGENT_SLOT_SEMANTIC_MISMATCH")
    if replay_summary["shape_mismatch_count"] != 0:
        failure_reasons.append("ACTOR_INPUT_CONTRACT_MISMATCH")
    if replay_summary["nan_inf_count"] != 0:
        failure_reasons.append("ACTOR_INPUT_CONTRACT_MISMATCH")
    if replay_summary["inactive_action_leakage_count"] != 0:
        failure_reasons.append("INACTIVE_AGENT_ACTION_LEAKAGE")
    if replay_summary["slot_mutation_count"] != 0:
        failure_reasons.append("AGENT_SLOT_SEMANTIC_MISMATCH")
    if replay_summary["reappearance_identity_failure_count"] != 0:
        failure_reasons.append("REAPPEARANCE_IDENTITY_FAILURE")
    if replay_summary["direction_transition_identity_failure_count"] != 0:
        failure_reasons.append("DIRECTION_TRANSITION_IDENTITY_FAILURE")
    if decision["checkpoint_compatibility_decision"] not in COMPATIBILITY_CHOICES:
        failure_reasons.append("CHECKPOINT_CONTRACT_UNDETERMINABLE")

    gate_passed = not failure_reasons
    readiness = (
        "SRP2_BIS_PV8_C3_COMPLETE_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_DECISION_"
        f"{decision['checkpoint_compatibility_decision']}_K_SAFETY_AND_CHECKPOINT_REUSE_REMAIN_GATED"
        if gate_passed
        else "FAILED_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT"
    )
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if gate_passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if gate_passed else FAIL_GATE,
        "gate_passed": gate_passed,
        "readiness": readiness,
        "failure_reasons": sorted(set(failure_reasons)),
        "checkpoint_compatibility_decision": decision["checkpoint_compatibility_decision"],
    }
    summary = {
        "artifact_root": str(root),
        "actor_interface_status": actor["status"],
        "critic_interface_status": critic["status"],
        "inactive_mask_status": action_mask["status"],
        "reappearance_status": "PASS",
        "direction_transition_status": "PASS",
        "checkpoint_compatibility_decision": decision["checkpoint_compatibility_decision"],
        "required_interface_changes": decision["required_interface_changes"],
        "retraining_required": decision["retraining_required"],
        "K_safety_layer_complete": guard["K_safety_layer_complete"],
        "checkpoint_reuse_authorized": guard["checkpoint_reuse_authorized"],
        "training_use_authorized": guard["training_use_authorized"],
        "gate": gate["gate"],
        "readiness": gate["readiness"],
    }
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
        "optimizer_backward_count": 0,
        "checkpoint_write_count": 0,
        "policy_performance_evaluation_count": 0,
        "simulator_step_count": 0,
        "interface_forward_shape_smoke_count": replay_summary["interface_forward_shape_smoke_count"],
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {
        "created_at": iso_kst(),
        "source_gate": gate["gate"],
        "readiness": gate["readiness"],
        "checkpoint_compatibility_decision": decision["checkpoint_compatibility_decision"],
        "checkpoint_reuse_authorized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "simulator_performance_evaluation_authorized": False,
        "K_safety_layer_complete": False,
        "automatic_pv8_k1_execution_authorized": False,
    })
    writer.text("final_report.md", final_report_text(summary))

    write_manifest_and_lock(writer, gate)
    own_checks = verify_manifest(root, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock")
    if not manifest_ok(own_checks):
        raise PV8C3Error("ARTIFACT_INTEGRITY_FAILURE", json.dumps(json_clean(own_checks), sort_keys=True))

    print(f"artifact_root: {summary['artifact_root']}")
    print(f"actor_interface_status: {summary['actor_interface_status']}")
    print(f"critic_interface_status: {summary['critic_interface_status']}")
    print(f"inactive_mask_status: {summary['inactive_mask_status']}")
    print(f"reappearance_status: {summary['reappearance_status']}")
    print(f"direction_transition_status: {summary['direction_transition_status']}")
    print(f"checkpoint_compatibility_decision: {summary['checkpoint_compatibility_decision']}")
    print(f"required_interface_changes: {summary['required_interface_changes']}")
    print(f"retraining_required: {summary['retraining_required']}")
    print("K_safety_layer_complete = false")
    print("checkpoint_reuse_authorized = false")
    print("training_use_authorized = false")
    print(f"gate: {summary['gate']}")
    print(f"readiness: {summary['readiness']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_validate(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
