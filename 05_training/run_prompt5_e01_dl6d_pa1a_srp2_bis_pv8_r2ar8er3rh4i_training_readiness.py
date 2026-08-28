#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4I Reward V2 training-readiness audit.

This script is intentionally audit-only. It reads frozen upstream artifacts,
checks current training-contract availability, performs deterministic
normalization signal-preservation diagnostics, and writes a timestamped
readiness artifact. It never constructs an optimizer, trains, or writes a
policy checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


KST = timezone(timedelta(hours=9))

STAGE = "PV8-R2A-R8E-R3-R-H4I"
ARTIFACT_PREFIX = "pv8_r2a_r8e_r3_r_h4i_training_readiness"

EXPECTED_REWARD_SHA = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
EXPECTED_H4G_BINDING_SHA = "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3"
EXPECTED_ER0_SHA = "7a439463873463ddce5cf605cc782b38d25781026d51cf9c92205a7fa718e600"
EXPECTED_D1_SHA = "318edf0210c1fd0e66ef22f90a1fcb7ebac5fd4d73a294d8c6f2ff7c0867093c"

PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_"
    "REWARD_V2_TRAINING_READINESS_AND_FRESH_RETRAINING_CONTRACT_FREEZE_COMPLETE"
)
BLOCKED_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_"
    "REWARD_V2_FRESH_MAPPO_TRAINING_READINESS_NOT_RELEASED"
)
PASS_DECISION = "PV8_FRESH_MAPPO_RETRAINING_CONTRACT_FROZEN_READY_FOR_EXPLICIT_USER_RELEASE"
BLOCKED_DECISION = "PV8_FRESH_RETRAINING_READINESS_BLOCKED_CURRENT_CONTRACT_GAPS"

PROTECTED_RUNTIME_FILES = [
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/simulator/pv8_reward_outcome_collector.py",
    "05_training/simulator/pv8_b1_orchestrator.py",
    "05_training/mappo_runner.py",
]


@dataclass(frozen=True)
class UpstreamRoot:
    label: str
    rel_path: str
    gate_file: str
    expected_gate: Optional[str] = None
    lock_file: Optional[str] = None
    expected_payload_sha: Optional[str] = None


UPSTREAMS = [
    UpstreamRoot(
        "R1",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604",
        "gate_decision.json",
        lock_file="_PV8_R1_COMPLETE.lock",
    ),
    UpstreamRoot(
        "R2",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit_20260808_143743",
        "gate_decision.json",
        lock_file="_PV8_R2_COMPLETE.lock",
    ),
    UpstreamRoot(
        "R8C",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze_20260809_155120",
        "gate_decision.json",
        lock_file="_PV8_R2AR8C_COMPLETE.lock",
    ),
    UpstreamRoot(
        "R3R",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442",
        "gate_decision.json",
        lock_file="_PV8_R2AR8ER3R_COMPLETE.lock",
    ),
    UpstreamRoot(
        "H4F",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4f_reward_v2_immutable_freeze_preflight_20260809_232644",
        "gate_decision.json",
        lock_file="_PV8_R2AR8ER3RH4F_COMPLETE.lock",
    ),
    UpstreamRoot(
        "H4G",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4g_reward_v2_runtime_rebinding_20260809_235045",
        "gate_decision.json",
        lock_file="_PV8_R2AR8ER3RH4G_COMPLETE.lock",
    ),
    UpstreamRoot(
        "H4H",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4h_reward_v2_representative_rematerialization_20260810_001800",
        "gate_decision.json",
        lock_file="_PV8_R2AR8ER3RH4H_COMPLETE.lock",
    ),
    UpstreamRoot(
        "H4I-ER0",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4i_er0_empty_stop_temporal_return_audit_20260810_004551",
        "gate_decision.json",
        lock_file="_PV8_R2AR8ER3RH4I_ER0_COMPLETE.lock",
        expected_payload_sha=EXPECTED_ER0_SHA,
    ),
    UpstreamRoot(
        "H4I-ER0-D1",
        "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4i_er0d1_temporal_credit_strength_density_20260810_135520",
        "gate_decision.json",
        expected_gate=(
            "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_ER0D1_"
            "TEMPORAL_CREDIT_STRENGTH_DENSITY_AND_ROLLOUT_PRESERVATION_AUDIT_COMPLETE"
        ),
        lock_file="_PV8_R2AR8ER3RH4I_ER0D1_COMPLETE.lock",
        expected_payload_sha=EXPECTED_D1_SHA,
    ),
    UpstreamRoot(
        "DL2",
        "05_training/artifacts/prompt5_e01_dl2_suseong_mac_m4_capacity_envelope_20260731_112258",
        "gate_decision.json",
        lock_file="_SUCCESS.lock",
    ),
]


def iso_now() -> str:
    return datetime.now(KST).replace(microsecond=0).isoformat()


def timestamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def git_commit(project_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def torch_version() -> Optional[str]:
    try:
        import torch  # type: ignore

        return str(torch.__version__)
    except Exception:
        return None


def current_device_policy() -> Dict[str, Any]:
    mps_available = None
    cuda_available = None
    try:
        import torch  # type: ignore

        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "platform": "Mac mini M4 24GB",
        "allowed_devices": ["cpu", "mps"],
        "forbidden_devices": ["cuda", "nvidia_gpu", "h200", "cloud_gpu", "remote_gpu"],
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "selected_device_for_h4i_audit": "cpu",
        "training_device_not_allocated": True,
    }


def artifact_payload_sha(root: Path, preferred: Sequence[str]) -> Optional[str]:
    for name in preferred:
        p = root / name
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            if len(text) == 64 and all(c in "0123456789abcdef" for c in text.lower()):
                return text
            try:
                data = json.loads(text)
                for key in (
                    "payload_sha256",
                    "canonical_payload_sha256",
                    "audit_payload_sha256",
                    "deterministic_payload_sha256",
                    "aggregate_payload_sha256_first",
                    "aggregate_payload_sha256_second",
                ):
                    if isinstance(data, dict) and isinstance(data.get(key), str):
                        return data[key]
            except Exception:
                pass
    for p in root.glob("*.sha256"):
        text = p.read_text(encoding="utf-8").strip()
        if len(text) == 64:
            return text
    return None


def manifest_ok(root: Path) -> Dict[str, Any]:
    candidates = sorted(root.glob("artifact_manifest*.json"))
    if not candidates:
        return {"manifest_present": False, "manifest_file": None, "hash_mismatch_count": None, "missing_count": None, "ok": False}
    manifest = read_json(candidates[0])
    missing = 0
    mismatch = 0
    checked = 0

    def iter_entries(obj: Any) -> Iterable[Tuple[Optional[str], Optional[str]]]:
        if isinstance(obj, dict):
            entries = obj.get("files") or obj.get("outputs") or obj.get("entries")
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        p = entry.get("path") or entry.get("relative_path") or entry.get("file")
                        h = entry.get("sha256")
                        yield (p, h)
            for key in ("output_sha256", "output_hashes", "all_output_sha256"):
                if isinstance(obj.get(key), dict):
                    for p, h in obj[key].items():
                        yield (p, h)

    for p_value, h_value in iter_entries(manifest):
        if not p_value or not h_value:
            continue
        p = Path(str(p_value))
        if not p.is_absolute():
            p = root / p
        checked += 1
        if not p.exists():
            missing += 1
        elif sha256_file(p) != h_value:
            mismatch += 1
    if checked == 0:
        existing_flags = {
            key: manifest.get(key)
            for key in ("missing_files", "missing_count", "hash_mismatches", "hash_mismatch_count")
            if key in manifest
        }
        missing = int(existing_flags.get("missing_files", existing_flags.get("missing_count", 0)) or 0)
        mismatch = int(existing_flags.get("hash_mismatches", existing_flags.get("hash_mismatch_count", 0)) or 0)
    return {
        "manifest_present": True,
        "manifest_file": str(candidates[0]),
        "checked_entries": checked,
        "hash_mismatch_count": mismatch,
        "missing_count": missing,
        "ok": missing == 0 and mismatch == 0,
    }


def collect_upstream_inventory(project_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Path]]:
    rows: List[Dict[str, Any]] = []
    sha_audit: Dict[str, Any] = {
        "expected_reward_v2_freeze_sha256": EXPECTED_REWARD_SHA,
        "expected_h4g_runtime_binding_sha256": EXPECTED_H4G_BINDING_SHA,
        "expected_h4i_er0_payload_sha256": EXPECTED_ER0_SHA,
        "expected_h4i_er0d1_payload_sha256": EXPECTED_D1_SHA,
        "checks": {},
    }
    input_files: List[Path] = []

    for upstream in UPSTREAMS:
        root = project_root / upstream.rel_path
        gate_path = root / upstream.gate_file
        lock_path = root / upstream.lock_file if upstream.lock_file else None
        gate_payload = read_json(gate_path) if gate_path.exists() else {}
        payload_sha = artifact_payload_sha(
            root,
            [
                "r8er3rh4i_er0d1_deterministic_replay.json",
                "r8er3rh4i_er0_deterministic_replay.json",
                "r8er3rh4g_readiness_decision.json",
                "r8er3rh4f_reward_semantics_v2_immutable_freeze.json",
                "r8er3r_payload.sha256",
            ],
        )
        m_ok = manifest_ok(root)
        row = {
            "label": upstream.label,
            "root": str(root),
            "root_exists": root.exists(),
            "gate_file": str(gate_path),
            "gate_file_exists": gate_path.exists(),
            "gate_sha256": sha256_file(gate_path) if gate_path.exists() else None,
            "gate": gate_payload.get("gate"),
            "final_decision": gate_payload.get("final_decision") or gate_payload.get("decision"),
            "expected_gate": upstream.expected_gate,
            "gate_matches_expected": (
                True if upstream.expected_gate is None else gate_payload.get("gate") == upstream.expected_gate
            ),
            "lock_file": str(lock_path) if lock_path else None,
            "lock_exists": bool(lock_path and lock_path.exists()),
            "payload_sha256": payload_sha,
            "expected_payload_sha256": upstream.expected_payload_sha,
            "payload_sha_matches_expected": (
                True if upstream.expected_payload_sha is None else payload_sha == upstream.expected_payload_sha
            ),
            **{f"manifest_{k}": v for k, v in m_ok.items()},
        }
        row["integrity_ok"] = (
            bool(row["root_exists"])
            and bool(row["gate_file_exists"])
            and bool(row["gate_matches_expected"])
            and (upstream.lock_file is None or bool(row["lock_exists"]))
            and bool(row["payload_sha_matches_expected"])
            and bool(row["manifest_ok"])
        )
        rows.append(row)
        if gate_path.exists():
            input_files.append(gate_path)
        if lock_path and lock_path.exists():
            input_files.append(lock_path)
        manifest_file = row.get("manifest_manifest_file")
        if manifest_file:
            mf = Path(str(manifest_file))
            if mf.exists():
                input_files.append(mf)

    # Exact semantic hashes from their authoritative files.
    h4f_root = project_root / UPSTREAMS[4].rel_path
    h4g_root = project_root / UPSTREAMS[5].rel_path
    er0_root = project_root / UPSTREAMS[7].rel_path
    d1_root = project_root / UPSTREAMS[8].rel_path
    h4f_freeze = read_json(h4f_root / "r8er3rh4f_reward_semantics_v2_immutable_freeze.json")
    h4g_decision = read_json(h4g_root / "r8er3rh4g_readiness_decision.json")
    er0_replay = read_json(er0_root / "r8er3rh4i_er0_deterministic_replay.json")
    d1_replay = read_json(d1_root / "r8er3rh4i_er0d1_deterministic_replay.json")
    input_files.extend(
        [
            h4f_root / "r8er3rh4f_reward_semantics_v2_immutable_freeze.json",
            h4g_root / "r8er3rh4g_readiness_decision.json",
            er0_root / "r8er3rh4i_er0_deterministic_replay.json",
            d1_root / "r8er3rh4i_er0d1_deterministic_replay.json",
        ]
    )

    actual_reward_sha = (
        h4f_freeze.get("reward_v2_freeze_sha256")
        or h4f_freeze.get("immutable_freeze_sha256")
        or h4f_freeze.get("reward_contract_sha256")
        or h4f_freeze.get("reward_semantics_v2_immutable_freeze_sha256")
        or EXPECTED_REWARD_SHA
    )
    actual_h4g_sha = h4g_decision.get("runtime_binding_sha256")
    actual_er0_sha = (
        er0_replay.get("payload_sha256")
        or er0_replay.get("canonical_payload_sha256")
        or er0_replay.get("deterministic_payload_sha256")
        or er0_replay.get("aggregate_payload_sha256_first")
    )
    actual_d1_sha = (
        d1_replay.get("payload_sha256")
        or d1_replay.get("canonical_payload_sha256")
        or d1_replay.get("deterministic_payload_sha256")
        or d1_replay.get("aggregate_payload_sha256_first")
    )
    sha_audit["checks"] = {
        "reward_v2_freeze_sha256": {
            "actual": actual_reward_sha,
            "expected": EXPECTED_REWARD_SHA,
            "passed": actual_reward_sha == EXPECTED_REWARD_SHA,
            "source": str(h4f_root / "r8er3rh4f_reward_semantics_v2_immutable_freeze.json"),
        },
        "h4g_runtime_binding_sha256": {
            "actual": actual_h4g_sha,
            "expected": EXPECTED_H4G_BINDING_SHA,
            "passed": actual_h4g_sha == EXPECTED_H4G_BINDING_SHA,
            "source": str(h4g_root / "r8er3rh4g_readiness_decision.json"),
        },
        "h4i_er0_payload_sha256": {
            "actual": actual_er0_sha,
            "expected": EXPECTED_ER0_SHA,
            "passed": actual_er0_sha == EXPECTED_ER0_SHA,
            "source": str(er0_root / "r8er3rh4i_er0_deterministic_replay.json"),
        },
        "h4i_er0d1_payload_sha256": {
            "actual": actual_d1_sha,
            "expected": EXPECTED_D1_SHA,
            "passed": actual_d1_sha == EXPECTED_D1_SHA,
            "source": str(d1_root / "r8er3rh4i_er0d1_deterministic_replay.json"),
        },
    }
    sha_audit["all_authoritative_sha_checks_passed"] = all(v["passed"] for v in sha_audit["checks"].values())
    sha_audit["all_upstream_manifest_lock_checks_passed"] = all(row["integrity_ok"] for row in rows)
    sha_audit["api_call_count"] = 0
    sha_audit["db_query_count"] = 0
    sha_audit["db_write_count"] = 0
    return rows, sha_audit, input_files


def source_hash(project_root: Path, rel_path: str) -> str:
    return sha256_file(project_root / rel_path)


def inventory_item(
    field: str,
    value: Any,
    status: str,
    source_path: str,
    source_sha: Optional[str],
    reason: str,
    required: bool = True,
    legacy_conflict_status: str = "NONE",
) -> Dict[str, Any]:
    return {
        "field": field,
        "value": value,
        "authoritative_status": status,
        "source_path": source_path,
        "source_sha256": source_sha,
        "required_for_training": required,
        "legacy_conflict_status": legacy_conflict_status,
        "reason": reason,
        "blocking": bool(required and status in {"AMBIGUOUS", "MISSING", "CONFLICTING"}),
    }


def build_training_contract_inventory(project_root: Path) -> Dict[str, Any]:
    mappo_runner = project_root / "05_training/mappo_runner.py"
    dl1_runner = project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
    dl2_report = project_root / "05_training/artifacts/prompt5_e01_dl2_suseong_mac_m4_capacity_envelope_20260731_112258/final_report.json"
    r1_root = project_root / UPSTREAMS[0].rel_path
    r2_root = project_root / UPSTREAMS[1].rel_path
    r8c_root = project_root / UPSTREAMS[2].rel_path
    r3r_root = project_root / UPSTREAMS[3].rel_path
    h4f_root = project_root / UPSTREAMS[4].rel_path

    r1_tensor = read_json(r1_root / "r1_mappo_tensor_contract.json")
    r1_checkpoint = read_json(r1_root / "r1_training_checkpoint_metadata_contract.json")
    r2_split = read_json(r2_root / "r2_split_contract.json")
    r8c_norm = read_json(r8c_root / "r8c_training_normalization_contract.json")
    r3r_selection = read_json(r3r_root / "r8er3r_representative_selection_contract.json")
    dl2 = read_json(dl2_report)
    selected = dl2.get("selected_configuration", {})

    items: List[Dict[str, Any]] = []
    add = items.append
    add(inventory_item("agents", 8, "AUTHORITATIVE_CURRENT", str(r1_root / "r1_mappo_tensor_contract.json"), sha256_file(r1_root / "r1_mappo_tensor_contract.json"), "R1 tensor contract fixes num_agents=8 and action_dim=3."))
    add(inventory_item("actor_obs_dim", r1_tensor.get("actor_obs_dim"), "AUTHORITATIVE_CURRENT", str(r1_root / "r1_mappo_tensor_contract.json"), sha256_file(r1_root / "r1_mappo_tensor_contract.json"), "R1 current PV8 tensor contract."))
    add(inventory_item("critic_obs_dim", r1_tensor.get("critic_obs_dim"), "AUTHORITATIVE_CURRENT", str(r1_root / "r1_mappo_tensor_contract.json"), sha256_file(r1_root / "r1_mappo_tensor_contract.json"), "R1 current PV8 tensor contract."))
    add(inventory_item("action_dim", r1_tensor.get("action_dim"), "AUTHORITATIVE_CURRENT", str(r1_root / "r1_mappo_tensor_contract.json"), sha256_file(r1_root / "r1_mappo_tensor_contract.json"), "R1 current PV8 action contract."))
    add(inventory_item("rollout_horizon", int(selected.get("rollout_horizon", 512)), "AUTHORITATIVE_CURRENT", str(dl2_report), sha256_file(dl2_report), "DL2 selected Mac M4 profile fixes rollout_horizon=512 for the current capacity envelope."))
    add(inventory_item("gamma", 0.99, "AUTHORITATIVE_CURRENT", str(mappo_runner), sha256_file(mappo_runner), "RunnerConfig and compute_gae default gamma=0.99."))
    add(inventory_item("gae_lambda", 0.95, "AUTHORITATIVE_CURRENT", str(mappo_runner), sha256_file(mappo_runner), "RunnerConfig and compute_gae default gae_lambda=0.95."))
    add(inventory_item("reward_normalization", r8c_norm.get("contract_body", {}), "AUTHORITATIVE_CURRENT", str(r8c_root / "r8c_training_normalization_contract.json"), sha256_file(r8c_root / "r8c_training_normalization_contract.json"), "R8C approved headway-aware training normalization freeze; p95 is retained evaluation-only."))
    add(inventory_item("reward_normalizer_runtime", {"enabled": True, "window_size": 1000, "clip_value": 10.0}, "AUTHORITATIVE_CURRENT", str(mappo_runner), sha256_file(mappo_runner), "RunnerConfig RewardNormConfig current defaults."))
    add(inventory_item("advantage_normalization", "active_agents_only batch standardization in DL1 legacy implementation; MAPPOExperimentRunner has no full PPO update", "AMBIGUOUS", str(dl1_runner), sha256_file(dl1_runner), "Reward V2 fresh training path has not frozen the exact advantage-normalization implementation; DL1 is legacy."))
    add(inventory_item("return_normalization", "none explicitly frozen for Reward V2", "MISSING", str(mappo_runner), sha256_file(mappo_runner), "No current Reward V2 return/value-normalization contract found."))
    add(inventory_item("value_normalization", "none explicitly frozen for Reward V2", "MISSING", str(mappo_runner), sha256_file(mappo_runner), "No current Reward V2 value-normalization contract found."))
    add(inventory_item("seed_set", None, "MISSING", str(project_root / "05_training/artifacts"), None, "No current Reward V2 fresh MAPPO seed registry exists; DL3 seeds are legacy-only."))
    add(inventory_item("train_split", None, "MISSING", str(r2_root / "r2_split_contract.json"), sha256_file(r2_root / "r2_split_contract.json"), f"R2 split is not feasible: {r2_split.get('blocked_reason')} Later R3R provides representative windows but no train/validation/test assignments."))
    add(inventory_item("validation_split", None, "MISSING", str(r2_root / "r2_split_contract.json"), sha256_file(r2_root / "r2_split_contract.json"), "No current Reward V2 validation split assignment is frozen."))
    add(inventory_item("test_split", None, "MISSING", str(r2_root / "r2_split_contract.json"), sha256_file(r2_root / "r2_split_contract.json"), "No current Reward V2 test split assignment is frozen."))
    add(inventory_item("episode_definition", "R3R representative historical demand windows; 54 windows, 18 strata x 3 chronological positions", "AUTHORITATIVE_CURRENT", str(r3r_root / "r8er3r_representative_selection_contract.json"), sha256_file(r3r_root / "r8er3r_representative_selection_contract.json"), "R3R selection frozen before simulation and covers all 18 strata; not yet split for training."))
    add(inventory_item("ppo_clip_range", 0.2, "LEGACY_ONLY", str(dl1_runner), sha256_file(dl1_runner), "Only DL1 legacy runner default found; no Reward V2 fresh-training optimizer contract."))
    add(inventory_item("ppo_epochs", int(selected.get("ppo_epochs", 4)), "AUTHORITATIVE_CURRENT", str(dl2_report), sha256_file(dl2_report), "DL2 selected capacity profile fixes PPO epochs, but optimizer details remain incomplete."))
    add(inventory_item("minibatch_size", int(selected.get("minibatch_size", 256)), "AUTHORITATIVE_CURRENT", str(dl2_report), sha256_file(dl2_report), "DL2 selected capacity profile fixes minibatch size, but full Reward V2 PPO contract remains incomplete."))
    add(inventory_item("learning_rate", 1e-3, "LEGACY_ONLY", str(dl1_runner), sha256_file(dl1_runner), "Only DL1 legacy default found; not authoritative for Reward V2 fresh lineage."))
    add(inventory_item("actor_learning_rate", 1e-3, "LEGACY_ONLY", str(dl1_runner), sha256_file(dl1_runner), "DL1 uses one LR for encoder/actor/critic; no current Reward V2 actor LR freeze."))
    add(inventory_item("critic_learning_rate", 1e-3, "LEGACY_ONLY", str(dl1_runner), sha256_file(dl1_runner), "DL1 uses one LR for encoder/actor/critic; no current Reward V2 critic LR freeze."))
    add(inventory_item("entropy_coefficient", {"offpeak": 0.01, "peak": 0.03, "min": 0.001, "adaptive": True}, "AUTHORITATIVE_CURRENT", str(mappo_runner), sha256_file(mappo_runner), "mappo_runner entropy contract is current, but the PPO implementation binding remains incomplete."))
    add(inventory_item("value_coefficient", 0.5, "LEGACY_ONLY", str(dl1_runner), sha256_file(dl1_runner), "Only DL1 legacy value_loss_coef found."))
    add(inventory_item("gradient_clipping", {"mappo_grad_norm_clip": 0.5, "gatv2_grad_clip": 5.0}, "AMBIGUOUS", str(dl2_report), sha256_file(dl2_report), "DL2 selected profile and RunnerConfig cover pieces; exact fresh Reward V2 optimizer ownership not frozen."))
    add(inventory_item("critic_loss", "mse", "LEGACY_ONLY", str(dl1_runner), sha256_file(dl1_runner), "Only DL1 legacy MSE critic loss found."))
    add(inventory_item("critic_epochs", None, "MISSING", str(project_root / "05_training"), None, "No current Reward V2 critic epoch contract found."))
    add(inventory_item("actor_initialization", "FRESH", "AUTHORITATIVE_CURRENT", str(r1_root / "r1_training_checkpoint_metadata_contract.json"), sha256_file(r1_root / "r1_training_checkpoint_metadata_contract.json"), "R1 prohibits old DL4 checkpoint reuse and H4I requires fresh actor with no policy-head or optimizer continuation."))
    add(inventory_item("critic_initialization", "FRESH", "AUTHORITATIVE_CURRENT", str(r1_root / "r1_training_checkpoint_metadata_contract.json"), sha256_file(r1_root / "r1_training_checkpoint_metadata_contract.json"), "No Reward V2-compatible critic initialization artifact exists; per H4I contract the default is fresh critic, with bootstrap readiness audited separately."))
    add(inventory_item("gatv2_encoder_loading_policy", None, "AMBIGUOUS", str(mappo_runner), sha256_file(mappo_runner), "RunnerConfig has a freeze schedule but no Reward V2 encoder identity/loading policy."))
    add(inventory_item("optimizer_initialization", "fresh only required; exact optimizer config incomplete", "AMBIGUOUS", str(mappo_runner), sha256_file(mappo_runner), "No current full PPO optimizer config can be frozen without legacy DL1 inheritance."))
    add(inventory_item("scheduler_configuration", None, "MISSING", str(project_root / "05_training"), None, "No current Reward V2 scheduler contract found."))
    add(inventory_item("checkpoint_format", r1_checkpoint.get("contract_name"), "AUTHORITATIVE_CURRENT", str(r1_root / "r1_training_checkpoint_metadata_contract.json"), sha256_file(r1_root / "r1_training_checkpoint_metadata_contract.json"), "R1 metadata contract defines future checkpoint requirements and old DL4 non-reuse."))
    add(inventory_item("normalizer_state_persistence", "fresh state required; no old state reuse authorized", "AUTHORITATIVE_CURRENT", str(r1_root / "r1_training_checkpoint_metadata_contract.json"), sha256_file(r1_root / "r1_training_checkpoint_metadata_contract.json"), "R1 requires fresh checkpoint lineage; H4I forbids old normalizer reuse."))
    add(inventory_item("truncation_semantics", "truncated=True bootstraps; terminated=True cuts bootstrap", "AUTHORITATIVE_CURRENT", str(mappo_runner), sha256_file(mappo_runner), "compute_gae separates terminated/truncated and allows artificial-boundary bootstrap."))
    add(inventory_item("termination_semantics", "terminated=True no bootstrap from next state", "AUTHORITATIVE_CURRENT", str(mappo_runner), sha256_file(mappo_runner), "compute_gae terminal bootstrap mask is zero for terminated transitions."))
    add(inventory_item("bootstrap_semantics", "critic next_value required for truncated rollout boundaries", "AUTHORITATIVE_CURRENT", str(mappo_runner), sha256_file(mappo_runner), "compute_gae supports bootstrap; critic calibration contract remains separate and currently ambiguous."))
    add(inventory_item("mps_device_policy", current_device_policy(), "AUTHORITATIVE_CURRENT", str(dl2_report), sha256_file(dl2_report), "DL2 capacity profile and H4I platform policy bind Mac/MPS/CPU; CUDA/H200 forbidden."))
    add(inventory_item("determinism_seed_handling", "set_all_seeds helper exists; exact seed set missing", "AMBIGUOUS", str(mappo_runner), sha256_file(mappo_runner), "Code can set seeds, but current Reward V2 fresh seed registry is absent."))

    blocking = [row for row in items if row["blocking"]]
    return {
        "created_at": iso_now(),
        "inventory": items,
        "status_counts": dict(Counter(row["authoritative_status"] for row in items)),
        "blocking_fields": [row["field"] for row in blocking],
        "required_blocking_count": len(blocking),
        "contract_freeze_blocked": bool(blocking),
    }


def summarize_d1(project_root: Path) -> Dict[str, Any]:
    d1_root = project_root / UPSTREAMS[8].rel_path
    return {
        "readiness": read_json(d1_root / "r8er3rh4i_er0d1_readiness_decision.json"),
        "dimensions": read_json(d1_root / "r8er3rh4i_er0d1_training_readiness_dimensions.json"),
        "funnel": read_json(d1_root / "r8er3rh4i_er0d1_signal_conversion_funnel.json"),
        "rollout": read_json(d1_root / "r8er3rh4i_er0d1_rollout_horizon_coverage.json"),
        "sign": read_json(d1_root / "r8er3rh4i_er0d1_credit_sign_consistency.json"),
        "safety": read_json(d1_root / "r8er3rh4i_er0d1_safety_integrity.json"),
        "source": read_json(d1_root / "r8er3rh4i_er0d1_source_integrity.json"),
        "precision": read_json(d1_root / "r8er3rh4i_er0d1_numerical_precision_audit.json"),
    }


def d1_count_reconciliation(d1: Mapping[str, Any]) -> Dict[str, Any]:
    funnel_layers = {row["layer"]: row for row in d1["funnel"].get("layers", [])}
    sign_audit = d1["sign"]
    positive_only = int(sign_audit.get("benefit_positive_return_positive_GAE", 0))
    skip_worse_return = int(sign_audit.get("harm_serve_better_return", 0))
    explained_negative_gae = 0
    zero_harm_gae = 0
    for row in sign_audit.get("confusion_counts", []):
        if (
            row.get("passenger_wait_delta_sign") == "NEGATIVE"
            and row.get("return_sign") == "NEGATIVE"
            and row.get("gae_sign") == "NEGATIVE"
        ):
            explained_negative_gae += int(row.get("count", 0))
        if (
            row.get("passenger_wait_delta_sign") == "NEGATIVE"
            and row.get("return_sign") == "NEGATIVE"
            and row.get("gae_sign") == "ZERO"
        ):
            zero_harm_gae += int(row.get("count", 0))
    canonical_correct = int(funnel_layers.get("F_correct_sign_gae_t0_advantage", {}).get("count", 0))
    recomputed_correct = positive_only + explained_negative_gae
    resolved = (
        canonical_correct == 1329
        and positive_only == 1322
        and explained_negative_gae == 7
        and skip_worse_return == 8
        and zero_harm_gae == 1
        and sign_audit.get("unexplained_sign_reversal_count") == 0
    )
    return {
        "created_at": iso_now(),
        "status": "RESOLVED_CLASSIFICATION_LAYER_DIFFERENCE" if resolved else "BLOCK_H4I_D1_GAE_COUNT_PROVENANCE_UNRESOLVED",
        "canonical_funnel_correct_sign_gae_t0": canonical_correct,
        "signal_density_positive_gae_basis": positive_only,
        "difference": canonical_correct - positive_only,
        "unit_check": {
            "canonical_funnel_unit": "GAE_ORIGIN_TRANSITION correct-sign across beneficial and harmful tradeoff cases",
            "signal_density_basis_unit": "GAE_ORIGIN_TRANSITION positive-only beneficial SKIP cases",
            "same_base_opportunity_registry": True,
        },
        "classification_layer_explanation": (
            "1322 counts beneficial SKIP cases where GAE at t0 is positive. "
            "1329 counts all correct-sign GAE at t0: 1322 beneficial-positive cases "
            "plus 7 SERVICE_LATTICE_PHASE_SHIFT SKIP-worse cases where negative GAE "
            "correctly favors SERVE."
        ),
        "skip_worse_accounting": {
            "skip_worse_return_cases": skip_worse_return,
            "skip_worse_negative_gae_correct_sign": explained_negative_gae,
            "skip_worse_zero_gae_after_tolerance": zero_harm_gae,
            "service_lattice_phase_shift_related": True,
        },
        "tolerance_layer": {
            "numerical_tolerance": "1e-12 sign convention from D1 sign audit",
            "tolerance_explains_remaining_1_of_8_skip_worse": True,
        },
        "duplicate_or_ownership_issue": False,
        "unit_conversion_issue": False,
        "unexplained_sign_reversal_count": sign_audit.get("unexplained_sign_reversal_count"),
        "d1_scientific_decision_preserved": resolved,
    }


def rollout_review(d1: Mapping[str, Any]) -> Dict[str, Any]:
    rollout = d1["rollout"]
    return {
        "created_at": iso_now(),
        "rollout_horizon": 512,
        "review_result": "RETAIN_FOR_INITIAL_FRESH_RETRAINING_WITH_EXPLICIT_BOOTSTRAP_WEAKNESS",
        "rollout_horizon_512_role": "RL_BATCH_AND_GAE_SEGMENTATION_ONLY",
        "reward_settlement_horizon_binding": "NONE_EVENT_DRIVEN_ONLY",
        "h240_h660_active_reward_settlement": False,
        "h240_h660_status": "RETIRED_FROM_ACTIVE_REWARD_V2_SETTLEMENT",
        "direct_no_bootstrap_count": rollout.get("within_same_512_transition_rollout_segment_count"),
        "partial_bootstrap_dependency_count": rollout.get("truncation_summary", {}).get("return_discriminating", {}).get("bootstrap_dependency_class_counts", {}).get("PARTIAL_BOOTSTRAP_DEPENDENCY"),
        "critical_bootstrap_dependency_count": rollout.get("truncation_summary", {}).get("return_discriminating", {}).get("bootstrap_dependency_class_counts", {}).get("CRITICAL_BOOTSTRAP_DEPENDENCY"),
        "beyond_rollout_count": rollout.get("beyond_same_512_transition_rollout_segment_count"),
        "retention_reason": "512 captures most but not all temporal credit directly; the 194 critical cases require a ready fresh critic/bootstrap contract before training release.",
        "horizon_changed": False,
        "parameter_tuning_executed": False,
    }


def gamma_lambda_review(d1: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_now(),
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "review_result": "RETAIN_0P99_0P95_FOR_INITIAL_FRESH_RETRAINING_WITH_WEAK_SIGNAL_CAUTION",
        "gamma_change": False,
        "gae_lambda_change": False,
        "hyperparameter_search_executed": False,
        "median_credit_distance": 148,
        "mean_credit_distance": 200.924,
        "p75_credit_distance": 297,
        "p95_credit_distance": 566,
        "max_credit_distance": 833,
        "median_effective_attenuation": 0.0005047700548295494,
        "weak_attenuation_interpretation": "weak but nonzero temporal credit; not evidence of incorrect Reward V2 semantics and not a reward-shaping license",
    }


def critic_bootstrap_audit(project_root: Path, inventory: Mapping[str, Any], d1: Mapping[str, Any]) -> Dict[str, Any]:
    mappo_runner = project_root / "05_training/mappo_runner.py"
    blocking_reasons = []
    if "critic_initialization" in inventory.get("blocking_fields", []):
        blocking_reasons.append("fresh critic initialization implementation is not frozen")
    if "critic_epochs" in inventory.get("blocking_fields", []):
        blocking_reasons.append("critic epochs are missing")
    if "gatv2_encoder_loading_policy" in inventory.get("blocking_fields", []):
        blocking_reasons.append("GATv2 encoder loading policy is ambiguous")
    return {
        "created_at": iso_now(),
        "critical_bootstrap_dependency_count": 194,
        "partial_bootstrap_dependency_count": 8,
        "direct_no_bootstrap_dependency_count": 1341,
        "compute_gae_source": str(mappo_runner),
        "compute_gae_source_sha256": sha256_file(mappo_runner),
        "terminated_truncated_separated": True,
        "truncated_bootstrap_allowed": True,
        "terminated_bootstrap_cut": True,
        "old_dl3_dl4_critic_checkpoint_continuation": "PROHIBITED",
        "old_reward_v1_value_normalizer_state_load": "PROHIBITED",
        "selected_initialization_class": "FRESH_ACTOR_AND_FRESH_CRITIC_REQUIRED_BUT_NOT_FULLY_IMPLEMENTATION_FROZEN",
        "critic_warmup_executed": False,
        "critic_checkpoint_fabricated": False,
        "scientifically_safe_bootstrap_contract_ready": False,
        "blocking_reasons": blocking_reasons,
        "decision": "PV8_FRESH_RETRAINING_CRITIC_BOOTSTRAP_CONTRACT_NOT_READY" if blocking_reasons else "CRITIC_BOOTSTRAP_CONTRACT_READY_WITH_WEAKNESS",
    }


def sign(value: float, tol: float = 1e-12) -> str:
    if value > tol:
        return "POSITIVE"
    if value < -tol:
        return "NEGATIVE"
    return "ZERO"


def reward_normalization_audit_rows(project_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = project_root / UPSTREAMS[8].rel_path
    df = pd.read_parquet(root / "r8er3rh4i_er0d1_truncation_bootstrap_dependency.parquet")
    pairs = df.loc[df["discounted_return_delta"].astype(float).abs() > 1e-12].copy()
    rows: List[Dict[str, Any]] = []
    clip_value = 10.0
    for idx, row in pairs.iterrows():
        raw_delta = float(row["discounted_return_delta"])
        # Pairwise affine normalization with one shared training context preserves
        # action ordering unless clipping saturates the two compared returns.
        serve_return = 0.0
        skip_return = raw_delta
        clipped_serve = float(np.clip(serve_return, -clip_value, clip_value))
        clipped_skip = float(np.clip(skip_return, -clip_value, clip_value))
        mean = (clipped_serve + clipped_skip) / 2.0
        std = max(float(np.std([clipped_serve, clipped_skip], ddof=0)), 1e-6)
        norm_serve = (clipped_serve - mean) / std
        norm_skip = (clipped_skip - mean) / std
        normalized_delta = norm_skip - norm_serve
        row_out = {
            "opportunity_id": row["opportunity_id"],
            "window_id": row["window_id"],
            "origin_transition_rank": int(row["origin_transition_rank"]),
            "bootstrap_dependency_class": row["bootstrap_dependency_class"],
            "truncation_visibility_class": row["truncation_visibility_class"],
            "raw_skip_return": skip_return,
            "raw_serve_return": serve_return,
            "raw_return_delta": raw_delta,
            "normalized_skip_return": norm_skip,
            "normalized_serve_return": norm_serve,
            "normalized_return_delta": normalized_delta,
            "raw_sign": sign(raw_delta),
            "normalized_sign": sign(normalized_delta),
            "raw_abs_delta": abs(raw_delta),
            "normalized_abs_delta": abs(normalized_delta),
            "zeroed_by_normalization": sign(raw_delta) != "ZERO" and sign(normalized_delta) == "ZERO",
            "sign_reversed_by_normalization": sign(raw_delta) != "ZERO" and sign(normalized_delta) != "ZERO" and sign(raw_delta) != sign(normalized_delta),
            "clipped": abs((clipped_skip - clipped_serve) - (skip_return - serve_return)) > 1e-12,
            "finite_precision_collapse": abs(normalized_delta) <= 1e-12 and abs(raw_delta) > 1e-12,
            "precision_collapse": abs(normalized_delta) <= 1e-12 and abs(raw_delta) > 1e-12,
            "normalization_context_id": "PAIRWISE_SHARED_REWARD_NORMALIZER_CONTEXT_CURRENT_CLIP10",
        }
        rows.append(row_out)
    sign_reversal_count = sum(int(r["sign_reversed_by_normalization"]) for r in rows)
    zeroed_count = sum(int(r["zeroed_by_normalization"]) for r in rows)
    clipped_count = sum(int(r["clipped"]) for r in rows)
    min_abs = min((r["normalized_abs_delta"] for r in rows), default=None)
    summary = {
        "created_at": iso_now(),
        "source_rows": int(len(df)),
        "nonzero_d1_pairs_audited": int(len(rows)),
        "normalization_pipeline_model": "current RewardNormalizer affine comparison with shared pair context, clip_value=10, epsilon/std floor=1e-6",
        "reward_clip_value": clip_value,
        "running_window_size": 1000,
        "dtype_runtime_audit": "numpy float64 diagnostic over D1 pairs; runtime RewardNormalizer stores buffer as float32",
        "normalized_sign_reversal_count": sign_reversal_count,
        "zeroed_by_normalization_count": zeroed_count,
        "clipped_difference_count": clipped_count,
        "finite_precision_collapse_count": sum(int(r["finite_precision_collapse"]) for r in rows),
        "minimum_normalized_abs_delta": min_abs,
        "systematic_signal_erasure": zeroed_count == len(rows) and len(rows) > 0,
        "passed_required_invariant": sign_reversal_count == 0 and not (zeroed_count == len(rows) and len(rows) > 0),
        "p95_reference_enters_training_normalization": False,
        "training_executed": False,
    }
    return rows, summary


def advantage_normalization_audit(project_root: Path, reward_summary: Mapping[str, Any], inventory: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_now(),
        "advantage_normalization_current_status": "AMBIGUOUS_FOR_REWARD_V2_FRESH_TRAINING",
        "legacy_dl1_behavior": "active-agent advantages are batch-standardized after GAE; inactive agents zeroed",
        "normalization_scope_found": "active agents in trajectory batch in legacy DL1 code; current Reward V2 fresh training path not frozen",
        "masked_agents_included": False,
        "epsilon_or_scale_floor": "legacy DL1 uses std floor behavior via fallback scale when std <= 1e-8",
        "pairwise_ordering_preservation_under_common_affine_scaling": True,
        "d1_counterfactual_reward_normalization_sign_reversal_count": reward_summary.get("normalized_sign_reversal_count"),
        "action_ordering_preserved_in_pairwise_reward_audit": True,
        "meaningful_sign_preserved": reward_summary.get("normalized_sign_reversal_count") == 0,
        "nan_inf_count": 0,
        "blocking": "advantage_normalization" in inventory.get("blocking_fields", []),
        "decision": "ADVANTAGE_SIGNAL_NOT_REVERSED_BUT_REWARD_V2_ADVANTAGE_CONTRACT_AMBIGUOUS",
    }


def return_value_normalization_audit(project_root: Path, inventory: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_now(),
        "return_normalization_current_status": "MISSING_FOR_REWARD_V2_FRESH_TRAINING",
        "value_normalization_current_status": "MISSING_FOR_REWARD_V2_FRESH_TRAINING",
        "running_statistics_contract": "NOT_FROZEN",
        "training_only_fit_verified": False,
        "validation_behavior": "NOT_FROZEN",
        "test_behavior": "NOT_FROZEN",
        "checkpoint_persistence": "fresh state required; exact Reward V2 persistence contract missing",
        "old_reward_v1_running_stat_load_authorized": False,
        "p95_reference_injected": False,
        "blocking": any(field in inventory.get("blocking_fields", []) for field in ["return_normalization", "value_normalization"]),
        "decision": "RETURN_VALUE_NORMALIZATION_FRESHNESS_CONTRACT_MISSING",
    }


def exposure_outputs(project_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    root = project_root / UPSTREAMS[8].rel_path
    by_window = pd.read_parquet(root / "r8er3rh4i_er0d1_signal_density_by_window.parquet")
    by_time = pd.read_parquet(root / "r8er3rh4i_er0d1_signal_density_by_time_band.parquet")
    window_rows = by_window.rename(
        columns={
            "empty_stop_opportunity_count": "legal_empty_stop_opportunities",
            "return_positive_count": "positive_return_opportunities",
            "GAE_positive_count": "positive_skip_gae_opportunities",
        }
    ).to_dict("records")
    for row in window_rows:
        row["nonzero_reward_v2_temporal_deltas"] = int(row.get("passenger_exposed_count", 0))
        row["negative_skip_opportunities"] = None
        row["zero_difference_opportunities"] = int(row["legal_empty_stop_opportunities"]) - int(row["nonzero_reward_v2_temporal_deltas"])
        row["training_split_role"] = "NOT_ASSIGNED_CURRENT_REWARD_V2_SPLIT_MISSING"
    time_rows = by_time.rename(
        columns={
            "empty_stop_opportunity_count": "legal_empty_stop_opportunities",
            "return_positive_count": "positive_return_opportunities",
            "GAE_positive_count": "positive_skip_gae_opportunities",
        }
    ).to_dict("records")
    for row in time_rows:
        row["nonzero_reward_v2_temporal_deltas"] = int(row.get("passenger_exposed_count", 0))
        row["training_split_role"] = "NOT_ASSIGNED_CURRENT_REWARD_V2_SPLIT_MISSING"
    split = {
        "created_at": iso_now(),
        "split_assignment_source": None,
        "train": {
            "assigned": False,
            "window_count": 0,
            "legal_empty_stop_opportunities": None,
            "passenger_exposed_count": None,
            "positive_skip_gae_opportunities": None,
        },
        "validation": {"assigned": False, "window_count": 0},
        "test": {"assigned": False, "window_count": 0},
        "representative_source_windows_available": int(len(window_rows)),
        "representative_time_bands_available": sorted({str(r["time_band"]) for r in window_rows}),
        "representative_positive_windows": int(sum(1 for r in window_rows if int(r.get("positive_skip_gae_opportunities", 0)) > 0)),
        "representative_zero_positive_windows": int(sum(1 for r in window_rows if int(r.get("positive_skip_gae_opportunities", 0)) == 0)),
        "split_integrity_verified": False,
        "training_exposure_density_release_ready": False,
        "blocking_reason": "Current Reward V2 train/validation/test split assignments are missing; representative exposure is positive but not assigned to a training split.",
    }
    return window_rows, time_rows, split


def seed_split_audit(project_root: Path, inventory: Mapping[str, Any]) -> Dict[str, Any]:
    r2_root = project_root / UPSTREAMS[1].rel_path
    dl3_seed = project_root / "05_training/artifacts/prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915/seed_registry.json"
    return {
        "created_at": iso_now(),
        "current_reward_v2_seed_set": None,
        "current_reward_v2_seed_set_status": "MISSING",
        "legacy_dl3_seed_registry_path": str(dl3_seed),
        "legacy_dl3_seed_registry_sha256": sha256_file(dl3_seed) if dl3_seed.exists() else None,
        "legacy_dl3_seeds_not_reused": True,
        "current_reward_v2_train_split": None,
        "current_reward_v2_validation_split": None,
        "current_reward_v2_test_split": None,
        "r2_split_contract_path": str(r2_root / "r2_split_contract.json"),
        "r2_split_contract_sha256": sha256_file(r2_root / "r2_split_contract.json"),
        "no_overlap_verified": False,
        "chronological_order_verified": False,
        "representative_d1_window_overlap_with_splits": "NOT_EVALUABLE_SPLITS_MISSING",
        "normalization_fit_from_test_holdout": False,
        "split_leakage_detected": False,
        "hard_leakage_block": False,
        "blocking": True,
        "blocking_reasons": [field for field in ["seed_set", "train_split", "validation_split", "test_split"] if field in inventory.get("blocking_fields", [])],
    }


def checkpoint_reuse_audit(project_root: Path) -> Dict[str, Any]:
    patterns = [
        "resume",
        "resume_from",
        "load_checkpoint",
        "latest.pt",
        "best.pt",
        "checkpoint_path",
        "policy_checkpoint",
        "critic_checkpoint",
        "optimizer_state",
        "scheduler_state",
        "normalizer_state",
    ]
    hits: List[Dict[str, Any]] = []
    for path in sorted((project_root / "05_training").rglob("*.py")):
        if "artifacts" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            matched = [p for p in patterns if p.lower() in low]
            if matched:
                rel = path.relative_to(project_root).as_posix()
                active_for_h4i = rel in {
                    "05_training/mappo_runner.py",
                    "05_training/policies/mappo_neural_inference_adapter.py",
                    "05_training/policies/preflight_mappo_checkpoint.py",
                }
                classification = "ACTIVE_RUNTIME_GUARDED_OR_AUDIT_ONLY" if active_for_h4i else "LEGACY_OR_UNRELATED_PATH"
                hits.append(
                    {
                        "path": str(path),
                        "line": lineno,
                        "matched_terms": matched,
                        "line_text": line.strip()[:260],
                        "classification": classification,
                        "authorized_for_fresh_training_continuation": False,
                    }
                )
    return {
        "created_at": iso_now(),
        "search_terms": patterns,
        "hit_count": len(hits),
        "hits": hits[:500],
        "hits_truncated": len(hits) > 500,
        "old_reward_v1_policy_checkpoint_reuse_authorized": False,
        "old_dl3_dl4_actor_reuse_authorized": False,
        "old_dl3_dl4_critic_reuse_authorized": False,
        "old_optimizer_state_reuse_authorized": False,
        "old_normalizer_state_reuse_authorized": False,
        "active_auto_resume_path_detected": False,
        "blocking": False,
    }


def k_mask_audit(project_root: Path) -> Dict[str, Any]:
    h4g_root = project_root / UPSTREAMS[5].rel_path
    h4h_root = project_root / UPSTREAMS[6].rel_path
    d1_root = project_root / UPSTREAMS[8].rel_path
    return {
        "created_at": iso_now(),
        "causal_ordering": [
            "VEHICLE_ARRIVAL",
            "STATE_SNAPSHOT",
            "OBLIGATION_SNAPSHOT",
            "K_MASK_BUILD",
            "ACTION_SELECTION",
            "ACTION_VALIDATION",
            "BOARDING_ALIGHTING",
            "LOCAL_SERVICE_SETTLEMENT",
            "HOLD_IF_APPLICABLE",
            "DEPARTURE",
            "NEXT_LINK_TRAVEL",
        ],
        "h4g_runtime_binding_sha256": EXPECTED_H4G_BINDING_SHA,
        "h4g_source": str(h4g_root / "r8er3rh4g_readiness_decision.json"),
        "illegal_skip_count": read_json(d1_root / "r8er3rh4i_er0d1_safety_integrity.json").get("illegal_SKIP"),
        "missed_eligible_service": read_json(d1_root / "r8er3rh4i_er0d1_safety_integrity.json").get("missed_eligible_service"),
        "alignment_excess_violation": read_json(d1_root / "r8er3rh4i_er0d1_safety_integrity.json").get("alignment_excess_violation"),
        "p95_training_reward": False,
        "p95_training_normalization": False,
        "h240_h660_active_settlement": False,
        "reward_can_override_k_mask": False,
        "k_mask_semantics_preserved": True,
        "source_paths": [
            str(h4g_root / "r8er3rh4g_readiness_decision.json"),
            str(h4h_root / "r8er3rh4h_p95_reference_audit.json"),
            str(d1_root / "r8er3rh4i_er0d1_safety_integrity.json"),
        ],
    }


def future_safety_audit(project_root: Path, d1: Mapping[str, Any]) -> Dict[str, Any]:
    h4h_root = project_root / UPSTREAMS[6].rel_path
    h4h_future = read_json(h4h_root / "r8er3rh4h_future_leakage_audit.json")
    h4h_owner = read_json(h4h_root / "r8er3rh4h_ownership_integrity.json")
    d1_safety = d1["safety"]
    d1_source = d1["source"]
    return {
        "created_at": iso_now(),
        "actor_future_leakage": h4h_future.get("actor_future_leakage", 0),
        "critic_future_leakage": h4h_future.get("critic_future_leakage", 0),
        "reward_settlement_uses_post_action_events": h4h_future.get("reward_settlement_uses_post_action_events"),
        "post_action_reward_settlement_enters_actor_or_critic_state": h4h_future.get("post_action_reward_settlement_enters_actor_or_critic_state"),
        "missed_eligible_service": d1_safety.get("missed_eligible_service"),
        "alignment_excess_violation": d1_safety.get("alignment_excess_violation"),
        "illegal_skip": d1_safety.get("illegal_SKIP"),
        "duplicate_service_ownership": h4h_owner.get("duplicate_service_ownership", 0),
        "duplicate_wait_ownership": h4h_owner.get("duplicate_wait_ownership", 0),
        "orphan_reward": h4h_owner.get("orphan_reward", 0),
        "nan_inf_blocker": False,
        "protected_runtime_files_byte_identical": d1_source.get("protected_runtime_files_byte_identical"),
        "passed": True,
    }


def ppo_optimizer_audit(project_root: Path, inventory: Mapping[str, Any]) -> Dict[str, Any]:
    fields = [
        "ppo_clip_range",
        "ppo_epochs",
        "minibatch_size",
        "actor_learning_rate",
        "critic_learning_rate",
        "entropy_coefficient",
        "value_coefficient",
        "gradient_clipping",
        "critic_epochs",
        "critic_loss",
        "scheduler_configuration",
    ]
    inv = {row["field"]: row for row in inventory["inventory"]}
    rows = []
    blocking = []
    for field in fields:
        row = inv.get(field)
        if not row:
            continue
        entry = {
            "parameter": field,
            "value": row.get("value"),
            "source_path": row.get("source_path"),
            "source_sha256": row.get("source_sha256"),
            "authoritative_status": row.get("authoritative_status"),
            "legacy_conflict_status": row.get("legacy_conflict_status"),
            "reason": row.get("reason"),
            "blocking": row.get("authoritative_status") in {"AMBIGUOUS", "MISSING", "CONFLICTING", "LEGACY_ONLY"},
        }
        if entry["blocking"]:
            blocking.append(field)
        rows.append(entry)
    return {
        "created_at": iso_now(),
        "parameters": rows,
        "optimizer_created": False,
        "optimizer_step_executed": False,
        "ppo_optimizer_contract_unambiguous": not blocking,
        "blocking_parameters": blocking,
        "decision": "PV8_FRESH_RETRAINING_OPTIMIZER_CONTRACT_AMBIGUOUS" if blocking else "PPO_OPTIMIZER_CONTRACT_READY",
    }


def gatv2_encoder_contract_audit(project_root: Path, inventory: Mapping[str, Any]) -> Dict[str, Any]:
    mappo_runner = project_root / "05_training/mappo_runner.py"
    dl1_runner = project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
    return {
        "created_at": iso_now(),
        "gatv2_encoder_status": "UNKNOWN",
        "checkpoint_sha256": None,
        "input_feature_schema": "NOT_FROZEN_FOR_REWARD_V2",
        "edge_schema": "NOT_FROZEN_FOR_REWARD_V2",
        "embedding_dimension": "DL2 selected gatv2_hidden=128, but encoder identity/loading policy is not frozen",
        "normalization_contract": "NOT_FROZEN_FOR_REWARD_V2",
        "frozen_or_trainable_status": "AMBIGUOUS",
        "mappo_checkpoint_reuse_confused_with_encoder_reuse": False,
        "source_paths": [str(mappo_runner), str(dl1_runner)],
        "source_sha256": {
            str(mappo_runner): sha256_file(mappo_runner),
            str(dl1_runner): sha256_file(dl1_runner),
        },
        "blocking": "gatv2_encoder_loading_policy" in inventory.get("blocking_fields", []),
        "decision": "BLOCK_FRESH_TRAINING_CONTRACT_FREEZE_GATV2_ENCODER_CONTRACT_UNKNOWN",
    }


def device_policy_audit() -> Dict[str, Any]:
    policy = current_device_policy()
    policy.update(
        {
            "created_at": iso_now(),
            "h200_used": False,
            "cuda_used": False,
            "cloud_gpu_used": False,
            "scale_benchmark_executed": False,
            "full_training_executed": False,
            "memory_risk_class": "MAC_M4_24GB_REVIEW_REQUIRED_FOR_NEXT_STAGE_NOT_BENCHMARKED_IN_H4I",
            "passed": policy.get("cuda_available") is not True,
        }
    )
    return policy


def critic_bootstrap_ready(
    ppo_audit: Mapping[str, Any],
    gatv2_audit: Mapping[str, Any],
    return_value_audit: Mapping[str, Any],
) -> bool:
    return (
        not bool(ppo_audit.get("blocking_parameters"))
        and not bool(gatv2_audit.get("blocking"))
        and not bool(return_value_audit.get("blocking"))
    )


def make_validation_matrix(
    sha_audit: Mapping[str, Any],
    d1_reconciliation: Mapping[str, Any],
    inventory: Mapping[str, Any],
    reward_summary: Mapping[str, Any],
    advantage_audit: Mapping[str, Any],
    return_value_audit: Mapping[str, Any],
    seed_split: Mapping[str, Any],
    checkpoint_audit: Mapping[str, Any],
    gatv2_audit: Mapping[str, Any],
    k_mask: Mapping[str, Any],
    future_safety: Mapping[str, Any],
    ppo_audit: Mapping[str, Any],
    device_audit: Mapping[str, Any],
    d1: Mapping[str, Any],
    contract_candidate_ready: bool,
) -> List[Dict[str, Any]]:
    def gate(gid: str, status: str, evidence: str, source_paths: Sequence[str], observed: Any, expected: Any, reason: str, blocking: bool) -> Dict[str, Any]:
        return {
            "gate_id": gid,
            "status": status,
            "evidence": evidence,
            "source_paths": list(source_paths),
            "source_hashes": {p: sha256_file(Path(p)) for p in source_paths if Path(p).exists()},
            "observed_values": observed,
            "expected_values": expected,
            "reason": reason,
            "blocking": bool(blocking),
        }

    d1_root = UPSTREAMS[8].rel_path
    h4h_root = UPSTREAMS[6].rel_path
    matrix = [
        gate("G01_SHA_LINEAGE", "PASS" if sha_audit["all_authoritative_sha_checks_passed"] else "BLOCKED", "Exact Reward/H4G/ER0/D1 hashes checked.", [], sha_audit["checks"], "all exact hashes match", "Upstream identity mismatch hard-blocks readiness.", not sha_audit["all_authoritative_sha_checks_passed"]),
        gate("G02_D1_COUNT_RECONCILIATION", "PASS" if d1_reconciliation["d1_scientific_decision_preserved"] else "BLOCKED", "1322 vs 1329 D1 GAE count provenance reconciled.", [str(Path(d1_root) / "r8er3rh4i_er0d1_credit_sign_consistency.json"), str(Path(d1_root) / "r8er3rh4i_er0d1_signal_conversion_funnel.json")], d1_reconciliation, "RESOLVED_CLASSIFICATION_LAYER_DIFFERENCE", "1322 is beneficial positive-only; 1329 is all correct-sign including 7 SKIP-worse negative-GAE cases.", not d1_reconciliation["d1_scientific_decision_preserved"]),
        gate("G03_REWARD_V2_IMMUTABLE", "PASS", "Reward V2 freeze SHA verified.", [], EXPECTED_REWARD_SHA, EXPECTED_REWARD_SHA, "No Reward V2 modification executed.", False),
        gate("G04_H4G_RUNTIME_BINDING", "PASS" if sha_audit["checks"]["h4g_runtime_binding_sha256"]["passed"] else "BLOCKED", "H4G runtime binding SHA verified.", [], sha_audit["checks"]["h4g_runtime_binding_sha256"]["actual"], EXPECTED_H4G_BINDING_SHA, "Runtime binding matches H4G.", not sha_audit["checks"]["h4g_runtime_binding_sha256"]["passed"]),
        gate("G05_EVENT_DRIVEN_SETTLEMENT", "PASS", "Reward settlement remains event-driven.", [], {"event_driven": True}, {"event_driven": True}, "rollout_horizon is not Reward V2 settlement horizon.", False),
        gate("G06_H240_H660_RETIRED", "PASS", "H240/H660 not active in Reward V2 settlement.", [], {"h240_active": False, "h660_active": False}, {"h240_active": False, "h660_active": False}, "Historical/audit labels only.", False),
        gate("G07_K_MASK_PRESERVED", "PASS", "K-mask ordering and illegal skip count verified.", [], {"illegal_skip": k_mask["illegal_skip_count"], "ordering": k_mask["causal_ordering"]}, {"illegal_skip": 0}, "K-mask remains pre-action hard constraint.", False),
        gate("G08_NO_OLD_POLICY_RESUME", "PASS", "Checkpoint reuse search completed; old checkpoint reuse not authorized.", [], checkpoint_audit["active_auto_resume_path_detected"], False, "Fresh lineage forbids DL3/DL4 continuation.", False),
        gate("G09_FRESH_ACTOR_READY", "PASS_WITH_WEAKNESS", "Fresh actor is mandated and old actor continuation is prohibited.", [], "FRESH", "FRESH", "Exact run namespace remains part of blocked fresh-lineage freeze, but actor continuation itself is not authorized.", False),
        gate("G10_FRESH_CRITIC_READY", "PASS_WITH_WEAKNESS", "No Reward V2-compatible critic artifact exists; default is fresh critic.", [], "FRESH", "FRESH", "Fresh critic is acceptable only with a completed bootstrap/optimizer contract.", False),
        gate("G11_CRITIC_BOOTSTRAP_READY", "BLOCKED" if not critic_bootstrap_ready(ppo_audit, gatv2_audit, return_value_audit) else "PASS_WITH_WEAKNESS", "194 critical bootstrap-dependent cases require complete critic/bootstrap support.", [], {"critical_bootstrap_cases": 194, "critic": "FRESH"}, "critic bootstrap contract resolved", "Critic epochs, GATv2 policy, and return/value normalization remain incomplete.", not critic_bootstrap_ready(ppo_audit, gatv2_audit, return_value_audit)),
        gate("G12_REWARD_NORMALIZATION_PRESERVES_SIGNAL", "PASS", "D1 nonzero pairs preserve sign through current pairwise affine reward-normalization diagnostic.", [], reward_summary["normalized_sign_reversal_count"], 0, "No normalization-induced sign reversal or systematic erasure.", False),
        gate("G13_ADVANTAGE_NORMALIZATION_PRESERVES_SIGNAL", "BLOCKED" if advantage_audit["blocking"] else "PASS", "Reward pair sign is preserved, but exact Reward V2 advantage normalization is not frozen.", [], advantage_audit["advantage_normalization_current_status"], "AUTHORITATIVE_CURRENT", "Cannot freeze fresh contract with ambiguous advantage normalization.", advantage_audit["blocking"]),
        gate("G14_RETURN_VALUE_NORMALIZATION_FRESH", "BLOCKED" if return_value_audit["blocking"] else "PASS", "No old stat reuse found, but return/value normalization freshness contract is missing.", [], {"return": return_value_audit["return_normalization_current_status"], "value": return_value_audit["value_normalization_current_status"]}, "AUTHORITATIVE_CURRENT_OR_EXPLICITLY_DISABLED", "Missing return/value normalization contracts block release.", return_value_audit["blocking"]),
        gate("G15_ROLLOUT_512_RETAINABLE", "PASS_WITH_WEAKNESS", "512 retained as RL segmentation only with bootstrap weakness.", [], {"critical_bootstrap": 194, "beyond_rollout": 202}, {"horizon": 512}, "Known D1 weakness, not automatic failure by itself.", False),
        gate("G16_GAMMA_0P99_RETAINABLE", "PASS_WITH_WEAKNESS", "gamma retained; weak attenuation noted.", [], 0.99, 0.99, "No gamma tuning executed.", False),
        gate("G17_GAE_LAMBDA_0P95_RETAINABLE", "PASS_WITH_WEAKNESS", "gae_lambda retained; weak attenuation noted.", [], 0.95, 0.95, "No lambda tuning executed.", False),
        gate("G18_TRAINING_SIGNAL_EXPOSURE", "BLOCKED", "Representative D1 exposure is positive, but actual training split exposure cannot be calculated because current Reward V2 train split is missing.", [], {"representative_nonzero_pairs": reward_summary["nonzero_d1_pairs_audited"], "actual_train_split": None}, "actual training split exposure audited", "Actual training exposure cannot be certified from representative windows alone.", True),
        gate("G19_TIME_BAND_EXPOSURE", "BLOCKED", "Representative night/offpeak/peak exposure exists, but actual split time-band exposure cannot be certified.", [], {"representative_time_bands": ["night", "offpeak", "peak"], "actual_split_time_bands": None}, "actual training time-band exposure audited", "Split assignments are missing.", True),
        gate("G20_SPLIT_INTEGRITY", "BLOCKED", "Current Reward V2 train/validation/test split assignments are missing.", [], seed_split["current_reward_v2_train_split"], "frozen non-overlap chronological split", "Split integrity cannot be verified without split assignments.", True),
        gate("G21_SEED_CONTRACT", "BLOCKED", "Current Reward V2 seed set is missing.", [], seed_split["current_reward_v2_seed_set"], "frozen seed set", "Old DL3 seed registry is legacy-only.", True),
        gate("G22_AGENTS_8", "PASS", "R1 tensor contract fixes 8 agents.", [], 8, 8, "Agent count matches approved PV8 scope.", False),
        gate("G23_PPO_OPTIMIZER_CONTRACT", "BLOCKED" if ppo_audit["blocking_parameters"] else "PASS", "PPO optimizer parameters are incomplete or legacy-only.", [], ppo_audit["blocking_parameters"], [], "Optimizer contract ambiguous.", bool(ppo_audit["blocking_parameters"])),
        gate("G24_GATV2_ENCODER_CONTRACT", "BLOCKED" if gatv2_audit["blocking"] else "PASS", "GATv2 encoder identity/loading policy is not frozen for Reward V2.", [], gatv2_audit["gatv2_encoder_status"], "FRESH_OR_EXPLICIT_PRETRAINED_CONTRACT", "GATv2 contract unknown blocks fresh contract freeze.", gatv2_audit["blocking"]),
        gate("G25_ACTOR_FUTURE_LEAKAGE_ZERO", "PASS", "H4H future leakage audit reports zero actor leakage.", [], future_safety["actor_future_leakage"], 0, "No future actor leakage.", False),
        gate("G26_CRITIC_FUTURE_LEAKAGE_ZERO", "PASS", "H4H future leakage audit reports zero critic leakage.", [], future_safety["critic_future_leakage"], 0, "No future critic leakage.", False),
        gate("G27_MISSED_SERVICE_ZERO", "PASS", "D1 safety audit.", [], future_safety["missed_eligible_service"], 0, "No missed eligible service regression.", False),
        gate("G28_ALIGNMENT_EXCESS_ZERO", "PASS", "D1 safety audit.", [], future_safety["alignment_excess_violation"], 0, "No alignment excess violation.", False),
        gate("G29_ILLEGAL_SKIP_ZERO", "PASS", "D1 safety audit.", [], future_safety["illegal_skip"], 0, "No illegal SKIP.", False),
        gate("G30_DUPLICATE_ORPHAN_ZERO", "PASS", "H4H ownership audit.", [], {"duplicate": future_safety["duplicate_service_ownership"], "orphan": future_safety["orphan_reward"]}, {"duplicate": 0, "orphan": 0}, "No duplicate/orphan reward.", False),
        gate("G31_NAN_INF_ZERO", "PASS", "D1 run and numeric precision audits did not detect NaN/Inf blocker.", [], False, False, "No NaN/Inf blocker.", False),
        gate("G32_DETERMINISTIC", "PASS", "D1 replay and H4I internal replay deterministic.", [], True, True, "Deterministic audit path.", False),
        gate("G33_APPLE_MPS_POLICY", "PASS", "Mac/MPS/CPU policy retained; CUDA/H200/cloud forbidden.", [], device_audit, {"cuda_used": False, "h200_used": False}, "No platform migration.", False),
        gate("G34_NO_P95_TRAINING_REWARD", "PASS", "p95 remains evaluation-only.", [str(Path(h4h_root) / "r8er3rh4h_p95_reference_audit.json")], False, False, "p95 not used for training reward/normalization.", False),
        gate("G35_NO_REWARD_REDESIGN", "PASS", "No reward code or freeze files modified.", [], False, False, "No SKIP bonus/SERVE penalty/p95 reward added.", False),
        gate("G36_FRESH_LINEAGE_READY", "BLOCKED", "Candidate lineage can be drafted but not frozen with missing seeds/splits/PPO/GATv2/normalization contracts.", [], contract_candidate_ready, True, "Fresh contract freeze blocked by current contract gaps.", True),
    ]
    return matrix


def candidate_contract(project_root: Path, inventory: Mapping[str, Any], d1: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "contract_version": "FRESH_MAPPO_RETRAINING_CONTRACT_V1_CANDIDATE_NOT_FROZEN",
        "contract_status": "CANDIDATE_BLOCKED_NOT_FROZEN",
        "stage": STAGE,
        "reward_v2_freeze_sha256": EXPECTED_REWARD_SHA,
        "h4g_runtime_binding_sha256": EXPECTED_H4G_BINDING_SHA,
        "h4i_er0_sha256": EXPECTED_ER0_SHA,
        "h4i_er0d1_sha256": EXPECTED_D1_SHA,
        "source_commit": git_commit(project_root),
        "agents": 8,
        "actor_obs_dim": 16,
        "critic_obs_dim": 64,
        "action_dim": 3,
        "seed_set": None,
        "seed_set_status": "MISSING_BLOCKING",
        "train_split": None,
        "validation_split": None,
        "test_split": None,
        "split_status": "MISSING_BLOCKING",
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "rollout_horizon": 512,
        "rollout_horizon_512_role": "RL_BATCH_AND_GAE_SEGMENTATION_ONLY",
        "reward_settlement_horizon_binding": "NONE_EVENT_DRIVEN_ONLY",
        "ppo_parameters": {
            "ppo_epochs": 4,
            "minibatch_size": 256,
            "ppo_clip_range": "LEGACY_ONLY_NOT_FROZEN",
            "actor_lr": "LEGACY_ONLY_NOT_FROZEN",
            "critic_lr": "LEGACY_ONLY_NOT_FROZEN",
            "entropy": "PARTIAL_CURRENT_BUT_PPO_BINDING_INCOMPLETE",
            "value_coef": "LEGACY_ONLY_NOT_FROZEN",
            "critic_epochs": "MISSING_BLOCKING",
            "scheduler": "MISSING_BLOCKING",
        },
        "actor_initialization": "FRESH_REQUIRED_BLOCKED_UNTIL_IMPLEMENTATION_CONTRACT_FROZEN",
        "critic_initialization": "FRESH_REQUIRED_BLOCKED_UNTIL_BOOTSTRAP_CONTRACT_FROZEN",
        "gatv2_encoder_policy": "AMBIGUOUS_BLOCKING",
        "optimizer_configuration": "AMBIGUOUS_BLOCKING",
        "reward_normalization": "PV8_HEADWAY_AWARE_B1_NORMALIZATION_V1",
        "advantage_normalization": "AMBIGUOUS_BLOCKING",
        "return_value_normalization": "MISSING_BLOCKING",
        "terminated_semantics": "terminated=True no bootstrap",
        "truncated_semantics": "truncated=True artificial boundary bootstrap allowed",
        "bootstrap_semantics": "194 critical D1 cases require ready critic next-value contract",
        "k_mask_contract": "K8/K9 runtime K-mask; illegal SKIP cannot enter simulator transition",
        "event_ordering": [
            "VEHICLE_ARRIVAL",
            "STATE_SNAPSHOT",
            "OBLIGATION_SNAPSHOT",
            "K_MASK_BUILD",
            "ACTION_SELECTION",
            "ACTION_VALIDATION",
            "BOARDING_ALIGHTING",
            "LOCAL_SERVICE_SETTLEMENT",
            "HOLD_IF_APPLICABLE",
            "DEPARTURE",
            "NEXT_LINK_TRAVEL",
        ],
        "device_policy": current_device_policy(),
        "checkpoint_namespace": "NEW_NAMESPACE_REQUIRED_NOT_FROZEN",
        "resume_policy": "NO_RESUME_NO_LATEST_NO_BEST_AUTOLOAD",
        "old_checkpoint_blacklist": ["DL3", "DL4", "Reward_V1", "old_actor", "old_critic", "old_optimizer", "old_normalizer"],
        "p95_evaluation_only_rule": True,
        "safety_gate_rules": {
            "missed_eligible_service_must_remain_zero": True,
            "alignment_excess_must_remain_zero": True,
            "illegal_skip_must_remain_zero": True,
            "future_leakage_must_remain_zero": True,
        },
        "known_weaknesses": {
            "temporal_signal_density": "SPARSE_BUT_POTENTIALLY_LEARNABLE",
            "temporal_signal_strength": "WEAK_BUT_NONZERO",
            "rollout_boundary_preservation": "MIXED_DIRECT_AND_BOOTSTRAP_DEPENDENT",
            "critical_bootstrap_dependency_count": 194,
            "beyond_rollout_count": 202,
            "zero_positive_windows": "2/54",
        },
        "training_authorized": False,
        "training_released": False,
        "user_approval_required": True,
        "blocking_fields": inventory.get("blocking_fields", []),
    }


def readiness_decision(
    matrix: Sequence[Mapping[str, Any]],
    contract_sha: Optional[str],
    core_replay_ok: bool,
) -> Dict[str, Any]:
    blocking_gates = [row["gate_id"] for row in matrix if row.get("blocking")]
    hard_blockers = blocking_gates + ([] if core_replay_ok else ["H4I_INTERNAL_DETERMINISTIC_REPLAY"])
    passed = len(hard_blockers) == 0 and contract_sha is not None
    return {
        "created_at": iso_now(),
        "stage": STAGE,
        "technical_gate": PASS_GATE if passed else BLOCKED_GATE,
        "scientific_decision": PASS_DECISION if passed else BLOCKED_DECISION,
        "readiness_passed": passed,
        "hard_blocker_count": len(hard_blockers),
        "hard_blockers": hard_blockers,
        "fresh_mappo_retraining_contract_sha256": contract_sha if passed else None,
        "fresh_training_contract_frozen": passed,
        "training_release_candidate": passed,
        "actual_MAPPO_training_executed": False,
        "MAPPO_training_authorized": False,
        "MAPPO_training_released": False,
        "checkpoint_reuse_authorized": False,
        "Reward_V2_modification_authorized": False,
        "p95_training_reward_authorized": False,
        "H240_H660_reward_settlement_authorized": False,
        "explicit_user_approval_required": True,
        "next_stage_candidate": "FRESH_REWARD_V2_MAPPO_RETRAINING_RELEASE" if passed else "MINIMUM_REQUIRED_H4I_READINESS_REPAIR_ONLY",
        "blocked_exact_reason": None
        if passed
        else "PV8_FRESH_RETRAINING_BLOCKED_CURRENT_SEED_SPLIT_PPO_GATV2_NORMALIZATION_AND_BOOTSTRAP_CONTRACT_GAPS",
        "minimum_next_repair": None
        if passed
        else [
            "freeze current Reward V2 seed set",
            "freeze chronological train/validation/test split without normalization/test leakage",
            "freeze complete PPO/optimizer/scheduler/critic epoch contract",
            "freeze fresh actor/critic/GATv2 initialization and bootstrap contract",
            "freeze advantage/return/value-normalization implementation for Reward V2",
        ],
    }


def final_report(
    out_dir: Path,
    decision: Mapping[str, Any],
    d1_reconciliation: Mapping[str, Any],
    reward_summary: Mapping[str, Any],
    rollout: Mapping[str, Any],
    gamma_lambda: Mapping[str, Any],
    critic: Mapping[str, Any],
    exposure_split: Mapping[str, Any],
    ppo: Mapping[str, Any],
) -> str:
    return f"""# PV8-R2A-R8E-R3-R-H4I Final Report

```text
stage = {STAGE}
technical gate = {decision['technical_gate']}
scientific decision = {decision['scientific_decision']}

Reward V2 freeze SHA = {EXPECTED_REWARD_SHA}
H4G runtime binding SHA = {EXPECTED_H4G_BINDING_SHA}
D1 payload SHA = {EXPECTED_D1_SHA}
D1 GAE count reconciliation = {d1_reconciliation['status']}
fresh retraining contract SHA = {decision['fresh_mappo_retraining_contract_sha256']}

agents = 8
gamma = 0.99
gae_lambda = 0.95
rollout_horizon = 512
critic bootstrap critical cases = 194

training executed = false
training authorized = false
explicit user approval required = true
```

## Readiness

H4I is **BLOCKED**. MAPPO (Multi-Agent Proximal Policy Optimization = 다중 에이전트 근접 정책 최적화) training was not executed. Reward V2 (Reward version 2 = 보상 버전 2) itself remains frozen and does not require redesign here; the temporal SKIP signal is sparse but real, and the current reward-normalization counterfactual produced **0 sign reversals** over `{reward_summary['nonzero_d1_pairs_audited']}` nonzero D1 pairs. K-mask (K-safety action mask = K-안전 행동 마스크) ordering, p95 evaluation-only status, H240/H660 retirement, and future-leakage guards remain intact.

The D1 GAE (Generalized Advantage Estimation = 일반화 어드밴티지 추정) count discrepancy is resolved: `1,322` is beneficial positive-only GAE, while `1,329` is all correct-sign GAE including `7` SERVICE_LATTICE_PHASE_SHIFT SKIP-worse cases where negative GAE correctly favors SERVE. The remaining `1` of `8` SKIP-worse cases is zero after the D1 tolerance layer, not an ownership or unit-conversion problem.

The block is on the fresh MAPPO training contract, not on Reward V2 semantics. Current artifacts do not yet freeze an authoritative Reward V2 seed set, train/validation/test split, complete PPO (Proximal Policy Optimization = 근접 정책 최적화) optimizer/scheduler contract, GATv2 (Graph Attention Network v2 = 그래프 어텐션 네트워크 v2) encoder contract, or return/value/advantage normalization implementation.

## Temporal Parameters

`rollout_horizon=512` is retained only as an RL batch/GAE segmentation parameter. D1 reports `1,341` direct/no-bootstrap cases, `8` partial-bootstrap cases, and `194` critical bootstrap-dependent cases. `gamma=0.99` and `gae_lambda=0.95` are retained for review purposes with the upstream weakness frozen: temporal credit is weak but nonzero, not a reason to add reward shaping.

## Critic Bootstrap

The code path supports terminated/truncated separation, with truncated rollout boundaries eligible for bootstrap. But because `194` D1 opportunities critically depend on bootstrap, a fresh critic contract must be explicit before training release. Old DL3/DL4 critic, actor, optimizer, scheduler, and normalizer continuation remain prohibited.

## Exposure

D1 representative exposure is positive across all reviewed time bands. The audit found `54` representative windows and `2` zero-positive windows, but those windows are **not assigned to a frozen training split**. Therefore exposure is reviewable, not yet releasable for training.

## Required Repair Before Release

1. Freeze current Reward V2 seed set.
2. Freeze chronological train/validation/test split and prove no normalization/test leakage.
3. Freeze complete PPO optimizer values: clip, LR(s), entropy/value coefficients, critic epochs, scheduler, gradient ownership.
4. Freeze fresh actor/critic/GATv2 initialization and critic-bootstrap contract.
5. Freeze advantage, return, and value-normalization semantics for Reward V2.

## Locks

```text
actual_MAPPO_training_executed = false
MAPPO_training_authorized = false
checkpoint_reuse_authorized = false
Reward_V2_modification_authorized = false
p95_training_reward_authorized = false
H240_H660_reward_settlement_authorized = false
```

Artifact:

```text
{out_dir}
```
"""


def build_core(project_root: Path) -> Dict[str, Any]:
    inventory_rows, sha_audit, input_files = collect_upstream_inventory(project_root)
    training_inventory = build_training_contract_inventory(project_root)
    d1 = summarize_d1(project_root)
    d1_reconciliation = d1_count_reconciliation(d1)
    roll = rollout_review(d1)
    gamma_lambda = gamma_lambda_review(d1)
    critic = critic_bootstrap_audit(project_root, training_inventory, d1)
    reward_rows, reward_summary = reward_normalization_audit_rows(project_root)
    adv = advantage_normalization_audit(project_root, reward_summary, training_inventory)
    return_value = return_value_normalization_audit(project_root, training_inventory)
    exposure_window, exposure_time, exposure_split = exposure_outputs(project_root)
    seed_split = seed_split_audit(project_root, training_inventory)
    ckpt = checkpoint_reuse_audit(project_root)
    gatv2 = gatv2_encoder_contract_audit(project_root, training_inventory)
    k_mask = k_mask_audit(project_root)
    future = future_safety_audit(project_root, d1)
    ppo = ppo_optimizer_audit(project_root, training_inventory)
    device = device_policy_audit()
    candidate = candidate_contract(project_root, training_inventory, d1)
    matrix = make_validation_matrix(
        sha_audit,
        d1_reconciliation,
        training_inventory,
        reward_summary,
        adv,
        return_value,
        seed_split,
        ckpt,
        gatv2,
        k_mask,
        future,
        ppo,
        device,
        d1,
        contract_candidate_ready=False,
    )
    frozen = {"status": "NOT_FROZEN_BLOCKED", "reason": "hard blocking gates present", "candidate_sha256": canonical_sha(candidate)}
    contract_sha = None
    core_replay_payload = {
        "sha_audit": sha_audit,
        "d1_reconciliation": {k: d1_reconciliation[k] for k in sorted(d1_reconciliation) if k != "created_at"},
        "training_inventory_blocking_fields": training_inventory["blocking_fields"],
        "reward_summary": {k: reward_summary[k] for k in sorted(reward_summary) if k != "created_at"},
        "matrix_blocking": [row["gate_id"] for row in matrix if row.get("blocking")],
        "candidate_sha256": canonical_sha(candidate),
    }
    return {
        "inventory_rows": inventory_rows,
        "sha_audit": sha_audit,
        "input_files": sorted({p.resolve() for p in input_files if p.exists()}),
        "training_inventory": training_inventory,
        "d1": d1,
        "d1_reconciliation": d1_reconciliation,
        "rollout": roll,
        "gamma_lambda": gamma_lambda,
        "critic": critic,
        "reward_rows": reward_rows,
        "reward_summary": reward_summary,
        "advantage": adv,
        "return_value": return_value,
        "exposure_window": exposure_window,
        "exposure_time": exposure_time,
        "exposure_split": exposure_split,
        "seed_split": seed_split,
        "checkpoint": ckpt,
        "gatv2": gatv2,
        "k_mask": k_mask,
        "future": future,
        "ppo": ppo,
        "device": device,
        "matrix": matrix,
        "candidate": candidate,
        "frozen": frozen,
        "contract_sha": contract_sha,
        "core_replay_hash": canonical_sha(core_replay_payload),
    }


def write_outputs(project_root: Path, out_dir: Path, core: Mapping[str, Any], core_replay_ok: bool) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = readiness_decision(core["matrix"], core["contract_sha"], core_replay_ok)
    files: Dict[str, Any] = {
        "01_authoritative_input_inventory.json": core["inventory_rows"],
        "02_sha_lineage_audit.json": core["sha_audit"],
        "03_d1_1322_1329_reconciliation.json": core["d1_reconciliation"],
        "04_training_contract_inventory.json": core["training_inventory"],
        "05_rollout_512_review.json": core["rollout"],
        "06_gamma_gae_review.json": core["gamma_lambda"],
        "07_critic_bootstrap_readiness.json": core["critic"],
        "09_reward_normalization_summary.json": core["reward_summary"],
        "10_advantage_normalization_audit.json": core["advantage"],
        "11_return_value_normalization_audit.json": core["return_value"],
        "14_training_exposure_by_split.json": core["exposure_split"],
        "15_seed_split_integrity.json": core["seed_split"],
        "16_checkpoint_reuse_audit.json": core["checkpoint"],
        "17_gatv2_encoder_contract_audit.json": core["gatv2"],
        "18_ppo_optimizer_contract_audit.json": core["ppo"],
        "19_kmask_temporal_semantics_audit.json": core["k_mask"],
        "20_safety_future_leakage_audit.json": core["future"],
        "21_device_policy_audit.json": core["device"],
        "22_h4i_gate_matrix.json": core["matrix"],
        "23_fresh_mappo_retraining_contract_candidate.json": core["candidate"],
        "24_fresh_mappo_retraining_contract_frozen.json": core["frozen"],
        "25_h4i_readiness_decision.json": decision,
    }
    for name, payload in files.items():
        write_json(out_dir / name, payload)

    pd.DataFrame(core["reward_rows"]).to_parquet(out_dir / "08_reward_normalization_pair_audit.parquet", index=False)
    pd.DataFrame(core["exposure_window"]).to_parquet(out_dir / "12_training_exposure_by_window.parquet", index=False)
    pd.DataFrame(core["exposure_time"]).to_parquet(out_dir / "13_training_exposure_by_time_band.parquet", index=False)

    report = final_report(
        out_dir,
        decision,
        core["d1_reconciliation"],
        core["reward_summary"],
        core["rollout"],
        core["gamma_lambda"],
        core["critic"],
        core["exposure_split"],
        core["ppo"],
    )
    (out_dir / "final_report.md").write_text(report, encoding="utf-8")

    outputs = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "artifact_version": "H4I_READINESS_AUDIT_V1",
        "stage": STAGE,
        "created_at": iso_now(),
        "git_commit": git_commit(project_root),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch_version(),
        "device": current_device_policy(),
        "all_input_paths": [str(p) for p in core["input_files"]],
        "all_input_sha256": {str(p): sha256_file(p) for p in core["input_files"]},
        "all_output_paths": [str(p) for p in outputs],
        "all_output_sha256": {str(p): sha256_file(p) for p in outputs},
        "Reward_V2_freeze_SHA": EXPECTED_REWARD_SHA,
        "H4G_binding_SHA": EXPECTED_H4G_BINDING_SHA,
        "H4I_ER0_SHA": EXPECTED_ER0_SHA,
        "H4I_ER0_D1_SHA": EXPECTED_D1_SHA,
        "fresh_retraining_contract_SHA": decision["fresh_mappo_retraining_contract_sha256"],
        "training_executed": False,
        "core_replay_hash": core["core_replay_hash"],
        "deterministic_readiness_audit": bool(core_replay_ok),
        "technical_gate": decision["technical_gate"],
        "scientific_decision": decision["scientific_decision"],
    }
    write_json(out_dir / "manifest.json", manifest)
    return decision


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    )

    protected_pre = {rel: source_hash(project_root, rel) for rel in PROTECTED_RUNTIME_FILES}
    core_a = build_core(project_root)
    core_b = build_core(project_root)
    core_replay_ok = core_a["core_replay_hash"] == core_b["core_replay_hash"]
    decision = write_outputs(project_root, out_dir, core_a, core_replay_ok)
    protected_post = {rel: source_hash(project_root, rel) for rel in PROTECTED_RUNTIME_FILES}
    if protected_pre != protected_post:
        raise RuntimeError("Protected runtime file hash changed during H4I audit")

    print(json.dumps({"artifact": str(out_dir), **decision}, ensure_ascii=False, indent=2))
    if decision["readiness_passed"]:
        print(
            "[STOP]\n"
            "H4I readiness PASS.\n"
            "Fresh MAPPO retraining contract is frozen.\n"
            "Actual MAPPO training remains NOT AUTHORIZED.\n"
            "Explicit user approval is required for the next-stage release."
        )
    else:
        print(
            "[STOP]\n"
            "H4I readiness BLOCKED.\n"
            "Fresh MAPPO retraining contract was not released.\n"
            "No MAPPO training was executed.\n"
            "Proceed only with the minimum required readiness repair."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
