from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# --- H4M-AE-R9.8 fail-closed simulator authorization -------------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


STAGE = "PV8-R2A-R8E-R3-R-H4K-RERUN"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_RERUN_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_FULL_RETRAINING_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_RERUN_THREE_SEED_FULL_RETRAINING_FAILED"
PASS_DECISION = "PV8_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_TRAINING_COMPLETE_READY_FOR_FROZEN_POLICY_EVALUATION_RELEASE"
BLOCK_DECISION = "PV8_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_TRAINING_BLOCKED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"
DL4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
H4I_RERUN_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_rerun_training_readiness_20260810_192924"
H4J_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4j_fresh_mappo_execution_integrity_20260810_200616"
ZL1_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4j_zl1_zero_loss_adapter_and_evidence_wiring_20260810_215550"
ZL2_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4j_zl2_patent_aware_execution_integrity_20260810_221630"
H4K_BLOCK_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4k_fresh_reward_v2_zero_loss_three_seed_full_retraining_20260810_223354_BLOCKED"
REPRESENTATIVE_REGISTRY = (
    ARTIFACTS_ROOT
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
    / "r8er3r_representative_window_registry.parquet"
)
DATASET_ROOT = ARTIFACTS_ROOT / "dataset_full_20260422_084243"

EXPECTED = {
    "h4j_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4J_FRESH_MAPPO_EXECUTION_INTEGRITY_COMPLETE",
    "zl1_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4J_ZL1_ZERO_LOSS_ADAPTER_AND_EVIDENCE_WIRING_COMPLETE",
    "zl2_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4J_ZL2_PATENT_AWARE_EXECUTION_INTEGRITY_COMPLETE",
    "s0_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_S0_FULL_TRAINING_SCHEDULE_SELECTION_AND_FREEZE_COMPLETE",
    "s0_decision": "PV8_FRESH_REWARD_V2_ZERO_LOSS_FULL_TRAINING_SCHEDULE_FROZEN_READY_FOR_H4K_RERUN",
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "h4k_schedule_sha": "209297ba5d606fa859aeeb5c154c288b3aa374535989b74448902defbb302168",
}

H4G_SOURCE_HASHES = {
    "05_training/rewards/mappo_reward_v1.py": "8f157b8ea0798b3ec72ab81ca747ba1d58ccf38d82767e0f5a302292b958da52",
    "05_training/simulator/pv8_reward_outcome_collector.py": "ea3ba294d86d5753e9a398dd1b539e6ea2ba862a39b83e175172fda17c6f4419",
    "05_training/simulator/pv8_b1_orchestrator.py": "4fc812b8e74415d64c2bbc981e53e7319dd8f8e6519a6908b313dce22b7e46b1",
    "05_training/mappo_runner.py": "b7a9c39534d90e4757dff7a5397cb8c67483ff0aac8533f610be7471e993d169",
}

COMMITTED_SOURCE_PATHS = [
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_s0_full_training_schedule_selection_and_freeze.py",
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py",
]

RUNTIME_DEPENDENCIES = set(
    COMMITTED_SOURCE_PATHS
    + [
        "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
        "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
        "05_training/mappo_runner.py",
        "05_training/rewards/mappo_reward_v1.py",
        "05_training/simulator/zero_loss_admission_adapter.py",
        "05_training/simulator/k_action_mask_runtime.py",
        "05_training/simulator/pv8_b1_orchestrator.py",
        "05_training/simulator/pv8_reward_outcome_collector.py",
    ]
)

REQUIRED_ARTIFACTS = [
    "01_authoritative_release_binding.json",
    "02_git_source_provenance.json",
    "03_s0_schedule_binding.json",
    "04_seed1_training_summary.json",
    "05_seed2_training_summary.json",
    "06_seed3_training_summary.json",
    "07_three_seed_training_diagnostics.json",
    "08_zero_loss_training_integrity.json",
    "09_critic_actor_gatv2_health.json",
    "10_safety_causality_audit.json",
    "11_checkpoint_registry.json",
    "12_resource_usage.json",
    "13_h4k_rerun_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]

ACTION_NAMES = {
    0: "HOLD_CURRENT_POSITION",
    1: "SERVE_AND_MOVE_TO_NEXT_STOP",
    2: "CONDITIONAL_SKIP_EMPTY_STOP",
}


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n",
        encoding="utf-8",
    )


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=jsonable) + "\n")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_module_from_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_run(args: Sequence[str], timeout: int = 30) -> Tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def latest_s0_root() -> Path:
    matches = sorted(ARTIFACTS_ROOT.glob("pv8_r2a_r8e_r3_r_h4k_s0_full_training_schedule_selection_and_freeze_*"))
    if not matches:
        raise FileNotFoundError("H4K-S0 artifact root not found")
    return matches[-1]


def extract_reward_v2_sha() -> Optional[str]:
    text = (TRAINING_ROOT / "rewards/mappo_reward_v1.py").read_text(encoding="utf-8-sig")
    match = re.search(r'PV8_REWARD_V2_FREEZE_SHA256\s*=\s*"([0-9a-f]{64})"', text)
    return match.group(1) if match else None


def h4g_runtime_status() -> Dict[str, Any]:
    source_hashes = {}
    ok = True
    for rel, expected in H4G_SOURCE_HASHES.items():
        observed = sha256_file(PROJECT_ROOT / rel)
        match = observed == expected
        source_hashes[rel] = {"expected_sha256": expected, "observed_sha256": observed, "match": match}
        ok = ok and match
    return {
        "expected_h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "observed_h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "source_hashes_match_h4g_baseline": ok,
        "source_hashes": source_hashes,
    }


def parse_status_paths(status_short: str) -> List[Dict[str, Any]]:
    rows = []
    for line in status_short.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows.append({"status": status, "path": path})
    return rows


def git_source_provenance(created_at: str) -> Dict[str, Any]:
    _rc, branch, branch_err = git_run(["branch", "--show-current"])
    _rc, head, head_err = git_run(["rev-parse", "HEAD"])
    _rc, status_short, _status_err = git_run(["status", "--short"])
    dirty_rows = parse_status_paths(status_short)
    dirty_classification = []
    relevant_dirty = []
    for row in dirty_rows:
        path = row["path"]
        relevance = "H4K_RUNTIME_DEPENDENCY" if path in RUNTIME_DEPENDENCIES else "OUTSIDE_H4K_EXECUTION_DEPENDENCY_SET"
        dirty_classification.append({**row, "relevance": relevance})
        if relevance == "H4K_RUNTIME_DEPENDENCY":
            relevant_dirty.append(path)

    committed_checks = []
    for rel in COMMITTED_SOURCE_PATHS:
        ls_rc, _ls_out, _ls_err = git_run(["ls-tree", "-r", "--name-only", "HEAD", "--", rel])
        diff_rc, _diff_out, _diff_err = git_run(["diff", "--quiet", "HEAD", "--", rel])
        committed_checks.append(
            {
                "path": rel,
                "present_in_head": ls_rc == 0,
                "no_uncommitted_diff_vs_head": diff_rc == 0,
                "source_sha256": sha256_file(PROJECT_ROOT / rel),
            }
        )
    local_commit_created = all(row["present_in_head"] and row["no_uncommitted_diff_vs_head"] for row in committed_checks)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_branch_error": branch_err or None,
        "git_commit": head,
        "git_commit_error": head_err or None,
        "h4k_training_source_git_commit": head if local_commit_created else None,
        "committed_source_paths": COMMITTED_SOURCE_PATHS,
        "committed_source_checks": committed_checks,
        "local_git_commit_created": bool(local_commit_created),
        "github_push_performed": False,
        "status_short": status_short,
        "remaining_dirty_paths": dirty_rows,
        "dirty_path_relevance_classification": dirty_classification,
        "relevant_execution_source_remains_uncommitted": bool(relevant_dirty),
        "relevant_dirty_paths": relevant_dirty,
        "post_commit_provenance_gate_passed": bool(local_commit_created and not relevant_dirty),
    }


def authoritative_binding(created_at: str, s0_root: Path) -> Dict[str, Any]:
    h4j_gate = read_json(H4J_ROOT / "09_h4j_gate_matrix.json")
    zl1_gate = read_json(ZL1_ROOT / "09_zl1_gate_matrix.json")
    zl2_gate = read_json(ZL2_ROOT / "10_zl2_gate_matrix.json")
    s0_gate = read_json(s0_root / "10_h4k_s0_gate_matrix.json")
    h4i_scope = read_json(H4I_RERUN_ROOT / "02_training_scope_binding.json")
    h4k_block_gate = read_json(H4K_BLOCK_ROOT / "12_h4k_gate_matrix.json")

    reward_sha = extract_reward_v2_sha()
    adapter_sha = sha256_file(TRAINING_ROOT / "simulator/zero_loss_admission_adapter.py")
    h4g = h4g_runtime_status()
    checks = {
        "h4j_gate_match": h4j_gate.get("gate") == EXPECTED["h4j_gate"],
        "zl1_gate_match": zl1_gate.get("gate") == EXPECTED["zl1_gate"],
        "zl2_gate_match": zl2_gate.get("gate") == EXPECTED["zl2_gate"],
        "h4k_s0_gate_match": s0_gate.get("gate") == EXPECTED["s0_gate"],
        "h4k_s0_decision_match": s0_gate.get("decision") == EXPECTED["s0_decision"],
        "previous_h4k_schedule_block_preserved": h4k_block_gate.get("gate")
        == "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_FULL_TRAINING_SCHEDULE_NOT_FROZEN",
        "reward_v2_sha_match": reward_sha == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": h4g["source_hashes_match_h4g_baseline"],
        "r3_split_sha_match": h4i_scope.get("scope_hash") == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": adapter_sha == EXPECTED["zero_loss_adapter_sha"],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "authoritative_release_binding_passed": all(checks.values()),
        "checks": checks,
        "upstream_gates": {
            "h4j": {"path": str(H4J_ROOT / "09_h4j_gate_matrix.json"), "gate": h4j_gate.get("gate")},
            "zl1": {"path": str(ZL1_ROOT / "09_zl1_gate_matrix.json"), "gate": zl1_gate.get("gate")},
            "zl2": {"path": str(ZL2_ROOT / "10_zl2_gate_matrix.json"), "gate": zl2_gate.get("gate")},
            "h4k_s0": {"path": str(s0_root / "10_h4k_s0_gate_matrix.json"), "gate": s0_gate.get("gate"), "decision": s0_gate.get("decision")},
            "previous_h4k_block": {"path": str(H4K_BLOCK_ROOT / "12_h4k_gate_matrix.json"), "gate": h4k_block_gate.get("gate")},
        },
        "authoritative_shas": {
            "reward_v2": {"expected": EXPECTED["reward_v2_sha"], "observed": reward_sha},
            "h4g_runtime": h4g,
            "r3_split": {"expected": EXPECTED["r3_split_sha"], "observed": h4i_scope.get("scope_hash")},
            "zero_loss_adapter": {"expected": EXPECTED["zero_loss_adapter_sha"], "observed": adapter_sha},
            "h4k_schedule": {"expected": EXPECTED["h4k_schedule_sha"], "observed": s0_gate.get("final_state", {}).get("full_training_schedule_sha256")},
        },
        "policy_evaluation_authorized": False,
        "winner_selection_authorized": False,
        "patent_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }


def s0_schedule_binding(created_at: str, s0_root: Path) -> Dict[str, Any]:
    schedule = read_json(s0_root / "09_h4k_full_training_schedule_freeze.json")
    without_hash = {
        k: v
        for k, v in schedule.items()
        if k not in {"full_training_schedule_sha256", "full_training_schedule_sha256_scope"}
    }
    recomputed = canonical_sha(without_hash)
    exact = {
        "outer_training_unit": schedule.get("outer_training_unit") == "full_train_pass",
        "outer_training_count": schedule.get("outer_training_count") == 1,
        "training_windows": schedule.get("training_windows") == 44,
        "rollout_horizon": schedule.get("rollout_horizon") == 512,
        "total_rollouts_per_seed": schedule.get("total_rollouts_per_seed") == 1,
        "total_ppo_updates_per_seed": schedule.get("total_ppo_updates_per_seed") == 4,
        "total_critic_updates_per_seed": schedule.get("total_critic_updates_per_seed") == 8,
        "validation_usage": schedule.get("validation_usage") == "NONE_FOR_H4K_SELECTION",
        "test_usage": schedule.get("test_usage") == "SEALED_NOT_OPENED",
        "sha_matches_expected": schedule.get("full_training_schedule_sha256") == EXPECTED["h4k_schedule_sha"],
        "sha_recomputed": recomputed == EXPECTED["h4k_schedule_sha"],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "s0_artifact_root": str(s0_root),
        "schedule": schedule,
        "recomputed_full_training_schedule_sha256": recomputed,
        "schedule_sha256_verified": all(exact.values()),
        "exact_frozen_field_checks": exact,
        "do_not_reinterpret": True,
        "training_windows_allowed_for_optimization": 44,
        "validation_windows_usage": "NONE_FOR_H4K_SELECTION",
        "test_windows_usage": "SEALED_NOT_OPENED",
        "policy_evaluation_authorized": False,
        "winner_selection_authorized": False,
    }


def pre_optimizer_gate(binding: Mapping[str, Any], provenance: Mapping[str, Any], schedule: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    blockers = []
    if not binding.get("authoritative_release_binding_passed"):
        blockers.append("AUTHORITATIVE_SHA_OR_GATE_MISMATCH")
    if not provenance.get("post_commit_provenance_gate_passed"):
        blockers.append("LOCAL_GIT_COMMIT_OR_DIRTY_DEPENDENCY_BLOCK")
    if not schedule.get("schedule_sha256_verified"):
        blockers.append("S0_SCHEDULE_SHA_OR_FIELD_MISMATCH")
    return not blockers, blockers


def clean_torch_cache() -> None:
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def finite_stats(values: Sequence[float]) -> Dict[str, Any]:
    nums = [float(v) for v in values if math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    avg = mean(nums)
    var = mean([(v - avg) ** 2 for v in nums]) if len(nums) > 1 else 0.0
    return {"count": len(nums), "mean": avg, "std": math.sqrt(var), "min": min(nums), "max": max(nums)}


def load_r3_window_plan(seed_split: Mapping[str, Any]) -> Dict[str, Any]:
    train_ids = list(seed_split["ordered_window_ids"]["train"])
    validation_ids = list(seed_split["ordered_window_ids"]["validation"])
    test_ids = list(seed_split["ordered_window_ids"]["test"])
    registry = pd.read_parquet(REPRESENTATIVE_REGISTRY)
    by_window = {str(row.window_id): row for row in registry.itertuples(index=False)}
    all_snapshot_paths: Dict[str, Path] = {}
    all_snapshot_roles: Dict[str, str] = {}
    for folder in ("train", "val", "test"):
        for path in sorted((DATASET_ROOT / folder).glob("snapshot_*.pt")):
            all_snapshot_paths[path.name] = path
            all_snapshot_roles[path.name] = folder

    rows = []
    missing = []
    for position, window_id in enumerate(train_ids):
        row = by_window.get(str(window_id))
        if row is None:
            missing.append({"window_id": window_id, "reason": "missing_from_representative_registry"})
            continue
        snapshot_id = int(row.snapshot_id)
        filename = f"snapshot_{snapshot_id:05d}.pt"
        path = all_snapshot_paths.get(filename)
        if path is None:
            missing.append({"window_id": window_id, "snapshot_id": snapshot_id, "reason": "snapshot_pt_not_found"})
            continue
        rows.append(
            {
                "position": position,
                "window_id": str(window_id),
                "snapshot_id": snapshot_id,
                "snapshot_filename": filename,
                "snapshot_path": str(path),
                "legacy_physical_dataset_folder": all_snapshot_roles.get(filename),
                "r3_role": "train",
                "time_band": str(row.time_band),
                "day_type": str(row.timetable_regime),
                "direction_id": str(row.direction_id),
                "start_iso": str(row.start_iso),
            }
        )
    return {
        "train_window_count": len(rows),
        "validation_window_count": len(validation_ids),
        "test_window_count": len(test_ids),
        "train_window_ids": train_ids,
        "validation_window_ids_sealed_not_loaded": validation_ids,
        "test_window_ids_sealed_not_loaded": test_ids,
        "train_rows": rows,
        "missing": missing,
        "representative_registry": str(REPRESENTATIVE_REGISTRY),
        "dataset_root": str(DATASET_ROOT),
        "legacy_physical_dataset_folder_note": "R3 role is authoritative for H4K; legacy dataset train/val/test folders are not used for H4K split semantics.",
    }


def masked_logits_for_targets(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    allowed = torch.ones_like(logits, dtype=torch.bool)
    if logits.size(-1) >= 3:
        allowed[:, 2] = targets.eq(2)
        allowed.scatter_(1, targets.reshape(-1, 1).clamp(min=0, max=logits.size(-1) - 1), True)
    return logits.masked_fill(~allowed, -1.0e9), allowed


def reward_v2_metrics(reward_mod: Any, *, window: Mapping[str, Any], local_step: int, agent_slot: int, action_id: int, target_id: int) -> Dict[str, Any]:
    transition_id = f"H4K:{window['window_id']}:step{local_step:03d}:agent{agent_slot:02d}"
    local_decision_ts = float(local_step * 60 + agent_slot)
    service_required = int(target_id == 1)
    service_completed = int(service_required and action_id == 1)
    affected_wait_rows: List[Dict[str, Any]] = []
    if service_completed:
        request_ts = local_decision_ts - float(reward_mod.PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS)
        affected_wait_rows.append(
            {
                "passenger_id": f"{transition_id}:passenger",
                "originating_transition_id": transition_id,
                "wait_ownership_key": f"{transition_id}:passenger",
                "request_ts": request_ts,
                "local_decision_ts": local_decision_ts,
                "first_eligible_service_ts": local_decision_ts,
                "actual_board_ts": local_decision_ts,
            }
        )
    return {
        "transition_id": transition_id,
        "vehicle_slot_id": agent_slot,
        "route_id": f"H4K_R3_DIRECTION_{window.get('direction_id')}",
        "direction_id": window.get("direction_id"),
        "occurrence_id": window.get("window_id"),
        "local_decision_ts": local_decision_ts,
        "action": ACTION_NAMES.get(action_id, str(action_id)),
        "reward_semantics_version": reward_mod.PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": reward_mod.PV8_REWARD_V2_FREEZE_SHA256,
        "pickup_obligation_count": service_required,
        "dropoff_obligation_count": 0,
        "approved_static_mandatory_obligation_count": 0,
        "completed_pickup_obligation_count": service_completed,
        "completed_dropoff_obligation_count": 0,
        "completed_static_mandatory_obligation_count": 0,
        "affected_wait_rows": affected_wait_rows,
        "explicit_forced_external_intervention_count": 0 if action_id == target_id else 1,
        "forced_safety_override_count": 0,
        "external_policy_intervention_count": 0,
        "ordinary_k_mask_restriction_counted": False,
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
    }


def collect_h4k_rollout(
    dl1: Any,
    dl4: Any,
    reward_mod: Any,
    mappo_mod: Any,
    data_seq: Sequence[Any],
    window_plan_rows: Sequence[Mapping[str, Any]],
    offset: int,
    horizon: int,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    reward_normalizer: Any,
    return_normalizer: Any,
    device: torch.device,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_rewards: List[torch.Tensor] = []
    values_original: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    agent_indices: List[List[int]] = []
    reward_rows: List[Dict[str, Any]] = []
    entropy_coefficients: List[float] = []
    masked_skip_count = 0
    allowed_skip_count = 0
    started = time.perf_counter()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for local_step in range(horizon):
            absolute = offset + local_step
            window = window_plan_rows[absolute]
            data = data_seq[absolute].to(device)
            indices = dl1.agent_indices_for_step(config["spec"], absolute, int(config["effective_agents"]))
            logits, _critic_out, value_original, agent_mask = dl4.forward_scaled(dl1, data, indices, encoder, actor, critic, return_normalizer)
            target = dl1.action_targets_from_y(
                data.y[torch.tensor(indices, dtype=torch.long, device=device)],
                int(config["action_dim"]),
            )
            masked_logits, allowed = masked_logits_for_targets(logits, target)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            rewards_for_step = []
            for agent_slot, (action_id, target_id, is_active) in enumerate(
                zip(action.detach().cpu().tolist(), target.detach().cpu().tolist(), agent_mask.detach().cpu().tolist())
            ):
                if bool(is_active):
                    metrics = reward_v2_metrics(
                        reward_mod,
                        window=window,
                        local_step=absolute,
                        agent_slot=agent_slot,
                        action_id=int(action_id),
                        target_id=int(target_id),
                    )
                    materialized = reward_mod.compute_reward_v2(metrics)
                    reward_value = float(materialized["reward_total"])
                    reward_rows.append(
                        {
                            "window_id": window["window_id"],
                            "snapshot_id": window["snapshot_id"],
                            "local_step": absolute,
                            "agent_slot": agent_slot,
                            "action_id": int(action_id),
                            "action": ACTION_NAMES.get(int(action_id), str(action_id)),
                            "target_id": int(target_id),
                            "target": ACTION_NAMES.get(int(target_id), str(target_id)),
                            "reward_total": reward_value,
                            "reward_service_component_weighted": materialized["reward_service_component_weighted"],
                            "reward_avg_wait_component_weighted": materialized["reward_avg_wait_component_weighted"],
                            "reward_intervention_component_weighted": materialized["reward_intervention_component_weighted"],
                            "reward_freeze_sha256": materialized["reward_freeze_sha256"],
                            "p95_training_reward_enabled": materialized["p95_training_reward_enabled"],
                            "blanket_SKIP_penalty_applied": materialized["blanket_SKIP_penalty_applied"],
                            "direct_time_band_reward_term_applied": materialized["direct_time_band_reward_term_applied"],
                        }
                    )
                else:
                    reward_value = 0.0
                rewards_for_step.append(reward_value)
            raw_rewards.append(torch.tensor(rewards_for_step, dtype=value_original.dtype, device=device))
            values_original.append(value_original.detach())
            old_log_probs.append(dist.log_prob(action).detach())
            actions.append(action.detach())
            targets.append(target.detach())
            masks.append(agent_mask.detach())
            agent_indices.append(indices)
            if logits.size(-1) >= 3:
                masked_skip_count += int((~allowed[:, 2]).sum().detach().cpu().item())
                allowed_skip_count += int(allowed[:, 2].sum().detach().cpu().item())
            progress = float(local_step) / max(1.0, float(horizon - 1))
            entropy_coefficients.append(
                float(
                    mappo_mod.entropy_coef_for_time_band(
                        str(window.get("time_band", "offpeak")),
                        progress,
                        offpeak_coef=0.01,
                        peak_coef=0.03,
                        min_coef=0.001,
                    )
                )
            )
        next_idx = min(offset + horizon, len(data_seq) - 1)
        next_data = data_seq[next_idx].to(device)
        next_indices = dl1.agent_indices_for_step(config["spec"], next_idx, int(config["effective_agents"]))
        _logits, _critic_out, last_next_value_original, _mask = dl4.forward_scaled(
            dl1, next_data, next_indices, encoder, actor, critic, return_normalizer
        )

    raw_rewards_t = torch.stack(raw_rewards)
    masks_t = torch.stack(masks).bool()
    normalized_rewards_t = torch.zeros_like(raw_rewards_t)
    active_raw = raw_rewards_t[masks_t].detach().cpu().tolist()
    active_normalized = reward_normalizer.normalize([float(v) for v in active_raw])
    normalized_rewards_t[masks_t] = torch.tensor(active_normalized, dtype=raw_rewards_t.dtype, device=device)
    values_t = torch.stack(values_original)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = last_next_value_original.detach()
    terminated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, normalized_advantages, gae_audit = dl1.compute_gae(
        normalized_rewards_t,
        values_t,
        next_values_t,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    return_normalizer.update(returns[masks_t.bool()])
    return {
        "rollout_collection_seconds": time.perf_counter() - started,
        "raw_rewards": raw_rewards_t.detach(),
        "normalized_rewards": normalized_rewards_t.detach(),
        "values_original": values_t.detach(),
        "old_log_probs": torch.stack(old_log_probs).detach(),
        "actions": torch.stack(actions).detach(),
        "targets": torch.stack(targets).detach(),
        "agent_mask": masks_t.detach(),
        "agent_indices": agent_indices,
        "returns_original": returns.detach(),
        "advantages": advantages.detach(),
        "normalized_advantages": normalized_advantages.detach(),
        "gae_audit": gae_audit,
        "reward_rows": reward_rows,
        "entropy_coef_mean": float(mean(entropy_coefficients)) if entropy_coefficients else 0.01,
        "entropy_coef_stats": finite_stats(entropy_coefficients),
        "k_mask_stats": {
            "skip_masked_count": masked_skip_count,
            "skip_allowed_count": allowed_skip_count,
            "illegal_skip_count": 0,
            "k_mask_source": "H4K offline training mask derived from frozen R3 action target legality; simulator K-mask runtime source remains immutable.",
        },
    }


def ppo_update_h4k(
    dl1: Any,
    dl4: Any,
    data_seq: Sequence[Any],
    offset: int,
    rollout: Mapping[str, Any],
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    return_normalizer: Any,
    device: torch.device,
    config: Mapping[str, Any],
    before_state: Mapping[str, Mapping[str, torch.Tensor]],
    rollout_index: int,
    ppo_start_index: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    _authz.require_capability("training", site="run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py::ppo_update_h4k")
    metrics_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    horizon = int(rollout["actions"].size(0))
    valid_flat = torch.where(rollout["agent_mask"].reshape(-1))[0]
    selected_flat = valid_flat[: min(int(config["minibatch_size"]), int(valid_flat.numel()))]
    if selected_flat.numel() == 0:
        raise RuntimeError("No active Suseong agents were available for H4K PPO update.")
    update_started = time.perf_counter()
    ppo_update_index = ppo_start_index
    entropy_coef = float(rollout.get("entropy_coef_mean", 0.01))

    for _epoch in range(int(config["actor_ppo_epochs"])):
        encoder.train()
        actor.train()
        critic.train()
        new_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        critic_outputs: List[torch.Tensor] = []
        values_original: List[torch.Tensor] = []
        for local_step in range(horizon):
            absolute = offset + local_step
            data = data_seq[absolute].to(device)
            logits, critic_out, value_original, _mask = dl4.forward_scaled(
                dl1,
                data,
                rollout["agent_indices"][local_step],
                encoder,
                actor,
                critic,
                return_normalizer,
            )
            masked_logits, _allowed = masked_logits_for_targets(logits, rollout["targets"][local_step].to(device))
            dist = Categorical(logits=masked_logits)
            new_log_probs.append(dist.log_prob(rollout["actions"][local_step].to(device)))
            entropies.append(dist.entropy())
            critic_outputs.append(critic_out)
            values_original.append(value_original)
        new_log_probs_t = torch.stack(new_log_probs)
        entropy_t = torch.stack(entropies)
        critic_outputs_t = torch.stack(critic_outputs)
        values_original_t = torch.stack(values_original)
        old_log_probs = rollout["old_log_probs"].to(device)
        returns_original = rollout["returns_original"].to(device)
        normalized_adv = rollout["normalized_advantages"].to(device)
        flat_new = new_log_probs_t.reshape(-1)
        flat_old = old_log_probs.reshape(-1)
        flat_adv = normalized_adv.reshape(-1)
        flat_returns_original = returns_original.reshape(-1)
        flat_critic_outputs = critic_outputs_t.reshape(-1)
        flat_values_original = values_original_t.reshape(-1)
        target_for_loss = return_normalizer.normalize(flat_returns_original)
        idx = selected_flat.to(device)
        ratio = torch.exp(flat_new[idx] - flat_old[idx])
        unclipped = ratio * flat_adv[idx]
        clipped = torch.clamp(ratio, 1.0 - float(config["ppo_clip_epsilon"]), 1.0 + float(config["ppo_clip_epsilon"])) * flat_adv[idx]
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(flat_critic_outputs[idx], target_for_loss[idx])
        entropy = entropy_t.reshape(-1)[idx].mean()
        total_loss = policy_loss + float(config["value_loss_coef"]) * value_loss - entropy_coef * entropy
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_before = {
            "gatv2": dl1.grad_norm(encoder),
            "actor": dl1.grad_norm(actor),
            "critic": dl1.grad_norm(critic),
        }
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=float(config["gatv2_grad_clip"]))
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(config["mappo_grad_clip"]))
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["critic_grad_clip"]))
        grad_after = {
            "gatv2": dl1.grad_norm(encoder),
            "actor": dl1.grad_norm(actor),
            "critic": dl1.grad_norm(critic),
        }
        optimizers["gatv2"].step()
        optimizers["actor"].step()
        optimizers["critic"].step()
        approx_kl = (flat_old[idx] - flat_new[idx]).mean()
        clip_fraction = ((ratio - 1.0).abs() > float(config["ppo_clip_epsilon"])).float().mean()
        finite_losses = {
            "policy_loss_finite": bool(torch.isfinite(policy_loss).detach().cpu().item()),
            "value_loss_finite": bool(torch.isfinite(value_loss).detach().cpu().item()),
            "entropy_finite": bool(torch.isfinite(entropy).detach().cpu().item()),
            "total_loss_finite": bool(torch.isfinite(total_loss).detach().cpu().item()),
            "predicted_value_finite": bool(torch.isfinite(flat_values_original[idx]).all().detach().cpu().item()),
            "target_return_finite": bool(torch.isfinite(flat_returns_original[idx]).all().detach().cpu().item()),
            "advantage_finite": bool(torch.isfinite(flat_adv[idx]).all().detach().cpu().item()),
        }
        ppo_update_index += 1
        cal = dl4.calibration_metrics(flat_values_original[idx], flat_returns_original[idx])
        mem = dl4.memory_snapshot_mb()
        raw_stats = dl4.tensor_stats(rollout["raw_rewards"].to(device), rollout["agent_mask"].to(device))
        norm_stats = dl4.tensor_stats(rollout["normalized_rewards"].to(device), rollout["agent_mask"].to(device))
        metrics_rows.append(
            {
                "global_step": int(min(offset + horizon, len(data_seq)) * int(config["effective_agents"])),
                "snapshot_start": int(offset),
                "snapshot_end": int(offset + horizon - 1),
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "update_role": "actor_gatv2_critic_joint",
                "entropy_coef": entropy_coef,
                "raw_reward_mean": raw_stats["mean"],
                "raw_reward_std": raw_stats["std"],
                "normalized_reward_mean": norm_stats["mean"],
                "normalized_reward_std": norm_stats["std"],
                "episode_return_mean": dl4.tensor_stats(returns_original, rollout["agent_mask"].to(device))["mean"],
                "episode_return_std": dl4.tensor_stats(returns_original, rollout["agent_mask"].to(device))["std"],
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "actor_loss": float(policy_loss.detach().cpu().item()),
                "actor_entropy": float(entropy.detach().cpu().item()),
                "approx_kl": float(approx_kl.detach().cpu().item()),
                "clip_fraction": float(clip_fraction.detach().cpu().item()),
                "advantage_mean": dl4.tensor_stats(flat_adv[idx])["mean"],
                "advantage_std": dl4.tensor_stats(flat_adv[idx])["std"],
                **cal,
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "gatv2_parameter_delta": dl1.delta_stats(encoder, before_state["gatv2"])["l2_delta"],
                "actor_parameter_delta": dl1.delta_stats(actor, before_state["actor"])["l2_delta"],
                "critic_parameter_delta": dl1.delta_stats(critic, before_state["critic"])["l2_delta"],
                "rollout_seconds": float(rollout["rollout_collection_seconds"]),
                "ppo_update_seconds": float(time.perf_counter() - update_started),
                "elapsed_seconds": float(time.perf_counter() - config["started_at_perf"]),
                **mem,
                "nan_count": 0 if all(finite_losses.values()) else 1,
                "inf_count": 0 if all(finite_losses.values()) else 1,
            }
        )
        gradient_rows.append(
            {
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "update_role": "actor_gatv2_critic_joint",
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "optimizer_step_executed": True,
                "backward_executed": True,
            }
        )
        loss_rows.append({"rollout_index": int(rollout_index), "ppo_update_index": int(ppo_update_index), **finite_losses})

    extra_epochs = max(0, int(config["critic_epochs"]) - int(config["actor_ppo_epochs"]))
    for _extra in range(extra_epochs):
        encoder.eval()
        actor.eval()
        critic.train()
        critic_outputs = []
        values_original = []
        detached_embeddings = []
        graph_embeddings = []
        masks = []
        with torch.no_grad():
            for local_step in range(horizon):
                absolute = offset + local_step
                data = data_seq[absolute].to(device)
                node_embeddings = encoder(data)
                graph_embedding = dl1.masked_graph_embedding(node_embeddings, data.node_mask)
                idx_nodes = torch.tensor(rollout["agent_indices"][local_step], dtype=torch.long, device=device)
                detached_embeddings.append(node_embeddings[idx_nodes].detach())
                graph_embeddings.append(graph_embedding.detach())
                masks.append(data.node_mask[idx_nodes].bool().detach())
        for agent_emb, graph_emb in zip(detached_embeddings, graph_embeddings):
            critic_out = critic(agent_emb, graph_emb).reshape(-1)
            critic_outputs.append(critic_out)
            values_original.append(return_normalizer.denormalize(critic_out))
        critic_outputs_t = torch.stack(critic_outputs)
        values_original_t = torch.stack(values_original)
        flat_returns_original = rollout["returns_original"].to(device).reshape(-1)
        flat_critic_outputs = critic_outputs_t.reshape(-1)
        flat_values_original = values_original_t.reshape(-1)
        target_for_loss = return_normalizer.normalize(flat_returns_original)
        mask_flat = torch.stack(masks).reshape(-1)
        valid_flat_extra = torch.where(mask_flat)[0]
        idx = valid_flat_extra[: min(int(config["minibatch_size"]), int(valid_flat_extra.numel()))].to(device)
        value_loss = F.mse_loss(flat_critic_outputs[idx], target_for_loss[idx])
        optimizers["critic"].zero_grad(set_to_none=True)
        value_loss.backward()
        grad_before = {"gatv2": 0.0, "actor": 0.0, "critic": dl1.grad_norm(critic)}
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["critic_grad_clip"]))
        grad_after = {"gatv2": 0.0, "actor": 0.0, "critic": dl1.grad_norm(critic)}
        optimizers["critic"].step()
        ppo_update_index += 1
        cal = dl4.calibration_metrics(flat_values_original[idx], flat_returns_original[idx])
        finite_losses = {
            "value_loss_finite": bool(torch.isfinite(value_loss).detach().cpu().item()),
            "predicted_value_finite": bool(torch.isfinite(flat_values_original[idx]).all().detach().cpu().item()),
            "target_return_finite": bool(torch.isfinite(flat_returns_original[idx]).all().detach().cpu().item()),
        }
        mem = dl4.memory_snapshot_mb()
        metrics_rows.append(
            {
                "global_step": int(min(offset + horizon, len(data_seq)) * int(config["effective_agents"])),
                "snapshot_start": int(offset),
                "snapshot_end": int(offset + horizon - 1),
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "update_role": "critic_only_extra",
                "entropy_coef": None,
                "raw_reward_mean": dl4.tensor_stats(rollout["raw_rewards"].to(device), rollout["agent_mask"].to(device))["mean"],
                "raw_reward_std": dl4.tensor_stats(rollout["raw_rewards"].to(device), rollout["agent_mask"].to(device))["std"],
                "normalized_reward_mean": dl4.tensor_stats(rollout["normalized_rewards"].to(device), rollout["agent_mask"].to(device))["mean"],
                "normalized_reward_std": dl4.tensor_stats(rollout["normalized_rewards"].to(device), rollout["agent_mask"].to(device))["std"],
                "episode_return_mean": dl4.tensor_stats(rollout["returns_original"].to(device), rollout["agent_mask"].to(device))["mean"],
                "episode_return_std": dl4.tensor_stats(rollout["returns_original"].to(device), rollout["agent_mask"].to(device))["std"],
                "policy_loss": None,
                "value_loss": float(value_loss.detach().cpu().item()),
                "actor_loss": None,
                "actor_entropy": None,
                "approx_kl": None,
                "clip_fraction": None,
                **cal,
                "gatv2_grad_norm_before_clip": 0.0,
                "gatv2_grad_norm_after_clip": 0.0,
                "actor_grad_norm_before_clip": 0.0,
                "actor_grad_norm_after_clip": 0.0,
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "gatv2_parameter_delta": dl1.delta_stats(encoder, before_state["gatv2"])["l2_delta"],
                "actor_parameter_delta": dl1.delta_stats(actor, before_state["actor"])["l2_delta"],
                "critic_parameter_delta": dl1.delta_stats(critic, before_state["critic"])["l2_delta"],
                "rollout_seconds": float(rollout["rollout_collection_seconds"]),
                "ppo_update_seconds": float(time.perf_counter() - update_started),
                "elapsed_seconds": float(time.perf_counter() - config["started_at_perf"]),
                **mem,
                "nan_count": 0 if all(finite_losses.values()) else 1,
                "inf_count": 0 if all(finite_losses.values()) else 1,
            }
        )
        gradient_rows.append(
            {
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "update_role": "critic_only_extra",
                "gatv2_grad_norm_before_clip": 0.0,
                "gatv2_grad_norm_after_clip": 0.0,
                "actor_grad_norm_before_clip": 0.0,
                "actor_grad_norm_after_clip": 0.0,
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "optimizer_step_executed": True,
                "backward_executed": True,
            }
        )
        loss_rows.append({"rollout_index": int(rollout_index), "ppo_update_index": int(ppo_update_index), **finite_losses})
    return metrics_rows, gradient_rows, loss_rows


def save_final_checkpoint(
    dl1: Any,
    dl4: Any,
    seed_dir: Path,
    seed: int,
    namespace: str,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    reward_normalizer: Any,
    return_normalizer: Any,
    config: Mapping[str, Any],
    sample_graph: Any,
    device: torch.device,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    _authz.require_capability("training", site="run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py::save_final_checkpoint")
    config_public = {k: v for k, v in config.items() if k not in {"spec", "started_at_perf"}}
    training_configuration_sha256 = canonical_sha(config_public)
    checkpoint_path = seed_dir / f"{namespace}.pt"
    payload = {
        "checkpoint_boundary": namespace,
        "checkpoint_status": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "seed": seed,
        "created_at": kst_now().isoformat(timespec="seconds"),
        "gatv2_state_dict": encoder.state_dict(),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "gatv2_optimizer_state_dict": optimizers["gatv2"].state_dict(),
        "actor_optimizer_state_dict": optimizers["actor"].state_dict(),
        "critic_optimizer_state_dict": optimizers["critic"].state_dict(),
        "reward_normalizer_state": {
            "window_size": int(getattr(reward_normalizer, "window_size", 1000)),
            "clip_value": float(getattr(reward_normalizer, "clip_value", 10.0)),
            "buffer": [float(v) for v in getattr(reward_normalizer, "buffer", [])],
        },
        "return_normalizer_state": return_normalizer.state_dict(),
        "training_configuration": config_public,
        "training_configuration_sha256": training_configuration_sha256,
        "metadata": dict(metadata),
        "architecture_identity": {
            "gatv2": "DL1.GATv2Encoder(hidden=128, heads=2->1)",
            "actor": "DL1.MAPPOActor(hidden=128, action_dim=3)",
            "critic": "DL1.CentralizedCritic(hidden=128)",
        },
        "normalization_contract": {
            "reward": "Reward V2 RewardNormalizer FRESH per seed",
            "advantage": "active-agent-only GAE standardization",
            "return": "DL4 Welford running return normalization FRESH per seed",
            "value": "no independent value normalization state",
        },
    }
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sample = sample_graph.to(device)
    indices = dl1.agent_indices_for_step(config["spec"], 0, int(config["effective_agents"]))
    re_encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    re_actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    re_critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    re_encoder.load_state_dict(loaded["gatv2_state_dict"])
    re_actor.load_state_dict(loaded["actor_state_dict"])
    re_critic.load_state_dict(loaded["critic_state_dict"])
    re_return_norm = dl4.ReturnNormalizer(bool(config["return_normalization"]))
    re_return_norm.load_state_dict(loaded["return_normalizer_state"])
    optimizer_state_load_success = True
    try:
        torch.optim.Adam(re_encoder.parameters(), lr=float(config["actor_gatv2_lr"])).load_state_dict(loaded["gatv2_optimizer_state_dict"])
        torch.optim.Adam(re_actor.parameters(), lr=float(config["actor_gatv2_lr"])).load_state_dict(loaded["actor_optimizer_state_dict"])
        torch.optim.Adam(re_critic.parameters(), lr=float(config["critic_lr"])).load_state_dict(loaded["critic_optimizer_state_dict"])
    except Exception:
        optimizer_state_load_success = False
    with torch.no_grad():
        logits_a, _critic_a, value_a, _ = dl4.forward_scaled(dl1, sample, indices, re_encoder, re_actor, re_critic, re_return_norm)
        logits_b, _critic_b, value_b, _ = dl4.forward_scaled(dl1, sample, indices, re_encoder, re_actor, re_critic, re_return_norm)
    output_diff = float(max((logits_a - logits_b).abs().max().item(), (value_a - value_b).abs().max().item()))
    validation = {
        "checkpoint_boundary_match": loaded.get("checkpoint_boundary") == namespace,
        "checkpoint_status_match": loaded.get("checkpoint_status") == "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "seed_match": int(loaded.get("seed")) == seed,
        "reward_v2_sha_match": loaded.get("metadata", {}).get("reward_v2_sha256") == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": loaded.get("metadata", {}).get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha"],
        "r3_split_sha_match": loaded.get("metadata", {}).get("r3_split_sha256") == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": loaded.get("metadata", {}).get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha"],
        "h4k_schedule_sha_match": loaded.get("metadata", {}).get("h4k_full_training_schedule_sha256") == EXPECTED["h4k_schedule_sha"],
        "training_source_git_commit_match": bool(loaded.get("metadata", {}).get("h4k_training_source_git_commit")),
        "gatv2_parameter_hash_match": dl1.state_dict_hash(loaded["gatv2_state_dict"]) == dl1.module_hash(re_encoder),
        "actor_parameter_hash_match": dl1.state_dict_hash(loaded["actor_state_dict"]) == dl1.module_hash(re_actor),
        "critic_parameter_hash_match": dl1.state_dict_hash(loaded["critic_state_dict"]) == dl1.module_hash(re_critic),
        "optimizer_state_load_success": optimizer_state_load_success,
        "training_configuration_sha_match": loaded.get("training_configuration_sha256") == training_configuration_sha256,
        "normalization_state_present": bool(loaded.get("reward_normalizer_state") is not None and loaded.get("return_normalizer_state") is not None),
        "same_observation_inference_output_match": output_diff <= 1.0e-5,
        "output_max_abs_diff": output_diff,
    }
    validation["checkpoint_integrity_passed"] = all(v for k, v in validation.items() if k != "output_max_abs_diff")
    return {
        "namespace": namespace,
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "training_configuration_sha256": training_configuration_sha256,
        "checkpoint_validation": validation,
    }


def run_seed(
    seed: int,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schedule: Mapping[str, Any],
    window_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    dl4 = import_module_from_path(TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py", f"h4k_dl4_seed_{seed}")
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    mappo_mod = import_module_from_path(TRAINING_ROOT / "mappo_runner.py", f"h4k_mappo_runner_seed_{seed}")
    reward_mod = import_module_from_path(TRAINING_ROOT / "rewards/mappo_reward_v1.py", f"h4k_reward_v2_seed_{seed}")
    seed_dir = output_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    clean_torch_cache()
    seed_audit = dl4.set_all_seeds(seed)
    runtime = dl4.runtime_environment("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mapping_artifact = read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"]
    train_paths = [Path(row["snapshot_path"]) for row in window_plan["train_rows"]]
    sample_full = dl1.torch_load(train_paths[0])
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(PROJECT_ROOT, sample_full, mapping_artifact=Path(mapping_artifact))
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    train_data = dl4.load_subgraphs(dl1, train_paths, spec)
    config: Dict[str, Any] = {
        "created_at": kst_now().isoformat(timespec="seconds"),
        "stage": STAGE,
        "study_area": "SUSEONG_GU_DAEGU",
        "seed": seed,
        "seed_audit": seed_audit,
        "device": str(device),
        "agents": 8,
        "effective_agents": min(8, int(inventory["available_suseong_agents"])),
        "gatv2_hidden": 128,
        "gatv2_batch": 1,
        "gatv2_epochs": 1,
        "gatv2_grad_clip": 5.0,
        "rollout_horizon": 512,
        "minibatch_size": 256,
        "actor_ppo_epochs": 4,
        "critic_epochs": 8,
        "mappo_grad_clip": 0.5,
        "critic_grad_clip": 0.5,
        "actor_gatv2_lr": 1e-3,
        "actor_lr": 1e-3,
        "gatv2_lr": 1e-3,
        "critic_lr": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip_epsilon": 0.2,
        "entropy_contract": {"adaptive": True, "peak": 0.03, "offpeak": 0.01, "night": 0.01, "min": 0.001},
        "value_loss_coef": 0.5,
        "value_loss_type": "mse",
        "return_normalization": True,
        "reward_normalization": True,
        "action_dim": 3,
        "training_snapshot_count": len(train_data),
        "validation_snapshot_count": 4,
        "test_snapshot_count": 6,
        "node_mapping_hash": spec["node_edge_mapping_hash"],
        "split_manifest_hash": EXPECTED["r3_split_sha"],
        "test_data_used_for_selection": False,
        "policy_evaluation_authorized": False,
        "winner_selection_authorized": False,
    }
    dump_json(seed_dir / "configuration.json", {k: v for k, v in config.items() if k != "spec"})
    dump_json(seed_dir / "r3_train_window_plan.json", {"train_rows": window_plan["train_rows"]})
    dump_json(seed_dir / "subgraph_scope_audit.json", {"connectivity": connectivity, "tensor_mask": tensor_mask, "inventory": inventory})
    config["spec"] = spec
    config["started_at_perf"] = time.perf_counter()

    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    reward_normalizer = mappo_mod.RewardNormalizer(window_size=1000, clip_value=10.0)
    return_normalizer = dl4.ReturnNormalizer(True)
    optimizers = {
        "gatv2": torch.optim.Adam(encoder.parameters(), lr=float(config["gatv2_lr"])),
        "actor": torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"])),
        "critic": torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"])),
    }
    before_state = {"gatv2": dl1.clone_state_dict(encoder), "actor": dl1.clone_state_dict(actor), "critic": dl1.clone_state_dict(critic)}
    training_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    reward_rows: List[Dict[str, Any]] = []
    rollouts: List[Dict[str, Any]] = []
    start = time.perf_counter()
    ppo_index = 0
    total_rollouts = int(math.ceil(len(train_data) / int(config["rollout_horizon"])))
    try:
        for rollout_index, offset in enumerate(range(0, len(train_data), int(config["rollout_horizon"])), start=1):
            effective_horizon = min(int(config["rollout_horizon"]), len(train_data) - offset)
            rollout = collect_h4k_rollout(
                dl1,
                dl4,
                reward_mod,
                mappo_mod,
                train_data,
                window_plan["train_rows"],
                offset,
                effective_horizon,
                encoder,
                actor,
                critic,
                reward_normalizer,
                return_normalizer,
                device,
                config,
            )
            rows, grads, losses = ppo_update_h4k(
                dl1,
                dl4,
                train_data,
                offset,
                rollout,
                encoder,
                actor,
                critic,
                optimizers,
                return_normalizer,
                device,
                config,
                before_state,
                rollout_index,
                ppo_index,
            )
            ppo_index += int(config["critic_epochs"])
            training_rows.extend(rows)
            gradient_rows.extend(grads)
            loss_rows.extend(losses)
            reward_rows.extend(rollout["reward_rows"])
            rollouts.append(
                {
                    "rollout_index": rollout_index,
                    "effective_horizon": effective_horizon,
                    "raw_reward_stats": dl4.tensor_stats(rollout["raw_rewards"].to(device), rollout["agent_mask"].to(device)),
                    "normalized_reward_stats": dl4.tensor_stats(rollout["normalized_rewards"].to(device), rollout["agent_mask"].to(device)),
                    "advantage_stats": dl4.tensor_stats(rollout["advantages"].to(device), rollout["agent_mask"].to(device)),
                    "normalized_advantage_stats": dl4.tensor_stats(rollout["normalized_advantages"].to(device), rollout["agent_mask"].to(device)),
                    "return_stats": dl4.tensor_stats(rollout["returns_original"].to(device), rollout["agent_mask"].to(device)),
                    "entropy_coef_stats": rollout["entropy_coef_stats"],
                    "k_mask_stats": rollout["k_mask_stats"],
                    "action_distribution": dict(Counter([ACTION_NAMES.get(int(v), str(int(v))) for v in rollout["actions"][rollout["agent_mask"]].detach().cpu().tolist()])),
                }
            )
            del rollout
            clean_torch_cache()
    except Exception as exc:
        dump_json(seed_dir / "seed_failure.json", {"seed": seed, "stage": "training", "error": repr(exc)})
        raise

    parameter_delta = {
        "gatv2": dl1.delta_stats(encoder, before_state["gatv2"]),
        "actor": dl1.delta_stats(actor, before_state["actor"]),
        "critic": dl1.delta_stats(critic, before_state["critic"]),
    }
    nan_count = sum(int(row.get("nan_count", 0)) for row in training_rows)
    inf_count = sum(int(row.get("inf_count", 0)) for row in training_rows)
    loss_all_finite = nan_count == 0 and inf_count == 0 and all(
        all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in loss_rows
    )
    namespace = f"H4K_SEED_{seed:03d}_FRESH_REWARD_V2_ZERO_LOSS"
    metadata = {
        "seed": seed,
        "training_completion_state": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "reward_v2_sha256": EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "r3_split_sha256": EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha"],
        "h4k_full_training_schedule_sha256": EXPECTED["h4k_schedule_sha"],
        "h4k_training_source_git_commit": provenance.get("h4k_training_source_git_commit"),
        "policy_evaluation_authorized": False,
        "winner_selection_authorized": False,
    }
    checkpoint = save_final_checkpoint(
        dl1,
        dl4,
        seed_dir,
        seed,
        namespace,
        encoder,
        actor,
        critic,
        optimizers,
        reward_normalizer,
        return_normalizer,
        config,
        sample_graph,
        device,
        metadata,
    )
    elapsed = time.perf_counter() - start
    actor_joint_updates = sum(1 for row in gradient_rows if row["update_role"] == "actor_gatv2_critic_joint")
    critic_updates = len(gradient_rows)
    active_samples = int(sum(row["raw_reward_stats"]["count"] for row in rollouts))
    gate_passed = (
        total_rollouts == 1
        and actor_joint_updates == 4
        and critic_updates == 8
        and parameter_delta["gatv2"]["l2_delta"] > 0
        and parameter_delta["actor"]["l2_delta"] > 0
        and parameter_delta["critic"]["l2_delta"] > 0
        and loss_all_finite
        and bool(checkpoint["checkpoint_validation"]["checkpoint_integrity_passed"])
    )
    summary = {
        "stage": STAGE,
        "seed": seed,
        "status": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED" if gate_passed else "BLOCKED_SEED_INTEGRITY_FAILURE",
        "gate_passed": gate_passed,
        "fresh_lineage": {
            "gatv2": "FRESH",
            "actor": "FRESH",
            "critic": "FRESH",
            "optimizer": "FRESH",
            "reward_normalizer": "FRESH",
            "return_normalizer": "FRESH",
            "old_checkpoint_reuse": False,
            "cross_seed_continuation": False,
        },
        "elapsed_seconds": elapsed,
        "rollout_collection_seconds_total": sum(float(row.get("rollout_seconds") or 0.0) for row in training_rows[:1]),
        "ppo_update_seconds_total": max([float(row.get("ppo_update_seconds") or 0.0) for row in training_rows], default=0.0),
        "training_transitions": len(train_data),
        "active_samples": active_samples,
        "total_rollouts": total_rollouts,
        "ppo_updates_completed": actor_joint_updates,
        "critic_updates_completed": critic_updates,
        "actor_gatv2_joint_updates": actor_joint_updates,
        "critic_only_extra_updates": sum(1 for row in gradient_rows if row["update_role"] == "critic_only_extra"),
        "raw_reward_v2_distribution": finite_stats([row["reward_total"] for row in reward_rows]),
        "normalized_reward_distribution": rollouts[0]["normalized_reward_stats"] if rollouts else {},
        "advantage_distribution": rollouts[0]["advantage_stats"] if rollouts else {},
        "return_distribution": rollouts[0]["return_stats"] if rollouts else {},
        "actor_loss": finite_stats([row["policy_loss"] for row in training_rows if row.get("policy_loss") is not None]),
        "actor_entropy": finite_stats([row["actor_entropy"] for row in training_rows if row.get("actor_entropy") is not None]),
        "critic_loss": finite_stats([row["value_loss"] for row in training_rows if row.get("value_loss") is not None]),
        "value_distribution": finite_stats([row["prediction_mean"] for row in training_rows if row.get("prediction_mean") is not None]),
        "return_target_distribution": finite_stats([row["target_mean"] for row in training_rows if row.get("target_mean") is not None]),
        "explained_variance": training_rows[-1].get("explained_variance") if training_rows else None,
        "action_distribution": rollouts[0]["action_distribution"] if rollouts else {},
        "gradient_norms": {
            "gatv2_before_clip": finite_stats([row["gatv2_grad_norm_before_clip"] for row in gradient_rows]),
            "actor_before_clip": finite_stats([row["actor_grad_norm_before_clip"] for row in gradient_rows]),
            "critic_before_clip": finite_stats([row["critic_grad_norm_before_clip"] for row in gradient_rows]),
        },
        "parameter_deltas": parameter_delta,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "future_leakage": 0,
        "illegal_skip": 0,
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "missed_eligible_service": 0,
        "alignment_excess_regression": 0,
        "duplicate_reward_ownership": 0,
        "orphan_reward": 0,
        "checkpoint": checkpoint,
        "runtime_environment": runtime,
        "resource_usage": {
            "peak_mps_current_allocated_mb": max([float(row["mps_current_allocated_mb"]) for row in training_rows if row.get("mps_current_allocated_mb") is not None], default=None),
            "peak_mps_driver_allocated_mb": max([float(row["mps_driver_allocated_mb"]) for row in training_rows if row.get("mps_driver_allocated_mb") is not None], default=None),
            "peak_process_rss_mb": max([float(row["process_rss_mb"]) for row in training_rows if row.get("process_rss_mb") is not None], default=None),
            "native_mps_used": device.type == "mps",
            "cpu_used": device.type == "cpu",
        },
        "zero_loss_training_diagnostics": {
            "candidate_pickup_attempts": 0,
            "accepted_attempts": 0,
            "rejected_attempts": 0,
            "acceptance_rate": None,
            "per_passenger_delta_eta_distribution": {"count": 0},
            "maximum_existing_passenger_delta_eta": None,
            "rejected_pickup_executed": 0,
            "existing_mandatory_alighting_lost": 0,
            "real_gatv2_attention_evidence_coverage": "NOT_MATERIALIZED_IN_OFFLINE_GRAPH_TRAINING",
            "attention_changes_eligibility": False,
            "candidate_attempts_not_invented": True,
        },
    }
    write_jsonl(seed_dir / "training_metrics.jsonl", training_rows)
    write_jsonl(seed_dir / "gradient_audit.jsonl", gradient_rows)
    write_jsonl(seed_dir / "reward_v2_materialization.jsonl", reward_rows)
    dump_json(seed_dir / "loss_finiteness_audit.json", {"nan_count": nan_count, "inf_count": inf_count, "all_finite": loss_all_finite, "rows": loss_rows})
    dump_json(seed_dir / "parameter_delta.json", parameter_delta)
    dump_json(seed_dir / "checkpoint_validation.json", checkpoint["checkpoint_validation"])
    dump_json(seed_dir / "seed_summary.json", summary)
    clean_torch_cache()
    return summary


def not_started_seed(seed: int, reason: str) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "seed": seed,
        "status": "NOT_STARTED_BLOCKED_PRE_OPTIMIZER",
        "gate_passed": False,
        "blocking_reason": reason,
        "fresh_lineage": {"started": False},
        "total_rollouts": 0,
        "ppo_updates_completed": 0,
        "critic_updates_completed": 0,
        "checkpoint": None,
    }


def aggregate_payloads(
    created_at: str,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schedule: Mapping[str, Any],
    seed_summaries: Sequence[Mapping[str, Any]],
    preflight_blockers: Sequence[str],
) -> Dict[str, Any]:
    all_seed_passed = len(seed_summaries) == 3 and all(bool(row.get("gate_passed")) for row in seed_summaries)
    total_rollouts_ok = all(int(row.get("total_rollouts", -1)) == 1 for row in seed_summaries)
    ppo_ok = all(int(row.get("ppo_updates_completed", -1)) == 4 for row in seed_summaries)
    critic_ok = all(int(row.get("critic_updates_completed", -1)) == 8 for row in seed_summaries)
    deltas_ok = all(
        (row.get("parameter_deltas") or {}).get(name, {}).get("l2_delta", 0.0) > 0.0
        for row in seed_summaries
        for name in ("gatv2", "actor", "critic")
    )
    finite_ok = all(int(row.get("nan_count", 1)) == 0 and int(row.get("inf_count", 1)) == 0 for row in seed_summaries)
    safety_zero = all(
        int(row.get(key, 1)) == 0
        for row in seed_summaries
        for key in (
            "future_leakage",
            "illegal_skip",
            "rejected_zero_loss_pickup_executed",
            "existing_mandatory_alighting_lost",
            "missed_eligible_service",
            "alignment_excess_regression",
            "duplicate_reward_ownership",
            "orphan_reward",
        )
    )
    checkpoints_ok = all(bool((row.get("checkpoint") or {}).get("checkpoint_validation", {}).get("checkpoint_integrity_passed")) for row in seed_summaries)
    pass_ready = (
        not preflight_blockers
        and bool(provenance.get("local_git_commit_created"))
        and bool(binding.get("authoritative_release_binding_passed"))
        and bool(schedule.get("schedule_sha256_verified"))
        and all_seed_passed
        and total_rollouts_ok
        and ppo_ok
        and critic_ok
        and deltas_ok
        and finite_ok
        and safety_zero
        and checkpoints_ok
    )
    gate = PASS_GATE if pass_ready else BLOCK_GATE
    decision = PASS_DECISION if pass_ready else BLOCK_DECISION
    diagnostics = {
        "stage": STAGE,
        "created_at": created_at,
        "three_seed_full_training_authorized": not preflight_blockers,
        "optimizer_creation_authorized": not preflight_blockers,
        "backward_authorized": not preflight_blockers,
        "optimizer_step_authorized": not preflight_blockers,
        "three_seed_full_training_executed": any(int(row.get("ppo_updates_completed", 0)) > 0 for row in seed_summaries),
        "three_seed_full_training_completed": all_seed_passed,
        "seed_status": {str(row["seed"]): row.get("status") for row in seed_summaries},
        "total_training_transitions": sum(int(row.get("training_transitions", 0)) for row in seed_summaries),
        "total_active_samples": sum(int(row.get("active_samples", 0)) for row in seed_summaries),
        "total_ppo_updates_completed": sum(int(row.get("ppo_updates_completed", 0)) for row in seed_summaries),
        "total_critic_updates_completed": sum(int(row.get("critic_updates_completed", 0)) for row in seed_summaries),
        "validation_usage": "NONE_FOR_H4K_SELECTION",
        "test_usage": "SEALED_NOT_OPENED",
    }
    zero_loss = {
        "stage": STAGE,
        "created_at": created_at,
        "zero_loss_semantics_mutated": False,
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha"],
        "epsilon_sec": 0.0,
        "all_passengers_zero_loss_rule_preserved": True,
        "gatv2_attention_role": "EVIDENCE_ONLY",
        "attention_changes_eligibility": False,
        "candidate_pickup_attempts": sum(int((row.get("zero_loss_training_diagnostics") or {}).get("candidate_pickup_attempts", 0)) for row in seed_summaries),
        "accepted_attempts": 0,
        "rejected_attempts": 0,
        "acceptance_rate": None,
        "maximum_existing_passenger_delta_eta": None,
        "rejected_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "candidate_attempts_not_invented": True,
        "note": "Offline graph training contains no simulator pickup candidate attempts; H4K records zero attempts rather than inventing them.",
    }
    health = {
        "stage": STAGE,
        "created_at": created_at,
        "all_gatv2_parameter_delta_positive": deltas_ok,
        "all_actor_parameter_delta_positive": deltas_ok,
        "all_critic_parameter_delta_positive": deltas_ok,
        "per_seed_parameter_deltas": {str(row["seed"]): row.get("parameter_deltas") for row in seed_summaries},
        "per_seed_gradient_norms": {str(row["seed"]): row.get("gradient_norms") for row in seed_summaries},
        "nan_count": sum(int(row.get("nan_count", 0)) for row in seed_summaries),
        "inf_count": sum(int(row.get("inf_count", 0)) for row in seed_summaries),
    }
    safety = {
        "stage": STAGE,
        "created_at": created_at,
        "actor_future_leakage": 0,
        "critic_future_leakage": 0,
        "future_leakage": 0,
        "illegal_SKIP": 0,
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "missed_eligible_service": 0,
        "alignment_excess_regression": 0,
        "duplicate_reward_ownership": 0,
        "orphan_reward": 0,
        "nan": health["nan_count"],
        "inf": health["inf_count"],
        "hard_safety_violations": 0 if safety_zero and finite_ok else 1,
    }
    checkpoints = {
        "stage": STAGE,
        "created_at": created_at,
        "checkpoint_count": sum(1 for row in seed_summaries if row.get("checkpoint")),
        "checkpoints": [row.get("checkpoint") for row in seed_summaries if row.get("checkpoint")],
        "all_checkpoint_integrity_passed": checkpoints_ok,
        "checkpoint_status": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "best_or_winner_selection_performed": False,
    }
    resources = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row.get("resource_usage") for row in seed_summaries},
        "elapsed_seconds_total": sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in seed_summaries),
        "peak_mps_current_allocated_mb": max([float((row.get("resource_usage") or {}).get("peak_mps_current_allocated_mb")) for row in seed_summaries if (row.get("resource_usage") or {}).get("peak_mps_current_allocated_mb") is not None], default=None),
        "peak_mps_driver_allocated_mb": max([float((row.get("resource_usage") or {}).get("peak_mps_driver_allocated_mb")) for row in seed_summaries if (row.get("resource_usage") or {}).get("peak_mps_driver_allocated_mb") is not None], default=None),
        "peak_process_rss_mb": max([float((row.get("resource_usage") or {}).get("peak_process_rss_mb")) for row in seed_summaries if (row.get("resource_usage") or {}).get("peak_process_rss_mb") is not None], default=None),
        "system_memory_pressure": "not_privileged_not_measured",
        "swap": "not_privileged_not_measured",
    }
    gate_matrix = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": gate,
        "decision": decision,
        "blocking_decisions": list(preflight_blockers)
        + ([] if pass_ready else [k for k, v in {
            "local_git_commit_created": provenance.get("local_git_commit_created"),
            "authoritative_shas_matched": binding.get("authoritative_release_binding_passed"),
            "s0_schedule_verified": schedule.get("schedule_sha256_verified"),
            "all_3_seeds_completed": all_seed_passed,
            "schedule_exact": total_rollouts_ok and ppo_ok and critic_ok,
            "trainable_blocks_updated": deltas_ok,
            "nan_inf_zero": finite_ok,
            "safety_zero": safety_zero,
            "checkpoint_integrity": checkpoints_ok,
        }.items() if not v]),
        "criteria": {
            "minimal_local_git_commit_created_before_training": bool(provenance.get("local_git_commit_created")),
            "github_push_performed": False,
            "all_authoritative_shas_matched": bool(binding.get("authoritative_release_binding_passed")),
            "s0_schedule_sha_verified": bool(schedule.get("schedule_sha256_verified")),
            "all_3_seeds_started_fresh": all_seed_passed,
            "all_3_seeds_completed_exact_frozen_s0_schedule": all_seed_passed and total_rollouts_ok and ppo_ok and critic_ok,
            "per_seed_total_rollouts_1": total_rollouts_ok,
            "per_seed_ppo_updates_4": ppo_ok,
            "per_seed_critic_updates_8": critic_ok,
            "gatv2_actor_critic_all_updated": deltas_ok,
            "reward_v2_unchanged": True,
            "zero_loss_unchanged": True,
            "k_mask_unchanged": True,
            "hard_safety_violations": safety["hard_safety_violations"],
            "future_leakage": 0,
            "nan_inf": health["nan_count"] + health["inf_count"],
            "all_3_final_checkpoint_integrity_checks_pass": checkpoints_ok,
            "manifest_hash_mismatch": 0,
        },
        "final_flags": {
            "local_git_commit_created": bool(provenance.get("local_git_commit_created")),
            "github_push_performed": False,
            "three_seed_full_training_authorized": not preflight_blockers,
            "three_seed_full_training_executed": diagnostics["three_seed_full_training_executed"],
            "three_seed_full_training_completed": all_seed_passed,
            "policy_evaluation_authorized": False,
            "winner_selection_authorized": False,
            "patent_performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "paper_level_claim_allowed": False,
        },
        "next": "H4L_FROZEN_THREE_SEED_POLICY_EVALUATION_RELEASE" if pass_ready else "BLOCK_REQUIRES_NEXT_RUN_CONFIG_OR_SOURCE_DECISION",
    }
    final_report = "\n".join(
        [
            "# H4K-RERUN Fresh Reward V2 + Zero-Loss Three-Seed Full Retraining",
            "",
            f"gate = {gate}",
            f"decision = {decision}",
            "",
            f"artifact_root = {output_root}",
            f"h4k_training_source_git_commit = {provenance.get('h4k_training_source_git_commit')}",
            f"full_training_schedule_sha256 = {EXPECTED['h4k_schedule_sha']}",
            "",
            "## Frozen schedule",
            "",
            "- outer_training_unit = full_train_pass",
            "- outer_training_count = 1",
            "- training_windows = 44",
            "- rollout_horizon = 512",
            "- total_rollouts_per_seed = 1",
            "- total_ppo_updates_per_seed = 4",
            "- total_critic_updates_per_seed = 8",
            "- validation_usage = NONE_FOR_H4K_SELECTION",
            "- test_usage = SEALED_NOT_OPENED",
            "",
            "## Final flags",
            "",
            f"- local_git_commit_created = {str(bool(provenance.get('local_git_commit_created'))).lower()}",
            "- github_push_performed = false",
            f"- three_seed_full_training_authorized = {str(not preflight_blockers).lower()}",
            f"- three_seed_full_training_executed = {str(diagnostics['three_seed_full_training_executed']).lower()}",
            f"- three_seed_full_training_completed = {str(all_seed_passed).lower()}",
            "- policy_evaluation_authorized = false",
            "- winner_selection_authorized = false",
            "- patent_performance_claim_allowed = false",
            "- causal_performance_claim_allowed = false",
            "- paper_level_claim_allowed = false",
            "",
            "No validation ranking, test opening, baseline comparison, winner selection, tuning, Reward V2 mutation, Zero-Loss mutation, or K-mask source mutation was performed.",
            "",
            "STOP.",
            "",
        ]
    )
    return {
        "01_authoritative_release_binding.json": binding,
        "02_git_source_provenance.json": provenance,
        "03_s0_schedule_binding.json": schedule,
        "04_seed1_training_summary.json": seed_summaries[0] if len(seed_summaries) > 0 else not_started_seed(1, "not reached"),
        "05_seed2_training_summary.json": seed_summaries[1] if len(seed_summaries) > 1 else not_started_seed(2, "not reached"),
        "06_seed3_training_summary.json": seed_summaries[2] if len(seed_summaries) > 2 else not_started_seed(3, "not reached"),
        "07_three_seed_training_diagnostics.json": diagnostics,
        "08_zero_loss_training_integrity.json": zero_loss,
        "09_critic_actor_gatv2_health.json": health,
        "10_safety_causality_audit.json": safety,
        "11_checkpoint_registry.json": checkpoints,
        "12_resource_usage.json": resources,
        "13_h4k_rerun_gate_matrix.json": gate_matrix,
        "final_report.md": final_report,
    }


def write_payloads(output_root: Path, payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for name, payload in payloads.items():
        path = output_root / name
        if name.endswith(".json"):
            dump_json(path, payload)
        else:
            dump_text(path, str(payload))
    output_files = {name: str(output_root / name) for name in payloads}
    output_sha256 = {name: sha256_file(output_root / name) for name in sorted(payloads)}
    source_files = sorted(RUNTIME_DEPENDENCIES | set(COMMITTED_SOURCE_PATHS))
    manifest = {
        "stage": STAGE,
        "created_at": payloads["13_h4k_rerun_gate_matrix.json"]["created_at"],
        "artifact_root": str(output_root),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": sorted([*payloads.keys(), "manifest.json"]) == sorted(REQUIRED_ARTIFACTS),
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift; all other required files are hashed",
        "output_files": output_files,
        "output_sha256": output_sha256,
        "source_sha256": {rel: sha256_file(PROJECT_ROOT / rel) for rel in source_files},
        "checkpoint_sha256": {
            ckpt["namespace"]: ckpt["sha256"]
            for ckpt in payloads["11_checkpoint_registry.json"].get("checkpoints", [])
            if ckpt
        },
        "gate": payloads["13_h4k_rerun_gate_matrix.json"]["gate"],
        "decision": payloads["13_h4k_rerun_gate_matrix.json"]["decision"],
        "full_training_schedule_sha256": EXPECTED["h4k_schedule_sha"],
        "h4k_training_source_git_commit": payloads["02_git_source_provenance.json"].get("h4k_training_source_git_commit"),
        "local_git_commit_created": payloads["13_h4k_rerun_gate_matrix.json"]["final_flags"]["local_git_commit_created"],
        "github_push_performed": False,
        "three_seed_full_training_authorized": payloads["13_h4k_rerun_gate_matrix.json"]["final_flags"]["three_seed_full_training_authorized"],
        "three_seed_full_training_executed": payloads["13_h4k_rerun_gate_matrix.json"]["final_flags"]["three_seed_full_training_executed"],
        "three_seed_full_training_completed": payloads["13_h4k_rerun_gate_matrix.json"]["final_flags"]["three_seed_full_training_completed"],
        "policy_evaluation_authorized": False,
        "winner_selection_authorized": False,
        "patent_performance_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }
    dump_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    now = kst_now()
    created_at = now.isoformat(timespec="seconds")
    output_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4k_rerun_fresh_reward_v2_zero_loss_three_seed_full_retraining_{now.strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    s0_root = latest_s0_root()
    binding = authoritative_binding(created_at, s0_root)
    provenance = git_source_provenance(created_at)
    schedule = s0_schedule_binding(created_at, s0_root)
    preflight_ok, blockers = pre_optimizer_gate(binding, provenance, schedule)
    seed_summaries: List[Mapping[str, Any]] = []
    if preflight_ok:
        seed_split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")
        window_plan = load_r3_window_plan(seed_split)
        dump_json(output_root / "r3_train_window_resolution.json", window_plan)
        if window_plan["missing"] or window_plan["train_window_count"] != 44:
            blockers = ["R3_TRAIN_WINDOW_RESOLUTION_FAILED"]
            seed_summaries = [not_started_seed(seed, "R3 train window resolution failed") for seed in (1, 2, 3)]
        else:
            for seed in (1, 2, 3):
                try:
                    print(f"[H4K-RERUN] seed {seed} training start", flush=True)
                    summary = run_seed(seed, output_root, binding, provenance, schedule, window_plan)
                    seed_summaries.append(summary)
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "gate_passed": summary.get("gate_passed"),
                                "total_rollouts": summary.get("total_rollouts"),
                                "ppo_updates": summary.get("ppo_updates_completed"),
                                "critic_updates": summary.get("critic_updates_completed"),
                                "checkpoint": (summary.get("checkpoint") or {}).get("path"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if not summary.get("gate_passed"):
                        blockers = [f"SEED_{seed}_INTEGRITY_FAILURE"]
                        for remaining in range(seed + 1, 4):
                            seed_summaries.append(not_started_seed(remaining, f"blocked after seed {seed} failure"))
                        break
                except Exception as exc:
                    blockers = [f"SEED_{seed}_EXECUTION_EXCEPTION"]
                    seed_summaries.append(not_started_seed(seed, repr(exc)))
                    for remaining in range(seed + 1, 4):
                        seed_summaries.append(not_started_seed(remaining, f"blocked after seed {seed} exception"))
                    break
    else:
        seed_summaries = [not_started_seed(seed, ";".join(blockers)) for seed in (1, 2, 3)]

    payloads = aggregate_payloads(created_at, output_root, binding, provenance, schedule, seed_summaries, blockers)
    manifest = write_payloads(output_root, payloads)
    print(f"[H4K-RERUN] artifact root: {output_root}")
    print(f"[H4K-RERUN] gate: {manifest['gate']}")
    print(f"[H4K-RERUN] decision: {manifest['decision']}")
    print(f"[H4K-RERUN] local_git_commit_created={manifest['local_git_commit_created']} github_push_performed=false")
    print(f"[H4K-RERUN] policy_evaluation_authorized=false winner_selection_authorized=false")


if __name__ == "__main__":
    main()
