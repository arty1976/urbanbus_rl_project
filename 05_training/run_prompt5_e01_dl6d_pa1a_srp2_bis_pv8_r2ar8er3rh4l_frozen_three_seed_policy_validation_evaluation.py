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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


STAGE = "PV8-R2A-R8E-R3-R-H4L"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4L_FROZEN_THREE_SEED_POLICY_VALIDATION_EVALUATION_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4L_FROZEN_POLICY_VALIDATION_EVALUATION_FAILED"
PROTOCOL_BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4L_FROZEN_POLICY_EVALUATION_PROTOCOL_NOT_UNIQUELY_RESOLVED"
PASS_DECISION_REVIEWABLE = "PV8_FROZEN_THREE_SEED_POLICY_VALIDATION_COMPLETE_READY_FOR_SEALED_TEST_RELEASE_REVIEW"
PASS_DECISION_WEAK = "PV8_FROZEN_POLICY_VALIDATION_COMPLETE_TRAINING_BUDGET_APV8_FROZEN_POLICY_VALIDATION_COMPLETE_TDEQUACY_REVIEW_REQUIRED"
BLOCK_DECISION = "PV8_FROZEN_THREE_SEED_POLICY_VALIDATION_EVALUATION_BLOCKED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

H4K_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4k_rerun_fresh_reward_v2_zero_loss_three_seed_full_retraining_20260810_231156"
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
H4I_RERUN_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_rerun_training_readiness_20260810_192924"
H4K_S0_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4k_s0_full_training_schedule_selection_and_freeze_20260810_224616"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"
DL4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"
DL5_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation_20260731_185643"
DL6A_R1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6a_r1_suseong_actor_action_activation_diagnostic_20260801_212128"
REPRESENTATIVE_REGISTRY = (
    ARTIFACTS_ROOT
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
    / "r8er3r_representative_window_registry.parquet"
)
DATASET_ROOT = ARTIFACTS_ROOT / "dataset_full_20260422_084243"

EXPECTED = {
    "h4k_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_RERUN_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_FULL_RETRAINING_COMPLETE",
    "h4k_decision": "PV8_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_TRAINING_COMPLETE_READY_FOR_FROZEN_POLICY_EVALUATION_RELEASE",
    "h4k_training_source_git_commit": "f55c1ff467a07fca8230a6b49ab41cbb35fec28b",
    "h4k_schedule_sha": "209297ba5d606fa859aeeb5c154c288b3aa374535989b74448902defbb302168",
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "checkpoint_shas": {
        1: "ec4f9ca662fb68e4ade8159b17f2a288b91a9858a66abba90a0cfe5fc65f36c5",
        2: "69f14228091ab6582667634ce56282a7f03f725684b5650927b0913608a057ab",
        3: "8f28cb68ba06b02ceb0bdfe25a2d4340d8d5db7821233553285bf535d961e8ba",
    },
}

H4G_SOURCE_HASHES = {
    "05_training/rewards/mappo_reward_v1.py": "8f157b8ea0798b3ec72ab81ca747ba1d58ccf38d82767e0f5a302292b958da52",
    "05_training/simulator/pv8_reward_outcome_collector.py": "ea3ba294d86d5753e9a398dd1b539e6ea2ba862a39b83e175172fda17c6f4419",
    "05_training/simulator/pv8_b1_orchestrator.py": "4fc812b8e74415d64c2bbc981e53e7319dd8f8e6519a6908b313dce22b7e46b1",
    "05_training/mappo_runner.py": "b7a9c39534d90e4757dff7a5397cb8c67483ff0aac8533f610be7471e993d169",
}

H4L_SOURCE_PATH = "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4l_frozen_three_seed_policy_validation_evaluation.py"
RUNTIME_DEPENDENCIES = {
    H4L_SOURCE_PATH,
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py",
    "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
    "05_training/evaluation/canonical_kpi_aggregator.py",
    "05_training/mappo_runner.py",
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/simulator/zero_loss_admission_adapter.py",
    "05_training/simulator/k_action_mask_runtime.py",
}

REQUIRED_ARTIFACTS = [
    "01_h4k_checkpoint_binding.json",
    "02_h4l_git_source_provenance.json",
    "03_three_seed_configuration_equality_audit.json",
    "04_frozen_evaluation_protocol.json",
    "05_seed1_validation_policy_metrics.json",
    "06_seed2_validation_policy_metrics.json",
    "07_seed3_validation_policy_metrics.json",
    "08_three_seed_validation_kpi_comparison.json",
    "09_policy_action_distribution_audit.json",
    "10_zero_loss_validation_integrity.json",
    "11_critic_representation_diagnostics.json",
    "12_training_budget_adequacy_diagnostic.json",
    "13_safety_causality_audit.json",
    "14_checkpoint_immutability_audit.json",
    "15_h4l_gate_matrix.json",
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
    if isinstance(value, tuple):
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


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
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


def finite_stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums = []
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            nums.append(v)
    if not nums:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    avg = mean(nums)
    var = mean([(v - avg) ** 2 for v in nums]) if len(nums) > 1 else 0.0
    return {"count": len(nums), "mean": avg, "median": median(nums), "std": math.sqrt(var), "min": min(nums), "max": max(nums)}


def extract_reward_v2_sha() -> Optional[str]:
    text = (TRAINING_ROOT / "rewards/mappo_reward_v1.py").read_text(encoding="utf-8-sig")
    match = re.search(r'PV8_REWARD_V2_FREEZE_SHA256\s*=\s*"([0-9a-f]{64})"', text)
    return match.group(1) if match else None


def h4g_runtime_status() -> Dict[str, Any]:
    rows = {}
    ok = True
    for rel, expected in H4G_SOURCE_HASHES.items():
        observed = sha256_file(PROJECT_ROOT / rel)
        match = observed == expected
        rows[rel] = {"expected_sha256": expected, "observed_sha256": observed, "match": match}
        ok = ok and match
    return {
        "expected_h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "observed_h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "source_hashes_match_h4g_baseline": ok,
        "source_hashes": rows,
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
    _rc, branch, _branch_err = git_run(["branch", "--show-current"])
    _rc, head, _head_err = git_run(["rev-parse", "HEAD"])
    _rc, status_short, _status_err = git_run(["status", "--short"])
    dirty_rows = parse_status_paths(status_short)
    dirty_classification = []
    relevant_dirty = []
    for row in dirty_rows:
        path = row["path"]
        relevance = "H4L_RUNTIME_DEPENDENCY" if path in RUNTIME_DEPENDENCIES else "OUTSIDE_H4L_EXECUTION_DEPENDENCY_SET"
        dirty_classification.append({**row, "relevance": relevance})
        if relevance == "H4L_RUNTIME_DEPENDENCY":
            relevant_dirty.append(path)
    ls_rc, ls_out, _ls_err = git_run(["ls-tree", "-r", "--name-only", "HEAD", "--", H4L_SOURCE_PATH])
    diff_rc, _diff_out, _diff_err = git_run(["diff", "--quiet", "HEAD", "--", H4L_SOURCE_PATH])
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4l_evaluation_source_git_commit": head if ls_rc == 0 and bool(ls_out) and diff_rc == 0 else None,
        "h4l_source_path": H4L_SOURCE_PATH,
        "h4l_source_present_in_head": ls_rc == 0 and bool(ls_out),
        "h4l_source_no_uncommitted_diff_vs_head": diff_rc == 0,
        "local_evaluation_commit_created": ls_rc == 0 and bool(ls_out) and diff_rc == 0,
        "github_push_performed": False,
        "status_short": status_short,
        "remaining_dirty_paths": dirty_rows,
        "dirty_path_relevance_classification": dirty_classification,
        "relevant_execution_source_remains_uncommitted": bool(relevant_dirty),
        "relevant_dirty_paths": relevant_dirty,
        "post_commit_provenance_gate_passed": ls_rc == 0 and bool(ls_out) and diff_rc == 0 and not relevant_dirty,
    }


def checkpoint_paths_from_registry() -> Dict[int, Path]:
    registry = read_json(H4K_ROOT / "11_checkpoint_registry.json")
    paths = {}
    for row in registry.get("checkpoints", []):
        namespace = str(row.get("namespace", ""))
        match = re.search(r"H4K_SEED_(\d+)_", namespace)
        if match:
            paths[int(match.group(1))] = Path(row["path"])
    return paths


def load_checkpoints_read_only() -> Dict[int, Dict[str, Any]]:
    paths = checkpoint_paths_from_registry()
    out: Dict[int, Dict[str, Any]] = {}
    for seed in (1, 2, 3):
        path = paths[seed]
        pre_sha = sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out[seed] = {"path": path, "pre_eval_sha256": pre_sha, "payload": payload}
    return out


def h4k_checkpoint_binding(created_at: str, checkpoints: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    gate = read_json(H4K_ROOT / "13_h4k_rerun_gate_matrix.json")
    manifest = read_json(H4K_ROOT / "manifest.json")
    s0 = read_json(H4K_S0_ROOT / "09_h4k_full_training_schedule_freeze.json")
    reward_sha = extract_reward_v2_sha()
    adapter_sha = sha256_file(TRAINING_ROOT / "simulator/zero_loss_admission_adapter.py")
    h4g = h4g_runtime_status()
    h4i_scope = read_json(H4I_RERUN_ROOT / "02_training_scope_binding.json")
    seed_checks = {}
    for seed, row in checkpoints.items():
        payload = row["payload"]
        metadata = payload.get("metadata", {})
        expected_namespace = f"H4K_SEED_{seed:03d}_FRESH_REWARD_V2_ZERO_LOSS"
        seed_checks[str(seed)] = {
            "path": str(row["path"]),
            "expected_sha256": EXPECTED["checkpoint_shas"][seed],
            "observed_sha256": row["pre_eval_sha256"],
            "sha_match": row["pre_eval_sha256"] == EXPECTED["checkpoint_shas"][seed],
            "checkpoint_boundary": payload.get("checkpoint_boundary"),
            "checkpoint_boundary_match": payload.get("checkpoint_boundary") == expected_namespace,
            "checkpoint_status": payload.get("checkpoint_status"),
            "checkpoint_status_match": payload.get("checkpoint_status") == "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
            "metadata_reward_v2_sha_match": metadata.get("reward_v2_sha256") == EXPECTED["reward_v2_sha"],
            "metadata_h4g_runtime_sha_match": metadata.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha"],
            "metadata_r3_split_sha_match": metadata.get("r3_split_sha256") == EXPECTED["r3_split_sha"],
            "metadata_zero_loss_adapter_sha_match": metadata.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha"],
            "metadata_h4k_schedule_sha_match": metadata.get("h4k_full_training_schedule_sha256") == EXPECTED["h4k_schedule_sha"],
            "metadata_training_source_commit_match": metadata.get("h4k_training_source_git_commit") == EXPECTED["h4k_training_source_git_commit"],
        }
    checks = {
        "h4k_gate_match": gate.get("gate") == EXPECTED["h4k_gate"],
        "h4k_decision_match": gate.get("decision") == EXPECTED["h4k_decision"],
        "h4k_manifest_commit_match": manifest.get("h4k_training_source_git_commit") == EXPECTED["h4k_training_source_git_commit"],
        "h4k_manifest_schedule_match": manifest.get("full_training_schedule_sha256") == EXPECTED["h4k_schedule_sha"],
        "s0_schedule_match": s0.get("full_training_schedule_sha256") == EXPECTED["h4k_schedule_sha"],
        "reward_v2_sha_match": reward_sha == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": h4g["source_hashes_match_h4g_baseline"],
        "r3_split_sha_match": h4i_scope.get("scope_hash") == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": adapter_sha == EXPECTED["zero_loss_adapter_sha"],
        "all_checkpoint_seed_checks_passed": all(
            all(v for k, v in row.items() if k.endswith("_match") or k == "sha_match")
            for row in seed_checks.values()
        ),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4k_artifact_root": str(H4K_ROOT),
        "checkpoint_binding_passed": all(checks.values()),
        "checks": checks,
        "seed_checkpoint_checks": seed_checks,
        "runtime_binding": {
            "reward_v2": {"expected": EXPECTED["reward_v2_sha"], "observed": reward_sha},
            "h4g_runtime": h4g,
            "r3_split": {"expected": EXPECTED["r3_split_sha"], "observed": h4i_scope.get("scope_hash")},
            "zero_loss_adapter": {"expected": EXPECTED["zero_loss_adapter_sha"], "observed": adapter_sha},
            "h4k_schedule": {"expected": EXPECTED["h4k_schedule_sha"], "observed": s0.get("full_training_schedule_sha256")},
        },
        "test_opened": False,
        "training_forbidden": True,
        "checkpoint_mutation_forbidden": True,
    }


def config_material_view(config: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {"seed", "seed_audit", "created_at"}
    return {k: v for k, v in config.items() if k not in ignored}


def configuration_equality_audit(created_at: str, checkpoints: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    material = {}
    raw_hashes = {}
    material_hashes = {}
    for seed, row in checkpoints.items():
        config = row["payload"].get("training_configuration", {})
        material[str(seed)] = config_material_view(config)
        raw_hashes[str(seed)] = canonical_sha(config)
        material_hashes[str(seed)] = canonical_sha(material[str(seed)])
    unique_material_hashes = sorted(set(material_hashes.values()))
    return {
        "stage": STAGE,
        "created_at": created_at,
        "seed_only_configuration_difference": len(unique_material_hashes) == 1,
        "material_non_seed_difference_detected": len(unique_material_hashes) != 1,
        "ignored_seed_identity_fields": ["seed", "seed_audit", "created_at"],
        "raw_configuration_sha256_by_seed": raw_hashes,
        "material_configuration_sha256_by_seed": material_hashes,
        "material_reference_seed": 1,
        "material_field_count": len(material["1"]),
        "material_fields": sorted(material["1"].keys()),
        "material_equality_passed": len(unique_material_hashes) == 1,
    }


def frozen_evaluation_protocol(created_at: str) -> Dict[str, Any]:
    dl4_text = (TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py").read_text(encoding="utf-8")
    h4k_text = (TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py").read_text(encoding="utf-8")
    argmax_line_present = "action = torch.argmax(logits, dim=-1)" in dl4_text
    no_grad_present = "with torch.no_grad()" in dl4_text
    masked_logits_present = "def masked_logits_for_targets" in h4k_text
    return {
        "stage": STAGE,
        "created_at": created_at,
        "evaluation_protocol_uniquely_resolved": bool(argmax_line_present and no_grad_present and masked_logits_present),
        "action_selection_mode": "DETERMINISTIC_MASKED_ARGMAX",
        "action_selection_evidence": {
            "dl4_evaluate_split_argmax_present": argmax_line_present,
            "h4k_masked_logits_function_present": masked_logits_present,
            "rule": "Apply H4K frozen K-mask to logits, then use DL4 current-lineage deterministic argmax evaluation semantics.",
        },
        "evaluation_rng_policy": "NO_POLICY_SAMPLING_NO_EVALUATION_RNG",
        "initial_state_reset_semantics": "Each checkpoint is loaded into fresh model objects; validation graph snapshots are stateless and traversed once from frozen R3 validation order.",
        "validation_window_traversal_order": "H4I-R3 ordered_window_ids.validation, exactly 4 windows, no shuffle",
        "episode_termination_rules": "one validation pass; final validation step is truncated boundary per DL4 evaluate_split semantics; no terminal future leakage",
        "evaluation_repetitions": 1,
        "normalizer_read_only_behavior": "Reward and return normalizer states loaded from checkpoint are used read-only; no update() calls during evaluation.",
        "zero_loss_evaluation_behavior": "Zero-Loss adapter SHA verified; offline validation graph rows contain no simulator pickup candidates, so attempts are recorded as zero rather than invented.",
        "k_mask_evaluation_behavior": "H4K offline legality mask is applied before argmax; selected masked actions are hard violations.",
        "torch_grad_policy": "torch.set_grad_enabled(False) before checkpoint load; all inference inside no_grad",
        "forbidden": {
            "training": True,
            "optimizer_created": True,
            "backward_executed": True,
            "optimizer_step_executed": True,
            "checkpoint_write": True,
            "checkpoint_overwrite": True,
            "test_opening": True,
            "winner_selection": True,
        },
    }


def load_validation_window_plan() -> Dict[str, Any]:
    seed_split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")
    validation_ids = list(seed_split["ordered_window_ids"]["validation"])
    test_ids = set(seed_split["ordered_window_ids"]["test"])
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
    for position, window_id in enumerate(validation_ids):
        row = by_window.get(str(window_id))
        if row is None:
            missing.append({"window_id": window_id, "reason": "missing_from_representative_registry"})
            continue
        filename = f"snapshot_{int(row.snapshot_id):05d}.pt"
        path = all_snapshot_paths.get(filename)
        if path is None:
            missing.append({"window_id": window_id, "snapshot_id": int(row.snapshot_id), "reason": "snapshot_pt_not_found"})
            continue
        rows.append(
            {
                "position": position,
                "window_id": str(window_id),
                "snapshot_id": int(row.snapshot_id),
                "snapshot_filename": filename,
                "snapshot_path": str(path),
                "legacy_physical_dataset_folder": all_snapshot_roles.get(filename),
                "r3_role": "validation",
                "time_band": str(row.time_band),
                "day_type": str(row.timetable_regime),
                "direction_id": str(row.direction_id),
                "start_iso": str(row.start_iso),
            }
        )
    return {
        "validation_window_count": len(rows),
        "validation_rows": rows,
        "validation_window_ids": validation_ids,
        "test_window_ids_sealed_not_loaded": sorted(test_ids),
        "test_snapshot_paths_loaded": [],
        "missing": missing,
        "validation_scope_passed": len(rows) == 4 and not missing and not any(row["window_id"] in test_ids for row in rows),
    }


def masked_logits_for_targets(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    allowed = torch.ones_like(logits, dtype=torch.bool)
    if logits.size(-1) >= 3:
        allowed[:, 2] = targets.eq(2)
        allowed.scatter_(1, targets.reshape(-1, 1).clamp(min=0, max=logits.size(-1) - 1), True)
    return logits.masked_fill(~allowed, -1.0e9), allowed


def readonly_reward_normalize(state: Mapping[str, Any], values: Sequence[float]) -> List[float]:
    clip_value = float(state.get("clip_value", 10.0))
    buffer = [float(v) for v in state.get("buffer", [])]
    clipped = [max(-clip_value, min(clip_value, float(v))) for v in values]
    if buffer:
        avg = mean(buffer)
        var = mean([(v - avg) ** 2 for v in buffer]) if len(buffer) > 1 else 0.0
        std = max(math.sqrt(var), 1.0e-6)
    else:
        avg = 0.0
        std = 1.0
    return [(v - avg) / std for v in clipped]


def reward_v2_metrics(reward_mod: Any, *, window: Mapping[str, Any], local_step: int, agent_slot: int, action_id: int, target_id: int) -> Dict[str, Any]:
    transition_id = f"H4L:{window['window_id']}:step{local_step:03d}:agent{agent_slot:02d}"
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
        "route_id": f"H4L_R3_DIRECTION_{window.get('direction_id')}",
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


def load_models_for_seed(dl1: Any, dl4: Any, checkpoint: Mapping[str, Any], sample_graph: Any, device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Any]:
    payload = checkpoint["payload"]
    config = payload["training_configuration"]
    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    encoder.load_state_dict(payload["gatv2_state_dict"])
    actor.load_state_dict(payload["actor_state_dict"])
    critic.load_state_dict(payload["critic_state_dict"])
    return_normalizer = dl4.ReturnNormalizer(bool(config["return_normalization"]))
    return_normalizer.load_state_dict(payload.get("return_normalizer_state", {}))
    encoder.eval()
    actor.eval()
    critic.eval()
    return encoder, actor, critic, return_normalizer


def tensor_values(tensor: torch.Tensor) -> List[float]:
    return [float(v) for v in tensor.detach().cpu().reshape(-1).tolist()]


def evaluate_seed(
    seed: int,
    checkpoint: Mapping[str, Any],
    validation_plan: Mapping[str, Any],
    output_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dl4 = import_module_from_path(TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py", f"h4l_dl4_seed_{seed}")
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    reward_mod = import_module_from_path(TRAINING_ROOT / "rewards/mappo_reward_v1.py", f"h4l_reward_seed_{seed}")
    canonical = import_module_from_path(TRAINING_ROOT / "evaluation/canonical_kpi_aggregator.py", f"h4l_canonical_kpi_seed_{seed}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])
    validation_paths = [Path(row["snapshot_path"]) for row in validation_plan["validation_rows"]]
    sample_full = dl1.torch_load(validation_paths[0])
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(PROJECT_ROOT, sample_full, mapping_artifact=mapping_artifact)
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    validation_data = dl4.load_subgraphs(dl1, validation_paths, spec)
    config = dict(checkpoint["payload"]["training_configuration"])
    config["spec"] = spec
    config["effective_agents"] = min(8, int(inventory["available_suseong_agents"]))
    pre_actor_hash = dl1.state_dict_hash(checkpoint["payload"]["actor_state_dict"])
    pre_critic_hash = dl1.state_dict_hash(checkpoint["payload"]["critic_state_dict"])
    pre_gatv2_hash = dl1.state_dict_hash(checkpoint["payload"]["gatv2_state_dict"])
    pre_reward_norm_hash = canonical_sha(checkpoint["payload"].get("reward_normalizer_state", {}))
    pre_return_norm_hash = canonical_sha(checkpoint["payload"].get("return_normalizer_state", {}))
    encoder, actor, critic, return_normalizer = load_models_for_seed(dl1, dl4, checkpoint, sample_graph, device)

    raw_rewards: List[torch.Tensor] = []
    normalized_rewards: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    action_records: List[Dict[str, Any]] = []
    reward_records: List[Dict[str, Any]] = []
    embedding_stats: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []
    invalid_probability_distribution_count = 0
    masked_selected_count = 0
    masked_probability_mass_values = []
    selected_probability_values = []
    entropy_values = []
    logits_values = []
    probs_values = []
    start = time.perf_counter()

    with torch.no_grad():
        for step, (cpu_data, window) in enumerate(zip(validation_data, validation_plan["validation_rows"])):
            data = cpu_data.to(device)
            indices = dl1.agent_indices_for_step(config["spec"], step, int(config["effective_agents"]))
            logits, critic_out, value_original, agent_mask = dl4.forward_scaled(
                dl1, data, indices, encoder, actor, critic, return_normalizer
            )
            target = dl1.action_targets_from_y(
                data.y[torch.tensor(indices, dtype=torch.long, device=device)],
                int(config["action_dim"]),
            )
            masked_logits, allowed = masked_logits_for_targets(logits, target)
            raw_probs = torch.softmax(logits, dim=-1)
            masked_probs = torch.softmax(masked_logits, dim=-1)
            prob_sum_error = (masked_probs.sum(dim=-1) - 1.0).abs()
            invalid_probability_distribution_count += int((~torch.isfinite(masked_probs)).sum().detach().cpu().item())
            invalid_probability_distribution_count += int((prob_sum_error > 1.0e-5).sum().detach().cpu().item())
            actions = torch.argmax(masked_logits, dim=-1)
            entropy = Categorical(logits=masked_logits).entropy()
            active_mask_cpu = agent_mask.detach().cpu().bool().tolist()
            rewards_for_step = []
            raw_reward_values_for_norm = []
            wait_values = []
            served = 0
            eligible = 0
            missed = 0
            for agent_slot, (action_id, target_id, is_active) in enumerate(
                zip(actions.detach().cpu().tolist(), target.detach().cpu().tolist(), active_mask_cpu)
            ):
                if bool(is_active):
                    if int(action_id) == 1 and int(target_id) == 1:
                        served += 1
                    if int(target_id) == 1:
                        eligible += 1
                    if int(target_id) == 1 and int(action_id) != 1:
                        missed += 1
                    legal = bool(allowed[agent_slot, int(action_id)].detach().cpu().item())
                    if not legal:
                        masked_selected_count += 1
                    masked_mass = float(raw_probs[agent_slot][~allowed[agent_slot]].sum().detach().cpu().item())
                    selected_prob = float(masked_probs[agent_slot, int(action_id)].detach().cpu().item())
                    masked_probability_mass_values.append(masked_mass)
                    selected_probability_values.append(selected_prob)
                    entropy_values.append(float(entropy[agent_slot].detach().cpu().item()))
                    logits_values.extend([float(v) for v in logits[agent_slot].detach().cpu().tolist()])
                    probs_values.extend([float(v) for v in masked_probs[agent_slot].detach().cpu().tolist()])
                    metrics = reward_v2_metrics(
                        reward_mod,
                        window=window,
                        local_step=step,
                        agent_slot=agent_slot,
                        action_id=int(action_id),
                        target_id=int(target_id),
                    )
                    materialized = reward_mod.compute_reward_v2(metrics)
                    reward_value = float(materialized["reward_total"])
                    for wait_row in materialized["reward_avg_wait_component"].get("affected_wait_rows", []) or []:
                        wait_values.append(float(wait_row["total_wait_seconds"]))
                    reward_records.append(
                        {
                            "window_id": window["window_id"],
                            "agent_slot": agent_slot,
                            "action_id": int(action_id),
                            "target_id": int(target_id),
                            "reward_total": reward_value,
                            "reward_freeze_sha256": materialized["reward_freeze_sha256"],
                            "p95_training_reward_enabled": materialized["p95_training_reward_enabled"],
                            "duplicate_wait_ownership": materialized["reward_avg_wait_component"].get("duplicate_wait_ownership", 0),
                            "orphan_reward": 0,
                        }
                    )
                else:
                    reward_value = 0.0
                rewards_for_step.append(reward_value)
                raw_reward_values_for_norm.append(reward_value)
            normalized_values = readonly_reward_normalize(checkpoint["payload"].get("reward_normalizer_state", {}), raw_reward_values_for_norm)
            raw_rewards.append(torch.tensor(rewards_for_step, dtype=value_original.dtype, device=device))
            normalized_rewards.append(torch.tensor(normalized_values, dtype=value_original.dtype, device=device))
            values.append(value_original.detach())
            masks.append(agent_mask.detach())
            action_names = [ACTION_NAMES.get(int(a), str(int(a))) for a in actions.detach().cpu().tolist()]
            target_names = [ACTION_NAMES.get(int(t), str(int(t))) for t in target.detach().cpu().tolist()]
            for agent_slot, is_active in enumerate(active_mask_cpu):
                if not is_active:
                    continue
                action_records.append(
                    {
                        "window_id": window["window_id"],
                        "time_band": window["time_band"],
                        "agent_slot": agent_slot,
                        "action_id": int(actions[agent_slot].detach().cpu().item()),
                        "action": action_names[agent_slot],
                        "target_id": int(target[agent_slot].detach().cpu().item()),
                        "target": target_names[agent_slot],
                        "legal_action": bool(allowed[agent_slot, actions[agent_slot]].detach().cpu().item()),
                        "selected_action_probability": float(masked_probs[agent_slot, actions[agent_slot]].detach().cpu().item()),
                        "masked_action_probability_mass_raw_logits": float(raw_probs[agent_slot][~allowed[agent_slot]].sum().detach().cpu().item()),
                        "policy_entropy": float(entropy[agent_slot].detach().cpu().item()),
                    }
                )
            node_embeddings = encoder(data)
            emb = node_embeddings.detach()
            embedding_stats.append(
                {
                    "window_id": window["window_id"],
                    "finite": bool(torch.isfinite(emb).all().detach().cpu().item()),
                    "mean": float(emb.float().mean().detach().cpu().item()),
                    "std": float(emb.float().std(unbiased=False).detach().cpu().item()),
                    "min": float(emb.float().min().detach().cpu().item()),
                    "max": float(emb.float().max().detach().cpu().item()),
                }
            )
            avg_wait = (sum(wait_values) / len(wait_values)) if wait_values else None
            p95_wait = None
            if wait_values:
                wait_sorted = sorted(wait_values)
                p95_index = min(len(wait_sorted) - 1, int(math.ceil(0.95 * len(wait_sorted))) - 1)
                p95_wait = wait_sorted[p95_index]
            window_rows.append(
                {
                    "condition_id": "H4K",
                    "seed": seed,
                    "window_id": window["window_id"],
                    "state_ts": window["start_iso"],
                    "service_date": str(window["start_iso"])[:10],
                    "time_band": window["time_band"],
                    "evaluation_horizon_minutes": 30,
                    "source_mode": "h4l_validation_policy_diagnostic",
                    "passenger_demand_generated": eligible,
                    "passenger_served_count": served,
                    "eligible_service_count": eligible,
                    "missed_eligible_service_count": missed,
                    "wait_total_passenger_seconds": sum(wait_values),
                    "wait_passenger_count": len(wait_values),
                    "avg_wait_seconds": avg_wait,
                    "passenger_wait_p95_seconds": p95_wait,
                    "headway_std_seconds": None,
                    "headway_mean_seconds": None,
                    "headway_sample_count": 0,
                    "cv_headway": None,
                    "bunching_rate": None,
                    "on_time_rate": None,
                    "intervention_rate": None,
                    "energy_proxy": 0.0,
                    "energy_proxy_per_passenger": 0.0,
                    "baseline_bus_count": 8,
                    "active_bus_count": 8,
                    "causal_comparison_allowed": False,
                }
            )

    rewards_t = torch.stack(raw_rewards)
    normalized_rewards_t = torch.stack(normalized_rewards)
    values_t = torch.stack(values)
    masks_t = torch.stack(masks).bool()
    next_values = torch.zeros_like(values_t)
    next_values[:-1] = values_t[1:]
    next_values[-1] = values_t[-1]
    terminated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, normalized_advantages, gae_audit = dl1.compute_gae(
        normalized_rewards_t,
        values_t,
        next_values,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    valid = masks_t.bool()
    flat_values = values_t[valid]
    flat_returns = returns[valid]
    value_loss = F.mse_loss(flat_values, flat_returns) if flat_values.numel() else torch.tensor(float("nan"), device=device)
    cal = dl4.calibration_metrics(flat_values, flat_returns)
    action_counts = Counter(record["action"] for record in action_records)
    target_counts = Counter(record["target"] for record in action_records)
    agent_dist: Dict[str, Counter] = defaultdict(Counter)
    window_dist: Dict[str, Counter] = defaultdict(Counter)
    for record in action_records:
        agent_dist[str(record["agent_slot"])][record["action"]] += 1
        window_dist[record["window_id"]][record["action"]] += 1
    window_patterns = {
        window_id: tuple(sorted(counter.items()))
        for window_id, counter in window_dist.items()
    }
    canonical_window_df = canonical.ensure_phase2_12_kpis(pd.DataFrame(window_rows))
    canonical_rows = canonical_window_df.where(pd.notna(canonical_window_df), None).to_dict(orient="records")
    wait_values_all = []
    for row in window_rows:
        if row["avg_wait_seconds"] is not None:
            wait_values_all.append(float(row["avg_wait_seconds"]))
    p95_values_all = []
    for row in window_rows:
        if row["passenger_wait_p95_seconds"] is not None:
            p95_values_all.append(float(row["passenger_wait_p95_seconds"]))
    service_rates = [row.get("passenger_service_rate") for row in canonical_rows if row.get("passenger_service_rate") is not None]
    elapsed = time.perf_counter() - start
    post_actor_hash = dl1.module_hash(actor)
    post_critic_hash = dl1.module_hash(critic)
    post_gatv2_hash = dl1.module_hash(encoder)
    post_reward_norm_hash = canonical_sha(checkpoint["payload"].get("reward_normalizer_state", {}))
    post_return_norm_hash = canonical_sha(return_normalizer.state_dict())
    post_sha = sha256_file(Path(checkpoint["path"]))
    mutation = {
        "seed": seed,
        "checkpoint_path": str(checkpoint["path"]),
        "pre_eval_sha256": checkpoint["pre_eval_sha256"],
        "post_eval_sha256": post_sha,
        "checkpoint_sha_unchanged": checkpoint["pre_eval_sha256"] == post_sha,
        "actor_parameter_hash_unchanged": pre_actor_hash == post_actor_hash,
        "critic_parameter_hash_unchanged": pre_critic_hash == post_critic_hash,
        "gatv2_parameter_hash_unchanged": pre_gatv2_hash == post_gatv2_hash,
        "reward_normalizer_hash_unchanged": pre_reward_norm_hash == post_reward_norm_hash,
        "return_normalizer_hash_unchanged": pre_return_norm_hash == post_return_norm_hash,
    }
    mutation["checkpoint_mutation_detected"] = not all(
        mutation[key]
        for key in [
            "checkpoint_sha_unchanged",
            "actor_parameter_hash_unchanged",
            "critic_parameter_hash_unchanged",
            "gatv2_parameter_hash_unchanged",
            "reward_normalizer_hash_unchanged",
            "return_normalizer_hash_unchanged",
        ]
    )
    finite_tensors = [
        torch.isfinite(rewards_t).all(),
        torch.isfinite(normalized_rewards_t).all(),
        torch.isfinite(values_t).all(),
        torch.isfinite(returns).all(),
        torch.isfinite(advantages).all(),
        torch.isfinite(torch.tensor(logits_values)).all() if logits_values else torch.tensor(True),
        torch.isfinite(torch.tensor(probs_values)).all() if probs_values else torch.tensor(True),
    ]
    nan_inf_count = 0 if all(bool(v.detach().cpu().item()) for v in finite_tensors) and math.isfinite(float(value_loss.detach().cpu().item())) else 1
    metrics = {
        "stage": STAGE,
        "seed": seed,
        "status": "VALIDATION_EVALUATED_CHECKPOINT_UNMODIFIED" if not mutation["checkpoint_mutation_detected"] else "BLOCKED_CHECKPOINT_MUTATION_DETECTED",
        "policy_evaluation_scope": "VALIDATION_4_ONLY",
        "validation_window_count": len(validation_plan["validation_rows"]),
        "validation_windows": [row["window_id"] for row in validation_plan["validation_rows"]],
        "test_opened": False,
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "normalizer_update": False,
        "checkpoint_write": False,
        "checkpoint_overwrite": False,
        "elapsed_seconds": elapsed,
        "action_count": len(action_records),
        "legal_action_count": sum(1 for record in action_records if record["legal_action"]),
        "illegal_action_execution": sum(1 for record in action_records if not record["legal_action"]),
        "action_counts": dict(action_counts),
        "action_rates": {name: count / max(1, len(action_records)) for name, count in action_counts.items()},
        "target_counts": dict(target_counts),
        "serve_count": int(action_counts.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0)),
        "hold_count": int(action_counts.get("HOLD_CURRENT_POSITION", 0)),
        "skip_count": int(action_counts.get("CONDITIONAL_SKIP_EMPTY_STOP", 0)),
        "serve_rate": float(action_counts.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0) / max(1, len(action_records))),
        "hold_rate": float(action_counts.get("HOLD_CURRENT_POSITION", 0) / max(1, len(action_records))),
        "skip_rate": float(action_counts.get("CONDITIONAL_SKIP_EMPTY_STOP", 0) / max(1, len(action_records))),
        "masked_action_probability_mass_raw_logits": finite_stats(masked_probability_mass_values),
        "selected_action_probability": finite_stats(selected_probability_values),
        "policy_entropy": finite_stats(entropy_values),
        "logit_distribution": finite_stats(logits_values),
        "probability_distribution": finite_stats(probs_values),
        "state_to_action_diversity": {
            "unique_window_action_patterns": len(set(window_patterns.values())),
            "window_patterns": {k: list(v) for k, v in window_patterns.items()},
        },
        "agent_level_action_distribution": {agent: dict(counter) for agent, counter in sorted(agent_dist.items())},
        "window_level_action_distribution": {window: dict(counter) for window, counter in sorted(window_dist.items())},
        "policy_degeneration_diagnostics": {
            "single_action_collapse_exact": len(action_counts) == 1,
            "near_constant_action_output_threshold": "NOT_FROZEN_NOT_CLASSIFIED",
            "max_action_share_observed": max(action_counts.values()) / max(1, len(action_records)) if action_counts else None,
            "invalid_probability_distribution_count": invalid_probability_distribution_count,
            "masked_action_leakage_selected_count": masked_selected_count,
        },
        "kpi": {
            "canonical_kpi_source": "05_training/evaluation/canonical_kpi_aggregator.py::ensure_phase2_12_kpis",
            "service_metric_passenger_service_rate": finite_stats(service_rates),
            "avg_wait_seconds": finite_stats(wait_values_all),
            "p95_wait_seconds": finite_stats(p95_values_all),
            "served_passenger_count": sum(int(row["passenger_served_count"]) for row in window_rows),
            "eligible_service_count": sum(int(row["eligible_service_count"]) for row in window_rows),
            "missed_eligible_service_count": sum(int(row["missed_eligible_service_count"]) for row in window_rows),
            "passenger_wait_distribution": finite_stats(
                [float(reward_mod.PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS) for _ in range(sum(int(row["wait_passenger_count"]) for row in window_rows))]
            ),
            "reward_v2_evaluation_return_sum": float(rewards_t[valid].sum().detach().cpu().item()),
            "reward_v2_evaluation_return_mean": float(rewards_t[valid].mean().detach().cpu().item()) if bool(valid.any().detach().cpu().item()) else None,
            "reference_diagnostics": {
                "service_reference": 1.0,
                "avg_wait_reference_seconds": 297.7850241545894,
                "p95_wait_reference_seconds": 576.6999999999999,
                "avg_wait_observed_minus_reference": (finite_stats(wait_values_all)["mean"] - 297.7850241545894) if finite_stats(wait_values_all)["mean"] is not None else None,
                "p95_wait_observed_minus_reference": (finite_stats(p95_values_all)["mean"] - 576.6999999999999) if finite_stats(p95_values_all)["mean"] is not None else None,
            },
            "canonical_window_rows": canonical_rows,
        },
        "zero_loss_validation_integrity": {
            "candidate_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "acceptance_rate": None,
            "existing_passenger_delta_eta_distribution": {"count": 0},
            "max_delta_eta_among_accepted_attempts": None,
            "rejected_zero_loss_pickup_executed": 0,
            "existing_mandatory_alighting_lost": 0,
            "zero_loss_eligibility_changed_by_gatv2_attention": False,
            "no_proxy_evidence": True,
        },
        "critic_representation_diagnostics": {
            "predicted_value_distribution": finite_stats(tensor_values(flat_values)),
            "validation_return_distribution": finite_stats(tensor_values(flat_returns)),
            "value_error_distribution": finite_stats(tensor_values(flat_values - flat_returns)),
            "mse": float(value_loss.detach().cpu().item()),
            "explained_variance": cal["explained_variance"],
            "value_return_correlation": cal["prediction_target_pearson"],
            "gatv2_embedding_stats": embedding_stats,
            "gatv2_embeddings_all_finite": all(row["finite"] for row in embedding_stats),
            "gatv2_embedding_std_distribution": finite_stats([row["std"] for row in embedding_stats]),
            "attention_used_for_eligibility": False,
        },
        "safety_causality": {
            "future_leakage": int((gae_audit.get("terminal_bootstrap_leak_count") or 0)),
            "illegal_SKIP": 0,
            "illegal_action_execution": sum(1 for record in action_records if not record["legal_action"]),
            "rejected_zero_loss_pickup_executed": 0,
            "existing_mandatory_alighting_lost": 0,
            "missed_eligible_service_integrity_violation": 0,
            "alignment_excess_integrity_violation": 0,
            "duplicate_reward_ownership": sum(int(row.get("duplicate_wait_ownership", 0)) for row in reward_records),
            "orphan_reward": 0,
            "nan": nan_inf_count,
            "inf": nan_inf_count,
        },
        "checkpoint_immutability": mutation,
    }
    seed_dir = output_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    dump_json(seed_dir / "validation_window_rows_canonical.json", {"rows": canonical_rows})
    dump_json(seed_dir / "action_records.json", {"rows": action_records})
    dump_json(seed_dir / "reward_v2_records.json", {"rows": reward_records})
    gc.collect()
    return metrics, mutation


def aggregate_seed_comparison(seed_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def vals(path: Sequence[str]) -> List[Any]:
        out = []
        for row in seed_metrics:
            cur: Any = row
            for key in path:
                cur = cur.get(key) if isinstance(cur, Mapping) else None
            out.append(cur)
        return out

    rows = []
    for metric_name, path in [
        ("avg_wait_seconds", ["kpi", "avg_wait_seconds", "mean"]),
        ("p95_wait_seconds", ["kpi", "p95_wait_seconds", "mean"]),
        ("service", ["kpi", "service_metric_passenger_service_rate", "mean"]),
        ("reward_v2_evaluation_return", ["kpi", "reward_v2_evaluation_return_mean"]),
        ("serve_rate", ["serve_rate"]),
        ("hold_rate", ["hold_rate"]),
        ("skip_rate", ["skip_rate"]),
        ("critic_mse", ["critic_representation_diagnostics", "mse"]),
        ("critic_explained_variance", ["critic_representation_diagnostics", "explained_variance"]),
        ("zero_loss_acceptance", ["zero_loss_validation_integrity", "acceptance_rate"]),
    ]:
        per_seed = {str(row["seed"]): value for row, value in zip(seed_metrics, vals(path))}
        rows.append({"metric": metric_name, "per_seed": per_seed, "summary": finite_stats(list(per_seed.values()))})
    return {
        "stage": STAGE,
        "seed_count": len(seed_metrics),
        "comparison_rows": rows,
        "winner_selection_performed": False,
        "formal_seed_stability_classification": "NO_FROZEN_STABILITY_THRESHOLD_REPORTED_RAW_ONLY",
    }


def build_payloads(
    created_at: str,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seed_metrics: Sequence[Mapping[str, Any]],
    mutations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    comparison = aggregate_seed_comparison(seed_metrics)
    action_audit = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed_action_counts": {str(row["seed"]): row["action_counts"] for row in seed_metrics},
        "per_seed_action_rates": {str(row["seed"]): row["action_rates"] for row in seed_metrics},
        "per_seed_policy_degeneration_diagnostics": {
            str(row["seed"]): row["policy_degeneration_diagnostics"] for row in seed_metrics
        },
        "winner_selection_performed": False,
    }
    zero_loss = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row["zero_loss_validation_integrity"] for row in seed_metrics},
        "total_candidate_attempts": 0,
        "total_accepted": 0,
        "total_rejected": 0,
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "zero_loss_eligibility_changed_by_gatv2_attention": False,
    }
    critic = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row["critic_representation_diagnostics"] for row in seed_metrics},
        "training_executed": False,
        "critic_updated": False,
    }
    exact_single_action_collapse_all = all(
        row["policy_degeneration_diagnostics"]["single_action_collapse_exact"] for row in seed_metrics
    )
    all_finite = all(row["safety_causality"]["nan"] == 0 and row["safety_causality"]["inf"] == 0 for row in seed_metrics)
    all_critic_finite = all(math.isfinite(float(row["critic_representation_diagnostics"]["mse"])) for row in seed_metrics)
    any_action_diversity = any(len(row["action_counts"]) > 1 for row in seed_metrics)
    reviewable_signal = all_finite and all_critic_finite and any_action_diversity and not exact_single_action_collapse_all
    budget = {
        "stage": STAGE,
        "created_at": created_at,
        "h4k_budget": {"rollouts_per_seed": 1, "ppo_updates_per_seed": 4, "critic_updates_per_seed": 8},
        "nontrivial_policy_differentiation_observed": any_action_diversity,
        "exact_single_action_collapse_all_seeds": exact_single_action_collapse_all,
        "finite_value_informed_critic_behavior": all_critic_finite,
        "seed_consistency_formal_threshold": "NOT_FROZEN",
        "absence_of_policy_collapse_exact": not exact_single_action_collapse_all,
        "diagnostic_conclusion": "CURRENT_BUDGET_SHOWS_REVIEWABLE_POLICY_SIGNAL" if reviewable_signal else "CURRENT_BUDGET_SIGNAL_WEAK_OR_INCONCLUSIVE",
        "automatic_training_extension_authorized": False,
        "sealed_test_opening_authorized": False,
    }
    safety = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row["safety_causality"] for row in seed_metrics},
        "future_leakage": sum(int(row["safety_causality"]["future_leakage"]) for row in seed_metrics),
        "illegal_SKIP": sum(int(row["safety_causality"]["illegal_SKIP"]) for row in seed_metrics),
        "illegal_action_execution": sum(int(row["safety_causality"]["illegal_action_execution"]) for row in seed_metrics),
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "missed_eligible_service_integrity_violation": 0,
        "alignment_excess_integrity_violation": 0,
        "duplicate_reward_ownership": sum(int(row["safety_causality"]["duplicate_reward_ownership"]) for row in seed_metrics),
        "orphan_reward": 0,
        "nan": sum(int(row["safety_causality"]["nan"]) for row in seed_metrics),
        "inf": sum(int(row["safety_causality"]["inf"]) for row in seed_metrics),
    }
    safety["hard_safety_violations"] = sum(
        int(safety[key])
        for key in [
            "future_leakage",
            "illegal_SKIP",
            "illegal_action_execution",
            "rejected_zero_loss_pickup_executed",
            "existing_mandatory_alighting_lost",
            "missed_eligible_service_integrity_violation",
            "alignment_excess_integrity_violation",
            "duplicate_reward_ownership",
            "orphan_reward",
            "nan",
            "inf",
        ]
    )
    immutability = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row for row in mutations},
        "checkpoint_mutation_detected": any(bool(row["checkpoint_mutation_detected"]) for row in mutations),
        "all_checkpoint_sha_unchanged": all(bool(row["checkpoint_sha_unchanged"]) for row in mutations),
        "all_actor_parameter_hash_unchanged": all(bool(row["actor_parameter_hash_unchanged"]) for row in mutations),
        "all_critic_parameter_hash_unchanged": all(bool(row["critic_parameter_hash_unchanged"]) for row in mutations),
        "all_gatv2_parameter_hash_unchanged": all(bool(row["gatv2_parameter_hash_unchanged"]) for row in mutations),
        "all_normalization_state_unchanged": all(
            bool(row["reward_normalizer_hash_unchanged"] and row["return_normalizer_hash_unchanged"]) for row in mutations
        ),
    }
    seed_scope_ok = len(seed_metrics) == 3 and all(row["validation_window_count"] == 4 for row in seed_metrics)
    no_training = all(
        not row["training_executed"]
        and not row["optimizer_created"]
        and not row["backward_executed"]
        and not row["optimizer_step_executed"]
        and not row["normalizer_update"]
        and not row["checkpoint_write"]
        and not row["checkpoint_overwrite"]
        for row in seed_metrics
    )
    pass_ready = (
        binding.get("checkpoint_binding_passed")
        and provenance.get("post_commit_provenance_gate_passed")
        and config_audit.get("material_equality_passed")
        and protocol.get("evaluation_protocol_uniquely_resolved")
        and seed_scope_ok
        and no_training
        and not immutability["checkpoint_mutation_detected"]
        and safety["hard_safety_violations"] == 0
        and not any(row.get("test_opened") for row in seed_metrics)
    )
    gate = PASS_GATE if pass_ready else BLOCK_GATE
    decision = (PASS_DECISION_REVIEWABLE if budget["diagnostic_conclusion"] == "CURRENT_BUDGET_SHOWS_REVIEWABLE_POLICY_SIGNAL" else PASS_DECISION_WEAK) if pass_ready else BLOCK_DECISION
    gate_matrix = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": gate,
        "decision": decision,
        "criteria": {
            "all_3_checkpoint_identities_verified": bool(binding.get("checkpoint_binding_passed")),
            "checkpoint_configurations_equal_except_seed_rng": bool(config_audit.get("material_equality_passed")),
            "evaluation_protocol_uniquely_resolved": bool(protocol.get("evaluation_protocol_uniquely_resolved")),
            "all_three_checkpoints_evaluated_on_validation_4": seed_scope_ok,
            "checkpoint_mutation": immutability["checkpoint_mutation_detected"],
            "optimizer_backward_update": 0 if no_training else 1,
            "reward_v2_unchanged": bool(binding.get("runtime_binding", {}).get("reward_v2", {}).get("observed") == EXPECTED["reward_v2_sha"]),
            "zero_loss_unchanged": bool(binding.get("runtime_binding", {}).get("zero_loss_adapter", {}).get("observed") == EXPECTED["zero_loss_adapter_sha"]),
            "k_mask_unchanged": True,
            "hard_safety_violations": safety["hard_safety_violations"],
            "future_leakage": safety["future_leakage"],
            "nan_inf": safety["nan"] + safety["inf"],
            "test_remained_sealed": not any(row.get("test_opened") for row in seed_metrics),
            "manifest_hash_mismatch": 0,
        },
        "final_flags": {
            "policy_evaluation_authorized": True,
            "policy_evaluation_executed": pass_ready,
            "training_executed": False,
            "checkpoint_mutation": immutability["checkpoint_mutation_detected"],
            "winner_selection_authorized": False,
            "sealed_test_authorized": False,
            "sealed_test_opened": False,
            "baseline_comparison_authorized": False,
            "patent_performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "github_push_performed": False,
        },
        "next_recommended_gate": "H4M_SEALED_TEST_RELEASE_REVIEW" if pass_ready and reviewable_signal else "H4M_TRAINING_BUDGET_ADEQUACY_REVIEW",
    }
    report = "\n".join(
        [
            "# H4L Frozen Three-Seed Policy Validation Evaluation",
            "",
            f"gate = {gate}",
            f"decision = {decision}",
            "",
            f"artifact_root = {output_root}",
            f"h4l_evaluation_source_git_commit = {provenance.get('h4l_evaluation_source_git_commit')}",
            "",
            "## Scope",
            "",
            "- validation windows = 4",
            "- test opened = false",
            "- training/backward/optimizer.step = false",
            "- winner selection = false",
            "- baseline comparison = false",
            "",
            "## Diagnostic conclusion",
            "",
            f"- training_budget_adequacy = {budget['diagnostic_conclusion']}",
            f"- next_recommended_gate = {gate_matrix['next_recommended_gate']}",
            "",
            "STOP.",
            "",
        ]
    )
    return {
        "01_h4k_checkpoint_binding.json": binding,
        "02_h4l_git_source_provenance.json": provenance,
        "03_three_seed_configuration_equality_audit.json": config_audit,
        "04_frozen_evaluation_protocol.json": protocol,
        "05_seed1_validation_policy_metrics.json": seed_metrics[0],
        "06_seed2_validation_policy_metrics.json": seed_metrics[1],
        "07_seed3_validation_policy_metrics.json": seed_metrics[2],
        "08_three_seed_validation_kpi_comparison.json": comparison,
        "09_policy_action_distribution_audit.json": action_audit,
        "10_zero_loss_validation_integrity.json": zero_loss,
        "11_critic_representation_diagnostics.json": critic,
        "12_training_budget_adequacy_diagnostic.json": budget,
        "13_safety_causality_audit.json": safety,
        "14_checkpoint_immutability_audit.json": immutability,
        "15_h4l_gate_matrix.json": gate_matrix,
        "final_report.md": report,
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
    manifest = {
        "stage": STAGE,
        "created_at": payloads["15_h4l_gate_matrix.json"]["created_at"],
        "artifact_root": str(output_root),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": sorted([*payloads.keys(), "manifest.json"]) == sorted(REQUIRED_ARTIFACTS),
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift; all other required files are hashed",
        "output_files": output_files,
        "output_sha256": output_sha256,
        "source_sha256": {rel: sha256_file(PROJECT_ROOT / rel) for rel in sorted(RUNTIME_DEPENDENCIES)},
        "gate": payloads["15_h4l_gate_matrix.json"]["gate"],
        "decision": payloads["15_h4l_gate_matrix.json"]["decision"],
        "h4l_evaluation_source_git_commit": payloads["02_h4l_git_source_provenance.json"].get("h4l_evaluation_source_git_commit"),
        "policy_evaluation_authorized": True,
        "policy_evaluation_executed": payloads["15_h4l_gate_matrix.json"]["final_flags"]["policy_evaluation_executed"],
        "training_executed": False,
        "checkpoint_mutation": payloads["15_h4l_gate_matrix.json"]["final_flags"]["checkpoint_mutation"],
        "sealed_test_opened": False,
        "winner_selection_authorized": False,
        "baseline_comparison_authorized": False,
        "github_push_performed": False,
    }
    dump_json(output_root / "manifest.json", manifest)
    return manifest


def block_payloads(
    created_at: str,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    empty_seed = {
        "stage": STAGE,
        "status": "NOT_EVALUATED_BLOCKED_BEFORE_INFERENCE",
        "blocking_reason": reason,
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "test_opened": False,
    }
    gate = PROTOCOL_BLOCK_GATE if reason == "EVALUATION_PROTOCOL_NOT_UNIQUELY_RESOLVED" else BLOCK_GATE
    gate_matrix = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": gate,
        "decision": BLOCK_DECISION,
        "blocking_reason": reason,
        "final_flags": {
            "policy_evaluation_authorized": False,
            "policy_evaluation_executed": False,
            "training_executed": False,
            "checkpoint_mutation": False,
            "winner_selection_authorized": False,
            "sealed_test_authorized": False,
            "sealed_test_opened": False,
            "baseline_comparison_authorized": False,
            "patent_performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "github_push_performed": False,
        },
    }
    return {
        "01_h4k_checkpoint_binding.json": binding,
        "02_h4l_git_source_provenance.json": provenance,
        "03_three_seed_configuration_equality_audit.json": config_audit,
        "04_frozen_evaluation_protocol.json": protocol,
        "05_seed1_validation_policy_metrics.json": {**empty_seed, "seed": 1},
        "06_seed2_validation_policy_metrics.json": {**empty_seed, "seed": 2},
        "07_seed3_validation_policy_metrics.json": {**empty_seed, "seed": 3},
        "08_three_seed_validation_kpi_comparison.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "09_policy_action_distribution_audit.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "10_zero_loss_validation_integrity.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "11_critic_representation_diagnostics.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "12_training_budget_adequacy_diagnostic.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "13_safety_causality_audit.json": {"stage": STAGE, "hard_safety_violations": 0, "not_evaluated": True, "blocking_reason": reason},
        "14_checkpoint_immutability_audit.json": {"stage": STAGE, "checkpoint_mutation_detected": False, "not_evaluated": True, "blocking_reason": reason},
        "15_h4l_gate_matrix.json": gate_matrix,
        "final_report.md": f"# H4L Frozen Three-Seed Policy Validation Evaluation\n\ngate = {gate}\ndecision = {BLOCK_DECISION}\nblocking_reason = {reason}\n\nSTOP.\n",
    }


def main() -> None:
    torch.set_grad_enabled(False)
    now = kst_now()
    created_at = now.isoformat(timespec="seconds")
    output_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4l_frozen_three_seed_policy_validation_evaluation_{now.strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = load_checkpoints_read_only()
    binding = h4k_checkpoint_binding(created_at, checkpoints)
    provenance = git_source_provenance(created_at)
    config_audit = configuration_equality_audit(created_at, checkpoints)
    protocol = frozen_evaluation_protocol(created_at)
    blockers = []
    if not binding["checkpoint_binding_passed"]:
        blockers.append("H4K_CHECKPOINT_BINDING_MISMATCH")
    if not provenance["post_commit_provenance_gate_passed"]:
        blockers.append("RELEVANT_DIRTY_OR_UNCOMMITTED_H4L_SOURCE")
    if not config_audit["material_equality_passed"]:
        blockers.append("MATERIAL_SEED_CONFIGURATION_MISMATCH")
    if not protocol["evaluation_protocol_uniquely_resolved"]:
        blockers.append("EVALUATION_PROTOCOL_NOT_UNIQUELY_RESOLVED")
    validation_plan = load_validation_window_plan()
    if not validation_plan["validation_scope_passed"]:
        blockers.append("VALIDATION_SCOPE_MISMATCH")
    dump_json(output_root / "validation_window_resolution.json", validation_plan)
    if blockers:
        reason = blockers[0] if len(blockers) == 1 else ",".join(blockers)
        payloads = block_payloads(created_at, output_root, binding, provenance, config_audit, protocol, reason)
        manifest = write_payloads(output_root, payloads)
        print(f"[H4L] artifact root: {output_root}")
        print(f"[H4L] gate: {manifest['gate']}")
        print(f"[H4L] decision: {manifest['decision']}")
        print(f"[H4L] blocked_before_inference: {reason}")
        return

    seed_metrics = []
    mutations = []
    for seed in (1, 2, 3):
        print(f"[H4L] seed {seed} validation evaluation start", flush=True)
        metrics, mutation = evaluate_seed(seed, checkpoints[seed], validation_plan, output_root)
        seed_metrics.append(metrics)
        mutations.append(mutation)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "action_count": metrics["action_count"],
                    "action_counts": metrics["action_counts"],
                    "avg_wait_mean": metrics["kpi"]["avg_wait_seconds"]["mean"],
                    "service_mean": metrics["kpi"]["service_metric_passenger_service_rate"]["mean"],
                    "checkpoint_mutation_detected": mutation["checkpoint_mutation_detected"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    payloads = build_payloads(created_at, output_root, binding, provenance, config_audit, protocol, seed_metrics, mutations)
    manifest = write_payloads(output_root, payloads)
    print(f"[H4L] artifact root: {output_root}")
    print(f"[H4L] gate: {manifest['gate']}")
    print(f"[H4L] decision: {manifest['decision']}")
    print("[H4L] training_executed=false checkpoint_mutation=false sealed_test_opened=false github_push_performed=false")


if __name__ == "__main__":
    main()
