#!/usr/bin/env python3
"""H4M-O actor-target-conditioned retraining outcome review.

Read-only diagnosis over H4M-N traces/checkpoints. This stage decomposes the
SERVE collapse into target×action credit, analytic PPO logit-pressure, shared
vs target-routed pressure, and cyclewise gradient interference. It selects the
next minimal repair contract when supported by H4M-N evidence only.

No training, repair implementation, backward, optimizer step, TEST6/validation,
environment expansion, winner/baseline selection, or GitHub push is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import py_compile
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


STAGE = "PV8-R2A-R8E-R3-R-H4M-O"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_O_ACTOR_TARGET_CONDITIONED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_O_ACTOR_TARGET_CONDITIONED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_N_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_n_fresh_actor_target_conditioned_three_seed_retraining_20260817_134046+0900"
H4M_M_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_m_actor_target_sensitivity_repair_implementation_equivalence_validation_20260817_114202+0900"
H4M_L_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_l_target_context_repair_outcome_review_next_decision_20260817_105652+0900"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"

EXPECTED = {
    "h4m_n_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_N_FRESH_ACTOR_TARGET_CONDITIONED_THREE_SEED_RETRAINING_COMPLETE",
    "h4m_n_source_commit": "848ef852bae745b87908bc8948b6de5f30eaa998",
    "h4m_n_decision": "ACTOR_DIRECT_CONDITIONING_FAILED_POLICY_COLLAPSE_PERSISTS",
    "h4m_n_next_gate": "H4M-O_ACTOR_TARGET_CONDITIONED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION",
    "h4m_m_source_commit": "8390bffd96e161ac7113eb2a58411414e9dc9529",
    "h4m_l_repair_contract_sha256": "5637381f6f450cbda6f01a76126a5940b31984edf6b782d71ca224c4accf8754",
    "active_mps_instrumentation_contract_sha256": "e6c73da48c12edfd573069730d2eeec32c74fea74f590105e55ad3502a729c92",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
ACTION_NAMES = {0: ACTION_HOLD, 1: ACTION_SERVE, 2: ACTION_SKIP}
TARGET_NAMES = {0: "target=HOLD", 1: "target=SERVE", 2: "target=SKIP"}
ROOT_CAUSE = "MIXED_TARGET_CONDITIONED_GRADIENT_IMBALANCE"
SELECTED_REPAIR = "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_REPAIR"
NEXT_GATE = "H4M-P_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION"

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_target_action_credit_decomposition.json",
    "03_actor_gradient_pressure_decomposition.json",
    "04_shared_vs_target_branch_gradient.json",
    "05_gradient_interference_audit.json",
    "06_negative_advantage_policy_effect.json",
    "07_cyclewise_collapse_origin.json",
    "08_root_cause_attribution.json",
    "09_repair_candidate_comparison.json",
    "10_h4m_o_repair_contract.json",
    "11_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def numeric_stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "sum": 0.0, "std": None}
    mu = float(mean(nums))
    var = sum((v - mu) ** 2 for v in nums) / len(nums)
    return {
        "count": len(nums),
        "mean": mu,
        "median": float(median(nums)),
        "min": min(nums),
        "max": max(nums),
        "sum": float(sum(nums)),
        "std": math.sqrt(var),
    }


def safe_rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else float(num) / float(den)


def py_compile_audit() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h4m_o_pycompile_") as tmp:
        cfile = Path(tmp) / (SOURCE_REL.name + ".pyc")
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(cfile), doraise=True)
            py_ok = True
            err = None
        except Exception as exc:
            py_ok = False
            err = repr(exc)
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {
        "py_compile_passed": py_ok,
        "py_compile_error": err,
        "git_diff_cached_check_passed": cached.returncode == 0,
        "git_diff_cached_check_stdout": cached.stdout.strip(),
        "git_diff_cached_check_stderr": cached.stderr.strip(),
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    latest = git_run(["log", "-1", "--format=%H", "--", str(SOURCE_REL)]).stdout.strip()
    present = git_run(["cat-file", "-e", f"HEAD:{SOURCE_REL}"], check=False).returncode == 0
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_o_source_git_commit": latest,
        "source_rel": str(SOURCE_REL),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "source_present_in_head": present,
        "head_commit_files": head_files,
        "head_commit_source_only": head_files == [str(SOURCE_REL)],
        "status_short": status_short,
        "local_source_only_commit_created_before_audit": present and latest == head and head_files == [str(SOURCE_REL)] and status_short == "",
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_static: Mapping[str, Any]) -> Dict[str, Any]:
    h4mn_gate = read_json(H4M_N_ROOT / "12_gate_matrix.json")
    h4mn_binding = read_json(H4M_N_ROOT / "01_authoritative_binding.json")
    h4mn_manifest = read_json(H4M_N_ROOT / "manifest.json")
    h4mn_integrity = read_json(H4M_N_ROOT / "02_training_integrity.json")
    h4mm_binding = read_json(H4M_M_ROOT / "01_authoritative_binding.json")
    h4ml_contract = read_json(H4M_L_ROOT / "09_h4m_l_repair_contract.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    close_contract_sha = sha256_file(H4M_G_CLOSE_ROOT / "12_final_mps_equivalence_contract.json")
    h4mn_sha = h4mn_binding.get("sha_bindings", {})
    checks = {
        "source_only_commit_before_audit": provenance.get("local_source_only_commit_created_before_audit") is True,
        "py_compile_passed": compile_static.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_static.get("git_diff_cached_check_passed") is True,
        "h4m_n_gate_match": h4mn_gate.get("gate") == EXPECTED["h4m_n_gate"],
        "h4m_n_source_commit_match": h4mn_binding.get("source_provenance", {}).get("h4m_n_source_git_commit")
        == EXPECTED["h4m_n_source_commit"],
        "h4m_n_decision_match": h4mn_gate.get("decision") == EXPECTED["h4m_n_decision"],
        "h4m_n_next_gate_match": h4mn_gate.get("exact_next_gate") == EXPECTED["h4m_n_next_gate"],
        "h4m_n_training_integrity_pass": h4mn_integrity.get("training_integrity_passed") is True,
        "h4m_m_source_commit_match": h4mm_binding.get("source_provenance", {}).get("h4m_m_source_git_commit")
        == EXPECTED["h4m_m_source_commit"],
        "h4m_l_repair_contract_sha_match": h4ml_contract.get("contract_sha256")
        == EXPECTED["h4m_l_repair_contract_sha256"],
        "active_mps_contract_sha_match": close_contract_sha == EXPECTED["active_mps_instrumentation_contract_sha256"]
        and h4mn_sha.get("active_mps_instrumentation_contract_sha256") == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_b_schedule_sha_match": h4m_b_schedule.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"]
        and h4mn_sha.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": h4mn_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "r3_split_sha_match": h4mn_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "h4g_runtime_sha_match": h4mn_sha.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "zero_loss_adapter_sha_match": h4mn_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "test6_sealed": h4mn_manifest.get("TEST6_opened") is False,
        "github_push_false": h4mn_manifest.get("github_push_performed") is False,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_n": str(H4M_N_ROOT),
            "h4m_m": str(H4M_M_ROOT),
            "h4m_l": str(H4M_L_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "source_provenance": provenance,
        "compile_static_checks": compile_static,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "sha_bindings": {
            "h4m_n_source_commit": EXPECTED["h4m_n_source_commit"],
            "h4m_m_source_commit": EXPECTED["h4m_m_source_commit"],
            "h4m_l_repair_contract_sha256": EXPECTED["h4m_l_repair_contract_sha256"],
            "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "hard_lock_attestation": {
            "h4m_n_trace_read_only": True,
            "training_executed": False,
            "repair_implemented": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "reward_gae_ppo_changed": False,
            "model_weights_changed": False,
            "environment_expanded": False,
            "validation_or_test6_executed": False,
            "github_push_performed": False,
        },
    }


def parquet_paths_by_key(manifest: Mapping[str, Any]) -> Dict[str, List[Path]]:
    keys = [
        "actual_pre_action_trace",
        "ppo_policy_surrogate_sample",
        "actor_logit_pressure_by_sample",
        "advantage_normalization_sample",
        "actual_critic_td_gae_trace",
        "actual_reward_trace",
        "conditional_policy_by_sample",
    ]
    out: Dict[str, List[Path]] = {key: [] for key in keys}
    for rel, path in manifest.get("output_files", {}).items():
        if not rel.endswith(".parquet"):
            continue
        for key in keys:
            if key in rel:
                out[key].append(Path(path))
    return out


def read_parquets(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_h4mn_frames() -> Dict[str, pd.DataFrame]:
    manifest = read_json(H4M_N_ROOT / "manifest.json")
    paths = parquet_paths_by_key(manifest)
    frames = {key: read_parquets(sorted(value)) for key, value in paths.items() if key != "conditional_policy_by_sample"}
    frames["conditional_policy_by_sample"] = pd.read_parquet(paths["conditional_policy_by_sample"][0])
    return frames


def prepare_rollout_rows(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pre_cols = [
        "sample_uid",
        "seed",
        "outer_cycle",
        "target_id",
        "target",
        "action_id",
        "action",
        "masked_probability_0",
        "masked_probability_1",
        "policy_entropy",
        "actor_direct_conditioning_active",
        "actor_target_context_dim",
        "actor_conditioned_input_shape",
    ]
    reward_cols = [
        "sample_uid",
        "seed",
        "outer_cycle",
        "reward_v2_raw_total",
        "reward_after_runtime_normalization",
    ]
    td_cols = ["sample_uid", "seed", "outer_cycle", "td_delta", "raw_gae_advantage", "raw_return_target"]
    adv_cols = ["sample_uid", "seed", "outer_cycle", "normalized_advantage"]
    cond_cols = [
        "sample_uid",
        "seed",
        "cycle",
        "classification",
        "delta_full_bootstrapped_return_serve_minus_hold",
    ]
    rows = frames["actual_pre_action_trace"][pre_cols].merge(
        frames["actual_reward_trace"][reward_cols], on=["sample_uid", "seed", "outer_cycle"], how="left"
    )
    rows = rows.merge(frames["actual_critic_td_gae_trace"][td_cols], on=["sample_uid", "seed", "outer_cycle"], how="left")
    rows = rows.merge(frames["advantage_normalization_sample"][adv_cols], on=["sample_uid", "seed", "outer_cycle"], how="left")
    cond = frames["conditional_policy_by_sample"][cond_cols].rename(columns={"cycle": "outer_cycle"})
    rows = rows.merge(cond, on=["sample_uid", "seed", "outer_cycle"], how="left")
    rows["target_action_group"] = rows["target_id"].map(TARGET_NAMES) + " / sampled=" + rows["action"].astype(str)
    return rows


def prepare_ppo_rows(frames: Mapping[str, pd.DataFrame], rollout_rows: pd.DataFrame) -> pd.DataFrame:
    pre_cols = [
        "sample_uid",
        "seed",
        "outer_cycle",
        "target_id",
        "target",
        "action_id",
        "action",
        "masked_probability_0",
        "masked_probability_1",
        "actor_direct_conditioning_active",
        "actor_target_context_dim",
        "actor_conditioned_input_shape",
        "classification",
        "reward_v2_raw_total",
        "reward_after_runtime_normalization",
        "td_delta",
        "raw_gae_advantage",
    ]
    pressure_cols = [
        "sample_uid",
        "seed",
        "outer_cycle",
        "ppo_epoch",
        "hold_logit_pressure",
        "serve_logit_pressure",
        "effective_direction_on_sampled_action_logit",
    ]
    rows = frames["ppo_policy_surrogate_sample"].merge(
        rollout_rows[pre_cols], on=["sample_uid", "seed", "outer_cycle"], how="left", suffixes=("", "_rollout")
    )
    rows = rows.merge(frames["actor_logit_pressure_by_sample"][pressure_cols], on=["sample_uid", "seed", "outer_cycle", "ppo_epoch"], how="left")
    selected = rows["sampled_action_id"].astype(int)
    p_selected = rows["current_action_probability"].astype(float)
    rows["p_hold_current"] = np.where(selected == 0, p_selected, 1.0 - p_selected)
    rows["p_serve_current"] = np.where(selected == 1, p_selected, 1.0 - p_selected)
    rows["gradient_active_unclipped"] = ~rows["clip_active"].astype(bool)
    rows["policy_mass"] = np.where(
        rows["gradient_active_unclipped"],
        rows["normalized_advantage"].astype(float) * rows["probability_ratio"].astype(float),
        0.0,
    )
    rows["hold_logit_update_pressure_scalar"] = np.where(
        selected == 0,
        rows["policy_mass"] * (1.0 - rows["p_hold_current"]),
        rows["policy_mass"] * (0.0 - rows["p_hold_current"]),
    )
    rows["serve_logit_update_pressure_scalar"] = np.where(
        selected == 1,
        rows["policy_mass"] * (1.0 - rows["p_serve_current"]),
        rows["policy_mass"] * (0.0 - rows["p_serve_current"]),
    )
    rows["net_serve_minus_hold_pressure_scalar"] = (
        rows["serve_logit_update_pressure_scalar"] - rows["hold_logit_update_pressure_scalar"]
    )
    rows["target_action_group"] = rows["target_id"].map(TARGET_NAMES) + " / sampled=" + rows["sampled_action_name"].astype(str)
    return rows


def stat_payload(frame: pd.DataFrame, col: str) -> Dict[str, Any]:
    return numeric_stats(frame[col].tolist()) if col in frame.columns else numeric_stats([])


def target_action_credit_decomposition(artifact_root: Path, rollout: pd.DataFrame, ppo: pd.DataFrame) -> Dict[str, Any]:
    rollout_rows = []
    for keys, group in rollout.groupby(["seed", "outer_cycle", "target_id", "action_id"], dropna=False):
        seed, cycle, target_id, action_id = [int(k) for k in keys]
        rollout_rows.append(
            {
                "seed": seed,
                "outer_cycle": cycle,
                "target_id": target_id,
                "target": ACTION_NAMES[target_id],
                "sampled_action_id": action_id,
                "sampled_action": ACTION_NAMES[action_id],
                "sample_count": int(len(group)),
                "raw_reward": stat_payload(group, "reward_v2_raw_total"),
                "runtime_normalized_reward": stat_payload(group, "reward_after_runtime_normalization"),
                "td_delta": stat_payload(group, "td_delta"),
                "raw_gae": stat_payload(group, "raw_gae_advantage"),
                "normalized_advantage": stat_payload(group, "normalized_advantage"),
                "raw_gae_positive_rate": safe_rate(int((group["raw_gae_advantage"] > 0).sum()), len(group)),
                "normalized_advantage_positive_rate": safe_rate(int((group["normalized_advantage"] > 0).sum()), len(group)),
                "normalized_advantage_negative_rate": safe_rate(int((group["normalized_advantage"] < 0).sum()), len(group)),
            }
        )
    ppo_rows = []
    for keys, group in ppo.groupby(["seed", "outer_cycle", "ppo_epoch", "target_id", "sampled_action_id"], dropna=False):
        seed, cycle, epoch, target_id, action_id = [int(k) for k in keys]
        ppo_rows.append(
            {
                "seed": seed,
                "outer_cycle": cycle,
                "ppo_epoch": epoch,
                "target_id": target_id,
                "target": ACTION_NAMES[target_id],
                "sampled_action_id": action_id,
                "sampled_action": ACTION_NAMES[action_id],
                "ppo_row_count": int(len(group)),
                "gradient_active_rate": safe_rate(int(group["gradient_active_unclipped"].sum()), len(group)),
                "probability_ratio": stat_payload(group, "probability_ratio"),
                "unclipped_surrogate": stat_payload(group, "unclipped_surrogate"),
                "clipped_surrogate": stat_payload(group, "clipped_surrogate"),
                "effective_surrogate_contribution": stat_payload(group, "effective_policy_surrogate_contribution"),
                "entropy_contribution": stat_payload(group, "entropy_contribution"),
                "policy_mass": stat_payload(group, "policy_mass"),
                "net_serve_minus_hold_pressure": stat_payload(group, "net_serve_minus_hold_pressure_scalar"),
            }
        )
    rollout_path = artifact_root / "target_action_credit_rollout_rows.parquet"
    ppo_path = artifact_root / "target_action_credit_ppo_update_rows.parquet"
    pd.DataFrame(rollout_rows).to_parquet(rollout_path, index=False)
    pd.DataFrame(ppo_rows).to_parquet(ppo_path, index=False)
    totals = []
    for keys, roll_group in rollout.groupby(["target_id", "action_id"], dropna=False):
        target_id, action_id = [int(k) for k in keys]
        ppo_group = ppo[(ppo["target_id"].astype(int) == target_id) & (ppo["sampled_action_id"].astype(int) == action_id)]
        totals.append(
            {
                "target_id": target_id,
                "target": ACTION_NAMES[target_id],
                "sampled_action_id": action_id,
                "sampled_action": ACTION_NAMES[action_id],
                "rollout_sample_count": int(len(roll_group)),
                "ppo_row_count": int(len(ppo_group)),
                "raw_reward_sum": float(roll_group["reward_v2_raw_total"].sum()),
                "td_delta_sum": float(roll_group["td_delta"].sum()),
                "raw_gae_sum": float(roll_group["raw_gae_advantage"].sum()),
                "normalized_advantage_sum": float(roll_group["normalized_advantage"].sum()),
                "positive_raw_gae_rate": safe_rate(int((roll_group["raw_gae_advantage"] > 0).sum()), len(roll_group)),
                "positive_normalized_advantage_rate": safe_rate(int((roll_group["normalized_advantage"] > 0).sum()), len(roll_group)),
                "effective_surrogate_sum": float(ppo_group["effective_policy_surrogate_contribution"].sum()),
                "unclipped_surrogate_sum": float(ppo_group["unclipped_surrogate"].sum()),
                "clipped_surrogate_sum": float(ppo_group["clipped_surrogate"].sum()),
                "entropy_contribution_sum": float(ppo_group["entropy_contribution"].sum()),
                "policy_mass_sum": float(ppo_group["policy_mass"].sum()),
                "net_serve_minus_hold_pressure_sum": float(ppo_group["net_serve_minus_hold_pressure_scalar"].sum()),
            }
        )
    required_groups = {(0, 0), (0, 1), (1, 0), (1, 1)}
    observed_groups = {(int(row["target_id"]), int(row["sampled_action_id"])) for row in totals}
    return {
        "stage": STAGE,
        "decomposition_scope": "H4M-N frozen rollout samples plus PPO surrogate rows; SKIP absent in selected H4M-N action samples",
        "required_four_groups_present": required_groups.issubset(observed_groups),
        "rollout_row_count": int(len(rollout)),
        "ppo_row_count": int(len(ppo)),
        "aggregate_target_action_totals": totals,
        "row_level_artifacts": {
            "rollout_credit_rows": str(rollout_path),
            "rollout_credit_rows_sha256": sha256_file(rollout_path),
            "ppo_update_rows": str(ppo_path),
            "ppo_update_rows_sha256": sha256_file(ppo_path),
        },
        "target_action_credit_decomposition_passed": required_groups.issubset(observed_groups) and len(rollout) == 11616 and len(ppo) == 33792,
    }


def pressure_group_summary(group: pd.DataFrame) -> Dict[str, Any]:
    return {
        "ppo_row_count": int(len(group)),
        "policy_mass_sum": float(group["policy_mass"].sum()),
        "hold_logit_update_pressure_sum": float(group["hold_logit_update_pressure_scalar"].sum()),
        "serve_logit_update_pressure_sum": float(group["serve_logit_update_pressure_scalar"].sum()),
        "net_serve_minus_hold_pressure_sum": float(group["net_serve_minus_hold_pressure_scalar"].sum()),
        "hold_logit_update_pressure_mean": float(group["hold_logit_update_pressure_scalar"].mean()) if len(group) else None,
        "serve_logit_update_pressure_mean": float(group["serve_logit_update_pressure_scalar"].mean()) if len(group) else None,
        "net_serve_minus_hold_pressure_mean": float(group["net_serve_minus_hold_pressure_scalar"].mean()) if len(group) else None,
        "gradient_active_rate": safe_rate(int(group["gradient_active_unclipped"].sum()), len(group)),
        "sample_count_weighted_contribution": float(group["net_serve_minus_hold_pressure_scalar"].sum()) / float(len(group))
        if len(group)
        else None,
    }


def actor_gradient_pressure_decomposition(artifact_root: Path, ppo: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for keys, group in ppo.groupby(["seed", "outer_cycle", "ppo_epoch", "target_id", "sampled_action_id"], dropna=False):
        seed, cycle, epoch, target_id, action_id = [int(k) for k in keys]
        rows.append(
            {
                "seed": seed,
                "outer_cycle": cycle,
                "ppo_epoch": epoch,
                "target_id": target_id,
                "target": ACTION_NAMES[target_id],
                "sampled_action_id": action_id,
                "sampled_action": ACTION_NAMES[action_id],
                **pressure_group_summary(group),
            }
        )
    path = artifact_root / "actor_gradient_pressure_rows.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    target_action = []
    for keys, group in ppo.groupby(["target_id", "sampled_action_id"], dropna=False):
        target_id, action_id = [int(k) for k in keys]
        target_action.append(
            {
                "target_id": target_id,
                "target": ACTION_NAMES[target_id],
                "sampled_action_id": action_id,
                "sampled_action": ACTION_NAMES[action_id],
                **pressure_group_summary(group),
            }
        )
    by_target = []
    for target_id, group in ppo.groupby("target_id", dropna=False):
        by_target.append({"target_id": int(target_id), "target": ACTION_NAMES[int(target_id)], **pressure_group_summary(group)})
    shared = pressure_group_summary(ppo)
    label_consistency = {
        "trace_direction_labels_available": ppo["hold_logit_pressure"].notna().all() and ppo["serve_logit_pressure"].notna().all(),
        "skip_selected_count": int((ppo["sampled_action_id"].astype(int) == 2).sum()),
        "positive_net_serve_rows": int((ppo["net_serve_minus_hold_pressure_scalar"] > 0).sum()),
        "negative_net_serve_rows": int((ppo["net_serve_minus_hold_pressure_scalar"] < 0).sum()),
    }
    return {
        "stage": STAGE,
        "pressure_formula": "For unclipped PPO rows: update_pressure(logit_j)=advantage*ratio*(1[j=sampled]-current_policy_probability_j); clipped rows carry zero policy-gradient pressure. Entropy is reported separately as trace contribution.",
        "backward_or_optimizer_executed": False,
        "target_action_pressure_totals": target_action,
        "by_target_pressure_totals": by_target,
        "shared_actor_pressure_total": shared,
        "label_consistency": label_consistency,
        "row_level_artifact": {"path": str(path), "sha256": sha256_file(path)},
        "actor_gradient_pressure_decomposition_passed": len(ppo) == 33792
        and int((ppo["sampled_action_id"].astype(int) == 2).sum()) == 0
        and all(math.isfinite(row["net_serve_minus_hold_pressure_sum"]) for row in target_action),
    }


def shared_vs_target_branch_gradient(pressure: Mapping[str, Any]) -> Dict[str, Any]:
    by_target = {row["target_id"]: row for row in pressure["by_target_pressure_totals"]}
    hold = by_target[0]
    serve = by_target[1]
    shared_net = float(pressure["shared_actor_pressure_total"]["net_serve_minus_hold_pressure_sum"])
    hold_net = float(hold["net_serve_minus_hold_pressure_sum"])
    serve_net = float(serve["net_serve_minus_hold_pressure_sum"])
    dominance_ratio = abs(serve_net) / abs(hold_net) if hold_net != 0.0 else math.inf
    return {
        "stage": STAGE,
        "separation_semantics": {
            "shared_actor_weights": "receive aggregate pressure from all target contexts",
            "target_conditioning_specific_columns": "one-hot target columns route target=HOLD rows to HOLD-context column and target=SERVE rows to SERVE-context column; same-column cross-target interference is structurally absent in this trace-reconstructable pressure view",
            "exact_hidden_parameter_gradient_limitation": "H4M-N trace stores conditioned-input schema and logits but not per-sample hidden backprop vectors; therefore this audit separates analytically reconstructed logit-pressure by parameter route rather than mutating checkpoints with backward.",
        },
        "shared_actor_gradient_result": {
            "net_serve_minus_hold_pressure_sum": shared_net,
            "result": "SHARED_ACTOR_NET_SERVE_PRESSURE" if shared_net > 0.0 else "SHARED_ACTOR_NET_HOLD_PRESSURE",
        },
        "target_conditioning_gradient_result": {
            "target_hold_branch_net_serve_minus_hold_pressure_sum": hold_net,
            "target_hold_branch_result": "CORRECTIVE_HOLD_PRESSURE" if hold_net < 0.0 else "SERVE_PRESSURE",
            "target_serve_branch_net_serve_minus_hold_pressure_sum": serve_net,
            "target_serve_branch_result": "SERVE_PRESSURE" if serve_net > 0.0 else "HOLD_PRESSURE",
            "serve_context_abs_pressure_over_hold_context_abs_pressure": dominance_ratio,
            "same_column_target_interference": 0,
        },
        "branch_gradient_passed": hold_net < 0.0 and serve_net > 0.0 and shared_net > 0.0,
    }


def vector_cosine(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    av = np.array(a, dtype=np.float64)
    bv = np.array(b, dtype=np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return None
    return float(np.dot(av, bv) / denom)


def gradient_interference_audit(artifact_root: Path, ppo: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for cycle in sorted(int(v) for v in ppo["outer_cycle"].unique()):
        h = ppo[(ppo["outer_cycle"].astype(int) == cycle) & (ppo["target_id"].astype(int) == 0)]
        s = ppo[(ppo["outer_cycle"].astype(int) == cycle) & (ppo["target_id"].astype(int) == 1)]
        hv = [float(h["hold_logit_update_pressure_scalar"].sum()), float(h["serve_logit_update_pressure_scalar"].sum())]
        sv = [float(s["hold_logit_update_pressure_scalar"].sum()), float(s["serve_logit_update_pressure_scalar"].sum())]
        cos = vector_cosine(hv, sv)
        h_norm = float(np.linalg.norm(hv))
        s_norm = float(np.linalg.norm(sv))
        if cos is None:
            classification = "not_measurable"
        elif cos < 0.0:
            classification = "conflicting"
        elif cos > 0.0:
            classification = "aligned"
        else:
            classification = "orthogonal"
        rows.append(
            {
                "outer_cycle": cycle,
                "target_hold_vector_hold_serve": hv,
                "target_serve_vector_hold_serve": sv,
                "target_hold_norm": h_norm,
                "target_serve_norm": s_norm,
                "target_serve_norm_over_target_hold_norm": s_norm / h_norm if h_norm else None,
                "cosine_similarity": cos,
                "dot_product": float(np.dot(np.array(hv), np.array(sv))),
                "classification": classification,
                "target_hold_net_serve_minus_hold": hv[1] - hv[0],
                "target_serve_net_serve_minus_hold": sv[1] - sv[0],
                "shared_net_serve_minus_hold": (hv[1] + sv[1]) - (hv[0] + sv[0]),
                "serve_context_dominant": s_norm > h_norm,
            }
        )
    path = artifact_root / "gradient_interference_cycle_rows.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    counts = Counter(row["classification"] for row in rows)
    serve_dominant_count = sum(1 for row in rows if row["serve_context_dominant"])
    return {
        "stage": STAGE,
        "shared_parameter_interference": {
            "cycle_rows": rows,
            "classification_counts": dict(counts),
            "conflicting_cycle_count": int(counts.get("conflicting", 0)),
            "aligned_cycle_count": int(counts.get("aligned", 0)),
            "serve_context_dominant_cycle_count": serve_dominant_count,
            "overall_result": "CONFLICTING_WITH_SERVE_CONTEXT_DOMINANCE"
            if counts.get("conflicting", 0) > 0 and serve_dominant_count > 0
            else "NOT_UNIQUE",
        },
        "target_conditioning_parameter_interference": {
            "same_column_interference": 0,
            "result": "STRUCTURALLY_ROUTED_BY_ONE_HOT_TARGET_CONTEXT",
        },
        "row_level_artifact": {"path": str(path), "sha256": sha256_file(path)},
        "gradient_interference_audit_passed": len(rows) == 11 and counts.get("conflicting", 0) > 0,
    }


def negative_advantage_policy_effect(ppo: pd.DataFrame, rollout: pd.DataFrame) -> Dict[str, Any]:
    target_hold_ppo = ppo[ppo["target_id"].astype(int) == 0]
    by_action = []
    for action_id, group in target_hold_ppo.groupby("sampled_action_id"):
        action_id = int(action_id)
        by_action.append(
            {
                "sampled_action_id": action_id,
                "sampled_action": ACTION_NAMES[action_id],
                "ppo_row_count": int(len(group)),
                "normalized_advantage_mean": float(group["normalized_advantage"].mean()) if len(group) else None,
                "negative_normalized_advantage_rate": safe_rate(int((group["normalized_advantage"] < 0).sum()), len(group)),
                "hold_logit_update_pressure_sum": float(group["hold_logit_update_pressure_scalar"].sum()),
                "serve_logit_update_pressure_sum": float(group["serve_logit_update_pressure_scalar"].sum()),
                "net_serve_minus_hold_pressure_sum": float(group["net_serve_minus_hold_pressure_scalar"].sum()),
            }
        )
    target_hold_rollout = rollout[rollout["target_id"].astype(int) == 0]
    raw_credit = []
    for action_id, group in target_hold_rollout.groupby("action_id"):
        action_id = int(action_id)
        raw_credit.append(
            {
                "sampled_action_id": action_id,
                "sampled_action": ACTION_NAMES[action_id],
                "rollout_sample_count": int(len(group)),
                "raw_gae_mean": float(group["raw_gae_advantage"].mean()) if len(group) else None,
                "raw_gae_negative_rate": safe_rate(int((group["raw_gae_advantage"] < 0).sum()), len(group)),
            }
        )
    by_action_map = {row["sampled_action_id"]: row for row in by_action}
    hold_action_net = by_action_map[0]["net_serve_minus_hold_pressure_sum"]
    serve_action_net = by_action_map[1]["net_serve_minus_hold_pressure_sum"]
    aggregate_net = hold_action_net + serve_action_net
    return {
        "stage": STAGE,
        "target_hold_raw_credit_by_sampled_action": raw_credit,
        "target_hold_ppo_policy_effect_by_sampled_action": by_action,
        "softmax_coupling_answer": {
            "sampled_HOLD_negative_advantage_effect": "decreases HOLD sampled-action logit and, via binary softmax coupling, increases SERVE relative logit",
            "sampled_SERVE_negative_advantage_effect": "decreases SERVE sampled-action logit and increases HOLD relative logit",
            "sampled_HOLD_net_serve_minus_hold_pressure": hold_action_net,
            "sampled_SERVE_net_serve_minus_hold_pressure": serve_action_net,
            "target_HOLD_aggregate_net_serve_minus_hold_pressure": aggregate_net,
            "answer": "Within target=HOLD, negative sampled-HOLD rows create SERVE-increasing pressure, but negative sampled-SERVE rows create a larger HOLD-corrective pressure. Therefore target=HOLD negative credit is not the aggregate cause of global SERVE collapse; SERVE-context positive pressure dominates the shared actor update.",
        },
        "negative_advantage_policy_effect_passed": aggregate_net < 0.0 and hold_action_net > 0.0 and serve_action_net < 0.0,
    }


def cyclewise_collapse_origin(ppo: pd.DataFrame) -> Dict[str, Any]:
    conditional = read_json(H4M_N_ROOT / "04_conditional_policy_discrimination.json")
    probability = {
        int(row["outer_cycle"]): {
            "P_HOLD": row["P_HOLD"]["mean"],
            "P_SERVE": row["P_SERVE"]["mean"],
            "probability_dominant_action": row["probability_dominant_action"],
            "sampled_action_counts": row["sampled_action_counts"],
        }
        for row in conditional["overall_policy_evolution"]
    }
    rows = []
    for cycle in sorted(int(v) for v in ppo["outer_cycle"].unique()):
        cdf = ppo[ppo["outer_cycle"].astype(int) == cycle]
        h = cdf[cdf["target_id"].astype(int) == 0]
        s = cdf[cdf["target_id"].astype(int) == 1]
        target_hold_net = float(h["net_serve_minus_hold_pressure_scalar"].sum())
        target_serve_net = float(s["net_serve_minus_hold_pressure_scalar"].sum())
        total_net = float(cdf["net_serve_minus_hold_pressure_scalar"].sum())
        prob = probability[cycle]
        rows.append(
            {
                "outer_cycle": cycle,
                "target_hold_net_serve_minus_hold_pressure": target_hold_net,
                "target_serve_net_serve_minus_hold_pressure": target_serve_net,
                "shared_total_net_serve_minus_hold_pressure": total_net,
                "target_hold_corrective_pressure_insufficient": target_hold_net >= 0.0,
                "serve_context_abs_pressure_exceeds_hold_context_abs_pressure": abs(target_serve_net) > abs(target_hold_net),
                "P_HOLD": prob["P_HOLD"],
                "P_SERVE": prob["P_SERVE"],
                "probability_dominant_action": prob["probability_dominant_action"],
                "sampled_action_counts": prob["sampled_action_counts"],
            }
        )
    earliest_net_serve = next((row["outer_cycle"] for row in rows if row["shared_total_net_serve_minus_hold_pressure"] > 0.0), None)
    earliest_prob_serve = next((row["outer_cycle"] for row in rows if row["P_SERVE"] > row["P_HOLD"]), None)
    earliest_hold_insufficient = next((row["outer_cycle"] for row in rows if row["target_hold_corrective_pressure_insufficient"]), None)
    return {
        "stage": STAGE,
        "cycle_rows": rows,
        "earliest_net_serve_pressure_dominant_cycle": earliest_net_serve,
        "earliest_probability_SERVE_dominant_cycle": earliest_prob_serve,
        "earliest_target_HOLD_corrective_gradient_insufficient_cycle": earliest_hold_insufficient,
        "collapse_origin": {
            "earliest_collapse_cycle": earliest_prob_serve,
            "first_transition": "cycle 1 is HOLD-dominant, cycle 2 becomes SERVE-dominant",
            "pressure_to_policy_link": "net SERVE pressure becomes positive at cycle 2, the same cycle where P(SERVE) overtakes P(HOLD)",
        },
        "cyclewise_collapse_origin_identified": earliest_net_serve == 2 and earliest_prob_serve == 2,
    }


def root_cause_attribution(
    pressure: Mapping[str, Any],
    branch: Mapping[str, Any],
    interference: Mapping[str, Any],
    negative_effect: Mapping[str, Any],
    cyclewise: Mapping[str, Any],
) -> Dict[str, Any]:
    target = {row["target_id"]: row for row in pressure["by_target_pressure_totals"]}
    hold_net = float(target[0]["net_serve_minus_hold_pressure_sum"])
    serve_net = float(target[1]["net_serve_minus_hold_pressure_sum"])
    shared_net = float(pressure["shared_actor_pressure_total"]["net_serve_minus_hold_pressure_sum"])
    evidence = {
        "target_hold_net_serve_minus_hold_pressure_sum": hold_net,
        "target_serve_net_serve_minus_hold_pressure_sum": serve_net,
        "shared_actor_net_serve_minus_hold_pressure_sum": shared_net,
        "serve_context_abs_pressure_over_hold_context_abs_pressure": branch["target_conditioning_gradient_result"][
            "serve_context_abs_pressure_over_hold_context_abs_pressure"
        ],
        "shared_interference_result": interference["shared_parameter_interference"]["overall_result"],
        "target_hold_negative_credit_result": negative_effect["softmax_coupling_answer"],
        "earliest_collapse_cycle": cyclewise["collapse_origin"]["earliest_collapse_cycle"],
    }
    supported = hold_net < 0.0 and serve_net > 0.0 and abs(serve_net) > abs(hold_net) and shared_net > 0.0
    return {
        "stage": STAGE,
        "root_cause": ROOT_CAUSE if supported else "G_CAUSAL_ATTRIBUTION_STILL_NOT_UNIQUE",
        "submechanisms_supported": {
            "A_SERVE_CONTEXT_GRADIENT_MAGNITUDE_DOMINANCE": serve_net > 0.0 and abs(serve_net) > abs(hold_net),
            "B_SHARED_ACTOR_GRADIENT_INTERFERENCE": interference["shared_parameter_interference"]["conflicting_cycle_count"] > 0
            and shared_net > 0.0,
            "C_HOLD_CONTEXT_NEGATIVE_ADVANTAGE_CANNOT_CREATE_POSITIVE_HOLD_REINFORCEMENT": False,
            "D_TARGET_CONDITIONING_BRANCH_GRADIENT_TOO_WEAK": False,
        },
        "why_not_C": "target=HOLD aggregate PPO pressure is HOLD-corrective (net SERVE-minus-HOLD < 0); negative sampled-HOLD rows create SERVE pressure, but negative sampled-SERVE rows more than offset it.",
        "why_not_D": "target-conditioning branch is active, finite/nonzero from H4M-N, and target=HOLD branch receives corrective pressure; the failure is dominated shared update balance, not absent target-branch usage.",
        "evidence": evidence,
        "root_cause_attribution_passed": supported,
    }


def repair_candidate_comparison(root: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = [
        {
            "candidate": "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_REPAIR",
            "category": "target-conditioned policy parameterization / Actor branch-context specialization",
            "directly_addresses": [
                "SERVE-context gradients dominating shared actor decision head",
                "shared actor interference between target=HOLD and target=SERVE pressure vectors",
            ],
            "minimal_change": "Replace single shared Actor decision head with target-routed actor head bank selected by existing pre-action target one-hot; keep GATv2, Critic, Reward, GAE/PPO, 12D observation, and action contract frozen.",
            "new_information_added": False,
            "reward_or_gae_ppo_change": False,
            "selected": True,
            "rationale": "H4M-O evidence shows target=HOLD pressure is corrective but SERVE-context pressure dominates the shared actor. Isolating action-decision head parameters by target context is the smallest repair that directly removes cross-target head interference without changing reward/PPO semantics.",
        },
        {
            "candidate": "SAMPLING_OR_UPDATE_BALANCING",
            "category": "sampling/update balancing",
            "directly_addresses": ["sample-count and magnitude imbalance"],
            "minimal_change": "Reweight PPO updates by target×action group.",
            "new_information_added": False,
            "reward_or_gae_ppo_change": True,
            "selected": False,
            "rationale": "Could address magnitude imbalance, but changes PPO/update weighting semantics before isolating the observed shared-head interference.",
        },
        {
            "candidate": "REWARD_V2_CHANGE",
            "category": "reward repair",
            "selected": False,
            "rationale": "Not selected: target-conditioned credit direction is already correct for target=HOLD and target=SERVE.",
        },
    ]
    selected = next(row for row in candidates if row.get("selected"))
    return {
        "stage": STAGE,
        "root_cause": root.get("root_cause"),
        "candidates": candidates,
        "selected_minimal_repair": selected["candidate"],
        "block_pending_explicit_repair_selection": False,
        "repair_candidate_comparison_passed": root.get("root_cause") == ROOT_CAUSE,
    }


def repair_contract(root: Mapping[str, Any], candidates: Mapping[str, Any]) -> Dict[str, Any]:
    payload = {
        "stage": STAGE,
        "contract_name": "PV8_H4M_O_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_REPAIR_CONTRACT",
        "root_cause": root.get("root_cause"),
        "selected_repair": candidates.get("selected_minimal_repair"),
        "failing_mechanism": "SERVE-context PPO pressure has larger magnitude than target=HOLD corrective pressure and dominates the shared Actor decision head, producing global SERVE collapse despite correct target-conditioned credit signs.",
        "exact_parameter_update_path": {
            "current_path": "GATv2 embedding + 3D target context -> single shared MAPPOActor head -> HOLD/SERVE logits",
            "failing_shared_path": "target=HOLD and target=SERVE PPO pressures both update the same shared Actor head parameters",
            "target_branch_observation": "target-context one-hot columns are active, but the single shared head still mixes target-context gradients downstream.",
        },
        "proposed_minimal_change": {
            "repair_class": "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION",
            "description": "Use the existing pre-action target one-hot to route each sample to a target-specific Actor decision head. The GATv2 trunk, Critic, Reward V2, GAE/PPO formula, normalization, action mask, 12D observation, and environment stay frozen.",
            "new_features": False,
            "reward_or_ppo_semantics_change": False,
        },
        "what_remains_frozen": [
            "Reward V2",
            "Zero-Loss",
            "K-mask/action contract",
            "12D observation",
            "GATv2 encoder",
            "Critic",
            "GAE/PPO equations and normalization",
            "H4M-B 11-cycle TRAIN44 schedule",
            "environment/data/agents",
        ],
        "expected_acceptance_test": [
            "source-only implementation and py_compile PASS",
            "branch routing uses only existing pre-action target one-hot",
            "target=HOLD samples update only HOLD-target Actor head; target=SERVE samples update only SERVE-target Actor head",
            "no future/post-action leakage",
            "instrumentation captures sample_uid -> target branch -> logits -> PPO pressure",
            "fresh seeds 1/2/3 retraining required; no H4M-N checkpoint continuation",
        ],
        "no_future_leakage": True,
        "fresh_retraining_requirement": {
            "required": True,
            "seeds": [1, 2, 3],
            "schedule": "H4M-B 11-cycle TRAIN44",
            "checkpoint_reuse_allowed": False,
        },
        "exact_next_gate": NEXT_GATE,
    }
    contract_sha = canonical_sha(payload)
    payload["contract_sha256"] = contract_sha
    return payload


def gate_matrix(
    binding: Mapping[str, Any],
    credit: Mapping[str, Any],
    pressure: Mapping[str, Any],
    branch: Mapping[str, Any],
    interference: Mapping[str, Any],
    negative: Mapping[str, Any],
    cyclewise: Mapping[str, Any],
    root: Mapping[str, Any],
    candidates: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_pass": binding.get("authoritative_binding_passed") is True,
        "target_action_credit_fully_decomposed": credit.get("target_action_credit_decomposition_passed") is True,
        "actor_gradient_pressure_quantified": pressure.get("actor_gradient_pressure_decomposition_passed") is True,
        "shared_vs_target_branch_separated": branch.get("branch_gradient_passed") is True,
        "gradient_interference_assessed": interference.get("gradient_interference_audit_passed") is True,
        "negative_advantage_policy_effect_answered": negative.get("negative_advantage_policy_effect_passed") is True,
        "cyclewise_collapse_origin_identified": cyclewise.get("cyclewise_collapse_origin_identified") is True,
        "root_cause_supported": root.get("root_cause_attribution_passed") is True,
        "repair_selected": candidates.get("block_pending_explicit_repair_selection") is False
        and candidates.get("selected_minimal_repair") == SELECTED_REPAIR,
        "repair_contract_sha_created": bool(contract.get("contract_sha256")),
        "no_training_repair_test6_push": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": root.get("root_cause") if passed else "H4M_O_OUTCOME_REVIEW_BLOCKED",
        "selected_minimal_repair": candidates.get("selected_minimal_repair") if passed else None,
        "repair_contract_sha256": contract.get("contract_sha256") if passed else None,
        "exact_next_gate": contract.get("exact_next_gate") if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "final_flags": {
            "training_executed": False,
            "repair_implemented": False,
            "reward_gae_ppo_modified": False,
            "environment_expanded": False,
            "validation_or_test6_executed": False,
            "github_push_performed": False,
        },
    }


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    files: Dict[str, str] = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(artifact_root))] = str(path)
    return {
        "stage": STAGE,
        "created_at": kst_now(),
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "selected_minimal_repair": gate.get("selected_minimal_repair"),
        "repair_contract_sha256": gate.get("repair_contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": files,
        "output_sha256": {rel: sha256_file(Path(path)) for rel, path in files.items()},
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def final_report(
    binding: Mapping[str, Any],
    credit: Mapping[str, Any],
    pressure: Mapping[str, Any],
    branch: Mapping[str, Any],
    interference: Mapping[str, Any],
    negative: Mapping[str, Any],
    cyclewise: Mapping[str, Any],
    root: Mapping[str, Any],
    contract: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    return f"""# H4M-O Actor Target-Conditioned Retraining Outcome Review

gate = {gate['gate']}
source_commit = {binding['source_provenance']['h4m_o_source_git_commit']}
decision = {gate['decision']}
selected_minimal_repair = {gate['selected_minimal_repair']}
repair_contract_sha256 = {gate['repair_contract_sha256']}
exact_next_gate = {gate['exact_next_gate']}

## Target×action credit totals

```json
{json.dumps(credit['aggregate_target_action_totals'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Actor gradient pressure

```json
{json.dumps(pressure['by_target_pressure_totals'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Shared vs target branch result

```json
{json.dumps({'shared': branch['shared_actor_gradient_result'], 'target': branch['target_conditioning_gradient_result']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Gradient interference

```json
{json.dumps(interference['shared_parameter_interference'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Why target=HOLD negative credit fails to prevent SERVE collapse

```json
{json.dumps(negative['softmax_coupling_answer'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Cyclewise collapse origin

```json
{json.dumps(cyclewise['collapse_origin'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Root cause

```json
{json.dumps(root, ensure_ascii=False, indent=2, default=jsonable)}
```

## Repair contract

```json
{json.dumps(contract, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no training, no repair implementation, no Reward/GAE/PPO changes, no environment expansion, no validation/TEST6, no GitHub push.
"""


def write_block_outputs(artifact_root: Path, created_at: str, binding: Mapping[str, Any], reason: str) -> None:
    gate = {
        "stage": STAGE,
        "gate": BLOCK_GATE,
        "decision": "H4M_O_OUTCOME_REVIEW_BLOCKED",
        "block_reason": reason,
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "01_authoritative_binding.json", binding)
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    for name in REQUIRED_ARTIFACTS[1:10]:
        write_json(artifact_root / name, empty)
    write_json(artifact_root / "11_gate_matrix.json", gate)
    (artifact_root / "final_report.md").write_text(f"# H4M-O\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-O] artifact root: {artifact_root}")
    print(f"[H4M-O] gate: {BLOCK_GATE}")
    print(f"[H4M-O] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_o_actor_target_conditioned_retraining_outcome_review_next_decision_{stamp}"
    provenance = source_provenance(created_at)
    compile_static = py_compile_audit()
    binding = authoritative_binding(created_at, provenance, compile_static)
    if not binding["authoritative_binding_passed"]:
        write_block_outputs(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return

    frames = load_h4mn_frames()
    rollout = prepare_rollout_rows(frames)
    ppo = prepare_ppo_rows(frames, rollout)

    artifact_root.mkdir(parents=True, exist_ok=True)
    credit = target_action_credit_decomposition(artifact_root, rollout, ppo)
    pressure = actor_gradient_pressure_decomposition(artifact_root, ppo)
    branch = shared_vs_target_branch_gradient(pressure)
    interference = gradient_interference_audit(artifact_root, ppo)
    negative = negative_advantage_policy_effect(ppo, rollout)
    cyclewise = cyclewise_collapse_origin(ppo)
    root = root_cause_attribution(pressure, branch, interference, negative, cyclewise)
    candidates = repair_candidate_comparison(root)
    contract = repair_contract(root, candidates)
    gate = gate_matrix(binding, credit, pressure, branch, interference, negative, cyclewise, root, candidates, contract)
    report = final_report(binding, credit, pressure, branch, interference, negative, cyclewise, root, contract, gate)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_target_action_credit_decomposition.json": credit,
        "03_actor_gradient_pressure_decomposition.json": pressure,
        "04_shared_vs_target_branch_gradient.json": branch,
        "05_gradient_interference_audit.json": interference,
        "06_negative_advantage_policy_effect.json": negative,
        "07_cyclewise_collapse_origin.json": cyclewise,
        "08_root_cause_attribution.json": root,
        "09_repair_candidate_comparison.json": candidates,
        "10_h4m_o_repair_contract.json": contract,
        "11_gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    print(f"[H4M-O] artifact root: {artifact_root}")
    print(f"[H4M-O] gate: {gate['gate']}")
    print(f"[H4M-O] source_commit: {binding['source_provenance']['h4m_o_source_git_commit']}")
    print(f"[H4M-O] shared_net_serve_minus_hold={branch['shared_actor_gradient_result']['net_serve_minus_hold_pressure_sum']}")
    print(
        "[H4M-O] target_hold_net_serve_minus_hold="
        f"{branch['target_conditioning_gradient_result']['target_hold_branch_net_serve_minus_hold_pressure_sum']}"
    )
    print(
        "[H4M-O] target_serve_net_serve_minus_hold="
        f"{branch['target_conditioning_gradient_result']['target_serve_branch_net_serve_minus_hold_pressure_sum']}"
    )
    print(f"[H4M-O] earliest_collapse_cycle={cyclewise['collapse_origin']['earliest_collapse_cycle']}")
    print(f"[H4M-O] root_cause={gate['decision']}")
    print(f"[H4M-O] selected_repair={gate['selected_minimal_repair']}")
    print(f"[H4M-O] repair_contract_sha256={gate['repair_contract_sha256']}")
    print(f"[H4M-O] next_gate={gate['exact_next_gate']}")
    print("[H4M-O] STOP: no training, no repair implementation, no TEST6, github_push=false")


if __name__ == "__main__":
    main()
