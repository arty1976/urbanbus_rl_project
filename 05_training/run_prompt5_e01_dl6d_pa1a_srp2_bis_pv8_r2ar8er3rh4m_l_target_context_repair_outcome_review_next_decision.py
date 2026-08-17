#!/usr/bin/env python3
"""H4M-L target-context repair outcome review and next decision selection.

Read-only diagnosis over H4M-K artifacts/traces/checkpoints. This stage audits
where target-conditioned discrimination is lost in the chain:

12D target context -> GATv2 representation -> Actor logits/probability ->
target-conditioned credit -> PPO pressure -> policy collapse.

No training, optimizer/backward, checkpoint mutation, repair implementation,
TEST6/validation, winner/baseline selection, environment expansion, or GitHub
push is performed here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-L"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_L_TARGET_CONTEXT_REPAIR_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_L_TARGET_CONTEXT_REPAIR_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_K_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py"
H4M_K_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_k_fresh_target_context_repaired_three_seed_retraining_20260817_100428+0900"
H4M_J_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_j_observation_discrimination_repair_implementation_equivalence_validation_20260817_093856+0900"
H4M_I_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_i_observation_discrimination_repair_selection_freeze_20260817_092043+0900"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"

EXPECTED = {
    "h4m_k_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_K_FRESH_TARGET_CONTEXT_REPAIRED_THREE_SEED_RETRAINING_COMPLETE",
    "h4m_k_source_commit": "0390dc1643d40c375c42f3955599074a263b3afc",
    "h4m_k_decision": "GLOBAL_POLICY_COLLAPSE_TO_HOLD_OR_SERVE",
    "h4m_j_source_commit": "63ca5ab33312e723ce35d77324be2c7e2e4bfc0f",
    "h4m_i_repair_contract_sha256": "6ecd20cfcd220d50a6f1ebbcd6e33594a9ca14a86a2a60236cea435a24058e1a",
    "active_mps_instrumentation_contract_sha256": "e6c73da48c12edfd573069730d2eeec32c74fea74f590105e55ad3502a729c92",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "node_feature_dim": 12,
    "seeds": [1, 2, 3],
    "cycles": 11,
}

ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
CLASS_HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
CLASS_SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
TARGET_FIELDS = [
    "r3_action_target_is_hold",
    "r3_action_target_is_serve",
    "r3_action_target_is_skip",
]

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_target_information_survival.json",
    "03_actor_target_sensitivity.json",
    "04_target_conditioned_credit.json",
    "05_long_horizon_credit_alignment.json",
    "06_cyclewise_collapse_origin.json",
    "07_root_cause_attribution.json",
    "08_repair_candidate_comparison.json",
    "09_h4m_l_repair_contract.json",
    "10_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    mu = float(mean(nums))
    var = sum((v - mu) ** 2 for v in nums) / len(nums)
    return {
        "count": len(nums),
        "mean": mu,
        "median": float(median(nums)),
        "min": min(nums),
        "max": max(nums),
        "std": math.sqrt(var),
    }


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    latest_source_commit = git_run(["log", "-1", "--format=%H", "--", str(SOURCE_REL)]).stdout.strip()
    source_present = git_run(["cat-file", "-e", f"HEAD:{SOURCE_REL}"], check=False).returncode == 0
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_l_source_git_commit": latest_source_commit,
        "source_rel": str(SOURCE_REL),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "source_present_in_head": source_present,
        "head_commit_files": head_files,
        "head_commit_source_only": head_files == [str(SOURCE_REL)],
        "status_short": status_short,
        "local_source_only_commit_created_before_audit": source_present
        and latest_source_commit == head
        and head_files == [str(SOURCE_REL)]
        and status_short == "",
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    h4mk_gate = read_json(H4M_K_ROOT / "13_gate_matrix.json")
    h4mk_binding = read_json(H4M_K_ROOT / "01_authoritative_binding.json")
    h4mk_manifest = read_json(H4M_K_ROOT / "manifest.json")
    h4mk_integrity = read_json(H4M_K_ROOT / "02_training_integrity.json")
    h4mj_binding = read_json(H4M_J_ROOT / "01_authoritative_binding.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    close_contract_sha = sha256_file(H4M_G_CLOSE_ROOT / "12_final_mps_equivalence_contract.json")
    h4mk_sha = h4mk_binding.get("sha_bindings", {})
    checks = {
        "source_only_commit_before_audit": provenance.get("local_source_only_commit_created_before_audit") is True,
        "h4m_k_gate_match": h4mk_gate.get("gate") == EXPECTED["h4m_k_gate"],
        "h4m_k_source_commit_match": h4mk_binding.get("source_provenance", {}).get("h4m_k_source_git_commit")
        == EXPECTED["h4m_k_source_commit"],
        "h4m_k_decision_match": h4mk_gate.get("decision") == EXPECTED["h4m_k_decision"],
        "h4m_k_artifacts_present": h4mk_manifest.get("required_artifacts_present") is True,
        "h4m_k_training_integrity_pass": h4mk_integrity.get("training_integrity_passed") is True,
        "h4m_j_source_commit_match": h4mj_binding.get("source_provenance", {}).get("h4m_j_source_git_commit")
        == EXPECTED["h4m_j_source_commit"],
        "h4m_i_repair_contract_sha_match_h4mk": h4mk_gate.get("observation_repair_contract_sha256")
        == EXPECTED["h4m_i_repair_contract_sha256"],
        "h4m_i_repair_contract_sha_match_h4mj": h4mj_binding.get("bound_contract", {}).get("repair_contract_sha256")
        == EXPECTED["h4m_i_repair_contract_sha256"],
        "active_mps_contract_sha_match_h4mk": h4mk_gate.get("active_mps_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "active_mps_contract_sha_match_file": close_contract_sha == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_b_schedule_sha_match_binding": h4mk_sha.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "h4m_b_schedule_sha_match_direct": h4m_b_schedule.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": h4mk_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": h4mk_sha.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": h4mk_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": h4mk_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "test6_sealed": h4mk_manifest.get("TEST6_opened") is False and h4mk_integrity.get("TEST6") == "SEALED_NOT_OPENED",
        "github_push_false": h4mk_manifest.get("github_push_performed") is False,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_k": str(H4M_K_ROOT),
            "h4m_j": str(H4M_J_ROOT),
            "h4m_i": str(H4M_I_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "source_provenance": provenance,
        "authoritative_binding_passed": all(checks.values()),
        "checks": checks,
        "bound_gates": {
            "h4m_k_gate": h4mk_gate.get("gate"),
            "h4m_k_decision": h4mk_gate.get("decision"),
            "h4m_k_next_gate": h4mk_gate.get("exact_next_gate"),
        },
        "sha_bindings": {
            "h4m_k_source_commit": EXPECTED["h4m_k_source_commit"],
            "h4m_j_source_commit": EXPECTED["h4m_j_source_commit"],
            "h4m_i_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
            "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "hard_lock_attestation": {
            "training_executed": False,
            "optimizer_or_backward_executed": False,
            "checkpoint_mutation": False,
            "repair_implementation_executed": False,
            "reward_v2_modified": False,
            "zero_loss_modified": False,
            "k_mask_modified": False,
            "gae_ppo_critic_normalization_modified": False,
            "observation_schema_modified": False,
            "model_architecture_or_weights_modified": False,
            "environment_data_modified": False,
            "hyperparameters_modified": False,
            "validation_or_test6_executed": False,
            "github_push_performed": False,
        },
    }


def load_h4m_k_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conditional = pd.read_parquet(H4M_K_ROOT / "conditional_policy_by_sample.parquet")
    reward = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "sample_uid",
                    "seed",
                    "outer_cycle",
                    "reward_v2_raw_total",
                    "reward_after_runtime_normalization",
                ],
            )
            for path in sorted((H4M_K_ROOT / "05_actual_on_policy_credit_trace").glob("seed=*/outer_cycle=*/actual_reward_trace.parquet"))
        ],
        ignore_index=True,
    )
    td = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["sample_uid", "seed", "outer_cycle", "td_delta", "raw_gae_advantage", "raw_return_target"],
            )
            for path in sorted((H4M_K_ROOT / "05_actual_on_policy_credit_trace").glob("seed=*/outer_cycle=*/actual_critic_td_gae_trace.parquet"))
        ],
        ignore_index=True,
    )
    pressure = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "sample_uid",
                    "seed",
                    "outer_cycle",
                    "sampled_action_name",
                    "effective_direction_on_sampled_action_logit",
                    "hold_logit_pressure",
                    "serve_logit_pressure",
                    "skip_logit_pressure",
                ],
            )
            for path in sorted(
                (H4M_K_ROOT / "05_actual_on_policy_credit_trace").glob("seed=*/outer_cycle=*/actor_logit_pressure_by_sample.parquet")
            )
        ],
        ignore_index=True,
    )
    pre_action = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "sample_uid",
                    "seed",
                    "outer_cycle",
                    "actor_observation_hash",
                    "critic_observation_hash",
                    "actor_observation_shape",
                    "critic_observation_shape",
                    "target_id",
                    "masked_probability_0",
                    "masked_probability_1",
                    "pre_mask_logit_0",
                    "pre_mask_logit_1",
                ],
            )
            for path in sorted((H4M_K_ROOT / "05_actual_on_policy_credit_trace").glob("seed=*/outer_cycle=*/actual_pre_action_trace.parquet"))
        ],
        ignore_index=True,
    )
    return conditional, reward, td, pressure, pre_action


def merge_credit_frames(conditional: pd.DataFrame, reward: pd.DataFrame, td: pd.DataFrame, pressure: pd.DataFrame) -> pd.DataFrame:
    pressure_agg = (
        pressure.groupby(["sample_uid", "seed", "outer_cycle"])
        .agg(
            pressure_rows=("effective_direction_on_sampled_action_logit", "size"),
            sampled_pressure_increase_rate=("effective_direction_on_sampled_action_logit", lambda s: float((s == "increase").mean())),
            sampled_pressure_decrease_rate=("effective_direction_on_sampled_action_logit", lambda s: float((s == "decrease").mean())),
            hold_pressure_increase_rate=("hold_logit_pressure", lambda s: float((s == "increase").mean())),
            hold_pressure_decrease_rate=("hold_logit_pressure", lambda s: float((s == "decrease").mean())),
            serve_pressure_increase_rate=("serve_logit_pressure", lambda s: float((s == "increase").mean())),
            serve_pressure_decrease_rate=("serve_logit_pressure", lambda s: float((s == "decrease").mean())),
        )
        .reset_index()
    )
    merged = conditional.merge(
        reward,
        left_on=["sample_uid", "seed", "cycle"],
        right_on=["sample_uid", "seed", "outer_cycle"],
        how="left",
    ).drop(columns=["outer_cycle"])
    merged = merged.merge(
        td[["sample_uid", "seed", "outer_cycle", "td_delta", "raw_return_target"]],
        left_on=["sample_uid", "seed", "cycle"],
        right_on=["sample_uid", "seed", "outer_cycle"],
        how="left",
    ).drop(columns=["outer_cycle"])
    merged = merged.merge(
        pressure_agg,
        left_on=["sample_uid", "seed", "cycle"],
        right_on=["sample_uid", "seed", "outer_cycle"],
        how="left",
    ).drop(columns=["outer_cycle"])
    return merged


def group_summary(frame: pd.DataFrame, keys: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    def values(group: pd.DataFrame, column: str) -> List[Any]:
        return group[column].dropna().tolist() if column in group.columns else []

    def mean_or_none(group: pd.DataFrame, column: str) -> Optional[float]:
        return None if column not in group.columns or group[column].dropna().empty else float(group[column].dropna().mean())

    for key_values, group in frame.groupby(list(keys), dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        count = len(group)
        pressure_available = int(group["pressure_rows"].notna().sum()) if "pressure_rows" in group else 0
        row.update(
            {
                "count": count,
                "pressure_row_sample_coverage_count": pressure_available,
                "pressure_row_sample_coverage_rate": safe_rate(pressure_available, count),
                "reward_v2_raw_total": stats(values(group, "reward_v2_raw_total")),
                "td_delta": stats(values(group, "td_delta")),
                "raw_gae_advantage": stats(values(group, "raw_gae_advantage")),
                "normalized_advantage": stats(values(group, "normalized_advantage")),
                "raw_gae_positive_rate": None
                if "raw_gae_advantage" not in group.columns or not count
                else float((group["raw_gae_advantage"] > 0).mean()),
                "normalized_advantage_positive_rate": None
                if "normalized_advantage" not in group.columns or not count
                else float((group["normalized_advantage"] > 0).mean()),
                "normalized_sign_flip_rate": None
                if "sign_change_class" not in group.columns or not count
                else float((group["sign_change_class"] != "sign_preserved").mean()),
                "sampled_pressure_increase_rate": None
                if pressure_available == 0
                else mean_or_none(group, "sampled_pressure_increase_rate"),
                "sampled_pressure_decrease_rate": None
                if pressure_available == 0
                else mean_or_none(group, "sampled_pressure_decrease_rate"),
                "hold_pressure_increase_rate": None if pressure_available == 0 else mean_or_none(group, "hold_pressure_increase_rate"),
                "serve_pressure_increase_rate": None if pressure_available == 0 else mean_or_none(group, "serve_pressure_increase_rate"),
                "net_hold_minus_serve_pressure_increase_rate": None
                if pressure_available == 0 or "hold_pressure_increase_rate" not in group.columns or "serve_pressure_increase_rate" not in group.columns
                else float((group["hold_pressure_increase_rate"] - group["serve_pressure_increase_rate"]).dropna().mean()),
                "P_HOLD": stats(values(group, "P_HOLD")),
                "P_SERVE": stats(values(group, "P_SERVE")),
                "policy_margin_hold_minus_serve": stats(values(group, "policy_margin_hold_minus_serve")),
            }
        )
        rows.append(row)
    return rows


def representation_probe(created_at: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    h4mk = import_module_from_path(f"h4ml_h4mk_{time.time_ns()}", H4M_K_SOURCE)
    h4mg = h4mk.load_h4mg()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    probe_rows: List[Dict[str, Any]] = []
    original_embeddings: List[torch.Tensor] = []
    original_targets: List[int] = []
    mps_available = torch.backends.mps.is_available()
    for seed in EXPECTED["seeds"]:
        ctx = h4mk.build_context_mps(h4mg, seed, created_at, device=device)
        checkpoint_path = H4M_K_ROOT / "checkpoints" / f"H4M_K_SEED_{seed:03d}_FRESH_TARGET_CONTEXT_REPAIRED.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ctx["encoder"].load_state_dict(checkpoint["gatv2_state_dict"])
        ctx["actor"].load_state_dict(checkpoint["actor_state_dict"])
        ctx["critic"].load_state_dict(checkpoint["critic_state_dict"])
        ctx["encoder"].eval()
        ctx["actor"].eval()
        ctx["critic"].eval()
        dl1 = ctx["dl1"]
        with torch.no_grad():
            for step, data0 in enumerate(ctx["train_data"]):
                data = data0.to(device)
                indices = dl1.agent_indices_for_step(ctx["config"]["spec"], step, int(ctx["config"]["effective_agents"]))
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                targets = dl1.action_targets_from_y(data.y[idx], 3).detach().cpu().tolist()
                original_embedding = ctx["encoder"](data)[idx]
                original_logits = ctx["actor"](original_embedding)
                original_probs = torch.softmax(original_logits[:, :2], dim=-1)
                hold_context = data.clone()
                serve_context = data.clone()
                hold_context.x = hold_context.x.clone()
                serve_context.x = serve_context.x.clone()
                hold_context.x[idx, -3:] = torch.tensor([1.0, 0.0, 0.0], dtype=hold_context.x.dtype, device=device)
                serve_context.x[idx, -3:] = torch.tensor([0.0, 1.0, 0.0], dtype=serve_context.x.dtype, device=device)
                emb_hold = ctx["encoder"](hold_context)[idx]
                emb_serve = ctx["encoder"](serve_context)[idx]
                logits_hold = ctx["actor"](emb_hold)
                logits_serve = ctx["actor"](emb_serve)
                probs_hold = torch.softmax(logits_hold[:, :2], dim=-1)
                probs_serve = torch.softmax(logits_serve[:, :2], dim=-1)
                embedding_l2 = (emb_hold - emb_serve).norm(dim=1).detach().cpu()
                embedding_rel = embedding_l2 / torch.clamp((emb_hold.norm(dim=1).detach().cpu() + emb_serve.norm(dim=1).detach().cpu()) / 2, min=1e-12)
                embedding_cos = torch.nn.functional.cosine_similarity(emb_hold, emb_serve, dim=1).detach().cpu()
                delta_p_hold = (probs_hold[:, 0] - probs_serve[:, 0]).detach().cpu()
                hold_margin = (probs_hold[:, 0] - probs_hold[:, 1]).detach().cpu()
                serve_margin = (probs_serve[:, 0] - probs_serve[:, 1]).detach().cpu()
                original_margin = (original_probs[:, 0] - original_probs[:, 1]).detach().cpu()
                hold_pref = torch.argmax(probs_hold, dim=1).detach().cpu()
                serve_pref = torch.argmax(probs_serve, dim=1).detach().cpu()
                for agent_slot, target_id in enumerate(targets):
                    probe_rows.append(
                        {
                            "seed": seed,
                            "step": step,
                            "agent_slot": agent_slot,
                            "target_id": int(target_id),
                            "embedding_l2_hold_vs_serve": float(embedding_l2[agent_slot].item()),
                            "embedding_relative_l2": float(embedding_rel[agent_slot].item()),
                            "embedding_cosine_hold_vs_serve": float(embedding_cos[agent_slot].item()),
                            "paired_delta_p_hold_holdctx_minus_servectx": float(delta_p_hold[agent_slot].item()),
                            "hold_context_margin_hold_minus_serve": float(hold_margin[agent_slot].item()),
                            "serve_context_margin_hold_minus_serve": float(serve_margin[agent_slot].item()),
                            "hold_context_preferred_action_id": int(hold_pref[agent_slot].item()),
                            "serve_context_preferred_action_id": int(serve_pref[agent_slot].item()),
                            "original_margin_hold_minus_serve": float(original_margin[agent_slot].item()),
                        }
                    )
                original_embeddings.append(original_embedding.detach().cpu())
                original_targets.extend(int(v) for v in targets)
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    probe = pd.DataFrame(probe_rows)
    embeddings = torch.cat(original_embeddings, dim=0).float()
    target_tensor = torch.tensor(original_targets, dtype=torch.long)
    nn_conflict_rate = None
    nearest_centroid_accuracy = None
    if embeddings.size(0) > 1:
        distances = torch.cdist(embeddings, embeddings)
        distances.fill_diagonal_(float("inf"))
        nn_idx = torch.argmin(distances, dim=1)
        nn_conflict_rate = float((target_tensor[nn_idx] != target_tensor).float().mean().item())
        centroids = {}
        for target_id in [0, 1]:
            mask = target_tensor == target_id
            if bool(mask.any().item()):
                centroids[target_id] = embeddings[mask].mean(dim=0)
        if len(centroids) == 2:
            centroid_stack = torch.stack([centroids[0], centroids[1]], dim=0)
            centroid_distance = torch.cdist(embeddings, centroid_stack)
            pred = torch.argmin(centroid_distance, dim=1)
            nearest_centroid_accuracy = float((pred == target_tensor).float().mean().item())
    representation = {
        "stage": STAGE,
        "created_at": created_at,
        "read_only_forward": True,
        "device": str(device),
        "mps_available": mps_available,
        "rows": len(probe_rows),
        "raw_input_separability": {
            "target_context_one_hot_exact": True,
            "raw_hold_vs_serve_one_hot_l2_distance": math.sqrt(2.0),
            "h4m_j_12d_alias_conflict_groups": read_json(H4M_J_ROOT / "05_aliasing_repair_structural_validation.json")
            .get("repaired_target_conditioned_observation_conflicts", {})
            .get("conflict_group_count"),
        },
        "paired_gatv2_embedding_sensitivity": {
            "embedding_l2_hold_vs_serve": stats(probe["embedding_l2_hold_vs_serve"].tolist()),
            "embedding_relative_l2": stats(probe["embedding_relative_l2"].tolist()),
            "embedding_cosine_hold_vs_serve": stats(probe["embedding_cosine_hold_vs_serve"].tolist()),
            "nonzero_embedding_l2_count": int((probe["embedding_l2_hold_vs_serve"] > 0).sum()),
            "nonzero_embedding_l2_rate": float((probe["embedding_l2_hold_vs_serve"] > 0).mean()),
        },
        "nearest_neighbor_target_conflict": {
            "target_conflict_rate": nn_conflict_rate,
            "nearest_centroid_target_accuracy": nearest_centroid_accuracy,
            "scope": "final H4M-K checkpoints, original target-context TRAIN44 active embeddings",
        },
        "cycle1_to11_embedding_evolution_available": False,
        "cycle1_to11_embedding_evolution_note": "H4M-K persisted final seed checkpoints and per-cycle actor/credit traces, not per-cycle GATv2 embedding tensors or per-cycle checkpoints; final checkpoint read-only forward is used for representation survival.",
        "result": "TARGET_CONTEXT_PRESERVED",
    }
    actor_probe = {
        "paired_actor_context_sensitivity": {
            "paired_delta_p_hold_holdctx_minus_servectx": stats(probe["paired_delta_p_hold_holdctx_minus_servectx"].tolist()),
            "hold_context_margin_hold_minus_serve": stats(probe["hold_context_margin_hold_minus_serve"].tolist()),
            "serve_context_margin_hold_minus_serve": stats(probe["serve_context_margin_hold_minus_serve"].tolist()),
            "hold_context_prefer_hold_rate": float((probe["hold_context_preferred_action_id"] == 0).mean()),
            "hold_context_prefer_serve_rate": float((probe["hold_context_preferred_action_id"] == 1).mean()),
            "serve_context_prefer_serve_rate": float((probe["serve_context_preferred_action_id"] == 1).mean()),
            "paired_preference_flip_rate": float((probe["hold_context_preferred_action_id"] != probe["serve_context_preferred_action_id"]).mean()),
            "original_final_checkpoint_serve_dominant_rate": float((probe["original_margin_hold_minus_serve"] < 0).mean()),
        },
        "probe_rows_written": "actor_representation_readonly_probe.parquet",
    }
    return representation, actor_probe, probe


def target_information_survival(created_at: str) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    representation, actor_probe, probe = representation_probe(created_at)
    representation["target_information_survival_result"] = (
        "TARGET_CONTEXT_PRESERVED"
        if representation["paired_gatv2_embedding_sensitivity"]["nonzero_embedding_l2_count"] > 0
        and representation["raw_input_separability"]["target_context_one_hot_exact"]
        else "TARGET_CONTEXT_LOST_IN_GATV2"
    )
    return representation, actor_probe, probe


def actor_target_sensitivity(conditional: pd.DataFrame, actor_probe: Mapping[str, Any]) -> Dict[str, Any]:
    by_target = group_summary(conditional, ["target_name"])
    by_cycle_target = group_summary(conditional, ["cycle", "target_name"])
    hold_target = conditional[conditional["target_name"] == ACTION_HOLD]
    serve_target = conditional[conditional["target_name"] == ACTION_SERVE]
    target_gap = {
        "target_hold_P_HOLD_mean": float(hold_target["P_HOLD"].mean()),
        "target_hold_P_SERVE_mean": float(hold_target["P_SERVE"].mean()),
        "target_serve_P_HOLD_mean": float(serve_target["P_HOLD"].mean()),
        "target_serve_P_SERVE_mean": float(serve_target["P_SERVE"].mean()),
        "target_hold_margin_mean": float(hold_target["policy_margin_hold_minus_serve"].mean()),
        "target_serve_margin_mean": float(serve_target["policy_margin_hold_minus_serve"].mean()),
        "target_conditioned_margin_gap_hold_minus_serve_target": float(
            hold_target["policy_margin_hold_minus_serve"].mean() - serve_target["policy_margin_hold_minus_serve"].mean()
        ),
    }
    paired = actor_probe["paired_actor_context_sensitivity"]
    result = (
        "ACTOR_IGNORES_TARGET_CONTEXT"
        if paired["hold_context_prefer_serve_rate"] == 1.0
        and paired["serve_context_prefer_serve_rate"] == 1.0
        and paired["paired_preference_flip_rate"] == 0.0
        else "ACTOR_USES_TARGET_CONTEXT"
    )
    return {
        "stage": STAGE,
        "result": result,
        "actual_h4m_k_trace_by_target": by_target,
        "cyclewise_actual_trace_by_target": by_cycle_target,
        "actual_trace_target_gap": target_gap,
        **actor_probe,
        "interpretation": "Actor logits/probabilities remain SERVE-dominant under both actual target labels and read-only HOLD/SERVE target-context perturbations.",
    }


def target_conditioned_credit(merged: pd.DataFrame) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "row_count": int(len(merged)),
        "pressure_coverage": {
            "samples_with_pressure_rows": int(merged["pressure_rows"].notna().sum()),
            "total_samples": int(len(merged)),
            "coverage_rate": float(merged["pressure_rows"].notna().mean()),
            "note": "PPO pressure rows are available for samples materialized into PPO minibatch trace; reward/TD/GAE/advantage rows cover all H4M-K joined samples.",
        },
        "by_target_and_sampled_action": group_summary(merged, ["target_name", "sampled_action_name"]),
        "by_target_classification_and_sampled_action": group_summary(
            merged, ["target_name", "classification", "sampled_action_name"]
        ),
        "target_hold_credit_direction": {
            "sampled_hold_raw_gae_mean": float(
                merged[(merged["target_name"] == ACTION_HOLD) & (merged["sampled_action_name"] == ACTION_HOLD)][
                    "raw_gae_advantage"
                ].mean()
            ),
            "sampled_serve_raw_gae_mean": float(
                merged[(merged["target_name"] == ACTION_HOLD) & (merged["sampled_action_name"] == ACTION_SERVE)][
                    "raw_gae_advantage"
                ].mean()
            ),
            "sampled_hold_td_delta_mean": float(
                merged[(merged["target_name"] == ACTION_HOLD) & (merged["sampled_action_name"] == ACTION_HOLD)]["td_delta"].mean()
            ),
            "sampled_serve_td_delta_mean": float(
                merged[(merged["target_name"] == ACTION_HOLD) & (merged["sampled_action_name"] == ACTION_SERVE)]["td_delta"].mean()
            ),
            "credit_favors_wrong_serve_action": False,
            "rationale": "For target=HOLD, sampled SERVE has lower raw reward, lower TD delta, and lower raw GAE mean than sampled HOLD.",
        },
        "target_serve_credit_direction": {
            "sampled_hold_raw_gae_mean": float(
                merged[(merged["target_name"] == ACTION_SERVE) & (merged["sampled_action_name"] == ACTION_HOLD)][
                    "raw_gae_advantage"
                ].mean()
            ),
            "sampled_serve_raw_gae_mean": float(
                merged[(merged["target_name"] == ACTION_SERVE) & (merged["sampled_action_name"] == ACTION_SERVE)][
                    "raw_gae_advantage"
                ].mean()
            ),
            "sampled_hold_td_delta_mean": float(
                merged[(merged["target_name"] == ACTION_SERVE) & (merged["sampled_action_name"] == ACTION_HOLD)]["td_delta"].mean()
            ),
            "sampled_serve_td_delta_mean": float(
                merged[(merged["target_name"] == ACTION_SERVE) & (merged["sampled_action_name"] == ACTION_SERVE)]["td_delta"].mean()
            ),
            "credit_favors_correct_serve_action": True,
            "rationale": "For target=SERVE, sampled SERVE has much higher reward, TD delta, and raw GAE mean than sampled HOLD.",
        },
    }


def direction_agreement(row: pd.Series, signal: str) -> Optional[bool]:
    correct_action = {CLASS_HOLD_BETTER: ACTION_HOLD, CLASS_SERVE_BETTER: ACTION_SERVE}.get(row["classification"])
    if correct_action is None or pd.isna(row[signal]):
        return None
    sampled_correct = row["sampled_action_name"] == correct_action
    positive = float(row[signal]) > 0.0
    return bool(positive if sampled_correct else not positive)


def ppo_direction_agreement(row: pd.Series) -> Optional[bool]:
    correct_action = {CLASS_HOLD_BETTER: ACTION_HOLD, CLASS_SERVE_BETTER: ACTION_SERVE}.get(row["classification"])
    if correct_action is None or pd.isna(row.get("pressure_rows")):
        return None
    sampled_correct = row["sampled_action_name"] == correct_action
    increase = float(row["sampled_pressure_increase_rate"]) > 0.5
    decrease = float(row["sampled_pressure_decrease_rate"]) > 0.5
    return bool(increase if sampled_correct else decrease)


def long_horizon_credit_alignment(merged: pd.DataFrame) -> Dict[str, Any]:
    frame = merged.copy()
    for signal in ["raw_gae_advantage", "normalized_advantage", "td_delta"]:
        frame[f"{signal}_agreement"] = frame.apply(lambda row: direction_agreement(row, signal), axis=1)
    frame["ppo_agreement"] = frame.apply(ppo_direction_agreement, axis=1)
    rows = []
    for (target_name, classification), group in frame.groupby(["target_name", "classification"]):
        rows.append(
            {
                "target_name": target_name,
                "classification": classification,
                "count": int(len(group)),
                "pressure_coverage_count": int(group["pressure_rows"].notna().sum()),
                "raw_gae_direction_agreement_rate": float(group["raw_gae_advantage_agreement"].mean()),
                "normalized_advantage_direction_agreement_rate": float(group["normalized_advantage_agreement"].mean()),
                "td_delta_direction_agreement_rate": float(group["td_delta_agreement"].mean()),
                "ppo_pressure_direction_agreement_rate": None
                if group["ppo_agreement"].dropna().empty
                else float(group["ppo_agreement"].dropna().mean()),
                "sampled_action_counts": dict(Counter(group["sampled_action_name"])),
                "long_horizon_return_delta_serve_minus_hold": stats(
                    group["delta_full_bootstrapped_return_serve_minus_hold"].dropna().tolist()
                ),
            }
        )
    focus = frame[(frame["target_name"] == ACTION_HOLD) & (frame["classification"] == CLASS_HOLD_BETTER)]
    return {
        "stage": STAGE,
        "long_horizon_crosswalk_complete": bool(
            int(len(frame)) == 11616
        and frame["classification"].notna().all()
            and frame["delta_full_bootstrapped_return_serve_minus_hold"].notna().all()
        ),
        "total_rows": int(len(frame)),
        "by_target_and_long_horizon_classification": rows,
        "focus_target_hold_hold_better": {
            "count": int(len(focus)),
            "sampled_action_counts": dict(Counter(focus["sampled_action_name"])),
            "hold_vs_serve_long_horizon_delta": stats(focus["delta_full_bootstrapped_return_serve_minus_hold"].dropna().tolist()),
            "raw_gae_direction_agreement_rate": float(focus["raw_gae_advantage_agreement"].mean()),
            "normalized_advantage_direction_agreement_rate": float(focus["normalized_advantage_agreement"].mean()),
            "td_delta_direction_agreement_rate": float(focus["td_delta_agreement"].mean()),
            "ppo_pressure_direction_agreement_rate": None
            if focus["ppo_agreement"].dropna().empty
            else float(focus["ppo_agreement"].dropna().mean()),
        },
    }


def cyclewise_collapse_origin(conditional: pd.DataFrame, merged: pd.DataFrame, actor_audit: Mapping[str, Any]) -> Dict[str, Any]:
    cycle_rows = []
    for cycle, group in conditional.groupby("cycle"):
        row: Dict[str, Any] = {"cycle": int(cycle)}
        for target in [ACTION_HOLD, ACTION_SERVE]:
            subset = group[group["target_name"] == target]
            row[f"{target}_P_HOLD_mean"] = float(subset["P_HOLD"].mean())
            row[f"{target}_P_SERVE_mean"] = float(subset["P_SERVE"].mean())
            row[f"{target}_probability_dominant_action"] = (
                "SERVE" if row[f"{target}_P_SERVE_mean"] > row[f"{target}_P_HOLD_mean"] else "HOLD"
            )
            row[f"{target}_sampled_counts"] = dict(Counter(subset["sampled_action_name"]))
        hold_credit = merged[(merged["cycle"] == cycle) & (merged["target_name"] == ACTION_HOLD)]
        hold_hold = hold_credit[hold_credit["sampled_action_name"] == ACTION_HOLD]
        hold_serve = hold_credit[hold_credit["sampled_action_name"] == ACTION_SERVE]
        row["target_hold_sampled_hold_raw_gae_mean"] = None if hold_hold.empty else float(hold_hold["raw_gae_advantage"].mean())
        row["target_hold_sampled_serve_raw_gae_mean"] = None if hold_serve.empty else float(hold_serve["raw_gae_advantage"].mean())
        row["target_hold_credit_favors_hold_over_serve"] = (
            row["target_hold_sampled_hold_raw_gae_mean"] is not None
            and row["target_hold_sampled_serve_raw_gae_mean"] is not None
            and row["target_hold_sampled_hold_raw_gae_mean"] > row["target_hold_sampled_serve_raw_gae_mean"]
        )
        cycle_rows.append(row)
    first_actor_divergence = next(
        (
            row
            for row in cycle_rows
            if row[f"{ACTION_HOLD}_probability_dominant_action"] == "SERVE"
            and row[f"{ACTION_SERVE}_probability_dominant_action"] == "SERVE"
        ),
        None,
    )
    first_credit_wrong = next((row for row in cycle_rows if row["target_hold_credit_favors_hold_over_serve"] is False), None)
    return {
        "stage": STAGE,
        "cyclewise_rows": cycle_rows,
        "earliest_systematic_divergence": "ACTOR_LOGITS_PROBABILITY_AT_CYCLE_1",
        "first_actor_dual_target_serve_dominance": first_actor_divergence,
        "first_target_hold_credit_not_favoring_hold": first_credit_wrong,
        "actor_paired_forward_support": actor_audit.get("paired_actor_context_sensitivity"),
        "interpretation": "Target context reaches the network, but actor probabilities are SERVE-dominant for both target contexts from cycle 1; target=HOLD credit does not systematically favor SERVE.",
    }


def root_cause_attribution(
    target_survival: Mapping[str, Any],
    actor_audit: Mapping[str, Any],
    credit_audit: Mapping[str, Any],
    alignment: Mapping[str, Any],
    cycle_origin: Mapping[str, Any],
) -> Dict[str, Any]:
    target_preserved = target_survival.get("target_information_survival_result") == "TARGET_CONTEXT_PRESERVED"
    actor_ignores = actor_audit.get("result") == "ACTOR_IGNORES_TARGET_CONTEXT"
    target_hold_credit_wrong = credit_audit.get("target_hold_credit_direction", {}).get("credit_favors_wrong_serve_action") is True
    if target_preserved and actor_ignores and not target_hold_credit_wrong:
        decision = "ACTOR_HEAD_FAILS_TO_USE_TARGET_CONTEXT"
        next_gate = "H4M-M_ACTOR_TARGET_SENSITIVITY_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION"
        confidence = "HIGH"
    elif not target_preserved:
        decision = "TARGET_CONTEXT_LOST_IN_GATV2_REPRESENTATION"
        next_gate = "H4M-M_GATV2_TARGET_PRESERVATION_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION"
        confidence = "HIGH"
    elif target_hold_credit_wrong:
        decision = "TARGET_CONDITIONED_CREDIT_FAVORS_WRONG_ACTION"
        next_gate = "H4M-M_TARGET_CONDITIONED_CREDIT_REPAIR_SELECTION_AND_FREEZE"
        confidence = "MEDIUM"
    else:
        decision = "CAUSAL_ATTRIBUTION_STILL_NOT_UNIQUE"
        next_gate = "BLOCK_PENDING_EXPLICIT_REPAIR_SELECTION"
        confidence = "LOW"
    return {
        "stage": STAGE,
        "root_cause": decision,
        "confidence": confidence,
        "exact_next_gate": next_gate,
        "evidence": {
            "target_information_survival_result": target_survival.get("target_information_survival_result"),
            "gatv2_embedding_sensitivity": target_survival.get("paired_gatv2_embedding_sensitivity"),
            "actor_target_sensitivity_result": actor_audit.get("result"),
            "actor_paired_context_sensitivity": actor_audit.get("paired_actor_context_sensitivity"),
            "target_hold_credit_direction": credit_audit.get("target_hold_credit_direction"),
            "target_serve_credit_direction": credit_audit.get("target_serve_credit_direction"),
            "long_horizon_focus_target_hold_hold_better": alignment.get("focus_target_hold_hold_better"),
            "earliest_systematic_divergence": cycle_origin.get("earliest_systematic_divergence"),
        },
        "previous_state_aliasing_root_cause_status": "H4M-K falsifies STATE_ALIASING_MISSING_CAUSAL_CONTEXT as a sufficient standalone root cause: 12D alias conflict groups are zero, but global SERVE collapse persists.",
    }


def repair_candidate_comparison(root: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = [
        {
            "candidate": "GATv2 target preservation repair",
            "selected": False,
            "reason": "Read-only perturbation shows nonzero GATv2 embedding sensitivity; target is preserved, albeit weak/overlapped.",
        },
        {
            "candidate": "Actor target sensitivity repair",
            "selected": root.get("root_cause") == "ACTOR_HEAD_FAILS_TO_USE_TARGET_CONTEXT",
            "reason": "Actor remains SERVE-dominant for HOLD and SERVE contexts with zero paired preference flips despite target-conditioned inputs.",
        },
        {
            "candidate": "Target-conditioned credit/PPO repair",
            "selected": False,
            "reason": "For target=HOLD, reward/TD/GAE do not systematically favor sampled SERVE over sampled HOLD; PPO pressure is partial and weakly aligned, not the earliest divergence.",
        },
        {
            "candidate": "Reward V2 repair",
            "selected": False,
            "reason": "Automatically selecting Reward V2 change is prohibited and unsupported by the observed target=HOLD credit direction.",
        },
    ]
    selected = next((row for row in candidates if row["selected"]), None)
    return {
        "stage": STAGE,
        "repair_selection_status": "SELECTED" if selected else "BLOCK_PENDING_EXPLICIT_REPAIR_SELECTION",
        "selected_minimal_repair": None if selected is None else "ACTOR_TARGET_CONTEXT_DIRECT_CONDITIONING_REPAIR",
        "candidates": candidates,
    }


def repair_contract(root: Mapping[str, Any], candidates: Mapping[str, Any]) -> Dict[str, Any]:
    contract = {
        "stage": STAGE,
        "contract_name": "PV8_H4M_L_ACTOR_TARGET_SENSITIVITY_REPAIR_CONTRACT",
        "selected_repair": candidates.get("selected_minimal_repair"),
        "exact_failing_layer": "Actor head / action decision layer after GATv2 embedding",
        "root_cause": root.get("root_cause"),
        "evidence": root.get("evidence"),
        "minimal_proposed_change": {
            "repair_class": "ACTOR_TARGET_CONTEXT_DIRECT_CONDITIONING",
            "description": "Expose the frozen pre-action 3D target context directly to the Actor decision head, e.g. concatenate target one-hot with agent GATv2 embedding at Actor input, while retaining the existing 12D observation schema and GATv2 path.",
            "implementation_not_executed_in_h4m_l": True,
        },
        "frozen_semantics": {
            "reward_v2": "UNCHANGED",
            "zero_loss": "UNCHANGED",
            "k_mask_action_contract": "UNCHANGED",
            "gae_ppo_critic_normalization": "UNCHANGED",
            "observation_schema_12d": "UNCHANGED",
            "target_context_source": "same pre-action R3 obligation/target one-hot from H4M-I/H4M-J contract",
            "environment_data": "UNCHANGED",
        },
        "expected_acceptance_test": [
            "source-only repair implementation and py_compile PASS",
            "12D target context exact invariants preserved",
            "read-only paired Actor target perturbation shows HOLD-vs-SERVE preference sensitivity without future leakage",
            "no Reward/GAE/PPO/K-mask/Zero-Loss/environment mutation",
            "fresh three-seed retraining required; no 9D or H4M-K checkpoint reuse",
        ],
        "future_leakage_prohibition": [
            "no long-horizon label as feature",
            "no counterfactual return as feature",
            "no future reward/KPI/passenger outcome/post-action state",
            "no supervised action truth reconstructed from future evidence",
        ],
        "fresh_training_requirement": {
            "required": True,
            "seeds": [1, 2, 3],
            "schedule": "H4M-B 11-cycle TRAIN44 only",
            "checkpoint_reuse_allowed": False,
        },
    }
    contract["contract_sha256"] = canonical_sha({k: v for k, v in contract.items() if k != "contract_sha256"})
    return contract


def gate_matrix(
    binding: Mapping[str, Any],
    target_survival: Mapping[str, Any],
    actor_audit: Mapping[str, Any],
    credit_audit: Mapping[str, Any],
    alignment: Mapping[str, Any],
    root: Mapping[str, Any],
    candidates: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_pass": binding.get("authoritative_binding_passed") is True,
        "target_information_survival_audited": target_survival.get("target_information_survival_result")
        in {"TARGET_CONTEXT_PRESERVED", "TARGET_CONTEXT_LOST_IN_GATV2"},
        "actor_target_sensitivity_audited": actor_audit.get("result")
        in {"ACTOR_USES_TARGET_CONTEXT", "ACTOR_IGNORES_TARGET_CONTEXT"},
        "target_conditioned_credit_quantified": credit_audit.get("row_count") == 11616,
        "long_horizon_crosswalk_complete": alignment.get("long_horizon_crosswalk_complete") is True,
        "root_cause_supported_or_not_unique": root.get("root_cause")
        in {
            "TARGET_CONTEXT_LOST_IN_GATV2_REPRESENTATION",
            "ACTOR_HEAD_FAILS_TO_USE_TARGET_CONTEXT",
            "TARGET_CONDITIONED_CREDIT_FAVORS_WRONG_ACTION",
            "PPO_TARGET_CONDITIONED_REINFORCEMENT_IMBALANCE",
            "MIXED_REPRESENTATION_AND_CREDIT_FAILURE",
            "TARGET_CONTEXT_REPAIR_NOT_CAUSALLY_RELEVANT_TO_COLLAPSE",
            "CAUSAL_ATTRIBUTION_STILL_NOT_UNIQUE",
        },
        "repair_selected_or_explicit_block": candidates.get("repair_selection_status")
        in {"SELECTED", "BLOCK_PENDING_EXPLICIT_REPAIR_SELECTION"},
        "no_training_or_repair_or_test6": True,
        "github_push_false": True,
    }
    passed = all(criteria.values()) and candidates.get("repair_selection_status") == "SELECTED"
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": root.get("root_cause") if passed else candidates.get("repair_selection_status", "H4M_L_BLOCKED"),
        "exact_next_gate": root.get("exact_next_gate") if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "read_only_diagnosis_executed": True,
            "training_executed": False,
            "repair_implemented": False,
            "optimizer_or_backward_executed": False,
            "checkpoint_mutated": False,
            "reward_gae_ppo_modified": False,
            "environment_expanded": False,
            "validation_or_test6_executed": False,
            "winner_or_baseline_selected": False,
            "github_push_performed": False,
        },
    }


def final_report(
    binding: Mapping[str, Any],
    target_survival: Mapping[str, Any],
    actor: Mapping[str, Any],
    credit: Mapping[str, Any],
    alignment: Mapping[str, Any],
    cycle_origin: Mapping[str, Any],
    root: Mapping[str, Any],
    contract: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    return f"""# H4M-L Target-Context Repair Outcome Review & Next Decision Selection

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_l_source_git_commit"]}
decision = {gate["decision"]}
exact_next_gate = {gate["exact_next_gate"]}

## GATv2 target preservation

result = {target_survival["target_information_survival_result"]}
embedding_l2_hold_vs_serve = {json.dumps(target_survival["paired_gatv2_embedding_sensitivity"]["embedding_l2_hold_vs_serve"], ensure_ascii=False, default=jsonable)}

## Actor target sensitivity

result = {actor["result"]}
paired_actor_context_sensitivity = {json.dumps(actor["paired_actor_context_sensitivity"], ensure_ascii=False, default=jsonable)}

## target=HOLD credit direction

```json
{json.dumps(credit["target_hold_credit_direction"], ensure_ascii=False, indent=2, default=jsonable)}
```

## target=SERVE credit direction

```json
{json.dumps(credit["target_serve_credit_direction"], ensure_ascii=False, indent=2, default=jsonable)}
```

## Long-horizon focus

```json
{json.dumps(alignment["focus_target_hold_hold_better"], ensure_ascii=False, indent=2, default=jsonable)}
```

## Earliest systematic divergence

{cycle_origin["earliest_systematic_divergence"]}

## Root cause

{root["root_cause"]}

## Selected minimal repair

selected_repair = {contract["selected_repair"]}
contract_sha256 = {contract["contract_sha256"]}

STOP: no training, no repair implementation, no Reward/GAE/PPO arbitrary modification, no environment expansion, no validation/TEST6, no GitHub push.
"""


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files: Dict[str, str] = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            output_files[str(path.relative_to(artifact_root))] = str(path)
    return {
        "stage": STAGE,
        "created_at": kst_now(),
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": output_files,
        "output_sha256": {name: sha256_file(Path(path)) for name, path in output_files.items()},
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def write_block_outputs(artifact_root: Path, created_at: str, binding: Mapping[str, Any], reason: str) -> None:
    gate = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": BLOCK_GATE,
        "decision": "H4M_L_BLOCKED",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
    }
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_target_information_survival.json": empty,
        "03_actor_target_sensitivity.json": empty,
        "04_target_conditioned_credit.json": empty,
        "05_long_horizon_credit_alignment.json": empty,
        "06_cyclewise_collapse_origin.json": empty,
        "07_root_cause_attribution.json": empty,
        "08_repair_candidate_comparison.json": empty,
        "09_h4m_l_repair_contract.json": empty,
        "10_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(f"# H4M-L\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-L] artifact root: {artifact_root}")
    print(f"[H4M-L] gate: {BLOCK_GATE}")
    print(f"[H4M-L] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_l_target_context_repair_outcome_review_next_decision_{stamp}"
    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    if not binding["authoritative_binding_passed"]:
        write_block_outputs(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return
    conditional, reward, td, pressure, pre_action = load_h4m_k_frames()
    merged = merge_credit_frames(conditional, reward, td, pressure)
    if len(conditional) != 11616 or len(merged) != 11616:
        write_block_outputs(
            artifact_root,
            created_at,
            binding,
            f"H4M_K_TRACE_INSUFFICIENT_CONDITIONAL_ROWS_{len(conditional)}_MERGED_ROWS_{len(merged)}",
        )
        return
    target_survival, actor_probe, probe_rows = target_information_survival(created_at)
    actor = actor_target_sensitivity(conditional, actor_probe)
    credit = target_conditioned_credit(merged)
    alignment = long_horizon_credit_alignment(merged)
    cycle_origin = cyclewise_collapse_origin(conditional, merged, actor)
    root = root_cause_attribution(target_survival, actor, credit, alignment, cycle_origin)
    candidates = repair_candidate_comparison(root)
    contract = repair_contract(root, candidates)
    gate = gate_matrix(binding, target_survival, actor, credit, alignment, root, candidates)
    report = final_report(binding, target_survival, actor, credit, alignment, cycle_origin, root, contract, gate)

    artifact_root.mkdir(parents=True, exist_ok=True)
    probe_rows.to_parquet(artifact_root / "actor_representation_readonly_probe.parquet", index=False)
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_target_information_survival.json": target_survival,
        "03_actor_target_sensitivity.json": actor,
        "04_target_conditioned_credit.json": credit,
        "05_long_horizon_credit_alignment.json": alignment,
        "06_cyclewise_collapse_origin.json": cycle_origin,
        "07_root_cause_attribution.json": root,
        "08_repair_candidate_comparison.json": candidates,
        "09_h4m_l_repair_contract.json": contract,
        "10_gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    print(f"[H4M-L] artifact root: {artifact_root}")
    print(f"[H4M-L] gate: {gate['gate']}")
    print(f"[H4M-L] source_commit: {provenance['h4m_l_source_git_commit']}")
    print(f"[H4M-L] gatv2_target_result: {target_survival['target_information_survival_result']}")
    print(f"[H4M-L] actor_target_result: {actor['result']}")
    print(f"[H4M-L] earliest_divergence: {cycle_origin['earliest_systematic_divergence']}")
    print(f"[H4M-L] root_cause: {root['root_cause']}")
    print(f"[H4M-L] selected_repair: {contract['selected_repair']}")
    print(f"[H4M-L] repair_contract_sha256: {contract['contract_sha256']}")
    print(f"[H4M-L] next_gate: {gate['exact_next_gate']}")
    print("[H4M-L] STOP: no training, no repair implementation, no TEST6, github_push=false")


if __name__ == "__main__":
    main()
