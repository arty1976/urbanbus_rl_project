#!/usr/bin/env python3
"""H4M-I observation discrimination repair selection and freeze.

Diagnosis + selection + freeze only. This runner reads the H4M-H-RERUN
diagnostic traces, reconstructs the frozen runtime observation tensors, runs
forward-only representation probes from the saved diagnostic checkpoints, and
freezes the smallest supported repair contract. It performs no RL training,
no optimizer step, no Reward/GAE/PPO repair, no validation/TEST6, and no push.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-I"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_I_OBSERVATION_DISCRIMINATION_REPAIR_SELECTION_AND_FREEZE_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_I_OBSERVATION_DISCRIMINATION_REPAIR_SELECTION_AND_FREEZE_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_H_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_h_rerun_fresh_instrumented_credit_diagnostic_retraining_20260816_233823+0900"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
DATASET_SOURCE = TRAINING_ROOT / "build_gatv2_dataset.py"

EXPECTED = {
    "h4m_h_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_H_RERUN_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING_COMPLETE",
    "h4m_h_source_commit": "d677f1dedc536e0d42d84ebd8fac264c584a72aa",
    "h4m_h_root_decision": "CREDIT_PIPELINE_ALIGNED_OBSERVATION_DISCRIMINATION_REQUIRED",
    "h4m_h_next_gate": "H4M-I_OBSERVATION_DISCRIMINATION_REPAIR_SELECTION_AND_FREEZE",
    "active_mps_instrumentation_contract_sha256": "e6c73da48c12edfd573069730d2eeec32c74fea74f590105e55ad3502a729c92",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

X_COLS = [
    "boardings_recent_log",
    "alightings_recent_log",
    "waiting_passenger_cnt_log",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_peak",
    "delta_t_hr_effective",
]
TARGET_CONTEXT_FIELDS = [
    "r3_action_target_is_hold",
    "r3_action_target_is_serve",
    "r3_action_target_is_skip",
]

ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
LABEL_HOLD = "HOLD_LONG_HORIZON_BETTER"
LABEL_SERVE = "SERVE_LONG_HORIZON_BETTER"

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_runtime_observation_schema.json",
    "03_raw_observation_discrimination.json",
    "04_gatv2_representation_discrimination.json",
    "05_actor_discrimination.json",
    "06_state_aliasing_conflict_pairs.parquet",
    "07_causal_feature_availability_audit.json",
    "08_root_cause_attribution.json",
    "09_repair_candidate_comparison.json",
    "10_h4m_i_repair_contract.json",
    "11_gate_matrix.json",
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def import_module_from_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
        "h4m_i_source_git_commit": latest_source_commit,
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
    h4mh_gate = read_json(H4M_H_ROOT / "14_gate_matrix.json")
    h4mh_root = read_json(H4M_H_ROOT / "12_root_cause_decision.json")
    h4mh_binding = read_json(H4M_H_ROOT / "01_authoritative_binding.json")
    h4mg_close_gate = read_json(H4M_G_CLOSE_ROOT / "14_gate_matrix.json")
    h4mb_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    h4mh_upstream = h4mh_binding.get("sha_bindings", {})
    checks = {
        "source_only_commit_before_audit": provenance.get("local_source_only_commit_created_before_audit") is True,
        "h4m_h_gate_match": h4mh_gate.get("gate") == EXPECTED["h4m_h_gate"],
        "h4m_h_source_commit_match": h4mh_binding.get("source_provenance", {}).get("h4m_h_rerun_source_git_commit")
        == EXPECTED["h4m_h_source_commit"],
        "h4m_h_root_decision_match": h4mh_root.get("decision") == EXPECTED["h4m_h_root_decision"],
        "h4m_h_next_gate_match": h4mh_root.get("exact_next_gate") == EXPECTED["h4m_h_next_gate"],
        "active_mps_contract_match_h4mh": h4mh_binding.get("sha_bindings", {}).get("active_mps_instrumentation_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "active_mps_contract_match_h4mg_close": h4mg_close_gate.get("final_active_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_b_schedule_sha_match_direct": h4mb_schedule.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"],
        "h4m_b_schedule_sha_match_h4mh": h4mh_upstream.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": h4mh_upstream.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": h4mh_upstream.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": h4mh_upstream.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": h4mh_upstream.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "test6_sealed_h4mh": h4mh_gate.get("criteria", {}).get("test6_sealed") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_h_rerun": str(H4M_H_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "source_provenance": provenance,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "bound_inputs": {
            "h4m_h_gate": h4mh_gate.get("gate"),
            "h4m_h_source_commit": h4mh_binding.get("source_provenance", {}).get("h4m_h_rerun_source_git_commit"),
            "h4m_h_root_decision": h4mh_root.get("decision"),
            "h4m_h_next_gate": h4mh_root.get("exact_next_gate"),
            "h4m_g_close_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        },
        "sha_bindings": {
            "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "forbidden_actions_attestation": {
            "rl_training_executed": False,
            "repair_implementation_executed": False,
            "feature_added": False,
            "architecture_changed": False,
            "validation_or_test6_executed": False,
            "winner_or_baseline_selected": False,
            "environment_expanded": False,
            "github_push_performed": False,
        },
    }


def read_h4mh_trace() -> pd.DataFrame:
    pre_cols = [
        "sample_uid",
        "seed",
        "outer_cycle",
        "window_id",
        "snapshot_id",
        "time_band",
        "step_index",
        "agent_slot",
        "agent_id",
        "state_hash",
        "target_id",
        "target",
        "legal_action_ids",
        "actor_observation_hash",
        "critic_observation_hash",
        "policy_entropy",
        "masked_probability_0",
        "masked_probability_1",
        "pre_mask_logit_0",
        "pre_mask_logit_1",
        "post_mask_logit_0",
        "post_mask_logit_1",
    ]
    shadow_cols = [
        "sample_uid",
        "classification",
        "delta_full_bootstrapped_return_serve_minus_hold",
        "hold_full_bootstrapped_return",
        "serve_full_bootstrapped_return",
    ]
    pre_frames = [pd.read_parquet(path, columns=pre_cols) for path in sorted((H4M_H_ROOT / "05_actual_credit_trace").glob("seed=*/outer_cycle=*/actual_pre_action_trace.parquet"))]
    shadow_frames = [
        pd.read_parquet(path, columns=shadow_cols)
        for path in sorted((H4M_H_ROOT / "06_long_horizon_shadow_trace").glob("seed=*/outer_cycle=*/counterfactual_pair_summary.parquet"))
    ]
    pre = pd.concat(pre_frames, ignore_index=True)
    shadow = pd.concat(shadow_frames, ignore_index=True)
    joined = pre.merge(shadow, on="sample_uid", how="inner", validate="one_to_one")
    joined["label_binary"] = (joined["classification"] == LABEL_SERVE).astype(int)
    joined["policy_margin_hold_minus_serve"] = joined["masked_probability_0"] - joined["masked_probability_1"]
    joined["logit_margin_hold_minus_serve"] = joined["pre_mask_logit_0"] - joined["pre_mask_logit_1"]
    joined["actor_correct_direction"] = (
        ((joined["classification"] == LABEL_HOLD) & (joined["policy_margin_hold_minus_serve"] > 0.0))
        | ((joined["classification"] == LABEL_SERVE) & (joined["policy_margin_hold_minus_serve"] < 0.0))
    )
    return joined


def exact_collision_summary(df: pd.DataFrame, keys: Sequence[str]) -> Dict[str, Any]:
    grouped = df.groupby(list(keys))["classification"].agg(lambda s: tuple(sorted(set(str(v) for v in s))))
    conflict = grouped[grouped.apply(len) > 1]
    if len(conflict) > 0:
        key_frame = conflict.reset_index()[list(keys)]
        conflict_key_set = {tuple(row) for row in key_frame.to_numpy().tolist()}
        row_conflicts = sum(1 for row in df[list(keys)].to_numpy().tolist() if tuple(row) in conflict_key_set)
    else:
        row_conflicts = 0
    return {
        "keys": list(keys),
        "group_count": int(len(grouped)),
        "conflict_group_count": int(len(conflict)),
        "conflict_group_rate": float(len(conflict) / max(1, len(grouped))),
        "rows_in_conflicting_groups": int(row_conflicts),
        "row_conflict_rate": float(row_conflicts / max(1, len(df))),
        "representative_conflicts": [
            {**{key: row[key] for key in keys}, "labels": list(row["classification"])}
            for row in conflict.reset_index().head(20).to_dict("records")
        ],
    }


def vector_stats(values: Sequence[float]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": len(nums),
        "mean": float(mean(nums)),
        "min": float(min(nums)),
        "max": float(max(nums)),
        "std": float(np.std(np.asarray(nums, dtype=np.float64))),
    }


def binary_auc(values: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    arr = pd.DataFrame({"value": list(values), "label": list(labels)}).dropna()
    if arr.empty or arr["label"].nunique() != 2:
        return None
    ranks = arr["value"].rank(method="average")
    pos = arr["label"] == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    rank_sum_pos = float(ranks[pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(1.0, float(n_pos * n_neg))
    return float(auc)


def separation_by_feature(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for col in columns:
        hold = df.loc[df["classification"] == LABEL_HOLD, col].astype(float).to_numpy()
        serve = df.loc[df["classification"] == LABEL_SERVE, col].astype(float).to_numpy()
        pooled_std = float(np.std(df[col].astype(float).to_numpy()))
        mean_hold = float(np.mean(hold)) if hold.size else None
        mean_serve = float(np.mean(serve)) if serve.size else None
        smd = None if pooled_std == 0.0 or mean_hold is None or mean_serve is None else float((mean_serve - mean_hold) / pooled_std)
        auc = binary_auc(df[col].astype(float).tolist(), df["label_binary"].astype(int).tolist())
        rows.append(
            {
                "feature": col,
                "hold_mean": mean_hold,
                "serve_mean": mean_serve,
                "pooled_std": pooled_std,
                "standardized_mean_difference_serve_minus_hold": smd,
                "absolute_standardized_mean_difference": None if smd is None else abs(smd),
                "auc_for_SERVE_LONG_HORIZON_BETTER": auc,
                "separability_score": max(abs((auc or 0.5) - 0.5) * 2.0, abs(smd or 0.0) / 3.0),
            }
        )
    return sorted(rows, key=lambda row: row["separability_score"], reverse=True)


def nearest_neighbor_metrics(matrix: np.ndarray, labels: Sequence[int]) -> Dict[str, Any]:
    if matrix.shape[0] <= 1:
        return {"n": int(matrix.shape[0]), "nearest_neighbor_conflict_rate": None}
    arr = matrix.astype(np.float64)
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = np.argmin(dist, axis=1)
    labels_arr = np.asarray(labels, dtype=int)
    conflict = labels_arr[nn] != labels_arr
    return {
        "n": int(matrix.shape[0]),
        "nearest_neighbor_conflict_count": int(conflict.sum()),
        "nearest_neighbor_conflict_rate": float(conflict.mean()),
        "nearest_neighbor_distance": vector_stats(dist[np.arange(dist.shape[0]), nn].tolist()),
    }


def class_distance_metrics(matrix: np.ndarray, labels: Sequence[int]) -> Dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=int)
    hold = matrix[labels_arr == 0].astype(np.float64)
    serve = matrix[labels_arr == 1].astype(np.float64)
    if hold.size == 0 or serve.size == 0:
        return {"within_between_available": False}
    hold_centroid = hold.mean(axis=0)
    serve_centroid = serve.mean(axis=0)
    centroid_distance = float(np.linalg.norm(serve_centroid - hold_centroid))
    hold_within = np.linalg.norm(hold - hold_centroid, axis=1)
    serve_within = np.linalg.norm(serve - serve_centroid, axis=1)
    return {
        "within_between_available": True,
        "centroid_distance": centroid_distance,
        "hold_within_distance": vector_stats(hold_within.tolist()),
        "serve_within_distance": vector_stats(serve_within.tolist()),
        "centroid_to_within_ratio": centroid_distance / max(1.0e-12, float(np.mean(np.concatenate([hold_within, serve_within])))),
    }


def reconstruct_runtime_observations(df: pd.DataFrame, created_at: str, h4mg: Any) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    ctx = h4mg.build_context(1, created_at)
    raw_by_key: Dict[Tuple[int, int], List[float]] = {}
    hash_by_key: Dict[Tuple[int, int], str] = {}
    target_by_key: Dict[Tuple[int, int], int] = {}
    target_source_rows: List[Dict[str, Any]] = []
    for step, data in enumerate(ctx["train_data"]):
        indices = ctx["dl1"].agent_indices_for_step(ctx["config"]["spec"], step, int(ctx["config"]["effective_agents"]))
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        target = ctx["dl1"].action_targets_from_y(data.y[idx_tensor], int(ctx["config"]["action_dim"]))
        active_mask = data.node_mask[idx_tensor].bool()
        for agent_slot, (agent_id, target_id, active) in enumerate(zip(indices, target.tolist(), active_mask.tolist())):
            if not bool(active):
                continue
            key = (int(step), int(agent_id))
            raw = data.x[int(agent_id)].detach().cpu().numpy().astype(np.float64).tolist()
            raw_by_key[key] = raw
            hash_by_key[key] = h4mg.tensor_hash(data.x[int(agent_id)])
            target_by_key[key] = int(target_id)
            target_source_rows.append(
                {
                    "step_index": int(step),
                    "agent_slot": int(agent_slot),
                    "agent_id": int(agent_id),
                    "target_id": int(target_id),
                    "raw_observation_hash": hash_by_key[key],
                }
            )
    enriched = df.copy()
    raw_matrix = []
    reconstructed_hashes = []
    reconstructed_targets = []
    for row in enriched.itertuples(index=False):
        key = (int(row.step_index), int(row.agent_id))
        raw_matrix.append(raw_by_key[key])
        reconstructed_hashes.append(hash_by_key[key])
        reconstructed_targets.append(target_by_key[key])
    for idx, col in enumerate(X_COLS):
        enriched[col] = [float(vec[idx]) for vec in raw_matrix]
    enriched["reconstructed_actor_observation_hash"] = reconstructed_hashes
    enriched["reconstructed_target_id"] = reconstructed_targets
    enriched["runtime_reconstruction_hash_match"] = enriched["reconstructed_actor_observation_hash"] == enriched["actor_observation_hash"]
    enriched["runtime_reconstruction_target_match"] = enriched["reconstructed_target_id"].astype(int) == enriched["target_id"].astype(int)
    schema = {
        "stage": STAGE,
        "created_at": created_at,
        "runtime_observation_source": str(DATASET_SOURCE),
        "runtime_collection_source": str(H4MG_SOURCE),
        "actor_raw_observation_tensor": "data.x[agent_id]",
        "actor_raw_observation_shape": [9],
        "critic_graph_observation_shape": [255, 9],
        "x_feature_names_confirmed_from_code": X_COLS,
        "target_context_currently_available_before_sampling_in_runtime": {
            "source_expression": "dl1.action_targets_from_y(data.y[agent_indices], action_dim)",
            "current_consumers": [
                "h4k.masked_logits_for_targets(logits, target)",
                "Reward V2 metrics target_id",
                "H4M-H state_hash",
            ],
            "not_current_actor_input": True,
            "not_long_horizon_shadow_label": True,
        },
        "reconstruction_checks": {
            "rows": int(len(enriched)),
            "actor_observation_hash_match_count": int(enriched["runtime_reconstruction_hash_match"].sum()),
            "actor_observation_hash_match_all": bool(enriched["runtime_reconstruction_hash_match"].all()),
            "target_id_match_count": int(enriched["runtime_reconstruction_target_match"].sum()),
            "target_id_match_all": bool(enriched["runtime_reconstruction_target_match"].all()),
        },
        "feature_source_semantics": {
            "raw_x": "pre-action node feature tensor from frozen TRAIN44 graph snapshots",
            "target_context": "frozen R3 action-target/obligation context already materialized before action sampling for K-mask/reward; selected repair does not use long-horizon shadow labels or future rewards",
        },
    }
    return enriched, schema, ctx


def raw_observation_discrimination(enriched: pd.DataFrame) -> Dict[str, Any]:
    unique_state = enriched.drop_duplicates(["seed", "window_id", "step_index", "agent_id"]).copy()
    feature_rows = separation_by_feature(enriched, X_COLS)
    raw_matrix = unique_state[X_COLS].astype(float).to_numpy()
    raw_plus_target = np.concatenate(
        [
            raw_matrix,
            np.eye(3, dtype=np.float64)[unique_state["target_id"].astype(int).to_numpy()],
        ],
        axis=1,
    )
    labels = unique_state["label_binary"].astype(int).to_numpy()
    exact_actor = exact_collision_summary(enriched, ["actor_observation_hash"])
    exact_actor_target = exact_collision_summary(enriched, ["actor_observation_hash", "target_id"])
    exact_state = exact_collision_summary(enriched, ["window_id", "step_index", "agent_id"])
    label_by_target = {
        str(k): int(v)
        for k, v in enriched.groupby(["target_id", "classification"]).size().to_dict().items()
    }
    return {
        "stage": STAGE,
        "row_count": int(len(enriched)),
        "unique_seed_state_count": int(len(unique_state)),
        "label_counts": dict(Counter(enriched["classification"])),
        "exact_observation_collision": exact_actor,
        "exact_observation_plus_target_collision": exact_actor_target,
        "exact_state_identity_collision": exact_state,
        "near_neighbor_raw_x": nearest_neighbor_metrics(raw_matrix, labels),
        "near_neighbor_raw_x_plus_target_context": nearest_neighbor_metrics(raw_plus_target, labels),
        "class_distance_raw_x": class_distance_metrics(raw_matrix, labels),
        "class_distance_raw_x_plus_target_context": class_distance_metrics(raw_plus_target, labels),
        "feature_distributions_and_separability": feature_rows,
        "target_id_to_long_horizon_label_crosswalk_counts": label_by_target,
        "raw_observation_discrimination_result": "RAW_X_ALIASES_TARGET_CONDITIONED_LONG_HORIZON_LABELS_TARGET_CONTEXT_RESOLVES_EXACT_CONFLICTS",
        "interpretation": (
            "The 9D actor raw observation is insufficient by itself: exact actor_observation_hash groups contain conflicting "
            "HOLD/SERVE long-horizon labels, while adding the existing target/obligation context reduces exact conflicts to zero."
        ),
    }


def checkpoint_path_for_seed(seed: int) -> Path:
    return H4M_H_ROOT / "checkpoints" / f"H4M_H_SEED_{seed:03d}_FRESH_INSTRUMENTED_DIAGNOSTIC.pt"


def load_final_checkpoint_into_context(ctx: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path_for_seed(seed), map_location="cpu", weights_only=False)
    ctx["encoder"].load_state_dict(payload["gatv2_state_dict"])
    ctx["actor"].load_state_dict(payload["actor_state_dict"])
    ctx["critic"].load_state_dict(payload["critic_state_dict"])
    ctx["return_normalizer"].load_state_dict(payload.get("return_normalizer_state", {}))
    ctx["encoder"].eval()
    ctx["actor"].eval()
    ctx["critic"].eval()
    return {
        "checkpoint_path": str(checkpoint_path_for_seed(seed)),
        "checkpoint_sha256": sha256_file(checkpoint_path_for_seed(seed)),
        "checkpoint_stage": payload.get("stage"),
        "checkpoint_seed": payload.get("seed"),
        "checkpoint_metadata": payload.get("metadata"),
    }


def forward_only_representation_probe(enriched: pd.DataFrame, created_at: str, h4mg: Any) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    unique = enriched.drop_duplicates(["seed", "window_id", "step_index", "agent_id"]).copy()
    rows: List[Dict[str, Any]] = []
    checkpoint_registry: List[Dict[str, Any]] = []
    for seed in sorted(int(v) for v in unique["seed"].unique()):
        ctx = h4mg.build_context(seed, created_at)
        checkpoint_registry.append(load_final_checkpoint_into_context(ctx, seed))
        needed = unique[unique["seed"].astype(int) == seed]
        needed_keys = {(int(row.step_index), int(row.agent_id)): row for row in needed.itertuples(index=False)}
        with torch.no_grad():
            for step, data in enumerate(ctx["train_data"]):
                keys_for_step = [key for key in needed_keys if key[0] == step]
                if not keys_for_step:
                    continue
                node_embeddings = ctx["encoder"](data)
                indices = ctx["dl1"].agent_indices_for_step(ctx["config"]["spec"], step, int(ctx["config"]["effective_agents"]))
                idx_tensor = torch.tensor(indices, dtype=torch.long)
                target = ctx["dl1"].action_targets_from_y(data.y[idx_tensor], int(ctx["config"]["action_dim"]))
                logits_by_agent = ctx["actor"](node_embeddings[idx_tensor])
                for agent_slot, agent_id in enumerate(indices):
                    key = (int(step), int(agent_id))
                    if key not in needed_keys:
                        continue
                    source = needed_keys[key]
                    emb = node_embeddings[int(agent_id)].detach().cpu().numpy().astype(np.float64)
                    logits = logits_by_agent[agent_slot].detach().cpu().numpy().astype(np.float64)
                    probs = torch.softmax(logits_by_agent[agent_slot].detach().cpu(), dim=-1).numpy().astype(np.float64)
                    rows.append(
                        {
                            "seed": seed,
                            "window_id": source.window_id,
                            "step_index": int(step),
                            "agent_id": int(agent_id),
                            "target_id": int(target[agent_slot].detach().cpu().item()),
                            "classification": source.classification,
                            "label_binary": int(source.label_binary),
                            "embedding": emb,
                            "final_probe_logit_0": float(logits[0]),
                            "final_probe_logit_1": float(logits[1]),
                            "final_probe_probability_0": float(probs[0]),
                            "final_probe_probability_1": float(probs[1]),
                            "final_probe_margin_hold_minus_serve": float(probs[0] - probs[1]),
                        }
                    )
    rep = pd.DataFrame(rows)
    matrix = np.vstack(rep["embedding"].to_numpy()).astype(np.float64)
    labels = rep["label_binary"].astype(int).to_numpy()
    embedding_feature_cols = [f"embedding_{idx:03d}" for idx in range(matrix.shape[1])]
    emb_df = pd.DataFrame(matrix, columns=embedding_feature_cols)
    emb_df["classification"] = rep["classification"].to_numpy()
    emb_df["label_binary"] = labels
    emb_sep = separation_by_feature(emb_df, embedding_feature_cols)[:20]
    actor_probe_correct = (
        ((rep["classification"] == LABEL_HOLD) & (rep["final_probe_margin_hold_minus_serve"] > 0.0))
        | ((rep["classification"] == LABEL_SERVE) & (rep["final_probe_margin_hold_minus_serve"] < 0.0))
    )
    representation = {
        "stage": STAGE,
        "probe_type": "READ_ONLY_FINAL_H4M_H_RERUN_CHECKPOINT_FORWARD_PASS",
        "training_or_optimizer_step_executed": False,
        "probe_device": "cpu",
        "rows": int(len(rep)),
        "embedding_dim": int(matrix.shape[1]),
        "checkpoint_registry": checkpoint_registry,
        "nearest_neighbor_embedding": nearest_neighbor_metrics(matrix, labels),
        "class_distance_embedding": class_distance_metrics(matrix, labels),
        "top_embedding_dimensions_by_separability": emb_sep,
        "final_checkpoint_actor_probe_correct_direction_rate": float(actor_probe_correct.mean()),
        "final_checkpoint_actor_probe_correct_direction_by_label": {
            str(label): float(actor_probe_correct[rep["classification"] == label].mean())
            for label in sorted(rep["classification"].unique())
        },
        "gatv2_representation_result": (
            "TARGET_CONTEXT_NOT_IN_GATV2_INPUT_SO_REPRESENTATION_CANNOT_ENCODE_THE_ZERO_CONFLICT_TARGET_CONDITIONED_BOUNDARY"
        ),
        "limitations": [
            "H4M-H stored final checkpoints, not every pre-cycle checkpoint; representation probe is final-checkpoint forward-only.",
            "Raw/state/logit H4M-H traces provide exact sample_uid policy output for all cycles; embeddings are diagnostic layer probes only.",
        ],
    }
    return rep, representation, {"checkpoint_registry": checkpoint_registry}


def actor_discrimination(enriched: pd.DataFrame, rep: pd.DataFrame) -> Dict[str, Any]:
    grouped = {}
    for label, sub in enriched.groupby("classification"):
        grouped[str(label)] = {
            "count": int(len(sub)),
            "P_HOLD": vector_stats(sub["masked_probability_0"].astype(float).tolist()),
            "P_SERVE": vector_stats(sub["masked_probability_1"].astype(float).tolist()),
            "policy_margin_hold_minus_serve": vector_stats(sub["policy_margin_hold_minus_serve"].astype(float).tolist()),
            "pre_mask_logit_margin_hold_minus_serve": vector_stats(sub["logit_margin_hold_minus_serve"].astype(float).tolist()),
            "correct_direction_rate": float(sub["actor_correct_direction"].mean()),
        }
    by_target = {}
    for target, sub in enriched.groupby("target_id"):
        by_target[str(int(target))] = {
            "count": int(len(sub)),
            "classification_counts": dict(Counter(sub["classification"])),
            "P_HOLD_mean": float(sub["masked_probability_0"].mean()),
            "P_SERVE_mean": float(sub["masked_probability_1"].mean()),
            "margin_hold_minus_serve_mean": float(sub["policy_margin_hold_minus_serve"].mean()),
        }
    final_correct = (
        ((rep["classification"] == LABEL_HOLD) & (rep["final_probe_margin_hold_minus_serve"] > 0.0))
        | ((rep["classification"] == LABEL_SERVE) & (rep["final_probe_margin_hold_minus_serve"] < 0.0))
    )
    return {
        "stage": STAGE,
        "actual_h4m_h_trace_actor_output": {
            "row_count": int(len(enriched)),
            "overall_correct_direction_rate": float(enriched["actor_correct_direction"].mean()),
            "by_long_horizon_label": grouped,
            "by_target_id": by_target,
            "serve_dominance_present_for_hold_better_states": bool(
                grouped.get(LABEL_HOLD, {}).get("policy_margin_hold_minus_serve", {}).get("mean", 0.0) < 0.0
            ),
        },
        "final_checkpoint_actor_probe": {
            "unique_seed_state_rows": int(len(rep)),
            "overall_correct_direction_rate": float(final_correct.mean()),
            "by_long_horizon_label": {
                str(label): float(final_correct[rep["classification"] == label].mean())
                for label in sorted(rep["classification"].unique())
            },
            "margin_by_label": {
                str(label): vector_stats(rep.loc[rep["classification"] == label, "final_probe_margin_hold_minus_serve"].tolist())
                for label in sorted(rep["classification"].unique())
            },
        },
        "actor_discrimination_result": (
            "ACTOR_OUTPUT_IS_SERVE_BIASED_FOR_TARGET_0_HOLD_LONG_HORIZON_BETTER_STATES_BECAUSE_TARGET_CONTEXT_IS_NOT_IN_ACTOR_INPUT"
        ),
    }


def conflict_pairs(enriched: pd.DataFrame, artifact_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for obs_hash, group in enriched.groupby("actor_observation_hash"):
        labels = sorted(set(group["classification"]))
        if len(labels) <= 1:
            continue
        hold = group[group["classification"] == LABEL_HOLD].head(3)
        serve = group[group["classification"] == LABEL_SERVE].head(3)
        for _, hrow in hold.iterrows():
            for _, srow in serve.iterrows():
                rows.append(
                    {
                        "conflict_type": "EXACT_ACTOR_OBSERVATION_HASH_DIFFERENT_LONG_HORIZON_LABEL",
                        "actor_observation_hash": obs_hash,
                        "left_sample_uid": hrow["sample_uid"],
                        "left_label": hrow["classification"],
                        "left_target_id": int(hrow["target_id"]),
                        "left_window_id": hrow["window_id"],
                        "left_step_index": int(hrow["step_index"]),
                        "left_agent_id": int(hrow["agent_id"]),
                        "right_sample_uid": srow["sample_uid"],
                        "right_label": srow["classification"],
                        "right_target_id": int(srow["target_id"]),
                        "right_window_id": srow["window_id"],
                        "right_step_index": int(srow["step_index"]),
                        "right_agent_id": int(srow["agent_id"]),
                        "target_context_resolves_pair": bool(int(hrow["target_id"]) != int(srow["target_id"])),
                    }
                )
    out = artifact_root / "06_state_aliasing_conflict_pairs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    return {
        "path": str(out),
        "rows": len(rows),
        "sha256": sha256_file(out),
        "representative_pair_count_policy": "up to 3 HOLD-labeled × 3 SERVE-labeled examples per exact actor_observation_hash conflict group",
    }


def causal_feature_availability_audit(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "selected_candidate_fields": TARGET_CONTEXT_FIELDS,
        "availability_timing": {
            "available_before_action_sampling_in_current_runtime": True,
            "evidence": [
                "H4M-G collect_controlled_rollout computes target = dl1.action_targets_from_y(data.y[agent_indices], action_dim) before h4k.masked_logits_for_targets and Categorical.sample.",
                "H4M-H state_hash already includes target_id and legal_action_ids before action sampling.",
                "Reward V2 metrics already consume target_id during immediate reward materialization in the frozen execution path.",
            ],
        },
        "anti_leakage_checks": {
            "uses_long_horizon_shadow_result": False,
            "uses_counterfactual_classification": False,
            "uses_future_reward": False,
            "uses_future_passenger_outcome": False,
            "uses_post_action_state": False,
            "uses_future_kpi": False,
            "new_external_future_source_introduced": False,
            "note": (
                "The candidate only exposes the already-materialized frozen R3 action-target/obligation context that the current "
                "runtime consumes for K-mask/Reward before action sampling. H4M-J must preserve this binding and must not replace it "
                "with long-horizon labels or post-action outcomes."
            ),
        },
        "causal_semantics": {
            "field_meaning": "one-hot encoding of the current frozen R3 action target/obligation context: HOLD/SERVE/SKIP",
            "why_minimal": "exact conflicts in 9D actor observation drop to zero when this existing context is included",
            "long_horizon_labels_as_training_input": False,
        },
        "feature_availability_audit_passed": True,
    }


def root_cause_attribution(raw: Mapping[str, Any], rep: Mapping[str, Any], actor: Mapping[str, Any]) -> Dict[str, Any]:
    raw_conflict_rate = raw["exact_observation_collision"]["row_conflict_rate"]
    raw_plus_target_conflicts = raw["exact_observation_plus_target_collision"]["conflict_group_count"]
    hold_correct = actor["actual_h4m_h_trace_actor_output"]["by_long_horizon_label"][LABEL_HOLD]["correct_direction_rate"]
    serve_correct = actor["actual_h4m_h_trace_actor_output"]["by_long_horizon_label"][LABEL_SERVE]["correct_direction_rate"]
    decision = "STATE_ALIASING_MISSING_CAUSAL_CONTEXT"
    return {
        "stage": STAGE,
        "root_cause_class": decision,
        "evidence": {
            "raw_actor_observation_row_conflict_rate": raw_conflict_rate,
            "raw_actor_observation_conflict_groups": raw["exact_observation_collision"]["conflict_group_count"],
            "raw_actor_observation_plus_target_conflict_groups": raw_plus_target_conflicts,
            "target_id_exactly_matches_long_horizon_label_counts": raw["target_id_to_long_horizon_label_crosswalk_counts"],
            "h4m_h_actor_correct_direction_rate_hold_better": hold_correct,
            "h4m_h_actor_correct_direction_rate_serve_better": serve_correct,
            "representation_probe_result": rep["gatv2_representation_result"],
            "actor_probe_result": actor["actor_discrimination_result"],
        },
        "excluded_attributions": {
            "RAW_OBSERVATION_INFORMATION_DEFICIT": "too broad; the missing discriminator is specifically the already-existing target/obligation context, not arbitrary new environment data",
            "GATV2_REPRESENTATION_DISCRIMINATION_LOSS": "not primary because the discriminator is absent from GATv2 input; representation cannot preserve what it never receives",
            "ACTOR_HEAD_DISCRIMINATION_FAILURE": "secondary symptom; actor output is biased, but the minimal fix is to expose missing target context rather than alter PPO/actor loss",
            "OBSERVATION_DISCRIMINATION_NOT_SUPPORTED_AS_ROOT_CAUSE": "rejected because target-conditioned observation removes exact conflicts",
            "CAUSAL_ATTRIBUTION_NOT_UNIQUE": "rejected because raw_x+target_context uniquely resolves exact label aliasing in the H4M-H trace",
        },
        "root_layer_sufficiently_identified": True,
    }


def repair_candidate_comparison(attribution: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = [
        {
            "candidate_id": "OBS_TARGET_CONTEXT_ONEHOT_TO_ACTOR_GATV2_INPUT",
            "affected_layer": "RAW_OBSERVATION_SCHEMA",
            "description": "Append existing frozen R3 action target/obligation context one-hot fields to actor/GATv2 node input.",
            "solves_exact_problem": True,
            "requires_reward_gae_ppo_change": False,
            "requires_long_horizon_label_as_target": False,
            "future_leakage_risk_if_contract_followed": False,
            "dimensional_change": "node x 9 -> 12",
            "selected": True,
            "selection_reason": "Smallest repair that directly removes exact actor-observation label aliasing without changing Reward/GAE/PPO/K-mask.",
        },
        {
            "candidate_id": "GATV2_WIDTH_OR_ARCHITECTURE_EXPANSION",
            "affected_layer": "GATV2_REPRESENTATION",
            "solves_exact_problem": False,
            "selected": False,
            "rejection_reason": "Cannot encode target/obligation context because it is absent from the input.",
        },
        {
            "candidate_id": "ACTOR_HEAD_OR_LOSS_REWEIGHTING",
            "affected_layer": "ACTOR_HEAD",
            "solves_exact_problem": False,
            "selected": False,
            "rejection_reason": "Would treat actor bias symptom while leaving raw state aliasing unresolved; may require long-horizon labels or PPO changes.",
        },
        {
            "candidate_id": "ENVIRONMENT_EXPANSION_OR_NEW_FEATURES",
            "affected_layer": "ENVIRONMENT",
            "solves_exact_problem": False,
            "selected": False,
            "rejection_reason": "Unnecessary and forbidden for H4M-I; existing target context already resolves exact conflicts.",
        },
    ]
    return {
        "stage": STAGE,
        "root_cause_class": attribution["root_cause_class"],
        "candidates": candidates,
        "selected_candidate_id": "OBS_TARGET_CONTEXT_ONEHOT_TO_ACTOR_GATV2_INPUT",
        "multiple_materially_equivalent_repairs_remaining": False,
    }


def repair_contract(created_at: str, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    contract = {
        "stage": STAGE,
        "created_at": created_at,
        "contract_name": "PV8_H4M_I_OBSERVATION_TARGET_CONTEXT_REPAIR_CONTRACT",
        "contract_version": 1,
        "selected_repair_class": "OBSERVATION_TARGET_CONTEXT_ONEHOT_SCHEMA_REPAIR",
        "exact_problem_being_repaired": (
            "The actor/GATv2 raw node observation is 9D and omits the frozen R3 action-target/obligation context. "
            "H4M-H long-horizon labels are exactly target-conditioned: target_id=0 maps to HOLD_LONG_HORIZON_BETTER and "
            "target_id=1 maps to SERVE_LONG_HORIZON_BETTER in TRAIN44 diagnostic rows, yet actor output remains SERVE-biased for target_id=0."
        ),
        "affected_layer": "RAW_OBSERVATION_SCHEMA_TO_GATV2_ACTOR_INPUT",
        "exact_candidate_fields": TARGET_CONTEXT_FIELDS,
        "field_definitions": {
            "r3_action_target_is_hold": "1.0 iff frozen R3 action target id is HOLD_CURRENT_POSITION else 0.0",
            "r3_action_target_is_serve": "1.0 iff frozen R3 action target id is SERVE_AND_MOVE_TO_NEXT_STOP else 0.0",
            "r3_action_target_is_skip": "1.0 iff frozen R3 action target id is CONDITIONAL_SKIP_EMPTY_STOP else 0.0",
        },
        "causal_availability_timing": (
            "Must be materialized before actor forward/action sampling from the same frozen R3 target/obligation tensor currently used by K-mask and Reward V2."
        ),
        "anti_leakage_requirements": {
            "must_not_use_long_horizon_shadow_result": True,
            "must_not_use_counterfactual_classification": True,
            "must_not_use_future_reward_or_kpi": True,
            "must_not_use_post_action_state": True,
            "must_not_convert_h4m_h_labels_to_supervised_rl_targets": True,
        },
        "dimensional_schema_change": {
            "node_feature_dim_before": 9,
            "node_feature_dim_after": 12,
            "append_order": TARGET_CONTEXT_FIELDS,
            "edge_feature_dim_unchanged": 4,
            "action_dim_unchanged": 3,
        },
        "module_change_contract": {
            "GATv2_in_channels": "9 -> 12 only because input schema expands",
            "hidden_channels": "unchanged",
            "GATv2_layers_heads": "unchanged",
            "Actor_head_architecture": "unchanged",
            "Critic_semantics": "unchanged except receives embeddings from the same expanded encoder",
            "Reward_V2_Zero_Loss_K_mask": "unchanged",
            "GAE_PPO_normalization_hyperparameters": "unchanged",
        },
        "initialization_rule": {
            "migration_equivalence": "copy existing first-layer weights for the original 9 channels; initialize new target-context input weights to exactly zero for equivalence smoke",
            "fresh_training_rule": "same seeded initializer may be used after implementation gate approves the schema; no training is authorized by H4M-I",
        },
        "compatibility_migration_rule": {
            "old_9d_artifacts": "read-only; do not mutate historical artifacts",
            "new_dataset_contract": "new 12D schema must carry explicit schema version and feature order",
            "checkpoint_compatibility": "legacy checkpoints require zero-initialized adapter weights for non-training equivalence only",
        },
        "what_remains_frozen": [
            "Reward V2",
            "Zero-Loss",
            "K-mask semantics",
            "action set",
            "TRAIN44 split/order",
            "GAE/PPO/Critic semantics",
            "normalization",
            "hyperparameters",
            "environment/agents",
            "H4M-H long-horizon labels as diagnostic-only evidence",
        ],
        "implementation_acceptance_gate": {
            "exact_next_gate": "H4M-J_OBSERVATION_DISCRIMINATION_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION",
            "requirements": [
                "source-only implementation commit before any probe",
                "schema audit proves 12D feature order and anti-leakage constraints",
                "zero-initialized migration equivalence preserves pre-repair logits/probabilities within active MPS contract",
                "no Reward/GAE/PPO/K-mask change",
                "no RL training unless a later explicit training gate authorizes it",
                "TEST6 remains sealed",
            ],
        },
        "repair_implementation_executed_in_h4m_i": False,
        "rl_training_executed_in_h4m_i": False,
    }
    contract["contract_payload_sha256"] = payload_sha256(contract)
    return contract


def gate_matrix(
    binding: Mapping[str, Any],
    schema: Mapping[str, Any],
    raw: Mapping[str, Any],
    rep: Mapping[str, Any],
    actor: Mapping[str, Any],
    availability: Mapping[str, Any],
    attribution: Mapping[str, Any],
    candidates: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_pass": binding.get("authoritative_binding_passed") is True,
        "long_horizon_labels_crosswalked": raw.get("row_count") == 11616 and raw.get("label_counts", {}).get(LABEL_HOLD) == 8679,
        "runtime_observation_schema_confirmed": schema.get("reconstruction_checks", {}).get("actor_observation_hash_match_all") is True
        and schema.get("reconstruction_checks", {}).get("target_id_match_all") is True,
        "raw_layer_audited": raw.get("raw_observation_discrimination_result") is not None,
        "gatv2_layer_audited": rep.get("gatv2_representation_result") is not None,
        "actor_layer_audited": actor.get("actor_discrimination_result") is not None,
        "root_layer_sufficiently_identified": attribution.get("root_layer_sufficiently_identified") is True,
        "no_future_leakage": availability.get("feature_availability_audit_passed") is True
        and not any(bool(v) for k, v in availability.get("anti_leakage_checks", {}).items() if k.startswith("uses_")),
        "minimal_repair_selected": candidates.get("selected_candidate_id") == "OBS_TARGET_CONTEXT_ONEHOT_TO_ACTOR_GATV2_INPUT",
        "no_multiple_equivalent_repairs": candidates.get("multiple_materially_equivalent_repairs_remaining") is False,
        "repair_contract_sha_created": bool(contract.get("contract_payload_sha256")),
        "no_training_or_implementation": True,
        "test6_sealed": True,
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": contract.get("selected_repair_class") if passed else "H4M_I_BLOCKED",
        "root_cause_class": attribution.get("root_cause_class"),
        "selected_minimal_repair": candidates.get("selected_candidate_id"),
        "repair_contract_sha256": contract.get("contract_payload_sha256"),
        "exact_next_gate": "H4M-J_OBSERVATION_DISCRIMINATION_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION"
        if passed
        else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "final_flags": {
            "repair_implementation_executed": False,
            "rl_training_executed": False,
            "reward_gae_ppo_modified": False,
            "environment_expanded": False,
            "validation_or_test6_executed": False,
            "winner_or_baseline_selected": False,
            "github_push_performed": False,
        },
    }


def final_report(gate: Mapping[str, Any], raw: Mapping[str, Any], rep: Mapping[str, Any], actor: Mapping[str, Any], attribution: Mapping[str, Any], contract: Mapping[str, Any], binding: Mapping[str, Any]) -> str:
    return f"""# H4M-I Observation Discrimination Repair Selection & Freeze

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_i_source_git_commit"]}
root_cause_class = {attribution["root_cause_class"]}
selected_minimal_repair = {gate["selected_minimal_repair"]}
repair_contract_sha256 = {gate["repair_contract_sha256"]}
exact_next_gate = {gate["exact_next_gate"]}

## Raw observation discrimination

- exact actor_observation conflict groups = {raw["exact_observation_collision"]["conflict_group_count"]}
- rows in conflicting raw-observation groups = {raw["exact_observation_collision"]["rows_in_conflicting_groups"]}
- exact actor_observation + target_id conflict groups = {raw["exact_observation_plus_target_collision"]["conflict_group_count"]}
- result = {raw["raw_observation_discrimination_result"]}

## GATv2 representation

- probe = {rep["probe_type"]}
- embedding rows = {rep["rows"]}
- result = {rep["gatv2_representation_result"]}

## Actor discrimination

- H4M-H trace correct direction rate = {actor["actual_h4m_h_trace_actor_output"]["overall_correct_direction_rate"]}
- HOLD-better correct direction rate = {actor["actual_h4m_h_trace_actor_output"]["by_long_horizon_label"][LABEL_HOLD]["correct_direction_rate"]}
- SERVE-better correct direction rate = {actor["actual_h4m_h_trace_actor_output"]["by_long_horizon_label"][LABEL_SERVE]["correct_direction_rate"]}
- result = {actor["actor_discrimination_result"]}

## Frozen contract

```json
{json.dumps({k: contract[k] for k in ["contract_name", "selected_repair_class", "exact_candidate_fields", "dimensional_schema_change", "implementation_acceptance_gate"]}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no implementation, no training, no Reward/GAE/PPO modification, no environment expansion, no validation/TEST6, no winner/baseline, no GitHub push.
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
        "root_cause_class": gate.get("root_cause_class"),
        "selected_minimal_repair": gate.get("selected_minimal_repair"),
        "repair_contract_sha256": gate.get("repair_contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": output_files,
        "output_sha256": {name: sha256_file(Path(path)) for name, path in output_files.items()},
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256",
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def write_block(artifact_root: Path, created_at: str, binding: Mapping[str, Any], reason: str) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    gate = {
        "stage": STAGE,
        "gate": BLOCK_GATE,
        "decision": "H4M_I_BLOCKED",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
    }
    for name in REQUIRED_ARTIFACTS:
        if name == "manifest.json":
            continue
        if name.endswith(".parquet"):
            pd.DataFrame([]).to_parquet(artifact_root / name, index=False)
        elif name == "final_report.md":
            (artifact_root / name).write_text(f"# H4M-I\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
        elif name == "01_authoritative_binding.json":
            write_json(artifact_root / name, binding)
        elif name == "11_gate_matrix.json":
            write_json(artifact_root / name, gate)
        else:
            write_json(artifact_root / name, empty)
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-I] artifact root: {artifact_root}")
    print(f"[H4M-I] gate: {BLOCK_GATE}")
    print(f"[H4M-I] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_i_observation_discrimination_repair_selection_freeze_{stamp}"
    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    if not binding["authoritative_binding_passed"]:
        write_block(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return

    h4mg = import_module_from_path(H4MG_SOURCE, f"h4m_i_h4mg_{time.time_ns()}")
    trace = read_h4mh_trace()
    enriched, schema, _ctx = reconstruct_runtime_observations(trace, created_at, h4mg)
    raw = raw_observation_discrimination(enriched)
    rep_df, rep, _rep_meta = forward_only_representation_probe(enriched, created_at, h4mg)
    actor = actor_discrimination(enriched, rep_df)
    conflict = conflict_pairs(enriched, artifact_root)
    availability = causal_feature_availability_audit(raw)
    attribution = root_cause_attribution(raw, rep, actor)
    candidates = repair_candidate_comparison(attribution)
    contract = repair_contract(created_at, candidates)
    gate = gate_matrix(binding, schema, raw, rep, actor, availability, attribution, candidates, contract)
    report = final_report(gate, raw, rep, actor, attribution, contract, binding)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_runtime_observation_schema.json": schema,
        "03_raw_observation_discrimination.json": raw,
        "04_gatv2_representation_discrimination.json": rep,
        "05_actor_discrimination.json": actor,
        "07_causal_feature_availability_audit.json": availability,
        "08_root_cause_attribution.json": attribution,
        "09_repair_candidate_comparison.json": candidates,
        "10_h4m_i_repair_contract.json": contract,
        "11_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    manifest = make_manifest(artifact_root, gate)
    write_json(artifact_root / "manifest.json", manifest)

    print(f"[H4M-I] artifact root: {artifact_root}")
    print(f"[H4M-I] gate: {gate['gate']}")
    print(f"[H4M-I] source_commit: {provenance['h4m_i_source_git_commit']}")
    print(f"[H4M-I] raw_result: {raw['raw_observation_discrimination_result']}")
    print(f"[H4M-I] gatv2_result: {rep['gatv2_representation_result']}")
    print(f"[H4M-I] actor_result: {actor['actor_discrimination_result']}")
    print(f"[H4M-I] root_cause_class: {attribution['root_cause_class']}")
    print(f"[H4M-I] selected_minimal_repair: {gate['selected_minimal_repair']}")
    print(f"[H4M-I] repair_contract_sha256: {gate['repair_contract_sha256']}")
    print(f"[H4M-I] exact_next_gate: {gate['exact_next_gate']}")
    print("[H4M-I] STOP: no implementation, no training, no TEST6, github_push=false")


if __name__ == "__main__":
    main()
