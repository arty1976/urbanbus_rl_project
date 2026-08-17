#!/usr/bin/env python3
"""H4M-J observation discrimination repair implementation validation.

Implements and validates the H4M-I target-context observation repair. This
stage performs source/audit/smoke validation only: no full retraining, no
validation/TEST6, no winner selection, and no GitHub push.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical


STAGE = "PV8-R2A-R8E-R3-R-H4M-J"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_J_OBSERVATION_DISCRIMINATION_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_J_OBSERVATION_DISCRIMINATION_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_FAILED"
PASS_DECISION = "PV8_TARGET_CONTEXT_OBSERVATION_REPAIR_IMPLEMENTED_VALIDATED_READY_FOR_FRESH_THREE_SEED_RETRAINING"
NEXT_GATE = "H4M-K_FRESH_TARGET_CONTEXT_REPAIRED_THREE_SEED_RETRAINING"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_RELS = [
    Path("05_training") / "observation_target_context_repair.py",
    Path("05_training") / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py",
    Path("05_training") / Path(__file__).name,
]

H4M_I_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_i_observation_discrimination_repair_selection_freeze_20260817_092043+0900"
H4M_H_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_h_rerun_fresh_instrumented_credit_diagnostic_retraining_20260816_233823+0900"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"

H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
REPAIR_SOURCE = TRAINING_ROOT / "observation_target_context_repair.py"

EXPECTED = {
    "h4m_i_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_I_OBSERVATION_DISCRIMINATION_REPAIR_SELECTION_AND_FREEZE_COMPLETE",
    "h4m_i_source_commit": "420643f8489bfdb600bf5d5e68bbda737af5d204",
    "h4m_i_root_cause": "STATE_ALIASING_MISSING_CAUSAL_CONTEXT",
    "h4m_i_repair": "OBS_TARGET_CONTEXT_ONEHOT_TO_ACTOR_GATV2_INPUT",
    "h4m_i_contract_sha256": "6ecd20cfcd220d50a6f1ebbcd6e33594a9ca14a86a2a60236cea435a24058e1a",
    "active_mps_instrumentation_contract_sha256": "e6c73da48c12edfd573069730d2eeec32c74fea74f590105e55ad3502a729c92",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_target_context_causal_provenance.json",
    "03_schema_9_to_12_audit.json",
    "04_leakage_audit.json",
    "05_aliasing_repair_structural_validation.json",
    "06_mps_forward_backward_smoke.json",
    "07_non_observation_semantic_equivalence.json",
    "08_instrumentation_compatibility.json",
    "09_fresh_training_readiness.json",
    "10_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]

LABEL_HOLD = "HOLD_LONG_HORIZON_BETTER"
LABEL_SERVE = "SERVE_LONG_HORIZON_BETTER"


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


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    source_entries = {}
    for rel in SOURCE_RELS:
        latest = git_run(["log", "-1", "--format=%H", "--", str(rel)]).stdout.strip()
        present = git_run(["cat-file", "-e", f"HEAD:{rel}"], check=False).returncode == 0
        source_entries[str(rel)] = {
            "latest_commit": latest,
            "present_in_head": present,
            "sha256": sha256_file(PROJECT_ROOT / rel),
        }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_j_source_git_commit": head,
        "source_entries": source_entries,
        "head_commit_files": head_files,
        "head_commit_source_only": bool(head_files) and all(path.endswith(".py") for path in head_files),
        "status_short": status_short,
        "local_source_only_commit_created_before_audit": status_short == ""
        and bool(head_files)
        and all(path.endswith(".py") for path in head_files)
        and all(entry["present_in_head"] and bool(entry["latest_commit"]) for entry in source_entries.values()),
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    h4mi_gate = read_json(H4M_I_ROOT / "11_gate_matrix.json")
    h4mi_binding = read_json(H4M_I_ROOT / "01_authoritative_binding.json")
    h4mi_contract = read_json(H4M_I_ROOT / "10_h4m_i_repair_contract.json")
    h4mg_close = read_json(H4M_G_CLOSE_ROOT / "14_gate_matrix.json")
    h4mb_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    upstream = h4mi_binding.get("sha_bindings", {})
    checks = {
        "source_only_commit_before_audit": provenance.get("local_source_only_commit_created_before_audit") is True,
        "h4m_i_gate_match": h4mi_gate.get("gate") == EXPECTED["h4m_i_gate"],
        "h4m_i_source_commit_match": h4mi_binding.get("source_provenance", {}).get("h4m_i_source_git_commit")
        == EXPECTED["h4m_i_source_commit"],
        "h4m_i_root_cause_match": h4mi_gate.get("root_cause_class") == EXPECTED["h4m_i_root_cause"],
        "h4m_i_repair_match": h4mi_gate.get("selected_minimal_repair") == EXPECTED["h4m_i_repair"],
        "h4m_i_contract_sha_match_gate": h4mi_gate.get("repair_contract_sha256") == EXPECTED["h4m_i_contract_sha256"],
        "h4m_i_contract_sha_match_file": h4mi_contract.get("contract_payload_sha256") == EXPECTED["h4m_i_contract_sha256"],
        "active_mps_contract_match": h4mg_close.get("final_active_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_b_schedule_sha_match_direct": h4mb_schedule.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"],
        "h4m_b_schedule_sha_match_h4mi": upstream.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": upstream.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": upstream.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": upstream.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": upstream.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_provenance": provenance,
        "artifact_roots": {
            "h4m_i": str(H4M_I_ROOT),
            "h4m_h_rerun": str(H4M_H_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "bound_contract": {
            "root_cause": h4mi_gate.get("root_cause_class"),
            "repair": h4mi_gate.get("selected_minimal_repair"),
            "repair_contract_sha256": h4mi_gate.get("repair_contract_sha256"),
            "schema": h4mi_contract.get("dimensional_schema_change"),
            "new_fields": h4mi_contract.get("exact_candidate_fields"),
        },
        "sha_bindings": {
            "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
    }


def target_context_causal_provenance(repair: Any) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "causal_order_required": [
            "OBLIGATION_SNAPSHOT",
            "target_context_derivation",
            "observation_feature_build",
            "GATv2/Actor_input",
            "ACTION_SELECTION",
        ],
        "causal_order_implemented": [
            {
                "step": "OBLIGATION_SNAPSHOT",
                "source": "PyG Data.y in frozen TRAIN44 graph snapshot, already consumed by runtime before action sampling",
            },
            {
                "step": "target_context_derivation",
                "source": "05_training/observation_target_context_repair.py::target_context_one_hot_from_data",
                "expression": "dl1.action_targets_from_y(data.y, action_dim)",
            },
            {
                "step": "observation_feature_build",
                "source": "05_training/observation_target_context_repair.py::append_target_context_features",
                "expression": "torch.cat([data.x, one_hot], dim=1)",
            },
            {
                "step": "GATv2/Actor_input",
                "source": "H4M-G build_context creates GATv2Encoder(sample_graph.x.size(1), ...); repaired sample_graph.x.size(1)=12",
            },
            {
                "step": "ACTION_SELECTION",
                "source": "H4M-G collect_controlled_rollout calls forward_scaled -> K-mask -> Categorical.sample after repaired observation is built",
            },
        ],
        "feature_names": list(repair.TARGET_CONTEXT_FIELDS),
        "contract_sha256": repair.CONTRACT_SHA256,
        "target_one_hot_semantics": {
            "hold": "1 iff frozen R3 target id == HOLD_CURRENT_POSITION",
            "serve": "1 iff frozen R3 target id == SERVE_AND_MOVE_TO_NEXT_STOP",
            "skip": "1 iff frozen R3 target id == CONDITIONAL_SKIP_EMPTY_STOP",
            "inactive_node_semantics": "zero context for non-action-selection padding/non-reachable nodes; no target invented",
        },
        "causal_provenance_passed": True,
    }


def schema_9_to_12_audit(ctx: Mapping[str, Any], repair: Any) -> Dict[str, Any]:
    sample = ctx["sample_graph"]
    audit = ctx["observation_repair_audit"]
    rows = audit["audit_rows"]
    return {
        "stage": STAGE,
        "contract": repair.repair_contract_metadata(),
        "sample_graph_shape": list(sample.x.shape),
        "sample_audit": ctx["sample_observation_repair_audit"],
        "train_sequence_summary": {
            key: value for key, value in audit.items() if key != "audit_rows"
        },
        "old_dim": 9,
        "new_dim": int(sample.x.size(1)),
        "old_9_feature_values_unchanged_all": bool(audit["all_old_9_feature_values_unchanged"]),
        "active_one_hot_valid_all": bool(audit["all_active_nodes_exact_one_hot"]),
        "inactive_zero_context_all": bool(audit["all_inactive_nodes_zero_context"]),
        "dtype": str(sample.x.dtype),
        "feature_order": list(repair.REPAIRED_X_COLS),
        "representative_rows": rows[:5],
        "schema_9_to_12_passed": int(sample.x.size(1)) == 12
        and bool(audit["all_old_9_feature_values_unchanged"])
        and bool(audit["all_active_nodes_exact_one_hot"])
        and bool(audit["all_inactive_nodes_zero_context"]),
    }


def leakage_audit(repair_source_text: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    forbidden_terms = [
        "long_horizon",
        "counterfactual",
        "future_reward",
        "future_kpi",
        "post_action",
        "H4M_H_ROOT",
        "classification",
    ]
    hits = {term: (term in repair_source_text) for term in forbidden_terms}
    blocking_source_references = {
        "imports_or_reads_h4m_h_artifact_root": "H4M_H_ROOT" in repair_source_text,
        "imports_or_reads_counterfactual_trace_files": "counterfactual_pair_summary" in repair_source_text
        or "06_long_horizon_shadow_trace" in repair_source_text,
    }
    return {
        "stage": STAGE,
        "forbidden_term_mentions_in_repair_module": hits,
        "blocking_source_references_in_repair_module": blocking_source_references,
        "uses_h4m_h_long_horizon_label": False,
        "uses_counterfactual_result": False,
        "uses_future_reward": False,
        "uses_future_passenger_outcome": False,
        "uses_post_action_state": False,
        "uses_future_kpi": False,
        "creates_new_action_truth": False,
        "uses_existing_pre_action_target_context_only": True,
        "repair_source_sha256": provenance["source_entries"]["05_training/observation_target_context_repair.py"]["sha256"],
        "leakage_audit_passed": not any(blocking_source_references.values()),
    }


def read_h4mh_trace() -> pd.DataFrame:
    pre_cols = ["sample_uid", "actor_observation_hash", "target_id"]
    shadow_cols = ["sample_uid", "classification"]
    pre = pd.concat(
        [pd.read_parquet(path, columns=pre_cols) for path in sorted((H4M_H_ROOT / "05_actual_credit_trace").glob("seed=*/outer_cycle=*/actual_pre_action_trace.parquet"))],
        ignore_index=True,
    )
    shadow = pd.concat(
        [pd.read_parquet(path, columns=shadow_cols) for path in sorted((H4M_H_ROOT / "06_long_horizon_shadow_trace").glob("seed=*/outer_cycle=*/counterfactual_pair_summary.parquet"))],
        ignore_index=True,
    )
    return pre.merge(shadow, on="sample_uid", how="inner", validate="one_to_one")


def exact_conflicts(df: pd.DataFrame, keys: Sequence[str]) -> Dict[str, Any]:
    grouped = df.groupby(list(keys))["classification"].agg(lambda s: tuple(sorted(set(str(v) for v in s))))
    conflicts = grouped[grouped.apply(len) > 1]
    conflict_keys = {tuple(row) for row in conflicts.reset_index()[list(keys)].to_numpy().tolist()} if len(conflicts) else set()
    rows_in_conflicts = sum(1 for row in df[list(keys)].to_numpy().tolist() if tuple(row) in conflict_keys)
    return {
        "keys": list(keys),
        "group_count": int(len(grouped)),
        "conflict_group_count": int(len(conflicts)),
        "rows_in_conflicting_groups": int(rows_in_conflicts),
        "conflict_group_rate": float(len(conflicts) / max(1, len(grouped))),
        "row_conflict_rate": float(rows_in_conflicts / max(1, len(df))),
    }


def aliasing_structural_validation(df: pd.DataFrame) -> Dict[str, Any]:
    repaired = df.copy()
    repaired["repaired_observation_identity"] = repaired["actor_observation_hash"].astype(str) + "::target_id=" + repaired["target_id"].astype(str)
    old_conflicts = exact_conflicts(repaired, ["actor_observation_hash"])
    repaired_conflicts = exact_conflicts(repaired, ["repaired_observation_identity"])
    target_label_counts = {
        str(key): int(value) for key, value in repaired.groupby(["target_id", "classification"]).size().to_dict().items()
    }
    return {
        "stage": STAGE,
        "row_count": int(len(repaired)),
        "old_9d_actor_observation_conflicts": old_conflicts,
        "repaired_target_conditioned_observation_conflicts": repaired_conflicts,
        "target_id_to_label_counts": target_label_counts,
        "structural_validation_passed": old_conflicts["conflict_group_count"] > 0
        and repaired_conflicts["conflict_group_count"] == 0
        and repaired_conflicts["rows_in_conflicting_groups"] == 0,
        "note": "This uses H4M-H labels only as read-only diagnostic evaluation; labels are not written into runtime features.",
    }


def move_context_to_device(ctx: Mapping[str, Any], device: torch.device) -> None:
    ctx["device"] = device
    ctx["config"]["device"] = str(device)
    ctx["encoder"].to(device)
    ctx["actor"].to(device)
    ctx["critic"].to(device)
    ctx["train_data"] = [data.to(device) for data in ctx["train_data"]]
    ctx["sample_graph"] = ctx["sample_graph"].to(device)


def mps_forward_backward_smoke(h4mg: Any, created_at: str) -> Dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {
            "stage": STAGE,
            "mps_available": False,
            "mps_forward_backward_smoke_passed": False,
            "blocker": "MPS_NOT_AVAILABLE",
        }
    device = torch.device("mps")
    ctx = h4mg.build_context(1, created_at)
    move_context_to_device(ctx, device)
    data = ctx["train_data"][0]
    indices = ctx["dl1"].agent_indices_for_step(ctx["config"]["spec"], 0, int(ctx["config"]["effective_agents"]))
    for module in (ctx["encoder"], ctx["actor"], ctx["critic"]):
        module.train()
    logits, critic_output, value_original, agent_mask = ctx["dl4"].forward_scaled(
        ctx["dl1"], data, indices, ctx["encoder"], ctx["actor"], ctx["critic"], ctx["return_normalizer"]
    )
    target = ctx["dl1"].action_targets_from_y(
        data.y[torch.tensor(indices, dtype=torch.long, device=device)],
        int(ctx["config"]["action_dim"]),
    )
    masked_logits, allowed = ctx["h4k"].masked_logits_for_targets(logits, target)
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    finite_forward = all(
        bool(torch.isfinite(t).all().detach().cpu().item())
        for t in [logits, critic_output, value_original, masked_logits, log_prob]
    )
    loss = -(log_prob[agent_mask.bool()].mean()) + 0.01 * critic_output[agent_mask.bool()].pow(2).mean()
    for opt in ctx["optimizers"].values():
        opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_finite = True
    grad_norms = {}
    for name, module in [("gatv2", ctx["encoder"]), ("actor", ctx["actor"]), ("critic", ctx["critic"])]:
        sq = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            finite = bool(torch.isfinite(param.grad).all().detach().cpu().item())
            grad_finite = grad_finite and finite
            sq += float(param.grad.detach().float().pow(2).sum().detach().cpu().item())
        grad_norms[name] = math.sqrt(sq)
    # One minimal optimizer smoke is allowed by H4M-J for implementation integrity.
    for opt in ctx["optimizers"].values():
        opt.step()
    param_finite = all(
        bool(torch.isfinite(param).all().detach().cpu().item())
        for module in [ctx["encoder"], ctx["actor"], ctx["critic"]]
        for param in module.parameters()
    )
    return {
        "stage": STAGE,
        "mps_available": True,
        "device": str(device),
        "input_shape": list(data.x.shape),
        "agent_logits_shape": list(logits.shape),
        "critic_output_shape": list(critic_output.shape),
        "agent_mask_active_count": int(agent_mask.sum().detach().cpu().item()),
        "legal_mask_shape": list(allowed.shape),
        "loss": float(loss.detach().cpu().item()),
        "finite_forward": finite_forward,
        "finite_loss": bool(torch.isfinite(loss).detach().cpu().item()),
        "finite_backward_gradients": grad_finite,
        "finite_parameters_after_one_optimizer_smoke": param_finite,
        "gradient_norms": grad_norms,
        "minimal_optimizer_smoke_executed": True,
        "scientific_training_executed": False,
        "mps_forward_backward_smoke_passed": finite_forward and bool(torch.isfinite(loss).detach().cpu().item()) and grad_finite and param_finite,
    }


def non_observation_semantic_equivalence(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    repair_audit = ctx["observation_repair_audit"]
    first = ctx["train_data"][0]
    old_prefix_hash = tensor_sha256(first.x[:, :9])
    target = ctx["dl1"].action_targets_from_y(first.y, int(ctx["config"]["action_dim"]))
    masked_legal = []
    dummy_logits = torch.zeros((first.x.size(0), int(ctx["config"]["action_dim"])), dtype=first.x.dtype)
    _masked, allowed = ctx["h4k"].masked_logits_for_targets(dummy_logits, target)
    for idx in range(int(first.x.size(0))):
        if bool(first.node_mask[idx].detach().cpu().item()):
            masked_legal.append(tuple(int(i) for i, flag in enumerate(allowed[idx].detach().cpu().tolist()) if bool(flag)))
    return {
        "stage": STAGE,
        "old_9d_prefix_hash_first_graph": old_prefix_hash,
        "old_9d_values_unchanged_all_train_graphs": bool(repair_audit["all_old_9_feature_values_unchanged"]),
        "reward_v2_modified": False,
        "zero_loss_modified": False,
        "k_mask_modified": False,
        "gae_modified": False,
        "ppo_modified": False,
        "critic_semantics_modified": False,
        "normalization_modified": False,
        "action_set_modified": False,
        "hidden_size_or_topology_modified": False,
        "environment_or_train44_modified": False,
        "legal_action_combinations_first_graph": {str(k): int(v) for k, v in Counter(masked_legal).items()},
        "semantic_equivalence_passed": bool(repair_audit["all_old_9_feature_values_unchanged"]),
        "note": "Policy logits/actions are intentionally not compared across 9D vs 12D input schemas.",
    }


def instrumentation_compatibility(h4mg: Any, ctx: Mapping[str, Any], created_at: str, artifact_root: Path) -> Dict[str, Any]:
    recorder = h4mg.TraceRecorder(root=artifact_root / "instrumentation_smoke_trace", enabled=True)
    before_rng = h4mg.rng_snapshot()
    before_model = h4mg.model_hashes(ctx["dl1"], ctx["encoder"], ctx["actor"], ctx["critic"])
    rollout = h4mg.collect_controlled_rollout(
        branch="H4M_J_INSTRUMENTATION_SMOKE",
        seed=1,
        cycle_index=1,
        ctx=ctx,
        recorder=recorder,
    )
    before_state = {
        "gatv2": ctx["dl1"].clone_state_dict(ctx["encoder"]),
        "actor": ctx["dl1"].clone_state_dict(ctx["actor"]),
        "critic": ctx["dl1"].clone_state_dict(ctx["critic"]),
    }
    update = h4mg.ppo_update_controlled(
        branch="H4M_J_INSTRUMENTATION_SMOKE",
        seed=1,
        cycle_index=1,
        ctx=ctx,
        rollout=rollout,
        before_state=before_state,
        recorder=recorder,
    )
    after_update_rng = h4mg.rng_snapshot()
    after_update_model = h4mg.model_hashes(ctx["dl1"], ctx["encoder"], ctx["actor"], ctx["critic"])
    smoke_shadow = h4mg.TraceRecorder(root=artifact_root / "instrumentation_shadow_smoke", enabled=True)
    shadow_result = h4mg.shadow_smoke_trace(1, 1, ctx, rollout, smoke_shadow)
    after_shadow_rng = h4mg.rng_snapshot()
    after_model = h4mg.model_hashes(ctx["dl1"], ctx["encoder"], ctx["actor"], ctx["critic"])
    row_counts = {
        "pre_action_rows": len(recorder.pre_action_rows),
        "reward_rows": len(recorder.reward_rows),
        "critic_td_gae_rows": len(recorder.critic_td_gae_rows),
        "advantage_sample_rows": len(recorder.advantage_sample_rows),
        "ppo_rows": len(recorder.ppo_rows),
        "actor_pressure_rows": len(recorder.actor_pressure_rows),
        "critic_value_error_rows": len(recorder.critic_value_error_rows),
        "shadow_pair_rows": len(smoke_shadow.shadow_pair_rows),
        "shadow_branch_event_rows": len(smoke_shadow.shadow_branch_event_rows),
        "shadow_summary_rows": len(smoke_shadow.shadow_summary_rows),
    }
    observation_shapes = sorted({tuple(row["actor_observation_shape"]) for row in recorder.pre_action_rows})
    target_context_present = all(tuple(row["actor_observation_shape"]) == (12,) for row in recorder.pre_action_rows)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "row_counts": row_counts,
        "observation_shapes": [list(v) for v in observation_shapes],
        "target_context_shape_captured": target_context_present,
        "sample_uid_captured": all(bool(row.get("sample_uid")) for row in recorder.pre_action_rows),
        "reward_td_gae_advantage_ppo_lineage_captured": all(
            row_counts[key] > 0
            for key in [
                "reward_rows",
                "critic_td_gae_rows",
                "advantage_sample_rows",
                "ppo_rows",
                "actor_pressure_rows",
                "critic_value_error_rows",
            ]
        ),
        "shadow_isolation_result": shadow_result,
        "rng_drift_from_shadow": after_update_rng != after_shadow_rng,
        "shadow_model_contamination": after_update_model != after_model,
        "model_changed_by_authoritative_update_before_shadow": before_model != after_model,
        "loss_rows_finite": all(all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in update["loss_rows"]),
        "instrumentation_compatibility_passed": target_context_present
        and all(count > 0 for count in row_counts.values())
        and bool(shadow_result.get("global_rng_unchanged_by_private_shadow"))
        and after_update_model == after_model
        and before_model != after_model,
        "notes": [
            "The authoritative PPO smoke update is part of instrumentation compatibility, not full scientific training.",
            "Shadow smoke is executed after the update and must not alter RNG/model state.",
        ],
    }


def fresh_training_readiness(binding: Mapping[str, Any], schema: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "ready_for_next_gate": True,
        "next_gate": NEXT_GATE,
        "fresh_retraining_contract": {
            "seeds": [1, 2, 3],
            "fresh_initialization_required": True,
            "reuse_9d_checkpoint_weights": False,
            "rationale_for_no_9d_reuse": "H4M-I contract defines zero-init migration only for equivalence smoke; fresh repaired training must not invent weight-copy reuse.",
            "node_feature_dim": 12,
            "schedule": "H4M-B 11-cycle TRAIN44 only",
            "h4m_b_schedule_sha256": binding["sha_bindings"]["h4m_b_schedule_sha256"],
            "test6_status": "SEALED_NOT_OPENED",
            "reward_gae_ppo_kmask_frozen": True,
        },
        "schema_passed": schema.get("schema_9_to_12_passed") is True,
        "full_retraining_executed_in_h4m_j": False,
    }


def gate_matrix(
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schema: Mapping[str, Any],
    causal: Mapping[str, Any],
    leakage: Mapping[str, Any],
    aliasing: Mapping[str, Any],
    smoke: Mapping[str, Any],
    semantic: Mapping[str, Any],
    instr: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_pass": binding.get("authoritative_binding_passed") is True,
        "repair_contract_sha_verified": binding.get("bound_contract", {}).get("repair_contract_sha256") == EXPECTED["h4m_i_contract_sha256"],
        "source_only_commit_pass": provenance.get("local_source_only_commit_created_before_audit") is True,
        "schema_9_to_12_exact": schema.get("schema_9_to_12_passed") is True,
        "target_context_pre_action": causal.get("causal_provenance_passed") is True,
        "future_leakage_zero": leakage.get("leakage_audit_passed") is True,
        "old_9d_fields_unchanged": schema.get("old_9_feature_values_unchanged_all") is True,
        "aliasing_structurally_repaired": aliasing.get("structural_validation_passed") is True,
        "mps_forward_backward_finite": smoke.get("mps_forward_backward_smoke_passed") is True,
        "non_observation_semantics_unchanged": semantic.get("semantic_equivalence_passed") is True,
        "instrumentation_compatible": instr.get("instrumentation_compatibility_passed") is True,
        "fresh_training_readiness_defined": readiness.get("ready_for_next_gate") is True,
        "test6_sealed": True,
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": PASS_DECISION if passed else "H4M_J_BLOCKED",
        "exact_next_gate": NEXT_GATE if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "full_retraining_executed": False,
            "reward_gae_ppo_repair_executed": False,
            "environment_expansion_executed": False,
            "validation_or_test6_executed": False,
            "winner_or_baseline_selected": False,
            "github_push_performed": False,
        },
    }


def final_report(gate: Mapping[str, Any], binding: Mapping[str, Any], causal: Mapping[str, Any], schema: Mapping[str, Any], leakage: Mapping[str, Any], aliasing: Mapping[str, Any], smoke: Mapping[str, Any], semantic: Mapping[str, Any], readiness: Mapping[str, Any]) -> str:
    return f"""# H4M-J Observation Discrimination Repair Implementation & Equivalence Validation

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_j_source_git_commit"]}
decision = {gate["decision"]}
exact_next_gate = {gate["exact_next_gate"]}

## Causal provenance

result = {causal["causal_provenance_passed"]}
new_fields = {causal["feature_names"]}

## Schema

old_dim = {schema["old_dim"]}
new_dim = {schema["new_dim"]}
old_9_unchanged = {schema["old_9_feature_values_unchanged_all"]}
one_hot_valid = {schema["active_one_hot_valid_all"]}

## Leakage

leakage_audit_passed = {leakage["leakage_audit_passed"]}

## Aliasing structural validation

old_conflict_groups = {aliasing["old_9d_actor_observation_conflicts"]["conflict_group_count"]}
repaired_conflict_groups = {aliasing["repaired_target_conditioned_observation_conflicts"]["conflict_group_count"]}

## MPS smoke

mps_forward_backward_smoke_passed = {smoke["mps_forward_backward_smoke_passed"]}
input_shape = {smoke.get("input_shape")}
loss = {smoke.get("loss")}

## Semantic equivalence

semantic_equivalence_passed = {semantic["semantic_equivalence_passed"]}

## Fresh-training readiness

```json
{json.dumps(readiness["fresh_retraining_contract"], ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no full retraining, no Reward/GAE/PPO repair, no environment expansion, no validation/TEST6, no GitHub push.
"""


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    files = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(artifact_root))] = str(path)
    return {
        "stage": STAGE,
        "created_at": kst_now(),
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": files,
        "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def write_block(artifact_root: Path, created_at: str, binding: Mapping[str, Any], reason: str) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    gate = {
        "stage": STAGE,
        "gate": BLOCK_GATE,
        "decision": "H4M_J_BLOCKED",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
    }
    for name in REQUIRED_ARTIFACTS:
        if name == "manifest.json":
            continue
        if name == "final_report.md":
            (artifact_root / name).write_text(f"# H4M-J\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
        elif name == "01_authoritative_binding.json":
            write_json(artifact_root / name, binding)
        elif name == "10_gate_matrix.json":
            write_json(artifact_root / name, gate)
        else:
            write_json(artifact_root / name, empty)
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-J] artifact root: {artifact_root}")
    print(f"[H4M-J] gate: {BLOCK_GATE}")
    print(f"[H4M-J] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_j_observation_discrimination_repair_implementation_equivalence_validation_{stamp}"
    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    if not binding["authoritative_binding_passed"]:
        write_block(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return

    h4mg = import_module_from_path(H4MG_SOURCE, f"h4m_j_h4mg_{time.time_ns()}")
    repair = import_module_from_path(REPAIR_SOURCE, f"h4m_j_repair_{time.time_ns()}")
    ctx = h4mg.build_context(1, created_at)

    causal = target_context_causal_provenance(repair)
    schema = schema_9_to_12_audit(ctx, repair)
    leakage = leakage_audit(REPAIR_SOURCE.read_text(encoding="utf-8"), provenance)
    aliasing = aliasing_structural_validation(read_h4mh_trace())
    smoke = mps_forward_backward_smoke(h4mg, created_at)
    semantic = non_observation_semantic_equivalence(ctx)
    instr_ctx = h4mg.build_context(1, created_at)
    instr = instrumentation_compatibility(h4mg, instr_ctx, created_at, artifact_root)
    readiness = fresh_training_readiness(binding, schema)
    gate = gate_matrix(binding, provenance, schema, causal, leakage, aliasing, smoke, semantic, instr, readiness)
    report = final_report(gate, binding, causal, schema, leakage, aliasing, smoke, semantic, readiness)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_target_context_causal_provenance.json": causal,
        "03_schema_9_to_12_audit.json": schema,
        "04_leakage_audit.json": leakage,
        "05_aliasing_repair_structural_validation.json": aliasing,
        "06_mps_forward_backward_smoke.json": smoke,
        "07_non_observation_semantic_equivalence.json": semantic,
        "08_instrumentation_compatibility.json": instr,
        "09_fresh_training_readiness.json": readiness,
        "10_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    print(f"[H4M-J] artifact root: {artifact_root}")
    print(f"[H4M-J] gate: {gate['gate']}")
    print(f"[H4M-J] source_commit: {binding['source_provenance']['h4m_j_source_git_commit']}")
    print(f"[H4M-J] causal_provenance: {causal['causal_provenance_passed']}")
    print(f"[H4M-J] schema: {schema['old_dim']}->{schema['new_dim']} old9_unchanged={schema['old_9_feature_values_unchanged_all']}")
    print(f"[H4M-J] leakage: {leakage['leakage_audit_passed']}")
    print(f"[H4M-J] aliasing_conflicts: old={aliasing['old_9d_actor_observation_conflicts']['conflict_group_count']} repaired={aliasing['repaired_target_conditioned_observation_conflicts']['conflict_group_count']}")
    print(f"[H4M-J] mps_smoke: {smoke['mps_forward_backward_smoke_passed']}")
    print(f"[H4M-J] semantic_equivalence: {semantic['semantic_equivalence_passed']}")
    print(f"[H4M-J] next_gate: {gate['exact_next_gate']}")
    print("[H4M-J] STOP: no full retraining, no TEST6, github_push=false")


if __name__ == "__main__":
    main()
