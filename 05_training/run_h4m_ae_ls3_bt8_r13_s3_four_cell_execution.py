#!/usr/bin/env python3
"""BT8-R13: exact M2 four-cell S3 on-policy causal execution.

The runner intentionally creates four fresh causal rollouts.  It does not reuse
the historical F1 training batch because the cross cells would violate the
on-policy binding.  The only historical runtime inputs reused are the six
lossless F1 review snapshots, and those are replayed without a simulator,
candidate generation, Zero-Loss evaluation, or optimizer exposure.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R13"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_S3_FOUR_CELL_ON_POLICY_CAUSAL_EXECUTION_COMPLETE"
PASS_CLASS_COMPLETE = "A_SUSEONG_LS3_S3_FACTORIAL_ACTOR_TRAINING_COMPLETE_READY_FOR_FROZEN_FACTOR_REVIEW"
PASS_CLASS_INCOMPLETE = "B_SUSEONG_LS3_S3_EXECUTION_VALID_BUT_FACTORIAL_CREDIT_SUPPORT_INCOMPLETE"
MPS_BLOCK = "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
AUTH_BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_EXECUTION_AUTHORITY_OR_BINDING_FAILURE"
INTEGRITY_BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_EXECUTION_INTEGRITY_FAILURE"

R12_SOURCE = "6e22b95d80720255ca17b1ab0520417b89e5a5c9"
R11_SOURCE = "f467feef8246c6c77dc67a360aecc60e0102913e"
E1_SOURCE = "a3cc98280e434c41520245f3fa13e3a76dd5d438"
R4A_SOURCE = "eac4a209e09e696380bde3bbc437a4fd13c45e99"
F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
S3_CONTRACT_SHA256 = "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"
E1_CONTRACT_SHA256 = "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"
R12_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R12_S3_FACTORIAL_EXECUTION_AUTHORITY_SELECTION_COMPLETE"
R11_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R11_MINIMAL_SEED_DIVERGENCE_CAUSE_ISOLATION_AND_REPAIR_SELECTION_COMPLETE"
R8_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R8_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
R4A_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4A_F1_EXECUTION_AUTHORITY_COMPLETION"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R12 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection_20260826_145345+09:00"
R11 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection_20260826_123135+09:00"
R8 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r8_e1_eligibility_validation_20260825_190658+09:00"
R4A = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4a_f1_execution_authority_completion_20260823_132257+0900"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
R4 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4_v2_actor_fresh_training_design_20260823_122459+0900"
R7 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility_20260825_124641+09:00"

SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r13_s3_four_cell_execution.py",
    "05_training/test_h4m_ae_ls3_bt8_r13_s3_four_cell_execution.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}


class R13Error(RuntimeError):
    """A contract failure stops the bounded executor without a workaround."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R13Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path, code: str = AUTH_BLOCK) -> Any:
    require(path.is_file(), code, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256", {})
    require(isinstance(expected, Mapping), AUTH_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in expected.items()
                  if not (root / str(name)).is_file() or sha256(root / str(name)) != str(digest)]
    return {"manifest_sha256": sha256(root / "manifest.json"), "declared_file_count": len(expected),
            "mismatches": mismatches, "all_match": not mismatches}


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R12_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r12": git(["merge-base", R12_SOURCE, "HEAD"]) == R12_SOURCE,
        "changed_files_since_r12": changed,
        "source_only_local_commit": set(changed) == SOURCE_FILES,
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r13_s3_four_cell_execution_{stamp}"


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    groups = []
    for group in state["param_groups"]:
        groups.append({key: value for key, value in group.items() if key != "params"} | {"parameter_count": len(group["params"])})
    return {"state_item_count": len(state["state"]), "parameter_groups": groups,
            "fresh_empty": len(state["state"]) == 0}


def rng_digest() -> str:
    return hashlib.sha256(torch.get_rng_state().detach().cpu().numpy().tobytes()).hexdigest()


def initial_counters() -> dict[str, int]:
    return {
        "training": 0,
        "mps_training": 0,
        "causal_rollout": 0,
        "simulator_step": 0,
        "authoritative_candidate_generation": 0,
        "candidate_regeneration_after_selection": 0,
        "candidate_regeneration_during_ppo": 0,
        "local_search_rerun_during_ppo": 0,
        "zero_loss_reevaluation_during_ppo": 0,
        "actor_optimizer_step": 0,
        "critic_optimizer_step": 0,
        "raw_optimizer_step": 0,
        "unauthorized_optimizer_step": 0,
        "checkpoint_write": 0,
        "checkpoint_promotion": 0,
        "review_optimizer_rows": 0,
        "review_batch_inclusion": 0,
        "review_regeneration": 0,
        "future_leakage": 0,
        "test6_access": 0,
        "github_push": 0,
        "nan_or_inf": 0,
        "seed_cell_mixing": 0,
        "cross_trajectory_contamination": 0,
        "candidate_identity_mismatch": 0,
        "candidate_plan_execution_collapse": 0,
        "serve_fallback": 0,
        "zero_loss_violation": 0,
        "illegal_or_masked_selection": 0,
        "source_state_mutation_during_shadow_evaluation": 0,
    }


@dataclass
class CellBudget:
    """Hard budget guard checked immediately before a bounded mutation."""

    cell_id: str
    allowed_actor_steps: int = 0
    decisions: int = 0
    trajectories: int = 0
    transitions: int = 0
    ppo_cycles: int = 0
    actor_steps: int = 0
    critic_steps: int = 0
    decision_ids: set[str] = field(default_factory=set)
    trajectory_ids: set[str] = field(default_factory=set)

    def decision(self, decision_id: str) -> None:
        require(self.decisions < 24 and decision_id not in self.decision_ids,
                INTEGRITY_BLOCK, f"decision_overrun_or_duplicate={self.cell_id}:{decision_id}")
        self.decisions += 1
        self.decision_ids.add(decision_id)

    def trajectory(self, trajectory_id: str) -> None:
        require(self.trajectories < 6 and trajectory_id not in self.trajectory_ids,
                INTEGRITY_BLOCK, f"trajectory_overrun_or_duplicate={self.cell_id}:{trajectory_id}")
        self.trajectories += 1
        self.trajectory_ids.add(trajectory_id)

    def transition(self) -> None:
        require(self.transitions < 48, INTEGRITY_BLOCK, f"transition_overrun={self.cell_id}")
        self.transitions += 1

    def cycle(self) -> None:
        require(self.ppo_cycles < 3, INTEGRITY_BLOCK, f"ppo_cycle_overrun={self.cell_id}")
        self.ppo_cycles += 1

    def actor_step(self) -> None:
        require(self.actor_steps < self.allowed_actor_steps, INTEGRITY_BLOCK, f"actor_step_overrun={self.cell_id}")
        self.actor_steps += 1

    def critic_step(self) -> None:
        require(self.critic_steps < 3, INTEGRITY_BLOCK, f"critic_step_overrun={self.cell_id}")
        self.critic_steps += 1

    def finalize(self) -> None:
        require((self.decisions, self.trajectories, self.transitions, self.ppo_cycles, self.critic_steps) == (24, 6, 48, 3, 3),
                INTEGRITY_BLOCK, f"cell_budget_mismatch={self.cell_id}")
        require(self.actor_steps == self.allowed_actor_steps, INTEGRITY_BLOCK, f"actor_step_count={self.cell_id}")


def mps_preflight(*, H: Any, JL: Any, MC: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "required_device": "mps:0",
        "cpu_fallback": 0,
        "capability_grant": 0,
        "authoritative_candidate_generation": 0,
        "causal_rollout": 0,
        "optimizer_step": 0,
        "checkpoint_write": 0,
    }
    if not evidence["mps_built"] or not evidence["mps_available"]:
        return evidence | {"passed": False, "failure_reason": "MPS_BUILT_OR_AVAILABLE_FALSE"}
    try:
        device = torch.device("mps:0")
        adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
        torch.manual_seed(20260824)
        actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
            global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim).to(device)
        torch.manual_seed(20260826)
        critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim,
                                           safe_summary_dim=1 + 2 * cdim).to(device)
        inputs = {
            "global_feats": torch.zeros((1, 8), dtype=torch.float32, device=device),
            "demand_feats": torch.zeros((1, 6), dtype=torch.float32, device=device),
            "agent_feats": torch.zeros((1, 8, adim), dtype=torch.float32, device=device),
            "agent_mask": torch.ones((1, 8), dtype=torch.bool, device=device),
            "candidate_feats": torch.zeros((1, 8, cdim), dtype=torch.float32, device=device),
            "pair_agent_index": torch.arange(8, dtype=torch.long, device=device).reshape(1, 8),
            "safe_mask": torch.ones((1, 8), dtype=torch.bool, device=device),
        }
        with torch.no_grad():
            logits, no_assign = actor(**inputs)
            value = critic(global_feats=inputs["global_feats"], demand_feats=inputs["demand_feats"],
                           agent_feats=inputs["agent_feats"], agent_mask=inputs["agent_mask"],
                           safe_summary=JL.safe_set_summary(inputs["candidate_feats"], inputs["safe_mask"]))
        torch.mps.synchronize()
        finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item()
                      and torch.isfinite(value).all().item())
        return evidence | {
            "passed": finite, "selected_device": str(logits.device), "finite_probe": finite,
            "scorer_input_dimension": int(actor.scorer[0].in_features),
            "scorer_first_weight_shape": list(actor.scorer[0].weight.shape),
            "shape_contract": {"agents": 8, "candidate_pairs": 8, "candidate_feature_dim": cdim},
            "mps_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    except Exception as exc:  # noqa: BLE001
        return evidence | {"passed": False, "failure_reason": f"{type(exc).__name__}:{exc}"}


def _require_gate(root: Path, expected_gate: str, expected_source: str, label: str) -> dict[str, Any]:
    gate = load_json(root / "gate_decision.json")
    require(gate.get("gate") == expected_gate and gate.get("source_commit") == expected_source,
            AUTH_BLOCK, f"gate={label}")
    return gate


def load_authority(*, R12MOD: Any, E1: Any) -> dict[str, Any]:
    """Bind every R13 execution input before a capability can be granted."""
    source = source_provenance()
    manifests = {"r12": manifest_audit(R12), "r11": manifest_audit(R11), "r8": manifest_audit(R8),
                 "r4a": manifest_audit(R4A), "f1": manifest_audit(F1), "r4": manifest_audit(R4)}
    require(all(item["all_match"] for item in manifests.values()), AUTH_BLOCK, "authoritative_manifest_mismatch")
    _require_gate(R12, R12_GATE, R12_SOURCE, "r12")
    _require_gate(R11, R11_GATE, R11_SOURCE, "r11")
    _require_gate(R8, R8_GATE, E1_SOURCE, "r8")
    _require_gate(R4A, R4A_GATE, R4A_SOURCE, "r4a")
    _require_gate(F1, F1_GATE, F1_SOURCE, "f1")
    r12_cells = load_json(R12 / "bt8r12_s3_cell_contract.json")
    r12_mode = load_json(R12 / "bt8r12_on_policy_mode_selection.json")
    r12_budget = load_json(R12 / "bt8r12_cell_update_budget.json")
    r12_review = load_json(R12 / "bt8r12_review_support_contract.json")
    r12_authority = load_json(R12 / "bt8r12_execution_authority.json")
    require(r12_cells.get("s3_contract_sha256") == S3_CONTRACT_SHA256 and r12_mode.get("selected_mode") == "M2"
            and r12_mode.get("M2", {}).get("selected") is True and r12_authority.get("selected_mode") == "M2",
            AUTH_BLOCK, "r12_m2_authority")
    cells = r12_cells.get("cells")
    require(isinstance(cells, list) and [row.get("cell_id") for row in cells] == ["AC-R1", "AC-R2", "BD-R1", "BD-R2"],
            AUTH_BLOCK, "r12_cell_order")
    expected_seeds = {
        "AC-R1": (20260822, 20260824, 20260826), "AC-R2": (20260823, 20260824, 20260826),
        "BD-R1": (20260822, 20260825, 20260827), "BD-R2": (20260823, 20260825, 20260827),
    }
    for cell in cells:
        actual = (int(cell["environment_seed"]), int(cell["actor_seed"]), int(cell["critic_seed"]))
        require(actual == expected_seeds[str(cell["cell_id"])], AUTH_BLOCK, f"cell_seed={cell['cell_id']}")
    aggregate = r12_budget.get("aggregate_authorized_maximum", {})
    require(aggregate == {"train_windows_per_cell": 6, "decisions": 96, "trajectories": 24,
                          "causal_transitions": 192, "actor_optimizer_steps_maximum": 12,
                          "critic_optimizer_steps_exact": 12, "raw_optimizer_step_calls_maximum": 24,
                          "supplemental_steps": 0}, AUTH_BLOCK, "r12_budget")
    require(r12_review.get("collection_digest") == "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"
            and r12_review.get("unique_snapshot_count") == 6
            and r12_review.get("same_environment_ac_bd_identical_inputs") is True,
            AUTH_BLOCK, "r12_review_contract")
    s3 = load_json(R11 / "bt8r11_selected_seed_repair_contract.json")
    without_sha = {key: value for key, value in s3.items() if key != "sha256"}
    require(s3.get("sha256") == S3_CONTRACT_SHA256 and canonical_sha256(without_sha) == S3_CONTRACT_SHA256
            and s3.get("selected_option") == "S3", AUTH_BLOCK, "s3_contract")
    e1_runtime = load_json(R8 / "bt8r8_e1_runtime_contract.json")
    e1_contract = load_json(R7 / "bt8r7_selected_minimal_contract.json")
    E1.bind_e1_contract(e1_contract)
    require(e1_runtime.get("contract_sha256") == E1_CONTRACT_SHA256, AUTH_BLOCK, "e1_contract")
    bridge = load_json(R4A / "bt8r4a_candidate_plan_bridge_contract.json")
    bridge_audit = load_json(R4A / "bt8r4a_candidate_plan_execution_audit.json")
    credit_audit = load_json(R4A / "bt8r4a_credit_identity_audit.json")
    require(bridge.get("candidate_regeneration_after_selection") is False and bridge.get("serve_fallback") is False
            and bridge.get("source_state_mutation") is False and bridge_audit.get("identity_chain_all") is True
            and bridge_audit.get("candidate_a_b_distinct_applied_state") is True and credit_audit.get("verified") is True,
            AUTH_BLOCK, "candidate_plan_bridge")
    envelope = load_json(R4 / "bt8r4_selected_training_envelope.json")
    train_windows, review_windows = list(envelope.get("train_windows", [])), list(envelope.get("review_windows", []))
    train_ids, review_ids = [str(row["window_id"]) for row in train_windows], [str(row["window_id"]) for row in review_windows]
    require(len(train_windows) == 6 and len(review_windows) == 3 and len(set(train_ids)) == 6
            and len(set(review_ids)) == 3 and set(train_ids).isdisjoint(review_ids)
            and train_ids == list(r12_cells.get("train_window_ids", [])) and review_ids == list(r12_cells.get("review_window_ids", [])),
            AUTH_BLOCK, "frozen_train_review_split")
    review_collection = load_json(F1 / "bt8f1_review_snapshot_collection.json")
    stripped_review = dict(review_collection); review_digest = stripped_review.pop("collection_digest", None)
    require(review_digest == "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"
            and canonical_sha256(stripped_review) == review_digest and review_collection.get("snapshot_count") == 6,
            AUTH_BLOCK, "historical_review_collection")
    training_collection = load_json(F1 / "bt8f1_training_snapshot_collection.json")
    require(training_collection.get("collection_digest") == "587e45422e0d5cbfcfc0f7279eae34178e00414d21229951ce6683c2aa4785d8",
            AUTH_BLOCK, "historical_training_collection")
    initial_manifest = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
    lineage = load_json(F1 / "bt8f1_initialization_lineage.json")
    lineage_by_id = {str(row["replicate_id"]): row for row in lineage.get("replicates", [])}
    require(set(initial_manifest) == {"F1_R1", "F1_R2"} and set(lineage_by_id) == {"F1_R1", "F1_R2"}
            and lineage.get("fresh_v2_actor") is True and lineage.get("fresh_critic") is True
            and lineage.get("v1_transfer_count") == 0 and lineage.get("prior_checkpoint_or_optimizer_reuse_count") == 0,
            AUTH_BLOCK, "initialization_lineage")
    historical_sha: dict[str, str] = {}
    for replicate_id in ("F1_R1", "F1_R2"):
        entry = initial_manifest[replicate_id]
        path = F1 / "initial_checkpoints" / str(entry["path"])
        historical_sha[replicate_id] = sha256(path)
        require(historical_sha[replicate_id] == entry.get("sha256"), AUTH_BLOCK, f"historical_checkpoint={replicate_id}")
    r12_frozen = load_json(R12 / "frozen_hash_before_after.json")
    frozen_before = R12MOD.frozen_hashes()
    require(frozen_before == r12_frozen.get("after") and r12_frozen.get("all_unchanged") is True,
            AUTH_BLOCK, "frozen_hash_before")
    require(source["source_lineage_descends_from_r12"] and source["source_only_local_commit"], AUTH_BLOCK, "source_scope")
    return {
        "source": source, "manifests": manifests, "cells": cells, "train_windows": train_windows,
        "review_windows": review_windows, "review_collection": review_collection,
        "historical_training_collection_digest": training_collection["collection_digest"],
        "initial_manifest": initial_manifest, "lineage_by_id": lineage_by_id,
        "historical_initial_checkpoint_sha256": historical_sha, "frozen_before": frozen_before,
        "r12_mode": r12_mode, "r12_budget": r12_budget, "r12_review": r12_review,
    }


def load_cell_models(*, cell: Mapping[str, Any], authority: Mapping[str, Any], config: Mapping[str, Any],
                     H: Any, JL: Any, MC: Any, device: torch.device) -> dict[str, Any]:
    """Clone a bound AC/BD source checkpoint into a cell-local model pair."""
    cell_id, init_rep = str(cell["cell_id"]), str(cell["initialization_replicate"])
    manifest = authority["initial_manifest"][init_rep]
    historical_path = F1 / "initial_checkpoints" / str(manifest["path"])
    payload = torch.load(historical_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping) and set(payload).issuperset({"actor", "critic", "meta"})
            and not any("optimizer" in str(key).lower() for key in payload), AUTH_BLOCK, f"checkpoint_schema={cell_id}")
    torch.manual_seed(int(cell["actor_seed"])); actor_rng = rng_digest()
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]), candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    torch.manual_seed(int(cell["critic_seed"])); critic_rng = rng_digest()
    critic = JL.JointAssignmentCritic(global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
                                      agent_dim=int(config["agent_dim"]), safe_summary_dim=1 + 2 * int(config["candidate_dim"])).to(device)
    critic.load_state_dict(payload["critic"], strict=True)
    actor_opt, critic_opt = torch.optim.Adam(actor.parameters(), lr=1e-4), torch.optim.Adam(critic.parameters(), lr=1e-4)
    actor_initial, critic_initial = module_digest(actor), module_digest(critic)
    require(actor_initial == str(cell["actor_initial_state_digest"]) and critic_initial == str(cell["critic_initial_state_digest"]),
            AUTH_BLOCK, f"initial_state_digest={cell_id}")
    require(optimizer_summary(actor_opt)["fresh_empty"] and optimizer_summary(critic_opt)["fresh_empty"],
            AUTH_BLOCK, f"fresh_optimizer={cell_id}")
    return {
        **dict(cell), "actor": actor, "critic": critic, "actor_opt": actor_opt, "critic_opt": critic_opt,
        "actor_initial_digest": actor_initial, "critic_initial_digest": critic_initial,
        "actor_rng_digest": actor_rng, "critic_rng_digest": critic_rng,
        "actor_optimizer_initial": optimizer_summary(actor_opt), "critic_optimizer_initial": optimizer_summary(critic_opt),
        "behavior_actor_checkpoint_sha256": str(cell["actor_initial_checkpoint_sha256"]),
        "historical_initial_checkpoint_path": historical_path,
    }


def assert_model_independence(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    actor_ids, critic_ids, actor_opt_ids, critic_opt_ids = [], [], [], []
    parameter_ids: list[int] = []
    for item in cells.values():
        actor_ids.append(id(item["actor"])); critic_ids.append(id(item["critic"]))
        actor_opt_ids.append(id(item["actor_opt"])); critic_opt_ids.append(id(item["critic_opt"]))
        parameter_ids.extend(id(parameter) for parameter in item["actor"].parameters())
        parameter_ids.extend(id(parameter) for parameter in item["critic"].parameters())
    independent = len(set(actor_ids)) == len(actor_ids) == 4 and len(set(critic_ids)) == len(critic_ids) == 4 \
        and len(set(actor_opt_ids)) == len(actor_opt_ids) == 4 and len(set(critic_opt_ids)) == len(critic_opt_ids) == 4 \
        and len(set(parameter_ids)) == len(parameter_ids)
    require(independent, AUTH_BLOCK, "cell_model_or_optimizer_aliasing")
    return {"actor_object_alias_count": len(actor_ids) - len(set(actor_ids)),
            "critic_object_alias_count": len(critic_ids) - len(set(critic_ids)),
            "optimizer_object_alias_count": (len(actor_opt_ids) - len(set(actor_opt_ids))) + (len(critic_opt_ids) - len(set(critic_opt_ids))),
            "parameter_object_alias_count": len(parameter_ids) - len(set(parameter_ids)),
            "independent": True}


def _finite_model_outputs(*, logits: torch.Tensor, no_assign: torch.Tensor, value: torch.Tensor,
                          safe_mask: torch.Tensor) -> bool:
    return bool(torch.isfinite(logits[safe_mask]).all().item() and torch.isfinite(no_assign).all().item()
                and torch.isfinite(value).all().item())


def _gradient_abs_sum(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().abs().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).sum().detach().cpu()) if values else 0.0


def _gradient_norm(module: torch.nn.Module) -> float:
    values = [(parameter.grad.detach() ** 2).sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).detach().cpu()) if values else 0.0


def _clip_summary(*, ratio: torch.Tensor, advantage: torch.Tensor, active: torch.Tensor, epsilon: float) -> dict[str, Any]:
    rows = []
    for value, adv, included in zip(ratio.detach().cpu().tolist(), advantage.detach().cpu().tolist(), active.detach().cpu().tolist()):
        clipped = (adv > 0.0 and value > 1.0 + epsilon) or (adv < 0.0 and value < 1.0 - epsilon)
        rows.append({"ratio": float(value), "advantage": float(adv), "actor_active": bool(included), "clipped": bool(clipped)})
    active_rows = [row for row in rows if row["actor_active"]]
    return {"ratio_min": min(row["ratio"] for row in rows), "ratio_mean": sum(row["ratio"] for row in rows) / len(rows),
            "ratio_max": max(row["ratio"] for row in rows), "actor_active_rows": len(active_rows),
            "actor_active_clipped_rows": sum(row["clipped"] for row in active_rows), "rows": rows}


def _review_entries_by_environment(collection: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    groups = {"F1_R1": [], "F1_R2": []}
    for entry in collection.get("entries", []):
        seed = int(entry["seed"])
        env = "F1_R1" if seed == 20260822 else "F1_R2" if seed == 20260823 else None
        require(env is not None, AUTH_BLOCK, f"review_seed={seed}")
        groups[env].append(entry)
    for env, entries in groups.items():
        entries.sort(key=lambda row: int(row["decision_index"]))
        require(len(entries) == 3 and len({str(row["snapshot_digest"]) for row in entries}) == 3, AUTH_BLOCK, f"review_entries={env}")
    return groups


def review_replay(*, models: Mapping[str, Mapping[str, Any]], review_entries: Mapping[str, Sequence[Mapping[str, Any]]],
                  FPS: Any, F1MOD: Any, H: Any, TIE: Any, device: torch.device, kind: str,
                  checkpoint_by_cell: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pure replay of persisted F1 review snapshots; no simulator-facing calls."""
    rows: list[dict[str, Any]] = []
    for cell_id, model in models.items():
        entries = review_entries[str(model["environment_replicate"])]
        for entry in entries:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / str(entry["relative_path"]))
            replay = F1MOD.replay(model["actor"], payload, H, TIE, device)
            require(replay["finite"] and replay["snapshot_digest"] == entry["snapshot_digest"], INTEGRITY_BLOCK, f"review_replay={cell_id}")
            rows.append({"cell_id": cell_id, "environment_replicate": model["environment_replicate"],
                         "review_snapshot_digest": str(entry["snapshot_digest"]), "window_id": str(entry["window_id"]),
                         "checkpoint_sha256": checkpoint_by_cell[cell_id]["sha256"], "strict_checkpoint_load": bool(checkpoint_by_cell[cell_id].get("strict_load", False)),
                         "replay_kind": kind, **replay})
    require(len(rows) == 12 and len({(row["cell_id"], row["review_snapshot_digest"]) for row in rows}) == 12,
            INTEGRITY_BLOCK, f"review_replay_count={kind}")
    return rows


def _write_checkpoint(*, path: Path, model: Mapping[str, Any], kind: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": kind, "cell_id": str(model["cell_id"]), "environment_seed": int(model["environment_seed"]),
        "test_only": True, "bounded": True, "non_promotable": True, "winner": False, "best_model": False,
        "promotion": False, "performance_claim_allowed": False, "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False, **dict(extra),
    }
    torch.save({"actor": {name: tensor.detach().cpu() for name, tensor in model["actor"].state_dict().items()},
                "critic": {name: tensor.detach().cpu() for name, tensor in model["critic"].state_dict().items()}, "meta": meta}, path)
    return {"path": path.relative_to(path.parent.parent).as_posix(), "sha256": sha256(path), **meta}


def _strict_actor_from_checkpoint(*, path: Path, expected_actor_digest: str, config: Mapping[str, Any], H: Any,
                                  JL: Any, MC: Any, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=adim,
        candidate_dim=cdim, hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
    critic = JL.JointAssignmentCritic(global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=adim,
                                      safe_summary_dim=1 + 2 * cdim).to(device)
    actor.load_state_dict(payload["actor"], strict=True); critic.load_state_dict(payload["critic"], strict=True)
    require(module_digest(actor) == expected_actor_digest, INTEGRITY_BLOCK, f"strict_actor_load={path.name}")
    return actor.eval()


def _check_historical_initials(authority: Mapping[str, Any]) -> dict[str, str]:
    return {rep: sha256(F1 / "initial_checkpoints" / str(entry["path"]))
            for rep, entry in authority["initial_manifest"].items()}


def _normalization_summary(raw: Sequence[float], normalized: Sequence[float]) -> dict[str, Any]:
    mean = sum(float(value) for value in raw) / len(raw)
    variance = sum((float(value) - mean) ** 2 for value in raw) / len(raw)
    return {"scope": "cell-local 24 train rows before E1 mask", "row_count": len(raw),
            "raw_mean": mean, "raw_std": max(variance ** 0.5, 1e-8),
            "normalized_mean": sum(float(value) for value in normalized) / len(normalized),
            "formula": "(raw - cell-local mean) / max(cell-local std, 1e-8)"}


def _identity_ids(plan: Any) -> tuple[str, str, str]:
    return str(plan.selected_candidate_id), str(plan.applied_candidate_id), str(plan.credited_candidate_id)


def rollout_cell(*, model: Mapping[str, Any], authority: Mapping[str, Any], factory: Any, agent_slots: Mapping[str, int],
                 config: Mapping[str, Any], features: Mapping[str, Any], frozen: Mapping[str, str], root: Path,
                 snapshot_entries: list[dict[str, Any]], F1MOD: Any, H: Any, JL: Any, CC: Any, CB: Any, PE: Any,
                 MC: Any, BT1: Any, BT6: Any, R3: Any, compute_reward_v2: Any, FPS: Any, device: torch.device,
                 counters: dict[str, int]) -> dict[str, Any]:
    """Run the exact six-window, four-decision fresh causal M2 path for one cell."""
    cell_id = str(model["cell_id"])
    budget = CellBudget(cell_id=cell_id)
    rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    train_root = root / "bt8r13_training_snapshots"
    for window_index, window in enumerate(authority["train_windows"]):
        trajectory_id = f"{cell_id}:{window['window_id']}"
        budget.trajectory(trajectory_id)
        adapter = R3.imp("bridge", R3.BRIDGE).PV8CausalKpiAdapter(
            **R3.authoritative_adapter_inputs({"window_id": window["window_id"]}, num_agents=8, seed=int(model["environment_seed"])))
        trajectory: list[dict[str, Any]] = []
        for decision_index in range(4):
            decision_id = f"BT8_R13:{cell_id}:{window_index}:{decision_index}"
            budget.decision(decision_id)
            support = factory.build(source_group=window["source_group"], decision_group=decision_id)
            counters["authoritative_candidate_generation"] += 1
            packed, ready = F1MOD.model_pack(support=support, H=H, BT6=BT6, cdim=int(config["candidate_dim"]), device=device)
            captured = F1MOD.snapshot(root=train_root, store=snapshot_entries, FPS=FPS, decision_id=decision_id,
                                      window=window, seed=int(model["environment_seed"]), index=decision_index,
                                      packed=packed, model_packed=ready, support=support, MC=MC, config=config,
                                      features=features, frozen=frozen, source_commit=str(authority["source"]["source_commit"]), device=device)
            # Read immediately after lossless serialization: metadata-only fallback is not accepted.
            loaded = FPS.load_snapshot(captured["directory"])
            require(loaded["snapshot_digest"] == captured["digest"] and loaded["metadata"]["decision_id"] == decision_id,
                    INTEGRITY_BLOCK, f"snapshot_roundtrip={decision_id}")
            model["actor"].eval(); model["critic"].eval()
            with torch.no_grad():
                raw, no_assign = model["actor"](global_feats=ready["global_feats"], demand_feats=ready["demand_feats"],
                                                   agent_feats=ready["agent_feats"], agent_mask=ready["agent_mask"],
                                                   candidate_feats=ready["candidate_feats"], pair_agent_index=ready["pair_agent_index"],
                                                   safe_mask=ready["safe_mask"])
                pair_count = len(packed["pair_keys"])
                logits, mask = raw[:, :pair_count], ready["safe_mask"][:, :pair_count]
                # H.select is the frozen training-time selection path.  T1 is bound only to review replay.
                output = H.select(decision_id, packed["pair_keys"], logits, no_assign, mask)
                old_log = float(JL.masked_log_probs(logits, no_assign, mask)[0, output.selected_index])
                value = float(model["critic"](global_feats=ready["global_feats"], demand_feats=ready["demand_feats"],
                                                agent_feats=ready["agent_feats"], agent_mask=ready["agent_mask"],
                                                safe_summary=JL.safe_set_summary(ready["candidate_feats"], ready["safe_mask"]))[0])
            require(math.isfinite(old_log) and math.isfinite(value), INTEGRITY_BLOCK, f"nonfinite_rollout={decision_id}")
            if output.selected_is_no_assign:
                selected_type = "NO_ASSIGN"
            else:
                selected_type = "CANDIDATE"
                support["snapshot"].assert_selectable(agent_id=output.selected_pair[0], candidate_id=output.selected_pair[1])
            plan = F1MOD.selected_plan(snapshot_value=support["snapshot"], output=output, decision_id=decision_id, CB=CB, PE=PE)
            selected_id, applied_id, credited_id = _identity_ids(plan)
            identity_ok = selected_id == applied_id == credited_id
            counters["candidate_identity_mismatch"] += int(not identity_ok)
            counters["serve_fallback"] += int(bool(plan.serve_fallback_used))
            require(identity_ok and not bool(plan.serve_fallback_used), INTEGRITY_BLOCK, f"plan_identity={decision_id}")
            actions = {agent: 0 for agent in range(8)}
            if not output.selected_is_no_assign:
                require(output.selected_pair[0] in agent_slots, INTEGRITY_BLOCK, f"agent_slot={decision_id}")
                actions[int(agent_slots[output.selected_pair[0]])] = BT1.SERVE
            rewards, operational = [], []
            for operation in range(2):
                budget.transition()
                result = adapter.step(actions, legal_mask={agent: [True, True, True] for agent in actions}, target_ids=actions,
                                      provenance={"arm_id": "BT8_R13", "policy_source": "fresh_v2_joint_assignment_actor",
                                                  "candidate_plan_transition_id": plan.transition_id,
                                                  "candidate_plan_digest": plan.candidate_plan_digest,
                                                  "applied_plan_digest": plan.applied_plan_digest})
                agent_rewards = []
                for event in result["events"]:
                    metric = BT1.agent_reward_metrics(
                        transition_id=f"{decision_id}:{operation}:{event['agent_id']}", agent_id=event["agent_id"],
                        action_id=event["action_id"], boarded=int(event["passenger_served"]), served=int(event["passenger_served"]),
                        wait_rows=[], decision_ts=int(event["event_ts"]), intervened=False)
                    reward = float(compute_reward_v2(metric)["reward_total"])
                    require(math.isfinite(reward), INTEGRITY_BLOCK, f"reward_v2_nonfinite={decision_id}")
                    agent_rewards.append(reward)
                rewards.append(CC.team_reward(agent_rewards, [True] * len(agent_rewards)))
                operational.append({"step": operation, "team_reward": rewards[-1], "causal_state_digest": adapter.state_identity()})
                counters["causal_rollout"] += 1; counters["simulator_step"] += 1
            terminal = decision_index == 3
            transition = CC.AssignmentTransition(
                assignment_step_id=decision_id, decision_group_id=decision_id, episode_id=cell_id, window_id=window["window_id"],
                decision_ts=decision_index * 2, next_assignment_ts=None if terminal else (decision_index + 1) * 2,
                delta_operational_steps=2, pre_state_digest=plan.events[0]["source_state_digest"],
                next_state_digest=plan.next_state.state_digest, safe_pair_ids=list(packed["pair_keys"]),
                safe_pair_mask=[True] * len(packed["pair_keys"]), no_assign_index=len(packed["pair_keys"]),
                selected_agent_id=None if output.selected_is_no_assign else output.selected_pair[0],
                selected_candidate_id=None if output.selected_is_no_assign else output.selected_pair[1],
                selected_is_no_assign=bool(output.selected_is_no_assign), valid_action_count=len(packed["pair_keys"]) + 1,
                forced_action=len(packed["pair_keys"]) == 0, old_log_prob=old_log, old_value=value,
                team_reward_sequence=rewards, assignment_discounted_reward=CC.assignment_return(rewards),
                terminated=terminal, truncated=False, policy_version=H.CANDIDATE_SENSITIVE_HEAD_VERSION,
                credit_contract_version=CC.CONTRACT_VERSION, seed=int(model["environment_seed"]),
                provenance={"trajectory_id": trajectory_id, "cell_id": cell_id,
                            "candidate_support_digest": support["snapshot"].snapshot_digest,
                            "candidate_plan_digest": plan.candidate_plan_digest, "applied_plan_digest": plan.applied_plan_digest,
                            "selected_candidate_id": selected_id, "applied_candidate_id": applied_id,
                            "credited_candidate_id": credited_id, "candidate_regenerated_during_ppo": False,
                            "operational": operational, "no_assign": plan.no_assign})
            trajectory.append({"t": transition, "packed": packed, "snapshot": support["snapshot"], "plan": plan,
                               "snapshot_digest": captured["digest"], "action_type": selected_type,
                               "zero_loss_selective": bool(support["snapshot"].rejected),
                               "multi_candidate": len(packed["pair_keys"]) >= 2})
        require(len(trajectory) == 4, INTEGRITY_BLOCK, f"trajectory_length={trajectory_id}")
        rows.extend(trajectory)
        windows.append({"window_id": str(window["window_id"]), "trajectory_id": trajectory_id, "trajectory_length": 4,
                        "causal_transitions": 8, "source_group": str(window["source_group"])})
    require(len(rows) == 24 and len(windows) == 6, INTEGRITY_BLOCK, f"rollout_scope={cell_id}")
    return {"rows": rows, "windows": windows, "budget": budget,
            "factory_source_mutations": int(factory.source_mutations), "factory_evaluations": int(factory.evaluations)}


def build_credit_and_batch(*, rollout: Mapping[str, Any], model: Mapping[str, Any], JL: Any, CC: Any, E1: Any,
                           R7MOD: Any, FC: Any, BT6: Any, device: torch.device, adim: int, cdim: int) -> dict[str, Any]:
    """Compute trajectory-local GAE, cell-local N0, then the immutable E1 mask."""
    rows = list(rollout["rows"])
    require(len(rows) == 24 and len({str(row["t"].assignment_step_id) for row in rows}) == 24,
            INTEGRITY_BLOCK, f"credit_rows={model['cell_id']}")
    values = [float(row["t"].old_value) for row in rows]
    next_values = [values[index + 1] if index + 1 < len(rows)
                   and rows[index + 1]["t"].window_id == row["t"].window_id else 0.0
                   for index, row in enumerate(rows)]
    transitions = [row["t"] for row in rows]
    gae = JL.compute_assignment_gae(transitions, values, next_values)
    normalized = FC.normalize_advantages_for_replicate(gae["assignment_advantage"])
    trace = [{"decision_id": row["t"].assignment_step_id,
              "trajectory_id": str(row["t"].provenance["trajectory_id"]),
              "nonterminal": not bool(row["t"].terminated),
              "reward": float(row["t"].assignment_discounted_reward)} for row in rows]
    fixed = R7MOD.fixed_gae(trace, {str(row["t"].assignment_step_id): float(row["t"].old_value) for row in rows},
                            gamma=CC.GAMMA, lam=CC.GAE_LAMBDA)
    fixed_by_id = {str(item["decision_id"]): item for item in fixed}
    require(len(fixed_by_id) == 24, INTEGRITY_BLOCK, f"fixed_gae_count={model['cell_id']}")
    e1_rows: list[dict[str, Any]] = []
    gae_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        transition, plan = row["t"], row["plan"]
        decision_id = str(transition.assignment_step_id)
        fixed_row = fixed_by_id[decision_id]
        selected, applied, credited = _identity_ids(plan)
        identity_valid = selected == applied == credited
        raw_gae = float(gae["assignment_advantage"][index])
        # The independently re-evaluated R7 decomposition makes reward ancestry auditable.
        require(math.isfinite(raw_gae) and math.isfinite(float(fixed_row["reward_gae_component"]))
                and math.isfinite(float(fixed_row["critic_bootstrap_gae_component"])),
                INTEGRITY_BLOCK, f"credit_nonfinite={decision_id}")
        eligibility_input = {
            "decision_id": decision_id, "replicate_id": str(model["cell_id"]), "data_role": "TRAIN",
            "identity_chain_valid": identity_valid, "selected_action_type": row["action_type"],
            "selected_candidate_id": selected, "applied_candidate_id": applied, "credited_candidate_id": credited,
            "reward": float(transition.assignment_discounted_reward),
            "reward_gae_component": float(fixed_row["reward_gae_component"]), "raw_gae": raw_gae,
            "critic_eligible_e1": True,
        }
        category = E1.category_from_row(eligibility_input)
        eligibility_input["category"] = category
        eligibility_input["actor_eligible_e1"] = category in E1.ACTOR_ELIGIBLE_CATEGORIES
        e1_rows.append(eligibility_input)
        gae_rows.append({
            "decision_id": decision_id, "trajectory_id": str(transition.provenance["trajectory_id"]),
            "window_id": str(transition.window_id), "reward": float(transition.assignment_discounted_reward),
            "Team Reward": float(transition.assignment_discounted_reward), "critic_value": float(transition.old_value),
            "next_value": float(next_values[index]), "critic_target": float(gae["assignment_return"][index]),
            "td_residual": float(gae["assignment_td_residual"][index]), "raw_gae": raw_gae,
            "normalized_advantage": float(normalized[index]),
            "gae_recursive_term": raw_gae - float(gae["assignment_td_residual"][index]),
            "temporally_propagated": abs(raw_gae - float(gae["assignment_td_residual"][index])) > 1e-12,
            "nonterminal": not bool(transition.terminated),
            "reward_gae_component": float(fixed_row["reward_gae_component"]),
            "critic_bootstrap_gae_component": float(fixed_row["critic_bootstrap_gae_component"]),
            "raw_additivity_residual": float(fixed_row["raw_additivity_residual"]),
            "selected_action_type": row["action_type"],
        })
    mask = E1.build_e1_eligibility_mask(e1_rows, expected_replicate_id=str(model["cell_id"]))
    batch = BT6.batch_rollout(rows, device=device, adim=adim, cdim=cdim)
    batch["advantage"] = torch.tensor(normalized, dtype=torch.float32, device=device)
    batch["value_target"] = torch.tensor(gae["assignment_return"], dtype=torch.float32, device=device)
    batch["actor_eligibility_mask"] = torch.tensor(mask.actor_eligible, dtype=torch.bool, device=device)
    require(all(bool(torch.isfinite(batch[name]).all().item()) for name in
                ("global_feats", "demand_feats", "agent_feats", "candidate_feats", "old_log_prob", "advantage", "value_target")),
            INTEGRITY_BLOCK, f"batch_nonfinite={model['cell_id']}")
    active = batch["actor_eligibility_mask"] & ~batch["forced_action"]
    return {"rows": rows, "gae": gae, "gae_rows": gae_rows, "e1_rows": e1_rows, "mask": mask, "batch": batch,
            "active_actor_rows": int(active.sum().item()), "normalization": _normalization_summary(gae["assignment_advantage"], normalized),
            "fixed_gae_additivity_residual_max_abs": max(abs(float(item["raw_additivity_residual"])) for item in fixed),
            "fixed_gae_raw_delta_max_abs": max(abs(float(gae["assignment_advantage"][index]) - float(item["raw_gae"]))
                                                for index, item in enumerate(fixed))}


def train_cell(*, model: Mapping[str, Any], prepared: Mapping[str, Any], JL: Any, AUTH: Any,
               device: torch.device, counters: dict[str, int]) -> dict[str, Any]:
    """Exact 3x full-batch PPO with actor gradients gated by E1 only."""
    cell_id, batch = str(model["cell_id"]), prepared["batch"]
    active_count = int(prepared["active_actor_rows"])
    budget = prepared["rollout_budget"]
    budget.allowed_actor_steps = 3 if active_count > 0 else 0
    actor, critic = model["actor"], model["critic"]
    actor_opt, critic_opt = model["actor_opt"], model["critic_opt"]
    updates: list[dict[str, Any]] = []
    ratio_rows: list[list[float]] = [[] for _ in range(24)]
    for epoch in range(3):
        budget.cycle()
        for row in prepared["rows"]:
            row["snapshot"].replay_guard(support_digest=row["t"].provenance["candidate_support_digest"], regeneration_requested=False)
        actor.train(); critic.train()
        logits, no_assign = actor(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                                  agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                                  candidate_feats=batch["candidate_feats"], pair_agent_index=batch["pair_agent_index"],
                                  safe_mask=batch["safe_mask"])
        value = critic(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                       agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                       safe_summary=JL.safe_set_summary(batch["candidate_feats"], batch["safe_mask"]))
        require(_finite_model_outputs(logits=logits, no_assign=no_assign, value=value, safe_mask=batch["safe_mask"]),
                INTEGRITY_BLOCK, f"training_output_nonfinite={cell_id}")
        loss = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign,
                                      safe_mask=batch["safe_mask"], action_index=batch["action_index"],
                                      old_log_prob=batch["old_log_prob"], advantage=batch["advantage"], value_pred=value,
                                      value_target=batch["value_target"], forced_action=batch["forced_action"],
                                      actor_eligibility_mask=batch["actor_eligibility_mask"])
        require(bool(torch.isfinite(loss["ratio"]).all().item() and torch.isfinite(loss["actor_loss"]).item()
                     and torch.isfinite(loss["critic_loss"]).item()), INTEGRITY_BLOCK, f"loss_nonfinite={cell_id}")
        logits.retain_grad(); no_assign.retain_grad()
        active = batch["actor_eligibility_mask"] & ~batch["forced_action"]
        require(int(loss["actor_eligible_rows"]) == active_count and bool(loss["actor_update_skipped"]) == (active_count == 0),
                INTEGRITY_BLOCK, f"actor_skip_contract={cell_id}")
        if active_count:
            AUTH.require_capability(AUTH.TRAINING, site=f"{STAGE}:{cell_id}:actor_epoch_{epoch + 1}")
            actor_opt.zero_grad(set_to_none=True)
            loss["actor_loss"].backward()
            inactive = ~active
            inactive_grad = max(float(logits.grad[inactive].detach().abs().max().cpu()) if bool(inactive.any()) else 0.0,
                                float(no_assign.grad[inactive].detach().abs().max().cpu()) if bool(inactive.any()) else 0.0)
            eligible_grad = float(logits.grad[active].detach().abs().sum().cpu()) + float(no_assign.grad[active].detach().abs().sum().cpu())
            actor_grad_abs, actor_grad_norm = _gradient_abs_sum(actor), _gradient_norm(actor)
            require(inactive_grad == 0.0 and eligible_grad > 0.0 and actor_grad_abs > 0.0,
                    INTEGRITY_BLOCK, f"actor_gradient_isolation={cell_id}")
            budget.actor_step(); actor_opt.step()
            counters["actor_optimizer_step"] += 1; counters["raw_optimizer_step"] += 1
        else:
            grad_logits, grad_no_assign = torch.autograd.grad(loss["actor_loss"], (logits, no_assign), retain_graph=True)
            inactive_grad = max(float(grad_logits.detach().abs().max().cpu()), float(grad_no_assign.detach().abs().max().cpu()))
            eligible_grad, actor_grad_abs, actor_grad_norm = 0.0, 0.0, 0.0
            require(inactive_grad == 0.0 and all(parameter.grad is None for parameter in actor.parameters()),
                    INTEGRITY_BLOCK, f"zero_eligible_actor_gradient={cell_id}")
        AUTH.require_capability(AUTH.TRAINING, site=f"{STAGE}:{cell_id}:critic_epoch_{epoch + 1}")
        critic_opt.zero_grad(set_to_none=True)
        loss["critic_loss"].backward()
        critic_grad_abs, critic_grad_norm = _gradient_abs_sum(critic), _gradient_norm(critic)
        require(critic_grad_abs > 0.0 and critic_grad_norm > 0.0, INTEGRITY_BLOCK, f"critic_gradient_missing={cell_id}")
        budget.critic_step(); critic_opt.step()
        counters["critic_optimizer_step"] += 1; counters["raw_optimizer_step"] += 1
        torch.mps.synchronize()
        values = (actor_grad_abs, actor_grad_norm, critic_grad_abs, critic_grad_norm,
                  float(loss["actor_loss"].detach().cpu()), float(loss["critic_loss"].detach().cpu()))
        counters["nan_or_inf"] += int(not all(math.isfinite(value_) for value_ in values))
        for index, ratio in enumerate(loss["ratio"].detach().cpu().tolist()):
            ratio_rows[index].append(float(ratio))
        updates.append({
            "epoch": epoch + 1, "actor_step": active_count > 0, "critic_step": True,
            "actor_eligible_semantic_rows": int(prepared["mask"].actor_eligible_count),
            "actor_active_rows": active_count, "actor_ineligible_rows": int(prepared["mask"].actor_ineligible_count),
            "actor_update_skipped": bool(loss["actor_update_skipped"]),
            "actor_loss": float(loss["actor_loss"].detach().cpu()), "policy_loss": float(loss["policy_loss"].detach().cpu()),
            "entropy": float(loss["entropy"].detach().cpu()), "critic_loss": float(loss["critic_loss"].detach().cpu()),
            "actor_grad_abs_sum": actor_grad_abs, "actor_grad_norm": actor_grad_norm,
            "critic_grad_abs_sum": critic_grad_abs, "critic_grad_norm": critic_grad_norm,
            "ineligible_actor_logit_gradient_max_abs": inactive_grad,
            "eligible_actor_logit_gradient_abs_sum": eligible_grad,
            "ppo": _clip_summary(ratio=loss["ratio"], advantage=batch["advantage"], active=active,
                                 epsilon=float(JL.CC.PPO_CLIP_EPSILON)),
        })
    budget.finalize()
    actor_final, critic_final = module_digest(actor), module_digest(critic)
    actor_changed = actor_final != str(model["actor_initial_digest"])
    critic_changed = critic_final != str(model["critic_initial_digest"])
    require(critic_changed and (actor_changed if active_count else not actor_changed),
            INTEGRITY_BLOCK, f"parameter_change_contract={cell_id}")
    require(max(item["ineligible_actor_logit_gradient_max_abs"] for item in updates) == 0.0,
            INTEGRITY_BLOCK, f"ineligible_gradient_leakage={cell_id}")
    return {
        "updates": updates, "ratios": ratio_rows,
        "step_ledger": {"ppo_cycles": budget.ppo_cycles, "actor_optimizer_steps": budget.actor_steps,
                        "critic_optimizer_steps": budget.critic_steps,
                        "raw_optimizer_step_calls": budget.actor_steps + budget.critic_steps, "supplemental_steps": 0},
        "initial": {"actor_digest": model["actor_initial_digest"], "critic_digest": model["critic_initial_digest"],
                    "actor_optimizer": model["actor_optimizer_initial"], "critic_optimizer": model["critic_optimizer_initial"]},
        "final": {"actor_digest": actor_final, "critic_digest": critic_final,
                  "actor_optimizer": optimizer_summary(actor_opt), "critic_optimizer": optimizer_summary(critic_opt)},
        "parameter_change": {"actor_changed": actor_changed, "critic_changed": critic_changed},
        "gradient_isolation": {"max_ineligible_actor_logit_gradient_abs": max(item["ineligible_actor_logit_gradient_max_abs"] for item in updates),
                               "min_eligible_actor_logit_gradient_abs": min(item["eligible_actor_logit_gradient_abs_sum"] for item in updates),
                               "actor_explicit_skip": active_count == 0},
    }


def make_credit_audit_rows(*, model: Mapping[str, Any], prepared: Mapping[str, Any], training: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join identity, credit, reward ancestry, and last PPO ratio by decision ID."""
    rows: list[dict[str, Any]] = []
    for index, rollout_row in enumerate(prepared["rows"]):
        transition, plan = rollout_row["t"], rollout_row["plan"]
        gae_row, e1_row = prepared["gae_rows"][index], prepared["e1_rows"][index]
        selected, applied, credited = _identity_ids(plan)
        ratios = training["ratios"][index]
        rows.append({
            "cell_id": str(model["cell_id"]), "replicate_id": str(model["cell_id"]),
            "environment_seed": int(model["environment_seed"]), "actor_seed": int(model["actor_seed"]),
            "critic_seed": int(model["critic_seed"]), "decision_id": str(transition.assignment_step_id),
            "trajectory_id": str(transition.provenance["trajectory_id"]), "window_id": str(transition.window_id),
            "agent_id": str(plan.events[0]["agent_id"]), "candidate_id": credited,
            "selected_candidate_id": selected, "applied_candidate_id": applied, "credited_candidate_id": credited,
            "selected_action_type": str(rollout_row["action_type"]), "candidate_plan_digest": str(plan.candidate_plan_digest),
            "applied_plan_digest": str(plan.applied_plan_digest), "source_state_digest": str(plan.events[0]["source_state_digest"]),
            "next_state_digest": str(plan.next_state.state_digest), "transition_id": str(plan.transition_id),
            "snapshot_digest": str(rollout_row["snapshot_digest"]),
            "candidate_support_digest": str(transition.provenance["candidate_support_digest"]),
            "Team Reward": float(gae_row["Team Reward"]), "reward": float(gae_row["reward"]),
            "critic_value": float(gae_row["critic_value"]), "next_value": float(gae_row["next_value"]),
            "critic_target": float(gae_row["critic_target"]), "TD residual": float(gae_row["td_residual"]),
            "td_residual": float(gae_row["td_residual"]), "GAE advantage": float(gae_row["raw_gae"]),
            "raw_gae": float(gae_row["raw_gae"]), "normalized advantage": float(gae_row["normalized_advantage"]),
            "normalized_advantage": float(gae_row["normalized_advantage"]),
            "reward_gae_component": float(gae_row["reward_gae_component"]),
            "critic_bootstrap_gae_component": float(gae_row["critic_bootstrap_gae_component"]),
            "gae_recursive_term": float(gae_row["gae_recursive_term"]), "temporally_propagated": bool(gae_row["temporally_propagated"]),
            "category": str(e1_row["category"]), "actor_eligible_e1": bool(e1_row["actor_eligible_e1"]),
            "critic_eligible_e1": True, "ppo_ratios": ratios,
            "policy-gradient contribution": float(gae_row["normalized_advantage"]) * float(ratios[-1]),
        })
    require(len(rows) == 24 and all(row["selected_candidate_id"] == row["applied_candidate_id"] == row["credited_candidate_id"] for row in rows),
            INTEGRITY_BLOCK, f"credit_audit_identity={model['cell_id']}")
    return rows


def _cell_training_audit(*, model: Mapping[str, Any], rollout: Mapping[str, Any], prepared: Mapping[str, Any],
                         training: Mapping[str, Any], initial_checkpoint: Mapping[str, Any],
                         final_checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    credit_rows = make_credit_audit_rows(model=model, prepared=prepared, training=training)
    categories = Counter(str(row["category"]) for row in credit_rows)
    return {
        "binding": {
            "cell_id": str(model["cell_id"]), "s3_cell_id": str(model["s3_cell_id"]),
            "environment_replicate": str(model["environment_replicate"]),
            "initialization_replicate": str(model["initialization_replicate"]),
            "environment_seed": int(model["environment_seed"]), "actor_seed": int(model["actor_seed"]),
            "critic_seed": int(model["critic_seed"]),
            "behavior_actor_checkpoint_sha256": str(model["behavior_actor_checkpoint_sha256"]),
            "bound_actor_initial_checkpoint_sha256": str(model["actor_initial_checkpoint_sha256"]),
            "on_policy_sha_match": str(model["behavior_actor_checkpoint_sha256"]) == str(model["actor_initial_checkpoint_sha256"]),
            "historical_initial_checkpoint_path": str(model["historical_initial_checkpoint_path"]),
            "initial_evidence_checkpoint_sha256": str(initial_checkpoint["sha256"]),
        },
        "rollout": {"train_window_ids": [str(row["window_id"]) for row in rollout["windows"]],
                    "trajectory_ids": sorted(rollout["budget"].trajectory_ids), "decision_ids": sorted(rollout["budget"].decision_ids),
                    "train_windows": 6, "trajectories": 6, "decisions": 24, "causal_transitions": 48,
                    "authoritative_candidate_generation": 24, "factory_evaluations": int(rollout["factory_evaluations"]),
                    "factory_source_mutations": int(rollout["factory_source_mutations"])},
        "initial": training["initial"] | {"checkpoint_sha256": str(initial_checkpoint["sha256"]),
                                                "actor_rng_digest": str(model["actor_rng_digest"]),
                                                "critic_rng_digest": str(model["critic_rng_digest"])},
        "final": training["final"] | {"checkpoint_sha256": str(final_checkpoint["sha256"])},
        "batch": {"rows": 24, "trajectories": 6, "normalization": prepared["normalization"],
                  "historical_f1_train_batch_reused": False},
        "eligibility": {"e1_contract_sha256": E1_CONTRACT_SHA256,
                        "actor_eligible_semantic_rows": int(prepared["mask"].actor_eligible_count),
                        "actor_active_rows": int(prepared["active_actor_rows"]),
                        "actor_ineligible_rows": int(prepared["mask"].actor_ineligible_count),
                        "critic_eligible_rows": 24, "category_counts": dict(prepared["mask"].category_counts),
                        "actor_status": "ACTOR_TRAINED_E1_ELIGIBLE" if prepared["active_actor_rows"] else "ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT"},
        "reward_ancestry": {"category_counts": dict(categories),
                            "fixed_gae_additivity_residual_max_abs": prepared["fixed_gae_additivity_residual_max_abs"],
                            "fixed_gae_raw_delta_max_abs": prepared["fixed_gae_raw_delta_max_abs"]},
        "updates": training["updates"], "step_ledger": training["step_ledger"],
        "parameter_change": training["parameter_change"], "gradient_isolation": training["gradient_isolation"],
        "learning_signal": {"informative_decisions": sum(float(row["reward"]) != 0.0 for row in credit_rows),
                            "multi_candidate_decisions": sum(bool(row["selected_action_type"] == "CANDIDATE") for row in credit_rows),
                            "temporally_propagated_advantages": sum(bool(row["temporally_propagated"]) for row in credit_rows),
                            "zero_loss_selective_states": sum(bool(row["snapshot_digest"]) for row in credit_rows)},
        "integrity": {"selected_applied_credited_identity": True, "candidate_regeneration_after_selection": 0,
                      "candidate_regeneration_during_ppo": 0, "review_optimizer_rows": 0, "future_leakage": 0,
                      "nan_or_inf": 0, "source_state_mutation_during_shadow_evaluation": int(rollout["factory_source_mutations"])},
    }


def _write_block(*, root: Path, source: Mapping[str, Any], code: str, detail: str,
                 preflight: Mapping[str, Any], counters: Mapping[str, Any], frozen: Mapping[str, Any] | None = None) -> None:
    """Produce a complete append-only failure artifact without a repair attempt."""
    outputs = {
        "bt8r13_mps_preflight.json": dict(preflight),
        "bt8r13_execution_manifest.json": {"source": dict(source), "authorized": False, "failure": detail,
                                            "execution_mode": "M2_FOUR_CELL_FRESH_CAUSAL_ROLLOUT"},
        "bt8r13_cell_training_audit.json": {"not_completed": True},
        "bt8r13_credit_eligibility.json": {"not_completed": True},
        "bt8r13_candidate_plan_credit.json": {"not_completed": True},
        "bt8r13_review_binding.json": {"not_completed": True},
        "bt8r13_checkpoint_manifest.json": {"not_completed": True},
        "test_results.json": {"execution_counters": dict(counters), "hard_failures": [detail], "warnings": [],
                              "global_locks": LOCKS},
        "frozen_hash_before_after.json": dict(frozen or {"not_completed": True}),
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                               "source_commit": source.get("source_commit"), "hard_failures": [detail], "warnings": [],
                               "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# BT8-R13 blocked\n\n- gate: `{code}`\n- detail: `{detail}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*")
                if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source.get("source_commit"),
                                  "github_push_performed": False, "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "simulator"))
    import joint_assignment_credit_contract as CC
    import joint_assignment_e1_eligibility as E1
    import joint_assignment_f1_execution_contract as FC
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import joint_candidate_plan_causal_bridge as CB
    import joint_candidate_plan_execution as PE
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_f1_bounded_training as F1MOD
    import run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection as R12MOD
    import run_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility as R7MOD
    import simulator_authorization as AUTH
    import test_h4m_ae_r3_causal_kpi_bridge as R3
    from rewards.mappo_reward_v1 import compute_reward_v2

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), AUTH_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    counters = initial_counters()
    AUTH.reset_audit_log()
    preflight = mps_preflight(H=H, JL=JL, MC=MC)
    if not preflight.get("passed"):
        _write_block(root=root, source=source, code=MPS_BLOCK, detail=str(preflight.get("failure_reason", "MPS_PREFLIGHT_FAILED")),
                     preflight=preflight, counters=counters)
        print(f"[BLOCKED] {MPS_BLOCK}")
        print(f"artifact: {root.relative_to(PROJECT)}")
        return
    device = torch.device("mps:0")
    require(preflight.get("selected_device") == "mps:0" and preflight.get("scorer_input_dimension") == 640
            and preflight.get("cpu_fallback") == 0, AUTH_BLOCK, "mps_preflight_shape_device")
    authority: dict[str, Any] | None = None
    frozen_before: dict[str, Any] | None = None
    extra_before: dict[str, str] | None = None
    try:
        authority = load_authority(R12MOD=R12MOD, E1=E1)
        frozen_before = dict(authority["frozen_before"])
        extra_before = {"e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
                        "candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
                        "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}
        adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
        config = F1MOD.actor_config(FPS, H, adim, cdim)
        features = F1MOD.feature_contract(FPS, adim, cdim)
        require(config.get("actor_head_version") == "LS3_BT8_R3_V2" and int(config.get("candidate_dim", -1)) == 8,
                AUTH_BLOCK, "v2_actor_config")
        cells = {str(cell["cell_id"]): load_cell_models(cell=cell, authority=authority, config=config, H=H, JL=JL,
                                                          MC=MC, device=device) for cell in authority["cells"]}
        independence = assert_model_independence(cells)
        historical_initial_before = _check_historical_initials(authority)
        require(historical_initial_before == authority["historical_initial_checkpoint_sha256"], AUTH_BLOCK, "historical_checkpoint_before")
        initial_dir, final_dir = root / "initial_checkpoints", root / "final_checkpoints"
        initial_dir.mkdir(); final_dir.mkdir()
        initial_checkpoints: dict[str, dict[str, Any]] = {}
        for cell_id, model in cells.items():
            record = _write_checkpoint(path=initial_dir / f"{cell_id}_initial.pt", model=model,
                                       kind="s3_m2_initial_evidence_copy", extra={
                                           "initialization_replicate": str(model["initialization_replicate"]),
                                           "historical_initial_checkpoint_sha256": str(model["actor_initial_checkpoint_sha256"]),
                                           "actor_initial_digest": str(model["actor_initial_digest"]),
                                           "critic_initial_digest": str(model["critic_initial_digest"]),
                                           "actor_seed": int(model["actor_seed"]), "critic_seed": int(model["critic_seed"]),
                                           "optimizer_state": "fresh_empty_not_serialized", "v1_transfer_count": 0,
                                           "prior_checkpoint_or_optimizer_reuse_count": 0,
                                       })
            record["absolute_path"] = str(initial_dir / f"{cell_id}_initial.pt")
            record["strict_load"] = True
            initial_checkpoints[cell_id] = record
            counters["checkpoint_write"] += 1
        review_entries = _review_entries_by_environment(authority["review_collection"])
        # Strictly load the just-written initial evidence before initial pure replay.
        initial_replay_models = {cell_id: {**model, "actor": _strict_actor_from_checkpoint(
            path=Path(record["absolute_path"]), expected_actor_digest=str(model["actor_initial_digest"]), config=config,
            H=H, JL=JL, MC=MC, device=device)} for cell_id, (model, record) in
            {key: (cells[key], initial_checkpoints[key]) for key in cells}.items()}
        initial_replays = review_replay(models=initial_replay_models, review_entries=review_entries, FPS=FPS, F1MOD=F1MOD,
                                        H=H, TIE=TIE, device=device, kind="initial", checkpoint_by_cell=initial_checkpoints)
        require(all(row["tolerance"] == 0.0 and row["selector"] == TIE.TIE_BREAK_CONTRACT_ID for row in initial_replays),
                INTEGRITY_BLOCK, "t1_review_binding")
        # The capability envelope opens only after all authority, model, and review preconditions are complete.
        train_entries: list[dict[str, Any]] = []
        cell_rollouts: dict[str, dict[str, Any]] = {}
        prepared: dict[str, dict[str, Any]] = {}
        training: dict[str, dict[str, Any]] = {}
        AUTH.reset_audit_log()
        with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                          reason="BT8-R13 explicitly authorized M2 four-cell fresh causal execution"):
            for cell_id, model in cells.items():
                torch.manual_seed(int(model["environment_seed"]))
                factory = BT6.RepairedSupportFactory()
                agent_slots = F1MOD.f1_agent_slot_mapping(
                    eligible=factory.eligible, source_groups=[str(row["source_group"]) for row in authority["train_windows"]], slots=8)
                rollout = rollout_cell(model=model, authority=authority, factory=factory, agent_slots=agent_slots,
                                       config=config, features=features, frozen=frozen_before, root=root, snapshot_entries=train_entries,
                                       F1MOD=F1MOD, H=H, JL=JL, CC=CC, CB=CB, PE=PE, MC=MC, BT1=BT1, BT6=BT6, R3=R3,
                                       compute_reward_v2=compute_reward_v2, FPS=FPS, device=device, counters=counters)
                counters["source_state_mutation_during_shadow_evaluation"] += int(rollout["factory_source_mutations"])
                prepared_cell = build_credit_and_batch(rollout=rollout, model=model, JL=JL, CC=CC, E1=E1, R7MOD=R7MOD,
                                                        FC=FC, BT6=BT6, device=device, adim=adim, cdim=cdim)
                prepared_cell["rollout_budget"] = rollout["budget"]
                training_cell = train_cell(model=model, prepared=prepared_cell, JL=JL, AUTH=AUTH, device=device, counters=counters)
                cell_rollouts[cell_id], prepared[cell_id], training[cell_id] = rollout, prepared_cell, training_cell
            counters["training"] = 1; counters["mps_training"] = 1
        capabilities_after = AUTH.authorization_state()["capabilities"]
        require(not any(capabilities_after.values()), INTEGRITY_BLOCK, "capability_not_revoked")
        require((counters["causal_rollout"], counters["simulator_step"], counters["critic_optimizer_step"]) == (192, 192, 12)
                and counters["actor_optimizer_step"] <= 12 and counters["raw_optimizer_step"] <= 24,
                INTEGRITY_BLOCK, "aggregate_budget")
        require(len(train_entries) == 96 and len({str(row["decision_id"]) for row in train_entries}) == 96,
                INTEGRITY_BLOCK, "training_snapshot_count")
        final_checkpoints: dict[str, dict[str, Any]] = {}
        for cell_id, model in cells.items():
            audit = training[cell_id]
            record = _write_checkpoint(path=final_dir / f"{cell_id}_final.pt", model=model,
                                       kind="s3_m2_final_frozen_evidence", extra={
                                           "initial_checkpoint_sha256": str(initial_checkpoints[cell_id]["sha256"]),
                                           "historical_initial_checkpoint_sha256": str(model["actor_initial_checkpoint_sha256"]),
                                           "actor_initial_digest": str(model["actor_initial_digest"]),
                                           "critic_initial_digest": str(model["critic_initial_digest"]),
                                           "actor_final_digest": str(audit["final"]["actor_digest"]),
                                           "critic_final_digest": str(audit["final"]["critic_digest"]),
                                           "actor_optimizer_steps": int(audit["step_ledger"]["actor_optimizer_steps"]),
                                           "critic_optimizer_steps": int(audit["step_ledger"]["critic_optimizer_steps"]),
                                       })
            record["absolute_path"] = str(final_dir / f"{cell_id}_final.pt")
            record["strict_load"] = True
            final_checkpoints[cell_id] = record
            counters["checkpoint_write"] += 1
        final_replay_models = {cell_id: {**model, "actor": _strict_actor_from_checkpoint(
            path=Path(record["absolute_path"]), expected_actor_digest=str(training[cell_id]["final"]["actor_digest"]), config=config,
            H=H, JL=JL, MC=MC, device=device)} for cell_id, (model, record) in
            {key: (cells[key], final_checkpoints[key]) for key in cells}.items()}
        final_replays = review_replay(models=final_replay_models, review_entries=review_entries, FPS=FPS, F1MOD=F1MOD,
                                      H=H, TIE=TIE, device=device, kind="final", checkpoint_by_cell=final_checkpoints)
        historical_initial_after = _check_historical_initials(authority)
        frozen_after = R12MOD.frozen_hashes()
        extra_after = {"e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
                       "candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
                       "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}
        require(historical_initial_before == historical_initial_after and frozen_before == frozen_after and extra_before == extra_after,
                INTEGRITY_BLOCK, "frozen_or_historical_input_changed")
        cell_audits = {cell_id: _cell_training_audit(model=cells[cell_id], rollout=cell_rollouts[cell_id],
                                                     prepared=prepared[cell_id], training=training[cell_id],
                                                     initial_checkpoint=initial_checkpoints[cell_id], final_checkpoint=final_checkpoints[cell_id])
                       for cell_id in cells}
        credit_rows = [row for cell_id in cells for row in make_credit_audit_rows(model=cells[cell_id], prepared=prepared[cell_id], training=training[cell_id])]
        require(len(credit_rows) == 96 and len({row["decision_id"] for row in credit_rows}) == 96, INTEGRITY_BLOCK, "credit_row_count")
        all_identity = all(row["selected_candidate_id"] == row["applied_candidate_id"] == row["credited_candidate_id"] for row in credit_rows)
        require(all_identity, INTEGRITY_BLOCK, "candidate_credit_identity")
        train_collection = F1MOD.finish_snapshot_store(
            root=root / "bt8r13_training_snapshots", kind="training", entries=train_entries, FPS=FPS,
            checkpoints={"initial_actor_checkpoint_sha256_by_cell": {key: value["sha256"] for key, value in initial_checkpoints.items()},
                         "final_actor_checkpoint_sha256_by_cell": {key: value["sha256"] for key, value in final_checkpoints.items()},
                         "historical_f1_training_collection_reused": False})
        # The collection digest binds all 96 lossless inputs; it does not mutate or recreate review snapshots.
        review_binding = {
            "historical_review_collection_path": str(F1 / "bt8f1_review_snapshots" / "collection_manifest.json"),
            "review_collection_digest": authority["review_collection"]["collection_digest"], "unique_snapshot_count": 6,
            "environment_bindings": {env: {"snapshot_digests": [str(row["snapshot_digest"]) for row in entries],
                                             "cells": ["AC-R1", "BD-R1"] if env == "F1_R1" else ["AC-R2", "BD-R2"],
                                             "same_input_for_cells": True} for env, entries in review_entries.items()},
            "initial_replays": initial_replays, "final_replays": final_replays,
            "t1_selector": TIE.TIE_BREAK_CONTRACT_ID, "t1_tolerance": 0.0,
            "training_time_selector": "H.select frozen F1 rollout path; T1 not used in training-time rollout",
            "optimizer_exposure": 0, "training_batch_inclusion": 0, "candidate_regeneration": 0,
            "simulator_execution": 0, "local_search_rerun": 0, "zero_loss_reevaluation": 0,
            "interpretation_performed": False,
        }
        all_zero_integrity = all(counters[name] == 0 for name in (
            "candidate_regeneration_after_selection", "candidate_regeneration_during_ppo", "local_search_rerun_during_ppo",
            "zero_loss_reevaluation_during_ppo", "unauthorized_optimizer_step", "checkpoint_promotion", "review_optimizer_rows",
            "review_batch_inclusion", "review_regeneration", "future_leakage", "test6_access", "github_push", "nan_or_inf",
            "seed_cell_mixing", "cross_trajectory_contamination", "candidate_identity_mismatch", "candidate_plan_execution_collapse",
            "serve_fallback", "zero_loss_violation", "illegal_or_masked_selection", "source_state_mutation_during_shadow_evaluation"))
        require(all_zero_integrity, INTEGRITY_BLOCK, "nonzero_integrity_counter")
        zero_eligible_cells = [cell_id for cell_id, item in prepared.items() if int(item["active_actor_rows"]) == 0]
        classification = PASS_CLASS_COMPLETE if not zero_eligible_cells else PASS_CLASS_INCOMPLETE
        outputs = {
            "bt8r13_mps_preflight.json": preflight,
            "bt8r13_execution_manifest.json": {
                "source": authority["source"], "r12_source": R12_SOURCE, "r11_s3_contract_sha256": S3_CONTRACT_SHA256,
                "e1_source": E1_SOURCE, "e1_contract_sha256": E1_CONTRACT_SHA256, "r4a_bridge_source": R4A_SOURCE,
                "execution_mode": "M2_FOUR_CELL_FRESH_CAUSAL_ROLLOUT", "cells": authority["cells"],
                "cell_independence": independence, "on_policy_release": {cell_id: audit["binding"]["on_policy_sha_match"] for cell_id, audit in cell_audits.items()},
                "authorized_budget": authority["r12_budget"], "actual_counters": counters,
                "historical_f1_training_collection_reused": False, "review_collection_recreated": False,
                "capability_events": AUTH.audit_log(), "capabilities_after": capabilities_after, "global_locks_after": LOCKS,
            },
            "bt8r13_cell_training_audit.json": cell_audits,
            "bt8r13_credit_eligibility.json": {
                "contract_id": E1.E1_CONTRACT_ID, "contract_sha256": E1_CONTRACT_SHA256,
                "normalization_order": "trajectory-local GAE -> cell-local N0 over 24 -> E1 actor mask",
                "cells": {cell_id: {"mask_evidence_digest": prepared[cell_id]["mask"].evidence_digest,
                                    "category_counts": dict(prepared[cell_id]["mask"].category_counts),
                                    "actor_eligible_semantic_rows": prepared[cell_id]["mask"].actor_eligible_count,
                                    "actor_active_rows": prepared[cell_id]["active_actor_rows"], "critic_eligible_rows": 24,
                                    "rows": [row for row in credit_rows if row["cell_id"] == cell_id]} for cell_id in cells},
                "insufficient_identity_credit_rows": sum(row["category"] == "INSUFFICIENT_IDENTITY_CREDIT" for row in credit_rows),
            },
            "bt8r13_candidate_plan_credit.json": {
                "rows": credit_rows, "identity_chain": "selected == applied == credited", "identity_chain_all": all_identity,
                "mismatches": 0, "candidate_plan_execution_collapse": 0, "candidate_regeneration_after_selection": 0,
                "serve_fallback": 0, "source_mutation_during_shadow_evaluation": 0,
            },
            "bt8r13_review_binding.json": review_binding,
            "bt8r13_checkpoint_manifest.json": {
                "initial": initial_checkpoints, "final": final_checkpoints,
                "historical_source_initial_checkpoint_sha256": authority["historical_initial_checkpoint_sha256"],
                "strict_load_validation": True, "test_only": True, "bounded": True, "non_promotable": True,
                "winner": False, "best_model": False, "promotion": False,
            },
            "bt8r13_training_snapshot_collection.json": train_collection,
            "test_results.json": {
                "execution_counters": counters, "authorized_vs_actual": {"decisions": [96, 96], "trajectories": [24, 24],
                    "causal_transitions": [192, counters["causal_rollout"]], "critic_steps": [12, counters["critic_optimizer_step"]],
                    "actor_steps_upper_bound": [12, counters["actor_optimizer_step"]], "raw_optimizer_upper_bound": [24, counters["raw_optimizer_step"]]},
                "training_snapshots": {"preserved": len(train_entries), "expected": 96, "unique": len({row["decision_id"] for row in train_entries}),
                                       "collection_digest": train_collection["collection_digest"]},
                "review_snapshots": {"preserved_historical": 6, "expected": 6, "recreated": 0},
                "hard_failures": [], "warnings": [], "global_locks": LOCKS, "github_push_performed": False,
            },
            "frozen_hash_before_after.json": {
                "before": frozen_before, "after": frozen_after, "all_unchanged": frozen_before == frozen_after,
                "extra_before": extra_before, "extra_after": extra_after, "extra_all_unchanged": extra_before == extra_after,
                "historical_initial_checkpoint_raw_digest_before": historical_initial_before,
                "historical_initial_checkpoint_raw_digest_after": historical_initial_after,
                "historical_initial_checkpoint_unchanged": historical_initial_before == historical_initial_after,
                "cell_parameter_changes": {cell_id: training[cell_id]["parameter_change"] for cell_id in cells},
            },
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": classification,
                                   "source_commit": authority["source"]["source_commit"], "hard_failures": [], "warnings": [],
                                   "global_locks": LOCKS, "next_step": "S3 same-support frozen-policy factor review"},
        }
        for name, payload in outputs.items():
            dump(root / name, payload)
        (root / "final_report.md").write_text(
            "# BT8-R13 final report\n\n"
            f"- gate: `{PASS_GATE}`\n- classification: `{classification}`\n- source commit: `{authority['source']['source_commit']}`\n"
            "- mode: `M2_FOUR_CELL_FRESH_CAUSAL_ROLLOUT`\n"
            f"- cells with Actor status `ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT`: `{zero_eligible_cells}`\n"
            "- review interpretation performed: `false`\n"
            "- next step: `S3 same-support frozen-policy factor review`\n", encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*")
                    if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE, "classification": classification,
                                      "source_commit": authority["source"]["source_commit"], "github_push_performed": False,
                                      "file_sha256": manifest})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(f"classification: {classification}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except R13Error as exc:
        frozen = {"before": frozen_before, "after": R12MOD.frozen_hashes() if frozen_before is not None else None,
                  "all_unchanged": frozen_before == R12MOD.frozen_hashes() if frozen_before is not None else None}
        _write_block(root=root, source=source, code=exc.code if exc.code == MPS_BLOCK else INTEGRITY_BLOCK,
                     detail=str(exc), preflight=preflight, counters=counters, frozen=frozen)
        print(f"[BLOCKED] {exc.code}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        frozen = {"before": frozen_before, "after": R12MOD.frozen_hashes() if frozen_before is not None else None,
                  "all_unchanged": frozen_before == R12MOD.frozen_hashes() if frozen_before is not None else None}
        _write_block(root=root, source=source, code=INTEGRITY_BLOCK, detail=f"{type(exc).__name__}:{exc}",
                     preflight=preflight, counters=counters, frozen=frozen)
        print(f"[BLOCKED] {INTEGRITY_BLOCK}")
        print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
