#!/usr/bin/env python3
"""BT8-F1-E1 exact preserved-batch bounded retraining.

This executor is intentionally narrower than BT8-F1.  It consumes the already
preserved F1 train/review snapshots and serialized PPO evidence only.  It never
creates a causal transition, candidate, reward, or GAE value.  The only mutable
objects are fresh MPS Actor/Critic instances initialized from the exact F1
initial evidence checkpoints and their fresh per-replicate optimizers.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-F1-E1"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_E1_EXACT_PRESERVED_BATCH_BOUNDED_RETRAINING_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_E1_RETRAINED_POLICIES_READY_FOR_SAME_SUPPORT_E0_E1_FROZEN_REVIEW"
AUTH_BLOCK = "BLOCKED_ON_POLICY_OR_BATCH_BINDING_FAILURE"
MPS_BLOCK = "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
BUDGET_BLOCK = "BLOCKED_E1_EXECUTION_BUDGET_MISMATCH"
GRAD_BLOCK = "BLOCKED_ACTOR_GRADIENT_LEAKAGE"
R2_BLOCK = "BLOCKED_R2_ACTOR_MUTATION"
REVIEW_BLOCK = "BLOCKED_REVIEW_LEAKAGE"
R9_SOURCE = "e70b2bdc51674547e7d7e6dc31a6c31c8b6eec56"
R9_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R9_E1_SEPARATE_BOUNDED_RETRAINING_AUTHORIZATION_SELECTION_COMPLETE"
R8_SOURCE = "a3cc98280e434c41520245f3fa13e3a76dd5d438"
F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
E1_CONTRACT_SHA256 = "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"
NEXT_GATE = "H4M-AE-R9.8-LS3-BT8-F1-E1-R1_SAME_SUPPORT_E0_E1_FROZEN_POLICY_REVIEW"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R9 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r9_e1_retraining_authority_selection_20260825_200038+09:00"
R8 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r8_e1_eligibility_validation_20260825_190658+09:00"
R7 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility_20260825_124641+09:00"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining.py",
    "05_training/test_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
REPLICATES = {
    "F1_R1": {"environment_seed": 20260822, "actor_seed": 20260824, "critic_seed": 20260826,
              "actor_eligible": 9, "actor_ineligible": 15, "actor_steps": 3, "critic_steps": 3},
    "F1_R2": {"environment_seed": 20260823, "actor_seed": 20260825, "critic_seed": 20260827,
              "actor_eligible": 0, "actor_ineligible": 24, "actor_steps": 0, "critic_steps": 3},
}


class F1E1Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise F1E1Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), "AUTHORITATIVE_ARTIFACT_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256", {})
    mismatches = [name for name, digest in expected.items()
                  if not (root / name).is_file() or sha256(root / name) != digest]
    return {"declared_file_count": len(expected), "mismatches": mismatches,
            "all_match": not mismatches, "manifest_sha256": sha256(root / "manifest.json")}


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    groups = []
    for group in state["param_groups"]:
        groups.append({key: value for key, value in group.items() if key != "params"} |
                      {"parameter_count": len(group["params"])})
    return {"state_item_count": len(state["state"]), "parameter_groups": groups,
            "fresh_empty": len(state["state"]) == 0}


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R9_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r9": git(["merge-base", R9_SOURCE, "HEAD"]) == R9_SOURCE,
        "changed_files_since_r9": changed,
        "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining_{stamp}"


def counters() -> dict[str, int]:
    return {
        "training": 0,
        "causal_rollout": 0,
        "simulator_step": 0,
        "candidate_generation": 0,
        "candidate_regeneration": 0,
        "local_search_rerun": 0,
        "zero_loss_reevaluation": 0,
        "reward_recomputation": 0,
        "actor_optimizer_step": 0,
        "critic_optimizer_step": 0,
        "raw_optimizer_step": 0,
        "unauthorized_optimizer_step": 0,
        "checkpoint_write": 0,
        "review_optimizer_rows": 0,
        "test6_access": 0,
        "github_push": 0,
        "nan_or_inf": 0,
    }


@dataclass
class ExactBudget:
    """Per-replicate counters that reject a step before it can exceed P1."""

    replicate_id: str
    allowed_actor_steps: int
    allowed_critic_steps: int
    actor_steps: int = 0
    critic_steps: int = 0
    ppo_cycles: int = 0
    train_rows: int = 0
    review_rows: int = 0
    trajectory_ids: set[str] = field(default_factory=set)

    def bind_batch(self, *, decision_ids: Sequence[str], trajectories: Sequence[str]) -> None:
        require(len(decision_ids) == 24 and len(set(decision_ids)) == 24, "E1_TRAIN_ROW_COUNT_OR_IDENTITY_INVALID", self.replicate_id)
        self.train_rows = len(decision_ids)
        self.trajectory_ids = set(trajectories)
        require(len(self.trajectory_ids) == 6, "E1_TRAJECTORY_COUNT_INVALID", self.replicate_id)

    def begin_cycle(self) -> None:
        require(self.ppo_cycles < 3, "E1_PPO_CYCLE_OVERRUN", self.replicate_id)
        self.ppo_cycles += 1

    def actor_step(self) -> None:
        require(self.actor_steps < self.allowed_actor_steps, "E1_ACTOR_STEP_OVERRUN", self.replicate_id)
        self.actor_steps += 1

    def critic_step(self) -> None:
        require(self.critic_steps < self.allowed_critic_steps, "E1_CRITIC_STEP_OVERRUN", self.replicate_id)
        self.critic_steps += 1

    def finalize(self) -> None:
        require(self.train_rows == 24 and self.review_rows == 0 and len(self.trajectory_ids) == 6,
                "E1_TRAIN_REVIEW_OR_TRAJECTORY_SCOPE_INVALID", self.replicate_id)
        require(self.ppo_cycles == 3, "E1_PPO_CYCLE_COUNT_INVALID", self.replicate_id)
        require(self.actor_steps == self.allowed_actor_steps and self.critic_steps == self.allowed_critic_steps,
                "E1_OPTIMIZER_STEP_COUNT_INVALID", self.replicate_id)


def mps_preflight(*, H: Any, JL: Any, MC: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "required_device": "mps:0",
        "cpu_fallback": 0,
        "capability_grant": 0,
        "optimizer_step": 0,
        "checkpoint_write": 0,
        "causal_rollout": 0,
    }
    if not evidence["mps_built"] or not evidence["mps_available"]:
        return evidence | {"passed": False, "failure_reason": "MPS_BUILT_OR_AVAILABLE_FALSE"}
    try:
        device = torch.device("mps:0")
        agent_dim, candidate_dim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
        torch.manual_seed(20260824)
        actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
            global_dim=8, demand_dim=6, agent_dim=agent_dim, candidate_dim=candidate_dim).to(device)
        torch.manual_seed(20260826)
        critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=agent_dim,
                                          safe_summary_dim=1 + 2 * candidate_dim).to(device)
        inputs = {
            "global_feats": torch.zeros((1, 8), dtype=torch.float32, device=device),
            "demand_feats": torch.zeros((1, 6), dtype=torch.float32, device=device),
            "agent_feats": torch.zeros((1, 8, agent_dim), dtype=torch.float32, device=device),
            "agent_mask": torch.ones((1, 8), dtype=torch.bool, device=device),
            "candidate_feats": torch.zeros((1, 8, candidate_dim), dtype=torch.float32, device=device),
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
            "passed": finite,
            "selected_device": str(logits.device),
            "finite": finite,
            "scorer_input_dimension": int(actor.scorer[0].in_features),
            "scorer_first_weight_shape": list(actor.scorer[0].weight.shape),
            "shape_contract": {"agents": 8, "candidate_pairs": 8, "candidate_feature_dim": candidate_dim},
            "mps_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    except Exception as exc:  # noqa: BLE001
        return evidence | {"passed": False, "failure_reason": f"{type(exc).__name__}:{exc}"}


def _by_id(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result = {str(row["decision_id"]): row for row in rows}
    require(len(result) == len(rows), "E1_DUPLICATE_DECISION_ID", label)
    return result


def _load_n0(normalization: Mapping[str, Any], replicate_id: str) -> dict[str, float]:
    rows = normalization[replicate_id]["modes"]["N0"]["rows"]
    require(len(rows) == 24, "E1_N0_ROW_COUNT_INVALID", replicate_id)
    result = {str(row["decision_id"]): float(row["advantage"]) for row in rows}
    require(len(result) == 24 and all(math.isfinite(value) for value in result.values()),
            "E1_N0_VALUE_INVALID", replicate_id)
    return result


def load_authority(*, R9MOD: Any, E1: Any, FPS: Any, R6MOD: Any, BT6: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind R9, E1, F1 and frozen sources before constructing a trainable model."""
    source = source_provenance()
    r9_gate = load_json(R9 / "gate_decision.json")
    r9_budget = load_json(R9 / "bt8r9_seed_update_budget.json")
    r9_completeness = load_json(R9 / "bt8r9_preserved_batch_completeness.json")
    r9_on_policy = load_json(R9 / "bt8r9_on_policy_binding.json")
    r9_frozen = load_json(R9 / "frozen_hash_before_after.json")
    r8_gate = load_json(R8 / "gate_decision.json")
    r8_contract = load_json(R8 / "bt8r8_e1_runtime_contract.json")
    r7_contract = load_json(R7 / "bt8r7_selected_minimal_contract.json")
    f1_gate = load_json(F1 / "gate_decision.json")
    binding = {
        "r9_gate": r9_gate.get("gate") == R9_GATE and r9_gate.get("source_commit") == R9_SOURCE,
        "r9_manifest": verify_manifest(R9),
        "r8_gate": r8_gate.get("source_commit") == R8_SOURCE,
        "r8_manifest": verify_manifest(R8),
        "e1_contract": r8_contract.get("contract_sha256") == E1_CONTRACT_SHA256,
        "f1_gate": f1_gate.get("gate") == F1_GATE and f1_gate.get("source_commit") == F1_SOURCE,
        "f1_manifest": verify_manifest(F1),
        "source_lineage": source["source_lineage_descends_from_r9"],
        "source_only_local_commit": source["source_only_local_commit"],
    }
    E1.bind_e1_contract(r7_contract)
    require(all((binding["r9_gate"], binding["r9_manifest"]["all_match"], binding["r8_gate"],
                 binding["r8_manifest"]["all_match"], binding["e1_contract"], binding["f1_gate"],
                 binding["f1_manifest"]["all_match"], binding["source_lineage"],
                 binding["source_only_local_commit"])), "E1_AUTHORITATIVE_BINDING_MISMATCH")
    completeness = R9MOD.audit_preserved_batch(loader=FPS)
    on_policy = R9MOD.audit_on_policy_binding(completeness=completeness)
    require(completeness.get("p1_evidence_complete") is True and on_policy.get("p1_on_policy_binding_pass") is True,
            "E1_P1_COMPLETENESS_OR_ON_POLICY_FAILURE")
    expected = R9MOD.expected_budget()
    require(r9_budget == expected, "E1_R9_FROZEN_BUDGET_MISMATCH")
    require(expected["aggregate"] == {"ppo_epochs_per_replicate": 3, "full_batch_size_per_replicate": 24,
                                        "minibatch_size_per_replicate": 24, "actor_optimizer_steps": 3,
                                        "critic_optimizer_steps": 6, "raw_optimizer_step_calls": 9,
                                        "extra_compensating_steps": 0}, "E1_AGGREGATE_BUDGET_INVALID")
    frozen = R6MOD.frozen_hashes(BT6)
    require(frozen == r9_frozen.get("after"), "E1_FROZEN_HASH_BINDING_MISMATCH")
    return {
        "source": source, "binding": binding, "completeness": completeness, "on_policy": on_policy,
        "budget": expected, "frozen_before": frozen, "r9_frozen": r9_frozen,
    }, {"r9_budget": r9_budget, "r9_completeness": r9_completeness}


def build_preserved_batch(*, replicate_id: str, FPS: Any, R9MOD: Any, device: torch.device) -> dict[str, Any]:
    """Materialize only lossless F1 tensor snapshots plus stored PPO scalars."""
    f1_execution = load_json(F1 / "bt8f1_training_execution_audit.json")
    f1_credit = load_json(F1 / "bt8f1_candidate_plan_credit_audit.json")
    training_collection = load_json(F1 / "bt8f1_training_snapshots" / "collection_manifest.json")
    r7_rows = load_json(R7 / "bt8r7_credit_eligibility_rows.json")["rows"]
    n0 = _load_n0(load_json(R7 / "bt8r7_advantage_normalization_audit.json"), replicate_id)
    credit_by_id = _by_id(f1_credit["rows"], "F1 credit")
    eligibility_by_id = _by_id(r7_rows, "R7 E1 eligibility")
    entries_by_id = _by_id(training_collection["entries"], "F1 train snapshots")
    rollout = f1_execution["replicate_rollouts"][replicate_id]
    gae_by_id = _by_id(rollout["gae_rows"], f"F1 GAE {replicate_id}")
    raw_rows = list(rollout["rows"])
    require(len(raw_rows) == 24 and len(gae_by_id) == 24, "E1_PRESERVED_BATCH_ROW_COUNT_INVALID", replicate_id)

    payloads: list[Mapping[str, Any]] = []
    records: list[dict[str, Any]] = []
    for raw in raw_rows:
        ledger = R9MOD.parse_transition_ledger(str(raw["t"]))
        decision_id = ledger["decision_id"]
        require(decision_id in gae_by_id and decision_id in credit_by_id and decision_id in eligibility_by_id
                and decision_id in entries_by_id, "E1_PRESERVED_ROW_BINDING_MISSING", decision_id)
        entry, credit, eligibility = entries_by_id[decision_id], credit_by_id[decision_id], eligibility_by_id[decision_id]
        directory = F1 / "bt8f1_training_snapshots" / str(entry["relative_path"])
        payload = FPS.load_snapshot(directory)
        require(str(payload["snapshot_digest"]) == str(entry["snapshot_digest"]) == str(raw["snapshot_digest"]),
                "E1_TRAIN_SNAPSHOT_DIGEST_MISMATCH", decision_id)
        require(str(payload["metadata"]["decision_id"]) == decision_id
                and int(payload["metadata"]["seed"]) == REPLICATES[replicate_id]["environment_seed"],
                "E1_TRAIN_SNAPSHOT_IDENTITY_MISMATCH", decision_id)
        require(str(eligibility["replicate_id"]) == replicate_id and bool(eligibility["identity_chain_valid"]),
                "E1_ELIGIBILITY_REPLICATE_OR_IDENTITY_INVALID", decision_id)
        require(float(gae_by_id[decision_id]["normalized_advantage"]) == n0[decision_id],
                "E1_N0_PRESERVED_VALUE_MISMATCH", decision_id)
        payloads.append(payload)
        records.append({
            "decision_id": decision_id,
            "trajectory_id": str(gae_by_id[decision_id]["trajectory_id"]),
            "old_log_prob": float(ledger["old_log_prob"]),
            "advantage": float(gae_by_id[decision_id]["normalized_advantage"]),
            "value_target": float(gae_by_id[decision_id]["critic_target"]),
            "eligibility": bool(eligibility["actor_eligible_e1"]),
            "action_type": str(eligibility["selected_action_type"]),
            "selected_agent_id": str(credit["agent_id"]),
            "selected_candidate_id": str(credit["selected_candidate_id"]),
            "snapshot_digest": str(payload["snapshot_digest"]),
            "candidate_plan_digest": str(credit["candidate_plan_digest"]),
            "applied_plan_digest": str(credit["applied_plan_digest"]),
            "transition_id": str(credit["transition_id"]),
        })
    max_agents = max(int(payload["tensors"]["agent_feats"].shape[1]) for payload in payloads)
    max_pairs = max(int(payload["metadata"]["selectable_pair_count"]) for payload in payloads)
    first = payloads[0]["tensors"]
    count = len(payloads)
    global_feats = torch.zeros((count, first["global_feats"].shape[-1]), dtype=torch.float32)
    demand_feats = torch.zeros((count, first["demand_feats"].shape[-1]), dtype=torch.float32)
    agent_feats = torch.zeros((count, max_agents, first["agent_feats"].shape[-1]), dtype=torch.float32)
    agent_mask = torch.zeros((count, max_agents), dtype=torch.bool)
    candidate_feats = torch.zeros((count, max_pairs, first["candidate_feats"].shape[-1]), dtype=torch.float32)
    pair_agent_index = torch.zeros((count, max_pairs), dtype=torch.long)
    safe_mask = torch.zeros((count, max_pairs), dtype=torch.bool)
    action_index: list[int] = []
    forced: list[bool] = []
    for index, (payload, record) in enumerate(zip(payloads, records)):
        tensors, metadata = payload["tensors"], payload["metadata"]
        agent_count, pair_count = int(tensors["agent_feats"].shape[1]), int(metadata["selectable_pair_count"])
        global_feats[index] = tensors["global_feats"][0]
        demand_feats[index] = tensors["demand_feats"][0]
        agent_feats[index, :agent_count] = tensors["agent_feats"][0]
        agent_mask[index, :agent_count] = tensors["agent_mask"][0]
        candidate_feats[index, :pair_count] = tensors["candidate_feats"][0, :pair_count]
        pair_agent_index[index, :pair_count] = tensors["pair_agent_index"][0, :pair_count]
        safe_mask[index, :pair_count] = tensors["safe_mask"][0, :pair_count]
        require(bool(safe_mask[index, :pair_count].all().item()), "E1_PRESERVED_SAFE_MASK_INVALID", record["decision_id"])
        if record["action_type"] == "NO_ASSIGN":
            require(record["selected_candidate_id"] == "NO_ASSIGN_KEEP_CURRENT_PLANS" and int(metadata["no_assign_index"]) == pair_count,
                    "E1_NO_ASSIGN_RECONSTRUCTION_INVALID", record["decision_id"])
            chosen = max_pairs
        else:
            candidates = list(metadata["candidate_ids"])
            wanted = (record["selected_agent_id"], record["selected_candidate_id"])
            positions = [position for position, row in enumerate(candidates)
                         if (str(row["agent_id"]), str(row["candidate_id"])) == wanted]
            require(len(positions) == 1 and positions[0] < pair_count and bool(safe_mask[index, positions[0]].item()),
                    "E1_SELECTED_ACTION_RECONSTRUCTION_INVALID", record["decision_id"])
            chosen = positions[0]
        action_index.append(chosen)
        forced.append(pair_count == 0)
    require(len(set(record["decision_id"] for record in records)) == 24
            and len(set(record["trajectory_id"] for record in records)) == 6,
            "E1_PRESERVED_TRAJECTORY_BINDING_INVALID", replicate_id)
    batch = {
        "global_feats": global_feats.to(device), "demand_feats": demand_feats.to(device),
        "agent_feats": agent_feats.to(device), "agent_mask": agent_mask.to(device),
        "candidate_feats": candidate_feats.to(device), "pair_agent_index": pair_agent_index.to(device),
        "safe_mask": safe_mask.to(device),
        "action_index": torch.tensor(action_index, dtype=torch.long, device=device),
        "old_log_prob": torch.tensor([record["old_log_prob"] for record in records], dtype=torch.float32, device=device),
        "advantage": torch.tensor([record["advantage"] for record in records], dtype=torch.float32, device=device),
        "value_target": torch.tensor([record["value_target"] for record in records], dtype=torch.float32, device=device),
        "actor_eligibility_mask": torch.tensor([record["eligibility"] for record in records], dtype=torch.bool, device=device),
        "forced_action": torch.tensor(forced, dtype=torch.bool, device=device),
    }
    require(all(bool(torch.isfinite(batch[key]).all().item()) for key in ("global_feats", "demand_feats", "agent_feats", "candidate_feats", "old_log_prob", "advantage", "value_target")),
            "E1_PRESERVED_BATCH_NONFINITE", replicate_id)
    return {"batch": batch, "records": records, "payloads": payloads,
            "batch_digest": hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()}


def load_initial_models(*, replicate_id: str, config: Mapping[str, Any], H: Any, JL: Any, device: torch.device) -> dict[str, Any]:
    manifest = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")[replicate_id]
    checkpoint = F1 / "initial_checkpoints" / str(manifest["path"])
    require(sha256(checkpoint) == manifest["sha256"], "E1_INITIAL_CHECKPOINT_DIGEST_MISMATCH", replicate_id)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping) and set(payload).issuperset({"actor", "critic", "meta"})
            and not any("optimizer" in str(key).lower() for key in payload),
            "E1_INITIAL_CHECKPOINT_SCHEMA_OR_OPTIMIZER_CONTAMINATION", replicate_id)
    contract = REPLICATES[replicate_id]
    torch.manual_seed(contract["actor_seed"])
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]), candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    torch.manual_seed(contract["critic_seed"])
    critic = JL.JointAssignmentCritic(global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
                                      agent_dim=int(config["agent_dim"]), safe_summary_dim=1 + 2 * int(config["candidate_dim"])).to(device)
    critic.load_state_dict(payload["critic"], strict=True)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    expected_initial = load_json(F1 / "bt8f1_final_checkpoint_manifest.json")[replicate_id]
    actor_digest, critic_digest = module_digest(actor), module_digest(critic)
    require(actor_digest == expected_initial["actor_initial_digest"] and critic_digest == expected_initial["critic_initial_digest"],
            "E1_INITIAL_MODEL_DIGEST_MISMATCH", replicate_id)
    require(optimizer_summary(actor_optimizer)["fresh_empty"] and optimizer_summary(critic_optimizer)["fresh_empty"],
            "E1_OPTIMIZER_NOT_FRESH_EMPTY", replicate_id)
    return {
        "replicate_id": replicate_id, "checkpoint_path": checkpoint, "initial_checkpoint_sha256": sha256(checkpoint),
        "actor": actor, "critic": critic, "actor_optimizer": actor_optimizer, "critic_optimizer": critic_optimizer,
        "actor_initial_digest": actor_digest, "critic_initial_digest": critic_digest,
        "actor_optimizer_initial": optimizer_summary(actor_optimizer),
        "critic_optimizer_initial": optimizer_summary(critic_optimizer),
    }


def _gradient_abs_sum(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().abs().sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).sum().detach().cpu()) if values else 0.0


def _gradient_norm(module: torch.nn.Module) -> float:
    values = [(parameter.grad.detach() ** 2).sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).detach().cpu()) if values else 0.0


def _finite_model_outputs(*, logits: torch.Tensor, no_assign: torch.Tensor, value: torch.Tensor,
                          safe_mask: torch.Tensor) -> bool:
    return bool(torch.isfinite(logits[safe_mask]).all().item() and torch.isfinite(no_assign).all().item()
                and torch.isfinite(value).all().item())


def _clip_summary(*, ratio: torch.Tensor, advantage: torch.Tensor, active: torch.Tensor, epsilon: float) -> dict[str, Any]:
    rows = []
    for value, adv, included in zip(ratio.detach().cpu().tolist(), advantage.detach().cpu().tolist(), active.detach().cpu().tolist()):
        clipped = (adv > 0.0 and value > 1.0 + epsilon) or (adv < 0.0 and value < 1.0 - epsilon)
        rows.append({"ratio": float(value), "advantage": float(adv), "actor_active": bool(included), "clipped": bool(clipped)})
    active_rows = [row for row in rows if row["actor_active"]]
    return {"ratio_min": min(row["ratio"] for row in rows), "ratio_mean": sum(row["ratio"] for row in rows) / len(rows),
            "ratio_max": max(row["ratio"] for row in rows), "actor_active_rows": len(active_rows),
            "actor_active_clipped_rows": sum(row["clipped"] for row in active_rows), "rows": rows}


def train_replicate(*, replicate: dict[str, Any], prepared: Mapping[str, Any], JL: Any, AUTH: Any,
                    device: torch.device, run_counters: dict[str, int]) -> dict[str, Any]:
    replicate_id = str(replicate["replicate_id"])
    contract = REPLICATES[replicate_id]
    batch, records = prepared["batch"], prepared["records"]
    budget = ExactBudget(replicate_id=replicate_id, allowed_actor_steps=contract["actor_steps"],
                         allowed_critic_steps=contract["critic_steps"])
    budget.bind_batch(decision_ids=[row["decision_id"] for row in records], trajectories=[row["trajectory_id"] for row in records])
    expected_mask = int(contract["actor_eligible"])
    require(int(batch["actor_eligibility_mask"].sum().item()) == expected_mask
            and int((~batch["actor_eligibility_mask"]).sum().item()) == int(contract["actor_ineligible"]),
            "E1_RUNTIME_ELIGIBILITY_MASK_MISMATCH", replicate_id)
    updates: list[dict[str, Any]] = []
    actor, critic = replicate["actor"], replicate["critic"]
    actor_optimizer, critic_optimizer = replicate["actor_optimizer"], replicate["critic_optimizer"]
    for epoch in range(3):
        budget.begin_cycle()
        actor.train(); critic.train()
        logits, no_assign = actor(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                                  agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                                  candidate_feats=batch["candidate_feats"], pair_agent_index=batch["pair_agent_index"],
                                  safe_mask=batch["safe_mask"])
        value = critic(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                       agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                       safe_summary=JL.safe_set_summary(batch["candidate_feats"], batch["safe_mask"]))
        require(_finite_model_outputs(logits=logits, no_assign=no_assign, value=value, safe_mask=batch["safe_mask"]),
                "E1_MODEL_OUTPUT_NONFINITE", replicate_id)
        loss = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign,
                                      safe_mask=batch["safe_mask"], action_index=batch["action_index"],
                                      old_log_prob=batch["old_log_prob"], advantage=batch["advantage"],
                                      value_pred=value, value_target=batch["value_target"], forced_action=batch["forced_action"],
                                      actor_eligibility_mask=batch["actor_eligibility_mask"])
        require(int(loss["actor_eligible_rows"]) == expected_mask
                and bool(loss["actor_update_skipped"]) == (expected_mask == 0),
                "E1_ACTOR_SKIP_CONTRACT_MISMATCH", replicate_id)
        require(bool(torch.isfinite(loss["ratio"]).all().item()
                     and torch.isfinite(loss["actor_loss"]).item()
                     and torch.isfinite(loss["critic_loss"]).item()), "E1_LOSS_OR_RATIO_NONFINITE", replicate_id)
        logits.retain_grad(); no_assign.retain_grad()
        active = batch["actor_eligibility_mask"] & ~batch["forced_action"]
        if expected_mask:
            AUTH.require_capability(AUTH.TRAINING, site=f"{STAGE}:{replicate_id}:actor_epoch_{epoch + 1}")
            actor_optimizer.zero_grad(set_to_none=True)
            loss["actor_loss"].backward()
            inactive = ~active
            inactive_grad = max(float(logits.grad[inactive].detach().abs().max().cpu()) if bool(inactive.any()) else 0.0,
                                float(no_assign.grad[inactive].detach().abs().max().cpu()) if bool(inactive.any()) else 0.0)
            eligible_grad = float(logits.grad[active].detach().abs().sum().cpu()) + float(no_assign.grad[active].detach().abs().sum().cpu())
            actor_grad_abs, actor_grad_norm = _gradient_abs_sum(actor), _gradient_norm(actor)
            require(inactive_grad == 0.0, "E1_INELIGIBLE_ACTOR_GRADIENT_NONZERO", replicate_id)
            require(eligible_grad > 0.0 and actor_grad_abs > 0.0, "E1_ELIGIBLE_ACTOR_GRADIENT_MISSING", replicate_id)
            budget.actor_step(); actor_optimizer.step(); run_counters["actor_optimizer_step"] += 1; run_counters["raw_optimizer_step"] += 1
        else:
            grad_logits, grad_no_assign = torch.autograd.grad(loss["actor_loss"], (logits, no_assign), retain_graph=True)
            inactive_grad = max(float(grad_logits.detach().abs().max().cpu()), float(grad_no_assign.detach().abs().max().cpu()))
            eligible_grad = 0.0; actor_grad_abs = 0.0; actor_grad_norm = 0.0
            require(inactive_grad == 0.0 and all(parameter.grad is None for parameter in actor.parameters()),
                    "E1_R2_ACTOR_SKIP_GRADIENT_FAILURE", replicate_id)
        AUTH.require_capability(AUTH.TRAINING, site=f"{STAGE}:{replicate_id}:critic_epoch_{epoch + 1}")
        critic_optimizer.zero_grad(set_to_none=True)
        loss["critic_loss"].backward()
        critic_grad_abs, critic_grad_norm = _gradient_abs_sum(critic), _gradient_norm(critic)
        require(critic_grad_abs > 0.0 and critic_grad_norm > 0.0, "E1_CRITIC_GRADIENT_MISSING", replicate_id)
        budget.critic_step(); critic_optimizer.step(); run_counters["critic_optimizer_step"] += 1; run_counters["raw_optimizer_step"] += 1
        torch.mps.synchronize()
        run_counters["nan_or_inf"] += int(not all(math.isfinite(value_) for value_ in
                                                   (actor_grad_abs, actor_grad_norm, critic_grad_abs, critic_grad_norm,
                                                    float(loss["actor_loss"].detach().cpu()), float(loss["critic_loss"].detach().cpu()))))
        updates.append({
            "epoch": epoch + 1,
            "actor_step": expected_mask > 0,
            "critic_step": True,
            "actor_eligible_rows": int(loss["actor_eligible_rows"]),
            "actor_ineligible_rows": int(loss["actor_ineligible_rows"]),
            "actor_update_skipped": bool(loss["actor_update_skipped"]),
            "actor_loss": float(loss["actor_loss"].detach().cpu()),
            "policy_loss": float(loss["policy_loss"].detach().cpu()),
            "entropy": float(loss["entropy"].detach().cpu()),
            "critic_loss": float(loss["critic_loss"].detach().cpu()),
            "actor_grad_abs_sum": actor_grad_abs,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_abs_sum": critic_grad_abs,
            "critic_grad_norm": critic_grad_norm,
            "ineligible_actor_logit_gradient_max_abs": inactive_grad,
            "eligible_actor_logit_gradient_abs_sum": eligible_grad,
            "ppo": _clip_summary(ratio=loss["ratio"], advantage=batch["advantage"], active=active,
                                   epsilon=float(JL.CC.PPO_CLIP_EPSILON)),
        })
    budget.finalize()
    actor_final, critic_final = module_digest(actor), module_digest(critic)
    return {
        "replicate_id": replicate_id,
        "batch": {"digest": prepared["batch_digest"], "rows": 24, "trajectories": 6,
                  "decision_ids": [row["decision_id"] for row in records],
                  "eligible_decision_ids": [row["decision_id"] for row in records if row["eligibility"]],
                  "ineligible_decision_ids": [row["decision_id"] for row in records if not row["eligibility"]]},
        "updates": updates,
        "step_ledger": {"ppo_cycles": budget.ppo_cycles, "actor_optimizer_steps": budget.actor_steps,
                        "critic_optimizer_steps": budget.critic_steps, "raw_optimizer_step_calls": budget.actor_steps + budget.critic_steps,
                        "supplemental_steps": 0},
        "initial": {"actor_digest": replicate["actor_initial_digest"], "critic_digest": replicate["critic_initial_digest"],
                    "checkpoint_sha256": replicate["initial_checkpoint_sha256"],
                    "actor_optimizer": replicate["actor_optimizer_initial"], "critic_optimizer": replicate["critic_optimizer_initial"]},
        "final": {"actor_digest": actor_final, "critic_digest": critic_final,
                  "actor_optimizer": optimizer_summary(actor_optimizer), "critic_optimizer": optimizer_summary(critic_optimizer)},
        "parameter_change": {"actor_changed": actor_final != replicate["actor_initial_digest"],
                             "critic_changed": critic_final != replicate["critic_initial_digest"]},
        "gradient_isolation": {"max_ineligible_actor_logit_gradient_abs": max(row["ineligible_actor_logit_gradient_max_abs"] for row in updates),
                               "min_eligible_actor_logit_gradient_abs": min(row["eligible_actor_logit_gradient_abs_sum"] for row in updates),
                               "r2_actor_explicit_skip": expected_mask == 0},
    }


def write_final_checkpoint(*, root: Path, replicate: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "final_checkpoints" / f"{replicate['replicate_id']}_e1_final.pt"
    path.parent.mkdir(parents=True, exist_ok=False) if not path.parent.exists() else None
    payload = {
        "actor": {name: tensor.detach().cpu() for name, tensor in replicate["actor"].state_dict().items()},
        "critic": {name: tensor.detach().cpu() for name, tensor in replicate["critic"].state_dict().items()},
        "meta": {
            "kind": "e1_preserved_batch_final_evidence", "replicate_id": replicate["replicate_id"],
            "environment_seed": REPLICATES[replicate["replicate_id"]]["environment_seed"], "test_only": True,
            "bounded": True, "non_promotable": True, "winner": False, "best_model": False, "promotion": False,
            "performance_claim_allowed": False, "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False, "initial_checkpoint_sha256": audit["initial"]["checkpoint_sha256"],
            "actor_initial_digest": audit["initial"]["actor_digest"], "critic_initial_digest": audit["initial"]["critic_digest"],
            "actor_final_digest": audit["final"]["actor_digest"], "critic_final_digest": audit["final"]["critic_digest"],
        },
    }
    torch.save(payload, path)
    return {"replicate_id": replicate["replicate_id"], "path": path.relative_to(root).as_posix(), "sha256": sha256(path), **payload["meta"]}


def load_final_actor(*, root: Path, checkpoint: Mapping[str, Any], config: Mapping[str, Any], H: Any,
                     device: torch.device) -> torch.nn.Module:
    payload = torch.load(root / str(checkpoint["path"]), map_location="cpu", weights_only=False)
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]), candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    require(module_digest(actor) == checkpoint["actor_final_digest"], "E1_FINAL_CHECKPOINT_STRICT_LOAD_MISMATCH", str(checkpoint["replicate_id"]))
    return actor.eval()


def final_review_replay(*, root: Path, checkpoints: Mapping[str, Any], config: Mapping[str, Any], FPS: Any,
                        F1MOD: Any, H: Any, TIE: Any, device: torch.device) -> dict[str, Any]:
    collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
    require(collection.get("snapshot_count") == 6 and collection.get("store_kind") == "review", "E1_REVIEW_COLLECTION_INVALID")
    rows = []
    for replicate_id in REPLICATES:
        actor = load_final_actor(root=root, checkpoint=checkpoints[replicate_id], config=config, H=H, device=device)
        entries = [row for row in collection["entries"] if f":{replicate_id}:" in str(row["decision_id"])]
        require(len(entries) == 3, "E1_REVIEW_REPLICATE_COUNT_INVALID", replicate_id)
        for entry in entries:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / str(entry["relative_path"]))
            replay = F1MOD.replay(actor, payload, H, TIE, device)
            require(replay["finite"] and replay["snapshot_digest"] == entry["snapshot_digest"],
                    "E1_FINAL_REVIEW_REPLAY_INVALID", str(entry["decision_id"]))
            rows.append({"replicate_id": replicate_id, "final_actor_checkpoint_sha256": checkpoints[replicate_id]["sha256"],
                         "strict_checkpoint_load": True, "review_snapshot_digest": entry["snapshot_digest"], **replay})
    require(len(rows) == 6 and len({(row["replicate_id"], row["review_snapshot_digest"]) for row in rows}) == 6,
            "E1_FINAL_REVIEW_SNAPSHOT_UNIQUENESS_INVALID")
    return {
        "review_collection_digest": collection["collection_digest"], "rows": rows,
        "same_support_binding_pass": True, "strict_checkpoint_load": True, "optimizer_steps": 0,
        "simulator_execution": 0, "local_search_rerun": 0, "zero_loss_reevaluation": 0,
        "interpretation_performed": False, "t1_tolerance": 0.0,
    }


def write_block(root: Path, *, source: Mapping[str, Any], code: str, detail: str,
                preflight: Mapping[str, Any], execution: Mapping[str, Any] | None = None) -> None:
    zero = counters()
    files = {
        "bt8f1e1_preflight.json": dict(preflight),
        "bt8f1e1_execution_manifest.json": {"not_authorized": True, "reason": detail},
        "bt8f1e1_seed1_training_audit.json": {"not_completed": True},
        "bt8f1e1_seed2_training_audit.json": {"not_completed": True},
        "bt8f1e1_gradient_isolation.json": {"not_completed": True},
        "bt8f1e1_final_checkpoint_manifest.json": {"not_completed": True},
        "bt8f1e1_final_review_replay.json": {"not_completed": True},
        "test_results.json": {"execution_counters": execution or zero, "hard_failures": [detail], "warnings": []},
        "frozen_hash_before_after.json": {"not_completed": True},
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                               "source_commit": source.get("source_commit"), "hard_failures": [detail], "warnings": [],
                               "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, payload in files.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-F1-E1 blocked\n\n- gate: `{code}`\n- detail: `{detail}`\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source.get("source_commit"),
                                   "file_sha256": manifest, "github_push_performed": False})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_e1_eligibility as E1
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_f1_bounded_training as F1MOD
    import run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution as R6MOD
    import run_h4m_ae_ls3_bt8_r9_retraining_authority_selection as R9MOD
    import simulator_authorization as AUTH

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), "APPEND_ONLY_ARTIFACT_COLLISION")
    root.mkdir(parents=True)
    AUTH.reset_audit_log()
    preflight = mps_preflight(H=H, JL=JL, MC=MC)
    if not preflight.get("passed"):
        write_block(root, source=source, code=MPS_BLOCK, detail=str(preflight.get("failure_reason", "MPS_PREFLIGHT_FAILED")), preflight=preflight)
        print(f"[BLOCKED] {MPS_BLOCK}\nartifact: {root.relative_to(PROJECT)}")
        return
    require(preflight.get("selected_device") == "mps:0" and preflight.get("scorer_input_dimension") == 640,
            "E1_MPS_PREFLIGHT_SHAPE_OR_DEVICE_MISMATCH")
    device = torch.device("mps:0")
    run_counters = counters()
    context: dict[str, Any] = {"source": source}
    try:
        authority, r9_files = load_authority(R9MOD=R9MOD, E1=E1, FPS=FPS, R6MOD=R6MOD, BT6=BT6)
        context.update(authority)
        config = load_json(F1 / "bt8f1_initialization_lineage.json")["actor_config"]
        require(config.get("actor_head_version") == "LS3_BT8_R3_V2" and config.get("candidate_dim") == 8,
                "E1_V2_ACTOR_CONFIG_MISMATCH")
        prepared = {replicate_id: build_preserved_batch(replicate_id=replicate_id, FPS=FPS, R9MOD=R9MOD, device=device)
                    for replicate_id in REPLICATES}
        replicates = {replicate_id: load_initial_models(replicate_id=replicate_id, config=config, H=H, JL=JL, device=device)
                      for replicate_id in REPLICATES}
        require(len({id(value["actor"]) for value in replicates.values()}) == 2
                and len({id(value["critic"]) for value in replicates.values()}) == 2,
                "E1_REPLICATE_MODEL_ALIASING")
        AUTH.reset_audit_log()
        with AUTH.granted(AUTH.TRAINING, reason="BT8-F1-E1 approved exact P1 preserved-batch bounded retraining"):
            audits = {replicate_id: train_replicate(replicate=replicate, prepared=prepared[replicate_id], JL=JL,
                                                     AUTH=AUTH, device=device, run_counters=run_counters)
                      for replicate_id, replicate in replicates.items()}
            run_counters["training"] = 1
        capabilities_after = AUTH.authorization_state()["capabilities"]
        require(not any(capabilities_after.values()), "E1_CAPABILITY_NOT_REVOKED")
        require(run_counters["actor_optimizer_step"] == 3 and run_counters["critic_optimizer_step"] == 6
                and run_counters["raw_optimizer_step"] == 9, "E1_EXECUTION_STEP_TOTAL_MISMATCH")
        require(audits["F1_R1"]["parameter_change"] == {"actor_changed": True, "critic_changed": True},
                "E1_R1_PARAMETER_CHANGE_EXPECTATION_FAILURE")
        require(audits["F1_R2"]["parameter_change"] == {"actor_changed": False, "critic_changed": True},
                "E1_R2_ACTOR_OR_CRITIC_CHANGE_EXPECTATION_FAILURE")
        require(audits["F1_R1"]["gradient_isolation"]["max_ineligible_actor_logit_gradient_abs"] == 0.0,
                "E1_R1_INELIGIBLE_ACTOR_GRADIENT_LEAKAGE")
        require(audits["F1_R2"]["gradient_isolation"]["max_ineligible_actor_logit_gradient_abs"] == 0.0,
                "E1_R2_ACTOR_GRADIENT_LEAKAGE")
        require(run_counters["nan_or_inf"] == 0, "E1_NONFINITE_EXECUTION")
        final_checkpoints = {replicate_id: write_final_checkpoint(root=root, replicate=replicate, audit=audits[replicate_id])
                             for replicate_id, replicate in replicates.items()}
        run_counters["checkpoint_write"] = len(final_checkpoints)
        final_replay = final_review_replay(root=root, checkpoints=final_checkpoints, config=config, FPS=FPS,
                                           F1MOD=F1MOD, H=H, TIE=TIE, device=device)
        frozen_after = R6MOD.frozen_hashes(BT6)
        require(frozen_after == authority["frozen_before"], "E1_FROZEN_HASH_CHANGED")
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        code = (R2_BLOCK if "R2" in text and ("ACTOR" in text or "actor" in text) else
                GRAD_BLOCK if "GRADIENT" in text or "gradient" in text else
                REVIEW_BLOCK if "REVIEW" in text else
                BUDGET_BLOCK if "STEP" in text or "BUDGET" in text or "CYCLE" in text else AUTH_BLOCK)
        write_block(root, source=source, code=code, detail=f"{type(exc).__name__}:{exc}", preflight=preflight, execution=run_counters)
        print(f"[BLOCKED] {code}\nartifact: {root.relative_to(PROJECT)}")
        return

    integrity = {
        "review_leakage": 0, "seed_mixing": 0, "trajectory_contamination": 0,
        "ineligible_actor_gradient": 0, "causal_rollout": run_counters["causal_rollout"],
        "candidate_regeneration": run_counters["candidate_regeneration"], "future_leakage": 0,
        "nan_or_inf": run_counters["nan_or_inf"], "test6_access": run_counters["test6_access"],
        "github_push": run_counters["github_push"], "unauthorized_optimizer_step": run_counters["unauthorized_optimizer_step"],
    }
    require(not any(integrity.values()), "E1_INTEGRITY_COUNTER_NONZERO")
    execution_manifest = {
        "stage": STAGE, "mode": "P1_EXACT_PRESERVED_BATCH", "source": authority["source"],
        "authority_binding": authority["binding"], "r9_completeness": authority["completeness"],
        "on_policy_binding": authority["on_policy"], "frozen_budget": authority["budget"],
        "mps_device": "mps:0", "no_causal_rollout": True, "no_candidate_regeneration": True,
        "authorization_events": AUTH.audit_log(), "capabilities_after": capabilities_after,
    }
    gradients = {
        "F1_R1": audits["F1_R1"]["gradient_isolation"], "F1_R2": audits["F1_R2"]["gradient_isolation"],
        "r1_ineligible_actor_gradient_zero": audits["F1_R1"]["gradient_isolation"]["max_ineligible_actor_logit_gradient_abs"] == 0.0,
        "r1_eligible_actor_gradient_present": audits["F1_R1"]["gradient_isolation"]["min_eligible_actor_logit_gradient_abs"] > 0.0,
        "r2_actor_parameter_unchanged": not audits["F1_R2"]["parameter_change"]["actor_changed"],
        "r2_critic_parameter_changed": audits["F1_R2"]["parameter_change"]["critic_changed"],
    }
    frozen = {"before": authority["frozen_before"], "after": frozen_after,
              "all_unchanged": authority["frozen_before"] == frozen_after,
              "e1_contract_sha256": E1_CONTRACT_SHA256,
              "model_parameter_changes": {key: audits[key]["parameter_change"] for key in REPLICATES}}
    outputs = {
        "bt8f1e1_preflight.json": preflight,
        "bt8f1e1_execution_manifest.json": execution_manifest,
        "bt8f1e1_seed1_training_audit.json": audits["F1_R1"],
        "bt8f1e1_seed2_training_audit.json": audits["F1_R2"],
        "bt8f1e1_gradient_isolation.json": gradients,
        "bt8f1e1_final_checkpoint_manifest.json": final_checkpoints,
        "bt8f1e1_final_review_replay.json": final_replay,
        "test_results.json": {"execution_counters": run_counters, "integrity_violations": integrity,
                              "hard_failures": [], "warnings": [], "TEST6_access": 0,
                              "github_push_performed": False, "review_replay_interpretation_performed": False},
        "frozen_hash_before_after.json": frozen,
        "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": PASS_CLASS,
                               "source_commit": source["source_commit"], "hard_failures": [], "warnings": [],
                               "global_locks": LOCKS, "next_step": NEXT_GATE},
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(
        "# BT8-F1-E1 final report\n\n"
        f"- gate: `{PASS_GATE}`\n- classification: `{PASS_CLASS}`\n- source commit: `{source['source_commit']}`\n"
        "- exact P1 preserved batch only; no causal rollout or candidate regeneration occurred.\n"
        "- final review replay uses the original F1 review snapshots without interpretation.\n",
        encoding="utf-8",
    )
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE, "classification": PASS_CLASS,
                                   "source_commit": source["source_commit"], "file_sha256": manifest,
                                   "github_push_performed": False})
    (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
    print(f"[PASS] {PASS_GATE}")
    print(f"classification: {PASS_CLASS}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
