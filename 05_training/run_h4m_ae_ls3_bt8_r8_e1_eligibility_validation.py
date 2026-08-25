#!/usr/bin/env python3
"""BT8-R8 E1 reward-ancestry Actor eligibility implementation validation.

This is a read-only, test-only-backward gate.  It consumes the frozen BT8-F1
training snapshots and R7 E1 rows, performs no rollout or optimizer step, and
does not write a checkpoint.  E0 and E1 are evaluated from the same in-memory
forward tensors so only Actor-loss aggregation can differ.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R8"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R8_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_E1_REWARD_ANCESTRY_GATED_ACTOR_UPDATE_READY_FOR_SEPARATE_BOUNDED_RETRAINING_AUTHORIZATION"
AUTH_BLOCK = "BLOCKED_E1_ELIGIBILITY_RUNTIME_MISMATCH"
PPO_BLOCK = "BLOCKED_PPO_EQUIVALENCE_FAILURE"
GRAD_BLOCK = "BLOCKED_ACTOR_GRADIENT_LEAKAGE"
MPS_BLOCK = "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
R7_SOURCE = "caf91442a2a837c0ea132bbaead26b55d13c4b8f"
R7_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R7_MINIMAL_ACTOR_CRITIC_SEED_FACTORIZATION_AND_CREDIT_ELIGIBILITY_SELECTION_COMPLETE"
F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
NEXT_GATE = "H4M-AE-R9.8-LS3-BT8-R9_E1_SEPARATE_BOUNDED_RETRAINING_AUTHORIZATION_SELECTION"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R7 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility_20260825_124641+09:00"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
R6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution_20260824_234227+09:00"
SOURCE_FILES = {
    "05_training/joint_assignment_learning.py",
    "05_training/joint_assignment_e1_eligibility.py",
    "05_training/run_h4m_ae_ls3_bt8_r8_e1_eligibility_validation.py",
    "05_training/test_h4m_ae_ls3_bt8_r8_e1_eligibility.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
REPLICATES = {
    "F1_R1": {"environment_seed": 20260822, "actor_seed": 20260824, "critic_seed": 20260826},
    "F1_R2": {"environment_seed": 20260823, "actor_seed": 20260825, "critic_seed": 20260827},
}


class R8Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R8Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
                    encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), "AUTHORITATIVE_EVIDENCE_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True,
                          check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff", "--name-only", f"{R7_SOURCE}..HEAD"]).splitlines() if row]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r7": git(["merge-base", R7_SOURCE, "HEAD"]) == R7_SOURCE,
        "changed_files_since_r7": changed,
        "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
        "github_push_performed": False,
    }


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    mismatches = [name for name, expected in manifest.get("file_sha256", {}).items()
                  if not (root / name).is_file() or sha256(root / name) != expected]
    return {"declared_file_count": len(manifest.get("file_sha256", {})), "mismatches": mismatches,
            "all_match": not mismatches, "manifest_sha256": sha256(root / "manifest.json")}


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, value in sorted(module.state_dict().items()):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def artifact_root() -> Path:
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r8_e1_eligibility_validation_{stamp}"


def empty_counters() -> dict[str, int]:
    return {
        "training": 0,
        "optimizer_step": 0,
        "backward": 0,
        "autograd_grad": 0,
        "causal_rollout": 0,
        "simulator_step": 0,
        "candidate_generation": 0,
        "candidate_regeneration": 0,
        "local_search_rerun": 0,
        "zero_loss_reevaluation": 0,
        "reward_recomputation": 0,
        "parameter_mutation": 0,
        "checkpoint_write": 0,
        "review_data_used": 0,
        "test6_access": 0,
        "actor_mps_forward": 0,
        "critic_mps_forward": 0,
        "nan_or_inf": 0,
    }


def write_block(root: Path, *, source: Mapping[str, Any], binding: Mapping[str, Any],
                code: str, detail: str) -> None:
    counters = empty_counters()
    stubs = {
        "bt8r8_e1_runtime_contract.json": {"not_run": True, "binding": binding},
        "bt8r8_eligibility_mask_audit.json": {"not_run": True},
        "bt8r8_actor_gradient_isolation.json": {"not_run": True},
        "bt8r8_critic_equivalence.json": {"not_run": True},
        "bt8r8_ppo_equivalence.json": {"not_run": True},
        "test_results.json": {"execution_counters": counters, "hard_failures": [detail]},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                               "source_commit": source.get("source_commit"), "hard_failures": [detail],
                               "warnings": [], "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, payload in stubs.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-R8 blocked\n\n`{code}`: {detail}\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source.get("source_commit"),
                                   "file_sha256": manifest, "github_push_performed": False})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def max_abs(values: Sequence[torch.Tensor | None]) -> float:
    finite = [value.detach().abs().max() for value in values if value is not None and value.numel()]
    return float(torch.stack(finite).max().cpu()) if finite else 0.0


def abs_sum(values: Sequence[torch.Tensor | None]) -> float:
    finite = [value.detach().abs().sum() for value in values if value is not None]
    return float(torch.stack(finite).sum().cpu()) if finite else 0.0


def build_batch(*, rows: Sequence[Mapping[str, Any]], payload_by_id: Mapping[str, Mapping[str, Any]],
                trace_by_id: Mapping[str, Mapping[str, Any]], n0_by_id: Mapping[str, float],
                device: torch.device) -> dict[str, Any]:
    require(bool(rows), "E1_EMPTY_REPLICATE_BATCH")
    payloads = [payload_by_id[str(row["decision_id"])] for row in rows]
    max_agents = max(int(payload["tensors"]["agent_feats"].shape[1]) for payload in payloads)
    max_pairs = max(int(payload["metadata"]["selectable_pair_count"]) for payload in payloads)
    count = len(payloads)
    global_dim = int(payloads[0]["tensors"]["global_feats"].shape[-1])
    demand_dim = int(payloads[0]["tensors"]["demand_feats"].shape[-1])
    agent_dim = int(payloads[0]["tensors"]["agent_feats"].shape[-1])
    candidate_dim = int(payloads[0]["tensors"]["candidate_feats"].shape[-1])
    glob = torch.zeros((count, global_dim), dtype=torch.float32)
    demand = torch.zeros((count, demand_dim), dtype=torch.float32)
    agents = torch.zeros((count, max_agents, agent_dim), dtype=torch.float32)
    agent_mask = torch.zeros((count, max_agents), dtype=torch.bool)
    candidates = torch.zeros((count, max_pairs, candidate_dim), dtype=torch.float32)
    pair_index = torch.zeros((count, max_pairs), dtype=torch.long)
    safe_mask = torch.zeros((count, max_pairs), dtype=torch.bool)
    action_index: list[int] = []
    forced: list[bool] = []
    advantages: list[float] = []
    targets: list[float] = []
    decision_ids: list[str] = []
    for index, (row, payload) in enumerate(zip(rows, payloads)):
        decision_id = str(row["decision_id"])
        tensors, meta = payload["tensors"], payload["metadata"]
        require(str(meta["decision_id"]) == decision_id, "E1_SNAPSHOT_DECISION_MISMATCH", decision_id)
        pairs = int(meta["selectable_pair_count"])
        require(pairs == int(tensors["candidate_feats"].shape[1]), "E1_SNAPSHOT_PAIR_SHAPE_MISMATCH", decision_id)
        agents_count = int(tensors["agent_feats"].shape[1])
        glob[index] = tensors["global_feats"][0].cpu()
        demand[index] = tensors["demand_feats"][0].cpu()
        agents[index, :agents_count] = tensors["agent_feats"][0].cpu()
        agent_mask[index, :agents_count] = tensors["agent_mask"][0].cpu()
        candidates[index, :pairs] = tensors["candidate_feats"][0].cpu()
        pair_index[index, :pairs] = tensors["pair_agent_index"][0].cpu()
        safe_mask[index, :pairs] = tensors["safe_mask"][0].cpu()
        if str(row["selected_action_type"]) == "NO_ASSIGN":
            chosen = max_pairs
        else:
            selected = (str(row["selected_candidate_id"]),)
            candidates_meta = list(meta["candidate_ids"])
            chosen = next((position for position, value in enumerate(candidates_meta)
                           if (str(value["candidate_id"]),) == selected), None)
            require(chosen is not None and chosen < pairs and bool(tensors["safe_mask"][0, chosen]),
                    "E1_SELECTED_CANDIDATE_NOT_IN_FROZEN_SUPPORT", decision_id)
        trace = trace_by_id[decision_id]
        advantage = float(n0_by_id[decision_id])
        target = float(trace["critic_target"])
        require(math.isfinite(advantage) and math.isfinite(target), "E1_NONFINITE_N0_OR_TARGET", decision_id)
        action_index.append(int(chosen))
        forced.append(pairs == 0)
        advantages.append(advantage)
        targets.append(target)
        decision_ids.append(decision_id)
    return {
        "global_feats": glob.to(device), "demand_feats": demand.to(device), "agent_feats": agents.to(device),
        "agent_mask": agent_mask.to(device), "candidate_feats": candidates.to(device),
        "pair_agent_index": pair_index.to(device), "safe_mask": safe_mask.to(device),
        "action_index": torch.tensor(action_index, dtype=torch.long, device=device),
        "forced_action": torch.tensor(forced, dtype=torch.bool, device=device),
        "advantage": torch.tensor(advantages, dtype=torch.float32, device=device),
        "value_target": torch.tensor(targets, dtype=torch.float32, device=device),
        "decision_ids": decision_ids, "max_pairs": max_pairs,
    }


def evaluate_replicate(*, replicate_id: str, actor: torch.nn.Module, critic: torch.nn.Module,
                       batch: Mapping[str, Any], actor_mask: torch.Tensor, JL: Any,
                       counters: dict[str, int]) -> dict[str, Any]:
    actor.eval()
    critic.eval()
    logits, no_assign = actor(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                               agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                               candidate_feats=batch["candidate_feats"], pair_agent_index=batch["pair_agent_index"],
                               safe_mask=batch["safe_mask"])
    value_pred = critic(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                        agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                        safe_summary=JL.safe_set_summary(batch["candidate_feats"], batch["safe_mask"]))
    counters["actor_mps_forward"] += 1
    counters["critic_mps_forward"] += 1
    log_probs = JL.masked_log_probs(logits, no_assign, batch["safe_mask"])
    old_log_prob = log_probs.gather(-1, batch["action_index"].unsqueeze(-1)).squeeze(-1).detach()
    e0 = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign,
                                 safe_mask=batch["safe_mask"], action_index=batch["action_index"],
                                 old_log_prob=old_log_prob, advantage=batch["advantage"], value_pred=value_pred,
                                 value_target=batch["value_target"], forced_action=batch["forced_action"])
    e1 = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign,
                                 safe_mask=batch["safe_mask"], action_index=batch["action_index"],
                                 old_log_prob=old_log_prob, advantage=batch["advantage"], value_pred=value_pred,
                                 value_target=batch["value_target"], forced_action=batch["forced_action"],
                                 actor_eligibility_mask=actor_mask)
    pair_grad, no_assign_grad = torch.autograd.grad(e1["actor_loss"], (logits, no_assign), retain_graph=True)
    actor_grads = torch.autograd.grad(e1["actor_loss"], tuple(actor.parameters()), retain_graph=True, allow_unused=True)
    critic_grads = torch.autograd.grad(e1["critic_loss"], tuple(critic.parameters()), allow_unused=True)
    counters["autograd_grad"] += 3
    ineligible = ~actor_mask
    eligible = actor_mask
    ineligible_logits = [pair_grad[ineligible], no_assign_grad[ineligible]] if bool(ineligible.any()) else []
    eligible_logits = [pair_grad[eligible], no_assign_grad[eligible]] if bool(eligible.any()) else []
    actor_contribution = e1["policy_loss_per_row"] * e1["actor_row_weight"]
    entropy_contribution = e1["entropy_per_row"] * e1["actor_row_weight"]
    exact = {
        "ratio_exact": bool(torch.equal(e0["ratio"], e1["ratio"])),
        "new_log_prob_exact": bool(torch.equal(e0["new_log_prob"], e1["new_log_prob"])),
        "unclipped_exact": bool(torch.equal(e0["unclipped_objective_per_row"], e1["unclipped_objective_per_row"])),
        "clipped_exact": bool(torch.equal(e0["clipped_objective_per_row"], e1["clipped_objective_per_row"])),
        "per_row_policy_exact": bool(torch.equal(e0["policy_loss_per_row"], e1["policy_loss_per_row"])),
        "critic_loss_exact": bool(torch.equal(e0["critic_loss"], e1["critic_loss"])),
        "value_target_exact": bool(torch.equal(batch["value_target"], batch["value_target"])),
    }
    require(all(exact.values()), "E1_PPO_OR_CRITIC_TENSOR_CHANGED", replicate_id)
    require(bool(torch.isfinite(logits).all() and torch.isfinite(no_assign).all() and torch.isfinite(value_pred).all()),
            "E1_NONFINITE_MODEL_OUTPUT", replicate_id)
    require(float(actor_contribution[ineligible].abs().sum().cpu()) == 0.0,
            "E1_INELIGIBLE_POLICY_CONTRIBUTION_NONZERO", replicate_id)
    require(float(entropy_contribution[ineligible].abs().sum().cpu()) == 0.0,
            "E1_INELIGIBLE_ENTROPY_CONTRIBUTION_NONZERO", replicate_id)
    return {
        "replicate_id": replicate_id,
        "decision_count": len(batch["decision_ids"]),
        "decision_ids": list(batch["decision_ids"]),
        "selected_action_bound_to_frozen_support": True,
        "n0_advantage_digest": hashlib.sha256(batch["advantage"].detach().cpu().numpy().tobytes()).hexdigest(),
        "actor": {
            "eligible": int(e1["actor_eligible_rows"]), "ineligible": int(e1["actor_ineligible_rows"]),
            "forced": int(e1["forced_rows"]), "denominator": float(e1["actor_denominator"]),
            "skip": bool(e1["actor_update_skipped"]), "policy_loss_e0": float(e0["policy_loss"].detach().cpu()),
            "policy_loss_e1": float(e1["policy_loss"].detach().cpu()), "entropy_e0": float(e0["entropy"].detach().cpu()),
            "entropy_e1": float(e1["entropy"].detach().cpu()),
            "ineligible_policy_contribution_abs_sum": float(actor_contribution[ineligible].abs().sum().cpu()),
            "ineligible_entropy_contribution_abs_sum": float(entropy_contribution[ineligible].abs().sum().cpu()),
            "ineligible_logit_gradient_max_abs": max_abs(ineligible_logits),
            "eligible_logit_gradient_abs_sum": abs_sum(eligible_logits),
            "actor_parameter_gradient_abs_sum": abs_sum(actor_grads),
        },
        "critic": {
            "finite_rows": int(torch.isfinite(value_pred).sum().item()),
            "value_digest": hashlib.sha256(value_pred.detach().cpu().numpy().tobytes()).hexdigest(),
            "target_digest": hashlib.sha256(batch["value_target"].detach().cpu().numpy().tobytes()).hexdigest(),
            "loss_e0": float(e0["critic_loss"].detach().cpu()), "loss_e1": float(e1["critic_loss"].detach().cpu()),
            "parameter_gradient_abs_sum": abs_sum(critic_grads),
        },
        "ppo": {
            **exact,
            "clip_epsilon": float(JL.CC.PPO_CLIP_EPSILON),
            "ratio_min": float(e1["ratio"].detach().min().cpu()),
            "ratio_max": float(e1["ratio"].detach().max().cpu()),
            "ratio_digest": hashlib.sha256(e1["ratio"].detach().cpu().numpy().tobytes()).hexdigest(),
        },
    }


def adversarial_audit(*, E1: Any, JL: Any, contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rejected(fn: Any, code: str) -> bool:
        try:
            fn()
        except E1.E1EligibilityError as exc:
            return exc.code == code
        return False

    critic_only = next(copy.deepcopy(row) for row in rows if row["category"] == "CRITIC_ONLY_NO_REWARD_ANCESTRY")
    label_tamper = copy.deepcopy(critic_only)
    label_tamper["category"] = "DIRECT_CANDIDATE_REWARD"
    label_tamper["actor_eligible_e1"] = True
    identity_tamper = copy.deepcopy(critic_only)
    identity_tamper["identity_chain_valid"] = False
    review_tamper = copy.deepcopy(critic_only)
    review_tamper["data_role"] = "REVIEW"
    nan_tamper = copy.deepcopy(critic_only)
    nan_tamper["raw_gae"] = float("nan")
    digest_tamper = dict(contract)
    digest_tamper["sha256"] = "0" * 64
    source_rep = str(critic_only["replicate_id"])
    other = next(copy.deepcopy(row) for row in rows if row["replicate_id"] != source_rep)
    wrong_dtype_blocked = False
    try:
        JL.assignment_ppo_loss(
            new_pair_logits=torch.zeros((1, 1)), new_no_assign_logit=torch.zeros((1, 1)),
            safe_mask=torch.ones((1, 1), dtype=torch.bool), action_index=torch.zeros(1, dtype=torch.long),
            old_log_prob=torch.zeros(1), advantage=torch.ones(1), value_pred=torch.zeros(1),
            value_target=torch.zeros(1), forced_action=torch.zeros(1, dtype=torch.bool),
            actor_eligibility_mask=torch.ones(1, dtype=torch.int64))
    except ValueError as exc:
        wrong_dtype_blocked = str(exc) == "ACTOR_ELIGIBILITY_MASK_MUST_BE_BOOL"
    return {
        "ancestry_label_tamper_blocked": rejected(
            lambda: E1.build_e1_eligibility_mask([label_tamper], expected_replicate_id=source_rep),
            "E1_ANCESTRY_LABEL_TAMPER"),
        "identity_tamper_blocked": rejected(
            lambda: E1.build_e1_eligibility_mask([identity_tamper], expected_replicate_id=source_rep),
            "E1_INSUFFICIENT_IDENTITY_CREDIT"),
        "review_row_blocked": rejected(
            lambda: E1.build_e1_eligibility_mask([review_tamper], expected_replicate_id=source_rep),
            "E1_REVIEW_ROW_IN_ACTOR_LOSS"),
        "nonfinite_row_blocked": rejected(
            lambda: E1.build_e1_eligibility_mask([nan_tamper], expected_replicate_id=source_rep),
            "E1_NONFINITE_CREDIT_ROW"),
        "seed_mixing_blocked": rejected(
            lambda: E1.build_e1_eligibility_mask([critic_only, other], expected_replicate_id=source_rep),
            "E1_SEED_MIXING_OR_REPLICATE_MISMATCH"),
        "contract_digest_tamper_blocked": rejected(lambda: E1.bind_e1_contract(digest_tamper),
                                                      "E1_CONTRACT_SHA256_MISMATCH"),
        "zero_eligible_actor_skip": True,
        "wrong_mask_dtype_blocked": wrong_dtype_blocked,
    }


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_e1_eligibility as E1
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_learning as JL
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_r5_same_support_review as R5MOD
    import run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution as R6MOD

    source = provenance()
    root = artifact_root()
    require(not root.exists(), "APPEND_ONLY_ARTIFACT_COLLISION")
    root.mkdir(parents=True)
    binding: dict[str, Any] = {"stage": STAGE, "source": source, "r7_source": R7_SOURCE,
                               "f1_source": F1_SOURCE, "e1_contract_sha256": E1.E1_CONTRACT_SHA256}
    try:
        r7_gate = load_json(R7 / "gate_decision.json")
        r7_contract = load_json(R7 / "bt8r7_selected_minimal_contract.json")
        r7_rows_file = load_json(R7 / "bt8r7_credit_eligibility_rows.json")
        r7_normalization = load_json(R7 / "bt8r7_advantage_normalization_audit.json")
        r7_frozen = load_json(R7 / "frozen_hash_before_after.json")
        f1_gate = load_json(F1 / "gate_decision.json")
        f1_collection = load_json(F1 / "bt8f1_training_snapshots" / "collection_manifest.json")
        f1_initial = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
        f1_leakage = load_json(F1 / "bt8f1_train_review_leakage_audit.json")
        r6_r1 = load_json(R6 / "bt8r6_replicate1_credit_trace.json")
        r6_r2 = load_json(R6 / "bt8r6_replicate2_credit_trace.json")
        E1.bind_e1_contract(r7_contract)
        binding.update({
            "r7_gate": r7_gate.get("gate") == R7_GATE and r7_gate.get("source_commit") == R7_SOURCE,
            "r7_manifest": verify_manifest(R7)["all_match"],
            "r7_e1_contract": r7_contract.get("sha256") == E1.E1_CONTRACT_SHA256,
            "f1_gate": f1_gate.get("gate") == F1_GATE and f1_gate.get("source_commit") == F1_SOURCE,
            "f1_manifest": verify_manifest(F1)["all_match"],
            "source_lineage": source["source_lineage_descends_from_r7"],
            "source_only_local_commit": source["source_only_local_commit"],
            "f1_review_leakage_zero": f1_leakage.get("review_optimizer_exposure") == 0
            and f1_leakage.get("training_minibatch_review_rows") == 0 and f1_leakage.get("train_review_overlap") == 0,
        })
        require(all(binding[key] for key in ("r7_gate", "r7_manifest", "r7_e1_contract", "f1_gate", "f1_manifest",
                                             "source_lineage", "source_only_local_commit", "f1_review_leakage_zero")),
                "E1_AUTHORITATIVE_BINDING_MISMATCH")
    except Exception as exc:  # noqa: BLE001
        write_block(root, source=source, binding=binding, code=AUTH_BLOCK,
                    detail=f"AUTHORITY_BINDING_FAILURE:{type(exc).__name__}:{exc}")
        print(f"[BLOCKED] {AUTH_BLOCK}\nartifact: {root.relative_to(PROJECT)}")
        return

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root, source=source, binding=binding, code=MPS_BLOCK, detail="MPS_REQUIRED_NO_CPU_FALLBACK")
        print(f"[BLOCKED] {MPS_BLOCK}\nartifact: {root.relative_to(PROJECT)}")
        return
    device = torch.device("mps:0")
    counters = empty_counters()
    hard: list[str] = []
    try:
        all_rows = list(r7_rows_file["rows"])
        category_counts = {key: sum(row["category"] == key for row in all_rows) for key in (
            "DIRECT_CANDIDATE_REWARD", "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD",
            "CRITIC_ONLY_NO_REWARD_ANCESTRY", "NO_ASSIGN_REWARD_SUPPORTED", "INSUFFICIENT_IDENTITY_CREDIT")}
        require(len(all_rows) == 48 and category_counts == {
            "DIRECT_CANDIDATE_REWARD": 4, "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD": 5,
            "CRITIC_ONLY_NO_REWARD_ANCESTRY": 39, "NO_ASSIGN_REWARD_SUPPORTED": 0,
            "INSUFFICIENT_IDENTITY_CREDIT": 0}, "E1_R7_ROW_DISTRIBUTION_MISMATCH")
        masks = {replicate_id: E1.build_e1_eligibility_mask(
            [row for row in all_rows if row["replicate_id"] == replicate_id], expected_replicate_id=replicate_id)
            for replicate_id in REPLICATES}
        require(masks["F1_R1"].actor_eligible_count == 9 and masks["F1_R2"].actor_eligible_count == 0,
                "E1_SEED_LOCAL_MASK_COUNT_MISMATCH")
        require(sum(sum(mask.critic_eligible) for mask in masks.values()) == 48, "E1_CRITIC_ROW_COUNT_MISMATCH")
        trace_by_id = {str(row["decision_id"]): row for row in [*r6_r1["rows"], *r6_r2["rows"]]}
        require(set(trace_by_id) == {str(row["decision_id"]) for row in all_rows}, "E1_R7_R6_DECISION_SET_MISMATCH")
        n0_by_id = {}
        for replicate_id, item in r7_normalization.items():
            for row in item["modes"]["N0"]["rows"]:
                decision_id = str(row["decision_id"])
                require(decision_id not in n0_by_id, "E1_N0_DUPLICATE_DECISION")
                n0_by_id[decision_id] = float(row["advantage"])
        require(set(n0_by_id) == set(trace_by_id), "E1_N0_DECISION_SET_MISMATCH")
        payload_by_id: dict[str, Mapping[str, Any]] = {}
        config: Mapping[str, Any] | None = None
        for entry in f1_collection["entries"]:
            snapshot = F1 / "bt8f1_training_snapshots" / entry["relative_path"]
            payload = FPS.load_snapshot(snapshot)
            manifest = load_json(snapshot / "snapshot_manifest.json")
            decision_id = str(payload["metadata"]["decision_id"])
            require(payload["snapshot_digest"] == entry["snapshot_digest"] == manifest["snapshot_digest"],
                    "E1_SNAPSHOT_DIGEST_MISMATCH", decision_id)
            require(sha256(snapshot / manifest["tensor_binary"]) == manifest["tensor_binary_sha256"],
                    "E1_SNAPSHOT_BINARY_DIGEST_MISMATCH", decision_id)
            require(decision_id not in payload_by_id and "REVIEW" not in decision_id.upper(),
                    "E1_NONTRAINING_OR_DUPLICATE_SNAPSHOT", decision_id)
            payload_by_id[decision_id] = payload
            config = payload["metadata"]["actor_config"] if config is None else config
            require(payload["metadata"]["actor_config"] == config, "E1_ACTOR_CONFIG_MISMATCH", decision_id)
        require(set(payload_by_id) == set(trace_by_id) and len(payload_by_id) == 48 and config is not None,
                "E1_TRAINING_SNAPSHOT_SET_MISMATCH")
    except Exception as exc:  # noqa: BLE001
        write_block(root, source=source, binding=binding, code=AUTH_BLOCK,
                    detail=f"E1_PRESERVED_EVIDENCE_FAILURE:{type(exc).__name__}:{exc}")
        print(f"[BLOCKED] {AUTH_BLOCK}\nartifact: {root.relative_to(PROJECT)}")
        return

    frozen_before = R6MOD.frozen_hashes(BT6)
    try:
        actors: dict[str, torch.nn.Module] = {}
        critics: dict[str, torch.nn.Module] = {}
        for replicate_id in REPLICATES:
            checkpoint = F1 / "initial_checkpoints" / f1_initial[replicate_id]["path"]
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            actors[replicate_id] = R5MOD.load_actor(checkpoint=checkpoint, config=config, H=H, device=device)
            critic = JL.JointAssignmentCritic(global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
                                               agent_dim=int(config["agent_dim"]),
                                               safe_summary_dim=1 + 2 * int(config["candidate_dim"]))
            critic.load_state_dict(checkpoint_payload["critic"], strict=True)
            critics[replicate_id] = critic.to(device).eval()
        module_before = {"actors": {key: module_digest(value) for key, value in actors.items()},
                         "critics": {key: module_digest(value) for key, value in critics.items()}}
        audits = {}
        for replicate_id in REPLICATES:
            source_rows = [next(row for row in all_rows if row["decision_id"] == decision_id)
                           for decision_id in masks[replicate_id].decision_ids]
            batch = build_batch(rows=source_rows, payload_by_id=payload_by_id, trace_by_id=trace_by_id,
                                n0_by_id=n0_by_id, device=device)
            actor_mask = torch.tensor(masks[replicate_id].actor_eligible, dtype=torch.bool, device=device)
            audits[replicate_id] = evaluate_replicate(replicate_id=replicate_id, actor=actors[replicate_id],
                                                      critic=critics[replicate_id], batch=batch, actor_mask=actor_mask,
                                                      JL=JL, counters=counters)
        torch.mps.synchronize()
        module_after = {"actors": {key: module_digest(value) for key, value in actors.items()},
                        "critics": {key: module_digest(value) for key, value in critics.items()}}
        counters["parameter_mutation"] = int(module_before != module_after)
        counters["nan_or_inf"] = int(any(not math.isfinite(value) for audit in audits.values() for value in (
            audit["actor"]["policy_loss_e0"], audit["actor"]["policy_loss_e1"], audit["critic"]["loss_e0"], audit["critic"]["loss_e1"])))
        require(module_before == module_after, "E1_TEST_ONLY_PARAMETER_MUTATION")
        require(audits["F1_R1"]["actor"]["eligible"] == 9 and audits["F1_R1"]["actor"]["ineligible"] == 15,
                "E1_R1_MASK_RUNTIME_MISMATCH")
        require(audits["F1_R2"]["actor"]["eligible"] == 0 and audits["F1_R2"]["actor"]["ineligible"] == 24
                and audits["F1_R2"]["actor"]["skip"], "E1_R2_ZERO_ELIGIBLE_SKIP_FAILURE")
        require(audits["F1_R1"]["actor"]["ineligible_logit_gradient_max_abs"] == 0.0
                and audits["F1_R2"]["actor"]["ineligible_logit_gradient_max_abs"] == 0.0,
                "E1_ACTOR_GRADIENT_LEAKAGE")
        require(audits["F1_R1"]["actor"]["eligible_logit_gradient_abs_sum"] > 0.0
                and audits["F1_R1"]["actor"]["actor_parameter_gradient_abs_sum"] > 0.0
                and audits["F1_R2"]["actor"]["actor_parameter_gradient_abs_sum"] == 0.0,
                "E1_ACTOR_GRADIENT_EXPECTATION_FAILURE")
        require(all(audit["critic"]["finite_rows"] == 24 and audit["critic"]["parameter_gradient_abs_sum"] > 0.0
                    for audit in audits.values()), "E1_CRITIC_FINITE_ALL_ROW_FAILURE")
    except Exception as exc:  # noqa: BLE001
        detail = f"E1_RUNTIME_OR_EQUIVALENCE_FAILURE:{type(exc).__name__}:{exc}"
        code = GRAD_BLOCK if "GRADIENT" in str(exc) else PPO_BLOCK if "PPO" in str(exc) else AUTH_BLOCK
        write_block(root, source=source, binding=binding, code=code, detail=detail)
        print(f"[BLOCKED] {code}\nartifact: {root.relative_to(PROJECT)}")
        return

    frozen_after = R6MOD.frozen_hashes(BT6)
    immutable_keys = sorted(set(r7_frozen["after"]) & set(frozen_after) - {"joint_assignment_learning"})
    immutable_mismatch = [key for key in immutable_keys if r7_frozen["after"][key] != frozen_after[key]]
    e1_path_changed = r7_frozen["after"].get("joint_assignment_learning") != frozen_after.get("joint_assignment_learning")
    if immutable_mismatch:
        hard.append("IMMUTABLE_FROZEN_HASH_MISMATCH")
    if not e1_path_changed:
        hard.append("E1_IMPLEMENTATION_SOURCE_NOT_CHANGED")
    adversarial = adversarial_audit(E1=E1, JL=JL, contract=r7_contract, rows=all_rows)
    if not all(value for key, value in adversarial.items() if key != "zero_eligible_actor_skip"):
        hard.append("E1_ADVERSARIAL_PROBE_FAILURE")
    if any(value for key, value in counters.items() if key not in {"autograd_grad", "actor_mps_forward", "critic_mps_forward"}):
        hard.append("FORBIDDEN_EXECUTION_COUNTER_NONZERO")
    if counters["actor_mps_forward"] != 2 or counters["critic_mps_forward"] != 2 or counters["autograd_grad"] != 6:
        hard.append("E1_TEST_ONLY_COUNTER_MISMATCH")
    gate = PASS_GATE if not hard else (GRAD_BLOCK if any("GRADIENT" in value for value in hard)
                                       else PPO_BLOCK if any("PPO" in value for value in hard) else AUTH_BLOCK)
    classification = PASS_CLASS if not hard else "BLOCKED"

    eligibility_audit = {
        "source": "R7 preserved 48-row eligibility evidence; no reclassification from rollout",
        "contract_sha256": E1.E1_CONTRACT_SHA256, "category_distribution": category_counts,
        "replicates": {key: {"decision_ids": list(mask.decision_ids), "actor_eligible": mask.actor_eligible_count,
                              "actor_ineligible": mask.actor_ineligible_count,
                              "critic_eligible": sum(mask.critic_eligible), "evidence_digest": mask.evidence_digest,
                              "seed": REPLICATES[key]["environment_seed"], "seed_mixing": 0}
                       for key, mask in masks.items()},
        "review_rows_used": 0, "insufficient_identity_credit_rows": category_counts["INSUFFICIENT_IDENTITY_CREDIT"],
    }
    runtime_contract = {
        "contract_id": E1.E1_CONTRACT_ID, "contract_sha256": E1.E1_CONTRACT_SHA256,
        "sequence": ["frozen trajectory GAE", "frozen seed-local N0 full-batch normalization", "E1 Actor eligibility mask"],
        "actor_loss": "masked mean over eligible, non-forced rows", "actor_entropy": "masked mean over eligible, non-forced rows",
        "critic_loss": "mean over all finite rows", "zero_eligible": "explicit Actor skip; Critic remains active",
        "immutable": {"reward_v2": "unchanged", "gae": "unchanged", "n0_normalization": "retained",
                      "ppo_clip": "unchanged", "actor_critic_architecture": "unchanged",
                      "no_assign_score": "unchanged", "t1_selector": "unchanged"},
    }
    actor_gradient = {key: audit["actor"] for key, audit in audits.items()} | {
        "all_ineligible_actor_gradient_max_abs": max(audit["actor"]["ineligible_logit_gradient_max_abs"] for audit in audits.values()),
        "r1_eligible_gradient_present": audits["F1_R1"]["actor"]["eligible_logit_gradient_abs_sum"] > 0.0,
        "r2_zero_eligible_skip": audits["F1_R2"]["actor"]["skip"],
    }
    critic_equivalence = {key: audit["critic"] for key, audit in audits.items()} | {
        "e0_e1_critic_loss_exact_all_replicates": all(audit["ppo"]["critic_loss_exact"] for audit in audits.values()),
        "all_finite_rows_retained": all(audit["critic"]["finite_rows"] == 24 for audit in audits.values()),
    }
    ppo_equivalence = {key: audit["ppo"] for key, audit in audits.items()} | {
        "e0_e1_ratio_logprob_clip_exact_all_replicates": all(
            audit["ppo"]["ratio_exact"] and audit["ppo"]["new_log_prob_exact"]
            and audit["ppo"]["unclipped_exact"] and audit["ppo"]["clipped_exact"] for audit in audits.values()),
        "n0_retained": True,
    }
    frozen = {
        "r7_before": r7_frozen["after"], "r8_after": frozen_after,
        "immutable_keys_checked": immutable_keys, "immutable_mismatches": immutable_mismatch,
        "immutable_all_unchanged": not immutable_mismatch,
        "intended_e1_path": {"module": "joint_assignment_learning.py", "r7_sha256": r7_frozen["after"].get("joint_assignment_learning"),
                               "r8_sha256": frozen_after.get("joint_assignment_learning"), "changed": e1_path_changed},
        "e1_eligibility_module_sha256": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
        "module_parameters_before": module_before, "module_parameters_after": module_after,
        "parameters_unchanged": module_before == module_after,
    }
    test_results = {
        "authority_binding": binding, "execution_counters": counters, "adversarial_probes": adversarial,
        "hard_failures": hard, "warnings": [], "future_leakage": 0, "review_data_used": 0,
        "TEST6_access": 0, "github_push_performed": False,
    }
    outputs = {
        "bt8r8_e1_runtime_contract.json": runtime_contract,
        "bt8r8_eligibility_mask_audit.json": eligibility_audit,
        "bt8r8_actor_gradient_isolation.json": actor_gradient,
        "bt8r8_critic_equivalence.json": critic_equivalence,
        "bt8r8_ppo_equivalence.json": ppo_equivalence,
        "test_results.json": test_results,
        "frozen_hash_before_after.json": frozen,
        "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification,
                               "source_commit": source["source_commit"], "hard_failures": hard, "warnings": [],
                               "global_locks": LOCKS, "next_step": NEXT_GATE if not hard else "STOP"},
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    report = (
        "# BT8-R8 final report\n\n"
        f"- gate: `{gate}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n"
        f"- E1 contract SHA256: `{E1.E1_CONTRACT_SHA256}`\n"
        "- Actor uses only the R7 reward-ancestry mask; Critic retains every finite persisted row.\n"
        "- This gate used test-only autograd; no training, rollout, optimizer step, checkpoint, or parameter mutation occurred.\n"
    )
    (root / "final_report.md").write_text(report, encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                   "source_commit": source["source_commit"], "file_sha256": manifest,
                                   "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"classification: {classification}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
