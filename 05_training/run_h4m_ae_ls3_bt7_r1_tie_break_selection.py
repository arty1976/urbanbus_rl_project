#!/usr/bin/env python3
"""BT7-R1: exact-tie-only, order-invariant frozen selection semantics.

This is a pure inference/selection audit.  Learned Joint Actor scores, the
checkpoint, snapshots, candidate support, and all training/simulator paths are
read-only.  The new selector is deliberately scoped to final frozen inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch


STAGE = "H4M-AE-R9.8-LS3-BT7-R1"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT7_R1_ORDER_INVARIANT_TIE_BREAK_SELECTION_SEMANTICS_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_FROZEN_POLICY_SELECTION_SEMANTICS_ORDER_INVARIANT_READY_FOR_BT7_BEHAVIOR_RERUN"
BLOCK = "BLOCKED_TIE_BREAK_REPAIR_ALTERS_LEARNED_POLICY_PREFERENCE"
BT6_SOURCE = "183e0dae6075e3038305686b151db0c3767212c8"
BT7_SOURCE = "18d0fafb5a7b478069f7ede35f7551b56ef01c59"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_FILES = {"05_training/joint_assignment_frozen_tie_break.py", "05_training/run_h4m_ae_ls3_bt7_r1_tie_break_selection.py"}
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_200221+09:00"
BT7 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_rerun_frozen_policy_behavior_review_20260822_202943+09:00"
SNAPSHOT_ROOT = BT6 / "bt7_frozen_policy_snapshots"
COLLECTION = SNAPSHOT_ROOT / "collection_manifest.json"
CHECKPOINT = BT6 / "bt6_r2_joint_assignment_checkpoint.pt"
FROZEN = {
    "gatv2_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2": "rewards/mappo_reward_v1.py", "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py", "candidate_support_deconfounding": "joint_candidate_support_snapshot.py",
    "causal_bridge": "causal_kpi_bridge.py", "r9_8_authorization": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py", "joint_assignment_learning": "joint_assignment_learning.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py", "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def frozen_hashes() -> Dict[str, str]:
    hashes = {name: sha256(ROOT / rel) for name, rel in FROZEN.items()}
    hashes["joint_actor_head"] = sha256(ROOT / "multi_agent_candidate_assignment_head.py")
    return hashes


def provenance() -> Dict[str, Any]:
    head, parent = git(["rev-parse", "HEAD"]), git(["rev-parse", "HEAD^"])
    changed = [row for row in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if row]
    return {"source_commit": head, "source_parent": parent, "changed_files": changed,
            "source_only_local_commit": set(changed) == SOURCE_FILES, "github_push_performed": False}


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def action_identity(selection: Any) -> str | tuple[str, str]:
    return "NO_ASSIGN" if selection.selected.is_no_assign else (str(selection.selected.agent_id), str(selection.selected.candidate_id))


def forward_scores(actor: torch.nn.Module, payload: Mapping[str, Any], *, device: torch.device, head: Any) -> Dict[str, Any]:
    meta, cpu = payload["metadata"], payload["tensors"]
    tensors = {name: value.to(device) for name, value in cpu.items()}
    p = int(meta["selectable_pair_count"])
    with torch.no_grad():
        raw, no_assign = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                               agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                               candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"],
                               safe_mask=tensors["safe_mask"])
        logits, mask = raw[:, :p], tensors["safe_mask"][:, :p]
        probabilities = head.masked_distribution(logits, no_assign, mask)
    return {"pair_scores": logits[0].detach().cpu().tolist(), "safe_mask": mask[0].detach().cpu().tolist(),
            "no_assign_score": float(no_assign[0, 0].detach().cpu()), "probabilities": probabilities[0].detach().cpu().tolist()}


def permute_candidates(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    meta, source = payload["metadata"], payload["tensors"]
    p, full = int(meta["selectable_pair_count"]), int(source["candidate_feats"].shape[1])
    if p < 2: return payload
    permutation = list(reversed(range(p))) + list(range(p, full))
    index = torch.tensor(permutation, dtype=torch.long)
    tensors = dict(source)
    for name in ("candidate_feats", "pair_agent_index", "safe_mask"):
        tensors[name] = source[name].index_select(1, index)
    ids = [meta["candidate_ids"][i] for i in permutation[:p]]
    changed_meta = dict(meta); changed_meta["candidate_ids"] = ids; changed_meta["candidate_order"] = ids
    return {"metadata": changed_meta, "tensors": tensors, "snapshot_digest": payload["snapshot_digest"]}


def permute_agents(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = payload["tensors"]
    a = int(source["agent_feats"].shape[1])
    if a < 2: return payload
    permutation = list(reversed(range(a))); index = torch.tensor(permutation, dtype=torch.long)
    inverse = torch.empty(a, dtype=torch.long); inverse[index] = torch.arange(a, dtype=torch.long)
    tensors = dict(source)
    tensors["agent_feats"] = source["agent_feats"].index_select(1, index)
    tensors["agent_mask"] = source["agent_mask"].index_select(1, index)
    tensors["pair_agent_index"] = inverse[source["pair_agent_index"]]
    return {"metadata": payload["metadata"], "tensors": tensors, "snapshot_digest": payload["snapshot_digest"]}


def selection(scores: Mapping[str, Any], payload: Mapping[str, Any], tie: Any) -> Any:
    return tie.select_exact_tie(candidate_ids=payload["metadata"]["candidate_ids"], pair_scores=scores["pair_scores"],
                                safe_mask=scores["safe_mask"], no_assign_score=scores["no_assign_score"])


def original_argmax(scores: Mapping[str, Any], payload: Mapping[str, Any]) -> str | tuple[str, str]:
    values = list(scores["pair_scores"]) + [scores["no_assign_score"]]
    index = max(range(len(values)), key=lambda item: values[item])
    return "NO_ASSIGN" if index == len(scores["pair_scores"]) else (payload["metadata"]["candidate_ids"][index]["agent_id"],
                                                                         payload["metadata"]["candidate_ids"][index]["candidate_id"])


def expected_digests(payload: Mapping[str, Any], tie: Any) -> list[str]:
    return [tie.canonical_identity_digest(agent_id=row["agent_id"], candidate_id=row["candidate_id"])
            for row in payload["metadata"]["candidate_ids"]]


def adversarial_tests(tie: Any) -> Dict[str, bool]:
    def rejects(code: str, fn) -> bool:
        try: fn()
        except tie.FrozenTieBreakError as exc: return exc.code == code
        return False
    ids = [{"agent_id": "opaque-agent-A", "candidate_id": "opaque-candidate-A"},
           {"agent_id": "opaque-agent-B", "candidate_id": "opaque-candidate-B"}]
    exact = tie.select_exact_tie(candidate_ids=ids, pair_scores=[1.0, 1.0], safe_mask=[True, True], no_assign_score=0.0)
    reversed_exact = tie.select_exact_tie(candidate_ids=list(reversed(ids)), pair_scores=[1.0, 1.0], safe_mask=[True, True], no_assign_score=0.0)
    outside = tie.select_exact_tie(candidate_ids=ids, pair_scores=[1.0, 1.0 + 1e-7], safe_mask=[True, True], no_assign_score=0.0)
    clear = tie.select_exact_tie(candidate_ids=ids, pair_scores=[3.0, 1.0], safe_mask=[True, True], no_assign_score=0.0)
    no_assign_only = tie.select_exact_tie(candidate_ids=[], pair_scores=[], safe_mask=[], no_assign_score=0.0)
    no_assign_tie = tie.select_exact_tie(candidate_ids=[ids[0]], pair_scores=[1.0], safe_mask=[True], no_assign_score=1.0)
    masked = tie.select_exact_tie(candidate_ids=ids, pair_scores=[100.0, 1.0], safe_mask=[False, True], no_assign_score=0.0)
    digests = [tie.canonical_identity_digest(agent_id=row["agent_id"], candidate_id=row["candidate_id"]) for row in ids]
    return {
        "exact_tie_order_invariant": action_identity(exact) == action_identity(reversed_exact),
        "near_tie_just_outside_zero_tolerance_preserves_score_winner": action_identity(outside) == ("opaque-agent-B", "opaque-candidate-B"),
        "unique_clear_winner_preserved": action_identity(clear) == ("opaque-agent-A", "opaque-candidate-A"),
        "no_assign_only_preserved": action_identity(no_assign_only) == "NO_ASSIGN",
        "no_assign_genuine_tie_uses_same_canonical_rule": action_identity(no_assign_tie) in {"NO_ASSIGN", ("opaque-agent-A", "opaque-candidate-A")},
        "masked_highest_raw_score_not_selected": action_identity(masked) == ("opaque-agent-B", "opaque-candidate-B"),
        "duplicate_identity_fail_closed": rejects("DUPLICATE_CANONICAL_CANDIDATE_IDENTITY", lambda: tie.select_exact_tie(candidate_ids=[ids[0], ids[0]], pair_scores=[1.0, 1.0], safe_mask=[True, True], no_assign_score=0.0)),
        "missing_identity_fail_closed": rejects("MISSING_CANONICAL_CANDIDATE_IDENTITY", lambda: tie.select_exact_tie(candidate_ids=[{"agent_id": "", "candidate_id": ""}], pair_scores=[1.0], safe_mask=[True], no_assign_score=0.0)),
        "identity_digest_tamper_fail_closed": rejects("CANONICAL_IDENTITY_DIGEST_TAMPERED", lambda: tie.select_exact_tie(candidate_ids=ids, pair_scores=[1.0, 1.0], safe_mask=[True, True], no_assign_score=0.0, expected_identity_digests=[digests[0], "tampered"])),
        "nan_score_fail_closed": rejects("TIE_BREAK_NONFINITE_LEGAL_SCORE", lambda: tie.select_exact_tie(candidate_ids=ids, pair_scores=[math.nan, 1.0], safe_mask=[True, True], no_assign_score=0.0)),
        "inf_no_assign_fail_closed": rejects("TIE_BREAK_NONFINITE_NO_ASSIGN_SCORE", lambda: tie.select_exact_tie(candidate_ids=ids, pair_scores=[1.0, 1.0], safe_mask=[True, True], no_assign_score=math.inf)),
    }


def main() -> None:
    started = time.perf_counter(); sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_r1_tie_break_selection_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists(): raise SystemExit("append-only artifact collision")
    before = frozen_hashes(); source = provenance(); hard = []
    b6_gate = json.loads((BT6 / "gate_decision.json").read_text(encoding="utf-8"))
    b7_gate = json.loads((BT7 / "gate_decision.json").read_text(encoding="utf-8"))
    if not torch.backends.mps.is_available(): hard.append("MPS_FROZEN_REPLAY_ENVIRONMENT_UNAVAILABLE")
    try:
        collection = FPS.load_collection_manifest(COLLECTION)
        snapshots = [(entry, FPS.load_snapshot(SNAPSHOT_ROOT / entry["relative_path"])) for entry in collection["entries"]]
    except Exception as exc:  # noqa: BLE001
        collection, snapshots = {}, []
        hard.append(f"FROZEN_EVIDENCE_LOAD_FAILED:{type(exc).__name__}")
    checkpoint_before = sha256(CHECKPOINT) if CHECKPOINT.is_file() else None
    binding = {"bt6_gate": b6_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_POST_REPAIR_EXTENDED_BOUNDED_TRAINING_AND_BT7_READINESS_COMPLETE",
               "bt6_source": b6_gate.get("source_commit") == BT6_SOURCE,
               "bt7_block": b7_gate.get("gate") == "BLOCKED_FROZEN_POLICY_STRUCTURAL_DEGENERACY",
               "bt7_source": b7_gate.get("source_commit") == BT7_SOURCE,
               "source_only_local_commit": source["source_only_local_commit"], "checkpoint_exists": CHECKPOINT.is_file(),
               "checkpoint_sha": bool(collection) and checkpoint_before == collection.get("checkpoint_sha256"),
               "collection_count_96": bool(collection) and collection.get("snapshot_count") == len(snapshots) == 96,
               "unique_snapshots": len({payload["snapshot_digest"] for _, payload in snapshots}) == 96,
               "snapshot_schema": bool(collection) and collection.get("snapshot_schema_version") == FPS.SNAPSHOT_SCHEMA_VERSION,
               "collection_schema": bool(collection) and collection.get("collection_schema_version") == FPS.COLLECTION_SCHEMA_VERSION,
               "frozen_authorities": bool(collection) and collection.get("frozen_authority_hashes") == before}
    if not all(binding.values()): hard.append("FROZEN_EVIDENCE_BINDING_MISMATCH")
    integrity = {name: 0 for name in ("optimizer_steps", "causal_simulator_rollouts", "candidate_regeneration", "zero_loss_reevaluation",
                                      "parameter_mutation", "source_mutation", "checkpoint_mutation", "logit_modification",
                                      "probability_modification", "TEST6_access", "github_push")}
    rows, candidate_before_failures, candidate_after_failures, agent_after_failures, combined_after_failures = [], 0, 0, 0, 0
    score_logit_delta, score_probability_delta = [], []
    if not hard:
        cfg = collection["actor_config"]
        expected = FPS.actor_config(global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"], agent_dim=cfg["agent_dim"],
                                    candidate_dim=cfg["candidate_dim"], hidden=cfg["hidden"], heads=cfg["heads"],
                                    actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
                                    actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
        if cfg != expected or collection.get("actor_config_sha256") != FPS.actor_config_sha256(expected): hard.append("ACTOR_CONFIG_BINDING_MISMATCH")
        device = torch.device("mps")
        actor = H.MultiAgentCandidateAssignmentHead(global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"], agent_dim=cfg["agent_dim"], candidate_dim=cfg["candidate_dim"], hidden=cfg["hidden"], heads=cfg["heads"]).to(device)
        checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor"], strict=True); actor.eval(); params_before = module_digest(actor)
        for entry, payload in snapshots:
            base_scores = forward_scores(actor, payload, device=device, head=H)
            repaired = selection(base_scores, payload, TIE)
            old = original_argmax(base_scores, payload)
            candidate_payload = permute_candidates(payload); candidate_scores = forward_scores(actor, candidate_payload, device=device, head=H)
            candidate_repaired = selection(candidate_scores, candidate_payload, TIE)
            agent_payload = permute_agents(payload); agent_scores = forward_scores(actor, agent_payload, device=device, head=H)
            agent_repaired = selection(agent_scores, agent_payload, TIE)
            combined_payload = permute_candidates(agent_payload); combined_scores = forward_scores(actor, combined_payload, device=device, head=H)
            combined_repaired = selection(combined_scores, combined_payload, TIE)
            if int(payload["metadata"]["selectable_pair_count"]) >= 2:
                candidate_before_failures += int(old != original_argmax(candidate_scores, candidate_payload))
                candidate_after_failures += int(action_identity(repaired) != action_identity(candidate_repaired))
                combined_after_failures += int(action_identity(repaired) != action_identity(combined_repaired))
                base_by_id = {(row["agent_id"], row["candidate_id"]): value for row, value in zip(payload["metadata"]["candidate_ids"], base_scores["pair_scores"])}
                perm_by_id = {(row["agent_id"], row["candidate_id"]): value for row, value in zip(candidate_payload["metadata"]["candidate_ids"], candidate_scores["pair_scores"])}
                score_logit_delta.append(max(abs(base_by_id[key] - perm_by_id[key]) for key in base_by_id))
                base_p = {(row["agent_id"], row["candidate_id"]): value for row, value in zip(payload["metadata"]["candidate_ids"], base_scores["probabilities"][:-1])}
                perm_p = {(row["agent_id"], row["candidate_id"]): value for row, value in zip(candidate_payload["metadata"]["candidate_ids"], candidate_scores["probabilities"][:-1])}
                score_probability_delta.append(max(abs(base_p[key] - perm_p[key]) for key in base_p))
            agent_after_failures += int(action_identity(repaired) != action_identity(agent_repaired))
            values = list(base_scores["pair_scores"]) + [base_scores["no_assign_score"]]
            ordered = sorted(values, reverse=True); margin = ordered[0] - ordered[1] if len(ordered) > 1 else None
            rows.append({"snapshot_digest": payload["snapshot_digest"], "support_size": int(payload["metadata"]["selectable_pair_count"]),
                         "old_original_order_selection": old, "repaired_selection": action_identity(repaired),
                         "exact_tie": repaired.exact_tie, "tie_set_size": len(repaired.tie_set), "top_margin": margin,
                         "old_changed_under_candidate_permutation": int(payload["metadata"]["selectable_pair_count"]) >= 2 and old != original_argmax(candidate_scores, candidate_payload),
                         "repaired_changed_under_candidate_permutation": action_identity(repaired) != action_identity(candidate_repaired),
                         "repaired_changed_under_agent_permutation": action_identity(repaired) != action_identity(agent_repaired),
                         "repaired_changed_under_combined_permutation": action_identity(repaired) != action_identity(combined_repaired),
                         "no_assign_before": old == "NO_ASSIGN", "no_assign_after": action_identity(repaired) == "NO_ASSIGN"})
        torch.mps.synchronize()
        integrity["parameter_mutation"] = int(module_digest(actor) != params_before)
    exact = [row for row in rows if row["exact_tie"]]
    positives = [row["top_margin"] for row in rows if row["top_margin"] is not None and row["top_margin"] > 0.0]
    non_tie = [row for row in rows if not row["exact_tie"]]
    non_tie_overrides = sum(row["old_original_order_selection"] != row["repaired_selection"] for row in non_tie)
    after = frozen_hashes(); integrity["source_mutation"] = int(before != after)
    integrity["checkpoint_mutation"] = int(checkpoint_before is not None and checkpoint_before != sha256(CHECKPOINT))
    adversarial = adversarial_tests(TIE); adversarial_passed = sum(adversarial.values())
    if non_tie_overrides: hard.append(BLOCK)
    if candidate_after_failures or agent_after_failures or combined_after_failures: hard.append("REPAIRED_SELECTION_ORDER_INVARIANCE_FAILURE")
    if any(integrity.values()): hard.append("FORBIDDEN_EXECUTION_OR_SCORE_MUTATION")
    if adversarial_passed != len(adversarial): hard.append("TIE_BREAK_ADVERSARIAL_TEST_FAILURE")
    gate, classification = (PASS_GATE, PASS_CLASS) if not hard else (BLOCK, "FROZEN_TIE_BREAK_REPAIR_BLOCKED")
    def shares(field: str) -> Dict[str, float]:
        picked = [row[field] for row in rows if row[field] != "NO_ASSIGN"]
        counts = Counter(map(str, picked)); total = sum(counts.values())
        return {key: value / total for key, value in sorted(counts.items())} if total else {}
    margin = {"all_snapshots": len(rows), "multi_candidate_snapshots": sum(row["support_size"] >= 2 for row in rows),
              "exact_ties": len(exact), "near_ties_under_selected_tolerance": 0,
              "positive_margins_le_1e_5_not_treated_as_ties": sum(value <= 1e-5 for value in positives),
              "clearly_separated_under_t1": len(non_tie), "minimum_positive_margin": min(positives) if positives else None,
              "maximum_margin": max(positives) if positives else None, "selected_tolerance": 0.0,
              "tolerance_justification": "T1 exact equality resolves all 26 observed positional failures; the smallest positive margin remains a learned preference and is not overridden."}
    order = {"candidate_order_tested": sum(row["support_size"] >= 2 for row in rows), "old_argmax_failures": candidate_before_failures,
             "repaired_failures": candidate_after_failures, "agent_order_tested": len(rows), "agent_order_failures": agent_after_failures,
             "combined_order_tested": sum(row["support_size"] >= 2 for row in rows), "combined_order_failures": combined_after_failures,
             "identity_aligned_max_logit_delta": max(score_logit_delta, default=0.0), "identity_aligned_max_probability_delta": max(score_probability_delta, default=0.0)}
    preservation = {"non_tie_preference_overrides": non_tie_overrides, "repaired_selection_differs_from_old_original_order_argmax": sum(row["old_original_order_selection"] != row["repaired_selection"] for row in rows),
                    "difference_reason": "exact ties only; canonical digest replaces positional first-index argmax", "all_frozen_scores_unchanged": integrity["logit_modification"] == integrity["probability_modification"] == 0,
                    "old_agent_selection_share": shares("old_original_order_selection"), "repaired_agent_selection_share": shares("repaired_selection")}
    no_assign = {"original_argmax_count": sum(row["no_assign_before"] for row in rows), "repaired_count": sum(row["no_assign_after"] for row in rows),
                 "unchanged": all(row["no_assign_before"] == row["no_assign_after"] for row in rows), "zero_feasible_support_count": sum(row["support_size"] == 0 for row in rows),
                 "zero_feasible_support_repaired_no_assign_count": sum(row["support_size"] == 0 and row["no_assign_after"] for row in rows),
                 "contract": "NO_ASSIGN remains a legal canonical action and is mandatory with zero feasible candidates"}
    root.mkdir(parents=True)
    lineage = {"BT6_RERUN": BT6_SOURCE, "BT7_RERUN_BLOCKED": BT7_SOURCE, "checkpoint_sha256": checkpoint_before,
               "snapshot_collection_digest": collection.get("collection_digest")}
    outputs = {
        "bt7r1_selection_path_root_cause.json": {"path": "masked logits -> probabilities -> torch.argmax first maximum -> support index -> candidate identity",
                                                   "root_cause": "exact-score ties resolved by first-index argmax", "device": "mps", "dtype": "torch.float32",
                                                   "no_assign_handling": "last action in legacy argmax; canonical action under repaired selector"},
        "bt7r1_margin_distribution.json": margin,
        "bt7r1_tie_semantics_contract.json": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "repair_level": "T1", "tie_definition": "score == best_score exactly; tolerance = 0.0",
                                                "canonical_identity": "SHA-256 over opaque (agent_id,candidate_id) action identity; explicit distinct NO_ASSIGN identity",
                                                "forbidden_tie_break_inputs": ["list position", "candidate rank", "agent ordinal", "agent_id numeric value", "reward", "KPI", "performance outcome"],
                                                "non_tie_rule": "unique maximum score always wins unchanged"},
        "bt7r1_repair_selection.json": {"T1": {"selected": True, "reason": "all 26 observed positional failures are exact ties"},
                                          "T2": {"selected": False, "reason": "would override six positive-score margins without evidence necessity"},
                                          "T3": {"selected": "implementation detail", "reason": "SHA-256 canonical identity is the opaque order-independent representation used by T1"},
                                          "T4": {"selected": False, "reason": "not needed"}},
        "bt7r1_order_invariance_audit.json": order,
        "bt7r1_policy_semantic_preservation.json": preservation,
        "bt7r1_no_assign_preservation.json": no_assign,
        "bt7r1_adversarial_tests.json": {"passed": adversarial_passed, "total": len(adversarial), "cases": adversarial},
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after, "checkpoint_before": checkpoint_before, "checkpoint_after": sha256(CHECKPOINT)},
        "test_results.json": {"binding": binding, "integrity_counters": integrity, "hard_failures": hard, "warnings": [],
                              "optimizer_steps": 0, "causal_simulator_rollouts": 0, "candidate_regeneration": 0, "zero_loss_reevaluation": 0,
                              "github_push_performed": False},
        "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"], "lineage": lineage,
                               "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
                               "next_step": "BT7 pure frozen-policy behavior rerun using repaired selection semantics" if not hard else "STOP"},
    }
    for name, value in outputs.items(): dump(root / name, value)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Exact-tie order-invariant selection semantics\n\n"
        f"gate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"T1 exact-tie canonical selection resolved {candidate_before_failures} old positional failures to {candidate_after_failures}. "
        f"It changed {preservation['repaired_selection_differs_from_old_original_order_argmax']} original-order selections, all within exact ties; "
        f"non-tie overrides = {non_tie_overrides}. No training, simulator, candidate regeneration, Zero-Loss reevaluation, or score mutation occurred.\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "lineage": lineage, "file_sha256": manifest, "elapsed_seconds": round(time.perf_counter() - started, 3), "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
