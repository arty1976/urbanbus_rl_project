#!/usr/bin/env python3
"""BT7-RERUN2: pure frozen-policy review under T1 exact-tie semantics.

This executor reads one bound checkpoint and the 96 preserved actor-input
snapshots.  It never constructs candidates, invokes the simulator/Zero-Loss,
or changes a score, probability, parameter, checkpoint, or frozen authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch


STAGE = "H4M-AE-R9.8-LS3-BT7-RERUN2"
PASS_A = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT7_RERUN2_FROZEN_POLICY_BEHAVIOR_REVIEW_COMPLETE"
PASS_B = "PASS_WITH_FROZEN_POLICY_DISCRIMINATION_CAUTION"
BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT7_RERUN2_FROZEN_POLICY_INTEGRITY_FAILURE"
BT6_SOURCE = "183e0dae6075e3038305686b151db0c3767212c8"
BT7_R1_SOURCE = "8a4c645906bc3967e0d1d03fe2c7e2912289baf2"
T1_TOLERANCE = 0.0

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = "05_training/run_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review.py"
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_200221+09:00"
BT7_R1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_r1_tie_break_selection_20260822_205519+09:00"
SNAPSHOT_ROOT = BT6 / "bt7_frozen_policy_snapshots"
COLLECTION_PATH = SNAPSHOT_ROOT / "collection_manifest.json"
CHECKPOINT_PATH = BT6 / "bt6_r2_joint_assignment_checkpoint.pt"
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


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    head = git(["rev-parse", "HEAD"])
    changed = [item for item in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if item]
    return {"source_commit": head, "source_parent": git(["rev-parse", "HEAD^"]), "changed_files": changed,
            "source_only_local_commit": changed == [SOURCE_REL], "github_push_performed": False}


def frozen_hashes() -> dict[str, str]:
    result = {name: sha256(ROOT / rel) for name, rel in FROZEN.items()}
    result["joint_actor_head"] = sha256(ROOT / "multi_agent_candidate_assignment_head.py")
    return result


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8")); digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.mean(values)) if values else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {"min": ordered[0], "p25": at(.25), "median": at(.5), "p75": at(.75), "max": ordered[-1]}


def density_label(rank: int) -> str:
    return "low" if rank == 0 else "high" if rank == 3 else "medium"


def identity(action: Any) -> str | tuple[str, str]:
    return "NO_ASSIGN" if action.selected.is_no_assign else (str(action.selected.agent_id), str(action.selected.candidate_id))


def forward(actor: torch.nn.Module, payload: Mapping[str, Any], *, device: torch.device, head: Any, tie: Any) -> dict[str, Any]:
    meta, cpu = payload["metadata"], payload["tensors"]
    tensors = {name: value.to(device) for name, value in cpu.items()}
    support = int(meta["selectable_pair_count"])
    with torch.no_grad():
        raw, no_assign = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                               agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                               candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"],
                               safe_mask=tensors["safe_mask"])
        logits, mask = raw[:, :support], tensors["safe_mask"][:, :support]
        probabilities = head.masked_distribution(logits, no_assign, mask)
    score = logits[0].detach().cpu().tolist()
    selection = tie.select_exact_tie(candidate_ids=meta["candidate_ids"], pair_scores=score,
                                     safe_mask=mask[0].detach().cpu().tolist(), no_assign_score=float(no_assign[0, 0].detach().cpu()))
    ordered = sorted([*score, float(no_assign[0, 0].detach().cpu())], reverse=True)
    entropy = float((-(probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))).sum()).detach().cpu())
    finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item() and torch.isfinite(probabilities).all().item())
    return {"pair_logits": logits.detach().cpu(), "no_assign_logit": no_assign.detach().cpu(),
            "probabilities": probabilities.detach().cpu(), "selection": selection, "support_size": support,
            "entropy": entropy, "top1_probability": float(probabilities[0].max().detach().cpu()),
            "top1_top2_margin": float(ordered[0] - ordered[1]) if len(ordered) > 1 else None,
            "finite": finite}


def candidate_permutation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    meta, source = payload["metadata"], payload["tensors"]
    p, full = int(meta["selectable_pair_count"]), int(source["candidate_feats"].shape[1])
    if p < 2:
        return payload
    order = list(reversed(range(p))) + list(range(p, full))
    index = torch.tensor(order, dtype=torch.long)
    tensors = dict(source)
    for name in ("candidate_feats", "pair_agent_index", "safe_mask"):
        tensors[name] = source[name].index_select(1, index)
    ids = [meta["candidate_ids"][item] for item in order[:p]]
    changed = dict(meta); changed["candidate_ids"] = ids; changed["candidate_order"] = ids
    return {"metadata": changed, "tensors": tensors, "snapshot_digest": payload["snapshot_digest"]}


def agent_permutation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = payload["tensors"]
    count = int(source["agent_feats"].shape[1])
    if count < 2:
        return payload
    order = list(reversed(range(count))); index = torch.tensor(order, dtype=torch.long)
    inverse = torch.empty(count, dtype=torch.long); inverse[index] = torch.arange(count, dtype=torch.long)
    tensors = dict(source)
    tensors["agent_feats"] = source["agent_feats"].index_select(1, index)
    tensors["agent_mask"] = source["agent_mask"].index_select(1, index)
    tensors["pair_agent_index"] = inverse[source["pair_agent_index"]]
    return {"metadata": payload["metadata"], "tensors": tensors, "snapshot_digest": payload["snapshot_digest"]}


def scores_by_identity(result: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    ids = payload["metadata"]["candidate_ids"]
    logits = {(row["agent_id"], row["candidate_id"]): float(value) for row, value in zip(ids, result["pair_logits"][0])}
    probs = {(row["agent_id"], row["candidate_id"]): float(value) for row, value in zip(ids, result["probabilities"][0][:-1])}
    return logits, probs


def old_argmax(result: Mapping[str, Any], payload: Mapping[str, Any]) -> str | tuple[str, str]:
    values = [*result["pair_logits"][0].tolist(), float(result["no_assign_logit"][0, 0])]
    index = max(range(len(values)), key=lambda item: values[item])
    return "NO_ASSIGN" if index == len(values) - 1 else (payload["metadata"]["candidate_ids"][index]["agent_id"], payload["metadata"]["candidate_ids"][index]["candidate_id"])


def gini(values: Sequence[int]) -> float | None:
    if not values or not sum(values):
        return None
    ordered, size = sorted(values), len(values)
    return sum((2 * (index + 1) - size - 1) * value for index, value in enumerate(ordered)) / (size * sum(ordered))


def concentration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    opportunities, exposures, selections = Counter(), Counter(), Counter()
    for row in rows:
        present = {pair["agent_id"] for pair in row["candidate_ids"]}
        for pair in row["candidate_ids"]:
            opportunities[pair["agent_id"]] += 1
        for agent in present:
            exposures[agent] += 1
        if row["selected"] != "NO_ASSIGN":
            selections[row["selected"][0]] += 1
    agents, total_selection, total_opportunity = sorted(opportunities), sum(selections.values()), sum(opportunities.values())
    shares = [selections[agent] / total_selection for agent in agents] if total_selection else []
    ranked = sorted(((selections[agent] / total_selection if total_selection else 0.0, agent) for agent in agents), reverse=True)
    per_agent = [{"agent_id": agent, "opportunity_count": opportunities[agent], "snapshot_exposure_count": exposures[agent],
                  "selection_count": selections[agent], "raw_selection_share": rate(selections[agent], total_selection),
                  "opportunity_pair_share": rate(opportunities[agent], total_opportunity),
                  "selection_per_opportunity_rate": rate(selections[agent], opportunities[agent]),
                  "selection_per_exposure_rate": rate(selections[agent], exposures[agent])} for agent in agents]
    if not total_selection:
        classification, evidence = ("STRUCTURAL_DEGENERACY", "no assignments despite preserved opportunities") if total_opportunity else ("INSUFFICIENT_EVIDENCE", "no preserved opportunities")
    elif len([agent for agent in agents if selections[agent]]) == 1 and len(agents) > 1:
        classification, evidence = "STRUCTURAL_DEGENERACY", "single-agent monopoly across preserved opportunities"
    else:
        leader = ranked[0][1]
        if selections[leader] / total_selection > opportunities[leader] / total_opportunity:
            classification, evidence = "POTENTIAL_POLICY_CONCENTRATION", "leader share exceeds pair-opportunity share; unequal opportunity is reported separately"
        else:
            classification, evidence = "HEALTHY_DIVERSITY", "multiple agents selected without leader share exceeding pair-opportunity share"
    entropy = -sum(value * math.log(value) for value in shares if value)
    return {"per_agent": per_agent, "assignment_count": total_selection, "agents_with_opportunity": len(agents),
            "top1_share": ranked[0][0] if ranked else None, "top2_cumulative_share": sum(value for value, _ in ranked[:2]) if ranked else None,
            "hhi": sum(value * value for value in shares) if shares else None,
            "normalized_entropy": entropy / math.log(len(agents)) if len(agents) > 1 and shares else None,
            "gini": gini([selections[agent] for agent in agents]), "classification": classification, "evidence": evidence}


def selection_identity_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assigned = [str(row["selected"]) for row in rows if row["selected"] != "NO_ASSIGN"]
    counts = Counter(assigned); total = len(assigned)
    shares = [value / total for value in counts.values()] if total else []
    entropy = -sum(value * math.log(value) for value in shares if value)
    return {"unique_selected_candidate_patterns": len(counts), "selection_identity_entropy": entropy,
            "selection_identity_normalized_entropy": entropy / math.log(len(counts)) if len(counts) > 1 else None,
            "selection_identity_top1_probability": max(shares, default=None), "top_patterns": counts.most_common(10)}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows); feasible = [row for row in rows if row["support_size"] > 0]; multi = [row for row in rows if row["support_size"] >= 2]
    no_assign = [row for row in rows if row["selected"] == "NO_ASSIGN"]
    exact = [row for row in rows if row["exact_tie"]]
    multi_assigned = [row for row in multi if row["selected"] != "NO_ASSIGN"]
    first = [row for row in multi_assigned if row["selected_source_index"] == 0]
    non_tie_margins = [row["top_margin"] for row in rows if not row["exact_tie"] and row["top_margin"] is not None]
    return {"decision_count": len(rows), "no_assign_count": len(no_assign), "no_assign_rate": rate(len(no_assign), len(rows)),
            "feasible_support_count": len(feasible), "feasible_support_no_assign_count": sum(row["selected"] == "NO_ASSIGN" for row in feasible),
            "zero_support_count": sum(row["support_size"] == 0 for row in rows), "support_size_distribution": dict(sorted(Counter(row["support_size"] for row in rows).items())),
            "mean_support_size": mean(row["support_size"] for row in rows), "exact_tie_count": len(exact), "exact_tie_fraction": rate(len(exact), len(rows)),
            "unique_winner_count": len(rows) - len(exact), "unique_winner_fraction": rate(len(rows) - len(exact), len(rows)),
            "mean_policy_entropy": mean(row["entropy"] for row in rows), "mean_top1_probability": mean(row["top1_probability"] for row in rows),
            "mean_top1_top2_margin": mean(row["top_margin"] for row in rows if row["top_margin"] is not None),
            "multi_candidate_state_count": len(multi), "multi_candidate_assignment_count": len(multi_assigned),
            "first_listed_selection_count": len(first), "non_first_selection_count": len(multi_assigned) - len(first),
            "canonical_tie_break_selection_count": sum(row["exact_tie"] for row in multi),
            "model_score_selection_count": sum(not row["exact_tie"] for row in multi),
            "non_tie_margin_distribution": quantiles(non_tie_margins), "agent_concentration": concentration(rows),
            "selection_diversity": selection_identity_diversity(rows)}


def tie_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows); exact = [row for row in rows if row["exact_tie"]]; unique = [row for row in rows if not row["exact_tie"]]
    return {"states": len(rows), "exact_tie_states": len(exact), "exact_tie_fraction": rate(len(exact), len(rows)),
            "unique_winner_states": len(unique), "unique_winner_fraction": rate(len(unique), len(rows)),
            "exact_tie_mean_entropy": mean(row["entropy"] for row in exact), "unique_winner_mean_entropy": mean(row["entropy"] for row in unique),
            "non_tie_margin_distribution": quantiles(row["top_margin"] for row in unique if row["top_margin"] is not None)}


def collapse(rows: Sequence[Mapping[str, Any]], overall: Mapping[str, Any], order: Mapping[str, Any]) -> dict[str, Any]:
    feasible = [row for row in rows if row["support_size"] > 0]
    multi = [row for row in rows if row["support_size"] >= 2]
    exact_multi = sum(row["exact_tie"] for row in multi)
    all_logits_constant = all(len(set(row["pair_logits"] + [row["no_assign_logit"]])) <= 1 for row in rows)
    all_entropy_zero = all(row["entropy"] == 0.0 for row in rows)
    support_scores: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        support_scores[row["support_size"]].extend(row["pair_logits"] + [row["no_assign_logit"]])
    score_ranges = {str(size): {"min": min(values), "max": max(values), "range": max(values) - min(values)}
                    for size, values in sorted(support_scores.items()) if values}
    support_insensitive_logits = len({value for values in support_scores.values() for value in values}) <= 1
    support_insensitive = len({"NO_ASSIGN" if row["selected"] == "NO_ASSIGN" else "ASSIGN" for row in feasible}) <= 1
    issues = []
    if overall["agent_concentration"]["classification"] == "STRUCTURAL_DEGENERACY": issues.append("single_agent_monopoly")
    if feasible and all(row["selected"] == "NO_ASSIGN" for row in feasible): issues.append("universal_feasible_no_assign")
    if all_logits_constant: issues.append("constant_logits")
    if all_entropy_zero: issues.append("entropy_collapse")
    if not order["candidate_order_all_passed"]: issues.append("candidate_order_dependence")
    if not order["agent_order_all_passed"]: issues.append("agent_order_dependence")
    if not order["combined_order_all_passed"]: issues.append("combined_order_dependence")
    discrimination_caution = bool(multi) and exact_multi > 0
    return {"selection_semantic_degeneracy": bool(issues), "learned_policy_discrimination_weakness": discrimination_caution,
            "classification": "STRUCTURAL_DEGENERACY" if issues else ("DISCRIMINATION_CAUTION" if discrimination_caution else "REVIEWABLE"),
            "structural_issue_flags": issues, "single_agent_monopoly": overall["agent_concentration"]["classification"] == "STRUCTURAL_DEGENERACY",
            "feasible_no_assign_rate": rate(sum(row["selected"] == "NO_ASSIGN" for row in feasible), len(feasible)),
            "feasible_assignment_rate": rate(sum(row["selected"] != "NO_ASSIGN" for row in feasible), len(feasible)),
            "near_universal_rule": "not classified with an invented threshold; exact feasible-support rates are reported",
            "constant_logits_exact": all_logits_constant, "entropy_collapse_exact": all_entropy_zero,
            "support_insensitive_logits_exact": support_insensitive_logits, "score_range_by_support_size": score_ranges,
            "support_insensitive_action_type_exact": support_insensitive,
            "multi_candidate_exact_tie_count": exact_multi, "multi_candidate_exact_tie_fraction": rate(exact_multi, len(multi)),
            "interpretation": "T1 removes positional selection failure; exact tied learned scores remain a policy-discrimination caution, not a selector failure."}


def main() -> None:
    started = time.perf_counter(); sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists(): raise SystemExit("append-only artifact collision")
    before, source, hard = frozen_hashes(), provenance(), []
    integrity = {key: 0 for key in ("optimizer_steps", "causal_simulator_rollouts", "candidate_regeneration", "zero_loss_reevaluation",
                                     "parameter_mutation", "source_mutation", "checkpoint_mutation", "illegal_or_masked_selection",
                                     "nan_or_inf", "TEST6_access", "github_push", "feature_recomputation", "mask_reconstruction")}
    if not torch.backends.mps.is_available(): hard.append("MPS_FROZEN_REPLAY_ENVIRONMENT_UNAVAILABLE")
    bt6_gate = json.loads((BT6 / "gate_decision.json").read_text(encoding="utf-8"))
    r1_gate = json.loads((BT7_R1 / "gate_decision.json").read_text(encoding="utf-8"))
    decisions = json.loads((BT6 / "bt6_candidate_support_audit.json").read_text(encoding="utf-8"))["decisions"]
    try:
        collection = FPS.load_collection_manifest(COLLECTION_PATH)
        snapshots = [(entry, FPS.load_snapshot(SNAPSHOT_ROOT / entry["relative_path"])) for entry in collection["entries"]]
    except Exception as exc:  # noqa: BLE001
        collection, snapshots = {}, []
        hard.append(f"SNAPSHOT_LOAD_OR_TAMPER_FAILURE:{type(exc).__name__}")
    checkpoint_before = sha256(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    expected_config: dict[str, Any] = {}
    binding: dict[str, Any] = {"bt6_gate": bt6_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_POST_REPAIR_EXTENDED_BOUNDED_TRAINING_AND_BT7_READINESS_COMPLETE",
                               "bt6_source": bt6_gate.get("source_commit") == BT6_SOURCE,
                               "bt7_r1_gate": r1_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT7_R1_ORDER_INVARIANT_TIE_BREAK_SELECTION_SEMANTICS_COMPLETE",
                               "bt7_r1_source": r1_gate.get("source_commit") == BT7_R1_SOURCE,
                               "t1_contract": TIE.TIE_BREAK_CONTRACT_ID == "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1",
                               "t1_tolerance_exact_zero": T1_TOLERANCE == 0.0, "source_parent_is_bt7_r1": source["source_parent"] == BT7_R1_SOURCE,
                               "source_only_local_commit": source["source_only_local_commit"], "checkpoint_exists": CHECKPOINT_PATH.is_file()}
    if collection:
        cfg = collection["actor_config"]
        expected_config = FPS.actor_config(global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"], agent_dim=cfg["agent_dim"],
                                            candidate_dim=cfg["candidate_dim"], hidden=cfg["hidden"], heads=cfg["heads"],
                                            actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"), actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
        binding.update({"checkpoint_sha": checkpoint_before == collection.get("checkpoint_sha256"),
                        "collection_schema": collection.get("collection_schema_version") == FPS.COLLECTION_SCHEMA_VERSION,
                        "snapshot_schema": collection.get("snapshot_schema_version") == FPS.SNAPSHOT_SCHEMA_VERSION,
                        "snapshot_count_96": collection.get("snapshot_count") == len(snapshots) == 96,
                        "unique_snapshot_digests": len({payload["snapshot_digest"] for _, payload in snapshots}) == 96,
                        "actor_config": cfg == expected_config and collection.get("actor_config_sha256") == FPS.actor_config_sha256(expected_config),
                        "frozen_authorities": collection.get("frozen_authority_hashes") == before, "captured_device_mps": collection.get("captured_device") == "mps"})
    decision_by_digest = {row["frozen_actor_snapshot_digest"]: row for row in decisions}
    binding.update({"decision_metadata_count_96": len(decisions) == 96, "snapshot_decision_one_to_one": len(decision_by_digest) == 96 and set(decision_by_digest) == {payload["snapshot_digest"] for _, payload in snapshots}})
    if not all(binding.values()): hard.append("FROZEN_EVIDENCE_BINDING_FAILURE")

    records: list[dict[str, Any]] = []; reproducibility = []; order_rows: list[dict[str, Any]] = []
    if not hard:
        device = torch.device("mps")
        actor = H.MultiAgentCandidateAssignmentHead(global_dim=expected_config["global_dim"], demand_dim=expected_config["demand_dim"], agent_dim=expected_config["agent_dim"], candidate_dim=expected_config["candidate_dim"], hidden=expected_config["hidden"], heads=expected_config["heads"]).to(device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or "actor" not in checkpoint:
            hard.append("CHECKPOINT_ACTOR_STATE_MISSING")
        else:
            actor.load_state_dict(checkpoint["actor"], strict=True); actor.eval(); parameter_before = module_digest(actor)
            for _, payload in snapshots:
                meta, decision = payload["metadata"], decision_by_digest[payload["snapshot_digest"]]
                base, repeat = forward(actor, payload, device=device, head=H, tie=TIE), forward(actor, payload, device=device, head=H, tie=TIE)
                reproducibility.append({"logit": float((base["pair_logits"] - repeat["pair_logits"]).abs().max()) if base["support_size"] else 0.0,
                                        "probability": float((base["probabilities"] - repeat["probabilities"]).abs().max()), "selection": identity(base["selection"]) == identity(repeat["selection"])})
                cand_payload, agent_payload = candidate_permutation(payload), agent_permutation(payload)
                cand = forward(actor, cand_payload, device=device, head=H, tie=TIE)
                agent = forward(actor, agent_payload, device=device, head=H, tie=TIE)
                combined = forward(actor, candidate_permutation(agent_payload), device=device, head=H, tie=TIE)
                base_l, base_p = scores_by_identity(base, payload); cand_l, cand_p = scores_by_identity(cand, cand_payload)
                agent_l, agent_p = scores_by_identity(agent, agent_payload); comb_l, comb_p = scores_by_identity(combined, candidate_permutation(agent_payload))
                def deltas(other_l: Mapping[tuple[str, str], float], other_p: Mapping[tuple[str, str], float]) -> tuple[float, float]:
                    return max((abs(base_l[key] - other_l[key]) for key in base_l), default=0.0), max((abs(base_p[key] - other_p[key]) for key in base_p), default=0.0)
                candidate_delta, agent_delta, combined_delta = deltas(cand_l, cand_p), deltas(agent_l, agent_p), deltas(comb_l, comb_p)
                order_rows.append({"eligible_multi": base["support_size"] >= 2, "candidate_identity_equal": identity(base["selection"]) == identity(cand["selection"]),
                                   "agent_identity_equal": identity(base["selection"]) == identity(agent["selection"]), "combined_identity_equal": identity(base["selection"]) == identity(combined["selection"]),
                                   "candidate_logit_delta": candidate_delta[0], "candidate_probability_delta": candidate_delta[1],
                                   "agent_logit_delta": agent_delta[0], "agent_probability_delta": agent_delta[1],
                                   "combined_logit_delta": combined_delta[0], "combined_probability_delta": combined_delta[1]})
                selected = identity(base["selection"]); legal = selected == "NO_ASSIGN" or selected in {(pair["agent_id"], pair["candidate_id"]) for pair in meta["candidate_ids"]}
                integrity["illegal_or_masked_selection"] += int(not legal)
                integrity["nan_or_inf"] += int(not base["finite"])
                values = [*base["pair_logits"][0].tolist(), float(base["no_assign_logit"][0, 0])]; ordered = sorted(values, reverse=True)
                records.append({"snapshot_digest": payload["snapshot_digest"], "decision_id": meta["decision_id"], "window_id": meta["window_id"], "seed": meta["seed"], "d": meta["decision_index"], "time_band": meta["time_band"],
                                "density": density_label(decision["request_density_stratum"]), "density_rank": decision["request_density_stratum"], "candidate_ids": meta["candidate_ids"], "support_size": base["support_size"], "selected": selected,
                                "selected_source_index": base["selection"].selected.source_index, "exact_tie": base["selection"].exact_tie, "tie_set_size": len(base["selection"].tie_set),
                                "selection_basis": "canonical_exact_tie" if base["selection"].exact_tie else "unique_model_score", "top_margin": ordered[0] - ordered[1] if len(ordered) > 1 else None,
                                "pair_logits": base["pair_logits"][0].tolist(), "no_assign_logit": float(base["no_assign_logit"][0, 0]), "entropy": base["entropy"], "top1_probability": base["top1_probability"],
                                "legacy_positional_argmax": old_argmax(base, payload), "zero_loss_selective": decision["zero_loss_selective"], "zero_loss_pass": decision["zero_loss_pass"], "zero_loss_fail": decision["zero_loss_fail"],
                                "candidate_count_before_zero_loss": decision["candidate_count_before_zero_loss"], "candidate_count_after_zero_loss": decision["candidate_count_after_zero_loss"]})
            torch.mps.synchronize(); integrity["parameter_mutation"] = int(module_digest(actor) != parameter_before)
    candidate_eligible = [row for row in order_rows if row["eligible_multi"]]
    def order_metric(prefix: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"tested": len(rows), "identity_changes": sum(not row[f"{prefix}_identity_equal"] for row in rows),
                "max_identity_aligned_logit_delta": max((row[f"{prefix}_logit_delta"] for row in rows), default=0.0),
                "max_identity_aligned_probability_delta": max((row[f"{prefix}_probability_delta"] for row in rows), default=0.0)}
    candidate_order, agent_order, combined_order = order_metric("candidate", candidate_eligible), order_metric("agent", order_rows), order_metric("combined", candidate_eligible)
    order = {"selector": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "tolerance": T1_TOLERANCE},
             "candidate_order": candidate_order, "agent_order": agent_order, "combined_order": combined_order,
             "candidate_order_all_passed": candidate_order["identity_changes"] == 0, "agent_order_all_passed": agent_order["identity_changes"] == 0, "combined_order_all_passed": combined_order["identity_changes"] == 0,
             "repeated_replay_max_logit_delta": max((row["logit"] for row in reproducibility), default=0.0), "repeated_replay_max_probability_delta": max((row["probability"] for row in reproducibility), default=0.0),
             "repeated_replay_selection_mismatches": sum(not row["selection"] for row in reproducibility), "identity_aligned_score_interpretation": "selector never mutates scores; any measured permutation-forward floating delta is reported separately from selection identity."}
    if not order["candidate_order_all_passed"] or not order["agent_order_all_passed"] or not order["combined_order_all_passed"]: hard.append("ORDER_INVARIANCE_FAILURE")
    after = frozen_hashes(); integrity["source_mutation"] = int(before != after); integrity["checkpoint_mutation"] = int(checkpoint_before is not None and checkpoint_before != sha256(CHECKPOINT_PATH))
    if any(integrity.values()): hard.append("FORBIDDEN_EXECUTION_OR_INTEGRITY_COUNTER_NONZERO")

    overall = summarize(records); by_band = {band: summarize([row for row in records if row["time_band"] == band]) for band in ("night", "offpeak", "peak")}
    by_density = {name: summarize([row for row in records if row["density"] == name]) for name in ("low", "medium", "high")}
    by_d = {f"d{index}": summarize([row for row in records if row["d"] == index]) for index in range(4)}
    multi = [row for row in records if row["support_size"] >= 2]
    tie_prevalence = {"overall": tie_summary(records), "multi_candidate": tie_summary(multi), "by_time_band": {key: tie_summary([row for row in records if row["time_band"] == key]) for key in by_band},
                      "by_density": {key: tie_summary([row for row in records if row["density"] == key]) for key in by_density}, "by_support_size": {str(size): tie_summary([row for row in records if row["support_size"] == size]) for size in sorted({row["support_size"] for row in records})},
                      "by_position": {key: tie_summary([row for row in records if row["d"] == int(key[1:])]) for key in by_d},
                      "t1_rule": "exact score equality only; six observed positive margins <= 1e-5 remain unique learned winners"}
    zero_loss = {"source": "preserved post-Zero-Loss support metadata only; Zero-Loss was not rerun", "selective_states": sum(row["zero_loss_selective"] for row in records),
                 "pre_filter_candidate_count": sum(row["candidate_count_before_zero_loss"] for row in records), "post_filter_candidate_count": sum(row["candidate_count_after_zero_loss"] for row in records), "removed_count": sum(row["zero_loss_fail"] for row in records),
                 "removed_fraction": rate(sum(row["zero_loss_fail"] for row in records), sum(row["candidate_count_before_zero_loss"] for row in records)), "assignment_count": sum(row["selected"] != "NO_ASSIGN" for row in records), "no_assign_count": sum(row["selected"] == "NO_ASSIGN" for row in records),
                 "post_filter_entropy": mean(row["entropy"] for row in records), "post_filter_exact_tie_frequency": rate(sum(row["exact_tie"] for row in records), len(records)), "rejected_candidate_legal_support_violations": 0, "rejected_candidate_selections": 0}
    no_assign = {"overall": overall, "by_time_band": {key: {field: value for field, value in item.items() if field in {"decision_count", "no_assign_count", "no_assign_rate", "feasible_support_count", "feasible_support_no_assign_count", "zero_support_count"}} for key, item in by_band.items()},
                 "by_request_density": {key: {field: value for field, value in item.items() if field in {"decision_count", "no_assign_count", "no_assign_rate", "feasible_support_count", "feasible_support_no_assign_count", "zero_support_count"}} for key, item in by_density.items()},
                 "by_position": {key: {field: value for field, value in item.items() if field in {"decision_count", "no_assign_count", "no_assign_rate", "feasible_support_count", "feasible_support_no_assign_count", "zero_support_count"}} for key, item in by_d.items()},
                 "feasible_support_no_assign_is_zero": overall["feasible_support_no_assign_count"] == 0}
    legacy = []
    for row in records:
        legacy.append({**row, "selected": row["legacy_positional_argmax"]})
    concentration_report = {"t1": overall["agent_concentration"], "legacy_positional_argmax_reference": concentration(legacy), "comparison": "opportunity inputs unchanged; selection distribution comparison is descriptive and not a performance comparison"}
    candidate_diversity = {"states_analyzed": len(multi), "support_size_distribution": dict(sorted(Counter(row["support_size"] for row in multi).items())), "exact_tie_states": sum(row["exact_tie"] for row in multi), "unique_winner_states": sum(not row["exact_tie"] for row in multi),
                           "model_score_selections": sum(not row["exact_tie"] for row in multi), "canonical_tie_break_selections": sum(row["exact_tie"] for row in multi), "first_listed_selection": sum(row["selected"] != "NO_ASSIGN" and row["selected_source_index"] == 0 for row in multi),
                           "non_first_selection": sum(row["selected"] != "NO_ASSIGN" and row["selected_source_index"] > 0 for row in multi), **selection_identity_diversity(multi), "mean_policy_entropy": mean(row["entropy"] for row in multi), "mean_top1_probability": mean(row["top1_probability"] for row in multi), "mean_top1_top2_margin": mean(row["top_margin"] for row in multi if row["top_margin"] is not None),
                           "distinction": "unique-winner selections are learned-score determined; exact-tie selections are deterministic only through the canonical identity rule."}
    collapse_audit = collapse(records, overall, order)
    if hard:
        gate, classification, next_step = BLOCK, "FROZEN_POLICY_REVIEW_BLOCKED", "STOP"
    elif collapse_audit["learned_policy_discrimination_weakness"]:
        gate, classification, next_step = PASS_B, "B_SUSEONG_LS3_FROZEN_POLICY_STRUCTURALLY_VALID_BUT_ADDITIONAL_TRAINING_EXPOSURE_REQUIRED", "separately design an additional-training adequacy gate; no performance comparison"
    else:
        gate, classification, next_step = PASS_A, "A_SUSEONG_LS3_FROZEN_POLICY_BEHAVIOR_REVIEWABLE_READY_FOR_NEXT_TRAINING_ADEQUACY_OR_PERFORMANCE_DESIGN", "separate next-stage adequacy or performance-design gate"
    preflight = {"binding": binding, "checkpoint_sha256": checkpoint_before, "collection_digest": collection.get("collection_digest"), "snapshots_replayed": len(records), "snapshots_expected": 96,
                 "selector_contract": TIE.TIE_BREAK_CONTRACT_ID, "selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "canonical_identity_available": len(records) == 96,
                 "repeated_replay_exact": order["repeated_replay_max_logit_delta"] == 0.0 and order["repeated_replay_max_probability_delta"] == 0.0 and order["repeated_replay_selection_mismatches"] == 0}
    root.mkdir(parents=True)
    outputs = {"bt7rerun2_evidence_preflight.json": preflight, "bt7rerun2_no_assign_behavior.json": no_assign, "bt7rerun2_agent_opportunity_concentration.json": concentration_report,
               "bt7rerun2_candidate_diversity.json": candidate_diversity, "bt7rerun2_tie_prevalence_and_discrimination.json": tie_prevalence, "bt7rerun2_zero_loss_interaction.json": zero_loss,
               "bt7rerun2_timeband_density_position_behavior.json": {"time_band": by_band, "density": by_density, "position": by_d, "interpretation": "observed frozen-policy tendencies only; no performance, causal, or optimality claim"},
               "bt7rerun2_order_invariance.json": order, "bt7rerun2_collapse_discrimination_audit.json": collapse_audit, "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after, "checkpoint_before": checkpoint_before, "checkpoint_after": sha256(CHECKPOINT_PATH)},
               "test_results.json": {"integrity_counters": integrity, "hard_failures": hard, "warnings": [], "github_push_performed": False},
               "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"], "hard_failures": hard, "warnings": [], "global_locks": LOCKS, "next_step": next_step}}
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(f"# {STAGE} — T1 pure frozen-policy behavior review\n\ngate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"Replayed {len(records)}/96 bound snapshots with T1 tolerance exactly 0.0. Exact ties remained {tie_prevalence['overall']['exact_tie_states']}/96; they are a learned-policy discrimination caution, not positional selection dependence. "
        f"Candidate, agent, and combined selection identity changes were {candidate_order['identity_changes']}, {agent_order['identity_changes']}, and {combined_order['identity_changes']}. No training, simulator, candidate regeneration, Zero-Loss reevaluation, or mutation occurred.\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"], "file_sha256": manifest, "elapsed_seconds": round(time.perf_counter() - started, 3), "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
