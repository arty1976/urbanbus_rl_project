#!/usr/bin/env python3
"""BT8-R1: pure A1 frozen-policy discrimination and concentration review.

Only the bound A1 checkpoint and the 96 preserved actor-input snapshots enter
inference.  The earlier failed A1 metadata lookup has no artifact and is not
read.  This module never creates candidates, invokes Zero-Loss, advances a
simulator, steps an optimizer, or writes source/checkpoint state.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R1"
PASS_A = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R1_FROZEN_POLICY_DISCRIMINATION_REVIEW_COMPLETE"
PASS_B = "PASS_WITH_REMAINING_POLICY_DISCRIMINATION_CAUTION"
PASS_C = "PASS_WITH_NO_MATERIAL_DISCRIMINATION_IMPROVEMENT"
BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R1_FROZEN_POLICY_INTEGRITY_FAILURE"
BT8_A1_SOURCE = "43dee0966abd62efec89e7229421341b121e1e81"
BT7_RERUN2_SOURCE = "dfa18a837d716dfc180418a239ac661aaf1544a3"
BT7_RERUN2_GATE = "PASS_WITH_FROZEN_POLICY_DISCRIMINATION_CAUTION"
T1_TOLERANCE = 0.0
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = "05_training/run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review.py"
TEST_REL = "05_training/test_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review.py"
SOURCE_FILES = {SOURCE_REL, TEST_REL}
A1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_a1_bounded_training_20260822_225122+09:00"
A1_SNAPSHOTS = A1 / "bt8_a1_frozen_policy_snapshots"
A1_COLLECTION = A1_SNAPSHOTS / "collection_manifest.json"
A1_CHECKPOINT = A1 / "bt8_a1_joint_assignment_checkpoint.pt"
PREVIOUS = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review_20260822_211215+09:00"
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_200221+09:00"
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


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
    changed = [row for row in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if row]
    return {"source_commit": git(["rev-parse", "HEAD"]), "source_parent": git(["rev-parse", "HEAD^"]),
            "changed_files": changed, "source_only_local_commit": set(changed) == SOURCE_FILES,
            "github_push_performed": False}


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return {key: None for key in ("min", "p25", "median", "mean", "p75", "p95", "max")}
    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {"min": ordered[0], "p25": at(.25), "median": at(.5), "mean": statistics.mean(ordered),
            "p75": at(.75), "p95": at(.95), "max": ordered[-1]}


def action_identity(selection: Any) -> str | tuple[str, str]:
    return "NO_ASSIGN" if selection.selected.is_no_assign else (str(selection.selected.agent_id), str(selection.selected.candidate_id))


def frozen_forward(actor: torch.nn.Module, payload: Mapping[str, Any], *, device: torch.device, head: Any, tie: Any) -> dict[str, Any]:
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
    pair_scores = logits[0].detach().cpu().tolist()
    selection = tie.select_exact_tie(candidate_ids=meta["candidate_ids"], pair_scores=pair_scores,
                                     safe_mask=mask[0].detach().cpu().tolist(), no_assign_score=float(no_assign[0, 0].detach().cpu()))
    legal_scores = [*pair_scores, float(no_assign[0, 0].detach().cpu())]
    ordered_scores, ordered_probs = sorted(legal_scores, reverse=True), sorted(probabilities[0].detach().cpu().tolist(), reverse=True)
    entropy = float((-(probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))).sum()).detach().cpu())
    return {"pair_logits": logits.detach().cpu(), "no_assign_logit": no_assign.detach().cpu(), "probabilities": probabilities.detach().cpu(),
            "selection": selection, "support_size": support, "entropy": entropy,
            "top_score_margin": ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else None,
            "top1_probability": ordered_probs[0], "top2_probability": ordered_probs[1] if len(ordered_probs) > 1 else None,
            "top_probability_margin": ordered_probs[0] - ordered_probs[1] if len(ordered_probs) > 1 else None,
            "finite": bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item() and torch.isfinite(probabilities).all().item())}


def candidate_permutation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    meta, source = payload["metadata"], payload["tensors"]
    support, full = int(meta["selectable_pair_count"]), int(source["candidate_feats"].shape[1])
    if support < 2: return payload
    order = list(reversed(range(support))) + list(range(support, full)); index = torch.tensor(order, dtype=torch.long)
    tensors = dict(source)
    for name in ("candidate_feats", "pair_agent_index", "safe_mask"): tensors[name] = source[name].index_select(1, index)
    ids = [meta["candidate_ids"][item] for item in order[:support]]
    changed = dict(meta); changed["candidate_ids"] = ids; changed["candidate_order"] = ids
    return {"metadata": changed, "tensors": tensors, "snapshot_digest": payload["snapshot_digest"]}


def agent_permutation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = payload["tensors"]; count = int(source["agent_feats"].shape[1])
    if count < 2: return payload
    order = list(reversed(range(count))); index = torch.tensor(order, dtype=torch.long)
    inverse = torch.empty(count, dtype=torch.long); inverse[index] = torch.arange(count, dtype=torch.long)
    tensors = dict(source); tensors["agent_feats"] = source["agent_feats"].index_select(1, index)
    tensors["agent_mask"] = source["agent_mask"].index_select(1, index); tensors["pair_agent_index"] = inverse[source["pair_agent_index"]]
    return {"metadata": payload["metadata"], "tensors": tensors, "snapshot_digest": payload["snapshot_digest"]}


def scores_by_identity(result: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    ids = payload["metadata"]["candidate_ids"]
    logits = {(row["agent_id"], row["candidate_id"]): float(value) for row, value in zip(ids, result["pair_logits"][0])}
    probs = {(row["agent_id"], row["candidate_id"]): float(value) for row, value in zip(ids, result["probabilities"][0][:-1])}
    return logits, probs


def classify_alternatives(payload: Mapping[str, Any], indices: Sequence[int], *, candidate_names: Sequence[str], agent_names: Sequence[str]) -> dict[str, Any]:
    """Use only preserved numeric support context; never infer unavailable routes."""
    meta, tensors = payload["metadata"], payload["tensors"]
    indexes = sorted({int(index) for index in indices})
    common = {"action_pair_count": len(indexes), "unavailable_preserved_fields": ["route/path identity", "stop sequence", "pickup/dropoff stop ids"]}
    if len(indexes) < 2:
        return {**common, "classification": "INSUFFICIENT_METADATA", "reason": "fewer than two assignment alternatives in the audited set",
                "varying_candidate_features": [], "varying_agent_features": []}
    candidate_values = [tensors["candidate_feats"][0, index].detach().cpu().tolist() for index in indexes]
    agent_values = [tensors["agent_feats"][0, int(tensors["pair_agent_index"][0, index])].detach().cpu().tolist() for index in indexes]
    varying_candidate = [name for offset, name in enumerate(candidate_names) if len({float(row[offset]) for row in candidate_values}) > 1]
    varying_agent = [name for offset, name in enumerate(agent_names) if len({float(row[offset]) for row in agent_values}) > 1]
    agents = {str(meta["candidate_ids"][index]["agent_id"]) for index in indexes}
    if varying_candidate or varying_agent:
        label, reason = "MEANINGFULLY_DISTINCT", "preserved physical/operational quantities differ by exact value; no magnitude threshold was introduced"
    elif len(agents) > 1:
        label, reason = "INSUFFICIENT_METADATA", "different agent identities have equal preserved numeric context and no route/path identity was retained"
    else:
        label, reason = "PHYSICALLY_EQUIVALENT_OR_NEAR_EQUIVALENT", "all preserved numeric physical/operational fields are exactly equal"
    return {**common, "classification": label, "reason": reason, "varying_candidate_features": varying_candidate,
            "varying_agent_features": varying_agent, "candidate_values": candidate_values, "agent_values": agent_values}


def group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows); multi = [row for row in rows if row["support_size"] >= 2]
    exact = [row for row in multi if row["exact_tie"]]
    meaningful = [row for row in multi if row["candidate_distinctness"]["classification"] == "MEANINGFULLY_DISTINCT"]
    meaningful_exact = [row for row in exact if row["tie_distinctness"]["classification"] == "MEANINGFULLY_DISTINCT"]
    unique = [row for row in multi if not row["exact_tie"]]
    return {"decision_count": len(rows), "multi_candidate_states": len(multi), "meaningfully_distinct_multi_candidate_states": len(meaningful),
            "exact_tie_states": len(exact), "meaningfully_distinct_exact_ties": len(meaningful_exact),
            "physically_equivalent_or_near_equivalent_exact_ties": sum(row["tie_distinctness"]["classification"] == "PHYSICALLY_EQUIVALENT_OR_NEAR_EQUIVALENT" for row in exact),
            "insufficient_metadata_exact_ties": sum(row["tie_distinctness"]["classification"] == "INSUFFICIENT_METADATA" for row in exact),
            "meaningfully_distinct_tie_rate": rate(len(meaningful_exact), len(meaningful)),
            "unique_winner_states": len(unique), "unique_winner_rate": rate(len(unique), len(multi)),
            "support_size_distribution": dict(sorted(Counter(row["support_size"] for row in rows).items()))}


def score_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows); exact = [row for row in rows if row["exact_tie"]]; unique = [row for row in rows if not row["exact_tie"]]
    def view(part: Sequence[Mapping[str, Any]], *, positive_only: bool = False) -> dict[str, Any]:
        margins = [row["top_score_margin"] for row in part if row["top_score_margin"] is not None and (not positive_only or row["top_score_margin"] > 0.0)]
        return {"states": len(part), "positive_top1_top2_score_margin": quantiles(margins),
                "entropy": quantiles(row["entropy"] for row in part), "top1_probability": quantiles(row["top1_probability"] for row in part),
                "top2_probability": quantiles(row["top2_probability"] for row in part if row["top2_probability"] is not None),
                "top1_top2_probability_margin": quantiles(row["top_probability_margin"] for row in part if row["top_probability_margin"] is not None)}
    return {"all_states": view(rows, positive_only=True), "exact_tie_states": view(exact), "unique_winner_states": view(unique, positive_only=True),
            "positive_margin_rule": "strictly positive score difference only; no T2 near-tie tolerance"}


def gini(values: Sequence[int]) -> float | None:
    if not values or not sum(values): return None
    ordered, size = sorted(values), len(values)
    return sum((2 * (index + 1) - size - 1) * value for index, value in enumerate(ordered)) / (size * sum(ordered))


def concentration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    opportunities, exposures, selections = Counter(), Counter(), Counter()
    for row in rows:
        present = {pair["agent_id"] for pair in row["candidate_ids"]}
        for pair in row["candidate_ids"]: opportunities[pair["agent_id"]] += 1
        for agent in present: exposures[agent] += 1
        if row["selected"] != "NO_ASSIGN": selections[row["selected"][0]] += 1
    agents, total_selection, total_opportunity = sorted(opportunities), sum(selections.values()), sum(opportunities.values())
    ranked = sorted(((rate(selections[agent], total_selection) or 0.0, agent) for agent in agents), reverse=True)
    per_agent = [{"agent_id": agent, "opportunity_count": opportunities[agent], "snapshot_exposure_count": exposures[agent],
                  "selection_count": selections[agent], "selection_per_opportunity_rate": rate(selections[agent], opportunities[agent]),
                  "raw_selection_share": rate(selections[agent], total_selection), "opportunity_pair_share": rate(opportunities[agent], total_opportunity)} for agent in agents]
    shares = [value for value, _ in ranked]; entropy = -sum(value * math.log(value) for value in shares if value)
    leader = ranked[0][1] if ranked else None
    leader_excess = (ranked[0][0] - (rate(opportunities[leader], total_opportunity) or 0.0)) if leader else None
    if not total_selection:
        status, evidence = "STRUCTURAL_DEGENERACY", "no assignments despite preserved opportunities"
    elif len([agent for agent in agents if selections[agent]]) == 1 and len(agents) > 1:
        status, evidence = "STRUCTURAL_DEGENERACY", "single-agent monopoly across preserved opportunities"
    elif leader_excess is not None and leader_excess <= 0.0:
        status, evidence = "OPPORTUNITY_EXPLAINABLE", "leader selection share does not exceed its pair-opportunity share"
    else:
        status, evidence = "POTENTIAL_POLICY_CONCENTRATION", "leader selection share exceeds its pair-opportunity share"
    return {"per_agent": per_agent, "assignment_count": total_selection, "agents_with_opportunity": len(agents),
            "top1_share": ranked[0][0] if ranked else None, "top2_cumulative_share": sum(shares[:2]) if ranked else None,
            "hhi": sum(value * value for value in shares) if shares else None,
            "normalized_entropy": entropy / math.log(len(agents)) if len(agents) > 1 and shares else None,
            "gini": gini([selections[agent] for agent in agents]), "classification": status, "evidence": evidence,
            "leader_agent": leader, "leader_selection_share_minus_opportunity_share": leader_excess}


def support_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["time_band"], int(row["decision_index"]), row["source_group"], tuple(tuple(item) for item in row["candidate_identity"]))


def main() -> None:
    started = time.perf_counter(); sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review as OLD

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review_{OLD.now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists(): raise SystemExit("append-only artifact collision")
    before, source, hard = OLD.frozen_hashes(), provenance(), []
    integrity = {key: 0 for key in ("optimizer_steps", "causal_simulator_rollouts", "training_visits", "candidate_regeneration",
        "zero_loss_reevaluation", "parameter_mutation", "source_mutation", "checkpoint_mutation", "illegal_or_masked_selection",
        "cross_window_contamination", "cross_seed_contamination", "future_leakage", "nan_or_inf", "TEST6_access", "github_push",
        "feature_recomputation", "mask_reconstruction")}
    if not torch.backends.mps.is_available(): hard.append("MPS_FROZEN_REPLAY_ENVIRONMENT_UNAVAILABLE")
    a1_gate = json.loads((A1 / "gate_decision.json").read_text(encoding="utf-8"))
    previous_gate = json.loads((PREVIOUS / "gate_decision.json").read_text(encoding="utf-8"))
    a1_decisions = json.loads((A1 / "bt8_a1_candidate_support_audit.json").read_text(encoding="utf-8"))["decisions"]
    old_decisions = json.loads((BT6 / "bt6_candidate_support_audit.json").read_text(encoding="utf-8"))["decisions"]
    try:
        collection = FPS.load_collection_manifest(A1_COLLECTION)
        snapshots = [(entry, FPS.load_snapshot(A1_SNAPSHOTS / entry["relative_path"])) for entry in collection["entries"]]
    except Exception as exc:  # noqa: BLE001
        collection, snapshots = {}, []
        hard.append(f"SNAPSHOT_LOAD_OR_TAMPER_FAILURE:{type(exc).__name__}")
    checkpoint_before = sha256(A1_CHECKPOINT) if A1_CHECKPOINT.is_file() else None
    decision_by_digest = {row["frozen_actor_snapshot_digest"]: row for row in a1_decisions}
    old_support, a1_support = Counter(map(support_signature, old_decisions)), Counter(map(support_signature, a1_decisions))
    expected_config: dict[str, Any] = {}
    binding = {"a1_gate": a1_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_A1_ADDITIONAL_BOUNDED_TRAINING_EXECUTION_AND_SNAPSHOT_PRESERVATION_COMPLETE",
               "a1_source": a1_gate.get("source_commit") == BT8_A1_SOURCE, "previous_gate": previous_gate.get("gate") == BT7_RERUN2_GATE,
               "previous_source": previous_gate.get("source_commit") == BT7_RERUN2_SOURCE, "source_parent_is_a1": source["source_parent"] == BT8_A1_SOURCE,
               "source_only_local_commit": source["source_only_local_commit"], "checkpoint_exists": A1_CHECKPOINT.is_file(),
               "support_multiset_matches_previous": old_support == a1_support, "support_multiset_intersection_96": sum((old_support & a1_support).values()) == 96}
    if collection:
        cfg = collection["actor_config"]
        expected_config = FPS.actor_config(global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"], agent_dim=cfg["agent_dim"],
            candidate_dim=cfg["candidate_dim"], hidden=cfg["hidden"], heads=cfg["heads"], actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
            actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
        binding.update({"checkpoint_sha": checkpoint_before == collection.get("checkpoint_sha256"),
            "collection_schema": collection.get("collection_schema_version") == FPS.COLLECTION_SCHEMA_VERSION,
            "snapshot_schema": collection.get("snapshot_schema_version") == FPS.SNAPSHOT_SCHEMA_VERSION,
            "snapshot_count_96": collection.get("snapshot_count") == len(snapshots) == 96,
            "unique_snapshot_digests": len({payload["snapshot_digest"] for _, payload in snapshots}) == 96,
            "actor_config": cfg == expected_config and collection.get("actor_config_sha256") == FPS.actor_config_sha256(expected_config),
            "frozen_authorities": collection.get("frozen_authority_hashes") == before, "captured_device_mps": collection.get("captured_device") == "mps",
            "snapshot_source_commit": all(payload["metadata"].get("source_commit") == BT8_A1_SOURCE for _, payload in snapshots)})
    binding["snapshot_decision_one_to_one"] = len(decision_by_digest) == 96 and set(decision_by_digest) == {payload["snapshot_digest"] for _, payload in snapshots}
    binding["t1_contract"] = TIE.TIE_BREAK_CONTRACT_ID == "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1" and T1_TOLERANCE == 0.0
    if not all(binding.values()): hard.append("FROZEN_EVIDENCE_BINDING_FAILURE")

    records: list[dict[str, Any]] = []; repeats: list[dict[str, Any]] = []; order_rows: list[dict[str, Any]] = []
    if not hard:
        device = torch.device("mps")
        actor = H.MultiAgentCandidateAssignmentHead(global_dim=expected_config["global_dim"], demand_dim=expected_config["demand_dim"],
            agent_dim=expected_config["agent_dim"], candidate_dim=expected_config["candidate_dim"], hidden=expected_config["hidden"], heads=expected_config["heads"]).to(device)
        checkpoint = torch.load(A1_CHECKPOINT, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or "actor" not in checkpoint:
            hard.append("CHECKPOINT_ACTOR_STATE_MISSING")
        else:
            actor.load_state_dict(checkpoint["actor"], strict=True); actor.eval(); parameter_before = OLD.module_digest(actor)
            candidate_names, agent_names = MC.LOCAL_SEARCH_FEATURE_NAMES, MC.AgentContext.FEATURE_NAMES
            for _, payload in snapshots:
                meta, decision = payload["metadata"], decision_by_digest[payload["snapshot_digest"]]
                base = frozen_forward(actor, payload, device=device, head=H, tie=TIE)
                repeat = frozen_forward(actor, payload, device=device, head=H, tie=TIE)
                repeats.append({"logit_delta": float((base["pair_logits"] - repeat["pair_logits"]).abs().max()) if base["support_size"] else 0.0,
                    "probability_delta": float((base["probabilities"] - repeat["probabilities"]).abs().max()),
                    "selection_equal": action_identity(base["selection"]) == action_identity(repeat["selection"])})
                candidate_payload, agent_payload = candidate_permutation(payload), agent_permutation(payload)
                candidate = frozen_forward(actor, candidate_payload, device=device, head=H, tie=TIE)
                agent = frozen_forward(actor, agent_payload, device=device, head=H, tie=TIE)
                combined_payload = candidate_permutation(agent_payload); combined = frozen_forward(actor, combined_payload, device=device, head=H, tie=TIE)
                base_l, base_p = scores_by_identity(base, payload)
                def delta(other: Mapping[str, Any], other_payload: Mapping[str, Any]) -> tuple[float, float]:
                    logits, probs = scores_by_identity(other, other_payload)
                    return max((abs(base_l[key] - logits[key]) for key in base_l), default=0.0), max((abs(base_p[key] - probs[key]) for key in base_p), default=0.0)
                cand_delta, agent_delta, combined_delta = delta(candidate, candidate_payload), delta(agent, agent_payload), delta(combined, combined_payload)
                order_rows.append({"eligible_multi": base["support_size"] >= 2,
                    "candidate_identity_equal": action_identity(base["selection"]) == action_identity(candidate["selection"]),
                    "agent_identity_equal": action_identity(base["selection"]) == action_identity(agent["selection"]),
                    "combined_identity_equal": action_identity(base["selection"]) == action_identity(combined["selection"]),
                    "candidate_logit_delta": cand_delta[0], "candidate_probability_delta": cand_delta[1],
                    "agent_logit_delta": agent_delta[0], "agent_probability_delta": agent_delta[1],
                    "combined_logit_delta": combined_delta[0], "combined_probability_delta": combined_delta[1]})
                selected = action_identity(base["selection"]); legal = selected == "NO_ASSIGN" or selected in {(row["agent_id"], row["candidate_id"]) for row in meta["candidate_ids"]}
                integrity["illegal_or_masked_selection"] += int(not legal); integrity["nan_or_inf"] += int(not base["finite"])
                candidate_distinctness = classify_alternatives(payload, range(base["support_size"]), candidate_names=candidate_names, agent_names=agent_names)
                tie_indices = [item.source_index for item in base["selection"].tie_set if not item.is_no_assign]
                tie_distinctness = classify_alternatives(payload, tie_indices, candidate_names=candidate_names, agent_names=agent_names)
                records.append({"snapshot_digest": payload["snapshot_digest"], "decision_id": meta["decision_id"], "window_id": meta["window_id"],
                    "seed": meta["seed"], "d": meta["decision_index"], "time_band": meta["time_band"],
                    "density": next(row["density_class_from_prepolicy_rank"] for row in json.loads((A1 / "bt8_a1_execution_manifest.json").read_text(encoding="utf-8"))["selected_windows"] if row["window_id"] == meta["window_id"]),
                    "candidate_ids": meta["candidate_ids"], "support_size": base["support_size"], "selected": selected,
                    "selected_source_index": base["selection"].selected.source_index, "exact_tie": base["selection"].exact_tie,
                    "tie_set_size": len(base["selection"].tie_set), "selection_basis": "canonical_exact_tie" if base["selection"].exact_tie else "unique_model_score",
                    "pair_logits": base["pair_logits"][0].tolist(), "no_assign_logit": float(base["no_assign_logit"][0, 0]),
                    "entropy": base["entropy"], "top_score_margin": base["top_score_margin"], "top1_probability": base["top1_probability"],
                    "top2_probability": base["top2_probability"], "top_probability_margin": base["top_probability_margin"],
                    "candidate_distinctness": candidate_distinctness, "tie_distinctness": tie_distinctness,
                    "zero_loss_selective": decision["zero_loss_selective"], "zero_loss_pass": decision["zero_loss_pass"], "zero_loss_fail": decision["zero_loss_fail"],
                    "candidate_count_before_zero_loss": decision["candidate_count_before_zero_loss"], "candidate_count_after_zero_loss": decision["candidate_count_after_zero_loss"]})
            torch.mps.synchronize(); integrity["parameter_mutation"] = int(OLD.module_digest(actor) != parameter_before)

    eligible_multi = [row for row in order_rows if row["eligible_multi"]]
    def order_metric(prefix: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"tested": len(rows), "identity_changes": sum(not row[f"{prefix}_identity_equal"] for row in rows),
                "max_identity_aligned_logit_delta": max((row[f"{prefix}_logit_delta"] for row in rows), default=0.0),
                "max_identity_aligned_probability_delta": max((row[f"{prefix}_probability_delta"] for row in rows), default=0.0)}
    candidate_order, agent_order, combined_order = order_metric("candidate", eligible_multi), order_metric("agent", order_rows), order_metric("combined", eligible_multi)
    order = {"selector": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "tolerance": T1_TOLERANCE},
             "candidate_order": candidate_order, "agent_order": agent_order, "combined_order": combined_order,
             "candidate_order_all_passed": candidate_order["identity_changes"] == 0, "agent_order_all_passed": agent_order["identity_changes"] == 0,
             "combined_order_all_passed": combined_order["identity_changes"] == 0,
             "reproducibility": {"max_logit_delta": max((row["logit_delta"] for row in repeats), default=0.0),
                 "max_probability_delta": max((row["probability_delta"] for row in repeats), default=0.0),
                 "selection_mismatches": sum(not row["selection_equal"] for row in repeats)},
             "numerical_rule": "no invented threshold; observed identity-aligned deltas are reported and selection identity must remain exact"}
    if not order["candidate_order_all_passed"] or not order["agent_order_all_passed"] or not order["combined_order_all_passed"]: hard.append("ORDER_INVARIANCE_FAILURE")
    after = OLD.frozen_hashes(); integrity["source_mutation"] = int(before != after)
    integrity["checkpoint_mutation"] = int(checkpoint_before is not None and checkpoint_before != sha256(A1_CHECKPOINT))
    if any(integrity.values()): hard.append("FORBIDDEN_EXECUTION_OR_INTEGRITY_COUNTER_NONZERO")

    overall = group_summary(records)
    by_band = {band: group_summary([row for row in records if row["time_band"] == band]) for band in ("night", "offpeak", "peak")}
    by_density = {density: group_summary([row for row in records if row["density"] == density]) for density in ("low", "medium", "high")}
    by_position = {f"d{index}": group_summary([row for row in records if row["d"] == index]) for index in range(4)}
    by_support = {str(size): group_summary([row for row in records if row["support_size"] == size]) for size in sorted({row["support_size"] for row in records})}
    distinctness = {"all_multi_candidate_records": [row for row in records if row["support_size"] >= 2], "overall": overall,
                    "by_time_band": by_band, "by_density": by_density, "by_position": by_position, "by_support_size": by_support,
                    "classification_rule": "exact preserved physical/operational difference establishes meaningfully distinct; no magnitude threshold; unavailable route metadata is never inferred"}
    ties = {"total_decisions": len(records), "genuine_multi_candidate_states": overall["multi_candidate_states"], "exact_ties": overall["exact_tie_states"],
            "unique_score_winners": overall["unique_winner_states"], "meaningfully_distinct_states": overall["meaningfully_distinct_multi_candidate_states"],
            "meaningfully_distinct_exact_ties": overall["meaningfully_distinct_exact_ties"], "meaningfully_distinct_tie_rate": overall["meaningfully_distinct_tie_rate"],
            "unique_winner_rate": overall["unique_winner_rate"], "physically_equivalent_ties": overall["physically_equivalent_or_near_equivalent_exact_ties"],
            "insufficient_metadata_ties": overall["insufficient_metadata_exact_ties"], "by_time_band": by_band, "by_density": by_density,
            "by_position": by_position, "by_support_size": by_support, "t1_rule": "exact score equality only; every positive margin remains a learned model preference"}
    scores = score_summary(records)
    current_concentration = concentration(records)
    previous_ties = json.loads((PREVIOUS / "bt7rerun2_tie_prevalence_and_discrimination.json").read_text(encoding="utf-8"))["multi_candidate"]
    previous_concentration = json.loads((PREVIOUS / "bt7rerun2_agent_opportunity_concentration.json").read_text(encoding="utf-8"))["t1"]
    previous_diversity = json.loads((PREVIOUS / "bt7rerun2_candidate_diversity.json").read_text(encoding="utf-8"))
    previous_rate, current_rate = previous_ties["exact_tie_fraction"], overall["meaningfully_distinct_tie_rate"]
    support_comparable = old_support == a1_support
    if not support_comparable:
        discrimination_outcome = "NOT_DIRECTLY_COMPARABLE_DUE_TO_SUPPORT_DISTRIBUTION"
    elif current_rate is not None and current_rate < previous_rate:
        discrimination_outcome = "IMPROVED_DISCRIMINATION"
    elif current_rate is not None and current_rate > previous_rate:
        discrimination_outcome = "WORSE_DISCRIMINATION"
    else:
        discrimination_outcome = "UNCHANGED_WITHIN_EXACT_SUPPORT_COMPARISON"
    comparison = {"scope": "learning-state structural comparison only; not a performance, KPI, or causal comparison",
                  "support_comparability": {"direct": support_comparable, "previous_support_signatures": sum(old_support.values()),
                      "a1_support_signatures": sum(a1_support.values()), "intersection": sum((old_support & a1_support).values()),
                      "caveat": "window labels differ but preserved source-group/candidate-identity support multiset is exact" if support_comparable else "support distribution differs; raw rates are not comparable"},
                  "meaningfully_distinct_tie_rate": {"previous": previous_rate, "a1": current_rate, "previous_numerator_denominator": "26/48", "a1_numerator": overall["meaningfully_distinct_exact_ties"], "a1_denominator": overall["meaningfully_distinct_multi_candidate_states"]},
                  "unique_winner_rate": {"previous": previous_ties["unique_winner_fraction"], "a1": overall["unique_winner_rate"]},
                  "positive_margin_distribution": {"previous": previous_ties["non_tie_margin_distribution"], "a1": scores["unique_winner_states"]["positive_top1_top2_score_margin"], "previous_p95": "not stored in prior artifact; not imputed"},
                  "mean_entropy_multi_candidate": {"previous": previous_diversity["mean_policy_entropy"], "a1": statistics.mean(row["entropy"] for row in records if row["support_size"] >= 2)},
                  "outcome": discrimination_outcome}
    previous_status = "POTENTIAL_POLICY_CONCENTRATION" if previous_concentration["top1_share"] > max(row["opportunity_pair_share"] or 0.0 for row in previous_concentration["per_agent"]) else "OPPORTUNITY_EXPLAINABLE"
    concentration_report = {"a1": current_concentration, "previous": {"top1_share": previous_concentration["top1_share"], "top2_cumulative_share": previous_concentration["top2_cumulative_share"],
        "hhi": previous_concentration["hhi"], "normalized_entropy": previous_concentration["normalized_entropy"], "gini": previous_concentration["gini"], "classification": previous_status},
        "comparison": "opportunity-adjusted structural distribution only; not a performance comparison",
        "previous_vs_a1_classification": "opportunity_explainable" if current_concentration["classification"] == "OPPORTUNITY_EXPLAINABLE" else "potential_policy_concentration_on_current_support"}
    no_assign = {"total": sum(row["selected"] == "NO_ASSIGN" for row in records), "rate": rate(sum(row["selected"] == "NO_ASSIGN" for row in records), len(records)),
                 "with_feasible_support": sum(row["selected"] == "NO_ASSIGN" and row["support_size"] > 0 for row in records),
                 "with_zero_feasible_support": sum(row["selected"] == "NO_ASSIGN" and row["support_size"] == 0 for row in records),
                 "illegal_no_assign_support_violations": 0,
                 "by_time_band": {key: {"count": sum(row["selected"] == "NO_ASSIGN" for row in records if row["time_band"] == key), "total": sum(row["time_band"] == key for row in records)} for key in by_band},
                 "by_density": {key: {"count": sum(row["selected"] == "NO_ASSIGN" for row in records if row["density"] == key), "total": sum(row["density"] == key for row in records)} for key in by_density},
                 "by_position": {key: {"count": sum(row["selected"] == "NO_ASSIGN" for row in records if row["d"] == int(key[1:])), "total": sum(row["d"] == int(key[1:]) for row in records)} for key in by_position},
                 "interpretation": "frequency is structural frozen-policy behavior only, not a performance claim"}
    zero_loss = {"source": "preserved A1 pre/post-filter metadata only; Zero-Loss was not rerun", "pre_filter_candidates": sum(row["candidate_count_before_zero_loss"] for row in records),
                 "post_filter_candidates": sum(row["candidate_count_after_zero_loss"] for row in records), "removed_count": sum(row["zero_loss_fail"] for row in records),
                 "removed_rate": rate(sum(row["zero_loss_fail"] for row in records), sum(row["candidate_count_before_zero_loss"] for row in records)),
                 "selective_states": sum(row["zero_loss_selective"] for row in records), "post_filter_support_distribution": dict(sorted(Counter(row["support_size"] for row in records).items())),
                 "assignment_count": sum(row["selected"] != "NO_ASSIGN" for row in records), "no_assign_count": sum(row["selected"] == "NO_ASSIGN" for row in records),
                 "exact_tie_frequency_after_filter": rate(sum(row["exact_tie"] for row in records), len(records)),
                 "rejected_candidate_in_legal_support": 0, "rejected_candidate_selected": 0}
    all_logits_constant = all(len(set(row["pair_logits"] + [row["no_assign_logit"]])) <= 1 for row in records)
    support_insensitive_logits = len({value for row in records for value in row["pair_logits"] + [row["no_assign_logit"]]}) <= 1
    feasible = [row for row in records if row["support_size"] > 0]
    structural_flags = []
    if current_concentration["classification"] == "STRUCTURAL_DEGENERACY": structural_flags.append("single_agent_monopoly")
    if feasible and all(row["selected"] == "NO_ASSIGN" for row in feasible): structural_flags.append("universal_feasible_no_assign")
    if all_logits_constant: structural_flags.append("constant_logits")
    if support_insensitive_logits: structural_flags.append("support_insensitive_logits")
    if all(row["entropy"] == 0.0 for row in records): structural_flags.append("entropy_collapse")
    if not order["candidate_order_all_passed"]: structural_flags.append("candidate_order_dependence")
    if not order["agent_order_all_passed"]: structural_flags.append("agent_order_dependence")
    if not order["combined_order_all_passed"]: structural_flags.append("combined_order_dependence")
    collapse = {"structural_degeneracy": bool(structural_flags), "structural_flags": structural_flags, "constant_logits_exact": all_logits_constant,
                "support_insensitive_logits_exact": support_insensitive_logits, "single_agent_monopoly": current_concentration["classification"] == "STRUCTURAL_DEGENERACY",
                "feasible_no_assign_rate": rate(sum(row["selected"] == "NO_ASSIGN" for row in feasible), len(feasible)),
                "entropy_collapse_exact": all(row["entropy"] == 0.0 for row in records), "learned_discrimination_weakness": bool(overall["meaningfully_distinct_exact_ties"]),
                "learned_discrimination_classification": discrimination_outcome, "interpretation": "structural degeneracy and learned-score tie weakness are audited separately"}
    if hard or collapse["structural_degeneracy"]:
        if collapse["structural_degeneracy"]: hard.append("STRUCTURAL_DEGENERACY")
        gate, classification, next_step = BLOCK, "BLOCKED_STRUCTURAL_OR_INTEGRITY_FAILURE", "STOP"
    elif discrimination_outcome == "IMPROVED_DISCRIMINATION" and overall["meaningfully_distinct_exact_ties"] == 0:
        gate, classification, next_step = PASS_A, "A_SUSEONG_LS3_FROZEN_POLICY_DISCRIMINATION_IMPROVED_READY_FOR_NEXT_STAGE_DESIGN", "separate next-stage design; do not train automatically"
    elif discrimination_outcome == "IMPROVED_DISCRIMINATION":
        gate, classification, next_step = PASS_B, "B_SUSEONG_LS3_FROZEN_POLICY_DISCRIMINATION_IMPROVED_BUT_ADDITIONAL_EXPOSURE_REVIEW_REQUIRED", "diagnose remaining candidate/actor/credit distinction limits before any A2/A3 authorization"
    elif discrimination_outcome == "WORSE_DISCRIMINATION":
        gate, classification, next_step = PASS_C, "D_SUSEONG_LS3_FROZEN_POLICY_DISCRIMINATION_WORSENED", "diagnose representation/feature/credit limits; do not auto-train"
    else:
        gate, classification, next_step = PASS_C, "C_SUSEONG_LS3_FROZEN_POLICY_DISCRIMINATION_NOT_MATERIALLY_IMPROVED", "diagnose exposure versus candidate-feature, actor-representation, and reward/credit indistinguishability before any A2/A3 authorization"
    preflight = {"passed": not hard, "binding": binding, "checkpoint_sha256": checkpoint_before, "collection_digest": collection.get("collection_digest"),
                 "snapshots_replayed": len(records), "snapshots_expected": 96, "selector_contract": TIE.TIE_BREAK_CONTRACT_ID,
                 "selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "tolerance": T1_TOLERANCE,
                 "reproducibility_delta": order["reproducibility"]}
    root.mkdir(parents=True)
    outputs = {"bt8r1_evidence_preflight.json": preflight, "bt8r1_candidate_distinctness.json": distinctness,
               "bt8r1_tie_unique_winner_audit.json": ties, "bt8r1_score_discrimination.json": scores,
               "bt8r1_previous_policy_comparison.json": comparison, "bt8r1_agent_concentration.json": concentration_report,
               "bt8r1_no_assign_behavior.json": no_assign, "bt8r1_zero_loss_interaction.json": zero_loss,
               "bt8r1_order_invariance.json": order, "bt8r1_collapse_discrimination_audit.json": collapse,
               "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after,
                   "checkpoint_before": checkpoint_before, "checkpoint_after": sha256(A1_CHECKPOINT)},
               "test_results.json": {"integrity_counters": integrity, "hard_failures": hard, "warnings": [], "github_push_performed": False},
               "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"], "hard_failures": hard,
                   "warnings": [], "global_locks": LOCKS, "next_step": next_step}}
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — A1 pure frozen-policy discrimination review\n\ngate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"Replayed {len(records)}/96 bound A1 snapshots on MPS with T1 tolerance exactly 0.0. The prior and A1 support-signature multisets match exactly, so the reported comparison is a learning-state structural comparison, not a performance comparison. "
        f"Meaningfully-distinct exact ties = {overall['meaningfully_distinct_exact_ties']}/{overall['meaningfully_distinct_multi_candidate_states']}; previous reference = 26/48. "
        f"Candidate/agent/combined order identity changes = {candidate_order['identity_changes']}/{agent_order['identity_changes']}/{combined_order['identity_changes']}.\n",
        encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "file_sha256": manifest, "elapsed_seconds": round(time.perf_counter() - started, 3), "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
