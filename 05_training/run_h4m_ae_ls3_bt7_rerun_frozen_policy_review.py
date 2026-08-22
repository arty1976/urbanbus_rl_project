#!/usr/bin/env python3
"""BT7-RERUN: pure frozen Joint Assignment behavior characterization.

The only policy inputs are the 96 device-neutral snapshots emitted by the
authoritative BT6-RERUN artifact.  Companion decision metadata is joined by its
snapshot digest solely for pre-preserved time-band, density, and Zero-Loss
support descriptions.  This module never rebuilds a candidate or calls the
simulator, Local Search, Zero-Loss, reward, PPO, or an optimizer.
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
from typing import Any, Dict, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch


STAGE = "H4M-AE-R9.8-LS3-BT7-RERUN"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT7_RERUN_FROZEN_POLICY_BEHAVIOR_REVIEW_COMPLETE"
BLOCK_BINDING = "BLOCKED_FROZEN_POLICY_EVIDENCE_BINDING_MISMATCH"
BLOCK_DEGENERACY = "BLOCKED_FROZEN_POLICY_STRUCTURAL_DEGENERACY"
BT6_SOURCE = "183e0dae6075e3038305686b151db0c3767212c8"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_200221+09:00"
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
PERMUTATION_TOLERANCE = 1e-4  # Existing Joint Assignment identity-aligned permutation contract.


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


def frozen_hashes() -> Dict[str, str]:
    hashes = {name: sha256(ROOT / rel) for name, rel in FROZEN.items()}
    hashes["joint_actor_head"] = sha256(ROOT / "multi_agent_candidate_assignment_head.py")
    return hashes


def provenance() -> Dict[str, Any]:
    head, parent = git(["rev-parse", "HEAD"]), git(["rev-parse", "HEAD^"])
    changed = [row for row in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if row]
    return {"source_commit": head, "source_parent": parent, "changed_files": changed,
            "source_only_local_commit": changed == [SOURCE_REL.as_posix()], "github_push_performed": False}


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.mean(values)) if values else None


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.median(values)) if values else None


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def density_label(rank: int) -> str:
    # BT6-S0 froze four ascending within-band ranks.  This reporting-only map
    # uses no reward/KPI outcome: endpoints are low/high, inner ranks medium.
    return "low" if rank == 0 else "high" if rank == 3 else "medium"


def forward(actor: torch.nn.Module, payload: Mapping[str, Any], *, device: torch.device, head: Any) -> Dict[str, Any]:
    metadata, cpu_tensors = payload["metadata"], payload["tensors"]
    tensors = {name: value.to(device) for name, value in cpu_tensors.items()}
    selectable = int(metadata["selectable_pair_count"])
    with torch.no_grad():
        raw_pair_logits, no_assign_logit = actor(
            global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
            agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
            candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"],
            safe_mask=tensors["safe_mask"])
        pair_logits = raw_pair_logits[:, :selectable]
        safe_mask = tensors["safe_mask"][:, :selectable]
        probabilities = head.masked_distribution(pair_logits, no_assign_logit, safe_mask)
        selected = int(torch.argmax(probabilities[0]).item())
    finite = bool(torch.isfinite(no_assign_logit).all().item() and torch.isfinite(probabilities).all().item()
                  and (selectable == 0 or torch.isfinite(pair_logits).all().item()))
    ordered = torch.sort(probabilities[0], descending=True).values
    top1, top2 = float(ordered[0].detach().cpu()), float(ordered[1].detach().cpu()) if len(ordered) > 1 else 0.0
    entropy = float((-(probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))).sum()).detach().cpu())
    selected_is_no_assign = selected == selectable
    candidate_ids = metadata["candidate_ids"]
    selected_pair = None if selected_is_no_assign else candidate_ids[selected]
    return {"pair_logits": pair_logits.detach().cpu(), "no_assign_logit": no_assign_logit.detach().cpu(),
            "probabilities": probabilities.detach().cpu(), "argmax_index": selected,
            "selected_is_no_assign": selected_is_no_assign, "selected_pair": selected_pair,
            "support_size": selectable, "top1_probability": top1, "top2_probability": top2,
            "top1_top2_margin": top1 - top2, "entropy": entropy, "finite": finite}


def candidate_permutation(actor: torch.nn.Module, payload: Mapping[str, Any], base: Mapping[str, Any], *, device: torch.device, head: Any) -> Dict[str, Any]:
    metadata, source = payload["metadata"], payload["tensors"]
    p = int(metadata["selectable_pair_count"])
    if p < 2:
        return {"tested": False, "passed": True, "max_logit_delta": 0.0, "max_probability_delta": 0.0,
                "argmax_identity_equal": True}
    full = int(source["candidate_feats"].shape[1])
    permutation = list(reversed(range(p))) + list(range(p, full))
    index = torch.tensor(permutation, dtype=torch.long)
    changed = dict(source)
    for name in ("candidate_feats", "pair_agent_index", "safe_mask"):
        changed[name] = source[name].index_select(1, index)
    candidate_ids = [metadata["candidate_ids"][i] for i in permutation[:p]]
    changed_metadata = dict(metadata)
    changed_metadata["candidate_ids"] = candidate_ids
    changed_metadata["candidate_order"] = candidate_ids
    changed_payload = {"metadata": changed_metadata, "tensors": changed, "snapshot_digest": payload["snapshot_digest"]}
    got = forward(actor, changed_payload, device=device, head=head)
    base_by_id = {(row["agent_id"], row["candidate_id"]): float(base["pair_logits"][0, i])
                  for i, row in enumerate(metadata["candidate_ids"])}
    got_by_id = {(row["agent_id"], row["candidate_id"]): float(got["pair_logits"][0, i])
                 for i, row in enumerate(candidate_ids)}
    logit_delta = max(abs(base_by_id[key] - got_by_id[key]) for key in base_by_id)
    base_prob_by_id = {(row["agent_id"], row["candidate_id"]): float(base["probabilities"][0, i])
                       for i, row in enumerate(metadata["candidate_ids"])}
    got_prob_by_id = {(row["agent_id"], row["candidate_id"]): float(got["probabilities"][0, i])
                      for i, row in enumerate(candidate_ids)}
    prob_delta = max(abs(base_prob_by_id[key] - got_prob_by_id[key]) for key in base_prob_by_id)
    base_argmax = "NO_ASSIGN" if base["selected_is_no_assign"] else (base["selected_pair"]["agent_id"], base["selected_pair"]["candidate_id"])
    got_argmax = "NO_ASSIGN" if got["selected_is_no_assign"] else (got["selected_pair"]["agent_id"], got["selected_pair"]["candidate_id"])
    return {"tested": True, "passed": logit_delta <= PERMUTATION_TOLERANCE and prob_delta <= PERMUTATION_TOLERANCE
            and base_argmax == got_argmax, "max_logit_delta": logit_delta, "max_probability_delta": prob_delta,
            "argmax_identity_equal": base_argmax == got_argmax}


def agent_permutation(actor: torch.nn.Module, payload: Mapping[str, Any], base: Mapping[str, Any], *, device: torch.device, head: Any) -> Dict[str, Any]:
    metadata, source = payload["metadata"], payload["tensors"]
    agents = int(source["agent_feats"].shape[1])
    if agents < 2:
        return {"tested": False, "passed": True, "max_logit_delta": 0.0, "max_probability_delta": 0.0,
                "argmax_identity_equal": True}
    permutation = list(reversed(range(agents)))
    index = torch.tensor(permutation, dtype=torch.long)
    inverse = torch.empty(agents, dtype=torch.long)
    inverse[index] = torch.arange(agents, dtype=torch.long)
    changed = dict(source)
    changed["agent_feats"] = source["agent_feats"].index_select(1, index)
    changed["agent_mask"] = source["agent_mask"].index_select(1, index)
    changed["pair_agent_index"] = inverse[source["pair_agent_index"]]
    changed_payload = {"metadata": metadata, "tensors": changed, "snapshot_digest": payload["snapshot_digest"]}
    got = forward(actor, changed_payload, device=device, head=head)
    logit_delta = float((base["pair_logits"] - got["pair_logits"]).abs().max()) if base["support_size"] else 0.0
    prob_delta = float((base["probabilities"] - got["probabilities"]).abs().max())
    base_argmax = "NO_ASSIGN" if base["selected_is_no_assign"] else (base["selected_pair"]["agent_id"], base["selected_pair"]["candidate_id"])
    got_argmax = "NO_ASSIGN" if got["selected_is_no_assign"] else (got["selected_pair"]["agent_id"], got["selected_pair"]["candidate_id"])
    return {"tested": True, "passed": logit_delta <= PERMUTATION_TOLERANCE and prob_delta <= PERMUTATION_TOLERANCE
            and base_argmax == got_argmax, "max_logit_delta": logit_delta, "max_probability_delta": prob_delta,
            "argmax_identity_equal": base_argmax == got_argmax}


def selection_concentration(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    opportunities, exposure_snapshots, selections = Counter(), Counter(), Counter()
    for row in rows:
        present = set()
        for pair in row["candidate_ids"]:
            agent = pair["agent_id"]
            opportunities[agent] += 1
            present.add(agent)
        for agent in present:
            exposure_snapshots[agent] += 1
        if not row["selected_is_no_assign"]:
            selections[row["selected_pair"]["agent_id"]] += 1
    agents = sorted(opportunities)
    assigned = sum(selections.values())
    total_opportunities = sum(opportunities.values())
    values = [selections[agent] for agent in agents]
    shares = [value / assigned for value in values] if assigned else []
    entropy = -sum(value * math.log(value) for value in shares if value) if shares else 0.0
    normalized_entropy = entropy / math.log(len(agents)) if len(agents) > 1 and shares else None
    ordered = sorted(((selections[agent] / assigned if assigned else 0.0, agent) for agent in agents), reverse=True)
    top1 = ordered[0][0] if ordered else None
    top2 = sum(value for value, _ in ordered[:2]) if ordered else None
    if assigned and len(values) > 1:
        sorted_values = sorted(values)
        gini = sum((2 * (index + 1) - len(values) - 1) * value for index, value in enumerate(sorted_values)) / (len(values) * sum(values))
    else:
        gini = 0.0 if assigned else None
    per_agent = []
    for agent in agents:
        selected = selections[agent]
        per_agent.append({"agent_id": agent, "candidate_opportunity_count": opportunities[agent],
                          "snapshot_exposure_count": exposure_snapshots[agent], "selection_count": selected,
                          "opportunity_pair_share": rate(opportunities[agent], total_opportunities),
                          "selection_share": rate(selected, assigned),
                          "selection_share_conditional_on_snapshot_exposure": rate(selected, exposure_snapshots[agent])})
    if not assigned:
        classification = "STRUCTURAL_DEGENERACY" if total_opportunities else "INSUFFICIENT_EVIDENCE"
        evidence = "no assignment was selected despite preserved feasible opportunities" if total_opportunities else "no feasible opportunity"
    elif len([value for value in values if value]) == 1 and len(agents) > 1:
        classification, evidence = "STRUCTURAL_DEGENERACY", "one agent received every non-NO_ASSIGN frozen selection while other agents had preserved opportunities"
    else:
        leader = ordered[0][1]
        leader_selection = selections[leader] / assigned
        leader_opportunity = opportunities[leader] / total_opportunities
        if leader_selection > leader_opportunity:
            classification, evidence = "POTENTIAL_POLICY_CONCENTRATION", "leader selection share exceeds its candidate-opportunity share"
        elif len([value for value in values if value]) >= 2:
            classification, evidence = "HEALTHY_DIVERSITY", "multiple agents are selected and no leader exceeds its candidate-opportunity share"
        else:
            classification, evidence = "CONCENTRATED_BUT_OPPORTUNITY_EXPLAINABLE", "selection distribution is bounded by preserved opportunity exposure"
    return {"non_no_assign_selection_count": assigned, "agents_with_opportunity": len(agents), "per_agent": per_agent,
            "top1_agent_share": top1, "top2_agent_share": top2, "hhi": sum(value * value for value in shares) if shares else None,
            "normalized_entropy": normalized_entropy, "gini": gini, "classification": classification, "evidence": evidence}


def behavior_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    no_assign = sum(row["selected_is_no_assign"] for row in rows)
    feasible = [row for row in rows if row["support_size"] >= 1]
    multi = [row for row in rows if row["support_size"] >= 2]
    multi_assigned = [row for row in multi if not row["selected_is_no_assign"]]
    first = sum(row["argmax_index"] == 0 for row in multi_assigned)
    support_counts = Counter(row["support_size"] for row in rows)
    return {"decision_count": len(rows), "no_assign_count": no_assign, "no_assign_rate": rate(no_assign, len(rows)),
            "feasible_support_count": len(feasible), "feasible_support_no_assign_count": sum(row["selected_is_no_assign"] for row in feasible),
            "feasible_support_no_assign_rate": rate(sum(row["selected_is_no_assign"] for row in feasible), len(feasible)),
            "support_size_distribution": dict(sorted(support_counts.items())), "mean_legal_support_size": mean(row["support_size"] for row in rows),
            "median_legal_support_size": median(row["support_size"] for row in rows), "mean_policy_entropy": mean(row["entropy"] for row in rows),
            "mean_top1_probability": mean(row["top1_probability"] for row in rows),
            "mean_top1_top2_margin": mean(row["top1_top2_margin"] for row in rows),
            "multi_candidate_state_count": len(multi), "multi_candidate_assignment_count": len(multi_assigned),
            "first_listed_candidate_selection_count": first, "first_listed_candidate_selection_rate": rate(first, len(multi_assigned)),
            "non_first_candidate_selection_count": len(multi_assigned) - first,
            "non_first_candidate_selection_rate": rate(len(multi_assigned) - first, len(multi_assigned)),
            "agent_concentration": selection_concentration(rows)}


def collapse_audit(rows: Sequence[Mapping[str, Any]], concentration: Mapping[str, Any], invariance: Mapping[str, Any]) -> Dict[str, Any]:
    feasible = [row for row in rows if row["support_size"] >= 1]
    multi = [row for row in rows if row["support_size"] >= 2]
    multi_assigned = [row for row in multi if not row["selected_is_no_assign"]]
    logits_constant = all((not row["pair_logits"] or max(row["pair_logits"]) == min(row["pair_logits"]))
                          and row["no_assign_logit"] == (row["pair_logits"][0] if row["pair_logits"] else row["no_assign_logit"])
                          for row in rows)
    zero_entropy = all(row["entropy"] == 0.0 for row in rows)
    universal_feasible_no_assign = bool(feasible) and all(row["selected_is_no_assign"] for row in feasible)
    universal_feasible_assignment = bool(feasible) and all(not row["selected_is_no_assign"] for row in feasible)
    positional_collapse = bool(multi_assigned) and all(row["argmax_index"] == 0 for row in multi_assigned)
    issues = []
    if concentration["classification"] == "STRUCTURAL_DEGENERACY": issues.append("single_agent_monopoly_or_zero_assignment")
    if universal_feasible_no_assign: issues.append("universal_no_assign_with_feasible_support")
    if logits_constant: issues.append("constant_logits")
    if zero_entropy: issues.append("zero_entropy")
    if not invariance["candidate_order_all_passed"]: issues.append("candidate_order_dependence")
    if not invariance["agent_order_all_passed"]: issues.append("agent_order_dependence")
    if issues:
        classification = "STRUCTURAL_DEGENERACY"
    elif concentration["classification"] in {"POTENTIAL_POLICY_CONCENTRATION", "CONCENTRATED_BUT_OPPORTUNITY_EXPLAINABLE"} or positional_collapse:
        classification = "REVIEWABLE_WITH_CONCENTRATION_CAUTION"
    else:
        classification = "DIVERSE_AND_REVIEWABLE"
    return {"classification": classification, "structural_issue_flags": issues,
            "single_agent_monopoly": concentration["classification"] == "STRUCTURAL_DEGENERACY",
            "universal_feasible_no_assign": universal_feasible_no_assign,
            "universal_feasible_assignment": universal_feasible_assignment,
            "first_listed_candidate_all_multi_assignments": positional_collapse,
            "constant_logits_exact": logits_constant, "zero_entropy_exact": zero_entropy,
            "distinct_support_sizes": sorted({row["support_size"] for row in rows}),
            "support_insensitive_action_type_exact": len({"NO_ASSIGN" if row["selected_is_no_assign"] else "ASSIGN" for row in feasible}) <= 1,
            "evidence_rule": "no arbitrary near-universal threshold: only exact structural conditions and opportunity-adjusted concentration are classified"}


def main() -> None:
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import multi_agent_candidate_assignment_head as H

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_rerun_frozen_policy_behavior_review_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists(): raise SystemExit("append-only artifact collision")
    before = frozen_hashes()
    source = provenance()
    hard: list[str] = []
    if not torch.backends.mps.is_available(): hard.append("MPS_FROZEN_REPLAY_ENVIRONMENT_UNAVAILABLE")
    bt6_gate = json.loads((BT6 / "gate_decision.json").read_text(encoding="utf-8"))
    decisions = json.loads((BT6 / "bt6_candidate_support_audit.json").read_text(encoding="utf-8"))["decisions"]
    try:
        collection = FPS.load_collection_manifest(COLLECTION_PATH)
        snapshots = [(entry, FPS.load_snapshot(SNAPSHOT_ROOT / entry["relative_path"])) for entry in collection["entries"]]
    except Exception as exc:  # noqa: BLE001
        collection, snapshots = {}, []
        hard.append(f"SNAPSHOT_COLLECTION_OR_DIGEST_INVALID:{type(exc).__name__}")
    checkpoint_sha_before = sha256(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    expected_config: Dict[str, Any] = {}
    binding: Dict[str, Any] = {"authoritative_bt6_only": BT6.is_dir(),
                               "bt6_gate": bt6_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_POST_REPAIR_EXTENDED_BOUNDED_TRAINING_AND_BT7_READINESS_COMPLETE",
                               "bt6_source": bt6_gate.get("source_commit") == BT6_SOURCE,
                               "source_only_local_commit": source["source_only_local_commit"],
                               "checkpoint_exists": CHECKPOINT_PATH.is_file()}
    if collection:
        cfg = collection.get("actor_config", {})
        expected_config = FPS.actor_config(global_dim=cfg.get("global_dim", -1), demand_dim=cfg.get("demand_dim", -1),
                                            agent_dim=cfg.get("agent_dim", -1), candidate_dim=cfg.get("candidate_dim", -1),
                                            hidden=cfg.get("hidden", -1), heads=cfg.get("heads", -1),
                                            actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
                                            actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
        binding.update({"snapshot_schema": collection.get("snapshot_schema_version") == FPS.SNAPSHOT_SCHEMA_VERSION,
                        "collection_schema": collection.get("collection_schema_version") == FPS.COLLECTION_SCHEMA_VERSION,
                        "collection_count_96": collection.get("snapshot_count") == len(collection.get("entries", [])) == 96,
                        "unique_snapshot_digests": len({entry.get("snapshot_digest") for entry in collection.get("entries", [])}) == 96,
                        "checkpoint_sha": checkpoint_sha_before == collection.get("checkpoint_sha256"),
                        "actor_config": cfg == expected_config and collection.get("actor_config_sha256") == FPS.actor_config_sha256(expected_config),
                        "frozen_authorities": collection.get("frozen_authority_hashes") == before,
                        "captured_device": collection.get("captured_device") == "mps"})
    snapshot_by_digest = {payload["snapshot_digest"]: payload for _, payload in snapshots}
    decision_by_digest = {row.get("frozen_actor_snapshot_digest"): row for row in decisions}
    binding.update({"all_snapshots_loaded": len(snapshots) == 96, "decision_metadata_count_96": len(decisions) == 96,
                    "decision_snapshot_one_to_one": len(decision_by_digest) == 96 and set(decision_by_digest) == set(snapshot_by_digest)})
    if not all(binding.values()): hard.append(BLOCK_BINDING)
    records: list[Dict[str, Any]] = []
    candidate_tests, agent_tests = [], []
    reproducibility_logit, reproducibility_probability = [], []
    integrity = {name: 0 for name in ("optimizer_steps", "causal_simulator_rollouts", "candidate_regeneration", "zero_loss_reevaluation",
                                      "parameter_mutation", "source_mutation", "checkpoint_mutation", "candidate_identity_mismatch",
                                      "illegal_or_masked_selection", "nan_or_inf", "TEST6_access", "github_push", "feature_recomputation",
                                      "mask_reconstruction", "candidate_order_dependence", "agent_order_dependence")}
    if not hard:
        device = torch.device("mps")
        actor = H.MultiAgentCandidateAssignmentHead(global_dim=expected_config["global_dim"], demand_dim=expected_config["demand_dim"],
                                                    agent_dim=expected_config["agent_dim"], candidate_dim=expected_config["candidate_dim"],
                                                    hidden=expected_config["hidden"], heads=expected_config["heads"]).to(device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or "actor" not in checkpoint:
            hard.append("CHECKPOINT_ACTOR_STATE_MISSING")
        else:
            actor.load_state_dict(checkpoint["actor"], strict=True)
            actor.eval()
            param_before = module_digest(actor)
            for entry, payload in snapshots:
                meta = payload["metadata"]
                decision = decision_by_digest[payload["snapshot_digest"]]
                result = forward(actor, payload, device=device, head=H)
                repeat = forward(actor, payload, device=device, head=H)
                reproducibility_logit.append(float((result["pair_logits"] - repeat["pair_logits"]).abs().max()) if result["support_size"] else 0.0)
                reproducibility_probability.append(float((result["probabilities"] - repeat["probabilities"]).abs().max()))
                candidate_tests.append(candidate_permutation(actor, payload, result, device=device, head=H))
                agent_tests.append(agent_permutation(actor, payload, result, device=device, head=H))
                candidate_ids = meta["candidate_ids"]
                candidate_identity = [(row["agent_id"], row["candidate_id"]) for row in candidate_ids]
                integrity["candidate_identity_mismatch"] += int(decision["candidate_count_after_zero_loss"] != result["support_size"]
                                                                  or decision["candidate_identity"] != [list(pair) for pair in candidate_identity])
                integrity["illegal_or_masked_selection"] += int(not result["selected_is_no_assign"] and result["argmax_index"] >= result["support_size"])
                integrity["nan_or_inf"] += int(not result["finite"])
                records.append({"snapshot_digest": payload["snapshot_digest"], "decision_id": meta["decision_id"], "window_id": meta["window_id"],
                                "seed": meta["seed"], "decision_index": meta["decision_index"], "time_band": meta["time_band"],
                                "request_density_rank": decision["request_density_stratum"], "request_density": density_label(decision["request_density_stratum"]),
                                "candidate_ids": candidate_ids, "agent_ids": meta["agent_ids"], "support_size": result["support_size"],
                                "no_assign_index": meta["no_assign_index"], "selected_is_no_assign": result["selected_is_no_assign"],
                                "selected_pair": result["selected_pair"], "argmax_index": result["argmax_index"],
                                "pair_logits": result["pair_logits"][0].tolist(), "no_assign_logit": float(result["no_assign_logit"][0, 0]),
                                "probabilities": result["probabilities"][0].tolist(), "entropy": result["entropy"],
                                "top1_probability": result["top1_probability"], "top2_probability": result["top2_probability"],
                                "top1_top2_margin": result["top1_top2_margin"], "zero_loss_pass": decision["zero_loss_pass"],
                                "zero_loss_fail": decision["zero_loss_fail"], "zero_loss_selective": decision["zero_loss_selective"],
                                "candidate_count_before_zero_loss": decision["candidate_count_before_zero_loss"],
                                "candidate_count_after_zero_loss": decision["candidate_count_after_zero_loss"]})
            torch.mps.synchronize()
            integrity["parameter_mutation"] = int(module_digest(actor) != param_before)
    candidate_invariance = {"tested": sum(item["tested"] for item in candidate_tests), "total": len(candidate_tests),
                            "all_passed": all(item["passed"] for item in candidate_tests),
                            "max_logit_delta": max((item["max_logit_delta"] for item in candidate_tests), default=None),
                            "max_probability_delta": max((item["max_probability_delta"] for item in candidate_tests), default=None),
                            "argmax_identity_mismatches": sum(not item["argmax_identity_equal"] for item in candidate_tests)}
    agent_invariance = {"tested": sum(item["tested"] for item in agent_tests), "total": len(agent_tests),
                        "all_passed": all(item["passed"] for item in agent_tests),
                        "max_logit_delta": max((item["max_logit_delta"] for item in agent_tests), default=None),
                        "max_probability_delta": max((item["max_probability_delta"] for item in agent_tests), default=None),
                        "argmax_identity_mismatches": sum(not item["argmax_identity_equal"] for item in agent_tests)}
    # Stored candidate identities matched their bound snapshot metadata above.
    # A changed deterministic argmax identity after a *permuted input* is a
    # separate policy order-dependence finding, not evidence corruption.
    integrity["candidate_order_dependence"] = candidate_invariance["argmax_identity_mismatches"]
    integrity["agent_order_dependence"] = agent_invariance["argmax_identity_mismatches"]
    if not candidate_invariance["all_passed"] or not agent_invariance["all_passed"]: hard.append("ORDER_INVARIANCE_FAILURE")
    after = frozen_hashes()
    integrity["source_mutation"] = int(before != after)
    integrity["checkpoint_mutation"] = int(checkpoint_sha_before is not None and checkpoint_sha_before != sha256(CHECKPOINT_PATH))
    if any(integrity.values()): hard.append("FORBIDDEN_EXECUTION_OR_INTEGRITY_COUNTER_NONZERO")
    overall = behavior_summary(records)
    by_band = {band: behavior_summary([row for row in records if row["time_band"] == band]) for band in ("night", "offpeak", "peak")}
    by_density = {label: behavior_summary([row for row in records if row["request_density"] == label]) for label in ("low", "medium", "high")}
    by_position = {f"d{index}": behavior_summary([row for row in records if row["decision_index"] == index]) for index in range(4)}
    multi = [row for row in records if row["support_size"] >= 2]
    multi_assigned = [row for row in multi if not row["selected_is_no_assign"]]
    pattern_counter = Counter(tuple((pair["agent_id"], pair["candidate_id"]) for pair in row["candidate_ids"]) +
                              (("NO_ASSIGN", "NO_ASSIGN") if row["selected_is_no_assign"] else
                               (row["selected_pair"]["agent_id"], row["selected_pair"]["candidate_id"]),) for row in multi)
    candidate_diversity = {"states_analyzed": len(multi), "support_size_distribution": dict(sorted(Counter(row["support_size"] for row in multi).items())),
                           "assigned_states": len(multi_assigned), "first_listed_candidate_selected_count": sum(row["argmax_index"] == 0 for row in multi_assigned),
                           "first_listed_candidate_selected_rate": rate(sum(row["argmax_index"] == 0 for row in multi_assigned), len(multi_assigned)),
                           "non_first_candidate_selected_count": sum(row["argmax_index"] > 0 for row in multi_assigned),
                           "non_first_candidate_selected_rate": rate(sum(row["argmax_index"] > 0 for row in multi_assigned), len(multi_assigned)),
                           "no_assign_count": sum(row["selected_is_no_assign"] for row in multi), "unique_candidate_selection_patterns": len(pattern_counter),
                           "top_patterns": [{"pattern": [list(pair) for pair in key], "count": value} for key, value in pattern_counter.most_common(10)],
                           "mean_policy_entropy": mean(row["entropy"] for row in multi), "mean_top1_probability": mean(row["top1_probability"] for row in multi),
                           "mean_top2_probability": mean(row["top2_probability"] for row in multi), "mean_top1_top2_margin": mean(row["top1_top2_margin"] for row in multi),
                           "candidate_order_invariant": candidate_invariance["all_passed"]}
    selective = [row for row in records if row["zero_loss_selective"]]
    zero_loss = {"selective_states_analyzed": len(selective), "pre_filter_candidate_count": sum(row["candidate_count_before_zero_loss"] for row in records),
                 "post_filter_candidate_count": sum(row["candidate_count_after_zero_loss"] for row in records),
                 "removed_candidate_count": sum(row["zero_loss_fail"] for row in records),
                 "removed_candidate_fraction": rate(sum(row["zero_loss_fail"] for row in records), sum(row["candidate_count_before_zero_loss"] for row in records)),
                 "assignment_count_after_filtering": sum(not row["selected_is_no_assign"] for row in records),
                 "assignment_rate_after_filtering": rate(sum(not row["selected_is_no_assign"] for row in records), len(records)),
                 "no_assign_count_after_filtering": sum(row["selected_is_no_assign"] for row in records),
                 "no_assign_rate_after_filtering": rate(sum(row["selected_is_no_assign"] for row in records), len(records)),
                 "entropy_by_post_filter_support_size": {str(size): mean(row["entropy"] for row in records if row["support_size"] == size)
                                                        for size in sorted({row["support_size"] for row in records})},
                 "zero_loss_fail_candidate_in_legal_support": 0, "zero_loss_fail_candidate_selected": 0,
                 "evidence": "every actor candidate identity is the persisted post-Zero-Loss support; no Zero-Loss recomputation was performed"}
    invariance = {"permutation_tolerance": PERMUTATION_TOLERANCE, "candidate_order": candidate_invariance,
                  "agent_order": agent_invariance, "candidate_order_all_passed": candidate_invariance["all_passed"],
                  "agent_order_all_passed": agent_invariance["all_passed"],
                  "replay_reproducible": max(reproducibility_logit, default=0.0) == 0.0 and max(reproducibility_probability, default=0.0) == 0.0,
                  "repeat_max_logit_delta": max(reproducibility_logit, default=None),
                  "repeat_max_probability_delta": max(reproducibility_probability, default=None)}
    collapse = collapse_audit(records, overall["agent_concentration"], invariance)
    if collapse["classification"] == "STRUCTURAL_DEGENERACY": hard.append(BLOCK_DEGENERACY)
    if hard:
        gate = BLOCK_DEGENERACY if BLOCK_DEGENERACY in hard else BLOCK_BINDING
        classification = "FROZEN_POLICY_REVIEW_BLOCKED"
    else:
        gate = PASS_GATE
        classification = ("A_SUSEONG_LS3_FROZEN_JOINT_ASSIGNMENT_BEHAVIOR_DIVERSE_AND_REVIEWABLE"
                          if collapse["classification"] == "DIVERSE_AND_REVIEWABLE"
                          else "B_SUSEONG_LS3_FROZEN_JOINT_ASSIGNMENT_BEHAVIOR_REVIEWABLE_WITH_CONCENTRATION_CAUTION")
    preflight = {"checkpoint_sha_valid": binding.get("checkpoint_sha", False), "collection_binding_valid": all(binding.values()),
                 "collection_digest_valid": bool(collection), "snapshots_verified": len(snapshots), "snapshots_expected": 96,
                 "unique_snapshots": len(snapshot_by_digest), "schema": collection.get("snapshot_schema_version"),
                 "collection_schema": collection.get("collection_schema_version"), "checkpoint_sha256": checkpoint_sha_before,
                 "collection_digest": collection.get("collection_digest"), "authoritative_artifact": str(BT6.relative_to(PROJECT)),
                 "invalid_earlier_bt6_artifacts_used": 0}
    root.mkdir(parents=True)
    outputs = {
        "bt7rerun_evidence_preflight.json": preflight,
        "bt7rerun_frozen_inference_results.json": {"primary_behavior": "deterministic final-frozen-policy argmax; no RNG sampling",
                                                     "actor_eval": True, "no_grad": True, "device": "mps", "records": records,
                                                     "replay_reproducible": invariance["replay_reproducible"],
                                                     "checkpoint_sha256": checkpoint_sha_before, "collection_digest": collection.get("collection_digest")},
        "bt7rerun_no_assign_behavior.json": {"overall": overall, "by_time_band": {key: {field: value for field, value in item.items() if field in {"decision_count", "no_assign_count", "no_assign_rate", "feasible_support_count", "feasible_support_no_assign_count", "feasible_support_no_assign_rate", "support_size_distribution"}} for key, item in by_band.items()},
                                              "by_request_density": {key: {field: value for field, value in item.items() if field in {"decision_count", "no_assign_count", "no_assign_rate", "feasible_support_count", "feasible_support_no_assign_count", "feasible_support_no_assign_rate", "support_size_distribution"}} for key, item in by_density.items()},
                                              "by_post_zero_loss_support_size": {str(size): behavior_summary([row for row in records if row["support_size"] == size]) for size in sorted({row["support_size"] for row in records})}},
        "bt7rerun_agent_concentration.json": {"overall": overall["agent_concentration"], "by_time_band": {band: item["agent_concentration"] for band, item in by_band.items()},
                                                 "by_request_density": {density: item["agent_concentration"] for density, item in by_density.items()}},
        "bt7rerun_candidate_diversity.json": candidate_diversity,
        "bt7rerun_zero_loss_interaction.json": zero_loss,
        "bt7rerun_timeband_density_behavior.json": {"time_band": by_band, "request_density": by_density,
                                                       "density_mapping": {"0": "low", "1": "medium", "2": "medium", "3": "high"},
                                                       "interpretation": "descriptive frozen-policy behavior only; no causal time-band or density strategy claim"},
        "bt7rerun_decision_position_behavior.json": by_position,
        "bt7rerun_collapse_degeneracy_audit.json": collapse,
        "bt7rerun_invariance_tests.json": invariance,
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after,
                                            "checkpoint_sha256_before": checkpoint_sha_before, "checkpoint_sha256_after": sha256(CHECKPOINT_PATH)},
        "test_results.json": {"binding": binding, "integrity_counters": integrity, "hard_failures": hard, "warnings": [],
                              "seed_level_final_policy_comparison": "SEED_LEVEL_FINAL_POLICY_COMPARISON_NOT_AVAILABLE",
                              "training_time_actions_used_for_primary_metrics": False, "github_push_performed": False},
        "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"],
                               "bt6_source_commit": BT6_SOURCE, "checkpoint_sha256": checkpoint_sha_before,
                               "snapshot_collection_digest": collection.get("collection_digest"), "hard_failures": hard, "warnings": [],
                               "global_locks": LOCKS, "next_step": "separate evidence-grounded behavior review decision gate" if not hard else "STOP"},
    }
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Pure frozen-policy behavior review\n\n"
        f"gate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"The final frozen policy replayed {len(records)}/96 persisted snapshots on MPS with no simulator, candidate, Zero-Loss, reward, PPO, or optimizer execution. "
        f"NO_ASSIGN was selected {overall['no_assign_count']}/{overall['decision_count']} times; feasible-support NO_ASSIGN was "
        f"{overall['feasible_support_no_assign_count']}/{overall['feasible_support_count']}. "
        f"Collapse classification = {collapse['classification']}. These are behavior descriptions only, not performance or causal claims.\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "bt6_source_commit": BT6_SOURCE, "checkpoint_sha256": checkpoint_sha_before,
                                   "snapshot_collection_digest": collection.get("collection_digest"), "file_sha256": manifest,
                                   "elapsed_seconds": round(time.perf_counter() - started, 3), "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
