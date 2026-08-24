#!/usr/bin/env python3
"""BT8-R5: read-only initial-versus-final V2 review on F1's same snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R5"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R5_SAME_SUPPORT_FROZEN_V2_DISCRIMINATION_REVIEW_COMPLETE"
BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R5_FROZEN_EVIDENCE_OR_STRUCTURAL_INTEGRITY_FAILURE"
F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
R4A_SOURCE = "eac4a209e09e696380bde3bbc437a4fd13c45e99"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
SOURCE_FILES = {"05_training/run_h4m_ae_ls3_bt8_r5_same_support_review.py",
                "05_training/test_h4m_ae_ls3_bt8_r5_same_support_review.py"}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}
REPLICATES = {
    20260822: {"id": "R1", "actor_seed": 20260824, "critic_seed": 20260826},
    20260823: {"id": "R2", "actor_seed": 20260825, "critic_seed": 20260827},
}


class R5Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(value: bool, code: str, detail: str = "") -> None:
    if not value:
        raise R5Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True,
                          check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff", "--name-only", f"{F1_SOURCE}..HEAD"]).splitlines() if row]
    return {"source_commit": git(["rev-parse", "HEAD"]),
            "source_lineage_descends_from_f1": git(["merge-base", F1_SOURCE, "HEAD"]) == F1_SOURCE,
            "changed_files_since_f1": changed,
            "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
            "github_push_performed": False}


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    values = sorted(float(value) for value in values)
    if not values:
        return {key: None for key in ("min", "median", "mean", "max")}
    return {"min": values[0], "median": values[(len(values) - 1) // 2],
            "mean": statistics.mean(values), "max": values[-1]}


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, value in sorted(module.state_dict().items()):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_hashes(BT6: Any) -> dict[str, str]:
    return {**BT6.frozen_hashes(), "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
            "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}


def load_json(path: Path) -> Any:
    require(path.is_file(), "F1_EVIDENCE_FILE_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def load_actor(*, checkpoint: Path, config: Mapping[str, Any], H: Any, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping) and "actor" in payload, "ACTOR_CHECKPOINT_STATE_MISSING")
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]), candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]), heads=int(config["heads"]))
    actor.load_state_dict(payload["actor"], strict=True)
    return actor.to(device).eval()


def action_identity(selection: Any) -> str | tuple[str, str]:
    return "NO_ASSIGN" if selection.selected.is_no_assign else (str(selection.selected.agent_id), str(selection.selected.candidate_id))


def actor_record(*, actor: torch.nn.Module, payload: Mapping[str, Any], H: Any, TIE: Any,
                 R1: Any, MC: Any, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    result = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE)
    meta = payload["metadata"]
    selected = action_identity(result["selection"])
    legal = selected == "NO_ASSIGN" or selected in {(row["agent_id"], row["candidate_id"]) for row in meta["candidate_ids"]}
    distinct = R1.classify_alternatives(payload, range(result["support_size"]),
                                        candidate_names=MC.LOCAL_SEARCH_FEATURE_NAMES,
                                        agent_names=MC.AgentContext.FEATURE_NAMES)
    record = {"snapshot_digest": payload["snapshot_digest"], "window_id": meta["window_id"], "seed": int(meta["seed"]),
              "decision_id": meta["decision_id"], "candidate_ids": meta["candidate_ids"],
              "candidate_support_digest": meta["candidate_support_digest"], "support_size": result["support_size"],
              "meaningfully_distinct": distinct["classification"] == "MEANINGFULLY_DISTINCT", "selected": selected,
              "selected_source_index": result["selection"].selected.source_index, "exact_tie": result["selection"].exact_tie,
              "tie_set_size": len(result["selection"].tie_set), "unique_winner": not result["selection"].exact_tie,
              "pair_logits": result["pair_logits"][0].detach().cpu().tolist(),
              "no_assign_logit": float(result["no_assign_logit"][0, 0]), "probabilities": result["probabilities"][0].detach().cpu().tolist(),
              "entropy": result["entropy"], "top_score_margin": result["top_score_margin"],
              "finite": bool(result["finite"]), "legal_selection": legal, "candidate_distinctness": distinct,
              "tensor_mask_candidate_identity_digest": payload["snapshot_digest"]}
    return record, result


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    meaningful = [row for row in rows if row["support_size"] >= 2 and row["meaningfully_distinct"]]
    ties = [row for row in meaningful if row["exact_tie"]]
    unique = [row for row in meaningful if row["unique_winner"]]
    positive = [float(row["top_score_margin"]) for row in meaningful if row["top_score_margin"] is not None and float(row["top_score_margin"]) > 0.0]
    entropy = [float(row["entropy"]) for row in meaningful]
    return {"review_rows": len(rows), "meaningfully_distinct_multi_candidate_states": len(meaningful),
            "exact_ties": len(ties), "exact_tie_rate": rate(len(ties), len(meaningful)),
            "unique_winners": len(unique), "unique_winner_rate": rate(len(unique), len(meaningful)),
            "positive_margin_count": len(positive), "positive_margin_distribution": quantiles(positive),
            "entropy_distribution": quantiles(entropy), "mean_entropy": statistics.mean(entropy) if entropy else None,
            "no_assign_count": sum(row["selected"] == "NO_ASSIGN" for row in rows),
            "feasible_no_assign_count": sum(row["selected"] == "NO_ASSIGN" and row["support_size"] > 0 for row in rows)}


def discrimination_change(initial: Mapping[str, Any], final: Mapping[str, Any]) -> str:
    if not initial["meaningfully_distinct_multi_candidate_states"]:
        return "INSUFFICIENT_REVIEW_SAMPLES"
    if final["exact_ties"] < initial["exact_ties"]:
        return "IMPROVED"
    if final["exact_ties"] > initial["exact_ties"]:
        return "WORSENED"
    return "UNCHANGED"


def concentration(rows: Sequence[Mapping[str, Any]], R1: Any) -> dict[str, Any]:
    source = [{"candidate_ids": row["candidate_ids"], "selected": row["selected"]} for row in rows]
    audit = R1.concentration(source)
    feasible = [row for row in rows if row["support_size"] > 0]
    all_logits_constant = all(len(set(row["pair_logits"] + [row["no_assign_logit"]])) <= 1 for row in rows)
    entropy_collapse = all(float(row["entropy"]) == 0.0 for row in rows)
    universal_no_assign = bool(feasible) and all(row["selected"] == "NO_ASSIGN" for row in feasible)
    return {"opportunity_adjusted": audit, "feasible_no_assign_rate": rate(sum(row["selected"] == "NO_ASSIGN" for row in feasible), len(feasible)),
            "near_universal_no_assign": "reported_as_rate_only; no invented near-universal threshold",
            "single_agent_monopoly": audit["classification"] == "STRUCTURAL_DEGENERACY",
            "constant_logits_exact": all_logits_constant, "entropy_collapse_exact": entropy_collapse,
            "universal_feasible_no_assign_exact": universal_no_assign,
            "structural_collapse": bool(audit["classification"] == "STRUCTURAL_DEGENERACY" or all_logits_constant or entropy_collapse or universal_no_assign)}


def order_audit(*, policies: Mapping[str, Mapping[str, torch.nn.Module]], snapshots: Mapping[str, Sequence[Mapping[str, Any]]],
                H: Any, TIE: Any, R1: Any, device: torch.device) -> tuple[dict[str, Any], list[str]]:
    rows, hard = [], []
    for replicate_id, states in policies.items():
        for state_name, actor in states.items():
            before = module_digest(actor)
            for payload in snapshots[replicate_id]:
                base = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE)
                candidate_payload, agent_payload = R1.candidate_permutation(payload), R1.agent_permutation(payload)
                candidate = R1.frozen_forward(actor, candidate_payload, device=device, head=H, tie=TIE)
                agent = R1.frozen_forward(actor, agent_payload, device=device, head=H, tie=TIE)
                combined_payload = R1.candidate_permutation(agent_payload)
                combined = R1.frozen_forward(actor, combined_payload, device=device, head=H, tie=TIE)
                base_logits, base_probs = R1.scores_by_identity(base, payload)
                def delta(other: Mapping[str, Any], other_payload: Mapping[str, Any]) -> tuple[float, float]:
                    logits, probabilities = R1.scores_by_identity(other, other_payload)
                    return (max((abs(base_logits[key] - logits[key]) for key in base_logits), default=0.0),
                            max((abs(base_probs[key] - probabilities[key]) for key in base_probs), default=0.0))
                candidate_delta, agent_delta, combined_delta = delta(candidate, candidate_payload), delta(agent, agent_payload), delta(combined, combined_payload)
                rows.append({"replicate_id": replicate_id, "policy_state": state_name, "snapshot_digest": payload["snapshot_digest"],
                             "multi_candidate": base["support_size"] >= 2,
                             "candidate_identity_equal": action_identity(base["selection"]) == action_identity(candidate["selection"]),
                             "agent_identity_equal": action_identity(base["selection"]) == action_identity(agent["selection"]),
                             "combined_identity_equal": action_identity(base["selection"]) == action_identity(combined["selection"]),
                             "candidate_logit_delta": candidate_delta[0], "candidate_probability_delta": candidate_delta[1],
                             "agent_logit_delta": agent_delta[0], "agent_probability_delta": agent_delta[1],
                             "combined_logit_delta": combined_delta[0], "combined_probability_delta": combined_delta[1]})
            if module_digest(actor) != before:
                hard.append("ACTOR_PARAMETER_MUTATION_DURING_ORDER_AUDIT")
    def metric(prefix: str, scope: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"tested": len(scope), "identity_changes": sum(not row[f"{prefix}_identity_equal"] for row in scope),
                "max_identity_aligned_logit_delta": max((row[f"{prefix}_logit_delta"] for row in scope), default=0.0),
                "max_identity_aligned_probability_delta": max((row[f"{prefix}_probability_delta"] for row in scope), default=0.0)}
    multi = [row for row in rows if row["multi_candidate"]]
    candidate, agent, combined = metric("candidate", multi), metric("agent", rows), metric("combined", multi)
    if candidate["identity_changes"] or agent["identity_changes"] or combined["identity_changes"]:
        hard.append("ORDER_INVARIANCE_FAILURE")
    return {"selector": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "tolerance": 0.0},
            "candidate_order": candidate, "agent_order": agent, "combined_order": combined,
            "candidate_order_all_passed": candidate["identity_changes"] == 0,
            "agent_order_all_passed": agent["identity_changes"] == 0,
            "combined_order_all_passed": combined["identity_changes"] == 0,
            "numerical_rule": "no invented tolerance; observed identity-aligned deltas are recorded, while identity must be exact"}, hard


def write_block(*, root: Path, source: Mapping[str, Any], preflight: Mapping[str, Any], reason: str) -> None:
    counters = {"training": 0, "optimizer_step": 0, "causal_rollout": 0, "candidate_regeneration": 0,
                "local_search_rerun": 0, "zero_loss_reevaluation": 0, "parameter_mutation": 0, "checkpoint_write": 0}
    outputs = {"bt8r5_evidence_preflight.json": preflight, "bt8r5_replicate1_same_support_review.json": {"not_run": True},
               "bt8r5_replicate2_same_support_review.json": {"not_run": True}, "bt8r5_cross_replicate_summary.json": {"not_run": True},
               "bt8r5_order_invariance.json": {"not_run": True}, "bt8r5_concentration_collapse_audit.json": {"not_run": True},
               "test_results.json": {"hard_failures": [reason], "execution_counters": counters}, "frozen_hash_before_after.json": {"not_run": True},
               "gate_decision.json": {"stage": STAGE, "gate": BLOCK, "classification": "BLOCKED", "source_commit": source["source_commit"],
                                      "hard_failures": [reason], "warnings": [], "global_locks": LOCKS, "next_step": "STOP"}}
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-R5 blocked\n\n{reason}\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": BLOCK, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(BLOCK + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT)); source = provenance()
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1

    stamp = BT6.now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r5_same_support_frozen_v2_review_{stamp}"
    require(not root.exists(), "APPEND_ONLY_ARTIFACT_COLLISION"); root.mkdir(parents=True)
    gate, initial_manifest, final_manifest = load_json(F1 / "gate_decision.json"), load_json(F1 / "bt8f1_initial_checkpoint_manifest.json"), load_json(F1 / "bt8f1_final_checkpoint_manifest.json")
    collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
    before = frozen_hashes(BT6); hard: list[str] = []
    preflight = {"f1_gate": gate.get("gate") == F1_GATE, "f1_source": gate.get("source_commit") == F1_SOURCE,
                 "r5_source_lineage": source["source_lineage_descends_from_f1"], "r5_source_only": source["source_only_local_commit"],
                 "r4a_source": R4A_SOURCE, "collection_digest": collection.get("collection_digest"),
                 "collection_schema": collection.get("collection_schema_version") == FPS.COLLECTION_SCHEMA_VERSION,
                 "snapshot_schema": collection.get("snapshot_schema_version") == FPS.SNAPSHOT_SCHEMA_VERSION,
                 "snapshot_count": collection.get("snapshot_count") == len(collection.get("entries", [])) == 6,
                 "unique_snapshot_digests": len({row.get("snapshot_digest") for row in collection.get("entries", [])}) == 6,
                 "t1_contract": TIE.TIE_BREAK_CONTRACT_ID == "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1",
                 "t1_selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "t1_tolerance": 0.0}
    try:
        check = dict(collection); supplied = check.pop("collection_digest")
        require(supplied == FPS.canonical_sha256(check), "REVIEW_COLLECTION_DIGEST_MISMATCH")
        snapshots: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in REPLICATES.values()}
        config = None
        for entry in collection["entries"]:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / entry["relative_path"])
            require(payload["snapshot_digest"] == entry["snapshot_digest"], "REVIEW_SNAPSHOT_ENTRY_DIGEST_MISMATCH")
            meta = payload["metadata"]; seed = int(meta["seed"]); require(seed in REPLICATES, "UNAUTHORIZED_REVIEW_SEED")
            expected = REPLICATES[seed]["id"]; require(meta["seed"] == seed, "REVIEW_SNAPSHOT_SEED_MISMATCH")
            snapshots[expected].append(payload); config = meta["actor_config"] if config is None else config
            require(meta["actor_config"] == config and meta["actor_config_sha256"] == FPS.actor_config_sha256(config), "ACTOR_CONFIG_BINDING_MISMATCH")
            require(meta["candidate_ids"] == meta["candidate_order"], "CANDIDATE_ORDER_METADATA_MISMATCH")
            require(meta["captured_device"] == "mps:0", "REVIEW_SNAPSHOT_DEVICE_MISMATCH")
        require(all(len(rows) == 3 for rows in snapshots.values()), "REPLICATE_REVIEW_SNAPSHOT_COUNT_MISMATCH")
        require(config is not None and config.get("actor_head_id") == H.CANDIDATE_SENSITIVE_HEAD_ID and config.get("actor_head_version") == H.CANDIDATE_SENSITIVE_HEAD_VERSION, "V2_ACTOR_CONFIG_MISMATCH")
        binding = collection.get("checkpoint_binding", {})
        for seed, contract in REPLICATES.items():
            replicate_id = "F1_" + contract["id"]
            initial, final = initial_manifest.get(replicate_id), final_manifest.get(replicate_id)
            require(isinstance(initial, Mapping) and isinstance(final, Mapping), "REPLICATE_CHECKPOINT_MANIFEST_MISSING", replicate_id)
            initial_path, final_path = F1 / "initial_checkpoints" / initial["path"], F1 / "final_checkpoints" / final["path"]
            require(sha256(initial_path) == initial["sha256"] == binding["initial_actor_checkpoint_sha256_by_replicate"][replicate_id], "INITIAL_CHECKPOINT_SHA_MISMATCH", replicate_id)
            require(sha256(final_path) == final["sha256"] == binding["final_actor_checkpoint_sha256_by_replicate"][replicate_id], "FINAL_CHECKPOINT_SHA_MISMATCH", replicate_id)
            require(initial.get("actor_init_seed") == contract["actor_seed"] and initial.get("critic_init_seed") == contract["critic_seed"], "INITIALIZATION_SEED_BINDING_MISMATCH", replicate_id)
    except Exception as exc:  # noqa: BLE001
        hard.append(f"EVIDENCE_PREFLIGHT_FAILURE:{type(exc).__name__}:{exc}")
        snapshots, config = {}, None
    if not all(value for key, value in preflight.items() if key not in {"collection_digest", "t1_selector_sha256", "r4a_source", "t1_tolerance"}): hard.append("AUTHORITATIVE_BINDING_FAILURE")
    if hard:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": hard}, reason=hard[0]); print(f"[BLOCKED] {root.relative_to(PROJECT)}"); return
    if not torch.backends.mps.is_available():
        write_block(root=root, source=source, preflight=preflight | {"mps_available": False}, reason="MPS_REQUIRED_FOR_BOUND_SNAPSHOT_REPLAY"); print(f"[BLOCKED] {root.relative_to(PROJECT)}"); return
    device = torch.device("mps:0")
    policies: dict[str, dict[str, torch.nn.Module]] = {}
    checkpoint_before: dict[str, str] = {}
    for seed, contract in REPLICATES.items():
        key, replicate_id = contract["id"], "F1_" + contract["id"]
        initial_path, final_path = F1 / "initial_checkpoints" / initial_manifest[replicate_id]["path"], F1 / "final_checkpoints" / final_manifest[replicate_id]["path"]
        checkpoint_before[replicate_id] = sha256(initial_path) + ":" + sha256(final_path)
        policies[key] = {"initial": load_actor(checkpoint=initial_path, config=config, H=H, device=device),
                         "final": load_actor(checkpoint=final_path, config=config, H=H, device=device)}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {key: {"initial": [], "final": []} for key in policies}
    integrity = {"training": 0, "optimizer_step": 0, "causal_rollout": 0, "candidate_regeneration": 0,
                 "local_search_rerun": 0, "zero_loss_reevaluation": 0, "parameter_mutation": 0, "checkpoint_write": 0,
                 "nan_or_inf": 0, "illegal_or_masked_selection": 0, "zero_loss_fail_in_legal_support": 0}
    for replicate_id, states in policies.items():
        for state_name, actor in states.items():
            digest_before = module_digest(actor)
            for payload in snapshots[replicate_id]:
                row, _ = actor_record(actor=actor, payload=payload, H=H, TIE=TIE, R1=R1, MC=MC, device=device)
                records[replicate_id][state_name].append(row)
                integrity["nan_or_inf"] += int(not row["finite"]); integrity["illegal_or_masked_selection"] += int(not row["legal_selection"])
            integrity["parameter_mutation"] += int(module_digest(actor) != digest_before)
    torch.mps.synchronize()
    order, order_hard = order_audit(policies=policies, snapshots=snapshots, H=H, TIE=TIE, R1=R1, device=device); hard.extend(order_hard)
    reviews, concentration_audit = {}, {}
    for key in ("R1", "R2"):
        initial, final = summary(records[key]["initial"]), summary(records[key]["final"])
        change = discrimination_change(initial, final)
        pair_rows = []
        initial_by = {row["snapshot_digest"]: row for row in records[key]["initial"]}
        for final_row in records[key]["final"]:
            start = initial_by[final_row["snapshot_digest"]]
            require(start["candidate_ids"] == final_row["candidate_ids"] and start["candidate_support_digest"] == final_row["candidate_support_digest"], "SAME_SUPPORT_COMPARISON_MISMATCH")
            pair_rows.append({"snapshot_digest": final_row["snapshot_digest"], "initial_selected": start["selected"], "final_selected": final_row["selected"],
                              "initial_exact_tie": start["exact_tie"], "final_exact_tie": final_row["exact_tie"],
                              "initial_margin": start["top_score_margin"], "final_margin": final_row["top_score_margin"],
                              "initial_entropy": start["entropy"], "final_entropy": final_row["entropy"]})
        reviews[key] = {"replicate_id": key, "initial": initial, "final": final, "change": change, "same_support_pairs": pair_rows,
                        "margin_mean_delta": (final["positive_margin_distribution"]["mean"] - initial["positive_margin_distribution"]["mean"] if initial["positive_margin_distribution"]["mean"] is not None and final["positive_margin_distribution"]["mean"] is not None else None),
                        "entropy_mean_delta": (final["mean_entropy"] - initial["mean_entropy"] if initial["mean_entropy"] is not None and final["mean_entropy"] is not None else None)}
        concentration_audit[key] = {"initial": concentration(records[key]["initial"], R1), "final": concentration(records[key]["final"], R1)}
    directions = {key: reviews[key]["change"] for key in reviews}
    collapsed = any(value[state]["structural_collapse"] for value in concentration_audit.values() for state in ("initial", "final"))
    if collapsed or "WORSENED" in directions.values(): classification = "D_V2_CANDIDATE_DISCRIMINATION_WORSENED_OR_COLLAPSED"
    elif all(value == "IMPROVED" for value in directions.values()): classification = "A_V2_CANDIDATE_DISCRIMINATION_IMPROVED_ACROSS_BOTH_REPLICATES"
    elif "IMPROVED" in directions.values(): classification = "B_V2_CANDIDATE_DISCRIMINATION_IMPROVED_BUT_SEED_SENSITIVE"
    else: classification = "C_V2_CANDIDATE_DISCRIMINATION_NOT_MATERIALLY_IMPROVED"
    after = frozen_hashes(BT6); integrity["checkpoint_write"] = 0
    checkpoint_after = {"F1_R1": sha256(F1 / "initial_checkpoints" / initial_manifest["F1_R1"]["path"]) + ":" + sha256(F1 / "final_checkpoints" / final_manifest["F1_R1"]["path"]),
                        "F1_R2": sha256(F1 / "initial_checkpoints" / initial_manifest["F1_R2"]["path"]) + ":" + sha256(F1 / "final_checkpoints" / final_manifest["F1_R2"]["path"])}
    if before != after: hard.append("FROZEN_HASH_MUTATION")
    if checkpoint_before != checkpoint_after: hard.append("CHECKPOINT_MUTATION")
    if any(integrity.values()): hard.append("FORBIDDEN_EXECUTION_OR_INTEGRITY_COUNTER_NONZERO")
    gate_out = PASS_GATE if not hard else BLOCK
    if hard: classification = "BLOCKED"
    cross = {"replicate_directions": directions, "consistency": ("BOTH_IMPROVED" if all(value == "IMPROVED" for value in directions.values()) else "ONE_POLICY_IMPROVED" if "IMPROVED" in directions.values() else "BOTH_UNCHANGED" if all(value == "UNCHANGED" for value in directions.values()) else "OPPOSITE_OR_INSUFFICIENT"),
             "combination_rule": "replicate weights and outcomes are never averaged", "review_sample_scope": "three same-support snapshots per independent replicate", "classification": classification}
    outputs = {"bt8r5_evidence_preflight.json": preflight | {"checkpoint_binding": checkpoint_before, "actor_config": config, "actor_config_sha256": FPS.actor_config_sha256(config)},
               "bt8r5_replicate1_same_support_review.json": reviews["R1"], "bt8r5_replicate2_same_support_review.json": reviews["R2"],
               "bt8r5_cross_replicate_summary.json": cross, "bt8r5_order_invariance.json": order,
               "bt8r5_concentration_collapse_audit.json": concentration_audit,
               "test_results.json": {"integrity_counters": integrity, "hard_failures": hard, "warnings": [], "execution_counters": integrity, "github_push_performed": False},
               "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after, "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after},
               "gate_decision.json": {"stage": STAGE, "gate": gate_out, "classification": classification, "source_commit": source["source_commit"], "hard_failures": hard, "warnings": [], "global_locks": LOCKS, "next_step": "separate frozen V2 review outcome interpretation / next-gate selection" if not hard else "STOP"}}
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-R5 final report\n\n- gate: `{gate_out}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n\nThis is a same-input structural frozen-policy review only; it makes no KPI, performance, or causal-performance claim.\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate_out, "classification": classification, "source_commit": source["source_commit"], "file_sha256": manifest, "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate_out + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate_out}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
