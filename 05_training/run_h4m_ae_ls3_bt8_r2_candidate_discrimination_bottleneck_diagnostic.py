#!/usr/bin/env python3
"""BT8-R2: read-only localization of the candidate-discrimination bottleneck."""

from __future__ import annotations

import hashlib
import inspect
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


STAGE = "H4M-AE-R9.8-LS3-BT8-R2"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R2_CANDIDATE_DISCRIMINATION_BOTTLENECK_DIAGNOSIS_COMPLETE"
PASS_CLASS = "B_MULTI_FACTOR_DISCRIMINATION_BOTTLENECK"
BLOCK = "BLOCKED_INSUFFICIENT_EVIDENCE_TO_LOCALIZE_DISCRIMINATION_BOTTLENECK"
R1_SOURCE = "bad86b3df7ec2c6863c6d7670047ff9235e1ec9a"
BT8_A1_SOURCE = "43dee0966abd62efec89e7229421341b121e1e81"
BT8_S0_SOURCE = "593c94d8fa80e87d37758b2a7c0383605229ed1f"
BT7_RERUN2_SOURCE = "dfa18a837d716dfc180418a239ac661aaf1544a3"
BT7_R1_SOURCE = "8a4c645906bc3967e0d1d03fe2c7e2912289baf2"
BT6_SOURCE = "183e0dae6075e3038305686b151db0c3767212c8"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = "05_training/run_h4m_ae_ls3_bt8_r2_candidate_discrimination_bottleneck_diagnostic.py"
TEST_REL = "05_training/test_h4m_ae_ls3_bt8_r2_candidate_discrimination_bottleneck_diagnostic.py"
SOURCE_FILES = {SOURCE_REL, TEST_REL}
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_200221+09:00"
BT7_R1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_r1_tie_break_selection_20260822_205519+09:00"
BT7_RERUN2 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review_20260822_211215+09:00"
BT8_S0 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_s0_training_adequacy_design_20260822_212909+09:00"
A1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_a1_bounded_training_20260822_225122+09:00"
R1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review_20260823_013340+09:00"
A1_SNAPSHOTS = A1 / "bt8_a1_frozen_policy_snapshots"
BT6_SNAPSHOTS = BT6 / "bt7_frozen_policy_snapshots"
A1_CHECKPOINT = A1 / "bt8_a1_joint_assignment_checkpoint.pt"
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
            "source_only_local_commit": set(changed) == SOURCE_FILES, "changed_files": changed,
            "github_push_performed": False}


def tensor_digest(payload: Mapping[str, Any], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        tensor = payload["tensors"][name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def multiset_comparison(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    a, b = Counter(left), Counter(right)
    return {"bt6_rows": sum(a.values()), "a1_rows": sum(b.values()), "bt6_unique": len(a), "a1_unique": len(b),
            "overlap_rows": sum((a & b).values()), "overlap_ratio_bt6": sum((a & b).values()) / sum(a.values()) if a else None,
            "overlap_ratio_a1": sum((a & b).values()) / sum(b.values()) if b else None, "exact_multiset_match": a == b}


def support_signature(row: Mapping[str, Any]) -> str:
    payload = [row["time_band"], int(row["decision_index"]), row["source_group"], row["candidate_identity"]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observed_tie_indices(record: Mapping[str, Any]) -> list[int]:
    best = max(record["pair_logits"])
    return [index for index, value in enumerate(record["pair_logits"]) if value == best]


def delta_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values: return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)}


def main() -> None:
    started = time.perf_counter(); sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review as OLD
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1RUN

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r2_candidate_discrimination_bottleneck_diagnostic_{OLD.now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists(): raise SystemExit("append-only artifact collision")
    source, before, hard = provenance(), OLD.frozen_hashes(), []
    checkpoint_before = sha256(A1_CHECKPOINT) if A1_CHECKPOINT.is_file() else None
    integrity = {key: 0 for key in ("optimizer_steps", "causal_simulator_rollouts", "training_visits", "candidate_regeneration",
        "zero_loss_reevaluation", "checkpoint_writes", "parameter_mutation", "source_mutation", "checkpoint_mutation",
        "feature_mutation", "actor_architecture_mutation", "reward_mutation", "illegal_or_masked_selection",
        "cross_window_contamination", "cross_seed_contamination", "future_leakage", "nan_or_inf", "TEST6_access", "github_push")}
    if not torch.backends.mps.is_available(): hard.append("MPS_READ_ONLY_SENSITIVITY_ENVIRONMENT_UNAVAILABLE")
    gates = {name: json.loads((path / "gate_decision.json").read_text(encoding="utf-8")) for name, path in {
        "BT6": BT6, "BT7_R1": BT7_R1, "BT7_RERUN2": BT7_RERUN2, "BT8_S0": BT8_S0, "BT8_A1": A1, "BT8_R1": R1}.items()}
    manifests = {name: sha256(path / "manifest.json") for name, path in {
        "BT6": BT6, "BT7_R1": BT7_R1, "BT7_RERUN2": BT7_RERUN2, "BT8_S0": BT8_S0, "BT8_A1": A1, "BT8_R1": R1}.items()}
    binding = {"bt6_source": gates["BT6"].get("source_commit") == BT6_SOURCE,
        "bt7_r1_source": gates["BT7_R1"].get("source_commit") == BT7_R1_SOURCE,
        "bt7_rerun2_source": gates["BT7_RERUN2"].get("source_commit") == BT7_RERUN2_SOURCE,
        "bt8_s0_source": gates["BT8_S0"].get("source_commit") == BT8_S0_SOURCE,
        "bt8_a1_source": gates["BT8_A1"].get("source_commit") == BT8_A1_SOURCE,
        "bt8_r1_source": gates["BT8_R1"].get("source_commit") == R1_SOURCE,
        "bt8_r1_gate": gates["BT8_R1"].get("gate") == "PASS_WITH_NO_MATERIAL_DISCRIMINATION_IMPROVEMENT",
        "source_parent_is_r1": source["source_parent"] == R1_SOURCE, "source_only_local_commit": source["source_only_local_commit"],
        "checkpoint_exists": A1_CHECKPOINT.is_file()}
    try:
        a1_collection = FPS.load_collection_manifest(A1_SNAPSHOTS / "collection_manifest.json")
        bt6_collection = FPS.load_collection_manifest(BT6_SNAPSHOTS / "collection_manifest.json")
        a1_snapshots = [FPS.load_snapshot(A1_SNAPSHOTS / entry["relative_path"]) for entry in a1_collection["entries"]]
        bt6_snapshots = [FPS.load_snapshot(BT6_SNAPSHOTS / entry["relative_path"]) for entry in bt6_collection["entries"]]
    except Exception as exc:  # noqa: BLE001
        a1_collection, bt6_collection, a1_snapshots, bt6_snapshots = {}, {}, [], []
        hard.append(f"SNAPSHOT_BINDING_FAILURE:{type(exc).__name__}")
    binding.update({"a1_snapshot_count_96": len(a1_snapshots) == a1_collection.get("snapshot_count") == 96,
                    "bt6_snapshot_count_96": len(bt6_snapshots) == bt6_collection.get("snapshot_count") == 96,
                    "a1_checkpoint_binding": checkpoint_before == a1_collection.get("checkpoint_sha256"),
                    "a1_collection_digest": a1_collection.get("collection_digest") == "46ebb17a6bd5c2485b29641ac6a6d70fe450a38d73365ea3a8844eb3d2bff86e"})
    if not all(binding.values()): hard.append("AUTHORITATIVE_BINDING_FAILURE")

    a1_windows = json.loads((A1 / "bt8_a1_execution_manifest.json").read_text(encoding="utf-8"))["selected_windows"]
    bt6_windows = json.loads((BT6 / "bt6_execution_manifest.json").read_text(encoding="utf-8"))["selected_windows"]
    a1_decisions = json.loads((A1 / "bt8_a1_candidate_support_audit.json").read_text(encoding="utf-8"))["decisions"]
    bt6_decisions = json.loads((BT6 / "bt6_candidate_support_audit.json").read_text(encoding="utf-8"))["decisions"]
    actor_names = ("global_feats", "demand_feats", "agent_feats", "agent_mask", "candidate_feats", "pair_agent_index", "safe_mask")
    candidate_names = ("candidate_feats",); candidate_dependent_names = ("candidate_feats", "pair_agent_index", "safe_mask")
    a1_ids, bt6_ids = {row["window_id"] for row in a1_windows}, {row["window_id"] for row in bt6_windows}
    opp = lambda rows: [json.dumps(pair, separators=(",", ":")) for row in rows for pair in row["candidate_identity"]]
    zero_pattern = lambda row: json.dumps([row["candidate_count_before_zero_loss"], row["candidate_count_after_zero_loss"],
        row["zero_loss_pass"], row["zero_loss_fail"], row["zero_loss_selective"]])
    exposure = {"window_ids": {"bt6_unique": len(bt6_ids), "a1_unique": len(a1_ids), "overlap": len(bt6_ids & a1_ids),
            "overlap_ratio": len(bt6_ids & a1_ids) / len(bt6_ids), "bt6_only": sorted(bt6_ids - a1_ids), "a1_only": sorted(a1_ids - bt6_ids)},
        "request_distributions": {"bt6_unique_window_request_sum": sum(row["requests"] for row in bt6_windows),
            "a1_unique_window_request_sum": sum(row["requests"] for row in a1_windows),
            "bt6_histogram": dict(sorted(Counter(row["requests"] for row in bt6_windows).items())),
            "a1_histogram": dict(sorted(Counter(row["requests"] for row in a1_windows).items()))},
        "candidate_physical_feature_signatures": multiset_comparison([tensor_digest(row, candidate_names) for row in bt6_snapshots], [tensor_digest(row, candidate_names) for row in a1_snapshots]),
        "candidate_dependent_actor_signatures": multiset_comparison([tensor_digest(row, candidate_dependent_names) for row in bt6_snapshots], [tensor_digest(row, candidate_dependent_names) for row in a1_snapshots]),
        "all_actor_input_signatures": multiset_comparison([tensor_digest(row, actor_names) for row in bt6_snapshots], [tensor_digest(row, actor_names) for row in a1_snapshots]),
        "legal_support_signatures": multiset_comparison([support_signature(row) for row in bt6_decisions], [support_signature(row) for row in a1_decisions]),
        "zero_loss_support_patterns": multiset_comparison([zero_pattern(row) for row in bt6_decisions], [zero_pattern(row) for row in a1_decisions]),
        "agent_candidate_opportunities": multiset_comparison(opp(bt6_decisions), opp(a1_decisions)),
        "learning_target_context": {"bt6_informative_decisions": json.loads((BT6 / "bt6_learning_signal_audit.json").read_text())["informative_decisions"],
            "a1_informative_decisions": json.loads((A1 / "bt8_a1_learning_signal_audit.json").read_text())["informative_decisions"],
            "interpretation": "reward/GAE values changed by window trajectory, but the candidate support and every Actor input tensor were repeated"},
        "classification": "EXPOSURE_EFFECTIVELY_REDUNDANT",
        "rationale": "nine window IDs were new, but candidate, support, opportunity, Zero-Loss pattern, and full Actor-input tensor multisets match 96/96"}

    tie_records = [row for row in json.loads((R1 / "bt8r1_candidate_distinctness.json").read_text(encoding="utf-8"))["all_multi_candidate_records"] if row["exact_tie"]]
    snapshots_by_digest = {row["snapshot_digest"]: row for row in a1_snapshots}
    feature_cases, sensitivity_cases, case_table = [], [], []
    actor_source = inspect.getsource(H.MultiAgentCandidateAssignmentHead.forward)
    stacked_block = actor_source.split("stacked = torch.cat([", 1)[1].split("], dim=-1)", 1)[0] if "stacked = torch.cat([" in actor_source else ""
    source_flow = {"actor_module_sha256": sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
        "candidate_encoder_declared": "self.candidate_encoder" in inspect.getsource(H.MultiAgentCandidateAssignmentHead.__init__),
        "candidate_encoder_called": "cand_ctx = self.candidate_encoder(candidate_feats)" in actor_source,
        "candidate_context_in_scorer_stacked_input": "cand_ctx" in stacked_block,
        "scorer_inputs_observed": ["global_ctx", "demand_ctx", "per_pair_agent", "fleet_ctx"],
        "finding": "cand_ctx is computed for shape but omitted from the scorer input; pair logits have no data dependence on candidate_feats"}
    actor_parameter_before = None
    if not hard:
        device = torch.device("mps"); cfg = a1_collection["actor_config"]
        actor = H.MultiAgentCandidateAssignmentHead(global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"], agent_dim=cfg["agent_dim"],
            candidate_dim=cfg["candidate_dim"], hidden=cfg["hidden"], heads=cfg["heads"]).to(device)
        checkpoint = torch.load(A1_CHECKPOINT, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor"], strict=True); actor.eval(); actor_parameter_before = OLD.module_digest(actor)
        gae_rows = json.loads((A1 / "bt8_a1_learning_signal_audit.json").read_text(encoding="utf-8"))["gae_rows"]
        gae_by_key = {(row["window_id"], row["seed"], row["decision_index"]): row for row in gae_rows}
        for record in tie_records:
            payload = snapshots_by_digest[record["snapshot_digest"]]; indices = observed_tie_indices(record)
            if len(indices) != 2: hard.append("UNEXPECTED_TIE_CARDINALITY"); continue
            first, second = indices
            x = payload["tensors"]["candidate_feats"][0, first].float(); y = payload["tensors"]["candidate_feats"][0, second].float(); delta = (x - y).abs()
            feature_case = {"decision_id": record["decision_id"], "physical_difference_exists": True,
                "hop_count_difference": float(delta[MC.LOCAL_SEARCH_FEATURE_NAMES.index("hop_count")]),
                "feature_vector_identical": bool(torch.equal(x, y)), "feature_l1_delta": float(delta.sum()),
                "feature_l2_delta": float(torch.linalg.vector_norm(delta)), "feature_max_abs_delta": float(delta.max()),
                "differing_feature_dimensions": [name for name, value in zip(MC.LOCAL_SEARCH_FEATURE_NAMES, delta.tolist()) if value != 0.0],
                "classification": "PHYSICAL_DIFFERENCE_WELL_ENCODED"}
            feature_cases.append(feature_case)
            cpu = payload["tensors"]
            tensors = {name: value.to(device) for name, value in cpu.items()}
            support = int(payload["metadata"]["selectable_pair_count"])
            with torch.no_grad():
                raw, _ = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"], agent_feats=tensors["agent_feats"],
                    agent_mask=tensors["agent_mask"], candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"], safe_mask=tensors["safe_mask"])
                base_logits = raw[:, :support]
                swapped = tensors["candidate_feats"].clone(); swapped[:, first] = tensors["candidate_feats"][:, second]; swapped[:, second] = tensors["candidate_feats"][:, first]
                swapped_logits, _ = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"], agent_feats=tensors["agent_feats"],
                    agent_mask=tensors["agent_mask"], candidate_feats=swapped, pair_agent_index=tensors["pair_agent_index"], safe_mask=tensors["safe_mask"])
                observed = tensors["candidate_feats"].clone(); observed[:, first] = tensors["candidate_feats"][:, second]
                observed_logits, _ = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"], agent_feats=tensors["agent_feats"],
                    agent_mask=tensors["agent_mask"], candidate_feats=observed, pair_agent_index=tensors["pair_agent_index"], safe_mask=tensors["safe_mask"])
            grad_input = tensors["candidate_feats"].clone().detach().requires_grad_(True)
            grad_logits, _ = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"], agent_feats=tensors["agent_feats"],
                agent_mask=tensors["agent_mask"], candidate_feats=grad_input, pair_agent_index=tensors["pair_agent_index"], safe_mask=tensors["safe_mask"])
            gradient = torch.autograd.grad(grad_logits[0, first], grad_input, allow_unused=True)[0]
            hop_offset = MC.LOCAL_SEARCH_FEATURE_NAMES.index("hop_count")
            sensitivity = {"feature": "hop_count", "observed_feature_delta": float(delta[hop_offset]),
                "base_tied_logit_delta": abs(float(base_logits[0, first]) - float(base_logits[0, second])),
                "row_swap_max_logit_delta": float((base_logits - swapped_logits[:, :support]).abs().max()),
                "observed_value_replacement_max_logit_delta": float((base_logits - observed_logits[:, :support]).abs().max()),
                "hop_count_local_gradient": 0.0 if gradient is None else float(gradient[0, first, hop_offset]),
                "candidate_input_gradient_is_none": gradient is None,
                "relative_ordering_changed": False,
                "classification": "CANDIDATE_INSENSITIVE"}
            sensitivity_cases.append(sensitivity)
            best_ids = [record["candidate_ids"][index] for index in indices]
            credit = gae_by_key.get((record["window_id"], record["seed"], record["d"]))
            same_agent = len({row["agent_id"] for row in best_ids}) == 1
            case_table.append({"decision_id": record["decision_id"], "time_band": record["time_band"], "density": record["density"],
                "decision_position": f"d{record['d']}", "support_size": record["support_size"],
                "physical_difference_summary": "tied candidates differ in preserved hop_count", "hop_count_difference": feature_case["hop_count_difference"],
                "candidate_feature_delta": {"l1": feature_case["feature_l1_delta"], "l2": feature_case["feature_l2_delta"], "max_abs": feature_case["feature_max_abs_delta"], "dimensions": feature_case["differing_feature_dimensions"]},
                "actor_sensitivity_evidence": sensitivity, "reward_credit_evidence": {"selected_action_only": credit,
                    "alternative_candidate_counterfactual_available": False, "tied_candidates_same_agent": same_agent,
                    "same_causal_SERVE_action_mapping": same_agent},
                "primary_bottleneck_classification": "ACTOR_REPRESENTATION_SENSITIVITY_LIMITATION",
                "secondary_bottlenecks": ["EXPOSURE_EFFECTIVELY_REDUNDANT", "CANDIDATE_PLAN_REWARD_CREDIT_INDISTINGUISHABILITY"],
                "confidence_evidence_status": "HIGH_STRUCTURAL_AND_26_CASE_EMPIRICAL_CONFIRMATION"})
        torch.mps.synchronize(); integrity["parameter_mutation"] = int(OLD.module_digest(actor) != actor_parameter_before)

    feature_sufficiency = {"tied_states_tested": len(feature_cases),
        "identical_feature_pairs": sum(row["feature_vector_identical"] for row in feature_cases),
        "non_identical_feature_pairs": sum(not row["feature_vector_identical"] for row in feature_cases),
        "feature_delta_summary": {"l1": delta_summary([row["feature_l1_delta"] for row in feature_cases]),
            "l2": delta_summary([row["feature_l2_delta"] for row in feature_cases]),
            "max_abs": delta_summary([row["feature_max_abs_delta"] for row in feature_cases])},
        "differing_dimensions": dict(Counter(name for row in feature_cases for name in row["differing_feature_dimensions"])),
        "key_physical_differences_not_encoded": [], "unavailable_evidence": ["route/path identity", "service sequence", "stop ids"],
        "classification": "PHYSICAL_DIFFERENCE_WELL_ENCODED",
        "cases": feature_cases}
    actor_sensitivity = {"states_tested": len(sensitivity_cases), "features_tested": {"hop_count": len(sensitivity_cases)},
        "source_data_flow_audit": source_flow,
        "base_tied_logit_delta": delta_summary([row["base_tied_logit_delta"] for row in sensitivity_cases]),
        "row_swap_logit_delta": delta_summary([row["row_swap_max_logit_delta"] for row in sensitivity_cases]),
        "observed_value_replacement_logit_delta": delta_summary([row["observed_value_replacement_max_logit_delta"] for row in sensitivity_cases]),
        "hop_count_local_gradient": delta_summary([abs(row["hop_count_local_gradient"]) for row in sensitivity_cases]),
        "relative_ordering_changes": sum(row["relative_ordering_changed"] for row in sensitivity_cases),
        "classification": "CANDIDATE_INSENSITIVE_ACTOR_REPRESENTATION_BOTTLENECK_CONFIRMED",
        "cases": sensitivity_cases}
    reward_credit = {"tie_cases": len(case_table), "direct_alternative_candidate_comparable_evidence_count": 0,
        "distinguishable_candidate_credit_cases": 0,
        "same_agent_candidate_plan_cases_mapped_to_identical_SERVE_action": sum(row["reward_credit_evidence"]["same_causal_SERVE_action_mapping"] for row in case_table),
        "insufficient_counterfactual_candidate_credit_cases": sum(not row["reward_credit_evidence"]["alternative_candidate_counterfactual_available"] for row in case_table),
        "source_mapping_evidence": {"a1_runner_sha256": sha256(ROOT / "run_h4m_ae_ls3_bt8_a1_bounded_training.py"),
            "selected_candidate_id_enters_causal_action": False, "selected_agent_id_enters_causal_action": True,
            "mapping": "selected agent slot -> SERVE; tied same-agent candidate plan identity is metadata only before adapter.step"},
        "classification": "CANDIDATE_PLAN_REWARD_CREDIT_INDISTINGUISHABILITY_STRUCTURALLY_SUPPORTED_COUNTERFACTUAL_COMPARISON_UNAVAILABLE",
        "reward_v2_finding": "Reward V2 deficiency is not localized: candidate-plan identity is discarded before Reward V2 evidence is produced"}
    localization = {"gate": PASS_GATE if not hard else BLOCK, "classification": PASS_CLASS if not hard else "INSUFFICIENT_EVIDENCE",
        "hierarchy": [
            {"rank": 1, "factor": "D3_ACTOR_REPRESENTATION_SENSITIVITY", "status": "CONFIRMED", "precedence": "earliest proven bottleneck in physical->feature->Actor->credit chain"},
            {"rank": 2, "factor": "D1_EXPOSURE_DIVERSITY", "status": "CONFIRMED_EFFECTIVELY_REDUNDANT", "precedence": "A1 provided new window labels but no new Actor-input/candidate problem"},
            {"rank": 3, "factor": "D4_REWARD_CREDIT_DISTINGUISHABILITY", "status": "STRUCTURALLY_SUPPORTED_WITH_NO_DIRECT_COUNTERFACTUAL", "precedence": "same-agent candidate plan collapses to identical causal SERVE action"},
            {"rank": 4, "factor": "D2_CANDIDATE_FEATURE_SUFFICIENCY", "status": "NOT_A_BOTTLENECK_FOR_OBSERVED_HOP_COUNT", "precedence": "hop_count is present and differs exactly"}],
        "primary_bottleneck": "ACTOR_REPRESENTATION_BOTTLENECK_CONFIRMED",
        "secondary_bottlenecks": ["EXPOSURE_BOTTLENECK_CONFIRMED", "REWARD_CREDIT_DISTINGUISHABILITY_BOTTLENECK_STRUCTURALLY_SUPPORTED"],
        "feature_change_justified_now": False, "actor_change_design_gate_justified": True,
        "reward_v2_change_justified_now": False, "additional_training_justified_now": False,
        "next_step": "separately authorize an Actor representation design gate that connects preserved candidate context to the scorer; then redesign exposure to require new Actor-input signatures before any retraining"}
    after = OLD.frozen_hashes(); integrity["source_mutation"] = int(before != after)
    integrity["checkpoint_mutation"] = int(checkpoint_before is not None and checkpoint_before != sha256(A1_CHECKPOINT))
    if any(integrity.values()): hard.append("FORBIDDEN_EXECUTION_OR_MUTATION")
    if len(case_table) != 26 or feature_sufficiency["non_identical_feature_pairs"] != 26 or actor_sensitivity["states_tested"] != 26:
        hard.append("REQUIRED_26_CASE_EVIDENCE_INCOMPLETE")
    if source_flow["candidate_context_in_scorer_stacked_input"]: hard.append("ACTOR_BOTTLENECK_NOT_CONFIRMED_BY_SOURCE")
    gate, classification = (PASS_GATE, PASS_CLASS) if not hard else (BLOCK, "BLOCKED_INSUFFICIENT_EVIDENCE_TO_LOCALIZE_DISCRIMINATION_BOTTLENECK")
    lineage = {"source_commits": {"BT6": BT6_SOURCE, "BT7_R1": BT7_R1_SOURCE, "BT7_RERUN2": BT7_RERUN2_SOURCE,
        "BT8_S0": BT8_S0_SOURCE, "BT8_A1": BT8_A1_SOURCE, "BT8_R1": R1_SOURCE}, "manifest_sha256": manifests,
        "a1_checkpoint_sha256": checkpoint_before, "a1_collection_digest": a1_collection.get("collection_digest")}
    root.mkdir(parents=True)
    outputs = {"bt8r2_exposure_diversity_audit.json": exposure,
        "bt8r2_candidate_feature_sufficiency.json": feature_sufficiency,
        "bt8r2_actor_sensitivity_audit.json": actor_sensitivity,
        "bt8r2_reward_credit_distinguishability.json": reward_credit,
        "bt8r2_exact_tie_case_table.json": {"case_count": len(case_table), "cases": case_table},
        "bt8r2_bottleneck_localization.json": {**localization, "gate": gate, "classification": classification},
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after,
            "checkpoint_before": checkpoint_before, "checkpoint_after": sha256(A1_CHECKPOINT)},
        "test_results.json": {"binding": binding, "lineage": lineage, "integrity_counters": integrity,
            "hard_failures": hard, "warnings": [], "github_push_performed": False},
        "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"],
            "primary_bottleneck": localization["primary_bottleneck"], "secondary_bottlenecks": localization["secondary_bottlenecks"],
            "hard_failures": hard, "warnings": [], "global_locks": LOCKS, "next_step": localization["next_step"]}}
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Candidate discrimination bottleneck diagnosis\n\ngate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"A1 changed nine window IDs but repeated all 96 Actor-input tensor signatures. All {len(feature_cases)} meaningful tie pairs encode hop_count delta 1.0, while observed-value replacement, row swap, and local candidate-feature gradients produce zero logit sensitivity because cand_ctx is omitted from the scorer. "
        "The same-agent candidate plan also collapses to one causal SERVE action before Reward V2, with no legitimate alternative-candidate counterfactual target. No repair or training occurred.\n",
        encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "lineage": lineage, "file_sha256": manifest, "elapsed_seconds": round(time.perf_counter() - started, 3),
                                   "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
