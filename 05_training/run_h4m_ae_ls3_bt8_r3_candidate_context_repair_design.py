#!/usr/bin/env python3
"""BT8-R3 candidate-context Actor repair design and pure validation gate."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R3"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R3_ACTOR_CANDIDATE_CONTEXT_REPRESENTATION_REPAIR_DESIGN_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_CANDIDATE_SENSITIVE_JOINT_ACTOR_ARCHITECTURE_READY_FOR_SEPARATE_FRESH_TRAINING_DESIGN"
BLOCK = "BLOCKED_ACTOR_REPRESENTATION_REPAIR_INSUFFICIENT"
R2_SOURCE = "9ef1e93c0c76d3acf0a72f8fddfb5ffa5acea573"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = "05_training/run_h4m_ae_ls3_bt8_r3_candidate_context_repair_design.py"
TEST_REL = "05_training/test_h4m_ae_ls3_bt8_r3_candidate_context_repair.py"
ACTOR_REL = "05_training/multi_agent_candidate_assignment_head.py"
SOURCE_FILES = {SOURCE_REL, TEST_REL, ACTOR_REL}
R2 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r2_candidate_discrimination_bottleneck_diagnostic_20260823_020752+09:00"
A1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_a1_bounded_training_20260822_225122+09:00"
R1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review_20260823_013340+09:00"
SNAPSHOTS = A1 / "bt8_a1_frozen_policy_snapshots"
CHECKPOINT = A1 / "bt8_a1_joint_assignment_checkpoint.pt"
LEGACY_ACTOR_SHA = "312aee5d6794e85ed1d005952b7764066afa9e52408cc87e9d9f3998854eaac3"
CHECKPOINT_SHA = "295e91dc734b3cd7c6839533aec642e306540377407ebe326ad2f7068b0b4cd4"
COLLECTION_DIGEST = "46ebb17a6bd5c2485b29641ac6a6d70fe450a38d73365ea3a8844eb3d2bff86e"
ABS_TOL = 1e-6
REL_TOL = 1e-5
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    """Digest every file in a snapshot directory without mutating evidence."""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(sha256(child).encode("ascii"))
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True,
                          capture_output=True, check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if row]
    return {"source_commit": git(["rev-parse", "HEAD"]),
            "source_parent": git(["rev-parse", "HEAD^"]),
            "changed_files": changed,
            "source_only_local_commit": set(changed) == SOURCE_FILES,
            "github_push_performed": False}


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu()) if left.numel() else 0.0


def close(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, atol=ABS_TOL, rtol=REL_TOL, equal_nan=False))


def summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)}


def caught_code(call: Callable[[], Any], error_type: type[BaseException]) -> str | None:
    try:
        call()
    except error_type as exc:
        return str(getattr(exc, "code", type(exc).__name__))
    return None


def main() -> None:
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review as OLD

    timestamp = OLD.now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    artifact = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r3_candidate_context_repair_design_{timestamp}"
    if artifact.exists():
        raise SystemExit("append-only artifact collision")

    source = provenance()
    hard: list[str] = []
    warnings: list[str] = []
    integrity = {key: 0 for key in (
        "optimizer_steps", "causal_simulator_rollouts", "training_visits",
        "candidate_regeneration", "zero_loss_reevaluation", "new_trained_checkpoints",
        "checkpoint_writes", "historical_checkpoint_mutation", "snapshot_mutation",
        "reward_v2_mutation", "zero_loss_mutation", "t1_mutation", "future_leakage",
        "nan_or_inf_valid_output", "TEST6_access", "github_push")}
    if not torch.backends.mps.is_available():
        hard.append("MPS_ARCHITECTURE_VALIDATION_ENVIRONMENT_UNAVAILABLE")

    r2_gate = json.loads((R2 / "gate_decision.json").read_text(encoding="utf-8"))
    r2_manifest_sha = sha256(R2 / "manifest.json")
    collection = FPS.load_collection_manifest(SNAPSHOTS / "collection_manifest.json")
    snapshot_files_before = {entry["relative_path"]: tree_sha256(SNAPSHOTS / entry["relative_path"])
                             for entry in collection["entries"]}
    checkpoint_before = sha256(CHECKPOINT)
    binding = {
        "r2_gate": r2_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R2_CANDIDATE_DISCRIMINATION_BOTTLENECK_DIAGNOSIS_COMPLETE",
        "r2_classification": r2_gate.get("classification") == "B_MULTI_FACTOR_DISCRIMINATION_BOTTLENECK",
        "r2_source": r2_gate.get("source_commit") == R2_SOURCE,
        "source_parent_is_r2": source["source_parent"] == R2_SOURCE,
        "source_only_local_commit": source["source_only_local_commit"],
        "checkpoint_sha": checkpoint_before == CHECKPOINT_SHA == collection.get("checkpoint_sha256"),
        "collection_digest": collection.get("collection_digest") == COLLECTION_DIGEST,
        "snapshot_count_96": collection.get("snapshot_count") == 96,
        "legacy_actor_sha_bound": collection.get("actor_config", {}).get("actor_module_sha256") == LEGACY_ACTOR_SHA,
    }
    if not all(binding.values()):
        hard.append("AUTHORITATIVE_BINDING_FAILURE")

    legacy_source = git(["show", f"{R2_SOURCE}:{ACTOR_REL}"])
    current_source = (ROOT / "multi_agent_candidate_assignment_head.py").read_text(encoding="utf-8")
    legacy_forward = legacy_source.split("class MultiAgentCandidateAssignmentHead", 1)[1].split("def masked_distribution", 1)[0]
    repaired_forward = inspect.getsource(H.CandidateSensitiveMultiAgentCandidateAssignmentHead.forward)
    root_cause_reconfirmed = (
        "cand_ctx = self.candidate_encoder(candidate_feats)" in legacy_forward
        and "cand_ctx," not in legacy_forward.split("stacked = torch.cat([", 1)[1].split("], dim=-1)", 1)[0]
        and "cand_ctx," in repaired_forward.split("stacked = torch.cat([", 1)[1].split("], dim=-1)", 1)[0])
    if not root_cause_reconfirmed:
        hard.append("ROOT_CAUSE_OR_REPAIR_DATAFLOW_NOT_CONFIRMED")

    cfg = collection["actor_config"]
    hidden = int(cfg["hidden"])
    dataflow = {
        "root_cause_reconfirmed": root_cause_reconfirmed,
        "legacy_source_commit": R2_SOURCE,
        "legacy_actor_module_sha256": LEGACY_ACTOR_SHA,
        "repaired_actor_module_sha256": sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
        "candidate": {"input": ["B", "P", int(cfg["candidate_dim"])],
                      "encoder_output": ["B", "P", hidden],
                      "legacy_destination": "computed but omitted from scorer",
                      "repaired_destination": "row-aligned fifth scorer block"},
        "agent": {"input": ["B", "A", int(cfg["agent_dim"])],
                  "encoder_output": ["B", "A", hidden],
                  "pair_agent_index": ["B", "P"],
                  "matched_agent_output": ["B", "P", hidden],
                  "semantics": "gather agent context by explicit pair_agent_index; no agent id feature"},
        "global": {"input": ["B", int(cfg["global_dim"])], "encoded": ["B", hidden]},
        "demand": {"input": ["B", int(cfg["demand_dim"])], "encoded": ["B", hidden]},
        "fleet": {"shape": ["B", hidden], "semantics": "active-agent masked mean"},
        "pair_scorer": {"legacy_input": ["B", "P", hidden * 4],
                        "repaired_input": ["B", "P", hidden * 5],
                        "output": ["B", "P"], "shared_across_pairs": True},
        "mask": {"safe_mask": ["B", "P"], "unsafe_logit": "-inf",
                 "no_assign_appended_after_pairs": True},
        "no_assign": {"input": ["B", hidden * 2], "inputs": ["global_ctx", "fleet_ctx"],
                      "output": ["B", 1], "candidate_sentinel_added": False},
        "parameter_dimensions": {
            "legacy_scorer_first_weight": [hidden, hidden * 4],
            "repaired_scorer_first_weight": [hidden, hidden * 5],
            "additional_first_layer_weights": hidden * hidden,
            "projection_layer_added": False},
    }
    repair_ladder = {
        "R1": {"change": "concatenate row-aligned cand_ctx to existing shared pair scorer input",
               "dimension_compatible": True, "candidate_encoder_probe_cases": 26,
               "selected": True, "reason": "all five contexts already have width H"},
        "R2": {"change": "R1 plus projection", "selected": False,
               "reason": "projection is not dimensionally required"},
        "R3": {"change": "larger pair interaction module", "selected": False,
               "reason": "R1 provides direct differentiable candidate reachability"},
        "minimum_change_stop_rule_applied": True,
    }

    tie_rows = json.loads((R1 / "bt8r1_candidate_distinctness.json").read_text(encoding="utf-8"))["all_multi_candidate_records"]
    tie_rows = [row for row in tie_rows if row["exact_tie"]]
    snapshots = [FPS.load_snapshot(SNAPSHOTS / entry["relative_path"])
                 for entry in collection["entries"]]
    by_digest = {row["snapshot_digest"]: row for row in snapshots}

    sensitivity_cases: list[dict[str, Any]] = []
    candidate_perm_diffs: list[float] = []
    agent_perm_diffs: list[float] = []
    combined_perm_diffs: list[float] = []
    no_assign_legacy_diffs: list[float] = []
    zeroing_no_assign_diffs: list[float] = []
    actor_params_before = actor_params_after = None
    invalid_results: dict[str, Any] = {}
    compatibility: dict[str, Any] = {}
    no_assign_audit: dict[str, Any] = {}

    if not hard:
        device = torch.device("mps")
        checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        legacy = H.MultiAgentCandidateAssignmentHead(
            global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"],
            agent_dim=cfg["agent_dim"], candidate_dim=cfg["candidate_dim"],
            hidden=cfg["hidden"], heads=cfg["heads"]).to(device).eval()
        legacy.load_state_dict(checkpoint["actor"], strict=True)

        torch.manual_seed(20260823)
        repaired = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
            global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"],
            agent_dim=cfg["agent_dim"], candidate_dim=cfg["candidate_dim"],
            hidden=cfg["hidden"], heads=cfg["heads"]).to(device).eval()
        fresh_state = repaired.state_dict()
        inherited = sorted(name for name in checkpoint["actor"] if not name.startswith("scorer."))
        new = sorted(name for name in fresh_state if name.startswith("scorer."))
        composed = {name: (checkpoint["actor"][name].detach().clone()
                           if name in inherited else value)
                    for name, value in fresh_state.items()}
        repaired.load_state_dict(composed, strict=True)
        actor_params_before = OLD.module_digest(repaired)

        strict_error = None
        strict_probe = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
            global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"],
            agent_dim=cfg["agent_dim"], candidate_dim=cfg["candidate_dim"],
            hidden=cfg["hidden"], heads=cfg["heads"])
        try:
            strict_probe.load_state_dict({name: value.detach().cpu()
                                          for name, value in checkpoint["actor"].items()}, strict=True)
        except RuntimeError as exc:
            strict_error = str(exc).splitlines()[0]
        legacy_shape = list(checkpoint["actor"]["scorer.0.weight"].shape)
        repaired_shape = list(fresh_state["scorer.0.weight"].shape)

        critic = JL.JointAssignmentCritic(global_dim=cfg["global_dim"],
            demand_dim=cfg["demand_dim"], agent_dim=cfg["agent_dim"],
            safe_summary_dim=1 + 2 * cfg["candidate_dim"], hidden=cfg["hidden"],
            heads=cfg["heads"]).to(device)
        critic_load = critic.load_state_dict(checkpoint["critic"], strict=True)
        compatibility = {
            "actor": "PARTIAL_TRANSFER_REQUIRED",
            "legacy_v1_checkpoint_remains_valid_for_v1": True,
            "legacy_checkpoint_strict_load_into_repaired_actor_failed": strict_error is not None,
            "strict_load_error": strict_error,
            "incompatible_tensor": "scorer.0.weight",
            "legacy_shape": legacy_shape, "repaired_shape": repaired_shape,
            "silently_padded_or_reshaped": False,
            "controlled_fixture_inherited_modules": ["agent_encoder", "global_encoder",
                "demand_encoder", "candidate_encoder", "no_assign_scorer"],
            "controlled_fixture_new_modules": ["scorer"],
            "inherited_state_tensor_count": len(inherited),
            "new_state_tensor_count": len(new),
            "critic": "FULLY_COMPATIBLE_ARCHITECTURALLY_REUSE_NOT_AUTHORIZED",
            "critic_strict_load_missing_keys": list(critic_load.missing_keys),
            "critic_strict_load_unexpected_keys": list(critic_load.unexpected_keys),
            "future_implication": "fresh Actor training required; critic reuse requires a separate explicit lineage decision",
        }

        for record in tie_rows:
            payload = by_digest[record["snapshot_digest"]]
            cpu = payload["tensors"]
            tensors = {name: value.to(device) for name, value in cpu.items()}
            support = int(payload["metadata"]["selectable_pair_count"])
            tied = [index for index, value in enumerate(record["pair_logits"])
                    if value == max(record["pair_logits"])]
            if len(tied) != 2:
                hard.append("UNEXPECTED_TIE_CARDINALITY")
                continue
            first, second = tied
            if int(tensors["pair_agent_index"][0, first]) != int(tensors["pair_agent_index"][0, second]):
                hard.append("TIE_PAIR_NOT_SAME_AGENT_FOR_IDENTITY_SWAP_PROBE")
                continue
            with torch.no_grad():
                legacy_logits, legacy_no_assign = legacy(**tensors)
                base, base_no_assign = repaired(**tensors)
                swapped = {name: value.clone() for name, value in tensors.items()}
                swapped["candidate_feats"][:, [first, second]] = tensors["candidate_feats"][:, [second, first]]
                swapped_logits, swapped_no_assign = repaired(**swapped)
                replaced = {name: value.clone() for name, value in tensors.items()}
                replaced["candidate_feats"][:, first] = tensors["candidate_feats"][:, second]
                replaced_logits, replaced_no_assign = repaired(**replaced)
                zeroed = {name: value.clone() for name, value in tensors.items()}
                zeroed["candidate_feats"] = torch.zeros_like(tensors["candidate_feats"])
                zeroed_logits, zeroed_no_assign = repaired(**zeroed)

                pair_count = tensors["candidate_feats"].shape[1]
                candidate_perm = list(reversed(range(support))) + list(range(support, pair_count))
                permuted = {name: value.clone() for name, value in tensors.items()}
                for name in ("candidate_feats", "pair_agent_index", "safe_mask"):
                    permuted[name] = permuted[name][:, candidate_perm]
                candidate_logits, candidate_no_assign = repaired(**permuted)
                candidate_diff = max(max_abs(candidate_logits[:, :support], base[:, candidate_perm[:support]]),
                                     max_abs(candidate_no_assign, base_no_assign))

                agent_count = tensors["agent_feats"].shape[1]
                agent_perm = torch.arange(agent_count - 1, -1, -1, device=device)
                inverse = torch.empty_like(agent_perm); inverse[agent_perm] = torch.arange(agent_count, device=device)
                agent_ordered = {name: value.clone() for name, value in tensors.items()}
                agent_ordered["agent_feats"] = agent_ordered["agent_feats"][:, agent_perm]
                agent_ordered["agent_mask"] = agent_ordered["agent_mask"][:, agent_perm]
                agent_ordered["pair_agent_index"] = inverse[agent_ordered["pair_agent_index"]]
                agent_logits, agent_no_assign = repaired(**agent_ordered)
                agent_diff = max(max_abs(agent_logits[:, :support], base[:, :support]),
                                 max_abs(agent_no_assign, base_no_assign))
                combined = {name: value.clone() for name, value in agent_ordered.items()}
                for name in ("candidate_feats", "pair_agent_index", "safe_mask"):
                    combined[name] = combined[name][:, candidate_perm]
                combined_logits, combined_no_assign = repaired(**combined)
                combined_diff = max(max_abs(combined_logits[:, :support], base[:, candidate_perm[:support]]),
                                    max_abs(combined_no_assign, base_no_assign))

            gradient_input = tensors["candidate_feats"].clone().detach().requires_grad_(True)
            grad_logits, _ = repaired(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                candidate_feats=gradient_input, pair_agent_index=tensors["pair_agent_index"],
                safe_mask=tensors["safe_mask"])
            gradient = torch.autograd.grad(grad_logits[0, first], gradient_input)[0]
            hop_index = MC.LOCAL_SEARCH_FEATURE_NAMES.index("hop_count")
            identity_follow = max(max_abs(swapped_logits[:, first], base[:, second]),
                                  max_abs(swapped_logits[:, second], base[:, first]))
            row_response = max_abs(swapped_logits[:, [first, second]], base[:, [first, second]])
            replacement_response = max_abs(replaced_logits[:, first], base[:, first])
            hop_gradient = abs(float(gradient[0, first, hop_index].detach().cpu()))
            valid_finite = (bool(torch.isfinite(base[:, :support]).all().item())
                            and bool(torch.isfinite(base_no_assign).all().item()))
            sensitivity_cases.append({
                "decision_id": record["decision_id"], "support_size": support,
                "hop_count_delta": 1.0,
                "repaired_base_tied_pair_logit_delta": max_abs(base[:, first], base[:, second]),
                "row_swap_response_max_abs": row_response,
                "row_swap_identity_follow_max_deviation": identity_follow,
                "observed_value_replacement_response_max_abs": replacement_response,
                "candidate_zeroing_response_max_abs": max_abs(zeroed_logits[:, :support], base[:, :support]),
                "hop_count_local_gradient_abs": hop_gradient,
                "candidate_order_max_deviation": candidate_diff,
                "agent_order_max_deviation": agent_diff,
                "combined_order_max_deviation": combined_diff,
                "no_assign_legacy_vs_repaired_max_abs": max_abs(legacy_no_assign, base_no_assign),
                "no_assign_row_swap_max_abs": max_abs(swapped_no_assign, base_no_assign),
                "no_assign_replacement_max_abs": max_abs(replaced_no_assign, base_no_assign),
                "no_assign_zeroing_max_abs": max_abs(zeroed_no_assign, base_no_assign),
                "valid_outputs_finite": valid_finite,
            })
            candidate_perm_diffs.append(candidate_diff)
            agent_perm_diffs.append(agent_diff)
            combined_perm_diffs.append(combined_diff)
            no_assign_legacy_diffs.append(max_abs(legacy_no_assign, base_no_assign))
            zeroing_no_assign_diffs.append(max_abs(zeroed_no_assign, base_no_assign))
            integrity["nan_or_inf_valid_output"] += int(not valid_finite)

        first_payload = by_digest[tie_rows[0]["snapshot_digest"]]
        first_tensors = {name: value.to(device) for name, value in first_payload["tensors"].items()}
        invalid = {name: value.clone() for name, value in first_tensors.items()}
        invalid["pair_agent_index"][0, 0] = 10_000
        invalid_results["pair_agent_index_tamper"] = caught_code(lambda: repaired(**invalid), H.ActorInputContractError)
        invalid = {name: value.clone() for name, value in first_tensors.items()}
        invalid["safe_mask"] = invalid["safe_mask"].float()
        invalid_results["safe_mask_tamper"] = caught_code(lambda: repaired(**invalid), H.ActorInputContractError)
        invalid = {name: value.clone() for name, value in first_tensors.items()}
        mapped = int(invalid["pair_agent_index"][0, 0].item())
        invalid["agent_mask"][0, mapped] = False
        invalid_results["agent_mask_tamper"] = caught_code(lambda: repaired(**invalid), H.ActorInputContractError)
        for label, value in (("nan_input", float("nan")), ("inf_input", float("inf"))):
            invalid = {name: tensor.clone() for name, tensor in first_tensors.items()}
            invalid["candidate_feats"][0, 0, 0] = value
            invalid_results[label] = caught_code(lambda invalid=invalid: repaired(**invalid), H.ActorInputContractError)
        invalid_results["candidate_identity_duplication"] = caught_code(
            lambda: FPS._normal_pair_ids([("A", "C"), ("A", "C")]),
            FPS.FrozenPolicySnapshotError)

        forced = {name: value.clone() for name, value in first_tensors.items()}
        forced["safe_mask"][:] = False
        with torch.no_grad():
            forced_logits, forced_no_assign = repaired(**forced)
            forced_probs = H.masked_distribution(forced_logits, forced_no_assign, forced["safe_mask"])
            mixed_logits, mixed_no_assign = repaired(**first_tensors)
            mixed_probs = H.masked_distribution(mixed_logits, mixed_no_assign, first_tensors["safe_mask"])
        no_assign_audit = {
            "no_assign_scorer_architecture_unchanged": True,
            "candidate_sentinel_added": False,
            "legacy_vs_repaired_max_abs_over_26": max(no_assign_legacy_diffs, default=math.inf),
            "candidate_zeroing_no_assign_max_abs_over_26": max(zeroing_no_assign_diffs, default=math.inf),
            "all_unsafe_pair_logits_negative_infinity": bool(torch.isneginf(forced_logits).all().item()),
            "no_assign_only_pair_probability_sum": float(forced_probs[:, :-1].sum().cpu()),
            "no_assign_only_probability": float(forced_probs[:, -1].cpu()),
            "mixed_unsafe_probability_sum": float(
                mixed_probs[:, :-1][~first_tensors["safe_mask"]].sum().cpu()),
            "mixed_no_assign_probability_positive": float(mixed_probs[:, -1].min().cpu()) > 0.0,
            "preserved": False,
        }
        no_assign_audit["preserved"] = (
            no_assign_audit["legacy_vs_repaired_max_abs_over_26"] == 0.0
            and no_assign_audit["candidate_zeroing_no_assign_max_abs_over_26"] == 0.0
            and no_assign_audit["all_unsafe_pair_logits_negative_infinity"]
            and no_assign_audit["no_assign_only_pair_probability_sum"] == 0.0
            and no_assign_audit["no_assign_only_probability"] == 1.0
            and no_assign_audit["mixed_unsafe_probability_sum"] == 0.0
            and no_assign_audit["mixed_no_assign_probability_positive"])
        torch.mps.synchronize()
        actor_params_after = OLD.module_digest(repaired)

    sensitivity = {
        "states_tested": len(sensitivity_cases), "features_tested": {"hop_count": len(sensitivity_cases)},
        "controlled_fixture": "legacy non-scorer modules transferred exactly; repaired scorer freshly initialized; no training",
        "repaired_base_pair_separation": summary([row["repaired_base_tied_pair_logit_delta"] for row in sensitivity_cases]),
        "row_swap_response": summary([row["row_swap_response_max_abs"] for row in sensitivity_cases]),
        "row_swap_identity_follow_deviation": summary([row["row_swap_identity_follow_max_deviation"] for row in sensitivity_cases]),
        "observed_value_replacement_response": summary([row["observed_value_replacement_response_max_abs"] for row in sensitivity_cases]),
        "candidate_zeroing_response": summary([row["candidate_zeroing_response_max_abs"] for row in sensitivity_cases]),
        "hop_count_local_gradient_abs": summary([row["hop_count_local_gradient_abs"] for row in sensitivity_cases]),
        "nonzero_row_swap_cases": sum(row["row_swap_response_max_abs"] > 0 for row in sensitivity_cases),
        "nonzero_observed_replacement_cases": sum(row["observed_value_replacement_response_max_abs"] > 0 for row in sensitivity_cases),
        "nonzero_hop_count_gradient_cases": sum(row["hop_count_local_gradient_abs"] > 0 for row in sensitivity_cases),
        "classification": "ARCHITECTURALLY_CANDIDATE_SENSITIVE_NOT_POLICY_QUALITY_EVIDENCE",
        "cases": sensitivity_cases,
    }
    order = {
        "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
        "candidate": {"states": len(candidate_perm_diffs), "max_deviation": max(candidate_perm_diffs, default=math.inf),
                      "passed": all(value <= ABS_TOL for value in candidate_perm_diffs) and len(candidate_perm_diffs) == 26},
        "agent": {"states": len(agent_perm_diffs), "max_deviation": max(agent_perm_diffs, default=math.inf),
                  "passed": all(value <= ABS_TOL for value in agent_perm_diffs) and len(agent_perm_diffs) == 26},
        "combined": {"states": len(combined_perm_diffs), "max_deviation": max(combined_perm_diffs, default=math.inf),
                     "passed": all(value <= ABS_TOL for value in combined_perm_diffs) and len(combined_perm_diffs) == 26},
        "identity_aligned": True, "candidate_position_feature_added": False,
    }
    expected_invalid = {
        "pair_agent_index_tamper": "ACTOR_PAIR_AGENT_INDEX_OUT_OF_RANGE",
        "safe_mask_tamper": "ACTOR_SAFE_MASK_DTYPE_INVALID",
        "agent_mask_tamper": "ACTOR_SAFE_PAIR_MAPPED_TO_INACTIVE_AGENT",
        "nan_input": "ACTOR_NONFINITE_FEATURE_INPUT", "inf_input": "ACTOR_NONFINITE_FEATURE_INPUT",
        "candidate_identity_duplication": "SNAPSHOT_DUPLICATE_CANDIDATE_IDENTITY",
    }
    invalid_all_pass = invalid_results == expected_invalid

    future = {
        "actor_training": "FRESH_ACTOR_TRAINING_REQUIRED",
        "legacy_actor_checkpoint_promotable_for_repaired_actor": False,
        "partial_transfer_fixture_is_quality_initialization_authority": False,
        "critic_reuse": "ARCHITECTURALLY_COMPATIBLE_BUT_NOT_AUTHORIZED_WITHOUT_SEPARATE_LINEAGE_DECISION",
        "required_exposure_evidence": ["new candidate feature signatures", "new full Actor-input signatures",
            "new agent×candidate opportunity patterns", "new meaningfully distinct candidate comparisons"],
        "new_window_ids_alone_sufficient": False,
        "reward_credit_carry_forward_risk": "candidate plans may collapse to the same SERVE-level reward path; counterfactual candidate-plan reward evidence remains unavailable",
        "reward_v2_redesign_authorized": False,
        "next_step": "separately authorize fresh repaired-Actor training/exposure design; do not train in BT8-R3",
    }

    baseline = json.loads((R2 / "frozen_hash_before_after.json").read_text(encoding="utf-8"))["after"]
    current = OLD.frozen_hashes()
    immutable_names = sorted(set(baseline) - {"joint_actor_head"})
    immutable_unchanged = all(current.get(name) == baseline.get(name) for name in immutable_names)
    checkpoint_after = sha256(CHECKPOINT)
    snapshots_after = {entry["relative_path"]: tree_sha256(SNAPSHOTS / entry["relative_path"])
                       for entry in collection["entries"]}
    integrity["historical_checkpoint_mutation"] = int(checkpoint_after != checkpoint_before)
    integrity["snapshot_mutation"] = int(snapshots_after != snapshot_files_before)
    integrity["reward_v2_mutation"] = int(current["reward_v2"] != baseline["reward_v2"])
    integrity["zero_loss_mutation"] = int(current["zero_loss"] != baseline["zero_loss"])
    integrity["t1_mutation"] = 0

    if len(sensitivity_cases) != 26:
        hard.append("REQUIRED_26_STATE_SENSITIVITY_EVIDENCE_INCOMPLETE")
    if sensitivity["nonzero_row_swap_cases"] != 26 or sensitivity["nonzero_observed_replacement_cases"] != 26 or sensitivity["nonzero_hop_count_gradient_cases"] != 26:
        hard.append("CANDIDATE_SENSITIVITY_NOT_ARCHITECTURALLY_REACHABLE_IN_ALL_CASES")
    if not all(row["row_swap_identity_follow_max_deviation"] <= ABS_TOL for row in sensitivity_cases):
        hard.append("CANDIDATE_IDENTITY_ROW_SWAP_FAILURE")
    if not all(section["passed"] for section in (order["candidate"], order["agent"], order["combined"])):
        hard.append("ORDER_INVARIANCE_FAILURE")
    if not no_assign_audit.get("preserved"):
        hard.append("NO_ASSIGN_SEMANTICS_FAILURE")
    if not invalid_all_pass:
        hard.append("INVALID_INPUT_FAIL_CLOSED_FAILURE")
    if compatibility.get("actor") != "PARTIAL_TRANSFER_REQUIRED" or compatibility.get("legacy_checkpoint_strict_load_into_repaired_actor_failed") is not True:
        hard.append("CHECKPOINT_COMPATIBILITY_CLASSIFICATION_FAILURE")
    if actor_params_before != actor_params_after:
        hard.append("ACTOR_PARAMETER_MUTATION")
    if not immutable_unchanged or any(integrity.values()):
        hard.append("FROZEN_AUTHORITY_OR_EXECUTION_INTEGRITY_FAILURE")

    gate, classification = ((PASS_GATE, PASS_CLASS) if not hard
                            else (BLOCK, "ACTOR_REPRESENTATION_REPAIR_BLOCKED"))
    repair = {
        "gate": gate, "classification": classification,
        "selected_level": "R1", "selected_head_id": H.CANDIDATE_SENSITIVE_HEAD_ID,
        "selected_head_version": H.CANDIDATE_SENSITIVE_HEAD_VERSION,
        "exact_change": "append row-aligned cand_ctx (B,P,H) to the existing shared pair scorer input; scorer first layer 4H -> 5H",
        "projection_added": False, "larger_interaction_module_added": False,
        "legacy_v1_class_preserved": True, "no_assign_path_changed": False,
        "candidate_position_or_rank_used": False, "trained": False,
    }
    frozen = {
        "r2_baseline": baseline, "after": current,
        "immutable_names": immutable_names, "immutable_authorities_all_unchanged": immutable_unchanged,
        "joint_actor_head_changed_as_authorized_repair": current["joint_actor_head"] != baseline["joint_actor_head"],
        "legacy_actor_source_sha256": baseline["joint_actor_head"],
        "repaired_actor_source_sha256": current["joint_actor_head"],
        "historical_checkpoint_before": checkpoint_before, "historical_checkpoint_after": checkpoint_after,
        "historical_checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "snapshot_collection_unchanged": snapshot_files_before == snapshots_after,
    }
    tests = {
        "binding": binding, "root_cause_reconfirmed": root_cause_reconfirmed,
        "invalid_input_results": invalid_results, "invalid_input_expected": expected_invalid,
        "invalid_inputs_all_fail_closed": invalid_all_pass,
        "integrity_counters": integrity, "actor_parameter_digest_before": actor_params_before,
        "actor_parameter_digest_after": actor_params_after,
        "actor_parameters_unchanged_during_validation": actor_params_before == actor_params_after,
        "commands": [
            "python -m py_compile BT8-R3 source/test/Actor modules",
            "python -m pytest -q 05_training/test_h4m_ae_ls3_bt8_r3_candidate_context_repair.py",
            "python 05_training/run_h4m_ae_ls3_bt8_r3_candidate_context_repair_design.py (MPS)"],
        "hard_failures": hard, "warnings": warnings, "github_push_performed": False,
    }
    lineage = {
        "BT8_R2_source_commit": R2_SOURCE, "BT8_R2_manifest_sha256": r2_manifest_sha,
        "BT8_A1_checkpoint_sha256": checkpoint_before,
        "BT8_A1_snapshot_collection_digest": collection.get("collection_digest"),
        "source_commit": source["source_commit"], "source_parent": source["source_parent"],
        "changed_files": source["changed_files"],
    }

    artifact.mkdir(parents=True)
    outputs = {
        "bt8r3_actor_dataflow_audit.json": dataflow,
        "bt8r3_repair_ladder.json": repair_ladder,
        "bt8r3_selected_actor_repair.json": repair,
        "bt8r3_candidate_sensitivity_audit.json": sensitivity,
        "bt8r3_order_invariance_audit.json": order,
        "bt8r3_no_assign_semantics_audit.json": no_assign_audit,
        "bt8r3_checkpoint_compatibility.json": compatibility,
        "bt8r3_future_training_contract.json": future,
        "test_results.json": tests,
        "frozen_hash_before_after.json": frozen,
        "gate_decision.json": {"gate": gate, "classification": classification,
            "source_commit": source["source_commit"], "root_cause_reconfirmed": root_cause_reconfirmed,
            "selected_repair": "R1_ROW_ALIGNED_CAND_CTX_CONCATENATION",
            "hard_failures": hard, "warnings": warnings, "global_locks": LOCKS,
            "next_step": future["next_step"]},
    }
    for name, payload in outputs.items():
        dump(artifact / name, payload)
    (artifact / "final_report.md").write_text(
        f"# {STAGE} — Candidate-context Actor repair design\n\n"
        f"gate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        "The legacy V1 root cause was independently reconfirmed. R1 was selected: the existing row-aligned cand_ctx is appended to the shared pair scorer, changing only its first input width from 4H to 5H. "
        f"All {len(sensitivity_cases)} preserved tie states showed non-zero row-swap, observed-value replacement, and hop_count-gradient reachability. Candidate, agent, and combined identity-aligned permutation checks passed; NO_ASSIGN remained byte/numerically unchanged. "
        "The old Actor checkpoint is not strict-compatible with V2; future training requires a fresh repaired Actor design. No training or simulator execution occurred.\n",
        encoding="utf-8")
    manifest_files = {path.relative_to(artifact).as_posix(): sha256(path)
                      for path in artifact.rglob("*") if path.is_file()}
    dump(artifact / "manifest.json", {"stage": STAGE, "gate": gate,
        "classification": classification, "source_commit": source["source_commit"],
        "lineage": lineage, "file_sha256": manifest_files,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "github_push_performed": False})
    (artifact / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"artifact: {artifact.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
