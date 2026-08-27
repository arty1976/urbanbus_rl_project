#!/usr/bin/env python3
"""R18-R5: post-update credit to logit direction attribution audit.

This runner reads only existing R18-R2/R18-R3/R18-R4/R17/R16/R15 evidence.
It deliberately has no training, rollout, simulator, candidate generation,
reward recomputation, optimizer, backward, checkpoint-write, mutation, or
promotion path.  If R18-R3 did not persist the row-level credit/GAE/PPO evidence
needed for attribution, the runner must stop with a complete blocked artifact
instead of reconstructing forbidden evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R5"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R5_POST_UPDATE_CREDIT_TO_LOGIT_DIRECTION_ATTRIBUTION_AUDIT_COMPLETE"
UPSTREAM_BLOCK = "BLOCKED_R18R5_UPSTREAM_EVIDENCE_BINDING_FAILURE"
ACTION_IDENTITY_BLOCK = "BLOCKED_R18R5_ACTION_IDENTITY_OR_CREDIT_BINDING_FAILURE"

R18_R3_TRAINING_SOURCE = "7bd0e2e223779455e6112584abd0b5cb6441c228"
R18_R4_REVIEW_SOURCE = "be2683e68eea94699a16604a60d7cea3848a39f4"
R18_R4_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R4_SAME_INPUT_FROZEN_POLICY_INITIAL_FINAL_REVIEW_COMPLETE"
R18_R4_CLASSIFICATION = "B_R18_R3_POLICY_DISTRIBUTION_SHIFT_WITH_STABLE_T1_SELECTIONS"
R18_R3_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_E1_MINIMAL_BOUNDED_TRAINING_AND_CAUSAL_LEARNING_PATH_EVIDENCE_COMPLETE"
R18_R2_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R2_DIGEST_DOMAIN_REPAIR_AND_FROZEN_INFERENCE_EQUIVALENCE_COMPLETE"
R17_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R17_E1_BOUNDED_TRAINING_EXECUTION_AUTHORIZATION_AND_ENVELOPE_FREEZE_COMPLETE"
R16_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R16_FROZEN_POLICY_MASKED_CATEGORICAL_EXPLORATION_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
R15_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R15_MINIMAL_EXPLORATION_DEADLOCK_REPAIR_SELECTION_AUDIT_COMPLETE"
REVIEW_COLLECTION_DIGEST = "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R18_R2 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r2_digest_domain_repair_validation_20260828_005726+09:00"
R18_R3 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r3_e1_bounded_training_execution_20260828_005726+09:00"
R18_R4 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review_20260828_011753+09:00"
R17 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization_20260827_235637+09:00"
R16 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r16_frozen_policy_masked_categorical_exploration_validation_20260827_205833+09:00"
R15 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit_20260827_194419+09:00"

SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r18_r5_credit_to_logit_direction_audit.py",
    "05_training/test_h4m_ae_ls3_bt8_r18_r5_credit_to_logit_direction_audit.py",
}
LOCKS = {
    "training_allowed": False,
    "rollout_allowed": False,
    "simulator_allowed": False,
    "candidate_generation_allowed": False,
    "reward_recomputation_allowed": False,
    "optimizer_creation_allowed": False,
    "optimizer_step_allowed": False,
    "backward_allowed": False,
    "checkpoint_write_allowed": False,
    "policy_mutation_allowed": False,
    "Reward_V2_mutation_allowed": False,
    "Zero_Loss_mutation_allowed": False,
    "GATv2_mutation_allowed": False,
    "TEST6_open_allowed": False,
    "GitHub_push_allowed": False,
}
EXPECTED_ELIGIBLE = {"AC_CONTROL_R1": 9, "BD_E1_R1": 5}
REQUIRED_R18_R3_ROW_EVIDENCE = {
    "eligible_row_credit_trace": (
        "eligible_row_credit_trace.parquet",
        "bt8r13_candidate_plan_credit.json",
        "r18_candidate_plan_credit.json",
    ),
    "credit_eligibility_rows": (
        "bt8r13_credit_eligibility.json",
        "r18_credit_eligibility.json",
    ),
    "gae_rows": (
        "gae_rows.parquet",
        "r18_gae_rows.json",
        "bt8r13_credit_eligibility.json",
    ),
    "ppo_ratio_rows": (
        "ppo_ratio_rows.parquet",
        "r18_ppo_ratio_rows.json",
        "r18_ppo_epoch_direction_audit.json",
    ),
    "per_epoch_loss_contributions": (
        "per_epoch_actor_loss_contributions.parquet",
        "r18_actor_loss_contributions.json",
    ),
}
TRACE_COLUMNS = [
    "arm", "window_id", "time_band", "trajectory_id", "decision_id",
    "selected_semantic_candidate_identity", "no_assign_identity", "candidate_support_digest",
    "selected_action_old_probability", "no_assign_old_probability", "selected_action_old_log_prob",
    "reward_bearing_transitions", "reward_ancestry_class", "assignment_reward",
    "raw_return", "raw_gae_advantage", "normalized_advantage", "ppo_old_log_prob",
    "ppo_new_log_prob", "ppo_ratio", "clip_status", "epoch", "actor_loss_contribution",
    "selected_candidate_logit_delta", "no_assign_logit_delta", "relative_logit_delta",
    "selected_candidate_probability_delta", "no_assign_probability_delta",
    "not_populated_reason",
]


class R18R5Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R18R5Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path, *, code: str = UPSTREAM_BLOCK) -> Any:
    require(path.is_file(), code, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def source_provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff", "--name-only", f"{R18_R4_REVIEW_SOURCE}..HEAD"]).splitlines() if row]
    return {
        "audit_source_commit": git(["rev-parse", "HEAD"]),
        "r18_r4_review_source": R18_R4_REVIEW_SOURCE,
        "lineage_descends_from_r18_r4_review_source": git(["merge-base", R18_R4_REVIEW_SOURCE, "HEAD"]) == R18_R4_REVIEW_SOURCE,
        "changed_files_since_r18_r4_review_source": changed,
        "source_only_r18_r5_audit": set(changed) == SOURCE_FILES,
        "git_status_porcelain": git(["status", "--porcelain=v1"]),
        "github_push_performed": False,
    }


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256")
    require(isinstance(expected, Mapping), UPSTREAM_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in expected.items()
                  if not (root / str(name)).is_file() or sha256(root / str(name)) != str(digest)]
    return {
        "root": str(root),
        "manifest_sha256": sha256(root / "manifest.json"),
        "declared_file_count": len(expected),
        "mismatches": mismatches,
        "all_match": not mismatches,
    }


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        require(isinstance(tensor, torch.Tensor), UPSTREAM_BLOCK, f"checkpoint_actor_tensor={name}")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r5_credit_to_logit_direction_audit_{stamp}"


def counters() -> dict[str, int]:
    return {
        "training": 0,
        "mps_training": 0,
        "rollout": 0,
        "simulator_execution": 0,
        "candidate_generation": 0,
        "reward_recomputation": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward": 0,
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "test6_access": 0,
        "github_push": 0,
        "checkpoint_mutation": 0,
        "frozen_replay_math_rows": 0,
    }


def find_any(root: Path, candidates: Sequence[str]) -> list[str]:
    return [name for name in candidates if (root / name).is_file()]


def row_evidence_availability(r18_root: Path) -> dict[str, Any]:
    availability = {
        key: {"required_alternatives": list(alternatives), "present": find_any(r18_root, alternatives)}
        for key, alternatives in REQUIRED_R18_R3_ROW_EVIDENCE.items()
    }
    missing = [key for key, item in availability.items() if not item["present"]]
    return {
        "required_primary_unit": "each R18-R3 Actor-eligible assignment row",
        "expected_actor_eligible_rows": EXPECTED_ELIGIBLE | {"total": sum(EXPECTED_ELIGIBLE.values())},
        "availability": availability,
        "missing_required_evidence_classes": missing,
        "row_level_trace_possible_without_forbidden_reconstruction": not missing,
        "forbidden_reconstruction_paths": [
            "rerun rollout",
            "rerun simulator",
            "candidate generation",
            "reward recomputation",
            "optimizer creation",
            "optimizer step",
            "backward",
        ],
    }


def bind_checkpoint(path: Path, expected_sha: str, expected_actor_digest: str) -> dict[str, Any]:
    require(path.is_file() and sha256(path) == expected_sha, UPSTREAM_BLOCK, f"checkpoint_sha={path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping) and isinstance(payload.get("actor"), Mapping), UPSTREAM_BLOCK, f"checkpoint_schema={path}")
    actor_digest = state_dict_digest(payload["actor"])
    require(actor_digest == expected_actor_digest, UPSTREAM_BLOCK, f"actor_digest={path}")
    meta = dict(payload.get("meta", {}))
    require(meta.get("test_only") is True and meta.get("bounded") is True and meta.get("non_promotable") is True
            and meta.get("promotion") is False, UPSTREAM_BLOCK, f"checkpoint_flags={path}")
    return {"path": str(path), "sha256": expected_sha, "actor_digest": actor_digest, "meta": meta}


def bind_evidence(source: Mapping[str, Any]) -> dict[str, Any]:
    require(source["lineage_descends_from_r18_r4_review_source"] and source["source_only_r18_r5_audit"]
            and source["git_status_porcelain"] == "", UPSTREAM_BLOCK, "audit_source_scope")
    roots = {"r18_r2": R18_R2, "r18_r3": R18_R3, "r18_r4": R18_R4, "r17": R17, "r16": R16, "r15": R15}
    manifests = {name: manifest_audit(path) for name, path in roots.items()}
    require(all(item["all_match"] for item in manifests.values()), UPSTREAM_BLOCK, "manifest_hash_mismatch")

    gates = {name: load_json(path / "gate_decision.json") for name, path in roots.items()}
    require(gates["r18_r2"].get("gate") == R18_R2_GATE and gates["r18_r2"].get("source_commit") == R18_R3_TRAINING_SOURCE,
            UPSTREAM_BLOCK, "r18_r2_gate")
    require(gates["r18_r3"].get("gate") == R18_R3_GATE and gates["r18_r3"].get("source_commit") == R18_R3_TRAINING_SOURCE,
            UPSTREAM_BLOCK, "r18_r3_gate")
    require(gates["r18_r4"].get("gate") == R18_R4_GATE and gates["r18_r4"].get("classification") == R18_R4_CLASSIFICATION
            and gates["r18_r4"].get("review_source_commit") == R18_R4_REVIEW_SOURCE
            and gates["r18_r4"].get("training_source_commit") == R18_R3_TRAINING_SOURCE, UPSTREAM_BLOCK, "r18_r4_gate")
    require(gates["r17"].get("gate") == R17_GATE, UPSTREAM_BLOCK, "r17_gate")
    require(gates["r16"].get("gate") == R16_GATE, UPSTREAM_BLOCK, "r16_gate")
    require(gates["r15"].get("gate") == R15_GATE, UPSTREAM_BLOCK, "r15_gate")

    auth = load_json(R18_R2 / "r18r3_bounded_training_authorization_manifest.json")
    auth_body = dict(auth)
    auth_sha = str(auth_body.pop("authorization_sha256", ""))
    require(canonical_sha256(auth_body) == auth_sha and auth.get("source_commit") == R18_R3_TRAINING_SOURCE,
            UPSTREAM_BLOCK, "r18_r3_authorization_sha")
    timestamped = dict(dict(auth.get("upstream", {})).get("timestamped_artifacts", {}))
    require(Path(timestamped.get("r15", "")) == R15 and Path(timestamped.get("r16", "")) == R16, UPSTREAM_BLOCK,
            "r15_r16_timestamp_binding")

    execution = load_json(R18_R3 / "r18_execution_manifest.json")
    r3_counters = dict(execution.get("counters", {}))
    expected_r3_counters = {
        "training": 1, "mps_training": 1, "authoritative_candidate_generation": 48,
        "causal_rollout": 96, "simulator_step": 96, "actor_optimizer_step": 6,
        "critic_optimizer_step": 6, "raw_optimizer_step": 12, "checkpoint_write": 2,
    }
    require(all(int(r3_counters.get(key, -1)) == value for key, value in expected_r3_counters.items()),
            UPSTREAM_BLOCK, "r18_r3_expected_counters")
    zero_integrity = (
        "action_support_mutation", "candidate_identity_mismatch", "candidate_plan_execution_collapse",
        "candidate_regeneration_after_selection", "candidate_regeneration_during_ppo", "cross_window_gae",
        "duplicate_reward_ancestry", "future_leakage", "github_push", "illegal_or_masked_selection",
        "local_search_rerun_during_ppo", "nan_or_inf", "review_batch_inclusion", "review_optimizer_rows",
        "review_regeneration", "serve_fallback", "source_state_mutation_during_shadow_evaluation",
        "test6_access", "unauthorized_optimizer_step", "zero_loss_reevaluation_during_ppo", "zero_loss_violation",
    )
    require(all(int(r3_counters.get(key, -1)) == 0 for key in zero_integrity), UPSTREAM_BLOCK, "r18_r3_integrity_counters")

    support = load_json(R18_R3 / "r18_candidate_support_audit.json")
    for arm_id in EXPECTED_ELIGIBLE:
        arm = dict(support.get(arm_id, {}))
        require(arm.get("behavior_equals_update_support") is True and int(arm.get("checked", -1)) == 24
                and int(arm.get("mismatched", -1)) == 0, UPSTREAM_BLOCK, f"support_roundtrip={arm_id}")

    learning = load_json(R18_R3 / "r18_learning_path_audit.json")
    cells = dict(learning.get("cells", {}))
    for arm_id, expected in EXPECTED_ELIGIBLE.items():
        cell = dict(cells.get(arm_id, {}))
        require(int(cell.get("actor_eligible_count", -1)) == expected
                and int(cell.get("assignment_actor_optimizer_steps", -1)) == 3
                and int(cell.get("assignment_critic_optimizer_steps", -1)) == 3
                and cell.get("policy_tensor_delta") is True and cell.get("critic_tensor_delta") is True,
                UPSTREAM_BLOCK, f"learning_path={arm_id}")
    require(learning.get("review_optimizer_exposure") == 0 and learning.get("review_candidate_regeneration") == 0,
            UPSTREAM_BLOCK, "r18_r3_review_exposure")

    checkpoint_manifest = load_json(R18_R3 / "r18_checkpoint_manifest.json")
    final_entries = dict(checkpoint_manifest["final"])
    initial_entries = dict(dict(auth["checkpoint_contract"])["initial_inputs"])
    checkpoints = {
        "AC_CONTROL_R1:initial": bind_checkpoint(
            Path(initial_entries["initial:AC-R1"]["path"]), str(initial_entries["initial:AC-R1"]["sha256"]),
            str(final_entries["AC_CONTROL_R1"]["initial_actor_digest"])),
        "AC_CONTROL_R1:final": bind_checkpoint(
            Path(final_entries["AC_CONTROL_R1"]["path"]), str(final_entries["AC_CONTROL_R1"]["sha256"]),
            str(final_entries["AC_CONTROL_R1"]["final_actor_digest"])),
        "BD_E1_R1:initial": bind_checkpoint(
            Path(initial_entries["initial:BD-R1"]["path"]), str(initial_entries["initial:BD-R1"]["sha256"]),
            str(final_entries["BD_E1_R1"]["initial_actor_digest"])),
        "BD_E1_R1:final": bind_checkpoint(
            Path(final_entries["BD_E1_R1"]["path"]), str(final_entries["BD_E1_R1"]["sha256"]),
            str(final_entries["BD_E1_R1"]["final_actor_digest"])),
    }

    r4_delta = load_json(R18_R4 / "r18r4_initial_final_delta.json")
    r4_discrimination = load_json(R18_R4 / "r18r4_candidate_discrimination.json")
    r4_tests = load_json(R18_R4 / "test_results.json")
    require(dict(r4_tests.get("execution_counters", {})).get("training") == 0
            and dict(r4_tests.get("execution_counters", {})).get("optimizer_step") == 0
            and dict(r4_tests.get("execution_counters", {})).get("checkpoint_mutation") == 0,
            UPSTREAM_BLOCK, "r18_r4_zero_training")
    r4_summary = dict(r4_delta.get("summary", {}))
    require(set(r4_summary) == set(EXPECTED_ELIGIBLE), UPSTREAM_BLOCK, "r18_r4_delta_arms")

    availability = row_evidence_availability(R18_R3)
    return {
        "source": dict(source),
        "roots": {name: str(path) for name, path in roots.items()},
        "manifests": manifests,
        "gates": gates,
        "authorization_sha256": auth_sha,
        "r18_r3_execution_counters": r3_counters,
        "r18_r3_candidate_support_audit": support,
        "r18_r3_learning_path_cells": cells,
        "checkpoint_bindings": checkpoints,
        "r18_r4_delta_summary": r4_summary,
        "r18_r4_candidate_discrimination_summary": dict(r4_discrimination.get("arms", {})),
        "row_evidence_availability": availability,
    }


def blocked_artifacts(*, root: Path, binding: Mapping[str, Any], reason: str) -> None:
    zero = counters()
    missing = dict(binding["row_evidence_availability"])
    qna = {
        "Q1": "Cannot prove per-row selected/executed/credited/PPO identity from R18-R3 artifacts; only aggregate identity counters are bound.",
        "Q2": "Cannot count positive vs negative normalized advantage by eligible row; R18-R3 did not persist eligible-row advantage values.",
        "Q3": "Cannot test sign flips; raw and normalized per-row advantages were not persisted.",
        "Q4": "Cannot determine material clipping effect; per-row ratios/clip status/loss contribution were not persisted.",
        "Q5": "Cannot determine selected-candidate relative-logit consistency with advantage signs; the selected eligible row set and signs are unavailable.",
        "Q6": "R18-R4 shows AC moved away from NO_ASSIGN on fixed snapshots, but attribution to credit/advantage is not evidentially justified.",
        "Q7": "R18-R4 shows BD moved toward NO_ASSIGN on fixed snapshots, but attribution to unfavorable candidate experience or batch dominance is not evidentially justified.",
        "Q8": "Indeterminate: no implementation/credit bug is confirmed, but expected learning direction cannot be proven from persisted R18-R3 evidence.",
        "Q9": "No. Evidence does not justify reward, exploration, or architecture change yet.",
        "Q10": "Minimum next step: run a zero-training evidence-recovery audit only if row-level PPO/GAE evidence exists elsewhere; otherwise rerun the same bounded envelope with durable per-row credit/GAE/PPO trace instrumentation under a new explicit authorization.",
    }
    pd.DataFrame([{column: None for column in TRACE_COLUMNS} | {
        "not_populated_reason": "R18-R3 did not persist row-level eligible credit/GAE/PPO evidence; forbidden reconstruction was not attempted."
    }]).iloc[0:0].to_parquet(root / "eligible_row_credit_trace.parquet", index=False)
    outputs = {
        "evidence_binding_audit.json": binding | {
            "binding_passed_for_available_artifacts": True,
            "required_row_level_evidence_available": False,
            "block_reason": reason,
        },
        "eligible_row_direction_classification.json": {
            "status": "BLOCKED",
            "primary_unit": "R18-R3 actor-eligible assignment row",
            "expected_eligible_rows": EXPECTED_ELIGIBLE | {"total": sum(EXPECTED_ELIGIBLE.values())},
            "traced_eligible_rows": 0,
            "row_evidence_availability": missing,
            "primary_classification": "E_MIXED_OR_INDETERMINATE_REQUIRES_MINIMAL_FOLLOWUP",
            "classification_basis": "Required per-row advantage, identity, ratio, clip, and loss-contribution evidence is absent from R18-R3.",
        },
        "advantage_sign_audit.json": {
            "status": "BLOCKED",
            "available_aggregate_actor_eligible_counts": EXPECTED_ELIGIBLE | {"total": sum(EXPECTED_ELIGIBLE.values())},
            "positive_normalized_advantage_count": None,
            "negative_normalized_advantage_count": None,
            "zero_normalized_advantage_count": None,
            "raw_vs_normalized_sign_flip_count": None,
            "reason": "Raw and normalized per-row advantages were not persisted in R18-R3.",
        },
        "ppo_epoch_direction_audit.json": {
            "status": "BLOCKED",
            "available_optimizer_step_counters": {
                arm: {
                    "actor_steps": int(cell["assignment_actor_optimizer_steps"]),
                    "critic_steps": int(cell["assignment_critic_optimizer_steps"]),
                }
                for arm, cell in dict(binding["r18_r3_learning_path_cells"]).items() if arm in EXPECTED_ELIGIBLE
            },
            "per_epoch_policy_loss": None,
            "entropy_term": None,
            "clip_fraction": None,
            "mean_ratio": None,
            "gradient_norm": None,
            "parameter_update_norm": None,
            "reason": "R18-R3 persisted aggregate step counts but not per-epoch PPO loss/ratio/clip/gradient/update rows.",
        },
        "action_identity_binding_audit.json": {
            "status": "BLOCKED",
            "aggregate_r18_r3_identity_counters": {
                "candidate_identity_mismatch": int(dict(binding["r18_r3_execution_counters"]).get("candidate_identity_mismatch", -1)),
                "candidate_plan_execution_collapse": int(dict(binding["r18_r3_execution_counters"]).get("candidate_plan_execution_collapse", -1)),
                "action_support_mutation": int(dict(binding["r18_r3_execution_counters"]).get("action_support_mutation", -1)),
                "illegal_or_masked_selection": int(dict(binding["r18_r3_execution_counters"]).get("illegal_or_masked_selection", -1)),
            },
            "per_eligible_row_identity_proof_available": False,
            "mismatch_failures": None,
            "reason": "Aggregate zero counters are bound, but 14/14 per-row semantic identity chain was not persisted.",
        },
        "bd_no_assign_shift_attribution.json": {
            "status": "BLOCKED",
            "observed_r18_r4_bd_shift": {
                "mean_no_assign_probability_delta": dict(binding["r18_r4_delta_summary"])["BD_E1_R1"]["mean_no_assign_probability_delta"],
                "mean_best_pair_minus_no_assign_delta": dict(binding["r18_r4_delta_summary"])["BD_E1_R1"]["mean_best_pair_minus_no_assign_delta"],
                "selection_changed_count": dict(binding["r18_r4_delta_summary"])["BD_E1_R1"]["selection_changed_count"],
                "best_pair_changed_count": dict(binding["r18_r4_delta_summary"])["BD_E1_R1"]["best_pair_changed_count"],
            },
            "cause_classification": "E_MIXED_OR_INDETERMINATE_REQUIRES_MINIMAL_FOLLOWUP",
            "ruled_out_by_available_aggregate_evidence": [
                "support/action index mismatch aggregate counter",
                "illegal selection aggregate counter",
                "NaN/Inf aggregate counter",
            ],
            "not_ruled_out_due_missing_row_evidence": [
                "candidate rows had negative advantage",
                "positive candidate rows dominated by other batch gradients",
                "PPO clipping changed effective contribution",
                "normalization sign/magnitude effect",
                "entropy materiality",
                "unexpected per-row gradient/update direction",
            ],
        },
        "ac_bd_update_direction_comparison.json": {
            "status": "PARTIAL_AGGREGATE_ONLY",
            "r18_r4_delta_summary": binding["r18_r4_delta_summary"],
            "available_r18_r3_learning_counts": {
                arm: {
                    "actor_eligible_count": int(cell["actor_eligible_count"]),
                    "actor_steps": int(cell["assignment_actor_optimizer_steps"]),
                    "critic_steps": int(cell["assignment_critic_optimizer_steps"]),
                    "policy_tensor_delta": bool(cell["policy_tensor_delta"]),
                }
                for arm, cell in dict(binding["r18_r3_learning_path_cells"]).items() if arm in EXPECTED_ELIGIBLE
            },
            "attribution_limit": "Direction comparison cannot be mapped to reward ancestry/advantage signs without R18-R3 eligible-row traces.",
        },
        "r18r4_delta_reconciliation.json": {
            "status": "PARTIAL_OBSERVATION_ONLY",
            "r18_r4_classification": R18_R4_CLASSIFICATION,
            "observed": binding["r18_r4_delta_summary"],
            "reconciliation": "R18-R4 fixed-snapshot deltas are bound and internally consistent, but R18-R5 cannot connect them to actual R18-R3 credit/advantage rows.",
            "performance_claim": False,
        },
        "test_results.json": {
            "execution_counters": zero,
            "hard_failures": [reason],
            "warnings": [],
            "training_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_mutation": 0,
            "github_push_performed": False,
        },
        "gate_decision.json": {
            "stage": STAGE,
            "gate": UPSTREAM_BLOCK,
            "classification": "BLOCKED_REQUIRED_R18_R3_ROW_LEVEL_CREDIT_GAE_PPO_EVIDENCE_NOT_PERSISTED",
            "primary_classification_requested": "E_MIXED_OR_INDETERMINATE_REQUIRES_MINIMAL_FOLLOWUP",
            "audit_source_commit": dict(binding["source"])["audit_source_commit"],
            "r18_r3_training_source": R18_R3_TRAINING_SOURCE,
            "r18_r4_review_source": R18_R4_REVIEW_SOURCE,
            "hard_failures": [reason],
            "global_locks": LOCKS,
            "final_questions": qna,
            "next_step": "STOP; no attribution claim without row-level evidence recovery or newly authorized durable-trace rerun",
        },
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(
        "# R18-R5 final report\n\n"
        f"- gate: `{UPSTREAM_BLOCK}`\n"
        "- classification: `BLOCKED_REQUIRED_R18_R3_ROW_LEVEL_CREDIT_GAE_PPO_EVIDENCE_NOT_PERSISTED`\n"
        f"- audit source: `{dict(binding['source'])['audit_source_commit']}`\n"
        f"- R18-R3 training source: `{R18_R3_TRAINING_SOURCE}`\n"
        f"- R18-R4 review source: `{R18_R4_REVIEW_SOURCE}`\n\n"
        "Available upstream artifacts bind cleanly, including R18-R3/R18-R4 manifests, checkpoint SHA/digests, support roundtrip counts, "
        "aggregate optimizer-step counters, and aggregate parameter deltas. The required actor-eligible row-level credit, GAE, PPO ratio, "
        "clip, and loss-contribution traces were not persisted by R18-R3, so this audit does not reconstruct them by rerunning rollout, "
        "candidate generation, simulator steps, reward computation, or optimizer math.\n\n"
        "Answer: BD's fixed-snapshot NO_ASSIGN increase is observed and bound, but cannot be attributed to expected PPO direction versus a "
        "credit/action/normalization/gradient bug from the available evidence alone.\n",
        encoding="utf-8",
    )
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*")
                if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {
        "stage": STAGE,
        "gate": UPSTREAM_BLOCK,
        "classification": "BLOCKED_REQUIRED_R18_R3_ROW_LEVEL_CREDIT_GAE_PPO_EVIDENCE_NOT_PERSISTED",
        "file_sha256": manifest,
        "github_push_performed": False,
    })
    (root / "_BLOCKED.lock").write_text(UPSTREAM_BLOCK + "\n", encoding="utf-8")


def main() -> None:
    root = artifact_root()
    require(not root.exists(), UPSTREAM_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    source = source_provenance()
    try:
        binding = bind_evidence(source)
        availability = dict(binding["row_evidence_availability"])
        if not availability["row_level_trace_possible_without_forbidden_reconstruction"]:
            missing = ", ".join(availability["missing_required_evidence_classes"])
            reason = f"R18-R3 row-level evidence unavailable: {missing}"
            blocked_artifacts(root=root, binding=binding, reason=reason)
            print(f"[BLOCKED] {UPSTREAM_BLOCK}")
            print(f"artifact: {root.relative_to(PROJECT)}")
            return
        raise R18R5Error(UPSTREAM_BLOCK, "unexpected row evidence layout; audited parser not implemented")
    except R18R5Error as exc:
        fallback = {"source": source, "row_evidence_availability": row_evidence_availability(R18_R3)}
        blocked_artifacts(root=root, binding=fallback, reason=str(exc))
        print(f"[BLOCKED] {exc.code}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        fallback = {"source": source, "row_evidence_availability": row_evidence_availability(R18_R3)}
        blocked_artifacts(root=root, binding=fallback, reason=f"{type(exc).__name__}:{exc}")
        print(f"[BLOCKED] {UPSTREAM_BLOCK}")
        print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
