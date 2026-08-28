#!/usr/bin/env python3
"""BT8-R9: choose a zero-rollout E1 retraining authority mode.

This gate deliberately does not instantiate an Actor or Critic, execute a
simulator transition, or create a checkpoint.  It proves whether the frozen
BT8-F1 batch is a complete, behavior-policy-bound PPO batch that can be
reused from the exact initial checkpoint with only the already-validated E1
Actor eligibility mask changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE = "H4M-AE-R9.8-LS3-BT8-R9"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R9_E1_SEPARATE_BOUNDED_RETRAINING_AUTHORIZATION_SELECTION_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_E1_EXACT_PRESERVED_BATCH_RETRAINING_READY_FOR_SEPARATE_EXECUTION"
EVIDENCE_BLOCK = "BLOCKED_E1_RETRAINING_EVIDENCE_INCOMPLETE"
ON_POLICY_BLOCK = "BLOCKED_ON_POLICY_BINDING_FAILURE"
R8_SOURCE = "a3cc98280e434c41520245f3fa13e3a76dd5d438"
R8_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R8_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
R7_SOURCE = "caf91442a2a837c0ea132bbaead26b55d13c4b8f"
R7_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R7_MINIMAL_ACTOR_CRITIC_SEED_FACTORIZATION_AND_CREDIT_ELIGIBILITY_SELECTION_COMPLETE"
R4A_SOURCE = "eac4a209e09e696380bde3bbc437a4fd13c45e99"
R4A_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4A_F1_EXECUTION_AUTHORITY_COMPLETION"
E1_CONTRACT_ID = "LS3_BT8_R7_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_V1"
E1_CONTRACT_SHA256 = "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"
NEXT_GATE = "H4M-AE-R9.8-LS3-BT8-F1-E1_EXACT_PRESERVED_BATCH_E1_BOUNDED_RETRAINING_EXECUTION_AUTHORIZATION"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R8 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r8_e1_eligibility_validation_20260825_190658+09:00"
R7 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility_20260825_124641+09:00"
R6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution_20260824_234227+09:00"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
R4A = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4a_f1_execution_authority_completion_20260823_132257+0900"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r9_retraining_authority_selection.py",
    "05_training/test_h4m_ae_ls3_bt8_r9_retraining_authority_selection.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
REPLICATES = {
    "F1_R1": {
        "environment_seed": 20260822,
        "actor_seed": 20260824,
        "critic_seed": 20260826,
        "actor_eligible": 9,
        "actor_ineligible": 15,
        "actor_steps": 3,
        "critic_steps": 3,
    },
    "F1_R2": {
        "environment_seed": 20260823,
        "actor_seed": 20260825,
        "critic_seed": 20260827,
        "actor_eligible": 0,
        "actor_ineligible": 24,
        "actor_steps": 0,
        "critic_steps": 3,
    },
}


class R9Error(RuntimeError):
    """An unbound preserved batch must fail closed before authorization."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R9Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    return {
        "declared_file_count": len(expected),
        "mismatches": mismatches,
        "all_match": not mismatches,
        "manifest_sha256": sha256(root / "manifest.json"),
    }


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R8_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r8": git(["merge-base", R8_SOURCE, "HEAD"]) == R8_SOURCE,
        "changed_files_since_r8": changed,
        "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r9_e1_retraining_authority_selection_{stamp}"


def zero_counters() -> dict[str, int]:
    return {
        "training": 0,
        "causal_rollout": 0,
        "simulator_step": 0,
        "optimizer_step": 0,
        "raw_optimizer_step": 0,
        "checkpoint_write": 0,
        "checkpoint_load": 0,
        "parameter_mutation": 0,
        "candidate_generation": 0,
        "candidate_regeneration": 0,
        "local_search_rerun": 0,
        "zero_loss_reevaluation": 0,
        "review_row_optimizer_exposure": 0,
        "test6_access": 0,
        "github_push": 0,
    }


def parse_transition_ledger(text: str) -> dict[str, Any]:
    """Extract only persisted scalar/identity evidence from F1's text ledger."""

    def one(name: str, pattern: str) -> str:
        found = re.findall(pattern, text)
        require(len(found) == 1, "PRESERVED_TRANSITION_FIELD_INVALID", name)
        return str(found[0])

    decision_id = one("assignment_step_id", r"assignment_step_id='([^']+)'")
    seed_text = one("seed", r"seed=(\d+)")
    old_log_text = one("old_log_prob", r"old_log_prob=([-+0-9.eE]+)")
    old_log = float(old_log_text)
    require(math.isfinite(old_log), "PRESERVED_OLD_LOG_PROB_NONFINITE", decision_id)
    return {"decision_id": decision_id, "seed": int(seed_text), "old_log_prob": old_log}


def select_retraining_mode(*, p1_complete: bool, on_policy_bound: bool,
                           p2_complete: bool = False) -> dict[str, Any]:
    """P1 is strictly preferred; P2 is never selected while P1 is valid."""
    if p1_complete and on_policy_bound:
        return {
            "selected_mode": "P1",
            "classification": PASS_CLASS,
            "p1": "SELECTED_EXACT_PRESERVED_BATCH_REUSE_NO_CAUSAL_ROLLOUT",
            "p2": "NOT_SELECTED_P1_COMPLETE",
            "p3": "FORBIDDEN_NEW_WINDOWS_OR_ADDITIONAL_TRAINING",
        }
    if p2_complete:
        return {
            "selected_mode": "P2",
            "classification": "B_SUSEONG_LS3_E1_EXACT_CAUSAL_RERUN_REQUIRED",
            "p1": "UNAVAILABLE",
            "p2": "SELECTED_ONLY_BECAUSE_P1_EVIDENCE_INCOMPLETE",
            "p3": "FORBIDDEN_NEW_WINDOWS_OR_ADDITIONAL_TRAINING",
        }
    return {
        "selected_mode": "BLOCKED",
        "classification": "BLOCKED",
        "p1": "UNAVAILABLE",
        "p2": "NOT_AUTHORIZED_WITHOUT_COMPLETE_CAUSAL_RERUN_AUTHORITY",
        "p3": "FORBIDDEN_NEW_WINDOWS_OR_ADDITIONAL_TRAINING",
    }


def expected_budget() -> dict[str, Any]:
    replicates = []
    for replicate_id, contract in REPLICATES.items():
        actor_steps, critic_steps = int(contract["actor_steps"]), int(contract["critic_steps"])
        replicates.append({
            "replicate_id": replicate_id,
            "environment_seed": contract["environment_seed"],
            "actor_initialization_seed": contract["actor_seed"],
            "critic_initialization_seed": contract["critic_seed"],
            "train_rows": 24,
            "trajectories": 6,
            "trajectory_length": 4,
            "ppo_update_cycles": 3,
            "ppo_epochs": 3,
            "full_batch_size": 24,
            "minibatch_size": 24,
            "actor_eligible_rows": contract["actor_eligible"],
            "actor_ineligible_rows": contract["actor_ineligible"],
            "actor_optimizer_steps": actor_steps,
            "critic_optimizer_steps": critic_steps,
            "raw_optimizer_step_calls": actor_steps + critic_steps,
            "zero_eligible_actor_behavior": "EXPLICIT_SKIP" if actor_steps == 0 else "MASKED_ELIGIBLE_MEAN",
            "optimizer_state": "FRESH_EMPTY_PER_REPLICATE_NO_HISTORICAL_REUSE",
            "gae_scope": "each four-decision trajectory; never across trajectories or seeds",
            "normalization": "existing N0 over this replicate's 24 persisted train rows before E1 masking",
        })
    return {
        "contract_id": "LS3_BT8_R9_E1_P1_PRESERVED_BATCH_RETRAINING_V1",
        "mode": "P1",
        "calculation_order": [
            "persisted per-trajectory GAE/return evidence",
            "persisted seed-local N0 full-batch normalization over 24 rows",
            "E1 Actor eligibility mask",
            "Actor policy/entropy mean over eligible non-forced rows",
            "Critic mean over all finite 24 rows",
        ],
        "replicates": replicates,
        "aggregate": {
            "ppo_epochs_per_replicate": 3,
            "full_batch_size_per_replicate": 24,
            "minibatch_size_per_replicate": 24,
            "actor_optimizer_steps": sum(row["actor_optimizer_steps"] for row in replicates),
            "critic_optimizer_steps": sum(row["critic_optimizer_steps"] for row in replicates),
            "raw_optimizer_step_calls": sum(row["raw_optimizer_step_calls"] for row in replicates),
            "extra_compensating_steps": 0,
        },
        "immutable": {
            "reward_v2": "unchanged",
            "gae": "unchanged",
            "n0_normalization": "unchanged",
            "ppo_clip": "unchanged",
            "actor_critic_architecture": "unchanged",
            "t1_selector": "unchanged",
        },
    }


def _by_id(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result = {str(row["decision_id"]): row for row in rows}
    require(len(result) == len(rows), "PRESERVED_DECISION_ID_DUPLICATE", label)
    return result


def _load_n0_rows(normalization: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for replicate_id in REPLICATES:
        rows = normalization[replicate_id]["modes"]["N0"]["rows"]
        require(len(rows) == 24, "PRESERVED_N0_ROW_COUNT_INVALID", replicate_id)
        for row in rows:
            decision_id = str(row["decision_id"])
            require(decision_id not in result, "PRESERVED_N0_DECISION_DUPLICATE", decision_id)
            value = float(row["advantage"])
            require(math.isfinite(value), "PRESERVED_N0_NONFINITE", decision_id)
            result[decision_id] = value
    require(len(result) == 48, "PRESERVED_N0_TOTAL_ROW_COUNT_INVALID")
    return result


def _review_summary(*, collection: Mapping[str, Any], loader: Any) -> dict[str, Any]:
    require(collection.get("store_kind") == "review", "REVIEW_COLLECTION_KIND_INVALID")
    require(collection.get("snapshot_count") == 6 and len(collection.get("entries", [])) == 6,
            "REVIEW_COLLECTION_COUNT_INVALID")
    require(collection.get("collection_schema_version") == "LS3_BT7_FROZEN_POLICY_SNAPSHOT_COLLECTION_V1",
            "REVIEW_COLLECTION_SCHEMA_INVALID")
    by_replicate: dict[str, list[str]] = {replicate_id: [] for replicate_id in REPLICATES}
    for entry in collection["entries"]:
        decision_id = str(entry["decision_id"])
        expected_replicate = next((key for key in REPLICATES if f":{key}:" in decision_id), None)
        require(expected_replicate is not None, "REVIEW_DECISION_REPLICATE_INVALID", decision_id)
        directory = F1 / "bt8f1_review_snapshots" / str(entry["relative_path"])
        payload = loader.load_snapshot(directory)
        metadata = payload["metadata"]
        require(str(metadata["decision_id"]) == decision_id and int(metadata["seed"]) == REPLICATES[expected_replicate]["environment_seed"],
                "REVIEW_SNAPSHOT_IDENTITY_MISMATCH", decision_id)
        require(str(payload["snapshot_digest"]) == str(entry["snapshot_digest"]), "REVIEW_SNAPSHOT_DIGEST_MISMATCH", decision_id)
        by_replicate[expected_replicate].append(str(entry["snapshot_digest"]))
    require(all(len(values) == 3 and len(set(values)) == 3 for values in by_replicate.values()),
            "REVIEW_SNAPSHOT_REPLICATE_COUNT_INVALID")
    return {
        "collection_digest": collection["collection_digest"],
        "snapshot_count": 6,
        "by_replicate": {key: {"count": len(value), "unique": len(set(value)), "digests": value}
                         for key, value in by_replicate.items()},
        "replay_rule": collection["replay_rule"],
        "initial_then_final_same_persisted_inputs": collection["checkpoint_binding"].get("initial_then_final_same_persisted_inputs") is True,
    }


def audit_preserved_batch(*, loader: Any) -> dict[str, Any]:
    """Cross-bind every P1 input from independent immutable F1/R6/R7/R8 records."""
    f1_execution = load_json(F1 / "bt8f1_training_execution_audit.json")
    f1_credit = load_json(F1 / "bt8f1_candidate_plan_credit_audit.json")
    f1_initial = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
    f1_final = load_json(F1 / "bt8f1_final_checkpoint_manifest.json")
    training_collection = load_json(F1 / "bt8f1_training_snapshots" / "collection_manifest.json")
    review_collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
    f1_leakage = load_json(F1 / "bt8f1_train_review_leakage_audit.json")
    r6_rows = [*load_json(R6 / "bt8r6_replicate1_credit_trace.json")["rows"],
               *load_json(R6 / "bt8r6_replicate2_credit_trace.json")["rows"]]
    r7_rows = load_json(R7 / "bt8r7_credit_eligibility_rows.json")["rows"]
    r7_n0 = _load_n0_rows(load_json(R7 / "bt8r7_advantage_normalization_audit.json"))
    r8_masks = load_json(R8 / "bt8r8_eligibility_mask_audit.json")

    require(training_collection.get("store_kind") == "training", "TRAINING_COLLECTION_KIND_INVALID")
    require(training_collection.get("snapshot_count") == 48 and len(training_collection.get("entries", [])) == 48,
            "TRAINING_COLLECTION_COUNT_INVALID")
    require(training_collection.get("collection_schema_version") == "LS3_BT7_FROZEN_POLICY_SNAPSHOT_COLLECTION_V1",
            "TRAINING_COLLECTION_SCHEMA_INVALID")
    require(f1_credit.get("verified") is True and f1_credit.get("mismatches") == 0
            and f1_credit.get("candidate_plan_execution_collapse") == 0
            and f1_credit.get("candidate_regeneration_after_selection") == 0,
            "F1_CREDIT_IDENTITY_AUDIT_INVALID")
    require(f1_leakage.get("review_optimizer_exposure") == 0
            and f1_leakage.get("training_minibatch_review_rows") == 0
            and f1_leakage.get("train_review_overlap") == 0,
            "F1_REVIEW_LEAKAGE_AUDIT_INVALID")

    credit_by_id = _by_id(f1_credit["rows"], "F1 credit")
    trace_by_id = _by_id(r6_rows, "R6 trace")
    eligibility_by_id = _by_id(r7_rows, "R7 eligibility")
    entries_by_id = _by_id(training_collection["entries"], "F1 training snapshots")
    require(set(credit_by_id) == set(trace_by_id) == set(eligibility_by_id) == set(entries_by_id),
            "PRESERVED_DECISION_UNIVERSE_MISMATCH")
    require(len(credit_by_id) == 48, "PRESERVED_DECISION_UNIVERSE_COUNT_INVALID")

    review = _review_summary(collection=review_collection, loader=loader)
    train_windows = set(str(item) for item in f1_leakage["train_windows"])
    review_windows = set(str(item) for item in f1_leakage["review_windows"])
    require(not train_windows & review_windows, "PRESERVED_TRAIN_REVIEW_WINDOW_OVERLAP")

    batch_binding = training_collection.get("checkpoint_binding", {})
    initial_binding = batch_binding.get("initial_actor_checkpoint_sha256_by_replicate", {})
    results: dict[str, Any] = {}
    old_log_all: list[float] = []
    for replicate_id, contract in REPLICATES.items():
        rollout = f1_execution.get("replicate_rollouts", {}).get(replicate_id)
        require(isinstance(rollout, Mapping), "F1_REPLICATE_ROLLOUT_MISSING", replicate_id)
        raw_rows = list(rollout.get("rows", []))
        gae_rows = list(rollout.get("gae_rows", []))
        require(len(raw_rows) == len(gae_rows) == 24, "PRESERVED_REPLICATE_ROW_COUNT_INVALID", replicate_id)
        require(rollout.get("budget") == {"train_samples": 24, "updates": 3}, "F1_REPLICATE_BUDGET_INVALID", replicate_id)
        require(len(rollout.get("window_rows", [])) == 6
                and all(row.get("trajectory_length") == 4 and row.get("causal_transitions") == 8 for row in rollout["window_rows"]),
                "PRESERVED_TRAJECTORY_LAYOUT_INVALID", replicate_id)

        checkpoint = f1_initial.get(replicate_id, {})
        require(checkpoint.get("environment_seed") == contract["environment_seed"]
                and checkpoint.get("actor_init_seed") == contract["actor_seed"]
                and checkpoint.get("critic_init_seed") == contract["critic_seed"],
                "PRESERVED_INITIALIZATION_SEED_MISMATCH", replicate_id)
        checkpoint_path = F1 / "initial_checkpoints" / str(checkpoint.get("path", ""))
        checkpoint_digest = sha256(checkpoint_path)
        require(checkpoint_digest == checkpoint.get("sha256"), "PRESERVED_INITIAL_CHECKPOINT_DIGEST_MISMATCH", replicate_id)
        require(initial_binding.get(replicate_id) == checkpoint_digest, "ON_POLICY_BEHAVIOR_CHECKPOINT_MISMATCH", replicate_id)
        require(checkpoint.get("prior_checkpoint_or_optimizer_reuse_count") == 0 and checkpoint.get("v1_transfer_count") == 0,
                "PRESERVED_INITIAL_CHECKPOINT_NOT_FRESH", replicate_id)

        gae_by_id = _by_id(gae_rows, f"F1 GAE {replicate_id}")
        raw_ids: list[str] = []
        trajectory_ids: dict[str, int] = {}
        support_and_mask_pass = True
        identity_pass = True
        scalar_pass = True
        old_logs: list[float] = []
        snapshot_digests: list[str] = []
        for raw in raw_rows:
            ledger = parse_transition_ledger(str(raw.get("t", "")))
            decision_id = ledger["decision_id"]
            raw_ids.append(decision_id)
            require(decision_id in gae_by_id and decision_id in entries_by_id and decision_id in credit_by_id
                    and decision_id in trace_by_id and decision_id in eligibility_by_id,
                    "PRESERVED_ROW_CROSS_BINDING_MISSING", decision_id)
            require(ledger["seed"] == contract["environment_seed"], "PRESERVED_TRANSITION_SEED_MISMATCH", decision_id)
            require(str(raw.get("snapshot_digest")) == str(entries_by_id[decision_id]["snapshot_digest"]),
                    "PRESERVED_EXECUTION_SNAPSHOT_MISMATCH", decision_id)
            snapshot_digests.append(str(raw["snapshot_digest"]))
            old_logs.append(float(ledger["old_log_prob"]))

            credit, trace, eligibility, gae = (credit_by_id[decision_id], trace_by_id[decision_id],
                                                eligibility_by_id[decision_id], gae_by_id[decision_id])
            require(str(credit["trajectory_id"]) == str(trace["trajectory_id"]) == str(eligibility["trajectory_id"]) == str(gae["trajectory_id"]),
                    "PRESERVED_TRAJECTORY_ID_MISMATCH", decision_id)
            trajectory_id = str(gae["trajectory_id"])
            trajectory_ids[trajectory_id] = trajectory_ids.get(trajectory_id, 0) + 1
            require(str(eligibility["replicate_id"]) == replicate_id and f":{replicate_id}:" in decision_id,
                    "PRESERVED_REPLICATE_ID_MISMATCH", decision_id)
            require(bool(eligibility["identity_chain_valid"]), "PRESERVED_ELIGIBILITY_IDENTITY_FAILURE", decision_id)
            identity = (str(credit["selected_candidate_id"]), str(credit["applied_candidate_id"]), str(credit["credited_candidate_id"]))
            require(identity[0] == identity[1] == identity[2], "PRESERVED_SELECTED_APPLIED_CREDITED_MISMATCH", decision_id)
            require(identity[0] == str(eligibility["selected_candidate_id"])
                    and identity[2] == str(eligibility["credited_candidate_id"]),
                    "PRESERVED_R7_IDENTITY_MISMATCH", decision_id)
            identity_pass = identity_pass and True

            require(float(gae["raw_gae"]) == float(trace["raw_gae"]) == float(eligibility["raw_gae"])
                    == float(credit["GAE advantage"]), "PRESERVED_RAW_GAE_MISMATCH", decision_id)
            require(float(gae["normalized_advantage"]) == float(trace["normalized_advantage"])
                    == float(r7_n0[decision_id]) == float(credit["normalized advantage"]),
                    "PRESERVED_N0_ADVANTAGE_MISMATCH", decision_id)
            require(float(gae["critic_target"]) == float(trace["critic_target"]) == float(credit["critic target"]),
                    "PRESERVED_CRITIC_TARGET_RETURN_MISMATCH", decision_id)
            require(math.isfinite(float(gae["raw_gae"])) and math.isfinite(float(gae["normalized_advantage"]))
                    and math.isfinite(float(gae["critic_target"])), "PRESERVED_CREDIT_SCALAR_NONFINITE", decision_id)
            scalar_pass = scalar_pass and True

            entry = entries_by_id[decision_id]
            directory = F1 / "bt8f1_training_snapshots" / str(entry["relative_path"])
            payload = loader.load_snapshot(directory)
            metadata, tensors = payload["metadata"], payload["tensors"]
            require(str(payload["snapshot_digest"]) == str(entry["snapshot_digest"])
                    and str(metadata["decision_id"]) == decision_id
                    and int(metadata["seed"]) == contract["environment_seed"],
                    "PRESERVED_SNAPSHOT_IDENTITY_MISMATCH", decision_id)
            pair_count = int(metadata["selectable_pair_count"])
            safe_mask = tensors["safe_mask"]
            require(safe_mask.dtype.__str__() == "torch.bool" and safe_mask.shape[0] == 1
                    and safe_mask.shape[1] >= pair_count
                    and bool(safe_mask[0, :pair_count].all().item()),
                    "PRESERVED_SAFE_SUPPORT_MASK_INVALID", decision_id)
            candidate_ids = list(metadata["candidate_ids"])
            require(len(candidate_ids) == pair_count, "PRESERVED_CANDIDATE_SUPPORT_COUNT_INVALID", decision_id)
            if str(eligibility["selected_action_type"]) == "CANDIDATE":
                selected_pair = (str(credit["agent_id"]), str(credit["selected_candidate_id"]))
                positions = [index for index, row in enumerate(candidate_ids)
                             if (str(row["agent_id"]), str(row["candidate_id"])) == selected_pair]
                require(len(positions) == 1 and bool(safe_mask[0, positions[0]].item()),
                        "PRESERVED_SELECTED_ACTION_NOT_IN_LEGAL_SUPPORT", decision_id)
            else:
                require(str(eligibility["selected_action_type"]) == "NO_ASSIGN"
                        and str(credit["selected_candidate_id"]) == "NO_ASSIGN_KEEP_CURRENT_PLANS"
                        and int(metadata["no_assign_index"]) == pair_count,
                        "PRESERVED_NO_ASSIGN_ACTION_CONTRACT_INVALID", decision_id)
            support_and_mask_pass = support_and_mask_pass and True

        require(len(set(raw_ids)) == 24 and set(raw_ids) == set(gae_by_id), "PRESERVED_RAW_GAE_DECISION_SET_MISMATCH", replicate_id)
        require(len(trajectory_ids) == 6 and all(length == 4 for length in trajectory_ids.values()),
                "PRESERVED_TRAJECTORY_COUNT_OR_LENGTH_INVALID", replicate_id)
        require(len(set(snapshot_digests)) == 24, "PRESERVED_SNAPSHOT_UNIQUENESS_INVALID", replicate_id)
        require(all(math.isfinite(value) for value in old_logs), "PRESERVED_OLD_LOG_PROB_NONFINITE", replicate_id)
        r8_replicate = r8_masks.get("replicates", {}).get(replicate_id, {})
        eligible_count = sum(bool(eligibility_by_id[decision_id]["actor_eligible_e1"]) for decision_id in raw_ids)
        require(eligible_count == contract["actor_eligible"]
                and 24 - eligible_count == contract["actor_ineligible"]
                and r8_replicate.get("actor_eligible") == eligible_count
                and r8_replicate.get("actor_ineligible") == 24 - eligible_count
                and r8_replicate.get("critic_eligible") == 24,
                "PRESERVED_E1_MASK_COUNT_MISMATCH", replicate_id)
        old_log_all.extend(old_logs)
        results[replicate_id] = {
            "seeds": {"environment": contract["environment_seed"], "actor": contract["actor_seed"], "critic": contract["critic_seed"]},
            "initial_actor_critic_checkpoint": {
                "relative_path": str(checkpoint["path"]),
                "sha256": checkpoint_digest,
                "actor_behavior_checkpoint_sha256": initial_binding[replicate_id],
                "planned_retraining_start_actor_checkpoint_sha256": checkpoint_digest,
                "planned_retraining_start_critic_checkpoint_sha256": checkpoint_digest,
                "all_sha_equal": checkpoint_digest == initial_binding[replicate_id],
                "historical_reuse_counts": {"v1_transfer": checkpoint["v1_transfer_count"], "prior_checkpoint_or_optimizer": checkpoint["prior_checkpoint_or_optimizer_reuse_count"]},
            },
            "rows": {"train": 24, "unique_decisions": len(set(raw_ids)), "trajectory_count": len(trajectory_ids),
                     "trajectory_lengths": sorted(trajectory_ids.values()), "seed_mixing": 0},
            "old_log_probability": {"count": len(old_logs), "finite": all(math.isfinite(value) for value in old_logs),
                                    "digest": hashlib.sha256(json.dumps(old_logs, separators=(",", ":")).encode()).hexdigest()},
            "credit_fields": {"raw_gae_exact_across_f1_r6_r7_credit": scalar_pass,
                              "n0_normalized_advantage_exact_across_f1_r6_r7_credit": scalar_pass,
                              "critic_target_equals_persisted_return_exact": scalar_pass},
            "action_support_mask": {"all_selected_actions_legal": support_and_mask_pass,
                                    "snapshot_count": len(snapshot_digests), "unique_snapshot_count": len(set(snapshot_digests))},
            "candidate_identity": {"selected_applied_credited_exact": identity_pass, "mismatches": 0},
            "e1": {"actor_eligible": eligible_count, "actor_ineligible": 24 - eligible_count, "critic_eligible": 24},
        }

    require(len(old_log_all) == 48, "PRESERVED_OLD_LOG_TOTAL_ROW_COUNT_INVALID")
    return {
        "p1_evidence_complete": True,
        "training_collection": {
            "collection_digest": training_collection["collection_digest"], "snapshot_count": 48,
            "replay_rule": training_collection["replay_rule"],
            "checkpoint_binding": batch_binding,
        },
        "review_collection": review,
        "train_review_leakage": {"review_optimizer_exposure": 0, "training_minibatch_review_rows": 0,
                                  "train_review_overlap": 0, "future_leakage": int(f1_leakage.get("future_leakage", -1))},
        "replicates": results,
        "e0_final_checkpoint_sha256_by_replicate": {
            key: sha256(F1 / "final_checkpoints" / str(f1_final[key]["path"])) for key in REPLICATES
        },
    }


def audit_on_policy_binding(*, completeness: Mapping[str, Any]) -> dict[str, Any]:
    historical = git(["show", f"{F1_SOURCE}:05_training/run_h4m_ae_ls3_bt8_f1_bounded_training.py"])
    old_log_site = historical.find("old_log = float(JL.masked_log_probs")
    update_site = historical.find("for update_number in range(3):")
    require(old_log_site >= 0 and update_site >= 0 and old_log_site < update_site
            and "old_log_prob=old_log" in historical,
            "ON_POLICY_HISTORICAL_BEHAVIOR_LOG_BINDING_MISSING")
    rows: dict[str, Any] = {}
    for replicate_id in REPLICATES:
        checkpoint = completeness["replicates"][replicate_id]["initial_actor_critic_checkpoint"]
        matches = checkpoint["all_sha_equal"] is True
        require(matches, "ON_POLICY_BEHAVIOR_START_ACTOR_SHA_MISMATCH", replicate_id)
        rows[replicate_id] = {
            "behavior_policy_actor_checkpoint_sha256": checkpoint["actor_behavior_checkpoint_sha256"],
            "planned_retraining_start_actor_checkpoint_sha256": checkpoint["planned_retraining_start_actor_checkpoint_sha256"],
            "same_sha_exact": matches,
            "old_log_probability_rows": completeness["replicates"][replicate_id]["old_log_probability"]["count"],
            "same_preserved_training_input_collection": completeness["training_collection"]["collection_digest"],
            "actor_optimizer_state": "FRESH_EMPTY_NOT_REUSED",
            "critic_optimizer_state": "FRESH_EMPTY_NOT_REUSED",
        }
    return {
        "p1_on_policy_binding_pass": all(row["same_sha_exact"] for row in rows.values()),
        "behavior_policy_capture": "historical F1 records old_log_prob before the three PPO epochs",
        "historical_f1_runner_source_sha256": hashlib.sha256(historical.encode("utf-8")).hexdigest(),
        "replicates": rows,
        "ppo_semantics": "same behavior policy and persisted batch at retraining start; three frozen PPO epochs retain the original F1 epoch count",
        "no_causal_rollout_required": True,
    }


def comparison_contract(*, completeness: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": "LS3_BT8_R9_E0_E1_SAME_SUPPORT_FROZEN_COMPARISON_V1",
        "status": "FROZEN_FOR_FUTURE_SEPARATE_REVIEW_NOT_EXECUTED_IN_R9",
        "baseline_e0": {
            "source": "BT8-F1 historical final checkpoint",
            "initial_checkpoint_sha256_by_replicate": {
                key: completeness["replicates"][key]["initial_actor_critic_checkpoint"]["sha256"] for key in REPLICATES
            },
            "final_checkpoint_sha256_by_replicate": completeness["e0_final_checkpoint_sha256_by_replicate"],
        },
        "future_e1": {
            "start_checkpoint_sha256_by_replicate": {
                key: completeness["replicates"][key]["initial_actor_critic_checkpoint"]["sha256"] for key in REPLICATES
            },
            "training_mode": "P1 persisted batch; E1 mask only",
            "final_checkpoint": "NOT_CREATED_BY_R9; future bounded non-promotable evidence only",
        },
        "same_support_review": {
            "collection_digest": completeness["review_collection"]["collection_digest"],
            "snapshot_count": 6,
            "per_replicate": 3,
            "required_bindings": ["same review snapshot digest", "same actor architecture/config", "same T1 exact-tie tolerance=0", "same support/mask/candidate identity"],
        },
        "review_dimensions_only": ["NO_ASSIGN", "candidate score margin", "exact tie", "agent selection concentration", "seed-specific difference"],
        "prohibited_interpretations": ["KPI comparison", "performance claim", "policy superiority claim", "promotion", "paper-level claim"],
        "budget_digest": canonical_sha256(budget),
    }


def write_block(root: Path, *, source: Mapping[str, Any], code: str, detail: str,
                binding: Mapping[str, Any]) -> None:
    counters = zero_counters()
    files = {
        "bt8r9_retraining_mode_selection.json": {"not_authorized": True, "reason": detail},
        "bt8r9_preserved_batch_completeness.json": {"not_completed": True, "binding": binding},
        "bt8r9_on_policy_binding.json": {"not_completed": True},
        "bt8r9_seed_update_budget.json": {"not_authorized": True},
        "bt8r9_e0_e1_comparison_contract.json": {"not_authorized": True},
        "test_results.json": {"execution_counters": counters, "hard_failures": [detail], "warnings": []},
        "frozen_hash_before_after.json": {"not_completed": True},
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                               "source_commit": source.get("source_commit"), "hard_failures": [detail], "warnings": [],
                               "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, payload in files.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-R9 blocked\n\n- gate: `{code}`\n- detail: `{detail}`\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source.get("source_commit"),
                                   "file_sha256": manifest, "github_push_performed": False})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as snapshots
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution as R6MOD

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), "APPEND_ONLY_ARTIFACT_COLLISION")
    root.mkdir(parents=True)
    binding: dict[str, Any] = {"stage": STAGE, "source": source, "e1_contract_sha256": E1_CONTRACT_SHA256}
    try:
        r8_gate = load_json(R8 / "gate_decision.json")
        r8_contract = load_json(R8 / "bt8r8_e1_runtime_contract.json")
        r8_frozen = load_json(R8 / "frozen_hash_before_after.json")
        r7_gate = load_json(R7 / "gate_decision.json")
        r7_contract = load_json(R7 / "bt8r7_selected_minimal_contract.json")
        f1_gate = load_json(F1 / "gate_decision.json")
        r4a_gate = load_json(R4A / "gate_decision.json")
        r4a_seed = load_json(R4A / "bt8r4a_seed_update_contract.json")
        r4a_ppo = load_json(R4A / "bt8r4a_ppo_partition_contract.json")
        binding.update({
            "r8_gate": r8_gate.get("gate") == R8_GATE and r8_gate.get("source_commit") == R8_SOURCE,
            "r8_manifest": verify_manifest(R8),
            "r8_e1_contract": r8_contract.get("contract_id") == E1_CONTRACT_ID and r8_contract.get("contract_sha256") == E1_CONTRACT_SHA256,
            "r7_gate": r7_gate.get("gate") == R7_GATE and r7_gate.get("source_commit") == R7_SOURCE,
            "r7_manifest": verify_manifest(R7),
            "r7_e1_contract": r7_contract.get("sha256") == E1_CONTRACT_SHA256,
            "f1_gate": f1_gate.get("gate") == F1_GATE and f1_gate.get("source_commit") == F1_SOURCE,
            "f1_manifest": verify_manifest(F1),
            "r4a_gate": r4a_gate.get("gate") == R4A_GATE and r4a_gate.get("source_commit") == R4A_SOURCE,
            "r4a_manifest": verify_manifest(R4A),
            "r4a_seed_ppo_contract": r4a_seed.get("contract_id") == r4a_ppo.get("contract_id") == "LS3_BT8_R4A_F1_SEED_PPO_PARTITION_V1"
                and r4a_ppo.get("ppo_epochs") == 3 and r4a_ppo.get("full_batch_size") == r4a_ppo.get("minibatch_size") == 24,
            "source_lineage": source["source_lineage_descends_from_r8"],
            "source_only_local_commit": source["source_only_local_commit"],
        })
        require(all((binding["r8_gate"], binding["r8_manifest"]["all_match"], binding["r8_e1_contract"],
                     binding["r7_gate"], binding["r7_manifest"]["all_match"], binding["r7_e1_contract"],
                     binding["f1_gate"], binding["f1_manifest"]["all_match"], binding["r4a_gate"],
                     binding["r4a_manifest"]["all_match"], binding["r4a_seed_ppo_contract"],
                     binding["source_lineage"], binding["source_only_local_commit"])),
                "AUTHORITATIVE_BINDING_MISMATCH")
        frozen_before = R6MOD.frozen_hashes(BT6)
        require(frozen_before == r8_frozen.get("r8_after"), "R8_FROZEN_HASH_BINDING_MISMATCH")
        completeness = audit_preserved_batch(loader=snapshots)
        on_policy = audit_on_policy_binding(completeness=completeness)
        mode = select_retraining_mode(p1_complete=bool(completeness["p1_evidence_complete"]),
                                      on_policy_bound=bool(on_policy["p1_on_policy_binding_pass"]))
        require(mode["selected_mode"] == "P1", "ON_POLICY_P1_SELECTION_FAILED")
        budget = expected_budget()
        require(budget["aggregate"] == {"ppo_epochs_per_replicate": 3, "full_batch_size_per_replicate": 24,
                                          "minibatch_size_per_replicate": 24, "actor_optimizer_steps": 3,
                                          "critic_optimizer_steps": 6, "raw_optimizer_step_calls": 9,
                                          "extra_compensating_steps": 0}, "E1_UPDATE_BUDGET_MISMATCH")
        comparison = comparison_contract(completeness=completeness, budget=budget)
        frozen_after = R6MOD.frozen_hashes(BT6)
        require(frozen_before == frozen_after, "READ_ONLY_FROZEN_HASH_MUTATION")
    except Exception as exc:  # noqa: BLE001
        code = ON_POLICY_BLOCK if "ON_POLICY" in str(exc) else EVIDENCE_BLOCK
        write_block(root, source=source, code=code, detail=f"{type(exc).__name__}:{exc}", binding=binding)
        print(f"[BLOCKED] {code}")
        print(f"artifact: {root.relative_to(PROJECT)}")
        return

    counters = zero_counters()
    frozen = {
        "r8_authoritative_after": r8_frozen["r8_after"],
        "before": frozen_before,
        "after": frozen_after,
        "all_unchanged": frozen_before == frozen_after == r8_frozen["r8_after"],
        "intended_e1_learning_path_sha256": frozen_after.get("joint_assignment_learning"),
        "e1_eligibility_module_sha256": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
        "model_or_checkpoint_parameter_mutation": 0,
    }
    outputs = {
        "bt8r9_retraining_mode_selection.json": {
            "selected_mode": mode, "p1_preferred": True, "p1_causal_rollout": 0,
            "p2_condition": "only if P1 is incomplete and an exact causal rerun authority is separately completed",
            "p3": "forbidden", "source_f1": F1_SOURCE,
        },
        "bt8r9_preserved_batch_completeness.json": completeness,
        "bt8r9_on_policy_binding.json": on_policy,
        "bt8r9_seed_update_budget.json": budget,
        "bt8r9_e0_e1_comparison_contract.json": comparison,
        "test_results.json": {
            "authority_binding": binding, "execution_counters": counters, "hard_failures": [], "warnings": [],
            "snapshot_reads_only": 54, "future_leakage": 0, "review_data_used_in_optimizer": 0,
            "TEST6_access": 0, "github_push_performed": False,
        },
        "frozen_hash_before_after.json": frozen,
        "gate_decision.json": {
            "stage": STAGE, "gate": PASS_GATE, "classification": PASS_CLASS,
            "source_commit": source["source_commit"], "hard_failures": [], "warnings": [],
            "global_locks": LOCKS, "next_step": NEXT_GATE,
        },
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(
        "# BT8-R9 final report\n\n"
        f"- gate: `{PASS_GATE}`\n"
        f"- classification: `{PASS_CLASS}`\n"
        f"- source commit: `{source['source_commit']}`\n"
        "- selected mode: `P1`, exact frozen F1 batch reuse from the same initial Actor/Critic checkpoint.\n"
        "- execution: no training, rollout, optimizer step, checkpoint write, parameter mutation, or GitHub push.\n",
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
