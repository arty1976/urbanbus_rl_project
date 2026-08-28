#!/usr/bin/env python3
"""BT8-R12: freeze a uniform four-cell S3 execution authority.

This stage only reads prior artifacts and source files.  It deliberately does
not deserialize model checkpoints, construct an environment, generate a
candidate, advance a transition, grant a capability, or execute an optimizer.
Its result is a later-execution contract, not an execution grant.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE = "H4M-AE-R9.8-LS3-BT8-R12"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R12_S3_FACTORIAL_EXECUTION_AUTHORITY_SELECTION_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_S3_FOUR_CELL_ON_POLICY_CAUSAL_EXECUTION_READY_FOR_SEPARATE_AUTHORIZATION"
ON_POLICY_BLOCK = "BLOCKED_S3_ON_POLICY_MODE_UNRESOLVED"
CELL_BINDING_BLOCK = "BLOCKED_S3_CELL_BINDING_INCOMPLETE"
REVIEW_BLOCK = "BLOCKED_S3_REVIEW_SUPPORT_CONTRACT_INCOMPLETE"

R11_SOURCE = "f467feef8246c6c77dc67a360aecc60e0102913e"
S3_CONTRACT_SHA256 = "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"
E1_SOURCE = "a3cc98280e434c41520245f3fa13e3a76dd5d438"
R4A_SOURCE = "eac4a209e09e696380bde3bbc437a4fd13c45e99"
R9_SOURCE = "e70b2bdc51674547e7d7e6dc31a6c31c8b6eec56"
F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
R4_SOURCE = "32fbf2b3043166188bde891a5c450702e7db3607"
E1_CONTRACT_SHA256 = "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"
R11_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R11_MINIMAL_SEED_DIVERGENCE_CAUSE_ISOLATION_AND_REPAIR_SELECTION_COMPLETE"
R8_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R8_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
R4A_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4A_F1_EXECUTION_AUTHORITY_COMPLETION"
R9_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R9_E1_SEPARATE_BOUNDED_RETRAINING_AUTHORIZATION_SELECTION_COMPLETE"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
R4_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_FRESH_V2_ACTOR_TRAINING_AND_NOVEL_EXPOSURE_DESIGN_COMPLETE"
S3_CONTRACT_ID = "LS3_BT8_R11_S3_PAIRED_ACTOR_CRITIC_ENV_ELIGIBILITY_V1"
E1_CONTRACT_ID = "LS3_BT8_R7_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_V1"
BRIDGE_CONTRACT_ID = "LS3_BT8_R4A_CANDIDATE_PLAN_CAUSAL_BRIDGE_V1"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R4 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4_v2_actor_fresh_training_design_20260823_122459+0900"
R4A = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4a_f1_execution_authority_completion_20260823_132257+0900"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
R8 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r8_e1_eligibility_validation_20260825_190658+09:00"
R9 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r9_e1_retraining_authority_selection_20260825_200038+09:00"
R11 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection_20260826_123135+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection.py",
    "05_training/test_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
FROZEN_RELATIVE = {
    "gatv2_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2": "rewards/mappo_reward_v1.py",
    "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py",
    "candidate_support_deconfounding": "joint_candidate_support_snapshot.py",
    "causal_bridge": "causal_kpi_bridge.py",
    "r9_8_authorization": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py",
    "joint_assignment_learning": "joint_assignment_learning.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py",
    "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
    "joint_actor_head": "multi_agent_candidate_assignment_head.py",
    "t1_selector": "joint_assignment_frozen_tie_break.py",
}


class R12Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R12Error(code, detail)


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path, *, code: str = CELL_BINDING_BLOCK) -> Any:
    require(path.is_file(), code, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256")
    require(isinstance(expected, Mapping), CELL_BINDING_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in expected.items()
                  if not (root / str(name)).is_file() or sha256(root / str(name)) != str(digest)]
    return {"manifest_sha256": sha256(root / "manifest.json"), "declared_file_count": len(expected),
            "mismatches": mismatches, "all_match": not mismatches}


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R11_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r11": git(["merge-base", R11_SOURCE, "HEAD"]) == R11_SOURCE,
        "changed_files_since_r11": changed,
        "source_only_local_commit": set(changed) == SOURCE_FILES,
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection_{stamp}"


def zero_counters() -> dict[str, int]:
    return {
        "training": 0,
        "causal_rollout": 0,
        "simulator_step": 0,
        "optimizer_step": 0,
        "raw_optimizer_step": 0,
        "checkpoint_load": 0,
        "checkpoint_write": 0,
        "parameter_mutation": 0,
        "candidate_generation": 0,
        "candidate_regeneration": 0,
        "local_search_rerun": 0,
        "zero_loss_reevaluation": 0,
        "review_optimizer_rows": 0,
        "future_leakage": 0,
        "test6_access": 0,
        "github_push": 0,
    }


def frozen_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / relative) for name, relative in FROZEN_RELATIVE.items()}


def make_cells(s3: Mapping[str, Any], checkpoint_manifest: Mapping[str, Any],
               initial_state_digests: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    rows = s3.get("seed_combination_table")
    require(isinstance(rows, list) and len(rows) == 4, CELL_BINDING_BLOCK, "s3_cell_count")
    source_by_id = {str(row.get("cell_id")): row for row in rows}
    require(len(source_by_id) == 4, CELL_BINDING_BLOCK, "s3_cell_id_duplicate")
    expected = (
        ("AC-R1", "AC×F1_R1", "F1_R1", "F1_R1"),
        ("AC-R2", "AC×F1_R2", "F1_R2", "F1_R1"),
        ("BD-R1", "BD×F1_R1", "F1_R1", "F1_R2"),
        ("BD-R2", "BD×F1_R2", "F1_R2", "F1_R2"),
    )
    result: list[dict[str, Any]] = []
    for cell_id, s3_id, environment_replicate, initialization_replicate in expected:
        row = source_by_id.get(s3_id)
        require(isinstance(row, Mapping), CELL_BINDING_BLOCK, f"missing_s3_cell={s3_id}")
        initial = checkpoint_manifest.get(initialization_replicate)
        require(isinstance(initial, Mapping), CELL_BINDING_BLOCK, f"initial_checkpoint={initialization_replicate}")
        state_digests = initial_state_digests.get(initialization_replicate)
        require(isinstance(state_digests, Mapping), CELL_BINDING_BLOCK, f"initial_state_digest={initialization_replicate}")
        expected_seeds = {
            "F1_R1": (20260822, 20260824, 20260826),
            "F1_R2": (20260823, 20260825, 20260827),
        }
        environment_seed, actor_seed, critic_seed = expected_seeds[environment_replicate][0], expected_seeds[initialization_replicate][1], expected_seeds[initialization_replicate][2]
        require((int(row.get("environment_seed")), int(row.get("actor_seed")), int(row.get("critic_seed"))) ==
                (environment_seed, actor_seed, critic_seed), CELL_BINDING_BLOCK, f"seed_binding={cell_id}")
        digest = str(initial.get("sha256"))
        require(str(row.get("actor_initial_checkpoint_sha256")) == digest and
                str(row.get("critic_initial_checkpoint_sha256")) == digest, CELL_BINDING_BLOCK, f"checkpoint_binding={cell_id}")
        result.append({
            "cell_id": cell_id,
            "s3_cell_id": s3_id,
            "environment_replicate": environment_replicate,
            "initialization_replicate": initialization_replicate,
            "environment_seed": environment_seed,
            "actor_seed": actor_seed,
            "critic_seed": critic_seed,
            "actor_initial_checkpoint_sha256": digest,
            "critic_initial_checkpoint_sha256": digest,
            "actor_initial_state_digest": str(state_digests["actor"]),
            "critic_initial_state_digest": str(state_digests["critic"]),
            "parameter_state": "COPY_FROM_BOUND_INITIAL_CHECKPOINT_THEN_INDEPENDENT_PER_CELL",
            "optimizer_state": "FRESH_EMPTY_PER_CELL_NO_REUSE",
        })
    return result


def select_on_policy_mode(cells: Sequence[Mapping[str, Any]], behavior_actor_sha_by_environment: Mapping[str, str]) -> dict[str, Any]:
    require(len(cells) == 4 and set(behavior_actor_sha_by_environment) == {"F1_R1", "F1_R2"}, ON_POLICY_BLOCK, "mode_matrix_scope")
    rows = []
    for cell in cells:
        environment = str(cell["environment_replicate"])
        behavior_sha = str(behavior_actor_sha_by_environment[environment])
        initial_sha = str(cell["actor_initial_checkpoint_sha256"])
        rows.append({
            "cell_id": str(cell["cell_id"]),
            "environment_replicate": environment,
            "preserved_batch_behavior_actor_sha256": behavior_sha,
            "cell_rollout_actor_sha256": initial_sha,
            "strict_sha_match": behavior_sha == initial_sha,
        })
    m0_valid = all(bool(row["strict_sha_match"]) for row in rows)
    diagonal = [row for row in rows if row["cell_id"] in {"AC-R1", "BD-R2"}]
    cross = [row for row in rows if row["cell_id"] in {"AC-R2", "BD-R1"}]
    require(len(diagonal) == 2 and len(cross) == 2, ON_POLICY_BLOCK, "mode_diagonal_partition")
    m1_mixed_origin = True
    m2_preconditions = all(isinstance(row["cell_rollout_actor_sha256"], str) and len(str(row["cell_rollout_actor_sha256"])) == 64 for row in rows)
    require(m2_preconditions, ON_POLICY_BLOCK, "m2_actor_sha")
    return {
        "cell_behavior_actor_sha_matrix": rows,
        "M0": {
            "selected": False,
            "result": "REJECTED_CROSS_CELL_PRESERVED_BATCH_BEHAVIOR_ACTOR_SHA_MISMATCH" if not m0_valid else "NOT_SELECTED",
            "all_four_cells_on_policy": m0_valid,
            "diagonal_cells_strict_match": all(bool(row["strict_sha_match"]) for row in diagonal),
            "cross_cells_strict_match": all(bool(row["strict_sha_match"]) for row in cross),
        },
        "M1": {
            "selected": False,
            "result": "REJECTED_MIXED_PRESERVED_AND_FRESH_EXECUTION_ORIGINS_CONFOUND_FACTORIAL_COMPARISON",
            "diagonal_origin": "PRESERVED_BATCH",
            "cross_origin": "FRESH_CAUSAL_ROLLOUT",
            "uniform_execution_origin": not m1_mixed_origin,
        },
        "M2": {
            "selected": True,
            "result": "SELECTED_UNIFORM_FRESH_CAUSAL_ROLLOUT_PER_CELL",
            "all_cells_origin": "FRESH_CAUSAL_ROLLOUT",
            "on_policy_release_rule": "for every cell, behavior_actor_checkpoint_sha256 == that cell actor_initial_checkpoint_sha256 before its first causal decision",
            "execution_authorized": False,
        },
        "selected_mode": "M2",
    }


def cell_update_budget(cells: Sequence[Mapping[str, Any]], train_window_ids: Sequence[str]) -> dict[str, Any]:
    require(len(cells) == 4 and len(train_window_ids) == 6 and len(set(train_window_ids)) == 6, CELL_BINDING_BLOCK, "budget_scope")
    per_cell = []
    for cell in cells:
        per_cell.append({
            "cell_id": str(cell["cell_id"]),
            "train_window_ids": list(train_window_ids),
            "train_windows": 6,
            "train_trajectories": 6,
            "train_decisions": 24,
            "causal_transitions": 48,
            "ppo_epochs": 3,
            "batch_size": 24,
            "minibatch_size": 24,
            "critic_optimizer_steps": 3,
            "actor_eligibility": {
                "minimum_actor_eligible_rows": 1,
                "eligible_gte_one": {"actor_optimizer_steps": 3, "status": "ACTOR_TRAINED_E1_ELIGIBLE"},
                "eligible_zero": {"actor_optimizer_steps": 0, "status": "ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT"},
            },
            "normalization_and_gae": "trajectory-local GAE then cell-local N0 over its 24 train rows, then E1 eligibility mask",
            "supplemental_steps": 0,
        })
    return {
        "mode": "M2",
        "per_cell": per_cell,
        "aggregate_authorized_maximum": {
            "train_windows_per_cell": 6,
            "decisions": 96,
            "trajectories": 24,
            "causal_transitions": 192,
            "actor_optimizer_steps_maximum": 12,
            "critic_optimizer_steps_exact": 12,
            "raw_optimizer_step_calls_maximum": 24,
            "supplemental_steps": 0,
        },
    }


def review_support_contract(review_collection: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(review_collection.get("collection_digest") == "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e", REVIEW_BLOCK, "review_digest")
    entries = review_collection.get("entries")
    require(isinstance(entries, list) and len(entries) == 6 and len(cells) == 4, REVIEW_BLOCK, "review_counts")
    by_environment: dict[str, list[Mapping[str, Any]]] = {"F1_R1": [], "F1_R2": []}
    for entry in entries:
        seed = int(entry.get("seed", -1))
        environment = "F1_R1" if seed == 20260822 else "F1_R2" if seed == 20260823 else ""
        require(environment, REVIEW_BLOCK, "review_environment_seed")
        by_environment[environment].append(entry)
    result: dict[str, Any] = {}
    used_digests: list[str] = []
    for environment, expected_cells in (("F1_R1", ["AC-R1", "BD-R1"]), ("F1_R2", ["AC-R2", "BD-R2"])):
        rows = sorted(by_environment[environment], key=lambda row: int(row["decision_index"]))
        digests = [str(row["snapshot_digest"]) for row in rows]
        window_ids = [str(row["window_id"]) for row in rows]
        require(len(rows) == 3 and len(set(digests)) == 3 and len(set(window_ids)) == 3, REVIEW_BLOCK, f"review_rows={environment}")
        require({str(cell["cell_id"]) for cell in cells if str(cell["environment_replicate"]) == environment} == set(expected_cells), REVIEW_BLOCK, f"review_cells={environment}")
        result[environment] = {
            "review_snapshot_digests": digests,
            "review_window_ids": window_ids,
            "cells_reusing_exact_same_inputs": expected_cells,
            "snapshot_regeneration": False,
            "candidate_support_recomputation": False,
            "review_optimizer_rows": 0,
        }
        used_digests.extend(digests)
    require(len(set(used_digests)) == 6, REVIEW_BLOCK, "review_unique_total")
    return {
        "collection_digest": str(review_collection["collection_digest"]),
        "unique_snapshot_count": 6,
        "environment_contracts": result,
        "same_environment_ac_bd_identical_inputs": True,
        "cross_environment_candidate_identity_relabeling": False,
        "replay_rule": "persisted review inputs only; no Local Search, Zero-Loss, candidate regeneration, feature or mask recomputation",
    }


def write_block(*, root: Path, source: Mapping[str, Any], preflight: Mapping[str, Any], code: str) -> None:
    outputs = {
        "bt8r12_s3_cell_contract.json": {"not_run": True},
        "bt8r12_on_policy_mode_selection.json": {"not_run": True},
        "bt8r12_cell_update_budget.json": {"not_run": True},
        "bt8r12_review_support_contract.json": {"not_run": True},
        "bt8r12_execution_authority.json": {"not_run": True},
        "test_results.json": {"execution_counters": zero_counters(), "hard_failures": [code], "warnings": []},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED", "source_commit": source["source_commit"],
                               "hard_failures": [code], "warnings": [], "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(f"# BT8-R12 blocked\n\n- gate: `{code}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), CELL_BINDING_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    preflight: dict[str, Any] = {
        "source": source,
        "required_sources": {"r11": R11_SOURCE, "e1": E1_SOURCE, "r4a_bridge": R4A_SOURCE},
        "s3_contract_sha256": S3_CONTRACT_SHA256,
    }
    try:
        manifests = {"r11": manifest_audit(R11), "r8": manifest_audit(R8), "r9": manifest_audit(R9),
                     "r4a": manifest_audit(R4A), "f1": manifest_audit(F1), "r4": manifest_audit(R4)}
        require(all(audit["all_match"] for audit in manifests.values()), CELL_BINDING_BLOCK, "authoritative_manifest")
        r11_gate = load_json(R11 / "gate_decision.json")
        r8_gate = load_json(R8 / "gate_decision.json")
        r9_gate = load_json(R9 / "gate_decision.json")
        r4a_gate = load_json(R4A / "gate_decision.json")
        f1_gate = load_json(F1 / "gate_decision.json")
        r4_gate = load_json(R4 / "gate_decision.json")
        require(r11_gate.get("gate") == R11_GATE and r11_gate.get("source_commit") == R11_SOURCE, CELL_BINDING_BLOCK, "r11_gate")
        require(r8_gate.get("gate") == R8_GATE and r8_gate.get("source_commit") == E1_SOURCE, CELL_BINDING_BLOCK, "e1_gate")
        require(r9_gate.get("gate") == R9_GATE and r9_gate.get("source_commit") == R9_SOURCE, CELL_BINDING_BLOCK, "r9_gate")
        require(r4a_gate.get("gate") == R4A_GATE and r4a_gate.get("source_commit") == R4A_SOURCE, CELL_BINDING_BLOCK, "r4a_gate")
        require(f1_gate.get("gate") == F1_GATE and f1_gate.get("source_commit") == F1_SOURCE, CELL_BINDING_BLOCK, "f1_gate")
        require(r4_gate.get("gate") == R4_GATE and r4_gate.get("source_commit") == R4_SOURCE, CELL_BINDING_BLOCK, "r4_gate")
        s3 = load_json(R11 / "bt8r11_selected_seed_repair_contract.json")
        s3_without_sha = {key: value for key, value in s3.items() if key != "sha256"}
        require(s3.get("contract_id") == S3_CONTRACT_ID and s3.get("selected_option") == "S3" and
                s3.get("sha256") == S3_CONTRACT_SHA256 and canonical_sha256(s3_without_sha) == S3_CONTRACT_SHA256 and
                s3.get("execution_authorized") is False, CELL_BINDING_BLOCK, "s3_contract")
        e1_runtime = load_json(R8 / "bt8r8_e1_runtime_contract.json")
        require(e1_runtime.get("contract_id") == E1_CONTRACT_ID and e1_runtime.get("contract_sha256") == E1_CONTRACT_SHA256 and
                e1_runtime.get("zero_eligible") == "explicit Actor skip; Critic remains active", CELL_BINDING_BLOCK, "e1_contract")
        bridge_contract = load_json(R4A / "bt8r4a_candidate_plan_bridge_contract.json")
        bridge_audit = load_json(R4A / "bt8r4a_candidate_plan_execution_audit.json")
        credit_audit = load_json(R4A / "bt8r4a_credit_identity_audit.json")
        require(bridge_contract.get("contract_id") == BRIDGE_CONTRACT_ID and bridge_contract.get("candidate_regeneration_after_selection") is False and
                bridge_contract.get("serve_fallback") is False and bridge_contract.get("source_state_mutation") is False and
                bridge_audit.get("identity_chain_all") is True and bridge_audit.get("candidate_a_b_distinct_applied_state") is True and
                bridge_audit.get("candidate_a_b_distinct_applied_plan") is True and credit_audit.get("verified") is True,
                CELL_BINDING_BLOCK, "candidate_plan_bridge")
        envelope = load_json(R4 / "bt8r4_selected_training_envelope.json")
        leakage = load_json(F1 / "bt8f1_train_review_leakage_audit.json")
        train_window_ids = [str(row["window_id"]) for row in envelope.get("train_windows", [])]
        review_window_ids = [str(row["window_id"]) for row in envelope.get("review_windows", [])]
        require(len(train_window_ids) == 6 and len(review_window_ids) == 3 and len(set(train_window_ids)) == 6 and
                len(set(review_window_ids)) == 3 and set(train_window_ids).isdisjoint(review_window_ids), CELL_BINDING_BLOCK, "f1_split")
        require(leakage.get("train_windows") == train_window_ids and leakage.get("review_windows") == review_window_ids and
                leakage.get("train_review_overlap") == 0 and leakage.get("review_optimizer_exposure") == 0 and
                leakage.get("future_leakage") == 0, CELL_BINDING_BLOCK, "f1_leakage")
        initial_manifest = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
        initialization_lineage = load_json(F1 / "bt8f1_initialization_lineage.json")
        lineage_rows = {str(row.get("replicate_id")): row for row in initialization_lineage.get("replicates", [])}
        require(initialization_lineage.get("fresh_v2_actor") is True and initialization_lineage.get("fresh_critic") is True and
                initialization_lineage.get("v1_transfer_count") == 0 and initialization_lineage.get("prior_checkpoint_or_optimizer_reuse_count") == 0 and
                set(lineage_rows) == {"F1_R1", "F1_R2"}, CELL_BINDING_BLOCK, "initialization_lineage")
        initial_state_digests = {
            replicate_id: {
                "actor": str(lineage_rows[replicate_id].get("actor_initial_digest")),
                "critic": str(lineage_rows[replicate_id].get("critic_initial_digest")),
            }
            for replicate_id in ("F1_R1", "F1_R2")
        }
        require(all(len(value) == 64 for row in initial_state_digests.values() for value in row.values()), CELL_BINDING_BLOCK, "initialization_state_digest_shape")
        checkpoint_raw_before: dict[str, str] = {}
        for replicate_id in ("F1_R1", "F1_R2"):
            entry = initial_manifest.get(replicate_id)
            require(isinstance(entry, Mapping), CELL_BINDING_BLOCK, f"initial_manifest={replicate_id}")
            path = F1 / "initial_checkpoints" / str(entry.get("path"))
            checkpoint_raw_before[replicate_id] = sha256(path)
            require(checkpoint_raw_before[replicate_id] == entry.get("sha256"), CELL_BINDING_BLOCK, f"initial_checkpoint_sha={replicate_id}")
        cells = make_cells(s3, initial_manifest, initial_state_digests)
        r9_preserved = load_json(R9 / "bt8r9_preserved_batch_completeness.json")
        r9_policy = load_json(R9 / "bt8r9_on_policy_binding.json")
        behavior_sha: dict[str, str] = {}
        for environment in ("F1_R1", "F1_R2"):
            preserved = r9_preserved.get("replicates", {}).get(environment, {})
            policy = r9_policy.get("replicates", {}).get(environment, {})
            behavior = str(preserved.get("initial_actor_critic_checkpoint", {}).get("actor_behavior_checkpoint_sha256"))
            require(behavior == str(policy.get("behavior_policy_actor_checkpoint_sha256")) and
                    policy.get("same_sha_exact") is True and preserved.get("rows", {}).get("train") == 24 and
                    preserved.get("rows", {}).get("trajectory_count") == 6, ON_POLICY_BLOCK, f"preserved_behavior={environment}")
            behavior_sha[environment] = behavior
        mode = select_on_policy_mode(cells, behavior_sha)
        require(mode.get("selected_mode") == "M2" and mode["M0"]["all_four_cells_on_policy"] is False and
                mode["M1"]["uniform_execution_origin"] is False, ON_POLICY_BLOCK, "on_policy_selection")
        training_collection = load_json(F1 / "bt8f1_training_snapshot_collection.json")
        review_collection = load_json(F1 / "bt8f1_review_snapshot_collection.json")
        for collection, expected_digest, expected_count in ((training_collection, "587e45422e0d5cbfcfc0f7279eae34178e00414d21229951ce6683c2aa4785d8", 48),
                                                            (review_collection, "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e", 6)):
            stripped = dict(collection); digest = stripped.pop("collection_digest", None)
            require(digest == expected_digest and canonical_sha256(stripped) == digest and collection.get("snapshot_count") == expected_count,
                    REVIEW_BLOCK if expected_count == 6 else CELL_BINDING_BLOCK, "snapshot_collection")
        review = review_support_contract(review_collection, cells)
        budget = cell_update_budget(cells, train_window_ids)
        expected_frozen = load_json(R11 / "frozen_hash_before_after.json")
        frozen_before = frozen_hashes()
        require(frozen_before == expected_frozen.get("after") and expected_frozen.get("all_unchanged") is True, CELL_BINDING_BLOCK, "frozen_before")
        require(source["source_lineage_descends_from_r11"] and source["source_only_local_commit"], CELL_BINDING_BLOCK, "source_scope")
        preflight |= {
            "manifests": manifests,
            "s3_contract_bound": True,
            "e1_contract_bound": True,
            "candidate_plan_bridge_bound": True,
            "f1_split_bound": True,
            "behavior_actor_sha_by_environment": behavior_sha,
            "checkpoint_raw_digest_verified": checkpoint_raw_before,
            "initial_state_digest_verified": initial_state_digests,
            "review_collection_digest": review["collection_digest"],
            "training_collection_digest": training_collection["collection_digest"],
        }
    except R12Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, code=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}"); return
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, code=CELL_BINDING_BLOCK)
        print(f"[BLOCKED] {CELL_BINDING_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return

    counters = zero_counters()
    frozen_after = frozen_hashes()
    checkpoint_raw_after = {replicate_id: sha256(F1 / "initial_checkpoints" / str(initial_manifest[replicate_id]["path"]))
                            for replicate_id in ("F1_R1", "F1_R2")}
    hard: list[str] = []
    if frozen_before != frozen_after or checkpoint_raw_before != checkpoint_raw_after:
        hard.append(CELL_BINDING_BLOCK)
    if any(value != 0 for value in counters.values()):
        hard.append(CELL_BINDING_BLOCK)
    authority = {
        "selected_mode": "M2",
        "execution_authorized": False,
        "later_explicit_authorization_required": True,
        "future_execution_preconditions": {
            "device": "mps:0",
            "cpu_fallback": 0,
            "behavior_actor_sha_equals_cell_initial_actor_sha": True,
            "fresh_optimizer_per_cell": True,
            "candidate_plan_aware_authoritative_bridge": BRIDGE_CONTRACT_ID,
            "selected_applied_credited_identity": "exact",
            "snapshot_preservation": "100_percent",
            "t1_exact_tie_tolerance": 0,
            "review_optimizer_rows": 0,
        },
        "immutable": {
            "reward_v2": "unchanged",
            "zero_loss": "unchanged",
            "actor_critic_architecture": "unchanged",
            "e1_eligibility": E1_CONTRACT_SHA256,
            "ppo_and_n0": "unchanged_and_cell_local",
            "candidate_regeneration_during_ppo": False,
        },
        "overrun_policy": "fail_closed on an unbound cell/window/seed, decision 97, trajectory 25, transition 193, critic step 13, Actor step 13, raw optimizer call 25, or any supplemental step",
        "global_locks_after_selection": LOCKS,
    }
    gate, classification = (PASS_GATE, PASS_CLASS) if not hard else (hard[0], "BLOCKED")
    outputs = {
        "bt8r12_s3_cell_contract.json": {
            "contract_id": "LS3_BT8_R12_S3_FOUR_CELL_CAUSAL_EXECUTION_V1",
            "s3_contract_sha256": S3_CONTRACT_SHA256,
            "cells": cells,
            "train_window_ids": train_window_ids,
            "review_window_ids": review_window_ids,
            "independence_rule": "same initialization pair is copied from the same bound initial hash; parameters, optimizer, GAE statistics, batch, and later updates are independent per cell",
            "candidate_plan_credit_rule": "selected candidate_id == applied candidate_id == credited candidate_id; no post-selection Local Search regeneration",
            "train_snapshot_rule": "the historical F1 train collection is evidence-only and must not be reused; M2 preserves 24 fresh train decision snapshots per cell (96 total) from that cell's own rollout",
        },
        "bt8r12_on_policy_mode_selection.json": mode,
        "bt8r12_cell_update_budget.json": budget,
        "bt8r12_review_support_contract.json": review,
        "bt8r12_execution_authority.json": authority,
        "test_results.json": {"execution_counters": counters, "hard_failures": hard, "warnings": [],
                              "selected_mode": "M2", "execution_authorized": False, "github_push_performed": False},
        "frozen_hash_before_after.json": {"before": frozen_before, "after": frozen_after,
                                            "all_unchanged": frozen_before == frozen_after,
                                            "initial_checkpoint_raw_digest_before": checkpoint_raw_before,
                                            "initial_checkpoint_raw_digest_after": checkpoint_raw_after,
                                            "initial_checkpoint_unchanged": checkpoint_raw_before == checkpoint_raw_after},
        "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                               "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
                               "next_step": "separate S3 four-cell execution authorization" if not hard else "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(
        f"# BT8-R12 final report\n\n- gate: `{gate}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n"
        "- selected mode: `M2` uniform fresh causal rollout per cell (not executed here)\n"
        "- execution authorization: `false`\n",
        encoding="utf-8",
    )
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "github_push_performed": False, "file_sha256": manifest})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"classification: {classification}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
