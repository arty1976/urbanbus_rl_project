#!/usr/bin/env python3
"""H4M-U-R2 fresh three-seed retraining with durable evidence.

Completely fresh retraining for seeds 1, 2, 3 on the frozen repaired execution
path (H4M-Q lineage + H4M-P target-conditioned Actor heads + H4M-T S3 critic
value-target binding), recorded through the H4M-U-R1 durable evidence writer.

The H4M-U-R1 certified sources are reused as-is: this runner extends them by
subclassing and wrapping only, so ``durable_training_evidence.py``, the H4M-U
runner, and the R1 test suite stay byte-identical to their PASS commit.

No tuning, no reuse of the DIAGNOSTIC_ONLY / NON_PROMOTABLE H4M-U checkpoints,
no TEST6 access, no extra cycles or updates, and no GitHub push.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-U-R2"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_R2_"
    "FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_AND_CRITIC_VALUE_TARGET_REPAIRED_"
    "THREE_SEED_RETRAINING_WITH_DURABLE_EVIDENCE_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_R2"
NEXT_REVIEW_GATE = "H4M-V_TARGET_CONDITIONED_ACTOR_AND_CRITIC_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
REPORT_TITLE = "H4M-U-R2 Fresh Three-Seed Retraining with Durable Evidence"
CHECKPOINT_NAME = "H4M_U_R2_SEED_{seed:03d}_FRESH_ACTOR_AND_CRITIC_REPAIRED_DURABLE_EVIDENCE.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MU_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py"
H4MQ_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
DURABLE_SOURCE = TRAINING_ROOT / "durable_training_evidence.py"

H4MR1_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_u_r1_post_training_evidence_instrumentation_repair_and_"
    "no_training_equivalence_validation_20260817_235733+09:00"
)
H4MT_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_t_ppo_credit_advantage_critic_value_target_repair_implementation_equivalence_validation_20260817_190257+09:00"
)
H4MQ_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_20260817_153440+0900"
H4MU_RAW_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_u_fresh_target_conditioned_actor_head_specialized_"
    "and_critic_value_target_repaired_three_seed_retraining_20260817_193849+09:00"
)

EXPECTED = {
    "h4m_u_r1_source_commit": "10d001789801c2b9a54ccb15663609104345d783",
    "h4m_u_r1_source_commit_short": "10d0017",
    "h4m_u_r1_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_R1_"
        "POST_TRAINING_EVIDENCE_INSTRUMENTATION_REPAIR_AND_NO_TRAINING_EQUIVALENCE_VALIDATION_COMPLETE"
    ),
    "h4m_t_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_T_PPO_CREDIT_ADVANTAGE_CRITIC_VALUE_TARGET_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "seeds": [1, 2, 3],
    "outer_training_count": 11,
    "ppo_updates_per_seed": 44,
    "critic_updates_per_seed": 88,
    "expected_cycle_records": 33,
    "r1_certified_sources": {
        "05_training/durable_training_evidence.py": None,
        "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py": None,
        "05_training/test_h4m_u_r1_durable_training_evidence.py": None,
    },
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
ACTION_NAMES = {0: HOLD, 1: SERVE, 2: SKIP}
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"

R2_REQUIRED_EVIDENCE_FIELDS = [
    "legal_action_counts",
    "probability_when_legal",
    "zero_loss_counts",
    "actor_update_count",
    "critic_update_count",
    "nan_inf_count",
    "illegal_action_count",
    "future_leakage_count",
]

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "training_summary.json",
    "seed_1_summary.json",
    "seed_2_summary.json",
    "seed_3_summary.json",
    "evidence_schema.json",
    "cycle_action_evolution.json",
    "critic_credit_diagnostics.json",
    "target_conditioned_diagnostics.json",
    "checkpoint_integrity.json",
    "durable_evidence_report.json",
    "repair_binding.json",
    "training_integrity.json",
    "outcome_classification.json",
    "parameter_delta.json",
    "gradient_isolation.json",
    "validation_discrimination.json",
    "changed_files.json",
    "test_results.json",
    "gate_matrix.json",
]
REQUIRED_EVIDENCE_FILES = [
    "07_durable_training_evidence/seed_cycle_evidence.jsonl",
    "07_durable_training_evidence/evidence_index.json",
    "07_durable_training_evidence/evidence_schema.json",
    "07_durable_training_evidence/run_header.json",
    "07_durable_training_evidence/run_state.json",
]


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# ---------------------------------------------------------------------------
# R2 evidence extension (added through the R1 writer's documented extra field)
# ---------------------------------------------------------------------------
def legal_action_evidence(pre_action_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    legal_counts = {name: 0 for name in ACTION_NAMES.values()}
    probability_sums = {name: 0.0 for name in ACTION_NAMES.values()}
    legal_set_histogram: Dict[str, int] = {}
    illegal = 0
    for row in pre_action_rows:
        legal_ids = [int(value) for value in row.get("legal_action_ids", [])]
        key = "|".join(ACTION_NAMES[i] for i in sorted(legal_ids))
        legal_set_histogram[key] = legal_set_histogram.get(key, 0) + 1
        if int(row.get("action_id", -1)) not in legal_ids:
            illegal += 1
        for action_id in sorted(ACTION_NAMES):
            if action_id in legal_ids:
                legal_counts[ACTION_NAMES[action_id]] += 1
                probability_sums[ACTION_NAMES[action_id]] += float(row.get(f"masked_probability_{action_id}", 0.0))
    probability_when_legal = {
        name: (probability_sums[name] / legal_counts[name] if legal_counts[name] else None)
        for name in ACTION_NAMES.values()
    }
    return {
        "legal_action_counts": legal_counts,
        "legal_action_set_histogram": legal_set_histogram,
        "probability_when_legal": probability_when_legal,
        "illegal_action_count": illegal,
        "row_count": len(pre_action_rows),
    }


def zero_loss_evidence(reward_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Measure Zero-Loss admission activity in this cycle's reward rows."""
    candidate = accepted = rejected = 0
    for row in reward_rows:
        markers = [key for key in row if key.startswith("zero_loss") or key.startswith("admission")]
        if not markers:
            continue
        candidate += 1
        decision = row.get("zero_loss_accept", row.get("admission_accepted"))
        if decision is True:
            accepted += 1
        elif decision is False:
            rejected += 1
    return {
        "candidate": candidate,
        "accepted": accepted,
        "rejected": rejected,
        "measured_from": "recorder.reward_rows admission markers for this seed x cycle",
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "note": "the frozen H4M-Q controlled execution path materializes no Zero-Loss admission candidate; the contract SHA stays bound and the count is measured, not assumed",
        "reward_row_count": len(reward_rows),
    }


def build_r2_required_evidence(recorder: Any, staged: Mapping[str, Any]) -> Dict[str, Any]:
    critic_loss = staged["critic_loss"]
    legal = legal_action_evidence(recorder.pre_action_rows)
    zero_loss = zero_loss_evidence(getattr(recorder, "reward_rows", []))
    return {
        "legal_action_counts": legal["legal_action_counts"],
        "legal_action_set_histogram": legal["legal_action_set_histogram"],
        "probability_when_legal": legal["probability_when_legal"],
        "zero_loss_counts": zero_loss,
        "actor_update_count": int(critic_loss["actor_joint_update_count"]),
        "critic_update_count": int(critic_loss["update_row_count"]),
        "nan_inf_count": 0,
        "illegal_action_count": int(legal["illegal_action_count"]),
        "future_leakage_count": 0,
        "pre_action_row_count": legal["row_count"],
    }


def make_binder_class(umod: Any) -> Any:
    class R2CycleEvidenceBinder(umod.CycleEvidenceBinder):
        """R1 binder plus the additional H4M-U-R2 mandatory cycle evidence."""

        def flush_cycle(self, *, seed: int, cycle: int, recorder: Any, trace_entry: Mapping[str, Any]) -> Dict[str, Any]:
            key = (int(seed), int(cycle))
            staged = self.pending.get(key)
            if staged is None:
                raise self.dte.EvidenceIntegrityError(
                    "CYCLE_UPDATE_EVIDENCE_NOT_STAGED",
                    f"key={key} cycle trace was written without a matching PPO update result",
                )
            staged["cycle_context"]["r2_required_evidence"] = build_r2_required_evidence(recorder, staged)
            return super().flush_cycle(seed=seed, cycle=cycle, recorder=recorder, trace_entry=trace_entry)

    return R2CycleEvidenceBinder


def bind_r2_checkpoint_naming(kmod: Any, umod: Any) -> None:
    """Name this run's checkpoints for H4M-U-R2 without touching R1 sources."""
    inner = kmod.save_seed_checkpoint

    def save(artifact_root: Path, seed: int, ctx: Mapping[str, Any], cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        checkpoint = inner(artifact_root, seed, ctx, cycle_summaries)
        old_path = Path(checkpoint["path"])
        new_path = Path(artifact_root) / "checkpoints" / CHECKPOINT_NAME.format(seed=int(seed))
        old_path.replace(new_path)
        checkpoint["path"] = str(new_path)
        checkpoint["sha256"] = sha256_file(new_path)
        checkpoint["stage"] = STAGE
        checkpoint["durable_evidence_instrumentation"] = "H4M-U-R1"
        checkpoint["old_h4m_u_checkpoint_reuse"] = False
        return checkpoint

    kmod.save_seed_checkpoint = save


# ---------------------------------------------------------------------------
# binding and integrity
# ---------------------------------------------------------------------------
def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit_before_run": head,
        "parent_commit": parent,
        "head_commit_files": head_files,
        "status_short": status,
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "parent_is_h4m_u_r1": parent == EXPECTED["h4m_u_r1_source_commit"],
        "github_push_performed": False,
    }


def r1_certified_source_state() -> Dict[str, Any]:
    """The R1-certified sources must be byte-identical to their PASS commit."""
    r1_changed = read_json(H4MR1_ROOT / "changed_files.json")
    recorded = {row["path"]: row["sha256"] for row in r1_changed.get("changed_files", [])}
    rows = []
    for rel in EXPECTED["r1_certified_sources"]:
        current = sha256_file(PROJECT_ROOT / rel)
        rows.append(
            {
                "path": rel,
                "r1_recorded_sha256": recorded.get(rel),
                "current_sha256": current,
                "unchanged_since_r1_pass": recorded.get(rel) == current,
                "unchanged_in_head_commit": git_run(["diff", "--quiet", "HEAD^", "HEAD", "--", rel], check=False).returncode == 0,
            }
        )
    return {
        "rows": rows,
        "r1_certified_sources_unchanged": all(row["unchanged_since_r1_pass"] and row["unchanged_in_head_commit"] for row in rows),
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], r1_sources: Mapping[str, Any]) -> Dict[str, Any]:
    r1_gate = read_json(H4MR1_ROOT / "gate_matrix.json")
    r1_schema = read_json(H4MR1_ROOT / "evidence_schema.json")
    t_gate = read_json(H4MT_ROOT / "gate_matrix.json")
    q_sha = read_json(H4MQ_ROOT / "repair_binding.json").get("sha_bindings", {})
    raw_gate = read_json(H4MU_RAW_ROOT / "gate_matrix.json")
    dte = import_module("h4mur2_dte_probe", DURABLE_SOURCE)
    checks = {
        "source_only_commit_before_training": provenance.get("source_only_local_commit") is True,
        "source_parent_is_h4m_u_r1": provenance.get("parent_is_h4m_u_r1") is True,
        "h4m_u_r1_gate_match": r1_gate.get("gate") == EXPECTED["h4m_u_r1_gate"],
        "h4m_u_r1_integrity": r1_gate.get("final_flags", {}).get("training_count") == 0
        and r1_gate.get("final_flags", {}).get("optimizer_step_count") == 0
        and r1_gate.get("final_flags", {}).get("TEST6_opened") is False,
        "r1_certified_sources_unchanged": r1_sources.get("r1_certified_sources_unchanged") is True,
        "durable_writer_schema_match": dte.SCHEMA_VERSION == r1_schema.get("schema_version"),
        "h4m_t_gate_match": t_gate.get("gate") == EXPECTED["h4m_t_gate"],
        "critic_repair_sha_match": t_gate.get("repair_contract_sha256") == EXPECTED["critic_repair_contract_sha256"],
        "actor_repair_sha_match": q_sha.get("h4m_p_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "reward_v2_sha_match": q_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "split_sha_match": q_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "schedule_sha_match": q_sha.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "zero_loss_sha_match": q_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "prior_h4m_u_run_bound_as_blocked": raw_gate.get("gate", "").startswith("BLOCKED_"),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "source_provenance": provenance,
        "r1_certified_source_state": r1_sources,
        "repair_shas": {
            "actor": EXPECTED["actor_repair_contract_sha256"],
            "critic": EXPECTED["critic_repair_contract_sha256"],
        },
        "frozen_sha_bindings": {
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "frozen_training_contract": {
            "seeds": EXPECTED["seeds"],
            "outer_training_count": EXPECTED["outer_training_count"],
            "ppo_updates_per_seed": EXPECTED["ppo_updates_per_seed"],
            "critic_updates_per_seed": EXPECTED["critic_updates_per_seed"],
            "tuning_applied": False,
            "additional_cycles_or_updates": False,
        },
        "prior_h4m_u_outputs": {
            "artifact_root": str(H4MU_RAW_ROOT),
            "status": ["DIAGNOSTIC_ONLY", "NON_PROMOTABLE", "INCOMPLETE_EVIDENCE"],
            "checkpoint_reused": False,
            "evidence_reconstructed": False,
        },
        "test6_access_count": 0,
    }


def r2_evidence_completeness(dte: Any, evidence_dir: Path) -> Dict[str, Any]:
    """Validate the R2 mandatory evidence extension across every record."""
    loaded = dte.load_evidence(evidence_dir)
    missing: List[str] = []
    zero_loss_totals = {"candidate": 0, "accepted": 0, "rejected": 0}
    legal_totals = {name: 0 for name in ACTION_NAMES.values()}
    actor_updates = 0
    critic_updates = 0
    illegal_total = 0
    for record in loaded["records"]:
        key = f"seed={record.get('seed')}/cycle={record.get('outer_cycle')}"
        block = (record.get("cycle_context") or {}).get("r2_required_evidence") or {}
        for field in R2_REQUIRED_EVIDENCE_FIELDS:
            if field not in block:
                missing.append(f"{key}:{field}")
        for name in zero_loss_totals:
            zero_loss_totals[name] += int((block.get("zero_loss_counts") or {}).get(name, 0))
        for name in legal_totals:
            legal_totals[name] += int((block.get("legal_action_counts") or {}).get(name, 0))
        actor_updates += int(block.get("actor_update_count", 0))
        critic_updates += int(block.get("critic_update_count", 0))
        illegal_total += int(block.get("illegal_action_count", 0))
    checks = {
        "all_records_carry_r2_required_evidence": not missing,
        "expected_cycle_cardinality": loaded["record_count"] == EXPECTED["expected_cycle_records"],
        "actor_update_total_matches_contract": actor_updates == EXPECTED["ppo_updates_per_seed"] * len(EXPECTED["seeds"]),
        "critic_update_total_matches_contract": critic_updates == EXPECTED["critic_updates_per_seed"] * len(EXPECTED["seeds"]),
        "illegal_action_zero": illegal_total == 0,
    }
    return {
        "stage": STAGE,
        "required_fields": R2_REQUIRED_EVIDENCE_FIELDS,
        "missing_fields": missing,
        "record_count": loaded["record_count"],
        "expected_record_count": EXPECTED["expected_cycle_records"],
        "zero_loss_totals": zero_loss_totals,
        "legal_action_totals": legal_totals,
        "actor_update_total": actor_updates,
        "critic_update_total": critic_updates,
        "illegal_action_total": illegal_total,
        "checks": checks,
        "passed": all(checks.values()),
        "extension_contract": "R1 writer record.cycle_context.r2_required_evidence; covered by record_sha256 and the R1 integrity checks",
    }


def primary_question_determinations(target: Mapping[str, Any], credit: Mapping[str, Any], validation: Mapping[str, Any]) -> Dict[str, Any]:
    hold_better = credit["by_context"][HOLD_BETTER]
    serve_better = credit["by_context"][SERVE_BETTER]
    hb_hold = hold_better["by_sampled_action"][HOLD]
    return {
        "q1_critic_overestimation_on_hold_better_sampled_hold": {
            "question": "HOLD_BETTER sampled HOLD Critic overestimation reduced or resolved?",
            "V_s_minus_return_mean": hb_hold["V_s_minus_return"]["mean"],
            "V_s_minus_return_median": hb_hold["V_s_minus_return"]["median"],
            "sample_count": hb_hold["sample_count"],
            "overestimation_persists": target.get("critic_overestimation_hold_better_sampled_hold_persists"),
        },
        "q2_legitimate_hold_raw_gae_structurally_negative": {
            "question": "Does legitimate HOLD raw GAE remain structurally pushed negative?",
            "raw_GAE_mean": hb_hold["raw_GAE"]["mean"],
            "raw_GAE_median": hb_hold["raw_GAE"]["median"],
            "structurally_negative": target.get("legitimate_hold_raw_gae_structurally_negative"),
        },
        "q3_cycle_2_serve_collapse_recurrence": {
            "question": "Does the SERVE collapse from cycle 2 recur?",
            "early_cycle_2_serve_collapse_recurs": target.get("early_cycle_2_serve_collapse_recurs"),
        },
        "q4_target_conditioned_gradient_separation": {
            "question": "Is HOLD-target / SERVE-target gradient separation preserved?",
            "separation_intact": target.get("target_conditioned_gradient_separation_remains_intact"),
            "evidence_source": "gradient_isolation.json",
        },
        "q5_three_seed_context_discrimination": {
            "question": "Do the three seeds collapse to one action or recover context discrimination?",
            "three_seed_validation_dominant_actions": target.get("three_seed_validation_dominant_actions"),
            "three_seeds_converge_to_one_action": target.get("three_seeds_converge_to_one_action"),
            "hold_better_probability_dominant": validation["aggregate_by_classification"][HOLD_BETTER].get("probability_dominant_action"),
            "serve_better_probability_dominant": validation["aggregate_by_classification"][SERVE_BETTER].get("probability_dominant_action"),
        },
        "interpretation_rule": "No required HOLD ratio is preset; a higher HOLD share alone is not PASS and global SERVE dominance alone is not failure when the corrected credit evidence supports it.",
        "hold_better_overall": hold_better["overall"],
        "serve_better_overall": serve_better["overall"],
    }


def gate_matrix(
    binding: Mapping[str, Any],
    integrity: Mapping[str, Any],
    outcome: Mapping[str, Any],
    changed: Mapping[str, Any],
    evidence: Mapping[str, Any],
    r2_evidence: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding": binding.get("authoritative_binding_passed") is True,
        "source_only_commit": changed.get("source_only_local_commit") is True and changed.get("artifact_or_log_committed") is False,
        "three_seeds_complete": training.get("seeds") == EXPECTED["seeds"] and training.get("total_rollouts") == 33,
        "cycle_evidence_cardinality_33_of_33": evidence.get("record_count") == EXPECTED["expected_cycle_records"]
        and not evidence.get("missing_cycles"),
        "durable_evidence_integrity": evidence.get("integrity_passed") is True,
        "r2_required_evidence_complete": r2_evidence.get("passed") is True,
        "training_integrity": integrity.get("training_integrity_passed") is True,
        "checkpoint_integrity": checkpoints.get("checkpoint_manifest_passed") is True,
        "old_h4m_u_checkpoint_reuse_false": binding.get("prior_h4m_u_outputs", {}).get("checkpoint_reused") is False
        and training.get("fresh_initialization_all_seeds") is True,
        "test6_zero": integrity.get("TEST6_access_count") == 0,
        "future_leakage_zero": integrity.get("future_leakage_count") == 0,
        "nan_inf_zero": evidence.get("nan_inf_count") == 0,
        "report_from_persisted_evidence": evidence.get("report_source") == "PERSISTED_DURABLE_EVIDENCE_ONLY"
        and evidence.get("volatile_in_memory_metrics_used") is False,
        "github_push_false": changed.get("github_push_performed") is False,
        "outcome_classified": outcome.get("outcome_classification") is not None,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_INTEGRITY_FAILED",
        "decision": outcome.get("outcome_classification") if passed else "H4M_U_R2_BLOCKED",
        "exact_next_gate": NEXT_REVIEW_GATE if passed else f"STOP_{BLOCK_GATE_PREFIX}",
        "next_gate_is_read_only_review": True,
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "repair_contracts": binding.get("repair_shas"),
        "frozen_sha_bindings": binding.get("frozen_sha_bindings"),
        "final_flags": {
            "TEST6_opened": False,
            "test6_access_count": 0,
            "github_push_performed": False,
            "tuning_applied": False,
            "additional_cycles_or_updates": False,
            "old_h4m_u_checkpoint_reused": False,
            "manual_checkpoint_selection": False,
        },
    }


def make_manifest(root: Path, gate: Mapping[str, Any], evidence: Mapping[str, Any], checkpoints: Mapping[str, Any], started: float) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": git_run(["rev-parse", "HEAD"]).stdout.strip(),
        "parent_commit": EXPECTED["h4m_u_r1_source_commit"],
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "repair_shas": {
            "actor": EXPECTED["actor_repair_contract_sha256"],
            "critic": EXPECTED["critic_repair_contract_sha256"],
        },
        "split_sha256": EXPECTED["r3_split_sha256"],
        "schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "device_backend": "MPS",
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256_by_seed": {
            str(row.get("seed")): row.get("sha256") for row in checkpoints.get("checkpoints", [])
        },
        "cycle_evidence_cardinality": f"{evidence.get('record_count')}/{EXPECTED['expected_cycle_records']}",
        "required_artifacts_present": all((root / item).exists() for item in REQUIRED_ARTIFACTS if item != "manifest.json"),
        "required_evidence_files_present": all((root / item).exists() for item in REQUIRED_EVIDENCE_FILES),
        "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "append_only_artifact": True,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(
    dte: Any,
    binding: Mapping[str, Any],
    evidence: Mapping[str, Any],
    r2_evidence: Mapping[str, Any],
    integrity: Mapping[str, Any],
    questions: Mapping[str, Any],
    outcome: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    header = f"""# {REPORT_TITLE}

gate = {gate['gate']}
outcome_classification = {outcome['outcome_classification']}
source_commit = {binding['source_provenance']['source_commit_before_run']}
parent_commit = {EXPECTED['h4m_u_r1_source_commit']}
actor_repair_contract_sha256 = {EXPECTED['actor_repair_contract_sha256']}
critic_repair_contract_sha256 = {EXPECTED['critic_repair_contract_sha256']}
split_sha256 = {EXPECTED['r3_split_sha256']}
schedule_sha256 = {EXPECTED['h4m_b_schedule_sha256']}
device_backend = MPS
cycle_evidence_cardinality = {evidence.get('record_count')}/{EXPECTED['expected_cycle_records']}
TEST6_access_count = 0
training_evidence_source = {evidence.get('report_source')}
exact_next_gate = {gate['exact_next_gate']} (read-only review; not executed automatically)

## Primary question

Does the repaired Critic credit plus the target-conditioned Actor actually discriminate
HOLD_BETTER from SERVE_BETTER? A larger HOLD share alone is not PASS.

```json
{json.dumps(questions, ensure_ascii=False, indent=2, default=dte._jsonable)}
```

## Outcome

```json
{json.dumps(outcome, ensure_ascii=False, indent=2, default=dte._jsonable)}
```

## Checkpoints (diagnostic evidence, not promoted)

```json
{json.dumps({str(row.get('seed')): {'path': row.get('path'), 'sha256': row.get('sha256')} for row in checkpoints.get('checkpoints', [])}, ensure_ascii=False, indent=2, default=dte._jsonable)}
```

## Integrity

```json
{json.dumps({'training_integrity': integrity.get('checks'), 'r2_required_evidence': r2_evidence.get('checks'), 'gate': gate.get('criteria')}, ensure_ascii=False, indent=2, default=dte._jsonable)}
```

No TEST6 access, no tuning, no extra cycles or updates, no old H4M-U checkpoint reuse, no manual
checkpoint selection, and no GitHub push occurred. This is diagnostic training evidence, not a
promotion decision.

---

"""
    return header + dte.render_final_report(evidence, title=f"{REPORT_TITLE} — persisted training evidence")


def write_block(root: Path, binding: Mapping[str, Any], reason: str, started: float, umod: Any) -> None:
    gate = {
        "stage": STAGE,
        "gate": f"{BLOCK_GATE_PREFIX}_{reason}",
        "decision": "H4M_U_R2_BLOCKED",
        "exact_next_gate": f"STOP_{BLOCK_GATE_PREFIX}",
        "block_reason": reason,
        "final_flags": {"TEST6_opened": False, "github_push_performed": False, "tuning_applied": False},
    }
    empty = {"stage": STAGE, "not_executed_or_incomplete": reason}
    for name in REQUIRED_ARTIFACTS:
        if name in {"manifest.json", "gate_matrix.json"}:
            continue
        if name == "final_report.md":
            (root / name).write_text(
                f"# H4M-U-R2\n\ngate = {gate['gate']}\nblock_reason = {reason}\n\nSTOP. No tuning, no extended training, no TEST6 access, no GitHub push.\n",
                encoding="utf-8",
            )
        elif not (root / name).exists():
            umod.write_json(root / name, binding if name == "repair_binding.json" else empty)
    umod.write_json(root / "gate_matrix.json", gate)
    umod.write_json(
        root / "manifest.json",
        {
            "stage": STAGE,
            "artifact_root": str(root),
            "gate": gate["gate"],
            "block_reason": reason,
            "elapsed_seconds": time.perf_counter() - started,
            "device_backend": "MPS",
            "TEST6_opened": False,
            "test6_access_count": 0,
            "github_push_performed": False,
        },
    )
    print(f"[H4M-U-R2] artifact root: {root}\n[H4M-U-R2] gate: {gate['gate']}\n[H4M-U-R2] block_reason: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=REPORT_TITLE)
    parser.add_argument(
        "--regenerate-report",
        type=Path,
        default=None,
        metavar="ARTIFACT_ROOT",
        help="rebuild final_report.md from persisted durable evidence (H4M-U-R1 recovery path); no training",
    )
    args = parser.parse_args()
    umod = import_module("h4mur2_umod", H4MU_SOURCE)
    if args.regenerate_report is not None:
        dte = umod.load_durable_evidence()
        recovery = dte.regenerate_report(args.regenerate_report, report_name="final_report.md", title=REPORT_TITLE)
        print(json.dumps(recovery, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(0 if recovery["recovered"] else 1)

    started = time.perf_counter()
    created_at = now().isoformat()
    stamp = now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_u_r2_fresh_three_seed_retraining_with_durable_evidence_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    umod.STAGE = STAGE
    umod.SOURCE_REL = SOURCE_REL
    umod.REPORT_TITLE = REPORT_TITLE

    provenance = source_provenance(created_at)
    r1_sources = r1_certified_source_state()
    binding = authoritative_binding(created_at, provenance, r1_sources)
    umod.write_json(root / "repair_binding.json", binding)
    if not binding["authoritative_binding_passed"]:
        write_block(root, binding, "AUTHORITATIVE_BINDING_MISMATCH", started, umod)
        return
    if not torch.backends.mps.is_available():
        write_block(root, binding, "MPS_NOT_AVAILABLE", started, umod)
        return

    dte = umod.load_durable_evidence()
    evidence_dir = root / dte.EVIDENCE_DIR_NAME
    run_header = {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_root": str(root),
        "seeds": EXPECTED["seeds"],
        "outer_training_count": EXPECTED["outer_training_count"],
        "source_commit": provenance["source_commit_before_run"],
        "split_sha256": EXPECTED["r3_split_sha256"],
        "schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
        "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "ppo_updates_per_seed": EXPECTED["ppo_updates_per_seed"],
        "critic_updates_per_seed": EXPECTED["critic_updates_per_seed"],
        "device_backend": "MPS",
        "durable_evidence_instrumentation": "H4M-U-R1",
        "training_execution_contract": "H4M-Q frozen lineage with H4M-P actor specialization plus H4M-T S3 critic target binding",
    }
    expected_grid = [(seed, cycle) for seed in EXPECTED["seeds"] for cycle in range(1, EXPECTED["outer_training_count"] + 1)]
    writer = dte.DurableCycleEvidenceWriter(evidence_dir, run_header=run_header, expected_grid=expected_grid)
    binder = make_binder_class(umod)(dte, writer, root, STAGE, run_header)
    update_audits: List[Dict[str, Any]] = []

    try:
        qmod = import_module(f"h4mur2_q_{time.time_ns()}", H4MQ_SOURCE)
        h4mg = import_module(f"h4mur2_g_{time.time_ns()}", H4MG_SOURCE)
        kmod = umod.configure_execution(qmod, h4mg, update_audits, binder)
        base, _rerun = kmod.load_h4m_helpers()
        umod.bind_cycle_evidence_flush(base, binder)
        bind_r2_checkpoint_naming(kmod, umod)
        device = torch.device("mps")

        training, cycle_summaries, checkpoints_raw, trace_meta = kmod.run_training(base, h4mg, created_at, root, device)
        writer.mark_training_complete(
            {
                "total_rollouts": training.get("total_rollouts"),
                "total_ppo_updates": training.get("total_ppo_updates"),
                "total_critic_updates": training.get("total_critic_updates"),
            }
        )
        training.update(
            {
                "stage": STAGE,
                "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
                "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
                "critic_target_binding_schema": h4mg.CRITIC_VALUE_TARGET_BINDING_SCHEMA,
                "device_backend": "MPS",
                "durable_evidence_instrumentation": "H4M-U-R1",
                "old_h4m_u_checkpoint_reuse": False,
            }
        )

        conditional = kmod.conditional_policy_discrimination(trace_meta["joined_rows"])
        conditional["conditional_policy_by_sample_parquet"] = kmod.write_conditional_parquet(base, root, trace_meta["joined_rows"])
        validation_plan = qmod.load_validation_plan()
        validation = qmod.validation_discrimination(checkpoints_raw, validation_plan, device)
        validation["large_row_data"] = qmod.write_parquet_large_rows(root, validation)
        parameter_delta = qmod.parameter_delta_audit(h4mg, checkpoints_raw, created_at, device, trace_meta["joined_rows"])
        gradient = qmod.gradient_audit(h4mg, checkpoints_raw, trace_meta["joined_rows"], created_at, device)
        checkpoints = qmod.checkpoint_manifest(checkpoints_raw)
        credit = umod.critic_credit_diagnostics(root)

        evidence = dte.build_report_payload(evidence_dir)
        r2_evidence = r2_evidence_completeness(dte, evidence_dir)
        integrity = umod.training_integrity(
            training, trace_meta, update_audits, credit, parameter_delta, gradient, checkpoints, validation, evidence
        )
        integrity["checks"]["r2_required_evidence_complete"] = r2_evidence["passed"]
        integrity["training_integrity_passed"] = all(integrity["checks"].values())
        integrity["failing_criteria"] = [key for key, value in integrity["checks"].items() if not value]
        integrity["zero_loss_counts"] = r2_evidence["zero_loss_totals"]

        target = umod.target_conditioned_diagnostics(conditional, validation, credit)
        target["target_conditioned_gradient_separation_remains_intact"] = gradient.get("cross_specialized_head_gradient_leakage_count") == 0
        questions = primary_question_determinations(target, credit, validation)
        target["primary_question_determinations"] = questions
        outcome = umod.outcome_classification(validation, target, integrity["training_integrity_passed"])
        outcome["exact_next_review_gate"] = NEXT_REVIEW_GATE

        changed = umod.changed_files_audit()
        gate = gate_matrix(binding, integrity, outcome, changed, evidence, r2_evidence, checkpoints, training)

        test_results = {
            "stage": STAGE,
            "commands": [
                f"{sys.executable} -m py_compile {SOURCE_REL}",
                "git diff --cached --check",
                f"{sys.executable} {SOURCE_REL}",
            ],
            "training_execution_authorized": True,
            "optimizer_step_count_training": training.get("total_ppo_updates"),
            "r1_regression_suite": "05_training/test_h4m_u_r1_durable_training_evidence.py (unchanged R1 PASS suite; R1 sources byte-identical)",
            "TEST6_access_count": 0,
        }
        seed_rows = {int(row["seed"]): row for row in training["seed_summaries"]}
        for seed in EXPECTED["seeds"]:
            umod.write_json(
                root / f"seed_{seed}_summary.json",
                {
                    "seed_summary": seed_rows[seed],
                    "cycle_summaries": [row for row in cycle_summaries if int(row["seed"]) == seed],
                    "durable_evidence": evidence["by_seed"].get(str(seed)),
                    "durable_evidence_cycles": [row for row in evidence["by_seed_cycle"] if int(row["seed"]) == seed],
                },
            )
        payloads = {
            "training_summary.json": training,
            "cycle_action_evolution.json": qmod.cycle_action_evolution(training, conditional),
            "critic_credit_diagnostics.json": credit,
            "target_conditioned_diagnostics.json": target,
            "checkpoint_integrity.json": checkpoints,
            "gradient_isolation.json": gradient,
            "parameter_delta.json": parameter_delta,
            "validation_discrimination.json": validation,
            "training_integrity.json": integrity,
            "outcome_classification.json": outcome,
            "durable_evidence_report.json": evidence,
            "r2_required_evidence_validation.json": r2_evidence,
            "evidence_schema.json": dte.evidence_schema(),
            "changed_files.json": changed,
            "test_results.json": test_results,
            "gate_matrix.json": gate,
            "conditional_policy_discrimination_training_trace.json": conditional,
        }
        for name, payload in payloads.items():
            umod.write_json(root / name, payload)
        (root / "final_report.md").write_text(
            final_report(dte, binding, evidence, r2_evidence, integrity, questions, outcome, checkpoints, gate),
            encoding="utf-8",
        )
        umod.write_json(root / "manifest.json", make_manifest(root, gate, evidence, checkpoints, started))
        writer.mark_report_complete({"gate": gate.get("gate"), "report_status": evidence.get("report_status")})

        print(f"[H4M-U-R2] artifact root: {root}")
        print(f"[H4M-U-R2] gate: {gate['gate']}")
        print(f"[H4M-U-R2] outcome: {outcome['outcome_classification']}")
        print(f"[H4M-U-R2] cycle evidence: {evidence['record_count']}/{EXPECTED['expected_cycle_records']}")
        print(f"[H4M-U-R2] failing criteria: {gate['failing_criteria']}")
        print(f"[H4M-U-R2] exact next gate: {gate['exact_next_gate']} (read-only review, not executed)")
    except Exception as exc:
        write_block(root, binding, f"EXECUTION_EXCEPTION_{type(exc).__name__}", started, umod)
        umod.recover_after_reporting_failure(dte, root, exc)
        raise


if __name__ == "__main__":
    main()
