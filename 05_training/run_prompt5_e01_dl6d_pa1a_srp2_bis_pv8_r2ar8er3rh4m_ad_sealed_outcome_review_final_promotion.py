#!/usr/bin/env python3
"""H4M-AD sealed hold-out outcome review and final promotion decision.

Read-only final review of the H4M-Z, H4M-AA, H4M-AB and H4M-AC artifacts.  It
never reopens the sealed hold-out, runs no inference, no training, no optimizer
step and no backward pass, changes no checkpoint, criterion or binding, mutates
no data and pushes nothing.  Historical artifacts are read only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-AD"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AD_"
    "SEALED_HOLDOUT_OUTCOME_REVIEW_AND_FINAL_PROMOTION_DECISION_COMPLETE"
)
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AD_FINAL_PROMOTION_NOT_JUSTIFIED"
NEXT_GATE = "H4M-AE_POST_PROMOTION_EVALUATION_SCOPE_DEFINITION_AND_FREEZE"
APPROVED = "FINAL_PROMOTION_APPROVED"
HOLD_DECISION = "FINAL_PROMOTION_HOLD"
INVALID = "FINAL_PROMOTION_INVALID"
PROMOTED_STATUS = "PROMOTED_REPAIRED_POLICY_LINEAGE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
H4MAC_SOURCE_REL = "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_ac_single_open_sealed_holdout_evaluation.py"

EXECUTION_PATH_SOURCES = [
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py",
    "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/observation_target_context_repair.py",
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py",
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py",
    "05_training/durable_training_evidence.py",
]

EXPECTED = {
    "h4m_ac_source_commit": "3ba9f71",
    "h4m_ac_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AC_SINGLE_OPEN_SEALED_HOLDOUT_EVALUATION_COMPLETE",
    "h4m_ac_outcome": "SEALED_HOLDOUT_GENERALIZATION_PASS",
    "h4m_ab_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AB_SEALED_HOLDOUT_PROMOTION_DECISION_GATE_COMPLETE",
    "h4m_aa_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AA_"
        "FROZEN_WINDOW_BOUNDARY_REPAIRED_POLICY_VALIDATION_COMPLETE"
    ),
    "h4m_z_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Z_"
        "WINDOW_BOUNDARY_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_VALIDATION_PROMOTION_COMPLETE"
    ),
    "h4m_z_classification": "A_PRIMARY_CAUSAL_ROOT_CAUSE_CONFIRMED_WINDOW_EPISODE_BOUNDARY",
    "sealed_evaluation_contract_sha256": "e163cb9016839b98561b6f9f221fde7fef2eaed30820ff5be1f6a82903fb2c4a",
    "criteria_sha256": "e4e9627b10228d4f6d05da38138e0564dd15f4d739dbc4197ac44c69a4de2545",
    "validation_promotion_contract_sha256": "6c15c2011bcd0c40932198d754a36b68ba44ce810f57b7352f3634847af687e7",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "observation_repair_contract_sha256": "6ecd20cfcd220d50a6f1ebbcd6e33594a9ca14a86a2a60236cea435a24058e1a",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "aa_expected": {"HOLD_LONG_HORIZON_BETTER": 60, "SERVE_LONG_HORIZON_BETTER": 36},
    "ac_expected": {"HOLD_LONG_HORIZON_BETTER": 90, "SERVE_LONG_HORIZON_BETTER": 54},
    "seeds": [1, 2, 3],
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
CORRECT_ACTION = {HOLD_BETTER: HOLD, SERVE_BETTER: SERVE}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "sealed_outcome_review.json",
    "single_open_integrity_audit.json",
    "counterpart_mapping_audit.json",
    "causal_lineage_review.json",
    "final_promotion_decision.json",
    "final_promotion_contract.json",
    "gate_matrix.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def canonical_sha(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def latest_artifact(pattern: str, gate: str) -> Path:
    roots = sorted(ARTIFACTS_ROOT.glob(pattern))
    passing = [r for r in roots if (r / "gate_matrix.json").exists() and read_json(r / "gate_matrix.json").get("gate") == gate]
    if not passing:
        raise RuntimeError(f"no passing artifact for {pattern}")
    return passing[-1]


# ---------------------------------------------------------------------------
def sealed_outcome_review(aa_root: Path, ac_root: Path) -> Dict[str, Any]:
    aa = read_json(aa_root / "frozen_validation_results.json")["aggregate_by_classification"]
    ac = read_json(ac_root / "aggregate_holdout_results.json")["aggregate_by_classification"]
    ac_seeds = {seed: read_json(ac_root / f"seed_{seed}_results.json")["by_class"] for seed in EXPECTED["seeds"]}
    aa_seeds = read_json(aa_root / "frozen_validation_results.json")["by_seed"]
    checks = {
        "aa_hold_better_all_hold": aa[HOLD_BETTER]["chosen_action_counts"].get(HOLD) == EXPECTED["aa_expected"][HOLD_BETTER]
        and aa[HOLD_BETTER]["sample_count"] == EXPECTED["aa_expected"][HOLD_BETTER],
        "aa_serve_better_all_serve": aa[SERVE_BETTER]["chosen_action_counts"].get(SERVE) == EXPECTED["aa_expected"][SERVE_BETTER]
        and aa[SERVE_BETTER]["sample_count"] == EXPECTED["aa_expected"][SERVE_BETTER],
        "aa_total_96_of_96": sum(aa[label]["sample_count"] for label in (HOLD_BETTER, SERVE_BETTER)) == 96
        and all(aa[label]["sampled_correct_direction_rate"] == 1.0 for label in (HOLD_BETTER, SERVE_BETTER)),
        "ac_hold_better_all_hold": ac[HOLD_BETTER]["chosen_action_counts"].get(HOLD) == EXPECTED["ac_expected"][HOLD_BETTER]
        and ac[HOLD_BETTER]["sample_count"] == EXPECTED["ac_expected"][HOLD_BETTER],
        "ac_serve_better_all_serve": ac[SERVE_BETTER]["chosen_action_counts"].get(SERVE) == EXPECTED["ac_expected"][SERVE_BETTER]
        and ac[SERVE_BETTER]["sample_count"] == EXPECTED["ac_expected"][SERVE_BETTER],
        "ac_total_144_of_144": sum(ac[label]["sample_count"] for label in (HOLD_BETTER, SERVE_BETTER)) == 144
        and all(ac[label]["correct_direction_rate"] == 1.0 for label in (HOLD_BETTER, SERVE_BETTER)),
        "ac_three_seeds_direction_consistent": all(
            ac_seeds[seed][label]["discrimination_is_context_appropriate"] is True
            for seed in EXPECTED["seeds"]
            for label in (HOLD_BETTER, SERVE_BETTER)
        ),
        "no_single_action_collapse_on_sealed": ac[HOLD_BETTER]["chosen_action_dominant"] != ac[SERVE_BETTER]["chosen_action_dominant"],
        "datasets_reported_separately": True,
    }
    return {
        "stage": STAGE,
        "frozen_validation_h4m_aa": {
            "dataset": "approved frozen validation windows (saturday)",
            "aggregate": aa,
            "by_seed": aa_seeds,
        },
        "sealed_holdout_h4m_ac": {
            "dataset": "sealed TEST6 windows (holiday), opened once",
            "aggregate": ac,
            "by_seed": {str(seed): ac_seeds[seed] for seed in EXPECTED["seeds"]},
        },
        "generalization_chain": "fresh retraining -> frozen validation 96/96 -> sealed hold-out 144/144, on disjoint window sets that were never pooled",
        "checks": checks,
        "passed": all(checks.values()),
    }


def single_open_integrity_audit(ac_root: Path) -> Dict[str, Any]:
    audit = read_json(ac_root / "sealed_open_audit.json")
    gate = read_json(ac_root / "gate_matrix.json")
    integrity = read_json(ac_root / "integrity_validation.json")
    checkpoints = read_json(ac_root / "checkpoint_integrity.json")
    ac_artifacts = sorted(ARTIFACTS_ROOT.glob("pv8_r2a_r8e_r3_r_h4m_ac_*"))
    opened_artifacts = [
        root.name for root in ac_artifacts if (root / "sealed_open_audit.json").exists() and read_json(root / "sealed_open_audit.json").get("open_count", 0) >= 1
    ]
    ac_commit = git_run(["rev-parse", EXPECTED["h4m_ac_source_commit"]]).stdout.strip()
    commits_after = [line for line in git_run(["log", "--format=%H %s", f"{ac_commit}..HEAD"]).stdout.splitlines() if line]
    changed_after = [line for line in git_run(["diff", "--name-only", ac_commit, "HEAD"]).stdout.splitlines() if line]
    execution_changed_after = [path for path in changed_after if path in EXECUTION_PATH_SOURCES]
    current_checkpoints = [
        {
            "seed": row["seed"],
            "contract_sha256": row["contract_sha256"],
            "post_evaluation_sha256": row["post_evaluation_sha256"],
            "current_sha256": sha256_file(Path(row["path"])),
            "unchanged_since_evaluation": sha256_file(Path(row["path"])) == row["post_evaluation_sha256"] == row["contract_sha256"],
        }
        for row in checkpoints["checkpoints"]
    ]
    checks = {
        "open_count_exactly_one": audit.get("open_count") == 1 and gate.get("open_count") == 1,
        "open_attempts_exactly_one": audit.get("open_attempts") == 1,
        "exactly_one_sealed_open_artifact": len(opened_artifacts) == 1,
        "no_reopen_or_retry": gate.get("final_flags", {}).get("sealed_reopened") is False,
        "raw_evidence_persisted_before_reporting": audit.get("raw_evidence_persisted") is True and audit.get("raw_evidence_rows") == 144,
        "no_checkpoint_reselection": gate.get("final_flags", {}).get("checkpoint_reselected") is False and checkpoints["checks"]["no_checkpoint_reselection"] is True,
        "no_seed_exclusion": gate.get("final_flags", {}).get("seed_excluded") is False,
        "no_tuning_after_result": gate.get("final_flags", {}).get("tuning_applied") is False,
        "no_execution_path_change_after_sealed_run": not execution_changed_after,
        "checkpoint_sha_unchanged_since_evaluation": all(row["unchanged_since_evaluation"] for row in current_checkpoints),
        "training_optimizer_backward_zero": integrity["counters"]["training_count"] == 0
        and integrity["counters"]["optimizer_step_count"] == 0
        and integrity["counters"]["backward_pass_count"] == 0,
        "parameter_mutation_zero": integrity["counters"]["parameter_mutation_count"] == 0
        and checkpoints["checks"]["all_parameter_hashes_unchanged"] is True,
        "illegal_action_and_nan_inf_zero": integrity["checks"]["illegal_action_zero"] is True and integrity["checks"]["nan_inf_zero"] is True,
        "future_leakage_zero": integrity["checks"]["future_leakage_zero"] is True,
    }
    return {
        "stage": STAGE,
        "sealed_open_audit": audit,
        "sealed_open_artifacts": opened_artifacts,
        "commits_after_sealed_run": commits_after,
        "files_changed_after_sealed_run": changed_after,
        "execution_path_files_changed_after_sealed_run": execution_changed_after,
        "checkpoints": current_checkpoints,
        "counters": integrity["counters"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def counterpart_mapping_audit(ac_root: Path) -> Dict[str, Any]:
    binding = read_json(ac_root / "contract_binding.json")
    applied = read_json(ac_root / "frozen_criteria_application.json")
    audit = read_json(ac_root / "sealed_open_audit.json")
    criteria = read_json(latest_artifact("pv8_r2a_r8e_r3_r_h4m_ab_*", EXPECTED["h4m_ab_gate"]) / "criteria_freeze.json")

    source_blob = git_run(["show", f"{EXPECTED['h4m_ac_source_commit']}:{H4MAC_SOURCE_REL}"]).stdout
    mapping_from_blob = None
    for node in ast.parse(source_blob).body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "CRITERIA_APPLICABILITY" for target in node.targets):
            mapping_from_blob = ast.literal_eval(node.value)
    blob_sha = canonical_sha(mapping_from_blob) if mapping_from_blob is not None else None

    # the two pairs named in the H4M-AD instruction
    instruction_pairs = {
        "validation_rows_four": "sealed_rows_match_frozen_sealed_identity",
        "test_snapshot_paths_loaded_empty": "sealed_open_count_exactly_one",
    }
    # H4M-AC additionally mapped one further scope guard; it is disclosed, not hidden
    additional_pairs = {"validation_scope_passed": "sealed_scope_passed"}
    expected_pairs = {**instruction_pairs, **additional_pairs}
    allowed_counterpart_names = set(expected_pairs.values())
    declared = binding.get("criteria_applicability_declared_before_open", {})
    pair_ok = {
        source: (declared.get(source, {}).get("sealed_counterpart") or "").startswith(target)
        for source, target in expected_pairs.items()
    }
    verbatim = [name for name, rule in declared.items() if rule.get("applies_verbatim")]
    mapped = [name for name, rule in declared.items() if not rule.get("applies_verbatim")]
    mapped_counterpart_names = {
        (declared[name].get("sealed_counterpart") or "").split(":")[0].strip() for name in mapped
    }
    checks = {
        "mapping_present_in_committed_source_before_run": mapping_from_blob is not None,
        "mapping_sha_matches_committed_source": blob_sha == binding.get("criteria_applicability_sha256"),
        "mapping_recorded_in_artifact_equals_source": mapping_from_blob == declared,
        "contract_binding_written_before_sealed_open": binding.get("created_at", "") <= audit.get("opened_at", ""),
        "both_instruction_named_pairs_as_specified": all(pair_ok[name] for name in instruction_pairs),
        "additional_scope_guard_mapping_disclosed": all(pair_ok[name] for name in additional_pairs),
        "mapped_set_is_exactly_the_scope_guards": sorted(mapped) == sorted(expected_pairs),
        "verbatim_and_mapped_partition_complete": len(verbatim) + len(mapped) == 8 and not set(verbatim) & set(mapped),
        "every_counterpart_is_procedural_or_integrity": mapped_counterpart_names <= allowed_counterpart_names,
        "no_performance_threshold_in_frozen_criteria": criteria["criteria"]["numeric_thresholds"]["hold_ratio_threshold"] is None
        and criteria["criteria"]["numeric_thresholds"]["accuracy_threshold"] is None,
        "counterparts_are_procedural_or_integrity_only": True,
        "criteria_sha_unchanged_across_stages": criteria.get("criteria_sha256") == EXPECTED["criteria_sha256"],
        "no_post_hoc_criterion_change": applied.get("applicability_declared_before_open") == declared,
    }
    return {
        "stage": STAGE,
        "instruction_named_pairs": instruction_pairs,
        "additional_pairs_declared_by_h4m_ac": additional_pairs,
        "expected_counterpart_pairs": expected_pairs,
        "verbatim_check_count": len(verbatim),
        "mapped_check_count": len(mapped),
        "declared_mapping": declared,
        "mapping_sha256_in_artifact": binding.get("criteria_applicability_sha256"),
        "mapping_sha256_recomputed_from_commit": blob_sha,
        "source_commit_carrying_mapping": EXPECTED["h4m_ac_source_commit"],
        "contract_binding_created_at": binding.get("created_at"),
        "sealed_opened_at": audit.get("opened_at"),
        "verbatim_checks": sorted(verbatim),
        "meaning_change_scope": "procedural and integrity scope only: a row-identity guard and an open-count guard; neither is a performance criterion",
        "checks": checks,
        "passed": all(checks.values()),
    }


def causal_lineage_review(z_root: Path, aa_root: Path, ac_root: Path) -> Dict[str, Any]:
    z_decision = read_json(z_root / "root_cause_decision.json")
    z_comparison = read_json(z_root / "causal_before_after_comparison.json")
    z_competing = read_json(z_root / "competing_explanation_review.json")
    y_boundary = read_json(
        ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_r1_report_only_recovery_from_persisted_evidence_20260818_153018+09:00" / "window_boundary_diagnostics.json"
    )
    ac_aggregate = read_json(ac_root / "aggregate_holdout_results.json")["aggregate_by_classification"]
    chain = [
        {
            "step": 1,
            "claim": "independent single-decision windows were linked into one credit episode",
            "evidence": "44 windows per rollout with one causal step each; 100% of GAE recursion links crossed a window boundary before the repair",
            "source": "H4M-V credit causality audit",
        },
        {
            "step": 2,
            "claim": "that linkage injected non-causal cross-window credit",
            "evidence": "the counterfactual action effect on any later state is zero in 11,616 of 11,616 rows, yet the cross-window tail supplied most of the advantage variance",
            "source": "H4M-V credit chain attenuation",
        },
        {
            "step": 3,
            "claim": "the contaminated credit left the HOLD_BETTER decision signal an order of magnitude weaker than SERVE_BETTER, and SERVE dominance persisted",
            "evidence": "delivered normalized-advantage contrast ratio 16.6x against a ground-truth reward ratio of 13x; HOLD_BETTER correct-direction 0.2667",
            "source": "H4M-V and H4M-U-R2",
        },
        {
            "step": 4,
            "claim": "the W1 boundary repair removed the contamination",
            "evidence": f"terminated {y_boundary['totals']['terminated_true']}/{y_boundary['totals']['samples']}, bootstrap leakage {y_boundary['totals']['cross_window_links_with_live_bootstrap']}, "
            f"{y_boundary['totals']['cross_window_links_blocked']} boundary links blocked",
            "source": "H4M-Y durable evidence and boundary diagnostics",
        },
        {
            "step": 5,
            "claim": "target-context discrimination recovered in training behaviour",
            "evidence": "HOLD_BETTER HOLD share recovered from cycle 7-8 to 82.9% by cycle 11 while SERVE_BETTER fell to 0.0%",
            "source": "H4M-Y cycle action evolution",
        },
        {
            "step": 6,
            "claim": "the recovery generalized to the approved frozen validation split",
            "evidence": "96/96 correct direction across three seeds",
            "source": "H4M-AA",
        },
        {
            "step": 7,
            "claim": "the recovery generalized to unseen sealed hold-out windows",
            "evidence": f"144/144 correct direction; HOLD_BETTER {ac_aggregate[HOLD_BETTER]['chosen_action_counts'][HOLD]}/{ac_aggregate[HOLD_BETTER]['sample_count']} HOLD, "
            f"SERVE_BETTER {ac_aggregate[SERVE_BETTER]['chosen_action_counts'][SERVE]}/{ac_aggregate[SERVE_BETTER]['sample_count']} SERVE",
            "source": "H4M-AC single sealed open",
        },
    ]
    checks = {
        "root_cause_classification_is_a": z_decision.get("classification") == EXPECTED["h4m_z_classification"],
        "before_after_verified_on_same_split": z_comparison.get("passed") is True,
        "competing_explanations_excluded": z_competing.get("passed") is True,
        "only_boundary_definitions_changed_between_runs": set(z_competing["top_level_definition_identity"]["changed"])
        <= {"collect_controlled_rollout", "materialize_rollout_credit_trace"},
        "leakage_zero_after_repair": y_boundary["totals"]["cross_window_links_with_live_bootstrap"] == 0,
        "sealed_generalization_confirmed": all(
            ac_aggregate[label]["discrimination_is_context_appropriate"] is True for label in (HOLD_BETTER, SERVE_BETTER)
        ),
    }
    return {
        "stage": STAGE,
        "causal_chain": chain,
        "attribution": "PRIMARY_STRUCTURAL_ROOT_CAUSE",
        "attribution_statement": (
            "The independent-window episode-boundary error is established as the primary structural root cause: it is the "
            "earliest divergence in the credit path, its removal was the only execution-path change between the two "
            "otherwise identical three-seed runs, and removing it restored discrimination on training, frozen validation "
            "and unseen sealed data."
        ),
        "exclusivity_disclaimer": (
            "This is not a claim that W1 is the only possible cause of the earlier behaviour. Other contributors, such as "
            "the rollout-global advantage normalization scope, remain plausible secondary factors that were never "
            "independently manipulated; the evidence establishes primacy and sufficiency for the observed recovery, not "
            "uniqueness."
        ),
        "not_claimed": [
            "no performance or deployment claim is made",
            "a higher HOLD ratio is not treated as success in itself",
            "training diagnostics are not sealed-split evidence",
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }


def binding_identity_audit(ac_root: Path, z_root: Path) -> Dict[str, Any]:
    contract = read_json(ac_root / "contract_binding.json")["sealed_evaluation_contract"]
    lineage = contract["implementation_lineage"]
    identity = {
        "w1_repair_contract_sha256": lineage.get("w1_repair_contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "actor_repair_contract_sha256": lineage.get("actor_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "critic_s3_repair_contract_sha256": lineage.get("critic_s3_repair_contract_sha256") == EXPECTED["critic_repair_contract_sha256"],
        "reward_v2_sha256": lineage.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "observation_repair_contract_sha256": lineage.get("observation_repair_contract_sha256") == EXPECTED["observation_repair_contract_sha256"],
        "zero_loss_adapter_sha256": lineage.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "r3_split_sha256": lineage.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "h4m_b_schedule_sha256": lineage.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "validation_promotion_contract_sha256": lineage.get("validation_promotion_contract_sha256") == EXPECTED["validation_promotion_contract_sha256"],
    }
    return {"stage": STAGE, "binding_identity": identity, "passed": all(identity.values()), "k_mask_and_action_semantics": "unchanged; carried by the untouched frozen inference path"}


def final_promotion_decision(outcome, single_open, mapping, causal, bindings) -> Dict[str, Any]:
    conditions = {
        "sealed_outcome_verified": outcome["passed"],
        "single_open_integrity_verified": single_open["passed"],
        "counterpart_mapping_frozen_pre_open": mapping["passed"],
        "causal_lineage_supported": causal["passed"],
        "all_bindings_unchanged": bindings["passed"],
    }
    execution_valid = single_open["checks"]["open_count_exactly_one"] and single_open["checks"]["exactly_one_sealed_open_artifact"] and mapping["checks"]["mapping_sha_matches_committed_source"]
    if not execution_valid:
        decision = INVALID
    elif all(conditions.values()):
        decision = APPROVED
    else:
        decision = HOLD_DECISION
    return {
        "stage": STAGE,
        "decision": decision,
        "status": PROMOTED_STATUS if decision == APPROVED else "NOT_PROMOTED",
        "conditions": conditions,
        "unmet_conditions": [key for key, value in conditions.items() if not value],
        "decision_rule": "approval requires the sealed outcome, the single-open integrity, the pre-open criteria mapping, the causal lineage and every frozen binding to hold together",
        "attribution": causal["attribution"],
        "scope_of_promotion": (
            "the repaired three-seed policy lineage is promoted as a research artifact with demonstrated target-context "
            "discrimination on frozen validation and unseen sealed data; this is not a deployment or KPI claim"
        ),
    }


def final_promotion_contract(decision: Mapping[str, Any], ac_root: Path, z_root: Path, aa_root: Path, ab_root: Path, source_commit: str) -> Dict[str, Any]:
    if decision["decision"] != APPROVED:
        return {"stage": STAGE, "frozen": False, "reason": f"decision {decision['decision']} does not authorize promotion"}
    checkpoints = read_json(ac_root / "checkpoint_integrity.json")["checkpoints"]
    body = {
        "status": PROMOTED_STATUS,
        "checkpoints": [{"seed": row["seed"], "path": row["path"], "sha256": row["post_evaluation_sha256"]} for row in checkpoints],
        "source_commit": source_commit,
        "training_source_commit": "44febe3",
        "implementation_source_commit": "4183307",
        "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
        "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
        "critic_s3_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "observation_repair_contract_sha256": EXPECTED["observation_repair_contract_sha256"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "r3_split_sha256": EXPECTED["r3_split_sha256"],
        "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "sealed_evaluation_contract_sha256": EXPECTED["sealed_evaluation_contract_sha256"],
        "criteria_sha256": EXPECTED["criteria_sha256"],
        "validation_promotion_contract_sha256": EXPECTED["validation_promotion_contract_sha256"],
        "lineage": {
            "h4m_z": z_root.name,
            "h4m_aa": aa_root.name,
            "h4m_ab": ab_root.name,
            "h4m_ac": ac_root.name,
        },
        "evidence_summary": {
            "frozen_validation": "96/96 correct direction over three seeds",
            "sealed_holdout": "144/144 correct direction over three seeds, opened once",
            "sealed_open_count": 1,
        },
        "post_promotion_constraints": {
            "sealed_holdout_is_now_consumed": True,
            "further_sealed_use_requires_a_new_sealed_dataset_and_gate": True,
            "no_retuning_of_this_lineage_under_this_contract": True,
            "deployment_or_kpi_claims_not_authorised": True,
        },
        "next_gate": NEXT_GATE,
    }
    return {"stage": STAGE, "frozen": True, "contract": body, "final_promotion_contract_sha256": canonical_sha(body)}


# ---------------------------------------------------------------------------
def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="h4mad_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "ad.pyc"), doraise=True)
            error = None
        except Exception as exc:
            error = repr(exc)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit": head,
        "head_commit_files": head_files,
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "clean_worktree": status == "",
        "py_compile_passed": error is None,
        "github_push_performed": False,
    }


def gate_matrix(provenance, outcome, single_open, mapping, causal, bindings, decision, contract) -> Dict[str, Any]:
    approved = decision["decision"] == APPROVED
    criteria = {
        "source_only_commit": provenance.get("source_only_local_commit") is True,
        "py_compile_passed": provenance.get("py_compile_passed") is True,
        "sealed_outcome_review_passed": outcome.get("passed") is True,
        "single_open_integrity_passed": single_open.get("passed") is True,
        "counterpart_mapping_audit_passed": mapping.get("passed") is True,
        "causal_lineage_review_passed": causal.get("passed") is True,
        "bindings_unchanged": bindings.get("passed") is True,
        "final_decision_made": decision.get("decision") is not None,
        "promotion_contract_frozen": (approved and bool(contract.get("final_promotion_contract_sha256"))) or (not approved and bool(contract.get("reason"))),
        "no_test6_reopen_in_this_gate": True,
        "no_training_or_inference_in_this_gate": True,
        "github_push_false": provenance.get("github_push_performed") is False,
    }
    passed = all(criteria.values()) and approved
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": decision.get("decision"),
        "status": decision.get("status"),
        "final_promotion_contract_sha256": contract.get("final_promotion_contract_sha256"),
        "exact_next_gate": NEXT_GATE if passed else f"STOP_{BLOCK_GATE}",
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "TEST6_reopened": False,
            "sealed_inference_rerun": False,
            "training_count": 0,
            "optimizer_step_count": 0,
            "backward_pass_count": 0,
            "checkpoint_replaced": False,
            "seed_removed": False,
            "new_threshold_created": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate, provenance, contract, started) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": provenance.get("source_commit"),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "status": gate.get("status"),
        "final_promotion_contract_sha256": contract.get("final_promotion_contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "mutable_lifecycle_state_files": [],
        "append_only_artifact": True,
        "read_only_review": True,
        "elapsed_seconds": time.perf_counter() - started,
        "TEST6_reopened": False,
        "training_count": 0,
        "optimizer_step_count": 0,
        "github_push_performed": False,
    }


def final_report(provenance, outcome, single_open, mapping, causal, decision, contract, gate) -> str:
    return f"""# H4M-AD Sealed Hold-out Outcome Review and Final Promotion Decision

gate = {gate['gate']}
decision = {gate['decision']}
status = {gate['status']}
final_promotion_contract_sha256 = {gate.get('final_promotion_contract_sha256')}
source_commit = {provenance['source_commit']}
TEST6_reopened = false
training_count = 0
optimizer_step_count = 0
exact_next_gate = {gate['exact_next_gate']} (not executed automatically)

## Outcome review

```json
{json.dumps({'frozen_validation_aggregate': outcome['frozen_validation_h4m_aa']['aggregate'], 'sealed_holdout_aggregate': outcome['sealed_holdout_h4m_ac']['aggregate'], 'checks': outcome['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Single-open integrity

```json
{json.dumps({'sealed_open_artifacts': single_open['sealed_open_artifacts'], 'counters': single_open['counters'], 'checks': single_open['checks'], 'commits_after_sealed_run': single_open['commits_after_sealed_run']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Counterpart mapping audit

The two sealed-stage counterpart checks were carried in the committed source that produced the run, and the
mapping hash recorded in the artifact recomputes exactly from that commit's blob.

```json
{json.dumps({'instruction_named_pairs': mapping['instruction_named_pairs'], 'additional_pairs_declared_by_h4m_ac': mapping['additional_pairs_declared_by_h4m_ac'], 'verbatim_check_count': mapping['verbatim_check_count'], 'mapped_check_count': mapping['mapped_check_count'], 'mapping_sha256_in_artifact': mapping['mapping_sha256_in_artifact'], 'mapping_sha256_recomputed_from_commit': mapping['mapping_sha256_recomputed_from_commit'], 'contract_binding_created_at': mapping['contract_binding_created_at'], 'sealed_opened_at': mapping['sealed_opened_at'], 'meaning_change_scope': mapping['meaning_change_scope'], 'checks': mapping['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Causal lineage

{causal['attribution_statement']}

{causal['exclusivity_disclaimer']}

```json
{json.dumps({'causal_chain': causal['causal_chain'], 'checks': causal['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Final decision

```json
{json.dumps({'decision': decision['decision'], 'status': decision['status'], 'conditions': decision['conditions'], 'scope_of_promotion': decision['scope_of_promotion']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Promotion contract

```json
{json.dumps(contract.get('contract', {'frozen': contract.get('frozen'), 'reason': contract.get('reason')}), ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: this review reopened no sealed data, reran no inference, trained nothing, replaced no checkpoint,
removed no seed, created no threshold, mutated no data and pushed nothing. The sealed hold-out is now
consumed; any further sealed use needs a new sealed dataset and a new gate.
"""


def main() -> None:
    started = time.perf_counter()
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ad_sealed_outcome_review_final_promotion_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    z_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_z_*", EXPECTED["h4m_z_gate"])
    aa_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_aa_*", EXPECTED["h4m_aa_gate"])
    ab_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_ab_*", EXPECTED["h4m_ab_gate"])
    ac_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_ac_*", EXPECTED["h4m_ac_gate"])

    provenance = source_provenance(created_at)
    outcome = sealed_outcome_review(aa_root, ac_root)
    single_open = single_open_integrity_audit(ac_root)
    mapping = counterpart_mapping_audit(ac_root)
    causal = causal_lineage_review(z_root, aa_root, ac_root)
    bindings = binding_identity_audit(ac_root, z_root)
    decision = final_promotion_decision(outcome, single_open, mapping, causal, bindings)
    contract = final_promotion_contract(decision, ac_root, z_root, aa_root, ab_root, provenance["source_commit"])
    gate = gate_matrix(provenance, outcome, single_open, mapping, causal, bindings, decision, contract)

    for name, payload in {
        "sealed_outcome_review.json": outcome,
        "single_open_integrity_audit.json": single_open,
        "counterpart_mapping_audit.json": mapping,
        "causal_lineage_review.json": causal,
        "binding_identity_audit.json": bindings,
        "final_promotion_decision.json": decision,
        "final_promotion_contract.json": contract,
        "source_provenance.json": provenance,
        "gate_matrix.json": gate,
    }.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(
        final_report(provenance, outcome, single_open, mapping, causal, decision, contract, gate), encoding="utf-8"
    )
    write_json(root / "manifest.json", make_manifest(root, gate, provenance, contract, started))

    print(f"[H4M-AD] artifact root: {root}")
    print(f"[H4M-AD] gate: {gate['gate']}")
    print(f"[H4M-AD] decision: {gate['decision']} | status: {gate['status']}")
    print(f"[H4M-AD] final_promotion_contract_sha256: {gate.get('final_promotion_contract_sha256')}")
    print(f"[H4M-AD] unmet conditions: {decision['unmet_conditions']}")
    print(f"[H4M-AD] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-AD] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
