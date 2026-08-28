#!/usr/bin/env python3
"""H4M-Z window-boundary repaired retraining outcome review and promotion.

Read-only causal review of the W1-before (H4M-U-R2) and W1-after (H4M-Y)
evidence, followed by a conditional promotion of the repaired lineage to the
frozen validation stage.

No training, no optimizer step, no backward pass, no tuning, no reward or model
change, no W2/W3/W4, no extra cycles, no database mutation, no TEST6 access, and
no GitHub push.  Historical artifacts are read only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-Z"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Z_"
    "WINDOW_BOUNDARY_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_VALIDATION_PROMOTION_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Z"
NEXT_GATE = "H4M-AA_FROZEN_WINDOW_BOUNDARY_REPAIRED_POLICY_VALIDATION"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

BEFORE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_u_r2_fresh_three_seed_retraining_with_durable_evidence_20260818_001408+09:00"
AFTER_TRAINING_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_fresh_window_boundary_repaired_three_seed_retraining_20260818_150812+09:00"
AFTER_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_r1_report_only_recovery_from_persisted_evidence_20260818_153018+09:00"
H4MX_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_x_window_boundary_repair_implementation_equivalence_validation_20260818_133041+09:00"
H4MW_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_w_window_episode_boundary_credit_repair_selection_freeze_20260818_131002+09:00"
H4MV_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_v_repaired_retraining_outcome_review_next_decision_20260818_010904+09:00"

EXECUTION_PATH_SOURCES = [
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py",
    "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/observation_target_context_repair.py",
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py",
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py",
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_h_fresh_instrumented_credit_diagnostic_retraining.py",
    "05_training/durable_training_evidence.py",
]

EXPECTED = {
    "before_training_commit": "be8602e736f231ed87e4f3df976e0f0b954023a5",
    "after_training_commit": "44febe3",
    "h4m_x_source_commit": "418330767b2a3d0e3030009744a2814b2889fb9b",
    "h4m_y_recovery_commit": "58805af",
    "h4m_y_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Y_"
        "FRESH_WINDOW_EPISODE_BOUNDARY_REPAIRED_THREE_SEED_RETRAINING_WITH_DURABLE_EVIDENCE_COMPLETE"
    ),
    "h4m_y_outcome": "WINDOW_BOUNDARY_REPAIR_RESTORED_TARGET_CONTEXT_DISCRIMINATION",
    "w1_repair_id": "W1_WINDOW_EPISODE_BOUNDARY_MASKING",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "seeds": [1, 2, 3],
    "outer_training_count": 11,
    "expected_cycle_records": 33,
    "expected_samples": 11616,
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "causal_before_after_comparison.json",
    "three_seed_recovery_analysis.json",
    "competing_explanation_review.json",
    "root_cause_decision.json",
    "validation_promotion_contract.json",
    "gate_matrix.json",
    "authoritative_binding.json",
    "changed_files.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
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


def fnum(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def validation_view(path: Path) -> Dict[str, Any]:
    aggregate = read_json(path).get("aggregate_by_classification", {})
    view: Dict[str, Any] = {}
    for label in (HOLD_BETTER, SERVE_BETTER):
        row = aggregate.get(label, {})
        view[label] = {
            "count": row.get("count"),
            "P_HOLD_mean": fnum((row.get("P_HOLD") or {}).get("mean")),
            "P_SERVE_mean": fnum((row.get("P_SERVE") or {}).get("mean")),
            "probability_dominant_action": row.get("probability_dominant_action"),
            "sampled_action_counts": row.get("sampled_action_counts"),
            "probability_correct_direction_rate": fnum(row.get("probability_correct_direction_rate")),
            "sampled_correct_direction_rate": fnum(row.get("sampled_correct_direction_rate")),
            "by_seed": row.get("by_seed"),
        }
    return view


# ---------------------------------------------------------------------------
# 1. causal before/after comparison
# ---------------------------------------------------------------------------
def causal_before_after_comparison() -> Dict[str, Any]:
    before = validation_view(BEFORE_ROOT / "validation_discrimination.json")
    after = validation_view(AFTER_ROOT / "validation_discrimination.json")
    before_credit = read_json(BEFORE_ROOT / "critic_credit_diagnostics.json")
    after_credit = read_json(AFTER_ROOT / "context_credit_diagnostics.json")
    after_boundary = read_json(AFTER_ROOT / "window_boundary_diagnostics.json")
    before_leakage = read_json(H4MV_ROOT / "credit_causality_audit.json")

    def credit_row(payload: Mapping[str, Any], label: str, action: str) -> Dict[str, Any]:
        row = payload["by_context"][label]["by_sampled_action"][action]
        return {
            "sample_count": row["sample_count"],
            "V_s_minus_return_mean": fnum(row["V_s_minus_return"]["mean"]),
            "raw_GAE_mean": fnum(row["raw_GAE"]["mean"]),
            "normalized_advantage_mean": fnum(row["normalized_advantage"]["mean"]),
            "P_HOLD_mean": fnum(row["P_HOLD"]["mean"]),
        }

    total_correct_after = sum(
        int(round((after[label]["probability_correct_direction_rate"] or 0.0) * (after[label]["count"] or 0)))
        for label in (HOLD_BETTER, SERVE_BETTER)
    )
    total_after = sum(after[label]["count"] or 0 for label in (HOLD_BETTER, SERVE_BETTER))

    checks = {
        "before_hold_better_correct_direction_0_2667": abs((before[HOLD_BETTER]["probability_correct_direction_rate"] or 0) - 0.2667) < 5e-4,
        "before_serve_better_correct_direction_1_0": before[SERVE_BETTER]["probability_correct_direction_rate"] == 1.0,
        "before_p_hold_hold_better_0_3458": abs((before[HOLD_BETTER]["P_HOLD_mean"] or 0) - 0.3458) < 5e-4,
        "before_p_hold_serve_better_0_028": abs((before[SERVE_BETTER]["P_HOLD_mean"] or 0) - 0.028) < 1e-3,
        "after_hold_better_all_hold": (after[HOLD_BETTER]["sampled_action_counts"] or {}).get(HOLD) == after[HOLD_BETTER]["count"]
        and after[HOLD_BETTER]["probability_correct_direction_rate"] == 1.0,
        "after_serve_better_all_serve": (after[SERVE_BETTER]["sampled_action_counts"] or {}).get(SERVE) == after[SERVE_BETTER]["count"]
        and after[SERVE_BETTER]["probability_correct_direction_rate"] == 1.0,
        "after_p_hold_hold_better_0_8985": abs((after[HOLD_BETTER]["P_HOLD_mean"] or 0) - 0.8985) < 5e-4,
        "after_p_hold_serve_better_0_0014": abs((after[SERVE_BETTER]["P_HOLD_mean"] or 0) - 0.0014) < 5e-4,
        "after_validation_correct_direction_96_of_96": total_correct_after == total_after == 96,
        "after_cross_window_leakage_zero": after_boundary["totals"]["cross_window_links_with_live_bootstrap"] == 0,
        "after_terminated_all_samples": after_boundary["totals"]["terminated_true"] == EXPECTED["expected_samples"]
        and after_boundary["totals"]["samples"] == EXPECTED["expected_samples"],
        "after_bootstrap_mask_zero_all_samples": after_boundary["totals"]["bootstrap_mask_zero"] == EXPECTED["expected_samples"],
        "before_had_full_cross_window_recursion": (before_leakage.get("gae_cross_window_fraction") or 0.0) > 0.99,
    }
    return {
        "stage": STAGE,
        "before": {
            "artifact": BEFORE_ROOT.name,
            "lineage": "Actor repair + S3 critic repair, no W1",
            "validation": before,
            "credit": {
                "hold_better_sampled_hold": credit_row(before_credit, HOLD_BETTER, HOLD),
                "hold_better_sampled_serve": credit_row(before_credit, HOLD_BETTER, SERVE),
                "serve_better_sampled_serve": credit_row(before_credit, SERVE_BETTER, SERVE),
            },
            "cross_window_gae_fraction": before_leakage.get("gae_cross_window_fraction"),
            "cross_window_tail_variance_share": read_json(H4MV_ROOT / "credit_chain_attenuation.json").get(
                "cross_window_tail_variance_share_of_advantage"
            ),
        },
        "after": {
            "artifact": AFTER_ROOT.name,
            "training_artifact": AFTER_TRAINING_ROOT.name,
            "lineage": "Actor repair + S3 critic repair + W1 window episode boundary",
            "validation": after,
            "credit": {
                "hold_better_sampled_hold": credit_row(after_credit, HOLD_BETTER, HOLD),
                "hold_better_sampled_serve": credit_row(after_credit, HOLD_BETTER, SERVE),
                "serve_better_sampled_serve": credit_row(after_credit, SERVE_BETTER, SERVE),
            },
            "boundary_totals": after_boundary["totals"],
            "action_contrast_normalized_advantage": after_credit.get("action_contrast_normalized_advantage"),
        },
        "deltas": {
            "hold_better_correct_direction": (after[HOLD_BETTER]["probability_correct_direction_rate"] or 0)
            - (before[HOLD_BETTER]["probability_correct_direction_rate"] or 0),
            "hold_better_P_HOLD": (after[HOLD_BETTER]["P_HOLD_mean"] or 0) - (before[HOLD_BETTER]["P_HOLD_mean"] or 0),
            "serve_better_correct_direction": (after[SERVE_BETTER]["probability_correct_direction_rate"] or 0)
            - (before[SERVE_BETTER]["probability_correct_direction_rate"] or 0),
            "serve_better_P_HOLD": (after[SERVE_BETTER]["P_HOLD_mean"] or 0) - (before[SERVE_BETTER]["P_HOLD_mean"] or 0),
        },
        "serve_better_not_degraded": (after[SERVE_BETTER]["probability_correct_direction_rate"] or 0)
        >= (before[SERVE_BETTER]["probability_correct_direction_rate"] or 0),
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# 2. three-seed recovery analysis
# ---------------------------------------------------------------------------
def three_seed_recovery_analysis() -> Dict[str, Any]:
    evolution = read_json(AFTER_ROOT / "cycle_action_evolution.json")["by_cycle"]
    after = validation_view(AFTER_ROOT / "validation_discrimination.json")
    per_seed_trajectory: Dict[str, List[Dict[str, Any]]] = {str(seed): [] for seed in EXPECTED["seeds"]}
    for entry in evolution:
        for seed_row in entry["by_class"][HOLD_BETTER]["per_seed"]:
            per_seed_trajectory[str(seed_row["seed"])].append(
                {
                    "outer_cycle": entry["outer_cycle"],
                    "hold_share": seed_row["hold_share"],
                    "sampled_dominant_action": seed_row["sampled_dominant_action"],
                    "P_HOLD_mean": seed_row["P_HOLD_mean"],
                }
            )
    serve_better_trajectory = [
        {
            "outer_cycle": entry["outer_cycle"],
            "mean_hold_share": entry["by_class"][SERVE_BETTER]["mean_hold_share"],
            "all_seeds_serve_dominant": entry["by_class"][SERVE_BETTER]["all_seeds_serve_dominant"],
        }
        for entry in evolution
    ]
    cycle1 = evolution[0]["by_class"][HOLD_BETTER]
    cycle2 = evolution[1]["by_class"][HOLD_BETTER]
    hold_dominant_cycles = [entry["outer_cycle"] for entry in evolution if entry["by_class"][HOLD_BETTER]["all_seeds_hold_dominant"]]
    recovery_cycles = [cycle for cycle in hold_dominant_cycles if cycle >= 2]
    single_action_collapse_cycles = [
        entry["outer_cycle"]
        for entry in evolution
        if any(row["hold_share"] in (0.0, 1.0) for row in entry["by_class"][HOLD_BETTER]["per_seed"])
    ]
    per_seed_recovery_cycle = {
        seed: next((row["outer_cycle"] for row in rows if row["outer_cycle"] >= 2 and row["sampled_dominant_action"] == HOLD), None)
        for seed, rows in per_seed_trajectory.items()
    }
    per_seed_validation = {
        seed: {
            HOLD_BETTER: after[HOLD_BETTER]["by_seed"].get(seed),
            SERVE_BETTER: after[SERVE_BETTER]["by_seed"].get(seed),
        }
        for seed in per_seed_trajectory
    }
    checks = {
        "cycle_2_serve_transition_recurred": cycle1["all_seeds_hold_dominant"] and cycle2["all_seeds_serve_dominant"],
        "transition_did_not_become_persistent_collapse": bool(recovery_cycles),
        "no_single_action_collapse_in_hold_better": not single_action_collapse_cycles,
        "all_three_seeds_recovered_hold_dominance": all(value is not None for value in per_seed_recovery_cycle.values()),
        "all_three_seeds_hold_better_correct_direction_1_0": all(
            (row[HOLD_BETTER] or {}).get("probability_correct_direction_rate") == 1.0 for row in per_seed_validation.values()
        ),
        "all_three_seeds_serve_better_correct_direction_1_0": all(
            (row[SERVE_BETTER] or {}).get("probability_correct_direction_rate") == 1.0 for row in per_seed_validation.values()
        ),
        "serve_better_stays_serve_dominant_after_transition": all(
            row["all_seeds_serve_dominant"] for row in serve_better_trajectory if row["outer_cycle"] >= 2
        ),
    }
    return {
        "stage": STAGE,
        "distinction": "a cycle-2 transition to SERVE dominance is not the same failure as a persistent single-action collapse",
        "hold_better_per_seed_trajectory": per_seed_trajectory,
        "serve_better_trajectory": serve_better_trajectory,
        "cycle_1_all_seeds_hold_dominant": cycle1["all_seeds_hold_dominant"],
        "cycle_2_all_seeds_serve_dominant": cycle2["all_seeds_serve_dominant"],
        "cycles_where_all_seeds_hold_dominant": hold_dominant_cycles,
        "first_recovery_cycle_per_seed": per_seed_recovery_cycle,
        "single_action_collapse_cycles": single_action_collapse_cycles,
        "final_cycle_mean_hold_share": {
            HOLD_BETTER: evolution[-1]["by_class"][HOLD_BETTER]["mean_hold_share"],
            SERVE_BETTER: evolution[-1]["by_class"][SERVE_BETTER]["mean_hold_share"],
        },
        "per_seed_validation": per_seed_validation,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# 3. competing explanation review
# ---------------------------------------------------------------------------
W1_EXPECTED_CHANGED_DEFS = {"collect_controlled_rollout", "materialize_rollout_credit_trace"}
W1_EXPECTED_ADDED_DEFS = {
    "window_episode_boundary_mask",
    "const::W1_REPAIR_ID",
    "const::W1_REPAIR_CONTRACT_SHA256",
    "const::WINDOW_EPISODE_BOUNDARY_SCHEMA",
}
CREDIT_CRITICAL_DEFS = [
    "ppo_update_controlled",
    "bind_critic_value_target",
    "apply_return_normalizer_update_after_critic_update",
    "return_normalizer_from_state",
    "shadow_smoke_trace",
    "TraceRecorder",
]


def top_level_definition_digests(text: str) -> Dict[str, str]:
    """SHA256 of every top-level definition and module constant."""
    tree = ast.parse(text)
    digests: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            digests[node.name] = hashlib.sha256((ast.get_source_segment(text, node) or "").encode("utf-8")).hexdigest()
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    digests[f"const::{target.id}"] = hashlib.sha256(
                        (ast.get_source_segment(text, node) or "").encode("utf-8")
                    ).hexdigest()
    return digests


def git_show(revision: str, path: str) -> str:
    return git_run(["show", f"{revision}:{path}"]).stdout


def competing_explanation_review() -> Dict[str, Any]:
    before_commit = EXPECTED["before_training_commit"]
    after_commit = git_run(["rev-parse", EXPECTED["after_training_commit"]]).stdout.strip()
    numstat = git_run(["diff", "--numstat", before_commit, after_commit]).stdout.strip().splitlines()
    changed = {}
    for line in numstat:
        added, removed, path = line.split("\t")
        changed[path] = {"added": int(added), "removed": int(removed)}
    execution_changed = {path: row for path, row in changed.items() if path in EXECUTION_PATH_SOURCES}
    non_execution_changed = sorted(path for path in changed if path not in EXECUTION_PATH_SOURCES)
    before_defs = top_level_definition_digests(git_show(before_commit, EXECUTION_PATH_SOURCES[0]))
    after_defs = top_level_definition_digests(git_show(after_commit, EXECUTION_PATH_SOURCES[0]))
    added_defs = sorted(set(after_defs) - set(before_defs))
    removed_defs = sorted(set(before_defs) - set(after_defs))
    changed_defs = sorted(name for name in set(before_defs) & set(after_defs) if before_defs[name] != after_defs[name])
    unchanged_defs = sorted(name for name in set(before_defs) & set(after_defs) if before_defs[name] == after_defs[name])
    credit_critical_identical = {name: before_defs.get(name) == after_defs.get(name) for name in CREDIT_CRITICAL_DEFS}

    before_manifest = read_json(BEFORE_ROOT / "manifest.json")
    after_header = read_json(AFTER_ROOT / "recovery_binding.json")["run_header"]
    before_training = read_json(BEFORE_ROOT / "training_summary.json")
    after_training = read_json(AFTER_ROOT / "training_summary.json")
    x_equivalence = read_json(H4MX_ROOT / "equivalence_validation.json")

    frozen_identical = {
        "reward_v2_sha256": before_manifest.get("reward_v2_sha256") == after_header.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "actor_repair_contract_sha256": (before_manifest.get("repair_shas") or {}).get("actor")
        == after_header.get("actor_repair_contract_sha256")
        == EXPECTED["actor_repair_contract_sha256"],
        "critic_repair_contract_sha256": (before_manifest.get("repair_shas") or {}).get("critic")
        == after_header.get("critic_repair_contract_sha256")
        == EXPECTED["critic_repair_contract_sha256"],
        "r3_split_sha256": before_manifest.get("split_sha256") == after_header.get("split_sha256") == EXPECTED["r3_split_sha256"],
        "h4m_b_schedule_sha256": before_manifest.get("schedule_sha256") == after_header.get("schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
    }
    budget_identical = {
        "seeds": before_training.get("seeds") == after_training.get("seeds") == EXPECTED["seeds"],
        "total_rollouts": before_training.get("total_rollouts") == after_training.get("total_rollouts") == 33,
        "total_ppo_updates": before_training.get("total_ppo_updates") == after_training.get("total_ppo_updates") == 132,
        "total_critic_updates": before_training.get("total_critic_updates") == after_training.get("total_critic_updates") == 264,
        "fresh_initialization_both_runs": before_training.get("fresh_initialization_all_seeds") is True
        and after_training.get("fresh_initialization_all_seeds") is True,
        "no_old_checkpoint_reuse": before_training.get("h4m_c_or_h4m_h_checkpoint_continuation") in (False, None)
        and after_training.get("old_checkpoint_reuse") is False,
    }
    checks = {
        "only_one_execution_path_file_changed": list(execution_changed) == [EXECUTION_PATH_SOURCES[0]],
        "td_gae_actor_critic_source_unchanged": EXECUTION_PATH_SOURCES[1] not in changed,
        "reward_source_unchanged": EXECUTION_PATH_SOURCES[3] not in changed,
        "observation_source_unchanged": EXECUTION_PATH_SOURCES[4] not in changed,
        "training_loop_source_unchanged": EXECUTION_PATH_SOURCES[5] not in changed,
        "execution_lineage_source_unchanged": EXECUTION_PATH_SOURCES[6] not in changed,
        "durable_evidence_source_unchanged": EXECUTION_PATH_SOURCES[8] not in changed,
        "only_rollout_assembly_definitions_changed": set(changed_defs) <= W1_EXPECTED_CHANGED_DEFS,
        "only_w1_declarations_added": set(added_defs) <= W1_EXPECTED_ADDED_DEFS,
        "no_definition_removed": not removed_defs,
        "credit_critical_definitions_identical": all(credit_critical_identical.values()),
        "compute_gae_equivalence_proved_by_h4m_x": x_equivalence.get("checks", {}).get("compute_gae_source_diff_zero") is True,
        "reward_actor_critic_equivalence_proved_by_h4m_x": x_equivalence.get("checks", {}).get("reward_v2_output_diff_zero") is True
        and x_equivalence.get("checks", {}).get("actor_logits_masks_diff_zero") is True
        and x_equivalence.get("checks", {}).get("critic_diff_zero") is True,
        "frozen_shas_identical_across_runs": all(frozen_identical.values()),
        "budget_and_schedule_identical": all(budget_identical.values()),
        "no_tuning_between_runs": all(budget_identical.values()) and all(frozen_identical.values()),
    }
    return {
        "stage": STAGE,
        "before_training_commit": before_commit,
        "after_training_commit": after_commit,
        "changed_files_between_runs": changed,
        "execution_path_files_changed": execution_changed,
        "non_execution_files_changed": non_execution_changed,
        "non_execution_files_role": "new read-only review, selection, validation and harness runners; none is imported by the frozen training path",
        "top_level_definition_identity": {
            "definitions_before": len(before_defs),
            "definitions_after": len(after_defs),
            "added": added_defs,
            "removed": removed_defs,
            "changed": changed_defs,
            "unchanged_count": len(unchanged_defs),
            "credit_critical_identical": credit_critical_identical,
        },
        "frozen_sha_identity": frozen_identical,
        "budget_identity": budget_identical,
        "competing_explanations_considered": {
            "reward_change": "excluded: reward source unchanged and H4M-X proved identical Reward V2 output",
            "actor_or_critic_change": "excluded: dl1 unchanged and H4M-X proved identical logits, masks and critic values",
            "gae_equation_change": "excluded: compute_gae source diff is zero",
            "normalization_change": "excluded: the advantage normalization block is inside the unchanged compute_gae",
            "split_schedule_or_budget_change": "excluded: identical SHAs and identical 33/132/264 counts",
            "checkpoint_warm_start": "excluded: both runs report fresh initialization for all three seeds",
            "hyperparameter_tuning": "excluded: no hyperparameter source changed between the two commits",
            "instrumentation_change": "the durable evidence writer is unchanged; the only recorded-value change is bootstrap_mask and td_delta now reporting what the credit path actually used",
            "some_other_edit_inside_the_changed_file": "excluded by AST identity: only the rollout assembly and its trace materializer changed; ppo_update_controlled, the S3 binding, the shadow trace and the recorder are byte-identical",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# 4. root-cause decision
# ---------------------------------------------------------------------------
def root_cause_decision(comparison: Mapping[str, Any], seeds: Mapping[str, Any], competing: Mapping[str, Any]) -> Dict[str, Any]:
    mechanism = read_json(H4MV_ROOT / "root_cause_review.json")
    requirements = {
        "root_cause_mechanism_demonstrated": mechanism.get("unique_root_cause") == "V2_TARGET_RETURN_CONSTRUCTION"
        and read_json(H4MV_ROOT / "earliest_remaining_divergence.json").get("earliest_remaining_divergence_stage")
        == "rollout_trajectory_assembly_and_episode_boundary",
        "cross_window_contamination_removed": comparison["checks"]["after_cross_window_leakage_zero"]
        and comparison["checks"]["after_terminated_all_samples"]
        and comparison["checks"]["before_had_full_cross_window_recursion"],
        "both_target_contexts_correct": comparison["checks"]["after_hold_better_all_hold"]
        and comparison["checks"]["after_serve_better_all_serve"],
        "three_seed_consistency": seeds["checks"]["all_three_seeds_hold_better_correct_direction_1_0"]
        and seeds["checks"]["all_three_seeds_serve_better_correct_direction_1_0"]
        and seeds["checks"]["all_three_seeds_recovered_hold_dominance"],
        "no_competing_upstream_change_explains_recovery": competing["passed"] is True,
        "serve_better_not_degraded": comparison["serve_better_not_degraded"] is True,
        "before_after_values_match_authoritative_lineage": comparison["passed"] is True,
    }
    if all(requirements.values()):
        classification = "A_PRIMARY_CAUSAL_ROOT_CAUSE_CONFIRMED_WINDOW_EPISODE_BOUNDARY"
    elif requirements["cross_window_contamination_removed"] and requirements["both_target_contexts_correct"] and not requirements["no_competing_upstream_change_explains_recovery"]:
        classification = "B_STRONG_ASSOCIATION_BUT_CAUSAL_ATTRIBUTION_NOT_YET_SUFFICIENT"
    elif requirements["cross_window_contamination_removed"] and not requirements["both_target_contexts_correct"]:
        classification = "C_WINDOW_BOUNDARY_REPAIR_HELPED_BUT_OTHER_PRIMARY_CAUSE_REMAINS"
    else:
        classification = "D_INSUFFICIENT_EVIDENCE"
    return {
        "stage": STAGE,
        "classification": classification,
        "requirements_for_A": requirements,
        "unmet_requirements": [key for key, value in requirements.items() if not value],
        "mechanism": mechanism.get("mechanism"),
        "not_claimed": [
            "A is not claimed because HOLD increased",
            "the repaired checkpoints are not promoted to any sealed hold-out",
            "no performance or deployment claim is made",
        ],
        "evidence_summary": {
            "hold_better_correct_direction": {
                "before": comparison["before"]["validation"][HOLD_BETTER]["probability_correct_direction_rate"],
                "after": comparison["after"]["validation"][HOLD_BETTER]["probability_correct_direction_rate"],
            },
            "serve_better_correct_direction": {
                "before": comparison["before"]["validation"][SERVE_BETTER]["probability_correct_direction_rate"],
                "after": comparison["after"]["validation"][SERVE_BETTER]["probability_correct_direction_rate"],
            },
            "cross_window_gae_fraction_before": comparison["before"]["cross_window_gae_fraction"],
            "cross_window_leakage_after": comparison["after"]["boundary_totals"]["cross_window_links_with_live_bootstrap"],
            "execution_path_files_changed_between_runs": list(competing["execution_path_files_changed"]),
        },
    }


# ---------------------------------------------------------------------------
# 5. validation promotion contract
# ---------------------------------------------------------------------------
def validation_promotion_contract(decision: Mapping[str, Any]) -> Dict[str, Any]:
    if not decision["classification"].startswith("A_"):
        return {
            "stage": STAGE,
            "promoted": False,
            "reason": f"classification {decision['classification']} does not authorize promotion",
        }
    checkpoints = read_json(AFTER_ROOT / "checkpoint_integrity.json")["checkpoints"]
    body = {
        "promotion": "PROMOTED_TO_FROZEN_VALIDATION_STAGE",
        "promoted_lineage": "H4M-P actor head specialization + H4M-T S3 critic value target + H4M-W W1 window episode boundary",
        "w1_repair_id": EXPECTED["w1_repair_id"],
        "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
        "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
        "critic_s3_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "r3_split_sha256": EXPECTED["r3_split_sha256"],
        "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "training_checkpoints": [
            {"seed": row["seed"], "path": row["path"], "sha256": row["sha256"], "fresh_initialization": row["fresh_initialization"]}
            for row in checkpoints
        ],
        "implementation_source_commit": EXPECTED["h4m_x_source_commit"],
        "training_source_commit": EXPECTED["after_training_commit"],
        "recovery_source_commit": EXPECTED["h4m_y_recovery_commit"],
        "evidence_lineage": {
            "training_artifact": AFTER_TRAINING_ROOT.name,
            "training_artifact_gate": read_json(AFTER_TRAINING_ROOT / "gate_matrix.json").get("gate"),
            "recovery_artifact": AFTER_ROOT.name,
            "recovery_artifact_gate": read_json(AFTER_ROOT / "gate_matrix.json").get("gate"),
            "cycle_evidence_cardinality": "33/33",
        },
        "validation_scope": "already-approved frozen validation split only",
        "test6_authorised": False,
        "checkpoint_test_promotion_status": "NOT_TEST_PROMOTED_UNTIL_VALIDATION_PASSES",
        "validation_must_be_read_only": {
            "training": 0,
            "optimizer_step": 0,
            "backward": 0,
            "parameter_mutation": 0,
        },
        "next_gate": NEXT_GATE,
    }
    return {"stage": STAGE, "promoted": True, "contract": body, "contract_sha256": canonical_sha(body)}


# ---------------------------------------------------------------------------
def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit": head,
        "parent_commit": parent,
        "head_commit_files": head_files,
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "clean_worktree": status == "",
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    y_gate = read_json(AFTER_ROOT / "gate_matrix.json")
    y_outcome = read_json(AFTER_ROOT / "outcome_classification.json")
    y_binding = read_json(AFTER_ROOT / "recovery_binding.json")
    x_gate = read_json(H4MX_ROOT / "gate_matrix.json")
    w_freeze = read_json(H4MW_ROOT / "repair_freeze_contract.json")
    with tempfile.TemporaryDirectory(prefix="h4mz_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "z.pyc"), doraise=True)
            compile_error = None
        except Exception as exc:
            compile_error = repr(exc)
    checks = {
        "source_only_local_commit": provenance.get("source_only_local_commit") is True,
        "py_compile_passed": compile_error is None,
        "h4m_y_gate_match": y_gate.get("gate") == EXPECTED["h4m_y_gate"],
        "h4m_y_outcome_match": y_outcome.get("outcome_classification") == EXPECTED["h4m_y_outcome"],
        "h4m_y_no_retraining": y_gate.get("final_flags", {}).get("optimizer_step_count") == 0
        and y_gate.get("final_flags", {}).get("backward_pass_count") == 0,
        "h4m_y_training_artifact_unmutated": y_binding["read_only_attestation"]["training_artifact_mutated"] is False,
        "h4m_x_gate_pass": str(x_gate.get("gate", "")).startswith("PASS_"),
        "w1_contract_sha_match": w_freeze.get("contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "blocked_training_artifact_present": (AFTER_TRAINING_ROOT / "gate_matrix.json").exists(),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "source_provenance": provenance,
        "py_compile_error": compile_error,
        "authoritative_artifacts": {
            "w1_before_training": str(BEFORE_ROOT),
            "w1_after_training_blocked": str(AFTER_TRAINING_ROOT),
            "w1_after_recovery": str(AFTER_ROOT),
            "w1_implementation": str(H4MX_ROOT),
            "w1_selection": str(H4MW_ROOT),
            "root_cause_review": str(H4MV_ROOT),
        },
        "read_only_attestation": {
            "training_count": 0,
            "optimizer_step_count": 0,
            "backward_pass_count": 0,
            "test6_access_count": 0,
            "tuning_applied": False,
            "reward_or_model_change": False,
            "w2_w3_w4_applied": False,
            "additional_cycles": False,
            "database_or_data_mutation": False,
            "historical_artifacts_modified": False,
            "github_push_performed": False,
        },
    }


def changed_files_audit() -> Dict[str, Any]:
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    return {
        "stage": STAGE,
        "head_commit_files": head_files,
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()],
        "artifact_or_log_committed": any("artifacts/" in path for path in head_files),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "github_push_performed": False,
    }


def gate_matrix(binding, comparison, seeds, competing, decision, contract, changed) -> Dict[str, Any]:
    promoted = contract.get("promoted") is True
    criteria = {
        "authoritative_binding": binding.get("authoritative_binding_passed") is True,
        "source_only_commit": changed.get("source_only_local_commit") is True and changed.get("artifact_or_log_committed") is False,
        "before_after_comparison_verified": comparison.get("passed") is True,
        "three_seed_recovery_verified": seeds.get("passed") is True,
        "competing_explanations_excluded": competing.get("passed") is True,
        "root_cause_classified": decision.get("classification") is not None,
        "decision_is_evidence_backed": not decision.get("unmet_requirements") if promoted else True,
        "promotion_contract_valid": (contract.get("promoted") is True and bool(contract.get("contract_sha256")))
        or (contract.get("promoted") is False and bool(contract.get("reason"))),
        "no_training": binding["read_only_attestation"]["training_count"] == 0,
        "no_optimizer_step": binding["read_only_attestation"]["optimizer_step_count"] == 0,
        "test6_zero": binding["read_only_attestation"]["test6_access_count"] == 0,
        "historical_artifacts_unmodified": binding["read_only_attestation"]["historical_artifacts_modified"] is False,
        "github_push_false": changed.get("github_push_performed") is False,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_{decision.get('classification', 'REVIEW_FAILED')}",
        "root_cause_classification": decision.get("classification"),
        "promotion": "PROMOTED_TO_FROZEN_VALIDATION_STAGE" if promoted else "NOT_PROMOTED",
        "validation_promotion_contract_sha256": contract.get("contract_sha256"),
        "exact_next_gate": NEXT_GATE if (passed and promoted) else f"STOP_{BLOCK_GATE_PREFIX}",
        "next_gate_scope": "frozen approved validation split only; inference/read-only; no TEST6",
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_count": 0,
            "optimizer_step_count": 0,
            "TEST6_opened": False,
            "checkpoint_test_promoted": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate, binding, contract) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": binding["source_provenance"]["source_commit"],
        "authoritative_artifacts": binding["authoritative_artifacts"],
        "gate": gate.get("gate"),
        "root_cause_classification": gate.get("root_cause_classification"),
        "promotion": gate.get("promotion"),
        "validation_promotion_contract_sha256": contract.get("contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "mutable_lifecycle_state_files": [],
        "append_only_artifact": True,
        "read_only_review": True,
        "training_count": 0,
        "optimizer_step_count": 0,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(binding, comparison, seeds, competing, decision, contract, gate) -> str:
    return f"""# H4M-Z Window-Boundary Repaired Retraining Outcome Review and Validation Promotion

gate = {gate['gate']}
root_cause_classification = {decision['classification']}
promotion = {gate['promotion']}
validation_promotion_contract_sha256 = {contract.get('contract_sha256')}
source_commit = {binding['source_provenance']['source_commit']}
training_count = 0
optimizer_step_count = 0
TEST6_access_count = 0
exact_next_gate = {gate['exact_next_gate']}

## W1 before and after on the same frozen validation split

```json
{json.dumps({'before': comparison['before']['validation'], 'after': comparison['after']['validation'], 'deltas': comparison['deltas']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Credit mechanism

```json
{json.dumps({'before': comparison['before']['credit'], 'after': comparison['after']['credit'], 'before_cross_window_gae_fraction': comparison['before']['cross_window_gae_fraction'], 'after_boundary_totals': comparison['after']['boundary_totals']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Three-seed recovery, transition versus collapse

```json
{json.dumps({'checks': seeds['checks'], 'first_recovery_cycle_per_seed': seeds['first_recovery_cycle_per_seed'], 'final_cycle_mean_hold_share': seeds['final_cycle_mean_hold_share'], 'single_action_collapse_cycles': seeds['single_action_collapse_cycles']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Competing explanations

```json
{json.dumps({'execution_path_files_changed': competing['execution_path_files_changed'], 'considered': competing['competing_explanations_considered'], 'frozen_sha_identity': competing['frozen_sha_identity'], 'budget_identity': competing['budget_identity']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Root-cause decision

```json
{json.dumps({'classification': decision['classification'], 'requirements_for_A': decision['requirements_for_A'], 'unmet': decision['unmet_requirements'], 'not_claimed': decision['not_claimed']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Promotion

```json
{json.dumps(contract.get('contract', {'promoted': contract.get('promoted'), 'reason': contract.get('reason')}), ensure_ascii=False, indent=2, default=jsonable)}
```

STOP conditions honoured: no training, no optimizer step, no tuning, no reward or model change, no
W2/W3/W4, no additional cycles, no data mutation, no TEST6 access, and no GitHub push. The blocked H4M-Y
training artifact and the H4M-Y-R1 recovery artifact were read only. The repaired checkpoints stay
non-test-promoted until the frozen validation stage passes.
"""


def main() -> None:
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_z_window_boundary_outcome_review_validation_promotion_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    comparison = causal_before_after_comparison()
    seeds = three_seed_recovery_analysis()
    competing = competing_explanation_review()
    decision = root_cause_decision(comparison, seeds, competing)
    contract = validation_promotion_contract(decision)
    changed = changed_files_audit()
    gate = gate_matrix(binding, comparison, seeds, competing, decision, contract, changed)

    for name, payload in {
        "authoritative_binding.json": binding,
        "causal_before_after_comparison.json": comparison,
        "three_seed_recovery_analysis.json": seeds,
        "competing_explanation_review.json": competing,
        "root_cause_decision.json": decision,
        "validation_promotion_contract.json": contract,
        "changed_files.json": changed,
        "gate_matrix.json": gate,
    }.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(final_report(binding, comparison, seeds, competing, decision, contract, gate), encoding="utf-8")
    write_json(root / "manifest.json", make_manifest(root, gate, binding, contract))

    print(f"[H4M-Z] artifact root: {root}")
    print(f"[H4M-Z] gate: {gate['gate']}")
    print(f"[H4M-Z] root cause: {decision['classification']}")
    print(f"[H4M-Z] promotion: {gate['promotion']}")
    print(f"[H4M-Z] contract sha256: {contract.get('contract_sha256')}")
    print(f"[H4M-Z] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-Z] exact next gate: {gate['exact_next_gate']}")


if __name__ == "__main__":
    main()
