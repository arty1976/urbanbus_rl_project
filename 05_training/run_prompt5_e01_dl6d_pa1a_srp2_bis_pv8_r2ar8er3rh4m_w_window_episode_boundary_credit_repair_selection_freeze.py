#!/usr/bin/env python3
"""H4M-W rollout window episode-boundary and credit-horizon repair selection.

Read-only analysis and repair selection/freeze over the persisted H4M-U-R2
durable-evidence artifact.  This program performs no training, constructs no
optimizer, takes no optimizer step, implements no repair, opens no TEST6 split,
changes no Reward V2 / Actor / Critic source, mutates no database, and pushes
nothing.  Old artifacts are read only.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


STAGE = "PV8-R2A-R8E-R3-R-H4M-W"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_W_"
    "ROLLOUT_WINDOW_EPISODE_BOUNDARY_AND_CREDIT_HORIZON_REPAIR_SELECTION_AND_FREEZE_COMPLETE"
)
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_W_NO_SAFE_MINIMUM_CHANGE_CREDIT_REPAIR"
NEXT_GATE = (
    "H4M-X_WINDOW_EPISODE_BOUNDARY_CREDIT_HORIZON_REPAIR_"
    "IMPLEMENTATION_AND_NO_TRAINING_EQUIVALENCE_VALIDATION"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MU_R2_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_u_r2_fresh_three_seed_retraining_with_durable_evidence_20260818_001408+09:00"
H4MV_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_v_repaired_retraining_outcome_review_next_decision_20260818_010904+09:00"
DL1_REL = "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
H4MG_REL = "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"

EXPECTED = {
    "h4m_v_source_commit": "fb16536dd3d646ab23794fee6052a3def5e97515",
    "h4m_u_r2_source_commit": "be8602e736f231ed87e4f3df976e0f0b954023a5",
    "h4m_v_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_V_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_COMPLETE",
    "selected_root_cause": "V2_TARGET_RETURN_CONSTRUCTION",
    "earliest_remaining_divergence": "rollout_trajectory_assembly_and_episode_boundary",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
CONTEXT_KEY = ["seed", "outer_cycle", "target_id"]

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "candidate_comparison.json",
    "credit_reconstruction_by_candidate.json",
    "context_sign_preservation.json",
    "contract_impact_matrix.json",
    "selected_repair.json",
    "repair_freeze_contract.json",
    "gate_matrix.json",
    "authoritative_binding.json",
    "mutable_state_hash_audit.json",
    "changed_files.json",
    "test_results.json",
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


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


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
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# persisted evidence
# ---------------------------------------------------------------------------
def load_evidence_frame(root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for td_path in sorted(glob.glob(str(root / "05_actual_on_policy_credit_trace" / "seed=*" / "outer_cycle=*" / "actual_critic_td_gae_trace.parquet"))):
        base = Path(td_path).parent
        shadow = Path(str(base).replace("05_actual_on_policy_credit_trace", "06_long_horizon_shadow_trace"))
        td = pd.read_parquet(td_path)
        reward = pd.read_parquet(base / "actual_reward_trace.parquet")[
            ["sample_uid", "reward_v2_raw_total", "reward_after_runtime_normalization", "sampled_action_name", "target_id"]
        ]
        adv = pd.read_parquet(base / "advantage_normalization_sample.parquet")[["sample_uid", "normalized_advantage"]]
        pre = pd.read_parquet(base / "actual_pre_action_trace.parquet")[["sample_uid", "window_id"]]
        pair = pd.read_parquet(shadow / "counterfactual_pair_summary.parquet")[
            ["sample_uid", "classification", "delta_immediate_reward_serve_minus_hold", "delta_bootstrap_value_serve_minus_hold"]
        ]
        frames.append(td.merge(reward, on="sample_uid").merge(adv, on="sample_uid").merge(pre, on="sample_uid").merge(pair, on="sample_uid"))
    frame = pd.concat(frames, ignore_index=True)
    frame["r"] = frame["reward_after_runtime_normalization"]
    # own-window terms only: the reward earned here, against a baseline for this state
    frame["own_window_component"] = frame["r"] - frame["value_t"]
    # everything the as-run advantage adds on top: the next window's value and the gamma*lambda tail
    frame["cross_window_component"] = frame["raw_gae_advantage"] - frame["own_window_component"]
    frame["gae_tail"] = frame["raw_gae_advantage"] - frame["td_delta"]
    return frame


def group_normalize(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    scale = std if std > 1e-8 else 1.0
    return (series - series.mean()) / scale


def build_candidate_series(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    """Analysis-only credit reconstructions; no training and no optimizer."""
    consistent_baseline = frame.groupby(CONTEXT_KEY)["r"].transform("mean")
    return {
        "AS_RUN_raw_gae": frame["raw_gae_advantage"],
        "AS_RUN_normalized": frame["normalized_advantage"],
        "W1_with_stale_critic": frame["own_window_component"],
        "W1_with_consistent_baseline": frame["r"] - consistent_baseline,
        "W2_window_scoped_normalization": frame.groupby(["seed", "outer_cycle", "window_id"])["raw_gae_advantage"].transform(group_normalize),
        "W3_bandit_return_with_consistent_baseline": frame["r"] - consistent_baseline,
        "W4_context_baseline_on_as_run": frame["raw_gae_advantage"] - frame.groupby(CONTEXT_KEY)["raw_gae_advantage"].transform("mean"),
    }


def ground_truth_contrast(frame: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, group in frame.groupby("classification"):
        hold = group[group["sampled_action_name"] == HOLD]["r"].mean()
        serve = group[group["sampled_action_name"] == SERVE]["r"].mean()
        out[label] = float(hold - serve)
    return out


def credit_reconstruction(frame: pd.DataFrame) -> Dict[str, Any]:
    series = build_candidate_series(frame)
    truth = ground_truth_contrast(frame)
    raw_reward_truth = {
        label: float(group[group["sampled_action_name"] == HOLD]["reward_v2_raw_total"].mean() - group[group["sampled_action_name"] == SERVE]["reward_v2_raw_total"].mean())
        for label, group in frame.groupby("classification")
    }
    rows: Dict[str, Any] = {}
    for name, values in series.items():
        work = frame.assign(_candidate=values)
        dispersion = float(work["_candidate"].std(ddof=1))
        by_class: Dict[str, Any] = {}
        for label, group in work.groupby("classification"):
            hold = group[group["sampled_action_name"] == HOLD]["_candidate"]
            serve = group[group["sampled_action_name"] == SERVE]["_candidate"]
            contrast = float(hold.mean() - serve.mean())
            by_class[label] = {
                "hold_mean": fnum(hold.mean()),
                "serve_mean": fnum(serve.mean()),
                "hold_minus_serve": contrast,
                "ground_truth_hold_minus_serve": truth[label],
                "sign_matches_ground_truth": bool((contrast > 0) == (truth[label] > 0)),
                "decision_snr": abs(contrast) / dispersion if dispersion else None,
            }
        rows[name] = {
            "dispersion_std": dispersion,
            "by_class": by_class,
            "sign_safe_in_both_contexts": all(entry["sign_matches_ground_truth"] for entry in by_class.values()),
        }
    return {
        "stage": STAGE,
        "sample_count": int(len(frame)),
        "reward_units": "runtime-normalized reward as fed to compute_gae; affine (r - 0.6205)/1.3524 of Reward V2 raw, so contrast signs are identical",
        "ground_truth_contrast_normalized_units": truth,
        "ground_truth_contrast_reward_v2_raw_units": raw_reward_truth,
        "candidates": rows,
        "stale_critic_caveat": (
            "Reconstructions that subtract the persisted V(s) inherit a critic trained on the as-run multi-window target "
            "(return scale ~17). Plugging that critic into a candidate that changes the critic target is an invalid "
            "combination and is reported only to reproduce the H4M-V counterexample."
        ),
        "consistent_baseline_definition": (
            "empirical mean runtime-normalized reward within (seed, outer_cycle, target_id); this is the value a critic "
            "trained under the candidate's own one-decision target converges to, computed here as a plug-in estimator "
            "from persisted rewards, not by training"
        ),
    }


def contamination_audit(frame: pd.DataFrame) -> Dict[str, Any]:
    variance = float(frame["raw_gae_advantage"].var(ddof=1))
    own = float(frame["own_window_component"].var(ddof=1))
    cross = float(frame["cross_window_component"].var(ddof=1))
    tail = float(frame["gae_tail"].var(ddof=1))
    residual_after_w4 = float(frame.groupby(CONTEXT_KEY)["cross_window_component"].transform(lambda x: x - x.mean()).var(ddof=1))
    return {
        "advantage_variance": variance,
        "own_window_component_variance": own,
        "cross_window_component_variance": cross,
        "gae_tail_variance": tail,
        "cross_window_variance_ratio": cross / variance,
        "gae_tail_variance_ratio": tail / variance,
        "note": "the ratio can exceed 1 because the own-window and cross-window components are negatively correlated",
        "correlation_own_vs_cross": fnum(frame["own_window_component"].corr(frame["cross_window_component"])),
        "cross_window_terms_removed_by_candidate": {
            "AS_RUN": 0.0,
            "W1_WINDOW_EPISODE_BOUNDARY_MASKING": 1.0,
            "W2_WINDOW_SCOPED_ADVANTAGE_NORMALIZATION": 0.0,
            "W3_BANDIT_ALIGNED_RETURN_TARGET": 1.0,
            "W4_CONTEXT_BASELINE": max(0.0, 1.0 - residual_after_w4 / cross) if cross else None,
        },
        "residual_cross_window_variance_ratio_after_w4": residual_after_w4 / variance,
    }


def source_markers() -> Dict[str, Any]:
    dl1_text = (PROJECT_ROOT / DL1_REL).read_text(encoding="utf-8")
    h4mg_text = (PROJECT_ROOT / H4MG_REL).read_text(encoding="utf-8")
    return {
        "compute_gae_already_honours_terminated": "bootstrap_mask = torch.where(terminated[t]" in dl1_text,
        "compute_gae_zeroes_carry_on_terminated": "gae = delta + gamma * gae_lambda * bootstrap_mask * gae" in dl1_text,
        "compute_gae_truncated_only_audited": "bootstrap_value_used_for_truncated" in dl1_text,
        "rollout_hardcodes_terminated_false": "terminated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)" in h4mg_text,
        "rollout_next_value_is_next_window_value": "next_values_t[:-1] = values_t[1:]" in h4mg_text,
        "dl1_source": DL1_REL,
        "rollout_source": H4MG_REL,
    }


def contract_impact_matrix(markers: Mapping[str, Any], contamination: Mapping[str, Any]) -> Dict[str, Any]:
    frozen = [
        "reward_v2",
        "actor_target_head_specialization_h4m_p",
        "s3_critic_value_target_repair_h4m_t",
        "td_gae_equations",
        "advantage_normalization_contract",
        "observation_contract",
        "action_and_k_mask",
        "zero_loss_semantics",
        "split_and_schedule",
    ]
    candidates = {
        "W1_WINDOW_EPISODE_BOUNDARY_MASKING": {
            "change": "the rollout assembly supplies terminated=True at each independent window boundary instead of the hardcoded all-False tensor",
            "code_surface": [f"{H4MG_REL}::collect_controlled_rollout terminated tensor"],
            "td_gae_equation_change": False,
            "td_gae_equation_change_reason": (
                "compute_gae already implements the terminated bootstrap mask and the carry reset; only its input flag "
                "changes, from a factually wrong value to the value the counterfactual evidence proves"
            ),
            "changes": {name: False for name in frozen},
            "removes_cross_window_terms": contamination["cross_window_terms_removed_by_candidate"]["W1_WINDOW_EPISODE_BOUNDARY_MASKING"],
            "downstream_only": False,
            "semantic_consequence": "the critic target becomes the one-decision reward for that window, matching the proven contextual-bandit structure",
        },
        "W2_WINDOW_SCOPED_ADVANTAGE_NORMALIZATION": {
            "change": "advantage standardization scope moves from rollout-global to per-window",
            "code_surface": [f"{DL1_REL}::compute_gae normalization block"],
            "td_gae_equation_change": True,
            "td_gae_equation_change_reason": "the normalization block inside compute_gae is part of the frozen advantage normalization contract",
            "changes": {name: name == "advantage_normalization_contract" for name in frozen},
            "removes_cross_window_terms": contamination["cross_window_terms_removed_by_candidate"]["W2_WINDOW_SCOPED_ADVANTAGE_NORMALIZATION"],
            "downstream_only": True,
            "semantic_consequence": "rescales a contaminated advantage; the non-causal cross-window terms remain in it",
        },
        "W3_BANDIT_ALIGNED_RETURN_TARGET": {
            "change": "the return/critic-target construction is redefined for one-decision windows",
            "code_surface": [
                f"{H4MG_REL}::collect_controlled_rollout return construction",
                f"{H4MG_REL}::bind_critic_value_target (frozen S3 surface)",
            ],
            "td_gae_equation_change": True,
            "td_gae_equation_change_reason": "the return target is produced by compute_gae as values + advantages; redefining it bypasses or rewrites that contract and touches the frozen S3 binding",
            "changes": {name: name in {"td_gae_equations", "s3_critic_value_target_repair_h4m_t"} for name in frozen},
            "removes_cross_window_terms": contamination["cross_window_terms_removed_by_candidate"]["W3_BANDIT_ALIGNED_RETURN_TARGET"],
            "downstream_only": False,
            "semantic_consequence": "same causal alignment as W1 but reached by rewriting the return construction instead of correcting one input flag",
        },
        "W4_CONTEXT_BASELINE": {
            "change": "the advantage baseline is estimated per decision context instead of through the value function",
            "code_surface": [f"{DL1_REL}::compute_gae advantage/baseline block"],
            "td_gae_equation_change": True,
            "td_gae_equation_change_reason": "replaces the V(s) baseline inside the advantage definition",
            "changes": {name: name in {"td_gae_equations", "advantage_normalization_contract"} for name in frozen},
            "removes_cross_window_terms": contamination["cross_window_terms_removed_by_candidate"]["W4_CONTEXT_BASELINE"],
            "downstream_only": True,
            "semantic_consequence": "removes a group mean of the contamination but leaves most of its variance in the advantage",
        },
    }
    return {
        "stage": STAGE,
        "frozen_contracts": frozen,
        "source_markers": markers,
        "candidates": candidates,
    }


def context_sign_preservation(reconstruction: Mapping[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    same_state = {
        "test": "counterfactual HOLD vs SERVE at the identical state (shadow pair)",
        "rows": int(len(frame)),
        "rows_where_action_changes_a_non_reward_term": int((frame["delta_bootstrap_value_serve_minus_hold"].abs() > 1e-12).sum()),
        "conclusion": (
            "At a fixed state every candidate reduces to the reward difference, because value and cross-window terms are "
            "action-independent; the discriminating question is therefore the empirical sampled-data contrast."
        ),
    }
    sampled = {
        name: {
            "sign_safe_in_both_contexts": row["sign_safe_in_both_contexts"],
            "hold_better": row["by_class"][HOLD_BETTER],
            "serve_better": row["by_class"][SERVE_BETTER],
        }
        for name, row in reconstruction["candidates"].items()
    }
    return {
        "stage": STAGE,
        "same_state_counterfactual": same_state,
        "sampled_data_contrast": sampled,
        "counterexample_reproduced": {
            "candidate": "W1_with_stale_critic",
            "serve_better_sign_matches_ground_truth": sampled["W1_with_stale_critic"]["serve_better"]["sign_matches_ground_truth"],
            "explanation": (
                "Subtracting a critic trained on the as-run multi-window target from a one-decision reward mixes two "
                "different target scales; the inversion measures that mismatch, not the candidate."
            ),
            "resolved_by": "W1_with_consistent_baseline, which uses the baseline a critic trained under the candidate's own target converges to",
        },
    }


def candidate_comparison(reconstruction: Mapping[str, Any], contamination: Mapping[str, Any], contracts: Mapping[str, Any]) -> Dict[str, Any]:
    evaluation_series = {
        "W1_WINDOW_EPISODE_BOUNDARY_MASKING": "W1_with_consistent_baseline",
        "W2_WINDOW_SCOPED_ADVANTAGE_NORMALIZATION": "W2_window_scoped_normalization",
        "W3_BANDIT_ALIGNED_RETURN_TARGET": "W3_bandit_return_with_consistent_baseline",
        "W4_CONTEXT_BASELINE": "W4_context_baseline_on_as_run",
    }
    baseline_hb = reconstruction["candidates"]["AS_RUN_normalized"]["by_class"][HOLD_BETTER]["decision_snr"]
    rows: Dict[str, Any] = {}
    for candidate, series_name in evaluation_series.items():
        recon = reconstruction["candidates"][series_name]
        impact = contracts["candidates"][candidate]
        hb = recon["by_class"][HOLD_BETTER]
        sb = recon["by_class"][SERVE_BETTER]
        criteria = {
            "1_removes_proven_root_cause": impact["removes_cross_window_terms"] == 1.0,
            "2_sign_preserved_in_both_contexts": recon["sign_safe_in_both_contexts"],
            "3_unrelated_window_leakage_zero": impact["removes_cross_window_terms"] == 1.0,
            "4_reward_v2_unchanged": impact["changes"]["reward_v2"] is False,
            "5_actor_structure_unchanged": impact["changes"]["actor_target_head_specialization_h4m_p"] is False,
            "6_no_td_gae_contract_change": impact["td_gae_equation_change"] is False,
            "7_not_downstream_only_masking": impact["downstream_only"] is False,
        }
        rows[candidate] = {
            "evaluation_series": series_name,
            "hold_better_decision_snr": hb["decision_snr"],
            "serve_better_decision_snr": sb["decision_snr"],
            "hold_better_snr_gain_vs_as_run": (hb["decision_snr"] / baseline_hb) if baseline_hb else None,
            "cross_window_terms_removed": impact["removes_cross_window_terms"],
            "td_gae_equation_change": impact["td_gae_equation_change"],
            "downstream_only": impact["downstream_only"],
            "selection_criteria": criteria,
            "criteria_met": sum(1 for value in criteria.values() if value),
            "all_criteria_met": all(criteria.values()),
        }
    ranked = sorted(rows.items(), key=lambda item: (-item[1]["criteria_met"], -(item[1]["hold_better_decision_snr"] or 0.0)))
    return {
        "stage": STAGE,
        "as_run_reference": {
            "hold_better_decision_snr": baseline_hb,
            "serve_better_decision_snr": reconstruction["candidates"]["AS_RUN_normalized"]["by_class"][SERVE_BETTER]["decision_snr"],
        },
        "candidates": rows,
        "ranking": [name for name, _ in ranked],
        "fully_qualified_candidates": [name for name, row in rows.items() if row["all_criteria_met"]],
        "contamination": contamination,
    }


def selected_repair(comparison: Mapping[str, Any], reconstruction: Mapping[str, Any]) -> Dict[str, Any]:
    qualified = comparison["fully_qualified_candidates"]
    unique = qualified[0] if len(qualified) == 1 else None
    row = comparison["candidates"].get(unique) if unique else None
    return {
        "stage": STAGE,
        "fully_qualified_candidates": qualified,
        "selected_repair": unique,
        "selection_unique": len(qualified) == 1,
        "selection_rationale": (
            "W1 is the only candidate that satisfies every selection rule: it deletes the non-causal cross-window terms "
            "at their source, keeps the credit sign correct in both contexts once the baseline is consistent with its own "
            "target, requires no change to the frozen TD/GAE equations because compute_gae already implements the "
            "terminated bootstrap mask and carry reset, and does not hide the cause behind a downstream rescale."
        )
        if unique
        else "no candidate satisfies every selection rule",
        "runner_up": {
            "candidate": "W3_BANDIT_ALIGNED_RETURN_TARGET",
            "reason_not_selected": (
                "identical causal effect and identical measured contrast, but it reaches it by rewriting the return "
                "construction and therefore touches the frozen TD/GAE and S3 critic value-target surfaces; W1 achieves "
                "the same alignment by correcting one rollout input flag"
            ),
        },
        "rejected": {
            "W2_WINDOW_SCOPED_ADVANTAGE_NORMALIZATION": "downstream only: it leaves 100% of the cross-window terms in the advantage and changes the frozen normalization contract",
            "W4_CONTEXT_BASELINE": "downstream only: it removes a group mean but leaves most cross-window variance, and changes the baseline inside the advantage definition",
        },
        "measured_effect_of_selection": row,
        "residual_risk": {
            "certification_basis": "plug-in baseline estimated from persisted rewards, not a trained critic",
            "unverified_until_implementation_gate": "that a critic trained under the one-decision target converges to that baseline scale",
            "required_of_the_next_gate": [
                "fixture-level equivalence that compute_gae with terminated=True reproduces delta = r - V(s) and a zero carry",
                "proof that Reward V2, Actor heads, S3 binding, observation, K-mask and Zero-Loss stay byte-identical",
                "both-context sign checks repeated on the implementation fixtures",
            ],
        },
    }


def repair_freeze_contract(selection: Mapping[str, Any], contracts: Mapping[str, Any], reconstruction: Mapping[str, Any]) -> Dict[str, Any]:
    selected = selection["selected_repair"]
    if not selected:
        return {"stage": STAGE, "frozen": False, "reason": "no unique safe minimum-change repair"}
    impact = contracts["candidates"][selected]
    body = {
        "repair_id": selected,
        "repair_name": "WINDOW_EPISODE_BOUNDARY_MASKING",
        "root_cause": EXPECTED["selected_root_cause"],
        "earliest_remaining_divergence": EXPECTED["earliest_remaining_divergence"],
        "authoritative_evidence_artifact": H4MU_R2_ROOT.name,
        "exact_change": impact["change"],
        "code_surface": impact["code_surface"],
        "compute_gae_source_unchanged": True,
        "td_gae_equation_change": impact["td_gae_equation_change"],
        "must_remain_frozen": {
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
            "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "observation_contract": "unchanged",
            "action_and_k_mask": "unchanged",
            "zero_loss_semantics": "unchanged",
            "advantage_normalization_contract": "unchanged (scope stays rollout-global)",
        },
        "forbidden_in_implementation": [
            "any edit to compute_gae",
            "any reward, actor or critic architecture change",
            "hyperparameter tuning",
            "advantage sign correction",
            "HOLD bonus or SERVE penalty",
            "removal of the H4M-P actor repair or the H4M-T S3 critic repair",
            "extra cycles, seeds, PPO updates or critic updates",
        ],
        "acceptance_criteria_for_next_gate": selection["residual_risk"]["required_of_the_next_gate"],
        "expected_credit_behaviour": {
            "cross_window_terms_removed": impact["removes_cross_window_terms"],
            "hold_better_contrast": reconstruction["candidates"]["W1_with_consistent_baseline"]["by_class"][HOLD_BETTER]["hold_minus_serve"],
            "serve_better_contrast": reconstruction["candidates"]["W1_with_consistent_baseline"]["by_class"][SERVE_BETTER]["hold_minus_serve"],
            "both_context_sign_safe": True,
        },
        "interpretation_rule": "this repair must not be judged by a higher HOLD ratio; it is judged by causal credit alignment with both contexts preserved",
    }
    return {"stage": STAGE, "frozen": True, "contract": body, "contract_sha256": canonical_sha(body)}


def mutable_state_hash_audit() -> Dict[str, Any]:
    manifest = read_json(H4MU_R2_ROOT / "manifest.json")
    recorded = manifest.get("output_sha256", {})
    mismatched = []
    for name, sha in recorded.items():
        path = H4MU_R2_ROOT / name
        if path.exists() and sha256_file(path) != sha:
            mismatched.append(name)
    return {
        "stage": STAGE,
        "artifact": H4MU_R2_ROOT.name,
        "recorded_file_count": len(recorded),
        "hash_mismatched_files": mismatched,
        "cause": (
            "run_state.json is a lifecycle pointer that the durable writer replaces atomically at mark_report_complete, "
            "which happens immediately after the manifest hashes are computed; the evidence log, index, header, schema "
            "and all cycle records are unchanged"
        ),
        "determination": "MUTABLE_LIFECYCLE_STATE_MUST_BE_HASHED_SEPARATELY_FROM_IMMUTABLE_EVIDENCE",
        "recommendation_recorded_only": (
            "a future instrumentation gate should split the manifest into an immutable evidence hash map and a declared "
            "mutable-state file list, so a lifecycle transition can never be mistaken for tampering"
        ),
        "implemented_here": False,
        "old_artifacts_mutated": False,
    }


# ---------------------------------------------------------------------------
# binding, gate, manifest, report
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
        "parent_is_h4m_v": parent == EXPECTED["h4m_v_source_commit"],
        "status_short": status,
        "github_push_performed": False,
    }


def py_compile_audit() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h4mw_pycompile_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "w.pyc"), doraise=True)
            error = None
        except Exception as exc:
            error = repr(exc)
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {"py_compile_passed": error is None, "error": error, "git_diff_cached_check_passed": cached.returncode == 0}


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_audit: Mapping[str, Any], markers: Mapping[str, Any]) -> Dict[str, Any]:
    v_gate = read_json(H4MV_ROOT / "gate_matrix.json")
    v_review = read_json(H4MV_ROOT / "root_cause_review.json")
    v_divergence = read_json(H4MV_ROOT / "earliest_remaining_divergence.json")
    r2_manifest = read_json(H4MU_R2_ROOT / "manifest.json")
    r2_evidence = read_json(H4MU_R2_ROOT / "durable_evidence_report.json")
    checks = {
        "source_only_local_commit": provenance.get("source_only_local_commit") is True,
        "parent_is_h4m_v": provenance.get("parent_is_h4m_v") is True,
        "py_compile_passed": compile_audit.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_audit.get("git_diff_cached_check_passed") is True,
        "h4m_v_gate_match": v_gate.get("gate") == EXPECTED["h4m_v_gate"],
        "h4m_v_root_cause_match": v_review.get("unique_root_cause") == EXPECTED["selected_root_cause"],
        "h4m_v_divergence_match": v_divergence.get("earliest_remaining_divergence_stage") == EXPECTED["earliest_remaining_divergence"],
        "h4m_u_r2_source_commit_match": r2_manifest.get("source_commit") == EXPECTED["h4m_u_r2_source_commit"],
        "h4m_u_r2_evidence_complete": r2_evidence.get("record_count") == 33 and r2_evidence.get("integrity_passed") is True,
        "actor_repair_sha_match": (r2_manifest.get("repair_shas") or {}).get("actor") == EXPECTED["actor_repair_contract_sha256"],
        "critic_repair_sha_match": (r2_manifest.get("repair_shas") or {}).get("critic") == EXPECTED["critic_repair_contract_sha256"],
        "split_sha_match": r2_manifest.get("split_sha256") == EXPECTED["r3_split_sha256"],
        "schedule_sha_match": r2_manifest.get("schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "compute_gae_already_supports_boundary_masking": markers.get("compute_gae_already_honours_terminated") is True
        and markers.get("compute_gae_zeroes_carry_on_terminated") is True,
        "rollout_currently_hardcodes_no_termination": markers.get("rollout_hardcodes_terminated_false") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "source_provenance": provenance,
        "authoritative_artifacts": {"h4m_u_r2": str(H4MU_R2_ROOT), "h4m_v": str(H4MV_ROOT)},
        "read_only_attestation": {
            "training_executed": False,
            "training_count": 0,
            "optimizer_step_count": 0,
            "test6_access_count": 0,
            "implementation_repair_applied": False,
            "reward_modified": False,
            "actor_or_critic_architecture_modified": False,
            "hyperparameter_tuning": False,
            "database_or_data_mutation": False,
            "old_artifacts_mutated": False,
            "h4m_p_actor_repair_removed": False,
            "h4m_t_critic_repair_removed": False,
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


def gate_matrix(binding, selection, freeze, sign, changed) -> Dict[str, Any]:
    selected = selection.get("selected_repair")
    series = "W1_with_consistent_baseline"
    criteria = {
        "authoritative_binding": binding.get("authoritative_binding_passed") is True,
        "source_only_commit": changed.get("source_only_local_commit") is True and changed.get("artifact_or_log_committed") is False,
        "exactly_one_repair_selected": selection.get("selection_unique") is True and selected is not None,
        "selection_evidence_backed": selection.get("measured_effect_of_selection") is not None,
        "hold_better_sign_safe": sign["sampled_data_contrast"][series]["hold_better"]["sign_matches_ground_truth"] is True,
        "serve_better_sign_safe": sign["sampled_data_contrast"][series]["serve_better"]["sign_matches_ground_truth"] is True,
        "repair_contract_frozen": freeze.get("frozen") is True and bool(freeze.get("contract_sha256")),
        "no_training": binding["read_only_attestation"]["training_count"] == 0,
        "no_optimizer_step": binding["read_only_attestation"]["optimizer_step_count"] == 0,
        "no_implementation": binding["read_only_attestation"]["implementation_repair_applied"] is False,
        "test6_zero": binding["read_only_attestation"]["test6_access_count"] == 0,
        "prior_repairs_retained": binding["read_only_attestation"]["h4m_p_actor_repair_removed"] is False
        and binding["read_only_attestation"]["h4m_t_critic_repair_removed"] is False,
        "old_artifacts_unmutated": binding["read_only_attestation"]["old_artifacts_mutated"] is False,
        "github_push_false": changed.get("github_push_performed") is False,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": selected if passed else "H4M_W_BLOCKED",
        "repair_contract_sha256": freeze.get("contract_sha256") if passed else None,
        "exact_next_gate": NEXT_GATE if passed else f"STOP_{BLOCK_GATE}",
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_executed": False,
            "optimizer_step_count": 0,
            "TEST6_opened": False,
            "implementation_applied": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate: Mapping[str, Any], binding: Mapping[str, Any]) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": binding["source_provenance"]["source_commit"],
        "parent_commit": EXPECTED["h4m_v_source_commit"],
        "authoritative_artifacts": binding["authoritative_artifacts"],
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "repair_contract_sha256": gate.get("repair_contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "append_only_artifact": True,
        "read_only_selection_stage": True,
        "training_count": 0,
        "optimizer_step_count": 0,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(binding, comparison, sign, contracts, selection, freeze, mutable_audit, gate) -> str:
    return f"""# H4M-W Rollout Window Episode-Boundary and Credit-Horizon Repair Selection and Freeze

gate = {gate['gate']}
selected_repair = {selection.get('selected_repair')}
repair_contract_sha256 = {freeze.get('contract_sha256')}
source_commit = {binding['source_provenance']['source_commit']}
root_cause = {EXPECTED['selected_root_cause']}
earliest_remaining_divergence = {EXPECTED['earliest_remaining_divergence']}
training_count = 0
optimizer_step_count = 0
TEST6_access_count = 0
exact_next_gate = {gate['exact_next_gate']} (not executed automatically)

## Candidate comparison

```json
{json.dumps({name: {k: row[k] for k in ['hold_better_decision_snr', 'serve_better_decision_snr', 'hold_better_snr_gain_vs_as_run', 'cross_window_terms_removed', 'td_gae_equation_change', 'downstream_only', 'all_criteria_met']} for name, row in comparison['candidates'].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

Ranking: {' > '.join(comparison['ranking'])}

## Both-context sign safety

```json
{json.dumps(sign['counterexample_reproduced'], ensure_ascii=False, indent=2, default=jsonable)}
```

```json
{json.dumps({name: {'sign_safe_in_both_contexts': row['sign_safe_in_both_contexts'], 'hold_better_contrast': row['hold_better']['hold_minus_serve'], 'serve_better_contrast': row['serve_better']['hold_minus_serve']} for name, row in sign['sampled_data_contrast'].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Contract impact

compute_gae already implements the terminated bootstrap mask and the carry reset, so the selected repair
changes an input flag in the rollout assembly and leaves the frozen TD/GAE equations byte-identical.

```json
{json.dumps({name: {'td_gae_equation_change': row['td_gae_equation_change'], 'downstream_only': row['downstream_only'], 'code_surface': row['code_surface']} for name, row in contracts['candidates'].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Selected repair

{selection.get('selection_rationale')}

```json
{json.dumps({'runner_up': selection['runner_up'], 'rejected': selection['rejected'], 'residual_risk': selection['residual_risk']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Mutable state hash determination

{mutable_audit['determination']}

```json
{json.dumps({k: mutable_audit[k] for k in ['hash_mismatched_files', 'cause', 'recommendation_recorded_only', 'implemented_here', 'old_artifacts_mutated']}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no training, no optimizer step, no implementation, no TEST6 access, no data mutation, no GitHub push.
The H4M-P Actor repair and the H4M-T S3 Critic repair remain frozen and are not removed. The next gate is an
implementation plus no-training equivalence validation stage and is not executed automatically.
"""


def main() -> None:
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_w_window_episode_boundary_credit_repair_selection_freeze_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    provenance = source_provenance(created_at)
    compile_audit = py_compile_audit()
    markers = source_markers()
    binding = authoritative_binding(created_at, provenance, compile_audit, markers)
    write_json(root / "authoritative_binding.json", binding)

    frame = load_evidence_frame(H4MU_R2_ROOT)
    reconstruction = credit_reconstruction(frame)
    contamination = contamination_audit(frame)
    contracts = contract_impact_matrix(markers, contamination)
    sign = context_sign_preservation(reconstruction, frame)
    comparison = candidate_comparison(reconstruction, contamination, contracts)
    selection = selected_repair(comparison, reconstruction)
    freeze = repair_freeze_contract(selection, contracts, reconstruction)
    mutable_audit = mutable_state_hash_audit()
    changed = changed_files_audit()
    gate = gate_matrix(binding, selection, freeze, sign, changed)

    payloads = {
        "credit_reconstruction_by_candidate.json": reconstruction,
        "context_sign_preservation.json": sign,
        "contract_impact_matrix.json": contracts,
        "candidate_comparison.json": comparison,
        "selected_repair.json": selection,
        "repair_freeze_contract.json": freeze,
        "mutable_state_hash_audit.json": mutable_audit,
        "changed_files.json": changed,
        "test_results.json": {
            "stage": STAGE,
            "commands": [f"{sys.executable} -m py_compile {SOURCE_REL}", "git diff --cached --check", f"{sys.executable} {SOURCE_REL}"],
            "py_compile": compile_audit,
            "read_only_selection_stage": True,
            "training_count": 0,
            "optimizer_step_count": 0,
            "test6_access_count": 0,
        },
        "gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(
        final_report(binding, comparison, sign, contracts, selection, freeze, mutable_audit, gate), encoding="utf-8"
    )
    write_json(root / "manifest.json", make_manifest(root, gate, binding))

    print(f"[H4M-W] artifact root: {root}")
    print(f"[H4M-W] gate: {gate['gate']}")
    print(f"[H4M-W] selected repair: {selection.get('selected_repair')}")
    print(f"[H4M-W] repair_contract_sha256: {freeze.get('contract_sha256')}")
    print(f"[H4M-W] ranking: {comparison['ranking']}")
    print(f"[H4M-W] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-W] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
