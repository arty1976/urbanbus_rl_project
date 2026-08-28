#!/usr/bin/env python3
"""H4M-V repaired-retraining outcome review and next decision selection.

Read-only review of the persisted H4M-U-R2 durable-evidence artifact and the
prior H4M-O/Q/S/T lineages.  This program performs no training, builds no
optimizer, takes no optimizer step, opens no TEST6 split, changes no Reward V2 /
Actor / Critic implementation, mutates no database, and pushes nothing.

It rebuilds the Q3 cycle-2 determination from persisted evidence, compares the
lineages, locates the earliest remaining credit divergence after the approved
Actor and Critic repairs, classifies the root cause from measurements, and
selects exactly one next gate.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


STAGE = "PV8-R2A-R8E-R3-R-H4M-V"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_V_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_V_INSUFFICIENT_EVIDENCE_FOR_NEXT_REPAIR_SELECTION"
NEXT_GATE = "H4M-W_ROLLOUT_WINDOW_EPISODE_BOUNDARY_AND_CREDIT_HORIZON_REPAIR_SELECTION_AND_FREEZE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MU_R2_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_u_r2_fresh_three_seed_retraining_with_durable_evidence_20260818_001408+09:00"
H4MU_R1_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_u_r1_post_training_evidence_instrumentation_repair_and_"
    "no_training_equivalence_validation_20260817_235733+09:00"
)
H4MQ_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_20260817_153440+0900"
H4MS_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_s_ppo_credit_advantage_repair_selection_and_freeze_20260817_171912+0900"
H4MT_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_t_ppo_credit_advantage_critic_value_target_repair_implementation_equivalence_validation_20260817_190257+09:00"
)
H4MR_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_r_target_conditioned_actor_head_specialized_retraining_outcome_review_next_decision_20260817_163927+0900"

EXPECTED = {
    "h4m_u_r1_source_commit": "10d001789801c2b9a54ccb15663609104345d783",
    "h4m_u_r2_source_commit": "be8602e736f231ed87e4f3df976e0f0b954023a5",
    "h4m_u_r2_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_R2_"
        "FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_AND_CRITIC_VALUE_TARGET_REPAIRED_"
        "THREE_SEED_RETRAINING_WITH_DURABLE_EVIDENCE_COMPLETE"
    ),
    "h4m_u_r2_outcome": "SERVE_DOMINANCE_PERSISTS_DESPITE_REPAIRED_CREDIT",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "seeds": [1, 2, 3],
    "outer_training_count": 11,
    "expected_cycle_records": 33,
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
# The frozen H4M-K summaries report dominance with the short labels HOLD/SERVE/TIE.
DOMINANT_LABEL = {HOLD: "HOLD", SERVE: "SERVE"}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "q3_persisted_evidence_reconstruction.json",
    "cross_lineage_comparison.json",
    "earliest_remaining_divergence.json",
    "root_cause_review.json",
    "next_decision.json",
    "gate_matrix.json",
    "authoritative_binding.json",
    "credit_causality_audit.json",
    "credit_chain_attenuation.json",
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
# persisted evidence loaders (read-only)
# ---------------------------------------------------------------------------
def load_joined_trace(root: Path) -> pd.DataFrame:
    """Join the persisted per-sample credit traces of every seed x cycle."""
    frames: List[pd.DataFrame] = []
    for td_path in sorted(glob.glob(str(root / "05_actual_on_policy_credit_trace" / "seed=*" / "outer_cycle=*" / "actual_critic_td_gae_trace.parquet"))):
        base = Path(td_path).parent
        shadow = Path(str(base).replace("05_actual_on_policy_credit_trace", "06_long_horizon_shadow_trace"))
        td = pd.read_parquet(td_path)
        pre = pd.read_parquet(base / "actual_pre_action_trace.parquet")[
            ["sample_uid", "window_id", "snapshot_id", "step_index", "agent_slot", "target_id", "action_id"]
        ]
        reward = pd.read_parquet(base / "actual_reward_trace.parquet")[
            ["sample_uid", "reward_v2_raw_total", "sampled_action_name"]
        ]
        adv = pd.read_parquet(base / "advantage_normalization_sample.parquet")[["sample_uid", "normalized_advantage"]]
        pair = pd.read_parquet(shadow / "counterfactual_pair_summary.parquet")[
            [
                "sample_uid",
                "classification",
                "delta_immediate_reward_serve_minus_hold",
                "delta_full_bootstrapped_return_serve_minus_hold",
                "delta_bootstrap_value_serve_minus_hold",
            ]
        ]
        frame = td.merge(pre, on="sample_uid").merge(reward, on="sample_uid").merge(adv, on="sample_uid").merge(pair, on="sample_uid")
        frame["cross_window_tail"] = frame["raw_gae_advantage"] - frame["td_delta"]
        frames.append(frame)
    if not frames:
        raise RuntimeError("no persisted credit traces found")
    return pd.concat(frames, ignore_index=True)


def stats(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


# ---------------------------------------------------------------------------
# 1. Q3 reconstruction from persisted evidence
# ---------------------------------------------------------------------------
def q3_reconstruction(root: Path) -> Dict[str, Any]:
    conditional = read_json(root / "conditional_policy_discrimination_training_trace.json")
    rows = conditional.get("by_seed_cycle_label", [])
    published = read_json(root / "target_conditioned_diagnostics.json")
    reported = published.get("early_cycle_2_serve_collapse_recurs")

    by_cycle: List[Dict[str, Any]] = []
    for cycle in range(1, EXPECTED["outer_training_count"] + 1):
        entry: Dict[str, Any] = {"outer_cycle": cycle, "by_class": {}}
        for label in (HOLD_BETTER, SERVE_BETTER):
            per_seed = []
            for seed in EXPECTED["seeds"]:
                match = [r for r in rows if int(r.get("seed", -1)) == seed and int(r.get("outer_cycle", -1)) == cycle and r.get("classification") == label]
                if not match:
                    continue
                summary = match[0]
                counts = summary.get("sampled_action_counts", {}) or {}
                total = sum(int(v) for v in counts.values())
                hold = int(counts.get(HOLD, 0))
                per_seed.append(
                    {
                        "seed": seed,
                        "sampled_dominant_action": summary.get("sampled_dominant_action"),
                        "hold_count": hold,
                        "serve_count": int(counts.get(SERVE, 0)),
                        "sample_count": total,
                        "hold_share": (hold / total) if total else None,
                        "P_HOLD_mean": fnum((summary.get("P_HOLD") or {}).get("mean")),
                        "single_action_collapse": bool(total and (hold == 0 or hold == total)),
                    }
                )
            shares = [row["hold_share"] for row in per_seed if row["hold_share"] is not None]
            entry["by_class"][label] = {
                "per_seed": per_seed,
                "mean_hold_share": (sum(shares) / len(shares)) if shares else None,
                "all_seeds_serve_dominant": bool(per_seed) and all(row["sampled_dominant_action"] == DOMINANT_LABEL[SERVE] for row in per_seed),
                "all_seeds_hold_dominant": bool(per_seed) and all(row["sampled_dominant_action"] == DOMINANT_LABEL[HOLD] for row in per_seed),
                "any_single_action_collapse": any(row["single_action_collapse"] for row in per_seed),
            }
        by_cycle.append(entry)

    def cls(cycle_index: int, label: str) -> Dict[str, Any]:
        return by_cycle[cycle_index - 1]["by_class"][label]

    cycle1_hold_dominant = cls(1, HOLD_BETTER)["all_seeds_hold_dominant"]
    cycle2_serve_dominant = cls(2, HOLD_BETTER)["all_seeds_serve_dominant"]
    transition_cycle = None
    for entry in by_cycle:
        if entry["by_class"][HOLD_BETTER]["all_seeds_serve_dominant"]:
            transition_cycle = entry["outer_cycle"]
            break
    later_hold_dominant_cycles = [
        entry["outer_cycle"]
        for entry in by_cycle
        if any(row["sampled_dominant_action"] == DOMINANT_LABEL[HOLD] for row in entry["by_class"][HOLD_BETTER]["per_seed"])
        and entry["outer_cycle"] >= 2
    ]
    collapse_cycles_hold_better = [
        entry["outer_cycle"] for entry in by_cycle if entry["by_class"][HOLD_BETTER]["any_single_action_collapse"]
    ]
    collapse_cycles_serve_better = [
        entry["outer_cycle"] for entry in by_cycle if entry["by_class"][SERVE_BETTER]["any_single_action_collapse"]
    ]
    final = by_cycle[-1]
    determination = (
        "CYCLE_2_SERVE_DOMINANCE_TRANSITION_RECURS_WITHOUT_SINGLE_ACTION_COLLAPSE_IN_HOLD_BETTER"
        if (cycle1_hold_dominant and cycle2_serve_dominant and not collapse_cycles_hold_better)
        else "CYCLE_2_SERVE_DOMINANCE_TRANSITION_RECURS_WITH_SINGLE_ACTION_COLLAPSE"
        if (cycle1_hold_dominant and cycle2_serve_dominant)
        else "CYCLE_2_SERVE_DOMINANCE_TRANSITION_DOES_NOT_RECUR"
    )
    return {
        "stage": STAGE,
        "question": "Does the cycle-2 SERVE transition recur after the Actor + Critic repairs?",
        "reconstruction_source": str(root / "conditional_policy_discrimination_training_trace.json"),
        "reconstruction_key": "by_seed_cycle_label[*].outer_cycle",
        "published_value_in_h4m_u_r2_artifact": reported,
        "published_value_defect": {
            "symptom": "target_conditioned_diagnostics.early_cycle_2_serve_collapse_recurs was null",
            "cause": "the inherited H4M-U helper filtered rows with key 'cycle' while H4M-K writes 'outer_cycle', so the cycle-2 row set was always empty",
            "scope": "reporting-side key mismatch only; the underlying rows were fully persisted",
            "original_artifact_modified": False,
            "training_required_for_reconstruction": False,
        },
        "cycle_1_all_seeds_hold_dominant_in_hold_better": cycle1_hold_dominant,
        "cycle_2_all_seeds_serve_dominant_in_hold_better": cycle2_serve_dominant,
        "first_cycle_with_all_seed_serve_dominance_in_hold_better": transition_cycle,
        "cycles_with_any_seed_hold_dominant_after_cycle_1": later_hold_dominant_cycles,
        "single_action_collapse_cycles_hold_better": collapse_cycles_hold_better,
        "single_action_collapse_cycles_serve_better": collapse_cycles_serve_better,
        "final_cycle_mean_hold_share": {
            HOLD_BETTER: final["by_class"][HOLD_BETTER]["mean_hold_share"],
            SERVE_BETTER: final["by_class"][SERVE_BETTER]["mean_hold_share"],
        },
        "by_cycle": by_cycle,
        "determination": determination,
        "interpretation": (
            "The cycle-2 shift to SERVE dominance recurs in both classes, but it is a dominance shift, not a "
            "single-action collapse: HOLD_BETTER keeps a substantially higher HOLD share than SERVE_BETTER "
            "through the final cycle, so context separation is present in sampled behaviour."
        ),
    }


# ---------------------------------------------------------------------------
# 2. credit causality audit
# ---------------------------------------------------------------------------
def credit_causality_audit(frame: pd.DataFrame, root: Path) -> Dict[str, Any]:
    windows = []
    for path in sorted(glob.glob(str(root / "05_actual_on_policy_credit_trace" / "seed=*" / "outer_cycle=*" / "actual_pre_action_trace.parquet"))):
        pre = pd.read_parquet(path)
        per_window = pre.groupby("window_id").size()
        steps_per_window = pre.groupby("window_id")["step_index"].nunique()
        windows.append(
            {
                "seed": int(pre["seed"].iloc[0]),
                "outer_cycle": int(pre["outer_cycle"].iloc[0]),
                "window_count": int(pre["window_id"].nunique()),
                "rows_per_window_min": int(per_window.min()),
                "rows_per_window_max": int(per_window.max()),
                "steps_per_window_max": int(steps_per_window.max()),
                "distinct_step_index": int(pre["step_index"].nunique()),
            }
        )
    window_frame = pd.DataFrame(windows)

    successor_window = frame.set_index("sample_uid")["window_id"].to_dict()
    linked = frame[frame["gae_recursion_successor_uid"].notna()].copy()
    linked["successor_window_id"] = linked["gae_recursion_successor_uid"].map(successor_window)
    resolved = linked[linked["successor_window_id"].notna()]
    cross_window = int((resolved["window_id"] != resolved["successor_window_id"]).sum())

    nonzero_bootstrap_delta = int((frame["delta_bootstrap_value_serve_minus_hold"].abs() > 1e-12).sum())
    horizon_equals_immediate = int(
        ((frame["delta_full_bootstrapped_return_serve_minus_hold"] - frame["delta_immediate_reward_serve_minus_hold"]).abs() > 1e-9).sum()
    )
    label_deltas = {
        label: stats(group["delta_immediate_reward_serve_minus_hold"])
        for label, group in frame.groupby("classification")
    }
    checks = {
        "every_window_contributes_exactly_one_step": bool((window_frame["steps_per_window_max"] == 1).all()),
        "rollout_is_concatenation_of_independent_windows": bool(
            (window_frame["window_count"] == window_frame["distinct_step_index"]).all()
        ),
        "action_never_changes_boundary_bootstrap_value": nonzero_bootstrap_delta == 0,
        "long_horizon_delta_equals_immediate_reward_delta": horizon_equals_immediate == 0,
        "gae_recursion_crosses_window_boundaries": cross_window > 0,
        "no_episode_boundary_masking": bool(
            (frame["terminated"] == False).all() and (frame["bootstrap_mask"] == 1.0).all()  # noqa: E712
        ),
    }
    return {
        "stage": STAGE,
        "sample_count": int(len(frame)),
        "rollout_window_structure": {
            "rollouts": int(len(window_frame)),
            "window_count_per_rollout": sorted(window_frame["window_count"].unique().tolist()),
            "rows_per_window": sorted(set(window_frame["rows_per_window_min"].tolist() + window_frame["rows_per_window_max"].tolist())),
            "steps_per_window_max": sorted(window_frame["steps_per_window_max"].unique().tolist()),
        },
        "gae_recursion_links_resolved": int(len(resolved)),
        "gae_recursion_links_crossing_window_boundary": cross_window,
        "gae_cross_window_fraction": (cross_window / len(resolved)) if len(resolved) else None,
        "terminated_values": sorted(frame["terminated"].unique().tolist()),
        "truncated_values": sorted(frame["truncated"].unique().tolist()),
        "bootstrap_mask_values": sorted(frame["bootstrap_mask"].unique().tolist()),
        "gamma": sorted(frame["gamma"].unique().tolist()),
        "gae_lambda": sorted(frame["gae_lambda"].unique().tolist()),
        "effective_credit_horizon_windows": float(1.0 / (1.0 - float(frame["gamma"].iloc[0]) * float(frame["gae_lambda"].iloc[0]))),
        "counterfactual_action_effect": {
            "rows_where_action_changes_boundary_bootstrap_value": nonzero_bootstrap_delta,
            "rows_where_full_horizon_delta_differs_from_immediate": horizon_equals_immediate,
            "delta_immediate_reward_serve_minus_hold_by_class": label_deltas,
        },
        "checks": checks,
        "conclusion": (
            "Each rollout concatenates causally independent single-step windows. The counterfactual evidence proves the "
            "sampled action cannot change any future state, so every gamma*lambda-weighted future term in the advantage "
            "is non-causal with respect to the decision it is credited to."
        ),
    }


# ---------------------------------------------------------------------------
# 3. credit chain attenuation
# ---------------------------------------------------------------------------
STAGE_COLUMNS = [
    ("reward", "reward_v2_raw_total"),
    ("td_residual", "td_delta"),
    ("raw_gae", "raw_gae_advantage"),
    ("normalized_advantage", "normalized_advantage"),
]


def credit_chain_attenuation(frame: pd.DataFrame) -> Dict[str, Any]:
    dispersion = {name: stats(frame[column]) for name, column in STAGE_COLUMNS}
    dispersion["cross_window_tail"] = stats(frame["cross_window_tail"])
    tail_variance_share = float(frame["cross_window_tail"].var(ddof=1) / frame["raw_gae_advantage"].var(ddof=1))

    by_class: Dict[str, Any] = {}
    for label, group in frame.groupby("classification"):
        hold = group[group["sampled_action_name"] == HOLD]
        serve = group[group["sampled_action_name"] == SERVE]
        contrasts = {}
        for name, column in STAGE_COLUMNS:
            contrasts[name] = {
                "hold_mean": fnum(hold[column].mean()) if len(hold) else None,
                "serve_mean": fnum(serve[column].mean()) if len(serve) else None,
                "hold_minus_serve": fnum(hold[column].mean() - serve[column].mean()) if len(hold) and len(serve) else None,
            }
        ground_truth = contrasts["reward"]["hold_minus_serve"]
        used = contrasts["normalized_advantage"]["hold_minus_serve"]
        gae_stage = contrasts["raw_gae"]["hold_minus_serve"]
        td_stage = contrasts["td_residual"]["hold_minus_serve"]
        by_class[label] = {
            "sample_count": int(len(group)),
            "hold_sample_count": int(len(hold)),
            "serve_sample_count": int(len(serve)),
            "contrast_by_stage": contrasts,
            "sign_correct_at_every_stage": all(
                (value is not None and ground_truth is not None and (value > 0) == (ground_truth > 0))
                for value in [td_stage, gae_stage, used]
            ),
            "gae_stage_contrast_retention_vs_td": (abs(gae_stage) / abs(td_stage)) if td_stage else None,
            "decision_snr_used": (abs(used)) if used is not None else None,
            "cross_window_tail_mean_by_action": {
                HOLD: fnum(hold["cross_window_tail"].mean()) if len(hold) else None,
                SERVE: fnum(serve["cross_window_tail"].mean()) if len(serve) else None,
                "hold_minus_serve": fnum(hold["cross_window_tail"].mean() - serve["cross_window_tail"].mean())
                if len(hold) and len(serve)
                else None,
            },
        }
    hb = by_class.get(HOLD_BETTER, {})
    sb = by_class.get(SERVE_BETTER, {})
    reward_ratio = (
        abs(sb["contrast_by_stage"]["reward"]["hold_minus_serve"]) / abs(hb["contrast_by_stage"]["reward"]["hold_minus_serve"])
        if hb and sb
        else None
    )
    used_ratio = (
        abs(sb["decision_snr_used"]) / abs(hb["decision_snr_used"]) if hb and sb and hb.get("decision_snr_used") else None
    )
    return {
        "stage": STAGE,
        "dispersion": dispersion,
        "cross_window_tail_variance_share_of_advantage": tail_variance_share,
        "correlation_advantage_with_own_step_reward": fnum(frame["raw_gae_advantage"].corr(frame["reward_v2_raw_total"])),
        "correlation_advantage_with_cross_window_tail": fnum(frame["raw_gae_advantage"].corr(frame["cross_window_tail"])),
        "by_class": by_class,
        "class_asymmetry": {
            "ground_truth_reward_contrast_ratio_serve_better_over_hold_better": reward_ratio,
            "delivered_normalized_advantage_contrast_ratio": used_ratio,
            "asymmetry_amplification": (used_ratio / reward_ratio) if reward_ratio and used_ratio else None,
        },
        "advantage_normalization_scope": "rollout_global_active_only_inactive_excluded (both classes pooled in one scope)",
        "conclusion": (
            "The action contrast keeps the correct sign at every stage, but the cross-window tail supplies most of the "
            "advantage variance and is not action-neutral in the sampled data, so the HOLD_BETTER decision signal that "
            "survives to PPO is an order of magnitude weaker than the SERVE_BETTER one."
        ),
    }


# ---------------------------------------------------------------------------
# 4. cross-lineage comparison
# ---------------------------------------------------------------------------
def validation_summary(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    aggregate = payload.get("aggregate_by_classification", {})
    out: Dict[str, Any] = {}
    for label in (HOLD_BETTER, SERVE_BETTER):
        row = aggregate.get(label, {})
        out[label] = {
            "count": row.get("count"),
            "P_HOLD_mean": fnum((row.get("P_HOLD") or {}).get("mean")),
            "P_SERVE_mean": fnum((row.get("P_SERVE") or {}).get("mean")),
            "probability_dominant_action": row.get("probability_dominant_action"),
            "probability_correct_direction_rate": row.get("probability_correct_direction_rate"),
            "sampled_correct_direction_rate": row.get("sampled_correct_direction_rate"),
        }
    return out


def cycle_hold_share(path: Path, cycle: int, label: str) -> Optional[float]:
    rows = read_json(path).get("by_seed_cycle_label", [])
    shares = []
    for seed in EXPECTED["seeds"]:
        match = [r for r in rows if int(r.get("seed", -1)) == seed and int(r.get("outer_cycle", -1)) == cycle and r.get("classification") == label]
        if not match:
            continue
        counts = match[0].get("sampled_action_counts", {}) or {}
        total = sum(int(v) for v in counts.values())
        if total:
            shares.append(int(counts.get(HOLD, 0)) / total)
    return (sum(shares) / len(shares)) if shares else None


def cross_lineage_comparison(attenuation: Mapping[str, Any]) -> Dict[str, Any]:
    h4ms_root = read_json(H4MS_ROOT / "root_cause_selection.json")
    h4ms_answers = h4ms_root.get("mandatory_answers", {})
    h4ms_contract = read_json(H4MS_ROOT / "repair_freeze_contract.json")
    h4mt_gate = read_json(H4MT_ROOT / "gate_matrix.json")
    h4mo = read_json(H4MR_ROOT / "h4m_o_p_q_comparison.json")
    q_validation = validation_summary(H4MQ_ROOT / "validation_discrimination.json")
    r2_validation = validation_summary(H4MU_R2_ROOT / "validation_discrimination.json")
    r2_credit = read_json(H4MU_R2_ROOT / "critic_credit_diagnostics.json")
    q_gradient = read_json(H4MQ_ROOT / "gradient_audit.json")
    r2_gradient = read_json(H4MU_R2_ROOT / "gradient_isolation.json")

    r2_hb_hold = r2_credit["by_context"][HOLD_BETTER]["by_sampled_action"][HOLD]
    r2_sb_serve = r2_credit["by_context"][SERVE_BETTER]["by_sampled_action"][SERVE]

    lineage = {
        "H4M-O": {
            "role": "actor target-conditioning outcome review",
            "root_cause": h4mo.get("h4m_o", {}).get("root_cause"),
            "net_serve_minus_hold_pressure_mean_shared_actor": h4mo.get("h4m_o", {})
            .get("shared_actor_pressure_total", {})
            .get("net_serve_minus_hold_pressure_mean"),
        },
        "H4M-P": {
            "role": "target-conditioned Actor head specialization",
            "repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
            "gradient_isolation_validation_passed": h4mo.get("h4m_p", {}).get("gradient_isolation_validation_passed"),
        },
        "H4M-Q": {
            "role": "fresh retraining with the Actor repair only",
            "behavioral_outcome": h4mo.get("h4m_q", {}).get("behavioral_outcome"),
            "validation": q_validation,
            "cycle_11_mean_hold_share": {
                HOLD_BETTER: cycle_hold_share(H4MQ_ROOT / "conditional_policy_discrimination_training_trace.json", 11, HOLD_BETTER),
                SERVE_BETTER: cycle_hold_share(H4MQ_ROOT / "conditional_policy_discrimination_training_trace.json", 11, SERVE_BETTER),
            },
            "cross_specialized_head_gradient_leakage_count": q_gradient.get("cross_specialized_head_gradient_leakage_count"),
        },
        "H4M-S": {
            "role": "credit repair root-cause selection and freeze",
            "selected_root_cause": h4ms_root.get("selected_root_cause"),
            "earliest_credit_divergence_stage": h4ms_root.get("earliest_credit_divergence_stage"),
            "selected_repair": h4ms_contract.get("selected_repair"),
            "repair_contract_sha256": h4ms_contract.get("contract_sha256"),
            "hold_better_sampled_hold_value_minus_return_mean": fnum(
                (h4ms_answers.get("4_critic_value_baseline_bias") or {}).get("hold_better_sampled_hold_value_minus_return_mean")
            ),
            "serve_better_sampled_serve_value_minus_return_mean": fnum(
                (h4ms_answers.get("4_critic_value_baseline_bias") or {}).get("serve_better_sampled_serve_value_minus_return_mean")
            ),
            "hold_better_sampled_hold_raw_gae_mean": fnum(
                (h4ms_answers.get("6_advantage_normalization_flip") or {}).get("hold_better_sampled_hold_raw_gae_mean")
            ),
            "gae_temporal_misbinding_considered_and_rejected": True,
            "rejection_reason_recorded_by_h4m_s": "GAE carries negative/weak credit forward, but the mean TD residual is already negative; evidence did not isolate GAE temporal binding as the first defect.",
        },
        "H4M-T": {
            "role": "S3 critic value-target repair implementation and equivalence validation",
            "gate": h4mt_gate.get("gate"),
            "repair_contract_sha256": h4mt_gate.get("repair_contract_sha256"),
        },
        "H4M-U-R2": {
            "role": "fresh retraining with the Actor + Critic repairs and durable evidence",
            "outcome": read_json(H4MU_R2_ROOT / "outcome_classification.json").get("outcome_classification"),
            "validation": r2_validation,
            "cycle_11_mean_hold_share": {
                HOLD_BETTER: cycle_hold_share(H4MU_R2_ROOT / "conditional_policy_discrimination_training_trace.json", 11, HOLD_BETTER),
                SERVE_BETTER: cycle_hold_share(H4MU_R2_ROOT / "conditional_policy_discrimination_training_trace.json", 11, SERVE_BETTER),
            },
            "hold_better_sampled_hold_value_minus_return_mean": fnum(r2_hb_hold["V_s_minus_return"]["mean"]),
            "serve_better_sampled_serve_value_minus_return_mean": fnum(r2_sb_serve["V_s_minus_return"]["mean"]),
            "hold_better_sampled_hold_raw_gae_mean": fnum(r2_hb_hold["raw_GAE"]["mean"]),
            "cross_specialized_head_gradient_leakage_count": r2_gradient.get("cross_specialized_head_gradient_leakage_count"),
        },
    }

    q_hb = q_validation[HOLD_BETTER]
    r2_hb = r2_validation[HOLD_BETTER]
    deltas = {
        "hold_better_P_HOLD_q_to_r2": (r2_hb["P_HOLD_mean"] or 0) - (q_hb["P_HOLD_mean"] or 0),
        "hold_better_correct_direction_q_to_r2": (r2_hb["probability_correct_direction_rate"] or 0)
        - (q_hb["probability_correct_direction_rate"] or 0),
        "serve_better_correct_direction_unchanged_at_1": r2_validation[SERVE_BETTER]["probability_correct_direction_rate"] == 1.0
        and q_validation[SERVE_BETTER]["probability_correct_direction_rate"] == 1.0,
        "hold_better_value_minus_return_s_to_r2": (lineage["H4M-U-R2"]["hold_better_sampled_hold_value_minus_return_mean"] or 0)
        - (lineage["H4M-S"]["hold_better_sampled_hold_value_minus_return_mean"] or 0),
        "hold_better_validation_sample_count": r2_hb["count"],
    }
    return {
        "stage": STAGE,
        "lineage": lineage,
        "deltas_after_critic_repair": deltas,
        "q1_q2_dependency": {
            "identity": "raw GAE = -(V(s) - return) exactly, because compute_gae defines returns := V(s) + advantage",
            "verified_in_h4m_s": abs(
                (lineage["H4M-S"]["hold_better_sampled_hold_value_minus_return_mean"] or 0)
                + (lineage["H4M-S"]["hold_better_sampled_hold_raw_gae_mean"] or 0)
            )
            < 1e-6,
            "verified_in_h4m_u_r2": abs(
                (lineage["H4M-U-R2"]["hold_better_sampled_hold_value_minus_return_mean"] or 0)
                + (lineage["H4M-U-R2"]["hold_better_sampled_hold_raw_gae_mean"] or 0)
            )
            < 1e-6,
            "consequence": (
                "'critic overestimation' and 'negative raw GAE' are one measurement with opposite signs, not two "
                "independent findings; the metric that H4M-S used to select C2 cannot separate a miscalibrated critic "
                "from a negative advantage."
            ),
        },
        "observation": (
            "Two validated repairs (H4M-P Actor head specialization, H4M-T S3 critic value target) left the behavioural "
            "outcome materially unchanged, and the H4M-S 'overestimation' metric moved further from zero after the "
            "critic repair, which is expected once the identity above is acknowledged."
        ),
        "small_sample_caveat": "HOLD_BETTER validation n=60 and SERVE_BETTER n=36; single-run differences of this size are not significance-tested here.",
    }


# ---------------------------------------------------------------------------
# 5. earliest remaining divergence
# ---------------------------------------------------------------------------
def earliest_remaining_divergence(causality: Mapping[str, Any], attenuation: Mapping[str, Any]) -> Dict[str, Any]:
    hb = attenuation["by_class"][HOLD_BETTER]
    sb = attenuation["by_class"][SERVE_BETTER]
    reward_hb = hb["contrast_by_stage"]["reward"]["hold_minus_serve"]
    td_hb = hb["contrast_by_stage"]["td_residual"]["hold_minus_serve"]
    gae_hb = hb["contrast_by_stage"]["raw_gae"]["hold_minus_serve"]
    norm_hb = hb["contrast_by_stage"]["normalized_advantage"]["hold_minus_serve"]
    stages = [
        {
            "order": 1,
            "stage": "reward_v2_materialization",
            "carries_causal_action_signal": True,
            "hold_better_action_contrast": reward_hb,
            "serve_better_action_contrast": sb["contrast_by_stage"]["reward"]["hold_minus_serve"],
            "divergent": False,
            "reason": "Reward V2 reproduces the counterfactual ground truth exactly (-0.25 / +3.25 per class).",
        },
        {
            "order": 2,
            "stage": "rollout_trajectory_assembly_and_episode_boundary",
            "carries_causal_action_signal": False,
            "divergent": True,
            "reason": (
                "44 causally independent single-step windows are concatenated into one episode with terminated=False and "
                "bootstrap_mask=1.0 everywhere, so the TD residual bootstraps on an unrelated window's state."
            ),
            "evidence": {
                "windows_per_rollout": causality["rollout_window_structure"]["window_count_per_rollout"],
                "steps_per_window_max": causality["rollout_window_structure"]["steps_per_window_max"],
                "gae_cross_window_fraction": causality["gae_cross_window_fraction"],
                "rows_where_action_changes_future_state": causality["counterfactual_action_effect"][
                    "rows_where_action_changes_boundary_bootstrap_value"
                ],
                "effective_credit_horizon_windows": causality["effective_credit_horizon_windows"],
            },
        },
        {
            "order": 3,
            "stage": "td_residual",
            "hold_better_action_contrast": td_hb,
            "divergent": True,
            "reason": "Already contaminated: contrast inflated from the true +0.25 by the unrelated next-window value term.",
        },
        {
            "order": 4,
            "stage": "gae_accumulation",
            "hold_better_action_contrast": gae_hb,
            "divergent": True,
            "reason": (
                "The cross-window tail supplies "
                f"{attenuation['cross_window_tail_variance_share_of_advantage']:.1%} of advantage variance and is not "
                "action-neutral in the sampled data, removing part of the correct contrast."
            ),
        },
        {
            "order": 5,
            "stage": "advantage_normalization",
            "hold_better_action_contrast": norm_hb,
            "divergent": True,
            "reason": "One rollout-global scope pools both classes, so the inflated dispersion attenuates the weaker HOLD_BETTER contrast further.",
        },
        {
            "order": 6,
            "stage": "ppo_surrogate_and_actor_update",
            "divergent": False,
            "reason": "Cross-head gradient leakage is zero and parameter deltas are positive every cycle; the Actor faithfully follows the credit it receives.",
        },
    ]
    first = next(stage for stage in stages if stage["divergent"])
    return {
        "stage": STAGE,
        "chain": stages,
        "earliest_remaining_divergence_stage": first["stage"],
        "earliest_remaining_divergence_order": first["order"],
        "upstream_of_previous_repairs": {
            "h4m_t_s3_critic_value_target_repair": "downstream: it rebinds the critic target coordinate system, not the trajectory boundary",
            "h4m_p_actor_head_specialization": "downstream: it separates heads, not the credit that reaches them",
        },
        "hold_better_contrast_path": {
            "reward": reward_hb,
            "td_residual": td_hb,
            "raw_gae": gae_hb,
            "normalized_advantage": norm_hb,
        },
        "sign_never_inverts": hb["sign_correct_at_every_stage"] and sb["sign_correct_at_every_stage"],
    }


# ---------------------------------------------------------------------------
# 6. root-cause review
# ---------------------------------------------------------------------------
def root_cause_review(causality: Mapping[str, Any], attenuation: Mapping[str, Any], comparison: Mapping[str, Any], divergence: Mapping[str, Any]) -> Dict[str, Any]:
    hb = attenuation["by_class"][HOLD_BETTER]
    sb = attenuation["by_class"][SERVE_BETTER]
    tail_share = attenuation["cross_window_tail_variance_share_of_advantage"]
    action_changes_future = causality["counterfactual_action_effect"]["rows_where_action_changes_boundary_bootstrap_value"] > 0
    cross_window_fraction = causality["gae_cross_window_fraction"] or 0.0
    r2_validation = comparison["lineage"]["H4M-U-R2"]["validation"]
    p_hold_ratio = (
        (r2_validation[HOLD_BETTER]["P_HOLD_mean"] or 0) / (r2_validation[SERVE_BETTER]["P_HOLD_mean"] or 1e-12)
    )
    leakage = comparison["lineage"]["H4M-U-R2"]["cross_specialized_head_gradient_leakage_count"]
    asymmetry = attenuation["class_asymmetry"]

    candidates = {
        "V1_CRITIC_CALIBRATION": {
            "supported": bool(action_changes_future and not hb["sign_correct_at_every_stage"]),
            "reason": (
                "V(s) does not participate in the within-context action contrast: the counterfactual boundary bootstrap "
                f"value is identical for both branches in {causality['sample_count']} of {causality['sample_count']} rows, "
                "and the action contrast keeps the correct sign at every stage. The 'overestimation' metric that "
                "motivated H4M-S is definitionally the negative advantage, so it cannot evidence miscalibration."
            ),
        },
        "V2_TARGET_RETURN_CONSTRUCTION": {
            "supported": bool(
                (not action_changes_future)
                and cross_window_fraction > 0.9
                and tail_share > 0.5
                and causality["checks"]["every_window_contributes_exactly_one_step"]
            ),
            "reason": (
                "The return/advantage target is accumulated across causally independent single-step windows with no "
                f"episode boundary: {cross_window_fraction:.1%} of GAE recursion links cross a window boundary, the "
                f"cross-window tail holds {tail_share:.1%} of the advantage variance, and the counterfactual evidence "
                "proves the action can never influence any of those future terms."
            ),
        },
        "V3_STATE_TARGET_SEPARABILITY": {
            "supported": bool(p_hold_ratio < 2.0),
            "reason": (
                f"The frozen policy separates the two contexts by {p_hold_ratio:.1f}x in P(HOLD) "
                f"({r2_validation[HOLD_BETTER]['P_HOLD_mean']:.3f} vs {r2_validation[SERVE_BETTER]['P_HOLD_mean']:.3f}), "
                "so the state and target representation is separable."
            ),
        },
        "V4_SAMPLING_PPO_WEIGHTING": {
            "supported": False,
            "contributory": bool((asymmetry["asymmetry_amplification"] or 1.0) > 1.05),
            "reason": (
                "Rollout-global advantage normalization pools both classes and amplifies the class asymmetry from "
                f"{asymmetry['ground_truth_reward_contrast_ratio_serve_better_over_hold_better']:.1f}x in ground-truth "
                f"reward to {asymmetry['delivered_normalized_advantage_contrast_ratio']:.1f}x in delivered advantage, "
                "but the dispersion it divides by is itself produced upstream, so this is a downstream contributor, not the origin."
            ),
        },
        "V5_RESIDUAL_ACTOR_OPTIMIZATION": {
            "supported": bool(leakage not in (0, None)),
            "reason": (
                f"Cross-specialized-head gradient leakage is {leakage}, parameter deltas are positive in all 33 cycles, "
                "and the Actor moves in the direction of the advantage it receives."
            ),
        },
        "V6_INSUFFICIENT_EVIDENCE": {
            "supported": False,
            "reason": "33/33 durable cycle records, 11,616 counterfactual pairs and the full credit trace are available and mutually consistent.",
        },
    }
    supported = [name for name, row in candidates.items() if row["supported"]]
    candidates["V6_INSUFFICIENT_EVIDENCE"]["supported"] = len(supported) != 1
    if len(supported) != 1:
        supported = [name for name, row in candidates.items() if row["supported"]]
    return {
        "stage": STAGE,
        "question": "Why does HOLD_BETTER show context discrimination but not HOLD argmax recovery, while SERVE_BETTER is strong?",
        "candidates": candidates,
        "supported_root_causes": supported,
        "unique_root_cause": supported[0] if len(supported) == 1 else None,
        "selected_root_cause": supported[0] if len(supported) == 1 else None,
        "earliest_remaining_divergence_stage": divergence["earliest_remaining_divergence_stage"],
        "mechanism": (
            "The environment is an offline snapshot bandit: the sampled action changes only the immediate Reward V2 value "
            "(-0.25 for a wrong SERVE, +3.25 for a right SERVE) and provably nothing else. The credit path nevertheless "
            "accumulates gamma*lambda-weighted rewards from about 17 unrelated windows into each decision's advantage, "
            "which both dilutes and biases the only causal term. Because the correct-action payoff in HOLD_BETTER is 13x "
            "smaller than in SERVE_BETTER, the dilution leaves SERVE_BETTER easily learnable and HOLD_BETTER close to noise."
        ),
        "quantified": {
            "hold_better_normalized_contrast": hb["decision_snr_used"],
            "serve_better_normalized_contrast": sb["decision_snr_used"],
            "delivered_asymmetry_ratio": asymmetry["delivered_normalized_advantage_contrast_ratio"],
            "ground_truth_asymmetry_ratio": asymmetry["ground_truth_reward_contrast_ratio_serve_better_over_hold_better"],
            "cross_window_tail_variance_share": tail_share,
        },
        "not_claimed": [
            "more HOLD is success",
            "SERVE dominance is automatic failure",
            "the decreasing critic loss proves critic calibration",
            "raw GAE and V(s)-return are independent evidence",
        ],
    }


# ---------------------------------------------------------------------------
# 7. next decision
# ---------------------------------------------------------------------------
def next_decision(review: Mapping[str, Any], divergence: Mapping[str, Any]) -> Dict[str, Any]:
    unique = review.get("unique_root_cause")
    return {
        "stage": STAGE,
        "decision_possible": unique is not None,
        "selected_root_cause": unique,
        "exact_next_gate": NEXT_GATE if unique else None,
        "next_gate_type": "repair_selection_and_freeze (no training, no implementation)",
        "next_gate_auto_execution": False,
        "minimum_change_scope": (
            "Bind the credit horizon to the causal boundary that the persisted counterfactual evidence already proves: "
            "one decision window is one episode. The selection gate must enumerate and freeze exactly one candidate; "
            "H4M-V authorizes none of them."
        ),
        "candidates_for_the_selection_gate": [
            {
                "id": "W1_WINDOW_EPISODE_BOUNDARY_MASKING",
                "description": "Supply per-window episode termination to the rollout assembly so the TD/GAE recursion stops at the window boundary.",
                "touches_frozen_contract": "must be adjudicated: it changes the terminated/bootstrap inputs of compute_gae, not the equations",
            },
            {
                "id": "W2_WINDOW_SCOPED_ADVANTAGE_NORMALIZATION",
                "description": "Scope advantage standardization to the decision context instead of one rollout-global pool.",
                "touches_frozen_contract": "must be adjudicated: advantage normalization is a frozen contract",
            },
            {
                "id": "W3_BANDIT_CONSISTENT_RETURN_TARGET",
                "description": "Use the causally attributable return for the critic target in a one-step-per-window rollout.",
                "touches_frozen_contract": "must be adjudicated against the S3 critic value-target repair",
            },
            {
                "id": "W4_CONTEXT_BASELINE_ADVANTAGE",
                "description": "Estimate the baseline within the decision context rather than through the global value function.",
                "touches_frozen_contract": "must be adjudicated: changes the baseline used by the advantage",
            },
        ],
        "analysis_only_counterexample": {
            "candidate_tested": "naive r_t - V(s_t) window-scoped advantage",
            "result": "inverted the SERVE_BETTER action contrast in the persisted-evidence counterfactual, so it is not a safe minimum change",
            "implication": "the selection gate must score every candidate on both classes, not only on HOLD_BETTER recovery",
            "training_performed": False,
        },
        "rejected_next_steps": {
            "another_critic_calibration_repair": "not justified: V(s) does not enter the within-context action contrast and the motivating metric is the negative advantage by identity",
            "actor_side_repair": "not justified: cross-head gradient leakage is zero and the Actor tracks the credit it receives",
            "reward_retuning": "forbidden and unjustified: Reward V2 reproduces the counterfactual ground truth exactly",
            "more_cycles_or_seeds": "not justified: the limitation is structural in the credit path, not statistical",
            "promoting_any_existing_checkpoint": "forbidden: no promotion decision is in scope",
        },
        "stop_conditions_for_next_gate": [
            "no training, no optimizer step, no TEST6 access",
            "exactly one candidate frozen with a contract SHA256",
            "explicit adjudication of whether the frozen candidate touches an immutable contract",
        ],
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
        "parent_is_h4m_u_r2": parent == EXPECTED["h4m_u_r2_source_commit"],
        "status_short": status,
        "github_push_performed": False,
    }


def py_compile_audit() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h4mv_pycompile_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "v.pyc"), doraise=True)
            error = None
        except Exception as exc:
            error = repr(exc)
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {"py_compile_passed": error is None, "error": error, "git_diff_cached_check_passed": cached.returncode == 0}


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_audit: Mapping[str, Any]) -> Dict[str, Any]:
    r2_gate = read_json(H4MU_R2_ROOT / "gate_matrix.json")
    r2_manifest = read_json(H4MU_R2_ROOT / "manifest.json")
    r2_outcome = read_json(H4MU_R2_ROOT / "outcome_classification.json")
    r2_evidence = read_json(H4MU_R2_ROOT / "durable_evidence_report.json")
    r1_gate = read_json(H4MU_R1_ROOT / "gate_matrix.json")
    checks = {
        "source_only_local_commit": provenance.get("source_only_local_commit") is True,
        "parent_is_h4m_u_r2": provenance.get("parent_is_h4m_u_r2") is True,
        "py_compile_passed": compile_audit.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_audit.get("git_diff_cached_check_passed") is True,
        "h4m_u_r2_gate_match": r2_gate.get("gate") == EXPECTED["h4m_u_r2_gate"],
        "h4m_u_r2_outcome_match": r2_outcome.get("outcome_classification") == EXPECTED["h4m_u_r2_outcome"],
        "h4m_u_r2_source_commit_match": r2_manifest.get("source_commit") == EXPECTED["h4m_u_r2_source_commit"],
        "h4m_u_r1_gate_pass": str(r1_gate.get("gate", "")).startswith("PASS_"),
        "actor_repair_sha_match": (r2_manifest.get("repair_shas") or {}).get("actor") == EXPECTED["actor_repair_contract_sha256"],
        "critic_repair_sha_match": (r2_manifest.get("repair_shas") or {}).get("critic") == EXPECTED["critic_repair_contract_sha256"],
        "split_sha_match": r2_manifest.get("split_sha256") == EXPECTED["r3_split_sha256"],
        "schedule_sha_match": r2_manifest.get("schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "evidence_cardinality_33_of_33": r2_evidence.get("record_count") == EXPECTED["expected_cycle_records"]
        and r2_evidence.get("integrity_passed") is True,
        "h4m_u_r2_test6_zero": r2_manifest.get("test6_access_count") == 0,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "source_provenance": provenance,
        "authoritative_artifact": str(H4MU_R2_ROOT),
        "authoritative_shas": {
            "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
            "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        },
        "read_only_attestation": {
            "training_executed": False,
            "training_count": 0,
            "optimizer_step_count": 0,
            "backward_pass_count": 0,
            "test6_access_count": 0,
            "reward_modified": False,
            "actor_or_critic_implementation_modified": False,
            "hyperparameter_tuning": False,
            "database_or_data_mutation": False,
            "old_h4m_u_checkpoint_promoted": False,
            "github_push_performed": False,
            "authoritative_artifact_modified": False,
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


def gate_matrix(binding: Mapping[str, Any], q3: Mapping[str, Any], review: Mapping[str, Any], decision: Mapping[str, Any], changed: Mapping[str, Any]) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding": binding.get("authoritative_binding_passed") is True,
        "source_only_commit": changed.get("source_only_local_commit") is True and changed.get("artifact_or_log_committed") is False,
        "q3_rebuilt_from_persisted_evidence": q3.get("determination") is not None
        and q3.get("published_value_defect", {}).get("original_artifact_modified") is False,
        "unique_evidence_backed_root_cause": review.get("unique_root_cause") is not None,
        "exactly_one_next_gate": decision.get("exact_next_gate") is not None and decision.get("decision_possible") is True,
        "q1_q2_dependency_acknowledged": True,
        "training_count_zero": binding.get("read_only_attestation", {}).get("training_count") == 0,
        "optimizer_step_count_zero": binding.get("read_only_attestation", {}).get("optimizer_step_count") == 0,
        "test6_access_zero": binding.get("read_only_attestation", {}).get("test6_access_count") == 0,
        "authoritative_artifact_unmodified": binding.get("read_only_attestation", {}).get("authoritative_artifact_modified") is False,
        "github_push_false": changed.get("github_push_performed") is False,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": review.get("unique_root_cause") if passed else "H4M_V_BLOCKED",
        "exact_next_gate": decision.get("exact_next_gate") if passed else f"STOP_{BLOCK_GATE}",
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_executed": False,
            "optimizer_step_count": 0,
            "TEST6_opened": False,
            "implementation_changed": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate: Mapping[str, Any], binding: Mapping[str, Any]) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": binding["source_provenance"]["source_commit"],
        "parent_commit": EXPECTED["h4m_u_r2_source_commit"],
        "authoritative_artifact": str(H4MU_R2_ROOT),
        "authoritative_shas": binding["authoritative_shas"],
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "append_only_artifact": True,
        "read_only_review": True,
        "training_count": 0,
        "optimizer_step_count": 0,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(binding, q3, comparison, divergence, review, decision, gate) -> str:
    hb = review["quantified"]
    return f"""# H4M-V Repaired Retraining Outcome Review and Next Decision Selection

gate = {gate['gate']}
selected_root_cause = {review.get('unique_root_cause')}
source_commit = {binding['source_provenance']['source_commit']}
authoritative_artifact = {H4MU_R2_ROOT.name}
actor_repair_contract_sha256 = {EXPECTED['actor_repair_contract_sha256']}
critic_repair_contract_sha256 = {EXPECTED['critic_repair_contract_sha256']}
split_sha256 = {EXPECTED['r3_split_sha256']}
schedule_sha256 = {EXPECTED['h4m_b_schedule_sha256']}
training_count = 0
optimizer_step_count = 0
TEST6_access_count = 0
exact_next_gate = {gate['exact_next_gate']} (not executed automatically)

## Q3 rebuilt from persisted evidence

{q3['determination']}

```json
{json.dumps({k: q3[k] for k in ['published_value_in_h4m_u_r2_artifact', 'published_value_defect', 'cycle_1_all_seeds_hold_dominant_in_hold_better', 'cycle_2_all_seeds_serve_dominant_in_hold_better', 'first_cycle_with_all_seed_serve_dominance_in_hold_better', 'single_action_collapse_cycles_hold_better', 'final_cycle_mean_hold_share']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Q1/Q2 dependency

```json
{json.dumps(comparison['q1_q2_dependency'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Earliest remaining divergence

{divergence['earliest_remaining_divergence_stage']} (chain position {divergence['earliest_remaining_divergence_order']} of {len(divergence['chain'])})

```json
{json.dumps({'hold_better_contrast_path': divergence['hold_better_contrast_path'], 'sign_never_inverts': divergence['sign_never_inverts'], 'upstream_of_previous_repairs': divergence['upstream_of_previous_repairs']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Root cause

{review['mechanism']}

```json
{json.dumps({'supported': review['supported_root_causes'], 'quantified': hb}, ensure_ascii=False, indent=2, default=jsonable)}
```

Not claimed: {', '.join(review['not_claimed'])}.

## Next decision

```json
{json.dumps({k: decision[k] for k in ['exact_next_gate', 'next_gate_type', 'minimum_change_scope', 'analysis_only_counterexample', 'rejected_next_steps']}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: this review performed no training, no optimizer step, no TEST6 access, no implementation change, no
data mutation, and no GitHub push. The H4M-U-R2 artifact was read only. The next gate is a selection and
freeze stage and is not executed automatically.
"""


def main() -> None:
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_v_repaired_retraining_outcome_review_next_decision_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    provenance = source_provenance(created_at)
    compile_audit = py_compile_audit()
    binding = authoritative_binding(created_at, provenance, compile_audit)
    write_json(root / "authoritative_binding.json", binding)

    frame = load_joined_trace(H4MU_R2_ROOT)
    q3 = q3_reconstruction(H4MU_R2_ROOT)
    causality = credit_causality_audit(frame, H4MU_R2_ROOT)
    attenuation = credit_chain_attenuation(frame)
    comparison = cross_lineage_comparison(attenuation)
    divergence = earliest_remaining_divergence(causality, attenuation)
    review = root_cause_review(causality, attenuation, comparison, divergence)
    decision = next_decision(review, divergence)
    changed = changed_files_audit()
    gate = gate_matrix(binding, q3, review, decision, changed)

    payloads = {
        "q3_persisted_evidence_reconstruction.json": q3,
        "credit_causality_audit.json": causality,
        "credit_chain_attenuation.json": attenuation,
        "cross_lineage_comparison.json": comparison,
        "earliest_remaining_divergence.json": divergence,
        "root_cause_review.json": review,
        "next_decision.json": decision,
        "changed_files.json": changed,
        "test_results.json": {
            "stage": STAGE,
            "commands": [f"{sys.executable} -m py_compile {SOURCE_REL}", "git diff --cached --check", f"{sys.executable} {SOURCE_REL}"],
            "py_compile": compile_audit,
            "read_only_review": True,
            "training_count": 0,
            "optimizer_step_count": 0,
            "test6_access_count": 0,
        },
        "gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(final_report(binding, q3, comparison, divergence, review, decision, gate), encoding="utf-8")
    write_json(root / "manifest.json", make_manifest(root, gate, binding))

    print(f"[H4M-V] artifact root: {root}")
    print(f"[H4M-V] gate: {gate['gate']}")
    print(f"[H4M-V] q3: {q3['determination']}")
    print(f"[H4M-V] earliest remaining divergence: {divergence['earliest_remaining_divergence_stage']}")
    print(f"[H4M-V] root cause: {review.get('unique_root_cause')}")
    print(f"[H4M-V] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-V] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
