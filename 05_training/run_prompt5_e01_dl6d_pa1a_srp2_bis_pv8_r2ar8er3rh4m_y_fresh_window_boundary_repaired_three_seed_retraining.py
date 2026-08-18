#!/usr/bin/env python3
"""H4M-Y fresh window-boundary-repaired three-seed retraining with durable evidence.

Fresh retraining for seeds 1, 2, 3 on the frozen repaired execution path with
the H4M-W W1 window episode-boundary repair implemented in H4M-X, recorded
through the H4M-U-R1 durable evidence writer.

The H4M-U-R1 and H4M-U-R2 sources are reused as-is: this runner extends them by
subclassing and wrapping only.  No tuning, no old checkpoint reuse, no TEST6
access, no extra cycles or updates, no W2/W3/W4, and no GitHub push.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-Y"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Y_"
    "FRESH_WINDOW_EPISODE_BOUNDARY_REPAIRED_THREE_SEED_RETRAINING_WITH_DURABLE_EVIDENCE_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Y"
NEXT_REVIEW_GATE = "H4M-Z_WINDOW_BOUNDARY_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
REPORT_TITLE = "H4M-Y Fresh Window-Boundary-Repaired Three-Seed Retraining with Durable Evidence"
CHECKPOINT_NAME = "H4M_Y_SEED_{seed:03d}_WINDOW_EPISODE_BOUNDARY_REPAIRED.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MU_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py"
H4MU_R2_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_r2_fresh_three_seed_retraining_with_durable_evidence.py"
H4MQ_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"

H4MX_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_x_window_boundary_repair_implementation_equivalence_validation_20260818_133041+09:00"
H4MW_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_w_window_episode_boundary_credit_repair_selection_freeze_20260818_131002+09:00"
H4MU_R2_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_u_r2_fresh_three_seed_retraining_with_durable_evidence_20260818_001408+09:00"
H4MQ_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_20260817_153440+0900"

EXPECTED = {
    "h4m_x_source_commit": "418330767b2a3d0e3030009744a2814b2889fb9b",
    "h4m_x_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_X_"
        "WINDOW_EPISODE_BOUNDARY_CREDIT_HORIZON_REPAIR_IMPLEMENTATION_AND_NO_TRAINING_EQUIVALENCE_VALIDATION_COMPLETE"
    ),
    "w1_repair_id": "W1_WINDOW_EPISODE_BOUNDARY_MASKING",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "boundary_schema": "independent_window_causal_horizon_termination_v1",
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
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
DOMINANT_LABEL = {HOLD: "HOLD", SERVE: "SERVE"}

W1_REQUIRED_EVIDENCE_FIELDS = [
    "terminated_true_count",
    "bootstrap_mask_zero_count",
    "cross_window_leakage_count",
    "window_count",
    "w1_repair_contract_sha256",
    "boundary_schema",
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
    "context_credit_diagnostics.json",
    "window_boundary_diagnostics.json",
    "checkpoint_integrity.json",
    "durable_evidence_report.json",
    "primary_questions.json",
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
]
MUTABLE_STATE_FILES = ["07_durable_training_evidence/run_state.json"]


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
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fnum(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


# ---------------------------------------------------------------------------
# W1 per-cycle evidence
# ---------------------------------------------------------------------------
def build_w1_boundary_evidence(recorder: Any, h4mg: Any) -> Dict[str, Any]:
    """Boundary and leakage evidence measured from this cycle's own rows."""
    window_by_uid = {row["sample_uid"]: row["window_id"] for row in recorder.pre_action_rows}
    rows = recorder.critic_td_gae_rows
    terminated_true = sum(1 for row in rows if bool(row.get("terminated")))
    bootstrap_zero = sum(1 for row in rows if float(row.get("bootstrap_mask", 1.0)) == 0.0)
    leakage = 0
    linked = 0
    for row in rows:
        successor = row.get("gae_recursion_successor_uid")
        if successor is None or successor not in window_by_uid:
            continue
        linked += 1
        if window_by_uid.get(row["sample_uid"]) != window_by_uid.get(successor):
            # a boundary link only leaks credit when the bootstrap survived it
            if float(row.get("bootstrap_mask", 1.0)) != 0.0:
                leakage += 1
    return {
        "terminated_true_count": terminated_true,
        "bootstrap_mask_zero_count": bootstrap_zero,
        "cross_window_leakage_count": leakage,
        "cross_window_linked_count": linked,
        "row_count": len(rows),
        "window_count": len(set(window_by_uid.values())),
        "w1_repair_contract_sha256": getattr(h4mg, "W1_REPAIR_CONTRACT_SHA256", None),
        "boundary_schema": getattr(h4mg, "WINDOW_EPISODE_BOUNDARY_SCHEMA", None),
    }


def make_y_binder_class(umod: Any, r2mod: Any, h4mg: Any) -> Any:
    base = r2mod.make_binder_class(umod)

    class H4MYCycleEvidenceBinder(base):  # type: ignore[misc, valid-type]
        """R2 binder plus the W1 boundary and leakage evidence."""

        def flush_cycle(self, *, seed: int, cycle: int, recorder: Any, trace_entry: Mapping[str, Any]) -> Dict[str, Any]:
            key = (int(seed), int(cycle))
            staged = self.pending.get(key)
            if staged is None:
                raise self.dte.EvidenceIntegrityError(
                    "CYCLE_UPDATE_EVIDENCE_NOT_STAGED",
                    f"key={key} cycle trace was written without a matching PPO update result",
                )
            boundary = build_w1_boundary_evidence(recorder, h4mg)
            if boundary["cross_window_leakage_count"] != 0:
                raise self.dte.EvidenceIntegrityError(
                    "CROSS_WINDOW_CREDIT_LEAKAGE_DETECTED",
                    f"key={key} leakage={boundary['cross_window_leakage_count']}",
                )
            if boundary["w1_repair_contract_sha256"] != EXPECTED["w1_repair_contract_sha256"]:
                raise self.dte.EvidenceIntegrityError("W1_REPAIR_SHA_MISMATCH", f"key={key}")
            staged["cycle_context"]["w1_boundary_evidence"] = boundary
            return super().flush_cycle(seed=seed, cycle=cycle, recorder=recorder, trace_entry=trace_entry)

    return H4MYCycleEvidenceBinder


def w1_evidence_completeness(dte: Any, evidence_dir: Path) -> Dict[str, Any]:
    loaded = dte.load_evidence(evidence_dir)
    missing: List[str] = []
    leakage_total = 0
    terminated_total = 0
    bootstrap_zero_total = 0
    rows_total = 0
    sha_mismatch = 0
    for record in loaded["records"]:
        key = f"seed={record.get('seed')}/cycle={record.get('outer_cycle')}"
        block = (record.get("cycle_context") or {}).get("w1_boundary_evidence") or {}
        for field in W1_REQUIRED_EVIDENCE_FIELDS:
            if field not in block:
                missing.append(f"{key}:{field}")
        leakage_total += int(block.get("cross_window_leakage_count", 1))
        terminated_total += int(block.get("terminated_true_count", 0))
        bootstrap_zero_total += int(block.get("bootstrap_mask_zero_count", 0))
        rows_total += int(block.get("row_count", 0))
        if block.get("w1_repair_contract_sha256") != EXPECTED["w1_repair_contract_sha256"]:
            sha_mismatch += 1
    checks = {
        "all_records_carry_w1_evidence": not missing,
        "cross_window_leakage_zero": leakage_total == 0,
        "every_sample_terminated_at_its_window": terminated_total == rows_total and rows_total > 0,
        "every_sample_bootstrap_masked": bootstrap_zero_total == rows_total and rows_total > 0,
        "w1_sha_bound_in_every_record": sha_mismatch == 0,
    }
    return {
        "stage": STAGE,
        "required_fields": W1_REQUIRED_EVIDENCE_FIELDS,
        "missing_fields": missing,
        "cross_window_leakage_total": leakage_total,
        "terminated_true_total": terminated_total,
        "bootstrap_mask_zero_total": bootstrap_zero_total,
        "sample_row_total": rows_total,
        "w1_sha_mismatch_records": sha_mismatch,
        "checks": checks,
        "passed": all(checks.values()),
    }


def window_boundary_diagnostics(root: Path) -> Dict[str, Any]:
    """Independent boundary audit straight from the persisted parquet traces."""
    rows: List[Dict[str, Any]] = []
    for td_path in sorted(glob.glob(str(root / "05_actual_on_policy_credit_trace" / "seed=*" / "outer_cycle=*" / "actual_critic_td_gae_trace.parquet"))):
        base = Path(td_path).parent
        td = pd.read_parquet(td_path)
        pre = pd.read_parquet(base / "actual_pre_action_trace.parquet")[["sample_uid", "window_id"]]
        merged = td.merge(pre, on="sample_uid")
        window_by_uid = dict(zip(merged["sample_uid"], merged["window_id"]))
        merged["successor_window_id"] = merged["gae_recursion_successor_uid"].map(window_by_uid)
        linked = merged[merged["successor_window_id"].notna()]
        cross = linked[linked["window_id"] != linked["successor_window_id"]]
        leaking = cross[cross["bootstrap_mask"] != 0.0]
        rows.append(
            {
                "seed": int(merged["seed"].iloc[0]),
                "outer_cycle": int(merged["outer_cycle"].iloc[0]),
                "samples": int(len(merged)),
                "terminated_true": int(merged["terminated"].sum()),
                "truncated_true": int(merged["truncated"].sum()),
                "bootstrap_mask_zero": int((merged["bootstrap_mask"] == 0.0).sum()),
                "cross_window_links": int(len(cross)),
                "cross_window_links_with_live_bootstrap": int(len(leaking)),
                "next_value_used_count": int((merged["bootstrap_mask"] != 0.0).sum()),
                "windows": int(merged["window_id"].nunique()),
                "td_delta_equals_reward_minus_value": bool(
                    ((merged["td_delta"] - (merged["raw_return_target"] - merged["value_t"])).abs() < 1e-5).all()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    checks = {
        "all_33_cycles_audited": len(rows) == EXPECTED["expected_cycle_records"],
        "every_sample_terminated": bool((frame["terminated_true"] == frame["samples"]).all()),
        "every_sample_bootstrap_masked": bool((frame["bootstrap_mask_zero"] == frame["samples"]).all()),
        "no_truncation_claimed": bool((frame["truncated_true"] == 0).all()),
        "cross_window_leakage_zero": bool((frame["cross_window_links_with_live_bootstrap"] == 0).all()),
        "next_window_value_never_used": bool((frame["next_value_used_count"] == 0).all()),
    }
    return {
        "stage": STAGE,
        "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
        "boundary_schema": EXPECTED["boundary_schema"],
        "by_seed_cycle": rows,
        "totals": {
            "samples": int(frame["samples"].sum()),
            "terminated_true": int(frame["terminated_true"].sum()),
            "bootstrap_mask_zero": int(frame["bootstrap_mask_zero"].sum()),
            "cross_window_links": int(frame["cross_window_links"].sum()),
            "cross_window_links_with_live_bootstrap": int(frame["cross_window_links_with_live_bootstrap"].sum()),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# primary questions
# ---------------------------------------------------------------------------
def cycle_dominance(conditional: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per-cycle dominance from the conditional summary.

    The summary is passed in rather than read back from the artifact root: it is
    only written at the end of main(), so reading it here would make reporting
    depend on a file that does not exist yet.
    """
    rows = conditional.get("by_seed_cycle_label", [])
    out: List[Dict[str, Any]] = []
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
                        "hold_share": (hold / total) if total else None,
                        "P_HOLD_mean": fnum((summary.get("P_HOLD") or {}).get("mean")),
                        "sample_count": total,
                    }
                )
            shares = [row["hold_share"] for row in per_seed if row["hold_share"] is not None]
            entry["by_class"][label] = {
                "per_seed": per_seed,
                "mean_hold_share": (sum(shares) / len(shares)) if shares else None,
                "all_seeds_serve_dominant": bool(per_seed) and all(r["sampled_dominant_action"] == DOMINANT_LABEL[SERVE] for r in per_seed),
                "all_seeds_hold_dominant": bool(per_seed) and all(r["sampled_dominant_action"] == DOMINANT_LABEL[HOLD] for r in per_seed),
            }
        out.append(entry)
    return out


def validation_summary(path: Path) -> Dict[str, Any]:
    aggregate = read_json(path).get("aggregate_by_classification", {})
    return {
        label: {
            "count": aggregate.get(label, {}).get("count"),
            "P_HOLD_mean": fnum((aggregate.get(label, {}).get("P_HOLD") or {}).get("mean")),
            "P_SERVE_mean": fnum((aggregate.get(label, {}).get("P_SERVE") or {}).get("mean")),
            "probability_dominant_action": aggregate.get(label, {}).get("probability_dominant_action"),
            "probability_correct_direction_rate": aggregate.get(label, {}).get("probability_correct_direction_rate"),
            "sampled_correct_direction_rate": aggregate.get(label, {}).get("sampled_correct_direction_rate"),
        }
        for label in (HOLD_BETTER, SERVE_BETTER)
    }


def primary_questions(conditional: Mapping[str, Any], boundary: Mapping[str, Any], w1_evidence: Mapping[str, Any], credit: Mapping[str, Any], validation: Mapping[str, Any], gradient: Mapping[str, Any]) -> Dict[str, Any]:
    dominance = cycle_dominance(conditional)
    baseline = validation_summary(H4MU_R2_ROOT / "validation_discrimination.json")
    current = {
        label: {
            "count": validation["aggregate_by_classification"][label].get("count"),
            "P_HOLD_mean": fnum((validation["aggregate_by_classification"][label].get("P_HOLD") or {}).get("mean")),
            "P_SERVE_mean": fnum((validation["aggregate_by_classification"][label].get("P_SERVE") or {}).get("mean")),
            "probability_dominant_action": validation["aggregate_by_classification"][label].get("probability_dominant_action"),
            "probability_correct_direction_rate": validation["aggregate_by_classification"][label].get("probability_correct_direction_rate"),
            "sampled_correct_direction_rate": validation["aggregate_by_classification"][label].get("sampled_correct_direction_rate"),
        }
        for label in (HOLD_BETTER, SERVE_BETTER)
    }
    cycle1 = dominance[0]["by_class"][HOLD_BETTER]
    cycle2 = dominance[1]["by_class"][HOLD_BETTER]
    hb_hold = credit["by_context"][HOLD_BETTER]["by_sampled_action"][HOLD]
    sb_serve = credit["by_context"][SERVE_BETTER]["by_sampled_action"][SERVE]
    hb_improvement = (current[HOLD_BETTER]["probability_correct_direction_rate"] or 0.0) - (
        baseline[HOLD_BETTER]["probability_correct_direction_rate"] or 0.0
    )
    return {
        "stage": STAGE,
        "baseline_reference": {"artifact": H4MU_R2_ROOT.name, "validation": baseline},
        "current_validation": current,
        "Q1_cross_window_leakage_zero": {
            "question": "Does cross-window credit leakage remain zero?",
            "per_cycle_evidence_leakage_total": w1_evidence["cross_window_leakage_total"],
            "persisted_trace_leakage_total": boundary["totals"]["cross_window_links_with_live_bootstrap"],
            "next_window_value_never_used": boundary["checks"]["next_window_value_never_used"],
            "answer": bool(w1_evidence["cross_window_leakage_total"] == 0 and boundary["totals"]["cross_window_links_with_live_bootstrap"] == 0),
        },
        "Q2_cycle_2_serve_transition": {
            "question": "Does the cycle-2 SERVE transition recur?",
            "cycle_1_all_seeds_hold_dominant": cycle1["all_seeds_hold_dominant"],
            "cycle_2_all_seeds_serve_dominant": cycle2["all_seeds_serve_dominant"],
            "cycle_1_mean_hold_share": cycle1["mean_hold_share"],
            "cycle_2_mean_hold_share": cycle2["mean_hold_share"],
            "answer": bool(cycle1["all_seeds_hold_dominant"] and cycle2["all_seeds_serve_dominant"]),
        },
        "Q3_hold_better_correct_direction": {
            "question": "Does the HOLD_BETTER correct-direction rate improve?",
            "baseline_rate": baseline[HOLD_BETTER]["probability_correct_direction_rate"],
            "current_rate": current[HOLD_BETTER]["probability_correct_direction_rate"],
            "delta": hb_improvement,
            "sample_count": current[HOLD_BETTER]["count"],
            "answer": bool(hb_improvement > 0.0),
        },
        "Q4_serve_better_discrimination": {
            "question": "Does SERVE_BETTER discrimination remain intact?",
            "baseline_rate": baseline[SERVE_BETTER]["probability_correct_direction_rate"],
            "current_rate": current[SERVE_BETTER]["probability_correct_direction_rate"],
            "current_P_SERVE_mean": current[SERVE_BETTER]["P_SERVE_mean"],
            "answer": bool(
                (current[SERVE_BETTER]["probability_correct_direction_rate"] or 0.0)
                >= (baseline[SERVE_BETTER]["probability_correct_direction_rate"] or 0.0) - 1e-12
            ),
        },
        "Q5_three_seed_context_recovery": {
            "question": "Do the three seeds recover context-dependent HOLD/SERVE, or does SERVE dominance persist?",
            "hold_better_probability_dominant": current[HOLD_BETTER]["probability_dominant_action"],
            "serve_better_probability_dominant": current[SERVE_BETTER]["probability_dominant_action"],
            "final_cycle_mean_hold_share": {
                HOLD_BETTER: dominance[-1]["by_class"][HOLD_BETTER]["mean_hold_share"],
                SERVE_BETTER: dominance[-1]["by_class"][SERVE_BETTER]["mean_hold_share"],
            },
            "context_separation_ratio_P_HOLD": (
                (current[HOLD_BETTER]["P_HOLD_mean"] or 0.0) / (current[SERVE_BETTER]["P_HOLD_mean"] or 1e-12)
            ),
            "cross_specialized_head_gradient_leakage_count": gradient.get("cross_specialized_head_gradient_leakage_count"),
            "answer": (
                "CONTEXT_DEPENDENT_DOMINANCE_RECOVERED"
                if current[HOLD_BETTER]["probability_dominant_action"] == HOLD and current[SERVE_BETTER]["probability_dominant_action"] == SERVE
                else "SERVE_DOMINANCE_PERSISTS"
                if current[HOLD_BETTER]["probability_dominant_action"] == SERVE and current[SERVE_BETTER]["probability_dominant_action"] == SERVE
                else "MIXED"
            ),
        },
        "credit_statistics": {
            "hold_better_sampled_hold": {
                "V_s_minus_return_mean": hb_hold["V_s_minus_return"]["mean"],
                "raw_GAE_mean": hb_hold["raw_GAE"]["mean"],
                "normalized_advantage_mean": hb_hold["normalized_advantage"]["mean"],
                "sample_count": hb_hold["sample_count"],
            },
            "serve_better_sampled_serve": {
                "V_s_minus_return_mean": sb_serve["V_s_minus_return"]["mean"],
                "raw_GAE_mean": sb_serve["raw_GAE"]["mean"],
                "normalized_advantage_mean": sb_serve["normalized_advantage"]["mean"],
                "sample_count": sb_serve["sample_count"],
            },
        },
        "cycle_dominance": dominance,
        "interpretation_rule": "PASS is not defined as a higher HOLD ratio; HOLD_BETTER credit/selection should be supported while SERVE_BETTER preference stays supported.",
    }


def outcome_classification(questions: Mapping[str, Any], integrity_ok: bool) -> Dict[str, Any]:
    hold_dom = questions["current_validation"][HOLD_BETTER]["probability_dominant_action"]
    serve_dom = questions["current_validation"][SERVE_BETTER]["probability_dominant_action"]
    improved = questions["Q3_hold_better_correct_direction"]["answer"]
    serve_intact = questions["Q4_serve_better_discrimination"]["answer"]
    if not integrity_ok:
        outcome = "OTHER_EVIDENCE_BASED_CLASSIFICATION"
    elif hold_dom == HOLD and serve_dom == SERVE:
        outcome = "WINDOW_BOUNDARY_REPAIR_RESTORED_TARGET_CONTEXT_DISCRIMINATION"
    elif improved and serve_intact:
        outcome = "PARTIAL_TARGET_CONTEXT_RECOVERY"
    elif hold_dom == SERVE and serve_dom == SERVE:
        outcome = "SERVE_DOMINANCE_PERSISTS_AFTER_WINDOW_BOUNDARY_REPAIR"
    else:
        outcome = "OTHER_EVIDENCE_BASED_CLASSIFICATION"
    return {
        "stage": STAGE,
        "outcome_classification": outcome,
        "exact_next_review_gate": NEXT_REVIEW_GATE,
        "evidence": {
            "hold_better_probability_dominant": hold_dom,
            "serve_better_probability_dominant": serve_dom,
            "hold_better_correct_direction_improved": improved,
            "hold_better_correct_direction_delta": questions["Q3_hold_better_correct_direction"]["delta"],
            "serve_better_discrimination_intact": serve_intact,
            "cross_window_leakage_zero": questions["Q1_cross_window_leakage_zero"]["answer"],
            "cycle_2_serve_transition_recurs": questions["Q2_cycle_2_serve_transition"]["answer"],
        },
        "classification_rule": (
            "restored requires both contexts to be dominant in their own direction; partial requires a strictly improved "
            "HOLD_BETTER correct-direction rate with SERVE_BETTER intact; otherwise persistence or other"
        ),
        "interpretation_rule": "diagnostic training evidence only, not a promotion or performance claim",
    }


# ---------------------------------------------------------------------------
# binding, gate, manifest
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
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "parent_is_h4m_x": parent == EXPECTED["h4m_x_source_commit"],
        "status_short": status,
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], h4mg: Any) -> Dict[str, Any]:
    x_gate = read_json(H4MX_ROOT / "gate_matrix.json")
    x_binding = read_json(H4MX_ROOT / "repair_binding.json")
    w_freeze = read_json(H4MW_ROOT / "repair_freeze_contract.json")
    q_sha = read_json(H4MQ_ROOT / "repair_binding.json").get("sha_bindings", {})
    r2_manifest = read_json(H4MU_R2_ROOT / "manifest.json")
    checks = {
        "source_only_commit_before_training": provenance.get("source_only_local_commit") is True,
        "source_parent_is_h4m_x": provenance.get("parent_is_h4m_x") is True,
        "h4m_x_gate_match": x_gate.get("gate") == EXPECTED["h4m_x_gate"],
        "h4m_x_integrity": x_gate.get("final_flags", {}).get("training_count") == 0
        and x_gate.get("final_flags", {}).get("optimizer_step_count") == 0
        and x_gate.get("final_flags", {}).get("TEST6_opened") is False,
        "w1_contract_sha_match": w_freeze.get("contract_sha256") == EXPECTED["w1_repair_contract_sha256"]
        and x_binding.get("w1_repair_contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "w1_bound_in_execution_source": getattr(h4mg, "W1_REPAIR_CONTRACT_SHA256", None) == EXPECTED["w1_repair_contract_sha256"]
        and getattr(h4mg, "W1_REPAIR_ID", None) == EXPECTED["w1_repair_id"],
        "boundary_schema_bound": getattr(h4mg, "WINDOW_EPISODE_BOUNDARY_SCHEMA", None) == EXPECTED["boundary_schema"],
        "boundary_builder_present": callable(getattr(h4mg, "window_episode_boundary_mask", None)),
        "s3_binding_retained": getattr(h4mg, "CRITIC_VALUE_TARGET_BINDING_SCHEMA", None) == "rollout_pre_update_return_normalizer_state_v1",
        "actor_repair_sha_match": q_sha.get("h4m_p_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "critic_repair_sha_match": (r2_manifest.get("repair_shas") or {}).get("critic") == EXPECTED["critic_repair_contract_sha256"],
        "reward_v2_sha_match": q_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "split_sha_match": q_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "schedule_sha_match": q_sha.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "zero_loss_sha_match": q_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "source_provenance": provenance,
        "repair_shas": {
            "w1_window_episode_boundary": EXPECTED["w1_repair_contract_sha256"],
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
            "w2_w3_w4_applied": False,
            "old_checkpoint_reuse": False,
        },
        "test6_access_count": 0,
    }


def gate_matrix(binding, integrity, outcome, changed, evidence, r2_evidence, w1_evidence, boundary, checkpoints, training) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding": binding.get("authoritative_binding_passed") is True,
        "source_only_commit": changed.get("source_only_local_commit") is True and changed.get("artifact_or_log_committed") is False,
        "three_seeds_complete": training.get("seeds") == EXPECTED["seeds"] and training.get("total_rollouts") == 33,
        "cycle_evidence_cardinality_33_of_33": evidence.get("record_count") == EXPECTED["expected_cycle_records"] and not evidence.get("missing_cycles"),
        "durable_evidence_integrity": evidence.get("integrity_passed") is True,
        "r2_required_evidence_complete": r2_evidence.get("passed") is True,
        "w1_boundary_evidence_complete": w1_evidence.get("passed") is True,
        "cross_window_leakage_zero": w1_evidence.get("cross_window_leakage_total") == 0
        and boundary.get("totals", {}).get("cross_window_links_with_live_bootstrap") == 0,
        "window_boundary_diagnostics_pass": boundary.get("passed") is True,
        "training_integrity": integrity.get("training_integrity_passed") is True,
        "checkpoint_integrity": checkpoints.get("checkpoint_manifest_passed") is True,
        "old_checkpoint_reuse_false": training.get("fresh_initialization_all_seeds") is True,
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
        "decision": outcome.get("outcome_classification") if passed else "H4M_Y_BLOCKED",
        "exact_next_gate": NEXT_REVIEW_GATE if passed else f"STOP_{BLOCK_GATE_PREFIX}",
        "next_gate_is_read_only_review": True,
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "repair_shas": binding.get("repair_shas"),
        "frozen_sha_bindings": binding.get("frozen_sha_bindings"),
        "final_flags": {
            "TEST6_opened": False,
            "test6_access_count": 0,
            "github_push_performed": False,
            "tuning_applied": False,
            "w2_w3_w4_applied": False,
            "old_checkpoint_reused": False,
            "additional_cycles_or_updates": False,
        },
    }


def make_manifest(root: Path, gate, evidence, checkpoints, boundary, started) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    immutable = {name: path for name, path in files.items() if name not in MUTABLE_STATE_FILES}
    mutable = {name: path for name, path in files.items() if name in MUTABLE_STATE_FILES}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": git_run(["rev-parse", "HEAD"]).stdout.strip(),
        "parent_commit": EXPECTED["h4m_x_source_commit"],
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "repair_shas": {
            "w1_window_episode_boundary": EXPECTED["w1_repair_contract_sha256"],
            "actor": EXPECTED["actor_repair_contract_sha256"],
            "critic": EXPECTED["critic_repair_contract_sha256"],
        },
        "split_sha256": EXPECTED["r3_split_sha256"],
        "schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "device_backend": "MPS",
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256_by_seed": {str(row.get("seed")): row.get("sha256") for row in checkpoints.get("checkpoints", [])},
        "cycle_evidence_cardinality": f"{evidence.get('record_count')}/{EXPECTED['expected_cycle_records']}",
        "cross_window_leakage_total": boundary.get("totals", {}).get("cross_window_links_with_live_bootstrap"),
        "required_artifacts_present": all((root / item).exists() for item in REQUIRED_ARTIFACTS if item != "manifest.json"),
        "required_evidence_files_present": all((root / item).exists() for item in REQUIRED_EVIDENCE_FILES),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in immutable.items()},
        "mutable_lifecycle_state_files": sorted(mutable),
        "hash_scope_policy": "mutable lifecycle state is declared separately from immutable evidence hashes",
        "append_only_artifact": True,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(dte, binding, evidence, questions, outcome, boundary, integrity, checkpoints, gate) -> str:
    header = f"""# {REPORT_TITLE}

gate = {gate['gate']}
outcome_classification = {outcome['outcome_classification']}
source_commit = {binding['source_provenance']['source_commit_before_run']}
parent_commit = {EXPECTED['h4m_x_source_commit']}
w1_repair_contract_sha256 = {EXPECTED['w1_repair_contract_sha256']}
actor_repair_contract_sha256 = {EXPECTED['actor_repair_contract_sha256']}
critic_repair_contract_sha256 = {EXPECTED['critic_repair_contract_sha256']}
split_sha256 = {EXPECTED['r3_split_sha256']}
schedule_sha256 = {EXPECTED['h4m_b_schedule_sha256']}
device_backend = MPS
cycle_evidence_cardinality = {evidence.get('record_count')}/{EXPECTED['expected_cycle_records']}
cross_window_leakage_total = {boundary['totals']['cross_window_links_with_live_bootstrap']}
TEST6_access_count = 0
training_evidence_source = {evidence.get('report_source')}
exact_next_gate = {gate['exact_next_gate']} (read-only review; not executed automatically)

## Primary questions

```json
{json.dumps({k: questions[k] for k in ['Q1_cross_window_leakage_zero', 'Q2_cycle_2_serve_transition', 'Q3_hold_better_correct_direction', 'Q4_serve_better_discrimination', 'Q5_three_seed_context_recovery']}, ensure_ascii=False, indent=2, default=dte._jsonable)}
```

## Credit statistics by context

```json
{json.dumps(questions['credit_statistics'], ensure_ascii=False, indent=2, default=dte._jsonable)}
```

## Window boundary diagnostics

```json
{json.dumps(boundary['checks'], ensure_ascii=False, indent=2, default=dte._jsonable)}
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
{json.dumps({'training_integrity': integrity.get('checks'), 'gate': gate.get('criteria')}, ensure_ascii=False, indent=2, default=dte._jsonable)}
```

{questions['interpretation_rule']}

No TEST6 access, no tuning, no extra cycles or updates, no old checkpoint reuse, no W2/W3/W4, and no GitHub
push occurred. This is diagnostic training evidence, not a promotion decision.

---

"""
    return header + dte.render_final_report(evidence, title=f"{REPORT_TITLE} — persisted training evidence")


def write_block(root: Path, binding, reason: str, started: float, umod: Any) -> None:
    gate = {
        "stage": STAGE,
        "gate": f"{BLOCK_GATE_PREFIX}_{reason}",
        "decision": "H4M_Y_BLOCKED",
        "exact_next_gate": f"STOP_{BLOCK_GATE_PREFIX}",
        "block_reason": reason,
        "final_flags": {"TEST6_opened": False, "github_push_performed": False, "tuning_applied": False, "w2_w3_w4_applied": False},
    }
    empty = {"stage": STAGE, "not_executed_or_incomplete": reason}
    for name in REQUIRED_ARTIFACTS:
        if name in {"manifest.json", "gate_matrix.json"} or (root / name).exists():
            continue
        if name == "final_report.md":
            (root / name).write_text(
                f"# H4M-Y\n\ngate = {gate['gate']}\nblock_reason = {reason}\n\nSTOP. No tuning, no extended training, no W2/W3/W4, no TEST6, no push.\n",
                encoding="utf-8",
            )
        else:
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
    print(f"[H4M-Y] artifact root: {root}\n[H4M-Y] gate: {gate['gate']}\n[H4M-Y] block_reason: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=REPORT_TITLE)
    parser.add_argument("--regenerate-report", type=Path, default=None, metavar="ARTIFACT_ROOT")
    args = parser.parse_args()
    umod = import_module("h4my_umod", H4MU_SOURCE)
    r2mod = import_module("h4my_r2mod", H4MU_R2_SOURCE)
    if args.regenerate_report is not None:
        dte = umod.load_durable_evidence()
        recovery = dte.regenerate_report(args.regenerate_report, report_name="final_report.md", title=REPORT_TITLE)
        print(json.dumps(recovery, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(0 if recovery["recovered"] else 1)

    started = time.perf_counter()
    created_at = now().isoformat()
    stamp = now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_y_fresh_window_boundary_repaired_three_seed_retraining_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    umod.STAGE = STAGE
    umod.SOURCE_REL = SOURCE_REL
    umod.REPORT_TITLE = REPORT_TITLE
    r2mod.STAGE = STAGE
    r2mod.CHECKPOINT_NAME = CHECKPOINT_NAME

    provenance = source_provenance(created_at)
    h4mg_probe = import_module("h4my_g_probe", H4MG_SOURCE)
    binding = authoritative_binding(created_at, provenance, h4mg_probe)
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
        "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
        "window_episode_boundary_schema": EXPECTED["boundary_schema"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "ppo_updates_per_seed": EXPECTED["ppo_updates_per_seed"],
        "critic_updates_per_seed": EXPECTED["critic_updates_per_seed"],
        "device_backend": "MPS",
        "durable_evidence_instrumentation": "H4M-U-R1",
        "training_execution_contract": "H4M-Q frozen lineage + H4M-P actor heads + H4M-T S3 critic target + H4M-W W1 window episode boundary",
    }
    expected_grid = [(seed, cycle) for seed in EXPECTED["seeds"] for cycle in range(1, EXPECTED["outer_training_count"] + 1)]
    writer = dte.DurableCycleEvidenceWriter(evidence_dir, run_header=run_header, expected_grid=expected_grid)
    update_audits: List[Dict[str, Any]] = []

    try:
        qmod = import_module(f"h4my_q_{time.time_ns()}", H4MQ_SOURCE)
        h4mg = import_module(f"h4my_g_{time.time_ns()}", H4MG_SOURCE)
        binder = make_y_binder_class(umod, r2mod, h4mg)(dte, writer, root, STAGE, run_header)
        kmod = umod.configure_execution(qmod, h4mg, update_audits, binder)
        base, _rerun = kmod.load_h4m_helpers()
        umod.bind_cycle_evidence_flush(base, binder)
        r2mod.bind_r2_checkpoint_naming(kmod, umod)
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
                "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
                "window_episode_boundary_schema": EXPECTED["boundary_schema"],
                "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
                "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
                "critic_target_binding_schema": h4mg.CRITIC_VALUE_TARGET_BINDING_SCHEMA,
                "device_backend": "MPS",
                "old_checkpoint_reuse": False,
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
        r2_evidence = r2mod.r2_evidence_completeness(dte, evidence_dir)
        w1_evidence = w1_evidence_completeness(dte, evidence_dir)
        boundary = window_boundary_diagnostics(root)
        integrity = umod.training_integrity(
            training, trace_meta, update_audits, credit, parameter_delta, gradient, checkpoints, validation, evidence
        )
        integrity["checks"]["r2_required_evidence_complete"] = r2_evidence["passed"]
        integrity["checks"]["w1_boundary_evidence_complete"] = w1_evidence["passed"]
        integrity["checks"]["cross_window_leakage_zero"] = boundary["totals"]["cross_window_links_with_live_bootstrap"] == 0
        integrity["training_integrity_passed"] = all(integrity["checks"].values())
        integrity["failing_criteria"] = [key for key, value in integrity["checks"].items() if not value]
        integrity["zero_loss_counts"] = r2_evidence["zero_loss_totals"]

        questions = primary_questions(conditional, boundary, w1_evidence, credit, validation, gradient)
        outcome = outcome_classification(questions, integrity["training_integrity_passed"])
        changed = umod.changed_files_audit()
        gate = gate_matrix(binding, integrity, outcome, changed, evidence, r2_evidence, w1_evidence, boundary, checkpoints, training)

        seed_rows = {int(row["seed"]): row for row in training["seed_summaries"]}
        for seed in EXPECTED["seeds"]:
            umod.write_json(
                root / f"seed_{seed}_summary.json",
                {
                    "seed_summary": seed_rows[seed],
                    "cycle_summaries": [row for row in cycle_summaries if int(row["seed"]) == seed],
                    "durable_evidence": evidence["by_seed"].get(str(seed)),
                    "durable_evidence_cycles": [row for row in evidence["by_seed_cycle"] if int(row["seed"]) == seed],
                    "window_boundary": [row for row in boundary["by_seed_cycle"] if int(row["seed"]) == seed],
                },
            )
        payloads = {
            "training_summary.json": training,
            "cycle_action_evolution.json": qmod.cycle_action_evolution(training, conditional),
            "context_credit_diagnostics.json": credit,
            "window_boundary_diagnostics.json": boundary,
            "checkpoint_integrity.json": checkpoints,
            "gradient_isolation.json": gradient,
            "parameter_delta.json": parameter_delta,
            "validation_discrimination.json": validation,
            "training_integrity.json": integrity,
            "primary_questions.json": questions,
            "outcome_classification.json": outcome,
            "durable_evidence_report.json": evidence,
            "r2_required_evidence_validation.json": r2_evidence,
            "w1_boundary_evidence_validation.json": w1_evidence,
            "evidence_schema.json": dte.evidence_schema(),
            "changed_files.json": changed,
            "test_results.json": {
                "stage": STAGE,
                "commands": [f"{sys.executable} -m py_compile {SOURCE_REL}", "git diff --cached --check", f"{sys.executable} {SOURCE_REL}"],
                "training_execution_authorized": True,
                "optimizer_step_count_training": training.get("total_ppo_updates"),
                "h4m_x_fixture_suite": "05_training/test_h4m_x_window_episode_boundary.py (unchanged H4M-X PASS suite)",
                "TEST6_access_count": 0,
            },
            "gate_matrix.json": gate,
            "conditional_policy_discrimination_training_trace.json": conditional,
        }
        for name, payload in payloads.items():
            umod.write_json(root / name, payload)
        (root / "final_report.md").write_text(
            final_report(dte, binding, evidence, questions, outcome, boundary, integrity, checkpoints, gate), encoding="utf-8"
        )
        umod.write_json(root / "manifest.json", make_manifest(root, gate, evidence, checkpoints, boundary, started))
        writer.mark_report_complete({"gate": gate.get("gate"), "report_status": evidence.get("report_status")})

        print(f"[H4M-Y] artifact root: {root}")
        print(f"[H4M-Y] gate: {gate['gate']}")
        print(f"[H4M-Y] outcome: {outcome['outcome_classification']}")
        print(f"[H4M-Y] cycle evidence: {evidence['record_count']}/{EXPECTED['expected_cycle_records']}")
        print(f"[H4M-Y] cross-window leakage: {boundary['totals']['cross_window_links_with_live_bootstrap']}")
        print(f"[H4M-Y] failing criteria: {gate['failing_criteria']}")
        print(f"[H4M-Y] exact next gate: {gate['exact_next_gate']} (read-only review, not executed)")
    except Exception as exc:
        write_block(root, binding, f"EXECUTION_EXCEPTION_{type(exc).__name__}", started, umod)
        umod.recover_after_reporting_failure(dte, root, exc)
        raise


if __name__ == "__main__":
    main()
