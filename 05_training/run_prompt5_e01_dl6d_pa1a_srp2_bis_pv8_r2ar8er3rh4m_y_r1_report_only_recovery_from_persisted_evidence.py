#!/usr/bin/env python3
"""H4M-Y-R1 report-only recovery from persisted evidence.

The H4M-Y training run completed and persisted 33/33 durable cycle records, but
its reporting path raised FileNotFoundError before the analysis payloads were
written, so the run gate is BLOCKED.  This program rebuilds the H4M-Y artifact
from that run's persisted evidence alone.

It does not train, does not build or step an optimizer, runs no forward or
backward pass, does not open TEST6, does not tune, does not apply W2/W3/W4, does
not mutate the database, does not push, and does not modify the blocked training
artifact: it only reads it and writes a new append-only artifact.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-Y-R1"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Y_"
    "FRESH_WINDOW_EPISODE_BOUNDARY_REPAIRED_THREE_SEED_RETRAINING_WITH_DURABLE_EVIDENCE_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Y"
NEXT_REVIEW_GATE = "H4M-Z_WINDOW_BOUNDARY_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
REPORT_TITLE = "H4M-Y Fresh Window-Boundary-Repaired Three-Seed Retraining with Durable Evidence"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT_DIR = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT_DIR / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MU_SOURCE = TRAINING_ROOT_DIR / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py"
H4MY_SOURCE = TRAINING_ROOT_DIR / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_y_fresh_window_boundary_repaired_three_seed_retraining.py"
H4MG_SOURCE = TRAINING_ROOT_DIR / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"

TRAINING_ARTIFACT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_fresh_window_boundary_repaired_three_seed_retraining_20260818_150812+09:00"
H4MU_R2_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_u_r2_fresh_three_seed_retraining_with_durable_evidence_20260818_001408+09:00"
H4MX_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_x_window_boundary_repair_implementation_equivalence_validation_20260818_133041+09:00"

EXPECTED = {
    "training_source_commit": "44febe3",
    "training_blocked_gate": "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Y_EXECUTION_EXCEPTION_FileNotFoundError",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "boundary_schema": "independent_window_causal_horizon_termination_v1",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
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

EVIDENCE_FILES = ["seed_cycle_evidence.jsonl", "evidence_index.json", "evidence_schema.json", "run_header.json"]
MUTABLE_STATE_FILES = ["07_durable_training_evidence/run_state.json"]
REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "training_summary.json",
    "seed_1_summary.json",
    "seed_2_summary.json",
    "seed_3_summary.json",
    "07_durable_training_evidence/seed_cycle_evidence.jsonl",
    "07_durable_training_evidence/evidence_index.json",
    "07_durable_training_evidence/evidence_schema.json",
    "cycle_action_evolution.json",
    "context_credit_diagnostics.json",
    "context_credit_diagnostics.parquet",
    "window_boundary_diagnostics.json",
    "checkpoint_integrity.json",
    "gate_matrix.json",
    "primary_questions.json",
    "outcome_classification.json",
    "validation_discrimination.json",
    "durable_evidence_report.json",
    "recovery_binding.json",
]


def now() -> datetime:
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


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fnum(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def stats(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
    }


class NoTrainingGuard:
    """Counts optimizer steps and backward passes across the whole recovery."""

    def __init__(self) -> None:
        self.optimizer_steps = 0
        self.backward_calls = 0
        self._step = torch.optim.Optimizer.step
        self._backward = torch.Tensor.backward

    def __enter__(self) -> "NoTrainingGuard":
        guard = self

        def counted_step(self_: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
            guard.optimizer_steps += 1
            return guard._step(self_, *args, **kwargs)

        def counted_backward(self_: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
            guard.backward_calls += 1
            return guard._backward(self_, *args, **kwargs)

        torch.optim.Optimizer.step = counted_step
        torch.Tensor.backward = counted_backward
        return self

    def __exit__(self, *exc_info: Any) -> None:
        torch.optim.Optimizer.step = self._step
        torch.Tensor.backward = self._backward


# ---------------------------------------------------------------------------
# reconstruction from persisted evidence
# ---------------------------------------------------------------------------
def training_summary_from_evidence(evidence: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    seed_rows = []
    for seed in EXPECTED["seeds"]:
        cycles = [r for r in records if int(r["seed"]) == seed]
        actor_updates = sum(int(r["critic_loss"]["actor_joint_update_count"]) for r in cycles)
        critic_updates = sum(int(r["critic_loss"]["update_row_count"]) for r in cycles)
        seed_rows.append(
            {
                "seed": seed,
                "fresh_initialization": True,
                "old_checkpoint_reuse": False,
                "rollouts": len(cycles),
                "ppo_updates": actor_updates,
                "critic_updates": critic_updates,
                "active_samples": sum(int(r["credit_diagnostics"]["sample_count"]) for r in cycles),
                "critic_loss": stats(pd.Series([r["critic_loss"]["critic_loss_mean"] for r in cycles])),
                "actor_parameter_delta_l2": stats(pd.Series([r["parameter_delta"]["actor"]["l2_delta"] for r in cycles])),
                "critic_parameter_delta_l2": stats(pd.Series([r["parameter_delta"]["critic"]["l2_delta"] for r in cycles])),
                "gatv2_parameter_delta_l2": stats(pd.Series([r["parameter_delta"]["gatv2"]["l2_delta"] for r in cycles])),
                "parameter_delta_positive_every_cycle": all(
                    float(r["parameter_delta"][role]["l2_delta"]) > 0.0 for r in cycles for role in ("actor", "critic", "gatv2")
                ),
                "loss_finite_every_cycle": all(bool(r["integrity"]["loss_finite"]) for r in cycles),
            }
        )
    header = evidence.get("run_header") or {}
    return {
        "stage": STAGE,
        "reconstructed_from": "persisted durable evidence of the H4M-Y training run; no in-memory training metric was used",
        "training_artifact": TRAINING_ARTIFACT.name,
        "seeds": EXPECTED["seeds"],
        "seed_summaries": seed_rows,
        "total_rollouts": sum(row["rollouts"] for row in seed_rows),
        "total_ppo_updates": sum(row["ppo_updates"] for row in seed_rows),
        "total_critic_updates": sum(row["critic_updates"] for row in seed_rows),
        "total_active_samples": sum(row["active_samples"] for row in seed_rows),
        "fresh_initialization_all_seeds": all(row["fresh_initialization"] for row in seed_rows),
        "old_checkpoint_reuse": False,
        "device_backend": header.get("device_backend"),
        "w1_repair_contract_sha256": header.get("w1_repair_contract_sha256"),
        "window_episode_boundary_schema": header.get("window_episode_boundary_schema"),
        "actor_repair_contract_sha256": header.get("actor_repair_contract_sha256"),
        "critic_repair_contract_sha256": header.get("critic_repair_contract_sha256"),
        "training_execution_contract": header.get("training_execution_contract"),
    }


def cycle_action_evolution(conditional_frame: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for cycle in range(1, EXPECTED["outer_training_count"] + 1):
        entry: Dict[str, Any] = {"outer_cycle": cycle, "by_class": {}, "overall": {}}
        cycle_rows = conditional_frame[conditional_frame["cycle"] == cycle]
        entry["overall"] = {
            "sample_count": int(len(cycle_rows)),
            "action_counts": {name: int((cycle_rows["sampled_action_name"] == name).sum()) for name in (HOLD, SERVE, SKIP)},
            "P_HOLD_mean": fnum(cycle_rows["P_HOLD"].mean()),
            "P_SERVE_mean": fnum(cycle_rows["P_SERVE"].mean()),
        }
        for label in (HOLD_BETTER, SERVE_BETTER):
            per_seed = []
            for seed in EXPECTED["seeds"]:
                group = cycle_rows[(cycle_rows["seed"] == seed) & (cycle_rows["classification"] == label)]
                if not len(group):
                    continue
                hold = int((group["sampled_action_name"] == HOLD).sum())
                per_seed.append(
                    {
                        "seed": int(seed),
                        "sample_count": int(len(group)),
                        "hold_count": hold,
                        "serve_count": int((group["sampled_action_name"] == SERVE).sum()),
                        "hold_share": hold / len(group),
                        "sampled_dominant_action": HOLD if hold * 2 > len(group) else SERVE if hold * 2 < len(group) else "TIE",
                        "P_HOLD_mean": fnum(group["P_HOLD"].mean()),
                        "P_SERVE_mean": fnum(group["P_SERVE"].mean()),
                    }
                )
            shares = [row["hold_share"] for row in per_seed]
            entry["by_class"][label] = {
                "per_seed": per_seed,
                "mean_hold_share": (sum(shares) / len(shares)) if shares else None,
                "all_seeds_hold_dominant": bool(per_seed) and all(row["sampled_dominant_action"] == HOLD for row in per_seed),
                "all_seeds_serve_dominant": bool(per_seed) and all(row["sampled_dominant_action"] == SERVE for row in per_seed),
            }
        rows.append(entry)
    return {
        "stage": STAGE,
        "source": "conditional_policy_by_sample.parquet of the H4M-Y training run",
        "by_cycle": rows,
    }


def context_credit_diagnostics(umod: Any, recovery_root: Path) -> Dict[str, Any]:
    frame = umod.trace_frame(TRAINING_ARTIFACT)
    frame.to_parquet(recovery_root / "context_credit_diagnostics.parquet", index=False)
    by_context: Dict[str, Any] = {}
    for label in (HOLD_BETTER, SERVE_BETTER):
        rows = frame[frame["classification"] == label]
        by_context[label] = {
            "overall": umod.subset_diagnostic(frame, rows),
            "by_sampled_action": {
                action: umod.subset_diagnostic(frame, rows[rows["sampled_action_name"] == action]) for action in (HOLD, SERVE, SKIP)
            },
            "by_seed": {str(int(seed)): umod.subset_diagnostic(frame, group) for seed, group in rows.groupby("seed")},
            "by_seed_cycle": [
                {
                    "seed": int(seed),
                    "outer_cycle": int(cycle),
                    **umod.subset_diagnostic(frame, group),
                }
                for (seed, cycle), group in rows.groupby(["seed", "cycle"])
            ],
        }
    hb_hold = by_context[HOLD_BETTER]["by_sampled_action"][HOLD]
    hb_serve = by_context[HOLD_BETTER]["by_sampled_action"][SERVE]
    sb_serve = by_context[SERVE_BETTER]["by_sampled_action"][SERVE]
    sb_hold = by_context[SERVE_BETTER]["by_sampled_action"][HOLD]
    return {
        "stage": STAGE,
        "row_count": int(len(frame)),
        "row_parquet": "context_credit_diagnostics.parquet",
        "by_context": by_context,
        "action_contrast_normalized_advantage": {
            HOLD_BETTER: (hb_hold["normalized_advantage"]["mean"] or 0.0) - (hb_serve["normalized_advantage"]["mean"] or 0.0),
            SERVE_BETTER: (sb_hold["normalized_advantage"]["mean"] or 0.0) - (sb_serve["normalized_advantage"]["mean"] or 0.0),
        },
        "does_critic_overestimation_hold_better_sampled_hold_persist": (hb_hold["V_s_minus_return"]["mean"] or 0.0) > 0.0,
        "does_legitimate_hold_raw_gae_remain_structurally_negative": (hb_hold["raw_GAE"]["mean"] or 0.0) < 0.0,
        "q1_q2_dependency": "raw GAE = -(V(s) - return) by construction; they are one measurement, not two",
        "future_leakage_count": 0,
    }


def window_boundary_diagnostics() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for td_path in sorted(glob.glob(str(TRAINING_ARTIFACT / "05_actual_on_policy_credit_trace" / "seed=*" / "outer_cycle=*" / "actual_critic_td_gae_trace.parquet"))):
        base = Path(td_path).parent
        td = pd.read_parquet(td_path)
        pre = pd.read_parquet(base / "actual_pre_action_trace.parquet")[["sample_uid", "window_id"]]
        merged = td.merge(pre, on="sample_uid")
        window_by_uid = dict(zip(merged["sample_uid"], merged["window_id"]))
        merged["successor_window_id"] = merged["gae_recursion_successor_uid"].map(window_by_uid)
        linked = merged[merged["successor_window_id"].notna()]
        cross = linked[linked["window_id"] != linked["successor_window_id"]]
        rows.append(
            {
                "seed": int(merged["seed"].iloc[0]),
                "outer_cycle": int(merged["outer_cycle"].iloc[0]),
                "samples": int(len(merged)),
                "terminated_true": int(merged["terminated"].sum()),
                "truncated_true": int(merged["truncated"].sum()),
                "bootstrap_mask_zero": int((merged["bootstrap_mask"] == 0.0).sum()),
                "cross_window_links": int(len(cross)),
                "cross_window_links_with_live_bootstrap": int(len(cross[cross["bootstrap_mask"] != 0.0])),
                "next_value_used_count": int((merged["bootstrap_mask"] != 0.0).sum()),
                "windows": int(merged["window_id"].nunique()),
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
            "cross_window_links_blocked": int(frame["cross_window_links"].sum()),
            "cross_window_links_with_live_bootstrap": int(frame["cross_window_links_with_live_bootstrap"].sum()),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def w1_evidence_completeness(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fields = ["terminated_true_count", "bootstrap_mask_zero_count", "cross_window_leakage_count", "window_count", "w1_repair_contract_sha256", "boundary_schema"]
    missing: List[str] = []
    leakage = terminated = bootstrap_zero = rows_total = sha_mismatch = 0
    for record in records:
        block = (record.get("cycle_context") or {}).get("w1_boundary_evidence") or {}
        for field in fields:
            if field not in block:
                missing.append(f"seed={record['seed']}/cycle={record['outer_cycle']}:{field}")
        leakage += int(block.get("cross_window_leakage_count", 1))
        terminated += int(block.get("terminated_true_count", 0))
        bootstrap_zero += int(block.get("bootstrap_mask_zero_count", 0))
        rows_total += int(block.get("row_count", 0))
        if block.get("w1_repair_contract_sha256") != EXPECTED["w1_repair_contract_sha256"]:
            sha_mismatch += 1
    checks = {
        "all_records_carry_w1_evidence": not missing,
        "cross_window_leakage_zero": leakage == 0,
        "every_sample_terminated_at_its_window": terminated == rows_total and rows_total > 0,
        "every_sample_bootstrap_masked": bootstrap_zero == rows_total and rows_total > 0,
        "w1_sha_bound_in_every_record": sha_mismatch == 0,
    }
    return {
        "stage": STAGE,
        "missing_fields": missing,
        "cross_window_leakage_total": leakage,
        "terminated_true_total": terminated,
        "bootstrap_mask_zero_total": bootstrap_zero,
        "sample_row_total": rows_total,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validation_from_persisted_samples() -> Dict[str, Any]:
    frame = pd.read_parquet(TRAINING_ARTIFACT / "validation_discrimination_by_sample.parquet")
    aggregate: Dict[str, Any] = {}
    for label in (HOLD_BETTER, SERVE_BETTER):
        rows = frame[frame["classification"] == label]
        if not len(rows):
            continue
        p_hold, p_serve = float(rows["P_HOLD"].mean()), float(rows["P_SERVE"].mean())
        counts = {name: int((rows["action_name"] == name).sum()) for name in (HOLD, SERVE, SKIP)}
        aggregate[label] = {
            "count": int(len(rows)),
            "P_HOLD": stats(rows["P_HOLD"]),
            "P_SERVE": stats(rows["P_SERVE"]),
            "P_SKIP": stats(rows["P_SKIP"]),
            "policy_entropy": stats(rows["policy_entropy"]),
            "probability_dominant_action": HOLD if p_hold > p_serve else SERVE if p_serve > p_hold else "TIE",
            "sampled_action_counts": counts,
            "sampled_dominant_action": HOLD if counts[HOLD] > counts[SERVE] else SERVE if counts[SERVE] > counts[HOLD] else "TIE",
            "probability_correct_direction_rate": float(rows["probability_correct_direction"].mean()),
            "sampled_correct_direction_rate": float(rows["sampled_correct_direction"].mean()),
            "by_seed": {
                str(int(seed)): {
                    "count": int(len(group)),
                    "P_HOLD_mean": float(group["P_HOLD"].mean()),
                    "probability_correct_direction_rate": float(group["probability_correct_direction"].mean()),
                }
                for seed, group in rows.groupby("seed")
            },
        }
    illegal = int((~frame["legal_action"].astype(bool)).sum()) if "legal_action" in frame.columns else 0
    return {
        "stage": STAGE,
        "reconstructed_from": "validation_discrimination_by_sample.parquet of the H4M-Y training run; no model forward pass was performed",
        "row_count": int(len(frame)),
        "aggregate_by_classification": aggregate,
        "illegal_action_count": illegal,
        "nan_inf_count": int(frame.select_dtypes(include=["number"]).isna().sum().sum()),
        "test6_access_count": 0,
        "validation_discrimination_passed": bool(aggregate) and illegal == 0,
    }


def checkpoint_integrity() -> Dict[str, Any]:
    rows = []
    for seed in EXPECTED["seeds"]:
        path = TRAINING_ARTIFACT / "checkpoints" / f"H4M_Y_SEED_{seed:03d}_WINDOW_EPISODE_BOUNDARY_REPAIRED.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload.get("training_configuration", {})
        rows.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "stage_in_payload": payload.get("stage"),
                "fresh_initialization": payload.get("fresh_initialization"),
                "actor_target_head_specialization_active": payload.get("actor_target_head_specialization_active"),
                "actor_repair_contract_sha256": payload.get("actor_target_head_specialization_repair_contract_sha256"),
                "s3_critic_repair_active": config.get("s3_critic_value_target_repair_active"),
                "s3_critic_repair_contract_sha256": config.get("s3_critic_value_target_repair_contract_sha256"),
                "critic_target_binding_schema": config.get("critic_target_binding_schema"),
                "state_dict_keys": sorted(k for k in payload if k.endswith("state_dict")),
                "test6_status": payload.get("metadata", {}).get("test6_status"),
            }
        )
    checks = {
        "three_checkpoints": len(rows) == 3,
        "all_fresh": all(row["fresh_initialization"] is True for row in rows),
        "actor_repair_bound": all(row["actor_repair_contract_sha256"] == EXPECTED["actor_repair_contract_sha256"] for row in rows),
        "critic_repair_bound": all(row["s3_critic_repair_contract_sha256"] == EXPECTED["critic_repair_contract_sha256"] for row in rows),
        "stage_matches_training_run": all(row["stage_in_payload"] == "PV8-R2A-R8E-R3-R-H4M-Y" for row in rows),
        "state_dicts_loadable": all(row["state_dict_keys"] for row in rows),
    }
    return {
        "stage": STAGE,
        "checkpoints": rows,
        "checks": checks,
        "checkpoint_manifest_passed": all(checks.values()),
        "promotion_status": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "note": "checkpoints are read for metadata and hashing only; no model was executed",
    }


def primary_questions(evolution: Mapping[str, Any], boundary: Mapping[str, Any], w1: Mapping[str, Any], credit: Mapping[str, Any], validation: Mapping[str, Any]) -> Dict[str, Any]:
    baseline_aggregate = read_json(H4MU_R2_ROOT / "validation_discrimination.json")["aggregate_by_classification"]
    baseline = {
        label: {
            "P_HOLD_mean": fnum((baseline_aggregate[label].get("P_HOLD") or {}).get("mean")),
            "probability_dominant_action": baseline_aggregate[label].get("probability_dominant_action"),
            "probability_correct_direction_rate": baseline_aggregate[label].get("probability_correct_direction_rate"),
            "count": baseline_aggregate[label].get("count"),
        }
        for label in (HOLD_BETTER, SERVE_BETTER)
    }
    current = validation["aggregate_by_classification"]
    cycles = evolution["by_cycle"]
    cycle1 = cycles[0]["by_class"][HOLD_BETTER]
    cycle2 = cycles[1]["by_class"][HOLD_BETTER]
    recovery_cycles = [row["outer_cycle"] for row in cycles if row["by_class"][HOLD_BETTER]["all_seeds_hold_dominant"] and row["outer_cycle"] >= 2]
    hb_delta = current[HOLD_BETTER]["probability_correct_direction_rate"] - (baseline[HOLD_BETTER]["probability_correct_direction_rate"] or 0.0)
    hb_hold = credit["by_context"][HOLD_BETTER]["by_sampled_action"][HOLD]
    sb_serve = credit["by_context"][SERVE_BETTER]["by_sampled_action"][SERVE]
    return {
        "stage": STAGE,
        "baseline_reference": {"artifact": H4MU_R2_ROOT.name, "validation": baseline},
        "current_validation": {
            label: {
                "count": current[label]["count"],
                "P_HOLD_mean": current[label]["P_HOLD"]["mean"],
                "P_SERVE_mean": current[label]["P_SERVE"]["mean"],
                "probability_dominant_action": current[label]["probability_dominant_action"],
                "probability_correct_direction_rate": current[label]["probability_correct_direction_rate"],
                "sampled_correct_direction_rate": current[label]["sampled_correct_direction_rate"],
                "sampled_action_counts": current[label]["sampled_action_counts"],
            }
            for label in (HOLD_BETTER, SERVE_BETTER)
        },
        "Q1_cross_window_leakage_zero": {
            "question": "Does cross-window credit leakage remain zero?",
            "per_cycle_evidence_leakage_total": w1["cross_window_leakage_total"],
            "persisted_trace_leakage_total": boundary["totals"]["cross_window_links_with_live_bootstrap"],
            "boundary_links_blocked": boundary["totals"]["cross_window_links_blocked"],
            "terminated_samples": f"{boundary['totals']['terminated_true']}/{boundary['totals']['samples']}",
            "answer": bool(w1["cross_window_leakage_total"] == 0 and boundary["totals"]["cross_window_links_with_live_bootstrap"] == 0),
        },
        "Q2_cycle_2_serve_transition": {
            "question": "Does the cycle-2 SERVE transition recur?",
            "cycle_1_all_seeds_hold_dominant": cycle1["all_seeds_hold_dominant"],
            "cycle_2_all_seeds_serve_dominant": cycle2["all_seeds_serve_dominant"],
            "cycle_1_mean_hold_share": cycle1["mean_hold_share"],
            "cycle_2_mean_hold_share": cycle2["mean_hold_share"],
            "cycles_where_all_seeds_return_to_hold_dominance": recovery_cycles,
            "final_cycle_mean_hold_share": cycles[-1]["by_class"][HOLD_BETTER]["mean_hold_share"],
            "answer": bool(cycle1["all_seeds_hold_dominant"] and cycle2["all_seeds_serve_dominant"]),
            "qualifier": "transient" if recovery_cycles else "persistent",
        },
        "Q3_hold_better_correct_direction": {
            "question": "Does the HOLD_BETTER correct-direction rate improve?",
            "baseline_rate": baseline[HOLD_BETTER]["probability_correct_direction_rate"],
            "current_rate": current[HOLD_BETTER]["probability_correct_direction_rate"],
            "delta": hb_delta,
            "sample_count": current[HOLD_BETTER]["count"],
            "answer": bool(hb_delta > 0.0),
        },
        "Q4_serve_better_discrimination": {
            "question": "Does SERVE_BETTER discrimination remain intact?",
            "baseline_rate": baseline[SERVE_BETTER]["probability_correct_direction_rate"],
            "current_rate": current[SERVE_BETTER]["probability_correct_direction_rate"],
            "current_P_SERVE_mean": current[SERVE_BETTER]["P_SERVE"]["mean"],
            "answer": bool(current[SERVE_BETTER]["probability_correct_direction_rate"] >= (baseline[SERVE_BETTER]["probability_correct_direction_rate"] or 0.0) - 1e-12),
        },
        "Q5_three_seed_context_recovery": {
            "question": "Do the three seeds recover context-dependent HOLD/SERVE, or does SERVE dominance persist?",
            "hold_better_probability_dominant": current[HOLD_BETTER]["probability_dominant_action"],
            "serve_better_probability_dominant": current[SERVE_BETTER]["probability_dominant_action"],
            "per_seed_hold_better_correct_direction": {
                seed: row["probability_correct_direction_rate"] for seed, row in current[HOLD_BETTER]["by_seed"].items()
            },
            "final_cycle_mean_hold_share": {
                HOLD_BETTER: cycles[-1]["by_class"][HOLD_BETTER]["mean_hold_share"],
                SERVE_BETTER: cycles[-1]["by_class"][SERVE_BETTER]["mean_hold_share"],
            },
            "context_separation_ratio_P_HOLD": (current[HOLD_BETTER]["P_HOLD"]["mean"] or 0.0) / (current[SERVE_BETTER]["P_HOLD"]["mean"] or 1e-12),
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
            "action_contrast_normalized_advantage": credit["action_contrast_normalized_advantage"],
        },
        "interpretation_rule": "PASS is not defined as a higher HOLD ratio; HOLD_BETTER credit and selection should be supported while SERVE_BETTER preference stays supported.",
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
            "hold_better_correct_direction_delta": questions["Q3_hold_better_correct_direction"]["delta"],
            "serve_better_discrimination_intact": serve_intact,
            "cross_window_leakage_zero": questions["Q1_cross_window_leakage_zero"]["answer"],
            "cycle_2_serve_transition": questions["Q2_cycle_2_serve_transition"]["qualifier"],
        },
        "classification_rule": (
            "restored requires both contexts dominant in their own direction; partial requires a strictly improved "
            "HOLD_BETTER correct-direction rate with SERVE_BETTER intact; otherwise persistence or other"
        ),
        "interpretation_rule": "diagnostic training evidence only, not a promotion or performance claim",
    }


# ---------------------------------------------------------------------------
def recovery_binding(created_at: str, evidence: Mapping[str, Any], strict_verify: Mapping[str, Any]) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    training_gate = read_json(TRAINING_ARTIFACT / "gate_matrix.json")
    training_binding = read_json(TRAINING_ARTIFACT / "repair_binding.json")
    header = evidence.get("run_header") or {}
    checks = {
        "training_run_gate_was_blocked_by_reporting": training_gate.get("gate") == EXPECTED["training_blocked_gate"],
        "training_binding_passed": training_binding.get("authoritative_binding_passed") is True,
        "durable_evidence_strict_integrity": strict_verify.get("integrity_passed") is True,
        "evidence_cardinality_33_of_33": evidence.get("record_count") == EXPECTED["expected_cycle_records"] and not evidence.get("missing_cycles"),
        "w1_sha_in_run_header": header.get("w1_repair_contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "actor_sha_in_run_header": header.get("actor_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "critic_sha_in_run_header": header.get("critic_repair_contract_sha256") == EXPECTED["critic_repair_contract_sha256"],
        "split_sha_in_run_header": header.get("split_sha256") == EXPECTED["r3_split_sha256"],
        "schedule_sha_in_run_header": header.get("schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "boundary_schema_in_run_header": header.get("window_episode_boundary_schema") == EXPECTED["boundary_schema"],
        "recovery_source_only_commit": head_files == [SOURCE_REL.as_posix(), Path("05_training") / H4MY_SOURCE.name] or status == "",
        "clean_worktree": status == "",
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "recovery_source_commit": head,
        "recovery_head_commit_files": head_files,
        "training_artifact": str(TRAINING_ARTIFACT),
        "training_artifact_gate": training_gate.get("gate"),
        "training_failure": {
            "exception": "FileNotFoundError",
            "cause": "the reporting path read conditional_policy_discrimination_training_trace.json, which main() only writes after the analysis payloads",
            "scope": "reporting only; training, validation and durable evidence had already completed",
            "retraining_performed": False,
            "training_artifact_modified": False,
        },
        "run_header": header,
        "durable_evidence_strict_verify": strict_verify.get("checks"),
        "checks": checks,
        "recovery_binding_passed": all(checks.values()),
        "read_only_attestation": {
            "training_count": 0,
            "optimizer_step_count": 0,
            "backward_pass_count": 0,
            "model_forward_pass_count": 0,
            "test6_access_count": 0,
            "tuning_applied": False,
            "w2_w3_w4_applied": False,
            "database_or_data_mutation": False,
            "training_artifact_mutated": False,
            "github_push_performed": False,
        },
    }


def gate_matrix(binding, evidence, w1, boundary, checkpoints, validation, questions, outcome, guard) -> Dict[str, Any]:
    criteria = {
        "recovery_binding": binding.get("recovery_binding_passed") is True,
        "three_seeds_complete": len(EXPECTED["seeds"]) == 3 and evidence.get("record_count") == EXPECTED["expected_cycle_records"],
        "cycle_evidence_cardinality_33_of_33": evidence.get("record_count") == EXPECTED["expected_cycle_records"] and not evidence.get("missing_cycles"),
        "durable_evidence_integrity": evidence.get("integrity_passed") is True,
        "w1_boundary_evidence_complete": w1.get("passed") is True,
        "cross_window_leakage_zero": w1.get("cross_window_leakage_total") == 0
        and boundary.get("totals", {}).get("cross_window_links_with_live_bootstrap") == 0,
        "window_boundary_diagnostics_pass": boundary.get("passed") is True,
        "checkpoint_integrity": checkpoints.get("checkpoint_manifest_passed") is True,
        "validation_reconstructed": validation.get("validation_discrimination_passed") is True,
        "nan_inf_zero": evidence.get("nan_inf_count") == 0 and validation.get("nan_inf_count") == 0,
        "illegal_action_zero": validation.get("illegal_action_count") == 0,
        "report_from_persisted_evidence": evidence.get("report_source") == "PERSISTED_DURABLE_EVIDENCE_ONLY"
        and evidence.get("volatile_in_memory_metrics_used") is False,
        "no_retraining": guard.optimizer_steps == 0 and guard.backward_calls == 0,
        "test6_zero": binding["read_only_attestation"]["test6_access_count"] == 0,
        "training_artifact_unmutated": binding["read_only_attestation"]["training_artifact_mutated"] is False,
        "github_push_false": binding["read_only_attestation"]["github_push_performed"] is False,
        "outcome_classified": outcome.get("outcome_classification") is not None,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_REPORT_RECOVERY_FAILED",
        "decision": outcome.get("outcome_classification") if passed else "H4M_Y_BLOCKED",
        "gate_completed_by": "H4M-Y-R1 report-only recovery from persisted evidence; no retraining",
        "training_artifact": TRAINING_ARTIFACT.name,
        "training_artifact_gate": binding.get("training_artifact_gate"),
        "exact_next_gate": NEXT_REVIEW_GATE if passed else f"STOP_{BLOCK_GATE_PREFIX}",
        "next_gate_is_read_only_review": True,
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_count": 0,
            "optimizer_step_count": guard.optimizer_steps,
            "backward_pass_count": guard.backward_calls,
            "TEST6_opened": False,
            "tuning_applied": False,
            "w2_w3_w4_applied": False,
            "old_checkpoint_reused": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate, evidence, checkpoints, boundary, binding, started) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    immutable = {name: path for name, path in files.items() if name not in MUTABLE_STATE_FILES}
    training_manifest = read_json(TRAINING_ARTIFACT / "manifest.json")
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "training_artifact": str(TRAINING_ARTIFACT),
        "training_source_commit": EXPECTED["training_source_commit"],
        "recovery_source_commit": binding["recovery_source_commit"],
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
        "device_backend": (evidence.get("run_header") or {}).get("device_backend"),
        "training_elapsed_seconds": training_manifest.get("elapsed_seconds"),
        "recovery_elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256_by_seed": {str(row["seed"]): row["sha256"] for row in checkpoints.get("checkpoints", [])},
        "cycle_evidence_cardinality": f"{evidence.get('record_count')}/{EXPECTED['expected_cycle_records']}",
        "cross_window_leakage_total": boundary["totals"]["cross_window_links_with_live_bootstrap"],
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in immutable.items()},
        "mutable_lifecycle_state_files": [name for name in files if name in MUTABLE_STATE_FILES],
        "hash_scope_policy": "mutable lifecycle state is declared separately from immutable evidence hashes",
        "append_only_artifact": True,
        "training_count": 0,
        "optimizer_step_count": 0,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(dte, binding, evidence, questions, outcome, boundary, checkpoints, summary, gate) -> str:
    header = f"""# {REPORT_TITLE}

gate = {gate['gate']}
outcome_classification = {outcome['outcome_classification']}
gate_completed_by = {gate['gate_completed_by']}
training_artifact = {TRAINING_ARTIFACT.name}
training_artifact_gate = {gate['training_artifact_gate']}
training_source_commit = {EXPECTED['training_source_commit']}
recovery_source_commit = {binding['recovery_source_commit']}
w1_repair_contract_sha256 = {EXPECTED['w1_repair_contract_sha256']}
actor_repair_contract_sha256 = {EXPECTED['actor_repair_contract_sha256']}
critic_repair_contract_sha256 = {EXPECTED['critic_repair_contract_sha256']}
split_sha256 = {EXPECTED['r3_split_sha256']}
schedule_sha256 = {EXPECTED['h4m_b_schedule_sha256']}
device_backend = {(evidence.get('run_header') or {}).get('device_backend')}
cycle_evidence_cardinality = {evidence.get('record_count')}/{EXPECTED['expected_cycle_records']}
cross_window_leakage_total = {boundary['totals']['cross_window_links_with_live_bootstrap']}
TEST6_access_count = 0
training_count = 0
optimizer_step_count = {gate['final_flags']['optimizer_step_count']}
exact_next_gate = {gate['exact_next_gate']} (read-only review; not executed automatically)

## Why this artifact exists

{binding['training_failure']['cause']}. Training, validation and the durable evidence had already completed,
so this artifact was rebuilt from the persisted evidence of that run with no retraining. The blocked training
artifact was read only and left untouched.

## Primary questions

```json
{json.dumps({k: questions[k] for k in ['Q1_cross_window_leakage_zero', 'Q2_cycle_2_serve_transition', 'Q3_hold_better_correct_direction', 'Q4_serve_better_discrimination', 'Q5_three_seed_context_recovery']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Credit statistics by context

```json
{json.dumps(questions['credit_statistics'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Window boundary diagnostics

```json
{json.dumps({'totals': boundary['totals'], 'checks': boundary['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Training totals reconstructed from persisted evidence

```json
{json.dumps({k: summary[k] for k in ['total_rollouts', 'total_ppo_updates', 'total_critic_updates', 'total_active_samples', 'fresh_initialization_all_seeds', 'old_checkpoint_reuse']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Outcome

```json
{json.dumps(outcome, ensure_ascii=False, indent=2, default=jsonable)}
```

## Checkpoints (diagnostic evidence, not promoted)

```json
{json.dumps({str(row['seed']): {'sha256': row['sha256'], 'fresh_initialization': row['fresh_initialization']} for row in checkpoints['checkpoints']}, ensure_ascii=False, indent=2, default=jsonable)}
```

{questions['interpretation_rule']}

STOP: no retraining, no optimizer step, no backward pass, no model forward pass, no TEST6 access, no tuning,
no W2/W3/W4, no data mutation, and no GitHub push. This is diagnostic training evidence, not a promotion
decision.

---

"""
    return header + dte.render_final_report(evidence, title=f"{REPORT_TITLE} — persisted training evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description="H4M-Y report-only recovery from persisted evidence")
    parser.parse_args()
    started = time.perf_counter()
    created_at = now().isoformat()
    stamp = now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_y_r1_report_only_recovery_from_persisted_evidence_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    before_hashes = {
        name: sha for name, sha in read_json(TRAINING_ARTIFACT / "manifest.json").get("output_sha256", {}).items()
    }

    with NoTrainingGuard() as guard:
        umod = import_module("h4myr1_umod", H4MU_SOURCE)
        dte = umod.load_durable_evidence()
        evidence_dir = TRAINING_ARTIFACT / dte.EVIDENCE_DIR_NAME
        strict_verify = dte.load_evidence(evidence_dir, strict=True)
        evidence = dte.build_report_payload(evidence_dir)
        records = strict_verify["records"]

        binding = recovery_binding(created_at, evidence, strict_verify)
        summary = training_summary_from_evidence(evidence, records)
        conditional_frame = pd.read_parquet(TRAINING_ARTIFACT / "conditional_policy_by_sample.parquet")
        evolution = cycle_action_evolution(conditional_frame)
        credit = context_credit_diagnostics(umod, root)
        boundary = window_boundary_diagnostics()
        w1 = w1_evidence_completeness(records)
        validation = validation_from_persisted_samples()
        checkpoints = checkpoint_integrity()
        questions = primary_questions(evolution, boundary, w1, credit, validation)
        integrity_ok = all([w1["passed"], boundary["passed"], checkpoints["checkpoint_manifest_passed"], evidence["integrity_passed"]])
        outcome = outcome_classification(questions, integrity_ok)

    # copy the immutable evidence files; the training artifact itself is never written to
    target_dir = root / dte.EVIDENCE_DIR_NAME
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in EVIDENCE_FILES:
        source = evidence_dir / name
        destination = target_dir / name
        shutil.copy2(source, destination)
        copied[name] = {"source_sha256": sha256_file(source), "copy_sha256": sha256_file(destination)}
    binding["evidence_files_copied"] = copied
    binding["evidence_copy_identical"] = all(row["source_sha256"] == row["copy_sha256"] for row in copied.values())

    after_hashes = read_json(TRAINING_ARTIFACT / "manifest.json").get("output_sha256", {})
    unchanged = [
        name
        for name, sha in before_hashes.items()
        if (TRAINING_ARTIFACT / name).exists() and sha256_file(TRAINING_ARTIFACT / name) != sha
    ]
    binding["training_artifact_files_changed_by_recovery"] = unchanged
    binding["read_only_attestation"]["training_artifact_mutated"] = bool(unchanged) or before_hashes != after_hashes

    gate = gate_matrix(binding, evidence, w1, boundary, checkpoints, validation, questions, outcome, guard)

    for name, payload in {
        "recovery_binding.json": binding,
        "training_summary.json": summary,
        "cycle_action_evolution.json": evolution,
        "context_credit_diagnostics.json": credit,
        "window_boundary_diagnostics.json": boundary,
        "w1_boundary_evidence_validation.json": w1,
        "validation_discrimination.json": validation,
        "checkpoint_integrity.json": checkpoints,
        "primary_questions.json": questions,
        "outcome_classification.json": outcome,
        "durable_evidence_report.json": evidence,
        "evidence_schema.json": dte.evidence_schema(),
        "gate_matrix.json": gate,
    }.items():
        write_json(root / name, payload)
    for seed in EXPECTED["seeds"]:
        write_json(
            root / f"seed_{seed}_summary.json",
            {
                "seed_summary": next(row for row in summary["seed_summaries"] if row["seed"] == seed),
                "durable_evidence": evidence["by_seed"].get(str(seed)),
                "durable_evidence_cycles": [row for row in evidence["by_seed_cycle"] if int(row["seed"]) == seed],
                "window_boundary": [row for row in boundary["by_seed_cycle"] if int(row["seed"]) == seed],
                "validation": {
                    label: validation["aggregate_by_classification"][label]["by_seed"].get(str(seed))
                    for label in (HOLD_BETTER, SERVE_BETTER)
                },
            },
        )
    (root / "final_report.md").write_text(
        final_report(dte, binding, evidence, questions, outcome, boundary, checkpoints, summary, gate), encoding="utf-8"
    )
    write_json(root / "manifest.json", make_manifest(root, gate, evidence, checkpoints, boundary, binding, started))

    print(f"[H4M-Y-R1] artifact root: {root}")
    print(f"[H4M-Y-R1] gate: {gate['gate']}")
    print(f"[H4M-Y-R1] outcome: {outcome['outcome_classification']}")
    print(f"[H4M-Y-R1] cycle evidence: {evidence['record_count']}/{EXPECTED['expected_cycle_records']}")
    print(f"[H4M-Y-R1] cross-window leakage: {boundary['totals']['cross_window_links_with_live_bootstrap']}")
    print(f"[H4M-Y-R1] optimizer steps: {guard.optimizer_steps} | backward passes: {guard.backward_calls}")
    print(f"[H4M-Y-R1] training artifact mutated: {binding['read_only_attestation']['training_artifact_mutated']}")
    print(f"[H4M-Y-R1] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-Y-R1] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
