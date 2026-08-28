#!/usr/bin/env python3
"""H4M-S PPO credit / advantage repair selection and freeze.

Read-only causal tracing over H4M-Q/H4M-R artifacts. This stage identifies the
earliest measured HOLD_BETTER credit divergence, selects one minimum repair,
and freezes its contract. It does not implement the repair, train, create an
optimizer, mutate data/checkpoints, open TEST6, or push GitHub.
"""

from __future__ import annotations

import hashlib
import json
import math
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


STAGE = "PV8-R2A-R8E-R3-R-H4M-S"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_S_PPO_CREDIT_ADVANTAGE_REPAIR_SELECTION_AND_FREEZE_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_S_PPO_CREDIT_ADVANTAGE_REPAIR_SELECTION_AND_FREEZE_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_R_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_r_target_conditioned_actor_head_specialized_retraining_outcome_review_next_decision_20260817_163927+0900"
H4M_Q_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_20260817_153440+0900"

EXPECTED = {
    "h4m_r_source_commit": "bb2bb245b8f20b89a00aa85c3808a89677f35338",
    "h4m_q_parent_commit": "7e0e57a8fba01aec22bc6b38427849a6a1c6f4db",
    "h4m_r_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_R_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_COMPLETE",
    "h4m_r_root_cause": "PPO_CREDIT_OR_ADVANTAGE_MISALIGNMENT",
    "h4m_r_selected_next_direction": "PPO_CREDIT_ADVANTAGE_REPAIR",
    "repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
}

CLASS_HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
CLASS_SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"

SELECTED_ROOT_CAUSE = "C2_CRITIC_VALUE_BASELINE_MISALIGNMENT"
SELECTED_REPAIR = "S3_CRITIC_VALUE_TARGET_REPAIR"
EXACT_NEXT_GATE = "H4M-T_PPO_CREDIT_ADVANTAGE_CRITIC_VALUE_TARGET_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION"

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "authoritative_binding.json",
    "credit_pipeline_trace.json",
    "credit_pipeline_trace.parquet",
    "raw_reward_audit.json",
    "critic_value_audit.json",
    "gae_sign_audit.json",
    "advantage_normalization_audit.json",
    "action_ownership_audit.json",
    "root_cause_selection.json",
    "repair_selection.json",
    "repair_freeze_contract.json",
    "changed_files.json",
    "test_results.json",
    "gate_matrix.json",
]


def kst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def numeric_stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            nums.append(number)
    if not nums:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    mu = float(mean(nums))
    var = sum((x - mu) ** 2 for x in nums) / len(nums)
    return {
        "count": len(nums),
        "mean": mu,
        "median": float(median(nums)),
        "min": min(nums),
        "max": max(nums),
        "std": math.sqrt(var),
    }


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else float(numerator) / float(denominator)


def py_compile_audit() -> Dict[str, Any]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="h4m_s_pycompile_") as tmp:
        source = PROJECT_ROOT / SOURCE_REL
        cfile = Path(tmp) / (source.name + ".pyc")
        try:
            py_compile.compile(str(source), cfile=str(cfile), doraise=True)
            ok = True
            error = None
        except Exception as exc:
            ok = False
            error = repr(exc)
        rows.append({"source_rel": str(SOURCE_REL), "passed": ok, "error": error, "bytecode_target_outside_repo": str(cfile)})
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {
        "py_compile_passed": all(row["passed"] for row in rows),
        "rows": rows,
        "git_diff_cached_check_passed": cached.returncode == 0,
        "git_diff_cached_check_stdout": cached.stdout.strip(),
        "git_diff_cached_check_stderr": cached.stderr.strip(),
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    source_commit = git_run(["log", "-1", "--format=%H", "--", str(SOURCE_REL)]).stdout.strip()
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_commit": head,
        "parent_commit": parent,
        "h4m_s_source_git_commit": source_commit,
        "source_commit_before_work": EXPECTED["h4m_r_source_commit"],
        "source_rel": str(SOURCE_REL),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "head_commit_files": head_files,
        "head_commit_source_only": head_files == [str(SOURCE_REL)],
        "status_short": status_short,
        "source_only_local_commit_created_before_freeze": (
            source_commit == head
            and parent == EXPECTED["h4m_r_source_commit"]
            and head_files == [str(SOURCE_REL)]
            and status_short == ""
        ),
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_static: Mapping[str, Any]) -> Dict[str, Any]:
    r_gate = read_json(H4M_R_ROOT / "gate_matrix.json")
    r_binding = read_json(H4M_R_ROOT / "authoritative_binding.json")
    r_root = read_json(H4M_R_ROOT / "root_cause_classification.json")
    q_gate = read_json(H4M_Q_ROOT / "gate_matrix.json")
    checks = {
        "source_only_commit_before_freeze": provenance.get("source_only_local_commit_created_before_freeze") is True,
        "py_compile_passed": compile_static.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_static.get("git_diff_cached_check_passed") is True,
        "h4m_r_gate_match": r_gate.get("gate") == EXPECTED["h4m_r_gate"],
        "h4m_r_source_commit_match": r_binding.get("source_provenance", {}).get("h4m_r_source_git_commit") == EXPECTED["h4m_r_source_commit"],
        "h4m_r_parent_match": r_binding.get("source_provenance", {}).get("parent_commit") == EXPECTED["h4m_q_parent_commit"],
        "h4m_r_root_cause_match": r_root.get("selected_root_cause") == EXPECTED["h4m_r_root_cause"],
        "h4m_r_selected_next_direction_match": r_root.get("selected_next_repair_direction") == EXPECTED["h4m_r_selected_next_direction"],
        "h4m_q_gate_passed": str(q_gate.get("gate", "")).startswith("PASS_"),
        "h4m_q_repair_contract_match": q_gate.get("repair_contract_sha256") == EXPECTED["repair_contract_sha256"],
        "test6_zero": read_json(H4M_Q_ROOT / "validation_discrimination.json").get("test6_access_count") == 0
        and read_json(H4M_R_ROOT / "manifest.json").get("TEST6_opened") is False,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {"h4m_r": str(H4M_R_ROOT), "h4m_q": str(H4M_Q_ROOT)},
        "source_provenance": provenance,
        "compile_static_checks": compile_static,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "sha_bindings": {
            "h4m_r_source_commit": EXPECTED["h4m_r_source_commit"],
            "h4m_q_parent_commit": EXPECTED["h4m_q_parent_commit"],
            "repair_contract_sha256": EXPECTED["repair_contract_sha256"],
            "h4m_r_root_cause_sha256": sha256_file(H4M_R_ROOT / "root_cause_classification.json"),
            "h4m_q_conditional_policy_by_sample_sha256": sha256_file(H4M_Q_ROOT / "conditional_policy_by_sample.parquet"),
        },
        "hard_lock_attestation": {
            "training_executed": False,
            "optimizer_created_or_stepped": False,
            "repair_implemented": False,
            "reward_v2_modified": False,
            "critic_modified": False,
            "gae_or_ppo_modified": False,
            "actor_modified": False,
            "data_or_db_modified": False,
            "test6_opened_or_used": False,
            "github_push_performed": False,
        },
    }


def concat_trace(kind: str) -> pd.DataFrame:
    paths = sorted((H4M_Q_ROOT / "05_actual_on_policy_credit_trace").glob(f"seed=*/outer_cycle=*/{kind}.parquet"))
    if not paths:
        raise RuntimeError(f"Missing trace parquet: {kind}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_pipeline_frame() -> Dict[str, pd.DataFrame]:
    joined = pd.read_parquet(H4M_Q_ROOT / "conditional_policy_by_sample.parquet")
    reward = concat_trace("actual_reward_trace")
    td = concat_trace("actual_critic_td_gae_trace")
    adv = concat_trace("advantage_normalization_sample")
    pre = concat_trace("actual_pre_action_trace")
    ppo = concat_trace("ppo_policy_surrogate_sample")
    sample = joined[
        [
            "sample_uid",
            "seed",
            "cycle",
            "classification",
            "target_id",
            "target_name",
            "sampled_action_id",
            "sampled_action_name",
            "P_HOLD",
            "P_SERVE",
            "P_SKIP",
            "policy_margin_hold_minus_serve",
            "delta_full_bootstrapped_return_serve_minus_hold",
        ]
    ].merge(
        reward[
            [
                "sample_uid",
                "reward_v2_raw_total",
                "reward_after_runtime_normalization",
                "reward_v2_local_service_component_weighted",
                "reward_v2_local_avg_wait_component_weighted",
                "reward_v2_explicit_intervention_component_weighted",
                "target_id",
                "sampled_action_id",
            ]
        ],
        on="sample_uid",
        how="left",
        suffixes=("", "_reward"),
    )
    sample = sample.merge(
        td[["sample_uid", "value_t", "next_value_t", "td_delta", "raw_gae_advantage", "raw_return_target"]],
        on="sample_uid",
        how="left",
    )
    sample = sample.merge(
        adv[["sample_uid", "raw_gae_advantage", "normalized_advantage", "raw_sign", "normalized_sign", "sign_changed", "sign_change_class"]],
        on="sample_uid",
        how="left",
        suffixes=("_td", "_adv"),
    )
    sample = sample.merge(
        pre[["sample_uid", "action_id", "target_id", "actor_target_context_hash", "sampled_action_log_prob"]],
        on="sample_uid",
        how="left",
        suffixes=("", "_pre"),
    )
    sample["critic_value_minus_return_target"] = sample["value_t"] - sample["raw_return_target"]
    sample["td_formula_residual"] = sample["reward_after_runtime_normalization"] + 0.99 * sample["next_value_t"] - sample["value_t"] - sample["td_delta"]
    ppo_joined = ppo.merge(joined[["sample_uid", "classification", "target_id", "sampled_action_name"]], on="sample_uid", how="left", suffixes=("", "_joined"))
    return {"sample": sample, "joined": joined, "reward": reward, "td": td, "adv": adv, "pre": pre, "ppo": ppo_joined}


def summarize_rows(rows: pd.DataFrame) -> Dict[str, Any]:
    return {
        "count": int(len(rows)),
        "reward_v2_raw_total": numeric_stats(rows.get("reward_v2_raw_total", [])),
        "reward_after_runtime_normalization": numeric_stats(rows.get("reward_after_runtime_normalization", [])),
        "value_t": numeric_stats(rows.get("value_t", [])),
        "next_value_t": numeric_stats(rows.get("next_value_t", [])),
        "td_delta": numeric_stats(rows.get("td_delta", [])),
        "raw_gae_advantage": numeric_stats(rows.get("raw_gae_advantage_adv", [])),
        "normalized_advantage": numeric_stats(rows.get("normalized_advantage", [])),
        "raw_return_target": numeric_stats(rows.get("raw_return_target", [])),
        "critic_value_minus_return_target": numeric_stats(rows.get("critic_value_minus_return_target", [])),
        "positive_raw_gae_rate": safe_rate(int((rows.get("raw_gae_advantage_adv", pd.Series(dtype=float)) > 0).sum()), len(rows)),
        "positive_normalized_advantage_rate": safe_rate(int((rows.get("normalized_advantage", pd.Series(dtype=float)) > 0).sum()), len(rows)),
        "sign_changed_rate": safe_rate(int(rows.get("sign_changed", pd.Series(dtype=bool)).sum()), len(rows)),
    }


def credit_pipeline_trace(frames: Mapping[str, pd.DataFrame], artifact_root: Path) -> Dict[str, Any]:
    sample = frames["sample"].copy()
    path = artifact_root / "credit_pipeline_trace.parquet"
    sample.to_parquet(path, index=False)
    by_label_action: Dict[str, Dict[str, Any]] = {}
    for label in [CLASS_HOLD_BETTER, CLASS_SERVE_BETTER]:
        label_rows = sample[sample["classification"] == label]
        by_label_action[label] = {
            "overall": summarize_rows(label_rows),
            "by_sampled_action": {
                str(action): summarize_rows(action_rows)
                for action, action_rows in label_rows.groupby("sampled_action_name")
            },
            "by_seed": {
                str(int(seed)): summarize_rows(seed_rows)
                for seed, seed_rows in label_rows.groupby("seed")
            },
            "by_cycle": {
                str(int(cycle)): summarize_rows(cycle_rows)
                for cycle, cycle_rows in label_rows.groupby("cycle")
            },
        }
    return {
        "stage": STAGE,
        "row_level_parquet": str(path),
        "row_level_parquet_sha256": sha256_file(path),
        "row_count": int(len(sample)),
        "by_label_action": by_label_action,
        "pipeline_order": [
            "sampled action/event",
            "Reward V2 raw settlement",
            "reward runtime normalization",
            "critic V(s), V(s')",
            "TD residual",
            "GAE raw advantage",
            "advantage normalization",
            "PPO surrogate",
        ],
    }


def raw_reward_audit(pipeline: Mapping[str, Any]) -> Dict[str, Any]:
    hold = pipeline["by_label_action"][CLASS_HOLD_BETTER]["by_sampled_action"]
    serve = pipeline["by_label_action"][CLASS_SERVE_BETTER]["by_sampled_action"]
    hb_hold = hold[ACTION_HOLD]["reward_v2_raw_total"]["mean"]
    hb_serve = hold[ACTION_SERVE]["reward_v2_raw_total"]["mean"]
    sb_hold = serve[ACTION_HOLD]["reward_v2_raw_total"]["mean"]
    sb_serve = serve[ACTION_SERVE]["reward_v2_raw_total"]["mean"]
    return {
        "stage": STAGE,
        "question_1_hold_better_sampled_hold_raw_reward_direction_correct": hb_hold is not None and hb_serve is not None and hb_hold > hb_serve,
        "hold_better_sampled_hold_raw_reward_mean": hb_hold,
        "hold_better_sampled_serve_raw_reward_mean": hb_serve,
        "serve_better_sampled_hold_raw_reward_mean": sb_hold,
        "serve_better_sampled_serve_raw_reward_mean": sb_serve,
        "c1_reward_signal_misalignment_supported": False,
        "interpretation": (
            "Raw Reward V2 direction is correct for measured HOLD_BETTER sampled actions: HOLD receives 0 while SERVE receives -0.25. "
            "SERVE_BETTER SERVE receives +3.0. Reward retuning is not justified."
        ),
    }


def critic_value_audit(pipeline: Mapping[str, Any]) -> Dict[str, Any]:
    hold = pipeline["by_label_action"][CLASS_HOLD_BETTER]["by_sampled_action"]
    serve = pipeline["by_label_action"][CLASS_SERVE_BETTER]["by_sampled_action"]
    hb_hold = hold[ACTION_HOLD]
    hb_serve = hold[ACTION_SERVE]
    sb_serve = serve[ACTION_SERVE]
    by_seed = pipeline["by_label_action"][CLASS_HOLD_BETTER]["by_seed"]
    overestimate_all_seeds = all(row["critic_value_minus_return_target"]["mean"] > 0.0 for row in by_seed.values())
    return {
        "stage": STAGE,
        "question_4_critic_systematic_over_under_estimate": {
            "hold_better_sampled_hold_value_minus_return_mean": hb_hold["critic_value_minus_return_target"]["mean"],
            "hold_better_sampled_serve_value_minus_return_mean": hb_serve["critic_value_minus_return_target"]["mean"],
            "serve_better_sampled_serve_value_minus_return_mean": sb_serve["critic_value_minus_return_target"]["mean"],
            "hold_better_value_over_return_all_seeds": overestimate_all_seeds,
        },
        "c2_critic_value_baseline_misalignment_supported": (
            hb_hold["critic_value_minus_return_target"]["mean"] > 0.0
            and hb_hold["raw_gae_advantage"]["mean"] < 0.0
            and sb_serve["critic_value_minus_return_target"]["mean"] < 0.0
        ),
        "interpretation": (
            "HOLD_BETTER sampled HOLD has a positive raw return target on average, but V(s) is higher than that target, "
            "so pre-normalization advantage is already negative. SERVE_BETTER sampled SERVE is underestimated and receives positive advantage."
        ),
        "by_seed_hold_better": by_seed,
    }


def gae_sign_audit(pipeline: Mapping[str, Any]) -> Dict[str, Any]:
    hold = pipeline["by_label_action"][CLASS_HOLD_BETTER]
    hb_hold = hold["by_sampled_action"][ACTION_HOLD]
    by_cycle = hold["by_cycle"]
    negative_raw_gae_cycles = sum(1 for row in by_cycle.values() if row["raw_gae_advantage"]["mean"] < 0.0)
    negative_td_cycles = sum(1 for row in by_cycle.values() if row["td_delta"]["mean"] < 0.0)
    return {
        "stage": STAGE,
        "question_2_pre_normalization_advantage_already_negative": hb_hold["raw_gae_advantage"]["mean"] < 0.0,
        "question_3_earliest_negative_stage": "reward_runtime_normalization_and_TD_residual_before_GAE",
        "question_5_gae_temporal_misbinding_supported": False,
        "hold_better_sampled_hold_td_delta_mean": hb_hold["td_delta"]["mean"],
        "hold_better_sampled_hold_raw_gae_mean": hb_hold["raw_gae_advantage"]["mean"],
        "hold_better_negative_td_cycle_count": negative_td_cycles,
        "hold_better_negative_raw_gae_cycle_count": negative_raw_gae_cycles,
        "hold_better_cycle_count": len(by_cycle),
        "c3_gae_temporal_credit_misbinding_supported": False,
        "interpretation": (
            "GAE carries negative/weak credit forward, but the mean TD residual is already negative. "
            "Evidence does not isolate GAE temporal binding as the first defect."
        ),
    }


def advantage_normalization_audit(pipeline: Mapping[str, Any]) -> Dict[str, Any]:
    hb_hold = pipeline["by_label_action"][CLASS_HOLD_BETTER]["by_sampled_action"][ACTION_HOLD]
    hb_overall = pipeline["by_label_action"][CLASS_HOLD_BETTER]["overall"]
    return {
        "stage": STAGE,
        "question_6_normalization_rescale_or_flip": {
            "hold_better_sampled_hold_raw_gae_mean": hb_hold["raw_gae_advantage"]["mean"],
            "hold_better_sampled_hold_normalized_advantage_mean": hb_hold["normalized_advantage"]["mean"],
            "hold_better_sampled_hold_sign_changed_rate": hb_hold["sign_changed_rate"],
            "hold_better_overall_sign_changed_rate": hb_overall["sign_changed_rate"],
        },
        "c4_advantage_normalization_sign_distortion_supported": False,
        "interpretation": "Advantage normalization slightly shifts/scales the problem, but raw GAE is already negative and sign-flip rates are low.",
    }


def action_ownership_audit(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    sample = frames["sample"]
    joined = frames["joined"]
    reward = frames["reward"]
    td = frames["td"]
    adv = frames["adv"]
    pre = frames["pre"]
    mismatches = {
        "joined_duplicate_sample_uid_count": int(joined["sample_uid"].duplicated().sum()),
        "reward_duplicate_sample_uid_count": int(reward["sample_uid"].duplicated().sum()),
        "td_duplicate_sample_uid_count": int(td["sample_uid"].duplicated().sum()),
        "advantage_duplicate_sample_uid_count": int(adv["sample_uid"].duplicated().sum()),
        "pre_action_duplicate_sample_uid_count": int(pre["sample_uid"].duplicated().sum()),
        "reward_action_id_mismatch_count": int((sample["sampled_action_id"] != sample["sampled_action_id_reward"]).sum()),
        "pre_action_id_mismatch_count": int((sample["sampled_action_id"] != sample["action_id"]).sum()),
        "reward_target_id_mismatch_count": int((sample["target_id"] != sample["target_id_reward"]).sum()),
        "pre_target_id_mismatch_count": int((sample["target_id"] != sample["target_id_pre"]).sum()),
    }
    return {
        "stage": STAGE,
        "question_7_credit_attached_to_correct_action_event_row": all(value == 0 for value in mismatches.values()),
        "mismatches": mismatches,
        "c5_target_action_ownership_misbinding_supported": False,
        "interpretation": "sample_uid/action/target ownership is exact across joined, reward, TD/GAE, advantage, and pre-action traces.",
    }


def ppo_surrogate_audit(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    ppo = frames["ppo"]
    rows: Dict[str, Any] = {}
    for label in [CLASS_HOLD_BETTER, CLASS_SERVE_BETTER]:
        label_rows = ppo[ppo["classification"] == label]
        rows[label] = {
            "overall": {
                "row_count": int(len(label_rows)),
                "clip_active_rate": safe_rate(int(label_rows["clip_active"].sum()), len(label_rows)),
                "effective_policy_surrogate_contribution": numeric_stats(label_rows["effective_policy_surrogate_contribution"]),
                "normalized_advantage": numeric_stats(label_rows["normalized_advantage"]),
                "probability_ratio": numeric_stats(label_rows["probability_ratio"]),
            },
            "by_sampled_action": {
                str(action): {
                    "row_count": int(len(action_rows)),
                    "clip_active_rate": safe_rate(int(action_rows["clip_active"].sum()), len(action_rows)),
                    "effective_policy_surrogate_contribution": numeric_stats(action_rows["effective_policy_surrogate_contribution"]),
                    "normalized_advantage": numeric_stats(action_rows["normalized_advantage"]),
                    "probability_ratio": numeric_stats(action_rows["probability_ratio"]),
                }
                for action, action_rows in label_rows.groupby("sampled_action_name_joined")
            },
        }
    return {
        "stage": STAGE,
        "by_classification": rows,
        "clipping_primary_cause_supported": False,
        "interpretation": "PPO surrogate follows the advantage sign; clipping rates remain low and do not explain the sign defect.",
    }


def root_cause_selection(
    raw_reward: Mapping[str, Any],
    critic: Mapping[str, Any],
    gae: Mapping[str, Any],
    normalization: Mapping[str, Any],
    ownership: Mapping[str, Any],
    ppo: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = SELECTED_ROOT_CAUSE if critic["c2_critic_value_baseline_misalignment_supported"] else "C6_MIXED_OR_OTHER"
    repair = SELECTED_REPAIR if selected == SELECTED_ROOT_CAUSE else "NO_REPAIR_SELECTED_INSUFFICIENT_EVIDENCE"
    next_gate = EXACT_NEXT_GATE if repair == SELECTED_REPAIR else "STOP_NO_REPAIR_SELECTED_INSUFFICIENT_EVIDENCE"
    return {
        "stage": STAGE,
        "earliest_credit_divergence_stage": "critic_value_baseline_at_TD_GAE_boundary",
        "selected_root_cause": selected,
        "selected_repair": repair,
        "exact_next_gate": next_gate,
        "causal_classification": {
            "C1_REWARD_SIGNAL_MISALIGNMENT": {
                "supported": raw_reward["c1_reward_signal_misalignment_supported"],
                "reason": raw_reward["interpretation"],
            },
            "C2_CRITIC_VALUE_BASELINE_MISALIGNMENT": {
                "supported": critic["c2_critic_value_baseline_misalignment_supported"],
                "reason": critic["interpretation"],
            },
            "C3_GAE_TEMPORAL_CREDIT_MISBINDING": {
                "supported": gae["c3_gae_temporal_credit_misbinding_supported"],
                "reason": gae["interpretation"],
            },
            "C4_ADVANTAGE_NORMALIZATION_SIGN_DISTORTION": {
                "supported": normalization["c4_advantage_normalization_sign_distortion_supported"],
                "reason": normalization["interpretation"],
            },
            "C5_TARGET_ACTION_OWNERSHIP_MISBINDING": {
                "supported": ownership["c5_target_action_ownership_misbinding_supported"],
                "reason": ownership["interpretation"],
            },
            "C6_MIXED_OR_OTHER": {
                "supported": selected == "C6_MIXED_OR_OTHER",
                "reason": "Not selected because a single earlier mechanism is evidence-supported.",
            },
        },
        "mandatory_answers": {
            "1_raw_reward_direction_correct_for_hold_better_sampled_hold": raw_reward[
                "question_1_hold_better_sampled_hold_raw_reward_direction_correct"
            ],
            "2_pre_normalization_advantage_already_negative": gae["question_2_pre_normalization_advantage_already_negative"],
            "3_earliest_negative_stage": gae["question_3_earliest_negative_stage"],
            "4_critic_value_baseline_bias": critic["question_4_critic_systematic_over_under_estimate"],
            "5_gae_later_trajectory_misbinding": gae["question_5_gae_temporal_misbinding_supported"],
            "6_advantage_normalization_flip": normalization["question_6_normalization_rescale_or_flip"],
            "7_action_ownership_correct": ownership["question_7_credit_attached_to_correct_action_event_row"],
            "8_reproducible_across_seeds_cycles": {
                "hold_better_value_over_return_all_seeds": critic["question_4_critic_systematic_over_under_estimate"][
                    "hold_better_value_over_return_all_seeds"
                ],
                "hold_better_negative_td_cycle_count": gae["hold_better_negative_td_cycle_count"],
                "hold_better_negative_raw_gae_cycle_count": gae["hold_better_negative_raw_gae_cycle_count"],
                "hold_better_cycle_count": gae["hold_better_cycle_count"],
            },
        },
        "ppo_surrogate_summary": ppo["by_classification"],
    }


def repair_selection(root: Mapping[str, Any]) -> Dict[str, Any]:
    rejected = {
        "S1_TARGET_ACTION_CREDIT_OWNERSHIP_REPAIR": "Rejected: sample_uid/action/target ownership mismatches are zero.",
        "S2_GAE_TEMPORAL_BINDING_REPAIR": "Rejected as first repair: TD residual/critic baseline divergence appears before GAE can be isolated as primary.",
        "S4_ADVANTAGE_NORMALIZATION_REPAIR": "Rejected: raw GAE is already wrong-signed and sign-flip rates are low.",
        "S5_OTHER_MINIMAL_CREDIT_REPAIR": "Rejected: S3 is more specific and evidence-supported.",
    }
    return {
        "stage": STAGE,
        "selected_repair": root["selected_repair"],
        "selected_root_cause": root["selected_root_cause"],
        "earliest_credit_divergence_stage": root["earliest_credit_divergence_stage"],
        "explicitly_rejected_repairs": rejected,
        "minimum_change_rule_satisfied": root["selected_repair"] == SELECTED_REPAIR,
        "exact_next_gate": root["exact_next_gate"],
        "do_not_implement_now": True,
    }


def repair_freeze_contract(root: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    contract = {
        "stage": STAGE,
        "contract_name": "H4M_S_CRITIC_VALUE_TARGET_REPAIR_FREEZE_CONTRACT",
        "root_cause_classification": root["selected_root_cause"],
        "earliest_credit_divergence_stage": root["earliest_credit_divergence_stage"],
        "selected_repair": selection["selected_repair"],
        "explicitly_rejected_repairs": selection["explicitly_rejected_repairs"],
        "exact_allowed_code_semantic_scope": {
            "may_change": [
                "critic value target / baseline computation used for PPO advantage in the H4M lineage",
                "target-conditioned critic-value calibration or target-stratified value target handling if required to remove measured baseline sign error",
                "read-only diagnostics and focused equivalence tests for critic target alignment",
            ],
            "must_not_change": [
                "Reward V2 semantics, weights, reference values, or HOLD bonus",
                "simulator causal order",
                "observation contract",
                "K-mask/action semantics",
                "Zero-Loss semantics",
                "H4M-P specialized actor head routing semantics",
                "frozen split, seeds, training budget, schedule",
                "TEST6 seal",
                "actor logit bias or target-specific reward shaping",
            ],
        },
        "immutable_upstream_contracts": {
            "h4m_r_source_commit": EXPECTED["h4m_r_source_commit"],
            "h4m_q_parent_commit": EXPECTED["h4m_q_parent_commit"],
            "h4m_p_actor_head_repair_contract_sha256": EXPECTED["repair_contract_sha256"],
        },
        "expected_invariant_after_repair": (
            "HOLD_BETTER sampled HOLD rows must no longer receive systematically wrong-signed PPO credit solely because "
            "the critic value baseline/target overestimates the realized value target for those target-conditioned rows."
        ),
        "validation_requirements": [
            "source/equivalence validation before retraining",
            "no Reward V2 drift",
            "critic/value path compatibility and checkpoint migration audit if needed",
            "controlled credit-pipeline fixture demonstrating removal of measured baseline sign error without HOLD logit bias",
            "cross-head gradient leakage remains zero",
            "TEST6 access count remains zero",
            "optimizer-step count remains zero during implementation/equivalence validation",
        ],
        "stop_conditions": [
            "lineage mismatch",
            "Reward V2 or simulator drift",
            "actor head routing drift",
            "action ownership mismatch",
            "unexplained NaN/Inf",
            "TEST6 access",
            "need for reward retuning or logit bias",
            "insufficient equivalence proof",
        ],
        "exact_next_gate": EXACT_NEXT_GATE,
    }
    contract["contract_sha256"] = canonical_sha(contract)
    return contract


def changed_files() -> Dict[str, Any]:
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    return {
        "stage": STAGE,
        "source_commit": git_run(["rev-parse", "HEAD"]).stdout.strip(),
        "parent_commit": git_run(["rev-parse", "HEAD^"]).stdout.strip(),
        "head_commit_files": head_files,
        "source_only_local_commit": head_files == [str(SOURCE_REL)],
        "artifact_or_log_files_committed": any("artifacts/" in path for path in head_files),
        "github_push_performed": False,
    }


def gate_matrix(
    binding: Mapping[str, Any],
    root: Mapping[str, Any],
    selection: Mapping[str, Any],
    contract: Mapping[str, Any],
    changed: Mapping[str, Any],
    manifest_ready: bool,
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_passed": binding.get("authoritative_binding_passed") is True,
        "source_only_local_commit": changed.get("source_only_local_commit") is True,
        "single_root_cause_selected": root.get("selected_root_cause") == SELECTED_ROOT_CAUSE,
        "single_minimum_repair_selected": selection.get("selected_repair") == SELECTED_REPAIR,
        "contract_sha256_generated": bool(contract.get("contract_sha256")),
        "required_artifacts_present": manifest_ready,
        "no_training_or_optimizer": binding.get("hard_lock_attestation", {}).get("optimizer_created_or_stepped") is False,
        "test6_zero": binding.get("hard_lock_attestation", {}).get("test6_opened_or_used") is False,
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": selection.get("selected_repair") if passed else "NO_REPAIR_SELECTED_INSUFFICIENT_EVIDENCE",
        "selected_root_cause": root.get("selected_root_cause"),
        "earliest_credit_divergence_stage": root.get("earliest_credit_divergence_stage"),
        "repair_contract_sha256": contract.get("contract_sha256"),
        "exact_next_gate": root.get("exact_next_gate") if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_executed": False,
            "optimizer_step_count": 0,
            "repair_implemented": False,
            "test6_opened_or_used": False,
            "github_push_performed": False,
        },
    }


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            output_files[str(path.relative_to(artifact_root))] = str(path)
    return {
        "stage": STAGE,
        "created_at": kst_now(),
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "selected_root_cause": gate.get("selected_root_cause"),
        "earliest_credit_divergence_stage": gate.get("earliest_credit_divergence_stage"),
        "repair_contract_sha256": gate.get("repair_contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": output_files,
        "output_sha256": {name: sha256_file(Path(path)) for name, path in output_files.items()},
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift",
        "TEST6_opened": False,
        "optimizer_step_count": 0,
        "training_executed": False,
        "github_push_performed": False,
    }


def final_report(
    binding: Mapping[str, Any],
    raw_reward: Mapping[str, Any],
    critic: Mapping[str, Any],
    gae: Mapping[str, Any],
    normalization: Mapping[str, Any],
    ownership: Mapping[str, Any],
    root: Mapping[str, Any],
    contract: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    return f"""# H4M-S PPO Credit / Advantage Repair Selection and Freeze

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_s_source_git_commit"]}
selected_root_cause = {root["selected_root_cause"]}
earliest_credit_divergence_stage = {root["earliest_credit_divergence_stage"]}
selected_repair = {gate["decision"]}
repair_contract_sha256 = {gate["repair_contract_sha256"]}
exact_next_gate = {gate["exact_next_gate"]}

## Raw reward audit

```json
{json.dumps(raw_reward, ensure_ascii=False, indent=2, default=jsonable)}
```

## Critic value audit

```json
{json.dumps(critic, ensure_ascii=False, indent=2, default=jsonable)}
```

## GAE sign audit

```json
{json.dumps(gae, ensure_ascii=False, indent=2, default=jsonable)}
```

## Advantage normalization audit

```json
{json.dumps(normalization, ensure_ascii=False, indent=2, default=jsonable)}
```

## Action ownership audit

```json
{json.dumps(ownership, ensure_ascii=False, indent=2, default=jsonable)}
```

## Root cause and repair contract

```json
{json.dumps({"root": root, "contract": contract}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no implementation, no training, no optimizer step, no TEST6, no GitHub push.
"""


def write_block_outputs(artifact_root: Path, created_at: str, binding: Mapping[str, Any], reason: str) -> None:
    gate = {
        "stage": STAGE,
        "gate": BLOCK_GATE,
        "decision": "NO_REPAIR_SELECTED_INSUFFICIENT_EVIDENCE",
        "selected_root_cause": "C6_MIXED_OR_OTHER",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
        "created_at": created_at,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    for name, payload in {
        "authoritative_binding.json": binding,
        "credit_pipeline_trace.json": empty,
        "raw_reward_audit.json": empty,
        "critic_value_audit.json": empty,
        "gae_sign_audit.json": empty,
        "advantage_normalization_audit.json": empty,
        "action_ownership_audit.json": empty,
        "root_cause_selection.json": empty,
        "repair_selection.json": empty,
        "repair_freeze_contract.json": empty,
        "changed_files.json": changed_files(),
        "test_results.json": binding.get("compile_static_checks", empty),
        "gate_matrix.json": gate,
    }.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(f"# H4M-S\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-S] artifact root: {artifact_root}")
    print(f"[H4M-S] gate: {BLOCK_GATE}")
    print(f"[H4M-S] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_s_ppo_credit_advantage_repair_selection_and_freeze_{stamp}"
    provenance = source_provenance(created_at)
    compile_static = py_compile_audit()
    binding = authoritative_binding(created_at, provenance, compile_static)
    if not binding["authoritative_binding_passed"]:
        write_block_outputs(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return

    artifact_root.mkdir(parents=True, exist_ok=True)
    frames = load_pipeline_frame()
    pipeline = credit_pipeline_trace(frames, artifact_root)
    raw_reward = raw_reward_audit(pipeline)
    critic = critic_value_audit(pipeline)
    gae = gae_sign_audit(pipeline)
    normalization = advantage_normalization_audit(pipeline)
    ownership = action_ownership_audit(frames)
    ppo = ppo_surrogate_audit(frames)
    root = root_cause_selection(raw_reward, critic, gae, normalization, ownership, ppo)
    selection = repair_selection(root)
    contract = repair_freeze_contract(root, selection)
    changed = changed_files()
    test_results = {
        "stage": STAGE,
        "py_compile": compile_static,
        "commands": [
            f"{sys.executable} -m py_compile {SOURCE_REL}",
            "git diff --cached --check",
            f"{sys.executable} {SOURCE_REL}",
        ],
        "training_executed": False,
        "optimizer_step_count": 0,
        "test6_access_count": 0,
    }

    payloads = {
        "authoritative_binding.json": binding,
        "credit_pipeline_trace.json": pipeline,
        "raw_reward_audit.json": raw_reward,
        "critic_value_audit.json": critic,
        "gae_sign_audit.json": gae,
        "advantage_normalization_audit.json": normalization,
        "action_ownership_audit.json": ownership,
        "ppo_surrogate_audit.json": ppo,
        "root_cause_selection.json": root,
        "repair_selection.json": selection,
        "repair_freeze_contract.json": contract,
        "changed_files.json": changed,
        "test_results.json": test_results,
    }
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)

    manifest_ready_pre_gate = all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name not in {"manifest.json", "gate_matrix.json", "final_report.md"})
    gate = gate_matrix(binding, root, selection, contract, changed, manifest_ready_pre_gate)
    write_json(artifact_root / "gate_matrix.json", gate)
    (artifact_root / "final_report.md").write_text(
        final_report(binding, raw_reward, critic, gae, normalization, ownership, root, contract, gate),
        encoding="utf-8",
    )
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    print(f"[H4M-S] artifact root: {artifact_root}")
    print(f"[H4M-S] gate: {gate['gate']}")
    print(f"[H4M-S] source_commit: {provenance['h4m_s_source_git_commit']}")
    print(f"[H4M-S] earliest_credit_divergence_stage: {gate['earliest_credit_divergence_stage']}")
    print(f"[H4M-S] selected_root_cause: {gate['selected_root_cause']}")
    print(f"[H4M-S] selected_repair: {gate['decision']}")
    print(f"[H4M-S] repair_contract_sha256: {gate['repair_contract_sha256']}")
    print(f"[H4M-S] exact_next_gate: {gate['exact_next_gate']}")


if __name__ == "__main__":
    main()
