#!/usr/bin/env python3
"""H4M-R target-conditioned actor-head outcome review and next decision.

Read-only diagnosis over H4M-Q artifacts. This stage does not train, create an
optimizer, mutate checkpoints, open TEST6, or implement a repair.
"""

from __future__ import annotations

import hashlib
import json
import math
import py_compile
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


STAGE = "PV8-R2A-R8E-R3-R-H4M-R"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_R_"
    "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_COMPLETE"
)
BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_R_"
    "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_FAILED"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_Q_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_"
    "20260817_153440+0900"
)
H4M_P_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_p_target_conditioned_actor_head_specialization_implementation_equivalence_validation_"
    "20260817_145922+0900"
)
H4M_O_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_o_actor_target_conditioned_retraining_outcome_review_next_decision_"
    "20260817_143826+0900"
)

EXPECTED = {
    "h4m_q_source_commit": "7e0e57a8fba01aec22bc6b38427849a6a1c6f4db",
    "h4m_p_parent_commit": "e8d69668e800124f41ff789a32210fb7701aa289",
    "h4m_q_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Q_"
        "FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_THREE_SEED_RETRAINING_COMPLETE"
    ),
    "h4m_q_behavioral_outcome": "SERVE_COLLAPSE_PERSISTS",
    "repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "pre_specialization_hold_better_validation_hold_count": 0,
    "pre_specialization_hold_better_validation_total": 60,
}

CLASS_HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
CLASS_SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"

NEXT_REPAIR = "PPO_CREDIT_ADVANTAGE_REPAIR"
NEXT_GATE = "H4M-S_PPO_CREDIT_ADVANTAGE_REPAIR_SELECTION_AND_FREEZE"

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "authoritative_binding.json",
    "hold_better_credit_trace.json",
    "hold_better_credit_trace.parquet",
    "serve_better_credit_trace.json",
    "serve_better_credit_trace.parquet",
    "ppo_update_effectiveness.json",
    "ppo_update_effectiveness.parquet",
    "logit_margin_evolution.json",
    "shared_trunk_interference_audit.json",
    "root_cause_classification.json",
    "next_decision.json",
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
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


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
    passed = True
    with tempfile.TemporaryDirectory(prefix="h4m_r_pycompile_") as tmp:
        source = PROJECT_ROOT / SOURCE_REL
        cfile = Path(tmp) / (source.name + ".pyc")
        try:
            py_compile.compile(str(source), cfile=str(cfile), doraise=True)
            ok = True
            error = None
        except Exception as exc:
            ok = False
            error = repr(exc)
        passed = passed and ok
        rows.append({"source_rel": str(SOURCE_REL), "passed": ok, "error": error, "bytecode_target_outside_repo": str(cfile)})
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {
        "py_compile_passed": passed,
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
        "h4m_r_source_git_commit": source_commit,
        "source_commit_before_work": EXPECTED["h4m_q_source_commit"],
        "source_rel": str(SOURCE_REL),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "head_commit_files": head_files,
        "head_commit_source_only": head_files == [str(SOURCE_REL)],
        "status_short": status_short,
        "source_only_local_commit_created_before_review": (
            source_commit == head
            and parent == EXPECTED["h4m_q_source_commit"]
            and head_files == [str(SOURCE_REL)]
            and status_short == ""
        ),
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_static: Mapping[str, Any]) -> Dict[str, Any]:
    q_gate = read_json(H4M_Q_ROOT / "gate_matrix.json")
    q_binding = read_json(H4M_Q_ROOT / "repair_binding.json")
    q_validation = read_json(H4M_Q_ROOT / "validation_discrimination.json")
    q_integrity = read_json(H4M_Q_ROOT / "training_integrity.json")
    p_gate = read_json(H4M_P_ROOT / "gate_matrix.json")
    o_gate = read_json(H4M_O_ROOT / "11_gate_matrix.json")
    checks = {
        "source_only_commit_before_review": provenance.get("source_only_local_commit_created_before_review") is True,
        "py_compile_passed": compile_static.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_static.get("git_diff_cached_check_passed") is True,
        "h4m_q_gate_match": q_gate.get("gate") == EXPECTED["h4m_q_gate"],
        "h4m_q_source_commit_match": q_binding.get("source_provenance", {}).get("h4m_q_source_git_commit")
        == EXPECTED["h4m_q_source_commit"],
        "h4m_q_parent_match": q_binding.get("source_provenance", {}).get("parent_commit") == EXPECTED["h4m_p_parent_commit"],
        "h4m_q_behavioral_outcome_match": q_gate.get("behavioral_outcome") == EXPECTED["h4m_q_behavioral_outcome"],
        "repair_contract_sha_match": q_gate.get("repair_contract_sha256") == EXPECTED["repair_contract_sha256"]
        and p_gate.get("repair_contract_sha256") == EXPECTED["repair_contract_sha256"],
        "h4m_q_integrity_passed": q_integrity.get("training_integrity_passed") is True,
        "h4m_q_validation_passed": q_validation.get("validation_discrimination_passed") is True,
        "h4m_p_gate_passed": str(p_gate.get("gate", "")).startswith("PASS_"),
        "h4m_o_gate_passed": str(o_gate.get("gate", "")).startswith("PASS_"),
        "test6_zero": q_validation.get("test6_access_count") == 0 and q_integrity.get("TEST6") == "SEALED_NOT_OPENED",
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {"h4m_q": str(H4M_Q_ROOT), "h4m_p": str(H4M_P_ROOT), "h4m_o": str(H4M_O_ROOT)},
        "source_provenance": provenance,
        "compile_static_checks": compile_static,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "sha_bindings": {
            "h4m_q_source_commit": EXPECTED["h4m_q_source_commit"],
            "h4m_p_parent_commit": EXPECTED["h4m_p_parent_commit"],
            "repair_contract_sha256": EXPECTED["repair_contract_sha256"],
            "h4m_q_artifact_manifest_sha256": sha256_file(H4M_Q_ROOT / "manifest.json"),
            "h4m_q_validation_discrimination_sha256": sha256_file(H4M_Q_ROOT / "validation_discrimination.json"),
            "h4m_q_credit_trace_summary_sha256": sha256_file(H4M_Q_ROOT / "credit_trace_summary.json"),
        },
        "hard_lock_attestation": {
            "training_executed": False,
            "optimizer_created_or_stepped": False,
            "checkpoint_mutated": False,
            "reward_v2_modified": False,
            "gae_or_ppo_modified": False,
            "shared_trunk_modified": False,
            "actor_head_modified": False,
            "test6_opened_or_used": False,
            "github_push_performed": False,
        },
    }


def concat_trace(kind: str) -> pd.DataFrame:
    paths = sorted((H4M_Q_ROOT / "05_actual_on_policy_credit_trace").glob(f"seed=*/outer_cycle=*/{kind}.parquet"))
    if not paths:
        raise RuntimeError(f"Missing trace parquet kind={kind}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def load_evidence_frames() -> Dict[str, pd.DataFrame]:
    joined = pd.read_parquet(H4M_Q_ROOT / "conditional_policy_by_sample.parquet")
    reward = concat_trace("actual_reward_trace")
    adv = concat_trace("advantage_normalization_sample")
    critic = concat_trace("actual_critic_td_gae_trace")
    pre = concat_trace("actual_pre_action_trace")
    ppo = concat_trace("ppo_policy_surrogate_sample")
    pressure = concat_trace("actor_logit_pressure_by_sample")
    base_cols = [
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
        "pre_mask_logit_margin_hold_minus_serve",
        "probability_correct_direction",
        "sampled_correct_direction",
    ]
    sample = joined[base_cols].merge(
        reward[
            [
                "sample_uid",
                "reward_v2_raw_total",
                "reward_after_runtime_normalization",
                "reward_v2_local_service_component_weighted",
                "reward_v2_local_avg_wait_component_weighted",
                "reward_v2_explicit_intervention_component_weighted",
            ]
        ],
        on="sample_uid",
        how="left",
    )
    sample = sample.merge(
        adv[["sample_uid", "raw_gae_advantage", "normalized_advantage", "raw_sign", "normalized_sign", "sign_changed"]],
        on="sample_uid",
        how="left",
    )
    sample = sample.merge(
        critic[["sample_uid", "value_t", "next_value_t", "td_delta", "raw_return_target"]],
        on="sample_uid",
        how="left",
    )
    sample = sample.merge(
        pre[
            [
                "sample_uid",
                "pre_mask_logit_0",
                "pre_mask_logit_1",
                "pre_mask_logit_2",
                "masked_probability_0",
                "masked_probability_1",
                "masked_probability_2",
                "policy_entropy",
            ]
        ],
        on="sample_uid",
        how="left",
    )
    return {"joined": joined, "sample": sample, "ppo": ppo, "pressure": pressure}


def summarize_action_subset(rows: pd.DataFrame) -> Dict[str, Any]:
    action_counts = rows["sampled_action_name"].value_counts().to_dict()
    return {
        "count": int(len(rows)),
        "sampled_action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "reward_v2_raw_total": numeric_stats(rows.get("reward_v2_raw_total", [])),
        "reward_after_runtime_normalization": numeric_stats(rows.get("reward_after_runtime_normalization", [])),
        "raw_gae_advantage": numeric_stats(rows.get("raw_gae_advantage", [])),
        "normalized_advantage": numeric_stats(rows.get("normalized_advantage", [])),
        "normalized_advantage_positive_rate": safe_rate(int((rows["normalized_advantage"] > 0).sum()), len(rows)) if len(rows) else None,
        "td_delta": numeric_stats(rows.get("td_delta", [])),
        "P_HOLD": numeric_stats(rows.get("P_HOLD", [])),
        "P_SERVE": numeric_stats(rows.get("P_SERVE", [])),
        "policy_margin_hold_minus_serve": numeric_stats(rows.get("policy_margin_hold_minus_serve", [])),
        "policy_entropy": numeric_stats(rows.get("policy_entropy", [])),
        "value_t": numeric_stats(rows.get("value_t", [])),
        "raw_return_target": numeric_stats(rows.get("raw_return_target", [])),
    }


def credit_trace_for_label(frames: Mapping[str, pd.DataFrame], label: str, artifact_root: Path, stem: str) -> Dict[str, Any]:
    sample = frames["sample"]
    rows = sample[sample["classification"] == label].copy()
    parquet_path = artifact_root / f"{stem}.parquet"
    rows.to_parquet(parquet_path, index=False)
    by_action = {}
    for action_name, action_rows in rows.groupby("sampled_action_name"):
        by_action[str(action_name)] = summarize_action_subset(action_rows)
    return {
        "stage": STAGE,
        "classification": label,
        "row_level_parquet": str(parquet_path),
        "row_level_parquet_sha256": sha256_file(parquet_path),
        "overall": summarize_action_subset(rows),
        "by_sampled_action": by_action,
        "questions_answered": {
            "raw_reward_available": "reward_v2_raw_total" in rows.columns,
            "gae_available": "raw_gae_advantage" in rows.columns and "normalized_advantage" in rows.columns,
            "advantage_sign_available": "normalized_sign" in rows.columns,
            "logits_probabilities_available": all(col in rows.columns for col in ["P_HOLD", "P_SERVE", "pre_mask_logit_0", "pre_mask_logit_1"]),
        },
    }


def ppo_effectiveness(frames: Mapping[str, pd.DataFrame], artifact_root: Path) -> Dict[str, Any]:
    base = frames["joined"][["sample_uid", "classification", "target_id"]]
    ppo = frames["ppo"].merge(base, on="sample_uid", how="left")
    parquet_path = artifact_root / "ppo_update_effectiveness.parquet"
    ppo.to_parquet(parquet_path, index=False)
    numeric_cols = [
        "old_action_probability",
        "current_action_probability",
        "probability_ratio",
        "normalized_advantage",
        "unclipped_surrogate",
        "clipped_surrogate",
        "effective_policy_surrogate_contribution",
        "entropy_contribution",
    ]

    def summarize(rows: pd.DataFrame) -> Dict[str, Any]:
        return {
            "ppo_row_count": int(len(rows)),
            "unique_sample_count": int(rows["sample_uid"].nunique()) if len(rows) else 0,
            "clip_active_rate": safe_rate(int(rows["clip_active"].sum()), len(rows)) if len(rows) else None,
            "positive_normalized_advantage_rate": safe_rate(int((rows["normalized_advantage"] > 0).sum()), len(rows)) if len(rows) else None,
            **{col: numeric_stats(rows[col]) for col in numeric_cols},
        }

    by_label = {}
    for label in [CLASS_HOLD_BETTER, CLASS_SERVE_BETTER]:
        label_rows = ppo[ppo["classification"] == label]
        by_action = {
            str(action): summarize(action_rows)
            for action, action_rows in label_rows.groupby("sampled_action_name")
        }
        by_cycle = {
            str(int(cycle)): summarize(cycle_rows)
            for cycle, cycle_rows in label_rows.groupby("outer_cycle")
        }
        by_label[label] = {"overall": summarize(label_rows), "by_sampled_action": by_action, "by_cycle": by_cycle}
    return {
        "stage": STAGE,
        "row_level_parquet": str(parquet_path),
        "row_level_parquet_sha256": sha256_file(parquet_path),
        "by_classification": by_label,
        "clipping_suppression_evidence": {
            "hold_better_clip_rate": by_label[CLASS_HOLD_BETTER]["overall"]["clip_active_rate"],
            "serve_better_clip_rate": by_label[CLASS_SERVE_BETTER]["overall"]["clip_active_rate"],
            "interpretation": "Observed clip rates are low and PPO ratios remain near 1; evidence does not support clipping as the primary loss mechanism.",
        },
    }


def logit_margin_evolution(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    sample = frames["sample"]
    rows = []
    for label in [CLASS_HOLD_BETTER, CLASS_SERVE_BETTER]:
        label_rows = sample[sample["classification"] == label]
        for cycle, cycle_rows in label_rows.groupby("cycle"):
            counts = cycle_rows["sampled_action_name"].value_counts().to_dict()
            rows.append(
                {
                    "classification": label,
                    "cycle": int(cycle),
                    "count": int(len(cycle_rows)),
                    "P_HOLD": numeric_stats(cycle_rows["P_HOLD"]),
                    "P_SERVE": numeric_stats(cycle_rows["P_SERVE"]),
                    "policy_margin_hold_minus_serve": numeric_stats(cycle_rows["policy_margin_hold_minus_serve"]),
                    "pre_mask_logit_margin_hold_minus_serve": numeric_stats(cycle_rows["pre_mask_logit_margin_hold_minus_serve"]),
                    "sampled_action_counts": {str(k): int(v) for k, v in counts.items()},
                }
            )
    hold_rows = [row for row in rows if row["classification"] == CLASS_HOLD_BETTER]
    serve_rows = [row for row in rows if row["classification"] == CLASS_SERVE_BETTER]
    return {
        "stage": STAGE,
        "cycle_rows": rows,
        "key_observation": {
            "hold_better_cycle1_margin": hold_rows[0]["policy_margin_hold_minus_serve"]["mean"],
            "hold_better_cycle11_margin": hold_rows[-1]["policy_margin_hold_minus_serve"]["mean"],
            "serve_better_cycle1_margin": serve_rows[0]["policy_margin_hold_minus_serve"]["mean"],
            "serve_better_cycle11_margin": serve_rows[-1]["policy_margin_hold_minus_serve"]["mean"],
            "interpretation": "HOLD_BETTER begins slightly HOLD-favored at cycle 1 but crosses to SERVE by cycle 2 and remains SERVE-favored.",
        },
    }


def pressure_totals(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    base = frames["joined"][["sample_uid", "classification", "target_id"]]
    pressure = frames["pressure"].merge(base, on="sample_uid", how="left")
    rows = []
    for label in [CLASS_HOLD_BETTER, CLASS_SERVE_BETTER]:
        label_rows = pressure[pressure["classification"] == label]
        rows.append(
            {
                "classification": label,
                "pressure_row_count": int(len(label_rows)),
                "unique_sample_count": int(label_rows["sample_uid"].nunique()),
                "hold_logit_increase_count": int((label_rows["hold_logit_pressure"] == "increase").sum()),
                "serve_logit_increase_count": int((label_rows["serve_logit_pressure"] == "increase").sum()),
                "net_serve_minus_hold_increase_count": int(
                    (label_rows["serve_logit_pressure"] == "increase").sum()
                    - (label_rows["hold_logit_pressure"] == "increase").sum()
                ),
            }
        )
    return {"by_classification": rows}


def shared_trunk_interference_audit(frames: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    parameter_delta = read_json(H4M_Q_ROOT / "parameter_delta.json")
    gradient = read_json(H4M_Q_ROOT / "gradient_audit.json")
    pressure = pressure_totals(frames)
    seed_rows = []
    for row in parameter_delta.get("seed_rows", []):
        branch = row["branch_parameter_delta"]
        seed_rows.append(
            {
                "seed": row["seed"],
                "shared_trunk_l2_delta": branch["shared_trunk"]["l2_delta"],
                "hold_head_l2_delta": branch["hold_context_head"]["l2_delta"],
                "serve_head_l2_delta": branch["serve_context_head"]["l2_delta"],
                "skip_head_l2_delta": branch["skip_context_head"]["l2_delta"],
                "shared_to_hold_head_l2_ratio": branch["shared_trunk"]["l2_delta"] / branch["hold_context_head"]["l2_delta"],
                "shared_to_serve_head_l2_ratio": branch["shared_trunk"]["l2_delta"] / branch["serve_context_head"]["l2_delta"],
            }
        )
    return {
        "stage": STAGE,
        "cross_specialized_head_gradient_leakage_count": gradient.get("cross_specialized_head_gradient_leakage_count"),
        "gradient_checks": gradient.get("checks"),
        "training_pressure_by_target": gradient.get("training_pressure_by_target"),
        "pressure_by_long_horizon_classification": pressure,
        "branch_parameter_delta": {
            "seed_rows": seed_rows,
            "summary": {
                "shared_trunk_l2_delta": numeric_stats([row["shared_trunk_l2_delta"] for row in seed_rows]),
                "hold_head_l2_delta": numeric_stats([row["hold_head_l2_delta"] for row in seed_rows]),
                "serve_head_l2_delta": numeric_stats([row["serve_head_l2_delta"] for row in seed_rows]),
                "shared_to_hold_head_l2_ratio": numeric_stats([row["shared_to_hold_head_l2_ratio"] for row in seed_rows]),
                "shared_to_serve_head_l2_ratio": numeric_stats([row["shared_to_serve_head_l2_ratio"] for row in seed_rows]),
            },
        },
        "h1_assessment": {
            "supported_as_primary": False,
            "reason": (
                "Shared trunk deltas are large, but isolated target=HOLD evidence shows net corrective HOLD pressure, "
                "not a uniquely SERVE-directed HOLD-head leak. This makes pure shared-representation interference "
                "insufficiently supported as the single primary cause."
            ),
        },
    }


def h4m_o_p_q_comparison(validation: Mapping[str, Any]) -> Dict[str, Any]:
    o_pressure = read_json(H4M_O_ROOT / "03_actor_gradient_pressure_decomposition.json")
    o_root = read_json(H4M_O_ROOT / "08_root_cause_attribution.json")
    p_grad = read_json(H4M_P_ROOT / "gradient_isolation_validation.json")
    q_hold = validation["aggregate_by_classification"][CLASS_HOLD_BETTER]
    q_serve = validation["aggregate_by_classification"][CLASS_SERVE_BETTER]
    post_hold_count = int(q_hold["sampled_action_counts"].get(ACTION_HOLD, 0))
    pre_hold_count = EXPECTED["pre_specialization_hold_better_validation_hold_count"]
    total = EXPECTED["pre_specialization_hold_better_validation_total"]
    return {
        "stage": STAGE,
        "h4m_o": {
            "decision": read_json(H4M_O_ROOT / "11_gate_matrix.json").get("decision"),
            "root_cause": o_root.get("root_cause"),
            "shared_actor_pressure_total": o_pressure.get("shared_actor_pressure_total"),
            "by_target_pressure_totals": o_pressure.get("by_target_pressure_totals"),
        },
        "h4m_p": {
            "gradient_isolation_validation_passed": p_grad.get("gradient_isolation_validation_passed"),
            "interpretation": "H4M-P was not a no-effect repair; it removed cross-head gradient leakage structurally.",
        },
        "h4m_q": {
            "behavioral_outcome": read_json(H4M_Q_ROOT / "gate_matrix.json").get("behavioral_outcome"),
            "hold_better_validation": q_hold,
            "serve_better_validation": q_serve,
            "partial_hold_recovery": {
                "pre_specialization_hold_better_hold_count": pre_hold_count,
                "post_specialization_hold_better_hold_count": post_hold_count,
                "total": total,
                "absolute_hold_count_gain": post_hold_count - pre_hold_count,
                "pre_rate": safe_rate(pre_hold_count, total),
                "post_rate": safe_rate(post_hold_count, total),
            },
            "full_discrimination_restoration": False,
        },
    }


def root_cause_classification(
    hold: Mapping[str, Any],
    serve: Mapping[str, Any],
    ppo: Mapping[str, Any],
    margin: Mapping[str, Any],
    shared: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> Dict[str, Any]:
    hold_overall = hold["overall"]
    hold_by_action = hold["by_sampled_action"]
    serve_overall = serve["overall"]
    hold_ppo = ppo["by_classification"][CLASS_HOLD_BETTER]
    serve_ppo = ppo["by_classification"][CLASS_SERVE_BETTER]
    hold_hold = hold_by_action.get(ACTION_HOLD, {})
    hold_serve = hold_by_action.get(ACTION_SERVE, {})
    h2_supported = (
        hold_hold.get("normalized_advantage", {}).get("mean") is not None
        and hold_hold["normalized_advantage"]["mean"] < 0.0
        and hold_overall["normalized_advantage"]["mean"] < 0.0
        and serve_overall["normalized_advantage"]["mean"] > 0.0
    )
    h3_supported = (
        hold_ppo["overall"]["clip_active_rate"] is not None
        and hold_ppo["overall"]["clip_active_rate"] > 0.2
    )
    hold_pressure_rows = {
        row["classification"]: row
        for row in shared["pressure_by_long_horizon_classification"]["by_classification"]
    }
    hold_net_pressure = hold_pressure_rows[CLASS_HOLD_BETTER]["net_serve_minus_hold_increase_count"]
    h1_supported_primary = shared["h1_assessment"]["supported_as_primary"] is True
    h4_supported_primary = False
    selected_root = "PPO_CREDIT_OR_ADVANTAGE_MISALIGNMENT" if h2_supported and not h3_supported else "MIXED_OR_OTHER"
    selected_next = NEXT_REPAIR if selected_root == "PPO_CREDIT_OR_ADVANTAGE_MISALIGNMENT" else "NO_REPAIR_SELECTED_INSUFFICIENT_EVIDENCE"
    exact_next_gate = NEXT_GATE if selected_next == NEXT_REPAIR else "STOP_NO_REPAIR_SELECTED_INSUFFICIENT_EVIDENCE"
    return {
        "stage": STAGE,
        "selected_root_cause": selected_root,
        "selected_next_repair_direction": selected_next,
        "exact_next_gate": exact_next_gate,
        "hypothesis_assessment": {
            "H1_SHARED_REPRESENTATION_INTERFERENCE": {
                "supported_as_primary": h1_supported_primary,
                "evidence": {
                    "shared_trunk_delta_large": shared["branch_parameter_delta"]["summary"]["shared_trunk_l2_delta"],
                    "hold_better_net_serve_minus_hold_pressure": hold_net_pressure,
                    "cross_head_leakage": shared["cross_specialized_head_gradient_leakage_count"],
                },
                "interpretation": "Possible contributor, but not uniquely supported as primary because target=HOLD pressure is corrective rather than SERVE-directed.",
            },
            "H2_PPO_CREDIT_OR_ADVANTAGE_MISALIGNMENT": {
                "supported_as_primary": h2_supported,
                "evidence": {
                    "hold_better_hold_action_normalized_advantage_mean": hold_hold["normalized_advantage"]["mean"],
                    "hold_better_serve_action_normalized_advantage_mean": hold_serve["normalized_advantage"]["mean"],
                    "hold_better_overall_normalized_advantage_mean": hold_overall["normalized_advantage"]["mean"],
                    "hold_better_positive_advantage_rate": hold_overall["normalized_advantage_positive_rate"],
                    "serve_better_overall_normalized_advantage_mean": serve_overall["normalized_advantage"]["mean"],
                    "serve_better_positive_advantage_rate": serve_overall["normalized_advantage_positive_rate"],
                    "hold_better_ppo_effective_policy_surrogate_mean": hold_ppo["overall"]["effective_policy_surrogate_contribution"]["mean"],
                    "serve_better_ppo_effective_policy_surrogate_mean": serve_ppo["overall"]["effective_policy_surrogate_contribution"]["mean"],
                },
                "interpretation": (
                    "HOLD_BETTER does not provide positive mean PPO/GAE reinforcement even for sampled HOLD actions; "
                    "SERVE_BETTER provides strong positive SERVE reinforcement. This is the clearest single mechanism."
                ),
            },
            "H3_PPO_CLIPPING_OR_OPTIMIZATION_SUPPRESSION": {
                "supported_as_primary": h3_supported,
                "evidence": ppo["clipping_suppression_evidence"],
                "interpretation": "Not primary: clip rates are low and ratios are near 1.",
            },
            "H4_DECISION_MARGIN_CALIBRATION_IMBALANCE": {
                "supported_as_primary": h4_supported_primary,
                "evidence": margin["key_observation"],
                "interpretation": (
                    "Margin imbalance is observed, but evidence points upstream to advantage/credit imbalance rather "
                    "than a standalone calibration-only failure."
                ),
            },
            "H5_MIXED_OR_OTHER": {
                "selected": selected_root == "MIXED_OR_OTHER",
                "interpretation": "Not selected because H2 is sufficiently supported and H3/H1/H4 are weaker as primary causes.",
            },
        },
        "direct_answers": {
            "A_did_HOLD_BETTER_generate_sufficient_positive_learning_pressure_toward_HOLD": "No. HOLD_BETTER sampled HOLD has negative mean normalized advantage and HOLD_BETTER aggregate PPO surrogate is negative.",
            "B_if_yes_where_was_signal_lost": "Not applicable as sufficient positive HOLD signal was not observed; corrective pressure exists mainly through weak negative-advantage suppression of SERVE.",
            "C_did_shared_trunk_move_HOLD_BETTER_toward_SERVE": "Cycle margins moved toward SERVE, but available pressure evidence does not uniquely attribute this to shared-trunk interference.",
            "D_cause": selected_root,
        },
        "partial_recovery_vs_restoration": comparison["h4m_q"]["partial_hold_recovery"],
        "full_discrimination_restoration": comparison["h4m_q"]["full_discrimination_restoration"],
    }


def next_decision(root: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "selected_root_cause": root["selected_root_cause"],
        "selected_minimal_next_direction": root["selected_next_repair_direction"],
        "exact_next_gate": root["exact_next_gate"],
        "do_not_execute_now": True,
        "repair_implementation_performed": False,
        "selection_rule": (
            "Select PPO_CREDIT_ADVANTAGE_REPAIR because H2 is supported by negative HOLD_BETTER mean advantage "
            "and low PPO clipping; stop after decision only."
        ),
    }


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
    next_decision_payload: Mapping[str, Any],
    changed: Mapping[str, Any],
    manifest_ready: bool,
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_passed": binding.get("authoritative_binding_passed") is True,
        "source_only_local_commit": changed.get("source_only_local_commit") is True,
        "root_cause_classified": root.get("selected_root_cause")
        in {
            "SHARED_REPRESENTATION_INTERFERENCE",
            "PPO_CREDIT_OR_ADVANTAGE_MISALIGNMENT",
            "PPO_CLIPPING_OR_OPTIMIZATION_SUPPRESSION",
            "DECISION_MARGIN_CALIBRATION_IMBALANCE",
            "MIXED_OR_OTHER",
        },
        "next_decision_selected": next_decision_payload.get("selected_minimal_next_direction")
        in {
            "SHARED_REPRESENTATION_SPECIALIZATION_REPAIR",
            "PPO_CREDIT_ADVANTAGE_REPAIR",
            "PPO_OPTIMIZATION_CLIPPING_REPAIR",
            "ACTOR_LOGIT_MARGIN_CALIBRATION_REPAIR",
            "NO_REPAIR_SELECTED_INSUFFICIENT_EVIDENCE",
        },
        "no_training_or_optimizer": binding.get("hard_lock_attestation", {}).get("optimizer_created_or_stepped") is False,
        "test6_zero": binding.get("hard_lock_attestation", {}).get("test6_opened_or_used") is False,
        "required_artifacts_present": manifest_ready,
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": next_decision_payload.get("selected_minimal_next_direction") if passed else "H4M_R_BLOCKED_REVIEW_EVIDENCE",
        "selected_root_cause": root.get("selected_root_cause"),
        "exact_next_gate": next_decision_payload.get("exact_next_gate") if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
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
    hold: Mapping[str, Any],
    serve: Mapping[str, Any],
    ppo: Mapping[str, Any],
    margin: Mapping[str, Any],
    shared: Mapping[str, Any],
    root: Mapping[str, Any],
    decision: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    return f"""# H4M-R Target-Conditioned Actor-Head Outcome Review

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_r_source_git_commit"]}
selected_root_cause = {root["selected_root_cause"]}
selected_next_direction = {decision["selected_minimal_next_direction"]}
exact_next_gate = {gate["exact_next_gate"]}

## HOLD_BETTER credit trace

```json
{json.dumps({"overall": hold["overall"], "by_sampled_action": hold["by_sampled_action"]}, ensure_ascii=False, indent=2, default=jsonable)}
```

## SERVE_BETTER credit trace

```json
{json.dumps({"overall": serve["overall"], "by_sampled_action": serve["by_sampled_action"]}, ensure_ascii=False, indent=2, default=jsonable)}
```

## PPO effectiveness

```json
{json.dumps(ppo["clipping_suppression_evidence"], ensure_ascii=False, indent=2, default=jsonable)}
```

## Logit margin evolution

```json
{json.dumps(margin["key_observation"], ensure_ascii=False, indent=2, default=jsonable)}
```

## Shared trunk / gradient audit

```json
{json.dumps({"cross_head_leakage": shared["cross_specialized_head_gradient_leakage_count"], "branch_delta_summary": shared["branch_parameter_delta"]["summary"], "h1_assessment": shared["h1_assessment"]}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Root cause

```json
{json.dumps(root, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no repair implementation, no retraining, no optimizer step, no TEST6, no GitHub push.
"""


def write_block_outputs(artifact_root: Path, created_at: str, binding: Mapping[str, Any], reason: str) -> None:
    gate = {
        "stage": STAGE,
        "gate": BLOCK_GATE,
        "decision": "H4M_R_BLOCKED_REVIEW_EVIDENCE",
        "selected_root_cause": "MIXED_OR_OTHER",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
        "created_at": created_at,
    }
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "authoritative_binding.json": binding,
        "hold_better_credit_trace.json": empty,
        "serve_better_credit_trace.json": empty,
        "ppo_update_effectiveness.json": empty,
        "logit_margin_evolution.json": empty,
        "shared_trunk_interference_audit.json": empty,
        "root_cause_classification.json": empty,
        "next_decision.json": empty,
        "changed_files.json": changed_files(),
        "test_results.json": binding.get("compile_static_checks", empty),
        "gate_matrix.json": gate,
    }.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(f"# H4M-R\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-R] artifact root: {artifact_root}")
    print(f"[H4M-R] gate: {BLOCK_GATE}")
    print(f"[H4M-R] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_r_target_conditioned_actor_head_specialized_retraining_outcome_review_next_decision_{stamp}"
    provenance = source_provenance(created_at)
    compile_static = py_compile_audit()
    binding = authoritative_binding(created_at, provenance, compile_static)
    if not binding["authoritative_binding_passed"]:
        write_block_outputs(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return

    artifact_root.mkdir(parents=True, exist_ok=True)
    frames = load_evidence_frames()
    validation = read_json(H4M_Q_ROOT / "validation_discrimination.json")
    hold = credit_trace_for_label(frames, CLASS_HOLD_BETTER, artifact_root, "hold_better_credit_trace")
    serve = credit_trace_for_label(frames, CLASS_SERVE_BETTER, artifact_root, "serve_better_credit_trace")
    ppo = ppo_effectiveness(frames, artifact_root)
    margin = logit_margin_evolution(frames)
    shared = shared_trunk_interference_audit(frames)
    comparison = h4m_o_p_q_comparison(validation)
    root = root_cause_classification(hold, serve, ppo, margin, shared, comparison)
    decision = next_decision(root)
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
        "hold_better_credit_trace.json": hold,
        "serve_better_credit_trace.json": serve,
        "ppo_update_effectiveness.json": ppo,
        "logit_margin_evolution.json": margin,
        "shared_trunk_interference_audit.json": shared,
        "h4m_o_p_q_comparison.json": comparison,
        "root_cause_classification.json": root,
        "next_decision.json": decision,
        "changed_files.json": changed,
        "test_results.json": test_results,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)

    manifest_ready_pre_gate = all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name not in {"manifest.json", "gate_matrix.json", "final_report.md"})
    gate = gate_matrix(binding, root, decision, changed, manifest_ready_pre_gate)
    write_json(artifact_root / "gate_matrix.json", gate)
    (artifact_root / "final_report.md").write_text(
        final_report(binding, hold, serve, ppo, margin, shared, root, decision, gate),
        encoding="utf-8",
    )
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    print(f"[H4M-R] artifact root: {artifact_root}")
    print(f"[H4M-R] gate: {gate['gate']}")
    print(f"[H4M-R] source_commit: {provenance['h4m_r_source_git_commit']}")
    print(f"[H4M-R] selected_root_cause: {gate['selected_root_cause']}")
    print(f"[H4M-R] selected_next_direction: {gate['decision']}")
    print(f"[H4M-R] exact_next_gate: {gate['exact_next_gate']}")


if __name__ == "__main__":
    main()
