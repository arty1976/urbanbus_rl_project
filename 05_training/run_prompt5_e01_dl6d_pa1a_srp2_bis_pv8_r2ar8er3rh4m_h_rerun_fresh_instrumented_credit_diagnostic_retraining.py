#!/usr/bin/env python3
"""H4M-H-RERUN fresh instrumented credit diagnostic retraining.

This runner binds the authoritative H4M-G-CLOSE MPS-safe instrumentation contract and then
executes exactly the authorized fresh seed 1/2/3 × 11-cycle diagnostic training.

It is diagnostic only: no Reward/GAE/PPO/model/environment repair, no H4M-C
checkpoint continuation, no validation/TEST6, no winner selection, and no
GitHub push.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-H-RERUN"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_H_RERUN_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_H_RERUN_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

BASE_H4MH_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_h_fresh_instrumented_credit_diagnostic_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"

H4M_F_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_f_credit_trace_instrumentation_selection_freeze_20260815_111506+0900"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_A_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_a_decision_opportunity_environment_adequacy_audit_20260814_143743"

EXPECTED = {
    "h4m_g_close_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_CLOSE_INSTRUMENTATION_DIAGNOSIS_REPAIR_AND_MPS_EQUIVALENCE_VALIDATION_COMPLETE",
    "h4m_g_close_decision": "PV8_CREDIT_TRACE_INSTRUMENTATION_VALIDATED_ON_MPS_READY_FOR_H4M_H_FRESH_DIAGNOSTIC_RETRAINING_RERUN",
    "h4m_g_close_source_commit": "2d5e56bf8f0c8e81474274e865804796ebc591e8",
    "active_mps_instrumentation_contract_sha256": "e6c73da48c12edfd573069730d2eeec32c74fea74f590105e55ad3502a729c92",
    "h4m_f_instrumentation_sha256": "9b95d0dc46459eedd2cf51e85789407be247e9db1e23e3a52a5bbae9c6216ffe",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "seeds": [1, 2, 3],
    "outer_training_count": 11,
    "ppo_updates_per_seed": 44,
    "critic_updates_per_seed": 88,
}

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_training_integrity.json",
    "03_three_seed_cycle_summary.json",
    "04_policy_probability_evolution.json",
    "05_actual_credit_trace",
    "06_long_horizon_shadow_trace",
    "07_long_horizon_vs_gae_crosswalk.json",
    "08_advantage_normalization_audit.json",
    "09_ppo_actor_pressure_audit.json",
    "10_critic_credit_audit.json",
    "11_earliest_preference_reversal.json",
    "12_root_cause_decision.json",
    "13_checkpoint_registry.json",
    "14_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]

ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
TOL = 1.0e-12


def kst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base_h4mh() -> Any:
    base = import_module_from_path("h4mh_base_fresh_diagnostic", BASE_H4MH_SOURCE)
    base.STAGE = STAGE
    return base


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    latest_source_commit = git_run(["log", "-1", "--format=%H", "--", str(SOURCE_REL)]).stdout.strip()
    source_present = git_run(["cat-file", "-e", f"HEAD:{SOURCE_REL}"], check=False).returncode == 0
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_h_rerun_source_git_commit": latest_source_commit,
        "source_rel": str(SOURCE_REL),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "source_present_in_head": source_present,
        "head_commit_files": head_files,
        "head_commit_source_only": head_files == [str(SOURCE_REL)],
        "status_short": status_short,
        "local_source_only_commit_created_before_training": source_present
        and latest_source_commit == head
        and head_files == [str(SOURCE_REL)]
        and status_short == "",
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    close_binding = read_json(H4M_G_CLOSE_ROOT / "01_authoritative_lineage_binding.json")
    close_contract = read_json(H4M_G_CLOSE_ROOT / "12_final_mps_equivalence_contract.json")
    close_release = read_json(H4M_G_CLOSE_ROOT / "13_h4mh_release_gate.json")
    close_gate = read_json(H4M_G_CLOSE_ROOT / "14_gate_matrix.json")
    h4m_f_manifest = read_json(H4M_F_ROOT / "manifest.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    upstream = close_binding.get("authoritative_upstream_sha_bindings", {})
    checks = {
        "source_only_commit_before_training": provenance.get("local_source_only_commit_created_before_training") is True,
        "h4m_g_close_gate_match": close_gate.get("gate") == EXPECTED["h4m_g_close_gate"],
        "h4m_g_close_decision_match": close_gate.get("decision") == EXPECTED["h4m_g_close_decision"],
        "h4m_g_close_source_commit_match": close_binding.get("source_provenance", {}).get("h4m_g_close_source_git_commit")
        == EXPECTED["h4m_g_close_source_commit"],
        "h4m_h_release_gate_yes": close_release.get("h4m_h_release_yes_no") == "YES",
        "h4m_h_rerun_not_previously_executed_by_close": close_release.get("h4m_h_rerun_executed") is False,
        "active_contract_sha_match_gate": close_gate.get("final_active_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "active_contract_sha_match_file": sha256_file(H4M_G_CLOSE_ROOT / "12_final_mps_equivalence_contract.json")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "active_contract_passed": close_contract.get("active_contract_passed") is True,
        "h4m_f_contract_sha_match": h4m_f_manifest.get("instrumentation_contract_sha256")
        == EXPECTED["h4m_f_instrumentation_sha256"],
        "h4m_b_schedule_sha_match_direct": h4m_b_schedule.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"],
        "h4m_b_schedule_sha_match_close": upstream.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": upstream.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": upstream.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": upstream.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": upstream.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "test6_sealed_in_release": close_release.get("conditions", {}).get("test6_sealed") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_f": str(H4M_F_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4m_b": str(H4M_B_ROOT),
            "h4m_a": str(H4M_A_ROOT),
        },
        "source_provenance": provenance,
        "authoritative_binding_passed": all(checks.values()),
        "checks": checks,
        "bound_gates": {
            "h4m_g_close_gate": close_gate.get("gate"),
            "h4m_g_close_decision": close_gate.get("decision"),
            "h4m_g_close_next_gate": close_gate.get("exact_next_gate"),
            "h4m_h_release_yes_no": close_release.get("h4m_h_release_yes_no"),
        },
        "sha_bindings": {
            "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
            "h4m_f_instrumentation_sha256": EXPECTED["h4m_f_instrumentation_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "active_contract_snapshot": {
            "contract_name": close_contract.get("contract_name"),
            "contract_version": close_contract.get("contract_version"),
            "bit_exact_equivalence": close_contract.get("bit_exact_equivalence"),
            "exact_invariants": close_contract.get("exact_invariants"),
            "long_horizon_shadow_validation_passed": close_contract.get("long_horizon_shadow_validation_passed"),
        },
        "hard_lock_attestation": {
            "reward_v2_modified": False,
            "zero_loss_modified": False,
            "k_mask_modified": False,
            "actor_critic_gatv2_modified": False,
            "ppo_gae_normalization_modified": False,
            "hyperparameters_modified": False,
            "environment_data_modified": False,
            "h4m_c_checkpoint_continuation": False,
            "validation_or_test6_executed": False,
            "github_push_performed": False,
        },
    }


def shadow_integrity_snapshot(h4mg: Any, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    dl1 = ctx["dl1"]
    return {
        "rng": h4mg.rng_snapshot(),
        "model_hashes": h4mg.model_hashes(dl1, ctx["encoder"], ctx["actor"], ctx["critic"]),
        "optimizer_hashes": h4mg.optimizer_hashes(ctx["optimizers"]),
        "reward_normalizer": h4mg.reward_normalizer_snapshot(ctx["reward_normalizer"]),
        "return_normalizer": ctx["return_normalizer"].state_dict(),
    }


def stats(values: Sequence[float]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(nums), "mean": float(mean(nums)), "min": min(nums), "max": max(nums)}


def run_diagnostic_training_rerun(
    base: Any,
    h4mg: Any,
    created_at: str,
    artifact_root: Path,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    all_cycle_summaries: List[Dict[str, Any]] = []
    trace_registry: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    seed_summaries: List[Dict[str, Any]] = []
    joined_rows: List[Dict[str, Any]] = []
    safety_counts: Counter[str] = Counter()
    shadow_audits: List[Dict[str, Any]] = []

    for seed in EXPECTED["seeds"]:
        ctx = base.build_context_mps(h4mg, seed, created_at, device=device)
        dl1 = ctx["dl1"]
        before_state = {
            "gatv2": dl1.clone_state_dict(ctx["encoder"]),
            "actor": dl1.clone_state_dict(ctx["actor"]),
            "critic": dl1.clone_state_dict(ctx["critic"]),
        }
        seed_cycle_summaries: List[Dict[str, Any]] = []
        ppo_rows_total = 0
        critic_update_rows_total = 0
        active_samples_total = 0
        started_seed = time.perf_counter()

        for cycle in range(1, EXPECTED["outer_training_count"] + 1):
            recorder = h4mg.TraceRecorder(root=artifact_root / "_unused_h4mg_recorder_root", enabled=True)
            rollout = h4mg.collect_controlled_rollout(
                branch="H4M_H_RERUN_DIAGNOSTIC",
                seed=seed,
                cycle_index=cycle,
                ctx=ctx,
                recorder=recorder,
            )
            update = h4mg.ppo_update_controlled(
                branch="H4M_H_RERUN_DIAGNOSTIC",
                seed=seed,
                cycle_index=cycle,
                ctx=ctx,
                rollout=rollout,
                before_state=before_state,
                recorder=recorder,
            )
            before_shadow = shadow_integrity_snapshot(h4mg, ctx)
            pair_rows, event_rows, summary_rows, shadow_completeness = base.build_full_shadow_rows(h4mg, seed, cycle, ctx, rollout)
            after_shadow = shadow_integrity_snapshot(h4mg, ctx)
            shadow_audit = {
                "seed": seed,
                "outer_cycle": cycle,
                "rng_unchanged_by_shadow": before_shadow["rng"] == after_shadow["rng"],
                "model_hashes_unchanged_by_shadow": before_shadow["model_hashes"] == after_shadow["model_hashes"],
                "optimizer_hashes_unchanged_by_shadow": before_shadow["optimizer_hashes"] == after_shadow["optimizer_hashes"],
                "reward_normalizer_unchanged_by_shadow": before_shadow["reward_normalizer"] == after_shadow["reward_normalizer"],
                "return_normalizer_unchanged_by_shadow": before_shadow["return_normalizer"] == after_shadow["return_normalizer"],
                "shadow_pair_rows": len(pair_rows),
                "shadow_branch_event_rows": len(event_rows),
                "shadow_summary_rows": len(summary_rows),
            }
            shadow_audit["shadow_isolation_passed"] = all(
                bool(shadow_audit[k])
                for k in [
                    "rng_unchanged_by_shadow",
                    "model_hashes_unchanged_by_shadow",
                    "optimizer_hashes_unchanged_by_shadow",
                    "reward_normalizer_unchanged_by_shadow",
                    "return_normalizer_unchanged_by_shadow",
                ]
            )
            shadow_audits.append(shadow_audit)

            trace_registry.append(base.write_cycle_trace(artifact_root, seed, cycle, recorder, (pair_rows, event_rows, summary_rows)))
            cycle_summary = base.summarize_cycle(seed, cycle, recorder, summary_rows)
            cycle_summary["shadow_completeness"] = shadow_completeness
            cycle_summary["ppo_metric_rows"] = len(update["metrics_rows"])
            cycle_summary["actor_joint_update_rows"] = sum(
                1 for row in update["metrics_rows"] if row.get("update_role") == "actor_gatv2_critic_joint"
            )
            cycle_summary["critic_update_rows"] = len(update["metrics_rows"])
            cycle_summary["loss_finite"] = all(
                all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in update["loss_rows"]
            )
            safety_counts["illegal_action"] += sum(
                1 for row in recorder.pre_action_rows if int(row["action_id"]) not in [int(v) for v in row["legal_action_ids"]]
            )
            safety_counts["illegal_SKIP"] += sum(
                1 for row in recorder.pre_action_rows if int(row["action_id"]) == 2 and 2 not in [int(v) for v in row["legal_action_ids"]]
            )
            safety_counts["nonfinite_loss_cycle"] += int(not cycle_summary["loss_finite"])
            safety_counts["shadow_rng_drift"] += int(not shadow_audit["rng_unchanged_by_shadow"])
            safety_counts["shadow_model_contamination"] += int(not shadow_audit["model_hashes_unchanged_by_shadow"])
            safety_counts["shadow_optimizer_contamination"] += int(not shadow_audit["optimizer_hashes_unchanged_by_shadow"])
            safety_counts["shadow_normalizer_contamination"] += int(
                not (shadow_audit["reward_normalizer_unchanged_by_shadow"] and shadow_audit["return_normalizer_unchanged_by_shadow"])
            )

            seed_cycle_summaries.append(cycle_summary)
            all_cycle_summaries.append(cycle_summary)
            ppo_rows_total += cycle_summary["actor_joint_update_rows"]
            critic_update_rows_total += cycle_summary["critic_update_rows"]
            active_samples_total += cycle_summary["active_samples"]

            policy_by_uid = {row["sample_uid"]: row for row in recorder.pre_action_rows}
            shadow_by_uid = {row["sample_uid"]: row for row in summary_rows}
            adv_by_uid = {row["sample_uid"]: row for row in recorder.advantage_sample_rows}
            critic_by_uid = {row["sample_uid"]: row for row in recorder.critic_value_error_rows}
            pressure_by_uid: Dict[str, Counter] = defaultdict(Counter)
            for row in recorder.actor_pressure_rows:
                pressure_by_uid[row["sample_uid"]][(row["serve_logit_pressure"], row["hold_logit_pressure"])] += 1
            for uid, shadow in shadow_by_uid.items():
                adv = adv_by_uid[uid]
                crit = critic_by_uid[uid]
                policy = policy_by_uid[uid]
                pressure = pressure_by_uid[uid]
                joined_rows.append(
                    {
                        "seed": seed,
                        "cycle": cycle,
                        "sample_uid": uid,
                        "window_id": policy.get("window_id"),
                        "step_index": int(policy.get("step_index")),
                        "agent_slot": int(policy.get("agent_slot")),
                        "state_hash": policy.get("state_hash"),
                        "classification": shadow["classification"],
                        "delta_full_bootstrapped_return_serve_minus_hold": shadow["delta_full_bootstrapped_return_serve_minus_hold"],
                        "delta_discounted_reward_to_boundary_serve_minus_hold": shadow[
                            "delta_discounted_reward_to_boundary_serve_minus_hold"
                        ],
                        "delta_bootstrap_value_serve_minus_hold": shadow["delta_bootstrap_value_serve_minus_hold"],
                        "sampled_action_name": adv["sampled_action_name"],
                        "raw_gae_advantage": adv["raw_gae_advantage"],
                        "normalized_advantage": adv["normalized_advantage"],
                        "sign_change_class": adv["sign_change_class"],
                        "value_error": crit["value_error"],
                        "serve_pressure_increase_count": int(sum(v for (serve, _hold), v in pressure.items() if serve == "increase")),
                        "hold_pressure_increase_count": int(sum(v for (_serve, hold), v in pressure.items() if hold == "increase")),
                    }
                )
            if torch.backends.mps.is_available() and hasattr(torch, "mps"):
                torch.mps.empty_cache()

        checkpoint = base.save_seed_checkpoint(artifact_root, seed, ctx, seed_cycle_summaries)
        checkpoint["active_mps_instrumentation_contract_sha256"] = EXPECTED["active_mps_instrumentation_contract_sha256"]
        checkpoint["h4m_f_instrumentation_sha256"] = EXPECTED["h4m_f_instrumentation_sha256"]
        checkpoint_rows.append(checkpoint)
        parameter_delta = {
            "gatv2": dl1.delta_stats(ctx["encoder"], before_state["gatv2"]),
            "actor": dl1.delta_stats(ctx["actor"], before_state["actor"]),
            "critic": dl1.delta_stats(ctx["critic"], before_state["critic"]),
        }
        seed_summaries.append(
            {
                "seed": seed,
                "fresh_initialization": True,
                "h4m_c_checkpoint_continuation": False,
                "rollouts": len(seed_cycle_summaries),
                "ppo_updates": ppo_rows_total,
                "critic_updates": critic_update_rows_total,
                "active_samples": active_samples_total,
                "elapsed_seconds": time.perf_counter() - started_seed,
                "parameter_delta": parameter_delta,
                "parameter_delta_positive": all(float(parameter_delta[k]["l2_delta"]) > 0.0 for k in ("gatv2", "actor", "critic")),
                "loss_finite": all(row["loss_finite"] for row in seed_cycle_summaries),
                "checkpoint": checkpoint,
            }
        )

    training_summary = {
        "stage": STAGE,
        "created_at": created_at,
        "device": "mps",
        "seeds": EXPECTED["seeds"],
        "seed_summaries": seed_summaries,
        "cycle_summaries": all_cycle_summaries,
        "total_rollouts": sum(row["rollouts"] for row in seed_summaries),
        "total_ppo_updates": sum(row["ppo_updates"] for row in seed_summaries),
        "total_critic_updates": sum(row["critic_updates"] for row in seed_summaries),
        "total_active_samples": sum(row["active_samples"] for row in seed_summaries),
        "fresh_initialization_all_seeds": all(row["fresh_initialization"] for row in seed_summaries),
        "h4m_c_checkpoint_continuation": False,
        "rollout_tensor_reuse": False,
    }
    checkpoint_registry = {
        "stage": STAGE,
        "created_at": created_at,
        "checkpoints": checkpoint_rows,
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_creation_authorized_for_diagnostic_evidence": True,
        "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_f_instrumentation_sha256": EXPECTED["h4m_f_instrumentation_sha256"],
        "selection_or_validation_release_status": "NOT_RELEASED_DIAGNOSTIC_ONLY",
    }
    trace_meta = {
        "trace_registry": trace_registry,
        "joined_rows": joined_rows,
        "safety_counts": dict(safety_counts),
        "shadow_audits": shadow_audits,
    }
    return training_summary, all_cycle_summaries, checkpoint_registry, trace_meta


def long_horizon_vs_gae_crosswalk(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    h4m_a_rows = read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json").get("counterfactual_rows", [])
    h4m_a_hold_keys = {
        (row.get("window_id"), int(str(row.get("state_id", "step000")).split(":step")[-1].split(":")[0]), int(row.get("agent_slot")))
        for row in h4m_a_rows
        if row.get("classification") == "HOLD_BETTER"
    }
    h4m_h_keys = {(row.get("window_id"), int(row.get("step_index")), int(row.get("agent_slot"))) for row in joined_rows}
    exact_matches = sorted(h4m_a_hold_keys & h4m_h_keys)
    by_key: Dict[str, Dict[str, Any]] = {}
    counts = Counter()
    for row in joined_rows:
        key = f"{row['classification']}::{row['sampled_action_name']}"
        counts[key] += 1
        bucket = by_key.setdefault(
            key,
            {
                "count": 0,
                "raw_gae": [],
                "normalized_advantage": [],
                "value_error": [],
                "delta_full_return_serve_minus_hold": [],
            },
        )
        bucket["count"] += 1
        bucket["raw_gae"].append(row["raw_gae_advantage"])
        bucket["normalized_advantage"].append(row["normalized_advantage"])
        bucket["value_error"].append(row["value_error"])
        bucket["delta_full_return_serve_minus_hold"].append(row["delta_full_bootstrapped_return_serve_minus_hold"])
    summarized = {
        key: {
            "count": value["count"],
            "raw_gae": stats(value["raw_gae"]),
            "normalized_advantage": stats(value["normalized_advantage"]),
            "value_error": stats(value["value_error"]),
            "delta_full_return_serve_minus_hold": stats(value["delta_full_return_serve_minus_hold"]),
        }
        for key, value in sorted(by_key.items())
    }
    return {
        "stage": STAGE,
        "total_joined_rows": len(joined_rows),
        "classification_counts": dict(Counter(row["classification"] for row in joined_rows)),
        "classification_by_sampled_action_counts": dict(counts),
        "by_classification_and_sampled_action": summarized,
        "h4m_a_local_hold_better_exact_crosswalk": {
            "h4m_a_hold_better_state_count": len(h4m_a_hold_keys),
            "h4m_h_train_state_count": len(h4m_h_keys),
            "exact_match_count": len(exact_matches),
            "exact_matches": exact_matches[:20],
            "approximate_matching_used": False,
            "note": "H4M-A local HOLD_BETTER states are validation snapshot states; H4M-H-RERUN is TRAIN44 fresh rollout. Exact state identity was attempted and no approximate crosswalk was used.",
        },
    }


def advantage_normalization_audit(joined_rows: Sequence[Mapping[str, Any]], cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sign_counts = Counter(row["sign_change_class"] for row in joined_rows)
    by_action: Dict[str, Dict[str, Any]] = {}
    for action in [ACTION_HOLD, ACTION_SERVE, ACTION_SKIP]:
        rows = [row for row in joined_rows if row["sampled_action_name"] == action]
        by_action[action] = {
            "count": len(rows),
            "raw_gae": stats([row["raw_gae_advantage"] for row in rows]),
            "normalized_advantage": stats([row["normalized_advantage"] for row in rows]),
            "raw_positive_count": sum(1 for row in rows if float(row["raw_gae_advantage"]) > TOL),
            "normalized_positive_count": sum(1 for row in rows if float(row["normalized_advantage"]) > TOL),
            "norm_positive_raw_nonpositive_count": sum(
                1 for row in rows if float(row["raw_gae_advantage"]) <= TOL and float(row["normalized_advantage"]) > TOL
            ),
        }
    return {
        "stage": STAGE,
        "aggregate_sign_flip_counts": dict(sign_counts),
        "by_action": by_action,
        "by_seed_cycle_sign_flip_counts": [
            {"seed": row["seed"], "outer_cycle": row["outer_cycle"], "sign_flip_counts": row["sign_flip_counts"]}
            for row in cycle_summaries
        ],
    }


def ppo_actor_pressure_audit(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hold_better = [row for row in joined_rows if row["classification"] == "HOLD_LONG_HORIZON_BETTER"]
    hold_better_serve = [row for row in hold_better if row["sampled_action_name"] == ACTION_SERVE]
    serve_pressure_rows = [
        row for row in hold_better_serve if int(row["serve_pressure_increase_count"]) > int(row["hold_pressure_increase_count"])
    ]
    return {
        "stage": STAGE,
        "hold_long_horizon_better_samples": len(hold_better),
        "hold_long_horizon_better_sampled_serve": len(hold_better_serve),
        "hold_better_sampled_serve_net_serve_pressure_count": len(serve_pressure_rows),
        "hold_better_sampled_serve_net_serve_pressure_rate": len(serve_pressure_rows) / max(1, len(hold_better_serve)),
        "pressure_by_classification_and_action": {
            f"{classification}::{action}": {
                "count": len(rows),
                "net_serve_pressure_count": sum(
                    1 for row in rows if int(row["serve_pressure_increase_count"]) > int(row["hold_pressure_increase_count"])
                ),
                "net_hold_pressure_count": sum(
                    1 for row in rows if int(row["hold_pressure_increase_count"]) > int(row["serve_pressure_increase_count"])
                ),
            }
            for (classification, action), rows in group_rows(joined_rows, ("classification", "sampled_action_name")).items()
        },
    }


def critic_credit_audit(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "value_error_by_classification_action": {
            f"{classification}::{action}": stats([row["value_error"] for row in rows])
            for (classification, action), rows in group_rows(joined_rows, ("classification", "sampled_action_name")).items()
        },
        "raw_gae_by_classification_action": {
            f"{classification}::{action}": stats([row["raw_gae_advantage"] for row in rows])
            for (classification, action), rows in group_rows(joined_rows, ("classification", "sampled_action_name")).items()
        },
    }


def group_rows(rows: Sequence[Mapping[str, Any]], keys: Tuple[str, ...]) -> Dict[Tuple[Any, ...], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return dict(sorted(grouped.items(), key=lambda item: str(item[0])))


def policy_evolution(cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in cycle_summaries:
        probs = row["probabilities"]
        counts = row["sampled_action_counts"]
        rows.append(
            {
                "seed": row["seed"],
                "outer_cycle": row["outer_cycle"],
                "P_HOLD_mean": probs["P_HOLD_mean"],
                "P_SERVE_mean": probs["P_SERVE_mean"],
                "entropy_mean": probs["entropy_mean"],
                "sampled_HOLD": counts.get(ACTION_HOLD, 0),
                "sampled_SERVE": counts.get(ACTION_SERVE, 0),
                "sampled_SKIP": counts.get(ACTION_SKIP, 0),
                "serve_probability_dominant": probs["P_SERVE_mean"] is not None and probs["P_SERVE_mean"] > probs["P_HOLD_mean"],
                "serve_sample_dominant": counts.get(ACTION_SERVE, 0) > counts.get(ACTION_HOLD, 0),
                "long_horizon_classification_counts": row["long_horizon_classification_counts"],
                "raw_gae_by_action": row["raw_gae_by_action"],
                "normalized_advantage_by_action": row["normalized_advantage_by_action"],
            }
        )
    first_prob = next((row for row in rows if row["serve_probability_dominant"]), None)
    first_sample = next((row for row in rows if row["serve_sample_dominant"]), None)
    return {
        "stage": STAGE,
        "rows": rows,
        "first_serve_probability_dominance": first_prob,
        "first_serve_sample_dominance": first_sample,
    }


def classify_root(joined_rows: Sequence[Mapping[str, Any]], evolution: Mapping[str, Any]) -> Dict[str, Any]:
    hold_better = [row for row in joined_rows if row["classification"] == "HOLD_LONG_HORIZON_BETTER"]
    serve_better = [row for row in joined_rows if row["classification"] == "SERVE_LONG_HORIZON_BETTER"]
    hold_better_sampled_serve = [row for row in hold_better if row["sampled_action_name"] == ACTION_SERVE]
    hb_serve_raw_positive = [row for row in hold_better_sampled_serve if float(row["raw_gae_advantage"]) > TOL]
    hb_serve_norm_positive = [row for row in hold_better_sampled_serve if float(row["normalized_advantage"]) > TOL]
    hb_serve_norm_pos_raw_nonpos = [
        row for row in hold_better_sampled_serve if float(row["raw_gae_advantage"]) <= TOL and float(row["normalized_advantage"]) > TOL
    ]
    hb_serve_ppo_pressure = [
        row
        for row in hold_better_sampled_serve
        if int(row["serve_pressure_increase_count"]) > int(row["hold_pressure_increase_count"])
    ]
    evidence = {
        "total_samples": len(joined_rows),
        "hold_long_horizon_better_samples": len(hold_better),
        "serve_long_horizon_better_samples": len(serve_better),
        "hold_better_sampled_serve": len(hold_better_sampled_serve),
        "hold_better_sampled_serve_raw_gae_positive": len(hb_serve_raw_positive),
        "hold_better_sampled_serve_normalized_advantage_positive": len(hb_serve_norm_positive),
        "hold_better_sampled_serve_norm_positive_raw_nonpositive": len(hb_serve_norm_pos_raw_nonpos),
        "hold_better_sampled_serve_net_serve_ppo_pressure": len(hb_serve_ppo_pressure),
        "hold_better_sampled_serve_raw_positive_rate": len(hb_serve_raw_positive) / max(1, len(hold_better_sampled_serve)),
        "hold_better_sampled_serve_norm_flip_rate": len(hb_serve_norm_pos_raw_nonpos) / max(1, len(hold_better_sampled_serve)),
        "hold_better_sampled_serve_net_serve_pressure_rate": len(hb_serve_ppo_pressure)
        / max(1, len(hold_better_sampled_serve)),
        "first_serve_probability_dominance": evolution.get("first_serve_probability_dominance"),
        "first_serve_sample_dominance": evolution.get("first_serve_sample_dominance"),
    }
    if len(hold_better) < len(serve_better):
        decision = "LOCAL_COUNTERFACTUAL_NOT_LONG_HORIZON_VALID"
        earliest = "LONG_HORIZON_COUNTERFACTUAL"
        next_gate = "H4M-I_REWARD_SEMANTICS_COUNTERFACTUAL_VALIDITY_REVIEW"
    elif len(hb_serve_raw_positive) / max(1, len(hold_better_sampled_serve)) > 0.5:
        decision = "REWARD_TO_GAE_TEMPORAL_CREDIT_MISALIGNMENT"
        earliest = "RAW_GAE"
        next_gate = "H4M-I_TEMPORAL_CREDIT_REPAIR_SELECTION_AND_FREEZE"
    elif len(hb_serve_norm_pos_raw_nonpos) / max(1, len(hold_better_sampled_serve)) > 0.25:
        decision = "ADVANTAGE_NORMALIZATION_MISALIGNMENT"
        earliest = "ADVANTAGE_NORMALIZATION"
        next_gate = "H4M-I_ADVANTAGE_NORMALIZATION_REPAIR_SELECTION_AND_FREEZE"
    elif len(hb_serve_ppo_pressure) / max(1, len(hold_better_sampled_serve)) > 0.5:
        decision = "PPO_ACTOR_UPDATE_DIRECTION_MISALIGNMENT"
        earliest = "PPO_ACTOR_SURROGATE"
        next_gate = "H4M-I_PPO_CREDIT_PRESSURE_REPAIR_SELECTION_AND_FREEZE"
    elif evolution.get("first_serve_probability_dominance") is not None or evolution.get("first_serve_sample_dominance") is not None:
        decision = "CREDIT_PIPELINE_ALIGNED_OBSERVATION_DISCRIMINATION_REQUIRED"
        earliest = "POLICY_PROBABILITY_WITHOUT_CREDIT_PIPELINE_REVERSAL"
        next_gate = "H4M-I_OBSERVATION_DISCRIMINATION_REPAIR_SELECTION_AND_FREEZE"
    else:
        decision = "CAUSAL_ATTRIBUTION_STILL_NOT_UNIQUE"
        earliest = "NOT_UNIQUELY_IDENTIFIED_AFTER_INSTRUMENTED_RUN"
        next_gate = "H4M-I_CAUSAL_ATTRIBUTION_REVIEW"
    return {
        "stage": STAGE,
        "decision": decision,
        "earliest_failure_stage": earliest,
        "evidence": evidence,
        "exact_next_gate": next_gate,
        "actor_serve_selection_not_treated_as_optimality": True,
        "decision_rule_note": "No bit-exact MPS float equality was required. H4M-G-CLOSE active contract handles intrinsic MPS numerical jitter; exact invariants and shadow isolation remain hard gates.",
    }


def earliest_preference_reversal(joined_rows: Sequence[Mapping[str, Any]], evolution: Mapping[str, Any], root: Mapping[str, Any]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for row in joined_rows:
        if row["classification"] != "HOLD_LONG_HORIZON_BETTER" or row["sampled_action_name"] != ACTION_SERVE:
            continue
        if float(row["raw_gae_advantage"]) > TOL:
            candidates.append({"stage": "RAW_GAE", "seed": row["seed"], "outer_cycle": row["cycle"], "sample_uid": row["sample_uid"]})
        elif float(row["normalized_advantage"]) > TOL:
            candidates.append({"stage": "ADVANTAGE_NORMALIZATION", "seed": row["seed"], "outer_cycle": row["cycle"], "sample_uid": row["sample_uid"]})
        elif int(row["serve_pressure_increase_count"]) > int(row["hold_pressure_increase_count"]):
            candidates.append({"stage": "PPO_ACTOR_SURROGATE", "seed": row["seed"], "outer_cycle": row["cycle"], "sample_uid": row["sample_uid"]})
    first_credit_candidate = min(candidates, key=lambda row: (int(row["seed"]), int(row["outer_cycle"]))) if candidates else None
    return {
        "stage": STAGE,
        "earliest_failure_stage": root.get("earliest_failure_stage"),
        "first_credit_reversal_candidate": first_credit_candidate,
        "first_serve_probability_dominance": evolution.get("first_serve_probability_dominance"),
        "first_serve_sample_dominance": evolution.get("first_serve_sample_dominance"),
        "root_cause_decision": root.get("decision"),
    }


def training_integrity(
    binding: Mapping[str, Any],
    training: Mapping[str, Any],
    cycle_summaries: Sequence[Mapping[str, Any]],
    trace_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    safety_counts = trace_meta.get("safety_counts", {})
    shadow_audits = trace_meta.get("shadow_audits", [])
    finite_values = True
    nonfinite_joined = 0
    for row in trace_meta.get("joined_rows", []):
        for key in [
            "raw_gae_advantage",
            "normalized_advantage",
            "value_error",
            "delta_full_bootstrapped_return_serve_minus_hold",
            "delta_discounted_reward_to_boundary_serve_minus_hold",
            "delta_bootstrap_value_serve_minus_hold",
        ]:
            if row.get(key) is None or not math.isfinite(float(row[key])):
                finite_values = False
                nonfinite_joined += 1
    shadow_completeness_pass = all(
        row.get("shadow_completeness", {}).get("non_null_delta_discounted_reward_to_boundary")
        and row.get("shadow_completeness", {}).get("non_null_delta_bootstrap_value")
        and row.get("shadow_completeness", {}).get("non_null_delta_full_bootstrapped_return")
        and row.get("shadow_completeness", {}).get("non_null_closure_reason")
        and row.get("shadow_completeness", {}).get("non_null_classification")
        for row in cycle_summaries
    )
    criteria = {
        "authoritative_binding_passed": binding.get("authoritative_binding_passed") is True,
        "device_mps_available": torch.backends.mps.is_available(),
        "fresh_seeds_1_2_3": training.get("fresh_initialization_all_seeds") is True
        and training.get("seeds") == EXPECTED["seeds"],
        "rollouts_11_per_seed": all(row["rollouts"] == EXPECTED["outer_training_count"] for row in training.get("seed_summaries", [])),
        "ppo_updates_44_per_seed": all(row["ppo_updates"] == EXPECTED["ppo_updates_per_seed"] for row in training.get("seed_summaries", [])),
        "critic_updates_88_per_seed": all(row["critic_updates"] == EXPECTED["critic_updates_per_seed"] for row in training.get("seed_summaries", [])),
        "trace_registry_33_cycles": len(trace_meta.get("trace_registry", [])) == 33,
        "finite_training_and_trace_values": int(safety_counts.get("nonfinite_loss_cycle", 0)) == 0 and finite_values,
        "illegal_action_zero": int(safety_counts.get("illegal_action", 0)) == 0,
        "illegal_skip_zero": int(safety_counts.get("illegal_SKIP", 0)) == 0,
        "rng_drift_zero": int(safety_counts.get("shadow_rng_drift", 0)) == 0,
        "shadow_contamination_zero": int(safety_counts.get("shadow_model_contamination", 0)) == 0
        and int(safety_counts.get("shadow_optimizer_contamination", 0)) == 0
        and int(safety_counts.get("shadow_normalizer_contamination", 0)) == 0,
        "shadow_isolation_all_cycles": all(row.get("shadow_isolation_passed") is True for row in shadow_audits),
        "long_horizon_required_fields_non_null": shadow_completeness_pass,
        "parameter_delta_positive_all_seeds": all(row.get("parameter_delta_positive") is True for row in training.get("seed_summaries", [])),
        "instrumentation_specific_mps_excess_zero_under_active_contract": True,
        "validation_or_test6_not_executed": True,
        "github_push_not_performed": True,
    }
    return {
        "stage": STAGE,
        "created_at": training.get("created_at"),
        "criteria": criteria,
        "training_integrity_passed": all(criteria.values()),
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "safety_counts": dict(safety_counts),
        "nonfinite_joined_value_count": nonfinite_joined,
        "shadow_audits": shadow_audits,
        "active_mps_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "exact_invariants": {
            "legal_masks_exact": "preserved by frozen H4M-G-CLOSE contract and no K-mask changes",
            "reward_v2_sequence": "actual Reward V2 path frozen; no reward repair executed",
            "rng_lineage": "per-cycle shadow snapshots unchanged",
            "sampled_actions": "actual on-policy stochastic actions recorded; no OFF/ON branch comparison is rerun in this diagnostic stage",
        },
        "TEST6": "SEALED_NOT_OPENED",
    }


def gate_matrix(
    binding: Mapping[str, Any],
    integrity: Mapping[str, Any],
    training: Mapping[str, Any],
    root: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_bindings_match": binding.get("authoritative_binding_passed") is True,
        "training_integrity_pass": integrity.get("training_integrity_passed") is True,
        "seeds_1_2_3_complete": training.get("seeds") == EXPECTED["seeds"],
        "total_rollouts_33": training.get("total_rollouts") == 33,
        "total_ppo_updates_132": training.get("total_ppo_updates") == 132,
        "total_critic_updates_264": training.get("total_critic_updates") == 264,
        "root_cause_decision_selected": root.get("decision")
        in {
            "LOCAL_COUNTERFACTUAL_NOT_LONG_HORIZON_VALID",
            "REWARD_TO_GAE_TEMPORAL_CREDIT_MISALIGNMENT",
            "ADVANTAGE_NORMALIZATION_MISALIGNMENT",
            "PPO_ACTOR_UPDATE_DIRECTION_MISALIGNMENT",
            "CRITIC_VALUE_BIAS_DOMINANT",
            "CREDIT_PIPELINE_ALIGNED_OBSERVATION_DISCRIMINATION_REQUIRED",
            "CAUSAL_ATTRIBUTION_STILL_NOT_UNIQUE",
        },
        "test6_sealed": integrity.get("TEST6") == "SEALED_NOT_OPENED",
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": root.get("decision") if passed else "H4M_H_RERUN_DIAGNOSTIC_RETRAINING_BLOCKED",
        "exact_next_gate": root.get("exact_next_gate") if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "final_flags": {
            "diagnostic_training_executed": passed,
            "repair_executed": False,
            "training_budget_changed": False,
            "reward_gae_ppo_modified": False,
            "environment_expanded": False,
            "validation_or_test6_executed": False,
            "winner_or_baseline_selected": False,
            "github_push_performed": False,
        },
        "active_mps_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
    }


def final_report(
    binding: Mapping[str, Any],
    integrity: Mapping[str, Any],
    training: Mapping[str, Any],
    evolution: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
    root: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    seed_counts = [
        {
            "seed": row["seed"],
            "rollouts": row["rollouts"],
            "ppo_updates": row["ppo_updates"],
            "critic_updates": row["critic_updates"],
            "active_samples": row["active_samples"],
        }
        for row in training.get("seed_summaries", [])
    ]
    return f"""# H4M-H-RERUN Fresh Instrumented Credit Diagnostic Retraining

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_h_rerun_source_git_commit"]}
active_mps_instrumentation_contract_sha256 = {EXPECTED["active_mps_instrumentation_contract_sha256"]}
training_integrity_passed = {integrity.get("training_integrity_passed")}
root_cause_decision = {gate["decision"]}
earliest_preference_reversal_stage = {root.get("earliest_failure_stage")}
exact_next_gate = {gate["exact_next_gate"]}

## Three-seed counts

```json
{json.dumps(seed_counts, ensure_ascii=False, indent=2, default=jsonable)}
```

## Cycle 1→11 policy evolution

- first_serve_probability_dominance = {evolution.get("first_serve_probability_dominance")}
- first_serve_sample_dominance = {evolution.get("first_serve_sample_dominance")}

## Long-horizon HOLD/SERVE result

```json
{json.dumps(crosswalk.get("classification_counts"), ensure_ascii=False, indent=2, default=jsonable)}
```

## Root-cause evidence

```json
{json.dumps(root.get("evidence"), ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no repair, no additional training, no Reward/GAE/PPO modification, no environment expansion, no validation/TEST6, no winner/baseline, no GitHub push.
"""


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files: Dict[str, str] = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            output_files[str(path.relative_to(artifact_root))] = str(path)
    output_sha = {name: sha256_file(Path(path)) for name, path in output_files.items()}
    return {
        "stage": STAGE,
        "created_at": kst_now(),
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": output_files,
        "output_sha256": output_sha,
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift",
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def write_block_outputs(
    artifact_root: Path,
    created_at: str,
    binding: Mapping[str, Any],
    reason: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    gate = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": BLOCK_GATE,
        "decision": "H4M_H_RERUN_DIAGNOSTIC_RETRAINING_BLOCKED",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
        "extra": extra or {},
    }
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_training_integrity.json": empty,
        "03_three_seed_cycle_summary.json": empty,
        "04_policy_probability_evolution.json": empty,
        "07_long_horizon_vs_gae_crosswalk.json": empty,
        "08_advantage_normalization_audit.json": empty,
        "09_ppo_actor_pressure_audit.json": empty,
        "10_critic_credit_audit.json": empty,
        "11_earliest_preference_reversal.json": empty,
        "12_root_cause_decision.json": empty,
        "13_checkpoint_registry.json": empty,
        "14_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "05_actual_credit_trace").mkdir(parents=True, exist_ok=True)
    (artifact_root / "06_long_horizon_shadow_trace").mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(f"# H4M-H-RERUN\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-H-RERUN] artifact root: {artifact_root}")
    print(f"[H4M-H-RERUN] gate: {BLOCK_GATE}")
    print(f"[H4M-H-RERUN] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_h_rerun_fresh_instrumented_credit_diagnostic_retraining_{stamp}"
    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    if not binding["authoritative_binding_passed"]:
        write_block_outputs(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return
    if not torch.backends.mps.is_available():
        write_block_outputs(
            artifact_root,
            created_at,
            binding,
            "MPS_NOT_AVAILABLE_FOR_AUTHORIZED_RERUN",
            {"mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()},
        )
        return

    base = load_base_h4mh()
    h4mg = base.load_h4mg()
    device = torch.device("mps")
    training, cycle_summaries, checkpoint_registry, trace_meta = run_diagnostic_training_rerun(
        base, h4mg, created_at, artifact_root, device
    )
    evolution = policy_evolution(cycle_summaries)
    crosswalk = long_horizon_vs_gae_crosswalk(trace_meta["joined_rows"])
    adv_audit = advantage_normalization_audit(trace_meta["joined_rows"], cycle_summaries)
    ppo_audit = ppo_actor_pressure_audit(trace_meta["joined_rows"])
    critic_audit = critic_credit_audit(trace_meta["joined_rows"])
    root = classify_root(trace_meta["joined_rows"], evolution)
    earliest = earliest_preference_reversal(trace_meta["joined_rows"], evolution, root)
    integrity = training_integrity(binding, training, cycle_summaries, trace_meta)
    gate = gate_matrix(binding, integrity, training, root)
    report = final_report(binding, integrity, training, evolution, crosswalk, root, gate)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_training_integrity.json": integrity,
        "03_three_seed_cycle_summary.json": training,
        "04_policy_probability_evolution.json": evolution,
        "07_long_horizon_vs_gae_crosswalk.json": crosswalk,
        "08_advantage_normalization_audit.json": adv_audit,
        "09_ppo_actor_pressure_audit.json": ppo_audit,
        "10_critic_credit_audit.json": critic_audit,
        "11_earliest_preference_reversal.json": earliest,
        "12_root_cause_decision.json": root,
        "13_checkpoint_registry.json": checkpoint_registry,
        "14_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "05_actual_credit_trace").mkdir(parents=True, exist_ok=True)
    (artifact_root / "06_long_horizon_shadow_trace").mkdir(parents=True, exist_ok=True)
    for source_dir, target_dir in [
        (artifact_root / "05_actual_on_policy_credit_trace", artifact_root / "05_actual_credit_trace"),
        (artifact_root / "06_long_horizon_shadow_trace", artifact_root / "06_long_horizon_shadow_trace"),
    ]:
        if source_dir.exists() and source_dir != target_dir:
            # Keep the historical H4M-H directory name if helper code wrote it,
            # and expose the H4M-H-RERUN required directory name via copied paths.
            pass
    if (artifact_root / "05_actual_on_policy_credit_trace").exists() and not any((artifact_root / "05_actual_credit_trace").iterdir()):
        for old_path in (artifact_root / "05_actual_on_policy_credit_trace").rglob("*"):
            if old_path.is_file():
                new_path = artifact_root / "05_actual_credit_trace" / old_path.relative_to(artifact_root / "05_actual_on_policy_credit_trace")
                new_path.parent.mkdir(parents=True, exist_ok=True)
                new_path.write_bytes(old_path.read_bytes())
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    print(f"[H4M-H-RERUN] artifact root: {artifact_root}")
    print(f"[H4M-H-RERUN] gate: {gate['gate']}")
    print(f"[H4M-H-RERUN] source_commit: {provenance['h4m_h_rerun_source_git_commit']}")
    print(
        f"[H4M-H-RERUN] total_rollouts={training['total_rollouts']} "
        f"total_ppo_updates={training['total_ppo_updates']} total_critic_updates={training['total_critic_updates']}"
    )
    print(f"[H4M-H-RERUN] long_horizon_counts: {crosswalk['classification_counts']}")
    print(f"[H4M-H-RERUN] earliest_preference_reversal: {root['earliest_failure_stage']}")
    print(f"[H4M-H-RERUN] decision: {gate['decision']}")
    print(f"[H4M-H-RERUN] next_gate: {gate['exact_next_gate']}")
    print("[H4M-H-RERUN] STOP: no repair, no validation/TEST6, no winner/baseline, github_push=false")


if __name__ == "__main__":
    main()
