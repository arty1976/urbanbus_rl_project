#!/usr/bin/env python3
"""BT6 mandatory MPS preflight and fail-closed blocked-artifact writer.

This module deliberately stops before capability grant, simulator mutation,
optimizer use, or checkpoint creation when Apple MPS is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence
from zoneinfo import ZoneInfo

import torch


STAGE = "H4M-AE-R9.8-LS3-BT6"
BLOCK_GATE = "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
BT6_S0_SOURCE = "34742b1c11f3d7ea437c990d6d549c7e45372e6d"
BT6_S0_ARTIFACT = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_s0_postrepair_scale_redesign_20260822_152944+09:00"
BT5R_SOURCE = "f14fab7dbc72f42968a1aea5dcbbb652dddbeb48"
BT5_SOURCE = "1df8284d161e58e8caeb07a1e0b037d95c808b7d"
BT4_SOURCE = "99617bd31a0ae383f2541ab58d677f42c84bec74"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
FROZEN = {
    "gatv2_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2": "rewards/mappo_reward_v1.py",
    "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py",
    "candidate_support_deconfounding": "joint_candidate_support_snapshot.py",
    "causal_bridge": "causal_kpi_bridge.py",
    "r9_8_authorization": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py",
    "joint_assignment_learning": "joint_assignment_learning.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py",
    "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
}
ENVELOPE = {
    "label": "R2_MODERATE_POST_REPAIR", "distinct_windows": 12, "window_visits": 24,
    "requests": 97, "agents": 8, "seeds": [20260822, 20260823],
    "assignment_decisions": 96, "trajectories": 24, "trajectory_length": 4,
    "causal_transitions": 192, "optimizer_updates": 10,
}
LOCKS = {
    "training_allowed": False, "simulator_execution_allowed": False,
    "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, check=True, text=True,
                          capture_output=True).stdout.strip()


def frozen_hashes() -> Dict[str, str]:
    return {name: sha256(ROOT / relative) for name, relative in FROZEN.items()}


def mps_preflight() -> Dict[str, Any]:
    available = bool(torch.backends.mps.is_available())
    return {
        "required": {"mps_available": True, "device": "mps", "cpu_fallback_allowed": False},
        "observed": {
            "torch_version": torch.__version__, "python_version": sys.version,
            "platform": platform.platform(), "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": available, "selected_device": "mps" if available else None,
            "cpu_device_selected": False,
        },
        "passed": available,
        "failure_reason": None if available else "torch.backends.mps.is_available() returned false",
        "environment_mutation": False,
    }


def provenance() -> Dict[str, Any]:
    head = git(["rev-parse", "HEAD"])
    parent = git(["rev-parse", "HEAD^"])
    files = [line for line in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if line]
    return {
        "source_commit": head, "source_parent": parent, "source_only_local_commit": files == [SOURCE_REL.as_posix()],
        "parent_is_bt6_s0": parent == BT6_S0_SOURCE, "changed_files": files,
        "github_push_performed": False,
    }


def empty_candidate_audit() -> Dict[str, Any]:
    return {
        "execution_status": "NOT_EXECUTED_MPS_PREFLIGHT_BLOCKED", "required_decisions": 96,
        "recorded_decisions": 0, "candidate_count_before_zero_loss": 0,
        "candidate_count_after_zero_loss": 0, "joint_support_including_no_assign": 0,
        "selected_agent": [], "selected_candidate": [], "candidate_identity": [],
        "by_time_band": {band: {"decisions": 0, "zero_candidate": 0, "one_candidate": 0,
                                  "genuine_2plus_candidate": 0} for band in ("night", "offpeak", "peak")},
        "zero_candidate_decisions": 0, "one_candidate_decisions": 0,
        "genuine_2plus_candidate_decisions": 0, "candidate_regeneration": 0,
        "candidate_identity_mismatch": 0, "request_density_strata": [],
    }


def empty_zero_loss_audit() -> Dict[str, Any]:
    return {
        "execution_status": "NOT_EXECUTED_MPS_PREFLIGHT_BLOCKED", "candidates_evaluated": 0,
        "pass_count": 0, "fail_count": 0, "selective_states": 0,
        "selective_by_time_band": {band: 0 for band in ("night", "offpeak", "peak")},
        "zero_loss_violations": 0, "rejected_candidate_selected": 0,
    }


def empty_learning_audit() -> Dict[str, Any]:
    return {
        "execution_status": "NOT_EXECUTED_MPS_PREFLIGHT_BLOCKED", "informative_decisions": 0,
        "non_zero_team_reward_fraction": None, "multi_step_trajectories": 0,
        "nonterminal_transitions": 0, "gae_recursive_term_count": 0,
        "temporally_propagated_advantages": 0, "advantage_distribution": None,
        "critic_target_distribution": None, "actor_gradient_norms": [], "critic_gradient_norms": [],
        "ppo_ratio_distribution": None, "entropy": [], "policy_loss": [], "critic_loss": [],
        "cross_window_contamination": 0, "cross_seed_contamination": 0,
        "legacy_advantage_contamination": 0, "future_leakage": 0,
    }


def main() -> None:
    started = time.perf_counter()
    preflight = mps_preflight()
    if preflight["passed"]:
        raise SystemExit("This fail-closed preflight writer is only valid for a missing-MPS block; do not grant BT6 capability through it.")
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_mps_preflight_block_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    before = frozen_hashes()
    source = provenance()
    root.mkdir(parents=True)
    counters = {
        "distinct_windows": 0, "window_visits": 0, "requests": 0, "seeds_completed": [], "agents": 0,
        "assignment_decisions": 0, "trajectories": 0, "causal_transitions": 0, "optimizer_steps": 0,
        "simulator_mutations": 0, "checkpoint_writes": 0, "temporary_capability_grants": 0,
        "TEST6_access": 0,
    }
    candidate = empty_candidate_audit()
    zero_loss = empty_zero_loss_audit()
    learning = empty_learning_audit()
    violations = {
        "zero_loss": 0, "illegal_or_masked_selection": 0, "candidate_identity_mismatch": 0,
        "candidate_regeneration": 0, "no_assign_violation": 0, "cross_window_contamination": 0,
        "cross_seed_contamination": 0, "future_leakage": 0, "source_state_mutation": 0,
        "nan_or_inf": 0, "unauthorized_optimizer_step": 0,
    }
    after = frozen_hashes()
    execution = {
        "stage": STAGE, "gate": BLOCK_GATE, "classification": "MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE_BEFORE_BT6_CAPABILITY_GRANT",
        "source_commit": source["source_commit"], "required_envelope": ENVELOPE, "actual_counts": counters,
        "execution_started": False, "block_boundary": "mandatory MPS preflight before any temporary capability grant",
        "checkpoint": {"created": False, "test_only": True, "bounded": True, "non_promotable": True,
                       "winner": False, "best_model": False, "promotion": False,
                       "performance_claim_allowed": False, "paper_level_claim_allowed": False,
                       "causal_performance_claim_allowed": False}, "global_locks": LOCKS,
    }
    readiness = {
        "passed": False, "status": "NOT_EVALUABLE_MPS_PREFLIGHT_BLOCKED",
        "failed_or_unevaluated_criteria": ["MPS preflight", "all 12 windows", "all 24 visits", "both approved seeds",
                                             "all time bands", "informative decisions >= 16", "genuine 2+ candidates >= 3 and >= 1/band",
                                             "Zero-Loss selective states >= 3 and >= 1/band", "24 four-step trajectories",
                                             "GAE recursion", "finite actor/critic gradients"],
        "integrity_violations": violations,
    }
    adversarial = {
        "missing_mps": {"expected_fail_closed": True, "observed": True, "passed": True},
        "training_without_capability": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "simulator_without_capability": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "budget_overrun": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "unapproved_seed": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "candidate_order_or_id_tamper": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "training_time_regeneration": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "zero_loss_fail_forced_selection": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "no_assign_removal": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "trajectory_merge": "NOT_EXECUTED_PRECONDITION_BLOCKED",
        "frozen_mutation": {"before_after_equal": before == after, "passed": before == after},
        "checkpoint_promotion_attempt": "NOT_EXECUTED_PRECONDITION_BLOCKED",
    }
    files = {
        "bt6_mps_preflight.json": preflight,
        "bt6_execution_manifest.json": execution,
        "bt6_authorization_audit.json": {"mps_preflight_passed": False, "temporary_capability_grants": [],
                                            "temporary_capability_revoked": True, "global_locks": LOCKS,
                                            "authorization_before_mutation": True},
        "bt6_candidate_support_audit.json": candidate,
        "bt6_zero_loss_selectivity_audit.json": zero_loss,
        "bt6_learning_signal_audit.json": learning,
        "bt6_bt7_readiness_audit.json": readiness,
        "bt6_optimizer_audit.json": {"optimizer_steps": 0, "actor_gradient_evidence": [], "critic_gradient_evidence": [],
                                        "ppo_ratio_distribution": None, "blocked_before_optimizer": True},
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after,
                                            "joint_assignment_actor_parameters": "not_instantiated", "joint_assignment_critic_parameters": "not_instantiated"},
        "test_results.json": {"preflight": preflight, "source_provenance": source, "adversarial_probes": adversarial,
                                "actual_counts": counters, "integrity_violations": violations, "hard_failures": [BLOCK_GATE],
                                "warnings": [], "github_push_performed": False},
        "gate_decision.json": {"gate": BLOCK_GATE, "classification": execution["classification"],
                                 "source_commit": source["source_commit"], "BT6_S0_source": BT6_S0_SOURCE,
                                 "lineage": {"BT4": BT4_SOURCE, "BT5": BT5_SOURCE, "BT5_R": BT5R_SOURCE,
                                             "BT6_S0": BT6_S0_SOURCE}, "hard_failures": [BLOCK_GATE], "warnings": [],
                                 "global_locks": LOCKS, "next_step": "Restore an Apple MPS runtime, then rerun BT6 preflight; no execution is authorized in this runtime."},
    }
    for name, payload in files.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — MPS preflight block\n\n"
        f"gate = {BLOCK_GATE}\nsource_commit = {source['source_commit']}\n\n"
        f"MPS built / available = {preflight['observed']['mps_built']} / {preflight['observed']['mps_available']}. "
        "No temporary capability was granted, and no simulator mutation, optimizer step, or checkpoint write occurred.\n\n"
        "BT7 readiness was not evaluated because the mandatory execution precondition failed. "
        "Restore an Apple MPS runtime and rerun BT6 preflight before any execution authorization.\n", encoding="utf-8")
    hashed = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": BLOCK_GATE, "source_commit": source["source_commit"],
                                   "file_sha256": hashed, "elapsed_seconds": round(time.perf_counter() - started, 3),
                                   **counters, "github_push_performed": False})
    (root / "_BLOCKED.lock").write_text(BLOCK_GATE + "\n", encoding="utf-8")
    print(f"[BLOCKED] {BLOCK_GATE}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
