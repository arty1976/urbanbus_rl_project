#!/usr/bin/env python3
"""BT7 frozen-policy review: fail closed when BT6 lacks replayable snapshots.

This audit deliberately reads only the BT6 artifact.  It never rebuilds a
candidate, instantiates an optimizer, advances the causal adapter, or loads a
policy for substitute inference when the authoritative stored inputs are not
sufficient to reproduce that inference.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence
from zoneinfo import ZoneInfo


STAGE = "H4M-AE-R9.8-LS3-BT7"
BLOCK = "BLOCKED_INSUFFICIENT_FROZEN_DECISION_SNAPSHOT_EVIDENCE"
BT6_SOURCE = "b404014ac35747ad9c95936fd8ad3fd142a619e6"
BT6_S0_SOURCE = "34742b1c11f3d7ea437c990d6d549c7e45372e6d"
BT5R_SOURCE = "f14fab7dbc72f42968a1aea5dcbbb652dddbeb48"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_160245+09:00"
BT6_S0 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_s0_postrepair_scale_redesign_20260822_152944+09:00"
BT5R = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt5r_candidate_support_repair_selection_20260822_151517+09:00"
FROZEN = {
    "gatv2_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2": "rewards/mappo_reward_v1.py", "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py", "candidate_support_deconfounding": "joint_candidate_support_snapshot.py",
    "causal_bridge": "causal_kpi_bridge.py", "r9_8_authorization": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py", "joint_assignment_learning": "joint_assignment_learning.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py", "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


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
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def frozen_hashes() -> Dict[str, str]:
    return {name: sha256(ROOT / relative) for name, relative in FROZEN.items()}


def provenance() -> Dict[str, Any]:
    head, parent = git(["rev-parse", "HEAD"]), git(["rev-parse", "HEAD^"])
    files = [line for line in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if line]
    return {"source_commit": head, "source_parent": parent, "source_only_local_commit": files == [SOURCE_REL.as_posix()],
            "changed_files": files, "github_push_performed": False}


def snapshot_evidence() -> Dict[str, Any]:
    candidate = json.loads((BT6 / "bt6_candidate_support_audit.json").read_text(encoding="utf-8"))
    decisions = candidate.get("decisions", [])
    required = {"global_feats", "demand_feats", "agent_feats", "agent_mask", "candidate_feats", "safe_mask",
                "pair_agent_index", "frozen_snapshot_digest"}
    present = set().union(*(set(row) for row in decisions)) if decisions else set()
    missing = sorted(required - present)
    checkpoint = BT6 / "bt6_r2_joint_assignment_checkpoint.pt"
    return {"authoritative_decision_rows": len(decisions), "expected_decision_rows": 96,
            "checkpoint_exists": checkpoint.is_file(), "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
            "stored_decision_fields": sorted(present), "required_replay_fields": sorted(required), "missing_replay_fields": missing,
            "identity_only_evidence": all("candidate_identity" in row for row in decisions),
            "replayable_frozen_actor_input_for_all_96": len(decisions) == 96 and not missing,
            "candidate_regeneration_performed": 0, "causal_rollout_performed": 0,
            "finding": "BT6 preserves decision metadata and checkpoint weights but not the tensor/safe-mask inputs required for frozen Actor replay."}


def main() -> None:
    started = time.perf_counter()
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_frozen_behavior_review_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    before = frozen_hashes()
    source = provenance()
    evidence = snapshot_evidence()
    bt6_gate = json.loads((BT6 / "gate_decision.json").read_text(encoding="utf-8"))
    binding = {"bt6_gate": bt6_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_POST_REPAIR_EXTENDED_BOUNDED_TRAINING_AND_BT7_READINESS_COMPLETE",
               "bt6_source": bt6_gate.get("source_commit") == BT6_SOURCE, "bt6s0_exists": BT6_S0.is_dir(),
               "bt5r_exists": BT5R.is_dir(), "source_only_local_commit": source["source_only_local_commit"]}
    after = frozen_hashes()
    if evidence["replayable_frozen_actor_input_for_all_96"]:
        raise SystemExit("This blocker is inapplicable: replayable snapshots exist; use the BT7 inference executor.")
    root.mkdir(parents=True)
    unavailable = {"status": "NOT_EVALUATED_FROZEN_INPUT_SNAPSHOTS_INSUFFICIENT", "reason": BLOCK,
                   "candidate_regeneration_performed": 0, "causal_rollout_performed": 0, "optimizer_steps": 0,
                   "parameter_mutation": 0, "new_training_checkpoint": 0}
    outputs = {
        "bt7_frozen_inference_manifest.json": {"status": "BLOCKED_BEFORE_INFERENCE", "snapshot_evidence": evidence,
                                                 "checkpoint_read_for_inference": False, "frozen_policy_inference_reproducible": False},
        "bt7_no_assign_behavior.json": unavailable,
        "bt7_agent_concentration.json": unavailable,
        "bt7_candidate_diversity.json": unavailable,
        "bt7_zero_loss_interaction.json": unavailable,
        "bt7_timeband_density_behavior.json": unavailable,
        "bt7_seed_stability.json": {**unavailable, "status": "SEED_LEVEL_FROZEN_POLICY_COMPARISON_NOT_AVAILABLE"},
        "bt7_collapse_degeneracy_audit.json": {**unavailable, "classification": "INSUFFICIENT_EVIDENCE"},
        "bt7_invariance_tests.json": {**unavailable, "candidate_order_invariance": "NOT_EVALUABLE_WITHOUT_FROZEN_TENSORS",
                                        "agent_order_invariance": "NOT_EVALUABLE_WITHOUT_FROZEN_TENSORS"},
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after},
        "test_results.json": {"binding": binding, "snapshot_evidence": evidence, "optimizer_steps": 0,
                              "causal_simulator_rollouts": 0, "candidate_regeneration": 0, "parameter_mutation": 0,
                              "TEST6_access": 0, "hard_failures": [BLOCK], "warnings": [], "github_push_performed": False},
        "gate_decision.json": {"gate": BLOCK, "classification": "BT7_FROZEN_POLICY_INFERENCE_NOT_REPRODUCIBLE_FROM_PRESERVED_EVIDENCE",
                               "source_commit": source["source_commit"], "lineage": {"BT5_R": BT5R_SOURCE, "BT6_S0": BT6_S0_SOURCE, "BT6": BT6_SOURCE,
                               "bt5r_manifest_sha256": sha256(BT5R / "manifest.json"), "bt6s0_manifest_sha256": sha256(BT6_S0 / "manifest.json"),
                               "bt6_manifest_sha256": sha256(BT6 / "manifest.json")}, "hard_failures": [BLOCK], "warnings": [], "global_locks": LOCKS,
                               "next_step": "Snapshot-preservation repair selection gate; do not rerun the causal simulator."},
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Frozen policy behavior review blocked\n\n"
        f"gate = {BLOCK}\nsource_commit = {source['source_commit']}\n\n"
        "BT6 checkpoint weights are present, but the 96 decision records omit the global/demand/agent/candidate feature tensors, legal mask, "
        "pair-to-agent index, and frozen support digest required to reproduce a final-policy forward pass. Rebuilding them would be forbidden candidate regeneration.\n\n"
        "No policy was loaded for substitute inference. Optimizer steps, parameter mutation, causal rollout, candidate regeneration, and new checkpoint writes are all zero.\n",
        encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": BLOCK, "source_commit": source["source_commit"], "file_sha256": manifest,
                                   "elapsed_seconds": round(time.perf_counter() - started, 3), "github_push_performed": False})
    (root / "_BLOCKED.lock").write_text(BLOCK + "\n", encoding="utf-8")
    print(f"[BLOCKED] {BLOCK}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
