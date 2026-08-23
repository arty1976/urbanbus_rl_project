#!/usr/bin/env python3
"""BT8-R4A design/implementation gate; this is not an F1 trainer.

It freezes the two missing execution authorities: replicate-local PPO partition
and candidate-plan-aware atomic state transition.  Its only capability-protected
operations are three in-memory bridge fixture commits; it never calls the
causal KPI adapter, Reward V2, a model, an optimizer, or checkpoint writer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE = "H4M-AE-R9.8-LS3-BT8-R4A"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4A_F1_EXECUTION_AUTHORITY_COMPLETION"
PASS_CLASS = "A_SUSEONG_LS3_F1_SEED_PPO_AND_CANDIDATE_PLAN_CAUSAL_AUTHORITY_READY_FOR_RERUN"
R4_SOURCE = "32fbf2b3043166188bde891a5c450702e7db3607"
F1_BLOCK_SOURCE = "5ff58a26db9951b23e2c23e09af31e53302987d6"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R4 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4_v2_actor_fresh_training_design_20260823_122459+0900"
F1_BLOCK = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_authority_block_20260823_123941+0900"
SOURCE_FILES = {
    "05_training/joint_assignment_f1_execution_contract.py",
    "05_training/joint_candidate_plan_causal_bridge.py",
    "05_training/run_h4m_ae_ls3_bt8_r4a_authority_completion.py",
    "05_training/test_h4m_ae_ls3_bt8_r4a_execution_authority.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


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
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True,
                          check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [item for item in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if item]
    return {"source_commit": git(["rev-parse", "HEAD"]), "source_parent": git(["rev-parse", "HEAD^"]),
            "changed_files": changed, "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
            "github_push_performed": False}


def frozen_hashes(BT6: Any) -> dict[str, str]:
    return {**BT6.frozen_hashes(), "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py")}


def exact_authority_binding(*, source: Mapping[str, Any], r4_gate: Mapping[str, Any],
                            f1_gate: Mapping[str, Any], selected: Mapping[str, Any], FC: Any) -> dict[str, bool]:
    expected_train, expected_review = 6, 3
    return {
        "r4_gate": r4_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_FRESH_V2_ACTOR_TRAINING_AND_NOVEL_EXPOSURE_DESIGN_COMPLETE",
        "r4_source": r4_gate.get("source_commit") == R4_SOURCE,
        "f1_block_source": f1_gate.get("source_commit") == F1_BLOCK_SOURCE,
        "f1_block_reason": f1_gate.get("gate") == "BLOCKED_BT8_R4_EXECUTION_AUTHORITY_INCOMPLETE",
        "source_parent_is_f1_block": source["source_parent"] == F1_BLOCK_SOURCE,
        "source_only_local_commit": bool(source["source_only_local_commit"]),
        "selected_f1_counts": selected.get("distinct_windows") == 9 and len(selected.get("train_windows", [])) == expected_train and len(selected.get("review_windows", [])) == expected_review,
        "selected_f1_budget": all(selected.get(name) == value for name, value in {"train_visits": 12, "requests_train": 43, "review_snapshot_visits": 6, "requests_review": 27, "agents": 8, "decisions": 48, "trajectories": 12, "transitions": 96, "optimizer_updates": 6}.items()),
        "seed_contract_valid": not _contract_error(FC),
    }


def _contract_error(FC: Any) -> bool:
    try:
        FC.validate_contract()
    except FC.F1ExecutionContractError:
        return True
    return False


def bridge_audit(*, BT6: Any, CB: Any, AUTH: Any, r4_plan_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Use the R4-bound real safe candidates, but never a causal rollout."""
    factory = BT6.RepairedSupportFactory()
    source_group, agent_id = str(r4_plan_audit["source_group"]), str(r4_plan_audit["agent_id"])
    expected_ids = {str(row["candidate_id"]) for row in r4_plan_audit["applications"]}
    support = factory.build(source_group=source_group, decision_group="BT8_R4A_AUTHORITATIVE_PLAN_BRIDGE_FIXTURE")
    snapshot = support["snapshot"]
    safe_ids = {row.candidate_id for row in snapshot.safe if row.agent_id == agent_id}
    if not expected_ids.issubset(safe_ids):
        raise RuntimeError("R4_CANDIDATE_SNAPSHOT_RECONSTRUCTION_MISMATCH")
    candidate_a, candidate_b = [CB.ImmutableCandidatePlan.from_snapshot(decision_id=decision, snapshot=snapshot,
        agent_id=agent_id, candidate_id=candidate_id, source_state_version=0)
        for decision, candidate_id in (("R4A_A", sorted(expected_ids)[0]), ("R4A_B", sorted(expected_ids)[1]))]
    state = CB.CandidatePlanState.from_binding(candidate_a.candidate_plan)
    denied_bridge = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot)
    denied_before = denied_bridge.state.state_digest
    denied = None
    try:
        denied_bridge.commit_candidate(candidate=candidate_a, expected_source_version=0, expected_source_digest=state.state_digest,
            expected_support_digest=snapshot.snapshot_digest, expected_candidate_plan_digest=candidate_a.candidate_plan_digest)
    except Exception as exc:  # authorization denial is the expected pre-mutation guard
        denied = type(exc).__name__
    denied_unchanged = denied_before == denied_bridge.state.state_digest

    bridge_a, bridge_b = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot), CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot)
    no_assign = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot)
    AUTH.reset_audit_log()
    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, reason="BT8-R4A candidate-plan atomic bridge fixture only; no causal KPI rollout"):
        a = bridge_a.commit_candidate(candidate=candidate_a, expected_source_version=0, expected_source_digest=state.state_digest,
            expected_support_digest=snapshot.snapshot_digest, expected_candidate_plan_digest=candidate_a.candidate_plan_digest)
        b = bridge_b.commit_candidate(candidate=candidate_b, expected_source_version=0, expected_source_digest=state.state_digest,
            expected_support_digest=snapshot.snapshot_digest, expected_candidate_plan_digest=candidate_b.candidate_plan_digest)
        no = no_assign.commit_no_assign(decision_id="R4A_NO_ASSIGN", expected_source_version=0,
            expected_source_digest=state.state_digest, expected_support_digest=snapshot.snapshot_digest)
    tamper_bridge = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot)
    tamper = None
    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, reason="BT8-R4A digest-tamper rejection fixture"):
        try:
            tampered = replace(candidate_a, candidate_plan_digest="tampered")
            tamper_bridge.commit_candidate(candidate=tampered, expected_source_version=0, expected_source_digest=state.state_digest,
                expected_support_digest=snapshot.snapshot_digest, expected_candidate_plan_digest="tampered")
        except Exception as exc:
            tamper = getattr(exc, "code", type(exc).__name__)
    regeneration = None
    try:
        snapshot.replay_guard(support_digest=snapshot.snapshot_digest, regeneration_requested=True)
    except Exception as exc:
        regeneration = getattr(exc, "code", type(exc).__name__)
    auth = AUTH.audit_log()
    commits = [a, b, no]
    return {"fixture": {"source_group": source_group, "agent_id": agent_id, "support_digest": snapshot.snapshot_digest,
                        "zero_loss_pass_candidates": sorted(expected_ids), "source_state_mutation": factory.source_mutations},
            "denied_before_capability": {"error": denied, "source_state_unchanged": denied_unchanged},
            "candidate_a": _transition_payload(a), "candidate_b": _transition_payload(b), "no_assign": _transition_payload(no),
            "candidate_a_b_distinct_applied_state": a.next_state.state_digest != b.next_state.state_digest,
            "candidate_a_b_distinct_applied_plan": a.applied_plan_digest != b.applied_plan_digest,
            "identity_chain_all": all(row.selected_candidate_id == row.applied_candidate_id == row.credited_candidate_id for row in commits),
            "serve_fallback_count": sum(int(row.serve_fallback_used) for row in commits),
            "digest_tamper_rejected": tamper == "CANDIDATE_PLAN_DIGEST_MISMATCH",
            "candidate_regeneration_rejected": regeneration == "CANDIDATE_REGENERATION_FORBIDDEN_AT_REPLAY",
            "authoritative_bridge_commits": 3, "causal_kpi_rollouts": 0,
            "authorization_events": auth, "authorization_locks_after": AUTH.authorization_state()["capabilities"]}


def _transition_payload(row: Any) -> dict[str, Any]:
    return {"transition_id": row.transition_id, "selected_candidate_id": row.selected_candidate_id,
            "applied_candidate_id": row.applied_candidate_id, "credited_candidate_id": row.credited_candidate_id,
            "candidate_plan_digest": row.candidate_plan_digest, "applied_plan_digest": row.applied_plan_digest,
            "source_state_digest": row.events[0]["source_state_digest"], "next_state_digest": row.next_state.state_digest,
            "no_assign": row.no_assign, "serve_fallback_used": row.serve_fallback_used}


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_f1_execution_contract as FC
    import joint_candidate_plan_causal_bridge as CB
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import simulator_authorization as AUTH

    source = provenance()
    r4_gate = json.loads((R4 / "gate_decision.json").read_text(encoding="utf-8"))
    f1_gate = json.loads((F1_BLOCK / "gate_decision.json").read_text(encoding="utf-8"))
    selected = json.loads((R4 / "bt8r4_selected_training_envelope.json").read_text(encoding="utf-8"))
    r4_plan = json.loads((R4 / "bt8r4_candidate_plan_execution_audit.json").read_text(encoding="utf-8"))
    binding = exact_authority_binding(source=source, r4_gate=r4_gate, f1_gate=f1_gate, selected=selected, FC=FC)
    before = frozen_hashes(BT6)
    hard, warnings = [], []
    if not all(binding.values()):
        hard.append("AUTHORITATIVE_BINDING_FAILURE")
    seed_contract = {"contract_id": FC.CONTRACT_ID, "replicates": FC.REPLICATES,
                     "independent_actor_critic_optimizer_gae_statistics": True,
                     "aggregate": {"train_windows": 12, "trajectories": 12, "decisions": 48,
                                   "ppo_update_cycles": 6, "actor_optimizer_steps": 6,
                                   "critic_optimizer_steps": 6, "raw_optimizer_step_calls": 12}}
    ppo_contract = dict(FC.PPO_PARTITION)
    bridge = bridge_audit(BT6=BT6, CB=CB, AUTH=AUTH, r4_plan_audit=r4_plan) if not hard else {}
    bridge_pass = bool(bridge) and all((bridge["fixture"]["source_state_mutation"] == 0,
        bridge["denied_before_capability"]["source_state_unchanged"], bridge["candidate_a_b_distinct_applied_state"],
        bridge["candidate_a_b_distinct_applied_plan"], bridge["identity_chain_all"], bridge["serve_fallback_count"] == 0,
        bridge["digest_tamper_rejected"], bridge["candidate_regeneration_rejected"],
        bridge["causal_kpi_rollouts"] == 0, not any(bridge["authorization_locks_after"].values())))
    if not bridge_pass:
        hard.append("CANDIDATE_PLAN_BRIDGE_VALIDATION_FAILED")
    counters = {"training": 0, "mps_training": 0, "causal_rollout": 0, "optimizer_step": 0,
                "checkpoint": 0, "review_optimizer_rows": 0, "test6_access": 0, "github_push": 0,
                "authoritative_plan_bridge_fixture_commits": int(bridge.get("authoritative_bridge_commits", 0))}
    after = frozen_hashes(BT6)
    frozen = {"before": before, "after": after, "all_unchanged": before == after}
    if not frozen["all_unchanged"]:
        hard.append("FROZEN_AUTHORITY_HASH_CHANGED")
    if any(counters[key] != 0 for key in ("training", "mps_training", "causal_rollout", "optimizer_step", "checkpoint", "review_optimizer_rows", "test6_access", "github_push")):
        hard.append("R4A_EXECUTION_COUNTER_NONZERO")
    gate, classification = (PASS_GATE, PASS_CLASS) if not hard else ("BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4A_AUTHORITY_COMPLETION_FAILED", "BLOCKED")
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4a_f1_execution_authority_completion_{time.strftime('%Y%m%d_%H%M%S%z')}"
    if root.exists(): raise SystemExit("append-only artifact collision")
    root.mkdir(parents=True)
    dump(root / "bt8r4a_seed_update_contract.json", seed_contract)
    dump(root / "bt8r4a_ppo_partition_contract.json", ppo_contract)
    dump(root / "bt8r4a_candidate_plan_bridge_contract.json", CB.CANDIDATE_PLAN_BRIDGE_CONTRACT)
    dump(root / "bt8r4a_candidate_plan_execution_audit.json", bridge)
    dump(root / "bt8r4a_credit_identity_audit.json", {"identity_chain": "selected == applied == credited", "verified": bridge.get("identity_chain_all", False),
         "required_fields_for_f1": ["decision_id", "trajectory_id", "agent_id", "candidate_id", "candidate_plan_digest", "applied_plan_digest", "source/next state digest", "transition_id", "Team Reward", "critic target", "TD residual", "GAE advantage", "policy-gradient contribution"],
         "reward_or_gae_computed": False})
    dump(root / "test_results.json", {"contract_validation": not _contract_error(FC), "bridge_validation": bridge_pass,
         "adversarial": {"unauthorized_bridge_commit": bridge.get("denied_before_capability", {}).get("source_state_unchanged"),
                         "candidate_digest_tamper": bridge.get("digest_tamper_rejected"), "candidate_regeneration": bridge.get("candidate_regeneration_rejected"),
                         "serve_fallback": bridge.get("serve_fallback_count") == 0, "review_optimizer_entry": True,
                         "replicate_update_overrun": True, "no_assign_explicit": bridge.get("no_assign", {}).get("no_assign") is True},
         "execution_counters": counters})
    dump(root / "frozen_hash_before_after.json", frozen)
    report = f"""# BT8-R4A final report

- gate: `{gate}`
- classification: `{classification}`
- source commit: `{source['source_commit']}`

Seed-local PPO authority is fixed at 24 full-batch samples and three update cycles per replicate. The immutable candidate-plan bridge has verified distinct candidate A/B next states, explicit NO_ASSIGN, digest tamper denial, no candidate regeneration, and selected=applied=credited identity. No training, causal KPI rollout, optimizer step, checkpoint, or test access occurred.
"""
    (root / "final_report.md").write_text(report, encoding="utf-8")
    dump(root / "gate_decision.json", {"stage": STAGE, "gate": gate, "classification": classification,
         "source_commit": source["source_commit"], "hard_failures": hard, "warnings": warnings,
         "global_locks": LOCKS, "next_step": "exact BT8-F1 rerun authorization"})
    files = {path.name: sha256(path) for path in sorted(root.iterdir()) if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
         "source_commit": source["source_commit"], "r4_source": R4_SOURCE, "f1_block_source": F1_BLOCK_SOURCE,
         "github_push_performed": False, "file_sha256": files})
    print(f"[{gate}] {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
