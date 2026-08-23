"""Focused BT8-R4A contracts; no model, optimizer, rollout, or checkpoint."""

from __future__ import annotations

from dataclasses import replace

import pytest

import joint_assignment_f1_execution_contract as FC
import joint_candidate_plan_causal_bridge as CB
import joint_candidate_support_snapshot as SS
import simulator_authorization as AUTH


def evidence(candidate_id: str, path: list[str]) -> SS.CandidateEvidence:
    local = {"candidate_id": candidate_id, "path_stop_ids": path, "generalized_cost": float(len(path))}
    return SS.CandidateEvidence(
        agent_id="AGENT_A", candidate_id=candidate_id, source_candidate_id=candidate_id,
        source_candidate_digest=SS.canonical_sha256(local), source_generalized_cost=float(len(path)),
        local_search_features={"distance_m": 1.0, "time_sec": 1.0, "generalized_cost": float(len(path)), "hop_count": float(len(path) - 1)},
        zero_loss_status=SS.PASS, zero_loss_evidence_digest=f"ZL-{candidate_id}", source_state_digest="AUTHORITATIVE_SOURCE",
        payload={"local_search": local, "plan_stop_ids": ["S0", "S1", "S2", "S3"], "pickup_position": 1,
                 "dropoff_position": 3, "onboard_passenger_count": 0, "max_delta_eta_sec": 0.0})


def fixture() -> tuple[SS.JointCandidateSupportSnapshot, CB.ImmutableCandidatePlan, CB.ImmutableCandidatePlan, CB.CandidatePlanState]:
    rows = [evidence("C1", ["S1", "A", "S3"]), evidence("C2", ["S1", "B", "C", "S3"])]
    snapshot = SS.freeze_joint_support(decision_group_id="R4A_D", no_assign_option=SS.NO_ASSIGN, records=rows,
        raw_candidate_ids_by_agent={"AGENT_A": ["C1", "C2"]}, source_state_digests={"AGENT_A": "AUTHORITATIVE_SOURCE"})
    one = CB.ImmutableCandidatePlan.from_snapshot(decision_id="D1", snapshot=snapshot, agent_id="AGENT_A", candidate_id="C1", source_state_version=0)
    two = CB.ImmutableCandidatePlan.from_snapshot(decision_id="D2", snapshot=snapshot, agent_id="AGENT_A", candidate_id="C2", source_state_version=0)
    return snapshot, one, two, CB.CandidatePlanState.from_binding(one.candidate_plan)


def commit(bridge: CB.CandidatePlanAuthoritativeBridge, candidate: CB.ImmutableCandidatePlan) -> CB.PurePlanTransition:
    state = bridge.state
    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, reason="R4A atomic bridge fixture"):
        return bridge.commit_candidate(candidate=candidate, expected_source_version=state.version,
            expected_source_digest=state.state_digest, expected_support_digest=candidate.candidate_support_digest,
            expected_candidate_plan_digest=candidate.candidate_plan_digest)


def test_seed_ppo_contract_is_exact_and_replicate_local() -> None:
    FC.validate_contract()
    first, second = FC.ReplicateBudget("F1_R1"), FC.ReplicateBudget("F1_R2")
    first.add_train_samples(24); second.add_train_samples(24)
    for _ in range(3): first.update(); second.update()
    first.finalize(); second.finalize()
    with pytest.raises(FC.F1ExecutionContractError) as caught:
        first.update()
    assert caught.value.code == "F1_REPLICATE_UPDATE_BUDGET_EXCEEDED"
    with pytest.raises(FC.F1ExecutionContractError) as caught:
        FC.ReplicateBudget("review").add_review_sample()
    assert caught.value.code == "REVIEW_ROW_ENTERED_OPTIMIZER_PATH"
    normalized = FC.normalize_advantages_for_replicate(range(24))
    assert len(normalized) == 24 and abs(sum(normalized)) < 1e-10


def test_candidate_a_b_distinct_atomic_transitions_and_identity() -> None:
    snapshot, one, two, state = fixture()
    source_digest = state.state_digest
    with pytest.raises(AUTH.AuthorizationDenied):
        CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot).commit_candidate(
            candidate=one, expected_source_version=0, expected_source_digest=state.state_digest,
            expected_support_digest=snapshot.snapshot_digest, expected_candidate_plan_digest=one.candidate_plan_digest)
    assert state.state_digest == source_digest
    a = commit(CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot), one)
    b = commit(CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot), two)
    assert a.next_state.state_digest != b.next_state.state_digest
    assert a.applied_plan_digest != b.applied_plan_digest
    assert a.selected_candidate_id == a.applied_candidate_id == a.credited_candidate_id == "C1"
    assert b.selected_candidate_id == b.applied_candidate_id == b.credited_candidate_id == "C2"
    assert state.state_digest == source_digest


def test_tamper_version_regeneration_double_commit_and_no_assign_fail_closed() -> None:
    snapshot, one, _, state = fixture()
    bridge = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot)
    with pytest.raises(CB.CandidatePlanBridgeError) as caught:
        commit(bridge, replace(one, candidate_plan_digest="tampered"))
    assert caught.value.code == "CANDIDATE_PLAN_DIGEST_MISMATCH"
    result = commit(bridge, one)
    with pytest.raises(CB.CandidatePlanBridgeError) as caught:
        commit(bridge, one)
    assert caught.value.code == "DOUBLE_AUTHORITATIVE_COMMIT"
    with pytest.raises(SS.CandidateSupportError) as caught:
        snapshot.replay_guard(support_digest=snapshot.snapshot_digest, regeneration_requested=True)
    assert caught.value.code == "CANDIDATE_REGENERATION_FORBIDDEN_AT_REPLAY"
    no_assign_bridge = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot)
    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, reason="R4A explicit no assign fixture"):
        no_assign = no_assign_bridge.commit_no_assign(decision_id="NO_ASSIGN_D", expected_source_version=0,
            expected_source_digest=state.state_digest, expected_support_digest=snapshot.snapshot_digest)
    assert no_assign.no_assign and not no_assign.serve_fallback_used
    assert no_assign.next_state.route_plan_stop_ids == state.route_plan_stop_ids
    assert result.serve_fallback_used is False
