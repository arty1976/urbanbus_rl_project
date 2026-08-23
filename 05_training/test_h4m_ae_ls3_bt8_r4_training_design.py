"""BT8-R4 pure contract fixtures; no optimizer, rollout, or reward execution."""

from __future__ import annotations

from dataclasses import replace

import pytest

import joint_candidate_plan_execution as PE
import joint_candidate_support_snapshot as SS
import run_h4m_ae_ls3_bt8_r4_training_design as R4


def evidence(candidate_id: str, path: list[str]) -> SS.CandidateEvidence:
    local = {"candidate_id": candidate_id, "path_stop_ids": path,
             "generalized_cost": float(len(path))}
    return SS.CandidateEvidence(
        agent_id="AGENT_A", candidate_id=candidate_id, source_candidate_id=candidate_id,
        source_candidate_digest=SS.canonical_sha256(local),
        source_generalized_cost=float(len(path)),
        local_search_features={"distance_m": 1.0, "time_sec": 1.0,
                               "generalized_cost": float(len(path)), "hop_count": float(len(path) - 1)},
        zero_loss_status=SS.PASS, zero_loss_evidence_digest="zero-loss", source_state_digest="source-state",
        payload={"local_search": local, "plan_stop_ids": ["S0", "S1", "S2", "S3"],
                 "pickup_position": 1, "dropoff_position": 3,
                 "onboard_passenger_count": 0, "max_delta_eta_sec": 0.0})


def snapshot() -> SS.JointCandidateSupportSnapshot:
    first, second = evidence("C1", ["S1", "X", "S3"]), evidence("C2", ["S1", "Y", "Z", "S3"])
    return SS.freeze_joint_support(decision_group_id="D", no_assign_option=SS.NO_ASSIGN,
        records=[first, second], raw_candidate_ids_by_agent={"AGENT_A": ["C1", "C2"]},
        source_state_digests={"AGENT_A": "source-state"})


def test_distinct_safe_candidates_apply_to_distinct_disposable_plan_states() -> None:
    frozen = snapshot()
    one = PE.apply_on_disposable_route(PE.binding_for_selected_pair(frozen, agent_id="AGENT_A", candidate_id="C1"))
    two = PE.apply_on_disposable_route(PE.binding_for_selected_pair(frozen, agent_id="AGENT_A", candidate_id="C2"))
    assert one.candidate_plan_digest != two.candidate_plan_digest
    assert one.applied_plan_digest != two.applied_plan_digest
    assert one.applied_state_digest != two.applied_state_digest
    assert one.applied_plan_stop_ids == ("S0", "S1", "X", "S3")
    assert two.applied_plan_stop_ids == ("S0", "S1", "Y", "Z", "S3")


def test_credit_row_preserves_selected_applied_attributed_candidate_identity() -> None:
    frozen = snapshot()
    applied = PE.apply_on_disposable_route(PE.binding_for_selected_pair(frozen, agent_id="AGENT_A", candidate_id="C1"))
    row = PE.candidate_plan_credit_row(decision_id="D0", agent_id="AGENT_A", candidate_id="C1",
        applied=applied, transition_id="T0", trajectory_id="TR0")
    PE.validate_candidate_plan_credit_row(row)
    assert row["candidate_id"] == applied.candidate_id
    assert row["candidate_plan_digest"] != row["applied_plan_digest"]


@pytest.mark.parametrize("mutation", ["candidate", "digest", "endpoint"])
def test_candidate_plan_identity_mismatches_fail_closed(mutation: str) -> None:
    frozen = snapshot()
    if mutation == "candidate":
        with pytest.raises(PE.CandidatePlanExecutionError) as caught:
            PE.binding_for_selected_pair(frozen, agent_id="AGENT_A", candidate_id="UNKNOWN")
        assert caught.value.code == "REJECTED_OR_UNKNOWN_CANDIDATE_FORCED_SELECTION"
    elif mutation == "digest":
        applied = PE.apply_on_disposable_route(PE.binding_for_selected_pair(frozen, agent_id="AGENT_A", candidate_id="C1"))
        row = PE.candidate_plan_credit_row(decision_id="D0", agent_id="AGENT_A", candidate_id="C1",
            applied=applied, transition_id="T0", trajectory_id="TR0")
        row["applied_plan_digest"] = row["candidate_plan_digest"]
        with pytest.raises(PE.CandidatePlanExecutionError) as caught:
            PE.validate_candidate_plan_credit_row(row)
        assert caught.value.code == "CANDIDATE_PLAN_EXECUTION_COLLAPSE"
    else:
        bad = replace(evidence("C1", ["BAD", "X", "S3"]), candidate_id="C1")
        with pytest.raises(PE.CandidatePlanExecutionError) as caught:
            PE.CandidatePlanBinding.from_evidence(bad)
        assert caught.value.code == "CANDIDATE_PLAN_ENDPOINT_ROUTE_MISMATCH"


def test_fresh_lineage_and_split_contracts_reject_reuse_and_review_exposure() -> None:
    R4.validate_initialization_authority(R4.ACTOR_INITIALIZATION)
    R4.validate_critic_lineage(R4.CRITIC_LINEAGE)
    with pytest.raises(R4.DesignContractError) as caught:
        R4.validate_initialization_authority({**R4.ACTOR_INITIALIZATION, "transfer_allowed": True})
    assert caught.value.code == "UNAUTHORIZED_PARTIAL_ACTOR_TRANSFER"
    with pytest.raises(R4.DesignContractError) as caught:
        R4.validate_critic_lineage({**R4.CRITIC_LINEAGE, "selected_policy": "C2_WARM_START_CRITIC"})
    assert caught.value.code == "UNAUTHORIZED_CRITIC_REUSE"
    with pytest.raises(R4.DesignContractError) as caught:
        R4.validate_train_review_split([{"window_id": "TRAIN"}], [{"window_id": "REVIEW"}], optimizer_steps=1)
    assert caught.value.code == "REVIEW_WINDOW_OPTIMIZER_EXPOSURE"


def test_novelty_and_non_outcome_selection_are_fail_closed() -> None:
    profile = {
        "candidate_feature_signature": "candidate", "full_v2_actor_input_signature": "actor",
        "opportunity_signature": "opportunity", "comparison_signature": "comparison",
        "legal_support_signature": "legal", "zero_loss_support_pattern_signature": "zero",
        "meaningfully_distinct_comparison_count": 1,
    }
    prior = {name: {profile[key]} for name, key in {
        "candidate_feature": "candidate_feature_signature", "full_actor_input": "full_v2_actor_input_signature",
        "opportunity": "opportunity_signature", "comparison": "comparison_signature",
        "legal_support": "legal_support_signature", "zero_loss_pattern": "zero_loss_support_pattern_signature",
    }.items()}
    classification, flags = R4.classify_exposure(profile, prior)
    assert classification == "REDUNDANT"
    assert not any(flags.values())
    with pytest.raises(R4.DesignContractError) as caught:
        R4.validate_selection_inputs(["window_id", "Reward"])
    assert caught.value.code == "OUTCOME_OR_REWARD_BASED_WINDOW_SELECTION"
