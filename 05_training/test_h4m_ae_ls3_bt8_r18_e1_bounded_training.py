"""Static guards for the R18 executor; these tests never execute training."""

from __future__ import annotations

import copy
import hashlib
import inspect
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18  # noqa: E402
import joint_assignment_credit_contract as CC  # noqa: E402


def _support_roundtrip_row() -> dict[str, object]:
    pairs = [("AGENT_007", "CANDIDATE_A"), ("AGENT_011", "CANDIDATE_B")]
    snapshot_support_digest = "a" * 64
    transition = CC.AssignmentTransition(
        assignment_step_id="R18_R2:SUPPORT:0", decision_group_id="R18_R2:SUPPORT:0",
        episode_id="R18_R2", window_id="WINDOW_0", decision_ts=0, next_assignment_ts=None,
        delta_operational_steps=1, pre_state_digest="pre", next_state_digest="next",
        safe_pair_ids=pairs, safe_pair_mask=[True, True], no_assign_index=2,
        selected_agent_id="AGENT_007", selected_candidate_id="CANDIDATE_A",
        selected_is_no_assign=False, valid_action_count=3, forced_action=False,
        old_log_prob=-0.5, old_value=0.0, team_reward_sequence=[0.0],
        assignment_discounted_reward=0.0, terminated=True, truncated=False,
        policy_version="R18_R2_TEST", credit_contract_version=CC.CONTRACT_VERSION, seed=20260822,
        provenance={"candidate_support_digest": snapshot_support_digest},
    )
    loaded = {"metadata": {
        "candidate_ids": [{"agent_id": agent, "candidate_id": candidate} for agent, candidate in pairs],
        "candidate_support_digest": snapshot_support_digest,
        "selectable_pair_count": 2,
        "no_assign_index": 2,
    }}
    return {"t": transition, "loaded": loaded}


def test_r18_rollout_uses_sealed_r16_selector_and_not_historical_argmax() -> None:
    source = inspect.getsource(R18._rollout_arm)
    assert "make_frozen_masked_distribution_view" in source
    assert "select_frozen_policy_action" in source
    assert "FROZEN_MASKED_CATEGORICAL_TRAINING" in source
    assert "H.select" not in source
    assert "torch.multinomial" not in source
    assert "masked_log_probs" not in source


def test_r18_train_path_asserts_support_before_every_epoch_and_requires_e1_rows() -> None:
    source = inspect.getsource(R18._train_arm)
    assert "behavior_equals_update_support" in source
    assert "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE" in source
    assert "actor_eligible" in source
    assert "ineligible_actor_gradient" in source


def test_r18_r2_support_roundtrip_accepts_distinct_bound_digest_domains() -> None:
    row = _support_roundtrip_row()
    transition, metadata = row["t"], row["loaded"]["metadata"]
    assert metadata["candidate_support_digest"] != transition.action_support_digest
    result = R18._assert_support_roundtrip(rows=[row], CC=CC)
    assert result == {
        "checked": 1, "mismatched": 0, "behavior_equals_update_support": True,
        "snapshot_support_digest_domain_checked": 1,
        "action_support_digest_domain_checked": 1,
    }


@pytest.mark.parametrize("mutation", ["pair_identity", "snapshot_digest", "pair_count", "no_assign_index"])
def test_r18_r2_support_roundtrip_fails_closed_on_dynamic_mutation(mutation: str) -> None:
    row = copy.deepcopy(_support_roundtrip_row())
    metadata = row["loaded"]["metadata"]
    if mutation == "pair_identity":
        metadata["candidate_ids"][0]["candidate_id"] = "MUTATED_CANDIDATE"
    elif mutation == "snapshot_digest":
        metadata["candidate_support_digest"] = "b" * 64
    elif mutation == "pair_count":
        metadata["selectable_pair_count"] = 1
    else:
        metadata["no_assign_index"] = 1
    with pytest.raises(R18.R18Error, match="ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE"):
        R18._assert_support_roundtrip(rows=[row], CC=CC)


def test_r18_r6_keeps_rollout_and_frozen_inference_hashes_and_freezes_instrumented_train_hash() -> None:
    expected = {
        "_rollout_arm": "9011b5903a0a1e6be56777f40f1a3e2ba4e1e2094af7b2c2fa3f21f475f756fc",
        "_train_arm": "3f1e3178288b80b73319fd7708877ec6fff6e6e5f320aff796e72f83c7a0ad6e",
        "_frozen_review_replay": "ddc010943f5bfb024ed2a33309e2719ede6ac8dc3f38442a7b7ff98847fd622c",
    }
    actual = {name: hashlib.sha256(inspect.getsource(getattr(R18, name)).encode()).hexdigest()
              for name in expected}
    assert actual == expected
    train_source = inspect.getsource(R18._train_arm)
    assert "TRACE.build_epoch_trace_rows" in train_source
    assert "TRACE.actor_batch_forward_no_grad" in train_source


def test_r18_captures_same_support_t1_review_deltas_without_optimizer_exposure() -> None:
    source = inspect.getsource(R18._frozen_review_replay) + inspect.getsource(R18._learning_counters) + inspect.getsource(R18.execute)
    assert "TIE_BREAK_CONTRACT_ID" in source
    assert "interpretation_performed" in source
    assert "logit_probability_shift_after_authorized_update" in source
    assert "duplicate_reward_ancestry" in source


def test_exact_execute_flag_is_required_by_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        R18._parse_args(["--authorization-manifest", str(tmp_path / "a.json"), "--authorization-sha256", "a"])
    parsed = R18._parse_args(["--authorization-manifest", str(tmp_path / "a.json"), "--authorization-sha256", "a", "--dry-run"])
    assert parsed.dry_run is True
    assert parsed.execute_exact_r17_envelope is False


def test_dry_run_is_zero_execution() -> None:
    report = R18.dry_run_report({"source_commit": "source"})
    assert report["training"] == report["rollout"] == report["optimizer_step"] == report["checkpoint_write"] == 0
