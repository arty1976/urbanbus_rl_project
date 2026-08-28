"""Focused pure tests for the BT8-R6 attribution audit."""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "05_training")

import run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution as R6


def test_ppo_clip_state_matches_canonical_signed_ppo_rule() -> None:
    assert R6.ppo_clip_state(1.0, 1.21) == "CLIPPED_HIGH_ZERO_GRADIENT"
    assert R6.ppo_clip_state(1.0, 0.70) == "ACTIVE"
    assert R6.ppo_clip_state(-1.0, 0.79) == "CLIPPED_LOW_ZERO_GRADIENT"
    assert R6.ppo_clip_state(-1.0, 1.30) == "ACTIVE"
    assert R6.ppo_clip_state(0.0, 1.0) == "ZERO_ADVANTAGE"


def test_pressure_labels_keep_action_type_and_sign_explicit() -> None:
    assert R6.pressure_label(selected_no_assign=False, advantage=-1.0, clip_state="ACTIVE") \
        == "SUPPRESS_SELECTED_CANDIDATE_DIRECT_RELATIVE_NO_ASSIGN_UP"
    assert R6.pressure_label(selected_no_assign=True, advantage=-1.0, clip_state="ACTIVE") \
        == "SUPPRESS_NO_ASSIGN_DIRECT_RELATIVE_CANDIDATE_UP"
    assert R6.pressure_label(selected_no_assign=True, advantage=1.0, clip_state="ACTIVE") \
        == "REINFORCE_NO_ASSIGN_DIRECT"
    assert R6.pressure_label(selected_no_assign=False, advantage=1.0,
                             clip_state="CLIPPED_HIGH_ZERO_GRADIENT") == "ZERO_POLICY_PRESSURE"


def test_exact_cross_replay_matrix_identifies_initialization_driver() -> None:
    matrix = {
        "R1_INITIAL_ACTOR": {"R1_INPUT_NO_ASSIGN": 0, "R2_INPUT_NO_ASSIGN": 0},
        "R2_INITIAL_ACTOR": {"R1_INPUT_NO_ASSIGN": 24, "R2_INPUT_NO_ASSIGN": 24},
    }
    assert R6.initial_action_support_driver(matrix) \
        == "ACTOR_INITIALIZATION_DOMINANT_ACROSS_BOTH_PRESERVED_INPUT_COLLECTIONS"


def test_root_classification_requires_both_seed_specific_chains() -> None:
    pattern = {
        "initial_driver": "ACTOR_INITIALIZATION_DOMINANT_ACROSS_BOTH_PRESERVED_INPUT_COLLECTIONS",
        "input_tensor_multisets_exact": True,
        "r1_candidate_selected": 24, "r1_reward_rows": 4, "r1_raw_positive": 24,
        "r1_positive_to_negative": 15, "r2_no_assign_selected": 24, "r2_reward_rows": 0,
        "r2_raw_negative": 24, "r2_negative_to_positive": 9, "r2_direct_candidate_credit": 0,
    }
    assert R6.classify_root(pattern) == R6.ROOT_CLASS
    assert R6.classify_root({**pattern, "r2_direct_candidate_credit": 1}) == R6.INDETERMINATE_CLASS


def test_sign_transition_and_empty_evidence_are_exact() -> None:
    assert R6.sign_transition(0.5, -0.1) == "POSITIVE_TO_NEGATIVE"
    assert R6.sign_transition(-0.5, 0.1) == "NEGATIVE_TO_POSITIVE"
    assert R6.describe([])["count"] == 0
    assert R6.describe([])["mean"] is None


def test_decision_id_binds_replicate_without_index_drift() -> None:
    assert R6.replicate_from_decision_id("BT8_F1:F1_R1:0:0") == "F1_R1"


def test_contract_no_assign_is_detected_as_an_explicit_action() -> None:
    row = {"selected_candidate_id": "NO_ASSIGN_KEEP_CURRENT_PLANS"}
    assert R6.selected_no_assign(row, "NO_ASSIGN_KEEP_CURRENT_PLANS") is True
    assert R6.selected_no_assign({"selected_candidate_id": None}, "NO_ASSIGN_KEEP_CURRENT_PLANS") is False


def test_gae_reward_and_critic_components_are_additive() -> None:
    rows = [
        {"decision_id": "BT8_F1:F1_R1:0:0", "trajectory_id": "t", "nonterminal": True,
         "reward": 1.0, "critic_value": 0.0, "next_value": 0.0, "raw_gae": 1.0,
         "normalized_advantage": 1.0},
        {"decision_id": "BT8_F1:F1_R1:0:1", "trajectory_id": "t", "nonterminal": False,
         "reward": 0.0, "critic_value": 0.0, "next_value": 0.0, "raw_gae": 0.0,
         "normalized_advantage": -1.0},
    ]
    components, audit = R6.decompose_gae(rows)
    assert [row["reward_gae_component"] for row in components] == [1.0, 0.0]
    assert [row["critic_bootstrap_gae_component"] for row in components] == [0.0, 0.0]
    assert audit["max_abs_raw_additivity_residual"] == 0.0
    assert audit["max_abs_normalized_additivity_residual"] == 0.0


def test_actor_route_partition_and_shapley_are_complete() -> None:
    assert R6.actor_parameter_group("global_encoder.0.weight") == "SHARED"
    assert R6.actor_parameter_group("candidate_encoder.0.weight") == "PAIR_ONLY"
    assert R6.actor_parameter_group("no_assign_scorer.0.weight") == "NO_ASSIGN_ONLY"
    values = {}
    for mask in range(8):
        subset = frozenset(group for index, group in enumerate(R6.PARAMETER_GROUPS) if mask & (1 << index))
        values[subset] = float(sum(index + 1 for index, group in enumerate(R6.PARAMETER_GROUPS) if group in subset))
    result = R6.shapley_from_coalitions(values)
    assert result == {"SHARED": 1.0, "PAIR_ONLY": 2.0, "NO_ASSIGN_ONLY": 3.0}


def test_source_scope_is_closed_to_runner_and_tests() -> None:
    assert R6.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution.py",
        "05_training/test_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution.py",
    }
    assert Path("05_training/run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution.py").is_file()


def test_pinned_authoritative_hashes_match_exactly() -> None:
    pairs = {
        R6.F1 / "manifest.json": R6.F1_MANIFEST_SHA256,
        R6.R5 / "manifest.json": R6.R5_MANIFEST_SHA256,
        R6.F1 / "bt8f1_candidate_plan_credit_audit.json": R6.F1_CREDIT_SHA256,
        R6.F1 / "bt8f1_training_execution_audit.json": R6.F1_EXECUTION_SHA256,
        R6.F1 / "bt8f1_learning_signal_audit.json": R6.F1_LEARNING_SHA256,
    }
    assert {path: R6.sha256(path) for path in pairs} == pairs


def test_authoritative_training_tensor_multisets_are_seed_identical() -> None:
    root = R6.F1 / "bt8f1_training_snapshots"
    collection = json.loads((root / "collection_manifest.json").read_text())
    by_seed: dict[int, list[str]] = {20260822: [], 20260823: []}
    for entry in collection["entries"]:
        manifest = json.loads((root / entry["relative_path"] / "snapshot_manifest.json").read_text())
        by_seed[int(manifest["metadata"]["seed"])].append(manifest["tensor_binary_sha256"])
    assert len(by_seed[20260822]) == len(by_seed[20260823]) == 24
    assert Counter(by_seed[20260822]) == Counter(by_seed[20260823])
    assert len(set(by_seed[20260822] + by_seed[20260823])) == 6


def test_authoritative_gae_has_rewardless_r2_critic_component() -> None:
    execution = json.loads((R6.F1 / "bt8f1_training_execution_audit.json").read_text())
    audits = {}
    for replicate_id in R6.REPLICATES:
        rows = execution["replicate_rollouts"][replicate_id]["gae_rows"]
        _, audits[replicate_id] = R6.decompose_gae(rows)
    assert audits["F1_R1"]["reward_component_abs_sum"] > 0.0
    assert audits["F1_R1"]["critic_bootstrap_component_abs_sum"] > 0.0
    assert audits["F1_R2"]["reward_component_abs_sum"] == 0.0
    assert audits["F1_R2"]["critic_bootstrap_component_abs_sum"] > 0.0


def test_r2_full_action_margin_switch_is_not_uniform_pair_compression() -> None:
    initial = json.loads((R6.F1 / "bt8f1_initial_review_replay.json").read_text())["rows"]
    final = json.loads((R6.F1 / "bt8f1_final_review_replay.json").read_text())["rows"]
    initial_by = {row["snapshot_digest"]: row for row in initial if row["replicate_id"] == "F1_R2"}
    pair_deltas = []
    for row in final:
        if row["replicate_id"] != "F1_R2" or len(row["pair_logits"]) < 2:
            continue
        first = initial_by[row["snapshot_digest"]]
        initial_pair = sorted(first["pair_logits"], reverse=True)
        final_pair = sorted(row["pair_logits"], reverse=True)
        pair_deltas.append((final_pair[0] - final_pair[1]) - (initial_pair[0] - initial_pair[1]))
        assert first["selected_identity"] == "NO_ASSIGN"
        assert row["selected_identity"] != "NO_ASSIGN"
    assert len(pair_deltas) == 2
    assert any(value > 0.0 for value in pair_deltas)
    assert any(value < 0.0 for value in pair_deltas)


def test_runner_has_no_training_or_rollout_mutation_entrypoint() -> None:
    source = inspect.getsource(R6)
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "apply_assignment_update(" not in source
    assert "adapter.step(" not in source
    assert "factory.build(" not in source
    assert "require_capability(" not in source
