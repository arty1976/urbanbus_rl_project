"""Policy sampling identity regression tests for R18-R9."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_snapshot as FPS  # noqa: E402
import joint_assignment_frozen_policy_selector as S  # noqa: E402


def _payload(*, source_commit: str = "SOURCE_A",
             agent_order: tuple[str, ...] = ("AGENT_A", "AGENT_B"),
             candidate_order: tuple[tuple[str, str], ...] = (("AGENT_A", "CAND_A"), ("AGENT_B", "CAND_B"))):
    agent_feature_by_id = {
        "AGENT_A": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "AGENT_B": torch.tensor([5.0, 6.0, 7.0, 8.0]),
    }
    candidate_feature_by_id = {
        ("AGENT_A", "CAND_A"): torch.arange(8, dtype=torch.float32),
        ("AGENT_B", "CAND_B"): torch.arange(8, dtype=torch.float32) + 10.0,
    }
    agent_index = {agent_id: index for index, agent_id in enumerate(agent_order)}
    tensors = {
        "global_feats": torch.ones((1, 8), dtype=torch.float32),
        "demand_feats": torch.ones((1, 6), dtype=torch.float32),
        "agent_feats": torch.stack([torch.stack([agent_feature_by_id[item] for item in agent_order])]),
        "agent_mask": torch.ones((1, len(agent_order)), dtype=torch.bool),
        "candidate_feats": torch.stack([torch.stack([candidate_feature_by_id[item] for item in candidate_order])]),
        "pair_agent_index": torch.tensor([[agent_index[agent_id] for agent_id, _ in candidate_order]], dtype=torch.long),
        "safe_mask": torch.ones((1, len(candidate_order)), dtype=torch.bool),
    }
    metadata = {
        "snapshot_schema_version": FPS.SNAPSHOT_SCHEMA_VERSION,
        "decision_id": "D0",
        "window_id": "W0",
        "seed": 20260822,
        "decision_index": 0,
        "time_band": "synthetic",
        "agent_ids": list(agent_order),
        "candidate_ids": [{"agent_id": agent, "candidate_id": candidate} for agent, candidate in candidate_order],
        "candidate_order": [{"agent_id": agent, "candidate_id": candidate} for agent, candidate in candidate_order],
        "selectable_pair_count": len(candidate_order),
        "actor_tensor_pair_count": len(candidate_order),
        "storage_padding_pair_count": 0,
        "no_assign_option": "NO_ASSIGN_KEEP_CURRENT_PLANS",
        "no_assign_index": len(candidate_order),
        "candidate_support_digest": "a" * 64,
        "candidate_support_contract_id": FPS.CANDIDATE_SUPPORT_CONTRACT_ID,
        "feature_contract": {
            "feature_contract_id": FPS.FEATURE_CONTRACT_ID,
            "candidate_support_contract_id": FPS.CANDIDATE_SUPPORT_CONTRACT_ID,
        },
        "feature_contract_id": FPS.FEATURE_CONTRACT_ID,
        "actor_config": {
            "actor_class": "CandidateSensitiveMultiAgentCandidateAssignmentHead",
            "actor_module_sha256": "source-file-hash",
            "actor_head_id": "MULTI_AGENT_JOINT_ASSIGNMENT_HEAD_V2_CANDIDATE_CONTEXT",
            "actor_head_version": "LS3_BT8_R3_V2",
            "global_dim": 8,
            "demand_dim": 6,
            "agent_dim": 4,
            "candidate_dim": 8,
            "hidden": 128,
            "heads": 4,
        },
        "actor_config_sha256": "synthetic",
        "frozen_authority_hashes": {"source": "hash"},
        "source_commit": source_commit,
        "captured_device": "cpu",
        "tensor_descriptors": {name: FPS._tensor_descriptor(value) for name, value in tensors.items()},
    }
    return metadata, tensors


def test_source_commit_changes_evidence_digest_but_not_policy_sampling_identity() -> None:
    meta_a, tensors = _payload(source_commit="A")
    meta_b, _ = _payload(source_commit="B")
    assert FPS._tensor_digest(meta_a, tensors) != FPS._tensor_digest(meta_b, tensors)
    assert FPS.policy_sampling_identity(metadata=meta_a, tensors=tensors) == FPS.policy_sampling_identity(metadata=meta_b, tensors=tensors)


def test_policy_sampling_identity_changes_for_state_and_support_mutations() -> None:
    metadata, tensors = _payload()
    base = FPS.policy_sampling_identity(metadata=metadata, tensors=tensors)
    mutated_tensors = {key: value.clone() for key, value in tensors.items()}
    mutated_tensors["demand_feats"][0, 0] += 1.0
    assert FPS.policy_sampling_identity(metadata=metadata, tensors=mutated_tensors) != base
    mutated_metadata = dict(metadata)
    mutated_metadata["candidate_support_digest"] = "b" * 64
    assert FPS.policy_sampling_identity(metadata=mutated_metadata, tensors=tensors) != base


def test_policy_sampling_identity_and_semantic_sampling_are_order_invariant() -> None:
    variants = [
        _payload(agent_order=("AGENT_A", "AGENT_B"), candidate_order=(("AGENT_A", "CAND_A"), ("AGENT_B", "CAND_B"))),
        _payload(agent_order=("AGENT_A", "AGENT_B"), candidate_order=(("AGENT_B", "CAND_B"), ("AGENT_A", "CAND_A"))),
        _payload(agent_order=("AGENT_B", "AGENT_A"), candidate_order=(("AGENT_A", "CAND_A"), ("AGENT_B", "CAND_B"))),
        _payload(agent_order=("AGENT_B", "AGENT_A"), candidate_order=(("AGENT_B", "CAND_B"), ("AGENT_A", "CAND_A"))),
    ]
    identities = [FPS.policy_sampling_identity(metadata=metadata, tensors=tensors) for metadata, tensors in variants]
    assert len(set(identities)) == 1
    semantic_logits = {("AGENT_A", "CAND_A"): 0.10, ("AGENT_B", "CAND_B"): 0.20}
    selections = []
    for identity, (metadata, _tensors) in zip(identities, variants):
        pair_keys = [(row["agent_id"], row["candidate_id"]) for row in metadata["candidate_ids"]]
        logits = torch.tensor([[semantic_logits[pair] for pair in pair_keys]], dtype=torch.float32)
        view = S.make_frozen_masked_distribution_view(
            pair_keys=pair_keys,
            pair_logits=logits,
            no_assign_logit=torch.tensor([[0.30]], dtype=torch.float32),
            safe_mask=torch.ones((1, len(pair_keys)), dtype=torch.bool),
        )
        selected = S.select_frozen_policy_action(
            distribution_view=view,
            mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
            policy_sampling_identity=identity,
            probe_seed=0,
        )
        selections.append(selected.semantic_identity)
    assert len(set(selections)) == 1
