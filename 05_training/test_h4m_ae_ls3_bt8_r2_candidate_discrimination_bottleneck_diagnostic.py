"""Read-only fixtures for BT8-R2 bottleneck localization."""

import inspect
import json

import multi_agent_candidate_assignment_head as H
import run_h4m_ae_ls3_bt8_r2_candidate_discrimination_bottleneck_diagnostic as r2


def test_candidate_context_is_computed_but_missing_from_scorer_input() -> None:
    source = inspect.getsource(H.MultiAgentCandidateAssignmentHead.forward)
    block = source.split("stacked = torch.cat([", 1)[1].split("], dim=-1)", 1)[0]
    assert "cand_ctx = self.candidate_encoder(candidate_feats)" in source
    assert "cand_ctx" not in block


def test_all_26_r1_ties_encode_only_hop_count_delta() -> None:
    rows = json.loads((r2.R1 / "bt8r1_candidate_distinctness.json").read_text())["all_multi_candidate_records"]
    ties = [row for row in rows if row["exact_tie"]]
    assert len(ties) == 26
    assert all(row["tie_distinctness"]["varying_candidate_features"] == ["hop_count"] for row in ties)


def test_a1_and_bt6_full_actor_inputs_match_as_multisets() -> None:
    import joint_assignment_frozen_policy_snapshot as FPS
    def load(root):
        collection = FPS.load_collection_manifest(root / "collection_manifest.json")
        return [FPS.load_snapshot(root / entry["relative_path"]) for entry in collection["entries"]]
    names = ("global_feats", "demand_feats", "agent_feats", "agent_mask", "candidate_feats", "pair_agent_index", "safe_mask")
    bt6, a1 = load(r2.BT6_SNAPSHOTS), load(r2.A1_SNAPSHOTS)
    assert r2.multiset_comparison([r2.tensor_digest(row, names) for row in bt6], [r2.tensor_digest(row, names) for row in a1])["exact_multiset_match"]
