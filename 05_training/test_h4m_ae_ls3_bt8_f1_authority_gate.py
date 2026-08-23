"""BT8-F1 must reject missing execution authority before MPS or training."""

from __future__ import annotations

import sys

sys.path.insert(0, "05_training")

import run_h4m_ae_ls3_bt8_f1_bounded_training as F1


def bridge_without_candidate_plan(agent_actions, *, legal_mask, target_ids, provenance):
    return None


def test_missing_seed_update_and_plan_application_authority_blocks() -> None:
    failures = F1.validate_r4_execution_authority(
        r4_gate={"gate": "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_FRESH_V2_ACTOR_TRAINING_AND_NOVEL_EXPOSURE_DESIGN_COMPLETE",
                 "source_commit": F1.R4_SOURCE},
        selected={"distinct_windows": 9, "train_visits": 12, "requests_train": 43,
                  "review_snapshot_visits": 6, "requests_review": 27, "agents": 8,
                  "decisions": 48, "trajectories": 12, "transitions": 96, "optimizer_updates": 6,
                  "seeds": [20260822, 20260823]},
        plan_contract={"disposable_shadow_application_only": True},
        bridge_step=bridge_without_candidate_plan,
    )
    assert {row["code"] for row in failures} == {
        "PER_REPLICATE_UPDATE_BUDGET_MISSING", "PPO_UPDATE_PARTITION_MISSING",
        "CANDIDATE_PLAN_AUTHORITATIVE_APPLICATION_UNBOUND", "CAUSAL_BRIDGE_CANDIDATE_PLAN_INPUT_UNBOUND",
    }
