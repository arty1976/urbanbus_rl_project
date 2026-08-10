from __future__ import annotations

import pandas as pd

from run_prompt5_e01_r1_validation_contract_repair import pair_join_audit, row_identity_audit, selection_contract


def test_selection_contract_is_hard_constraint_first_and_no_test() -> None:
    contract = selection_contract()
    assert contract["hard_constraint_first"][0] == "capacity_violation_zero"
    assert contract["lexicographic_order"][:3] == [
        "passenger_service_rate desc",
        "passenger_wait_p95_seconds asc",
        "avg_wait_seconds asc",
    ]
    assert contract["test_access_allowed"] is False
    assert contract["approved_for_e2_execution"] is False


def test_row_identity_audit_blocks_missing_and_duplicate_keys() -> None:
    rows = [
        {"split": "validation", "snapshot_id": 1, "state_ts": "t", "condition_id": "A", "seed": 1, "window_id": "w", "timestep": 0, "agent_id": 1},
        {"split": "validation", "snapshot_id": 1, "state_ts": "t", "condition_id": "A", "seed": 1, "window_id": "w", "timestep": 0, "agent_id": 1},
        {"split": "validation", "snapshot_id": 2},
    ]
    audit = row_identity_audit(rows, ["split", "snapshot_id", "state_ts", "condition_id", "seed", "window_id", "timestep", "agent_id"])
    assert audit["duplicate_key_count"] == 1
    assert audit["missing_key_count"] == 1
    assert audit["passed"] is False

def test_pair_join_audit_requires_matching_explicit_keys() -> None:
    e0 = [{"split": "validation", "snapshot_id": 1, "state_ts": "t", "condition_id": "A", "seed": 1, "window_id": "w", "timestep": 0, "agent_id": 1}]
    e1 = [{"split": "validation", "snapshot_id": 1, "state_ts": "t", "condition_id": "A", "seed": 1, "window_id": "w", "timestep": 0, "agent_id": 1}]
    assert pair_join_audit(e0, e1)["passed"] is True
    e1[0]["agent_id"] = 2
    audit = pair_join_audit(e0, e1)
    assert audit["E0_only_key_count"] == 1
    assert audit["E1_only_key_count"] == 1
    assert audit["passed"] is False
