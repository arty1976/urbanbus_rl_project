"""
Self-test for A-family policy interface bridge v1.

Run:
    python ./05_training/rollouts/test_a_family_policy_interface_bridge_v1.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from a_family_policy_interface_bridge_v1 import (
    apply_policy_action_to_rollout_row,
    build_a_family_row_with_policy_interface,
    derive_interface_observation_fields,
    sample_base_row,
    write_contract_checkpoint,
)


def make_checkpoint() -> Path:
    out = Path("artifacts/step36_policy_interface_bridge_test/sample_mappo_checkpoint_metadata.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_contract_checkpoint(out)


def test_derive_interface_observation_fields() -> None:
    row = sample_base_row()
    out = derive_interface_observation_fields(row)

    assert "avg_wait_seconds" in out
    assert "bunching_rate" in out
    assert "on_time_rate" in out
    assert "energy_proxy_per_passenger" in out
    assert out["qwen_trigger_rate"] == 0.0


def test_build_a80_row_with_policy_interface() -> None:
    root = Path("artifacts/step36_policy_interface_bridge_test")

    if root.exists():
        shutil.rmtree(root)

    ckpt = make_checkpoint()

    row = build_a_family_row_with_policy_interface(
        base_row=sample_base_row(),
        condition_id="A80",
        checkpoint_path=ckpt,
        require_existing_checkpoint=True,
    )

    assert row["condition_id"] == "A80"
    assert row["policy_source"] == "mappo_policy"
    assert row["source_mode"] == "causal_A80_mappo_policy_v1"
    assert row["qwen_trigger_rate"] == 0.0
    assert row["active_bus_count"] == 8.0
    assert row["action_version"] == "bus_control_action_v1"
    assert row["dispatch_delta"] == 0.0
    assert row["hold_seconds"] == 0.0
    assert row["skip_stop_flag"] is False
    assert row["target_headway_ratio"] == 1.0
    assert row["performance_claim_allowed"] is False
    assert "policy_action_debug" in row


def test_all_a_family_conditions() -> None:
    ckpt = make_checkpoint()

    expected_active_bus_count = {
        "A": 10.0,
        "A90": 9.0,
        "A80": 8.0,
        "A70": 7.0,
    }

    for condition_id, expected_active in expected_active_bus_count.items():
        row = build_a_family_row_with_policy_interface(
            base_row=sample_base_row(),
            condition_id=condition_id,
            checkpoint_path=ckpt,
            require_existing_checkpoint=True,
        )

        assert row["source_mode"] == f"causal_{condition_id}_mappo_policy_v1"
        assert row["active_bus_count"] == expected_active
        assert row["policy_source"] == "mappo_policy"
        assert row["qwen_trigger_rate"] == 0.0


def test_placeholder_action_rejected() -> None:
    row = {
        "condition_id": "A",
        "policy_source": "mappo_policy",
        "source_mode": "causal_A_mappo_policy_v1",
        "qwen_trigger_rate": 0.0,
        "active_bus_count": 10.0,
    }

    bad_action = {
        "action_version": "bus_control_action_v1",
        "condition_id": "A",
        "policy_source": "placeholder_policy",
        "source_mode": "stub_A_placeholder_smoke",
        "qwen_trigger_rate": 0.0,
        "dispatch_delta": 0.0,
        "hold_seconds": 0.0,
        "skip_stop_flag": False,
        "target_headway_ratio": 1.0,
        "active_bus_count": 10.0,
        "policy_debug": {},
    }

    try:
        apply_policy_action_to_rollout_row(row, bad_action)
    except ValueError:
        return

    raise AssertionError("placeholder action was not rejected")


def main() -> None:
    tests = [
        test_derive_interface_observation_fields,
        test_build_a80_row_with_policy_interface,
        test_all_a_family_conditions,
        test_placeholder_action_rejected,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] a_family_policy_interface_bridge_v1 self-test passed")


if __name__ == "__main__":
    main()
