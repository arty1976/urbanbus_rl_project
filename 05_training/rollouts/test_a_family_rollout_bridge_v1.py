"""
Self-test for a_family_rollout_bridge_v1.py.

Run:
    python .\05_training\rollouts\test_a_family_rollout_bridge_v1.py
"""

from a_family_rollout_bridge_v1 import (
    build_a_family_rollout_row,
    sample_base_row,
)


def test_placeholder_row():
    row = build_a_family_rollout_row(
        sample_base_row(),
        condition_id="A",
        policy_kind="placeholder",
        checkpoint_path=None,
        strict_claim_validation=False,
    )

    assert row["condition_id"] == "A"
    assert row["policy_source"] == "placeholder_policy"
    assert row["source_mode"] == "stub_A_placeholder_smoke"
    assert row["qwen_trigger_rate"] == 0.0


def test_actual_mappo_row_a80():
    row = build_a_family_rollout_row(
        sample_base_row(),
        condition_id="A80",
        policy_kind="mappo",
        checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
        require_existing_checkpoint=False,
        strict_claim_validation=True,
    )

    assert row["condition_id"] == "A80"
    assert row["policy_source"] == "mappo_policy"
    assert row["source_mode"] == "causal_A80_mappo_policy_v1"
    assert abs(row["fleet_reduction_ratio"] - 0.2) < 1e-9
    assert row["qwen_trigger_rate"] == 0.0


def test_actual_mappo_missing_checkpoint_rejected():
    try:
        build_a_family_rollout_row(
            sample_base_row(),
            condition_id="A",
            policy_kind="mappo",
            checkpoint_path="",
            require_existing_checkpoint=False,
            strict_claim_validation=False,
        )
    except FileNotFoundError:
        return

    raise AssertionError("missing MAPPO checkpoint_path was not rejected")


def test_fleet_ratios():
    expected = {
        "A": 0.0,
        "A90": 0.1,
        "A80": 0.2,
        "A70": 0.3,
    }

    for condition_id, ratio in expected.items():
        row = build_a_family_rollout_row(
            sample_base_row(),
            condition_id=condition_id,
            policy_kind="placeholder",
            checkpoint_path=None,
            strict_claim_validation=False,
        )
        assert abs(row["fleet_reduction_ratio"] - ratio) < 1e-9


def test_reward_fields_exist():
    row = build_a_family_rollout_row(
        sample_base_row(),
        condition_id="A",
        policy_kind="placeholder",
        checkpoint_path=None,
        strict_claim_validation=False,
    )

    for key in [
        "reward_total",
        "reward_service",
        "reward_avg_wait",
        "reward_long_wait",
        "reward_on_time",
        "reward_bunching",
        "reward_energy",
        "reward_fleet",
        "reward_constraint",
    ]:
        assert key in row


def main():
    tests = [
        test_placeholder_row,
        test_actual_mappo_row_a80,
        test_actual_mappo_missing_checkpoint_rejected,
        test_fleet_ratios,
        test_reward_fields_exist,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] a_family_rollout_bridge_v1 self-test passed")


if __name__ == "__main__":
    main()
