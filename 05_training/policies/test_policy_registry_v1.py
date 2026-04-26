"""
Self-test for policy_registry_v1.py.

Run:
    python .\05_training\policies\test_policy_registry_v1.py
"""

from policy_registry_v1 import (
    build_source_mode,
    default_actual_mappo_spec,
    default_noop_spec,
    default_placeholder_spec,
    default_rulebased_spec,
    make_policy_spec,
)


def test_placeholder_spec():
    spec = default_placeholder_spec("A")

    assert spec["condition_id"] == "A"
    assert spec["policy_kind"] == "placeholder"
    assert spec["policy_source"] == "placeholder_policy"
    assert spec["source_mode"] == "stub_A_placeholder_smoke"
    assert spec["qwen_enabled"] is False
    assert spec["checkpoint_required"] is False


def test_actual_mappo_spec_requires_checkpoint():
    spec = default_actual_mappo_spec(
        "A",
        "artifacts/experiment_A_v1/checkpoints/best.pt",
        require_existing_checkpoint=False,
    )

    assert spec["condition_id"] == "A"
    assert spec["policy_kind"] == "mappo"
    assert spec["policy_source"] == "mappo_policy"
    assert spec["source_mode"] == "causal_A_mappo_policy_v1"
    assert spec["qwen_enabled"] is False
    assert spec["checkpoint_required"] is True


def test_missing_mappo_checkpoint_rejected():
    try:
        default_actual_mappo_spec(
            "A",
            "",
            require_existing_checkpoint=False,
        )
    except FileNotFoundError:
        return

    raise AssertionError("missing MAPPO checkpoint was not rejected")


def test_causal_placeholder_rejected():
    try:
        make_policy_spec(
            condition_id="A",
            policy_kind="placeholder",
            checkpoint_path=None,
            qwen_enabled=False,
            causal=True,
        )
    except ValueError:
        return

    raise AssertionError("causal placeholder was not rejected")


def test_a_family_qwen_rejected():
    try:
        make_policy_spec(
            condition_id="A",
            policy_kind="mappo",
            checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
            qwen_enabled=True,
            causal=True,
        )
    except ValueError:
        return

    raise AssertionError("A-family Qwen intervention was not rejected")


def test_baseline_specs():
    noop = default_noop_spec()
    rulebased = default_rulebased_spec()

    assert noop["condition_id"] == "B1"
    assert noop["policy_kind"] == "noop"
    assert noop["source_mode"] == "replay_B1_noop_noncausal"

    assert rulebased["condition_id"] == "B2"
    assert rulebased["policy_kind"] == "rulebased"
    assert rulebased["source_mode"] == "replay_B2_rulebased_noncausal"


def test_source_mode_builder():
    assert build_source_mode("A", "mappo", causal=True) == "causal_A_mappo_policy_v1"
    assert build_source_mode("A90", "mappo", causal=True) == "causal_A90_mappo_policy_v1"
    assert build_source_mode("A", "placeholder", causal=False) == "stub_A_placeholder_smoke"


def main():
    tests = [
        test_placeholder_spec,
        test_actual_mappo_spec_requires_checkpoint,
        test_missing_mappo_checkpoint_rejected,
        test_causal_placeholder_rejected,
        test_a_family_qwen_rejected,
        test_baseline_specs,
        test_source_mode_builder,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] policy_registry_v1 self-test passed")


if __name__ == "__main__":
    main()
