"""
Self-test for policy_inference_boundary_v1.py.

Run:
    python .\05_training\policies\test_policy_inference_boundary_v1.py
"""

from policy_inference_boundary_v1 import (
    PolicyInferenceRequest,
    boundary_to_rollout_provenance,
    build_policy_inference_boundary,
    make_a_family_actual_boundary,
    make_a_family_placeholder_boundary,
    validate_boundary_for_rollout_claim,
)


def test_placeholder_boundary():
    boundary = build_policy_inference_boundary(
        PolicyInferenceRequest(
            condition_id="A",
            policy_kind="placeholder",
            checkpoint_path=None,
            qwen_enabled=False,
            causal=False,
            require_existing_checkpoint=False,
        )
    )

    assert boundary.condition_id == "A"
    assert boundary.policy_kind == "placeholder"
    assert boundary.placeholder_allowed is True
    assert boundary.causal_claim_allowed is False
    assert boundary.actual_mappo_inference_ready is False


def test_actual_mappo_boundary():
    boundary = build_policy_inference_boundary(
        PolicyInferenceRequest(
            condition_id="A",
            policy_kind="mappo",
            checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
            qwen_enabled=False,
            causal=True,
            require_existing_checkpoint=False,
        )
    )

    assert boundary.condition_id == "A"
    assert boundary.policy_kind == "mappo"
    assert boundary.policy_source == "mappo_policy"
    assert boundary.source_mode == "causal_A_mappo_policy_v1"
    assert boundary.placeholder_allowed is False
    assert boundary.causal_claim_allowed is True
    assert boundary.actual_mappo_inference_ready is True

    validate_boundary_for_rollout_claim(boundary)


def test_missing_checkpoint_path_rejected():
    try:
        build_policy_inference_boundary(
            PolicyInferenceRequest(
                condition_id="A",
                policy_kind="mappo",
                checkpoint_path="",
                qwen_enabled=False,
                causal=True,
                require_existing_checkpoint=False,
            )
        )
    except FileNotFoundError:
        return

    raise AssertionError("missing checkpoint_path was not rejected")


def test_causal_placeholder_rejected():
    try:
        build_policy_inference_boundary(
            PolicyInferenceRequest(
                condition_id="A",
                policy_kind="placeholder",
                checkpoint_path=None,
                qwen_enabled=False,
                causal=True,
                require_existing_checkpoint=False,
            )
        )
    except ValueError:
        return

    raise AssertionError("causal placeholder was not rejected")


def test_qwen_rejected_for_a_family():
    try:
        build_policy_inference_boundary(
            PolicyInferenceRequest(
                condition_id="A",
                policy_kind="mappo",
                checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
                qwen_enabled=True,
                causal=True,
                require_existing_checkpoint=False,
            )
        )
    except ValueError:
        return

    raise AssertionError("A-family Qwen intervention was not rejected")


def test_rollout_provenance_fields():
    boundary = build_policy_inference_boundary(
        PolicyInferenceRequest(
            condition_id="A90",
            policy_kind="mappo",
            checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
            qwen_enabled=False,
            causal=True,
            require_existing_checkpoint=False,
        )
    )

    provenance = boundary_to_rollout_provenance(boundary)

    assert provenance["policy_source"] == "mappo_policy"
    assert provenance["policy_checkpoint_path"] == "artifacts/experiment_A_v1/checkpoints/best.pt"
    assert provenance["source_mode"] == "causal_A90_mappo_policy_v1"
    assert provenance["qwen_trigger_rate"] == 0.0


def test_helper_dict_outputs():
    actual = make_a_family_actual_boundary(
        condition_id="A80",
        checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
        require_existing_checkpoint=False,
    )

    placeholder = make_a_family_placeholder_boundary("A70")

    assert actual["source_mode"] == "causal_A80_mappo_policy_v1"
    assert actual["causal_claim_allowed"] is True

    assert placeholder["source_mode"] == "stub_A70_placeholder_smoke"
    assert placeholder["causal_claim_allowed"] is False


def main():
    tests = [
        test_placeholder_boundary,
        test_actual_mappo_boundary,
        test_missing_checkpoint_path_rejected,
        test_causal_placeholder_rejected,
        test_qwen_rejected_for_a_family,
        test_rollout_provenance_fields,
        test_helper_dict_outputs,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] policy_inference_boundary_v1 self-test passed")


if __name__ == "__main__":
    main()
