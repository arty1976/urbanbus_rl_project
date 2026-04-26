"""
Self-test for mappo_policy_interface_v1.py.

Run:
    python ./05_training/policies/test_mappo_policy_interface_v1.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from mappo_policy_interface_v1 import (
    interface_contract,
    run_policy_interface_once,
    sample_rollout_row,
    validate_action_dict,
    validate_checkpoint_metadata_dict,
    validate_observation_dict,
    write_sample_metadata,
)


def test_checkpoint_metadata_validation_passes() -> None:
    contract = interface_contract()
    payload = dict(contract["expected_checkpoint_metadata"])
    payload["trained_model"] = True

    errors = validate_checkpoint_metadata_dict(payload)

    assert errors == []


def test_checkpoint_metadata_rejects_fake_untrained_model() -> None:
    contract = interface_contract()
    payload = dict(contract["expected_checkpoint_metadata"])
    payload["trained_model"] = False

    errors = validate_checkpoint_metadata_dict(payload)

    assert errors
    assert any("trained_model" in err for err in errors)


def test_observation_validation_passes() -> None:
    row = sample_rollout_row("A80")
    errors = validate_observation_dict(row)
    assert errors == []


def test_observation_rejects_qwen_trigger() -> None:
    row = sample_rollout_row("A")
    row["qwen_trigger_rate"] = 0.1

    errors = validate_observation_dict(row)

    assert errors
    assert any("qwen_trigger_rate" in err for err in errors)


def test_policy_interface_once() -> None:
    out_dir = Path("artifacts/step35_mappo_policy_interface_test")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    ckpt = write_sample_metadata(out_dir / "sample_checkpoint_metadata.json", trained_model=True)
    result = run_policy_interface_once(ckpt, sample_rollout_row("A80"))

    assert result["passed"] is True
    assert result["action"]["policy_source"] == "mappo_policy"
    assert result["action"]["source_mode"] == "causal_A80_mappo_policy_v1"
    assert result["action"]["qwen_trigger_rate"] == 0.0
    assert result["action"]["policy_debug"]["mock_action"] is True


def test_action_rejects_placeholder_source_mode() -> None:
    out_dir = Path("artifacts/step35_mappo_policy_interface_test_action")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    ckpt = write_sample_metadata(out_dir / "sample_checkpoint_metadata.json", trained_model=True)
    result = run_policy_interface_once(ckpt, sample_rollout_row("A70"))
    action = result["action"]
    action["source_mode"] = "stub_A70_placeholder_smoke"

    errors = validate_action_dict(action)

    assert errors
    assert any("source_mode" in err for err in errors)


def main() -> None:
    tests = [
        test_checkpoint_metadata_validation_passes,
        test_checkpoint_metadata_rejects_fake_untrained_model,
        test_observation_validation_passes,
        test_observation_rejects_qwen_trigger,
        test_policy_interface_once,
        test_action_rejects_placeholder_source_mode,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] mappo_policy_interface_v1 self-test passed")


if __name__ == "__main__":
    main()
