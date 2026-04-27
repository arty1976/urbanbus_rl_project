from __future__ import annotations

import json
import sys
from pathlib import Path


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def assert_raises(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        print(f"[OK] expected failure: {name}: {exc}")
        return
    raise AssertionError(f"{name}: expected failure but none was raised")


def main() -> None:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    from policies.a_family_rollout_policy_integration import (
        A_FAMILY_CONDITIONS,
        AFamilyPolicyIntegrationConfig,
        build_a_family_policy_integration_plan,
        build_matrix_preview,
        plan_to_dict,
    )

    out_dir = project_root / "artifacts" / "experiment_A_v1" / "a_family_policy_integration_plan_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke_plans = []
    actual_noncausal_plans = []
    actual_causal_plans = []

    for cid in A_FAMILY_CONDITIONS:
        smoke_plan = build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id=cid,
                policy_source_mode="mappo_smoke",
                checkpoint_path="C:/tmp/smoke.pt",
                checkpoint_validation_mode="smoke",
                causal_simulator=False,
            )
        )
        smoke_d = plan_to_dict(smoke_plan)

        if smoke_d["expected_policy_source"] != "mappo_policy":
            raise AssertionError("mappo_smoke expected_policy_source mismatch")
        if smoke_d["checkpoint_required"] is not True:
            raise AssertionError("mappo_smoke must require checkpoint")
        if smoke_d["checkpoint_validation_mode"] != "smoke":
            raise AssertionError("mappo_smoke validation mode mismatch")
        if smoke_d["actual_policy_claim_possible"] is not False:
            raise AssertionError("mappo_smoke must not allow actual policy claim")
        if smoke_d["causal_policy_claim_possible"] is not False:
            raise AssertionError("mappo_smoke must not allow causal policy claim")
        if not smoke_d["expected_source_mode"].startswith(f"noncausal_{cid}_mappo_policy_smoke"):
            raise AssertionError("mappo_smoke source_mode mismatch")

        smoke_plans.append(smoke_d)

        actual_noncausal = build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id=cid,
                policy_source_mode="mappo_actual",
                checkpoint_path="C:/tmp/best.pt",
                checkpoint_validation_mode="actual",
                causal_simulator=False,
            )
        )
        actual_noncausal_d = plan_to_dict(actual_noncausal)

        if actual_noncausal_d["actual_policy_claim_possible"] is not True:
            raise AssertionError("mappo_actual should allow actual policy claim after checkpoint validation")
        if actual_noncausal_d["causal_policy_claim_possible"] is not False:
            raise AssertionError("noncausal actual must not allow causal claim")
        if not actual_noncausal_d["expected_source_mode"].startswith(f"noncausal_{cid}_mappo_policy_actual"):
            raise AssertionError("mappo_actual noncausal source_mode mismatch")

        actual_noncausal_plans.append(actual_noncausal_d)

        actual_causal = build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id=cid,
                policy_source_mode="mappo_actual",
                checkpoint_path="C:/tmp/best.pt",
                checkpoint_validation_mode="actual",
                causal_simulator=True,
            )
        )
        actual_causal_d = plan_to_dict(actual_causal)

        if actual_causal_d["causal_policy_claim_possible"] is not True:
            raise AssertionError("causal actual should allow causal claim possibility")
        if not actual_causal_d["expected_source_mode"].startswith(f"causal_{cid}_mappo_policy"):
            raise AssertionError("mappo_actual causal source_mode mismatch")

        actual_causal_plans.append(actual_causal_d)

    mock_plan = build_a_family_policy_integration_plan(
        AFamilyPolicyIntegrationConfig(
            condition_id="A",
            policy_source_mode="mock_smoke",
            checkpoint_path="",
            checkpoint_validation_mode="",
            causal_simulator=False,
            allow_mock=True,
        )
    )
    mock_d = plan_to_dict(mock_plan)

    if mock_d["expected_policy_source"] != "mock_policy":
        raise AssertionError("mock expected_policy_source mismatch")
    if mock_d["checkpoint_required"] is not False:
        raise AssertionError("mock must not require checkpoint")
    if mock_d["actual_policy_claim_possible"] is not False:
        raise AssertionError("mock must not allow actual claim")

    assert_raises(
        "unsupported condition",
        lambda: build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id="B2",
                policy_source_mode="mappo_smoke",
                checkpoint_path="C:/tmp/smoke.pt",
                checkpoint_validation_mode="smoke",
            )
        ),
    )

    assert_raises(
        "missing checkpoint for mappo smoke",
        lambda: build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id="A",
                policy_source_mode="mappo_smoke",
                checkpoint_path="",
                checkpoint_validation_mode="smoke",
            )
        ),
    )

    assert_raises(
        "wrong validation mode for mappo actual",
        lambda: build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id="A",
                policy_source_mode="mappo_actual",
                checkpoint_path="C:/tmp/best.pt",
                checkpoint_validation_mode="smoke",
            )
        ),
    )

    assert_raises(
        "mock requires explicit allow_mock",
        lambda: build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id="A",
                policy_source_mode="mock_smoke",
                checkpoint_path="",
                checkpoint_validation_mode="",
                allow_mock=False,
            )
        ),
    )

    assert_raises(
        "qwen train forbidden",
        lambda: build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id="A",
                policy_source_mode="mappo_smoke",
                checkpoint_path="C:/tmp/smoke.pt",
                checkpoint_validation_mode="smoke",
                qwen_train=True,
            )
        ),
    )

    assert_raises(
        "qwen trigger forbidden",
        lambda: build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id="A",
                policy_source_mode="mappo_smoke",
                checkpoint_path="C:/tmp/smoke.pt",
                checkpoint_validation_mode="smoke",
                qwen_trigger_rate=0.5,
            )
        ),
    )

    matrix = build_matrix_preview(checkpoint_path="C:/tmp/required_checkpoint.pt")

    report = {
        "status": "PASS",
        "step": 51,
        "contract": "a_family_rollout_policy_integration_v1",
        "conditions": A_FAMILY_CONDITIONS,
        "mock_plan": mock_d,
        "smoke_plans": smoke_plans,
        "actual_noncausal_plans": actual_noncausal_plans,
        "actual_causal_plans": actual_causal_plans,
        "matrix_preview": matrix,
        "note": (
            "This validates the A-family rollout policy integration plan only. "
            "Actual rollout writer wiring is the next step."
        ),
    }

    out_report = out_dir / "step51_a_family_policy_integration_plan_report.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Step 51 A-family policy integration plan self-test PASS")
    print(f"[OK] report: {out_report}")


if __name__ == "__main__":
    main()
