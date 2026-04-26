"""
Self-test for rollout_schema_v1.py.

Run:
    python ./05_training/rollouts\test_rollout_schema_v1.py
"""

from rollout_schema_v1 import (
    augment_rollout_row,
    sample_causal_a_row,
    source_mode_allows_causal_claim,
    source_mode_is_placeholder,
    validate_qwen_for_condition,
    validate_rollout_rows,
    validate_same_demand_by_window,
)


def test_sample_row_strict_validation_passes():
    row = augment_rollout_row(sample_causal_a_row())
    summary = validate_rollout_rows([row], strict=True)

    assert summary["passed"] is True
    assert summary["error_count"] == 0


def test_derived_service_rate():
    row = sample_causal_a_row()
    row["passenger_service_rate"] = None
    row["passenger_demand_generated"] = 40
    row["passenger_served_count"] = 39

    out = augment_rollout_row(row)

    assert abs(out["passenger_service_rate"] - 0.975) < 1e-12


def test_derived_energy_per_passenger():
    row = sample_causal_a_row()
    row["energy_proxy"] = 390.0
    row["energy_proxy_per_passenger"] = None
    row["passenger_served_count"] = 39

    out = augment_rollout_row(row)

    assert abs(out["energy_proxy_per_passenger"] - 10.0) < 1e-12


def test_derived_fleet_reduction_ratio():
    row = sample_causal_a_row()
    row["active_bus_count"] = 8
    row["baseline_bus_count"] = 10
    row["fleet_reduction_ratio"] = None

    out = augment_rollout_row(row)

    assert abs(out["fleet_reduction_ratio"] - 0.2) < 1e-12


def test_qwen_violation_for_a_family():
    row = sample_causal_a_row()
    row["condition_id"] = "A"
    row["qwen_trigger_rate"] = 0.1

    errors = validate_qwen_for_condition(row)

    assert errors


def test_source_mode_causal_and_placeholder_detection():
    assert source_mode_allows_causal_claim("causal_A_mappo_policy_v1") is True
    assert source_mode_allows_causal_claim("replay_A_pure_mappo_noncausal") is False
    assert source_mode_allows_causal_claim("stub_A_pure_mappo_smoke") is False
    assert source_mode_is_placeholder("stub_A_pure_mappo_smoke") is True


def test_same_demand_by_window_passes():
    rows = []

    for condition in ["B1", "B2", "A", "A90", "A80", "A70"]:
        row = sample_causal_a_row()
        row["condition_id"] = condition
        row["window_id"] = "w001"
        row["passenger_demand_generated"] = 100
        row["source_mode"] = f"causal_{condition}_policy_v1"
        rows.append(row)

    summary = validate_same_demand_by_window(rows)

    assert summary["passed"] is True


def test_same_demand_by_window_fails_when_demand_differs():
    rows = []

    row_a = sample_causal_a_row()
    row_a["condition_id"] = "A"
    row_a["window_id"] = "w001"
    row_a["passenger_demand_generated"] = 100

    row_b = sample_causal_a_row()
    row_b["condition_id"] = "A90"
    row_b["window_id"] = "w001"
    row_b["passenger_demand_generated"] = 90

    rows.append(row_a)
    rows.append(row_b)

    summary = validate_same_demand_by_window(rows)

    assert summary["passed"] is False
    assert summary["violation_count"] == 1


def main():
    tests = [
        test_sample_row_strict_validation_passes,
        test_derived_service_rate,
        test_derived_energy_per_passenger,
        test_derived_fleet_reduction_ratio,
        test_qwen_violation_for_a_family,
        test_source_mode_causal_and_placeholder_detection,
        test_same_demand_by_window_passes,
        test_same_demand_by_window_fails_when_demand_differs,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] rollout_schema_v1 self-test passed")


if __name__ == "__main__":
    main()
