"""
Self-test for mappo_reward_v1.py.

Run:
    python .\05_training\rewards\test_mappo_reward_v1.py
"""

from mappo_reward_v1 import compute_total_reward


def base_metrics():
    return {
        "condition_id": "A",
        "avg_wait_seconds": 300.0,
        "passenger_wait_p95_seconds": 600.0,
        "passenger_service_rate": 0.98,
        "energy_proxy": 1000.0,
        "energy_proxy_per_passenger": 10.0,
        "on_time_rate": 0.85,
        "bunching_rate": 0.05,
        "fleet_reduction_ratio": 0.10,
        "qwen_trigger_rate": 0.0,
    }


def test_low_service_rate_constraint():
    m = base_metrics()
    m["passenger_service_rate"] = 0.70

    out = compute_total_reward(m)

    assert out["constraint_violation_flags"]["low_service_rate"] is True
    assert out["reward_constraint"] < 0.0


def test_p95_wait_penalty_increases():
    base = compute_total_reward(base_metrics())

    m = base_metrics()
    m["passenger_wait_p95_seconds"] = 900.0

    out = compute_total_reward(m)

    assert out["reward_long_wait"] < base["reward_long_wait"]
    assert out["constraint_violation_flags"]["p95_wait_exceeds_baseline_ratio"] is True


def test_lower_energy_per_passenger_improves_reward():
    low = base_metrics()
    low["energy_proxy_per_passenger"] = 5.0

    high = base_metrics()
    high["energy_proxy_per_passenger"] = 15.0

    out_low = compute_total_reward(low)
    out_high = compute_total_reward(high)

    assert out_low["reward_energy"] > out_high["reward_energy"]
    assert out_low["reward_total"] > out_high["reward_total"]


def test_fleet_bonus_cannot_hide_bad_service():
    good = base_metrics()
    good["fleet_reduction_ratio"] = 0.00
    good["passenger_service_rate"] = 0.98

    bad = base_metrics()
    bad["fleet_reduction_ratio"] = 0.70
    bad["passenger_service_rate"] = 0.70

    out_good = compute_total_reward(good)
    out_bad = compute_total_reward(bad)

    assert out_bad["constraint_violation_flags"]["low_service_rate"] is True
    assert out_bad["reward_total"] < out_good["reward_total"]


def test_qwen_trigger_violation_for_A():
    m = base_metrics()
    m["condition_id"] = "A"
    m["qwen_trigger_rate"] = 0.10

    out = compute_total_reward(m)

    assert out["constraint_violation_flags"]["qwen_trigger_rate_violation_for_A"] is True
    assert out["reward_constraint"] < 0.0


def main():
    tests = [
        test_low_service_rate_constraint,
        test_p95_wait_penalty_increases,
        test_lower_energy_per_passenger_improves_reward,
        test_fleet_bonus_cannot_hide_bad_service,
        test_qwen_trigger_violation_for_A,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] mappo_reward_v1 self-test passed")


if __name__ == "__main__":
    main()
