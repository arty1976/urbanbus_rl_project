from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def append_training_dir_to_path() -> Path:
    here = Path(__file__).resolve()
    training_dir = here.parents[1]
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def assert_raises(name: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"{name}: expected exception but none was raised")


def main() -> None:
    training_dir = append_training_dir_to_path()

    from rewards.energy_proxy_model_v1 import (
        DEFAULT_CONFIG,
        EnergyProxyConfig,
        compute_energy_kwh_equiv,
        compute_energy_proxy_for_record,
        compute_energy_proxy_for_records,
        compute_energy_proxy_from_components,
        expected_contract_fields,
        summarize_energy_proxy_records,
        validate_config_constants,
    )

    validate_config_constants(DEFAULT_CONFIG)

    fields = expected_contract_fields(DEFAULT_CONFIG)
    if fields["energy_proxy_model_version"] != "daegu_energy_proxy_v1":
        raise AssertionError("energy_proxy_model_version mismatch")
    assert_close("k_dist", fields["k_dist_kwh_per_m"], 0.0012)
    assert_close("k_acc", fields["k_acc_kwh_per_event"], 0.1800)
    assert_close("k_idle", fields["k_idle_kwh_per_sec"], 0.0080)

    zero = compute_energy_proxy_from_components(
        distance_m=0.0,
        acceleration_event_count=0.0,
        hold_seconds=0.0,
        passenger_served_count=10.0,
    )
    assert_close("zero energy", zero["energy_kwh_equiv"], 0.0)
    assert_close("zero per passenger", zero["energy_proxy_per_passenger"], 0.0)

    sample = compute_energy_proxy_from_components(
        distance_m=1000.0,
        acceleration_event_count=10.0,
        hold_seconds=60.0,
        passenger_served_count=30.0,
    )

    expected_energy = 0.0012 * 1000.0 + 0.1800 * 10.0 + 0.0080 * 60.0
    expected_per_passenger = expected_energy / 30.0

    assert_close("sample energy", sample["energy_kwh_equiv"], expected_energy)
    assert_close(
        "sample per passenger",
        sample["energy_proxy_per_passenger"],
        expected_per_passenger,
    )

    record = {
        "distance_m": 2500.0,
        "acceleration_event_count": 14.0,
        "hold_seconds": 90.0,
        "passenger_served_count": 45.0,
    }
    record_out = compute_energy_proxy_for_record(record)
    expected_record_energy = 0.0012 * 2500.0 + 0.1800 * 14.0 + 0.0080 * 90.0
    assert_close("record energy", record_out["energy_kwh_equiv"], expected_record_energy)

    batch = compute_energy_proxy_for_records(
        [
            {
                "distance_m": 100.0,
                "acceleration_event_count": 1.0,
                "hold_seconds": 5.0,
                "passenger_served_count": 3.0,
            },
            {
                "distance_m": 200.0,
                "acceleration_event_count": 2.0,
                "hold_seconds": 10.0,
                "passenger_served_count": 4.0,
            },
        ]
    )
    summary = summarize_energy_proxy_records(batch)
    if summary["record_count"] != 2:
        raise AssertionError("summary record_count mismatch")
    if summary["energy_kwh_equiv_total"] is None:
        raise AssertionError("summary energy total should not be None")

    zero_passenger = compute_energy_proxy_from_components(
        distance_m=10.0,
        acceleration_event_count=0.0,
        hold_seconds=0.0,
        passenger_served_count=0.0,
    )
    if not math.isfinite(float(zero_passenger["energy_proxy_per_passenger"])):
        raise AssertionError("zero passenger denominator should be epsilon-protected")

    assert_raises(
        "negative distance",
        lambda: compute_energy_kwh_equiv(
            distance_m=-1.0,
            acceleration_event_count=0.0,
            hold_seconds=0.0,
        ),
    )

    bad_config = EnergyProxyConfig(k_dist_kwh_per_m=0.9999)
    assert_raises(
        "bad config constants",
        lambda: validate_config_constants(bad_config),
    )

    out_dir = training_dir.parent / "artifacts" / "experiment_A_v1" / "energy_proxy_model_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "PASS",
        "step": 43,
        "model_version": "daegu_energy_proxy_v1",
        "energy_proxy_unit": "kwh_equivalent",
        "sample": sample,
        "record_out": record_out,
        "batch_summary": summary,
        "contract_fields": fields,
        "note": "This validates the KPI-side energy proxy model only. Reward integration is not yet performed.",
    }

    out_path = out_dir / "step43_energy_proxy_smoke_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] Step 43 Daegu energy proxy model self-test PASS")
    print(f"[OK] report: {out_path}")


if __name__ == "__main__":
    main()
