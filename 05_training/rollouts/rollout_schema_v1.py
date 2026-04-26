"""
rollout_schema_v1.py
====================

Rollout schema contract for UrbanBus MAPPO.

Purpose
-------
This module defines the official extended window_rollup schema that will be
used when replacing placeholder A/A90/A80/A70 policies with actual MAPPO
rollouts.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.
Qwen is disabled for A_pure_mappo.

Design rules
------------
1. B0R/B1/B2/A/A90/A80/A70 must use the same passenger_demand_generated
   for the same window.
2. A90/A80/A70 may reduce active_bus_count, but must not change demand.
3. Placeholder outputs and actual policy inference outputs must be separated
   by policy_source and source_mode.
4. Only causal_* source_mode rows may be used for paper-level performance
   claims.
5. A condition must have qwen_trigger_rate == 0.0.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "rollout_schema_v1"


VALID_TIME_BANDS = {
    "peak",
    "offpeak",
    "night",
}


VALID_CONDITION_IDS = {
    "B0R",
    "B0",
    "B1",
    "B2",
    "A",
    "A90",
    "A80",
    "A70",
}


CORE_ID_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
]


OFFICIAL_ROLLUP_RAW_COLUMNS = [
    "headway_mean_seconds",
    "headway_std_seconds",
    "headway_sample_count",
    "bunching_event_count",
    "headway_event_count",
    "wait_total_passenger_seconds",
    "wait_passenger_count",
    "ontime_event_count",
    "schedulable_arrival_count",
    "intervention_count",
    "decision_step_count",
    "energy_proxy_total",
]


EXTENDED_PASSENGER_COLUMNS = [
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "long_wait_passenger_count",
]


EXTENDED_ENERGY_FLEET_COLUMNS = [
    "energy_proxy",
    "energy_proxy_per_passenger",
    "active_bus_count",
    "baseline_bus_count",
    "fleet_reduction_ratio",
]


POLICY_PROVENANCE_COLUMNS = [
    "policy_source",
    "policy_checkpoint_path",
    "source_mode",
    "qwen_trigger_rate",
    "effective_replay_step_minutes",
]


REWARD_COLUMNS = [
    "reward_total",
    "reward_service",
    "reward_avg_wait",
    "reward_long_wait",
    "reward_on_time",
    "reward_bunching",
    "reward_energy",
    "reward_fleet",
    "reward_constraint",
]


STRICT_REQUIRED_COLUMNS = (
    CORE_ID_COLUMNS
    + OFFICIAL_ROLLUP_RAW_COLUMNS
    + EXTENDED_PASSENGER_COLUMNS
    + EXTENDED_ENERGY_FLEET_COLUMNS
    + POLICY_PROVENANCE_COLUMNS
    + REWARD_COLUMNS
)


MINIMAL_REQUIRED_COLUMNS = (
    CORE_ID_COLUMNS
    + [
        "source_mode",
        "policy_source",
    ]
)


PLACEHOLDER_SOURCE_PATTERNS = [
    "stub",
    "placeholder",
    "smoke",
]


NON_CAUSAL_SOURCE_PATTERNS = [
    "historical",
    "legacy",
    "replay",
    "noncausal",
    "non_causal",
    "non-causal",
    "smoke",
    "stub",
    "placeholder",
]


CAUSAL_SOURCE_PREFIX = "causal_"


def schema_contract() -> Dict[str, Any]:
    """Return the rollout schema contract as a plain dictionary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "valid_condition_ids": sorted(VALID_CONDITION_IDS),
        "valid_time_bands": sorted(VALID_TIME_BANDS),
        "core_id_columns": list(CORE_ID_COLUMNS),
        "official_rollup_raw_columns": list(OFFICIAL_ROLLUP_RAW_COLUMNS),
        "extended_passenger_columns": list(EXTENDED_PASSENGER_COLUMNS),
        "extended_energy_fleet_columns": list(EXTENDED_ENERGY_FLEET_COLUMNS),
        "policy_provenance_columns": list(POLICY_PROVENANCE_COLUMNS),
        "reward_columns": list(REWARD_COLUMNS),
        "minimal_required_columns": list(MINIMAL_REQUIRED_COLUMNS),
        "strict_required_columns": list(STRICT_REQUIRED_COLUMNS),
        "rules": {
            "same_demand_by_window": True,
            "a_family_qwen_disabled": True,
            "a90_a80_a70_reduce_active_bus_count_only": True,
            "causal_claims_require_causal_source_mode": True,
            "placeholder_must_not_be_mixed_with_actual_policy": True,
        },
    }


def expected_columns(strict: bool = True) -> List[str]:
    """Return expected rollout columns."""
    return list(STRICT_REQUIRED_COLUMNS if strict else MINIMAL_REQUIRED_COLUMNS)


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default

    try:
        return int(value)
    except Exception:
        return default


def _safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
    default: Optional[float] = None,
) -> Optional[float]:
    if numerator is None:
        return default

    if denominator is None or denominator <= 0:
        return default

    return float(numerator / denominator)


def normalize_condition_id(condition_id: Any) -> str:
    return str(condition_id).strip().upper()


def normalize_time_band(time_band: Any) -> str:
    return str(time_band).strip().lower()


def source_mode_allows_causal_claim(source_mode: Any) -> bool:
    """Return True only for source_mode values that start with causal_."""
    value = str(source_mode or "").strip().lower()

    if not value.startswith(CAUSAL_SOURCE_PREFIX):
        return False

    for pattern in NON_CAUSAL_SOURCE_PATTERNS:
        if pattern in value:
            return False

    return True


def source_mode_is_placeholder(source_mode: Any) -> bool:
    value = str(source_mode or "").strip().lower()
    return any(pattern in value for pattern in PLACEHOLDER_SOURCE_PATTERNS)


def validate_qwen_for_condition(row: Mapping[str, Any]) -> List[str]:
    """Validate that A family conditions do not use Qwen unless explicitly allowed later."""
    errors: List[str] = []

    condition_id = normalize_condition_id(row.get("condition_id"))
    qwen_trigger_rate = _as_float(row.get("qwen_trigger_rate"), default=0.0)

    if condition_id in {"A", "A90", "A80", "A70"}:
        if qwen_trigger_rate is None:
            qwen_trigger_rate = 0.0

        if abs(qwen_trigger_rate) > 1e-12:
            errors.append(
                "A-family conditions must have qwen_trigger_rate == 0.0"
            )

    return errors


def derive_passenger_service_rate(row: Mapping[str, Any]) -> Optional[float]:
    service_rate = _as_float(row.get("passenger_service_rate"), default=None)

    if service_rate is not None:
        return service_rate

    served = _as_float(row.get("passenger_served_count"), default=None)
    generated = _as_float(row.get("passenger_demand_generated"), default=None)

    return _safe_ratio(served, generated, default=None)


def derive_energy_proxy_per_passenger(row: Mapping[str, Any]) -> Optional[float]:
    existing = _as_float(row.get("energy_proxy_per_passenger"), default=None)

    if existing is not None:
        return existing

    energy_proxy = _as_float(row.get("energy_proxy"), default=None)

    if energy_proxy is None:
        energy_proxy = _as_float(row.get("energy_proxy_total"), default=None)

    served = _as_float(row.get("passenger_served_count"), default=None)

    if served is None:
        served = _as_float(row.get("wait_passenger_count"), default=None)

    return _safe_ratio(energy_proxy, served, default=None)


def derive_fleet_reduction_ratio(row: Mapping[str, Any]) -> Optional[float]:
    existing = _as_float(row.get("fleet_reduction_ratio"), default=None)

    if existing is not None:
        return existing

    active = _as_float(row.get("active_bus_count"), default=None)
    baseline = _as_float(row.get("baseline_bus_count"), default=None)

    if active is None or baseline is None or baseline <= 0:
        return None

    ratio = (baseline - active) / baseline

    if ratio < 0:
        ratio = 0.0

    if ratio > 1:
        ratio = 1.0

    return float(ratio)


def augment_rollout_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of row with derived Step 22 fields filled when possible.

    This function does not invent passenger demand. It only derives ratios when
    numerator and denominator are already present.
    """
    out = deepcopy(dict(row))

    if out.get("passenger_service_rate") is None:
        out["passenger_service_rate"] = derive_passenger_service_rate(out)

    if out.get("energy_proxy") is None and out.get("energy_proxy_total") is not None:
        out["energy_proxy"] = _as_float(out.get("energy_proxy_total"), default=None)

    if out.get("energy_proxy_per_passenger") is None:
        out["energy_proxy_per_passenger"] = derive_energy_proxy_per_passenger(out)

    if out.get("fleet_reduction_ratio") is None:
        out["fleet_reduction_ratio"] = derive_fleet_reduction_ratio(out)

    if out.get("qwen_trigger_rate") is None:
        out["qwen_trigger_rate"] = 0.0

    if out.get("policy_source") is None:
        out["policy_source"] = "unknown"

    if out.get("source_mode") is None:
        out["source_mode"] = "unknown"

    return out


def validate_rollout_row(
    row: Mapping[str, Any],
    strict: bool = False,
) -> List[str]:
    """Validate one rollout row and return a list of error messages."""
    errors: List[str] = []
    columns = set(row.keys())

    required = STRICT_REQUIRED_COLUMNS if strict else MINIMAL_REQUIRED_COLUMNS

    missing = [col for col in required if col not in columns]

    if missing:
        errors.append("missing columns: " + ", ".join(missing))

    condition_id = normalize_condition_id(row.get("condition_id"))

    if condition_id and condition_id not in VALID_CONDITION_IDS:
        errors.append(f"invalid condition_id: {condition_id}")

    time_band = normalize_time_band(row.get("time_band"))

    if time_band and time_band not in VALID_TIME_BANDS:
        errors.append(f"invalid time_band: {time_band}")

    horizon = _as_int(row.get("evaluation_horizon_minutes"), default=None)

    if horizon is not None and horizon != 30:
        errors.append("evaluation_horizon_minutes must be 30")

    demand_generated = _as_float(row.get("passenger_demand_generated"), default=None)
    served_count = _as_float(row.get("passenger_served_count"), default=None)
    service_rate = _as_float(row.get("passenger_service_rate"), default=None)

    if demand_generated is not None and demand_generated < 0:
        errors.append("passenger_demand_generated must be non-negative")

    if served_count is not None and served_count < 0:
        errors.append("passenger_served_count must be non-negative")

    if (
        demand_generated is not None
        and served_count is not None
        and served_count > demand_generated + 1e-9
    ):
        errors.append("passenger_served_count cannot exceed passenger_demand_generated")

    if service_rate is not None and (service_rate < 0 or service_rate > 1):
        errors.append("passenger_service_rate must be between 0 and 1")

    fleet_ratio = _as_float(row.get("fleet_reduction_ratio"), default=None)

    if fleet_ratio is not None and (fleet_ratio < 0 or fleet_ratio > 1):
        errors.append("fleet_reduction_ratio must be between 0 and 1")

    active_bus_count = _as_float(row.get("active_bus_count"), default=None)
    baseline_bus_count = _as_float(row.get("baseline_bus_count"), default=None)

    if active_bus_count is not None and active_bus_count < 0:
        errors.append("active_bus_count must be non-negative")

    if baseline_bus_count is not None and baseline_bus_count < 0:
        errors.append("baseline_bus_count must be non-negative")

    source_mode = str(row.get("source_mode", ""))

    if source_mode_is_placeholder(source_mode):
        policy_source = str(row.get("policy_source", "")).lower()

        if "placeholder" not in policy_source and "stub" not in policy_source:
            errors.append(
                "placeholder source_mode should use placeholder/stub policy_source"
            )

    errors.extend(validate_qwen_for_condition(row))

    return errors


def validate_rollout_rows(
    rows: Iterable[Mapping[str, Any]],
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate multiple rollout rows and return a summary."""
    row_count = 0
    error_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        row_count += 1
        augmented = augment_rollout_row(row)
        errors = validate_rollout_row(augmented, strict=strict)

        if errors:
            error_rows.append(
                {
                    "row_index": idx,
                    "condition_id": augmented.get("condition_id"),
                    "window_id": augmented.get("window_id"),
                    "errors": errors,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "strict": bool(strict),
        "row_count": row_count,
        "error_count": len(error_rows),
        "error_rows": error_rows,
        "passed": len(error_rows) == 0,
    }


def validate_same_demand_by_window(
    rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Validate fairness rule:
    for each window_id, all conditions should share passenger_demand_generated.
    """
    by_window: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        window_id = str(row.get("window_id"))
        condition_id = normalize_condition_id(row.get("condition_id"))
        demand = row.get("passenger_demand_generated")

        if demand is None:
            continue

        demand_value = _as_float(demand, default=None)

        if demand_value is None:
            continue

        if window_id not in by_window:
            by_window[window_id] = {}

        by_window[window_id][condition_id] = demand_value

    violations = []

    for window_id, condition_values in by_window.items():
        rounded_values = {
            condition: round(value, 9)
            for condition, value in condition_values.items()
        }
        unique_values = sorted(set(rounded_values.values()))

        if len(unique_values) > 1:
            violations.append(
                {
                    "window_id": window_id,
                    "condition_demands": rounded_values,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "checked_window_count": len(by_window),
        "violation_count": len(violations),
        "violations": violations,
        "passed": len(violations) == 0,
    }


def sample_causal_a_row() -> Dict[str, Any]:
    """Return a valid sample row for causal A MAPPO policy."""
    return {
        "condition_id": "A",
        "seed": 1,
        "window_id": "sample_window_001",
        "state_ts": "2023-01-01T08:00:00+09:00",
        "service_date": "2023-01-01",
        "time_band": "peak",
        "evaluation_horizon_minutes": 30,
        "headway_mean_seconds": 600.0,
        "headway_std_seconds": 90.0,
        "headway_sample_count": 6,
        "bunching_event_count": 0,
        "headway_event_count": 5,
        "wait_total_passenger_seconds": 12000.0,
        "wait_passenger_count": 40,
        "ontime_event_count": 4,
        "schedulable_arrival_count": 5,
        "intervention_count": 1,
        "decision_step_count": 5,
        "energy_proxy_total": 360.0,
        "passenger_demand_generated": 40,
        "passenger_served_count": 39,
        "passenger_service_rate": None,
        "passenger_wait_p95_seconds": 540.0,
        "long_wait_passenger_count": 2,
        "energy_proxy": None,
        "energy_proxy_per_passenger": None,
        "active_bus_count": 10,
        "baseline_bus_count": 10,
        "fleet_reduction_ratio": None,
        "policy_source": "mappo_policy",
        "policy_checkpoint_path": "artifacts/experiment_A_v1/checkpoints/best.pt",
        "source_mode": "causal_A_mappo_policy_v1",
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": 5.0,
        "reward_total": 0.0,
        "reward_service": 0.0,
        "reward_avg_wait": 0.0,
        "reward_long_wait": 0.0,
        "reward_on_time": 0.0,
        "reward_bunching": 0.0,
        "reward_energy": 0.0,
        "reward_fleet": 0.0,
        "reward_constraint": 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--print-columns", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.print_contract:
        print(json.dumps(schema_contract(), ensure_ascii=False, indent=2))
        return 0

    if args.print_columns:
        for col in expected_columns(strict=True):
            print(col)
        return 0

    if args.self_test:
        row = sample_causal_a_row()
        augmented = augment_rollout_row(row)
        summary = validate_rollout_rows([augmented], strict=True)

        if not summary["passed"]:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1

        if not source_mode_allows_causal_claim(augmented["source_mode"]):
            print("[FAIL] sample causal source_mode was not accepted")
            return 1

        print("[OK] rollout_schema_v1 self-test passed")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
