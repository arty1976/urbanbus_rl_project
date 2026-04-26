"""
mappo_reward_v1.py
==================

Reward contract for UrbanBus MAPPO.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화)
reward must prioritize passenger service quality first, and energy/fleet
efficiency second.

Important:
- A / A90 / A80 / A70 actual performance claims must come from causal rollout.
- Placeholder or replay outputs must remain separated by policy_source/source_mode.
- Qwen must not intervene in A_pure_mappo.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "artifact_version": "mappo_reward_v1",
    "weights": {
        "service_rate_weight": 3.0,
        "avg_wait_weight": 2.0,
        "long_wait_weight": 3.0,
        "on_time_weight": 1.0,
        "bunching_weight": 1.5,
        "energy_per_passenger_weight": 1.0,
        "fleet_reduction_weight": 0.5,
        "constraint_violation_weight": 10.0,
    },
    "constraints": {
        "min_service_rate": 0.95,
        "max_avg_wait_ratio_vs_baseline": 1.10,
        "max_p95_wait_ratio_vs_baseline": 1.15,
        "qwen_trigger_rate_required_for_A": 0.0,
    },
    "baselines": {
        "avg_wait_seconds": 300.0,
        "passenger_wait_p95_seconds": 600.0,
        "energy_proxy_per_passenger": 10.0,
    },
}


def merged_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge user config into DEFAULT_CONFIG one level deep."""
    out = deepcopy(DEFAULT_CONFIG)

    if not config:
        return out

    for section, value in config.items():
        if isinstance(value, Mapping) and isinstance(out.get(section), dict):
            out[section].update(value)
        else:
            out[section] = value

    return out


def _float(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)

    if value is None:
        return float(default)

    try:
        return float(value)
    except Exception:
        return float(default)


def _cfg_float(
    config: Mapping[str, Any],
    section: str,
    key: str,
    default: float,
) -> float:
    try:
        return float(config.get(section, {}).get(key, default))
    except Exception:
        return float(default)


def _safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if denominator <= 0:
        return float(default)
    return float(numerator / denominator)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def compute_constraint_flags(
    metrics: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, bool]:
    """
    Compute hard constraint violation flags.

    These flags are used to prevent fake improvements such as:
    - reducing buses while leaving passengers unserved
    - reducing energy by suppressing service
    - allowing Qwen intervention in A_pure_mappo
    """
    cfg = merged_config(config)

    min_service_rate = _cfg_float(
        cfg,
        "constraints",
        "min_service_rate",
        0.95,
    )
    max_avg_wait_ratio = _cfg_float(
        cfg,
        "constraints",
        "max_avg_wait_ratio_vs_baseline",
        1.10,
    )
    max_p95_wait_ratio = _cfg_float(
        cfg,
        "constraints",
        "max_p95_wait_ratio_vs_baseline",
        1.15,
    )
    qwen_required_for_a = _cfg_float(
        cfg,
        "constraints",
        "qwen_trigger_rate_required_for_A",
        0.0,
    )

    baseline_avg_wait = _cfg_float(
        cfg,
        "baselines",
        "avg_wait_seconds",
        300.0,
    )
    baseline_p95_wait = _cfg_float(
        cfg,
        "baselines",
        "passenger_wait_p95_seconds",
        600.0,
    )

    condition_id = str(metrics.get("condition_id", "")).upper()

    service_rate = _float(
        metrics,
        "passenger_service_rate",
        0.0,
    )
    avg_wait = _float(
        metrics,
        "avg_wait_seconds",
        baseline_avg_wait,
    )
    p95_wait = _float(
        metrics,
        "passenger_wait_p95_seconds",
        baseline_p95_wait,
    )
    qwen_trigger_rate = _float(
        metrics,
        "qwen_trigger_rate",
        0.0,
    )

    avg_wait_ratio = _safe_ratio(
        avg_wait,
        baseline_avg_wait,
        default=1.0,
    )
    p95_wait_ratio = _safe_ratio(
        p95_wait,
        baseline_p95_wait,
        default=1.0,
    )

    return {
        "low_service_rate": bool(service_rate < min_service_rate),
        "avg_wait_exceeds_baseline_ratio": bool(avg_wait_ratio > max_avg_wait_ratio),
        "p95_wait_exceeds_baseline_ratio": bool(p95_wait_ratio > max_p95_wait_ratio),
        "qwen_trigger_rate_violation_for_A": bool(
            condition_id == "A"
            and abs(qwen_trigger_rate - qwen_required_for_a) > 1e-12
        ),
    }


def compute_reward_components(
    metrics: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute reward components from KPI metrics.

    Required or recommended input metrics:
    - condition_id
    - avg_wait_seconds
    - passenger_wait_p95_seconds
    - passenger_service_rate
    - energy_proxy
    - energy_proxy_per_passenger
    - on_time_rate
    - bunching_rate
    - fleet_reduction_ratio
    - qwen_trigger_rate

    Output:
    - reward_total
    - reward_service
    - reward_avg_wait
    - reward_long_wait
    - reward_on_time
    - reward_bunching
    - reward_energy
    - reward_fleet
    - reward_constraint
    - constraint_violation_flags
    - reward_debug
    """
    cfg = merged_config(config)

    w_service = _cfg_float(
        cfg,
        "weights",
        "service_rate_weight",
        3.0,
    )
    w_avg_wait = _cfg_float(
        cfg,
        "weights",
        "avg_wait_weight",
        2.0,
    )
    w_long_wait = _cfg_float(
        cfg,
        "weights",
        "long_wait_weight",
        3.0,
    )
    w_on_time = _cfg_float(
        cfg,
        "weights",
        "on_time_weight",
        1.0,
    )
    w_bunching = _cfg_float(
        cfg,
        "weights",
        "bunching_weight",
        1.5,
    )
    w_energy = _cfg_float(
        cfg,
        "weights",
        "energy_per_passenger_weight",
        1.0,
    )
    w_fleet = _cfg_float(
        cfg,
        "weights",
        "fleet_reduction_weight",
        0.5,
    )
    w_constraint = _cfg_float(
        cfg,
        "weights",
        "constraint_violation_weight",
        10.0,
    )

    baseline_avg_wait = _cfg_float(
        cfg,
        "baselines",
        "avg_wait_seconds",
        300.0,
    )
    baseline_p95_wait = _cfg_float(
        cfg,
        "baselines",
        "passenger_wait_p95_seconds",
        600.0,
    )
    baseline_energy_pp = _cfg_float(
        cfg,
        "baselines",
        "energy_proxy_per_passenger",
        10.0,
    )

    service_rate = _float(
        metrics,
        "passenger_service_rate",
        0.0,
    )
    avg_wait = _float(
        metrics,
        "avg_wait_seconds",
        baseline_avg_wait,
    )
    p95_wait = _float(
        metrics,
        "passenger_wait_p95_seconds",
        baseline_p95_wait,
    )
    on_time_rate = _float(
        metrics,
        "on_time_rate",
        0.0,
    )
    bunching_rate = _float(
        metrics,
        "bunching_rate",
        0.0,
    )
    energy_pp = _float(
        metrics,
        "energy_proxy_per_passenger",
        baseline_energy_pp,
    )
    fleet_reduction_ratio = _float(
        metrics,
        "fleet_reduction_ratio",
        0.0,
    )

    service_rate_score = _clamp(
        service_rate,
        0.0,
        1.0,
    )
    avg_wait_penalty = max(
        0.0,
        _safe_ratio(avg_wait, baseline_avg_wait, default=1.0) - 1.0,
    )
    long_wait_penalty = max(
        0.0,
        _safe_ratio(p95_wait, baseline_p95_wait, default=1.0) - 1.0,
    )
    on_time_score = _clamp(
        on_time_rate,
        0.0,
        1.0,
    )
    bunching_penalty = _clamp(
        bunching_rate,
        0.0,
        1.0,
    )
    energy_pp_ratio = max(
        0.0,
        _safe_ratio(energy_pp, baseline_energy_pp, default=1.0),
    )
    fleet_score = _clamp(
        fleet_reduction_ratio,
        0.0,
        1.0,
    )

    flags = compute_constraint_flags(
        metrics,
        cfg,
    )
    violation_count = sum(
        1
        for value in flags.values()
        if value
    )

    reward_service = +w_service * service_rate_score
    reward_avg_wait = -w_avg_wait * avg_wait_penalty
    reward_long_wait = -w_long_wait * long_wait_penalty
    reward_on_time = +w_on_time * on_time_score
    reward_bunching = -w_bunching * bunching_penalty
    reward_energy = -w_energy * energy_pp_ratio
    reward_fleet = +w_fleet * fleet_score
    reward_constraint = -w_constraint * float(violation_count)

    reward_total = (
        reward_service
        + reward_avg_wait
        + reward_long_wait
        + reward_on_time
        + reward_bunching
        + reward_energy
        + reward_fleet
        + reward_constraint
    )

    return {
        "reward_total": float(reward_total),
        "reward_service": float(reward_service),
        "reward_avg_wait": float(reward_avg_wait),
        "reward_long_wait": float(reward_long_wait),
        "reward_on_time": float(reward_on_time),
        "reward_bunching": float(reward_bunching),
        "reward_energy": float(reward_energy),
        "reward_fleet": float(reward_fleet),
        "reward_constraint": float(reward_constraint),
        "constraint_violation_flags": flags,
        "reward_debug": {
            "service_rate_score": float(service_rate_score),
            "avg_wait_penalty": float(avg_wait_penalty),
            "long_wait_penalty": float(long_wait_penalty),
            "on_time_score": float(on_time_score),
            "bunching_penalty": float(bunching_penalty),
            "energy_proxy_per_passenger_ratio": float(energy_pp_ratio),
            "fleet_reduction_score": float(fleet_score),
            "constraint_violation_count": int(violation_count),
            "baseline_avg_wait_seconds": float(baseline_avg_wait),
            "baseline_p95_wait_seconds": float(baseline_p95_wait),
            "baseline_energy_proxy_per_passenger": float(baseline_energy_pp),
        },
    }


def compute_total_reward(
    metrics: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Alias for compute_reward_components."""
    return compute_reward_components(
        metrics=metrics,
        config=config,
    )


def reward_keys() -> Dict[str, Any]:
    """Return reward input/output key contract."""
    return {
        "required_or_recommended_input_metrics": [
            "condition_id",
            "avg_wait_seconds",
            "passenger_wait_p95_seconds",
            "passenger_service_rate",
            "energy_proxy",
            "energy_proxy_per_passenger",
            "on_time_rate",
            "bunching_rate",
            "fleet_reduction_ratio",
            "qwen_trigger_rate",
        ],
        "output_reward_columns": [
            "reward_total",
            "reward_service",
            "reward_avg_wait",
            "reward_long_wait",
            "reward_on_time",
            "reward_bunching",
            "reward_energy",
            "reward_fleet",
            "reward_constraint",
            "constraint_violation_flags",
        ],
    }


if __name__ == "__main__":
    sample = {
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

    result = compute_total_reward(sample)

    for key, value in result.items():
        print(f"{key}: {value}")
