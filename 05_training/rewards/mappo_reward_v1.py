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
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_CONFIG: Dict[str, Any] = {
    "artifact_version": "mappo_reward_v1",
    "reward_mode": "LEGACY_LINEAGE_REPLAY",
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

PV8_REWARD_SEMANTICS_VERSION = "PV8_REWARD_SEMANTICS_V2"
PV8_REWARD_V2_MODE = "PV8_REWARD_V2"
PV8_REWARD_V2_FREEZE_SHA256 = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
PV8_REWARD_V2_TRAINING_REFERENCE_VERSION = "PV8_REPRESENTATIVE_B1_TRAINING_REFERENCE_FROZEN_V1"
PV8_REWARD_V2_FORMULA = (
    "+ 3.0 * local_service_component "
    "+ 2.0 * local_affected_avg_wait_component "
    "- 0.25 * explicit_forced_external_intervention"
)
PV8_REWARD_V2_SERVICE_REFERENCE = 1.0
PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS = 297.7850241545894
PV8_REWARD_V2_P95_EVALUATION_REFERENCE_SECONDS = 576.6999999999999
PV8_REWARD_V2_NUMERIC_TOLERANCE = 1e-12


class RewardV2BindingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RewardV2DuplicateOwnershipError(RewardV2BindingError):
    pass


class RewardV2TransitionBindingError(RewardV2BindingError):
    pass


def validate_reward_v2_freeze_hash(reward_freeze_sha256: Any) -> None:
    if str(reward_freeze_sha256) != PV8_REWARD_V2_FREEZE_SHA256:
        raise RewardV2BindingError(
            "RUNTIME_REWARD_FREEZE_HASH_MISMATCH",
            "Reward V2 runtime result must carry the immutable H4F freeze SHA.",
        )


def _finite_float(value: Any, key: str, default: float = 0.0) -> float:
    if value is None:
        value = default
    try:
        out = float(value)
    except Exception as exc:
        raise RewardV2BindingError("RUNTIME_REWARD_NON_NUMERIC_INPUT", f"{key} must be numeric") from exc
    if not isfinite(out):
        raise RewardV2BindingError("RUNTIME_REWARD_NON_FINITE_INPUT", f"{key} must be finite")
    return out


def _nonnegative_int(metrics: Mapping[str, Any], key: str) -> int:
    value = int(_finite_float(metrics.get(key, 0), key, default=0.0))
    if value < 0:
        raise RewardV2BindingError("RUNTIME_REWARD_NEGATIVE_COUNT", f"{key} must be non-negative")
    return value


def _transition_identity(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    required = [
        "transition_id",
        "vehicle_slot_id",
        "route_id",
        "direction_id",
        "occurrence_id",
        "local_decision_ts",
        "action",
    ]
    missing = [key for key in required if metrics.get(key) in (None, "")]
    if missing:
        raise RewardV2TransitionBindingError(
            "RUNTIME_REWARD_TRANSITION_IDENTITY_MISSING",
            f"Reward V2 transition identity is incomplete: {missing}",
        )
    return {key: metrics[key] for key in required}


def validate_reward_v2_transition_binding(
    metrics: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> None:
    observed = _transition_identity(metrics)
    compare_keys = [
        "transition_id",
        "vehicle_slot_id",
        "route_id",
        "direction_id",
        "occurrence_id",
        "local_decision_ts",
        "action",
    ]
    for key in compare_keys:
        if str(observed.get(key)) != str(expected_identity.get(key)):
            code = {
                "transition_id": "WRONG_TRANSITION_OWNERSHIP_REJECTED",
                "vehicle_slot_id": "WRONG_VEHICLE_OWNERSHIP_REJECTED",
                "occurrence_id": "WRONG_OCCURRENCE_OWNERSHIP_REJECTED",
            }.get(key, "RUNTIME_REWARD_TRANSITION_BINDING_REJECTED")
            raise RewardV2TransitionBindingError(
                code,
                f"Reward V2 transition binding mismatch for {key}: {observed.get(key)!r} != {expected_identity.get(key)!r}",
            )


def compute_local_service_component(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    pickup = _nonnegative_int(metrics, "pickup_obligation_count")
    dropoff = _nonnegative_int(metrics, "dropoff_obligation_count")
    mandatory = _nonnegative_int(metrics, "approved_static_mandatory_obligation_count")
    completed_pickup = _nonnegative_int(metrics, "completed_pickup_obligation_count")
    completed_dropoff = _nonnegative_int(metrics, "completed_dropoff_obligation_count")
    completed_mandatory = _nonnegative_int(metrics, "completed_static_mandatory_obligation_count")
    required = pickup + dropoff + mandatory
    completed = completed_pickup + completed_dropoff + completed_mandatory
    if required == 0:
        return {
            "status": "NOT_APPLICABLE",
            "raw": 0.0,
            "weighted": 0.0,
            "required_obligation_count": 0,
            "completed_obligation_count": completed,
            "service_success_denominator_increment": 0,
            "positive_service_reward": 0.0,
        }
    if completed > required:
        raise RewardV2BindingError(
            "RUNTIME_REWARD_SERVICE_COMPLETION_EXCEEDS_REQUIRED",
            "Completed local obligations cannot exceed required obligations.",
        )
    raw = 1.0 if completed == required else 0.0
    return {
        "status": "SUCCESS" if raw == 1.0 else "INCOMPLETE",
        "raw": raw,
        "weighted": 3.0 * raw,
        "required_obligation_count": required,
        "completed_obligation_count": completed,
        "service_success_denominator_increment": 1,
        "positive_service_reward": 3.0 * raw,
    }


def _normalize_wait_rows(rows: Sequence[Mapping[str, Any]], transition_id: str) -> List[Dict[str, Any]]:
    seen_passengers: set[str] = set()
    seen_ownership_keys: set[str] = set()
    out: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        passenger_id = str(row.get("passenger_id") or "")
        if not passenger_id:
            raise RewardV2DuplicateOwnershipError("WAIT_OWNER_IDENTITY_MISSING", "affected wait row lacks passenger_id")
        if passenger_id in seen_passengers:
            raise RewardV2DuplicateOwnershipError(
                "DUPLICATE_WAIT_REWARD_OWNERSHIP_REJECTED",
                f"passenger wait is duplicated: {passenger_id}",
            )
        seen_passengers.add(passenger_id)
        owner = str(row.get("originating_transition_id") or transition_id)
        if owner != transition_id:
            raise RewardV2TransitionBindingError(
                "WAIT_REWARD_WRONG_TRANSITION_REJECTED",
                f"wait row belongs to {owner}, not {transition_id}",
            )
        ownership_key = str(row.get("wait_ownership_key") or f"{owner}:{passenger_id}")
        if ownership_key in seen_ownership_keys:
            raise RewardV2DuplicateOwnershipError(
                "DUPLICATE_WAIT_REWARD_OWNERSHIP_REJECTED",
                f"wait ownership key is duplicated: {ownership_key}",
            )
        seen_ownership_keys.add(ownership_key)
        request_ts = _finite_float(row.get("request_ts"), f"affected_wait_rows[{index}].request_ts")
        first_eligible = _finite_float(row.get("first_eligible_service_ts"), f"affected_wait_rows[{index}].first_eligible_service_ts")
        actual_board = _finite_float(row.get("actual_board_ts"), f"affected_wait_rows[{index}].actual_board_ts")
        local_decision_ts = _finite_float(row.get("local_decision_ts"), f"affected_wait_rows[{index}].local_decision_ts")
        if request_ts > local_decision_ts:
            raise RewardV2TransitionBindingError(
                "FUTURE_REQUEST_IN_REWARD_SET_REJECTED",
                "affected wait set cannot include future passenger requests",
            )
        if first_eligible < request_ts or actual_board < first_eligible:
            raise RewardV2BindingError("WAIT_DECOMPOSITION_INVALID", "wait timestamps violate frozen decomposition")
        schedule_wait = first_eligible - request_ts
        alignment_excess = actual_board - first_eligible
        total_wait = schedule_wait + alignment_excess
        out.append(
            {
                "passenger_id": passenger_id,
                "originating_transition_id": owner,
                "wait_ownership_key": ownership_key,
                "request_ts": request_ts,
                "local_decision_ts": local_decision_ts,
                "first_eligible_service_ts": first_eligible,
                "actual_board_ts": actual_board,
                "schedule_wait_seconds": schedule_wait,
                "alignment_excess_wait_seconds": alignment_excess,
                "total_wait_seconds": total_wait,
            }
        )
    return out


def compute_local_avg_wait_component(
    metrics: Mapping[str, Any],
    *,
    transition_id: str,
    local_service_raw: float,
) -> Dict[str, Any]:
    rows = _normalize_wait_rows(metrics.get("affected_wait_rows") or (), transition_id)
    if not rows:
        return {
            "status": "NOT_APPLICABLE",
            "raw": 0.0,
            "weighted": 0.0,
            "affected_wait_count": 0,
            "current_local_avg_wait_seconds": None,
            "positive_avg_wait_reward_blocked_by_service_gate": False,
            "duplicate_wait_ownership": 0,
        }
    waits = [float(row["total_wait_seconds"]) for row in rows]
    avg_wait = float(sum(waits) / len(waits))
    centered = 1.0 - avg_wait / PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS
    blocked = centered > 0.0 and float(local_service_raw) < 1.0
    raw = 0.0 if blocked else centered
    return {
        "status": "BOUND" if not blocked else "POSITIVE_IMPROVEMENT_BLOCKED_BY_SERVICE_GATE",
        "raw": float(raw),
        "weighted": float(2.0 * raw),
        "affected_wait_count": len(rows),
        "current_local_avg_wait_seconds": avg_wait,
        "affected_wait_rows": rows,
        "positive_avg_wait_reward_blocked_by_service_gate": blocked,
        "duplicate_wait_ownership": 0,
    }


def compute_intervention_component(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = _nonnegative_int(metrics, "explicit_forced_external_intervention_count")
    forced = _nonnegative_int(metrics, "forced_safety_override_count")
    external = _nonnegative_int(metrics, "external_policy_intervention_count")
    count = explicit if explicit else forced + external
    if bool(metrics.get("ordinary_k_mask_restriction_counted", False)):
        raise RewardV2BindingError(
            "ORDINARY_K_MASK_RESTRICTION_NOT_INTERVENTION",
            "ordinary K-mask availability restriction must not trigger intervention penalty",
        )
    return {
        "status": "BOUND",
        "raw": float(count),
        "weighted_penalty_magnitude": float(0.25 * count),
        "explicit_forced_external_intervention_count": int(count),
        "ordinary_k_mask_restriction_counted": False,
    }


def compute_reward_v2(
    metrics: Mapping[str, Any],
    *,
    expected_freeze_sha256: str = PV8_REWARD_V2_FREEZE_SHA256,
) -> Dict[str, Any]:
    if expected_freeze_sha256 != PV8_REWARD_V2_FREEZE_SHA256:
        raise RewardV2BindingError("RUNTIME_REWARD_EXPECTED_FREEZE_HASH_INVALID", "expected freeze hash is not H4F")
    validate_reward_v2_freeze_hash(metrics.get("reward_freeze_sha256"))
    if str(metrics.get("reward_semantics_version")) != PV8_REWARD_SEMANTICS_VERSION:
        raise RewardV2BindingError(
            "RUNTIME_REWARD_SEMANTICS_VERSION_MISMATCH",
            "Reward V2 runtime result must report PV8_REWARD_SEMANTICS_V2.",
        )
    identity = _transition_identity(metrics)
    if bool(metrics.get("p95_training_reward_enabled", False)) or bool(metrics.get("p95_training_normalization_active", False)):
        raise RewardV2BindingError("P95_TRAINING_REWARD_FORBIDDEN", "p95 is evaluation-only in Reward V2")
    service = compute_local_service_component(metrics)
    avg_wait = compute_local_avg_wait_component(
        metrics,
        transition_id=str(identity["transition_id"]),
        local_service_raw=float(service["raw"]),
    )
    intervention = compute_intervention_component(metrics)
    reward_total = float(service["weighted"] + avg_wait["weighted"] - intervention["weighted_penalty_magnitude"])
    if not isfinite(reward_total):
        raise RewardV2BindingError("RUNTIME_REWARD_NON_FINITE_TOTAL", "Reward V2 total must be finite")
    return {
        "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
        "reward_mode": PV8_REWARD_V2_MODE,
        "reward_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
        "reward_formula": PV8_REWARD_V2_FORMULA,
        "training_reference_version": PV8_REWARD_V2_TRAINING_REFERENCE_VERSION,
        "service_reference": PV8_REWARD_V2_SERVICE_REFERENCE,
        "avg_wait_reference_seconds": PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS,
        "p95_evaluation_reference_seconds": PV8_REWARD_V2_P95_EVALUATION_REFERENCE_SECONDS,
        "p95_semantics": "P95_EVALUATION_ONLY",
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
        "p95_local_transition_owner": None,
        "old_p95_weight_status": "INACTIVE_NOT_REDISTRIBUTED",
        "missed_eligible_service_role": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "alignment_excess_wait_role": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "transition_identity": identity,
        "transition_id": identity["transition_id"],
        "reward_service_component": service,
        "reward_avg_wait_component": avg_wait,
        "reward_intervention_component": intervention,
        "reward_service_component_raw": float(service["raw"]),
        "reward_service_component_weighted": float(service["weighted"]),
        "reward_avg_wait_component_raw": float(avg_wait["raw"]),
        "reward_avg_wait_component_weighted": float(avg_wait["weighted"]),
        "reward_intervention_component_raw": float(intervention["raw"]),
        "reward_intervention_component_weighted": float(-intervention["weighted_penalty_magnitude"]),
        "reward_total": reward_total,
        "reward_value_materialized": True,
        "representative_reward_rematerialization": False,
        "blanket_SKIP_penalty_applied": False,
        "direct_time_band_reward_term_applied": False,
        "H240_active_reward_settlement": False,
        "H660_active_reward_settlement": False,
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
    requested_mode = str(
        metrics.get("reward_mode")
        or metrics.get("reward_semantics_version")
        or (config or {}).get("reward_mode", "")
    )
    if requested_mode in {PV8_REWARD_V2_MODE, PV8_REWARD_SEMANTICS_VERSION}:
        return compute_reward_v2(
            metrics,
            expected_freeze_sha256=str((config or {}).get("reward_freeze_sha256", PV8_REWARD_V2_FREEZE_SHA256)),
        )

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
        "active_reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
        "pv8_reward_v2": {
            "mode": PV8_REWARD_V2_MODE,
            "freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
            "formula": PV8_REWARD_V2_FORMULA,
            "required_transition_identity": [
                "transition_id",
                "vehicle_slot_id",
                "route_id",
                "direction_id",
                "occurrence_id",
                "local_decision_ts",
                "action",
            ],
            "output_reward_columns": [
                "reward_total",
                "reward_service_component_raw",
                "reward_service_component_weighted",
                "reward_avg_wait_component_raw",
                "reward_avg_wait_component_weighted",
                "reward_intervention_component_raw",
                "reward_intervention_component_weighted",
                "reward_semantics_version",
                "reward_freeze_sha256",
                "transition_id",
            ],
        },
        "legacy_reward_mode": "LEGACY_LINEAGE_REPLAY",
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
