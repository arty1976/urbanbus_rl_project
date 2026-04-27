from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional


MODEL_VERSION = "daegu_energy_proxy_v1"
ENERGY_PROXY_UNIT = "kwh_equivalent"

K_DIST_KWH_PER_M = 0.0012
K_ACC_KWH_PER_EVENT = 0.1800
K_IDLE_KWH_PER_SEC = 0.0080

DEFAULT_EPSILON = 1.0e-9


@dataclass(frozen=True)
class EnergyProxyConfig:
    """
    Daegu bus energy proxy configuration.

    This is a proxy model, not a direct physics-grade battery/fuel model.
    It is intended to be used first as a KPI-side diagnostic lens before
    being promoted into MAPPO reward shaping.
    """

    model_version: str = MODEL_VERSION
    energy_proxy_unit: str = ENERGY_PROXY_UNIT
    k_dist_kwh_per_m: float = K_DIST_KWH_PER_M
    k_acc_kwh_per_event: float = K_ACC_KWH_PER_EVENT
    k_idle_kwh_per_sec: float = K_IDLE_KWH_PER_SEC
    epsilon: float = DEFAULT_EPSILON


DEFAULT_CONFIG = EnergyProxyConfig()


class EnergyProxyError(ValueError):
    pass


def config_as_dict(config: EnergyProxyConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    return asdict(config)


def expected_contract_fields(config: EnergyProxyConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """
    Fields that must match MAPPO checkpoint contract v1.
    """
    return {
        "energy_proxy_model_version": config.model_version,
        "energy_proxy_unit": config.energy_proxy_unit,
        "k_dist_kwh_per_m": config.k_dist_kwh_per_m,
        "k_acc_kwh_per_event": config.k_acc_kwh_per_event,
        "k_idle_kwh_per_sec": config.k_idle_kwh_per_sec,
    }


def validate_config_constants(config: EnergyProxyConfig = DEFAULT_CONFIG) -> None:
    if config.model_version != MODEL_VERSION:
        raise EnergyProxyError(
            f"model_version mismatch: expected {MODEL_VERSION}, got {config.model_version}"
        )

    if config.energy_proxy_unit != ENERGY_PROXY_UNIT:
        raise EnergyProxyError(
            f"energy_proxy_unit mismatch: expected {ENERGY_PROXY_UNIT}, got {config.energy_proxy_unit}"
        )

    checks = {
        "k_dist_kwh_per_m": (config.k_dist_kwh_per_m, K_DIST_KWH_PER_M),
        "k_acc_kwh_per_event": (config.k_acc_kwh_per_event, K_ACC_KWH_PER_EVENT),
        "k_idle_kwh_per_sec": (config.k_idle_kwh_per_sec, K_IDLE_KWH_PER_SEC),
    }

    for name, (actual, expected) in checks.items():
        if abs(float(actual) - float(expected)) > 1.0e-12:
            raise EnergyProxyError(
                f"{name} mismatch: expected {expected}, got {actual}"
            )

    if float(config.epsilon) <= 0.0:
        raise EnergyProxyError(f"epsilon must be positive, got {config.epsilon}")


def _has_negative(value: Any) -> bool:
    try:
        import numpy as np

        arr = np.asarray(value, dtype=float)
        return bool((arr < 0).any())
    except Exception:
        pass

    try:
        return float(value) < 0.0
    except Exception:
        try:
            return any(float(x) < 0.0 for x in value)
        except Exception:
            raise EnergyProxyError(f"cannot inspect numeric value for negativity: {value!r}")


def _ensure_non_negative(name: str, value: Any) -> None:
    if _has_negative(value):
        raise EnergyProxyError(f"{name} must be non-negative")


def _safe_max_passenger_count(passenger_served_count: Any, epsilon: float) -> Any:
    try:
        import numpy as np

        arr = np.asarray(passenger_served_count, dtype=float)
        return np.maximum(arr, float(epsilon))
    except Exception:
        pass

    try:
        return max(float(passenger_served_count), float(epsilon))
    except Exception:
        return [max(float(x), float(epsilon)) for x in passenger_served_count]


def _safe_divide(numerator: Any, denominator: Any) -> Any:
    try:
        import numpy as np

        return np.asarray(numerator, dtype=float) / np.asarray(denominator, dtype=float)
    except Exception:
        pass

    try:
        return float(numerator) / float(denominator)
    except Exception:
        return [float(n) / float(d) for n, d in zip(numerator, denominator)]


def _to_float_if_scalar(value: Any) -> Any:
    try:
        import numpy as np

        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        return value
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return value


def compute_energy_kwh_equiv(
    *,
    distance_m: Any,
    acceleration_event_count: Any,
    hold_seconds: Any,
    config: EnergyProxyConfig = DEFAULT_CONFIG,
) -> Any:
    """
    Compute Daegu bus energy proxy.

    energy_kwh_equiv =
        K_DIST * distance_m
        + K_ACC * acceleration_event_count
        + K_IDLE * hold_seconds

    Inputs may be scalars or vector-like objects such as numpy arrays or pandas Series.
    """

    validate_config_constants(config)

    _ensure_non_negative("distance_m", distance_m)
    _ensure_non_negative("acceleration_event_count", acceleration_event_count)
    _ensure_non_negative("hold_seconds", hold_seconds)

    energy = (
        float(config.k_dist_kwh_per_m) * distance_m
        + float(config.k_acc_kwh_per_event) * acceleration_event_count
        + float(config.k_idle_kwh_per_sec) * hold_seconds
    )

    return _to_float_if_scalar(energy)


def compute_energy_proxy_per_passenger(
    *,
    energy_kwh_equiv: Any,
    passenger_served_count: Any,
    config: EnergyProxyConfig = DEFAULT_CONFIG,
) -> Any:
    """
    Compute energy proxy per passenger.

    energy_proxy_per_passenger =
        energy_kwh_equiv / max(passenger_served_count, epsilon)
    """

    validate_config_constants(config)

    _ensure_non_negative("energy_kwh_equiv", energy_kwh_equiv)
    _ensure_non_negative("passenger_served_count", passenger_served_count)

    denom = _safe_max_passenger_count(
        passenger_served_count,
        epsilon=float(config.epsilon),
    )

    return _to_float_if_scalar(_safe_divide(energy_kwh_equiv, denom))


def compute_energy_proxy_from_components(
    *,
    distance_m: Any,
    acceleration_event_count: Any,
    hold_seconds: Any,
    passenger_served_count: Any,
    config: EnergyProxyConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    energy = compute_energy_kwh_equiv(
        distance_m=distance_m,
        acceleration_event_count=acceleration_event_count,
        hold_seconds=hold_seconds,
        config=config,
    )

    per_passenger = compute_energy_proxy_per_passenger(
        energy_kwh_equiv=energy,
        passenger_served_count=passenger_served_count,
        config=config,
    )

    return {
        "energy_proxy_model_version": config.model_version,
        "energy_proxy_unit": config.energy_proxy_unit,
        "distance_m": distance_m,
        "acceleration_event_count": acceleration_event_count,
        "hold_seconds": hold_seconds,
        "passenger_served_count": passenger_served_count,
        "energy_kwh_equiv": energy,
        "energy_proxy_per_passenger": per_passenger,
        "k_dist_kwh_per_m": config.k_dist_kwh_per_m,
        "k_acc_kwh_per_event": config.k_acc_kwh_per_event,
        "k_idle_kwh_per_sec": config.k_idle_kwh_per_sec,
        "epsilon": config.epsilon,
    }


def compute_energy_proxy_for_record(
    record: Mapping[str, Any],
    *,
    config: EnergyProxyConfig = DEFAULT_CONFIG,
    distance_key: str = "distance_m",
    acceleration_key: str = "acceleration_event_count",
    hold_key: str = "hold_seconds",
    passenger_key: str = "passenger_served_count",
) -> Dict[str, Any]:
    missing = [
        key
        for key in (distance_key, acceleration_key, hold_key, passenger_key)
        if key not in record
    ]

    if missing:
        raise EnergyProxyError(f"record missing required keys: {missing}")

    return compute_energy_proxy_from_components(
        distance_m=record[distance_key],
        acceleration_event_count=record[acceleration_key],
        hold_seconds=record[hold_key],
        passenger_served_count=record[passenger_key],
        config=config,
    )


def compute_energy_proxy_for_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config: EnergyProxyConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    return [
        compute_energy_proxy_for_record(record, config=config)
        for record in records
    ]


def summarize_energy_proxy_records(
    computed_records: Iterable[Mapping[str, Any]],
) -> Dict[str, Optional[float]]:
    values = list(computed_records)

    if not values:
        return {
            "record_count": 0,
            "energy_kwh_equiv_total": None,
            "energy_kwh_equiv_mean": None,
            "energy_proxy_per_passenger_mean": None,
        }

    energy_values = [float(r["energy_kwh_equiv"]) for r in values]
    per_passenger_values = [float(r["energy_proxy_per_passenger"]) for r in values]

    return {
        "record_count": int(len(values)),
        "energy_kwh_equiv_total": float(sum(energy_values)),
        "energy_kwh_equiv_mean": float(sum(energy_values) / len(energy_values)),
        "energy_proxy_per_passenger_mean": float(
            sum(per_passenger_values) / len(per_passenger_values)
        ),
    }


if __name__ == "__main__":
    sample = compute_energy_proxy_from_components(
        distance_m=1000.0,
        acceleration_event_count=10.0,
        hold_seconds=60.0,
        passenger_served_count=30.0,
    )
    for key, value in sample.items():
        print(f"{key}: {value}")
