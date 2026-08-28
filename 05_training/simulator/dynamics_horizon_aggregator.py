from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Mapping, Optional, Sequence

from simulator.dynamics_event_trace import DynamicsEvent, DynamicsEventType, PassengerWaitEvent


class KPIStatus:
    NATIVE = "NATIVE"
    DERIVED_FROM_EVENT_LOG = "DERIVED_FROM_EVENT_LOG"
    REQUIRES_EXTERNAL_CANONICAL_AGGREGATOR = "REQUIRES_EXTERNAL_CANONICAL_AGGREGATOR"
    UNAVAILABLE_MISSING_REQUIRED_DATA = "UNAVAILABLE_MISSING_REQUIRED_DATA"
    FORMULA_NOT_SELECTED = "FORMULA_NOT_SELECTED"


@dataclass(frozen=True)
class KPIValue:
    value: Optional[float]
    status: str
    reason: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {"value": self.value, "status": self.status, "reason": self.reason}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_horizon_kpis(
    events: Sequence[DynamicsEvent],
    passenger_wait_events: Sequence[PassengerWaitEvent],
    *,
    passenger_demand_count: Optional[int] = None,
    headway_values: Optional[Sequence[float]] = None,
    travel_seconds: Optional[float] = None,
    dwell_seconds: Optional[float] = None,
    load_time_integral: Optional[float] = None,
    distance_proxy_inputs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    served_count = sum(1 for event in events if event.event_type in {DynamicsEventType.PASSENGER_BOARD, DynamicsEventType.PASSENGER_ALIGHT})
    missed_pickup = sum(1 for event in events if event.event_type == DynamicsEventType.PASSENGER_MISSED_PICKUP)
    missed_dropoff = sum(1 for event in events if event.event_type == DynamicsEventType.PASSENGER_MISSED_DROPOFF)
    mandatory = sum(1 for event in events if event.event_type == DynamicsEventType.MANDATORY_STOP_VIOLATION)
    invalid_skip = sum(1 for event in events if event.event_type == DynamicsEventType.INVALID_SKIP)
    waits = [float(event.wait_seconds) for event in passenger_wait_events]
    if waits:
        avg_wait = KPIValue(float(mean(waits)), KPIStatus.DERIVED_FROM_EVENT_LOG)
        p95_wait = KPIValue(float(_percentile(waits, 0.95)), KPIStatus.DERIVED_FROM_EVENT_LOG)
    else:
        avg_wait = KPIValue(None, KPIStatus.UNAVAILABLE_MISSING_REQUIRED_DATA, "PASSENGER_LEVEL_WAIT_DISTRIBUTION_UNAVAILABLE")
        p95_wait = KPIValue(None, KPIStatus.UNAVAILABLE_MISSING_REQUIRED_DATA, "PASSENGER_LEVEL_WAIT_DISTRIBUTION_UNAVAILABLE")
    if passenger_demand_count is None or passenger_demand_count <= 0:
        service_rate = KPIValue(None, KPIStatus.UNAVAILABLE_MISSING_REQUIRED_DATA, "PASSENGER_DEMAND_COUNT_UNAVAILABLE")
    else:
        service_rate = KPIValue(float(served_count / passenger_demand_count), KPIStatus.DERIVED_FROM_EVENT_LOG)
    headway = KPIValue(float(mean(headway_values)), KPIStatus.DERIVED_FROM_EVENT_LOG) if headway_values else KPIValue(None, KPIStatus.REQUIRES_EXTERNAL_CANONICAL_AGGREGATOR, "HEADWAY_SERIES_UNAVAILABLE")
    return {
        "served_count": KPIValue(float(served_count), KPIStatus.NATIVE).to_payload(),
        "missed_pickup_count": KPIValue(float(missed_pickup), KPIStatus.NATIVE).to_payload(),
        "missed_dropoff_count": KPIValue(float(missed_dropoff), KPIStatus.NATIVE).to_payload(),
        "mandatory_stop_violation_count": KPIValue(float(mandatory), KPIStatus.NATIVE).to_payload(),
        "invalid_skip_execution_count": KPIValue(float(invalid_skip), KPIStatus.NATIVE).to_payload(),
        "avg_wait_seconds": avg_wait.to_payload(),
        "p95_wait_seconds": p95_wait.to_payload(),
        "passenger_service_rate": service_rate.to_payload(),
        "headway": headway.to_payload(),
        "cv_headway": KPIValue(None, KPIStatus.REQUIRES_EXTERNAL_CANONICAL_AGGREGATOR, "CV_HEADWAY_REQUIRES_CANONICAL_HEADWAY_AGGREGATOR").to_payload(),
        "bunching_rate": KPIValue(None, KPIStatus.REQUIRES_EXTERNAL_CANONICAL_AGGREGATOR, "BUNCHING_RATE_REQUIRES_CANONICAL_HEADWAY_AGGREGATOR").to_payload(),
        "on_time_rate": KPIValue(None, KPIStatus.REQUIRES_EXTERNAL_CANONICAL_AGGREGATOR, "ON_TIME_RATE_REQUIRES_CANONICAL_SCHEDULE_AGGREGATOR").to_payload(),
        "travel_seconds": KPIValue(float(travel_seconds), KPIStatus.DERIVED_FROM_EVENT_LOG).to_payload() if travel_seconds is not None else KPIValue(None, KPIStatus.UNAVAILABLE_MISSING_REQUIRED_DATA, "TRAVEL_SECONDS_UNAVAILABLE").to_payload(),
        "dwell_seconds": KPIValue(float(dwell_seconds), KPIStatus.DERIVED_FROM_EVENT_LOG).to_payload() if dwell_seconds is not None else KPIValue(None, KPIStatus.UNAVAILABLE_MISSING_REQUIRED_DATA, "DWELL_SECONDS_UNAVAILABLE").to_payload(),
        "distance_proxy_inputs": {"value": distance_proxy_inputs, "status": KPIStatus.DERIVED_FROM_EVENT_LOG if distance_proxy_inputs else KPIStatus.UNAVAILABLE_MISSING_REQUIRED_DATA, "reason": None if distance_proxy_inputs else "DISTANCE_INPUTS_UNAVAILABLE"},
        "load_time_integral": KPIValue(float(load_time_integral), KPIStatus.DERIVED_FROM_EVENT_LOG).to_payload() if load_time_integral is not None else KPIValue(None, KPIStatus.UNAVAILABLE_MISSING_REQUIRED_DATA, "LOAD_TIME_INTEGRAL_UNAVAILABLE").to_payload(),
        "energy_metric_status": KPIStatus.FORMULA_NOT_SELECTED,
        "new_energy_formula_created": False,
        "reward_status": "NOT_DEFINED_IN_ER1_IMPLEMENT",
        "new_reward_formula_created": False,
    }
