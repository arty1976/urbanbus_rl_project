from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from simulator.k_safety_state import ServiceObligationStateMachine, StaticGuardStatus
from simulator.suseong_service_transition_engine import build_distinct_three_action_mask


class KActionMaskRuntimeError(ValueError):
    pass


class RuntimeRuleHashError(KActionMaskRuntimeError):
    pass


class RuntimeRuleContractError(KActionMaskRuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeVersionBinding:
    k_safety_state_version: str
    static_rulebook_version: str
    static_rulebook_sha256: str
    occurrence_master_sha256: str
    dynamic_state_contract_version: str
    mask_predicate_version: str
    experiment_version: str

    def to_payload(self) -> Dict[str, str]:
        return {
            "k_safety_state_version": self.k_safety_state_version,
            "static_rulebook_version": self.static_rulebook_version,
            "static_rulebook_sha256": self.static_rulebook_sha256,
            "occurrence_master_sha256": self.occurrence_master_sha256,
            "dynamic_state_contract_version": self.dynamic_state_contract_version,
            "mask_predicate_version": self.mask_predicate_version,
            "experiment_version": self.experiment_version,
        }


@dataclass
class _RuntimeVehicle:
    agent_id: int
    vehicle_token: str
    route_key: Tuple[str, str]
    position: int = 0
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if text.lower() == "nan" or not text:
        return None
    return text


class FixedVehicleOccurrenceMaskRuntime:
    """Occurrence-exact K-mask adapter for an externally approved research contract.

    This adapter only materializes action masks. It does not select or execute an
    action and it does not authorize policy, checkpoint, evaluation, or training use.
    """

    REQUIRED_BOOLEAN_FIELDS = (
        "mandatory_stop",
        "protected_stop",
        "planned_itinerary_allows_skip",
        "terminal_or_turnaround_stop",
        "charging_or_driver_relief_stop",
        "valid_post_skip_path",
    )

    def __init__(
        self,
        *,
        rule_rows: Sequence[Mapping[str, Any]],
        fixed_vehicle_bindings: Mapping[int, str],
        approval_record: Mapping[str, Any],
        version_binding: RuntimeVersionBinding,
        actual_rulebook_sha256: str,
        actual_occurrence_master_sha256: str,
    ) -> None:
        if actual_rulebook_sha256 != version_binding.static_rulebook_sha256:
            raise RuntimeRuleHashError("static rulebook SHA-256 does not match runtime version binding")
        if actual_occurrence_master_sha256 != version_binding.occurrence_master_sha256:
            raise RuntimeRuleHashError("occurrence master SHA-256 does not match runtime version binding")
        if not bool(approval_record.get("approval_valid")) or not bool(approval_record.get("research_rule_approved")):
            raise RuntimeRuleContractError("explicit research-rule approval is required")
        if bool(approval_record.get("real_network_rule_claim_allowed")):
            raise RuntimeRuleContractError("research-rule approval cannot authorize a real-network rule claim")
        if str(approval_record.get("static_rulebook_sha256")) != version_binding.static_rulebook_sha256:
            raise RuntimeRuleHashError("approval record rulebook hash does not match runtime binding")
        if str(approval_record.get("occurrence_master_sha256")) != version_binding.occurrence_master_sha256:
            raise RuntimeRuleHashError("approval record occurrence hash does not match runtime binding")

        self.version_binding = version_binding
        self.fixed_vehicle_bindings = {int(agent): str(token) for agent, token in fixed_vehicle_bindings.items()}
        self.rules_by_occurrence_id: Dict[str, Dict[str, Any]] = {}
        self.rules_by_exact_key: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
        for raw in rule_rows:
            row = dict(raw)
            occurrence_id = str(row.get("route_stop_occurrence_id") or "")
            if not occurrence_id or occurrence_id in self.rules_by_occurrence_id:
                raise RuntimeRuleContractError("rulebook occurrence IDs must be present and unique")
            if str(row.get("rule_version")) != version_binding.static_rulebook_version:
                raise RuntimeRuleContractError("rule row version does not match runtime version binding")
            if str(row.get("rule_class")) != "CONTRACT_FIXED_RESEARCH_RULE":
                raise RuntimeRuleContractError("runtime rule rows must be research-contract rows")
            if not bool(row.get("static_rule_complete")):
                raise RuntimeRuleContractError("runtime rule rows must be complete")
            for field in self.REQUIRED_BOOLEAN_FIELDS:
                if not isinstance(row.get(field), bool):
                    raise RuntimeRuleContractError(f"{field} must be an explicit boolean")
            exact_key = (
                str(row["route_id"]),
                str(row["direction_id"]),
                int(row["stop_sequence"]),
                str(row["stop_id"]),
            )
            if exact_key in self.rules_by_exact_key:
                raise RuntimeRuleContractError("route-direction-sequence-stop occurrence key must be unique")
            self.rules_by_occurrence_id[occurrence_id] = row
            self.rules_by_exact_key[exact_key] = row

    def _fail_closed(
        self,
        *,
        agent_id: int,
        vehicle_token: str,
        active_bus_mask: bool,
        reason: str,
        identity_failure: bool = False,
        occurrence_lookup_failure: bool = False,
    ) -> Dict[str, Any]:
        return {
            "agent_id": int(agent_id),
            "vehicle_token": str(vehicle_token),
            "active_bus_mask": bool(active_bus_mask),
            "action_mask": [True, False, False] if active_bus_mask else [False, False, False],
            "hold_valid": bool(active_bus_mask),
            "serve_move_valid": False,
            "skip_valid": False,
            "skip_invalid_reason_codes": [reason],
            "identity_failure": bool(identity_failure),
            "occurrence_lookup_failure": bool(occurrence_lookup_failure),
            "inactive_action_leakage": False,
            "occurrence_lookup_integrity": False,
            "route_stop_occurrence_id": None,
            "dynamic_state_complete": False,
            "static_guard_state_complete": False,
            "decision_time_obligation_snapshot": None,
            "policy_execution_count": 0,
            "research_rule_approved": True,
            "K_action_mask_available": True,
            "version_binding": self.version_binding.to_payload(),
        }

    def evaluate(
        self,
        *,
        agent_id: int,
        vehicle_token: str,
        active_bus_mask: bool,
        route_id: Optional[str],
        direction_id: Optional[str],
        stop_sequence: Optional[int],
        stop_id: Optional[str],
        route_stop_occurrence_id: Optional[str],
        obligation_state_machine: Optional[ServiceObligationStateMachine],
        decision_ts: int,
    ) -> Dict[str, Any]:
        agent = int(agent_id)
        token = str(vehicle_token)
        expected_token = self.fixed_vehicle_bindings.get(agent)
        if expected_token is None or expected_token != token:
            return self._fail_closed(
                agent_id=agent,
                vehicle_token=token,
                active_bus_mask=active_bus_mask,
                reason="FIXED_VEHICLE_IDENTITY_MISMATCH",
                identity_failure=True,
            )
        if not active_bus_mask:
            result = self._fail_closed(
                agent_id=agent,
                vehicle_token=token,
                active_bus_mask=False,
                reason="INACTIVE_AGENT",
            )
            result["occurrence_lookup_integrity"] = True
            return result
        if route_id is None or direction_id is None or stop_sequence is None or stop_id is None:
            return self._fail_closed(
                agent_id=agent,
                vehicle_token=token,
                active_bus_mask=True,
                reason="OCCURRENCE_CONTEXT_INCOMPLETE",
                occurrence_lookup_failure=True,
            )
        exact_key = (str(route_id), str(direction_id), int(stop_sequence), str(stop_id))
        rule = self.rules_by_exact_key.get(exact_key)
        if rule is None:
            return self._fail_closed(
                agent_id=agent,
                vehicle_token=token,
                active_bus_mask=True,
                reason="OCCURRENCE_LOOKUP_MISSING",
                occurrence_lookup_failure=True,
            )
        occurrence_id = str(rule["route_stop_occurrence_id"])
        if route_stop_occurrence_id is None or str(route_stop_occurrence_id) != occurrence_id:
            return self._fail_closed(
                agent_id=agent,
                vehicle_token=token,
                active_bus_mask=True,
                reason="OCCURRENCE_ID_CONTEXT_MISMATCH",
                occurrence_lookup_failure=True,
            )

        candidate_stop = str(rule["stop_id"])
        post_stop = _optional_string(rule.get("post_skip_target_stop_id"))
        route_key = ("K8_RUNTIME", occurrence_id)
        route = [
            {"stop_id": "__K8_PREVIOUS__", "node_uid": "STOP:__K8_PREVIOUS__"},
            {"stop_id": candidate_stop, "node_uid": f"STOP:{candidate_stop}"},
        ]
        if post_stop is not None:
            route.append({"stop_id": post_stop, "node_uid": f"STOP:{post_stop}"})
        guard = StaticGuardStatus(
            rule["mandatory_stop"],
            rule["protected_stop"],
            rule["planned_itinerary_allows_skip"],
            rule["terminal_or_turnaround_stop"],
            rule["charging_or_driver_relief_stop"],
            rule["valid_post_skip_path"],
            True,
            (),
            "APPROVED_RESEARCH_RULE_WITH_DERIVED_SAFE_BLOCKERS",
        )
        result = build_distinct_three_action_mask(
            _RuntimeVehicle(agent, token, route_key),
            {route_key: route},
            obligation_state_machine=obligation_state_machine,
            decision_ts=int(decision_ts),
            vehicle_token=token,
            static_guard_status=guard,
        )
        result.update(
            {
                "agent_id": agent,
                "vehicle_token": token,
                "active_bus_mask": True,
                "identity_failure": False,
                "occurrence_lookup_failure": False,
                "inactive_action_leakage": False,
                "occurrence_lookup_integrity": True,
                "route_stop_occurrence_id": occurrence_id,
                "rule_version": rule["rule_version"],
                "rule_class": rule["rule_class"],
                "policy_execution_count": 0,
                "research_rule_approved": True,
                "K_action_mask_available": True,
                "version_binding": self.version_binding.to_payload(),
            }
        )
        return result
