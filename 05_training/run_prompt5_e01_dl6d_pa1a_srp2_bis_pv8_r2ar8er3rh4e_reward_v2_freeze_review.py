#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4E Reward V2 temporal/reference freeze review.

This stage creates a machine-readable freeze proposal after H4D. It does not
freeze contracts, rebind runtime reward, rematerialize rewards, evaluate a
policy, or train MAPPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4e_reward_v2_freeze_review.py"

H4C_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4c_reward_semantics_adjudication_20260809_215214"
H4D_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4d_reward_semantics_v2_temporal_binding_fixture_validation_20260809_222412"
R3R_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"

H4C_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4C_REWARD_SEMANTICS_ADJUDICATION_COMPLETE"
H4C_DECISION = "PV8_REWARD_SEMANTICS_V2_READY_FOR_EXPLICIT_APPROVAL_AND_FIXTURE"
H4D_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4D_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_FIXTURE_VALIDATION_COMPLETE"
H4D_DECISION = "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_READY_FOR_FREEZE_REVIEW"
R3R_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3R_REPRESENTATIVE_HISTORICAL_DEMAND_B1_REGENERATION_COMPLETE"

PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4E_REWARD_V2_TEMPORAL_AND_B1_REFERENCE_FREEZE_REVIEW_COMPLETE"
SUCCESS_DECISION = "PV8_REWARD_V2_TEMPORAL_AND_B1_REFERENCE_FREEZE_PROPOSAL_READY_FOR_EXPLICIT_APPROVAL"
READINESS = "REWARD_V2_TEMPORAL_AND_B1_REFERENCE_FREEZE_PROPOSAL_READY_EXPLICIT_APPROVAL_PENDING"

SEMANTICS_CANDIDATE = "PV8_REWARD_SEMANTICS_CANDIDATE_V2"
TEMPORAL_CANDIDATE = "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_CANDIDATE_V1"
TEMPORAL_PROPOSED = "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_FROZEN_V1"
TRAINING_REFERENCE_PROPOSED = "PV8_REPRESENTATIVE_B1_TRAINING_REFERENCE_FROZEN_V1"
TAIL_REFERENCE_PROPOSED = "PV8_REPRESENTATIVE_B1_TAIL_EVALUATION_REFERENCE_FROZEN_V1"
LOCAL_ANCHOR = "PV8_LOCAL_STOP_DECISION_ANCHOR_V1"
SETTLEMENT = "PV8_COMPONENT_SPECIFIC_EVENT_DRIVEN_SETTLEMENT_V1"

SERVICE_REFERENCE = 1.0
AVG_WAIT_REFERENCE = 297.7850241545894
P95_REFERENCE = 576.6999999999999
H4D_PAYLOAD_SHA256 = "6d16816437c51efe6d61c2809b7b71c87fc8e8e19803fdb1cfb040f8a679d4fb"

CANONICAL_EVENT_ORDER = [
    "VEHICLE_ARRIVAL",
    "STATE_SNAPSHOT",
    "OBLIGATION_SNAPSHOT",
    "K_MASK_BUILD",
    "ACTION_SELECTION",
    "ACTION_VALIDATION",
    "BOARDING_ALIGHTING",
    "LOCAL_SERVICE_SETTLEMENT",
    "HOLD_IF_APPLICABLE",
    "DEPARTURE",
    "NEXT_LINK_TRAVEL",
]

PAYLOADS = [
    "r8er3rh4e_upstream_binding.json",
    "r8er3rh4e_h4d_integrity_recheck.json",
    "r8er3rh4e_event_order_reconciliation.json",
    "r8er3rh4e_canonical_temporal_contract.json",
    "r8er3rh4e_reference_type_registry.json",
    "r8er3rh4e_training_reference_candidate.json",
    "r8er3rh4e_tail_evaluation_reference_candidate.json",
    "r8er3rh4e_p95_tolerance_status.json",
    "r8er3rh4e_h240_retirement_review.json",
    "r8er3rh4e_h660_status.json",
    "r8er3rh4e_reward_weight_registry.json",
    "r8er3rh4e_component_transform_contract_audit.json",
    "r8er3rh4e_reward_v2_normalization_terminology_audit.json",
    "r8er3rh4e_freeze_proposal.json",
    "r8er3rh4e_runtime_rebinding_readiness.json",
    "r8er3rh4e_deterministic_replay.json",
    "r8er3rh4e_integrity_audit.json",
    "r8er3rh4e_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4e_reward_v2_freeze_review"


class H4EError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    def fallback(item: Any) -> Any:
        if isinstance(item, (datetime, pd.Timestamp)):
            return item.isoformat()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"unsupported canonical type {type(item)!r}")

    payload = json.dumps(
        k5.json_clean(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=fallback,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: deterministic_content(item) for key, item in value.items() if key != "created_at"}
    if isinstance(value, list):
        return [deterministic_content(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deterministic_content(item) for item in value)
    return value


def with_contract_hash(body: Mapping[str, Any], key: str = "sha256") -> Dict[str, Any]:
    payload = dict(body)
    payload[key] = canonical_hash(body)
    return payload


def verify_gate(root: Path, manifest_name: str, lock_name: str, expected_gate: str, expected_decision: Optional[str] = None) -> Dict[str, Any]:
    manifest = k5.verify_manifest(root, manifest_name, lock_name)
    gate = k5.read_json(root / "gate_decision.json")
    observed_gate = gate.get("gate") or gate.get("terminal_gate")
    if not k5.manifest_ok(manifest) or observed_gate != expected_gate:
        raise H4EError(f"source lineage integrity failed: {root.name}")
    if expected_decision is not None and gate.get("final_decision") != expected_decision:
        raise H4EError(f"source decision drifted: {root.name}")
    return {
        "artifact_root": str(root),
        "manifest_integrity": manifest,
        "gate": observed_gate,
        "decision": gate.get("final_decision"),
        "manifest_sha256": k5.sha256_file(root / manifest_name),
        "lock_sha256": k5.sha256_file(root / lock_name),
    }


def upstream_binding() -> Dict[str, Any]:
    h4c = verify_gate(H4C_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4c.json", "_PV8_R2AR8ER3RH4C_COMPLETE.lock", H4C_GATE, H4C_DECISION)
    h4d = verify_gate(H4D_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4d.json", "_PV8_R2AR8ER3RH4D_COMPLETE.lock", H4D_GATE, H4D_DECISION)
    r3r = verify_gate(R3R_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3r.json", "_PV8_R2AR8ER3R_COMPLETE.lock", R3R_GATE)
    h4d_det = k5.read_json(H4D_ROOT / "r8er3rh4d_deterministic_replay.json")
    if h4d_det.get("payload_sha256") != H4D_PAYLOAD_SHA256:
        raise H4EError("H4D payload SHA drifted")
    return {
        "created_at": iso_kst(),
        "h4c": h4c,
        "h4d": h4d,
        "r3r": r3r,
        "approved_semantics": SEMANTICS_CANDIDATE,
        "h4d_payload_sha256": H4D_PAYLOAD_SHA256,
        "fixed_timestamped_h4d_artifact": H4D_ROOT.name,
        "mutable_latest_pointer_used": False,
    }


def h4d_integrity_recheck() -> Dict[str, Any]:
    integrity = k5.read_json(H4D_ROOT / "r8er3rh4d_integrity_audit.json")
    summary = k5.read_json(H4D_ROOT / "r8er3rh4d_fixture_summary.json")
    deterministic = k5.read_json(H4D_ROOT / "r8er3rh4d_deterministic_replay.json")
    required_zero = [
        "future_leakage_count",
        "critic_future_leakage_count",
        "duplicate_wait_ownership_count",
        "duplicate_service_ownership_count",
        "wrong_transition_accepted_count",
        "wrong_vehicle_accepted_count",
        "wrong_occurrence_accepted_count",
        "K_mask_regression_count",
        "illegal_SKIP_accepted_count",
        "p95_local_reward_generated_count",
        "blanket_SKIP_penalty_count",
        "direct_time_band_reward_term_count",
        "empty_stop_service_reward_inflation_count",
    ]
    zero_failures = {key: integrity.get(key) for key in required_zero if int(integrity.get(key, 0)) != 0}
    fixture_required = {
        **summary["core_fixture_results"],
        **summary["additional_fixture_results"],
    }
    failed_fixtures = [key for key, value in fixture_required.items() if not bool(value)]
    passed = not zero_failures and not failed_fixtures and bool(deterministic.get("identical")) and deterministic.get("payload_sha256") == H4D_PAYLOAD_SHA256
    return {
        "created_at": iso_kst(),
        "fixtures_executed": int(summary["fixtures_executed"]),
        "fixture_results": fixture_required,
        "failed_fixtures": failed_fixtures,
        "required_zero_fields": {key: int(integrity.get(key, 0)) for key in required_zero},
        "zero_failure_fields": zero_failures,
        "h4d_deterministic_replay": bool(deterministic.get("identical")),
        "h4d_payload_sha256": deterministic.get("payload_sha256"),
        "passed": passed,
    }


def event_order_reconciliation() -> Dict[str, Any]:
    candidate = k5.read_json(H4D_ROOT / "r8er3rh4d_temporal_binding_candidate.json")
    events = pd.read_parquet(H4D_ROOT / "r8er3rh4d_event_order_trace.parquet")
    required = ["STATE_SNAPSHOT", "OBLIGATION_SNAPSHOT", "K_MASK_BUILD", "ACTION_SELECTION"]
    by_transition = []
    order_failures = []
    duplicate_ordinal = 0
    for transition_id, group in events.groupby("transition_id"):
        ordinals = {row["event_type"]: int(row["event_ordinal"]) for row in group.to_dict("records")}
        duplicate_ordinal += int(group["event_ordinal"].duplicated().sum())
        if all(name in ordinals for name in required):
            obligation_before_mask = ordinals["OBLIGATION_SNAPSHOT"] < ordinals["K_MASK_BUILD"]
            mask_before_action = ordinals["K_MASK_BUILD"] < ordinals["ACTION_SELECTION"]
        else:
            obligation_before_mask = False
            mask_before_action = False
        exchange = group[group["event_type"].isin(["BOARDING", "ALIGHTING", "BOARDING_AND_ALIGHTING", "NO_PASSENGER_EXCHANGE", "NO_SERVICE_EXCHANGE_LEGAL_SKIP"])]
        if not exchange.empty and "ACTION_SELECTION" in ordinals:
            action_before_exchange = ordinals["ACTION_SELECTION"] < int(exchange["event_ordinal"].min())
        else:
            action_before_exchange = True
        record = {
            "transition_id": transition_id,
            "obligation_before_K_mask": bool(obligation_before_mask),
            "K_mask_before_action": bool(mask_before_action),
            "action_before_board_alight": bool(action_before_exchange),
        }
        by_transition.append(record)
        if not all(record[key] for key in ("obligation_before_K_mask", "K_mask_before_action", "action_before_board_alight")):
            order_failures.append(record)
    h4d_candidate_order = candidate.get("event_order", [])
    h4d_omitted_obligation = "OBLIGATION" not in h4d_candidate_order and "OBLIGATION_SNAPSHOT" not in h4d_candidate_order
    passed = not order_failures and duplicate_ordinal == 0
    return {
        "created_at": iso_kst(),
        "h4d_candidate": candidate.get("candidate"),
        "h4d_candidate_event_order": h4d_candidate_order,
        "h4d_candidate_omitted_obligation_stage": h4d_omitted_obligation,
        "resolution": "OBLIGATION_IS_SEPARATE_CANONICAL_EVENT_STATE_SNAPSHOT",
        "resolution_basis": "H4D event_order_trace contains OBLIGATION_SNAPSHOT with ordinal 30 before K_MASK_BUILD ordinal 40.",
        "canonical_event_order": CANONICAL_EVENT_ORDER,
        "obligation_before_K_mask": all(row["obligation_before_K_mask"] for row in by_transition),
        "K_mask_before_action": all(row["K_mask_before_action"] for row in by_transition),
        "action_before_board_alight": all(row["action_before_board_alight"] for row in by_transition),
        "same_timestamp_order_unambiguous": duplicate_ordinal == 0,
        "duplicate_event_ordinal_count": duplicate_ordinal,
        "transition_count": len(by_transition),
        "order_failures": order_failures,
        "freeze_blocked_by_event_order": not passed,
        "passed": passed,
    }


def canonical_temporal_contract(event_order: Mapping[str, Any]) -> Dict[str, Any]:
    body = {
        "proposed_version": TEMPORAL_PROPOSED,
        "status": "FREEZE_PROPOSAL_READY_NOT_APPROVED",
        "source_candidate": TEMPORAL_CANDIDATE,
        "decision_anchor": LOCAL_ANCHOR,
        "canonical_event_order": CANONICAL_EVENT_ORDER,
        "event_ordinal_rule": "same wall-clock timestamp is allowed; event_ordinal is the deterministic causal ordering authority",
        "transition_identity": [
            "fixture_id",
            "vehicle_slot_id",
            "route_id",
            "direction_id",
            "occurrence_id",
            "local_decision_ts",
            "action",
        ],
        "mandatory_ordering": {
            "current_obligations_known_before_K_mask": bool(event_order["obligation_before_K_mask"]),
            "K_mask_finalized_before_action_selection": bool(event_order["K_mask_before_action"]),
            "action_selection_before_same_stop_boarding_alighting": bool(event_order["action_before_board_alight"]),
        },
        "service_settlement": "LOCAL_CURRENT_STOP_SERVICE_COMPLETION at current-stop obligation completion",
        "avg_wait_settlement": "LOCAL_CAUSALLY_AFFECTED_WAIT_SET at affected passenger boarding resolution",
        "intervention_settlement": "typed forced/external intervention event only",
        "missed_service_gate": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "alignment_excess_gate": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "p95_semantics": "P95_EVALUATION_ONLY",
        "fixed_horizon_status": {
            "H240_legacy_reward_horizon": "RETAINED_FOR_LINEAGE_AND_AUDIT_ONLY",
            "H240_active_local_reward_settlement": "RETIRED_IN_REWARD_SEMANTICS_V2_PROPOSED_NOT_ACTIVATED",
            "H660": "NOT_APPROVED",
        },
        "runtime_reward_binding_allowed": False,
    }
    return with_contract_hash(body)


def r3r_reference_evidence() -> Dict[str, Any]:
    windows = pd.read_parquet(R3R_ROOT / "r8er3r_representative_window_registry.parquet")
    lifecycle = pd.read_parquet(R3R_ROOT / "r8er3r_passenger_lifecycle.parquet")
    norm = k5.read_json(R3R_ROOT / "r8er3r_normalization_candidate.json")
    stratum = k5.read_json(R3R_ROOT / "r8er3r_stratum_coverage.json")
    causal = k5.read_json(R3R_ROOT / "r8er3r_causal_integrity_audit.json")
    missed = k5.read_json(R3R_ROOT / "r8er3r_missed_service_alignment_audit.json")
    loo = k5.read_json(R3R_ROOT / "r8er3r_normalization_loo_window.json")
    prior = k5.read_json(R3R_ROOT / "r8er3r_prior_candidate_comparison.json")
    warmup = k5.read_json(R3R_ROOT / "r8er3r_warmup_convergence_audit.json")
    deterministic = k5.read_json(R3R_ROOT / "r8er3r_deterministic_regeneration.json")
    waits = lifecycle["total_wait_seconds"].astype(float)
    service = float(lifecycle["served_valid_demand"].astype(bool).sum() / lifecycle["eligible_valid_demand"].astype(bool).sum())
    recomputed = {
        "B1_service_reference": service,
        "B1_avg_wait_reference": float(waits.mean()),
        "B1_p95_wait_reference": float(waits.quantile(.95)),
    }
    expected = {
        "B1_service_reference": SERVICE_REFERENCE,
        "B1_avg_wait_reference": AVG_WAIT_REFERENCE,
        "B1_p95_wait_reference": P95_REFERENCE,
    }
    recompute_ok = all(abs(recomputed[key] - expected[key]) <= 1e-12 for key in expected)
    counts = {
        "representative_windows": int(len(windows)),
        "time_band_counts": {str(key): int(value) for key, value in windows["time_band"].value_counts().sort_index().items()},
        "timetable_regime_counts": {str(key): int(value) for key, value in windows["timetable_regime"].value_counts().sort_index().items()},
        "direction_counts": {str(key): int(value) for key, value in windows["direction_id"].astype(str).value_counts().sort_index().items()},
        "passengers_generated": int(len(lifecycle)),
        "passengers_served": int(lifecycle["served_valid_demand"].astype(bool).sum()),
    }
    integrity_pass = bool(
        stratum.get("all_18_strata_present")
        and warmup.get("warmup_converged")
        and causal.get("cold_start_artifact_count") == 0
        and missed.get("missed_eligible_service_count") == 0
        and missed.get("alignment_excess_count_gt_zero") == 0
        and causal.get("future_leakage_count") == 0
        and causal.get("passenger_identity_failure_count") == 0
        and causal.get("request_identity_failure_count") == 0
        and causal.get("vehicle_identity_failure_count") == 0
        and causal.get("headway_drift_count") == 0
        and deterministic.get("identical")
        and recompute_ok
    )
    return {
        "counts": counts,
        "normalization_candidate": norm,
        "stratum_coverage": stratum,
        "causal_integrity": causal,
        "missed_alignment": {
            "missed_eligible_service_count": missed.get("missed_eligible_service_count"),
            "alignment_excess_count_gt_zero": missed.get("alignment_excess_count_gt_zero"),
            "passenger_count": missed.get("passenger_count"),
        },
        "loo_maximum_relative_deviation": loo["maximum_relative_deviation"],
        "prior_candidate_comparison": prior,
        "warmup_convergence": warmup,
        "deterministic": deterministic,
        "recomputed_references": recomputed,
        "expected_references": expected,
        "recompute_matches_candidate": recompute_ok,
        "reference_integrity_pass": integrity_pass,
    }


def reference_type_registry(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "registry_version": "PV8_REWARD_V2_REFERENCE_TYPE_REGISTRY_PROPOSAL_V1",
        "source": R3R_ROOT.name,
        "training_reward_reference": {
            "proposed_version": TRAINING_REFERENCE_PROPOSED,
            "active_components": ["service_reference", "avg_wait_reference_seconds"],
            "service_reference": SERVICE_REFERENCE,
            "avg_wait_reference_seconds": AVG_WAIT_REFERENCE,
            "scope": "LOCAL_TRAINING_REWARD_ONLY",
        },
        "tail_evaluation_reference": {
            "proposed_version": TAIL_REFERENCE_PROPOSED,
            "active_components": ["p95_wait_reference_seconds"],
            "p95_wait_reference_seconds": P95_REFERENCE,
            "scope": "SYSTEM_LEVEL_EVALUATION_AND_PROMOTION_KPI_ONLY",
            "promotion_tolerance": "PENDING_SEPARATE_GATE",
        },
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
        "p95_evaluation_reference_active_after_eventual_freeze": True,
        "ambiguous_training_normalization_label_retired_in_proposal": True,
        "reference_integrity_pass": bool(evidence["reference_integrity_pass"]),
    }


def training_reference_candidate(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    body = {
        "proposed_version": TRAINING_REFERENCE_PROPOSED,
        "status": "FREEZE_PROPOSAL_READY_NOT_APPROVED",
        "source_artifact": R3R_ROOT.name,
        "service_reference": SERVICE_REFERENCE,
        "avg_wait_reference_seconds": AVG_WAIT_REFERENCE,
        "estimator": "passenger-weighted individual completed wait events: total_wait_seconds = actual_board_ts - request_ts",
        "sample_counts": {
            "representative_windows": evidence["counts"]["representative_windows"],
            "eligible_passenger_demand": evidence["counts"]["passengers_generated"],
            "served_passenger_demand": evidence["counts"]["passengers_served"],
        },
        "loo_maximum_relative_deviation": {
            "service": evidence["loo_maximum_relative_deviation"]["B1_service_reference"],
            "avg_wait": evidence["loo_maximum_relative_deviation"]["B1_avg_wait_reference"],
        },
        "excluded_from_training_reference": ["p95_wait_reference_seconds"],
        "training_reference_frozen": False,
    }
    return with_contract_hash(body)


def tail_reference_candidate(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    body = {
        "proposed_version": TAIL_REFERENCE_PROPOSED,
        "status": "FREEZE_PROPOSAL_READY_NOT_APPROVED",
        "source_artifact": R3R_ROOT.name,
        "p95_wait_reference_seconds": P95_REFERENCE,
        "estimator": "linear p95 over passenger-weighted individual completed wait events",
        "scope": "SYSTEM_LEVEL_EVALUATION_AND_PROMOTION_KPI_ONLY",
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
        "p95_evaluation_reference_active_after_eventual_freeze": True,
        "promotion_tolerance": "PENDING_SEPARATE_GATE",
        "loo_maximum_relative_deviation": {
            "p95_wait": evidence["loo_maximum_relative_deviation"]["B1_p95_wait_reference"],
        },
        "tail_evaluation_reference_frozen": False,
    }
    return with_contract_hash(body)


def p95_tolerance_status() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "p95_wait_reference_seconds": P95_REFERENCE,
        "reference_value_can_be_proposed_for_freeze": True,
        "promotion_tolerance": "P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING",
        "promotion_tolerance_status": "PENDING_SEPARATE_GATE",
        "numeric_tolerance_created_in_H4E": False,
        "reference_value_equals_acceptance_tolerance": False,
    }


def h240_retirement_review(horizon: Mapping[str, Any]) -> Dict[str, Any]:
    ok = bool(
        horizon.get("local_component_result_identical_across_H120_H240_H660")
        and not horizon.get("fixed_H4_technically_required_for_candidate_v2_local_components")
    )
    return {
        "created_at": iso_kst(),
        "H120_H240_H660_local_component_equivalence": bool(horizon.get("local_component_result_identical_across_H120_H240_H660")),
        "fixed_H4_technically_required_for_candidate_v2_local_components": bool(horizon.get("fixed_H4_technically_required_for_candidate_v2_local_components")),
        "H240_LEGACY_REWARD_HORIZON": "RETAINED_FOR_LINEAGE_AND_AUDIT_ONLY",
        "H240_ACTIVE_LOCAL_REWARD_SETTLEMENT": "RETIRED_IN_REWARD_SEMANTICS_V2_PROPOSED_NOT_ACTIVATED" if ok else "REVIEW_BLOCKED",
        "proposed_H240_status": "LEGACY_LINEAGE_ONLY_NOT_ACTIVE_LOCAL_SETTLEMENT" if ok else "CONFLICT",
        "historical_artifacts_preserved": True,
        "historical_R2A_R8D_R8E_rewritten": False,
        "H240_retired_from_runtime": False,
        "passed": ok,
    }


def h660_status() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "H660": "NOT_APPROVED",
        "H660_replaces_H240": False,
        "fixed_660_second_reward_window_required_by_candidate_v2": False,
    }


def reward_weight_registry() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "registry_version": "PV8_REWARD_V2_WEIGHT_REGISTRY_FREEZE_REVIEW_V1",
        "positive_magnitude_convention": True,
        "formula": "+ 3.0 * local_service_component + 2.0 * local_affected_avg_wait_component - 0.25 * explicit_forced_external_intervention",
        "weights": {
            "local_service_component": {"magnitude": 3.0, "formula_sign": "+"},
            "local_affected_avg_wait_component": {"magnitude": 2.0, "formula_sign": "+"},
            "explicit_forced_external_intervention": {"magnitude": 0.25, "formula_sign": "-"},
        },
        "inactive_lineage": {
            "old_p95_weight": 3.0,
            "old_p95_weight_status": "INACTIVE_NOT_REDISTRIBUTED",
            "active_local_p95_weight": 0.0,
            "p95_weight_redistribution": 0.0,
        },
    }


def component_transform_contract_audit() -> Dict[str, Any]:
    avg = k5.read_json(H4C_ROOT / "r8er3rh4c_avg_wait_local_semantics.json")
    service = k5.read_json(H4C_ROOT / "r8er3rh4c_service_local_semantics.json")
    candidate = k5.read_json(H4C_ROOT / "r8er3rh4c_reward_semantics_candidate_v2.json")
    avg_formula = str(avg.get("component_formula", ""))
    required_avg_parts = [
        "centered=1-local_mean_wait/297.7850241545894",
        "positive improvement retained only when local service score=1",
        "deterioration remains non-positive",
    ]
    avg_complete = all(part in avg_formula for part in required_avg_parts)
    service_formula = str(service.get("score_formula", ""))
    service_complete = "NOT_APPLICABLE" in service_formula and "1.0" in service_formula and "0.0" in service_formula
    return {
        "created_at": iso_kst(),
        "local_service_transform_source": str(H4C_ROOT / "r8er3rh4c_service_local_semantics.json"),
        "local_service_score_formula": service.get("score_formula"),
        "local_service_transform_complete": service_complete,
        "local_avg_wait_transform_source": str(H4C_ROOT / "r8er3rh4c_avg_wait_local_semantics.json"),
        "local_avg_wait_component_formula": avg_formula,
        "local_avg_wait_transform_complete": avg_complete,
        "wait_formula": avg.get("wait_formula"),
        "candidate_formula": candidate.get("candidate_training_formula"),
        "new_algebraic_wait_transform_invented_in_H4E": False,
        "component_transform_contract_complete": bool(service_complete and avg_complete),
    }


def terminology_audit() -> Dict[str, Any]:
    scanned = [
        H4C_ROOT / "r8er3rh4c_reward_semantics_candidate_v2.json",
        H4D_ROOT / "r8er3rh4d_temporal_binding_candidate.json",
        R3R_ROOT / "r8er3r_normalization_candidate.json",
    ]
    records = []
    active_p95_training = 0
    ambiguous_count = 0
    for path in scanned:
        data = k5.read_json(path)
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
        has_p95 = "p95" in text
        has_normalization = "normalization" in text.lower()
        active_training = "p95_training_normalization_active" in text and "true" in text
        ambiguous = bool(has_p95 and has_normalization and "candidate_only" in text.lower())
        active_p95_training += int(active_training)
        ambiguous_count += int(ambiguous)
        records.append(
            {
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "contains_p95_reference": has_p95,
                "contains_normalization_label": has_normalization,
                "active_p95_training_normalization_detected": active_training,
                "ambiguous_candidate_label_detected": ambiguous,
            }
        )
    return {
        "created_at": iso_kst(),
        "scanned_records": records,
        "ambiguous_candidate_label_count": ambiguous_count,
        "active_p95_training_normalization_detected_count": active_p95_training,
        "H4E_representation_patch": "split proposal into training reference registry and tail evaluation reference registry; historical artifacts are not rewritten",
        "freeze_blocked_by_p95_training_normalization": active_p95_training > 0,
        "passed": active_p95_training == 0,
    }


def runtime_rebinding_readiness(decision_ready: bool) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "runtime_reward_rebinding_ready_for_separate_authorization": bool(decision_ready),
        "reward_runtime_binding_allowed": False,
        "reward_values_rematerialized": False,
        "required_before_runtime_binding": [
            "explicit approval of temporal binding freeze",
            "explicit approval of B1 training reference freeze",
            "explicit approval of p95 evaluation reference freeze",
            "explicit approval of H240 retirement from active local reward settlement",
        ],
    }


def freeze_proposal(
    temporal: Mapping[str, Any],
    training_ref: Mapping[str, Any],
    tail_ref: Mapping[str, Any],
    h240: Mapping[str, Any],
    h660: Mapping[str, Any],
    weights: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> Dict[str, Any]:
    body = {
        "proposal_version": "PV8_REWARD_V2_TEMPORAL_AND_B1_REFERENCE_FREEZE_PROPOSAL_V1",
        "status": "FREEZE_PROPOSAL_READY_FOR_EXPLICIT_USER_APPROVAL",
        "proposed_temporal_binding": {
            "version": temporal["proposed_version"],
            "sha256": temporal["sha256"],
        },
        "proposed_training_reference": {
            "version": training_ref["proposed_version"],
            "sha256": training_ref["sha256"],
            "service_reference": SERVICE_REFERENCE,
            "avg_wait_reference_seconds": AVG_WAIT_REFERENCE,
        },
        "proposed_tail_evaluation_reference": {
            "version": tail_ref["proposed_version"],
            "sha256": tail_ref["sha256"],
            "p95_wait_reference_seconds": P95_REFERENCE,
            "promotion_tolerance": "PENDING_SEPARATE_GATE",
        },
        "proposed_H240_status": h240["proposed_H240_status"],
        "H660_status": h660["H660"],
        "reward_weights": weights["weights"],
        "p95_inactive_weight_lineage": weights["inactive_lineage"],
        "all_source_artifact_hashes": {
            "h4c_manifest_sha256": upstream["h4c"]["manifest_sha256"],
            "h4d_manifest_sha256": upstream["h4d"]["manifest_sha256"],
            "r3r_manifest_sha256": upstream["r3r"]["manifest_sha256"],
            "h4d_payload_sha256": H4D_PAYLOAD_SHA256,
        },
        "all_proposed_freeze_hashes": {
            "temporal_binding_sha256": temporal["sha256"],
            "training_reference_sha256": training_ref["sha256"],
            "tail_evaluation_reference_sha256": tail_ref["sha256"],
        },
        "runtime_mutation_allowed": False,
        "explicit_approval_required": True,
    }
    return with_contract_hash(body, key="proposal_sha256")


def integrity_audit(
    h4d: Mapping[str, Any],
    event_order: Mapping[str, Any],
    evidence: Mapping[str, Any],
    transform: Mapping[str, Any],
    terminology: Mapping[str, Any],
    h240: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "h4d_integrity_pass": bool(h4d["passed"]),
        "event_order_reconciled": bool(event_order["passed"]),
        "OBLIGATION_available_before_K_MASK": bool(event_order["obligation_before_K_mask"]),
        "same_timestamp_order_unambiguous": bool(event_order["same_timestamp_order_unambiguous"]),
        "b1_reference_integrity_pass": bool(evidence["reference_integrity_pass"]),
        "component_transform_contract_complete": bool(transform["component_transform_contract_complete"]),
        "p95_reference_role_misclassified": not bool(terminology["passed"]),
        "H240_retirement_semantics_conflict": not bool(h240["passed"]),
        "API_calls": 0,
        "DB_queries": 0,
        "DB_writes": 0,
        "reward_values_rematerialized": False,
        "reward_runtime_binding_allowed": False,
        "MAPPO_training_authorized": False,
    }


def decide(
    upstream_ok: bool,
    h4d: Mapping[str, Any],
    event_order: Mapping[str, Any],
    evidence: Mapping[str, Any],
    transform: Mapping[str, Any],
    terminology: Mapping[str, Any],
    h240: Mapping[str, Any],
) -> Tuple[str, str]:
    if not upstream_ok:
        return "PV8_FREEZE_SOURCE_LINEAGE_INTEGRITY_FAILED", "At least one source manifest, lock, or gate failed."
    if not event_order["passed"]:
        return "PV8_H4D_EVENT_ORDER_RECONCILIATION_REQUIRED", "OBLIGATION/K-mask/action ordering is not proven."
    if not transform["component_transform_contract_complete"]:
        return "PV8_REWARD_V2_COMPONENT_TRANSFORM_CONTRACT_INCOMPLETE", "The local service or avg-wait transform is not unambiguous."
    if not evidence["reference_integrity_pass"]:
        return "PV8_B1_REFERENCE_ESTIMATOR_INTEGRITY_FAILED", "Representative B1 estimator or lineage integrity failed."
    if not terminology["passed"]:
        return "PV8_P95_REFERENCE_ROLE_MISCLASSIFIED", "P95 appears as active training normalization."
    if not h240["passed"]:
        return "PV8_H240_RETIREMENT_SEMANTICS_CONFLICT", "H240 retirement from active local settlement conflicts with evidence."
    if not h4d["passed"]:
        return "PV8_REWARD_V2_FREEZE_REVIEW_FAILED", "H4D fixture or integrity preservation failed."
    return SUCCESS_DECISION, "Freeze proposal is internally coherent and ready for explicit user approval; no object was frozen or activated."


def review_once() -> Dict[str, Any]:
    upstream = upstream_binding()
    h4d = h4d_integrity_recheck()
    event_order = event_order_reconciliation()
    temporal = canonical_temporal_contract(event_order)
    evidence = r3r_reference_evidence()
    ref_registry = reference_type_registry(evidence)
    training_ref = training_reference_candidate(evidence)
    tail_ref = tail_reference_candidate(evidence)
    tolerance = p95_tolerance_status()
    h240 = h240_retirement_review(k5.read_json(H4D_ROOT / "r8er3rh4d_horizon_independence_audit.json"))
    h660 = h660_status()
    weights = reward_weight_registry()
    transform = component_transform_contract_audit()
    terminology = terminology_audit()
    integrity = integrity_audit(h4d, event_order, evidence, transform, terminology, h240)
    decision, rationale = decide(True, h4d, event_order, evidence, transform, terminology, h240)
    runtime = runtime_rebinding_readiness(decision == SUCCESS_DECISION)
    proposal = freeze_proposal(temporal, training_ref, tail_ref, h240, h660, weights, upstream)
    readiness = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "rationale": rationale,
        "reward_semantics_v2_design_approved": True,
        "reward_v2_fixture_validated": bool(h4d["passed"]),
        "temporal_binding_candidate_review_passed": bool(event_order["passed"]),
        "training_reference_review_passed": bool(evidence["reference_integrity_pass"]),
        "tail_evaluation_reference_review_passed": bool(evidence["reference_integrity_pass"] and terminology["passed"]),
        "H240_retirement_review_passed": bool(h240["passed"]),
        "runtime_rebinding_ready_for_separate_authorization": bool(runtime["runtime_reward_rebinding_ready_for_separate_authorization"]),
        "temporal_binding_frozen": False,
        "training_reference_frozen": False,
        "tail_evaluation_reference_frozen": False,
        "normalization_frozen": False,
        "H240_retired_from_runtime": False,
        "reward_runtime_binding_allowed": False,
        "reward_rematerialization_allowed": False,
        "policy_evaluation_authorized": False,
        "MAPPO_training_authorized": False,
        "next_step": "STOP pending explicit user approval/rejection of temporal binding freeze, B1 training reference freeze, p95 evaluation reference freeze, and H240 retirement.",
    }
    payload = {
        "upstream": deterministic_content(upstream),
        "h4d": deterministic_content(h4d),
        "event_order": deterministic_content(event_order),
        "temporal": deterministic_content(temporal),
        "ref_registry": deterministic_content(ref_registry),
        "training_ref": deterministic_content(training_ref),
        "tail_ref": deterministic_content(tail_ref),
        "tolerance": deterministic_content(tolerance),
        "h240": deterministic_content(h240),
        "h660": deterministic_content(h660),
        "weights": deterministic_content(weights),
        "transform": deterministic_content(transform),
        "terminology": deterministic_content(terminology),
        "proposal": deterministic_content(proposal),
        "runtime": deterministic_content(runtime),
        "integrity": deterministic_content(integrity),
        "readiness": deterministic_content(readiness),
    }
    return {
        "upstream": upstream,
        "h4d": h4d,
        "event_order": event_order,
        "temporal": temporal,
        "evidence": evidence,
        "ref_registry": ref_registry,
        "training_ref": training_ref,
        "tail_ref": tail_ref,
        "tolerance": tolerance,
        "h240": h240,
        "h660": h660,
        "weights": weights,
        "transform": transform,
        "terminology": terminology,
        "proposal": proposal,
        "runtime": runtime,
        "integrity": integrity,
        "readiness": readiness,
        "decision": decision,
        "rationale": rationale,
        "payload_sha256": canonical_hash(payload),
    }


def deterministic_replay(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "first_payload_sha256": first["payload_sha256"],
        "second_payload_sha256": second["payload_sha256"],
        "payload_sha256": first["payload_sha256"],
        "identical": first["payload_sha256"] == second["payload_sha256"],
        "canonical_event_order_identical": first["temporal"]["canonical_event_order"] == second["temporal"]["canonical_event_order"],
        "reference_registry_identical": canonical_hash(deterministic_content(first["ref_registry"])) == canonical_hash(deterministic_content(second["ref_registry"])),
        "freeze_proposal_identical": canonical_hash(deterministic_content(first["proposal"])) == canonical_hash(deterministic_content(second["proposal"])),
        "contract_sha256s_identical": (
            first["temporal"]["sha256"] == second["temporal"]["sha256"]
            and first["training_ref"]["sha256"] == second["training_ref"]["sha256"]
            and first["tail_ref"]["sha256"] == second["tail_ref"]["sha256"]
            and first["proposal"]["proposal_sha256"] == second["proposal"]["proposal_sha256"]
        ),
        "readiness_decision_identical": deterministic_content(first["readiness"]) == deterministic_content(second["readiness"]),
    }


def claim_guards(result: Mapping[str, Any]) -> Dict[str, Any]:
    success = result["decision"] == SUCCESS_DECISION
    return {
        "created_at": iso_kst(),
        "reward_semantics_v2_design_approved": True,
        "reward_v2_fixture_validated": bool(result["h4d"]["passed"]),
        "temporal_binding_candidate_review_passed": bool(result["event_order"]["passed"]),
        "training_reference_review_passed": bool(result["evidence"]["reference_integrity_pass"]),
        "tail_evaluation_reference_review_passed": bool(result["evidence"]["reference_integrity_pass"] and result["terminology"]["passed"]),
        "H240_retirement_review_passed": bool(result["h240"]["passed"]),
        "runtime_rebinding_ready_for_separate_authorization": bool(success),
        "temporal_binding_frozen": False,
        "training_reference_frozen": False,
        "tail_evaluation_reference_frozen": False,
        "normalization_frozen": False,
        "H240_retired_from_runtime": False,
        "reward_runtime_binding_allowed": False,
        "reward_rematerialization_allowed": False,
        "optimizer_creation_allowed": False,
        "policy_evaluation_authorized": False,
        "MAPPO_training_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }


def final_report(result: Mapping[str, Any], deterministic: Mapping[str, Any]) -> str:
    decision = result["decision"]
    evidence = result["evidence"]
    lines = [
        "# PV8-R2A-R8E-R3-R-H4E Final Report",
        "",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision}`",
        "- review only: no temporal binding, B1 reference, p95 reference, H240 retirement, runtime reward binding, reward rematerialization, policy evaluation, or MAPPO training was activated.",
        "",
        "## Four Objects",
        "",
        f"- A. Reward Semantics V2: `{SEMANTICS_CANDIDATE}` is already user-approved as design semantics.",
        f"- B. Temporal Binding: H4D validated `{TEMPORAL_CANDIDATE}`; freeze proposal `{TEMPORAL_PROPOSED}` is pending explicit approval, SHA-256 `{result['temporal']['sha256']}`.",
        f"- C. Training B1 Reference: service `{SERVICE_REFERENCE}`, avg wait `{AVG_WAIT_REFERENCE}` sec; proposal `{TRAINING_REFERENCE_PROPOSED}` pending explicit approval, SHA-256 `{result['training_ref']['sha256']}`.",
        f"- D. Tail Evaluation Reference: p95 `{P95_REFERENCE}` sec; proposal `{TAIL_REFERENCE_PROPOSED}` pending explicit approval, SHA-256 `{result['tail_ref']['sha256']}`.",
        "",
        "## Event Order",
        "",
        "- H4D candidate JSON omitted `OBLIGATION`, but H4D event trace records `OBLIGATION_SNAPSHOT` as a separate canonical event/state snapshot.",
        "- resolved canonical order: `VEHICLE_ARRIVAL -> STATE_SNAPSHOT -> OBLIGATION_SNAPSHOT -> K_MASK_BUILD -> ACTION_SELECTION -> ACTION_VALIDATION -> BOARDING_ALIGHTING -> LOCAL_SERVICE_SETTLEMENT -> HOLD_IF_APPLICABLE -> DEPARTURE -> NEXT_LINK_TRAVEL`.",
        f"- OBLIGATION before K-mask / K-mask before action / action before board-alight: `{result['event_order']['obligation_before_K_mask']} / {result['event_order']['K_mask_before_action']} / {result['event_order']['action_before_board_alight']}`.",
        f"- same-timestamp order unambiguous: `{result['event_order']['same_timestamp_order_unambiguous']}`.",
        "",
        "## B1 Reference",
        "",
        f"- representative windows: `{evidence['counts']['representative_windows']}`; time bands `{evidence['counts']['time_band_counts']}`; regimes `{evidence['counts']['timetable_regime_counts']}`; directions `{evidence['counts']['direction_counts']}`.",
        f"- generated/served passengers: `{evidence['counts']['passengers_generated']} / {evidence['counts']['passengers_served']}`.",
        f"- cold-start / missed service / alignment excess / future leakage / headway drift: `{evidence['causal_integrity']['cold_start_artifact_count']} / {evidence['missed_alignment']['missed_eligible_service_count']} / {evidence['missed_alignment']['alignment_excess_count_gt_zero']} / {evidence['causal_integrity']['future_leakage_count']} / {evidence['causal_integrity']['headway_drift_count']}`.",
        f"- LOO max relative deviation service/avg/p95: `{evidence['loo_maximum_relative_deviation']['B1_service_reference']} / {evidence['loo_maximum_relative_deviation']['B1_avg_wait_reference']} / {evidence['loo_maximum_relative_deviation']['B1_p95_wait_reference']}`.",
        "- stale `1.0 / 2062.5 / 5338.25` remains rejected as `COLD_START_CONTAMINATED_LINEAGE_ONLY`; old candidates were not averaged together.",
        "",
        "## H240 And P95",
        "",
        "- H240 is not scientifically required for Reward Semantics V2 local event-driven settlement.",
        "- The proposal is to retain H240 only as historical/audit lineage and retire it from active local reward settlement semantics. This retirement has NOT yet been activated.",
        "- H660 remains `NOT_APPROVED` and does not replace H240.",
        "- `p95_wait` is not a local training reward and is not part of active training normalization.",
        f"- `{P95_REFERENCE}` sec is the proposed representative B1 evaluation reference for system-level tail-wait assessment.",
        "- the numerical promotion tolerance remains `P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING`.",
        "",
        "## Integrity",
        "",
        f"- H4D fixture/integrity recheck: `{result['h4d']['passed']}`.",
        f"- component transform contract complete: `{result['transform']['component_transform_contract_complete']}`.",
        f"- active p95 training-normalization detections: `{result['terminology']['active_p95_training_normalization_detected_count']}`.",
        f"- runtime rebinding ready for separate authorization: `{result['runtime']['runtime_reward_rebinding_ready_for_separate_authorization']}`; runtime binding allowed now: `{result['runtime']['reward_runtime_binding_allowed']}`.",
        f"- deterministic review replay: `{deterministic['identical']}`; payload SHA-256 `{deterministic['payload_sha256']}`.",
        f"- freeze proposal SHA-256: `{result['proposal']['proposal_sha256']}`.",
        "- next user decision should explicitly approve or reject temporal binding freeze, B1 training reference freeze, p95 evaluation reference freeze, and H240 retirement from active local settlement.",
        "",
    ]
    return "\n".join(lines)


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4e.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4e.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_R2AR8ER3RH4E_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def run(root: Path) -> Path:
    first = review_once()
    second = review_once()
    deterministic = deterministic_replay(first, second)
    deterministic_flags = [value for key, value in deterministic.items() if key == "identical" or key.endswith("_identical")]
    if not all(bool(value) for value in deterministic_flags):
        raise H4EError("deterministic freeze review drifted")
    success = first["decision"] == SUCCESS_DECISION
    writer = k5.Writer(root)
    writer.json("r8er3rh4e_upstream_binding.json", first["upstream"])
    writer.json("r8er3rh4e_h4d_integrity_recheck.json", first["h4d"])
    writer.json("r8er3rh4e_event_order_reconciliation.json", first["event_order"])
    writer.json("r8er3rh4e_canonical_temporal_contract.json", first["temporal"])
    writer.json("r8er3rh4e_reference_type_registry.json", first["ref_registry"])
    writer.json("r8er3rh4e_training_reference_candidate.json", first["training_ref"])
    writer.json("r8er3rh4e_tail_evaluation_reference_candidate.json", first["tail_ref"])
    writer.json("r8er3rh4e_p95_tolerance_status.json", first["tolerance"])
    writer.json("r8er3rh4e_h240_retirement_review.json", first["h240"])
    writer.json("r8er3rh4e_h660_status.json", first["h660"])
    writer.json("r8er3rh4e_reward_weight_registry.json", first["weights"])
    writer.json("r8er3rh4e_component_transform_contract_audit.json", first["transform"])
    writer.json("r8er3rh4e_reward_v2_normalization_terminology_audit.json", first["terminology"])
    writer.json("r8er3rh4e_freeze_proposal.json", first["proposal"])
    writer.json("r8er3rh4e_runtime_rebinding_readiness.json", first["runtime"])
    writer.json("r8er3rh4e_deterministic_replay.json", deterministic)
    writer.json("r8er3rh4e_integrity_audit.json", first["integrity"])
    writer.json("r8er3rh4e_readiness_decision.json", first["readiness"])
    guards = claim_guards(first)
    writer.json("claim_guard_status.json", guards)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "gate_passed": True,
        "final_decision": first["decision"],
        "readiness": READINESS,
        "failure_reasons": [] if success else [first["rationale"]],
        "frozen_implied": False,
        "runtime_approval_implied": False,
    }
    writer.json("run_manifest.json", {"created_at": iso_kst(), "runner": str(RUNNER_PATH), "mode": "freeze_review", "python": sys.version, "platform": platform.platform(), "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "api_call_count": 0, "db_query_count": 0, "db_write_count": 0, "historical_dataset_regeneration_count": 0, "reward_values_rematerialized": False, "reward_runtime_binding_count": 0, "policy_evaluation_count": 0, "optimizer_creation_count": 0, "checkpoint_generation_count": 0})
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": first["decision"], "readiness": READINESS})
    writer.text("final_report.md", final_report(first, deterministic))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("review",), required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    root = k5.validate_artifact_root(Path(args.artifact_root))
    run(root)
    gate = k5.read_json(root / "gate_decision.json")
    print(f"artifact_root: {root}")
    print(f"gate: {gate['gate']}")
    print(f"decision: {gate['final_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
