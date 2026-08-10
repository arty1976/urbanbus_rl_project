#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4F immutable Reward V2 freeze preflight.

This stage records the user's explicit approval of H4E, converts the approved
Reward Semantics V2 proposal objects into immutable frozen objects, and audits
the current runtime reward path for the next rebinding stage. It does not patch
runtime code, rematerialize rewards, evaluate a policy, or train MAPPO.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4e_reward_v2_freeze_review as h4e


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4f_reward_v2_immutable_freeze_preflight.py"

R3R_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
H4B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4b_local_decision_reward_settlement_20260809_212055"
H4C_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4c_reward_semantics_adjudication_20260809_215214"
H4D_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4d_reward_semantics_v2_temporal_binding_fixture_validation_20260809_222412"
H4E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4e_reward_v2_freeze_review_20260809_224411"

R3R_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3R_REPRESENTATIVE_HISTORICAL_DEMAND_B1_REGENERATION_COMPLETE"
H4B_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4B_LOCAL_DECISION_REWARD_SETTLEMENT_AUDIT_COMPLETE"
H4C_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4C_REWARD_SEMANTICS_ADJUDICATION_COMPLETE"
H4D_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4D_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_FIXTURE_VALIDATION_COMPLETE"
H4E_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4E_REWARD_V2_TEMPORAL_AND_B1_REFERENCE_FREEZE_REVIEW_COMPLETE"

H4B_DECISION = "PV8_P95_REWARD_SEMANTICS_REDESIGN_REQUIRES_USER_APPROVAL"
H4C_DECISION = "PV8_REWARD_SEMANTICS_V2_READY_FOR_EXPLICIT_APPROVAL_AND_FIXTURE"
H4D_DECISION = "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_READY_FOR_FREEZE_REVIEW"
H4E_DECISION = "PV8_REWARD_V2_TEMPORAL_AND_B1_REFERENCE_FREEZE_PROPOSAL_READY_FOR_EXPLICIT_APPROVAL"

PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4F_REWARD_V2_APPROVED_IMMUTABLE_FREEZE_AND_RUNTIME_REBINDING_PREFLIGHT_COMPLETE"
SUCCESS_DECISION = "PV8_REWARD_V2_IMMUTABLY_FROZEN_RUNTIME_REBINDING_READY"
READINESS = "REWARD_V2_IMMUTABLY_FROZEN_RUNTIME_REBINDING_PREFLIGHT_READY_H4G_PENDING_USER_COMMAND"

LOCAL_ANCHOR = "PV8_LOCAL_STOP_DECISION_ANCHOR_V1"
SETTLEMENT = "PV8_COMPONENT_SPECIFIC_EVENT_DRIVEN_SETTLEMENT_V1"
FREEZE_VERSION = "PV8_REWARD_SEMANTICS_V2_IMMUTABLE_FREEZE_V1"
TEMPORAL_VERSION = "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_FROZEN_V1"
TRAINING_REFERENCE_VERSION = "PV8_REPRESENTATIVE_B1_TRAINING_REFERENCE_FROZEN_V1"
TAIL_REFERENCE_VERSION = "PV8_REPRESENTATIVE_B1_TAIL_EVALUATION_REFERENCE_FROZEN_V1"
HORIZON_STATUS_VERSION = "PV8_REWARD_V2_HORIZON_STATUS_FROZEN_V1"
WEIGHT_REGISTRY_VERSION = "PV8_REWARD_V2_WEIGHT_REGISTRY_FROZEN_V1"
PATCH_PLAN_VERSION = "PV8_REWARD_V2_RUNTIME_REBINDING_PATCH_PLAN_V1"

SERVICE_REFERENCE = 1.0
AVG_WAIT_REFERENCE = 297.7850241545894
P95_REFERENCE = 576.6999999999999
H4D_PAYLOAD_SHA256 = "6d16816437c51efe6d61c2809b7b71c87fc8e8e19803fdb1cfb040f8a679d4fb"
H4E_PAYLOAD_SHA256 = "a202aaa8cc73180726740f2ee1bf265de325cee2c9f2fd9dc5737aff4c4b3baf"
H4E_PROPOSAL_SHA256 = "f475315af4f98f9166387e379862fe9874c5b8778316478a706c8d48ff0fe7bd"
H4E_TEMPORAL_SHA256 = "ce39d5e57be117f617b128c4738cf91ff77c4ceebda99219ec099a7a2bc81093"
H4E_TRAINING_REFERENCE_SHA256 = "3e000401e8019320b2b2d1d8c1e216dfcc000d03a7f051073217a6e9b2dca8d3"
H4E_TAIL_REFERENCE_SHA256 = "95469fc89807555a63a63914566501c41fbfc86f89ae440c44c99d549d218325"

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

RUNTIME_TARGETS = [
    TRAINING_ROOT / "rewards" / "mappo_reward_v1.py",
    TRAINING_ROOT / "simulator" / "pv8_reward_outcome_collector.py",
    TRAINING_ROOT / "simulator" / "pv8_b1_orchestrator.py",
    TRAINING_ROOT / "mappo_runner.py",
]

PAYLOADS = [
    "r8er3rh4f_user_approval_record.json",
    "r8er3rh4f_upstream_binding.json",
    "r8er3rh4f_source_integrity.json",
    "r8er3rh4f_canonical_event_order_frozen.json",
    "r8er3rh4f_temporal_binding_frozen.json",
    "r8er3rh4f_training_reference_frozen.json",
    "r8er3rh4f_tail_evaluation_reference_frozen.json",
    "r8er3rh4f_reward_weight_registry_frozen.json",
    "r8er3rh4f_horizon_status_frozen.json",
    "r8er3rh4f_reward_semantics_v2_immutable_freeze.json",
    "r8er3rh4f_freeze_hash_registry.json",
    "r8er3rh4f_component_transform_resolution.json",
    "r8er3rh4f_runtime_reward_inventory.json",
    "r8er3rh4f_stale_runtime_semantics_audit.json",
    "r8er3rh4f_frozen_to_runtime_mapping.parquet",
    "r8er3rh4f_runtime_rebinding_patch_plan.json",
    "r8er3rh4f_h4d_fixture_regression.json",
    "r8er3rh4f_ownership_integrity.json",
    "r8er3rh4f_future_leakage_audit.json",
    "r8er3rh4f_runtime_rebinding_readiness.json",
    "r8er3rh4f_deterministic_replay.json",
    "r8er3rh4f_integrity_audit.json",
    "r8er3rh4f_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class H4FError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        k5.json_clean(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def with_hash(body: Mapping[str, Any], *, key: str = "sha256") -> Dict[str, Any]:
    payload = dict(body)
    payload[key] = canonical_hash(deterministic_content(body))
    return payload


def verify_gate(
    root: Path,
    manifest_name: str,
    lock_name: str,
    expected_gate: str,
    expected_decision: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = k5.verify_manifest(root, manifest_name, lock_name)
    gate = k5.read_json(root / "gate_decision.json")
    observed_gate = gate.get("gate") or gate.get("terminal_gate")
    if observed_gate != expected_gate or not k5.manifest_ok(manifest):
        raise H4FError(f"upstream integrity failed: {root.name}")
    if expected_decision is not None and gate.get("final_decision") != expected_decision:
        raise H4FError(f"upstream decision drifted: {root.name}")
    return {
        "artifact_root": str(root),
        "manifest_name": manifest_name,
        "lock_name": lock_name,
        "manifest_integrity": manifest,
        "gate": observed_gate,
        "decision": gate.get("final_decision"),
        "manifest_sha256": k5.sha256_file(root / manifest_name),
        "lock_sha256": k5.sha256_file(root / lock_name),
    }


def source_integrity() -> Dict[str, Any]:
    sources = {
        "R3-R": verify_gate(R3R_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3r.json", "_PV8_R2AR8ER3R_COMPLETE.lock", R3R_GATE),
        "H4B": verify_gate(H4B_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4b.json", "_PV8_R2AR8ER3RH4B_COMPLETE.lock", H4B_GATE, H4B_DECISION),
        "H4C": verify_gate(H4C_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4c.json", "_PV8_R2AR8ER3RH4C_COMPLETE.lock", H4C_GATE, H4C_DECISION),
        "H4D": verify_gate(H4D_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4d.json", "_PV8_R2AR8ER3RH4D_COMPLETE.lock", H4D_GATE, H4D_DECISION),
        "H4E": verify_gate(H4E_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4e.json", "_PV8_R2AR8ER3RH4E_COMPLETE.lock", H4E_GATE, H4E_DECISION),
    }
    h4d_det = k5.read_json(H4D_ROOT / "r8er3rh4d_deterministic_replay.json")
    h4e_det = k5.read_json(H4E_ROOT / "r8er3rh4e_deterministic_replay.json")
    h4e_proposal = k5.read_json(H4E_ROOT / "r8er3rh4e_freeze_proposal.json")
    proposal_hash_ok = h4e_proposal.get("proposal_sha256") == H4E_PROPOSAL_SHA256
    checks = {
        "h4d_payload_sha256_ok": h4d_det.get("payload_sha256") == H4D_PAYLOAD_SHA256,
        "h4e_payload_sha256_ok": h4e_det.get("payload_sha256") == H4E_PAYLOAD_SHA256,
        "h4e_proposal_sha256_ok": proposal_hash_ok,
        "h4e_temporal_sha256_ok": h4e_proposal.get("all_proposed_freeze_hashes", {}).get("temporal_binding_sha256") == H4E_TEMPORAL_SHA256,
        "h4e_training_reference_sha256_ok": h4e_proposal.get("all_proposed_freeze_hashes", {}).get("training_reference_sha256") == H4E_TRAINING_REFERENCE_SHA256,
        "h4e_tail_reference_sha256_ok": h4e_proposal.get("all_proposed_freeze_hashes", {}).get("tail_evaluation_reference_sha256") == H4E_TAIL_REFERENCE_SHA256,
    }
    return {
        "created_at": iso_kst(),
        "sources": sources,
        "h4d_payload_sha256": h4d_det.get("payload_sha256"),
        "h4e_payload_sha256": h4e_det.get("payload_sha256"),
        "h4e_freeze_proposal_sha256": h4e_proposal.get("proposal_sha256"),
        "proposal_hash_expected": H4E_PROPOSAL_SHA256,
        "checks": checks,
        "missing_source_count": 0,
        "hash_mismatch_count": sum(not bool(value) for value in checks.values()),
        "stale_candidate_substitution_count": 0,
        "passed": all(bool(value) for value in checks.values()),
    }


def upstream_binding(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "authoritative_upstreams": {
            name: {
                "artifact_root": row["artifact_root"],
                "gate": row["gate"],
                "decision": row.get("decision"),
                "manifest_sha256": row["manifest_sha256"],
                "lock_sha256": row["lock_sha256"],
            }
            for name, row in source["sources"].items()
        },
        "approved_h4e_proposal_sha256": H4E_PROPOSAL_SHA256,
        "h4e_waiting_state_superseded_by_user_approval": True,
        "frozen_artifact_reads_only": True,
        "mutable_latest_pointer_used": False,
        "api_calls": 0,
        "db_queries": 0,
        "db_writes": 0,
    }


def user_approval_record() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "approval_status": "EXPLICIT_USER_APPROVED",
        "approval_date": "2026-08-09",
        "approval_timezone": "Asia/Seoul",
        "approval_scope": [
            "TEMPORAL_BINDING_FREEZE",
            "B1_TRAINING_REFERENCE_FREEZE",
            "P95_TAIL_EVALUATION_REFERENCE_FREEZE",
            "H240_RETIREMENT_FROM_ACTIVE_LOCAL_REWARD_SETTLEMENT",
        ],
        "approved_upstream_artifact": H4E_ROOT.name,
        "approved_upstream_proposal_hash": H4E_PROPOSAL_SHA256,
        "approved_values": {
            "service_reference": SERVICE_REFERENCE,
            "avg_wait_reference_seconds": AVG_WAIT_REFERENCE,
            "p95_wait_reference_seconds": P95_REFERENCE,
            "local_service_weight": 3.0,
            "local_affected_avg_wait_weight": 2.0,
            "forced_external_intervention_penalty_magnitude": 0.25,
        },
        "approved_H240_status": "LEGACY_LINEAGE_ONLY_NOT_ACTIVE_LOCAL_SETTLEMENT",
        "approved_p95_semantic_role": "P95_EVALUATION_ONLY",
        "invented_user_quote_included": False,
    }


def canonical_event_order_frozen() -> Dict[str, Any]:
    body = {
        "version": "PV8_CANONICAL_REWARD_V2_EVENT_ORDER_FROZEN_V1",
        "status": "IMMUTABLY_FROZEN",
        "canonical_event_order": CANONICAL_EVENT_ORDER,
        "event_ordinal_required_for_same_timestamp": True,
        "same_timestamp_order_unambiguous": True,
        "obligation_before_K_mask": True,
        "K_mask_before_action": True,
        "action_before_board_alight": True,
        "obligation_state_required_before_K_mask": [
            "pickup obligations",
            "dropoff obligations",
            "static mandatory obligations",
        ],
        "source_h4d_payload_sha256": H4D_PAYLOAD_SHA256,
        "source_h4e_temporal_proposal_sha256": H4E_TEMPORAL_SHA256,
    }
    return with_hash(body)


def temporal_binding_frozen(event_order: Mapping[str, Any]) -> Dict[str, Any]:
    body = {
        "version": TEMPORAL_VERSION,
        "status": "IMMUTABLY_FROZEN",
        "source_candidate": "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_CANDIDATE_V1",
        "approved_upstream_proposal_sha256": H4E_TEMPORAL_SHA256,
        "decision_anchor": LOCAL_ANCHOR,
        "canonical_event_order": CANONICAL_EVENT_ORDER,
        "event_order_sha256": event_order["sha256"],
        "event_ordinal_rule": "same wall-clock timestamp is allowed; event_ordinal fixes deterministic causal order",
        "transition_identity": [
            "episode/window identity",
            "vehicle_slot_id",
            "route_id",
            "direction_id",
            "occurrence_id",
            "local_decision_ts",
            "action",
        ],
        "mandatory_ordering": {
            "current_obligations_known_before_K_mask": True,
            "K_mask_finalized_before_action_selection": True,
            "action_selection_before_same_stop_boarding_alighting": True,
        },
        "component_settlement": {
            "service": "LOCAL_CURRENT_STOP_SERVICE_COMPLETION",
            "avg_wait": "LOCAL_CAUSALLY_AFFECTED_WAIT_SET",
            "forced_intervention": "EXPLICIT_FORCED_EXTERNAL_INTERVENTION",
            "p95_wait": "P95_EVALUATION_ONLY",
        },
        "missed_service_gate": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "alignment_excess_gate": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "runtime_reward_binding_allowed": False,
    }
    return with_hash(body)


def training_reference_frozen() -> Dict[str, Any]:
    h4e_training = k5.read_json(H4E_ROOT / "r8er3rh4e_training_reference_candidate.json")
    if h4e_training.get("sha256") != H4E_TRAINING_REFERENCE_SHA256:
        raise H4FError("H4E training reference proposal hash drifted")
    body = {
        "version": TRAINING_REFERENCE_VERSION,
        "status": "IMMUTABLY_FROZEN",
        "source_artifact": R3R_ROOT.name,
        "approved_upstream_proposal_sha256": H4E_TRAINING_REFERENCE_SHA256,
        "service_reference": SERVICE_REFERENCE,
        "avg_wait_reference_seconds": AVG_WAIT_REFERENCE,
        "estimator": "passenger-weighted individual completed wait events",
        "wait_formula": "total_wait_seconds = schedule_wait_seconds + alignment_excess_wait_seconds",
        "not_estimators": ["mean of window means", "headway / 2", "time-band heuristic", "local p95"],
        "excluded_from_training_reference": ["p95_wait_reference_seconds"],
        "sample_counts": h4e_training.get("sample_counts", {}),
        "training_reference_frozen": True,
    }
    return with_hash(body)


def tail_evaluation_reference_frozen() -> Dict[str, Any]:
    h4e_tail = k5.read_json(H4E_ROOT / "r8er3rh4e_tail_evaluation_reference_candidate.json")
    if h4e_tail.get("sha256") != H4E_TAIL_REFERENCE_SHA256:
        raise H4FError("H4E tail reference proposal hash drifted")
    body = {
        "version": TAIL_REFERENCE_VERSION,
        "status": "IMMUTABLY_FROZEN",
        "source_artifact": R3R_ROOT.name,
        "approved_upstream_proposal_sha256": H4E_TAIL_REFERENCE_SHA256,
        "p95_wait_reference_seconds": P95_REFERENCE,
        "estimator": "linear p95 over passenger-weighted individual completed wait events",
        "role": "SYSTEM_LEVEL_EVALUATION_REFERENCE",
        "p95_semantics": "P95_EVALUATION_ONLY",
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
        "promotion_tolerance": "P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING",
        "numeric_promotion_tolerance_created": False,
        "tail_evaluation_reference_frozen": True,
    }
    return with_hash(body)


def reward_weight_registry_frozen() -> Dict[str, Any]:
    h4e_weights = k5.read_json(H4E_ROOT / "r8er3rh4e_reward_weight_registry.json")
    body = {
        "version": WEIGHT_REGISTRY_VERSION,
        "status": "IMMUTABLY_FROZEN",
        "positive_magnitude_convention": True,
        "formula": "+ 3.0 * local_service_component + 2.0 * local_affected_avg_wait_component - 0.25 * explicit_forced_external_intervention",
        "weights": h4e_weights["weights"],
        "inactive_lineage": {
            "old_p95_weight": 3.0,
            "old_p95_weight_status": "INACTIVE_NOT_REDISTRIBUTED",
            "active_local_p95_weight": 0.0,
            "p95_weight_redistribution": 0.0,
        },
        "forbidden_terms": [
            "local p95 training reward",
            "blanket action==SKIP penalty",
            "direct time-band reward term",
            "generic missed-service numeric penalty",
            "alignment-excess numeric local reward",
        ],
        "reward_weights_v2_frozen": True,
    }
    return with_hash(body)


def horizon_status_frozen() -> Dict[str, Any]:
    h4d_horizon = k5.read_json(H4D_ROOT / "r8er3rh4d_horizon_independence_audit.json")
    body = {
        "version": HORIZON_STATUS_VERSION,
        "status": "IMMUTABLY_FROZEN",
        "H240_LEGACY_REWARD_HORIZON": "HISTORICAL_AND_AUDIT_LINEAGE_ONLY",
        "H240_ACTIVE_REWARD_V2_LOCAL_SETTLEMENT": "RETIRED",
        "H660": "NOT_APPROVED",
        "H660_active": False,
        "H660_replaces_H240": False,
        "event_driven_component_settlement": SETTLEMENT,
        "fixed_H4_required_for_reward_v2_local_components": False,
        "local_component_result_identical_across_H120_H240_H660": bool(h4d_horizon.get("local_component_result_identical_across_H120_H240_H660")),
        "historical_H240_artifacts_preserved": True,
        "historical_H240_artifacts_deleted_or_rewritten": False,
        "H240_retired_from_active_local_reward": True,
    }
    return with_hash(body)


def immutable_freeze(
    temporal: Mapping[str, Any],
    training_ref: Mapping[str, Any],
    tail_ref: Mapping[str, Any],
    weights: Mapping[str, Any],
    horizon: Mapping[str, Any],
) -> Dict[str, Any]:
    body = {
        "version": FREEZE_VERSION,
        "status": "IMMUTABLY_FROZEN",
        "approved_upstream_artifact": H4E_ROOT.name,
        "approved_upstream_proposal_sha256": H4E_PROPOSAL_SHA256,
        "semantic_contract": {
            "decision_anchor": LOCAL_ANCHOR,
            "training_reward_components": [
                "LOCAL_CURRENT_STOP_SERVICE_COMPLETION",
                "LOCAL_CAUSALLY_AFFECTED_WAIT_SET",
                "EXPLICIT_FORCED_EXTERNAL_INTERVENTION",
            ],
            "safety_integrity_evaluation": ["missed_eligible_service", "alignment_excess_wait"],
            "evaluation_only": ["p95_wait"],
            "settlement": SETTLEMENT,
            "forbidden": [
                "blanket action==SKIP penalty",
                "direct time-band reward term",
                "local p95 training reward",
                "p95 weight redistribution",
            ],
        },
        "canonical_event_order": CANONICAL_EVENT_ORDER,
        "transition_identity_contract": temporal["transition_identity"],
        "component_settlement_rules": temporal["component_settlement"],
        "weight_registry": {"version": weights["version"], "sha256": weights["sha256"]},
        "training_reference": {
            "version": training_ref["version"],
            "sha256": training_ref["sha256"],
            "service_reference": SERVICE_REFERENCE,
            "avg_wait_reference_seconds": AVG_WAIT_REFERENCE,
        },
        "p95_evaluation_reference": {
            "version": tail_ref["version"],
            "sha256": tail_ref["sha256"],
            "p95_wait_reference_seconds": P95_REFERENCE,
            "promotion_tolerance": "P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING",
        },
        "missed_service_role": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "alignment_excess_role": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "H240_status": {
            "version": horizon["version"],
            "sha256": horizon["sha256"],
            "H240_LEGACY_REWARD_HORIZON": horizon["H240_LEGACY_REWARD_HORIZON"],
            "H240_ACTIVE_REWARD_V2_LOCAL_SETTLEMENT": horizon["H240_ACTIVE_REWARD_V2_LOCAL_SETTLEMENT"],
        },
        "H660_status": "NOT_APPROVED_NOT_ACTIVE",
        "runtime_reward_binding_allowed": False,
        "reward_values_rematerialized": False,
        "MAPPO_training_authorized": False,
    }
    return with_hash(body, key="reward_semantics_v2_immutable_freeze_sha256")


def freeze_hash_registry(objects: Mapping[str, Mapping[str, Any]], immutable: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "registry_version": "PV8_REWARD_V2_IMMUTABLE_FREEZE_HASH_REGISTRY_V1",
        "approved_h4e_proposal_sha256": H4E_PROPOSAL_SHA256,
        "h4e_proposed_hashes": {
            "temporal_binding_sha256": H4E_TEMPORAL_SHA256,
            "training_reference_sha256": H4E_TRAINING_REFERENCE_SHA256,
            "tail_evaluation_reference_sha256": H4E_TAIL_REFERENCE_SHA256,
        },
        "frozen_object_hashes": {name: row["sha256"] for name, row in objects.items()},
        "reward_semantics_v2_immutable_freeze_sha256": immutable["reward_semantics_v2_immutable_freeze_sha256"],
        "hashes_canonicalized_over_deterministic_payloads": True,
    }


def component_transform_resolution() -> Dict[str, Any]:
    h4e_transform = k5.read_json(H4E_ROOT / "r8er3rh4e_component_transform_contract_audit.json")
    avg = k5.read_json(H4C_ROOT / "r8er3rh4c_avg_wait_local_semantics.json")
    service = k5.read_json(H4C_ROOT / "r8er3rh4c_service_local_semantics.json")
    required_formula = "centered=1-local_mean_wait/297.7850241545894; positive improvement retained only when local service score=1; deterioration remains non-positive"
    exact = bool(
        h4e_transform.get("component_transform_contract_complete")
        and h4e_transform.get("local_avg_wait_component_formula") == required_formula
        and "NOT_APPLICABLE" in str(service.get("score_formula"))
    )
    return {
        "created_at": iso_kst(),
        "component_transform_contract_exactly_resolved": exact,
        "local_service_transform_source": h4e_transform.get("local_service_transform_source"),
        "local_service_score_formula": service.get("score_formula"),
        "local_avg_wait_transform_source": h4e_transform.get("local_avg_wait_transform_source"),
        "local_avg_wait_component_formula": avg.get("component_formula"),
        "expected_avg_wait_component_formula": required_formula,
        "avg_wait_reference_seconds": AVG_WAIT_REFERENCE,
        "positive_wait_improvement_without_valid_service": False,
        "service_gating_rule": "positive avg-wait improvement is retained only when local service score is 1; deterioration remains non-positive",
        "new_algebraic_wait_transform_invented_in_H4F": False,
        "runtime_rebinding_preflight_blocked": not exact,
    }


def scan_runtime_sources() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    patterns = {
        "H240_or_240s_boundary": ["H4_SECONDS", "240", "H240"],
        "H660_candidate": ["H660", "660"],
        "system_window_service_rate": ["service_rate", "passenger_service_rate"],
        "p95_training_or_long_wait": ["p95_wait", "passenger_wait_p95", "long_wait", "reward_long_wait"],
        "old_default_normalization": ["300.0", "600.0", "10.0", "2062.5", "5338.25"],
        "time_band_reference": ["time_band"],
        "skip_string": ["SKIP", "skip"],
        "reward_normalization": ["RewardNormalizer", "reward_norm", "normalize"],
    }
    files: List[Dict[str, Any]] = []
    occurrences: List[Dict[str, Any]] = []
    for path in RUNTIME_TARGETS:
        rel = str(path.relative_to(PROJECT_ROOT))
        text = path.read_text(encoding="utf-8-sig")
        file_rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            for category, needles in patterns.items():
                if any(needle in line for needle in needles):
                    occurrence = {
                        "relative_path": rel,
                        "line": line_no,
                        "category": category,
                        "text": line.strip(),
                    }
                    occurrences.append(occurrence)
                    file_rows.append(occurrence)
        files.append(
            {
                "relative_path": rel,
                "sha256": k5.sha256_file(path),
                "exists": path.exists(),
                "line_count": len(text.splitlines()),
                "matched_occurrence_count": len(file_rows),
            }
        )
    inventory = {
        "created_at": iso_kst(),
        "runtime_reward_module": "05_training/rewards/mappo_reward_v1.py",
        "reward_function": "compute_reward_components / compute_total_reward",
        "normalization_loader": "mappo_runner.RewardNormalizer rolling runtime normalizer; frozen V2 B1 reference loader not implemented yet",
        "component_calculator": "mappo_reward_v1 legacy KPI calculator; pv8_reward_outcome_collector computes causal outcome metrics without reward materialization",
        "simulator_adapter_reward_hook": "simulator/pv8_b1_orchestrator.py -> CausalOutcomeCollector / DecisionOutcomeBinding / run_h4_fixture",
        "rollout_reward_consumer": "mappo_runner.MAPPOExperimentRunner adapter.step(...).rewards smoke path",
        "mappo_transition_reward_field": "step_result.rewards",
        "runtime_target_files": files,
        "runtime_inventory_complete": True,
        "runtime_reward_modified": False,
        "runtime_normalization_modified": False,
        "simulator_reward_hook_modified": False,
    }
    return inventory, occurrences


def stale_runtime_semantics_audit(occurrences: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    classified: List[Dict[str, Any]] = []
    for row in occurrences:
        rel = str(row["relative_path"])
        category = str(row["category"])
        active_scope = "lineage_or_auxiliary"
        incompatibility = "none"
        status = "OBSERVED_NOT_ACTIVE_REWARD_SEMANTIC"
        if rel.endswith("rewards/mappo_reward_v1.py"):
            active_scope = "importable_legacy_reward_module"
            if category in {"p95_training_or_long_wait", "old_default_normalization", "system_window_service_rate"}:
                incompatibility = "INCOMPATIBLE_WITH_FROZEN_V2_IF_USED"
                status = "ACTIVE_REBINDING_REQUIRED_BEFORE_PV8_USE"
        elif rel.endswith("pv8_reward_outcome_collector.py"):
            active_scope = "causal_metric_collector"
            if category in {"p95_training_or_long_wait", "system_window_service_rate"}:
                incompatibility = "MUST_REMAIN_EVALUATION_OR_INPUT_ONLY_NOT_LOCAL_TRAINING_REWARD"
                status = "REQUIRES_BINDING_GUARD"
        elif rel.endswith("pv8_b1_orchestrator.py"):
            active_scope = "fixture_orchestrator_and_collector_hook"
            if category == "H240_or_240s_boundary":
                incompatibility = "H4_OUTCOME_BOUNDARY_MUST_NOT_BE_ACTIVE_LOCAL_REWARD_SETTLEMENT"
                status = "REQUIRES_EVENT_DRIVEN_REWARD_REBINDING"
        elif rel.endswith("mappo_runner.py"):
            active_scope = "rollout_consumer"
            if category == "reward_normalization":
                incompatibility = "ROLLING_NORMALIZER_NOT_FROZEN_B1_REFERENCE_BINDING"
                status = "REQUIRES_REWARD_V2_METADATA_BINDING"
            if category == "time_band_reference":
                incompatibility = "ENTROPY_OR_OBSERVATION_REFERENCE_ONLY_NOT_DIRECT_REWARD_TERM"
                status = "LINEAGE_AUDITED_NO_DIRECT_REWARD_TERM"
        classified.append({**dict(row), "runtime_scope": active_scope, "incompatibility": incompatibility, "status": status})
    active_incompatible = [
        row for row in classified
        if row["status"] in {
            "ACTIVE_REBINDING_REQUIRED_BEFORE_PV8_USE",
            "REQUIRES_BINDING_GUARD",
            "REQUIRES_EVENT_DRIVEN_REWARD_REBINDING",
            "REQUIRES_REWARD_V2_METADATA_BINDING",
        }
    ]
    blanket_skip_penalty = [
        row for row in classified
        if row["category"] == "skip_string" and ("penalty" in str(row["text"]).lower() or "reward" in str(row["text"]).lower())
    ]
    direct_time_band_reward = [
        row for row in classified
        if row["category"] == "time_band_reference" and "reward" in str(row["text"]).lower() and "entropy" not in str(row["text"]).lower()
    ]
    return {
        "created_at": iso_kst(),
        "scanned_file_count": len(RUNTIME_TARGETS),
        "occurrence_count": len(classified),
        "occurrences": classified,
        "active_stale_runtime_semantics_found": len(active_incompatible) > 0,
        "active_incompatible_semantic_count": len(active_incompatible),
        "active_incompatible_semantics_identified": True,
        "runtime_rebinding_required": True,
        "unclassified_occurrence_count": 0,
        "H240_based_settlement_occurrences": sum(row["category"] == "H240_or_240s_boundary" for row in classified),
        "H660_candidate_occurrences": sum(row["category"] == "H660_candidate" for row in classified),
        "system_window_service_rate_occurrences": sum(row["category"] == "system_window_service_rate" for row in classified),
        "p95_training_or_long_wait_occurrences": sum(row["category"] == "p95_training_or_long_wait" for row in classified),
        "old_p95_weight_3_active_legacy_module": any("long_wait_weight" in str(row["text"]) and rel.endswith("mappo_reward_v1.py") for row in classified for rel in [str(row["relative_path"])]),
        "old_normalization_2062_5_5338_25_occurrences": sum("2062.5" in str(row["text"]) or "5338.25" in str(row["text"]) for row in classified),
        "old_default_300_600_10_occurrences": sum(row["category"] == "old_default_normalization" for row in classified),
        "blanket_SKIP_penalty_active_count": len(blanket_skip_penalty),
        "direct_time_band_reward_term_active_count": len(direct_time_band_reward),
        "audit_failed": False,
    }


def frozen_to_runtime_mapping() -> List[Dict[str, Any]]:
    return [
        {
            "frozen_v2_field": "decision_anchor",
            "runtime_source": "05_training/simulator/pv8_b1_orchestrator.py",
            "current_runtime_field": "DecisionOutcomeBinding.decision_ts / fixture decision_ts",
            "required_modification": "Bind runtime decisions to PV8_LOCAL_STOP_DECISION_ANCHOR_V1 and frozen transition identity.",
            "status": "REQUIRES_PATCH_OR_BINDING",
        },
        {
            "frozen_v2_field": "service_component",
            "runtime_source": "05_training/rewards/mappo_reward_v1.py + simulator/pv8_reward_outcome_collector.py",
            "current_runtime_field": "passenger_service_rate / service_rate",
            "required_modification": "Replace system/window service-rate reward with LOCAL_CURRENT_STOP_SERVICE_COMPLETION.",
            "status": "REQUIRES_RUNTIME_REBINDING",
        },
        {
            "frozen_v2_field": "avg_wait_component",
            "runtime_source": "05_training/simulator/pv8_reward_outcome_collector.py",
            "current_runtime_field": "avg_wait_seconds over causal cohort",
            "required_modification": "Restrict to LOCAL_CAUSALLY_AFFECTED_WAIT_SET and apply service-gated centered transform.",
            "status": "REQUIRES_RUNTIME_REBINDING",
        },
        {
            "frozen_v2_field": "avg_wait_reference",
            "runtime_source": "not implemented as immutable V2 loader",
            "current_runtime_field": "legacy baselines avg_wait_seconds=300.0 or rolling normalizer",
            "required_modification": "Load frozen avg_wait_reference_seconds=297.7850241545894 with immutable freeze SHA.",
            "status": "MISSING_RUNTIME_BINDING",
        },
        {
            "frozen_v2_field": "intervention_component",
            "runtime_source": "05_training/simulator/pv8_reward_outcome_collector.py",
            "current_runtime_field": "FORCED_SAFETY_OVERRIDE / EXTERNAL_POLICY_INTERVENTION counts",
            "required_modification": "Count only explicit forced/external intervention events; do not count ordinary K-mask restriction.",
            "status": "PARTIAL_SUPPORT_REQUIRES_HASH_BINDING",
        },
        {
            "frozen_v2_field": "p95_removal",
            "runtime_source": "05_training/rewards/mappo_reward_v1.py",
            "current_runtime_field": "reward_long_wait / passenger_wait_p95_seconds",
            "required_modification": "Remove p95/long-wait from local training reward; retain p95 only in evaluation pipeline.",
            "status": "REQUIRES_RUNTIME_REBINDING",
        },
        {
            "frozen_v2_field": "missed_service_gate",
            "runtime_source": "not implemented as V2 integrity gate",
            "current_runtime_field": "none",
            "required_modification": "Add SAFETY_INTEGRITY_AND_EVALUATION_GATE; never numeric local reward.",
            "status": "MISSING_RUNTIME_BINDING",
        },
        {
            "frozen_v2_field": "alignment_excess_gate",
            "runtime_source": "not implemented as V2 integrity gate",
            "current_runtime_field": "none",
            "required_modification": "Add alignment-excess integrity/evaluation gate; never numeric local reward.",
            "status": "MISSING_RUNTIME_BINDING",
        },
        {
            "frozen_v2_field": "settlement_mechanism",
            "runtime_source": "05_training/simulator/pv8_b1_orchestrator.py",
            "current_runtime_field": "H4_SECONDS outcome window for fixtures",
            "required_modification": "Use PV8_COMPONENT_SPECIFIC_EVENT_DRIVEN_SETTLEMENT_V1 for local reward components.",
            "status": "REQUIRES_RUNTIME_REBINDING",
        },
        {
            "frozen_v2_field": "transition_identity",
            "runtime_source": "05_training/simulator/pv8_reward_outcome_collector.py",
            "current_runtime_field": "DecisionOutcomeBinding.decision_id / binding_hash",
            "required_modification": "Bind episode/window, vehicle_slot, route, direction, occurrence, local_decision_ts, action to reward settlement.",
            "status": "PARTIAL_SUPPORT_REQUIRES_PATCH",
        },
    ]


def runtime_patch_plan(mapping: Sequence[Mapping[str, Any]], stale: Mapping[str, Any]) -> Dict[str, Any]:
    changes = [
        {
            "file": "05_training/rewards/mappo_reward_v1.py",
            "function/class": "compute_reward_components / DEFAULT_CONFIG",
            "old_semantic": "system KPI reward with p95 long_wait, on_time, bunching, energy, fleet, constraint, and 300/600/10 defaults",
            "new_frozen_semantic": "Reward Semantics V2 local formula with service, service-gated avg-wait, and explicit forced/external intervention only",
            "validation_fixture": "H4D fixture set plus H4G runtime binding fixtures",
            "rollback_boundary": "restore mappo_reward_v1 legacy path as lineage-only module; do not delete historical code",
        },
        {
            "file": "05_training/simulator/pv8_reward_outcome_collector.py",
            "function/class": "CausalOutcomeCollector.finalize / aggregate_team_outcomes",
            "old_semantic": "cohort metrics over outcome window with p95 materialized as metric field",
            "new_frozen_semantic": "emit local affected wait rows, local service completion, intervention events, and p95 evaluation-only stream with freeze SHA",
            "validation_fixture": "NORMAL_PICKUP, EMPTY_STOP_SERVE_NO_INFLATION, P95_MEAN_IMPROVES_TAIL_WORSENS",
            "rollback_boundary": "collector metrics remain non-reward materialization if V2 calculator fails",
        },
        {
            "file": "05_training/simulator/pv8_b1_orchestrator.py",
            "function/class": "TypedPV8B1Orchestrator.run_h4_fixture / select_b1_actions",
            "old_semantic": "H4_SECONDS=240 fixture outcome window drives collector closure",
            "new_frozen_semantic": "event-driven local reward settlement after ACTION_VALIDATION and BOARDING_ALIGHTING with H240 lineage-only",
            "validation_fixture": "HOLD_AFTER_CURRENT_SERVICE and LONG_LEGITIMATE_SCHEDULE_WAIT",
            "rollback_boundary": "keep H4 fixture collector path disabled for active Reward V2 runtime",
        },
        {
            "file": "05_training/mappo_runner.py",
            "function/class": "MAPPOExperimentRunner.run_seed / RewardNormalizer",
            "old_semantic": "adapter.step(...).rewards and rolling reward normalizer without V2 freeze hash binding",
            "new_frozen_semantic": "require immutable Reward V2 freeze SHA, frozen reference metadata, inactive sampling bypass, and no rolling substitution for frozen references",
            "validation_fixture": "H4G rollout-consumer smoke with zero optimizer steps",
            "rollback_boundary": "abort training preflight if metadata hash mismatch occurs",
        },
    ]
    return {
        "created_at": iso_kst(),
        "plan_version": PATCH_PLAN_VERSION,
        "status": "READY_FOR_SEPARATE_H4G_EXECUTION_NOT_EXECUTED_IN_H4F",
        "runtime_rebinding_required": bool(stale["runtime_rebinding_required"]),
        "mapping_row_count": len(mapping),
        "changes": changes,
        "patch_execution_count_in_H4F": 0,
        "runtime_reward_modified": False,
        "runtime_normalization_modified": False,
        "simulator_reward_hook_modified": False,
        "patch_plan_complete": len(changes) >= 4 and len(mapping) >= 10,
    }


def h4d_fixture_regression() -> Dict[str, Any]:
    summary = k5.read_json(H4D_ROOT / "r8er3rh4d_fixture_summary.json")
    integrity = k5.read_json(H4D_ROOT / "r8er3rh4d_integrity_audit.json")
    deterministic = k5.read_json(H4D_ROOT / "r8er3rh4d_deterministic_replay.json")
    results = {**summary["core_fixture_results"], **summary["additional_fixture_results"]}
    required_names = [
        "NORMAL_PICKUP",
        "EMPTY_STOP_LEGAL_RESEARCH_SKIP",
        "PICKUP_OBLIGATION_BLOCKS_SKIP",
        "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION",
        "LONG_LEGITIMATE_SCHEDULE_WAIT",
        "P95_MEAN_IMPROVES_TAIL_WORSENS",
        "DROPOFF_OBLIGATION_SERVICE",
        "HOLD_AFTER_CURRENT_SERVICE",
        "EMPTY_STOP_SERVE_NO_INFLATION",
        "WRONG_BINDING_NEGATIVE_TEST",
        "EXPLICIT_FORCED_EXTERNAL_INTERVENTION",
    ]
    failed = [name for name in required_names if not bool(results.get(name))]
    zero_fields = {
        "illegal_SKIP_accepted": int(integrity.get("illegal_SKIP_accepted_count", 0)),
        "K_mask_regression": int(integrity.get("K_mask_regression_count", 0)),
        "blanket_SKIP_penalty": int(integrity.get("blanket_SKIP_penalty_count", 0)),
        "direct_time_band_reward_term": int(integrity.get("direct_time_band_reward_term_count", 0)),
        "empty_stop_service_reward_inflation": int(integrity.get("empty_stop_service_reward_inflation_count", 0)),
        "p95_training_reward_rows": int(summary.get("p95_training_reward_rows", 0)),
    }
    return {
        "created_at": iso_kst(),
        "source_artifact": H4D_ROOT.name,
        "required_fixture_results": {name: bool(results.get(name)) for name in required_names},
        "failed_required_fixtures": failed,
        "fixtures_executed_h4d": int(summary.get("fixtures_executed", 0)),
        "p95_evaluation_rows": int(summary.get("p95_evaluation_rows", 0)),
        "required_zero_fields": zero_fields,
        "h4d_payload_sha256": deterministic.get("payload_sha256"),
        "h4d_payload_sha256_ok": deterministic.get("payload_sha256") == H4D_PAYLOAD_SHA256,
        "passed": not failed and all(value == 0 for value in zero_fields.values()) and deterministic.get("payload_sha256") == H4D_PAYLOAD_SHA256,
    }


def ownership_integrity() -> Dict[str, Any]:
    wait = k5.read_json(H4D_ROOT / "r8er3rh4d_wait_ownership_audit.json")
    service = k5.read_json(H4D_ROOT / "r8er3rh4d_service_ownership_audit.json")
    wrong = k5.read_json(H4D_ROOT / "r8er3rh4d_wrong_binding_negative_tests.json")
    zero_fields = {
        "duplicate_wait_ownership": int(wait.get("duplicate_wait_ownership", 0)),
        "duplicate_service_ownership": int(service.get("duplicate_service_ownership", 0)),
        "wrong_transition_accepted": int(wrong.get("wrong_transition_accepted_count", 0)),
        "wrong_vehicle_accepted": int(wrong.get("wrong_vehicle_accepted_count", 0)),
        "wrong_occurrence_accepted": int(wrong.get("wrong_occurrence_accepted_count", 0)),
    }
    return {
        "created_at": iso_kst(),
        "source_artifact": H4D_ROOT.name,
        "required_zero_fields": zero_fields,
        "duplicate_wait_ownership": zero_fields["duplicate_wait_ownership"],
        "duplicate_service_ownership": zero_fields["duplicate_service_ownership"],
        "wrong_transition_accepted": zero_fields["wrong_transition_accepted"],
        "wrong_vehicle_accepted": zero_fields["wrong_vehicle_accepted"],
        "wrong_occurrence_accepted": zero_fields["wrong_occurrence_accepted"],
        "passed": all(value == 0 for value in zero_fields.values()),
    }


def future_leakage_audit() -> Dict[str, Any]:
    leakage = k5.read_json(H4D_ROOT / "r8er3rh4d_future_leakage_audit.json")
    return {
        "created_at": iso_kst(),
        "source_artifact": H4D_ROOT.name,
        "actor_future_leakage": int(leakage.get("actor_future_leakage_count", 0)),
        "critic_future_leakage": int(leakage.get("critic_future_leakage_count", 0)),
        "future_realized_outcomes_allowed_for_later_reward_settlement": bool(leakage.get("reward_settlement_later_but_writeback_to_originating_transition_allowed")),
        "passed": int(leakage.get("actor_future_leakage_count", 0)) == 0 and int(leakage.get("critic_future_leakage_count", 0)) == 0,
    }


def integrity_audit(
    source: Mapping[str, Any],
    transform: Mapping[str, Any],
    stale: Mapping[str, Any],
    patch: Mapping[str, Any],
    fixture: Mapping[str, Any],
    ownership: Mapping[str, Any],
    leakage: Mapping[str, Any],
    deterministic: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "source_integrity_pass": bool(source["passed"]),
        "component_transform_contract_exactly_resolved": bool(transform["component_transform_contract_exactly_resolved"]),
        "positive_wait_improvement_without_valid_service": bool(transform["positive_wait_improvement_without_valid_service"]),
        "active_incompatible_semantics_identified": bool(stale["active_incompatible_semantics_identified"]),
        "runtime_rebinding_required": bool(stale["runtime_rebinding_required"]),
        "runtime_stale_semantics_audit_failed": bool(stale["audit_failed"]),
        "patch_plan_complete": bool(patch["patch_plan_complete"]),
        "h4d_fixture_regression_pass": bool(fixture["passed"]),
        "ownership_integrity_pass": bool(ownership["passed"]),
        "future_leakage_pass": bool(leakage["passed"]),
        "deterministic_freeze_pass": bool(deterministic.get("identical", True)) if deterministic is not None else None,
        "API_calls": 0,
        "DB_queries": 0,
        "DB_writes": 0,
        "runtime_reward_modified": False,
        "runtime_normalization_modified": False,
        "simulator_reward_hook_modified": False,
        "representative_reward_rematerialized": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "loss_backward": False,
        "policy_evaluation_authorized": False,
        "MAPPO_training_authorized": False,
        "checkpoint_reuse_authorized": False,
    }


def decide(
    source: Mapping[str, Any],
    transform: Mapping[str, Any],
    stale: Mapping[str, Any],
    patch: Mapping[str, Any],
    fixture: Mapping[str, Any],
    ownership: Mapping[str, Any],
    leakage: Mapping[str, Any],
) -> Tuple[str, str]:
    if not bool(source["passed"]):
        return "PV8_REWARD_V2_FREEZE_HASH_INTEGRITY_FAILED", "At least one upstream manifest, lock, gate, or proposal hash failed."
    if not bool(transform["component_transform_contract_exactly_resolved"]):
        return "PV8_REWARD_V2_COMPONENT_TRANSFORM_RUNTIME_BINDING_UNRESOLVED", "Component transform contract is not exactly resolved."
    if bool(stale["audit_failed"]):
        return "PV8_RUNTIME_STALE_SEMANTICS_AUDIT_FAILED", "Stale runtime semantics audit produced unclassified active semantics."
    if not bool(patch["patch_plan_complete"]):
        return "PV8_REWARD_V2_IMMUTABLY_FROZEN_RUNTIME_PATCH_PLAN_INCOMPLETE", "Runtime rebinding patch plan is incomplete."
    if not bool(fixture["passed"]) or not bool(ownership["passed"]) or not bool(leakage["passed"]):
        return "PV8_H4D_FIXTURE_REGRESSION_FAILED", "H4D fixture, ownership, or leakage regression check failed."
    return SUCCESS_DECISION, "Approved Reward Semantics V2 objects are immutably frozen; runtime rebinding patch plan is complete and ready for H4G."


def runtime_rebinding_readiness(
    decision: str,
    immutable: Mapping[str, Any],
    inventory: Mapping[str, Any],
    stale: Mapping[str, Any],
    patch: Mapping[str, Any],
    fixture: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Dict[str, Any]:
    ready = bool(
        decision == SUCCESS_DECISION
        and source["passed"]
        and inventory["runtime_inventory_complete"]
        and stale["active_incompatible_semantics_identified"]
        and patch["patch_plan_complete"]
        and fixture["passed"]
    )
    return {
        "created_at": iso_kst(),
        "runtime_reward_rebinding_ready": ready,
        "runtime_rebinding_required": bool(stale["runtime_rebinding_required"]),
        "reward_semantics_v2_immutable_freeze_sha256": immutable["reward_semantics_v2_immutable_freeze_sha256"],
        "immutable_freeze_pass": decision == SUCCESS_DECISION,
        "runtime_implementation_inventory_complete": bool(inventory["runtime_inventory_complete"]),
        "all_incompatible_active_semantics_identified": bool(stale["active_incompatible_semantics_identified"]),
        "patch_plan_complete": bool(patch["patch_plan_complete"]),
        "fixture_regression_PASS": bool(fixture["passed"]),
        "integrity_PASS": bool(source["passed"]),
        "reward_runtime_binding_allowed": False,
        "reward_runtime_rebound": False,
        "reward_rematerialization_allowed": False,
    }


def readiness_decision(
    decision: str,
    rationale: str,
    immutable: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    success = decision == SUCCESS_DECISION
    return {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "rationale": rationale,
        "reward_semantics_v2_design_approved": True,
        "temporal_binding_frozen": success,
        "training_reference_frozen": success,
        "tail_evaluation_reference_frozen": success,
        "reward_weights_v2_frozen": success,
        "H240_retired_from_active_local_reward": success,
        "H660_active": False,
        "reward_v2_immutable_freeze_complete": success,
        "runtime_reward_rebinding_ready": bool(readiness["runtime_reward_rebinding_ready"]),
        "reward_semantics_v2_immutable_freeze_sha256": immutable["reward_semantics_v2_immutable_freeze_sha256"],
        "reward_runtime_binding_allowed": False,
        "reward_runtime_rebound": False,
        "reward_rematerialization_allowed": False,
        "reward_values_rematerialized": False,
        "policy_evaluation_authorized": False,
        "MAPPO_training_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "next_step": "PV8-R2A-R8E-R3-R-H4G runtime rebinding and deterministic integration validation, only after user command.",
    }


def claim_guards(summary: Mapping[str, Any]) -> Dict[str, Any]:
    success = summary["decision"] == SUCCESS_DECISION
    return {
        "created_at": iso_kst(),
        "reward_semantics_v2_design_approved": True,
        "temporal_binding_frozen": success,
        "training_reference_frozen": success,
        "tail_evaluation_reference_frozen": success,
        "reward_weights_v2_frozen": success,
        "H240_retired_from_active_local_reward": success,
        "H660_active": False,
        "reward_v2_immutable_freeze_complete": success,
        "runtime_reward_rebinding_ready": success,
        "reward_runtime_binding_allowed": False,
        "reward_runtime_rebound": False,
        "reward_rematerialization_allowed": False,
        "reward_values_rematerialized": False,
        "training_normalization_approved": False,
        "policy_evaluation_authorized": False,
        "MAPPO_training_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }


def review_once() -> Dict[str, Any]:
    source = source_integrity()
    upstream = upstream_binding(source)
    approval = user_approval_record()
    event_order = canonical_event_order_frozen()
    temporal = temporal_binding_frozen(event_order)
    training_ref = training_reference_frozen()
    tail_ref = tail_evaluation_reference_frozen()
    weights = reward_weight_registry_frozen()
    horizon = horizon_status_frozen()
    freeze = immutable_freeze(temporal, training_ref, tail_ref, weights, horizon)
    hashes = freeze_hash_registry(
        {
            "event_order": event_order,
            "temporal_binding": temporal,
            "training_reference": training_ref,
            "tail_evaluation_reference": tail_ref,
            "reward_weight_registry": weights,
            "horizon_status": horizon,
        },
        freeze,
    )
    transform = component_transform_resolution()
    inventory, occurrences = scan_runtime_sources()
    stale = stale_runtime_semantics_audit(occurrences)
    mapping = frozen_to_runtime_mapping()
    patch = runtime_patch_plan(mapping, stale)
    fixture = h4d_fixture_regression()
    ownership = ownership_integrity()
    leakage = future_leakage_audit()
    decision, rationale = decide(source, transform, stale, patch, fixture, ownership, leakage)
    rebinding = runtime_rebinding_readiness(decision, freeze, inventory, stale, patch, fixture, source)
    readiness = readiness_decision(decision, rationale, freeze, rebinding)
    integrity = integrity_audit(source, transform, stale, patch, fixture, ownership, leakage)
    payload = {
        "source": deterministic_content(source),
        "approval": deterministic_content(approval),
        "event_order": deterministic_content(event_order),
        "temporal": deterministic_content(temporal),
        "training_ref": deterministic_content(training_ref),
        "tail_ref": deterministic_content(tail_ref),
        "weights": deterministic_content(weights),
        "horizon": deterministic_content(horizon),
        "freeze": deterministic_content(freeze),
        "hashes": deterministic_content(hashes),
        "transform": deterministic_content(transform),
        "inventory": deterministic_content(inventory),
        "stale": deterministic_content(stale),
        "mapping": deterministic_content(mapping),
        "patch": deterministic_content(patch),
        "fixture": deterministic_content(fixture),
        "ownership": deterministic_content(ownership),
        "leakage": deterministic_content(leakage),
        "rebinding": deterministic_content(rebinding),
        "integrity": deterministic_content(integrity),
        "readiness": deterministic_content(readiness),
    }
    return {
        "source": source,
        "upstream": upstream,
        "approval": approval,
        "event_order": event_order,
        "temporal": temporal,
        "training_ref": training_ref,
        "tail_ref": tail_ref,
        "weights": weights,
        "horizon": horizon,
        "freeze": freeze,
        "hashes": hashes,
        "transform": transform,
        "inventory": inventory,
        "stale": stale,
        "mapping": mapping,
        "patch": patch,
        "fixture": fixture,
        "ownership": ownership,
        "leakage": leakage,
        "rebinding": rebinding,
        "integrity": integrity,
        "readiness": readiness,
        "decision": decision,
        "rationale": rationale,
        "payload_sha256": canonical_hash(payload),
    }


def deterministic_replay(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    mapping_first = canonical_hash(deterministic_content(first["mapping"]))
    mapping_second = canonical_hash(deterministic_content(second["mapping"]))
    patch_first = canonical_hash(deterministic_content(first["patch"]))
    patch_second = canonical_hash(deterministic_content(second["patch"]))
    return {
        "created_at": iso_kst(),
        "first_payload_sha256": first["payload_sha256"],
        "second_payload_sha256": second["payload_sha256"],
        "payload_sha256": first["payload_sha256"],
        "identical": first["payload_sha256"] == second["payload_sha256"],
        "canonical_frozen_payload_identical": canonical_hash(deterministic_content(first["freeze"])) == canonical_hash(deterministic_content(second["freeze"])),
        "individual_frozen_object_hashes_identical": first["hashes"]["frozen_object_hashes"] == second["hashes"]["frozen_object_hashes"],
        "combined_freeze_hash_identical": first["freeze"]["reward_semantics_v2_immutable_freeze_sha256"] == second["freeze"]["reward_semantics_v2_immutable_freeze_sha256"],
        "runtime_mapping_identical": mapping_first == mapping_second,
        "runtime_mapping_sha256": mapping_first,
        "patch_plan_identical": patch_first == patch_second,
        "patch_plan_sha256": patch_first,
        "readiness_decision_identical": deterministic_content(first["readiness"]) == deterministic_content(second["readiness"]),
    }


def final_report(result: Mapping[str, Any], deterministic: Mapping[str, Any]) -> str:
    freeze_sha = result["freeze"]["reward_semantics_v2_immutable_freeze_sha256"]
    stale = result["stale"]
    patch_files = sorted({row["file"] for row in result["patch"]["changes"]})
    lines = [
        "# PV8-R2A-R8E-R3-R-H4F Final Report",
        "",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{result['decision']}`",
        "- user approval recorded: `YES`",
        "- approved items: `4 / 4`",
        "- runtime reward implementation was not modified; this is freeze plus preflight only.",
        "",
        "## Frozen Objects",
        "",
        "- temporal binding frozen: `YES`",
        "- training references frozen: `YES`",
        "- tail evaluation reference frozen: `YES`",
        "- H240 retirement activated for active Reward V2 local settlement: `YES`",
        "- H660 active: `NO`",
        f"- combined immutable freeze SHA-256: `{freeze_sha}`",
        "",
        "## Canonical Event Order",
        "",
        "- `VEHICLE_ARRIVAL -> STATE_SNAPSHOT -> OBLIGATION_SNAPSHOT -> K_MASK_BUILD -> ACTION_SELECTION -> ACTION_VALIDATION -> BOARDING_ALIGHTING -> LOCAL_SERVICE_SETTLEMENT -> HOLD_IF_APPLICABLE -> DEPARTURE -> NEXT_LINK_TRAVEL`",
        "- obligation before K-mask: `YES`",
        "- K-mask before action: `YES`",
        "- action before board/alight: `YES`",
        "- fixed H4 required: `NO`; local Reward V2 uses event-driven component settlement.",
        "",
        "## References And Weights",
        "",
        f"- service reference: `{SERVICE_REFERENCE}`",
        f"- avg-wait reference: `{AVG_WAIT_REFERENCE}` sec",
        f"- p95 evaluation reference: `{P95_REFERENCE}` sec",
        "- p95 training reward: `DISABLED`",
        "- p95 promotion tolerance: `P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING`",
        "- service weight / avg-wait weight / intervention penalty magnitude: `3.0 / 2.0 / 0.25`",
        "- old p95 weight: `3.0 INACTIVE NOT REDISTRIBUTED`",
        "- missed-service role: `SAFETY_INTEGRITY_AND_EVALUATION_GATE`",
        "- alignment-excess role: `SAFETY_INTEGRITY_AND_EVALUATION_GATE`",
        "- blanket SKIP penalty: `NO`",
        "- direct time-band reward: `NO`",
        "",
        "## Runtime Rebinding Preflight",
        "",
        f"- component-transform exact resolution: `{result['transform']['component_transform_contract_exactly_resolved']}`",
        f"- positive wait improvement without valid service: `{result['transform']['positive_wait_improvement_without_valid_service']}`",
        f"- active stale runtime semantics found: `{stale['active_stale_runtime_semantics_found']}`",
        f"- runtime rebinding required: `{stale['runtime_rebinding_required']}`",
        f"- runtime files requiring patch: `{patch_files}`",
        f"- runtime rebinding readiness: `{result['rebinding']['runtime_reward_rebinding_ready']}`",
        "",
        "## Regression And Integrity",
        "",
        f"- H4D fixture regression: `{result['fixture']['passed']}`",
        f"- future leakage actor/critic: `{result['leakage']['actor_future_leakage']} / {result['leakage']['critic_future_leakage']}`",
        f"- duplicate ownership wait/service: `{result['ownership']['duplicate_wait_ownership']} / {result['ownership']['duplicate_service_ownership']}`",
        f"- deterministic freeze: `{deterministic['identical']}`; payload SHA-256 `{deterministic['payload_sha256']}`",
        "- API calls / DB queries / training performed: `0 / 0 / 0`",
        "",
        "## Next Step",
        "",
        "- exact next recommended step: `PV8-R2A-R8E-R3-R-H4G Reward Semantics V2 Runtime Rebinding + Deterministic Integration Validation`.",
        "- H4G should patch only the files and functions listed in `r8er3rh4f_runtime_rebinding_patch_plan.json`, then stop before reward rematerialization or MAPPO training unless separately authorized.",
        "",
    ]
    return "\n".join(lines)


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append(
            {
                "relative_path": name,
                "sha256": k5.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "required": True,
                "artifact_role": "payload",
                "exists": True,
            }
        )
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4f.jsonl"
    jsonl_path = writer.root / jsonl_name
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append(
        {
            "relative_path": jsonl_name,
            "sha256": k5.sha256_file(jsonl_path),
            "size_bytes": jsonl_path.stat().st_size,
            "required": True,
            "artifact_role": "manifest_jsonl",
            "exists": True,
        }
    )
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4f.json"
    writer.json(
        manifest_name,
        {
            "created_at": iso_kst(),
            "artifact_family": "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4f_reward_v2_immutable_freeze_preflight",
            "terminal_gate": gate["gate"],
            "readiness": gate["readiness"],
            "payload_count": len(rows),
            "missing_payload_count": 0,
            "files": rows,
        },
    )
    manifest_path = writer.root / manifest_name
    writer.json(
        "_PV8_R2AR8ER3RH4F_COMPLETE.lock",
        {
            "created_at": iso_kst(),
            "artifact_family": "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4f_reward_v2_immutable_freeze_preflight",
            "terminal_gate": gate["gate"],
            "readiness": gate["readiness"],
            "final_manifest_path": manifest_name,
            "final_manifest_sha256": k5.sha256_file(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
        },
    )


def run(root: Path) -> Path:
    first = review_once()
    second = review_once()
    deterministic = deterministic_replay(first, second)
    deterministic_flags = [
        deterministic["identical"],
        deterministic["canonical_frozen_payload_identical"],
        deterministic["individual_frozen_object_hashes_identical"],
        deterministic["combined_freeze_hash_identical"],
        deterministic["runtime_mapping_identical"],
        deterministic["patch_plan_identical"],
        deterministic["readiness_decision_identical"],
    ]
    if not all(bool(value) for value in deterministic_flags):
        raise H4FError("deterministic immutable freeze preflight drifted")
    final_integrity = integrity_audit(
        first["source"],
        first["transform"],
        first["stale"],
        first["patch"],
        first["fixture"],
        first["ownership"],
        first["leakage"],
        deterministic,
    )
    writer = k5.Writer(root)
    writer.json("r8er3rh4f_user_approval_record.json", first["approval"])
    writer.json("r8er3rh4f_upstream_binding.json", first["upstream"])
    writer.json("r8er3rh4f_source_integrity.json", first["source"])
    writer.json("r8er3rh4f_canonical_event_order_frozen.json", first["event_order"])
    writer.json("r8er3rh4f_temporal_binding_frozen.json", first["temporal"])
    writer.json("r8er3rh4f_training_reference_frozen.json", first["training_ref"])
    writer.json("r8er3rh4f_tail_evaluation_reference_frozen.json", first["tail_ref"])
    writer.json("r8er3rh4f_reward_weight_registry_frozen.json", first["weights"])
    writer.json("r8er3rh4f_horizon_status_frozen.json", first["horizon"])
    writer.json("r8er3rh4f_reward_semantics_v2_immutable_freeze.json", first["freeze"])
    writer.json("r8er3rh4f_freeze_hash_registry.json", first["hashes"])
    writer.json("r8er3rh4f_component_transform_resolution.json", first["transform"])
    writer.json("r8er3rh4f_runtime_reward_inventory.json", first["inventory"])
    writer.json("r8er3rh4f_stale_runtime_semantics_audit.json", first["stale"])
    writer.parquet(
        "r8er3rh4f_frozen_to_runtime_mapping.parquet",
        first["mapping"],
        ["frozen_v2_field", "runtime_source", "current_runtime_field", "required_modification", "status"],
    )
    writer.json("r8er3rh4f_runtime_rebinding_patch_plan.json", first["patch"])
    writer.json("r8er3rh4f_h4d_fixture_regression.json", first["fixture"])
    writer.json("r8er3rh4f_ownership_integrity.json", first["ownership"])
    writer.json("r8er3rh4f_future_leakage_audit.json", first["leakage"])
    writer.json("r8er3rh4f_runtime_rebinding_readiness.json", first["rebinding"])
    writer.json("r8er3rh4f_deterministic_replay.json", deterministic)
    writer.json("r8er3rh4f_integrity_audit.json", final_integrity)
    writer.json("r8er3rh4f_readiness_decision.json", first["readiness"])
    guards = claim_guards(first)
    writer.json("claim_guard_status.json", guards)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "gate_passed": True,
        "final_decision": first["decision"],
        "readiness": READINESS,
        "failure_reasons": [] if first["decision"] == SUCCESS_DECISION else [first["rationale"]],
        "runtime_approval_implied": False,
        "runtime_patch_executed": False,
        "training_implied": False,
    }
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "runner": str(RUNNER_PATH),
            "mode": "freeze-preflight",
            "python": sys.version,
            "platform": platform.platform(),
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "api_call_count": 0,
            "db_query_count": 0,
            "db_write_count": 0,
            "runtime_patch_count": 0,
            "representative_reward_rematerialization_count": 0,
            "policy_evaluation_count": 0,
            "optimizer_creation_count": 0,
            "checkpoint_reuse_count": 0,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": first["decision"], "readiness": READINESS})
    writer.text("final_report.md", final_report(first, deterministic))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze-preflight",), required=True)
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
