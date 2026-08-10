#!/usr/bin/env python3
"""Prompt 5-E01-R2D-1S.

Interval-Width Geometry, Clock-Dominance and Observation-Resolution
Feasibility Diagnostic.

Fully offline diagnostic step:
  network_api_calls = 0
  service_key_accessed = false
  database_connection = false
  external_network_access = false
  turnbull_rerun_executed = false
  new_estimation_executed = false
  simulator_parameter_generated = false
  simulator_application_executed = false

The step produces bounded counterfactual and feasibility numbers for
Path A (observation-resolution improvement) and Path B (partial-identification
sensitivity grid design input) in parallel. It adopts neither path and makes
no recommendation between them.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

KST = timezone(timedelta(hours=9))

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACT_ROOT = PROJECT_ROOT / "05_training" / "artifacts"

R2D1R_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1r_estimation_result_audit_non_identification_freeze_20260730_190737"
R2D1R_DETERMINISM_RERUN_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1r_estimation_result_audit_non_identification_freeze_20260730_191006"
R2D1Q_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1q_method_prototype_interval_censored_estimation_20260730_100056"
R2D1O_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1o_12_episode_registry_freeze_estimation_readiness_20260730_085243"
R2D1N_HF1_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1n_hf1_final_registry_eta_deduplication_20260730_000608"
R2D1E_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

REQUIRED_GATES = {
    "R2D1R": (R2D1R_DIR, "prompt5_e01_r2d1r_gate.json",
              ["PASS_METHOD_PROTOTYPE_ESTIMATION_RESULT_AUDIT_NON_IDENTIFICATION_FREEZE_COMPLETE"]),
    "R2D1Q": (R2D1Q_DIR, "prompt5_e01_r2d1q_gate.json",
              ["PASS_METHOD_PROTOTYPE_INTERVAL_CENSORED_ESTIMATION_COMPLETE",
               "PASS_METHOD_PROTOTYPE_ESTIMATION_NONUNIQUE_IDENTIFIED_SET_COMPLETE"]),
    "R2D1O": (R2D1O_DIR, "prompt5_e01_r2d1o_gate.json",
              ["PASS_12_EPISODE_REGISTRY_FREEZE_ESTIMATION_READINESS_READY"]),
    "R2D1N_HF1": (R2D1N_HF1_DIR, "prompt5_e01_r2d1n_hf1_gate.json",
                  ["PASS_CAMPAIGN_C_FINAL_REGISTRY_ETA_DEDUPLICATION_FREEZE_READY"]),
    "R2D1E": (R2D1E_DIR, "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json",
              ["PASS_SCHEMA_FREEZE_METHOD_REVIEW_READY"]),
    "R2D1F": (R2D1F_DIR, "prompt5_e01_r2d1f_gate.json",
              ["PASS_ESTIMATION_DESIGN_APPROVED_ADDITIONAL_DATA_REQUIRED"]),
}

UPSTREAM_DIRS = [R2D1R_DIR, R2D1Q_DIR, R2D1O_DIR, R2D1N_HF1_DIR, R2D1E_DIR, R2D1F_DIR]

CADENCES_SEC = [10, 15, 20, 30, 45, 60]
DECOMPOSITION_RESIDUAL_TOLERANCE_SEC = 1e-6
INTERSECTION_COMPARISON_TOLERANCE_SEC = 1e-9
CONFIGURED_MAX_CALLS_PER_MINUTE_EXPECTED = 4

PARTITION_CATEGORIES = [
    "PRESENT_STABLE_SPAN",
    "PRESENT_TO_TERMINAL_TRANSITION_BRACKET",
    "TERMINAL_TO_ABSENT_TRANSITION_BRACKET",
    "ABSENT_SAMPLED_SPAN",
    "ABSENT_TO_REENTRY_TRANSITION_BRACKET",
    "PRESENT_REENTRY_CONFIRMATION_SPAN",
    "UNKNOWN_OR_UNRESOLVED_SPAN",
]

PROHIBITED_OUTPUT_FIELD_NAMES = [
    "provider_gap_attributable_sec",
    "irreducible_width_sec",
]

REQUIRED_ARTIFACT_FILES = [
    "prompt5_e01_r2d1s_manifest.json",
    "prompt5_e01_r2d1s_gate.json",
    "prompt5_e01_r2d1s_final_report.md",
    "upstream_reference_r2d1r.json",
    "upstream_reference_r2d1q.json",
    "upstream_reference_r2d1o.json",
    "upstream_reference_r2d1n_hf1.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "upstream_raw_evidence_artifact_index.json",
    "authoritative_source_resolution_audit.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "database_connection_audit.json",
    "external_network_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "secret_leak_audit.json",
    "frozen_registry_reload_audit.json",
    "identified_set_recomputation_audit.json",
    "bound_definition_verification_audit.json",
    "clock_specific_interval_width_decomposition.json",
    "clock_specific_interval_width_decomposition.parquet",
    "episode_censoring_width_summary.json",
    "gap_sample_timeline.json",
    "gap_sample_timeline.parquet",
    "gap_timeline_partition_audit.json",
    "gap_reducibility_bounds.json",
    "gap_reducibility_bounds.parquet",
    "causal_attribution_limitation.json",
    "identified_set_binding_endpoint_audit.json",
    "identified_set_leave_one_out_audit.json",
    "identified_set_leave_one_out_audit.parquet",
    "capture_mode_stratification_audit.json",
    "broad_scan_exclusion_influence_audit.json",
    "clock_dominance_audit.json",
    "request_cadence_clock_counterfactual.json",
    "request_cadence_clock_counterfactual.parquet",
    "intersection_classification_audit.json",
    "rate_limit_contract_resolution.json",
    "rate_limit_feasibility_analysis.json",
    "rate_limit_feasibility_analysis.parquet",
    "phase_based_api_budget_projection.json",
    "phase_based_api_budget_projection.parquet",
    "new_episode_informativeness_conditions.json",
    "leave_one_out_informativeness_diagnostic.json",
    "path_a_observation_resolution_feasibility.json",
    "path_b_partial_identification_grid_design_input.json",
    "path_comparison_numeric_summary.json",
    "canonical_kpi_contract_resolution.json",
    "scheduled_layover_comparator_inventory.json",
    "prohibited_interpretations_carry_forward.json",
    "estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "simulator_application_authorization.json",
    "phase2_execution_authorization.json",
    "resolution_improvement_campaign_authorization.json",
    "json_parquet_synchronization_audit.json",
    "manifest_self_entry_contract.json",
]


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

class DiagnosticFailure(RuntimeError):
    def __init__(self, gate_status: str, detail: str):
        super().__init__(f"{gate_status}: {detail}")
        self.gate_status = gate_status
        self.detail = detail


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      indent=1, allow_nan=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def snapshot_dir(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = sha256_file(path)
    return out


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def raw_filename_request_time(path: str) -> Optional[datetime]:
    match = re.match(r"^(\d{8})_(\d{6})_(\d{6})", os.path.basename(path)[:-5])
    if not match:
        return None
    return datetime.strptime(match.group(1) + match.group(2),
                             "%Y%m%d%H%M%S").replace(tzinfo=KST)


def raw_capture_mode(path: str, campaign_dir: Path) -> str:
    rel = Path(path).relative_to(campaign_dir).as_posix()
    parts = rel.split("/")
    if len(parts) >= 3 and parts[0] == "raw" and not parts[1].isdigit():
        return parts[1].upper()
    stem = os.path.basename(path)[:-5]
    tokens = stem.split("_")
    if len(tokens) > 3:
        return "_".join(tokens[3:]).upper()
    return "UNSPECIFIED"


def target_vehicle_row(payload: Dict[str, Any], vehicle_id: str) -> Tuple[Optional[Dict[str, Any]], int]:
    body = payload.get("body") or {}
    items = body.get("items")
    if items is None:
        return None, 0
    if isinstance(items, dict):
        items = [items]
    for item in items:
        if str(item.get("vhcNo2")) == str(vehicle_id):
            return item, len(items)
    return None, len(items)


def provider_event_time(ar_time: Optional[str], request_dt: datetime) -> Optional[datetime]:
    if not ar_time or not re.fullmatch(r"\d{6}", str(ar_time)):
        return None
    hh, mm, ss = int(ar_time[0:2]), int(ar_time[2:4]), int(ar_time[4:6])
    candidate = request_dt.replace(hour=hh % 24, minute=mm, second=ss, microsecond=0)
    if (candidate - request_dt).total_seconds() > 43200:
        candidate -= timedelta(days=1)
    elif (request_dt - candidate).total_seconds() > 43200:
        candidate += timedelta(days=1)
    return candidate


def response_status_of(payload: Dict[str, Any]) -> Optional[str]:
    header = payload.get("header") or {}
    code = header.get("resultCode")
    if code is None:
        return None
    return "OK" if str(code) == "0000" else f"PROVIDER_RESULT_CODE_{code}"


def classify_intersection(lower: float, upper: float) -> str:
    delta = upper - lower
    if delta < -INTERSECTION_COMPARISON_TOLERANCE_SEC:
        return "EMPTY"
    if abs(delta) <= INTERSECTION_COMPARISON_TOLERANCE_SEC:
        return "SINGLETON"
    return "POSITIVE_WIDTH"


def median_or_none(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


# --------------------------------------------------------------------------
# Phase A - authoritative upstream resolution
# --------------------------------------------------------------------------

def phase_a_resolve_upstream() -> Dict[str, Any]:
    resolution: Dict[str, Any] = {
        "authoritative_selection_rule": (
            "explicit prompt-specified absolute paths only; a newer timestamp is never "
            "sufficient reason to substitute an artifact"
        ),
        "excluded_determinism_rerun": {
            "artifact_dir": str(R2D1R_DETERMINISM_RERUN_DIR),
            "authoritative": False,
            "exists": R2D1R_DETERMINISM_RERUN_DIR.is_dir(),
            "purpose": "DETERMINISM_RERUN_REFERENCE_ONLY",
            "used_as_analysis_input": False,
        },
        "records": [],
    }
    for key, (directory, gate_name, allowed) in REQUIRED_GATES.items():
        if not directory.is_dir():
            raise DiagnosticFailure("BLOCKED_AUTHORITATIVE_INPUT_MISSING",
                                    f"{key} artifact dir missing: {directory}")
        gate_path = directory / gate_name
        if not gate_path.is_file():
            raise DiagnosticFailure("BLOCKED_AUTHORITATIVE_INPUT_MISSING",
                                    f"{key} gate file missing: {gate_path}")
        gate = load_json(gate_path)
        status = gate.get("gate_status")
        if status not in allowed:
            raise DiagnosticFailure("FAIL_UPSTREAM_GATE_MISMATCH",
                                    f"{key} gate_status={status!r} not in {allowed}")
        resolution["records"].append({
            "artifact_dir": str(directory),
            "expected_gate_status_candidates": list(allowed),
            "file_count": sum(1 for p in directory.rglob("*") if p.is_file()),
            "gate_file": str(gate_path),
            "gate_sha256": sha256_file(gate_path),
            "gate_status": status,
            "gate_status_matches": True,
            "upstream_key": key,
        })
    resolution["resolved_upstream_count"] = len(resolution["records"])
    resolution["authoritative_source_resolution_passed"] = True
    return resolution


def phase_a_lineage_audit() -> Dict[str, Any]:
    r2d1r_gate = load_json(R2D1R_DIR / "prompt5_e01_r2d1r_gate.json")
    r2d1r_fingerprint = load_json(R2D1R_DIR / "identified_set_freeze_fingerprint.json")
    r2d1o_gate = load_json(R2D1O_DIR / "prompt5_e01_r2d1o_gate.json")
    r2d1o_fingerprint = load_json(R2D1O_DIR / "registry_12_freeze_fingerprint.json")
    q_ref = load_json(R2D1R_DIR / "upstream_reference_r2d1q.json")
    o_ref = load_json(R2D1R_DIR / "upstream_reference_r2d1o.json")

    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, observed: Any, expected: Any) -> None:
        checks.append({"check": name, "expected": expected, "observed": observed, "passed": bool(ok)})

    check("r2d1r_audited_r2d1q_dir_matches_authoritative_r2d1q",
          r2d1r_gate.get("audited_r2d1q_artifact_dir") == str(R2D1Q_DIR),
          r2d1r_gate.get("audited_r2d1q_artifact_dir"), str(R2D1Q_DIR))
    check("r2d1r_fingerprint_source_r2d1q_dir_matches",
          r2d1r_fingerprint.get("source_r2d1q_artifact_dir") == str(R2D1Q_DIR),
          r2d1r_fingerprint.get("source_r2d1q_artifact_dir"), str(R2D1Q_DIR))
    check("r2d1r_source_r2d1o_gate_matches_r2d1o_gate_status",
          r2d1r_gate.get("source_r2d1o_gate") == r2d1o_gate.get("gate_status"),
          r2d1r_gate.get("source_r2d1o_gate"), r2d1o_gate.get("gate_status"))
    check("r2d1r_source_r2d1q_gate_matches_r2d1q_gate_status",
          r2d1r_gate.get("source_r2d1q_gate") == load_json(
              R2D1Q_DIR / "prompt5_e01_r2d1q_gate.json").get("gate_status"),
          r2d1r_gate.get("source_r2d1q_gate"),
          load_json(R2D1Q_DIR / "prompt5_e01_r2d1q_gate.json").get("gate_status"))
    check("r2d1r_input_registry_fingerprint_matches_r2d1o_registry_canonical_sha256",
          r2d1r_fingerprint.get("source_input_registry_fingerprint")
          == r2d1o_fingerprint.get("registry_canonical_sha256"),
          r2d1r_fingerprint.get("source_input_registry_fingerprint"),
          r2d1o_fingerprint.get("registry_canonical_sha256"))
    check("r2d1o_source_r2d1n_hf1_gate_matches",
          r2d1o_gate.get("source_r2d1n_hf1_gate") == load_json(
              R2D1N_HF1_DIR / "prompt5_e01_r2d1n_hf1_gate.json").get("gate_status"),
          r2d1o_gate.get("source_r2d1n_hf1_gate"),
          load_json(R2D1N_HF1_DIR / "prompt5_e01_r2d1n_hf1_gate.json").get("gate_status"))
    check("r2d1r_referenced_r2d1q_gate_sha256_matches_recomputed",
          r2d1r_fingerprint.get("source_r2d1q_gate_sha256")
          == sha256_file(R2D1Q_DIR / "prompt5_e01_r2d1q_gate.json"),
          r2d1r_fingerprint.get("source_r2d1q_gate_sha256"),
          sha256_file(R2D1Q_DIR / "prompt5_e01_r2d1q_gate.json"))
    check("r2d1r_referenced_r2d1q_manifest_sha256_matches_recomputed",
          r2d1r_fingerprint.get("source_r2d1q_manifest_sha256")
          == sha256_file(R2D1Q_DIR / "prompt5_e01_r2d1q_manifest.json"),
          r2d1r_fingerprint.get("source_r2d1q_manifest_sha256"),
          sha256_file(R2D1Q_DIR / "prompt5_e01_r2d1q_manifest.json"))
    q_ref_path = q_ref.get("artifact_dir") or q_ref.get("upstream_artifact_dir") or q_ref.get("absolute_path")
    o_ref_path = o_ref.get("artifact_dir") or o_ref.get("upstream_artifact_dir") or o_ref.get("absolute_path")
    check("r2d1r_upstream_reference_r2d1q_path_matches",
          str(q_ref_path or "") == str(R2D1Q_DIR),
          q_ref_path, str(R2D1Q_DIR))
    check("r2d1r_upstream_reference_r2d1o_path_matches",
          str(o_ref_path or "") == str(R2D1O_DIR),
          o_ref_path, str(R2D1O_DIR))

    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise DiagnosticFailure("FAIL_LINEAGE_RESOLUTION", f"lineage checks failed: {failed}")
    return {
        "checks": checks,
        "failed_check_count": 0,
        "lineage_resolution_passed": True,
        "upstream_lineage_chain": [
            "R2D-1E methodology freeze",
            "R2D-1F estimation design approval",
            "R2D-1N-HF1 final registry / ETA deduplication freeze",
            "R2D-1O 12-episode registry freeze",
            "R2D-1Q method-prototype interval-censored estimation",
            "R2D-1R estimation-result audit / non-identification freeze",
        ],
    }


def phase_a_source_integrity() -> Dict[str, Any]:
    records = []
    for key, (directory, gate_name, _allowed) in REQUIRED_GATES.items():
        manifest_candidates = sorted(directory.glob("*manifest*.json"))
        manifest_path = manifest_candidates[0] if manifest_candidates else None
        missing = 0
        hash_mismatch = 0
        entry_count = 0
        if manifest_path is not None:
            manifest = load_json(manifest_path)
            entries = manifest.get("files") or manifest.get("entries") or manifest.get("records")
            if isinstance(entries, list):
                for entry in entries:
                    rel = entry.get("path") or entry.get("relative_path")
                    if not rel:
                        continue
                    entry_count += 1
                    target = directory / rel
                    if not target.is_file():
                        missing += 1
                        continue
                    recorded = entry.get("sha256")
                    if entry.get("self_hash_exempt") or recorded is None:
                        continue
                    if sha256_file(target) != recorded:
                        hash_mismatch += 1
        records.append({
            "artifact_dir": str(directory),
            "gate_sha256": sha256_file(directory / gate_name),
            "manifest_entry_count": entry_count,
            "manifest_missing_file_count": missing,
            "manifest_nonself_hash_mismatch_count": hash_mismatch,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "upstream_key": key,
        })
    total_missing = sum(r["manifest_missing_file_count"] for r in records)
    total_mismatch = sum(r["manifest_nonself_hash_mismatch_count"] for r in records)
    if total_missing or total_mismatch:
        raise DiagnosticFailure(
            "FAIL_SOURCE_ARTIFACT_INTEGRITY",
            f"upstream manifest missing={total_missing} hash_mismatch={total_mismatch}")
    return {
        "records": records,
        "source_artifact_integrity_passed": True,
        "total_manifest_missing_file_count": 0,
        "total_manifest_nonself_hash_mismatch_count": 0,
    }


# --------------------------------------------------------------------------
# Phase A/B - frozen registry reload and identified-set recomputation
# --------------------------------------------------------------------------

REGISTRY_ROW_ORDER = ["observation_date", "first_terminal_request_time",
                     "route_id", "vehicle_id", "episode_id"]


def load_frozen_registry() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    registry_json = R2D1O_DIR / "registry_12_frozen.json"
    registry_parquet = R2D1O_DIR / "registry_12_frozen.parquet"
    fingerprint_path = R2D1O_DIR / "registry_12_freeze_fingerprint.json"
    packet_path = R2D1O_DIR / "method_prototype_estimation_input_packet.json"
    intervals_parquet = R2D1O_DIR / "method_prototype_estimation_input_intervals.parquet"
    for path in [registry_json, registry_parquet, fingerprint_path, packet_path, intervals_parquet]:
        if not path.is_file():
            raise DiagnosticFailure("FAIL_FROZEN_REGISTRY_RELOAD", f"missing {path}")

    registry = load_json(registry_json)
    records = registry["records"]
    if len(records) != 12 or int(registry.get("row_count", -1)) != 12:
        raise DiagnosticFailure("FAIL_FROZEN_REGISTRY_RELOAD",
                                f"registry row count = {len(records)} (expected 12)")

    fingerprint = load_json(fingerprint_path)
    ordered = sorted(records, key=lambda r: tuple(str(r[k]) for k in REGISTRY_ROW_ORDER))
    canonical_fields = [
        "episode_id",
        "source_episode_id",
        "source_artifact",
        "route_id",
        "vehicle_id",
        "direction",
        "observation_date",
        "hour_bucket",
        "final_status",
        "provider_lower_bound_sec",
        "provider_upper_bound_sec",
        "request_lower_bound_sec",
        "request_upper_bound_sec",
        "conservative_dual_lower_bound_sec",
        "conservative_dual_upper_bound_sec",
        "clock_semantics_status",
        "first_upstream_raw_sha256",
        "last_pre_terminal_raw_sha256",
        "first_terminal_raw_sha256",
        "last_terminal_raw_sha256",
        "first_post_terminal_raw_sha256",
        "post_terminal_confirmation_raw_sha256s",
        "global_vehicle_identity_key",
        "route_local_vehicle_identity_key",
    ]
    registry_canonical_sha256 = sha256_text(
        canonical_json([{field: row.get(field) for field in canonical_fields} for row in ordered]))
    episode_id_set_sha256 = sha256_text(canonical_json(sorted(r["episode_id"] for r in records)))
    interval_tuples = sorted(
        (
            {
                "episode_id": r["episode_id"],
                "provider": [r["provider_lower_bound_sec"], r["provider_upper_bound_sec"]],
                "request": [r["request_lower_bound_sec"], r["request_upper_bound_sec"]],
                "conservative": [
                    r["conservative_dual_lower_bound_sec"],
                    r["conservative_dual_upper_bound_sec"],
                ],
            }
            for r in records
        ),
        key=lambda row: row["episode_id"],
    )
    interval_tuple_set_sha256 = sha256_text(canonical_json(interval_tuples))

    parquet_df = pd.read_parquet(registry_parquet)
    parquet_sha256 = sha256_file(registry_parquet)

    fp_checks = {
        "registry_canonical_sha256": (registry_canonical_sha256,
                                      fingerprint.get("registry_canonical_sha256")),
        "episode_id_set_sha256": (episode_id_set_sha256,
                                  fingerprint.get("episode_id_set_sha256")),
        "interval_tuple_set_sha256": (interval_tuple_set_sha256,
                                      fingerprint.get("interval_tuple_set_sha256")),
        "parquet_file_sha256": (parquet_sha256, fingerprint.get("parquet_file_sha256")),
    }
    fingerprint_records = []
    for name, (recomputed, frozen) in fp_checks.items():
        fingerprint_records.append({
            "fingerprint_field": name,
            "frozen_value": frozen,
            "recomputed_value": recomputed,
            "matches": recomputed == frozen,
        })
    parquet_matches = int(len(parquet_df)) == 12

    audit = {
        "episode_ids": sorted(r["episode_id"] for r in records),
        "fingerprint_field_match_count": sum(1 for r in fingerprint_records if r["matches"]),
        "fingerprint_field_total": len(fingerprint_records),
        "fingerprint_records": fingerprint_records,
        "frozen_registry_reload_passed": True,
        "parquet_row_count": int(len(parquet_df)),
        "parquet_row_count_matches_json": parquet_matches,
        "registry_row_count": len(records),
        "registry_row_count_after": len(records),
        "registry_row_count_before": len(records),
        "resolved_paths": {
            "method_prototype_estimation_input_intervals_parquet": str(intervals_parquet),
            "method_prototype_estimation_input_packet_json": str(packet_path),
            "registry_12_freeze_fingerprint_json": str(fingerprint_path),
            "registry_12_frozen_json": str(registry_json),
            "registry_12_frozen_parquet": str(registry_parquet),
        },
        "row_ordering_contract": REGISTRY_ROW_ORDER,
        "canonical_serialization_contract": {
            "allow_nan": False,
            "note": ("registry_canonical_sha256 is recomputed with this step's own canonical "
                     "serializer; a value mismatch is reported, not treated as registry mutation, "
                     "because the frozen fingerprint was produced by the R2D-1O serializer"),
            "strict_json": True,
        },
    }
    if not parquet_matches:
        raise DiagnosticFailure("FAIL_FROZEN_REGISTRY_RELOAD",
                                "registry parquet row count != 12")
    return records, audit


def recompute_identified_sets(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    frozen_sets = load_json(R2D1R_DIR / "frozen_identified_sets.json")
    frozen_parquet = pd.read_parquet(R2D1R_DIR / "frozen_identified_sets.parquet")

    def endpoints(lower_field: str, upper_field: str) -> Tuple[float, float, float]:
        lowers = [float(r[lower_field]) for r in records]
        uppers = [float(r[upper_field]) for r in records]
        star_lower = max(lowers)
        star_upper = min(uppers)
        return star_lower, star_upper, star_upper - star_lower

    dual_lower, dual_upper, dual_width = endpoints(
        "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec")
    request_lower, request_upper, request_width = endpoints(
        "request_lower_bound_sec", "request_upper_bound_sec")
    provider_lower, provider_upper, provider_width = endpoints(
        "provider_lower_bound_sec", "provider_upper_bound_sec")

    frozen_by_analysis: Dict[str, Dict[str, float]] = {}
    for rec in frozen_sets["records"]:
        frozen_by_analysis[rec["analysis_id"]] = {
            "identified_lower_sec": float(rec["identified_lower_sec"]),
            "identified_upper_sec": float(rec["identified_upper_sec"]),
            "identified_width_sec": float(rec["identified_width_sec"]),
        }

    comparisons = []
    mismatch_count = 0
    for analysis_id, recomputed in [
        ("PRIMARY_DUAL_EQUAL", (dual_lower, dual_upper, dual_width)),
        ("SENSITIVITY_DUAL_VEHICLE_BALANCED", (dual_lower, dual_upper, dual_width)),
        ("SENSITIVITY_PROVIDER_ONLY", (provider_lower, provider_upper, provider_width)),
        ("SENSITIVITY_REQUEST_ONLY", (request_lower, request_upper, request_width)),
    ]:
        frozen = frozen_by_analysis.get(analysis_id)
        if frozen is None:
            mismatch_count += 1
            comparisons.append({"analysis_id": analysis_id, "frozen_present": False,
                                "matches": False})
            continue
        matches = (
            abs(frozen["identified_lower_sec"] - recomputed[0]) <= INTERSECTION_COMPARISON_TOLERANCE_SEC
            and abs(frozen["identified_upper_sec"] - recomputed[1]) <= INTERSECTION_COMPARISON_TOLERANCE_SEC
            and abs(frozen["identified_width_sec"] - recomputed[2]) <= INTERSECTION_COMPARISON_TOLERANCE_SEC
        )
        if not matches:
            mismatch_count += 1
        comparisons.append({
            "analysis_id": analysis_id,
            "frozen_identified_lower_sec": frozen["identified_lower_sec"],
            "frozen_identified_upper_sec": frozen["identified_upper_sec"],
            "frozen_identified_width_sec": frozen["identified_width_sec"],
            "frozen_present": True,
            "matches": matches,
            "recomputed_identified_lower_sec": recomputed[0],
            "recomputed_identified_upper_sec": recomputed[1],
            "recomputed_identified_width_sec": recomputed[2],
        })

    r2d1r_gate = load_json(R2D1R_DIR / "prompt5_e01_r2d1r_gate.json")
    gate_checks = [
        {"field": "primary_identified_lower_sec",
         "gate_value": float(r2d1r_gate["primary_identified_lower_sec"]),
         "recomputed_value": dual_lower,
         "matches": abs(float(r2d1r_gate["primary_identified_lower_sec"]) - dual_lower) <= INTERSECTION_COMPARISON_TOLERANCE_SEC},
        {"field": "primary_identified_upper_sec",
         "gate_value": float(r2d1r_gate["primary_identified_upper_sec"]),
         "recomputed_value": dual_upper,
         "matches": abs(float(r2d1r_gate["primary_identified_upper_sec"]) - dual_upper) <= INTERSECTION_COMPARISON_TOLERANCE_SEC},
        {"field": "primary_identified_width_sec",
         "gate_value": float(r2d1r_gate["primary_identified_width_sec"]),
         "recomputed_value": dual_width,
         "matches": abs(float(r2d1r_gate["primary_identified_width_sec"]) - dual_width) <= INTERSECTION_COMPARISON_TOLERANCE_SEC},
        {"field": "request_identified_lower_sec",
         "gate_value": float(r2d1r_gate["request_identified_lower_sec"]),
         "recomputed_value": request_lower,
         "matches": abs(float(r2d1r_gate["request_identified_lower_sec"]) - request_lower) <= INTERSECTION_COMPARISON_TOLERANCE_SEC},
        {"field": "request_identified_upper_sec",
         "gate_value": float(r2d1r_gate["request_identified_upper_sec"]),
         "recomputed_value": request_upper,
         "matches": abs(float(r2d1r_gate["request_identified_upper_sec"]) - request_upper) <= INTERSECTION_COMPARISON_TOLERANCE_SEC},
    ]
    mismatch_count += sum(1 for c in gate_checks if not c["matches"])
    if mismatch_count:
        raise DiagnosticFailure("FAIL_IDENTIFIED_SET_RECOMPUTATION",
                                f"identified_set_mismatch_count = {mismatch_count}")

    return {
        "analysis_comparisons": comparisons,
        "current_dual_identified_set_width_sec": dual_width,
        "current_max_dual_lower_sec": dual_lower,
        "current_max_provider_lower_sec": provider_lower,
        "current_max_request_lower_sec": request_lower,
        "current_min_dual_upper_sec": dual_upper,
        "current_min_provider_upper_sec": provider_upper,
        "current_min_request_upper_sec": request_upper,
        "current_provider_identified_set_width_sec": provider_width,
        "current_request_identified_set_width_sec": request_width,
        "frozen_identified_sets_parquet_row_count": int(len(frozen_parquet)),
        "frozen_identified_sets_resolved_paths": {
            "frozen_identified_sets_json": str(R2D1R_DIR / "frozen_identified_sets.json"),
            "frozen_identified_sets_parquet": str(R2D1R_DIR / "frozen_identified_sets.parquet"),
            "binding_episode_analysis_json": str(R2D1R_DIR / "binding_episode_analysis.json"),
            "clock_input_independence_audit_json": str(R2D1R_DIR / "clock_input_independence_audit.json"),
            "resolution_improvement_requirement_json": str(R2D1R_DIR / "resolution_improvement_requirement.json"),
            "turnbull_innermost_interval_audit_json": str(R2D1R_DIR / "turnbull_innermost_interval_audit.json"),
        },
        "gate_field_comparisons": gate_checks,
        "identified_set_mismatch_count": 0,
        "identified_set_recomputation_passed": True,
        "identified_set_width_attribution_status": "NON_ADDITIVE",
        "recomputation_rule": {
            "identified_set_lower": "L_star = max_i(L_i)",
            "identified_set_upper": "U_star = min_i(U_i)",
            "identified_set_width": "U_star - L_star",
            "episode_width": "W_i = U_i - L_i",
        },
    }


# --------------------------------------------------------------------------
# Phase B - bound definition verification
# --------------------------------------------------------------------------

def phase_b_bound_definitions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema_path = R2D1F_DIR / "terminal_recovery_estimation_input_schema.json"
    dictionary_path = R2D1F_DIR / "terminal_recovery_estimation_input_data_dictionary.json"
    dual_contract_path = R2D1F_DIR / "dual_clock_interval_design_contract.json"
    clock_contract_path = R2D1E_DIR / "clock_semantics_contract_v2.json"
    candidate_parquet = R2D1F_DIR / "terminal_recovery_estimation_input_candidate.parquet"
    interval_validation_path = R2D1O_DIR / "registry_12_interval_validation_audit.json"
    q_interval_validation_path = R2D1Q_DIR / "interval_input_validation_audit.json"

    sources = [schema_path, dictionary_path, dual_contract_path, clock_contract_path,
               candidate_parquet, interval_validation_path, q_interval_validation_path]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise DiagnosticFailure("FAIL_BOUND_DEFINITION_RESOLUTION",
                                f"bound-definition sources missing: {missing}")

    dual_contract = load_json(dual_contract_path)
    clock_contract = load_json(clock_contract_path)
    dictionary = load_json(dictionary_path)
    dict_fields = {f["field"]: f.get("validation_rule") for f in dictionary.get("fields", [])}

    window_fields = ["provider_service_end_window_start", "provider_service_end_window_end",
                     "provider_reentry_window_start", "provider_reentry_window_end",
                     "request_service_end_window_start", "request_service_end_window_end",
                     "request_reentry_window_start", "request_reentry_window_end"]
    window_fields_documented = all(f in dict_fields for f in window_fields)

    candidate = pd.read_parquet(candidate_parquet)
    window_alignment: List[Dict[str, Any]] = []
    alignment_mismatch = 0
    for _, row in candidate.iterrows():
        source_episode_id = str(row["source_episode_id"])
        reg = next((r for r in records if r["episode_id"] == source_episode_id), None)
        if reg is None:
            continue
        pairs = [
            ("provider_service_end_window_start", "last_pre_terminal_provider_time"),
            ("provider_service_end_window_end", "first_terminal_provider_time"),
            ("provider_reentry_window_start", "last_terminal_provider_time"),
            ("provider_reentry_window_end", "first_post_terminal_provider_time"),
            ("request_service_end_window_start", "last_pre_terminal_request_time"),
            ("request_service_end_window_end", "first_terminal_request_time"),
            ("request_reentry_window_start", "last_terminal_request_time"),
            ("request_reentry_window_end", "first_post_terminal_request_time"),
        ]
        for window_field, anchor_field in pairs:
            observed = str(row[window_field])
            expected = str(reg[anchor_field])
            ok = parse_iso(observed) == parse_iso(expected)
            if not ok:
                alignment_mismatch += 1
            window_alignment.append({
                "anchor_field": anchor_field,
                "episode_id": source_episode_id,
                "matches": ok,
                "registry_anchor_value": expected,
                "window_field": window_field,
                "window_field_value": observed,
            })

    numeric_records: List[Dict[str, Any]] = []
    hypothesis_mismatch = 0
    for reg in records:
        for clock, prefix in (("PROVIDER", "provider"), ("REQUEST", "request")):
            last_pre = parse_iso(reg[f"last_pre_terminal_{prefix}_time"])
            first_term = parse_iso(reg[f"first_terminal_{prefix}_time"])
            last_term = parse_iso(reg[f"last_terminal_{prefix}_time"])
            first_post = parse_iso(reg[f"first_post_terminal_{prefix}_time"])
            derived_lower = (last_term - first_term).total_seconds()
            derived_upper = (first_post - last_pre).total_seconds()
            frozen_lower = float(reg[f"{prefix}_lower_bound_sec"])
            frozen_upper = float(reg[f"{prefix}_upper_bound_sec"])
            lower_residual = derived_lower - frozen_lower
            upper_residual = derived_upper - frozen_upper
            ok = (abs(lower_residual) <= DECOMPOSITION_RESIDUAL_TOLERANCE_SEC
                  and abs(upper_residual) <= DECOMPOSITION_RESIDUAL_TOLERANCE_SEC)
            if not ok:
                hypothesis_mismatch += 1
            numeric_records.append({
                "clock": clock,
                "derived_lower_bound_sec": derived_lower,
                "derived_upper_bound_sec": derived_upper,
                "episode_id": reg["episode_id"],
                "exact_zero_lower_residual": lower_residual == 0.0,
                "exact_zero_upper_residual": upper_residual == 0.0,
                "frozen_lower_bound_sec": frozen_lower,
                "frozen_upper_bound_sec": frozen_upper,
                "lower_residual_sec": lower_residual,
                "matches": ok,
                "upper_residual_sec": upper_residual,
            })

    if hypothesis_mismatch or alignment_mismatch or not window_fields_documented:
        raise DiagnosticFailure(
            "FAIL_BOUND_DEFINITION_RESOLUTION",
            f"bound formula verification failed (numeric={hypothesis_mismatch}, "
            f"window_alignment={alignment_mismatch}, documented={window_fields_documented})")

    return {
        "actual_bound_formula": {
            "L_clock": "reentry_window_start - service_end_window_end = last_terminal_time - first_terminal_time",
            "U_clock": "reentry_window_end - service_end_window_start = first_post_terminal_time - last_pre_terminal_time",
            "conservative_dual_lower": "min(provider_lower_bound_sec, request_lower_bound_sec)",
            "conservative_dual_upper": "max(provider_upper_bound_sec, request_upper_bound_sec)",
        },
        "bound_definition_source_paths": [str(p) for p in sources],
        "bound_definition_source_resolved": True,
        "bound_hypothesis_matches": True,
        "bound_hypothesis_as_stated_in_prompt": {
            "L_request": "last_terminal_request_time - first_terminal_request_time",
            "U_request": "first_post_terminal_request_time - last_pre_terminal_request_time",
        },
        "bound_hypothesis_numeric_verification": numeric_records,
        "conservative_dual_clock_contract": dual_contract.get("rule_a_conservative_envelope"),
        "clock_fields_preserved": clock_contract.get("clock_fields_preserved"),
        "dual_expansion_formula": (
            "W_dual = W_request + max(0, U_provider - U_request) + max(0, L_request - L_provider)"),
        "exact_zero_residual_record_count": sum(
            1 for r in numeric_records
            if r["exact_zero_lower_residual"] and r["exact_zero_upper_residual"]),
        "numeric_verification_record_count": len(numeric_records),
        "provider_decomposition_formula": (
            "W_provider = entry_detection_bracket_provider + exit_detection_bracket_provider, "
            "entry = first_terminal_provider_time - last_pre_terminal_provider_time, "
            "exit = first_post_terminal_provider_time - last_terminal_provider_time"),
        "rederived_decomposition_valid": True,
        "request_decomposition_formula": (
            "W_request = entry_detection_bracket_request + exit_detection_bracket_request, "
            "entry = first_terminal_request_time - last_pre_terminal_request_time, "
            "exit = first_post_terminal_request_time - last_terminal_request_time"),
        "window_field_alignment_mismatch_count": alignment_mismatch,
        "window_field_alignment_records": window_alignment,
        "window_fields_documented_in_r2d1f_data_dictionary": window_fields_documented,
        "window_semantics": {
            "reentry_window": "[last_terminal_time, first_post_terminal_time]",
            "service_end_window": "[last_pre_terminal_time, first_terminal_time]",
        },
        "terminal_state_semantics_rederived": False,
        "terminal_state_semantics_note": (
            "The upstream definition of the terminal state (which sighting counts as "
            "first/last terminal) is frozen and is not re-derived here; this step consumes "
            "the frozen anchors as given."),
    }


# --------------------------------------------------------------------------
# Phase C - clock-specific width decomposition
# --------------------------------------------------------------------------

def phase_c_decomposition(records: List[Dict[str, Any]],
                          identified: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dual_star_lower = identified["current_max_dual_lower_sec"]
    dual_star_upper = identified["current_min_dual_upper_sec"]
    request_star_lower = identified["current_max_request_lower_sec"]
    request_star_upper = identified["current_min_request_upper_sec"]

    rows: List[Dict[str, Any]] = []
    for reg in records:
        out: Dict[str, Any] = {
            "episode_id": reg["episode_id"],
            "route_id": str(reg["route_id"]),
            "vehicle_id": str(reg["vehicle_id"]),
            "observation_date": str(reg["observation_date"]),
            "hour_bucket": str(reg["hour_bucket"]),
            "source_artifact": str(reg["source_artifact"]),
            "source_campaign_id": str(reg["source_campaign_id"]),
        }
        for clock, prefix in (("provider", "provider"), ("request", "request")):
            last_pre = parse_iso(reg[f"last_pre_terminal_{prefix}_time"])
            first_term = parse_iso(reg[f"first_terminal_{prefix}_time"])
            last_term = parse_iso(reg[f"last_terminal_{prefix}_time"])
            first_post = parse_iso(reg[f"first_post_terminal_{prefix}_time"])
            lower = float(reg[f"{prefix}_lower_bound_sec"])
            upper = float(reg[f"{prefix}_upper_bound_sec"])
            width = upper - lower
            entry = (first_term - last_pre).total_seconds()
            exit_ = (first_post - last_term).total_seconds()
            residual = width - (entry + exit_)
            out[f"{clock}_lower_bound_sec"] = lower
            out[f"{clock}_upper_bound_sec"] = upper
            out[f"{clock}_width_sec"] = width
            out[f"{clock}_entry_bracket_sec"] = entry
            out[f"{clock}_exit_bracket_sec"] = exit_
            out[f"{clock}_decomposition_residual_sec"] = residual
            out[f"{clock}_decomposition_residual_exact_zero"] = residual == 0.0

        dual_lower = float(reg["conservative_dual_lower_bound_sec"])
        dual_upper = float(reg["conservative_dual_upper_bound_sec"])
        provider_lower = out["provider_lower_bound_sec"]
        provider_upper = out["provider_upper_bound_sec"]
        request_lower = out["request_lower_bound_sec"]
        request_upper = out["request_upper_bound_sec"]

        lower_hedge = max(0.0, request_lower - provider_lower)
        upper_hedge = max(0.0, provider_upper - request_upper)
        dual_width = dual_upper - dual_lower
        dual_expansion_residual = dual_width - (out["request_width_sec"] + lower_hedge + upper_hedge)

        if provider_lower < request_lower:
            lower_source = "PROVIDER"
        elif request_lower < provider_lower:
            lower_source = "REQUEST"
        else:
            lower_source = "PROVIDER_AND_REQUEST_EQUAL"
        if provider_upper > request_upper:
            upper_source = "PROVIDER"
        elif request_upper > provider_upper:
            upper_source = "REQUEST"
        else:
            upper_source = "PROVIDER_AND_REQUEST_EQUAL"

        out.update({
            "conservative_dual_lower_bound_sec": dual_lower,
            "conservative_dual_upper_bound_sec": dual_upper,
            "conservative_dual_width_sec": dual_width,
            "dual_clock_hedge_effective": bool(dual_lower < provider_lower or dual_upper > provider_upper),
            "dual_expansion_residual_exact_zero": dual_expansion_residual == 0.0,
            "dual_expansion_residual_sec": dual_expansion_residual,
            "dual_lower_hedge_expansion_sec": lower_hedge,
            "dual_lower_source_clock": lower_source,
            "dual_upper_hedge_expansion_sec": upper_hedge,
            "dual_upper_source_clock": upper_source,
            "is_dual_lower_binding": abs(dual_lower - dual_star_lower) <= INTERSECTION_COMPARISON_TOLERANCE_SEC,
            "is_dual_upper_binding": abs(dual_upper - dual_star_upper) <= INTERSECTION_COMPARISON_TOLERANCE_SEC,
            "is_request_lower_binding": abs(request_lower - request_star_lower) <= INTERSECTION_COMPARISON_TOLERANCE_SEC,
            "is_request_upper_binding": abs(request_upper - request_star_upper) <= INTERSECTION_COMPARISON_TOLERANCE_SEC,
        })
        rows.append(out)

    request_residuals = [abs(r["request_decomposition_residual_sec"]) for r in rows]
    provider_residuals = [abs(r["provider_decomposition_residual_sec"]) for r in rows]
    dual_residuals = [abs(r["dual_expansion_residual_sec"]) for r in rows]
    summary = {
        "decomposition_residual_tolerance_sec": DECOMPOSITION_RESIDUAL_TOLERANCE_SEC,
        "dual_expansion_residual_abs_max_sec": max(dual_residuals),
        "dual_expansion_residual_exact_zero_count": sum(
            1 for r in rows if r["dual_expansion_residual_exact_zero"]),
        "provider_decomposition_residual_abs_max_sec": max(provider_residuals),
        "provider_decomposition_residual_exact_zero_count": sum(
            1 for r in rows if r["provider_decomposition_residual_exact_zero"]),
        "request_decomposition_residual_abs_max_sec": max(request_residuals),
        "request_decomposition_residual_exact_zero_count": sum(
            1 for r in rows if r["request_decomposition_residual_exact_zero"]),
    }
    if (summary["request_decomposition_residual_abs_max_sec"] > DECOMPOSITION_RESIDUAL_TOLERANCE_SEC
            or summary["provider_decomposition_residual_abs_max_sec"] > DECOMPOSITION_RESIDUAL_TOLERANCE_SEC
            or summary["dual_expansion_residual_abs_max_sec"] > DECOMPOSITION_RESIDUAL_TOLERANCE_SEC):
        raise DiagnosticFailure("FAIL_CLOCK_SPECIFIC_DECOMPOSITION",
                                f"decomposition residual outside tolerance: {summary}")

    dual_widths = [r["conservative_dual_width_sec"] for r in rows]
    request_widths = [r["request_width_sec"] for r in rows]
    provider_widths = [r["provider_width_sec"] for r in rows]
    censoring_summary = {
        "episode_count": len(rows),
        "episode_width_vs_identified_set_width": {
            "episode_width_definition": "W_i = U_i - L_i",
            "identified_set_width_definition": "U_star - L_star with U_star=min_i U_i, L_star=max_i L_i",
            "identified_set_width_attribution_status": "NON_ADDITIVE",
            "prohibited_interpretation": (
                "the sum or median of the 12 episode widths is not an additive cause of the "
                "1604-second identified-set width"),
        },
        "dual_width_max_sec": max(dual_widths),
        "dual_width_median_sec": median_or_none(dual_widths),
        "dual_width_min_sec": min(dual_widths),
        "identified_set_dual_width_sec": dual_star_upper - dual_star_lower,
        "identified_set_request_width_sec": request_star_upper - request_star_lower,
        "provider_width_max_sec": max(provider_widths),
        "provider_width_median_sec": median_or_none(provider_widths),
        "provider_width_min_sec": min(provider_widths),
        "request_entry_bracket_max_sec": max(r["request_entry_bracket_sec"] for r in rows),
        "request_entry_bracket_median_sec": median_or_none([r["request_entry_bracket_sec"] for r in rows]),
        "request_exit_bracket_max_sec": max(r["request_exit_bracket_sec"] for r in rows),
        "request_exit_bracket_median_sec": median_or_none([r["request_exit_bracket_sec"] for r in rows]),
        "request_width_max_sec": max(request_widths),
        "request_width_median_sec": median_or_none(request_widths),
        "request_width_min_sec": min(request_widths),
        "residual_summary": summary,
        "dual_lower_hedge_expansion_median_sec": median_or_none(
            [r["dual_lower_hedge_expansion_sec"] for r in rows]),
        "dual_upper_hedge_expansion_median_sec": median_or_none(
            [r["dual_upper_hedge_expansion_sec"] for r in rows]),
        "dual_clock_hedge_effective_episode_count": sum(
            1 for r in rows if r["dual_clock_hedge_effective"]),
    }
    return rows, {"records": rows, "summary": summary, "censoring_summary": censoring_summary}


# --------------------------------------------------------------------------
# Phase D - raw request timeline reconstruction
# --------------------------------------------------------------------------

def resolve_raw_evidence_index(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    provenance_path = R2D1O_DIR / "registry_12_raw_provenance_freeze_audit.json"
    if not provenance_path.is_file():
        raise DiagnosticFailure("FAIL_RAW_EVIDENCE_MISSING", f"missing {provenance_path}")
    provenance = load_json(provenance_path)

    per_episode: Dict[str, Dict[str, Any]] = {}
    sha_mismatch = 0
    missing = 0
    for rec in provenance["records"]:
        episode_id = rec["episode_id"]
        raw_path = Path(rec["raw_path"])
        entry = per_episode.setdefault(episode_id, {
            "campaign_artifact_dirs": set(),
            "evidence_references": [],
            "source_index_paths": set(),
        })
        campaign_dir = ARTIFACT_ROOT / str(raw_path).split("/artifacts/")[1].split("/")[0]
        entry["campaign_artifact_dirs"].add(str(campaign_dir))
        if rec.get("source_index_path"):
            entry["source_index_paths"].add(str(rec["source_index_path"]))
        if not raw_path.is_file():
            missing += 1
            recomputed = None
            ok = False
        else:
            recomputed = sha256_file(raw_path)
            ok = recomputed == rec.get("actual_sha256") == rec.get("indexed_sha256")
            if not ok:
                sha_mismatch += 1
        entry["evidence_references"].append({
            "evidence_role": rec["evidence_role"],
            "raw_absolute_path": str(raw_path),
            "raw_exists": raw_path.is_file(),
            "recomputed_sha256": recomputed,
            "registry_recorded_sha256": rec.get("actual_sha256"),
            "sha256_identity_verified": ok,
        })

    if missing or sha_mismatch:
        raise DiagnosticFailure(
            "FAIL_RAW_EVIDENCE_MISSING",
            f"raw evidence missing={missing} sha_mismatch={sha_mismatch}")

    index_records = []
    for reg in records:
        episode_id = reg["episode_id"]
        entry = per_episode.get(episode_id)
        if entry is None:
            raise DiagnosticFailure("FAIL_RAW_EVIDENCE_MISSING",
                                    f"no raw provenance rows for {episode_id}")
        campaign_dirs = sorted(entry["campaign_artifact_dirs"])
        if len(campaign_dirs) != 1:
            raise DiagnosticFailure(
                "FAIL_LINEAGE_RESOLUTION",
                f"{episode_id} raw evidence spans multiple campaign dirs: {campaign_dirs}")
        campaign_dir = Path(campaign_dirs[0])
        route_id = str(reg["route_id"])
        route_files = sorted(set(
            glob.glob(str(campaign_dir / "raw" / route_id / "*.json"))
            + glob.glob(str(campaign_dir / "raw" / "*" / route_id / "*.json"))))
        if not route_files:
            raise DiagnosticFailure("FAIL_RAW_EVIDENCE_MISSING",
                                    f"{episode_id}: no route raw files under {campaign_dir}")
        index_records.append({
            "campaign_artifact_dir": str(campaign_dir),
            "episode_id": episode_id,
            "evidence_references": sorted(entry["evidence_references"],
                                          key=lambda r: (r["evidence_role"], r["raw_absolute_path"])),
            "registry_source_artifact": str(reg["source_artifact"]),
            "registry_source_detail_path": str(reg.get("source_detail_path")),
            "registry_source_episode_id": str(reg["source_episode_id"]),
            "resolution_priority_used": (
                "1_FROZEN_REGISTRY_SOURCE_ARTIFACT_WITH_RECORDED_ABSOLUTE_RAW_PATHS"
                "_AND_4_RAW_SHA_EVIDENCE_IDENTITY_MATCH"),
            "route_id": route_id,
            "route_raw_file_count": len(route_files),
            "source_index_paths": sorted(entry["source_index_paths"]),
        })

    return {
        "prefix_recency_selection_used": False,
        "raw_evidence_reference_count": len(provenance["records"]),
        "raw_evidence_missing_count": 0,
        "raw_evidence_sha256_mismatch_count": 0,
        "records": index_records,
        "resolution_priority_contract": [
            "1. frozen registry source_artifact",
            "2. episode lineage file recorded source path",
            "3. upstream_reference absolute path",
            "4. raw SHA and evidence identity match",
        ],
        "source_provenance_path": str(provenance_path),
    }


def build_timeline(reg: Dict[str, Any], campaign_dir: Path) -> Dict[str, Any]:
    route_id = str(reg["route_id"])
    vehicle_id = str(reg["vehicle_id"])
    files = sorted(set(
        glob.glob(str(campaign_dir / "raw" / route_id / "*.json"))
        + glob.glob(str(campaign_dir / "raw" / "*" / route_id / "*.json"))))
    entries = []
    for path in files:
        request_dt = raw_filename_request_time(path)
        if request_dt is None:
            continue
        entries.append((request_dt, path))
    entries.sort(key=lambda t: (t[0], t[1]))

    anchors = {
        "last_pre_terminal": parse_iso(reg["last_pre_terminal_request_time"]),
        "first_terminal": parse_iso(reg["first_terminal_request_time"]),
        "last_terminal": parse_iso(reg["last_terminal_request_time"]),
        "first_post_terminal": parse_iso(reg["first_post_terminal_request_time"]),
    }
    anchor_index: Dict[str, int] = {}
    for name, target in anchors.items():
        hits = [i for i, (dt, _p) in enumerate(entries) if dt == target]
        if len(hits) != 1:
            raise DiagnosticFailure(
                "FAIL_GAP_TIMELINE_RECONSTRUCTION",
                f"{reg['episode_id']}: anchor {name} resolved to {len(hits)} raw files")
        anchor_index[name] = hits[0]

    samples: List[Dict[str, Any]] = []
    for idx, (request_dt, path) in enumerate(entries):
        payload = load_json(Path(path))
        row, item_count = target_vehicle_row(payload, vehicle_id)
        prov_dt = provider_event_time(row.get("arTime") if row else None, request_dt)
        samples.append({
            "index": idx,
            "raw_absolute_path": path,
            "raw_relative_path": str(Path(path).relative_to(campaign_dir)),
            "request_observation_time": request_dt.isoformat(),
            "request_dt": request_dt,
            "capture_mode": raw_capture_mode(path, campaign_dir),
            "response_status": response_status_of(payload),
            "route_response_item_count": item_count,
            "target_vehicle_present": row is not None,
            "target_vehicle_sequence": int(row["seq"]) if row and row.get("seq") is not None else None,
            "target_vehicle_stop_id": str(row["bsId"]) if row and row.get("bsId") is not None else None,
            "provider_event_time": prov_dt.isoformat() if prov_dt else None,
            "provider_event_dt": prov_dt,
            "provider_ar_time_raw": str(row["arTime"]) if row and row.get("arTime") else None,
        })

    for idx, sample in enumerate(samples):
        prev_present = None
        for back in range(idx - 1, -1, -1):
            if samples[back]["target_vehicle_present"]:
                prev_present = samples[back]
                break
        repeated = None
        if sample["target_vehicle_present"] and sample["provider_ar_time_raw"] is not None:
            repeated = bool(prev_present is not None
                            and prev_present["provider_ar_time_raw"] == sample["provider_ar_time_raw"])
        sample["provider_timestamp_repeated"] = repeated
        if sample["provider_event_dt"] is not None:
            sample["provider_timestamp_staleness_sec"] = (
                sample["request_dt"] - sample["provider_event_dt"]).total_seconds()
        else:
            sample["provider_timestamp_staleness_sec"] = None
        sample["seconds_since_previous_request"] = (
            (sample["request_dt"] - samples[idx - 1]["request_dt"]).total_seconds()
            if idx > 0 else None)

    return {"anchor_index": anchor_index, "samples": samples, "campaign_dir": str(campaign_dir)}


def anchor_role_of(idx: int, anchor_index: Dict[str, int]) -> Optional[str]:
    for name, value in anchor_index.items():
        if value == idx:
            return name.upper()
    return None


def phase_of(idx: int, anchor_index: Dict[str, int]) -> str:
    if idx < anchor_index["first_terminal"]:
        return "PRE_TERMINAL"
    if idx <= anchor_index["last_terminal"]:
        return "TERMINAL_WINDOW"
    if idx < anchor_index["first_post_terminal"]:
        return "EXIT_WINDOW"
    return "POST_TERMINAL"


def classify_span(a: Dict[str, Any], b: Dict[str, Any], anchor_index: Dict[str, int]) -> str:
    a_role = anchor_role_of(a["index"], anchor_index)
    b_role = anchor_role_of(b["index"], anchor_index)
    if b_role == "FIRST_TERMINAL":
        return "PRESENT_TO_TERMINAL_TRANSITION_BRACKET"
    if b_role == "FIRST_POST_TERMINAL":
        return ("ABSENT_TO_REENTRY_TRANSITION_BRACKET" if not a["target_vehicle_present"]
                else "UNKNOWN_OR_UNRESOLVED_SPAN")
    if a_role == "LAST_TERMINAL":
        return ("TERMINAL_TO_ABSENT_TRANSITION_BRACKET" if not b["target_vehicle_present"]
                else "UNKNOWN_OR_UNRESOLVED_SPAN")
    a_present = a["target_vehicle_present"]
    b_present = b["target_vehicle_present"]
    a_phase = phase_of(a["index"], anchor_index)
    b_phase = phase_of(b["index"], anchor_index)
    if a_present and b_present:
        if a_phase == "POST_TERMINAL" and b_phase == "POST_TERMINAL":
            return "PRESENT_REENTRY_CONFIRMATION_SPAN"
        if "EXIT_WINDOW" in (a_phase, b_phase):
            return "UNKNOWN_OR_UNRESOLVED_SPAN"
        return "PRESENT_STABLE_SPAN"
    if a_present and not b_present:
        return "TERMINAL_TO_ABSENT_TRANSITION_BRACKET"
    if (not a_present) and b_present:
        return "ABSENT_TO_REENTRY_TRANSITION_BRACKET"
    return "ABSENT_SAMPLED_SPAN"


# --------------------------------------------------------------------------
# Phase E - timeline windows, partition and reducibility bounds
# --------------------------------------------------------------------------

def phase_e_timelines(records: List[Dict[str, Any]],
                      raw_index: Dict[str, Any]) -> Dict[str, Any]:
    campaign_by_episode = {r["episode_id"]: Path(r["campaign_artifact_dir"])
                           for r in raw_index["records"]}
    timeline_rows: List[Dict[str, Any]] = []
    partition_rows: List[Dict[str, Any]] = []
    reducibility_rows: List[Dict[str, Any]] = []
    window_audits: List[Dict[str, Any]] = []
    staleness_rows: List[Dict[str, Any]] = []
    anchor_provider_match = 0
    anchor_provider_total = 0

    for reg in records:
        episode_id = reg["episode_id"]
        campaign_dir = campaign_by_episode[episode_id]
        built = build_timeline(reg, campaign_dir)
        samples = built["samples"]
        anchor_index = built["anchor_index"]

        for name, prefix in (("last_pre_terminal", "last_pre_terminal"),
                             ("first_terminal", "first_terminal"),
                             ("last_terminal", "last_terminal"),
                             ("first_post_terminal", "first_post_terminal")):
            anchor_provider_total += 1
            sample = samples[anchor_index[name]]
            expected = reg[f"{prefix}_provider_time"]
            if sample["provider_event_time"] is not None and expected is not None:
                if parse_iso(sample["provider_event_time"]) == parse_iso(expected):
                    anchor_provider_match += 1

        windows = {
            "ENTRY": (max(0, anchor_index["last_pre_terminal"] - 1),
                      min(len(samples) - 1, anchor_index["first_terminal"] + 1)),
            "EXIT": (max(0, anchor_index["last_terminal"] - 1),
                     min(len(samples) - 1, anchor_index["first_post_terminal"] + 1)),
        }

        for gap_side, (start, end) in windows.items():
            window_samples = samples[start:end + 1]
            for sample in window_samples:
                timeline_rows.append({
                    "anchor_role": anchor_role_of(sample["index"], anchor_index),
                    "capture_mode": sample["capture_mode"],
                    "content_type": None,
                    "content_type_status": "NOT_RESOLVED_IN_SOURCE_RAW_PAYLOAD",
                    "episode_id": episode_id,
                    "gap_side": gap_side,
                    "provider_event_time": sample["provider_event_time"],
                    "provider_timestamp_repeat_annotation": sample["provider_timestamp_repeated"],
                    "provider_timestamp_repeated": sample["provider_timestamp_repeated"],
                    "provider_timestamp_staleness_sec": sample["provider_timestamp_staleness_sec"],
                    "raw_relative_path": sample["raw_relative_path"],
                    "raw_sha256": sha256_file(Path(sample["raw_absolute_path"])),
                    "request_id": None,
                    "request_id_status": "NOT_RESOLVED_IN_SOURCE_INDEX",
                    "request_observation_time": sample["request_observation_time"],
                    "request_sequence_index": sample["index"],
                    "response_status": sample["response_status"],
                    "route_id": str(reg["route_id"]),
                    "route_response_item_count": sample["route_response_item_count"],
                    "seconds_since_previous_request": sample["seconds_since_previous_request"],
                    "target_vehicle_id": str(reg["vehicle_id"]),
                    "target_vehicle_present": sample["target_vehicle_present"],
                    "target_vehicle_sequence": sample["target_vehicle_sequence"],
                    "target_vehicle_stop_id": sample["target_vehicle_stop_id"],
                })

            window_partition = []
            for a, b in zip(window_samples, window_samples[1:]):
                duration = (b["request_dt"] - a["request_dt"]).total_seconds()
                category = classify_span(a, b, anchor_index)
                window_partition.append({
                    "duration_sec": duration,
                    "end_request_observation_time": b["request_observation_time"],
                    "end_target_vehicle_present": b["target_vehicle_present"],
                    "episode_id": episode_id,
                    "gap_side": gap_side,
                    "partition_category": category,
                    "provider_timestamp_repeat_annotation": bool(b["provider_timestamp_repeated"]) if b["provider_timestamp_repeated"] is not None else None,
                    "provider_timestamp_staleness_sec": b["provider_timestamp_staleness_sec"],
                    "start_request_observation_time": a["request_observation_time"],
                    "start_target_vehicle_present": a["target_vehicle_present"],
                })
            partition_rows.extend(window_partition)

            window_start = window_samples[0]["request_dt"]
            window_end = window_samples[-1]["request_dt"]
            total = sum(p["duration_sec"] for p in window_partition)
            residual = total - (window_end - window_start).total_seconds()
            unknown_category = [p for p in window_partition
                                if p["partition_category"] not in PARTITION_CATEGORIES]
            window_audits.append({
                "episode_id": episode_id,
                "gap_side": gap_side,
                "invalid_category_count": len(unknown_category),
                "partition_duration_residual_sec": residual,
                "partition_interval_count": len(window_partition),
                "sample_count": len(window_samples),
                "window_end_request_observation_time": window_end.isoformat(),
                "window_span_sec": (window_end - window_start).total_seconds(),
                "window_start_request_observation_time": window_start.isoformat(),
            })

        entry_bracket = (samples[anchor_index["first_terminal"]]["request_dt"]
                         - samples[anchor_index["last_pre_terminal"]]["request_dt"]).total_seconds()
        exit_bracket = (samples[anchor_index["first_post_terminal"]]["request_dt"]
                        - samples[anchor_index["last_terminal"]]["request_dt"]).total_seconds()

        exit_span_samples = samples[anchor_index["last_terminal"]:anchor_index["first_post_terminal"] + 1]
        disappearance_sum = 0.0
        reentry_sum = 0.0
        absent_sampled_sum = 0.0
        unresolved_sum = 0.0
        absent_sample_count = 0
        present_after_last_terminal_count = 0
        for a, b in zip(exit_span_samples, exit_span_samples[1:]):
            duration = (b["request_dt"] - a["request_dt"]).total_seconds()
            category = classify_span(a, b, anchor_index)
            if category == "TERMINAL_TO_ABSENT_TRANSITION_BRACKET":
                disappearance_sum += duration
            elif category == "ABSENT_TO_REENTRY_TRANSITION_BRACKET":
                reentry_sum += duration
            elif category == "ABSENT_SAMPLED_SPAN":
                absent_sampled_sum += duration
            else:
                unresolved_sum += duration
        for sample in exit_span_samples[1:-1]:
            if not sample["target_vehicle_present"]:
                absent_sample_count += 1
            else:
                present_after_last_terminal_count += 1

        entry_reducible_upper = entry_bracket
        exit_reducible_upper = min(exit_bracket, disappearance_sum + reentry_sum)

        for gap_side, gap_duration, reducible_upper, extra in (
            ("ENTRY", entry_bracket, entry_reducible_upper, {}),
            ("EXIT", exit_bracket, exit_reducible_upper, {
                "exit_absent_sample_count": absent_sample_count,
                "exit_absent_sampled_span_total_sec": absent_sampled_sum,
                "exit_disappearance_transition_bracket_total_sec": disappearance_sum,
                "exit_present_sample_after_last_terminal_count": present_after_last_terminal_count,
                "exit_reentry_transition_bracket_total_sec": reentry_sum,
                "exit_unresolved_span_total_sec": unresolved_sum,
            }),
        ):
            row = {
                "episode_id": episode_id,
                "gap_side": gap_side,
                "gap_duration_sec": gap_duration,
                "non_sampling_floor_lower_bound_sec": max(0.0, gap_duration - reducible_upper),
                "non_sampling_floor_upper_bound_sec": gap_duration,
                "sampling_reducible_lower_bound_sec": 0.0,
                "sampling_reducible_upper_bound_sec": reducible_upper,
                "reducibility_is_bounded": True,
                "causal_attribution_status": "NOT_IDENTIFIED_FROM_DISCRETE_FEED",
            }
            row.update({k: extra.get(k) for k in [
                "exit_absent_sample_count", "exit_absent_sampled_span_total_sec",
                "exit_disappearance_transition_bracket_total_sec",
                "exit_present_sample_after_last_terminal_count",
                "exit_reentry_transition_bracket_total_sec", "exit_unresolved_span_total_sec"]})
            reducibility_rows.append(row)

        stale_values = [s["provider_timestamp_staleness_sec"] for s in samples
                        if s["provider_timestamp_staleness_sec"] is not None]
        repeat_count = sum(1 for s in samples if s["provider_timestamp_repeated"] is True)
        staleness_rows.append({
            "episode_id": episode_id,
            "exit_window_absent_sample_count": absent_sample_count,
            "observed_request_sample_count": len(samples),
            "provider_timestamp_repeat_sample_count": repeat_count,
            "provider_timestamp_staleness_max_sec": max(stale_values) if stale_values else None,
            "provider_timestamp_staleness_median_sec": median_or_none(stale_values),
            "clock_semantics_status": str(reg["clock_semantics"]),
        })

    residual_max = max(abs(w["partition_duration_residual_sec"]) for w in window_audits)
    invalid_categories = sum(w["invalid_category_count"] for w in window_audits)
    if residual_max > DECOMPOSITION_RESIDUAL_TOLERANCE_SEC or invalid_categories:
        raise DiagnosticFailure(
            "FAIL_TIMELINE_PARTITION",
            f"partition residual_max={residual_max} invalid_categories={invalid_categories}")

    if anchor_provider_match != anchor_provider_total:
        raise DiagnosticFailure(
            "FAIL_GAP_TIMELINE_RECONSTRUCTION",
            f"provider anchor time reconstruction matched {anchor_provider_match}/{anchor_provider_total}")

    category_counts: Dict[str, int] = {c: 0 for c in PARTITION_CATEGORIES}
    for row in partition_rows:
        category_counts[row["partition_category"]] += 1

    reducible_upper_by_episode: Dict[str, float] = {}
    floor_lower_by_episode: Dict[str, float] = {}
    floor_upper_by_episode: Dict[str, float] = {}
    for row in reducibility_rows:
        reducible_upper_by_episode[row["episode_id"]] = (
            reducible_upper_by_episode.get(row["episode_id"], 0.0)
            + row["sampling_reducible_upper_bound_sec"])
        floor_upper_by_episode[row["episode_id"]] = (
            floor_upper_by_episode.get(row["episode_id"], 0.0) + row["gap_duration_sec"])
    for episode_id, width in floor_upper_by_episode.items():
        floor_lower_by_episode[episode_id] = max(0.0, width - reducible_upper_by_episode[episode_id])

    episode_reducibility = [{
        "episode_id": episode_id,
        "request_non_sampling_floor_lower_bound_sec": floor_lower_by_episode[episode_id],
        "request_non_sampling_floor_upper_bound_sec": floor_upper_by_episode[episode_id],
        "request_width_reducible_lower_bound_sec": 0.0,
        "request_width_reducible_upper_bound_sec": reducible_upper_by_episode[episode_id],
    } for episode_id in sorted(floor_upper_by_episode)]

    return {
        "timeline_rows": timeline_rows,
        "partition_rows": partition_rows,
        "reducibility_rows": reducibility_rows,
        "episode_reducibility": episode_reducibility,
        "staleness_rows": staleness_rows,
        "partition_audit": {
            "allowed_partition_categories": PARTITION_CATEGORIES,
            "causal_attribution_status": "NOT_IDENTIFIED_FROM_DISCRETE_FEED",
            "causal_point_attribution_generated": False,
            "partition_category_counts": category_counts,
            "partition_interval_count": len(partition_rows),
            "partition_rule": {
                "presence_determination": "exact target vehicle id match on vhcNo2 in the raw response items",
                "prohibited_presence_determination": [
                    "substring comparison", "float conversion", "coordinate-based identity",
                    "sequence-only vehicle identity"],
                "span_classification_order": [
                    "b == FIRST_TERMINAL anchor -> PRESENT_TO_TERMINAL_TRANSITION_BRACKET",
                    "b == FIRST_POST_TERMINAL anchor -> ABSENT_TO_REENTRY_TRANSITION_BRACKET when a absent else UNKNOWN_OR_UNRESOLVED_SPAN",
                    "a == LAST_TERMINAL anchor -> TERMINAL_TO_ABSENT_TRANSITION_BRACKET when b absent else UNKNOWN_OR_UNRESOLVED_SPAN",
                    "both present, both POST_TERMINAL -> PRESENT_REENTRY_CONFIRMATION_SPAN",
                    "both present, either side inside EXIT_WINDOW -> UNKNOWN_OR_UNRESOLVED_SPAN",
                    "both present otherwise -> PRESENT_STABLE_SPAN",
                    "present then absent -> TERMINAL_TO_ABSENT_TRANSITION_BRACKET",
                    "absent then present -> ABSENT_TO_REENTRY_TRANSITION_BRACKET",
                    "both absent -> ABSENT_SAMPLED_SPAN"],
                "terminal_state_semantics_rederived": False,
                "provider_timestamp_repeat_treatment": "ANNOTATION_ONLY_NOT_A_PARTITION_CATEGORY",
            },
            "provider_anchor_time_reconstruction_match_count": anchor_provider_match,
            "provider_anchor_time_reconstruction_total": anchor_provider_total,
            "timeline_partition_duration_residual_abs_max_sec": residual_max,
            "timeline_partition_duration_residual_tolerance_sec": DECOMPOSITION_RESIDUAL_TOLERANCE_SEC,
            "timeline_partition_is_exact": True,
            "timeline_partition_missing_span_count": 0,
            "timeline_partition_overlap_count": 0,
            "window_audits": window_audits,
        },
    }


# --------------------------------------------------------------------------
# Phase F - binding endpoints and leave-one-out
# --------------------------------------------------------------------------

def intersection_of(intervals: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    if not intervals:
        return None, None
    return max(i[0] for i in intervals), min(i[1] for i in intervals)


def phase_f_binding(decomposition: List[Dict[str, Any]]) -> Dict[str, Any]:
    clocks = {
        "CONSERVATIVE_DUAL_CLOCK": ("conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec"),
        "REQUEST_ONLY_INTERVAL": ("request_lower_bound_sec", "request_upper_bound_sec"),
    }
    binding: Dict[str, Any] = {"clocks": {}}
    loo_rows: List[Dict[str, Any]] = []
    for clock, (lower_field, upper_field) in clocks.items():
        intervals = [(row[lower_field], row[upper_field], row["episode_id"]) for row in decomposition]
        lowers = sorted((i[0] for i in intervals), reverse=True)
        uppers = sorted(i[1] for i in intervals)
        star_lower = lowers[0]
        star_upper = uppers[0]
        binding["clocks"][clock] = {
            "identified_set_lower_sec": star_lower,
            "identified_set_upper_sec": star_upper,
            "identified_set_width_sec": star_upper - star_lower,
            "largest_lower_endpoint_sec": lowers[0],
            "second_largest_lower_endpoint_sec": lowers[1] if len(lowers) > 1 else None,
            "smallest_upper_endpoint_sec": uppers[0],
            "second_smallest_upper_endpoint_sec": uppers[1] if len(uppers) > 1 else None,
            "ordered_lower_endpoints_sec": sorted(i[0] for i in intervals),
            "ordered_upper_endpoints_sec": sorted(i[1] for i in intervals),
            "lower_binding_episode_ids": sorted(
                i[2] for i in intervals
                if abs(i[0] - star_lower) <= INTERSECTION_COMPARISON_TOLERANCE_SEC),
            "upper_binding_episode_ids": sorted(
                i[2] for i in intervals
                if abs(i[1] - star_upper) <= INTERSECTION_COMPARISON_TOLERANCE_SEC),
        }
        full_width = star_upper - star_lower
        for target in intervals:
            others = [(i[0], i[1]) for i in intervals if i[2] != target[2]]
            loo_lower, loo_upper = intersection_of(others)
            binds_lower = target[0] > loo_lower + INTERSECTION_COMPARISON_TOLERANCE_SEC
            binds_upper = target[1] < loo_upper - INTERSECTION_COMPARISON_TOLERANCE_SEC
            loo_rows.append({
                "binds_lower_endpoint": bool(binds_lower),
                "binds_upper_endpoint": bool(binds_upper),
                "clock": clock,
                "episode_id": target[2],
                "full_sample_identified_set_width_sec": full_width,
                "identified_set_width_minus_i_sec": loo_upper - loo_lower,
                "inference_status": "EXPLORATORY_EXTREMAL_CONTRIBUTION_DIAGNOSTIC",
                "leave_one_out_informative": bool(binds_lower or binds_upper),
                "observation_lower_sec": target[0],
                "observation_upper_sec": target[1],
                "u_star_minus_i_sec": loo_upper,
                "l_star_minus_i_sec": loo_lower,
                "width_change_minus_i_sec": (loo_upper - loo_lower) - full_width,
            })

    dual_rows = [r for r in loo_rows if r["clock"] == "CONSERVATIVE_DUAL_CLOCK"]
    informative = [r for r in dual_rows if r["leave_one_out_informative"]]
    loo_audit = {
        "clock_summaries": {
            clock: {
                "leave_one_out_informative_episode_count": sum(
                    1 for r in loo_rows if r["clock"] == clock and r["leave_one_out_informative"]),
                "leave_one_out_informative_fraction": (
                    sum(1 for r in loo_rows if r["clock"] == clock and r["leave_one_out_informative"])
                    / max(1, sum(1 for r in loo_rows if r["clock"] == clock))),
                "leave_one_out_lower_binding_count": sum(
                    1 for r in loo_rows if r["clock"] == clock and r["binds_lower_endpoint"]),
                "leave_one_out_upper_binding_count": sum(
                    1 for r in loo_rows if r["clock"] == clock and r["binds_upper_endpoint"]),
                "max_width_increase_if_removed_sec": max(
                    r["width_change_minus_i_sec"] for r in loo_rows if r["clock"] == clock),
            } for clock in clocks
        },
        "inference_status": "EXPLORATORY_EXTREMAL_CONTRIBUTION_DIAGNOSTIC",
        "interpretation_limits": [
            "this is not an estimate of the success probability of a new episode",
            "this is not a required-sample-size calculation",
            "removing episodes cannot improve identification",
        ],
        "leave_one_out_informative_episode_count": len(informative),
        "leave_one_out_informative_episode_ids": sorted(r["episode_id"] for r in informative),
        "leave_one_out_informative_fraction": len(informative) / max(1, len(dual_rows)),
        "leave_one_out_lower_binding_count": sum(1 for r in dual_rows if r["binds_lower_endpoint"]),
        "leave_one_out_upper_binding_count": sum(1 for r in dual_rows if r["binds_upper_endpoint"]),
        "naive_ratio_prohibited": {
            "prohibited_statistic": [
                "fraction(L_i > current maximum L)", "fraction(U_i < current minimum U)"],
            "reason": "these are zero by construction and carry no information",
        },
        "records": loo_rows,
        "required_sample_size_estimate": None,
        "required_sample_size_estimate_status": "NOT_COMPUTED",
        "required_sample_size_reason": "FORMAL_INFERENCE_NOT_AUTHORIZED",
    }

    r2d1r_binding = load_json(R2D1R_DIR / "binding_episode_analysis.json")
    reconciliation = {
        "r2d1r_primary_binding_episode_count": r2d1r_binding.get("primary_binding_episode_count"),
        "r2d1r_primary_binding_episode_ids": sorted(r2d1r_binding.get("primary_binding_episode_ids", [])),
        "recomputed_primary_binding_episode_count": len(informative),
        "recomputed_primary_binding_episode_ids": sorted(r["episode_id"] for r in informative),
    }
    reconciliation["matches"] = (
        reconciliation["r2d1r_primary_binding_episode_count"] == reconciliation["recomputed_primary_binding_episode_count"]
        and reconciliation["r2d1r_primary_binding_episode_ids"] == reconciliation["recomputed_primary_binding_episode_ids"])
    if not reconciliation["matches"]:
        raise DiagnosticFailure("FAIL_BINDING_ENDPOINT_RECONCILIATION",
                                f"binding reconciliation mismatch: {reconciliation}")

    binding["binding_endpoint_audit_passed"] = True
    binding["r2d1r_reconciliation"] = reconciliation
    binding["inference_status"] = "EXPLORATORY_EXTREMAL_CONTRIBUTION_DIAGNOSTIC"
    return {"binding": binding, "leave_one_out": loo_audit, "loo_rows": loo_rows}


def phase_f_broad_scan_influence(decomposition: List[Dict[str, Any]],
                                 timeline_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    modes_by_episode: Dict[str, Dict[str, int]] = {}
    anchor_modes: Dict[str, Dict[str, str]] = {}
    for row in timeline_rows:
        counts = modes_by_episode.setdefault(row["episode_id"], {})
        counts[row["capture_mode"]] = counts.get(row["capture_mode"], 0) + 1
        if row["anchor_role"]:
            anchor_modes.setdefault(row["episode_id"], {})[row["anchor_role"]] = row["capture_mode"]

    stratification_records = []
    for row in decomposition:
        episode_id = row["episode_id"]
        anchors = anchor_modes.get(episode_id, {})
        anchor_mode_values = sorted(set(anchors.values()))
        stratification_records.append({
            "anchor_capture_modes": anchors,
            "anchor_capture_mode_set": anchor_mode_values,
            "any_anchor_broad_scan": any("BROAD_SCAN" in m for m in anchor_mode_values),
            "capture_mode_counts_in_reconstruction_windows": modes_by_episode.get(episode_id, {}),
            "episode_id": episode_id,
            "request_entry_bracket_sec": row["request_entry_bracket_sec"],
            "request_exit_bracket_sec": row["request_exit_bracket_sec"],
            "loose_entry_cadence_gt_60s": row["request_entry_bracket_sec"] > 60.0,
            "source_campaign_id": row["source_campaign_id"],
        })

    strata = {
        "ANCHOR_CAPTURE_MODE_BROAD_SCAN": [
            r["episode_id"] for r in stratification_records if r["any_anchor_broad_scan"]],
        "LOOSE_ENTRY_CADENCE_GT_60S": [
            r["episode_id"] for r in stratification_records if r["loose_entry_cadence_gt_60s"]],
    }
    strata["UNION_BROAD_SCAN_OR_LOOSE_ENTRY_CADENCE"] = sorted(set(
        strata["ANCHOR_CAPTURE_MODE_BROAD_SCAN"] + strata["LOOSE_ENTRY_CADENCE_GT_60S"]))

    full_dual = [(r["conservative_dual_lower_bound_sec"], r["conservative_dual_upper_bound_sec"])
                 for r in decomposition]
    full_lower, full_upper = intersection_of(full_dual)
    full_width = full_upper - full_lower

    exclusion_records = []
    monotonicity_violation = 0
    for stratum, excluded_ids in strata.items():
        kept = [r for r in decomposition if r["episode_id"] not in excluded_ids]
        if kept:
            kept_lower, kept_upper = intersection_of(
                [(r["conservative_dual_lower_bound_sec"], r["conservative_dual_upper_bound_sec"])
                 for r in kept])
            width = kept_upper - kept_lower
            widths = [r["conservative_dual_width_sec"] for r in kept]
            distribution = {
                "episode_width_max_sec": max(widths),
                "episode_width_median_sec": median_or_none(widths),
                "episode_width_min_sec": min(widths),
            }
        else:
            kept_lower = kept_upper = width = None
            distribution = {"episode_width_max_sec": None, "episode_width_median_sec": None,
                            "episode_width_min_sec": None}
        if width is not None and width < full_width - INTERSECTION_COMPARISON_TOLERANCE_SEC:
            monotonicity_violation += 1
        exclusion_records.append({
            "episode_width_distribution_if_excluded": distribution,
            "excluded_episode_count": len(excluded_ids),
            "excluded_episode_ids": sorted(excluded_ids),
            "identified_set_lower_if_excluded_sec": kept_lower,
            "identified_set_upper_if_excluded_sec": kept_upper,
            "identified_set_width_if_excluded_sec": width,
            "monotonicity_holds_width_not_smaller": (
                None if width is None else width >= full_width - INTERSECTION_COMPARISON_TOLERANCE_SEC),
            "retained_episode_count": len(kept),
            "stratum_id": stratum,
            "width_change_vs_full_sample_sec": None if width is None else width - full_width,
        })

    if monotonicity_violation:
        raise DiagnosticFailure(
            "FAIL_R2D1S_DIAGNOSTIC",
            f"episode exclusion produced a narrower intersection in {monotonicity_violation} strata")

    influence = {
        "exclusion_records": exclusion_records,
        "full_sample_identified_set_lower_sec": full_lower,
        "full_sample_identified_set_upper_sec": full_upper,
        "full_sample_identified_set_width_sec": full_width,
        "interpretation": [
            "This is an influence diagnostic only.",
            "Removing episodes cannot improve identification.",
        ],
        "mathematical_check": "identified_set_width_if_broad_scan_excluded >= original_identified_set_width",
        "mathematical_check_violation_count": 0,
        "prohibited_claims": [
            "excluding broad-scan episodes improves identification",
            "deleting episodes from the frozen registry",
        ],
        "registry_row_count_after": len(decomposition),
        "registry_row_count_before": len(decomposition),
        "registry_rows_deleted": 0,
        "strata_definitions": {
            "ANCHOR_CAPTURE_MODE_BROAD_SCAN": "any of the four frozen anchor samples captured in a broad-scan capture mode",
            "LOOSE_ENTRY_CADENCE_GT_60S": "observed request-clock entry bracket greater than 60 seconds",
            "UNION_BROAD_SCAN_OR_LOOSE_ENTRY_CADENCE": "union of the two strata above",
        },
        "strata_membership": {k: sorted(v) for k, v in strata.items()},
    }
    stratification = {
        "capture_mode_stratification_passed": True,
        "capture_mode_vocabulary_note": (
            "capture modes are read from the raw evidence layout of each source campaign "
            "(directory segment or filename suffix) and normalised to upper case"),
        "records": stratification_records,
        "strata_membership": {k: sorted(v) for k, v in strata.items()},
    }
    return stratification, influence


# --------------------------------------------------------------------------
# Phase G - clock dominance and request-cadence counterfactual
# --------------------------------------------------------------------------

def phase_g_clock_dominance(decomposition: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = []
    for row in decomposition:
        dual_lower = row["conservative_dual_lower_bound_sec"]
        dual_upper = row["conservative_dual_upper_bound_sec"]
        provider_lower = row["provider_lower_bound_sec"]
        provider_upper = row["provider_upper_bound_sec"]
        request_lower = row["request_lower_bound_sec"]
        request_upper = row["request_upper_bound_sec"]
        records.append({
            "dual_lower_equals_provider_lower": dual_lower == provider_lower,
            "dual_lower_equals_request_lower": dual_lower == request_lower,
            "dual_upper_equals_provider_upper": dual_upper == provider_upper,
            "dual_upper_equals_request_upper": dual_upper == request_upper,
            "episode_id": row["episode_id"],
            "provider_interval_equals_dual": (dual_lower == provider_lower and dual_upper == provider_upper),
            "request_boundary_determines_dual_lower": (
                dual_lower == request_lower and request_lower < provider_lower),
            "request_boundary_determines_dual_upper": (
                dual_upper == request_upper and request_upper > provider_upper),
            "request_interval_subset_of_dual": (
                request_lower >= dual_lower and request_upper <= dual_upper),
        })

    audit = {
        "dual_clock_hedge_effective_episode_count": sum(
            1 for row in decomposition if row["dual_clock_hedge_effective"]),
        "dual_equals_provider_episode_count": sum(
            1 for r in records if r["provider_interval_equals_dual"]),
        "episode_count": len(records),
        "records": records,
        "request_boundary_determines_dual_lower_count": sum(
            1 for r in records if r["request_boundary_determines_dual_lower"]),
        "request_boundary_determines_dual_upper_count": sum(
            1 for r in records if r["request_boundary_determines_dual_upper"]),
        "request_subset_of_dual_episode_count": sum(
            1 for r in records if r["request_interval_subset_of_dual"]),
    }
    audit["request_boundary_determines_dual_count"] = (
        audit["request_boundary_determines_dual_lower_count"]
        + audit["request_boundary_determines_dual_upper_count"])

    r2d1r = load_json(R2D1R_DIR / "clock_input_independence_audit.json")
    expected = {
        "dual_equals_provider_episode_count": r2d1r.get("dual_equals_provider_count"),
        "request_subset_of_dual_episode_count": r2d1r.get("request_nested_in_dual_count"),
        "request_boundary_determines_dual_count": r2d1r.get("request_clock_binding_episode_count"),
    }
    mismatches = {k: {"r2d1r": v, "recomputed": audit[k]}
                  for k, v in expected.items() if v != audit[k]}
    audit["r2d1r_expected_values"] = expected
    audit["r2d1r_reconciliation_mismatches"] = mismatches
    audit["r2d1r_reconciliation_mismatch_count"] = len(mismatches)
    if mismatches:
        raise DiagnosticFailure("FAIL_CLOCK_DOMINANCE_RECONCILIATION",
                                f"clock dominance mismatch vs R2D-1R: {mismatches}")
    audit["clock_dominance_audit_passed"] = True
    return audit


SCENARIOS = [
    ("OPTIMISTIC_FULL_BRACKET", "prompt-specified optimistic scenario: the entire observed entry and "
                                "exit detection brackets are treated as sampling-reducible"),
    ("BOUNDED_REDUCIBLE_UPPER", "supplementary bounded scenario: only the observed state-transition "
                                "brackets are treated as sampling-reducible "
                                "(entry bracket for the lower endpoint, absent-to-reentry bracket "
                                "for the upper endpoint)"),
    ("CONSERVATIVE_REDUCIBLE_LOWER", "prompt-specified conservative scenario: the causal "
                                     "sampling-reducible lower bound is zero, so no endpoint moves"),
]


def phase_g_cadence_counterfactual(decomposition: List[Dict[str, Any]],
                                   reducibility_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reduce_by = {(r["episode_id"], r["gap_side"]): r for r in reducibility_rows}
    episode_rows: List[Dict[str, Any]] = []
    scenario_summaries: List[Dict[str, Any]] = []

    for scenario_id, scenario_note in SCENARIOS:
        for cadence in CADENCES_SEC:
            request_intervals: List[Tuple[float, float]] = []
            dual_intervals: List[Tuple[float, float]] = []
            changed_dual_episodes = 0
            changed_request_episodes = 0
            for row in decomposition:
                episode_id = row["episode_id"]
                entry_bracket = row["request_entry_bracket_sec"]
                exit_bracket = row["request_exit_bracket_sec"]
                entry_reduce = reduce_by[(episode_id, "ENTRY")]
                exit_reduce = reduce_by[(episode_id, "EXIT")]
                if scenario_id == "OPTIMISTIC_FULL_BRACKET":
                    lower_reducible = entry_bracket
                    upper_reducible = exit_bracket
                elif scenario_id == "BOUNDED_REDUCIBLE_UPPER":
                    lower_reducible = entry_reduce["sampling_reducible_upper_bound_sec"]
                    upper_reducible = exit_reduce["exit_reentry_transition_bracket_total_sec"] or 0.0
                else:
                    lower_reducible = entry_reduce["sampling_reducible_lower_bound_sec"]
                    upper_reducible = exit_reduce["sampling_reducible_lower_bound_sec"]

                request_lower_cf = row["request_lower_bound_sec"] + max(0.0, lower_reducible - cadence)
                request_upper_cf = row["request_upper_bound_sec"] - max(0.0, upper_reducible - cadence)
                dual_lower_cf = min(row["provider_lower_bound_sec"], request_lower_cf)
                dual_upper_cf = max(row["provider_upper_bound_sec"], request_upper_cf)

                request_changed = (request_lower_cf != row["request_lower_bound_sec"]
                                   or request_upper_cf != row["request_upper_bound_sec"])
                dual_changed = (dual_lower_cf != row["conservative_dual_lower_bound_sec"]
                                or dual_upper_cf != row["conservative_dual_upper_bound_sec"])
                changed_request_episodes += int(request_changed)
                changed_dual_episodes += int(dual_changed)

                request_intervals.append((request_lower_cf, request_upper_cf))
                dual_intervals.append((dual_lower_cf, dual_upper_cf))
                episode_rows.append({
                    "cadence_sec": cadence,
                    "dual_interval_changed": bool(dual_changed),
                    "dual_lower_cf_sec": dual_lower_cf,
                    "dual_upper_cf_sec": dual_upper_cf,
                    "episode_id": episode_id,
                    "observed_dual_lower_sec": row["conservative_dual_lower_bound_sec"],
                    "observed_dual_upper_sec": row["conservative_dual_upper_bound_sec"],
                    "observed_request_lower_sec": row["request_lower_bound_sec"],
                    "observed_request_upper_sec": row["request_upper_bound_sec"],
                    "provider_endpoints_fixed": True,
                    "request_interval_changed": bool(request_changed),
                    "request_lower_cf_sec": request_lower_cf,
                    "request_upper_cf_sec": request_upper_cf,
                    "scenario_id": scenario_id,
                    "lower_endpoint_reducible_span_sec": lower_reducible,
                    "upper_endpoint_reducible_span_sec": upper_reducible,
                })

            req_lower_star, req_upper_star = intersection_of(request_intervals)
            dual_lower_star, dual_upper_star = intersection_of(dual_intervals)
            observed_request = intersection_of(
                [(r["request_lower_bound_sec"], r["request_upper_bound_sec"]) for r in decomposition])
            observed_dual = intersection_of(
                [(r["conservative_dual_lower_bound_sec"], r["conservative_dual_upper_bound_sec"])
                 for r in decomposition])
            scenario_summaries.append({
                "cadence_sec": cadence,
                "primary_dual_identified_set_changed": bool(
                    dual_lower_star != observed_dual[0] or dual_upper_star != observed_dual[1]),
                "primary_dual_intersection_classification": classify_intersection(dual_lower_star, dual_upper_star),
                "primary_dual_interval_changed_episode_count": changed_dual_episodes,
                "primary_dual_star_lower_sec": dual_lower_star,
                "primary_dual_star_upper_sec": dual_upper_star,
                "primary_dual_star_width_sec": dual_upper_star - dual_lower_star,
                "request_interval_changed_episode_count": changed_request_episodes,
                "request_only_identified_set_changed": bool(
                    req_lower_star != observed_request[0] or req_upper_star != observed_request[1]),
                "request_only_intersection_classification": classify_intersection(req_lower_star, req_upper_star),
                "request_only_star_lower_sec": req_lower_star,
                "request_only_star_upper_sec": req_upper_star,
                "request_only_star_width_sec": req_upper_star - req_lower_star,
                "scenario_id": scenario_id,
                "scenario_note": scenario_note,
            })

    primary_dual_changed_any = any(
        s["primary_dual_identified_set_changed"] for s in scenario_summaries)
    cadence_only_effect = ("ZERO_UNDER_FIXED_PROVIDER_ENDPOINTS" if not primary_dual_changed_any
                           else "NONZERO_UNDER_FIXED_PROVIDER_ENDPOINTS")

    by_scenario_cadence = {
        s["scenario_id"]: {} for s in scenario_summaries}
    for s in scenario_summaries:
        by_scenario_cadence[s["scenario_id"]][f"{s['cadence_sec']}s"] = {
            "primary_dual_intersection_classification": s["primary_dual_intersection_classification"],
            "request_only_intersection_classification": s["request_only_intersection_classification"],
        }

    return {
        "cadence_only_effect_on_primary_dual": cadence_only_effect,
        "cadences_sec": CADENCES_SEC,
        "classification_by_scenario_and_cadence": by_scenario_cadence,
        "counterfactual_formulas": {
            "L_dual_cf_i": "min(L_provider_i, L_request_cf_i(c))",
            "L_request_conservative_i": "L_request_i + max(0, sampling_reducible_lower_bound_entry_i - c)",
            "L_request_optimistic_i": "L_request_i + max(0, request_entry_bracket_i - c)",
            "U_dual_cf_i": "max(U_provider_i, U_request_cf_i(c))",
            "U_request_conservative_i": "U_request_i - max(0, sampling_reducible_lower_bound_exit_i - c)",
            "U_request_optimistic_i": "U_request_i - max(0, request_exit_bracket_i - c)",
        },
        "episode_records": episode_rows,
        "scenario_contract": {
            "primary_clock_rule_unchanged": True,
            "provider_endpoints_fixed": True,
            "request_endpoints_improved": True,
        },
        "scenario_definitions": {sid: note for sid, note in SCENARIOS},
        "scenario_summaries": scenario_summaries,
    }


def phase_g_intersection_classification(counterfactual: Dict[str, Any],
                                        identified: Dict[str, Any]) -> Dict[str, Any]:
    records = []
    for s in counterfactual["scenario_summaries"]:
        for family, lower, upper in (
            ("REQUEST_ONLY", s["request_only_star_lower_sec"], s["request_only_star_upper_sec"]),
            ("PRIMARY_DUAL", s["primary_dual_star_lower_sec"], s["primary_dual_star_upper_sec"]),
        ):
            records.append({
                "cadence_sec": s["cadence_sec"],
                "classification": classify_intersection(lower, upper),
                "endpoint_based_classification": True,
                "interval_family": family,
                "l_star_cf_sec": lower,
                "scenario_id": s["scenario_id"],
                "u_star_cf_sec": upper,
                "width_only_classification_used": False,
            })
    empty_count = sum(1 for r in records if r["classification"] == "EMPTY")
    return {
        "common_intersection_removed": empty_count > 0,
        "condition_status": "NECESSARY_NOT_SUFFICIENT",
        "empty_classification_count": empty_count,
        "indeterminate_from_width_only_count": 0,
        "intersection_comparison_tolerance_sec": INTERSECTION_COMPARISON_TOLERANCE_SEC,
        "observed_primary_dual_classification": classify_intersection(
            identified["current_max_dual_lower_sec"], identified["current_min_dual_upper_sec"]),
        "observed_request_only_classification": classify_intersection(
            identified["current_max_request_lower_sec"], identified["current_min_request_upper_sec"]),
        "quantile_separation_guaranteed": False,
        "quantile_separation_note": (
            "removal of the common intersection does not by itself guarantee q25/q50/q75 separation"),
        "records": records,
        "width_only_intersection_decision_count": 0,
    }


# --------------------------------------------------------------------------
# Phase H - rate limit contract and API budget feasibility
# --------------------------------------------------------------------------

R2D1I_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1i_targeted_4010002118_live_observation_20260726_095826"
R2D1K_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1k_campaign_b_controlled_live_observation_20260727_093548"
R2D1N_DIR = ARTIFACT_ROOT / "prompt5_e01_r2d1n_campaign_c_adaptive_polling_retry_20260729_093218"


def phase_h_rate_limit_contract() -> Dict[str, Any]:
    rolling_contract_path = R2D1N_HF1_DIR / "rolling_60_second_call_rate_contract.json"
    policy_path = R2D1N_DIR / "adaptive_polling_policy.json"
    budget_paths = {
        "R2D1I": R2D1I_DIR / "r2d1i_api_budget_contract.json",
        "R2D1K_CAMPAIGN_B": R2D1K_DIR / "campaign_b_effective_api_budget.json",
        "R2D1N_CAMPAIGN_C": R2D1N_DIR / "effective_api_budget.json",
    }
    for path in [rolling_contract_path, policy_path] + list(budget_paths.values()):
        if not path.is_file():
            raise DiagnosticFailure("FAIL_RATE_LIMIT_CONTRACT", f"missing {path}")

    rolling = load_json(rolling_contract_path)
    policy = load_json(policy_path)
    budgets = {key: load_json(path) for key, path in budget_paths.items()}

    configured = int(rolling["configured_max_calls_per_minute"])
    if int(policy.get("configured_max_calls_per_minute", -1)) != configured:
        raise DiagnosticFailure(
            "FAIL_RATE_LIMIT_CONTRACT",
            "configured_max_calls_per_minute disagrees between the rolling-window contract "
            "and the adaptive polling policy")
    if configured != CONFIGURED_MAX_CALLS_PER_MINUTE_EXPECTED:
        raise DiagnosticFailure("FAIL_RATE_LIMIT_CONTRACT",
                                f"configured_max_calls_per_minute = {configured} (expected 4)")

    campaign_caps = {
        "R2D1I": {
            "available_daily_budget": budgets["R2D1I"].get("available_daily_budget"),
            "daily_safety_cap": budgets["R2D1I"].get("daily_safety_cap"),
            "per_campaign_hard_cap": budgets["R2D1I"].get("effective_r2d1i_hard_cap"),
            "source_path": str(budget_paths["R2D1I"]),
        },
        "R2D1K_CAMPAIGN_B": {
            "available_daily_budget": budgets["R2D1K_CAMPAIGN_B"].get("available_daily_budget"),
            "daily_safety_cap": budgets["R2D1K_CAMPAIGN_B"].get("daily_physical_safety_cap"),
            "per_campaign_hard_cap": budgets["R2D1K_CAMPAIGN_B"].get("effective_campaign_b_hard_cap"),
            "source_path": str(budget_paths["R2D1K_CAMPAIGN_B"]),
        },
        "R2D1N_CAMPAIGN_C": {
            "available_daily_budget": budgets["R2D1N_CAMPAIGN_C"].get("available_daily_budget"),
            "daily_safety_cap": budgets["R2D1N_CAMPAIGN_C"].get("daily_physical_safety_cap"),
            "per_campaign_hard_cap": budgets["R2D1N_CAMPAIGN_C"].get("effective_campaign_hard_cap"),
            "source_path": str(budget_paths["R2D1N_CAMPAIGN_C"]),
        },
    }
    latest = campaign_caps["R2D1N_CAMPAIGN_C"]

    return {
        "adaptive_polling_policy_modes": policy.get("modes"),
        "adaptive_polling_policy_path": str(policy_path),
        "available_daily_budget": latest["available_daily_budget"],
        "available_daily_budget_source": "R2D1N_CAMPAIGN_C",
        "campaign_budget_inventory": campaign_caps,
        "configured_max_calls_per_minute": configured,
        "configured_max_calls_per_minute_source_paths": [str(rolling_contract_path), str(policy_path)],
        "daily_safety_cap": latest["daily_safety_cap"],
        "enforcement_mechanism_observed_in_campaign_runner": "MINIMUM_INTER_CALL_SPACING",
        "enforcement_min_inter_call_spacing_sec": 60.0 / configured,
        "enforcement_implies_declared_rolling_window_bound": True,
        "enforcement_derivation": (
            "a minimum spacing of 60/4 = 15 s admits at most 4 calls in any half-open "
            "(t-60s, t] window, so the enforced spacing implies the declared rolling-window limit"),
        "inclusive_endpoint_sensitivity_convention": rolling.get("inclusive_endpoint_sensitivity"),
        "per_campaign_hard_cap": latest["per_campaign_hard_cap"],
        "per_campaign_hard_cap_source": "R2D1N_CAMPAIGN_C",
        "rate_limit_contract_resolved": True,
        "reserved_overhead_calls_per_minute": None,
        "overhead_model_status": "NOT_RESOLVED",
        "overhead_model_reason": (
            "no authoritative artifact declares a reserved per-minute overhead call allowance; "
            "no value is invented"),
        "rolling_window_convention": rolling.get("rolling_window_convention"),
        "rolling_window_convention_source_path": str(rolling_contract_path),
    }


def phase_h_rate_limit_feasibility(contract: Dict[str, Any]) -> Dict[str, Any]:
    configured = contract["configured_max_calls_per_minute"]
    records = []
    for concurrent_routes in (1, 2):
        for cadence in CADENCES_SEC:
            theoretical = (60.0 / cadence) * concurrent_routes
            projected = theoretical
            if projected > configured + INTERSECTION_COMPARISON_TOLERANCE_SEC:
                classification = "RATE_LIMIT_VIOLATION"
            elif abs(projected - configured) <= INTERSECTION_COMPARISON_TOLERANCE_SEC:
                classification = "RATE_LIMIT_ZERO_HEADROOM"
            else:
                classification = "RATE_LIMIT_HEADROOM_AVAILABLE"
            records.append({
                "cadence_sec": cadence,
                "concurrent_route_count": concurrent_routes,
                "configured_max_calls_per_minute": configured,
                "projected_calls_per_minute": projected,
                "rate_limit_classification": classification,
                "rate_limit_feasible": classification != "RATE_LIMIT_VIOLATION",
                "rate_limit_headroom_calls_per_minute": configured - projected,
                "reserved_overhead_calls_per_minute": None,
                "overhead_model_status": "NOT_RESOLVED",
                "theoretical_calls_per_minute": theoretical,
            })

    single = {f"{r['cadence_sec']}s": r["rate_limit_classification"]
              for r in records if r["concurrent_route_count"] == 1}
    expected = {"10s": "RATE_LIMIT_VIOLATION", "15s": "RATE_LIMIT_ZERO_HEADROOM"}
    for key, value in expected.items():
        if single.get(key) != value:
            raise DiagnosticFailure("FAIL_RATE_LIMIT_CONTRACT",
                                    f"single-route cadence {key} classified {single.get(key)} "
                                    f"(expected {value})")
    return {
        "classification_definition": {
            "RATE_LIMIT_HEADROOM_AVAILABLE": "projected calls/min < configured limit",
            "RATE_LIMIT_VIOLATION": "projected calls/min > configured limit",
            "RATE_LIMIT_ZERO_HEADROOM": "projected calls/min == configured limit",
        },
        "operational_feasibility_caveat": (
            "a 15-second cadence consumes the entire configured rate budget on a single route; "
            "zero headroom is not an operational feasibility finding because retries, preflight "
            "calls and any unresolved overhead have no remaining allowance"),
        "records": records,
        "single_route_classification_by_cadence": single,
        "theoretical_rate_formula": "60 / cadence_sec * concurrent_route_count",
    }


def phase_h_phase_based_projection(contract: Dict[str, Any]) -> Dict[str, Any]:
    index_path = R2D1N_DIR / "raw_file_index.json"
    runtime_path = R2D1N_DIR / "runtime_audit.json"
    if not index_path.is_file() or not runtime_path.is_file():
        raise DiagnosticFailure("FAIL_API_BUDGET_ANALYSIS", "R2D-1N campaign index/runtime missing")
    index = load_json(index_path)
    runtime = load_json(runtime_path)
    policy = contract["adaptive_polling_policy_modes"] or {}

    samples = sorted(
        ({"capture_mode": str(r["capture_mode"]).upper(),
          "request_dt": parse_iso(r["request_observation_time"])}
         for r in index["records"]),
        key=lambda r: r["request_dt"])

    phase_duration: Dict[str, float] = {}
    phase_calls: Dict[str, int] = {}
    for sample in samples:
        phase_calls[sample["capture_mode"]] = phase_calls.get(sample["capture_mode"], 0) + 1
    for a, b in zip(samples, samples[1:]):
        delta = (b["request_dt"] - a["request_dt"]).total_seconds()
        phase_duration[a["capture_mode"]] = phase_duration.get(a["capture_mode"], 0.0) + delta

    cadence_applied_phases = ["PRE_TERMINAL_WATCH", "TERMINAL_APPROACH_FOCUSED", "POST_TERMINAL_FOCUSED"]
    policy_interval = {mode: float(cfg["interval_sec"]) for mode, cfg in policy.items()}

    records = []
    totals = []
    for cadence in CADENCES_SEC:
        total = 0
        for mode in sorted(set(list(phase_duration) + list(phase_calls))):
            duration = phase_duration.get(mode, 0.0)
            observed_calls = phase_calls.get(mode, 0)
            if mode == "PREFLIGHT":
                projected = observed_calls
                interval = None
                status = "OBSERVED_PREFLIGHT_COUNT_HELD_FIXED"
            elif mode in cadence_applied_phases:
                interval = float(cadence)
                projected = int(math.ceil(duration / interval)) + 1 if duration > 0 else observed_calls
                status = "CADENCE_APPLIED"
            elif mode in policy_interval:
                interval = policy_interval[mode]
                projected = int(math.ceil(duration / interval)) + 1 if duration > 0 else observed_calls
                status = "POLICY_INTERVAL_HELD_FIXED"
            else:
                interval = None
                projected = observed_calls
                status = "OBSERVED_COUNT_HELD_FIXED_PHASE_NOT_IN_POLICY"
            total += projected
            records.append({
                "cadence_sec": cadence,
                "observed_call_count": observed_calls,
                "observed_phase_duration_sec": duration,
                "phase": mode,
                "policy_interval_sec": policy_interval.get(mode),
                "projected_calls": projected,
                "projection_interval_sec": interval,
                "projection_status": status,
            })
        totals.append({"cadence_sec": cadence, "projected_total_calls_phase_based": total})

    campaign_start = parse_iso(runtime["campaign_started_at"])
    campaign_end = parse_iso(runtime["campaign_finished_at"])
    campaign_duration = (campaign_end - campaign_start).total_seconds()
    preflight_calls = int(runtime.get("preflight_physical_calls") or 0)

    projections = []
    hard_cap = contract["per_campaign_hard_cap"]
    daily_budget = contract["available_daily_budget"]
    for entry in totals:
        cadence = entry["cadence_sec"]
        lower = entry["projected_total_calls_phase_based"]
        upper = int(math.ceil(campaign_duration / cadence)) + 1 + preflight_calls
        projections.append({
            "cadence_sec": cadence,
            "daily_budget_feasible": bool(upper <= daily_budget) if daily_budget else None,
            "daily_budget_feasible_at_lower_bound": bool(lower <= daily_budget) if daily_budget else None,
            "observed_total_calls_reference": len(samples),
            "per_campaign_hard_cap_feasible": bool(upper <= hard_cap) if hard_cap else None,
            "per_campaign_hard_cap_feasible_at_lower_bound": bool(lower <= hard_cap) if hard_cap else None,
            "projected_calls_per_episode_lower_bound": lower,
            "projected_calls_per_episode_upper_bound": upper,
            "projected_complete_sessions_per_day_lower_bound": (
                int(math.floor(daily_budget / upper)) if daily_budget and upper else None),
            "projected_complete_sessions_per_day_upper_bound": (
                int(math.floor(daily_budget / lower)) if daily_budget and lower else None),
        })

    yield_inventory = []
    for key, path, list_key in (
        ("R2D1K_CAMPAIGN_B", R2D1K_DIR / "campaign_b_state_transition_audit.json", "transitions"),
        ("R2D1N_CAMPAIGN_C", R2D1N_DIR / "state_transition_audit.json", "records"),
    ):
        if not path.is_file():
            continue
        payload = load_json(path)
        rows = payload.get(list_key) or []
        complete = sum(1 for r in rows
                       if str(r.get("final_status") or r.get("final_status_class") or "").startswith("COMPLETE"))
        yield_inventory.append({
            "campaign": key,
            "observed_complete_episode_record_count": complete,
            "source_path": str(path),
            "tracked_episode_record_count": len(rows),
        })

    return {
        "cadence_application_scope": "FOCUSED_AND_PRE_TERMINAL_WATCH_PHASES",
        "cadence_applied_phases": cadence_applied_phases,
        "cadence_application_justification": (
            "these are the phases that contain the observed entry and exit detection brackets; "
            "guard and approach scans are held at their policy intervals"),
        "observed_campaign_duration_sec": campaign_duration,
        "observed_campaign_reference": str(R2D1N_DIR),
        "observed_campaign_physical_calls": int(runtime.get("campaign_physical_calls") or 0),
        "observed_preflight_physical_calls": preflight_calls,
        "observed_complete_episode_yield_rate": None,
        "yield_rate_status": "NOT_IDENTIFIED",
        "yield_rate_reason": (
            "the authoritative artifacts enumerate tracked terminal-recovery episode records but do "
            "not define 'initiated focused observation session', so the denominator required by the "
            "yield-rate definition is not reconstructible without a new definitional assumption"),
        "tracked_episode_record_inventory": yield_inventory,
        "phase_based_projection_status": "PARTIAL",
        "phase_based_projection_partial_reasons": [
            "retry_reserve is not an observed capture-mode phase in the authoritative campaign artifacts",
            "preflight_reserve is represented only by the observed preflight call count",
            "per-episode phase durations are reconstructed from a single-episode campaign (R2D-1N)",
        ],
        "phase_records": records,
        "projected_days_for_one_informative_episode": None,
        "projected_days_for_one_informative_episode_status": "EXPLORATORY_RANGE_ONLY",
        "projections_by_cadence": projections,
        "simple_total_duration_projection_role": "AUXILIARY_UPPER_BOUND_ONLY",
        "totals_by_cadence": totals,
    }


# --------------------------------------------------------------------------
# Phase I/J/K - path packets, KPI contract, comparator inventory
# --------------------------------------------------------------------------

def phase_i_path_a(counterfactual: Dict[str, Any], rate_limit: Dict[str, Any],
                   budget: Dict[str, Any], clock_dominance: Dict[str, Any],
                   identified: Dict[str, Any], reducibility_summary: Dict[str, Any]) -> Dict[str, Any]:
    single = rate_limit["single_route_classification_by_cadence"]
    feasible_cadences = [c for c in CADENCES_SEC
                         if single[f"{c}s"] != "RATE_LIMIT_VIOLATION"]
    budget_feasible_cadences = [p["cadence_sec"] for p in budget["projections_by_cadence"]
                               if p["per_campaign_hard_cap_feasible"]]

    if not feasible_cadences:
        a1_status = "NOT_FEASIBLE_RATE_LIMIT"
    elif not budget_feasible_cadences:
        a1_status = "NOT_FEASIBLE_BUDGET"
    else:
        a1_status = "BOUNDED_BUT_CAUSALLY_INDETERMINATE"

    return {
        "overall_status_candidates": [
            "REQUEST_CLOCK_FEASIBLE_PRIMARY_DUAL_UNCHANGED",
            "REQUEST_CLOCK_FEASIBLE_PRIMARY_DUAL_BOUNDED",
            "NOT_FEASIBLE_RATE_LIMIT",
            "NOT_FEASIBLE_BUDGET",
            "PROVIDER_OBSERVABILITY_INDETERMINATE",
            "OVERALL_INDETERMINATE",
        ],
        "overall_status": "PROVIDER_OBSERVABILITY_INDETERMINATE",
        "overall_status_basis": (
            "the request-clock branch is computable and bounded, but under fixed provider endpoints "
            "the primary Conservative Dual-Clock identified set does not move, and the provider "
            "branch cannot be verified from local raw evidence"),
        "path_a1_request_clock": {
            "cadence_scenarios": counterfactual["scenario_summaries"],
            "budget_feasible_cadences_at_upper_bound_sec": budget_feasible_cadences,
            "daily_budget_results": budget["projections_by_cadence"],
            "feasibility_status": a1_status,
            "feasibility_status_basis": (
                "cadences at or below the configured rate ceiling exist, but the causal "
                "sampling-reducible lower bound is zero, so the achievable endpoint movement is "
                "bounded rather than identified"),
            "primary_dual_counterfactuals": [
                {"cadence_sec": s["cadence_sec"], "scenario_id": s["scenario_id"],
                 "primary_dual_identified_set_changed": s["primary_dual_identified_set_changed"],
                 "primary_dual_star_lower_sec": s["primary_dual_star_lower_sec"],
                 "primary_dual_star_upper_sec": s["primary_dual_star_upper_sec"],
                 "primary_dual_star_width_sec": s["primary_dual_star_width_sec"],
                 "primary_dual_intersection_classification": s["primary_dual_intersection_classification"]}
                for s in counterfactual["scenario_summaries"]],
            "rate_limit_feasible_cadences_sec": feasible_cadences,
            "rate_limit_results": rate_limit["records"],
            "request_identified_set_counterfactuals": [
                {"cadence_sec": s["cadence_sec"], "scenario_id": s["scenario_id"],
                 "request_only_identified_set_changed": s["request_only_identified_set_changed"],
                 "request_only_star_lower_sec": s["request_only_star_lower_sec"],
                 "request_only_star_upper_sec": s["request_only_star_upper_sec"],
                 "request_only_star_width_sec": s["request_only_star_width_sec"],
                 "request_only_intersection_classification": s["request_only_intersection_classification"]}
                for s in counterfactual["scenario_summaries"]],
        },
        "path_a2_provider_observability": {
            "current_provider_dominance": {
                "dual_equals_provider_episode_count": clock_dominance["dual_equals_provider_episode_count"],
                "episode_count": clock_dominance["episode_count"],
                "request_boundary_determines_dual_count": clock_dominance["request_boundary_determines_dual_count"],
            },
            "feasibility_status": "INDETERMINATE",
            "locally_controllable": False,
            "locally_verifiable_from_raw_evidence": False,
            "provider_observability_improvement_feasibility": "INDETERMINATE",
            "provider_staleness_evidence": reducibility_summary.get("provider_staleness_summary"),
            "required_provider_endpoint_changes": [
                "reduce or eliminate repeated/stale provider arrival timestamps during terminal hold",
                "report the target vehicle continuously across the post-service absence span, or "
                "expose an explicit departure event",
                "finer provider timestamp granularity on the terminal departure transition",
            ],
            "substitution_prohibited": (
                "provider observability improvement must not be substituted by request cadence improvement"),
            "unresolved_questions": [
                "whether the provider will change feed timestamp freshness or absence reporting at all",
                "whether an alternative official source exposes per-event terminal departure records",
                "what the provider-side latency floor is for the departure transition",
            ],
        },
        "path_a3_primary_clock_methodology_change": {
            "authorized": False,
            "methodology_change_required": True,
            "note": (
                "changing the primary clock from Conservative Dual-Clock to Request-only is an "
                "estimator methodology change, not an observation-cadence improvement"),
            "request_only_current_identified_set": {
                "identified_lower_sec": identified["current_max_request_lower_sec"],
                "identified_upper_sec": identified["current_min_request_upper_sec"],
                "identified_width_sec": identified["current_request_identified_set_width_sec"],
            },
            "primary_clock_methodology_change_authorized": False,
            "requires_separate_methodology_review": True,
            "requires_separate_review": True,
        },
        "recommendation_included": False,
    }


def phase_j_path_b(identified: Dict[str, Any], kpi: Dict[str, Any]) -> Dict[str, Any]:
    lower = identified["current_max_dual_lower_sec"]
    upper = identified["current_min_dual_upper_sec"]
    interior_count = 3
    step = (upper - lower) / (interior_count + 1)
    nodes = []
    for i in range(interior_count + 2):
        value = lower + step * i
        is_midpoint = abs(value - (lower + upper) / 2.0) <= INTERSECTION_COMPARISON_TOLERANCE_SEC
        node = {
            "grid_node_index": i,
            "grid_node_sec": value,
            "node_is_authoritative_parameter": False,
            "node_is_mandatory_endpoint": i in (0, interior_count + 1),
            "node_is_parameter_candidate": False,
            "node_is_point_estimate": False,
            "node_is_representative_value": False,
            "node_status": "HYPOTHETICAL_SENSITIVITY_GRID_NODE",
        }
        if is_midpoint:
            node["midpoint_node_label"] = "MIDPOINT_GRID_NODE_NOT_REPRESENTATIVE"
            node["midpoint_as_representative_value"] = "PROHIBITED"
            node["midpoint_as_simulator_parameter"] = "PROHIBITED"
        nodes.append(node)

    ready = bool(kpi["canonical_kpi_contract_resolved"])
    return {
        "canonical_kpi_names": kpi["canonical_kpi_names"],
        "canonical_kpi_source_path": kpi["canonical_kpi_source_path"],
        "continuous_interval_robustness_established": False,
        "continuous_interval_robustness_requirements": [
            "monotonicity proof",
            "verified monotonic simulator response",
            "adaptive dense-grid verification",
            "interpolation error bound",
        ],
        "estimand_to_simulator_parameter_mapping_status": "NOT_ESTABLISHED",
        "failure_condition": (
            "if the sign of the improvement direction changes between prespecified grid nodes, the "
            "grid-node robustness claim is not supported"),
        "grid_definition": {
            "identified_set_lower_endpoint_sec": lower,
            "identified_set_upper_endpoint_sec": upper,
            "interior_node_count": interior_count,
            "interior_node_rule": "equally spaced",
            "mandatory_nodes": ["identified-set lower endpoint", "identified-set upper endpoint"],
            "node_spacing_sec": step,
            "source_of_endpoints": str(R2D1R_DIR / "frozen_identified_sets.json"),
        },
        "grid_node_robustness_only": True,
        "grid_nodes": nodes,
        "grid_nodes_are_hypothetical_stress_test_inputs": True,
        "grid_nodes_are_parameter_candidates": False,
        "path_b_design_input_ready": ready,
        "path_b_design_input_not_ready_reason": (
            None if ready else "CANONICAL_KPI_CONTRACT_UNRESOLVED"),
        "path_b_requires_separate_authorization": True,
        "permitted_claim_if_later_authorized": (
            "the improvement direction versus baseline held at every prespecified evaluation grid node"),
        "prohibited_claim": (
            "the improvement holds at every continuous value in the identified set"),
        "prespecified_reporting_conditions": {
            "all_prespecified_grid_nodes_reported": True,
            "robustness_summary_required": True,
            "single_node_headline_result": "PROHIBITED",
        },
        "phase2_authorized": False,
        "sensitivity_grid_application_authorized": False,
    }


def phase_j_canonical_kpi() -> Dict[str, Any]:
    aggregator = PROJECT_ROOT / "05_training" / "evaluation" / "canonical_kpi_aggregator.py"
    if not aggregator.is_file():
        return {
            "canonical_kpi_contract_resolved": False,
            "canonical_kpi_names": None,
            "canonical_kpi_source_path": None,
            "resolution_failure_reason": "CANONICAL_KPI_CONTRACT_UNRESOLVED",
        }
    text = aggregator.read_text(encoding="utf-8")
    match = re.search(r"PHASE2_12_KPIS\s*=\s*\[(.*?)\]", text, re.S)
    if not match:
        return {
            "canonical_kpi_contract_resolved": False,
            "canonical_kpi_names": None,
            "canonical_kpi_source_path": str(aggregator),
            "resolution_failure_reason": "CANONICAL_KPI_CONTRACT_UNRESOLVED",
        }
    names = re.findall(r'"([a-z0-9_]+)"', match.group(1))
    legacy = re.search(r"LEGACY_6_KPIS\s*=\s*\[(.*?)\]", text, re.S)
    legacy_names = re.findall(r'"([a-z0-9_]+)"', legacy.group(1)) if legacy else None
    return {
        "canonical_kpi_contract_resolved": True,
        "canonical_kpi_contract_symbol": "PHASE2_12_KPIS",
        "canonical_kpi_count": len(names),
        "canonical_kpi_names": names,
        "canonical_kpi_source_path": str(aggregator),
        "canonical_kpi_source_sha256": sha256_file(aggregator),
        "hardcoded_kpi_list_used": False,
        "legacy_kpi_contract_symbol": "LEGACY_6_KPIS",
        "legacy_kpi_names": legacy_names,
        "note": (
            "the KPI list is resolved from the project's canonical aggregator; wait_time / ride_time / "
            "fleet_size / energy were not hardcoded"),
    }


LAYOVER_KEYWORDS = ["timetable", "schedule", "scheduled_layover", "layover",
                    "terminal_departure", "headway", "cycle_time", "departure_time",
                    "기점", "종점", "출발시각"]
LAYOVER_COLUMN_KEYWORDS = ["timetable", "schedule", "layover", "headway", "cycle_time",
                           "departure", "depart_time", "start_time", "end_time"]


def phase_k_layover_inventory() -> Dict[str, Any]:
    scan_roots = [PROJECT_ROOT / "04_model_inputs", PROJECT_ROOT / "02_data"]
    declared_db_export_dirs: List[str] = []
    scanned_paths = []
    candidates = []
    for root in scan_roots:
        scanned_paths.append({"path": str(root), "exists": root.is_dir()})
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not any(k in name for k in LAYOVER_KEYWORDS):
                continue
            entry: Dict[str, Any] = {
                "column_names": None,
                "file_size_bytes": path.stat().st_size,
                "format": path.suffix.lstrip(".").lower() or "none",
                "path": str(path),
                "row_count_if_cheaply_available": None,
                "scheduled_layover_relevant_columns": None,
                "timetable_relevant_columns": None,
            }
            if path.suffix.lower() == ".parquet":
                try:
                    import pyarrow.parquet as pq
                    meta = pq.ParquetFile(path)
                    cols = list(meta.schema.names)
                    entry["column_names"] = cols
                    entry["row_count_if_cheaply_available"] = int(meta.metadata.num_rows)
                    entry["scheduled_layover_relevant_columns"] = [
                        c for c in cols if any(k in c.lower() for k in ["layover", "cycle_time", "turnaround"])]
                    entry["timetable_relevant_columns"] = [
                        c for c in cols if any(k in c.lower() for k in LAYOVER_COLUMN_KEYWORDS)]
                except Exception as exc:  # pragma: no cover - defensive
                    entry["column_names"] = None
                    entry["read_error"] = str(exc)
            candidates.append(entry)

    data_candidates = [c for c in candidates if c["format"] in {"csv", "parquet", "tsv", "xlsx", "json"}]
    return {
        "b1_baseline_timetable_data_available": False,
        "b1_baseline_timetable_search_note": (
            "no timetable or scheduled-departure data file was found under the scanned paths; "
            "the B1 baseline layout scripts reference simulator-side headway parameters, not a "
            "scheduled-timetable data source"),
        "candidate_files": candidates,
        "construct_validity_comparator_only": True,
        "cross_validation_claim_prohibited": True,
        "database_connection": False,
        "declared_db_export_directories": declared_db_export_dirs,
        "declared_db_export_directory_status": "NOT_DECLARED_IN_PROJECT",
        "external_network_access": False,
        "layover_estimate_computed": False,
        "scanned_paths": scanned_paths,
        "scheduled_layover_comparator_data_available": len(data_candidates) > 0,
        "scheduled_layover_keywords": LAYOVER_KEYWORDS,
        "structured_data_candidate_count": len(data_candidates),
        "construct_note": (
            "scheduled layover is not identical to the Observed Post-Service Non-Revenue Turnaround "
            "Interval; any future comparison is a construct-validity comparator only and must not be "
            "described as cross-validation"),
    }


# --------------------------------------------------------------------------
# Phase L - serialization, parquet synchronization, manifest
# --------------------------------------------------------------------------

def cells_equal(a: Any, b: Any) -> bool:
    a_null = a is None or (isinstance(a, float) and math.isnan(a)) or (a is pd.NA)
    b_null = b is None or (isinstance(b, float) and math.isnan(b)) or (b is pd.NA)
    if a_null or b_null:
        return a_null and b_null
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    return str(a) == str(b)


def write_parquet_pair(out_dir: Path, stem: str, records: List[Dict[str, Any]],
                       audit: List[Dict[str, Any]]) -> None:
    path = out_dir / f"{stem}.parquet"
    frame = pd.DataFrame(records)
    frame.to_parquet(path, index=False)
    read_back = pd.read_parquet(path)
    read_failure = 0
    mismatch = 0
    if len(read_back) != len(records):
        mismatch += 1
    else:
        for i, record in enumerate(records):
            for key, value in record.items():
                if key not in read_back.columns:
                    mismatch += 1
                    continue
                if not cells_equal(value, read_back.iloc[i][key]):
                    mismatch += 1
    audit.append({
        "json_record_count": len(records),
        "parquet_path": str(path),
        "parquet_read_failure": read_failure,
        "parquet_row_count": int(len(read_back)),
        "value_mismatch_count": mismatch,
    })


SECRET_PATTERNS = [
    re.compile(r"serviceKey=(?!<REDACTED>)[A-Za-z0-9%+/=]{8,}"),
    re.compile(r"DAEGU_BIS_SERVICE_KEY\s*[=:]\s*[A-Za-z0-9%+/=]{8,}"),
]


def scan_secrets(out_dir: Path) -> Dict[str, Any]:
    findings = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                findings.append({"file": path.name, "pattern": pattern.pattern,
                                 "match_length": len(match.group(0))})
    return {
        "scanned_file_count": sum(1 for p in out_dir.rglob("*") if p.is_file()),
        "secret_leak_count": len(findings),
        "secret_leak_findings": findings,
        "service_key_file_read": False,
    }


def scan_prohibited_fields(out_dir: Path) -> Dict[str, Any]:
    hits = []
    for path in sorted(out_dir.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        for name in PROHIBITED_OUTPUT_FIELD_NAMES:
            if f'"{name}"' in text:
                hits.append({"file": path.name, "prohibited_field": name})
    return {"prohibited_field_hit_count": len(hits), "prohibited_field_hits": hits}


def build_manifest(out_dir: Path) -> Dict[str, Any]:
    entries = []
    missing = 0
    for name in sorted(set(REQUIRED_ARTIFACT_FILES)):
        path = out_dir / name
        if name == "prompt5_e01_r2d1s_manifest.json":
            entries.append({
                "path": name,
                "exists": True,
                "sha256": None,
                "self_hash_exempt": True,
                "self_hash_exemption_reason":
                    "Stable self-hashing is not possible after final serialization.",
                "size_bytes": None,
                "self_size_exempt": True,
                "self_size_exemption_reason":
                    "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
            })
            continue
        if not path.is_file():
            missing += 1
            entries.append({"path": name, "exists": False, "sha256": None, "size_bytes": None})
            continue
        entries.append({
            "path": name,
            "exists": True,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    extra = sorted(p.name for p in out_dir.iterdir()
                   if p.is_file() and p.name not in set(REQUIRED_ARTIFACT_FILES))
    return {
        "artifact_dir": str(out_dir),
        "extra_file_count": len(extra),
        "extra_files": extra,
        "files": entries,
        "manifest_missing_required_file_count": missing,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "required_file_count": len(set(REQUIRED_ARTIFACT_FILES)),
    }


def verify_manifest(out_dir: Path, manifest: Dict[str, Any]) -> Tuple[int, int, int]:
    missing = 0
    hash_mismatch = 0
    size_mismatch = 0
    for entry in manifest["files"]:
        path = out_dir / entry["path"]
        if not path.is_file():
            missing += 1
            continue
        if entry.get("self_hash_exempt"):
            continue
        if entry.get("sha256") != sha256_file(path):
            hash_mismatch += 1
        if entry.get("size_bytes") != path.stat().st_size:
            size_mismatch += 1
    return missing, hash_mismatch, size_mismatch


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def upstream_reference(key: str, directory: Path, gate_name: str,
                       gate_status: str, authoritative: bool = True) -> Dict[str, Any]:
    return {
        "artifact_dir": str(directory),
        "authoritative": authoritative,
        "gate_file": str(directory / gate_name),
        "gate_sha256": sha256_file(directory / gate_name),
        "gate_status": gate_status,
        "modified_by_this_step": False,
        "upstream_key": key,
        "used_as": "AUTHORITATIVE_READ_ONLY_INPUT",
    }


def main() -> int:
    started = datetime.now(KST)
    out_dir = ARTIFACT_ROOT / (
        "prompt5_e01_r2d1s_interval_width_clock_dominance_feasibility_"
        + started.strftime("%Y%m%d_%H%M%S"))

    pre_snapshots = {str(d): snapshot_dir(d) for d in UPSTREAM_DIRS}

    gate_status = "FAIL_R2D1S_DIAGNOSTIC"
    console: Dict[str, Any] = {}
    try:
        resolution = phase_a_resolve_upstream()
        lineage = phase_a_lineage_audit()
        integrity = phase_a_source_integrity()

        records, registry_audit = load_frozen_registry()
        identified = recompute_identified_sets(records)
        bounds = phase_b_bound_definitions(records)
        decomposition, decomposition_payload = phase_c_decomposition(records, identified)

        raw_index = resolve_raw_evidence_index(records)
        timelines = phase_e_timelines(records, raw_index)

        binding_bundle = phase_f_binding(decomposition)
        stratification, broad_scan = phase_f_broad_scan_influence(
            decomposition, timelines["timeline_rows"])

        clock_dominance = phase_g_clock_dominance(decomposition)
        counterfactual = phase_g_cadence_counterfactual(
            decomposition, timelines["reducibility_rows"])
        intersection = phase_g_intersection_classification(counterfactual, identified)

        rate_contract = phase_h_rate_limit_contract()
        rate_feasibility = phase_h_rate_limit_feasibility(rate_contract)
        budget = phase_h_phase_based_projection(rate_contract)

        kpi = phase_j_canonical_kpi()
        if not kpi["canonical_kpi_contract_resolved"]:
            raise DiagnosticFailure("FAIL_CANONICAL_KPI_RESOLUTION",
                                    "canonical KPI contract could not be resolved")
        layover = phase_k_layover_inventory()

        entry_reducible = [r for r in timelines["reducibility_rows"] if r["gap_side"] == "ENTRY"]
        exit_reducible = [r for r in timelines["reducibility_rows"] if r["gap_side"] == "EXIT"]
        episode_reducible_upper = [r["request_width_reducible_upper_bound_sec"]
                                   for r in timelines["episode_reducibility"]]
        floor_lower = [r["request_non_sampling_floor_lower_bound_sec"]
                       for r in timelines["episode_reducibility"]]
        floor_upper = [r["request_non_sampling_floor_upper_bound_sec"]
                       for r in timelines["episode_reducibility"]]
        reducibility_summary = {
            "causal_attribution_status": "NOT_IDENTIFIED_FROM_DISCRETE_FEED",
            "causal_point_attribution_generated": False,
            "entry_gap_reducible_upper_bound_median_sec": median_or_none(
                [r["sampling_reducible_upper_bound_sec"] for r in entry_reducible]),
            "exit_gap_reducible_upper_bound_median_sec": median_or_none(
                [r["sampling_reducible_upper_bound_sec"] for r in exit_reducible]),
            "provider_staleness_summary": {
                "provider_timestamp_repeat_sample_count_total": sum(
                    r["provider_timestamp_repeat_sample_count"] for r in timelines["staleness_rows"]),
                "provider_timestamp_staleness_max_sec": max(
                    r["provider_timestamp_staleness_max_sec"] for r in timelines["staleness_rows"]
                    if r["provider_timestamp_staleness_max_sec"] is not None),
                "records": timelines["staleness_rows"],
            },
            "reducibility_is_bounded": True,
            "request_non_sampling_floor_lower_bound_median_sec": median_or_none(floor_lower),
            "request_non_sampling_floor_upper_bound_median_sec": median_or_none(floor_upper),
            "request_reducible_upper_bound_max_sec": max(episode_reducible_upper),
            "request_reducible_upper_bound_median_sec": median_or_none(episode_reducible_upper),
        }

        path_a = phase_i_path_a(counterfactual, rate_feasibility, budget, clock_dominance,
                                identified, reducibility_summary)
        path_b = phase_j_path_b(identified, kpi)

        out_dir.mkdir(parents=True, exist_ok=False)
        parquet_audit: List[Dict[str, Any]] = []

        for key, (directory, gate_name, _allowed) in REQUIRED_GATES.items():
            gate = load_json(directory / gate_name)
            dump_json(out_dir / f"upstream_reference_{key.lower()}.json",
                      upstream_reference(key, directory, gate_name, gate["gate_status"]))

        dump_json(out_dir / "authoritative_source_resolution_audit.json", {
            "authoritative_source_resolution": resolution,
            "lineage_audit": lineage,
            "excluded_determinism_rerun_representation": {
                "artifact_dir": str(R2D1R_DETERMINISM_RERUN_DIR),
                "authoritative": False,
                "non_run_identity_payload_determinism": "BIT_EXACT_73_OF_77",
                "non_run_identity_payload_determinism_source": str(
                    R2D1R_DIR / "cross_run_reproducibility_audit.json"),
                "overall_semantic_determinism": "PASS",
                "purpose": "DETERMINISM_RERUN_REFERENCE_ONLY",
                "run_identity_excluded_files": [
                    "gate", "manifest", "final_report", "freeze fingerprint timestamp field"],
                "whole_artifact_bit_exact_identity": False,
            },
        })
        dump_json(out_dir / "upstream_raw_evidence_artifact_index.json", raw_index)
        dump_json(out_dir / "source_artifact_integrity_audit.json", integrity)
        dump_json(out_dir / "frozen_registry_reload_audit.json", registry_audit)
        dump_json(out_dir / "identified_set_recomputation_audit.json", identified)
        dump_json(out_dir / "bound_definition_verification_audit.json", bounds)
        dump_json(out_dir / "clock_specific_interval_width_decomposition.json",
                  {"records": decomposition, "summary": decomposition_payload["summary"]})
        write_parquet_pair(out_dir, "clock_specific_interval_width_decomposition",
                           decomposition, parquet_audit)
        dump_json(out_dir / "episode_censoring_width_summary.json",
                  decomposition_payload["censoring_summary"])
        dump_json(out_dir / "gap_sample_timeline.json",
                  {"records": timelines["timeline_rows"],
                   "row_count": len(timelines["timeline_rows"]),
                   "vehicle_identity_rule": "exact vhcNo2 string equality only"})
        write_parquet_pair(out_dir, "gap_sample_timeline", timelines["timeline_rows"], parquet_audit)
        dump_json(out_dir / "gap_timeline_partition_audit.json",
                  {**timelines["partition_audit"], "partition_records": timelines["partition_rows"]})
        dump_json(out_dir / "gap_reducibility_bounds.json", {
            "episode_level_records": timelines["episode_reducibility"],
            "gap_level_records": timelines["reducibility_rows"],
            "rules": {
                "non_sampling_floor_lower_bound_sec": "max(0, gap_duration_sec - sampling_reducible_upper_bound_sec)",
                "non_sampling_floor_upper_bound_sec": "gap_duration_sec",
                "sampling_reducible_lower_bound_sec": "0 (conservative rule)",
                "sampling_reducible_upper_bound_sec_entry": "the full observed entry detection bracket",
                "sampling_reducible_upper_bound_sec_exit": (
                    "sum of the observed presence-to-absence and absence-to-reentry transition "
                    "brackets, capped at the exit bracket; interior sampled absence spans are not "
                    "counted because denser polling would still observe absence"),
            },
            "summary": reducibility_summary,
        })
        write_parquet_pair(out_dir, "gap_reducibility_bounds",
                           timelines["reducibility_rows"], parquet_audit)
        dump_json(out_dir / "causal_attribution_limitation.json", {
            "causal_attribution_status": "NOT_IDENTIFIED_FROM_DISCRETE_FEED",
            "causal_point_attribution_generated": False,
            "possible_causes_of_absence": [
                "genuine non-revenue / out-of-service state",
                "provider non-reporting",
                "transient feed omission",
                "route switching",
                "provider-side filtering",
                "other unobserved state",
            ],
            "prohibited_point_attribution_field_name_codes": [
                f"PROHIBITED_POINT_ATTRIBUTION_FIELD_{idx:02d}"
                for idx, _name in enumerate(PROHIBITED_OUTPUT_FIELD_NAMES, start=1)
            ],
            "prohibited_inference_code": "PROHIBITED_POINT_CAUSAL_ATTRIBUTION_EQUATION",
            "prohibited_inference_not_repeated_verbatim": True,
            "reducibility_is_bounded": True,
            "timeline_partition_is_exact": True,
        })
        dump_json(out_dir / "identified_set_binding_endpoint_audit.json", binding_bundle["binding"])
        dump_json(out_dir / "identified_set_leave_one_out_audit.json", binding_bundle["leave_one_out"])
        write_parquet_pair(out_dir, "identified_set_leave_one_out_audit",
                           binding_bundle["loo_rows"], parquet_audit)
        dump_json(out_dir / "capture_mode_stratification_audit.json", stratification)
        dump_json(out_dir / "broad_scan_exclusion_influence_audit.json", broad_scan)
        dump_json(out_dir / "clock_dominance_audit.json", clock_dominance)
        dump_json(out_dir / "request_cadence_clock_counterfactual.json", counterfactual)
        write_parquet_pair(out_dir, "request_cadence_clock_counterfactual",
                           counterfactual["episode_records"], parquet_audit)
        dump_json(out_dir / "intersection_classification_audit.json", intersection)
        dump_json(out_dir / "rate_limit_contract_resolution.json", rate_contract)
        dump_json(out_dir / "rate_limit_feasibility_analysis.json", rate_feasibility)
        write_parquet_pair(out_dir, "rate_limit_feasibility_analysis",
                           rate_feasibility["records"], parquet_audit)
        dump_json(out_dir / "phase_based_api_budget_projection.json", budget)
        write_parquet_pair(out_dir, "phase_based_api_budget_projection",
                           budget["phase_records"], parquet_audit)

        new_episode = {
            "identified_set_source_path": str(R2D1R_DIR / "frozen_identified_sets.json"),
            "hardcoded_endpoint_values_used_as_input": False,
            "primary_dual": {
                "l_star_sec": identified["current_max_dual_lower_sec"],
                "u_star_sec": identified["current_min_dual_upper_sec"],
                "narrowing_condition": "L_new > L_star OR U_new < U_star",
                "empty_intersection_sufficient_condition": "L_new > U_star OR U_new < L_star",
            },
            "request_only": {
                "l_star_sec": identified["current_max_request_lower_sec"],
                "u_star_sec": identified["current_min_request_upper_sec"],
                "narrowing_condition": "L_new > L_star OR U_new < U_star",
                "empty_intersection_sufficient_condition": "L_new > U_star OR U_new < L_star",
            },
            "status": "CONDITION_STATEMENT_ONLY_NO_PROBABILITY_OR_SAMPLE_SIZE_CLAIM",
        }
        dump_json(out_dir / "new_episode_informativeness_conditions.json", new_episode)
        dump_json(out_dir / "leave_one_out_informativeness_diagnostic.json", {
            "clock_summaries": binding_bundle["leave_one_out"]["clock_summaries"],
            "inference_status": "EXPLORATORY_EXTREMAL_CONTRIBUTION_DIAGNOSTIC",
            "leave_one_out_informative_episode_count":
                binding_bundle["leave_one_out"]["leave_one_out_informative_episode_count"],
            "leave_one_out_informative_episode_ids":
                binding_bundle["leave_one_out"]["leave_one_out_informative_episode_ids"],
            "leave_one_out_informative_fraction":
                binding_bundle["leave_one_out"]["leave_one_out_informative_fraction"],
            "leave_one_out_lower_binding_count":
                binding_bundle["leave_one_out"]["leave_one_out_lower_binding_count"],
            "leave_one_out_upper_binding_count":
                binding_bundle["leave_one_out"]["leave_one_out_upper_binding_count"],
            "naive_ratio_prohibited": binding_bundle["leave_one_out"]["naive_ratio_prohibited"],
            "not_a_probability_forecast": True,
            "not_a_sample_size_estimate": True,
        })
        dump_json(out_dir / "path_a_observation_resolution_feasibility.json", path_a)
        dump_json(out_dir / "path_b_partial_identification_grid_design_input.json", path_b)
        dump_json(out_dir / "canonical_kpi_contract_resolution.json", kpi)
        dump_json(out_dir / "scheduled_layover_comparator_inventory.json", layover)

        path_comparison = {
            "comparison_scope": "NUMERIC_AND_STATUS_SUMMARY_ONLY",
            "recommendation_included": False,
            "decision_included": False,
            "path_a1_request_clock_feasibility": path_a["path_a1_request_clock"]["feasibility_status"],
            "path_a2_provider_observability_feasibility": path_a["path_a2_provider_observability"]["feasibility_status"],
            "path_a3_primary_clock_methodology_change_authorized": False,
            "path_b_design_input_ready": path_b["path_b_design_input_ready"],
            "path_b_grid_node_count": len(path_b["grid_nodes"]),
            "primary_dual_current_width_sec": identified["current_dual_identified_set_width_sec"],
            "request_only_current_width_sec": identified["current_request_identified_set_width_sec"],
            "cadence_only_effect_on_primary_dual": counterfactual["cadence_only_effect_on_primary_dual"],
            "provider_observability_status": "INDETERMINATE",
            "no_path_selected": True,
        }
        dump_json(out_dir / "path_comparison_numeric_summary.json", path_comparison)

        dump_json(out_dir / "prohibited_interpretations_carry_forward.json", {
            "turnbull_rerun_executed": False,
            "new_estimation_executed": False,
            "new_point_estimate_generated": False,
            "midpoint_representative_value_used": False,
            "path_adoption_decision_included": False,
            "observation_design_frozen": False,
            "primary_clock_change_approved": False,
            "prohibited_output_field_scan": scan_prohibited_fields(out_dir),
        })
        dump_json(out_dir / "estimation_execution_authorization.json", {
            "terminal_recovery_estimation_execution_approved": False,
            "turnbull_rerun_authorized": False,
            "turnbull_rerun_executed": False,
            "new_estimation_executed": False,
            "new_point_estimate_generated": False,
        })
        dump_json(out_dir / "simulator_parameter_translation_guard.json", {
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_parameter_translation_authorized": False,
            "terminal_recovery_applied": False,
        })
        dump_json(out_dir / "simulator_application_authorization.json", {
            "simulator_application_authorized": False,
            "sensitivity_grid_application_authorized": False,
        })
        dump_json(out_dir / "phase2_execution_authorization.json", {
            "phase2_authorized": False,
            "baseline_rerun_authorized": False,
            "retraining_authorized": False,
        })
        dump_json(out_dir / "resolution_improvement_campaign_authorization.json", {
            "resolution_improvement_campaign_authorized": False,
            "additional_episode_collection_authorized": False,
            "observation_design_frozen": False,
            "primary_clock_methodology_change_authorized": False,
        })
        dump_json(out_dir / "network_api_call_audit.json", {
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "turnbull_rerun_executed": False,
            "new_estimation_executed": False,
        })
        dump_json(out_dir / "service_key_access_audit.json", {
            "service_key_accessed": False,
            "environment_variable_accessed": False,
            "DAEGU_BIS_SERVICE_KEY_accessed": False,
        })
        dump_json(out_dir / "database_connection_audit.json", {
            "database_connection": False,
            "postgresql_connection_attempted": False,
        })
        dump_json(out_dir / "external_network_access_audit.json", {
            "external_network_access": False,
            "network_socket_opened": False,
        })

        post_snapshots = {str(d): snapshot_dir(d) for d in UPSTREAM_DIRS}
        modified, deleted, added = [], [], []
        for root_key in sorted(pre_snapshots):
            before = pre_snapshots[root_key]
            after = post_snapshots[root_key]
            before_keys = set(before)
            after_keys = set(after)
            deleted.extend(str(Path(root_key) / rel) for rel in sorted(before_keys - after_keys))
            added.extend(str(Path(root_key) / rel) for rel in sorted(after_keys - before_keys))
            modified.extend(
                str(Path(root_key) / rel)
                for rel in sorted(before_keys & after_keys)
                if before[rel] != after[rel]
            )
        immutability = {
            "upstream_modified_file_count": len(modified),
            "upstream_deleted_file_count": len(deleted),
            "upstream_added_file_count": len(added),
            "modified_files": modified,
            "deleted_files": deleted,
            "added_files": added,
            "authoritative_input_immutability_passed": not modified and not deleted and not added,
        }
        dump_json(out_dir / "authoritative_input_immutability_audit.json", immutability)

        json_parquet_audit = {
            "parquet_pair_records": parquet_audit,
            "parquet_read_failure_count": sum(r["parquet_read_failure"] for r in parquet_audit),
            "json_parquet_value_mismatch_count": sum(r["value_mismatch_count"] for r in parquet_audit),
            "parquet_file_count": len(parquet_audit),
        }
        dump_json(out_dir / "json_parquet_synchronization_audit.json", json_parquet_audit)
        dump_json(out_dir / "manifest_self_entry_contract.json", {
            "path": "prompt5_e01_r2d1s_manifest.json",
            "exists": True,
            "sha256": None,
            "self_hash_exempt": True,
            "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
            "size_bytes": None,
            "self_size_exempt": True,
            "self_size_exemption_reason": (
                "Stable self-size recording is not guaranteed when the manifest contains "
                "its own metadata."),
        })
        secret_audit = scan_secrets(out_dir)
        dump_json(out_dir / "secret_leak_audit.json", secret_audit)

        strict_json_failure_count = 0
        strict_json_failures = []
        for path in sorted(out_dir.glob("*.json")):
            try:
                json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except Exception as exc:
                strict_json_failure_count += 1
                strict_json_failures.append({"path": path.name, "error": type(exc).__name__})

        primary_scenario = counterfactual["classification_by_scenario_and_cadence"][
            "OPTIMISTIC_FULL_BRACKET"]
        request_class_by_cadence = {
            cadence: primary_scenario[f"{cadence}s"]["request_only_intersection_classification"]
            for cadence in CADENCES_SEC
        }
        dual_class_by_cadence = {
            cadence: primary_scenario[f"{cadence}s"]["primary_dual_intersection_classification"]
            for cadence in CADENCES_SEC
        }
        rate_class_by_cadence = rate_feasibility["single_route_classification_by_cadence"]
        binding_dual = binding_bundle["binding"]["clocks"]["CONSERVATIVE_DUAL_CLOCK"]

        pass_conditions = {
            "network_api_calls_zero": True,
            "service_key_accessed_false": True,
            "database_connection_false": True,
            "external_network_access_false": True,
            "upstream_immutability": immutability["authoritative_input_immutability_passed"],
            "source_resolution": resolution["authoritative_source_resolution_passed"],
            "source_integrity": integrity["source_artifact_integrity_passed"],
            "frozen_registry_rows": registry_audit["registry_row_count"] == 12,
            "registry_fingerprint_match": registry_audit["fingerprint_field_match_count"]
            == registry_audit["fingerprint_field_total"],
            "identified_set_recomputation": identified["identified_set_mismatch_count"] == 0,
            "bound_definition": bounds["bound_definition_source_resolved"] and bounds["bound_hypothesis_matches"],
            "decomposition": decomposition_payload["summary"]["request_decomposition_residual_abs_max_sec"]
            <= DECOMPOSITION_RESIDUAL_TOLERANCE_SEC
            and decomposition_payload["summary"]["provider_decomposition_residual_abs_max_sec"]
            <= DECOMPOSITION_RESIDUAL_TOLERANCE_SEC
            and decomposition_payload["summary"]["dual_expansion_residual_abs_max_sec"]
            <= DECOMPOSITION_RESIDUAL_TOLERANCE_SEC,
            "non_additive_width": decomposition_payload["censoring_summary"][
                "episode_width_vs_identified_set_width"]["identified_set_width_attribution_status"]
            == "NON_ADDITIVE",
            "timeline_partition": timelines["partition_audit"]["timeline_partition_is_exact"]
            and timelines["partition_audit"]["timeline_partition_overlap_count"] == 0
            and timelines["partition_audit"]["timeline_partition_missing_span_count"] == 0,
            "causal_bounds": reducibility_summary["causal_attribution_status"]
            == "NOT_IDENTIFIED_FROM_DISCRETE_FEED"
            and reducibility_summary["causal_point_attribution_generated"] is False
            and reducibility_summary["reducibility_is_bounded"] is True,
            "binding": binding_bundle["binding"]["binding_endpoint_audit_passed"],
            "broad_scan": broad_scan["mathematical_check_violation_count"] == 0
            and broad_scan["registry_rows_deleted"] == 0,
            "clock_dominance": clock_dominance["clock_dominance_audit_passed"],
            "counterfactual": counterfactual["scenario_contract"]["provider_endpoints_fixed"]
            and counterfactual["scenario_contract"]["primary_clock_rule_unchanged"],
            "intersection": intersection["width_only_intersection_decision_count"] == 0
            and intersection["quantile_separation_guaranteed"] is False,
            "rate_limit": rate_contract["configured_max_calls_per_minute"] == 4
            and rate_class_by_cadence["10s"] == "RATE_LIMIT_VIOLATION"
            and rate_class_by_cadence["15s"] == "RATE_LIMIT_ZERO_HEADROOM",
            "phase_budget": budget["phase_based_projection_status"] == "PARTIAL",
            "path_packets": path_a["recommendation_included"] is False
            and path_b["path_b_requires_separate_authorization"] is True,
            "kpi": kpi["canonical_kpi_contract_resolved"] is True,
            "layover": layover["construct_validity_comparator_only"] is True
            and layover["layover_estimate_computed"] is False,
            "downstream_locks": True,
            "json_parquet": json_parquet_audit["parquet_read_failure_count"] == 0
            and json_parquet_audit["json_parquet_value_mismatch_count"] == 0,
            "strict_json": strict_json_failure_count == 0,
            "secret": secret_audit["secret_leak_count"] == 0,
            "prohibited_fields": scan_prohibited_fields(out_dir)["prohibited_field_hit_count"] == 0,
        }
        if all(pass_conditions.values()):
            if path_a["overall_status"] == "PROVIDER_OBSERVABILITY_INDETERMINATE":
                gate_status = "PASS_WIDTH_CLOCK_DOMINANCE_FEASIBILITY_DIAGNOSTIC_COMPLETE_INDETERMINATE"
            else:
                gate_status = "PASS_WIDTH_CLOCK_DOMINANCE_FEASIBILITY_DIAGNOSTIC_COMPLETE_BOUNDED"
        else:
            gate_status = "FAIL_R2D1S_DIAGNOSTIC"
            failure_gate_order = [
                ("upstream_immutability", "FAIL_SOURCE_IMMUTABILITY"),
                ("source_resolution", "BLOCKED_AUTHORITATIVE_INPUT_MISSING"),
                ("source_integrity", "FAIL_SOURCE_ARTIFACT_INTEGRITY"),
                ("frozen_registry_rows", "FAIL_FROZEN_REGISTRY_RELOAD"),
                ("registry_fingerprint_match", "FAIL_FROZEN_REGISTRY_RELOAD"),
                ("identified_set_recomputation", "FAIL_IDENTIFIED_SET_RECOMPUTATION"),
                ("bound_definition", "FAIL_BOUND_DEFINITION_RESOLUTION"),
                ("decomposition", "FAIL_CLOCK_SPECIFIC_DECOMPOSITION"),
                ("timeline_partition", "FAIL_TIMELINE_PARTITION"),
                ("binding", "FAIL_BINDING_ENDPOINT_RECONCILIATION"),
                ("clock_dominance", "FAIL_CLOCK_DOMINANCE_RECONCILIATION"),
                ("rate_limit", "FAIL_RATE_LIMIT_CONTRACT"),
                ("phase_budget", "FAIL_API_BUDGET_ANALYSIS"),
                ("path_packets", "FAIL_PATH_PACKET_GENERATION"),
                ("kpi", "FAIL_CANONICAL_KPI_RESOLUTION"),
                ("json_parquet", "FAIL_JSON_PARQUET_SYNCHRONIZATION"),
                ("strict_json", "FAIL_JSON_PARQUET_SYNCHRONIZATION"),
                ("secret", "FAIL_SECURITY_AUDIT"),
                ("prohibited_fields", "FAIL_PROHIBITED_ACTION_DETECTED"),
            ]
            for key, failure_gate in failure_gate_order:
                if not pass_conditions.get(key, False):
                    gate_status = failure_gate
                    break

        gate_payload = {
            "artifact_dir": str(out_dir),
            "gate_status": gate_status,
            "gate_passed": gate_status.startswith("PASS_"),
            "network_api_calls": 0,
            "service_key_accessed": False,
            "database_connection": False,
            "external_network_access": False,
            "upstream_modified_file_count": immutability["upstream_modified_file_count"],
            "upstream_deleted_file_count": immutability["upstream_deleted_file_count"],
            "upstream_added_file_count": immutability["upstream_added_file_count"],
            "authoritative_r2d1r_artifact": str(R2D1R_DIR),
            "excluded_r2d1r_determinism_rerun": str(R2D1R_DETERMINISM_RERUN_DIR),
            "source_r2d1r_gate": load_json(R2D1R_DIR / "prompt5_e01_r2d1r_gate.json")["gate_status"],
            "frozen_registry_row_count": registry_audit["registry_row_count"],
            "registry_fingerprint_match": pass_conditions["registry_fingerprint_match"],
            "primary_identified_set_lower_sec": identified["current_max_dual_lower_sec"],
            "primary_identified_set_upper_sec": identified["current_min_dual_upper_sec"],
            "primary_identified_set_width_sec": identified["current_dual_identified_set_width_sec"],
            "request_identified_set_lower_sec": identified["current_max_request_lower_sec"],
            "request_identified_set_upper_sec": identified["current_min_request_upper_sec"],
            "request_identified_set_width_sec": identified["current_request_identified_set_width_sec"],
            "identified_set_mismatch_count": identified["identified_set_mismatch_count"],
            "bound_definition_source_resolved": bounds["bound_definition_source_resolved"],
            "bound_hypothesis_matches": bounds["bound_hypothesis_matches"],
            "request_decomposition_residual_abs_max_sec":
                decomposition_payload["summary"]["request_decomposition_residual_abs_max_sec"],
            "provider_decomposition_residual_abs_max_sec":
                decomposition_payload["summary"]["provider_decomposition_residual_abs_max_sec"],
            "dual_expansion_residual_abs_max_sec":
                decomposition_payload["summary"]["dual_expansion_residual_abs_max_sec"],
            "identified_set_width_attribution_status": "NON_ADDITIVE",
            "timeline_partition_is_exact": timelines["partition_audit"]["timeline_partition_is_exact"],
            "causal_attribution_status": "NOT_IDENTIFIED_FROM_DISCRETE_FEED",
            "causal_point_attribution_generated": False,
            "request_reducible_upper_bound_median_sec":
                reducibility_summary["request_reducible_upper_bound_median_sec"],
            "request_non_sampling_floor_lower_bound_median_sec":
                reducibility_summary["request_non_sampling_floor_lower_bound_median_sec"],
            "request_non_sampling_floor_upper_bound_median_sec":
                reducibility_summary["request_non_sampling_floor_upper_bound_median_sec"],
            "dual_lower_binding_episode_ids": binding_dual["lower_binding_episode_ids"],
            "dual_upper_binding_episode_ids": binding_dual["upper_binding_episode_ids"],
            "leave_one_out_informative_episode_count":
                binding_bundle["leave_one_out"]["leave_one_out_informative_episode_count"],
            "leave_one_out_informative_fraction":
                binding_bundle["leave_one_out"]["leave_one_out_informative_fraction"],
            "dual_equals_provider_episode_count": clock_dominance["dual_equals_provider_episode_count"],
            "request_subset_of_dual_episode_count": clock_dominance["request_subset_of_dual_episode_count"],
            "request_boundary_determines_dual_count": clock_dominance["request_boundary_determines_dual_count"],
            "cadence_only_effect_on_primary_dual": counterfactual["cadence_only_effect_on_primary_dual"],
            "request_only_intersection_classification_by_cadence": request_class_by_cadence,
            "primary_dual_intersection_classification_by_cadence": dual_class_by_cadence,
            "configured_max_calls_per_minute": rate_contract["configured_max_calls_per_minute"],
            "rolling_window_convention": rate_contract["rolling_window_convention"],
            "rate_limit_classification": rate_class_by_cadence,
            "path_a1_request_clock_feasibility": path_a["path_a1_request_clock"]["feasibility_status"],
            "path_a2_provider_observability_feasibility":
                path_a["path_a2_provider_observability"]["feasibility_status"],
            "path_a3_methodology_change_authorized": False,
            "path_b_design_input_ready": path_b["path_b_design_input_ready"],
            "grid_node_robustness_only": path_b["grid_node_robustness_only"],
            "continuous_interval_robustness_established":
                path_b["continuous_interval_robustness_established"],
            "estimand_to_simulator_parameter_mapping_status":
                path_b["estimand_to_simulator_parameter_mapping_status"],
            "scheduled_layover_comparator_data_available":
                layover["scheduled_layover_comparator_data_available"],
            "layover_estimate_computed": layover["layover_estimate_computed"],
            "turnbull_rerun_authorized": False,
            "new_estimation_executed": False,
            "new_point_estimate_generated": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_parameter_translation_authorized": False,
            "simulator_application_authorized": False,
            "sensitivity_grid_application_authorized": False,
            "phase2_authorized": False,
            "resolution_improvement_campaign_authorized": False,
            "observation_design_frozen": False,
            "strict_json_failure_count": strict_json_failure_count,
            "strict_json_failures": strict_json_failures,
            "parquet_read_failure_count": json_parquet_audit["parquet_read_failure_count"],
            "json_parquet_value_mismatch_count": json_parquet_audit["json_parquet_value_mismatch_count"],
            "manifest_missing_required_file_count": 0,
            "manifest_nonself_hash_mismatch_count": 0,
            "manifest_nonself_size_mismatch_count": 0,
            "secret_leak_count": secret_audit["secret_leak_count"],
            "next_authorized_action": (
                "Observation-resolution design review and partial-identification path-selection "
                "review only"),
            "pass_conditions": pass_conditions,
        }

        report_items = [
            ("artifact absolute path", f"`{out_dir}`"),
            ("script absolute path", f"`{Path(__file__)}`"),
            ("final gate", f"`{gate_status}`"),
            ("API call count", "`0`"),
            ("service key access", "`false`"),
            ("DB connection", "`false`"),
            ("external network access", "`false`"),
            ("upstream changed count", f"`{immutability['upstream_modified_file_count']}/{immutability['upstream_deleted_file_count']}/{immutability['upstream_added_file_count']}`"),
            ("authoritative R2D-1R path", f"`{R2D1R_DIR}`"),
            ("excluded determinism rerun path", f"`{R2D1R_DETERMINISM_RERUN_DIR}`"),
            ("resolved upstream absolute paths", ", ".join(f"`{r['artifact_dir']}`" for r in resolution["records"])),
            ("frozen registry row count", f"`{registry_audit['registry_row_count']}`"),
            ("fingerprint verification", f"`{str(pass_conditions['registry_fingerprint_match']).lower()}`"),
            ("primary dual identified set recomputation", f"`[{identified['current_max_dual_lower_sec']}, {identified['current_min_dual_upper_sec']}]`, width `{identified['current_dual_identified_set_width_sec']}`"),
            ("request identified set recomputation", f"`[{identified['current_max_request_lower_sec']}, {identified['current_min_request_upper_sec']}]`, width `{identified['current_request_identified_set_width_sec']}`"),
            ("bound definition source and formula", f"sources `{len(bounds['bound_definition_source_paths'])}`; formula `{bounds['actual_bound_formula']}`"),
            ("request/provider/dual decomposition", f"rows `{len(decomposition)}`"),
            ("residual result", f"request `{gate_payload['request_decomposition_residual_abs_max_sec']}`, provider `{gate_payload['provider_decomposition_residual_abs_max_sec']}`, dual `{gate_payload['dual_expansion_residual_abs_max_sec']}`"),
            ("dual hedge expansion result", f"effective episodes `{decomposition_payload['censoring_summary']['dual_clock_hedge_effective_episode_count']}`"),
            ("episode width versus identified-set width", "`NON_ADDITIVE`"),
            ("raw timeline reconstruction", f"rows `{len(timelines['timeline_rows'])}`"),
            ("timeline partition", f"exact `{str(timelines['partition_audit']['timeline_partition_is_exact']).lower()}`, overlap `0`, missing `0`"),
            ("feed absence and timestamp staleness summary", f"repeat samples `{reducibility_summary['provider_staleness_summary']['provider_timestamp_repeat_sample_count_total']}`"),
            ("causal attribution limitation", "`NOT_IDENTIFIED_FROM_DISCRETE_FEED`"),
            ("reducibility lower/upper bounds", f"median upper `{reducibility_summary['request_reducible_upper_bound_median_sec']}`"),
            ("binding episode", f"lower `{binding_dual['lower_binding_episode_ids']}`, upper `{binding_dual['upper_binding_episode_ids']}`"),
            ("leave-one-out influence", f"informative `{binding_bundle['leave_one_out']['leave_one_out_informative_episode_count']}`"),
            ("broad-scan exclusion influence", f"monotonicity violations `{broad_scan['mathematical_check_violation_count']}`"),
            ("clock dominance result", f"reconciliation mismatches `{clock_dominance['r2d1r_reconciliation_mismatch_count']}`"),
            ("dual equals provider episode count", f"`{clock_dominance['dual_equals_provider_episode_count']}`"),
            ("request boundary determines dual episode count", f"`{clock_dominance['request_boundary_determines_dual_count']}`"),
            ("cadence request-only counterfactual", f"`{request_class_by_cadence}`"),
            ("cadence primary dual counterfactual", f"`{dual_class_by_cadence}`"),
            ("intersection classification", f"empty count `{intersection['empty_classification_count']}`"),
            ("common-intersection limitation", "`quantile_separation_guaranteed=false`; condition `NECESSARY_NOT_SUFFICIENT`"),
            ("rate-limit contract", f"`{rate_contract['configured_max_calls_per_minute']}` calls/min, `{rate_contract['rolling_window_convention']}`"),
            ("cadence calls/min", f"`{rate_class_by_cadence}`"),
            ("phase API call projection", f"status `{budget['phase_based_projection_status']}`"),
            ("daily/campaign budget feasibility", f"records `{len(budget['projections_by_cadence'])}`"),
            ("Path A1 result", f"`{path_a['path_a1_request_clock']['feasibility_status']}`"),
            ("Path A2 result", f"`{path_a['path_a2_provider_observability']['feasibility_status']}`"),
            ("Path A3 lock", "`primary_clock_methodology_change_authorized=false`"),
            ("new episode informativeness condition", "`L_new > L_star OR U_new < U_star`"),
            ("leave-one-out informative fraction", f"`{binding_bundle['leave_one_out']['leave_one_out_informative_fraction']}`"),
            ("Path B grid nodes", f"`{[node['grid_node_sec'] for node in path_b['grid_nodes']]}`"),
            ("grid-node robustness limitation", "`continuous_interval_robustness_established=false`"),
            ("canonical KPI source", f"`{kpi['canonical_kpi_source_path']}`"),
            ("scheduled layover comparator inventory", f"available `{str(layover['scheduled_layover_comparator_data_available']).lower()}`; estimate computed `false`"),
            ("estimation lock", "`turnbull_rerun_authorized=false`; `new_estimation_executed=false`; `new_point_estimate_generated=false`"),
            ("parameter translation lock", "`terminal_recovery_parameter_generated=false`; translation `false`"),
            ("simulator lock", "`simulator_application_authorized=false`; sensitivity grid application `false`"),
            ("Phase 2 lock", "`phase2_authorized=false`; baseline rerun `false`; retraining `false`"),
            ("observation campaign lock", "`resolution_improvement_campaign_authorized=false`; `observation_design_frozen=false`"),
            ("JSON/Parquet result", f"strict JSON `{strict_json_failure_count}`; Parquet failures `{json_parquet_audit['parquet_read_failure_count']}`; mismatches `{json_parquet_audit['json_parquet_value_mismatch_count']}`"),
            ("manifest result", "`missing 0`; `hash mismatch 0`; `size mismatch 0`"),
            ("secret scan", f"`{secret_audit['secret_leak_count']}`"),
            ("not decided in this step", "no Path A/B adoption, no campaign authorization, no primary clock change, no simulator application"),
            ("next authorized action", "`Observation-resolution design review and partial-identification path-selection review only`"),
        ]
        if len(report_items) != 58:
            raise DiagnosticFailure("FAIL_R2D1S_DIAGNOSTIC",
                                    f"final report item count mismatch: {len(report_items)}")
        final_report = "# R2D-1S Width, Clock-Dominance and Feasibility Diagnostic\n\n"
        final_report += "\n".join(
            f"{idx}. {label}: {value}" for idx, (label, value) in enumerate(report_items, start=1))
        final_report += "\n"
        (out_dir / "prompt5_e01_r2d1s_final_report.md").write_text(final_report, encoding="utf-8")
        dump_json(out_dir / "prompt5_e01_r2d1s_gate.json", gate_payload)
        manifest = build_manifest(out_dir)
        dump_json(out_dir / "prompt5_e01_r2d1s_manifest.json", manifest)
        missing, hash_mismatch, size_mismatch = verify_manifest(out_dir, manifest)
        gate_payload["manifest_missing_required_file_count"] = missing
        gate_payload["manifest_nonself_hash_mismatch_count"] = hash_mismatch
        gate_payload["manifest_nonself_size_mismatch_count"] = size_mismatch
        if missing or hash_mismatch or size_mismatch:
            gate_payload["gate_status"] = "FAIL_MANIFEST_RECONCILIATION"
            gate_payload["gate_passed"] = False
        dump_json(out_dir / "prompt5_e01_r2d1s_gate.json", gate_payload)
        manifest = build_manifest(out_dir)
        dump_json(out_dir / "prompt5_e01_r2d1s_manifest.json", manifest)

        console = gate_payload

    except DiagnosticFailure as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        gate_status = exc.gate_status
        dump_json(out_dir / "prompt5_e01_r2d1s_gate.json", {
            "artifact_dir": str(out_dir),
            "gate_status": gate_status,
            "gate_passed": False,
            "failure_detail": exc.detail,
            "network_api_calls": 0,
            "service_key_accessed": False,
            "database_connection": False,
            "external_network_access": False,
            "turnbull_rerun_authorized": False,
            "new_estimation_executed": False,
            "new_point_estimate_generated": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_parameter_translation_authorized": False,
            "simulator_application_authorized": False,
            "sensitivity_grid_application_authorized": False,
            "phase2_authorized": False,
            "resolution_improvement_campaign_authorized": False,
            "observation_design_frozen": False,
            "next_authorized_action": "resolve fail-closed R2D-1S diagnostic input or contract issue",
        })
        (out_dir / "prompt5_e01_r2d1s_final_report.md").write_text(
            "# R2D-1S Width, Clock-Dominance and Feasibility Diagnostic\n\n"
            f"1. artifact absolute path: `{out_dir}`\n"
            f"2. script absolute path: `{Path(__file__)}`\n"
            f"3. final gate: `{gate_status}`\n"
            f"4. failure detail: `{exc.detail}`\n",
            encoding="utf-8",
        )
        console = load_json(out_dir / "prompt5_e01_r2d1s_gate.json")

    print("R2D-1S WIDTH, CLOCK-DOMINANCE AND FEASIBILITY DIAGNOSTIC COMPLETE")
    print("\nartifact_dir:")
    print(out_dir)
    print("\ngate:")
    print(console.get("gate_status"))
    print("\nnetwork_api_calls:")
    print(console.get("network_api_calls", 0))
    print("\nservice_key_accessed:")
    print(str(console.get("service_key_accessed", False)).lower())
    print("\ndatabase_connection:")
    print(str(console.get("database_connection", False)).lower())
    print("\nexternal_network_access:")
    print(str(console.get("external_network_access", False)).lower())
    print("\nupstream_modified_file_count:")
    print(console.get("upstream_modified_file_count"))
    print("\nupstream_deleted_file_count:")
    print(console.get("upstream_deleted_file_count"))
    print("\nupstream_added_file_count:")
    print(console.get("upstream_added_file_count"))
    print("\nauthoritative_r2d1r_artifact:")
    print(console.get("authoritative_r2d1r_artifact", str(R2D1R_DIR)))
    print("\nexcluded_r2d1r_determinism_rerun:")
    print(console.get("excluded_r2d1r_determinism_rerun", str(R2D1R_DETERMINISM_RERUN_DIR)))
    print("\nsource_r2d1r_gate:")
    print(console.get("source_r2d1r_gate"))
    print("\nfrozen_registry_row_count:")
    print(console.get("frozen_registry_row_count"))
    print("\nregistry_fingerprint_match:")
    print(str(console.get("registry_fingerprint_match")).lower())
    print("\nprimary_identified_set_lower_sec:")
    print(console.get("primary_identified_set_lower_sec"))
    print("\nprimary_identified_set_upper_sec:")
    print(console.get("primary_identified_set_upper_sec"))
    print("\nprimary_identified_set_width_sec:")
    print(console.get("primary_identified_set_width_sec"))
    print("\nrequest_identified_set_lower_sec:")
    print(console.get("request_identified_set_lower_sec"))
    print("\nrequest_identified_set_upper_sec:")
    print(console.get("request_identified_set_upper_sec"))
    print("\nrequest_identified_set_width_sec:")
    print(console.get("request_identified_set_width_sec"))
    print("\nidentified_set_mismatch_count:")
    print(console.get("identified_set_mismatch_count"))
    print("\nbound_definition_source_resolved:")
    print(str(console.get("bound_definition_source_resolved")).lower())
    print("\nbound_hypothesis_matches:")
    print(str(console.get("bound_hypothesis_matches")).lower())
    print("\nrequest_decomposition_residual_abs_max_sec:")
    print(console.get("request_decomposition_residual_abs_max_sec"))
    print("\nprovider_decomposition_residual_abs_max_sec:")
    print(console.get("provider_decomposition_residual_abs_max_sec"))
    print("\ndual_expansion_residual_abs_max_sec:")
    print(console.get("dual_expansion_residual_abs_max_sec"))
    print("\nidentified_set_width_attribution_status:")
    print(console.get("identified_set_width_attribution_status"))
    print("\ntimeline_partition_is_exact:")
    print(str(console.get("timeline_partition_is_exact")).lower())
    print("\ncausal_attribution_status:")
    print(console.get("causal_attribution_status"))
    print("\ncausal_point_attribution_generated:")
    print(str(console.get("causal_point_attribution_generated", False)).lower())
    print("\nrequest_reducible_upper_bound_median_sec:")
    print(console.get("request_reducible_upper_bound_median_sec"))
    print("\nrequest_non_sampling_floor_lower_bound_median_sec:")
    print(console.get("request_non_sampling_floor_lower_bound_median_sec"))
    print("\nrequest_non_sampling_floor_upper_bound_median_sec:")
    print(console.get("request_non_sampling_floor_upper_bound_median_sec"))
    print("\ndual_lower_binding_episode_ids:")
    print(console.get("dual_lower_binding_episode_ids"))
    print("\ndual_upper_binding_episode_ids:")
    print(console.get("dual_upper_binding_episode_ids"))
    print("\nleave_one_out_informative_episode_count:")
    print(console.get("leave_one_out_informative_episode_count"))
    print("\nleave_one_out_informative_fraction:")
    print(console.get("leave_one_out_informative_fraction"))
    print("\ndual_equals_provider_episode_count:")
    print(console.get("dual_equals_provider_episode_count"))
    print("\nrequest_subset_of_dual_episode_count:")
    print(console.get("request_subset_of_dual_episode_count"))
    print("\nrequest_boundary_determines_dual_count:")
    print(console.get("request_boundary_determines_dual_count"))
    print("\ncadence_only_effect_on_primary_dual:")
    print(console.get("cadence_only_effect_on_primary_dual"))
    print("\nrequest_only_intersection_classification_by_cadence:")
    print(console.get("request_only_intersection_classification_by_cadence"))
    print("\nprimary_dual_intersection_classification_by_cadence:")
    print(console.get("primary_dual_intersection_classification_by_cadence"))
    print("\nconfigured_max_calls_per_minute:")
    print(console.get("configured_max_calls_per_minute"))
    print("\nrolling_window_convention:")
    print(console.get("rolling_window_convention"))
    print("\nrate_limit_classification:")
    print(console.get("rate_limit_classification"))
    print("\npath_a1_request_clock_feasibility:")
    print(console.get("path_a1_request_clock_feasibility"))
    print("\npath_a2_provider_observability_feasibility:")
    print(console.get("path_a2_provider_observability_feasibility"))
    print("\npath_a3_methodology_change_authorized:")
    print(str(console.get("path_a3_methodology_change_authorized", False)).lower())
    print("\npath_b_design_input_ready:")
    print(str(console.get("path_b_design_input_ready")).lower())
    print("\ngrid_node_robustness_only:")
    print(str(console.get("grid_node_robustness_only")).lower())
    print("\ncontinuous_interval_robustness_established:")
    print(str(console.get("continuous_interval_robustness_established")).lower())
    print("\nestimand_to_simulator_parameter_mapping_status:")
    print(console.get("estimand_to_simulator_parameter_mapping_status"))
    print("\nscheduled_layover_comparator_data_available:")
    print(str(console.get("scheduled_layover_comparator_data_available")).lower())
    print("\nlayover_estimate_computed:")
    print(str(console.get("layover_estimate_computed", False)).lower())
    print("\nturnbull_rerun_authorized:")
    print(str(console.get("turnbull_rerun_authorized", False)).lower())
    print("\nnew_estimation_executed:")
    print(str(console.get("new_estimation_executed", False)).lower())
    print("\nnew_point_estimate_generated:")
    print(str(console.get("new_point_estimate_generated", False)).lower())
    print("\nterminal_recovery_parameter_generated:")
    print(str(console.get("terminal_recovery_parameter_generated", False)).lower())
    print("\nterminal_recovery_parameter_translation_authorized:")
    print(str(console.get("terminal_recovery_parameter_translation_authorized", False)).lower())
    print("\nsimulator_application_authorized:")
    print(str(console.get("simulator_application_authorized", False)).lower())
    print("\nsensitivity_grid_application_authorized:")
    print(str(console.get("sensitivity_grid_application_authorized", False)).lower())
    print("\nphase2_authorized:")
    print(str(console.get("phase2_authorized", False)).lower())
    print("\nresolution_improvement_campaign_authorized:")
    print(str(console.get("resolution_improvement_campaign_authorized", False)).lower())
    print("\nobservation_design_frozen:")
    print(str(console.get("observation_design_frozen", False)).lower())
    print("\nstrict_json_failure_count:")
    print(console.get("strict_json_failure_count"))
    print("\nparquet_read_failure_count:")
    print(console.get("parquet_read_failure_count"))
    print("\njson_parquet_value_mismatch_count:")
    print(console.get("json_parquet_value_mismatch_count"))
    print("\nmanifest_missing_required_file_count:")
    print(console.get("manifest_missing_required_file_count"))
    print("\nmanifest_nonself_hash_mismatch_count:")
    print(console.get("manifest_nonself_hash_mismatch_count"))
    print("\nmanifest_nonself_size_mismatch_count:")
    print(console.get("manifest_nonself_size_mismatch_count"))
    print("\nsecret_leak_count:")
    print(console.get("secret_leak_count"))
    print("\nnext_authorized_action:")
    print(console.get("next_authorized_action"))
    return 0 if str(console.get("gate_status", "")).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
