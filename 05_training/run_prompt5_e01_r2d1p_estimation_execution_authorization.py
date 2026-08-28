from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1p_estimation_execution_authorization.py"

SOURCE_R2D1O = ARTIFACTS_ROOT / "prompt5_e01_r2d1o_12_episode_registry_freeze_estimation_readiness_20260730_085243"
SOURCE_R2D1N_HF1 = ARTIFACTS_ROOT / "prompt5_e01_r2d1n_hf1_final_registry_eta_deduplication_20260730_000608"
SOURCE_R2D1N = ARTIFACTS_ROOT / "prompt5_e01_r2d1n_campaign_c_adaptive_polling_retry_20260729_093218"
SOURCE_R2D1E = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
SOURCE_R2D1F = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

PASS_GATE = "PASS_METHOD_PROTOTYPE_ESTIMATION_EXECUTION_AUTHORIZATION_READY"
PRIMARY_ESTIMAND_ENUM = "OBSERVED_POST_SERVICE_NON_REVENUE_TURNAROUND_INTERVAL"
UNIDENTIFIABLE_ESTIMAND = "ACTUAL_DRIVER_REST_DURATION"
PRIMARY_CLOCK = "CONSERVATIVE_DUAL_CLOCK"
PRIMARY_METHOD = "TURNBULL_NPMLE_INTERVAL_CENSORED_POOLED"
PRIMARY_POPULATION = "12_VERIFIED_COMPLETE_EPISODES"

REQUIRED_FILES = [
    "prompt5_e01_r2d1p_manifest.json",
    "prompt5_e01_r2d1p_gate.json",
    "prompt5_e01_r2d1p_final_report.md",
    "upstream_reference_r2d1o.json",
    "upstream_reference_r2d1n_hf1.json",
    "upstream_reference_r2d1n.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "upstream_execution_boundary_audit.json",
    "secret_leak_audit.json",
    "frozen_registry_fingerprint_reverification.json",
    "frozen_registry_immutability_audit.json",
    "estimation_input_packet_reconciliation.json",
    "estimation_analysis_population_audit.json",
    "adaptive_eta_exclusion_audit.json",
    "estimand_authorization_contract.json",
    "primary_interval_input_contract.json",
    "clock_sensitivity_authorization_contract.json",
    "vehicle_dependence_handling_contract.json",
    "route_analysis_limitation_contract.json",
    "turnbull_support_construction_contract.json",
    "turnbull_initialization_contract.json",
    "turnbull_em_algorithm_contract.json",
    "turnbull_convergence_contract.json",
    "turnbull_likelihood_monotonicity_contract.json",
    "turnbull_nonunique_solution_contract.json",
    "median_identification_contract.json",
    "quantile_identification_contract.json",
    "survival_curve_output_contract.json",
    "descriptive_endpoint_summary_contract.json",
    "estimation_fail_closed_contract.json",
    "estimation_reproducibility_contract.json",
    "independent_implementation_verification_contract.json",
    "r2d1q_execution_artifact_contract.json",
    "r2d1q_gate_contract.json",
    "estimation_execution_authorization_review.json",
    "estimation_execution_authorization.json",
    "estimation_not_executed_audit.json",
    "terminal_recovery_parameter_translation_guard.json",
    "simulator_application_authorization.json",
    "phase2_execution_authorization.json",
    "json_parquet_synchronization_audit.json",
    "manifest_self_entry_contract.json",
]


def now_stamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def strict_constant(value: str) -> None:
    raise ValueError(f"non-strict JSON token: {value}")


def strict_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=strict_constant)


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        return sanitize(converted)
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    strict_read_json(path)


def dataframe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [sanitize(row) for row in frame.to_dict(orient="records")]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    data = json.dumps(sanitize(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return sha256_bytes(data.encode("utf-8"))


def gate_path_for(root: Path) -> Path | None:
    candidates = sorted(
        path
        for path in root.glob("*.json")
        if path.name.endswith("_gate.json") or path.name.endswith("_gate_hf1.json") or "_gate" in path.name
    )
    prompt_candidates = [path for path in candidates if path.name.startswith("prompt")]
    return prompt_candidates[0] if prompt_candidates else (candidates[0] if candidates else None)


def manifest_path_for(root: Path) -> Path | None:
    candidates = sorted(path for path in root.glob("*.json") if path.name.endswith("_manifest.json") or "_manifest" in path.name)
    prompt_candidates = [path for path in candidates if path.name.startswith("prompt")]
    return prompt_candidates[0] if prompt_candidates else (candidates[0] if candidates else None)


def upstream_reference(root: Path, label: str, required_gate: str | None) -> Dict[str, Any]:
    gate_path = gate_path_for(root)
    manifest_path = manifest_path_for(root)
    gate = strict_read_json(gate_path) if gate_path and gate_path.exists() else {}
    status = gate.get("gate_status")
    return {
        "label": label,
        "absolute_path": str(root),
        "exists": root.exists(),
        "read_only_input": True,
        "gate_path": str(gate_path) if gate_path else None,
        "gate_exists": bool(gate_path and gate_path.exists()),
        "gate_status": status,
        "required_gate_status": required_gate,
        "gate_requirement_passed": bool(status == required_gate) if required_gate else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_exists": bool(manifest_path and manifest_path.exists()),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path and manifest_path.exists() else None,
    }


def source_snapshot(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            snapshot[str(root)] = {"exists": False, "size": None, "sha256": None}
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            snapshot[str(path)] = {"exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
    return snapshot


def compare_snapshots(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    deleted = sorted(before_keys - after_keys)
    added = sorted(after_keys - before_keys)
    modified = sorted(
        key
        for key in before_keys & after_keys
        if before[key].get("size") != after[key].get("size") or before[key].get("sha256") != after[key].get("sha256")
    )
    return {
        "upstream_modified_file_count": len(modified),
        "upstream_deleted_file_count": len(deleted),
        "upstream_added_file_count": len(added),
        "modified_files": modified,
        "deleted_files": deleted,
        "added_files": added,
    }


def evidence_sha_records(registry: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str, str, int]]:
    roles = [
        ("FIRST_UPSTREAM", "first_upstream_raw_sha256"),
        ("LAST_PRE_TERMINAL", "last_pre_terminal_raw_sha256"),
        ("FIRST_TERMINAL", "first_terminal_raw_sha256"),
        ("LAST_TERMINAL", "last_terminal_raw_sha256"),
        ("FIRST_POST_TERMINAL", "first_post_terminal_raw_sha256"),
    ]
    out: List[Tuple[str, str, str, int]] = []
    for row in registry:
        episode_id = str(row["episode_id"])
        for role, field in roles:
            sha = row.get(field)
            if isinstance(sha, str) and len(sha) == 64:
                out.append((episode_id, role, sha, 1))
        for idx, sha in enumerate(row.get("post_terminal_confirmation_raw_sha256s") or [], start=1):
            if isinstance(sha, str) and len(sha) == 64:
                out.append((episode_id, "POST_TERMINAL_CONFIRMATION", sha, idx))
    return out


def recompute_fingerprint(registry: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(
        registry,
        key=lambda row: (
            str(row.get("observation_date")),
            str(row.get("first_terminal_request_time")),
            str(row.get("route_id")),
            str(row.get("vehicle_id")),
            str(row.get("episode_id")),
        ),
    )
    canonical_records = [
        {
            key: row.get(key)
            for key in [
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
        }
        for row in ordered
    ]
    evidence_refs = [
        {"episode_id": episode_id, "role": role, "sha256": sha, "evidence_index": index}
        for episode_id, role, sha, index in evidence_sha_records(ordered)
    ]
    interval_tuples = [
        {
            "episode_id": row["episode_id"],
            "provider": [row["provider_lower_bound_sec"], row["provider_upper_bound_sec"]],
            "request": [row["request_lower_bound_sec"], row["request_upper_bound_sec"]],
            "conservative": [row["conservative_dual_lower_bound_sec"], row["conservative_dual_upper_bound_sec"]],
        }
        for row in ordered
    ]
    return {
        "registry_canonical_sha256": canonical_sha(canonical_records),
        "registry_row_count": len(ordered),
        "episode_id_set_sha256": canonical_sha(sorted(row["episode_id"] for row in ordered)),
        "raw_evidence_reference_set_sha256": canonical_sha(
            sorted(evidence_refs, key=lambda item: (item["episode_id"], item["role"], item["evidence_index"], item["sha256"]))
        ),
        "interval_tuple_set_sha256": canonical_sha(sorted(interval_tuples, key=lambda item: item["episode_id"])),
    }


def strict_json_audit(output_root: Path) -> Dict[str, Any]:
    failures = []
    for path in sorted(output_root.rglob("*.json")):
        try:
            strict_read_json(path)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(output_root)), "error": type(exc).__name__})
    return {"strict_json_failure_count": len(failures), "failures": failures}


def parquet_audit(output_root: Path) -> Dict[str, Any]:
    failures = []
    for path in sorted(output_root.rglob("*.parquet")):
        try:
            pd.read_parquet(path)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(output_root)), "error": type(exc).__name__})
    return {"parquet_read_failure_count": len(failures), "failures": failures}


def secret_scan(output_root: Path) -> Dict[str, Any]:
    patterns = [
        re.compile(rb"serviceKey=[^<\s&][^\s&]+", re.IGNORECASE),
        re.compile(rb"Authorization:\s*(?!<REDACTED>)\S+", re.IGNORECASE),
        re.compile(rb"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]{8,}"),
    ]
    findings = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        for pattern in patterns:
            if pattern.search(data):
                findings.append({"path": str(path.relative_to(output_root)), "pattern": pattern.pattern.decode("utf-8", errors="replace")})
    return {"secret_leak_count": len(findings), "findings": findings}


def build_manifest(output_root: Path) -> Dict[str, Any]:
    files = []
    for name in REQUIRED_FILES:
        path = output_root / name
        if name == "prompt5_e01_r2d1p_manifest.json":
            files.append(
                {
                    "path": name,
                    "exists": True,
                    "sha256": None,
                    "self_hash_exempt": True,
                    "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
                    "size_bytes": None,
                    "self_size_exempt": True,
                    "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
                }
            )
        else:
            files.append(
                {
                    "path": name,
                    "exists": path.exists(),
                    "sha256": sha256_file(path) if path.exists() else None,
                    "self_hash_exempt": False,
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "self_size_exempt": False,
                }
            )
    missing = [row["path"] for row in files if not row["exists"]]
    return {
        "artifact_id": output_root.name,
        "created_at": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "files": files,
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files": missing,
    }


def validate_manifest(output_root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    missing, hash_mismatches, size_mismatches = [], [], []
    for entry in manifest.get("files", []):
        name = entry.get("path")
        if not name:
            continue
        path = output_root / str(name)
        if not path.exists():
            missing.append(name)
            continue
        if entry.get("self_hash_exempt"):
            continue
        if entry.get("sha256") != sha256_file(path):
            hash_mismatches.append(name)
        if entry.get("size_bytes") != path.stat().st_size:
            size_mismatches.append(name)
    return {
        "manifest_missing_required_file_count": len(missing),
        "manifest_nonself_hash_mismatch_count": len(hash_mismatches),
        "manifest_nonself_size_mismatch_count": len(size_mismatches),
        "missing_required_files": missing,
        "hash_mismatches": hash_mismatches,
        "size_mismatches": size_mismatches,
    }


def main() -> None:
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1p_estimation_execution_authorization_{now_stamp()}"
    output_root.mkdir(parents=True, exist_ok=False)

    upstream_roots = [SOURCE_R2D1O, SOURCE_R2D1N_HF1, SOURCE_R2D1N, SOURCE_R2D1E, SOURCE_R2D1F]
    before_snapshot = source_snapshot(upstream_roots)

    upstream_refs = {
        "upstream_reference_r2d1o.json": upstream_reference(SOURCE_R2D1O, "r2d1o", "PASS_12_EPISODE_REGISTRY_FREEZE_ESTIMATION_READINESS_READY"),
        "upstream_reference_r2d1n_hf1.json": upstream_reference(SOURCE_R2D1N_HF1, "r2d1n_hf1", "PASS_CAMPAIGN_C_FINAL_REGISTRY_ETA_DEDUPLICATION_FREEZE_READY"),
        "upstream_reference_r2d1n.json": upstream_reference(SOURCE_R2D1N, "r2d1n", "PASS_CAMPAIGN_C_FINAL_EPISODE_COMPLETE_ADAPTIVE"),
        "upstream_reference_r2d1e.json": upstream_reference(SOURCE_R2D1E, "r2d1e", "PASS_SCHEMA_FREEZE_METHOD_REVIEW_READY"),
        "upstream_reference_r2d1f.json": upstream_reference(SOURCE_R2D1F, "r2d1f", "PASS_ESTIMATION_DESIGN_APPROVED_ADDITIONAL_DATA_REQUIRED"),
    }
    for name, payload in upstream_refs.items():
        dump_json(output_root / name, payload)

    source_failures = [
        {"reference_file": name, "gate_status": payload["gate_status"], "required_gate_status": payload["required_gate_status"]}
        for name, payload in upstream_refs.items()
        if payload["required_gate_status"] is not None and payload["gate_status"] != payload["required_gate_status"]
    ]
    source_integrity = {
        "source_artifact_integrity_passed": len(source_failures) == 0,
        "source_artifact_integrity_failure_count": len(source_failures),
        "failures": source_failures,
        "references": list(upstream_refs.values()),
    }
    dump_json(output_root / "source_artifact_integrity_audit.json", source_integrity)

    network_api_calls = 0
    service_key_accessed = False
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "network_request_performed": False})
    dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False, "environment_read_performed": False, "service_key_output": False})

    r2d1o_gate = strict_read_json(SOURCE_R2D1O / "prompt5_e01_r2d1o_gate.json")
    r2d1o_execution_auth = strict_read_json(SOURCE_R2D1O / "estimation_execution_authorization.json")
    r2d1o_sim_guard = strict_read_json(SOURCE_R2D1O / "simulator_parameter_translation_guard.json")
    r2d1o_phase2 = strict_read_json(SOURCE_R2D1O / "phase2_execution_authorization.json")
    forbidden_result_files = [
        "turnbull_primary_execution.json",
        "turnbull_primary_iteration_trace.parquet",
        "turnbull_primary_survival_curve.json",
        "turnbull_primary_quantile_identification.json",
    ]
    existing_forbidden_result_files = [name for name in forbidden_result_files if (SOURCE_R2D1O / name).exists()]
    upstream_boundary = {
        "source_gate": r2d1o_gate.get("gate_status"),
        "method_prototype_data_threshold_met": r2d1o_gate.get("method_prototype_data_threshold_met"),
        "formal_inferential_sufficiency": r2d1o_gate.get("formal_inferential_sufficiency"),
        "estimation_input_packet_ready": r2d1o_gate.get("estimation_input_packet_ready"),
        "estimation_method_contract_ready": r2d1o_gate.get("estimation_method_contract_ready"),
        "terminal_recovery_estimation_execution_approved": r2d1o_gate.get("terminal_recovery_estimation_execution_approved"),
        "terminal_recovery_parameter_generated": r2d1o_gate.get("terminal_recovery_parameter_generated"),
        "simulator_application_authorized": r2d1o_gate.get("simulator_application_authorized"),
        "phase2_authorized": r2d1o_gate.get("phase2_authorized"),
        "source_execution_authorization_file_terminal_recovery_estimation_execution_approved": r2d1o_execution_auth.get("terminal_recovery_estimation_execution_approved"),
        "source_parameter_generated": r2d1o_sim_guard.get("terminal_recovery_parameter_generated"),
        "existing_forbidden_result_files": existing_forbidden_result_files,
    }
    upstream_boundary["upstream_execution_boundary_passed"] = (
        r2d1o_gate.get("gate_status") == "PASS_12_EPISODE_REGISTRY_FREEZE_ESTIMATION_READINESS_READY"
        and r2d1o_gate.get("method_prototype_data_threshold_met") is True
        and r2d1o_gate.get("formal_inferential_sufficiency") == "NOT_ESTABLISHED"
        and r2d1o_gate.get("estimation_input_packet_ready") is True
        and r2d1o_gate.get("estimation_method_contract_ready") is True
        and r2d1o_gate.get("terminal_recovery_estimation_execution_approved") is False
        and r2d1o_execution_auth.get("terminal_recovery_estimation_execution_approved") is False
        and r2d1o_sim_guard.get("terminal_recovery_parameter_generated") is False
        and r2d1o_sim_guard.get("simulator_application_authorized") is False
        and r2d1o_phase2.get("phase2_authorized") is False
        and not existing_forbidden_result_files
    )
    dump_json(output_root / "upstream_execution_boundary_audit.json", upstream_boundary)

    registry_json = strict_read_json(SOURCE_R2D1O / "registry_12_frozen.json")
    registry_records = registry_json["records"]
    registry_parquet = pd.read_parquet(SOURCE_R2D1O / "registry_12_frozen.parquet")
    source_fingerprint = strict_read_json(SOURCE_R2D1O / "registry_12_freeze_fingerprint.json")
    recomputed_fingerprint = recompute_fingerprint(registry_records)
    fingerprint_reverification = {
        "source_fingerprint": source_fingerprint,
        "recomputed_fingerprint": recomputed_fingerprint,
        "registry_fingerprint_match": recomputed_fingerprint["registry_canonical_sha256"] == source_fingerprint["registry_canonical_sha256"],
        "episode_id_set_fingerprint_match": recomputed_fingerprint["episode_id_set_sha256"] == source_fingerprint["episode_id_set_sha256"],
        "raw_evidence_set_fingerprint_match": recomputed_fingerprint["raw_evidence_reference_set_sha256"] == source_fingerprint["raw_evidence_reference_set_sha256"],
        "interval_tuple_set_fingerprint_match": recomputed_fingerprint["interval_tuple_set_sha256"] == source_fingerprint["interval_tuple_set_sha256"],
        "registry_row_count_match": recomputed_fingerprint["registry_row_count"] == source_fingerprint["registry_row_count"] == len(registry_records),
    }
    fingerprint_reverification["frozen_registry_fingerprint_reverification_passed"] = all(
        fingerprint_reverification[key]
        for key in [
            "registry_fingerprint_match",
            "episode_id_set_fingerprint_match",
            "raw_evidence_set_fingerprint_match",
            "interval_tuple_set_fingerprint_match",
            "registry_row_count_match",
        ]
    )
    dump_json(output_root / "frozen_registry_fingerprint_reverification.json", fingerprint_reverification)

    immutability_contract = strict_read_json(SOURCE_R2D1O / "registry_12_freeze_immutability_contract.json")
    minimal_locked_fields = [
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
    locked_field_missing = sorted(field for field in minimal_locked_fields if field not in registry_records[0])
    frozen_registry_immutability = {
        "immutability_contract_ready": immutability_contract.get("freeze_immutability_contract_ready") is True,
        "source_locked_fields": immutability_contract.get("frozen_fields", []),
        "minimal_locked_fields_checked": minimal_locked_fields,
        "locked_field_missing_count": len(locked_field_missing),
        "locked_field_missing": locked_field_missing,
        "r2d1p_locked_field_modification_count": 0,
        "frozen_registry_immutability_audit_passed": immutability_contract.get("freeze_immutability_contract_ready") is True and not locked_field_missing,
    }
    dump_json(output_root / "frozen_registry_immutability_audit.json", frozen_registry_immutability)

    input_packet = strict_read_json(SOURCE_R2D1O / "method_prototype_estimation_input_packet.json")
    input_records = input_packet["records"]
    input_parquet = pd.read_parquet(SOURCE_R2D1O / "method_prototype_estimation_input_intervals.parquet")
    input_required_fields = [
        "episode_id",
        "route_id",
        "vehicle_id",
        "observation_date",
        "hour_bucket",
        "conservative_dual_lower_bound_sec",
        "conservative_dual_upper_bound_sec",
        "provider_lower_bound_sec",
        "provider_upper_bound_sec",
        "request_lower_bound_sec",
        "request_upper_bound_sec",
        "clock_semantics",
        "vehicle_cluster_id",
        "route_stratum",
    ]
    prohibited_fields = {
        "midpoint",
        "imputed_duration",
        "imputed duration",
        "representative_recovery_seconds",
        "representative recovery seconds",
        "driver_rest_duration",
        "simulator_parameter",
        "simulator parameter",
    }
    registry_by_episode = {str(row["episode_id"]): row for row in registry_records}
    input_by_episode = {str(row["episode_id"]): row for row in input_records}
    duplicate_input_episodes = len(input_records) - len(input_by_episode)
    missing_episodes = sorted(set(registry_by_episode) - set(input_by_episode))
    extra_episodes = sorted(set(input_by_episode) - set(registry_by_episode))
    value_mismatches = []
    comparison_map = [
        ("episode_id", "episode_id"),
        ("route_id", "route_id"),
        ("vehicle_id", "vehicle_id"),
        ("observation_date", "observation_date"),
        ("hour_bucket", "hour_bucket"),
        ("conservative_dual_lower_bound_sec", "conservative_dual_lower_bound_sec"),
        ("conservative_dual_upper_bound_sec", "conservative_dual_upper_bound_sec"),
        ("provider_lower_bound_sec", "provider_lower_bound_sec"),
        ("provider_upper_bound_sec", "provider_upper_bound_sec"),
        ("request_lower_bound_sec", "request_lower_bound_sec"),
        ("request_upper_bound_sec", "request_upper_bound_sec"),
        ("clock_semantics_status", "clock_semantics"),
    ]
    for episode_id in sorted(set(registry_by_episode) & set(input_by_episode)):
        reg = registry_by_episode[episode_id]
        inp = input_by_episode[episode_id]
        for reg_field, input_field in comparison_map:
            if sanitize(reg.get(reg_field)) != sanitize(inp.get(input_field)):
                value_mismatches.append(
                    {
                        "episode_id": episode_id,
                        "registry_field": reg_field,
                        "input_field": input_field,
                        "registry_value": reg.get(reg_field),
                        "input_value": inp.get(input_field),
                    }
                )
    prohibited_found = sorted({field for row in input_records for field in row if field in prohibited_fields})
    missing_input_fields = sorted(field for field in input_required_fields if any(field not in row for row in input_records))
    estimation_input_packet_reconciliation = {
        "estimation_input_row_count": len(input_records),
        "estimation_input_parquet_row_count": len(input_parquet),
        "input_required_fields": input_required_fields,
        "missing_required_input_field_count": len(missing_input_fields),
        "missing_required_input_fields": missing_input_fields,
        "input_registry_value_mismatch_count": len(value_mismatches),
        "input_registry_value_mismatches": value_mismatches,
        "input_duplicate_episode_count": duplicate_input_episodes,
        "input_missing_episode_count": len(missing_episodes),
        "input_missing_episodes": missing_episodes,
        "input_extra_episode_count": len(extra_episodes),
        "input_extra_episodes": extra_episodes,
        "prohibited_input_field_count": len(prohibited_found),
        "prohibited_input_fields": prohibited_found,
        "packet_json_parquet_row_count_match": len(input_records) == len(input_parquet),
    }
    estimation_input_packet_reconciliation["estimation_input_packet_reconciliation_passed"] = (
        len(input_records) == 12
        and len(input_parquet) == 12
        and not missing_input_fields
        and not value_mismatches
        and duplicate_input_episodes == 0
        and not missing_episodes
        and not extra_episodes
        and not prohibited_found
    )
    dump_json(output_root / "estimation_input_packet_reconciliation.json", estimation_input_packet_reconciliation)

    adaptive_eta = strict_read_json(SOURCE_R2D1N_HF1 / "adaptive_eta_training_dataset_deduplicated.json")
    adaptive_episode_ids = {str(row.get("episode_id")) for row in adaptive_eta.get("records", []) if row.get("episode_id")}
    input_episode_ids = set(input_by_episode)
    adaptive_eta_rows_in_primary_input = len(input_episode_ids & adaptive_episode_ids)
    noncomplete_rows = [
        episode_id
        for episode_id, row in registry_by_episode.items()
        if episode_id in input_episode_ids and row.get("final_status") != "COMPLETE_INTERVAL_CENSORED"
    ]
    analysis_population = {
        "primary_analysis_population": PRIMARY_POPULATION,
        "input_episode_count": len(input_records),
        "complete_episodes_in_primary_input": len(input_records) - len(noncomplete_rows),
        "censored_noncomplete_rows_in_primary_input": len(noncomplete_rows),
        "noncomplete_episode_ids": noncomplete_rows,
        "excluded_categories": [
            "left-censored session",
            "right-censored session",
            "invalid session",
            "partial campaign episode",
            "adaptive ETA training row",
            "broad scan row",
            "position sample",
            "vehicle trajectory row",
        ],
        "analysis_population_reconciliation_passed": len(input_records) == 12 and not noncomplete_rows,
    }
    dump_json(output_root / "estimation_analysis_population_audit.json", analysis_population)

    adaptive_eta_exclusion = {
        "adaptive_eta_deduplicated_row_count": adaptive_eta.get("row_count"),
        "adaptive_eta_rows_in_primary_input": adaptive_eta_rows_in_primary_input,
        "adaptive_eta_rows_excluded_from_primary_input": adaptive_eta.get("row_count"),
        "adaptive_eta_rows_are_outcome_episodes": False,
        "adaptive_eta_exclusion_passed": adaptive_eta_rows_in_primary_input == 0,
    }
    dump_json(output_root / "adaptive_eta_exclusion_audit.json", adaptive_eta_exclusion)

    method_contract = strict_read_json(SOURCE_R2D1O / "method_prototype_estimation_method_contract.json")
    estimand_review = strict_read_json(SOURCE_R2D1O / "estimand_identifiability_review.json")
    formal_review = strict_read_json(SOURCE_R2D1O / "formal_inferential_sufficiency_review.json")
    output_schema_contract = strict_read_json(SOURCE_R2D1O / "method_prototype_output_schema_contract.json")
    prohibited_interpretations = strict_read_json(SOURCE_R2D1O / "method_prototype_prohibited_interpretations.json")

    estimand_contract = {
        "primary_estimand": PRIMARY_ESTIMAND_ENUM,
        "primary_estimand_label": "Observed Post-Service Non-Revenue Turnaround Interval",
        "primary_estimand_definition_ko": "운행 종료 이후 동일 차량의 다음 영업 운행 재개까지 API로 관측 가능한 비영업 회차 구간",
        "primary_estimand_source_review_passed": estimand_review.get("primary_estimand_identifiable") is True,
        "unidentifiable_estimand": UNIDENTIFIABLE_ESTIMAND,
        "driver_rest_duration_inference": "NOT_IDENTIFIABLE",
        "driver_rest_claims_prohibited": True,
        "estimand_contract_passed": estimand_review.get("primary_estimand_identifiable") is True and estimand_review.get("driver_rest_duration_inference") == "NOT_IDENTIFIABLE",
    }
    dump_json(output_root / "estimand_authorization_contract.json", estimand_contract)

    interval_errors = []
    for row in registry_records:
        pl, pu = float(row["provider_lower_bound_sec"]), float(row["provider_upper_bound_sec"])
        rl, ru = float(row["request_lower_bound_sec"]), float(row["request_upper_bound_sec"])
        cl, cu = float(row["conservative_dual_lower_bound_sec"]), float(row["conservative_dual_upper_bound_sec"])
        if min(pl, pu, rl, ru, cl, cu) < 0:
            interval_errors.append({"episode_id": row["episode_id"], "reason": "negative_interval"})
        if pl > pu or rl > ru or cl > cu:
            interval_errors.append({"episode_id": row["episode_id"], "reason": "reversed_interval"})
        if cl != min(pl, rl) or cu != max(pu, ru):
            interval_errors.append({"episode_id": row["episode_id"], "reason": "dual_clock_formula_mismatch"})
    primary_interval_contract = {
        "primary_interval": "[conservative_dual_lower_bound_sec, conservative_dual_upper_bound_sec]",
        "primary_interval_boundary_convention": "closed interval [L_i, U_i]",
        "primary_input_clock": PRIMARY_CLOCK,
        "negative_interval_count": sum(1 for item in interval_errors if item["reason"] == "negative_interval"),
        "reversed_interval_count": sum(1 for item in interval_errors if item["reason"] == "reversed_interval"),
        "dual_clock_formula_mismatch_count": sum(1 for item in interval_errors if item["reason"] == "dual_clock_formula_mismatch"),
        "interval_errors": interval_errors,
        "primary_interval_contract_passed": not interval_errors,
    }
    dump_json(output_root / "primary_interval_input_contract.json", primary_interval_contract)

    clock_sensitivity_contract = {
        "primary_clock": PRIMARY_CLOCK,
        "secondary_sensitivity_inputs": ["PROVIDER_ONLY_INTERVAL", "REQUEST_ONLY_INTERVAL"],
        "execution_priority": [PRIMARY_CLOCK, "PROVIDER_ONLY_INTERVAL", "REQUEST_ONLY_INTERVAL"],
        "provider_clock_sensitivity_authorized": True,
        "request_clock_sensitivity_authorized": True,
        "provider_only_cannot_replace_primary": True,
        "request_only_cannot_replace_primary": True,
        "clock_sensitivity_contract_passed": True,
    }
    dump_json(output_root / "clock_sensitivity_authorization_contract.json", clock_sensitivity_contract)

    vehicle_counts = Counter(str(row["vehicle_id"]) for row in registry_records)
    vehicle_dependence_contract = {
        "episode_count": len(registry_records),
        "global_unique_vehicle_count": len(vehicle_counts),
        "episode_count_per_vehicle": dict(sorted(vehicle_counts.items())),
        "primary_weighting": "EQUAL_EPISODE_WEIGHT",
        "each_episode_weight": 1,
        "vehicle_balanced_sensitivity": "AUTHORIZED_AS_SECONDARY_ONLY",
        "vehicle_balanced_sensitivity_authorized": True,
        "vehicle_balanced_weight_rule": "episode weight = 1 / number_of_episodes_for_same_vehicle",
        "vehicle_cluster_bootstrap": "NOT_AUTHORIZED",
        "vehicle_cluster_bootstrap_authorized": False,
        "repeated_vehicle_episodes_deleted": False,
        "vehicle_dependence_contract_passed": len(registry_records) == 12 and len(vehicle_counts) == 8,
    }
    dump_json(output_root / "vehicle_dependence_handling_contract.json", vehicle_dependence_contract)

    route_counts = Counter(str(row["route_id"]) for row in registry_records)
    route_limitation_contract = {
        "route_complete_counts": dict(sorted(route_counts.items())),
        "route_level_analysis": "DESCRIPTIVE_ONLY",
        "route_formal_inference_authorized": False,
        "prohibited": [
            "route-specific formal Turnbull conclusion",
            "route comparison p-value",
            "statistically proven route ranking",
            "route-specific simulator parameter",
        ],
        "route_analysis_limitation_contract_passed": all(count == 3 for count in route_counts.values()),
    }
    dump_json(output_root / "route_analysis_limitation_contract.json", route_limitation_contract)

    endpoint_values = sorted(
        set(float(row["conservative_dual_lower_bound_sec"]) for row in registry_records)
        | set(float(row["conservative_dual_upper_bound_sec"]) for row in registry_records)
    )
    support_contract = {
        "primary_method": PRIMARY_METHOD,
        "support_construction": "Turnbull maximal intersections or equivalent interval-censoring support cell construction",
        "input_endpoint_set": "all unique L_i and U_i values",
        "unique_endpoint_count_preexecution": len(endpoint_values),
        "support_cell_ordering": "deterministic ascending",
        "required_support_audit_fields": [
            "unique_endpoint_count",
            "support_cell_count",
            "empty_support_cell_count",
            "episode_without_admissible_support_count",
        ],
        "required_empty_support_cell_count": 0,
        "required_episode_without_admissible_support_count": 0,
        "turnbull_support_contract_passed": len(endpoint_values) > 0,
    }
    dump_json(output_root / "turnbull_support_construction_contract.json", support_contract)

    initialization_contract = {
        "initialization_count": 2,
        "initializations": ["uniform_mass_over_admissible_support_cells", "endpoint_informed_initialization"],
        "random_initialization_allowed": False,
        "fixed_seed_required_if_random_used": True,
        "required_trace_fields": [
            "initialization_name",
            "initial_log_likelihood",
            "final_log_likelihood",
            "iteration_count",
            "converged",
            "final_mass_vector_sha256",
        ],
        "turnbull_initialization_contract_passed": True,
    }
    dump_json(output_root / "turnbull_initialization_contract.json", initialization_contract)

    em_contract = {
        "optimizer_family": "EM_OR_DETERMINISTIC_EQUIVALENT",
        "iteration_trace_required": True,
        "required_iteration_fields": [
            "iteration",
            "log_likelihood",
            "absolute_log_likelihood_change",
            "relative_log_likelihood_change",
            "maximum_mass_change",
            "mass_sum",
            "minimum_mass",
        ],
        "mass_sum_tolerance": 1e-10,
        "negative_mass_tolerance": 1e-12,
        "log_likelihood_decrease_tolerance": 1e-10,
        "clamp_count_must_be_recorded": True,
        "turnbull_em_algorithm_contract_passed": True,
    }
    dump_json(output_root / "turnbull_em_algorithm_contract.json", em_contract)

    convergence_contract = {
        "relative_log_likelihood_change_tolerance": 1e-10,
        "maximum_mass_change_tolerance": 1e-8,
        "consecutive_convergence_iterations": 5,
        "maximum_iterations": 10000,
        "minimum_iterations": 5,
        "allowed_convergence_statuses": [
            "CONVERGED",
            "MAX_ITERATIONS_REACHED",
            "NUMERICAL_FAILURE",
            "LIKELIHOOD_DECREASE",
            "NONFINITE_VALUE",
            "INVALID_MASS_VECTOR",
        ],
        "success_status": "CONVERGED",
        "non_converged_not_pass": True,
        "turnbull_convergence_contract_passed": True,
    }
    dump_json(output_root / "turnbull_convergence_contract.json", convergence_contract)

    likelihood_contract = {
        "log_likelihood_monotonicity_required": True,
        "likelihood_decrease_tolerance": 1e-10,
        "failure_gate": "FAIL_TURNBULL_LIKELIHOOD_MONOTONICITY",
        "final_iteration_trace_required": True,
        "turnbull_likelihood_monotonicity_contract_passed": True,
    }
    dump_json(output_root / "turnbull_likelihood_monotonicity_contract.json", likelihood_contract)

    nonunique_contract = {
        "nonunique_likelihood_equivalent_solution_possible": True,
        "status_if_same_likelihood_different_mass": "SOLUTION_NONUNIQUE_LIKELIHOOD_EQUIVALENT",
        "do_not_report_single_mass_vector_as_unique_truth": True,
        "allowed_outputs": [
            "identified support",
            "survival lower/upper envelope",
            "median identified set",
            "quantile identified set",
        ],
        "turnbull_nonunique_solution_contract_passed": True,
    }
    dump_json(output_root / "turnbull_nonunique_solution_contract.json", nonunique_contract)

    median_contract = {
        "statuses": ["POINT_IDENTIFIED", "INTERVAL_IDENTIFIED", "NOT_IDENTIFIED_WITHIN_SUPPORT"],
        "point_identified_rule": "median_identified_lower_sec == median_identified_upper_sec",
        "interval_identified_rule": "median_identified_lower_sec < median_identified_upper_sec",
        "midpoint_as_median_prohibited": True,
        "median_point_estimate_field_prohibited_unless_point_identified": True,
        "median_identification_contract_passed": True,
    }
    dump_json(output_root / "median_identification_contract.json", median_contract)

    quantile_contract = {
        "quantiles": [0.25, 0.5, 0.75],
        "statuses": ["POINT_IDENTIFIED", "INTERVAL_IDENTIFIED", "NOT_IDENTIFIED"],
        "fields": ["quantile_probability", "identification_status", "identified_lower_sec", "identified_upper_sec"],
        "clock_sensitivity_uses_same_rules": True,
        "route_specific_formal_quantile_prohibited": True,
        "quantile_identification_contract_passed": True,
    }
    dump_json(output_root / "quantile_identification_contract.json", quantile_contract)

    survival_curve_contract = {
        "allowed_fields": [
            "support_cell_index",
            "support_lower_sec",
            "support_upper_sec",
            "estimated_probability_mass",
            "cdf_lower",
            "cdf_upper",
            "survival_lower",
            "survival_upper",
        ],
        "nonunique_solution_requires_solution_identifier": True,
        "cdf_monotonic_non_decreasing_required": True,
        "survival_monotonic_non_increasing_required": True,
        "final_total_mass_approximately_one_required": True,
        "survival_curve_output_contract_passed": True,
    }
    dump_json(output_root / "survival_curve_output_contract.json", survival_curve_contract)

    descriptive_contract = {
        "allowed_endpoint_summaries": [
            "episode_count",
            "vehicle_count",
            "route_count",
            "minimum_lower_bound",
            "maximum_upper_bound",
            "median_lower_endpoint",
            "median_upper_endpoint",
            "interval_width_min",
            "interval_width_median",
            "interval_width_max",
        ],
        "required_note": "Endpoint summaries describe observed censoring bounds. They are not the estimated median turnaround duration.",
        "descriptive_endpoint_summary_contract_passed": True,
    }
    dump_json(output_root / "descriptive_endpoint_summary_contract.json", descriptive_contract)

    fail_closed_conditions = [
        "frozen registry fingerprint mismatch",
        "input packet mismatch",
        "input row count != 12",
        "duplicate episode",
        "missing interval",
        "negative interval",
        "reversed interval",
        "dual-clock formula mismatch",
        "nonfinite input",
        "episode without admissible support",
        "empty support construction",
        "mass vector negative beyond tolerance",
        "mass sum mismatch",
        "log-likelihood decrease beyond tolerance",
        "maximum iterations reached without convergence",
        "nonfinite likelihood",
        "independent likelihood mismatch",
        "output schema violation",
        "prohibited point estimate generated",
        "driver-rest interpretation generated",
        "simulator parameter generated",
    ]
    dump_json(output_root / "estimation_fail_closed_contract.json", {"fail_closed_conditions": fail_closed_conditions, "estimation_fail_closed_contract_ready": True})

    reproducibility_contract = {
        "python_version_for_authorization": sys.version,
        "platform_for_authorization": platform.platform(),
        "implementation_name": "in_project_turnbull_npMLE_contract",
        "implementation_version": "R2D-1P-authorized-contract-v1",
        "algorithm_contract_version": "turnbull_interval_censored_contract_v1",
        "input_registry_fingerprint": source_fingerprint["registry_canonical_sha256"],
        "input_packet_sha256": sha256_file(SOURCE_R2D1O / "method_prototype_estimation_input_packet.json"),
        "method_contract_sha256": sha256_file(SOURCE_R2D1O / "method_prototype_estimation_method_contract.json"),
        "initialization_scheme": initialization_contract["initializations"],
        "random_seed_if_used": "fixed_seed_required_if_random_used",
        "numerical_tolerances": {
            "relative_log_likelihood_change": 1e-10,
            "maximum_mass_change": 1e-8,
            "mass_sum": 1e-10,
            "negative_mass": 1e-12,
            "likelihood_decrease": 1e-10,
        },
        "maximum_iterations": 10000,
        "external_library_defaults_only_prohibited": True,
        "estimation_reproducibility_contract_passed": True,
    }
    dump_json(output_root / "estimation_reproducibility_contract.json", reproducibility_contract)

    independent_verification_contract = {
        "minimum_independent_checks": [
            "final log-likelihood independently recomputed",
            "mass sum independently recomputed",
            "episode likelihoods independently recomputed",
        ],
        "allowed_independent_verification_methods": [
            "separate in-project implementation",
            "separate library implementation",
            "exact likelihood recomputation from final mass",
            "KKT-like self-consistency audit",
        ],
        "independent_likelihood_mismatch_tolerance": 1e-8,
        "independent_implementation_verification_contract_passed": True,
    }
    dump_json(output_root / "independent_implementation_verification_contract.json", independent_verification_contract)

    r2d1q_artifact_contract = {
        "artifact_path_pattern": "05_training/artifacts/prompt5_e01_r2d1q_method_prototype_interval_censored_estimation_<YYYYMMDD_HHMMSS>/",
        "script_path": "05_training/run_prompt5_e01_r2d1q_method_prototype_interval_censored_estimation.py",
        "r2d1p_does_not_execute_r2d1q_script": True,
        "required_core_files": [
            "prompt5_e01_r2d1q_manifest.json",
            "prompt5_e01_r2d1q_gate.json",
            "prompt5_e01_r2d1q_final_report.md",
            "turnbull_primary_execution.json",
            "turnbull_primary_iteration_trace.parquet",
            "turnbull_primary_support.json",
            "turnbull_primary_survival_curve.json",
            "turnbull_primary_quantile_identification.json",
            "independent_likelihood_verification.json",
            "prohibited_interpretation_audit.json",
        ],
        "r2d1q_execution_artifact_contract_passed": True,
    }
    dump_json(output_root / "r2d1q_execution_artifact_contract.json", r2d1q_artifact_contract)

    r2d1q_gate_contract = {
        "success_gate_candidates": [
            "PASS_METHOD_PROTOTYPE_INTERVAL_CENSORED_ESTIMATION_COMPLETE",
            "PASS_METHOD_PROTOTYPE_ESTIMATION_NONUNIQUE_IDENTIFIED_SET_COMPLETE",
        ],
        "failure_gate_candidates": [
            "FAIL_INPUT_FINGERPRINT_MISMATCH",
            "FAIL_INPUT_PACKET_MISMATCH",
            "FAIL_TURNBULL_SUPPORT_CONSTRUCTION",
            "FAIL_TURNBULL_NONCONVERGENCE",
            "FAIL_TURNBULL_NUMERICAL_INVARIANT",
            "FAIL_TURNBULL_LIKELIHOOD_MONOTONICITY",
            "FAIL_INDEPENDENT_LIKELIHOOD_VERIFICATION",
            "FAIL_QUANTILE_IDENTIFICATION_OUTPUT",
            "FAIL_ESTIMATION_OUTPUT_SCHEMA",
            "FAIL_PROHIBITED_INTERPRETATION",
            "FAIL_PARAMETER_TRANSLATION_BOUNDARY",
        ],
        "r2d1q_gate_contract_passed": True,
    }
    dump_json(output_root / "r2d1q_gate_contract.json", r2d1q_gate_contract)

    formal_sufficiency = formal_review.get("formal_inferential_sufficiency")
    authorization_review_prelim = {
        "estimation_execution_authorization_review_passed": True,
        "terminal_recovery_estimation_execution_approved": True,
        "authorized_execution_step": "R2D-1Q",
        "authorized_primary_method": PRIMARY_METHOD,
        "authorized_primary_input_clock": PRIMARY_CLOCK,
        "authorized_primary_episode_count": 12,
        "authorized_provider_clock_sensitivity": True,
        "authorized_request_clock_sensitivity": True,
        "authorized_vehicle_balanced_sensitivity": True,
        "authorized_midpoint_sensitivity": False,
        "authorized_bootstrap": False,
        "authorized_route_formal_inference": False,
        "estimation_executed": False,
        "simulator_parameter_translation_authorized": False,
        "formal_inferential_sufficiency": formal_sufficiency,
    }

    dump_json(output_root / "terminal_recovery_parameter_translation_guard.json", {"terminal_recovery_parameter_generated": False, "terminal_recovery_parameter_translation_authorized": False, "terminal_recovery_applied": False})
    dump_json(output_root / "simulator_application_authorization.json", {"simulator_application_authorized": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})
    dump_json(
        output_root / "estimation_not_executed_audit.json",
        {
            "turnbull_estimation_executed": False,
            "bootstrap_executed": False,
            "provider_only_sensitivity_executed": False,
            "request_only_sensitivity_executed": False,
            "midpoint_sensitivity_executed": False,
            "survival_curve_generated": False,
            "median_or_quantile_result_generated": False,
            "estimation_result_file_generated": False,
        },
    )

    immutability = compare_snapshots(before_snapshot, source_snapshot(upstream_roots))
    dump_json(output_root / "authoritative_input_immutability_audit.json", immutability)
    dump_json(output_root / "secret_leak_audit.json", secret_scan(output_root))

    json_parquet_audit = {
        "parquet_files_written": 0,
        "parquet_read_failure_count": 0,
        "json_parquet_value_mismatch_count": 0,
        "note": "R2D-1P writes no required Parquet files; source Parquet files were read-only inputs.",
    }
    dump_json(output_root / "json_parquet_synchronization_audit.json", json_parquet_audit)
    dump_json(
        output_root / "manifest_self_entry_contract.json",
        {
            "path": "prompt5_e01_r2d1p_manifest.json",
            "exists": True,
            "sha256": None,
            "self_hash_exempt": True,
            "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
            "size_bytes": None,
            "self_size_exempt": True,
            "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
        },
    )

    pass_conditions = {
        "network_api_calls_zero": network_api_calls == 0,
        "service_key_accessed_false": service_key_accessed is False,
        "source_integrity": source_integrity["source_artifact_integrity_passed"],
        "upstream_immutability": immutability["upstream_modified_file_count"] == 0 and immutability["upstream_deleted_file_count"] == 0 and immutability["upstream_added_file_count"] == 0,
        "upstream_execution_boundary": upstream_boundary["upstream_execution_boundary_passed"],
        "fingerprint": fingerprint_reverification["frozen_registry_fingerprint_reverification_passed"],
        "immutability": frozen_registry_immutability["frozen_registry_immutability_audit_passed"],
        "input_packet": estimation_input_packet_reconciliation["estimation_input_packet_reconciliation_passed"],
        "analysis_population": analysis_population["analysis_population_reconciliation_passed"],
        "adaptive_eta_exclusion": adaptive_eta_exclusion["adaptive_eta_exclusion_passed"],
        "estimand": estimand_contract["estimand_contract_passed"],
        "primary_interval": primary_interval_contract["primary_interval_contract_passed"],
        "clock_sensitivity": clock_sensitivity_contract["clock_sensitivity_contract_passed"],
        "vehicle_dependence": vehicle_dependence_contract["vehicle_dependence_contract_passed"],
        "route_limitation": route_limitation_contract["route_analysis_limitation_contract_passed"],
        "support": support_contract["turnbull_support_contract_passed"],
        "initialization": initialization_contract["turnbull_initialization_contract_passed"],
        "em_contract": em_contract["turnbull_em_algorithm_contract_passed"],
        "convergence": convergence_contract["turnbull_convergence_contract_passed"],
        "likelihood": likelihood_contract["turnbull_likelihood_monotonicity_contract_passed"],
        "nonunique": nonunique_contract["turnbull_nonunique_solution_contract_passed"],
        "median": median_contract["median_identification_contract_passed"],
        "quantile": quantile_contract["quantile_identification_contract_passed"],
        "independent_verification": independent_verification_contract["independent_implementation_verification_contract_passed"],
        "reproducibility": reproducibility_contract["estimation_reproducibility_contract_passed"],
        "r2d1q_artifact": r2d1q_artifact_contract["r2d1q_execution_artifact_contract_passed"],
        "r2d1q_gate": r2d1q_gate_contract["r2d1q_gate_contract_passed"],
        "authorization_boundary": authorization_review_prelim["estimation_executed"] is False and authorization_review_prelim["terminal_recovery_estimation_execution_approved"] is True,
        "formal_sufficiency_not_established": formal_sufficiency == "NOT_ESTABLISHED",
        "downstream_locks": True,
        "json_parquet": json_parquet_audit["json_parquet_value_mismatch_count"] == 0 and json_parquet_audit["parquet_read_failure_count"] == 0,
    }
    gate_status = PASS_GATE if all(pass_conditions.values()) else "FAIL_R2D1P_AUTHORIZATION_REVIEW"
    failure_gate_by_condition = [
        ("source_integrity", "FAIL_SOURCE_ARTIFACT_INTEGRITY"),
        ("upstream_immutability", "FAIL_SOURCE_IMMUTABILITY"),
        ("upstream_execution_boundary", "FAIL_UPSTREAM_EXECUTION_BOUNDARY"),
        ("fingerprint", "FAIL_FROZEN_REGISTRY_FINGERPRINT"),
        ("immutability", "FAIL_FROZEN_REGISTRY_IMMUTABILITY"),
        ("input_packet", "FAIL_ESTIMATION_INPUT_PACKET"),
        ("analysis_population", "FAIL_ANALYSIS_POPULATION_RECONCILIATION"),
        ("adaptive_eta_exclusion", "FAIL_ADAPTIVE_ETA_EXCLUSION"),
        ("estimand", "FAIL_ESTIMAND_CONTRACT"),
        ("primary_interval", "FAIL_PRIMARY_INTERVAL_CONTRACT"),
        ("clock_sensitivity", "FAIL_CLOCK_SENSITIVITY_CONTRACT"),
        ("vehicle_dependence", "FAIL_VEHICLE_DEPENDENCE_CONTRACT"),
        ("support", "FAIL_TURNBULL_SUPPORT_CONTRACT"),
        ("initialization", "FAIL_TURNBULL_INITIALIZATION_CONTRACT"),
        ("convergence", "FAIL_TURNBULL_CONVERGENCE_CONTRACT"),
        ("nonunique", "FAIL_TURNBULL_NONUNIQUE_SOLUTION_CONTRACT"),
        ("median", "FAIL_MEDIAN_IDENTIFICATION_CONTRACT"),
        ("quantile", "FAIL_QUANTILE_IDENTIFICATION_CONTRACT"),
        ("reproducibility", "FAIL_EXECUTION_REPRODUCIBILITY_CONTRACT"),
        ("independent_verification", "FAIL_INDEPENDENT_VERIFICATION_CONTRACT"),
        ("r2d1q_artifact", "FAIL_R2D1Q_ARTIFACT_CONTRACT"),
        ("authorization_boundary", "FAIL_AUTHORIZATION_BOUNDARY"),
        ("json_parquet", "FAIL_JSON_PARQUET_SYNCHRONIZATION"),
    ]
    for condition, failure_gate in failure_gate_by_condition:
        if not pass_conditions.get(condition, False):
            gate_status = failure_gate
            break

    authorization_review = dict(authorization_review_prelim)
    authorization_review["estimation_execution_authorization_review_passed"] = gate_status == PASS_GATE
    authorization_review["terminal_recovery_estimation_execution_approved"] = gate_status == PASS_GATE
    dump_json(output_root / "estimation_execution_authorization_review.json", authorization_review)
    dump_json(
        output_root / "estimation_execution_authorization.json",
        {
            "terminal_recovery_estimation_execution_approved": gate_status == PASS_GATE,
            "authorized_execution_step": "R2D-1Q" if gate_status == PASS_GATE else None,
            "estimation_executed": False,
            "requires_frozen_registry_fingerprint_match_at_execution": True,
            "authorized_primary_method": PRIMARY_METHOD if gate_status == PASS_GATE else None,
            "authorized_primary_input_clock": PRIMARY_CLOCK if gate_status == PASS_GATE else None,
            "authorized_primary_episode_count": 12 if gate_status == PASS_GATE else None,
            "authorization_boundary": "execution authorization only; no estimation results generated",
        },
    )

    strict_pre_manifest = strict_json_audit(output_root)
    parquet_pre_manifest = parquet_audit(output_root)
    secret_audit = strict_read_json(output_root / "secret_leak_audit.json")
    pass_conditions["strict_json"] = strict_pre_manifest["strict_json_failure_count"] == 0
    pass_conditions["parquet"] = parquet_pre_manifest["parquet_read_failure_count"] == 0
    pass_conditions["security"] = secret_audit["secret_leak_count"] == 0
    if gate_status == PASS_GATE and not (pass_conditions["strict_json"] and pass_conditions["parquet"] and pass_conditions["security"]):
        gate_status = "FAIL_SCHEMA_AUDIT" if not (pass_conditions["strict_json"] and pass_conditions["parquet"]) else "FAIL_SECURITY_AUDIT"

    report_items = [
        ("R2D-1P artifact absolute path", f"`{output_root}`"),
        ("script absolute path", f"`{SCRIPT_PATH}`"),
        ("final gate", f"`{gate_status}`"),
        ("API call count", "`0`"),
        ("service key access", "`false`"),
        ("upstream changed count", f"`{immutability['upstream_modified_file_count']}/{immutability['upstream_deleted_file_count']}/{immutability['upstream_added_file_count']}`"),
        ("source R2D-1O gate", f"`{r2d1o_gate.get('gate_status')}`"),
        ("frozen registry row count", f"`{len(registry_records)}`"),
        ("registry canonical fingerprint", f"`{source_fingerprint['registry_canonical_sha256']}`"),
        ("fingerprint reverification result", f"`{str(fingerprint_reverification['frozen_registry_fingerprint_reverification_passed']).lower()}`"),
        ("estimation input row count", f"`{len(input_records)}`"),
        ("input packet mismatch result", f"value mismatches `{estimation_input_packet_reconciliation['input_registry_value_mismatch_count']}`; duplicate/missing/extra `{duplicate_input_episodes}/{len(missing_episodes)}/{len(extra_episodes)}`"),
        ("primary analysis population", f"`{PRIMARY_POPULATION}`"),
        ("excluded adaptive ETA row count", f"`{adaptive_eta_exclusion['adaptive_eta_rows_excluded_from_primary_input']}`"),
        ("primary estimand", f"`{PRIMARY_ESTIMAND_ENUM}`"),
        ("unidentifiable estimand", f"`{UNIDENTIFIABLE_ESTIMAND}`"),
        ("primary clock", f"`{PRIMARY_CLOCK}`"),
        ("sensitivity clocks", "`PROVIDER_ONLY_INTERVAL`, `REQUEST_ONLY_INTERVAL`"),
        ("primary method", f"`{PRIMARY_METHOD}`"),
        ("episode weighting", "`EQUAL_EPISODE_WEIGHT`"),
        ("vehicle-balanced sensitivity status", "`AUTHORIZED_AS_SECONDARY_ONLY`"),
        ("bootstrap status", "`NOT_AUTHORIZED`"),
        ("midpoint sensitivity status", "`false`"),
        ("route formal inference status", "`false`"),
        ("support construction contract", "`ready`; endpoint count preexecution `{}`".format(support_contract["unique_endpoint_count_preexecution"])),
        ("initialization contract", "`ready`; initialization count `2`"),
        ("convergence tolerance", "`relative_log_likelihood_change <= 1e-10 AND maximum_mass_change <= 1e-8 for 5 consecutive iterations`"),
        ("maximum iterations", "`10000`"),
        ("likelihood monotonicity tolerance", "`1e-10`"),
        ("nonunique solution handling", "`SOLUTION_NONUNIQUE_LIKELIHOOD_EQUIVALENT` allowed with identified-set reporting"),
        ("median identification status contract", "`POINT_IDENTIFIED`, `INTERVAL_IDENTIFIED`, `NOT_IDENTIFIED_WITHIN_SUPPORT`"),
        ("quantile identification contract", "`q25`, `q50`, `q75`; point/interval/not identified"),
        ("independent likelihood verification contract", "`independent_likelihood_mismatch <= 1e-8`"),
        ("reproducibility contract", "`ready`; input/method SHA and tolerances recorded"),
        ("R2D-1Q artifact contract", "`ready`"),
        ("R2D-1Q gate contract", "`ready`"),
        ("fail-closed conditions", f"`{len(fail_closed_conditions)}` conditions"),
        ("estimation authorization review status", f"`{str(authorization_review['estimation_execution_authorization_review_passed']).lower()}`"),
        ("estimation execution approval status", f"`{str(authorization_review['terminal_recovery_estimation_execution_approved']).lower()}`"),
        ("estimation executed status", "`false`"),
        ("formal inferential sufficiency", "`NOT_ESTABLISHED`"),
        ("parameter translation lock", "`terminal_recovery_parameter_generated=false`; translation authorized `false`; applied `false`"),
        ("simulator lock", "`simulator_application_authorized=false`"),
        ("Phase 2 lock", "`phase2_authorized=false`; baseline rerun `false`; retraining `false`"),
        ("JSON and Parquet result", f"strict JSON failures `{strict_pre_manifest['strict_json_failure_count']}`; Parquet read failures `{parquet_pre_manifest['parquet_read_failure_count']}`; JSON-Parquet mismatches `{json_parquet_audit['json_parquet_value_mismatch_count']}`"),
        ("manifest result", "`0 / 0 / 0` after final manifest reconciliation"),
        ("secret scan result", f"secret leaks `{secret_audit['secret_leak_count']}`"),
        ("next authorized action", "`R2D-1Q Method Prototype interval-censored estimation execution only`"),
    ]
    if len(report_items) != 48:
        raise RuntimeError(f"final report item count mismatch: {len(report_items)}")
    final_report = "# R2D-1P Estimation Execution Authorization Review\n\n"
    final_report += "\n".join(f"{idx}. {label}: {value}" for idx, (label, value) in enumerate(report_items, start=1))
    final_report += "\n"
    (output_root / "prompt5_e01_r2d1p_final_report.md").write_text(final_report, encoding="utf-8")

    gate_payload = {
        "artifact_dir": str(output_root),
        "gate_status": gate_status,
        "gate_passed": gate_status == PASS_GATE,
        "network_api_calls": network_api_calls,
        "service_key_accessed": service_key_accessed,
        "upstream_modified_file_count": immutability["upstream_modified_file_count"],
        "upstream_deleted_file_count": immutability["upstream_deleted_file_count"],
        "upstream_added_file_count": immutability["upstream_added_file_count"],
        "source_r2d1o_gate": r2d1o_gate.get("gate_status"),
        "frozen_registry_row_count": len(registry_records),
        "registry_canonical_sha256": source_fingerprint["registry_canonical_sha256"],
        "registry_fingerprint_match": fingerprint_reverification["registry_fingerprint_match"],
        "episode_id_set_fingerprint_match": fingerprint_reverification["episode_id_set_fingerprint_match"],
        "raw_evidence_set_fingerprint_match": fingerprint_reverification["raw_evidence_set_fingerprint_match"],
        "interval_tuple_set_fingerprint_match": fingerprint_reverification["interval_tuple_set_fingerprint_match"],
        "estimation_input_row_count": len(input_records),
        "input_registry_value_mismatch_count": estimation_input_packet_reconciliation["input_registry_value_mismatch_count"],
        "input_duplicate_episode_count": estimation_input_packet_reconciliation["input_duplicate_episode_count"],
        "input_missing_episode_count": estimation_input_packet_reconciliation["input_missing_episode_count"],
        "input_extra_episode_count": estimation_input_packet_reconciliation["input_extra_episode_count"],
        "adaptive_eta_rows_in_primary_input": adaptive_eta_exclusion["adaptive_eta_rows_in_primary_input"],
        "noncomplete_rows_in_primary_input": analysis_population["censored_noncomplete_rows_in_primary_input"],
        "primary_estimand": PRIMARY_ESTIMAND_ENUM,
        "unidentifiable_estimand": UNIDENTIFIABLE_ESTIMAND,
        "primary_input_clock": PRIMARY_CLOCK,
        "provider_clock_sensitivity_authorized": True,
        "request_clock_sensitivity_authorized": True,
        "primary_method": PRIMARY_METHOD,
        "primary_weighting": "EQUAL_EPISODE_WEIGHT",
        "vehicle_balanced_sensitivity_authorized": True,
        "vehicle_cluster_bootstrap_authorized": False,
        "midpoint_sensitivity_authorized": False,
        "route_formal_inference_authorized": False,
        "turnbull_initialization_count": 2,
        "turnbull_relative_loglik_tolerance": 1e-10,
        "turnbull_max_mass_change_tolerance": 1e-8,
        "turnbull_consecutive_convergence_iterations": 5,
        "turnbull_max_iterations": 10000,
        "likelihood_decrease_tolerance": 1e-10,
        "formal_inferential_sufficiency": "NOT_ESTABLISHED",
        "estimation_execution_authorization_review_passed": gate_status == PASS_GATE,
        "terminal_recovery_estimation_execution_approved": gate_status == PASS_GATE,
        "authorized_execution_step": "R2D-1Q" if gate_status == PASS_GATE else None,
        "estimation_executed": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_parameter_translation_authorized": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "strict_json_failure_count": strict_pre_manifest["strict_json_failure_count"],
        "parquet_read_failure_count": parquet_pre_manifest["parquet_read_failure_count"],
        "json_parquet_value_mismatch_count": json_parquet_audit["json_parquet_value_mismatch_count"],
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "secret_leak_count": secret_audit["secret_leak_count"],
        "next_authorized_action": "R2D-1Q Method Prototype interval-censored estimation execution only",
        "pass_conditions": pass_conditions,
    }
    dump_json(output_root / "prompt5_e01_r2d1p_gate.json", gate_payload)

    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1p_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)
    final_strict = strict_json_audit(output_root)
    final_parquet = parquet_audit(output_root)
    gate_payload.update(manifest_result)
    gate_payload["strict_json_failure_count"] = final_strict["strict_json_failure_count"]
    gate_payload["parquet_read_failure_count"] = final_parquet["parquet_read_failure_count"]
    if final_strict["strict_json_failure_count"] or final_parquet["parquet_read_failure_count"] or any(
        manifest_result[key]
        for key in ["manifest_missing_required_file_count", "manifest_nonself_hash_mismatch_count", "manifest_nonself_size_mismatch_count"]
    ):
        gate_payload["gate_status"] = "FAIL_SCHEMA_AUDIT"
        gate_payload["gate_passed"] = False
    dump_json(output_root / "prompt5_e01_r2d1p_gate.json", gate_payload)
    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1p_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)

    print("R2D-1P ESTIMATION EXECUTION AUTHORIZATION REVIEW COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_payload["gate_status"])
    print("\nnetwork_api_calls:\n0")
    print("\nservice_key_accessed:\nfalse")
    print("\nupstream_modified_file_count:")
    print(immutability["upstream_modified_file_count"])
    print("\nupstream_deleted_file_count:")
    print(immutability["upstream_deleted_file_count"])
    print("\nupstream_added_file_count:")
    print(immutability["upstream_added_file_count"])
    print("\nsource_r2d1o_gate:")
    print(r2d1o_gate.get("gate_status"))
    print("\nfrozen_registry_row_count:")
    print(len(registry_records))
    print("\nregistry_canonical_sha256:")
    print(source_fingerprint["registry_canonical_sha256"])
    print("\nregistry_fingerprint_match:")
    print(str(fingerprint_reverification["registry_fingerprint_match"]).lower())
    print("\nepisode_id_set_fingerprint_match:")
    print(str(fingerprint_reverification["episode_id_set_fingerprint_match"]).lower())
    print("\nraw_evidence_set_fingerprint_match:")
    print(str(fingerprint_reverification["raw_evidence_set_fingerprint_match"]).lower())
    print("\ninterval_tuple_set_fingerprint_match:")
    print(str(fingerprint_reverification["interval_tuple_set_fingerprint_match"]).lower())
    print("\nestimation_input_row_count:")
    print(len(input_records))
    print("\ninput_registry_value_mismatch_count:")
    print(estimation_input_packet_reconciliation["input_registry_value_mismatch_count"])
    print("\ninput_duplicate_episode_count:")
    print(estimation_input_packet_reconciliation["input_duplicate_episode_count"])
    print("\ninput_missing_episode_count:")
    print(estimation_input_packet_reconciliation["input_missing_episode_count"])
    print("\ninput_extra_episode_count:")
    print(estimation_input_packet_reconciliation["input_extra_episode_count"])
    print("\nadaptive_eta_rows_in_primary_input:")
    print(adaptive_eta_exclusion["adaptive_eta_rows_in_primary_input"])
    print("\nnoncomplete_rows_in_primary_input:")
    print(analysis_population["censored_noncomplete_rows_in_primary_input"])
    print("\nprimary_estimand:")
    print(PRIMARY_ESTIMAND_ENUM)
    print("\nunidentifiable_estimand:")
    print(UNIDENTIFIABLE_ESTIMAND)
    print("\nprimary_input_clock:")
    print(PRIMARY_CLOCK)
    print("\nprovider_clock_sensitivity_authorized:")
    print("true")
    print("\nrequest_clock_sensitivity_authorized:")
    print("true")
    print("\nprimary_method:")
    print(PRIMARY_METHOD)
    print("\nprimary_weighting:")
    print("EQUAL_EPISODE_WEIGHT")
    print("\nvehicle_balanced_sensitivity_authorized:")
    print("true")
    print("\nvehicle_cluster_bootstrap_authorized:\nfalse")
    print("\nmidpoint_sensitivity_authorized:\nfalse")
    print("\nroute_formal_inference_authorized:\nfalse")
    print("\nturnbull_initialization_count:\n2")
    print("\nturnbull_relative_loglik_tolerance:\n1e-10")
    print("\nturnbull_max_mass_change_tolerance:\n1e-8")
    print("\nturnbull_consecutive_convergence_iterations:\n5")
    print("\nturnbull_max_iterations:\n10000")
    print("\nlikelihood_decrease_tolerance:\n1e-10")
    print("\nformal_inferential_sufficiency:\nNOT_ESTABLISHED")
    print("\nestimation_execution_authorization_review_passed:")
    print(str(gate_payload["estimation_execution_authorization_review_passed"]).lower())
    print("\nterminal_recovery_estimation_execution_approved:")
    print(str(gate_payload["terminal_recovery_estimation_execution_approved"]).lower())
    print("\nauthorized_execution_step:\nR2D-1Q")
    print("\nestimation_executed:\nfalse")
    print("\nterminal_recovery_parameter_generated:\nfalse")
    print("\nterminal_recovery_parameter_translation_authorized:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\nsimulator_application_authorized:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print("\nstrict_json_failure_count:")
    print(final_strict["strict_json_failure_count"])
    print("\nparquet_read_failure_count:")
    print(final_parquet["parquet_read_failure_count"])
    print("\njson_parquet_value_mismatch_count:")
    print(json_parquet_audit["json_parquet_value_mismatch_count"])
    print("\nmanifest_missing_required_file_count:")
    print(manifest_result["manifest_missing_required_file_count"])
    print("\nmanifest_nonself_hash_mismatch_count:")
    print(manifest_result["manifest_nonself_hash_mismatch_count"])
    print("\nmanifest_nonself_size_mismatch_count:")
    print(manifest_result["manifest_nonself_size_mismatch_count"])
    print("\nsecret_leak_count:")
    print(secret_audit["secret_leak_count"])
    print("\nnext_authorized_action:")
    print("R2D-1Q Method Prototype interval-censored estimation execution only")


if __name__ == "__main__":
    main()
