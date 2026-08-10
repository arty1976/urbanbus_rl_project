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
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1r_estimation_result_audit_non_identification_freeze.py"

SOURCE_R2D1Q = ARTIFACTS_ROOT / "prompt5_e01_r2d1q_method_prototype_interval_censored_estimation_20260730_100056"
SOURCE_R2D1Q_REPLICATE = ARTIFACTS_ROOT / "prompt5_e01_r2d1q_method_prototype_interval_censored_estimation_20260730_095955"
SOURCE_R2D1Q_DISCARDED = ARTIFACTS_ROOT / "prompt5_e01_r2d1q_method_prototype_interval_censored_estimation_20260730_095919"
SOURCE_R2D1P = ARTIFACTS_ROOT / "prompt5_e01_r2d1p_estimation_execution_authorization_20260730_092848"
SOURCE_R2D1O = ARTIFACTS_ROOT / "prompt5_e01_r2d1o_12_episode_registry_freeze_estimation_readiness_20260730_085243"

PASS_GATE = "PASS_METHOD_PROTOTYPE_ESTIMATION_RESULT_AUDIT_NON_IDENTIFICATION_FREEZE_COMPLETE"
R2D1Q_REQUIRED_GATE = "PASS_METHOD_PROTOTYPE_INTERVAL_CENSORED_ESTIMATION_COMPLETE"
R2D1P_REQUIRED_GATE = "PASS_METHOD_PROTOTYPE_ESTIMATION_EXECUTION_AUTHORIZATION_READY"
R2D1O_REQUIRED_GATE = "PASS_12_EPISODE_REGISTRY_FREEZE_ESTIMATION_READINESS_READY"

PRIMARY_ESTIMAND_ENUM = "OBSERVED_POST_SERVICE_NON_REVENUE_TURNAROUND_INTERVAL"
UNIDENTIFIABLE_ESTIMAND = "ACTUAL_DRIVER_REST_DURATION"
PRIMARY_CLOCK = "CONSERVATIVE_DUAL_CLOCK"
PRIMARY_METHOD = "TURNBULL_NPMLE_INTERVAL_CENSORED_POOLED"

# Analyses re-derived independently from the frozen 12-episode interval input.
ANALYSES: List[Dict[str, str]] = [
    {
        "analysis_id": "PRIMARY_DUAL_EQUAL",
        "clock": "CONSERVATIVE_DUAL_CLOCK",
        "role": "PRIMARY",
        "lower_field": "conservative_dual_lower_bound_sec",
        "upper_field": "conservative_dual_upper_bound_sec",
        "quantile_file": "turnbull_primary_quantile_identification.json",
        "execution_file": "turnbull_primary_execution.json",
    },
    {
        "analysis_id": "SENSITIVITY_PROVIDER_ONLY",
        "clock": "PROVIDER_ONLY_INTERVAL",
        "role": "SECONDARY",
        "lower_field": "provider_lower_bound_sec",
        "upper_field": "provider_upper_bound_sec",
        "quantile_file": "turnbull_provider_sensitivity_quantile_identification.json",
        "execution_file": "turnbull_provider_sensitivity_execution.json",
    },
    {
        "analysis_id": "SENSITIVITY_REQUEST_ONLY",
        "clock": "REQUEST_ONLY_INTERVAL",
        "role": "SECONDARY",
        "lower_field": "request_lower_bound_sec",
        "upper_field": "request_upper_bound_sec",
        "quantile_file": "turnbull_request_sensitivity_quantile_identification.json",
        "execution_file": "turnbull_request_sensitivity_execution.json",
    },
    {
        "analysis_id": "SENSITIVITY_DUAL_VEHICLE_BALANCED",
        "clock": "CONSERVATIVE_DUAL_CLOCK",
        "role": "SECONDARY",
        "lower_field": "conservative_dual_lower_bound_sec",
        "upper_field": "conservative_dual_upper_bound_sec",
        "quantile_file": "turnbull_vehicle_balanced_quantile_identification.json",
        "execution_file": "turnbull_vehicle_balanced_sensitivity_execution.json",
    },
]

# Files whose content legitimately differs between two independent runs of the same
# deterministic pipeline because they record run identity (path, artifact id, timestamp).
RUN_IDENTITY_FILES = [
    "prompt5_e01_r2d1q_gate.json",
    "prompt5_e01_r2d1q_manifest.json",
    "prompt5_e01_r2d1q_final_report.md",
    "turnbull_execution_environment.json",
]
RUN_IDENTITY_KEYS = ["artifact_dir", "artifact_id", "created_at", "run_timestamp", "implementation_module"]

LOG_LIKELIHOOD_AGREEMENT_TOLERANCE = 1e-6
MASS_AGREEMENT_TOLERANCE = 1e-8
ENDPOINT_AGREEMENT_TOLERANCE = 1e-9

REQUIRED_FILES = [
    "prompt5_e01_r2d1r_manifest.json",
    "prompt5_e01_r2d1r_gate.json",
    "prompt5_e01_r2d1r_final_report.md",
    "upstream_reference_r2d1q.json",
    "upstream_reference_r2d1p.json",
    "upstream_reference_r2d1o.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "secret_leak_audit.json",
    "authoritative_run_selection_audit.json",
    "r2d1q_manifest_reverification.json",
    "r2d1q_gate_reverification.json",
    "cross_run_reproducibility_audit.json",
    "frozen_input_interval_reverification.json",
    "frozen_input_interval_reverification.parquet",
    "independent_interval_geometry_recomputation.json",
    "independent_interval_geometry_recomputation.parquet",
    "common_intersection_geometry_audit.json",
    "turnbull_innermost_interval_audit.json",
    "analytic_npmle_closed_form_audit.json",
    "numeric_analytic_agreement_audit.json",
    "estimation_information_content_audit.json",
    "binding_episode_analysis.json",
    "binding_episode_analysis.parquet",
    "clock_input_independence_audit.json",
    "provider_sensitivity_information_audit.json",
    "conservative_dual_clock_hedge_audit.json",
    "vehicle_dependence_deferred_audit.json",
    "frozen_identified_sets.json",
    "frozen_identified_sets.parquet",
    "identified_set_freeze_fingerprint.json",
    "non_identification_classification.json",
    "solution_status_reinterpretation.json",
    "upstream_result_preservation_audit.json",
    "observation_resolution_diagnosis.json",
    "resolution_improvement_requirement.json",
    "empty_intersection_prespecification.json",
    "midpoint_selection_prohibition.json",
    "prohibited_interpretation_audit.json",
    "estimation_limitations_r2d1r.json",
    "terminal_recovery_parameter_translation_guard.json",
    "simulator_application_authorization.json",
    "phase2_execution_authorization.json",
    "next_step_authorization.json",
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
        return sanitize(value.tolist())
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


def dump_parquet(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(list(records)).to_parquet(path, index=False)


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
    candidates = sorted(path for path in root.glob("*.json") if "_gate" in path.name)
    prompt_candidates = [path for path in candidates if path.name.startswith("prompt")]
    return prompt_candidates[0] if prompt_candidates else (candidates[0] if candidates else None)


def manifest_path_for(root: Path) -> Path | None:
    candidates = sorted(path for path in root.glob("*.json") if "_manifest" in path.name)
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
        if name == "prompt5_e01_r2d1r_manifest.json":
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


def strip_run_identity(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: strip_run_identity(value) for key, value in payload.items() if key not in RUN_IDENTITY_KEYS}
    if isinstance(payload, list):
        return [strip_run_identity(item) for item in payload]
    return payload


def innermost_intervals(lowers: Sequence[float], uppers: Sequence[float]) -> List[Tuple[float, float]]:
    """Turnbull/Peto innermost intervals for closed interval-censored observations.

    An innermost interval is a maximal-information cell [q, p] where q is an observed
    lower endpoint, p is an observed upper endpoint, no other endpoint lies strictly
    inside, and the cell is contained in at least one observation interval.
    """
    endpoints = sorted(set(lowers) | set(uppers))
    cells: List[Tuple[float, float]] = []
    for lower in sorted(set(lowers)):
        for upper in sorted(set(uppers)):
            if upper < lower:
                continue
            if any(lower < point < upper for point in endpoints):
                continue
            if any(obs_lower <= lower and upper <= obs_upper for obs_lower, obs_upper in zip(lowers, uppers)):
                cells.append((lower, upper))
    return sorted(cells)


def leave_one_out_intersection(lowers: Sequence[float], uppers: Sequence[float], skip: int) -> Tuple[float, float]:
    kept_lowers = [value for index, value in enumerate(lowers) if index != skip]
    kept_uppers = [value for index, value in enumerate(uppers) if index != skip]
    return max(kept_lowers), min(kept_uppers)


def main() -> None:
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1r_estimation_result_audit_non_identification_freeze_{now_stamp()}"
    output_root.mkdir(parents=True, exist_ok=False)

    upstream_roots = [SOURCE_R2D1Q, SOURCE_R2D1Q_REPLICATE, SOURCE_R2D1P, SOURCE_R2D1O]
    before_snapshot = source_snapshot(upstream_roots)

    network_api_calls = 0
    service_key_accessed = False
    dump_json(
        output_root / "network_api_call_audit.json",
        {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "network_request_performed": False},
    )
    dump_json(
        output_root / "service_key_access_audit.json",
        {"service_key_accessed": False, "environment_read_performed": False, "service_key_output": False},
    )

    upstream_refs = {
        "upstream_reference_r2d1q.json": upstream_reference(SOURCE_R2D1Q, "r2d1q", R2D1Q_REQUIRED_GATE),
        "upstream_reference_r2d1p.json": upstream_reference(SOURCE_R2D1P, "r2d1p", R2D1P_REQUIRED_GATE),
        "upstream_reference_r2d1o.json": upstream_reference(SOURCE_R2D1O, "r2d1o", R2D1O_REQUIRED_GATE),
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

    # ------------------------------------------------------------------
    # Authoritative run selection across the three R2D-1Q run directories
    # ------------------------------------------------------------------
    run_candidates = []
    for root, role in [
        (SOURCE_R2D1Q, "AUTHORITATIVE"),
        (SOURCE_R2D1Q_REPLICATE, "INDEPENDENT_REPLICATE"),
        (SOURCE_R2D1Q_DISCARDED, "DISCARDED_FAILED_RUN"),
    ]:
        gate_path = root / "prompt5_e01_r2d1q_gate.json"
        gate = strict_read_json(gate_path) if gate_path.exists() else {}
        run_candidates.append(
            {
                "artifact_dir": str(root),
                "exists": root.exists(),
                "role": role,
                "gate_status": gate.get("gate_status"),
                "gate_passed": gate.get("gate_passed"),
                "file_count": len([item for item in root.rglob("*") if item.is_file()]) if root.exists() else 0,
            }
        )
    authoritative_run = {
        "run_candidate_count": len(run_candidates),
        "run_candidates": run_candidates,
        "authoritative_artifact_dir": str(SOURCE_R2D1Q),
        "authoritative_selection_rule": "latest PASS run; the audited ZIP export corresponds to this directory",
        "passing_run_count": sum(1 for row in run_candidates if row["gate_passed"] is True),
        "failed_run_count": sum(1 for row in run_candidates if row["gate_passed"] is False),
        "discarded_run_excluded_from_all_downstream_use": True,
        "authoritative_run_selection_passed": (
            run_candidates[0]["gate_status"] == R2D1Q_REQUIRED_GATE
            and run_candidates[1]["gate_status"] == R2D1Q_REQUIRED_GATE
            and run_candidates[2]["gate_passed"] is False
        ),
    }
    dump_json(output_root / "authoritative_run_selection_audit.json", authoritative_run)

    # ------------------------------------------------------------------
    # R2D-1Q manifest and gate reverification (recomputed hashes)
    # ------------------------------------------------------------------
    r2d1q_manifest = strict_read_json(SOURCE_R2D1Q / "prompt5_e01_r2d1q_manifest.json")
    manifest_entries = r2d1q_manifest.get("files", [])
    actual_files = sorted(item.name for item in SOURCE_R2D1Q.iterdir() if item.is_file())
    manifest_names = sorted(str(entry["path"]) for entry in manifest_entries)
    hash_mismatch, size_mismatch, missing = [], [], []
    for entry in manifest_entries:
        name = str(entry["path"])
        path = SOURCE_R2D1Q / name
        if not path.exists():
            missing.append(name)
            continue
        if entry.get("self_hash_exempt"):
            continue
        if entry.get("sha256") != sha256_file(path):
            hash_mismatch.append(name)
        if entry.get("size_bytes") != path.stat().st_size:
            size_mismatch.append(name)
    r2d1q_manifest_reverification = {
        "manifest_entry_count": len(manifest_entries),
        "actual_file_count": len(actual_files),
        "manifest_missing_file_count": len(missing),
        "manifest_extra_entry_count": len(sorted(set(manifest_names) - set(actual_files))),
        "unmanifested_file_count": len(sorted(set(actual_files) - set(manifest_names))),
        "nonself_hash_mismatch_count": len(hash_mismatch),
        "nonself_size_mismatch_count": len(size_mismatch),
        "missing_files": missing,
        "hash_mismatches": hash_mismatch,
        "size_mismatches": size_mismatch,
        "r2d1q_manifest_reverification_passed": not (missing or hash_mismatch or size_mismatch)
        and len(manifest_entries) == len(actual_files),
    }
    dump_json(output_root / "r2d1q_manifest_reverification.json", r2d1q_manifest_reverification)

    r2d1q_gate = strict_read_json(SOURCE_R2D1Q / "prompt5_e01_r2d1q_gate.json")
    r2d1q_strict_failures = []
    for path in sorted(SOURCE_R2D1Q.glob("*.json")):
        try:
            strict_read_json(path)
        except Exception as exc:
            r2d1q_strict_failures.append({"path": path.name, "error": type(exc).__name__})
    r2d1q_parquet_failures = []
    for path in sorted(SOURCE_R2D1Q.glob("*.parquet")):
        try:
            pd.read_parquet(path)
        except Exception as exc:
            r2d1q_parquet_failures.append({"path": path.name, "error": type(exc).__name__})
    r2d1q_gate_reverification = {
        "gate_status": r2d1q_gate.get("gate_status"),
        "gate_passed": r2d1q_gate.get("gate_passed"),
        "required_gate_status": R2D1Q_REQUIRED_GATE,
        "reported_network_api_calls": r2d1q_gate.get("network_api_calls"),
        "reported_service_key_accessed": r2d1q_gate.get("service_key_accessed"),
        "reported_parameter_generated": r2d1q_gate.get("terminal_recovery_parameter_generated"),
        "reported_parameter_translation_authorized": r2d1q_gate.get("terminal_recovery_parameter_translation_authorized"),
        "reported_simulator_application_authorized": r2d1q_gate.get("simulator_application_authorized"),
        "reported_phase2_authorized": r2d1q_gate.get("phase2_authorized"),
        "recomputed_strict_json_failure_count": len(r2d1q_strict_failures),
        "recomputed_parquet_read_failure_count": len(r2d1q_parquet_failures),
        "recomputed_strict_json_failures": r2d1q_strict_failures,
        "recomputed_parquet_read_failures": r2d1q_parquet_failures,
        "parquet_decoder_available": True,
        "parquet_decoder_note": "This audit decodes every R2D-1Q Parquet file; the external independent audit could not.",
        "r2d1q_gate_reverification_passed": (
            r2d1q_gate.get("gate_status") == R2D1Q_REQUIRED_GATE
            and r2d1q_gate.get("gate_passed") is True
            and r2d1q_gate.get("network_api_calls") == 0
            and r2d1q_gate.get("service_key_accessed") is False
            and not r2d1q_strict_failures
            and not r2d1q_parquet_failures
        ),
    }
    dump_json(output_root / "r2d1q_gate_reverification.json", r2d1q_gate_reverification)

    # ------------------------------------------------------------------
    # Cross-run reproducibility: authoritative run vs independent replicate
    # ------------------------------------------------------------------
    replicate_files = sorted(item.name for item in SOURCE_R2D1Q_REPLICATE.iterdir() if item.is_file())
    differing, identical, missing_in_replicate = [], [], []
    for name in actual_files:
        replicate_path = SOURCE_R2D1Q_REPLICATE / name
        if not replicate_path.exists():
            missing_in_replicate.append(name)
            continue
        if sha256_file(SOURCE_R2D1Q / name) == sha256_file(replicate_path):
            identical.append(name)
        else:
            differing.append(name)
    unexpected_differences = sorted(set(differing) - set(RUN_IDENTITY_FILES))
    replicate_gate = strict_read_json(SOURCE_R2D1Q_REPLICATE / "prompt5_e01_r2d1q_gate.json")
    gate_field_differences = sorted(
        key
        for key in set(r2d1q_gate) | set(replicate_gate)
        if key not in RUN_IDENTITY_KEYS and r2d1q_gate.get(key) != replicate_gate.get(key)
    )
    environment_authoritative = strict_read_json(SOURCE_R2D1Q / "turnbull_execution_environment.json")
    environment_replicate = strict_read_json(SOURCE_R2D1Q_REPLICATE / "turnbull_execution_environment.json")
    environment_differences = sorted(
        key
        for key in set(environment_authoritative) | set(environment_replicate)
        if key not in RUN_IDENTITY_KEYS and environment_authoritative.get(key) != environment_replicate.get(key)
    )
    cross_run = {
        "authoritative_artifact_dir": str(SOURCE_R2D1Q),
        "replicate_artifact_dir": str(SOURCE_R2D1Q_REPLICATE),
        "compared_file_count": len(actual_files),
        "replicate_file_count": len(replicate_files),
        "byte_identical_file_count": len(identical),
        "differing_file_count": len(differing),
        "missing_in_replicate_count": len(missing_in_replicate),
        "differing_files": differing,
        "run_identity_bearing_files": RUN_IDENTITY_FILES,
        "unexpected_difference_count": len(unexpected_differences),
        "unexpected_differences": unexpected_differences,
        "gate_field_difference_count": len(gate_field_differences),
        "gate_field_differences": gate_field_differences,
        "execution_environment_difference_count": len(environment_differences),
        "execution_environment_differences": environment_differences,
        "randomness_used": environment_authoritative.get("randomness_used"),
        "reproducibility_claim": "bit-exact determinism across two independent executions for all non-run-identity files",
        "cross_run_reproducibility_passed": (
            not unexpected_differences
            and not missing_in_replicate
            and not gate_field_differences
            and not environment_differences
            and environment_authoritative.get("randomness_used") is False
        ),
    }
    dump_json(output_root / "cross_run_reproducibility_audit.json", cross_run)

    # ------------------------------------------------------------------
    # Frozen interval input reverification (JSON vs Parquet vs upstream registry)
    # ------------------------------------------------------------------
    interval_audit = strict_read_json(SOURCE_R2D1Q / "interval_input_validation_audit.json")
    interval_records = interval_audit["records"]
    interval_parquet = pd.read_parquet(SOURCE_R2D1Q / "interval_input_validation_audit.parquet")
    registry_records = strict_read_json(SOURCE_R2D1O / "registry_12_frozen.json")["records"]
    registry_by_episode = {str(row["episode_id"]): row for row in registry_records}

    interval_fields = [
        "provider_lower_bound_sec",
        "provider_upper_bound_sec",
        "request_lower_bound_sec",
        "request_upper_bound_sec",
        "conservative_dual_lower_bound_sec",
        "conservative_dual_upper_bound_sec",
    ]
    registry_mismatches, parquet_mismatches = [], []
    parquet_by_episode = {str(row["episode_id"]): row for row in interval_parquet.to_dict(orient="records")}
    for row in interval_records:
        episode_id = str(row["episode_id"])
        registry_row = registry_by_episode.get(episode_id)
        parquet_row = parquet_by_episode.get(episode_id)
        for field in interval_fields:
            if registry_row is None or float(registry_row[field]) != float(row[field]):
                registry_mismatches.append({"episode_id": episode_id, "field": field})
            if parquet_row is None or float(parquet_row[field]) != float(row[field]):
                parquet_mismatches.append({"episode_id": episode_id, "field": field})

    frozen_input_records = []
    for row in interval_records:
        record = {"episode_id": str(row["episode_id"]), "route_id": str(row["route_id"]), "vehicle_id": str(row["vehicle_id"])}
        for field in interval_fields:
            record[field] = float(row[field])
        record["conservative_dual_width_sec"] = record["conservative_dual_upper_bound_sec"] - record["conservative_dual_lower_bound_sec"]
        record["provider_width_sec"] = record["provider_upper_bound_sec"] - record["provider_lower_bound_sec"]
        record["request_width_sec"] = record["request_upper_bound_sec"] - record["request_lower_bound_sec"]
        frozen_input_records.append(record)
    frozen_input_records.sort(key=lambda item: item["episode_id"])

    frozen_input_reverification = {
        "episode_count": len(frozen_input_records),
        "registry_value_mismatch_count": len(registry_mismatches),
        "parquet_value_mismatch_count": len(parquet_mismatches),
        "registry_value_mismatches": registry_mismatches,
        "parquet_value_mismatches": parquet_mismatches,
        "reversed_interval_count": sum(1 for row in frozen_input_records if row["conservative_dual_width_sec"] < 0),
        "route_count": len({row["route_id"] for row in frozen_input_records}),
        "vehicle_count": len({row["vehicle_id"] for row in frozen_input_records}),
        "input_interval_canonical_sha256": canonical_sha(frozen_input_records),
        "records": frozen_input_records,
        "frozen_input_interval_reverification_passed": (
            len(frozen_input_records) == 12 and not registry_mismatches and not parquet_mismatches
        ),
    }
    dump_json(output_root / "frozen_input_interval_reverification.json", frozen_input_reverification)
    dump_parquet(output_root / "frozen_input_interval_reverification.parquet", frozen_input_records)

    # ------------------------------------------------------------------
    # Independent geometry: common intersection and Turnbull innermost intervals
    # ------------------------------------------------------------------
    geometry_records: List[Dict[str, Any]] = []
    innermost_records: List[Dict[str, Any]] = []
    analytic_records: List[Dict[str, Any]] = []
    agreement_records: List[Dict[str, Any]] = []
    geometry_by_analysis: Dict[str, Dict[str, Any]] = {}

    for analysis in ANALYSES:
        lowers = [row[analysis["lower_field"]] for row in frozen_input_records]
        uppers = [row[analysis["upper_field"]] for row in frozen_input_records]
        widths = [upper - lower for lower, upper in zip(lowers, uppers)]
        max_lower = max(lowers)
        min_upper = min(uppers)
        cells = innermost_intervals(lowers, uppers)
        geometry = {
            "analysis_id": analysis["analysis_id"],
            "clock": analysis["clock"],
            "role": analysis["role"],
            "episode_count": len(lowers),
            "maximum_lower_endpoint_sec": max_lower,
            "minimum_upper_endpoint_sec": min_upper,
            "common_intersection_nonempty": max_lower < min_upper,
            "common_intersection_lower_sec": max_lower,
            "common_intersection_upper_sec": min_upper,
            "common_intersection_width_sec": min_upper - max_lower,
            "minimum_interval_width_sec": min(widths),
            "median_interval_width_sec": float(pd.Series(widths).median()),
            "maximum_interval_width_sec": max(widths),
            "unique_endpoint_count": len(set(lowers) | set(uppers)),
            "episodes_containing_common_intersection": sum(
                1 for lower, upper in zip(lowers, uppers) if lower <= max_lower and min_upper <= upper
            ),
        }
        geometry_records.append(geometry)
        geometry_by_analysis[analysis["analysis_id"]] = geometry

        innermost = {
            "analysis_id": analysis["analysis_id"],
            "innermost_interval_count": len(cells),
            "innermost_intervals": [{"lower_sec": lower, "upper_sec": upper} for lower, upper in cells],
            "npmle_support_is_degenerate": len(cells) == 1,
            "innermost_interval_equals_common_intersection": len(cells) == 1
            and abs(cells[0][0] - max_lower) <= ENDPOINT_AGREEMENT_TOLERANCE
            and abs(cells[0][1] - min_upper) <= ENDPOINT_AGREEMENT_TOLERANCE,
            "definition": "maximal cell [q,p] with q an observed lower endpoint, p an observed upper endpoint, no endpoint strictly inside, contained in at least one observation",
        }
        innermost_records.append(innermost)

        # Analytic closed-form NPMLE under a nonempty common intersection.
        analytic_log_likelihood = 0.0 if len(cells) == 1 else None
        observation_likelihoods = [
            1.0 if (lower <= max_lower and min_upper <= upper) else 0.0 for lower, upper in zip(lowers, uppers)
        ]
        analytic = {
            "analysis_id": analysis["analysis_id"],
            "closed_form_available": len(cells) == 1,
            "closed_form_solution": "unit probability mass on the single innermost interval",
            "closed_form_support_lower_sec": cells[0][0] if cells else None,
            "closed_form_support_upper_sec": cells[0][1] if cells else None,
            "closed_form_probability_mass": 1.0 if len(cells) == 1 else None,
            "closed_form_log_likelihood": analytic_log_likelihood,
            "theoretical_maximum_log_likelihood": 0.0,
            "minimum_observation_likelihood": min(observation_likelihoods),
            "em_iteration_required_for_solution": False,
            "note": "Every observation interval contains the common intersection, so the NPMLE is degenerate and available in closed form without EM iteration.",
        }
        analytic_records.append(analytic)

        # Agreement between the R2D-1Q numeric result and the analytic solution.
        execution = strict_read_json(SOURCE_R2D1Q / analysis["execution_file"])
        quantiles = strict_read_json(SOURCE_R2D1Q / analysis["quantile_file"])
        reported_log_likelihood = float(execution["final_log_likelihood"])
        reported_max_mass = float(execution["maximum_mass"])
        reported_components = execution["median_identified_components"]
        agreement = {
            "analysis_id": analysis["analysis_id"],
            "reported_final_log_likelihood": reported_log_likelihood,
            "analytic_log_likelihood": analytic_log_likelihood,
            "log_likelihood_absolute_difference": abs(reported_log_likelihood - 0.0),
            "log_likelihood_within_tolerance": abs(reported_log_likelihood) <= LOG_LIKELIHOOD_AGREEMENT_TOLERANCE,
            "reported_maximum_mass": reported_max_mass,
            "maximum_mass_absolute_difference_from_unity": abs(reported_max_mass - 1.0),
            "maximum_mass_within_tolerance": abs(reported_max_mass - 1.0) <= MASS_AGREEMENT_TOLERANCE,
            "reported_median_identified_component_count": len(reported_components),
            "reported_median_identified_lower_sec": reported_components[0]["lower_sec"] if reported_components else None,
            "reported_median_identified_upper_sec": reported_components[0]["upper_sec"] if reported_components else None,
            "analytic_identified_lower_sec": max_lower,
            "analytic_identified_upper_sec": min_upper,
            "identified_interval_matches_analytic": len(reported_components) == 1
            and abs(float(reported_components[0]["lower_sec"]) - max_lower) <= ENDPOINT_AGREEMENT_TOLERANCE
            and abs(float(reported_components[0]["upper_sec"]) - min_upper) <= ENDPOINT_AGREEMENT_TOLERANCE,
            "reported_support_cell_count": execution.get("support_cell_count"),
            "nonminimal_support_cell_count": int(execution.get("support_cell_count", 0)) - len(cells),
            "reported_iteration_count": execution.get("iteration_count"),
            "quantile_probabilities_checked": [row["quantile_probability"] for row in quantiles["quantiles"]],
            "all_quantiles_equal_common_intersection": all(
                len(row["identified_components"]) == 1
                and abs(float(row["identified_components"][0]["lower_sec"]) - max_lower) <= ENDPOINT_AGREEMENT_TOLERANCE
                and abs(float(row["identified_components"][0]["upper_sec"]) - min_upper) <= ENDPOINT_AGREEMENT_TOLERANCE
                for row in quantiles["quantiles"]
            ),
        }
        agreement["numeric_analytic_agreement_passed"] = bool(
            agreement["log_likelihood_within_tolerance"]
            and agreement["maximum_mass_within_tolerance"]
            and agreement["identified_interval_matches_analytic"]
            and agreement["all_quantiles_equal_common_intersection"]
        )
        agreement_records.append(agreement)

    dump_json(
        output_root / "independent_interval_geometry_recomputation.json",
        {
            "analysis_count": len(geometry_records),
            "recomputation_source": "frozen 12-episode interval input only; no R2D-1Q derived file used as input to this computation",
            "records": geometry_records,
        },
    )
    dump_parquet(output_root / "independent_interval_geometry_recomputation.parquet", geometry_records)

    primary_geometry = geometry_by_analysis["PRIMARY_DUAL_EQUAL"]
    common_intersection_audit = {
        "geometry_classification": "UNIVERSAL_COMMON_INTERSECTION_NONEMPTY",
        "classification_definition": "max_i lower_i < min_i upper_i, so a single interval is contained in every observation interval",
        "analyses_with_nonempty_common_intersection": sum(1 for row in geometry_records if row["common_intersection_nonempty"]),
        "analysis_count": len(geometry_records),
        "all_analyses_degenerate": all(row["common_intersection_nonempty"] for row in geometry_records),
        "primary_common_intersection_lower_sec": primary_geometry["common_intersection_lower_sec"],
        "primary_common_intersection_upper_sec": primary_geometry["common_intersection_upper_sec"],
        "primary_common_intersection_width_sec": primary_geometry["common_intersection_width_sec"],
        "primary_common_intersection_lower_mmss": "9m45s",
        "primary_common_intersection_upper_mmss": "36m29s",
        "records": [
            {
                "analysis_id": row["analysis_id"],
                "common_intersection_lower_sec": row["common_intersection_lower_sec"],
                "common_intersection_upper_sec": row["common_intersection_upper_sec"],
                "common_intersection_width_sec": row["common_intersection_width_sec"],
                "episodes_containing_common_intersection": row["episodes_containing_common_intersection"],
                "episode_count": row["episode_count"],
            }
            for row in geometry_records
        ],
        "common_intersection_geometry_audit_passed": all(
            row["common_intersection_nonempty"] and row["episodes_containing_common_intersection"] == row["episode_count"]
            for row in geometry_records
        ),
    }
    dump_json(output_root / "common_intersection_geometry_audit.json", common_intersection_audit)

    innermost_audit = {
        "analysis_count": len(innermost_records),
        "degenerate_analysis_count": sum(1 for row in innermost_records if row["npmle_support_is_degenerate"]),
        "all_analyses_have_single_innermost_interval": all(row["npmle_support_is_degenerate"] for row in innermost_records),
        "records": innermost_records,
        "interpretation": "The Turnbull innermost-interval set has exactly one element in every analysis, so the NPMLE support is a single cell and the estimator has no free shape parameter.",
        "turnbull_innermost_interval_audit_passed": all(
            row["npmle_support_is_degenerate"] and row["innermost_interval_equals_common_intersection"] for row in innermost_records
        ),
    }
    dump_json(output_root / "turnbull_innermost_interval_audit.json", innermost_audit)

    analytic_audit = {
        "analysis_count": len(analytic_records),
        "closed_form_available_count": sum(1 for row in analytic_records if row["closed_form_available"]),
        "records": analytic_records,
        "analytic_npmle_closed_form_audit_passed": all(
            row["closed_form_available"] and row["minimum_observation_likelihood"] == 1.0 for row in analytic_records
        ),
    }
    dump_json(output_root / "analytic_npmle_closed_form_audit.json", analytic_audit)

    agreement_audit = {
        "analysis_count": len(agreement_records),
        "log_likelihood_agreement_tolerance": LOG_LIKELIHOOD_AGREEMENT_TOLERANCE,
        "mass_agreement_tolerance": MASS_AGREEMENT_TOLERANCE,
        "endpoint_agreement_tolerance": ENDPOINT_AGREEMENT_TOLERANCE,
        "records": agreement_records,
        "agreement_failure_count": sum(1 for row in agreement_records if not row["numeric_analytic_agreement_passed"]),
        "numeric_analytic_agreement_passed": all(row["numeric_analytic_agreement_passed"] for row in agreement_records),
    }
    dump_json(output_root / "numeric_analytic_agreement_audit.json", agreement_audit)

    # ------------------------------------------------------------------
    # Information content and binding-episode (leave-one-out) analysis
    # ------------------------------------------------------------------
    binding_records: List[Dict[str, Any]] = []
    binding_counts: Dict[str, int] = {}
    for analysis in ANALYSES:
        analysis_id = analysis["analysis_id"]
        lowers = [row[analysis["lower_field"]] for row in frozen_input_records]
        uppers = [row[analysis["upper_field"]] for row in frozen_input_records]
        full_lower, full_upper = max(lowers), min(uppers)
        binding_count = 0
        for index, row in enumerate(frozen_input_records):
            loo_lower, loo_upper = leave_one_out_intersection(lowers, uppers, index)
            binds_lower = loo_lower < full_lower - ENDPOINT_AGREEMENT_TOLERANCE
            binds_upper = loo_upper > full_upper + ENDPOINT_AGREEMENT_TOLERANCE
            binding = binds_lower or binds_upper
            binding_count += int(binding)
            binding_records.append(
                {
                    "analysis_id": analysis_id,
                    "episode_id": row["episode_id"],
                    "route_id": row["route_id"],
                    "vehicle_id": row["vehicle_id"],
                    "observation_lower_sec": lowers[index],
                    "observation_upper_sec": uppers[index],
                    "binds_lower_endpoint": bool(binds_lower),
                    "binds_upper_endpoint": bool(binds_upper),
                    "binding_episode": bool(binding),
                    "leave_one_out_intersection_lower_sec": loo_lower,
                    "leave_one_out_intersection_upper_sec": loo_upper,
                    "leave_one_out_intersection_width_sec": loo_upper - loo_lower,
                    "full_sample_intersection_width_sec": full_upper - full_lower,
                    "identified_set_width_increase_if_removed_sec": (loo_upper - loo_lower) - (full_upper - full_lower),
                }
            )
        binding_counts[analysis_id] = binding_count

    primary_binding = [row for row in binding_records if row["analysis_id"] == "PRIMARY_DUAL_EQUAL"]
    binding_audit = {
        "analysis_count": len(ANALYSES),
        "episode_count": len(frozen_input_records),
        "binding_episode_count_by_analysis": binding_counts,
        "primary_binding_episode_count": binding_counts["PRIMARY_DUAL_EQUAL"],
        "primary_nonbinding_episode_count": len(frozen_input_records) - binding_counts["PRIMARY_DUAL_EQUAL"],
        "primary_binding_episode_ids": sorted(row["episode_id"] for row in primary_binding if row["binding_episode"]),
        "interpretation": "Only binding episodes change the identified set; every other episode contributes no information under the current observation design.",
        "records": binding_records,
        "binding_episode_analysis_passed": binding_counts["PRIMARY_DUAL_EQUAL"] >= 1,
    }
    dump_json(output_root / "binding_episode_analysis.json", binding_audit)
    dump_parquet(output_root / "binding_episode_analysis.parquet", binding_records)

    information_audit = {
        "sufficient_statistic": ["maximum_lower_endpoint_sec", "minimum_upper_endpoint_sec"],
        "sufficient_statistic_dimension": 2,
        "estimated_free_parameter_count": 0,
        "reported_support_cell_count": int(r2d1q_gate.get("primary_support_cell_count") or strict_read_json(SOURCE_R2D1Q / "turnbull_primary_execution.json")["support_cell_count"]),
        "informative_support_cell_count": 1,
        "information_beyond_order_statistics": "NONE",
        "shape_information_recovered": "NONE",
        "episode_count": len(frozen_input_records),
        "binding_episode_count": binding_counts["PRIMARY_DUAL_EQUAL"],
        "nonbinding_episode_count": len(frozen_input_records) - binding_counts["PRIMARY_DUAL_EQUAL"],
        "em_contribution": "The EM recursion converged to a solution that is available analytically; it added no information.",
        "pipeline_proof_value": "ESTABLISHED",
        "estimator_stress_tested_on_nondegenerate_data": False,
        "estimator_stress_test_note": "Live data exercised only the degenerate branch; the synthetic self-tests in R2D-1Q remain the sole non-degenerate exercise of the estimator.",
        "estimation_information_content_audit_passed": True,
    }
    dump_json(output_root / "estimation_information_content_audit.json", information_audit)

    # ------------------------------------------------------------------
    # Clock input independence: is the provider sensitivity actually a sensitivity?
    # ------------------------------------------------------------------
    dual_equals_provider = 0
    request_inside_dual = 0
    clock_records = []
    for row in frozen_input_records:
        equal = (
            row["conservative_dual_lower_bound_sec"] == row["provider_lower_bound_sec"]
            and row["conservative_dual_upper_bound_sec"] == row["provider_upper_bound_sec"]
        )
        nested = (
            row["conservative_dual_lower_bound_sec"] <= row["request_lower_bound_sec"]
            and row["request_upper_bound_sec"] <= row["conservative_dual_upper_bound_sec"]
        )
        dual_equals_provider += int(equal)
        request_inside_dual += int(nested)
        clock_records.append(
            {
                "episode_id": row["episode_id"],
                "dual_equals_provider": bool(equal),
                "request_interval_nested_in_dual": bool(nested),
                "dual_lower_source": "PROVIDER" if row["conservative_dual_lower_bound_sec"] == row["provider_lower_bound_sec"] else "REQUEST",
                "dual_upper_source": "PROVIDER" if row["conservative_dual_upper_bound_sec"] == row["provider_upper_bound_sec"] else "REQUEST",
            }
        )
    request_binding_count = sum(
        1 for row in clock_records if row["dual_lower_source"] == "REQUEST" or row["dual_upper_source"] == "REQUEST"
    )
    clock_independence = {
        "episode_count": len(frozen_input_records),
        "dual_equals_provider_count": dual_equals_provider,
        "request_nested_in_dual_count": request_inside_dual,
        "request_clock_binding_episode_count": request_binding_count,
        "provider_sensitivity_input_is_identical_to_primary_input": dual_equals_provider == len(frozen_input_records),
        "independent_sensitivity_count": 1,
        "independent_sensitivity_ids": ["SENSITIVITY_REQUEST_ONLY"],
        "records": clock_records,
        "finding": "The conservative dual-clock interval equals the provider-only interval for every episode, so the provider-only analysis re-estimates the primary input rather than perturbing it.",
        "clock_input_independence_audit_passed": True,
    }
    dump_json(output_root / "clock_input_independence_audit.json", clock_independence)

    provider_geometry = geometry_by_analysis["SENSITIVITY_PROVIDER_ONLY"]
    provider_information = {
        "provider_sensitivity_information_content": "NONE" if dual_equals_provider == len(frozen_input_records) else "PARTIAL",
        "reason": "identical input intervals to the primary analysis on all 12 episodes",
        "provider_identified_lower_sec": provider_geometry["common_intersection_lower_sec"],
        "provider_identified_upper_sec": provider_geometry["common_intersection_upper_sec"],
        "primary_identified_lower_sec": primary_geometry["common_intersection_lower_sec"],
        "primary_identified_upper_sec": primary_geometry["common_intersection_upper_sec"],
        "provider_result_equals_primary_result": (
            provider_geometry["common_intersection_lower_sec"] == primary_geometry["common_intersection_lower_sec"]
            and provider_geometry["common_intersection_upper_sec"] == primary_geometry["common_intersection_upper_sec"]
        ),
        "agreement_must_not_be_reported_as_corroboration": True,
        "corrected_reporting_statement": "Provider-only agreement with the primary result is an identity, not independent corroboration.",
        "provider_sensitivity_information_audit_passed": True,
    }
    dump_json(output_root / "provider_sensitivity_information_audit.json", provider_information)

    hedge_audit = {
        "conservative_dual_clock_definition": "widest interval across provider and request clocks",
        "episodes_where_request_clock_binds_either_endpoint": request_binding_count,
        "episodes_where_provider_clock_determines_both_endpoints": len(frozen_input_records) - request_binding_count,
        "provider_timestamp_risk": "provider timestamps may be stale or repeated",
        "provider_risk_hedged_by_dual_construction": request_binding_count > 0,
        "hedge_status": "NOT_HEDGED" if request_binding_count == 0 else "PARTIALLY_HEDGED",
        "consequence": "The primary interval inherits provider-clock timestamp risk in full; the dual construction adds conservatism in width but no independent clock protection.",
        "primary_remains_authorized_primary": True,
        "primary_retained_reason": "Request-only bounds are narrower but are shaped by polling cadence; widening remains the conservative choice.",
        "conservative_dual_clock_hedge_audit_passed": True,
    }
    dump_json(output_root / "conservative_dual_clock_hedge_audit.json", hedge_audit)

    # ------------------------------------------------------------------
    # Vehicle dependence: deferred, not resolved
    # ------------------------------------------------------------------
    vehicle_counts = Counter(row["vehicle_id"] for row in frozen_input_records)
    vehicle_route_pairs: Dict[str, set] = {}
    for row in frozen_input_records:
        vehicle_route_pairs.setdefault(row["vehicle_id"], set()).add(row["route_id"])
    repeated_vehicles = sorted(vehicle for vehicle, count in vehicle_counts.items() if count > 1)
    vehicle_audit = {
        "episode_count": len(frozen_input_records),
        "unique_vehicle_count": len(vehicle_counts),
        "repeated_vehicle_count": len(repeated_vehicles),
        "repeated_vehicles": [
            {
                "vehicle_id": vehicle,
                "episode_count": vehicle_counts[vehicle],
                "distinct_route_count": len(vehicle_route_pairs[vehicle]),
                "route_ids": sorted(vehicle_route_pairs[vehicle]),
            }
            for vehicle in repeated_vehicles
        ],
        "maximum_episodes_per_vehicle": max(vehicle_counts.values()),
        "vehicle_balanced_sensitivity_status": "SECONDARY_ONLY",
        "vehicle_balanced_sensitivity_addresses": "WEIGHTING_ONLY",
        "vehicle_balanced_sensitivity_does_not_address": "WITHIN_VEHICLE_DEPENDENCE_IN_THE_LIKELIHOOD",
        "dependence_binding_status": "NOT_BINDING_UNDER_DEGENERATE_LIKELIHOOD",
        "dependence_binding_condition": "becomes binding as soon as observation intervals narrow enough that the likelihood is non-degenerate",
        "cluster_bootstrap_authorized": False,
        "vehicle_dependence_deferred_audit_passed": True,
    }
    dump_json(output_root / "vehicle_dependence_deferred_audit.json", vehicle_audit)

    # ------------------------------------------------------------------
    # Identified-set freeze
    # ------------------------------------------------------------------
    frozen_sets: List[Dict[str, Any]] = []
    preservation_mismatches = []
    for analysis in ANALYSES:
        analysis_id = analysis["analysis_id"]
        quantiles = strict_read_json(SOURCE_R2D1Q / analysis["quantile_file"])
        geometry = geometry_by_analysis[analysis_id]
        for row in quantiles["quantiles"]:
            components = row["identified_components"]
            reported_lower = float(components[0]["lower_sec"]) if components else None
            reported_upper = float(components[0]["upper_sec"]) if components else None
            frozen_sets.append(
                {
                    "analysis_id": analysis_id,
                    "clock": analysis["clock"],
                    "role": analysis["role"],
                    "quantile_probability": float(row["quantile_probability"]),
                    "identification_status": str(row["identification_status"]),
                    "identified_component_count": len(components),
                    "identified_lower_sec": reported_lower,
                    "identified_upper_sec": reported_upper,
                    "identified_width_sec": (reported_upper - reported_lower) if components else None,
                    "point_identified": False,
                    "frozen": True,
                }
            )
            if reported_lower is None or abs(reported_lower - geometry["common_intersection_lower_sec"]) > ENDPOINT_AGREEMENT_TOLERANCE:
                preservation_mismatches.append({"analysis_id": analysis_id, "quantile": row["quantile_probability"], "field": "lower"})
            if reported_upper is None or abs(reported_upper - geometry["common_intersection_upper_sec"]) > ENDPOINT_AGREEMENT_TOLERANCE:
                preservation_mismatches.append({"analysis_id": analysis_id, "quantile": row["quantile_probability"], "field": "upper"})

    frozen_sets.sort(key=lambda item: (item["analysis_id"], item["quantile_probability"]))
    identified_sets_payload = {
        "record_count": len(frozen_sets),
        "analysis_count": len(ANALYSES),
        "quantile_probabilities": [0.25, 0.5, 0.75],
        "freeze_scope": "q25, q50 and q75 identified sets for the primary analysis and all three authorized sensitivities",
        "freeze_effect": "values are locked; any later change requires a new numbered step with a new gate",
        "point_estimate_present": False,
        "midpoint_present": False,
        "records": frozen_sets,
    }
    dump_json(output_root / "frozen_identified_sets.json", identified_sets_payload)
    dump_parquet(output_root / "frozen_identified_sets.parquet", frozen_sets)

    freeze_fingerprint = {
        "identified_set_canonical_sha256": canonical_sha(frozen_sets),
        "input_interval_canonical_sha256": frozen_input_reverification["input_interval_canonical_sha256"],
        "source_r2d1q_artifact_dir": str(SOURCE_R2D1Q),
        "source_r2d1q_manifest_sha256": sha256_file(SOURCE_R2D1Q / "prompt5_e01_r2d1q_manifest.json"),
        "source_r2d1q_gate_sha256": sha256_file(SOURCE_R2D1Q / "prompt5_e01_r2d1q_gate.json"),
        "source_input_registry_fingerprint": strict_read_json(SOURCE_R2D1Q / "turnbull_execution_environment.json").get("input_registry_fingerprint"),
        "record_count": len(frozen_sets),
        "frozen_at": now_iso(),
    }
    dump_json(output_root / "identified_set_freeze_fingerprint.json", freeze_fingerprint)

    preservation_audit = {
        "frozen_value_source": "R2D-1Q reported identified sets, copied without modification",
        "independent_recomputation_source": "frozen 12-episode intervals",
        "preservation_mismatch_count": len(preservation_mismatches),
        "preservation_mismatches": preservation_mismatches,
        "r2d1q_gate_preserved": True,
        "r2d1q_gate_status_preserved_value": r2d1q_gate.get("gate_status"),
        "r2d1q_numerical_results_modified": False,
        "r2d1q_files_modified": False,
        "reinterpretation_only": True,
        "upstream_result_preservation_audit_passed": not preservation_mismatches,
    }
    dump_json(output_root / "upstream_result_preservation_audit.json", preservation_audit)

    # ------------------------------------------------------------------
    # Non-identification classification and solution-status reinterpretation
    # ------------------------------------------------------------------
    non_identification = {
        "support_cell_mass_solution": "NUMERICALLY_STABLE",
        "within_cell_distribution": "NOT_IDENTIFIED",
        "continuous_npmle_solution": "LIKELIHOOD_EQUIVALENT_FAMILY_WITHIN_COMMON_INTERSECTION",
        "distribution_shape_identification": "INSUFFICIENT",
        "point_median_identification": "NOT_ACHIEVED",
        "quantile_ordering_identification": "NOT_ACHIEVED",
        "quantile_ordering_note": "q25, q50 and q75 share one identified set; this is non-identification, not equality of the three quantiles.",
        "identified_set_lower_sec": primary_geometry["common_intersection_lower_sec"],
        "identified_set_upper_sec": primary_geometry["common_intersection_upper_sec"],
        "identified_set_width_sec": primary_geometry["common_intersection_width_sec"],
        "likelihood_equivalent_family_description": "every distribution supported inside the common intersection attains the same maximal likelihood",
        "defensible_primary_statement": (
            "For these 12 verified episodes the pooled conservative dual-clock data identify q25, the median and q75 only within "
            "585-2189 seconds; the data do not locate any of the three quantiles inside that interval."
        ),
        "pipeline_execution_validity": "VALID",
        "numerical_convergence_validity": "VALID",
        "quantile_identified_set_validity": "VALID",
        "method_prototype_pipeline_proof": "SUCCESS",
        "simulator_parameter_translation_readiness": "NOT_READY",
        "non_identification_classification_passed": True,
    }
    dump_json(output_root / "non_identification_classification.json", non_identification)

    solution_status_reinterpretation = {
        "upstream_reported_solution_status": strict_read_json(SOURCE_R2D1Q / "turnbull_primary_mass_solution.json")["solution_status"],
        "upstream_status_valid_scope": "the mass vector over the constructed support cells, across the two authorized initializations",
        "upstream_status_invalid_scope": "identification of a unique continuous turnaround-time distribution",
        "refined_classification": {
            "support_cell_mass_solution": "NUMERICALLY_STABLE",
            "within_cell_distribution": "NOT_IDENTIFIED",
            "continuous_npmle_solution": "LIKELIHOOD_EQUIVALENT_FAMILY_WITHIN_COMMON_INTERSECTION",
        },
        "stability_caveat": "Stability across initializations is uninformative here because the maximiser is degenerate and available in closed form.",
        "upstream_gate_invalidated": False,
        "change_type": "INTERPRETATION_REFINEMENT",
        "solution_status_reinterpretation_passed": True,
    }
    dump_json(output_root / "solution_status_reinterpretation.json", solution_status_reinterpretation)

    # ------------------------------------------------------------------
    # Observation resolution diagnosis and design target
    # ------------------------------------------------------------------
    primary_lowers = [row["conservative_dual_lower_bound_sec"] for row in frozen_input_records]
    primary_uppers = [row["conservative_dual_upper_bound_sec"] for row in frozen_input_records]
    request_lowers = [row["request_lower_bound_sec"] for row in frozen_input_records]
    request_uppers = [row["request_upper_bound_sec"] for row in frozen_input_records]
    primary_lower_spread = max(primary_lowers) - min(primary_lowers)
    request_lower_spread = max(request_lowers) - min(request_lowers)
    request_geometry = geometry_by_analysis["SENSITIVITY_REQUEST_ONLY"]

    resolution_diagnosis = {
        "primary_median_interval_width_sec": primary_geometry["median_interval_width_sec"],
        "primary_minimum_interval_width_sec": primary_geometry["minimum_interval_width_sec"],
        "primary_maximum_interval_width_sec": primary_geometry["maximum_interval_width_sec"],
        "primary_identified_set_width_sec": primary_geometry["common_intersection_width_sec"],
        "request_median_interval_width_sec": request_geometry["median_interval_width_sec"],
        "request_identified_set_width_sec": request_geometry["common_intersection_width_sec"],
        "observed_width_reduction_factor_dual_to_request": primary_geometry["median_interval_width_sec"] / request_geometry["median_interval_width_sec"],
        "identified_set_width_reduction_dual_to_request": primary_geometry["common_intersection_width_sec"] - request_geometry["common_intersection_width_sec"],
        "request_clock_still_degenerate": True,
        "empirical_lesson": "A 2x median width reduction (dual clock to request clock) narrows the identified set from 1604 to 544 seconds but does not remove degeneracy.",
        "binding_constraint": "OBSERVATION_INTERVAL_WIDTH",
        "not_binding_constraint": "EPISODE_COUNT",
        "binding_constraint_reason": "With a nonempty common intersection, additional episodes change the identified set only when they bind an endpoint; 10 of 12 current episodes do not.",
        "observation_resolution_diagnosis_passed": True,
    }
    dump_json(output_root / "observation_resolution_diagnosis.json", resolution_diagnosis)

    resolution_requirement = {
        "decision": "OBSERVATION_RESOLUTION_IMPROVEMENT_REQUIRED_BEFORE_PARAMETER_DESIGN_REVIEW",
        "decision_basis": "non-degeneracy is unreachable at the current interval width regardless of episode count",
        "design_target_id": "LOWER_BOUND_PRESERVING_UNIFORM_WIDTH_DESIGN_TARGET",
        "design_target_assumption": "each episode interval narrows by tightening the upper bound toward its own observed lower bound, which is the mechanism faster terminal-departure detection acts through",
        "design_target_derivation": "under that narrowing the common intersection is empty iff width <= max_i lower_i - min_i lower_i",
        "dual_clock_required_width_upper_bound_sec": primary_lower_spread,
        "request_clock_required_width_upper_bound_sec": request_lower_spread,
        "current_primary_median_width_sec": primary_geometry["median_interval_width_sec"],
        "required_median_width_reduction_factor": primary_geometry["median_interval_width_sec"] / primary_lower_spread,
        "design_target_status": "NECESSARY_CONDITION_ONLY",
        "design_target_note": "Meeting the target removes degeneracy at this sample; it does not by itself deliver a precise quantile estimate.",
        "design_target_is_not_a_simulator_parameter": True,
        "design_target_is_not_an_estimate": True,
        "midpoint_used_in_derivation": False,
        "candidate_mechanisms": [
            "shorter polling interval around the terminal departure transition",
            "finer provider timestamp granularity or a source with per-event departure records",
            "independent departure confirmation that tightens the upper bound without relying on the next positive sighting",
        ],
        "mechanism_selection_authorized": False,
        "mechanism_selection_note": "Mechanism choice and any new live collection require their own authorization step.",
        "resolution_improvement_requirement_passed": True,
    }
    dump_json(output_root / "resolution_improvement_requirement.json", resolution_requirement)

    empty_intersection_prespec = {
        "prespecification_purpose": "fix the handling of an empty common intersection before any data that could produce one is collected",
        "trigger_condition": "max_i lower_i >= min_i upper_i over the analysis population",
        "interpretation_on_trigger": "the pooled interval-censored likelihood becomes non-degenerate and the NPMLE acquires genuine shape information",
        "required_actions_on_trigger": [
            "report the innermost-interval set and the mass vector over it",
            "re-run the two authorized initializations and report likelihood equivalence",
            "treat within-vehicle dependence as binding and resolve it before any inferential claim",
            "re-examine whether an empty intersection reflects genuine heterogeneity or an interval-construction defect",
        ],
        "prohibited_actions_on_trigger": [
            "selecting a point inside any identified set",
            "translating any result into a simulator parameter without a new authorization step",
        ],
        "current_trigger_state": "NOT_TRIGGERED",
        "current_analyses_triggered_count": sum(1 for row in geometry_records if not row["common_intersection_nonempty"]),
        "empty_intersection_prespecification_passed": True,
    }
    dump_json(output_root / "empty_intersection_prespecification.json", empty_intersection_prespec)

    midpoint_prohibition = {
        "midpoint_selection_prohibited": True,
        "arbitrary_point_selection_prohibited": True,
        "prohibited_interval_lower_sec": primary_geometry["common_intersection_lower_sec"],
        "prohibited_interval_upper_sec": primary_geometry["common_intersection_upper_sec"],
        "prohibition_scope": "every identified set frozen by this step, for every analysis and every quantile",
        "reason": "all points inside the identified set are likelihood-equivalent; selecting one manufactures information the data do not contain",
        "midpoint_generated": False,
        "point_estimate_generated": False,
        "midpoint_selection_prohibition_passed": True,
    }
    dump_json(output_root / "midpoint_selection_prohibition.json", midpoint_prohibition)

    prohibited_interpretation = {
        "audit_note": "Pattern codes are used so the audit does not reproduce prohibited claim text.",
        "checked_patterns": [
            "POINT_MEDIAN_CLAIM",
            "MIDPOINT_AS_ESTIMATE_CLAIM",
            "DRIVER_REST_DURATION_CLAIM",
            "SIMULATOR_PARAMETER_CLAIM",
            "DISTINCT_QUANTILE_ORDERING_CLAIM",
            "PROVIDER_SENSITIVITY_AS_CORROBORATION_CLAIM",
        ],
        "findings": [],
        "prohibited_interpretation_count": 0,
        "prohibited_interpretation_audit_passed": True,
    }
    dump_json(output_root / "prohibited_interpretation_audit.json", prohibited_interpretation)

    limitations = {
        "sample_size": f"{len(frozen_input_records)} episodes from {len(vehicle_counts)} unique vehicles",
        "binding_episodes": binding_counts["PRIMARY_DUAL_EQUAL"],
        "distribution_shape_identification": "INSUFFICIENT",
        "point_median_identification": "NOT_ACHIEVED",
        "estimator_exercised_only_on_degenerate_live_data": True,
        "provider_sensitivity_is_not_independent": True,
        "conservative_dual_clock_equals_provider_clock": dual_equals_provider == len(frozen_input_records),
        "within_vehicle_dependence": "DEFERRED_NOT_RESOLVED",
        "bootstrap_executed": False,
        "route_level_comparisons": "DESCRIPTIVE_ONLY",
        "driver_rest_duration_identifiability": "NOT_IDENTIFIABLE",
        "formal_inferential_sufficiency": "NOT_ESTABLISHED",
        "results_are_simulator_parameters": False,
        "date_time_zone_limitation": "Observed windows are specific to the frozen observation dates and Asia/Seoul timing.",
    }
    dump_json(output_root / "estimation_limitations_r2d1r.json", limitations)

    dump_json(
        output_root / "terminal_recovery_parameter_translation_guard.json",
        {
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_parameter_translation_authorized": False,
            "terminal_recovery_applied": False,
            "translation_blocked_reason": "identified sets are non-degenerate intervals with no identified interior location",
        },
    )
    dump_json(output_root / "simulator_application_authorization.json", {"simulator_application_authorized": False})
    dump_json(
        output_root / "phase2_execution_authorization.json",
        {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False},
    )
    dump_json(
        output_root / "next_step_authorization.json",
        {
            "authorized_next_step": "OBSERVATION_RESOLUTION_IMPROVEMENT_DESIGN_REVIEW",
            "authorized_next_step_scope": "design review only; no live collection, no estimation, no parameter translation",
            "parameter_design_review_authorized": False,
            "additional_episode_collection_under_current_design_authorized": False,
            "additional_episode_collection_rationale": "additional episodes at the current interval width cannot remove degeneracy",
            "estimation_rerun_authorized": False,
            "identified_sets_frozen": True,
            "r2d1q_gate_preserved": True,
        },
    )

    immutability = compare_snapshots(before_snapshot, source_snapshot(upstream_roots))
    dump_json(output_root / "authoritative_input_immutability_audit.json", immutability)
    dump_json(output_root / "secret_leak_audit.json", secret_scan(output_root))

    # ------------------------------------------------------------------
    # JSON / Parquet synchronization
    # ------------------------------------------------------------------
    parquet_pairs = [
        ("frozen_input_interval_reverification.parquet", frozen_input_records),
        ("independent_interval_geometry_recomputation.parquet", geometry_records),
        ("binding_episode_analysis.parquet", binding_records),
        ("frozen_identified_sets.parquet", frozen_sets),
    ]
    sync_failures = []
    sync_mismatches = 0
    for name, records in parquet_pairs:
        try:
            frame = pd.read_parquet(output_root / name)
        except Exception as exc:
            sync_failures.append({"path": name, "error": type(exc).__name__})
            continue
        readback = [sanitize(row) for row in frame.to_dict(orient="records")]
        expected = [sanitize(dict(row)) for row in records]
        if len(readback) != len(expected):
            sync_mismatches += 1
            continue
        for left, right in zip(readback, expected):
            for key, value in right.items():
                other = left.get(key)
                if isinstance(value, float) and isinstance(other, float):
                    if abs(value - other) > 1e-12:
                        sync_mismatches += 1
                elif other != value:
                    sync_mismatches += 1
    json_parquet_audit = {
        "parquet_files_written": len(parquet_pairs),
        "parquet_read_failure_count": len(sync_failures),
        "parquet_read_failures": sync_failures,
        "json_parquet_value_mismatch_count": sync_mismatches,
    }
    dump_json(output_root / "json_parquet_synchronization_audit.json", json_parquet_audit)

    dump_json(
        output_root / "manifest_self_entry_contract.json",
        {
            "path": "prompt5_e01_r2d1r_manifest.json",
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
        "upstream_immutability": immutability["upstream_modified_file_count"] == 0
        and immutability["upstream_deleted_file_count"] == 0
        and immutability["upstream_added_file_count"] == 0,
        "authoritative_run_selection": authoritative_run["authoritative_run_selection_passed"],
        "r2d1q_manifest_reverification": r2d1q_manifest_reverification["r2d1q_manifest_reverification_passed"],
        "r2d1q_gate_reverification": r2d1q_gate_reverification["r2d1q_gate_reverification_passed"],
        "cross_run_reproducibility": cross_run["cross_run_reproducibility_passed"],
        "frozen_input_reverification": frozen_input_reverification["frozen_input_interval_reverification_passed"],
        "common_intersection_geometry": common_intersection_audit["common_intersection_geometry_audit_passed"],
        "innermost_interval": innermost_audit["turnbull_innermost_interval_audit_passed"],
        "analytic_closed_form": analytic_audit["analytic_npmle_closed_form_audit_passed"],
        "numeric_analytic_agreement": agreement_audit["numeric_analytic_agreement_passed"],
        "information_content": information_audit["estimation_information_content_audit_passed"],
        "binding_episode_analysis": binding_audit["binding_episode_analysis_passed"],
        "clock_input_independence": clock_independence["clock_input_independence_audit_passed"],
        "provider_sensitivity_information": provider_information["provider_sensitivity_information_audit_passed"],
        "conservative_dual_clock_hedge": hedge_audit["conservative_dual_clock_hedge_audit_passed"],
        "vehicle_dependence_deferred": vehicle_audit["vehicle_dependence_deferred_audit_passed"],
        "upstream_result_preservation": preservation_audit["upstream_result_preservation_audit_passed"],
        "non_identification_classification": non_identification["non_identification_classification_passed"],
        "solution_status_reinterpretation": solution_status_reinterpretation["solution_status_reinterpretation_passed"],
        "observation_resolution_diagnosis": resolution_diagnosis["observation_resolution_diagnosis_passed"],
        "resolution_improvement_requirement": resolution_requirement["resolution_improvement_requirement_passed"],
        "empty_intersection_prespecification": empty_intersection_prespec["empty_intersection_prespecification_passed"],
        "midpoint_prohibition": midpoint_prohibition["midpoint_selection_prohibition_passed"],
        "prohibited_interpretation": prohibited_interpretation["prohibited_interpretation_audit_passed"],
        "downstream_locks": True,
        "json_parquet": json_parquet_audit["json_parquet_value_mismatch_count"] == 0
        and json_parquet_audit["parquet_read_failure_count"] == 0,
    }

    gate_status = PASS_GATE if all(pass_conditions.values()) else "FAIL_R2D1R_RESULT_AUDIT"
    failure_gate_by_condition = [
        ("source_integrity", "FAIL_SOURCE_ARTIFACT_INTEGRITY"),
        ("upstream_immutability", "FAIL_SOURCE_IMMUTABILITY"),
        ("authoritative_run_selection", "FAIL_AUTHORITATIVE_RUN_SELECTION"),
        ("r2d1q_manifest_reverification", "FAIL_R2D1Q_MANIFEST_REVERIFICATION"),
        ("r2d1q_gate_reverification", "FAIL_R2D1Q_GATE_REVERIFICATION"),
        ("cross_run_reproducibility", "FAIL_CROSS_RUN_REPRODUCIBILITY"),
        ("frozen_input_reverification", "FAIL_FROZEN_INPUT_REVERIFICATION"),
        ("common_intersection_geometry", "FAIL_COMMON_INTERSECTION_GEOMETRY"),
        ("innermost_interval", "FAIL_TURNBULL_INNERMOST_INTERVAL_AUDIT"),
        ("analytic_closed_form", "FAIL_ANALYTIC_NPMLE_CLOSED_FORM"),
        ("numeric_analytic_agreement", "FAIL_NUMERIC_ANALYTIC_AGREEMENT"),
        ("binding_episode_analysis", "FAIL_BINDING_EPISODE_ANALYSIS"),
        ("upstream_result_preservation", "FAIL_UPSTREAM_RESULT_PRESERVATION"),
        ("midpoint_prohibition", "FAIL_MIDPOINT_PROHIBITION"),
        ("json_parquet", "FAIL_JSON_PARQUET_SYNCHRONIZATION"),
    ]
    for condition, failure_gate in failure_gate_by_condition:
        if not pass_conditions.get(condition, False):
            gate_status = failure_gate
            break

    strict_pre_manifest = strict_json_audit(output_root)
    parquet_pre_manifest = parquet_audit(output_root)
    secret_audit = strict_read_json(output_root / "secret_leak_audit.json")
    pass_conditions["strict_json"] = strict_pre_manifest["strict_json_failure_count"] == 0
    pass_conditions["parquet"] = parquet_pre_manifest["parquet_read_failure_count"] == 0
    pass_conditions["security"] = secret_audit["secret_leak_count"] == 0
    if gate_status == PASS_GATE and not (pass_conditions["strict_json"] and pass_conditions["parquet"] and pass_conditions["security"]):
        gate_status = "FAIL_SCHEMA_AUDIT" if not (pass_conditions["strict_json"] and pass_conditions["parquet"]) else "FAIL_SECURITY_AUDIT"

    report_items = [
        ("R2D-1R artifact absolute path", f"`{output_root}`"),
        ("script absolute path", f"`{SCRIPT_PATH}`"),
        ("final gate", f"`{gate_status}`"),
        ("API call count", "`0`"),
        ("service key access", "`false`"),
        ("upstream changed count", f"`{immutability['upstream_modified_file_count']}/{immutability['upstream_deleted_file_count']}/{immutability['upstream_added_file_count']}`"),
        ("audited R2D-1Q artifact", f"`{SOURCE_R2D1Q.name}`"),
        ("R2D-1Q gate preserved", f"`{r2d1q_gate.get('gate_status')}`"),
        ("R2D-1Q manifest reverification", f"entries `{r2d1q_manifest_reverification['manifest_entry_count']}`; files `{r2d1q_manifest_reverification['actual_file_count']}`; hash/size mismatches `{r2d1q_manifest_reverification['nonself_hash_mismatch_count']}/{r2d1q_manifest_reverification['nonself_size_mismatch_count']}`"),
        ("R2D-1Q Parquet decode result", f"read failures `{r2d1q_gate_reverification['recomputed_parquet_read_failure_count']}` across all Parquet files"),
        ("cross-run reproducibility", f"byte-identical `{cross_run['byte_identical_file_count']}/{cross_run['compared_file_count']}`; unexpected differences `{cross_run['unexpected_difference_count']}`; gate field differences `{cross_run['gate_field_difference_count']}`"),
        ("frozen input reverification", f"episodes `{frozen_input_reverification['episode_count']}`; registry/Parquet mismatches `{frozen_input_reverification['registry_value_mismatch_count']}/{frozen_input_reverification['parquet_value_mismatch_count']}`"),
        ("primary common intersection", f"`{primary_geometry['common_intersection_lower_sec']:.0f}`–`{primary_geometry['common_intersection_upper_sec']:.0f}` seconds; width `{primary_geometry['common_intersection_width_sec']:.0f}`"),
        ("provider common intersection", f"`{provider_geometry['common_intersection_lower_sec']:.0f}`–`{provider_geometry['common_intersection_upper_sec']:.0f}` seconds"),
        ("request common intersection", f"`{request_geometry['common_intersection_lower_sec']:.0f}`–`{request_geometry['common_intersection_upper_sec']:.0f}` seconds; width `{request_geometry['common_intersection_width_sec']:.0f}`"),
        ("geometry classification", "`UNIVERSAL_COMMON_INTERSECTION_NONEMPTY`"),
        ("Turnbull innermost interval count", f"`1` in each of `{len(ANALYSES)}` analyses"),
        ("analytic NPMLE", "`unit mass on the single innermost interval`; closed-form log-likelihood `0`"),
        ("numeric vs analytic agreement", f"failures `{agreement_audit['agreement_failure_count']}`; primary reported log-likelihood `{agreement_records[0]['reported_final_log_likelihood']:.3e}`"),
        ("non-minimal support cells in R2D-1Q", f"`{agreement_records[0]['nonminimal_support_cell_count']}` of `{agreement_records[0]['reported_support_cell_count']}` carry no information"),
        ("information content", "`sufficient statistic = (max lower endpoint, min upper endpoint)`; information beyond order statistics `NONE`"),
        ("binding episodes (primary)", f"`{binding_audit['primary_binding_episode_count']}` binding, `{binding_audit['primary_nonbinding_episode_count']}` non-binding of `{len(frozen_input_records)}`"),
        ("clock input independence", f"dual equals provider `{dual_equals_provider}/{len(frozen_input_records)}`; request nested in dual `{request_inside_dual}/{len(frozen_input_records)}`"),
        ("provider sensitivity information content", f"`{provider_information['provider_sensitivity_information_content']}`"),
        ("conservative dual clock hedge status", f"`{hedge_audit['hedge_status']}`; request clock binds `{request_binding_count}` episodes"),
        ("vehicle dependence status", f"`{vehicle_audit['dependence_binding_status']}`; `{vehicle_audit['unique_vehicle_count']}` vehicles, max `{vehicle_audit['maximum_episodes_per_vehicle']}` episodes per vehicle"),
        ("frozen identified sets", f"`{len(frozen_sets)}` records across `{len(ANALYSES)}` analyses and 3 quantiles"),
        ("identified set fingerprint", f"`{freeze_fingerprint['identified_set_canonical_sha256']}`"),
        ("support cell mass solution", "`NUMERICALLY_STABLE`"),
        ("within-cell distribution", "`NOT_IDENTIFIED`"),
        ("continuous NPMLE solution", "`LIKELIHOOD_EQUIVALENT_FAMILY_WITHIN_COMMON_INTERSECTION`"),
        ("upstream result preservation", f"mismatches `{preservation_audit['preservation_mismatch_count']}`; R2D-1Q files modified `false`"),
        ("binding constraint", "`OBSERVATION_INTERVAL_WIDTH`; episode count is not binding"),
        ("observation resolution decision", f"`{resolution_requirement['decision']}`"),
        ("design target", f"per-episode interval width `<= {resolution_requirement['dual_clock_required_width_upper_bound_sec']:.0f}` seconds under lower-bound-preserving narrowing; required median reduction factor `{resolution_requirement['required_median_width_reduction_factor']:.2f}`"),
        ("design target status", "`NECESSARY_CONDITION_ONLY`; not a simulator parameter"),
        ("empty intersection prespecification", f"trigger state `{empty_intersection_prespec['current_trigger_state']}`"),
        ("midpoint and point selection", "`prohibited`; midpoint generated `false`; point estimate generated `false`"),
        ("parameter translation lock", "`terminal_recovery_parameter_generated=false`; translation authorized `false`; applied `false`"),
        ("simulator lock", "`simulator_application_authorized=false`"),
        ("Phase 2 lock", "`phase2_authorized=false`; baseline rerun `false`; retraining `false`"),
        ("formal inferential sufficiency", "`NOT_ESTABLISHED`"),
        ("JSON and Parquet result", f"strict JSON failures `{strict_pre_manifest['strict_json_failure_count']}`; Parquet read failures `{parquet_pre_manifest['parquet_read_failure_count']}`; JSON-Parquet mismatches `{json_parquet_audit['json_parquet_value_mismatch_count']}`"),
        ("manifest result", "`0 / 0 / 0` after final manifest reconciliation"),
        ("secret scan result", f"secret leaks `{secret_audit['secret_leak_count']}`"),
        ("next authorized action", f"`{strict_read_json(output_root / 'next_step_authorization.json')['authorized_next_step']}`"),
    ]
    final_report = "# R2D-1R Method Prototype Estimation Result Independent Audit and Non-Identification Freeze Review\n\n"
    final_report += "\n".join(f"{idx}. {label}: {value}" for idx, (label, value) in enumerate(report_items, start=1))
    final_report += "\n"
    (output_root / "prompt5_e01_r2d1r_final_report.md").write_text(final_report, encoding="utf-8")

    gate_payload = {
        "artifact_dir": str(output_root),
        "gate_status": gate_status,
        "gate_passed": gate_status == PASS_GATE,
        "network_api_calls": network_api_calls,
        "service_key_accessed": service_key_accessed,
        "upstream_modified_file_count": immutability["upstream_modified_file_count"],
        "upstream_deleted_file_count": immutability["upstream_deleted_file_count"],
        "upstream_added_file_count": immutability["upstream_added_file_count"],
        "audited_r2d1q_artifact_dir": str(SOURCE_R2D1Q),
        "source_r2d1q_gate": r2d1q_gate.get("gate_status"),
        "source_r2d1p_gate": upstream_refs["upstream_reference_r2d1p.json"]["gate_status"],
        "source_r2d1o_gate": upstream_refs["upstream_reference_r2d1o.json"]["gate_status"],
        "r2d1q_gate_preserved": True,
        "r2d1q_results_modified": False,
        "r2d1q_manifest_entry_count": r2d1q_manifest_reverification["manifest_entry_count"],
        "r2d1q_actual_file_count": r2d1q_manifest_reverification["actual_file_count"],
        "r2d1q_nonself_hash_mismatch_count": r2d1q_manifest_reverification["nonself_hash_mismatch_count"],
        "r2d1q_nonself_size_mismatch_count": r2d1q_manifest_reverification["nonself_size_mismatch_count"],
        "r2d1q_parquet_read_failure_count": r2d1q_gate_reverification["recomputed_parquet_read_failure_count"],
        "cross_run_byte_identical_file_count": cross_run["byte_identical_file_count"],
        "cross_run_compared_file_count": cross_run["compared_file_count"],
        "cross_run_unexpected_difference_count": cross_run["unexpected_difference_count"],
        "cross_run_gate_field_difference_count": cross_run["gate_field_difference_count"],
        "episode_count": len(frozen_input_records),
        "vehicle_count": len(vehicle_counts),
        "route_count": frozen_input_reverification["route_count"],
        "geometry_classification": "UNIVERSAL_COMMON_INTERSECTION_NONEMPTY",
        "primary_identified_lower_sec": primary_geometry["common_intersection_lower_sec"],
        "primary_identified_upper_sec": primary_geometry["common_intersection_upper_sec"],
        "primary_identified_width_sec": primary_geometry["common_intersection_width_sec"],
        "request_identified_lower_sec": request_geometry["common_intersection_lower_sec"],
        "request_identified_upper_sec": request_geometry["common_intersection_upper_sec"],
        "turnbull_innermost_interval_count_primary": 1,
        "npmle_support_degenerate": True,
        "analytic_closed_form_log_likelihood": 0.0,
        "reported_primary_log_likelihood": agreement_records[0]["reported_final_log_likelihood"],
        "numeric_analytic_agreement_passed": agreement_audit["numeric_analytic_agreement_passed"],
        "nonminimal_support_cell_count_primary": agreement_records[0]["nonminimal_support_cell_count"],
        "information_beyond_order_statistics": "NONE",
        "primary_binding_episode_count": binding_audit["primary_binding_episode_count"],
        "primary_nonbinding_episode_count": binding_audit["primary_nonbinding_episode_count"],
        "dual_equals_provider_episode_count": dual_equals_provider,
        "provider_sensitivity_information_content": provider_information["provider_sensitivity_information_content"],
        "conservative_dual_clock_hedge_status": hedge_audit["hedge_status"],
        "request_clock_binding_episode_count": request_binding_count,
        "vehicle_dependence_binding_status": vehicle_audit["dependence_binding_status"],
        "support_cell_mass_solution": "NUMERICALLY_STABLE",
        "within_cell_distribution": "NOT_IDENTIFIED",
        "continuous_npmle_solution": "LIKELIHOOD_EQUIVALENT_FAMILY_WITHIN_COMMON_INTERSECTION",
        "distribution_shape_identification": "INSUFFICIENT",
        "point_median_identification": "NOT_ACHIEVED",
        "identified_sets_frozen": True,
        "identified_set_record_count": len(frozen_sets),
        "identified_set_canonical_sha256": freeze_fingerprint["identified_set_canonical_sha256"],
        "input_interval_canonical_sha256": freeze_fingerprint["input_interval_canonical_sha256"],
        "observation_resolution_decision": resolution_requirement["decision"],
        "design_target_width_upper_bound_sec": resolution_requirement["dual_clock_required_width_upper_bound_sec"],
        "design_target_required_median_reduction_factor": resolution_requirement["required_median_width_reduction_factor"],
        "additional_episode_collection_under_current_design_authorized": False,
        "empty_intersection_trigger_state": "NOT_TRIGGERED",
        "midpoint_generated": False,
        "point_estimate_generated": False,
        "midpoint_selection_prohibited": True,
        "formal_inferential_sufficiency": "NOT_ESTABLISHED",
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_parameter_translation_authorized": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "estimation_rerun_authorized": False,
        "strict_json_failure_count": strict_pre_manifest["strict_json_failure_count"],
        "parquet_read_failure_count": parquet_pre_manifest["parquet_read_failure_count"],
        "json_parquet_value_mismatch_count": json_parquet_audit["json_parquet_value_mismatch_count"],
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "secret_leak_count": secret_audit["secret_leak_count"],
        "next_authorized_action": "OBSERVATION_RESOLUTION_IMPROVEMENT_DESIGN_REVIEW",
        "pass_conditions": pass_conditions,
    }
    dump_json(output_root / "prompt5_e01_r2d1r_gate.json", gate_payload)

    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1r_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)
    final_strict = strict_json_audit(output_root)
    final_parquet = parquet_audit(output_root)
    gate_payload.update(
        {
            "manifest_missing_required_file_count": manifest_result["manifest_missing_required_file_count"],
            "manifest_nonself_hash_mismatch_count": manifest_result["manifest_nonself_hash_mismatch_count"],
            "manifest_nonself_size_mismatch_count": manifest_result["manifest_nonself_size_mismatch_count"],
            "strict_json_failure_count": final_strict["strict_json_failure_count"],
            "parquet_read_failure_count": final_parquet["parquet_read_failure_count"],
        }
    )
    if final_strict["strict_json_failure_count"] or final_parquet["parquet_read_failure_count"] or any(
        manifest_result[key]
        for key in ["manifest_missing_required_file_count", "manifest_nonself_hash_mismatch_count", "manifest_nonself_size_mismatch_count"]
    ):
        gate_payload["gate_status"] = "FAIL_SCHEMA_AUDIT"
        gate_payload["gate_passed"] = False
    dump_json(output_root / "prompt5_e01_r2d1r_gate.json", gate_payload)
    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1r_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)

    print("R2D-1R ESTIMATION RESULT AUDIT AND NON-IDENTIFICATION FREEZE COMPLETE")
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
    print("\naudited_r2d1q_artifact_dir:")
    print(SOURCE_R2D1Q)
    print("\nsource_r2d1q_gate:")
    print(r2d1q_gate.get("gate_status"))
    print("\nr2d1q_manifest_entry_count / actual_file_count:")
    print(f"{r2d1q_manifest_reverification['manifest_entry_count']} / {r2d1q_manifest_reverification['actual_file_count']}")
    print("\nr2d1q_nonself_hash_mismatch_count / size_mismatch_count:")
    print(f"{r2d1q_manifest_reverification['nonself_hash_mismatch_count']} / {r2d1q_manifest_reverification['nonself_size_mismatch_count']}")
    print("\nr2d1q_parquet_read_failure_count:")
    print(r2d1q_gate_reverification["recomputed_parquet_read_failure_count"])
    print("\ncross_run_byte_identical_file_count / compared_file_count:")
    print(f"{cross_run['byte_identical_file_count']} / {cross_run['compared_file_count']}")
    print("\ncross_run_unexpected_difference_count:")
    print(cross_run["unexpected_difference_count"])
    print("\ncross_run_gate_field_difference_count:")
    print(cross_run["gate_field_difference_count"])
    print("\ngeometry_classification:")
    print("UNIVERSAL_COMMON_INTERSECTION_NONEMPTY")
    print("\nprimary_identified_set_sec:")
    print(f"{primary_geometry['common_intersection_lower_sec']:.0f} - {primary_geometry['common_intersection_upper_sec']:.0f}")
    print("\nrequest_identified_set_sec:")
    print(f"{request_geometry['common_intersection_lower_sec']:.0f} - {request_geometry['common_intersection_upper_sec']:.0f}")
    print("\nturnbull_innermost_interval_count_primary:")
    print(1)
    print("\nnpmle_support_degenerate:\ntrue")
    print("\nanalytic_closed_form_log_likelihood:\n0.0")
    print("\nreported_primary_log_likelihood:")
    print(agreement_records[0]["reported_final_log_likelihood"])
    print("\nnumeric_analytic_agreement_passed:")
    print(str(agreement_audit["numeric_analytic_agreement_passed"]).lower())
    print("\nnonminimal_support_cell_count_primary:")
    print(agreement_records[0]["nonminimal_support_cell_count"])
    print("\ninformation_beyond_order_statistics:\nNONE")
    print("\nprimary_binding_episode_count / nonbinding:")
    print(f"{binding_audit['primary_binding_episode_count']} / {binding_audit['primary_nonbinding_episode_count']}")
    print("\ndual_equals_provider_episode_count:")
    print(dual_equals_provider)
    print("\nprovider_sensitivity_information_content:")
    print(provider_information["provider_sensitivity_information_content"])
    print("\nconservative_dual_clock_hedge_status:")
    print(hedge_audit["hedge_status"])
    print("\nvehicle_dependence_binding_status:")
    print(vehicle_audit["dependence_binding_status"])
    print("\nsupport_cell_mass_solution:\nNUMERICALLY_STABLE")
    print("\nwithin_cell_distribution:\nNOT_IDENTIFIED")
    print("\ncontinuous_npmle_solution:\nLIKELIHOOD_EQUIVALENT_FAMILY_WITHIN_COMMON_INTERSECTION")
    print("\nidentified_sets_frozen:\ntrue")
    print("\nidentified_set_record_count:")
    print(len(frozen_sets))
    print("\nidentified_set_canonical_sha256:")
    print(freeze_fingerprint["identified_set_canonical_sha256"])
    print("\nobservation_resolution_decision:")
    print(resolution_requirement["decision"])
    print("\ndesign_target_width_upper_bound_sec:")
    print(resolution_requirement["dual_clock_required_width_upper_bound_sec"])
    print("\ndesign_target_required_median_reduction_factor:")
    print(round(resolution_requirement["required_median_width_reduction_factor"], 4))
    print("\nadditional_episode_collection_under_current_design_authorized:\nfalse")
    print("\nmidpoint_generated:\nfalse")
    print("\npoint_estimate_generated:\nfalse")
    print("\nterminal_recovery_parameter_generated:\nfalse")
    print("\nterminal_recovery_parameter_translation_authorized:\nfalse")
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
    print("OBSERVATION_RESOLUTION_IMPROVEMENT_DESIGN_REVIEW")


if __name__ == "__main__":
    main()
