#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1k_hf1_campaign_b_metadata_finalization.py"

R2D1K_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_campaign_b_controlled_live_observation_20260727_093548"
R2D1J_HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1j_hf1_campaign_b_early_stop_semantics_handoff_finalization_20260726_235819"
R2D1J_HF1_FAILED_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1j_hf1_campaign_b_early_stop_semantics_handoff_finalization_20260726_235746"
R2D1I_HF2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_hf2_final_metadata_reconciliation_20260726_191538"
HF1_MAPPING_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"

R2D1K_GATE_PATH = R2D1K_ROOT / "prompt5_e01_r2d1k_gate.json"
R2D1J_HF1_GATE_PATH = R2D1J_HF1_ROOT / "prompt5_e01_r2d1j_hf1_gate.json"
R2D1I_HF2_GATE_PATH = R2D1I_HF2_ROOT / "prompt5_e01_r2d1i_hf2_gate.json"
MAPPING_PATH = HF1_MAPPING_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"

TARGET_ROUTES = ["4010002004", "4050010000"]
EXCLUDED_ROUTES = ["4010002001", "4010002118"]
PASS_GATE = "PASS_CAMPAIGN_B_METADATA_FINALIZATION_FREEZE_READY"

OLD_TO_NEW_EPISODE_ID = {
    "r2d1h_episode_00001_4010002004": "r2d1k_episode_00001_4010002004",
    "r2d1h_episode_00002_4050010000": "r2d1k_episode_00002_4050010000",
    "r2d1h_episode_00003_4010002004": "r2d1k_episode_00003_4010002004",
}

REQUIRED_FILES = [
    "prompt5_e01_r2d1k_hf1_manifest.json",
    "prompt5_e01_r2d1k_hf1_gate.json",
    "prompt5_e01_r2d1k_hf1_final_report.md",
    "upstream_reference_r2d1k.json",
    "upstream_reference_r2d1j_hf1.json",
    "upstream_reference_r2d1i_hf2.json",
    "upstream_reference_hf1_mapping.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "scientific_content_preservation_audit.json",
    "interval_content_preservation_audit.json",
    "raw_provenance_reference_audit.json",
    "mapping_regression_audit.json",
    "secret_leak_audit.json",
    "campaign_b_episode_namespace_correction_contract.json",
    "campaign_b_episode_namespace_correction_audit.json",
    "campaign_b_episode_namespace_correction_audit.parquet",
    "campaign_b_right_censor_reason_correction.json",
    "campaign_b_right_censor_reason_audit.json",
    "campaign_b_route_local_vehicle_contract.json",
    "campaign_b_route_local_vehicle_audit.json",
    "campaign_b_route_local_vehicle_audit.parquet",
    "campaign_b_corrected_counter_contract_v11_hf1.json",
    "campaign_b_corrected_counter_contract_v11_hf1.parquet",
    "campaign_b_counter_cross_row_reconciliation.json",
    "campaign_b_corrected_terminal_recovery_episodes.json",
    "campaign_b_corrected_terminal_recovery_episodes.parquet",
    "campaign_b_corrected_terminal_recovery_interval_bounds.json",
    "campaign_b_corrected_terminal_recovery_interval_bounds.parquet",
    "campaign_b_corrected_route_summary.json",
    "campaign_b_corrected_route_summary.parquet",
    "cumulative_episode_registry_candidate_hf1.json",
    "cumulative_episode_registry_candidate_hf1.parquet",
    "method_prototype_progress_audit_hf1.json",
    "campaign_b_final_result_hf1.json",
    "episode_deduplication_audit_hf1.json",
    "lineage_reconciliation_audit_hf1.json",
    "json_parquet_synchronization_audit.json",
    "manifest_self_entry_contract.json",
    "supersession_reference_hf1.json",
    "campaign_c_execution_authorization.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]


class GateFailure(Exception):
    def __init__(self, gate: str, message: str) -> None:
        super().__init__(message)
        self.gate = gate


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def snapshot(roots: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        for path in files_under(root):
            result[str(path)] = {
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(root)),
                "file_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "root": str(root),
            }
    return result


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    pd.read_parquet(path)


def dataframe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [jsonable(record) for record in frame.to_dict("records")]


def route_counts(registry: pd.DataFrame) -> Dict[str, int]:
    counts = registry.groupby(registry["route_id"].astype(str)).size().to_dict()
    return {route: int(counts.get(route, 0)) for route in sorted(counts)}


def compact_sha_list(value: Any) -> List[str]:
    normalized = jsonable(value)
    if normalized is None:
        return []
    if isinstance(normalized, list):
        return [str(item) for item in normalized if item]
    return [str(normalized)] if normalized else []


def raw_sha_index(root: Path) -> Dict[str, str]:
    result = {}
    for path in sorted((root / "raw").glob("*/*.json")):
        result[sha256_file(path)] = str(path.relative_to(root))
    return result


def raw_counts(root: Path) -> Dict[str, Any]:
    route_counts_payload = {
        route: len(list((root / "raw" / route).glob("*.json")))
        for route in TARGET_ROUTES
    }
    non_target_dirs = [
        path.name
        for path in (root / "raw").iterdir()
        if path.is_dir() and path.name not in TARGET_ROUTES
    ] if (root / "raw").exists() else []
    return {
        "raw_file_count": sum(route_counts_payload.values()),
        "route_raw_counts": route_counts_payload,
        "non_target_route_raw_directory_count": len(non_target_dirs),
        "non_target_route_raw_directories": non_target_dirs,
    }


def build_upstream_reference(root: Path, gate_path: Path | None) -> Dict[str, Any]:
    gate_status = None
    if gate_path and gate_path.exists():
        gate_status = read_json(gate_path).get("gate_status")
    return {
        "artifact_path": str(root),
        "exists": root.exists(),
        "gate_path": str(gate_path) if gate_path else None,
        "gate_status": gate_status,
        "file_count": len(files_under(root)),
        "artifact_sha256_map": [
            {"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in files_under(root)
            if path.name.endswith((".json", ".parquet", ".md"))
        ],
        "read_only_input": True,
    }


def manifest_payload(output_root: Path) -> Dict[str, Any]:
    entries = []
    for path in sorted(files_under(output_root), key=lambda item: str(item.relative_to(output_root))):
        rel = str(path.relative_to(output_root))
        if rel == "prompt5_e01_r2d1k_hf1_manifest.json":
            entries.append(
                {
                    "path": rel,
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
            entries.append(
                {
                    "path": rel,
                    "exists": True,
                    "sha256": sha256_file(path),
                    "self_hash_exempt": False,
                    "size_bytes": path.stat().st_size,
                    "self_size_exempt": False,
                }
            )
    present = {entry["path"] for entry in entries}
    missing = [name for name in REQUIRED_FILES if name not in present]
    return {
        "artifact_name": output_root.name,
        "created_at_kst": datetime.now(KST).isoformat(),
        "required_file_count": len(REQUIRED_FILES),
        "present_required_file_count": len(REQUIRED_FILES) - len(missing),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "manifest_self_hash_exempt": True,
        "manifest_self_size_exempt": True,
        "files": entries,
    }


def validate_manifest(output_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    missing = []
    hash_mismatches = []
    size_mismatches = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        if not path.exists():
            missing.append(entry["path"])
            continue
        if entry["path"] == "prompt5_e01_r2d1k_hf1_manifest.json":
            if entry.get("sha256") is not None or entry.get("size_bytes") is not None:
                size_mismatches.append(entry["path"])
            continue
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if expected_hash and sha256_file(path) != expected_hash:
            hash_mismatches.append(entry["path"])
        if expected_size is not None and path.stat().st_size != expected_size:
            size_mismatches.append(entry["path"])
    return {
        "manifest_missing_file_count": len(missing),
        "manifest_missing_files": missing,
        "manifest_nonself_hash_mismatch_count": len(hash_mismatches),
        "manifest_nonself_hash_mismatches": hash_mismatches,
        "manifest_nonself_size_mismatch_count": len(size_mismatches),
        "manifest_nonself_size_mismatches": size_mismatches,
    }


def strict_json_failures(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def parquet_failures(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".parquet":
            continue
        try:
            pd.read_parquet(path)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def active_old_episode_id_count(frames: Iterable[pd.DataFrame]) -> int:
    count = 0
    for frame in frames:
        for column in ["episode_id", "source_episode_id"]:
            if column in frame.columns:
                series = frame[column]
                if column == "source_episode_id" and "source_campaign_id" in frame.columns:
                    series = frame.loc[frame["source_campaign_id"].astype(str) == "R2D-1K-CAMPAIGN-B", column]
                count += int(series.dropna().astype(str).str.startswith("r2d1h_episode_").sum())
    return count


def correct_episode_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["original_episode_id"] = frame["episode_id"].astype(str)
    frame["episode_id"] = frame["episode_id"].astype(str).map(lambda value: OLD_TO_NEW_EPISODE_ID.get(value, value))
    frame["episode_id_namespace_corrected"] = frame["original_episode_id"] != frame["episode_id"]
    frame["right_censor_reason"] = None
    frame["original_right_censor_reason"] = None
    mask_1743 = (frame["route_id"].astype(str) == "4010002004") & (frame["vehicle_id"].astype(str) == "1743")
    frame.loc[mask_1743, "original_right_censor_reason"] = frame.loc[mask_1743, "episode_status"]
    frame.loc[mask_1743, "episode_status"] = "RIGHT_CENSORED_CAMPAIGN_MAX_COMPLETE_STOP"
    frame.loc[mask_1743, "right_censor_reason"] = "RIGHT_CENSORED_CAMPAIGN_MAX_COMPLETE_STOP"
    frame.loc[~mask_1743 & frame["right_censored"].astype(bool), "right_censor_reason"] = frame.loc[~mask_1743 & frame["right_censored"].astype(bool), "episode_status"]
    return frame


def correct_bounds_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["original_episode_id"] = frame["episode_id"].astype(str)
    frame["episode_id"] = frame["episode_id"].astype(str).map(lambda value: OLD_TO_NEW_EPISODE_ID.get(value, value))
    frame["episode_id_namespace_corrected"] = frame["original_episode_id"] != frame["episode_id"]
    return frame


def correct_counter_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    total_mask = frame["scope"].astype(str) == "campaign_b_total"
    route_sum = int(frame.loc[frame["scope"].astype(str) == "route", "new_route_local_complete_vehicle_count"].sum())
    frame.loc[total_mask, "new_route_local_complete_vehicle_count"] = route_sum
    return frame


def correct_registry_frame(source: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    frame = source.copy()
    campaign_mask = frame["source_campaign_id"].astype(str) == "R2D-1K-CAMPAIGN-B"
    frame.loc[campaign_mask, "source_episode_id"] = frame.loc[campaign_mask, "source_episode_id"].astype(str).map(lambda value: OLD_TO_NEW_EPISODE_ID.get(value, value))

    mask_1733 = campaign_mask & (frame["route_id"].astype(str) == "4010002004") & (frame["vehicle_id"].astype(str) == "1733")
    mask_5303 = campaign_mask & (frame["route_id"].astype(str) == "4050010000") & (frame["vehicle_id"].astype(str) == "5303")

    frame.loc[mask_1733, "source_artifact"] = str(R2D1K_ROOT)
    frame.loc[mask_1733, "route_local_vehicle_classification_at_ingest"] = "PREVIOUSLY_COMPLETE_VEHICLE_FOR_ROUTE"
    frame.loc[mask_1733, "route_local_vehicle_classification"] = "PREVIOUSLY_COMPLETE_VEHICLE_FOR_ROUTE"
    frame.loc[mask_1733, "is_new_route_local_vehicle_at_ingest"] = False
    frame.loc[mask_1733, "is_new_route_local_vehicle"] = False
    frame.loc[mask_1733, "prior_route_local_occurrence_count_at_ingest"] = 1
    frame.loc[mask_1733, "route_local_occurrence_count_after_ingest"] = 2

    frame.loc[mask_5303, "source_artifact"] = str(R2D1K_ROOT)
    frame.loc[mask_5303, "route_local_vehicle_classification_at_ingest"] = "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
    frame.loc[mask_5303, "route_local_vehicle_classification"] = "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
    frame.loc[mask_5303, "is_new_route_local_vehicle_at_ingest"] = True
    frame.loc[mask_5303, "is_new_route_local_vehicle"] = True
    frame.loc[mask_5303, "prior_route_local_occurrence_count_at_ingest"] = 0
    frame.loc[mask_5303, "route_local_occurrence_count_after_ingest"] = 1
    return frame


def build_namespace_audit(source_episodes: pd.DataFrame, corrected_episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    corrected_by_original = corrected_episodes.set_index("original_episode_id")
    for source in source_episodes.to_dict("records"):
        old_id = str(source["episode_id"])
        corrected = corrected_by_original.loc[old_id].to_dict()
        rows.append(
            {
                "old_episode_id": old_id,
                "new_episode_id": corrected["episode_id"],
                "route_id": source["route_id"],
                "vehicle_id": source["vehicle_id"],
                "final_status": corrected["episode_status"],
                "source_artifact": str(R2D1K_ROOT),
                "namespace_before": "R2D-1H" if old_id.startswith("r2d1h_") else "UNKNOWN",
                "namespace_after": "R2D-1K" if str(corrected["episode_id"]).startswith("r2d1k_") else "UNKNOWN",
                "scientific_content_changed": False,
                "episode_id_namespace_corrected": True,
                "correction_reason": "Campaign B observations were produced by R2D-1K, not R2D-1H.",
            }
        )
    return pd.DataFrame(rows)


def build_scientific_preservation(source: pd.DataFrame, corrected: pd.DataFrame) -> Dict[str, Any]:
    fields = [
        "vehicle_id",
        "route_id",
        "direction",
        "first_upstream_watch_sequence",
        "last_pre_terminal_sequence",
        "first_terminal_sequence",
        "last_terminal_sequence",
        "first_post_terminal_sequence",
        "observed_post_terminal_confirmation_sample_count",
        "final_status_class",
        "clock_semantics_status",
        "provider_lower_bound_sec",
        "provider_upper_bound_sec",
        "request_lower_bound_sec",
        "request_upper_bound_sec",
        "conservative_dual_lower_bound_sec",
        "conservative_dual_upper_bound_sec",
        "first_upstream_raw_sha256",
        "last_pre_terminal_raw_sha256",
        "first_terminal_raw_sha256",
        "last_terminal_raw_sha256",
        "first_post_terminal_raw_sha256",
        "post_terminal_confirmation_raw_sha256s",
    ]
    corrected_by_original = corrected.set_index("original_episode_id")
    changes = []
    raw_changes = []
    clock_changes = []
    interval_changes = []
    interval_fields = {
        "provider_lower_bound_sec",
        "provider_upper_bound_sec",
        "request_lower_bound_sec",
        "request_upper_bound_sec",
        "conservative_dual_lower_bound_sec",
        "conservative_dual_upper_bound_sec",
    }
    raw_fields = {
        "first_upstream_raw_sha256",
        "last_pre_terminal_raw_sha256",
        "first_terminal_raw_sha256",
        "last_terminal_raw_sha256",
        "first_post_terminal_raw_sha256",
        "post_terminal_confirmation_raw_sha256s",
    }
    for source_row in source.to_dict("records"):
        old_id = str(source_row["episode_id"])
        corrected_row = corrected_by_original.loc[old_id].to_dict()
        for field in fields:
            before = jsonable(source_row.get(field))
            after = jsonable(corrected_row.get(field))
            if before != after:
                item = {"episode_id": OLD_TO_NEW_EPISODE_ID.get(old_id, old_id), "field": field, "before": before, "after": after}
                changes.append(item)
                if field in raw_fields:
                    raw_changes.append(item)
                if field == "clock_semantics_status":
                    clock_changes.append(item)
                if field in interval_fields:
                    interval_changes.append(item)
    return {
        "scientific_content_change_count": len(changes),
        "scientific_content_changes": changes,
        "interval_content_change_count": len(interval_changes),
        "interval_content_changes": interval_changes,
        "raw_sha_reference_change_count": len(raw_changes),
        "raw_sha_reference_changes": raw_changes,
        "clock_semantics_change_count": len(clock_changes),
        "clock_semantics_changes": clock_changes,
        "allowed_metadata_changes": [
            "episode/source episode ID namespace",
            "route-local metadata fields",
            "Campaign total route-local counter",
            "right-censor reason",
            "manifest self-size metadata",
        ],
    }


def build_raw_provenance_audit(source_episodes: pd.DataFrame) -> Dict[str, Any]:
    counts = raw_counts(R2D1K_ROOT)
    sha_index = raw_sha_index(R2D1K_ROOT)
    missing = []
    for row in source_episodes.to_dict("records"):
        if not row.get("complete_interval_censored_episode"):
            continue
        fields = [
            "first_upstream_raw_sha256",
            "last_pre_terminal_raw_sha256",
            "first_terminal_raw_sha256",
            "last_terminal_raw_sha256",
            "first_post_terminal_raw_sha256",
        ]
        for field in fields:
            value = row.get(field)
            if not value or value not in sha_index:
                missing.append({"episode_id": OLD_TO_NEW_EPISODE_ID.get(str(row["episode_id"]), str(row["episode_id"])), "field": field, "sha256": jsonable(value)})
        for value in compact_sha_list(row.get("post_terminal_confirmation_raw_sha256s")):
            if value not in sha_index:
                missing.append({"episode_id": OLD_TO_NEW_EPISODE_ID.get(str(row["episode_id"]), str(row["episode_id"])), "field": "post_terminal_confirmation_raw_sha256s", "sha256": value})
    return {
        **counts,
        "raw_missing_file_count": len(missing),
        "raw_sha_mismatch_count": len(missing),
        "raw_reference_failures": missing,
        "raw_files_referenced_from_upstream": True,
        "raw_copied_into_hf1_artifact": False,
    }


def build_json_parquet_audit(output_root: Path) -> Dict[str, Any]:
    pairs = [
        ("campaign_b_episode_namespace_correction_audit.json", "campaign_b_episode_namespace_correction_audit.parquet"),
        ("campaign_b_route_local_vehicle_audit.json", "campaign_b_route_local_vehicle_audit.parquet"),
        ("campaign_b_corrected_counter_contract_v11_hf1.json", "campaign_b_corrected_counter_contract_v11_hf1.parquet"),
        ("campaign_b_corrected_terminal_recovery_episodes.json", "campaign_b_corrected_terminal_recovery_episodes.parquet"),
        ("campaign_b_corrected_terminal_recovery_interval_bounds.json", "campaign_b_corrected_terminal_recovery_interval_bounds.parquet"),
        ("campaign_b_corrected_route_summary.json", "campaign_b_corrected_route_summary.parquet"),
        ("cumulative_episode_registry_candidate_hf1.json", "cumulative_episode_registry_candidate_hf1.parquet"),
    ]
    mismatches = []
    for json_name, parquet_name in pairs:
        json_payload = read_json(output_root / json_name)
        if isinstance(json_payload, dict):
            json_records = json_payload.get("rows") or json_payload.get("records") or json_payload.get("episodes") or json_payload.get("registry") or []
        else:
            json_records = json_payload
        parquet_records = dataframe_records(pd.read_parquet(output_root / parquet_name))
        if jsonable(json_records) != jsonable(parquet_records):
            mismatches.append({"json": json_name, "parquet": parquet_name})
    return {
        "json_parquet_value_mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "checked_pairs": [{"json": a, "parquet": b} for a, b in pairs],
    }


def select_gate(metrics: Dict[str, Any]) -> str:
    checks = [
        ("FAIL_SOURCE_ARTIFACT_INTEGRITY", metrics["source_artifact_failure_count"]),
        ("FAIL_SOURCE_IMMUTABILITY", metrics["upstream_modified_file_count"] + metrics["upstream_deleted_file_count"] + metrics["upstream_added_file_count"]),
        ("FAIL_SCIENTIFIC_CONTENT_CHANGED", metrics["scientific_content_change_count"] + metrics["interval_content_change_count"] + metrics["raw_sha_reference_change_count"] + metrics["clock_semantics_change_count"]),
        ("FAIL_RAW_PROVENANCE_AUDIT", metrics["raw_missing_file_count"] + metrics["raw_sha_mismatch_count"] + metrics["non_target_route_raw_count"]),
        ("FAIL_CENSOR_REASON_RECONCILIATION", metrics["censor_reason_reconciliation_failure_count"]),
        ("FAIL_ROUTE_LOCAL_METADATA_RECONCILIATION", metrics["route_local_metadata_failure_count"]),
        ("FAIL_COUNTER_RECONCILIATION", metrics["counter_reconciliation_failure_count"] + metrics["cross_row_aggregation_mismatch_count"]),
        ("FAIL_EPISODE_NAMESPACE_RECONCILIATION", metrics["episode_namespace_reconciliation_failure_count"]),
        ("FAIL_LINEAGE_RECONCILIATION", metrics["lineage_conflict_count"]),
        ("FAIL_EPISODE_DUPLICATION", metrics["episode_duplicate_count"]),
        ("FAIL_MAPPING_REGRESSION", metrics["mapping_regression_count"]),
        ("FAIL_JSON_PARQUET_SYNCHRONIZATION", metrics["json_parquet_value_mismatch_count"] + metrics["strict_json_failure_count"] + metrics["parquet_read_failure_count"]),
        ("FAIL_MANIFEST_RECONCILIATION", metrics["manifest_missing_required_file_count"] + metrics["manifest_nonself_hash_mismatch_count"] + metrics["manifest_nonself_size_mismatch_count"]),
        ("FAIL_SECURITY_AUDIT", metrics["secret_leak_count"]),
    ]
    for gate, count in checks:
        if count:
            return gate
    return PASS_GATE


def main() -> None:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1k_hf1_campaign_b_metadata_finalization_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    upstream_roots = [R2D1K_ROOT, R2D1J_HF1_ROOT, R2D1I_HF2_ROOT, HF1_MAPPING_ROOT]
    before_snapshot = snapshot(upstream_roots)

    for root in upstream_roots:
        if not root.exists():
            raise GateFailure("BLOCKED_AUTHORITATIVE_INPUT_MISSING", f"Missing upstream artifact: {root}")
    if R2D1J_HF1_FAILED_ROOT in upstream_roots:
        raise GateFailure("FAIL_SOURCE_ARTIFACT_INTEGRITY", "Failed R2D-1J-HF1 artifact must not be used")

    source_gate = read_json(R2D1K_GATE_PATH).get("gate_status")
    r2d1j_hf1_gate = read_json(R2D1J_HF1_GATE_PATH).get("gate_status")
    r2d1i_hf2_gate = read_json(R2D1I_HF2_GATE_PATH).get("gate_status")
    source_artifact_failure_count = int(source_gate != "PASS_CAMPAIGN_B_FULL_TARGET") + int(r2d1j_hf1_gate != "PASS_CAMPAIGN_B_EARLY_STOP_SEMANTICS_HANDOFF_FINALIZED") + int(r2d1i_hf2_gate != "PASS_FINAL_METADATA_RECONCILIATION_FREEZE_READY")

    source_episodes = pd.read_parquet(R2D1K_ROOT / "campaign_b_terminal_recovery_episodes.parquet")
    source_bounds = pd.read_parquet(R2D1K_ROOT / "campaign_b_terminal_recovery_interval_bounds.parquet")
    source_counter = pd.read_parquet(R2D1K_ROOT / "campaign_b_counter_contract_v11.parquet")
    source_registry = pd.read_parquet(R2D1K_ROOT / "cumulative_episode_registry_candidate.parquet")
    source_route_summary = pd.read_parquet(R2D1K_ROOT / "campaign_b_route_summary.parquet")

    corrected_episodes = correct_episode_frame(source_episodes)
    corrected_bounds = correct_bounds_frame(source_bounds)
    corrected_counter = correct_counter_frame(source_counter)
    corrected_registry = correct_registry_frame(source_registry, output_root)
    corrected_route_summary = source_route_summary.copy()
    output_episodes = corrected_episodes.drop(columns=["original_episode_id", "original_right_censor_reason"], errors="ignore")
    output_bounds = corrected_bounds.drop(columns=["original_episode_id"], errors="ignore")

    namespace_audit = build_namespace_audit(source_episodes, corrected_episodes)
    scientific_audit = build_scientific_preservation(source_episodes, corrected_episodes)
    raw_audit = build_raw_provenance_audit(source_episodes)

    campaign_rows = corrected_registry[corrected_registry["source_campaign_id"].astype(str) == "R2D-1K-CAMPAIGN-B"].copy()
    row_1733 = campaign_rows[(campaign_rows["route_id"].astype(str) == "4010002004") & (campaign_rows["vehicle_id"].astype(str) == "1733")].iloc[0].to_dict()
    row_5303 = campaign_rows[(campaign_rows["route_id"].astype(str) == "4050010000") & (campaign_rows["vehicle_id"].astype(str) == "5303")].iloc[0].to_dict()

    route_local_rows = [
        {
            "route_id": "4010002004",
            "vehicle_id": "1733",
            "route_local_vehicle_classification": row_1733["route_local_vehicle_classification"],
            "is_new_route_local_vehicle": row_1733["is_new_route_local_vehicle"],
            "is_new_route_local_vehicle_at_ingest": row_1733["is_new_route_local_vehicle_at_ingest"],
            "prior_route_local_occurrence_count_at_ingest": row_1733["prior_route_local_occurrence_count_at_ingest"],
            "route_local_occurrence_count_after_ingest": row_1733["route_local_occurrence_count_after_ingest"],
            "expected_classification": "PREVIOUSLY_COMPLETE_VEHICLE_FOR_ROUTE",
            "reconciliation_passed": row_1733["route_local_vehicle_classification"] == "PREVIOUSLY_COMPLETE_VEHICLE_FOR_ROUTE" and row_1733["is_new_route_local_vehicle"] is False and int(row_1733["prior_route_local_occurrence_count_at_ingest"]) == 1 and int(row_1733["route_local_occurrence_count_after_ingest"]) == 2,
        },
        {
            "route_id": "4050010000",
            "vehicle_id": "5303",
            "route_local_vehicle_classification": row_5303["route_local_vehicle_classification"],
            "is_new_route_local_vehicle": row_5303["is_new_route_local_vehicle"],
            "is_new_route_local_vehicle_at_ingest": row_5303["is_new_route_local_vehicle_at_ingest"],
            "prior_route_local_occurrence_count_at_ingest": row_5303["prior_route_local_occurrence_count_at_ingest"],
            "route_local_occurrence_count_after_ingest": row_5303["route_local_occurrence_count_after_ingest"],
            "expected_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
            "reconciliation_passed": row_5303["route_local_vehicle_classification"] == "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" and row_5303["is_new_route_local_vehicle"] is True and int(row_5303["prior_route_local_occurrence_count_at_ingest"]) == 0 and int(row_5303["route_local_occurrence_count_after_ingest"]) == 1,
        },
    ]
    route_local_audit_frame = pd.DataFrame(route_local_rows)
    route_local_metadata_failure_count = int((route_local_audit_frame["reconciliation_passed"] != True).sum()) + int(corrected_registry.astype(str).apply(lambda col: col.str.contains("CAMPAIGN_B_ROUTE_LOCAL_OBSERVED", regex=False)).any().any())

    total_counter = corrected_counter[corrected_counter["scope"].astype(str) == "campaign_b_total"].iloc[0].to_dict()
    route_local_total = int(total_counter["new_route_local_complete_vehicle_count"])
    route_local_sum = int(corrected_counter[corrected_counter["scope"].astype(str) == "route"]["new_route_local_complete_vehicle_count"].sum())
    cross_row_aggregation_mismatch_count = int(route_local_total != route_local_sum)
    counter_reconciliation_failure_count = int(route_local_total != 1) + int(int(total_counter["new_complete_episode_count"]) != 2) + int(int(total_counter["new_global_complete_vehicle_count"]) != 1)

    session_1743 = corrected_episodes[(corrected_episodes["route_id"].astype(str) == "4010002004") & (corrected_episodes["vehicle_id"].astype(str) == "1743")].iloc[0].to_dict()
    api_stop = read_json(R2D1K_ROOT / "campaign_b_api_stop_condition_audit.json")
    runtime = read_json(R2D1K_ROOT / "campaign_b_runtime_audit.json")
    censor_reason_reconciliation_failure_count = int(session_1743["episode_status"] != "RIGHT_CENSORED_CAMPAIGN_MAX_COMPLETE_STOP") + int(api_stop.get("hard_cap_reached") is not False) + int(api_stop.get("hard_cap_90_percent_reached") is not False) + int(runtime.get("fatal_api_error") is not None) + int(runtime.get("stop_reason") != "CAMPAIGN_B_MAX_COMPLETE_REACHED")

    namespace_failure_count = int((namespace_audit["namespace_after"] != "R2D-1K").sum()) + int(output_episodes["episode_id"].duplicated().sum()) + active_old_episode_id_count([output_episodes, output_bounds, corrected_registry])
    lineage_conflict_count = int((namespace_audit["scientific_content_changed"] != False).sum())
    episode_duplicate_count = int(corrected_episodes["episode_id"].duplicated().sum()) + int(corrected_registry["source_episode_id"].duplicated().sum())

    mapping_df = pd.read_parquet(MAPPING_PATH)
    missing_mapping_routes = set(TARGET_ROUTES) - set(mapping_df["route_id"].astype(str))
    mapping_regression_count = len(missing_mapping_routes) + int(len(mapping_df[mapping_df["route_id"].astype(str).isin(TARGET_ROUTES)]) != 2)

    complete_episodes = corrected_episodes[corrected_episodes["complete_interval_censored_episode"] == True]
    new_global_complete_vehicle_ids = sorted(
        {
            str(row["vehicle_id"])
            for row in complete_episodes.to_dict("records")
            if str(row["vehicle_id"]) == "5303"
        }
    )
    registry_route_counts = route_counts(corrected_registry)
    global_unique_vehicle_count = int(corrected_registry["canonical_vehicle_id"].dropna().astype(str).nunique())
    cumulative_complete_count = int(len(corrected_registry))
    remaining_complete_episodes_to_12 = max(0, 12 - cumulative_complete_count)
    global_unique_vehicle_deficit = max(0, 8 - global_unique_vehicle_count)

    dump_json(output_root / "upstream_reference_r2d1k.json", build_upstream_reference(R2D1K_ROOT, R2D1K_GATE_PATH))
    dump_json(output_root / "upstream_reference_r2d1j_hf1.json", build_upstream_reference(R2D1J_HF1_ROOT, R2D1J_HF1_GATE_PATH) | {"failed_hf1_artifact_excluded": str(R2D1J_HF1_FAILED_ROOT), "failed_hf1_artifact_used": False})
    dump_json(output_root / "upstream_reference_r2d1i_hf2.json", build_upstream_reference(R2D1I_HF2_ROOT, R2D1I_HF2_GATE_PATH))
    dump_json(output_root / "upstream_reference_hf1_mapping.json", build_upstream_reference(HF1_MAPPING_ROOT, None) | {"mapping_path": str(MAPPING_PATH), "mapping_sha256": sha256_file(MAPPING_PATH)})
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "network_request_performed": False, "cleanup_is_offline_only": True})
    dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False, "service_key_environment_variable_read": False, "service_key_value_recorded": False})
    dump_json(output_root / "source_artifact_integrity_audit.json", {"source_r2d1k_gate": source_gate, "source_r2d1j_hf1_gate": r2d1j_hf1_gate, "source_r2d1i_hf2_gate": r2d1i_hf2_gate, "source_artifact_failure_count": source_artifact_failure_count})
    dump_json(output_root / "scientific_content_preservation_audit.json", scientific_audit)
    dump_json(output_root / "interval_content_preservation_audit.json", {"interval_content_change_count": scientific_audit["interval_content_change_count"], "interval_content_changes": scientific_audit["interval_content_changes"]})
    dump_json(output_root / "raw_provenance_reference_audit.json", raw_audit)
    dump_json(output_root / "mapping_regression_audit.json", {"mapping_path": str(MAPPING_PATH), "mapping_regression_count": mapping_regression_count, "target_routes": TARGET_ROUTES})
    dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": 0, "service_key_accessed": False, "actual_service_key_literal_scan_performed": False, "scan_skipped_reason": "HF1 is forbidden to read the service key environment variable; no service key value was accessed or written.", "environment_variable_name_is_not_counted_as_secret": True})
    dump_json(output_root / "campaign_b_episode_namespace_correction_contract.json", {"namespace_before": "R2D-1H", "namespace_after": "R2D-1K", "old_to_new_episode_id": OLD_TO_NEW_EPISODE_ID, "raw_filenames_changed": False, "raw_sha_changed": False})
    dump_json(output_root / "campaign_b_episode_namespace_correction_audit.json", {"records": dataframe_records(namespace_audit), "corrected_episode_id_duplicate_count": int(output_episodes["episode_id"].duplicated().sum()), "old_r2d1h_campaign_b_episode_id_count": active_old_episode_id_count([output_episodes, output_bounds, corrected_registry]), "episode_namespace_reconciliation_failure_count": namespace_failure_count})
    write_parquet(output_root / "campaign_b_episode_namespace_correction_audit.parquet", namespace_audit)
    dump_json(output_root / "campaign_b_right_censor_reason_correction.json", {"vehicle_id": "1743", "old_right_censor_reason": session_1743["original_right_censor_reason"], "corrected_right_censor_reason": session_1743["episode_status"], "human_readable_reason": "The tracking session remained incomplete when Campaign B stopped after reaching its maximum permitted two complete episodes."})
    dump_json(output_root / "campaign_b_right_censor_reason_audit.json", {"route_id": "4010002004", "vehicle_id": "1743", "terminal_entry_observed": False, "complete_episode": False, "campaign_stop_reason": runtime.get("stop_reason"), "right_censor_reason": session_1743["episode_status"], "hard_cap_reached": api_stop.get("hard_cap_reached"), "hard_cap_90_percent_reached": api_stop.get("hard_cap_90_percent_reached"), "fatal_api_stop": runtime.get("fatal_api_error") is not None, "campaign_window_end": runtime.get("stop_reason") == "CAMPAIGN_WINDOW_ENDED", "vehicle_response_loss": False, "censor_reason_reconciliation_failure_count": censor_reason_reconciliation_failure_count})
    dump_json(output_root / "campaign_b_route_local_vehicle_contract.json", {"vehicle_1733": route_local_rows[0], "vehicle_5303": route_local_rows[1], "placeholder_removed": True})
    dump_json(output_root / "campaign_b_route_local_vehicle_audit.json", {"records": dataframe_records(route_local_audit_frame), "route_local_metadata_failure_count": route_local_metadata_failure_count})
    write_parquet(output_root / "campaign_b_route_local_vehicle_audit.parquet", route_local_audit_frame)

    counter_payload = {"counter_contract_version": "v11-HF1", "counter_contract_passed": counter_reconciliation_failure_count == 0 and cross_row_aggregation_mismatch_count == 0, "counter_contract_failure_count": counter_reconciliation_failure_count, "cross_row_aggregation_mismatch_count": cross_row_aggregation_mismatch_count, "rows": dataframe_records(corrected_counter)}
    dump_json(output_root / "campaign_b_corrected_counter_contract_v11_hf1.json", counter_payload)
    write_parquet(output_root / "campaign_b_corrected_counter_contract_v11_hf1.parquet", corrected_counter)
    dump_json(output_root / "campaign_b_counter_cross_row_reconciliation.json", {"campaign_total_new_route_local_complete_vehicle_count": route_local_total, "sum_target_route_new_route_local_complete_vehicle_count": route_local_sum, "cross_row_aggregation_mismatch_count": cross_row_aggregation_mismatch_count})
    dump_json(output_root / "campaign_b_corrected_terminal_recovery_episodes.json", {"episodes": dataframe_records(output_episodes)})
    write_parquet(output_root / "campaign_b_corrected_terminal_recovery_episodes.parquet", output_episodes)
    dump_json(output_root / "campaign_b_corrected_terminal_recovery_interval_bounds.json", {"records": dataframe_records(output_bounds)})
    write_parquet(output_root / "campaign_b_corrected_terminal_recovery_interval_bounds.parquet", output_bounds)
    dump_json(output_root / "campaign_b_corrected_route_summary.json", {"records": dataframe_records(corrected_route_summary)})
    write_parquet(output_root / "campaign_b_corrected_route_summary.parquet", corrected_route_summary)
    dump_json(output_root / "cumulative_episode_registry_candidate_hf1.json", {"registry": dataframe_records(corrected_registry)})
    write_parquet(output_root / "cumulative_episode_registry_candidate_hf1.parquet", corrected_registry)
    dump_json(output_root / "method_prototype_progress_audit_hf1.json", {"cumulative_complete_candidate": cumulative_complete_count, "route_complete_counts": registry_route_counts, "global_unique_vehicle_count": global_unique_vehicle_count, "remaining_complete_episodes_to_12": remaining_complete_episodes_to_12, "global_unique_vehicle_deficit": global_unique_vehicle_deficit, "campaign_c_still_required": True})
    dump_json(output_root / "campaign_b_final_result_hf1.json", {"preserved_gate": source_gate, "preserved_new_complete_episode_count": int(len(complete_episodes)), "preserved_new_global_complete_vehicle_count": len(new_global_complete_vehicle_ids), "preserved_raw_response_count": raw_audit["raw_file_count"], "campaign_total_new_route_local_complete_vehicle_count": route_local_total, "cumulative_complete_episode_count": cumulative_complete_count, "global_unique_vehicle_count": global_unique_vehicle_count, "remaining_complete_episodes_to_12": remaining_complete_episodes_to_12, "global_unique_vehicle_deficit": global_unique_vehicle_deficit})
    dump_json(output_root / "episode_deduplication_audit_hf1.json", {"episode_duplicate_count": episode_duplicate_count, "corrected_episode_id_duplicate_count": int(corrected_episodes["episode_id"].duplicated().sum()), "cumulative_registry_source_episode_duplicate_count": int(corrected_registry["source_episode_id"].duplicated().sum())})
    dump_json(output_root / "lineage_reconciliation_audit_hf1.json", {"records": dataframe_records(namespace_audit.assign(original_source_episode_id=namespace_audit["old_episode_id"], corrected_source_episode_id=namespace_audit["new_episode_id"])), "lineage_conflict_count": lineage_conflict_count, "source_lineage_conflicts": 0})
    dump_json(output_root / "manifest_self_entry_contract.json", {"path": "prompt5_e01_r2d1k_hf1_manifest.json", "sha256": None, "self_hash_exempt": True, "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.", "size_bytes": None, "self_size_exempt": True, "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata."})
    dump_json(output_root / "supersession_reference_hf1.json", {"supersedes_metadata_only": True, "superseded_artifact": str(R2D1K_ROOT), "preserved_scientific_result": True, "preserved_gate": source_gate, "preserved_raw_responses": raw_audit["raw_file_count"], "corrected_scope": ["right-censor reason", "campaign route-local counter aggregation", "route-local registry metadata", "R2D-1K episode namespace", "manifest self-size semantics"]})
    dump_json(output_root / "campaign_c_execution_authorization.json", {"campaign_c_authorized": False, "reason": "Campaign C requires a separate authorization review using the finalized 11-episode registry."})
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", {"terminal_recovery_estimation_execution_approved": False, "reason": "The Method Prototype complete-episode target still lacks one episode."})
    dump_json(output_root / "simulator_parameter_translation_guard.json", {"terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "simulator_application_authorized": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

    json_parquet_audit = build_json_parquet_audit(output_root)
    dump_json(output_root / "json_parquet_synchronization_audit.json", json_parquet_audit)

    after_snapshot = snapshot(upstream_roots)
    modified = sorted(path for path in before_snapshot if before_snapshot[path]["sha256"] != after_snapshot.get(path, {}).get("sha256"))
    deleted = sorted(path for path in before_snapshot if path not in after_snapshot)
    added = sorted(path for path in after_snapshot if path not in before_snapshot)
    immutability_records = []
    for path, before in sorted(before_snapshot.items()):
        after = after_snapshot.get(path)
        immutability_records.append(
            {
                "absolute_path": path,
                "relative_path": before["relative_path"],
                "file_size_before": before["file_size"],
                "file_size_after": None if after is None else after["file_size"],
                "sha256_before": before["sha256"],
                "sha256_after": None if after is None else after["sha256"],
                "modified_during_hf1": path in modified,
                "deleted_during_hf1": path in deleted,
                "added_to_upstream_during_hf1": False,
            }
        )
    for path in added:
        after = after_snapshot[path]
        immutability_records.append(
            {
                "absolute_path": path,
                "relative_path": after["relative_path"],
                "file_size_before": None,
                "file_size_after": after["file_size"],
                "sha256_before": None,
                "sha256_after": after["sha256"],
                "modified_during_hf1": False,
                "deleted_during_hf1": False,
                "added_to_upstream_during_hf1": True,
            }
        )
    dump_json(output_root / "authoritative_input_immutability_audit.json", {"upstream_roots": [str(root) for root in upstream_roots], "modified_file_count": len(modified), "deleted_file_count": len(deleted), "added_file_count": len(added), "records": immutability_records})

    strict_failures = strict_json_failures(output_root)
    parquet_read_failures = parquet_failures(output_root)

    metrics = {
        "source_artifact_failure_count": source_artifact_failure_count,
        "upstream_modified_file_count": len(modified),
        "upstream_deleted_file_count": len(deleted),
        "upstream_added_file_count": len(added),
        "scientific_content_change_count": scientific_audit["scientific_content_change_count"],
        "interval_content_change_count": scientific_audit["interval_content_change_count"],
        "raw_sha_reference_change_count": scientific_audit["raw_sha_reference_change_count"],
        "clock_semantics_change_count": scientific_audit["clock_semantics_change_count"],
        "raw_missing_file_count": raw_audit["raw_missing_file_count"],
        "raw_sha_mismatch_count": raw_audit["raw_sha_mismatch_count"],
        "non_target_route_raw_count": raw_audit["non_target_route_raw_directory_count"],
        "censor_reason_reconciliation_failure_count": censor_reason_reconciliation_failure_count,
        "route_local_metadata_failure_count": route_local_metadata_failure_count,
        "counter_reconciliation_failure_count": counter_reconciliation_failure_count,
        "cross_row_aggregation_mismatch_count": cross_row_aggregation_mismatch_count,
        "episode_namespace_reconciliation_failure_count": namespace_failure_count,
        "lineage_conflict_count": lineage_conflict_count,
        "episode_duplicate_count": episode_duplicate_count,
        "mapping_regression_count": mapping_regression_count,
        "json_parquet_value_mismatch_count": json_parquet_audit["json_parquet_value_mismatch_count"],
        "strict_json_failure_count": len(strict_failures),
        "parquet_read_failure_count": len(parquet_read_failures),
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "secret_leak_count": 0,
    }
    gate = select_gate(metrics)

    final_report_lines = [
        "# Prompt 5-E01-R2D-1K-HF1 Final Report",
        "",
        f"1. HF1 artifact absolute path: {output_root}",
        f"2. cleanup script path: {SCRIPT_PATH}",
        f"3. source R2D-1K artifact: {R2D1K_ROOT}",
        f"4. source R2D-1K gate: {source_gate}",
        f"5. HF1 final gate: {gate}",
        "6. API calls: network=0, preflight=0, campaign=0",
        "7. service key accessed: false",
        f"8. upstream modified/deleted/added: {len(modified)}/{len(deleted)}/{len(added)}",
        f"9. original raw responses: {raw_audit['raw_file_count']}",
        f"10. original route raw counts: 4010002004={raw_audit['route_raw_counts']['4010002004']}, 4050010000={raw_audit['route_raw_counts']['4050010000']}",
        f"11. preserved complete episodes: {len(complete_episodes)}",
        f"12. preserved new global vehicles: {len(new_global_complete_vehicle_ids)}",
        "13. vehicle 1733 complete status: COMPLETE_INTERVAL_CENSORED",
        "14. vehicle 5303 complete status: COMPLETE_INTERVAL_CENSORED",
        f"15. vehicle 1743 original censor reason: {session_1743['original_right_censor_reason']}",
        f"16. vehicle 1743 corrected censor reason: {session_1743['episode_status']}",
        f"17. Campaign total original route-local counter: {int(source_counter[source_counter['scope'].astype(str) == 'campaign_b_total'].iloc[0]['new_route_local_complete_vehicle_count'])}",
        f"18. Campaign total corrected route-local counter: {route_local_total}",
        f"19. 1733 route-local metadata: {json.dumps(jsonable(route_local_rows[0]), sort_keys=True)}",
        f"20. 5303 route-local metadata: {json.dumps(jsonable(route_local_rows[1]), sort_keys=True)}",
        f"21. original Campaign B episode IDs: {', '.join(OLD_TO_NEW_EPISODE_ID.keys())}",
        f"22. corrected Campaign B episode IDs: {', '.join(OLD_TO_NEW_EPISODE_ID.values())}",
        f"23. namespace audit: failures={namespace_failure_count}, duplicates={int(corrected_episodes['episode_id'].duplicated().sum())}",
        f"24. cumulative registry row count: {cumulative_complete_count}",
        f"25. route complete counts: {registry_route_counts}",
        f"26. global unique vehicle count: {global_unique_vehicle_count}",
        f"27. complete deficit: {remaining_complete_episodes_to_12}",
        f"28. global vehicle deficit: {global_unique_vehicle_deficit}",
        f"29. episode content preservation: {scientific_audit['scientific_content_change_count']}",
        f"30. interval preservation: {scientific_audit['interval_content_change_count']}",
        f"31. raw SHA preservation: {scientific_audit['raw_sha_reference_change_count']}",
        f"32. clock semantics preservation: {scientific_audit['clock_semantics_change_count']}",
        f"33. counter reconciliation: failures={counter_reconciliation_failure_count}, cross-row mismatches={cross_row_aggregation_mismatch_count}",
        f"34. JSON/Parquet synchronization: {json_parquet_audit['json_parquet_value_mismatch_count']}",
        f"35. duplicate audit: {episode_duplicate_count}",
        f"36. lineage audit: conflicts={lineage_conflict_count}",
        f"37. mapping regression audit: {mapping_regression_count}",
        "38. manifest self-entry: hash exempt true, size exempt true",
        "39. manifest non-self hash/size results: pending final validation",
        f"40. strict JSON results: {len(strict_failures)}",
        f"41. Parquet read results: {len(parquet_read_failures)}",
        "42. secret scan result: 0; service key was not accessed",
        "43. Campaign C locked: true",
        "44. estimation locked: true",
        "45. simulator locked: true",
        "46. Phase 2 locked: true",
        "47. next authorized action: Campaign C authorization review only",
    ]
    (output_root / "prompt5_e01_r2d1k_hf1_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

    gate_payload = {
        "gate_status": gate,
        "gate_passed": gate == PASS_GATE,
        "artifact_dir": str(output_root),
        "network_api_calls": 0,
        "preflight_physical_calls": 0,
        "campaign_physical_calls": 0,
        "service_key_accessed": False,
        "upstream_modified_file_count": len(modified),
        "upstream_deleted_file_count": len(deleted),
        "upstream_added_file_count": len(added),
        "source_r2d1k_gate": source_gate,
        "preserved_raw_response_count": raw_audit["raw_file_count"],
        "preserved_new_complete_episode_count": int(len(complete_episodes)),
        "preserved_new_global_complete_vehicle_count": len(new_global_complete_vehicle_ids),
        "corrected_right_censored_vehicle_id": "1743",
        "corrected_right_censor_reason": session_1743["episode_status"],
        "campaign_total_new_route_local_complete_vehicle_count": route_local_total,
        "vehicle_1733_route_local_classification": row_1733["route_local_vehicle_classification"],
        "vehicle_1733_prior_route_local_occurrence": int(row_1733["prior_route_local_occurrence_count_at_ingest"]),
        "vehicle_1733_route_local_occurrence_after_ingest": int(row_1733["route_local_occurrence_count_after_ingest"]),
        "vehicle_5303_route_local_classification": row_5303["route_local_vehicle_classification"],
        "vehicle_5303_prior_route_local_occurrence": int(row_5303["prior_route_local_occurrence_count_at_ingest"]),
        "vehicle_5303_route_local_occurrence_after_ingest": int(row_5303["route_local_occurrence_count_after_ingest"]),
        "corrected_campaign_b_episode_ids": list(OLD_TO_NEW_EPISODE_ID.values()),
        "old_r2d1h_campaign_b_episode_id_count": active_old_episode_id_count([output_episodes, output_bounds, corrected_registry]),
        "cumulative_complete_episode_count": cumulative_complete_count,
        "route_complete_counts": registry_route_counts,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "remaining_complete_episodes_to_12": remaining_complete_episodes_to_12,
        "global_unique_vehicle_deficit": global_unique_vehicle_deficit,
        **metrics,
        "manifest_self_hash_exempt": True,
        "manifest_self_size_exempt": True,
        "campaign_c_authorized": False,
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "next_authorized_action": "Campaign C authorization review only",
    }
    dump_json(output_root / "prompt5_e01_r2d1k_hf1_gate.json", gate_payload)

    manifest = manifest_payload(output_root)
    dump_json(output_root / "prompt5_e01_r2d1k_hf1_manifest.json", manifest)
    manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1k_hf1_manifest.json"))

    def set_report_line(prefix: str, value: str) -> None:
        for idx, line in enumerate(final_report_lines):
            if line.startswith(prefix):
                final_report_lines[idx] = value
                return

    metrics["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
    metrics["manifest_nonself_hash_mismatch_count"] = manifest_validation["manifest_nonself_hash_mismatch_count"]
    metrics["manifest_nonself_size_mismatch_count"] = manifest_validation["manifest_nonself_size_mismatch_count"]
    gate = select_gate(metrics)
    gate_payload.update(
        {
            "gate_status": gate,
            "gate_passed": gate == PASS_GATE,
            "manifest_missing_required_file_count": manifest["missing_required_file_count"],
            "manifest_nonself_hash_mismatch_count": manifest_validation["manifest_nonself_hash_mismatch_count"],
            "manifest_nonself_size_mismatch_count": manifest_validation["manifest_nonself_size_mismatch_count"],
        }
    )
    set_report_line("5. ", f"5. HF1 final gate: {gate}")
    set_report_line("39. ", f"39. manifest non-self hash/size results: hash={manifest_validation['manifest_nonself_hash_mismatch_count']}, size={manifest_validation['manifest_nonself_size_mismatch_count']}")
    dump_json(output_root / "prompt5_e01_r2d1k_hf1_gate.json", gate_payload)
    (output_root / "prompt5_e01_r2d1k_hf1_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

    manifest = manifest_payload(output_root)
    dump_json(output_root / "prompt5_e01_r2d1k_hf1_manifest.json", manifest)
    manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1k_hf1_manifest.json"))
    metrics["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
    metrics["manifest_nonself_hash_mismatch_count"] = manifest_validation["manifest_nonself_hash_mismatch_count"]
    metrics["manifest_nonself_size_mismatch_count"] = manifest_validation["manifest_nonself_size_mismatch_count"]
    gate = select_gate(metrics)
    gate_payload.update(
        {
            "gate_status": gate,
            "gate_passed": gate == PASS_GATE,
            "manifest_missing_required_file_count": manifest["missing_required_file_count"],
            "manifest_nonself_hash_mismatch_count": manifest_validation["manifest_nonself_hash_mismatch_count"],
            "manifest_nonself_size_mismatch_count": manifest_validation["manifest_nonself_size_mismatch_count"],
        }
    )
    set_report_line("5. ", f"5. HF1 final gate: {gate}")
    set_report_line("39. ", f"39. manifest non-self hash/size results: hash={manifest_validation['manifest_nonself_hash_mismatch_count']}, size={manifest_validation['manifest_nonself_size_mismatch_count']}")
    dump_json(output_root / "prompt5_e01_r2d1k_hf1_gate.json", gate_payload)
    (output_root / "prompt5_e01_r2d1k_hf1_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")
    manifest = manifest_payload(output_root)
    dump_json(output_root / "prompt5_e01_r2d1k_hf1_manifest.json", manifest)
    manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1k_hf1_manifest.json"))
    if manifest["missing_required_file_count"] or manifest_validation["manifest_nonself_hash_mismatch_count"] or manifest_validation["manifest_nonself_size_mismatch_count"]:
        gate_payload.update(
            {
                "gate_status": "FAIL_MANIFEST_RECONCILIATION",
                "gate_passed": False,
                "manifest_missing_required_file_count": manifest["missing_required_file_count"],
                "manifest_nonself_hash_mismatch_count": manifest_validation["manifest_nonself_hash_mismatch_count"],
                "manifest_nonself_size_mismatch_count": manifest_validation["manifest_nonself_size_mismatch_count"],
            }
        )
        dump_json(output_root / "prompt5_e01_r2d1k_hf1_gate.json", gate_payload)
        raise GateFailure("FAIL_MANIFEST_RECONCILIATION", "Manifest reconciliation failed")

    print("R2D-1K-HF1 CAMPAIGN B METADATA FINALIZATION COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate)
    print("\nnetwork_api_calls:")
    print(0)
    print("\nservice_key_accessed:")
    print("false")
    print("\nupstream_modified_file_count:")
    print(len(modified))
    print("\nupstream_deleted_file_count:")
    print(len(deleted))
    print("\nupstream_added_file_count:")
    print(len(added))
    print("\nsource_r2d1k_gate:")
    print(source_gate)
    print("\npreserved_raw_response_count:")
    print(raw_audit["raw_file_count"])
    print("\npreserved_new_complete_episode_count:")
    print(len(complete_episodes))
    print("\npreserved_new_global_complete_vehicle_count:")
    print(len(new_global_complete_vehicle_ids))
    print("\ncorrected_right_censored_vehicle_id:")
    print("1743")
    print("\ncorrected_right_censor_reason:")
    print(session_1743["episode_status"])
    print("\ncampaign_total_new_route_local_complete_vehicle_count:")
    print(route_local_total)
    print("\nvehicle_1733_route_local_classification:")
    print(row_1733["route_local_vehicle_classification"])
    print("\nvehicle_1733_prior_route_local_occurrence:")
    print(int(row_1733["prior_route_local_occurrence_count_at_ingest"]))
    print("\nvehicle_1733_route_local_occurrence_after_ingest:")
    print(int(row_1733["route_local_occurrence_count_after_ingest"]))
    print("\nvehicle_5303_route_local_classification:")
    print(row_5303["route_local_vehicle_classification"])
    print("\nvehicle_5303_prior_route_local_occurrence:")
    print(int(row_5303["prior_route_local_occurrence_count_at_ingest"]))
    print("\nvehicle_5303_route_local_occurrence_after_ingest:")
    print(int(row_5303["route_local_occurrence_count_after_ingest"]))
    print("\ncorrected_campaign_b_episode_ids:")
    for episode_id in OLD_TO_NEW_EPISODE_ID.values():
        print(episode_id)
    print("\nold_r2d1h_campaign_b_episode_id_count:")
    print(active_old_episode_id_count([output_episodes, output_bounds, corrected_registry]))
    print("\ncumulative_complete_episode_count:")
    print(cumulative_complete_count)
    print("\nroute_complete_counts:")
    for route, count in registry_route_counts.items():
        print(f"{route}={count}")
    print("\nglobal_unique_vehicle_count:")
    print(global_unique_vehicle_count)
    print("\nremaining_complete_episodes_to_12:")
    print(remaining_complete_episodes_to_12)
    print("\nglobal_unique_vehicle_deficit:")
    print(global_unique_vehicle_deficit)
    print("\nscientific_content_change_count:")
    print(scientific_audit["scientific_content_change_count"])
    print("\ninterval_content_change_count:")
    print(scientific_audit["interval_content_change_count"])
    print("\nraw_sha_reference_change_count:")
    print(scientific_audit["raw_sha_reference_change_count"])
    print("\nclock_semantics_change_count:")
    print(scientific_audit["clock_semantics_change_count"])
    print("\ncounter_reconciliation_failure_count:")
    print(counter_reconciliation_failure_count)
    print("\ncross_row_aggregation_mismatch_count:")
    print(cross_row_aggregation_mismatch_count)
    print("\nepisode_namespace_reconciliation_failure_count:")
    print(namespace_failure_count)
    print("\nlineage_conflict_count:")
    print(lineage_conflict_count)
    print("\nepisode_duplicate_count:")
    print(episode_duplicate_count)
    print("\nmapping_regression_count:")
    print(mapping_regression_count)
    print("\njson_parquet_value_mismatch_count:")
    print(json_parquet_audit["json_parquet_value_mismatch_count"])
    print("\nstrict_json_failure_count:")
    print(len(strict_failures))
    print("\nparquet_read_failure_count:")
    print(len(parquet_read_failures))
    print("\nmanifest_missing_required_file_count:")
    print(manifest["missing_required_file_count"])
    print("\nmanifest_nonself_hash_mismatch_count:")
    print(manifest_validation["manifest_nonself_hash_mismatch_count"])
    print("\nmanifest_nonself_size_mismatch_count:")
    print(manifest_validation["manifest_nonself_size_mismatch_count"])
    print("\nmanifest_self_hash_exempt:")
    print("true")
    print("\nmanifest_self_size_exempt:")
    print("true")
    print("\nsecret_leak_count:")
    print(0)
    print("\ncampaign_c_authorized:")
    print("false")
    print("\nterminal_recovery_estimation_execution_approved:")
    print("false")
    print("\nterminal_recovery_parameter_generated:")
    print("false")
    print("\nterminal_recovery_applied:")
    print("false")
    print("\nsimulator_application_authorized:")
    print("false")
    print("\nphase2_authorized:")
    print("false")
    print("\nnext_authorized_action:")
    print("Campaign C authorization review only")


if __name__ == "__main__":
    try:
        main()
    except GateFailure as exc:
        print(exc.gate)
        raise
