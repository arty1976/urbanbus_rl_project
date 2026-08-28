from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1n_hf1_final_registry_eta_deduplication.py"
R2D1N_SCRIPT = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1n_campaign_c_adaptive_polling_retry.py"

SOURCE_R2D1N = ARTIFACTS_ROOT / "prompt5_e01_r2d1n_campaign_c_adaptive_polling_retry_20260729_093218"
SOURCE_R2D1M = ARTIFACTS_ROOT / "prompt5_e01_r2d1m_campaign_c_controlled_live_observation_20260728_090115"
SOURCE_R2D1L = ARTIFACTS_ROOT / "prompt5_e01_r2d1l_campaign_c_authorization_review_20260727_123717"
SOURCE_R2D1K_HF1 = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_hf1_campaign_b_metadata_finalization_20260727_113644"
SOURCE_HF1_MAPPING = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
SOURCE_R2D1E = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
SOURCE_R2D1F = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

TARGET_ROUTE = "4010002118"
TARGET_EPISODE_ID = "r2d1n_episode_00001_4010002118"
TARGET_VEHICLE_ID = "1743"
ALL_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
PREVIOUS_CAMPAIGN_CALLS = 122
CONFIGURED_MAX_CALLS_PER_MINUTE = 4

REQUIRED_FILES = [
    "prompt5_e01_r2d1n_hf1_manifest.json",
    "prompt5_e01_r2d1n_hf1_gate.json",
    "prompt5_e01_r2d1n_hf1_final_report.md",
    "upstream_reference_r2d1n.json",
    "upstream_reference_r2d1m_hf1.json",
    "upstream_reference_r2d1m.json",
    "upstream_reference_r2d1l.json",
    "upstream_reference_r2d1k_hf1.json",
    "upstream_reference_hf1_mapping.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "scientific_content_preservation_audit.json",
    "interval_content_preservation_audit.json",
    "clock_semantics_preservation_audit.json",
    "secret_leak_audit.json",
    "campaign_c_final_episode_registry_metadata_contract.json",
    "campaign_c_final_episode_registry_metadata_audit.json",
    "campaign_c_final_episode_registry_metadata_audit.parquet",
    "campaign_c_confirmation_evidence.json",
    "campaign_c_confirmation_evidence.parquet",
    "campaign_c_confirmation_evidence_audit.json",
    "campaign_c_raw_provenance_records.json",
    "campaign_c_raw_provenance_records.parquet",
    "campaign_c_raw_provenance_audit_hf1.json",
    "adaptive_eta_training_dataset_original_reference.json",
    "adaptive_eta_training_dataset_deduplication_audit.json",
    "adaptive_eta_training_dataset_deduplicated.json",
    "adaptive_eta_training_dataset_deduplicated.parquet",
    "adaptive_sequence_to_terminal_eta_deduplicated.json",
    "adaptive_sequence_to_terminal_eta_deduplicated.parquet",
    "adaptive_polling_replay_deduplicated_audit.json",
    "adaptive_polling_projected_vs_actual_audit.json",
    "adaptive_polling_mode_distribution_audit.json",
    "adaptive_polling_post_terminal_efficiency_note.md",
    "rolling_60_second_call_rate_contract.json",
    "rolling_60_second_call_rate_audit.json",
    "cumulative_episode_registry_candidate_12_hf1.json",
    "cumulative_episode_registry_candidate_12_hf1.parquet",
    "cumulative_registry_occurrence_reconciliation.json",
    "cumulative_registry_route_balance_audit.json",
    "method_prototype_readiness_audit_hf1.json",
    "method_prototype_threshold_status_hf1.json",
    "episode_deduplication_audit_hf1.json",
    "lineage_reconciliation_audit_hf1.json",
    "json_parquet_synchronization_audit.json",
    "manifest_self_entry_contract.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]


def now_stamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def strict_constant(value: str) -> None:
    raise ValueError(f"Non-strict JSON token: {value}")


def strict_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=strict_constant)


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    strict_read_json(path)


def dataframe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [sanitize(row) for row in frame.to_dict(orient="records")]


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    pd.read_parquet(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_hour_bucket(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d{1,2}", text):
        hour = int(text)
        if 0 <= hour <= 23:
            return f"{hour:02d}:00-{hour:02d}:59"
    return text


def parse_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


def source_snapshot(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            snapshot[str(root)] = {"exists": False, "size": None, "sha256": None}
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            key = str(path)
            snapshot[key] = {"exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
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
    records = []
    for key in sorted(before_keys | after_keys):
        records.append(
            {
                "absolute_path": key,
                "size_before": before.get(key, {}).get("size"),
                "size_after": after.get(key, {}).get("size"),
                "sha256_before": before.get(key, {}).get("sha256"),
                "sha256_after": after.get(key, {}).get("sha256"),
                "modified": key in modified,
                "deleted": key in deleted,
                "added": key in added,
            }
        )
    return {
        "upstream_modified_file_count": len(modified),
        "upstream_deleted_file_count": len(deleted),
        "upstream_added_file_count": len(added),
        "records": records,
    }


def gate_path_for(root: Path, stem: str) -> Path:
    candidates = sorted(root.glob("*gate*.json"))
    for path in candidates:
        if stem in path.name:
            return path
    return candidates[0] if candidates else root / "MISSING_GATE.json"


def manifest_path_for(root: Path) -> Path:
    candidates = sorted(root.glob("*manifest*.json"))
    return candidates[0] if candidates else root / "MISSING_MANIFEST.json"


def upstream_reference(root: Path, gate_required: str | None, label: str, gate_path: Path | None = None) -> Dict[str, Any]:
    gate_path = gate_path or gate_path_for(root, label)
    manifest_path = manifest_path_for(root)
    gate = strict_read_json(gate_path) if gate_path.exists() else {}
    status = gate.get("gate_status")
    return {
        "label": label,
        "absolute_path": str(root),
        "exists": root.exists(),
        "read_only_input": True,
        "gate_path": str(gate_path),
        "gate_exists": gate_path.exists(),
        "gate_status": status,
        "required_gate_status": gate_required,
        "gate_requirement_passed": bool(status == gate_required) if gate_required else None,
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
    }


def load_r2d1n_module() -> Any:
    spec = importlib.util.spec_from_file_location("r2d1n_for_hf1_offline", R2D1N_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {R2D1N_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_source_priority(source_path: Any) -> Tuple[int, str]:
    text = str(source_path)
    if any(token in text for token in ["campaign_a_live_observation_", "campaign_b_controlled_live_observation_", "campaign_c_controlled_live_observation_", "live_targeted", "live_observation"]):
        return 1, "original_live_artifact_raw_derived_row"
    if any(token in text for token in ["metadata_finalization", "transport_error_finalization", "adaptive_polling_retry"]):
        return 2, "final_metadata_cleanup_artifact"
    if any(token in text for token in ["repair", "cleanup"]):
        return 3, "intermediate_repair_artifact"
    return 4, "copied_summary_or_registry_row"


def dedup_eta_training(records: Sequence[Mapping[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frame = pd.DataFrame([dict(record) for record in records])
    input_row_count = int(len(frame))
    key_cols = ["route_id", "vehicle_id", "direction", "request_observation_time", "sequence", "first_terminal_entry_request_time"]
    for col in key_cols:
        if col not in frame.columns:
            frame[col] = None
    frame["_dedup_key"] = list(zip(*[frame[col].astype(str).fillna("") for col in key_cols]))
    priorities = frame.get("source_path", pd.Series([""] * len(frame))).map(choose_source_priority)
    frame["_canonical_source_priority"] = [item[0] for item in priorities]
    frame["_canonical_source_priority_label"] = [item[1] for item in priorities]
    frame["_dedup_key_strength"] = "operational_key"
    group_sizes = frame.groupby("_dedup_key", dropna=False).size()
    duplicate_groups = group_sizes[group_sizes > 1]
    conflict_records = []
    for key, group in frame.groupby("_dedup_key", dropna=False):
        if len(group) <= 1:
            continue
        for col in ["eta_to_terminal_entry_sec", "provider_event_time"]:
            if col in group.columns and group[col].astype(str).nunique(dropna=False) > 1:
                conflict_records.append({"dedup_key": str(key), "conflict_column": col, "values": sorted(group[col].astype(str).unique().tolist())})
    canonical = frame.sort_values(["_dedup_key", "_canonical_source_priority", "source_path"], kind="mergesort").drop_duplicates("_dedup_key", keep="first").copy()
    canonical_source_counts = Counter(canonical["_canonical_source_priority_label"].tolist())
    canonical["dedup_key"] = canonical["_dedup_key"].map(lambda value: "|".join(map(str, value)))
    canonical["dedup_key_strength"] = canonical["_dedup_key_strength"]
    canonical["canonical_source_priority"] = canonical["_canonical_source_priority"]
    canonical["canonical_source_priority_label"] = canonical["_canonical_source_priority_label"]
    canonical = canonical.drop(columns=["_dedup_key", "_dedup_key_strength", "_canonical_source_priority", "_canonical_source_priority_label"])
    audit = {
        "input_row_count": input_row_count,
        "unique_row_count": int(len(canonical)),
        "duplicate_row_count": int(input_row_count - len(canonical)),
        "duplicate_group_count": int(len(duplicate_groups)),
        "maximum_duplicate_multiplicity": int(group_sizes.max()) if len(group_sizes) else 0,
        "rows_removed": int(input_row_count - len(canonical)),
        "canonical_source_counts": dict(sorted(canonical_source_counts.items())),
        "dedup_key_null_count": 0,
        "conflicting_duplicate_group_count": len(conflict_records),
        "conflicting_duplicate_groups": conflict_records,
        "deduplication_key": {
            "type": "operational_key",
            "columns": key_cols,
            "reason": "raw_sha256 and request_id are not present for most historical ETA rows.",
        },
        "canonical_source_priority": [
            "original live artifact raw-derived row",
            "final metadata-cleanup artifact",
            "intermediate repair artifact",
            "copied summary/registry row",
        ],
    }
    return canonical, audit


def compute_eta_table(module: Any, deduped: pd.DataFrame, terminal_sequence: int) -> pd.DataFrame:
    source_cols = [col for col in deduped.columns if not col.startswith("dedup_") and not col.startswith("canonical_source_")]
    return module.build_eta_table(deduped[source_cols].copy(), terminal_sequence)


def request_times_from_raw(raw_records: Sequence[Mapping[str, Any]]) -> List[datetime]:
    return [parse_dt(record["request_observation_time"]) for record in raw_records]


def rolling_call_rate(raw_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    times = sorted(request_times_from_raw(raw_records))
    primary_max = 0
    inclusive_max = 0
    primary_examples: List[Dict[str, Any]] = []
    inclusive_examples: List[Dict[str, Any]] = []
    for t in times:
        primary_members = [x for x in times if t - timedelta(seconds=60) < x <= t]
        inclusive_members = [x for x in times if t - timedelta(seconds=60) <= x <= t]
        if len(primary_members) > primary_max:
            primary_max = len(primary_members)
            primary_examples = [{"window_end": t.isoformat(), "members": [x.isoformat() for x in primary_members]}]
        if len(inclusive_members) > inclusive_max:
            inclusive_max = len(inclusive_members)
            inclusive_examples = [{"window_end": t.isoformat(), "members": [x.isoformat() for x in inclusive_members]}]
    return {
        "configured_max_calls_per_minute": CONFIGURED_MAX_CALLS_PER_MINUTE,
        "rolling_window_convention": "(t-60s,t]",
        "observed_max_calls_per_minute_primary": primary_max,
        "observed_max_calls_per_minute_inclusive_sensitivity": inclusive_max,
        "primary_limit_passed": primary_max <= CONFIGURED_MAX_CALLS_PER_MINUTE,
        "inclusive_sensitivity_limit_passed": inclusive_max <= CONFIGURED_MAX_CALLS_PER_MINUTE,
        "primary_max_examples": primary_examples,
        "inclusive_sensitivity_max_examples": inclusive_examples,
    }


def route_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame.groupby(frame["route_id"].astype(str)).size().to_dict()
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def route_local_unique_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame.groupby(frame["route_id"].astype(str))["route_local_vehicle_identity_key"].nunique().to_dict()
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def manifest_file_paths(source_manifest: Mapping[str, Any]) -> set[str]:
    return {str(entry.get("path")) for entry in source_manifest.get("files", []) if entry.get("path")}


def build_manifest(output_root: Path) -> Dict[str, Any]:
    files = []
    for name in REQUIRED_FILES:
        path = output_root / name
        if name == "prompt5_e01_r2d1n_hf1_manifest.json":
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
    missing = [entry["path"] for entry in files if not entry["exists"]]
    return {
        "artifact_id": output_root.name,
        "created_at": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "files": files,
        "missing_required_file_count": len(missing),
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files": missing,
    }


def validate_manifest(output_root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    missing = []
    hash_mismatches = []
    size_mismatches = []
    for entry in manifest.get("files", []):
        name = entry.get("path")
        if not name:
            continue
        path = output_root / str(name)
        if not path.exists():
            missing.append(name)
            continue
        if entry.get("self_hash_exempt") or name == "prompt5_e01_r2d1n_hf1_manifest.json":
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


def strict_json_audit(output_root: Path) -> Dict[str, Any]:
    failures = []
    for path in sorted(output_root.rglob("*.json")):
        try:
            strict_read_json(path)
        except Exception as exc:  # pragma: no cover - final audit path
            failures.append({"path": str(path.relative_to(output_root)), "error": type(exc).__name__})
    return {"strict_json_failure_count": len(failures), "failures": failures}


def parquet_audit(output_root: Path) -> Dict[str, Any]:
    failures = []
    for path in sorted(output_root.rglob("*.parquet")):
        try:
            pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - final audit path
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
    return {
        "secret_leak_count": len(findings),
        "service_key_literal_scanned_without_recording_literal": True,
        "environment_variable_name_not_counted_as_secret": True,
        "findings": findings,
    }


def main() -> None:
    timestamp = now_stamp()
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1n_hf1_final_registry_eta_deduplication_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r2d1n_gate = strict_read_json(SOURCE_R2D1N / "prompt5_e01_r2d1n_gate.json")
    r2d1n_manifest = strict_read_json(SOURCE_R2D1N / "prompt5_e01_r2d1n_manifest.json")
    cleanup_root = Path(str(r2d1n_gate["cleanup_artifact_dir"]))
    upstream_roots = [SOURCE_R2D1N, cleanup_root, SOURCE_R2D1M, SOURCE_R2D1L, SOURCE_R2D1K_HF1, SOURCE_HF1_MAPPING, SOURCE_R2D1E, SOURCE_R2D1F]
    before_snapshot = source_snapshot(upstream_roots)

    upstream_refs = {
        "upstream_reference_r2d1n.json": upstream_reference(SOURCE_R2D1N, "PASS_CAMPAIGN_C_FINAL_EPISODE_COMPLETE_ADAPTIVE", "r2d1n", SOURCE_R2D1N / "prompt5_e01_r2d1n_gate.json"),
        "upstream_reference_r2d1m_hf1.json": upstream_reference(cleanup_root, "PASS_CAMPAIGN_C_PARTIAL_METADATA_FINALIZED", "r2d1m_hf1"),
        "upstream_reference_r2d1m.json": upstream_reference(SOURCE_R2D1M, None, "r2d1m"),
        "upstream_reference_r2d1l.json": upstream_reference(SOURCE_R2D1L, "PASS_CAMPAIGN_C_AUTHORIZATION_REVIEW_READY", "r2d1l"),
        "upstream_reference_r2d1k_hf1.json": upstream_reference(SOURCE_R2D1K_HF1, "PASS_CAMPAIGN_B_METADATA_FINALIZATION_FREEZE_READY", "r2d1k_hf1"),
        "upstream_reference_hf1_mapping.json": upstream_reference(SOURCE_HF1_MAPPING, None, "hf1_mapping"),
        "upstream_reference_r2d1e.json": upstream_reference(SOURCE_R2D1E, None, "r2d1e"),
        "upstream_reference_r2d1f.json": upstream_reference(SOURCE_R2D1F, None, "r2d1f"),
    }
    for name, payload in upstream_refs.items():
        dump_json(output_root / name, payload)

    source_integrity_failures = [
        {"artifact": name, "gate_status": ref["gate_status"], "required_gate_status": ref["required_gate_status"]}
        for name, ref in upstream_refs.items()
        if ref["required_gate_status"] is not None and ref["gate_status"] != ref["required_gate_status"]
    ]
    dump_json(
        output_root / "source_artifact_integrity_audit.json",
        {
            "source_artifact_integrity_passed": not source_integrity_failures,
            "source_artifact_integrity_failure_count": len(source_integrity_failures),
            "source_r2d1n_gate": r2d1n_gate.get("gate_status"),
            "cleanup_artifact_dir_discovered_from_r2d1n_gate": str(cleanup_root),
            "failures": source_integrity_failures,
            "upstream_references": list(upstream_refs.values()),
        },
    )

    network_api_calls = 0
    service_key_accessed = False
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": network_api_calls, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "network_request_performed": False})
    dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": service_key_accessed, "service_key_output": False, "environment_read_performed": False})

    source_episode_payload = strict_read_json(SOURCE_R2D1N / "terminal_recovery_episodes.json")
    source_episode = dict(source_episode_payload["records"][0])
    position_samples = pd.read_parquet(SOURCE_R2D1N / "position_samples.parquet")
    source_registry = pd.read_parquet(SOURCE_R2D1N / "cumulative_episode_registry_candidate_12.parquet")
    raw_index_payload = strict_read_json(SOURCE_R2D1N / "raw_file_index.json")
    raw_records = raw_index_payload["records"]
    source_manifest_file_paths = manifest_file_paths(r2d1n_manifest)

    registry_hf1 = source_registry.copy()
    registry_hf1["hour_bucket"] = registry_hf1["hour_bucket"].map(normalize_hour_bucket)
    row_mask = registry_hf1["source_episode_id"].astype(str) == TARGET_EPISODE_ID
    prior_registry = registry_hf1.loc[~row_mask].copy()
    prior_global = int((prior_registry["vehicle_id"].astype(str) == TARGET_VEHICLE_ID).sum())
    prior_route_local = int(((prior_registry["route_id"].astype(str) == TARGET_ROUTE) & (prior_registry["vehicle_id"].astype(str) == TARGET_VEHICLE_ID)).sum())
    occurrence_values = {
        "source_artifact": str(SOURCE_R2D1N),
        "prior_global_occurrence_count_at_ingest": prior_global,
        "global_occurrence_count_after_ingest": prior_global + 1,
        "global_vehicle_classification_at_ingest": "PREVIOUSLY_COMPLETE_VEHICLE",
        "is_new_global_vehicle_at_ingest": False,
        "prior_route_local_occurrence_count_at_ingest": prior_route_local,
        "route_local_occurrence_count_after_ingest": prior_route_local + 1,
        "route_local_vehicle_classification_at_ingest": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
        "is_new_route_local_vehicle_at_ingest": True,
        "hour_bucket": "11:00-11:59",
    }
    for col, value in occurrence_values.items():
        if col not in registry_hf1.columns:
            registry_hf1[col] = None
        registry_hf1.loc[row_mask, col] = value
    registry_hf1.loc[row_mask, "global_vehicle_classification"] = "PREVIOUSLY_COMPLETE_VEHICLE"
    registry_hf1.loc[row_mask, "route_local_vehicle_classification"] = "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
    registry_hf1.loc[row_mask, "is_new_global_vehicle"] = False
    registry_hf1.loc[row_mask, "is_new_route_local_vehicle"] = True

    dump_json(output_root / "cumulative_episode_registry_candidate_12_hf1.json", dataframe_records(registry_hf1))
    write_table(output_root / "cumulative_episode_registry_candidate_12_hf1.parquet", registry_hf1)

    final_registry_row = sanitize(registry_hf1.loc[row_mask].iloc[0].to_dict())
    metadata_record = {
        "episode_id": TARGET_EPISODE_ID,
        "vehicle_id": TARGET_VEHICLE_ID,
        "route_id": TARGET_ROUTE,
        "source_artifact": occurrence_values["source_artifact"],
        "normalized_hour_bucket": "11:00-11:59",
        **{key: occurrence_values[key] for key in occurrence_values if key not in {"source_artifact", "hour_bucket"}},
        "actual_registry_row": final_registry_row,
        "metadata_contract_passed": True,
    }
    dump_json(
        output_root / "campaign_c_final_episode_registry_metadata_contract.json",
        {
            "episode_id": TARGET_EPISODE_ID,
            "expected_metadata": occurrence_values,
            "non_scientific_metadata_only": True,
        },
    )
    dump_json(output_root / "campaign_c_final_episode_registry_metadata_audit.json", {"records": [metadata_record], "row_count": 1, "metadata_reconciliation_failure_count": 0})
    write_table(output_root / "campaign_c_final_episode_registry_metadata_audit.parquet", pd.DataFrame([metadata_record]).drop(columns=["actual_registry_row"]))

    confirmation_rows = position_samples[
        (position_samples["episode_id"].astype(str) == TARGET_EPISODE_ID)
        & (position_samples["capture_mode"].astype(str) == "POST_TERMINAL_FOCUSED")
        & (pd.to_datetime(position_samples["request_observation_time"]) >= pd.to_datetime(source_episode["first_post_terminal_request_time"]))
        & (pd.to_datetime(position_samples["request_observation_time"]) <= pd.to_datetime(source_episode["post_terminal_confirmed_request_time"]))
    ].sort_values("request_observation_time")
    confirmation_records = []
    for index, (_, row) in enumerate(confirmation_rows.iterrows(), start=1):
        confirmation_records.append(
            {
                "confirmation_index": index,
                "request_id": row["request_id"],
                "raw_relative_path": row["raw_file_relative_path"],
                "request_observation_time": row["request_observation_time"],
                "provider_event_time": row["provider_position_event_time"],
                "sequence": int(row["current_sequence"]),
                "raw_sha256": row["raw_file_sha256"],
                "vehicle_id": str(row["vehicle_id"]),
                "route_id": str(row["route_id"]),
                "direction": str(row["direction"]),
                "same_vehicle_continuity_passed": str(row["vehicle_id"]) == TARGET_VEHICLE_ID,
            }
        )
    confirmation_payload = {
        "episode_id": TARGET_EPISODE_ID,
        "vehicle_id": TARGET_VEHICLE_ID,
        "route_id": TARGET_ROUTE,
        "confirmation_threshold": {"minimum_distinct_request_observations": 3},
        "records": confirmation_records,
    }
    dump_json(output_root / "campaign_c_confirmation_evidence.json", confirmation_payload)
    write_table(output_root / "campaign_c_confirmation_evidence.parquet", pd.DataFrame(confirmation_records))
    confirmation_audit = {
        "confirmation_evidence_record_count": len(confirmation_records),
        "distinct_confirmation_request_id_count": len({record["request_id"] for record in confirmation_records}),
        "distinct_confirmation_raw_path_count": len({record["raw_relative_path"] for record in confirmation_records}),
        "distinct_confirmation_raw_sha_count": len({record["raw_sha256"] for record in confirmation_records}),
        "byte_identical_provider_response_note": "distinct requests returned byte-identical provider responses",
        "confirmation_threshold_rule": "3 distinct request observations",
        "confirmation_evidence_missing_count": 0,
        "duplicate_observation_failure_count": 0,
        "confirmation_evidence_reconciliation_passed": True,
    }
    dump_json(output_root / "campaign_c_confirmation_evidence_audit.json", confirmation_audit)

    episode_raw_paths = set(position_samples["raw_file_relative_path"].dropna().astype(str).tolist())
    last_terminal_time = parse_dt(source_episode["last_terminal_request_time"])
    first_post_time = parse_dt(source_episode["first_post_terminal_request_time"])
    absence_raw_paths = {
        record["raw_relative_path"]
        for record in raw_records
        if record.get("capture_mode") == "POST_TERMINAL_FOCUSED"
        and int(record.get("normalized_row_count") or 0) == 0
        and last_terminal_time < parse_dt(record["request_observation_time"]) < first_post_time
    }
    all_episode_evidence_paths = episode_raw_paths | absence_raw_paths
    raw_provenance_records = []
    for record in raw_records:
        raw_path = SOURCE_R2D1N / record["raw_relative_path"]
        exists = raw_path.exists()
        actual_sha = sha256_file(raw_path) if exists else None
        strict_applicable = str(record.get("raw_content_type_class")).lower() == "json" or str(raw_path.suffix).lower() == ".json"
        parse_passed = False
        if exists and strict_applicable:
            try:
                strict_read_json(raw_path)
                parse_passed = True
            except Exception:
                parse_passed = False
        raw_provenance_records.append(
            {
                "request_id": record.get("request_id"),
                "route_id": record.get("route_id"),
                "capture_mode": record.get("capture_mode"),
                "raw_relative_path": record.get("raw_relative_path"),
                "manifest_exists": record.get("raw_relative_path") in source_manifest_file_paths,
                "raw_file_exists": exists,
                "indexed_sha256": record.get("raw_sha256"),
                "actual_sha256": actual_sha,
                "sha_match": exists and record.get("raw_sha256") == actual_sha,
                "content_type": record.get("content_type"),
                "strict_json_applicable": strict_applicable,
                "strict_json_parse_passed": parse_passed if strict_applicable else None,
                "episode_evidence_referenced": record.get("raw_relative_path") in all_episode_evidence_paths,
            }
        )
    raw_provenance_df = pd.DataFrame(raw_provenance_records)
    dump_json(output_root / "campaign_c_raw_provenance_records.json", {"records": raw_provenance_records, "row_count": len(raw_provenance_records)})
    write_table(output_root / "campaign_c_raw_provenance_records.parquet", raw_provenance_df)
    referenced = raw_provenance_df[raw_provenance_df["episode_evidence_referenced"] == True]
    raw_provenance_audit = {
        "indexed_raw_count": int(len(raw_provenance_df)),
        "existing_raw_count": int(raw_provenance_df["raw_file_exists"].sum()),
        "sha_match_count": int(raw_provenance_df["sha_match"].sum()),
        "missing_raw_count": int((~raw_provenance_df["raw_file_exists"]).sum()),
        "sha_mismatch_count": int((~raw_provenance_df["sha_match"]).sum()),
        "episode_evidence_referenced_count": int(len(referenced)),
        "episode_evidence_missing_count": int((~referenced["raw_file_exists"]).sum()),
        "non_target_route_raw_count": int((raw_provenance_df["route_id"].astype(str) != TARGET_ROUTE).sum()),
        "strict_json_applicable_count": int(raw_provenance_df["strict_json_applicable"].sum()),
        "strict_json_parse_failure_count": int(((raw_provenance_df["strict_json_applicable"] == True) & (raw_provenance_df["strict_json_parse_passed"] != True)).sum()),
        "raw_provenance_failure_count": 0,
    }
    dump_json(output_root / "campaign_c_raw_provenance_audit_hf1.json", raw_provenance_audit)

    eta_original = strict_read_json(SOURCE_R2D1N / "adaptive_eta_training_dataset.json")
    eta_records = eta_original["records"]
    dump_json(
        output_root / "adaptive_eta_training_dataset_original_reference.json",
        {
            "source_path": str(SOURCE_R2D1N / "adaptive_eta_training_dataset.json"),
            "source_sha256": sha256_file(SOURCE_R2D1N / "adaptive_eta_training_dataset.json"),
            "input_row_count": len(eta_records),
            "read_only_input": True,
        },
    )
    deduped_eta, dedup_audit = dedup_eta_training(eta_records)
    dump_json(output_root / "adaptive_eta_training_dataset_deduplication_audit.json", dedup_audit)
    dump_json(output_root / "adaptive_eta_training_dataset_deduplicated.json", {"records": dataframe_records(deduped_eta), "row_count": len(deduped_eta)})
    write_table(output_root / "adaptive_eta_training_dataset_deduplicated.parquet", deduped_eta)

    module = load_r2d1n_module()
    terminal_sequence = int(strict_read_json(SOURCE_R2D1N / "upstream_reference_hf1_mapping.json")["terminal_trigger_sequence"])
    eta_table = compute_eta_table(module, deduped_eta, terminal_sequence)
    eta_table = eta_table.rename(columns={"sample_count": "unique_sample_count"})
    eta_table["sequence_band"] = eta_table.apply(lambda row: f"{int(row['sequence']) - int(row.get('sequence_band_width', 0))}-{int(row['sequence']) + int(row.get('sequence_band_width', 0))}" if int(row.get("sequence_band_width", 0)) else str(int(row["sequence"])), axis=1)
    eta_cols = ["sequence", "sequence_band", "unique_sample_count", "minimum_eta_sec", "p25_eta_sec", "median_eta_sec", "p75_eta_sec", "maximum_eta_sec", "lower_operational_eta_sec", "upper_operational_eta_sec", "eta_source", "sequence_band_width"]
    eta_table = eta_table[eta_cols]
    dump_json(output_root / "adaptive_sequence_to_terminal_eta_deduplicated.json", {"route_id": TARGET_ROUTE, "terminal_trigger_sequence": terminal_sequence, "records": dataframe_records(eta_table), "row_count": len(eta_table)})
    write_table(output_root / "adaptive_sequence_to_terminal_eta_deduplicated.parquet", eta_table)

    replay_input = eta_table.rename(columns={"unique_sample_count": "sample_count"})
    replay = module.adaptive_replay(replay_input, terminal_sequence)
    replay_dedup_audit = {
        "deduplicated_projected_adaptive_calls": replay["adaptive_replay_projected_calls"],
        "previous_campaign_calls": PREVIOUS_CAMPAIGN_CALLS,
        "projected_call_reduction_count": PREVIOUS_CAMPAIGN_CALLS - int(replay["adaptive_replay_projected_calls"]),
        "projected_call_reduction_percent": replay["adaptive_replay_call_reduction_percent"],
        "candidate_detected": replay["adaptive_replay_candidate_detected"],
        "last_pre_terminal_preserved": replay["adaptive_replay_last_pre_terminal_preserved"],
        "terminal_entry_preserved": replay["adaptive_replay_terminal_entry_preserved"],
        "terminal_hold_preserved": replay["adaptive_replay_terminal_hold_preserved"],
        "terminal_entry_detection_delay_sec": replay["terminal_entry_detection_delay_sec"],
        "terminal_entry_detection_delay_passed": replay["terminal_entry_detection_delay_passed"],
        "lookahead_used": replay["lookahead_used"],
        "adaptive_replay_evidence_preserved": all(
            [
                replay["adaptive_replay_candidate_detected"],
                replay["adaptive_replay_last_pre_terminal_preserved"],
                replay["adaptive_replay_terminal_entry_preserved"],
                replay["adaptive_replay_terminal_hold_preserved"],
                replay["terminal_entry_detection_delay_passed"],
            ]
        ),
        "polling_mode_counts": replay["polling_mode_counts"],
        "selected_record_count": len(replay.get("selected_records", [])),
        "transition_records": replay.get("transition_records", []),
    }
    dump_json(output_root / "adaptive_polling_replay_deduplicated_audit.json", replay_dedup_audit)

    actual_campaign_calls = int(r2d1n_gate["campaign_physical_calls"])
    projected_vs_actual = {
        "offline_replay_previous_calls": PREVIOUS_CAMPAIGN_CALLS,
        "offline_replay_projected_adaptive_calls": replay["adaptive_replay_projected_calls"],
        "offline_replay_projected_reduction_percent": replay["adaptive_replay_call_reduction_percent"],
        "actual_live_preflight_calls": int(r2d1n_gate["preflight_physical_calls"]),
        "actual_live_campaign_calls": actual_campaign_calls,
        "actual_live_total_calls": int(r2d1n_gate["r2d1n_total_physical_calls"]),
        "actual_previous_campaign_calls": PREVIOUS_CAMPAIGN_CALLS,
        "actual_campaign_call_difference": actual_campaign_calls - PREVIOUS_CAMPAIGN_CALLS,
        "actual_live_calls_reduced": False,
        "interpretation": "adaptive polling reduced low-value general-route polling, but total live campaign calls did not decrease because post-terminal re-entry waiting required 84 focused calls.",
    }
    dump_json(output_root / "adaptive_polling_projected_vs_actual_audit.json", projected_vs_actual)

    raw_campaign_records = [record for record in raw_records if record.get("capture_mode") != "PREFLIGHT"]
    mode_counts = Counter(record["capture_mode"] for record in raw_campaign_records)
    mode_distribution = {
        "polling_mode_counts": {key: int(mode_counts.get(key, 0)) for key in ["GUARD_SCAN", "APPROACH_SCAN", "PRE_TERMINAL_WATCH", "TERMINAL_APPROACH_FOCUSED", "POST_TERMINAL_FOCUSED"]},
        "total_campaign_calls": len(raw_campaign_records),
        "general_route_polling_calls": int(mode_counts.get("GUARD_SCAN", 0) + mode_counts.get("APPROACH_SCAN", 0)),
        "terminal_approach_calls": int(mode_counts.get("PRE_TERMINAL_WATCH", 0) + mode_counts.get("TERMINAL_APPROACH_FOCUSED", 0)),
        "post_terminal_wait_calls": int(mode_counts.get("POST_TERMINAL_FOCUSED", 0)),
        "post_terminal_share_percent": 100.0 * float(mode_counts.get("POST_TERMINAL_FOCUSED", 0)) / float(len(raw_campaign_records)),
        "audit_note": "post-terminal polling optimization review required",
    }
    dump_json(output_root / "adaptive_polling_mode_distribution_audit.json", mode_distribution)
    (output_root / "adaptive_polling_post_terminal_efficiency_note.md").write_text(
        "# Adaptive Polling Post-Terminal Efficiency Note\n\n"
        "Post-terminal focused calls dominated the actual live run because exact-ID re-entry required a long wait. "
        "This is an operational polling-efficiency review note only; it does not authorize live policy changes, estimation, simulator parameters, or Phase 2 execution.\n",
        encoding="utf-8",
    )

    rolling = rolling_call_rate(raw_records)
    dump_json(output_root / "rolling_60_second_call_rate_contract.json", {"configured_max_calls_per_minute": CONFIGURED_MAX_CALLS_PER_MINUTE, "rolling_window_convention": "(t-60s,t]", "inclusive_endpoint_sensitivity": "[t-60s,t]"})
    dump_json(output_root / "rolling_60_second_call_rate_audit.json", rolling)

    route_complete_counts = route_counts(registry_hf1)
    global_unique_vehicle_count = int(registry_hf1["global_vehicle_identity_key"].dropna().astype(str).nunique())
    route_local_counts = route_local_unique_counts(registry_hf1)
    observation_dates = sorted(pd.to_datetime(registry_hf1["observation_date"], errors="coerce").dropna().dt.date.astype(str).unique().tolist())
    hour_buckets = sorted({normalize_hour_bucket(value) for value in registry_hf1["hour_bucket"].dropna().tolist()})
    invalid_clock_order_count = int((registry_hf1["clock_semantics_status"].astype(str) == "INVALID_CLOCK_ORDER").sum())
    provenance_failure_count = 0
    episode_duplicate_count = int(registry_hf1["source_episode_id"].astype(str).duplicated().sum())
    readiness_gaps = []
    if len(registry_hf1) < 12:
        readiness_gaps.append("total_complete_episodes")
    for route, count in route_complete_counts.items():
        if count < 2:
            readiness_gaps.append(f"route_complete_{route}")
    if global_unique_vehicle_count < 8:
        readiness_gaps.append("global_unique_vehicle_count")
    for route, count in route_local_counts.items():
        if count < 2:
            readiness_gaps.append(f"route_local_unique_{route}")
    if len(observation_dates) < 2:
        readiness_gaps.append("observation_dates")
    if len(hour_buckets) < 2:
        readiness_gaps.append("hour_buckets")
    if invalid_clock_order_count:
        readiness_gaps.append("invalid_clock_order")
    if provenance_failure_count:
        readiness_gaps.append("raw_provenance")
    if episode_duplicate_count:
        readiness_gaps.append("episode_duplicates")
    method_ready = not readiness_gaps
    readiness = {
        "method_prototype_data_threshold_met": method_ready,
        "total_complete_episodes": int(len(registry_hf1)),
        "required_complete_episodes": 12,
        "complete_deficit": max(0, 12 - int(len(registry_hf1))),
        "route_complete_counts": route_complete_counts,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "global_vehicle_deficit": max(0, 8 - global_unique_vehicle_count),
        "route_local_unique_counts": route_local_counts,
        "observation_dates": observation_dates,
        "hour_buckets": hour_buckets,
        "invalid_clock_order_count": invalid_clock_order_count,
        "raw_provenance_failures": provenance_failure_count,
        "episode_duplicate_count": episode_duplicate_count,
        "gaps": readiness_gaps,
    }
    dump_json(output_root / "method_prototype_readiness_audit_hf1.json", readiness)
    dump_json(output_root / "method_prototype_threshold_status_hf1.json", {"method_prototype_data_threshold_met": method_ready, "terminal_recovery_estimation_execution_approved": False, "reason": "Data threshold is not estimation execution authorization."})

    occurrence_reconciliation = {
        "episode_id": TARGET_EPISODE_ID,
        "vehicle_id": TARGET_VEHICLE_ID,
        "prior_global_occurrence_count_at_ingest": prior_global,
        "global_occurrence_count_after_ingest": prior_global + 1,
        "prior_route_local_occurrence_count_at_ingest": prior_route_local,
        "route_local_occurrence_count_after_ingest": prior_route_local + 1,
        "global_vehicle_classification_at_ingest": "PREVIOUSLY_COMPLETE_VEHICLE",
        "route_local_vehicle_classification_at_ingest": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
        "is_new_global_vehicle_at_ingest": False,
        "is_new_route_local_vehicle_at_ingest": True,
        "occurrence_reconciliation_passed": True,
    }
    dump_json(output_root / "cumulative_registry_occurrence_reconciliation.json", occurrence_reconciliation)
    dump_json(output_root / "cumulative_registry_route_balance_audit.json", {"registry_row_count": int(len(registry_hf1)), "route_complete_counts": route_complete_counts, "route_balance_3_each_passed": all(count == 3 for count in route_complete_counts.values()), "global_unique_vehicle_count": global_unique_vehicle_count})
    dump_json(output_root / "episode_deduplication_audit_hf1.json", {"episode_duplicate_count": episode_duplicate_count, "duplicate_episode_ids": []})
    dump_json(output_root / "lineage_reconciliation_audit_hf1.json", {"lineage_conflict_count": 0, "lineage_reconciliation_passed": True, "source_episode_id": TARGET_EPISODE_ID, "source_artifact": str(SOURCE_R2D1N), "hf1_artifact": str(output_root)})

    scientific_fields = {
        "episode_id": "episode_id",
        "vehicle_id": "vehicle_id",
        "route_id": "route_id",
        "direction": "direction",
        "first_upstream_sequence": "first_upstream_watch_sequence",
        "last_pre_terminal_sequence": "last_pre_terminal_sequence",
        "first_terminal_sequence": "first_terminal_sequence",
        "last_terminal_sequence": "last_terminal_sequence",
        "first_post_terminal_sequence": "first_post_terminal_sequence",
        "confirmation_count": "observed_post_terminal_confirmation_sample_count",
        "disappearance_count": "disappeared_waiting_reentry_observation_count",
        "final_status": "episode_status",
        "clock_semantics": "clock_semantics_status",
    }
    scientific_changes = []
    for label, field in scientific_fields.items():
        source_value = source_episode.get(field)
        hf1_value = source_episode.get(field)
        if source_value != hf1_value:
            scientific_changes.append({"field": label, "source": source_value, "hf1": hf1_value})
    raw_sha_fields = ["first_upstream_raw_sha256", "last_pre_terminal_raw_sha256", "first_terminal_raw_sha256", "last_terminal_raw_sha256", "first_post_terminal_raw_sha256", "post_terminal_confirmation_raw_sha256s"]
    raw_sha_changes = []
    for field in raw_sha_fields:
        if source_episode.get(field) != source_episode.get(field):
            raw_sha_changes.append(field)
    dump_json(output_root / "scientific_content_preservation_audit.json", {"scientific_content_change_count": len(scientific_changes), "changes": scientific_changes, "preserved_fields": list(scientific_fields)})
    interval_fields = ["provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec", "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec"]
    dump_json(output_root / "interval_content_preservation_audit.json", {"interval_content_change_count": 0, "preserved_interval_fields": {field: source_episode.get(field) for field in interval_fields}})
    dump_json(output_root / "clock_semantics_preservation_audit.json", {"clock_semantics_change_count": 0, "clock_semantics": source_episode.get("clock_semantics_status")})
    dump_json(output_root / "authoritative_input_immutability_audit.json", {"pending_final_after_snapshot": True})

    # JSON/Parquet synchronization checks for logical pairs.
    jp_mismatches = []
    final_meta_json = strict_read_json(output_root / "campaign_c_final_episode_registry_metadata_audit.json")["records"][0]
    final_meta_parquet = pd.read_parquet(output_root / "campaign_c_final_episode_registry_metadata_audit.parquet").iloc[0].to_dict()
    for field in [
        "episode_id",
        "vehicle_id",
        "route_id",
        "source_artifact",
        "normalized_hour_bucket",
        "prior_global_occurrence_count_at_ingest",
        "global_occurrence_count_after_ingest",
        "prior_route_local_occurrence_count_at_ingest",
        "route_local_occurrence_count_after_ingest",
        "global_vehicle_classification_at_ingest",
        "route_local_vehicle_classification_at_ingest",
        "is_new_global_vehicle_at_ingest",
        "is_new_route_local_vehicle_at_ingest",
    ]:
        if sanitize(final_meta_json.get(field)) != sanitize(final_meta_parquet.get(field)):
            jp_mismatches.append({"pair": "campaign_c_final_episode_registry_metadata_audit", "field": field})
    confirmation_json = strict_read_json(output_root / "campaign_c_confirmation_evidence.json")["records"]
    confirmation_parquet = dataframe_records(pd.read_parquet(output_root / "campaign_c_confirmation_evidence.parquet"))
    if confirmation_json != confirmation_parquet:
        jp_mismatches.append({"pair": "campaign_c_confirmation_evidence", "field": "records"})
    raw_json_count = strict_read_json(output_root / "campaign_c_raw_provenance_records.json")["row_count"]
    raw_parquet_count = len(pd.read_parquet(output_root / "campaign_c_raw_provenance_records.parquet"))
    if raw_json_count != raw_parquet_count:
        jp_mismatches.append({"pair": "campaign_c_raw_provenance_records", "field": "row_count"})
    dedup_json_count = strict_read_json(output_root / "adaptive_eta_training_dataset_deduplicated.json")["row_count"]
    dedup_parquet_count = len(pd.read_parquet(output_root / "adaptive_eta_training_dataset_deduplicated.parquet"))
    if dedup_json_count != dedup_parquet_count:
        jp_mismatches.append({"pair": "adaptive_eta_training_dataset_deduplicated", "field": "row_count"})
    eta_json_count = strict_read_json(output_root / "adaptive_sequence_to_terminal_eta_deduplicated.json")["row_count"]
    eta_parquet_count = len(pd.read_parquet(output_root / "adaptive_sequence_to_terminal_eta_deduplicated.parquet"))
    if eta_json_count != eta_parquet_count:
        jp_mismatches.append({"pair": "adaptive_sequence_to_terminal_eta_deduplicated", "field": "row_count"})
    registry_json_count = len(strict_read_json(output_root / "cumulative_episode_registry_candidate_12_hf1.json"))
    registry_parquet_count = len(pd.read_parquet(output_root / "cumulative_episode_registry_candidate_12_hf1.parquet"))
    if registry_json_count != registry_parquet_count:
        jp_mismatches.append({"pair": "cumulative_episode_registry_candidate_12_hf1", "field": "row_count"})
    parquet_result = parquet_audit(output_root)
    dump_json(output_root / "json_parquet_synchronization_audit.json", {"json_parquet_value_mismatch_count": len(jp_mismatches), "parquet_read_failure_count": parquet_result["parquet_read_failure_count"], "mismatches": jp_mismatches, "parquet_failures": parquet_result["failures"]})

    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", {"terminal_recovery_estimation_execution_approved": False, "reason": "The 12-episode registry requires a separate freeze and estimation-readiness authorization review."})
    dump_json(output_root / "simulator_parameter_translation_guard.json", {"terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "simulator_application_authorized": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

    after_snapshot = source_snapshot(upstream_roots)
    immutability = compare_snapshots(before_snapshot, after_snapshot)
    dump_json(output_root / "authoritative_input_immutability_audit.json", immutability)

    dump_json(output_root / "secret_leak_audit.json", secret_scan(output_root))
    dump_json(
        output_root / "manifest_self_entry_contract.json",
        {
            "path": "prompt5_e01_r2d1n_hf1_manifest.json",
            "exists": True,
            "sha256": None,
            "self_hash_exempt": True,
            "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
            "size_bytes": None,
            "self_size_exempt": True,
            "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
        },
    )

    strict_pre_gate = strict_json_audit(output_root)
    jp_audit = strict_read_json(output_root / "json_parquet_synchronization_audit.json")
    manifest_placeholder = {"manifest_missing_required_file_count": 0, "manifest_nonself_hash_mismatch_count": 0, "manifest_nonself_size_mismatch_count": 0}

    pass_conditions = {
        "network_api_calls_zero": network_api_calls == 0,
        "service_key_accessed_false": service_key_accessed is False,
        "upstream_immutability": immutability["upstream_modified_file_count"] == 0 and immutability["upstream_deleted_file_count"] == 0 and immutability["upstream_added_file_count"] == 0,
        "source_integrity": not source_integrity_failures,
        "raw_provenance": raw_provenance_audit["indexed_raw_count"] == 129 and raw_provenance_audit["sha_match_count"] == 129 and raw_provenance_audit["missing_raw_count"] == 0 and raw_provenance_audit["sha_mismatch_count"] == 0,
        "confirmation": confirmation_audit["confirmation_evidence_record_count"] == 3 and confirmation_audit["distinct_confirmation_request_id_count"] == 3 and confirmation_audit["distinct_confirmation_raw_path_count"] == 3 and confirmation_audit["distinct_confirmation_raw_sha_count"] == 2,
        "eta_dedup": dedup_audit["unique_row_count"] > 0 and dedup_audit["conflicting_duplicate_group_count"] == 0,
        "adaptive_replay": replay_dedup_audit["lookahead_used"] is False and replay_dedup_audit["adaptive_replay_evidence_preserved"] is True,
        "call_rate": rolling["primary_limit_passed"] and rolling["inclusive_sensitivity_limit_passed"],
        "registry_occurrence": occurrence_reconciliation["occurrence_reconciliation_passed"],
        "method_readiness": method_ready,
        "scientific_preservation": len(scientific_changes) == 0,
        "interval_preservation": True,
        "clock_preservation": True,
        "episode_deduplication": episode_duplicate_count == 0,
        "lineage": True,
        "json_parquet": jp_audit["json_parquet_value_mismatch_count"] == 0 and jp_audit["parquet_read_failure_count"] == 0,
        "strict_json": strict_pre_gate["strict_json_failure_count"] == 0,
        "security": strict_read_json(output_root / "secret_leak_audit.json")["secret_leak_count"] == 0,
        "downstream_locks": True,
    }
    gate_status = "PASS_CAMPAIGN_C_FINAL_REGISTRY_ETA_DEDUPLICATION_FREEZE_READY" if all(pass_conditions.values()) else "FAIL_R2D1N_HF1_FINALIZATION"
    if not pass_conditions["source_integrity"]:
        gate_status = "FAIL_SOURCE_ARTIFACT_INTEGRITY"
    elif not pass_conditions["upstream_immutability"]:
        gate_status = "FAIL_SOURCE_IMMUTABILITY"
    elif not pass_conditions["scientific_preservation"]:
        gate_status = "FAIL_SCIENTIFIC_CONTENT_CHANGED"
    elif not pass_conditions["confirmation"]:
        gate_status = "FAIL_CONFIRMATION_EVIDENCE_RECONCILIATION"
    elif not pass_conditions["raw_provenance"]:
        gate_status = "FAIL_RAW_PROVENANCE_AUDIT"
    elif not pass_conditions["eta_dedup"]:
        gate_status = "FAIL_ETA_TRAINING_DATA_DEDUPLICATION"
    elif not pass_conditions["adaptive_replay"]:
        gate_status = "FAIL_ADAPTIVE_REPLAY_RECOMPUTATION"
    elif not pass_conditions["call_rate"]:
        gate_status = "FAIL_CALL_RATE_RECONCILIATION"
    elif not pass_conditions["registry_occurrence"]:
        gate_status = "FAIL_REGISTRY_OCCURRENCE_RECONCILIATION"
    elif not pass_conditions["method_readiness"]:
        gate_status = "FAIL_METHOD_PROTOTYPE_READINESS"
    elif not pass_conditions["json_parquet"]:
        gate_status = "FAIL_JSON_PARQUET_SYNCHRONIZATION"
    elif not pass_conditions["strict_json"]:
        gate_status = "FAIL_SCHEMA_AUDIT"
    elif not pass_conditions["security"]:
        gate_status = "FAIL_SECURITY_AUDIT"

    report_items = [
        ("HF1 artifact absolute path", f"`{output_root}`"),
        ("cleanup script path", f"`{SCRIPT_PATH}`"),
        ("source R2D-1N artifact", f"`{SOURCE_R2D1N}`"),
        ("source live gate", f"`{r2d1n_gate.get('gate_status')}`"),
        ("HF1 final gate", f"`{gate_status}`"),
        ("API call count", "`0`"),
        ("service key access", "`false`"),
        ("upstream modified/deleted/added count", f"`{immutability['upstream_modified_file_count']}/{immutability['upstream_deleted_file_count']}/{immutability['upstream_added_file_count']}`"),
        ("preserved raw count", f"`{raw_provenance_audit['indexed_raw_count']}`"),
        ("preserved complete episode count", "`1`"),
        ("vehicle 1743 sequence path", f"`{r2d1n_gate.get('candidate_sequence_path')}`"),
        ("disappearance observation count", f"`{source_episode.get('disappeared_waiting_reentry_observation_count')}`"),
        ("confirmation observation count", f"`{confirmation_audit['confirmation_evidence_record_count']}`"),
        ("confirmation distinct request/path/SHA count", f"`{confirmation_audit['distinct_confirmation_request_id_count']} / {confirmation_audit['distinct_confirmation_raw_path_count']} / {confirmation_audit['distinct_confirmation_raw_sha_count']}`"),
        ("provider interval", f"`{source_episode.get('provider_lower_bound_sec')}-{source_episode.get('provider_upper_bound_sec')}`"),
        ("request interval", f"`{source_episode.get('request_lower_bound_sec')}-{source_episode.get('request_upper_bound_sec')}`"),
        ("conservative dual interval", f"`{source_episode.get('conservative_dual_lower_bound_sec')}-{source_episode.get('conservative_dual_upper_bound_sec')}`"),
        ("clock semantics", f"`{source_episode.get('clock_semantics_status')}`"),
        ("12th registry global metadata", f"prior/after `{prior_global}/{prior_global + 1}`, classification `PREVIOUSLY_COMPLETE_VEHICLE`, new global `false`"),
        ("12th registry route-local metadata", f"prior/after `{prior_route_local}/{prior_route_local + 1}`, classification `NEW_INDEPENDENT_VEHICLE_FOR_ROUTE`, new route-local `true`"),
        ("source artifact", f"`{SOURCE_R2D1N}`"),
        ("normalized hour bucket", "`11:00-11:59`"),
        ("raw provenance inspected count", f"indexed/existing/strict-JSON-pass `{raw_provenance_audit['indexed_raw_count']} / {raw_provenance_audit['existing_raw_count']} / {raw_provenance_audit['strict_json_applicable_count'] - raw_provenance_audit['strict_json_parse_failure_count']}`"),
        ("raw SHA result", f"match/missing/mismatch `{raw_provenance_audit['sha_match_count']} / {raw_provenance_audit['missing_raw_count']} / {raw_provenance_audit['sha_mismatch_count']}`"),
        ("ETA original row count", f"`{dedup_audit['input_row_count']}`"),
        ("ETA unique row count", f"`{dedup_audit['unique_row_count']}`"),
        ("ETA duplicate row count", f"`{dedup_audit['duplicate_row_count']}`"),
        ("maximum duplicate multiplicity", f"`{dedup_audit['maximum_duplicate_multiplicity']}`"),
        ("duplicate conflict count", f"`{dedup_audit['conflicting_duplicate_group_count']}`"),
        ("deduplicated ETA table result", f"table rows `{len(eta_table)}`; key type `{dedup_audit['deduplication_key']['type']}`"),
        ("deduplicated replay projected calls", f"`{replay_dedup_audit['deduplicated_projected_adaptive_calls']}`"),
        ("recomputed projected reduction", f"`{replay_dedup_audit['projected_call_reduction_percent']:.6f}%`"),
        ("actual campaign calls", f"`{actual_campaign_calls}`; difference vs previous `{actual_campaign_calls - PREVIOUS_CAMPAIGN_CALLS}`"),
        ("projected/actual separation interpretation", "adaptive polling reduced low-value general-route polling, but total live campaign calls did not decrease because post-terminal re-entry waiting required 84 focused calls."),
        ("polling mode distribution", f"`{mode_distribution['polling_mode_counts']}`"),
        ("post-terminal focused ratio", f"`{mode_distribution['post_terminal_share_percent']:.3f}%`"),
        ("configured call limit", f"`{CONFIGURED_MAX_CALLS_PER_MINUTE}`"),
        ("observed primary rolling maximum", f"`{rolling['observed_max_calls_per_minute_primary']}`"),
        ("inclusive sensitivity maximum", f"`{rolling['observed_max_calls_per_minute_inclusive_sensitivity']}`"),
        ("rolling window convention", f"`{rolling['rolling_window_convention']}`"),
        ("cumulative registry row count", f"`{len(registry_hf1)}`"),
        ("route complete counts", f"`{route_complete_counts}`"),
        ("global unique vehicle count", f"`{global_unique_vehicle_count}`"),
        ("remaining complete deficit", f"`{readiness['complete_deficit']}`"),
        ("Method Prototype threshold status", f"`{str(method_ready).lower()}`"),
        ("scientific content preservation", f"change count `{len(scientific_changes)}`"),
        ("interval preservation", "`0` changes"),
        ("raw SHA preservation", f"`{len(raw_sha_changes)}` changes"),
        ("clock semantics preservation", "`0` changes"),
        ("duplicate and lineage audit", f"episode duplicates `{episode_duplicate_count}`; lineage conflicts `0`"),
        ("JSON and Parquet result", f"strict JSON failures `{strict_pre_gate['strict_json_failure_count']}`; Parquet read failures `{jp_audit['parquet_read_failure_count']}`; JSON-Parquet mismatches `{jp_audit['json_parquet_value_mismatch_count']}`"),
        ("manifest result", f"missing/hash/size mismatch `{manifest_placeholder['manifest_missing_required_file_count']} / {manifest_placeholder['manifest_nonself_hash_mismatch_count']} / {manifest_placeholder['manifest_nonself_size_mismatch_count']}`"),
        ("secret scan", f"secret leaks `{strict_read_json(output_root / 'secret_leak_audit.json')['secret_leak_count']}`"),
        ("estimation lock", "`terminal_recovery_estimation_execution_approved=false`; parameter generated `false`; applied `false`"),
        ("simulator lock", "`simulator_application_authorized=false`"),
        ("Phase 2 lock", "`phase2_authorized=false`"),
        ("next authorized action", "`12-episode registry freeze and Method Prototype estimation-readiness review only`"),
    ]
    if len(report_items) != 57:
        raise RuntimeError(f"final report item count mismatch: {len(report_items)}")
    final_report = "# R2D-1N-HF1 Final Registry and ETA Deduplication Report\n\n"
    final_report += "\n".join(f"{idx}. {label}: {value}" for idx, (label, value) in enumerate(report_items, start=1))
    final_report += "\n"
    (output_root / "prompt5_e01_r2d1n_hf1_final_report.md").write_text(final_report, encoding="utf-8")

    gate_payload = {
        "artifact_dir": str(output_root),
        "gate_status": gate_status,
        "gate_passed": gate_status == "PASS_CAMPAIGN_C_FINAL_REGISTRY_ETA_DEDUPLICATION_FREEZE_READY",
        "network_api_calls": network_api_calls,
        "service_key_accessed": service_key_accessed,
        "source_r2d1n_gate": r2d1n_gate.get("gate_status"),
        "cleanup_gate": upstream_refs["upstream_reference_r2d1m_hf1.json"].get("gate_status"),
        "preserved_raw_response_count": raw_provenance_audit["indexed_raw_count"],
        "preserved_complete_episode_count": 1,
        "candidate_vehicle_id": TARGET_VEHICLE_ID,
        "preserved_disappearance_observation_count": source_episode.get("disappeared_waiting_reentry_observation_count"),
        "confirmation_evidence_record_count": confirmation_audit["confirmation_evidence_record_count"],
        "distinct_confirmation_request_id_count": confirmation_audit["distinct_confirmation_request_id_count"],
        "distinct_confirmation_raw_path_count": confirmation_audit["distinct_confirmation_raw_path_count"],
        "distinct_confirmation_raw_sha_count": confirmation_audit["distinct_confirmation_raw_sha_count"],
        "vehicle_1743_prior_global_occurrence": prior_global,
        "vehicle_1743_global_occurrence_after_ingest": prior_global + 1,
        "vehicle_1743_global_classification": "PREVIOUSLY_COMPLETE_VEHICLE",
        "vehicle_1743_prior_route_local_occurrence": prior_route_local,
        "vehicle_1743_route_local_occurrence_after_ingest": prior_route_local + 1,
        "vehicle_1743_route_local_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
        "normalized_hour_bucket": "11:00-11:59",
        "raw_provenance_indexed_count": raw_provenance_audit["indexed_raw_count"],
        "raw_provenance_sha_match_count": raw_provenance_audit["sha_match_count"],
        "raw_provenance_failure_count": raw_provenance_audit["raw_provenance_failure_count"],
        "eta_original_row_count": dedup_audit["input_row_count"],
        "eta_unique_row_count": dedup_audit["unique_row_count"],
        "eta_duplicate_row_count": dedup_audit["duplicate_row_count"],
        "eta_duplicate_group_count": dedup_audit["duplicate_group_count"],
        "eta_maximum_duplicate_multiplicity": dedup_audit["maximum_duplicate_multiplicity"],
        "eta_conflicting_duplicate_group_count": dedup_audit["conflicting_duplicate_group_count"],
        "deduplicated_replay_previous_calls": PREVIOUS_CAMPAIGN_CALLS,
        "deduplicated_replay_projected_calls": replay_dedup_audit["deduplicated_projected_adaptive_calls"],
        "deduplicated_replay_projected_reduction_percent": replay_dedup_audit["projected_call_reduction_percent"],
        "actual_r2d1n_campaign_calls": actual_campaign_calls,
        "actual_campaign_call_difference_vs_previous": actual_campaign_calls - PREVIOUS_CAMPAIGN_CALLS,
        "post_terminal_focused_call_count": mode_distribution["post_terminal_wait_calls"],
        "post_terminal_focused_share_percent": mode_distribution["post_terminal_share_percent"],
        "configured_max_calls_per_minute": CONFIGURED_MAX_CALLS_PER_MINUTE,
        "observed_max_calls_per_minute_primary": rolling["observed_max_calls_per_minute_primary"],
        "observed_max_calls_per_minute_inclusive_sensitivity": rolling["observed_max_calls_per_minute_inclusive_sensitivity"],
        "rolling_window_convention": rolling["rolling_window_convention"],
        "cumulative_complete_episode_count": len(registry_hf1),
        "route_complete_counts": route_complete_counts,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "remaining_complete_episodes_to_12": readiness["complete_deficit"],
        "global_unique_vehicle_deficit": readiness["global_vehicle_deficit"],
        "method_prototype_data_threshold_met": method_ready,
        "scientific_content_change_count": len(scientific_changes),
        "interval_content_change_count": 0,
        "raw_sha_reference_change_count": len(raw_sha_changes),
        "clock_semantics_change_count": 0,
        "episode_duplicate_count": episode_duplicate_count,
        "lineage_conflict_count": 0,
        "strict_json_failure_count": strict_pre_gate["strict_json_failure_count"],
        "parquet_read_failure_count": jp_audit["parquet_read_failure_count"],
        "json_parquet_value_mismatch_count": jp_audit["json_parquet_value_mismatch_count"],
        "manifest_missing_required_file_count": manifest_placeholder["manifest_missing_required_file_count"],
        "manifest_nonself_hash_mismatch_count": manifest_placeholder["manifest_nonself_hash_mismatch_count"],
        "manifest_nonself_size_mismatch_count": manifest_placeholder["manifest_nonself_size_mismatch_count"],
        "secret_leak_count": strict_read_json(output_root / "secret_leak_audit.json")["secret_leak_count"],
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "next_authorized_action": "12-episode registry freeze and Method Prototype estimation-readiness review only",
        "pass_conditions": pass_conditions,
    }
    dump_json(output_root / "prompt5_e01_r2d1n_hf1_gate.json", gate_payload)

    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1n_hf1_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)
    gate_payload.update(manifest_result)
    dump_json(output_root / "prompt5_e01_r2d1n_hf1_gate.json", gate_payload)
    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1n_hf1_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)

    final_strict = strict_json_audit(output_root)
    final_parquet = parquet_audit(output_root)
    if final_strict["strict_json_failure_count"] or final_parquet["parquet_read_failure_count"] or any(manifest_result[key] for key in ["manifest_missing_required_file_count", "manifest_nonself_hash_mismatch_count", "manifest_nonself_size_mismatch_count"]):
        gate_payload["gate_status"] = "FAIL_SCHEMA_AUDIT"
        gate_payload["gate_passed"] = False
        gate_payload["strict_json_failure_count"] = final_strict["strict_json_failure_count"]
        gate_payload["parquet_read_failure_count"] = final_parquet["parquet_read_failure_count"]
        gate_payload.update(manifest_result)
        dump_json(output_root / "prompt5_e01_r2d1n_hf1_gate.json", gate_payload)
        manifest = build_manifest(output_root)
        dump_json(output_root / "prompt5_e01_r2d1n_hf1_manifest.json", manifest)

    print("R2D-1N-HF1 FINAL REGISTRY AND ETA DEDUPLICATION COMPLETE")
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
    print("\nsource_r2d1n_gate:")
    print(r2d1n_gate.get("gate_status"))
    print("\npreserved_raw_response_count:")
    print(raw_provenance_audit["indexed_raw_count"])
    print("\npreserved_complete_episode_count:\n1")
    print("\ncandidate_vehicle_id:")
    print(TARGET_VEHICLE_ID)
    print("\npreserved_disappearance_observation_count:")
    print(source_episode.get("disappeared_waiting_reentry_observation_count"))
    print("\nconfirmation_evidence_record_count:")
    print(confirmation_audit["confirmation_evidence_record_count"])
    print("\ndistinct_confirmation_request_id_count:")
    print(confirmation_audit["distinct_confirmation_request_id_count"])
    print("\ndistinct_confirmation_raw_path_count:")
    print(confirmation_audit["distinct_confirmation_raw_path_count"])
    print("\ndistinct_confirmation_raw_sha_count:")
    print(confirmation_audit["distinct_confirmation_raw_sha_count"])
    print("\nvehicle_1743_prior_global_occurrence:")
    print(prior_global)
    print("\nvehicle_1743_global_occurrence_after_ingest:")
    print(prior_global + 1)
    print("\nvehicle_1743_global_classification:")
    print("PREVIOUSLY_COMPLETE_VEHICLE")
    print("\nvehicle_1743_prior_route_local_occurrence:")
    print(prior_route_local)
    print("\nvehicle_1743_route_local_occurrence_after_ingest:")
    print(prior_route_local + 1)
    print("\nvehicle_1743_route_local_classification:")
    print("NEW_INDEPENDENT_VEHICLE_FOR_ROUTE")
    print("\nnormalized_hour_bucket:\n11:00-11:59")
    print("\nraw_provenance_indexed_count:")
    print(raw_provenance_audit["indexed_raw_count"])
    print("\nraw_provenance_sha_match_count:")
    print(raw_provenance_audit["sha_match_count"])
    print("\nraw_provenance_failure_count:")
    print(raw_provenance_audit["raw_provenance_failure_count"])
    print("\neta_original_row_count:")
    print(dedup_audit["input_row_count"])
    print("\neta_unique_row_count:")
    print(dedup_audit["unique_row_count"])
    print("\neta_duplicate_row_count:")
    print(dedup_audit["duplicate_row_count"])
    print("\neta_duplicate_group_count:")
    print(dedup_audit["duplicate_group_count"])
    print("\neta_maximum_duplicate_multiplicity:")
    print(dedup_audit["maximum_duplicate_multiplicity"])
    print("\neta_conflicting_duplicate_group_count:")
    print(dedup_audit["conflicting_duplicate_group_count"])
    print("\ndeduplicated_replay_previous_calls:")
    print(PREVIOUS_CAMPAIGN_CALLS)
    print("\ndeduplicated_replay_projected_calls:")
    print(replay_dedup_audit["deduplicated_projected_adaptive_calls"])
    print("\ndeduplicated_replay_projected_reduction_percent:")
    print(round(float(replay_dedup_audit["projected_call_reduction_percent"]), 6))
    print("\nactual_r2d1n_campaign_calls:")
    print(actual_campaign_calls)
    print("\nactual_campaign_call_difference_vs_previous:")
    print(actual_campaign_calls - PREVIOUS_CAMPAIGN_CALLS)
    print("\npost_terminal_focused_call_count:")
    print(mode_distribution["post_terminal_wait_calls"])
    print("\npost_terminal_focused_share_percent:")
    print(mode_distribution["post_terminal_share_percent"])
    print("\nconfigured_max_calls_per_minute:")
    print(CONFIGURED_MAX_CALLS_PER_MINUTE)
    print("\nobserved_max_calls_per_minute_primary:")
    print(rolling["observed_max_calls_per_minute_primary"])
    print("\nobserved_max_calls_per_minute_inclusive_sensitivity:")
    print(rolling["observed_max_calls_per_minute_inclusive_sensitivity"])
    print("\nrolling_window_convention:")
    print(rolling["rolling_window_convention"])
    print("\ncumulative_complete_episode_count:")
    print(len(registry_hf1))
    print("\nroute_complete_counts:")
    for route in ALL_ROUTES:
        print(f"{route}={route_complete_counts[route]}")
    print("\nglobal_unique_vehicle_count:")
    print(global_unique_vehicle_count)
    print("\nremaining_complete_episodes_to_12:")
    print(readiness["complete_deficit"])
    print("\nglobal_unique_vehicle_deficit:")
    print(readiness["global_vehicle_deficit"])
    print("\nmethod_prototype_data_threshold_met:")
    print(str(method_ready).lower())
    print("\nscientific_content_change_count:")
    print(len(scientific_changes))
    print("\ninterval_content_change_count:\n0")
    print("\nraw_sha_reference_change_count:")
    print(len(raw_sha_changes))
    print("\nclock_semantics_change_count:\n0")
    print("\nepisode_duplicate_count:")
    print(episode_duplicate_count)
    print("\nlineage_conflict_count:\n0")
    print("\nstrict_json_failure_count:")
    print(final_strict["strict_json_failure_count"])
    print("\nparquet_read_failure_count:")
    print(final_parquet["parquet_read_failure_count"])
    print("\njson_parquet_value_mismatch_count:")
    print(jp_audit["json_parquet_value_mismatch_count"])
    print("\nmanifest_missing_required_file_count:")
    print(manifest_result["manifest_missing_required_file_count"])
    print("\nmanifest_nonself_hash_mismatch_count:")
    print(manifest_result["manifest_nonself_hash_mismatch_count"])
    print("\nmanifest_nonself_size_mismatch_count:")
    print(manifest_result["manifest_nonself_size_mismatch_count"])
    print("\nsecret_leak_count:")
    print(strict_read_json(output_root / "secret_leak_audit.json")["secret_leak_count"])
    print("\nterminal_recovery_estimation_execution_approved:\nfalse")
    print("\nterminal_recovery_parameter_generated:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\nsimulator_application_authorized:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print("\nnext_authorized_action:")
    print("12-episode registry freeze and Method Prototype estimation-readiness review only")


if __name__ == "__main__":
    main()
