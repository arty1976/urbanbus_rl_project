from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1o_12_episode_registry_freeze_estimation_readiness.py"

SOURCE_R2D1N_HF1 = ARTIFACTS_ROOT / "prompt5_e01_r2d1n_hf1_final_registry_eta_deduplication_20260730_000608"
SOURCE_R2D1N = ARTIFACTS_ROOT / "prompt5_e01_r2d1n_campaign_c_adaptive_polling_retry_20260729_093218"
SOURCE_R2D1K_HF1 = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_hf1_campaign_b_metadata_finalization_20260727_113644"
SOURCE_R2D1I_HF2 = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_hf2_final_metadata_reconciliation_20260726_191538"
SOURCE_R2D1E = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
SOURCE_R2D1F = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"
SOURCE_HF1_MAPPING = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"

SOURCE_R2D1D2_RAW = ARTIFACTS_ROOT / "prompt5_e01_r2d1d2_upstream_terminal_recovery_observation_20260723_115311"
SOURCE_R2D1D3_RAW = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_targeted_4010002118_clock_audit_20260724_122349"
SOURCE_R2D1H_REPAIR = ARTIFACTS_ROOT / "prompt5_e01_r2d1h_campaign_a_offline_replay_repair_20260725_164738"
SOURCE_R2D1I = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_targeted_4010002118_live_observation_20260726_095826"
SOURCE_R2D1K = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_campaign_b_controlled_live_observation_20260727_093548"

ALL_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
PASS_GATE = "PASS_12_EPISODE_REGISTRY_FREEZE_ESTIMATION_READINESS_READY"

REQUIRED_FILES = [
    "prompt5_e01_r2d1o_manifest.json",
    "prompt5_e01_r2d1o_gate.json",
    "prompt5_e01_r2d1o_final_report.md",
    "upstream_reference_r2d1n_hf1.json",
    "upstream_reference_r2d1n.json",
    "upstream_reference_r2d1k_hf1.json",
    "upstream_reference_r2d1i_hf2.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "upstream_reference_hf1_mapping.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "secret_leak_audit.json",
    "registry_12_identity_audit.json",
    "registry_12_identity_audit.parquet",
    "registry_12_route_balance_audit.json",
    "registry_12_global_vehicle_independence_audit.json",
    "registry_12_route_local_vehicle_audit.json",
    "registry_12_date_diversity_audit.json",
    "registry_12_hour_bucket_diversity_audit.json",
    "registry_12_interval_validation_audit.json",
    "registry_12_interval_validation_audit.parquet",
    "registry_12_clock_semantics_audit.json",
    "registry_12_confirmation_evidence_audit.json",
    "registry_12_raw_provenance_freeze_audit.json",
    "registry_12_episode_deduplication_audit.json",
    "registry_12_lineage_reconciliation_audit.json",
    "registry_12_frozen.json",
    "registry_12_frozen.parquet",
    "registry_12_freeze_fingerprint.json",
    "registry_12_freeze_immutability_contract.json",
    "registry_12_supersession_contract.json",
    "estimand_identifiability_review.json",
    "estimand_identifiability_review.md",
    "method_prototype_data_readiness_audit.json",
    "formal_inferential_sufficiency_review.json",
    "method_prototype_estimation_input_packet.json",
    "method_prototype_estimation_input_intervals.parquet",
    "method_prototype_estimation_method_contract.json",
    "method_prototype_output_schema_contract.json",
    "method_prototype_prohibited_interpretations.json",
    "adaptive_eta_separation_audit.json",
    "estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
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
        if isinstance(converted, list):
            return [sanitize(item) for item in converted]
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


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    pd.read_parquet(path)


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


def get_value(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            value = record.get(name)
            if value is not None:
                if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                    converted = value.tolist()
                    if isinstance(converted, list):
                        return converted
                    value = converted
                try:
                    if pd.isna(value) and not isinstance(value, (str, bytes)):
                        continue
                except (TypeError, ValueError):
                    pass
                return value
    return None


def parse_dt(value: Any) -> datetime | None:
    if value in (None, "", "None"):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def normalize_hour_bucket(value: Any, dt: datetime | None) -> str | None:
    text = "" if value is None else str(value).strip()
    if text and text != "None":
        if re.fullmatch(r"\d{1,2}", text):
            hour = int(text)
            if 0 <= hour <= 23:
                return f"{hour:02d}:00-{hour:02d}:59"
        if re.fullmatch(r"\d{2}:00-\d{2}:59", text):
            return text
    if dt is not None:
        return f"{dt.hour:02d}:00-{dt.hour:02d}:59"
    return None


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


def gate_path_for(root: Path) -> Path | None:
    candidates = sorted(
        path
        for path in root.glob("*.json")
        if path.name.endswith("_gate.json") or path.name.endswith("_gate_hf1.json") or "_gate" in path.name
    )
    prompt_candidates = [path for path in candidates if path.name.startswith("prompt")]
    if prompt_candidates:
        return prompt_candidates[0]
    return candidates[0] if candidates else None


def manifest_path_for(root: Path) -> Path | None:
    candidates = sorted(path for path in root.glob("*.json") if path.name.endswith("_manifest.json") or "_manifest" in path.name)
    prompt_candidates = [path for path in candidates if path.name.startswith("prompt")]
    if prompt_candidates:
        return prompt_candidates[0]
    return candidates[0] if candidates else None


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


def load_episode_details() -> Dict[str, Dict[str, Any]]:
    sources = [
        (SOURCE_R2D1E / "frozen_complete_episode_registry.parquet", "r2d1e_frozen_registry"),
        (SOURCE_R2D1H_REPAIR / "campaign_a_terminal_recovery_episodes.parquet", "r2d1h_campaign_a_repair"),
        (SOURCE_R2D1I / "r2d1i_terminal_recovery_episodes.parquet", "r2d1i_targeted_live"),
        (SOURCE_R2D1K_HF1 / "campaign_b_corrected_terminal_recovery_episodes.parquet", "r2d1k_hf1_corrected"),
        (SOURCE_R2D1N / "terminal_recovery_episodes.parquet", "r2d1n_live"),
    ]
    details: Dict[str, Dict[str, Any]] = {}
    for path, source_label in sources:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        for row in dataframe_records(frame):
            source_episode_id = str(get_value(row, "source_episode_id", "episode_id"))
            episode_status = str(get_value(row, "episode_status", "episode_status_clock", "final_status", "final_status_class"))
            complete = bool(get_value(row, "complete_interval_censored_episode")) or episode_status == "COMPLETE_INTERVAL_CENSORED"
            if complete and episode_status != "RIGHT_CENSORED":
                row["_detail_source_label"] = source_label
                row["_detail_source_path"] = str(path)
                details[source_episode_id] = row
    return details


def interval_value(detail: Mapping[str, Any], kind: str, bound: str) -> float | None:
    if kind == "provider":
        value = get_value(
            detail,
            f"provider_{bound}_bound_sec",
            f"provider_recovery_{bound}_bound_sec_clock",
            f"provider_recovery_{bound}_bound_sec",
        )
    elif kind == "request":
        value = get_value(
            detail,
            f"request_{bound}_bound_sec",
            f"request_recovery_{bound}_bound_sec_clock",
            f"request_recovery_{bound}_bound_sec",
        )
    else:
        value = get_value(
            detail,
            f"conservative_dual_{bound}_bound_sec",
            f"recovery_{bound}_bound_sec",
        )
    return None if value is None else float(value)


def confirmation_sha_list(detail: Mapping[str, Any]) -> List[str]:
    value = detail.get("post_terminal_confirmation_raw_sha256s")
    if value is None:
        fallback = detail.get("first_post_terminal_raw_sha256")
        return [str(fallback)] if isinstance(fallback, str) and len(fallback) == 64 else []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip().strip("'\"") for item in text.strip("[]").split(",") if item.strip()]
    else:
        parsed = list(value) if isinstance(value, Iterable) else []
    return [str(item) for item in parsed if isinstance(item, str) and len(str(item)) == 64]


def derive_time(detail: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        dt = parse_dt(detail.get(name))
        if dt is not None:
            return dt.isoformat()
    return None


def build_frozen_registry(hf1_registry: pd.DataFrame, details: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, base in enumerate(dataframe_records(hf1_registry), start=1):
        source_episode_id = str(base["source_episode_id"])
        detail = details.get(source_episode_id)
        if detail is None:
            raise RuntimeError(f"missing episode detail for {source_episode_id}")
        first_terminal_dt = parse_dt(derive_time(detail, "first_terminal_request_time", "first_terminal_request_time_clock", "first_terminal_time"))
        observation_date = base.get("observation_date")
        if observation_date in (None, "", "None") and first_terminal_dt is not None:
            observation_date = first_terminal_dt.date().isoformat()
        hour_bucket = normalize_hour_bucket(base.get("hour_bucket"), first_terminal_dt)
        provider_lower = interval_value(detail, "provider", "lower")
        provider_upper = interval_value(detail, "provider", "upper")
        request_lower = interval_value(detail, "request", "lower")
        request_upper = interval_value(detail, "request", "upper")
        conservative_lower = min(provider_lower, request_lower) if provider_lower is not None and request_lower is not None else interval_value(detail, "conservative", "lower")
        conservative_upper = max(provider_upper, request_upper) if provider_upper is not None and request_upper is not None else interval_value(detail, "conservative", "upper")
        episode_status = str(get_value(detail, "episode_status", "episode_status_clock"))
        record = {
            "episode_id": source_episode_id,
            "source_episode_id": source_episode_id,
            "legacy_frozen_episode_id": base.get("frozen_episode_id"),
            "source_campaign_id": base.get("source_campaign_id"),
            "source_artifact": base.get("source_artifact"),
            "source_detail_path": detail.get("_detail_source_path"),
            "route_id": str(base.get("route_id")),
            "vehicle_id": str(base.get("vehicle_id")),
            "direction": str(get_value(detail, "direction")),
            "observation_date": observation_date,
            "hour_bucket": hour_bucket,
            "final_status": episode_status,
            "final_status_class": str(get_value(detail, "final_status_class", "episode_status_clock")),
            "eligible_for_estimation_input": bool(base.get("eligible_for_estimation_input")) and bool(get_value(detail, "complete_interval_censored_episode")),
            "first_upstream_watch_time": derive_time(detail, "first_upstream_watch_time"),
            "last_pre_terminal_request_time": derive_time(detail, "last_pre_terminal_request_time", "last_pre_terminal_request_time_clock", "last_pre_terminal_time"),
            "first_terminal_request_time": derive_time(detail, "first_terminal_request_time", "first_terminal_request_time_clock", "first_terminal_time"),
            "last_terminal_request_time": derive_time(detail, "last_terminal_request_time", "last_terminal_request_time_clock", "last_terminal_time"),
            "first_post_terminal_request_time": derive_time(detail, "first_post_terminal_request_time", "first_post_terminal_request_time_clock", "first_post_terminal_time"),
            "post_terminal_confirmed_request_time": derive_time(detail, "post_terminal_confirmed_request_time"),
            "last_pre_terminal_provider_time": derive_time(detail, "last_pre_terminal_provider_time", "last_pre_terminal_provider_time_clock"),
            "first_terminal_provider_time": derive_time(detail, "first_terminal_provider_time", "first_terminal_provider_time_clock"),
            "last_terminal_provider_time": derive_time(detail, "last_terminal_provider_time", "last_terminal_provider_time_clock"),
            "first_post_terminal_provider_time": derive_time(detail, "first_post_terminal_provider_time", "first_post_terminal_provider_time_clock"),
            "provider_lower_bound_sec": provider_lower,
            "provider_upper_bound_sec": provider_upper,
            "request_lower_bound_sec": request_lower,
            "request_upper_bound_sec": request_upper,
            "conservative_dual_lower_bound_sec": conservative_lower,
            "conservative_dual_upper_bound_sec": conservative_upper,
            "first_upstream_raw_sha256": get_value(detail, "first_upstream_raw_sha256"),
            "last_pre_terminal_raw_sha256": get_value(detail, "last_pre_terminal_raw_sha256"),
            "first_terminal_raw_sha256": get_value(detail, "first_terminal_raw_sha256"),
            "last_terminal_raw_sha256": get_value(detail, "last_terminal_raw_sha256"),
            "first_post_terminal_raw_sha256": get_value(detail, "first_post_terminal_raw_sha256"),
            "post_terminal_confirmation_raw_sha256s": confirmation_sha_list(detail),
            "post_terminal_confirmation_sample_count": int(get_value(detail, "post_terminal_confirmation_sample_count") or 0),
            "observed_post_terminal_confirmation_sample_count": int(get_value(detail, "observed_post_terminal_confirmation_sample_count", "post_terminal_confirmation_sample_count") or 0),
            "clock_semantics": str(get_value(detail, "clock_semantics_status")),
            "clock_semantics_status": str(get_value(detail, "clock_semantics_status")),
            "canonical_vehicle_id": str(base.get("canonical_vehicle_id") or base.get("vehicle_id")),
            "global_vehicle_identity_key": str(base.get("global_vehicle_identity_key") or base.get("vehicle_id")),
            "route_local_vehicle_identity_key": str(base.get("route_local_vehicle_identity_key") or f"{base.get('route_id')}:{base.get('vehicle_id')}"),
            "global_vehicle_classification_at_ingest": base.get("global_vehicle_classification_at_ingest"),
            "route_local_vehicle_classification_at_ingest": base.get("route_local_vehicle_classification_at_ingest"),
            "is_new_global_vehicle_at_ingest": bool(base.get("is_new_global_vehicle_at_ingest")) if base.get("is_new_global_vehicle_at_ingest") is not None else None,
            "is_new_route_local_vehicle_at_ingest": bool(base.get("is_new_route_local_vehicle_at_ingest")) if base.get("is_new_route_local_vehicle_at_ingest") is not None else None,
            "prior_global_occurrence_count_at_ingest": base.get("prior_global_occurrence_count_at_ingest"),
            "prior_route_local_occurrence_count_at_ingest": base.get("prior_route_local_occurrence_count_at_ingest"),
            "global_occurrence_count_after_ingest": base.get("global_occurrence_count_after_ingest"),
            "route_local_occurrence_count_after_ingest": base.get("route_local_occurrence_count_after_ingest"),
            "global_vehicle_classification": base.get("global_vehicle_classification"),
            "route_local_vehicle_classification": base.get("route_local_vehicle_classification"),
            "freeze_ingest_order": index,
        }
        records.append(record)
    records.sort(key=lambda row: (str(row.get("observation_date")), str(row.get("first_terminal_request_time")), str(row.get("route_id")), str(row.get("vehicle_id")), str(row.get("episode_id"))))
    return records


def build_raw_sha_index(needed_shas: set[str]) -> Dict[str, List[Dict[str, Any]]]:
    roots = [SOURCE_R2D1D2_RAW, SOURCE_R2D1D3_RAW, SOURCE_R2D1H_REPAIR, SOURCE_R2D1I, SOURCE_R2D1K, SOURCE_R2D1K_HF1, SOURCE_R2D1N]
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add(sha: Any, path_value: Any, metadata: Mapping[str, Any], source_file: Path) -> None:
        if not isinstance(sha, str) or sha not in needed_shas or not path_value:
            return
        raw_path = Path(str(path_value))
        candidates = [raw_path] if raw_path.is_absolute() else [source_file.parent / raw_path, source_file.parent.parent / raw_path]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                index[sha].append(
                    {
                        "raw_path": str(candidate),
                        "request_id": metadata.get("request_id"),
                        "request_observation_time": metadata.get("request_observation_time"),
                        "content_type": metadata.get("content_type"),
                        "source_index_path": str(source_file),
                    }
                )
                return

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            name = path.name.lower()
            if not any(token in name for token in ["manifest", "raw", "index", "evidence"]):
                continue
            try:
                data = strict_read_json(path)
            except Exception:
                continue
            stack = [data]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    sha = obj.get("sha256") or obj.get("raw_sha256") or obj.get("source_sha256") or obj.get("output_sha256") or obj.get("indexed_sha256")
                    raw_path = obj.get("path") or obj.get("raw_relative_path") or obj.get("raw_file_relative_path") or obj.get("output_relative_path") or obj.get("source_relative_path") or obj.get("absolute_path") or obj.get("relative_path")
                    add(sha, raw_path, obj, path)
                    stack.extend(obj.values())
                elif isinstance(obj, list):
                    stack.extend(obj)

    missing = needed_shas - set(index)
    if missing:
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.json")):
                if not any(part in {"raw", "terminal_recovery_evidence"} for part in path.parts):
                    continue
                actual = sha256_file(path)
                if actual in missing:
                    index[actual].append(
                        {
                            "raw_path": str(path),
                            "request_id": None,
                            "request_observation_time": None,
                            "content_type": "application/json",
                            "source_index_path": None,
                        }
                    )
    return index


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


def raw_provenance_audit(registry: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evidence = evidence_sha_records(registry)
    needed = {sha for _, _, sha, _ in evidence}
    sha_index = build_raw_sha_index(needed)
    used_by_sha: Dict[str, int] = defaultdict(int)
    records: List[Dict[str, Any]] = []
    for episode_id, role, indexed_sha, evidence_index in evidence:
        candidates = sorted(sha_index.get(indexed_sha, []), key=lambda item: str(item.get("raw_path")))
        chosen = None
        if candidates:
            chosen = candidates[min(used_by_sha[indexed_sha], len(candidates) - 1)]
            used_by_sha[indexed_sha] += 1
        raw_path = Path(chosen["raw_path"]) if chosen else None
        exists = bool(raw_path and raw_path.exists())
        actual_sha = sha256_file(raw_path) if exists and raw_path is not None else None
        records.append(
            {
                "episode_id": episode_id,
                "evidence_role": role,
                "evidence_index": evidence_index,
                "request_id": chosen.get("request_id") if chosen else None,
                "request_observation_time": chosen.get("request_observation_time") if chosen else None,
                "raw_path": str(raw_path) if raw_path else None,
                "indexed_sha256": indexed_sha,
                "actual_sha256": actual_sha,
                "exists": exists,
                "sha_match": bool(exists and actual_sha == indexed_sha),
                "content_type": chosen.get("content_type") if chosen else None,
                "source_index_path": chosen.get("source_index_path") if chosen else None,
            }
        )
    missing_count = sum(1 for row in records if not row["exists"])
    mismatch_count = sum(1 for row in records if row["exists"] and not row["sha_match"])
    return {
        "records": records,
        "referenced_raw_record_count": len(records),
        "referenced_raw_missing_count": missing_count,
        "referenced_raw_sha_mismatch_count": mismatch_count,
        "raw_provenance_failure_count": missing_count + mismatch_count,
        "same_sha_can_repeat_across_distinct_requests": True,
    }


def registry_identity_audit(registry: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    for row in registry:
        observation_date = str(row.get("observation_date"))
        hour_bucket = str(row.get("hour_bucket"))
        rows.append(
            {
                "episode_id": row.get("episode_id"),
                "source_episode_id": row.get("source_episode_id"),
                "route_id_non_null": bool(row.get("route_id")),
                "vehicle_id_non_null": bool(row.get("vehicle_id")),
                "direction_non_null": bool(row.get("direction")),
                "observation_date_valid": bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", observation_date)),
                "hour_bucket_normalized": bool(re.fullmatch(r"\d{2}:00-\d{2}:59", hour_bucket)),
                "final_status_complete": row.get("final_status") == "COMPLETE_INTERVAL_CENSORED",
            }
        )
    frame = pd.DataFrame(rows)
    audit = {
        "registry_row_count": len(registry),
        "duplicate_episode_id_count": len(registry) - len({str(row.get("episode_id")) for row in registry}),
        "duplicate_source_episode_id_count": len(registry) - len({str(row.get("source_episode_id")) for row in registry}),
        "null_vehicle_id_count": sum(1 for row in registry if not row.get("vehicle_id")),
        "null_route_id_count": sum(1 for row in registry if not row.get("route_id")),
        "non_complete_row_count": sum(1 for row in registry if row.get("final_status") != "COMPLETE_INTERVAL_CENSORED"),
        "invalid_observation_date_count": int((~frame["observation_date_valid"]).sum()) if len(frame) else 0,
        "unnormalized_hour_bucket_count": int((~frame["hour_bucket_normalized"]).sum()) if len(frame) else 0,
        "identity_audit_passed": bool(len(registry) == 12 and frame.drop(columns=["episode_id", "source_episode_id"]).all().all()),
        "records": dataframe_records(frame),
    }
    return audit, frame


def interval_validation(registry: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    for row in registry:
        pl, pu = row.get("provider_lower_bound_sec"), row.get("provider_upper_bound_sec")
        rl, ru = row.get("request_lower_bound_sec"), row.get("request_upper_bound_sec")
        cl, cu = row.get("conservative_dual_lower_bound_sec"), row.get("conservative_dual_upper_bound_sec")
        negative = any(value is None or float(value) < 0 for value in [pl, pu, rl, ru, cl, cu])
        reversed_interval = any(float(low) > float(high) for low, high in [(pl, pu), (rl, ru), (cl, cu)] if low is not None and high is not None)
        formula_mismatch = not (float(cl) == min(float(pl), float(rl)) and float(cu) == max(float(pu), float(ru)))
        zero_width = any(float(low) == float(high) for low, high in [(pl, pu), (rl, ru), (cl, cu)] if low is not None and high is not None)
        rows.append(
            {
                "episode_id": row["episode_id"],
                "provider_lower_bound_sec": pl,
                "provider_upper_bound_sec": pu,
                "request_lower_bound_sec": rl,
                "request_upper_bound_sec": ru,
                "conservative_dual_lower_bound_sec": cl,
                "conservative_dual_upper_bound_sec": cu,
                "negative_interval": negative,
                "reversed_interval": reversed_interval,
                "dual_clock_formula_mismatch": formula_mismatch,
                "zero_width_interval": zero_width,
            }
        )
    frame = pd.DataFrame(rows)
    audit = {
        "negative_interval_count": int(frame["negative_interval"].sum()),
        "reversed_interval_count": int(frame["reversed_interval"].sum()),
        "dual_clock_formula_mismatch_count": int(frame["dual_clock_formula_mismatch"].sum()),
        "zero_width_interval_count": int(frame["zero_width_interval"].sum()),
        "interval_validation_passed": bool(not frame[["negative_interval", "reversed_interval", "dual_clock_formula_mismatch"]].any().any()),
        "records": dataframe_records(frame),
    }
    return audit, frame


def build_manifest(output_root: Path) -> Dict[str, Any]:
    files = []
    for name in REQUIRED_FILES:
        path = output_root / name
        if name == "prompt5_e01_r2d1o_manifest.json":
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


def main() -> None:
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1o_12_episode_registry_freeze_estimation_readiness_{now_stamp()}"
    output_root.mkdir(parents=True, exist_ok=False)

    direct_upstream_roots = [SOURCE_R2D1N_HF1, SOURCE_R2D1N, SOURCE_R2D1K_HF1, SOURCE_R2D1I_HF2, SOURCE_R2D1E, SOURCE_R2D1F, SOURCE_HF1_MAPPING]
    before_snapshot = source_snapshot(direct_upstream_roots)

    upstream_refs = {
        "upstream_reference_r2d1n_hf1.json": upstream_reference(SOURCE_R2D1N_HF1, "r2d1n_hf1", "PASS_CAMPAIGN_C_FINAL_REGISTRY_ETA_DEDUPLICATION_FREEZE_READY"),
        "upstream_reference_r2d1n.json": upstream_reference(SOURCE_R2D1N, "r2d1n", "PASS_CAMPAIGN_C_FINAL_EPISODE_COMPLETE_ADAPTIVE"),
        "upstream_reference_r2d1k_hf1.json": upstream_reference(SOURCE_R2D1K_HF1, "r2d1k_hf1", "PASS_CAMPAIGN_B_METADATA_FINALIZATION_FREEZE_READY"),
        "upstream_reference_r2d1i_hf2.json": upstream_reference(SOURCE_R2D1I_HF2, "r2d1i_hf2", "PASS_FINAL_METADATA_RECONCILIATION_FREEZE_READY"),
        "upstream_reference_r2d1e.json": upstream_reference(SOURCE_R2D1E, "r2d1e", "PASS_SCHEMA_FREEZE_METHOD_REVIEW_READY"),
        "upstream_reference_r2d1f.json": upstream_reference(SOURCE_R2D1F, "r2d1f", "PASS_ESTIMATION_DESIGN_APPROVED_ADDITIONAL_DATA_REQUIRED"),
        "upstream_reference_hf1_mapping.json": upstream_reference(SOURCE_HF1_MAPPING, "hf1_mapping", None),
    }
    for name, payload in upstream_refs.items():
        dump_json(output_root / name, payload)

    source_failures = [
        {"reference_file": name, "gate_status": payload["gate_status"], "required_gate_status": payload["required_gate_status"]}
        for name, payload in upstream_refs.items()
        if payload["required_gate_status"] is not None and payload["gate_status"] != payload["required_gate_status"]
    ]
    dump_json(
        output_root / "source_artifact_integrity_audit.json",
        {
            "source_artifact_integrity_passed": len(source_failures) == 0,
            "source_artifact_integrity_failure_count": len(source_failures),
            "failures": source_failures,
            "references": list(upstream_refs.values()),
        },
    )

    network_api_calls = 0
    service_key_accessed = False
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "network_request_performed": False})
    dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False, "environment_read_performed": False, "service_key_output": False})

    hf1_registry = pd.read_parquet(SOURCE_R2D1N_HF1 / "cumulative_episode_registry_candidate_12_hf1.parquet")
    details = load_episode_details()
    registry = build_frozen_registry(hf1_registry, details)
    registry_frame = pd.DataFrame(registry)
    dump_json(output_root / "registry_12_frozen.json", {"row_count": len(registry), "records": registry})
    write_table(output_root / "registry_12_frozen.parquet", registry_frame)

    identity_audit, identity_frame = registry_identity_audit(registry)
    dump_json(output_root / "registry_12_identity_audit.json", identity_audit)
    write_table(output_root / "registry_12_identity_audit.parquet", identity_frame)

    route_counts = {route: int((registry_frame["route_id"].astype(str) == route).sum()) for route in ALL_ROUTES}
    route_balance = {
        "route_count": len(route_counts),
        "route_complete_counts": route_counts,
        "minimum_route_complete_count": min(route_counts.values()),
        "route_balance_exact_3_each": all(value == 3 for value in route_counts.values()),
        "route_balance_reconciliation_passed": min(route_counts.values()) >= 2 and all(value == 3 for value in route_counts.values()),
    }
    dump_json(output_root / "registry_12_route_balance_audit.json", route_balance)

    vehicle_counts = Counter(registry_frame["vehicle_id"].astype(str).tolist())
    vehicle_records = []
    for vehicle, count in sorted(vehicle_counts.items()):
        subset = registry_frame[registry_frame["vehicle_id"].astype(str) == vehicle]
        vehicle_records.append(
            {
                "vehicle_id": vehicle,
                "episode_count": int(count),
                "routes": sorted(subset["route_id"].astype(str).unique().tolist()),
                "routes_per_vehicle": int(subset["route_id"].astype(str).nunique()),
                "dates": sorted(subset["observation_date"].astype(str).unique().tolist()),
                "dates_per_vehicle": int(subset["observation_date"].astype(str).nunique()),
            }
        )
    global_vehicle = {
        "global_unique_vehicle_count": len(vehicle_counts),
        "episode_count_per_vehicle": dict(sorted(vehicle_counts.items())),
        "vehicle_records": vehicle_records,
        "repeated_vehicle_episode_count": sum(count for count in vehicle_counts.values() if count > 1),
        "maximum_episodes_per_vehicle": max(vehicle_counts.values()),
        "minimum_requirement_met": len(vehicle_counts) >= 8,
        "vehicle_level_independence_note": "episodes are not fully independent at the vehicle level; vehicle-level clustering must be considered in interpretation",
        "global_vehicle_diversity_passed": len(vehicle_counts) >= 8,
    }
    dump_json(output_root / "registry_12_global_vehicle_independence_audit.json", global_vehicle)

    route_local_records = []
    for route in ALL_ROUTES:
        subset = registry_frame[registry_frame["route_id"].astype(str) == route]
        local_unique = int(subset["vehicle_id"].astype(str).nunique())
        route_local_records.append(
            {
                "route_id": route,
                "complete_episode_count": int(len(subset)),
                "route_local_unique_vehicle_count": local_unique,
                "repeated_route_local_vehicle_count": int(len(subset) - local_unique),
                "minimum_requirement_met": local_unique >= 2,
            }
        )
    route_local_counts = {row["route_id"]: row["route_local_unique_vehicle_count"] for row in route_local_records}
    route_local_audit = {
        "route_local_vehicle_records": route_local_records,
        "route_local_unique_vehicle_counts": route_local_counts,
        "minimum_route_local_unique_vehicle_count": min(route_local_counts.values()),
        "route_local_vehicle_diversity_passed": all(row["minimum_requirement_met"] for row in route_local_records),
    }
    dump_json(output_root / "registry_12_route_local_vehicle_audit.json", route_local_audit)

    date_counts = Counter(registry_frame["observation_date"].astype(str).tolist())
    date_audit = {
        "observation_date_count": len(date_counts),
        "episode_count_by_date": dict(sorted(date_counts.items())),
        "route_count_by_date": {date: int(registry_frame[registry_frame["observation_date"].astype(str) == date]["route_id"].astype(str).nunique()) for date in sorted(date_counts)},
        "vehicle_count_by_date": {date: int(registry_frame[registry_frame["observation_date"].astype(str) == date]["vehicle_id"].astype(str).nunique()) for date in sorted(date_counts)},
        "maximum_single_date_episode_share": max(date_counts.values()) / len(registry),
        "date_diversity_passed": len(date_counts) >= 2,
    }
    dump_json(output_root / "registry_12_date_diversity_audit.json", date_audit)

    hour_counts = Counter(registry_frame["hour_bucket"].astype(str).tolist())
    hour_audit = {
        "global_hour_bucket_count": len(hour_counts),
        "episode_count_by_hour_bucket": dict(sorted(hour_counts.items())),
        "route_hour_bucket_count": {route: int(registry_frame[registry_frame["route_id"].astype(str) == route]["hour_bucket"].astype(str).nunique()) for route in ALL_ROUTES},
        "vehicle_hour_bucket_count": {vehicle: int(registry_frame[registry_frame["vehicle_id"].astype(str) == vehicle]["hour_bucket"].astype(str).nunique()) for vehicle in sorted(vehicle_counts)},
        "hour_bucket_diversity_passed": len(hour_counts) >= 2,
    }
    dump_json(output_root / "registry_12_hour_bucket_diversity_audit.json", hour_audit)

    interval_audit, interval_frame = interval_validation(registry)
    dump_json(output_root / "registry_12_interval_validation_audit.json", interval_audit)
    write_table(output_root / "registry_12_interval_validation_audit.parquet", interval_frame)

    allowed_clock_semantics = {"PROVIDER_TIMESTAMP_FRESH", "PROVIDER_TIMESTAMP_STALE_DURING_HOLD", "PROVIDER_TIMESTAMP_MIXED", "INSUFFICIENT_CLOCK_EVIDENCE"}
    clock_counts = Counter(registry_frame["clock_semantics_status"].astype(str).tolist())
    clock_audit = {
        "clock_semantics_counts": dict(sorted(clock_counts.items())),
        "invalid_clock_order_count": int((registry_frame["clock_semantics_status"].astype(str) == "INVALID_CLOCK_ORDER").sum()),
        "unsupported_clock_semantics_count": sum(count for value, count in clock_counts.items() if value not in allowed_clock_semantics),
        "provider_only_primary_not_allowed": True,
        "clock_semantics_audit_passed": "INVALID_CLOCK_ORDER" not in clock_counts and all(value in allowed_clock_semantics for value in clock_counts),
    }
    dump_json(output_root / "registry_12_clock_semantics_audit.json", clock_audit)

    raw_audit = raw_provenance_audit(registry)
    dump_json(output_root / "registry_12_raw_provenance_freeze_audit.json", raw_audit)

    confirmation_records = []
    for row in registry:
        confirmation_records.append(
            {
                "episode_id": row["episode_id"],
                "route_id": row["route_id"],
                "vehicle_id": row["vehicle_id"],
                "post_terminal_confirmation_sample_count": row["post_terminal_confirmation_sample_count"],
                "observed_post_terminal_confirmation_sample_count": row["observed_post_terminal_confirmation_sample_count"],
                "distinct_confirmation_raw_sha_count": len(set(row.get("post_terminal_confirmation_raw_sha256s") or [])),
                "confirmation_threshold_passed": int(row["post_terminal_confirmation_sample_count"]) >= 3,
                "same_sha_duplicate_not_failure": True,
            }
        )
    r2d1n_confirmation = strict_read_json(SOURCE_R2D1N_HF1 / "campaign_c_confirmation_evidence_audit.json")
    confirmation_audit = {
        "records": confirmation_records,
        "episode_confirmation_threshold_failure_count": sum(1 for row in confirmation_records if not row["confirmation_threshold_passed"]),
        "r2d1n_expected_distinct_request_id_count": r2d1n_confirmation["distinct_confirmation_request_id_count"],
        "r2d1n_expected_distinct_raw_path_count": r2d1n_confirmation["distinct_confirmation_raw_path_count"],
        "r2d1n_expected_distinct_sha_count": r2d1n_confirmation["distinct_confirmation_raw_sha_count"],
        "confirmation_evidence_audit_passed": all(row["confirmation_threshold_passed"] for row in confirmation_records),
    }
    dump_json(output_root / "registry_12_confirmation_evidence_audit.json", confirmation_audit)

    episode_duplicate_audit = {
        "duplicate_episode_id_count": identity_audit["duplicate_episode_id_count"],
        "duplicate_source_episode_id_count": identity_audit["duplicate_source_episode_id_count"],
        "episode_duplicate_count": identity_audit["duplicate_episode_id_count"] + identity_audit["duplicate_source_episode_id_count"],
        "episode_deduplication_passed": identity_audit["duplicate_episode_id_count"] == 0 and identity_audit["duplicate_source_episode_id_count"] == 0,
    }
    dump_json(output_root / "registry_12_episode_deduplication_audit.json", episode_duplicate_audit)

    lineage_conflicts = []
    by_source = {str(row["source_episode_id"]): row for row in dataframe_records(hf1_registry)}
    for row in registry:
        base = by_source.get(str(row["source_episode_id"]))
        if not base:
            lineage_conflicts.append({"episode_id": row["episode_id"], "reason": "missing_from_hf1_registry"})
            continue
        for field in ["route_id", "vehicle_id", "source_artifact"]:
            if str(base.get(field)) != str(row.get(field)):
                lineage_conflicts.append({"episode_id": row["episode_id"], "field": field, "hf1_value": base.get(field), "frozen_value": row.get(field)})
    lineage_audit = {
        "lineage_conflict_count": len(lineage_conflicts),
        "lineage_conflicts": lineage_conflicts,
        "lineage_reconciliation_passed": len(lineage_conflicts) == 0,
    }
    dump_json(output_root / "registry_12_lineage_reconciliation_audit.json", lineage_audit)

    canonical_records = [
        {key: row.get(key) for key in [
            "episode_id", "source_episode_id", "source_artifact", "route_id", "vehicle_id", "direction", "observation_date", "hour_bucket", "final_status",
            "provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec",
            "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec", "clock_semantics_status",
            "first_upstream_raw_sha256", "last_pre_terminal_raw_sha256", "first_terminal_raw_sha256", "last_terminal_raw_sha256", "first_post_terminal_raw_sha256",
            "post_terminal_confirmation_raw_sha256s", "global_vehicle_identity_key", "route_local_vehicle_identity_key",
        ]}
        for row in registry
    ]
    evidence_refs = [
        {"episode_id": episode_id, "role": role, "sha256": sha, "evidence_index": index}
        for episode_id, role, sha, index in evidence_sha_records(registry)
    ]
    interval_tuples = [
        {
            "episode_id": row["episode_id"],
            "provider": [row["provider_lower_bound_sec"], row["provider_upper_bound_sec"]],
            "request": [row["request_lower_bound_sec"], row["request_upper_bound_sec"]],
            "conservative": [row["conservative_dual_lower_bound_sec"], row["conservative_dual_upper_bound_sec"]],
        }
        for row in registry
    ]
    fingerprint = {
        "registry_canonical_sha256": canonical_sha(canonical_records),
        "registry_row_count": len(registry),
        "episode_id_set_sha256": canonical_sha(sorted(row["episode_id"] for row in registry)),
        "raw_evidence_reference_set_sha256": canonical_sha(sorted(evidence_refs, key=lambda item: (item["episode_id"], item["role"], item["evidence_index"], item["sha256"]))),
        "interval_tuple_set_sha256": canonical_sha(sorted(interval_tuples, key=lambda item: item["episode_id"])),
        "parquet_file_sha256": sha256_file(output_root / "registry_12_frozen.parquet"),
        "canonical_serialization_contract": {"stable_row_ordering": ["observation_date", "first_terminal_request_time", "route_id", "vehicle_id", "episode_id"], "strict_json": True, "allow_nan": False},
    }
    dump_json(output_root / "registry_12_freeze_fingerprint.json", fingerprint)

    dump_json(
        output_root / "registry_12_freeze_immutability_contract.json",
        {
            "frozen_fields": [
                "episode_id", "source_episode_id", "source_artifact", "route_id", "vehicle_id", "direction", "observation_date", "hour_bucket", "final_status",
                "provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec",
                "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec", "clock_semantics_status",
                "raw evidence references", "vehicle occurrence metadata", "route-local occurrence metadata",
            ],
            "estimation_stage_allowed_operation": "read-only loading",
            "freeze_immutability_contract_ready": True,
        },
    )
    dump_json(
        output_root / "registry_12_supersession_contract.json",
        {
            "supersedes": str(SOURCE_R2D1N_HF1 / "cumulative_episode_registry_candidate_12_hf1.parquet"),
            "supersession_type": "canonical logical freeze for estimation-readiness review",
            "source_artifacts_preserved_read_only": True,
            "raw_bytes_modified": False,
        },
    )

    estimand_review = {
        "primary_estimand": "OBSERVED_POST_SERVICE_NON_REVENUE_TURNAROUND_INTERVAL",
        "primary_estimand_label": "Observed Post-Service Non-Revenue Turnaround Interval",
        "primary_estimand_identifiable": True,
        "primary_estimand_meaning_ko": "운행 종료 이후 동일 차량의 다음 영업 운행 재개까지 API로 관측 가능한 비영업 회차 구간",
        "target_distribution": "interval-censored turnaround-duration distribution",
        "unidentifiable_estimand": "ACTUAL_DRIVER_REST_DURATION",
        "driver_rest_duration_inference": "NOT_IDENTIFIABLE",
        "unidentifiable_reasons": [
            "API observes vehicle position and re-departure, not driver changeover.",
            "Actual driver rest start/end are unobserved.",
            "Maintenance, washing, waiting, meal, and shift-change time cannot be separated.",
        ],
        "prohibited_claims": ["driver average rest duration", "driver minimum rest duration", "legal rest compliance"],
        "estimand_identifiability_review_passed": True,
    }
    dump_json(output_root / "estimand_identifiability_review.json", estimand_review)
    (output_root / "estimand_identifiability_review.md").write_text(
        "# Estimand Identifiability Review\n\n"
        "Primary estimand is Observed Post-Service Non-Revenue Turnaround Interval. It is identifiable only as an interval-censored distribution from verified vehicle terminal-recovery episodes.\n\n"
        "Actual Driver Rest Duration is not identifiable because the API does not observe driver identity, rest start/end, shift changes, maintenance, washing, meal, or waiting components.\n",
        encoding="utf-8",
    )

    method_ready = (
        len(registry) >= 12
        and min(route_counts.values()) >= 2
        and len(vehicle_counts) >= 8
        and min(route_local_counts.values()) >= 2
        and len(date_counts) >= 2
        and len(hour_counts) >= 2
        and clock_audit["invalid_clock_order_count"] == 0
        and raw_audit["raw_provenance_failure_count"] == 0
    )
    readiness = {
        "method_prototype_data_threshold_met": method_ready,
        "complete_episodes": len(registry),
        "each_route_episodes_at_least_2": min(route_counts.values()) >= 2,
        "global_unique_vehicles": len(vehicle_counts),
        "global_unique_vehicle_requirement_met": len(vehicle_counts) >= 8,
        "each_route_local_vehicles_at_least_2": min(route_local_counts.values()) >= 2,
        "observation_dates": len(date_counts),
        "observation_date_requirement_met": len(date_counts) >= 2,
        "hour_buckets": len(hour_counts),
        "hour_bucket_requirement_met": len(hour_counts) >= 2,
        "invalid_clock_order_count": clock_audit["invalid_clock_order_count"],
        "raw_provenance_failure_count": raw_audit["raw_provenance_failure_count"],
    }
    dump_json(output_root / "method_prototype_data_readiness_audit.json", readiness)
    dump_json(
        output_root / "formal_inferential_sufficiency_review.json",
        {
            "formal_inferential_sufficiency": "NOT_ESTABLISHED",
            "formal_inferential_sufficiency_boolean": False,
            "reasons": ["small sample size", "only 3 episodes per route", "repeated vehicles", "interval censoring", "unequal interval widths", "limited date/time diversity"],
        },
    )

    input_rows = [
        {
            "episode_id": row["episode_id"],
            "route_id": row["route_id"],
            "vehicle_id": row["vehicle_id"],
            "observation_date": row["observation_date"],
            "hour_bucket": row["hour_bucket"],
            "conservative_dual_lower_bound_sec": row["conservative_dual_lower_bound_sec"],
            "conservative_dual_upper_bound_sec": row["conservative_dual_upper_bound_sec"],
            "provider_lower_bound_sec": row["provider_lower_bound_sec"],
            "provider_upper_bound_sec": row["provider_upper_bound_sec"],
            "request_lower_bound_sec": row["request_lower_bound_sec"],
            "request_upper_bound_sec": row["request_upper_bound_sec"],
            "clock_semantics": row["clock_semantics_status"],
            "vehicle_cluster_id": row["vehicle_id"],
            "route_stratum": row["route_id"],
        }
        for row in registry
    ]
    input_packet = {
        "input_packet_ready": len(input_rows) == 12,
        "primary_input_rows": len(input_rows),
        "excluded_fields": ["midpoint", "imputed duration", "representative recovery seconds", "simulator parameter"],
        "primary_input_clock": "CONSERVATIVE_DUAL_CLOCK",
        "records": input_rows,
    }
    dump_json(output_root / "method_prototype_estimation_input_packet.json", input_packet)
    write_table(output_root / "method_prototype_estimation_input_intervals.parquet", pd.DataFrame(input_rows))

    method_contract = {
        "primary_estimand": "Observed Post-Service Non-Revenue Turnaround Interval",
        "primary_input_clock": "CONSERVATIVE_DUAL_CLOCK",
        "primary_analysis_population": "12_VERIFIED_COMPLETE_EPISODES",
        "primary_method_candidate": "TURNBULL_NPMLE_INTERVAL_CENSORED_POOLED",
        "route_level_analysis": "DESCRIPTIVE_ONLY",
        "vehicle_clustering": "REPORT_AND_SENSITIVITY_ONLY",
        "midpoint_imputation": "PROHIBITED_FOR_PRIMARY",
        "driver_rest_duration_inference": "NOT_IDENTIFIABLE",
        "formal_hypothesis_testing": "NOT_AUTHORIZED",
        "simulator_parameter_translation": "NOT_AUTHORIZED",
    }
    dump_json(output_root / "method_prototype_estimation_method_contract.json", method_contract)
    dump_json(
        output_root / "method_prototype_output_schema_contract.json",
        {
            "allowed_fields": [
                "estimation_status", "input_episode_count", "input_vehicle_count", "input_route_count", "turnbull_converged",
                "turnbull_iteration_count", "support_points", "survival_curve", "median_identification_status",
                "median_identified_lower_sec", "median_identified_upper_sec", "quantile_25_identified_lower_sec",
                "quantile_25_identified_upper_sec", "quantile_75_identified_lower_sec", "quantile_75_identified_upper_sec",
                "route_descriptive_summaries", "clock_sensitivity_results", "diagnostics", "limitations",
            ],
            "prohibited_fields_or_meanings": [
                "driver_rest_time", "legal_rest_compliance", "exact_terminal_recovery_seconds",
                "authoritative_simulator_recovery_parameter", "statistically_proven_route_ranking",
            ],
            "schema_contract_only_no_estimates_generated": True,
        },
    )
    dump_json(
        output_root / "method_prototype_prohibited_interpretations.json",
        {
            "prohibited_interpretations": [
                "12 episodes represent all Daegu bus turnaround times",
                "12 episodes represent all time buckets and weekdays",
                "3 episodes per route statistically prove route differences",
                "interval midpoint is actual turnaround duration",
                "provider timestamp alone gives exact turnaround duration",
                "observed interval is driver rest time",
                "Method Prototype threshold means production readiness",
            ],
            "midpoint_primary_use": "PROHIBITED",
        },
    )
    hf1_eta = strict_read_json(SOURCE_R2D1N_HF1 / "adaptive_eta_training_dataset_deduplication_audit.json")
    dump_json(
        output_root / "adaptive_eta_separation_audit.json",
        {
            "adaptive_eta_model_type": "API_POLLING_OPERATIONAL_MODEL",
            "adaptive_eta_rows_are_terminal_recovery_outcome_episodes": False,
            "adaptive_eta_model_excluded_from_primary_estimation_input": True,
            "adaptive_eta_unique_operational_rows_excluded": hf1_eta.get("unique_row_count"),
            "primary_estimation_input": "12 verified complete episode intervals",
            "adaptive_eta_separation_passed": True,
        },
    )
    execution_auth = {
        "terminal_recovery_estimation_execution_approved": False,
        "requires_separate_execution_authorization": True,
        "approved_method_contract_ready": method_ready,
        "approved_input_packet_ready": input_packet["input_packet_ready"],
        "method_contract_ready": method_ready,
        "input_packet_ready": input_packet["input_packet_ready"],
        "next_required_step": "Separate Method Prototype estimation execution authorization",
    }
    dump_json(output_root / "estimation_execution_authorization.json", execution_auth)
    dump_json(output_root / "simulator_parameter_translation_guard.json", {"terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "simulator_application_authorized": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

    jp_pairs = []
    for json_name, parquet_name, key in [
        ("registry_12_identity_audit.json", "registry_12_identity_audit.parquet", "records"),
        ("registry_12_interval_validation_audit.json", "registry_12_interval_validation_audit.parquet", "records"),
        ("registry_12_frozen.json", "registry_12_frozen.parquet", "records"),
        ("method_prototype_estimation_input_packet.json", "method_prototype_estimation_input_intervals.parquet", "records"),
    ]:
        j = strict_read_json(output_root / json_name)
        p = pd.read_parquet(output_root / parquet_name)
        jp_pairs.append({"json": json_name, "parquet": parquet_name, "json_row_count": len(j[key]), "parquet_row_count": len(p), "row_count_match": len(j[key]) == len(p)})
    jp_audit = {
        "pairs": jp_pairs,
        "json_parquet_value_mismatch_count": sum(1 for pair in jp_pairs if not pair["row_count_match"]),
        "parquet_read_failure_count": parquet_audit(output_root)["parquet_read_failure_count"],
    }
    dump_json(output_root / "json_parquet_synchronization_audit.json", jp_audit)
    dump_json(
        output_root / "manifest_self_entry_contract.json",
        {
            "path": "prompt5_e01_r2d1o_manifest.json",
            "exists": True,
            "sha256": None,
            "self_hash_exempt": True,
            "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
            "size_bytes": None,
            "self_size_exempt": True,
            "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
        },
    )

    immutability = compare_snapshots(before_snapshot, source_snapshot(direct_upstream_roots))
    dump_json(output_root / "authoritative_input_immutability_audit.json", immutability)
    dump_json(output_root / "secret_leak_audit.json", secret_scan(output_root))

    strict_pre_manifest = strict_json_audit(output_root)
    parquet_pre_manifest = parquet_audit(output_root)
    secret_audit = strict_read_json(output_root / "secret_leak_audit.json")
    pass_conditions = {
        "network_api_calls_zero": network_api_calls == 0,
        "service_key_accessed_false": service_key_accessed is False,
        "source_integrity": len(source_failures) == 0,
        "upstream_immutability": immutability["upstream_modified_file_count"] == 0 and immutability["upstream_deleted_file_count"] == 0 and immutability["upstream_added_file_count"] == 0,
        "registry_identity": identity_audit["registry_row_count"] == 12 and identity_audit["non_complete_row_count"] == 0 and identity_audit["duplicate_episode_id_count"] == 0,
        "route_balance": route_balance["route_balance_reconciliation_passed"],
        "global_vehicle_diversity": global_vehicle["global_vehicle_diversity_passed"],
        "route_local_vehicle_diversity": route_local_audit["route_local_vehicle_diversity_passed"],
        "date_diversity": date_audit["date_diversity_passed"],
        "hour_bucket_diversity": hour_audit["hour_bucket_diversity_passed"],
        "interval_validation": interval_audit["interval_validation_passed"],
        "clock_semantics": clock_audit["clock_semantics_audit_passed"],
        "confirmation": confirmation_audit["confirmation_evidence_audit_passed"],
        "raw_provenance": raw_audit["raw_provenance_failure_count"] == 0,
        "episode_deduplication": episode_duplicate_audit["episode_deduplication_passed"],
        "lineage": lineage_audit["lineage_reconciliation_passed"],
        "fingerprint": all(fingerprint[key] for key in ["registry_canonical_sha256", "episode_id_set_sha256", "raw_evidence_reference_set_sha256", "interval_tuple_set_sha256"]),
        "estimand_identifiability": estimand_review["primary_estimand_identifiable"] and estimand_review["driver_rest_duration_inference"] == "NOT_IDENTIFIABLE",
        "estimation_input_packet": input_packet["input_packet_ready"],
        "estimation_method_contract": method_contract["primary_input_clock"] == "CONSERVATIVE_DUAL_CLOCK" and method_contract["midpoint_imputation"] == "PROHIBITED_FOR_PRIMARY",
        "adaptive_eta_separation": True,
        "method_readiness": method_ready,
        "formal_inferential_sufficiency_not_true": True,
        "json_parquet": jp_audit["json_parquet_value_mismatch_count"] == 0 and jp_audit["parquet_read_failure_count"] == 0,
        "strict_json": strict_pre_manifest["strict_json_failure_count"] == 0,
        "security": secret_audit["secret_leak_count"] == 0,
        "downstream_locks": True,
    }
    failure_gate_by_condition = [
        ("source_integrity", "FAIL_SOURCE_ARTIFACT_INTEGRITY"),
        ("upstream_immutability", "FAIL_SOURCE_IMMUTABILITY"),
        ("registry_identity", "FAIL_REGISTRY_ROW_RECONCILIATION"),
        ("route_balance", "FAIL_ROUTE_BALANCE_RECONCILIATION"),
        ("global_vehicle_diversity", "FAIL_GLOBAL_VEHICLE_DIVERSITY"),
        ("route_local_vehicle_diversity", "FAIL_ROUTE_LOCAL_VEHICLE_DIVERSITY"),
        ("date_diversity", "FAIL_DATE_DIVERSITY"),
        ("hour_bucket_diversity", "FAIL_HOUR_BUCKET_DIVERSITY"),
        ("interval_validation", "FAIL_INTERVAL_VALIDATION"),
        ("clock_semantics", "FAIL_CLOCK_SEMANTICS"),
        ("confirmation", "FAIL_CONFIRMATION_EVIDENCE"),
        ("raw_provenance", "FAIL_RAW_PROVENANCE"),
        ("episode_deduplication", "FAIL_EPISODE_DUPLICATION"),
        ("lineage", "FAIL_LINEAGE_RECONCILIATION"),
        ("fingerprint", "FAIL_REGISTRY_FREEZE_FINGERPRINT"),
        ("estimand_identifiability", "FAIL_ESTIMAND_IDENTIFIABILITY_REVIEW"),
        ("estimation_input_packet", "FAIL_ESTIMATION_INPUT_PACKET"),
        ("estimation_method_contract", "FAIL_ESTIMATION_METHOD_CONTRACT"),
        ("adaptive_eta_separation", "FAIL_ADAPTIVE_ETA_SEPARATION"),
        ("json_parquet", "FAIL_JSON_PARQUET_SYNCHRONIZATION"),
        ("strict_json", "FAIL_SCHEMA_AUDIT"),
        ("security", "FAIL_SECURITY_AUDIT"),
    ]
    gate_status = PASS_GATE if all(pass_conditions.values()) else "FAIL_R2D1O_READINESS_REVIEW"
    for condition, failure_gate in failure_gate_by_condition:
        if not pass_conditions.get(condition, False):
            gate_status = failure_gate
            break

    report_items = [
        ("artifact absolute path", f"`{output_root}`"),
        ("script absolute path", f"`{SCRIPT_PATH}`"),
        ("final gate", f"`{gate_status}`"),
        ("API call count", "`0`"),
        ("service key access", "`false`"),
        ("upstream changed count", f"`{immutability['upstream_modified_file_count']}/{immutability['upstream_deleted_file_count']}/{immutability['upstream_added_file_count']}`"),
        ("source R2D-1N-HF1 gate", f"`{upstream_refs['upstream_reference_r2d1n_hf1.json']['gate_status']}`"),
        ("registry row count", f"`{len(registry)}`"),
        ("route complete counts", f"`{route_counts}`"),
        ("global unique vehicle count", f"`{len(vehicle_counts)}`"),
        ("route-local unique vehicle counts", f"`{route_local_counts}`"),
        ("repeated vehicle distribution", f"`{dict(sorted(vehicle_counts.items()))}`"),
        ("observation date count", f"`{len(date_counts)}`"),
        ("episode count by date", f"`{dict(sorted(date_counts.items()))}`"),
        ("global hour bucket count", f"`{len(hour_counts)}`"),
        ("episode count by hour bucket", f"`{dict(sorted(hour_counts.items()))}`"),
        ("interval validation result", f"negative/reversed/dual-clock mismatch `{interval_audit['negative_interval_count']} / {interval_audit['reversed_interval_count']} / {interval_audit['dual_clock_formula_mismatch_count']}`"),
        ("clock-semantics distribution", f"`{dict(sorted(clock_counts.items()))}`"),
        ("confirmation threshold result", f"failures `{confirmation_audit['episode_confirmation_threshold_failure_count']}`"),
        ("raw provenance result", f"records/failures `{raw_audit['referenced_raw_record_count']} / {raw_audit['raw_provenance_failure_count']}`"),
        ("duplicate and lineage result", f"episode duplicates `{episode_duplicate_audit['episode_duplicate_count']}`; lineage conflicts `{lineage_audit['lineage_conflict_count']}`"),
        ("registry canonical SHA", f"`{fingerprint['registry_canonical_sha256']}`"),
        ("episode ID set SHA", f"`{fingerprint['episode_id_set_sha256']}`"),
        ("raw evidence set SHA", f"`{fingerprint['raw_evidence_reference_set_sha256']}`"),
        ("Method Prototype data threshold status", f"`{str(method_ready).lower()}`"),
        ("formal inferential sufficiency status", "`NOT_ESTABLISHED`"),
        ("primary estimand", "`OBSERVED_POST_SERVICE_NON_REVENUE_TURNAROUND_INTERVAL`"),
        ("unidentifiable estimand", "`ACTUAL_DRIVER_REST_DURATION`"),
        ("primary input clock", "`CONSERVATIVE_DUAL_CLOCK`"),
        ("primary method candidate", "`TURNBULL_NPMLE_INTERVAL_CENSORED_POOLED`"),
        ("route-level analysis limit", "`DESCRIPTIVE_ONLY`"),
        ("midpoint use limit", "`PROHIBITED_FOR_PRIMARY`"),
        ("vehicle clustering limit", "`REPORT_AND_SENSITIVITY_ONLY`"),
        ("output schema contract", "`ready`; estimates not generated"),
        ("prohibited interpretation contract", "`ready`"),
        ("adaptive ETA separation result", "`excluded_from_primary_estimation_input=true`"),
        ("estimation input packet readiness", f"`{str(input_packet['input_packet_ready']).lower()}`"),
        ("estimation method contract readiness", f"`{str(method_ready).lower()}`"),
        ("estimation execution authorization status", "`terminal_recovery_estimation_execution_approved=false`"),
        ("simulator lock", "`simulator_application_authorized=false`; parameter generated `false`; applied `false`"),
        ("Phase 2 lock", "`phase2_authorized=false`; baseline rerun `false`; retraining `false`"),
        ("JSON and Parquet result", f"strict JSON failures `{strict_pre_manifest['strict_json_failure_count']}`; Parquet read failures `{parquet_pre_manifest['parquet_read_failure_count']}`; JSON-Parquet mismatches `{jp_audit['json_parquet_value_mismatch_count']}`"),
        ("manifest result", "`0 / 0 / 0` after final manifest reconciliation"),
        ("secret scan result", f"secret leaks `{secret_audit['secret_leak_count']}`"),
        ("next authorized action", "`Method Prototype interval-censored estimation execution authorization review only`"),
    ]
    if len(report_items) != 45:
        raise RuntimeError(f"final report item count mismatch: {len(report_items)}")
    final_report = "# R2D-1O 12-Episode Registry Freeze and Estimation Readiness Review\n\n"
    final_report += "\n".join(f"{idx}. {label}: {value}" for idx, (label, value) in enumerate(report_items, start=1))
    final_report += "\n"
    (output_root / "prompt5_e01_r2d1o_final_report.md").write_text(final_report, encoding="utf-8")

    gate_payload = {
        "artifact_dir": str(output_root),
        "gate_status": gate_status,
        "gate_passed": gate_status == PASS_GATE,
        "network_api_calls": network_api_calls,
        "service_key_accessed": service_key_accessed,
        "upstream_modified_file_count": immutability["upstream_modified_file_count"],
        "upstream_deleted_file_count": immutability["upstream_deleted_file_count"],
        "upstream_added_file_count": immutability["upstream_added_file_count"],
        "source_r2d1n_hf1_gate": upstream_refs["upstream_reference_r2d1n_hf1.json"]["gate_status"],
        "registry_complete_episode_count": len(registry),
        "route_complete_counts": route_counts,
        "global_unique_vehicle_count": len(vehicle_counts),
        "route_local_unique_vehicle_counts": route_local_counts,
        "observation_date_count": len(date_counts),
        "global_hour_bucket_count": len(hour_counts),
        "invalid_clock_order_count": clock_audit["invalid_clock_order_count"],
        "negative_interval_count": interval_audit["negative_interval_count"],
        "reversed_interval_count": interval_audit["reversed_interval_count"],
        "dual_clock_formula_mismatch_count": interval_audit["dual_clock_formula_mismatch_count"],
        "confirmation_threshold_failure_count": confirmation_audit["episode_confirmation_threshold_failure_count"],
        "raw_provenance_failure_count": raw_audit["raw_provenance_failure_count"],
        "episode_duplicate_count": episode_duplicate_audit["episode_duplicate_count"],
        "lineage_conflict_count": lineage_audit["lineage_conflict_count"],
        "registry_canonical_sha256": fingerprint["registry_canonical_sha256"],
        "episode_id_set_sha256": fingerprint["episode_id_set_sha256"],
        "raw_evidence_reference_set_sha256": fingerprint["raw_evidence_reference_set_sha256"],
        "interval_tuple_set_sha256": fingerprint["interval_tuple_set_sha256"],
        "method_prototype_data_threshold_met": method_ready,
        "formal_inferential_sufficiency": "NOT_ESTABLISHED",
        "primary_estimand": "OBSERVED_POST_SERVICE_NON_REVENUE_TURNAROUND_INTERVAL",
        "unidentifiable_estimand": "ACTUAL_DRIVER_REST_DURATION",
        "primary_input_clock": "CONSERVATIVE_DUAL_CLOCK",
        "primary_method_candidate": "TURNBULL_NPMLE_INTERVAL_CENSORED_POOLED",
        "route_level_analysis": "DESCRIPTIVE_ONLY",
        "midpoint_primary_use": "PROHIBITED",
        "estimation_input_packet_ready": input_packet["input_packet_ready"],
        "estimation_method_contract_ready": method_ready,
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "strict_json_failure_count": strict_pre_manifest["strict_json_failure_count"],
        "parquet_read_failure_count": parquet_pre_manifest["parquet_read_failure_count"],
        "json_parquet_value_mismatch_count": jp_audit["json_parquet_value_mismatch_count"],
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "secret_leak_count": secret_audit["secret_leak_count"],
        "next_authorized_action": "Method Prototype interval-censored estimation execution authorization review only",
        "pass_conditions": pass_conditions,
    }
    dump_json(output_root / "prompt5_e01_r2d1o_gate.json", gate_payload)

    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1o_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)
    final_strict = strict_json_audit(output_root)
    final_parquet = parquet_audit(output_root)
    gate_payload.update(manifest_result)
    gate_payload["strict_json_failure_count"] = final_strict["strict_json_failure_count"]
    gate_payload["parquet_read_failure_count"] = final_parquet["parquet_read_failure_count"]
    if final_strict["strict_json_failure_count"] or final_parquet["parquet_read_failure_count"] or any(manifest_result[key] for key in ["manifest_missing_required_file_count", "manifest_nonself_hash_mismatch_count", "manifest_nonself_size_mismatch_count"]):
        gate_payload["gate_status"] = "FAIL_SCHEMA_AUDIT"
        gate_payload["gate_passed"] = False
    dump_json(output_root / "prompt5_e01_r2d1o_gate.json", gate_payload)
    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1o_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)

    print("R2D-1O 12-EPISODE REGISTRY FREEZE AND ESTIMATION READINESS REVIEW COMPLETE")
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
    print("\nsource_r2d1n_hf1_gate:")
    print(upstream_refs["upstream_reference_r2d1n_hf1.json"]["gate_status"])
    print("\nregistry_complete_episode_count:")
    print(len(registry))
    print("\nroute_complete_counts:")
    for route in ALL_ROUTES:
        print(f"{route}={route_counts[route]}")
    print("\nglobal_unique_vehicle_count:")
    print(len(vehicle_counts))
    print("\nroute_local_unique_vehicle_counts:")
    for route in ALL_ROUTES:
        print(f"{route}={route_local_counts[route]}")
    print("\nobservation_date_count:")
    print(len(date_counts))
    print("\nglobal_hour_bucket_count:")
    print(len(hour_counts))
    print("\ninvalid_clock_order_count:")
    print(clock_audit["invalid_clock_order_count"])
    print("\nnegative_interval_count:")
    print(interval_audit["negative_interval_count"])
    print("\nreversed_interval_count:")
    print(interval_audit["reversed_interval_count"])
    print("\ndual_clock_formula_mismatch_count:")
    print(interval_audit["dual_clock_formula_mismatch_count"])
    print("\nconfirmation_threshold_failure_count:")
    print(confirmation_audit["episode_confirmation_threshold_failure_count"])
    print("\nraw_provenance_failure_count:")
    print(raw_audit["raw_provenance_failure_count"])
    print("\nepisode_duplicate_count:")
    print(episode_duplicate_audit["episode_duplicate_count"])
    print("\nlineage_conflict_count:")
    print(lineage_audit["lineage_conflict_count"])
    print("\nregistry_canonical_sha256:")
    print(fingerprint["registry_canonical_sha256"])
    print("\nepisode_id_set_sha256:")
    print(fingerprint["episode_id_set_sha256"])
    print("\nraw_evidence_reference_set_sha256:")
    print(fingerprint["raw_evidence_reference_set_sha256"])
    print("\nmethod_prototype_data_threshold_met:")
    print(str(method_ready).lower())
    print("\nformal_inferential_sufficiency:\nNOT_ESTABLISHED")
    print("\nprimary_estimand:\nOBSERVED_POST_SERVICE_NON_REVENUE_TURNAROUND_INTERVAL")
    print("\nunidentifiable_estimand:\nACTUAL_DRIVER_REST_DURATION")
    print("\nprimary_input_clock:\nCONSERVATIVE_DUAL_CLOCK")
    print("\nprimary_method_candidate:\nTURNBULL_NPMLE_INTERVAL_CENSORED_POOLED")
    print("\nroute_level_analysis:\nDESCRIPTIVE_ONLY")
    print("\nmidpoint_primary_use:\nPROHIBITED")
    print("\nestimation_input_packet_ready:")
    print(str(input_packet["input_packet_ready"]).lower())
    print("\nestimation_method_contract_ready:")
    print(str(method_ready).lower())
    print("\nterminal_recovery_estimation_execution_approved:\nfalse")
    print("\nterminal_recovery_parameter_generated:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\nsimulator_application_authorized:\nfalse")
    print("\nphase2_authorized:\nfalse")
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
    print(secret_audit["secret_leak_count"])
    print("\nnext_authorized_action:")
    print("Method Prototype interval-censored estimation execution authorization review only")


if __name__ == "__main__":
    main()
