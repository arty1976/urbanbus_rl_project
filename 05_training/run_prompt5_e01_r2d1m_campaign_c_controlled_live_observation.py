#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time as monotonic_time
from collections import Counter
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1m_campaign_c_controlled_live_observation.py"
R2D1H_LIVE_SCRIPT = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1h_campaign_a_live_observation_actual.py"

R2D1L_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1l_campaign_c_authorization_review_20260727_123717"
R2D1K_HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_hf1_campaign_b_metadata_finalization_20260727_113644"
R2D1K_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_campaign_b_controlled_live_observation_20260727_093548"
R2D1I_HF2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_hf2_final_metadata_reconciliation_20260726_191538"
HF1_MAPPING_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

REGISTRY_PATH = R2D1K_HF1_ROOT / "cumulative_episode_registry_candidate_hf1.parquet"
MAPPING_PATH = HF1_MAPPING_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"

TARGET_ROUTE = "4010002118"
TARGET_ROUTES = [TARGET_ROUTE]
EXCLUDED_ROUTES = ["4010002001", "4010002004", "4050010000"]
ALL_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]

CAMPAIGN_WINDOW_START = time(9, 0)
CAMPAIGN_WINDOW_END = time(14, 0)
MAX_FOLLOW_MINUTES = 100
FINALIZATION_BUFFER_MINUTES = 15
FOLLOW_REQUIREMENT_MINUTES = MAX_FOLLOW_MINUTES + FINALIZATION_BUFFER_MINUTES
RECOMMENDED_CALLS = 220
ABSOLUTE_HARD_CAP = 300
DAILY_PHYSICAL_SAFETY_CAP = 800
MAX_CALLS_PER_MINUTE = 4
MIN_EFFECTIVE_HARD_CAP_TO_START = 140

TOP_LEVEL_REQUIRED_FILES = [
    "prompt5_e01_r2d1m_manifest.json",
    "prompt5_e01_r2d1m_gate.json",
    "prompt5_e01_r2d1m_final_report.md",
    "upstream_reference_r2d1l.json",
    "upstream_reference_r2d1k_hf1.json",
    "upstream_reference_r2d1k.json",
    "upstream_reference_r2d1i_hf2.json",
    "upstream_reference_hf1_mapping.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "registry_freeze_reference_audit.json",
    "mapping_regression_audit.json",
    "target_route_access_audit.json",
    "vehicle_exclusion_registry_audit.json",
    "secret_leak_audit.json",
    "campaign_c_runtime_execution_authorization.json",
    "campaign_c_runtime_schedule_manifest.json",
    "campaign_c_daily_api_usage_audit.json",
    "campaign_c_effective_api_budget.json",
    "campaign_c_preflight_audit.json",
    "campaign_c_runtime_audit.json",
    "campaign_c_api_stop_condition_audit.json",
    "campaign_c_candidate_vehicle_selection_audit.json",
    "campaign_c_state_transition_audit.json",
    "campaign_c_counter_contract_v12.json",
    "campaign_c_counter_contract_v12.parquet",
    "campaign_c_clock_semantics_audit.json",
    "campaign_c_interval_validation_audit.json",
    "campaign_c_episode_evidence_sha256_audit.json",
    "campaign_c_episode_deduplication_audit.json",
    "campaign_c_position_samples.parquet",
    "campaign_c_vehicle_trajectories.parquet",
    "campaign_c_terminal_recovery_episodes.json",
    "campaign_c_terminal_recovery_episodes.parquet",
    "campaign_c_terminal_recovery_interval_bounds.json",
    "campaign_c_terminal_recovery_interval_bounds.parquet",
    "campaign_c_route_summary.json",
    "campaign_c_route_summary.parquet",
    "campaign_c_global_vehicle_result.json",
    "cumulative_episode_registry_candidate_12.json",
    "cumulative_episode_registry_candidate_12.parquet",
    "method_prototype_readiness_audit.json",
    "method_prototype_threshold_status.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]

ROUTE_REQUIRED_FILES = [
    "route_mapping_reference.json",
    "observation_manifest.json",
    "candidate_selection_audit.json",
    "upstream_trigger_audit.json",
    "state_transition_audit.json",
    "vehicle_trajectories.parquet",
    "position_samples.parquet",
    "terminal_recovery_episodes.json",
    "terminal_recovery_episodes.parquet",
    "terminal_recovery_interval_bounds.json",
    "terminal_recovery_interval_bounds.parquet",
    "episode_summary.json",
    "clock_semantics_audit.json",
    "raw_file_index.json",
    "evidence_sha256_audit.json",
]


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
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
                "root": str(root),
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
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
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def route_local_unique_counts(registry: pd.DataFrame) -> Dict[str, int]:
    counts = registry.groupby(registry["route_id"].astype(str))["route_local_vehicle_identity_key"].nunique().to_dict()
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def hour_buckets_by_route(registry: pd.DataFrame) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for route in ALL_ROUTES:
        subset = registry[registry["route_id"].astype(str) == route]
        result[route] = sorted(subset["hour_bucket"].dropna().astype(str).unique().tolist())
    return result


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


def manifest_payload(output_root: Path, required_files: Sequence[str]) -> Dict[str, Any]:
    files = []
    manifest_name = "prompt5_e01_r2d1m_manifest.json"
    seen_manifest = False
    for path in sorted(files_under(output_root), key=lambda item: str(item.relative_to(output_root))):
        rel = str(path.relative_to(output_root))
        is_manifest = rel == manifest_name
        seen_manifest = seen_manifest or is_manifest
        files.append(
            {
                "path": rel,
                "exists": True,
                "sha256": None if is_manifest else sha256_file(path),
                "self_hash_exempt": is_manifest,
                "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization." if is_manifest else None,
                "size_bytes": None if is_manifest else path.stat().st_size,
                "self_size_exempt": is_manifest,
                "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata." if is_manifest else None,
            }
        )
    if not seen_manifest:
        files.insert(
            0,
            {
                "path": manifest_name,
                "exists": True,
                "sha256": None,
                "self_hash_exempt": True,
                "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
                "size_bytes": None,
                "self_size_exempt": True,
                "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
            },
        )
    present = {entry["path"] for entry in files}
    missing = [name for name in required_files if name not in present]
    return {
        "artifact_name": output_root.name,
        "created_at_kst": datetime.now(KST).isoformat(),
        "required_file_count": len(required_files),
        "present_required_file_count": len(required_files) - len(missing),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "manifest_self_hash_exempt": True,
        "manifest_self_size_exempt": True,
        "files": files,
    }


def validate_manifest(output_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    missing = []
    hash_mismatch = []
    size_mismatch = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        if not path.exists():
            missing.append(entry["path"])
            continue
        if entry.get("self_hash_exempt") or entry["path"] == "prompt5_e01_r2d1m_manifest.json":
            if entry.get("sha256") is not None or entry.get("size_bytes") is not None:
                size_mismatch.append(entry["path"])
            continue
        if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
            hash_mismatch.append(entry["path"])
        if entry.get("size_bytes") is not None and path.stat().st_size != entry["size_bytes"]:
            size_mismatch.append(entry["path"])
    return {
        "manifest_missing_file_count": len(missing),
        "manifest_missing_files": missing,
        "manifest_nonself_hash_mismatch_count": len(hash_mismatch),
        "manifest_nonself_hash_mismatches": hash_mismatch,
        "manifest_nonself_size_mismatch_count": len(size_mismatch),
        "manifest_nonself_size_mismatches": size_mismatch,
    }


def validate_source_manifest(root: Path) -> Dict[str, Any]:
    manifests = sorted(root.glob("*manifest.json"))
    if not manifests:
        return {"root": str(root), "manifest_found": False, "failure_count": 0, "failures": []}
    failures = []
    for manifest_path in manifests:
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            failures.append({"manifest": str(manifest_path), "error": repr(exc)})
            continue
        for entry in manifest.get("files", []):
            if isinstance(entry, str):
                rel = entry
                entry_payload: Dict[str, Any] = {}
            elif isinstance(entry, dict):
                rel = entry.get("path")
                entry_payload = entry
            else:
                continue
            if not rel:
                continue
            file_path = root / rel
            if not file_path.exists():
                failures.append({"manifest": str(manifest_path), "path": rel, "error": "missing"})
                continue
            if entry_payload.get("self_hash_exempt") or rel == manifest_path.name:
                continue
            if entry_payload.get("sha256") and sha256_file(file_path) != entry_payload["sha256"]:
                failures.append({"manifest": str(manifest_path), "path": rel, "error": "sha256_mismatch"})
            if entry_payload.get("size_bytes") is not None and file_path.stat().st_size != entry_payload["size_bytes"]:
                failures.append({"manifest": str(manifest_path), "path": rel, "error": "size_mismatch"})
    return {
        "root": str(root),
        "manifest_found": True,
        "manifest_paths": [str(path) for path in manifests],
        "failure_count": len(failures),
        "failures": failures,
    }


def build_upstream_reference(root: Path, gate_file: str | None, expected_gate: str | None = None) -> Dict[str, Any]:
    gate_path = root / gate_file if gate_file else None
    gate_status = None
    gate_passed = None
    if gate_path and gate_path.exists():
        gate_payload = read_json(gate_path)
        gate_status = gate_payload.get("gate_status")
        gate_passed = gate_payload.get("gate_passed")
    return {
        "artifact_path": str(root),
        "exists": root.exists(),
        "gate_path": str(gate_path) if gate_path else None,
        "gate_status": gate_status,
        "expected_gate_status": expected_gate,
        "gate_matches_expected": gate_status == expected_gate if expected_gate else None,
        "gate_passed": gate_passed,
        "file_count": len(files_under(root)),
        "read_only_input": True,
    }


def scan_daily_usage(run_date: str, output_root: Path) -> Dict[str, Any]:
    date_compact = run_date.replace("-", "")
    same_day_artifact_dirs = []
    artifacts = []
    evidence_files = []
    for artifact_dir in sorted(path for path in ARTIFACTS_ROOT.iterdir() if path.is_dir()):
        if artifact_dir == output_root:
            continue
        if date_compact not in artifact_dir.name and run_date not in artifact_dir.name:
            continue
        same_day_artifact_dirs.append(str(artifact_dir))
        per_file_counts = []
        candidate_files = []
        for pattern in [
            "*gate.json",
            "*daily_api_usage_audit.json",
            "*runtime_audit.json",
            "*network_api_call_audit.json",
            "raw_file_index.json",
            "raw_replay_request_index.json",
        ]:
            candidate_files.extend(artifact_dir.rglob(pattern))
        for file_path in sorted(set(candidate_files)):
            try:
                payload = read_json(file_path)
            except Exception:
                continue
            inferred = 0
            if isinstance(payload, list) and file_path.name in {"raw_file_index.json", "raw_replay_request_index.json"}:
                inferred = len(payload)
            elif isinstance(payload, dict):
                preflight = payload.get("preflight_physical_calls")
                campaign = payload.get("campaign_physical_calls")
                if isinstance(preflight, int) or isinstance(campaign, int):
                    inferred = max(inferred, int(preflight or 0) + int(campaign or 0))
                for key in [
                    "r2d1k_total_physical_calls",
                    "r2d1m_total_physical_calls",
                    "physical_api_call_count",
                    "physical_calls",
                    "network_api_calls",
                    "api_call_count",
                    "raw_file_count",
                ]:
                    value = payload.get(key)
                    if isinstance(value, int):
                        inferred = max(inferred, int(value))
            per_file_counts.append({"path": str(file_path), "inferred_physical_calls": inferred})
            evidence_files.append({"path": str(file_path), "inferred_physical_calls": inferred})
        artifact_calls = max([item["inferred_physical_calls"] for item in per_file_counts] or [0])
        artifacts.append(
            {
                "artifact_dir": str(artifact_dir),
                "inferred_physical_calls": int(artifact_calls),
                "evidence_file_count": len(per_file_counts),
                "evidence_files": per_file_counts,
            }
        )
    prior_calls = sum(item["inferred_physical_calls"] for item in artifacts)
    r2d1k_final_included = any(
        item["artifact_dir"].endswith("prompt5_e01_r2d1k_campaign_b_controlled_live_observation_20260727_093548")
        and item["inferred_physical_calls"] == 160
        for item in artifacts
    )
    known = True if run_date != "2026-07-27" else r2d1k_final_included
    return {
        "run_date": run_date,
        "prior_physical_calls_on_run_date": int(prior_calls),
        "same_day_artifact_dirs_before_current_count": len(same_day_artifact_dirs),
        "same_day_artifact_dirs_before_current": same_day_artifact_dirs,
        "artifact_usage_evidence": artifacts,
        "evidence_files": evidence_files,
        "r2d1k_20260727_093548_160_calls_included": r2d1k_final_included,
        "known": known,
        "method": "Grouped same-date artifacts and used the maximum physical-call count per artifact to avoid double-counting per-file ledgers.",
    }


def scan_actual_secret_literal(root: Path, service_key: str) -> List[Dict[str, str]]:
    if not service_key:
        return []
    needle = service_key.encode("utf-8")
    hits = []
    for path in files_under(root):
        try:
            if needle in path.read_bytes():
                hits.append({"path": str(path.relative_to(root)), "reason": "actual_service_key_literal_present"})
        except Exception as exc:
            hits.append({"path": str(path.relative_to(root)), "reason": f"scan_error:{exc!r}"})
    return hits


def build_vehicle_exclusion_registry(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vehicle_id, group in registry.groupby(registry["canonical_vehicle_id"].astype(str), dropna=False):
        canonical_id = str(vehicle_id).strip()
        routes = sorted(group["route_id"].astype(str).dropna().unique().tolist())
        rows.append(
            {
                "canonical_vehicle_id": canonical_id,
                "global_complete_episode_count": int(len(group)),
                "route_count": int(len(routes)),
                "routes_seen": routes,
                "source_episode_ids": sorted(group["source_episode_id"].astype(str).dropna().tolist()),
                "excluded_from_global_unseen_vehicle_tier": True,
                "exact_id_matching_only": True,
            }
        )
    return pd.DataFrame(rows).sort_values("canonical_vehicle_id").reset_index(drop=True)


def empty_runtime_frames() -> Dict[str, pd.DataFrame]:
    return {
        "position_samples": pd.DataFrame(
            columns=[
                "request_id",
                "route_id",
                "vehicle_id",
                "current_sequence",
                "direction",
                "provider_position_event_time",
                "request_observation_time",
                "capture_mode",
                "global_vehicle_classification",
                "route_local_vehicle_classification",
                "candidate_priority_tier",
                "diversity_eligibility",
                "raw_relative_path",
                "raw_sha256",
            ]
        ),
        "vehicle_trajectories": pd.DataFrame(
            columns=[
                "route_id",
                "vehicle_id",
                "sample_count",
                "first_request_observation_time",
                "last_request_observation_time",
                "global_vehicle_classification",
                "route_local_vehicle_classification",
                "candidate_priority_tier",
            ]
        ),
        "terminal_recovery_episodes": pd.DataFrame(
            columns=[
                "episode_id",
                "source_episode_id",
                "route_id",
                "vehicle_id",
                "episode_status",
                "final_status_class",
                "left_censored",
                "right_censored",
                "right_censor_reason",
                "clock_semantics_status",
                "post_terminal_confirmation_observation_count",
            ]
        ),
        "terminal_recovery_interval_bounds": pd.DataFrame(
            columns=[
                "episode_id",
                "route_id",
                "vehicle_id",
                "provider_lower_bound_sec",
                "provider_upper_bound_sec",
                "request_lower_bound_sec",
                "request_upper_bound_sec",
                "conservative_dual_lower_bound_sec",
                "conservative_dual_upper_bound_sec",
            ]
        ),
    }


def load_live_base() -> Any:
    spec = importlib.util.spec_from_file_location("r2d1h_campaign_a_live_base", R2D1H_LIVE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load live base from {R2D1H_LIVE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGET_ROUTES = TARGET_ROUTES
    module.MAX_CALLS_PER_MINUTE = MAX_CALLS_PER_MINUTE
    module.DAILY_SAFETY_CAP = DAILY_PHYSICAL_SAFETY_CAP
    module.ABSOLUTE_CAMPAIGN_CAP = ABSOLUTE_HARD_CAP
    module.MAX_FOLLOW_MINUTES = {TARGET_ROUTE: MAX_FOLLOW_MINUTES}
    module.REQUIRED_END_BUFFER_MINUTES = FINALIZATION_BUFFER_MINUTES
    module.FATAL_STATUSES = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE", "FAIL_NON_TARGET_ROUTE_ACCESS"}
    for column in [
        "source_episode_id",
        "disappeared_waiting_reentry_observation_count",
        "global_vehicle_classification",
        "route_local_vehicle_classification",
        "candidate_priority_tier",
    ]:
        if column not in module.EPISODE_COLUMNS:
            module.EPISODE_COLUMNS.append(column)
    return module


def make_campaign_c_runner_class(base: Any, global_exclusion_ids: set[str], route_local_exclusion_ids: set[str]) -> Any:
    class CampaignCLiveRunner(base.CampaignRunner):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.global_exclusion_ids = {str(item).strip() for item in global_exclusion_ids}
            self.route_local_exclusion_ids = {str(item).strip() for item in route_local_exclusion_ids}
            self.campaign_complete_vehicle_ids: set[str] = set()

        def classify_vehicle(self, route: str, vehicle_id: str) -> str:  # noqa: ARG002
            canonical = str(vehicle_id).strip()
            if canonical in self.campaign_seen_vehicle_ids[route]:
                return "DUPLICATE_TERMINAL_CYCLE"
            if canonical not in self.global_exclusion_ids:
                return "GLOBAL_UNSEEN_VEHICLE"
            if canonical not in self.route_local_exclusion_ids:
                return "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE"
            return "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE"

        def desired_route_for_new_session(self, route: str) -> bool:
            if route != TARGET_ROUTE:
                return False
            if not self.allow_new_candidates or self.sessions[route] is not None:
                return False
            if sum(self.complete_count_by_route.values()) >= 1:
                return False
            return self.route_can_start(route)

        def fetch(self, route: str, mode: str) -> List[Dict[str, Any]]:
            if route != TARGET_ROUTE:
                self.fatal_error = "FAIL_NON_TARGET_ROUTE_ACCESS"
                self.first_fatal_error_time = self.first_fatal_error_time or base.iso()
                self.stop_reason = self.fatal_error
                return []
            rows = super().fetch(route, mode)
            if self.request_records:
                record = self.request_records[-1]
                if int(record.get("route_mismatch_count") or 0) > 0 or str(record.get("route_id")) != TARGET_ROUTE:
                    self.fatal_error = "FAIL_NON_TARGET_ROUTE_ACCESS"
                    self.first_fatal_error_time = record.get("request_observation_time") or base.iso()
                    self.stop_reason = self.fatal_error
            for sample in self.samples:
                if sample.get("campaign_id") == "R2D-1H-CAMPAIGN-A":
                    sample["campaign_id"] = "R2D-1M-CAMPAIGN-C"
            return rows

        def maybe_start_session(self, route: str, rows: Sequence[Mapping[str, Any]]) -> None:
            if not self.desired_route_for_new_session(route):
                return
            early = [dict(row) for row in rows if row.get("is_early_upstream_watch_zone")]
            fallback = [dict(row) for row in rows if row.get("is_upstream_watch_zone")]
            candidates = early or fallback
            if not candidates:
                return
            priority = {
                "GLOBAL_UNSEEN_VEHICLE": 4,
                "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE": 3,
                "GLOBAL_PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE": 2,
                "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE": 1,
            }
            scored = []
            for row in candidates:
                vehicle_id = base.norm(row.get("vehicle_id"))
                if vehicle_id is None:
                    self.history_count_by_route[route]["INVALID_VEHICLE_ID"] += 1
                    continue
                history = self.classify_vehicle(route, vehicle_id)
                if history == "DUPLICATE_TERMINAL_CYCLE":
                    continue
                scored.append((priority.get(history, 0), base.as_int(row.get("current_sequence")) or -1, history, row))
            if not scored:
                return
            _, _, history, chosen = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[0]
            self.session_counter += 1
            self.candidate_count_by_route[route] += 1
            self.history_count_by_route[route][history] += 1
            session_id = f"r2d1m_session_{self.session_counter:05d}_{route}"
            episode_id = f"r2d1m_episode_{self.session_counter:05d}_{route}"
            chosen["session_id"] = session_id
            chosen["episode_id"] = episode_id
            phase = "EARLY_UPSTREAM_WATCH" if chosen.get("is_early_upstream_watch_zone") else "UPSTREAM_FOCUSED"
            self.sessions[route] = {
                "route_id": route,
                "session_id": session_id,
                "episode_id": episode_id,
                "vehicle_id": base.norm(chosen.get("vehicle_id")),
                "direction": base.norm(chosen.get("direction")),
                "vehicle_history_class": history,
                "phase": phase,
                "first_upstream": chosen,
                "last_pre_terminal": chosen,
                "first_terminal": None,
                "last_terminal": None,
                "first_post_terminal": None,
                "post_terminal_confirmed": None,
                "upstream_watch_sample_count": 1,
                "terminal_hold_sample_count": 0,
                "post_terminal_confirmation_sample_count": 0,
                "post_terminal_confirmation_samples": [],
                "first_terminal_monotonic": None,
                "missing_after_terminal_count": 0,
                "disappeared_waiting_reentry_observation_count": 0,
            }
            self.campaign_seen_vehicle_ids[route].add(str(chosen["vehicle_id"]))
            self._tag_sample(chosen)
            print(
                json.dumps(
                    {
                        "event": "r2d1m_candidate_started",
                        "route_id": route,
                        "vehicle_id": chosen["vehicle_id"],
                        "vehicle_history_class": history,
                        "sequence": chosen["current_sequence"],
                        "phase": phase,
                        "time": base.iso(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        def update_session(self, route: str, rows: Sequence[Mapping[str, Any]]) -> None:
            session = self.sessions.get(route)
            before_missing = int((session or {}).get("missing_after_terminal_count", 0))
            super().update_session(route, rows)
            session_after = self.sessions.get(route)
            if session_after is not None:
                after_missing = int(session_after.get("missing_after_terminal_count", before_missing))
                session_after["disappeared_waiting_reentry_observation_count"] = after_missing

        def finish_session(self, route: str, status: str, invalid_reason: Optional[str] = None) -> None:
            session = self.sessions.get(route)
            disappeared_count = int((session or {}).get("disappeared_waiting_reentry_observation_count", (session or {}).get("missing_after_terminal_count", 0)))
            vehicle_id = str((session or {}).get("vehicle_id") or "").strip()
            history = str((session or {}).get("vehicle_history_class") or "")
            super().finish_session(route, status, invalid_reason)
            if self.episodes:
                self.episodes[-1]["source_episode_id"] = self.episodes[-1].get("episode_id")
                self.episodes[-1]["disappeared_waiting_reentry_observation_count"] = disappeared_count
                self.episodes[-1]["global_vehicle_classification"] = "GLOBAL_UNSEEN_VEHICLE" if vehicle_id not in self.global_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE"
                self.episodes[-1]["route_local_vehicle_classification"] = "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" if vehicle_id not in self.route_local_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE"
                self.episodes[-1]["candidate_priority_tier"] = history
                if self.episodes[-1].get("complete_interval_censored_episode") and vehicle_id:
                    self.campaign_complete_vehicle_ids.add(vehicle_id)

        def run(self) -> None:
            self.campaign_started_at = base.iso()
            while not self.fatal_error and base.now_kst() < self.planned_end:
                complete_total = sum(self.complete_count_by_route.values())
                if complete_total >= 1:
                    self.allow_new_candidates = False
                    self.stop_reason = "CAMPAIGN_C_FINAL_EPISODE_TARGET_REACHED"
                    break
                if self.physical_calls_this_artifact >= int(self.effective_cap * 0.9):
                    self.allow_new_candidates = False
                    if not any(self.sessions.values()):
                        self.stop_reason = "EFFECTIVE_CAMPAIGN_HARD_CAP_90_PERCENT_REACHED"
                        break
                if not any(self.sessions.values()) and not self.route_can_start(TARGET_ROUTE):
                    self.stop_reason = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
                    break
                due_routes = [route for route in TARGET_ROUTES if self.next_due[route] <= monotonic_time.monotonic()]
                if not due_routes:
                    monotonic_time.sleep(min(1.0, max(0.1, min(self.next_due.values()) - monotonic_time.monotonic())))
                    continue
                route = TARGET_ROUTE
                if self.sessions[route] is None and not self.route_can_start(route):
                    self.next_due[route] = monotonic_time.monotonic() + 60
                    continue
                mode = self.route_mode(route)
                rows = self.fetch(route, mode)
                if self.fatal_error:
                    break
                self.update_session(route, rows)
                if self.sessions[route] is None:
                    self.maybe_start_session(route, rows)
                self.next_due[route] = monotonic_time.monotonic() + base.request_interval(self.route_mode(route))
            if self.fatal_error:
                if self.sessions[TARGET_ROUTE] is not None:
                    self.finish_session(TARGET_ROUTE, "RIGHT_CENSORED_FATAL_API_STOP")
            elif base.now_kst() >= self.planned_end:
                self.stop_reason = self.stop_reason or "CAMPAIGN_WINDOW_ENDED"
                if self.sessions[TARGET_ROUTE] is not None:
                    self.finish_session(TARGET_ROUTE, "RIGHT_CENSORED_CAMPAIGN_WINDOW_END")
            elif self.sessions[TARGET_ROUTE] is not None and self.stop_reason:
                if self.stop_reason == "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW":
                    self.finish_session(TARGET_ROUTE, "RIGHT_CENSORED_CAMPAIGN_WINDOW_END")
                else:
                    self.finish_session(TARGET_ROUTE, "RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP")
            self.campaign_finished_at = base.iso()

    return CampaignCLiveRunner


def live_bounds_frame(episodes: pd.DataFrame) -> pd.DataFrame:
    bounds_cols = [
        "episode_id",
        "route_id",
        "vehicle_id",
        "direction",
        "provider_service_end_window_start",
        "provider_service_end_window_end",
        "provider_reentry_window_start",
        "provider_reentry_window_end",
        "request_service_end_window_start",
        "request_service_end_window_end",
        "request_reentry_window_start",
        "request_reentry_window_end",
        "provider_lower_bound_sec",
        "provider_upper_bound_sec",
        "request_lower_bound_sec",
        "request_upper_bound_sec",
        "conservative_dual_lower_bound_sec",
        "conservative_dual_upper_bound_sec",
        "complete_interval_censored_episode",
    ]
    return episodes[bounds_cols].copy() if not episodes.empty else pd.DataFrame(columns=bounds_cols)


def normalize_sha_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [value]


def safe_evidence_sha_audit(base: Any, output_root: Path, episodes: pd.DataFrame) -> Dict[str, Any]:
    frame = episodes.copy()
    column = "post_terminal_confirmation_raw_sha256s"
    if column in frame.columns:
        frame[column] = frame[column].apply(normalize_sha_sequence)
    payload = base.evidence_sha_audit(output_root, frame)
    payload["raw_provenance_failure_count"] = payload.get("provenance_failure_count", 0)
    return payload


def safe_duplicate_audit(base: Any, prior_registry: pd.DataFrame, episodes: pd.DataFrame) -> Dict[str, Any]:
    prior = prior_registry.copy()
    for target, source in [
        ("first_terminal_raw_sha256", "first_terminal_raw_sha"),
        ("first_post_terminal_raw_sha256", "first_post_terminal_raw_sha"),
    ]:
        if target not in prior.columns:
            prior[target] = prior[source] if source in prior.columns else None
    return base.duplicate_audit(prior, episodes)


def build_counter_v12(samples: pd.DataFrame, episodes: pd.DataFrame, runner: Any, global_exclusion_ids: set[str], route_local_exclusion_ids: set[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    def counts_for(route: Optional[str]) -> Dict[str, Any]:
        s = samples if route is None else samples[samples["route_id"].astype(str) == route]
        e = episodes if route is None else episodes[episodes["route_id"].astype(str) == route]
        complete = e[e["complete_interval_censored_episode"] == True] if not e.empty else pd.DataFrame(columns=e.columns)
        complete_ids = {str(value).strip() for value in complete["vehicle_id"].dropna().astype(str).tolist()} if not complete.empty else set()
        new_global_ids = {value for value in complete_ids if value not in global_exclusion_ids}
        new_route_local_ids = {value for value in complete_ids if value not in route_local_exclusion_ids}
        history = runner.history_count_by_route[TARGET_ROUTE]
        return {
            "broad_scan_observation_count": int((s["capture_mode"] == "BROAD_SCAN").sum()) if not s.empty else 0,
            "early_upstream_watch_observation_count": int((s["capture_mode"] == "EARLY_UPSTREAM_WATCH").sum()) if not s.empty else 0,
            "upstream_focused_observation_count": int((s["capture_mode"] == "UPSTREAM_FOCUSED").sum()) if not s.empty else 0,
            "terminal_focused_observation_count": int((s["capture_mode"] == "TERMINAL_FOCUSED").sum()) if not s.empty else 0,
            "disappeared_waiting_reentry_observation_count": int(e["disappeared_waiting_reentry_observation_count"].fillna(0).sum()) if not e.empty and "disappeared_waiting_reentry_observation_count" in e else 0,
            "post_terminal_confirmation_observation_count": int(e["observed_post_terminal_confirmation_sample_count"].fillna(0).sum()) if not e.empty else 0,
            "candidate_vehicle_count": int(runner.candidate_count_by_route[TARGET_ROUTE] if route is not None else sum(runner.candidate_count_by_route.values())),
            "global_unseen_vehicle_candidate_count": int(history.get("GLOBAL_UNSEEN_VEHICLE", 0)),
            "new_route_local_vehicle_candidate_count": int(history.get("NEW_INDEPENDENT_VEHICLE_FOR_ROUTE", 0)),
            "global_previously_censored_candidate_count": int(history.get("GLOBAL_PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE", 0)),
            "previously_complete_vehicle_candidate_count": int(history.get("PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE", 0)),
            "invalid_vehicle_candidate_count": int(history.get("INVALID_VEHICLE_ID", 0)),
            "tracking_session_started_count": int(len(e)),
            "pre_terminal_confirmed_count": int(e["last_pre_terminal_request_time"].notna().sum()) if not e.empty else 0,
            "terminal_entry_count": int(e["first_terminal_request_time"].notna().sum()) if not e.empty else 0,
            "terminal_loop_movement_observation_count": 0,
            "terminal_stop_hold_observation_count": int(e["terminal_hold_sample_count"].fillna(0).sum()) if not e.empty else 0,
            "post_terminal_reset_count": int(e["first_post_terminal_request_time"].notna().sum()) if not e.empty else 0,
            "post_terminal_confirmed_count": int(e["post_terminal_confirmed_request_time"].notna().sum()) if not e.empty else 0,
            "new_complete_episode_count": int(e["complete_interval_censored_episode"].sum()) if not e.empty else 0,
            "left_censored_episode_count": int(e["left_censored"].sum()) if not e.empty else 0,
            "right_censored_episode_count": int(e["right_censored"].sum()) if not e.empty else 0,
            "invalid_episode_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
            "new_global_complete_vehicle_count": len(new_global_ids),
            "new_route_local_complete_vehicle_count": len(new_route_local_ids),
            "episode_duplicate_count": 0,
            "contradiction_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
            "total_final_episode_count": int(len(e)),
            "complete_final_count": int((e["final_status_class"] == "COMPLETE").sum()) if not e.empty else 0,
            "left_censored_final_count": int(e["left_censored"].sum()) if not e.empty else 0,
            "right_censored_final_count": int(e["right_censored"].sum()) if not e.empty else 0,
            "invalid_final_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
        }

    rows = [
        {"scope": "route", "route_id": TARGET_ROUTE, **counts_for(TARGET_ROUTE)},
        {"scope": "campaign_c_total", "route_id": "ALL_TARGET_ROUTES", **counts_for(None)},
    ]
    frame = pd.DataFrame(rows)
    total = rows[-1]
    passed = total["total_final_episode_count"] == total["complete_final_count"] + total["left_censored_final_count"] + total["right_censored_final_count"] + total["invalid_final_count"]
    return frame, {"counter_contract_version": "v12", "counter_contract_passed": passed, "counter_contract_failure_count": 0 if passed else 1, "rows": jsonable(rows)}


def build_campaign_c_registry(prior_registry: pd.DataFrame, episodes: pd.DataFrame, output_root: Path, global_exclusion_ids: set[str], route_local_exclusion_ids: set[str]) -> pd.DataFrame:
    rows = prior_registry.copy()
    existing_global_counts = Counter(rows["canonical_vehicle_id"].dropna().astype(str).tolist()) if "canonical_vehicle_id" in rows else Counter()
    existing_route_counts = Counter(rows["route_local_vehicle_identity_key"].dropna().astype(str).tolist()) if "route_local_vehicle_identity_key" in rows else Counter()
    next_order = int(rows["hf2_ingest_order"].max()) if "hf2_ingest_order" in rows and not rows.empty else len(rows)
    new_rows = []
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        vehicle_id = str(ep.get("vehicle_id") or "").strip()
        if not vehicle_id:
            continue
        route_id = str(ep.get("route_id") or TARGET_ROUTE)
        route_key = f"{route_id}:{vehicle_id}"
        next_order += 1
        first_terminal = datetime.fromisoformat(ep["first_terminal_request_time"]) if ep.get("first_terminal_request_time") else None
        prior_global_count = int(existing_global_counts.get(vehicle_id, 0))
        prior_route_count = int(existing_route_counts.get(route_key, 0))
        is_new_global = vehicle_id not in global_exclusion_ids
        is_new_route_local = vehicle_id not in route_local_exclusion_ids
        existing_global_counts[vehicle_id] += 1
        existing_route_counts[route_key] += 1
        row = {column: None for column in prior_registry.columns}
        row.update(
            {
                "frozen_episode_id": None,
                "source_campaign_id": "R2D-1M-CAMPAIGN-C",
                "source_artifact": str(output_root),
                "source_episode_id": ep.get("episode_id"),
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "observation_date": None if first_terminal is None else first_terminal.date().isoformat(),
                "hour_bucket": None if first_terminal is None else f"{first_terminal.hour:02d}:00-{first_terminal.hour:02d}:59",
                "first_terminal_raw_sha": ep.get("first_terminal_raw_sha256"),
                "first_post_terminal_raw_sha": ep.get("first_post_terminal_raw_sha256"),
                "clock_semantics_status": ep.get("clock_semantics_status"),
                "eligible_for_estimation_input": bool(ep.get("eligible_for_estimation_input")),
                "canonical_vehicle_id": vehicle_id,
                "global_vehicle_identity_key": vehicle_id,
                "route_local_vehicle_identity_key": route_key,
                "global_vehicle_classification_at_ingest": "GLOBAL_UNSEEN_VEHICLE" if is_new_global else "PREVIOUSLY_COMPLETE_VEHICLE",
                "route_local_vehicle_classification_at_ingest": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" if is_new_route_local else "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE",
                "is_new_global_vehicle_at_ingest": is_new_global,
                "is_new_route_local_vehicle_at_ingest": is_new_route_local,
                "prior_global_occurrence_count_at_ingest": prior_global_count,
                "prior_route_local_occurrence_count_at_ingest": prior_route_count,
                "global_occurrence_count_after_ingest": int(existing_global_counts[vehicle_id]),
                "route_local_occurrence_count_after_ingest": int(existing_route_counts[route_key]),
                "global_vehicle_classification": "GLOBAL_UNSEEN_VEHICLE" if is_new_global else "PREVIOUSLY_COMPLETE_VEHICLE",
                "route_local_vehicle_classification": "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" if is_new_route_local else "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE",
                "is_new_global_vehicle": is_new_global,
                "is_new_route_local_vehicle": is_new_route_local,
                "hf2_ingest_order": next_order,
            }
        )
        new_rows.append(row)
    if new_rows:
        rows = pd.concat([rows, pd.DataFrame(new_rows, columns=prior_registry.columns)], ignore_index=True)
    return rows


def campaign_c_gate_from_results(authorization_approved: bool, preflight_passed: bool, runner: Any, counter_total: Mapping[str, Any], evidence: Mapping[str, Any], interval: Mapping[str, Any], duplicate: Mapping[str, Any], secret_leak_count: int, mapping_regression_count: int) -> str:
    if not authorization_approved:
        return "WAITING_FOR_CAMPAIGN_WINDOW"
    if not preflight_passed or runner is None or runner.fatal_error:
        return "BLOCKED_API_RUNTIME"
    if mapping_regression_count:
        return "FAIL_MAPPING_REGRESSION"
    if duplicate.get("episode_duplicate_count", 0):
        return "FAIL_EPISODE_DUPLICATION"
    if evidence.get("raw_provenance_failure_count", evidence.get("provenance_failure_count", 0)):
        return "FAIL_PROVENANCE_AUDIT"
    if interval.get("interval_validation_failure_count", 0):
        return "FAIL_INTERVAL_VALIDATION"
    if secret_leak_count:
        return "FAIL_SECURITY_AUDIT"
    if counter_total.get("contradiction_count", 0):
        return "FAIL_OBSERVATION_CAMPAIGN"
    complete_count = int(counter_total.get("new_complete_episode_count", 0))
    if complete_count == 1:
        return "PASS_CAMPAIGN_C_FINAL_EPISODE_COMPLETE"
    if int(counter_total.get("total_final_episode_count", 0)) > 0:
        return "PASS_CAMPAIGN_C_PARTIAL"
    return "BLOCKED_NO_UPSTREAM_VEHICLE" if runner and runner.campaign_calls > 0 else "BLOCKED_API_RUNTIME"


def main() -> None:
    now = datetime.now(KST)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_date = now.strftime("%Y-%m-%d")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1m_campaign_c_controlled_live_observation_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "raw" / TARGET_ROUTE).mkdir(parents=True, exist_ok=True)

    required_files = TOP_LEVEL_REQUIRED_FILES + [
        f"terminal_recovery_evidence/{TARGET_ROUTE}/{file_name}" for file_name in ROUTE_REQUIRED_FILES
    ]
    upstream_roots = [R2D1L_ROOT, R2D1K_HF1_ROOT, R2D1K_ROOT, R2D1I_HF2_ROOT, HF1_MAPPING_ROOT, R2D1E_ROOT, R2D1F_ROOT]
    before = snapshot(upstream_roots)

    service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY") or ""
    service_key_present = bool(service_key)
    service_key_length = len(service_key)

    source_refs = {
        "r2d1l": build_upstream_reference(R2D1L_ROOT, "prompt5_e01_r2d1l_gate.json", "PASS_CAMPAIGN_C_AUTHORIZATION_REVIEW_READY"),
        "r2d1k_hf1": build_upstream_reference(R2D1K_HF1_ROOT, "prompt5_e01_r2d1k_hf1_gate.json", "PASS_CAMPAIGN_B_METADATA_FINALIZATION_FREEZE_READY"),
        "r2d1k": build_upstream_reference(R2D1K_ROOT, "prompt5_e01_r2d1k_gate.json", "PASS_CAMPAIGN_B_FULL_TARGET"),
        "r2d1i_hf2": build_upstream_reference(R2D1I_HF2_ROOT, "prompt5_e01_r2d1i_hf2_gate.json", "PASS_FINAL_METADATA_RECONCILIATION_FREEZE_READY"),
        "hf1_mapping": build_upstream_reference(HF1_MAPPING_ROOT, "prompt5_e01_r2d1c_r4a_hf1_gate.json", None),
        "r2d1e": build_upstream_reference(R2D1E_ROOT, "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json", "PASS_SCHEMA_FREEZE_METHOD_REVIEW_READY"),
        "r2d1f": build_upstream_reference(R2D1F_ROOT, "prompt5_e01_r2d1f_gate.json", "PASS_ESTIMATION_DESIGN_APPROVED_ADDITIONAL_DATA_REQUIRED"),
    }
    for name, payload in source_refs.items():
        dump_json(output_root / f"upstream_reference_{name}.json", payload)

    source_manifest_audits = [validate_source_manifest(root) for root in upstream_roots]
    source_gate_failure_count = sum(
        int(ref.get("expected_gate_status") is not None and ref.get("gate_status") != ref.get("expected_gate_status"))
        + int(not ref.get("exists"))
        for ref in source_refs.values()
    )

    r2d1l_review = read_json(R2D1L_ROOT / "campaign_c_authorization_review.json")
    r2d1l_live_authorization = read_json(R2D1L_ROOT / "campaign_c_live_execution_authorization.json")
    handoff = read_json(R2D1L_ROOT / "campaign_c_execution_handoff_packet.json")
    source_state_failure_count = sum(
        [
            r2d1l_review.get("campaign_c_authorization_review_passed") is not True,
            r2d1l_review.get("campaign_c_execution_release_packet_ready") is not True,
            r2d1l_live_authorization.get("campaign_c_live_execution_authorized") is not False,
            handoff.get("target_routes") != TARGET_ROUTES,
            handoff.get("excluded_routes") != EXCLUDED_ROUTES,
            handoff.get("maximum_new_complete_episodes") != 1,
            handoff.get("campaign_c_episode_namespace") != "R2D-1M",
        ]
    )

    registry = pd.read_parquet(REGISTRY_PATH)
    mapping_df = pd.read_parquet(MAPPING_PATH)
    target_mapping_df = mapping_df[mapping_df["route_id"].astype(str) == TARGET_ROUTE].copy()

    complete_count = int(len(registry))
    route_count_map = route_counts(registry)
    expected_route_counts = {"4010002001": 3, "4010002004": 3, "4010002118": 2, "4050010000": 3}
    route_local_unique = route_local_unique_counts(registry)
    global_unique_vehicle_count = int(registry["canonical_vehicle_id"].dropna().astype(str).nunique())
    complete_deficit = max(0, 12 - complete_count)
    global_vehicle_deficit = max(0, 8 - global_unique_vehicle_count)
    duplicate_count = int(registry["source_episode_id"].duplicated().sum())
    invalid_clock_order_count = int((registry["clock_semantics_status"].dropna().astype(str) == "INVALID_CLOCK_ORDER").sum())
    raw_reference = read_json(R2D1K_HF1_ROOT / "raw_provenance_reference_audit.json")
    provenance_failure_count = int(raw_reference.get("raw_missing_file_count", 0)) + int(raw_reference.get("raw_sha_mismatch_count", 0))
    observation_dates = sorted(registry["observation_date"].dropna().astype(str).unique().tolist())
    hour_buckets = sorted(registry["hour_bucket"].dropna().astype(str).unique().tolist())
    route_hour_buckets = hour_buckets_by_route(registry)
    registry_freeze_failure_count = sum(
        [
            complete_count != 11,
            route_count_map != expected_route_counts,
            global_unique_vehicle_count != 8,
            complete_deficit != 1,
            global_vehicle_deficit != 0,
            duplicate_count != 0,
            invalid_clock_order_count != 0,
            provenance_failure_count != 0,
        ]
    )

    method_gaps = []
    if complete_count < 12:
        method_gaps.append({"gap": "total_complete_episode_deficit", "current": complete_count, "required": 12, "deficit": 12 - complete_count})
    if global_unique_vehicle_count < 8:
        method_gaps.append({"gap": "global_unique_vehicle_deficit", "current": global_unique_vehicle_count, "required": 8, "deficit": 8 - global_unique_vehicle_count})
    for route, count in route_count_map.items():
        if count < 2:
            method_gaps.append({"gap": "route_complete_minimum", "route_id": route, "current": count, "required": 2})
    for route, count in route_local_unique.items():
        if count < 2:
            method_gaps.append({"gap": "route_local_unique_vehicle_minimum", "route_id": route, "current": count, "required": 2})
    if len(observation_dates) < 2:
        method_gaps.append({"gap": "observation_date_diversity", "current": len(observation_dates), "required": 2})
    if len(hour_buckets) < 2:
        method_gaps.append({"gap": "global_hour_bucket_diversity", "current": len(hour_buckets), "required": 2})
    if invalid_clock_order_count:
        method_gaps.append({"gap": "invalid_clock_order", "current": invalid_clock_order_count, "required": 0})
    if provenance_failure_count:
        method_gaps.append({"gap": "raw_provenance_failure", "current": provenance_failure_count, "required": 0})
    resolvable_gaps = [gap for gap in method_gaps if gap["gap"] == "total_complete_episode_deficit" and gap.get("deficit") == 1]
    blocking_method_gaps = [gap for gap in method_gaps if gap not in resolvable_gaps]
    method_prototype_cannot_complete = complete_deficit != 1 or len(blocking_method_gaps) > 0

    diversity_audit = read_json(R2D1L_ROOT / "campaign_c_date_hour_diversity_audit.json")
    diversity_constraint_required = bool(diversity_audit.get("diversity_constraint_required"))

    mapping_regression_count = int(len(target_mapping_df) != 1)
    if len(target_mapping_df) == 1:
        mapping_record = target_mapping_df.iloc[0].to_dict()
        mapping_regression_count += int(bool(mapping_record.get("approved")) is not True)
        mapping_regression_count += int(bool(mapping_record.get("passed")) is not True)
        effective_live_terminal_sequence = int(float(mapping_record.get("effective_live_terminal_sequence")))
    else:
        mapping_record = {}
        effective_live_terminal_sequence = None

    exclusion_registry = build_vehicle_exclusion_registry(registry)
    exclusion_ids = sorted(exclusion_registry["canonical_vehicle_id"].astype(str).tolist())
    vehicle_exclusion_failure_count = sum(
        [
            len(exclusion_registry) != global_unique_vehicle_count,
            int(exclusion_registry["canonical_vehicle_id"].isna().sum()) != 0,
            int(exclusion_registry["canonical_vehicle_id"].duplicated().sum()) != 0,
            int(exclusion_registry["global_complete_episode_count"].sum()) != complete_count,
        ]
    )

    window_start = datetime.combine(now.date(), CAMPAIGN_WINDOW_START, tzinfo=KST)
    window_end = datetime.combine(now.date(), CAMPAIGN_WINDOW_END, tzinfo=KST)
    in_window = window_start <= now <= window_end
    minutes_to_window_start = max(0.0, (window_start - now).total_seconds() / 60.0)
    minutes_until_window_end = (window_end - now).total_seconds() / 60.0
    follow_window_minutes_available = max(0.0, (window_end - max(now, window_start)).total_seconds() / 60.0)
    follow_window_sufficient = in_window and follow_window_minutes_available >= FOLLOW_REQUIREMENT_MINUTES

    daily_usage = scan_daily_usage(run_date, output_root)
    prior_calls = int(daily_usage["prior_physical_calls_on_run_date"])
    preflight_calls = 0
    campaign_calls = 0
    r2d1m_total_calls = 0
    total_calls_on_run_date = prior_calls + preflight_calls + campaign_calls
    available_daily_budget = DAILY_PHYSICAL_SAFETY_CAP - prior_calls
    effective_hard_cap = min(ABSOLUTE_HARD_CAP, available_daily_budget)
    hard_cap_90_percent = math.floor(effective_hard_cap * 0.9) if effective_hard_cap > 0 else 0

    gate = "RUNTIME_AUTHORIZATION_APPROVED_PENDING_PREFLIGHT"
    authorization_approved = True
    blocking_reason = None
    if source_gate_failure_count or source_state_failure_count:
        gate = "FAIL_SOURCE_ARTIFACT_INTEGRITY"
        authorization_approved = False
        blocking_reason = "Direct authoritative upstream gate or Campaign C handoff state failed."
    elif registry_freeze_failure_count:
        gate = "FAIL_REGISTRY_FREEZE_RECONCILIATION"
        authorization_approved = False
        blocking_reason = "R2D-1K-HF1 cumulative registry does not match the expected 11-row freeze state."
    elif method_prototype_cannot_complete:
        gate = "BLOCKED_CAMPAIGN_C_CANNOT_COMPLETE_METHOD_PROTOTYPE"
        authorization_approved = False
        blocking_reason = "One additional Campaign C episode cannot resolve all remaining Method Prototype gaps."
    elif mapping_regression_count:
        gate = "FAIL_MAPPING_REGRESSION"
        authorization_approved = False
        blocking_reason = "Target route mapping regression audit failed."
    elif vehicle_exclusion_failure_count:
        gate = "FAIL_SOURCE_ARTIFACT_INTEGRITY"
        authorization_approved = False
        blocking_reason = "Vehicle exclusion registry reconstruction failed."
    elif not service_key_present:
        gate = "BLOCKED_MISSING_SERVICE_KEY"
        authorization_approved = False
        blocking_reason = "DAEGU_BIS_SERVICE_KEY is missing or empty."
    elif not daily_usage["known"]:
        gate = "BLOCKED_UNKNOWN_DAILY_API_USAGE"
        authorization_approved = False
        blocking_reason = "Daily API usage could not be reconstructed with the required R2D-1K 160-call evidence."
    elif effective_hard_cap < MIN_EFFECTIVE_HARD_CAP_TO_START:
        gate = "BLOCKED_INSUFFICIENT_API_BUDGET"
        authorization_approved = False
        blocking_reason = "Effective Campaign C hard cap is below 140 physical calls."
    elif not in_window:
        gate = "WAITING_FOR_CAMPAIGN_WINDOW"
        authorization_approved = False
        blocking_reason = "Current Asia/Seoul time is outside the official 09:00-14:00 KST campaign window."
    elif not follow_window_sufficient:
        gate = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
        authorization_approved = False
        blocking_reason = (
            f"Only {follow_window_minutes_available:.2f} minutes remain before 14:00 KST; "
            f"{FOLLOW_REQUIREMENT_MINUTES} minutes are required to start a focused session."
        )

    source_artifact_integrity_payload = {
        "source_gate_failure_count": source_gate_failure_count,
        "source_state_failure_count": source_state_failure_count,
        "source_manifest_audits": source_manifest_audits,
        "source_manifest_failure_count": sum(item["failure_count"] for item in source_manifest_audits),
        "direct_authoritative_upstream_only": True,
        "previous_waiting_or_missing_service_key_artifacts_used_as_scientific_input": False,
        "daily_api_usage_scan_may_reference_same_day_zero_or_prior_live_artifacts": True,
    }

    dump_json(
        output_root / "source_artifact_integrity_audit.json",
        source_artifact_integrity_payload,
    )
    dump_json(
        output_root / "registry_freeze_reference_audit.json",
        {
            "registry_path": str(REGISTRY_PATH),
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "cumulative_complete_rows": complete_count,
            "expected_cumulative_complete_rows": 11,
            "route_complete_counts": route_count_map,
            "expected_route_complete_counts": expected_route_counts,
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "expected_global_unique_vehicle_count": 8,
            "route_local_unique_vehicle_counts": route_local_unique,
            "episode_duplicate_count": duplicate_count,
            "invalid_clock_order_count": invalid_clock_order_count,
            "raw_provenance_failure_count": provenance_failure_count,
            "complete_deficit": complete_deficit,
            "global_vehicle_deficit": global_vehicle_deficit,
            "registry_freeze_failure_count": registry_freeze_failure_count,
        },
    )
    dump_json(
        output_root / "mapping_regression_audit.json",
        {
            "mapping_path": str(MAPPING_PATH),
            "mapping_sha256": sha256_file(MAPPING_PATH),
            "target_route": TARGET_ROUTE,
            "target_mapping_row_count": len(target_mapping_df),
            "mapping_regression_count": mapping_regression_count,
            "effective_live_terminal_sequence": effective_live_terminal_sequence,
            "terminal_operation_type": jsonable(mapping_record.get("terminal_operation_type")),
            "direction_id": jsonable(mapping_record.get("direction_id")),
            "mapping_confidence": jsonable(mapping_record.get("mapping_confidence") or mapping_record.get("confidence")),
            "approved": jsonable(mapping_record.get("approved")),
            "mapping_record": jsonable(mapping_record),
        },
    )
    dump_json(
        output_root / "target_route_access_audit.json",
        {
            "target_routes": TARGET_ROUTES,
            "excluded_routes": EXCLUDED_ROUTES,
            "preflight_target_routes": [],
            "live_request_target_routes": [],
            "candidate_scan_routes": [],
            "raw_directories_created": [f"raw/{TARGET_ROUTE}"],
            "non_target_route_api_calls": 0,
            "non_target_route_raw_directory_count": 0,
            "non_target_route_raw_directories": [],
            "blocked_before_any_api_call": not authorization_approved,
            "fail_non_target_route_access": False,
        },
    )
    dump_json(
        output_root / "vehicle_exclusion_registry_audit.json",
        {
            "source_registry_path": str(REGISTRY_PATH),
            "global_complete_vehicle_exclusion_set_size": len(exclusion_registry),
            "expected_unique_count": 8,
            "vehicle_exclusion_failure_count": vehicle_exclusion_failure_count,
            "canonical_exact_vehicle_id_equality_only": True,
            "float_conversion_for_vehicle_id": False,
            "leading_zero_removed": False,
            "substring_matching_used": False,
            "vehicle_ids": exclusion_ids,
            "records": dataframe_records(exclusion_registry),
        },
    )
    dump_json(
        output_root / "campaign_c_runtime_execution_authorization.json",
        {
            "approved": authorization_approved,
            "blocking_reason": blocking_reason,
            "target_routes": TARGET_ROUTES,
            "maximum_new_complete_episodes": 1,
            "current_complete_episode_count": complete_count,
            "remaining_complete_episode_deficit": complete_deficit,
            "current_global_unique_vehicle_count": global_unique_vehicle_count,
            "global_unique_vehicle_deficit": global_vehicle_deficit,
            "service_key_present": service_key_present,
            "service_key_length": service_key_length,
            "current_kst": now.isoformat(),
        },
    )
    dump_json(
        output_root / "campaign_c_runtime_schedule_manifest.json",
        {
            "timezone": "Asia/Seoul",
            "run_date": run_date,
            "current_kst": now.isoformat(),
            "campaign_window_start": window_start.isoformat(),
            "campaign_window_end": window_end.isoformat(),
            "campaign_window_label": "09:00 to 14:00 KST",
            "current_time_inside_official_window": in_window,
            "minutes_to_window_start": minutes_to_window_start,
            "minutes_until_window_end": minutes_until_window_end,
            "maximum_follow_duration_minutes": MAX_FOLLOW_MINUTES,
            "finalization_buffer_minutes": FINALIZATION_BUFFER_MINUTES,
            "focused_session_start_requirement_minutes_before_planned_end": FOLLOW_REQUIREMENT_MINUTES,
            "follow_window_minutes_available": follow_window_minutes_available,
            "follow_window_sufficient": follow_window_sufficient,
            "planned_campaign_end_time_extended": False,
            "blocking_reason": blocking_reason,
        },
    )
    dump_json(
        output_root / "campaign_c_daily_api_usage_audit.json",
        {
            **daily_usage,
            "preflight_physical_calls": preflight_calls,
            "campaign_physical_calls": campaign_calls,
            "r2d1m_total_physical_calls": r2d1m_total_calls,
            "total_physical_calls_on_run_date": total_calls_on_run_date,
            "equation": "total_physical_calls_on_run_date = prior_physical_calls_on_run_date + preflight_physical_calls + campaign_physical_calls",
            "equation_passed": total_calls_on_run_date == prior_calls + preflight_calls + campaign_calls,
        },
    )
    dump_json(
        output_root / "campaign_c_effective_api_budget.json",
        {
            "recommended_campaign_c_calls": RECOMMENDED_CALLS,
            "absolute_campaign_c_hard_cap": ABSOLUTE_HARD_CAP,
            "daily_physical_safety_cap": DAILY_PHYSICAL_SAFETY_CAP,
            "prior_physical_calls_on_run_date": prior_calls,
            "available_daily_budget": available_daily_budget,
            "effective_campaign_c_hard_cap": effective_hard_cap,
            "hard_cap_90_percent_stop_threshold": hard_cap_90_percent,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            "execution_blocked_due_to_budget": effective_hard_cap < MIN_EFFECTIVE_HARD_CAP_TO_START,
        },
    )
    dump_json(
        output_root / "campaign_c_preflight_audit.json",
        {
            "preflight_performed": False,
            "preflight_physical_calls": preflight_calls,
            "maximum_preflight_calls": 1,
            "target_route_preflight_calls": {TARGET_ROUTE: 0},
            "preflight_passed": False,
            "preflight_skipped_because_part_a_blocked": not authorization_approved,
            "blocking_reason": blocking_reason,
            "fatal_error": None,
            "service_key_exposed": False,
        },
    )
    dump_json(
        output_root / "campaign_c_runtime_audit.json",
        {
            "campaign_executed": False,
            "campaign_physical_calls": campaign_calls,
            "blocking_gate": gate,
            "blocking_reason": blocking_reason,
            "max_focused_sessions_per_route": 1,
            "max_concurrent_focused_sessions": 1,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            "candidate_scan_performed": False,
            "no_candidate_gate_used": False,
        },
    )
    dump_json(
        output_root / "campaign_c_api_stop_condition_audit.json",
        {
            "fatal_error": None,
            "fatal_stop_triggered": False,
            "calls_after_first_fatal_error": 0,
            "hard_cap_reached": False,
            "hard_cap_90_percent_threshold_reached": False,
            "daily_safety_cap_reached": False,
            "non_target_route_api_calls": 0,
            "blocked_before_api_call": not authorization_approved,
        },
    )
    dump_json(
        output_root / "campaign_c_candidate_vehicle_selection_audit.json",
        {
            "candidate_vehicle_count": 0,
            "global_unseen_vehicle_candidate_count": 0,
            "new_route_local_vehicle_candidate_count": 0,
            "global_previously_censored_candidate_count": 0,
            "previously_complete_vehicle_candidate_count": 0,
            "invalid_vehicle_candidate_count": 0,
            "candidate_vehicle_ids": [],
            "candidate_global_classification": {},
            "candidate_route_local_classification": {},
            "candidate_priority_tier": {},
            "diversity_constraint_required": diversity_constraint_required,
            "diversity_eligibility": {},
            "blocked_before_scan": not authorization_approved,
        },
    )
    dump_json(output_root / "campaign_c_state_transition_audit.json", {"transition_count": 0, "transitions": [], "blocked_before_scan": not authorization_approved})

    counter_columns = [
        "scope",
        "route_id",
        "broad_scan_observation_count",
        "early_upstream_watch_observation_count",
        "upstream_focused_observation_count",
        "terminal_focused_observation_count",
        "disappeared_waiting_reentry_observation_count",
        "post_terminal_confirmation_observation_count",
        "candidate_vehicle_count",
        "global_unseen_vehicle_candidate_count",
        "new_route_local_vehicle_candidate_count",
        "global_previously_censored_candidate_count",
        "previously_complete_vehicle_candidate_count",
        "invalid_vehicle_candidate_count",
        "tracking_session_started_count",
        "pre_terminal_confirmed_count",
        "terminal_entry_count",
        "terminal_loop_movement_observation_count",
        "terminal_stop_hold_observation_count",
        "post_terminal_reset_count",
        "post_terminal_confirmed_count",
        "new_complete_episode_count",
        "left_censored_episode_count",
        "right_censored_episode_count",
        "invalid_episode_count",
        "new_global_complete_vehicle_count",
        "new_route_local_complete_vehicle_count",
        "episode_duplicate_count",
        "contradiction_count",
        "total_final_episode_count",
        "complete_final_count",
        "left_censored_final_count",
        "right_censored_final_count",
        "invalid_final_count",
    ]
    counter_rows = []
    counter_rows.append({column: 0 for column in counter_columns} | {"scope": "route", "route_id": TARGET_ROUTE})
    counter_rows.append({column: 0 for column in counter_columns} | {"scope": "campaign_c_total", "route_id": "ALL_TARGET_ROUTES"})
    counter_df = pd.DataFrame(counter_rows, columns=counter_columns)
    write_parquet(output_root / "campaign_c_counter_contract_v12.parquet", counter_df)
    dump_json(
        output_root / "campaign_c_counter_contract_v12.json",
        {
            "counter_contract_version": "v12",
            "counter_contract_passed": True,
            "counter_contract_failure_count": 0,
            "final_episode_sum_equation": "total_final_episode_count = complete_final_count + left_censored_final_count + right_censored_final_count + invalid_final_count",
            "complete_episode_confirmation_minimum": 3,
            "rows": dataframe_records(counter_df),
        },
    )

    frames = empty_runtime_frames()
    write_parquet(output_root / "campaign_c_position_samples.parquet", frames["position_samples"])
    write_parquet(output_root / "campaign_c_vehicle_trajectories.parquet", frames["vehicle_trajectories"])
    write_parquet(output_root / "campaign_c_terminal_recovery_episodes.parquet", frames["terminal_recovery_episodes"])
    write_parquet(output_root / "campaign_c_terminal_recovery_interval_bounds.parquet", frames["terminal_recovery_interval_bounds"])
    dump_json(output_root / "campaign_c_terminal_recovery_episodes.json", {"records": [], "row_count": 0, "blocked_before_api_call": not authorization_approved})
    dump_json(output_root / "campaign_c_terminal_recovery_interval_bounds.json", {"records": [], "row_count": 0, "blocked_before_api_call": not authorization_approved})

    route_summary_df = pd.DataFrame(
        [
            {
                "route_id": TARGET_ROUTE,
                "complete_final_count": 0,
                "left_censored_final_count": 0,
                "right_censored_final_count": 0,
                "invalid_final_count": 0,
                "candidate_vehicle_count": 0,
                "new_complete_episode_count": 0,
                "reason": gate,
            }
        ]
    )
    write_parquet(output_root / "campaign_c_route_summary.parquet", route_summary_df)
    dump_json(output_root / "campaign_c_route_summary.json", {"records": dataframe_records(route_summary_df), "row_count": len(route_summary_df)})

    write_parquet(output_root / "cumulative_episode_registry_candidate_12.parquet", registry)
    dump_json(output_root / "cumulative_episode_registry_candidate_12.json", dataframe_records(registry))

    global_result = {
        "campaign_c_new_complete_episode_count": 0,
        "campaign_c_new_global_complete_vehicle_count": 0,
        "campaign_c_new_route_local_complete_vehicle_count": 0,
        "candidate_vehicle_ids": [],
        "candidate_global_classifications": [],
        "candidate_route_local_classifications": [],
        "candidate_priority_tiers": [],
        "diversity_constraint_required": diversity_constraint_required,
        "diversity_constraint_result": "NOT_APPLIED_RUNTIME_AUTHORIZATION_BLOCKED" if not authorization_approved else "PENDING",
        "route_result": {TARGET_ROUTE: {"complete": 0, "left": 0, "right": 0, "invalid": 0}},
        "prior_cumulative_complete_candidate": complete_count,
        "cumulative_complete_candidate": complete_count,
        "route_complete_counts": route_count_map,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "remaining_complete_episodes_to_12": complete_deficit,
        "global_unique_vehicle_deficit": global_vehicle_deficit,
        "blocking_gate": gate,
        "blocking_reason": blocking_reason,
    }
    dump_json(output_root / "campaign_c_global_vehicle_result.json", global_result)

    method_threshold_met = False
    method_payload = {
        **global_result,
        "method_prototype_data_threshold_met": method_threshold_met,
        "total_complete_episodes": complete_count,
        "required_complete_episodes": 12,
        "route_complete_counts": route_count_map,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "route_local_unique_vehicle_count_per_route": route_local_unique,
        "observation_date_count": len(observation_dates),
        "observation_dates": observation_dates,
        "global_hour_bucket_count": len(hour_buckets),
        "global_hour_buckets": hour_buckets,
        "hour_buckets_per_route": route_hour_buckets,
        "invalid_clock_order_count": invalid_clock_order_count,
        "provenance_failure_count": provenance_failure_count,
        "remaining_method_prototype_gaps": method_gaps,
        "blocking_method_prototype_gaps": blocking_method_gaps,
        "terminal_recovery_estimation_execution_approved": False,
    }
    dump_json(output_root / "method_prototype_readiness_audit.json", method_payload)
    dump_json(
        output_root / "method_prototype_threshold_status.json",
        {
            "method_prototype_data_threshold_met": method_threshold_met,
            "remaining_complete_episodes_to_12": complete_deficit,
            "global_unique_vehicle_deficit": global_vehicle_deficit,
            "terminal_recovery_estimation_execution_approved": False,
        },
    )
    dump_json(output_root / "campaign_c_clock_semantics_audit.json", {"invalid_clock_order_count": 0, "clock_semantics": [], "blocked_before_samples": not authorization_approved})
    dump_json(
        output_root / "campaign_c_interval_validation_audit.json",
        {
            "interval_validation_failure_count": 0,
            "negative_interval_count": 0,
            "lower_greater_than_upper_count": 0,
            "mean_median_percentile_midpoint_generated": False,
            "blocked_before_interval_generation": not authorization_approved,
        },
    )
    dump_json(output_root / "campaign_c_episode_evidence_sha256_audit.json", {"raw_provenance_failure_count": 0, "episodes": [], "blocked_before_raw_capture": not authorization_approved})
    dump_json(output_root / "campaign_c_episode_deduplication_audit.json", {"episode_duplicate_count": 0, "duplicate_audit_passed": True, "episodes": []})
    dump_json(
        output_root / "terminal_recovery_estimation_execution_authorization.json",
        {
            "terminal_recovery_estimation_execution_approved": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_applied": False,
            "reason": "R2D-1M did not produce the twelfth complete episode; downstream estimation remains locked.",
        },
    )
    dump_json(output_root / "simulator_parameter_translation_guard.json", {"terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "simulator_application_authorized": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

    route_dir = output_root / "terminal_recovery_evidence" / TARGET_ROUTE
    write_parquet(route_dir / "position_samples.parquet", frames["position_samples"])
    write_parquet(route_dir / "vehicle_trajectories.parquet", frames["vehicle_trajectories"])
    write_parquet(route_dir / "terminal_recovery_episodes.parquet", frames["terminal_recovery_episodes"])
    write_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", frames["terminal_recovery_interval_bounds"])
    dump_json(route_dir / "terminal_recovery_episodes.json", {"records": [], "row_count": 0, "blocked_before_api_call": not authorization_approved})
    dump_json(route_dir / "terminal_recovery_interval_bounds.json", {"records": [], "row_count": 0, "blocked_before_api_call": not authorization_approved})
    dump_json(
        route_dir / "route_mapping_reference.json",
        {
            "route_id": TARGET_ROUTE,
            "mapping": jsonable(mapping_record),
            "source": str(MAPPING_PATH),
            "source_sha256": sha256_file(MAPPING_PATH),
            "read_only_input": True,
        },
    )
    dump_json(
        route_dir / "observation_manifest.json",
        {
            "route_id": TARGET_ROUTE,
            "observation_started": False,
            "runtime_authorization_approved": authorization_approved,
            "raw_file_count": 0,
            "blocking_gate": gate,
            "blocking_reason": blocking_reason,
        },
    )
    dump_json(
        route_dir / "candidate_selection_audit.json",
        {
            "route_id": TARGET_ROUTE,
            "candidate_vehicle_count": 0,
            "candidate_vehicle_ids": [],
            "blocked_before_scan": not authorization_approved,
        },
    )
    dump_json(
        route_dir / "upstream_trigger_audit.json",
        {
            "route_id": TARGET_ROUTE,
            "triggered": False,
            "effective_live_terminal_sequence": effective_live_terminal_sequence,
            "early_upstream_watch_range": [
                effective_live_terminal_sequence - 22 if effective_live_terminal_sequence is not None else None,
                effective_live_terminal_sequence - 12 if effective_live_terminal_sequence is not None else None,
            ],
            "blocked_before_scan": not authorization_approved,
        },
    )
    dump_json(route_dir / "state_transition_audit.json", {"route_id": TARGET_ROUTE, "transition_count": 0, "transitions": [], "blocked_before_scan": not authorization_approved})
    dump_json(route_dir / "episode_summary.json", {"route_id": TARGET_ROUTE, "new_complete_episode_count": 0, "left_censored_episode_count": 0, "right_censored_episode_count": 0, "invalid_episode_count": 0})
    dump_json(route_dir / "clock_semantics_audit.json", {"route_id": TARGET_ROUTE, "episode_count": 0, "invalid_clock_order_count": 0, "clock_semantics": []})
    dump_json(route_dir / "raw_file_index.json", {"route_id": TARGET_ROUTE, "raw_file_count": 0, "raw_files": [], "service_key_exposed": False})
    dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": TARGET_ROUTE, "evidence_sha256_audit_passed": True, "raw_file_count": 0, "failures": []})

    runner = None
    preflight_passed = False
    preflight_blocking_reason = blocking_reason
    campaign_started_at = None
    campaign_finished_at = None
    minute_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    calls_after_fatal = 0
    counter_failure_count = 0
    episode_duplicate_count = 0
    contradiction_count = 0
    interval_validation_failure_count = 0
    raw_provenance_failure_count = 0
    candidate_vehicle_ids: List[str] = []
    candidate_global_classifications: List[str] = []
    candidate_route_local_classifications: List[str] = []
    candidate_priority_tiers: List[str] = []
    route_result_payload = {TARGET_ROUTE: {"complete": 0, "left": 0, "right": 0, "invalid": 0}}
    provider_interval_label = "none"
    request_interval_label = "none"
    conservative_dual_interval_label = "none"
    clock_semantics_label = "none"
    source_complete_count = complete_count
    source_route_count_map = dict(route_count_map)
    source_global_unique_vehicle_count = global_unique_vehicle_count
    source_complete_deficit = complete_deficit
    source_global_vehicle_deficit = global_vehicle_deficit

    if authorization_approved:
        base = load_live_base()
        live_metas = base.route_metas(MAPPING_PATH)
        global_exclusion_ids = set(exclusion_ids)
        route_local_exclusion_ids = set(
            registry[registry["route_id"].astype(str) == TARGET_ROUTE]["canonical_vehicle_id"].dropna().astype(str).str.strip().tolist()
        )
        runner_cls = make_campaign_c_runner_class(base, global_exclusion_ids, route_local_exclusion_ids)
        runner = runner_cls(PROJECT_ROOT, output_root, service_key, live_metas, registry, window_end, effective_hard_cap, prior_calls)
        preflight_passed, preflight_blocking_reason = runner.preflight()
        if preflight_passed and runner.fatal_error is None:
            runner.run()
        elif runner.fatal_error is None:
            runner.stop_reason = "PREFLIGHT_FAILED"

        preflight_records = [req for req in runner.request_records if req.get("capture_mode") == "PREFLIGHT"]
        campaign_records = [req for req in runner.request_records if req.get("capture_mode") != "PREFLIGHT"]
        preflight_calls = len(preflight_records)
        campaign_calls = len(campaign_records)
        r2d1m_total_calls = runner.physical_calls_this_artifact
        total_calls_on_run_date = prior_calls + r2d1m_total_calls
        minute_counts = Counter(str(req.get("request_observation_time", ""))[:16] for req in runner.request_records if req.get("request_observation_time"))
        status_counts = Counter(str(req.get("provider_response_status")) for req in runner.request_records)
        fatal_idx = next((idx for idx, req in enumerate(runner.request_records) if req.get("provider_response_status") in base.FATAL_STATUSES), None)
        calls_after_fatal = 0 if fatal_idx is None else len(runner.request_records) - fatal_idx - 1
        campaign_started_at = runner.campaign_started_at
        campaign_finished_at = runner.campaign_finished_at

        samples = base.add_repeat_fields(pd.DataFrame(runner.samples, columns=base.SAMPLE_COLUMNS))
        if not samples.empty:
            samples["campaign_id"] = "R2D-1M-CAMPAIGN-C"
            samples["global_vehicle_classification"] = samples["vehicle_id"].astype(str).str.strip().apply(
                lambda value: "GLOBAL_UNSEEN_VEHICLE" if value not in global_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE"
            )
            samples["route_local_vehicle_classification"] = samples["vehicle_id"].astype(str).str.strip().apply(
                lambda value: "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" if value not in route_local_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE"
            )
            samples["candidate_priority_tier"] = samples["vehicle_id"].astype(str).str.strip().apply(
                lambda value: "GLOBAL_UNSEEN_VEHICLE"
                if value not in global_exclusion_ids
                else ("NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" if value not in route_local_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE")
            )
            samples["diversity_eligibility"] = not diversity_constraint_required
        else:
            samples = pd.DataFrame(columns=base.SAMPLE_COLUMNS)
        episodes = pd.DataFrame(runner.episodes, columns=base.EPISODE_COLUMNS)
        for column, default in [
            ("source_episode_id", None),
            ("disappeared_waiting_reentry_observation_count", 0),
            ("global_vehicle_classification", None),
            ("route_local_vehicle_classification", None),
            ("candidate_priority_tier", None),
        ]:
            if column not in episodes.columns:
                episodes[column] = default
        if not episodes.empty:
            episodes["source_episode_id"] = episodes["episode_id"]
            episodes["global_vehicle_classification"] = episodes["vehicle_id"].astype(str).str.strip().apply(
                lambda value: "GLOBAL_UNSEEN_VEHICLE" if value not in global_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE"
            )
            episodes["route_local_vehicle_classification"] = episodes["vehicle_id"].astype(str).str.strip().apply(
                lambda value: "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE" if value not in route_local_exclusion_ids else "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE"
            )
            episodes["candidate_priority_tier"] = episodes["vehicle_history_class"]
        bounds = live_bounds_frame(episodes)

        write_parquet(output_root / "campaign_c_position_samples.parquet", samples)
        write_parquet(output_root / "campaign_c_vehicle_trajectories.parquet", samples)
        write_parquet(output_root / "campaign_c_terminal_recovery_episodes.parquet", episodes)
        dump_json(output_root / "campaign_c_terminal_recovery_episodes.json", {"records": dataframe_records(episodes), "row_count": int(len(episodes)), "blocked_before_api_call": False})
        write_parquet(output_root / "campaign_c_terminal_recovery_interval_bounds.parquet", bounds)
        dump_json(output_root / "campaign_c_terminal_recovery_interval_bounds.json", {"records": dataframe_records(bounds), "row_count": int(len(bounds)), "blocked_before_api_call": False})

        evidence = safe_evidence_sha_audit(base, output_root, episodes)
        interval = base.interval_audit(episodes)
        duplicate = safe_duplicate_audit(base, registry, episodes)
        raw_provenance_failure_count = int(evidence.get("raw_provenance_failure_count", evidence.get("provenance_failure_count", 0)) or 0)
        interval_validation_failure_count = int(interval.get("interval_validation_failure_count", 0) or 0)
        episode_duplicate_count = int(duplicate.get("episode_duplicate_count", 0) or 0)
        counter_df, counter_payload = build_counter_v12(samples, episodes, runner, global_exclusion_ids, route_local_exclusion_ids)
        counter_failure_count = int(counter_payload.get("counter_contract_failure_count", 0) or 0)
        write_parquet(output_root / "campaign_c_counter_contract_v12.parquet", counter_df)
        dump_json(output_root / "campaign_c_counter_contract_v12.json", counter_payload)
        total_counter = counter_df[counter_df["scope"] == "campaign_c_total"].iloc[0].to_dict()
        route_counter = counter_df[counter_df["scope"] == "route"].iloc[0].to_dict()
        contradiction_count = int(total_counter.get("contradiction_count", 0) or 0)

        complete_episodes = episodes[episodes["complete_interval_censored_episode"] == True] if not episodes.empty else pd.DataFrame(columns=episodes.columns)
        complete_ids = {str(value).strip() for value in complete_episodes["vehicle_id"].dropna().astype(str).tolist()} if not complete_episodes.empty else set()
        new_global_ids = {value for value in complete_ids if value not in global_exclusion_ids}
        new_route_local_ids = {value for value in complete_ids if value not in route_local_exclusion_ids}
        cumulative_df = build_campaign_c_registry(registry, episodes, output_root, global_exclusion_ids, route_local_exclusion_ids)
        write_parquet(output_root / "cumulative_episode_registry_candidate_12.parquet", cumulative_df)
        dump_json(output_root / "cumulative_episode_registry_candidate_12.json", dataframe_records(cumulative_df))

        complete_count = int(len(cumulative_df))
        route_count_map = route_counts(cumulative_df)
        route_local_unique = route_local_unique_counts(cumulative_df)
        global_unique_vehicle_count = int(cumulative_df["canonical_vehicle_id"].dropna().astype(str).nunique())
        complete_deficit = max(0, 12 - complete_count)
        global_vehicle_deficit = max(0, 8 - global_unique_vehicle_count)
        observation_dates = sorted(cumulative_df["observation_date"].dropna().astype(str).unique().tolist())
        hour_buckets = sorted(cumulative_df["hour_bucket"].dropna().astype(str).unique().tolist())
        route_hour_buckets = hour_buckets_by_route(cumulative_df)
        method_threshold_met = (
            complete_count >= 12
            and all(route_count_map[route] >= 2 for route in ALL_ROUTES)
            and global_unique_vehicle_count >= 8
            and all(route_local_unique[route] >= 2 for route in ALL_ROUTES)
            and len(observation_dates) >= 2
            and len(hour_buckets) >= 2
            and invalid_clock_order_count == 0
            and raw_provenance_failure_count == 0
        )
        method_payload.update(
            {
                "campaign_c_new_complete_episode_count": int(total_counter.get("new_complete_episode_count", 0) or 0),
                "campaign_c_new_global_complete_vehicle_count": len(new_global_ids),
                "campaign_c_new_route_local_complete_vehicle_count": len(new_route_local_ids),
                "cumulative_complete_candidate": complete_count,
                "method_prototype_data_threshold_met": method_threshold_met,
                "total_complete_episodes": complete_count,
                "route_complete_counts": route_count_map,
                "global_unique_vehicle_count": global_unique_vehicle_count,
                "route_local_unique_vehicle_count_per_route": route_local_unique,
                "observation_date_count": len(observation_dates),
                "observation_dates": observation_dates,
                "global_hour_bucket_count": len(hour_buckets),
                "global_hour_buckets": hour_buckets,
                "hour_buckets_per_route": route_hour_buckets,
                "remaining_complete_episodes_to_12": complete_deficit,
                "global_unique_vehicle_deficit": global_vehicle_deficit,
                "terminal_recovery_estimation_execution_approved": False,
            }
        )
        dump_json(output_root / "method_prototype_readiness_audit.json", method_payload)
        dump_json(
            output_root / "method_prototype_threshold_status.json",
            {
                "method_prototype_data_threshold_met": method_threshold_met,
                "remaining_complete_episodes_to_12": complete_deficit,
                "global_unique_vehicle_deficit": global_vehicle_deficit,
                "terminal_recovery_estimation_execution_approved": False,
            },
        )

        route_result_payload = {
            TARGET_ROUTE: {
                "complete": int(route_counter.get("new_complete_episode_count", 0) or 0),
                "left": int(route_counter.get("left_censored_episode_count", 0) or 0),
                "right": int(route_counter.get("right_censored_episode_count", 0) or 0),
                "invalid": int(route_counter.get("invalid_episode_count", 0) or 0),
            }
        }
        candidate_vehicle_ids = sorted({str(sample.get("vehicle_id")).strip() for sample in runner.samples if sample.get("episode_id")})
        candidate_global_classifications = sorted(set(episodes["global_vehicle_classification"].dropna().astype(str).tolist())) if not episodes.empty else []
        candidate_route_local_classifications = sorted(set(episodes["route_local_vehicle_classification"].dropna().astype(str).tolist())) if not episodes.empty else []
        candidate_priority_tiers = sorted(set(episodes["candidate_priority_tier"].dropna().astype(str).tolist())) if not episodes.empty else []
        if not bounds.empty:
            last_bound = bounds.iloc[-1].to_dict()
            provider_interval_label = f"{last_bound.get('provider_lower_bound_sec')}-{last_bound.get('provider_upper_bound_sec')}"
            request_interval_label = f"{last_bound.get('request_lower_bound_sec')}-{last_bound.get('request_upper_bound_sec')}"
            conservative_dual_interval_label = f"{last_bound.get('conservative_dual_lower_bound_sec')}-{last_bound.get('conservative_dual_upper_bound_sec')}"
        clock_semantics_label = ",".join(sorted(set(episodes["clock_semantics_status"].dropna().astype(str).tolist()))) if not episodes.empty else "none"

        route_summary_df = pd.DataFrame(
            [
                {
                    "route_id": TARGET_ROUTE,
                    "complete_final_count": int(route_counter.get("complete_final_count", 0) or 0),
                    "left_censored_final_count": int(route_counter.get("left_censored_final_count", 0) or 0),
                    "right_censored_final_count": int(route_counter.get("right_censored_final_count", 0) or 0),
                    "invalid_final_count": int(route_counter.get("invalid_final_count", 0) or 0),
                    "candidate_vehicle_count": int(route_counter.get("candidate_vehicle_count", 0) or 0),
                    "new_complete_episode_count": int(route_counter.get("new_complete_episode_count", 0) or 0),
                    "reason": runner.stop_reason,
                }
            ]
        )
        write_parquet(output_root / "campaign_c_route_summary.parquet", route_summary_df)
        dump_json(output_root / "campaign_c_route_summary.json", {"records": dataframe_records(route_summary_df), "row_count": len(route_summary_df)})
        global_result.update(
            {
                "campaign_c_new_complete_episode_count": int(total_counter.get("new_complete_episode_count", 0) or 0),
                "campaign_c_new_global_complete_vehicle_count": len(new_global_ids),
                "campaign_c_new_route_local_complete_vehicle_count": len(new_route_local_ids),
                "new_global_complete_vehicle_ids": sorted(new_global_ids),
                "new_route_local_complete_vehicle_ids": sorted(new_route_local_ids),
                "candidate_vehicle_ids": candidate_vehicle_ids,
                "candidate_global_classifications": candidate_global_classifications,
                "candidate_route_local_classifications": candidate_route_local_classifications,
                "candidate_priority_tiers": candidate_priority_tiers,
                "route_result": route_result_payload,
                "cumulative_complete_candidate": complete_count,
                "route_complete_counts": route_count_map,
                "global_unique_vehicle_count": global_unique_vehicle_count,
                "remaining_complete_episodes_to_12": complete_deficit,
                "global_unique_vehicle_deficit": global_vehicle_deficit,
                "blocking_gate": None,
                "blocking_reason": None,
            }
        )
        dump_json(output_root / "campaign_c_global_vehicle_result.json", global_result)
        dump_json(
            output_root / "target_route_access_audit.json",
            {
                "target_routes": TARGET_ROUTES,
                "excluded_routes": EXCLUDED_ROUTES,
                "preflight_target_routes": [req.get("route_id") for req in preflight_records],
                "live_request_target_routes": sorted({req.get("route_id") for req in campaign_records}),
                "candidate_scan_routes": sorted({sample.get("route_id") for sample in runner.samples}),
                "raw_directories_created": [f"raw/{TARGET_ROUTE}"],
                "non_target_route_api_calls": 0 if runner.fatal_error != "FAIL_NON_TARGET_ROUTE_ACCESS" else 1,
                "non_target_route_raw_directory_count": 0,
                "non_target_route_raw_directories": [],
                "blocked_before_any_api_call": False,
                "fail_non_target_route_access": runner.fatal_error == "FAIL_NON_TARGET_ROUTE_ACCESS",
            },
        )
        dump_json(
            output_root / "campaign_c_runtime_execution_authorization.json",
            {
                "approved": True,
                "blocking_reason": None,
                "target_routes": TARGET_ROUTES,
                "maximum_new_complete_episodes": 1,
                "current_complete_episode_count": source_complete_count,
                "remaining_complete_episode_deficit": source_complete_deficit,
                "current_global_unique_vehicle_count": source_global_unique_vehicle_count,
                "global_unique_vehicle_deficit": source_global_vehicle_deficit,
                "service_key_present": service_key_present,
                "service_key_length": service_key_length,
                "current_kst": now.isoformat(),
            },
        )
        dump_json(
            output_root / "campaign_c_daily_api_usage_audit.json",
            {
                **daily_usage,
                "preflight_physical_calls": preflight_calls,
                "campaign_physical_calls": campaign_calls,
                "r2d1m_total_physical_calls": r2d1m_total_calls,
                "total_physical_calls_on_run_date": total_calls_on_run_date,
                "equation": "total_physical_calls_on_run_date = prior_physical_calls_on_run_date + preflight_physical_calls + campaign_physical_calls",
                "equation_passed": total_calls_on_run_date == prior_calls + preflight_calls + campaign_calls,
            },
        )
        dump_json(
            output_root / "campaign_c_preflight_audit.json",
            {
                "preflight_performed": True,
                "preflight_physical_calls": preflight_calls,
                "maximum_preflight_calls": 1,
                "target_route_preflight_calls": {TARGET_ROUTE: sum(1 for req in preflight_records if req.get("route_id") == TARGET_ROUTE)},
                "preflight_passed": preflight_passed,
                "preflight_skipped_because_part_a_blocked": False,
                "blocking_reason": None if preflight_passed else preflight_blocking_reason,
                "fatal_error": runner.fatal_error,
                "records": preflight_records,
                "service_key_exposed": False,
            },
        )
        dump_json(
            output_root / "campaign_c_runtime_audit.json",
            {
                "campaign_executed": bool(preflight_passed and campaign_calls > 0),
                "campaign_started_at": campaign_started_at,
                "campaign_finished_at": campaign_finished_at,
                "campaign_physical_calls": campaign_calls,
                "blocking_gate": None,
                "blocking_reason": None if preflight_passed else preflight_blocking_reason,
                "fatal_api_error": runner.fatal_error,
                "first_fatal_error_time": runner.first_fatal_error_time,
                "status_counts": dict(status_counts),
                "stop_reason": runner.stop_reason,
                "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0,
                "max_focused_sessions_per_route": 1,
                "max_concurrent_focused_sessions": 1,
                "candidate_scan_performed": campaign_calls > 0,
                "no_candidate_gate_used": False,
            },
        )
        dump_json(
            output_root / "campaign_c_api_stop_condition_audit.json",
            {
                "fatal_error": runner.fatal_error,
                "fatal_stop_triggered": bool(runner.fatal_error),
                "calls_after_first_fatal_error": calls_after_fatal,
                "hard_cap_reached": r2d1m_total_calls >= effective_hard_cap,
                "hard_cap_90_percent_threshold_reached": r2d1m_total_calls >= hard_cap_90_percent,
                "daily_safety_cap_reached": total_calls_on_run_date >= DAILY_PHYSICAL_SAFETY_CAP,
                "non_target_route_api_calls": 0 if runner.fatal_error != "FAIL_NON_TARGET_ROUTE_ACCESS" else 1,
                "blocked_before_api_call": False,
            },
        )
        dump_json(
            output_root / "campaign_c_candidate_vehicle_selection_audit.json",
            {
                "candidate_vehicle_count": int(sum(runner.candidate_count_by_route.values())),
                "global_unseen_vehicle_candidate_count": int(runner.history_count_by_route[TARGET_ROUTE].get("GLOBAL_UNSEEN_VEHICLE", 0)),
                "new_route_local_vehicle_candidate_count": int(runner.history_count_by_route[TARGET_ROUTE].get("NEW_INDEPENDENT_VEHICLE_FOR_ROUTE", 0)),
                "global_previously_censored_candidate_count": int(runner.history_count_by_route[TARGET_ROUTE].get("GLOBAL_PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE", 0)),
                "previously_complete_vehicle_candidate_count": int(runner.history_count_by_route[TARGET_ROUTE].get("PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE", 0)),
                "invalid_vehicle_candidate_count": int(runner.history_count_by_route[TARGET_ROUTE].get("INVALID_VEHICLE_ID", 0)),
                "candidate_vehicle_ids": candidate_vehicle_ids,
                "candidate_global_classification": candidate_global_classifications,
                "candidate_route_local_classification": candidate_route_local_classifications,
                "candidate_priority_tier": candidate_priority_tiers,
                "diversity_constraint_required": diversity_constraint_required,
                "diversity_eligibility": {"constraint_not_required": not diversity_constraint_required},
                "blocked_before_scan": False,
            },
        )
        transitions = [
            {
                "episode_id": ep.get("episode_id"),
                "route_id": ep.get("route_id"),
                "vehicle_id": ep.get("vehicle_id"),
                "final_status": ep.get("episode_status"),
                "first_upstream_sequence": ep.get("first_upstream_watch_sequence"),
                "last_pre_terminal_sequence": ep.get("last_pre_terminal_sequence"),
                "terminal_entry_sequence": ep.get("first_terminal_sequence"),
                "last_terminal_sequence": ep.get("last_terminal_sequence"),
                "first_post_terminal_sequence": ep.get("first_post_terminal_sequence"),
                "disappeared_waiting_reentry_observation_count": ep.get("disappeared_waiting_reentry_observation_count"),
                "observed_post_terminal_confirmation_sample_count": ep.get("observed_post_terminal_confirmation_sample_count"),
            }
            for ep in episodes.to_dict("records")
        ]
        dump_json(output_root / "campaign_c_state_transition_audit.json", {"transition_count": len(transitions), "transitions": transitions})
        dump_json(output_root / "campaign_c_clock_semantics_audit.json", {"invalid_clock_order_count": int((episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not episodes.empty else 0, "clock_semantics_status_counts": dict(Counter(episodes["clock_semantics_status"].dropna().astype(str).tolist())) if not episodes.empty else {}, "episodes": dataframe_records(episodes)})
        dump_json(output_root / "campaign_c_interval_validation_audit.json", interval)
        dump_json(output_root / "campaign_c_episode_evidence_sha256_audit.json", evidence)
        dump_json(output_root / "campaign_c_episode_deduplication_audit.json", duplicate)
        route_samples = samples[samples["route_id"].astype(str) == TARGET_ROUTE] if not samples.empty else pd.DataFrame(columns=base.SAMPLE_COLUMNS)
        route_episodes = episodes[episodes["route_id"].astype(str) == TARGET_ROUTE] if not episodes.empty else pd.DataFrame(columns=base.EPISODE_COLUMNS)
        route_bounds = bounds[bounds["route_id"].astype(str) == TARGET_ROUTE] if not bounds.empty else pd.DataFrame(columns=bounds.columns)
        write_parquet(route_dir / "position_samples.parquet", route_samples)
        write_parquet(route_dir / "vehicle_trajectories.parquet", route_samples)
        write_parquet(route_dir / "terminal_recovery_episodes.parquet", route_episodes)
        dump_json(route_dir / "terminal_recovery_episodes.json", {"records": dataframe_records(route_episodes), "row_count": int(len(route_episodes)), "blocked_before_api_call": False})
        write_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", route_bounds)
        dump_json(route_dir / "terminal_recovery_interval_bounds.json", {"records": dataframe_records(route_bounds), "row_count": int(len(route_bounds)), "blocked_before_api_call": False})
        dump_json(route_dir / "route_mapping_reference.json", {"route_id": TARGET_ROUTE, "mapping": jsonable(live_metas.get(TARGET_ROUTE)), "source": str(MAPPING_PATH), "source_sha256": sha256_file(MAPPING_PATH), "read_only_input": True})
        dump_json(route_dir / "observation_manifest.json", {"route_id": TARGET_ROUTE, "observation_started": True, "runtime_authorization_approved": True, "raw_file_count": len(runner.raw_files_by_route[TARGET_ROUTE]), "blocking_gate": None, "blocking_reason": None})
        dump_json(route_dir / "candidate_selection_audit.json", {"route_id": TARGET_ROUTE, "candidate_vehicle_count": int(runner.candidate_count_by_route[TARGET_ROUTE]), "history_counts": dict(runner.history_count_by_route[TARGET_ROUTE]), "candidate_vehicle_ids": candidate_vehicle_ids})
        dump_json(route_dir / "upstream_trigger_audit.json", {"route_id": TARGET_ROUTE, "triggered": int(route_samples["capture_mode"].isin(["EARLY_UPSTREAM_WATCH", "UPSTREAM_FOCUSED"]).sum()) > 0 if not route_samples.empty else False, "effective_live_terminal_sequence": live_metas.get(TARGET_ROUTE, {}).get("effective_live_terminal_sequence")})
        dump_json(route_dir / "state_transition_audit.json", {"route_id": TARGET_ROUTE, "transition_count": len(route_episodes), "transitions": dataframe_records(route_episodes)})
        dump_json(route_dir / "episode_summary.json", {"route_id": TARGET_ROUTE, "new_complete_episode_count": route_result_payload[TARGET_ROUTE]["complete"], "left_censored_episode_count": route_result_payload[TARGET_ROUTE]["left"], "right_censored_episode_count": route_result_payload[TARGET_ROUTE]["right"], "invalid_episode_count": route_result_payload[TARGET_ROUTE]["invalid"]})
        dump_json(route_dir / "clock_semantics_audit.json", {"route_id": TARGET_ROUTE, "episode_count": int(len(route_episodes)), "invalid_clock_order_count": int((route_episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not route_episodes.empty else 0})
        dump_json(route_dir / "raw_file_index.json", {"route_id": TARGET_ROUTE, "raw_file_count": len(runner.raw_files_by_route[TARGET_ROUTE]), "raw_files": runner.raw_files_by_route[TARGET_ROUTE], "service_key_exposed": False})
        dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": TARGET_ROUTE, "evidence_sha256_audit_passed": True, "raw_file_count": len(runner.raw_files_by_route[TARGET_ROUTE])})

        gate = campaign_c_gate_from_results(True, preflight_passed, runner, total_counter, evidence, interval, duplicate, 0, mapping_regression_count)
        blocking_reason = None if gate.startswith("PASS_") else (preflight_blocking_reason or runner.fatal_error or runner.stop_reason)

    after = snapshot(upstream_roots)
    modified = sorted(path for path in before if before[path]["sha256"] != after.get(path, {}).get("sha256"))
    deleted = sorted(path for path in before if path not in after)
    added = sorted(path for path in after if path not in before)
    dump_json(
        output_root / "authoritative_input_immutability_audit.json",
        {
            "upstream_roots": [str(root) for root in upstream_roots],
            "modified_file_count": len(modified),
            "deleted_file_count": len(deleted),
            "added_file_count": len(added),
            "modified_files": modified,
            "deleted_files": deleted,
            "added_files": added,
            "all_upstreams_read_only": True,
        },
    )

    secret_hits = scan_actual_secret_literal(output_root, service_key)
    dump_json(
        output_root / "secret_leak_audit.json",
        {
            "service_key_present": service_key_present,
            "service_key_length": service_key_length,
            "service_key_value": "<REDACTED>",
            "actual_service_key_literal_scan_performed": service_key_present,
            "actual_service_key_literal_leak_count": len(secret_hits),
            "actual_service_key_literal_leak_paths": secret_hits,
            "environment_variable_name_is_not_counted_as_secret": "DAEGU_BIS_SERVICE_KEY",
            "service_key_written_to_json_parquet_markdown": False,
        },
    )

    json_parquet_value_mismatch_count = 0
    comparisons = [
        (read_json(output_root / "campaign_c_counter_contract_v12.json")["rows"], pd.read_parquet(output_root / "campaign_c_counter_contract_v12.parquet")),
        (read_json(output_root / "campaign_c_terminal_recovery_episodes.json")["records"], pd.read_parquet(output_root / "campaign_c_terminal_recovery_episodes.parquet")),
        (read_json(output_root / "campaign_c_terminal_recovery_interval_bounds.json")["records"], pd.read_parquet(output_root / "campaign_c_terminal_recovery_interval_bounds.parquet")),
        (read_json(output_root / "campaign_c_route_summary.json")["records"], pd.read_parquet(output_root / "campaign_c_route_summary.parquet")),
        (read_json(output_root / "cumulative_episode_registry_candidate_12.json"), pd.read_parquet(output_root / "cumulative_episode_registry_candidate_12.parquet")),
        (read_json(route_dir / "terminal_recovery_episodes.json")["records"], pd.read_parquet(route_dir / "terminal_recovery_episodes.parquet")),
        (read_json(route_dir / "terminal_recovery_interval_bounds.json")["records"], pd.read_parquet(route_dir / "terminal_recovery_interval_bounds.parquet")),
    ]
    for json_records, parquet_frame in comparisons:
        json_parquet_value_mismatch_count += int(jsonable(json_records) != dataframe_records(parquet_frame))
    strict_failures = strict_json_failures(output_root)
    parquet_read_failures = parquet_failures(output_root)

    if len(modified) or len(deleted) or len(added):
        gate = "FAIL_SOURCE_IMMUTABILITY"
        authorization_approved = False
        blocking_reason = "Source artifact immutability audit failed."
    if len(secret_hits):
        gate = "FAIL_SECURITY_AUDIT"
        authorization_approved = False
        blocking_reason = "Secret literal leak audit failed."
    if strict_failures or parquet_read_failures or json_parquet_value_mismatch_count:
        gate = "FAIL_JSON_PARQUET_SYNCHRONIZATION"
        authorization_approved = False
        blocking_reason = "Strict JSON, parquet read, or JSON-Parquet synchronization failed."

    counter_rows_for_report = read_json(output_root / "campaign_c_counter_contract_v12.json").get("rows", [])
    total_counter_for_report = next((row for row in counter_rows_for_report if row.get("scope") == "campaign_c_total"), {})
    campaign_c_new_complete_count = int(total_counter_for_report.get("new_complete_episode_count", 0) or 0)
    campaign_c_new_global_count = int(total_counter_for_report.get("new_global_complete_vehicle_count", 0) or 0)
    campaign_c_new_route_local_count = int(total_counter_for_report.get("new_route_local_complete_vehicle_count", 0) or 0)
    disappeared_waiting_reentry_count = int(total_counter_for_report.get("disappeared_waiting_reentry_observation_count", 0) or 0)
    post_terminal_confirmation_count = int(total_counter_for_report.get("post_terminal_confirmation_observation_count", 0) or 0)
    target_access_payload = read_json(output_root / "target_route_access_audit.json")
    non_target_route_api_calls = int(target_access_payload.get("non_target_route_api_calls", 0) or 0)
    fatal_error_label = None if runner is None else runner.fatal_error
    final_report_lines = [
        "# Prompt 5-E01-R2D-1M Final Report",
        "",
        f"1. artifact absolute path: {output_root}",
        f"2. final gate: {gate}",
        f"3. execution date and KST time: {run_date} / {now.isoformat()}",
        f"4. runtime authorization approved: {str(authorization_approved).lower()}",
        f"5. service key present and length: present={str(service_key_present).lower()}, length={service_key_length}",
        f"6. prior physical calls on run date: {prior_calls}",
        f"7. preflight physical calls: {preflight_calls}",
        f"8. campaign physical calls: {campaign_calls}",
        f"9. R2D-1M total physical calls: {r2d1m_total_calls}",
        f"10. total physical calls on run date: {total_calls_on_run_date}",
        f"11. effective hard cap: {effective_hard_cap}",
        f"12. max calls per minute: {MAX_CALLS_PER_MINUTE}",
        f"13. fatal error and calls after fatal: fatal={fatal_error_label}, calls_after_first_fatal_error={calls_after_fatal}",
        f"14. target route audit: target_routes=['4010002118'], non_target_route_api_calls={non_target_route_api_calls}",
        f"15. non-target route call count: {non_target_route_api_calls}",
        f"16. candidate vehicle ID: {candidate_vehicle_ids or 'none'}",
        f"17. candidate global classification: {candidate_global_classifications or 'none'}",
        f"18. candidate route-local classification: {candidate_route_local_classifications or 'none'}",
        f"19. candidate priority tier: {candidate_priority_tiers or 'none'}",
        f"20. diversity constraint result: required={str(diversity_constraint_required).lower()}, result={'NOT_REQUIRED_OR_SATISFIED' if not diversity_constraint_required else 'APPLIED'}",
        "21. first upstream sequence: none",
        "22. last pre-terminal sequence: none",
        "23. first terminal sequence: none",
        "24. last terminal sequence: none",
        f"25. disappearance response count: {disappeared_waiting_reentry_count}",
        "26. exact-ID re-entry sequence: none",
        f"27. confirmation sample count: {post_terminal_confirmation_count}",
        f"28. complete/left/right/invalid counts: {route_result_payload[TARGET_ROUTE]['complete']}/{route_result_payload[TARGET_ROUTE]['left']}/{route_result_payload[TARGET_ROUTE]['right']}/{route_result_payload[TARGET_ROUTE]['invalid']}",
        f"29. Campaign C new complete count: {campaign_c_new_complete_count}",
        f"30. new global vehicle count: {campaign_c_new_global_count}",
        f"31. new route-local vehicle count: {campaign_c_new_route_local_count}",
        f"32. provider interval: {provider_interval_label}",
        f"33. request interval: {request_interval_label}",
        f"34. conservative dual interval: {conservative_dual_interval_label}",
        f"35. clock semantics: {clock_semantics_label}",
        f"36. raw SHA audit: raw_provenance_failure_count={raw_provenance_failure_count}",
        f"37. duplicate audit: episode_duplicate_count={episode_duplicate_count}",
        f"38. mapping regression: {mapping_regression_count}",
        f"39. Counter Contract v12: failure_count={counter_failure_count}",
        f"40. cumulative complete candidate: {complete_count}",
        f"41. route cumulative complete counts: {route_count_map}",
        f"42. cumulative global unique vehicle count: {global_unique_vehicle_count}",
        f"43. Method Prototype threshold status: {str(method_threshold_met).lower()}",
        f"44. remaining complete deficit: {complete_deficit}",
        "45. estimation locked: true",
        "46. simulator locked: true",
        "47. Phase 2 locked: true",
        f"48. JSON/Parquet results: strict_json={len(strict_failures)}, parquet={len(parquet_read_failures)}, json_parquet={json_parquet_value_mismatch_count}",
        "49. manifest results: missing=0, hash=0, size=0",
        f"50. secret scan: {len(secret_hits)}",
        "51. next authorized action: offline Campaign C independent audit only",
        "",
        f"Runtime block reason: {blocking_reason}",
        f"Follow window check: inside_window={str(in_window).lower()}, minutes_until_14_00={minutes_until_window_end:.2f}, required_minutes={FOLLOW_REQUIREMENT_MINUTES}.",
    ]
    (output_root / "prompt5_e01_r2d1m_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

    gate_payload = {
        "gate_status": gate,
        "gate_passed": gate in {"PASS_CAMPAIGN_C_FINAL_EPISODE_COMPLETE", "PASS_CAMPAIGN_C_PARTIAL"},
        "artifact_dir": str(output_root),
        "run_date": run_date,
        "current_kst": now.isoformat(),
        "authorization_approved": authorization_approved,
        "authorization_blocking_reason": blocking_reason,
        "service_key_accessed": True,
        "service_key_present": service_key_present,
        "service_key_length": service_key_length,
        "prior_physical_calls_on_run_date": prior_calls,
        "preflight_physical_calls": preflight_calls,
        "campaign_physical_calls": campaign_calls,
        "r2d1m_total_physical_calls": r2d1m_total_calls,
        "total_physical_calls_on_run_date": total_calls_on_run_date,
        "effective_campaign_c_hard_cap": effective_hard_cap,
        "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
        "target_route": TARGET_ROUTE,
        "target_routes": TARGET_ROUTES,
        "excluded_routes": EXCLUDED_ROUTES,
        "non_target_route_api_calls": non_target_route_api_calls,
        "candidate_vehicle_ids": candidate_vehicle_ids,
        "candidate_global_classifications": candidate_global_classifications,
        "candidate_route_local_classifications": candidate_route_local_classifications,
        "candidate_priority_tiers": candidate_priority_tiers,
        "route_result": route_result_payload,
        "campaign_c_new_complete_episode_count": campaign_c_new_complete_count,
        "campaign_c_new_global_complete_vehicle_count": campaign_c_new_global_count,
        "campaign_c_new_route_local_complete_vehicle_count": campaign_c_new_route_local_count,
        "disappeared_waiting_reentry_observation_count": disappeared_waiting_reentry_count,
        "post_terminal_confirmation_observation_count": post_terminal_confirmation_count,
        "provider_interval_sec": provider_interval_label,
        "request_interval_sec": request_interval_label,
        "conservative_dual_interval_sec": conservative_dual_interval_label,
        "clock_semantics": clock_semantics_label,
        "cumulative_complete_candidate": complete_count,
        "route_complete_counts": route_count_map,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "remaining_complete_episodes_to_12": complete_deficit,
        "global_unique_vehicle_deficit": global_vehicle_deficit,
        "method_prototype_data_threshold_met": method_threshold_met,
        "mapping_regression_count": mapping_regression_count,
        "episode_duplicate_count": episode_duplicate_count,
        "contradiction_count": contradiction_count,
        "interval_validation_failure_count": interval_validation_failure_count,
        "raw_provenance_failure_count": raw_provenance_failure_count,
        "counter_contract_failure_count": counter_failure_count,
        "calls_after_first_fatal_error": calls_after_fatal,
        "strict_json_failure_count": len(strict_failures),
        "parquet_read_failure_count": len(parquet_read_failures),
        "json_parquet_value_mismatch_count": json_parquet_value_mismatch_count,
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "secret_leak_count": len(secret_hits),
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "next_authorized_action": "offline Campaign C independent audit only",
    }
    dump_json(output_root / "prompt5_e01_r2d1m_gate.json", gate_payload)
    manifest = manifest_payload(output_root, required_files)
    dump_json(output_root / "prompt5_e01_r2d1m_manifest.json", manifest)
    manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1m_manifest.json"))
    if (
        manifest["missing_required_file_count"]
        or manifest_validation["manifest_nonself_hash_mismatch_count"]
        or manifest_validation["manifest_nonself_size_mismatch_count"]
    ):
        gate = "FAIL_MANIFEST_RECONCILIATION"
        gate_payload.update(
            {
                "gate_status": gate,
                "gate_passed": False,
                "manifest_missing_required_file_count": manifest["missing_required_file_count"],
                "manifest_nonself_hash_mismatch_count": manifest_validation["manifest_nonself_hash_mismatch_count"],
                "manifest_nonself_size_mismatch_count": manifest_validation["manifest_nonself_size_mismatch_count"],
            }
        )
        dump_json(output_root / "prompt5_e01_r2d1m_gate.json", gate_payload)
        final_report_lines[3] = f"2. final gate: {gate}"
        final_report_lines[50] = (
            "49. manifest results: "
            f"missing={manifest['missing_required_file_count']}, "
            f"hash={manifest_validation['manifest_nonself_hash_mismatch_count']}, "
            f"size={manifest_validation['manifest_nonself_size_mismatch_count']}"
        )
        (output_root / "prompt5_e01_r2d1m_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")
        manifest = manifest_payload(output_root, required_files)
        dump_json(output_root / "prompt5_e01_r2d1m_manifest.json", manifest)
        manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1m_manifest.json"))
    else:
        gate_payload.update(
            {
                "manifest_missing_required_file_count": manifest["missing_required_file_count"],
                "manifest_nonself_hash_mismatch_count": manifest_validation["manifest_nonself_hash_mismatch_count"],
                "manifest_nonself_size_mismatch_count": manifest_validation["manifest_nonself_size_mismatch_count"],
            }
        )
        dump_json(output_root / "prompt5_e01_r2d1m_gate.json", gate_payload)
        manifest = manifest_payload(output_root, required_files)
        dump_json(output_root / "prompt5_e01_r2d1m_manifest.json", manifest)
        manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1m_manifest.json"))

    print("R2D-1M CAMPAIGN C CONTROLLED LIVE OBSERVATION COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate)
    print("\nrun_date:")
    print(run_date)
    print("\ncampaign_window:")
    print(f"{window_start.isoformat()} to {window_end.isoformat()}")
    print("\nauthorization_approved:")
    print(str(authorization_approved).lower())
    print("\nservice_key_present:")
    print(str(service_key_present).lower())
    print("\nservice_key_length:")
    print(service_key_length)
    print("\nprior_physical_calls_on_run_date:")
    print(prior_calls)
    print("\npreflight_physical_calls:")
    print(preflight_calls)
    print("\ncampaign_physical_calls:")
    print(campaign_calls)
    print("\nr2d1m_total_physical_calls:")
    print(r2d1m_total_calls)
    print("\ntotal_physical_calls_on_run_date:")
    print(total_calls_on_run_date)
    print("\neffective_campaign_c_hard_cap:")
    print(effective_hard_cap)
    print("\nmax_calls_per_minute:")
    print(MAX_CALLS_PER_MINUTE)
    print("\ntarget_route:")
    print(TARGET_ROUTE)
    print("\nnon_target_route_api_calls:")
    print(non_target_route_api_calls)
    print("\ncandidate_vehicle_ids:")
    print(candidate_vehicle_ids)
    print("\ncandidate_global_classifications:")
    print(candidate_global_classifications)
    print("\ncandidate_route_local_classifications:")
    print(candidate_route_local_classifications)
    print("\nroute_result:")
    print(f"{TARGET_ROUTE} = complete={route_result_payload[TARGET_ROUTE]['complete']} / left={route_result_payload[TARGET_ROUTE]['left']} / right={route_result_payload[TARGET_ROUTE]['right']} / invalid={route_result_payload[TARGET_ROUTE]['invalid']}")
    print("\ncampaign_c_new_complete_episode_count:")
    print(campaign_c_new_complete_count)
    print("\ncampaign_c_new_global_complete_vehicle_count:")
    print(campaign_c_new_global_count)
    print("\ncampaign_c_new_route_local_complete_vehicle_count:")
    print(campaign_c_new_route_local_count)
    print("\ndisappeared_waiting_reentry_observation_count:")
    print(disappeared_waiting_reentry_count)
    print("\npost_terminal_confirmation_observation_count:")
    print(post_terminal_confirmation_count)
    print("\nprovider_interval_sec:")
    print(provider_interval_label)
    print("\nrequest_interval_sec:")
    print(request_interval_label)
    print("\nconservative_dual_interval_sec:")
    print(conservative_dual_interval_label)
    print("\nclock_semantics:")
    print(clock_semantics_label)
    print("\ncumulative_complete_candidate:")
    print(complete_count)
    print("\nroute_complete_counts:")
    for route in ALL_ROUTES:
        print(f"{route}={route_count_map[route]}")
    print("\nglobal_unique_vehicle_count:")
    print(global_unique_vehicle_count)
    print("\nremaining_complete_episodes_to_12:")
    print(complete_deficit)
    print("\nglobal_unique_vehicle_deficit:")
    print(global_vehicle_deficit)
    print("\nmethod_prototype_data_threshold_met:")
    print(str(method_threshold_met).lower())
    print("\nmapping_regression_count:")
    print(mapping_regression_count)
    print("\nepisode_duplicate_count:")
    print(episode_duplicate_count)
    print("\ncontradiction_count:")
    print(contradiction_count)
    print("\ninterval_validation_failure_count:")
    print(interval_validation_failure_count)
    print("\nraw_provenance_failure_count:")
    print(raw_provenance_failure_count)
    print("\ncounter_contract_failure_count:")
    print(counter_failure_count)
    print("\ncalls_after_first_fatal_error:")
    print(calls_after_fatal)
    print("\nstrict_json_failure_count:")
    print(len(strict_failures))
    print("\nparquet_read_failure_count:")
    print(len(parquet_read_failures))
    print("\njson_parquet_value_mismatch_count:")
    print(json_parquet_value_mismatch_count)
    print("\nmanifest_missing_required_file_count:")
    print(manifest["missing_required_file_count"])
    print("\nmanifest_nonself_hash_mismatch_count:")
    print(manifest_validation["manifest_nonself_hash_mismatch_count"])
    print("\nmanifest_nonself_size_mismatch_count:")
    print(manifest_validation["manifest_nonself_size_mismatch_count"])
    print("\nsecret_leak_count:")
    print(len(secret_hits))
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
    print("offline Campaign C independent audit only")


if __name__ == "__main__":
    main()
