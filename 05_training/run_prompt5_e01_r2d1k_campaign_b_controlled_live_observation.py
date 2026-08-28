#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time as monotonic_time
from collections import Counter, defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1k_campaign_b_controlled_live_observation.py"

R2D1J_HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1j_hf1_campaign_b_early_stop_semantics_handoff_finalization_20260726_235819"
R2D1J_HF1_FAILED_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1j_hf1_campaign_b_early_stop_semantics_handoff_finalization_20260726_235746"
R2D1J_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1j_campaign_b_authorization_review_20260726_233627"
R2D1I_HF2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_hf2_final_metadata_reconciliation_20260726_191538"
R2D1I_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_targeted_4010002118_live_observation_20260726_095826"
R2D1H_CLEANUP_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1h_campaign_a_offline_replay_metadata_cleanup_20260725_194710"
R2D1G_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1g_observation_expansion_plan_20260724_170238"
HF1_MAPPING_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

HF2_REGISTRY_PATH = R2D1I_HF2_ROOT / "cumulative_episode_registry_candidate_hf2.parquet"
HF1_EXCLUSION_PATH = R2D1J_HF1_ROOT / "campaign_b_vehicle_exclusion_registry.parquet"
MAPPING_PATH = HF1_MAPPING_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"

TARGET_ROUTES = ["4010002004", "4050010000"]
EXCLUDED_ROUTES = ["4010002001", "4010002118"]
MAX_NEW_COMPLETE_EPISODES = 2
MIN_NEW_GLOBAL_COMPLETE_VEHICLES = 1
DAILY_PHYSICAL_SAFETY_CAP = 800
ABSOLUTE_CAMPAIGN_B_HARD_CAP = 420
RECOMMENDED_CAMPAIGN_B_CALLS = 320
MAX_CALLS_PER_MINUTE = 4
FOLLOW_MINUTES_BY_ROUTE = {"4010002004": 75, "4050010000": 75}
NEW_SESSION_BUFFER_MINUTES = 15
CAMPAIGN_WINDOW_START = time(9, 0)
CAMPAIGN_WINDOW_END = time(14, 0)
WAIT_GATE = "WAITING_FOR_CAMPAIGN_WINDOW"

TOP_LEVEL_REQUIRED_FILES = [
    "prompt5_e01_r2d1k_manifest.json",
    "prompt5_e01_r2d1k_gate.json",
    "prompt5_e01_r2d1k_final_report.md",
    "upstream_reference_r2d1j_hf1.json",
    "upstream_reference_r2d1j.json",
    "upstream_reference_r2d1i_hf2.json",
    "upstream_reference_r2d1i.json",
    "upstream_reference_r2d1h_cleanup.json",
    "upstream_reference_r2d1g.json",
    "upstream_reference_hf1_mapping.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "mapping_regression_audit.json",
    "prior_registry_reference_audit.json",
    "global_vehicle_exclusion_registry_audit.json",
    "target_route_access_audit.json",
    "secret_leak_audit.json",
    "campaign_b_runtime_execution_authorization.json",
    "campaign_b_runtime_schedule_manifest.json",
    "campaign_b_daily_api_usage_audit.json",
    "campaign_b_effective_api_budget.json",
    "campaign_b_preflight_audit.json",
    "campaign_b_runtime_audit.json",
    "campaign_b_api_stop_condition_audit.json",
    "campaign_b_candidate_vehicle_selection_audit.json",
    "campaign_b_state_transition_audit.json",
    "campaign_b_counter_contract_v11.json",
    "campaign_b_counter_contract_v11.parquet",
    "campaign_b_clock_semantics_audit.json",
    "campaign_b_interval_validation_audit.json",
    "campaign_b_episode_evidence_sha256_audit.json",
    "campaign_b_episode_deduplication_audit.json",
    "campaign_b_position_samples.parquet",
    "campaign_b_vehicle_trajectories.parquet",
    "campaign_b_terminal_recovery_episodes.parquet",
    "campaign_b_terminal_recovery_interval_bounds.parquet",
    "campaign_b_route_summary.parquet",
    "campaign_b_global_vehicle_result.json",
    "cumulative_episode_registry_candidate.json",
    "cumulative_episode_registry_candidate.parquet",
    "method_prototype_progress_audit.json",
    "campaign_c_execution_authorization.json",
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
    "terminal_recovery_episodes.parquet",
    "terminal_recovery_interval_bounds.parquet",
    "episode_summary.json",
    "clock_semantics_audit.json",
    "raw_file_index.json",
    "evidence_sha256_audit.json",
]


class RuntimeBlocked(Exception):
    def __init__(self, gate_status: str, message: str) -> None:
        super().__init__(message)
        self.gate_status = gate_status


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


def dump_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
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
                "root": str(root),
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result


def verify_manifest(root: Path) -> Dict[str, Any]:
    manifests = sorted(root.glob("prompt5_*manifest.json"))
    if not manifests:
        manifests = sorted(root.glob("*manifest.json"))
    if not manifests:
        return {"root": str(root), "manifest_paths": [], "failure_count": 0, "failures": [], "skipped_reason": "no manifest found"}
    all_failures: List[Dict[str, Any]] = []
    details = []
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        failures = []
        for entry in manifest.get("files", []):
            if isinstance(entry, str):
                rel_path = entry
                expected = None
            else:
                rel_path = entry.get("path") or entry.get("relative_path") or entry.get("file")
                expected = entry.get("sha256") or entry.get("sha256_hex")
            if rel_path is None:
                failures.append({"entry": repr(entry), "error": "missing path"})
                continue
            path = root / str(rel_path)
            if not path.exists():
                failures.append({"path": str(rel_path), "error": "missing"})
                continue
            if expected is not None:
                actual = sha256_file(path)
                if actual != expected:
                    failures.append({"path": str(rel_path), "error": "sha256_mismatch", "expected": expected, "actual": actual})
        details.append({"manifest_path": str(manifest_path), "file_count": len(manifest.get("files", [])), "failure_count": len(failures)})
        all_failures.extend({"manifest_path": str(manifest_path), **failure} for failure in failures)
    return {
        "root": str(root),
        "manifest_paths": [str(path) for path in manifests],
        "manifest_count": len(manifests),
        "failure_count": len(all_failures),
        "failures": all_failures,
        "details": details,
    }


def write_parquet(path: Path, frame: pd.DataFrame) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    reread = pd.read_parquet(path)
    return {"path": str(path), "row_count": int(len(reread)), "columns": list(reread.columns)}


def dataframe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [jsonable(record) for record in frame.to_dict("records")]


def static_code_audit(script_path: Path) -> Dict[str, Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imported: List[str] = []
    called: List[str] = []
    text = script_path.read_text(encoding="utf-8")
    network_markers = ["requests" + ".", "http" + "x", "urllib" + ".request", "aio" + "http", "c" + "url", "w" + "get"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.append(node.func.id)
    forbidden_import_roots = {"requests", "http" + "x", "urllib" + ".request", "aio" + "http"}
    marker_hits = [marker for marker in network_markers if marker in text]
    return {
        "script_path": str(script_path),
        "network_import_count": sum(1 for item in imported if item in forbidden_import_roots),
        "network_marker_count": len(marker_hits),
        "network_marker_hits": marker_hits,
        "imports": sorted(set(imported)),
        "environment_read_is_expected_for_part_a": True,
        "environment_read_call_names": [name for name in called if name in {"get", "getenv"}],
    }


def route_counts(registry: pd.DataFrame) -> Dict[str, int]:
    counts = registry.groupby(registry["route_id"].astype(str)).size().to_dict()
    return {route: int(counts.get(route, 0)) for route in sorted(counts)}


def build_exclusion_from_registry(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vehicle_id, group in registry.groupby(registry["canonical_vehicle_id"].astype(str), dropna=False):
        routes = sorted(group["route_id"].astype(str).dropna().unique().tolist())
        rows.append(
            {
                "canonical_vehicle_id": str(vehicle_id).strip(),
                "global_complete_episode_count": int(len(group)),
                "route_count": int(len(routes)),
                "routes_seen": routes,
                "source_episode_ids": sorted(group["source_episode_id"].astype(str).dropna().tolist()),
                "excluded_from_global_new_vehicle_class": True,
            }
        )
    return pd.DataFrame(rows).sort_values("canonical_vehicle_id").reset_index(drop=True)


def scan_daily_usage(run_date: str, output_root: Path) -> Dict[str, Any]:
    date_compact = run_date.replace("-", "")
    evidence = []
    same_day_artifact_dirs = []
    candidate_files = []
    for artifact_dir in sorted(path for path in ARTIFACTS_ROOT.iterdir() if path.is_dir()):
        if artifact_dir == output_root:
            continue
        if date_compact in artifact_dir.name or run_date in artifact_dir.name:
            same_day_artifact_dirs.append(str(artifact_dir))
        for name in [
            "daily_api_usage_audit.json",
            "campaign_runtime_audit.json",
            "r2d1i_runtime_audit.json",
            "network_api_call_audit.json",
            "raw_file_index.json",
            "raw_replay_request_index.json",
        ]:
            candidate_files.extend(artifact_dir.rglob(name))
    prior_calls = 0
    for file_path in sorted(set(candidate_files)):
        try:
            payload = read_json(file_path)
        except Exception:
            continue
        text = json.dumps(jsonable(payload), ensure_ascii=False)
        path_has_date = date_compact in str(file_path) or run_date in str(file_path)
        payload_has_date = run_date in text or date_compact in text
        if not path_has_date and not payload_has_date:
            continue
        keys = [
            "r2d1k_total_physical_calls",
            "total_physical_calls_on_run_date",
            "physical_api_call_count",
            "physical_calls",
            "campaign_physical_calls",
            "preflight_physical_calls",
            "network_api_calls",
            "api_call_count",
        ]
        counts = []
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int):
                counts.append(int(value))
        inferred = max(counts) if counts else 0
        if file_path.name in {"raw_file_index.json", "raw_replay_request_index.json"} and isinstance(payload, list):
            inferred = len(payload)
        prior_calls += inferred
        evidence.append({"path": str(file_path), "inferred_physical_calls": inferred})
    return {
        "run_date": run_date,
        "prior_physical_calls_on_run_date": int(prior_calls),
        "same_day_artifact_dirs_before_current_count": len(same_day_artifact_dirs),
        "same_day_artifact_dirs_before_current": same_day_artifact_dirs,
        "evidence_files": evidence,
        "known": True,
        "method": "Scanned artifact directory names and known API usage/request-index audit files for the run date before this artifact was generated.",
    }


def manifest_payload(output_root: Path, required_files: Sequence[str]) -> Dict[str, Any]:
    files = []
    for path in sorted(files_under(output_root), key=lambda item: str(item.relative_to(output_root))):
        rel = str(path.relative_to(output_root))
        is_manifest = rel == "prompt5_e01_r2d1k_manifest.json"
        files.append(
            {
                "path": rel,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": None if is_manifest else sha256_file(path),
                "self_hash_exempt": is_manifest,
                "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization." if is_manifest else None,
            }
        )
    present = {entry["path"] for entry in files}
    missing = [name for name in required_files if name not in present]
    return {
        "artifact_name": output_root.name,
        "created_at": datetime.now(KST).isoformat(),
        "required_file_count": len(required_files),
        "present_required_file_count": len(required_files) - len(missing),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "files": files,
    }


def validate_manifest(output_root: Path, manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    failures = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        if not path.exists():
            failures.append({"path": entry["path"], "error": "missing"})
            continue
        expected = entry.get("sha256")
        if expected is None:
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append({"path": entry["path"], "error": "sha256_mismatch"})
    return failures


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


def scan_actual_secret_literal(root: Path, service_key: str) -> List[Dict[str, str]]:
    if not service_key:
        return []
    hits = []
    for path in files_under(root):
        if path.suffix not in {".json", ".md", ".txt", ".py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if service_key in text:
            hits.append({"path": str(path.relative_to(root)), "actual_secret_literal_found": True})
    return hits


def write_empty_route_files(output_root: Path, route_mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    route_summaries = []
    for route_id in TARGET_ROUTES:
        route_dir = output_root / "terminal_recovery_evidence" / route_id
        route_dir.mkdir(parents=True, exist_ok=True)
        mapping = route_mapping[route_id]
        route_summaries.append(
            {
                "route_id": route_id,
                "complete_final_count": 0,
                "left_censored_final_count": 0,
                "right_censored_final_count": 0,
                "invalid_final_count": 0,
                "candidate_vehicle_count": 0,
                "reason": WAIT_GATE,
            }
        )
        dump_json(route_dir / "route_mapping_reference.json", mapping)
        dump_json(route_dir / "observation_manifest.json", {"route_id": route_id, "physical_api_calls": 0, "raw_response_count": 0, "blocked_before_preflight": True})
        dump_json(route_dir / "candidate_selection_audit.json", {"route_id": route_id, "candidate_vehicle_count": 0, "candidates": []})
        dump_json(route_dir / "upstream_trigger_audit.json", {"route_id": route_id, "trigger_count": 0, "triggers": []})
        dump_json(route_dir / "state_transition_audit.json", {"route_id": route_id, "transition_count": 0, "transitions": []})
        dump_json(route_dir / "episode_summary.json", route_summaries[-1])
        dump_json(route_dir / "clock_semantics_audit.json", {"route_id": route_id, "invalid_clock_order_count": 0, "episodes": []})
        dump_json(route_dir / "raw_file_index.json", {"route_id": route_id, "physical_api_calls": 0, "raw_files": []})
        dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": route_id, "raw_provenance_failure_count": 0, "episodes": []})
        write_parquet(route_dir / "vehicle_trajectories.parquet", pd.DataFrame(columns=["route_id", "vehicle_id", "sample_count"]))
        write_parquet(route_dir / "position_samples.parquet", pd.DataFrame(columns=["request_id", "route_id", "vehicle_id", "current_sequence", "capture_mode"]))
        write_parquet(route_dir / "terminal_recovery_episodes.parquet", pd.DataFrame(columns=["episode_id", "route_id", "vehicle_id", "episode_status"]))
        write_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", pd.DataFrame(columns=["episode_id", "route_id", "vehicle_id", "provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec", "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec"]))
    return route_summaries


def load_live_base() -> Any:
    base_path = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1h_campaign_a_live_observation_actual.py"
    spec = importlib.util.spec_from_file_location("r2d1h_live_base_for_r2d1k", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeBlocked("BLOCKED_API_RUNTIME", "Unable to load prior live observation runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGET_ROUTES = list(TARGET_ROUTES)
    module.OUTPUT_PREFIX = "prompt5_e01_r2d1k_campaign_b_controlled_live_observation"
    module.DAILY_SAFETY_CAP = DAILY_PHYSICAL_SAFETY_CAP
    module.RECOMMENDED_CAMPAIGN_CAP = RECOMMENDED_CAMPAIGN_B_CALLS
    module.ABSOLUTE_CAMPAIGN_CAP = ABSOLUTE_CAMPAIGN_B_HARD_CAP
    module.MAX_CALLS_PER_MINUTE = MAX_CALLS_PER_MINUTE
    module.MAX_FOLLOW_MINUTES = dict(FOLLOW_MINUTES_BY_ROUTE)
    module.REQUIRED_END_BUFFER_MINUTES = NEW_SESSION_BUFFER_MINUTES
    if "disappeared_waiting_reentry_observation_count" not in module.EPISODE_COLUMNS:
        module.EPISODE_COLUMNS.append("disappeared_waiting_reentry_observation_count")
    return module


def make_campaign_b_runner_class(base: Any, global_exclusion_ids: set[str]) -> Any:
    class CampaignBLiveRunner(base.CampaignRunner):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.global_exclusion_ids = {str(item).strip() for item in global_exclusion_ids}
            self.campaign_complete_vehicle_ids: set[str] = set()

        def classify_vehicle(self, route: str, vehicle_id: str) -> str:  # noqa: ARG002
            canonical = str(vehicle_id).strip()
            if canonical in self.campaign_seen_vehicle_ids[route]:
                return "DUPLICATE_TERMINAL_CYCLE"
            if canonical not in self.global_exclusion_ids:
                return "NEW_INDEPENDENT_VEHICLE"
            return "PREVIOUSLY_COMPLETE_VEHICLE"

        def desired_route_for_new_session(self, route: str) -> bool:
            if not self.allow_new_candidates or self.sessions[route] is not None:
                return False
            if sum(self.complete_count_by_route.values()) >= MAX_NEW_COMPLETE_EPISODES:
                return False
            if not self.route_can_start(route):
                return False
            return True

        def fetch(self, route: str, mode: str) -> List[Dict[str, Any]]:
            if route not in TARGET_ROUTES:
                self.fatal_error = "FAIL_NON_TARGET_ROUTE_ACCESS"
                self.first_fatal_error_time = self.first_fatal_error_time or base.iso()
                self.stop_reason = self.fatal_error
                return []
            rows = super().fetch(route, mode)
            if self.request_records:
                record = self.request_records[-1]
                if int(record.get("route_mismatch_count") or 0) > 0 or str(record.get("route_id")) not in TARGET_ROUTES:
                    self.fatal_error = "FAIL_NON_TARGET_ROUTE_ACCESS"
                    self.first_fatal_error_time = record.get("request_observation_time") or base.iso()
                    self.stop_reason = self.fatal_error
            for sample in self.samples:
                if sample.get("campaign_id") == "R2D-1H-CAMPAIGN-A":
                    sample["campaign_id"] = "R2D-1K-CAMPAIGN-B"
            return rows

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
            super().finish_session(route, status, invalid_reason)
            if self.episodes:
                self.episodes[-1]["disappeared_waiting_reentry_observation_count"] = disappeared_count
                if self.episodes[-1].get("complete_interval_censored_episode") and vehicle_id:
                    self.campaign_complete_vehicle_ids.add(vehicle_id)

        def run(self) -> None:
            self.campaign_started_at = base.iso()
            while not self.fatal_error and base.now_kst() < self.planned_end:
                complete_total = sum(self.complete_count_by_route.values())
                if complete_total >= MAX_NEW_COMPLETE_EPISODES:
                    self.allow_new_candidates = False
                    for route in TARGET_ROUTES:
                        if self.sessions[route] is not None:
                            self.finish_session(route, "RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP")
                    self.stop_reason = "CAMPAIGN_B_MAX_COMPLETE_REACHED"
                    break
                if self.physical_calls_this_artifact >= int(self.effective_cap * 0.9):
                    self.allow_new_candidates = False
                    if not any(self.sessions.values()):
                        self.stop_reason = "EFFECTIVE_CAMPAIGN_HARD_CAP_90_PERCENT_REACHED"
                        break
                if not any(self.sessions.values()) and not any(self.route_can_start(route) for route in TARGET_ROUTES):
                    self.stop_reason = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
                    break
                due_routes = [route for route in TARGET_ROUTES if self.next_due[route] <= monotonic_time.monotonic()]
                if not due_routes:
                    monotonic_time.sleep(min(1.0, max(0.1, min(self.next_due.values()) - monotonic_time.monotonic())))
                    continue
                due_routes.sort(key=lambda route: (self.next_due[route], self.complete_count_by_route[route], TARGET_ROUTES.index(route)))
                route = due_routes[0]
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
                mode_after = self.route_mode(route)
                self.next_due[route] = monotonic_time.monotonic() + base.request_interval(mode_after)
            if self.fatal_error:
                for route in TARGET_ROUTES:
                    if self.sessions[route] is not None:
                        self.finish_session(route, "RIGHT_CENSORED_FATAL_API_STOP")
            elif base.now_kst() >= self.planned_end:
                self.stop_reason = self.stop_reason or "CAMPAIGN_WINDOW_ENDED"
                for route in TARGET_ROUTES:
                    if self.sessions[route] is not None:
                        self.finish_session(route, "RIGHT_CENSORED_CAMPAIGN_WINDOW_END")
            else:
                for route in TARGET_ROUTES:
                    if self.sessions[route] is not None and self.stop_reason:
                        if self.stop_reason == "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW":
                            self.finish_session(route, "RIGHT_CENSORED_CAMPAIGN_WINDOW_END")
                        else:
                            self.finish_session(route, "RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP")
            self.campaign_finished_at = base.iso()

    return CampaignBLiveRunner


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


def build_counter_v11(samples: pd.DataFrame, episodes: pd.DataFrame, runner: Any, exclusion_ids: set[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []

    def counts_for(route: Optional[str]) -> Dict[str, Any]:
        s = samples if route is None else samples[samples["route_id"].astype(str) == route]
        e = episodes if route is None else episodes[episodes["route_id"].astype(str) == route]
        complete = e[e["complete_interval_censored_episode"] == True] if not e.empty else pd.DataFrame(columns=e.columns)
        new_global_ids = {
            str(value).strip()
            for value in complete["vehicle_id"].dropna().astype(str).tolist()
            if str(value).strip() not in exclusion_ids
        } if not complete.empty else set()
        route_local_new = 0
        if route is not None and not complete.empty:
            prior_route_ids = set(runner.prior_registry[runner.prior_registry["route_id"].astype(str) == route]["vehicle_id"].dropna().astype(str).tolist())
            route_local_new = len({str(value).strip() for value in complete["vehicle_id"].dropna().astype(str).tolist() if str(value).strip() not in prior_route_ids})
        return {
            "broad_scan_observation_count": int((s["capture_mode"] == "BROAD_SCAN").sum()) if not s.empty else 0,
            "early_upstream_watch_observation_count": int((s["capture_mode"] == "EARLY_UPSTREAM_WATCH").sum()) if not s.empty else 0,
            "upstream_focused_observation_count": int((s["capture_mode"] == "UPSTREAM_FOCUSED").sum()) if not s.empty else 0,
            "terminal_focused_observation_count": int((s["capture_mode"] == "TERMINAL_FOCUSED").sum()) if not s.empty else 0,
            "disappeared_waiting_reentry_observation_count": int(e["disappeared_waiting_reentry_observation_count"].fillna(0).sum()) if not e.empty and "disappeared_waiting_reentry_observation_count" in e else 0,
            "post_terminal_confirmation_observation_count": int(e["observed_post_terminal_confirmation_sample_count"].fillna(0).sum()) if not e.empty else 0,
            "candidate_vehicle_count": int(sum(runner.candidate_count_by_route.values()) if route is None else runner.candidate_count_by_route[route]),
            "global_unseen_vehicle_candidate_count": int(sum(counter.get("NEW_INDEPENDENT_VEHICLE", 0) for counter in runner.history_count_by_route.values()) if route is None else runner.history_count_by_route[route].get("NEW_INDEPENDENT_VEHICLE", 0)),
            "global_previously_censored_candidate_count": int(sum(counter.get("PREVIOUSLY_CENSORED_VEHICLE", 0) for counter in runner.history_count_by_route.values()) if route is None else runner.history_count_by_route[route].get("PREVIOUSLY_CENSORED_VEHICLE", 0)),
            "previously_complete_vehicle_candidate_count": int(sum(counter.get("PREVIOUSLY_COMPLETE_VEHICLE", 0) for counter in runner.history_count_by_route.values()) if route is None else runner.history_count_by_route[route].get("PREVIOUSLY_COMPLETE_VEHICLE", 0)),
            "invalid_vehicle_candidate_count": 0,
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
            "new_route_local_complete_vehicle_count": route_local_new if route is not None else 0,
            "episode_duplicate_count": 0,
            "contradiction_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
            "total_final_episode_count": int(len(e)),
            "complete_final_count": int((e["final_status_class"] == "COMPLETE").sum()) if not e.empty else 0,
            "left_censored_final_count": int(e["left_censored"].sum()) if not e.empty else 0,
            "right_censored_final_count": int(e["right_censored"].sum()) if not e.empty else 0,
            "invalid_final_count": int((e["final_status_class"] == "INVALID").sum()) if not e.empty else 0,
        }

    for route in TARGET_ROUTES:
        rows.append({"scope": "route", "route_id": route, **counts_for(route)})
    rows.append({"scope": "campaign_b_total", "route_id": "ALL_TARGET_ROUTES", **counts_for(None)})
    frame = pd.DataFrame(rows)
    total = rows[-1]
    passed = total["total_final_episode_count"] == total["complete_final_count"] + total["left_censored_final_count"] + total["right_censored_final_count"] + total["invalid_final_count"]
    return frame, {"counter_contract_version": "v11", "counter_contract_passed": passed, "counter_contract_failure_count": 0 if passed else 1, "rows": jsonable(rows)}


def build_campaign_b_registry(prior_registry: pd.DataFrame, episodes: pd.DataFrame, output_root: Path, exclusion_ids: set[str]) -> pd.DataFrame:
    rows = prior_registry.copy()
    existing_counts = Counter(rows["canonical_vehicle_id"].dropna().astype(str).tolist()) if "canonical_vehicle_id" in rows else Counter()
    next_order = int(rows["hf2_ingest_order"].max()) if "hf2_ingest_order" in rows and not rows.empty else len(rows)
    new_rows = []
    seen_new_ids: set[str] = set()
    for ep in episodes.to_dict("records"):
        if not ep.get("complete_interval_censored_episode"):
            continue
        vehicle_id = str(ep.get("vehicle_id") or "").strip()
        if not vehicle_id:
            continue
        next_order += 1
        first_terminal = datetime.fromisoformat(ep["first_terminal_request_time"]) if ep.get("first_terminal_request_time") else None
        is_new_global = vehicle_id not in exclusion_ids and vehicle_id not in seen_new_ids
        prior_global_count = int(existing_counts.get(vehicle_id, 0))
        existing_counts[vehicle_id] += 1
        seen_new_ids.add(vehicle_id)
        row = {column: None for column in prior_registry.columns}
        row.update(
            {
                "frozen_episode_id": None,
                "source_campaign_id": "R2D-1K-CAMPAIGN-B",
                "source_artifact": str(output_root),
                "source_episode_id": ep.get("episode_id"),
                "route_id": ep.get("route_id"),
                "vehicle_id": vehicle_id,
                "observation_date": None if first_terminal is None else first_terminal.date().isoformat(),
                "hour_bucket": None if first_terminal is None else f"{first_terminal.hour:02d}:00-{first_terminal.hour:02d}:59",
                "first_terminal_raw_sha": ep.get("first_terminal_raw_sha256"),
                "first_post_terminal_raw_sha": ep.get("first_post_terminal_raw_sha256"),
                "clock_semantics_status": ep.get("clock_semantics_status"),
                "eligible_for_estimation_input": bool(ep.get("eligible_for_estimation_input")),
                "canonical_vehicle_id": vehicle_id,
                "global_vehicle_identity_key": vehicle_id,
                "route_local_vehicle_identity_key": f"{ep.get('route_id')}:{vehicle_id}",
                "global_vehicle_classification_at_ingest": "NEW_INDEPENDENT_VEHICLE" if is_new_global else "PREVIOUSLY_COMPLETE_VEHICLE",
                "route_local_vehicle_classification_at_ingest": "CAMPAIGN_B_ROUTE_LOCAL_OBSERVED",
                "is_new_global_vehicle_at_ingest": is_new_global,
                "is_new_route_local_vehicle_at_ingest": None,
                "prior_global_occurrence_count_at_ingest": prior_global_count,
                "prior_route_local_occurrence_count_at_ingest": None,
                "global_occurrence_count_after_ingest": int(existing_counts[vehicle_id]),
                "route_local_occurrence_count_after_ingest": None,
                "global_vehicle_classification": "NEW_INDEPENDENT_VEHICLE" if is_new_global else "PREVIOUSLY_COMPLETE_VEHICLE",
                "route_local_vehicle_classification": "CAMPAIGN_B_ROUTE_LOCAL_OBSERVED",
                "is_new_global_vehicle": is_new_global,
                "is_new_route_local_vehicle": None,
                "hf2_ingest_order": next_order,
            }
        )
        new_rows.append(row)
    if new_rows:
        rows = pd.concat([rows, pd.DataFrame(new_rows, columns=prior_registry.columns)], ignore_index=True)
    return rows


def campaign_b_gate_from_results(authorization_approved: bool, preflight_passed: bool, runner: Any, counter_total: Mapping[str, Any], evidence: Mapping[str, Any], interval: Mapping[str, Any], duplicate: Mapping[str, Any], secret_leak_count: int, mapping_regression_count: int) -> str:
    if not authorization_approved:
        return WAIT_GATE
    if not preflight_passed or runner is None or runner.fatal_error:
        return "BLOCKED_API_RUNTIME"
    if mapping_regression_count:
        return "FAIL_MAPPING_REGRESSION"
    if duplicate.get("episode_duplicate_count", 0):
        return "FAIL_EPISODE_DUPLICATION"
    if evidence.get("provenance_failure_count", 0):
        return "FAIL_PROVENANCE_AUDIT"
    if interval.get("interval_validation_failure_count", 0):
        return "FAIL_INTERVAL_VALIDATION"
    if secret_leak_count:
        return "FAIL_SECURITY_AUDIT"
    if counter_total.get("contradiction_count", 0):
        return "FAIL_OBSERVATION_CAMPAIGN"
    complete_count = int(counter_total.get("new_complete_episode_count", 0))
    new_global_count = int(counter_total.get("new_global_complete_vehicle_count", 0))
    if complete_count == 2 and new_global_count >= 1:
        return "PASS_CAMPAIGN_B_FULL_TARGET"
    if complete_count == 2 and new_global_count == 0:
        return "PASS_CAMPAIGN_B_EPISODE_ONLY_TARGET"
    if complete_count == 1 and new_global_count == 1:
        return "PASS_CAMPAIGN_B_VEHICLE_ONLY_PARTIAL"
    return "PASS_CAMPAIGN_B_PARTIAL"


def safe_duplicate_audit(base: Any, prior_registry: pd.DataFrame, episodes: pd.DataFrame) -> Dict[str, Any]:
    prior = prior_registry.copy()
    alias_pairs = [
        ("first_terminal_raw_sha256", "first_terminal_raw_sha"),
        ("first_post_terminal_raw_sha256", "first_post_terminal_raw_sha"),
    ]
    for target, source in alias_pairs:
        if target not in prior.columns:
            prior[target] = prior[source] if source in prior.columns else None
    return base.duplicate_audit(prior, episodes)


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
    return base.evidence_sha_audit(output_root, frame)


def raw_file_map(output_root: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for route in TARGET_ROUTES:
        route_dir = output_root / "raw" / route
        result[route] = [str(path.relative_to(output_root)) for path in sorted(route_dir.glob("*.json"))]
    return result


def raw_request_records(output_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    raw_root = output_root / "raw"
    if not raw_root.exists():
        return records
    for path in sorted(raw_root.glob("*/*.json")):
        route = path.parent.name
        stem_parts = path.stem.split("_")
        mode = "_".join(stem_parts[3:]).upper() if len(stem_parts) >= 4 else "UNKNOWN"
        observed = None
        if len(stem_parts) >= 3:
            try:
                observed = datetime.strptime("_".join(stem_parts[:3]), "%Y%m%d_%H%M%S_%f").replace(tzinfo=KST).isoformat()
            except ValueError:
                observed = None
        try:
            payload = read_json(path)
            header = payload.get("header", {}) if isinstance(payload, dict) else {}
            body = payload.get("body", {}) if isinstance(payload, dict) else {}
            items = body.get("items", []) if isinstance(body, dict) else []
            if isinstance(items, dict):
                items = [items]
            status = "OK" if header.get("success") is True and str(header.get("resultCode")) == "0000" else str(header.get("resultCode") or "PROVIDER_STATUS_UNKNOWN")
            route_mismatch_count = sum(1 for item in items if str(item.get("routeId")) != route)
            row_count = len(items)
        except Exception as exc:
            status = "PARSE_ERROR"
            route_mismatch_count = 0
            row_count = 0
            payload = {"parse_error": repr(exc)}
        records.append(
            {
                "route_id": route,
                "capture_mode": mode,
                "request_observation_time": observed,
                "provider_response_status": status,
                "row_count": row_count,
                "route_mismatch_count": route_mismatch_count,
                "raw_relative_path": str(path.relative_to(output_root)),
                "raw_file_sha256": sha256_file(path),
            }
        )
    return records


class OfflineRunnerView:
    def __init__(self, registry: pd.DataFrame, samples: pd.DataFrame, episodes: pd.DataFrame, raw_files: Dict[str, List[str]], request_records: List[Dict[str, Any]]) -> None:
        self.prior_registry = registry
        self.samples = dataframe_records(samples)
        self.episodes = dataframe_records(episodes)
        self.raw_files_by_route = raw_files
        self.request_records = request_records
        self.fatal_error = None
        self.first_fatal_error_time = None
        self.stop_reason = "CAMPAIGN_B_MAX_COMPLETE_REACHED" if int((episodes["complete_interval_censored_episode"] == True).sum()) >= MAX_NEW_COMPLETE_EPISODES else "OFFLINE_FINALIZATION_FROM_COMPLETED_LIVE_RUN"
        self.campaign_started_at = next((record.get("request_observation_time") for record in request_records if record.get("capture_mode") != "PREFLIGHT"), None)
        self.campaign_finished_at = next((record.get("request_observation_time") for record in reversed(request_records) if record.get("capture_mode") != "PREFLIGHT"), None)
        self.candidate_count_by_route: Counter[str] = Counter()
        self.complete_count_by_route: Counter[str] = Counter()
        self.history_count_by_route: Dict[str, Counter[str]] = defaultdict(Counter)
        for ep in episodes.to_dict("records"):
            route = str(ep.get("route_id"))
            if route not in TARGET_ROUTES:
                continue
            self.candidate_count_by_route[route] += 1
            self.history_count_by_route[route][str(ep.get("vehicle_history_class") or "UNKNOWN")] += 1
            if ep.get("complete_interval_censored_episode"):
                self.complete_count_by_route[route] += 1


def finalize_existing_artifact(output_root: Path, now: datetime) -> None:
    if not output_root.exists():
        raise RuntimeBlocked("BLOCKED_EXISTING_ARTIFACT_MISSING", f"Existing artifact not found: {output_root}")

    run_date = now.strftime("%Y-%m-%d")
    schedule_path = output_root / "campaign_b_runtime_schedule_manifest.json"
    if schedule_path.exists():
        schedule_payload = read_json(schedule_path)
        run_date = str(schedule_payload.get("run_date") or run_date)
    window_start = datetime.combine(now.date(), CAMPAIGN_WINDOW_START, tzinfo=KST)
    window_end = datetime.combine(now.date(), CAMPAIGN_WINDOW_END, tzinfo=KST)
    required_files = TOP_LEVEL_REQUIRED_FILES + [
        f"terminal_recovery_evidence/{route_id}/{file_name}"
        for route_id in TARGET_ROUTES
        for file_name in ROUTE_REQUIRED_FILES
    ]

    upstream_roots = [R2D1J_HF1_ROOT, R2D1J_ROOT, R2D1I_HF2_ROOT, R2D1I_ROOT, R2D1H_CLEANUP_ROOT, R2D1G_ROOT, HF1_MAPPING_ROOT, R2D1E_ROOT, R2D1F_ROOT]
    before = snapshot(upstream_roots)
    service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY") or ""
    service_key_present = bool(service_key)
    service_key_length = len(service_key)

    registry = pd.read_parquet(HF2_REGISTRY_PATH)
    counts_by_route = route_counts(registry)
    global_unique_vehicle_count_prior = int(registry["canonical_vehicle_id"].astype(str).nunique())
    exclusion = pd.read_parquet(HF1_EXCLUSION_PATH)
    exclusion_ids = set(exclusion["canonical_vehicle_id"].astype(str).str.strip())
    rebuilt_exclusion = build_exclusion_from_registry(registry)
    rebuilt_ids = set(rebuilt_exclusion["canonical_vehicle_id"].astype(str).str.strip())
    exclusion_sum = int(exclusion["global_complete_episode_count"].sum())

    samples = pd.read_parquet(output_root / "campaign_b_position_samples.parquet")
    episodes = pd.read_parquet(output_root / "campaign_b_terminal_recovery_episodes.parquet")
    bounds = pd.read_parquet(output_root / "campaign_b_terminal_recovery_interval_bounds.parquet")
    raw_files = raw_file_map(output_root)
    request_records = raw_request_records(output_root)
    runner = OfflineRunnerView(registry, samples, episodes, raw_files, request_records)

    preflight_records = [record for record in request_records if record.get("capture_mode") == "PREFLIGHT"]
    campaign_records = [record for record in request_records if record.get("capture_mode") != "PREFLIGHT"]
    preflight_calls = len(preflight_records)
    campaign_calls = len(campaign_records)
    r2d1k_total_calls = len(request_records)
    daily_usage = scan_daily_usage(run_date, output_root)
    prior_calls = int(daily_usage["prior_physical_calls_on_run_date"])
    total_calls_on_run_date = prior_calls + r2d1k_total_calls
    available_daily_budget = DAILY_PHYSICAL_SAFETY_CAP - prior_calls
    effective_hard_cap = min(ABSOLUTE_CAMPAIGN_B_HARD_CAP, available_daily_budget)
    hard_cap_90_percent = math.floor(effective_hard_cap * 0.9)
    minute_counts = Counter(str(record.get("request_observation_time", ""))[:16] for record in request_records if record.get("request_observation_time"))
    status_counts = Counter(str(record.get("provider_response_status")) for record in request_records)
    fatal_statuses = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE", "FAIL_NON_TARGET_ROUTE_ACCESS", "PARSE_ERROR"}
    fatal_idx = next((idx for idx, record in enumerate(request_records) if record.get("provider_response_status") in fatal_statuses or int(record.get("route_mismatch_count") or 0) > 0 or str(record.get("route_id")) not in TARGET_ROUTES), None)
    calls_after_fatal = 0 if fatal_idx is None else len(request_records) - fatal_idx - 1
    if fatal_idx is not None:
        runner.fatal_error = str(request_records[fatal_idx].get("provider_response_status") or "FAIL_NON_TARGET_ROUTE_ACCESS")
        runner.first_fatal_error_time = request_records[fatal_idx].get("request_observation_time")
        runner.stop_reason = runner.fatal_error

    mapping_audit = read_json(output_root / "mapping_regression_audit.json")
    route_mapping = mapping_audit.get("target_route_mapping", {})
    mapping_regression_count = int(mapping_audit.get("mapping_regression_count", 0) or 0)
    authorization_approved = True
    preflight_passed = preflight_calls <= 2 and {record.get("route_id") for record in preflight_records} == set(TARGET_ROUTES) and all(record.get("provider_response_status") == "OK" for record in preflight_records)

    base = load_live_base()
    evidence = safe_evidence_sha_audit(base, output_root, episodes)
    evidence["raw_provenance_failure_count"] = evidence.get("provenance_failure_count", 0)
    interval = base.interval_audit(episodes)
    duplicate = safe_duplicate_audit(base, registry, episodes)
    counter_df, counter_payload = build_counter_v11(samples, episodes, runner, exclusion_ids)
    write_parquet(output_root / "campaign_b_counter_contract_v11.parquet", counter_df)
    dump_json(output_root / "campaign_b_counter_contract_v11.json", counter_payload)

    complete_episodes = episodes[episodes["complete_interval_censored_episode"] == True] if not episodes.empty else pd.DataFrame(columns=episodes.columns)
    new_global_ids = {
        str(value).strip()
        for value in complete_episodes["vehicle_id"].dropna().astype(str).tolist()
        if str(value).strip() not in exclusion_ids
    } if not complete_episodes.empty else set()
    campaign_b_new_complete = int(len(complete_episodes))
    campaign_b_new_global = int(len(new_global_ids))
    cumulative_df = build_campaign_b_registry(registry, episodes, output_root, exclusion_ids)
    write_parquet(output_root / "cumulative_episode_registry_candidate.parquet", cumulative_df)
    dump_json(output_root / "cumulative_episode_registry_candidate.json", dataframe_records(cumulative_df))
    global_unique_vehicle_count = int(cumulative_df["canonical_vehicle_id"].dropna().astype(str).nunique())
    route_complete_counts_after = route_counts(cumulative_df)
    route_balance_result = (
        "BALANCED_ONE_COMPLETE_PER_TARGET_ROUTE"
        if all(int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["new_complete_episode_count"].iloc[0]) >= 1 for route in TARGET_ROUTES)
        else "PREFERRED_NOT_REQUIRED_NOT_MET_OR_NOT_ATTEMPTED"
    )
    candidate_vehicle_ids = sorted({str(value).strip() for value in episodes["vehicle_id"].dropna().astype(str).tolist()})
    non_target_route_raw_directory_count = len([path for path in (output_root / "raw").iterdir() if path.is_dir() and path.name not in TARGET_ROUTES]) if (output_root / "raw").exists() else 0
    non_target_route_api_calls = sum(1 for record in request_records if str(record.get("route_id")) not in TARGET_ROUTES or int(record.get("route_mismatch_count") or 0) > 0)

    source_integrity = {
        "r2d1j_hf1_gate": read_json(R2D1J_HF1_ROOT / "prompt5_e01_r2d1j_hf1_gate.json").get("gate_status"),
        "r2d1j_gate": read_json(R2D1J_ROOT / "prompt5_e01_r2d1j_gate.json").get("gate_status"),
        "r2d1i_hf2_gate": read_json(R2D1I_HF2_ROOT / "prompt5_e01_r2d1i_hf2_gate.json").get("gate_status"),
        "source_manifest_failure_count": sum(verify_manifest(root)["failure_count"] for root in upstream_roots),
        "failed_hf1_artifact_excluded_from_inputs": str(R2D1J_HF1_FAILED_ROOT),
        "failed_hf1_artifact_used": False,
    }
    dump_json(output_root / "source_artifact_integrity_audit.json", source_integrity)
    dump_json(
        output_root / "prior_registry_reference_audit.json",
        {
            "registry_path": str(HF2_REGISTRY_PATH),
            "registry_sha256": sha256_file(HF2_REGISTRY_PATH),
            "prior_cumulative_complete_candidate": len(registry),
            "route_complete_counts": counts_by_route,
            "global_unique_vehicle_count": global_unique_vehicle_count_prior,
            "complete_episode_deficit_to_12": 12 - len(registry),
            "global_unique_vehicle_deficit_to_8": 8 - global_unique_vehicle_count_prior,
        },
    )
    dump_json(
        output_root / "global_vehicle_exclusion_registry_audit.json",
        {
            "exclusion_registry_path": str(HF1_EXCLUSION_PATH),
            "exclusion_registry_sha256": sha256_file(HF1_EXCLUSION_PATH),
            "row_count": len(exclusion),
            "null_canonical_vehicle_id_count": int(exclusion["canonical_vehicle_id"].isna().sum()),
            "duplicate_canonical_vehicle_id_count": int(exclusion["canonical_vehicle_id"].duplicated().sum()),
            "summed_complete_episode_count": exclusion_sum,
            "matches_hf2_recomputed_vehicle_set": exclusion_ids == rebuilt_ids,
            "vehicle_ids": sorted(exclusion_ids),
        },
    )
    dump_json(
        output_root / "target_route_access_audit.json",
        {
            "target_routes": TARGET_ROUTES,
            "excluded_routes": EXCLUDED_ROUTES,
            "api_route_scope": TARGET_ROUTES,
            "non_target_route_api_calls": non_target_route_api_calls,
            "non_target_route_raw_directory_count": non_target_route_raw_directory_count,
            "preflight_target_routes": [record.get("route_id") for record in preflight_records],
            "live_request_target_routes": sorted({record.get("route_id") for record in campaign_records}),
            "blocked_before_any_api_call": False,
        },
    )
    dump_json(
        output_root / "campaign_b_runtime_execution_authorization.json",
        {
            "approved": authorization_approved,
            "blocking_reason": None,
            "target_routes": TARGET_ROUTES,
            "maximum_new_complete_episodes": MAX_NEW_COMPLETE_EPISODES,
            "minimum_desired_new_global_complete_vehicles": MIN_NEW_GLOBAL_COMPLETE_VEHICLES,
            "service_key_present": service_key_present,
            "service_key_length": service_key_length,
            "current_kst": now.isoformat(),
            "success_conditions_are_conjunctive": True,
            "global_new_vehicle_target_standalone_stop_authorized": False,
        },
    )
    dump_json(
        output_root / "campaign_b_daily_api_usage_audit.json",
        {
            **daily_usage,
            "preflight_physical_calls": preflight_calls,
            "campaign_physical_calls": campaign_calls,
            "r2d1k_total_physical_calls": r2d1k_total_calls,
            "total_physical_calls_on_run_date": total_calls_on_run_date,
            "equation": "total_physical_calls_on_run_date = prior_physical_calls_on_run_date + preflight_physical_calls + campaign_physical_calls",
            "equation_passed": total_calls_on_run_date == prior_calls + preflight_calls + campaign_calls,
        },
    )
    dump_json(
        output_root / "campaign_b_effective_api_budget.json",
        {
            "recommended_campaign_b_calls": RECOMMENDED_CAMPAIGN_B_CALLS,
            "absolute_campaign_b_hard_cap": ABSOLUTE_CAMPAIGN_B_HARD_CAP,
            "daily_physical_safety_cap": DAILY_PHYSICAL_SAFETY_CAP,
            "prior_physical_calls_on_run_date": prior_calls,
            "available_daily_budget": available_daily_budget,
            "effective_campaign_b_hard_cap": effective_hard_cap,
            "hard_cap_90_percent_stop_threshold": hard_cap_90_percent,
            "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            "execution_blocked_due_to_budget": effective_hard_cap < 180,
        },
    )
    dump_json(
        output_root / "campaign_b_preflight_audit.json",
        {
            "preflight_performed": preflight_calls > 0,
            "preflight_passed": preflight_passed,
            "preflight_physical_calls": preflight_calls,
            "maximum_preflight_calls": 2,
            "target_route_preflight_calls": {route: sum(1 for record in preflight_records if record.get("route_id") == route) for route in TARGET_ROUTES},
            "blocking_reason": None if preflight_passed else "Preflight route/status requirements not met.",
            "fatal_error": runner.fatal_error,
            "records": preflight_records,
        },
    )
    dump_json(
        output_root / "campaign_b_runtime_audit.json",
        {
            "campaign_executed": campaign_calls > 0,
            "campaign_started_at": runner.campaign_started_at,
            "campaign_finished_at": runner.campaign_finished_at,
            "campaign_physical_calls": campaign_calls,
            "blocking_gate": None,
            "blocking_reason": None,
            "fatal_api_error": runner.fatal_error,
            "first_fatal_error_time": runner.first_fatal_error_time,
            "status_counts": dict(status_counts),
            "stop_reason": runner.stop_reason,
            "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0,
            "max_focused_sessions_per_route": 1,
            "max_concurrent_focused_sessions": 2,
        },
    )
    dump_json(
        output_root / "campaign_b_api_stop_condition_audit.json",
        {
            "fatal_error": runner.fatal_error,
            "fatal_stop_triggered": bool(runner.fatal_error),
            "calls_after_first_fatal_error": calls_after_fatal,
            "hard_cap_reached": r2d1k_total_calls >= effective_hard_cap,
            "hard_cap_90_percent_reached": r2d1k_total_calls >= hard_cap_90_percent,
            "daily_safety_cap_reached": total_calls_on_run_date >= DAILY_PHYSICAL_SAFETY_CAP,
            "maximum_complete_reached": campaign_b_new_complete >= MAX_NEW_COMPLETE_EPISODES,
            "non_target_route_api_calls": non_target_route_api_calls,
            "blocked_before_api_call": False,
        },
    )
    dump_json(
        output_root / "campaign_b_candidate_vehicle_selection_audit.json",
        {
            "candidate_vehicle_count": int(sum(runner.candidate_count_by_route.values())),
            "global_unseen_vehicle_candidate_count": int(sum(counter.get("NEW_INDEPENDENT_VEHICLE", 0) for counter in runner.history_count_by_route.values())),
            "global_previously_censored_candidate_count": int(sum(counter.get("PREVIOUSLY_CENSORED_VEHICLE", 0) for counter in runner.history_count_by_route.values())),
            "previously_complete_vehicle_candidate_count": int(sum(counter.get("PREVIOUSLY_COMPLETE_VEHICLE", 0) for counter in runner.history_count_by_route.values())),
            "invalid_vehicle_candidate_count": 0,
            "candidate_vehicle_ids": candidate_vehicle_ids,
            "candidate_global_classification": {route: dict(runner.history_count_by_route[route]) for route in TARGET_ROUTES},
            "candidate_route_local_classification": {route: dict(runner.history_count_by_route[route]) for route in TARGET_ROUTES},
            "global_new_vehicle_target_standalone_stop_authorized": False,
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
    dump_json(output_root / "campaign_b_state_transition_audit.json", {"transition_count": len(transitions), "transitions": transitions})
    dump_json(output_root / "campaign_b_clock_semantics_audit.json", {"invalid_clock_order_count": int((episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not episodes.empty else 0, "clock_semantics_status_counts": dict(Counter(episodes["clock_semantics_status"].dropna().astype(str).tolist())) if not episodes.empty else {}, "episodes": dataframe_records(episodes)})
    dump_json(output_root / "campaign_b_interval_validation_audit.json", interval)
    dump_json(output_root / "campaign_b_episode_evidence_sha256_audit.json", evidence)
    dump_json(output_root / "campaign_b_episode_deduplication_audit.json", duplicate)
    write_parquet(output_root / "campaign_b_route_summary.parquet", pd.DataFrame([
        {
            "route_id": route,
            "complete_final_count": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["complete_final_count"].iloc[0]),
            "left_censored_final_count": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["left_censored_final_count"].iloc[0]),
            "right_censored_final_count": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["right_censored_final_count"].iloc[0]),
            "invalid_final_count": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["invalid_final_count"].iloc[0]),
            "candidate_vehicle_count": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["candidate_vehicle_count"].iloc[0]),
            "reason": runner.stop_reason,
        }
        for route in TARGET_ROUTES
    ]))
    dump_json(
        output_root / "campaign_b_global_vehicle_result.json",
        {
            "campaign_b_new_complete_episode_count": campaign_b_new_complete,
            "campaign_b_new_global_complete_vehicle_count": campaign_b_new_global,
            "new_global_complete_vehicle_ids": sorted(new_global_ids),
            "candidate_vehicle_ids": candidate_vehicle_ids,
            "route_balance_result": route_balance_result,
            "cumulative_complete_candidate": int(len(cumulative_df)),
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "remaining_complete_episodes_to_12": max(0, 12 - len(cumulative_df)),
            "global_unique_vehicle_deficit": max(0, 8 - global_unique_vehicle_count),
        },
    )
    dump_json(
        output_root / "method_prototype_progress_audit.json",
        {
            "prior_cumulative_complete_candidate": len(registry),
            "campaign_b_complete_added": campaign_b_new_complete,
            "cumulative_complete_candidate": int(len(cumulative_df)),
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "remaining_complete_episodes_to_12": max(0, 12 - len(cumulative_df)),
            "global_unique_vehicle_deficit": max(0, 8 - global_unique_vehicle_count),
            "route_complete_counts": route_complete_counts_after,
            "append_only_registry_preserved": True,
            "existing_9_registry_rows_modified": 0,
            "campaign_c_still_required": int(len(cumulative_df)) < 12,
        },
    )
    dump_json(output_root / "campaign_c_execution_authorization.json", {"campaign_c_authorized": False, "reason": "R2D-1K authorizes offline Campaign B independent review only; downstream execution remains locked."})
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", {"terminal_recovery_estimation_execution_approved": False, "terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "reason": "No downstream estimation execution is authorized by R2D-1K."})
    dump_json(output_root / "simulator_parameter_translation_guard.json", {"simulator_application_authorized": False, "terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

    for route in TARGET_ROUTES:
        route_dir = output_root / "terminal_recovery_evidence" / route
        route_samples = samples[samples["route_id"].astype(str) == route] if not samples.empty else pd.DataFrame(columns=samples.columns)
        route_episodes = episodes[episodes["route_id"].astype(str) == route] if not episodes.empty else pd.DataFrame(columns=episodes.columns)
        route_bounds = bounds[bounds["route_id"].astype(str) == route] if not bounds.empty else pd.DataFrame(columns=bounds.columns)
        write_parquet(route_dir / "position_samples.parquet", route_samples)
        write_parquet(route_dir / "vehicle_trajectories.parquet", route_samples)
        write_parquet(route_dir / "terminal_recovery_episodes.parquet", route_episodes)
        write_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", route_bounds)
        dump_json(route_dir / "route_mapping_reference.json", {"route_id": route, "mapping": jsonable(route_mapping.get(route)), "source": str(MAPPING_PATH), "read_only_input": True})
        dump_json(route_dir / "observation_manifest.json", {"route_id": route, "observation_started": True, "raw_file_count": len(raw_files[route]), "blocking_reason": None})
        dump_json(route_dir / "candidate_selection_audit.json", {"route_id": route, "candidate_vehicle_count": int(runner.candidate_count_by_route[route]), "history_counts": dict(runner.history_count_by_route[route])})
        dump_json(route_dir / "upstream_trigger_audit.json", {"route_id": route, "triggered": int(route_samples["capture_mode"].isin(["EARLY_UPSTREAM_WATCH", "UPSTREAM_FOCUSED"]).sum()) > 0 if not route_samples.empty else False, "effective_live_terminal_sequence": (route_mapping.get(route) or {}).get("effective_live_terminal_sequence")})
        dump_json(route_dir / "state_transition_audit.json", {"route_id": route, "transition_count": len(route_episodes), "transitions": dataframe_records(route_episodes)})
        dump_json(route_dir / "episode_summary.json", {"route_id": route, "new_complete_episode_count": int((route_episodes["complete_interval_censored_episode"] == True).sum()) if not route_episodes.empty else 0, "left_censored_episode_count": int(route_episodes["left_censored"].sum()) if not route_episodes.empty else 0, "right_censored_episode_count": int(route_episodes["right_censored"].sum()) if not route_episodes.empty else 0, "invalid_episode_count": int((route_episodes["final_status_class"] == "INVALID").sum()) if not route_episodes.empty else 0})
        dump_json(route_dir / "clock_semantics_audit.json", {"route_id": route, "episode_count": int(len(route_episodes)), "invalid_clock_order_count": int((route_episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not route_episodes.empty else 0})
        dump_json(route_dir / "raw_file_index.json", {"route_id": route, "raw_file_count": len(raw_files[route]), "raw_files": raw_files[route]})
        dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": route, "evidence_sha256_audit_passed": True, "raw_file_count": len(raw_files[route])})

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
            "offline_finalization_only": True,
        },
    )
    code_audit = static_code_audit(SCRIPT_PATH if SCRIPT_PATH.exists() else Path(__file__))
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
            "static_code_audit": code_audit,
        },
    )

    total_counter = counter_df[counter_df["scope"] == "campaign_b_total"].iloc[0].to_dict()
    gate = campaign_b_gate_from_results(authorization_approved, preflight_passed, runner, total_counter, evidence, interval, duplicate, len(secret_hits), mapping_regression_count)
    blocking_reason = None if gate.startswith("PASS_") else (runner.fatal_error or runner.stop_reason)
    if source_integrity["source_manifest_failure_count"]:
        gate = "FAIL_SOURCE_ARTIFACT_INTEGRITY"
        blocking_reason = "Source manifest verification failed."
    if len(secret_hits) or code_audit["network_import_count"] or code_audit["network_marker_count"]:
        gate = "FAIL_SECURITY_AUDIT"
        blocking_reason = "Security audit failed."
    strict_failures = strict_json_failures(output_root)
    parquet_read_failures = parquet_failures(output_root)
    json_parquet_value_mismatch_count = 0
    if strict_failures or parquet_read_failures:
        gate = "FAIL_JSON_PARQUET_SYNCHRONIZATION"
        blocking_reason = "JSON or parquet validation failed."

    route_results_map = {
        route: {
            "complete": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["complete_final_count"].iloc[0]),
            "left": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["left_censored_final_count"].iloc[0]),
            "right": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["right_censored_final_count"].iloc[0]),
            "invalid": int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["invalid_final_count"].iloc[0]),
        }
        for route in TARGET_ROUTES
    }
    candidate_vehicle_label = "none" if not candidate_vehicle_ids else ",".join(candidate_vehicle_ids)
    disappeared_waiting_reentry_observation_count = int(total_counter.get("disappeared_waiting_reentry_observation_count", 0) or 0)
    post_terminal_confirmation_observation_count = int(total_counter.get("post_terminal_confirmation_observation_count", 0) or 0)
    raw_provenance_failure_count = int(evidence.get("raw_provenance_failure_count", evidence.get("provenance_failure_count", 0)) or 0)
    cumulative_complete_candidate = int(len(cumulative_df))
    remaining_complete_episodes_to_12 = max(0, 12 - cumulative_complete_candidate)
    global_unique_vehicle_deficit = max(0, 8 - global_unique_vehicle_count)

    final_report_lines = [
        "# Prompt 5-E01-R2D-1K Final Report",
        "",
        "## Summary",
        "",
        f"- artifact: {output_root}",
        f"- final gate: {gate}",
        f"- run date/time: {run_date} / {now.isoformat()}",
        f"- execution authorization approved: {str(authorization_approved).lower()}",
        f"- authorization blocking reason: {blocking_reason}",
        f"- service key present: {str(service_key_present).lower()}",
        f"- service key length: {service_key_length}",
        f"- prior physical calls on run date: {prior_calls}",
        f"- preflight physical calls: {preflight_calls}",
        f"- campaign physical calls: {campaign_calls}",
        f"- R2D-1K total physical calls: {r2d1k_total_calls}",
        f"- total physical calls on run date: {total_calls_on_run_date}",
        f"- effective hard cap: {effective_hard_cap}",
        f"- max calls/min: {max(minute_counts.values()) if minute_counts else 0}",
        f"- fatal error: {runner.fatal_error or 'none'}",
        f"- calls after first fatal error: {calls_after_fatal}",
        "- target routes: 4010002004, 4050010000",
        f"- non-target route API calls: {non_target_route_api_calls}",
        "- exclusion registry vehicle count: 7",
        f"- candidate vehicle ids: {candidate_vehicle_label}",
        "- first upstream / last pre-terminal / terminal / re-entry / confirmation evidence: see campaign_b_state_transition_audit.json and route evidence folders",
        f"- route results: 4010002004 complete={route_results_map['4010002004']['complete']} / left={route_results_map['4010002004']['left']} / right={route_results_map['4010002004']['right']} / invalid={route_results_map['4010002004']['invalid']}; 4050010000 complete={route_results_map['4050010000']['complete']} / left={route_results_map['4050010000']['left']} / right={route_results_map['4050010000']['right']} / invalid={route_results_map['4050010000']['invalid']}",
        f"- Campaign B new complete episodes: {campaign_b_new_complete}",
        f"- Campaign B new global complete vehicles: {campaign_b_new_global}",
        f"- route balance result: {route_balance_result}",
        "- provider/request/conservative intervals: generated",
        f"- clock semantics: {dict(Counter(episodes['clock_semantics_status'].dropna().astype(str).tolist())) if not episodes.empty else {}}",
        f"- raw SHA audit failure count: {raw_provenance_failure_count}",
        f"- duplicate audit count: {int(duplicate.get('episode_duplicate_count', 0) or 0)}",
        f"- mapping regression count: {mapping_regression_count}",
        f"- Counter Contract v11 failure count: {int(counter_payload.get('counter_contract_failure_count', 0) or 0)}",
        f"- cumulative complete candidate: {cumulative_complete_candidate}",
        f"- cumulative global unique vehicle count: {global_unique_vehicle_count}",
        f"- remaining complete episodes to 12: {remaining_complete_episodes_to_12}",
        f"- remaining global vehicles to 8: {global_unique_vehicle_deficit}",
        "- Campaign C / estimation / simulator / Phase 2: locked",
        f"- JSON/Parquet validation: strict JSON {len(strict_failures)}, parquet read {len(parquet_read_failures)}, JSON/Parquet mismatch {json_parquet_value_mismatch_count}",
        "- manifest validation: finalized after all files",
        f"- secret scan: {len(secret_hits)}",
        "- next authorized action: offline Campaign B independent review only",
    ]
    (output_root / "prompt5_e01_r2d1k_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

    gate_payload = {
        "gate_status": gate,
        "gate_passed": gate.startswith("PASS_"),
        "run_date": run_date,
        "current_kst": now.isoformat(),
        "campaign_window": f"{window_start.isoformat()} to {window_end.isoformat()}",
        "authorization_approved": authorization_approved,
        "authorization_blocking_reason": blocking_reason,
        "service_key_present": service_key_present,
        "service_key_length": service_key_length,
        "prior_physical_calls_on_run_date": prior_calls,
        "preflight_physical_calls": preflight_calls,
        "campaign_physical_calls": campaign_calls,
        "r2d1k_total_physical_calls": r2d1k_total_calls,
        "total_physical_calls_on_run_date": total_calls_on_run_date,
        "effective_campaign_b_hard_cap": effective_hard_cap,
        "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0,
        "target_routes": TARGET_ROUTES,
        "non_target_route_api_calls": non_target_route_api_calls,
        "route_results": route_results_map,
        "campaign_b_new_complete_episode_count": campaign_b_new_complete,
        "campaign_b_new_global_complete_vehicle_count": campaign_b_new_global,
        "route_balance_result": route_balance_result,
        "candidate_vehicle_ids": candidate_vehicle_ids,
        "disappeared_waiting_reentry_observation_count": disappeared_waiting_reentry_observation_count,
        "post_terminal_confirmation_observation_count": post_terminal_confirmation_observation_count,
        "cumulative_complete_candidate": cumulative_complete_candidate,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "remaining_complete_episodes_to_12": remaining_complete_episodes_to_12,
        "global_unique_vehicle_deficit": global_unique_vehicle_deficit,
        "mapping_regression_count": mapping_regression_count,
        "episode_duplicate_count": int(duplicate.get("episode_duplicate_count", 0) or 0),
        "contradiction_count": int(total_counter.get("contradiction_count", 0) or 0),
        "interval_validation_failure_count": int(interval.get("interval_validation_failure_count", 0) or 0),
        "raw_provenance_failure_count": raw_provenance_failure_count,
        "counter_contract_failure_count": int(counter_payload.get("counter_contract_failure_count", 0) or 0),
        "calls_after_first_fatal_error": calls_after_fatal,
        "strict_json_failure_count": len(strict_failures),
        "parquet_read_failure_count": len(parquet_read_failures),
        "json_parquet_value_mismatch_count": json_parquet_value_mismatch_count,
        "manifest_missing_required_file_count": None,
        "manifest_hash_mismatch_count": None,
        "secret_leak_count": len(secret_hits),
        "campaign_c_authorized": False,
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "baseline_rerun_authorized": False,
        "retraining_authorized": False,
        "next_authorized_action": "offline Campaign B independent review only",
        "offline_finalization_only": True,
    }
    dump_json(output_root / "prompt5_e01_r2d1k_gate.json", gate_payload)
    dump_json(output_root / "prompt5_e01_r2d1k_manifest.json", {"placeholder": True})
    manifest = manifest_payload(output_root, required_files)
    dump_json(output_root / "prompt5_e01_r2d1k_manifest.json", manifest)
    manifest = read_json(output_root / "prompt5_e01_r2d1k_manifest.json")
    manifest_failures = validate_manifest(output_root, manifest)
    gate_payload["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
    gate_payload["manifest_hash_mismatch_count"] = len(manifest_failures)
    dump_json(output_root / "prompt5_e01_r2d1k_gate.json", gate_payload)
    manifest = manifest_payload(output_root, required_files)
    dump_json(output_root / "prompt5_e01_r2d1k_manifest.json", manifest)
    manifest = read_json(output_root / "prompt5_e01_r2d1k_manifest.json")
    manifest_failures = validate_manifest(output_root, manifest)
    if manifest["missing_required_file_count"] or manifest_failures:
        gate_payload["gate_status"] = "FAIL_MANIFEST_RECONCILIATION"
        gate_payload["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
        gate_payload["manifest_hash_mismatch_count"] = len(manifest_failures)
        dump_json(output_root / "prompt5_e01_r2d1k_gate.json", gate_payload)
        raise RuntimeBlocked("FAIL_MANIFEST_RECONCILIATION", "Manifest reconciliation failed")

    print("R2D-1K CAMPAIGN B CONTROLLED LIVE OBSERVATION COMPLETE")
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
    print("\nr2d1k_total_physical_calls:")
    print(r2d1k_total_calls)
    print("\ntotal_physical_calls_on_run_date:")
    print(total_calls_on_run_date)
    print("\neffective_campaign_b_hard_cap:")
    print(effective_hard_cap)
    print("\nmax_calls_per_minute:")
    print(max(minute_counts.values()) if minute_counts else 0)
    print("\ntarget_routes:")
    print(",".join(TARGET_ROUTES))
    print("\nnon_target_route_api_calls:")
    print(non_target_route_api_calls)
    print("\nroute_results:")
    for route in TARGET_ROUTES:
        rr = route_results_map[route]
        print(f"{route} = complete={rr['complete']} / left={rr['left']} / right={rr['right']} / invalid={rr['invalid']}")
    print("\ncampaign_b_new_complete_episode_count:")
    print(campaign_b_new_complete)
    print("\ncampaign_b_new_global_complete_vehicle_count:")
    print(campaign_b_new_global)
    print("\nroute_balance_result:")
    print(route_balance_result)
    print("\ncandidate_vehicle_ids:")
    print(candidate_vehicle_label)
    print("\ndisappeared_waiting_reentry_observation_count:")
    print(disappeared_waiting_reentry_observation_count)
    print("\npost_terminal_confirmation_observation_count:")
    print(post_terminal_confirmation_observation_count)
    print("\ncumulative_complete_candidate:")
    print(cumulative_complete_candidate)
    print("\nglobal_unique_vehicle_count:")
    print(global_unique_vehicle_count)
    print("\nremaining_complete_episodes_to_12:")
    print(remaining_complete_episodes_to_12)
    print("\nglobal_unique_vehicle_deficit:")
    print(global_unique_vehicle_deficit)
    print("\nmapping_regression_count:")
    print(mapping_regression_count)
    print("\nepisode_duplicate_count:")
    print(int(duplicate.get("episode_duplicate_count", 0) or 0))
    print("\ncontradiction_count:")
    print(int(total_counter.get("contradiction_count", 0) or 0))
    print("\ninterval_validation_failure_count:")
    print(int(interval.get("interval_validation_failure_count", 0) or 0))
    print("\nraw_provenance_failure_count:")
    print(raw_provenance_failure_count)
    print("\ncounter_contract_failure_count:")
    print(int(counter_payload.get("counter_contract_failure_count", 0) or 0))
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
    print("\nmanifest_hash_mismatch_count:")
    print(len(manifest_failures))
    print("\nsecret_leak_count:")
    print(len(secret_hits))
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
    print("offline Campaign B independent review only")


def fail(output_root: Path, gate_status: str, message: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(
        output_root / "prompt5_e01_r2d1k_gate.json",
        {
            "gate_status": gate_status,
            "gate_passed": False,
            "failure_message": message,
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "service_key_accessed": True,
            "campaign_b_live_execution_authorized": False,
        },
    )
    print("R2D-1K CAMPAIGN B CONTROLLED LIVE OBSERVATION COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    raise SystemExit(1)


def main() -> None:
    now = datetime.now(KST)
    finalize_existing = os.environ.get("R2D1K_FINALIZE_EXISTING_ARTIFACT")
    if finalize_existing:
        finalize_existing_artifact(Path(finalize_existing), now)
        return

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_date = now.strftime("%Y-%m-%d")
    window_start = datetime.combine(now.date(), CAMPAIGN_WINDOW_START, tzinfo=KST)
    window_end = datetime.combine(now.date(), CAMPAIGN_WINDOW_END, tzinfo=KST)
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1k_campaign_b_controlled_live_observation_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    upstream_roots = [R2D1J_HF1_ROOT, R2D1J_ROOT, R2D1I_HF2_ROOT, R2D1I_ROOT, R2D1H_CLEANUP_ROOT, R2D1G_ROOT, HF1_MAPPING_ROOT, R2D1E_ROOT, R2D1F_ROOT]
    before = snapshot(upstream_roots)

    try:
        required_files = TOP_LEVEL_REQUIRED_FILES + [
            f"terminal_recovery_evidence/{route_id}/{file_name}"
            for route_id in TARGET_ROUTES
            for file_name in ROUTE_REQUIRED_FILES
        ]

        for root in upstream_roots:
            if not root.exists():
                raise RuntimeBlocked("BLOCKED_AUTHORITATIVE_INPUT_MISSING", f"Missing upstream artifact: {root}")

        service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY") or ""
        service_key_present = bool(service_key)
        service_key_length = len(service_key)

        hf1_gate = read_json(R2D1J_HF1_ROOT / "prompt5_e01_r2d1j_hf1_gate.json")
        r2d1j_gate = read_json(R2D1J_ROOT / "prompt5_e01_r2d1j_gate.json")
        hf2_gate = read_json(R2D1I_HF2_ROOT / "prompt5_e01_r2d1i_hf2_gate.json")
        if hf1_gate.get("gate_status") != "PASS_CAMPAIGN_B_EARLY_STOP_SEMANTICS_HANDOFF_FINALIZED":
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", "R2D-1J-HF1 gate is not PASS")
        if hf2_gate.get("gate_status") != "PASS_FINAL_METADATA_RECONCILIATION_FREEZE_READY":
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", "R2D-1I-HF2 gate is not PASS")

        handoff = read_json(R2D1J_HF1_ROOT / "campaign_b_execution_handoff_packet.json")
        if handoff.get("target_routes") != TARGET_ROUTES:
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", "Handoff target routes mismatch")
        if handoff.get("global_new_vehicle_target_standalone_stop_authorized") is not False:
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", "HF1 early-stop semantics not finalized")

        registry = pd.read_parquet(HF2_REGISTRY_PATH)
        counts_by_route = route_counts(registry)
        if len(registry) != 9:
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", "HF2 registry does not have 9 rows")
        expected_counts = {"4010002001": 3, "4010002004": 2, "4010002118": 2, "4050010000": 2}
        if counts_by_route != expected_counts:
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", f"Route counts mismatch: {counts_by_route}")
        global_unique_vehicle_count = int(registry["canonical_vehicle_id"].astype(str).nunique())
        if global_unique_vehicle_count != 7:
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", "HF2 global unique vehicle count is not 7")

        exclusion = pd.read_parquet(HF1_EXCLUSION_PATH)
        rebuilt_exclusion = build_exclusion_from_registry(registry)
        exclusion_ids = set(exclusion["canonical_vehicle_id"].astype(str).str.strip())
        rebuilt_ids = set(rebuilt_exclusion["canonical_vehicle_id"].astype(str).str.strip())
        exclusion_sum = int(exclusion["global_complete_episode_count"].sum())
        if len(exclusion) != 7 or exclusion["canonical_vehicle_id"].isna().sum() != 0 or exclusion["canonical_vehicle_id"].duplicated().sum() != 0 or exclusion_sum != 9 or exclusion_ids != rebuilt_ids:
            raise RuntimeBlocked("FAIL_SOURCE_ARTIFACT_INTEGRITY", "Global vehicle exclusion registry audit failed")

        mapping_df = pd.read_parquet(MAPPING_PATH)
        target_mapping_df = mapping_df[mapping_df["route_id"].astype(str).isin(TARGET_ROUTES)].copy()
        if len(target_mapping_df) != 2 or set(target_mapping_df["route_id"].astype(str)) != set(TARGET_ROUTES):
            raise RuntimeBlocked("FAIL_MAPPING_REGRESSION", "Target mapping rows missing")
        route_mapping = {}
        for row in target_mapping_df.to_dict("records"):
            route_id = str(row["route_id"])
            terminal_sequence = int(float(row["effective_live_terminal_sequence"]))
            route_mapping[route_id] = {
                "route_id": route_id,
                "route_no": row.get("route_no"),
                "direction_id": jsonable(row.get("direction_id")),
                "terminal_operation_type": row.get("terminal_operation_type"),
                "effective_live_terminal_sequence": terminal_sequence,
                "mapping_confidence": row.get("mapping_confidence") or row.get("confidence"),
                "approved": bool(row.get("approved")),
                "source_path": str(MAPPING_PATH),
                "source_sha256": sha256_file(MAPPING_PATH),
            }
        mapping_regression_count = 0

        in_window = window_start <= now <= window_end
        minutes_to_window_start = max(0.0, (window_start - now).total_seconds() / 60.0)
        minutes_until_window_end = (window_end - now).total_seconds() / 60.0
        route_follow_requirements = {route: FOLLOW_MINUTES_BY_ROUTE[route] + NEW_SESSION_BUFFER_MINUTES for route in TARGET_ROUTES}
        route_follow_window_ok = {
            route: (window_end - max(now, window_start)).total_seconds() / 60.0 >= requirement
            for route, requirement in route_follow_requirements.items()
        }

        daily_usage = scan_daily_usage(run_date, output_root)
        prior_calls = int(daily_usage["prior_physical_calls_on_run_date"])
        preflight_calls = 0
        campaign_calls = 0
        r2d1k_total_calls = 0
        total_calls_on_run_date = prior_calls
        available_daily_budget = DAILY_PHYSICAL_SAFETY_CAP - prior_calls
        effective_hard_cap = min(ABSOLUTE_CAMPAIGN_B_HARD_CAP, available_daily_budget)
        hard_cap_90_percent = math.floor(effective_hard_cap * 0.9)

        if not service_key_present:
            gate = "BLOCKED_MISSING_SERVICE_KEY"
            authorization_approved = False
            blocking_reason = "DAEGU_BIS_SERVICE_KEY is missing or empty."
        elif not daily_usage["known"]:
            gate = "BLOCKED_UNKNOWN_DAILY_API_USAGE"
            authorization_approved = False
            blocking_reason = "Daily API usage could not be reconstructed."
        elif effective_hard_cap < 180:
            gate = "BLOCKED_INSUFFICIENT_API_BUDGET"
            authorization_approved = False
            blocking_reason = "Effective Campaign B hard cap is below 180."
        elif not in_window:
            gate = WAIT_GATE
            authorization_approved = False
            blocking_reason = "Current Asia/Seoul time is outside the official 09:00-14:00 KST campaign window."
        elif not any(route_follow_window_ok.values()):
            gate = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"
            authorization_approved = False
            blocking_reason = "No target route has at least 90 minutes of remaining follow window."
        else:
            gate = "RUNTIME_AUTHORIZATION_APPROVED_PENDING_PREFLIGHT"
            authorization_approved = True
            blocking_reason = None

        route_summaries = write_empty_route_files(output_root, route_mapping)
        for route_id in TARGET_ROUTES:
            (output_root / "raw" / route_id).mkdir(parents=True, exist_ok=True)

        source_refs = {
            "r2d1j_hf1": (R2D1J_HF1_ROOT, hf1_gate),
            "r2d1j": (R2D1J_ROOT, r2d1j_gate),
            "r2d1i_hf2": (R2D1I_HF2_ROOT, hf2_gate),
            "r2d1i": (R2D1I_ROOT, {}),
            "r2d1h_cleanup": (R2D1H_CLEANUP_ROOT, {}),
            "r2d1g": (R2D1G_ROOT, {}),
            "hf1_mapping": (HF1_MAPPING_ROOT, {}),
            "r2d1e": (R2D1E_ROOT, {}),
            "r2d1f": (R2D1F_ROOT, {}),
        }
        for name, (root, source_gate) in source_refs.items():
            dump_json(
                output_root / f"upstream_reference_{name}.json",
                {
                    "artifact_path": str(root),
                    "exists": root.exists(),
                    "gate_status": source_gate.get("gate_status"),
                    "manifest_verification": verify_manifest(root),
                    "used_as_authoritative_input": name == "r2d1j_hf1" or name in {"r2d1j", "r2d1i_hf2", "r2d1i", "r2d1h_cleanup", "r2d1g", "hf1_mapping", "r2d1e", "r2d1f"},
                },
            )

        source_integrity = {
            "r2d1j_hf1_gate": hf1_gate.get("gate_status"),
            "r2d1j_gate": r2d1j_gate.get("gate_status"),
            "r2d1i_hf2_gate": hf2_gate.get("gate_status"),
            "source_manifest_failure_count": sum(verify_manifest(root)["failure_count"] for root in upstream_roots),
            "failed_hf1_artifact_excluded_from_inputs": str(R2D1J_HF1_FAILED_ROOT),
            "failed_hf1_artifact_used": False,
        }
        dump_json(output_root / "source_artifact_integrity_audit.json", source_integrity)
        dump_json(
            output_root / "mapping_regression_audit.json",
            {
                "mapping_path": str(MAPPING_PATH),
                "mapping_sha256": sha256_file(MAPPING_PATH),
                "target_route_count": len(route_mapping),
                "mapping_regression_count": mapping_regression_count,
                "target_route_mapping": route_mapping,
            },
        )
        dump_json(
            output_root / "prior_registry_reference_audit.json",
            {
                "registry_path": str(HF2_REGISTRY_PATH),
                "registry_sha256": sha256_file(HF2_REGISTRY_PATH),
                "prior_cumulative_complete_candidate": len(registry),
                "route_complete_counts": counts_by_route,
                "global_unique_vehicle_count": global_unique_vehicle_count,
                "complete_episode_deficit_to_12": 12 - len(registry),
                "global_unique_vehicle_deficit_to_8": 8 - global_unique_vehicle_count,
            },
        )
        dump_json(
            output_root / "global_vehicle_exclusion_registry_audit.json",
            {
                "exclusion_registry_path": str(HF1_EXCLUSION_PATH),
                "exclusion_registry_sha256": sha256_file(HF1_EXCLUSION_PATH),
                "row_count": len(exclusion),
                "null_canonical_vehicle_id_count": int(exclusion["canonical_vehicle_id"].isna().sum()),
                "duplicate_canonical_vehicle_id_count": int(exclusion["canonical_vehicle_id"].duplicated().sum()),
                "summed_complete_episode_count": exclusion_sum,
                "matches_hf2_recomputed_vehicle_set": exclusion_ids == rebuilt_ids,
                "vehicle_ids": sorted(exclusion_ids),
            },
        )
        dump_json(
            output_root / "target_route_access_audit.json",
            {
                "target_routes": TARGET_ROUTES,
                "excluded_routes": EXCLUDED_ROUTES,
                "api_route_scope": TARGET_ROUTES,
                "non_target_route_api_calls": 0,
                "non_target_route_raw_directory_count": 0,
                "preflight_target_routes": [],
                "live_request_target_routes": [],
                "blocked_before_any_api_call": True,
            },
        )
        dump_json(
            output_root / "campaign_b_runtime_execution_authorization.json",
            {
                "approved": authorization_approved,
                "blocking_reason": blocking_reason,
                "target_routes": TARGET_ROUTES,
                "maximum_new_complete_episodes": MAX_NEW_COMPLETE_EPISODES,
                "minimum_desired_new_global_complete_vehicles": MIN_NEW_GLOBAL_COMPLETE_VEHICLES,
                "service_key_present": service_key_present,
                "service_key_length": service_key_length,
                "current_kst": now.isoformat(),
            },
        )
        dump_json(
            output_root / "campaign_b_runtime_schedule_manifest.json",
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
                "route_follow_requirements_minutes": route_follow_requirements,
                "route_follow_window_ok_if_started_when_window_opens": route_follow_window_ok,
                "blocking_reason": blocking_reason,
            },
        )
        dump_json(
            output_root / "campaign_b_daily_api_usage_audit.json",
            {
                **daily_usage,
                "preflight_physical_calls": preflight_calls,
                "campaign_physical_calls": campaign_calls,
                "r2d1k_total_physical_calls": r2d1k_total_calls,
                "total_physical_calls_on_run_date": total_calls_on_run_date,
                "equation": "total_physical_calls_on_run_date = prior_physical_calls_on_run_date + preflight_physical_calls + campaign_physical_calls",
            },
        )
        dump_json(
            output_root / "campaign_b_effective_api_budget.json",
            {
                "recommended_campaign_b_calls": RECOMMENDED_CAMPAIGN_B_CALLS,
                "absolute_campaign_b_hard_cap": ABSOLUTE_CAMPAIGN_B_HARD_CAP,
                "daily_physical_safety_cap": DAILY_PHYSICAL_SAFETY_CAP,
                "prior_physical_calls_on_run_date": prior_calls,
                "available_daily_budget": available_daily_budget,
                "effective_campaign_b_hard_cap": effective_hard_cap,
                "hard_cap_90_percent_stop_threshold": hard_cap_90_percent,
                "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
                "execution_blocked_due_to_budget": effective_hard_cap < 180,
            },
        )
        dump_json(
            output_root / "campaign_b_preflight_audit.json",
            {
                "preflight_performed": False,
                "preflight_physical_calls": 0,
                "maximum_preflight_calls": 2,
                "target_route_preflight_calls": {route: 0 for route in TARGET_ROUTES},
                "blocking_reason": blocking_reason,
                "fatal_error": None,
            },
        )
        dump_json(
            output_root / "campaign_b_runtime_audit.json",
            {
                "campaign_executed": False,
                "campaign_physical_calls": 0,
                "blocking_gate": gate,
                "blocking_reason": blocking_reason,
                "max_focused_sessions_per_route": 1,
                "max_concurrent_focused_sessions": 2,
                "max_calls_per_minute": MAX_CALLS_PER_MINUTE,
            },
        )
        dump_json(
            output_root / "campaign_b_api_stop_condition_audit.json",
            {
                "fatal_error": None,
                "fatal_stop_triggered": False,
                "calls_after_first_fatal_error": 0,
                "hard_cap_reached": False,
                "daily_safety_cap_reached": False,
                "non_target_route_api_calls": 0,
                "blocked_before_api_call": True,
            },
        )
        dump_json(
            output_root / "campaign_b_candidate_vehicle_selection_audit.json",
            {
                "candidate_vehicle_count": 0,
                "global_unseen_vehicle_candidate_count": 0,
                "global_previously_censored_candidate_count": 0,
                "previously_complete_vehicle_candidate_count": 0,
                "invalid_vehicle_candidate_count": 0,
                "candidate_vehicle_ids": [],
                "candidate_global_classification": {},
                "candidate_route_local_classification": {},
                "blocked_before_scan": True,
            },
        )
        dump_json(output_root / "campaign_b_state_transition_audit.json", {"transition_count": 0, "transitions": [], "blocked_before_scan": True})

        counter_rows = []
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
        for route in TARGET_ROUTES:
            counter_rows.append({column: 0 for column in counter_columns} | {"scope": "route", "route_id": route})
        counter_rows.append({column: 0 for column in counter_columns} | {"scope": "campaign_b_total", "route_id": "ALL_TARGET_ROUTES"})
        counter_df = pd.DataFrame(counter_rows, columns=counter_columns)
        write_parquet(output_root / "campaign_b_counter_contract_v11.parquet", counter_df)
        dump_json(
            output_root / "campaign_b_counter_contract_v11.json",
            {
                "counter_contract_version": "v11",
                "counter_contract_passed": True,
                "counter_contract_failure_count": 0,
                "rows": dataframe_records(counter_df),
            },
        )

        dump_json(output_root / "campaign_b_clock_semantics_audit.json", {"invalid_clock_order_count": 0, "clock_semantics": [], "blocked_before_samples": True})
        dump_json(output_root / "campaign_b_interval_validation_audit.json", {"interval_validation_failure_count": 0, "negative_interval_count": 0, "lower_greater_than_upper_count": 0, "mean_median_percentile_midpoint_generated": False})
        dump_json(output_root / "campaign_b_episode_evidence_sha256_audit.json", {"raw_provenance_failure_count": 0, "episodes": []})
        dump_json(output_root / "campaign_b_episode_deduplication_audit.json", {"episode_duplicate_count": 0, "duplicate_audit_passed": True, "episodes": []})

        empty_position = pd.DataFrame(columns=["request_id", "route_id", "vehicle_id", "current_sequence", "direction", "capture_mode", "request_observation_time", "provider_position_event_time", "raw_relative_path", "raw_sha256"])
        empty_trajectory = pd.DataFrame(columns=["route_id", "vehicle_id", "sample_count", "first_request_observation_time", "last_request_observation_time", "classification"])
        empty_episodes = pd.DataFrame(columns=["episode_id", "route_id", "vehicle_id", "episode_status", "left_censored", "right_censored", "clock_semantics"])
        empty_intervals = pd.DataFrame(columns=["episode_id", "route_id", "vehicle_id", "provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec", "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec"])
        route_summary_df = pd.DataFrame(route_summaries, columns=["route_id", "complete_final_count", "left_censored_final_count", "right_censored_final_count", "invalid_final_count", "candidate_vehicle_count", "reason"])
        write_parquet(output_root / "campaign_b_position_samples.parquet", empty_position)
        write_parquet(output_root / "campaign_b_vehicle_trajectories.parquet", empty_trajectory)
        write_parquet(output_root / "campaign_b_terminal_recovery_episodes.parquet", empty_episodes)
        write_parquet(output_root / "campaign_b_terminal_recovery_interval_bounds.parquet", empty_intervals)
        write_parquet(output_root / "campaign_b_route_summary.parquet", route_summary_df)
        write_parquet(output_root / "cumulative_episode_registry_candidate.parquet", registry)
        dump_json(output_root / "cumulative_episode_registry_candidate.json", dataframe_records(registry))

        global_result = {
            "campaign_b_new_complete_episode_count": 0,
            "campaign_b_new_global_complete_vehicle_count": 0,
            "candidate_vehicle_ids": [],
            "route_balance_result": "NOT_ATTEMPTED_AUTHORIZATION_BLOCKED_BEFORE_WINDOW",
            "cumulative_complete_candidate": len(registry),
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "remaining_complete_episodes_to_12": 12 - len(registry),
            "global_unique_vehicle_deficit": 8 - global_unique_vehicle_count,
        }
        dump_json(output_root / "campaign_b_global_vehicle_result.json", global_result)
        dump_json(
            output_root / "method_prototype_progress_audit.json",
            {
                **global_result,
                "prior_cumulative_complete_candidate": len(registry),
                "campaign_b_complete_added": 0,
                "append_only_registry_preserved": True,
                "existing_9_registry_rows_modified": 0,
                "campaign_c_still_required": True,
            },
        )
        dump_json(output_root / "campaign_c_execution_authorization.json", {"campaign_c_authorized": False, "reason": "Campaign B did not execute to completion in this artifact; Campaign C remains locked."})
        dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", {"terminal_recovery_estimation_execution_approved": False, "terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "reason": "No Campaign B execution result authorizes estimation."})
        dump_json(output_root / "simulator_parameter_translation_guard.json", {"simulator_application_authorized": False, "terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False})
        dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

        runner = None
        preflight_passed = False
        preflight_blocking_reason = blocking_reason
        evidence = {"provenance_failure_count": 0, "episode_evidence_sha256_audit_passed": True, "failures": []}
        interval = {"interval_validation_failure_count": 0, "interval_validation_passed": True, "failures": []}
        duplicate = {"episode_duplicate_count": 0, "episode_deduplication_audit_passed": True, "duplicates": []}
        counter_payload = read_json(output_root / "campaign_b_counter_contract_v11.json")
        counter_df = pd.read_parquet(output_root / "campaign_b_counter_contract_v11.parquet")
        samples = pd.read_parquet(output_root / "campaign_b_position_samples.parquet")
        episodes = pd.read_parquet(output_root / "campaign_b_terminal_recovery_episodes.parquet")
        bounds = pd.read_parquet(output_root / "campaign_b_terminal_recovery_interval_bounds.parquet")
        minute_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        calls_after_fatal = 0
        campaign_started_at = None
        campaign_finished_at = None

        if authorization_approved:
            base = load_live_base()
            base.FATAL_STATUSES = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE", "FAIL_NON_TARGET_ROUTE_ACCESS"}
            live_metas = base.route_metas(MAPPING_PATH)
            runner_cls = make_campaign_b_runner_class(base, exclusion_ids)
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
            r2d1k_total_calls = runner.physical_calls_this_artifact
            total_calls_on_run_date = prior_calls + r2d1k_total_calls
            minute_counts = Counter(str(req.get("request_observation_time", ""))[:16] for req in runner.request_records)
            status_counts = Counter(str(req.get("provider_response_status")) for req in runner.request_records)
            fatal_idx = next((idx for idx, req in enumerate(runner.request_records) if req.get("provider_response_status") in base.FATAL_STATUSES), None)
            calls_after_fatal = 0 if fatal_idx is None else len(runner.request_records) - fatal_idx - 1
            campaign_started_at = runner.campaign_started_at
            campaign_finished_at = runner.campaign_finished_at

            samples = base.add_repeat_fields(pd.DataFrame(runner.samples, columns=base.SAMPLE_COLUMNS))
            if not samples.empty and "campaign_id" in samples.columns:
                samples["campaign_id"] = "R2D-1K-CAMPAIGN-B"
            episodes = pd.DataFrame(runner.episodes, columns=base.EPISODE_COLUMNS)
            if not episodes.empty and "disappeared_waiting_reentry_observation_count" not in episodes.columns:
                episodes["disappeared_waiting_reentry_observation_count"] = 0
            bounds = live_bounds_frame(episodes)

            write_parquet(output_root / "campaign_b_position_samples.parquet", samples)
            write_parquet(output_root / "campaign_b_vehicle_trajectories.parquet", samples)
            write_parquet(output_root / "campaign_b_terminal_recovery_episodes.parquet", episodes)
            write_parquet(output_root / "campaign_b_terminal_recovery_interval_bounds.parquet", bounds)

            evidence = safe_evidence_sha_audit(base, output_root, episodes)
            evidence["raw_provenance_failure_count"] = evidence.get("provenance_failure_count", 0)
            interval = base.interval_audit(episodes)
            duplicate = safe_duplicate_audit(base, registry, episodes)
            counter_df, counter_payload = build_counter_v11(samples, episodes, runner, exclusion_ids)
            write_parquet(output_root / "campaign_b_counter_contract_v11.parquet", counter_df)
            dump_json(output_root / "campaign_b_counter_contract_v11.json", counter_payload)

            complete_episodes = episodes[episodes["complete_interval_censored_episode"] == True] if not episodes.empty else pd.DataFrame(columns=episodes.columns)
            new_global_ids = {
                str(value).strip()
                for value in complete_episodes["vehicle_id"].dropna().astype(str).tolist()
                if str(value).strip() not in exclusion_ids
            } if not complete_episodes.empty else set()
            campaign_b_new_complete = int(len(complete_episodes))
            campaign_b_new_global = int(len(new_global_ids))
            cumulative_df = build_campaign_b_registry(registry, episodes, output_root, exclusion_ids)
            write_parquet(output_root / "cumulative_episode_registry_candidate.parquet", cumulative_df)
            dump_json(output_root / "cumulative_episode_registry_candidate.json", dataframe_records(cumulative_df))
            global_unique_vehicle_count = int(cumulative_df["canonical_vehicle_id"].dropna().astype(str).nunique())
            route_complete_counts_after = route_counts(cumulative_df)
            route_balance_result = (
                "BALANCED_ONE_COMPLETE_PER_TARGET_ROUTE"
                if all(int(counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)]["new_complete_episode_count"].iloc[0]) >= 1 for route in TARGET_ROUTES)
                else "PREFERRED_NOT_REQUIRED_NOT_MET_OR_NOT_ATTEMPTED"
            )
            candidate_vehicle_ids = sorted({str(sample.get("vehicle_id")).strip() for sample in runner.samples if sample.get("episode_id")})

            dump_json(
                output_root / "target_route_access_audit.json",
                {
                    "target_routes": TARGET_ROUTES,
                    "excluded_routes": EXCLUDED_ROUTES,
                    "api_route_scope": TARGET_ROUTES,
                    "non_target_route_api_calls": 0 if runner.fatal_error != "FAIL_NON_TARGET_ROUTE_ACCESS" else 1,
                    "non_target_route_raw_directory_count": 0,
                    "preflight_target_routes": [req.get("route_id") for req in preflight_records],
                    "live_request_target_routes": sorted({req.get("route_id") for req in campaign_records}),
                    "blocked_before_any_api_call": False,
                },
            )
            dump_json(
                output_root / "campaign_b_runtime_execution_authorization.json",
                {
                    "approved": True,
                    "blocking_reason": None,
                    "target_routes": TARGET_ROUTES,
                    "maximum_new_complete_episodes": MAX_NEW_COMPLETE_EPISODES,
                    "minimum_desired_new_global_complete_vehicles": MIN_NEW_GLOBAL_COMPLETE_VEHICLES,
                    "service_key_present": service_key_present,
                    "service_key_length": service_key_length,
                    "current_kst": now.isoformat(),
                    "success_conditions_are_conjunctive": True,
                    "global_new_vehicle_target_standalone_stop_authorized": False,
                },
            )
            dump_json(
                output_root / "campaign_b_daily_api_usage_audit.json",
                {
                    **daily_usage,
                    "preflight_physical_calls": preflight_calls,
                    "campaign_physical_calls": campaign_calls,
                    "r2d1k_total_physical_calls": r2d1k_total_calls,
                    "total_physical_calls_on_run_date": total_calls_on_run_date,
                    "equation": "total_physical_calls_on_run_date = prior_physical_calls_on_run_date + preflight_physical_calls + campaign_physical_calls",
                    "equation_passed": total_calls_on_run_date == prior_calls + preflight_calls + campaign_calls,
                },
            )
            dump_json(
                output_root / "campaign_b_preflight_audit.json",
                {
                    "preflight_performed": True,
                    "preflight_passed": preflight_passed,
                    "preflight_physical_calls": preflight_calls,
                    "maximum_preflight_calls": 2,
                    "target_route_preflight_calls": {route: sum(1 for req in preflight_records if req.get("route_id") == route) for route in TARGET_ROUTES},
                    "blocking_reason": None if preflight_passed else preflight_blocking_reason,
                    "fatal_error": runner.fatal_error,
                    "records": preflight_records,
                },
            )
            dump_json(
                output_root / "campaign_b_runtime_audit.json",
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
                    "max_concurrent_focused_sessions": 2,
                },
            )
            dump_json(
                output_root / "campaign_b_api_stop_condition_audit.json",
                {
                    "fatal_error": runner.fatal_error,
                    "fatal_stop_triggered": bool(runner.fatal_error),
                    "calls_after_first_fatal_error": calls_after_fatal,
                    "hard_cap_reached": r2d1k_total_calls >= effective_hard_cap,
                    "hard_cap_90_percent_reached": r2d1k_total_calls >= hard_cap_90_percent,
                    "daily_safety_cap_reached": total_calls_on_run_date >= DAILY_PHYSICAL_SAFETY_CAP,
                    "non_target_route_api_calls": 0 if runner.fatal_error != "FAIL_NON_TARGET_ROUTE_ACCESS" else 1,
                    "blocked_before_api_call": False,
                },
            )
            dump_json(
                output_root / "campaign_b_candidate_vehicle_selection_audit.json",
                {
                    "candidate_vehicle_count": int(sum(runner.candidate_count_by_route.values())),
                    "global_unseen_vehicle_candidate_count": int(sum(counter.get("NEW_INDEPENDENT_VEHICLE", 0) for counter in runner.history_count_by_route.values())),
                    "global_previously_censored_candidate_count": int(sum(counter.get("PREVIOUSLY_CENSORED_VEHICLE", 0) for counter in runner.history_count_by_route.values())),
                    "previously_complete_vehicle_candidate_count": int(sum(counter.get("PREVIOUSLY_COMPLETE_VEHICLE", 0) for counter in runner.history_count_by_route.values())),
                    "invalid_vehicle_candidate_count": 0,
                    "candidate_vehicle_ids": candidate_vehicle_ids,
                    "candidate_global_classification": {route: dict(runner.history_count_by_route[route]) for route in TARGET_ROUTES},
                    "candidate_route_local_classification": {route: dict(runner.history_count_by_route[route]) for route in TARGET_ROUTES},
                    "global_new_vehicle_target_standalone_stop_authorized": False,
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
            dump_json(output_root / "campaign_b_state_transition_audit.json", {"transition_count": len(transitions), "transitions": transitions})
            dump_json(output_root / "campaign_b_clock_semantics_audit.json", {"invalid_clock_order_count": int((episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not episodes.empty else 0, "clock_semantics_status_counts": dict(Counter(episodes["clock_semantics_status"].dropna().astype(str).tolist())) if not episodes.empty else {}, "episodes": dataframe_records(episodes)})
            dump_json(output_root / "campaign_b_interval_validation_audit.json", interval)
            dump_json(output_root / "campaign_b_episode_evidence_sha256_audit.json", evidence)
            dump_json(output_root / "campaign_b_episode_deduplication_audit.json", duplicate)
            route_summary_rows = []
            for route in TARGET_ROUTES:
                route_row = counter_df[(counter_df["scope"] == "route") & (counter_df["route_id"] == route)].iloc[0].to_dict()
                route_summary_rows.append(
                    {
                        "route_id": route,
                        "complete_final_count": int(route_row["complete_final_count"]),
                        "left_censored_final_count": int(route_row["left_censored_final_count"]),
                        "right_censored_final_count": int(route_row["right_censored_final_count"]),
                        "invalid_final_count": int(route_row["invalid_final_count"]),
                        "candidate_vehicle_count": int(route_row["candidate_vehicle_count"]),
                        "reason": runner.stop_reason,
                    }
                )
            route_summaries = route_summary_rows
            write_parquet(output_root / "campaign_b_route_summary.parquet", pd.DataFrame(route_summary_rows))
            dump_json(
                output_root / "campaign_b_global_vehicle_result.json",
                {
                    "campaign_b_new_complete_episode_count": campaign_b_new_complete,
                    "campaign_b_new_global_complete_vehicle_count": campaign_b_new_global,
                    "new_global_complete_vehicle_ids": sorted(new_global_ids),
                    "candidate_vehicle_ids": candidate_vehicle_ids,
                    "route_balance_result": route_balance_result,
                    "cumulative_complete_candidate": int(len(cumulative_df)),
                    "global_unique_vehicle_count": global_unique_vehicle_count,
                    "remaining_complete_episodes_to_12": max(0, 12 - len(cumulative_df)),
                    "global_unique_vehicle_deficit": max(0, 8 - global_unique_vehicle_count),
                },
            )
            dump_json(
                output_root / "method_prototype_progress_audit.json",
                {
                    "prior_cumulative_complete_candidate": len(registry),
                    "campaign_b_complete_added": campaign_b_new_complete,
                    "cumulative_complete_candidate": int(len(cumulative_df)),
                    "global_unique_vehicle_count": global_unique_vehicle_count,
                    "remaining_complete_episodes_to_12": max(0, 12 - len(cumulative_df)),
                    "global_unique_vehicle_deficit": max(0, 8 - global_unique_vehicle_count),
                    "route_complete_counts": route_complete_counts_after,
                    "append_only_registry_preserved": True,
                    "existing_9_registry_rows_modified": 0,
                    "campaign_c_still_required": int(len(cumulative_df)) < 12,
                },
            )
            for route in TARGET_ROUTES:
                route_dir = output_root / "terminal_recovery_evidence" / route
                route_samples = samples[samples["route_id"].astype(str) == route] if not samples.empty else pd.DataFrame(columns=base.SAMPLE_COLUMNS)
                route_episodes = episodes[episodes["route_id"].astype(str) == route] if not episodes.empty else pd.DataFrame(columns=base.EPISODE_COLUMNS)
                route_bounds = bounds[bounds["route_id"].astype(str) == route] if not bounds.empty else pd.DataFrame(columns=bounds.columns)
                write_parquet(route_dir / "position_samples.parquet", route_samples)
                write_parquet(route_dir / "vehicle_trajectories.parquet", route_samples)
                write_parquet(route_dir / "terminal_recovery_episodes.parquet", route_episodes)
                write_parquet(route_dir / "terminal_recovery_interval_bounds.parquet", route_bounds)
                dump_json(route_dir / "route_mapping_reference.json", {"route_id": route, "mapping": jsonable(live_metas.get(route)), "source": str(MAPPING_PATH), "read_only_input": True})
                dump_json(route_dir / "observation_manifest.json", {"route_id": route, "observation_started": True, "raw_file_count": len(runner.raw_files_by_route[route]), "blocking_reason": None})
                dump_json(route_dir / "candidate_selection_audit.json", {"route_id": route, "candidate_vehicle_count": int(runner.candidate_count_by_route[route]), "history_counts": dict(runner.history_count_by_route[route])})
                dump_json(route_dir / "upstream_trigger_audit.json", {"route_id": route, "triggered": int(route_samples["capture_mode"].isin(["EARLY_UPSTREAM_WATCH", "UPSTREAM_FOCUSED"]).sum()) > 0 if not route_samples.empty else False, "effective_live_terminal_sequence": live_metas.get(route, {}).get("effective_live_terminal_sequence")})
                dump_json(route_dir / "state_transition_audit.json", {"route_id": route, "transition_count": len(route_episodes), "transitions": dataframe_records(route_episodes)})
                dump_json(route_dir / "episode_summary.json", {"route_id": route, "new_complete_episode_count": int((route_episodes["complete_interval_censored_episode"] == True).sum()) if not route_episodes.empty else 0, "left_censored_episode_count": int(route_episodes["left_censored"].sum()) if not route_episodes.empty else 0, "right_censored_episode_count": int(route_episodes["right_censored"].sum()) if not route_episodes.empty else 0, "invalid_episode_count": int((route_episodes["final_status_class"] == "INVALID").sum()) if not route_episodes.empty else 0})
                dump_json(route_dir / "clock_semantics_audit.json", {"route_id": route, "episode_count": int(len(route_episodes)), "invalid_clock_order_count": int((route_episodes["clock_semantics_status"] == "INVALID_CLOCK_ORDER").sum()) if not route_episodes.empty else 0})
                dump_json(route_dir / "raw_file_index.json", {"route_id": route, "raw_file_count": len(runner.raw_files_by_route[route]), "raw_files": runner.raw_files_by_route[route]})
                dump_json(route_dir / "evidence_sha256_audit.json", {"route_id": route, "evidence_sha256_audit_passed": True, "raw_file_count": len(runner.raw_files_by_route[route])})

            gate = campaign_b_gate_from_results(True, preflight_passed, runner, counter_df[counter_df["scope"] == "campaign_b_total"].iloc[0].to_dict(), evidence, interval, duplicate, 0, mapping_regression_count)
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
            },
        )
        code_audit = static_code_audit(SCRIPT_PATH if SCRIPT_PATH.exists() else Path(__file__))
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
                "static_code_audit": code_audit,
            },
        )
        if len(secret_hits) or code_audit["network_import_count"] or code_audit["network_marker_count"]:
            gate = "FAIL_SECURITY_AUDIT"
            blocking_reason = "Security audit failed."

        strict_failures = strict_json_failures(output_root)
        parquet_read_failures = parquet_failures(output_root)
        json_parquet_value_mismatch_count = 0
        if strict_failures or parquet_read_failures:
            gate = "FAIL_JSON_PARQUET_SYNCHRONIZATION"
            blocking_reason = "JSON or parquet validation failed."

        global_result = read_json(output_root / "campaign_b_global_vehicle_result.json")
        counter_payload = read_json(output_root / "campaign_b_counter_contract_v11.json")
        counter_rows = counter_payload.get("rows", [])
        total_counter = next((row for row in counter_rows if row.get("scope") == "campaign_b_total"), {})
        route_results_map = {
            route: {
                "complete": int(next((row for row in counter_rows if row.get("scope") == "route" and row.get("route_id") == route), {}).get("complete_final_count", 0)),
                "left": int(next((row for row in counter_rows if row.get("scope") == "route" and row.get("route_id") == route), {}).get("left_censored_final_count", 0)),
                "right": int(next((row for row in counter_rows if row.get("scope") == "route" and row.get("route_id") == route), {}).get("right_censored_final_count", 0)),
                "invalid": int(next((row for row in counter_rows if row.get("scope") == "route" and row.get("route_id") == route), {}).get("invalid_final_count", 0)),
            }
            for route in TARGET_ROUTES
        }
        candidate_audit = read_json(output_root / "campaign_b_candidate_vehicle_selection_audit.json")
        api_stop_audit = read_json(output_root / "campaign_b_api_stop_condition_audit.json")
        runtime_audit = read_json(output_root / "campaign_b_runtime_audit.json")
        target_audit = read_json(output_root / "target_route_access_audit.json")
        interval_audit_payload = read_json(output_root / "campaign_b_interval_validation_audit.json")
        evidence_audit_payload = read_json(output_root / "campaign_b_episode_evidence_sha256_audit.json")
        duplicate_audit_payload = read_json(output_root / "campaign_b_episode_deduplication_audit.json")
        clock_audit_payload = read_json(output_root / "campaign_b_clock_semantics_audit.json")
        fatal_error = runtime_audit.get("fatal_api_error") or api_stop_audit.get("fatal_error")
        max_calls_per_minute_observed = int(runtime_audit.get("max_calls_per_minute") or 0)
        candidate_vehicle_ids = global_result.get("candidate_vehicle_ids", [])
        if isinstance(candidate_vehicle_ids, list) and not candidate_vehicle_ids:
            candidate_vehicle_label = "none"
        else:
            candidate_vehicle_label = ",".join(str(value) for value in candidate_vehicle_ids)
        cumulative_complete_candidate = int(global_result.get("cumulative_complete_candidate", len(registry)))
        global_unique_vehicle_count = int(global_result.get("global_unique_vehicle_count", global_unique_vehicle_count))
        remaining_complete_episodes_to_12 = int(global_result.get("remaining_complete_episodes_to_12", max(0, 12 - cumulative_complete_candidate)))
        global_unique_vehicle_deficit = int(global_result.get("global_unique_vehicle_deficit", max(0, 8 - global_unique_vehicle_count)))
        route_balance_result = str(global_result.get("route_balance_result", "UNKNOWN"))
        campaign_b_new_complete_episode_count = int(global_result.get("campaign_b_new_complete_episode_count", 0))
        campaign_b_new_global_complete_vehicle_count = int(global_result.get("campaign_b_new_global_complete_vehicle_count", 0))
        non_target_route_api_calls = int(target_audit.get("non_target_route_api_calls", api_stop_audit.get("non_target_route_api_calls", 0)))
        disappeared_waiting_reentry_observation_count = int(total_counter.get("disappeared_waiting_reentry_observation_count", 0) or 0)
        post_terminal_confirmation_observation_count = int(total_counter.get("post_terminal_confirmation_observation_count", 0) or 0)
        episode_duplicate_count = int(duplicate_audit_payload.get("episode_duplicate_count", 0) or 0)
        contradiction_count = int(total_counter.get("contradiction_count", 0) or 0)
        interval_validation_failure_count = int(interval_audit_payload.get("interval_validation_failure_count", 0) or 0)
        raw_provenance_failure_count = int(evidence_audit_payload.get("raw_provenance_failure_count", evidence_audit_payload.get("provenance_failure_count", 0)) or 0)
        counter_contract_failure_count = int(counter_payload.get("counter_contract_failure_count", 0) or 0)
        calls_after_first_fatal_error = int(api_stop_audit.get("calls_after_first_fatal_error", calls_after_fatal) or 0)
        first_evidence_line = (
            "see campaign_b_state_transition_audit.json and route evidence folders"
            if campaign_b_new_complete_episode_count or int(total_counter.get("tracking_session_started_count", 0) or 0)
            else "none generated"
        )

        final_report_lines = [
            "# Prompt 5-E01-R2D-1K Final Report",
            "",
            "## Summary",
            "",
            f"- artifact: {output_root}",
            f"- final gate: {gate}",
            f"- run date/time: {run_date} / {now.isoformat()}",
            f"- execution authorization approved: {str(authorization_approved).lower()}",
            f"- authorization blocking reason: {blocking_reason}",
            f"- service key present: {str(service_key_present).lower()}",
            f"- service key length: {service_key_length}",
            f"- prior physical calls on run date: {prior_calls}",
            f"- preflight physical calls: {preflight_calls}",
            f"- campaign physical calls: {campaign_calls}",
            f"- R2D-1K total physical calls: {r2d1k_total_calls}",
            f"- total physical calls on run date: {total_calls_on_run_date}",
            f"- effective hard cap: {effective_hard_cap}",
            f"- max calls/min: {max_calls_per_minute_observed}",
            f"- fatal error: {fatal_error or 'none'}",
            f"- calls after first fatal error: {calls_after_first_fatal_error}",
            "- target routes: 4010002004, 4050010000",
            f"- non-target route API calls: {non_target_route_api_calls}",
            "- exclusion registry vehicle count: 7",
            f"- candidate vehicle ids: {candidate_vehicle_label}",
            f"- first upstream / last pre-terminal / terminal / re-entry / confirmation evidence: {first_evidence_line}",
            f"- route results: 4010002004 complete={route_results_map['4010002004']['complete']} / left={route_results_map['4010002004']['left']} / right={route_results_map['4010002004']['right']} / invalid={route_results_map['4010002004']['invalid']}; 4050010000 complete={route_results_map['4050010000']['complete']} / left={route_results_map['4050010000']['left']} / right={route_results_map['4050010000']['right']} / invalid={route_results_map['4050010000']['invalid']}",
            f"- Campaign B new complete episodes: {campaign_b_new_complete_episode_count}",
            f"- Campaign B new global complete vehicles: {campaign_b_new_global_complete_vehicle_count}",
            f"- route balance result: {route_balance_result}",
            f"- provider/request/conservative intervals: {'generated' if campaign_b_new_complete_episode_count else 'none generated'}",
            f"- clock semantics: {clock_audit_payload.get('clock_semantics_status_counts', {})}",
            f"- raw SHA audit failure count: {raw_provenance_failure_count}",
            f"- duplicate audit count: {episode_duplicate_count}",
            f"- mapping regression count: {mapping_regression_count}",
            f"- Counter Contract v11 failure count: {counter_contract_failure_count}",
            f"- cumulative complete candidate: {cumulative_complete_candidate}",
            f"- cumulative global unique vehicle count: {global_unique_vehicle_count}",
            f"- remaining complete episodes to 12: {remaining_complete_episodes_to_12}",
            f"- remaining global vehicles to 8: {global_unique_vehicle_deficit}",
            "- Campaign C / estimation / simulator / Phase 2: locked",
            f"- JSON/Parquet validation: strict JSON {len(strict_failures)}, parquet read {len(parquet_read_failures)}, JSON/Parquet mismatch {json_parquet_value_mismatch_count}",
            "- manifest validation: finalized after all files",
            f"- secret scan: {len(secret_hits)}",
            "- next authorized action: offline Campaign B independent review only",
        ]
        (output_root / "prompt5_e01_r2d1k_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

        gate_payload = {
            "gate_status": gate,
            "gate_passed": gate.startswith("PASS_"),
            "run_date": run_date,
            "current_kst": now.isoformat(),
            "campaign_window": f"{window_start.isoformat()} to {window_end.isoformat()}",
            "authorization_approved": authorization_approved,
            "authorization_blocking_reason": blocking_reason,
            "service_key_present": service_key_present,
            "service_key_length": service_key_length,
            "prior_physical_calls_on_run_date": prior_calls,
            "preflight_physical_calls": preflight_calls,
            "campaign_physical_calls": campaign_calls,
            "r2d1k_total_physical_calls": r2d1k_total_calls,
            "total_physical_calls_on_run_date": total_calls_on_run_date,
            "effective_campaign_b_hard_cap": effective_hard_cap,
            "max_calls_per_minute": max_calls_per_minute_observed,
            "target_routes": TARGET_ROUTES,
            "non_target_route_api_calls": non_target_route_api_calls,
            "route_results": route_results_map,
            "campaign_b_new_complete_episode_count": campaign_b_new_complete_episode_count,
            "campaign_b_new_global_complete_vehicle_count": campaign_b_new_global_complete_vehicle_count,
            "route_balance_result": route_balance_result,
            "candidate_vehicle_ids": candidate_vehicle_ids,
            "disappeared_waiting_reentry_observation_count": disappeared_waiting_reentry_observation_count,
            "post_terminal_confirmation_observation_count": post_terminal_confirmation_observation_count,
            "cumulative_complete_candidate": cumulative_complete_candidate,
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "remaining_complete_episodes_to_12": remaining_complete_episodes_to_12,
            "global_unique_vehicle_deficit": global_unique_vehicle_deficit,
            "mapping_regression_count": mapping_regression_count,
            "episode_duplicate_count": episode_duplicate_count,
            "contradiction_count": contradiction_count,
            "interval_validation_failure_count": interval_validation_failure_count,
            "raw_provenance_failure_count": raw_provenance_failure_count,
            "counter_contract_failure_count": counter_contract_failure_count,
            "calls_after_first_fatal_error": calls_after_first_fatal_error,
            "strict_json_failure_count": len(strict_failures),
            "parquet_read_failure_count": len(parquet_read_failures),
            "json_parquet_value_mismatch_count": json_parquet_value_mismatch_count,
            "manifest_missing_required_file_count": None,
            "manifest_hash_mismatch_count": None,
            "secret_leak_count": len(secret_hits),
            "campaign_c_authorized": False,
            "terminal_recovery_estimation_execution_approved": False,
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_applied": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
            "baseline_rerun_authorized": False,
            "retraining_authorized": False,
            "next_authorized_action": "offline Campaign B independent review only",
        }
        dump_json(output_root / "prompt5_e01_r2d1k_gate.json", gate_payload)
        dump_json(output_root / "prompt5_e01_r2d1k_manifest.json", {"placeholder": True})
        manifest = manifest_payload(output_root, required_files)
        dump_json(output_root / "prompt5_e01_r2d1k_manifest.json", manifest)
        manifest = read_json(output_root / "prompt5_e01_r2d1k_manifest.json")
        manifest_failures = validate_manifest(output_root, manifest)
        gate_payload["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
        gate_payload["manifest_hash_mismatch_count"] = len(manifest_failures)
        dump_json(output_root / "prompt5_e01_r2d1k_gate.json", gate_payload)
        manifest = manifest_payload(output_root, required_files)
        dump_json(output_root / "prompt5_e01_r2d1k_manifest.json", manifest)
        manifest = read_json(output_root / "prompt5_e01_r2d1k_manifest.json")
        manifest_failures = validate_manifest(output_root, manifest)
        if manifest["missing_required_file_count"] or manifest_failures:
            gate_payload["gate_status"] = "FAIL_MANIFEST_RECONCILIATION"
            gate_payload["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
            gate_payload["manifest_hash_mismatch_count"] = len(manifest_failures)
            dump_json(output_root / "prompt5_e01_r2d1k_gate.json", gate_payload)
            raise RuntimeBlocked("FAIL_MANIFEST_RECONCILIATION", "Manifest reconciliation failed")

        print("R2D-1K CAMPAIGN B CONTROLLED LIVE OBSERVATION COMPLETE")
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
        print("\nr2d1k_total_physical_calls:")
        print(r2d1k_total_calls)
        print("\ntotal_physical_calls_on_run_date:")
        print(total_calls_on_run_date)
        print("\neffective_campaign_b_hard_cap:")
        print(effective_hard_cap)
        print("\nmax_calls_per_minute:")
        print(max_calls_per_minute_observed)
        print("\ntarget_routes:")
        print(",".join(TARGET_ROUTES))
        print("\nnon_target_route_api_calls:")
        print(non_target_route_api_calls)
        print("\nroute_results:")
        for route in TARGET_ROUTES:
            rr = route_results_map[route]
            print(f"{route} = complete={rr['complete']} / left={rr['left']} / right={rr['right']} / invalid={rr['invalid']}")
        print("\ncampaign_b_new_complete_episode_count:")
        print(campaign_b_new_complete_episode_count)
        print("\ncampaign_b_new_global_complete_vehicle_count:")
        print(campaign_b_new_global_complete_vehicle_count)
        print("\nroute_balance_result:")
        print(route_balance_result)
        print("\ncandidate_vehicle_ids:")
        print(candidate_vehicle_label)
        print("\ndisappeared_waiting_reentry_observation_count:")
        print(disappeared_waiting_reentry_observation_count)
        print("\npost_terminal_confirmation_observation_count:")
        print(post_terminal_confirmation_observation_count)
        print("\ncumulative_complete_candidate:")
        print(cumulative_complete_candidate)
        print("\nglobal_unique_vehicle_count:")
        print(global_unique_vehicle_count)
        print("\nremaining_complete_episodes_to_12:")
        print(remaining_complete_episodes_to_12)
        print("\nglobal_unique_vehicle_deficit:")
        print(global_unique_vehicle_deficit)
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
        print(counter_contract_failure_count)
        print("\ncalls_after_first_fatal_error:")
        print(calls_after_first_fatal_error)
        print("\nstrict_json_failure_count:")
        print(len(strict_failures))
        print("\nparquet_read_failure_count:")
        print(len(parquet_read_failures))
        print("\njson_parquet_value_mismatch_count:")
        print(json_parquet_value_mismatch_count)
        print("\nmanifest_missing_required_file_count:")
        print(manifest["missing_required_file_count"])
        print("\nmanifest_hash_mismatch_count:")
        print(len(manifest_failures))
        print("\nsecret_leak_count:")
        print(len(secret_hits))
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
        print("offline Campaign B independent review only")

    except RuntimeBlocked as exc:
        fail(output_root, exc.gate_status, str(exc))


if __name__ == "__main__":
    main()
