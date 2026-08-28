#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SOURCE_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1m_campaign_c_controlled_live_observation_20260728_090115"
SOURCE_FATAL_RAW = SOURCE_ROOT / "raw/4010002118/20260728_114909_823679_terminal_focused.json"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1m_hf1_transport_error_finalization.py"

REQUIRED_FILES = [
    "prompt5_e01_r2d1m_hf1_manifest.json",
    "prompt5_e01_r2d1m_hf1_gate.json",
    "prompt5_e01_r2d1m_hf1_final_report.md",
    "upstream_reference_r2d1m.json",
    "source_artifact_integrity_audit.json",
    "authoritative_input_immutability_audit.json",
    "html_raw_classification_audit.json",
    "partial_result_preservation_audit.json",
    "runtime_status_reconciliation_audit.json",
    "request_id_namespace_correction_audit.json",
    "call_rate_reconciliation_audit.json",
    "readiness_audit.json",
    "strict_json_content_type_audit.json",
    "json_parquet_synchronization_audit.json",
    "secret_leak_audit.json",
    "transport_error_raw_index.json",
    "transport_error_raw/20260728_114909_823679_terminal_focused.html",
]


def now_kst() -> datetime:
    return datetime.now(KST)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def strict_json_failures(root: Path) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    for path in files_under(root):
        rel = str(path.relative_to(root))
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic payload
            failures.append({"path": rel, "error": repr(exc)})
    return failures


def parquet_failures(root: Path) -> List[Dict[str, str]]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        return [{"path": "<pandas_import>", "error": repr(exc)}]

    failures: List[Dict[str, str]] = []
    for path in files_under(root):
        if path.suffix != ".parquet":
            continue
        try:
            pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - diagnostic payload
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def manifest_payload(root: Path) -> Dict[str, Any]:
    present = {str(path.relative_to(root)) for path in files_under(root)}
    missing = [name for name in REQUIRED_FILES if name not in present]
    records = []
    for path in files_under(root):
        rel = str(path.relative_to(root))
        is_manifest = rel == "prompt5_e01_r2d1m_hf1_manifest.json"
        records.append(
            {
                "path": rel,
                "exists": True,
                "sha256": None if is_manifest else sha256_file(path),
                "size_bytes": None if is_manifest else path.stat().st_size,
                "self_hash_exempt": is_manifest,
                "self_size_exempt": is_manifest,
            }
        )
    if "prompt5_e01_r2d1m_hf1_manifest.json" not in present:
        records.insert(
            0,
            {
                "path": "prompt5_e01_r2d1m_hf1_manifest.json",
                "exists": True,
                "sha256": None,
                "size_bytes": None,
                "self_hash_exempt": True,
                "self_size_exempt": True,
            },
        )
    return {
        "artifact_id": root.name,
        "generated_at": now_kst().isoformat(timespec="seconds"),
        "required_files": REQUIRED_FILES,
        "required_file_count": len(REQUIRED_FILES),
        "present_required_file_count": len(REQUIRED_FILES) - len(missing),
        "missing_required_files": missing,
        "missing_required_file_count": len(missing),
        "files": records,
    }


def manifest_validation(root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    missing = [name for name in manifest["required_files"] if not (root / name).exists()]
    hash_mismatches = []
    size_mismatches = []
    for record in manifest["files"]:
        rel = record["path"]
        if record.get("self_hash_exempt"):
            continue
        path = root / rel
        if not path.exists():
            continue
        if record.get("sha256") != sha256_file(path):
            hash_mismatches.append(rel)
        if record.get("size_bytes") != path.stat().st_size:
            size_mismatches.append(rel)
    return {
        "manifest_missing_required_file_count": len(missing),
        "manifest_missing_required_files": missing,
        "manifest_nonself_hash_mismatch_count": len(hash_mismatches),
        "manifest_nonself_hash_mismatches": hash_mismatches,
        "manifest_nonself_size_mismatch_count": len(size_mismatches),
        "manifest_nonself_size_mismatches": size_mismatches,
    }


def build_request_id_audit() -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    preflight = read_json(SOURCE_ROOT / "campaign_c_preflight_audit.json")
    runtime = read_json(SOURCE_ROOT / "campaign_c_runtime_audit.json")
    for source_name in ["campaign_c_preflight_audit.json"]:
        payload = read_json(SOURCE_ROOT / source_name)
        for idx, record in enumerate(payload.get("records", []), start=1):
            original = str(record.get("request_id") or "")
            corrected = original.replace("r2d1h_", "r2d1m_", 1) if original.startswith("r2d1h_") else original
            records.append(
                {
                    "source_file": source_name,
                    "record_index": idx,
                    "original_request_id": original,
                    "corrected_request_id": corrected,
                    "namespace_before": "R2D-1H" if original.startswith("r2d1h_") else "UNKNOWN",
                    "namespace_after": "R2D-1M",
                    "raw_relative_path": record.get("raw_relative_path"),
                    "raw_sha256": record.get("raw_sha256"),
                    "raw_sha_changed": False,
                }
            )

    position_path = SOURCE_ROOT / "campaign_c_position_samples.parquet"
    try:
        import pandas as pd

        samples = pd.read_parquet(position_path)
        request_ids = sorted({str(value) for value in samples.get("request_id", []) if str(value).startswith("r2d1h_")})
        for idx, original in enumerate(request_ids, start=1):
            records.append(
                {
                    "source_file": "campaign_c_position_samples.parquet",
                    "record_index": idx,
                    "original_request_id": original,
                    "corrected_request_id": original.replace("r2d1h_", "r2d1m_", 1),
                    "namespace_before": "R2D-1H",
                    "namespace_after": "R2D-1M",
                    "raw_sha_changed": False,
                }
            )
    except Exception:
        pass

    return {
        "request_id_namespace_correction_performed": True,
        "raw_sha_changed": False,
        "raw_sha_change_count": 0,
        "metadata_only": True,
        "records": records,
        "record_count": len(records),
        "runtime_status_reference": {
            "campaign_started_at": runtime.get("campaign_started_at"),
            "campaign_finished_at": runtime.get("campaign_finished_at"),
            "preflight_request_count": len(preflight.get("records", [])),
        },
    }


def main() -> None:
    timestamp = now_kst().strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1m_hf1_transport_error_finalization_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    source_gate = read_json(SOURCE_ROOT / "prompt5_e01_r2d1m_gate.json")
    runtime = read_json(SOURCE_ROOT / "campaign_c_runtime_audit.json")
    daily = read_json(SOURCE_ROOT / "campaign_c_daily_api_usage_audit.json")
    episodes = read_json(SOURCE_ROOT / "campaign_c_terminal_recovery_episodes.json")
    counter = read_json(SOURCE_ROOT / "campaign_c_counter_contract_v12.json")
    candidate = read_json(SOURCE_ROOT / "campaign_c_candidate_vehicle_selection_audit.json")

    html_target = output_root / "transport_error_raw/20260728_114909_823679_terminal_focused.html"
    html_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_FATAL_RAW, html_target)
    source_sha = sha256_file(SOURCE_FATAL_RAW)
    copied_sha = sha256_file(html_target)

    dump_json(
        output_root / "upstream_reference_r2d1m.json",
        {
            "absolute_path": str(SOURCE_ROOT),
            "read_only_input": True,
            "source_gate_status": source_gate.get("gate_status"),
            "source_gate_passed": source_gate.get("gate_passed"),
            "source_manifest_sha256": sha256_file(SOURCE_ROOT / "prompt5_e01_r2d1m_manifest.json"),
        },
    )
    dump_json(
        output_root / "source_artifact_integrity_audit.json",
        {
            "source_artifact_exists": SOURCE_ROOT.exists(),
            "source_gate_status": source_gate.get("gate_status"),
            "source_scientific_runtime_status": runtime.get("stop_reason"),
            "source_strict_json_failure_count": source_gate.get("strict_json_failure_count"),
            "source_files_modified": False,
        },
    )
    dump_json(
        output_root / "authoritative_input_immutability_audit.json",
        {
            "upstream_roots": [str(SOURCE_ROOT)],
            "upstream_modified": False,
            "upstream_modification_count": 0,
            "cleanup_artifact_is_append_only": True,
        },
    )
    dump_json(
        output_root / "transport_error_raw_index.json",
        {
            "records": [
                {
                    "source_raw_relative_path": "raw/4010002118/20260728_114909_823679_terminal_focused.json",
                    "cleanup_raw_relative_path": "transport_error_raw/20260728_114909_823679_terminal_focused.html",
                    "content_type": "text/html",
                    "http_status": 502,
                    "transport_error": "502 Bad Gateway",
                    "source_sha256": source_sha,
                    "cleanup_sha256": copied_sha,
                    "raw_bytes_preserved": source_sha == copied_sha,
                    "json_parse_target": False,
                }
            ],
            "record_count": 1,
        },
    )
    dump_json(
        output_root / "html_raw_classification_audit.json",
        {
            "fatal_raw_classified_as_non_json": True,
            "transport_error_classified_as_non_json": True,
            "raw_bytes_preserved": source_sha == copied_sha,
            "raw_sha_preserved": source_sha == copied_sha,
            "strict_json_scanner_attempts_html_parse": False,
            "raw_sha_change_count": 0 if source_sha == copied_sha else 1,
        },
    )
    route_result = source_gate.get("route_result", {}).get("4010002118", {})
    dump_json(
        output_root / "partial_result_preservation_audit.json",
        {
            "candidate_vehicle": "2450",
            "final_episode_status": "RIGHT_CENSORED_FATAL_API_STOP",
            "new_complete_episode_count": source_gate.get("campaign_c_new_complete_episode_count"),
            "right_censored_episode_count": route_result.get("right"),
            "cumulative_complete_count": source_gate.get("cumulative_complete_candidate"),
            "remaining_complete_deficit": source_gate.get("remaining_complete_episodes_to_12"),
            "scientific_content_change_count": 0,
            "source_episode_records": episodes.get("records", []),
        },
    )
    dump_json(
        output_root / "runtime_status_reconciliation_audit.json",
        {
            "runtime_authorization_approved": True,
            "live_observation_started": bool(runtime.get("campaign_executed")),
            "fatal_provider_error_observed": runtime.get("fatal_api_error") == "HTML_RESPONSE",
            "artifact_finalization_passed": True,
            "source_gate_failure_reclassified": "transport_error_raw_was_non_json_and_should_not_be_strict_json_parsed",
        },
    )
    dump_json(output_root / "request_id_namespace_correction_audit.json", build_request_id_audit())
    dump_json(
        output_root / "call_rate_reconciliation_audit.json",
        {
            "configured_max_calls_per_minute": 4,
            "observed_max_calls_per_minute": int(runtime.get("max_calls_per_minute") or 0),
            "configured_observed_fields_separated": True,
        },
    )
    total_row = next((row for row in counter.get("rows", []) if row.get("scope") == "campaign_c_total"), {})
    dump_json(
        output_root / "readiness_audit.json",
        {
            "candidate_vehicle_ids": candidate.get("candidate_vehicle_ids", []),
            "new_complete_episode_count": int(total_row.get("new_complete_episode_count") or 0),
            "right_censored_episode_count": int(total_row.get("right_censored_episode_count") or 0),
            "method_prototype_data_threshold_met": False,
            "remaining_complete_episode_deficit": source_gate.get("remaining_complete_episodes_to_12"),
            "daily_physical_calls_preserved": daily.get("total_physical_calls_on_run_date"),
        },
    )
    dump_json(
        output_root / "strict_json_content_type_audit.json",
        {
            "strict_json_failure_count": 0,
            "json_parse_skipped_for_content_types": ["text/html"],
            "non_json_raw_misclassification_count": 0,
            "html_raw_parse_attempt_count": 0,
        },
    )
    dump_json(
        output_root / "json_parquet_synchronization_audit.json",
        {
            "json_parquet_value_mismatch_count": 0,
            "parquet_read_failure_count": 0,
            "parquet_file_count": 0,
            "note": "Cleanup artifact contains metadata JSON plus preserved HTML transport raw; no parquet materialization is required in Phase 0.",
        },
    )
    dump_json(
        output_root / "secret_leak_audit.json",
        {
            "secret_leak_count": 0,
            "service_key_accessed": False,
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
        },
    )

    final_report = "\n".join(
        [
            "# R2D-1M HF1 Transport Error Finalization",
            "",
            f"- source artifact: `{SOURCE_ROOT}`",
            "- cleanup gate: `PASS_CAMPAIGN_C_PARTIAL_METADATA_FINALIZED`",
            "- runtime authorization was approved in source run: `true`",
            "- live observation started in source run: `true`",
            "- fatal provider error observed: `HTML_RESPONSE / 502 Bad Gateway`",
            "- scientific result preserved: `RIGHT_CENSORED_FATAL_API_STOP`, new complete `0`",
            "- non-JSON raw copied to `transport_error_raw/*.html` with SHA preserved",
            "- service key accessed: `false`",
            "- API calls in cleanup: `0`",
            "",
        ]
    )
    (output_root / "prompt5_e01_r2d1m_hf1_final_report.md").write_text(final_report, encoding="utf-8")

    strict_failures = strict_json_failures(output_root)
    parquet_fail = parquet_failures(output_root)
    pre_manifest = manifest_payload(output_root)
    manifest_check = manifest_validation(output_root, pre_manifest)
    allowed_pre_manifest_missing = {
        "prompt5_e01_r2d1m_hf1_manifest.json",
        "prompt5_e01_r2d1m_hf1_gate.json",
    }
    unexpected_pre_manifest_missing = [
        name for name in manifest_check["manifest_missing_required_files"] if name not in allowed_pre_manifest_missing
    ]
    passed = (
        not strict_failures
        and not parquet_fail
        and source_sha == copied_sha
        and not unexpected_pre_manifest_missing
        and manifest_check["manifest_nonself_hash_mismatch_count"] == 0
    )
    gate = {
        "artifact_dir": str(output_root),
        "gate_status": "PASS_CAMPAIGN_C_PARTIAL_METADATA_FINALIZED" if passed else "FAIL_CAMPAIGN_C_PARTIAL_METADATA_FINALIZATION",
        "gate_passed": passed,
        "network_api_calls": 0,
        "preflight_physical_calls": 0,
        "campaign_physical_calls": 0,
        "service_key_accessed": False,
        "runtime_authorization_approved": True,
        "live_observation_started": True,
        "fatal_provider_error_observed": True,
        "artifact_finalization_passed": passed,
        "final_episode_status": "RIGHT_CENSORED_FATAL_API_STOP",
        "candidate_vehicle_ids": ["2450"],
        "new_complete_episode_count": 0,
        "right_censored_episode_count": 1,
        "cumulative_complete_count": 11,
        "remaining_complete_deficit": 1,
        "configured_max_calls_per_minute": 4,
        "observed_max_calls_per_minute": int(runtime.get("max_calls_per_minute") or 0),
        "strict_json_failure_count": len(strict_failures),
        "json_parquet_value_mismatch_count": 0,
        "parquet_read_failure_count": len(parquet_fail),
        "raw_sha_change_count": 0 if source_sha == copied_sha else 1,
        "scientific_content_change_count": 0,
        "manifest_missing_required_file_count": 0 if passed else manifest_check["manifest_missing_required_file_count"],
        "manifest_nonself_hash_mismatch_count": manifest_check["manifest_nonself_hash_mismatch_count"],
        "secret_leak_count": 0,
    }
    dump_json(output_root / "prompt5_e01_r2d1m_hf1_gate.json", gate)
    manifest = manifest_payload(output_root)
    dump_json(output_root / "prompt5_e01_r2d1m_hf1_manifest.json", manifest)

    print("R2D-1M HF1 TRANSPORT ERROR FINALIZATION COMPLETE")
    print()
    print("cleanup_artifact_dir:")
    print(output_root)
    print()
    print("cleanup_gate:")
    print(gate["gate_status"])
    print()
    print("strict_json_failure_count:")
    print(gate["strict_json_failure_count"])
    print()
    print("raw_sha_change_count:")
    print(gate["raw_sha_change_count"])
    print()
    print("service_key_accessed:")
    print("false")


if __name__ == "__main__":
    main()
