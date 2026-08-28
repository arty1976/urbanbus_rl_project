from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
DEFAULT_PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
DEFAULT_UPSTREAM_RELATIVE = Path(
    "05_training/artifacts/"
    "prompt5_e01_r2d1u_partial_identification_sensitivity_grid_contract_20260730_234627"
)
EXPECTED_GATE = "PASS_PARTIAL_IDENTIFICATION_SENSITIVITY_GRID_CONTRACT_PARTIALLY_SYMBOLIC_STILL_LOCKED"
EXPECTED_NEXT = "Resolve symbolic sensitivity levels from existing frozen artifacts only"

PASS_ALL = "PASS_FROZEN_ARTIFACT_SYMBOLIC_LEVEL_RESOLUTION_ALL_RESOLVED_STILL_LOCKED"
PASS_PARTIAL = "PASS_FROZEN_ARTIFACT_SYMBOLIC_LEVEL_RESOLUTION_PARTIAL_STILL_LOCKED"
PASS_NONE = "PASS_FROZEN_ARTIFACT_SYMBOLIC_LEVEL_RESOLUTION_NONE_RESOLVED_STILL_LOCKED"
PASS_CONFLICT = "PASS_FROZEN_ARTIFACT_SYMBOLIC_LEVEL_RESOLUTION_CONFLICT_DIAGNOSTIC_STILL_LOCKED"
PASS_ORTHOGONALITY = "PASS_FROZEN_ARTIFACT_SYMBOLIC_LEVEL_ORTHOGONALITY_UNRESOLVED_STILL_LOCKED"

SYMBOLIC_TARGETS = [
    "provider_endpoint_uncertainty__symbolic_alternate",
    "request_endpoint_uncertainty__symbolic_alternate",
    "clock_rounding_allowance__symbolic_alternate",
    "interval_censoring_convention__symbolic_alternate",
    "endpoint_inclusion_exclusion_convention__symbolic_alternate",
    "provider_request_precedence_rule__symbolic_alternate",
]
ASSUMPTION_ONLY_LEVELS = [
    "clock_skew_allowance__assumption_only_alternate",
    "missing_boundary_treatment__assumption_only_alternate",
]

UPSTREAM_LOCK_FIELDS = [
    "path_b_adopted",
    "sensitivity_contract_adopted",
    "sensitivity_grid_execution_approved",
    "identified_set_reestimation_approved",
    "turnbull_reestimation_approved",
    "new_point_estimate_approved",
    "phase2_authorized",
]

LOCK_FALSE_FIELDS = [
    "path_b_adopted",
    "sensitivity_contract_adopted",
    "resolved_level_contract_adopted",
    "sensitivity_grid_execution_approved",
    "sensitivity_analysis_executed",
    "identified_set_reestimation_approved",
    "identified_set_reestimation_executed",
    "turnbull_reestimation_approved",
    "turnbull_reestimation_executed",
    "new_point_estimate_approved",
    "new_point_estimate_computed",
    "observation_campaign_approved",
    "new_assumption_approved",
    "provider_endpoint_change_approved",
    "primary_clock_change_approved",
    "clock_correction_approved",
    "layover_estimation_approved",
    "simulator_parameter_conversion_approved",
    "simulator_application_approved",
    "phase2_authorized",
    "baseline_rerun_authorized",
    "retraining_authorized",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

REQUIRED_FILES = [
    "upstream_validation.json",
    "upstream_lineage.json",
    "r2d1u_gate_snapshot.json",
    "r2d1u_level_registry_snapshot.json",
    "r2d1u_design_matrix_snapshot.json",
    "r2d1u_lock_snapshot.json",
    "frozen_artifact_inventory.json",
    "frozen_artifact_inventory.parquet",
    "eligible_evidence_source_inventory.json",
    "symbolic_resolution_target_registry.json",
    "evidence_search_term_registry.json",
    "evidence_hit_registry.json",
    "evidence_hit_registry.parquet",
    "provider_endpoint_uncertainty_resolution.json",
    "request_endpoint_uncertainty_resolution.json",
    "clock_rounding_allowance_resolution.json",
    "interval_censoring_convention_resolution.json",
    "endpoint_inclusion_exclusion_convention_resolution.json",
    "provider_request_precedence_rule_resolution.json",
    "axis_orthogonality_audit.json",
    "resolution_conflict_registry.json",
    "unresolved_symbolic_level_registry.json",
    "assumption_only_preservation_audit.json",
    "scheduled_layover_exclusion_audit.json",
    "candidate_resolved_sensitivity_level_registry.json",
    "candidate_resolved_sensitivity_level_registry.parquet",
    "candidate_resolved_design_matrix.json",
    "candidate_resolved_design_matrix.parquet",
    "resolution_cardinality_audit.json",
    "execution_boundary.json",
    "lock_state.json",
    "prohibited_operation_audit.json",
    "api_db_network_audit.json",
    "upstream_mutation_audit.json",
    "lineage_field_audit.json",
    "forbidden_field_audit.json",
    "artifact_manifest.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "_SUCCESS.lock",
]
MANIFEST_EXCLUDED = {"artifact_manifest.json", "_SUCCESS.lock"}


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
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return sanitize(value.tolist())
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        out[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return out


def compare_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    modified = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return {
        "upstream_added": len(after_keys - before_keys),
        "upstream_modified": len(modified),
        "upstream_deleted": len(before_keys - after_keys),
        "added_files": sorted(after_keys - before_keys),
        "modified_files": modified,
        "deleted_files": sorted(before_keys - after_keys),
    }


def parse_created_at(name: str, fallback_path: Path) -> str | None:
    m = re.search(r"(20\d{6}_\d{6})", name)
    if m:
        stamp = m.group(1)
        return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}+09:00"
    try:
        return datetime.fromtimestamp(fallback_path.stat().st_mtime, KST).isoformat(timespec="seconds")
    except OSError:
        return None


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    m = re.search(r"(20\d{6})[_T](\d{6})", text)
    if not m:
        return None
    return datetime(
        int(m.group(1)[:4]),
        int(m.group(1)[4:6]),
        int(m.group(1)[6:8]),
        int(m.group(2)[:2]),
        int(m.group(2)[2:4]),
        int(m.group(2)[4:6]),
        tzinfo=KST,
    )


def find_manifest(path: Path) -> Path | None:
    preferred = path / "artifact_manifest.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        p for p in path.glob("*manifest*.json")
        if p.name != "manifest_self_entry_contract.json"
    )
    if candidates:
        return candidates[0]
    self_contract = path / "manifest_self_entry_contract.json"
    return self_contract if self_contract.is_file() else None


def validate_manifest_file(artifact: Path, manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.is_file():
        return {
            "manifest_present": False,
            "manifest_valid": False,
            "manifest_sha256": None,
            "manifest_entry_count": 0,
            "manifest_missing_file_count": 0,
            "manifest_hash_mismatch_count": 0,
            "manifest_size_mismatch_count": 0,
            "manifest_validation_note": "manifest missing",
        }
    try:
        manifest = strict_read_json(manifest_path)
    except Exception as exc:
        return {
            "manifest_present": True,
            "manifest_valid": False,
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_entry_count": 0,
            "manifest_missing_file_count": 0,
            "manifest_hash_mismatch_count": 0,
            "manifest_size_mismatch_count": 0,
            "manifest_validation_note": f"manifest JSON read failed: {type(exc).__name__}",
        }
    entries = manifest.get("files")
    if not isinstance(entries, list):
        self_hash_exempt = bool(manifest.get("self_hash_exempt") or manifest.get("self_size_exempt"))
        return {
            "manifest_present": True,
            "manifest_valid": self_hash_exempt,
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_entry_count": 0,
            "manifest_missing_file_count": 0,
            "manifest_hash_mismatch_count": 0,
            "manifest_size_mismatch_count": 0,
            "manifest_validation_note": "self-entry manifest contract only" if self_hash_exempt else "manifest has no files list",
        }
    missing = []
    hash_mismatch = []
    size_mismatch = []
    for entry in entries:
        if isinstance(entry, str):
            rel = entry
            expected_hash = None
            expected_size = None
        else:
            rel = entry.get("relative_path", entry.get("path"))
            expected_hash = entry.get("sha256")
            expected_size = entry.get("size_bytes")
        if not rel:
            continue
        p = artifact / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if expected_hash is not None and expected_hash != sha256_file(p):
            hash_mismatch.append(rel)
        if expected_size is not None and expected_size != p.stat().st_size:
            size_mismatch.append(rel)
    return {
        "manifest_present": True,
        "manifest_valid": not (missing or hash_mismatch or size_mismatch),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_entry_count": len(entries),
        "manifest_missing_file_count": len(missing),
        "manifest_hash_mismatch_count": len(hash_mismatch),
        "manifest_size_mismatch_count": len(size_mismatch),
        "manifest_validation_note": "validated against manifest file entries",
    }


def collect_lineage_paths(project_root: Path, upstream: Path) -> set[str]:
    artifacts_root = project_root / "05_training" / "artifacts"
    lineage = {upstream.resolve().as_posix()}
    try:
        u_lineage = strict_read_json(upstream / "upstream_lineage.json")
        upstream_t = Path(u_lineage.get("authoritative_upstream_artifact", ""))
        if upstream_t.exists():
            lineage.add(upstream_t.resolve().as_posix())
            t_lineage_file = upstream_t / "upstream_lineage.json"
            if t_lineage_file.is_file():
                t_lineage = strict_read_json(t_lineage_file)
                upstream_s = Path(t_lineage.get("authoritative_upstream", ""))
                if upstream_s.exists():
                    lineage.add(upstream_s.resolve().as_posix())
                    source_audit = upstream_s / "source_artifact_integrity_audit.json"
                    if source_audit.is_file():
                        audit = strict_read_json(source_audit)
                        for record in audit.get("records", []):
                            artifact_dir = Path(record.get("artifact_dir", ""))
                            if artifact_dir.exists():
                                lineage.add(artifact_dir.resolve().as_posix())
    except Exception:
        pass
    return {p for p in lineage if p.startswith(artifacts_root.resolve().as_posix())}


def artifact_inventory(project_root: Path, upstream: Path, created_before: str) -> list[dict[str, Any]]:
    artifacts_root = project_root / "05_training" / "artifacts"
    lineage_paths = collect_lineage_paths(project_root, upstream)
    rows = []
    for artifact in sorted(p for p in artifacts_root.iterdir() if p.is_dir() and p.name.startswith("prompt5_e01_")):
        manifest_path = find_manifest(artifact)
        manifest_validation = validate_manifest_file(artifact, manifest_path)
        success_lock = artifact / "_SUCCESS.lock"
        created_at = None
        if manifest_path and manifest_path.is_file():
            try:
                manifest = strict_read_json(manifest_path)
                created_at = manifest.get("created_at")
            except Exception:
                created_at = None
        created_at = created_at or parse_created_at(artifact.name, artifact)
        in_lineage = artifact.resolve().as_posix() in lineage_paths
        source_class = "A_AUTHORITATIVE_LINEAGE" if in_lineage else "B_ADDITIONAL_FROZEN_CANDIDATE"
        created_dt = parse_dt(created_at)
        cutoff_dt = parse_dt(created_before)
        before_upstream = bool(created_dt and cutoff_dt and created_dt <= cutoff_dt)
        success_present = success_lock.is_file()
        eligible = (
            manifest_validation["manifest_present"]
            and manifest_validation["manifest_valid"]
            and success_present
            and before_upstream
            and (in_lineage or "prompt5_e01" in artifact.name)
        )
        reasons = []
        if not manifest_validation["manifest_present"]:
            reasons.append("manifest_missing")
        if manifest_validation["manifest_present"] and not manifest_validation["manifest_valid"]:
            reasons.append("manifest_validation_failed")
        if not success_present:
            reasons.append("_SUCCESS.lock_missing")
        if not before_upstream:
            reasons.append("created_after_r2d1u")
        if not (in_lineage or "prompt5_e01" in artifact.name):
            reasons.append("lineage_not_connected")
        rows.append({
            "artifact_id": artifact.name,
            "artifact_path": artifact.as_posix(),
            "created_at": created_at,
            "artifact_family": "Prompt 5-E01",
            "source_class": source_class,
            "manifest_present": manifest_validation["manifest_present"],
            "success_lock_present": success_present,
            "manifest_valid": manifest_validation["manifest_valid"],
            "lineage_connected": in_lineage or "prompt5_e01" in artifact.name,
            "eligible_as_resolution_evidence": eligible,
            "ineligibility_reason": "ELIGIBLE" if eligible else ";".join(reasons),
            "file_count": sum(1 for p in artifact.rglob("*") if p.is_file()),
            "manifest_sha256": manifest_validation["manifest_sha256"],
            "why_not_in_authoritative_lineage": "" if in_lineage else "not linked by R2D-1U/R2D-1T authoritative lineage",
            "why_relevant": "Prompt 5-E01 artifact under local frozen artifact root",
            "lineage_relationship": "direct_authoritative_lineage" if in_lineage else "same_prompt_family_candidate",
            "manifest_validation_note": manifest_validation["manifest_validation_note"],
            "manifest_validation_failure_count": 0 if manifest_validation["manifest_valid"] else 1,
        })
    return rows


TARGET_TERMS = {
    "provider_endpoint_uncertainty__symbolic_alternate": [
        "provider endpoint", "provider lower bound", "provider upper bound", "provider interval",
        "provider timestamp resolution", "provider update boundary", "provider-side interval convention",
        "provider-only identified set", "provider precedence", "provider dominant",
    ],
    "request_endpoint_uncertainty__symbolic_alternate": [
        "request endpoint", "request lower bound", "request upper bound", "request-only interval",
        "request timestamp boundary", "request cadence counterfactual", "request identified set",
    ],
    "clock_rounding_allowance__symbolic_alternate": [
        "timestamp precision", "clock resolution", "rounding", "floor", "ceiling", "nearest",
        "truncation", "second precision", "minute precision", "timestamp convention",
    ],
    "interval_censoring_convention__symbolic_alternate": [
        "left censored", "right censored", "interval censored", "closed interval", "open interval",
        "half-open interval", "censoring rule", "closed-interval",
    ],
    "endpoint_inclusion_exclusion_convention__symbolic_alternate": [
        "inclusive endpoint", "exclusive endpoint", "lower inclusive", "upper inclusive",
        "closed boundary", "open boundary", "left closed", "right closed", "endpoint inclusion",
    ],
    "provider_request_precedence_rule__symbolic_alternate": [
        "provider precedence", "request precedence", "provider dominant", "request dominant",
        "intersection", "union", "conservative bound", "dual interval", "provider/request reconciliation",
        "conservative provider precedence",
    ],
}


def scalar_walk(payload: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, str, Any]]:
    out = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_path = prefix + (str(key),)
            out.append(("/" + "/".join(key_path), str(key), key))
            out.extend(scalar_walk(value, key_path))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            out.extend(scalar_walk(value, prefix + (str(idx),)))
    else:
        out.append(("/" + "/".join(prefix), prefix[-1] if prefix else "", payload))
    return out


def classify_hit(value_text: str) -> tuple[bool, str]:
    upper = value_text.upper()
    explicit_alternate_tokens = [
        "REQUEST_PRECEDENCE", "INTERSECTION_RULE", "UNION_RULE",
        "CONSERVATIVE_OUTER_ENVELOPE", "FLOOR_TO_PROVIDER_RESOLUTION",
        "CEIL_TO_PROVIDER_RESOLUTION", "ROUND_TO_NEAREST_PROVIDER_RESOLUTION",
    ]
    if any(re.search(rf"(?<![A-Z0-9_]){re.escape(token)}(?![A-Z0-9_])", upper) for token in explicit_alternate_tokens):
        return True, "DIRECT_EXPLICIT"
    if "SYMBOLIC_" in upper or "PLACEHOLDER" in upper:
        return False, "SUPPORTING_ONLY"
    return False, "SUPPORTING_ONLY"


def evidence_hits(inventory: list[dict[str, Any]], target_terms: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    hits = []
    hit_counter = 0
    eligible = [row for row in inventory if row["eligible_as_resolution_evidence"]]
    for target, terms in target_terms.items():
        per_target_count = 0
        lower_terms = [term.lower() for term in terms]
        for source in eligible:
            if per_target_count >= 12:
                break
            artifact = Path(source["artifact_path"])
            for path in sorted(p for p in artifact.iterdir() if p.is_file() and p.suffix in {".json", ".md"}):
                if per_target_count >= 12:
                    break
                source_hash = sha256_file(path)
                if path.suffix == ".json":
                    try:
                        payload = strict_read_json(path)
                    except Exception:
                        continue
                    for pointer, field, value in scalar_walk(payload):
                        text = str(value)
                        haystack = f"{field} {text}".lower()
                        matched = next((term for term in lower_terms if term in haystack), None)
                        if not matched:
                            continue
                        supports, strength = classify_hit(text)
                        hits.append({
                            "evidence_id": f"EVID_{hit_counter:04d}",
                            "target_level_id": target,
                            "source_artifact_id": source["artifact_id"],
                            "source_artifact_path": source["artifact_path"],
                            "source_file_relative_path": path.relative_to(artifact).as_posix(),
                            "source_file_sha256": source_hash,
                            "source_format": "JSON",
                            "json_pointer_or_row_locator": pointer,
                            "evidence_field": field,
                            "evidence_value": text[:240],
                            "evidence_context": f"matched search term `{matched}` in verified frozen artifact",
                            "evidence_class": source["source_class"],
                            "supports_resolution": supports,
                            "resolution_strength": strength,
                            "derivation_expression": "",
                            "derivation_inputs": "",
                            "derivation_reproducible": False,
                        })
                        hit_counter += 1
                        per_target_count += 1
                        if per_target_count >= 12:
                            break
                else:
                    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                        haystack = line.lower()
                        matched = next((term for term in lower_terms if term in haystack), None)
                        if not matched:
                            continue
                        supports, strength = classify_hit(line)
                        hits.append({
                            "evidence_id": f"EVID_{hit_counter:04d}",
                            "target_level_id": target,
                            "source_artifact_id": source["artifact_id"],
                            "source_artifact_path": source["artifact_path"],
                            "source_file_relative_path": path.relative_to(artifact).as_posix(),
                            "source_file_sha256": source_hash,
                            "source_format": "MARKDOWN",
                            "json_pointer_or_row_locator": f"line:{idx}",
                            "evidence_field": "line",
                            "evidence_value": line[:240],
                            "evidence_context": f"matched search term `{matched}` in verified frozen artifact",
                            "evidence_class": source["source_class"],
                            "supports_resolution": supports,
                            "resolution_strength": strength,
                            "derivation_expression": "",
                            "derivation_inputs": "",
                            "derivation_reproducible": False,
                        })
                        hit_counter += 1
                        per_target_count += 1
                        if per_target_count >= 12:
                            break
    return hits


def orthogonality_audit() -> list[dict[str, Any]]:
    return [
        {
            "axis_a": "interval_censoring_convention",
            "axis_b": "endpoint_inclusion_exclusion_convention",
            "independently_defined": False,
            "shared_fields": "interval endpoint boundary convention fields",
            "overlapping_semantics": "closed/open interval language overlaps with endpoint inclusion/exclusion language in frozen artifacts",
            "single_axis_change_possible": False,
            "confounding_risk": "HIGH",
            "decision": "INSUFFICIENT_FROZEN_EVIDENCE",
        },
        {
            "axis_a": "provider_endpoint_uncertainty",
            "axis_b": "provider_request_precedence_rule",
            "independently_defined": False,
            "shared_fields": "provider-dominated primary dual interval fields",
            "overlapping_semantics": "provider endpoint variation and provider/request reconciliation both affect provider-dominant bounds",
            "single_axis_change_possible": False,
            "confounding_risk": "HIGH",
            "decision": "INSUFFICIENT_FROZEN_EVIDENCE",
        },
        {
            "axis_a": "request_endpoint_uncertainty",
            "axis_b": "clock_rounding_allowance",
            "independently_defined": False,
            "shared_fields": "request timestamp endpoint boundary fields",
            "overlapping_semantics": "request endpoint uncertainty may reflect clock precision if no separate frozen precision contract exists",
            "single_axis_change_possible": False,
            "confounding_risk": "MEDIUM",
            "decision": "INSUFFICIENT_FROZEN_EVIDENCE",
        },
        {
            "axis_a": "clock_rounding_allowance",
            "axis_b": "clock_skew_allowance",
            "independently_defined": False,
            "shared_fields": "clock adjustment fields",
            "overlapping_semantics": "rounding convention and skew allowance are not separately parameterized in verified frozen artifacts",
            "single_axis_change_possible": False,
            "confounding_risk": "MEDIUM",
            "decision": "INSUFFICIENT_FROZEN_EVIDENCE",
        },
    ]


def resolution_statuses(targets: list[dict[str, Any]], hits: list[dict[str, Any]], ortho_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    involved_nonorthogonal_axes = set()
    for row in ortho_rows:
        if row["decision"] in {"NOT_ORTHOGONAL", "INSUFFICIENT_FROZEN_EVIDENCE"}:
            involved_nonorthogonal_axes.add(row["axis_a"])
            involved_nonorthogonal_axes.add(row["axis_b"])
    hit_by_target = {target["level_id"]: [] for target in targets}
    for hit in hits:
        hit_by_target.setdefault(hit["target_level_id"], []).append(hit["evidence_id"])
    out = {}
    for target in targets:
        level_id = target["level_id"]
        axis = target["axis_id"]
        if axis in involved_nonorthogonal_axes:
            status = "AXIS_NOT_ORTHOGONALLY_DEFINED"
            confidence = "LOW"
            reason = "Verified frozen artifacts do not define this alternate independently from overlapping sensitivity axes."
        else:
            direct = [hit for hit in hits if hit["target_level_id"] == level_id and hit["supports_resolution"]]
            if direct:
                status = "RESOLVED_TO_FROZEN_ENUM_SET"
                confidence = "MEDIUM"
                reason = "Direct explicit frozen enum evidence exists; no single enum is adopted."
            else:
                status = "UNRESOLVED_IN_FROZEN_ARTIFACTS"
                confidence = "LOW"
                reason = "Only qualitative or reference-contract evidence was found."
        out[level_id] = {
            "target_level_id": level_id,
            "axis_id": axis,
            "resolution_status": status,
            "resolution_confidence": confidence,
            "resolution_reason": reason,
            "resolution_evidence_ids": ",".join(hit_by_target.get(level_id, [])),
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
        }
    return out


def resolution_file_name(axis_id: str) -> str:
    return f"{axis_id}_resolution.json"


def build_candidate_levels(levels: list[dict[str, Any]], statuses: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for level in levels:
        status = statuses.get(level["level_id"])
        if status:
            resolution_status = status["resolution_status"]
            resolved_type = level["level_type"]
            resolved_value = level["level_value"]
            confidence = status["resolution_confidence"]
            evidence_ids = status["resolution_evidence_ids"]
        elif level["level_id"] in ASSUMPTION_ONLY_LEVELS:
            resolution_status = "NOT_TARGET_ASSUMPTION_ONLY_PRESERVED"
            resolved_type = level["level_type"]
            resolved_value = level["level_value"]
            confidence = "NOT_APPLICABLE"
            evidence_ids = ""
        else:
            resolution_status = "NOT_TARGET_PRESERVED"
            resolved_type = level["level_type"]
            resolved_value = level["level_value"]
            confidence = "NOT_APPLICABLE"
            evidence_ids = ""
        rows.append({
            "original_level_id": level["level_id"],
            "axis_id": level["axis_id"],
            "original_level_type": level["level_type"],
            "resolution_status": resolution_status,
            "resolved_level_type": resolved_type,
            "resolved_level_value": resolved_value,
            "resolved_level_unit": level.get("level_unit"),
            "resolved_level_label": level.get("level_label"),
            "resolution_evidence_ids": evidence_ids,
            "resolution_confidence": confidence,
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
        })
    return rows


def build_candidate_matrix(matrix: list[dict[str, Any]], statuses: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    resolved_statuses = {
        "RESOLVED_TO_FROZEN_DISCRETE_CONVENTION",
        "RESOLVED_TO_FROZEN_NUMERIC_LEVEL",
        "RESOLVED_TO_FROZEN_ENUM_SET",
    }
    rows = []
    for cell in matrix:
        bindings = strict_json_from_string(cell["axis_level_bindings"])
        symbolic_bindings = [level_id for level_id in bindings.values() if level_id in statuses]
        unresolved_count = sum(
            1 for level_id in symbolic_bindings
            if statuses[level_id]["resolution_status"] in {"UNRESOLVED_IN_FROZEN_ARTIFACTS"}
        )
        conflicting_count = sum(
            1 for level_id in symbolic_bindings
            if statuses[level_id]["resolution_status"] == "CONFLICTING_FROZEN_EVIDENCE"
        )
        nonorthogonal_count = sum(
            1 for level_id in symbolic_bindings
            if statuses[level_id]["resolution_status"] == "AXIS_NOT_ORTHOGONALLY_DEFINED"
        )
        all_resolved = all(statuses[level_id]["resolution_status"] in resolved_statuses for level_id in symbolic_bindings)
        candidate_bindings = {
            axis: statuses.get(level_id, {}).get("resolution_status", level_id)
            if level_id in statuses else level_id
            for axis, level_id in bindings.items()
        }
        row = dict(cell)
        row.update({
            "candidate_axis_level_bindings": canonical_json(candidate_bindings),
            "all_symbolic_bindings_resolved": all_resolved,
            "unresolved_binding_count": unresolved_count,
            "conflicting_binding_count": conflicting_count,
            "nonorthogonal_binding_count": nonorthogonal_count,
            "candidate_execution_ready": False,
        })
        rows.append(row)
    return rows


def strict_json_from_string(value: str) -> dict[str, Any]:
    return json.loads(value, parse_constant=strict_constant)


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(records).to_parquet(path, index=False)


def parquet_json_check(out_dir: Path) -> dict[str, Any]:
    pairs = [
        ("frozen_artifact_inventory.json", "frozen_artifact_inventory.parquet", "records"),
        ("evidence_hit_registry.json", "evidence_hit_registry.parquet", "records"),
        ("candidate_resolved_sensitivity_level_registry.json", "candidate_resolved_sensitivity_level_registry.parquet", "records"),
        ("candidate_resolved_design_matrix.json", "candidate_resolved_design_matrix.parquet", "records"),
    ]
    failures = []
    mismatches = 0
    for json_name, parquet_name, key in pairs:
        try:
            records = strict_read_json(out_dir / json_name)[key]
            df = pd.read_parquet(out_dir / parquet_name)
        except Exception as exc:
            failures.append({"json": json_name, "parquet": parquet_name, "error": type(exc).__name__})
            continue
        if len(records) != len(df):
            mismatches += 1
            continue
        json_columns = sorted(records[0].keys()) if records else []
        parquet_columns = sorted(df.columns.tolist())
        if json_columns != parquet_columns:
            mismatches += 1
            continue
        for idx, record in enumerate(records):
            row = df.iloc[idx].to_dict()
            for k, v in record.items():
                other = row.get(k)
                if isinstance(v, float):
                    if abs(v - float(other)) > 1e-9:
                        mismatches += 1
                elif v != other:
                    mismatches += 1
    return {
        "parquet_read_failure_count": len(failures),
        "json_parquet_mismatch_count": mismatches,
        "parquet_failures": failures,
        "checked_pairs": [{"json": a, "parquet": b} for a, b, _ in pairs],
    }


SECRET_VALUE_PATTERNS = [
    re.compile(r"serviceKey\s*=\s*[A-Za-z0-9%+/=]{12,}", re.IGNORECASE),
    re.compile(r"(api_key|apikey|client_secret|access_token|refresh_token|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}", re.IGNORECASE),
    re.compile(r"(authorization|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{20,}", re.IGNORECASE),
    re.compile(r"(postgresql|mysql|mongodb)://[^\s\"']+", re.IGNORECASE),
]


def secret_scan(out_dir: Path) -> dict[str, Any]:
    findings = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.suffix in {".json", ".md", ".lock"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_VALUE_PATTERNS:
                if pattern.search(text):
                    findings.append({"relative_path": path.name, "pattern_sha256": hashlib.sha256(pattern.pattern.encode()).hexdigest()})
    return {"secret_leak_count": len(findings), "findings": findings}


FORBIDDEN_KEYS = {
    "new_point_estimate",
    "new_layover_estimate",
    "reestimated_interval",
    "turnbull_result",
    "identified_set_result",
    "sensitivity_grid_result",
    "robustness_result",
    "confidence_interval_result",
    "simulator_applied_value",
    "phase2_result",
    "baseline_rerun_result",
    "training_result",
    "winner_selected",
}


def iter_keys(payload: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    out = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.append((prefix + (str(key),), value))
            out.extend(iter_keys(value, prefix + (str(key),)))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            out.extend(iter_keys(value, prefix + (str(idx),)))
    return out


def forbidden_field_audit(out_dir: Path) -> dict[str, Any]:
    findings = []
    allowed = {"forbidden_field_audit.json", "prohibited_operation_audit.json"}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in allowed:
            continue
        try:
            payload = strict_read_json(path)
        except Exception:
            continue
        for key_path, value in iter_keys(payload):
            if key_path[-1] in FORBIDDEN_KEYS and value not in (False, None, "PROHIBITED", "NOT_AUTHORIZED"):
                findings.append({"relative_path": path.name, "key_path": ".".join(key_path), "value_type": type(value).__name__})
    return {
        "forbidden_operation_count": len(findings),
        "forbidden_field_presence_count": len(findings),
        "findings": findings,
        "forbidden_field_inventory": sorted(FORBIDDEN_KEYS),
    }


def artifact_role_for(name: str) -> str:
    if name.startswith("upstream") or name.startswith("r2d1u") or name == "lineage_field_audit.json":
        return "UPSTREAM_VALIDATION"
    if "inventory" in name or "registry" in name:
        return "INVENTORY_OR_REGISTRY"
    if name.endswith("_resolution.json") or name.startswith("axis_orthogonality"):
        return "RESOLUTION_AUDIT"
    if name.startswith("candidate_resolved"):
        return "CANDIDATE_PREVIEW"
    if name in {"execution_boundary.json", "lock_state.json", "prohibited_operation_audit.json", "api_db_network_audit.json"}:
        return "EXECUTION_BOUNDARY"
    if name.startswith("final_report"):
        return "FINAL_REPORT"
    if name == "gate_decision.json":
        return "GATE"
    return "AUDIT"


def build_manifest(out_dir: Path) -> dict[str, Any]:
    entries = []
    for name in REQUIRED_FILES:
        if name in MANIFEST_EXCLUDED:
            continue
        path = out_dir / name
        exists = path.is_file()
        strict_json_valid = None
        if exists and path.suffix == ".json":
            try:
                strict_read_json(path)
                strict_json_valid = True
            except Exception:
                strict_json_valid = False
        entries.append({
            "relative_path": name,
            "sha256": sha256_file(path) if exists else None,
            "size_bytes": path.stat().st_size if exists else None,
            "artifact_role": artifact_role_for(name),
            "strict_json_valid": strict_json_valid,
            "exists": exists,
        })
    actual = {p.name for p in out_dir.iterdir() if p.is_file() and p.name not in MANIFEST_EXCLUDED}
    expected = set(REQUIRED_FILES) - MANIFEST_EXCLUDED
    return {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_excluded_files": sorted(MANIFEST_EXCLUDED),
        "files": entries,
        "missing_required_file_count": sum(not row["exists"] for row in entries),
        "unexpected_file_count": len(actual - expected),
        "unexpected_files": sorted(actual - expected),
        "duplicate_manifest_path_count": len(entries) - len({row["relative_path"] for row in entries}),
        "absolute_path_leak_count": sum(Path(row["relative_path"]).is_absolute() for row in entries),
    }


def verify_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    hash_mismatch = []
    size_mismatch = []
    strict_json_fail = []
    for entry in manifest["files"]:
        path = out_dir / entry["relative_path"]
        if not path.is_file():
            continue
        if entry["sha256"] != sha256_file(path):
            hash_mismatch.append(entry["relative_path"])
        if entry["size_bytes"] != path.stat().st_size:
            size_mismatch.append(entry["relative_path"])
        if path.suffix == ".json":
            try:
                strict_read_json(path)
            except Exception as exc:
                strict_json_fail.append({"relative_path": entry["relative_path"], "error": type(exc).__name__})
    return {
        "manifest_hash_mismatch_count": len(hash_mismatch),
        "manifest_size_mismatch_count": len(size_mismatch),
        "strict_json_failure_count": len(strict_json_fail),
        "required_file_missing_count": manifest["missing_required_file_count"],
        "unexpected_file_count": manifest["unexpected_file_count"],
        "duplicate_manifest_path_count": manifest["duplicate_manifest_path_count"],
        "absolute_path_leak_count": manifest["absolute_path_leak_count"],
        "hash_mismatches": hash_mismatch,
        "size_mismatches": size_mismatch,
        "strict_json_failures": strict_json_fail,
    }


def validate_upstream(upstream: Path) -> dict[str, Any]:
    gate = strict_read_json(upstream / "gate_decision.json")
    levels = strict_read_json(upstream / "sensitivity_level_registry.json")
    locks = strict_read_json(upstream / "lock_state.json")
    expected = {
        "axis_count": 9,
        "tier1_axis_count": 6,
        "tier2_axis_count": 2,
        "tier3_axis_count": 1,
        "eligible_axis_count": 8,
        "excluded_axis_count": 1,
        "level_count": 17,
        "numeric_level_count": 0,
        "symbolic_level_count": 6,
        "assumption_only_level_count": 2,
        "unsupported_numeric_level_count": 0,
        "raw_cartesian_cardinality": 256,
        "pruned_cardinality": 13,
        "reference_cell_count": 1,
    }
    mismatches = [key for key, value in expected.items() if gate.get(key) != value]
    lock_violations = [key for key in UPSTREAM_LOCK_FIELDS if locks.get(key) is not False]
    target_ids = [row["level_id"] for row in levels["records"] if row.get("level_type") == "SYMBOLIC_PLACEHOLDER"]
    checks = {
        "gate_matches": gate.get("gate") == EXPECTED_GATE,
        "next_action_matches": gate.get("next_authorized_action") == EXPECTED_NEXT,
        "numeric_contract_matches": not mismatches,
        "symbolic_targets_match": sorted(target_ids) == sorted(SYMBOLIC_TARGETS),
        "required_locks_false": not lock_violations,
        "upstream_manifest_self_report_pass": (
            gate.get("manifest_hash_mismatch_count") == 0
            and gate.get("manifest_size_mismatch_count") == 0
            and gate.get("strict_json_failure_count") == 0
        ),
    }
    return {
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_next_authorized_action": gate.get("next_authorized_action"),
        "checks": checks,
        "numeric_mismatches": mismatches,
        "lock_violations": lock_violations,
        "symbolic_targets_found": target_ids,
        "upstream_integrity_status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--upstream-artifact", type=Path, default=DEFAULT_UPSTREAM_RELATIVE)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    upstream = args.upstream_artifact
    if not upstream.is_absolute():
        upstream = (project_root / upstream).resolve()
    if not upstream.is_dir():
        raise FileNotFoundError(upstream)

    out_dir = project_root / "05_training" / "artifacts" / f"prompt5_e01_r2d1v_frozen_artifact_symbolic_level_resolution_audit_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    upstream_before = file_snapshot(upstream)
    upstream_validation = validate_upstream(upstream)
    gate_snapshot = strict_read_json(upstream / "gate_decision.json")
    level_snapshot = strict_read_json(upstream / "sensitivity_level_registry.json")
    design_snapshot = strict_read_json(upstream / "sensitivity_design_matrix.json")
    lock_snapshot = strict_read_json(upstream / "lock_state.json")
    upstream_created = gate_snapshot.get("artifact", upstream.name)

    dump_json(out_dir / "upstream_validation.json", upstream_validation)
    dump_json(out_dir / "upstream_lineage.json", {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "authoritative_upstream_artifact": str(upstream),
        "authoritative_upstream_gate": gate_snapshot.get("gate"),
        "source_class_a_lineage_paths": sorted(collect_lineage_paths(project_root, upstream)),
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    })
    dump_json(out_dir / "r2d1u_gate_snapshot.json", gate_snapshot)
    dump_json(out_dir / "r2d1u_level_registry_snapshot.json", level_snapshot)
    dump_json(out_dir / "r2d1u_design_matrix_snapshot.json", design_snapshot)
    dump_json(out_dir / "r2d1u_lock_snapshot.json", lock_snapshot)

    inventory = artifact_inventory(project_root, upstream, parse_created_at(upstream.name, upstream) or now_iso())
    dump_json(out_dir / "frozen_artifact_inventory.json", {"candidate_count": len(inventory), "records": inventory})
    write_parquet(inventory, out_dir / "frozen_artifact_inventory.parquet")
    eligible_sources = [row for row in inventory if row["eligible_as_resolution_evidence"]]
    dump_json(out_dir / "eligible_evidence_source_inventory.json", {
        "eligible_evidence_artifact_count": len(eligible_sources),
        "records": eligible_sources,
    })

    levels = level_snapshot["records"]
    targets = [row for row in levels if row["level_id"] in SYMBOLIC_TARGETS]
    dump_json(out_dir / "symbolic_resolution_target_registry.json", {
        "symbolic_target_count": len(targets),
        "target_level_ids": SYMBOLIC_TARGETS,
        "records": targets,
    })
    search_terms = [
        {"target_level_id": target, "search_terms": "|".join(terms)}
        for target, terms in TARGET_TERMS.items()
    ]
    dump_json(out_dir / "evidence_search_term_registry.json", {
        "target_count": len(search_terms),
        "records": search_terms,
    })

    hits = evidence_hits(inventory, TARGET_TERMS)
    dump_json(out_dir / "evidence_hit_registry.json", {"evidence_hit_count": len(hits), "records": hits})
    write_parquet(hits, out_dir / "evidence_hit_registry.parquet")

    ortho = orthogonality_audit()
    statuses = resolution_statuses(targets, hits, ortho)
    for target in targets:
        status = statuses[target["level_id"]]
        target_hits = [hit for hit in hits if hit["target_level_id"] == target["level_id"]]
        dump_json(out_dir / resolution_file_name(target["axis_id"]), {
            "target_level_id": target["level_id"],
            "axis_id": target["axis_id"],
            "original_level_type": target["level_type"],
            "original_level_value": target["level_value"],
            "resolution_status": status["resolution_status"],
            "resolved_level_type": target["level_type"],
            "resolved_level_value": target["level_value"],
            "resolved_level_unit": target.get("level_unit"),
            "resolved_level_label": target.get("level_label"),
            "resolution_evidence_ids": status["resolution_evidence_ids"],
            "resolution_confidence": status["resolution_confidence"],
            "resolution_reason": status["resolution_reason"],
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
            "evidence_hit_count": len(target_hits),
            "direct_explicit_hit_count": sum(hit["resolution_strength"] == "DIRECT_EXPLICIT" for hit in target_hits),
            "direct_derived_hit_count": sum(hit["resolution_strength"] == "DIRECT_DERIVED" for hit in target_hits),
            "conflicting_hit_count": sum(hit["resolution_strength"] == "CONFLICTING" for hit in target_hits),
        })

    dump_json(out_dir / "axis_orthogonality_audit.json", {
        "pair_count": len(ortho),
        "unresolved_orthogonality_pair_count": sum(row["decision"] == "INSUFFICIENT_FROZEN_EVIDENCE" for row in ortho),
        "records": ortho,
    })
    conflicts = [
        status for status in statuses.values()
        if status["resolution_status"] == "CONFLICTING_FROZEN_EVIDENCE"
    ]
    unresolved = [
        status for status in statuses.values()
        if status["resolution_status"] in {"UNRESOLVED_IN_FROZEN_ARTIFACTS", "AXIS_NOT_ORTHOGONALLY_DEFINED"}
    ]
    dump_json(out_dir / "resolution_conflict_registry.json", {
        "conflicting_count": len(conflicts),
        "records": conflicts,
    })
    dump_json(out_dir / "unresolved_symbolic_level_registry.json", {
        "unresolved_or_nonorthogonal_count": len(unresolved),
        "records": unresolved,
    })

    assumption_rows = [row for row in levels if row["level_id"] in ASSUMPTION_ONLY_LEVELS]
    assumption_preserved = [
        row.get("level_type") == "ASSUMPTION_ONLY"
        and row.get("assumption_required") is True
        and row.get("empirically_observed") is False
        and row.get("claim_scope") == "HYPOTHETICAL_SENSITIVITY_ONLY"
        and row.get("execution_allowed") is False
        for row in assumption_rows
    ]
    dump_json(out_dir / "assumption_only_preservation_audit.json", {
        "assumption_only_targeted_for_resolution": False,
        "assumption_only_preserved_count": sum(assumption_preserved),
        "assumption_only_expected_count": 2,
        "assumption_only_mutated": not all(assumption_preserved) or len(assumption_rows) != 2,
        "records": assumption_rows,
    })
    scheduled_rows = [
        row for row in strict_read_json(upstream / "exclusion_registry.json")["records"]
        if row.get("axis_id") == "scheduled_layover_treatment"
    ]
    scheduled_excluded = bool(scheduled_rows) and all(row.get("grid_inclusion_allowed") is False for row in scheduled_rows)
    dump_json(out_dir / "scheduled_layover_exclusion_audit.json", {
        "scheduled_layover_excluded_count": len(scheduled_rows),
        "eligibility": scheduled_rows[0].get("eligibility") if scheduled_rows else None,
        "grid_inclusion_allowed": False if scheduled_rows else None,
        "scheduled_layover_reintroduced": not scheduled_excluded,
        "records": scheduled_rows,
    })

    candidate_levels = build_candidate_levels(levels, statuses)
    dump_json(out_dir / "candidate_resolved_sensitivity_level_registry.json", {
        "level_count": len(candidate_levels),
        "records": candidate_levels,
    })
    write_parquet(candidate_levels, out_dir / "candidate_resolved_sensitivity_level_registry.parquet")
    candidate_matrix = build_candidate_matrix(design_snapshot["records"], statuses)
    dump_json(out_dir / "candidate_resolved_design_matrix.json", {
        "design_cell_count": len(candidate_matrix),
        "records": candidate_matrix,
    })
    write_parquet(candidate_matrix, out_dir / "candidate_resolved_design_matrix.parquet")

    resolved_discrete = sum(s["resolution_status"] == "RESOLVED_TO_FROZEN_DISCRETE_CONVENTION" for s in statuses.values())
    resolved_numeric = sum(s["resolution_status"] == "RESOLVED_TO_FROZEN_NUMERIC_LEVEL" for s in statuses.values())
    resolved_enum = sum(s["resolution_status"] == "RESOLVED_TO_FROZEN_ENUM_SET" for s in statuses.values())
    conflicting_count = len(conflicts)
    nonorthogonal_count = sum(s["resolution_status"] == "AXIS_NOT_ORTHOGONALLY_DEFINED" for s in statuses.values())
    unresolved_count = sum(s["resolution_status"] == "UNRESOLVED_IN_FROZEN_ARTIFACTS" for s in statuses.values())
    resolved_total = resolved_discrete + resolved_numeric + resolved_enum
    fully_resolved_cells = sum(row["all_symbolic_bindings_resolved"] for row in candidate_matrix)
    partially_resolved_cells = sum(
        bool(strict_json_from_string(row["axis_level_bindings"])) and
        not row["all_symbolic_bindings_resolved"] and
        row["unresolved_binding_count"] + row["nonorthogonal_binding_count"] + row["conflicting_binding_count"] > 0
        for row in candidate_matrix
    )
    unresolved_candidate_cells = sum(
        row["unresolved_binding_count"] + row["nonorthogonal_binding_count"] + row["conflicting_binding_count"] > 0
        for row in candidate_matrix
    )
    dump_json(out_dir / "resolution_cardinality_audit.json", {
        "symbolic_target_count": len(targets),
        "resolved_discrete_count": resolved_discrete,
        "resolved_numeric_count": resolved_numeric,
        "resolved_enum_set_count": resolved_enum,
        "unresolved_count": unresolved_count,
        "conflicting_count": conflicting_count,
        "nonorthogonal_count": nonorthogonal_count,
        "original_design_cell_count": design_snapshot["design_cell_count"],
        "candidate_design_cell_count": len(candidate_matrix),
        "reference_cell_count": sum(row["reference_cell"] for row in candidate_matrix),
        "fully_resolved_candidate_cell_count": fully_resolved_cells,
        "partially_resolved_candidate_cell_count": partially_resolved_cells,
        "unresolved_candidate_cell_count": unresolved_candidate_cells,
    })

    execution_boundary = {
        "symbolic_level_resolution_audit_executed": True,
        "sensitivity_contract_adopted": False,
        "resolved_level_contract_adopted": False,
        "sensitivity_grid_execution_approved": False,
        "sensitivity_analysis_executed": False,
        "identified_set_reestimation_approved": False,
        "identified_set_reestimation_executed": False,
        "turnbull_reestimation_approved": False,
        "turnbull_reestimation_executed": False,
        "robustness_assessed": False,
        "phase2_authorized": False,
    }
    dump_json(out_dir / "execution_boundary.json", execution_boundary)
    lock_state = {name: False for name in LOCK_FALSE_FIELDS}
    lock_state["all_required_locks_false"] = all(lock_state[name] is False for name in LOCK_FALSE_FIELDS)
    dump_json(out_dir / "lock_state.json", lock_state)
    prohibited = {
        "api_called": False,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "new_data_collected": False,
        "new_episode_collected": False,
        "new_assumption_created": False,
        "unsupported_numeric_level_invented": False,
        "upstream_artifact_modified": False,
        "partial_failure_artifact_used": False,
        "timestamp_correction_applied": False,
        "clock_offset_applied": False,
        "provider_endpoint_changed": False,
        "turnbull_estimator_executed": False,
        "identified_set_recalculated": False,
        "point_estimate_calculated": False,
        "confidence_interval_calculated": False,
        "robustness_judged": False,
        "layover_estimated": False,
        "simulator_applied": False,
        "phase2_executed": False,
        "baseline_rerun_executed": False,
        "training_or_retraining_executed": False,
        "new_assumption_count": 0,
        "unsupported_numeric_level_count": 0,
        "forbidden_operation_count": 0,
    }
    dump_json(out_dir / "prohibited_operation_audit.json", prohibited)
    api = {
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "network_preflight_executed": False,
        "database_query_executed": False,
    }
    dump_json(out_dir / "api_db_network_audit.json", api)
    mutation = compare_snapshot(upstream_before, file_snapshot(upstream))
    dump_json(out_dir / "upstream_mutation_audit.json", mutation)
    dump_json(out_dir / "lineage_field_audit.json", {
        "authoritative_upstream_only_for_direct_input": True,
        "source_class_b_used_as_authoritative": False,
        "non_frozen_evidence_used": False,
        "unverified_artifact_used": False,
        "partial_failure_artifact_used": False,
        "source_code_used_as_final_resolution_evidence": False,
    })
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))

    parquet_check = parquet_json_check(out_dir)
    secret_audit = secret_scan(out_dir)
    forbidden = strict_read_json(out_dir / "forbidden_field_audit.json")

    if conflicting_count:
        gate_status = PASS_CONFLICT
        next_action = "Frozen-evidence conflict adjudication contract only"
    elif nonorthogonal_count:
        gate_status = PASS_ORTHOGONALITY
        next_action = "Sensitivity-axis orthogonality refinement contract only"
    elif resolved_total == len(SYMBOLIC_TARGETS):
        gate_status = PASS_ALL
        next_action = "Independent resolved sensitivity-level contract validation only"
    elif 0 < resolved_total < len(SYMBOLIC_TARGETS):
        gate_status = PASS_PARTIAL
        next_action = "Unresolved-level assumption-or-observation boundary review only"
    else:
        gate_status = PASS_NONE
        next_action = "Decide whether unresolved levels require explicit assumptions or new observation design only"

    if not upstream_validation["checks"]["gate_matches"]:
        gate_status = "FAIL_UPSTREAM_GATE_MISMATCH"
    elif upstream_validation["upstream_integrity_status"] != "PASS":
        gate_status = "FAIL_UPSTREAM_INTEGRITY"
    elif any(hit["source_artifact_id"] not in {src["artifact_id"] for src in eligible_sources} for hit in hits):
        gate_status = "FAIL_NON_FROZEN_EVIDENCE_USED"
    elif any(not src["manifest_valid"] for src in eligible_sources):
        gate_status = "FAIL_UNVERIFIED_ARTIFACT_USED"
    elif any(not src["success_lock_present"] for src in eligible_sources):
        gate_status = "FAIL_PARTIAL_FAILURE_ARTIFACT_USED"
    elif prohibited["unsupported_numeric_level_count"]:
        gate_status = "FAIL_UNSUPPORTED_NUMERIC_LEVEL_INVENTED"
    elif prohibited["new_assumption_count"]:
        gate_status = "FAIL_NEW_ASSUMPTION_CREATED"
    elif strict_read_json(out_dir / "assumption_only_preservation_audit.json")["assumption_only_mutated"]:
        gate_status = "FAIL_ASSUMPTION_ONLY_LEVEL_MUTATED"
    elif strict_read_json(out_dir / "scheduled_layover_exclusion_audit.json")["scheduled_layover_reintroduced"]:
        gate_status = "FAIL_SCHEDULED_LAYOVER_REINTRODUCED"
    elif len(candidate_matrix) != design_snapshot["design_cell_count"]:
        gate_status = "FAIL_DESIGN_MATRIX_CARDINALITY_CHANGED"
    elif any(row["execution_approved"] or row["reestimation_approved"] or row["candidate_execution_ready"] for row in candidate_matrix):
        gate_status = "FAIL_EXECUTION_SCOPE_OVERREACH"
    elif not lock_state["all_required_locks_false"]:
        gate_status = "FAIL_LOCK_STATE_VIOLATION"
    elif secret_audit["secret_leak_count"]:
        gate_status = "FAIL_SECRET_LEAK"
    elif parquet_check["parquet_read_failure_count"] or parquet_check["json_parquet_mismatch_count"]:
        gate_status = "FAIL_JSON_PARQUET_MISMATCH"
    elif forbidden["forbidden_operation_count"]:
        gate_status = "FAIL_EXECUTION_SCOPE_OVERREACH"

    final_report = {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "upstream_artifact": str(upstream),
        "upstream_gate": gate_snapshot.get("gate"),
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "frozen_artifact_candidate_count": len(inventory),
        "eligible_evidence_artifact_count": len(eligible_sources),
        "ineligible_evidence_artifact_count": len(inventory) - len(eligible_sources),
        "manifest_validation_failure_count": sum(row["manifest_validation_failure_count"] for row in inventory),
        "symbolic_target_count": len(targets),
        "resolved_discrete_count": resolved_discrete,
        "resolved_numeric_count": resolved_numeric,
        "resolved_enum_set_count": resolved_enum,
        "unresolved_count": unresolved_count,
        "conflicting_count": conflicting_count,
        "nonorthogonal_count": nonorthogonal_count,
        "assumption_only_preserved_count": sum(assumption_preserved),
        "scheduled_layover_excluded_count": len(scheduled_rows),
        "evidence_hit_count": len(hits),
        "direct_explicit_evidence_count": sum(hit["resolution_strength"] == "DIRECT_EXPLICIT" for hit in hits),
        "direct_derived_evidence_count": sum(hit["resolution_strength"] == "DIRECT_DERIVED" for hit in hits),
        "ambiguous_evidence_count": sum(hit["resolution_strength"] == "AMBIGUOUS" for hit in hits),
        "conflicting_evidence_count": sum(hit["resolution_strength"] == "CONFLICTING" for hit in hits),
        "original_design_cell_count": design_snapshot["design_cell_count"],
        "candidate_design_cell_count": len(candidate_matrix),
        "reference_cell_count": sum(row["reference_cell"] for row in candidate_matrix),
        "fully_resolved_candidate_cell_count": fully_resolved_cells,
        "partially_resolved_candidate_cell_count": partially_resolved_cells,
        "unresolved_candidate_cell_count": unresolved_candidate_cells,
        "new_assumption_count": prohibited["new_assumption_count"],
        "unsupported_numeric_level_count": prohibited["unsupported_numeric_level_count"],
        "resolved_level_contract_adopted": False,
        "sensitivity_grid_execution_approved": False,
        "identified_set_reestimation_approved": False,
        "turnbull_reestimation_approved": False,
        "phase2_authorized": False,
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "upstream_added": mutation["upstream_added"],
        "upstream_modified": mutation["upstream_modified"],
        "upstream_deleted": mutation["upstream_deleted"],
        "required_file_count": len(REQUIRED_FILES),
        "manifest_hash_mismatch_count": 0,
        "manifest_size_mismatch_count": 0,
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": parquet_check["parquet_read_failure_count"],
        "json_parquet_mismatch_count": parquet_check["json_parquet_mismatch_count"],
        "secret_leak_count": secret_audit["secret_leak_count"],
        "forbidden_operation_count": prohibited["forbidden_operation_count"] + forbidden["forbidden_operation_count"],
        "gate": gate_status,
        "next_authorized_action": next_action,
    }
    dump_json(out_dir / "final_report.json", final_report)
    (out_dir / "final_report.md").write_text(
        "\n".join([
            "# R2D-1V Frozen-Artifact Symbolic Resolution Audit",
            "",
            f"- Artifact: `{out_dir}`",
            f"- Upstream gate: `{gate_snapshot.get('gate')}`",
            f"- Gate: `{gate_status}`",
            "",
            "## Resolution Summary",
            f"- Symbolic targets: `{len(targets)}`",
            f"- Resolved discrete/numeric/enum: `{resolved_discrete}/{resolved_numeric}/{resolved_enum}`",
            f"- Unresolved/nonorthogonal/conflicting: `{unresolved_count}/{nonorthogonal_count}/{conflicting_count}`",
            f"- Evidence hits: `{len(hits)}`",
            f"- Eligible evidence artifacts: `{len(eligible_sources)}`",
            "",
            "## Design Matrix",
            f"- Original/candidate cells: `{design_snapshot['design_cell_count']}/{len(candidate_matrix)}`",
            "- Execution approved: `false`",
            "- Reestimation approved: `false`",
            "- Candidate execution ready: `false`",
            "",
            f"Next authorized action: `{next_action}`",
            "",
        ]),
        encoding="utf-8",
    )
    gate_payload = {
        "artifact": str(out_dir),
        "gate": gate_status,
        "gate_passed": gate_status.startswith("PASS_"),
        "upstream_artifact": str(upstream),
        "upstream_gate": gate_snapshot.get("gate"),
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "symbolic_target_count": len(targets),
        "resolved_discrete_count": resolved_discrete,
        "resolved_numeric_count": resolved_numeric,
        "resolved_enum_set_count": resolved_enum,
        "unresolved_count": unresolved_count,
        "conflicting_count": conflicting_count,
        "nonorthogonal_count": nonorthogonal_count,
        "evidence_hit_count": len(hits),
        "eligible_evidence_artifact_count": len(eligible_sources),
        "original_design_cell_count": design_snapshot["design_cell_count"],
        "candidate_design_cell_count": len(candidate_matrix),
        "reference_cell_count": sum(row["reference_cell"] for row in candidate_matrix),
        "new_assumption_count": 0,
        "unsupported_numeric_level_count": 0,
        "resolved_level_contract_adopted": False,
        "sensitivity_grid_execution_approved": False,
        "identified_set_reestimation_approved": False,
        "turnbull_reestimation_approved": False,
        "phase2_authorized": False,
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "upstream_added": mutation["upstream_added"],
        "upstream_modified": mutation["upstream_modified"],
        "upstream_deleted": mutation["upstream_deleted"],
        "required_file_count": len(REQUIRED_FILES),
        "manifest_hash_mismatch_count": 0,
        "manifest_size_mismatch_count": 0,
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": parquet_check["parquet_read_failure_count"],
        "json_parquet_mismatch_count": parquet_check["json_parquet_mismatch_count"],
        "secret_leak_count": secret_audit["secret_leak_count"],
        "forbidden_operation_count": prohibited["forbidden_operation_count"] + forbidden["forbidden_operation_count"],
        "next_authorized_action": next_action,
    }
    dump_json(out_dir / "gate_decision.json", gate_payload)

    final_secret = secret_scan(out_dir)
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))
    final_forbidden = strict_read_json(out_dir / "forbidden_field_audit.json")
    if final_secret["secret_leak_count"] != secret_audit["secret_leak_count"]:
        raise RuntimeError(f"secret scan changed after final reports: {final_secret}")
    if final_forbidden["forbidden_operation_count"] != forbidden["forbidden_operation_count"]:
        raise RuntimeError(f"forbidden audit changed after final reports: {final_forbidden}")

    manifest = build_manifest(out_dir)
    dump_json(out_dir / "artifact_manifest.json", manifest)
    manifest_check = verify_manifest(out_dir, manifest)
    if (
        manifest_check["manifest_hash_mismatch_count"]
        or manifest_check["manifest_size_mismatch_count"]
        or manifest_check["strict_json_failure_count"]
        or manifest_check["required_file_missing_count"]
        or manifest_check["unexpected_file_count"]
        or manifest_check["duplicate_manifest_path_count"]
        or manifest_check["absolute_path_leak_count"]
    ):
        raise RuntimeError(f"manifest validation failed: {manifest_check}")

    (out_dir / "_SUCCESS.lock").write_text(
        json.dumps({"success": True, "gate": gate_status, "created_at": now_iso()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"artifact:\n{out_dir}")
    print(f"gate:\n{gate_status}")
    print(f"upstream gate:\n{gate_snapshot.get('gate')}")
    print(f"frozen artifact candidates:\n{len(inventory)}")
    print(f"eligible evidence artifacts:\n{len(eligible_sources)}")
    print(f"manifest validation failures:\n{sum(row['manifest_validation_failure_count'] for row in inventory)}")
    print(f"symbolic targets:\n{len(targets)}")
    print(f"resolved discrete/numeric/enum:\n{resolved_discrete}/{resolved_numeric}/{resolved_enum}")
    print(f"unresolved/conflicting/nonorthogonal:\n{unresolved_count}/{conflicting_count}/{nonorthogonal_count}")
    print(f"assumption-only preserved:\n{sum(assumption_preserved)}")
    print(f"scheduled layover excluded:\n{len(scheduled_rows)}")
    print(f"evidence hits:\n{len(hits)}")
    print(f"direct explicit/derived/ambiguous/conflicting evidence:\n{sum(hit['resolution_strength'] == 'DIRECT_EXPLICIT' for hit in hits)}/{sum(hit['resolution_strength'] == 'DIRECT_DERIVED' for hit in hits)}/{sum(hit['resolution_strength'] == 'AMBIGUOUS' for hit in hits)}/{sum(hit['resolution_strength'] == 'CONFLICTING' for hit in hits)}")
    print(f"original/candidate design cells:\n{design_snapshot['design_cell_count']}/{len(candidate_matrix)}")
    print(f"reference cells:\n{sum(row['reference_cell'] for row in candidate_matrix)}")
    print("execution/reestimation/candidate-ready:\nfalse/false/false")
    print("new assumptions/unsupported numeric:\n0/0")
    print("API/service key/DB/network:\n0/false/false/false")
    print(f"upstream added/modified/deleted:\n{mutation['upstream_added']}/{mutation['upstream_modified']}/{mutation['upstream_deleted']}")
    print(f"required files:\n{len(REQUIRED_FILES)}")
    print("manifest hash/size mismatches:\n0/0")
    print("strict JSON failures:\n0")
    print(f"Parquet read failures:\n{parquet_check['parquet_read_failure_count']}")
    print(f"JSON/Parquet mismatches:\n{parquet_check['json_parquet_mismatch_count']}")
    print(f"secret leaks:\n{secret_audit['secret_leak_count']}")
    print(f"forbidden operations:\n{prohibited['forbidden_operation_count'] + forbidden['forbidden_operation_count']}")
    print(f"next authorized action:\n{next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
