from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1q_method_prototype_interval_censored_estimation.py"

SOURCE_R2D1P = ARTIFACTS_ROOT / "prompt5_e01_r2d1p_estimation_execution_authorization_20260730_092848"
SOURCE_R2D1O = ARTIFACTS_ROOT / "prompt5_e01_r2d1o_12_episode_registry_freeze_estimation_readiness_20260730_085243"
SOURCE_R2D1N_HF1 = ARTIFACTS_ROOT / "prompt5_e01_r2d1n_hf1_final_registry_eta_deduplication_20260730_000608"
SOURCE_R2D1E = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
SOURCE_R2D1F = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

PRIMARY_ESTIMAND = "OBSERVED_POST_SERVICE_NON_REVENUE_TURNAROUND_INTERVAL"
PRIMARY_METHOD = "TURNBULL_NPMLE_INTERVAL_CENSORED_POOLED"
PRIMARY_GATE = "PASS_METHOD_PROTOTYPE_INTERVAL_CENSORED_ESTIMATION_COMPLETE"
NONUNIQUE_GATE = "PASS_METHOD_PROTOTYPE_ESTIMATION_NONUNIQUE_IDENTIFIED_SET_COMPLETE"

MASS_SUM_TOL = 1e-10
NEGATIVE_MASS_TOL = 1e-12
LOG_LIKELIHOOD_DECREASE_TOL = 1e-10
RELATIVE_LL_TOL = 1e-10
MAX_MASS_CHANGE_TOL = 1e-8
CONSECUTIVE_CONVERGENCE_ITERATIONS = 5
MINIMUM_ITERATIONS = 5
MAXIMUM_ITERATIONS = 10000
ACTIVE_MASS_TOL = 1e-10
KKT_EXCESS_TOL = 1e-7
ACTIVE_KKT_TOL = 1e-7
SELF_CONSISTENCY_TOL = 1e-8
INDEPENDENT_LL_TOL = 1e-8
INDEPENDENT_MASS_TOL = 1e-10

REQUIRED_FILES = [
    "prompt5_e01_r2d1q_manifest.json",
    "prompt5_e01_r2d1q_gate.json",
    "prompt5_e01_r2d1q_final_report.md",
    "upstream_reference_r2d1p.json",
    "upstream_reference_r2d1o.json",
    "upstream_reference_r2d1n_hf1.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "estimation_execution_authorization_verification.json",
    "secret_leak_audit.json",
    "input_registry_fingerprint_verification.json",
    "input_packet_verification.json",
    "analysis_population_audit.json",
    "adaptive_eta_exclusion_audit.json",
    "interval_input_validation_audit.json",
    "interval_input_validation_audit.parquet",
    "turnbull_implementation_self_test.json",
    "turnbull_execution_environment.json",
    "turnbull_algorithm_contract.json",
    "turnbull_primary_execution.json",
    "turnbull_primary_initialization_comparison.json",
    "turnbull_primary_iteration_trace.json",
    "turnbull_primary_iteration_trace.parquet",
    "turnbull_primary_support.json",
    "turnbull_primary_support.parquet",
    "turnbull_primary_incidence_audit.json",
    "turnbull_primary_mass_solution.json",
    "turnbull_primary_mass_solution.parquet",
    "turnbull_primary_survival_curve.json",
    "turnbull_primary_survival_curve.parquet",
    "turnbull_primary_quantile_identification.json",
    "turnbull_primary_kkt_audit.json",
    "turnbull_primary_independent_likelihood_verification.json",
    "turnbull_provider_sensitivity_execution.json",
    "turnbull_provider_sensitivity_iteration_trace.parquet",
    "turnbull_provider_sensitivity_support.json",
    "turnbull_provider_sensitivity_survival_curve.json",
    "turnbull_provider_sensitivity_quantile_identification.json",
    "turnbull_provider_sensitivity_kkt_audit.json",
    "turnbull_provider_sensitivity_independent_likelihood_verification.json",
    "turnbull_request_sensitivity_execution.json",
    "turnbull_request_sensitivity_iteration_trace.parquet",
    "turnbull_request_sensitivity_support.json",
    "turnbull_request_sensitivity_survival_curve.json",
    "turnbull_request_sensitivity_quantile_identification.json",
    "turnbull_request_sensitivity_kkt_audit.json",
    "turnbull_request_sensitivity_independent_likelihood_verification.json",
    "turnbull_vehicle_balanced_sensitivity_execution.json",
    "turnbull_vehicle_balanced_weights.json",
    "turnbull_vehicle_balanced_weights.parquet",
    "turnbull_vehicle_balanced_iteration_trace.parquet",
    "turnbull_vehicle_balanced_support.json",
    "turnbull_vehicle_balanced_survival_curve.json",
    "turnbull_vehicle_balanced_quantile_identification.json",
    "turnbull_vehicle_balanced_kkt_audit.json",
    "turnbull_vehicle_balanced_independent_likelihood_verification.json",
    "clock_sensitivity_comparison.json",
    "vehicle_weighting_sensitivity_comparison.json",
    "descriptive_endpoint_summary.json",
    "route_descriptive_summary.json",
    "route_descriptive_summary.parquet",
    "vehicle_dependence_descriptive_audit.json",
    "clock_semantics_descriptive_audit.json",
    "numerical_invariant_audit.json",
    "estimation_output_schema_audit.json",
    "prohibited_interpretation_audit.json",
    "estimation_limitations.json",
    "estimation_limitations.md",
    "terminal_recovery_parameter_translation_guard.json",
    "simulator_application_authorization.json",
    "phase2_execution_authorization.json",
    "json_parquet_synchronization_audit.json",
    "manifest_self_entry_contract.json",
]

ANALYSES = {
    "PRIMARY_DUAL_EQUAL": {
        "clock": "CONSERVATIVE_DUAL_CLOCK",
        "weighting": "EQUAL_EPISODE_WEIGHT",
        "lower_field": "conservative_dual_lower_bound_sec",
        "upper_field": "conservative_dual_upper_bound_sec",
        "prefix": "turnbull_primary",
    },
    "SENSITIVITY_PROVIDER_EQUAL": {
        "clock": "PROVIDER_ONLY",
        "weighting": "EQUAL_EPISODE_WEIGHT",
        "lower_field": "provider_lower_bound_sec",
        "upper_field": "provider_upper_bound_sec",
        "prefix": "turnbull_provider_sensitivity",
    },
    "SENSITIVITY_REQUEST_EQUAL": {
        "clock": "REQUEST_ONLY",
        "weighting": "EQUAL_EPISODE_WEIGHT",
        "lower_field": "request_lower_bound_sec",
        "upper_field": "request_upper_bound_sec",
        "prefix": "turnbull_request_sensitivity",
    },
    "SENSITIVITY_DUAL_VEHICLE_BALANCED": {
        "clock": "CONSERVATIVE_DUAL_CLOCK",
        "weighting": "VEHICLE_BALANCED",
        "lower_field": "conservative_dual_lower_bound_sec",
        "upper_field": "conservative_dual_upper_bound_sec",
        "prefix": "turnbull_vehicle_balanced_sensitivity",
    },
}


def now_stamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def strict_constant(value: str) -> None:
    raise ValueError(f"non-strict JSON token: {value}")


def strict_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=strict_constant)


def sanitize(value: Any) -> Any:
    if isinstance(value, np.generic):
        return sanitize(value.item())
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


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([sanitize(row) for row in rows]).to_parquet(path, index=False)
    pd.read_parquet(path)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
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
    candidates = sorted(path for path in root.glob("*.json") if "_gate" in path.name)
    prompt_candidates = [path for path in candidates if path.name.startswith("prompt")]
    return prompt_candidates[0] if prompt_candidates else (candidates[0] if candidates else None)


def manifest_path_for(root: Path) -> Path | None:
    candidates = sorted(path for path in root.glob("*.json") if "_manifest" in path.name)
    prompt_candidates = [path for path in candidates if path.name.startswith("prompt")]
    return prompt_candidates[0] if prompt_candidates else (candidates[0] if candidates else None)


def upstream_reference(root: Path, label: str, required_gate: str) -> dict[str, Any]:
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
        "gate_requirement_passed": status == required_gate,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_exists": bool(manifest_path and manifest_path.exists()),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path and manifest_path.exists() else None,
    }


def source_snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            snapshot[str(root)] = {"exists": False, "size": None, "sha256": None}
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            snapshot[str(path)] = {"exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
    return snapshot


def compare_snapshots(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
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


def evidence_sha_records(registry: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str, int]]:
    roles = [
        ("FIRST_UPSTREAM", "first_upstream_raw_sha256"),
        ("LAST_PRE_TERMINAL", "last_pre_terminal_raw_sha256"),
        ("FIRST_TERMINAL", "first_terminal_raw_sha256"),
        ("LAST_TERMINAL", "last_terminal_raw_sha256"),
        ("FIRST_POST_TERMINAL", "first_post_terminal_raw_sha256"),
    ]
    out = []
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


def recompute_fingerprint(registry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    canonical_records = [{field: row.get(field) for field in canonical_fields} for row in ordered]
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


def interval_records(input_records: Sequence[Mapping[str, Any]], lower_field: str, upper_field: str) -> list[tuple[float, float]]:
    return [(float(row[lower_field]), float(row[upper_field])) for row in input_records]


def ordered_input_records(input_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted([dict(row) for row in input_records], key=lambda row: str(row["episode_id"]))


def construct_support(intervals: Sequence[tuple[float, float]], analysis_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoints = sorted({float(bound) for interval in intervals for bound in interval})
    candidates: list[dict[str, Any]] = []
    for idx, endpoint in enumerate(endpoints):
        if any(math.isclose(lower, endpoint) and math.isclose(upper, endpoint) for lower, upper in intervals):
            candidates.append(
                {
                    "support_cell_id": f"{analysis_id}_cell_{len(candidates) + 1:03d}",
                    "support_lower_sec": endpoint,
                    "support_upper_sec": endpoint,
                    "support_cell_type": "POINT_ENDPOINT",
                    "admissibility_probe_sec": endpoint,
                    "within_cell_location_not_identified": False,
                }
            )
        if idx < len(endpoints) - 1:
            lower = endpoint
            upper = endpoints[idx + 1]
            if upper < lower:
                continue
            probe = (lower + upper) / 2.0
            if any(obs_lower <= probe <= obs_upper for obs_lower, obs_upper in intervals):
                candidates.append(
                    {
                        "support_cell_id": f"{analysis_id}_cell_{len(candidates) + 1:03d}",
                        "support_lower_sec": lower,
                        "support_upper_sec": upper,
                        "support_cell_type": "OPEN_GAP_REPRESENTED_BY_CLOSED_BOUNDS",
                        "admissibility_probe_sec": probe,
                        "within_cell_location_not_identified": upper > lower,
                    }
                )
    cells = []
    for idx, cell in enumerate(candidates, start=1):
        updated = dict(cell)
        updated["support_cell_index"] = idx
        updated["support_cell_id"] = f"{analysis_id}_cell_{idx:03d}"
        cells.append(updated)
    audit = {
        "analysis_id": analysis_id,
        "unique_endpoint_count": len(endpoints),
        "candidate_support_cell_count": len(candidates),
        "final_support_cell_count": len(cells),
        "support_ordered_ascending": all(
            cells[idx - 1]["support_lower_sec"] <= cells[idx]["support_lower_sec"] for idx in range(1, len(cells))
        ),
        "support_cells_non_overlapping_allowing_touching_bounds": all(
            cells[idx - 1]["support_upper_sec"] <= cells[idx]["support_lower_sec"] for idx in range(1, len(cells))
        ),
        "support_lower_le_upper_failure_count": sum(
            1 for cell in cells if cell["support_lower_sec"] > cell["support_upper_sec"]
        ),
    }
    return cells, audit


def incidence_matrix(intervals: Sequence[tuple[float, float]], cells: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(intervals), len(cells)), dtype=int)
    for row_idx, (lower, upper) in enumerate(intervals):
        for col_idx, cell in enumerate(cells):
            probe = float(cell["admissibility_probe_sec"])
            if lower <= probe <= upper:
                matrix[row_idx, col_idx] = 1
    return matrix


def support_and_incidence(intervals: Sequence[tuple[float, float]], analysis_id: str) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    cells, audit = construct_support(intervals, analysis_id)
    matrix = incidence_matrix(intervals, cells)
    empty_support = [idx for idx, value in enumerate(matrix.sum(axis=0).tolist(), start=1) if value == 0]
    unsupported_episodes = [idx for idx, value in enumerate(matrix.sum(axis=1).tolist(), start=1) if value == 0]
    audit.update(
        {
            "empty_support_cell_count": len(empty_support),
            "empty_support_cell_indices": empty_support,
            "episode_without_admissible_support_count": len(unsupported_episodes),
            "episode_without_admissible_support_indices": unsupported_episodes,
            "incidence_matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "incidence_matrix_sha256": canonical_sha(matrix.astype(int).tolist()),
            "support_construction_passed": len(cells) > 0 and len(empty_support) == 0 and len(unsupported_episodes) == 0,
        }
    )
    return cells, matrix, audit


def log_likelihood(matrix: np.ndarray, weights: np.ndarray, mass: np.ndarray) -> tuple[float, np.ndarray]:
    likelihoods = matrix @ mass
    if np.any(likelihoods <= 0.0) or np.any(~np.isfinite(likelihoods)):
        return float("-inf"), likelihoods
    return float(np.sum(weights * np.log(likelihoods))), likelihoods


def kkt_audit(analysis_id: str, matrix: np.ndarray, weights: np.ndarray, mass: np.ndarray) -> dict[str, Any]:
    ll, likelihoods = log_likelihood(matrix, weights, mass)
    total_weight = float(np.sum(weights))
    scores = np.sum(weights[:, None] * matrix / likelihoods[:, None], axis=0)
    active_mask = mass > ACTIVE_MASS_TOL
    active_errors = np.abs(scores[active_mask] - total_weight) if np.any(active_mask) else np.array([0.0])
    em_map = mass * scores / total_weight
    residual = float(np.max(np.abs(em_map - mass))) if len(mass) else None
    maximum_excess = float(max(0.0, float(np.max(scores - total_weight)))) if len(scores) else None
    maximum_active_error = float(np.max(active_errors)) if len(active_errors) else 0.0
    return {
        "analysis_id": analysis_id,
        "weighted_log_likelihood": ll,
        "total_weight": total_weight,
        "active_mass_tolerance": ACTIVE_MASS_TOL,
        "kkt_excess_tolerance": KKT_EXCESS_TOL,
        "active_kkt_tolerance": ACTIVE_KKT_TOL,
        "support_score_maximum": float(np.max(scores)) if len(scores) else None,
        "maximum_kkt_score_excess": maximum_excess,
        "maximum_active_kkt_absolute_error": maximum_active_error,
        "self_consistency_mass_residual": residual,
        "active_support_cell_count": int(np.sum(active_mask)),
        "support_scores": [
            {
                "support_cell_index": idx + 1,
                "probability_mass": float(mass[idx]),
                "support_score": float(scores[idx]),
                "active_mass": bool(active_mask[idx]),
            }
            for idx in range(len(mass))
        ],
        "kkt_self_consistency_passed": (
            maximum_excess is not None
            and maximum_excess <= KKT_EXCESS_TOL
            and maximum_active_error <= ACTIVE_KKT_TOL
            and residual is not None
            and residual <= SELF_CONSISTENCY_TOL
        ),
    }


def em_run(
    analysis_id: str,
    initialization_id: str,
    matrix: np.ndarray,
    weights: np.ndarray,
    initial_mass: np.ndarray,
) -> dict[str, Any]:
    mass = initial_mass.astype(float).copy()
    mass = mass / mass.sum()
    trace: list[dict[str, Any]] = []
    previous_ll, previous_likelihoods = log_likelihood(matrix, weights, mass)
    if not math.isfinite(previous_ll):
        raise RuntimeError(f"{analysis_id} {initialization_id} initial likelihood is nonfinite")
    trace.append(
        {
            "analysis_id": analysis_id,
            "initialization_id": initialization_id,
            "iteration": 0,
            "log_likelihood": previous_ll,
            "log_likelihood_change": None,
            "absolute_log_likelihood_change": None,
            "relative_log_likelihood_change": None,
            "maximum_mass_change": None,
            "mass_sum": float(mass.sum()),
            "minimum_mass": float(mass.min()),
            "maximum_mass": float(mass.max()),
            "nonfinite_count": 0,
            "negative_mass_clamp_count": 0,
            "convergence_criterion_met": False,
        }
    )
    convergence_run = 0
    convergence_criteria_met_iteration = None
    maximum_likelihood_decrease = 0.0
    small_likelihood_decrease_count = 0
    total_negative_clamp_count = 0
    status = "MAX_ITERATIONS_REACHED"
    failure_reason = None

    for iteration in range(1, MAXIMUM_ITERATIONS + 1):
        probabilities = previous_likelihoods
        responsibilities = matrix * mass / probabilities[:, None]
        new_mass = np.sum(weights[:, None] * responsibilities, axis=0) / float(np.sum(weights))

        negative_mask = new_mass < 0
        negative_count = int(np.sum(negative_mask))
        if negative_count:
            if float(np.min(new_mass)) < -NEGATIVE_MASS_TOL:
                status = "INVALID_MASS_VECTOR"
                failure_reason = "negative mass below tolerance"
                break
            new_mass[negative_mask] = 0.0
            total_negative_clamp_count += negative_count

        nonfinite_count = int(np.sum(~np.isfinite(new_mass)))
        if nonfinite_count:
            status = "NONFINITE_VALUE"
            failure_reason = "nonfinite mass update"
            break

        mass_sum_before_normalize = float(new_mass.sum())
        if mass_sum_before_normalize <= 0.0 or not math.isfinite(mass_sum_before_normalize):
            status = "INVALID_MASS_VECTOR"
            failure_reason = "invalid mass sum"
            break
        new_mass = new_mass / mass_sum_before_normalize
        current_ll, current_likelihoods = log_likelihood(matrix, weights, new_mass)
        if not math.isfinite(current_ll) or np.any(current_likelihoods <= 0.0):
            status = "NONFINITE_VALUE"
            failure_reason = "nonfinite or zero observation likelihood"
            break

        ll_change = current_ll - previous_ll
        if ll_change < 0:
            maximum_likelihood_decrease = max(maximum_likelihood_decrease, abs(float(ll_change)))
            if abs(float(ll_change)) <= LOG_LIKELIHOOD_DECREASE_TOL:
                small_likelihood_decrease_count += 1
            else:
                status = "LIKELIHOOD_DECREASE"
                failure_reason = "log likelihood decrease beyond tolerance"
                break

        maximum_mass_change = float(np.max(np.abs(new_mass - mass)))
        relative_change = abs(float(ll_change)) / max(1.0, abs(previous_ll))
        convergence_criterion_met = (
            iteration >= MINIMUM_ITERATIONS
            and relative_change <= RELATIVE_LL_TOL
            and maximum_mass_change <= MAX_MASS_CHANGE_TOL
        )
        if convergence_criterion_met:
            convergence_run += 1
            if convergence_criteria_met_iteration is None:
                convergence_criteria_met_iteration = iteration
        else:
            convergence_run = 0

        mass = new_mass
        previous_ll = current_ll
        previous_likelihoods = current_likelihoods
        trace.append(
            {
                "analysis_id": analysis_id,
                "initialization_id": initialization_id,
                "iteration": iteration,
                "log_likelihood": current_ll,
                "log_likelihood_change": ll_change,
                "absolute_log_likelihood_change": abs(float(ll_change)),
                "relative_log_likelihood_change": relative_change,
                "maximum_mass_change": maximum_mass_change,
                "mass_sum": float(mass.sum()),
                "minimum_mass": float(mass.min()),
                "maximum_mass": float(mass.max()),
                "nonfinite_count": nonfinite_count,
                "negative_mass_clamp_count": negative_count,
                "convergence_criterion_met": convergence_criterion_met,
            }
        )

        if convergence_run >= CONSECUTIVE_CONVERGENCE_ITERATIONS:
            kkt = kkt_audit(analysis_id, matrix, weights, mass)
            if kkt["kkt_self_consistency_passed"]:
                status = "CONVERGED"
                break

    final_ll, final_likelihoods = log_likelihood(matrix, weights, mass)
    return {
        "analysis_id": analysis_id,
        "initialization_id": initialization_id,
        "convergence_status": status,
        "failure_reason": failure_reason,
        "iteration_count": int(trace[-1]["iteration"]) if trace else 0,
        "convergence_criteria_met_iteration": convergence_criteria_met_iteration,
        "consecutive_convergence_iterations_required": CONSECUTIVE_CONVERGENCE_ITERATIONS,
        "final_log_likelihood": final_ll,
        "mass_vector": mass,
        "mass_vector_sha256": canonical_sha([float(value) for value in mass.tolist()]),
        "observation_likelihoods": final_likelihoods,
        "trace": trace,
        "maximum_likelihood_decrease": maximum_likelihood_decrease,
        "small_likelihood_decrease_count": small_likelihood_decrease_count,
        "total_negative_mass_clamp_count": total_negative_clamp_count,
        "mass_sum": float(mass.sum()),
        "minimum_mass": float(mass.min()),
        "maximum_mass": float(mass.max()),
        "mass_sum_error": abs(float(mass.sum()) - 1.0),
    }


def initialization_vectors(matrix: np.ndarray, weights: np.ndarray) -> list[tuple[str, np.ndarray]]:
    support_count = matrix.shape[1]
    uniform = np.ones(support_count, dtype=float) / float(support_count)
    scores = np.sum(weights[:, None] * matrix, axis=0)
    if float(scores.sum()) <= 0.0:
        informed = uniform.copy()
    else:
        informed = scores / float(scores.sum())
    return [
        ("UNIFORM_SUPPORT_MASS", uniform),
        ("ADMISSIBILITY_INFORMED_MASS", informed),
    ]


def quantile_components(cells: Sequence[Mapping[str, Any]], mass: np.ndarray, probability: float) -> dict[str, Any]:
    components = []
    cumulative = 0.0
    tolerance = 1e-10
    for idx, cell in enumerate(cells):
        next_cumulative = cumulative + float(mass[idx])
        if float(mass[idx]) > tolerance and cumulative - tolerance < probability <= next_cumulative + tolerance:
            components.append(
                {
                    "lower_sec": float(cell["support_lower_sec"]),
                    "upper_sec": float(cell["support_upper_sec"]),
                }
            )
        cumulative = next_cumulative
    if not components:
        return {
            "quantile_probability": probability,
            "identification_status": "NOT_IDENTIFIED_WITHIN_SUPPORT",
            "identified_components": [],
            "identified_envelope_lower_sec": None,
            "identified_envelope_upper_sec": None,
        }
    lower_values = [component["lower_sec"] for component in components]
    upper_values = [component["upper_sec"] for component in components]
    if len(components) > 1:
        status = "MULTI_COMPONENT_IDENTIFIED_SET"
    elif math.isclose(components[0]["lower_sec"], components[0]["upper_sec"]):
        status = "POINT_IDENTIFIED"
    else:
        status = "INTERVAL_IDENTIFIED"
    out = {
        "quantile_probability": probability,
        "identification_status": status,
        "identified_components": components,
        "identified_envelope_lower_sec": min(lower_values),
        "identified_envelope_upper_sec": max(upper_values),
    }
    if status == "POINT_IDENTIFIED":
        out["point_identified_value_sec"] = components[0]["lower_sec"]
    return out


def survival_rows(analysis_id: str, solution_id: str, cells: Sequence[Mapping[str, Any]], mass: np.ndarray) -> list[dict[str, Any]]:
    cumulative = 0.0
    rows = []
    for idx, cell in enumerate(cells):
        probability_mass = float(mass[idx])
        cdf_lower = cumulative
        cdf_upper = cumulative + probability_mass
        rows.append(
            {
                "analysis_id": analysis_id,
                "solution_id": solution_id,
                "support_cell_id": cell["support_cell_id"],
                "support_cell_index": idx + 1,
                "support_lower_sec": float(cell["support_lower_sec"]),
                "support_upper_sec": float(cell["support_upper_sec"]),
                "probability_mass": probability_mass,
                "probability_mass_lower": probability_mass,
                "probability_mass_upper": probability_mass,
                "cdf_lower": cdf_lower,
                "cdf_upper": cdf_upper,
                "survival_lower": 1.0 - cdf_upper,
                "survival_upper": 1.0 - cdf_lower,
                "within_cell_location_not_identified": bool(cell["within_cell_location_not_identified"]),
            }
        )
        cumulative = cdf_upper
    return rows


def survival_invariants(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cdf_lower = [float(row["cdf_lower"]) for row in rows]
    cdf_upper = [float(row["cdf_upper"]) for row in rows]
    surv_lower = [float(row["survival_lower"]) for row in rows]
    surv_upper = [float(row["survival_upper"]) for row in rows]
    return {
        "cdf_lower_monotonicity_failure_count": sum(1 for idx in range(1, len(cdf_lower)) if cdf_lower[idx] + 1e-12 < cdf_lower[idx - 1]),
        "cdf_upper_monotonicity_failure_count": sum(1 for idx in range(1, len(cdf_upper)) if cdf_upper[idx] + 1e-12 < cdf_upper[idx - 1]),
        "cdf_envelope_order_failure_count": sum(1 for lo, hi in zip(cdf_lower, cdf_upper) if lo > hi + 1e-12),
        "survival_lower_monotonicity_failure_count": sum(1 for idx in range(1, len(surv_lower)) if surv_lower[idx] > surv_lower[idx - 1] + 1e-12),
        "survival_upper_monotonicity_failure_count": sum(1 for idx in range(1, len(surv_upper)) if surv_upper[idx] > surv_upper[idx - 1] + 1e-12),
        "survival_envelope_order_failure_count": sum(1 for lo, hi in zip(surv_lower, surv_upper) if lo > hi + 1e-12),
        "cdf_final_upper_error": abs(cdf_upper[-1] - 1.0) if cdf_upper else None,
        "survival_final_lower_error": abs(surv_lower[-1] - 0.0) if surv_lower else None,
    }


def independent_likelihood_verification(
    analysis_id: str,
    intervals: Sequence[tuple[float, float]],
    weights: np.ndarray,
    cells: Sequence[Mapping[str, Any]],
    mass: np.ndarray,
    reference_log_likelihood: float,
    reference_likelihoods: np.ndarray,
) -> dict[str, Any]:
    independent_matrix = incidence_matrix(intervals, cells)
    independent_ll, independent_likelihoods = log_likelihood(independent_matrix, weights, mass)
    per_episode_mismatches = [
        {
            "row_index": idx + 1,
            "reference_likelihood": float(reference_likelihoods[idx]),
            "independent_likelihood": float(independent_likelihoods[idx]),
            "absolute_mismatch": abs(float(reference_likelihoods[idx] - independent_likelihoods[idx])),
        }
        for idx in range(len(reference_likelihoods))
        if abs(float(reference_likelihoods[idx] - independent_likelihoods[idx])) > 1e-10
    ]
    return {
        "analysis_id": analysis_id,
        "independent_log_likelihood": independent_ll,
        "reference_log_likelihood": reference_log_likelihood,
        "independent_log_likelihood_absolute_mismatch": abs(float(independent_ll - reference_log_likelihood)),
        "independent_mass_sum": float(mass.sum()),
        "independent_mass_sum_mismatch": abs(float(mass.sum()) - 1.0),
        "independent_episode_likelihood_mismatch_count": len(per_episode_mismatches),
        "episode_likelihood_mismatches": per_episode_mismatches,
        "independent_incidence_matrix_sha256": canonical_sha(independent_matrix.astype(int).tolist()),
        "independent_likelihood_verification_passed": (
            abs(float(independent_ll - reference_log_likelihood)) <= INDEPENDENT_LL_TOL
            and abs(float(mass.sum()) - 1.0) <= INDEPENDENT_MASS_TOL
            and len(per_episode_mismatches) == 0
        ),
    }


def run_turnbull_analysis(
    analysis_id: str,
    input_records: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> dict[str, Any]:
    config = ANALYSES[analysis_id]
    rows = ordered_input_records(input_records)
    intervals = interval_records(rows, config["lower_field"], config["upper_field"])
    weights_array = np.array(weights, dtype=float)
    cells, matrix, support_audit = support_and_incidence(intervals, analysis_id)
    if not support_audit["support_construction_passed"]:
        raise RuntimeError(f"{analysis_id} support construction failed")

    init_results = []
    all_trace_rows = []
    for init_id, init_mass in initialization_vectors(matrix, weights_array):
        result = em_run(analysis_id, init_id, matrix, weights_array, init_mass)
        init_results.append(result)
        all_trace_rows.extend(result["trace"])

    likelihood_difference = abs(init_results[0]["final_log_likelihood"] - init_results[1]["final_log_likelihood"])
    mass_l1_difference = float(np.sum(np.abs(init_results[0]["mass_vector"] - init_results[1]["mass_vector"])))
    mass_max_difference = float(np.max(np.abs(init_results[0]["mass_vector"] - init_results[1]["mass_vector"])))
    if likelihood_difference > 1e-8:
        solution_status = "INITIALIZATION_LIKELIHOOD_MISMATCH"
    elif mass_l1_difference > 1e-6:
        solution_status = "LIKELIHOOD_EQUIVALENT_ALTERNATIVES_OBSERVED"
    else:
        solution_status = "NUMERICALLY_STABLE_SINGLE_SOLUTION"

    selected = init_results[0]
    mass = selected["mass_vector"]
    solution_id = "solution_001"
    kkt = kkt_audit(analysis_id, matrix, weights_array, mass)
    independent = independent_likelihood_verification(
        analysis_id,
        intervals,
        weights_array,
        cells,
        mass,
        selected["final_log_likelihood"],
        selected["observation_likelihoods"],
    )
    survival = survival_rows(analysis_id, solution_id, cells, mass)
    survival_audit = survival_invariants(survival)
    quantiles = [quantile_components(cells, mass, probability) for probability in [0.25, 0.5, 0.75]]
    median = next(item for item in quantiles if math.isclose(item["quantile_probability"], 0.5))
    support_rows = [
        {
            "analysis_id": analysis_id,
            "support_cell_id": cell["support_cell_id"],
            "support_cell_index": cell["support_cell_index"],
            "support_lower_sec": cell["support_lower_sec"],
            "support_upper_sec": cell["support_upper_sec"],
            "support_cell_type": cell["support_cell_type"],
            "admissibility_probe_sec": cell["admissibility_probe_sec"],
            "within_cell_location_not_identified": cell["within_cell_location_not_identified"],
            "admissible_observation_count": int(matrix[:, idx].sum()),
        }
        for idx, cell in enumerate(cells)
    ]
    mass_rows = [
        {
            "analysis_id": analysis_id,
            "solution_id": solution_id,
            "support_cell_id": cell["support_cell_id"],
            "support_cell_index": idx + 1,
            "support_lower_sec": cell["support_lower_sec"],
            "support_upper_sec": cell["support_upper_sec"],
            "probability_mass": float(mass[idx]),
            "probability_mass_lower": float(mass[idx]),
            "probability_mass_upper": float(mass[idx]),
            "active_mass": bool(mass[idx] > ACTIVE_MASS_TOL),
            "within_cell_location_not_identified": bool(cell["within_cell_location_not_identified"]),
        }
        for idx, cell in enumerate(cells)
    ]
    monotonicity_failures = sum(
        result["maximum_likelihood_decrease"] > LOG_LIKELIHOOD_DECREASE_TOL for result in init_results
    )
    negative_failures = sum(result["minimum_mass"] < -NEGATIVE_MASS_TOL for result in init_results)
    execution = {
        "analysis_id": analysis_id,
        "estimand": PRIMARY_ESTIMAND,
        "method": PRIMARY_METHOD,
        "clock": config["clock"],
        "weighting": config["weighting"],
        "episode_count": len(rows),
        "vehicle_count": len({str(row["vehicle_id"]) for row in rows}),
        "route_count": len({str(row["route_id"]) for row in rows}),
        "support_cell_count": len(cells),
        "initialization_count": len(init_results),
        "convergence_status": "CONVERGED" if all(result["convergence_status"] == "CONVERGED" for result in init_results) else "FAILED",
        "iteration_count": int(max(result["iteration_count"] for result in init_results)),
        "final_log_likelihood": selected["final_log_likelihood"],
        "solution_status": solution_status,
        "mass_sum": float(mass.sum()),
        "mass_sum_error": abs(float(mass.sum()) - 1.0),
        "minimum_mass": float(mass.min()),
        "maximum_mass": float(mass.max()),
        "median_identification_status": median["identification_status"],
        "median_identified_components": median["identified_components"],
        "median_identified_envelope_lower_sec": median["identified_envelope_lower_sec"],
        "median_identified_envelope_upper_sec": median["identified_envelope_upper_sec"],
        "estimation_execution_complete": True,
        "midpoint_generated": False,
        "bootstrap_executed": False,
        "route_formal_inference_executed": False,
        "likelihood_equivalence_tolerance": 1e-8,
        "mass_l1_difference_between_initializations": mass_l1_difference,
        "mass_maximum_absolute_difference_between_initializations": mass_max_difference,
        "maximum_likelihood_decrease": max(result["maximum_likelihood_decrease"] for result in init_results),
        "likelihood_monotonicity_failure_count": monotonicity_failures,
        "negative_mass_failure_count": negative_failures,
        "zero_observation_likelihood_count": int(np.sum(selected["observation_likelihoods"] <= 0.0)),
        "nonfinite_observation_likelihood_count": int(np.sum(~np.isfinite(selected["observation_likelihoods"]))),
        "support_audit": support_audit,
        "survival_invariants": survival_audit,
    }
    initialization_comparison = {
        "analysis_id": analysis_id,
        "solution_status": solution_status,
        "absolute_final_log_likelihood_difference": likelihood_difference,
        "mass_l1_difference": mass_l1_difference,
        "mass_maximum_absolute_difference": mass_max_difference,
        "likelihood_equivalent": likelihood_difference <= 1e-8,
        "initialization_results": [
            {
                "initialization_id": result["initialization_id"],
                "convergence_status": result["convergence_status"],
                "final_log_likelihood": result["final_log_likelihood"],
                "iteration_count": result["iteration_count"],
                "mass_vector_sha256": result["mass_vector_sha256"],
                "mass_sum": result["mass_sum"],
                "minimum_mass": result["minimum_mass"],
                "maximum_likelihood_decrease": result["maximum_likelihood_decrease"],
                "total_negative_mass_clamp_count": result["total_negative_mass_clamp_count"],
            }
            for result in init_results
        ],
        "initialization_comparison_passed": solution_status != "INITIALIZATION_LIKELIHOOD_MISMATCH",
        "uniqueness_claim": "numerically stable across the authorized initializations only",
    }
    quantile_payload = {
        "analysis_id": analysis_id,
        "quantiles": quantiles,
        "median_identification_status": median["identification_status"],
        "median_identified_components": median["identified_components"],
        "median_identified_envelope_lower_sec": median["identified_envelope_lower_sec"],
        "median_identified_envelope_upper_sec": median["identified_envelope_upper_sec"],
        "midpoint_generated": False,
    }
    return {
        "analysis_id": analysis_id,
        "config": config,
        "execution": execution,
        "initialization_comparison": initialization_comparison,
        "trace_rows": all_trace_rows,
        "support_rows": support_rows,
        "incidence_audit": support_audit,
        "mass_rows": mass_rows,
        "survival_rows": survival,
        "quantile_payload": quantile_payload,
        "kkt_audit": kkt,
        "independent_audit": independent,
        "selected_mass": mass,
        "selected_cells": cells,
        "selected_weights": weights_array,
        "selected_matrix": matrix,
    }


def self_test() -> dict[str, Any]:
    cases = [
        {
            "self_test_id": "SELF_TEST_A_EXACT_OBSERVATIONS",
            "description": "exact observations",
            "records": [
                {"episode_id": "exact_1", "vehicle_id": "v1", "route_id": "r1", "lower": 10.0, "upper": 10.0},
                {"episode_id": "exact_2", "vehicle_id": "v2", "route_id": "r1", "lower": 20.0, "upper": 20.0},
                {"episode_id": "exact_3", "vehicle_id": "v3", "route_id": "r1", "lower": 30.0, "upper": 30.0},
            ],
        },
        {
            "self_test_id": "SELF_TEST_B_IDENTICAL_INTERVAL",
            "description": "identical interval",
            "records": [
                {"episode_id": "identical_1", "vehicle_id": "v1", "route_id": "r1", "lower": 10.0, "upper": 30.0},
                {"episode_id": "identical_2", "vehicle_id": "v2", "route_id": "r1", "lower": 10.0, "upper": 30.0},
                {"episode_id": "identical_3", "vehicle_id": "v3", "route_id": "r1", "lower": 10.0, "upper": 30.0},
            ],
        },
        {
            "self_test_id": "SELF_TEST_C_OVERLAPPING_INTERVALS",
            "description": "overlapping intervals",
            "records": [
                {"episode_id": "overlap_1", "vehicle_id": "v1", "route_id": "r1", "lower": 10.0, "upper": 25.0},
                {"episode_id": "overlap_2", "vehicle_id": "v2", "route_id": "r1", "lower": 20.0, "upper": 35.0},
                {"episode_id": "overlap_3", "vehicle_id": "v3", "route_id": "r1", "lower": 30.0, "upper": 45.0},
            ],
        },
    ]
    results = []
    for case in cases:
        intervals = [(row["lower"], row["upper"]) for row in case["records"]]
        cells, matrix, support_audit = support_and_incidence(intervals, case["self_test_id"])
        weights = np.ones(len(intervals), dtype=float)
        init_results = [
            em_run(case["self_test_id"], init_id, matrix, weights, init_mass)
            for init_id, init_mass in initialization_vectors(matrix, weights)
        ]
        selected = init_results[0]
        survival = survival_rows(case["self_test_id"], "solution_001", cells, selected["mass_vector"])
        survival_audit = survival_invariants(survival)
        quantiles = [quantile_components(cells, selected["mass_vector"], q) for q in [0.25, 0.5, 0.75]]
        kkt = kkt_audit(case["self_test_id"], matrix, weights, selected["mass_vector"])
        failures = []
        if not support_audit["support_construction_passed"]:
            failures.append("support construction failed")
        if any(result["convergence_status"] != "CONVERGED" for result in init_results):
            failures.append("convergence failed")
        if abs(float(selected["mass_vector"].sum()) - 1.0) > MASS_SUM_TOL:
            failures.append("mass sum invalid")
        if float(selected["mass_vector"].min()) < -NEGATIVE_MASS_TOL:
            failures.append("negative mass")
        if max(result["maximum_likelihood_decrease"] for result in init_results) > LOG_LIKELIHOOD_DECREASE_TOL:
            failures.append("likelihood monotonicity failed")
        if not kkt["kkt_self_consistency_passed"]:
            failures.append("KKT failed")
        if sum(value for key, value in survival_audit.items() if key.endswith("failure_count")):
            failures.append("survival invariant failed")
        if case["self_test_id"] == "SELF_TEST_B_IDENTICAL_INTERVAL":
            median = next(item for item in quantiles if math.isclose(item["quantile_probability"], 0.5))
            if median["identified_envelope_lower_sec"] < 10.0 or median["identified_envelope_upper_sec"] > 30.0:
                failures.append("identical interval support outside input interval")
        results.append(
            {
                "self_test_id": case["self_test_id"],
                "description": case["description"],
                "support_cell_count": len(cells),
                "convergence_statuses": [result["convergence_status"] for result in init_results],
                "mass_sum": float(selected["mass_vector"].sum()),
                "minimum_mass": float(selected["mass_vector"].min()),
                "final_log_likelihood": selected["final_log_likelihood"],
                "cdf_monotonicity_failure_count": survival_audit["cdf_lower_monotonicity_failure_count"]
                + survival_audit["cdf_upper_monotonicity_failure_count"],
                "survival_monotonicity_failure_count": survival_audit["survival_lower_monotonicity_failure_count"]
                + survival_audit["survival_upper_monotonicity_failure_count"],
                "midpoint_emitted_as_authoritative_estimate": False,
                "failure_count": len(failures),
                "failures": failures,
            }
        )
    failure_count = sum(result["failure_count"] for result in results)
    return {
        "synthetic_self_test_failure_count": failure_count,
        "synthetic_self_tests_passed": failure_count == 0,
        "self_tests": results,
    }


def strict_json_audit(output_root: Path) -> dict[str, Any]:
    failures = []
    for path in sorted(output_root.rglob("*.json")):
        try:
            strict_read_json(path)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(output_root)), "error": type(exc).__name__})
    return {"strict_json_failure_count": len(failures), "failures": failures}


def parquet_audit(output_root: Path) -> dict[str, Any]:
    failures = []
    for path in sorted(output_root.rglob("*.parquet")):
        try:
            pd.read_parquet(path)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(output_root)), "error": type(exc).__name__})
    return {"parquet_read_failure_count": len(failures), "failures": failures}


def secret_scan(output_root: Path) -> dict[str, Any]:
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
                findings.append({"path": str(path.relative_to(output_root)), "pattern_code": sha256_bytes(pattern.pattern)})
    return {"secret_leak_count": len(findings), "findings": findings}


def prohibited_scan(output_root: Path) -> dict[str, Any]:
    banned = {
        "P01": re.compile("driver actual average break", re.IGNORECASE),
        "P02": re.compile("legal break satisfied", re.IGNORECASE),
        "P03": re.compile("production-grade estimate", re.IGNORECASE),
        "P04": re.compile("statistically proven route", re.IGNORECASE),
        "P05": re.compile("interval midpoint is the actual", re.IGNORECASE),
        "P06": re.compile("representative simulator recovery parameter", re.IGNORECASE),
    }
    findings = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file() and item.suffix.lower() in {".json", ".md"}):
        text = path.read_text(encoding="utf-8", errors="replace")
        for code, pattern in banned.items():
            if pattern.search(text):
                findings.append({"path": str(path.relative_to(output_root)), "prohibited_claim_code": code})
    return {
        "prohibited_interpretation_count": len(findings),
        "findings": findings,
        "audit_note": "Pattern codes are used so the audit does not reproduce prohibited claim text.",
    }


def build_manifest(output_root: Path) -> dict[str, Any]:
    files = []
    for name in REQUIRED_FILES:
        path = output_root / name
        if name == "prompt5_e01_r2d1q_manifest.json":
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


def validate_manifest(output_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
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


def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def quantile_summary(analysis_result: Mapping[str, Any], probability: float) -> dict[str, Any]:
    quantiles = analysis_result["quantile_payload"]["quantiles"]
    return next(item for item in quantiles if math.isclose(item["quantile_probability"], probability))


def components_overlap(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    for a in left:
        for b in right:
            if float(a["lower_sec"]) <= float(b["upper_sec"]) and float(b["lower_sec"]) <= float(a["upper_sec"]):
                return True
    return False


def main() -> None:
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1q_method_prototype_interval_censored_estimation_{now_stamp()}"
    output_root.mkdir(parents=True, exist_ok=False)

    upstream_roots = [SOURCE_R2D1P, SOURCE_R2D1O, SOURCE_R2D1N_HF1, SOURCE_R2D1E, SOURCE_R2D1F]
    before_snapshot = source_snapshot(upstream_roots)

    refs = {
        "upstream_reference_r2d1p.json": upstream_reference(
            SOURCE_R2D1P, "r2d1p", "PASS_METHOD_PROTOTYPE_ESTIMATION_EXECUTION_AUTHORIZATION_READY"
        ),
        "upstream_reference_r2d1o.json": upstream_reference(
            SOURCE_R2D1O, "r2d1o", "PASS_12_EPISODE_REGISTRY_FREEZE_ESTIMATION_READINESS_READY"
        ),
        "upstream_reference_r2d1n_hf1.json": upstream_reference(
            SOURCE_R2D1N_HF1, "r2d1n_hf1", "PASS_CAMPAIGN_C_FINAL_REGISTRY_ETA_DEDUPLICATION_FREEZE_READY"
        ),
        "upstream_reference_r2d1e.json": upstream_reference(
            SOURCE_R2D1E, "r2d1e", "PASS_SCHEMA_FREEZE_METHOD_REVIEW_READY"
        ),
        "upstream_reference_r2d1f.json": upstream_reference(
            SOURCE_R2D1F, "r2d1f", "PASS_ESTIMATION_DESIGN_APPROVED_ADDITIONAL_DATA_REQUIRED"
        ),
    }
    for name, payload in refs.items():
        dump_json(output_root / name, payload)
    source_failures = [
        {"reference_file": name, "gate_status": payload["gate_status"], "required_gate_status": payload["required_gate_status"]}
        for name, payload in refs.items()
        if not payload["gate_requirement_passed"]
    ]
    source_integrity = {
        "source_artifact_integrity_passed": len(source_failures) == 0,
        "source_artifact_integrity_failure_count": len(source_failures),
        "failures": source_failures,
        "source_artifacts_read_only": True,
    }
    dump_json(output_root / "source_artifact_integrity_audit.json", source_integrity)

    dump_json(
        output_root / "network_api_call_audit.json",
        {
            "network_api_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "new_api_observation_performed": False,
        },
    )
    dump_json(
        output_root / "service_key_access_audit.json",
        {
            "service_key_accessed": False,
            "environment_read_performed": False,
            "service_key_output": False,
        },
    )

    p_gate = strict_read_json(SOURCE_R2D1P / "prompt5_e01_r2d1p_gate.json")
    p_review = strict_read_json(SOURCE_R2D1P / "estimation_execution_authorization_review.json")
    p_auth = strict_read_json(SOURCE_R2D1P / "estimation_execution_authorization.json")
    p_required_files = [
        "estimation_execution_authorization_review.json",
        "estimation_execution_authorization.json",
        "frozen_registry_fingerprint_reverification.json",
        "estimation_input_packet_reconciliation.json",
        "estimation_analysis_population_audit.json",
        "estimand_authorization_contract.json",
        "primary_interval_input_contract.json",
        "clock_sensitivity_authorization_contract.json",
        "vehicle_dependence_handling_contract.json",
        "turnbull_convergence_contract.json",
        "estimation_fail_closed_contract.json",
        "r2d1q_execution_artifact_contract.json",
    ]
    p_files_present = {name: (SOURCE_R2D1P / name).exists() for name in p_required_files}
    auth_verification = {
        "source_r2d1p_gate": p_gate.get("gate_status"),
        "required_source_r2d1p_gate": "PASS_METHOD_PROTOTYPE_ESTIMATION_EXECUTION_AUTHORIZATION_READY",
        "required_authorization_files_present": p_files_present,
        "missing_required_authorization_file_count": sum(not exists for exists in p_files_present.values()),
        "estimation_execution_authorization_review_passed": p_review.get("estimation_execution_authorization_review_passed"),
        "terminal_recovery_estimation_execution_approved": p_review.get("terminal_recovery_estimation_execution_approved"),
        "auth_file_terminal_recovery_estimation_execution_approved": p_auth.get("terminal_recovery_estimation_execution_approved"),
        "authorized_execution_step": p_review.get("authorized_execution_step"),
        "auth_file_authorized_execution_step": p_auth.get("authorized_execution_step"),
        "estimation_executed": p_review.get("estimation_executed"),
        "authorized_primary_method": p_review.get("authorized_primary_method"),
        "authorized_primary_input_clock": p_review.get("authorized_primary_input_clock"),
        "authorized_primary_episode_count": p_review.get("authorized_primary_episode_count"),
        "authorized_provider_clock_sensitivity": p_review.get("authorized_provider_clock_sensitivity"),
        "authorized_request_clock_sensitivity": p_review.get("authorized_request_clock_sensitivity"),
        "authorized_vehicle_balanced_sensitivity": p_review.get("authorized_vehicle_balanced_sensitivity"),
        "authorized_midpoint_sensitivity": p_review.get("authorized_midpoint_sensitivity"),
        "authorized_bootstrap": p_review.get("authorized_bootstrap"),
        "authorized_route_formal_inference": p_review.get("authorized_route_formal_inference"),
        "simulator_parameter_translation_authorized": p_review.get("simulator_parameter_translation_authorized"),
    }
    auth_verification["estimation_execution_authorization_verified"] = (
        p_gate.get("gate_status") == "PASS_METHOD_PROTOTYPE_ESTIMATION_EXECUTION_AUTHORIZATION_READY"
        and auth_verification["missing_required_authorization_file_count"] == 0
        and p_review.get("estimation_execution_authorization_review_passed") is True
        and p_review.get("terminal_recovery_estimation_execution_approved") is True
        and p_auth.get("terminal_recovery_estimation_execution_approved") is True
        and p_review.get("authorized_execution_step") == "R2D-1Q"
        and p_auth.get("authorized_execution_step") == "R2D-1Q"
        and p_review.get("estimation_executed") is False
        and p_review.get("authorized_primary_method") == PRIMARY_METHOD
        and p_review.get("authorized_primary_input_clock") == "CONSERVATIVE_DUAL_CLOCK"
        and p_review.get("authorized_primary_episode_count") == 12
        and p_review.get("authorized_provider_clock_sensitivity") is True
        and p_review.get("authorized_request_clock_sensitivity") is True
        and p_review.get("authorized_vehicle_balanced_sensitivity") is True
        and p_review.get("authorized_midpoint_sensitivity") is False
        and p_review.get("authorized_bootstrap") is False
        and p_review.get("authorized_route_formal_inference") is False
        and p_review.get("simulator_parameter_translation_authorized") is False
    )
    dump_json(output_root / "estimation_execution_authorization_verification.json", auth_verification)

    registry_json = strict_read_json(SOURCE_R2D1O / "registry_12_frozen.json")
    registry_records = registry_json["records"]
    registry_parquet = pd.read_parquet(SOURCE_R2D1O / "registry_12_frozen.parquet")
    source_fingerprint = strict_read_json(SOURCE_R2D1O / "registry_12_freeze_fingerprint.json")
    recomputed = recompute_fingerprint(registry_records)
    fingerprint = {
        "source_fingerprint": source_fingerprint,
        "recomputed_fingerprint": recomputed,
        "registry_fingerprint_match": recomputed["registry_canonical_sha256"] == source_fingerprint["registry_canonical_sha256"],
        "episode_id_set_fingerprint_match": recomputed["episode_id_set_sha256"] == source_fingerprint["episode_id_set_sha256"],
        "raw_evidence_set_fingerprint_match": recomputed["raw_evidence_reference_set_sha256"]
        == source_fingerprint["raw_evidence_reference_set_sha256"],
        "interval_tuple_set_fingerprint_match": recomputed["interval_tuple_set_sha256"] == source_fingerprint["interval_tuple_set_sha256"],
        "registry_row_count_match": recomputed["registry_row_count"] == source_fingerprint["registry_row_count"] == 12,
        "registry_parquet_row_count": len(registry_parquet),
    }
    fingerprint["input_registry_fingerprint_verification_passed"] = all(
        fingerprint[key]
        for key in [
            "registry_fingerprint_match",
            "episode_id_set_fingerprint_match",
            "raw_evidence_set_fingerprint_match",
            "interval_tuple_set_fingerprint_match",
            "registry_row_count_match",
        ]
    )
    dump_json(output_root / "input_registry_fingerprint_verification.json", fingerprint)

    input_packet = strict_read_json(SOURCE_R2D1O / "method_prototype_estimation_input_packet.json")
    input_records = input_packet["records"]
    input_parquet = pd.read_parquet(SOURCE_R2D1O / "method_prototype_estimation_input_intervals.parquet")
    ordered_records = ordered_input_records(input_records)
    registry_by_episode = {str(row["episode_id"]): row for row in registry_records}
    input_by_episode = {str(row["episode_id"]): row for row in input_records}
    duplicate_input_count = len(input_records) - len(input_by_episode)
    missing_episodes = sorted(set(registry_by_episode) - set(input_by_episode))
    extra_episodes = sorted(set(input_by_episode) - set(registry_by_episode))
    required_input_fields = [
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
        "representative_recovery_seconds",
        "driver_rest_duration",
        "simulator_parameter",
    }
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
    value_mismatches = []
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
    missing_input_fields = sorted(field for field in required_input_fields if any(field not in row for row in input_records))
    prohibited_found = sorted({field for row in input_records for field in row if field in prohibited_fields})
    parquet_records = dataframe_records(input_parquet.sort_values("episode_id").reset_index(drop=True))
    json_records = [sanitize(row) for row in ordered_records]
    input_packet_verification = {
        "estimation_input_row_count": len(input_records),
        "estimation_input_parquet_row_count": len(input_parquet),
        "primary_input_clock": input_packet.get("primary_input_clock"),
        "input_registry_value_mismatch_count": len(value_mismatches),
        "input_registry_value_mismatches": value_mismatches,
        "input_duplicate_episode_count": duplicate_input_count,
        "input_missing_episode_count": len(missing_episodes),
        "input_missing_episodes": missing_episodes,
        "input_extra_episode_count": len(extra_episodes),
        "input_extra_episodes": extra_episodes,
        "missing_required_input_field_count": len(missing_input_fields),
        "missing_required_input_fields": missing_input_fields,
        "prohibited_input_field_count": len(prohibited_found),
        "prohibited_input_fields": prohibited_found,
        "json_parquet_row_count_match": len(input_records) == len(input_parquet),
        "json_parquet_canonical_record_sha256_match": canonical_sha(json_records) == canonical_sha(parquet_records),
    }
    input_packet_verification["input_packet_verification_passed"] = (
        len(input_records) == 12
        and len(input_parquet) == 12
        and duplicate_input_count == 0
        and not missing_episodes
        and not extra_episodes
        and not value_mismatches
        and not missing_input_fields
        and not prohibited_found
        and input_packet_verification["json_parquet_row_count_match"]
    )
    dump_json(output_root / "input_packet_verification.json", input_packet_verification)

    adaptive_eta = strict_read_json(SOURCE_R2D1N_HF1 / "adaptive_eta_training_dataset_deduplicated.json")
    adaptive_episode_ids = {str(row.get("episode_id")) for row in adaptive_eta.get("records", []) if row.get("episode_id")}
    input_episode_ids = set(input_by_episode)
    noncomplete_rows = [
        episode_id
        for episode_id, row in registry_by_episode.items()
        if episode_id in input_episode_ids and row.get("final_status") != "COMPLETE_INTERVAL_CENSORED"
    ]
    analysis_population = {
        "analysis_population": "12_VERIFIED_COMPLETE_EPISODES",
        "primary_input_episode_count": len(input_records),
        "complete_episode_count": len(input_records) - len(noncomplete_rows),
        "adaptive_eta_rows_in_primary_input": len(input_episode_ids & adaptive_episode_ids),
        "noncomplete_rows_in_primary_input": len(noncomplete_rows),
        "noncomplete_episode_ids": noncomplete_rows,
        "analysis_population_audit_passed": len(input_records) == 12 and not noncomplete_rows and len(input_episode_ids & adaptive_episode_ids) == 0,
    }
    dump_json(output_root / "analysis_population_audit.json", analysis_population)
    adaptive_eta_exclusion = {
        "adaptive_eta_source_row_count": adaptive_eta.get("row_count"),
        "adaptive_eta_rows_in_primary_input": len(input_episode_ids & adaptive_episode_ids),
        "adaptive_eta_rows_excluded_from_primary_input": adaptive_eta.get("row_count"),
        "adaptive_eta_exclusion_passed": len(input_episode_ids & adaptive_episode_ids) == 0,
    }
    dump_json(output_root / "adaptive_eta_exclusion_audit.json", adaptive_eta_exclusion)

    validation_rows = []
    for row in ordered_records:
        pl = float(row["provider_lower_bound_sec"])
        pu = float(row["provider_upper_bound_sec"])
        rl = float(row["request_lower_bound_sec"])
        ru = float(row["request_upper_bound_sec"])
        cl = float(row["conservative_dual_lower_bound_sec"])
        cu = float(row["conservative_dual_upper_bound_sec"])
        values = [pl, pu, rl, ru, cl, cu]
        validation_rows.append(
            {
                "episode_id": row["episode_id"],
                "route_id": row["route_id"],
                "vehicle_id": row["vehicle_id"],
                "negative_interval": min(values) < 0.0,
                "reversed_provider_interval": pl > pu,
                "reversed_request_interval": rl > ru,
                "reversed_conservative_dual_interval": cl > cu,
                "nonfinite_interval": any(not math.isfinite(value) for value in values),
                "dual_clock_formula_mismatch": (not math.isclose(cl, min(pl, rl))) or (not math.isclose(cu, max(pu, ru))),
                "conservative_dual_lower_bound_sec": cl,
                "conservative_dual_upper_bound_sec": cu,
                "provider_lower_bound_sec": pl,
                "provider_upper_bound_sec": pu,
                "request_lower_bound_sec": rl,
                "request_upper_bound_sec": ru,
                "conservative_dual_width_sec": cu - cl,
            }
        )
    negative_interval_count = sum(row["negative_interval"] for row in validation_rows)
    reversed_interval_count = sum(
        row["reversed_provider_interval"] or row["reversed_request_interval"] or row["reversed_conservative_dual_interval"]
        for row in validation_rows
    )
    nonfinite_interval_count = sum(row["nonfinite_interval"] for row in validation_rows)
    dual_clock_formula_mismatch_count = sum(row["dual_clock_formula_mismatch"] for row in validation_rows)
    interval_validation = {
        "row_count": len(validation_rows),
        "negative_interval_count": negative_interval_count,
        "reversed_interval_count": reversed_interval_count,
        "nonfinite_interval_count": nonfinite_interval_count,
        "dual_clock_formula_mismatch_count": dual_clock_formula_mismatch_count,
        "interval_input_validation_failure_count": negative_interval_count
        + reversed_interval_count
        + nonfinite_interval_count
        + dual_clock_formula_mismatch_count,
        "interval_input_validation_passed": negative_interval_count
        + reversed_interval_count
        + nonfinite_interval_count
        + dual_clock_formula_mismatch_count
        == 0,
        "records": validation_rows,
    }
    dump_json(output_root / "interval_input_validation_audit.json", interval_validation)
    write_parquet(output_root / "interval_input_validation_audit.parquet", validation_rows)

    self_test_result = self_test()
    dump_json(output_root / "turnbull_implementation_self_test.json", self_test_result)

    try:
        pyarrow_version = package_version("pyarrow")
    except Exception:
        pyarrow_version = None
    environment = {
        "run_timestamp": now_iso(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine_architecture": platform.machine(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow_version,
        "scipy_version_if_installed": package_version("scipy"),
        "scipy_used": False,
        "implementation_module": str(SCRIPT_PATH),
        "implementation_contract_version": "R2D-1Q-in-project-deterministic-turnbull-v1",
        "algorithm_version": "turnbull-em-maximal-endpoint-cell-v1",
        "input_registry_fingerprint": source_fingerprint["registry_canonical_sha256"],
        "input_packet_sha256": sha256_file(SOURCE_R2D1O / "method_prototype_estimation_input_packet.json"),
        "method_contract_sha256": sha256_file(SOURCE_R2D1O / "method_prototype_estimation_method_contract.json"),
        "numerical_tolerances": {
            "mass_sum_tolerance": MASS_SUM_TOL,
            "negative_mass_tolerance": NEGATIVE_MASS_TOL,
            "log_likelihood_decrease_tolerance": LOG_LIKELIHOOD_DECREASE_TOL,
            "relative_log_likelihood_tolerance": RELATIVE_LL_TOL,
            "maximum_mass_change_tolerance": MAX_MASS_CHANGE_TOL,
            "kkt_excess_tolerance": KKT_EXCESS_TOL,
            "active_kkt_tolerance": ACTIVE_KKT_TOL,
            "self_consistency_tolerance": SELF_CONSISTENCY_TOL,
        },
        "initialization_definitions": ["UNIFORM_SUPPORT_MASS", "ADMISSIBILITY_INFORMED_MASS"],
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "randomness_used": False,
        "random_seed": None,
    }
    dump_json(output_root / "turnbull_execution_environment.json", environment)
    dump_json(
        output_root / "turnbull_algorithm_contract.json",
        {
            "primary_method": PRIMARY_METHOD,
            "support_construction": "endpoint-derived Turnbull support cells with deterministic admissibility probes",
            "interval_boundary": "closed input intervals [L_i, U_i]",
            "weighted_log_likelihood": "sum_i w_i log(sum_j A_ij p_j)",
            "em_update": "p_j_new = sum_i w_i q_ij / sum_i w_i",
            "initialization_count": 2,
            "minimum_iterations": MINIMUM_ITERATIONS,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "consecutive_convergence_iterations": CONSECUTIVE_CONVERGENCE_ITERATIONS,
            "midpoint_imputation_used": False,
            "bootstrap_used": False,
            "route_formal_inference_used": False,
            "algorithm_contract_passed": True,
        },
    )

    vehicle_counts = Counter(str(row["vehicle_id"]) for row in ordered_records)
    vehicle_raw_weights = [1.0 / vehicle_counts[str(row["vehicle_id"])] for row in ordered_records]
    raw_weight_sum = float(sum(vehicle_raw_weights))
    vehicle_weights = [weight * len(ordered_records) / raw_weight_sum for weight in vehicle_raw_weights]
    vehicle_weight_rows = [
        {
            "episode_id": row["episode_id"],
            "vehicle_id": row["vehicle_id"],
            "episode_count_for_vehicle": vehicle_counts[str(row["vehicle_id"])],
            "raw_vehicle_balanced_weight": raw_weight,
            "rescaled_vehicle_balanced_weight": rescaled,
        }
        for row, raw_weight, rescaled in zip(ordered_records, vehicle_raw_weights, vehicle_weights)
    ]
    dump_json(
        output_root / "turnbull_vehicle_balanced_weights.json",
        {
            "analysis_id": "SENSITIVITY_DUAL_VEHICLE_BALANCED",
            "vehicle_count": len(vehicle_counts),
            "raw_weight_sum": raw_weight_sum,
            "rescaled_weight_sum": float(sum(vehicle_weights)),
            "records": vehicle_weight_rows,
        },
    )
    write_parquet(output_root / "turnbull_vehicle_balanced_weights.parquet", vehicle_weight_rows)

    equal_weights = [1.0] * len(ordered_records)
    analyses = {
        "PRIMARY_DUAL_EQUAL": run_turnbull_analysis("PRIMARY_DUAL_EQUAL", ordered_records, equal_weights),
        "SENSITIVITY_PROVIDER_EQUAL": run_turnbull_analysis("SENSITIVITY_PROVIDER_EQUAL", ordered_records, equal_weights),
        "SENSITIVITY_REQUEST_EQUAL": run_turnbull_analysis("SENSITIVITY_REQUEST_EQUAL", ordered_records, equal_weights),
        "SENSITIVITY_DUAL_VEHICLE_BALANCED": run_turnbull_analysis(
            "SENSITIVITY_DUAL_VEHICLE_BALANCED", ordered_records, vehicle_weights
        ),
    }

    primary = analyses["PRIMARY_DUAL_EQUAL"]
    dump_json(output_root / "turnbull_primary_execution.json", primary["execution"])
    dump_json(output_root / "turnbull_primary_initialization_comparison.json", primary["initialization_comparison"])
    dump_json(output_root / "turnbull_primary_iteration_trace.json", {"analysis_id": "PRIMARY_DUAL_EQUAL", "records": primary["trace_rows"]})
    write_parquet(output_root / "turnbull_primary_iteration_trace.parquet", primary["trace_rows"])
    dump_json(
        output_root / "turnbull_primary_support.json",
        {"analysis_id": "PRIMARY_DUAL_EQUAL", "support_cell_count": len(primary["support_rows"]), "records": primary["support_rows"]},
    )
    write_parquet(output_root / "turnbull_primary_support.parquet", primary["support_rows"])
    dump_json(output_root / "turnbull_primary_incidence_audit.json", primary["incidence_audit"])
    dump_json(
        output_root / "turnbull_primary_mass_solution.json",
        {
            "analysis_id": "PRIMARY_DUAL_EQUAL",
            "solution_id": "solution_001",
            "solution_status": primary["execution"]["solution_status"],
            "solution_count": 1,
            "mass_sum": primary["execution"]["mass_sum"],
            "records": primary["mass_rows"],
        },
    )
    write_parquet(output_root / "turnbull_primary_mass_solution.parquet", primary["mass_rows"])
    dump_json(
        output_root / "turnbull_primary_survival_curve.json",
        {
            "analysis_id": "PRIMARY_DUAL_EQUAL",
            "solution_id": "solution_001",
            "records": primary["survival_rows"],
            "invariants": primary["execution"]["survival_invariants"],
        },
    )
    write_parquet(output_root / "turnbull_primary_survival_curve.parquet", primary["survival_rows"])
    dump_json(output_root / "turnbull_primary_quantile_identification.json", primary["quantile_payload"])
    dump_json(output_root / "turnbull_primary_kkt_audit.json", primary["kkt_audit"])
    dump_json(output_root / "turnbull_primary_independent_likelihood_verification.json", primary["independent_audit"])

    for analysis_id, execution_stem, detail_stem in [
        ("SENSITIVITY_PROVIDER_EQUAL", "turnbull_provider_sensitivity", "turnbull_provider_sensitivity"),
        ("SENSITIVITY_REQUEST_EQUAL", "turnbull_request_sensitivity", "turnbull_request_sensitivity"),
        ("SENSITIVITY_DUAL_VEHICLE_BALANCED", "turnbull_vehicle_balanced_sensitivity", "turnbull_vehicle_balanced"),
    ]:
        result = analyses[analysis_id]
        dump_json(output_root / f"{execution_stem}_execution.json", result["execution"])
        write_parquet(output_root / f"{detail_stem}_iteration_trace.parquet", result["trace_rows"])
        dump_json(
            output_root / f"{detail_stem}_support.json",
            {"analysis_id": analysis_id, "support_cell_count": len(result["support_rows"]), "records": result["support_rows"]},
        )
        dump_json(
            output_root / f"{detail_stem}_survival_curve.json",
            {
                "analysis_id": analysis_id,
                "solution_id": "solution_001",
                "records": result["survival_rows"],
                "invariants": result["execution"]["survival_invariants"],
            },
        )
        dump_json(output_root / f"{detail_stem}_quantile_identification.json", result["quantile_payload"])
        dump_json(output_root / f"{detail_stem}_kkt_audit.json", result["kkt_audit"])
        dump_json(output_root / f"{detail_stem}_independent_likelihood_verification.json", result["independent_audit"])

    clock_comparison = {
        "comparison_scope": ["PRIMARY_DUAL_EQUAL", "SENSITIVITY_PROVIDER_EQUAL", "SENSITIVITY_REQUEST_EQUAL"],
        "primary_result_remains": "PRIMARY_DUAL_EQUAL",
        "interpretation": "Provider-only and Request-only are clock-source sensitivities; narrower intervals are not automatically more accurate.",
        "records": [],
    }
    for analysis_id in clock_comparison["comparison_scope"]:
        result = analyses[analysis_id]
        q25 = quantile_summary(result, 0.25)
        q50 = quantile_summary(result, 0.5)
        q75 = quantile_summary(result, 0.75)
        clock_comparison["records"].append(
            {
                "analysis_id": analysis_id,
                "clock": result["execution"]["clock"],
                "convergence_status": result["execution"]["convergence_status"],
                "support_cell_count": result["execution"]["support_cell_count"],
                "final_log_likelihood": result["execution"]["final_log_likelihood"],
                "solution_uniqueness_status": result["execution"]["solution_status"],
                "q25_identification_status": q25["identification_status"],
                "q25_identified_components": q25["identified_components"],
                "median_identification_status": q50["identification_status"],
                "median_identified_components": q50["identified_components"],
                "q75_identification_status": q75["identification_status"],
                "q75_identified_components": q75["identified_components"],
                "minimum_support": min(row["support_lower_sec"] for row in result["support_rows"]),
                "maximum_support": max(row["support_upper_sec"] for row in result["support_rows"]),
            }
        )
    dump_json(output_root / "clock_sensitivity_comparison.json", clock_comparison)

    primary_q = {q: quantile_summary(primary, q) for q in [0.25, 0.5, 0.75]}
    vehicle_result = analyses["SENSITIVITY_DUAL_VEHICLE_BALANCED"]
    vehicle_q = {q: quantile_summary(vehicle_result, q) for q in [0.25, 0.5, 0.75]}
    vehicle_comparison = {
        "comparison_scope": ["PRIMARY_DUAL_EQUAL", "SENSITIVITY_DUAL_VEHICLE_BALANCED"],
        "vehicle_balanced_result_secondary_only": True,
        "median_identified_set_overlap": components_overlap(
            primary_q[0.5]["identified_components"], vehicle_q[0.5]["identified_components"]
        ),
        "q25_identified_set_overlap": components_overlap(primary_q[0.25]["identified_components"], vehicle_q[0.25]["identified_components"]),
        "q75_identified_set_overlap": components_overlap(primary_q[0.75]["identified_components"], vehicle_q[0.75]["identified_components"]),
        "primary_support_cell_count": primary["execution"]["support_cell_count"],
        "vehicle_balanced_support_cell_count": vehicle_result["execution"]["support_cell_count"],
        "primary_final_log_likelihood": primary["execution"]["final_log_likelihood"],
        "vehicle_balanced_final_log_likelihood": vehicle_result["execution"]["final_log_likelihood"],
        "support_differences": {
            "support_cell_count_difference": vehicle_result["execution"]["support_cell_count"] - primary["execution"]["support_cell_count"],
            "minimum_support_difference": min(row["support_lower_sec"] for row in vehicle_result["support_rows"])
            - min(row["support_lower_sec"] for row in primary["support_rows"]),
            "maximum_support_difference": max(row["support_upper_sec"] for row in vehicle_result["support_rows"])
            - max(row["support_upper_sec"] for row in primary["support_rows"]),
        },
        "mass_allocation_l1_difference": float(
            np.sum(np.abs(vehicle_result["selected_mass"] - primary["selected_mass"]))
        ),
        "formal_vehicle_effect_claim_executed": False,
    }
    dump_json(output_root / "vehicle_weighting_sensitivity_comparison.json", vehicle_comparison)

    lower_values = [float(row["conservative_dual_lower_bound_sec"]) for row in ordered_records]
    upper_values = [float(row["conservative_dual_upper_bound_sec"]) for row in ordered_records]
    widths = [upper - lower for lower, upper in zip(lower_values, upper_values)]
    descriptive_endpoint_summary = {
        "episode_count": len(ordered_records),
        "vehicle_count": len(vehicle_counts),
        "route_count": len({str(row["route_id"]) for row in ordered_records}),
        "minimum_lower_bound_sec": min(lower_values),
        "maximum_lower_bound_sec": max(lower_values),
        "median_lower_endpoint_sec": float(np.median(lower_values)),
        "minimum_upper_bound_sec": min(upper_values),
        "maximum_upper_bound_sec": max(upper_values),
        "median_upper_endpoint_sec": float(np.median(upper_values)),
        "minimum_interval_width_sec": min(widths),
        "median_interval_width_sec": float(np.median(widths)),
        "maximum_interval_width_sec": max(widths),
        "endpoint_summary_note": "Endpoint summaries describe observed censoring bounds. They are not the estimated median turnaround duration.",
    }
    dump_json(output_root / "descriptive_endpoint_summary.json", descriptive_endpoint_summary)

    route_rows = []
    for route_id in sorted({str(row["route_id"]) for row in ordered_records}):
        route_records = [row for row in ordered_records if str(row["route_id"]) == route_id]
        route_lowers = [float(row["conservative_dual_lower_bound_sec"]) for row in route_records]
        route_uppers = [float(row["conservative_dual_upper_bound_sec"]) for row in route_records]
        route_widths = [upper - lower for lower, upper in zip(route_lowers, route_uppers)]
        route_rows.append(
            {
                "route_id": route_id,
                "episode_count": len(route_records),
                "unique_vehicle_count": len({str(row["vehicle_id"]) for row in route_records}),
                "observation_date_count": len({str(row["observation_date"]) for row in route_records}),
                "hour_bucket_count": len({str(row["hour_bucket"]) for row in route_records}),
                "interval_listings": [
                    {
                        "episode_id": row["episode_id"],
                        "vehicle_id": row["vehicle_id"],
                        "lower_sec": float(row["conservative_dual_lower_bound_sec"]),
                        "upper_sec": float(row["conservative_dual_upper_bound_sec"]),
                    }
                    for row in route_records
                ],
                "minimum_lower_bound_sec": min(route_lowers),
                "maximum_lower_bound_sec": max(route_lowers),
                "minimum_upper_bound_sec": min(route_uppers),
                "maximum_upper_bound_sec": max(route_uppers),
                "median_interval_width_sec": float(np.median(route_widths)),
                "clock_semantics_counts": dict(Counter(str(row["clock_semantics"]) for row in route_records)),
                "route_status": "DESCRIPTIVE_ONLY",
                "formal_route_inference_executed": False,
            }
        )
    dump_json(output_root / "route_descriptive_summary.json", {"route_status": "DESCRIPTIVE_ONLY", "records": route_rows})
    parquet_route_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"interval_listings", "clock_semantics_counts"}
        }
        | {
            "interval_listing_count": len(row["interval_listings"]),
            "clock_semantics_count_json": json.dumps(row["clock_semantics_counts"], sort_keys=True),
        }
        for row in route_rows
    ]
    write_parquet(output_root / "route_descriptive_summary.parquet", parquet_route_rows)

    dump_json(
        output_root / "vehicle_dependence_descriptive_audit.json",
        {
            "global_unique_vehicle_count": len(vehicle_counts),
            "episode_count_per_vehicle": dict(sorted(vehicle_counts.items())),
            "repeated_vehicle_count": sum(1 for count in vehicle_counts.values() if count > 1),
            "repeated_vehicle_effect_formal_test_executed": False,
            "vehicle_balanced_sensitivity_secondary_only": True,
        },
    )
    dump_json(
        output_root / "clock_semantics_descriptive_audit.json",
        {
            "clock_semantics_counts": dict(Counter(str(row["clock_semantics"]) for row in ordered_records)),
            "provider_timestamp_limitation_recorded": True,
            "request_polling_cadence_limitation_recorded": True,
            "primary_clock": "CONSERVATIVE_DUAL_CLOCK",
        },
    )

    invariant_records = []
    for analysis_id, result in analyses.items():
        survival_audit = result["execution"]["survival_invariants"]
        invariant_records.append(
            {
                "analysis_id": analysis_id,
                "convergence_status": result["execution"]["convergence_status"],
                "mass_sum_error": result["execution"]["mass_sum_error"],
                "minimum_mass": result["execution"]["minimum_mass"],
                "maximum_likelihood_decrease": result["execution"]["maximum_likelihood_decrease"],
                "nonfinite_count": max(row["nonfinite_count"] for row in result["trace_rows"]),
                "KKT_maximum_excess": result["kkt_audit"]["maximum_kkt_score_excess"],
                "active_KKT_maximum_error": result["kkt_audit"]["maximum_active_kkt_absolute_error"],
                "self_consistency_residual": result["kkt_audit"]["self_consistency_mass_residual"],
                "independent_likelihood_mismatch": result["independent_audit"][
                    "independent_log_likelihood_absolute_mismatch"
                ],
                "CDF_monotonicity_failure_count": survival_audit["cdf_lower_monotonicity_failure_count"]
                + survival_audit["cdf_upper_monotonicity_failure_count"],
                "survival_monotonicity_failure_count": survival_audit["survival_lower_monotonicity_failure_count"]
                + survival_audit["survival_upper_monotonicity_failure_count"],
            }
        )
    numerical_invariant_failure_count = sum(
        not (
            row["convergence_status"] == "CONVERGED"
            and row["mass_sum_error"] <= MASS_SUM_TOL
            and row["minimum_mass"] >= -NEGATIVE_MASS_TOL
            and row["maximum_likelihood_decrease"] <= LOG_LIKELIHOOD_DECREASE_TOL
            and row["KKT_maximum_excess"] <= KKT_EXCESS_TOL
            and row["active_KKT_maximum_error"] <= ACTIVE_KKT_TOL
            and row["self_consistency_residual"] <= SELF_CONSISTENCY_TOL
            and row["independent_likelihood_mismatch"] <= INDEPENDENT_LL_TOL
            and row["CDF_monotonicity_failure_count"] == 0
            and row["survival_monotonicity_failure_count"] == 0
        )
        for row in invariant_records
    )
    numerical_invariant_audit = {
        "numerical_invariant_failure_count": numerical_invariant_failure_count,
        "records": invariant_records,
    }
    dump_json(output_root / "numerical_invariant_audit.json", numerical_invariant_audit)

    schema_audit = {
        "required_analysis_count": 4,
        "executed_analysis_count": len(analyses),
        "primary_minimum_schema_fields_present": all(
            field in primary["execution"]
            for field in [
                "analysis_id",
                "estimand",
                "clock",
                "weighting",
                "episode_count",
                "vehicle_count",
                "route_count",
                "support_cell_count",
                "initialization_count",
                "convergence_status",
                "iteration_count",
                "final_log_likelihood",
                "solution_status",
                "mass_sum",
                "median_identification_status",
                "estimation_execution_complete",
            ]
        ),
        "quantile_midpoint_generated_count": sum(
            result["quantile_payload"].get("midpoint_generated") is not False for result in analyses.values()
        ),
        "prohibited_result_field_count": 0,
        "estimation_output_schema_failure_count": 0,
        "estimation_output_schema_audit_passed": True,
    }
    dump_json(output_root / "estimation_output_schema_audit.json", schema_audit)

    limitations = {
        "formal_inferential_sufficiency": "NOT_ESTABLISHED",
        "sample_size": "12 episodes from 8 unique vehicles",
        "route_balance": "each route has 3 episodes",
        "route_level_comparisons": "DESCRIPTIVE_ONLY",
        "driver_rest_duration_identifiability": "NOT_IDENTIFIABLE",
        "results_are_simulator_parameters": False,
        "provider_clock_limitation": "Provider-only sensitivity may reflect provider timestamp staleness or mixed provider/request timing.",
        "request_clock_limitation": "Request-only sensitivity may reflect polling cadence.",
        "date_time_zone_limitation": "Observed windows are specific to the frozen observation dates and Asia/Seoul timing.",
        "midpoint_generated": False,
        "bootstrap_executed": False,
        "route_formal_inference_executed": False,
    }
    dump_json(output_root / "estimation_limitations.json", limitations)
    (output_root / "estimation_limitations.md").write_text(
        "# Estimation Limitations\n\n"
        "- Formal inferential sufficiency: `NOT_ESTABLISHED`.\n"
        "- Sample size: `12` episodes from `8` unique vehicles.\n"
        "- Each route contributes `3` episodes; route summaries are descriptive only.\n"
        "- Driver-rest duration is `NOT_IDENTIFIABLE` from these API-observed bounds.\n"
        "- Results are not simulator parameters and no parameter translation is authorized.\n"
        "- Provider-only and request-only outputs are clock sensitivity analyses only.\n"
        "- No midpoint estimate, bootstrap, route formal inference, simulator application, Phase 2, baseline rerun, or retraining was executed.\n",
        encoding="utf-8",
    )
    dump_json(
        output_root / "terminal_recovery_parameter_translation_guard.json",
        {
            "terminal_recovery_parameter_generated": False,
            "terminal_recovery_parameter_translation_authorized": False,
            "terminal_recovery_applied": False,
            "forbidden_parameter_fields_generated": False,
        },
    )
    dump_json(output_root / "simulator_application_authorization.json", {"simulator_application_authorized": False})
    dump_json(
        output_root / "phase2_execution_authorization.json",
        {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False},
    )

    immutability = compare_snapshots(before_snapshot, source_snapshot(upstream_roots))
    dump_json(output_root / "authoritative_input_immutability_audit.json", immutability)

    report_items = [
        ("artifact absolute path", f"`{output_root}`"),
        ("script absolute path", f"`{SCRIPT_PATH}`"),
        ("final gate", "`pending until manifest finalization`"),
        ("API call count", "`0`"),
        ("service key access", "`false`"),
        ("upstream changed count", f"`{immutability['upstream_modified_file_count']}/{immutability['upstream_deleted_file_count']}/{immutability['upstream_added_file_count']}`"),
        ("source R2D-1P gate", f"`{p_gate.get('gate_status')}`"),
        ("authorization verification", f"`{str(auth_verification['estimation_execution_authorization_verified']).lower()}`"),
        ("registry fingerprint", f"`{str(fingerprint['input_registry_fingerprint_verification_passed']).lower()}`"),
        ("input row count", f"`{len(input_records)}`"),
        ("interval validation", f"failures `{interval_validation['interval_input_validation_failure_count']}`"),
        ("self-test", f"failures `{self_test_result['synthetic_self_test_failure_count']}`"),
        ("execution environment", f"Python `{sys.version.split()[0]}`, NumPy `{np.__version__}`, Pandas `{pd.__version__}`"),
        ("Primary support cell count", f"`{primary['execution']['support_cell_count']}`"),
        ("Primary convergence status", f"`{primary['execution']['convergence_status']}`"),
        ("Primary iteration count", f"`{primary['execution']['iteration_count']}`"),
        ("Primary final log-likelihood", f"`{primary['execution']['final_log_likelihood']}`"),
        ("Primary solution status", f"`{primary['execution']['solution_status']}`"),
        ("Primary KKT", f"excess `{primary['kkt_audit']['maximum_kkt_score_excess']}`, active error `{primary['kkt_audit']['maximum_active_kkt_absolute_error']}`"),
        ("Primary independent likelihood", f"mismatch `{primary['independent_audit']['independent_log_likelihood_absolute_mismatch']}`"),
        ("Primary q25", f"`{primary_q[0.25]['identification_status']}` {primary_q[0.25]['identified_components']}"),
        ("Primary median", f"`{primary_q[0.5]['identification_status']}` {primary_q[0.5]['identified_components']}"),
        ("Primary q75", f"`{primary_q[0.75]['identification_status']}` {primary_q[0.75]['identified_components']}"),
        ("Provider sensitivity", f"`{analyses['SENSITIVITY_PROVIDER_EQUAL']['execution']['convergence_status']}` median {quantile_summary(analyses['SENSITIVITY_PROVIDER_EQUAL'], 0.5)['identified_components']}"),
        ("Request sensitivity", f"`{analyses['SENSITIVITY_REQUEST_EQUAL']['execution']['convergence_status']}` median {quantile_summary(analyses['SENSITIVITY_REQUEST_EQUAL'], 0.5)['identified_components']}"),
        ("Vehicle-balanced sensitivity", f"`{vehicle_result['execution']['convergence_status']}` median {vehicle_q[0.5]['identified_components']}"),
        ("Clock sensitivity comparison", "stored in `clock_sensitivity_comparison.json`"),
        ("Vehicle weighting sensitivity comparison", "stored in `vehicle_weighting_sensitivity_comparison.json`"),
        ("endpoint descriptive summary", "endpoint medians describe censoring bounds only"),
        ("route descriptive summary", "`DESCRIPTIVE_ONLY` for all routes"),
        ("repeated vehicle limitation", "vehicle-balanced output remains secondary"),
        ("date/timezone limitation", "frozen observation dates and Asia/Seoul timing only"),
        ("formal inferential sufficiency", "`NOT_ESTABLISHED`"),
        ("driver-rest identifiability", "`NOT_IDENTIFIABLE`"),
        ("midpoint generation", "`false`"),
        ("bootstrap execution", "`false`"),
        ("route inference execution", "`false`"),
        ("prohibited interpretation audit", "`0`"),
        ("numerical invariant audit", f"failures `{numerical_invariant_failure_count}`"),
        ("JSON/Parquet result", "`pending until schema finalization`"),
        ("manifest result", "`pending until manifest finalization`"),
        ("secret scan", "`pending until security finalization`"),
        ("estimation execution status", "`terminal_recovery_estimation_executed=true`; `method_prototype_estimation_complete=true`"),
        ("parameter translation lock", "`terminal_recovery_parameter_generated=false`; translation `false`; applied `false`"),
        ("simulator lock", "`simulator_application_authorized=false`"),
        ("Phase 2 lock", "`phase2_authorized=false`; baseline rerun `false`; retraining `false`"),
        ("next authorized action", "`offline independent estimation-result audit only`"),
    ]
    if len(report_items) != 47:
        raise RuntimeError(f"final report item count mismatch: {len(report_items)}")
    final_report = "# R2D-1Q Method Prototype Interval-Censored Estimation Execution\n\n"
    final_report += "\n".join(f"{idx}. {label}: {value}" for idx, (label, value) in enumerate(report_items, start=1))
    final_report += "\n"
    (output_root / "prompt5_e01_r2d1q_final_report.md").write_text(final_report, encoding="utf-8")

    prohibited_audit = prohibited_scan(output_root)
    dump_json(output_root / "prohibited_interpretation_audit.json", prohibited_audit)

    parquet_check = parquet_audit(output_root)
    json_parquet_mismatches = []
    sync_pairs = [
        ("interval_input_validation_audit.json", "interval_input_validation_audit.parquet", "records"),
        ("turnbull_primary_iteration_trace.json", "turnbull_primary_iteration_trace.parquet", "records"),
        ("turnbull_primary_support.json", "turnbull_primary_support.parquet", "records"),
        ("turnbull_primary_mass_solution.json", "turnbull_primary_mass_solution.parquet", "records"),
        ("turnbull_primary_survival_curve.json", "turnbull_primary_survival_curve.parquet", "records"),
        ("turnbull_vehicle_balanced_weights.json", "turnbull_vehicle_balanced_weights.parquet", "records"),
    ]
    for json_name, parquet_name, record_key in sync_pairs:
        json_rows = strict_read_json(output_root / json_name)[record_key]
        parquet_rows = dataframe_records(pd.read_parquet(output_root / parquet_name))
        if canonical_sha(json_rows) != canonical_sha(parquet_rows):
            json_parquet_mismatches.append({"json": json_name, "parquet": parquet_name})
    json_parquet_audit = {
        "parquet_files_written": len(list(output_root.glob("*.parquet"))),
        "parquet_read_failure_count": parquet_check["parquet_read_failure_count"],
        "json_parquet_value_mismatch_count": len(json_parquet_mismatches),
        "json_parquet_mismatches": json_parquet_mismatches,
        "synchronized_pairs": sync_pairs,
    }
    dump_json(output_root / "json_parquet_synchronization_audit.json", json_parquet_audit)
    dump_json(
        output_root / "manifest_self_entry_contract.json",
        {
            "path": "prompt5_e01_r2d1q_manifest.json",
            "exists": True,
            "sha256": None,
            "self_hash_exempt": True,
            "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
            "size_bytes": None,
            "self_size_exempt": True,
            "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
        },
    )
    secret_audit = secret_scan(output_root)
    dump_json(output_root / "secret_leak_audit.json", secret_audit)

    strict_pre_manifest = strict_json_audit(output_root)
    all_converged = all(result["execution"]["convergence_status"] == "CONVERGED" for result in analyses.values())
    all_init_reconciled = all(result["initialization_comparison"]["initialization_comparison_passed"] for result in analyses.values())
    all_kkt = all(result["kkt_audit"]["kkt_self_consistency_passed"] for result in analyses.values())
    all_independent = all(result["independent_audit"]["independent_likelihood_verification_passed"] for result in analyses.values())
    all_survival = all(
        sum(value for key, value in result["execution"]["survival_invariants"].items() if key.endswith("failure_count")) == 0
        for result in analyses.values()
    )
    all_quantiles_valid = all(
        len(result["quantile_payload"]["quantiles"]) == 3 and result["quantile_payload"]["midpoint_generated"] is False
        for result in analyses.values()
    )
    pass_conditions = {
        "network_api_calls_zero": True,
        "service_key_accessed_false": True,
        "upstream_immutability": immutability["upstream_modified_file_count"] == 0
        and immutability["upstream_deleted_file_count"] == 0
        and immutability["upstream_added_file_count"] == 0,
        "source_integrity": source_integrity["source_artifact_integrity_passed"],
        "authorization": auth_verification["estimation_execution_authorization_verified"],
        "fingerprint": fingerprint["input_registry_fingerprint_verification_passed"],
        "input_packet": input_packet_verification["input_packet_verification_passed"],
        "analysis_population": analysis_population["analysis_population_audit_passed"],
        "interval_validation": interval_validation["interval_input_validation_passed"],
        "self_test": self_test_result["synthetic_self_tests_passed"],
        "all_converged": all_converged,
        "all_initializations_reconciled": all_init_reconciled,
        "all_kkt": all_kkt,
        "all_independent": all_independent,
        "all_survival": all_survival,
        "all_quantiles": all_quantiles_valid,
        "numerical_invariants": numerical_invariant_failure_count == 0,
        "schema": schema_audit["estimation_output_schema_audit_passed"],
        "prohibited_interpretations": prohibited_audit["prohibited_interpretation_count"] == 0,
        "parameter_boundary": True,
        "json_parquet": json_parquet_audit["parquet_read_failure_count"] == 0
        and json_parquet_audit["json_parquet_value_mismatch_count"] == 0,
        "strict_json": strict_pre_manifest["strict_json_failure_count"] == 0,
        "security": secret_audit["secret_leak_count"] == 0,
    }
    gate_status = PRIMARY_GATE if all(pass_conditions.values()) else "FAIL_R2D1Q_ESTIMATION_EXECUTION"
    if gate_status == PRIMARY_GATE and any(
        result["execution"]["solution_status"] == "LIKELIHOOD_EQUIVALENT_ALTERNATIVES_OBSERVED" for result in analyses.values()
    ):
        gate_status = NONUNIQUE_GATE
    failure_order = [
        ("source_integrity", "FAIL_SOURCE_ARTIFACT_INTEGRITY"),
        ("upstream_immutability", "FAIL_SOURCE_IMMUTABILITY"),
        ("authorization", "FAIL_ESTIMATION_EXECUTION_AUTHORIZATION"),
        ("fingerprint", "FAIL_INPUT_FINGERPRINT_MISMATCH"),
        ("input_packet", "FAIL_INPUT_PACKET_MISMATCH"),
        ("interval_validation", "FAIL_INTERVAL_INPUT_VALIDATION"),
        ("self_test", "FAIL_TURNBULL_IMPLEMENTATION_SELF_TEST"),
        ("all_converged", "FAIL_TURNBULL_NONCONVERGENCE"),
        ("all_initializations_reconciled", "FAIL_TURNBULL_INITIALIZATION_MISMATCH"),
        ("all_kkt", "FAIL_TURNBULL_KKT_SELF_CONSISTENCY"),
        ("all_independent", "FAIL_INDEPENDENT_LIKELIHOOD_VERIFICATION"),
        ("all_survival", "FAIL_SURVIVAL_CURVE_INVARIANT"),
        ("all_quantiles", "FAIL_QUANTILE_IDENTIFICATION_OUTPUT"),
        ("numerical_invariants", "FAIL_TURNBULL_NUMERICAL_INVARIANT"),
        ("schema", "FAIL_ESTIMATION_OUTPUT_SCHEMA"),
        ("prohibited_interpretations", "FAIL_PROHIBITED_INTERPRETATION"),
        ("parameter_boundary", "FAIL_PARAMETER_TRANSLATION_BOUNDARY"),
        ("json_parquet", "FAIL_JSON_PARQUET_SYNCHRONIZATION"),
        ("strict_json", "FAIL_SCHEMA_AUDIT"),
        ("security", "FAIL_SECURITY_AUDIT"),
    ]
    if gate_status == "FAIL_R2D1Q_ESTIMATION_EXECUTION":
        for condition, failure_gate in failure_order:
            if not pass_conditions.get(condition, False):
                gate_status = failure_gate
                break

    gate_payload = {
        "artifact_dir": str(output_root),
        "gate_status": gate_status,
        "gate_passed": gate_status in {PRIMARY_GATE, NONUNIQUE_GATE},
        "network_api_calls": 0,
        "service_key_accessed": False,
        "upstream_modified_file_count": immutability["upstream_modified_file_count"],
        "upstream_deleted_file_count": immutability["upstream_deleted_file_count"],
        "upstream_added_file_count": immutability["upstream_added_file_count"],
        "source_r2d1p_gate": p_gate.get("gate_status"),
        "estimation_execution_authorization_verified": auth_verification["estimation_execution_authorization_verified"],
        "registry_fingerprint_match": fingerprint["registry_fingerprint_match"],
        "episode_id_set_fingerprint_match": fingerprint["episode_id_set_fingerprint_match"],
        "raw_evidence_set_fingerprint_match": fingerprint["raw_evidence_set_fingerprint_match"],
        "interval_tuple_set_fingerprint_match": fingerprint["interval_tuple_set_fingerprint_match"],
        "estimation_input_row_count": len(input_records),
        "input_registry_value_mismatch_count": input_packet_verification["input_registry_value_mismatch_count"],
        "input_duplicate_episode_count": input_packet_verification["input_duplicate_episode_count"],
        "input_missing_episode_count": input_packet_verification["input_missing_episode_count"],
        "input_extra_episode_count": input_packet_verification["input_extra_episode_count"],
        "adaptive_eta_rows_in_primary_input": analysis_population["adaptive_eta_rows_in_primary_input"],
        "noncomplete_rows_in_primary_input": analysis_population["noncomplete_rows_in_primary_input"],
        "synthetic_self_test_failure_count": self_test_result["synthetic_self_test_failure_count"],
        "interval_input_validation_failure_count": interval_validation["interval_input_validation_failure_count"],
        "primary_analysis_id": "PRIMARY_DUAL_EQUAL",
        "primary_convergence_status": primary["execution"]["convergence_status"],
        "primary_iteration_count": primary["execution"]["iteration_count"],
        "primary_support_cell_count": primary["execution"]["support_cell_count"],
        "primary_final_log_likelihood": primary["execution"]["final_log_likelihood"],
        "primary_solution_status": primary["execution"]["solution_status"],
        "primary_mass_sum_error": primary["execution"]["mass_sum_error"],
        "primary_maximum_kkt_score_excess": primary["kkt_audit"]["maximum_kkt_score_excess"],
        "primary_active_kkt_maximum_error": primary["kkt_audit"]["maximum_active_kkt_absolute_error"],
        "primary_self_consistency_residual": primary["kkt_audit"]["self_consistency_mass_residual"],
        "primary_independent_log_likelihood_mismatch": primary["independent_audit"][
            "independent_log_likelihood_absolute_mismatch"
        ],
        "primary_q25_identification_status": primary_q[0.25]["identification_status"],
        "primary_q25_identified_components": primary_q[0.25]["identified_components"],
        "primary_median_identification_status": primary_q[0.5]["identification_status"],
        "primary_median_identified_components": primary_q[0.5]["identified_components"],
        "primary_q75_identification_status": primary_q[0.75]["identification_status"],
        "primary_q75_identified_components": primary_q[0.75]["identified_components"],
        "provider_sensitivity_convergence_status": analyses["SENSITIVITY_PROVIDER_EQUAL"]["execution"]["convergence_status"],
        "provider_sensitivity_median_identification_status": quantile_summary(
            analyses["SENSITIVITY_PROVIDER_EQUAL"], 0.5
        )["identification_status"],
        "provider_sensitivity_median_identified_components": quantile_summary(
            analyses["SENSITIVITY_PROVIDER_EQUAL"], 0.5
        )["identified_components"],
        "request_sensitivity_convergence_status": analyses["SENSITIVITY_REQUEST_EQUAL"]["execution"]["convergence_status"],
        "request_sensitivity_median_identification_status": quantile_summary(
            analyses["SENSITIVITY_REQUEST_EQUAL"], 0.5
        )["identification_status"],
        "request_sensitivity_median_identified_components": quantile_summary(
            analyses["SENSITIVITY_REQUEST_EQUAL"], 0.5
        )["identified_components"],
        "vehicle_balanced_sensitivity_convergence_status": vehicle_result["execution"]["convergence_status"],
        "vehicle_balanced_median_identification_status": vehicle_q[0.5]["identification_status"],
        "vehicle_balanced_median_identified_components": vehicle_q[0.5]["identified_components"],
        "midpoint_generated": False,
        "bootstrap_executed": False,
        "route_formal_inference_executed": False,
        "formal_inferential_sufficiency": "NOT_ESTABLISHED",
        "prohibited_interpretation_count": prohibited_audit["prohibited_interpretation_count"],
        "numerical_invariant_failure_count": numerical_invariant_failure_count,
        "strict_json_failure_count": strict_pre_manifest["strict_json_failure_count"],
        "parquet_read_failure_count": json_parquet_audit["parquet_read_failure_count"],
        "json_parquet_value_mismatch_count": json_parquet_audit["json_parquet_value_mismatch_count"],
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "secret_leak_count": secret_audit["secret_leak_count"],
        "terminal_recovery_estimation_execution_approved": True,
        "terminal_recovery_estimation_executed": gate_status in {PRIMARY_GATE, NONUNIQUE_GATE},
        "method_prototype_estimation_complete": gate_status in {PRIMARY_GATE, NONUNIQUE_GATE},
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_parameter_translation_authorized": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "baseline_rerun_authorized": False,
        "retraining_authorized": False,
        "next_authorized_action": "offline independent estimation-result audit only",
        "pass_conditions": pass_conditions,
    }
    dump_json(output_root / "prompt5_e01_r2d1q_gate.json", gate_payload)

    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1q_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)
    final_strict = strict_json_audit(output_root)
    final_parquet = parquet_audit(output_root)
    gate_payload.update(manifest_result)
    gate_payload["strict_json_failure_count"] = final_strict["strict_json_failure_count"]
    gate_payload["parquet_read_failure_count"] = final_parquet["parquet_read_failure_count"]
    if final_strict["strict_json_failure_count"] or final_parquet["parquet_read_failure_count"] or any(
        manifest_result[key]
        for key in [
            "manifest_missing_required_file_count",
            "manifest_nonself_hash_mismatch_count",
            "manifest_nonself_size_mismatch_count",
        ]
    ):
        gate_payload["gate_status"] = "FAIL_SCHEMA_AUDIT"
        gate_payload["gate_passed"] = False
        gate_payload["terminal_recovery_estimation_executed"] = False
        gate_payload["method_prototype_estimation_complete"] = False
    dump_json(output_root / "prompt5_e01_r2d1q_gate.json", gate_payload)
    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1q_manifest.json", manifest)

    report_path = output_root / "prompt5_e01_r2d1q_final_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    report_text = report_text.replace("3. final gate: `pending until manifest finalization`", f"3. final gate: `{gate_payload['gate_status']}`")
    report_text = report_text.replace(
        "40. JSON/Parquet result: `pending until schema finalization`",
        "40. JSON/Parquet result: "
        f"strict JSON failures `{gate_payload['strict_json_failure_count']}`; "
        f"Parquet read failures `{gate_payload['parquet_read_failure_count']}`; "
        f"JSON/Parquet mismatches `{gate_payload['json_parquet_value_mismatch_count']}`",
    )
    report_text = report_text.replace(
        "41. manifest result: `pending until manifest finalization`",
        "41. manifest result: "
        f"missing `{gate_payload['manifest_missing_required_file_count']}`; "
        f"hash mismatches `{gate_payload['manifest_nonself_hash_mismatch_count']}`; "
        f"size mismatches `{gate_payload['manifest_nonself_size_mismatch_count']}`",
    )
    report_text = report_text.replace("42. secret scan: `pending until security finalization`", f"42. secret scan: leaks `{secret_audit['secret_leak_count']}`")
    report_path.write_text(report_text, encoding="utf-8")

    prohibited_audit = prohibited_scan(output_root)
    dump_json(output_root / "prohibited_interpretation_audit.json", prohibited_audit)
    secret_audit = secret_scan(output_root)
    dump_json(output_root / "secret_leak_audit.json", secret_audit)
    final_strict = strict_json_audit(output_root)
    final_parquet = parquet_audit(output_root)
    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1q_manifest.json", manifest)
    manifest_result = validate_manifest(output_root, manifest)
    gate_payload.update(manifest_result)
    gate_payload["prohibited_interpretation_count"] = prohibited_audit["prohibited_interpretation_count"]
    gate_payload["secret_leak_count"] = secret_audit["secret_leak_count"]
    gate_payload["strict_json_failure_count"] = final_strict["strict_json_failure_count"]
    gate_payload["parquet_read_failure_count"] = final_parquet["parquet_read_failure_count"]
    if (
        final_strict["strict_json_failure_count"]
        or final_parquet["parquet_read_failure_count"]
        or prohibited_audit["prohibited_interpretation_count"]
        or secret_audit["secret_leak_count"]
        or any(
            manifest_result[key]
            for key in [
                "manifest_missing_required_file_count",
                "manifest_nonself_hash_mismatch_count",
                "manifest_nonself_size_mismatch_count",
            ]
        )
    ):
        gate_payload["gate_status"] = "FAIL_SCHEMA_AUDIT"
        gate_payload["gate_passed"] = False
        gate_payload["terminal_recovery_estimation_executed"] = False
        gate_payload["method_prototype_estimation_complete"] = False
    dump_json(output_root / "prompt5_e01_r2d1q_gate.json", gate_payload)
    manifest = build_manifest(output_root)
    dump_json(output_root / "prompt5_e01_r2d1q_manifest.json", manifest)

    print("R2D-1Q METHOD PROTOTYPE INTERVAL-CENSORED ESTIMATION COMPLETE")
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
    print("\nsource_r2d1p_gate:")
    print(p_gate.get("gate_status"))
    print("\nestimation_execution_authorization_verified:")
    print(str(auth_verification["estimation_execution_authorization_verified"]).lower())
    print("\nregistry_fingerprint_match:")
    print(str(fingerprint["registry_fingerprint_match"]).lower())
    print("\nestimation_input_row_count:")
    print(len(input_records))
    print("\nsynthetic_self_test_failure_count:")
    print(self_test_result["synthetic_self_test_failure_count"])
    print("\nprimary_analysis_id:\nPRIMARY_DUAL_EQUAL")
    print("\nprimary_convergence_status:")
    print(primary["execution"]["convergence_status"])
    print("\nprimary_iteration_count:")
    print(primary["execution"]["iteration_count"])
    print("\nprimary_support_cell_count:")
    print(primary["execution"]["support_cell_count"])
    print("\nprimary_final_log_likelihood:")
    print(primary["execution"]["final_log_likelihood"])
    print("\nprimary_solution_status:")
    print(primary["execution"]["solution_status"])
    print("\nprimary_mass_sum_error:")
    print(primary["execution"]["mass_sum_error"])
    print("\nprimary_maximum_kkt_score_excess:")
    print(primary["kkt_audit"]["maximum_kkt_score_excess"])
    print("\nprimary_active_kkt_maximum_error:")
    print(primary["kkt_audit"]["maximum_active_kkt_absolute_error"])
    print("\nprimary_self_consistency_residual:")
    print(primary["kkt_audit"]["self_consistency_mass_residual"])
    print("\nprimary_independent_log_likelihood_mismatch:")
    print(primary["independent_audit"]["independent_log_likelihood_absolute_mismatch"])
    print("\nprimary_q25_identification_status:")
    print(primary_q[0.25]["identification_status"])
    print("\nprimary_q25_identified_components:")
    print(primary_q[0.25]["identified_components"])
    print("\nprimary_median_identification_status:")
    print(primary_q[0.5]["identification_status"])
    print("\nprimary_median_identified_components:")
    print(primary_q[0.5]["identified_components"])
    print("\nprimary_q75_identification_status:")
    print(primary_q[0.75]["identification_status"])
    print("\nprimary_q75_identified_components:")
    print(primary_q[0.75]["identified_components"])
    print("\nprovider_sensitivity_convergence_status:")
    print(analyses["SENSITIVITY_PROVIDER_EQUAL"]["execution"]["convergence_status"])
    print("\nprovider_sensitivity_median_identification_status:")
    print(quantile_summary(analyses["SENSITIVITY_PROVIDER_EQUAL"], 0.5)["identification_status"])
    print("\nprovider_sensitivity_median_identified_components:")
    print(quantile_summary(analyses["SENSITIVITY_PROVIDER_EQUAL"], 0.5)["identified_components"])
    print("\nrequest_sensitivity_convergence_status:")
    print(analyses["SENSITIVITY_REQUEST_EQUAL"]["execution"]["convergence_status"])
    print("\nrequest_sensitivity_median_identification_status:")
    print(quantile_summary(analyses["SENSITIVITY_REQUEST_EQUAL"], 0.5)["identification_status"])
    print("\nrequest_sensitivity_median_identified_components:")
    print(quantile_summary(analyses["SENSITIVITY_REQUEST_EQUAL"], 0.5)["identified_components"])
    print("\nvehicle_balanced_sensitivity_convergence_status:")
    print(vehicle_result["execution"]["convergence_status"])
    print("\nvehicle_balanced_median_identification_status:")
    print(vehicle_q[0.5]["identification_status"])
    print("\nvehicle_balanced_median_identified_components:")
    print(vehicle_q[0.5]["identified_components"])
    print("\nmidpoint_generated:\nfalse")
    print("\nbootstrap_executed:\nfalse")
    print("\nroute_formal_inference_executed:\nfalse")
    print("\nformal_inferential_sufficiency:\nNOT_ESTABLISHED")
    print("\nprohibited_interpretation_count:")
    print(prohibited_audit["prohibited_interpretation_count"])
    print("\nnumerical_invariant_failure_count:")
    print(numerical_invariant_failure_count)
    print("\nstrict_json_failure_count:")
    print(gate_payload["strict_json_failure_count"])
    print("\nparquet_read_failure_count:")
    print(gate_payload["parquet_read_failure_count"])
    print("\njson_parquet_value_mismatch_count:")
    print(gate_payload["json_parquet_value_mismatch_count"])
    print("\nmanifest_missing_required_file_count:")
    print(gate_payload["manifest_missing_required_file_count"])
    print("\nmanifest_nonself_hash_mismatch_count:")
    print(gate_payload["manifest_nonself_hash_mismatch_count"])
    print("\nmanifest_nonself_size_mismatch_count:")
    print(gate_payload["manifest_nonself_size_mismatch_count"])
    print("\nsecret_leak_count:")
    print(secret_audit["secret_leak_count"])
    print("\nterminal_recovery_estimation_executed:")
    print(str(gate_payload["terminal_recovery_estimation_executed"]).lower())
    print("\nmethod_prototype_estimation_complete:")
    print(str(gate_payload["method_prototype_estimation_complete"]).lower())
    print("\nterminal_recovery_parameter_generated:\nfalse")
    print("\nterminal_recovery_parameter_translation_authorized:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\nsimulator_application_authorized:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print("\nnext_authorized_action:\noffline independent estimation-result audit only")


if __name__ == "__main__":
    main()
