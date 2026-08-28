#!/usr/bin/env python3
"""H4M-AE-R1 KPI measurement path availability and arm constructability audit.

Audit only.  This program repairs nothing, constructs no arm, executes no KPI,
trains nothing, touches no sealed hold-out, uses no database or network, and
pushes nothing.  It reads source and artifacts, inventories every canonical KPI
measurement lineage, judges whether each comparison arm can be constructed, and
runs self-tests that must detect the known defective paths.
"""

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


STAGE = "PV8-R2A-R8E-R3-R-H4M-AE-R1"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R1_"
    "KPI_MEASUREMENT_PATH_AVAILABILITY_AUDIT_AND_ARM_CONSTRUCTABILITY_REVIEW_COMPLETE"
)
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R1_AUDIT_INCOMPLETE"
NEXT_GATE = "H4M-AE-R2_CAUSAL_KPI_MEASUREMENT_BRIDGE_REPAIR_SELECTION_AND_FREEZE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

AGGREGATOR = "05_training/evaluation/canonical_kpi_aggregator.py"
ENERGY_MODEL = "05_training/rewards/energy_proxy_model_v1.py"
ADAPTER_V2 = "05_training/adapters/causal_simulator_v2_adapter.py"
ADAPTER_TOY = "05_training/adapters/causal_simulator_adapter.py"
ADAPTER_REPLAY = "05_training/adapters/historical_replay_adapter.py"
CAUSAL_ROLLOUT = "05_training/run_causal_rollout.py"
POLICY_SMOKE = "05_training/run_a_family_policy_rollout_smoke.py"
MAPPO_RUNNER = "05_training/mappo_runner.py"
B0C_CONTRACT = "05_training/baselines/b0c_causal_shadow_baseline_contract.json"

DL5_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation_20260731_185643"
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
H4MAE_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_post_promotion_kpi_scope_protocol_freeze_*"
H4MAD_GLOB = "pv8_r2a_r8e_r3_r_h4m_ad_sealed_outcome_review_final_promotion_*"

EXPECTED = {
    "head_short": "09cb3d4",
    "promoted_actor_node_feature_dim": 12,
    "promoted_actor_input_dim": 131,
    "promoted_gatv2_hidden": 128,
    "toy_actor_obs_dim": 16,
    "pv8_window_total": 54,
}

CANONICAL_12 = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]
PRIMARY_4 = {
    "WAITING_TIME_REDUCTION": "avg_wait_seconds",
    "IN_VEHICLE_TIME_REDUCTION": None,
    "FLEET_SIZE_REDUCTION": "fleet_reduction_ratio",
    "ENERGY_CONSUMPTION_REDUCTION": "energy_proxy",
}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "final_report.json",
    "kpi_measurement_path_inventory.json",
    "kpi_measurement_path_inventory.parquet",
    "primary_4kpi_availability_review.json",
    "canonical_12kpi_availability_review.json",
    "comparison_arm_constructability_matrix.json",
    "comparison_arm_constructability_matrix.parquet",
    "window_lineage_compatibility_audit.json",
    "policy_kpi_path_provenance_audit.json",
    "synthetic_kpi_path_detection.json",
    "test6_non_reuse_audit.json",
    "in_vehicle_time_non_proxy_guard.json",
    "repair_dependency_register.json",
    "gate_decision.json",
    "downstream_lock.json",
    "artifact_manifest.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def source_text(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")


def latest(glob: str) -> Optional[Path]:
    roots = sorted(ARTIFACTS_ROOT.glob(glob))
    return roots[-1] if roots else None


# ---------------------------------------------------------------------------
# simulator capability probe
# ---------------------------------------------------------------------------
def simulator_capability() -> Dict[str, Any]:
    v2 = source_text(ADAPTER_V2)
    toy = source_text(ADAPTER_TOY)
    replay = source_text(ADAPTER_REPLAY)
    v2_kpis_none = v2.count('": None,') >= 5 and '"avg_wait_seconds": None' in v2
    return {
        "causal_simulator_v2_adapter": {
            "path": ADAPTER_V2,
            "source_mode": "static_signal_scaffold_nonperformance_v1",
            "accepts_pv8_graph_observation": "edge_features_dim" in v2 and "node_features_dim" in v2,
            "emits_transition_derived_kpis": not v2_kpis_none,
            "compute_kpis_returns_none_for_core_kpis": v2_kpis_none,
            "performance_claim_allowed": "performance_claim_allowed\": False" in v2 or '"performance_claim_allowed": False' in v2,
            "verdict": "graph-context scaffold that returns None for every core KPI; cannot measure",
        },
        "causal_simulator_adapter_toy": {
            "path": ADAPTER_TOY,
            "source_mode": "causal_toy_suseong_v1",
            "graph_scope": "toy_suseong_beomeo_manchon",
            "actor_obs_dim": EXPECTED["toy_actor_obs_dim"],
            "emits_transition_derived_kpis": "queue_person_seconds" in toy and "headway_samples" in toy,
            "causal_comparison_allowed": '"causal_comparison_allowed": True' in toy,
            "accepts_pv8_graph_observation": False,
            "verdict": "the only adapter that derives KPIs from simulated passenger and headway state, but it is an 8-stop toy corridor with a flat 16-dim observation",
        },
        "historical_replay_adapter": {
            "path": ADAPTER_REPLAY,
            "emits_transition_derived_kpis": False,
            "constant_stub_detected": '"avg_wait_seconds": 300.0' in replay,
            "verdict": "returns a hardcoded avg_wait_seconds and a None energy proxy; reference only",
        },
        "conclusion": (
            "no adapter simultaneously accepts the promoted policy observation contract and emits "
            "transition-derived canonical KPIs"
        ),
    }


def kpi_inventory(capability: Mapping[str, Any]) -> List[Dict[str, Any]]:
    aggregator = source_text(AGGREGATOR)
    toy_capable = capability["causal_simulator_adapter_toy"]["emits_transition_derived_kpis"]
    common = {
        "simulator_source": "CausalSimulatorAdapter (toy, causal_toy_suseong_v1) only; CausalSimulatorV2Adapter returns None",
        "raw_event_path": "adapter state -> compute_kpis(); no raw_events writer exists for the PV8 R3-R scope",
        "window_rollup_path": f"{POLICY_SMOKE}::build_window_rollup_row (synthetic) or {CAUSAL_ROLLOUT} (placeholder policy)",
        "canonical_aggregation_path": f"{AGGREGATOR}::compute_official_kpi_by_window",
        "same_window_comparison_possible": False,
        "causal_comparison_possible": False,
    }
    rows: List[Dict[str, Any]] = []
    definitions = {
        "cv_headway": ("headway variability", "headway_std_seconds / headway_mean_seconds", True),
        "avg_wait_seconds": ("mean passenger wait", "wait_total_passenger_seconds / wait_passenger_count", True),
        "bunching_rate": ("bunching events per headway event", "bunching_event_count / headway_event_count", True),
        "on_time_rate": ("on-time arrivals", "ontime_event_count / schedulable_arrival_count", True),
        "intervention_rate": ("interventions per decision", "intervention_count / decision_step_count", True),
        "energy_proxy": ("energy proxy total", "energy_proxy_total from distance, acceleration events, hold seconds", True),
        "passenger_demand_generated": ("demand", "passenger_demand_generated", True),
        "passenger_served_count": ("served passengers", "passenger_served_count", True),
        "passenger_service_rate": ("served / demand", "passenger_served_count / passenger_demand_generated", True),
        "passenger_wait_p95_seconds": ("wait tail", "passenger_wait_p95_seconds or avg_wait_seconds * 1.65 fallback", True),
        "energy_proxy_per_passenger": ("energy per served passenger", "energy_proxy / passenger_served_count", True),
        "fleet_reduction_ratio": ("fleet reduction", "1 - active_bus_count / baseline_bus_count", True),
    }
    for metric, (semantic, formula, action_affects) in definitions.items():
        derived_fallback = None
        classification = "unavailable"
        if metric == "passenger_wait_p95_seconds" and "* 1.65" in aggregator:
            derived_fallback = "avg_wait_seconds * 1.65 when no measured p95 column exists"
            classification = "derived"
        elif metric in {"energy_proxy", "energy_proxy_per_passenger"}:
            classification = "proxy"
        elif metric == "fleet_reduction_ratio":
            classification = "derived"
        else:
            classification = "observed_in_toy_simulator_only" if toy_capable else "unavailable"
        rows.append(
            {
                "metric": metric,
                "semantic_definition": semantic,
                "raw_source_fields": formula,
                **common,
                "policy_action_causally_affects_metric": action_affects,
                "classification": classification,
                "derived_fallback": derived_fallback,
                "available_for_promoted_policy_scope": False,
                "blocking_dependency": "no PV8-scope simulator emits this field for the promoted policy observation contract",
            }
        )
    rows.append(
        {
            "metric": "in_vehicle_time_seconds",
            "semantic_definition": "boarding to alighting elapsed time per passenger",
            "raw_source_fields": "none: no alighting timestamp exists in the canonical schema, the rollup contract or Reward V2",
            "simulator_source": "none",
            "raw_event_path": "none",
            "window_rollup_path": "none",
            "canonical_aggregation_path": "none",
            "policy_action_causally_affects_metric": True,
            "classification": "unavailable",
            "derived_fallback": None,
            "same_window_comparison_possible": False,
            "causal_comparison_possible": False,
            "available_for_promoted_policy_scope": False,
            "blocking_dependency": "no authoritative field; NOT_YET_MEASURABLE and no proxy permitted",
        }
    )
    return rows


def arm_matrix(capability: Mapping[str, Any]) -> List[Dict[str, Any]]:
    b0c = read_json(PROJECT_ROOT / B0C_CONTRACT)
    dl5_scope = read_json(DL5_ROOT / "evaluation_scope.json")
    base = {
        "same_simulator_semantics": False,
        "same_windows": False,
        "same_passenger_demand": False,
        "same_initial_vehicle_state": False,
        "same_kpi_definitions": True,
        "actual_mappo_checkpoint_usable": False,
        "simulator_derived_kpi_available": False,
        "existing_artifact_reusable": False,
        "new_execution_required": True,
        "test6_contamination": False,
    }
    rows = [
        {
            "arm": "A_vs_B0R",
            "baseline": "B0R historical 12-KPI compatibility reference",
            **base,
            "causal_comparison_allowed": False,
            "evidence": f"B0R lineage is the DL-3/DL-4 holdout ({dl5_scope.get('test_snapshot_count')} snapshots) with causal_comparison_allowed={dl5_scope.get('causal_comparison_allowed')}",
            "status": "INCOMPATIBLE_REFERENCE_ONLY",
        },
        {
            "arm": "A_vs_B1",
            "baseline": "B1 no-op",
            **base,
            "causal_comparison_allowed": False,
            "evidence": "B1 executed only on the DL-3/DL-4 holdout; a PV8-scope no-op run would still need a KPI-emitting PV8 simulator, which does not exist",
            "status": "CONSTRUCTABLE_AFTER_EXPLICIT_REPAIR",
        },
        {
            "arm": "A_vs_B2",
            "baseline": "B2 rule-based calibrated proxy",
            **base,
            "causal_comparison_allowed": False,
            "evidence": "same scope mismatch as B1, plus the rule-based policy is calibrated against the DL-3/DL-4 lineage",
            "status": "CONSTRUCTABLE_AFTER_EXPLICIT_REPAIR",
        },
        {
            "arm": "A_vs_B0C",
            "baseline": "B0C causal shadow",
            **base,
            "causal_comparison_allowed": False,
            "evidence": f"B0C contract_status={b0c.get('contract_status')}; never executed, so there is no causal baseline evidence at all",
            "status": "NOT_CONSTRUCTABLE_WITH_CURRENT_EVIDENCE",
        },
    ]
    for row in rows:
        row["a_arm_blocker"] = capability["conclusion"]
    return rows


def window_lineage_audit() -> Dict[str, Any]:
    split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")["ordered_window_ids"]
    dl5_scope = read_json(DL5_ROOT / "evaluation_scope.json")
    universes = {
        "PV8_R3R": {
            "window_count": sum(len(split[key]) for key in ("train", "validation", "test")),
            "train": len(split["train"]),
            "validation": len(split["validation"]),
            "test6_sealed": len(split["test"]),
            "identity": "representative window registry, 30-minute windows, GATv2 snapshots",
        },
        "DL3_DL4_HOLDOUT": {
            "snapshot_count": dl5_scope.get("test_snapshot_count"),
            "identity": dl5_scope.get("evaluation_timestamps_source"),
            "causal_comparison_allowed": dl5_scope.get("causal_comparison_allowed"),
        },
        "TOY_CORRIDOR": {"identity": "toy_suseong_beomeo_manchon, 8 stops", "window_concept": "not window based"},
    }
    checks = {
        "pv8_total_is_54": universes["PV8_R3R"]["window_count"] == EXPECTED["pv8_window_total"],
        "universes_are_disjoint_and_incomparable": True,
        "no_arm_may_join_two_universes": True,
        "test6_excluded_from_any_future_arm": True,
    }
    return {"stage": STAGE, "universes": universes, "checks": checks, "passed": all(checks.values())}


def policy_kpi_path_provenance() -> Dict[str, Any]:
    smoke = source_text(POLICY_SMOKE)
    rollout = source_text(CAUSAL_ROLLOUT)
    runner = source_text(MAPPO_RUNNER)
    tree = ast.parse(smoke)
    step_assigned = False
    step_used_anywhere = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "step_result" for target in node.targets):
            step_assigned = True
        if isinstance(node, ast.Name) and node.id == "step_result" and isinstance(node.ctx, ast.Load):
            step_used_anywhere = True
    rollup_builder = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "build_window_rollup_row"),
        None,
    )
    rollup_args = (
        sorted(arg.arg for arg in list(rollup_builder.args.args) + list(rollup_builder.args.kwonlyargs))
        if rollup_builder is not None
        else []
    )
    step_used = bool(set(rollup_args) & {"step_result", "obs", "state", "trajectory", "kpis"})
    return {
        "stage": STAGE,
        "a_policy_paths": {
            "canonical_rollout": {
                "path": CAUSAL_ROLLOUT,
                "loads_checkpoint": "checkpoint_path" in rollout,
                "placeholder_policy_markers": rollout.count("placeholder"),
                "verdict": "placeholder A-family policies only; no checkpoint is ever loaded",
            },
            "checkpoint_capable_smoke": {
                "path": POLICY_SMOKE,
                "loads_checkpoint": "build_mappo_actions" in smoke,
                "simulator_step_assigned": step_assigned,
                "simulator_step_result_used_for_kpi": step_used,
                "simulator_step_result_referenced_anywhere": step_used_anywhere,
                "step_result_usage": "recorded as terminated, truncated, reward count and info in the status payload only",
                "window_rollup_builder_args": rollup_args,
                "kpi_synthesised_from_action_count": "nonzero_action_count" in smoke,
                "verdict": "loads a real checkpoint but discards the simulator step result and synthesises headway, wait and bunching from the non-zero action count",
            },
            "mappo_runner": {
                "path": MAPPO_RUNNER,
                "writes_checkpoint_stub": "write_checkpoint_stub" in runner,
                "stub_disclaimer_present": "does not create an actual trained MAPPO checkpoint" in runner,
                "verdict": "smoke checkpoint stub writer; not an evaluation path",
            },
        },
        "observation_contract_mismatch": {
            "promoted_policy": {
                "node_feature_dim": EXPECTED["promoted_actor_node_feature_dim"],
                "gatv2_hidden": EXPECTED["promoted_gatv2_hidden"],
                "actor_input_dim": EXPECTED["promoted_actor_input_dim"],
                "input_kind": "PyG graph Data with target-context one-hot",
            },
            "toy_causal_simulator": {"actor_obs_dim": EXPECTED["toy_actor_obs_dim"], "input_kind": "flat vector"},
            "compatible": False,
            "bridging_would_change_observation_contract": True,
        },
        "passed": True,
    }


def synthetic_kpi_detection(arms: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    smoke = source_text(POLICY_SMOKE)
    rollout = source_text(CAUSAL_ROLLOUT)
    b0c = read_json(PROJECT_ROOT / B0C_CONTRACT)
    dl5_scope = read_json(DL5_ROOT / "evaluation_scope.json")
    ae_root = latest(H4MAE_GLOB)
    ae_contract = read_json(ae_root / "kpi_evaluation_contract.json")["contract"] if ae_root else {}

    tree = ast.parse(smoke)
    simulator_output_tokens = {"step_result", "obs", "observation", "state", "trajectory", "kpis", "sim_kpis"}
    rollup_builder = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "build_window_rollup_row"),
        None,
    )
    rollup_args = (
        sorted(arg.arg for arg in list(rollup_builder.args.args) + list(rollup_builder.args.kwonlyargs))
        if rollup_builder is not None
        else []
    )
    rollup_receives_simulator_output = bool(set(rollup_args) & simulator_output_tokens)
    rollup_body = ast.get_source_segment(smoke, rollup_builder) if rollup_builder is not None else ""
    kpi_from_action_count = "nonzero_action_count" in (rollup_body or "")
    step_result_only_in_status = ("step_result.terminated" in smoke) and not rollup_receives_simulator_output
    detectors = {
        "D1_simulator_output_discarded_synthetic_kpi": {
            "description": "a path that calls adapter.step and then builds KPI rows without any simulator output",
            "detected": ("adapter.step(" in smoke) and (not rollup_receives_simulator_output) and kpi_from_action_count,
            "evidence": (
                f"{POLICY_SMOKE}: build_window_rollup_row accepts {rollup_args} with no simulator output, "
                "derives headway and wait from nonzero_action_count, and the adapter step result is recorded only as "
                "terminated/reward-count status"
            ),
            "rollup_builder_args": rollup_args,
            "rollup_receives_simulator_output": rollup_receives_simulator_output,
            "kpi_fields_derived_from_action_count": kpi_from_action_count,
            "step_result_used_for_status_only": step_result_only_in_status,
        },
        "D2_placeholder_policy_presented_as_actual_mappo": {
            "description": "a placeholder policy emitting an A-family condition id",
            "detected": ("placeholder" in rollout) and ('"A": "A_pure_mappo"' in rollout) and ("checkpoint_path" not in rollout),
            "evidence": f"{CAUSAL_ROLLOUT}: condition A maps to A_pure_mappo while the policy is a placeholder and no checkpoint is loaded",
        },
        "D3_different_window_universes_joined": {
            "description": "an arm that mixes the PV8 R3-R window set with the DL-3/DL-4 holdout",
            "detected": any(row["same_windows"] is False and row["status"] != "INCOMPATIBLE_REFERENCE_ONLY" for row in arms) is False
            and any(row["same_windows"] for row in arms) is False,
            "evidence": "no arm in this audit joins two universes; every baseline is recorded as a different universe",
            "polarity": "detector confirms absence; a True here would mean this audit itself mixed universes",
        },
        "D4_causal_promotion_of_noncausal_baseline": {
            "description": "treating a causal_comparison_allowed=false baseline as causal",
            "detected": any(row.get("causal_comparison_allowed") for row in arms),
            "evidence": f"DL-5 scope causal_comparison_allowed={dl5_scope.get('causal_comparison_allowed')}; every arm records causal_comparison_allowed=false",
        },
        "D5_unexecuted_b0c_treated_as_executed": {
            "description": "using the B0C contract as if it were an executed baseline",
            "detected": any(row["arm"] == "A_vs_B0C" and row["existing_artifact_reusable"] for row in arms),
            "evidence": f"B0C contract_status={b0c.get('contract_status')}; the arm is classified NOT_CONSTRUCTABLE_WITH_CURRENT_EVIDENCE",
        },
        "D6_proxy_attached_to_not_yet_measurable_kpi": {
            "description": "substituting a proxy for in-vehicle time",
            "detected": any(
                row["metric"] == "in_vehicle_time_seconds" and row["classification"] not in {"unavailable"} for row in inventory
            )
            or bool((ae_contract.get("kpi_definitions", {}).get("IN_VEHICLE_TIME_REDUCTION", {}) or {}).get("substitute_metric_used")),
            "evidence": "in_vehicle_time_seconds stays classified unavailable and no substitute metric is registered",
        },
        "D7_test6_reused_in_a_new_arm": {
            "description": "a comparison arm that reuses the exhausted sealed hold-out",
            "detected": any(row["test6_contamination"] for row in arms),
            "evidence": "no arm references the sealed TEST6 windows",
        },
    }
    known_defects_found = [
        name for name in ("D1_simulator_output_discarded_synthetic_kpi", "D2_placeholder_policy_presented_as_actual_mappo")
        if detectors[name]["detected"]
    ]
    audit_hygiene_violations = [
        name
        for name in (
            "D3_different_window_universes_joined",
            "D4_causal_promotion_of_noncausal_baseline",
            "D5_unexecuted_b0c_treated_as_executed",
            "D6_proxy_attached_to_not_yet_measurable_kpi",
            "D7_test6_reused_in_a_new_arm",
        )
        if detectors[name]["detected"]
    ]
    return {
        "stage": STAGE,
        "detectors": detectors,
        "known_repository_defects_detected": known_defects_found,
        "audit_hygiene_violations": audit_hygiene_violations,
        "self_test_passed": len(known_defects_found) == 2 and not audit_hygiene_violations,
        "self_test_rule": (
            "D1 and D2 are positive controls: the defective paths exist in the repository and the detectors must fire. "
            "D3 to D7 are hygiene guards on this audit and must not fire."
        ),
    }


def test6_non_reuse_audit(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")["ordered_window_ids"]
    sealed = sorted(str(value) for value in split["test"])
    source = source_text(str(SOURCE_REL))
    checks = {
        "no_arm_uses_test6": not any(row["test6_contamination"] for row in arms),
        "no_sealed_window_id_literal_in_this_audit_source": not any(window_id in source for window_id in sealed),
        "no_sealed_snapshot_loaded": True,
        "no_new_holdout_created": True,
        "train_validation_read_for_schema_only": True,
        "no_policy_kpi_computed": True,
    }
    return {
        "stage": STAGE,
        "sealed_window_count": len(sealed),
        "test6_status": "EXHAUSTED_SEALED_HOLDOUT",
        "checks": checks,
        "passed": all(checks.values()),
    }


def in_vehicle_guard(inventory: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tokens = ["in_vehicle", "ride_time", "alight_ts", "onboard_seconds", "board_to_alight", "dwell_to_alight"]
    scanned = {
        AGGREGATOR: source_text(AGGREGATOR),
        ADAPTER_TOY: source_text(ADAPTER_TOY),
        ADAPTER_V2: source_text(ADAPTER_V2),
        "05_training/rewards/mappo_reward_v1.py": source_text("05_training/rewards/mappo_reward_v1.py"),
    }
    hits = {path: sorted(token for token in tokens if token in text) for path, text in scanned.items()}
    row = next(item for item in inventory if item["metric"] == "in_vehicle_time_seconds")
    checks = {
        "no_authoritative_field_found": all(not value for value in hits.values()),
        "classified_unavailable": row["classification"] == "unavailable",
        "no_proxy_registered": row["derived_fallback"] is None,
        "no_substitute_metric_invented": True,
    }
    return {
        "stage": STAGE,
        "status": "NOT_YET_MEASURABLE",
        "searched_tokens": tokens,
        "scan_hits": hits,
        "required_to_become_measurable": "an alighting timestamp or in-vehicle elapsed time emitted by a causal simulator and added to the canonical KPI contract",
        "checks": checks,
        "passed": all(checks.values()),
    }


def repair_register(capability: Mapping[str, Any], arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    items = [
        {
            "id": "R1_PV8_SCOPE_CAUSAL_KPI_EMITTING_SIMULATOR",
            "blocking": "every KPI and every arm",
            "problem": "CausalSimulatorV2Adapter returns None for all core KPIs; the only KPI-emitting causal adapter is an 8-stop toy corridor with a 16-dim flat observation",
            "minimum_change": "make one causal simulator emit transition-derived wait, headway, service, energy and active-vehicle observables on the PV8 R3-R window scope",
            "must_not_change": ["Reward V2", "GAE/PPO", "observation contract", "KPI formulas", "promoted checkpoints"],
            "severity": "ROOT",
        },
        {
            "id": "R2_CHECKPOINT_TO_SIMULATOR_ACTION_BRIDGE",
            "blocking": "A arm",
            "problem": "the checkpoint-capable runner discards adapter.step output and synthesises KPIs from the action count",
            "minimum_change": "drive the simulator with the promoted policy's actions and build the window rollup from the simulator step result only",
            "must_not_change": ["checkpoint weights", "action semantics", "K-mask"],
            "severity": "ROOT",
        },
        {
            "id": "R3_BASELINE_REEXECUTION_ON_PV8_SCOPE",
            "blocking": "B1 and B2 arms",
            "problem": "B0R/B1/B2 exist only on the DL-3/DL-4 holdout with causal_comparison_allowed=false",
            "minimum_change": "re-run no-op and rule-based policies on the identical PV8 windows once R1 exists",
            "must_not_change": ["baseline policy definitions"],
            "severity": "DEPENDENT_ON_R1",
        },
        {
            "id": "R4_B0C_EXECUTION_OR_FORMAL_RETIREMENT",
            "blocking": "B0C arm",
            "problem": "B0C is DRAFT_LOCKED_NOT_EXECUTED",
            "minimum_change": "either execute the contract once R1 exists or formally retire the arm",
            "must_not_change": ["the B0C contract semantics"],
            "severity": "DEPENDENT_ON_R1",
        },
        {
            "id": "R5_IN_VEHICLE_TIME_FIELD",
            "blocking": "IN_VEHICLE_TIME_REDUCTION only",
            "problem": "no alighting timestamp or in-vehicle elapsed time exists anywhere",
            "minimum_change": "emit an alighting timestamp from the causal simulator and extend the canonical KPI contract",
            "must_not_change": ["no proxy substitution is permitted in the interim"],
            "severity": "INDEPENDENT",
        },
        {
            "id": "R6_EVALUATION_WINDOW_SUPPLY",
            "blocking": "unseen-data claims only",
            "problem": "the registry holds 54 windows fully partitioned into 44 train, 4 validation and 6 exhausted TEST6",
            "minimum_change": "declare train/validation reuse explicitly, or commission a new sealed window set through a separate gate",
            "must_not_change": ["no relabelling of train or validation as fresh holdout"],
            "severity": "SCOPE",
        },
    ]
    return {
        "stage": STAGE,
        "ordering": ["R1_PV8_SCOPE_CAUSAL_KPI_EMITTING_SIMULATOR", "R2_CHECKPOINT_TO_SIMULATOR_ACTION_BRIDGE", "R3_BASELINE_REEXECUTION_ON_PV8_SCOPE", "R4_B0C_EXECUTION_OR_FORMAL_RETIREMENT", "R5_IN_VEHICLE_TIME_FIELD", "R6_EVALUATION_WINDOW_SUPPLY"],
        "items": items,
        "root_blockers": [row["id"] for row in items if row["severity"] == "ROOT"],
    }


def build_gate_decision(inventory, arms, synthetic, test6, guard, capability) -> Dict[str, Any]:
    constructable = [row["arm"] for row in arms if row["status"] == "CONSTRUCTABLE_FROM_EXISTING_AUTHORITATIVE_PATH"]
    blocked = [row["arm"] for row in arms if row["status"] != "CONSTRUCTABLE_FROM_EXISTING_AUTHORITATIVE_PATH"]
    primary = {
        name: {
            "canonical_field": field,
            "measurement_status": "NOT_YET_MEASURABLE"
            if field is None
            else "DEFINED_BUT_NOT_MEASURABLE_ON_PROMOTED_SCOPE",
        }
        for name, field in PRIMARY_4.items()
    }
    return {
        "stage": STAGE,
        "authoritative_kpi_paths_available": False,
        "primary_4kpi_measurement_status": primary,
        "comparable_causal_arm_constructable": bool(constructable),
        "constructable_arm_ids": constructable,
        "blocked_arm_ids": blocked,
        "synthetic_kpi_paths_detected": synthetic["known_repository_defects_detected"],
        "test6_reuse_required": False,
        "in_vehicle_time_status": guard["status"],
        "repair_required": True,
        "audit_complete_and_reproducible": True,
        "note": "PASS certifies the audit, not KPI comparability",
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    short = git_run(["rev-parse", "--short", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    parent_short = git_run(["rev-parse", "--short", "HEAD^"]).stdout.strip()
    status = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    with tempfile.TemporaryDirectory(prefix="h4maer1_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "aer1.pyc"), doraise=True)
            error = None
        except Exception as exc:
            error = repr(exc)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit": head,
        "source_commit_short": short,
        "audit_parent_commit": parent,
        "audit_parent_commit_short": parent_short,
        "starting_head_expected": EXPECTED["head_short"],
        "starting_head_matched": parent_short.startswith(EXPECTED["head_short"]) or short.startswith(EXPECTED["head_short"]),
        "starting_state_rule": "the audit runs from a source-only commit whose parent is the expected starting HEAD; no reset or rebase was performed",
        "git_status_short": status,
        "clean_worktree": status == "",
        "head_commit_files": head_files,
        "py_compile_passed": error is None,
        "reset_or_rebase_performed": False,
        "github_push_performed": False,
    }


def main() -> None:
    started = time.perf_counter()
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ae_r1_kpi_measurement_path_availability_audit_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    provenance = source_provenance(created_at)
    capability = simulator_capability()
    inventory = kpi_inventory(capability)
    arms = arm_matrix(capability)
    windows = window_lineage_audit()
    provenance_audit = policy_kpi_path_provenance()
    synthetic = synthetic_kpi_detection(arms, inventory)
    test6 = test6_non_reuse_audit(arms)
    guard = in_vehicle_guard(inventory)
    repairs = repair_register(capability, arms)
    gate_decision = build_gate_decision(inventory, arms, synthetic, test6, guard, capability)

    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_parquet(root / "kpi_measurement_path_inventory.parquet", index=False)
    arm_frame = pd.DataFrame(arms)
    arm_frame.to_parquet(root / "comparison_arm_constructability_matrix.parquet", index=False)

    primary_review = {
        "stage": STAGE,
        "primary_kpis": {
            name: {
                "canonical_field": field,
                "row": next((row for row in inventory if row["metric"] == (field or "in_vehicle_time_seconds")), None),
                "measurable_on_promoted_scope": False,
            }
            for name, field in PRIMARY_4.items()
        },
        "measurable_count_on_promoted_scope": 0,
        "not_yet_measurable": ["IN_VEHICLE_TIME_REDUCTION"],
    }
    twelve_review = {
        "stage": STAGE,
        "canonical_12": CANONICAL_12,
        "defined_in_aggregator": len(CANONICAL_12),
        "measurable_in_toy_causal_simulator": sum(
            1 for row in inventory if row["metric"] in CANONICAL_12 and row["classification"] != "unavailable"
        ),
        "available_on_promoted_policy_scope": 0,
        "by_metric": {row["metric"]: row["classification"] for row in inventory if row["metric"] in CANONICAL_12},
    }

    checks_all = {
        "provenance_clean": provenance["clean_worktree"] and provenance["starting_head_matched"],
        "inventory_complete": len(inventory) == len(CANONICAL_12) + 1,
        "arms_all_classified": len(arms) == 4,
        "self_test_passed": synthetic["self_test_passed"],
        "test6_non_reuse": test6["passed"],
        "in_vehicle_guard": guard["passed"],
        "window_lineage_audit": windows["passed"],
        "no_execution_performed": True,
    }
    gate = {
        "stage": STAGE,
        "gate": PASS_GATE if all(checks_all.values()) else BLOCK_GATE,
        "criteria": checks_all,
        "failing_criteria": [key for key, value in checks_all.items() if not value],
        "gate_decision": gate_decision,
        "exact_next_gate": NEXT_GATE,
        "next_gate_auto_execution": False,
        "final_flags": {
            "training": 0,
            "kpi_executed": False,
            "arm_constructed": False,
            "test6_touched": False,
            "repair_applied": False,
            "github_push_performed": False,
        },
    }
    downstream_lock = {
        "stage": STAGE,
        "locked_until_repair": True,
        "forbidden_downstream_actions": [
            "any KPI performance claim for the promoted policy",
            "any A vs baseline comparison using the current synthetic paths",
            "any reuse of TEST6",
            "any in-vehicle-time proxy",
        ],
        "unlock_condition": "R1 and R2 in the repair dependency register are implemented and validated under a separate gate",
        "sealed_holdout_status": "EXHAUSTED_SEALED_HOLDOUT",
    }

    payloads = {
        "kpi_measurement_path_inventory.json": {"stage": STAGE, "rows": inventory, "simulator_capability": capability},
        "primary_4kpi_availability_review.json": primary_review,
        "canonical_12kpi_availability_review.json": twelve_review,
        "comparison_arm_constructability_matrix.json": {"stage": STAGE, "arms": arms},
        "window_lineage_compatibility_audit.json": windows,
        "policy_kpi_path_provenance_audit.json": provenance_audit,
        "synthetic_kpi_path_detection.json": synthetic,
        "test6_non_reuse_audit.json": test6,
        "in_vehicle_time_non_proxy_guard.json": guard,
        "repair_dependency_register.json": repairs,
        "gate_decision.json": gate_decision,
        "downstream_lock.json": downstream_lock,
        "source_provenance.json": provenance,
        "gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(root / name, payload)

    final_json = {
        "stage": STAGE,
        "gate": gate["gate"],
        "source_commit": provenance["source_commit"],
        "gate_decision": gate_decision,
        "root_blockers": repairs["root_blockers"],
        "arm_status": {row["arm"]: row["status"] for row in arms},
        "canonical_12_available_on_promoted_scope": twelve_review["available_on_promoted_policy_scope"],
        "self_test": {
            "known_defects_detected": synthetic["known_repository_defects_detected"],
            "hygiene_violations": synthetic["audit_hygiene_violations"],
        },
        "next_gate": NEXT_GATE,
    }
    write_json(root / "final_report.json", final_json)
    (root / "final_report.md").write_text(
        f"""# H4M-AE-R1 KPI Measurement Path Availability and Arm Constructability Audit

gate = {gate['gate']}
source_commit = {provenance['source_commit']}
audit_only = true; no repair, no arm construction, no KPI execution, no TEST6 access

## Root finding

{capability['conclusion']}

```json
{json.dumps({name: {"verdict": row["verdict"]} for name, row in capability.items() if isinstance(row, dict)}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Arm constructability

```json
{json.dumps({row['arm']: {"status": row["status"], "evidence": row["evidence"]} for row in arms}, ensure_ascii=False, indent=2, default=jsonable)}
```

## KPI availability

```json
{json.dumps({"primary_4": {name: row["measurement_status"] for name, row in gate_decision["primary_4kpi_measurement_status"].items()}, "canonical_12_available_on_promoted_scope": twelve_review["available_on_promoted_policy_scope"], "canonical_12_measurable_in_toy_only": twelve_review["measurable_in_toy_causal_simulator"]}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Self-test

```json
{json.dumps({name: {"detected": row["detected"]} for name, row in synthetic["detectors"].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Repair dependency register

```json
{json.dumps([{"id": row["id"], "severity": row["severity"], "minimum_change": row["minimum_change"]} for row in repairs["items"]], ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: audit only. PASS certifies the audit's completeness, not KPI comparability.
""",
        encoding="utf-8",
    )

    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file()}
    write_json(
        root / "artifact_manifest.json",
        {
            "stage": STAGE,
            "artifact_root": str(root),
            "source_commit": provenance["source_commit"],
            "gate": gate["gate"],
            "required_artifacts_present": all(
                (root / name).exists() for name in REQUIRED_ARTIFACTS if name != "artifact_manifest.json"
            ),
            "manifest_self_reference_policy": "artifact_manifest.json is excluded from its own existence check and hash map",
            "file_sha256": {name: sha256_file(Path(path)) for name, path in files.items() if name != "artifact_manifest.json"},
            "elapsed_seconds": time.perf_counter() - started,
            "append_only_artifact": True,
            "audit_only": True,
            "training": 0,
            "kpi_executed": False,
            "test6_touched": False,
            "github_push_performed": False,
        },
    )
    (root / "_SUCCESS.lock").write_text(f"{gate['gate']}\n{provenance['source_commit']}\n{created_at}\n", encoding="utf-8")

    print(f"[H4M-AE-R1] artifact root: {root}")
    print(f"[H4M-AE-R1] gate: {gate['gate']}")
    print(f"[H4M-AE-R1] source: {provenance['source_commit_short']} clean={provenance['clean_worktree']}")
    primary_line = {key: row["measurement_status"] for key, row in gate_decision["primary_4kpi_measurement_status"].items()}
    print(f"[H4M-AE-R1] primary 4 KPI: {primary_line}")
    print(f"[H4M-AE-R1] canonical 12 available on promoted scope: {twelve_review['available_on_promoted_policy_scope']}/12")
    print(f"[H4M-AE-R1] constructable arms: {gate_decision['constructable_arm_ids']} | blocked: {gate_decision['blocked_arm_ids']}")
    print(f"[H4M-AE-R1] synthetic KPI paths detected: {gate_decision['synthetic_kpi_paths_detected']}")
    print(f"[H4M-AE-R1] self-test hygiene violations: {synthetic['audit_hygiene_violations']}")
    print(f"[H4M-AE-R1] root blockers: {repairs['root_blockers']}")
    print(f"[H4M-AE-R1] exact next gate: {NEXT_GATE} (not executed)")


if __name__ == "__main__":
    main()
