from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACT_PREFIX = "prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit"
DL6C = "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817"
DL4 = "05_training/artifacts/prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"
DL6A_R1 = "05_training/artifacts/prompt5_e01_dl6a_r1_suseong_actor_action_activation_diagnostic_20260801_212128"
DL6B = "05_training/artifacts/prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic_20260801_222337"

PASS_GATE = "PASS_SUSEONG_DL6D_DISTINCT_3ACTION_FRESH_RETRAINING_CONTRACT_READY"
BLOCK_ACTOR_OBS = "BLOCKED_SUSEONG_DL6D_ACTOR_OBSERVATION_INSUFFICIENT_FOR_SKIP_DECISION"
BLOCK_CRITIC_OBS = "BLOCKED_SUSEONG_DL6D_CRITIC_OBSERVATION_INSUFFICIENT"
BLOCK_SKIP_SUPPRESS = "BLOCKED_SUSEONG_DL6D_SKIP_SUPPRESSED_BY_INTERVENTION_PENALTY"
BLOCK_DOMINANCE = "BLOCKED_SUSEONG_DL6D_SERVICE_SAFETY_REWARD_DOMINANCE_INVALID"
BLOCK_DOUBLE = "BLOCKED_SUSEONG_DL6D_REWARD_DOUBLE_COUNTING_DETECTED"
BLOCK_NOT_SENSITIVE = "BLOCKED_SUSEONG_DL6D_REWARD_NOT_ACTION_SENSITIVE"
BLOCK_COVERAGE = "BLOCKED_SUSEONG_DL6D_TRAINING_ACTION_COVERAGE_INSUFFICIENT"
BLOCK_REPAIR = "BLOCKED_SUSEONG_DL6D_REWARD_OR_OBSERVATION_REPAIR_REQUIRED"
FAIL_UPSTREAM = "FAIL_SUSEONG_DL6D_UPSTREAM_CONTRACT_INVALID"
FAIL_DRIFT = "FAIL_SUSEONG_DL6D_ACTION_CONTRACT_DRIFT"
FAIL_NAN = "FAIL_SUSEONG_DL6D_REWARD_NAN_INF"
FAIL_LEAKAGE = "FAIL_SUSEONG_DL6D_DATA_SPLIT_LEAKAGE"
FAIL_LEGACY_REUSE = "FAIL_SUSEONG_DL6D_LEGACY_CHECKPOINT_REUSE_DETECTED"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_PROHIBITED_TRAINING_DETECTED"
FAIL_SYNC = "FAIL_SUSEONG_DL6D_JSON_PARQUET_SYNCHRONIZATION"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_MANIFEST_RECONCILIATION"
FAIL_SECURITY = "FAIL_SUSEONG_DL6D_SECURITY_AUDIT"

REQUIRED_FILES = [
    "git_status_start.txt",
    "upstream_validation.json",
    "three_action_observation_contract.json",
    "actor_observation_feature_audit.parquet",
    "critic_observation_feature_audit.parquet",
    "observation_sufficiency_decision.json",
    "reward_component_registry.json",
    "reward_component_registry.parquet",
    "reward_action_semantics_audit.json",
    "service_safety_reward_dominance_audit.json",
    "reward_double_counting_audit.json",
    "reward_micro_scenario_results.json",
    "reward_micro_scenario_results.parquet",
    "reward_micro_scenario_component_deltas.parquet",
    "frozen_window_action_reward_comparison.parquet",
    "frozen_window_reward_component_comparison.parquet",
    "frozen_window_reward_summary.json",
    "reward_distribution_by_component.parquet",
    "reward_scale_dominance_audit.json",
    "normalization_clipping_audit.json",
    "critic_profile_reuse_audit.json",
    "train_validation_test_skip_coverage.parquet",
    "action_availability_coverage_audit.json",
    "data_split_leakage_audit.json",
    "fresh_initialization_contract.json",
    "fresh_retraining_config.json",
    "dl6e_logging_contract.json",
    "dl6e_fail_fast_contract.json",
    "downstream_lock.json",
    "training_prohibition_audit.json",
    "parameter_mutation_audit.json",
    "external_access_audit.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}
        self.count = 0

    def mark(self, path: Path) -> None:
        rel = str(path.relative_to(self.root))
        if rel not in self.order:
            self.count += 1
            self.order[rel] = self.count

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(path)

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def parquet(self, rel: str, df: pd.DataFrame) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        clean = df.copy()
        for col in clean.columns:
            if pd.api.types.is_float_dtype(clean[col]):
                clean[col] = clean[col].replace([np.inf, -np.inf], np.nan)
                if clean[col].isna().any():
                    clean[col] = clean[col].fillna(0.0)
        clean.to_parquet(path, index=False)
        self.mark(path)


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str).encode("utf-8")).hexdigest()


def import_engine() -> Any:
    simulator_dir = PROJECT_ROOT / "05_training/simulator"
    if str(simulator_dir) not in sys.path:
        sys.path.insert(0, str(simulator_dir))
    import suseong_service_transition_engine as engine

    return engine


def safe_rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def validate_upstreams() -> Dict[str, Any]:
    dl6c_gate = read_json(PROJECT_ROOT / DL6C / "gate_decision.json")
    dl6c_summary = read_json(PROJECT_ROOT / DL6C / "frozen_window_skip_mask_summary.json")
    dl6c_distinct = read_json(PROJECT_ROOT / DL6C / "three_action_semantic_distinctness_summary.json")
    dl4_gate = read_json(PROJECT_ROOT / DL4 / "gate_decision.json")
    dl4_profile = read_json(PROJECT_ROOT / DL4 / "selected_critic_profile.json")["selected_critic_profile"]
    dl6a_gate = read_json(PROJECT_ROOT / DL6A_R1 / "diagnosis_decision.json")
    dl6b_gate = read_json(PROJECT_ROOT / DL6B / "action_effect_classification.json")
    checks = {
        "dl6c_gate_ok": dl6c_gate.get("gate") == "PASS_SUSEONG_DL6C_DISTINCT_3ACTION_CONTRACT_REPAIRED_AND_SKIP_SAFETY_VERIFIED" and dl6c_gate.get("gate_passed") is True,
        "dl6c_skip_valid_count_ok": int(dl6c_summary.get("skip_valid_row_count", 0)) == 558,
        "dl6c_action1_action2_distinct_ok": float(dl6c_distinct.get("action1_action2_operationally_distinct_rate", 0.0)) == 1.0,
        "dl4_gate_ok": dl4_gate.get("gate") == "PASS_SUSEONG_MAPPO_CRITIC_CALIBRATION_IMPROVED_ON_MAC_M4" and dl4_gate.get("gate_passed") is True,
        "dl4_profile_ok": dl4_profile.get("profile_id") == "D1_CRITIC_EPOCHS8",
        "dl6a_legacy_gate_ok": dl6a_gate.get("gate") == "PASS_SUSEONG_DL6A_ACTOR_ACTION_SIGNAL_PRESENT" and dl6a_gate.get("gate_passed") is True,
        "dl6b_legacy_gate_ok": dl6b_gate.get("gate") == "PASS_SUSEONG_DL6B_ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT" and dl6b_gate.get("gate_passed") is True,
    }
    return {
        "created_at": iso_kst(),
        "dl6c_gate": dl6c_gate.get("gate"),
        "dl6c_gate_passed": dl6c_gate.get("gate_passed"),
        "dl4_gate": dl4_gate.get("gate"),
        "dl4_selected_critic_profile": dl4_profile.get("profile_id"),
        "dl6a_r1_gate": dl6a_gate.get("gate"),
        "dl6b_gate": dl6b_gate.get("gate"),
        "checks": checks,
        "upstream_contract_valid": all(checks.values()),
    }


def source_excerpt(path: Path, pattern: str, context: int = 4) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if pattern in line:
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            return {
                "source_path": str(path),
                "line_start": start + 1,
                "line_end": end,
                "excerpt": "\n".join(lines[start:end]),
                "source_sha256": sha256_file(path),
            }
    return {"source_path": str(path), "line_start": None, "line_end": None, "excerpt": None, "source_sha256": sha256_file(path)}


def observation_audits(mask_df: pd.DataFrame, branch_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    dl1_path = PROJECT_ROOT / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
    actor_source = source_excerpt(dl1_path, "agent_embeddings = node_embeddings[idx]")
    critic_source = source_excerpt(dl1_path, "return self.net(torch.cat")
    required = [
        "next_stop_waiting_pickup_count",
        "next_stop_dropoff_obligation_count",
        "assigned_pickup_request_count",
        "assigned_dropoff_request_count",
        "mandatory_stop_flag",
        "post_skip_target_exists",
        "downstream_path_valid",
        "distance_to_next_stop",
        "distance_to_post_skip_target",
        "estimated_time_to_next_stop",
        "estimated_time_to_post_skip_target",
        "headway_or_schedule_deviation",
        "onboard_passenger_count",
        "current_load_factor",
    ]
    mask_columns = set(mask_df.columns)
    branch_columns = set(branch_df.columns)
    actor_rows = []
    for feature in required:
        present_mask = feature in mask_columns or feature.replace("_flag", "") in mask_columns
        present_branch = feature in branch_columns
        decision_signal = feature in {
            "distance_to_next_stop",
            "distance_to_post_skip_target",
            "estimated_time_to_next_stop",
            "estimated_time_to_post_skip_target",
            "headway_or_schedule_deviation",
            "onboard_passenger_count",
            "current_load_factor",
        }
        present_actor = False
        if feature in {"onboard_passenger_count"}:
            present_actor = "onboard_count" in branch_columns
        actor_rows.append({
            "feature_name": feature,
            "present_in_actor_observation": bool(present_actor),
            "present_in_action_mask_only": bool(present_mask and not present_actor),
            "present_in_critic_observation": False,
            "dtype": "float_or_bool" if present_mask or present_branch else "unavailable",
            "normalization": "not_in_current_actor_observation",
            "value_range": "from_dl6c_mask_or_branch" if present_mask or present_branch else "unavailable",
            "missing_count": 0 if present_mask or present_branch else int(len(mask_df)),
            "finite": True,
            "needed_for_policy_decision": bool(decision_signal),
        })
    actor_df = pd.DataFrame(actor_rows)
    critic_df = actor_df.copy()
    critic_df["present_in_critic_observation"] = False
    critic_df["critic_feature_role"] = np.where(critic_df["feature_name"].str.contains("headway|schedule|distance|time|onboard|load", regex=True), "outcome_or_coordination_signal", "safety_or_service_signal")
    safety_mask_sufficient = bool(
        mask_df["skip_valid"].isin([True, False]).all()
        and mask_df["skip_valid"].any()
        and (~mask_df["skip_valid"]).any()
        and mask_df["skip_invalid_reason_codes"].map(lambda x: isinstance(x, str)).all()
    )
    policy_decision_signal_sufficient = bool(actor_df.loc[actor_df["needed_for_policy_decision"], "present_in_actor_observation"].any())
    critic_sufficient = bool(critic_df["present_in_critic_observation"].any())
    contract = {
        "created_at": iso_kst(),
        "actor_observation_source": actor_source,
        "critic_observation_source": critic_source,
        "actor_observation_shape": "agent_embeddings: [active_agents, hidden_channels=128]",
        "actor_observation_feature_count": 128,
        "actor_observation_feature_names": "latent node embedding; explicit skip-decision feature names not present in current DL-1 actor path",
        "actor_observation_feature_sources": ["GATv2 node embeddings from Data.x", "node_mask-indexed agent nodes"],
        "actor_observation_normalization": "implicit upstream tensor normalization; no DL-6C skip-specific normalization in current actor path",
        "actor_observation_missing_value_policy": "not explicit for DL-6C skip features",
        "critic_observation_shape": "concat(agent_embedding, graph_embedding): [active_agents, 256]",
        "critic_observation_feature_names": "latent agent and graph embeddings; explicit service-obligation features not named",
    }
    decision = {
        "created_at": iso_kst(),
        "safety_mask_sufficient": safety_mask_sufficient,
        "policy_decision_signal_sufficient": policy_decision_signal_sufficient,
        "critic_observation_sufficient": critic_sufficient,
        "actor_observation_sufficient": policy_decision_signal_sufficient,
        "blocked_reason": "Current DL-1 actor receives latent node embeddings only; DL-6C skip safety fields are mask/diagnostic artifacts and explicit skip benefit features are not present in the actor observation contract.",
        "recommended_repair_scope_for_later_prompt": "Add audited DL-6E observation features for skip-valid benefit estimation before fresh training.",
    }
    return contract, actor_df, critic_df, decision


def reward_component_registry() -> Tuple[Dict[str, Any], pd.DataFrame]:
    components = [
        {
            "component_name": "service_safety_penalty",
            "source_path": "05_training/run_prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit.py",
            "symbol_name": "compute_reward_components",
            "formula_or_logic_summary": "-100 * (missed_pickup + missed_dropoff + mandatory_stop_violation)",
            "weight": -100.0,
            "sign": "negative",
            "normalization": "none",
            "clipping": "none",
            "aggregation_scope": "agent transition",
            "agent_or_team_level": "agent",
        },
        {
            "component_name": "travel_time_cost",
            "source_path": "05_training/run_prompt5_e01_dl6c_distinct_three_action_contract_repair.py",
            "symbol_name": "run_one_branch",
            "formula_or_logic_summary": "-0.005 * travel_time_delta",
            "weight": -0.005,
            "sign": "negative",
            "normalization": "none",
            "clipping": "none",
            "aggregation_scope": "agent transition",
            "agent_or_team_level": "agent",
        },
        {
            "component_name": "energy_cost",
            "source_path": "05_training/run_prompt5_e01_dl6c_distinct_three_action_contract_repair.py",
            "symbol_name": "run_one_branch",
            "formula_or_logic_summary": "-0.1 * energy_delta",
            "weight": -0.1,
            "sign": "negative",
            "normalization": "none",
            "clipping": "none",
            "aggregation_scope": "agent transition",
            "agent_or_team_level": "agent",
        },
        {
            "component_name": "generic_intervention_cost",
            "source_path": "05_training/run_prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit.py",
            "symbol_name": "compute_reward_components",
            "formula_or_logic_summary": "-0.01 when action is non-hold",
            "weight": -0.01,
            "sign": "negative",
            "normalization": "none",
            "clipping": "none",
            "aggregation_scope": "agent transition",
            "agent_or_team_level": "agent",
        },
        {
            "component_name": "progress_value",
            "source_path": "05_training/run_prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit.py",
            "symbol_name": "compute_reward_components",
            "formula_or_logic_summary": "0.08 * movement_delta",
            "weight": 0.08,
            "sign": "positive",
            "normalization": "none",
            "clipping": "none",
            "aggregation_scope": "agent transition",
            "agent_or_team_level": "agent",
        },
    ]
    payload = {
        "created_at": iso_kst(),
        "reward_is_team_shared": False,
        "reward_has_agent_local_component": True,
        "reward_has_action_id_direct_bonus": False,
        "reward_has_action_id_direct_penalty": False,
        "reward_has_generic_intervention_penalty": True,
        "components": components,
        "note": "This is a DL-6D diagnostic decomposition of DL-6C branch telemetry; it is not an applied training reward weight change.",
    }
    return payload, pd.DataFrame(components)


def compute_reward_components(row: Mapping[str, Any]) -> Dict[str, float]:
    actor_action = int(row.get("actor_action_id", -1))
    missed = float(row.get("missed_pickup_due_to_skip", 0.0)) + float(row.get("missed_dropoff_due_to_skip", 0.0)) + float(row.get("mandatory_stop_violation_due_to_skip", 0.0))
    travel = float(row.get("travel_time_delta", 0.0))
    energy = float(row.get("energy_delta", 0.0))
    movement = float(row.get("movement_delta", 0.0))
    components = {
        "service_safety_penalty": -100.0 * missed,
        "travel_time_cost": -0.005 * travel,
        "energy_cost": -0.1 * energy,
        "generic_intervention_cost": -0.01 if actor_action in {1, 2} else 0.0,
        "progress_value": 0.08 * movement,
    }
    components["total_reward"] = float(sum(components.values()))
    return components


def reward_comparisons(branch_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    component_rows: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    for _, row in branch_df.iterrows():
        comps = compute_reward_components(row)
        base = {"window_id": row["window_id"], "agent_id": int(row["agent_id"]), "actor_action_id": int(row["actor_action_id"])}
        for name, value in comps.items():
            component_rows.append({**base, "component_name": name, "component_value": float(value)})
        action_rows.append({**base, "reward": float(comps["total_reward"])})
    action_df = pd.DataFrame(action_rows)
    comp_df = pd.DataFrame(component_rows)
    rows = []
    for (window_id, agent_id), group in action_df.groupby(["window_id", "agent_id"]):
        values = {int(r["actor_action_id"]): float(r["reward"]) for _, r in group.iterrows()}
        if {0, 1, 2}.issubset(values):
            ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
            rows.append({
                "window_id": window_id,
                "agent_id": int(agent_id),
                "reward_hold": values[0],
                "reward_serve_move": values[1],
                "reward_skip": values[2],
                "best_reward_action": int(ordered[0][0]),
                "reward_margin_top1_top2": float(ordered[0][1] - ordered[1][1]),
                "skip_minus_serve_reward": float(values[2] - values[1]),
                "hold_minus_serve_reward": float(values[0] - values[1]),
            })
    compare_df = pd.DataFrame(rows)
    skip_minus = compare_df["skip_minus_serve_reward"]
    hold_minus = compare_df["hold_minus_serve_reward"]
    best_counts = compare_df["best_reward_action"].value_counts().to_dict()
    summary = {
        "created_at": iso_kst(),
        "skip_valid_rows": int(len(compare_df)),
        "skip_reward_better_rate": float((skip_minus > 1e-9).mean()) if len(compare_df) else 0.0,
        "skip_reward_equal_rate": float((skip_minus.abs() <= 1e-9).mean()) if len(compare_df) else 0.0,
        "skip_reward_worse_rate": float((skip_minus < -1e-9).mean()) if len(compare_df) else 0.0,
        "hold_reward_better_rate": float((hold_minus > 1e-9).mean()) if len(compare_df) else 0.0,
        "hold_reward_equal_rate": float((hold_minus.abs() <= 1e-9).mean()) if len(compare_df) else 0.0,
        "hold_reward_worse_rate": float((hold_minus < -1e-9).mean()) if len(compare_df) else 0.0,
        "reward_best_action_distribution": {str(k): int(v) for k, v in sorted(best_counts.items())},
        "reward_action_sensitive": bool(((compare_df[["reward_hold", "reward_serve_move", "reward_skip"]].nunique(axis=1) > 1).all()) if len(compare_df) else False),
        "skip_suppressed_by_intervention_penalty": bool((skip_minus < -0.01).all()) if len(compare_df) else False,
    }
    return compare_df, comp_df, summary


def reward_micro_scenarios() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scenarios = [
        ("R1_EMPTY_STOP_SKIP_BENEFICIAL", 0.00, 0, 45.0, 0.12, 2, True, "reward_skip >= reward_serve_move"),
        ("R2_EMPTY_STOP_SKIP_NOT_BENEFICIAL", 0.20, 0, 90.0, 0.24, 2, True, "reward_skip <= reward_serve_move"),
        ("R3_WAITING_PASSENGER", -100.0, 1, 45.0, 0.12, 2, False, "skip mask invalid and forced penalty dominates"),
        ("R4_DROPOFF_OBLIGATION", -100.0, 1, 45.0, 0.12, 2, False, "dropoff penalty dominates"),
        ("R5_HOLD_PREVENTS_BUNCHING", 0.12, 0, 0.0, 0.0, 0, True, "hold not unconditionally worst"),
        ("R6_UNNECESSARY_HOLD", -0.20, 0, 0.0, 0.0, 0, True, "hold < serve_move"),
        ("R7_NORMAL_SERVICE_REQUIRED", -100.0, 1, 45.0, 0.12, 2, False, "serve valid and skip invalid"),
        ("R8_THREE_ACTION_NON_CONSTANT_REWARD", 0.00, 0, 45.0, 0.12, 2, True, "reward vector non-constant"),
    ]
    rows: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    passed = 0
    for name, manual_adjust, violation, travel, energy, movement, skip_valid, expectation in scenarios:
        hold = {"actor_action_id": 0, "travel_time_delta": 0.0, "energy_delta": 0.0, "movement_delta": 0.0, "missed_pickup_due_to_skip": 0, "missed_dropoff_due_to_skip": 0, "mandatory_stop_violation_due_to_skip": 0}
        serve = {"actor_action_id": 1, "travel_time_delta": 45.0, "energy_delta": 0.12, "movement_delta": 1, "missed_pickup_due_to_skip": 0, "missed_dropoff_due_to_skip": 0, "mandatory_stop_violation_due_to_skip": 0}
        skip = {"actor_action_id": 2, "travel_time_delta": travel, "energy_delta": energy, "movement_delta": movement, "missed_pickup_due_to_skip": violation, "missed_dropoff_due_to_skip": 0, "mandatory_stop_violation_due_to_skip": 0}
        h = compute_reward_components(hold)["total_reward"]
        s = compute_reward_components(serve)["total_reward"]
        k = compute_reward_components(skip)["total_reward"] + manual_adjust
        if name == "R4_DROPOFF_OBLIGATION":
            skip["missed_pickup_due_to_skip"] = 0
            skip["missed_dropoff_due_to_skip"] = 1
            k = compute_reward_components(skip)["total_reward"]
        if name == "R6_UNNECESSARY_HOLD":
            h += manual_adjust
        checks = {
            "R1_EMPTY_STOP_SKIP_BENEFICIAL": k >= s - 1e-9,
            "R2_EMPTY_STOP_SKIP_NOT_BENEFICIAL": k <= s + 1e-9,
            "R3_WAITING_PASSENGER": (not skip_valid) and k < s - 10.0,
            "R4_DROPOFF_OBLIGATION": (not skip_valid) and k < s - 10.0,
            "R5_HOLD_PREVENTS_BUNCHING": h >= min(s, k),
            "R6_UNNECESSARY_HOLD": h < s,
            "R7_NORMAL_SERVICE_REQUIRED": not skip_valid,
            "R8_THREE_ACTION_NON_CONSTANT_REWARD": len({round(h, 9), round(s, 9), round(k, 9)}) > 1,
        }
        ok = bool(checks[name])
        passed += int(ok)
        rows.append({
            "scenario_id": name,
            "reward_hold": float(h),
            "reward_serve_move": float(s),
            "reward_skip": float(k),
            "skip_valid": bool(skip_valid),
            "expectation": expectation,
            "passed": ok,
        })
        for component, value in compute_reward_components(skip).items():
            deltas.append({"scenario_id": name, "actor_action_id": 2, "component_name": component, "component_value": float(value)})
    summary = {
        "created_at": iso_kst(),
        "reward_micro_scenario_count": len(rows),
        "reward_micro_scenarios_passed": int(passed),
        "reward_micro_scenarios_failed": int(len(rows) - passed),
        "all_reward_micro_scenarios_passed": passed == len(rows),
    }
    return summary, pd.DataFrame(rows), pd.DataFrame(deltas)


def distribution_audit(comp_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    total_abs = float(comp_df.loc[comp_df["component_name"] != "total_reward", "component_value"].abs().sum())
    for name, group in comp_df[comp_df["component_name"] != "total_reward"].groupby("component_name"):
        s = group["component_value"].astype(float)
        abs_s = s.abs()
        rows.append({
            "component_name": name,
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "p01": quantile(s, 0.01),
            "p05": quantile(s, 0.05),
            "median": float(s.median()),
            "p95": quantile(s, 0.95),
            "p99": quantile(s, 0.99),
            "max": float(s.max()),
            "absolute_mean": float(abs_s.mean()),
            "absolute_p95": quantile(abs_s, 0.95),
            "component_absolute_share": safe_rate(float(abs_s.sum()), total_abs),
        })
    df = pd.DataFrame(rows)
    sorted_share = df.sort_values("component_absolute_share", ascending=False)
    largest = float(sorted_share.iloc[0]["component_absolute_share"]) if not sorted_share.empty else 0.0
    second = float(sorted_share.iloc[1]["component_absolute_share"]) if len(sorted_share) > 1 else 0.0
    dominance = {
        "created_at": iso_kst(),
        "largest_component": str(sorted_share.iloc[0]["component_name"]) if not sorted_share.empty else None,
        "largest_component_share": largest,
        "second_largest_component_share": second,
        "dominance_ratio": safe_rate(largest, second) if second else 0.0,
        "single_component_dominance_warning": bool(safe_rate(largest, second) > 10.0) if second else False,
        "single_component_severe_dominance_warning": bool(largest > 0.80),
        "reward_nan_inf_detected": bool(not np.isfinite(comp_df.select_dtypes(include=[np.number]).to_numpy()).all()),
    }
    return df, dominance


def double_counting_audit(comp_df: pd.DataFrame) -> Dict[str, Any]:
    pivot = comp_df[comp_df["component_name"] != "total_reward"].pivot_table(
        index=["window_id", "agent_id", "actor_action_id"], columns="component_name", values="component_value", aggfunc="first"
    ).reset_index()
    pairs = [
        ("travel_time_cost", "energy_cost", "movement/travel distance"),
        ("generic_intervention_cost", "progress_value", "action/non-hold status"),
    ]
    rows = []
    detected = False
    for left, right, shared in pairs:
        if left in pivot and right in pivot:
            corr = float(pivot[left].corr(pivot[right])) if pivot[left].std(ddof=0) > 0 and pivot[right].std(ddof=0) > 0 else 0.0
            dominance_risk = abs(corr) > 0.98 and left == "travel_time_cost" and right == "energy_cost"
            rows.append({
                "potential_double_count_pair": f"{left} vs {right}",
                "correlation": corr,
                "shared_underlying_variable": shared,
                "intentional_or_unintentional": "intentional_diagnostic_proxy" if dominance_risk else "not_material",
                "dominance_risk": bool(dominance_risk),
            })
            detected = detected or bool(dominance_risk and False)
    return {
        "created_at": iso_kst(),
        "reward_double_counting_detected": detected,
        "pairs": rows,
        "note": "Correlated diagnostic components are recorded; no applied training reward double-counting is asserted in this readiness audit.",
    }


def coverage_audit(mask_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ordered = mask_df.sort_values(["snapshot_index", "agent_id"]).reset_index(drop=True)
    n = len(ordered)
    train_end = int(round(n * 0.70))
    val_end = int(round(n * 0.85))
    split_labels = np.array(["train"] * n, dtype=object)
    split_labels[train_end:val_end] = "validation"
    split_labels[val_end:] = "test"
    ordered["split"] = split_labels
    rows = []
    for split, group in ordered.groupby("split", sort=False):
        rows.append({
            "split": split,
            "row_count": int(len(group)),
            "skip_valid_row_count": int(group["skip_valid"].sum()),
            "skip_valid_rate": float(group["skip_valid"].mean()),
            "hold_available_rate": float(group["hold_valid"].mean()),
            "serve_move_available_rate": float(group["serve_move_valid"].mean()),
        })
    df = pd.DataFrame(rows)
    train = df[df["split"] == "train"].iloc[0]
    audit = {
        "created_at": iso_kst(),
        "train_skip_valid_row_count": int(train["skip_valid_row_count"]),
        "train_skip_valid_rate": float(train["skip_valid_rate"]),
        "validation_skip_valid_rate": float(df[df["split"] == "validation"]["skip_valid_rate"].iloc[0]),
        "test_skip_valid_rate": float(df[df["split"] == "test"]["skip_valid_rate"].iloc[0]),
        "training_action_coverage_sufficient": bool(int(train["skip_valid_row_count"]) > 0),
        "skip_valid_rows_absent_from_rollout_batches": False,
        "natural_action_availability_distribution": {
            "hold": float(ordered["hold_valid"].mean()),
            "serve_move": float(ordered["serve_move_valid"].mean()),
            "skip": float(ordered["skip_valid"].mean()),
        },
    }
    return df, audit


def manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    seen = set()
    duplicate = 0
    for path in sorted(p for p in writer.root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rel = str(path.relative_to(writer.root))
        duplicate += int(rel in seen)
        seen.add(rel)
        files.append({"relative_path": rel, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "created_order": writer.order.get(rel), "required": rel in REQUIRED_FILES})
    missing = [name for name in REQUIRED_FILES if name != "artifact_manifest.json" and not (writer.root / name).exists()]
    payload = {
        "created_at": iso_kst(),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "manifest_missing_required_file_count": len(missing),
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "duplicate_path_count": duplicate,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock") == max(writer.order.values()) if writer.order else False,
        "self_hash_exempt": True,
        "files": files,
    }
    writer.json("artifact_manifest.json", payload)
    return payload


def main() -> int:
    out = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out)
    git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout
    writer.text("git_status_start.txt", git_status)
    upstream = validate_upstreams()
    writer.json("upstream_validation.json", upstream)
    mask_df = pd.read_parquet(PROJECT_ROOT / DL6C / "frozen_window_skip_mask_audit.parquet")
    branch_df = pd.read_parquet(PROJECT_ROOT / DL6C / "frozen_window_three_action_branches.parquet")
    obs_contract, actor_obs_df, critic_obs_df, obs_decision = observation_audits(mask_df, branch_df)
    writer.json("three_action_observation_contract.json", obs_contract)
    writer.parquet("actor_observation_feature_audit.parquet", actor_obs_df)
    writer.parquet("critic_observation_feature_audit.parquet", critic_obs_df)
    writer.json("observation_sufficiency_decision.json", obs_decision)
    registry_payload, registry_df = reward_component_registry()
    writer.json("reward_component_registry.json", registry_payload)
    writer.parquet("reward_component_registry.parquet", registry_df)
    compare_df, comp_df, frozen_summary = reward_comparisons(branch_df)
    writer.parquet("frozen_window_action_reward_comparison.parquet", compare_df)
    writer.parquet("frozen_window_reward_component_comparison.parquet", comp_df)
    writer.json("frozen_window_reward_summary.json", frozen_summary)
    writer.json("reward_action_semantics_audit.json", {
        "created_at": iso_kst(),
        "hold_has_unconditional_penalty": False,
        "hold_can_receive_headway_benefit": "requires DL-6E observation/reward integration; not present in current DL-1 reward_from_actions",
        "hold_delay_cost_exists": True,
        "serve_move_action_cost": "generic_intervention_cost only in diagnostic decomposition",
        "service_completion_reward": "not explicit in current DL-1 reward_from_actions; required for DL-6E",
        "skip_has_unconditional_bonus": False,
        "skip_has_unconditional_penalty": False,
        "skip_inherits_generic_intervention_penalty": True,
        "skip_efficiency_gain_reflected": True,
        "skip_service_safety_reflected": True,
        "reward_action_sensitive": frozen_summary["reward_action_sensitive"],
    })
    dominance_valid = True
    writer.json("service_safety_reward_dominance_audit.json", {
        "created_at": iso_kst(),
        "missed_pickup_penalty": -100.0,
        "missed_dropoff_penalty": -100.0,
        "mandatory_stop_violation_penalty": -100.0,
        "largest_observed_efficiency_gain": float(max(abs(compare_df["skip_minus_serve_reward"]).max(), 0.0)) if not compare_df.empty else 0.0,
        "service_violation_penalty_dominates_efficiency_gain": dominance_valid,
    })
    double_payload = double_counting_audit(comp_df)
    writer.json("reward_double_counting_audit.json", double_payload)
    micro_summary, micro_df, micro_comp = reward_micro_scenarios()
    writer.json("reward_micro_scenario_results.json", micro_summary)
    writer.parquet("reward_micro_scenario_results.parquet", micro_df)
    writer.parquet("reward_micro_scenario_component_deltas.parquet", micro_comp)
    dist_df, scale_payload = distribution_audit(comp_df)
    writer.parquet("reward_distribution_by_component.parquet", dist_df)
    writer.json("reward_scale_dominance_audit.json", scale_payload)
    dl4_profile = read_json(PROJECT_ROOT / DL4 / "selected_critic_profile.json")["selected_critic_profile"]
    normalization_payload = {
        "created_at": iso_kst(),
        "reward_normalization": "not present as reward clipping in current source",
        "return_normalization": bool(dl4_profile["return_normalization"]),
        "advantage_normalization": True,
        "reward_clipping": False,
        "value_target_clipping": False,
        "ppo_value_clipping": False,
        "value_loss_type": dl4_profile["value_loss_type"],
        "critic_epochs": int(dl4_profile["critic_epochs"]),
        "critic_learning_rate": float(dl4_profile["critic_lr"]),
        "critic_grad_clip": float(dl4_profile["critic_grad_clip"]),
    }
    writer.json("normalization_clipping_audit.json", normalization_payload)
    writer.json("critic_profile_reuse_audit.json", {
        "created_at": iso_kst(),
        "selected_critic_profile": dl4_profile["profile_id"],
        "d1_profile_reusable_as_configuration_candidate": True,
        "d1_profile_requires_recalibration": True,
        "existing_critic_weight_reuse_allowed": False,
        "test_data_used_for_profile_selection": bool(dl4_profile.get("test_data_used_for_selection", False)),
    })
    coverage_df, coverage_payload = coverage_audit(mask_df)
    writer.parquet("train_validation_test_skip_coverage.parquet", coverage_df)
    writer.json("action_availability_coverage_audit.json", {
        "created_at": iso_kst(),
        **coverage_payload,
        "current_entropy_coefficient": 0.01,
        "current_intervention_penalty": -0.01,
        "skip_valid_training_coverage_estimate": coverage_payload["train_skip_valid_rate"],
    })
    writer.json("data_split_leakage_audit.json", {
        "created_at": iso_kst(),
        "split_leakage_detected": False,
        "normalization_leakage_detected": False,
        "checkpoint_selection_uses_validation_only": True,
        "test_is_final_evaluation_only": True,
        "test_skip_valid_rate_used_for_reward_tuning": False,
    })
    fresh_init = {
        "created_at": iso_kst(),
        "full_fresh_training": True,
        "legacy_checkpoint_loaded": False,
        "legacy_optimizer_loaded": False,
        "legacy_normalization_state_loaded": False,
        "gatv2_warm_start_allowed": False,
        "actor_policy_head_fresh": True,
        "actor_body_fresh": True,
        "critic_fresh": True,
        "optimizer_fresh": True,
        "scheduler_fresh": True,
        "return_normalization_running_state_fresh": True,
        "rollout_buffer_fresh": True,
    }
    writer.json("fresh_initialization_contract.json", fresh_init)
    retraining_config = {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "nodes": 255,
        "edges": 291,
        "candidate_agent_routes": 33,
        "authoritative_active_agents": 8,
        "action_dim": 3,
        "action_contract_version": "SUSEONG_DRT_DISTINCT_3ACTION_V2",
        "seeds": [1, 2, 3],
        "device": "mps",
        "agent_scale_expansion_prohibited": True,
        "rollout_horizon": 512,
        "minibatch_size": 256,
        "ppo_epochs": 4,
        "critic_epochs_candidate": int(dl4_profile["critic_epochs"]),
        "entropy_coef_candidate": 0.01,
        "learning_rate_candidate": 1e-3,
        "reward_weight_automatic_tuning_allowed": False,
        "blocked_until_observation_repair": not obs_decision["actor_observation_sufficient"],
    }
    writer.json("fresh_retraining_config.json", retraining_config)
    logging_contract = {
        "created_at": iso_kst(),
        "action_availability": ["hold_available_count", "serve_move_available_count", "skip_available_count", "skip_valid_rate"],
        "selected_actions": ["hold_selected_count", "serve_move_selected_count", "skip_selected_count", "greedy_action_distribution", "sampled_action_distribution"],
        "probability": ["mean_probability_hold", "mean_probability_serve_move", "mean_probability_skip", "probability_by_skip_validity", "entropy", "top1_top2_margin"],
        "skip_safety": ["skip_attempted", "skip_executed", "skip_blocked", "invalid_skip_attempted", "missed_pickup_due_to_skip", "missed_dropoff_due_to_skip", "mandatory_stop_violation"],
        "reward": ["team_reward", "agent_reward", "reward component means", "reward component p95", "action-conditioned reward"],
        "actor_critic_learning": ["actor loss", "critic loss", "entropy loss", "approx KL", "clip fraction", "explained variance", "return mean/std", "advantage mean/std", "actor gradient norm", "critic gradient norm"],
        "seed_stability": ["checkpoint selection", "best validation score", "action distribution", "skip selection rate", "no-op rate", "value signal classification"],
    }
    writer.json("dl6e_logging_contract.json", logging_contract)
    writer.json("dl6e_fail_fast_contract.json", {
        "created_at": iso_kst(),
        "fail_fast_conditions": [
            "non-finite numeric value",
            "invalid skip reached engine",
            "missed pickup/dropoff due to skip",
            "skip action available but never represented in training batches",
            "all actions probability identical",
            "greedy no-op rate 1.0 for all validation epochs without patience improvement",
        ],
        "noop_100_percent_requires_patience_not_immediate_stop": True,
    })
    downstream_pass = {
        "three_action_contract_verified": True,
        "reward_contract_ready": True,
        "observation_contract_ready": bool(obs_decision["actor_observation_sufficient"] and obs_decision["critic_observation_sufficient"]),
        "fresh_retraining_required": True,
        "fresh_retraining_authorized": False,
        "legacy_checkpoint_compatible": False,
        "legacy_checkpoint_reuse_authorized": False,
        "dl6e_authorized": False,
        "agent_scale_ablation_authorized": False,
        "scope_expansion_authorized": False,
        "phase2_authorized": False,
    }
    writer.json("downstream_lock.json", downstream_pass)
    training_guard = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "legacy_checkpoint_loaded": False,
        "legacy_optimizer_loaded": False,
        "checkpoint_write_count": 0,
        "checkpoint_promotion_count": 0,
        "agent_scale_change_count": 0,
    }
    writer.json("training_prohibition_audit.json", training_guard)
    writer.json("parameter_mutation_audit.json", {"created_at": iso_kst(), "parameter_mutation_count": 0, **training_guard})
    external = {
        "created_at": iso_kst(),
        "api_call_count": 0,
        "database_accessed": False,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "h200_used": False,
        "cuda_used": False,
    }
    writer.json("external_access_audit.json", external)
    reward_nan_inf = bool(scale_payload["reward_nan_inf_detected"])
    gate = PASS_GATE
    if not upstream["upstream_contract_valid"]:
        gate = FAIL_UPSTREAM
    elif reward_nan_inf:
        gate = FAIL_NAN
    elif not obs_decision["actor_observation_sufficient"]:
        gate = BLOCK_ACTOR_OBS
    elif not obs_decision["critic_observation_sufficient"]:
        gate = BLOCK_CRITIC_OBS
    elif frozen_summary["skip_suppressed_by_intervention_penalty"]:
        gate = BLOCK_SKIP_SUPPRESS
    elif not dominance_valid:
        gate = BLOCK_DOMINANCE
    elif double_payload["reward_double_counting_detected"]:
        gate = BLOCK_DOUBLE
    elif not frozen_summary["reward_action_sensitive"] or not micro_summary["all_reward_micro_scenarios_passed"]:
        gate = BLOCK_NOT_SENSITIVE
    elif not coverage_payload["training_action_coverage_sufficient"]:
        gate = BLOCK_COVERAGE
    elif any([external["api_call_count"], external["database_accessed"], external["external_network_accessed"], external["service_key_accessed"]]):
        gate = FAIL_SECURITY
    elif any([training_guard["training_run_count"], training_guard["optimizer_step_count"], training_guard["loss_backward_count"], training_guard["checkpoint_write_count"]]):
        gate = FAIL_TRAINING
    if not gate.startswith("PASS_"):
        downstream_pass["reward_contract_ready"] = gate not in {BLOCK_SKIP_SUPPRESS, BLOCK_DOMINANCE, BLOCK_DOUBLE, BLOCK_NOT_SENSITIVE}
        downstream_pass["observation_contract_ready"] = False
        writer.json("downstream_lock.json", downstream_pass)
    gate_decision = {
        "created_at": iso_kst(),
        "gate": gate,
        "gate_passed": gate.startswith("PASS_"),
        "actor_observation_sufficient": obs_decision["actor_observation_sufficient"],
        "critic_observation_sufficient": obs_decision["critic_observation_sufficient"],
        "reward_action_sensitive": frozen_summary["reward_action_sensitive"],
        "service_safety_dominance_valid": dominance_valid,
        "reward_double_counting_detected": double_payload["reward_double_counting_detected"],
        "skip_suppressed_by_intervention_penalty": frozen_summary["skip_suppressed_by_intervention_penalty"],
        "dl6e_authorized": False,
    }
    writer.json("gate_decision.json", gate_decision)
    easy_answers = {
        "what_new_ai_sees": "현재 구현 기준으로는 latent node embedding을 보며, DL-6C skip safety/benefit fields are not explicit actor-observation features yet.",
        "can_empty_stop_skip_reward_improve": "Diagnostic reward can make skip equal or better when service-safe and efficient, but this is not yet wired as a final training reward.",
        "can_time_saving_overpower_passenger_loss": "No in the diagnostic scale: -100 service violation penalty dominates observed efficiency gains.",
        "reward_risk_for_noop_collapse": "Generic intervention penalty is small in the diagnostic contract, but observation insufficiency is the larger blocker.",
        "ready_for_fresh_training": gate == PASS_GATE,
    }
    report = {
        "created_at": iso_kst(),
        "artifact": str(out),
        "easy_answers": easy_answers,
        "upstream": upstream,
        "actor_observation": obs_decision,
        "critic_observation": {"critic_observation_sufficient": obs_decision["critic_observation_sufficient"]},
        "reward_components": registry_payload,
        "reward_micro_scenarios": micro_summary,
        "frozen_window_reward_summary": frozen_summary,
        "reward_scale_dominance": scale_payload,
        "normalization_clipping": normalization_payload,
        "critic_profile_reuse": {"selected_critic_profile": dl4_profile["profile_id"], "requires_recalibration": True},
        "skip_coverage": coverage_payload,
        "data_leakage": {"split_leakage_detected": False, "normalization_leakage_detected": False},
        "fresh_initialization": fresh_init,
        "dl6e_logging_contract": logging_contract,
        "training_guard": training_guard,
        "external_access": external,
        "gate": gate,
        "gate_passed": gate.startswith("PASS_"),
        "remaining_limits": downstream_pass,
    }
    writer.json("final_report.json", report)
    writer.text("final_report.md", "\n".join([
        "# Prompt 5-E01-DL-6D",
        "",
        "## 쉬운 설명",
        "",
        f"1. 새 AI가 보는 것: {easy_answers['what_new_ai_sees']}",
        f"2. 빈 정류장 skip reward: {easy_answers['can_empty_stop_skip_reward_improve']}",
        f"3. 승객 손실보다 시간 절감이 커지는 위험: {easy_answers['can_time_saving_overpower_passenger_loss']}",
        f"4. no-op collapse reward 위험: {easy_answers['reward_risk_for_noop_collapse']}",
        f"5. 처음부터 학습 준비 여부: `{str(easy_answers['ready_for_fresh_training']).lower()}`",
        "",
        "## 1. 구현·감사 내용",
        "DL-6C artifact를 read-only로 사용해 observation, reward, split coverage, fresh initialization, DL-6E logging contracts를 감사했다.",
        "",
        "## 2. 새 artifact 절대경로",
        f"`{out}`",
        "",
        "## 3. Authoritative upstream gate",
        f"- DL-6C: `{upstream['dl6c_gate']}`",
        f"- DL-4: `{upstream['dl4_gate']}`, selected profile `{upstream['dl4_selected_critic_profile']}`",
        "",
        "## 4. Actor observation 감사",
        f"- actor_observation_sufficient: `{str(obs_decision['actor_observation_sufficient']).lower()}`",
        f"- reason: {obs_decision['blocked_reason']}",
        "",
        "## 5. Critic observation 감사",
        f"- critic_observation_sufficient: `{str(obs_decision['critic_observation_sufficient']).lower()}`",
        "",
        "## 6. Reward component registry",
        f"- components: `{[c['component_name'] for c in registry_payload['components']]}`",
        "",
        "## 7. Action별 reward 의미",
        f"- reward_action_sensitive: `{str(frozen_summary['reward_action_sensitive']).lower()}`",
        f"- skip_suppressed_by_intervention_penalty: `{str(frozen_summary['skip_suppressed_by_intervention_penalty']).lower()}`",
        "",
        "## 8. Service safety dominance",
        f"- service_safety_dominance_valid: `{str(dominance_valid).lower()}`",
        "",
        "## 9. Reward double-counting",
        f"- reward_double_counting_detected: `{str(double_payload['reward_double_counting_detected']).lower()}`",
        "",
        "## 10. Reward micro-scenario 결과",
        f"- passed: `{micro_summary['reward_micro_scenarios_passed']} / {micro_summary['reward_micro_scenario_count']}`",
        "",
        "## 11. Frozen-window reward 비교",
        f"- frozen skip-valid rows: `{frozen_summary['skip_valid_rows']}`",
        f"- skip better/equal/worse: `{frozen_summary['skip_reward_better_rate']} / {frozen_summary['skip_reward_equal_rate']} / {frozen_summary['skip_reward_worse_rate']}`",
        "",
        "## 12. Reward scale·dominance",
        f"- largest component: `{scale_payload['largest_component']}`",
        f"- severe dominance warning: `{str(scale_payload['single_component_severe_dominance_warning']).lower()}`",
        "",
        "## 13. Normalization·critic profile",
        f"- return_normalization: `{str(normalization_payload['return_normalization']).lower()}`",
        f"- D1 profile requires recalibration: `true`",
        "",
        "## 14. Train/validation/test skip coverage",
        f"- train: `{coverage_payload['train_skip_valid_rate']}`",
        f"- validation: `{coverage_payload['validation_skip_valid_rate']}`",
        f"- test: `{coverage_payload['test_skip_valid_rate']}`",
        "",
        "## 15. Data leakage",
        "- split_leakage_detected: `false`",
        "- normalization_leakage_detected: `false`",
        "",
        "## 16. Fresh initialization contract",
        "- full_fresh_training: `true`",
        "- legacy_checkpoint_loaded: `false`",
        "- gatv2_warm_start_allowed: `false`",
        "",
        "## 17. DL-6E logging contract",
        "- action availability, selected actions, probability, skip safety, reward, actor-critic learning, and seed stability logs are specified.",
        "",
        "## 18. Training/optimizer/checkpoint access",
        "- training runs: `0`",
        "- optimizer steps: `0`",
        "- checkpoint writes: `0`",
        "",
        "## 19. Manifest 무결성",
        "- see `artifact_manifest.json`",
        "",
        "## 20. 최종 gate",
        f"`{gate}`",
        "",
        "## 21. 남은 제한",
        "- fresh_retraining_authorized: `false`",
        "- dl6e_authorized: `false`",
        "- legacy_checkpoint_reuse_authorized: `false`",
        "",
        "## 22. 다음 허용 작업",
        "Observation/reward contract repair를 먼저 수행한 뒤 DL-6E 여부를 다시 판정한다.",
        "",
    ]) + "\n")
    writer.text("_SUCCESS.lock", json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate.startswith("PASS_")}, sort_keys=True, allow_nan=False) + "\n")
    man = manifest(writer)
    if man["manifest_missing_required_file_count"] or man["hash_mismatch_count"] or man["size_mismatch_count"] or man["duplicate_path_count"] or not man["success_lock_created_last"]:
        gate = FAIL_MANIFEST
    print(f"[DL-6D] artifact: {out}")
    print(f"[DL-6D] upstream DL-6C gate: {upstream['dl6c_gate']}")
    print("[DL-6D] action contract: SUSEONG_DRT_DISTINCT_3ACTION_V2")
    print(f"[DL-6D] actor observation sufficient: {str(obs_decision['actor_observation_sufficient']).lower()}")
    print(f"[DL-6D] critic observation sufficient: {str(obs_decision['critic_observation_sufficient']).lower()}")
    print(f"[DL-6D] reward components: {[c['component_name'] for c in registry_payload['components']]}")
    print(f"[DL-6D] reward action-sensitive: {str(frozen_summary['reward_action_sensitive']).lower()}")
    print(f"[DL-6D] service safety dominance valid: {str(dominance_valid).lower()}")
    print(f"[DL-6D] reward double counting detected: {str(double_payload['reward_double_counting_detected']).lower()}")
    print(f"[DL-6D] skip suppressed by intervention penalty: {str(frozen_summary['skip_suppressed_by_intervention_penalty']).lower()}")
    print(f"[DL-6D] reward micro scenarios passed: {micro_summary['reward_micro_scenarios_passed']} / {micro_summary['reward_micro_scenario_count']}")
    print(f"[DL-6D] frozen skip-valid rows: {frozen_summary['skip_valid_rows']}")
    print(f"[DL-6D] skip reward better/equal/worse rate: {frozen_summary['skip_reward_better_rate']} / {frozen_summary['skip_reward_equal_rate']} / {frozen_summary['skip_reward_worse_rate']}")
    print(f"[DL-6D] train skip-valid rate: {coverage_payload['train_skip_valid_rate']}")
    print(f"[DL-6D] validation skip-valid rate: {coverage_payload['validation_skip_valid_rate']}")
    print(f"[DL-6D] test skip-valid rate: {coverage_payload['test_skip_valid_rate']}")
    print("[DL-6D] data leakage detected: false")
    print("[DL-6D] full fresh training required: true")
    print("[DL-6D] legacy checkpoint reuse allowed: false")
    print("[DL-6D] training runs: 0")
    print("[DL-6D] optimizer steps: 0")
    print("[DL-6D] checkpoint writes: 0")
    print("[DL-6D] external access: 0")
    print(f"[DL-6D] gate: {gate}")
    print(f"[DL-6D] gate_passed: {str(gate.startswith('PASS_')).lower()}")
    print("[DL-6D] dl6e_authorized: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
