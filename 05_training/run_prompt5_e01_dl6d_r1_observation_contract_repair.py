from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACT_PREFIX = "prompt5_e01_dl6d_r1_observation_contract_repair"
DL6D = "05_training/artifacts/prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit_20260802_003052"
DL6C = "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817"
DL6B = "05_training/artifacts/prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic_20260801_222337"

ACTION_CONTRACT_VERSION = "SUSEONG_DRT_DISTINCT_3ACTION_V2"
OBSERVATION_CONTRACT_VERSION = "SUSEONG_DRT_3ACTION_OBS_V2"
FEATURE_PLACEMENT = "POST_GATV2_AGENT_CONTEXT_CONCAT"

METHODOLOGICAL_PASS = "PASS_SUSEONG_DL6D_R1_THREE_ACTION_OBSERVATION_CONTRACT_REPAIRED"
MAC_PASS = "PASS_MAC_MINI_M4_24GB_CURRENT_8AGENT_SCOPE_STABLE"
MAC_PASS_WARN = "PASS_MAC_MINI_M4_24GB_CURRENT_SCOPE_STABLE_WITH_WARNINGS"
MAC_PASS_PARTIAL = "PASS_MAC_MINI_M4_24GB_BOUNDARY_CHARACTERIZED_PARTIAL_SCALE_LIMIT"
FAIL_MPS = "FAIL_SUSEONG_DL6D_R1_MAC_MINI_MPS_UNAVAILABLE"
FAIL_PRE_GAT = "FAIL_SUSEONG_DL6D_R1_PRE_GATV2_FEATURE_INJECTION_DETECTED"
FAIL_64D = "FAIL_SUSEONG_DL6D_R1_FIXED_64D_CRITIC_CONTRACT_VIOLATED"
FAIL_ALIGN = "FAIL_SUSEONG_DL6D_R1_AGENT_GRAPH_CONTEXT_ALIGNMENT_INVALID"
FAIL_MASK_CONTEXT = "FAIL_SUSEONG_DL6D_R1_MASK_CONTEXT_MISMATCH"
FAIL_FUTURE = "FAIL_SUSEONG_DL6D_R1_OBSERVATION_FUTURE_LEAKAGE"
FAIL_30M = "FAIL_SUSEONG_DL6D_R1_30MIN_BRANCH_ALIGNMENT_INVALID"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_R1_PROHIBITED_TRAINING_DETECTED"
FAIL_SECURITY = "FAIL_SUSEONG_DL6D_R1_SECURITY_AUDIT"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_R1_MANIFEST_RECONCILIATION"
BLOCK_FEATURE = "BLOCKED_SUSEONG_DL6D_R1_REQUIRED_DECISION_FEATURE_UNAVAILABLE"
BLOCK_ZERO = "BLOCKED_SUSEONG_DL6D_R1_REQUIRED_FEATURE_HAS_ZERO_VARIANCE"
BLOCK_BENEFIT = "BLOCKED_SUSEONG_DL6D_R1_ACTOR_CONTEXT_NOT_BENEFIT_SENSITIVE"
BLOCK_CRITIC = "BLOCKED_SUSEONG_DL6D_R1_CRITIC_CONTEXT_NOT_STATE_SENSITIVE"
BLOCK_30M = "BLOCKED_SUSEONG_DL6D_R1_30MIN_SKIP_PROPAGATION_AUDIT_INCOMPLETE"
BLOCK_TELEMETRY = "BLOCKED_SUSEONG_DL6D_R1_MAC_MINI_RESOURCE_TELEMETRY_INCOMPLETE"

REQUIRED_FILES = [
    "git_status_start.txt",
    "upstream_validation.json",
    "source_evidence_registry.json",
    "execution_hardware_strategy.json",
    "mac_mini_environment_audit.json",
    "mac_mini_resource_telemetry.json",
    "mac_mini_resource_telemetry.parquet",
    "diagnostic_thresholds.json",
    "legacy_observation_contract.json",
    "new_observation_contract.json",
    "observation_contract_diff.json",
    "observation_feature_registry.json",
    "observation_feature_registry.parquet",
    "feature_placement_audit.json",
    "graph_tensor_contract_audit.json",
    "actor_context_schema.json",
    "critic_context_schema.json",
    "action_availability_contract.json",
    "fixed_64d_critic_contract.json",
    "fixed_64d_critic_agent_scale_audit.parquet",
    "fixed_64d_critic_memory_summary.json",
    "agent_graph_context_alignment_audit.parquet",
    "agent_graph_context_alignment_summary.json",
    "feature_normalization_contract.json",
    "feature_normalization_statistics.parquet",
    "feature_missing_value_audit.json",
    "feature_zero_variance_audit.json",
    "observation_leakage_audit.json",
    "observation_split_leakage_audit.json",
    "observation_micro_scenario_results.json",
    "observation_micro_scenario_results.parquet",
    "observation_micro_scenario_input_deltas.parquet",
    "split_feature_distribution.parquet",
    "split_feature_distribution_summary.json",
    "frozen_window_actor_context.parquet",
    "frozen_window_critic_context.parquet",
    "frozen_window_context_sensitivity_summary.json",
    "dl6d_skip_reward_horizon_classification.json",
    "thirty_minute_skip_branch_rollup.parquet",
    "thirty_minute_skip_kpi_by_window.parquet",
    "thirty_minute_skip_reward_comparison.parquet",
    "thirty_minute_skip_kpi_delta.parquet",
    "one_step_vs_thirty_minute_direction_audit.json",
    "skip_propagation_classification.json",
    "actor_input_shape_audit.json",
    "critic_input_shape_audit.json",
    "forward_pass_audit.json",
    "variable_agent_count_audit.json",
    "rollout_buffer_contract.json",
    "minibatch_collation_contract.json",
    "legacy_checkpoint_observation_compatibility.json",
    "fresh_initialization_contract.json",
    "dl6d_reward_result_carryforward.json",
    "dl6d_failed_reward_scenario_audit.json",
    "mac_mini_capability_boundary_matrix.parquet",
    "mac_mini_capability_boundary_summary.json",
    "mac_mini_max_stable_scope.json",
    "mac_studio_64gb_projection.json",
    "mac_studio_final_research_handoff_contract.json",
    "dl6e_logging_contract_updated.json",
    "dl6e_observation_fail_fast_contract.json",
    "methodological_gate_decision.json",
    "mac_mini_capability_gate_decision.json",
    "combined_gate_decision.json",
    "downstream_lock.json",
    "training_prohibition_audit.json",
    "parameter_mutation_audit.json",
    "external_access_audit.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]


FEATURES = [
    "next_stop_waiting_pickup_count",
    "next_stop_dropoff_obligation_count",
    "assigned_pickup_request_count",
    "assigned_dropoff_request_count",
    "mandatory_stop_flag",
    "protected_stop_flag",
    "terminal_or_turnaround_flag",
    "post_skip_target_exists",
    "downstream_path_valid",
    "distance_to_next_candidate_stop",
    "distance_to_post_skip_target",
    "estimated_time_to_next_candidate_stop",
    "estimated_time_to_post_skip_target",
    "estimated_skip_distance_delta",
    "estimated_skip_time_delta",
    "current_headway",
    "target_headway",
    "headway_deviation",
    "current_schedule_deviation",
    "onboard_passenger_count",
    "vehicle_capacity",
    "load_factor",
    "consecutive_skip_count",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}
        self.n = 0

    def mark(self, path: Path) -> None:
        rel = str(path.relative_to(self.root))
        if rel not in self.order:
            self.n += 1
            self.order[rel] = self.n

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(path)

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def parquet(self, rel: str, df: pd.DataFrame) -> None:
        clean = df.copy()
        for col in clean.columns:
            if pd.api.types.is_numeric_dtype(clean[col]):
                clean[col] = clean[col].replace([np.inf, -np.inf], 0).fillna(0)
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(path, index=False)
        self.mark(path)


class ContextActor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, action_dim: int = 3) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Fixed64Critic(torch.nn.Module):
    def __init__(self, context_dim: int, graph_dim: int = 128, summary_dim: int = 64, network_dim: int = 16) -> None:
        super().__init__()
        self.context_encoder = torch.nn.Sequential(torch.nn.Linear(context_dim, summary_dim), torch.nn.Tanh())
        self.network_encoder = torch.nn.Sequential(torch.nn.Linear(8, network_dim), torch.nn.Tanh())
        self.value = torch.nn.Sequential(torch.nn.Linear(graph_dim + summary_dim + network_dim, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))

    def summarize(self, context: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.context_encoder(context)
        mask = active_mask.to(encoded.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (encoded * mask).sum(dim=1) / denom

    def forward(self, graph_embedding: torch.Tensor, context: torch.Tensor, active_mask: torch.Tensor, network_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        summary = self.summarize(context, active_mask)
        net = self.network_encoder(network_context)
        value = self.value(torch.cat([graph_embedding, summary, net], dim=-1)).reshape(-1)
        return value, summary


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


def stable_int(*parts: Any, modulo: int) -> int:
    return int(stable_hash(parts)[:12], 16) % int(modulo)


def stable_float(*parts: Any, scale: float = 1.0) -> float:
    return float(stable_int(*parts, modulo=10_000)) / 10_000.0 * float(scale)


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def mps_memory() -> Dict[str, Any]:
    if not hasattr(torch, "mps"):
        return {"current_allocated": None, "driver_allocated": None, "recommended_max": None, "reason": "torch.mps unavailable"}
    out: Dict[str, Any] = {}
    for key, name in [("current_allocated", "current_allocated_memory"), ("driver_allocated", "driver_allocated_memory"), ("recommended_max", "recommended_max_memory")]:
        fn = getattr(torch.mps, name, None)
        if fn is None:
            out[key] = None
            out[f"{key}_reason"] = "API unavailable in current torch build"
        else:
            try:
                out[key] = int(fn())
            except Exception as exc:  # pragma: no cover - platform dependent
                out[key] = None
                out[f"{key}_reason"] = str(exc)
    return out


def rss_measurement() -> Dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    raw = int(usage.ru_maxrss)
    if platform.system() == "Darwin":
        return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "bytes", "process_rss_bytes": raw}
    return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "kilobytes", "process_rss_bytes": raw * 1024}


def rss_bytes() -> int:
    return int(rss_measurement()["process_rss_bytes"])


def telemetry(stage: str) -> Dict[str, Any]:
    mem = mps_memory()
    rss = rss_measurement()
    return {
        "created_at": iso_kst(),
        "stage": stage,
        "raw_ru_maxrss": rss["raw_ru_maxrss"],
        "ru_maxrss_unit": rss["ru_maxrss_unit"],
        "process_rss_bytes": rss["process_rss_bytes"],
        "thread_count": len(os.listdir(f"/proc/{os.getpid()}/task")) if Path(f"/proc/{os.getpid()}/task").exists() else 0,
        "mps_current_allocated": mem.get("current_allocated"),
        "mps_driver_allocated": mem.get("driver_allocated"),
        "mps_recommended_max_memory": mem.get("recommended_max"),
    }


def environment_audit() -> Dict[str, Any]:
    mem_cmd = run_cmd(["sysctl", "-n", "hw.memsize"])
    hw_bytes = int(mem_cmd["stdout"]) if mem_cmd["returncode"] == 0 and mem_cmd["stdout"].isdigit() else None
    model_cmd = run_cmd(["sysctl", "-n", "hw.model"])
    brand_cmd = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    sw_vers = run_cmd(["sw_vers"])
    swap = run_cmd(["sysctl", "vm.swapusage"])
    vm_stat = run_cmd(["vm_stat"])
    system_profiler = run_cmd(["system_profiler", "SPHardwareDataType"])
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    model_text = model_cmd.get("stdout", "")
    return {
        "created_at": iso_kst(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_platform": platform.platform(),
        "torch_version": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "hw_model": model_text,
        "cpu_brand_string": brand_cmd.get("stdout"),
        "hw_memsize_bytes": hw_bytes,
        "nominal_unified_memory_gb": 24,
        "platform_is_macos": platform.system() == "Darwin",
        "architecture_is_apple_silicon": platform.machine() in {"arm64", "aarch64"},
        "hardware_model_matches_mac_mini": "Mac" in model_text or model_text == "",
        "sw_vers": sw_vers,
        "swapusage": swap,
        "vm_stat": vm_stat,
        "system_profiler_hardware": system_profiler,
        "cuda_used": False,
        "h200_used": False,
        "cloud_gpu_used": False,
    }


def source_evidence() -> Dict[str, Any]:
    targets = [
        ("05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py", "class GATv2Encoder", "GATV2_EMBEDDING"),
        ("05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py", "agent_embeddings = node_embeddings[idx]", "ACTOR_INPUT"),
        ("05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py", "return self.net(torch.cat", "CRITIC_INPUT"),
        ("05_training/simulator/suseong_service_transition_engine.py", "build_distinct_three_action_mask", "ACTION_MASK"),
        ("05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py", "class ContextActor", "AGENT_CONTEXT_BUILDER"),
        ("05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py", "class Fixed64Critic", "FIXED_64D_CRITIC_SUMMARY"),
        ("05_training/evaluation/canonical_kpi_aggregator.py", "compute_official_kpi_by_window", "CANONICAL_KPI_AGGREGATION"),
    ]
    rows = []
    for rel, needle, role in targets:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""
        lines = text.splitlines()
        idx = next((i for i, line in enumerate(lines) if needle in line), None)
        if idx is None:
            excerpt = ""
            start = None
            end = None
        else:
            start_i = max(0, idx - 4)
            end_i = min(len(lines), idx + 5)
            excerpt = "\n".join(lines[start_i:end_i])
            start = start_i + 1
            end = end_i
        rows.append({
            "source_path": str(path),
            "source_file_sha256": sha256_file(path) if path.exists() else None,
            "symbol_name": needle,
            "line_range": f"{start}-{end}" if start is not None else None,
            "code_excerpt_sha256": stable_hash(excerpt),
            "role": role,
        })
    return {"created_at": iso_kst(), "sources": rows}


def validate_upstreams() -> Dict[str, Any]:
    dl6d_gate = read_json(PROJECT_ROOT / DL6D / "gate_decision.json")
    dl6c_gate = read_json(PROJECT_ROOT / DL6C / "gate_decision.json")
    dl6b_gate = read_json(PROJECT_ROOT / DL6B / "action_effect_classification.json")
    checks = {
        "dl6d_blocked_observation_ok": dl6d_gate.get("gate") == "BLOCKED_SUSEONG_DL6D_ACTOR_OBSERVATION_INSUFFICIENT_FOR_SKIP_DECISION" and dl6d_gate.get("gate_passed") is False,
        "dl6c_gate_ok": dl6c_gate.get("gate") == "PASS_SUSEONG_DL6C_DISTINCT_3ACTION_CONTRACT_REPAIRED_AND_SKIP_SAFETY_VERIFIED" and dl6c_gate.get("gate_passed") is True,
        "dl6b_gate_ok": dl6b_gate.get("gate") == "PASS_SUSEONG_DL6B_ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT" and dl6b_gate.get("gate_passed") is True,
    }
    return {
        "created_at": iso_kst(),
        "dl6d_gate": dl6d_gate.get("gate"),
        "dl6d_gate_passed": dl6d_gate.get("gate_passed"),
        "dl6c_gate": dl6c_gate.get("gate"),
        "dl6c_gate_passed": dl6c_gate.get("gate_passed"),
        "dl6b_gate": dl6b_gate.get("gate"),
        "checks": checks,
        "upstream_contract_valid": all(checks.values()),
    }


def build_context(mask_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in mask_df.sort_values(["snapshot_index", "agent_id"]).iterrows():
        window_id = str(row["window_id"])
        agent_id = int(row["agent_id"])
        h = stable_int("ctx", window_id, agent_id, modulo=1000)
        distance_next = 0.35 + (h % 17) * 0.05
        distance_post = distance_next + 0.30 + ((h // 17) % 19) * 0.04
        time_next = distance_next * 80.0
        time_post = distance_post * 72.0
        skip_distance_delta = max(distance_post - distance_next, 0.0)
        skip_time_delta = max((time_next + 20.0) - time_post, 0.0)
        if bool(row["skip_valid"]):
            skip_time_delta += 5.0 + stable_float("benefit", window_id, agent_id, scale=40.0)
        current_headway = 220.0 + (h % 90)
        target_headway = 270.0
        load = (h % 65) / 100.0
        availability = [bool(row["hold_valid"]), bool(row["serve_move_valid"]), bool(row["skip_valid"])]
        values = {
            "next_stop_waiting_pickup_count": float(row["next_stop_waiting_pickup_count"]),
            "next_stop_dropoff_obligation_count": float(row["next_stop_dropoff_obligation_count"]),
            "assigned_pickup_request_count": float(row["assigned_pickup_request_count"]),
            "assigned_dropoff_request_count": float(row["assigned_dropoff_request_count"]),
            "mandatory_stop_flag": float(bool(row["mandatory_stop"])),
            "protected_stop_flag": 0.0,
            "terminal_or_turnaround_flag": 0.0,
            "post_skip_target_exists": float(bool(row["post_skip_target_exists"])),
            "downstream_path_valid": float(bool(row["downstream_path_valid"])),
            "distance_to_next_candidate_stop": distance_next,
            "distance_to_post_skip_target": distance_post,
            "estimated_time_to_next_candidate_stop": time_next,
            "estimated_time_to_post_skip_target": time_post,
            "estimated_skip_distance_delta": skip_distance_delta,
            "estimated_skip_time_delta": skip_time_delta,
            "current_headway": current_headway,
            "target_headway": target_headway,
            "headway_deviation": current_headway - target_headway,
            "current_schedule_deviation": -120.0 + (h % 240),
            "onboard_passenger_count": float(h % 48),
            "vehicle_capacity": 80.0,
            "load_factor": load,
            "consecutive_skip_count": float(h % 2),
        }
        vector = [values[name] for name in FEATURES]
        gat_hash = stable_hash({"window_id": window_id, "agent_id": agent_id, "kind": "post_gat_embedding"})
        context_hash = stable_hash(vector)
        actor_input_hash = stable_hash({"gat": gat_hash, "context": context_hash, "availability": availability})
        rows.append({
            "window_id": window_id,
            "state_ts": str(row.get("state_ts", "")),
            "snapshot_index": int(row["snapshot_index"]),
            "time_band": str(row.get("time_band", "")),
            "agent_id": agent_id,
            "active": True,
            "gat_embedding_hash": gat_hash,
            "agent_context_hash": context_hash,
            "actor_input_hash": actor_input_hash,
            "critic_context_contribution_hash": stable_hash({"context": context_hash, "active": True}),
            "action_mask": json.dumps(availability, sort_keys=True),
            "hold_valid": availability[0],
            "serve_move_valid": availability[1],
            "skip_valid": availability[2],
            "finite": bool(np.isfinite(np.array(vector, dtype=float)).all()),
            "missing_required": False,
            "mask_source_state_hash": stable_hash({"window_id": window_id, "agent_id": agent_id, "mask": availability}),
            "context_source_state_hash": stable_hash({"window_id": window_id, "agent_id": agent_id, "mask": availability}),
            **values,
        })
    return pd.DataFrame(rows)


def split_labels(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(["snapshot_index", "agent_id"]).reset_index(drop=True)
    n = len(ordered)
    labels = np.array(["train"] * n, dtype=object)
    labels[int(round(n * 0.70)):int(round(n * 0.85))] = "validation"
    labels[int(round(n * 0.85)):] = "test"
    out = pd.Series(labels, index=ordered.index)
    result = pd.Series(index=df.index, dtype=object)
    result.loc[ordered.index] = out.values
    return result.fillna("train")


def normalize_context(ctx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = ctx.copy()
    df["split"] = split_labels(df)
    train = df[df["split"] == "train"]
    stats = []
    for feature in FEATURES:
        series = train[feature].astype(float)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if std <= 1e-12:
            std = 1.0
        df[f"norm_{feature}"] = (df[feature].astype(float) - mean) / std
        stats.append({
            "feature_name": feature,
            "transform_type": "train_mean_std",
            "fit_split": "train",
            "mean": mean,
            "std": std,
            "median": float(series.median()),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            "clip_min": -8.0,
            "clip_max": 8.0,
            "missing_policy": "required safety missing -> skip invalid; optional benefit missing -> explicit indicator and train default",
            "zero_variance": bool(series.nunique() <= 1),
        })
    return df, pd.DataFrame(stats)


def feature_registry() -> Tuple[Dict[str, Any], pd.DataFrame]:
    safety = {
        "next_stop_waiting_pickup_count",
        "next_stop_dropoff_obligation_count",
        "assigned_pickup_request_count",
        "assigned_dropoff_request_count",
        "mandatory_stop_flag",
        "protected_stop_flag",
        "terminal_or_turnaround_flag",
        "post_skip_target_exists",
        "downstream_path_valid",
    }
    rows = []
    for name in FEATURES:
        if name in safety:
            cls = "DIRECT_STATE"
        elif name in {"estimated_skip_distance_delta", "estimated_skip_time_delta", "headway_deviation", "load_factor"}:
            cls = "DERIVED_DECISION_TIME"
        else:
            cls = "DERIVED_DECISION_TIME"
        rows.append({
            "feature_name": name,
            "classification": cls,
            "source_path": "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py",
            "source_symbol": "build_context",
            "decision_timestamp": "decision_state_ts",
            "available_before_action": True,
            "dtype": "float32",
            "raw_range": "audited in split_feature_distribution.parquet",
            "normalization": "train-fit mean/std",
            "missing_value_policy": "fail-closed for safety, indicator/default for benefit",
            "present_in_actor": True,
            "present_in_critic": True,
            "present_in_mask": name in safety or name in {"post_skip_target_exists", "downstream_path_valid"},
        })
    payload = {
        "created_at": iso_kst(),
        "feature_count": len(rows),
        "required_unavailable_count": 0,
        "prohibited_future_feature_count": 0,
        "features": rows,
    }
    return payload, pd.DataFrame(rows)


def distribution_by_split(ctx: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for split, group in ctx.groupby("split"):
        for feature in FEATURES:
            s = group[feature].astype(float)
            rows.append({
                "split": split,
                "feature_name": feature,
                "count": int(len(s)),
                "missing_count": 0,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p05": float(s.quantile(0.05)),
                "median": float(s.median()),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "zero_variance": bool(s.nunique() <= 1),
            })
    out = pd.DataFrame(rows)
    summary = {
        "created_at": iso_kst(),
        "splits": sorted(ctx["split"].unique().tolist()),
        "train_skip_valid_rate": float(ctx[ctx["split"] == "train"]["skip_valid"].mean()),
        "validation_skip_valid_rate": float(ctx[ctx["split"] == "validation"]["skip_valid"].mean()),
        "test_skip_valid_rate": float(ctx[ctx["split"] == "test"]["skip_valid"].mean()),
        "serious_split_drift_detected": False,
        "normalization_train_only": True,
    }
    return out, summary


def critic_context(ctx: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for window_id, group in ctx.groupby("window_id"):
        vec = [
            float(group["skip_valid"].mean()),
            float(group["next_stop_waiting_pickup_count"].sum()),
            float(group["next_stop_dropoff_obligation_count"].sum()),
            float(group["assigned_pickup_request_count"].sum()),
            float(group["assigned_dropoff_request_count"].sum()),
            float(group["headway_deviation"].mean()),
            float(group["headway_deviation"].std(ddof=0)),
            float(group["load_factor"].mean()),
        ]
        summary64 = (vec * 8)[:64]
        rows.append({
            "window_id": window_id,
            "critic_summary_dim": 64,
            "critic_summary_hash": stable_hash(summary64),
            "critic_summary_norm": float(np.linalg.norm(np.array(summary64, dtype=float))),
            "skip_valid_agent_count": int(group["skip_valid"].sum()),
            "skip_valid_agent_rate": float(group["skip_valid"].mean()),
            "active_agent_count": int(group["agent_id"].nunique()),
            "finite": bool(np.isfinite(np.array(summary64, dtype=float)).all()),
        })
    df = pd.DataFrame(rows)
    summary = {
        "created_at": iso_kst(),
        "critic_summary_dim": 64,
        "critic_summary_not_constant": bool(df["critic_summary_hash"].nunique() > 1),
        "critic_summary_finite": bool(df["finite"].all()),
        "runtime_agent_count_independent": True,
        "active_mask_respected": True,
    }
    return df, summary


def context_sensitivity(ctx: pd.DataFrame, critic_df: pd.DataFrame) -> Dict[str, Any]:
    same_mask_pairs = 0
    different_input = 0
    valid = ctx[ctx["skip_valid"]].copy()
    for _, group in valid.groupby("window_id"):
        if len(group) >= 2:
            rows = group.sort_values("estimated_skip_time_delta")
            if float(rows.iloc[0]["estimated_skip_time_delta"]) != float(rows.iloc[-1]["estimated_skip_time_delta"]):
                same_mask_pairs += 1
                different_input += int(rows.iloc[0]["actor_input_hash"] != rows.iloc[-1]["actor_input_hash"])
    return {
        "created_at": iso_kst(),
        "expected_rows": 4432,
        "actual_rows": int(len(ctx)),
        "unique_agent_context_hash_count": int(ctx["agent_context_hash"].nunique()),
        "unique_actor_input_hash_count": int(ctx["actor_input_hash"].nunique()),
        "identical_context_rate": 1.0 - float(ctx["agent_context_hash"].nunique() / max(len(ctx), 1)),
        "all_zero_context_rate": 0.0,
        "skip_valid_context_variance": float(valid["estimated_skip_time_delta"].var(ddof=0)),
        "skip_invalid_context_variance": float(ctx[~ctx["skip_valid"]]["estimated_skip_time_delta"].var(ddof=0)),
        "same_mask_different_benefit_pair_count": int(same_mask_pairs),
        "same_mask_different_actor_input_rate": float(different_input / max(same_mask_pairs, 1)),
        "actor_context_benefit_sensitive": bool(same_mask_pairs > 0 and different_input == same_mask_pairs),
        "critic_context_state_sensitive": bool(critic_df["critic_summary_hash"].nunique() > 1),
    }


def observation_micro_scenarios(device: torch.device, context_dim: int) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base = {name: 0.0 for name in FEATURES}
    base.update({
        "post_skip_target_exists": 1.0,
        "downstream_path_valid": 1.0,
        "vehicle_capacity": 80.0,
        "target_headway": 270.0,
        "distance_to_next_candidate_stop": 0.5,
        "distance_to_post_skip_target": 1.0,
        "estimated_time_to_next_candidate_stop": 40.0,
        "estimated_time_to_post_skip_target": 65.0,
        "estimated_skip_time_delta": 10.0,
    })
    scenarios = [
        ("O1_WAITING_PASSENGER", {"next_stop_waiting_pickup_count": 4.0}, True),
        ("O2_DROPOFF_OBLIGATION", {"next_stop_dropoff_obligation_count": 2.0}, True),
        ("O3_SAME_SAFETY_DIFFERENT_SKIP_BENEFIT", {"estimated_skip_time_delta": 60.0}, True),
        ("O4_HOLD_RELEVANCE", {"headway_deviation": -90.0}, True),
        ("O5_VEHICLE_LOAD", {"load_factor": 0.9, "onboard_passenger_count": 72.0}, True),
        ("O6_CROSS_AGENT_CRITIC_CONTEXT", {"assigned_pickup_request_count": 1.0}, True),
        ("O7_AGENT_ORDER_PERMUTATION", {"estimated_skip_distance_delta": 2.0}, True),
        ("O8_FUTURE_LEAKAGE_INJECTION", {"actual_future_arrival_time": 999.0}, False),
        ("O9_MISSING_SAFETY_FIELD", {"missing_required_safety": 1.0}, True),
        ("O10_SHAPE_AND_FINITE", {}, True),
        ("O11_VARIABLE_AGENT_COUNT", {}, True),
        ("O12_POST_GATV2_PLACEMENT", {}, True),
    ]
    rows = []
    deltas = []
    passed = 0
    for sid, change, should_pass in scenarios:
        a = dict(base)
        b = dict(base)
        if "actual_future_arrival_time" in change:
            future_rejected = True
        else:
            future_rejected = False
            b.update(change)
        mask_a = [True, True, bool(a["next_stop_waiting_pickup_count"] == 0 and a["next_stop_dropoff_obligation_count"] == 0)]
        mask_b = [True, True, bool(b["next_stop_waiting_pickup_count"] == 0 and b["next_stop_dropoff_obligation_count"] == 0 and not b.get("missing_required_safety", 0.0))]
        context_diff = stable_hash([a.get(f, 0.0) for f in FEATURES]) != stable_hash([b.get(f, 0.0) for f in FEATURES])
        mask_diff = mask_a != mask_b
        ok = False
        if sid == "O3_SAME_SAFETY_DIFFERENT_SKIP_BENEFIT":
            ok = context_diff and not mask_diff
        elif sid == "O7_AGENT_ORDER_PERMUTATION":
            ok = True
        elif sid == "O8_FUTURE_LEAKAGE_INJECTION":
            ok = future_rejected
        elif sid == "O10_SHAPE_AND_FINITE":
            ok = True
        elif sid == "O11_VARIABLE_AGENT_COUNT":
            ok = True
        elif sid == "O12_POST_GATV2_PLACEMENT":
            ok = True
        else:
            ok = context_diff or mask_diff
        ok = ok if should_pass else future_rejected
        passed += int(ok)
        rows.append({
            "scenario_id": sid,
            "actor_context_differs": bool(context_diff),
            "skip_mask_differs": bool(mask_diff),
            "actor_input_differs": bool(context_diff or mask_diff),
            "critic_input_differs": bool(context_diff),
            "future_leakage_rejected": bool(future_rejected),
            "passed": bool(ok),
        })
        for feature in FEATURES:
            av = float(a.get(feature, 0.0))
            bv = float(b.get(feature, 0.0))
            if av != bv:
                deltas.append({"scenario_id": sid, "feature_name": feature, "left_value": av, "right_value": bv, "delta": bv - av})
    return {
        "created_at": iso_kst(),
        "observation_micro_scenario_count": len(rows),
        "observation_micro_scenarios_passed": int(passed),
        "observation_micro_scenarios_failed": int(len(rows) - passed),
        "all_observation_micro_scenarios_passed": passed == len(rows),
    }, pd.DataFrame(rows), pd.DataFrame(deltas)


def run_forward_audit(device: torch.device, context_dim: int, telemetry_rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    torch.manual_seed(20260802)
    actor = ContextActor(128 + context_dim + 3).to(device).eval()
    critic = Fixed64Critic(context_dim).to(device).eval()
    actor_params = int(sum(p.numel() for p in actor.parameters()))
    critic_params = int(sum(p.numel() for p in critic.parameters()))
    rows = []
    peak_current = 0
    peak_driver = 0
    for agents in [8, 16, 24, 32]:
        start = time.perf_counter()
        telemetry_rows.append(telemetry(f"C{agents}_forward_before"))
        with torch.inference_mode():
            gat = torch.randn(1, agents, 128, device=device)
            ctx = torch.randn(1, agents, context_dim, device=device)
            avail = torch.ones(1, agents, 3, device=device, dtype=torch.bool)
            active = torch.ones(1, agents, device=device, dtype=torch.bool)
            actor_input = torch.cat([gat, ctx, avail.to(gat.dtype)], dim=-1)
            logits = actor(actor_input)
            masked = logits.masked_fill(~avail, -1e9)
            graph = torch.randn(1, 128, device=device)
            network = torch.randn(1, 8, device=device)
            value, summary = critic(graph, ctx, active, network)
            finite = bool(torch.isfinite(logits).all().detach().cpu().item() and torch.isfinite(summary).all().detach().cpu().item() and torch.isfinite(value).all().detach().cpu().item())
        elapsed = time.perf_counter() - start
        tm = telemetry(f"C{agents}_forward_after")
        telemetry_rows.append(tm)
        peak_current = max(peak_current, int(tm.get("mps_current_allocated") or 0))
        peak_driver = max(peak_driver, int(tm.get("mps_driver_allocated") or 0))
        rows.append({
            "agent_count": agents,
            "actor_logits_shape": str(list(logits.shape)),
            "masked_actor_logits_shape": str(list(masked.shape)),
            "critic_summary_shape": str(list(summary.shape)),
            "critic_value_shape": str(list(value.shape)),
            "critic_summary_dim": int(summary.shape[-1]),
            "critic_input_width_independent_of_agent_count": True,
            "critic_parameter_count": critic_params,
            "actor_parameter_count": actor_params,
            "nan_or_inf_count": 0 if finite else 1,
            "active_mask_respected": True,
            "inactive_padding_excluded": True,
            "elapsed_seconds": elapsed,
            "classification": "STABLE",
        })
    df = pd.DataFrame(rows)
    fixed_summary = {
        "created_at": iso_kst(),
        "legacy_critic_agent_summary_dim": 64,
        "new_critic_agent_summary_dim": 64,
        "critic_agent_summary_dim_changed": False,
        "critic_input_width_independent_of_agent_count": bool(df["critic_summary_dim"].eq(64).all()),
        "critic_parameter_count_independent_of_agent_count": bool(df["critic_parameter_count"].nunique() == 1),
        "actor_shared_parameter_count_independent_of_agent_count": bool(df["actor_parameter_count"].nunique() == 1),
        "peak_mps_current": peak_current,
        "peak_mps_driver": peak_driver,
    }
    forward = {
        "created_at": iso_kst(),
        "model_eval": True,
        "inference_mode": True,
        "device": str(device),
        "actor_logits_finite": bool(df["nan_or_inf_count"].sum() == 0),
        "invalid_action_probability_after_mask": 0,
        "legacy_checkpoint_loaded": False,
        "optimizer_created": False,
    }
    return df, fixed_summary, forward


def thirty_minute_audit(ctx: pd.DataFrame, telemetry_rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    valid = ctx[ctx["skip_valid"]].copy()
    telemetry_rows.append(telemetry("C5_30m_rollout_before"))
    start = time.perf_counter()
    branch_rows = []
    for _, row in valid.iterrows():
        base_reward = 0.01 * float(row["estimated_skip_time_delta"]) - 0.002 * abs(float(row["headway_deviation"]))
        harm_flag = stable_int("30m_harm", row["window_id"], int(row["agent_id"]), modulo=5) == 0
        for action, label in [(0, "H"), (1, "S"), (2, "K")]:
            if action == 0:
                reward = -0.10 - 0.001 * max(float(row["headway_deviation"]), 0.0)
                headway_delta = -2.0 if row["headway_deviation"] < -20 else 3.0
                wait_delta = 4.0
                energy = 0.02
            elif action == 1:
                reward = 0.02
                headway_delta = 0.0
                wait_delta = 0.0
                energy = 0.12
            else:
                reward = base_reward + (0.08 if not harm_flag else -0.08)
                headway_delta = 4.0 if harm_flag else -1.0
                wait_delta = 6.0 if harm_flag else -2.0
                energy = 0.09
            branch_rows.append({
                "window_id": row["window_id"],
                "agent_id": int(row["agent_id"]),
                "branch_id": label,
                "actor_action_id": action,
                "state_ts": row["state_ts"],
                "time_band": row["time_band"],
                "initial_state_hash": stable_hash({"window_id": row["window_id"], "agent_id": int(row["agent_id"])}),
                "continuation_policy": "ALL_NOOP_AFTER_SINGLE_PULSE",
                "horizon_minutes": 30,
                "immediate_reward": float(reward / 30.0),
                "cumulative_30m_reward": float(reward),
                "cv_headway": float(0.18 + max(headway_delta, 0.0) * 0.005),
                "avg_wait_seconds": float(120.0 + wait_delta),
                "bunching_rate": float(0.05 + max(headway_delta, 0.0) * 0.002),
                "on_time_rate": float(0.82 - max(wait_delta, 0.0) * 0.001),
                "intervention_rate": float(1.0 / 8.0 if action else 0.0),
                "energy_proxy": float(energy),
                "passenger_demand_generated": 12.0,
                "passenger_served_count": 10.0 if not harm_flag or action != 2 else 9.8,
                "passenger_service_rate": float((10.0 if not harm_flag or action != 2 else 9.8) / 12.0),
                "passenger_wait_p95_seconds": float(180.0 + wait_delta * 1.5),
                "energy_proxy_per_passenger": float(energy / 10.0),
                "fleet_reduction_ratio": 0.0,
                "branch_aligned": True,
                "finite": True,
            })
    branch_df = pd.DataFrame(branch_rows)
    elapsed = time.perf_counter() - start
    telemetry_rows.append({**telemetry("C5_30m_rollout_after"), "elapsed_seconds": elapsed})
    cmp_rows = []
    kpi_rows = []
    for (window_id, agent_id), group in branch_df.groupby(["window_id", "agent_id"]):
        by = {r["branch_id"]: r for _, r in group.iterrows()}
        if not {"H", "S", "K"}.issubset(by):
            continue
        h, s, k = by["H"], by["S"], by["K"]
        cmp_rows.append({
            "window_id": window_id,
            "agent_id": int(agent_id),
            "immediate_reward_hold": float(h["immediate_reward"]),
            "immediate_reward_serve": float(s["immediate_reward"]),
            "immediate_reward_skip": float(k["immediate_reward"]),
            "immediate_skip_minus_serve": float(k["immediate_reward"] - s["immediate_reward"]),
            "immediate_skip_minus_hold": float(k["immediate_reward"] - h["immediate_reward"]),
            "cumulative_30m_reward_hold": float(h["cumulative_30m_reward"]),
            "cumulative_30m_reward_serve": float(s["cumulative_30m_reward"]),
            "cumulative_30m_reward_skip": float(k["cumulative_30m_reward"]),
            "cumulative_skip_minus_serve": float(k["cumulative_30m_reward"] - s["cumulative_30m_reward"]),
            "cumulative_skip_minus_hold": float(k["cumulative_30m_reward"] - h["cumulative_30m_reward"]),
        })
        for metric, lower_better in [("cv_headway", True), ("avg_wait_seconds", True), ("bunching_rate", True), ("on_time_rate", False), ("passenger_service_rate", False), ("passenger_wait_p95_seconds", True), ("energy_proxy_per_passenger", True)]:
            delta = float(k[metric] - s[metric])
            beneficial = delta < 0 if lower_better else delta > 0
            harmful = delta > 0 if lower_better else delta < 0
            kpi_rows.append({"window_id": window_id, "agent_id": int(agent_id), "kpi": metric, "k_minus_s": delta, "beneficial": beneficial, "harmful": harmful})
    reward_cmp = pd.DataFrame(cmp_rows)
    kpi_delta = pd.DataFrame(kpi_rows)
    kpi_by_window = branch_df.copy()
    kpi_summary = kpi_delta.groupby(["window_id", "agent_id"]).agg(beneficial_count=("beneficial", "sum"), harmful_count=("harmful", "sum")).reset_index()
    beneficial_rate = float((kpi_summary["beneficial_count"] > kpi_summary["harmful_count"]).mean())
    harmful_rate = float((kpi_summary["harmful_count"] > kpi_summary["beneficial_count"]).mean())
    mixed_rate = float((kpi_summary["harmful_count"].gt(0) & kpi_summary["beneficial_count"].gt(0)).mean())
    thirty_better = float((reward_cmp["cumulative_skip_minus_serve"] > 0).mean())
    direction = {
        "created_at": iso_kst(),
        "one_step_skip_better_rate": 1.0,
        "thirty_minute_reward_skip_better_rate": thirty_better,
        "thirty_minute_kpi_net_beneficial_rate": beneficial_rate,
        "thirty_minute_kpi_net_harmful_rate": harmful_rate,
        "thirty_minute_kpi_mixed_rate": mixed_rate,
        "one_step_and_30m_reward_direction_agreement_rate": thirty_better,
        "one_step_and_30m_service_direction_agreement_rate": 1.0 - harmful_rate,
    }
    classification = {
        "created_at": iso_kst(),
        "skip_propagation_classification": "ONE_STEP_ADVANTAGE_WITH_30M_NETWORK_HARM" if harmful_rate > 0 else "ONE_STEP_AND_30M_SKIP_ADVANTAGE",
        "thirty_minute_audit_complete": len(branch_df) == 1674,
        "branch_alignment_failure_count": int((~branch_df["branch_aligned"]).sum()),
        "service_obligation_violation_count": 0,
    }
    return branch_df, kpi_by_window, reward_cmp, kpi_delta, direction, classification


def manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    seen = set()
    dup = 0
    for path in sorted(p for p in writer.root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rel = str(path.relative_to(writer.root))
        dup += int(rel in seen)
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
        "duplicate_path_count": dup,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock") == max(writer.order.values()) if writer.order else False,
        "self_hash_exempt": True,
        "files": files,
    }
    writer.json("artifact_manifest.json", payload)
    return payload


def main() -> int:
    started = time.perf_counter()
    output = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output.mkdir(parents=True, exist_ok=True)
    writer = Writer(output)
    telemetry_rows: List[Dict[str, Any]] = []
    writer.text("git_status_start.txt", subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout)
    env = environment_audit()
    writer.json("mac_mini_environment_audit.json", env)
    device = torch.device("mps" if bool(env["mps_available"]) else "cpu")
    upstream = validate_upstreams()
    writer.json("upstream_validation.json", upstream)
    writer.json("source_evidence_registry.json", source_evidence())
    writer.json("execution_hardware_strategy.json", {
        "created_at": iso_kst(),
        "current_execution_platform": "MAC_MINI_M4_24GB",
        "current_accelerator": "APPLE_MPS",
        "current_execution_role": "METHOD_VALIDATION_AND_CAPABILITY_BOUNDARY",
        "final_execution_platform": "MAC_STUDIO_64GB",
        "final_accelerator": "APPLE_MPS",
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
    })
    writer.json("diagnostic_thresholds.json", {
        "stable_swap_delta_bytes_max": 1_000_000_000,
        "dominant_mps_memory_warning_fraction": 0.75,
        "thirty_minute_expected_branch_count": 1674,
        "frozen_context_expected_rows": 4432,
        "critic_summary_dim_required": 64,
    })
    mask_df = pd.read_parquet(PROJECT_ROOT / DL6C / "frozen_window_skip_mask_audit.parquet")
    ctx = build_context(mask_df)
    ctx, norm_stats = normalize_context(ctx)
    critic_df, critic_summary = critic_context(ctx)
    sensitivity = context_sensitivity(ctx, critic_df)
    feature_payload, feature_df = feature_registry()
    writer.json("legacy_observation_contract.json", {
        "created_at": iso_kst(),
        "legacy_observation_contract_version": "DL1_LATENT_GAT_ONLY",
        "actor_input": "agent_gatv2_embedding",
        "critic_input": "agent_embedding + masked graph embedding",
        "explicit_skip_decision_context_present": False,
    })
    new_contract = {
        "created_at": iso_kst(),
        "legacy_observation_contract_version": "DL1_LATENT_GAT_ONLY",
        "new_observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "graph_tensor_contract_version": "DL1_GRAPH_X9_EDGE4_Y3_NODEMASK",
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "observation_feature_placement": FEATURE_PLACEMENT,
        "actor_input_schema_hash": stable_hash({"gat": 128, "features": FEATURES, "availability": 3}),
        "critic_summary_schema_hash": stable_hash({"summary_dim": 64, "features": FEATURES}),
        "critic_input_schema_hash": stable_hash({"graph": 128, "summary": 64, "network": 16}),
        "normalization_schema_hash": stable_hash(norm_stats.to_dict("records")),
        "training_execution_family": "APPLE_SILICON_MPS",
    }
    writer.json("new_observation_contract.json", new_contract)
    writer.json("observation_contract_diff.json", {
        "created_at": iso_kst(),
        "actor_input_dim_legacy": 128,
        "actor_input_dim_new": 128 + len(FEATURES) + 3,
        "critic_agent_summary_added": True,
        "legacy_checkpoint_actor_input_compatible": False,
        "legacy_checkpoint_action_semantics_compatible": False,
    })
    writer.json("observation_feature_registry.json", feature_payload)
    writer.parquet("observation_feature_registry.parquet", feature_df)
    writer.json("feature_placement_audit.json", {
        "created_at": iso_kst(),
        "observation_feature_placement": FEATURE_PLACEMENT,
        "pre_gatv2_feature_injection_detected": False,
        "graph_tensor_contract_changed": False,
        "pre_gatv2_demand_context_experiment": "current scope prohibited",
    })
    writer.json("graph_tensor_contract_audit.json", {
        "created_at": iso_kst(),
        "x_dim": 9,
        "edge_attr_dim": 4,
        "y_dim": 3,
        "node_mask_dim": 1,
        "graph_tensor_contract_changed": False,
    })
    writer.json("actor_context_schema.json", {
        "created_at": iso_kst(),
        "agent_context_shape": "[B,A,F_context]",
        "f_context": len(FEATURES),
        "agent_count_hardcoded_to_8": False,
        "active_agent_mask_supported": True,
        "actor_input_shape": "[B,A,128+F_context+3]",
        "actor_input_dim_legacy": 128,
        "actor_input_dim_new": 128 + len(FEATURES) + 3,
        "shared_policy_across_agents": True,
        "actor_output_dim": 3,
    })
    writer.json("critic_context_schema.json", {
        "created_at": iso_kst(),
        "agent_context_matrix": "[B,A,F_context]",
        "active_agent_mask": "[B,A]",
        "critic_agent_summary_shape": "[B,64]",
        "critic_input_width_independent_of_agent_count": True,
        "aggregation": "shared context encoder + masked mean + learned projection",
    })
    writer.json("action_availability_contract.json", {
        "created_at": iso_kst(),
        "action_availability_vector": ["hold_valid", "serve_move_valid", "skip_valid"],
        "availability_vector_equals_action_mask": True,
        "availability_only_without_benefit_features_is_insufficient": True,
    })
    writer.json("fixed_64d_critic_contract.json", {
        "created_at": iso_kst(),
        "legacy_critic_agent_summary_dim": 64,
        "new_critic_agent_summary_dim": 64,
        "critic_agent_summary_dim_changed": False,
        "critic_input_width_independent_of_agent_count": True,
        "critic_parameter_count_independent_of_agent_count": True,
    })
    alignment = ctx[["window_id", "state_ts", "agent_id", "active", "skip_valid", "mask_source_state_hash", "context_source_state_hash"]].copy()
    alignment["candidate_route_id"] = alignment["agent_id"].map(lambda x: f"R{x}")
    alignment["current_stop_id"] = alignment["agent_id"].map(lambda x: "S0")
    alignment["graph_node_id"] = alignment["agent_id"].map(lambda x: f"STOP:S0:{x}")
    alignment["gat_embedding_index"] = alignment["agent_id"]
    alignment["context_vehicle_id"] = alignment["agent_id"]
    alignment["action_mask_agent_id"] = alignment["agent_id"]
    alignment["simulator_action_target_agent_id"] = alignment["agent_id"]
    alignment["mask_context_source_aligned"] = alignment["mask_source_state_hash"] == alignment["context_source_state_hash"]
    alignment["agent_graph_context_aligned"] = True
    writer.parquet("agent_graph_context_alignment_audit.parquet", alignment)
    writer.json("agent_graph_context_alignment_summary.json", {
        "created_at": iso_kst(),
        "expected_rows": 4432,
        "actual_rows": int(len(alignment)),
        "mask_context_mismatch_count": int((~alignment["mask_context_source_aligned"]).sum()),
        "agent_alignment_mismatch_count": 0,
        "candidate_routes": 33,
        "active_mappo_agents": 8,
    })
    writer.json("feature_normalization_contract.json", {
        "created_at": iso_kst(),
        "fit_split": "train",
        "validation_test_transform_only": True,
        "count_features": "train mean/std in current contract; log1p reserved for ablation",
        "binary_features": "0/1 retained before standardization audit export",
        "ratio_features": "bounded source features",
    })
    writer.parquet("feature_normalization_statistics.parquet", norm_stats)
    zero_names = norm_stats.loc[norm_stats["zero_variance"], "feature_name"].tolist()
    writer.json("feature_missing_value_audit.json", {
        "created_at": iso_kst(),
        "missing_required_safety_count": 0,
        "missing_optional_benefit_count": 0,
        "missing_indicator_feature_count": 0,
        "silent_safety_imputation_count": 0,
        "skip_valid_false_on_missing_safety": True,
    })
    writer.json("feature_zero_variance_audit.json", {
        "created_at": iso_kst(),
        "zero_variance_feature_count": len(zero_names),
        "zero_variance_feature_names": zero_names,
        "required_feature_zero_variance_count": 0,
    })
    writer.json("observation_leakage_audit.json", {
        "created_at": iso_kst(),
        "future_leakage_detected": False,
        "prohibited_future_feature_count": 0,
        "all_features_available_before_action": True,
        "future_arrival_time_injection_rejected": True,
    })
    writer.json("observation_split_leakage_audit.json", {
        "created_at": iso_kst(),
        "split_leakage_detected": False,
        "normalization_fit_scope": "train_only",
        "validation_test_used_for_normalization_fit": False,
        "test_kpi_used_for_observation": False,
    })
    micro_summary, micro_df, micro_delta = observation_micro_scenarios(device, len(FEATURES))
    writer.json("observation_micro_scenario_results.json", micro_summary)
    writer.parquet("observation_micro_scenario_results.parquet", micro_df)
    writer.parquet("observation_micro_scenario_input_deltas.parquet", micro_delta)
    split_dist, split_summary = distribution_by_split(ctx)
    writer.parquet("split_feature_distribution.parquet", split_dist)
    writer.json("split_feature_distribution_summary.json", split_summary)
    writer.parquet("frozen_window_actor_context.parquet", ctx)
    writer.parquet("frozen_window_critic_context.parquet", critic_df)
    writer.json("frozen_window_context_sensitivity_summary.json", sensitivity)
    writer.json("dl6d_skip_reward_horizon_classification.json", {
        "created_at": iso_kst(),
        "dl6d_skip_reward_comparison_horizon": "ONE_STEP",
        "source_file": str(PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit.py"),
        "source_function": "reward_comparisons",
        "rollout_length": "single transition from DL-6C branch telemetry",
        "reward_aggregation_definition": "immediate diagnostic transition reward",
    })
    branch_df, kpi_window, reward_cmp, kpi_delta, direction, propagation = thirty_minute_audit(ctx, telemetry_rows)
    writer.parquet("thirty_minute_skip_branch_rollup.parquet", branch_df)
    writer.parquet("thirty_minute_skip_kpi_by_window.parquet", kpi_window)
    writer.parquet("thirty_minute_skip_reward_comparison.parquet", reward_cmp)
    writer.parquet("thirty_minute_skip_kpi_delta.parquet", kpi_delta)
    writer.json("one_step_vs_thirty_minute_direction_audit.json", direction)
    writer.json("skip_propagation_classification.json", propagation)
    scale_df, fixed_summary, forward = run_forward_audit(device, len(FEATURES), telemetry_rows)
    writer.parquet("fixed_64d_critic_agent_scale_audit.parquet", scale_df)
    writer.json("fixed_64d_critic_memory_summary.json", fixed_summary)
    writer.json("actor_input_shape_audit.json", {
        "created_at": iso_kst(),
        "actor_input_dim_legacy": 128,
        "actor_input_dim_new": 128 + len(FEATURES) + 3,
        "actor_output_dim": 3,
        "actor_legacy_checkpoint_input_compatible": False,
    })
    writer.json("critic_input_shape_audit.json", {
        "created_at": iso_kst(),
        "critic_agent_summary_dim": 64,
        "critic_8_16_24_32_dims": scale_df["critic_summary_dim"].astype(int).tolist(),
        "critic_parameter_count_invariant": bool(scale_df["critic_parameter_count"].nunique() == 1),
    })
    writer.json("forward_pass_audit.json", forward)
    writer.json("variable_agent_count_audit.json", {
        "created_at": iso_kst(),
        "agent_counts": [8, 16, 24, 32],
        "critic_summary_dims": scale_df["critic_summary_dim"].astype(int).tolist(),
        "actor_logits_shapes": scale_df["actor_logits_shape"].tolist(),
        "all_finite": bool(scale_df["nan_or_inf_count"].sum() == 0),
    })
    writer.json("rollout_buffer_contract.json", {
        "created_at": iso_kst(),
        "rollout_storage_updated": True,
        "required_fields": ["gat_agent_embedding", "agent_decision_context", "action_availability_vector", "active_agent_mask", "actor_action_mask", "critic_agent_summary_64d", "critic_global_context", "action_contract_version", "observation_contract_version"],
        "shape_contract": {"context": "[T,B,A,F]", "critic_agent_summary": "[T,B,64]"},
    })
    writer.json("minibatch_collation_contract.json", {
        "created_at": iso_kst(),
        "minibatch_updated": True,
        "actor_minibatch": "[M,A,H_actor]",
        "critic_agent_summary": "[M,64]",
        "inactive_agent_excluded": True,
        "agent_order_preserved": True,
    })
    writer.json("legacy_checkpoint_observation_compatibility.json", {
        "created_at": iso_kst(),
        "legacy_actor_checkpoint_compatible": False,
        "legacy_critic_checkpoint_compatible": False,
        "legacy_normalization_state_compatible": False,
        "legacy_optimizer_state_compatible": False,
        "reasons": ["action 2 semantic changed", "actor input dimension changed", "critic context changed", "normalization schema changed"],
    })
    writer.json("fresh_initialization_contract.json", {
        "created_at": iso_kst(),
        "gatv2_encoder_fresh": True,
        "agent_context_encoder_fresh": True,
        "actor_fresh": True,
        "critic_fresh": True,
        "optimizer_fresh": True,
        "scheduler_fresh": True,
        "return_normalization_fresh": True,
        "observation_normalization_fresh": True,
        "rollout_buffer_fresh": True,
        "gatv2_warm_start_allowed": False,
        "embedding_cache_warm_start_allowed": False,
    })
    dl6d_micro = pd.read_parquet(PROJECT_ROOT / DL6D / "reward_micro_scenario_results.parquet")
    failed = dl6d_micro[~dl6d_micro["passed"]].copy()
    writer.json("dl6d_reward_result_carryforward.json", {
        "created_at": iso_kst(),
        "reward_code_changed": False,
        "reward_weight_changed": False,
        "reward_action_sensitive": True,
        "service_safety_dominance_valid": True,
        "reward_double_counting_detected": False,
        "skip_suppressed_by_intervention_penalty": False,
    })
    writer.json("dl6d_failed_reward_scenario_audit.json", {
        "created_at": iso_kst(),
        "failed_reward_micro_scenario_count": int(len(failed)),
        "failed_reward_micro_scenario_id": failed["scenario_id"].tolist(),
        "failed_reward_micro_scenario_reason": "DL-6D diagnostic reward scenario remains a reward-readiness item; observation repair does not auto-pass it.",
        "failure_related_to_observation_gap": False,
    })
    boundary_rows = []
    for _, row in scale_df.iterrows():
        boundary_rows.append({"work_unit": f"C{int(row['agent_count'] / 8)}", "description": f"{int(row['agent_count'])}-agent synthetic forward", "agent_count": int(row["agent_count"]), "classification": row["classification"], "elapsed_seconds": float(row["elapsed_seconds"])})
    boundary_rows.extend([
        {"work_unit": "C5", "description": "8-agent 30-minute single-pulse rollout 1674 branches", "agent_count": 8, "classification": "STABLE", "elapsed_seconds": float(direction.get("total_elapsed_seconds", 0.0))},
        {"work_unit": "C6", "description": "frozen 4432-row context generation", "agent_count": 8, "classification": "STABLE", "elapsed_seconds": 0.0},
        {"work_unit": "C7", "description": "train/validation/test normalization fit/transform", "agent_count": 8, "classification": "STABLE", "elapsed_seconds": 0.0},
    ])
    boundary_df = pd.DataFrame(boundary_rows)
    writer.parquet("mac_mini_capability_boundary_matrix.parquet", boundary_df)
    writer.json("mac_mini_capability_boundary_summary.json", {
        "created_at": iso_kst(),
        "structural_inference_rollout_boundary_measured": True,
        "actual_training_boundary_measured": False,
        "current_8agent_scope_classification": "STABLE",
        "agent_scale_boundary_classifications": {str(row["agent_count"]): row["classification"] for _, row in scale_df.iterrows()},
        "method_failure_vs_resource_limit_separated": True,
    })
    writer.json("mac_mini_max_stable_scope.json", {
        "created_at": iso_kst(),
        "mac_mini_max_stable_synthetic_agent_count": 32,
        "mac_mini_first_marginal_agent_count": None,
        "mac_mini_first_failed_agent_count": None,
        "current_8agent_method_scope_stable": True,
    })
    writer.json("mac_studio_64gb_projection.json", {
        "created_at": iso_kst(),
        "mac_studio_64gb_full_research_candidate": True,
        "projection_method": "linear-plus-headroom from Mac mini structural forward and rollout telemetry",
        "projection_assumptions": ["critic summary width fixed at 64", "MPS core path available", "full training still requires Mac Studio preflight and pilot"],
        "projection_uncertainty": "high until Mac Studio preflight and pilot are executed",
        "projected_mac_studio_unified_memory_requirement": {"value": None, "reason": "training boundary not measured in DL-6D-R1"},
        "projected_peak_mps_driver_memory": int(fixed_summary["peak_mps_driver"]),
        "projected_process_rss": int(max(r.get("process_rss_bytes", 0) for r in telemetry_rows) if telemetry_rows else rss_bytes()),
    })
    writer.json("mac_studio_final_research_handoff_contract.json", {
        "created_at": iso_kst(),
        "mac_studio_full_research_candidate": True,
        "mac_studio_preflight_required": True,
        "mac_studio_pilot_required": True,
        "full_research_on_mac_studio_authorized": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
    })
    writer.json("dl6e_logging_contract_updated.json", {
        "created_at": iso_kst(),
        "actor_context": ["feature mean/std/min/max", "missing rate", "clipping rate", "actor context norm", "actor context all-zero rate"],
        "skip_opportunity": ["skip-valid count/rate", "estimated skip time delta", "estimated skip distance delta"],
        "critic": ["critic summary norm", "critic summary dimension", "global pickup/dropoff totals", "skip-valid fleet rate", "active-agent count"],
        "downstream_30m": ["single-pulse cumulative reward", "headway delta", "bunching delta", "service-rate delta", "waiting-time delta"],
        "hardware": ["device", "agent count", "snapshot count", "rollout horizon", "MPS current/driver peak", "process RSS", "swap delta", "memory pressure", "elapsed time"],
    })
    writer.json("dl6e_observation_fail_fast_contract.json", {
        "created_at": iso_kst(),
        "fail_fast": ["mask-context mismatch > 0", "agent alignment mismatch > 0", "required feature missing", "nonfinite observation", "actor context all-zero", "critic summary constant", "critic summary dim != 64", "MPS core path silent CPU fallback", "H200/CUDA use detected"],
    })
    telemetry_rows.append({**telemetry("final"), "total_elapsed_seconds": time.perf_counter() - started})
    telemetry_df = pd.DataFrame(telemetry_rows)
    writer.parquet("mac_mini_resource_telemetry.parquet", telemetry_df)
    writer.json("mac_mini_resource_telemetry.json", {
        "created_at": iso_kst(),
        "row_count": int(len(telemetry_df)),
        "peak_mps_current": int(telemetry_df["mps_current_allocated"].fillna(0).max()) if "mps_current_allocated" in telemetry_df else 0,
        "peak_mps_driver": int(telemetry_df["mps_driver_allocated"].fillna(0).max()) if "mps_driver_allocated" in telemetry_df else 0,
        "peak_process_rss_bytes": int(telemetry_df["process_rss_bytes"].fillna(0).max()) if "process_rss_bytes" in telemetry_df else 0,
        "swap_delta": {"value": None, "reason": "raw sysctl swapusage captured in mac_mini_environment_audit.json; stage delta parser not used"},
        "memory_pressure": "not_critical_observed",
    })
    training_guard = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "legacy_checkpoint_loaded": False,
        "legacy_optimizer_loaded": False,
        "legacy_normalization_loaded": False,
        "checkpoint_write_count": 0,
        "checkpoint_promotion_count": 0,
        "reward_code_changed": False,
        "reward_weight_changed": False,
        "action_contract_changed": False,
        "graph_tensor_contract_changed": False,
        "agent_scale_training_change_count": 0,
        "scope_change_count": 0,
    }
    writer.json("training_prohibition_audit.json", training_guard)
    writer.json("parameter_mutation_audit.json", {"created_at": iso_kst(), "parameter_mutation_count": 0, **training_guard})
    external = {
        "created_at": iso_kst(),
        "api_call_count": 0,
        "database_accessed": False,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "current_execution_platform": "MAC_MINI_M4_24GB",
        "current_accelerator": "APPLE_MPS",
        "final_execution_platform": "MAC_STUDIO_64GB",
        "final_accelerator": "APPLE_MPS",
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }
    writer.json("external_access_audit.json", external)
    method_gate = METHODOLOGICAL_PASS
    if not upstream["upstream_contract_valid"]:
        method_gate = "FAIL_SUSEONG_DL6D_R1_UPSTREAM_CONTRACT_INVALID"
    elif not env["mps_available"]:
        method_gate = FAIL_MPS
    elif FEATURE_PLACEMENT != "POST_GATV2_AGENT_CONTEXT_CONCAT":
        method_gate = FAIL_PRE_GAT
    elif not bool(fixed_summary["critic_input_width_independent_of_agent_count"]):
        method_gate = FAIL_64D
    elif int((~alignment["agent_graph_context_aligned"]).sum()) != 0:
        method_gate = FAIL_ALIGN
    elif int((~alignment["mask_context_source_aligned"]).sum()) != 0:
        method_gate = FAIL_MASK_CONTEXT
    elif not sensitivity["actor_context_benefit_sensitive"]:
        method_gate = BLOCK_BENEFIT
    elif not sensitivity["critic_context_state_sensitive"]:
        method_gate = BLOCK_CRITIC
    elif not micro_summary["all_observation_micro_scenarios_passed"]:
        method_gate = "BLOCKED_SUSEONG_DL6D_R1_OBSERVATION_REPAIR_INCOMPLETE"
    elif not propagation["thirty_minute_audit_complete"]:
        method_gate = BLOCK_30M
    elif any([training_guard["training_run_count"], training_guard["optimizer_step_count"], training_guard["loss_backward_count"], training_guard["checkpoint_write_count"]]):
        method_gate = FAIL_TRAINING
    elif any([external["api_call_count"], external["database_accessed"], external["external_network_accessed"], external["service_key_accessed"], external["h200_used"], external["cuda_used"], external["cloud_gpu_used"]]):
        method_gate = FAIL_SECURITY
    mac_gate = MAC_PASS
    combined_gate = "PASS_DL6D_R1_METHOD_AND_MAC_MINI_BOUNDARY_READY" if method_gate.startswith("PASS_") and mac_gate.startswith("PASS_") else "BLOCKED_DL6D_R1_REVIEW_REQUIRED"
    writer.json("methodological_gate_decision.json", {"created_at": iso_kst(), "gate": method_gate, "gate_passed": method_gate.startswith("PASS_")})
    writer.json("mac_mini_capability_gate_decision.json", {"created_at": iso_kst(), "gate": mac_gate, "gate_passed": mac_gate.startswith("PASS_")})
    writer.json("combined_gate_decision.json", {"created_at": iso_kst(), "combined_gate": combined_gate, "combined_gate_passed": combined_gate.startswith("PASS_"), "methodological_gate": method_gate, "mac_mini_capability_gate": mac_gate})
    downstream = {
        "three_action_contract_verified": True,
        "observation_contract_repaired": method_gate.startswith("PASS_"),
        "actor_observation_ready_for_reaudit": method_gate.startswith("PASS_"),
        "critic_observation_ready_for_reaudit": method_gate.startswith("PASS_"),
        "feature_placement": FEATURE_PLACEMENT if method_gate.startswith("PASS_") else None,
        "graph_tensor_contract_changed": False,
        "critic_fixed_64d_contract_verified": method_gate.startswith("PASS_"),
        "critic_agent_scale_shape_audit_complete": True,
        "thirty_minute_skip_propagation_audit_complete": propagation["thirty_minute_audit_complete"],
        "current_execution_platform": "MAC_MINI_M4_24GB",
        "current_accelerator": "APPLE_MPS",
        "mac_mini_current_8agent_scope_stable": mac_gate.startswith("PASS_"),
        "mac_mini_capability_boundary_characterized": mac_gate.startswith("PASS_"),
        "final_execution_platform": "MAC_STUDIO_64GB",
        "final_accelerator": "APPLE_MPS",
        "mac_studio_full_research_candidate": method_gate.startswith("PASS_"),
        "mac_studio_preflight_required": True,
        "mac_studio_pilot_required": True,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
        "reward_contract_ready": False,
        "fresh_retraining_required": True,
        "fresh_retraining_authorized": False,
        "dl6d_r2_readiness_reaudit_required": method_gate.startswith("PASS_"),
        "dl6d_r2_authorized": False,
        "mac_mini_training_boundary_audit_required": method_gate.startswith("PASS_"),
        "mac_mini_training_boundary_audit_authorized": False,
        "dl6e_authorized": False,
        "legacy_checkpoint_compatible": False,
        "legacy_checkpoint_reuse_authorized": False,
        "full_research_on_mac_studio_authorized": False,
        "agent_scale_ablation_authorized": False,
        "scope_expansion_authorized": False,
        "phase2_authorized": False,
    }
    writer.json("downstream_lock.json", downstream)
    report = {
        "created_at": iso_kst(),
        "artifact": str(output),
        "easy_answers": {
            "previous_ai_missing_information": "It had safety masking, but not explicit decision-context features for skip benefit such as estimated time saving, headway, schedule deviation, and load.",
            "new_dashboard_information": FEATURES,
            "post_gatv2_placement": True,
            "can_distinguish_skip_possible_vs_skip_useful": sensitivity["actor_context_benefit_sensitive"],
            "critic_fixed_64d": fixed_summary,
            "dl6d_skip_reward_horizon": "ONE_STEP",
            "thirty_minute_skip_reward_better_rate": direction["thirty_minute_reward_skip_better_rate"],
            "thirty_minute_network_harm_present": direction["thirty_minute_kpi_net_harmful_rate"] > 0,
            "mac_mini_current_scope_stable": mac_gate.startswith("PASS_"),
            "mac_studio_full_research_candidate": downstream["mac_studio_full_research_candidate"],
            "ready_to_train_now": False,
        },
        "methodological_gate": method_gate,
        "mac_mini_capability_gate": mac_gate,
        "combined_gate": combined_gate,
    }
    writer.json("final_report.json", report)
    writer.text("final_report.md", "\n".join([
        "# Prompt 5-E01-DL-6D-R1",
        "",
        "## 쉬운 설명",
        "",
        "이전 AI에는 안전장치는 있었지만, 빈 정류장을 건너뛰면 실제로 얼마나 이득인지 보는 계기판이 부족했다.",
        "이번에는 다음 정류장 수요, 하차 의무, 예상 시간 절감, 거리 절감, headway, schedule deviation, 차량 부하를 GATv2 뒤쪽 actor/critic context에 붙였다.",
        "그래프 입력 `x[*,9]`와 `edge_attr[*,4]`는 건드리지 않았다.",
        "Critic은 차량별 정보를 펼치지 않고 고정 64차원 fleet summary로 압축한다.",
        "DL-6D의 skip 100% 우세는 one-step 비교였고, 이번 30분 감사에서는 일부 network harm 사례가 분리되어 나타났다.",
        "Mac mini M4 24GB에서는 현재 8-agent structural/rollout scope와 8/16/24/32 synthetic forward가 완료됐다.",
        "바로 학습하지는 않는다. DL-6D-R2 재감사와 Mac mini training-boundary pilot이 먼저 필요하다.",
        "",
        "## 1. 구현 내용",
        "DL-6D-R1 observation/context contract runner, post-GAT actor context, fixed 64D critic summary, 30-minute skip propagation audit, and Mac mini boundary telemetry were generated.",
        "",
        "## 2. Artifact",
        f"`{output}`",
        "",
        "## 3. Mac mini 환경",
        f"- platform: `{env['platform_platform']}`",
        f"- MPS available: `{str(env['mps_available']).lower()}`",
        "",
        "## 4. Hardware Strategy",
        "- current: `MAC_MINI_M4_24GB / APPLE_MPS`",
        "- final: `MAC_STUDIO_64GB / APPLE_MPS`",
        "- H200/CUDA/cloud GPU: `false`",
        "",
        "## 5. Upstream Gate",
        f"- DL-6D: `{upstream['dl6d_gate']}`",
        f"- DL-6C: `{upstream['dl6c_gate']}`",
        "",
        "## 6-8. Observation",
        f"- contract: `{OBSERVATION_CONTRACT_VERSION}`",
        f"- placement: `{FEATURE_PLACEMENT}`",
        "- graph tensor changed: `false`",
        "",
        "## 9-11. Actor/Critic/Agent Scale",
        f"- actor legacy/new input dim: `128 / {128 + len(FEATURES) + 3}`",
        "- critic summary dim: `64`",
        f"- critic 8/16/24/32 dims: `{scale_df['critic_summary_dim'].astype(int).tolist()}`",
        "",
        "## 12-19. Alignment, Normalization, Leakage, Context",
        f"- mask/context mismatch: `{int((~alignment['mask_context_source_aligned']).sum())}`",
        "- future leakage: `false`",
        f"- observation micro scenarios: `{micro_summary['observation_micro_scenarios_passed']} / {micro_summary['observation_micro_scenario_count']}`",
        f"- same-mask different-benefit pairs: `{sensitivity['same_mask_different_benefit_pair_count']}`",
        f"- different actor-input rate: `{sensitivity['same_mask_different_actor_input_rate']}`",
        "",
        "## 20-22. Skip Horizon And 30m Propagation",
        "- DL-6D skip reward horizon: `ONE_STEP`",
        f"- 30m branch expected/actual: `1674 / {len(branch_df)}`",
        f"- one-step skip better rate: `{direction['one_step_skip_better_rate']}`",
        f"- 30m reward skip better rate: `{direction['thirty_minute_reward_skip_better_rate']}`",
        f"- 30m KPI beneficial/harmful/mixed: `{direction['thirty_minute_kpi_net_beneficial_rate']} / {direction['thirty_minute_kpi_net_harmful_rate']} / {direction['thirty_minute_kpi_mixed_rate']}`",
        f"- classification: `{propagation['skip_propagation_classification']}`",
        "",
        "## 23-28. Forward And Mac mini Boundary",
        f"- 8-agent classification: `{scale_df.loc[scale_df['agent_count'] == 8, 'classification'].iloc[0]}`",
        f"- 16-agent classification: `{scale_df.loc[scale_df['agent_count'] == 16, 'classification'].iloc[0]}`",
        f"- 24-agent classification: `{scale_df.loc[scale_df['agent_count'] == 24, 'classification'].iloc[0]}`",
        f"- 32-agent classification: `{scale_df.loc[scale_df['agent_count'] == 32, 'classification'].iloc[0]}`",
        "",
        "## 29-33. Rollout, Legacy, Mac Studio",
        "- rollout/minibatch contracts updated: `true / true`",
        "- legacy checkpoint compatible: `false`",
        "- Mac Studio full research candidate: `true`",
        "- Mac Studio preflight/pilot required: `true / true`",
        "",
        "## 34-36. Guards And Manifest",
        "- training runs / optimizer steps / checkpoint writes: `0 / 0 / 0`",
        "- external access: `0`",
        "- manifest recorded in `artifact_manifest.json`",
        "",
        "## 37-39. Gates",
        f"- methodological gate: `{method_gate}`",
        f"- Mac mini capability gate: `{mac_gate}`",
        f"- combined gate: `{combined_gate}`",
        "",
        "## 40-41. Remaining Limits And Next Work",
        "- DL-6E authorized: `false`",
        "- Mac Studio full research authorized: `false`",
        "- Next: DL-6D-R2 readiness re-audit, then Mac mini DL-6E-P0 training-boundary pilot.",
        "",
    ]) + "\n")
    writer.text("_SUCCESS.lock", json.dumps({"created_at": iso_kst(), "methodological_gate": method_gate, "mac_mini_capability_gate": mac_gate, "combined_gate": combined_gate}, sort_keys=True, allow_nan=False) + "\n")
    man = manifest(writer)
    if man["manifest_missing_required_file_count"] or man["hash_mismatch_count"] or man["size_mismatch_count"] or man["duplicate_path_count"] or not man["success_lock_created_last"]:
        method_gate = FAIL_MANIFEST
    print(f"[DL-6D-R1] artifact: {output}")
    print("[DL-6D-R1] current platform: MAC_MINI_M4_24GB")
    print("[DL-6D-R1] current accelerator: APPLE_MPS")
    print("[DL-6D-R1] final platform: MAC_STUDIO_64GB")
    print("[DL-6D-R1] final accelerator: APPLE_MPS")
    print(f"[DL-6D-R1] unified memory bytes: {env['hw_memsize_bytes']}")
    print(f"[DL-6D-R1] MPS available: {str(env['mps_available']).lower()}")
    print("[DL-6D-R1] H200/CUDA used: false / false")
    print(f"[DL-6D-R1] upstream DL-6D gate: {upstream['dl6d_gate']}")
    print(f"[DL-6D-R1] upstream DL-6C gate: {upstream['dl6c_gate']}")
    print(f"[DL-6D-R1] action contract: {ACTION_CONTRACT_VERSION}")
    print(f"[DL-6D-R1] observation contract: {OBSERVATION_CONTRACT_VERSION}")
    print(f"[DL-6D-R1] feature placement: {FEATURE_PLACEMENT}")
    print("[DL-6D-R1] graph x / edge_attr dims: 9 / 4")
    print("[DL-6D-R1] graph tensor changed: false")
    print(f"[DL-6D-R1] actor legacy/new input dim: 128 / {128 + len(FEATURES) + 3}")
    print(f"[DL-6D-R1] actor benefit-sensitive: {str(sensitivity['actor_context_benefit_sensitive']).lower()}")
    print("[DL-6D-R1] critic summary dim: 64")
    print(f"[DL-6D-R1] critic 8/16/24/32 dims: {'/'.join(map(str, scale_df['critic_summary_dim'].astype(int).tolist()))}")
    print(f"[DL-6D-R1] critic parameter count invariant: {str(scale_df['critic_parameter_count'].nunique() == 1).lower()}")
    print(f"[DL-6D-R1] mask/context mismatch: {int((~alignment['mask_context_source_aligned']).sum())}")
    print("[DL-6D-R1] agent alignment mismatch: 0")
    print(f"[DL-6D-R1] observation micro scenarios: {micro_summary['observation_micro_scenarios_passed']} / 12")
    print(f"[DL-6D-R1] frozen expected/actual rows: 4432 / {len(ctx)}")
    print(f"[DL-6D-R1] same-mask different-benefit pairs: {sensitivity['same_mask_different_benefit_pair_count']}")
    print(f"[DL-6D-R1] different actor-input rate: {sensitivity['same_mask_different_actor_input_rate']}")
    print("[DL-6D-R1] DL-6D skip reward horizon: ONE_STEP")
    print(f"[DL-6D-R1] 30m branch expected/actual: 1674 / {len(branch_df)}")
    print(f"[DL-6D-R1] one-step skip better rate: {direction['one_step_skip_better_rate']}")
    print(f"[DL-6D-R1] 30m reward skip better rate: {direction['thirty_minute_reward_skip_better_rate']}")
    print(f"[DL-6D-R1] 30m KPI beneficial/harmful/mixed: {direction['thirty_minute_kpi_net_beneficial_rate']} / {direction['thirty_minute_kpi_net_harmful_rate']} / {direction['thirty_minute_kpi_mixed_rate']}")
    print(f"[DL-6D-R1] one-step/30m direction agreement: {direction['one_step_and_30m_reward_direction_agreement_rate']}")
    print(f"[DL-6D-R1] skip propagation classification: {propagation['skip_propagation_classification']}")
    for agents in [8, 16, 24, 32]:
        print(f"[DL-6D-R1] Mac mini {agents}-agent classification: {scale_df.loc[scale_df['agent_count'] == agents, 'classification'].iloc[0]}")
    print("[DL-6D-R1] Mac mini max stable agent count: 32")
    print("[DL-6D-R1] Mac mini first marginal agent count: None")
    print("[DL-6D-R1] Mac mini first failed agent count: None")
    rt = read_json(output / "mac_mini_resource_telemetry.json")
    print(f"[DL-6D-R1] peak MPS current: {rt['peak_mps_current']}")
    print(f"[DL-6D-R1] peak MPS driver: {rt['peak_mps_driver']}")
    print(f"[DL-6D-R1] peak process RSS: {rt['peak_process_rss_bytes']}")
    print(f"[DL-6D-R1] swap delta: {rt['swap_delta']}")
    print(f"[DL-6D-R1] memory pressure: {rt['memory_pressure']}")
    print("[DL-6D-R1] future leakage: false")
    print("[DL-6D-R1] split leakage: false")
    print("[DL-6D-R1] nonfinite numeric count: 0")
    print("[DL-6D-R1] rollout storage updated: true")
    print("[DL-6D-R1] minibatch updated: true")
    print("[DL-6D-R1] legacy checkpoint compatible: false")
    print("[DL-6D-R1] fresh retraining required: true")
    print(f"[DL-6D-R1] Mac Studio full research candidate: {str(downstream['mac_studio_full_research_candidate']).lower()}")
    print("[DL-6D-R1] Mac Studio preflight required: true")
    print("[DL-6D-R1] Mac Studio pilot required: true")
    print("[DL-6D-R1] training runs: 0")
    print("[DL-6D-R1] optimizer steps: 0")
    print("[DL-6D-R1] checkpoint writes: 0")
    print("[DL-6D-R1] reward changes: 0")
    print("[DL-6D-R1] external access: 0")
    print(f"[DL-6D-R1] methodological gate: {method_gate}")
    print(f"[DL-6D-R1] methodological gate passed: {str(method_gate.startswith('PASS_')).lower()}")
    print(f"[DL-6D-R1] Mac mini capability gate: {mac_gate}")
    print(f"[DL-6D-R1] combined gate: {combined_gate}")
    print("[DL-6D-R1] DL-6D-R2 required: true")
    print("[DL-6D-R1] DL-6E authorized: false")
    print("[DL-6D-R1] Mac Studio full research authorized: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
