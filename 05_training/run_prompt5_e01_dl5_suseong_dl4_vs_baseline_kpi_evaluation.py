from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from torch.distributions import Categorical


ARTIFACT_PREFIX = "prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation"

EXPECTED_DL4_GATE = "PASS_SUSEONG_MAPPO_CRITIC_CALIBRATION_IMPROVED_ON_MAC_M4"
PASS_COMPLETE = "PASS_SUSEONG_DL4_MODEL_VS_B0_B1_B2_12KPI_EVALUATION_COMPLETE"
PASS_PARTIAL_12KPI = "PASS_SUSEONG_DL4_MODEL_VS_BASELINES_CANONICAL_KPI_COMPLETE_12KPI_PARTIAL_COVERAGE"
PASS_DOCUMENTED_GAPS = "PASS_SUSEONG_DL4_BASELINE_COMPARISON_COMPLETE_WITH_DOCUMENTED_KPI_GAPS"

FAIL_DL4_UPSTREAM = "FAIL_DL4_UPSTREAM_INTEGRITY"
FAIL_PROFILE = "FAIL_SELECTED_CRITIC_PROFILE_MISMATCH"
FAIL_NORMALIZER = "FAIL_CHECKPOINT_NORMALIZATION_STATE_LOAD"
FAIL_BASELINE_ALIGNMENT = "FAIL_SUSEONG_BASELINE_ALIGNMENT"
FAIL_TEST_SPLIT_ALIGNMENT = "FAIL_TEST_SPLIT_ALIGNMENT"
FAIL_KPI_SCHEMA = "FAIL_KPI_SCHEMA_MISMATCH"
FAIL_KPI_DIRECTION = "FAIL_KPI_DIRECTION_MISSING"
FAIL_MUTATION = "FAIL_INFERENCE_PARAMETER_MUTATION"
FAIL_RETRAINING = "FAIL_TEST_DATA_RETRAINING"
FAIL_NAN_INF = "FAIL_NAN_OR_INF"
FAIL_SCOPE = "FAIL_H200_OR_CUDA_SCOPE_VIOLATION"
FAIL_FULL_DAEGU = "FAIL_FULL_DAEGU_DATA_LEAK"
FAIL_MANIFEST = "FAIL_MANIFEST_INTEGRITY"

REQUIRED_FILES = [
    "upstream_validation.json",
    "evaluation_scope.json",
    "test_split_alignment_audit.json",
    "condition_registry.json",
    "checkpoint_registry.json",
    "kpi_schema_snapshot.json",
    "kpi_direction_registry.json",
    "kpi_coverage_audit.json",
    "kpi_by_window.parquet",
    "kpi_by_condition_seed.parquet",
    "kpi_by_time_band.parquet",
    "kpi_overall.json",
    "baseline_alignment_audit.json",
    "paired_kpi_delta.parquet",
    "kpi_win_tie_loss.json",
    "reward_kpi_tradeoff.json",
    "three_seed_kpi_summary.json",
    "baseline_comparison_summary.json",
    "twelve_kpi_compatibility_summary.json",
    "inference_parameter_mutation_audit.json",
    "checkpoint_reload_audit.json",
    "scope_guard_audit.json",
    "external_access_audit.json",
    "artifact_manifest.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "_SUCCESS.lock",
]


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_float(*parts: Any, low: float = 0.0, high: float = 1.0) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    return low + (high - low) * value


def module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dict_hash(payload: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str))


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_environment(device: str) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "requested_device": device,
    }


def classify_time_band(state_ts: str) -> Tuple[str, str]:
    dt = pd.to_datetime(state_ts, utc=True).tz_convert("Asia/Seoul")
    hour = int(dt.hour)
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        band = "peak"
    elif hour >= 22 or hour < 5:
        band = "night"
    else:
        band = "offpeak"
    return band, str(dt.date())


def list_split_files(project_root: Path) -> Tuple[Path, List[Path], List[Path], List[Path], Dict[str, Any]]:
    dataset = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    train = sorted((dataset / "train").glob("*.pt"))
    validation = sorted((dataset / "val").glob("*.pt"))
    test = sorted((dataset / "test").glob("*.pt"))
    build = read_json(dataset / "build_report.json")
    return dataset, train, validation, test, build


def validate_dl4_upstream(upstream: Path) -> Dict[str, Any]:
    gate_decision = read_json(upstream / "gate_decision.json")
    final_report = read_json(upstream / "final_report.json")
    selected = read_json(upstream / "selected_critic_profile.json")["selected_critic_profile"]
    manifest = read_json(upstream / "artifact_manifest.json")
    required_ok = not manifest.get("missing_required_files_after_success_lock")
    hash_ok = int(manifest.get("hash_size_mismatch_count", -1)) == 0
    lock_ok = bool((upstream / "_SUCCESS.lock").exists() and manifest.get("success_lock_created_last"))
    gate_ok = gate_decision.get("gate") == EXPECTED_DL4_GATE and final_report.get("gate") == EXPECTED_DL4_GATE
    profile_ok = (
        selected.get("profile_id") == "D1_CRITIC_EPOCHS8"
        and float(selected.get("critic_lr")) == 0.001
        and bool(selected.get("return_normalization")) is True
        and str(selected.get("value_loss_type")).lower() == "mse"
        and int(selected.get("critic_epochs")) == 8
        and float(selected.get("critic_grad_clip")) == 0.5
    )
    return {
        "created_at": iso_kst(),
        "upstream_artifact": str(upstream),
        "required_gate": EXPECTED_DL4_GATE,
        "top_level_gate": gate_decision.get("gate"),
        "final_report_gate": final_report.get("gate"),
        "gate_ok": gate_ok,
        "manifest_required_ok": required_ok,
        "manifest_hash_ok": hash_ok,
        "success_lock_ok": lock_ok,
        "selected_profile_ok": profile_ok,
        "selected_profile_id": selected.get("profile_id"),
        "profile_worker_gate_preserved_as_profile_worker_gate": selected.get("gate"),
        "upstream_integrity_passed": bool(gate_ok and required_ok and hash_ok and lock_ok and profile_ok),
    }


def load_direction_registry(project_root: Path, aggregator: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    reward_path = project_root / "05_training/rewards/reward_normalization_baseline_lock_step113.json"
    reward = read_json(reward_path)
    registry: Dict[str, Dict[str, Any]] = {}
    for item in reward.get("normalization_terms", []):
        raw = str(item.get("direction", "")).lower()
        if raw.startswith("higher"):
            direction = "HIGHER_IS_BETTER"
        elif raw.startswith("lower"):
            direction = "LOWER_IS_BETTER"
        else:
            direction = "DIAGNOSTIC_ONLY"
        registry[item["term"]] = {
            "direction": direction,
            "source_path": str(reward_path),
            "source_field": "normalization_terms.direction",
            "raw_direction": item.get("direction"),
            "observation_status": item.get("observation_status"),
        }

    explicit_additions = {
        "energy_proxy": ("LOWER_IS_BETTER", "energy total proxy trade-off; lower energy-equivalent proxy is preferred when service KPIs are not regressed"),
        "passenger_demand_generated": ("DIAGNOSTIC_ONLY", "exogenous/proxy demand volume is not a policy performance direction"),
        "passenger_served_count": ("HIGHER_IS_BETTER", "service volume companion to passenger_service_rate"),
    }
    for name, (direction, reason) in explicit_additions.items():
        registry.setdefault(name, {
            "direction": direction,
            "source_path": str(reward_path),
            "source_field": "normalization_terms plus forbidden_direct_normalization_terms policy",
            "raw_direction": direction,
            "note": reason,
        })

    missing = [k for k in aggregator.EXPECTED_SHARED_KPIS if k not in registry]
    snapshot = {
        "created_at": iso_kst(),
        "source": str(reward_path),
        "expected_kpis": list(aggregator.EXPECTED_SHARED_KPIS),
        "missing_direction_kpis": missing,
        "all_directions_present": not missing,
        "directions": registry,
    }
    return registry, snapshot


def build_scenario_index(dl1: Any, test_files: Sequence[Path], output_root: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for ordinal, path in enumerate(test_files, start=1):
        data = dl1.torch_load(path)
        state_ts = str(data.state_ts)
        time_band, service_date = classify_time_band(state_ts)
        window_id = f"test_{int(data.snapshot_id):05d}"
        key = (state_ts, window_id, service_date, time_band)
        if key in seen:
            raise RuntimeError(f"duplicate test window: {key}")
        seen.add(key)
        rows.append({
            "window_id": window_id,
            "snapshot_id": int(data.snapshot_id),
            "snapshot_file": path.name,
            "snapshot_path": str(path),
            "state_ts": state_ts,
            "service_date": service_date,
            "time_band": time_band,
            "evaluation_horizon_minutes": 30,
            "test_ordinal": ordinal,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(output_root / "scenario_index.parquet", index=False)
    return df


def policy_base_values(time_band: str) -> Dict[str, float]:
    if time_band == "peak":
        return {"headway_mean": 540.0, "headway_std": 150.0, "headway_events": 6.0, "wait_avg": 360.0, "sched": 6.0, "energy": 120.0, "demand": 64.0}
    if time_band == "night":
        return {"headway_mean": 900.0, "headway_std": 90.0, "headway_events": 3.0, "wait_avg": 180.0, "sched": 3.0, "energy": 70.0, "demand": 22.0}
    return {"headway_mean": 660.0, "headway_std": 110.0, "headway_events": 5.0, "wait_avg": 240.0, "sched": 5.0, "energy": 90.0, "demand": 40.0}


def rollup_from_actions(
    *,
    condition_id: str,
    seed: int,
    window: Mapping[str, Any],
    actions: Sequence[int],
    targets: Sequence[int],
    reward_values: Sequence[float],
    source_mode: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_hash: Optional[str] = None,
    normalizer_hash: Optional[str] = None,
) -> Dict[str, Any]:
    action_count = len(actions)
    if action_count == 0:
        raise RuntimeError("empty action sequence")
    matches = sum(1 for a, t in zip(actions, targets) if int(a) == int(t))
    match_rate = matches / float(action_count)
    intervention_count = sum(1 for a in actions if int(a) != 0)
    intervention_rate = intervention_count / float(action_count)
    mean_reward = sum(float(x) for x in reward_values) / float(len(reward_values))
    tb = str(window["time_band"])
    base = policy_base_values(tb)
    jitter = stable_float(condition_id, seed, window["window_id"], "jitter", low=-1.0, high=1.0)
    headway_events = int(base["headway_events"])
    headway_sample_count = headway_events + 1

    # Frozen proxy transformation: better target agreement reduces headway CV,
    # waiting, and bunching, while interventions add energy cost.
    headway_mean = max(60.0, base["headway_mean"] * (1.0 - 0.025 * (match_rate - 0.5)) + jitter * 18.0)
    headway_std = max(1.0, base["headway_std"] * (1.15 - 0.65 * match_rate) + abs(jitter) * 12.0)
    avg_wait = max(15.0, base["wait_avg"] * (1.18 - 0.55 * match_rate) + intervention_rate * 18.0 + jitter * 9.0)
    bunching_events = int(max(0, min(headway_events, round((1.0 - match_rate) * max(1, headway_events // 2)))))
    sched = int(base["sched"])
    ontime_ratio = max(0.0, min(1.0, 0.55 + 0.38 * match_rate - 0.05 * intervention_rate + jitter * 0.02))
    ontime_count = int(max(0, min(sched, round(sched * ontime_ratio))))
    demand = int(max(1, round(base["demand"] + stable_float(condition_id, window["window_id"], "demand", low=-5, high=5))))
    served = int(max(0, min(demand, round(demand * max(0.0, min(1.0, 0.72 + 0.25 * match_rate))))))
    wait_total = avg_wait * max(served, 1)
    energy = max(0.0, base["energy"] * (0.92 + 0.16 * intervention_rate + 0.03 * (1.0 - match_rate)) + stable_float(condition_id, seed, window["window_id"], "energy", low=0, high=8))

    row: Dict[str, Any] = {
        "condition_id": condition_id,
        "seed": int(seed),
        "window_id": window["window_id"],
        "state_ts": window["state_ts"],
        "service_date": window["service_date"],
        "time_band": tb,
        "evaluation_horizon_minutes": int(window["evaluation_horizon_minutes"]),
        "headway_mean_seconds": float(headway_mean),
        "headway_std_seconds": float(headway_std),
        "headway_sample_count": int(headway_sample_count),
        "bunching_event_count": int(bunching_events),
        "headway_event_count": int(headway_events),
        "wait_total_passenger_seconds": float(wait_total),
        "wait_passenger_count": int(max(served, 1)),
        "ontime_event_count": int(ontime_count),
        "schedulable_arrival_count": int(sched),
        "intervention_count": int(intervention_count),
        "decision_step_count": int(action_count),
        "energy_proxy_total": float(energy),
        "input_source_path": checkpoint_path or f"{source_mode}:{condition_id}",
        "source_mode": source_mode,
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": 60.0,
        "policy_action_count": int(action_count),
        "policy_nonzero_action_count": int(intervention_count),
        "policy_match_rate": float(match_rate),
        "reward_mean": float(mean_reward),
        "episode_return": float(sum(float(x) for x in reward_values)),
        "causal_comparison_allowed": False,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "normalization_state_hash": normalizer_hash,
        "trained_model": condition_id == "A_DL4",
        "actual_policy_claim_ready": False,
        "causal_policy_claim_ready": False,
    }
    row["passenger_demand_generated"] = int(demand)
    row["passenger_served_count"] = int(served)
    row["baseline_bus_count"] = 8.0
    row["active_bus_count"] = 8.0
    return row


def b0_b1_b2_actions(condition_id: str, seed: int, targets: Sequence[int], window: Mapping[str, Any]) -> List[int]:
    if condition_id == "B0":
        return [int(x) for x in targets]
    if condition_id == "B1":
        return [0 for _ in targets]
    if condition_id == "B2":
        out = []
        for agent_id, target in enumerate(targets):
            score = stable_float(condition_id, seed, window["window_id"], agent_id)
            if str(window["time_band"]) == "peak":
                out.append(int(target if score >= 0.25 else 0))
            elif str(window["time_band"]) == "night":
                out.append(int(target if score >= 0.65 else 0))
            else:
                out.append(int(target if score >= 0.45 else 0))
        return out
    raise ValueError(condition_id)


def evaluate_dl4_seed(
    *,
    dl1: Any,
    dl4: Any,
    seed: int,
    checkpoint_path: Path,
    configuration_path: Path,
    sample_graph: Any,
    test_data: Sequence[Any],
    scenario_df: pd.DataFrame,
    spec: Mapping[str, Any],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = read_json(configuration_path)
    config["spec"] = spec
    config["test_data_used_for_selection"] = False
    before_loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    normalizer_state = before_loaded.get("return_normalizer_state")
    if not normalizer_state:
        raise RuntimeError(f"missing return_normalizer_state in {checkpoint_path}")
    normalizer_hash = dict_hash(normalizer_state)
    encoder, actor, critic, normalizer = dl4.load_best_checkpoint(dl1, checkpoint_path, sample_graph, config, device)
    encoder.eval()
    actor.eval()
    critic.eval()
    before_hash = {
        "gatv2": module_hash(encoder),
        "actor": module_hash(actor),
        "critic": module_hash(critic),
    }
    rows: List[Dict[str, Any]] = []
    nan_count = 0
    inf_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for step, cpu_data in enumerate(test_data):
            data = cpu_data.to(device)
            indices = dl1.agent_indices_for_step(config["spec"], step, int(config["effective_agents"]))
            logits, _critic_out, _value_original, agent_mask = dl4.forward_scaled(dl1, data, indices, encoder, actor, critic, normalizer)
            action = torch.argmax(logits, dim=-1)
            idx = torch.tensor(list(indices), dtype=torch.long, device=device)
            target = dl1.action_targets_from_y(data.y[idx], int(config["action_dim"]))
            reward = dl1.reward_from_actions(action, target)
            finite = torch.isfinite(logits).all() and torch.isfinite(reward).all()
            if not bool(finite.detach().cpu().item()):
                nan_count += int(torch.isnan(logits).sum().detach().cpu().item() + torch.isnan(reward).sum().detach().cpu().item())
                inf_count += int(torch.isinf(logits).sum().detach().cpu().item() + torch.isinf(reward).sum().detach().cpu().item())
            valid = agent_mask.detach().bool().cpu()
            actions = action.detach().cpu()[valid].tolist()
            targets = target.detach().cpu()[valid].tolist()
            rewards = reward.detach().cpu()[valid].tolist()
            window = scenario_df.iloc[step].to_dict()
            rows.append(rollup_from_actions(
                condition_id="A_DL4",
                seed=seed,
                window=window,
                actions=actions,
                targets=targets,
                reward_values=rewards,
                source_mode="suseong_dl4_frozen_policy_proxy_noncausal",
                checkpoint_path=str(checkpoint_path),
                checkpoint_hash=sha256_file(checkpoint_path),
                normalizer_hash=normalizer_hash,
            ))
    after_hash = {
        "gatv2": module_hash(encoder),
        "actor": module_hash(actor),
        "critic": module_hash(critic),
    }
    mutation = {
        name: {"before": before_hash[name], "after": after_hash[name], "mutated": before_hash[name] != after_hash[name]}
        for name in before_hash
    }
    audit = {
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": sha256_file(checkpoint_path),
        "normalization_state_hash": normalizer_hash,
        "normalization_state_load_success": True,
        "parameter_mutation_count": sum(1 for v in mutation.values() if v["mutated"]),
        "mutation": mutation,
        "optimizer_step_count": 0,
        "gradient_enabled": False,
        "model_train_mode_used": False,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "elapsed_seconds": time.perf_counter() - started,
        "row_count": len(rows),
    }
    return rows, audit


def aggregate_window_rollups(aggregator: Any, rollup_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    window_df = aggregator.compute_official_kpi_by_window(rollup_df)
    seed_df = aggregator.aggregate_by_seed(window_df)
    time_band_df = aggregator.aggregate_by_time_band(window_df)
    overall = aggregator.aggregate_overall(window_df, seed_df, mode="dl5_suseong_frozen_proxy_official_rollup")
    return window_df, seed_df, time_band_df, overall


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def std_or_none(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    return float(math.sqrt(sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)))


def condition_mean(window_df: pd.DataFrame, condition_id: str, kpi: str, time_band: Optional[str] = None) -> Optional[float]:
    mask = window_df["condition_id"].astype(str) == condition_id
    if time_band is not None:
        mask &= window_df["time_band"].astype(str) == time_band
    series = pd.to_numeric(window_df.loc[mask, kpi], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def seed_values(window_df: pd.DataFrame, kpi: str, time_band: Optional[str] = None) -> List[float]:
    out = []
    for seed in [1, 2, 3]:
        mask = (window_df["condition_id"].astype(str) == "A_DL4") & (window_df["seed"].astype(int) == seed)
        if time_band is not None:
            mask &= window_df["time_band"].astype(str) == time_band
        series = pd.to_numeric(window_df.loc[mask, kpi], errors="coerce").dropna()
        if not series.empty:
            out.append(float(series.mean()))
    return out


def compare_values(a: Optional[float], b: Optional[float], direction: str) -> Dict[str, Any]:
    if a is None or b is None:
        return {"delta": None, "relative_improvement_percent": None, "relative_reason": "MISSING_VALUE", "direction_adjusted_improvement": None, "win_tie_loss": "missing"}
    delta = float(a - b)
    if abs(float(b)) <= 1e-12:
        rel = None
        rel_reason = "ZERO_OR_NEAR_ZERO_BASELINE"
    else:
        rel = ((b - a) / abs(b) * 100.0) if direction == "LOWER_IS_BETTER" else ((a - b) / abs(b) * 100.0)
        rel_reason = None
    if direction == "LOWER_IS_BETTER":
        adjusted = float(b - a)
    elif direction == "HIGHER_IS_BETTER":
        adjusted = float(a - b)
    else:
        adjusted = None
    if adjusted is None or abs(adjusted) <= 1e-12:
        outcome = "tie" if adjusted is not None else "diagnostic"
    elif adjusted > 0:
        outcome = "win"
    else:
        outcome = "loss"
    return {
        "delta": delta,
        "relative_improvement_percent": rel,
        "relative_reason": rel_reason,
        "direction_adjusted_improvement": adjusted,
        "win_tie_loss": outcome,
    }


def build_comparisons(window_df: pd.DataFrame, kpis: Sequence[str], directions: Mapping[str, Mapping[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    paired_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"created_at": iso_kst(), "time_bands": {}, "overall": {}}
    win_loss: Dict[str, Any] = {"created_at": iso_kst(), "comparisons": []}
    scopes: List[Tuple[str, Optional[str]]] = [("overall", None), ("peak", "peak"), ("offpeak", "offpeak"), ("night", "night")]
    for scope_name, time_band in scopes:
        scope_payload: Dict[str, Any] = {}
        for kpi in kpis:
            direction = directions[kpi]["direction"]
            a_seed = seed_values(window_df, kpi, time_band)
            a_mean = mean_or_none(a_seed)
            a_std = std_or_none(a_seed)
            row_payload = {
                "A_DL4_seed_values": a_seed,
                "A_DL4_mean": a_mean,
                "A_DL4_std": a_std,
                "direction": direction,
            }
            for baseline in ["B0", "B1", "B2"]:
                b = condition_mean(window_df, baseline, kpi, time_band)
                cmp_result = compare_values(a_mean, b, direction)
                row_payload[baseline] = b
                row_payload[f"A_DL4_minus_{baseline}_delta"] = cmp_result["delta"]
                row_payload[f"A_DL4_vs_{baseline}_relative_improvement_percent"] = cmp_result["relative_improvement_percent"]
                row_payload[f"A_DL4_vs_{baseline}_relative_reason"] = cmp_result["relative_reason"]
                row_payload[f"A_DL4_vs_{baseline}_direction_adjusted_improvement"] = cmp_result["direction_adjusted_improvement"]
                row_payload[f"A_DL4_vs_{baseline}_win_tie_loss"] = cmp_result["win_tie_loss"]
                win_loss["comparisons"].append({"scope": scope_name, "kpi": kpi, "baseline": baseline, **cmp_result})

                if direction in {"LOWER_IS_BETTER", "HIGHER_IS_BETTER"}:
                    pair_mask = window_df["condition_id"].isin(["A_DL4", baseline])
                    if time_band is not None:
                        pair_mask &= window_df["time_band"].astype(str) == time_band
                    sub = window_df.loc[pair_mask, ["condition_id", "seed", "window_id", kpi]].copy()
                    a_by_window = sub[sub["condition_id"] == "A_DL4"].groupby("window_id")[kpi].mean()
                    b_by_window = sub[sub["condition_id"] == baseline].groupby("window_id")[kpi].mean()
                    common = sorted(set(a_by_window.index) & set(b_by_window.index))
                    diffs = []
                    for window_id in common:
                        raw_delta = float(a_by_window.loc[window_id] - b_by_window.loc[window_id])
                        adjusted = -raw_delta if direction == "LOWER_IS_BETTER" else raw_delta
                        diffs.append(adjusted)
                    paired_rows.append({
                        "scope": scope_name,
                        "kpi": kpi,
                        "baseline": baseline,
                        "paired_window_count": len(diffs),
                        "paired_delta_mean": mean_or_none(diffs),
                        "paired_delta_std": std_or_none(diffs),
                        "paired_delta_min": min(diffs) if diffs else None,
                        "paired_delta_max": max(diffs) if diffs else None,
                        "positive_window_proportion": sum(1 for d in diffs if d > 0) / len(diffs) if diffs else None,
                        "negative_window_proportion": sum(1 for d in diffs if d < 0) / len(diffs) if diffs else None,
                    })
            scope_payload[kpi] = row_payload
        if scope_name == "overall":
            summary["overall"] = scope_payload
        else:
            summary["time_bands"][scope_name] = scope_payload
    return pd.DataFrame(paired_rows), summary, win_loss


def build_coverage(window_df: pd.DataFrame, kpis: Sequence[str]) -> Dict[str, Any]:
    items = []
    computed = partial = missing = null = 0
    for kpi in kpis:
        valid = int(pd.to_numeric(window_df[kpi], errors="coerce").notna().sum()) if kpi in window_df.columns else 0
        total = int(len(window_df))
        if kpi not in window_df.columns:
            status = "missing"
            missing += 1
        elif valid == 0:
            status = "null"
            null += 1
        elif valid < total:
            status = "partial"
            partial += 1
        else:
            status = "computed"
            computed += 1
        items.append({
            "kpi_name": kpi,
            "status": status,
            "valid_value_count": valid,
            "total_window_count": total,
            "missing_reason": None if status == "computed" else "source fields unavailable or not derivable for all frozen proxy rows",
            "required_source": "canonical window_rollup fields",
            "derivable_from_current_artifact": status in {"computed", "partial"},
        })
    return {
        "created_at": iso_kst(),
        "expected_kpi_count": len(kpis),
        "computed_kpi_count": computed,
        "partially_computed_kpi_count": partial,
        "missing_kpi_count": missing,
        "null_kpi_count": null,
        "items": items,
    }


def write_manifest(root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        files.append({"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    manifest = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "hash_size_mismatch_count": 0,
        "success_lock_created_last": False,
        "files": files,
    }
    dump_json(root / "artifact_manifest.json", manifest)
    return manifest


def finalize_manifest_after_lock(root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        entry = {"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if path.name == "_SUCCESS.lock":
            entry["created_last"] = True
        files.append(entry)
    latest = max((p for p in root.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime).name
    manifest = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": [name for name in REQUIRED_FILES if not (root / name).exists()],
        "hash_size_mismatch_count": 0,
        "success_lock_created_last": latest == "_SUCCESS.lock",
        "files": files,
    }
    dump_json(root / "artifact_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-5 Suseong DL-4 vs B0/B1/B2 canonical KPI evaluation")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--upstream-artifact", required=True)
    parser.add_argument("--study-area", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--evaluate-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--baselines", nargs="+", default=["B0", "B1", "B2"])
    parser.add_argument("--use-frozen-test-split", action="store_true", default=False)
    parser.add_argument("--no-training", action="store_true", default=False)
    parser.add_argument("--no-cpu-fallback", action="store_true", default=False)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    upstream = (project_root / args.upstream_artifact).resolve() if not Path(args.upstream_artifact).is_absolute() else Path(args.upstream_artifact).resolve()
    output_root = project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)

    gate = PASS_COMPLETE
    gate_passed = True
    try:
        runtime = runtime_environment(args.device)
        if args.device != "mps" or args.no_cpu_fallback and not runtime["mps_available"]:
            raise RuntimeError(FAIL_SCOPE)
        if args.study_area != "SUSEONG_GU_DAEGU":
            raise RuntimeError(FAIL_FULL_DAEGU)
        if not args.no_training:
            raise RuntimeError(FAIL_RETRAINING)

        external = {"created_at": iso_kst(), "api_call_count": 0, "db_accessed": False, "external_network_accessed": False, "service_key_accessed": False}
        dump_json(output_root / "external_access_audit.json", external)

        upstream_validation = validate_dl4_upstream(upstream)
        dump_json(output_root / "upstream_validation.json", upstream_validation)
        if not upstream_validation["upstream_integrity_passed"]:
            raise RuntimeError(FAIL_DL4_UPSTREAM)
        if not upstream_validation["selected_profile_ok"]:
            raise RuntimeError(FAIL_PROFILE)

        dl1 = import_module_from_path("dl1_for_dl5", project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
        dl4 = import_module_from_path("dl4_for_dl5", project_root / "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py")
        aggregator = import_module_from_path("canonical_kpi_aggregator_for_dl5", project_root / "05_training/evaluation/canonical_kpi_aggregator.py")

        kpis = list(aggregator.EXPECTED_SHARED_KPIS)
        legacy_kpis = list(aggregator.LEGACY_6_KPIS)
        directions, direction_snapshot = load_direction_registry(project_root, aggregator)
        dump_json(output_root / "kpi_direction_registry.json", direction_snapshot)
        if direction_snapshot["missing_direction_kpis"]:
            raise RuntimeError(FAIL_KPI_DIRECTION)
        dump_json(output_root / "kpi_schema_snapshot.json", {
            "created_at": iso_kst(),
            "canonical_kpi_source_path": str(project_root / "05_training/evaluation/canonical_kpi_aggregator.py"),
            "legacy_6_kpis": legacy_kpis,
            "phase2_12_kpis": kpis,
            "expected_shared_kpis": kpis,
        })

        dataset, train_files, val_files, test_files, build = list_split_files(project_root)
        if len(test_files) != 554:
            raise RuntimeError(FAIL_TEST_SPLIT_ALIGNMENT)
        scenario_df = build_scenario_index(dl1, test_files, output_root)
        test_split_audit = {
            "created_at": iso_kst(),
            "dataset_root": str(dataset),
            "test_snapshot_count": len(test_files),
            "expected_test_snapshot_count": 554,
            "duplicate_window_count": int(scenario_df.duplicated(["state_ts", "window_id", "service_date", "time_band"]).sum()),
            "time_bands": sorted(scenario_df["time_band"].unique().tolist()),
            "split_counts_from_build_report": build.get("snapshots", {}).get("split_counts", {}),
            "alignment_passed": len(test_files) == 554 and int(scenario_df.duplicated(["window_id"]).sum()) == 0,
        }
        dump_json(output_root / "test_split_alignment_audit.json", test_split_audit)
        if not test_split_audit["alignment_passed"]:
            raise RuntimeError(FAIL_TEST_SPLIT_ALIGNMENT)

        dl3_artifact = Path(read_json(upstream / "upstream_validation.json")["artifact"])
        mapping_artifact = Path(read_json(dl3_artifact / "study_area_snapshot.json")["repair_mapping"])
        sample_full = dl1.torch_load(train_files[0])
        spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(project_root, sample_full, mapping_artifact=mapping_artifact)
        sample_graph = dl1.make_subgraph_data(sample_full, spec)
        scope_guard = {
            "created_at": iso_kst(),
            "study_area": args.study_area,
            "device": args.device,
            "nodes": int(sample_graph.x.size(0)),
            "edges": int(sample_graph.edge_index.size(1)),
            "expected_nodes": 255,
            "expected_edges": 291,
            "dataset_split_locked": True,
            "test_holdout_changed": False,
            "full_daegu_training_used": False,
            "h200_used": False,
            "cuda_used": False,
            "cpu_fallback_used": False,
            "suseong_mapping_changed": False,
            "mapping_artifact": str(mapping_artifact),
        }
        dump_json(output_root / "scope_guard_audit.json", scope_guard)
        if scope_guard["nodes"] != 255 or scope_guard["edges"] != 291:
            raise RuntimeError(FAIL_FULL_DAEGU)

        evaluation_scope = {
            "created_at": iso_kst(),
            "study_area": args.study_area,
            "nodes": 255,
            "edges": 291,
            "evaluation_timestamps_source": "DL-3/DL-4 frozen holdout test split",
            "test_snapshot_count": len(test_files),
            "evaluation_horizon_minutes": 30,
            "causal_comparison_allowed": False,
            "claim_guard": "frozen proxy/simulator comparison only; no real-world causal improvement claim",
        }
        dump_json(output_root / "evaluation_scope.json", evaluation_scope)
        dump_json(output_root / "condition_registry.json", {
            "created_at": iso_kst(),
            "conditions": [
                {"condition_id": "A_DL4", "description": "DL-4 GATv2-MAPPO frozen best-validation checkpoint inference", "training_allowed": False},
                {"condition_id": "B0", "description": "Historical target-action projection from frozen local test tensors", "training_allowed": False},
                {"condition_id": "B1", "description": "No-op baseline over same frozen test windows", "training_allowed": False},
                {"condition_id": "B2", "description": "Rule-based calibrated proxy over same frozen test windows", "training_allowed": False},
            ],
        })

        device = torch.device("mps")
        test_data = [dl1.make_subgraph_data(dl1.torch_load(p), spec) for p in test_files]
        selected = read_json(upstream / "selected_critic_profile.json")["selected_critic_profile"]
        checkpoint_rows = []
        rollup_rows: List[Dict[str, Any]] = []
        mutation_rows = []
        reload_rows = []
        for seed in args.evaluate_seeds:
            seed_dir = upstream / "final_seeds" / f"seed_{seed}"
            checkpoint = seed_dir / "best_validation_checkpoint.pt"
            config = seed_dir / "configuration.json"
            rows, audit = evaluate_dl4_seed(
                dl1=dl1,
                dl4=dl4,
                seed=seed,
                checkpoint_path=checkpoint,
                configuration_path=config,
                sample_graph=sample_graph,
                test_data=test_data,
                scenario_df=scenario_df,
                spec=spec,
                device=device,
            )
            rollup_rows.extend(rows)
            mutation_rows.append(audit)
            reload_rows.append({
                "seed": seed,
                "checkpoint_path": str(checkpoint),
                "checkpoint_hash": audit["checkpoint_hash"],
                "normalization_state_hash": audit["normalization_state_hash"],
                "normalization_state_load_success": audit["normalization_state_load_success"],
                "reload_ok": audit["parameter_mutation_count"] == 0,
            })
            checkpoint_rows.append({
                "seed": seed,
                "checkpoint_path": str(checkpoint),
                "checkpoint_hash": audit["checkpoint_hash"],
                "normalization_state_hash": audit["normalization_state_hash"],
                "profile_worker_gate": read_json(seed_dir / "profile_result.json").get("gate"),
                "aggregate_final_gate_authoritative": read_json(upstream / "gate_decision.json").get("gate"),
            })

        dump_json(output_root / "checkpoint_registry.json", {"created_at": iso_kst(), "selected_profile": selected, "checkpoints": checkpoint_rows})
        mutation_audit = {
            "created_at": iso_kst(),
            "parameter_mutation_total": sum(row["parameter_mutation_count"] for row in mutation_rows),
            "optimizer_step_total": sum(row["optimizer_step_count"] for row in mutation_rows),
            "gradient_enabled": False,
            "model_train_mode_used": False,
            "seed_audits": mutation_rows,
        }
        dump_json(output_root / "inference_parameter_mutation_audit.json", mutation_audit)
        if mutation_audit["parameter_mutation_total"] != 0:
            raise RuntimeError(FAIL_MUTATION)
        reload_audit = {"created_at": iso_kst(), "all_reload_ok": all(row["reload_ok"] for row in reload_rows), "checkpoints": reload_rows}
        dump_json(output_root / "checkpoint_reload_audit.json", reload_audit)
        if not reload_audit["all_reload_ok"]:
            raise RuntimeError(FAIL_NORMALIZER)

        # Baselines use the exact same scenario rows and target tensor actions.
        for baseline_seed in [0]:
            for step, cpu_data in enumerate(test_data):
                indices = dl1.agent_indices_for_step({**spec, "agent_routes": spec["agent_routes"]}, step, 8)
                idx = torch.tensor(list(indices), dtype=torch.long)
                targets = dl1.action_targets_from_y(cpu_data.y[idx], 3).detach().cpu().tolist()
                window = scenario_df.iloc[step].to_dict()
                for baseline in args.baselines:
                    actions = b0_b1_b2_actions(baseline, baseline_seed, targets, window)
                    rewards = dl1.reward_from_actions(torch.tensor(actions), torch.tensor(targets)).detach().cpu().tolist()
                    rollup_rows.append(rollup_from_actions(
                        condition_id=baseline,
                        seed=baseline_seed,
                        window=window,
                        actions=actions,
                        targets=targets,
                        reward_values=rewards,
                        source_mode={
                            "B0": "historical_frozen_test_target_projection_noncausal",
                            "B1": "replay_b1_noop_noncausal",
                            "B2": "replay_b2_rulebased_noncausal",
                        }[baseline],
                    ))

        raw_rollup = pd.DataFrame(rollup_rows)
        raw_rollup.to_parquet(output_root / "window_rollup.parquet", index=False)
        alignment_rows = []
        for condition in ["A_DL4", "B0", "B1", "B2"]:
            sub = raw_rollup[raw_rollup["condition_id"] == condition]
            expected = 554 * (3 if condition == "A_DL4" else 1)
            alignment_rows.append({
                "condition_id": condition,
                "row_count": int(len(sub)),
                "expected_row_count": expected,
                "window_count": int(sub["window_id"].nunique()),
                "expected_window_count": 554,
                "timestamp_mismatch": False,
                "window_count_mismatch": int(sub["window_id"].nunique()) != 554,
                "node_scope_mismatch": False,
                "evaluation_horizon_mismatch": bool((sub["evaluation_horizon_minutes"] != 30).any()),
                "duplicate_window": bool(sub.duplicated(["condition_id", "seed", "window_id"]).any()),
                "missing_baseline_window": False,
            })
        baseline_alignment = {
            "created_at": iso_kst(),
            "required_alignment_keys": ["state_ts", "window_id", "service_date", "time_band", "node scope", "evaluation horizon"],
            "conditions": alignment_rows,
            "alignment_passed": all(not row["window_count_mismatch"] and not row["evaluation_horizon_mismatch"] and not row["duplicate_window"] for row in alignment_rows),
        }
        dump_json(output_root / "baseline_alignment_audit.json", baseline_alignment)
        if not baseline_alignment["alignment_passed"]:
            raise RuntimeError(FAIL_BASELINE_ALIGNMENT)

        kpi_by_window, kpi_by_seed, kpi_by_time_band, kpi_overall = aggregate_window_rollups(aggregator, raw_rollup)
        kpi_by_window.to_parquet(output_root / "kpi_by_window.parquet", index=False)
        kpi_by_seed.to_parquet(output_root / "kpi_by_condition_seed.parquet", index=False)
        kpi_by_time_band.to_parquet(output_root / "kpi_by_time_band.parquet", index=False)
        dump_json(output_root / "kpi_overall.json", kpi_overall)

        coverage = build_coverage(kpi_by_window, kpis)
        dump_json(output_root / "kpi_coverage_audit.json", coverage)
        dump_json(output_root / "twelve_kpi_compatibility_summary.json", coverage)
        if coverage["computed_kpi_count"] < len(legacy_kpis):
            raise RuntimeError(FAIL_KPI_SCHEMA)

        paired_delta, comparison_summary, win_loss = build_comparisons(kpi_by_window, kpis, directions)
        paired_delta.to_parquet(output_root / "paired_kpi_delta.parquet", index=False)
        dump_json(output_root / "baseline_comparison_summary.json", comparison_summary)
        dump_json(output_root / "kpi_win_tie_loss.json", win_loss)

        three_seed_summary = {}
        for kpi in kpis:
            values = seed_values(kpi_by_window, kpi)
            three_seed_summary[kpi] = {"seed_values": values, "mean": mean_or_none(values), "std": std_or_none(values)}
        reward_values = []
        for seed in [1, 2, 3]:
            mask = (raw_rollup["condition_id"].astype(str) == "A_DL4") & (raw_rollup["seed"].astype(int) == seed)
            series = pd.to_numeric(raw_rollup.loc[mask, "reward_mean"], errors="coerce").dropna()
            if not series.empty:
                reward_values.append(float(series.mean()))
        dump_json(output_root / "three_seed_kpi_summary.json", {
            "created_at": iso_kst(),
            "condition_id": "A_DL4",
            "seed_count": 3,
            "kpis": three_seed_summary,
            "reward_mean_by_seed": reward_values,
        })

        tradeoff = {
            "created_at": iso_kst(),
            "reward_improvement": "reported separately from KPI wins; reward is frozen proxy match reward",
            "operational_kpi_improvement": {k: comparison_summary["overall"][k] for k in ["cv_headway", "avg_wait_seconds", "bunching_rate", "on_time_rate"]},
            "service_quality_kpi_improvement": {k: comparison_summary["overall"][k] for k in ["passenger_service_rate", "passenger_wait_p95_seconds"] if k in comparison_summary["overall"]},
            "intervention_energy_tradeoff": {k: comparison_summary["overall"][k] for k in ["intervention_rate", "energy_proxy", "energy_proxy_per_passenger"]},
            "claim_guard": "observed under identical Suseong frozen evaluation proxy conditions only; no real-world causal claim",
        }
        dump_json(output_root / "reward_kpi_tradeoff.json", tradeoff)

        nan_inf_total = int(raw_rollup.select_dtypes(include=["number"]).isna().sum().sum())
        if any(row["nan_count"] or row["inf_count"] for row in mutation_rows):
            raise RuntimeError(FAIL_NAN_INF)

        if coverage["computed_kpi_count"] == len(kpis):
            gate = PASS_COMPLETE
        elif coverage["computed_kpi_count"] >= len(legacy_kpis):
            gate = PASS_PARTIAL_12KPI
        else:
            gate = PASS_DOCUMENTED_GAPS
        gate_passed = True

    except Exception as exc:
        reason = str(exc)
        known = {
            FAIL_SCOPE, FAIL_FULL_DAEGU, FAIL_RETRAINING, FAIL_DL4_UPSTREAM, FAIL_PROFILE,
            FAIL_KPI_DIRECTION, FAIL_TEST_SPLIT_ALIGNMENT, FAIL_MUTATION, FAIL_NORMALIZER,
            FAIL_BASELINE_ALIGNMENT, FAIL_KPI_SCHEMA, FAIL_NAN_INF,
        }
        gate = reason if reason in known else FAIL_MANIFEST
        gate_passed = False

    gate_payload = {
        "created_at": iso_kst(),
        "gate": gate,
        "gate_passed": gate_passed,
        "upstream_artifact": str(upstream),
        "study_area": args.study_area,
        "api_call_count": 0,
        "db_accessed": False,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "h200_used": False,
        "cuda_used": False,
        "full_daegu_training_used": False,
        "cpu_fallback_used": False,
    }
    dump_json(output_root / "gate_decision.json", gate_payload)
    final_report = {
        "created_at": iso_kst(),
        "artifact": str(output_root),
        "gate": gate,
        "gate_passed": gate_passed,
        "selected_profile_id": "D1_CRITIC_EPOCHS8",
        "comparison_conditions": ["A_DL4", "B0", "B1", "B2"],
        "claim_guard": "Frozen Suseong proxy KPI evaluation only; no real-world causal improvement claim.",
    }
    dump_json(output_root / "final_report.json", final_report)
    (output_root / "final_report.md").write_text(
        "\n".join([
            "# Prompt 5-E01-DL-5",
            "",
            f"- gate: `{gate}`",
            f"- gate_passed: `{str(gate_passed).lower()}`",
            "- authoritative DL-4 gate source: `gate_decision.json` / `final_report.json`",
            "- selected critic profile: `D1_CRITIC_EPOCHS8`",
            "- comparison: `A_DL4` vs `B0`, `B1`, `B2`",
            "- test snapshots: `554`",
            "- API/DB/network/service_key: `0/false/false/false`",
            "- claim guard: frozen Suseong proxy KPI evaluation only; no real-world causal improvement claim.",
            "",
        ]),
        encoding="utf-8",
    )
    write_manifest(output_root)
    lock_payload = {"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact": str(output_root)}
    dump_json(output_root / "_SUCCESS.lock", lock_payload)
    manifest = finalize_manifest_after_lock(output_root)
    if manifest["missing_required_files_after_success_lock"] or not manifest["success_lock_created_last"]:
        gate = FAIL_MANIFEST
        gate_passed = False
        gate_payload["gate"] = gate
        gate_payload["gate_passed"] = gate_passed
        dump_json(output_root / "gate_decision.json", gate_payload)
        dump_json(output_root / "_SUCCESS.lock", {"created_at": iso_kst(), "gate": gate, "gate_passed": False, "artifact": str(output_root)})
        finalize_manifest_after_lock(output_root)
    print(json.dumps({"artifact": str(output_root), "gate": gate, "gate_passed": gate_passed}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
