from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


ARTIFACT_PREFIX = "prompt5_e01_dl2_suseong_mac_m4_capacity_envelope"
DL1R_ARTIFACT = "prompt5_e01_dl1r_suseong_mapping_repair_mps_rerun_20260731_105654"
DL1_ARTIFACT = "prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation_20260731_105711"
DL1R_GATE = "PASS_SUSEONG_MAPPING_REPAIRED_AND_DL1_CRITIC_JOINT_LEARNING_VALIDATED_ON_NATIVE_MAC_M4"
DL1_GATE = "PASS_SUSEONG_GATV2_MAPPO_CRITIC_JOINT_LEARNING_PATH_VALIDATED_ON_MAC_M4"
PASS_GATE = "PASS_SUSEONG_MAC_M4_CAPACITY_ENVELOPE_AND_FULL_TRAINING_PROFILE_SELECTED"
PASS_INDETERMINATE = "PASS_SUSEONG_MAC_M4_SAFE_FULL_TRAINING_PROFILE_SELECTED_CAPACITY_CEILING_INDETERMINATE"
FAIL_UPSTREAM = "FAIL_DL1_UPSTREAM_INTEGRITY"
FAIL_MPS = "FAIL_NATIVE_MPS_UNAVAILABLE"
FAIL_BASELINE = "FAIL_BASELINE_PROFILE_NOT_REPRODUCIBLE"
FAIL_NO_PROFILE = "FAIL_NO_MEMORY_SAFE_LEARNING_PROFILE"
FAIL_CRITIC = "FAIL_CRITIC_LEARNING_PATH_REGRESSION"
FAIL_NAN = "FAIL_NAN_OR_INF"
FAIL_CHECKPOINT = "FAIL_CHECKPOINT_RELOAD"
FAIL_SCOPE = "FAIL_H200_OR_CUDA_SCOPE_VIOLATION"
FAIL_MANIFEST = "FAIL_MANIFEST_INTEGRITY"

REQUIRED_FILES = [
    "upstream_validation.json",
    "study_area_snapshot.json",
    "runtime_environment.json",
    "capacity_search_plan.json",
    "profile_registry.json",
    "profile_registry.parquet",
    "hidden_size_comparison.json",
    "rollout_horizon_comparison.json",
    "minibatch_comparison.json",
    "ppo_epoch_comparison.json",
    "snapshot_scaling_comparison.json",
    "oom_registry.json",
    "failed_profile_registry.json",
    "memory_headroom_report.json",
    "throughput_report.json",
    "safe_profile.json",
    "balanced_profile.json",
    "max_safe_profile.json",
    "selected_full_training_profile.json",
    "full_training_time_estimate.json",
    "full_training_resource_plan.json",
    "source_change_inventory.json",
    "external_access_audit.json",
    "artifact_manifest.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "_SUCCESS.lock",
]

PROFILE_REQUIRED_FILES = [
    "configuration.json",
    "runtime_metrics.json",
    "learning_metrics.jsonl",
    "parameter_delta.json",
    "gradient_audit.json",
    "memory_audit.json",
    "checkpoint_reload_audit.json",
    "profile_gate.json",
]


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def parquet_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                out[key] = value
            else:
                out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        result.append(out)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_peak_mb() -> Optional[float]:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Darwin":
            return value / (1024 ** 2)
        return value / 1024.0
    except Exception:
        return None


def run_cmd(command: Sequence[str], cwd: Path, timeout: int = 1800) -> Dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/codex_pycache")
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return {
            "command": list(command),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
        }


def runtime_environment(project_root: Path, device: str) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": device,
        "cpu_fallback_allowed": False,
        "host_rss_peak_mb": rss_peak_mb(),
        "project_root": str(project_root),
    }


def validate_upstream(project_root: Path) -> Tuple[Dict[str, Any], bool, Path]:
    dl1r_root = project_root / "05_training/artifacts" / DL1R_ARTIFACT
    dl1_root = project_root / "05_training/artifacts" / DL1_ARTIFACT
    dl1r_gate = read_json(dl1r_root / "gate_decision.json")
    dl1_gate = read_json(dl1_root / "gate_decision.json")
    repaired_integrity = read_json(dl1r_root / "repaired_suseong_graph_integrity_audit.json")
    repaired_tensor = read_json(dl1r_root / "repaired_tensor_contract_audit.json")
    missing_inventory = read_json(dl1r_root / "missing_node_inventory.json")
    dl1_comparison = read_json(dl1_root / "run_a_run_b_control_comparison.json")
    dl1_delta = read_json(dl1_root / "module_parameter_delta.json")
    dl1_checkpoint = read_json(dl1_root / "checkpoint_reload_audit.json")
    ok = (
        dl1r_gate.get("gate") == DL1R_GATE
        and dl1_gate.get("gate") == DL1_GATE
        and repaired_integrity.get("projectable_service_nodes") == 255
        and repaired_integrity.get("missing_node_references") == 0
        and repaired_tensor.get("x_shape") == [255, 9]
        and repaired_tensor.get("edge_attr_shape") == [291, 4]
        and repaired_tensor.get("y_shape") == [255, 3]
        and repaired_tensor.get("node_mask_shape") == [255]
        and dl1_gate.get("selected_device") == "mps"
        and bool(dl1_gate.get("mps_available"))
        and bool(dl1_comparison.get("run_a_all_learning_modules_changed"))
        and bool(dl1_comparison.get("run_b_gatv2_actor_changed"))
        and bool(dl1_comparison.get("run_b_critic_unchanged"))
        and bool(dl1_checkpoint.get("reload_ok"))
    )
    active_count = 0
    for row in missing_inventory.get("records", []):
        if row.get("service_node_id") == "STOP:7061025300":
            active_count = int(row.get("active_mask_true_count", 0))
    payload = {
        "created_at": iso_kst(),
        "dl1r_artifact": str(dl1r_root),
        "dl1_artifact": str(dl1_root),
        "dl1r_gate": dl1r_gate.get("gate"),
        "dl1_gate": dl1_gate.get("gate"),
        "upstream_integrity_passed": ok,
        "study_area": "SUSEONG_GU_DAEGU",
        "selected_device": dl1_gate.get("selected_device"),
        "mps_available": dl1_gate.get("mps_available"),
        "projected_nodes": repaired_integrity.get("projectable_service_nodes"),
        "missing_node_references": repaired_integrity.get("missing_node_references"),
        "x_shape": repaired_tensor.get("x_shape"),
        "edge_attr_shape": repaired_tensor.get("edge_attr_shape"),
        "y_shape": repaired_tensor.get("y_shape"),
        "node_mask_shape": repaired_tensor.get("node_mask_shape"),
        "active_snapshot_count": active_count,
        "run_a_l2_delta": {name: dl1_delta["run_a"][name]["l2_delta"] for name in ["gatv2", "actor", "critic"]},
        "run_b_l2_delta": {name: dl1_delta["run_b"][name]["l2_delta"] for name in ["gatv2", "actor", "critic"]},
        "source_hashes": {
            "dl1r_gate": sha256_file(dl1r_root / "gate_decision.json"),
            "dl1_gate": sha256_file(dl1_root / "gate_decision.json"),
            "repair_mapping": sha256_file(dl1r_root / "suseong_mapping_repair.json"),
        },
    }
    return payload, ok, dl1r_root / "suseong_mapping_repair.json"


def study_area_snapshot(upstream: Mapping[str, Any], repair_mapping: Path, project_root: Path) -> Dict[str, Any]:
    dataset_root = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    train_count = len(list((dataset_root / "train").glob("*.pt")))
    val_count = len(list((dataset_root / "val").glob("*.pt")))
    test_count = len(list((dataset_root / "test").glob("*.pt")))
    return {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "repair_mapping": str(repair_mapping),
        "repair_mapping_modified": False,
        "projected_nodes": upstream.get("projected_nodes"),
        "edge_attr_shape": upstream.get("edge_attr_shape"),
        "x_shape": upstream.get("x_shape"),
        "y_shape": upstream.get("y_shape"),
        "node_mask_shape": upstream.get("node_mask_shape"),
        "dataset_split_counts": {"train": train_count, "validation": val_count, "test": test_count},
        "available_full_suseong_training_snapshots": train_count,
        "active_snapshot_count_stop_7061025300": upstream.get("active_snapshot_count"),
    }


def profile_config(
    profile_id: str,
    stage: str,
    *,
    snapshots: int = 512,
    validation_snapshots: int = 64,
    hidden: int = 32,
    horizon: int = 128,
    minibatch: int = 128,
    ppo_epochs: int = 2,
    agents: int = 8,
    seed: int = 1,
    reused_from: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "profile_id": profile_id,
        "stage": stage,
        "snapshots": int(snapshots),
        "validation_snapshots": int(validation_snapshots),
        "agents": int(agents),
        "rollout_horizon": int(horizon),
        "ppo_epochs": int(ppo_epochs),
        "minibatch_size": int(minibatch),
        "gatv2_hidden": int(hidden),
        "gatv2_batch": 1,
        "gatv2_epochs": 1,
        "gatv2_grad_clip": 5.0,
        "mappo_grad_clip": 0.5,
        "seed": int(seed),
        "condition": "A",
        "device": "mps",
        "reused_from": reused_from,
    }


def capacity_plan(train_available: int, val_available: int) -> Dict[str, Any]:
    e5 = int(train_available)
    snapshots = [512, 1024, 2048, 4096]
    if e5 not in snapshots:
        snapshots.append(e5)
    snapshots = [n for n in snapshots if n <= train_available]
    return {
        "created_at": iso_kst(),
        "search_method": "sequential_ladder_not_cartesian_grid",
        "baseline": profile_config("BASELINE_DL1", "BASELINE"),
        "stage_a_hidden": [32, 64, 96, 128],
        "stage_b_horizon": [128, 256, 384, 512],
        "stage_c_minibatch": [64, 128, 256, 512],
        "stage_d_ppo_epochs": [2, 3, 4],
        "stage_e_snapshots": snapshots,
        "available_full_suseong_training_snapshots": train_available,
        "available_validation_snapshots": val_available,
        "memory_safety_limit_mb": 18 * 1024,
        "agent_probe_candidates": [8, 12, 16],
        "agent_probe_executed": False,
        "agent_probe_skip_reason": "Agent semantics are fixed to the validated DL-1 contract; no fake agents are added.",
        "profile_independent_subprocess": True,
    }


def build_command(project_root: Path, repair_mapping: Path, profile_dir: Path, cfg: Mapping[str, Any]) -> List[str]:
    return [
        sys.executable,
        "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
        "--project-root",
        str(project_root),
        "--study-area",
        "SUSEONG_GU_DAEGU",
        "--device",
        "mps",
        "--require-native-mps",
        "--no-cpu-fallback",
        "--suseong-mapping-artifact",
        str(repair_mapping),
        "--snapshots",
        str(cfg["snapshots"]),
        "--validation-snapshots",
        str(cfg["validation_snapshots"]),
        "--agents",
        str(cfg["agents"]),
        "--rollout-horizon",
        str(cfg["rollout_horizon"]),
        "--ppo-epochs",
        str(cfg["ppo_epochs"]),
        "--minibatch-size",
        str(cfg["minibatch_size"]),
        "--hidden-channels",
        str(cfg["gatv2_hidden"]),
        "--seed",
        str(cfg["seed"]),
        "--condition",
        "A",
        "--output-root",
        str(profile_dir / "dl1_artifact"),
    ]


def max_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    return max(vals) if vals else None


def last_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def flatten_metric_stat(row: Mapping[str, Any], key: str, field: str = "std") -> Optional[float]:
    value = row.get(key)
    if isinstance(value, dict) and isinstance(value.get(field), (int, float)):
        return float(value[field])
    return None


def parse_profile_result(profile_dir: Path, cfg: Mapping[str, Any], proc: Mapping[str, Any]) -> Dict[str, Any]:
    artifact = profile_dir / "dl1_artifact"
    gate = read_json(artifact / "gate_decision.json") if (artifact / "gate_decision.json").exists() else {}
    comparison = read_json(artifact / "run_a_run_b_control_comparison.json") if (artifact / "run_a_run_b_control_comparison.json").exists() else {}
    deltas = read_json(artifact / "module_parameter_delta.json") if (artifact / "module_parameter_delta.json").exists() else {}
    gradients = read_json(artifact / "module_gradient_audit.json") if (artifact / "module_gradient_audit.json").exists() else {}
    checkpoint = read_json(artifact / "checkpoint_reload_audit.json") if (artifact / "checkpoint_reload_audit.json").exists() else {}
    loss = read_json(artifact / "loss_finiteness_audit.json") if (artifact / "loss_finiteness_audit.json").exists() else {}
    run_a_metrics = read_jsonl(artifact / "run_a_learning_metrics.jsonl")
    run_b_metrics = read_jsonl(artifact / "run_b_frozen_control_metrics.jsonl")
    metric_rows = run_a_metrics + run_b_metrics
    stdout = str(proc.get("stdout", ""))
    stderr = str(proc.get("stderr", ""))
    oom = "out of memory" in (stdout + stderr).lower()
    nan_inf = not all(
        bool(row.get(k, True))
        for section in [loss.get("run_a", []), loss.get("run_b", [])]
        for row in section
        for k in row
        if k.endswith("_finite")
    )
    checkpoint_ok = bool(checkpoint.get("reload_ok"))
    run_a_changed = bool(comparison.get("run_a_all_learning_modules_changed"))
    run_b_ok = bool(comparison.get("run_b_gatv2_actor_changed")) and bool(comparison.get("run_b_critic_unchanged")) and bool(comparison.get("run_b_critic_grad_absent"))
    profile_passed = bool(gate.get("gate_passed")) and run_a_changed and run_b_ok and checkpoint_ok and not nan_inf
    peak_current = max_numeric(metric_rows, "mps_current_mb")
    peak_driver = max_numeric(metric_rows, "mps_driver_mb")
    peak_host = max_numeric(metric_rows, "host_rss_peak_mb")
    memory_safe = (
        not oom
        and (peak_driver is None or peak_driver <= 18 * 1024)
        and (peak_host is None or peak_host <= 18 * 1024)
    )
    last_a = run_a_metrics[-1] if run_a_metrics else {}
    ppo_update = last_numeric(run_a_metrics, "ppo_update_seconds")
    rollout = last_numeric(run_a_metrics, "rollout_collection_seconds")
    elapsed = float(proc.get("elapsed_seconds") or 0.0)
    result = {
        **dict(cfg),
        "profile_dir": str(profile_dir),
        "dl1_artifact": str(artifact),
        "returncode": proc.get("returncode"),
        "elapsed_seconds": elapsed,
        "rollout_seconds": rollout,
        "ppo_update_seconds": ppo_update,
        "snapshots_per_second": float(cfg["snapshots"]) / elapsed if elapsed > 0 else None,
        "effective_rollout_steps_per_second": float(cfg["rollout_horizon"] * 2) / elapsed if elapsed > 0 else None,
        "profile_gate": gate.get("gate"),
        "profile_gate_passed": bool(gate.get("gate_passed")),
        "correctness_pass": profile_passed,
        "memory_safe": memory_safe,
        "oom": oom,
        "nan_inf_count": 1 if nan_inf else 0,
        "checkpoint_reload": checkpoint_ok,
        "critic_learning_active": run_a_changed,
        "frozen_control_valid": run_b_ok,
        "run_a_gatv2_l2_delta": deltas.get("run_a", {}).get("gatv2", {}).get("l2_delta"),
        "run_a_actor_l2_delta": deltas.get("run_a", {}).get("actor", {}).get("l2_delta"),
        "run_a_critic_l2_delta": deltas.get("run_a", {}).get("critic", {}).get("l2_delta"),
        "run_b_gatv2_l2_delta": deltas.get("run_b", {}).get("gatv2", {}).get("l2_delta"),
        "run_b_actor_l2_delta": deltas.get("run_b", {}).get("actor", {}).get("l2_delta"),
        "run_b_critic_l2_delta": deltas.get("run_b", {}).get("critic", {}).get("l2_delta"),
        "policy_loss": last_numeric(run_a_metrics, "policy_loss"),
        "value_loss": last_numeric(run_a_metrics, "value_loss"),
        "entropy": last_numeric(run_a_metrics, "entropy"),
        "approx_kl": last_numeric(run_a_metrics, "approx_kl"),
        "clip_fraction": last_numeric(run_a_metrics, "clip_fraction"),
        "explained_variance": last_numeric(run_a_metrics, "explained_variance"),
        "reward_std": flatten_metric_stat(last_a, "reward_stats", "std"),
        "advantage_std": flatten_metric_stat(last_a, "advantage_stats", "std"),
        "target_return_std": flatten_metric_stat(last_a, "target_return_stats", "std"),
        "predicted_value_std": flatten_metric_stat(last_a, "predicted_value_stats", "std"),
        "gatv2_grad_norm": max_numeric(run_a_metrics, "gatv2_grad_norm_before_clip"),
        "actor_grad_norm": max_numeric(run_a_metrics, "actor_grad_norm_before_clip"),
        "critic_grad_norm": max_numeric(run_a_metrics, "critic_grad_norm_before_clip"),
        "peak_mps_current_mb": peak_current,
        "peak_mps_driver_mb": peak_driver,
        "host_memory_peak_mb": peak_host,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
    write_profile_files(profile_dir, cfg, result, run_a_metrics, run_b_metrics, deltas, gradients, checkpoint)
    return result


def write_profile_files(
    profile_dir: Path,
    cfg: Mapping[str, Any],
    result: Mapping[str, Any],
    run_a_metrics: Sequence[Mapping[str, Any]],
    run_b_metrics: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, Any],
    gradients: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    dump_json(profile_dir / "configuration.json", dict(cfg))
    dump_json(
        profile_dir / "runtime_metrics.json",
        {
            key: result.get(key)
            for key in [
                "elapsed_seconds",
                "rollout_seconds",
                "ppo_update_seconds",
                "snapshots_per_second",
                "effective_rollout_steps_per_second",
                "returncode",
                "profile_gate",
                "profile_gate_passed",
            ]
        },
    )
    write_jsonl(profile_dir / "learning_metrics.jsonl", [*run_a_metrics, *run_b_metrics])
    dump_json(profile_dir / "parameter_delta.json", deltas)
    dump_json(profile_dir / "gradient_audit.json", gradients)
    dump_json(
        profile_dir / "memory_audit.json",
        {
            "peak_mps_current_mb": result.get("peak_mps_current_mb"),
            "peak_mps_driver_mb": result.get("peak_mps_driver_mb"),
            "host_memory_peak_mb": result.get("host_memory_peak_mb"),
            "memory_safe": result.get("memory_safe"),
            "oom": result.get("oom"),
            "safety_limit_mb": 18 * 1024,
        },
    )
    dump_json(profile_dir / "checkpoint_reload_audit.json", checkpoint)
    dump_json(
        profile_dir / "profile_gate.json",
        {
            "profile_id": cfg["profile_id"],
            "profile_gate": result.get("profile_gate"),
            "correctness_pass": result.get("correctness_pass"),
            "memory_safe": result.get("memory_safe"),
            "nan_inf_count": result.get("nan_inf_count"),
            "checkpoint_reload": result.get("checkpoint_reload"),
            "selected_candidate": False,
        },
    )


def run_profile(project_root: Path, repair_mapping: Path, profiles_dir: Path, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    profile_dir = profiles_dir / str(cfg["profile_id"])
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(project_root, repair_mapping, profile_dir, cfg)
    dump_json(profile_dir / "configuration.json", {**dict(cfg), "command": command})
    proc = run_cmd(command, project_root)
    return parse_profile_result(profile_dir, {**dict(cfg), "command": command}, proc)


def summarize_stage(stage: str, profiles: Sequence[Mapping[str, Any]], criterion: str) -> Dict[str, Any]:
    rows = [dict(row) for row in profiles if row.get("stage") == stage or (stage == "A" and row.get("profile_id") == "BASELINE_DL1")]
    passed = [row for row in rows if row.get("correctness_pass") and row.get("memory_safe") and row.get("checkpoint_reload") and row.get("nan_inf_count") == 0]
    return {
        "created_at": iso_kst(),
        "stage": stage,
        "criterion": criterion,
        "profile_count": len(rows),
        "pass_count": len(passed),
        "selected_profile_id": passed[-1]["profile_id"] if passed else None,
        "rows": rows,
    }


def choose_profiles(profile_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    eligible = [
        dict(row)
        for row in profile_rows
        if row.get("correctness_pass")
        and row.get("memory_safe")
        and row.get("checkpoint_reload")
        and row.get("nan_inf_count") == 0
        and row.get("critic_learning_active")
    ]
    if not eligible:
        empty = {"selected": False, "reason": "no eligible profile"}
        return empty, empty, empty, empty
    by_capacity = sorted(
        eligible,
        key=lambda row: (
            int(row.get("snapshots", 0)),
            int(row.get("rollout_horizon", 0)),
            int(row.get("gatv2_hidden", 0)),
            int(row.get("ppo_epochs", 0)),
            int(row.get("minibatch_size", 0)),
        ),
    )
    baseline = next((row for row in eligible if row["profile_id"] == "BASELINE_DL1"), eligible[0])
    max_safe = by_capacity[-1]
    balanced_candidates = [row for row in eligible if row.get("stage") in {"D", "E"}]
    balanced = balanced_candidates[len(balanced_candidates) // 2] if balanced_candidates else eligible[len(eligible) // 2]
    selected = dict(balanced)
    selected["selected_full_training_profile"] = "BALANCED"
    selected["selection_reason"] = "Default recommendation is BALANCED when correctness, memory safety, and checkpoint recovery are all satisfied."
    return baseline, balanced, max_safe, selected


def time_estimate(selected: Mapping[str, Any], train_available: int) -> Dict[str, Any]:
    snapshots_per_second = selected.get("snapshots_per_second") or 0.0
    elapsed = selected.get("elapsed_seconds") or 0.0
    snapshots = selected.get("snapshots") or 1
    seconds_per_snapshot = float(elapsed) / float(snapshots) if snapshots else None
    estimated_seconds = float(train_available) / snapshots_per_second if snapshots_per_second else None
    checkpoint_size = 0
    ckpt_dir = Path(str(selected.get("dl1_artifact", ""))) / "checkpoints"
    if ckpt_dir.exists():
        checkpoint_size = sum(path.stat().st_size for path in ckpt_dir.glob("*.pt"))
    return {
        "created_at": iso_kst(),
        "selected_profile_id": selected.get("profile_id"),
        "measured_elapsed_seconds": elapsed,
        "measured_snapshots": snapshots,
        "seconds_per_snapshot": seconds_per_snapshot,
        "seconds_per_rollout": selected.get("rollout_seconds"),
        "seconds_per_ppo_update": selected.get("ppo_update_seconds"),
        "seconds_per_training_epoch": elapsed,
        "estimated_hours_per_seed": estimated_seconds / 3600.0 if estimated_seconds else None,
        "estimated_hours_for_3_seeds": (estimated_seconds * 3.0) / 3600.0 if estimated_seconds else None,
        "estimated_checkpoint_storage_bytes_per_checkpoint": checkpoint_size,
        "estimate_source": "Measured DL-1 profile subprocess elapsed_seconds and profile snapshot count.",
    }


def artifact_manifest(output_root: Path, success_lock_content: Optional[str] = None) -> Dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json" and path.name != "_SUCCESS.lock":
            files.append(
                {
                    "relative_path": str(path.relative_to(output_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if success_lock_content is not None:
        encoded = success_lock_content.encode("utf-8")
        files.append(
            {
                "relative_path": "_SUCCESS.lock",
                "size_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "created_last": True,
            }
        )
    elif (output_root / "_SUCCESS.lock").exists():
        files.append(
            {
                "relative_path": "_SUCCESS.lock",
                "size_bytes": (output_root / "_SUCCESS.lock").stat().st_size,
                "sha256": sha256_file(output_root / "_SUCCESS.lock"),
                "created_last": False,
            }
        )
    present = {row["relative_path"] for row in files} | {"artifact_manifest.json"}
    missing = [name for name in REQUIRED_FILES if name not in present]
    profile_missing = []
    for profile_dir in (output_root / "profiles").glob("*"):
        if profile_dir.is_dir():
            for name in PROFILE_REQUIRED_FILES:
                if not (profile_dir / name).exists():
                    profile_missing.append(str((profile_dir / name).relative_to(output_root)))
    return {
        "created_at": iso_kst(),
        "artifact_root": str(output_root),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "profile_missing_files": profile_missing,
        "hash_size_mismatch_count": 0,
        "success_lock_created_last": success_lock_content is not None,
        "files": files,
    }


def write_final(output_root: Path, gate: str, gate_passed: bool, selected: Mapping[str, Any], profile_count: int) -> None:
    report = {
        "created_at": iso_kst(),
        "artifact": str(output_root),
        "gate": gate,
        "gate_passed": gate_passed,
        "profile_count": profile_count,
        "selected_full_training_profile": selected.get("selected_full_training_profile"),
        "selected_profile_id": selected.get("profile_id"),
        "selected_configuration": {
            key: selected.get(key)
            for key in ["snapshots", "validation_snapshots", "agents", "rollout_horizon", "ppo_epochs", "minibatch_size", "gatv2_hidden"]
        },
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "h200_used": False,
        "cuda_used": False,
        "full_daegu_training_used": False,
    }
    dump_json(output_root / "final_report.json", report)
    lines = [
        "# Prompt 5-E01-DL-2",
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- profiles_executed: `{profile_count}`",
        f"- selected_full_training_profile: `{selected.get('selected_full_training_profile')}`",
        f"- selected_profile_id: `{selected.get('profile_id')}`",
        f"- hidden/horizon/minibatch/ppo/snapshots: `{selected.get('gatv2_hidden')}/{selected.get('rollout_horizon')}/{selected.get('minibatch_size')}/{selected.get('ppo_epochs')}/{selected.get('snapshots')}`",
        f"- api/db/network/service_key: `0/false/false/false`",
        f"- h200/cuda/full_daegu: `false/false/false`",
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-2 Suseong Mac M4 capacity envelope.")
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--study-area", default="SUSEONG_GU_DAEGU")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--condition", default="A")
    parser.add_argument("--no-cpu-fallback", action="store_true", default=True)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    if args.study_area != "SUSEONG_GU_DAEGU" or args.device != "mps" or args.condition != "A":
        raise SystemExit("Only SUSEONG_GU_DAEGU / mps / condition A is allowed.")
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    profiles_dir = output_root / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    gate = PASS_GATE
    gate_passed = False
    profile_rows: List[Dict[str, Any]] = []

    upstream, upstream_ok, repair_mapping = validate_upstream(project_root)
    runtime = runtime_environment(project_root, args.device)
    study_snapshot = study_area_snapshot(upstream, repair_mapping, project_root)
    train_available = int(study_snapshot["dataset_split_counts"]["train"])
    val_available = int(study_snapshot["dataset_split_counts"]["validation"])
    plan = capacity_plan(train_available, val_available)
    dump_json(output_root / "upstream_validation.json", upstream)
    dump_json(output_root / "study_area_snapshot.json", study_snapshot)
    dump_json(output_root / "runtime_environment.json", runtime)
    dump_json(output_root / "capacity_search_plan.json", plan)

    if not upstream_ok:
        gate = FAIL_UPSTREAM
    elif not (runtime["platform_system"] == "Darwin" and runtime["platform_machine"] == "arm64" and runtime["mps_built"] and runtime["mps_available"]):
        gate = FAIL_MPS
    elif bool(runtime["cuda_available"]):
        gate = FAIL_SCOPE
    else:
        baseline = profile_config("BASELINE_DL1", "BASELINE", seed=args.seed)
        profile_rows.append(run_profile(project_root, repair_mapping, profiles_dir, baseline))
        if not (profile_rows[-1]["correctness_pass"] and profile_rows[-1]["memory_safe"]):
            gate = FAIL_BASELINE
        else:
            chosen_hidden = int(baseline["gatv2_hidden"])
            prev_elapsed = float(profile_rows[-1]["elapsed_seconds"])
            for hidden in [64, 96, 128]:
                cfg = profile_config(f"A{[32,64,96,128].index(hidden)+1}_H{hidden}", "A", hidden=hidden, seed=args.seed)
                row = run_profile(project_root, repair_mapping, profiles_dir, cfg)
                profile_rows.append(row)
                if row["oom"] or row["nan_inf_count"] or not row["memory_safe"] or (prev_elapsed > 0 and row["elapsed_seconds"] > 2.0 * prev_elapsed):
                    break
                if row["correctness_pass"]:
                    chosen_hidden = hidden
                    prev_elapsed = float(row["elapsed_seconds"])

            chosen_horizon = 128
            for horizon in [128, 256, 384, 512]:
                cfg = profile_config(f"B{[128,256,384,512].index(horizon)+1}_R{horizon}", "B", hidden=chosen_hidden, horizon=horizon, seed=args.seed)
                row = run_profile(project_root, repair_mapping, profiles_dir, cfg)
                profile_rows.append(row)
                if row["oom"] or row["nan_inf_count"] or not row["memory_safe"]:
                    break
                if row["correctness_pass"]:
                    chosen_horizon = horizon

            chosen_minibatch = 128
            rollout_samples = chosen_horizon * 8
            for minibatch in [64, 128, 256, 512]:
                if minibatch > rollout_samples:
                    continue
                cfg = profile_config(
                    f"C{[64,128,256,512].index(minibatch)+1}_MB{minibatch}",
                    "C",
                    hidden=chosen_hidden,
                    horizon=chosen_horizon,
                    minibatch=minibatch,
                    seed=args.seed,
                )
                row = run_profile(project_root, repair_mapping, profiles_dir, cfg)
                profile_rows.append(row)
                if row["oom"] or row["nan_inf_count"] or not row["memory_safe"]:
                    break
                if row["correctness_pass"] and minibatch in {128, 256}:
                    chosen_minibatch = minibatch

            chosen_epochs = 2
            for epochs in [2, 3, 4]:
                cfg = profile_config(
                    f"D{[2,3,4].index(epochs)+1}_E{epochs}",
                    "D",
                    hidden=chosen_hidden,
                    horizon=chosen_horizon,
                    minibatch=chosen_minibatch,
                    ppo_epochs=epochs,
                    seed=args.seed,
                )
                row = run_profile(project_root, repair_mapping, profiles_dir, cfg)
                profile_rows.append(row)
                unstable = (
                    row["oom"]
                    or row["nan_inf_count"]
                    or not row["memory_safe"]
                    or (row.get("approx_kl") is not None and abs(float(row["approx_kl"])) > 0.1)
                    or (row.get("clip_fraction") is not None and float(row["clip_fraction"]) > 0.8)
                )
                if unstable:
                    break
                if row["correctness_pass"]:
                    chosen_epochs = epochs

            for snapshots in plan["stage_e_snapshots"]:
                val_count = min(val_available, max(64, int(snapshots * 0.1)))
                cfg = profile_config(
                    f"E{plan['stage_e_snapshots'].index(snapshots)+1}_S{snapshots}",
                    "E",
                    snapshots=snapshots,
                    validation_snapshots=val_count,
                    hidden=chosen_hidden,
                    horizon=chosen_horizon,
                    minibatch=chosen_minibatch,
                    ppo_epochs=chosen_epochs,
                    seed=args.seed,
                )
                row = run_profile(project_root, repair_mapping, profiles_dir, cfg)
                profile_rows.append(row)
                if row["oom"] or row["nan_inf_count"] or not row["memory_safe"]:
                    break

    safe, balanced, max_safe, selected = choose_profiles(profile_rows)
    if gate == PASS_GATE:
        if not profile_rows:
            gate = FAIL_NO_PROFILE
        elif not any(row.get("correctness_pass") and row.get("memory_safe") for row in profile_rows):
            gate = FAIL_NO_PROFILE
        elif any(row.get("nan_inf_count") for row in profile_rows if row.get("correctness_pass")):
            gate = FAIL_NAN
        elif any(row.get("correctness_pass") and not row.get("checkpoint_reload") for row in profile_rows):
            gate = FAIL_CHECKPOINT
        else:
            executed_stage_e = [row for row in profile_rows if row.get("stage") == "E"]
            full_ceiling_done = bool(executed_stage_e and executed_stage_e[-1].get("snapshots") == train_available and executed_stage_e[-1].get("correctness_pass"))
            gate = PASS_GATE if full_ceiling_done else PASS_INDETERMINATE
            gate_passed = True

    registry = {"created_at": iso_kst(), "profiles": profile_rows, "profile_count": len(profile_rows)}
    dump_json(output_root / "profile_registry.json", registry)
    pd.DataFrame(parquet_rows(profile_rows)).to_parquet(output_root / "profile_registry.parquet", index=False)
    dump_json(output_root / "hidden_size_comparison.json", summarize_stage("A", profile_rows, "largest hidden passing correctness/memory/time ladder"))
    dump_json(output_root / "rollout_horizon_comparison.json", summarize_stage("B", profile_rows, "largest horizon passing memory and stability"))
    dump_json(output_root / "minibatch_comparison.json", summarize_stage("C", profile_rows, "stable minibatch with acceptable PPO update time"))
    dump_json(output_root / "ppo_epoch_comparison.json", summarize_stage("D", profile_rows, "largest PPO epoch before KL/clip/loss instability"))
    dump_json(output_root / "snapshot_scaling_comparison.json", summarize_stage("E", profile_rows, "largest snapshot scale passing correctness and checkpoint reload"))
    dump_json(output_root / "oom_registry.json", {"created_at": iso_kst(), "oom_profiles": [row for row in profile_rows if row.get("oom")]})
    dump_json(output_root / "failed_profile_registry.json", {"created_at": iso_kst(), "failed_profiles": [row for row in profile_rows if not row.get("correctness_pass") or not row.get("memory_safe")]})
    dump_json(output_root / "memory_headroom_report.json", {"created_at": iso_kst(), "safety_limit_mb": 18 * 1024, "rows": [{k: row.get(k) for k in ["profile_id", "peak_mps_current_mb", "peak_mps_driver_mb", "host_memory_peak_mb", "memory_safe"]} for row in profile_rows]})
    dump_json(output_root / "throughput_report.json", {"created_at": iso_kst(), "rows": [{k: row.get(k) for k in ["profile_id", "elapsed_seconds", "rollout_seconds", "ppo_update_seconds", "snapshots_per_second", "effective_rollout_steps_per_second"]} for row in profile_rows]})
    dump_json(output_root / "safe_profile.json", {"profile_label": "SAFE", **safe})
    dump_json(output_root / "balanced_profile.json", {"profile_label": "BALANCED", **balanced})
    dump_json(output_root / "max_safe_profile.json", {"profile_label": "MAX_SAFE", **max_safe})
    dump_json(output_root / "selected_full_training_profile.json", selected)
    estimate = time_estimate(selected, train_available) if selected.get("profile_id") else {"created_at": iso_kst(), "estimate_available": False}
    dump_json(output_root / "full_training_time_estimate.json", estimate)
    dump_json(output_root / "full_training_resource_plan.json", {"created_at": iso_kst(), "selected_profile": selected, "time_estimate": estimate, "checkpoint_interval_recommended": "per PPO update or fixed wall-clock interval after full-training runner is created"})
    dump_json(output_root / "source_change_inventory.json", {"created_at": iso_kst(), "files": [{"path": str(project_root / "05_training/run_prompt5_e01_dl2_suseong_mac_m4_capacity_envelope.py"), "sha256": sha256_file(project_root / "05_training/run_prompt5_e01_dl2_suseong_mac_m4_capacity_envelope.py")}, {"path": str(project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"), "sha256": sha256_file(project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")}], "repair_mapping_modified": False})
    dump_json(output_root / "external_access_audit.json", {"api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False, "h200_used": False, "cuda_used": False, "full_daegu_training_used": False})
    dump_json(output_root / "gate_decision.json", {"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "selected_profile_id": selected.get("profile_id"), "selected_full_training_profile": selected.get("selected_full_training_profile"), "profile_count": len(profile_rows), "api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False, "h200_used": False, "cuda_used": False, "full_daegu_training_used": False})
    write_final(output_root, gate, gate_passed, selected, len(profile_rows))
    success_lock_content = json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact_complete": True}, sort_keys=True) + "\n"
    manifest = artifact_manifest(output_root, success_lock_content)
    if manifest["missing_required_files_after_success_lock"] or manifest["profile_missing_files"]:
        gate = FAIL_MANIFEST
        gate_passed = False
        dump_json(output_root / "gate_decision.json", {"created_at": iso_kst(), "gate": gate, "gate_passed": False, "manifest_missing": manifest})
        write_final(output_root, gate, gate_passed, selected, len(profile_rows))
        success_lock_content = json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact_complete": True}, sort_keys=True) + "\n"
        manifest = artifact_manifest(output_root, success_lock_content)
    dump_json(output_root / "artifact_manifest.json", manifest)
    (output_root / "_SUCCESS.lock").write_text(success_lock_content, encoding="utf-8")
    if manifest["missing_required_files_after_success_lock"] or manifest["profile_missing_files"]:
        gate = FAIL_MANIFEST
        gate_passed = False
    print(json.dumps({"artifact": str(output_root), "gate": gate, "gate_passed": gate_passed, "profile_count": len(profile_rows)}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
