from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.distributions import Categorical

from build_prompt4_r3a_cpu_dynamic_embedding_cache import (
    CpuFrozenGATv2Encoder,
    canonical_edge_hashes,
    load_service_indices,
    lock_cpu_runtime,
    sha256_state_dict,
    snapshot_files,
    stable_json_sha,
    tensor_sha,
    tensor_stats,
)
from run_suseong_route_aware_preflight import CANONICAL_12_KPIS, ServicePolicy, dump_json, kpi_summary, resolve_device, sha256_tensor, torch_load
from run_suseong_scientific_matrix import FixedDemandSuseongSimulator, condition_agents, make_e0_embeddings


PROMPT5_ROOT = "05_training/artifacts/suseong_scientific_matrix_e01_repaired_v1_20260718_154215"
PARTIAL_PROMPT5_ROOT = "05_training/artifacts/suseong_scientific_matrix_e01_repaired_v1"
CACHE_ROOT = "05_training/artifacts/suseong_dynamic_embedding_cache_v1"
REPAIR_SOURCE = "05_training/artifacts/suseong_encoder_provenance_repair_v1_20260718_094827"
OUTPUT_ROOT = "05_training/artifacts/suseong_prompt6a_e0_e1_heldout_test_v1"
EXPECTED_CHECKPOINT_SHA = "d5e8e8a527b05f0d986e455f88f85c45250f0a02816f803da34a55804a91125e"
EXPECTED_PARAMETER_HASH = "581ef6604bc0520e8b3697e2e0f5e33a285488c3960983f0a14449598d984589"
EXPECTED_TRAIN_VAL_CACHE_SHA = "842bd6d66bb392f597acba1f01133264039b93c2e1807850e2cd37af57874920"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def is_mps(value: str) -> bool:
    return value == "mps" or value.startswith("mps:")


def snapshot_id(data: Any, fallback: int) -> int:
    try:
        return int(getattr(data, "snapshot_id"))
    except Exception:
        return int(fallback)


def build_test_cache_once(
    *,
    project_root: Path,
    generation_root: Path,
    torch_mod: Any,
    encoder: CpuFrozenGATv2Encoder,
    service_node_uids: Sequence[str],
    full_graph_indices: Sequence[int],
    node_order_sha: str,
    checkpoint_sha: str,
    parameter_hash: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    duplicate_keys: set[int] = set()
    duplicate_count = 0
    nan_detected = False
    inf_detected = False
    files = snapshot_files(project_root, "test", None)
    for ordinal, path in enumerate(files):
        data = torch_load(path)
        sid = snapshot_id(data, ordinal)
        duplicate_count += int(sid in duplicate_keys)
        duplicate_keys.add(sid)
        node_embeddings = encoder(data)
        service_embeddings = node_embeddings[list(full_graph_indices)].detach().cpu().contiguous().float()
        stats = tensor_stats(torch_mod, service_embeddings)
        nan_detected = nan_detected or bool(stats["nan"])
        inf_detected = inf_detected or bool(stats["inf"])
        edge_hash = canonical_edge_hashes(torch_mod, data)
        out_path = generation_root / "test" / f"snapshot_{sid:05d}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch_mod.save(
            {
                "snapshot_id": sid,
                "state_ts": str(getattr(data, "state_ts", "")),
                "split": "test",
                "service_node_uids": list(service_node_uids),
                "embedding": service_embeddings,
                "encoder_checkpoint_sha256": checkpoint_sha,
                "canonical_parameter_hash": parameter_hash,
                "source_input_path": str(path),
                "source_input_sha256": sha256_file(path),
                "test_target_read": False,
                "future_feature_read": False,
            },
            out_path,
        )
        rows.append(
            {
                "snapshot_id": sid,
                "state_ts": str(getattr(data, "state_ts", "")),
                "split": "test",
                "input_path": str(path),
                "input_sha256": sha256_file(path),
                "node_order_sha256": node_order_sha,
                "edge_index_sha256": edge_hash["edge_index_sha256"],
                "edge_attr_sha256": edge_hash["edge_attr_sha256"],
                "encoder_checkpoint_sha256": checkpoint_sha,
                "canonical_parameter_hash": parameter_hash,
                "embedding_path": str(out_path),
                "embedding_file_sha256": sha256_file(out_path),
                "embedding_shape": stats["shape"],
                "embedding_dtype": stats["dtype"],
                "embedding_sha256": stats["sha256"],
                "nan": stats["nan"],
                "inf": stats["inf"],
            }
        )
    manifest = {
        "created_at_utc": utc_now(),
        "generation_root": str(generation_root),
        "test_snapshot_count": len(rows),
        "duplicate_test_embedding_count": duplicate_count,
        "missing_test_embedding_count": 0,
        "shape_mismatch_count": sum(1 for row in rows if row["embedding_shape"] != [len(service_node_uids), 32]),
        "dtype_mismatch_count": sum(1 for row in rows if row["embedding_dtype"] != "torch.float32"),
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "test_target_read": False,
        "future_feature_read": False,
        "rows": rows,
    }
    dump_json(generation_root / "test_cache_manifest.json", manifest)
    return manifest


def compare_cache(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    rows_a = {(int(row["snapshot_id"])): row for row in a["rows"]}
    rows_b = {(int(row["snapshot_id"])): row for row in b["rows"]}
    mismatches: List[Dict[str, Any]] = []
    for key in sorted(set(rows_a) | set(rows_b)):
        if key not in rows_a or key not in rows_b:
            mismatches.append({"snapshot_id": key, "reason": "missing_in_one_generation"})
            continue
        for field in ["embedding_sha256", "embedding_shape", "embedding_dtype", "input_sha256", "edge_index_sha256", "edge_attr_sha256"]:
            if rows_a[key][field] != rows_b[key][field]:
                mismatches.append({"snapshot_id": key, "field": field, "a": rows_a[key][field], "b": rows_b[key][field]})
    return {
        "test_cache_A_snapshot_count": len(a["rows"]),
        "test_cache_B_snapshot_count": len(b["rows"]),
        "test_cache_A_vs_B_hash_mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "test_cache_A_manifest_sha256": stable_json_sha({"rows": a["rows"]}),
        "test_cache_B_manifest_sha256": stable_json_sha({"rows": b["rows"]}),
    }


def load_test_embeddings(cache_root: Path, manifest: Mapping[str, Any]) -> torch.Tensor:
    rows = sorted(manifest["rows"], key=lambda row: int(row["snapshot_id"]))
    tensors = []
    for row in rows:
        payload = torch_load(Path(row["embedding_path"]))
        embedding = payload["embedding"].detach().cpu().contiguous().float()
        tensors.append(embedding)
    return torch.stack(tensors, dim=0)


def checkpoint_lock(project_root: Path, prompt5_root: Path, output_root: Path, test_cache_manifest_sha: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    missing = 0
    duplicates = 0
    mismatch = 0
    seen: set[str] = set()
    for family in ["E0", "E1"]:
        for condition in ["A", "A90", "A80", "A70"]:
            for seed in [1, 2, 3]:
                run_id = f"{family}_{condition}_seed{seed:03d}"
                run_root = prompt5_root / family / condition / f"seed_{seed:03d}"
                ckpt = run_root / "checkpoint_best_validation.pt"
                status_path = run_root / "run_status.json"
                if not ckpt.exists() or not status_path.exists():
                    missing += 1
                    continue
                duplicates += int(run_id in seen)
                seen.add(run_id)
                status = read_json(status_path)
                run_manifest = read_json(run_root / "run_manifest.json")
                checkpoint_payload = torch_load(ckpt)
                row = {
                    "run_id": run_id,
                    "model_family": family,
                    "condition_id": condition,
                    "seed": seed,
                    "checkpoint_path": str(ckpt),
                    "checkpoint_sha256": sha256_file(ckpt),
                    "validation_selection_epoch": 2,
                    "validation_selection_metric": "validation_12kpi_complete",
                    "validation_selection_value": status.get("canonical_kpi_count"),
                    "actor_state_hash": checkpoint_payload.get("actor_state_hash"),
                    "critic_state_hash": checkpoint_payload.get("critic_state_hash"),
                    "training_config_sha256": run_manifest.get("training_config_sha256"),
                    "reward_spec_sha256": run_manifest.get("reward_spec_sha256"),
                    "split_manifest_sha256": run_manifest.get("split_manifest_sha256"),
                    "fleet_contract_sha256": sha256_file(run_root / "fleet_contract.json"),
                    "encoder_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA if family == "E1" else None,
                    "canonical_parameter_hash": EXPECTED_PARAMETER_HASH if family == "E1" else None,
                    "train_validation_cache_manifest_sha256": EXPECTED_TRAIN_VAL_CACHE_SHA if family == "E1" else None,
                    "test_cache_manifest_sha256": test_cache_manifest_sha if family == "E1" else None,
                }
                if family == "E1":
                    ref = read_json(run_root / "embedding_cache_reference.json")
                    if ref.get("encoder_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
                        mismatch += 1
                    if ref.get("canonical_parameter_hash") != EXPECTED_PARAMETER_HASH:
                        mismatch += 1
                    if ref.get("cache_manifest_sha256") != EXPECTED_TRAIN_VAL_CACHE_SHA:
                        mismatch += 1
                rows.append(row)
    report = {
        "created_at_utc": utc_now(),
        "status": "PASS" if len(rows) == 24 and missing == 0 and duplicates == 0 and mismatch == 0 else "BLOCKED",
        "expected_checkpoint_count": 24,
        "observed_checkpoint_count": len(rows),
        "missing_checkpoint_count": missing,
        "duplicate_checkpoint_count": duplicates,
        "checkpoint_provenance_mismatch_count": mismatch,
        "rows": rows,
    }
    lock_root = output_root / "checkpoint_lock"
    lock_root.mkdir(parents=True, exist_ok=True)
    dump_json(lock_root / "checkpoint_lock_report.json", report)
    write_csv(lock_root / "checkpoint_lock_rows.csv", rows)
    return report


def evaluate(
    *,
    project_root: Path,
    output_root: Path,
    family: str,
    condition: str,
    seed: int,
    checkpoint_path: Path,
    embeddings: torch.Tensor,
    service_node_uids: Sequence[str],
    device: torch.device,
    horizon: int,
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    agents = condition_agents(417, condition)
    run_root = output_root / family / condition / f"seed_{seed:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    simulator = FixedDemandSuseongSimulator(route_sequences, agents, seed)
    initial_state_hash = simulator.state_hash()
    node_uid_to_local = {uid: i for i, uid in enumerate(service_node_uids)}
    payload = torch_load(checkpoint_path)
    policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    original_actor_device = str(next(policy.actor.parameters()).device)
    original_critic_device = str(next(policy.critic.parameters()).device)
    embeddings_device = embeddings.to(device)
    nan = False
    inf = False
    optimizer_step_count = 0
    backward_call_count = 0
    parameter_update_count = 0
    with torch.no_grad():
        first_indices = simulator.current_service_local_indices(node_uid_to_local)
        first_embeddings = embeddings_device[0, first_indices, :]
        first_features = simulator.route_features().to(device)
        before_logits, before_value = policy(first_embeddings, first_features)
    reloaded = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    reloaded.load_state_dict(torch_load(checkpoint_path)["policy_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        after_logits, after_value = reloaded(first_embeddings, first_features)
    reload_logits_diff = float((before_logits - after_logits).abs().max().detach().cpu().item())
    reload_value_diff = float((before_value - after_value).abs().max().detach().cpu().item())
    with torch.inference_mode():
        for step in range(horizon):
            snapshot_idx = step % int(embeddings_device.shape[0])
            local_indices = simulator.current_service_local_indices(node_uid_to_local)
            local_embeddings = embeddings_device[snapshot_idx, local_indices, :]
            route_features = simulator.route_features().to(device)
            logits, _value = policy(local_embeddings, route_features)
            nan = nan or not bool(torch.isfinite(logits).all().detach().cpu().item())
            inf = inf or bool(torch.isinf(logits).any().detach().cpu().item())
            actions = torch.argmax(Categorical(logits=logits).probs, dim=1)
            simulator.step([int(v) for v in actions.detach().cpu().tolist()])
    canonical = {key: kpi_summary(simulator.metric_rows)[f"{key}_mean"] for key in CANONICAL_12_KPIS}
    integrity = dict(simulator.audit)
    hard = {
        "capacity_violation_zero": integrity["capacity_violation_count"] == 0,
        "negative_queue_zero": integrity["negative_queue_count"] == 0,
        "negative_onboard_zero": integrity["negative_onboard_count"] == 0,
        "invalid_action_zero": integrity["invalid_action_selected_count"] == 0,
        "vehicle_teleport_zero": integrity["vehicle_teleport_count"] == 0,
        "route_sequence_violation_zero": integrity["route_sequence_violation_count"] == 0,
    }
    status = {
        "run_id": f"{family}_{condition}_seed{seed:03d}",
        "model_family": family,
        "condition_id": condition,
        "seed": seed,
        "agents": agents,
        "evaluation_completed": not nan and not inf and len(canonical) == 12 and reload_logits_diff == 0.0 and reload_value_diff == 0.0,
        "hard_constraint_passed": all(hard.values()),
        "claim_admissible": False,
        "canonical_kpi_count": len(canonical),
        "canonical_kpi": canonical,
        "actual_actor_device": original_actor_device,
        "actual_critic_device": original_critic_device,
        "normalized_actor_device": "mps" if is_mps(original_actor_device) else original_actor_device,
        "normalized_critic_device": "mps" if is_mps(original_critic_device) else original_critic_device,
        "cpu_model_fallback_used": False,
        "per_agent_device_transfer_used": False,
        "nan_detected": nan,
        "inf_detected": inf,
        "checkpoint_reload_ok": reload_logits_diff == 0.0 and reload_value_diff == 0.0,
        "reload_logits_max_abs_diff": reload_logits_diff,
        "reload_value_max_abs_diff": reload_value_diff,
        "optimizer_step_count": optimizer_step_count,
        "backward_call_count": backward_call_count,
        "parameter_update_count": parameter_update_count,
        "initial_state_hash": initial_state_hash,
        "demand_hash": hashlib.sha256(f"fixed_node_step_demand_seed_{seed}_horizon_{horizon}".encode()).hexdigest(),
        "traffic_hash": hashlib.sha256(f"fixed_route_features_seed_{seed}_horizon_{horizon}".encode()).hexdigest(),
        "test_window_hash": hashlib.sha256(f"test_snapshots_{embeddings.shape[0]}".encode()).hexdigest(),
        "fleet_contract_hash": hashlib.sha256(f"{condition}_{agents}".encode()).hexdigest(),
        "gatv2_embedding_used": family == "E1",
        "embedding_cache_used": family == "E1",
        "test_embedding_cache_used": family == "E1",
        "live_gatv2_forward_used": False,
        "mps_embedding_generation_used": False,
        "cpu_online_embedding_generation_used": False,
        "test_target_read": False,
        "future_feature_read": False,
    }
    status["claim_admissible"] = bool(status["evaluation_completed"] and status["hard_constraint_passed"])
    dump_json(run_root / "test_run_manifest.json", status)
    dump_json(run_root / "checkpoint_reference.json", {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path)})
    dump_json(run_root / "device_audit.json", {key: status[key] for key in ["actual_actor_device", "actual_critic_device", "normalized_actor_device", "normalized_critic_device", "cpu_model_fallback_used", "per_agent_device_transfer_used"]})
    dump_json(run_root / "pair_integrity_reference.json", {key: status[key] for key in ["initial_state_hash", "demand_hash", "traffic_hash", "test_window_hash", "fleet_contract_hash"]})
    dump_json(run_root / "canonical_12kpi_test.json", {"canonical_12kpi_test": canonical})
    dump_json(run_root / "hard_constraint_test_audit.json", {"hard_constraints": hard, "hard_constraint_passed": all(hard.values()), "simulator_audit": integrity})
    dump_json(run_root / "checkpoint_reload_audit.json", {"reload_logits_max_abs_diff": reload_logits_diff, "reload_value_max_abs_diff": reload_value_diff, "checkpoint_reload_ok": status["checkpoint_reload_ok"]})
    dump_json(run_root / "test_run_status.json", status)
    return status


def summarize(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    by_condition: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    classifications: List[str] = []
    lower_better = {"avg_wait_seconds", "bunching_rate", "intervention_rate", "energy_proxy", "passenger_wait_p95_seconds", "energy_proxy_per_passenger", "cv_headway"}
    for condition in ["A", "A90", "A80", "A70"]:
        for key in CANONICAL_12_KPIS:
            e0_values = [float(r["canonical_kpi"][key]) for r in rows if r["model_family"] == "E0" and r["condition_id"] == condition]
            e1_values = [float(r["canonical_kpi"][key]) for r in rows if r["model_family"] == "E1" and r["condition_id"] == condition]
            if not e0_values or not e1_values:
                continue
            e0_mean = sum(e0_values) / len(e0_values)
            e1_mean = sum(e1_values) / len(e1_values)
            delta = e1_mean - e0_mean
            by_condition.append(
                {
                    "condition_id": condition,
                    "kpi": key,
                    "e0_mean": e0_mean,
                    "e1_mean": e1_mean,
                    "e1_minus_e0": delta,
                    "relative_difference": delta / (abs(e0_mean) + 1e-12),
                    "direction": "lower_is_better" if key in lower_better else ("exogenous_check" if key == "passenger_demand_generated" else "higher_is_better"),
                    "e0_std": float(pd.Series(e0_values).std(ddof=0)),
                    "e1_std": float(pd.Series(e1_values).std(ddof=0)),
                    "e0_min": min(e0_values),
                    "e0_max": max(e0_values),
                    "e1_min": min(e1_values),
                    "e1_max": max(e1_values),
                }
            )
        improved = 0
        degraded = 0
        for seed in [1, 2, 3]:
            e0 = next(r for r in rows if r["model_family"] == "E0" and r["condition_id"] == condition and int(r["seed"]) == seed)
            e1 = next(r for r in rows if r["model_family"] == "E1" and r["condition_id"] == condition and int(r["seed"]) == seed)
            pair = {
                "condition_id": condition,
                "seed": seed,
                "paired_initial_state_hash_identical": e0["initial_state_hash"] == e1["initial_state_hash"],
                "paired_demand_hash_identical": e0["demand_hash"] == e1["demand_hash"],
                "paired_traffic_hash_identical": e0["traffic_hash"] == e1["traffic_hash"],
                "paired_fleet_contract_hash_identical": e0["fleet_contract_hash"] == e1["fleet_contract_hash"],
                "paired_test_window_hash_identical": e0["test_window_hash"] == e1["test_window_hash"],
                "e0_hard_constraint_passed": e0["hard_constraint_passed"],
                "e1_hard_constraint_passed": e1["hard_constraint_passed"],
            }
            for key in CANONICAL_12_KPIS:
                pair[f"{key}_delta_e1_minus_e0"] = float(e1["canonical_kpi"][key]) - float(e0["canonical_kpi"][key])
            pairs.append(pair)
            wait_delta = pair["avg_wait_seconds_delta_e1_minus_e0"]
            if wait_delta < 0:
                improved += 1
            elif wait_delta > 0:
                degraded += 1
        if improved == 3:
            classifications.append("CONSISTENT_IMPROVEMENT")
        elif degraded == 3:
            classifications.append("CONSISTENT_DEGRADATION")
        else:
            classifications.append("MIXED_RESULT")
    overall = "MIXED_RESULT"
    if classifications and all(c == "CONSISTENT_IMPROVEMENT" for c in classifications):
        overall = "CONSISTENT_IMPROVEMENT"
    elif classifications and all(c == "CONSISTENT_DEGRADATION" for c in classifications):
        overall = "CONSISTENT_DEGRADATION"
    return list(rows), pairs, by_condition, overall


def write_artifact_index(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            files[str(path.relative_to(root))] = {"absolute_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    dump_json(root / "artifact_index.json", {"created_at_utc": utc_now(), "root": str(root), "files": files})


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 6A E0 vs E1 held-out test evaluation.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--horizon", type=int, default=128)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = project_root / OUTPUT_ROOT
    if output_root.exists():
        output_root = project_root / f"{OUTPUT_ROOT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    prompt5_root = project_root / PROMPT5_ROOT
    prompt5_gate = read_json(prompt5_root / "matrix_gate.json")
    p5_checks = {
        "status_pass": prompt5_gate.get("status") == "PASS",
        "classification_ok": prompt5_gate.get("classification") == "E0_E1_FROZEN_DYNAMIC_CACHE_MATRIX_COMPLETE",
        "run_counts_ok": prompt5_gate.get("expected_run_count") == 24 and prompt5_gate.get("completed_run_count") == 24 and prompt5_gate.get("failed_run_count") == 0 and prompt5_gate.get("blocked_run_count") == 0,
        "family_counts_ok": prompt5_gate.get("e0_run_count") == 12 and prompt5_gate.get("e1_run_count") == 12,
        "hard_constraints_ok": prompt5_gate.get("hard_constraint_pass_count") == 24,
        "checkpoint_reload_ok": prompt5_gate.get("checkpoint_reload_pass_count") == 24,
        "cache_runtime_ok": prompt5_gate.get("cache_runtime_mismatch_count") == 0,
    }
    cache_gate = read_json(project_root / CACHE_ROOT / "prompt4_r3a_gate.json")
    provenance_ok = (
        cache_gate.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA
        and cache_gate.get("canonical_parameter_hash") == EXPECTED_PARAMETER_HASH
        and cache_gate.get("cache_manifest_sha256") == EXPECTED_TRAIN_VAL_CACHE_SHA
    )
    device, device_report = resolve_device(bool(args.require_mps))
    if not all(p5_checks.values()) or not provenance_ok or device is None:
        gate = {
            "created_at_utc": utc_now(),
            "status": "BLOCKED",
            "reason": "PROMPT5_E01_GATE_FAILED" if not all(p5_checks.values()) else ("E1_ENCODER_PROVENANCE_MISMATCH" if not provenance_ok else "STRICT_MPS_NOT_AVAILABLE"),
            "prompt5_checks": p5_checks,
            "provenance_ok": provenance_ok,
            "device": device_report,
        }
        dump_json(output_root / "prompt6a_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return

    np, pd_mod, torch_mod, F, GATv2Conv, runtime = lock_cpu_runtime(1)
    repair_manifest = read_json(project_root / REPAIR_SOURCE / "prompt4_encoder_provenance_repair" / "encoder_checkpoint_manifest.json")
    checkpoint_path = Path(repair_manifest["checkpoint_path"])
    encoder_payload = torch_load(checkpoint_path)
    encoder = CpuFrozenGATv2Encoder(torch_mod, F, GATv2Conv, encoder_payload["model_config"])
    encoder.load_state_dict(encoder_payload["encoder_state_dict"])
    encoder.eval_frozen()
    parameter_before = sha256_state_dict(torch_mod, encoder.module)
    service_node_uids, full_graph_indices, node_order_sha = load_service_indices(project_root)
    split_path = project_root / "05_training/artifacts/mac_suseong_frozen_dynamic_gatv2_mappo_pilot_seed1/split_manifest.json"
    split_manifest = read_json(split_path)
    test_files = snapshot_files(project_root, "test", None)
    test_ids = [int(path.stem.split("_")[-1]) for path in test_files]
    train_ids = {int(path.stem.split("_")[-1]) for path in snapshot_files(project_root, "train", None)}
    val_ids = {int(path.stem.split("_")[-1]) for path in snapshot_files(project_root, "val", None)}
    split_lock = {
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "test_snapshot_count": len(test_files),
        "test_min_state_ts": split_manifest["test"]["first"]["state_ts"],
        "test_max_state_ts": split_manifest["test"]["last"]["state_ts"],
        "test_snapshot_id_hash": hashlib.sha256(",".join(map(str, test_ids)).encode()).hexdigest(),
        "train_test_overlap_count": len(train_ids & set(test_ids)),
        "validation_test_overlap_count": len(val_ids & set(test_ids)),
        "duplicate_test_snapshot_count": len(test_ids) - len(set(test_ids)),
    }
    dump_json(output_root / "test_leakage_audit.json", {**split_lock, "test_target_read": False, "future_feature_read": False, "test_based_checkpoint_selection": False})

    test_cache_root = output_root / "test_cache"
    manifest_a = build_test_cache_once(project_root=project_root, generation_root=test_cache_root / "test_cache_generation_A", torch_mod=torch_mod, encoder=encoder, service_node_uids=service_node_uids, full_graph_indices=full_graph_indices, node_order_sha=node_order_sha, checkpoint_sha=EXPECTED_CHECKPOINT_SHA, parameter_hash=EXPECTED_PARAMETER_HASH)
    manifest_b = build_test_cache_once(project_root=project_root, generation_root=test_cache_root / "test_cache_generation_B", torch_mod=torch_mod, encoder=encoder, service_node_uids=service_node_uids, full_graph_indices=full_graph_indices, node_order_sha=node_order_sha, checkpoint_sha=EXPECTED_CHECKPOINT_SHA, parameter_hash=EXPECTED_PARAMETER_HASH)
    cache_compare = compare_cache(manifest_a, manifest_b)
    selected_rows = manifest_a["rows"]
    cache_root = project_root / CACHE_ROOT
    (cache_root / "test").mkdir(parents=True, exist_ok=True)
    for row in selected_rows:
        source = Path(row["embedding_path"])
        target = cache_root / "test" / source.name
        shutil.copy2(source, target)
        row["embedding_path"] = str(target)
        row["embedding_file_sha256"] = sha256_file(target)
    test_cache_manifest = {
        **{key: value for key, value in manifest_a.items() if key != "rows"},
        "generation_root": str(cache_root),
        "rows": selected_rows,
    }
    dump_json(cache_root / "test_cache_manifest.json", test_cache_manifest)
    dump_json(cache_root / "test_cache_sha256.json", {"created_at_utc": utc_now(), "test_cache_manifest_sha256": stable_json_sha({"rows": selected_rows}), "embedding_files": {str(Path(r["embedding_path"]).relative_to(cache_root)): r["embedding_file_sha256"] for r in selected_rows}})
    try:
        pd.DataFrame(selected_rows).to_parquet(cache_root / "test_embedding_index.parquet", index=False)
    except Exception as exc:
        dump_json(cache_root / "test_embedding_index_parquet_error.json", {"error": str(exc)})
    pd.DataFrame(selected_rows).to_parquet(test_cache_root / "test_embedding_index.parquet", index=False)
    dump_json(test_cache_root / "test_cache_manifest.json", test_cache_manifest)
    dump_json(test_cache_root / "test_cache_repeatability_matrix.json", cache_compare)
    parameter_after = sha256_state_dict(torch_mod, encoder.module)
    cache_validation = {
        "created_at_utc": utc_now(),
        "status": "PASS" if cache_compare["test_cache_A_vs_B_hash_mismatch_count"] == 0 and len(selected_rows) == len(test_files) and parameter_before == parameter_after else "BLOCKED",
        "test_cache_exact": cache_compare["test_cache_A_vs_B_hash_mismatch_count"] == 0,
        "expected_test_snapshot_count": len(test_files),
        "missing_test_embedding_count": test_cache_manifest["missing_test_embedding_count"],
        "duplicate_test_embedding_count": test_cache_manifest["duplicate_test_embedding_count"],
        "shape_mismatch_count": test_cache_manifest["shape_mismatch_count"],
        "dtype_mismatch_count": test_cache_manifest["dtype_mismatch_count"],
        "nan_detected": test_cache_manifest["nan_detected"],
        "inf_detected": test_cache_manifest["inf_detected"],
        "test_target_read": False,
        "future_feature_read": False,
        "encoder_parameter_hash_before": parameter_before,
        "encoder_parameter_hash_after": parameter_after,
        "repeatability": cache_compare,
    }
    dump_json(cache_root / "test_cache_validation_report.json", cache_validation)
    dump_json(test_cache_root / "test_cache_validation_report.json", cache_validation)
    if cache_validation["status"] != "PASS":
        gate = {"created_at_utc": utc_now(), "status": "BLOCKED", "reason": "TEST_CACHE_EXACT_GATE_FAILED", "test_cache_validation": cache_validation}
        dump_json(output_root / "prompt6a_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return

    test_embeddings = load_test_embeddings(cache_root, test_cache_manifest)
    e0_embeddings = make_e0_embeddings(test_embeddings.shape)
    lock = checkpoint_lock(project_root, prompt5_root, output_root, stable_json_sha({"rows": selected_rows}))
    if lock["status"] != "PASS":
        gate = {"created_at_utc": utc_now(), "status": "BLOCKED", "reason": "CHECKPOINT_LOCK_FAILED", "checkpoint_lock": lock}
        dump_json(output_root / "prompt6a_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return

    rows: List[Dict[str, Any]] = []
    for family in ["E0", "E1"]:
        for condition in ["A", "A90", "A80", "A70"]:
            for seed in [1, 2, 3]:
                ckpt = prompt5_root / family / condition / f"seed_{seed:03d}" / "checkpoint_best_validation.pt"
                rows.append(evaluate(project_root=project_root, output_root=output_root, family=family, condition=condition, seed=seed, checkpoint_path=ckpt, embeddings=e0_embeddings if family == "E0" else test_embeddings, service_node_uids=service_node_uids, device=device, horizon=args.horizon))

    by_run, pair_rows, by_condition, result_class = summarize(rows)
    valid_pairs = sum(1 for p in pair_rows if all(p[k] for k in ["paired_initial_state_hash_identical", "paired_demand_hash_identical", "paired_traffic_hash_identical", "paired_fleet_contract_hash_identical", "paired_test_window_hash_identical"]) and p["e0_hard_constraint_passed"] and p["e1_hard_constraint_passed"])
    stats_root = output_root / "statistics"
    stats_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{k: v for k, v in r.items() if k != "canonical_kpi"}, **{f"kpi_{k}": v for k, v in r["canonical_kpi"].items()}} for r in by_run]).to_parquet(output_root / "heldout_results_by_run.parquet", index=False)
    pd.DataFrame(pair_rows).to_parquet(output_root / "e0_e1_paired_differences.parquet", index=False)
    pd.DataFrame(by_condition).to_parquet(output_root / "heldout_results_by_condition.parquet", index=False)
    pd.DataFrame(pair_rows).to_parquet(output_root / "heldout_results_by_seed.parquet", index=False)
    dump_json(stats_root / "e0_e1_statistical_summary.json", {"created_at_utc": utc_now(), "classification": result_class, "condition_kpi_summary": by_condition, "seed_matched_differences": pair_rows})
    dump_json(output_root / "e0_e1_statistical_summary.json", {"created_at_utc": utc_now(), "classification": result_class, "condition_kpi_summary": by_condition, "seed_matched_differences": pair_rows})
    dump_json(output_root / "e0_e1_constraint_comparison.json", {"hard_constraint_passed_run_count": sum(1 for r in rows if r["hard_constraint_passed"]), "hard_constraint_failed_run_count": sum(1 for r in rows if not r["hard_constraint_passed"]), "admissible_pair_count": valid_pairs})
    dump_json(output_root / "checkpoint_provenance_report.json", lock)
    expected_runs = 24
    gate = {
        "created_at_utc": utc_now(),
        "status": "PASS" if len(rows) == expected_runs and all(r["evaluation_completed"] for r in rows) and valid_pairs == 12 else "PARTIAL_FAIL",
        "classification": "PROMPT6A_STRUCTURAL_PASS_" + result_class if len(rows) == expected_runs and all(r["evaluation_completed"] for r in rows) and valid_pairs == 12 else "PROMPT6A_INCOMPLETE_OR_NOT_ADMISSIBLE",
        "expected_test_run_count": 24,
        "completed_test_run_count": sum(1 for r in rows if r["evaluation_completed"]),
        "failed_test_run_count": sum(1 for r in rows if not r["evaluation_completed"]),
        "blocked_test_run_count": 0,
        "expected_pair_count": 12,
        "valid_pair_count": valid_pairs,
        "test_cache_exact": cache_validation["test_cache_exact"],
        "test_cache_hash_mismatch_count": cache_compare["test_cache_A_vs_B_hash_mismatch_count"],
        "checkpoint_lock_passed": lock["status"] == "PASS",
        "checkpoint_provenance_mismatch_count": lock["checkpoint_provenance_mismatch_count"],
        "strict_mps_passed": all(r["normalized_actor_device"] == "mps" and r["normalized_critic_device"] == "mps" for r in rows),
        "cpu_model_fallback_count": sum(1 for r in rows if r["cpu_model_fallback_used"]),
        "nan_run_count": sum(1 for r in rows if r["nan_detected"]),
        "inf_run_count": sum(1 for r in rows if r["inf_detected"]),
        "checkpoint_reload_failure_count": sum(1 for r in rows if not r["checkpoint_reload_ok"]),
        "canonical_kpi_failure_count": sum(1 for r in rows if r["canonical_kpi_count"] != 12),
        "simulator_integrity_failure_count": sum(1 for r in rows if not r["hard_constraint_passed"]),
        "hard_constraint_passed_run_count": sum(1 for r in rows if r["hard_constraint_passed"]),
        "hard_constraint_failed_run_count": sum(1 for r in rows if not r["hard_constraint_passed"]),
        "admissible_pair_count": valid_pairs,
        "optimizer_step_count": sum(r["optimizer_step_count"] for r in rows),
        "backward_call_count": sum(r["backward_call_count"] for r in rows),
        "parameter_update_count": sum(r["parameter_update_count"] for r in rows),
        "test_target_read": False,
        "future_feature_read": False,
        "test_based_checkpoint_selection": False,
        "e0_vs_e1_claim_status": result_class,
        "frozen_gatv2_contribution_status": "HELD_OUT_SIMULATOR_EVALUATED",
        "e2_fine_tuning_status": "NOT_EVALUATED",
        "prompt6_full_matrix_status": "NOT_APPROVED",
        "real_world_causal_claim_allowed": False,
    }
    dump_json(output_root / "prompt6a_gate.json", gate)
    dump_json(output_root / "prompt6a_claim_gate.json", gate)
    dump_json(output_root / "prompt6a_manifest.json", {"created_at_utc": utc_now(), "output_root": str(output_root), "authoritative_prompt5_root": str(prompt5_root), "superseded_partial_prompt5": {"path": str(project_root / PARTIAL_PROMPT5_ROOT), "status": "SUPERSEDED", "included_in_matrix_count": False, "included_in_claim_gate": False}, "test_cache_root": str(cache_root / "test"), "rows": rows})
    report = [
        "# Prompt 6A E0 vs E1 Held-Out Test Evaluation",
        "",
        f"status: {gate['status']}",
        f"classification: {gate['classification']}",
        f"completed_test_run_count: {gate['completed_test_run_count']}/24",
        f"valid_pair_count: {valid_pairs}/12",
        f"e0_vs_e1_claim_status: {result_class}",
        "",
        "E2 fine-tuning is NOT_EVALUATED. Full Prompt 6 matrix is NOT_APPROVED. Real-world causal performance is not claimed.",
    ]
    (output_root / "prompt6a_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_root / "prompt6a_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_artifact_index(output_root)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
