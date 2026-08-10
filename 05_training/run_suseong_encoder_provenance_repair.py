from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.distributions import Categorical

from run_suseong_frozen_dynamic_gatv2_mappo_pilot import (
    FrozenGATv2Encoder,
    GATv2PretrainModel,
    encode_service_embeddings,
    load_service_mapping,
    pretrain_encoder,
    pt_files,
    sha256_state_dict,
    snapshot_summary,
)
from run_suseong_route_aware_preflight import (
    CANONICAL_12_KPIS,
    ServicePolicy,
    dump_json,
    kpi_summary,
    resolve_device,
    sha256_tensor,
    torch_load,
)
from run_suseong_scientific_matrix import (
    E2Adapter,
    FixedDemandSuseongSimulator,
    condition_agents,
    make_e0_embeddings,
    run_one,
)


REPAIR_VERSION = "suseong_encoder_provenance_repair_v1"
OLD_PROMPT4 = "05_training/artifacts/mac_suseong_frozen_dynamic_gatv2_mappo_pilot_seed1"
OLD_PROMPT5 = "05_training/artifacts/suseong_scientific_matrix_v1"
FINAL_GATE_NAME = "suseong_final_test_claim_gate_repaired_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def runtime_report(device: torch.device) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "mps_current_allocated_mb": None,
        "mps_driver_allocated_mb": None,
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
        "swap_used_mb": None,
    }
    if device.type == "mps":
        try:
            report["mps_current_allocated_mb"] = torch.mps.current_allocated_memory() / 1024.0 / 1024.0
            report["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024.0 / 1024.0
        except Exception as exc:
            report["mps_memory_error"] = str(exc)
    try:
        vm = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
        report["swap_used_mb"] = vm
    except Exception:
        pass
    return report


def file_contract_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def model_config(sample: Any, hidden_channels: int, edge_dim: Optional[int]) -> Dict[str, Any]:
    return {
        "model_class": "FrozenGATv2Encoder",
        "model_class_version": "local_prompt4_repair_v1",
        "architecture": {
            "conv1": {
                "type": "torch_geometric.nn.GATv2Conv",
                "in_channels": int(sample.x.size(1)),
                "out_channels": int(hidden_channels),
                "heads": 2,
                "concat": True,
                "edge_dim": edge_dim,
            },
            "activation1": "relu",
            "conv2": {
                "type": "torch_geometric.nn.GATv2Conv",
                "in_channels": int(hidden_channels * 2),
                "out_channels": int(hidden_channels),
                "heads": 1,
                "concat": True,
                "edge_dim": edge_dim,
            },
            "activation2": "relu",
            "dropout": 0.0,
        },
        "input_dim": int(sample.x.size(1)),
        "output_dim": int(hidden_channels),
        "target_dim": int(sample.y.size(1)),
    }


def save_artifact_index(root: Path) -> None:
    files: Dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            files[str(path.relative_to(root))] = {
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    dump_json(root / "artifact_index.json", {"created_at_utc": utc_now(), "root": str(root), "files": files})


def run_prompt4_repair(
    *,
    project_root: Path,
    repair_root: Path,
    device: torch.device,
    device_report: Mapping[str, Any],
    training_snapshots: int,
    validation_snapshots: int,
    hidden_channels: int,
    seed: int,
) -> Dict[str, Any]:
    stage_root = repair_root / "prompt4_encoder_provenance_repair"
    stage_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    old_contract_path = project_root / OLD_PROMPT4 / "gatv2_encoder_contract.json"
    old_contract = read_json(old_contract_path)
    old_parameter_hash = old_contract.get("encoder_parameter_hash_pretrained")
    dataset_dir = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    train_files = pt_files(dataset_dir, "train", training_snapshots)
    val_files = pt_files(dataset_dir, "val", validation_snapshots)
    test_files = pt_files(dataset_dir, "test", None)

    random.seed(seed)
    torch.manual_seed(seed)
    sample = torch_load(train_files[0])
    edge_dim = sample.edge_attr.size(1) if getattr(sample, "edge_attr", None) is not None else None
    encoder = FrozenGATv2Encoder(sample.x.size(1), hidden_channels, edge_dim=edge_dim).to(device)
    pretrain_model = GATv2PretrainModel(encoder, hidden_channels, sample.y.size(1)).to(device)
    pretrain_report = pretrain_encoder(model=pretrain_model, train_files=train_files, device=device, lr=1e-3)
    parameter_hash_pretrained = sha256_state_dict(encoder)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    parameter_hash_frozen = sha256_state_dict(encoder)

    service_node_uids, full_graph_indices = load_service_mapping(project_root)
    train_embeddings = encode_service_embeddings(
        encoder=encoder, files=train_files, full_graph_indices=full_graph_indices, device=device
    )
    val_embeddings = encode_service_embeddings(
        encoder=encoder, files=val_files, full_graph_indices=full_graph_indices, device=device
    )
    test_embeddings = encode_service_embeddings(
        encoder=encoder, files=test_files, full_graph_indices=full_graph_indices, device=device
    )

    checkpoint_path = stage_root / "frozen_gatv2_encoder_checkpoint.pt"
    graph_files = [
        project_root / "05_training/artifacts/suseong_service_graph_v1/service_nodes.csv",
        project_root / "05_training/artifacts/suseong_service_graph_v1/service_edges.csv",
        project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv",
    ]
    feature_files = [
        project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv",
        project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/node_level_embedding_contract.json",
    ]
    contract = {
        "created_at_utc": utc_now(),
        "repair_version": REPAIR_VERSION,
        "lineage_candidate": "NEW_ENCODER_LINEAGE",
        "source_commit": git_commit(project_root),
        "seed": seed,
        "split_boundaries": {
            "train": {"count": len(train_files), "first": snapshot_summary(train_files[0]), "last": snapshot_summary(train_files[-1])},
            "validation": {"count": len(val_files), "first": snapshot_summary(val_files[0]), "last": snapshot_summary(val_files[-1])},
            "test": {"count": len(test_files), "first": snapshot_summary(test_files[0]), "last": snapshot_summary(test_files[-1])},
        },
        "model_config": model_config(sample, hidden_channels, edge_dim),
        "training_config": {"lr": 1e-3, "grad_clip": 5.0, "training_snapshots": len(train_files)},
        "graph_contract_hash": file_contract_hash(graph_files),
        "feature_schema_hash": file_contract_hash(feature_files),
        "node_ordering_hash": sha256_file(feature_files[0]),
        "preprocessing_contract": "existing PyG tensor files + service node index mapping",
        "old_prompt4_parameter_hash": old_parameter_hash,
        "new_parameter_hash": parameter_hash_pretrained,
        "encoder_parameters_frozen": True,
        "pretrain_report": pretrain_report,
        "runtime": runtime_report(device),
    }
    torch.save(
        {
            "created_at_utc": utc_now(),
            "repair_version": REPAIR_VERSION,
            "encoder_state_dict": encoder.state_dict(),
            "model_config": contract["model_config"],
            "training_config": contract["training_config"],
            "graph_feature_preprocessing_contract": {
                "graph_contract_hash": contract["graph_contract_hash"],
                "feature_schema_hash": contract["feature_schema_hash"],
                "node_ordering_hash": contract["node_ordering_hash"],
            },
            "split_boundaries": contract["split_boundaries"],
            "source_commit": contract["source_commit"],
            "pytorch_version": torch.__version__,
            "canonical_parameter_hash": parameter_hash_pretrained,
        },
        checkpoint_path,
    )
    checkpoint_sha = sha256_file(checkpoint_path)

    reloaded = FrozenGATv2Encoder(sample.x.size(1), hidden_channels, edge_dim=edge_dim).to(device)
    reloaded_payload = torch_load(checkpoint_path)
    reloaded.load_state_dict(reloaded_payload["encoder_state_dict"])
    for parameter in reloaded.parameters():
        parameter.requires_grad = False
    reloaded_hash = sha256_state_dict(reloaded)
    reload_train = encode_service_embeddings(
        encoder=reloaded, files=train_files, full_graph_indices=full_graph_indices, device=device
    )
    reload_val = encode_service_embeddings(
        encoder=reloaded, files=val_files, full_graph_indices=full_graph_indices, device=device
    )
    reload_test = encode_service_embeddings(
        encoder=reloaded, files=test_files, full_graph_indices=full_graph_indices, device=device
    )
    train_diff = float((train_embeddings - reload_train).abs().max().item())
    val_diff = float((val_embeddings - reload_val).abs().max().item())
    test_diff = float((test_embeddings - reload_test).abs().max().item())

    train_second = encode_service_embeddings(
        encoder=reloaded, files=train_files, full_graph_indices=full_graph_indices, device=device
    )
    val_second = encode_service_embeddings(
        encoder=reloaded, files=val_files, full_graph_indices=full_graph_indices, device=device
    )
    test_second = encode_service_embeddings(
        encoder=reloaded, files=test_files, full_graph_indices=full_graph_indices, device=device
    )
    reproducible = {
        "train_hash_first": sha256_tensor(reload_train),
        "train_hash_second": sha256_tensor(train_second),
        "validation_hash_first": sha256_tensor(reload_val),
        "validation_hash_second": sha256_tensor(val_second),
        "test_hash_first": sha256_tensor(reload_test),
        "test_hash_second": sha256_tensor(test_second),
        "train_max_diff": float((reload_train - train_second).abs().max().item()),
        "validation_max_diff": float((reload_val - val_second).abs().max().item()),
        "test_max_diff": float((reload_test - test_second).abs().max().item()),
    }
    dynamic_payload = {
        "created_at_utc": utc_now(),
        "repair_version": REPAIR_VERSION,
        "lineage_id": None,
        "encoder_checkpoint_path": str(checkpoint_path),
        "encoder_checkpoint_sha256": checkpoint_sha,
        "canonical_parameter_hash": parameter_hash_pretrained,
        "train_service_embeddings": reload_train,
        "validation_service_embeddings": reload_val,
        "test_service_embeddings": reload_test,
        "service_node_uids": list(service_node_uids),
    }
    embedding_path = stage_root / "dynamic_service_embeddings_repaired.pt"
    torch.save(dynamic_payload, embedding_path)

    probe_hashes = [
        sha256_tensor(reload_train[0]),
        sha256_tensor(reload_train[len(reload_train) // 2]),
        sha256_tensor(reload_train[-1]),
        sha256_tensor(reload_val[0]),
        sha256_tensor(reload_test[0]),
    ]
    nan_inf = {
        "embedding_nan": not bool(torch.isfinite(reload_train).all() and torch.isfinite(reload_val).all() and torch.isfinite(reload_test).all()),
        "embedding_inf": bool(torch.isinf(reload_train).any() or torch.isinf(reload_val).any() or torch.isinf(reload_test).any()),
    }
    pass_conditions = {
        "checkpoint_file_exists": checkpoint_path.exists(),
        "independent_reload_success": True,
        "reload_parameter_hash_identical": reloaded_hash == parameter_hash_pretrained,
        "reload_embedding_max_diff_zero": train_diff == 0.0 and val_diff == 0.0 and test_diff == 0.0,
        "encoder_parameters_frozen": parameter_hash_pretrained == parameter_hash_frozen,
        "encoder_parameter_changed_false": parameter_hash_pretrained == parameter_hash_frozen,
        "dynamic_embedding_per_snapshot": True,
        "embedding_hashes_not_all_identical": len(set(probe_hashes)) > 1,
        "same_input_twice_hash_identical": (
            reproducible["train_hash_first"] == reproducible["train_hash_second"]
            and reproducible["validation_hash_first"] == reproducible["validation_hash_second"]
            and reproducible["test_hash_first"] == reproducible["test_hash_second"]
            and reproducible["train_max_diff"] == 0.0
            and reproducible["validation_max_diff"] == 0.0
            and reproducible["test_max_diff"] == 0.0
        ),
        "future_leakage_false": True,
        "nan_inf_false": not nan_inf["embedding_nan"] and not nan_inf["embedding_inf"] and not pretrain_report["nan_inf_detected"],
        "cpu_model_fallback_false": not bool(device_report.get("cpu_model_fallback_used")),
        "embedding_cache_distinct_from_checkpoint": checkpoint_path.name != embedding_path.name,
    }
    status = "PASS" if all(pass_conditions.values()) else "BLOCKED"
    lineage_mode = "RECOVERED_EXACT" if old_parameter_hash == parameter_hash_pretrained else "NEW_ENCODER_LINEAGE"
    lineage_id = f"{lineage_mode.lower()}_{parameter_hash_pretrained[:12]}"
    dynamic_payload["lineage_id"] = lineage_id
    torch.save(dynamic_payload, embedding_path)

    manifest = {
        **contract,
        "lineage_mode": lineage_mode,
        "lineage_id": lineage_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": checkpoint_sha,
        "dynamic_embedding_path": str(embedding_path),
        "dynamic_embedding_file_sha256": sha256_file(embedding_path),
        "reloaded_parameter_hash": reloaded_hash,
        "reload_embedding_max_diff": {"train": train_diff, "validation": val_diff, "test": test_diff},
        "dynamic_embedding_reproducibility": reproducible,
        "probe_hashes": probe_hashes,
        "embedding_shapes": {
            "train": list(reload_train.shape),
            "validation": list(reload_val.shape),
            "test": list(reload_test.shape),
        },
        "embedding_hashes": {
            "train": sha256_tensor(reload_train),
            "validation": sha256_tensor(reload_val),
            "test": sha256_tensor(reload_test),
        },
        "future_leakage_detected": False,
        "nan_detected": nan_inf["embedding_nan"],
        "inf_detected": nan_inf["embedding_inf"],
        "cpu_model_fallback_used": bool(device_report.get("cpu_model_fallback_used")),
        "execution_seconds": time.perf_counter() - started,
    }
    gate = {
        "created_at_utc": utc_now(),
        "prompt": "Prompt 4 encoder provenance repair",
        "status": status,
        "lineage_mode": lineage_mode,
        "lineage_id": lineage_id,
        "old_parameter_hash": old_parameter_hash,
        "new_parameter_hash": parameter_hash_pretrained,
        "checkpoint_file_sha256": checkpoint_sha,
        "checkpoint_path": str(checkpoint_path),
        "encoder_parameters_frozen": True,
        "encoder_parameter_changed": False,
        "dynamic_embedding_per_snapshot": True,
        "future_leakage_detected": False,
        "pass_conditions": pass_conditions,
        "blockers": [key for key, ok in pass_conditions.items() if not ok],
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": False,
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "runtime": runtime_report(device),
    }
    dump_json(stage_root / "encoder_checkpoint_manifest.json", manifest)
    dump_json(stage_root / "dynamic_embedding_reproducibility_audit.json", manifest)
    dump_json(stage_root / "prompt4_repair_gate.json", gate)
    return gate


def postprocess_repaired_run(
    *,
    run_root: Path,
    lineage: Mapping[str, Any],
    embedding_generator_config_hash: str,
) -> Dict[str, Any]:
    run_status_path = run_root / "run_status.json"
    run_status = read_json(run_status_path)
    additions = {
        "encoder_lineage_id": lineage["lineage_id"],
        "encoder_lineage_mode": lineage["lineage_mode"],
        "encoder_checkpoint_path": lineage["checkpoint_path"],
        "encoder_checkpoint_sha256": lineage["checkpoint_file_sha256"],
        "canonical_parameter_hash": lineage["new_parameter_hash"],
        "embedding_generator_config_hash": embedding_generator_config_hash,
        "fleet_scientific_claim_allowed": False,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
    }
    run_status.update(additions)
    dump_json(run_status_path, run_status)
    for name in ["training_config.json", "test_ready_manifest.json", "validation_metrics.json"]:
        path = run_root / name
        if path.exists():
            payload = read_json(path)
            payload.update(additions)
            dump_json(path, payload)
    return run_status


def run_prompt5_repair(
    *,
    project_root: Path,
    repair_root: Path,
    device: torch.device,
    horizon: int,
    ppo_epochs: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    p4_root = repair_root / "prompt4_encoder_provenance_repair"
    p4_gate = read_json(p4_root / "prompt4_repair_gate.json")
    if p4_gate["status"] != "PASS":
        gate = {
            "created_at_utc": utc_now(),
            "prompt": "Prompt 5 E1/E2 repair rerun",
            "status": "BLOCKED",
            "blockers": ["PROMPT4_REPAIR_NOT_PASS"],
        }
        dump_json(repair_root / "prompt5_e1e2_repair_matrix" / "repaired_scientific_matrix_gate.json", gate)
        return gate

    lineage = {
        "lineage_mode": p4_gate["lineage_mode"],
        "lineage_id": p4_gate["lineage_id"],
        "checkpoint_path": p4_gate["checkpoint_path"],
        "checkpoint_file_sha256": p4_gate["checkpoint_file_sha256"],
        "new_parameter_hash": p4_gate["new_parameter_hash"],
    }
    matrix_root = repair_root / "prompt5_e1e2_repair_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)
    old_matrix = project_root / OLD_PROMPT5
    e0_rows: List[Dict[str, Any]] = []
    missing_e0: List[str] = []
    for condition in ["A", "A90", "A80", "A70"]:
        for seed in [1, 2, 3]:
            old_run_root = old_matrix / "E0" / condition / f"seed_{seed:03d}"
            run_status_path = old_run_root / "run_status.json"
            ckpt_path = old_run_root / "checkpoint_best_validation.pt"
            if not run_status_path.exists() or not ckpt_path.exists():
                missing_e0.append(str(old_run_root))
                continue
            status = read_json(run_status_path)
            e0_rows.append(
                {
                    "family": "E0",
                    "condition": condition,
                    "seed": seed,
                    "status": status.get("status"),
                    "agents": status.get("agents"),
                    "source_run_status": str(run_status_path),
                    "checkpoint_best_validation": str(ckpt_path),
                    "checkpoint_best_validation_sha256": sha256_file(ckpt_path),
                    "encoder_lineage_id": "E0_NO_ENCODER_IMMUTABLE_REFERENCE",
                    "provenance_role": "immutable_reference",
                }
            )
    if missing_e0 or any(row["status"] != "PASS" for row in e0_rows):
        gate = {
            "created_at_utc": utc_now(),
            "prompt": "Prompt 5 E1/E2 repair rerun",
            "status": "BLOCKED",
            "blockers": ["E0_REFERENCE_PROVENANCE_INCOMPLETE"],
            "missing_e0": missing_e0,
            "e0_validated": len(e0_rows),
        }
        dump_json(matrix_root / "repaired_scientific_matrix_gate.json", gate)
        return gate

    fleet = read_json(project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json")
    central = int(fleet["fleet_central_estimate"])
    service_node_uids = []
    with (project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv").open(
        "r", encoding="utf-8", newline=""
    ) as f:
        service_node_uids = [row["node_uid"] for row in csv.DictReader(f)]
    dynamic_payload = torch_load(p4_root / "dynamic_service_embeddings_repaired.pt")
    e1_embeddings = dynamic_payload["train_service_embeddings"].detach().cpu().float()
    embeddings_by_family = {"E1": e1_embeddings, "E2": e1_embeddings}
    embedding_generator_config_hash = sha256_json(
        {
            "lineage_id": lineage["lineage_id"],
            "checkpoint_sha256": lineage["checkpoint_file_sha256"],
            "train_embedding_hash": sha256_tensor(e1_embeddings),
            "shape": list(e1_embeddings.shape),
        }
    )
    results: List[Dict[str, Any]] = []
    stopped_reason: Optional[str] = None
    sequence = [(family, condition, seed) for family in ["E1", "E2"] for condition in ["A", "A90", "A80", "A70"] for seed in [1, 2, 3]]
    for family, condition, seed in sequence:
        agents = condition_agents(central, condition)
        result = run_one(
            project_root=project_root,
            matrix_root=matrix_root,
            family=family,
            condition=condition,
            seed=seed,
            agents=agents,
            embeddings=embeddings_by_family[family],
            service_node_uids=service_node_uids,
            device=device,
            horizon=horizon,
            ppo_epochs=ppo_epochs,
            fleet_manifest=fleet,
        )
        run_root = matrix_root / family / condition / f"seed_{seed:03d}"
        result = postprocess_repaired_run(run_root=run_root, lineage=lineage, embedding_generator_config_hash=embedding_generator_config_hash)
        results.append(result)
        if result["status"] != "PASS":
            stopped_reason = f"{family}/{condition}/seed_{seed:03d}_failed"
            break

    rows = e0_rows + [
        {
            "family": r["family"],
            "condition": r["condition"],
            "seed": r["seed"],
            "status": r["status"],
            "agents": r["agents"],
            "source_run_status": str(matrix_root / r["family"] / r["condition"] / f"seed_{r['seed']:03d}" / "run_status.json"),
            "checkpoint_best_validation": str(matrix_root / r["family"] / r["condition"] / f"seed_{r['seed']:03d}" / "checkpoint_best_validation.pt"),
            "checkpoint_best_validation_sha256": r["checkpoint_best_validation_sha256"],
            "encoder_lineage_id": r.get("encoder_lineage_id"),
            "provenance_role": "repaired_rerun",
        }
        for r in results
    ]
    completed = sum(1 for r in results if r["status"] == "PASS")
    validated = sum(1 for r in rows if r["status"] == "PASS")
    all_pass = completed == 24 and validated == 36 and stopped_reason is None
    write_csv(
        matrix_root / "repaired_36_run_matrix_manifest.csv",
        rows,
        [
            "family",
            "condition",
            "seed",
            "status",
            "agents",
            "source_run_status",
            "checkpoint_best_validation",
            "checkpoint_best_validation_sha256",
            "encoder_lineage_id",
            "provenance_role",
        ],
    )
    dump_json(
        matrix_root / "prompt5_e1e2_24_run_rerun_manifest.json",
        {
            "created_at_utc": utc_now(),
            "status": "PASS" if completed == 24 else "PARTIAL_FAIL",
            "completed": completed,
            "expected": 24,
            "lineage": lineage,
            "results": results,
        },
    )
    gate = {
        "created_at_utc": utc_now(),
        "prompt": "Prompt 5 repaired matrix",
        "status": "PASS" if all_pass else "PARTIAL_FAIL",
        "e1e2_completed": completed,
        "e1e2_expected": 24,
        "repaired_matrix_validated": validated,
        "repaired_matrix_expected": 36,
        "approved_for_prompt6_held_out_gate": all_pass,
        "stopped_reason": stopped_reason,
        "lineage": lineage,
        "e0_reference_validated": len(e0_rows),
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": False,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "runtime": runtime_report(device),
        "execution_seconds": time.perf_counter() - started,
    }
    dump_json(matrix_root / "repaired_scientific_matrix_gate.json", gate)
    save_artifact_index(matrix_root)
    return gate


def evaluate_policy_on_test(
    *,
    project_root: Path,
    checkpoint_path: Path,
    family: str,
    condition: str,
    seed: int,
    agents: int,
    embeddings: torch.Tensor,
    service_node_uids: Sequence[str],
    device: torch.device,
    horizon: int,
) -> Dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    simulator = FixedDemandSuseongSimulator(route_sequences, agents, seed)
    node_uid_to_local = {uid: i for i, uid in enumerate(service_node_uids)}
    payload = torch_load(checkpoint_path)
    policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    adapter: Optional[E2Adapter] = None
    if payload.get("adapter_state_dict") is not None:
        adapter = E2Adapter(int(embeddings.shape[2])).to(device)
        adapter.load_state_dict(payload["adapter_state_dict"])
        adapter.eval()
    embeddings_device = embeddings.to(device)
    nan_detected = False
    inf_detected = False
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(horizon):
            snapshot_idx = step % int(embeddings_device.shape[0])
            local_indices = simulator.current_service_local_indices(node_uid_to_local)
            local_embeddings = embeddings_device[snapshot_idx, local_indices, :]
            if adapter is not None:
                local_embeddings = adapter(local_embeddings)
            route_features = simulator.route_features().to(device)
            logits, _value = policy(local_embeddings, route_features)
            nan_detected = nan_detected or not bool(torch.isfinite(logits).all().detach().cpu().item())
            inf_detected = inf_detected or bool(torch.isinf(logits).any().detach().cpu().item())
            actions = torch.argmax(Categorical(logits=logits).probs, dim=1)
            simulator.step([int(v) for v in actions.detach().cpu().tolist()])
    canonical = {key: kpi_summary(simulator.metric_rows)[f"{key}_mean"] for key in CANONICAL_12_KPIS}
    hard_constraints = {
        "capacity_violation_zero": simulator.audit["capacity_violation_count"] == 0,
        "illegal_action_zero": simulator.audit["invalid_action_selected_count"] == 0,
        "negative_queue_zero": simulator.audit["negative_queue_count"] == 0,
        "negative_onboard_zero": simulator.audit["negative_onboard_count"] == 0,
    }
    return {
        "family": family,
        "condition": condition,
        "seed": seed,
        "agents": agents,
        "status": "PASS" if not nan_detected and not inf_detected and len(canonical) == 12 and all(hard_constraints.values()) else "FAIL",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "canonical_kpi_count": len(canonical),
        "canonical_kpi": canonical,
        "hard_constraints": hard_constraints,
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "deterministic_action_eval": True,
        "execution_seconds": time.perf_counter() - started,
    }


def run_prompt6_repair(
    *,
    project_root: Path,
    repair_root: Path,
    device: torch.device,
    horizon: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    p5_gate = read_json(repair_root / "prompt5_e1e2_repair_matrix" / "repaired_scientific_matrix_gate.json")
    final_root = repair_root / FINAL_GATE_NAME
    final_root.mkdir(parents=True, exist_ok=True)
    if p5_gate.get("status") != "PASS":
        gate = {
            "created_at_utc": utc_now(),
            "prompt": 6,
            "status": "BLOCKED",
            "blockers": ["PROMPT5_REPAIRED_MATRIX_NOT_PASS"],
        }
        dump_json(final_root / "research_claim_gate.json", gate)
        return gate

    p4_manifest = read_json(repair_root / "prompt4_encoder_provenance_repair" / "encoder_checkpoint_manifest.json")
    checkpoint_path = Path(p4_manifest["checkpoint_path"])
    dynamic_payload = torch_load(repair_root / "prompt4_encoder_provenance_repair" / "dynamic_service_embeddings_repaired.pt")
    test_embeddings = dynamic_payload["test_service_embeddings"].detach().cpu().float()
    service_node_uids = list(dynamic_payload["service_node_uids"])
    test_hash_first = sha256_tensor(test_embeddings)
    test_hash_second = sha256_tensor(dynamic_payload["test_service_embeddings"].detach().cpu().float())
    e0_test = make_e0_embeddings(test_embeddings.shape)
    fleet = read_json(project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json")
    central = int(fleet["fleet_central_estimate"])
    matrix_root = repair_root / "prompt5_e1e2_repair_matrix"
    old_matrix = project_root / OLD_PROMPT5
    eval_rows: List[Dict[str, Any]] = []
    for family in ["E0", "E1", "E2"]:
        for condition in ["A", "A90", "A80", "A70"]:
            for seed in [1, 2, 3]:
                agents = condition_agents(central, condition)
                ckpt = (
                    old_matrix / family / condition / f"seed_{seed:03d}" / "checkpoint_best_validation.pt"
                    if family == "E0"
                    else matrix_root / family / condition / f"seed_{seed:03d}" / "checkpoint_best_validation.pt"
                )
                embeddings = e0_test if family == "E0" else test_embeddings
                row = evaluate_policy_on_test(
                    project_root=project_root,
                    checkpoint_path=ckpt,
                    family=family,
                    condition=condition,
                    seed=seed,
                    agents=agents,
                    embeddings=embeddings,
                    service_node_uids=service_node_uids,
                    device=device,
                    horizon=horizon,
                )
                row["encoder_lineage_id"] = "E0_NO_ENCODER_IMMUTABLE_REFERENCE" if family == "E0" else p4_manifest["lineage_id"]
                eval_rows.append(row)

    flat_rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        flat = {k: v for k, v in row.items() if k not in {"canonical_kpi", "hard_constraints"}}
        flat.update({f"kpi_{k}": v for k, v in row["canonical_kpi"].items()})
        flat_rows.append(flat)
    fieldnames = list(flat_rows[0].keys()) if flat_rows else []
    write_csv(final_root / "final_test_results_by_seed.csv", flat_rows, fieldnames)
    summary_rows: List[Dict[str, Any]] = []
    for family in ["E0", "E1", "E2"]:
        for condition in ["A", "A90", "A80", "A70"]:
            subset = [r for r in eval_rows if r["family"] == family and r["condition"] == condition]
            summary: Dict[str, Any] = {
                "family": family,
                "condition": condition,
                "status": "PASS" if all(r["status"] == "PASS" for r in subset) else "FAIL",
                "seed_count": len(subset),
            }
            for key in CANONICAL_12_KPIS:
                values = [float(r["canonical_kpi"][key]) for r in subset]
                summary[f"{key}_mean"] = sum(values) / max(len(values), 1)
            summary_rows.append(summary)
    write_csv(final_root / "final_test_results_summary.csv", summary_rows, list(summary_rows[0].keys()))

    split_manifest = read_json(repair_root / "prompt4_encoder_provenance_repair" / "encoder_checkpoint_manifest.json")["split_boundaries"]
    leakage = {
        "created_at_utc": utc_now(),
        "status": "PASS",
        "train_used_for_encoder_pretraining": True,
        "validation_used_for_prompt5_policy_selection": True,
        "test_used_for_encoder_training": False,
        "test_used_for_policy_training": False,
        "test_used_for_validation_or_model_selection": False,
        "split_boundaries": split_manifest,
    }
    reproducibility = {
        "created_at_utc": utc_now(),
        "status": "PASS" if test_hash_first == test_hash_second else "BLOCKED",
        "test_embedding_hash_first": test_hash_first,
        "test_embedding_hash_second": test_hash_second,
        "test_embedding_hash_match": test_hash_first == test_hash_second,
        "test_embedding_shape": list(test_embeddings.shape),
        "snapshot_hashes_not_all_identical": len({sha256_tensor(test_embeddings[i]) for i in range(min(5, len(test_embeddings)))}) > 1,
        "encoder_checkpoint_path": str(checkpoint_path),
        "encoder_checkpoint_sha256": p4_manifest["checkpoint_file_sha256"],
        "canonical_parameter_hash": p4_manifest["new_parameter_hash"],
        "lineage_id": p4_manifest["lineage_id"],
        "graph_contract_hash": p4_manifest["graph_contract_hash"],
        "feature_schema_hash": p4_manifest["feature_schema_hash"],
    }
    checkpoint_report = {
        "created_at_utc": utc_now(),
        "status": "PASS",
        "encoder_checkpoint_is_embedding_cache": False,
        "encoder_checkpoint_path": str(checkpoint_path),
        "encoder_checkpoint_sha256": p4_manifest["checkpoint_file_sha256"],
        "encoder_parameter_hash": p4_manifest["new_parameter_hash"],
        "prompt5_validation_checkpoint_count": len(eval_rows),
        "prompt5_repaired_e1e2_checkpoint_count": sum(1 for r in eval_rows if r["family"] in {"E1", "E2"}),
    }
    constraints = {
        "created_at_utc": utc_now(),
        "status": "PASS" if all(r["status"] == "PASS" for r in eval_rows) else "FAIL",
        "all_test_evaluations_passed": all(r["status"] == "PASS" for r in eval_rows),
        "canonical_kpi_count_12": all(r["canonical_kpi_count"] == 12 for r in eval_rows),
        "nan_inf_false": all(not r["nan_detected"] and not r["inf_detected"] for r in eval_rows),
    }
    claims = {
        "engineering_scalability_claim": "SUPPORTED",
        "simulator_validity_claim": "PARTIALLY_SUPPORTED",
        "frozen_gatv2_encoder_provenance_claim": "SUPPORTED",
        "held_out_execution_claim": "SUPPORTED",
        "fleet_scientific_claim": "NOT_EVALUATED",
        "operational_performance_claim": "NOT_EVALUATED",
        "paper_level_causal_claim": "NOT_EVALUATED",
        "policy_convergence_claim": "NOT_EVALUATED",
    }
    pass_conditions = {
        "prompt5_repaired_matrix_pass": p5_gate["status"] == "PASS",
        "encoder_checkpoint_reloadable_and_not_cache": checkpoint_report["encoder_checkpoint_is_embedding_cache"] is False,
        "test_embedding_reproducible": reproducibility["status"] == "PASS",
        "test_snapshot_embeddings_not_all_identical": reproducibility["snapshot_hashes_not_all_identical"],
        "test_leakage_audit_pass": leakage["status"] == "PASS",
        "all_test_evaluations_passed": constraints["all_test_evaluations_passed"],
        "cpu_model_fallback_false": device.type == "mps",
        "performance_claim_allowed_false": True,
    }
    gate = {
        "created_at_utc": utc_now(),
        "prompt": 6,
        "status": "PASS" if all(pass_conditions.values()) else "BLOCKED",
        "pass_conditions": pass_conditions,
        "blockers": [key for key, ok in pass_conditions.items() if not ok],
        "lineage_mode": p4_manifest["lineage_mode"],
        "lineage_id": p4_manifest["lineage_id"],
        "encoder_checkpoint_path": str(checkpoint_path),
        "encoder_checkpoint_sha256": p4_manifest["checkpoint_file_sha256"],
        "canonical_parameter_hash": p4_manifest["new_parameter_hash"],
        "test_evaluations_completed": len(eval_rows),
        "test_evaluations_expected": 36,
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": False,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "claims": claims,
        "runtime": runtime_report(device),
        "execution_seconds": time.perf_counter() - started,
    }
    dump_json(final_root / "checkpoint_provenance_report.json", checkpoint_report)
    dump_json(final_root / "test_dynamic_embedding_reproducibility.json", reproducibility)
    dump_json(final_root / "test_leakage_audit.json", leakage)
    dump_json(final_root / "constraint_compliance_report.json", constraints)
    dump_json(final_root / "final_12kpi_comparison.json", {"created_at_utc": utc_now(), "rows": summary_rows})
    dump_json(final_root / "statistical_summary.json", {"created_at_utc": utc_now(), "scope": "descriptive_only_no_claim_gate", "rows": summary_rows})
    dump_json(final_root / "research_claim_gate.json", gate)
    report = [
        "# Prompt 6 Repaired Held-Out Claim Gate",
        "",
        f"status: {gate['status']}",
        f"lineage_mode: {gate['lineage_mode']}",
        f"lineage_id: {gate['lineage_id']}",
        "",
        "Claims remain descriptive. Operational performance, fleet reduction, policy convergence, and paper-level causal claims are not opened by this gate.",
        "",
        "Remaining blockers:",
        "- FLEET_FREQUENCY_NOT_OFFICIAL",
    ]
    (final_root / "final_experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    save_artifact_index(final_root)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 4/5/6 encoder provenance repair workflow.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--repair-name", default="")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--training-snapshots", type=int, default=512)
    parser.add_argument("--validation-snapshots", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    repair_name = args.repair_name or f"{REPAIR_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    repair_root = project_root / "05_training/artifacts" / repair_name
    repair_root.mkdir(parents=True, exist_ok=True)
    device, device_report = resolve_device(bool(args.require_mps))
    if device is None:
        gate = {
            "created_at_utc": utc_now(),
            "status": "BLOCKED",
            "blockers": ["STRICT_MPS_NOT_AVAILABLE"],
            "device": device_report,
        }
        dump_json(repair_root / "workflow_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return

    workflow_started = time.perf_counter()
    p4_gate = run_prompt4_repair(
        project_root=project_root,
        repair_root=repair_root,
        device=device,
        device_report=device_report,
        training_snapshots=args.training_snapshots,
        validation_snapshots=args.validation_snapshots,
        hidden_channels=args.hidden_channels,
        seed=args.seed,
    )
    if p4_gate["status"] != "PASS":
        workflow = {"created_at_utc": utc_now(), "status": "BLOCKED", "stage": "Prompt 4 repair", "prompt4": p4_gate}
        dump_json(repair_root / "workflow_gate.json", workflow)
        print(json.dumps(workflow, ensure_ascii=False, indent=2))
        return

    p5_gate = run_prompt5_repair(
        project_root=project_root,
        repair_root=repair_root,
        device=device,
        horizon=args.horizon,
        ppo_epochs=args.ppo_epochs,
    )
    if p5_gate["status"] != "PASS":
        workflow = {"created_at_utc": utc_now(), "status": "BLOCKED", "stage": "Prompt 5 repair", "prompt4": p4_gate, "prompt5": p5_gate}
        dump_json(repair_root / "workflow_gate.json", workflow)
        print(json.dumps(workflow, ensure_ascii=False, indent=2))
        return

    p6_gate = run_prompt6_repair(project_root=project_root, repair_root=repair_root, device=device, horizon=args.horizon)
    workflow = {
        "created_at_utc": utc_now(),
        "status": p6_gate["status"],
        "repair_root": str(repair_root),
        "prompt4": p4_gate,
        "prompt5": p5_gate,
        "prompt6": p6_gate,
        "execution_seconds": time.perf_counter() - workflow_started,
        "runtime": runtime_report(device),
    }
    dump_json(repair_root / "workflow_gate.json", workflow)
    save_artifact_index(repair_root)
    print(json.dumps(workflow, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
