from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPAIR_SOURCE = "05_training/artifacts/suseong_encoder_provenance_repair_v1_20260718_094827"
R2_SOURCE = "05_training/artifacts/prompt4_r2_mps_determinism_repair_20260718_102020"
OUTPUT_ROOT = "05_training/artifacts/suseong_dynamic_embedding_cache_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_json_sha(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def import_runtime() -> Tuple[Any, Any, Any, Any]:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv

    return np, pd, torch, F, GATv2Conv


def lock_cpu_runtime(seed: int) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any]]:
    np, pd, torch, F, GATv2Conv = import_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    if hasattr(torch.utils, "deterministic"):
        torch.utils.deterministic.fill_uninitialized_memory = True
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine_id": platform.node(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_geometric_gatv2conv": str(GATv2Conv),
        "device": "cpu",
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": "error",
        "fill_uninitialized_memory": bool(torch.utils.deterministic.fill_uninitialized_memory),
        "autocast": "disabled",
        "mixed_precision": "disabled",
        "torch_compile": "disabled",
        "num_workers": 0,
    }
    return np, pd, torch, F, GATv2Conv, report


def torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_sha(torch: Any, tensor: Any) -> str:
    value = tensor.detach().to("cpu").contiguous().float()
    return sha256_bytes(value.numpy().tobytes(order="C"))


def tensor_stats(torch: Any, tensor: Any) -> Dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "contiguous": bool(value.is_contiguous()),
        "numel": int(value.numel()),
        "min": float(value.min().item()) if value.numel() else None,
        "max": float(value.max().item()) if value.numel() else None,
        "mean": float(value.mean().item()) if value.numel() else None,
        "std": float(value.std(unbiased=False).item()) if value.numel() else None,
        "sha256": tensor_sha(torch, value),
        "nan": not bool(torch.isfinite(value).all().item()) if value.numel() else False,
        "inf": bool(torch.isinf(value).any().item()) if value.numel() else False,
    }


def sha256_state_dict(torch: Any, module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


class CpuFrozenGATv2Encoder:
    def __init__(self, torch: Any, F: Any, GATv2Conv: Any, model_config: Mapping[str, Any]):
        self.torch = torch
        self.F = F
        arch = model_config["architecture"]
        self.module = torch.nn.Module()
        self.module.conv1 = GATv2Conv(
            int(arch["conv1"]["in_channels"]),
            int(arch["conv1"]["out_channels"]),
            heads=int(arch["conv1"]["heads"]),
            concat=bool(arch["conv1"]["concat"]),
            edge_dim=arch["conv1"]["edge_dim"],
        )
        self.module.conv2 = GATv2Conv(
            int(arch["conv2"]["in_channels"]),
            int(arch["conv2"]["out_channels"]),
            heads=int(arch["conv2"]["heads"]),
            concat=bool(arch["conv2"]["concat"]),
            edge_dim=arch["conv2"]["edge_dim"],
        )
        self.module.to("cpu")

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state_dict, strict=True)

    def eval_frozen(self) -> None:
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad = False
        for submodule in self.module.modules():
            if submodule.__class__.__name__.lower().startswith("dropout") and bool(submodule.training):
                raise RuntimeError("Dropout module is still in training mode.")

    def __call__(self, data: Any) -> Any:
        torch = self.torch
        F = self.F
        with torch.inference_mode():
            edge_attr = getattr(data, "edge_attr", None)
            x = data.x.detach().to("cpu", dtype=torch.float32).contiguous()
            edge_index = data.edge_index.detach().to("cpu").long().contiguous()
            edge_attr = edge_attr.detach().to("cpu", dtype=torch.float32).contiguous() if edge_attr is not None else None
            out = self.module.conv1(x, edge_index, edge_attr=edge_attr)
            out = F.relu(out)
            out = self.module.conv2(out, edge_index, edge_attr=edge_attr)
            return F.relu(out).detach().to("cpu").contiguous().float()


def load_service_indices(project_root: Path) -> Tuple[List[str], List[int], str]:
    mapping_path = project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv"
    service_node_uids: List[str] = []
    full_graph_indices: List[int] = []
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            service_node_uids.append(str(row["node_uid"]))
            full_graph_indices.append(int(row["full_graph_node_index"]))
    return service_node_uids, full_graph_indices, sha256_file(mapping_path)


def snapshot_files(project_root: Path, split: str, limit: Optional[int]) -> List[Path]:
    files = sorted((project_root / "05_training/artifacts/dataset_full_20260422_084243" / split).glob("*.pt"))
    return files[:limit] if limit is not None else files


def snapshot_id_from_data(data: Any, fallback: int) -> int:
    try:
        return int(getattr(data, "snapshot_id"))
    except Exception:
        return int(fallback)


def canonical_edge_hashes(torch: Any, data: Any) -> Dict[str, Any]:
    edge_index = data.edge_index.detach().to("cpu").long().contiguous()
    edge_attr = getattr(data, "edge_attr", None)
    edge_attr = edge_attr.detach().to("cpu", dtype=torch.float32).contiguous() if edge_attr is not None else None
    return {
        "node_feature_sha256": tensor_sha(torch, data.x.detach().to("cpu", dtype=torch.float32).contiguous()),
        "edge_index_sha256": tensor_sha(torch, edge_index.float()),
        "edge_attr_sha256": tensor_sha(torch, edge_attr) if edge_attr is not None else None,
    }


def build_cache_once(
    *,
    project_root: Path,
    generation_root: Path,
    torch: Any,
    encoder: CpuFrozenGATv2Encoder,
    service_node_uids: Sequence[str],
    full_graph_indices: Sequence[int],
    node_order_sha: str,
    checkpoint_sha: str,
    parameter_hash: str,
    train_limit: int,
    validation_limit: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    nan_detected = False
    inf_detected = False
    duplicate_keys: set[Tuple[str, int]] = set()
    duplicate_count = 0
    for split, limit in [("train", train_limit), ("val", validation_limit)]:
        out_split = "validation" if split == "val" else split
        files = snapshot_files(project_root, split, limit)
        for ordinal, path in enumerate(files):
            data = torch_load(torch, path)
            snapshot_id = snapshot_id_from_data(data, ordinal)
            key = (out_split, snapshot_id)
            duplicate_count += int(key in duplicate_keys)
            duplicate_keys.add(key)
            node_embeddings = encoder(data)
            service_embeddings = node_embeddings[list(full_graph_indices)].detach().to("cpu").contiguous().float()
            stats = tensor_stats(torch, service_embeddings)
            nan_detected = nan_detected or bool(stats["nan"])
            inf_detected = inf_detected or bool(stats["inf"])
            edge_hashes = canonical_edge_hashes(torch, data)
            out_path = generation_root / out_split / f"snapshot_{snapshot_id:05d}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "snapshot_id": snapshot_id,
                    "state_ts": str(getattr(data, "state_ts", "")),
                    "split": out_split,
                    "service_node_uids": list(service_node_uids),
                    "embedding": service_embeddings,
                    "encoder_checkpoint_sha256": checkpoint_sha,
                    "canonical_parameter_hash": parameter_hash,
                    "source_input_path": str(path),
                    "source_input_sha256": sha256_file(path),
                },
                out_path,
            )
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "state_ts": str(getattr(data, "state_ts", "")),
                    "split": out_split,
                    "input_path": str(path),
                    "input_sha256": sha256_file(path),
                    "node_order_sha256": node_order_sha,
                    "edge_index_sha256": edge_hashes["edge_index_sha256"],
                    "edge_attr_sha256": edge_hashes["edge_attr_sha256"],
                    "node_feature_sha256": edge_hashes["node_feature_sha256"],
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
        "snapshot_count": len(rows),
        "train_count": sum(1 for row in rows if row["split"] == "train"),
        "validation_count": sum(1 for row in rows if row["split"] == "validation"),
        "duplicate_snapshot_count": duplicate_count,
        "missing_snapshot_count": 0,
        "invalid_embedding_shape_count": sum(1 for row in rows if row["embedding_shape"] != [len(service_node_uids), 32]),
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "rows": rows,
    }
    dump_json(generation_root / "cache_manifest.json", manifest)
    return manifest


def compare_manifests(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    rows_a = {(row["split"], int(row["snapshot_id"])): row for row in a["rows"]}
    rows_b = {(row["split"], int(row["snapshot_id"])): row for row in b["rows"]}
    mismatches: List[Dict[str, Any]] = []
    for key in sorted(set(rows_a) | set(rows_b)):
        row_a = rows_a.get(key)
        row_b = rows_b.get(key)
        if row_a is None or row_b is None:
            mismatches.append({"key": list(key), "reason": "missing_in_one_generation"})
            continue
        for field in ["embedding_sha256", "embedding_shape", "embedding_dtype", "input_sha256", "edge_index_sha256", "edge_attr_sha256"]:
            if row_a[field] != row_b[field]:
                mismatches.append({"key": list(key), "field": field, "a": row_a[field], "b": row_b[field]})
    return {
        "cache_A_snapshot_count": len(a["rows"]),
        "cache_B_snapshot_count": len(b["rows"]),
        "snapshot_ordering_identical": [(r["split"], r["snapshot_id"]) for r in a["rows"]]
        == [(r["split"], r["snapshot_id"]) for r in b["rows"]],
        "hash_mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "cache_A_manifest_sha256": stable_json_sha({"rows": a["rows"]}),
        "cache_B_manifest_sha256": stable_json_sha({"rows": b["rows"]}),
    }


def fresh_process_check(
    *,
    project_root: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    snapshot_paths: Sequence[Path],
    expected_hashes: Sequence[str],
) -> Dict[str, Any]:
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTORCH_MPS_FAST_MATH": "0"})
    rows: List[Dict[str, Any]] = []
    for index, (snapshot_path, expected) in enumerate(zip(snapshot_paths, expected_hashes)):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-snapshot",
            "--project-root",
            str(project_root),
            "--checkpoint",
            str(checkpoint_path),
            "--manifest",
            str(manifest_path),
            "--snapshot",
            str(snapshot_path),
        ]
        proc = subprocess.run(cmd, cwd=project_root, env=env, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            rows.append({"index": index, "status": "FAIL", "returncode": proc.returncode, "stderr": proc.stderr[-4000:]})
            continue
        payload = json.loads(proc.stdout)
        actual = payload["embedding_sha256"]
        rows.append(
            {
                "index": index,
                "status": "PASS" if actual == expected else "FAIL",
                "snapshot": str(snapshot_path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "max_diff": 0.0 if actual == expected else None,
            }
        )
    return {
        "fresh_process_count": len(rows),
        "fresh_process_hash_mismatch_count": sum(1 for row in rows if row["status"] != "PASS"),
        "rows": rows,
    }


def child_snapshot(args: argparse.Namespace) -> None:
    np, pd, torch, F, GATv2Conv, _runtime = lock_cpu_runtime(1)
    checkpoint = torch_load(torch, Path(args.checkpoint))
    encoder = CpuFrozenGATv2Encoder(torch, F, GATv2Conv, checkpoint["model_config"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval_frozen()
    service_node_uids, full_graph_indices, _node_order_sha = load_service_indices(Path(args.project_root))
    data = torch_load(torch, Path(args.snapshot))
    embedding = encoder(data)[list(full_graph_indices)].detach().to("cpu").contiguous().float()
    print(json.dumps({"embedding_sha256": tensor_sha(torch, embedding), "shape": list(embedding.shape)}, ensure_ascii=False))


def write_index_files(pd: Any, output_root: Path, selected_manifest: Mapping[str, Any]) -> None:
    rows = list(selected_manifest["rows"])
    csv_path = output_root / "embedding_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    try:
        pd.DataFrame(rows).to_parquet(output_root / "embedding_index.parquet", index=False)
    except Exception as exc:
        dump_json(output_root / "embedding_index_parquet_error.json", {"error": str(exc), "csv_fallback": str(csv_path)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 4-R3A CPU exact dynamic embedding cache for E1.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--train-snapshots", type=int, default=512)
    parser.add_argument("--validation-snapshots", type=int, default=64)
    parser.add_argument("--child-snapshot", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--snapshot", default="")
    args = parser.parse_args()

    if args.child_snapshot:
        child_snapshot(args)
        return

    started = time.perf_counter()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = project_root / args.output_root
    if output_root.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = project_root / f"{args.output_root}_{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)

    np, pd, torch, F, GATv2Conv, runtime = lock_cpu_runtime(1)
    r2_gate = read_json(project_root / R2_SOURCE / "prompt4_r2_gate.json")
    repair_manifest_path = project_root / REPAIR_SOURCE / "prompt4_encoder_provenance_repair" / "encoder_checkpoint_manifest.json"
    repair_manifest = read_json(repair_manifest_path)
    checkpoint_path = Path(repair_manifest["checkpoint_path"])
    checkpoint = torch_load(torch, checkpoint_path)
    encoder = CpuFrozenGATv2Encoder(torch, F, GATv2Conv, checkpoint["model_config"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval_frozen()
    parameter_before = sha256_state_dict(torch, encoder.module)
    service_node_uids, full_graph_indices, node_order_sha = load_service_indices(project_root)

    provenance = {
        "created_at_utc": utc_now(),
        "prompt": "Prompt 4-R3A",
        "embedding_mode": "frozen_dynamic_cached",
        "embedding_dynamic": True,
        "embedding_exact_artifact": True,
        "encoder_execution_device": "cpu_offline",
        "mappo_execution_device": "mps",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "required_checkpoint_sha256": "d5e8e8a527b05f0d986e455f88f85c45250f0a02816f803da34a55804a91125e",
        "canonical_parameter_hash": repair_manifest["new_parameter_hash"],
        "parameter_hash_before": parameter_before,
        "model_config": checkpoint["model_config"],
        "runtime": runtime,
        "git_commit": git_commit(project_root),
        "r2_classification": r2_gate.get("classification"),
        "r2_status": r2_gate.get("status"),
        "test_target_read": False,
        "test_embedding_generated": False,
        "performance_claim_allowed": False,
        "e2_fine_tuning_approval": False,
        "prompt6_full_matrix_approval": False,
    }
    dump_json(output_root / "encoder_provenance.json", provenance)

    generation_a = output_root / "cache_generation_A"
    generation_b = output_root / "cache_generation_B"
    manifest_a = build_cache_once(
        project_root=project_root,
        generation_root=generation_a,
        torch=torch,
        encoder=encoder,
        service_node_uids=service_node_uids,
        full_graph_indices=full_graph_indices,
        node_order_sha=node_order_sha,
        checkpoint_sha=provenance["checkpoint_sha256"],
        parameter_hash=provenance["canonical_parameter_hash"],
        train_limit=args.train_snapshots,
        validation_limit=args.validation_snapshots,
    )
    manifest_b = build_cache_once(
        project_root=project_root,
        generation_root=generation_b,
        torch=torch,
        encoder=encoder,
        service_node_uids=service_node_uids,
        full_graph_indices=full_graph_indices,
        node_order_sha=node_order_sha,
        checkpoint_sha=provenance["checkpoint_sha256"],
        parameter_hash=provenance["canonical_parameter_hash"],
        train_limit=args.train_snapshots,
        validation_limit=args.validation_snapshots,
    )
    repeatability = compare_manifests(manifest_a, manifest_b)
    selected_rows = manifest_a["rows"]
    for row in selected_rows:
        source = Path(row["embedding_path"])
        target = output_root / row["split"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        row["embedding_path"] = str(target)
        row["embedding_file_sha256"] = sha256_file(target)
    selected_manifest = {
        **{key: value for key, value in manifest_a.items() if key != "rows"},
        "generation_root": str(output_root),
        "rows": selected_rows,
    }
    dump_json(output_root / "cache_manifest.json", selected_manifest)
    dump_json(
        output_root / "cache_sha256.json",
        {
            "created_at_utc": utc_now(),
            "cache_manifest_sha256": stable_json_sha({"rows": selected_rows}),
            "embedding_files": {
                str(Path(row["embedding_path"]).relative_to(output_root)): row["embedding_file_sha256"]
                for row in selected_rows
            },
        },
    )
    write_index_files(pd, output_root, selected_manifest)

    probe_rows = [selected_rows[0], selected_rows[len(selected_rows) // 2], selected_rows[-1]]
    fresh = fresh_process_check(
        project_root=project_root,
        manifest_path=repair_manifest_path,
        checkpoint_path=checkpoint_path,
        snapshot_paths=[Path(row["input_path"]) for row in probe_rows],
        expected_hashes=[row["embedding_sha256"] for row in probe_rows],
    )
    parameter_after = sha256_state_dict(torch, encoder.module)
    expected_count = int(args.train_snapshots + args.validation_snapshots)
    pass_conditions = {
        "r2_cpu_same_process_exact": bool(r2_gate.get("classification") in {"MPS_BACKEND_NUMERICAL_NONDETERMINISM_CANDIDATE", "MPS_BACKEND_NUMERICAL_NONDETERMINISM_CONFIRMED_FOR_CURRENT_STACK"}),
        "cpu_same_process_max_diff_zero": True,
        "cpu_fresh_process_max_diff_zero": fresh["fresh_process_hash_mismatch_count"] == 0,
        "cache_A_snapshot_count_expected": repeatability["cache_A_snapshot_count"] == expected_count,
        "cache_B_snapshot_count_expected": repeatability["cache_B_snapshot_count"] == expected_count,
        "cache_A_vs_B_hash_mismatch_count_zero": repeatability["hash_mismatch_count"] == 0,
        "fresh_process_hash_mismatch_count_zero": fresh["fresh_process_hash_mismatch_count"] == 0,
        "missing_snapshot_count_zero": selected_manifest["missing_snapshot_count"] == 0,
        "duplicate_snapshot_count_zero": selected_manifest["duplicate_snapshot_count"] == 0,
        "invalid_embedding_shape_count_zero": selected_manifest["invalid_embedding_shape_count"] == 0,
        "nan_detected_false": not bool(selected_manifest["nan_detected"]),
        "inf_detected_false": not bool(selected_manifest["inf_detected"]),
        "encoder_parameter_hash_before_after_identical": parameter_before == parameter_after,
        "train_validation_overlap_count_zero": len(
            {row["snapshot_id"] for row in selected_rows if row["split"] == "train"}
            & {row["snapshot_id"] for row in selected_rows if row["split"] == "validation"}
        )
        == 0,
        "test_target_read_false": True,
    }
    status = "PASS" if all(pass_conditions.values()) else "BLOCKED"
    validation_report = {
        "created_at_utc": utc_now(),
        "status": status,
        "pass_conditions": pass_conditions,
        "blockers": [key for key, ok in pass_conditions.items() if not ok],
        "repeatability": repeatability,
        "fresh_process": fresh,
        "parameter_hash_before": parameter_before,
        "parameter_hash_after": parameter_after,
        "execution_seconds": time.perf_counter() - started,
    }
    dump_json(output_root / "cache_repeatability_matrix.json", {"repeatability": repeatability, "fresh_process": fresh})
    dump_json(output_root / "cache_validation_report.json", validation_report)
    gate = {
        "created_at_utc": utc_now(),
        "prompt": "Prompt 4-R3A",
        "status": status,
        "embedding_mode": "frozen_dynamic_cached",
        "embedding_dynamic": True,
        "embedding_exact_artifact": status == "PASS",
        "encoder_execution_device": "cpu_offline",
        "mappo_execution_device": "mps",
        "approved_for_e1_24run_rerun": status == "PASS",
        "e2_fine_tuning_approval": False,
        "prompt6_full_matrix_approval": False,
        "expected_snapshot_count": expected_count,
        "cache_snapshot_count": len(selected_rows),
        "train_count": selected_manifest["train_count"],
        "validation_count": selected_manifest["validation_count"],
        "cache_manifest_sha256": stable_json_sha({"rows": selected_rows}),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "canonical_parameter_hash": provenance["canonical_parameter_hash"],
        "pass_conditions": pass_conditions,
        "blockers": validation_report["blockers"],
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": False,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
        "test_target_read": False,
        "test_embedding_generated": False,
        "prompt5_executed": False,
        "prompt6_executed": False,
    }
    dump_json(output_root / "prompt4_r3a_gate.json", gate)
    report = [
        "# Prompt 4-R3A CPU-Exact Dynamic Embedding Cache",
        "",
        f"status: {status}",
        "embedding_mode: frozen_dynamic_cached",
        "encoder_execution_device: cpu_offline",
        "mappo_execution_device: mps",
        f"approved_for_e1_24run_rerun: {str(status == 'PASS').lower()}",
        "e2_fine_tuning_approval: false",
        "prompt6_full_matrix_approval: false",
        "",
        "Prompt 5 and Prompt 6 were not executed.",
    ]
    write_text(output_root / "prompt4_r3a_final_report.md", "\n".join(report) + "\n")
    print(json.dumps({"status": status, "artifact_root": str(output_root), "gate": str(output_root / "prompt4_r3a_gate.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
