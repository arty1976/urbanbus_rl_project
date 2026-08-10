from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch


ARTIFACT_VERSION = "suseong_node_level_service_embedding_contract_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_snapshot_files(dataset_dir: Path, split: str, snapshot_count: int) -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(dataset_dir / split / "*.pt")))
    if len(files) < snapshot_count:
        raise FileNotFoundError(
            f"Need at least {snapshot_count} .pt files in {dataset_dir / split}; found {len(files)}"
        )
    return files[:snapshot_count]


def make_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scale = 1.0 / math.sqrt(max(in_dim, 1))
    return torch.randn((in_dim, out_dim), generator=generator, dtype=torch.float32) * scale


def build_contract(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).expanduser().resolve()
    service_graph_dir = Path(args.service_graph_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    service_nodes_path = service_graph_dir / "service_nodes.csv"
    graph_manifest_path = service_graph_dir / "service_graph_manifest.json"
    full_nodes_path = project_root / "05_training/artifacts/suseong_source_pack_v1/full_graph_nodes.parquet"

    service_nodes = pd.read_csv(service_nodes_path)
    full_nodes = pd.read_parquet(full_nodes_path)
    full_node_count = int(len(full_nodes))

    required_cols = {"node_uid", "node_index"}
    missing = sorted(required_cols - set(service_nodes.columns))
    if missing:
        raise RuntimeError(f"service_nodes.csv missing required columns: {missing}")

    service_nodes = service_nodes.sort_values("node_index").reset_index(drop=True)
    service_indices = [int(v) for v in service_nodes["node_index"].tolist()]
    service_uids = [str(v) for v in service_nodes["node_uid"].tolist()]
    duplicate_embedding_rows = int(pd.Series(service_indices).duplicated().sum())
    out_of_range = [idx for idx in service_indices if idx < 0 or idx >= full_node_count]
    if duplicate_embedding_rows != 0:
        raise RuntimeError(f"duplicate service node_index rows found: {duplicate_embedding_rows}")
    if out_of_range:
        raise RuntimeError(f"service node_index out of full graph range: {out_of_range[:10]}")

    snapshot_files = resolve_snapshot_files(dataset_dir, args.split, args.snapshot_count)
    first = torch_load(snapshot_files[0])
    if not hasattr(first, "x") or first.x.ndim != 2:
        raise RuntimeError(f"snapshot does not contain a 2D x tensor: {snapshot_files[0]}")
    if int(first.x.shape[0]) != full_node_count:
        raise RuntimeError(
            f"snapshot node count mismatch: snapshot={int(first.x.shape[0])}, full_nodes={full_node_count}"
        )

    projection = make_projection(int(first.x.shape[1]), int(args.embedding_dim), int(args.seed))
    embeddings: List[torch.Tensor] = []
    snapshot_rows: List[Dict[str, Any]] = []
    for path in snapshot_files:
        data = torch_load(path)
        x = data.x.detach().cpu().float()
        if int(x.shape[0]) != full_node_count:
            raise RuntimeError(f"snapshot node count mismatch for {path}: {tuple(x.shape)}")
        service_x = x[service_indices]
        service_embedding = torch.tanh(service_x @ projection)
        embeddings.append(service_embedding)
        snapshot_rows.append(
            {
                "snapshot_file": str(path),
                "snapshot_id": int(getattr(data, "snapshot_id", len(snapshot_rows))),
                "state_ts": str(getattr(data, "state_ts", "")),
                "service_embedding_shape": list(service_embedding.shape),
            }
        )

    stacked = torch.stack(embeddings, dim=0).contiguous()
    finite = bool(torch.isfinite(stacked).all().item())
    output_root.mkdir(parents=True, exist_ok=True)
    embedding_path = output_root / "service_node_embeddings.pt"
    mapping_csv = output_root / "service_node_embedding_mapping.csv"
    manifest_path = output_root / "node_level_embedding_contract.json"

    torch.save(
        {
            "artifact_version": ARTIFACT_VERSION,
            "created_at_utc": utc_now(),
            "service_node_embeddings": stacked,
            "service_node_indices": torch.tensor(service_indices, dtype=torch.long),
            "service_node_uids": service_uids,
            "projection_matrix": projection,
            "snapshot_rows": snapshot_rows,
            "source_method": "deterministic_service_node_feature_projection_for_prompt2_preflight_contract",
            "performance_claim_allowed": False,
        },
        embedding_path,
    )

    pd.DataFrame(
        {
            "service_local_index": list(range(len(service_indices))),
            "node_uid": service_uids,
            "full_graph_node_index": service_indices,
            "is_suseong_core": service_nodes.get("is_suseong_core", pd.Series([None] * len(service_nodes))).tolist(),
            "official_gu": service_nodes.get("official_gu", pd.Series([None] * len(service_nodes))).tolist(),
        }
    ).to_csv(mapping_csv, index=False)

    manifest = {
        "created_at_utc": utc_now(),
        "artifact_version": ARTIFACT_VERSION,
        "status": "PASS",
        "scope": "Prompt 2-R node-level service embedding contract",
        "not_scope": [
            "trained GATv2 encoder performance claim",
            "fleet scientific gate",
            "Prompt 2 full causal preflight final approval",
        ],
        "project_root": str(project_root),
        "service_graph_dir": str(service_graph_dir),
        "service_graph_manifest": str(graph_manifest_path),
        "dataset_dir": str(dataset_dir),
        "split": str(args.split),
        "snapshot_count": int(args.snapshot_count),
        "full_graph_node_count": full_node_count,
        "service_node_count": int(len(service_indices)),
        "embedding_dim": int(args.embedding_dim),
        "embedding_shape": list(stacked.shape),
        "embedding_dtype": str(stacked.dtype),
        "embedding_finite": finite,
        "service_node_to_full_graph_mapping_coverage_percent": 100.0,
        "unmapped_service_node_count": 0,
        "duplicate_embedding_row_count": duplicate_embedding_rows,
        "source_method": "deterministic_service_node_feature_projection_for_prompt2_preflight_contract",
        "source_method_guard": (
            "This artifact is a node-level contract and structural preflight input. "
            "It must not be described as a trained GATv2 performance embedding."
        ),
        "performance_claim_allowed": False,
        "embedding_path": str(embedding_path),
        "embedding_file_sha256": sha256_file(embedding_path),
        "embedding_tensor_sha256": sha256_tensor(stacked),
        "mapping_csv": str(mapping_csv),
        "mapping_csv_sha256": sha256_file(mapping_csv),
        "snapshot_rows": snapshot_rows,
        "blockers_preserved": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
    }
    dump_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Prompt 2-R node-level service embedding contract.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument(
        "--service-graph-dir",
        default="05_training/artifacts/suseong_service_graph_v1",
    )
    parser.add_argument(
        "--dataset-dir",
        default="05_training/artifacts/dataset_full_20260422_084243",
    )
    parser.add_argument(
        "--output-root",
        default="05_training/artifacts/suseong_node_level_embedding_contract_v1",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--snapshot-count", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main() -> None:
    manifest = build_contract(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
