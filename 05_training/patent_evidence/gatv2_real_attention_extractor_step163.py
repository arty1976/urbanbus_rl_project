from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

ARTIFACT_VERSION = "gatv2_real_attention_extractor_step163_v1"

NON_CLAIM_FLAGS = {
    "simulation_evidence_only": True,
    "actual_operational_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "trained_model_claim_allowed": False,
    "train_allowed": False,
}

REQUIRED_SNAPSHOT_ATTENTION_COLUMNS = [
    "state_ts",
    "layer_id",
    "head_id",
    "edge_rank",
    "src_node",
    "dst_node",
    "src_idx",
    "dst_idx",
    "attention_weight",
    "attention_source",
    "real_gatv2conv_attention_extracted",
]

REQUIRED_STEP160_ATTENTION_COLUMNS = [
    "attempt_id",
    "state_ts",
    "layer_id",
    "head_id",
    "src_node",
    "dst_node",
    "attention_weight",
]

VALID_SEGMENTS = {
    "existing_passenger_path",
    "candidate_pickup_path",
    "candidate_dropoff_path",
    "unrelated",
    "unknown",
}


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise RuntimeError(f"unsupported table format: {path}")


def write_table(path: Path, df: pd.DataFrame, preferred_format: str, warnings: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred_format = preferred_format.lower().strip()
    if preferred_format not in {"csv", "parquet"}:
        raise RuntimeError(f"unsupported file format: {preferred_format}")
    if preferred_format == "csv":
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        return out
    out = path.with_suffix(".parquet")
    try:
        df.to_parquet(out, index=False)
        return out
    except Exception as exc:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        warnings.append(f"parquet_write_failed_csv_fallback: {exc}")
        return fallback


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def normalize_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    s = str(value).strip()
    return s if s else default


def bool_from_any(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def try_import_torch_stack() -> Tuple[Any, Any, Any, Optional[str]]:
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GATv2Conv
        return torch, Data, GATv2Conv, None
    except Exception as exc:
        return None, None, None, repr(exc)


def make_sample_attempts(condition_id: str, seed: int, state_ts: str) -> pd.DataFrame:
    rows = []
    for i, (pickup, dropoff) in enumerate([("node_1", "node_4"), ("node_2", "node_5"), ("node_0", "node_3")], start=1):
        rows.append(
            {
                "attempt_id": f"att_step163_{i:03d}",
                "state_ts": state_ts,
                "condition_id": condition_id,
                "seed": int(seed),
                "vehicle_id": f"bus_{i:02d}",
                "existing_passenger_id": f"existing_{i:03d}",
                "new_passenger_id": f"new_{i:03d}",
                "route_id": "route_step163",
                "direction_id": "0",
                "pickup_stop_id": pickup,
                "dropoff_stop_id": dropoff,
                "decision": "accepted" if i != 2 else "rejected",
                "time_band": "peak",
                "window_id": "w_step163_sample",
            }
        )
    return pd.DataFrame(rows)


def make_sample_pyg_data() -> Tuple[Any, Dict[int, str], str]:
    torch, Data, _, err = try_import_torch_stack()
    if torch is None or Data is None:
        raise RuntimeError(f"torch_geometric stack unavailable: {err}")
    torch.manual_seed(163)
    x = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.0, 1.0, 1.0],
            [0.8, 0.1, 0.4, 0.2, 0.9, 1.0],
            [0.4, 0.3, 0.7, 0.4, 0.7, 1.0],
            [0.2, 0.5, 0.5, 0.6, 0.5, 1.0],
            [0.1, 0.4, 0.3, 0.8, 0.3, 1.0],
            [0.0, 0.2, 0.1, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [
            [120.0, 22.0, 1.02, 0.0],
            [120.0, 22.0, 1.02, 0.0],
            [210.0, 31.0, 1.10, 0.0],
            [210.0, 31.0, 1.10, 0.0],
            [180.0, 28.0, 1.08, 0.0],
            [180.0, 28.0, 1.08, 0.0],
            [260.0, 39.0, 1.20, 0.0],
            [260.0, 39.0, 1.20, 0.0],
            [300.0, 45.0, 1.30, 0.0],
            [300.0, 45.0, 1.30, 0.0],
        ],
        dtype=torch.float32,
    )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    node_map = {i: f"node_{i}" for i in range(x.shape[0])}
    return data, node_map, "2026-01-02T08:00:00+09:00"


def maybe_tensor_to_list(value: Any) -> Any:
    try:
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "cpu"):
            return value.cpu().tolist()
    except Exception:
        pass
    return value


def data_num_nodes(data: Any) -> int:
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        return int(data.num_nodes)
    if hasattr(data, "x"):
        return int(data.x.size(0))
    raise RuntimeError("cannot determine num_nodes from PyG Data")


def infer_node_map(data: Any) -> Dict[int, str]:
    n = data_num_nodes(data)
    # PyG Data may carry custom fields, but they may be tensors/lists/strings.
    for attr in ("node_uid", "node_ids", "node_id", "stop_id", "node_names"):
        if hasattr(data, attr):
            raw = maybe_tensor_to_list(getattr(data, attr))
            if isinstance(raw, (list, tuple)) and len(raw) == n:
                return {i: normalize_str(raw[i], f"node_{i}") for i in range(n)}
    return {i: f"node_{i}" for i in range(n)}


def load_pyg_data(path: Path) -> Tuple[Any, Dict[int, str], str]:
    torch, _, _, err = try_import_torch_stack()
    if torch is None:
        raise RuntimeError(f"torch/torch_geometric stack unavailable: {err}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    data = obj
    if isinstance(obj, dict):
        # Common wrappers.
        for key in ("data", "pyg_data", "snapshot", "sample"):
            if key in obj:
                data = obj[key]
                break
    if not hasattr(data, "x") or not hasattr(data, "edge_index"):
        raise RuntimeError(f"loaded object from {path} is not a PyG Data-like object")
    node_map = infer_node_map(data)
    state_ts = normalize_str(getattr(data, "state_ts", ""), "unknown_state_ts")
    return data, node_map, state_ts


class AttentionEnabledNodeLevelGATv2:
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_dim: Optional[int], heads1: int = 2, heads2: int = 1):
        torch, _, GATv2Conv, err = try_import_torch_stack()
        if torch is None or GATv2Conv is None:
            raise RuntimeError(f"torch_geometric stack unavailable: {err}")
        import torch.nn as nn
        self.torch = torch
        self.F = __import__("torch.nn.functional", fromlist=["relu"])
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads1, concat=True, edge_dim=edge_dim)
                self.conv2 = GATv2Conv(hidden_channels * heads1, hidden_channels, heads=heads2, concat=True, edge_dim=edge_dim)
                self.lin = nn.Linear(hidden_channels * heads2, out_channels)

            def forward(self, data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
                x = data.x.float()
                edge_index = data.edge_index
                edge_attr = getattr(data, "edge_attr", None)
                if edge_attr is not None:
                    edge_attr = edge_attr.float()
                x1, att1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                x1 = self.__class__.F.relu(x1) if hasattr(self.__class__, "F") else x1.relu()
                x2, att2 = self.conv2(x1, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                x2 = self.__class__.F.relu(x2) if hasattr(self.__class__, "F") else x2.relu()
                out = self.lin(x2)
                return out, [
                    {"layer_id": 1, "attention": att1},
                    {"layer_id": 2, "attention": att2},
                ]
        # Attach F to class to avoid closure serialization concerns.
        Model.F = self.F
        self.model = Model()

    def load_checkpoint(self, checkpoint_path: Optional[Path], strict: bool, warnings: List[str]) -> Dict[str, Any]:
        if checkpoint_path is None:
            return {"checkpoint_loaded": False, "checkpoint_path": ""}
        torch = self.torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "encoder_state_dict"):
                if key in ckpt:
                    state = ckpt[key]
                    break
        try:
            result = self.model.load_state_dict(state, strict=strict)
            missing = list(getattr(result, "missing_keys", []))
            unexpected = list(getattr(result, "unexpected_keys", []))
            if missing:
                warnings.append(f"checkpoint_missing_keys: {missing[:10]}")
            if unexpected:
                warnings.append(f"checkpoint_unexpected_keys: {unexpected[:10]}")
            return {
                "checkpoint_loaded": True,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_strict": bool(strict),
                "missing_keys_count": len(missing),
                "unexpected_keys_count": len(unexpected),
            }
        except Exception as exc:
            if strict:
                raise
            warnings.append(f"checkpoint_load_non_strict_failed: {exc}")
            return {
                "checkpoint_loaded": False,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_strict": bool(strict),
                "checkpoint_error": repr(exc),
            }

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        with self.torch.no_grad():
            return self.model(data)


def collect_real_gatv2_attention(
    data: Any,
    node_map: Dict[int, str],
    state_ts: str,
    hidden_channels: int,
    out_channels: int,
    checkpoint_path: Optional[Path],
    checkpoint_strict: bool,
    warnings: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    torch, _, _, err = try_import_torch_stack()
    if torch is None:
        raise RuntimeError(f"torch_geometric stack unavailable: {err}")
    in_channels = int(data.x.size(1))
    edge_dim = int(data.edge_attr.size(1)) if hasattr(data, "edge_attr") and data.edge_attr is not None else None
    model = AttentionEnabledNodeLevelGATv2(
        in_channels=in_channels,
        hidden_channels=int(hidden_channels),
        out_channels=int(out_channels),
        edge_dim=edge_dim,
    )
    checkpoint_info = model.load_checkpoint(checkpoint_path, strict=checkpoint_strict, warnings=warnings)
    model.eval()
    _, attentions = model(data)

    rows: List[Dict[str, Any]] = []
    edge_attr = getattr(data, "edge_attr", None)
    original_edge_count = int(data.edge_index.size(1))

    for layer in attentions:
        layer_id = int(layer["layer_id"])
        edge_index_out, alpha = layer["attention"]
        edge_index_out = edge_index_out.detach().cpu()
        alpha = alpha.detach().cpu()
        # alpha: [num_edges_after_self_loops, num_heads]
        for edge_rank in range(edge_index_out.size(1)):
            src_idx = int(edge_index_out[0, edge_rank].item())
            dst_idx = int(edge_index_out[1, edge_rank].item())
            src_node = node_map.get(src_idx, f"node_{src_idx}")
            dst_node = node_map.get(dst_idx, f"node_{dst_idx}")
            is_self_loop = bool(src_idx == dst_idx)
            # Edge attrs are available only for original edges. GATv2Conv may add self-loops.
            distance_m = None
            time_sec = None
            generalized_cost = None
            long_edge_5km_flag = None
            if edge_attr is not None and edge_rank < original_edge_count:
                vals = edge_attr[edge_rank].detach().cpu().tolist()
                if len(vals) > 0:
                    distance_m = float(vals[0])
                if len(vals) > 1:
                    time_sec = float(vals[1])
                if len(vals) > 2:
                    generalized_cost = float(vals[2])
                if len(vals) > 3:
                    long_edge_5km_flag = float(vals[3])
            for head_id in range(alpha.size(1)):
                rows.append(
                    {
                        "state_ts": state_ts,
                        "layer_id": layer_id,
                        "head_id": int(head_id),
                        "edge_rank": int(edge_rank),
                        "src_node": src_node,
                        "dst_node": dst_node,
                        "src_idx": src_idx,
                        "dst_idx": dst_idx,
                        "edge_id": f"gatv2_l{layer_id}_h{head_id}_e{edge_rank}_{src_idx}_{dst_idx}",
                        "attention_weight": float(alpha[edge_rank, head_id].item()),
                        "attention_source": "gatv2conv_return_attention_weights",
                        "real_gatv2conv_attention_extracted": True,
                        "distance_m": distance_m,
                        "time_sec": time_sec,
                        "generalized_cost": generalized_cost,
                        "long_edge_5km_flag": long_edge_5km_flag,
                        "is_self_loop": is_self_loop,
                    }
                )
    df = pd.DataFrame(rows)
    return df, checkpoint_info


def collect_fallback_attention(
    state_ts: str,
    seed: int,
    warnings: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Torch/PyG-free deterministic fallback for package portability. Not real GATv2Conv attention."""
    warnings.append("fallback_attention_used_not_real_gatv2conv")
    random.seed(int(seed))
    nodes = [f"node_{i}" for i in range(6)]
    directed_edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4)]
    rows: List[Dict[str, Any]] = []
    for layer_id, head_count in [(1, 2), (2, 1)]:
        for head_id in range(head_count):
            for edge_rank, (src, dst) in enumerate(directed_edges):
                h = stable_int(layer_id, head_id, src, dst, seed)
                rows.append(
                    {
                        "state_ts": state_ts,
                        "layer_id": layer_id,
                        "head_id": head_id,
                        "edge_rank": edge_rank,
                        "src_node": nodes[src],
                        "dst_node": nodes[dst],
                        "src_idx": src,
                        "dst_idx": dst,
                        "edge_id": f"fallback_l{layer_id}_h{head_id}_e{edge_rank}_{src}_{dst}",
                        "attention_weight": round(0.05 + (h % 100) / 300.0, 6),
                        "attention_source": "fallback_torch_free_contract_smoke",
                        "real_gatv2conv_attention_extracted": False,
                        "distance_m": float(120 + 20 * edge_rank),
                        "time_sec": float(20 + 4 * edge_rank),
                        "generalized_cost": round(1.0 + edge_rank / 100.0, 6),
                        "long_edge_5km_flag": 0.0,
                        "is_self_loop": False,
                    }
                )
    return pd.DataFrame(rows), {"checkpoint_loaded": False, "checkpoint_path": ""}


def classify_path_segment(row: pd.Series, attempt: pd.Series) -> str:
    src = normalize_str(row.get("src_node", ""))
    dst = normalize_str(row.get("dst_node", ""))
    pickup = normalize_str(attempt.get("pickup_stop_id", ""))
    dropoff = normalize_str(attempt.get("dropoff_stop_id", ""))
    existing_hint = normalize_str(attempt.get("current_stop_id", attempt.get("existing_current_stop_id", "")))
    if pickup and (src == pickup or dst == pickup):
        return "candidate_pickup_path"
    if dropoff and (src == dropoff or dst == dropoff):
        return "candidate_dropoff_path"
    if existing_hint and (src == existing_hint or dst == existing_hint):
        return "existing_passenger_path"
    return "unrelated"


def build_attempt_level_attention(
    snapshot_attention: pd.DataFrame,
    attempts: Optional[pd.DataFrame],
    top_k_per_attempt: int,
) -> pd.DataFrame:
    if attempts is None or attempts.empty:
        return pd.DataFrame(columns=REQUIRED_STEP160_ATTENTION_COLUMNS)

    require_columns(attempts, ["attempt_id", "state_ts", "pickup_stop_id", "dropoff_stop_id"], "attempt_events")
    rows: List[Dict[str, Any]] = []
    df = snapshot_attention.copy()
    df["attention_weight_numeric"] = pd.to_numeric(df["attention_weight"], errors="coerce")
    df = df.sort_values(["attention_weight_numeric"], ascending=False).reset_index(drop=True)
    if top_k_per_attempt > 0:
        base_top = df.head(top_k_per_attempt).copy()
    else:
        base_top = df.copy()

    for attempt in attempts.itertuples(index=False):
        a = pd.Series(attempt._asdict())
        attempt_id = normalize_str(a.get("attempt_id"))
        route_id = normalize_str(a.get("route_id", ""))
        direction_id = normalize_str(a.get("direction_id", ""))
        for _, r in base_top.iterrows():
            segment = classify_path_segment(r, a)
            rows.append(
                {
                    "attempt_id": attempt_id,
                    "state_ts": normalize_str(a.get("state_ts", r.get("state_ts", "")), normalize_str(r.get("state_ts", ""))),
                    "layer_id": int(r["layer_id"]),
                    "head_id": int(r["head_id"]),
                    "src_node": normalize_str(r["src_node"]),
                    "dst_node": normalize_str(r["dst_node"]),
                    "edge_id": normalize_str(r.get("edge_id", "")),
                    "attention_weight": float(r["attention_weight"]),
                    "path_segment": segment,
                    "route_id": route_id,
                    "direction_id": direction_id,
                    "distance_m": r.get("distance_m"),
                    "time_sec": r.get("time_sec"),
                    "generalized_cost": r.get("generalized_cost"),
                    "attention_source": normalize_str(r.get("attention_source", "")),
                    "real_gatv2conv_attention_extracted": bool_from_any(r.get("real_gatv2conv_attention_extracted", False)),
                }
            )
    return pd.DataFrame(rows)


def validate_attention_outputs(snapshot_attention: pd.DataFrame, attempt_attention: Optional[pd.DataFrame]) -> None:
    require_columns(snapshot_attention, REQUIRED_SNAPSHOT_ATTENTION_COLUMNS, "gatv2_real_attention_edges")
    if snapshot_attention.empty:
        raise RuntimeError("gatv2_real_attention_edges is empty")
    weights = pd.to_numeric(snapshot_attention["attention_weight"], errors="raise")
    if weights.isna().any():
        raise RuntimeError("snapshot attention_weight contains null values")
    if bool((weights < 0).any()):
        raise RuntimeError("snapshot attention_weight must be non-negative")
    if attempt_attention is not None and not attempt_attention.empty:
        require_columns(attempt_attention, REQUIRED_STEP160_ATTENTION_COLUMNS, "gatv2_attention")
        aw = pd.to_numeric(attempt_attention["attention_weight"], errors="raise")
        if aw.isna().any() or bool((aw < 0).any()):
            raise RuntimeError("attempt-level attention_weight must be finite and non-negative")


def attention_quality(snapshot_attention: pd.DataFrame, attempt_attention: pd.DataFrame) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "snapshot_attention_row_count": int(len(snapshot_attention)),
        "attempt_attention_row_count": int(len(attempt_attention)),
        "layer_count": int(snapshot_attention["layer_id"].nunique()) if not snapshot_attention.empty else 0,
        "head_count_total": int(snapshot_attention[["layer_id", "head_id"]].drop_duplicates().shape[0]) if not snapshot_attention.empty else 0,
        "real_gatv2conv_attention_extracted": bool(snapshot_attention["real_gatv2conv_attention_extracted"].astype(bool).all()) if not snapshot_attention.empty else False,
        "attention_source_counts": snapshot_attention["attention_source"].astype(str).value_counts().to_dict() if not snapshot_attention.empty else {},
        "attention_weight_min": float(pd.to_numeric(snapshot_attention["attention_weight"]).min()) if not snapshot_attention.empty else None,
        "attention_weight_max": float(pd.to_numeric(snapshot_attention["attention_weight"]).max()) if not snapshot_attention.empty else None,
        "attempt_count_with_attention": int(attempt_attention["attempt_id"].nunique()) if not attempt_attention.empty and "attempt_id" in attempt_attention.columns else 0,
    }
    return payload


def run_extractor(
    mode: str,
    output_root: Path,
    condition_id: str,
    seed: int,
    file_format: str,
    pyg_pt: Optional[Path],
    checkpoint: Optional[Path],
    checkpoint_strict: bool,
    attempt_events: Optional[Path],
    state_ts_arg: str,
    hidden_channels: int,
    out_channels: int,
    top_k_per_attempt: int,
    allow_fallback: bool,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    attempts: Optional[pd.DataFrame] = None
    input_files: Dict[str, str] = {}

    if mode == "sample":
        try:
            data, node_map, default_state_ts = make_sample_pyg_data()
            state_ts = normalize_str(state_ts_arg, default_state_ts) if state_ts_arg else default_state_ts
            attempts = make_sample_attempts(condition_id=condition_id, seed=seed, state_ts=state_ts)
            sample_attempt_path = output_root / "sample_inputs" / "pickup_attempt_events.csv"
            sample_attempt_path.parent.mkdir(parents=True, exist_ok=True)
            attempts.to_csv(sample_attempt_path, index=False, encoding="utf-8-sig")
            input_files["sample_pickup_attempt_events"] = str(sample_attempt_path)
            snapshot_attention, checkpoint_info = collect_real_gatv2_attention(
                data=data,
                node_map=node_map,
                state_ts=state_ts,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                checkpoint_path=checkpoint,
                checkpoint_strict=checkpoint_strict,
                warnings=warnings,
            )
        except Exception as exc:
            if not allow_fallback:
                raise
            state_ts = normalize_str(state_ts_arg, "2026-01-02T08:00:00+09:00")
            attempts = make_sample_attempts(condition_id=condition_id, seed=seed, state_ts=state_ts)
            sample_attempt_path = output_root / "sample_inputs" / "pickup_attempt_events.csv"
            sample_attempt_path.parent.mkdir(parents=True, exist_ok=True)
            attempts.to_csv(sample_attempt_path, index=False, encoding="utf-8-sig")
            input_files["sample_pickup_attempt_events"] = str(sample_attempt_path)
            warnings.append(f"real_gatv2conv_sample_failed_fallback_used: {exc}")
            snapshot_attention, checkpoint_info = collect_fallback_attention(state_ts=state_ts, seed=seed, warnings=warnings)
    elif mode == "from-pyg-pt":
        if pyg_pt is None:
            raise RuntimeError("--pyg-pt is required for mode from-pyg-pt")
        data, node_map, default_state_ts = load_pyg_data(pyg_pt)
        state_ts = normalize_str(state_ts_arg, default_state_ts) if state_ts_arg else default_state_ts
        input_files["pyg_pt"] = str(pyg_pt)
        if attempt_events is not None:
            attempts = read_table(attempt_events)
            input_files["attempt_events"] = str(attempt_events)
        snapshot_attention, checkpoint_info = collect_real_gatv2_attention(
            data=data,
            node_map=node_map,
            state_ts=state_ts,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            checkpoint_path=checkpoint,
            checkpoint_strict=checkpoint_strict,
            warnings=warnings,
        )
    else:
        raise RuntimeError(f"unsupported mode: {mode}")

    attempt_attention = build_attempt_level_attention(
        snapshot_attention=snapshot_attention,
        attempts=attempts,
        top_k_per_attempt=top_k_per_attempt,
    )
    validate_attention_outputs(snapshot_attention, attempt_attention)

    snapshot_path = write_table(output_root / "gatv2_real_attention_edges", snapshot_attention, file_format, warnings)
    attempt_attention_path = None
    if not attempt_attention.empty:
        attempt_attention_path = write_table(output_root / "gatv2_attention", attempt_attention, file_format, warnings)

    quality = attention_quality(snapshot_attention, attempt_attention)
    quality_path = output_root / "attention_quality_report.json"
    dump_json(quality_path, quality)

    outputs: Dict[str, str] = {
        "gatv2_real_attention_edges": str(snapshot_path),
        "attention_quality_report": str(quality_path),
    }
    output_hashes: Dict[str, str] = {
        "gatv2_real_attention_edges": sha256_file(snapshot_path),
        "attention_quality_report": sha256_file(quality_path),
    }
    if attempt_attention_path is not None:
        outputs["gatv2_attention"] = str(attempt_attention_path)
        output_hashes["gatv2_attention"] = sha256_file(attempt_attention_path)

    real_extracted = bool(quality.get("real_gatv2conv_attention_extracted", False))
    bundle_status = (
        "GATV2_REAL_ATTENTION_EXTRACTED_FOR_ZERO_LOSS_EVIDENCE_NONCLAIM"
        if real_extracted
        else "GATV2_ATTENTION_CONTRACT_FALLBACK_ONLY_NONCLAIM"
    )
    audit_status = "PASS" if real_extracted else "PASS_WITH_FALLBACK"

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": audit_status,
        "bundle_status": bundle_status,
        "mode": mode,
        "condition_id": condition_id,
        "seed": int(seed),
        "output_root": str(output_root),
        "input_files": input_files,
        "output_files": outputs,
        "output_sha256": output_hashes,
        "summary": {
            **quality,
            "checkpoint_loaded": bool(checkpoint_info.get("checkpoint_loaded", False)),
            "checkpoint_path": checkpoint_info.get("checkpoint_path", ""),
        },
        "checkpoint_info": checkpoint_info,
        "warnings": warnings,
        **NON_CLAIM_FLAGS,
    }
    manifest_path = output_root / "gatv2_real_attention_extractor_manifest.json"
    dump_json(manifest_path, manifest)

    print("[OK] Step 163 GATv2 real attention extractor completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] bundle_status             : {manifest['bundle_status']}")
    print(f"[OK] real_gatv2conv_attention_extracted : {real_extracted}")
    print(f"[OK] snapshot_attention_row_count       : {quality['snapshot_attention_row_count']}")
    print(f"[OK] attempt_attention_row_count        : {quality['attempt_attention_row_count']}")
    print(f"[OK] attempt_count_with_attention       : {quality['attempt_count_with_attention']}")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] manifest                  : {manifest_path}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 163 GATv2 real attention extractor")
    parser.add_argument("--mode", choices=["sample", "from-pyg-pt"], default="sample")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--file-format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--pyg-pt", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-strict", action="store_true")
    parser.add_argument("--attempt-events", default="")
    parser.add_argument("--state-ts", default="")
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--top-k-per-attempt", type=int, default=12)
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    run_extractor(
        mode=args.mode,
        output_root=Path(args.output_root),
        condition_id=args.condition_id,
        seed=args.seed,
        file_format=args.file_format,
        pyg_pt=Path(args.pyg_pt) if args.pyg_pt else None,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        checkpoint_strict=bool(args.checkpoint_strict),
        attempt_events=Path(args.attempt_events) if args.attempt_events else None,
        state_ts_arg=args.state_ts,
        hidden_channels=int(args.hidden_channels),
        out_channels=int(args.out_channels),
        top_k_per_attempt=int(args.top_k_per_attempt),
        allow_fallback=bool(args.allow_fallback),
    )


if __name__ == "__main__":
    main()
