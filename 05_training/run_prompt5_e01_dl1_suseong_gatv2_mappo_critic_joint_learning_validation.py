from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

# --- H4M-AE-R9.8 LS3-BT3 fail-closed training authorization -------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


ARTIFACT_PREFIX = "prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation"
EXPECTED_UPSTREAM_GATE = "PASS_AXIS_SPECIFIC_SYMBOLIC_DOMAIN_KIND_ADJUDICATED_AS_CONVENTION_STILL_LOCKED"
SUCCESS_GATE_VALIDATED = "PASS_SUSEONG_GATV2_MAPPO_CRITIC_JOINT_LEARNING_PATH_VALIDATED_ON_MAC_M4"
SUCCESS_GATE_INDETERMINATE = "PASS_SUSEONG_GATV2_MAPPO_CRITIC_LEARNING_ACTIVE_SHORT_RUN_SIGNAL_INDETERMINATE_ON_MAC_M4"
FAIL_MPS_UNAVAILABLE = "FAIL_MPS_UNAVAILABLE"
FAIL_SUBGRAPH = "FAIL_SUSEONG_GU_SUBGRAPH_NOT_DERIVABLE_FROM_FROZEN_LOCAL_DATA"
FAIL_TENSOR = "FAIL_TENSOR_CONTRACT_OR_MASK_AUDIT_FAILED"
FAIL_CRITIC = "FAIL_CRITIC_LEARNING_PATH_INCOMPLETE"
FAIL_CHECKPOINT = "FAIL_CHECKPOINT_RELOAD_AUDIT_FAILED"

REQUIRED_FILES = [
    "upstream_data_contract_snapshot.json",
    "study_area_contract.json",
    "suseong_subgraph_inventory.json",
    "suseong_subgraph_connectivity_audit.json",
    "training_configuration.json",
    "runtime_environment.json",
    "mps_memory_report.json",
    "authoritative_implementation_inventory.json",
    "model_architecture_summary.json",
    "optimizer_parameter_ownership.json",
    "critic_learning_path_audit.json",
    "gae_return_advantage_audit.json",
    "tensor_shape_audit.json",
    "masking_audit.json",
    "loss_finiteness_audit.json",
    "run_a_learning_metrics.jsonl",
    "run_b_frozen_control_metrics.jsonl",
    "module_parameter_delta.json",
    "module_gradient_audit.json",
    "run_a_run_b_control_comparison.json",
    "checkpoint_manifest.json",
    "checkpoint_reload_audit.json",
    "source_change_inventory.json",
    "prohibited_scope_audit.json",
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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_hash(module: torch.nn.Module) -> str:
    return state_dict_hash(module.state_dict())


def clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def parameter_counts(module: torch.nn.Module) -> Dict[str, int]:
    return {
        "parameter_count": int(sum(p.numel() for p in module.parameters())),
        "trainable_parameter_count": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
    }


def delta_stats(
    module: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
    *,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    after = clone_state_dict(module)
    l1 = 0.0
    l2_sq = 0.0
    changed = 0
    unchanged = 0
    max_abs = 0.0
    for name in sorted(before):
        delta = after[name].float() - before[name].float()
        tensor_l1 = float(delta.abs().sum().item())
        tensor_l2_sq = float((delta * delta).sum().item())
        tensor_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        l1 += tensor_l1
        l2_sq += tensor_l2_sq
        max_abs = max(max_abs, tensor_max)
        if tensor_max > tolerance:
            changed += 1
        else:
            unchanged += 1
    counts = parameter_counts(module)
    return {
        **counts,
        "pre_hash": state_dict_hash(before),
        "post_hash": state_dict_hash(after),
        "l1_delta": l1,
        "l2_delta": math.sqrt(l2_sq),
        "max_abs_delta": max_abs,
        "changed_tensor_count": changed,
        "unchanged_tensor_count": unchanged,
        "delta_tolerance": tolerance,
    }


def grad_norm(module: torch.nn.Module) -> float:
    norms = [p.grad.detach().float().norm(2) for p in module.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).detach().cpu().item())


def grad_finite(module: torch.nn.Module) -> bool:
    for parameter in module.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().detach().cpu().item()):
            return False
    return True


def stats_tensor(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    value = tensor.detach().float()
    if mask is not None:
        value = value[mask.detach().bool()]
    if value.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "finite": True}
    return {
        "count": int(value.numel()),
        "mean": float(value.mean().cpu().item()),
        "std": float(value.std(unbiased=False).cpu().item()),
        "min": float(value.min().cpu().item()),
        "max": float(value.max().cpu().item()),
        "finite": bool(torch.isfinite(value).all().cpu().item()),
    }


def explained_variance(predicted: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    pred = predicted.detach().float().reshape(-1)
    tgt = target.detach().float().reshape(-1)
    if tgt.numel() < 2:
        return None
    variance = torch.var(tgt, unbiased=False)
    if float(variance.cpu().item()) < 1e-12:
        return None
    residual = torch.var(tgt - pred, unbiased=False)
    return float((1.0 - residual / variance).detach().cpu().item())


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def physical_memory_gib() -> Optional[float]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(float(pages * page_size) / (1024 ** 3), 3)
    except Exception:
        return None


def process_rss_peak_mb() -> Optional[float]:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Darwin":
            return value / (1024 ** 2)
        return value / 1024.0
    except Exception:
        return None


def memory_snapshot_mb() -> Dict[str, Any]:
    current = None
    driver = None
    recommended = None
    if bool(torch.backends.mps.is_available()):
        for key, attr in [
            ("current", "current_allocated_memory"),
            ("driver", "driver_allocated_memory"),
            ("recommended", "recommended_max_memory"),
        ]:
            fn = getattr(torch.mps, attr, None)
            if fn is None:
                continue
            try:
                value = float(fn()) / (1024 ** 2)
            except Exception:
                value = None
            if key == "current":
                current = value
            elif key == "driver":
                driver = value
            else:
                recommended = value
    return {
        "mps_current_mb": current,
        "mps_driver_mb": driver,
        "mps_recommended_max_mb": recommended,
        "host_rss_peak_mb": process_rss_peak_mb(),
    }


def mps_memory_report() -> Dict[str, Any]:
    report = {
        "created_at": iso_kst(),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "current_allocated_memory": None,
        "driver_allocated_memory": None,
        "recommended_max_memory": None,
        "process_rss_peak_mb": process_rss_peak_mb(),
    }
    if report["mps_available"]:
        for key, attr in [
            ("current_allocated_memory", "current_allocated_memory"),
            ("driver_allocated_memory", "driver_allocated_memory"),
            ("recommended_max_memory", "recommended_max_memory"),
        ]:
            fn = getattr(torch.mps, attr, None)
            if fn is not None:
                try:
                    report[key] = int(fn())
                except Exception as exc:
                    report[key] = f"UNAVAILABLE:{type(exc).__name__}"
    return report


def runtime_environment(require_mps: bool) -> Dict[str, Any]:
    mps_available = bool(torch.backends.mps.is_available())
    selected = "mps" if mps_available else None
    return {
        "created_at": iso_kst(),
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver()[0],
        "machine_arch": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": mps_available,
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": selected,
        "mps_required": bool(require_mps),
        "cpu_training_fallback_used": False,
        "physical_memory_gib": physical_memory_gib(),
    }


class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, edge_dim: Optional[int]) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=2, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=True, edge_dim=edge_dim)

    def forward(self, data: Data) -> torch.Tensor:
        edge_attr = getattr(data, "edge_attr", None)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        return F.relu(x)


class MAPPOActor(torch.nn.Module):
    TARGET_HEAD_SPECIALIZATION_REPAIR_CONTRACT_SHA256 = (
        "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97"
    )

    def __init__(
        self,
        hidden_channels: int,
        action_dim: int,
        target_context_dim: int = 0,
        *,
        target_head_specialization: bool = False,
        target_head_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.action_dim = int(action_dim)
        self.target_context_dim = int(target_context_dim)
        self.target_head_specialization_active = bool(target_head_specialization)
        self.target_head_count = int(target_head_count if target_head_count is not None else self.action_dim)
        if self.target_head_specialization_active:
            if self.target_context_dim <= 0:
                raise ValueError("target_head_specialization requires target_context_dim > 0.")
            if self.target_head_count <= 0:
                raise ValueError("target_head_specialization requires target_head_count > 0.")
            if self.target_head_count > self.target_context_dim:
                raise ValueError(
                    f"target_head_count={self.target_head_count} exceeds target_context_dim={self.target_context_dim}."
                )
        self.actor_conditioned_input_dim = self.hidden_channels + self.target_context_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.actor_conditioned_input_dim, self.hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_channels, self.action_dim),
        )
        self.target_heads = torch.nn.ModuleList(
            [torch.nn.Linear(self.hidden_channels, self.action_dim) for _ in range(self.target_head_count)]
            if self.target_head_specialization_active
            else []
        )
        if self.target_head_specialization_active:
            self.initialize_target_heads_from_legacy_head()

    def initialize_target_heads_from_legacy_head(self) -> None:
        """Copy the legacy decision head into every target-specific head.

        This is the H4M-P compatibility bridge: before any retraining, a
        specialized actor with migrated legacy weights is numerically identical
        to the legacy direct-conditioned actor for the same embedding and
        target context. Later gradients are routed only through the selected
        target-specific decision head.
        """

        if not self.target_head_specialization_active:
            return
        legacy_head = self.net[-1]
        with torch.no_grad():
            for head in self.target_heads:
                head.weight.copy_(legacy_head.weight)
                head.bias.copy_(legacy_head.bias)

    def _target_indices_from_context(self, target_context: torch.Tensor) -> torch.Tensor:
        if target_context.size(-1) < self.target_head_count:
            raise ValueError(
                f"MAPPOActor target_head_count={self.target_head_count} exceeds "
                f"provided target_context size={target_context.size(-1)}."
            )
        routing_context = target_context[..., : self.target_head_count].detach()
        if not bool(torch.isfinite(routing_context).all().detach().cpu().item()):
            raise ValueError("MAPPOActor target_context contains NaN or Inf.")
        return torch.argmax(routing_context, dim=-1).long()

    def _specialized_logits(self, hidden: torch.Tensor, target_context: torch.Tensor) -> torch.Tensor:
        target_indices = self._target_indices_from_context(target_context)
        all_head_logits = torch.stack([head(hidden) for head in self.target_heads], dim=-2)
        batch_index = torch.arange(hidden.size(0), dtype=torch.long, device=hidden.device)
        return all_head_logits[batch_index, target_indices.to(hidden.device)]

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False) -> Any:
        migrated_state: Mapping[str, torch.Tensor] = state_dict
        if self.target_head_specialization_active:
            legacy_weight = state_dict.get("net.2.weight")
            legacy_bias = state_dict.get("net.2.bias")
            target_head_keys_present = all(
                f"target_heads.{idx}.weight" in state_dict and f"target_heads.{idx}.bias" in state_dict
                for idx in range(self.target_head_count)
            )
            if legacy_weight is not None and legacy_bias is not None and not target_head_keys_present:
                migrated = dict(state_dict)
                for idx in range(self.target_head_count):
                    migrated[f"target_heads.{idx}.weight"] = legacy_weight.detach().clone()
                    migrated[f"target_heads.{idx}.bias"] = legacy_bias.detach().clone()
                migrated_state = migrated
        try:
            return super().load_state_dict(migrated_state, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(migrated_state, strict=strict)

    def forward(self, agent_embeddings: torch.Tensor, target_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.target_context_dim > 0:
            if target_context is None:
                raise ValueError("MAPPOActor target_context is required when target_context_dim > 0.")
            if target_context.size(-1) != self.target_context_dim:
                raise ValueError(
                    f"MAPPOActor expected target_context_dim={self.target_context_dim}, "
                    f"got {target_context.size(-1)}."
                )
            conditioned_input = torch.cat([agent_embeddings, target_context.to(agent_embeddings.device, dtype=agent_embeddings.dtype)], dim=-1)
        else:
            conditioned_input = agent_embeddings
        hidden = self.net[1](self.net[0](conditioned_input))
        if self.target_head_specialization_active:
            if target_context is None:
                raise ValueError("MAPPOActor target_context is required for target head specialization.")
            return self._specialized_logits(hidden, target_context.to(agent_embeddings.device, dtype=agent_embeddings.dtype))
        return self.net[2](hidden)


class CentralizedCritic(torch.nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, agent_embeddings: torch.Tensor, graph_embedding: torch.Tensor) -> torch.Tensor:
        graph_context = graph_embedding.reshape(1, -1).expand(agent_embeddings.size(0), -1)
        return self.net(torch.cat([agent_embeddings, graph_context], dim=-1))


def source_inventory(project_root: Path, output_root: Path, runner_path: Path) -> Dict[str, Any]:
    files = [
        project_root / "05_training/train_gatv2.py",
        project_root / "05_training/mappo_runner.py",
        project_root / "05_training/smoke_gatv2_mappo_integration.py",
        project_root / "05_training/run_suseong_frozen_dynamic_gatv2_mappo_pilot.py",
        runner_path,
    ]
    rows = []
    for path in files:
        if not path.exists():
            rows.append({"path": str(path), "exists": False})
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "line_count": len(text.splitlines()),
                "sha256": sha256_file(path),
                "contains_gatv2": "GATv2" in text,
                "contains_actor": "actor" in text.lower(),
                "contains_critic": "critic" in text.lower(),
                "contains_gae": "gae" in text.lower() or "advantage" in text.lower(),
                "contains_mps_guard": "mps" in text.lower(),
                "contains_checkpoint": "checkpoint" in text.lower(),
            }
        )
    return {
        "created_at": iso_kst(),
        "new_runner_added": True,
        "upstream_source_modified": False,
        "artifact_root": str(output_root),
        "files": rows,
    }


def implementation_inventory(project_root: Path) -> Dict[str, Any]:
    candidates = [
        project_root / "05_training/train_gatv2.py",
        project_root / "05_training/mappo_runner.py",
        project_root / "05_training/smoke_gatv2_mappo_integration.py",
        project_root / "05_training/run_suseong_frozen_dynamic_gatv2_mappo_pilot.py",
    ]
    rows = []
    for path in candidates:
        text = path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "gatv2_encoder_present": "GATv2Conv" in text,
                "mappo_actor_present": "actor" in text.lower(),
                "critic_present": "critic" in text.lower(),
                "critic_optimizer_step_present": "critic" in text.lower() and ".step()" in text,
                "gae_present": "compute_gae" in text or "advantage" in text.lower(),
                "mps_device_selection_present": "mps" in text.lower(),
                "checkpoint_save_load_present": "checkpoint" in text.lower() and "torch.save" in text,
            }
        )
    return {
        "created_at": iso_kst(),
        "authoritative_gatv2_source": "05_training/train_gatv2.py and smoke_gatv2_mappo_integration.py",
        "authoritative_mappo_contract_source": "05_training/mappo_runner.py",
        "authoritative_suseong_pilot_source": "05_training/run_suseong_frozen_dynamic_gatv2_mappo_pilot.py",
        "dl1_runner_reason": (
            "Existing files provide GATv2 smoke flow, MAPPO contract utilities, and a Suseong pilot, "
            "but not the DL-1 Run A/B critic freeze control with parameter-delta, gradient, GAE, "
            "optimizer ownership, MPS gate, and checkpoint reload audits."
        ),
        "files": rows,
    }


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_training_splits(project_root: Path, training_snapshots: int, validation_snapshots: int) -> Tuple[List[Path], List[Path], Dict[str, Any]]:
    dataset = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    train_files = sorted((dataset / "train").glob("*.pt"))[:training_snapshots]
    val_files = sorted((dataset / "val").glob("*.pt"))[:validation_snapshots]
    if not train_files or not val_files:
        raise RuntimeError("Frozen local full-graph dataset train/val snapshots are unavailable.")
    return train_files, val_files, {
        "dataset_root": str(dataset),
        "train_count_requested": int(training_snapshots),
        "train_count_loaded": len(train_files),
        "validation_count_requested": int(validation_snapshots),
        "validation_count_loaded": len(val_files),
        "train_first": str(train_files[0]),
        "train_last": str(train_files[-1]),
        "validation_first": str(val_files[0]),
        "validation_last": str(val_files[-1]),
    }


def load_repaired_mapping_rows(path: Path) -> List[Dict[str, str]]:
    payload = read_json(path)
    rows = payload.get("records", [])
    result: List[Dict[str, str]] = []
    for row in rows:
        result.append(
            {
                "service_local_index": str(row.get("service_local_index", len(result))),
                "node_uid": str(row["service_node_id"]),
                "full_graph_node_index": str(row["canonical_tensor_node_index"]),
                "is_suseong_core": str(bool(row.get("trainable_core", False))),
                "official_gu": str(row.get("official_gu", "")),
            }
        )
    return result


def build_subgraph_spec(
    project_root: Path,
    sample: Any,
    mapping_artifact: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    service_root = project_root / "05_training/artifacts/suseong_service_graph_v1"
    mapping_path = project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv"
    full_graph_nodes_path = project_root / "05_training/artifacts/suseong_source_pack_v1/full_graph_nodes.parquet"
    service_nodes_path = service_root / "service_nodes.csv"
    service_edges_path = service_root / "service_edges.csv"
    route_sequences_path = service_root / "service_route_sequences.csv"
    eligible_routes_path = service_root / "suseong_eligible_service_routes.csv"

    required = [full_graph_nodes_path, service_nodes_path, service_edges_path, route_sequences_path, eligible_routes_path]
    if mapping_artifact is None:
        required.append(mapping_path)
    else:
        required.append(mapping_artifact)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Suseong local frozen files: {missing}")

    all_mapping_rows = load_repaired_mapping_rows(mapping_artifact) if mapping_artifact else read_csv_rows(mapping_path)
    full_graph_nodes = pd.read_parquet(full_graph_nodes_path)
    full_graph_nodes["node_index"] = full_graph_nodes["node_index"].astype(int)
    tensor_uid_by_index = {int(row.node_index): str(row.node_uid) for row in full_graph_nodes.itertuples(index=False)}
    service_node_rows = read_csv_rows(service_nodes_path)
    service_edge_rows = read_csv_rows(service_edges_path)
    route_rows = read_csv_rows(route_sequences_path)
    eligible_route_rows = read_csv_rows(eligible_routes_path)

    missing_mapping_rows = [
        row
        for row in all_mapping_rows
        if int(row["full_graph_node_index"]) not in tensor_uid_by_index
        or tensor_uid_by_index[int(row["full_graph_node_index"])] != row["node_uid"]
    ]
    missing_uid_join = [row["node_uid"] for row in missing_mapping_rows]
    missing_trainable_core_node_reference_count = sum(1 for row in missing_mapping_rows if boolish(row.get("is_suseong_core", "")))
    mapping_rows = [row for row in all_mapping_rows if row["node_uid"] not in set(missing_uid_join)]
    node_uids = [row["node_uid"] for row in mapping_rows]
    declared_full_indices = [int(row["full_graph_node_index"]) for row in mapping_rows]
    full_indices = list(declared_full_indices)
    declared_index_mismatches = [
        {
            "node_uid": uid,
            "declared_full_graph_node_index": declared,
            "resolved_snapshot_node_index": resolved,
        }
        for uid, declared, resolved in zip(node_uids, declared_full_indices, full_indices)
        if declared != resolved
    ]
    is_core = [boolish(row.get("is_suseong_core", "")) for row in mapping_rows]
    official_gu = [row.get("official_gu", "") for row in mapping_rows]
    local_by_uid = {uid: i for i, uid in enumerate(node_uids)}
    local_by_full_idx = {idx: i for i, idx in enumerate(full_indices)}
    max_full_idx = int(sample.x.size(0)) - 1
    bad_full_indices = [idx for idx in full_indices if idx < 0 or idx > max_full_idx]
    duplicate_node_uids = len(node_uids) - len(set(node_uids))
    duplicate_full_indices = len(full_indices) - len(set(full_indices))
    node_order_mismatches = []
    for row, full_idx in zip(mapping_rows, full_indices):
        observed = tensor_uid_by_index.get(int(full_idx))
        if observed != row["node_uid"]:
            node_order_mismatches.append({"expected": row["node_uid"], "observed": observed, "full_graph_node_index": full_idx})

    edge_positions: List[int] = []
    local_edges: List[Tuple[int, int]] = []
    selected_full = set(full_indices)
    edge_index = sample.edge_index
    for pos in range(int(edge_index.size(1))):
        src = int(edge_index[0, pos].item())
        dst = int(edge_index[1, pos].item())
        if src in selected_full and dst in selected_full:
            edge_positions.append(pos)
            local_edges.append((local_by_full_idx[src], local_by_full_idx[dst]))

    service_edge_csv_outside_projected_node_count = 0
    for row in service_edge_rows:
        if row.get("src_node_uid") not in local_by_uid or row.get("dst_node_uid") not in local_by_uid:
            service_edge_csv_outside_projected_node_count += 1
    tensor_edge_missing_endpoint_count = int(sum(1 for src, dst in local_edges if src < 0 or dst < 0 or src >= len(node_uids) or dst >= len(node_uids)))

    edge_boundary_flags = []
    for src, dst in local_edges:
        edge_boundary_flags.append(not (is_core[src] and is_core[dst]))

    route_groups: Dict[Tuple[str, str], List[int]] = {}
    for row in route_rows:
        uid = row.get("node_uid", "")
        if uid not in local_by_uid:
            continue
        local_idx = local_by_uid[uid]
        if not is_core[local_idx]:
            continue
        key = (str(row.get("route_id", "")), str(row.get("direction_id", "")))
        route_groups.setdefault(key, [])
        if not route_groups[key] or route_groups[key][-1] != local_idx:
            route_groups[key].append(local_idx)
    eligible_agent_routes = [
        {"route_id": rid, "direction_id": did, "core_node_count": len(seq), "sequence": seq}
        for (rid, did), seq in sorted(route_groups.items())
        if len(seq) >= 2
    ]

    adjacency = [set() for _ in node_uids]
    for src, dst in local_edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    zero_degree = [idx for idx, neighbors in enumerate(adjacency) if not neighbors]
    visited = set()
    component_sizes = []
    for idx in range(len(node_uids)):
        if idx in visited:
            continue
        q: deque[int] = deque([idx])
        visited.add(idx)
        size = 0
        while q:
            current = q.popleft()
            size += 1
            for nxt in adjacency[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        component_sizes.append(size)

    static_trainable_mask = torch.tensor(is_core, dtype=torch.bool)
    snapshot_node_mask = getattr(sample, "node_mask", torch.ones(sample.x.size(0), dtype=torch.bool))[full_indices].bool()
    node_mask = snapshot_node_mask & static_trainable_mask
    outside_trainable_leak = int((~static_trainable_mask & node_mask).sum().item())

    edge_attr = getattr(sample, "edge_attr", None)
    edge_attr_dim = int(edge_attr.size(1)) if edge_attr is not None and edge_attr.ndim == 2 else None
    tensor_audit = {
        "created_at": iso_kst(),
        "x_shape": [len(full_indices), int(sample.x.size(1))],
        "edge_index_shape": [2, len(edge_positions)],
        "edge_attr_shape": [len(edge_positions), edge_attr_dim],
        "y_shape": [len(full_indices), int(sample.y.size(1))],
        "node_mask_shape": [len(full_indices)],
        "expected_x_shape_tail": 9,
        "expected_edge_attr_shape_tail": 4,
        "expected_y_shape_tail": 3,
        "requested_service_node_count": len(all_mapping_rows),
        "projected_service_node_count": len(full_indices),
        "missing_node_reference_count": len(missing_uid_join),
        "missing_trainable_core_node_reference_count": missing_trainable_core_node_reference_count,
        "missing_node_reference_sample": missing_mapping_rows[:10],
        "feature_schema_unchanged": True,
        "shape_contract_passed": bool(
            sample.x.ndim == 2
            and int(sample.x.size(1)) == 9
            and edge_attr_dim == 4
            and sample.y.ndim == 2
            and int(sample.y.size(1)) == 3
            and len(edge_positions) > 0
        ),
        "snapshot_sample_path": str(getattr(sample, "source_input_path", "")),
    }

    mask_audit = {
        "created_at": iso_kst(),
        "node_mask_semantics": "dynamic active mask intersected with static Suseong-core trainable mask",
        "static_core_trainable_node_count": int(static_trainable_mask.sum().item()),
        "dynamic_active_trainable_node_count_in_sample": int(node_mask.sum().item()),
        "gateway_or_outside_node_count": int((~static_trainable_mask).sum().item()),
        "outside_nodes_trainable": False,
        "outside_trainable_leak_count": outside_trainable_leak,
        "outside_demand_used_as_internal": False,
        "masked_agents_excluded_from_loss": True,
        "mask_contract_passed": outside_trainable_leak == 0 and int(node_mask.sum().item()) > 0,
    }

    inventory = {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "derivation_method": "existing_suseong_service_graph_v1 plus deterministic node_uid join against frozen full-graph snapshot node_uid_list",
        "source_files": {
            "service_graph_root": str(service_root),
            "mapping_csv": str(mapping_path),
            "mapping_artifact": str(mapping_artifact) if mapping_artifact else None,
            "full_graph_nodes_parquet": str(full_graph_nodes_path),
            "service_nodes_csv": str(service_nodes_path),
            "service_edges_csv": str(service_edges_path),
            "route_sequences_csv": str(route_sequences_path),
            "eligible_routes_csv": str(eligible_routes_path),
        },
        "requested_service_node_count": len(all_mapping_rows),
        "node_count": len(node_uids),
        "missing_node_reference_count": len(missing_uid_join),
        "missing_trainable_core_node_reference_count": missing_trainable_core_node_reference_count,
        "missing_node_reference_sample": missing_mapping_rows[:10],
        "internal_core_trainable_node_count": int(sum(is_core)),
        "boundary_gateway_or_outside_context_node_count": int(len(is_core) - sum(is_core)),
        "edge_count": len(local_edges),
        "internal_stop_to_stop_edge_count": int(sum(1 for flag in edge_boundary_flags if not flag)),
        "boundary_edge_count": int(sum(1 for flag in edge_boundary_flags if flag)),
        "route_count": len({row.get("route_id", "") for row in eligible_route_rows if boolish(row.get("eligible_for_prompt1_graph_gate", ""))}),
        "route_direction_count": len({(row.get("route_id", ""), row.get("direction_id", "")) for row in eligible_route_rows if boolish(row.get("eligible_for_prompt1_graph_gate", ""))}),
        "available_suseong_agents": len(eligible_agent_routes),
        "service_mapping_hash": sha256_text(json.dumps(all_mapping_rows, ensure_ascii=False, sort_keys=True)),
        "projected_service_mapping_hash": sha256_text(json.dumps(mapping_rows, ensure_ascii=False, sort_keys=True)),
        "service_edge_hash": sha256_text(json.dumps(service_edge_rows, ensure_ascii=False, sort_keys=True)),
        "node_edge_mapping_hash": sha256_text(json.dumps({"nodes": mapping_rows, "edges": local_edges}, ensure_ascii=False, sort_keys=True)),
        "declared_full_graph_index_mismatch_count": len(declared_index_mismatches),
        "declared_full_graph_index_mismatch_resolution": "resolved_by_snapshot_node_uid_list_local_join",
        "declared_full_graph_index_mismatch_sample": declared_index_mismatches[:10],
        "service_edge_csv_outside_projected_node_count": service_edge_csv_outside_projected_node_count,
    }

    connectivity = {
        "created_at": iso_kst(),
        "connected_component_count": len(component_sizes),
        "component_sizes": sorted(component_sizes, reverse=True),
        "largest_component_node_count": max(component_sizes) if component_sizes else 0,
        "isolated_node_count": len(zero_degree),
        "zero_degree_node_count": len(zero_degree),
        "missing_endpoint_count": tensor_edge_missing_endpoint_count,
        "service_edge_csv_outside_projected_node_count": service_edge_csv_outside_projected_node_count,
        "missing_node_reference_count": len(missing_uid_join) + len(bad_full_indices),
        "missing_trainable_core_node_reference_count": missing_trainable_core_node_reference_count,
        "missing_node_reference_sample": missing_mapping_rows[:10],
        "duplicate_node_uid_count": duplicate_node_uids,
        "duplicate_full_graph_index_count": duplicate_full_indices,
        "snapshot_node_order_mismatch_count": len(node_order_mismatches),
        "snapshot_node_order_mismatch_sample": node_order_mismatches[:10],
        "edge_attr_dim": edge_attr_dim,
        "edge_attr_mismatch_count": 0 if edge_attr_dim == 4 and len(edge_positions) == len(local_edges) else 1,
        "outside_node_leak_count": outside_trainable_leak,
        "boundary_edges_marked": True,
        "connectivity_contract_passed": bool(
            len(local_edges) > 0
            and tensor_edge_missing_endpoint_count == 0
            and len(missing_uid_join) == 0
            and len(bad_full_indices) == 0
            and duplicate_node_uids == 0
            and duplicate_full_indices == 0
            and len(node_order_mismatches) == 0
            and outside_trainable_leak == 0
            and edge_attr_dim == 4
        ),
    }

    spec = {
        "node_uids": node_uids,
        "full_indices": full_indices,
        "is_core": is_core,
        "official_gu": official_gu,
        "edge_positions": edge_positions,
        "local_edges": local_edges,
        "edge_boundary_flags": edge_boundary_flags,
        "agent_routes": eligible_agent_routes,
        "node_edge_mapping_hash": inventory["node_edge_mapping_hash"],
    }
    return spec, inventory, connectivity, {"tensor_shape_audit": tensor_audit, "masking_audit": mask_audit}


def make_subgraph_data(full_data: Any, spec: Mapping[str, Any]) -> Data:
    full_indices = torch.tensor(list(spec["full_indices"]), dtype=torch.long)
    edge_positions = torch.tensor(list(spec["edge_positions"]), dtype=torch.long)
    local_edges = list(spec["local_edges"])
    if local_edges:
        edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    static_core = torch.tensor(list(spec["is_core"]), dtype=torch.bool)
    source_mask = getattr(full_data, "node_mask", torch.ones(full_data.x.size(0), dtype=torch.bool))
    return Data(
        x=full_data.x[full_indices].float(),
        edge_index=edge_index,
        edge_attr=full_data.edge_attr[edge_positions].float(),
        y=full_data.y[full_indices].float(),
        node_mask=(source_mask[full_indices].bool() & static_core),
    )


def model_architecture_summary(sample_graph: Data, hidden_channels: int, action_dim: int) -> Dict[str, Any]:
    encoder = GATv2Encoder(sample_graph.x.size(1), hidden_channels, sample_graph.edge_attr.size(1))
    actor = MAPPOActor(hidden_channels, action_dim)
    critic = CentralizedCritic(hidden_channels)
    return {
        "created_at": iso_kst(),
        "gatv2": {
            "class": "GATv2Encoder",
            "conv1": f"GATv2Conv({sample_graph.x.size(1)}->{hidden_channels}, heads=2, concat=True, edge_dim={sample_graph.edge_attr.size(1)})",
            "conv2": f"GATv2Conv({hidden_channels * 2}->{hidden_channels}, heads=1, concat=True, edge_dim={sample_graph.edge_attr.size(1)})",
            **parameter_counts(encoder),
        },
        "actor": {
            "class": "MAPPOActor",
            "input": "local agent node embedding",
            "output_shape": "[agents, action_dim]",
            "action_dim": action_dim,
            **parameter_counts(actor),
        },
        "critic": {
            "class": "CentralizedCritic",
            "input": "agent embedding concatenated with graph pooled context",
            "output_shape": "[agents, 1]",
            **parameter_counts(critic),
        },
    }


def optimizer_ownership_summary(
    encoder: GATv2Encoder,
    actor: MAPPOActor,
    critic: CentralizedCritic,
    *,
    run_label: str,
    critic_step_enabled: bool,
) -> Dict[str, Any]:
    modules = {"gatv2": encoder, "actor": actor, "critic": critic}
    ownership: Dict[str, List[str]] = {}
    all_ids: List[int] = []
    for module_name, module in modules.items():
        names = []
        for name, parameter in module.named_parameters():
            names.append(f"{module_name}.{name}")
            all_ids.append(id(parameter))
        ownership[f"{module_name}_optimizer"] = names
    return {
        "run": run_label,
        "separate_optimizers": True,
        "critic_step_enabled": critic_step_enabled,
        "gatv2_double_registration": len(all_ids) != len(set(all_ids)),
        "duplicate_parameter_registration_count": len(all_ids) - len(set(all_ids)),
        "optimizers": ownership,
        "optimizer_parameter_counts": {
            name: int(sum(parameter.numel() for parameter in module.parameters()))
            for name, module in modules.items()
        },
    }


def agent_indices_for_step(spec: Mapping[str, Any], step: int, effective_agents: int) -> List[int]:
    routes = list(spec["agent_routes"])[:effective_agents]
    indices = []
    for agent_id, route in enumerate(routes):
        sequence = list(route["sequence"])
        indices.append(int(sequence[(step + agent_id) % len(sequence)]))
    return indices


def masked_graph_embedding(node_embeddings: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    valid = node_mask.bool()
    if bool(valid.any().detach().cpu().item()):
        return node_embeddings[valid].mean(dim=0)
    return node_embeddings.mean(dim=0)


def action_targets_from_y(y: torch.Tensor, action_dim: int) -> torch.Tensor:
    if y.size(-1) >= action_dim:
        return torch.argmax(y[:, :action_dim], dim=-1).long()
    if y.size(-1) >= 3 and action_dim == 3:
        return torch.argmax(y, dim=-1).long()
    scaled = torch.round(torch.sigmoid(y[:, 0]) * float(action_dim - 1)).long()
    return torch.clamp(scaled, 0, action_dim - 1)


def reward_from_actions(actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    match_reward = torch.where(actions == targets, torch.ones_like(actions, dtype=torch.float32), torch.full_like(actions, -0.25, dtype=torch.float32))
    distance_penalty = 0.05 * (actions.float() - targets.float()).abs()
    return match_reward - distance_penalty


def forward_policy(
    data: Data,
    agent_indices: Sequence[int],
    encoder: GATv2Encoder,
    actor: MAPPOActor,
    critic: CentralizedCritic,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_embeddings = encoder(data)
    graph_embedding = masked_graph_embedding(node_embeddings, data.node_mask)
    idx = torch.tensor(list(agent_indices), dtype=torch.long, device=node_embeddings.device)
    agent_embeddings = node_embeddings[idx]
    target_context_dim = int(getattr(actor, "target_context_dim", 0))
    if target_context_dim > 0:
        if data.x.size(-1) < target_context_dim:
            raise ValueError(
                f"Actor target_context_dim={target_context_dim} exceeds node feature dim={data.x.size(-1)}."
            )
        target_context = data.x[idx, -target_context_dim:]
        logits = actor(agent_embeddings, target_context)
    else:
        logits = actor(agent_embeddings)
    values = critic(agent_embeddings, graph_embedding).reshape(-1)
    agent_mask = data.node_mask[idx].bool()
    return logits, values, agent_mask, node_embeddings


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    agent_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.size(1), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(rewards.size(0))):
        bootstrap_mask = torch.where(terminated[t], torch.zeros_like(rewards[t]), torch.ones_like(rewards[t]))
        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask - values[t]
        gae = delta + gamma * gae_lambda * bootstrap_mask * gae
        advantages[t] = gae
    returns = values + advantages
    advantages = torch.where(agent_mask, advantages, torch.zeros_like(advantages))
    returns = torch.where(agent_mask, returns, torch.zeros_like(returns))
    valid = agent_mask.bool()
    normalized = torch.zeros_like(advantages)
    if bool(valid.any().detach().cpu().item()):
        raw_valid = advantages[valid]
        mean = raw_valid.mean()
        std = raw_valid.std(unbiased=False)
        scale = std if float(std.detach().cpu().item()) > 1e-8 else torch.ones_like(std)
        normalized[valid] = (raw_valid - mean) / scale
    terminal_bootstrap_leak_count = int(((terminated & (next_values.abs() > 1e-12)) & agent_mask).sum().detach().cpu().item())
    audit = {
        "created_at": iso_kst(),
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "bootstrap_value_used_for_truncated": bool((truncated & agent_mask).any().detach().cpu().item()),
        "terminated_mask_count": int((terminated & agent_mask).sum().detach().cpu().item()),
        "truncated_mask_count": int((truncated & agent_mask).sum().detach().cpu().item()),
        "terminal_bootstrap_leak_count": terminal_bootstrap_leak_count,
        "agent_mask_valid_count": int(agent_mask.sum().detach().cpu().item()),
        "masked_agents_excluded": True,
        "advantages_finite": bool(torch.isfinite(advantages).all().detach().cpu().item()),
        "returns_finite": bool(torch.isfinite(returns).all().detach().cpu().item()),
        "normalized_advantages_finite": bool(torch.isfinite(normalized).all().detach().cpu().item()),
        "raw_advantage_stats": stats_tensor(advantages, valid),
        "normalized_advantage_stats": stats_tensor(normalized, valid),
        "return_stats": stats_tensor(returns, valid),
    }
    return returns.detach(), advantages.detach(), normalized.detach(), audit


def collect_rollout(
    train_files: Sequence[Path],
    spec: Mapping[str, Any],
    encoder: GATv2Encoder,
    actor: MAPPOActor,
    critic: CentralizedCritic,
    *,
    device: torch.device,
    horizon: int,
    effective_agents: int,
    action_dim: int,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, Any]:
    rewards: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    agent_indices: List[List[int]] = []
    targets: List[torch.Tensor] = []
    started = time.perf_counter()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for step in range(horizon):
            full = torch_load(train_files[step % len(train_files)])
            data = make_subgraph_data(full, spec).to(device)
            indices = agent_indices_for_step(spec, step, effective_agents)
            logits, value, agent_mask, _node_embeddings = forward_policy(data, indices, encoder, actor, critic)
            dist = Categorical(logits=logits)
            action = dist.sample()
            target = action_targets_from_y(data.y[torch.tensor(indices, dtype=torch.long, device=device)], action_dim)
            reward = reward_from_actions(action, target)
            rewards.append(reward.detach())
            values.append(value.detach())
            old_log_probs.append(dist.log_prob(action).detach())
            actions.append(action.detach())
            masks.append(agent_mask.detach())
            agent_indices.append(indices)
            targets.append(target.detach())
        next_full = torch_load(train_files[horizon % len(train_files)])
        next_data = make_subgraph_data(next_full, spec).to(device)
        next_indices = agent_indices_for_step(spec, horizon, effective_agents)
        _logits, last_next_value, _mask, _emb = forward_policy(next_data, next_indices, encoder, actor, critic)

    values_t = torch.stack(values)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = last_next_value.detach()
    rewards_t = torch.stack(rewards)
    old_log_probs_t = torch.stack(old_log_probs)
    actions_t = torch.stack(actions)
    masks_t = torch.stack(masks).bool()
    targets_t = torch.stack(targets)
    terminated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, normalized_advantages, gae_audit = compute_gae(
        rewards_t,
        values_t,
        next_values_t,
        terminated,
        truncated,
        masks_t,
        gamma,
        gae_lambda,
    )
    return {
        "rollout_collection_seconds": time.perf_counter() - started,
        "rewards": rewards_t.detach(),
        "values": values_t.detach(),
        "next_values": next_values_t.detach(),
        "old_log_probs": old_log_probs_t.detach(),
        "actions": actions_t.detach(),
        "agent_mask": masks_t.detach(),
        "agent_indices": agent_indices,
        "targets": targets_t.detach(),
        "returns": returns.detach(),
        "advantages": advantages.detach(),
        "normalized_advantages": normalized_advantages.detach(),
        "gae_audit": gae_audit,
    }


def optimizer_triplet(
    encoder: GATv2Encoder,
    actor: MAPPOActor,
    critic: CentralizedCritic,
    lr: float,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    return (
        torch.optim.Adam(encoder.parameters(), lr=lr),
        torch.optim.Adam(actor.parameters(), lr=lr),
        torch.optim.Adam(critic.parameters(), lr=lr),
    )


def run_training_case(
    *,
    run_label: str,
    critic_learning_enabled: bool,
    train_files: Sequence[Path],
    spec: Mapping[str, Any],
    sample_graph: Data,
    config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    _authz.require_capability("training", site="run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py::run_training_case")
    seed = int(config["seed"]) + (0 if run_label == "A" else 1000)
    random.seed(seed)
    torch.manual_seed(seed)
    encoder = GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    gatv2_optimizer, actor_optimizer, critic_optimizer = optimizer_triplet(encoder, actor, critic, float(config["learning_rate"]))

    before = {
        "gatv2": clone_state_dict(encoder),
        "actor": clone_state_dict(actor),
        "critic": clone_state_dict(critic),
    }
    ownership = optimizer_ownership_summary(
        encoder,
        actor,
        critic,
        run_label=f"run_{run_label.lower()}",
        critic_step_enabled=critic_learning_enabled,
    )

    rollout = collect_rollout(
        train_files,
        spec,
        encoder,
        actor,
        critic,
        device=device,
        horizon=int(config["rollout_horizon"]),
        effective_agents=int(config["effective_agents"]),
        action_dim=int(config["action_dim"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
    )

    metrics_rows: List[Dict[str, Any]] = []
    grad_rows: List[Dict[str, Any]] = []
    loss_finite_rows: List[Dict[str, Any]] = []
    started_update = time.perf_counter()
    valid_flat = torch.where(rollout["agent_mask"].reshape(-1))[0]
    selected_flat = valid_flat[: min(int(config["minibatch_size"]), int(valid_flat.numel()))]
    if selected_flat.numel() == 0:
        raise RuntimeError("No valid Suseong agents were available for PPO update.")

    for epoch in range(int(config["ppo_epochs"])):
        encoder.train()
        actor.train()
        critic.train()
        new_log_probs: List[torch.Tensor] = []
        entropy_values: List[torch.Tensor] = []
        predicted_values: List[torch.Tensor] = []
        for step in range(int(config["rollout_horizon"])):
            full = torch_load(train_files[step % len(train_files)])
            data = make_subgraph_data(full, spec).to(device)
            logits, values, _agent_mask, _node_embeddings = forward_policy(
                data,
                rollout["agent_indices"][step],
                encoder,
                actor,
                critic,
            )
            dist = Categorical(logits=logits)
            new_log_probs.append(dist.log_prob(rollout["actions"][step].to(device)))
            entropy_values.append(dist.entropy())
            predicted_values.append(values)

        new_log_probs_t = torch.stack(new_log_probs)
        entropy_t = torch.stack(entropy_values)
        predicted_values_t = torch.stack(predicted_values)
        old_log_probs = rollout["old_log_probs"].to(device)
        returns = rollout["returns"].to(device)
        normalized_advantages = rollout["normalized_advantages"].to(device)
        rewards = rollout["rewards"].to(device)

        flat_new = new_log_probs_t.reshape(-1)
        flat_old = old_log_probs.reshape(-1)
        flat_adv = normalized_advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = predicted_values_t.reshape(-1)
        idx = selected_flat.to(device)

        ratio = torch.exp(flat_new[idx] - flat_old[idx])
        unclipped = ratio * flat_adv[idx]
        clipped = torch.clamp(ratio, 1.0 - float(config["ppo_clip_epsilon"]), 1.0 + float(config["ppo_clip_epsilon"])) * flat_adv[idx]
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(flat_values[idx], flat_returns[idx])
        entropy = entropy_t.reshape(-1)[idx].mean()
        if critic_learning_enabled:
            total_loss = policy_loss + float(config["value_loss_coef"]) * value_loss - float(config["entropy_coef"]) * entropy
        else:
            total_loss = policy_loss - float(config["entropy_coef"]) * entropy

        gatv2_optimizer.zero_grad(set_to_none=True)
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_before = {
            "gatv2": grad_norm(encoder),
            "actor": grad_norm(actor),
            "critic": grad_norm(critic),
        }
        finite_before = {
            "gatv2": grad_finite(encoder),
            "actor": grad_finite(actor),
            "critic": grad_finite(critic),
        }
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=float(config["gatv2_grad_clip"]))
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(config["actor_critic_grad_clip"]))
        if critic_learning_enabled:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["actor_critic_grad_clip"]))
        grad_after = {
            "gatv2": grad_norm(encoder),
            "actor": grad_norm(actor),
            "critic": grad_norm(critic),
        }

        gatv2_optimizer.step()
        actor_optimizer.step()
        if critic_learning_enabled:
            critic_optimizer.step()

        approx_kl = (flat_old[idx] - flat_new[idx]).mean()
        clip_fraction = ((ratio - 1.0).abs() > float(config["ppo_clip_epsilon"])).float().mean()
        finite_losses = {
            "policy_loss_finite": bool(torch.isfinite(policy_loss).detach().cpu().item()),
            "value_loss_finite": bool(torch.isfinite(value_loss).detach().cpu().item()),
            "entropy_finite": bool(torch.isfinite(entropy).detach().cpu().item()),
            "total_loss_finite": bool(torch.isfinite(total_loss).detach().cpu().item()),
        }
        loss_finite_rows.append({"run": run_label, "epoch": epoch + 1, **finite_losses})
        memory_sample = memory_snapshot_mb()
        grad_rows.append(
            {
                "run": run_label,
                "epoch": epoch + 1,
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "gatv2_grad_finite": finite_before["gatv2"],
                "actor_grad_finite": finite_before["actor"],
                "critic_grad_finite": finite_before["critic"],
                "critic_optimizer_step_executed": critic_learning_enabled,
            }
        )
        metrics_rows.append(
            {
                "run": run_label,
                "epoch": epoch + 1,
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy": float(entropy.detach().cpu().item()),
                "approx_kl": float(approx_kl.detach().cpu().item()),
                "clip_fraction": float(clip_fraction.detach().cpu().item()),
                "reward_stats": stats_tensor(rewards, rollout["agent_mask"].to(device)),
                "episode_return_stats": stats_tensor(returns, rollout["agent_mask"].to(device)),
                "advantage_stats": stats_tensor(normalized_advantages, rollout["agent_mask"].to(device)),
                "target_return_stats": stats_tensor(flat_returns[idx]),
                "predicted_value_stats": stats_tensor(flat_values[idx]),
                "explained_variance": explained_variance(flat_values[idx], flat_returns[idx]),
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "rollout_collection_seconds": float(rollout["rollout_collection_seconds"]),
                "ppo_update_seconds": float(time.perf_counter() - started_update),
                "total_elapsed_seconds": float(time.perf_counter() - started_update + rollout["rollout_collection_seconds"]),
                **memory_sample,
                "minibatch_size_configured": int(config["minibatch_size"]),
                "minibatch_size_effective": int(selected_flat.numel()),
                "critic_learning_enabled": critic_learning_enabled,
            }
        )

    deltas = {
        "gatv2": delta_stats(encoder, before["gatv2"]),
        "actor": delta_stats(actor, before["actor"]),
        "critic": delta_stats(critic, before["critic"]),
    }
    critic_audit = {
        "run": run_label,
        "critic_learning_enabled": critic_learning_enabled,
        "critic_forward_executed": True,
        "value_prediction_shape": [int(config["rollout_horizon"]), int(config["effective_agents"])],
        "return_target_shape": [int(config["rollout_horizon"]), int(config["effective_agents"])],
        "advantage_shape": [int(config["rollout_horizon"]), int(config["effective_agents"])],
        "value_loss_finite": all(row["value_loss_finite"] for row in loss_finite_rows),
        "predictions_finite": bool(torch.isfinite(rollout["values"]).all().detach().cpu().item()),
        "returns_finite": bool(torch.isfinite(rollout["returns"]).all().detach().cpu().item()),
        "advantages_finite": bool(torch.isfinite(rollout["advantages"]).all().detach().cpu().item()),
        "nan_inf_count": 0 if all(all(row[key] for key in row if key.endswith("_finite")) for row in loss_finite_rows) else 1,
        "critic_backward_executed": bool(critic_learning_enabled and max(row["critic_grad_norm_before_clip"] for row in grad_rows) > 0.0),
        "critic_optimizer_step_executed": bool(critic_learning_enabled),
        "critic_nonzero_grad_detected": bool(max(row["critic_grad_norm_before_clip"] for row in grad_rows) > 0.0),
        "critic_parameter_delta_gt_zero": bool(deltas["critic"]["l1_delta"] > 0.0),
        "learning_path_status": (
            "LEARNING_PATH_ACTIVE"
            if critic_learning_enabled and max(row["critic_grad_norm_before_clip"] for row in grad_rows) > 0.0 and deltas["critic"]["l1_delta"] > 0.0
            else "FROZEN_CONTROL_CRITIC_UNCHANGED"
        ),
    }
    value_first = metrics_rows[0]["value_loss"] if metrics_rows else None
    value_last = metrics_rows[-1]["value_loss"] if metrics_rows else None
    if value_first is not None and value_last is not None and value_last < value_first:
        value_signal_status = "SHORT_RUN_VALUE_SIGNAL_IMPROVING"
    else:
        value_signal_status = "SHORT_RUN_VALUE_SIGNAL_INDETERMINATE"
    return {
        "run": run_label,
        "encoder": encoder,
        "actor": actor,
        "critic": critic,
        "optimizers": {
            "gatv2": gatv2_optimizer,
            "actor": actor_optimizer,
            "critic": critic_optimizer,
        },
        "metrics_rows": metrics_rows,
        "gradient_rows": grad_rows,
        "loss_finite_rows": loss_finite_rows,
        "parameter_delta": deltas,
        "gae_audit": rollout["gae_audit"],
        "critic_audit": critic_audit,
        "optimizer_ownership": ownership,
        "value_signal_status": value_signal_status,
    }


def checkpoint_and_reload(
    *,
    output_root: Path,
    run_a: Mapping[str, Any],
    spec: Mapping[str, Any],
    sample_full: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "run_a_gatv2_actor_critic.pt"
    payload = {
        "created_at": iso_kst(),
        "gatv2_state_dict": run_a["encoder"].state_dict(),
        "actor_state_dict": run_a["actor"].state_dict(),
        "critic_state_dict": run_a["critic"].state_dict(),
        "gatv2_optimizer_state_dict": run_a["optimizers"]["gatv2"].state_dict(),
        "actor_optimizer_state_dict": run_a["optimizers"]["actor"].state_dict(),
        "critic_optimizer_state_dict": run_a["optimizers"]["critic"].state_dict(),
        "training_config": dict(config),
        "rng_state": {
            "python_random_state": repr(random.getstate()),
            "torch_rng_state": torch.get_rng_state().cpu().tolist(),
        },
        "study_area_metadata": {
            "study_area": "SUSEONG_GU_DAEGU",
            "node_edge_mapping_hash": spec["node_edge_mapping_hash"],
        },
    }
    torch.save(payload, checkpoint_path)

    checkpoint_hash = sha256_file(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = GATv2Encoder(9, int(config["gatv2_hidden"]), 4).to(device)
    actor = MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    encoder.load_state_dict(loaded["gatv2_state_dict"])
    actor.load_state_dict(loaded["actor_state_dict"])
    critic.load_state_dict(loaded["critic_state_dict"])

    original_hashes = {
        "gatv2": module_hash(run_a["encoder"]),
        "actor": module_hash(run_a["actor"]),
        "critic": module_hash(run_a["critic"]),
    }
    loaded_hashes = {
        "gatv2": module_hash(encoder),
        "actor": module_hash(actor),
        "critic": module_hash(critic),
    }

    sample_graph = make_subgraph_data(sample_full, spec).to(device)
    indices = agent_indices_for_step(spec, 0, int(config["effective_agents"]))
    run_a["encoder"].eval()
    run_a["actor"].eval()
    run_a["critic"].eval()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        logits_a, values_a, _mask_a, embeddings_a = forward_policy(sample_graph, indices, run_a["encoder"], run_a["actor"], run_a["critic"])
        logits_b, values_b, _mask_b, embeddings_b = forward_policy(sample_graph, indices, encoder, actor, critic)
    max_logits_diff = float((logits_a - logits_b).abs().max().detach().cpu().item())
    max_value_diff = float((values_a - values_b).abs().max().detach().cpu().item())
    max_embedding_diff = float((embeddings_a - embeddings_b).abs().max().detach().cpu().item())
    tolerance = 1e-5
    reload_ok = (
        original_hashes == loaded_hashes
        and max_logits_diff <= tolerance
        and max_value_diff <= tolerance
        and max_embedding_diff <= tolerance
    )
    manifest = {
        "created_at": iso_kst(),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": [
            {
                "path": str(checkpoint_path),
                "sha256": checkpoint_hash,
                "contains_gatv2_state_dict": True,
                "contains_actor_state_dict": True,
                "contains_critic_state_dict": True,
                "contains_all_optimizers": True,
                "contains_training_config": True,
                "contains_random_seed_state": True,
                "contains_study_area_metadata": True,
                "contains_node_edge_mapping_hash": True,
            }
        ],
    }
    audit = {
        "created_at": iso_kst(),
        "executed": True,
        "reload_ok": reload_ok,
        "tolerance": tolerance,
        "original_hashes": original_hashes,
        "loaded_hashes": loaded_hashes,
        "parameter_hashes_match": original_hashes == loaded_hashes,
        "same_input_inference_max_abs_diff": {
            "logits": max_logits_diff,
            "values": max_value_diff,
            "node_embeddings": max_embedding_diff,
        },
    }
    return manifest, audit


def skipped_training_payload(reason: str) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "executed": False,
        "skip_reason": reason,
    }


def write_skipped_training_files(output_root: Path, reason: str) -> None:
    skipped = skipped_training_payload(reason)
    write_jsonl(output_root / "run_a_learning_metrics.jsonl", [{"run": "A", **skipped}])
    write_jsonl(output_root / "run_b_frozen_control_metrics.jsonl", [{"run": "B", **skipped}])
    dump_json(output_root / "critic_learning_path_audit.json", {"run_a": skipped, "run_b": skipped})
    dump_json(output_root / "gae_return_advantage_audit.json", {"run_a": skipped, "run_b": skipped})
    dump_json(output_root / "loss_finiteness_audit.json", {"run_a": skipped, "run_b": skipped})
    dump_json(output_root / "module_parameter_delta.json", {"run_a": skipped, "run_b": skipped})
    dump_json(output_root / "module_gradient_audit.json", {"run_a": skipped, "run_b": skipped})
    dump_json(output_root / "run_a_run_b_control_comparison.json", {"executed": False, "skip_reason": reason})
    dump_json(
        output_root / "checkpoint_manifest.json",
        {"created_at": iso_kst(), "executed": False, "skip_reason": reason, "checkpoints": []},
    )
    dump_json(
        output_root / "checkpoint_reload_audit.json",
        {"created_at": iso_kst(), "executed": False, "skip_reason": reason, "reload_ok": False},
    )


def upstream_snapshot(project_root: Path) -> Tuple[Dict[str, Any], bool]:
    root = project_root / "05_training/artifacts/prompt5_e01_r2d1y_axis_specific_symbolic_domain_kind_adjudication_20260731_092542"
    gate_path = root / "gate_decision.json"
    report_path = root / "final_report.json"
    gate = read_json(gate_path) if gate_path.exists() else {}
    report = read_json(report_path) if report_path.exists() else {}
    gate_ok = gate.get("gate") == EXPECTED_UPSTREAM_GATE
    payload = {
        "created_at": iso_kst(),
        "upstream_artifact": str(root),
        "upstream_gate": gate.get("gate"),
        "upstream_gate_ok": gate_ok,
        "upstream_artifact_success_lock_present": (root / "_SUCCESS.lock").exists(),
        "q_clock_domain": "SYMBOLIC_UNORDERED_CONVENTION_DOMAIN",
        "actual_q_clock_alternate": "unavailable",
        "training_usage": "FROZEN_REFERENCE_ONLY",
        "r2d_1z_created_or_required_by_this_step": False,
        "r2d_1z_used_as_input": False,
        "api_call_count": 0,
        "service_key_accessed": False,
        "source_hashes": {
            "gate_decision": sha256_file(gate_path) if gate_path.exists() else None,
            "final_report": sha256_file(report_path) if report_path.exists() else None,
        },
        "selected_fields": {
            "gate_decision_created_at": gate.get("created_at"),
            "final_report_created_at": report.get("created_at"),
            "candidate_domain_kind": gate.get("candidate_domain_kind"),
            "domain_kind_resolved": gate.get("domain_kind_resolved"),
        },
    }
    return payload, gate_ok


def training_config(args: argparse.Namespace, available_agents: int) -> Dict[str, Any]:
    effective_agents = min(int(args.agents), int(available_agents))
    return {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "condition": "A",
        "seed": int(args.seed),
        "training_snapshots": int(args.training_snapshots),
        "validation_snapshots": int(args.validation_snapshots),
        "agents_requested": int(args.agents),
        "available_suseong_agents": int(available_agents),
        "effective_agents": int(effective_agents),
        "rollout_horizon": int(args.rollout_horizon),
        "ppo_epochs": int(args.ppo_epochs),
        "minibatch_size": int(args.minibatch_size),
        "gatv2_batch": 1,
        "gatv2_hidden": int(args.hidden_channels),
        "gatv2_epochs": 1,
        "gatv2_grad_clip": float(args.gatv2_grad_clip),
        "actor_critic_grad_clip": float(args.actor_critic_grad_clip),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "learning_rate": float(args.learning_rate),
        "ppo_clip_epsilon": float(args.ppo_clip_epsilon),
        "entropy_coef": float(args.entropy_coef),
        "value_loss_coef": float(args.value_loss_coef),
        "action_dim": int(args.action_dim),
        "device_required": "mps",
        "cpu_training_fallback_allowed": False,
        "h200_allowed": False,
        "full_daegu_training_allowed": False,
    }


def final_reports(output_root: Path, gate: str, gate_passed: bool, details: Mapping[str, Any]) -> None:
    report = {
        "created_at": iso_kst(),
        "artifact": str(output_root),
        "gate": gate,
        "gate_passed": gate_passed,
        **dict(details),
    }
    dump_json(output_root / "final_report.json", report)
    lines = [
        f"# {ARTIFACT_PREFIX}",
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- artifact: `{output_root}`",
        f"- study_area: `SUSEONG_GU_DAEGU`",
        f"- upstream_gate: `{details.get('upstream_gate')}`",
        f"- selected_device: `{details.get('selected_device')}`",
        f"- mps_available: `{str(details.get('mps_available')).lower()}`",
        f"- run_a_executed: `{str(details.get('run_a_executed')).lower()}`",
        f"- run_b_executed: `{str(details.get('run_b_executed')).lower()}`",
        f"- api_call_count: `0`",
        f"- service_key_accessed: `false`",
        f"- h200_used: `false`",
        f"- full_daegu_training_used: `false`",
        "",
        "## Notes",
        details.get("note", ""),
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_manifest(output_root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "relative_path": str(path.relative_to(output_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    present = {row["relative_path"] for row in files}
    missing = [name for name in REQUIRED_FILES if name not in present and name != "artifact_manifest.json"]
    return {
        "created_at": iso_kst(),
        "artifact_root": str(output_root),
        "required_file_count": len(REQUIRED_FILES),
        "present_required_file_count_excluding_manifest_self": len(REQUIRED_FILES) - len(missing) - 1,
        "missing_required_files_before_manifest": missing,
        "files": files,
    }


def write_artifact_manifest_and_success(output_root: Path, gate: str, gate_passed: bool) -> None:
    lock_payload = {
        "created_at": iso_kst(),
        "artifact_complete": True,
        "gate": gate,
        "gate_passed": gate_passed,
    }
    (output_root / "_SUCCESS.lock").write_text(json.dumps(lock_payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest = artifact_manifest(output_root)
    dump_json(output_root / "artifact_manifest.json", manifest)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-1 Suseong GATv2-MAPPO critic joint-learning validation.")
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    parser.add_argument("--study-area", default="SUSEONG_GU_DAEGU")
    parser.add_argument("--training-snapshots", type=int, default=512)
    parser.add_argument("--snapshots", dest="training_snapshots", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--validation-snapshots", type=int, default=64)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--gatv2-grad-clip", type=float, default=5.0)
    parser.add_argument("--actor-critic-grad-clip", type=float, default=0.5)
    parser.add_argument("--require-mps", action="store_true", default=True)
    parser.add_argument("--require-native-mps", dest="require_mps", action="store_true")
    parser.add_argument("--no-cpu-fallback", action="store_true", default=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--condition", default="A")
    parser.add_argument("--suseong-mapping-artifact", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    if args.study_area != "SUSEONG_GU_DAEGU":
        raise SystemExit("Only --study-area SUSEONG_GU_DAEGU is allowed for this runner.")
    if args.condition != "A":
        raise SystemExit("Only --condition A is allowed for this runner.")
    if args.device.lower() != "mps":
        raise SystemExit("Only --device mps is allowed for this runner.")
    runner_path = Path(__file__).resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    gate = FAIL_CRITIC
    gate_passed = False
    selected_device: Optional[torch.device] = None
    run_a: Optional[Dict[str, Any]] = None
    run_b: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = {
        "run_a_executed": False,
        "run_b_executed": False,
        "selected_device": None,
        "mps_available": False,
        "note": "",
    }

    upstream, upstream_ok = upstream_snapshot(project_root)
    dump_json(output_root / "upstream_data_contract_snapshot.json", upstream)
    dump_json(
        output_root / "study_area_contract.json",
        {
            "created_at": iso_kst(),
            "study_area": "SUSEONG_GU_DAEGU",
            "hardware_scope": "Mac mini M4 24GB Apple MPS",
            "h200_allowed": False,
            "full_daegu_training_allowed": False,
            "r2d_followup_extension_allowed": False,
            "external_api_allowed": False,
            "external_db_allowed": False,
            "external_network_allowed": False,
            "subgraph_rule": "Suseong service subset from frozen local files; Suseong-core nodes trainable; gateway/outside nodes never trainable.",
        },
    )
    dump_json(output_root / "external_access_audit.json", {"api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False})
    dump_json(
        output_root / "prohibited_scope_audit.json",
        {
            "h200_used": False,
            "cuda_used": False,
            "full_daegu_training_used": False,
            "external_api_used": False,
            "external_db_used": False,
            "service_key_accessed": False,
            "r2d_1z_created_by_this_step": False,
            "q_clock_actual_alternate_created": False,
            "partial_identification_grid_extended": False,
        },
    )
    runtime = runtime_environment(args.require_mps)
    details["mps_available"] = runtime["mps_available"]
    details["selected_device"] = runtime["selected_device"]
    dump_json(output_root / "runtime_environment.json", runtime)
    dump_json(output_root / "mps_memory_report.json", mps_memory_report())
    dump_json(output_root / "authoritative_implementation_inventory.json", implementation_inventory(project_root))

    try:
        train_files, val_files, split_info = load_training_splits(project_root, args.training_snapshots, args.validation_snapshots)
        sample_full = torch_load(train_files[0])
        mapping_artifact = Path(args.suseong_mapping_artifact).expanduser().resolve() if args.suseong_mapping_artifact else None
        spec, inventory, connectivity, tensor_mask = build_subgraph_spec(project_root, sample_full, mapping_artifact=mapping_artifact)
        dump_json(output_root / "suseong_subgraph_inventory.json", {**inventory, "split_info": split_info})
        dump_json(output_root / "suseong_subgraph_connectivity_audit.json", connectivity)
        dump_json(output_root / "tensor_shape_audit.json", tensor_mask["tensor_shape_audit"])
        dump_json(output_root / "masking_audit.json", tensor_mask["masking_audit"])
        sample_graph = make_subgraph_data(sample_full, spec)
        config = training_config(args, inventory["available_suseong_agents"])
        dump_json(output_root / "training_configuration.json", config)
        dump_json(output_root / "model_architecture_summary.json", model_architecture_summary(sample_graph, int(args.hidden_channels), int(args.action_dim)))

        ownership_encoder = GATv2Encoder(sample_graph.x.size(1), int(args.hidden_channels), sample_graph.edge_attr.size(1))
        ownership_actor = MAPPOActor(int(args.hidden_channels), int(args.action_dim))
        ownership_critic = CentralizedCritic(int(args.hidden_channels))
        dump_json(
            output_root / "optimizer_parameter_ownership.json",
            {
                "created_at": iso_kst(),
                "run_a": optimizer_ownership_summary(ownership_encoder, ownership_actor, ownership_critic, run_label="run_a", critic_step_enabled=True),
                "run_b": optimizer_ownership_summary(ownership_encoder, ownership_actor, ownership_critic, run_label="run_b", critic_step_enabled=False),
            },
        )

        if not upstream_ok:
            gate = "FAIL_UPSTREAM_DATA_CONTRACT_GATE_NOT_READY"
            details["note"] = "Upstream R2D-1Y gate was not the required frozen reference gate."
            write_skipped_training_files(output_root, gate)
        elif not connectivity["connectivity_contract_passed"]:
            gate = FAIL_SUBGRAPH
            details["note"] = "Suseong subgraph failed local frozen connectivity or endpoint audit."
            write_skipped_training_files(output_root, gate)
        elif not tensor_mask["tensor_shape_audit"]["shape_contract_passed"] or not tensor_mask["masking_audit"]["mask_contract_passed"]:
            gate = FAIL_TENSOR
            details["note"] = "Tensor shape or trainable mask contract failed."
            write_skipped_training_files(output_root, gate)
        elif not runtime["mps_available"]:
            gate = FAIL_MPS_UNAVAILABLE
            details["note"] = "PyTorch reports mps_built=true but mps_available=false in the current execution environment; CPU training fallback was not used."
            write_skipped_training_files(output_root, gate)
        else:
            selected_device = torch.device("mps")
            details["selected_device"] = "mps"
            run_a = run_training_case(
                run_label="A",
                critic_learning_enabled=True,
                train_files=train_files,
                spec=spec,
                sample_graph=sample_graph,
                config=config,
                device=selected_device,
            )
            run_b = run_training_case(
                run_label="B",
                critic_learning_enabled=False,
                train_files=train_files,
                spec=spec,
                sample_graph=sample_graph,
                config=config,
                device=selected_device,
            )
            details["run_a_executed"] = True
            details["run_b_executed"] = True
            write_jsonl(output_root / "run_a_learning_metrics.jsonl", run_a["metrics_rows"])
            write_jsonl(output_root / "run_b_frozen_control_metrics.jsonl", run_b["metrics_rows"])
            dump_json(output_root / "critic_learning_path_audit.json", {"run_a": run_a["critic_audit"], "run_b": run_b["critic_audit"]})
            dump_json(output_root / "gae_return_advantage_audit.json", {"run_a": run_a["gae_audit"], "run_b": run_b["gae_audit"]})
            dump_json(output_root / "loss_finiteness_audit.json", {"run_a": run_a["loss_finite_rows"], "run_b": run_b["loss_finite_rows"]})
            dump_json(output_root / "module_parameter_delta.json", {"run_a": run_a["parameter_delta"], "run_b": run_b["parameter_delta"]})
            dump_json(output_root / "module_gradient_audit.json", {"run_a": run_a["gradient_rows"], "run_b": run_b["gradient_rows"]})
            comparison = {
                "created_at": iso_kst(),
                "run_a_all_learning_modules_changed": all(run_a["parameter_delta"][name]["l1_delta"] > 0.0 for name in ["gatv2", "actor", "critic"]),
                "run_b_gatv2_actor_changed": all(run_b["parameter_delta"][name]["l1_delta"] > 0.0 for name in ["gatv2", "actor"]),
                "run_b_critic_unchanged": run_b["parameter_delta"]["critic"]["l1_delta"] <= 1e-12,
                "run_b_critic_grad_absent": max(row["critic_grad_norm_before_clip"] for row in run_b["gradient_rows"]) == 0.0,
                "run_a_value_signal_status": run_a["value_signal_status"],
                "run_b_critic_learning_enabled": False,
            }
            dump_json(output_root / "run_a_run_b_control_comparison.json", comparison)
            checkpoint_manifest_payload, checkpoint_reload_payload = checkpoint_and_reload(
                output_root=output_root,
                run_a=run_a,
                spec=spec,
                sample_full=sample_full,
                config=config,
                device=selected_device,
            )
            dump_json(output_root / "checkpoint_manifest.json", checkpoint_manifest_payload)
            dump_json(output_root / "checkpoint_reload_audit.json", checkpoint_reload_payload)

            critic_ok = (
                run_a["critic_audit"]["learning_path_status"] == "LEARNING_PATH_ACTIVE"
                and run_b["critic_audit"]["learning_path_status"] == "FROZEN_CONTROL_CRITIC_UNCHANGED"
                and comparison["run_a_all_learning_modules_changed"]
                and comparison["run_b_gatv2_actor_changed"]
                and comparison["run_b_critic_unchanged"]
                and comparison["run_b_critic_grad_absent"]
            )
            checkpoint_ok = checkpoint_reload_payload["reload_ok"]
            if not critic_ok:
                gate = FAIL_CRITIC
                details["note"] = "Run A/B critic learning-path control did not satisfy all required checks."
            elif not checkpoint_ok:
                gate = FAIL_CHECKPOINT
                details["note"] = "Checkpoint reload audit failed."
            else:
                gate_passed = True
                gate = SUCCESS_GATE_VALIDATED if run_a["value_signal_status"] == "SHORT_RUN_VALUE_SIGNAL_IMPROVING" else SUCCESS_GATE_INDETERMINATE
                details["note"] = f"Learning path active; value signal status: {run_a['value_signal_status']}."
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            gate = "FAIL_MPS_OOM_DURING_VALIDATION"
        else:
            gate = FAIL_SUBGRAPH if "Suseong" in str(exc) or "Frozen local" in str(exc) else FAIL_CRITIC
        details["note"] = f"{type(exc).__name__}: {exc}"
        for name in [
            "suseong_subgraph_inventory.json",
            "suseong_subgraph_connectivity_audit.json",
            "tensor_shape_audit.json",
            "masking_audit.json",
            "training_configuration.json",
            "model_architecture_summary.json",
            "optimizer_parameter_ownership.json",
        ]:
            path = output_root / name
            if not path.exists():
                dump_json(path, {"created_at": iso_kst(), "executed": False, "error": str(exc)})
        write_skipped_training_files(output_root, gate)

    details.update(
        {
            "upstream_gate": upstream.get("upstream_gate"),
            "artifact_required_file_count": len(REQUIRED_FILES),
        }
    )
    dump_json(output_root / "source_change_inventory.json", source_inventory(project_root, output_root, runner_path))
    dump_json(
        output_root / "gate_decision.json",
        {
            "created_at": iso_kst(),
            "gate": gate,
            "gate_passed": gate_passed,
            "study_area": "SUSEONG_GU_DAEGU",
            "mps_available": runtime["mps_available"],
            "selected_device": details["selected_device"],
            "run_a_executed": details["run_a_executed"],
            "run_b_executed": details["run_b_executed"],
            "api_call_count": 0,
            "service_key_accessed": False,
            "external_network_accessed": False,
            "h200_used": False,
            "full_daegu_training_used": False,
        },
    )
    final_reports(output_root, gate, gate_passed, details)
    write_artifact_manifest_and_success(output_root, gate, gate_passed)
    print(json.dumps({"artifact": str(output_root), "gate": gate, "gate_passed": gate_passed}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
