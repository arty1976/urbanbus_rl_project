"""PV8 H4M-I/H4M-J target-context observation repair utilities.

This module implements the frozen H4M-I repair contract:

    9D existing node features
    + 3D frozen R3 action-target/obligation one-hot context
    = 12D repaired node features

The added fields are derived only from the pre-action target context already
present in the frozen runtime (`data.y` -> `action_targets_from_y`) and already
consumed before action sampling by K-mask and Reward V2. It does not read
long-horizon labels, counterfactual results, rewards, post-action state, or
future KPI artifacts.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


CONTRACT_NAME = "PV8_H4M_I_OBSERVATION_TARGET_CONTEXT_REPAIR_CONTRACT"
CONTRACT_SHA256 = "6ecd20cfcd220d50a6f1ebbcd6e33594a9ca14a86a2a60236cea435a24058e1a"
SCHEMA_VERSION = "PV8_TARGET_CONTEXT_OBSERVATION_SCHEMA_V1"

BASE_X_COLS = [
    "boardings_recent_log",
    "alightings_recent_log",
    "waiting_passenger_cnt_log",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_peak",
    "delta_t_hr_effective",
]

TARGET_CONTEXT_FIELDS = [
    "r3_action_target_is_hold",
    "r3_action_target_is_serve",
    "r3_action_target_is_skip",
]

REPAIRED_X_COLS = BASE_X_COLS + TARGET_CONTEXT_FIELDS


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def repair_contract_metadata() -> Dict[str, Any]:
    return {
        "contract_name": CONTRACT_NAME,
        "contract_sha256": CONTRACT_SHA256,
        "schema_version": SCHEMA_VERSION,
        "node_feature_dim_before": 9,
        "node_feature_dim_after": 12,
        "base_feature_names": list(BASE_X_COLS),
        "target_context_fields": list(TARGET_CONTEXT_FIELDS),
        "repaired_feature_names": list(REPAIRED_X_COLS),
        "causal_source": "pre-action frozen R3 action-target/obligation tensor via dl1.action_targets_from_y(data.y, action_dim)",
        "anti_leakage": {
            "uses_long_horizon_shadow_result": False,
            "uses_counterfactual_classification": False,
            "uses_future_reward": False,
            "uses_future_passenger_outcome": False,
            "uses_post_action_state": False,
            "uses_future_kpi": False,
        },
    }


def target_context_one_hot_from_data(data: Any, dl1: Any, action_dim: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target one-hot context for reachable pre-action nodes.

    Active/reachable nodes (`data.node_mask == True`) receive exactly one
    target context bit. Inactive graph padding/non-reachable nodes receive zero
    in all three new channels so the repair does not invent target semantics
    for nodes outside the action-selection population.
    """

    if action_dim != 3:
        raise ValueError(f"{CONTRACT_NAME} requires action_dim=3, got {action_dim}")
    if not hasattr(data, "x") or not hasattr(data, "y"):
        raise ValueError("data must contain x and y tensors")
    target = dl1.action_targets_from_y(data.y, action_dim).long()
    one_hot = F.one_hot(target, num_classes=action_dim).to(dtype=data.x.dtype, device=data.x.device)
    if hasattr(data, "node_mask"):
        active_mask = data.node_mask.bool().to(device=data.x.device)
    else:
        active_mask = torch.ones((int(data.x.size(0)),), dtype=torch.bool, device=data.x.device)
    one_hot = torch.where(active_mask.unsqueeze(-1), one_hot, torch.zeros_like(one_hot))
    return one_hot, target, active_mask


def one_hot_semantics_audit(one_hot: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, Any]:
    active = one_hot[active_mask.bool()]
    inactive = one_hot[~active_mask.bool()]
    active_row_sums = active.sum(dim=1) if active.numel() else torch.empty((0,), device=one_hot.device)
    inactive_abs_sum = inactive.abs().sum(dim=1) if inactive.numel() else torch.empty((0,), device=one_hot.device)
    return {
        "active_node_count": int(active.size(0)),
        "inactive_node_count": int(inactive.size(0)),
        "active_exact_one_hot": bool(
            active.numel() == 0
            or (
                torch.all((active == 0) | (active == 1)).detach().cpu().item()
                and torch.all(active_row_sums == 1).detach().cpu().item()
            )
        ),
        "inactive_zero_context": bool(inactive.numel() == 0 or torch.all(inactive_abs_sum == 0).detach().cpu().item()),
        "hold_count": int(active[:, 0].sum().detach().cpu().item()) if active.numel() else 0,
        "serve_count": int(active[:, 1].sum().detach().cpu().item()) if active.numel() else 0,
        "skip_count": int(active[:, 2].sum().detach().cpu().item()) if active.numel() else 0,
    }


def append_target_context_features(data: Any, dl1: Any, action_dim: int = 3) -> Tuple[Any, Dict[str, Any]]:
    """Clone `data` and append the 3D target-context channels to `data.x`."""

    old_dim = int(data.x.size(1))
    one_hot, _target, active_mask = target_context_one_hot_from_data(data, dl1, action_dim=action_dim)
    base_x_sha256 = tensor_sha256(data.x)
    if old_dim == 12:
        base_prefix = data.x[:, :9]
        observed_context = data.x[:, 9:12]
        context_match = bool(torch.equal(observed_context.detach().cpu(), one_hot.detach().cpu()))
        audit = {
            "schema_version": SCHEMA_VERSION,
            "already_repaired": True,
            "old_dim": old_dim,
            "new_dim": old_dim,
            "base_x_sha256": tensor_sha256(base_prefix),
            "repaired_x_sha256": tensor_sha256(data.x),
            "old_9_feature_values_unchanged": True,
            "target_context_matches_contract": context_match,
            **one_hot_semantics_audit(one_hot, active_mask),
        }
        return data, audit
    if old_dim != 9:
        raise ValueError(f"{CONTRACT_NAME} expects old_dim=9 or repaired_dim=12, got {old_dim}")
    repaired = data.clone()
    repaired.x = torch.cat([data.x, one_hot], dim=1)
    repaired.observation_schema_version = SCHEMA_VERSION
    repaired.feature_names = list(REPAIRED_X_COLS)
    repaired.target_context_fields = list(TARGET_CONTEXT_FIELDS)
    repaired.target_context_contract_sha256 = CONTRACT_SHA256
    repaired.target_context_causal_source = "pre-action frozen R3 action-target/obligation tensor"
    repaired_prefix_sha256 = tensor_sha256(repaired.x[:, :9])
    audit = {
        "schema_version": SCHEMA_VERSION,
        "already_repaired": False,
        "old_dim": old_dim,
        "new_dim": int(repaired.x.size(1)),
        "base_x_sha256": base_x_sha256,
        "repaired_x_prefix9_sha256": repaired_prefix_sha256,
        "repaired_x_sha256": tensor_sha256(repaired.x),
        "old_9_feature_values_unchanged": base_x_sha256 == repaired_prefix_sha256,
        "dtype": str(repaired.x.dtype),
        "shape": list(repaired.x.shape),
        **one_hot_semantics_audit(one_hot, active_mask),
    }
    return repaired, audit


def repair_data_sequence(data_seq: Sequence[Any], dl1: Any, action_dim: int = 3) -> Tuple[List[Any], Dict[str, Any]]:
    repaired_rows: List[Any] = []
    audit_rows: List[Dict[str, Any]] = []
    for index, data in enumerate(data_seq):
        repaired, audit = append_target_context_features(data, dl1, action_dim=action_dim)
        audit_rows.append({"sequence_index": int(index), **audit})
        repaired_rows.append(repaired)
    summary = {
        "contract": repair_contract_metadata(),
        "sequence_count": len(repaired_rows),
        "all_old_dim_9_or_already_12": all(row["old_dim"] in (9, 12) for row in audit_rows),
        "all_new_dim_12": all(row["new_dim"] == 12 for row in audit_rows),
        "all_old_9_feature_values_unchanged": all(row["old_9_feature_values_unchanged"] for row in audit_rows),
        "all_active_nodes_exact_one_hot": all(row["active_exact_one_hot"] for row in audit_rows),
        "all_inactive_nodes_zero_context": all(row["inactive_zero_context"] for row in audit_rows),
        "total_active_hold_context": int(sum(row["hold_count"] for row in audit_rows)),
        "total_active_serve_context": int(sum(row["serve_count"] for row in audit_rows)),
        "total_active_skip_context": int(sum(row["skip_count"] for row in audit_rows)),
        "audit_rows": audit_rows,
    }
    return repaired_rows, summary
