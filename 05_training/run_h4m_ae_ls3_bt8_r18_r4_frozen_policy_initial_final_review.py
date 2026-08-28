#!/usr/bin/env python3
"""R18-R4: same-input frozen-policy initial/final review for R18-R3.

This runner compares the R18-R3 initial and final actor checkpoints on the
same six losslessly preserved frozen review snapshots.  It is intentionally
limited to checkpoint reads, persisted snapshot reads, frozen MPS forwards, and
pure descriptive comparison.  It has no training, rollout, simulator, reward,
optimizer, backward, checkpoint-write, promotion, or KPI/performance path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R4"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R4_SAME_INPUT_FROZEN_POLICY_INITIAL_FINAL_REVIEW_COMPLETE"
BINDING_BLOCK = "BLOCKED_R18_R4_EVIDENCE_BINDING_FAILURE"
STRUCTURAL_BLOCK = "BLOCKED_R18_R4_FROZEN_REVIEW_STRUCTURAL_INTEGRITY_FAILURE"
MPS_BLOCK = "BLOCKED_R18_R4_MPS_FROZEN_REVIEW_ENVIRONMENT_UNAVAILABLE"

R18_SOURCE = "7bd0e2e223779455e6112584abd0b5cb6441c228"
R18_R2_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R2_DIGEST_DOMAIN_REPAIR_AND_FROZEN_INFERENCE_EQUIVALENCE_COMPLETE"
R18_R3_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_E1_MINIMAL_BOUNDED_TRAINING_AND_CAUSAL_LEARNING_PATH_EVIDENCE_COMPLETE"
R18_R3_CLASSIFICATION = "A_R18_E1_CAUSAL_LEARNING_PATH_OBSERVED"
REVIEW_COLLECTION_DIGEST = "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"
T1_CONTRACT = "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R18_R2 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r2_digest_domain_repair_validation_20260828_005726+09:00"
R18_R3 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r3_e1_bounded_training_execution_20260828_005726+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review.py",
    "05_training/test_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review.py",
}
ARMS = {
    "AC_CONTROL_R1": {"initial_key": "initial:AC-R1", "label": "AC"},
    "BD_E1_R1": {"initial_key": "initial:BD-R1", "label": "BD"},
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}


class R18R4Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R18R4Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path, *, code: str = BINDING_BLOCK) -> Any:
    require(path.is_file(), code, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256")
    require(isinstance(expected, Mapping), BINDING_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in expected.items()
                  if not (root / str(name)).is_file() or sha256(root / str(name)) != str(digest)]
    return {
        "manifest_sha256": sha256(root / "manifest.json"),
        "declared_file_count": len(expected),
        "mismatches": mismatches,
        "all_match": not mismatches,
    }


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        require(isinstance(tensor, torch.Tensor), BINDING_BLOCK, f"actor_tensor={name}")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R18_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "review_source_commit": git(["rev-parse", "HEAD"]),
        "training_source_commit": R18_SOURCE,
        "lineage_descends_from_r18_source": git(["merge-base", R18_SOURCE, "HEAD"]) == R18_SOURCE,
        "changed_files_since_r18_source": changed,
        "source_only_review_commit": set(changed) == SOURCE_FILES,
        "git_status_porcelain": git(["status", "--porcelain=v1"]),
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review_{stamp}"


def counters() -> dict[str, int]:
    return {
        "training": 0,
        "mps_training": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward": 0,
        "loss_update": 0,
        "causal_rollout": 0,
        "simulator_execution": 0,
        "candidate_generation": 0,
        "candidate_regeneration": 0,
        "local_search_rerun": 0,
        "zero_loss_reevaluation": 0,
        "reward_settlement": 0,
        "parameter_mutation": 0,
        "checkpoint_write": 0,
        "checkpoint_mutation": 0,
        "review_optimizer_rows": 0,
        "future_leakage": 0,
        "test6_access": 0,
        "github_push": 0,
        "nan_or_inf": 0,
        "illegal_or_masked_selection": 0,
        "frozen_policy_rows": 0,
    }


def action_identity(selection: Any) -> str:
    if selection.selected.is_no_assign:
        return "NO_ASSIGN"
    return f"{selection.selected.agent_id}::{selection.selected.candidate_id}"


def mean(values: Sequence[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.mean(finite) if finite else None


def direction(delta: float | None) -> str:
    if delta is None:
        return "NOT_AVAILABLE"
    if delta > 0.0:
        return "INCREASED"
    if delta < 0.0:
        return "DECREASED"
    return "UNCHANGED_EXACT"


def max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    require(len(left) == len(right), BINDING_BLOCK, "score_vector_length_mismatch")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def centered(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    value_mean = statistics.mean(float(value) for value in values)
    return [float(value) - value_mean for value in values]


def rank_map(scores: Sequence[float], identities: Sequence[str]) -> dict[str, int]:
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), str(identities[index])))
    return {str(identities[index]): rank for rank, index in enumerate(order, start=1)}


def actor_record(*, actor: torch.nn.Module, arm_id: str, checkpoint_state: str, checkpoint: Mapping[str, Any],
                 payload: Mapping[str, Any], R1: Any, H: Any, TIE: Any, MC: Any, device: torch.device) -> dict[str, Any]:
    result = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE)
    metadata, tensors = payload["metadata"], payload["tensors"]
    support = int(result["support_size"])
    candidate_ids = list(metadata["candidate_ids"])
    candidate_identities = [f"{row['agent_id']}::{row['candidate_id']}" for row in candidate_ids]
    pair_logits = [float(value) for value in result["pair_logits"][0].detach().cpu().tolist()]
    probabilities = [float(value) for value in result["probabilities"][0].detach().cpu().tolist()]
    pair_probabilities = probabilities[:support]
    no_assign_logit = float(result["no_assign_logit"][0, 0].detach().cpu())
    no_assign_probability = float(probabilities[-1])
    safe_mask = tensors["safe_mask"][0, :support].detach().cpu()
    selected = action_identity(result["selection"])
    legal = selected == "NO_ASSIGN" or selected in set(candidate_identities)
    require(payload["snapshot_digest"] == metadata.get("snapshot_digest", payload["snapshot_digest"]), BINDING_BLOCK, "snapshot_digest_echo")
    require(len(pair_logits) == support == len(candidate_ids) and len(probabilities) == support + 1,
            BINDING_BLOCK, f"actor_output_shape={metadata['decision_id']}")
    require(bool(safe_mask.all().item()), STRUCTURAL_BLOCK, f"unsafe_frozen_support={metadata['decision_id']}")
    require(legal, STRUCTURAL_BLOCK, f"illegal_selection={metadata['decision_id']}")
    finite = bool(result["finite"] and all(math.isfinite(value) for value in [*pair_logits, no_assign_logit, *probabilities]))
    require(finite, STRUCTURAL_BLOCK, f"nonfinite={metadata['decision_id']}")
    if support:
        top_order = sorted(range(support), key=lambda index: (-pair_logits[index], candidate_identities[index]))
        best_index = top_order[0]
        second_index = top_order[1] if support >= 2 else None
        best_pair_identity = candidate_identities[best_index]
        best_pair_logit = pair_logits[best_index]
        best_pair_probability = pair_probabilities[best_index]
    else:
        top_order, best_index, second_index = [], None, None, None
        best_pair_identity, best_pair_logit, best_pair_probability = None, None, None
    candidate_margin = None if second_index is None else pair_logits[best_index] - pair_logits[second_index]
    candidate_probability_margin = None if second_index is None else pair_probabilities[best_index] - pair_probabilities[second_index]
    candidate_range = max(pair_logits) - min(pair_logits) if support >= 2 else 0.0
    candidate_probability_range = max(pair_probabilities) - min(pair_probabilities) if support >= 2 else 0.0
    candidate_std = statistics.pstdev(pair_logits) if support >= 2 else 0.0
    pair_ranks = rank_map(pair_logits, candidate_identities)
    top2_distinctness = R1.classify_alternatives(
        payload, [index for index in (best_index, second_index) if index is not None],
        candidate_names=MC.LOCAL_SEARCH_FEATURE_NAMES, agent_names=MC.AgentContext.FEATURE_NAMES,
    )
    return {
        "arm_id": arm_id,
        "checkpoint_state": checkpoint_state,
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "actor_state_sha256": str(checkpoint["actor_digest"]),
        "decision_id": str(metadata["decision_id"]),
        "window_id": str(metadata["window_id"]),
        "time_band": str(metadata["time_band"]),
        "environment_seed": int(metadata["seed"]),
        "snapshot_digest": str(payload["snapshot_digest"]),
        "snapshot_manifest_sha256": str(checkpoint.get("snapshot_manifest_sha256", "")),
        "candidate_support_digest": str(metadata["candidate_support_digest"]),
        "candidate_ids": candidate_ids,
        "candidate_identities": candidate_identities,
        "candidate_ids_sha256": canonical_sha256(candidate_ids),
        "safe_mask_sha256": bytes_sha256(safe_mask.contiguous().numpy().tobytes()),
        "support_size": support,
        "multi_candidate": support >= 2,
        "selected_identity": selected,
        "selected_action_family": "NO_ASSIGN" if selected == "NO_ASSIGN" else "CANDIDATE",
        "legal_selection": legal,
        "exact_tie": bool(result["selection"].exact_tie),
        "tie_count": len(result["selection"].tie_set),
        "selector": TIE.TIE_BREAK_CONTRACT_ID,
        "t1_tolerance": 0.0,
        "pair_logits": pair_logits,
        "no_assign_logit": no_assign_logit,
        "probabilities": probabilities,
        "pair_probabilities": pair_probabilities,
        "no_assign_probability": no_assign_probability,
        "best_pair_identity": best_pair_identity,
        "best_pair_logit": best_pair_logit,
        "best_pair_probability": best_pair_probability,
        "best_pair_minus_no_assign": None if best_pair_logit is None else float(best_pair_logit) - no_assign_logit,
        "no_assign_minus_best_pair": None if best_pair_logit is None else no_assign_logit - float(best_pair_logit),
        "candidate_only_top1_top2_logit_margin": candidate_margin,
        "candidate_only_top1_top2_probability_margin": candidate_probability_margin,
        "candidate_only_logit_range": candidate_range,
        "candidate_only_probability_range": candidate_probability_range,
        "candidate_only_logit_std": candidate_std,
        "candidate_rank_by_identity": pair_ranks,
        "top2_candidate_distinctness": top2_distinctness,
        "entropy": float(result["entropy"]),
        "top_score_margin_including_no_assign": result["top_score_margin"],
        "top_probability_margin_including_no_assign": result["top_probability_margin"],
        "finite": finite,
    }


def rows_by_arm_state(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    result: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        result.setdefault((str(row["arm_id"]), str(row["checkpoint_state"])), []).append(row)
    for key, value in result.items():
        value.sort(key=lambda row: (int(row["environment_seed"]), str(row["window_id"]), str(row["decision_id"])))
        require(len(value) == 6 and len({str(row["snapshot_digest"]) for row in value}) == 6, BINDING_BLOCK, f"row_count={key}")
    return result


def compare_same_snapshot(initial: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, Any]:
    require(initial["arm_id"] == final["arm_id"] and initial["snapshot_digest"] == final["snapshot_digest"],
            BINDING_BLOCK, "initial_final_snapshot_binding")
    require(initial["candidate_support_digest"] == final["candidate_support_digest"]
            and initial["candidate_identities"] == final["candidate_identities"]
            and initial["safe_mask_sha256"] == final["safe_mask_sha256"], BINDING_BLOCK, "support_identity_binding")
    pair_logits_initial = [float(value) for value in initial["pair_logits"]]
    pair_logits_final = [float(value) for value in final["pair_logits"]]
    probabilities_initial = [float(value) for value in initial["probabilities"]]
    probabilities_final = [float(value) for value in final["probabilities"]]
    rank_initial = dict(initial["candidate_rank_by_identity"])
    rank_final = dict(final["candidate_rank_by_identity"])
    candidate_rank_changes = {identity: int(rank_final[identity]) - int(rank_initial[identity]) for identity in rank_initial}
    return {
        "arm_id": initial["arm_id"],
        "decision_id": initial["decision_id"],
        "window_id": initial["window_id"],
        "time_band": initial["time_band"],
        "environment_seed": initial["environment_seed"],
        "snapshot_digest": initial["snapshot_digest"],
        "candidate_support_digest": initial["candidate_support_digest"],
        "support_size": initial["support_size"],
        "selection_changed": initial["selected_identity"] != final["selected_identity"],
        "action_family_changed": initial["selected_action_family"] != final["selected_action_family"],
        "initial_selected_identity": initial["selected_identity"],
        "final_selected_identity": final["selected_identity"],
        "initial_selected_action_family": initial["selected_action_family"],
        "final_selected_action_family": final["selected_action_family"],
        "best_pair_changed": initial["best_pair_identity"] != final["best_pair_identity"],
        "initial_best_pair_identity": initial["best_pair_identity"],
        "final_best_pair_identity": final["best_pair_identity"],
        "candidate_rank_changed_count": sum(delta != 0 for delta in candidate_rank_changes.values()),
        "candidate_rank_changes": candidate_rank_changes,
        "max_abs_pair_logit_delta": max_abs_delta(pair_logits_initial, pair_logits_final),
        "mean_abs_pair_logit_delta": mean([abs(a - b) for a, b in zip(pair_logits_initial, pair_logits_final)]),
        "max_abs_centered_pair_logit_delta": max_abs_delta(centered(pair_logits_initial), centered(pair_logits_final)),
        "no_assign_logit_delta": float(final["no_assign_logit"]) - float(initial["no_assign_logit"]),
        "max_abs_probability_delta": max_abs_delta(probabilities_initial, probabilities_final),
        "no_assign_probability_delta": float(final["no_assign_probability"]) - float(initial["no_assign_probability"]),
        "best_pair_probability_delta": None if initial["best_pair_probability"] is None or final["best_pair_probability"] is None
        else float(final["best_pair_probability"]) - float(initial["best_pair_probability"]),
        "best_pair_minus_no_assign_delta": None if initial["best_pair_minus_no_assign"] is None or final["best_pair_minus_no_assign"] is None
        else float(final["best_pair_minus_no_assign"]) - float(initial["best_pair_minus_no_assign"]),
        "candidate_margin_delta": None if initial["candidate_only_top1_top2_logit_margin"] is None
        or final["candidate_only_top1_top2_logit_margin"] is None
        else float(final["candidate_only_top1_top2_logit_margin"]) - float(initial["candidate_only_top1_top2_logit_margin"]),
        "candidate_probability_margin_delta": None if initial["candidate_only_top1_top2_probability_margin"] is None
        or final["candidate_only_top1_top2_probability_margin"] is None
        else float(final["candidate_only_top1_top2_probability_margin"]) - float(initial["candidate_only_top1_top2_probability_margin"]),
        "candidate_range_delta": float(final["candidate_only_logit_range"]) - float(initial["candidate_only_logit_range"]),
        "candidate_probability_range_delta": float(final["candidate_only_probability_range"]) - float(initial["candidate_only_probability_range"]),
        "candidate_std_delta": float(final["candidate_only_logit_std"]) - float(initial["candidate_only_logit_std"]),
        "entropy_delta": float(final["entropy"]) - float(initial["entropy"]),
        "initial_candidate_margin": initial["candidate_only_top1_top2_logit_margin"],
        "final_candidate_margin": final["candidate_only_top1_top2_logit_margin"],
        "initial_candidate_logit_range": initial["candidate_only_logit_range"],
        "final_candidate_logit_range": final["candidate_only_logit_range"],
        "initial_candidate_logit_std": initial["candidate_only_logit_std"],
        "final_candidate_logit_std": final["candidate_only_logit_std"],
    }


def policy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    return {
        "rows": len(values),
        "candidate_selection_count": sum(row["selected_action_family"] == "CANDIDATE" for row in values),
        "no_assign_selection_count": sum(row["selected_action_family"] == "NO_ASSIGN" for row in values),
        "multi_candidate_count": sum(bool(row["multi_candidate"]) for row in values),
        "exact_tie_count": sum(bool(row["exact_tie"]) for row in values),
        "mean_entropy": mean([row["entropy"] for row in values]),
        "mean_no_assign_probability": mean([row["no_assign_probability"] for row in values]),
        "mean_best_pair_probability": mean([row["best_pair_probability"] for row in values]),
        "mean_best_pair_minus_no_assign": mean([row["best_pair_minus_no_assign"] for row in values]),
        "mean_candidate_margin": mean([row["candidate_only_top1_top2_logit_margin"] for row in values]),
        "mean_candidate_probability_margin": mean([row["candidate_only_top1_top2_probability_margin"] for row in values]),
        "mean_candidate_logit_range": mean([row["candidate_only_logit_range"] for row in values]),
        "mean_candidate_probability_range": mean([row["candidate_only_probability_range"] for row in values]),
        "mean_candidate_logit_std": mean([row["candidate_only_logit_std"] for row in values]),
        "min_candidate_margin": min((float(row["candidate_only_top1_top2_logit_margin"]) for row in values
                                     if row["candidate_only_top1_top2_logit_margin"] is not None), default=None),
        "max_candidate_margin": max((float(row["candidate_only_top1_top2_logit_margin"]) for row in values
                                     if row["candidate_only_top1_top2_logit_margin"] is not None), default=None),
    }


def delta_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    return {
        "paired_rows": len(values),
        "selection_changed_count": sum(bool(row["selection_changed"]) for row in values),
        "action_family_changed_count": sum(bool(row["action_family_changed"]) for row in values),
        "best_pair_changed_count": sum(bool(row["best_pair_changed"]) for row in values),
        "candidate_rank_changed_snapshot_count": sum(int(row["candidate_rank_changed_count"]) > 0 for row in values),
        "max_abs_pair_logit_delta": max((float(row["max_abs_pair_logit_delta"]) for row in values), default=0.0),
        "max_abs_centered_pair_logit_delta": max((float(row["max_abs_centered_pair_logit_delta"]) for row in values), default=0.0),
        "max_abs_probability_delta": max((float(row["max_abs_probability_delta"]) for row in values), default=0.0),
        "mean_entropy_delta": mean([row["entropy_delta"] for row in values]),
        "mean_no_assign_probability_delta": mean([row["no_assign_probability_delta"] for row in values]),
        "mean_best_pair_minus_no_assign_delta": mean([row["best_pair_minus_no_assign_delta"] for row in values]),
        "mean_candidate_margin_delta": mean([row["candidate_margin_delta"] for row in values]),
        "mean_candidate_probability_margin_delta": mean([row["candidate_probability_margin_delta"] for row in values]),
        "mean_candidate_range_delta": mean([row["candidate_range_delta"] for row in values]),
        "mean_candidate_probability_range_delta": mean([row["candidate_probability_range_delta"] for row in values]),
        "mean_candidate_std_delta": mean([row["candidate_std_delta"] for row in values]),
        "candidate_margin_direction_counts": {
            "INCREASED": sum(direction(row["candidate_margin_delta"]) == "INCREASED" for row in values),
            "DECREASED": sum(direction(row["candidate_margin_delta"]) == "DECREASED" for row in values),
            "UNCHANGED_EXACT": sum(direction(row["candidate_margin_delta"]) == "UNCHANGED_EXACT" for row in values),
            "NOT_AVAILABLE": sum(direction(row["candidate_margin_delta"]) == "NOT_AVAILABLE" for row in values),
        },
        "candidate_range_direction_counts": {
            "INCREASED": sum(direction(row["candidate_range_delta"]) == "INCREASED" for row in values),
            "DECREASED": sum(direction(row["candidate_range_delta"]) == "DECREASED" for row in values),
            "UNCHANGED_EXACT": sum(direction(row["candidate_range_delta"]) == "UNCHANGED_EXACT" for row in values),
        },
        "candidate_std_direction_counts": {
            "INCREASED": sum(direction(row["candidate_std_delta"]) == "INCREASED" for row in values),
            "DECREASED": sum(direction(row["candidate_std_delta"]) == "DECREASED" for row in values),
            "UNCHANGED_EXACT": sum(direction(row["candidate_std_delta"]) == "UNCHANGED_EXACT" for row in values),
        },
    }


def classification_from(delta_by_arm: Mapping[str, Mapping[str, Any]]) -> str:
    values = list(delta_by_arm.values())
    if any(int(value["selection_changed_count"]) > 0 for value in values):
        return "A_R18_R3_POLICY_SELECTION_CHANGED_ON_FIXED_REVIEW_SNAPSHOTS"
    if any(float(value["max_abs_pair_logit_delta"]) > 0.0 or float(value["max_abs_probability_delta"]) > 0.0 for value in values):
        return "B_R18_R3_POLICY_DISTRIBUTION_SHIFT_WITH_STABLE_T1_SELECTIONS"
    return "C_R18_R3_POLICY_OUTPUTS_UNCHANGED_ON_FIXED_REVIEW_SNAPSHOTS"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "arm_id", "environment_seed", "time_band", "window_id", "decision_id", "snapshot_digest",
        "support_size", "selection_changed", "action_family_changed", "best_pair_changed",
        "initial_selected_identity", "final_selected_identity", "initial_best_pair_identity", "final_best_pair_identity",
        "max_abs_pair_logit_delta", "max_abs_centered_pair_logit_delta", "max_abs_probability_delta",
        "no_assign_probability_delta", "best_pair_minus_no_assign_delta", "candidate_margin_delta",
        "candidate_probability_margin_delta", "candidate_range_delta", "candidate_probability_range_delta",
        "candidate_std_delta", "entropy_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_block(*, root: Path, source: Mapping[str, Any], code: str, detail: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    zero = counters()
    outputs = {
        "r18r4_evidence_preflight.json": {"source": source, "hard_failures": [detail]},
        "r18r4_frozen_policy_matrix.json": {"not_run": True},
        "r18r4_initial_final_delta.json": {"not_run": True},
        "r18r4_candidate_discrimination.json": {"not_run": True},
        "r18r4_checkpoint_immutability.json": {"not_run": True},
        "test_results.json": {"execution_counters": zero, "hard_failures": [detail], "warnings": [], "github_push_performed": False},
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                               "review_source_commit": source.get("review_source_commit"),
                               "training_source_commit": R18_SOURCE, "hard_failures": [detail],
                               "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(
        f"# R18-R4 blocked\n\n- gate: `{code}`\n- detail: `{detail}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt8_f1_bounded_training as F1MOD
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), BINDING_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    preflight: dict[str, Any] = {
        "source": source,
        "stage": STAGE,
        "r18_r2_root": str(R18_R2),
        "r18_r3_root": str(R18_R3),
        "training_allowed": False,
        "pure_frozen_inference_only": True,
        "performance_or_kpi_interpretation_allowed": False,
        "review_collection_digest": REVIEW_COLLECTION_DIGEST,
        "t1_selector_expected": T1_CONTRACT,
    }
    try:
        require(source["lineage_descends_from_r18_source"] and source["source_only_review_commit"]
                and source["git_status_porcelain"] == "", BINDING_BLOCK, "review_source_scope")
        r2_gate, r3_gate = load_json(R18_R2 / "gate_decision.json"), load_json(R18_R3 / "gate_decision.json")
        require(r2_gate.get("gate") == R18_R2_GATE and r2_gate.get("source_commit") == R18_SOURCE, BINDING_BLOCK, "r18_r2_gate")
        require(r3_gate.get("gate") == R18_R3_GATE and r3_gate.get("classification") == R18_R3_CLASSIFICATION
                and r3_gate.get("source_commit") == R18_SOURCE, BINDING_BLOCK, "r18_r3_gate")
        r2_manifest, r3_manifest = manifest_audit(R18_R2), manifest_audit(R18_R3)
        require(r2_manifest["all_match"] and r3_manifest["all_match"], BINDING_BLOCK, "upstream_manifest_hashes")

        auth = load_json(R18_R2 / "r18r3_bounded_training_authorization_manifest.json")
        supplied_auth = str(auth.get("authorization_sha256"))
        auth_body = dict(auth)
        auth_body.pop("authorization_sha256", None)
        require(canonical_sha256(auth_body) == supplied_auth and auth.get("source_commit") == R18_SOURCE, BINDING_BLOCK, "r18_r3_authorization")
        require(str(dict(auth.get("checkpoint_contract", {})).get("R18_output_root")) == str(R18_R3), BINDING_BLOCK, "r18_r3_output_root")

        review_binding = dict(dict(auth.get("upstream", {})).get("review_snapshot_binding", {}))
        require(review_binding.get("verified") is True and review_binding.get("collection_digest") == REVIEW_COLLECTION_DIGEST,
                BINDING_BLOCK, "review_binding")
        collection_path = Path(str(review_binding.get("collection_path", "")))
        collection = load_json(collection_path)
        collection_body = dict(collection)
        collection_digest = collection_body.pop("collection_digest", None)
        require(collection_digest == REVIEW_COLLECTION_DIGEST and FPS.canonical_sha256(collection_body) == REVIEW_COLLECTION_DIGEST,
                BINDING_BLOCK, "review_collection_digest")
        entries = [dict(row) for row in review_binding.get("entries", [])]
        require(len(entries) == 6 and len({row.get("snapshot_digest") for row in entries}) == 6, BINDING_BLOCK, "review_entry_count")
        require({row["snapshot_digest"] for row in entries} == {row["snapshot_digest"] for row in collection.get("entries", [])},
                BINDING_BLOCK, "review_entry_set")
        snapshots: list[dict[str, Any]] = []
        snapshot_rows: list[dict[str, Any]] = []
        config: Mapping[str, Any] | None = None
        for entry in entries:
            snapshot_root = Path(str(entry["snapshot_root"]))
            snapshot_manifest = load_json(snapshot_root / "snapshot_manifest.json")
            require(snapshot_root.is_dir() and snapshot_manifest.get("manifest_sha256") == entry["snapshot_manifest_sha256"]
                    and snapshot_manifest.get("snapshot_digest") == entry["snapshot_digest"],
                    BINDING_BLOCK, f"snapshot_manifest={entry['decision_id']}")
            payload = FPS.load_snapshot(snapshot_root)
            metadata, tensors = payload["metadata"], payload["tensors"]
            require(payload["snapshot_digest"] == entry["snapshot_digest"], BINDING_BLOCK, f"snapshot_digest={entry['decision_id']}")
            require(metadata["actor_config"] == (config if config is not None else metadata["actor_config"]), BINDING_BLOCK, "actor_config")
            config = metadata["actor_config"]
            require(str(metadata["actor_config_sha256"]) == FPS.actor_config_sha256(config), BINDING_BLOCK, "actor_config_sha")
            require(metadata["candidate_ids"] == metadata["candidate_order"], BINDING_BLOCK, "candidate_order")
            support = int(metadata["selectable_pair_count"])
            require(support > 0 and int(metadata["no_assign_index"]) == support
                    and bool(tensors["safe_mask"][0, :support].all().item()), STRUCTURAL_BLOCK, "snapshot_support")
            require(metadata.get("captured_device") == "mps:0", BINDING_BLOCK, "snapshot_device")
            snapshots.append(payload)
            snapshot_rows.append({
                "decision_id": str(metadata["decision_id"]),
                "window_id": str(metadata["window_id"]),
                "time_band": str(metadata["time_band"]),
                "environment_seed": int(metadata["seed"]),
                "snapshot_digest": str(payload["snapshot_digest"]),
                "snapshot_manifest_sha256": str(entry["snapshot_manifest_sha256"]),
                "candidate_support_digest": str(metadata["candidate_support_digest"]),
                "support_size": support,
            })
        require(config is not None and config.get("actor_head_id") == H.CANDIDATE_SENSITIVE_HEAD_ID
                and config.get("actor_head_version") == H.CANDIDATE_SENSITIVE_HEAD_VERSION, BINDING_BLOCK, "actor_config_head")
        require(TIE.TIE_BREAK_CONTRACT_ID == T1_CONTRACT, BINDING_BLOCK, "t1_selector")

        r3_checkpoints = load_json(R18_R3 / "r18_checkpoint_manifest.json")
        checkpoint_contract = dict(auth["checkpoint_contract"])
        initial_inputs = dict(checkpoint_contract["initial_inputs"])
        checkpoint_bindings: dict[str, dict[str, Any]] = {}
        checkpoint_sha_before: dict[str, str] = {}
        for arm_id, arm in ARMS.items():
            initial_entry = dict(initial_inputs[arm["initial_key"]])
            final_entry = dict(dict(r3_checkpoints["final"])[arm_id])
            for state, entry in (("initial", initial_entry), ("final", final_entry)):
                checkpoint_path = Path(str(entry["path"]))
                require(checkpoint_path.is_file() and sha256(checkpoint_path) == entry["sha256"], BINDING_BLOCK,
                        f"checkpoint_sha={arm_id}:{state}")
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                require(isinstance(payload, Mapping) and isinstance(payload.get("actor"), Mapping)
                        and isinstance(payload.get("critic"), Mapping) and isinstance(payload.get("meta"), Mapping)
                        and not any("optimizer" in str(key).lower() for key in payload), BINDING_BLOCK,
                        f"checkpoint_schema={arm_id}:{state}")
                actor_digest = state_dict_digest(payload["actor"])
                expected_digest = final_entry["initial_actor_digest"] if state == "initial" else final_entry["final_actor_digest"]
                require(actor_digest == expected_digest, BINDING_BLOCK, f"actor_digest={arm_id}:{state}")
                meta = dict(payload["meta"])
                require(meta.get("test_only") is True and meta.get("bounded") is True and meta.get("non_promotable") is True
                        and meta.get("winner") is False and meta.get("best_model") is False and meta.get("promotion") is False,
                        BINDING_BLOCK, f"checkpoint_flags={arm_id}:{state}")
                key = f"{arm_id}:{state}"
                checkpoint_bindings[key] = {"path": str(checkpoint_path), "sha256": str(entry["sha256"]),
                                            "actor_digest": actor_digest, "meta": meta}
                checkpoint_sha_before[key] = sha256(checkpoint_path)
        require(dict(r3_checkpoints).get("test_only") is True and dict(r3_checkpoints).get("non_promotable") is True
                and dict(r3_checkpoints).get("promotion") is False, BINDING_BLOCK, "r18_r3_checkpoint_manifest_flags")

        preflight |= {
            "r18_r2_manifest": r2_manifest,
            "r18_r3_manifest": r3_manifest,
            "authorization_sha256": supplied_auth,
            "review_snapshot_count": len(snapshots),
            "review_snapshot_unique": len({payload["snapshot_digest"] for payload in snapshots}),
            "review_snapshots": snapshot_rows,
            "actor_config": dict(config),
            "actor_config_sha256": FPS.actor_config_sha256(config),
            "checkpoint_binding": checkpoint_bindings,
            "checkpoint_sha_before": checkpoint_sha_before,
            "selector": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "tolerance": 0.0},
        }
    except R18R4Error as exc:
        write_block(root=root, source=source, code=exc.code, detail=str(exc))
        print(f"[BLOCKED] {exc.code}")
        print(f"artifact: {root.relative_to(PROJECT)}")
        return
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, code=BINDING_BLOCK, detail=f"{type(exc).__name__}:{exc}")
        print(f"[BLOCKED] {BINDING_BLOCK}")
        print(f"artifact: {root.relative_to(PROJECT)}")
        return

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root=root, source=source, code=MPS_BLOCK,
                    detail=f"mps_built={torch.backends.mps.is_built()} mps_available={torch.backends.mps.is_available()}")
        print(f"[BLOCKED] {MPS_BLOCK}")
        print(f"artifact: {root.relative_to(PROJECT)}")
        return

    device = torch.device("mps:0")
    execution = counters()
    hard: list[str] = []
    try:
        actors: dict[tuple[str, str], torch.nn.Module] = {}
        for arm_id in ARMS:
            for state in ("initial", "final"):
                checkpoint = checkpoint_bindings[f"{arm_id}:{state}"]
                actor = F1MOD.strict_load(Path(str(checkpoint["path"])), H, JL, MC, device)
                require(module_digest(actor) == checkpoint["actor_digest"], BINDING_BLOCK, f"strict_actor_load={arm_id}:{state}")
                actors[(arm_id, state)] = actor
        actor_digest_before = {f"{arm}:{state}": module_digest(actor) for (arm, state), actor in actors.items()}

        matrix: list[dict[str, Any]] = []
        for (arm_id, state), actor in actors.items():
            checkpoint = checkpoint_bindings[f"{arm_id}:{state}"]
            for payload, snapshot_row in zip(snapshots, snapshot_rows):
                row = actor_record(actor=actor, arm_id=arm_id, checkpoint_state=state, checkpoint=checkpoint,
                                   payload=payload, R1=R1, H=H, TIE=TIE, MC=MC, device=device)
                row["snapshot_manifest_sha256"] = snapshot_row["snapshot_manifest_sha256"]
                matrix.append(row)
                execution["frozen_policy_rows"] += 1
                execution["nan_or_inf"] += int(not row["finite"])
                execution["illegal_or_masked_selection"] += int(not row["legal_selection"])
        torch.mps.synchronize()
        actor_digest_after = {f"{arm}:{state}": module_digest(actor) for (arm, state), actor in actors.items()}
        execution["parameter_mutation"] = sum(actor_digest_before[name] != actor_digest_after[name] for name in actor_digest_before)

        grouped = rows_by_arm_state(matrix)
        deltas_by_arm: dict[str, list[dict[str, Any]]] = {}
        summaries_by_arm: dict[str, Any] = {}
        for arm_id in ARMS:
            initial_rows = grouped[(arm_id, "initial")]
            final_rows = grouped[(arm_id, "final")]
            initial_by_digest = {str(row["snapshot_digest"]): row for row in initial_rows}
            final_by_digest = {str(row["snapshot_digest"]): row for row in final_rows}
            require(set(initial_by_digest) == set(final_by_digest), BINDING_BLOCK, f"paired_snapshots={arm_id}")
            rows = [compare_same_snapshot(initial_by_digest[digest], final_by_digest[digest])
                    for digest in sorted(initial_by_digest)]
            deltas_by_arm[arm_id] = rows
            summaries_by_arm[arm_id] = {
                "initial": policy_summary(initial_rows),
                "final": policy_summary(final_rows),
                "delta": delta_summary(rows),
            }
        delta_summaries = {arm_id: summaries_by_arm[arm_id]["delta"] for arm_id in ARMS}

        checkpoint_sha_after = {key: sha256(Path(value["path"])) for key, value in checkpoint_bindings.items()}
        execution["checkpoint_mutation"] = sum(checkpoint_sha_before[name] != checkpoint_sha_after[name] for name in checkpoint_sha_before)
        forbidden = (
            "training", "mps_training", "optimizer_creation", "optimizer_step", "backward", "loss_update",
            "causal_rollout", "simulator_execution", "candidate_generation", "candidate_regeneration",
            "local_search_rerun", "zero_loss_reevaluation", "reward_settlement", "parameter_mutation",
            "checkpoint_write", "checkpoint_mutation", "review_optimizer_rows", "future_leakage",
            "test6_access", "github_push", "nan_or_inf", "illegal_or_masked_selection",
        )
        if any(execution[name] != 0 for name in forbidden):
            hard.append(STRUCTURAL_BLOCK)
        classification = "BLOCKED" if hard else classification_from(delta_summaries)
        gate = hard[0] if hard else PASS_GATE

        flat_deltas = [row for rows in deltas_by_arm.values() for row in rows]
        write_csv(root / "r18r4_initial_final_deltas.csv", flat_deltas)
        outputs = {
            "r18r4_evidence_preflight.json": preflight | {"mps_built": True, "mps_available": True, "device": "mps:0", "cpu_fallback": 0},
            "r18r4_frozen_policy_matrix.json": {
                "policy_rows": len(matrix),
                "arm_count": len(ARMS),
                "checkpoint_states": ["initial", "final"],
                "review_snapshot_count": len(snapshots),
                "rows": matrix,
                "per_arm_state_summary": {f"{arm_id}:{state}": policy_summary(rows) for (arm_id, state), rows in grouped.items()},
                "same_input_rule": "initial/final comparisons are permitted only inside the exact same snapshot digest and support digest.",
            },
            "r18r4_initial_final_delta.json": {
                "rows": deltas_by_arm,
                "summary": delta_summaries,
                "csv": "r18r4_initial_final_deltas.csv",
            },
            "r18r4_candidate_discrimination.json": {
                "definition": {
                    "candidate_margin": "top1 minus top2 pair logit, excluding no-assign; None for single-candidate support",
                    "candidate_range": "max minus min pair logit, excluding no-assign",
                    "candidate_std": "population standard deviation of pair logits, excluding no-assign",
                    "centered_pair_delta": "pair logit delta after subtracting each snapshot's mean pair logit",
                },
                "arms": summaries_by_arm,
                "no_performance_or_kpi_interpretation": True,
            },
            "r18r4_checkpoint_immutability.json": {
                "checkpoint_sha_before": checkpoint_sha_before,
                "checkpoint_sha_after": checkpoint_sha_after,
                "checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after,
                "actor_digest_before": actor_digest_before,
                "actor_digest_after": actor_digest_after,
                "actor_modules_unchanged_during_review": actor_digest_before == actor_digest_after,
            },
            "test_results.json": {
                "execution_counters": execution,
                "hard_failures": hard,
                "warnings": [],
                "github_push_performed": False,
                "training_performed": False,
                "performance_interpretation_performed": False,
            },
            "gate_decision.json": {
                "stage": STAGE,
                "gate": gate,
                "classification": classification,
                "review_source_commit": source["review_source_commit"],
                "training_source_commit": R18_SOURCE,
                "hard_failures": hard,
                "warnings": [],
                "global_locks": LOCKS,
                "next_step": "manual interpretation of frozen-policy deltas only; no automatic training or promotion",
            },
        }
        for name, value in outputs.items():
            dump(root / name, value)
        (root / "final_report.md").write_text(
            f"# R18-R4 final report\n\n- gate: `{gate}`\n- classification: `{classification}`\n"
            f"- review source: `{source['review_source_commit']}`\n- R18-R3 training source: `{R18_SOURCE}`\n\n"
            "This artifact is pure same-input frozen inference. It compares policy movement and candidate discrimination only; "
            "it makes no KPI, promotion, policy-quality, or causal-performance claim.\n",
            encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                       "review_source_commit": source["review_source_commit"],
                                       "training_source_commit": R18_SOURCE, "file_sha256": manifest,
                                       "github_push_performed": False})
        (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
        print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
        print(f"classification: {classification}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except R18R4Error as exc:
        write_block(root=root, source=source, code=exc.code, detail=str(exc))
        print(f"[BLOCKED] {exc.code}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, code=STRUCTURAL_BLOCK, detail=f"{type(exc).__name__}:{exc}")
        print(f"[BLOCKED] {STRUCTURAL_BLOCK}")
        print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
