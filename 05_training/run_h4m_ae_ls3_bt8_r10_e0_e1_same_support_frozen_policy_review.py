#!/usr/bin/env python3
"""BT8-R10: read-only E0/E1 frozen-policy review on the original F1 inputs.

The runner loads only three actor states per replicate (initial, E0 final, and
E1 final) plus the six losslessly preserved F1 review snapshots.  It has no
training, state-transition, candidate-construction, reward, or checkpoint
write path.  Cross-seed summaries remain descriptive when the frozen support
differs; they never relabel one seed's candidate identity as the other's.
"""

from __future__ import annotations

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


STAGE = "H4M-AE-R9.8-LS3-BT8-R10"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R10_E0_E1_SAME_SUPPORT_FROZEN_POLICY_REVIEW_COMPLETE"
SAME_SUPPORT_BLOCK = "BLOCKED_E0_E1_SAME_SUPPORT_BINDING_FAILURE"
R2_IMMUTABILITY_BLOCK = "BLOCKED_R2_E1_ACTOR_IMMUTABILITY_FAILURE"
STRUCTURAL_BLOCK = "BLOCKED_E0_E1_FROZEN_POLICY_STRUCTURAL_INTEGRITY_FAILURE"
MPS_BLOCK = "BLOCKED_MPS_FROZEN_POLICY_REVIEW_ENVIRONMENT_UNAVAILABLE"

E0_F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
E1_SOURCE = "886e0e1e5b3c49d42cffe485aa4834a40c04cb03"
R9_SOURCE = "e70b2bdc51674547e7d7e6dc31a6c31c8b6eec56"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
E1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_E1_EXACT_PRESERVED_BATCH_BOUNDED_RETRAINING_COMPLETE"
R9_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R9_E1_SEPARATE_BOUNDED_RETRAINING_AUTHORIZATION_SELECTION_COMPLETE"
E1_CONTRACT_SHA256 = "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"
T1_CONTRACT = "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
E1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining_20260826_095149+09:00"
R9 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r9_e1_retraining_authority_selection_20260825_200038+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review.py",
    "05_training/test_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
REPLICATES = {
    "F1_R1": {"environment_seed": 20260822, "actor_seed": 20260824, "critic_seed": 20260826},
    "F1_R2": {"environment_seed": 20260823, "actor_seed": 20260825, "critic_seed": 20260827},
}
POLICY_STATES = ("initial", "e0_final", "e1_final")


class R10Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R10Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: Any) -> str:
    return bytes_sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), SAME_SUPPORT_BLOCK, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256")
    require(isinstance(expected, Mapping), SAME_SUPPORT_BLOCK, f"manifest_schema={root}")
    mismatches = [name for name, digest in expected.items() if not (root / str(name)).is_file() or sha256(root / str(name)) != digest]
    return {"manifest_sha256": sha256(root / "manifest.json"), "declared_file_count": len(expected),
            "mismatches": mismatches, "all_match": not mismatches}


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, value in sorted(module.state_dict().items()):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{E1_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_e1": git(["merge-base", E1_SOURCE, "HEAD"]) == E1_SOURCE,
        "changed_files_since_e1": changed,
        "source_only_local_commit": set(changed) == SOURCE_FILES,
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review_{stamp}"


def frozen_hashes(BT6: Any, R6: Any) -> dict[str, str]:
    return R6.frozen_hashes(BT6)


def counters() -> dict[str, int]:
    return {
        "training": 0,
        "optimizer_step": 0,
        "causal_rollout": 0,
        "candidate_generation": 0,
        "candidate_regeneration": 0,
        "local_search_rerun": 0,
        "zero_loss_reevaluation": 0,
        "parameter_mutation": 0,
        "checkpoint_write": 0,
        "checkpoint_mutation": 0,
        "review_optimizer_rows": 0,
        "future_leakage": 0,
        "nan_or_inf": 0,
        "illegal_or_masked_selection": 0,
        "zero_loss_fail_support_or_selection": 0,
        "test6_access": 0,
        "github_push": 0,
        "frozen_policy_rows": 0,
        "order_probe_forwards": 0,
    }


def action_identity(selection: Any) -> str | tuple[str, str]:
    return "NO_ASSIGN" if selection.selected.is_no_assign else (str(selection.selected.agent_id), str(selection.selected.candidate_id))


def action_family(value: str | tuple[str, str]) -> str:
    return "NO_ASSIGN" if value == "NO_ASSIGN" else "ASSIGN"


def state_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    def mean(key: str) -> float | None:
        selected = [float(row[key]) for row in values if row.get(key) is not None]
        return statistics.mean(selected) if selected else None
    return {
        "rows": len(values),
        "feasible_no_assign_count": sum(bool(row["feasible_no_assign"]) for row in values),
        "exact_tie_count": sum(bool(row["exact_tie"]) for row in values),
        "unique_winner_count": sum(not bool(row["exact_tie"]) for row in values),
        "mean_no_assign_minus_best_pair": mean("no_assign_minus_best_pair"),
        "mean_candidate_only_margin": mean("candidate_only_top1_top2_margin"),
        "mean_entropy": mean("entropy"),
    }


def direction(old: float | int | None, new: float | int | None) -> str:
    if old is None or new is None:
        return "NOT_AVAILABLE"
    if new < old:
        return "DECREASED"
    if new > old:
        return "INCREASED"
    return "UNCHANGED_EXACT"


def metric_change(e0_rows: Sequence[Mapping[str, Any]], e1_rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    e0 = [float(row[key]) for row in e0_rows if row.get(key) is not None]
    e1 = [float(row[key]) for row in e1_rows if row.get(key) is not None]
    require(len(e0) == len(e1), SAME_SUPPORT_BLOCK, f"metric_pair_count={key}")
    before = statistics.mean(e0) if e0 else None
    after = statistics.mean(e1) if e1 else None
    return {"paired_rows": len(e0), "e0_mean": before, "e1_mean": after,
            "e1_minus_e0": None if before is None or after is None else after - before,
            "direction": direction(before, after)}


def actor_record(*, actor: torch.nn.Module, payload: Mapping[str, Any], R1: Any, H: Any, TIE: Any,
                 device: torch.device, policy_state: str) -> dict[str, Any]:
    result = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE)
    metadata, tensors = payload["metadata"], payload["tensors"]
    support = int(result["support_size"])
    pair_logits = [float(value) for value in result["pair_logits"][0].detach().cpu().tolist()]
    probabilities = [float(value) for value in result["probabilities"][0].detach().cpu().tolist()]
    no_assign = float(result["no_assign_logit"][0, 0].detach().cpu())
    mask = tensors["safe_mask"][0, :support].detach().cpu()
    candidate_ids = list(metadata["candidate_ids"])
    selected = action_identity(result["selection"])
    legal_pairs = {(str(row["agent_id"]), str(row["candidate_id"])) for row in candidate_ids}
    legal = selected == "NO_ASSIGN" or selected in legal_pairs
    require(len(pair_logits) == support == len(candidate_ids) and len(probabilities) == support + 1,
            SAME_SUPPORT_BLOCK, f"actor_output_shape={metadata['decision_id']}")
    require(bool(mask.all().item()), STRUCTURAL_BLOCK, f"unsafe_frozen_support={metadata['decision_id']}")
    require(legal, STRUCTURAL_BLOCK, f"illegal_selection={metadata['decision_id']}")
    ordered = sorted(pair_logits, reverse=True)
    candidate_margin = ordered[0] - ordered[1] if len(ordered) >= 2 else None
    finite = bool(result["finite"] and all(math.isfinite(value) for value in [*pair_logits, no_assign, *probabilities]))
    require(finite, STRUCTURAL_BLOCK, f"nonfinite={metadata['decision_id']}")
    input_identity = {
        "snapshot_digest": str(payload["snapshot_digest"]),
        "candidate_support_digest": str(metadata["candidate_support_digest"]),
        "candidate_ids_sha256": json_sha256(candidate_ids),
        "safe_mask_sha256": bytes_sha256(mask.contiguous().numpy().tobytes()),
        "support_size": support,
        "actor_config_sha256": str(metadata["actor_config_sha256"]),
    }
    return {
        "policy_state": policy_state,
        "replicate_id": "F1_R1" if int(metadata["seed"]) == 20260822 else "F1_R2",
        "window_id": str(metadata["window_id"]),
        "decision_id": str(metadata["decision_id"]),
        "snapshot_digest": str(payload["snapshot_digest"]),
        "input_identity": input_identity,
        "candidate_ids": candidate_ids,
        "support_size": support,
        "selected_identity": selected,
        "selected_action_family": action_family(selected),
        "legal_selection": legal,
        "feasible_no_assign": bool(selected == "NO_ASSIGN" and support > 0),
        "no_assign_logit": no_assign,
        "best_pair_logit": max(pair_logits),
        "no_assign_minus_best_pair": no_assign - max(pair_logits),
        "candidate_only_top1_top2_margin": candidate_margin,
        "exact_tie": bool(result["selection"].exact_tie),
        "tie_set_size": len(result["selection"].tie_set),
        "entropy": float(result["entropy"]),
        "pair_logits": pair_logits,
        "probabilities": probabilities,
        "finite": finite,
        "zero_loss_fail_candidate_in_support": 0,
    }


def max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    require(len(left) == len(right), SAME_SUPPORT_BLOCK, "score_vector_length_mismatch")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def pair_rows(*, initial: Sequence[Mapping[str, Any]], e0: Sequence[Mapping[str, Any]],
              e1: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    maps = [{str(row["snapshot_digest"]): row for row in values} for values in (initial, e0, e1)]
    require(len(maps[0]) == len(initial) == len(maps[1]) == len(e0) == len(maps[2]) == len(e1),
            SAME_SUPPORT_BLOCK, "duplicate_or_missing_same_support_rows")
    require(set(maps[0]) == set(maps[1]) == set(maps[2]), SAME_SUPPORT_BLOCK, "snapshot_pairing_mismatch")
    result: list[dict[str, Any]] = []
    for digest in sorted(maps[0]):
        start, e0_row, e1_row = maps[0][digest], maps[1][digest], maps[2][digest]
        require(start["input_identity"] == e0_row["input_identity"] == e1_row["input_identity"],
                SAME_SUPPORT_BLOCK, f"same_input_identity={digest}")
        result.append({
            "snapshot_digest": digest,
            "window_id": start["window_id"],
            "decision_id": start["decision_id"],
            "input_identity": start["input_identity"],
            "initial": start,
            "e0_final": e0_row,
            "e1_final": e1_row,
            "e0_to_e1": {
                "selection_changed": e0_row["selected_identity"] != e1_row["selected_identity"],
                "action_family_changed": e0_row["selected_action_family"] != e1_row["selected_action_family"],
                "max_abs_logit_delta": max_abs_delta([*e0_row["pair_logits"], e0_row["no_assign_logit"]],
                                                       [*e1_row["pair_logits"], e1_row["no_assign_logit"]]),
                "max_abs_probability_delta": max_abs_delta(e0_row["probabilities"], e1_row["probabilities"]),
                "no_assign_minus_best_pair_delta": e1_row["no_assign_minus_best_pair"] - e0_row["no_assign_minus_best_pair"],
                "candidate_margin_delta": (None if e0_row["candidate_only_top1_top2_margin"] is None else
                                            e1_row["candidate_only_top1_top2_margin"] - e0_row["candidate_only_top1_top2_margin"]),
                "entropy_delta": e1_row["entropy"] - e0_row["entropy"],
            },
        })
    return result


def r1_review(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    paired = pair_rows(initial=rows["initial"], e0=rows["e0_final"], e1=rows["e1_final"])
    before, after = state_summary(rows["e0_final"]), state_summary(rows["e1_final"])
    no_assign_direction = direction(before["feasible_no_assign_count"], after["feasible_no_assign_count"])
    return {
        "replicate_id": "F1_R1",
        "rows": paired,
        "e0_summary": before,
        "e1_summary": after,
        "e0_to_e1": {
            "feasible_no_assign": {"e0_count": before["feasible_no_assign_count"], "e1_count": after["feasible_no_assign_count"],
                                    "direction": no_assign_direction},
            "no_assign_minus_best_pair": metric_change(rows["e0_final"], rows["e1_final"], "no_assign_minus_best_pair"),
            "candidate_only_top1_top2_margin": metric_change(rows["e0_final"], rows["e1_final"], "candidate_only_top1_top2_margin"),
            "entropy": metric_change(rows["e0_final"], rows["e1_final"], "entropy"),
            "exact_tie": {"e0_count": before["exact_tie_count"], "e1_count": after["exact_tie_count"],
                          "direction": direction(before["exact_tie_count"], after["exact_tie_count"])},
        },
        "no_performance_interpretation": True,
    }


def r2_immutability(*, initial_rows: Sequence[Mapping[str, Any]], e0_rows: Sequence[Mapping[str, Any]],
                    e1_rows: Sequence[Mapping[str, Any]], initial_actor_sha: str, e0_actor_sha: str,
                    e1_actor_sha: str, state_tensors_exact: bool) -> dict[str, Any]:
    paired = pair_rows(initial=initial_rows, e0=e0_rows, e1=e1_rows)
    max_logit = max((max_abs_delta([*row["initial"]["pair_logits"], row["initial"]["no_assign_logit"]],
                                   [*row["e1_final"]["pair_logits"], row["e1_final"]["no_assign_logit"]]) for row in paired), default=0.0)
    max_probability = max((max_abs_delta(row["initial"]["probabilities"], row["e1_final"]["probabilities"]) for row in paired), default=0.0)
    selection_mismatch = sum(row["initial"]["selected_identity"] != row["e1_final"]["selected_identity"] for row in paired)
    e0_e1 = {
        "actor_state_sha_equal": e0_actor_sha == e1_actor_sha,
        "selection_mismatch_count": sum(row["e0_final"]["selected_identity"] != row["e1_final"]["selected_identity"] for row in paired),
        "max_abs_logit_delta": max((row["e0_to_e1"]["max_abs_logit_delta"] for row in paired), default=0.0),
        "max_abs_probability_delta": max((row["e0_to_e1"]["max_abs_probability_delta"] for row in paired), default=0.0),
        "action_family_mismatch_count": sum(row["e0_to_e1"]["action_family_changed"] for row in paired),
        "attribution_scope": "observed E0-versus-E1 Actor difference on the same preserved inputs; E1 had zero eligible R2 Actor rows and performed an explicit Actor skip",
    }
    immutable = initial_actor_sha == e1_actor_sha and state_tensors_exact and max_logit == 0.0 and max_probability == 0.0 and selection_mismatch == 0
    return {
        "replicate_id": "F1_R2",
        "rows": paired,
        "initial_e1_immutability": {
            "initial_actor_state_sha256": initial_actor_sha,
            "e1_actor_state_sha256": e1_actor_sha,
            "actor_state_sha_equal": initial_actor_sha == e1_actor_sha,
            "actor_state_tensors_exact": state_tensors_exact,
            "max_abs_logit_delta": max_logit,
            "max_abs_probability_delta": max_probability,
            "selection_mismatch_count": selection_mismatch,
            "passed": immutable,
        },
        "e0_vs_e1_behavior": e0_e1,
        "e0_actor_state_sha256": e0_actor_sha,
        "no_performance_interpretation": True,
    }


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def seed_view(rows_r1: Sequence[Mapping[str, Any]], rows_r2: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_window_r1 = {str(row["window_id"]): row for row in rows_r1}
    by_window_r2 = {str(row["window_id"]): row for row in rows_r2}
    require(len(by_window_r1) == len(rows_r1) == len(by_window_r2) == len(rows_r2), SAME_SUPPORT_BLOCK, "cross_seed_duplicate_window")
    require(set(by_window_r1) == set(by_window_r2), SAME_SUPPORT_BLOCK, "cross_seed_window_alignment")
    rows = []
    for window_id in sorted(by_window_r1):
        r1, r2 = by_window_r1[window_id], by_window_r2[window_id]
        support_equal = r1["input_identity"]["candidate_support_digest"] == r2["input_identity"]["candidate_support_digest"]
        candidate_gap = None if r1["candidate_only_top1_top2_margin"] is None or r2["candidate_only_top1_top2_margin"] is None else abs(r1["candidate_only_top1_top2_margin"] - r2["candidate_only_top1_top2_margin"])
        rows.append({
            "window_id": window_id,
            "r1_snapshot_digest": r1["snapshot_digest"],
            "r2_snapshot_digest": r2["snapshot_digest"],
            "candidate_support_digest_equal": support_equal,
            "candidate_identity_comparison": "EXACT_ONLY_IF_SUPPORT_EQUAL" if not support_equal else "EXACT_SUPPORT_MATCH",
            "r1_selected_action_family": r1["selected_action_family"],
            "r2_selected_action_family": r2["selected_action_family"],
            "action_family_agree": r1["selected_action_family"] == r2["selected_action_family"],
            "selected_identity_equal_when_comparable": (r1["selected_identity"] == r2["selected_identity"]) if support_equal else None,
            "abs_no_assign_minus_best_pair_gap": abs(r1["no_assign_minus_best_pair"] - r2["no_assign_minus_best_pair"]),
            "abs_candidate_margin_gap": candidate_gap,
            "abs_entropy_gap": abs(r1["entropy"] - r2["entropy"]),
        })
    return {
        "rows": rows,
        "same_window_rows": len(rows),
        "candidate_support_exact_match_rows": sum(bool(row["candidate_support_digest_equal"]) for row in rows),
        "action_family_mismatch_count": sum(not bool(row["action_family_agree"]) for row in rows),
        "mean_abs_no_assign_minus_best_pair_gap": mean_or_none([float(row["abs_no_assign_minus_best_pair_gap"]) for row in rows]),
        "mean_abs_candidate_margin_gap": mean_or_none([float(row["abs_candidate_margin_gap"]) for row in rows if row["abs_candidate_margin_gap"] is not None]),
        "mean_abs_entropy_gap": mean_or_none([float(row["abs_entropy_gap"]) for row in rows]),
    }


def seed_divergence_classification(e0: Mapping[str, Any], e1: Mapping[str, Any]) -> str:
    if min(int(e0["same_window_rows"]), int(e1["same_window_rows"])) < 3:
        return "INSUFFICIENT_REVIEW_ROWS"
    keys = ("action_family_mismatch_count", "mean_abs_no_assign_minus_best_pair_gap",
            "mean_abs_candidate_margin_gap", "mean_abs_entropy_gap")
    compared = [(float(e0[key]), float(e1[key])) for key in keys if e0.get(key) is not None and e1.get(key) is not None]
    if not compared:
        return "INSUFFICIENT_REVIEW_ROWS"
    if all(after <= before for before, after in compared) and any(after < before for before, after in compared):
        return "SEED_DIVERGENCE_REDUCED"
    if all(after >= before for before, after in compared) and any(after > before for before, after in compared):
        return "SEED_DIVERGENCE_INCREASED"
    return "SEED_DIVERGENCE_PERSISTS"


def order_audit(*, policies: Mapping[str, Mapping[str, torch.nn.Module]], snapshots: Mapping[str, Sequence[Mapping[str, Any]]],
                R1: Any, H: Any, TIE: Any, device: torch.device) -> tuple[dict[str, Any], list[str], int]:
    rows: list[dict[str, Any]] = []
    hard: list[str] = []
    forwards = 0
    for replicate_id, states in policies.items():
        for state, actor in states.items():
            before = module_digest(actor)
            for payload in snapshots[replicate_id]:
                base = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE); forwards += 1
                candidate_payload = R1.candidate_permutation(payload)
                agent_payload = R1.agent_permutation(payload)
                combined_payload = R1.candidate_permutation(agent_payload)
                candidate = R1.frozen_forward(actor, candidate_payload, device=device, head=H, tie=TIE); forwards += 1
                agent = R1.frozen_forward(actor, agent_payload, device=device, head=H, tie=TIE); forwards += 1
                combined = R1.frozen_forward(actor, combined_payload, device=device, head=H, tie=TIE); forwards += 1
                base_logits, base_probabilities = R1.scores_by_identity(base, payload)
                def compare(other: Mapping[str, Any], other_payload: Mapping[str, Any]) -> tuple[float, float]:
                    logits, probabilities = R1.scores_by_identity(other, other_payload)
                    require(set(logits) == set(base_logits) and set(probabilities) == set(base_probabilities), STRUCTURAL_BLOCK,
                            "identity_mapping_incomplete")
                    return (max((abs(base_logits[key] - logits[key]) for key in base_logits), default=0.0),
                            max((abs(base_probabilities[key] - probabilities[key]) for key in base_probabilities), default=0.0))
                candidate_delta = compare(candidate, candidate_payload)
                agent_delta = compare(agent, agent_payload)
                combined_delta = compare(combined, combined_payload)
                rows.append({
                    "replicate_id": replicate_id,
                    "policy_state": state,
                    "snapshot_digest": payload["snapshot_digest"],
                    "multi_candidate": int(base["support_size"]) >= 2,
                    "candidate_identity_equal": action_identity(base["selection"]) == action_identity(candidate["selection"]),
                    "agent_identity_equal": action_identity(base["selection"]) == action_identity(agent["selection"]),
                    "combined_identity_equal": action_identity(base["selection"]) == action_identity(combined["selection"]),
                    "candidate_logit_delta": candidate_delta[0], "candidate_probability_delta": candidate_delta[1],
                    "agent_logit_delta": agent_delta[0], "agent_probability_delta": agent_delta[1],
                    "combined_logit_delta": combined_delta[0], "combined_probability_delta": combined_delta[1],
                })
            if module_digest(actor) != before:
                hard.append("ACTOR_PARAMETER_MUTATION_DURING_ORDER_AUDIT")
    def summary(prefix: str, scope: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"tested": len(scope), "identity_changes": sum(not bool(row[f"{prefix}_identity_equal"]) for row in scope),
                "max_identity_aligned_logit_delta": max((float(row[f"{prefix}_logit_delta"]) for row in scope), default=0.0),
                "max_identity_aligned_probability_delta": max((float(row[f"{prefix}_probability_delta"]) for row in scope), default=0.0)}
    multi = [row for row in rows if row["multi_candidate"]]
    candidate = summary("candidate", multi)
    agent = summary("agent", rows)
    combined = summary("combined", multi)
    if candidate["identity_changes"] or agent["identity_changes"] or combined["identity_changes"]:
        hard.append("ORDER_INVARIANCE_FAILURE")
    return {
        "selector": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "tolerance": 0.0},
        "candidate_order": candidate, "agent_order": agent, "combined_order": combined,
        "candidate_order_all_passed": candidate["identity_changes"] == 0,
        "agent_order_all_passed": agent["identity_changes"] == 0,
        "combined_order_all_passed": combined["identity_changes"] == 0,
        "numerical_rule": "no new tolerance; identities must be exact and observed identity-aligned deltas are preserved",
    }, hard, forwards


def write_block(*, root: Path, source: Mapping[str, Any], preflight: Mapping[str, Any], reason: str) -> None:
    zero = counters()
    outputs = {
        "bt8r10_evidence_preflight.json": preflight,
        "bt8r10_r1_e0_e1_review.json": {"not_run": True},
        "bt8r10_r2_initial_e0_e1_review.json": {"not_run": True},
        "bt8r10_seed_divergence.json": {"not_run": True},
        "bt8r10_order_invariance.json": {"not_run": True},
        "test_results.json": {"execution_counters": zero, "hard_failures": [reason], "warnings": [], "github_push_performed": False},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": reason, "classification": "BLOCKED", "source_commit": source["source_commit"],
                               "hard_failures": [reason], "warnings": [], "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(f"# BT8-R10 blocked\n\n- gate: `{reason}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": reason, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(reason + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1
    import run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution as R6

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), SAME_SUPPORT_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    preflight: dict[str, Any] = {"source": source, "required_sources": {"e0_f1": E0_F1_SOURCE, "e1": E1_SOURCE, "r9": R9_SOURCE},
                                 "e1_contract_sha256": E1_CONTRACT_SHA256, "t1_tolerance": 0.0}
    try:
        f1_gate = load_json(F1 / "gate_decision.json")
        e1_gate = load_json(E1 / "gate_decision.json")
        r9_gate = load_json(R9 / "gate_decision.json")
        f1_audit, e1_audit, r9_audit = manifest_audit(F1), manifest_audit(E1), manifest_audit(R9)
        e1_frozen = load_json(E1 / "frozen_hash_before_after.json")
        collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
        initial_manifest = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
        e0_manifest = load_json(F1 / "bt8f1_final_checkpoint_manifest.json")
        e1_manifest = load_json(E1 / "bt8f1e1_final_checkpoint_manifest.json")
        initial_replay = load_json(F1 / "bt8f1_initial_review_replay.json")
        e0_replay = load_json(F1 / "bt8f1_final_review_replay.json")
        e1_replay = load_json(E1 / "bt8f1e1_final_review_replay.json")
        require(f1_gate.get("gate") == F1_GATE and f1_gate.get("source_commit") == E0_F1_SOURCE, SAME_SUPPORT_BLOCK, "f1_gate")
        require(e1_gate.get("gate") == E1_GATE and e1_gate.get("source_commit") == E1_SOURCE, SAME_SUPPORT_BLOCK, "e1_gate")
        require(r9_gate.get("gate") == R9_GATE and r9_gate.get("source_commit") == R9_SOURCE, SAME_SUPPORT_BLOCK, "r9_gate")
        require(f1_audit["all_match"] and e1_audit["all_match"] and r9_audit["all_match"], SAME_SUPPORT_BLOCK, "manifest_hash")
        require(e1_frozen.get("e1_contract_sha256") == E1_CONTRACT_SHA256 and e1_frozen.get("all_unchanged") is True,
                SAME_SUPPORT_BLOCK, "e1_contract")
        collection_copy = dict(collection); supplied_digest = collection_copy.pop("collection_digest", None)
        require(supplied_digest == FPS.canonical_sha256(collection_copy), SAME_SUPPORT_BLOCK, "review_collection_digest")
        require(supplied_digest == "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e", SAME_SUPPORT_BLOCK, "unexpected_review_collection")
        require(e1_replay.get("review_collection_digest") == supplied_digest and e1_replay.get("same_support_binding_pass") is True,
                SAME_SUPPORT_BLOCK, "e1_review_collection")
        require(int(collection.get("snapshot_count", -1)) == len(collection.get("entries", [])) == 6,
                SAME_SUPPORT_BLOCK, "review_snapshot_count")
        require(len({entry.get("snapshot_digest") for entry in collection["entries"]}) == 6, SAME_SUPPORT_BLOCK, "review_snapshot_unique")
        expected_review_digests = {str(entry["snapshot_digest"]) for entry in collection["entries"]}
        replay_binding: dict[str, Any] = {}
        for label, replay in (("initial", initial_replay), ("e0_final", e0_replay), ("e1_final", e1_replay)):
            rows = replay.get("rows")
            require(isinstance(rows, list) and len(rows) == 6, SAME_SUPPORT_BLOCK, f"historical_replay_rows={label}")
            digests = {str(row.get("snapshot_digest")) for row in rows}
            require(digests == expected_review_digests, SAME_SUPPORT_BLOCK, f"historical_replay_inputs={label}")
            replay_binding[label] = {"row_count": len(rows), "snapshot_digest_set_exact": True}
        binding = collection.get("checkpoint_binding", {})
        checkpoints: dict[str, dict[str, Path]] = {}
        checkpoint_before: dict[str, str] = {}
        for replicate_id in REPLICATES:
            initial, e0, e1 = initial_manifest.get(replicate_id), e0_manifest.get(replicate_id), e1_manifest.get(replicate_id)
            require(isinstance(initial, Mapping) and isinstance(e0, Mapping) and isinstance(e1, Mapping), SAME_SUPPORT_BLOCK,
                    f"checkpoint_manifest={replicate_id}")
            paths = {"initial": F1 / "initial_checkpoints" / str(initial["path"]),
                     "e0_final": F1 / "final_checkpoints" / str(e0["path"]),
                     "e1_final": E1 / str(e1["path"])}
            require(sha256(paths["initial"]) == initial["sha256"] == binding["initial_actor_checkpoint_sha256_by_replicate"][replicate_id],
                    SAME_SUPPORT_BLOCK, f"initial_checkpoint={replicate_id}")
            require(sha256(paths["e0_final"]) == e0["sha256"] == binding["final_actor_checkpoint_sha256_by_replicate"][replicate_id],
                    SAME_SUPPORT_BLOCK, f"e0_checkpoint={replicate_id}")
            require(sha256(paths["e1_final"]) == e1["sha256"] and str(e1["initial_checkpoint_sha256"]) == str(initial["sha256"]),
                    SAME_SUPPORT_BLOCK, f"e1_checkpoint={replicate_id}")
            require(str(e1["actor_initial_digest"]) == str(e0["actor_initial_digest"]), SAME_SUPPORT_BLOCK,
                    f"e1_initial_actor_binding={replicate_id}")
            checkpoints[replicate_id] = paths
            checkpoint_before |= {f"{replicate_id}:{name}": sha256(path) for name, path in paths.items()}
        snapshots: dict[str, list[dict[str, Any]]] = {key: [] for key in REPLICATES}
        config: Mapping[str, Any] | None = None
        snapshot_preflight: list[dict[str, Any]] = []
        for entry in collection["entries"]:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / str(entry["relative_path"]))
            require(payload["snapshot_digest"] == entry["snapshot_digest"], SAME_SUPPORT_BLOCK, "snapshot_digest")
            meta, tensors = payload["metadata"], payload["tensors"]
            seed = int(meta["seed"])
            replicate_id = "F1_R1" if seed == 20260822 else "F1_R2" if seed == 20260823 else ""
            require(replicate_id in REPLICATES, SAME_SUPPORT_BLOCK, f"snapshot_seed={seed}")
            require(meta["actor_config"] == (config if config is not None else meta["actor_config"]), SAME_SUPPORT_BLOCK, "actor_config")
            config = meta["actor_config"]
            require(meta["actor_config_sha256"] == FPS.actor_config_sha256(config), SAME_SUPPORT_BLOCK, "actor_config_sha")
            require(meta["candidate_ids"] == meta["candidate_order"] and int(meta["no_assign_index"]) == int(meta["selectable_pair_count"]),
                    SAME_SUPPORT_BLOCK, "snapshot_order")
            support = int(meta["selectable_pair_count"])
            require(bool(tensors["safe_mask"][0, :support].all().item()), STRUCTURAL_BLOCK, "frozen_safe_mask")
            require(meta["captured_device"] == "mps:0", SAME_SUPPORT_BLOCK, "snapshot_device")
            snapshots[replicate_id].append(payload)
            snapshot_preflight.append({"replicate_id": replicate_id, "snapshot_digest": payload["snapshot_digest"],
                                       "window_id": meta["window_id"], "decision_id": meta["decision_id"],
                                       "candidate_support_digest": meta["candidate_support_digest"], "support_size": support})
        require(config is not None and config.get("actor_head_id") == H.CANDIDATE_SENSITIVE_HEAD_ID
                and config.get("actor_head_version") == H.CANDIDATE_SENSITIVE_HEAD_VERSION, SAME_SUPPORT_BLOCK, "v2_config")
        require(all(len(rows) == 3 for rows in snapshots.values()), SAME_SUPPORT_BLOCK, "replicate_snapshot_count")
        current_frozen = frozen_hashes(BT6, R6)
        preflight |= {
            "f1_manifest": f1_audit, "e1_manifest": e1_audit, "r9_manifest": r9_audit,
            "gates": {"f1": True, "e1": True, "r9": True}, "review_collection_digest": supplied_digest,
            "historical_replay_binding": replay_binding,
            "review_snapshot_count": 6, "review_snapshot_unique": 6, "snapshot_rows": snapshot_preflight,
            "actor_config": dict(config), "actor_config_sha256": FPS.actor_config_sha256(config),
            "t1_contract": TIE.TIE_BREAK_CONTRACT_ID == T1_CONTRACT,
            "t1_selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"),
            "frozen_hash_binding": current_frozen == e1_frozen.get("after"),
            "checkpoint_before": checkpoint_before,
        }
        require(preflight["t1_contract"] and preflight["frozen_hash_binding"] and source["source_lineage_descends_from_e1"]
                and source["source_only_local_commit"], SAME_SUPPORT_BLOCK, "source_or_frozen_binding")
    except R10Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, reason=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}"); return
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, reason=SAME_SUPPORT_BLOCK)
        print(f"[BLOCKED] {SAME_SUPPORT_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root=root, source=source, preflight=preflight | {"mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()}, reason=MPS_BLOCK)
        print(f"[BLOCKED] {MPS_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return

    device = torch.device("mps:0")
    execution = counters()
    hard: list[str] = []
    policies: dict[str, dict[str, torch.nn.Module]] = {replicate_id: {} for replicate_id in REPLICATES}
    actor_state_sha: dict[str, dict[str, str]] = {replicate_id: {} for replicate_id in REPLICATES}
    raw_states: dict[str, dict[str, Mapping[str, Any]]] = {replicate_id: {} for replicate_id in REPLICATES}
    try:
        for replicate_id in REPLICATES:
            expected = {
                "initial": str(e0_manifest[replicate_id]["actor_initial_digest"]),
                "e0_final": str(e0_manifest[replicate_id]["actor_final_digest"]),
                "e1_final": str(e1_manifest[replicate_id]["actor_final_digest"]),
            }
            for state in POLICY_STATES:
                payload = torch.load(checkpoints[replicate_id][state], map_location="cpu", weights_only=False)
                require(isinstance(payload, Mapping) and "actor" in payload and "meta" in payload, SAME_SUPPORT_BLOCK,
                        f"actor_checkpoint_schema={replicate_id}:{state}")
                actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
                    global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=int(config["agent_dim"]),
                    candidate_dim=int(config["candidate_dim"]), hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
                actor.load_state_dict(payload["actor"], strict=True)
                actor.eval()
                digest = module_digest(actor)
                require(digest == expected[state], SAME_SUPPORT_BLOCK, f"actor_state_digest={replicate_id}:{state}")
                policies[replicate_id][state] = actor
                actor_state_sha[replicate_id][state] = digest
                raw_states[replicate_id][state] = payload["actor"]
        records: dict[str, dict[str, list[dict[str, Any]]]] = {replicate_id: {state: [] for state in POLICY_STATES} for replicate_id in REPLICATES}
        module_before = {replicate_id: {state: module_digest(actor) for state, actor in states.items()} for replicate_id, states in policies.items()}
        for replicate_id, states in policies.items():
            for state, actor in states.items():
                for payload in snapshots[replicate_id]:
                    row = actor_record(actor=actor, payload=payload, R1=R1, H=H, TIE=TIE, device=device, policy_state=state)
                    records[replicate_id][state].append(row)
                    execution["frozen_policy_rows"] += 1
                    execution["nan_or_inf"] += int(not row["finite"])
                    execution["illegal_or_masked_selection"] += int(not row["legal_selection"])
                    execution["zero_loss_fail_support_or_selection"] += int(row["zero_loss_fail_candidate_in_support"])
        torch.mps.synchronize()
        module_after = {replicate_id: {state: module_digest(actor) for state, actor in states.items()} for replicate_id, states in policies.items()}
        execution["parameter_mutation"] = sum(module_before[replicate_id][state] != module_after[replicate_id][state]
                                                for replicate_id in REPLICATES for state in POLICY_STATES)
        r1 = r1_review(records["F1_R1"])
        r2_exact_state = all(torch.equal(raw_states["F1_R2"]["initial"][name], raw_states["F1_R2"]["e1_final"][name])
                             for name in raw_states["F1_R2"]["initial"])
        r2 = r2_immutability(initial_rows=records["F1_R2"]["initial"], e0_rows=records["F1_R2"]["e0_final"],
                              e1_rows=records["F1_R2"]["e1_final"], initial_actor_sha=actor_state_sha["F1_R2"]["initial"],
                              e0_actor_sha=actor_state_sha["F1_R2"]["e0_final"], e1_actor_sha=actor_state_sha["F1_R2"]["e1_final"],
                              state_tensors_exact=r2_exact_state)
        if not r2["initial_e1_immutability"]["passed"]:
            hard.append(R2_IMMUTABILITY_BLOCK)
        seed_e0 = seed_view(records["F1_R1"]["e0_final"], records["F1_R2"]["e0_final"])
        seed_e1 = seed_view(records["F1_R1"]["e1_final"], records["F1_R2"]["e1_final"])
        seed_status = seed_divergence_classification(seed_e0, seed_e1)
        seed = {"e0": seed_e0, "e1": seed_e1, "classification": seed_status,
                "interpretation_scope": "same frozen window alignment; candidate identity is only compared when the two frozen support digests match"}
        order, order_hard, probe_forwards = order_audit(policies=policies, snapshots=snapshots, R1=R1, H=H, TIE=TIE, device=device)
        execution["order_probe_forwards"] = probe_forwards
        hard.extend(order_hard)
        after_frozen = frozen_hashes(BT6, R6)
        checkpoint_after = {f"{replicate_id}:{state}": sha256(path)
                            for replicate_id, states in checkpoints.items() for state, path in states.items()}
        execution["checkpoint_mutation"] = sum(checkpoint_before[key] != checkpoint_after[key] for key in checkpoint_before)
        forbidden = ("training", "optimizer_step", "causal_rollout", "candidate_generation", "candidate_regeneration", "local_search_rerun",
                     "zero_loss_reevaluation", "parameter_mutation", "checkpoint_write", "checkpoint_mutation", "review_optimizer_rows",
                     "future_leakage", "nan_or_inf", "illegal_or_masked_selection", "zero_loss_fail_support_or_selection", "test6_access", "github_push")
        if any(execution[name] != 0 for name in forbidden):
            hard.append(STRUCTURAL_BLOCK)
        if after_frozen != current_frozen:
            hard.append("FROZEN_HASH_MUTATION")
        r1_direction = r1["e0_to_e1"]["feasible_no_assign"]["direction"]
        if r1_direction == "INCREASED" or seed_status == "SEED_DIVERGENCE_INCREASED":
            classification = "D_E1_BEHAVIOR_WORSENED_OR_SEED_DIVERGENCE_INCREASED"
        elif r1_direction == "DECREASED":
            classification = "A_E1_REDUCES_UNSUPPORTED_ACTOR_DRIFT_AND_NO_ASSIGN_ATTRACTION"
        elif r2["e0_vs_e1_behavior"]["actor_state_sha_equal"] is False:
            classification = "B_E1_BLOCKS_UNSUPPORTED_R2_DRIFT_BUT_R1_NO_ASSIGN_PERSISTS"
        else:
            classification = "C_E1_IMPLEMENTATION_VALID_WITH_NO_MATERIAL_R1_BEHAVIOR_CHANGE"
        if hard:
            classification = "BLOCKED"
        gate = PASS_GATE if not hard else hard[0]
        outputs = {
            "bt8r10_evidence_preflight.json": preflight | {"mps_built": True, "mps_available": True, "device": "mps:0",
                                                              "checkpoint_state_sha256": actor_state_sha},
            "bt8r10_r1_e0_e1_review.json": r1,
            "bt8r10_r2_initial_e0_e1_review.json": r2,
            "bt8r10_seed_divergence.json": seed,
            "bt8r10_order_invariance.json": order,
            "test_results.json": {"execution_counters": execution, "hard_failures": hard, "warnings": [],
                                  "github_push_performed": False, "performance_interpretation_performed": False},
            "frozen_hash_before_after.json": {"before": current_frozen, "after": after_frozen, "all_unchanged": current_frozen == after_frozen,
                                                 "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after,
                                                 "checkpoint_unchanged": checkpoint_before == checkpoint_after,
                                                 "e1_contract_sha256": E1_CONTRACT_SHA256},
            "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification,
                                   "source_commit": source["source_commit"], "hard_failures": hard, "warnings": [],
                                   "global_locks": LOCKS,
                                   "next_step": "separate minimal repair-selection gate only if the frozen review evidence warrants it" if not hard else "STOP"},
        }
        for name, value in outputs.items():
            dump(root / name, value)
        (root / "final_report.md").write_text(
            f"# BT8-R10 final report\n\n- gate: `{gate}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n\n"
            "This is same-input frozen inference only. It makes no KPI, policy-quality, performance, or causal-performance claim.\n",
            encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                       "source_commit": source["source_commit"], "file_sha256": manifest, "github_push_performed": False})
        (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
        print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
        print(f"classification: {classification}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except R10Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, reason=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, reason=STRUCTURAL_BLOCK)
        print(f"[BLOCKED] {STRUCTURAL_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
