#!/usr/bin/env python3
"""BT8-R14: read-only S3 four-cell frozen-policy factor review.

Only the R13 evidence checkpoints and the original six lossless F1 review
snapshots are read.  This runner deliberately contains no training, rollout,
candidate construction, state transition, reward computation, optimizer, or
checkpoint-write path.  It is an input-preserving frozen inference audit.
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


STAGE = "H4M-AE-R9.8-LS3-BT8-R14"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R14_S3_SAME_INPUT_FROZEN_POLICY_FACTOR_REVIEW_COMPLETE"
BINDING_BLOCK = "BLOCKED_S3_FROZEN_FACTOR_EVIDENCE_BINDING_FAILURE"
BD_IMMUTABILITY_BLOCK = "BLOCKED_S3_BD_ACTOR_IMMUTABILITY_FAILURE"
STRUCTURAL_BLOCK = "BLOCKED_S3_FROZEN_FACTOR_STRUCTURAL_INTEGRITY_FAILURE"
MPS_BLOCK = "BLOCKED_MPS_FROZEN_POLICY_REVIEW_ENVIRONMENT_UNAVAILABLE"

R13_SOURCE = "a9818399c4a1a74734496a146b2b98fabab513b0"
R12_SOURCE = "6e22b95d80720255ca17b1ab0520417b89e5a5c9"
R11_SOURCE = "f467feef8246c6c77dc67a360aecc60e0102913e"
S3_CONTRACT_SHA256 = "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"
R13_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_S3_FOUR_CELL_ON_POLICY_CAUSAL_EXECUTION_COMPLETE"
R12_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R12_S3_FACTORIAL_EXECUTION_AUTHORITY_SELECTION_COMPLETE"
R11_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R11_MINIMAL_SEED_DIVERGENCE_CAUSE_ISOLATION_AND_REPAIR_SELECTION_COMPLETE"
T1_CONTRACT = "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1"
REVIEW_COLLECTION_DIGEST = "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R13 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r13_s3_four_cell_execution_20260826_174729+09:00"
R12 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection_20260826_145345+09:00"
R11 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection_20260826_123135+09:00"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review.py",
    "05_training/test_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
CELL_EXPECTATIONS = {
    "AC-R1": {"eligible": 9, "actor_steps": 3, "environment": "F1_R1"},
    "AC-R2": {"eligible": 9, "actor_steps": 3, "environment": "F1_R2"},
    "BD-R1": {"eligible": 0, "actor_steps": 0, "environment": "F1_R1"},
    "BD-R2": {"eligible": 0, "actor_steps": 0, "environment": "F1_R2"},
}
POLICIES = {
    "initial_AC": {"checkpoint_kind": "initial", "cell_id": "AC-R1", "expected_digest_key": "actor_initial_digest"},
    "final_AC_R1": {"checkpoint_kind": "final", "cell_id": "AC-R1", "expected_digest_key": "actor_final_digest"},
    "final_AC_R2": {"checkpoint_kind": "final", "cell_id": "AC-R2", "expected_digest_key": "actor_final_digest"},
    "initial_BD": {"checkpoint_kind": "initial", "cell_id": "BD-R1", "expected_digest_key": "actor_initial_digest"},
    "final_BD_R1": {"checkpoint_kind": "final", "cell_id": "BD-R1", "expected_digest_key": "actor_final_digest"},
    "final_BD_R2": {"checkpoint_kind": "final", "cell_id": "BD-R2", "expected_digest_key": "actor_final_digest"},
}


class R14Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R14Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return bytes_sha256(raw)


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


def state_dicts_exact(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in left)


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R13_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r13": git(["merge-base", R13_SOURCE, "HEAD"]) == R13_SOURCE,
        "changed_files_since_r13": changed,
        "source_only_local_commit": set(changed) == SOURCE_FILES,
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review_{stamp}"


def counters() -> dict[str, int]:
    return {
        "training": 0,
        "optimizer_step": 0,
        "causal_rollout": 0,
        "simulator_execution": 0,
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


def action_family(identity: str | tuple[str, str]) -> str:
    return "NO_ASSIGN" if identity == "NO_ASSIGN" else "CANDIDATE"


def max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    require(len(left) == len(right), BINDING_BLOCK, "score_vector_length_mismatch")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def actor_record(*, actor: torch.nn.Module, policy_name: str, checkpoint: Mapping[str, Any], payload: Mapping[str, Any],
                 R1: Any, H: Any, TIE: Any, device: torch.device) -> dict[str, Any]:
    result = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE)
    metadata, tensors = payload["metadata"], payload["tensors"]
    support = int(result["support_size"])
    pair_logits = [float(value) for value in result["pair_logits"][0].detach().cpu().tolist()]
    probabilities = [float(value) for value in result["probabilities"][0].detach().cpu().tolist()]
    no_assign_logit = float(result["no_assign_logit"][0, 0].detach().cpu())
    candidate_ids = list(metadata["candidate_ids"])
    safe_mask = tensors["safe_mask"][0, :support].detach().cpu()
    selected = action_identity(result["selection"])
    legal_pairs = {(str(row["agent_id"]), str(row["candidate_id"])) for row in candidate_ids}
    legal = selected == "NO_ASSIGN" or selected in legal_pairs
    require(len(pair_logits) == support == len(candidate_ids) and len(probabilities) == support + 1,
            BINDING_BLOCK, f"actor_output_shape={metadata['decision_id']}")
    require(bool(safe_mask.all().item()), STRUCTURAL_BLOCK, f"unsafe_frozen_support={metadata['decision_id']}")
    require(legal, STRUCTURAL_BLOCK, f"illegal_selection={metadata['decision_id']}")
    finite = bool(result["finite"] and all(math.isfinite(value) for value in [*pair_logits, no_assign_logit, *probabilities]))
    require(finite, STRUCTURAL_BLOCK, f"nonfinite={metadata['decision_id']}")
    ordered = sorted(pair_logits, reverse=True)
    candidate_margin = ordered[0] - ordered[1] if len(ordered) >= 2 else None
    return {
        "policy": policy_name,
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "actor_state_sha256": str(checkpoint["actor_digest"]),
        "source_cell_id": str(checkpoint["cell_id"]),
        "snapshot_digest": str(payload["snapshot_digest"]),
        "window_id": str(metadata["window_id"]),
        "decision_id": str(metadata["decision_id"]),
        "review_environment_seed": int(metadata["seed"]),
        "candidate_support_digest": str(metadata["candidate_support_digest"]),
        "candidate_ids_sha256": json_sha256(candidate_ids),
        "safe_mask_sha256": bytes_sha256(safe_mask.contiguous().numpy().tobytes()),
        "support_size": support,
        "selected_identity": selected,
        "selected_action_family": action_family(selected),
        "legal_selection": legal,
        "feasible_no_assign": bool(selected == "NO_ASSIGN" and support > 0),
        "no_assign_logit": no_assign_logit,
        "best_pair_logit": max(pair_logits),
        "no_assign_minus_best_pair": no_assign_logit - max(pair_logits),
        "candidate_only_top1_top2_margin": candidate_margin,
        "exact_tie": bool(result["selection"].exact_tie),
        "tie_set_size": len(result["selection"].tie_set),
        "entropy": float(result["entropy"]),
        "pair_logits": pair_logits,
        "probabilities": probabilities,
        "finite": finite,
        "zero_loss_fail_candidate_in_support": 0,
        "candidate_identity_cross_support_comparison": "PROHIBITED_UNLESS_SNAPSHOT_DIGEST_EQUAL",
    }


def rows_by_snapshot(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["snapshot_digest"]): row for row in rows}
    require(len(result) == len(rows), BINDING_BLOCK, "duplicate_snapshot_row")
    return result


def compare_same_snapshot(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    require(left["snapshot_digest"] == right["snapshot_digest"], BINDING_BLOCK, "cross_snapshot_direct_comparison")
    require(left["candidate_support_digest"] == right["candidate_support_digest"], BINDING_BLOCK, "support_mismatch_same_snapshot")
    return {
        "snapshot_digest": left["snapshot_digest"],
        "candidate_identity_comparison_allowed": True,
        "selection_mismatch": left["selected_identity"] != right["selected_identity"],
        "action_family_mismatch": left["selected_action_family"] != right["selected_action_family"],
        "max_abs_logit_delta": max_abs_delta([*left["pair_logits"], left["no_assign_logit"]],
                                               [*right["pair_logits"], right["no_assign_logit"]]),
        "max_abs_probability_delta": max_abs_delta(left["probabilities"], right["probabilities"]),
        "no_assign_minus_best_pair_delta": float(right["no_assign_minus_best_pair"]) - float(left["no_assign_minus_best_pair"]),
        "candidate_margin_delta": (None if left["candidate_only_top1_top2_margin"] is None else
                                    float(right["candidate_only_top1_top2_margin"]) - float(left["candidate_only_top1_top2_margin"])),
        "entropy_delta": float(right["entropy"]) - float(left["entropy"]),
    }


def environment_effect(rows: Sequence[Mapping[str, Any]], actor_states_exact: bool) -> dict[str, Any]:
    values = list(rows)
    require(values, BINDING_BLOCK, "empty_environment_effect")
    action_mismatches = sum(bool(row["action_family_mismatch"]) for row in values)
    selection_mismatches = sum(bool(row["selection_mismatch"]) for row in values)
    max_logit = max(float(row["max_abs_logit_delta"]) for row in values)
    max_probability = max(float(row["max_abs_probability_delta"]) for row in values)
    max_gap = max(abs(float(row["no_assign_minus_best_pair_delta"])) for row in values)
    max_margin = max((abs(float(row["candidate_margin_delta"])) for row in values if row["candidate_margin_delta"] is not None), default=0.0)
    max_entropy = max(abs(float(row["entropy_delta"])) for row in values)
    exact_output = selection_mismatches == 0 and max_logit == 0.0 and max_probability == 0.0 and max_gap == 0.0 and max_margin == 0.0 and max_entropy == 0.0
    if action_mismatches:
        classification = "ENVIRONMENT_EFFECT_MATERIAL"
    elif actor_states_exact and exact_output:
        classification = "ENVIRONMENT_EFFECT_LOW"
    else:
        classification = "MIXED_OR_INDETERMINATE"
    return {
        "paired_rows": len(values),
        "actor_states_exact": actor_states_exact,
        "action_family_agreement_count": len(values) - action_mismatches,
        "action_family_mismatch_count": action_mismatches,
        "selection_mismatch_count": selection_mismatches,
        "max_abs_logit_delta": max_logit,
        "max_abs_probability_delta": max_probability,
        "max_abs_no_assign_minus_best_pair_delta": max_gap,
        "max_abs_candidate_margin_delta": max_margin,
        "max_abs_entropy_delta": max_entropy,
        "classification": classification,
        "rule": "MATERIAL on any action-family mismatch; LOW only on exact Actor tensor and exact six-input output equality; otherwise MIXED_OR_INDETERMINATE. No numerical tolerance was introduced.",
    }


def bd_immutability(*, initial: Sequence[Mapping[str, Any]], final_r1: Sequence[Mapping[str, Any]],
                    final_r2: Sequence[Mapping[str, Any]], state_digests: Mapping[str, str], tensors_exact: bool) -> dict[str, Any]:
    initial_map, r1_map, r2_map = rows_by_snapshot(initial), rows_by_snapshot(final_r1), rows_by_snapshot(final_r2)
    require(set(initial_map) == set(r1_map) == set(r2_map), BINDING_BLOCK, "bd_snapshot_pairing")
    pairs = []
    for digest in sorted(initial_map):
        left, first, second = initial_map[digest], r1_map[digest], r2_map[digest]
        first_delta, second_delta = compare_same_snapshot(left, first), compare_same_snapshot(left, second)
        pairs.append({"snapshot_digest": digest, "initial_vs_final_bd_r1": first_delta, "initial_vs_final_bd_r2": second_delta})
    deltas = [item[key] for item in pairs for key in ("initial_vs_final_bd_r1", "initial_vs_final_bd_r2")]
    max_logit = max((float(row["max_abs_logit_delta"]) for row in deltas), default=0.0)
    max_probability = max((float(row["max_abs_probability_delta"]) for row in deltas), default=0.0)
    selection_mismatch = sum(bool(row["selection_mismatch"]) for row in deltas)
    state_digest_equal = len(set(state_digests.values())) == 1
    passed = state_digest_equal and tensors_exact and max_logit == 0.0 and max_probability == 0.0 and selection_mismatch == 0
    return {
        "state_digests": dict(state_digests),
        "actor_state_digest_equal": state_digest_equal,
        "actor_state_tensors_exact": tensors_exact,
        "paired_snapshot_count": len(pairs),
        "max_abs_logit_delta": max_logit,
        "max_abs_probability_delta": max_probability,
        "selection_mismatch_count": selection_mismatch,
        "passed": passed,
        "rows": pairs,
    }


def exploration_deadlock(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(cell_rows)
    require(rows, BINDING_BLOCK, "empty_deadlock_cell")
    feasible = sum(bool(row["feasible_candidate_exists"]) for row in rows)
    no_assign = sum(bool(row["selected_action_type"] == "NO_ASSIGN") for row in rows)
    candidate_execution = sum(bool(row["selected_action_type"] == "CANDIDATE") for row in rows)
    ancestry = sum(bool(row["reward_ancestry_supported"]) for row in rows)
    eligible = sum(bool(row["actor_eligible_e1"]) for row in rows)
    if feasible == len(rows) and no_assign == len(rows) and candidate_execution == 0 and ancestry == 0 and eligible == 0:
        classification = "INITIAL_POLICY_EXPLORATION_DEADLOCK_CONFIRMED"
    elif candidate_execution > 0 and ancestry == 0:
        classification = "CREDIT_GENERATION_LIMITATION"
    else:
        classification = "MIXED_OR_INDETERMINATE"
    return {
        "training_decisions": len(rows),
        "feasible_candidate_decisions": feasible,
        "no_assign_selections": no_assign,
        "candidate_plan_executions": candidate_execution,
        "reward_ancestry_rows": ancestry,
        "actor_eligible_rows": eligible,
        "classification": classification,
        "chain": "feasible candidate -> NO_ASSIGN selection -> candidate plan not executed -> reward ancestry -> E1 Actor eligibility -> Actor learning",
    }


def order_audit(*, actors: Mapping[str, torch.nn.Module], snapshots: Sequence[Mapping[str, Any]], R1: Any, H: Any,
                TIE: Any, device: torch.device) -> tuple[dict[str, Any], list[str], int]:
    rows: list[dict[str, Any]] = []
    hard: list[str] = []
    forwards = 0
    for policy_name, actor in actors.items():
        digest_before = module_digest(actor)
        for payload in snapshots:
            base = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE); forwards += 1
            candidate_payload = R1.candidate_permutation(payload)
            agent_payload = R1.agent_permutation(payload)
            combined_payload = R1.candidate_permutation(agent_payload)
            candidate = R1.frozen_forward(actor, candidate_payload, device=device, head=H, tie=TIE); forwards += 1
            agent = R1.frozen_forward(actor, agent_payload, device=device, head=H, tie=TIE); forwards += 1
            combined = R1.frozen_forward(actor, combined_payload, device=device, head=H, tie=TIE); forwards += 1
            base_logits, base_probs = R1.scores_by_identity(base, payload)

            def compare(other: Mapping[str, Any], other_payload: Mapping[str, Any]) -> tuple[float, float, float, float]:
                logits, probs = R1.scores_by_identity(other, other_payload)
                require(set(logits) == set(base_logits) and set(probs) == set(base_probs), STRUCTURAL_BLOCK,
                        "order_identity_mapping_incomplete")
                return (
                    max((abs(base_logits[key] - logits[key]) for key in base_logits), default=0.0),
                    max((abs(base_probs[key] - probs[key]) for key in base_probs), default=0.0),
                    abs(float(base["no_assign_logit"][0, 0]) - float(other["no_assign_logit"][0, 0])),
                    abs(float(base["probabilities"][0, -1]) - float(other["probabilities"][0, -1])),
                )

            candidate_delta = compare(candidate, candidate_payload)
            agent_delta = compare(agent, agent_payload)
            combined_delta = compare(combined, combined_payload)
            rows.append({
                "policy": policy_name,
                "snapshot_digest": str(payload["snapshot_digest"]),
                "multi_candidate": int(base["support_size"]) >= 2,
                "candidate_identity_equal": action_identity(base["selection"]) == action_identity(candidate["selection"]),
                "agent_identity_equal": action_identity(base["selection"]) == action_identity(agent["selection"]),
                "combined_identity_equal": action_identity(base["selection"]) == action_identity(combined["selection"]),
                "candidate_pair_logit_delta": candidate_delta[0], "candidate_pair_probability_delta": candidate_delta[1],
                "candidate_no_assign_logit_delta": candidate_delta[2], "candidate_no_assign_probability_delta": candidate_delta[3],
                "agent_pair_logit_delta": agent_delta[0], "agent_pair_probability_delta": agent_delta[1],
                "agent_no_assign_logit_delta": agent_delta[2], "agent_no_assign_probability_delta": agent_delta[3],
                "combined_pair_logit_delta": combined_delta[0], "combined_pair_probability_delta": combined_delta[1],
                "combined_no_assign_logit_delta": combined_delta[2], "combined_no_assign_probability_delta": combined_delta[3],
            })
        if module_digest(actor) != digest_before:
            hard.append("ACTOR_PARAMETER_MUTATION_DURING_ORDER_AUDIT")

    def summary(prefix: str, scope: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "tested": len(scope),
            "identity_changes": sum(not bool(row[f"{prefix}_identity_equal"]) for row in scope),
            "max_identity_aligned_pair_logit_delta": max((float(row[f"{prefix}_pair_logit_delta"]) for row in scope), default=0.0),
            "max_identity_aligned_pair_probability_delta": max((float(row[f"{prefix}_pair_probability_delta"]) for row in scope), default=0.0),
            "max_no_assign_logit_delta": max((float(row[f"{prefix}_no_assign_logit_delta"]) for row in scope), default=0.0),
            "max_no_assign_probability_delta": max((float(row[f"{prefix}_no_assign_probability_delta"]) for row in scope), default=0.0),
        }

    multi = [row for row in rows if row["multi_candidate"]]
    candidate, agent, combined = summary("candidate", multi), summary("agent", rows), summary("combined", multi)
    if candidate["identity_changes"] or agent["identity_changes"] or combined["identity_changes"]:
        hard.append("ORDER_INVARIANCE_FAILURE")
    return {
        "selector": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "tolerance": 0.0},
        "candidate_order": candidate,
        "agent_order": agent,
        "combined_order": combined,
        "candidate_order_all_passed": candidate["identity_changes"] == 0,
        "agent_order_all_passed": agent["identity_changes"] == 0,
        "combined_order_all_passed": combined["identity_changes"] == 0,
        "numerical_rule": "selected identities must be exact; all observed identity-aligned numerical deltas are recorded without a tolerance.",
        "rows": rows,
    }, hard, forwards


def review_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    def mean(name: str) -> float | None:
        items = [float(row[name]) for row in values if row[name] is not None]
        return statistics.mean(items) if items else None
    return {
        "rows": len(values),
        "assign_count": sum(row["selected_action_family"] == "CANDIDATE" for row in values),
        "no_assign_count": sum(row["selected_action_family"] == "NO_ASSIGN" for row in values),
        "exact_tie_count": sum(bool(row["exact_tie"]) for row in values),
        "mean_no_assign_minus_best_pair": mean("no_assign_minus_best_pair"),
        "mean_candidate_only_margin": mean("candidate_only_top1_top2_margin"),
        "mean_entropy": mean("entropy"),
    }


def write_block(*, root: Path, source: Mapping[str, Any], preflight: Mapping[str, Any], reason: str) -> None:
    zero = counters()
    outputs = {
        "bt8r14_evidence_preflight.json": preflight,
        "bt8r14_cross_replay_matrix.json": {"not_run": True},
        "bt8r14_ac_environment_effect.json": {"not_run": True},
        "bt8r14_bd_immutability.json": {"not_run": True},
        "bt8r14_exploration_deadlock_audit.json": {"not_run": True},
        "bt8r14_factor_summary.json": {"not_run": True},
        "bt8r14_order_invariance.json": {"not_run": True},
        "test_results.json": {"execution_counters": zero, "hard_failures": [reason], "warnings": [], "github_push_performed": False},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": reason, "classification": "BLOCKED", "source_commit": source["source_commit"],
                               "hard_failures": [reason], "warnings": [], "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(
        f"# BT8-R14 blocked\n\n- gate: `{reason}`\n- source commit: `{source['source_commit']}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": reason, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(reason + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1
    import run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection as R12MOD

    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), BINDING_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    preflight: dict[str, Any] = {
        "source": source,
        "required_sources": {"r13": R13_SOURCE, "r12": R12_SOURCE, "r11": R11_SOURCE},
        "s3_contract_sha256": S3_CONTRACT_SHA256,
        "t1_tolerance": 0.0,
        "pure_frozen_inference_only": True,
    }
    try:
        r13_gate, r12_gate, r11_gate = (load_json(R13 / "gate_decision.json"), load_json(R12 / "gate_decision.json"),
                                         load_json(R11 / "gate_decision.json"))
        r13_audit, r12_audit, r11_audit = manifest_audit(R13), manifest_audit(R12), manifest_audit(R11)
        require(r13_gate.get("gate") == R13_GATE and r13_gate.get("source_commit") == R13_SOURCE, BINDING_BLOCK, "r13_gate")
        require(r12_gate.get("gate") == R12_GATE and r12_gate.get("source_commit") == R12_SOURCE, BINDING_BLOCK, "r12_gate")
        require(r11_gate.get("gate") == R11_GATE and r11_gate.get("source_commit") == R11_SOURCE, BINDING_BLOCK, "r11_gate")
        require(r13_audit["all_match"] and r12_audit["all_match"] and r11_audit["all_match"], BINDING_BLOCK, "authority_manifest")

        r13_execution = load_json(R13 / "bt8r13_execution_manifest.json")
        r13_checkpoints = load_json(R13 / "bt8r13_checkpoint_manifest.json")
        r13_cells = load_json(R13 / "bt8r13_cell_training_audit.json")
        r13_eligibility = load_json(R13 / "bt8r13_credit_eligibility.json")
        r13_credit = load_json(R13 / "bt8r13_candidate_plan_credit.json")
        r13_review_binding = load_json(R13 / "bt8r13_review_binding.json")
        r13_frozen = load_json(R13 / "frozen_hash_before_after.json")
        r12_contract = load_json(R12 / "bt8r12_s3_cell_contract.json")
        r11_contract = load_json(R11 / "bt8r11_selected_seed_repair_contract.json")
        require(r13_execution.get("r11_s3_contract_sha256") == S3_CONTRACT_SHA256 and r12_contract.get("s3_contract_sha256") == S3_CONTRACT_SHA256
                and r11_contract.get("sha256") == S3_CONTRACT_SHA256, BINDING_BLOCK, "s3_contract_binding")
        require(r13_execution.get("execution_mode") == "M2_FOUR_CELL_FRESH_CAUSAL_ROLLOUT", BINDING_BLOCK, "r13_mode")
        require(r13_credit.get("identity_chain_all") is True and int(r13_credit.get("mismatches", -1)) == 0, BINDING_BLOCK, "r13_credit_chain")
        require(r13_frozen.get("all_unchanged") is True and r13_frozen.get("extra_all_unchanged") is True, BINDING_BLOCK, "r13_frozen")
        require(r13_execution.get("global_locks_after") == LOCKS, BINDING_BLOCK, "r13_locks")
        require(r13_review_binding.get("review_collection_digest") == REVIEW_COLLECTION_DIGEST and int(r13_review_binding.get("unique_snapshot_count", -1)) == 6,
                BINDING_BLOCK, "r13_review_binding")
        require(float(r13_review_binding.get("t1_tolerance", -1.0)) == 0.0 and r13_review_binding.get("t1_selector") == T1_CONTRACT,
                BINDING_BLOCK, "r13_t1")

        collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
        collection_for_digest = dict(collection); supplied_digest = collection_for_digest.pop("collection_digest", None)
        require(supplied_digest == FPS.canonical_sha256(collection_for_digest) == REVIEW_COLLECTION_DIGEST, BINDING_BLOCK, "review_collection_digest")
        entries = list(collection.get("entries", []))
        require(int(collection.get("snapshot_count", -1)) == len(entries) == 6 and len({entry.get("snapshot_digest") for entry in entries}) == 6,
                BINDING_BLOCK, "review_collection_cardinality")
        expected_review_digests = {str(entry["snapshot_digest"]) for entry in entries}
        review_environment_bindings = r13_review_binding.get("environment_bindings", {})
        bound_review_digests = {str(digest) for binding in review_environment_bindings.values() for digest in binding.get("snapshot_digests", [])}
        require(expected_review_digests == bound_review_digests, BINDING_BLOCK, "r13_review_input_set")

        snapshots: list[dict[str, Any]] = []
        snapshot_preflight: list[dict[str, Any]] = []
        config: Mapping[str, Any] | None = None
        for entry in entries:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / str(entry["relative_path"]))
            metadata, tensors = payload["metadata"], payload["tensors"]
            require(payload["snapshot_digest"] == entry["snapshot_digest"], BINDING_BLOCK, "snapshot_digest")
            require(metadata["actor_config"] == (config if config is not None else metadata["actor_config"]), BINDING_BLOCK, "actor_config")
            config = metadata["actor_config"]
            require(str(metadata["actor_config_sha256"]) == FPS.actor_config_sha256(config), BINDING_BLOCK, "actor_config_sha")
            require(metadata["candidate_ids"] == metadata["candidate_order"] and int(metadata["no_assign_index"]) == int(metadata["selectable_pair_count"]),
                    BINDING_BLOCK, "snapshot_candidate_order")
            support = int(metadata["selectable_pair_count"])
            require(support > 0 and bool(tensors["safe_mask"][0, :support].all().item()), STRUCTURAL_BLOCK, "review_safe_support")
            require(metadata.get("captured_device") == "mps:0", BINDING_BLOCK, "snapshot_device")
            snapshots.append(payload)
            snapshot_preflight.append({"snapshot_digest": str(payload["snapshot_digest"]), "window_id": str(metadata["window_id"]),
                                       "decision_id": str(metadata["decision_id"]), "environment_seed": int(metadata["seed"]),
                                       "candidate_support_digest": str(metadata["candidate_support_digest"]), "support_size": support})
        require(config is not None and config.get("actor_head_id") == H.CANDIDATE_SENSITIVE_HEAD_ID
                and config.get("actor_head_version") == H.CANDIDATE_SENSITIVE_HEAD_VERSION, BINDING_BLOCK, "v2_actor_config")

        all_evidence: dict[str, dict[str, Any]] = {}
        raw_checkpoint_before: dict[str, str] = {}
        for checkpoint_kind in ("initial", "final"):
            for cell_id in CELL_EXPECTATIONS:
                manifest_entry = r13_checkpoints[checkpoint_kind].get(cell_id)
                require(isinstance(manifest_entry, Mapping), BINDING_BLOCK, f"checkpoint_manifest={checkpoint_kind}:{cell_id}")
                checkpoint_path = R13 / str(manifest_entry["path"])
                expected_digest_key = "actor_initial_digest" if checkpoint_kind == "initial" else "actor_final_digest"
                require(checkpoint_path.is_file() and sha256(checkpoint_path) == manifest_entry["sha256"], BINDING_BLOCK,
                        f"checkpoint_sha={checkpoint_kind}:{cell_id}")
                require(manifest_entry.get("strict_load") is True, BINDING_BLOCK, f"strict_load={checkpoint_kind}:{cell_id}")
                stored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                require(isinstance(stored, Mapping) and isinstance(stored.get("actor"), Mapping), BINDING_BLOCK,
                        f"checkpoint_schema={checkpoint_kind}:{cell_id}")
                evidence_key = f"{checkpoint_kind}:{cell_id}"
                all_evidence[evidence_key] = {"path": checkpoint_path, "sha256": str(manifest_entry["sha256"]),
                                              "cell_id": cell_id, "actor_digest": str(manifest_entry[expected_digest_key]),
                                              "actor": stored["actor"], "strict_load": True}
                raw_checkpoint_before[evidence_key] = sha256(checkpoint_path)

        require(state_dicts_exact(all_evidence["initial:AC-R1"]["actor"], all_evidence["initial:AC-R2"]["actor"]), BINDING_BLOCK,
                "ac_initial_pair_binding")
        require(state_dicts_exact(all_evidence["final:AC-R1"]["actor"], all_evidence["final:AC-R2"]["actor"]), BINDING_BLOCK,
                "ac_final_pair_binding")
        require(state_dicts_exact(all_evidence["initial:BD-R1"]["actor"], all_evidence["initial:BD-R2"]["actor"]), BINDING_BLOCK,
                "bd_initial_pair_binding")
        require(state_dicts_exact(all_evidence["initial:BD-R1"]["actor"], all_evidence["final:BD-R1"]["actor"]), BINDING_BLOCK,
                "bd_r1_state_binding")
        require(state_dicts_exact(all_evidence["initial:BD-R1"]["actor"], all_evidence["final:BD-R2"]["actor"]), BINDING_BLOCK,
                "bd_r2_state_binding")

        checkpoints: dict[str, dict[str, Any]] = {}
        raw_actor_states: dict[str, Mapping[str, torch.Tensor]] = {}
        for policy_name, binding in POLICIES.items():
            evidence_key = f"{binding['checkpoint_kind']}:{binding['cell_id']}"
            evidence = all_evidence[evidence_key]
            checkpoints[policy_name] = {"path": evidence["path"], "sha256": evidence["sha256"], "cell_id": evidence["cell_id"],
                                        "actor_digest": evidence["actor_digest"], "strict_load": True}
            raw_actor_states[policy_name] = evidence["actor"]

        cells_binding: dict[str, Any] = {}
        for cell_id, expected in CELL_EXPECTATIONS.items():
            cell = r13_cells.get(cell_id, {})
            eligibility = cell.get("eligibility", {})
            final = r13_checkpoints["final"].get(cell_id, {})
            require(int(eligibility.get("actor_eligible_semantic_rows", -1)) == int(expected["eligible"]), BINDING_BLOCK, f"eligible={cell_id}")
            require(int(final.get("actor_optimizer_steps", -1)) == int(expected["actor_steps"]), BINDING_BLOCK, f"actor_steps={cell_id}")
            require(str(cell.get("binding", {}).get("environment_replicate")) == str(expected["environment"]), BINDING_BLOCK, f"cell_environment={cell_id}")
            cells_binding[cell_id] = {"eligible": int(expected["eligible"]), "actor_steps": int(expected["actor_steps"]),
                                      "environment_replicate": expected["environment"], "actor_status": eligibility.get("actor_status")}

        current_frozen = R12MOD.frozen_hashes()
        expected_frozen = r13_frozen.get("after")
        expected_extra = r13_frozen.get("extra_after")
        current_extra = {
            "candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
            "e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
            "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py"),
        }
        require(current_frozen == expected_frozen and current_extra == expected_extra, BINDING_BLOCK, "frozen_hash_binding")
        require(TIE.TIE_BREAK_CONTRACT_ID == T1_CONTRACT, BINDING_BLOCK, "t1_contract")
        require(source["source_lineage_descends_from_r13"] and source["source_only_local_commit"], BINDING_BLOCK, "source_scope")
        preflight |= {
            "r13_manifest": r13_audit, "r12_manifest": r12_audit, "r11_manifest": r11_audit,
            "authority_gates": {"r13": True, "r12": True, "r11": True},
            "r13_execution_mode": r13_execution["execution_mode"], "review_collection_digest": supplied_digest,
            "review_snapshot_count": len(snapshots), "review_snapshot_unique": len({payload["snapshot_digest"] for payload in snapshots}),
            "review_snapshot_rows": snapshot_preflight, "actor_config": dict(config), "actor_config_sha256": FPS.actor_config_sha256(config),
            "t1_selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"), "t1_contract": True,
            "cell_eligibility_and_steps": cells_binding, "checkpoint_before": raw_checkpoint_before,
            "frozen_hash_binding": True, "r13_credit_identity_chain": True,
        }
    except R14Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, reason=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}"); return
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, reason=BINDING_BLOCK)
        print(f"[BLOCKED] {BINDING_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root=root, source=source, preflight=preflight | {"mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()}, reason=MPS_BLOCK)
        print(f"[BLOCKED] {MPS_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return

    device = torch.device("mps:0")
    execution = counters()
    hard: list[str] = []
    try:
        actors: dict[str, torch.nn.Module] = {}
        for policy_name, checkpoint in checkpoints.items():
            stored = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
            actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
                global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=int(config["agent_dim"]),
                candidate_dim=int(config["candidate_dim"]), hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
            actor.load_state_dict(stored["actor"], strict=True)
            actor.eval()
            require(module_digest(actor) == checkpoint["actor_digest"], BINDING_BLOCK, f"strict_actor_load={policy_name}")
            actors[policy_name] = actor

        actor_before = {name: module_digest(actor) for name, actor in actors.items()}
        matrix: list[dict[str, Any]] = []
        records: dict[str, list[dict[str, Any]]] = {name: [] for name in POLICIES}
        for policy_name, actor in actors.items():
            for payload in snapshots:
                row = actor_record(actor=actor, policy_name=policy_name, checkpoint=checkpoints[policy_name], payload=payload,
                                   R1=R1, H=H, TIE=TIE, device=device)
                records[policy_name].append(row); matrix.append(row)
                execution["frozen_policy_rows"] += 1
                execution["nan_or_inf"] += int(not row["finite"])
                execution["illegal_or_masked_selection"] += int(not row["legal_selection"])
                execution["zero_loss_fail_support_or_selection"] += int(row["zero_loss_fail_candidate_in_support"])
        torch.mps.synchronize()
        actor_after = {name: module_digest(actor) for name, actor in actors.items()}
        execution["parameter_mutation"] = sum(actor_before[name] != actor_after[name] for name in actors)

        bd_tensors_exact = state_dicts_exact(raw_actor_states["initial_BD"], raw_actor_states["final_BD_R1"]) and state_dicts_exact(raw_actor_states["initial_BD"], raw_actor_states["final_BD_R2"])
        bd = bd_immutability(initial=records["initial_BD"], final_r1=records["final_BD_R1"], final_r2=records["final_BD_R2"],
                             state_digests={name: checkpoints[name]["actor_digest"] for name in ("initial_BD", "final_BD_R1", "final_BD_R2")},
                             tensors_exact=bd_tensors_exact)
        if not bd["passed"]:
            hard.append(BD_IMMUTABILITY_BLOCK)

        ac_pairs = [compare_same_snapshot(rows_by_snapshot(records["final_AC_R1"])[digest], rows_by_snapshot(records["final_AC_R2"])[digest])
                    for digest in sorted(rows_by_snapshot(records["final_AC_R1"]))]
        ac_tensors_exact = state_dicts_exact(raw_actor_states["final_AC_R1"], raw_actor_states["final_AC_R2"])
        ac = environment_effect(ac_pairs, ac_tensors_exact)
        ac |= {"rows": ac_pairs, "initial_actor_digest": checkpoints["initial_AC"]["actor_digest"],
               "final_actor_digest_ac_r1": checkpoints["final_AC_R1"]["actor_digest"],
               "final_actor_digest_ac_r2": checkpoints["final_AC_R2"]["actor_digest"]}

        initial_factor_rows = [compare_same_snapshot(rows_by_snapshot(records["initial_AC"])[digest], rows_by_snapshot(records["initial_BD"])[digest])
                               for digest in sorted(rows_by_snapshot(records["initial_AC"]))]
        initial_factor = {
            "paired_rows": len(initial_factor_rows),
            "action_family_agreement_count": sum(not row["action_family_mismatch"] for row in initial_factor_rows),
            "action_family_mismatch_count": sum(bool(row["action_family_mismatch"]) for row in initial_factor_rows),
            "candidate_identity_comparison": "permitted only because every pair uses the same exact persisted snapshot; no cross-support identity statement is made",
            "rows": initial_factor_rows,
        }

        support_by_training_snapshot: dict[str, int] = {}
        train_collection = load_json(R13 / "bt8r13_training_snapshots" / "collection_manifest.json")
        train_entries = list(train_collection.get("entries", []))
        require(len(train_entries) == 96 and len({entry.get("snapshot_digest") for entry in train_entries}) == 96, BINDING_BLOCK, "training_snapshot_collection")
        for entry in train_entries:
            payload = FPS.load_snapshot(R13 / "bt8r13_training_snapshots" / str(entry["relative_path"]))
            require(payload["snapshot_digest"] == entry["snapshot_digest"], BINDING_BLOCK, "training_snapshot_digest")
            support_by_training_snapshot[str(payload["snapshot_digest"])] = int(payload["metadata"]["selectable_pair_count"])
        credit_rows = list(r13_credit.get("rows", []))
        eligibility_by_cell = r13_eligibility.get("cells", {})
        cell_deadlock_rows: dict[str, list[dict[str, Any]]] = {cell_id: [] for cell_id in CELL_EXPECTATIONS}
        for row in credit_rows:
            cell_id, snapshot_digest = str(row["cell_id"]), str(row["snapshot_digest"])
            require(cell_id in cell_deadlock_rows and snapshot_digest in support_by_training_snapshot, BINDING_BLOCK, "credit_snapshot_binding")
            category = str(row["category"])
            require(category != "INSUFFICIENT_IDENTITY_CREDIT", BINDING_BLOCK, "insufficient_identity_credit")
            cell_deadlock_rows[cell_id].append({
                "decision_id": str(row["decision_id"]), "snapshot_digest": snapshot_digest,
                "feasible_candidate_exists": support_by_training_snapshot[snapshot_digest] > 0,
                "selected_action_type": str(row["selected_action_type"]),
                "reward_ancestry_supported": category in {"DIRECT_CANDIDATE_REWARD", "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD", "NO_ASSIGN_REWARD_SUPPORTED"},
                "actor_eligible_e1": bool(row["actor_eligible_e1"]), "category": category,
            })
        deadlock_cells: dict[str, Any] = {}
        for cell_id, rows in cell_deadlock_rows.items():
            require(len(rows) == 24, BINDING_BLOCK, f"deadlock_row_count={cell_id}")
            report = exploration_deadlock(rows)
            expected_eligible = int(eligibility_by_cell[cell_id]["actor_eligible_semantic_rows"])
            require(report["actor_eligible_rows"] == expected_eligible, BINDING_BLOCK, f"deadlock_eligibility={cell_id}")
            report |= {"actor_steps": CELL_EXPECTATIONS[cell_id]["actor_steps"], "cell_id": cell_id}
            deadlock_cells[cell_id] = report
        bd_deadlock = all(deadlock_cells[cell]["classification"] == "INITIAL_POLICY_EXPLORATION_DEADLOCK_CONFIRMED" for cell in ("BD-R1", "BD-R2"))
        deadlock = {"cells": deadlock_cells, "bd_chain_confirmed": bd_deadlock,
                    "classification": "INITIAL_POLICY_EXPLORATION_DEADLOCK_CONFIRMED" if bd_deadlock else "MIXED_OR_INDETERMINATE",
                    "credit_generation_limitation": False,
                    "source": "R13 persisted training snapshot support plus selected/applied/credited credit rows; no candidate regeneration or rollout"}

        order, order_hard, order_forwards = order_audit(actors=actors, snapshots=snapshots, R1=R1, H=H, TIE=TIE, device=device)
        execution["order_probe_forwards"] = order_forwards
        hard.extend(order_hard)
        after_frozen = R12MOD.frozen_hashes()
        after_extra = {"candidate_plan_bridge": sha256(ROOT / "joint_candidate_plan_causal_bridge.py"),
                       "e1_eligibility": sha256(ROOT / "joint_assignment_e1_eligibility.py"),
                       "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py")}
        raw_checkpoint_after = {name: sha256(evidence["path"]) for name, evidence in all_evidence.items()}
        execution["checkpoint_mutation"] = sum(raw_checkpoint_before[name] != raw_checkpoint_after[name] for name in raw_checkpoint_before)
        forbidden = ("training", "optimizer_step", "causal_rollout", "simulator_execution", "candidate_generation", "candidate_regeneration",
                     "local_search_rerun", "zero_loss_reevaluation", "parameter_mutation", "checkpoint_write", "checkpoint_mutation",
                     "review_optimizer_rows", "future_leakage", "nan_or_inf", "illegal_or_masked_selection",
                     "zero_loss_fail_support_or_selection", "test6_access", "github_push")
        if any(execution[name] != 0 for name in forbidden):
            hard.append(STRUCTURAL_BLOCK)
        if after_frozen != current_frozen or after_extra != current_extra:
            hard.append("FROZEN_HASH_MUTATION")
        if ac["classification"] == "ENVIRONMENT_EFFECT_LOW" and bd_deadlock:
            classification = "A_AC_LEARNING_ENVIRONMENT_ROBUST_BD_EXPLORATION_DEADLOCK_CONFIRMED"
        elif ac["classification"] == "ENVIRONMENT_EFFECT_MATERIAL" and bd_deadlock:
            classification = "B_AC_LEARNING_ENVIRONMENT_SENSITIVE_BD_EXPLORATION_DEADLOCK_CONFIRMED"
        elif deadlock["classification"] == "CREDIT_GENERATION_LIMITATION":
            classification = "B_CREDIT_GENERATION_LIMITATION_CONFIRMED"
        else:
            classification = "C_FACTORIAL_REVIEW_INCOMPLETE_OR_INDETERMINATE"
        gate = PASS_GATE if not hard else hard[0]
        if hard:
            classification = "BLOCKED"
        outputs = {
            "bt8r14_evidence_preflight.json": preflight | {"mps_built": True, "mps_available": True, "device": "mps:0",
                                                              "actor_checkpoint_state_sha256": {name: value["actor_digest"] for name, value in checkpoints.items()}},
            "bt8r14_cross_replay_matrix.json": {"policy_count": len(actors), "review_snapshot_count": len(snapshots), "rows": matrix,
                                                  "per_policy_summary": {name: review_summary(rows) for name, rows in records.items()},
                                                  "candidate_identity_cross_support_rule": "not compared across different snapshot digests/supports"},
            "bt8r14_ac_environment_effect.json": ac,
            "bt8r14_bd_immutability.json": bd,
            "bt8r14_exploration_deadlock_audit.json": deadlock,
            "bt8r14_factor_summary.json": {"actor_initialization_effect": initial_factor, "ac_environment_effect": ac["classification"],
                                              "bd_initial_no_assign_persistence": bd["passed"], "exploration_deadlock": deadlock["classification"],
                                              "no_performance_or_kpi_interpretation": True},
            "bt8r14_order_invariance.json": order,
            "test_results.json": {"execution_counters": execution, "hard_failures": hard, "warnings": [], "github_push_performed": False,
                                  "performance_interpretation_performed": False},
            "frozen_hash_before_after.json": {"before": current_frozen, "after": after_frozen, "all_unchanged": current_frozen == after_frozen,
                                                "extra_before": current_extra, "extra_after": after_extra, "extra_all_unchanged": current_extra == after_extra,
                                                "checkpoint_before": raw_checkpoint_before, "checkpoint_after": raw_checkpoint_after,
                                                "checkpoint_unchanged": raw_checkpoint_before == raw_checkpoint_after},
            "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
                                   "next_step": "minimal exploration-deadlock repair-selection audit; no automatic training authorization" if not hard else "STOP"},
        }
        for name, value in outputs.items():
            dump(root / name, value)
        (root / "final_report.md").write_text(
            f"# BT8-R14 final report\n\n- gate: `{gate}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n\n"
            "This artifact is pure frozen inference on six preserved review snapshots. It makes no KPI, policy-quality, performance, or causal-performance claim.\n",
            encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                       "file_sha256": manifest, "github_push_performed": False})
        (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
        print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
        print(f"classification: {classification}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except R14Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, reason=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, reason=STRUCTURAL_BLOCK)
        print(f"[BLOCKED] {STRUCTURAL_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
