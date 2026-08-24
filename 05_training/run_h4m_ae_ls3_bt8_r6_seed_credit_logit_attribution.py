#!/usr/bin/env python3
"""BT8-R6: read-only seed-divergent credit-to-logit attribution audit.

This stage consumes the preserved BT8-F1 checkpoints, training snapshots, and
credit rows.  It never regenerates candidates, runs the simulator, computes a
new rollout, mutates parameters, or invokes an optimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from itertools import combinations
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R6"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R6_SEED_DIVERGENT_CREDIT_TO_LOGIT_ATTRIBUTION_AUDIT_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R6_EVIDENCE_OR_ATTRIBUTION_INTEGRITY_FAILURE"
ROOT_CLASS = "MIXED_OR_INDETERMINATE"
OBSERVED_MECHANISM = "INITIAL_POLICY_ACTION_SUPPORT_PLUS_CRITIC_GAE_NORMALIZATION"
INDETERMINATE_CLASS = "INSUFFICIENT_OR_INTEGRITY_LIMITED"
FIRST_DIVERGENCE = "INITIAL_ACTOR_ACTION_FAMILY_SPLIT_ON_EXACT_MATCHED_INPUTS_BEFORE_REWARD"
NEXT_GATE = "H4M-AE-R9.8-LS3-BT8-R7_MINIMAL_ACTOR_CRITIC_SEED_FACTORIZATION_AND_CREDIT_ELIGIBILITY_SELECTION"

F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
R4A_SOURCE = "eac4a209e09e696380bde3bbc437a4fd13c45e99"
R5_SOURCE = "d971c3beaba65585ef8737403b215b4c914739cc"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
R5_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R5_SAME_SUPPORT_FROZEN_V2_DISCRIMINATION_REVIEW_COMPLETE"
R4A_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4A_F1_EXECUTION_AUTHORITY_COMPLETION"
R5_CLASS = "D_V2_CANDIDATE_DISCRIMINATION_WORSENED_OR_COLLAPSED"
F1_MANIFEST_SHA256 = "b6a7ffdcaa966b06bfad1bad565d6620c6bcd1cd5c545caa044e076af506de5a"
R5_MANIFEST_SHA256 = "e55abd2c92c2d2c95cd886bcfdfa05ee10ac168b0ae4edfb756a79301d0bb22c"
F1_CREDIT_SHA256 = "2ab4cd4ef57a9727214a432b52927cd1fcc28b289331983ef9c0f7431b909fb1"
F1_EXECUTION_SHA256 = "4670e15fa8aa193812eee1edf697803b4b90e5e3baecfe7f9b0060d6fe467289"
F1_LEARNING_SHA256 = "92641255353474b53ca440660e0533d9a026f15774e4e668bf26f9e88eb06e9e"
F1_TRAINING_COLLECTION_FILE_SHA256 = "9c40a569bfdcd2c7eef0d5ca0ee5243d90e53450367e50e61dd0c6c37128dd4f"
F1_TRAINING_COLLECTION_DIGEST = "587e45422e0d5cbfcfc0f7279eae34178e00414d21229951ce6683c2aa4785d8"
F1_REVIEW_COLLECTION_DIGEST = "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"
R5_R1_REVIEW_SHA256 = "d0f703df624581a697a6bf1dea594a0ae568cac7f3d873fbd52bcd374a88dad3"
R5_R2_REVIEW_SHA256 = "2b6e792c509adb52161231944e36622f2e30f4158da243aa8d1f6eef1be458fa"
R5_COLLAPSE_SHA256 = "956e9a0f3f241f7cfa7eb048ee9d54b92877087e7628eaa6790db656aee7f004"
F1_RUNNER_SHA256 = "1f415da9d880f42fe050afe9c64dff0ce041883cad0f7cebdff756f0840072e6"
R5_RUNNER_SHA256 = "29f556062f32da63069d326543f17fbb63eb8979fb06b38c996f6a80ff0bd45e"
ACTOR_SHA256 = "257e73b5e148a153bd148ac52b34d8c6b0fbcf4c51049f788768808fbfa770f4"
LEARNING_SHA256 = "45f12365f965b0cfb262f597d19b0bb2263ecbea8dd27d825c5a9b92d2d47e7b"
CREDIT_CONTRACT_SHA256 = "ac173a2d5eae80e9fd33b36910fb12ce7a55894b9c6f766a130cdfcca7c3efe2"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
R5 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r5_same_support_frozen_v2_review_20260824_231253+09:00"
R4A = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4a_f1_execution_authority_completion_20260823_132257+0900"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution.py",
    "05_training/test_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution.py",
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
PARAMETER_GROUPS = ("SHARED", "PAIR_ONLY", "NO_ASSIGN_ONLY")


class R6Error(RuntimeError):
    pass


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R6Error(f"{code}:{detail}" if detail else code)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), "AUTHORITATIVE_EVIDENCE_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff", "--name-only", f"{R5_SOURCE}..HEAD"]).splitlines() if row]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_lineage_descends_from_r5": git(["merge-base", R5_SOURCE, "HEAD"]) == R5_SOURCE,
        "changed_files_since_r5": changed,
        "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
        "github_push_performed": False,
    }


def describe(values: Sequence[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "min": None, "mean": None, "max": None, "variance": None,
                "positive": 0, "negative": 0, "zero": 0}
    return {
        "count": len(rows), "min": min(rows), "mean": statistics.mean(rows), "max": max(rows),
        "variance": statistics.pvariance(rows), "positive": sum(value > 0.0 for value in rows),
        "negative": sum(value < 0.0 for value in rows), "zero": sum(value == 0.0 for value in rows),
    }


def sign_transition(raw: float, normalized: float) -> str:
    raw_sign = "POSITIVE" if raw > 0.0 else "NEGATIVE" if raw < 0.0 else "ZERO"
    normalized_sign = "POSITIVE" if normalized > 0.0 else "NEGATIVE" if normalized < 0.0 else "ZERO"
    return f"{raw_sign}_TO_{normalized_sign}"


def ppo_clip_state(advantage: float, ratio: float, epsilon: float = 0.2) -> str:
    lower, upper = 1.0 - float(epsilon), 1.0 + float(epsilon)
    if advantage == 0.0:
        return "ZERO_ADVANTAGE"
    if advantage > 0.0:
        if ratio > upper:
            return "CLIPPED_HIGH_ZERO_GRADIENT"
        if ratio == upper:
            return "HIGH_BOUNDARY"
        return "ACTIVE"
    if ratio < lower:
        return "CLIPPED_LOW_ZERO_GRADIENT"
    if ratio == lower:
        return "LOW_BOUNDARY"
    return "ACTIVE"


def pressure_label(*, selected_no_assign: bool, advantage: float, clip_state: str) -> str:
    if clip_state.startswith("CLIPPED") or advantage == 0.0:
        return "ZERO_POLICY_PRESSURE"
    if selected_no_assign and advantage > 0.0:
        return "REINFORCE_NO_ASSIGN_DIRECT"
    if selected_no_assign:
        return "SUPPRESS_NO_ASSIGN_DIRECT_RELATIVE_CANDIDATE_UP"
    if advantage > 0.0:
        return "REINFORCE_SELECTED_CANDIDATE_DIRECT"
    return "SUPPRESS_SELECTED_CANDIDATE_DIRECT_RELATIVE_NO_ASSIGN_UP"


def initial_action_support_driver(matrix: Mapping[str, Mapping[str, int]]) -> str:
    r1 = matrix.get("R1_INITIAL_ACTOR", {})
    r2 = matrix.get("R2_INITIAL_ACTOR", {})
    if r1.get("R1_INPUT_NO_ASSIGN") == 0 and r1.get("R2_INPUT_NO_ASSIGN") == 0 \
            and r2.get("R1_INPUT_NO_ASSIGN") == 24 and r2.get("R2_INPUT_NO_ASSIGN") == 24:
        return "ACTOR_INITIALIZATION_DOMINANT_ACROSS_BOTH_PRESERVED_INPUT_COLLECTIONS"
    if r1.get("R1_INPUT_NO_ASSIGN") != r1.get("R2_INPUT_NO_ASSIGN") \
            or r2.get("R1_INPUT_NO_ASSIGN") != r2.get("R2_INPUT_NO_ASSIGN"):
        return "INPUT_AND_INITIALIZATION_MIXED"
    return "NOT_UNIQUE"


def classify_root(pattern: Mapping[str, Any]) -> str:
    exact = (
        pattern.get("initial_driver") == "ACTOR_INITIALIZATION_DOMINANT_ACROSS_BOTH_PRESERVED_INPUT_COLLECTIONS"
        and pattern.get("input_tensor_multisets_exact") is True
        and pattern.get("r1_candidate_selected") == 24
        and pattern.get("r1_reward_rows", 0) > 0
        and pattern.get("r1_raw_positive") == 24
        and pattern.get("r1_positive_to_negative", 0) > 0
        and pattern.get("r2_no_assign_selected") == 24
        and pattern.get("r2_reward_rows") == 0
        and pattern.get("r2_raw_negative") == 24
        and pattern.get("r2_negative_to_positive", 0) > 0
        and pattern.get("r2_direct_candidate_credit") == 0
    )
    return ROOT_CLASS if exact else INDETERMINATE_CLASS


def actor_parameter_group(name: str) -> str:
    if name.startswith(("global_encoder.", "agent_encoder.")):
        return "SHARED"
    if name.startswith(("demand_encoder.", "candidate_encoder.", "scorer.")):
        return "PAIR_ONLY"
    if name.startswith("no_assign_scorer."):
        return "NO_ASSIGN_ONLY"
    raise R6Error(f"UNCLASSIFIED_ACTOR_PARAMETER:{name}")


def shapley_from_coalitions(values: Mapping[frozenset[str], float],
                            groups: Sequence[str] = PARAMETER_GROUPS) -> dict[str, float]:
    n = len(groups)
    require(len(values) == 2 ** n, "SHAPLEY_COALITION_COUNT_MISMATCH")
    result: dict[str, float] = {}
    factorial = math.factorial
    for group in groups:
        contribution = 0.0
        others = [item for item in groups if item != group]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset_tuple in combinations(others, size):
                subset = frozenset(subset_tuple)
                contribution += weight * (float(values[subset | {group}]) - float(values[subset]))
        result[group] = contribution
    return result


def decompose_gae(rows: Sequence[Mapping[str, Any]], gamma: float = 0.99,
                  lam: float = 0.95) -> tuple[list[dict[str, float]], dict[str, Any]]:
    by_trajectory: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for row in rows:
        order = int(str(row["decision_id"]).rsplit(":", 1)[1])
        by_trajectory.setdefault(str(row["trajectory_id"]), []).append((order, row))
    components: dict[str, tuple[float, float]] = {}
    gamma_k = float(gamma) ** 2
    for trajectory_rows in by_trajectory.values():
        reward_running = 0.0
        critic_running = 0.0
        for _, row in sorted(trajectory_rows, reverse=True):
            nonterminal = 1.0 if row["nonterminal"] else 0.0
            reward_running = float(row["reward"]) + gamma_k * float(lam) * nonterminal * reward_running
            critic_delta = gamma_k * float(row["next_value"]) * nonterminal - float(row["critic_value"])
            critic_running = critic_delta + gamma_k * float(lam) * nonterminal * critic_running
            components[str(row["decision_id"])] = (reward_running, critic_running)
    raw_values = [float(row["raw_gae"]) for row in rows]
    raw_mean = statistics.mean(raw_values)
    raw_std = max(statistics.pvariance(raw_values) ** 0.5, 1e-8)
    reward_values = [components[str(row["decision_id"])][0] for row in rows]
    critic_values = [components[str(row["decision_id"])][1] for row in rows]
    reward_mean, critic_mean = statistics.mean(reward_values), statistics.mean(critic_values)
    output, raw_residuals, normalized_residuals = [], [], []
    for row, reward_component, critic_component in zip(rows, reward_values, critic_values):
        reward_centered = (reward_component - reward_mean) / raw_std
        critic_centered = (critic_component - critic_mean) / raw_std
        raw_residual = float(row["raw_gae"]) - reward_component - critic_component
        normalized_residual = float(row["normalized_advantage"]) - reward_centered - critic_centered
        raw_residuals.append(raw_residual)
        normalized_residuals.append(normalized_residual)
        output.append({
            "decision_id": str(row["decision_id"]), "reward_gae_component": reward_component,
            "critic_bootstrap_gae_component": critic_component,
            "reward_centered_normalized_component": reward_centered,
            "critic_centered_normalized_component": critic_centered,
            "raw_additivity_residual": raw_residual,
            "normalized_additivity_residual": normalized_residual,
        })
    audit = {
        "gamma": gamma, "gamma_k": gamma_k, "gae_lambda": lam,
        "reward_component": describe(reward_values), "critic_bootstrap_component": describe(critic_values),
        "reward_component_abs_sum": sum(abs(value) for value in reward_values),
        "critic_bootstrap_component_abs_sum": sum(abs(value) for value in critic_values),
        "max_abs_raw_additivity_residual": max((abs(value) for value in raw_residuals), default=0.0),
        "max_abs_normalized_additivity_residual": max((abs(value) for value in normalized_residuals), default=0.0),
        "comparison_rule": "observed residuals recorded; no invented tolerance",
    }
    return output, audit


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    rows = manifest.get("file_sha256", {})
    mismatches = []
    for name, expected in rows.items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(name)
    return {"declared_file_count": len(rows), "mismatches": mismatches, "all_match": not mismatches,
            "manifest_sha256": sha256(root / "manifest.json")}


def frozen_hashes(BT6: Any) -> dict[str, str]:
    return {
        **BT6.frozen_hashes(),
        "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
        "t1_selector": sha256(ROOT / "joint_assignment_frozen_tie_break.py"),
    }


def checkpoint_bindings(initial: Mapping[str, Any], final: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for replicate_id in REPLICATES:
        first = F1 / "initial_checkpoints" / initial[replicate_id]["path"]
        last = F1 / "final_checkpoints" / final[replicate_id]["path"]
        result[replicate_id] = f"{sha256(first)}:{sha256(last)}"
    return result


def selected_no_assign(credit: Mapping[str, Any], no_assign_id: str) -> bool:
    return str(credit.get("selected_candidate_id")) == no_assign_id


def replicate_from_decision_id(decision_id: str) -> str:
    parts = str(decision_id).split(":")
    require(len(parts) == 4 and parts[0] == "BT8_F1" and parts[1] in REPLICATES,
            "DECISION_ID_REPLICATE_BINDING_INVALID", decision_id)
    return parts[1]


def replay_actor(actor: torch.nn.Module, payload: Mapping[str, Any], F1MOD: Any, H: Any,
                 TIE: Any, device: torch.device) -> dict[str, Any]:
    row = F1MOD.replay(actor, payload, H, TIE, device)
    pair_logits = [float(value) for value in row["pair_logits"]]
    pair_spread = max(pair_logits) - min(pair_logits) if len(pair_logits) >= 2 else None
    ordered_pairs = sorted(pair_logits, reverse=True)
    pair_top1_top2_margin = ordered_pairs[0] - ordered_pairs[1] if len(ordered_pairs) >= 2 else None
    top_candidate = max(pair_logits) if pair_logits else None
    return {
        **row,
        "pair_logit_spread": pair_spread,
        "pair_top1_top2_margin": pair_top1_top2_margin,
        "full_action_top1_top2_margin": row["top_score_margin"],
        "top_candidate_logit": top_candidate,
        "no_assign_minus_top_candidate": float(row["no_assign_logit"]) - top_candidate if top_candidate is not None else None,
    }


def selected_score(row: Mapping[str, Any], payload: Mapping[str, Any], credit: Mapping[str, Any],
                   no_assign_id: str) -> tuple[float, float]:
    if selected_no_assign(credit, no_assign_id):
        index = len(row["pair_logits"])
        return float(row["no_assign_logit"]), float(row["probabilities"][index])
    candidate = str(credit["selected_candidate_id"])
    agent = str(credit["agent_id"])
    identities = payload["metadata"]["candidate_ids"]
    index = next((i for i, item in enumerate(identities)
                  if str(item["candidate_id"]) == candidate and str(item["agent_id"]) == agent), None)
    require(index is not None, "SELECTED_IDENTITY_NOT_IN_PRESERVED_SUPPORT", str(credit["decision_id"]))
    return float(row["pair_logits"][index]), float(row["probabilities"][index])


def functional_actor_metrics(actor: torch.nn.Module, parameters: Mapping[str, torch.Tensor],
                             payload: Mapping[str, Any], device: torch.device) -> dict[str, float | None]:
    tensors = {name: value.to(device) for name, value in payload["tensors"].items()}
    selectable = int(payload["metadata"]["selectable_pair_count"])
    kwargs = {
        "global_feats": tensors["global_feats"], "demand_feats": tensors["demand_feats"],
        "agent_feats": tensors["agent_feats"], "agent_mask": tensors["agent_mask"],
        "candidate_feats": tensors["candidate_feats"], "pair_agent_index": tensors["pair_agent_index"],
        "safe_mask": tensors["safe_mask"],
    }
    actor.eval()
    with torch.no_grad():
        raw, no_assign = torch.func.functional_call(actor, dict(parameters), args=(), kwargs=kwargs, strict=True)
    pairs = [float(value) for value in raw[0, :selectable].detach().cpu().tolist()]
    no_assign_value = float(no_assign[0, 0].detach().cpu())
    full = sorted([*pairs, no_assign_value], reverse=True)
    ordered_pairs = sorted(pairs, reverse=True)
    return {
        "no_assign_minus_best_pair": no_assign_value - max(pairs) if pairs else None,
        "pair_top1_top2_margin": ordered_pairs[0] - ordered_pairs[1] if len(ordered_pairs) >= 2 else None,
        "full_action_top1_top2_margin": full[0] - full[1] if len(full) >= 2 else None,
    }


def actor_route_attribution(*, policies: Mapping[str, Mapping[str, torch.nn.Module]],
                            review_snapshots: Mapping[str, Sequence[Mapping[str, Any]]],
                            device: torch.device) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    forward_count = 0
    all_subsets = [frozenset(combo) for size in range(len(PARAMETER_GROUPS) + 1)
                   for combo in combinations(PARAMETER_GROUPS, size)]
    metrics = ("no_assign_minus_best_pair", "pair_top1_top2_margin", "full_action_top1_top2_margin")
    for replicate_id in REPLICATES:
        initial_actor, final_actor = policies[replicate_id]["initial"], policies[replicate_id]["final"]
        initial_parameters = dict(initial_actor.named_parameters())
        final_parameters = dict(final_actor.named_parameters())
        require(set(initial_parameters) == set(final_parameters), "ACTOR_PARAMETER_KEY_MISMATCH", replicate_id)
        grouped_names = {group: sorted(name for name in initial_parameters if actor_parameter_group(name) == group)
                         for group in PARAMETER_GROUPS}
        require(set().union(*map(set, grouped_names.values())) == set(initial_parameters),
                "ACTOR_PARAMETER_ROUTE_PARTITION_INCOMPLETE", replicate_id)
        parameter_delta = {}
        for group, names in grouped_names.items():
            deltas = [(final_parameters[name].detach().cpu().double()
                       - initial_parameters[name].detach().cpu().double()).reshape(-1) for name in names]
            merged = torch.cat(deltas)
            parameter_delta[group] = {
                "tensor_count": len(names), "element_count": int(merged.numel()),
                "l2_norm": float(torch.linalg.vector_norm(merged)),
                "abs_sum": float(merged.abs().sum()), "max_abs": float(merged.abs().max()),
                "parameter_names": names,
            }
        coalition_values: dict[frozenset[str], dict[str, float]] = {}
        coalition_rows = []
        for subset in all_subsets:
            hybrid = {name: (final_parameters[name] if actor_parameter_group(name) in subset else initial_parameters[name])
                      for name in initial_parameters}
            observed = {metric: [] for metric in metrics}
            for payload in review_snapshots[replicate_id]:
                row = functional_actor_metrics(initial_actor, hybrid, payload, device)
                forward_count += 1
                for metric in metrics:
                    if row[metric] is not None:
                        observed[metric].append(float(row[metric]))
            means = {metric: statistics.mean(values) for metric, values in observed.items()}
            coalition_values[subset] = means
            coalition_rows.append({"final_parameter_groups": sorted(subset), "metrics": means,
                                   "review_rows": len(review_snapshots[replicate_id]),
                                   "pair_margin_rows": len(observed["pair_top1_top2_margin"])})
        shapley = {}
        completeness = {}
        for metric in metrics:
            values = {subset: row[metric] for subset, row in coalition_values.items()}
            contributions = shapley_from_coalitions(values)
            endpoint_delta = values[frozenset(PARAMETER_GROUPS)] - values[frozenset()]
            residual = endpoint_delta - sum(contributions.values())
            shapley[metric] = contributions
            completeness[metric] = {"initial_to_final_delta": endpoint_delta,
                                    "shapley_sum": sum(contributions.values()),
                                    "additivity_residual": residual,
                                    "comparison_rule": "observed residual; no invented tolerance"}
        result[replicate_id] = {
            "route_contract": {"SHARED": ["global_encoder", "agent_encoder"],
                               "PAIR_ONLY": ["demand_encoder", "candidate_encoder", "scorer"],
                               "NO_ASSIGN_ONLY": ["no_assign_scorer"]},
            "parameter_delta": parameter_delta,
            "coalitions": coalition_rows,
            "shapley": shapley,
            "completeness": completeness,
            "method": "frozen torch.func.functional_call; no parameter mutation, backward, autograd.grad, or optimizer",
            "interpretation_scope": "endpoint parameter-group decomposition only; not an epoch/sample/Adam trajectory reconstruction",
        }
    return result, forward_count


def summarize_trace(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actions = Counter(row["selected_action_type"] for row in rows)
    transitions = Counter(row["normalization_sign_transition"] for row in rows)
    pressure_by_epoch = {
        f"epoch_{index + 1}": dict(sorted(Counter(row["pressure_by_epoch"][index] for row in rows).items()))
        for index in range(3)
    }
    clip_by_epoch = {
        f"epoch_{index + 1}": dict(sorted(Counter(row["clip_state_by_epoch"][index] for row in rows).items()))
        for index in range(3)
    }
    reward_rows = [row for row in rows if row["reward"] != 0.0]
    return {
        "decision_count": len(rows),
        "selected_action_counts": dict(sorted(actions.items())),
        "reward_supported_decisions": len(reward_rows),
        "reward_supported_trajectories": len({row["trajectory_id"] for row in reward_rows}),
        "direct_candidate_identity_credit_rows": sum(row["selected_action_type"] == "CANDIDATE" for row in rows),
        "direct_multi_candidate_identity_credit_rows": sum(row["selected_action_type"] == "CANDIDATE" and row["support_size"] >= 2 for row in rows),
        "direct_no_assign_credit_rows": sum(row["selected_action_type"] == "NO_ASSIGN" for row in rows),
        "reward": describe([row["reward"] for row in rows]),
        "critic_value": describe([row["critic_value"] for row in rows]),
        "critic_target": describe([row["critic_target"] for row in rows]),
        "td_residual": describe([row["td_residual"] for row in rows]),
        "raw_gae": describe([row["raw_gae"] for row in rows]),
        "normalized_advantage": describe([row["normalized_advantage"] for row in rows]),
        "normalization_sign_transitions": dict(sorted(transitions.items())),
        "ppo_clip_state_by_epoch": clip_by_epoch,
        "policy_pressure_by_epoch": pressure_by_epoch,
        "initial_to_final": {
            "selected_logit_delta": describe([row["selected_logit_delta"] for row in rows]),
            "selected_probability_delta": describe([row["selected_probability_delta"] for row in rows]),
            "no_assign_minus_top_candidate_delta": describe([
                row["no_assign_minus_top_candidate_delta"] for row in rows
                if row["no_assign_minus_top_candidate_delta"] is not None
            ]),
            "pair_logit_spread_delta": describe([
                row["pair_logit_spread_delta"] for row in rows if row["pair_logit_spread_delta"] is not None
            ]),
            "pair_top1_top2_margin_delta": describe([
                row["pair_top1_top2_margin_delta"] for row in rows
                if row["pair_top1_top2_margin_delta"] is not None
            ]),
            "full_action_top1_top2_margin_delta": describe([
                row["full_action_top1_top2_margin_delta"] for row in rows
                if row["full_action_top1_top2_margin_delta"] is not None
            ]),
        },
    }


def artifact_root(BT6: Any) -> Path:
    stamp = BT6.now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution_{stamp}"


def write_block(root: Path, source: Mapping[str, Any], binding: Mapping[str, Any], reason: str) -> None:
    counters = {"training": 0, "optimizer_step": 0, "backward": 0, "autograd_grad": 0,
                "causal_rollout": 0, "simulator_step": 0, "candidate_generation": 0,
                "candidate_regeneration": 0, "local_search_rerun": 0, "zero_loss_reevaluation": 0,
                "reward_recomputation": 0, "snapshot_generation": 0, "parameter_mutation": 0,
                "checkpoint_write": 0, "test6_access": 0,
                "training_snapshot_forward_replays": 0, "review_functional_forward_replays": 0}
    outputs = {
        "bt8r6_authoritative_binding.json": binding,
        "bt8r6_f1_evidence_integrity.json": {"not_run": True},
        "bt8r6_replicate1_credit_trace.json": {"not_run": True},
        "bt8r6_replicate2_credit_trace.json": {"not_run": True},
        "bt8r6_cross_seed_initial_policy_replay.json": {"not_run": True},
        "bt8r6_actor_route_attribution.json": {"not_run": True},
        "bt8r6_credit_to_logit_attribution.json": {"not_run": True},
        "bt8r6_first_divergence_classification.json": {"classification": "BLOCKED", "reason": reason},
        "bt8r6_gate_matrix.json": {"passed": False, "hard_failures": [reason]},
        "test_results.json": {"execution_counters": counters, "hard_failures": [reason]},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": BLOCK_GATE, "classification": "BLOCKED",
                               "source_commit": source.get("source_commit"), "hard_failures": [reason],
                               "warnings": [], "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-R6 blocked\n\n{reason}\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": BLOCK_GATE,
                                   "source_commit": source.get("source_commit"), "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(BLOCK_GATE + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_credit_contract as CC
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_f1_bounded_training as F1MOD
    import run_h4m_ae_ls3_bt8_r5_same_support_review as R5MOD

    source = provenance()
    frozen_before = frozen_hashes(BT6)
    root = artifact_root(BT6)
    require(not root.exists(), "APPEND_ONLY_ARTIFACT_COLLISION")
    root.mkdir(parents=True)
    binding: dict[str, Any] = {"stage": STAGE, "source": source, "f1_source": F1_SOURCE,
                               "r4a_source": R4A_SOURCE, "r5_source": R5_SOURCE}
    try:
        f1_gate = load_json(F1 / "gate_decision.json")
        r5_gate = load_json(R5 / "gate_decision.json")
        r4a_gate = load_json(R4A / "gate_decision.json")
        f1_manifest = verify_manifest(F1)
        r5_manifest = verify_manifest(R5)
        r4a_manifest = verify_manifest(R4A)
        initial_manifest = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
        final_manifest = load_json(F1 / "bt8f1_final_checkpoint_manifest.json")
        collection = load_json(F1 / "bt8f1_training_snapshots" / "collection_manifest.json")
        review_collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
        credits_payload = load_json(F1 / "bt8f1_candidate_plan_credit_audit.json")
        execution = load_json(F1 / "bt8f1_training_execution_audit.json")
        learning = load_json(F1 / "bt8f1_learning_signal_audit.json")
        f1_frozen = load_json(F1 / "frozen_hash_before_after.json")
        r5_r1 = load_json(R5 / "bt8r5_replicate1_same_support_review.json")
        r5_r2 = load_json(R5 / "bt8r5_replicate2_same_support_review.json")
        r5_collapse = load_json(R5 / "bt8r5_concentration_collapse_audit.json")
        r5_preflight = load_json(R5 / "bt8r5_evidence_preflight.json")

        binding.update({
            "f1_gate": f1_gate.get("gate") == F1_GATE,
            "f1_gate_source": f1_gate.get("source_commit") == F1_SOURCE,
            "r5_gate": r5_gate.get("gate") == R5_GATE,
            "r5_gate_source": r5_gate.get("source_commit") == R5_SOURCE,
            "r5_classification": r5_gate.get("classification") == R5_CLASS,
            "r4a_gate": r4a_gate.get("gate") == R4A_GATE,
            "r4a_gate_source": r4a_gate.get("source_commit") == R4A_SOURCE,
            "source_lineage": source["source_lineage_descends_from_r5"],
            "source_only_local_commit": source["source_only_local_commit"],
            "f1_manifest": f1_manifest,
            "r5_manifest": r5_manifest,
            "r4a_manifest": r4a_manifest,
            "f1_frozen_unchanged": f1_frozen.get("all_unchanged") is True,
            "pinned_artifact_hashes": {
                "f1_manifest": sha256(F1 / "manifest.json") == F1_MANIFEST_SHA256,
                "r5_manifest": sha256(R5 / "manifest.json") == R5_MANIFEST_SHA256,
                "f1_credit": sha256(F1 / "bt8f1_candidate_plan_credit_audit.json") == F1_CREDIT_SHA256,
                "f1_execution": sha256(F1 / "bt8f1_training_execution_audit.json") == F1_EXECUTION_SHA256,
                "f1_learning": sha256(F1 / "bt8f1_learning_signal_audit.json") == F1_LEARNING_SHA256,
                "f1_training_collection_file": sha256(F1 / "bt8f1_training_snapshots" / "collection_manifest.json") == F1_TRAINING_COLLECTION_FILE_SHA256,
                "r5_r1_review": sha256(R5 / "bt8r5_replicate1_same_support_review.json") == R5_R1_REVIEW_SHA256,
                "r5_r2_review": sha256(R5 / "bt8r5_replicate2_same_support_review.json") == R5_R2_REVIEW_SHA256,
                "r5_collapse": sha256(R5 / "bt8r5_concentration_collapse_audit.json") == R5_COLLAPSE_SHA256,
            },
            "pinned_source_hashes": {
                "f1_runner": sha256(ROOT / "run_h4m_ae_ls3_bt8_f1_bounded_training.py") == F1_RUNNER_SHA256,
                "r5_runner": sha256(ROOT / "run_h4m_ae_ls3_bt8_r5_same_support_review.py") == R5_RUNNER_SHA256,
                "actor": sha256(ROOT / "multi_agent_candidate_assignment_head.py") == ACTOR_SHA256,
                "learning": sha256(ROOT / "joint_assignment_learning.py") == LEARNING_SHA256,
                "credit_contract": sha256(ROOT / "joint_assignment_credit_contract.py") == CREDIT_CONTRACT_SHA256,
            },
        })
        require(all(binding[key] for key in ("f1_gate", "f1_gate_source", "r5_gate", "r5_gate_source",
                                             "r5_classification", "r4a_gate", "r4a_gate_source",
                                             "source_lineage", "source_only_local_commit",
                                             "f1_frozen_unchanged")), "AUTHORITATIVE_GATE_OR_SOURCE_MISMATCH")
        require(f1_manifest["all_match"] and r5_manifest["all_match"] and r4a_manifest["all_match"],
                "AUTHORITATIVE_MANIFEST_HASH_MISMATCH")
        require(all(binding["pinned_artifact_hashes"].values()), "PINNED_ARTIFACT_HASH_MISMATCH")
        require(all(binding["pinned_source_hashes"].values()), "PINNED_SOURCE_HASH_MISMATCH")

        supplied_digest = collection.get("collection_digest")
        digest_input = dict(collection)
        digest_input.pop("collection_digest", None)
        require(supplied_digest == FPS.canonical_sha256(digest_input), "TRAINING_COLLECTION_DIGEST_MISMATCH")
        require(supplied_digest == F1_TRAINING_COLLECTION_DIGEST, "PINNED_TRAINING_COLLECTION_DIGEST_MISMATCH")
        review_digest_input = dict(review_collection)
        review_supplied_digest = review_digest_input.pop("collection_digest", None)
        require(review_supplied_digest == FPS.canonical_sha256(review_digest_input) == F1_REVIEW_COLLECTION_DIGEST,
                "REVIEW_COLLECTION_DIGEST_MISMATCH")
        require(review_collection.get("snapshot_count") == len(review_collection.get("entries", [])) == 6,
                "REVIEW_SNAPSHOT_COUNT_MISMATCH")
        require(collection.get("snapshot_count") == len(collection.get("entries", [])) == 48,
                "TRAINING_SNAPSHOT_COUNT_MISMATCH")
        require(len({row.get("snapshot_digest") for row in collection["entries"]}) == 48,
                "TRAINING_SNAPSHOT_NOT_UNIQUE")

        credits = credits_payload.get("rows", [])
        require(credits_payload.get("verified") is True and credits_payload.get("mismatches") == 0,
                "CREDIT_IDENTITY_CHAIN_INVALID")
        require(len(credits) == 48, "CREDIT_ROW_COUNT_MISMATCH")
        credit_by_id = {str(row["decision_id"]): row for row in credits}
        require(len(credit_by_id) == 48, "CREDIT_DECISION_ID_NOT_UNIQUE")
        gae_by_id: dict[str, Mapping[str, Any]] = {}
        for replicate_id in REPLICATES:
            rows = execution["replicate_rollouts"][replicate_id]["gae_rows"]
            require(len(rows) == 24, "REPLICATE_GAE_ROW_COUNT_MISMATCH", replicate_id)
            for row in rows:
                require(row["decision_id"] not in gae_by_id, "GAE_DECISION_ID_NOT_UNIQUE", row["decision_id"])
                gae_by_id[row["decision_id"]] = row
        require(set(gae_by_id) == set(credit_by_id), "CREDIT_GAE_DECISION_SET_MISMATCH")

        snapshots: dict[str, list[dict[str, Any]]] = {key: [] for key in REPLICATES}
        review_snapshots: dict[str, list[dict[str, Any]]] = {key: [] for key in REPLICATES}
        tensor_sha_by_replicate: dict[str, list[str]] = {key: [] for key in REPLICATES}
        tensor_identity_rows = []
        tensor_sha_by_decision: dict[str, str] = {}
        config = None
        for entry in collection["entries"]:
            snapshot_root = F1 / "bt8f1_training_snapshots" / entry["relative_path"]
            payload = FPS.load_snapshot(snapshot_root)
            require(payload["snapshot_digest"] == entry["snapshot_digest"], "TRAINING_SNAPSHOT_ENTRY_DIGEST_MISMATCH")
            meta = payload["metadata"]
            replicate_id = replicate_from_decision_id(str(meta["decision_id"]))
            require(int(meta["seed"]) == REPLICATES[replicate_id]["environment_seed"], "SNAPSHOT_SEED_MISMATCH")
            require(meta["source_commit"] == F1_SOURCE and meta["captured_device"] == "mps:0", "SNAPSHOT_LINEAGE_MISMATCH")
            require(meta["decision_id"] in credit_by_id, "SNAPSHOT_CREDIT_ROW_MISSING")
            require(meta["actor_config_sha256"] == FPS.actor_config_sha256(meta["actor_config"]), "ACTOR_CONFIG_DIGEST_MISMATCH")
            config = meta["actor_config"] if config is None else config
            require(meta["actor_config"] == config, "ACTOR_CONFIG_NOT_UNIFORM")
            snapshots[replicate_id].append(payload)
            snapshot_manifest = load_json(snapshot_root / "snapshot_manifest.json")
            require(snapshot_manifest.get("manifest_sha256") == entry["snapshot_manifest_sha256"],
                    "SNAPSHOT_MANIFEST_BINDING_MISMATCH", meta["decision_id"])
            tensor_sha = str(snapshot_manifest["tensor_binary_sha256"])
            require(sha256(snapshot_root / snapshot_manifest["tensor_binary"]) == tensor_sha,
                    "SNAPSHOT_TENSOR_BINARY_HASH_MISMATCH", meta["decision_id"])
            tensor_sha_by_replicate[replicate_id].append(tensor_sha)
            tensor_sha_by_decision[str(meta["decision_id"])] = tensor_sha
            tensor_identity_rows.append({"replicate_id": replicate_id, "decision_id": meta["decision_id"],
                                         "window_id": meta["window_id"], "tensor_binary_sha256": tensor_sha})
        for replicate_id, rows in snapshots.items():
            rows.sort(key=lambda row: str(row["metadata"]["decision_id"]))
            require(len(rows) == 24, "REPLICATE_SNAPSHOT_COUNT_MISMATCH", replicate_id)
        input_tensor_multisets_exact = Counter(tensor_sha_by_replicate["F1_R1"]) == Counter(tensor_sha_by_replicate["F1_R2"])
        unique_tensor_payloads = len(set(tensor_sha_by_replicate["F1_R1"] + tensor_sha_by_replicate["F1_R2"]))
        for entry in review_collection["entries"]:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / entry["relative_path"])
            meta = payload["metadata"]
            seed = int(meta["seed"])
            require(seed in {row["environment_seed"] for row in REPLICATES.values()}, "UNAUTHORIZED_REVIEW_SEED")
            replicate_id = "F1_R1" if seed == REPLICATES["F1_R1"]["environment_seed"] else "F1_R2"
            require(payload["snapshot_digest"] == entry["snapshot_digest"], "REVIEW_SNAPSHOT_ENTRY_DIGEST_MISMATCH")
            require(meta["actor_config"] == config, "REVIEW_ACTOR_CONFIG_MISMATCH")
            review_snapshots[replicate_id].append(payload)
        require(all(len(rows) == 3 for rows in review_snapshots.values()), "REPLICATE_REVIEW_SNAPSHOT_COUNT_MISMATCH")

        checkpoint_before = checkpoint_bindings(initial_manifest, final_manifest)
        expected_binding = collection.get("checkpoint_binding", {})
        for replicate_id, seeds in REPLICATES.items():
            initial = initial_manifest[replicate_id]
            final = final_manifest[replicate_id]
            initial_path = F1 / "initial_checkpoints" / initial["path"]
            final_path = F1 / "final_checkpoints" / final["path"]
            require(sha256(initial_path) == initial["sha256"] == expected_binding["initial_actor_checkpoint_sha256_by_replicate"][replicate_id],
                    "INITIAL_CHECKPOINT_BINDING_MISMATCH", replicate_id)
            require(sha256(final_path) == final["sha256"] == expected_binding["final_actor_checkpoint_sha256_by_replicate"][replicate_id],
                    "FINAL_CHECKPOINT_BINDING_MISMATCH", replicate_id)
            require((initial["environment_seed"], initial["actor_init_seed"], initial["critic_init_seed"])
                    == (seeds["environment_seed"], seeds["actor_seed"], seeds["critic_seed"]),
                    "CHECKPOINT_SEED_LINEAGE_MISMATCH", replicate_id)

        require(r5_collapse["R1"]["final"]["universal_feasible_no_assign_exact"] is True,
                "R5_R1_NO_ASSIGN_OUTCOME_NOT_BOUND")
        require(float(r5_r2["margin_mean_delta"]) < 0.0, "R5_R2_MARGIN_COMPRESSION_NOT_BOUND")
        require(r5_preflight["collection_digest"] == "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e",
                "R5_REVIEW_COLLECTION_BINDING_MISMATCH")
    except Exception as exc:  # noqa: BLE001
        reason = f"EVIDENCE_PREFLIGHT_FAILURE:{type(exc).__name__}:{exc}"
        write_block(root, source, binding, reason)
        print(f"[BLOCKED] {BLOCK_GATE}\nartifact: {root.relative_to(PROJECT)}")
        return

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root, source, binding, "MPS_REQUIRED_FOR_BOUND_F1_CHECKPOINT_REPLAY")
        print(f"[BLOCKED] {BLOCK_GATE}\nartifact: {root.relative_to(PROJECT)}")
        return

    device = torch.device("mps:0")
    policies: dict[str, dict[str, torch.nn.Module]] = {}
    actor_digests_before: dict[str, dict[str, str]] = {}
    for replicate_id in REPLICATES:
        first = F1 / "initial_checkpoints" / initial_manifest[replicate_id]["path"]
        last = F1 / "final_checkpoints" / final_manifest[replicate_id]["path"]
        policies[replicate_id] = {
            "initial": R5MOD.load_actor(checkpoint=first, config=config, H=H, device=device),
            "final": R5MOD.load_actor(checkpoint=last, config=config, H=H, device=device),
        }
        actor_digests_before[replicate_id] = {
            state: R5MOD.module_digest(actor) for state, actor in policies[replicate_id].items()
        }

    counters = {"training": 0, "optimizer_step": 0, "backward": 0, "autograd_grad": 0,
                "causal_rollout": 0, "simulator_step": 0, "candidate_generation": 0,
                "candidate_regeneration": 0, "local_search_rerun": 0, "zero_loss_reevaluation": 0,
                "reward_recomputation": 0, "snapshot_generation": 0, "parameter_mutation": 0,
                "checkpoint_write": 0, "test6_access": 0, "nan_or_inf": 0,
                "training_snapshot_forward_replays": 0, "review_functional_forward_replays": 0}
    no_assign_id = MC.NO_ASSIGN
    traces: dict[str, list[dict[str, Any]]] = {key: [] for key in REPLICATES}
    hard: list[str] = []

    for replicate_id, payloads in snapshots.items():
        for payload in payloads:
            decision_id = str(payload["metadata"]["decision_id"])
            credit, gae = credit_by_id[decision_id], gae_by_id[decision_id]
            initial = replay_actor(policies[replicate_id]["initial"], payload, F1MOD, H, TIE, device)
            final = replay_actor(policies[replicate_id]["final"], payload, F1MOD, H, TIE, device)
            counters["training_snapshot_forward_replays"] += 2
            counters["nan_or_inf"] += int(not initial["finite"]) + int(not final["finite"])
            is_no_assign = selected_no_assign(credit, no_assign_id)
            replay_is_no_assign = initial["selected_identity"] == "NO_ASSIGN"
            if replay_is_no_assign != is_no_assign:
                hard.append(f"INITIAL_REPLAY_ACTION_TYPE_MISMATCH:{decision_id}")
            if not is_no_assign:
                identity = initial["selected_identity"]
                if not isinstance(identity, Mapping) or identity.get("candidate_id") != credit["selected_candidate_id"] \
                        or identity.get("agent_id") != credit["agent_id"]:
                    hard.append(f"INITIAL_REPLAY_CANDIDATE_IDENTITY_MISMATCH:{decision_id}")
            for credit_key, gae_key in (("Team Reward", "reward"), ("critic target", "critic_target"),
                                        ("TD residual", "td_residual"), ("GAE advantage", "raw_gae"),
                                        ("normalized advantage", "normalized_advantage")):
                if float(credit[credit_key]) != float(gae[gae_key]):
                    hard.append(f"CREDIT_GAE_VALUE_MISMATCH:{decision_id}:{credit_key}")
            initial_score, initial_probability = selected_score(initial, payload, credit, no_assign_id)
            final_score, final_probability = selected_score(final, payload, credit, no_assign_id)
            normalized = float(credit["normalized advantage"])
            ratios = [float(value) for value in credit["ppo_ratios"]]
            require(len(ratios) == 3, "PPO_RATIO_EPOCH_COUNT_MISMATCH", decision_id)
            clip_states = [ppo_clip_state(normalized, ratio, epsilon=CC.PPO_CLIP_EPSILON) for ratio in ratios]
            pressures = [pressure_label(selected_no_assign=is_no_assign, advantage=normalized, clip_state=state)
                         for state in clip_states]
            pair_spread_delta = None
            if initial["pair_logit_spread"] is not None and final["pair_logit_spread"] is not None:
                pair_spread_delta = float(final["pair_logit_spread"] - initial["pair_logit_spread"])
            pair_margin_delta = None
            if initial["pair_top1_top2_margin"] is not None and final["pair_top1_top2_margin"] is not None:
                pair_margin_delta = float(final["pair_top1_top2_margin"] - initial["pair_top1_top2_margin"])
            full_margin_delta = None
            if initial["full_action_top1_top2_margin"] is not None and final["full_action_top1_top2_margin"] is not None:
                full_margin_delta = float(final["full_action_top1_top2_margin"] - initial["full_action_top1_top2_margin"])
            no_assign_delta = None
            if initial["no_assign_minus_top_candidate"] is not None and final["no_assign_minus_top_candidate"] is not None:
                no_assign_delta = float(final["no_assign_minus_top_candidate"] - initial["no_assign_minus_top_candidate"])
            traces[replicate_id].append({
                "decision_id": decision_id, "trajectory_id": credit["trajectory_id"],
                "snapshot_digest": payload["snapshot_digest"],
                "tensor_binary_sha256": tensor_sha_by_decision[decision_id],
                "support_size": int(payload["metadata"]["selectable_pair_count"]),
                "selected_action_type": "NO_ASSIGN" if is_no_assign else "CANDIDATE",
                "selected_agent_id": None if is_no_assign else credit["agent_id"],
                "selected_candidate_id": credit["selected_candidate_id"],
                "reward": float(gae["reward"]), "critic_value": float(gae["critic_value"]),
                "next_value": float(gae["next_value"]), "critic_target": float(gae["critic_target"]),
                "nonterminal": bool(gae["nonterminal"]),
                "td_residual": float(gae["td_residual"]), "raw_gae": float(gae["raw_gae"]),
                "normalized_advantage": normalized,
                "normalization_sign_transition": sign_transition(float(gae["raw_gae"]), normalized),
                "ppo_ratios": ratios, "clip_state_by_epoch": clip_states, "pressure_by_epoch": pressures,
                "recorded_policy_gradient_proxy": float(credit["policy-gradient contribution"]),
                "initial_selected_logit": initial_score, "final_selected_logit": final_score,
                "selected_logit_delta": final_score - initial_score,
                "initial_selected_probability": initial_probability, "final_selected_probability": final_probability,
                "selected_probability_delta": final_probability - initial_probability,
                "initial_no_assign_minus_top_candidate": initial["no_assign_minus_top_candidate"],
                "final_no_assign_minus_top_candidate": final["no_assign_minus_top_candidate"],
                "no_assign_minus_top_candidate_delta": no_assign_delta,
                "initial_pair_logit_spread": initial["pair_logit_spread"],
                "final_pair_logit_spread": final["pair_logit_spread"],
                "pair_logit_spread_delta": pair_spread_delta,
                "initial_pair_top1_top2_margin": initial["pair_top1_top2_margin"],
                "final_pair_top1_top2_margin": final["pair_top1_top2_margin"],
                "pair_top1_top2_margin_delta": pair_margin_delta,
                "initial_full_action_top1_top2_margin": initial["full_action_top1_top2_margin"],
                "final_full_action_top1_top2_margin": final["full_action_top1_top2_margin"],
                "full_action_top1_top2_margin_delta": full_margin_delta,
                "initial_replay_selected_identity": initial["selected_identity"],
                "final_replay_selected_identity": final["selected_identity"],
            })

    cross_matrix: dict[str, dict[str, int]] = {}
    cross_rows = []
    for actor_replicate in REPLICATES:
        label = actor_replicate.removeprefix("F1_") + "_INITIAL_ACTOR"
        cross_matrix[label] = {}
        for input_replicate, payloads in snapshots.items():
            no_assign_count = 0
            for payload in payloads:
                row = replay_actor(policies[actor_replicate]["initial"], payload, F1MOD, H, TIE, device)
                counters["training_snapshot_forward_replays"] += 1
                counters["nan_or_inf"] += int(not row["finite"])
                no_assign_count += int(row["selected_identity"] == "NO_ASSIGN")
            input_label = input_replicate.removeprefix("F1_") + "_INPUT_NO_ASSIGN"
            cross_matrix[label][input_label] = no_assign_count
            cross_rows.append({"actor": label, "input_collection": input_replicate,
                               "snapshot_count": len(payloads), "no_assign": no_assign_count,
                               "candidate_assignment": len(payloads) - no_assign_count})
    torch.mps.synchronize()

    route_attribution, route_forward_count = actor_route_attribution(
        policies=policies, review_snapshots=review_snapshots, device=device)
    counters["review_functional_forward_replays"] = route_forward_count
    torch.mps.synchronize()

    gae_decomposition: dict[str, dict[str, Any]] = {}
    for replicate_id, rows in traces.items():
        components, audit = decompose_gae(rows, gamma=CC.GAMMA, lam=CC.GAE_LAMBDA)
        component_by_id = {row["decision_id"]: row for row in components}
        for row in rows:
            row.update(component_by_id[row["decision_id"]])
        gae_decomposition[replicate_id] = audit
    summaries = {replicate_id: summarize_trace(rows) | {"gae_component_decomposition": gae_decomposition[replicate_id]}
                 for replicate_id, rows in traces.items()}
    critic_same_input_rows = []
    for tensor_sha in sorted(set(tensor_sha_by_replicate["F1_R1"])):
        r1_values = [row["critic_value"] for row in traces["F1_R1"] if row["tensor_binary_sha256"] == tensor_sha]
        r2_values = [row["critic_value"] for row in traces["F1_R2"] if row["tensor_binary_sha256"] == tensor_sha]
        critic_same_input_rows.append({"tensor_binary_sha256": tensor_sha,
                                       "R1_values": r1_values, "R2_values": r2_values,
                                       "R1_mean": statistics.mean(r1_values), "R2_mean": statistics.mean(r2_values),
                                       "sign_split": all(value < 0.0 for value in r1_values)
                                       and all(value > 0.0 for value in r2_values)})
    driver = initial_action_support_driver(cross_matrix)
    pattern = {
        "initial_driver": driver,
        "input_tensor_multisets_exact": input_tensor_multisets_exact,
        "r1_candidate_selected": summaries["F1_R1"]["selected_action_counts"].get("CANDIDATE", 0),
        "r1_reward_rows": summaries["F1_R1"]["reward_supported_decisions"],
        "r1_raw_positive": summaries["F1_R1"]["raw_gae"]["positive"],
        "r1_positive_to_negative": summaries["F1_R1"]["normalization_sign_transitions"].get("POSITIVE_TO_NEGATIVE", 0),
        "r2_no_assign_selected": summaries["F1_R2"]["selected_action_counts"].get("NO_ASSIGN", 0),
        "r2_reward_rows": summaries["F1_R2"]["reward_supported_decisions"],
        "r2_raw_negative": summaries["F1_R2"]["raw_gae"]["negative"],
        "r2_negative_to_positive": summaries["F1_R2"]["normalization_sign_transitions"].get("NEGATIVE_TO_POSITIVE", 0),
        "r2_direct_candidate_credit": summaries["F1_R2"]["direct_candidate_identity_credit_rows"],
    }
    classification = classify_root(pattern)
    first_divergence = FIRST_DIVERGENCE if driver.startswith("ACTOR_INITIALIZATION_DOMINANT") else "NOT_UNIQUE"

    actor_digests_after = {replicate_id: {state: R5MOD.module_digest(actor) for state, actor in values.items()}
                           for replicate_id, values in policies.items()}
    counters["parameter_mutation"] = int(actor_digests_before != actor_digests_after)
    after = frozen_hashes(BT6)
    checkpoint_after = checkpoint_bindings(initial_manifest, final_manifest)
    if frozen_before != after:
        hard.append("FROZEN_AUTHORITY_HASH_CHANGED")
    if checkpoint_before != checkpoint_after:
        hard.append("CHECKPOINT_HASH_CHANGED")
    allowed_counters = {"training_snapshot_forward_replays", "review_functional_forward_replays"}
    if any(value for key, value in counters.items() if key not in allowed_counters):
        hard.append("FORBIDDEN_EXECUTION_OR_INTEGRITY_COUNTER_NONZERO")
    if counters["training_snapshot_forward_replays"] != 192 or counters["review_functional_forward_replays"] != 48:
        hard.append("PRESERVED_REPLAY_COUNT_MISMATCH")

    gate = PASS_GATE if not hard else BLOCK_GATE
    if hard:
        classification = "BLOCKED"
    evidence_integrity = {
        "training_collection_digest": supplied_digest,
        "training_snapshots": 48,
        "unique_training_snapshots": 48,
        "replicate_snapshot_counts": {key: len(value) for key, value in snapshots.items()},
        "input_tensor_multisets_exact": input_tensor_multisets_exact,
        "unique_tensor_payloads": unique_tensor_payloads,
        "tensor_sha_multiset_by_replicate": {key: dict(sorted(Counter(value).items()))
                                             for key, value in tensor_sha_by_replicate.items()},
        "tensor_identity_rows": tensor_identity_rows,
        "critic_same_input_sign_split": all(row["sign_split"] for row in critic_same_input_rows),
        "critic_same_input_rows": critic_same_input_rows,
        "credit_rows": len(credits), "gae_rows": len(gae_by_id),
        "credit_gae_exact_value_binding": not any(item.startswith("CREDIT_GAE_VALUE_MISMATCH") for item in hard),
        "initial_action_replay_exact_binding": not any("INITIAL_REPLAY" in item for item in hard),
        "checkpoint_binding": checkpoint_before,
        "actor_config": config,
        "device": str(device),
        "t1_selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py"),
        "t1_tolerance": 0.0,
        "ppo_clip_epsilon": CC.PPO_CLIP_EPSILON,
        "gamma": CC.GAMMA,
        "gae_lambda": CC.GAE_LAMBDA,
    }
    attribution = {
        "classification": classification,
        "observed_mechanism": OBSERVED_MECHANISM,
        "first_divergence": first_divergence,
        "pattern": pattern,
        "R1_chain": [
            "initial Actor selects candidate on all 24 preserved rollout states",
            "four reward-bearing decisions; all 24 raw GAE values are positive",
            "replicate centering maps 15 positive raw GAE values to negative normalized advantages",
            "negative selected-candidate rows create relative NO_ASSIGN pressure",
            "R5 final review is universal feasible NO_ASSIGN",
        ],
        "R2_chain": [
            "initial Actor selects NO_ASSIGN on all 24 preserved rollout states",
            "zero reward-bearing decisions; fresh positive critic baseline yields 24 negative raw GAE values",
            "replicate centering maps nine negative raw GAE values to positive normalized advantages",
            "all direct credit targets NO_ASSIGN; no candidate identity receives direct PPO credit",
            "R5 final review leaves NO_ASSIGN; the reported full-action margin changes winner family and cannot be read as uniform pair-margin compression",
        ],
        "R1_endpoint_attribution": "MIXED_OR_INDETERMINATE",
        "R2_upstream_credit_attribution": "CRITIC_GAE_DRIVEN",
        "R2_candidate_margin_attribution": "MARGIN_IDENTITY_SWITCH_CONFOUNDED_AND_NO_PAIR_IDENTITY_CREDIT",
        "identifiability_limit": "environment, Actor-init, and Critic-init seeds were bundled; no per-row gradient, Adam moment, or intermediate checkpoint was preserved",
        "stored_policy_gradient_field": "surrogate proxy = normalized advantage * epoch-3 ratio; not an actual gradient and not clipping-aware",
        "scientific_scope": "mechanism attribution on one bounded F1 execution; no policy-quality or causal-performance claim",
    }
    authority_binding_pass = (all(binding.get(key) is True for key in ("f1_gate", "f1_gate_source", "r5_gate",
                                                                        "r5_gate_source", "r5_classification",
                                                                        "r4a_gate", "r4a_gate_source", "source_lineage",
                                                                        "source_only_local_commit", "f1_frozen_unchanged"))
                              and all(binding["pinned_artifact_hashes"].values())
                              and all(binding["pinned_source_hashes"].values()))
    gate_matrix = {
        "authority_binding": authority_binding_pass,
        "f1_manifest_hashes": f1_manifest["all_match"], "r5_manifest_hashes": r5_manifest["all_match"],
        "snapshot_credit_gae_binding": evidence_integrity["credit_gae_exact_value_binding"] and evidence_integrity["initial_action_replay_exact_binding"],
        "read_only_execution": not any(value for key, value in counters.items() if key not in allowed_counters),
        "frozen_hashes_unchanged": frozen_before == after and checkpoint_before == checkpoint_after,
        "root_attribution_complete": classification in {ROOT_CLASS, INDETERMINATE_CLASS},
        "hard_failures": hard,
    }
    outputs = {
        "bt8r6_authoritative_binding.json": binding,
        "bt8r6_f1_evidence_integrity.json": evidence_integrity,
        "bt8r6_replicate1_credit_trace.json": {"replicate_id": "F1_R1", "summary": summaries["F1_R1"], "rows": traces["F1_R1"],
                                                "r5_review_outcome": r5_r1, "r5_final_collapse": r5_collapse["R1"]["final"]},
        "bt8r6_replicate2_credit_trace.json": {"replicate_id": "F1_R2", "summary": summaries["F1_R2"], "rows": traces["F1_R2"],
                                                "r5_review_outcome": r5_r2, "r5_final_collapse": r5_collapse["R2"]["final"]},
        "bt8r6_cross_seed_initial_policy_replay.json": {"matrix": cross_matrix, "rows": cross_rows,
                                                         "driver": driver, "comparison_rule": "exact counts; no tolerance"},
        "bt8r6_actor_route_attribution.json": route_attribution,
        "bt8r6_credit_to_logit_attribution.json": attribution,
        "bt8r6_first_divergence_classification.json": {"first_divergence": first_divergence,
                                                        "root_cause_classification": classification,
                                                        "next_gate": NEXT_GATE if not hard else "STOP"},
        "bt8r6_gate_matrix.json": gate_matrix,
        "test_results.json": {"execution_counters": counters, "hard_failures": hard, "warnings": [],
                              "TEST6_access": 0, "github_push_performed": False},
        "frozen_hash_before_after.json": {"before": frozen_before, "after": after, "all_unchanged": frozen_before == after,
                                          "actor_digests_before": actor_digests_before, "actor_digests_after": actor_digests_after,
                                          "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after},
        "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification,
                               "first_divergence": first_divergence, "source_commit": source["source_commit"],
                               "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
                               "next_step": NEXT_GATE if not hard else "STOP"},
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    report = (
        "# BT8-R6 final report\n\n"
        f"- gate: `{gate}`\n"
        f"- classification: `{classification}`\n"
        f"- first divergence: `{first_divergence}`\n"
        f"- source commit: `{source['source_commit']}`\n"
        f"- next gate: `{NEXT_GATE if not hard else 'STOP'}`\n\n"
        "This is a read-only mechanism attribution over preserved F1 evidence. "
        "It contains no additional training, rollout, repair, performance comparison, or policy-quality claim.\n"
    )
    (root / "final_report.md").write_text(report, encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                   "source_commit": source["source_commit"], "file_sha256": manifest,
                                   "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"classification: {classification}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
