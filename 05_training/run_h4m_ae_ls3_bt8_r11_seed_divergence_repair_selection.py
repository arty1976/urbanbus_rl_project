#!/usr/bin/env python3
"""BT8-R11: read-only seed-cause factorization and repair-contract selection.

Only persisted initial checkpoints, review/training snapshots, reward lineages,
and E1 labels are read.  The 2×2 Actor/support and Critic/lineage calculations
are independent frozen forwards and algebra over stored transitions; they never
generate a candidate, advance a state, or update an optimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R11"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R11_MINIMAL_SEED_DIVERGENCE_CAUSE_ISOLATION_AND_REPAIR_SELECTION_COMPLETE"
COMMON_SUPPORT_BLOCK = "BLOCKED_INSUFFICIENT_COMMON_SUPPORT_EVIDENCE"
SELECTION_BLOCK = "BLOCKED_SEED_REPAIR_SELECTION_UNRESOLVED"
STRUCTURAL_BLOCK = "BLOCKED_SEED_CAUSE_AUDIT_STRUCTURAL_INTEGRITY_FAILURE"
MPS_BLOCK = "BLOCKED_MPS_SEED_CAUSE_AUDIT_ENVIRONMENT_UNAVAILABLE"

R10_SOURCE = "637251d74ee34fff8aa4b3181fbae82d46da1806"
E1_SOURCE = "886e0e1e5b3c49d42cffe485aa4834a40c04cb03"
R9_SOURCE = "e70b2bdc51674547e7d7e6dc31a6c31c8b6eec56"
E1_CONTRACT_SHA256 = "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"
R10_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R10_E0_E1_SAME_SUPPORT_FROZEN_POLICY_REVIEW_COMPLETE"
E1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_E1_EXACT_PRESERVED_BATCH_BOUNDED_RETRAINING_COMPLETE"
R9_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R9_E1_SEPARATE_BOUNDED_RETRAINING_AUTHORIZATION_SELECTION_COMPLETE"
R7_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R7_MINIMAL_ACTOR_CRITIC_SEED_FACTORIZATION_AND_CREDIT_ELIGIBILITY_SELECTION_COMPLETE"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
T1_CONTRACT = "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
R7 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility_20260825_124641+09:00"
R9 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r9_e1_retraining_authority_selection_20260825_200038+09:00"
E1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining_20260826_095149+09:00"
R10 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review_20260826_114330+09:00"
R6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution_20260824_234227+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection.py",
    "05_training/test_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection.py",
}
LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
REPLICATES = {
    "F1_R1": {"actor_label": "A", "critic_label": "C", "environment_seed": 20260822,
              "actor_seed": 20260824, "critic_seed": 20260826},
    "F1_R2": {"actor_label": "B", "critic_label": "D", "environment_seed": 20260823,
              "actor_seed": 20260825, "critic_seed": 20260827},
}
ACTORS = {"A": "F1_R1", "B": "F1_R2"}
CRITICS = {"C": "F1_R1", "D": "F1_R2"}
ELIGIBLE_CATEGORIES = {"DIRECT_CANDIDATE_REWARD", "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD", "NO_ASSIGN_REWARD_SUPPORTED"}


class R11Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R11Error(code, detail)


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
    require(path.is_file(), COMMON_SUPPORT_BLOCK, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    expected = manifest.get("file_sha256")
    require(isinstance(expected, Mapping), COMMON_SUPPORT_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in expected.items() if not (root / str(name)).is_file() or sha256(root / str(name)) != digest]
    return {"manifest_sha256": sha256(root / "manifest.json"), "declared_file_count": len(expected),
            "mismatches": mismatches, "all_match": not mismatches}


def source_provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R10_SOURCE}..HEAD"]).splitlines() if name]
    return {
        "source_commit": git(["rev-parse", "HEAD"]), "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r10": git(["merge-base", R10_SOURCE, "HEAD"]) == R10_SOURCE,
        "changed_files_since_r10": changed, "source_only_local_commit": set(changed) == SOURCE_FILES,
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection_{stamp}"


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def describe(values: Sequence[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "min": None, "mean": None, "max": None, "positive": 0, "negative": 0, "zero": 0}
    return {"count": len(rows), "min": min(rows), "mean": statistics.mean(rows), "max": max(rows),
            "positive": sum(value > 0.0 for value in rows), "negative": sum(value < 0.0 for value in rows),
            "zero": sum(value == 0.0 for value in rows)}


def action_identity(selection: Any) -> str | tuple[str, str]:
    return "NO_ASSIGN" if selection.selected.is_no_assign else (str(selection.selected.agent_id), str(selection.selected.candidate_id))


def action_family(identity: str | tuple[str, str]) -> str:
    return "NO_ASSIGN" if identity == "NO_ASSIGN" else "CANDIDATE"


def counter_template() -> dict[str, int]:
    return {
        "training": 0, "optimizer_step": 0, "backward": 0, "autograd_grad": 0, "causal_rollout": 0,
        "simulator_step": 0, "candidate_generation": 0, "candidate_regeneration": 0, "local_search_rerun": 0,
        "zero_loss_reevaluation": 0, "reward_recomputation": 0, "checkpoint_write": 0,
        "parameter_mutation": 0, "checkpoint_mutation": 0, "review_optimizer_rows": 0, "future_leakage": 0,
        "nan_or_inf": 0, "test6_access": 0, "github_push": 0, "actor_support_forward": 0,
        "critic_forward": 0,
    }


def actor_row(*, actor: torch.nn.Module, payload: Mapping[str, Any], R1: Any, H: Any, TIE: Any,
              device: torch.device, actor_label: str, support_label: str) -> dict[str, Any]:
    result = R1.frozen_forward(actor, payload, device=device, head=H, tie=TIE)
    metadata, tensors = payload["metadata"], payload["tensors"]
    support = int(result["support_size"])
    pair_logits = [float(value) for value in result["pair_logits"][0].detach().cpu().tolist()]
    probabilities = [float(value) for value in result["probabilities"][0].detach().cpu().tolist()]
    no_assign = float(result["no_assign_logit"][0, 0].detach().cpu())
    mask = tensors["safe_mask"][0, :support]
    selected = action_identity(result["selection"])
    legal_pairs = {(str(item["agent_id"]), str(item["candidate_id"])) for item in metadata["candidate_ids"]}
    legal = selected == "NO_ASSIGN" or selected in legal_pairs
    require(bool(mask.all().item()) and legal and len(pair_logits) == support and len(probabilities) == support + 1,
            STRUCTURAL_BLOCK, f"actor_support_legality={metadata['decision_id']}")
    finite = bool(result["finite"] and all(math.isfinite(item) for item in [*pair_logits, no_assign, *probabilities]))
    require(finite, STRUCTURAL_BLOCK, f"actor_support_nonfinite={metadata['decision_id']}")
    ordered = sorted(pair_logits, reverse=True)
    return {
        "actor_label": actor_label, "support_label": support_label, "window_id": str(metadata["window_id"]),
        "decision_id": str(metadata["decision_id"]), "snapshot_digest": str(payload["snapshot_digest"]),
        "candidate_support_digest": str(metadata["candidate_support_digest"]), "support_size": support,
        "selected_identity": selected, "action_family": action_family(selected), "feasible_no_assign": selected == "NO_ASSIGN" and support > 0,
        "no_assign_minus_best_pair": no_assign - max(pair_logits),
        "candidate_only_top1_top2_margin": ordered[0] - ordered[1] if len(ordered) >= 2 else None,
        "entropy": float(result["entropy"]), "exact_tie": bool(result["selection"].exact_tie), "finite": finite,
    }


def actor_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows), "action_family": dict(sorted(Counter(str(row["action_family"]) for row in rows).items())),
        "feasible_no_assign": sum(bool(row["feasible_no_assign"]) for row in rows),
        "no_assign_minus_best_pair": describe([float(row["no_assign_minus_best_pair"]) for row in rows]),
        "candidate_only_margin": describe([float(row["candidate_only_top1_top2_margin"]) for row in rows if row["candidate_only_top1_top2_margin"] is not None]),
        "entropy": describe([float(row["entropy"]) for row in rows]), "exact_ties": sum(bool(row["exact_tie"]) for row in rows),
    }


def matched_by_window(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["window_id"]): row for row in rows}
    require(len(result) == len(rows), COMMON_SUPPORT_BLOCK, "duplicate_window_in_factorization")
    return result


def actor_support_factorization(matrix: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    required = {"A×R1", "A×R2", "B×R1", "B×R2"}
    require(set(matrix) == required and all(len(rows) == 3 for rows in matrix.values()), COMMON_SUPPORT_BLOCK, "actor_support_matrix_shape")
    views = {key: matched_by_window(rows) for key, rows in matrix.items()}
    windows = set(views["A×R1"])
    require(all(set(view) == windows for view in views.values()), COMMON_SUPPORT_BLOCK, "actor_support_window_alignment")
    actor_pairs, support_pairs = [], []
    for support in ("R1", "R2"):
        for window in sorted(windows):
            a, b = views[f"A×{support}"][window], views[f"B×{support}"][window]
            actor_pairs.append({"support": support, "window_id": window,
                                "action_family_different": a["action_family"] != b["action_family"],
                                "no_assign_gap_abs": abs(float(a["no_assign_minus_best_pair"]) - float(b["no_assign_minus_best_pair"])),
                                "candidate_margin_abs": None if a["candidate_only_top1_top2_margin"] is None else abs(float(a["candidate_only_top1_top2_margin"]) - float(b["candidate_only_top1_top2_margin"])),
                                "entropy_abs": abs(float(a["entropy"]) - float(b["entropy"]))})
    for actor in ("A", "B"):
        for window in sorted(windows):
            r1, r2 = views[f"{actor}×R1"][window], views[f"{actor}×R2"][window]
            support_pairs.append({"actor": actor, "window_id": window,
                                  "candidate_support_digest_equal": r1["candidate_support_digest"] == r2["candidate_support_digest"],
                                  "action_family_different": r1["action_family"] != r2["action_family"],
                                  "no_assign_gap_abs": abs(float(r1["no_assign_minus_best_pair"]) - float(r2["no_assign_minus_best_pair"])),
                                  "candidate_margin_abs": None if r1["candidate_only_top1_top2_margin"] is None else abs(float(r1["candidate_only_top1_top2_margin"]) - float(r2["candidate_only_top1_top2_margin"])),
                                  "entropy_abs": abs(float(r1["entropy"]) - float(r2["entropy"]))})
    actor_mismatch = sum(bool(row["action_family_different"]) for row in actor_pairs)
    support_mismatch = sum(bool(row["action_family_different"]) for row in support_pairs)
    actor_signs = {label: {"negative": all(float(row["no_assign_minus_best_pair"]) < 0.0 for row in [*matrix[f"{label}×R1"], *matrix[f"{label}×R2"]]),
                             "positive": all(float(row["no_assign_minus_best_pair"]) > 0.0 for row in [*matrix[f"{label}×R1"], *matrix[f"{label}×R2"]])}
                   for label in ("A", "B")}
    if actor_mismatch == len(actor_pairs) and support_mismatch == 0 and actor_signs["A"]["negative"] and actor_signs["B"]["positive"]:
        classification = "ACTOR_INIT_DOMINANT"
    elif actor_mismatch == 0 and support_mismatch > 0:
        classification = "SUPPORT_DOMINANT"
    elif 0 < actor_mismatch < len(actor_pairs) or 0 < support_mismatch < len(support_pairs):
        classification = "ACTOR_SUPPORT_INTERACTION"
    else:
        classification = "MIXED"
    return {
        "matrix": {key: {"summary": actor_summary(rows), "rows": list(rows)} for key, rows in matrix.items()},
        "actor_effect_same_support": {"rows": actor_pairs, "action_family_mismatch_count": actor_mismatch,
                                        "no_assign_gap_abs": describe([float(row["no_assign_gap_abs"]) for row in actor_pairs])},
        "support_effect_same_actor": {"rows": support_pairs, "action_family_mismatch_count": support_mismatch,
                                        "candidate_support_exact_match_count": sum(bool(row["candidate_support_digest_equal"]) for row in support_pairs),
                                        "no_assign_gap_abs": describe([float(row["no_assign_gap_abs"]) for row in support_pairs])},
        "actor_sign_consistency": actor_signs, "classification": classification,
        "method": "2×2 frozen Actor×preserved-support replay; supports are not regenerated and cross-support candidate identities are never relabeled",
    }


def load_critic(*, checkpoint: Path, config: Mapping[str, Any], JL: Any, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping) and "critic" in payload, COMMON_SUPPORT_BLOCK, f"critic_state={checkpoint}")
    critic = JL.JointAssignmentCritic(global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
                                      agent_dim=int(config["agent_dim"]), safe_summary_dim=1 + 2 * int(config["candidate_dim"])).to(device)
    critic.load_state_dict(payload["critic"], strict=True)
    return critic.eval()


def critic_value(*, critic: torch.nn.Module, payload: Mapping[str, Any], JL: Any, device: torch.device) -> float:
    tensors = {name: value.to(device) for name, value in payload["tensors"].items()}
    with torch.no_grad():
        value = critic(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                       agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                       safe_summary=JL.safe_set_summary(tensors["candidate_feats"], tensors["safe_mask"]))
    result = float(value[0].detach().cpu())
    require(math.isfinite(result), STRUCTURAL_BLOCK, "critic_nonfinite")
    return result


def critic_lineage_rows(*, fixed: Sequence[Mapping[str, Any]], trace: Sequence[Mapping[str, Any]],
                        eligibility: Mapping[str, Mapping[str, Any]], R7MOD: Any) -> dict[str, Any]:
    require(len(fixed) == len(trace) == 24, COMMON_SUPPORT_BLOCK, "critic_lineage_count")
    n0_values, n0_scope = R7MOD.standardize([float(row["raw_gae"]) for row in fixed])
    rows = []
    for fixed_row, original, normalized in zip(fixed, trace, n0_values):
        decision_id = str(fixed_row["decision_id"])
        label = eligibility[decision_id]
        require(str(original["decision_id"]) == decision_id and bool(label["identity_chain_valid"]), COMMON_SUPPORT_BLOCK,
                f"critic_credit_identity={decision_id}")
        actor_eligible = bool(label["actor_eligible_e1"])
        direction = R7MOD.action_direction(str(original["selected_action_type"]), float(normalized)) if actor_eligible else "ACTOR_INELIGIBLE"
        rows.append({
            "decision_id": decision_id, "trajectory_id": fixed_row["trajectory_id"], "reward": fixed_row["reward"],
            "value": fixed_row["value"], "td_residual": fixed_row["td_residual"], "raw_gae": fixed_row["raw_gae"],
            "n0_normalized_advantage": normalized, "critic_target": fixed_row["critic_target"],
            "reward_gae_component": fixed_row["reward_gae_component"], "bootstrap_gae_component": fixed_row["critic_bootstrap_gae_component"],
            "eligibility_category": label["category"], "actor_eligible_e1": actor_eligible,
            "actor_gradient_direction_proxy": direction,
        })
    eligible_rows = [row for row in rows if row["actor_eligible_e1"]]
    return {
        "rows": rows, "n0_scope": n0_scope,
        "summary": {"value": describe([float(row["value"]) for row in rows]), "td_residual": describe([float(row["td_residual"]) for row in rows]),
                    "raw_gae": describe([float(row["raw_gae"]) for row in rows]), "n0_normalized_advantage": describe([float(row["n0_normalized_advantage"]) for row in rows]),
                    "critic_target": describe([float(row["critic_target"]) for row in rows]), "actor_eligible_rows": len(eligible_rows),
                    "actor_gradient_direction_proxy": dict(sorted(Counter(str(row["actor_gradient_direction_proxy"]) for row in eligible_rows).items()))},
    }


def critic_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_rows = {str(row["decision_id"]): row for row in left["rows"]}
    right_rows = {str(row["decision_id"]): row for row in right["rows"]}
    require(set(left_rows) == set(right_rows), COMMON_SUPPORT_BLOCK, "critic_pairing")
    pairs = []
    for decision_id in sorted(left_rows):
        c, d = left_rows[decision_id], right_rows[decision_id]
        require(c["reward"] == d["reward"] and c["reward_gae_component"] == d["reward_gae_component"], COMMON_SUPPORT_BLOCK,
                f"reward_lineage_changed={decision_id}")
        pairs.append({"decision_id": decision_id, "actor_eligible_e1": c["actor_eligible_e1"],
                      "value_delta_d_minus_c": float(d["value"]) - float(c["value"]),
                      "td_delta_d_minus_c": float(d["td_residual"]) - float(c["td_residual"]),
                      "raw_gae_delta_d_minus_c": float(d["raw_gae"]) - float(c["raw_gae"]),
                      "n0_delta_d_minus_c": float(d["n0_normalized_advantage"]) - float(c["n0_normalized_advantage"]),
                      "raw_gae_sign_changed": (float(c["raw_gae"]) > 0.0) != (float(d["raw_gae"]) > 0.0),
                      "n0_sign_changed": (float(c["n0_normalized_advantage"]) > 0.0) != (float(d["n0_normalized_advantage"]) > 0.0),
                      "actor_gradient_direction_changed": c["actor_gradient_direction_proxy"] != d["actor_gradient_direction_proxy"]})
    eligible = [row for row in pairs if row["actor_eligible_e1"]]
    return {"rows": pairs, "reward_lineage_exact": True,
            "summary": {"raw_gae_sign_changes": sum(bool(row["raw_gae_sign_changed"]) for row in pairs),
                        "n0_sign_changes": sum(bool(row["n0_sign_changed"]) for row in pairs),
                        "eligible_rows": len(eligible),
                        "eligible_raw_gae_sign_changes": sum(bool(row["raw_gae_sign_changed"]) for row in eligible),
                        "eligible_n0_sign_changes": sum(bool(row["n0_sign_changed"]) for row in eligible),
                        "eligible_gradient_direction_changes": sum(bool(row["actor_gradient_direction_changed"]) for row in eligible),
                        "abs_value_delta": describe([abs(float(row["value_delta_d_minus_c"])) for row in pairs]),
                        "abs_td_delta": describe([abs(float(row["td_delta_d_minus_c"])) for row in pairs]),
                        "abs_raw_gae_delta": describe([abs(float(row["raw_gae_delta_d_minus_c"])) for row in pairs]),
                        "eligible_abs_raw_gae_delta": describe([abs(float(row["raw_gae_delta_d_minus_c"])) for row in eligible])},
            "method": "stored reward/trajectory lineage plus frozen C/D value forwards; canonical TD/GAE recomputation only, no state transition"}


def zero_eligibility_policy(counts: Mapping[str, int]) -> dict[str, Any]:
    require(set(counts) == set(REPLICATES), COMMON_SUPPORT_BLOCK, "eligibility_replicate_scope")
    options = [
        {"option": "V0", "selected": False, "result": "REJECTED_ACTOR_STEP_ZERO_CANNOT_BE_CLASSIFIED_AS_TRAINED_POLICY_SUCCESS"},
        {"option": "V1", "selected": True, "result": "SELECTED_ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT",
         "rule": "actor_eligible_rows == 0 implies Actor optimizer steps = 0, classification ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT, and Critic-only diagnostic preservation; it is excluded from learned-Actor success claims"},
        {"option": "V2", "selected": False, "result": "PROHIBITED_POSTHOC_REWARD_BASED_WINDOW_SELECTION"},
    ]
    return {"eligible_rows_by_replicate": dict(counts), "options": options, "selected_option": "V1",
            "zero_eligible_replicates": [name for name, count in counts.items() if count == 0],
            "actor_not_trained_rule": options[1]["rule"]}


def select_repair(*, actor_classification: str, critic_material: bool, zero_eligible: bool,
                  initial_checkpoints: Mapping[str, str]) -> dict[str, Any]:
    options = [
        {"option": "S1", "selected": False,
         "scope": "one common Actor/Critic initial checkpoint pair across both environment seeds",
         "result": "REJECTED_DOES_NOT_MEASURE_CONFIRMED_ACTOR_INITIALIZATION_EFFECT_OR_CRITIC_SEED_EFFECT"},
        {"option": "S2", "selected": False,
         "scope": "Actor A/B × environment seed R1/R2 with one fixed Critic initialization",
         "result": "REJECTED_CRITIC_SEED_EFFECT_AND_ZERO_ELIGIBILITY_GUARD_REMAIN_UNRESOLVED"},
        {"option": "S3", "selected": False,
         "scope": "paired Actor/Critic A+C and B+D × both environment seeds, with per-cell Actor eligibility minimum",
         "result": "CANDIDATE"},
    ]
    selected = "S3" if actor_classification == "ACTOR_INIT_DOMINANT" and critic_material and zero_eligible else ""
    require(selected == "S3", SELECTION_BLOCK, "evidence_does_not_support_single_minimal_option")
    for option in options:
        option["selected"] = option["option"] == selected
        if option["selected"]:
            option["result"] = "SELECTED_MINIMAL_MULTI_FACTOR_CONTRACT"
    cells = []
    for pair, actor_rep, critic_rep in (("AC", "F1_R1", "F1_R1"), ("BD", "F1_R2", "F1_R2")):
        for environment in ("F1_R1", "F1_R2"):
            cells.append({"cell_id": f"{pair}×{environment}", "environment_seed": REPLICATES[environment]["environment_seed"],
                          "actor_initial_checkpoint_sha256": initial_checkpoints[actor_rep],
                          "critic_initial_checkpoint_sha256": initial_checkpoints[critic_rep],
                          "actor_seed": REPLICATES[actor_rep]["actor_seed"], "critic_seed": REPLICATES[critic_rep]["critic_seed"],
                          "optimizer_state": "FRESH_EMPTY_PER_CELL_NO_REUSE"})
    contract = {
        "contract_id": "LS3_BT8_R11_S3_PAIRED_ACTOR_CRITIC_ENV_ELIGIBILITY_V1", "selected_option": "S3",
        "seed_combination_table": cells,
        "checkpoint_rule": "both initial Actor/Critic pairs are included once on each environment seed; no behavior-favorable seed selection; no historical optimizer reuse",
        "actor_eligibility_minimum": {"minimum_actor_eligible_rows": 1,
                                       "zero_rule": "ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT; Actor steps=0; Critic-only diagnostic retention; no learned-Actor success or cross-policy conclusion"},
        "optimizer_step_accounting": {"ppo_epochs_per_cell": 3, "batch_and_minibatch_size": 24,
                                        "actor_steps_per_eligible_cell": 3, "actor_steps_per_zero_eligible_cell": 0,
                                        "critic_steps_per_cell": 3, "supplemental_steps": 0,
                                        "raw_steps_rule": "3 critic steps per cell plus 3 Actor steps only for cells meeting the frozen minimum; each cell is independently ledgered"},
        "same_support_review": "before/after frozen replay uses the same preserved review snapshots for each environment seed; no snapshot regeneration, support recomputation, or cross-seed candidate identity relabeling",
        "snapshot_checkpoint_binding": "strict SHA binding of initial/final checkpoints, actor config, T1 selector, review collection, train collection, candidate support/mask and credit identity before any later execution authorization",
        "immutable": {"reward_v2": "unchanged", "n0": "unchanged", "actor_critic_architecture": "unchanged", "no_assign_penalty": "none", "candidate_regeneration": False},
        "execution_authorized": False,
    }
    contract["sha256"] = canonical_sha256(contract)
    return {"options": options, "selected_contract": contract}


def write_block(*, root: Path, source: Mapping[str, Any], preflight: Mapping[str, Any], code: str) -> None:
    outputs = {
        "bt8r11_actor_support_factorization.json": {"not_run": True}, "bt8r11_critic_effect_audit.json": {"not_run": True},
        "bt8r11_zero_eligibility_policy.json": {"not_run": True}, "bt8r11_repair_option_comparison.json": {"not_run": True},
        "bt8r11_selected_seed_repair_contract.json": {"not_run": True}, "test_results.json": {"execution_counters": counter_template(), "hard_failures": [code], "warnings": []},
        "frozen_hash_before_after.json": {"not_run": True}, "bt8r11_evidence_preflight.json": preflight,
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED", "source_commit": source["source_commit"],
                               "hard_failures": [code], "warnings": [], "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, value in outputs.items(): dump(root / name, value)
    (root / "final_report.md").write_text(f"# BT8-R11 blocked\n\n- gate: `{code}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_credit_contract as CC
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as R1
    import run_h4m_ae_ls3_bt8_r5_same_support_review as R5
    import run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution as R6MOD
    import run_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility as R7MOD

    source = source_provenance(); root = artifact_root()
    require(not root.exists(), COMMON_SUPPORT_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    preflight: dict[str, Any] = {"source": source, "required_sources": {"r10": R10_SOURCE, "e1": E1_SOURCE, "r9": R9_SOURCE},
                                 "e1_contract_sha256": E1_CONTRACT_SHA256}
    try:
        r10_gate, e1_gate, r9_gate, r7_gate, f1_gate = (load_json(R10 / "gate_decision.json"), load_json(E1 / "gate_decision.json"),
                                                          load_json(R9 / "gate_decision.json"), load_json(R7 / "gate_decision.json"),
                                                          load_json(F1 / "gate_decision.json"))
        audits = {"r10": manifest_audit(R10), "e1": manifest_audit(E1), "r9": manifest_audit(R9), "r7": manifest_audit(R7),
                  "r6": manifest_audit(R6), "f1": manifest_audit(F1)}
        require(r10_gate.get("gate") == R10_GATE and r10_gate.get("source_commit") == R10_SOURCE, COMMON_SUPPORT_BLOCK, "r10_gate")
        require(e1_gate.get("gate") == E1_GATE and e1_gate.get("source_commit") == E1_SOURCE, COMMON_SUPPORT_BLOCK, "e1_gate")
        require(r9_gate.get("gate") == R9_GATE and r9_gate.get("source_commit") == R9_SOURCE, COMMON_SUPPORT_BLOCK, "r9_gate")
        require(r7_gate.get("gate") == R7_GATE and f1_gate.get("gate") == F1_GATE, COMMON_SUPPORT_BLOCK, "r7_or_f1_gate")
        require(all(audit["all_match"] for audit in audits.values()), COMMON_SUPPORT_BLOCK, "authoritative_manifest")
        r10_seed = load_json(R10 / "bt8r10_seed_divergence.json")
        r10_preflight = load_json(R10 / "bt8r10_evidence_preflight.json")
        e1_frozen = load_json(E1 / "frozen_hash_before_after.json")
        r7_contract = load_json(R7 / "bt8r7_repair_option_comparison.json")["selected_contract"]
        eligibility_document = load_json(R7 / "bt8r7_credit_eligibility_rows.json")
        traces = {"F1_R1": load_json(R6 / "bt8r6_replicate1_credit_trace.json")["rows"],
                  "F1_R2": load_json(R6 / "bt8r6_replicate2_credit_trace.json")["rows"]}
        require(r10_seed.get("classification") == "SEED_DIVERGENCE_PERSISTS", COMMON_SUPPORT_BLOCK, "r10_seed_result")
        require(r10_preflight.get("review_collection_digest") == "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e", COMMON_SUPPORT_BLOCK, "review_collection")
        require(e1_frozen.get("e1_contract_sha256") == E1_CONTRACT_SHA256 and r7_contract.get("sha256") == E1_CONTRACT_SHA256,
                COMMON_SUPPORT_BLOCK, "e1_contract")
        require(len(eligibility_document.get("rows", [])) == 48 and eligibility_document.get("actor_eligible_e1") == 9,
                COMMON_SUPPORT_BLOCK, "eligibility_rows")
        require(all(len(rows) == 24 for rows in traces.values()), COMMON_SUPPORT_BLOCK, "trace_count")
        review_collection = load_json(F1 / "bt8f1_review_snapshots" / "collection_manifest.json")
        training_collection = load_json(F1 / "bt8f1_training_snapshots" / "collection_manifest.json")
        for collection, expected_count in ((review_collection, 6), (training_collection, 48)):
            copy = dict(collection); digest = copy.pop("collection_digest", None)
            require(digest == FPS.canonical_sha256(copy) and int(collection["snapshot_count"]) == expected_count, COMMON_SUPPORT_BLOCK, "snapshot_collection")
        initial_manifest = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
        final_manifest = load_json(F1 / "bt8f1_final_checkpoint_manifest.json")
        checkpoint_paths = {replicate_id: F1 / "initial_checkpoints" / str(initial_manifest[replicate_id]["path"]) for replicate_id in REPLICATES}
        checkpoint_before = {replicate_id: sha256(path) for replicate_id, path in checkpoint_paths.items()}
        for replicate_id, spec in REPLICATES.items():
            initial = initial_manifest[replicate_id]
            require((initial["environment_seed"], initial["actor_init_seed"], initial["critic_init_seed"]) ==
                    (spec["environment_seed"], spec["actor_seed"], spec["critic_seed"]), COMMON_SUPPORT_BLOCK, f"initial_seed={replicate_id}")
            require(checkpoint_before[replicate_id] == initial["sha256"], COMMON_SUPPORT_BLOCK, f"initial_sha={replicate_id}")
        frozen_before = R6MOD.frozen_hashes(BT6)
        require(frozen_before == e1_frozen.get("after"), COMMON_SUPPORT_BLOCK, "frozen_binding")
        preflight |= {"manifests": audits, "r10_seed_divergence": r10_seed.get("classification"),
                      "review_collection_digest": review_collection["collection_digest"], "training_collection_digest": training_collection["collection_digest"],
                      "checkpoint_before": checkpoint_before, "frozen_hash_binding": True,
                      "source_lineage": source["source_lineage_descends_from_r10"], "source_only_local_commit": source["source_only_local_commit"]}
        require(source["source_lineage_descends_from_r10"] and source["source_only_local_commit"], COMMON_SUPPORT_BLOCK, "source_scope")
    except R11Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, code=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}"); return
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, code=COMMON_SUPPORT_BLOCK)
        print(f"[BLOCKED] {COMMON_SUPPORT_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root=root, source=source, preflight=preflight | {"mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()}, code=MPS_BLOCK)
        print(f"[BLOCKED] {MPS_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}"); return

    device = torch.device("mps:0"); execution = counter_template(); hard: list[str] = []
    try:
        snapshots: dict[str, list[Mapping[str, Any]]] = {key: [] for key in REPLICATES}
        config: Mapping[str, Any] | None = None
        for entry in review_collection["entries"]:
            payload = FPS.load_snapshot(F1 / "bt8f1_review_snapshots" / str(entry["relative_path"]))
            meta = payload["metadata"]; replicate_id = "F1_R1" if int(meta["seed"]) == 20260822 else "F1_R2"
            require(payload["snapshot_digest"] == entry["snapshot_digest"] and replicate_id in REPLICATES, COMMON_SUPPORT_BLOCK, "review_snapshot")
            config = meta["actor_config"] if config is None else config
            require(meta["actor_config"] == config and meta["actor_config_sha256"] == FPS.actor_config_sha256(config), COMMON_SUPPORT_BLOCK, "review_actor_config")
            require(meta["candidate_ids"] == meta["candidate_order"] and bool(payload["tensors"]["safe_mask"][0, :int(meta["selectable_pair_count"])].all().item()),
                    STRUCTURAL_BLOCK, "review_support")
            snapshots[replicate_id].append(payload)
        require(config is not None and all(len(rows) == 3 for rows in snapshots.values()) and TIE.TIE_BREAK_CONTRACT_ID == T1_CONTRACT,
                COMMON_SUPPORT_BLOCK, "review_matrix")
        actors = {label: R5.load_actor(checkpoint=checkpoint_paths[replicate_id], config=config, H=H, device=device)
                  for label, replicate_id in ACTORS.items()}
        critics = {label: load_critic(checkpoint=checkpoint_paths[replicate_id], config=config, JL=JL, device=device)
                   for label, replicate_id in CRITICS.items()}
        modules_before = {"actors": {label: module_digest(model) for label, model in actors.items()},
                          "critics": {label: module_digest(model) for label, model in critics.items()}}
        actor_matrix: dict[str, list[dict[str, Any]]] = {}
        for actor_label, actor in actors.items():
            for support_id in ("F1_R1", "F1_R2"):
                key = f"{actor_label}×{'R1' if support_id == 'F1_R1' else 'R2'}"
                actor_matrix[key] = []
                for payload in snapshots[support_id]:
                    actor_matrix[key].append(actor_row(actor=actor, payload=payload, R1=R1, H=H, TIE=TIE, device=device,
                                                       actor_label=actor_label, support_label=support_id))
                    execution["actor_support_forward"] += 1
        actor_factor = actor_support_factorization(actor_matrix)
        eligibility = {str(row["decision_id"]): row for row in eligibility_document["rows"]}
        payload_by_id: dict[str, Mapping[str, Any]] = {}
        for entry in training_collection["entries"]:
            payload = FPS.load_snapshot(F1 / "bt8f1_training_snapshots" / str(entry["relative_path"]))
            decision_id = str(payload["metadata"]["decision_id"])
            require(decision_id not in payload_by_id and decision_id in eligibility and payload["snapshot_digest"] == entry["snapshot_digest"],
                    COMMON_SUPPORT_BLOCK, "training_snapshot")
            payload_by_id[decision_id] = payload
        require(len(payload_by_id) == 48, COMMON_SUPPORT_BLOCK, "training_payload_count")
        critic_values: dict[str, dict[str, float]] = {label: {} for label in critics}
        for label, critic in critics.items():
            for decision_id, payload in payload_by_id.items():
                critic_values[label][decision_id] = critic_value(critic=critic, payload=payload, JL=JL, device=device)
                execution["critic_forward"] += 1
        torch.mps.synchronize()
        critic_audit: dict[str, Any] = {"lineages": {}, "comparisons": {}}
        for lineage_id, trace in traces.items():
            by_critic = {}
            for label, values in critic_values.items():
                fixed = R7MOD.fixed_gae(trace, values, gamma=CC.GAMMA, lam=CC.GAE_LAMBDA)
                by_critic[label] = critic_lineage_rows(fixed=fixed, trace=trace, eligibility=eligibility, R7MOD=R7MOD)
            critic_audit["lineages"][lineage_id] = by_critic
            critic_audit["comparisons"][lineage_id] = critic_comparison(by_critic["C"], by_critic["D"])
        eligible_counts = {replicate_id: sum(bool(eligibility[str(row["decision_id"])]["actor_eligible_e1"]) for row in trace)
                           for replicate_id, trace in traces.items()}
        zero_policy = zero_eligibility_policy(eligible_counts)
        critic_material = any(int(value["summary"]["eligible_raw_gae_sign_changes"]) > 0 or int(value["summary"]["eligible_gradient_direction_changes"]) > 0
                              or float(value["summary"]["eligible_abs_raw_gae_delta"]["max"] or 0.0) > 0.0
                              for value in critic_audit["comparisons"].values())
        repair = select_repair(actor_classification=str(actor_factor["classification"]), critic_material=critic_material,
                               zero_eligible=bool(zero_policy["zero_eligible_replicates"]), initial_checkpoints=checkpoint_before)
        classification = "B_MULTI_FACTOR_SEED_REPAIR_CONTRACT_READY" if repair["selected_contract"]["selected_option"] == "S3" else ""
        require(bool(classification), SELECTION_BLOCK, "classification")
        modules_after = {"actors": {label: module_digest(model) for label, model in actors.items()},
                         "critics": {label: module_digest(model) for label, model in critics.items()}}
        execution["parameter_mutation"] = int(modules_before != modules_after)
        checkpoint_after = {replicate_id: sha256(path) for replicate_id, path in checkpoint_paths.items()}
        execution["checkpoint_mutation"] = sum(checkpoint_before[key] != checkpoint_after[key] for key in checkpoint_before)
        frozen_after = R6MOD.frozen_hashes(BT6)
        forbidden = ("training", "optimizer_step", "backward", "autograd_grad", "causal_rollout", "simulator_step", "candidate_generation",
                     "candidate_regeneration", "local_search_rerun", "zero_loss_reevaluation", "reward_recomputation", "checkpoint_write",
                     "parameter_mutation", "checkpoint_mutation", "review_optimizer_rows", "future_leakage", "nan_or_inf", "test6_access", "github_push")
        if any(execution[name] != 0 for name in forbidden): hard.append(STRUCTURAL_BLOCK)
        if frozen_before != frozen_after: hard.append("FROZEN_HASH_MUTATION")
        if execution["actor_support_forward"] != 12 or execution["critic_forward"] != 96: hard.append("FORWARD_ACCOUNTING_MISMATCH")
        if actor_factor["classification"] not in {"ACTOR_INIT_DOMINANT", "SUPPORT_DOMINANT", "ACTOR_SUPPORT_INTERACTION", "MIXED"}: hard.append(SELECTION_BLOCK)
        gate = PASS_GATE if not hard else hard[0]
        if hard: classification = "BLOCKED"
        outputs = {
            "bt8r11_evidence_preflight.json": preflight | {"mps_built": True, "mps_available": True, "device": "mps:0",
                                                              "t1_selector_sha256": sha256(ROOT / "joint_assignment_frozen_tie_break.py")},
            "bt8r11_actor_support_factorization.json": actor_factor,
            "bt8r11_critic_effect_audit.json": {"critic_material_on_eligible_credit": critic_material, "audit": critic_audit,
                                                  "method": "C/D frozen forward × stored R1/R2 reward/trajectory lineages; N0 is recomputed per 24-row lineage only for audit"},
            "bt8r11_zero_eligibility_policy.json": zero_policy,
            "bt8r11_repair_option_comparison.json": repair["options"],
            "bt8r11_selected_seed_repair_contract.json": repair["selected_contract"],
            "test_results.json": {"execution_counters": execution, "hard_failures": hard, "warnings": [], "github_push_performed": False,
                                  "execution_authorized": False},
            "frozen_hash_before_after.json": {"before": frozen_before, "after": frozen_after, "all_unchanged": frozen_before == frozen_after,
                                                 "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after,
                                                 "checkpoint_unchanged": checkpoint_before == checkpoint_after, "e1_contract_sha256": E1_CONTRACT_SHA256},
            "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
                                   "next_step": "separate authorization gate for the frozen S3 contract" if not hard else "STOP"},
        }
        for name, value in outputs.items(): dump(root / name, value)
        (root / "final_report.md").write_text(
            f"# BT8-R11 final report\n\n- gate: `{gate}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n"
            f"- Actor/support factor: `{actor_factor['classification']}`\n- Critic material on eligible credit: `{critic_material}`\n"
            f"- selected repair: `{repair['selected_contract']['contract_id']}`\n\n"
            "This stage is audit and design only. It grants no later training or simulator execution authority.\n", encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                       "file_sha256": manifest, "github_push_performed": False})
        (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
        print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}"); print(f"classification: {classification}"); print(f"artifact: {root.relative_to(PROJECT)}")
    except R11Error as exc:
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [str(exc)]}, code=exc.code)
        print(f"[BLOCKED] {exc.code}"); print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, preflight=preflight | {"hard_failures": [repr(exc)]}, code=STRUCTURAL_BLOCK)
        print(f"[BLOCKED] {STRUCTURAL_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
