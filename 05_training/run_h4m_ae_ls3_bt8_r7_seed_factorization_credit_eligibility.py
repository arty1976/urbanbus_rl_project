#!/usr/bin/env python3
"""BT8-R7: read-only Actor/Critic seed factorization and credit eligibility gate.

The runner uses only persisted BT8-F1/R6 checkpoints, snapshots, and credit
rows.  A 2×2 Cartesian table is observational: Actor and Critic forwards are
evaluated independently on the same stored inputs, never used to create a new
transition, reward, rollout, optimizer step, or checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R7"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R7_MINIMAL_ACTOR_CRITIC_SEED_FACTORIZATION_AND_CREDIT_ELIGIBILITY_SELECTION_COMPLETE"
BLOCK = "BLOCKED_INSUFFICIENT_SEED_FACTORIZATION_EVIDENCE"
CREDIT_BLOCK = "BLOCKED_CREDIT_ELIGIBILITY_UNRESOLVED"
R6_SOURCE = "f18af102031dbc41ef9255555a9b6ca4a2de1b03"
R6_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R6_SEED_DIVERGENT_CREDIT_TO_LOGIT_ATTRIBUTION_AUDIT_COMPLETE"
R6_MANIFEST_SHA256 = "c64fdafbf1e4cd8c3c89e8915b103d68d5169c52c109e61c6c055a4106b3bd5b"
R6_GATE_SHA256 = "a515115fce6eadf1f0e9fbc0bb24860a0ce425d101954e98c485fc8bf5facbb7"
R6_R1_TRACE_SHA256 = "ad31d5aa98fb3d88127ac11716af06573f5406919100d4f4f4793f13af823a04"
R6_R2_TRACE_SHA256 = "69dcc6d8af09e8a17615add877463d61612f1f0cdfc9946eea96b1ee00639090"
R6_ATTRIBUTION_SHA256 = "1d6c510cca74199c2db3a5b3de4a70fccf1dcb34ef19296f5e4d6d65b3c1c681"
F1_SOURCE = "53c54bd5b18045b4eb3fb055a2aed0ae8bf169dd"
F1_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
NEXT_GATE = "H4M-AE-R9.8-LS3-BT8-R8_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution_20260824_234227+09:00"
F1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_20260823_135127+09:00"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility.py",
    "05_training/test_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility.py",
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
ACTOR_IDS = {"A": "F1_R1", "B": "F1_R2"}
CRITIC_IDS = {"C": "F1_R1", "D": "F1_R2"}
ELIGIBLE_CATEGORIES = {
    "DIRECT_CANDIDATE_REWARD",
    "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD",
    "NO_ASSIGN_REWARD_SUPPORTED",
}


class R7Error(RuntimeError):
    pass


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R7Error(f"{code}:{detail}" if detail else code)


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
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), "AUTHORITATIVE_EVIDENCE_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff", "--name-only", f"{R6_SOURCE}..HEAD"]).splitlines() if row]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_lineage_descends_from_r6": git(["merge-base", R6_SOURCE, "HEAD"]) == R6_SOURCE,
        "changed_files_since_r6": changed,
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


def sign(value: float) -> str:
    return "POSITIVE" if value > 0.0 else "NEGATIVE" if value < 0.0 else "ZERO"


def standardize(values: Sequence[float]) -> tuple[list[float], dict[str, Any]]:
    raw = [float(value) for value in values]
    if not raw:
        return [], {"row_count": 0, "mean": None, "std": None, "mode": "NO_ROWS"}
    mean = sum(raw) / len(raw)
    std = max((sum((value - mean) ** 2 for value in raw) / len(raw)) ** 0.5, 1e-8)
    return [(value - mean) / std for value in raw], {"row_count": len(raw), "mean": mean, "std": std,
                                                       "formula": "(raw - per-scope mean) / max(per-scope std, 1e-8)"}


def action_direction(action_type: str, advantage: float) -> str:
    if advantage == 0.0:
        return "ZERO_ADVANTAGE"
    if action_type == "CANDIDATE":
        return "CANDIDATE_RELATIVE_TO_NO_ASSIGN" if advantage > 0.0 else "NO_ASSIGN_RELATIVE_TO_CANDIDATE"
    if action_type == "NO_ASSIGN":
        return "NO_ASSIGN_RELATIVE_TO_CANDIDATE" if advantage > 0.0 else "CANDIDATE_RELATIVE_TO_NO_ASSIGN"
    return "UNRESOLVED_ACTION_TYPE"


def eligibility_category(*, action_type: str, reward: float, reward_component: float,
                         identity_chain_valid: bool, raw_gae: float) -> str:
    if not identity_chain_valid:
        return "INSUFFICIENT_IDENTITY_CREDIT"
    if action_type == "CANDIDATE" and reward != 0.0:
        return "DIRECT_CANDIDATE_REWARD"
    if action_type == "CANDIDATE" and reward == 0.0 and reward_component != 0.0:
        return "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD"
    if action_type == "NO_ASSIGN" and reward_component != 0.0:
        return "NO_ASSIGN_REWARD_SUPPORTED"
    if reward_component == 0.0 and raw_gae != 0.0:
        return "CRITIC_ONLY_NO_REWARD_ANCESTRY"
    return "INSUFFICIENT_IDENTITY_CREDIT"


def fixed_gae(rows: Sequence[Mapping[str, Any]], values: Mapping[str, float], *, gamma: float,
              lam: float) -> list[dict[str, Any]]:
    """Re-evaluate only the critic side on a persisted reward trajectory."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["trajectory_id"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    gamma_k = float(gamma) ** 2
    for trajectory_id, group in groups.items():
        ordered = sorted(group, key=lambda row: int(str(row["decision_id"]).rsplit(":", 1)[1]))
        running, reward_running, bootstrap_running = 0.0, 0.0, 0.0
        for position in range(len(ordered) - 1, -1, -1):
            row = ordered[position]
            decision_id = str(row["decision_id"])
            nonterminal = 1.0 if bool(row["nonterminal"]) else 0.0
            next_value = float(values[str(ordered[position + 1]["decision_id"])]) if nonterminal else 0.0
            value = float(values[decision_id])
            reward = float(row["reward"])
            td = reward + gamma_k * next_value * nonterminal - value
            bootstrap_td = gamma_k * next_value * nonterminal - value
            running = td + gamma_k * float(lam) * nonterminal * running
            reward_running = reward + gamma_k * float(lam) * nonterminal * reward_running
            bootstrap_running = bootstrap_td + gamma_k * float(lam) * nonterminal * bootstrap_running
            out[decision_id] = {
                "decision_id": decision_id, "trajectory_id": trajectory_id, "value": value,
                "next_value": next_value, "reward": reward, "td_residual": td,
                "raw_gae": running, "critic_target": running + value,
                "reward_gae_component": reward_running,
                "critic_bootstrap_gae_component": bootstrap_running,
                "raw_additivity_residual": running - reward_running - bootstrap_running,
            }
    return [out[str(row["decision_id"])] for row in rows]


def summarize_fixed_gae(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "value": describe([row["value"] for row in rows]),
        "td_residual": describe([row["td_residual"] for row in rows]),
        "raw_gae": describe([row["raw_gae"] for row in rows]),
        "critic_target": describe([row["critic_target"] for row in rows]),
        "reward_component": describe([row["reward_gae_component"] for row in rows]),
        "bootstrap_component": describe([row["critic_bootstrap_gae_component"] for row in rows]),
        "max_abs_additivity_residual": max((abs(float(row["raw_additivity_residual"])) for row in rows), default=0.0),
        "comparison_rule": "same persisted reward trajectory; no counterfactual transition or invented tolerance",
    }


def classify_factorization(actor: Mapping[str, Any], critic: Mapping[str, Any],
                           lineages: Mapping[str, Any]) -> dict[str, str]:
    actor_effect = "ACTOR_INIT_DOMINANT" if actor["A_candidate"] == actor["canonical_rows"] \
        and actor["B_no_assign"] == actor["canonical_rows"] else "MIXED_OR_INDETERMINATE"
    critic_effect = "CRITIC_INIT_DOMINANT" if critic["C_all_negative"] and critic["D_all_positive"] \
        else "MIXED_OR_INDETERMINATE"
    interaction = "ACTOR_CRITIC_INTERACTION" if actor_effect == "ACTOR_INIT_DOMINANT" \
        and critic_effect == "CRITIC_INIT_DOMINANT" and lineages["R1_reward_rows"] > 0 \
        and lineages["R2_reward_rows"] == 0 else "MIXED_OR_INDETERMINATE"
    return {"actor_seed_effect": actor_effect, "critic_seed_effect": critic_effect,
            "interaction": interaction}


def select_contract(eligibility_rows: Sequence[Mapping[str, Any]], normalization: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row["category"]) for row in eligibility_rows)
    eligible = [row for row in eligibility_rows if bool(row["actor_eligible_e1"])]
    ineligible = [row for row in eligibility_rows if not bool(row["actor_eligible_e1"])]
    identity_pass = all(bool(row["identity_chain_valid"]) for row in eligibility_rows)
    e0 = {
        "option": "E0", "actor_loss_rows": len(eligibility_rows), "critic_loss_rows": len(eligibility_rows),
        "critic_only_actor_rows": counts["CRITIC_ONLY_NO_REWARD_ANCESTRY"], "selected": False,
        "result": "REJECTED_CRITIC_ONLY_ROWS_WOULD_REMAIN_IN_ACTOR_LOSS",
    }
    e1 = {
        "option": "E1", "actor_loss_rows": len(eligible), "critic_loss_rows": len(eligibility_rows),
        "excluded_actor_rows": len(ineligible), "retained_categories": sorted(ELIGIBLE_CATEGORIES),
        "excluded_categories": ["CRITIC_ONLY_NO_REWARD_ANCESTRY", "INSUFFICIENT_IDENTITY_CREDIT"],
        "identity_chain_pass": identity_pass,
        "ratio_semantics": "retained-row old_log_prob, frozen legal support, ratio, clipping, and action distribution are unchanged; ineligible rows receive actor weight 0 before aggregation",
        "critic_semantics": "all finite rows remain in critic loss",
        "normalization_decision": "N0_CURRENT_FULL_REPLICATE_NORMALIZATION_RETAINED; N1 and N2 are audit counterfactuals only",
        "selected": identity_pass and bool(eligible),
        "result": "SELECTED_MINIMUM_NONSTRUCTURAL_CONTRACT" if identity_pass and eligible else "NOT_ELIGIBLE",
    }
    e2 = {
        "option": "E2", "selected": False,
        "result": "REJECTED_REQUIRES_ACTOR_ROUTE_OR_STRUCTURE_CHANGE_WITHOUT_RATIO_EQUIVALENCE_EVIDENCE",
        "reason": "hard locks prohibit Actor/Critic structure changes in this selection gate",
    }
    require(e1["selected"] is True, "CREDIT_ELIGIBILITY_UNRESOLVED")
    contract = {
        "contract_id": "LS3_BT8_R7_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_V1",
        "selected_option": "E1",
        "actor_eligible_categories": sorted(ELIGIBLE_CATEGORIES),
        "actor_excluded_categories": ["CRITIC_ONLY_NO_REWARD_ANCESTRY", "INSUFFICIENT_IDENTITY_CREDIT"],
        "critic_rows": "all finite persisted rows", "normalization": "N0 retained; no normalizer change selected",
        "invariants": {
            "reward_v2": "unchanged", "legal_support": "frozen", "old_log_prob": "unchanged for retained rows",
            "ppo_ratio_and_clipping": "unchanged for retained rows", "no_assign_penalty": "none",
            "actor_critic_architecture": "unchanged", "candidate_regeneration": False,
        },
        "implementation_boundary": "eligibility mask is applied only to Actor-loss aggregation; no actor forward, logits, masks, rewards, values, or critic loss are changed by this contract",
    }
    contract["sha256"] = canonical_sha256(contract)
    return {"options": [e0, e1, e2], "selected_contract": contract,
            "normalization_evidence": normalization}


def artifact_root(BT6: Any) -> Path:
    stamp = BT6.now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility_{stamp}"


def write_block(root: Path, source: Mapping[str, Any], binding: Mapping[str, Any], code: str, detail: str) -> None:
    counters = {"training": 0, "optimizer_step": 0, "backward": 0, "autograd_grad": 0,
                "causal_rollout": 0, "simulator_step": 0, "candidate_generation": 0,
                "candidate_regeneration": 0, "local_search_rerun": 0, "zero_loss_reevaluation": 0,
                "reward_recomputation": 0, "parameter_mutation": 0, "checkpoint_write": 0,
                "review_data_used": 0, "test6_access": 0, "mps_forward": 0}
    outputs = {
        "bt8r7_seed_factorization.json": {"not_run": True, "binding": binding},
        "bt8r7_advantage_normalization_audit.json": {"not_run": True},
        "bt8r7_credit_eligibility_rows.json": {"not_run": True},
        "bt8r7_repair_option_comparison.json": {"not_run": True},
        "bt8r7_selected_minimal_contract.json": {"not_run": True},
        "test_results.json": {"execution_counters": counters, "hard_failures": [detail]},
        "frozen_hash_before_after.json": {"not_run": True},
        "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                               "source_commit": source.get("source_commit"), "hard_failures": [detail],
                               "warnings": [], "global_locks": LOCKS, "next_step": "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    (root / "final_report.md").write_text(f"# BT8-R7 blocked\n\n{detail}\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source.get("source_commit"),
                                   "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_credit_contract as CC
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_f1_bounded_training as F1MOD
    import run_h4m_ae_ls3_bt8_r5_same_support_review as R5MOD
    import run_h4m_ae_ls3_bt8_r6_seed_credit_logit_attribution as R6MOD

    source = provenance()
    frozen_before = R6MOD.frozen_hashes(BT6)
    root = artifact_root(BT6)
    require(not root.exists(), "APPEND_ONLY_ARTIFACT_COLLISION")
    root.mkdir(parents=True)
    binding: dict[str, Any] = {"stage": STAGE, "source": source, "r6_source": R6_SOURCE,
                               "f1_source": F1_SOURCE}
    try:
        r6_gate = load_json(R6 / "gate_decision.json")
        f1_gate = load_json(F1 / "gate_decision.json")
        r6_manifest = R6MOD.verify_manifest(R6)
        f1_manifest = R6MOD.verify_manifest(F1)
        r6_r1 = load_json(R6 / "bt8r6_replicate1_credit_trace.json")
        r6_r2 = load_json(R6 / "bt8r6_replicate2_credit_trace.json")
        r6_attr = load_json(R6 / "bt8r6_credit_to_logit_attribution.json")
        r6_test = load_json(R6 / "test_results.json")
        r6_frozen = load_json(R6 / "frozen_hash_before_after.json")
        collection = load_json(F1 / "bt8f1_training_snapshots" / "collection_manifest.json")
        credit = load_json(F1 / "bt8f1_candidate_plan_credit_audit.json")
        initial_manifest = load_json(F1 / "bt8f1_initial_checkpoint_manifest.json")
        final_manifest = load_json(F1 / "bt8f1_final_checkpoint_manifest.json")
        f1_learning = load_json(F1 / "bt8f1_learning_signal_audit.json")
        f1_leakage = load_json(F1 / "bt8f1_train_review_leakage_audit.json")
        binding.update({
            "r6_gate": r6_gate.get("gate") == R6_GATE and r6_gate.get("source_commit") == R6_SOURCE,
            "r6_classification": r6_gate.get("classification") == "MIXED_OR_INDETERMINATE",
            "f1_gate": f1_gate.get("gate") == F1_GATE and f1_gate.get("source_commit") == F1_SOURCE,
            "source_lineage": source["source_lineage_descends_from_r6"],
            "source_only_local_commit": source["source_only_local_commit"],
            "r6_manifest": r6_manifest["all_match"], "f1_manifest": f1_manifest["all_match"],
            "pinned_r6_files": {
                "manifest": sha256(R6 / "manifest.json") == R6_MANIFEST_SHA256,
                "gate": sha256(R6 / "gate_decision.json") == R6_GATE_SHA256,
                "r1_trace": sha256(R6 / "bt8r6_replicate1_credit_trace.json") == R6_R1_TRACE_SHA256,
                "r2_trace": sha256(R6 / "bt8r6_replicate2_credit_trace.json") == R6_R2_TRACE_SHA256,
                "attribution": sha256(R6 / "bt8r6_credit_to_logit_attribution.json") == R6_ATTRIBUTION_SHA256,
            },
        })
        require(all(binding[key] for key in ("r6_gate", "r6_classification", "f1_gate", "source_lineage",
                                             "source_only_local_commit", "r6_manifest", "f1_manifest")),
                "AUTHORITATIVE_GATE_OR_SOURCE_MISMATCH")
        require(all(binding["pinned_r6_files"].values()), "PINNED_R6_ARTIFACT_HASH_MISMATCH")
        require(r6_attr["first_divergence"] == "INITIAL_ACTOR_ACTION_FAMILY_SPLIT_ON_EXACT_MATCHED_INPUTS_BEFORE_REWARD",
                "R6_FIRST_DIVERGENCE_MISMATCH")
        require(r6_frozen.get("all_unchanged") is True and not r6_test.get("hard_failures"), "R6_INTEGRITY_MISMATCH")
        require(all(value == 0 for key, value in r6_test["execution_counters"].items()
                    if key not in {"training_snapshot_forward_replays", "review_functional_forward_replays"}),
                "R6_FORBIDDEN_COUNTER_NONZERO")
        require(credit.get("verified") is True and credit.get("mismatches") == 0 and len(credit.get("rows", [])) == 48,
                "F1_CREDIT_IDENTITY_BINDING_MISMATCH")
        require(f1_learning.get("future_leakage") == 0 and f1_leakage.get("train_review_overlap") == 0,
                "F1_LEAKAGE_BINDING_MISMATCH")
        digest_input = dict(collection)
        collection_digest = digest_input.pop("collection_digest", None)
        require(collection_digest == FPS.canonical_sha256(digest_input) == R6MOD.F1_TRAINING_COLLECTION_DIGEST,
                "TRAINING_COLLECTION_DIGEST_MISMATCH")
        require(collection.get("snapshot_count") == len(collection.get("entries", [])) == 48,
                "TRAINING_SNAPSHOT_COUNT_MISMATCH")
    except Exception as exc:  # noqa: BLE001
        detail = f"AUTHORITY_BINDING_FAILURE:{type(exc).__name__}:{exc}"
        write_block(root, source, binding, BLOCK, detail)
        print(f"[BLOCKED] {BLOCK}\nartifact: {root.relative_to(PROJECT)}")
        return

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        write_block(root, source, binding, BLOCK, "MPS_REQUIRED_FOR_BOUND_CHECKPOINT_REPLAY")
        print(f"[BLOCKED] {BLOCK}\nartifact: {root.relative_to(PROJECT)}")
        return
    device = torch.device("mps:0")
    traces = {"F1_R1": r6_r1["rows"], "F1_R2": r6_r2["rows"]}
    credit_by_id = {str(row["decision_id"]): row for row in credit["rows"]}
    try:
        require(all(len(rows) == 24 for rows in traces.values()), "R6_TRACE_ROW_COUNT_MISMATCH")
        require(set().union(*(set(str(row["decision_id"]) for row in rows) for rows in traces.values())) == set(credit_by_id),
                "R6_F1_DECISION_SET_MISMATCH")
        payload_by_id: dict[str, Mapping[str, Any]] = {}
        tensor_to_payload: dict[str, Mapping[str, Any]] = {}
        config = None
        tensor_multiset: dict[str, list[str]] = {key: [] for key in REPLICATES}
        for entry in collection["entries"]:
            path = F1 / "bt8f1_training_snapshots" / entry["relative_path"]
            payload = FPS.load_snapshot(path)
            manifest = load_json(path / "snapshot_manifest.json")
            decision_id = str(payload["metadata"]["decision_id"])
            replicate_id = decision_id.split(":")[1]
            require(payload["snapshot_digest"] == entry["snapshot_digest"] and decision_id not in payload_by_id,
                    "TRAINING_SNAPSHOT_IDENTITY_MISMATCH", decision_id)
            require(sha256(path / manifest["tensor_binary"]) == manifest["tensor_binary_sha256"],
                    "TRAINING_TENSOR_HASH_MISMATCH", decision_id)
            payload_by_id[decision_id] = payload
            tensor_sha = str(manifest["tensor_binary_sha256"])
            tensor_to_payload.setdefault(tensor_sha, payload)
            tensor_multiset[replicate_id].append(tensor_sha)
            config = payload["metadata"]["actor_config"] if config is None else config
            require(payload["metadata"]["actor_config"] == config, "ACTOR_CONFIG_MISMATCH")
        require(len(payload_by_id) == 48 and len(tensor_to_payload) == 6, "MATCHED_INPUT_SCOPE_MISMATCH")
        require(Counter(tensor_multiset["F1_R1"]) == Counter(tensor_multiset["F1_R2"]), "SEED_INPUT_MULTISET_MISMATCH")
        require(config is not None, "ACTOR_CONFIG_MISSING")
        checkpoint_before = R6MOD.checkpoint_bindings(initial_manifest, final_manifest)
        for replicate_id, seed in REPLICATES.items():
            initial = initial_manifest[replicate_id]
            require((initial["environment_seed"], initial["actor_init_seed"], initial["critic_init_seed"])
                    == (seed["environment_seed"], seed["actor_seed"], seed["critic_seed"]),
                    "INITIAL_SEED_BINDING_MISMATCH", replicate_id)
    except Exception as exc:  # noqa: BLE001
        detail = f"MATCHED_INPUT_PREP_FAILURE:{type(exc).__name__}:{exc}"
        write_block(root, source, binding, BLOCK, detail)
        print(f"[BLOCKED] {BLOCK}\nartifact: {root.relative_to(PROJECT)}")
        return

    def load_critic(checkpoint: Path) -> torch.nn.Module:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actor_config = config
        critic = JL.JointAssignmentCritic(global_dim=int(actor_config["global_dim"]), demand_dim=int(actor_config["demand_dim"]),
                                           agent_dim=int(actor_config["agent_dim"]),
                                           safe_summary_dim=1 + 2 * int(actor_config["candidate_dim"]))
        critic.load_state_dict(payload["critic"], strict=True)
        return critic.to(device).eval()

    def critic_value(critic: torch.nn.Module, payload: Mapping[str, Any]) -> float:
        tensors = {name: value.to(device) for name, value in payload["tensors"].items()}
        with torch.no_grad():
            value = critic(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                           agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                           safe_summary=JL.safe_set_summary(tensors["candidate_feats"], tensors["safe_mask"]))
        return float(value[0].detach().cpu())

    actors, critics = {}, {}
    for label, replicate_id in ACTOR_IDS.items():
        actors[label] = R5MOD.load_actor(checkpoint=F1 / "initial_checkpoints" / initial_manifest[replicate_id]["path"],
                                          config=config, H=H, device=device)
    for label, replicate_id in CRITIC_IDS.items():
        critics[label] = load_critic(F1 / "initial_checkpoints" / initial_manifest[replicate_id]["path"])
    module_before = {"actors": {label: R5MOD.module_digest(model) for label, model in actors.items()},
                     "critics": {label: R5MOD.module_digest(model) for label, model in critics.items()}}
    counters = {"training": 0, "optimizer_step": 0, "backward": 0, "autograd_grad": 0,
                "causal_rollout": 0, "simulator_step": 0, "candidate_generation": 0,
                "candidate_regeneration": 0, "local_search_rerun": 0, "zero_loss_reevaluation": 0,
                "reward_recomputation": 0, "parameter_mutation": 0, "checkpoint_write": 0,
                "review_data_used": 0, "test6_access": 0, "nan_or_inf": 0,
                "actor_mps_forward": 0, "critic_mps_forward": 0}
    hard: list[str] = []
    canonical_payloads = [tensor_to_payload[key] for key in sorted(tensor_to_payload)]
    actor_metrics: dict[str, list[dict[str, Any]]] = {label: [] for label in actors}
    for label, actor in actors.items():
        for payload in canonical_payloads:
            row = R6MOD.replay_actor(actor, payload, F1MOD, H, TIE, device)
            counters["actor_mps_forward"] += 1
            counters["nan_or_inf"] += int(not row["finite"])
            actor_metrics[label].append({"tensor_binary_sha256": next(key for key, value in tensor_to_payload.items() if value is payload),
                                         "selected_identity": row["selected_identity"], "action_family": "NO_ASSIGN" if row["selected_identity"] == "NO_ASSIGN" else "CANDIDATE",
                                         "no_assign_minus_best_candidate": row["no_assign_minus_top_candidate"],
                                         "candidate_only_top1_top2_margin": row["pair_top1_top2_margin"],
                                         "entropy": row["entropy"], "finite": row["finite"]})
    critic_values: dict[str, dict[str, float]] = {label: {} for label in critics}
    for label, critic in critics.items():
        for decision_id, payload in payload_by_id.items():
            value = critic_value(critic, payload)
            counters["critic_mps_forward"] += 1
            counters["nan_or_inf"] += int(not math.isfinite(value))
            critic_values[label][decision_id] = value
    torch.mps.synchronize()

    def actor_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"canonical_rows": len(rows), "action_family": dict(Counter(str(row["action_family"]) for row in rows)),
                "no_assign_minus_best_candidate": describe([float(row["no_assign_minus_best_candidate"]) for row in rows]),
                "candidate_only_top1_top2_margin": describe([float(row["candidate_only_top1_top2_margin"]) for row in rows
                                                               if row["candidate_only_top1_top2_margin"] is not None]),
                "entropy": describe([float(row["entropy"]) for row in rows])}

    actor_summary_a, actor_summary_b = actor_summary(actor_metrics["A"]), actor_summary(actor_metrics["B"])
    actor_evidence = {"canonical_rows": len(canonical_payloads), "A": actor_summary_a, "B": actor_summary_b,
                      "A_candidate": actor_summary_a["action_family"].get("CANDIDATE", 0),
                      "B_no_assign": actor_summary_b["action_family"].get("NO_ASSIGN", 0),
                      "same_input_tensor_count": len(canonical_payloads)}
    fixed_lineages: dict[str, dict[str, Any]] = {}
    for critic_label, values in critic_values.items():
        fixed_lineages[critic_label] = {}
        for lineage_id, rows in traces.items():
            score_rows = fixed_gae(rows, values, gamma=CC.GAMMA, lam=CC.GAE_LAMBDA)
            fixed_lineages[critic_label][lineage_id] = {"rows": score_rows, "summary": summarize_fixed_gae(score_rows)}
    c_values = list(critic_values["C"].values())
    d_values = list(critic_values["D"].values())
    critic_evidence = {"C": describe(c_values), "D": describe(d_values),
                       "C_all_negative": all(value < 0.0 for value in c_values),
                       "D_all_positive": all(value > 0.0 for value in d_values),
                       "same_input_decisions": len(payload_by_id)}
    lineage_evidence = {"R1_reward_rows": sum(float(row["reward"]) != 0.0 for row in traces["F1_R1"]),
                        "R2_reward_rows": sum(float(row["reward"]) != 0.0 for row in traces["F1_R2"])}
    factor_class = classify_factorization(actor_evidence, critic_evidence, lineage_evidence)
    factor_matrix = []
    for actor_label in ("A", "B"):
        for critic_label in ("C", "D"):
            factor_matrix.append({"actor": actor_label, "critic": critic_label,
                                  "actor_same_input": actor_summary(actor_metrics[actor_label]),
                                  "critic_on_preserved_R1_reward_lineage": fixed_lineages[critic_label]["F1_R1"]["summary"],
                                  "critic_on_preserved_R2_reward_lineage": fixed_lineages[critic_label]["F1_R2"]["summary"],
                                  "scope": "independent forwards on persisted inputs/rewards; not a new actor-critic rollout"})

    eligibility_rows = []
    for replicate_id, rows in traces.items():
        for row in rows:
            decision_id = str(row["decision_id"])
            credit_row = credit_by_id[decision_id]
            identity_chain = (credit_row["selected_candidate_id"] == credit_row["applied_candidate_id"]
                              == credit_row["credited_candidate_id"])
            category = eligibility_category(action_type=str(row["selected_action_type"]), reward=float(row["reward"]),
                                            reward_component=float(row["reward_gae_component"]),
                                            identity_chain_valid=identity_chain, raw_gae=float(row["raw_gae"]))
            eligibility_rows.append({"replicate_id": replicate_id, "decision_id": decision_id,
                                     "trajectory_id": row["trajectory_id"], "selected_action_type": row["selected_action_type"],
                                     "selected_candidate_id": credit_row["selected_candidate_id"],
                                     "applied_candidate_id": credit_row["applied_candidate_id"],
                                     "credited_candidate_id": credit_row["credited_candidate_id"],
                                     "identity_chain_valid": identity_chain, "reward": row["reward"],
                                     "reward_gae_component": row["reward_gae_component"], "raw_gae": row["raw_gae"],
                                     "category": category, "actor_eligible_e1": category in ELIGIBLE_CATEGORIES,
                                     "critic_eligible_e1": math.isfinite(float(row["raw_gae"]))})
    require(len(eligibility_rows) == 48 and all(row["identity_chain_valid"] for row in eligibility_rows),
            "CREDIT_IDENTITY_LINEAGE_UNRESOLVED")

    normalization: dict[str, Any] = {}
    for replicate_id, rows in traces.items():
        eligibility_by_id = {str(row["decision_id"]): row for row in eligibility_rows if row["replicate_id"] == replicate_id}
        raw_values = [float(row["raw_gae"]) for row in rows]
        n0_values = [float(row["normalized_advantage"]) for row in rows]
        n1_values = raw_values
        eligible_ids = [str(row["decision_id"]) for row in rows if eligibility_by_id[str(row["decision_id"])]["actor_eligible_e1"]]
        n2_values, n2_scope = standardize([float(next(row["raw_gae"] for row in rows if str(row["decision_id"]) == decision_id))
                                           for decision_id in eligible_ids])
        n2_by_id = dict(zip(eligible_ids, n2_values))
        modes = {}
        for name, values, scope in (("N0", n0_values, {"scope": "current per-seed 24-row full batch", "row_count": 24}),
                                    ("N1", n1_values, {"scope": "no normalization", "row_count": 24}),
                                    ("N2", n2_values, {"scope": "E1 actor-eligible rows only", **n2_scope})):
            audit_rows = []
            for index, trace in enumerate(rows):
                decision_id = str(trace["decision_id"])
                used = name != "N2" or decision_id in n2_by_id
                value = float(values[index]) if name != "N2" else (n2_by_id.get(decision_id) if used else None)
                if value is None:
                    transition, direction = "EXCLUDED_FROM_N2", "EXCLUDED_FROM_ACTOR_NORMALIZATION"
                else:
                    transition = f"{sign(float(trace['raw_gae']))}_TO_{sign(float(value))}"
                    direction = action_direction(str(trace["selected_action_type"]), float(value))
                audit_rows.append({"decision_id": decision_id, "action_type": trace["selected_action_type"],
                                   "category": eligibility_by_id[decision_id]["category"], "raw_gae": trace["raw_gae"],
                                   "advantage": value, "raw_to_advantage_sign": transition,
                                   "preclip_direction": direction, "used_in_mode": used})
            modes[name] = {"scope": scope, "rows": audit_rows,
                           "sign_transitions": dict(sorted(Counter(row["raw_to_advantage_sign"] for row in audit_rows).items())),
                           "direction_counts": dict(sorted(Counter(row["preclip_direction"] for row in audit_rows).items())),
                           "by_action": {action: dict(sorted(Counter(row["raw_to_advantage_sign"] for row in audit_rows
                                                                      if row["action_type"] == action).items()))
                                         for action in ("CANDIDATE", "NO_ASSIGN")}}
        normalization[replicate_id] = {"seed": REPLICATES[replicate_id]["environment_seed"], "raw": describe(raw_values),
                                       "modes": modes, "seed_mixing": 0}

    repair = select_contract(eligibility_rows, normalization)
    selected_contract = repair["selected_contract"]
    overall_classification = "B_MULTI_FACTOR_MINIMAL_REPAIR_CONTRACT_READY" if factor_class["actor_seed_effect"] == "ACTOR_INIT_DOMINANT" \
        and factor_class["critic_seed_effect"] == "CRITIC_INIT_DOMINANT" and repair["selected_contract"]["selected_option"] == "E1" \
        else "PASS_NO_LEARNING_RULE_CHANGE_JUSTIFIED"

    module_after = {"actors": {label: R5MOD.module_digest(model) for label, model in actors.items()},
                    "critics": {label: R5MOD.module_digest(model) for label, model in critics.items()}}
    counters["parameter_mutation"] = int(module_before != module_after)
    frozen_after = R6MOD.frozen_hashes(BT6)
    checkpoint_after = R6MOD.checkpoint_bindings(initial_manifest, final_manifest)
    if module_before != module_after:
        hard.append("MODULE_PARAMETER_MUTATION")
    if frozen_before != frozen_after:
        hard.append("FROZEN_HASH_MUTATION")
    if checkpoint_before != checkpoint_after:
        hard.append("CHECKPOINT_HASH_MUTATION")
    allowed = {"actor_mps_forward", "critic_mps_forward"}
    if any(value for key, value in counters.items() if key not in allowed):
        hard.append("FORBIDDEN_EXECUTION_COUNTER_NONZERO")
    if counters["actor_mps_forward"] != 12 or counters["critic_mps_forward"] != 96:
        hard.append("FACTOR_FORWARD_COUNT_MISMATCH")
    if factor_class["actor_seed_effect"] != "ACTOR_INIT_DOMINANT" or factor_class["critic_seed_effect"] != "CRITIC_INIT_DOMINANT":
        hard.append("SEED_FACTORIZATION_NOT_RESOLVED")
    if not selected_contract.get("sha256"):
        hard.append("MINIMAL_CONTRACT_NOT_SELECTED")
    gate = PASS_GATE if not hard else (CREDIT_BLOCK if any("CREDIT" in row or "CONTRACT" in row for row in hard) else BLOCK)
    if hard:
        overall_classification = "BLOCKED"
    seed_factorization = {
        "same_input_contract": {"training_tensor_payloads": len(tensor_to_payload),
                                "R1_R2_tensor_multisets_exact": Counter(tensor_multiset["F1_R1"]) == Counter(tensor_multiset["F1_R2"]),
                                "actor_forward_rows": len(canonical_payloads), "critic_forward_rows": len(payload_by_id)},
        "actor_seed_effect": factor_class["actor_seed_effect"], "critic_seed_effect": factor_class["critic_seed_effect"],
        "interaction": factor_class["interaction"], "actor_evidence": actor_evidence,
        "critic_evidence": critic_evidence, "realized_lineage_evidence": lineage_evidence,
        "factor_matrix": factor_matrix,
        "method": "2×2 observational Cartesian factorization; Actor and Critic forwards are independent and never create new transitions",
    }
    test_results = {"authority_binding": binding, "execution_counters": counters, "hard_failures": hard,
                    "warnings": [], "future_leakage": 0, "review_data_used": 0, "TEST6_access": 0,
                    "github_push_performed": False}
    outputs = {
        "bt8r7_seed_factorization.json": seed_factorization,
        "bt8r7_advantage_normalization_audit.json": normalization,
        "bt8r7_credit_eligibility_rows.json": {"rows": eligibility_rows,
                                                 "category_distribution": dict(sorted(Counter(row["category"] for row in eligibility_rows).items())),
                                                 "actor_eligible_e1": sum(bool(row["actor_eligible_e1"]) for row in eligibility_rows),
                                                 "critic_eligible_e1": sum(bool(row["critic_eligible_e1"]) for row in eligibility_rows),
                                                 "identity_chain_failures": sum(not bool(row["identity_chain_valid"]) for row in eligibility_rows)},
        "bt8r7_repair_option_comparison.json": repair,
        "bt8r7_selected_minimal_contract.json": selected_contract,
        "test_results.json": test_results,
        "frozen_hash_before_after.json": {"before": frozen_before, "after": frozen_after,
                                            "all_unchanged": frozen_before == frozen_after,
                                            "module_before": module_before, "module_after": module_after,
                                            "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after},
        "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": overall_classification,
                               "source_commit": source["source_commit"], "hard_failures": hard, "warnings": [],
                               "global_locks": LOCKS, "next_step": NEXT_GATE if not hard else "STOP"},
    }
    for name, value in outputs.items():
        dump(root / name, value)
    report = (
        "# BT8-R7 final report\n\n"
        f"- gate: `{gate}`\n"
        f"- classification: `{overall_classification}`\n"
        f"- source commit: `{source['source_commit']}`\n"
        f"- Actor seed effect: `{factor_class['actor_seed_effect']}`\n"
        f"- Critic seed effect: `{factor_class['critic_seed_effect']}`\n"
        f"- selected contract: `{selected_contract['contract_id']}`\n"
        f"- next gate: `{NEXT_GATE if not hard else 'STOP'}`\n\n"
        "This is a read-only factorization and repair-selection gate. It performs no training, rollout, reward modification, checkpoint write, or structural model change.\n"
    )
    (root / "final_report.md").write_text(report, encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*")
                if path.is_file() and path.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": overall_classification,
                                   "source_commit": source["source_commit"], "file_sha256": manifest,
                                   "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"classification: {overall_classification}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
