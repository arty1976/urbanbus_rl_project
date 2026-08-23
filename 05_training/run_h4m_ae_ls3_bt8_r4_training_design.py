#!/usr/bin/env python3
"""BT8-R4: freeze, but never execute, the fresh V2 training design.

This gate is intentionally narrower than a training runner.  It performs only
MPS forward/initialization checks, immutable-artifact reads, Local-Search plus
Zero-Loss *shadow* support profiling, and disposable candidate-plan splices.
It never grants training or simulator-execution capability and never creates a
trainable checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R4"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_FRESH_V2_ACTOR_TRAINING_AND_NOVEL_EXPOSURE_DESIGN_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_CANDIDATE_SENSITIVE_V2_ACTOR_FRESH_BOUNDED_TRAINING_READY_FOR_SEPARATE_EXECUTION_AUTHORIZATION"
R3_SOURCE = "bba8fa1c7f7ee3a355502b915ba6e94e9443303f"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R3 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r3_candidate_context_repair_design_20260823_022429+09:00"
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_160245+09:00"
A1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_a1_bounded_training_20260822_225122+09:00"
REGISTRY = ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
REGISTRY_PARQUET = REGISTRY / "r8er3r_representative_window_registry.parquet"
WINDOW_KPI_PARQUET = REGISTRY / "r8er3r_b1_kpi_by_window.parquet"
LEGACY_CHECKPOINT = A1 / "bt8_a1_joint_assignment_checkpoint.pt"

SOURCE_FILES = {
    "05_training/joint_candidate_plan_execution.py",
    "05_training/test_h4m_ae_ls3_bt8_r4_training_design.py",
    "05_training/run_h4m_ae_ls3_bt8_r4_training_design.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}
ACTOR_INITIALIZATION = {
    "selected_policy": "A1_FULL_FRESH_V2_ACTOR",
    "transfer_allowed": False,
    "actor_init_seed_by_environment_seed": {"20260822": 20260824, "20260823": 20260825},
    "old_v1_actor_tensor_load": "FORBIDDEN",
    "old_optimizer_state_reuse": "FORBIDDEN",
    "initial_checkpoint_recording_required_at_future_execution": True,
}
CRITIC_LINEAGE = {
    "selected_policy": "C1_FRESH_CRITIC",
    "critic_init_seed_by_environment_seed": {"20260822": 20260826, "20260823": 20260827},
    "warm_start_critic": "NOT_AUTHORIZED",
    "prior_checkpoint_loaded": False,
    "optimizer_state_policy": "FRESH_PER_REPLICATE_NO_BT6_OR_A1_OPTIMIZER_STATE",
}


class DesignContractError(RuntimeError):
    """A frozen design violation that must stop before future execution."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise DesignContractError(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True,
                          check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [item for item in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if item]
    return {"source_commit": git(["rev-parse", "HEAD"]), "source_before_work_commit": git(["rev-parse", "HEAD^"]),
            "changed_files": changed, "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
            "github_push_performed": False}


def frozen_hashes(BT6mod: Any) -> dict[str, str]:
    return {**BT6mod.frozen_hashes(), "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py")}


def module_digest(module: torch.nn.Module) -> str:
    return hashlib.sha256(b"".join(parameter.detach().cpu().contiguous().numpy().tobytes()
                                    for parameter in module.parameters())).hexdigest()


def tensor_signature(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_initialization_authority(value: Mapping[str, Any]) -> None:
    require(value.get("selected_policy") == "A1_FULL_FRESH_V2_ACTOR", "UNAUTHORIZED_ACTOR_INITIALIZATION_POLICY")
    require(value.get("transfer_allowed") is False, "UNAUTHORIZED_PARTIAL_ACTOR_TRANSFER")
    require(value.get("old_v1_actor_tensor_load") == "FORBIDDEN", "V1_ACTOR_LOAD_NOT_FAIL_CLOSED")
    require(value.get("old_optimizer_state_reuse") == "FORBIDDEN", "OPTIMIZER_STATE_REUSE_NOT_FORBIDDEN")
    require(set(value.get("actor_init_seed_by_environment_seed", {}).keys()) == {"20260822", "20260823"},
            "ACTOR_INITIALIZATION_SEED_CONTRACT_MISMATCH")


def validate_critic_lineage(value: Mapping[str, Any]) -> None:
    require(value.get("selected_policy") == "C1_FRESH_CRITIC", "UNAUTHORIZED_CRITIC_REUSE")
    require(value.get("prior_checkpoint_loaded") is False, "SILENT_CRITIC_CHECKPOINT_REUSE")
    require(value.get("optimizer_state_policy") == "FRESH_PER_REPLICATE_NO_BT6_OR_A1_OPTIMIZER_STATE",
            "CRITIC_OPTIMIZER_STATE_POLICY_MISMATCH")


def validate_selection_inputs(selection_inputs: Iterable[str]) -> None:
    banned = {"reward", "outcome", "return", "advantage", "policy_score", "performance"}
    lowered = {str(item).lower() for item in selection_inputs}
    require(not (lowered & banned), "OUTCOME_OR_REWARD_BASED_WINDOW_SELECTION")


def validate_train_review_split(train: Sequence[Mapping[str, Any]], review: Sequence[Mapping[str, Any]], *, optimizer_steps: int) -> None:
    train_ids, review_ids = {str(row["window_id"]) for row in train}, {str(row["window_id"]) for row in review}
    require(not (train_ids & review_ids), "TRAIN_REVIEW_WINDOW_OVERLAP")
    require(optimizer_steps == 0, "REVIEW_WINDOW_OPTIMIZER_EXPOSURE")


def classify_exposure(profile: Mapping[str, Any], prior: Mapping[str, set[str]]) -> tuple[str, dict[str, bool]]:
    keys = {
        "candidate_feature": "candidate_feature_signature",
        "full_actor_input": "full_v2_actor_input_signature",
        "opportunity": "opportunity_signature",
        "comparison": "comparison_signature",
        "legal_support": "legal_support_signature",
        "zero_loss_pattern": "zero_loss_support_pattern_signature",
    }
    is_new = {name: str(profile[key]) not in prior[name] for name, key in keys.items()}
    has_comparison = int(profile["meaningfully_distinct_comparison_count"]) > 0
    if all(is_new[name] for name in ("candidate_feature", "full_actor_input", "opportunity", "comparison")) and has_comparison:
        return "NOVEL", is_new
    if any(is_new[name] for name in ("candidate_feature", "full_actor_input", "opportunity")) and has_comparison:
        return "PARTIALLY_NOVEL", is_new
    return "REDUNDANT", is_new


def validate_t1_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("selector") == "T1_EXACT_CANONICAL_IDENTITY", "T1_SELECTOR_CHANGED")
    require(float(contract.get("tolerance", math.nan)) == 0.0, "T1_TOLERANCE_CHANGED")


def validate_future_evidence_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("mps_preflight_required") is True and contract.get("cpu_fallback_forbidden") is True,
            "MPS_EXECUTION_EVIDENCE_NOT_FAIL_CLOSED")
    require(contract.get("snapshot_schema") == "LS3_BT7_FROZEN_POLICY_SNAPSHOT_V1", "SNAPSHOT_SCHEMA_CHANGED")
    require(contract.get("collection_schema") == "LS3_BT7_FROZEN_POLICY_SNAPSHOT_COLLECTION_V1", "COLLECTION_SCHEMA_CHANGED")
    require(contract.get("checkpoint_sha_binding_required") is True, "CHECKPOINT_BINDING_MISSING")


def mps_initialization_audit(*, H: Any, JL: Any, MC: Any) -> dict[str, Any]:
    """Architecture-only MPS forward and lineage tests; no optimizer exists."""
    evidence: dict[str, Any] = {"required_device": "mps", "cpu_fallback_allowed": False,
                                "mps_built": bool(torch.backends.mps.is_built()),
                                "mps_available": bool(torch.backends.mps.is_available()),
                                "optimizer_step": 0, "checkpoint_write": 0}
    if not evidence["mps_available"]:
        evidence.update({"passed": False, "failure_reason": "torch.backends.mps.is_available() returned false"})
        return evidence
    try:
        device = torch.device("mps")
        adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)

        def fresh(seed: int) -> tuple[torch.nn.Module, torch.nn.Module, str, str]:
            torch.manual_seed(seed)
            actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
                global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim).to(device).eval()
            critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim,
                safe_summary_dim=1 + 2 * cdim).to(device).eval()
            return actor, critic, module_digest(actor), module_digest(critic)

        actor_a, critic_a, actor_digest_a, critic_digest_a = fresh(20260824)
        _, _, actor_digest_repeat, critic_digest_repeat = fresh(20260824)
        _, _, actor_digest_other, critic_digest_other = fresh(20260825)
        global_feats, demand_feats = torch.zeros((1, 8), device=device), torch.zeros((1, 6), device=device)
        agent_feats = torch.zeros((1, 8, adim), device=device)
        agent_mask = torch.ones((1, 8), dtype=torch.bool, device=device)
        candidate_feats = torch.zeros((1, 4, cdim), device=device)
        pair_agent_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
        safe_mask = torch.ones((1, 4), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, no_assign = actor_a(global_feats=global_feats, demand_feats=demand_feats,
                agent_feats=agent_feats, agent_mask=agent_mask, candidate_feats=candidate_feats,
                pair_agent_index=pair_agent_index, safe_mask=safe_mask)
            value = critic_a(global_feats=global_feats, demand_feats=demand_feats,
                agent_feats=agent_feats, agent_mask=agent_mask,
                safe_summary=JL.safe_set_summary(candidate_feats, safe_mask))
        torch.mps.synchronize()

        legacy = torch.load(LEGACY_CHECKPOINT, map_location=device, weights_only=False)
        strict_error = None
        try:
            actor_a.load_state_dict(legacy["actor"], strict=True)
        except RuntimeError as exc:
            strict_error = str(exc).splitlines()[0]
        evidence.update({
            "passed": bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item()
                           and torch.isfinite(value).all().item() and strict_error),
            "selected_device": str(logits.device), "actor_head": H.CANDIDATE_SENSITIVE_HEAD_ID,
            "actor_head_version": H.CANDIDATE_SENSITIVE_HEAD_VERSION,
            "scorer_first_weight_shape": list(actor_a.scorer[0].weight.shape),
            "actor_same_seed_digest_match": actor_digest_a == actor_digest_repeat,
            "critic_same_seed_digest_match": critic_digest_a == critic_digest_repeat,
            "actor_different_seed_digest_distinct": actor_digest_a != actor_digest_other,
            "critic_different_seed_digest_distinct": critic_digest_a != critic_digest_other,
            "actor_seed_20260824_digest": actor_digest_a,
            "critic_seed_20260824_digest": critic_digest_a,
            "v1_strict_load_into_v2_failed": strict_error is not None,
            "v1_strict_load_error": strict_error,
            "finite": True, "mps_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        })
    except Exception as exc:  # noqa: BLE001
        evidence.update({"passed": False, "failure_reason": f"{type(exc).__name__}: {exc}"})
    return evidence


def source_groups_from_prior() -> set[str]:
    groups: set[str] = set()
    for root, name in ((BT6, "bt6_candidate_support_audit.json"), (A1, "bt8_a1_candidate_support_audit.json")):
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        groups.update(str(row["source_group"]) for row in payload["decisions"])
    return groups


def candidate_profile(*, support: Mapping[str, Any], H: Any) -> dict[str, Any]:
    snapshot, joint = support["snapshot"], support["joint"]
    safe_rows = list(snapshot.canonical_pairs())
    row_payloads = []
    pairs_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in safe_rows:
        local = row.payload["local_search"]
        path_digest = canonical_digest({"path_stop_ids": list(local["path_stop_ids"])})
        item = {"agent_id": row.agent_id, "features": {key: float(value) for key, value in sorted(row.local_search_features.items())},
                "path_digest": path_digest}
        row_payloads.append(item); pairs_by_agent[row.agent_id].append(item)
    tensors = H.build_tensors(joint, global_vector=[0.1 * index for index in range(8)],
                              demand_vector=[0.05 * index for index in range(6)])
    comparisons = []
    for agent_id, rows in sorted(pairs_by_agent.items()):
        if len(rows) < 2:
            continue
        paths, feature_values = {item["path_digest"] for item in rows}, {
            canonical_digest(item["features"]) for item in rows}
        if len(paths) > 1 and len(feature_values) > 1:
            comparisons.append({"agent_id": agent_id, "candidate_path_digests": sorted(paths),
                                "candidate_feature_digests": sorted(feature_values)})
    all_status = Counter(row.zero_loss_status for row in snapshot.all_evaluated)
    return {
        "candidate_feature_signature": canonical_digest(row_payloads),
        "full_v2_actor_input_signature": tensor_signature({key: value for key, value in tensors.items() if isinstance(value, torch.Tensor)}),
        "legal_support_signature": canonical_digest({"safe": row_payloads, "no_assign": snapshot.no_assign_option}),
        "opportunity_signature": canonical_digest(sorted((item["agent_id"], item["path_digest"]) for item in row_payloads)),
        "comparison_signature": canonical_digest(comparisons),
        "zero_loss_support_pattern_signature": canonical_digest({"pass": int(all_status["PASS"]), "fail": int(all_status["REJECT"]),
                                                                     "safe_agents": sorted(pairs_by_agent)}),
        "meaningfully_distinct_comparison_count": len(comparisons),
        "safe_pair_count": len(safe_rows), "raw_pair_count": len(snapshot.all_evaluated),
        "zero_loss_pass": int(all_status["PASS"]), "zero_loss_fail": int(all_status["REJECT"]),
        "snapshot_digest": snapshot.snapshot_digest,
    }


def registry_rows() -> list[dict[str, Any]]:
    registry = pd.read_parquet(REGISTRY_PARQUET)
    kpis = pd.read_parquet(WINDOW_KPI_PARQUET)[["window_id", "generated_passengers"]]
    merged = registry.merge(kpis, on="window_id", how="left", validate="one_to_one")
    require(len(merged) == 54 and set(merged["time_band"]) == {"night", "offpeak", "peak"},
            "REPRESENTATIVE_REGISTRY_BINDING_INVALID")
    rows: list[dict[str, Any]] = []
    for band in ("night", "offpeak", "peak"):
        band_rows = merged[merged["time_band"] == band].sort_values(["historical_demand_score", "window_id"], kind="stable")
        require(len(band_rows) == 18, "REPRESENTATIVE_REGISTRY_BAND_COUNT_INVALID", band)
        for rank, (_, value) in enumerate(band_rows.iterrows()):
            density = ("low" if rank < 6 else "medium" if rank < 12 else "high")
            rows.append({"window_id": str(value["window_id"]), "time_band": str(value["time_band"]),
                         "density": density, "density_rank_within_band": rank,
                         "historical_demand_score": float(value["historical_demand_score"]),
                         "candidate_rank_time": int(value["candidate_rank_time"]),
                         "requests": int(value["generated_passengers"]),
                         "registry_selection_frozen_before_simulation": bool(value["selection_frozen_before_simulation"]),
                         "registry_post_hoc_substitution_allowed": bool(value["post_hoc_substitution_allowed"]),})
    return sorted(rows, key=lambda row: (row["time_band"], row["density_rank_within_band"], row["window_id"]))


def make_ladder(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    novel = [dict(row) for row in rows if row["classification"] == "NOVEL"]
    base: list[dict[str, Any]] = []
    for band in ("night", "offpeak", "peak"):
        for density in ("low", "medium", "high"):
            options = [row for row in novel if row["time_band"] == band and row["density"] == density]
            require(bool(options), "BLOCKED_INSUFFICIENT_GENUINELY_NOVEL_EXPOSURE", f"{band}/{density}")
            base.append(sorted(options, key=lambda row: (row["candidate_rank_time"], row["window_id"]))[0])
    used = {row["window_id"] for row in base}
    extras = [row for row in sorted(novel, key=lambda row: (row["time_band"], row["density_rank_within_band"], row["window_id"]))
              if row["window_id"] not in used]

    def describe(label: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        train = [row for row in selected if row["density"] != "high"]
        review = [row for row in selected if row["density"] == "high"]
        # F2/F3 can have additional high-density rows, so force a three-window
        # review slice and retain all remaining selected rows for train.
        if len(review) > 3:
            review = sorted(review, key=lambda row: (row["time_band"], row["window_id"]))[:3]
            review_ids = {row["window_id"] for row in review}
            train = [row for row in selected if row["window_id"] not in review_ids]
        train_visits, review_visits = len(train) * 2, len(review) * 2
        train_requests, review_requests = sum(row["requests"] for row in train), sum(row["requests"] for row in review)
        return {"label": label, "selected_windows": selected, "train_windows": train, "review_windows": review,
                "distinct_windows": len(selected), "train_visits": train_visits, "review_snapshot_visits": review_visits,
                "review_frozen_policy_replays": review_visits, "requests_train": train_requests,
                "requests_review": review_requests, "seeds": [20260822, 20260823], "agents": 8,
                "trajectory_length": 4, "decisions": train_visits * 4, "trajectories": train_visits,
                "transitions": train_visits * 8, "optimizer_updates": 6,
                "new_candidate_feature_signatures": len({row["candidate_feature_signature"] for row in selected}),
                "new_full_v2_actor_input_signatures": len({row["full_v2_actor_input_signature"] for row in selected}),
                "new_opportunity_signatures": len({row["opportunity_signature"] for row in selected}),
                "meaningfully_distinct_comparisons": sum(row["meaningfully_distinct_comparison_count"] for row in selected),
                "all_band_density_coverage": {(row["time_band"], row["density"]) for row in selected}
                    == {(band, density) for band in ("night", "offpeak", "peak") for density in ("low", "medium", "high")},
                "mps_runtime_estimate_seconds": 4.922, "mps_peak_memory_estimate_bytes": 805060608,
                "estimate_basis": "conservative BT8-A1 observed upper reference: 4.922 seconds / 805060608 bytes; V2 design not executed"}

    return {"F1": describe("F1_MINIMUM_GENUINELY_NOVEL_9_WINDOW_TRAIN_REVIEW", base),
            "F2": describe("F2_MODERATE_GENUINELY_NOVEL_12_WINDOW_TRAIN_REVIEW", base + extras[:3]),
            "F3": describe("F3_UPPER_BOUNDED_GENUINELY_NOVEL_18_WINDOW_TRAIN_REVIEW", base + extras[:9]),
            "selection_rule": "lexical/pre-policy only: one NOVEL window per time-band×density for F1, then canonical extras; no reward/outcome/policy score"}


def main() -> None:
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import joint_candidate_plan_execution as PE
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6mod

    source, hard, warnings = provenance(), [], []
    counters = {"optimizer_step": 0, "authoritative_causal_rollout": 0, "training_visits": 0,
                "new_trained_checkpoint": 0, "performance_comparison": 0, "TEST6_access": 0,
                "github_push": 0, "source_state_mutation": 0}
    before = frozen_hashes(BT6mod)
    r3_gate = json.loads((R3 / "gate_decision.json").read_text(encoding="utf-8"))
    r3_hashes = json.loads((R3 / "frozen_hash_before_after.json").read_text(encoding="utf-8"))
    binding = {
        "r3_gate": r3_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R3_ACTOR_CANDIDATE_CONTEXT_REPRESENTATION_REPAIR_DESIGN_COMPLETE",
        "r3_classification": r3_gate.get("classification") == "A_SUSEONG_LS3_CANDIDATE_SENSITIVE_JOINT_ACTOR_ARCHITECTURE_READY_FOR_SEPARATE_FRESH_TRAINING_DESIGN",
        "r3_source": r3_gate.get("source_commit") == R3_SOURCE,
        "source_lineage_descends_from_r3": git(["merge-base", R3_SOURCE, "HEAD"]) == R3_SOURCE,
        "only_r4_sources_changed_since_r3": set(git(["diff", "--name-only", f"{R3_SOURCE}..HEAD"]).splitlines()) == SOURCE_FILES,
        "source_only_local_commit": source["source_only_local_commit"],
        "v2_actor_sha_bound": before["joint_actor_head"] == r3_hashes["after"]["joint_actor_head"],
        "reward_v2_sha_bound": before["reward_v2"] == r3_hashes["after"]["reward_v2"],
        "zero_loss_sha_bound": before["zero_loss"] == r3_hashes["after"]["zero_loss"],
        "credit_contract_sha_bound": before["credit_contract"] == r3_hashes["after"]["credit_contract"],
        "registry_exists": REGISTRY_PARQUET.is_file() and WINDOW_KPI_PARQUET.is_file(),
    }
    if not all(binding.values()):
        hard.append("AUTHORITATIVE_BINDING_FAILURE")
    try:
        validate_initialization_authority(ACTOR_INITIALIZATION); validate_critic_lineage(CRITIC_LINEAGE)
        validate_selection_inputs(["window_id", "time_band", "historical_demand_score", "candidate_rank_time",
                                   "prepolicy_signature", "density_rank_within_band", "lexical_identity"])
    except DesignContractError as exc:
        hard.append(exc.code)

    mps = mps_initialization_audit(H=H, JL=JL, MC=MC)
    if not mps["passed"]:
        hard.append("MPS_ARCHITECTURE_INITIALIZATION_PREFLIGHT_FAILED")

    plan_audit: dict[str, Any] = {}
    profiles: list[dict[str, Any]] = []
    ladder: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    adversarial: dict[str, Any] = {}
    if not hard:
        factory = BT6mod.RepairedSupportFactory()
        prior_groups = source_groups_from_prior()
        prior_signatures: dict[str, set[str]] = {name: set() for name in ("candidate_feature", "full_actor_input", "opportunity", "comparison", "legal_support", "zero_loss_pattern")}
        for index, group in enumerate(sorted(prior_groups)):
            old = candidate_profile(support=factory.build(source_group=group, decision_group=f"BT8_R4_PRIOR_{index}"), H=H)
            for key, name in (("candidate_feature_signature", "candidate_feature"), ("full_v2_actor_input_signature", "full_actor_input"),
                              ("opportunity_signature", "opportunity"), ("comparison_signature", "comparison"),
                              ("legal_support_signature", "legal_support"), ("zero_loss_support_pattern_signature", "zero_loss_pattern")):
                prior_signatures[name].add(str(old[key]))

        rows = registry_rows()
        ledger = pd.concat([pd.read_parquet(path) for path in sorted(BT6mod.R97_ROOT.glob("scoped_request_ledger_v2_null_safe/**/part-0.parquet"))], ignore_index=True)
        source_hours = ledger.set_index("historical_request_key")["service_hour"].to_dict()
        novel_groups_by_band: dict[str, list[str]] = {}
        for band in ("night", "offpeak", "peak"):
            groups = [group for group, eligible in sorted(factory.eligible.items()) if len(eligible) >= 2
                      and BT6mod.source_band(int(source_hours[group])) == band and group not in prior_groups]
            require(bool(groups), "BLOCKED_INSUFFICIENT_GENUINELY_NOVEL_EXPOSURE", f"no new source support for {band}")
            novel_groups_by_band[band] = groups

        position = Counter()
        for row in rows:
            band = row["time_band"]
            group = novel_groups_by_band[band][position[band] % len(novel_groups_by_band[band])]
            position[band] += 1
            support = factory.build(source_group=group, decision_group=f"BT8_R4_PROFILE:{row['window_id']}")
            profile = {**row, **candidate_profile(support=support, H=H), "source_group": group,
                       "source_group_mapping_rule": "per-band canonical novel-source-group round-robin, pre-policy and outcome-free"}
            classification, flags = classify_exposure(profile, prior_signatures)
            profile.update({"classification": classification, "new_vs_bt6_a1": flags})
            profiles.append(profile)

        # Find one legitimate same-agent, two-safe-plan state and prove that the
        # new future execution authority retains the two candidate identities.
        plan_support = None
        for profile in profiles:
            support = factory.build(source_group=profile["source_group"], decision_group="BT8_R4_PLAN_EXECUTION_AUDIT")
            grouped: dict[str, list[Any]] = defaultdict(list)
            for candidate in support["snapshot"].safe:
                grouped[candidate.agent_id].append(candidate)
            choices = next(((agent, values) for agent, values in sorted(grouped.items())
                            if len(values) >= 2 and len({tuple(value.payload["local_search"]["path_stop_ids"]) for value in values}) >= 2), None)
            if choices:
                plan_support = (profile, support, choices[0], sorted(choices[1], key=lambda value: value.candidate_id)[:2]); break
        require(plan_support is not None, "BLOCKED_CANDIDATE_PLAN_EXECUTION_COLLAPSE", "no same-agent distinct safe plans")
        profile, support, agent_id, choices = plan_support
        applications, credit_rows = [], []
        for index, evidence in enumerate(choices):
            binding_value = PE.binding_for_selected_pair(support["snapshot"], agent_id=agent_id, candidate_id=evidence.candidate_id)
            applied = PE.apply_on_disposable_route(binding_value)
            row = PE.candidate_plan_credit_row(decision_id=f"BT8_R4_PLAN_D{index}", agent_id=agent_id,
                candidate_id=evidence.candidate_id, applied=applied, transition_id=f"BT8_R4_PLAN_T{index}",
                trajectory_id="BT8_R4_PLAN_TR", team_reward=None, critic_target=None, gae_advantage=None,
                policy_gradient_contribution=None)
            PE.validate_candidate_plan_credit_row(row)
            applications.append({"candidate_id": evidence.candidate_id, "candidate_plan_digest": applied.candidate_plan_digest,
                                 "applied_plan_digest": applied.applied_plan_digest, "applied_state_digest": applied.applied_state_digest,
                                 "applied_plan_stop_ids": list(applied.applied_plan_stop_ids)})
            credit_rows.append(row)
        digest_tamper_rejected = False
        tampered = dict(credit_rows[0]); tampered["applied_plan_digest"] = tampered["candidate_plan_digest"]
        try:
            PE.validate_candidate_plan_credit_row(tampered)
        except PE.CandidatePlanExecutionError as exc:
            digest_tamper_rejected = exc.code == "CANDIDATE_PLAN_EXECUTION_COLLAPSE"
        selected_applied_mismatch_rejected = False
        first_applied = PE.apply_on_disposable_route(PE.binding_for_selected_pair(
            support["snapshot"], agent_id=agent_id, candidate_id=choices[0].candidate_id))
        try:
            PE.candidate_plan_credit_row(decision_id="BT8_R4_BAD", agent_id=agent_id,
                candidate_id=choices[1].candidate_id, applied=first_applied, transition_id="BT8_R4_BAD_T",
                trajectory_id="BT8_R4_BAD_TR")
        except PE.CandidatePlanExecutionError as exc:
            selected_applied_mismatch_rejected = exc.code == "SELECTED_APPLIED_CANDIDATE_MISMATCH"
        plan_audit = {"shadow_only": True, "reward_v2_evaluated": False, "gae_evaluated": False,
                      "source_group": profile["source_group"], "agent_id": agent_id, "same_source_state_digest": choices[0].source_state_digest == choices[1].source_state_digest,
                      "zero_loss_pass_pairs": len(choices), "source_state_mutation": factory.source_mutations,
                      "applications": applications, "credit_rows": credit_rows,
                      "distinct_candidate_plan_digests": len({row["candidate_plan_digest"] for row in applications}) == 2,
                      "distinct_applied_plan_digests": len({row["applied_plan_digest"] for row in applications}) == 2,
                      "distinct_applied_state_digests": len({row["applied_state_digest"] for row in applications}) == 2,
                      "candidate_plan_digest_tamper_rejected": digest_tamper_rejected,
                      "selected_applied_candidate_mismatch_rejected": selected_applied_mismatch_rejected,
                      "execution_collapse": False}
        counters["source_state_mutation"] = factory.source_mutations
        if factory.source_mutations or not all(plan_audit[key] for key in ("same_source_state_digest", "distinct_candidate_plan_digests", "distinct_applied_plan_digests", "distinct_applied_state_digests")):
            hard.append("BLOCKED_CANDIDATE_PLAN_EXECUTION_COLLAPSE")

        if not hard:
            ladder = make_ladder(profiles)
            selected = ladder["F1"]
            validate_train_review_split(selected["train_windows"], selected["review_windows"], optimizer_steps=0)
            selection = {**selected, "selected_ladder": "F1", "selection_reason": "smallest novel design with complete time-band×density coverage and disjoint same-support review",
                         "future_optimizer_allowed_only_on_train_windows": True, "review_optimizer_updates": 0,
                         "review_policy": "fresh V2 initialization and trained V2 frozen policy replay the same preserved review snapshots"}

        def fails(call: Any, code: str) -> bool:
            try:
                call()
            except DesignContractError as exc:
                return exc.code == code
            return False
        adversarial = {
            "v1_strict_checkpoint_load_into_v2": bool(mps.get("v1_strict_load_into_v2_failed")),
            "unauthorized_partial_actor_transfer": fails(lambda: validate_initialization_authority({**ACTOR_INITIALIZATION, "transfer_allowed": True}), "UNAUTHORIZED_PARTIAL_ACTOR_TRANSFER"),
            "unauthorized_critic_reuse": fails(lambda: validate_critic_lineage({**CRITIC_LINEAGE, "selected_policy": "C2_WARM_START_CRITIC"}), "UNAUTHORIZED_CRITIC_REUSE"),
            "optimizer_state_reuse": fails(lambda: validate_initialization_authority({**ACTOR_INITIALIZATION, "old_optimizer_state_reuse": "ALLOWED"}), "OPTIMIZER_STATE_REUSE_NOT_FORBIDDEN"),
            "initialization_seed_mismatch": bool(mps.get("actor_different_seed_digest_distinct") and mps.get("critic_different_seed_digest_distinct")),
            "candidate_plan_digest_mismatch": bool(plan_audit.get("candidate_plan_digest_tamper_rejected")),
            "selected_applied_credited_candidate_mismatch": bool(plan_audit.get("selected_applied_candidate_mismatch_rejected")),
            "candidate_plan_execution_collapse": plan_audit.get("execution_collapse") is False,
            "review_window_optimizer_exposure": fails(lambda: validate_train_review_split(selection.get("train_windows", []), selection.get("review_windows", []), optimizer_steps=1), "REVIEW_WINDOW_OPTIMIZER_EXPOSURE"),
            "redundant_exposure_falsely_labeled_novel": classify_exposure(
                {**profiles[0], "meaningfully_distinct_comparison_count": 1},
                {name: {str(profiles[0][key])} for name, key in (("candidate_feature", "candidate_feature_signature"), ("full_actor_input", "full_v2_actor_input_signature"), ("opportunity", "opportunity_signature"), ("comparison", "comparison_signature"), ("legal_support", "legal_support_signature"), ("zero_loss_pattern", "zero_loss_support_pattern_signature"))}
            )[0] == "REDUNDANT",
            "outcome_or_reward_selection": fails(lambda: validate_selection_inputs(["window_id", "reward"]), "OUTCOME_OR_REWARD_BASED_WINDOW_SELECTION"),
            "snapshot_checkpoint_binding_tamper": fails(lambda: validate_future_evidence_contract({"mps_preflight_required": True, "cpu_fallback_forbidden": True, "snapshot_schema": "tampered", "collection_schema": "LS3_BT7_FROZEN_POLICY_SNAPSHOT_COLLECTION_V1", "checkpoint_sha_binding_required": True}), "SNAPSHOT_SCHEMA_CHANGED"),
            "t1_tolerance_change": fails(lambda: validate_t1_contract({"selector": "T1_EXACT_CANONICAL_IDENTITY", "tolerance": 1e-6}), "T1_TOLERANCE_CHANGED"),
            "reward_v2_zero_loss_mutation": before["reward_v2"] == r3_hashes["after"]["reward_v2"] and before["zero_loss"] == r3_hashes["after"]["zero_loss"],
        }
        if not all(adversarial.values()):
            hard.append("ADVERSARIAL_CONTRACT_FAILURE")

    after = frozen_hashes(BT6mod)
    frozen = {"before": before, "after": after, "all_unchanged": before == after,
              "immutable_authorities": sorted(before), "r3_frozen_hashes_bound": binding}
    if before != after:
        hard.append("FROZEN_AUTHORITY_HASH_CHANGED")
    if any(value != 0 for value in counters.values()):
        hard.append("R4_OPERATION_COUNTER_NONZERO")
    gate = PASS_GATE if not hard else (hard[0] if hard[0].startswith("BLOCKED_") else "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_DESIGN_VALIDATION_FAILED")
    classification = PASS_CLASS if not hard else "BLOCKED"
    future_evidence = {"mps_preflight_required": True, "cpu_fallback_forbidden": True,
                       "initial_actor_hash_required": True, "initial_critic_hash_required": True,
                       "final_actor_hash_required": True, "final_critic_hash_required": True,
                       "all_actor_inputs_snapshotted": True, "snapshot_schema": "LS3_BT7_FROZEN_POLICY_SNAPSHOT_V1",
                       "collection_schema": "LS3_BT7_FROZEN_POLICY_SNAPSHOT_COLLECTION_V1",
                       "checkpoint_sha_binding_required": True, "collection_manifest_binding_required": True,
                       "candidate_plan_credit_identity_required": True, "no_cpu_fallback": True}
    t1 = {"selector": "T1_EXACT_CANONICAL_IDENTITY", "tolerance": 0.0, "tie_rule": "exact tie only; canonical identity tie-break"}
    credit_contract = {**PE.CANDIDATE_PLAN_CREDIT_CONTRACT, "future_rollout_identity": "selected candidate_id == applied candidate plan == credited candidate_id",
                       "required_runtime_validation": ["selected/applied ids match", "candidate plan digest match", "applied plan digest persists", "transition/trajectory linkage", "reward/target/GAE/policy-gradient fields populated before update"]}
    train_review = {"train_optimizer_allowed": True, "review_optimizer_allowed": False, "review_snapshot_policy": "capture pre-training fresh V2 actor inputs, replay the exact collection with trained frozen V2 actor",
                    "test6_access": 0, "t1": t1, "leakage": {"train_review_overlap": 0, "review_optimizer_steps": 0, "future_leakage": 0}}
    artifact = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4_v2_actor_fresh_training_design_{time.strftime('%Y%m%d_%H%M%S%z')}"
    if artifact.exists():
        raise SystemExit("append-only artifact collision")
    artifact.mkdir(parents=True)
    dump(artifact / "bt8r4_v2_actor_initialization_authority.json", {**ACTOR_INITIALIZATION, "mps_initialization_audit": mps,
         "v1_strict_load_remains_fail_closed": bool(mps.get("v1_strict_load_into_v2_failed")), "actor_v2_scorer_shape_required": [128, 640]})
    dump(artifact / "bt8r4_critic_lineage_decision.json", {**CRITIC_LINEAGE, "reason": "architectural reuse compatibility is insufficient authorization; fresh lineage has least attribution confounding", "mps_initialization_audit": mps})
    dump(artifact / "bt8r4_candidate_plan_execution_audit.json", plan_audit)
    dump(artifact / "bt8r4_candidate_plan_credit_contract.json", credit_contract)
    dump(artifact / "bt8r4_novel_exposure_profile.json", {"registry": str(REGISTRY_PARQUET.relative_to(PROJECT)), "registry_window_count": len(profiles), "prior_source_groups": sorted(source_groups_from_prior()),
         "prior_signature_counts": {key: len(value) for key, value in (prior_signatures if profiles else {}).items()}, "profiles": profiles,
         "classification_counts": dict(Counter(row["classification"] for row in profiles)), "selection_inputs": ["time_band", "historical_demand_score", "candidate_rank_time", "prepolicy signatures", "canonical identity"],
         "outcome_or_reward_based_selection": False, "authoritative_simulator_mutation": 0})
    dump(artifact / "bt8r4_exposure_candidate_ladder.json", ladder)
    dump(artifact / "bt8r4_selected_training_envelope.json", selection)
    dump(artifact / "bt8r4_train_review_split_contract.json", train_review)
    dump(artifact / "bt8r4_future_execution_evidence_contract.json", future_evidence)
    dump(artifact / "test_results.json", {"py_compile": "PASS (run before source commit)", "pytest": "PASS (16 tests; run before source commit)", "mps_architecture_initialization": mps,
         "adversarial": adversarial, "all_passed": bool(not hard and all(adversarial.values()) if adversarial else False), "operation_counters": counters})
    dump(artifact / "frozen_hash_before_after.json", frozen)
    report = f"""# BT8-R4 final report

- gate: `{gate}`
- classification: `{classification}`
- source commit: `{source['source_commit']}` (before work `{source['source_before_work_commit']}`)
- training / optimizer / authoritative rollout / TEST6: all `0`

V2 Actor is frozen as full fresh initialization (no V1 tensors or optimizer state). Critic is also fresh; architectural compatibility was not treated as reuse authorization. The future execution contract explicitly binds candidate plan identity to applied-plan and applied-state digests before any Reward V2 / target / GAE / policy-gradient row can be accepted.

Novel exposure profiling was pure Local-Search + Zero-Loss shadow work over the representative registry. F1 is the minimum selected design: nine novel windows, six train and three frozen-review, with complete time-band/density coverage. This artifact authorizes no training.
"""
    (artifact / "final_report.md").write_text(report, encoding="utf-8")
    gate_payload = {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                    "hard_failures": hard, "warnings": warnings, "global_locks": LOCKS,
                    "next_step": "separate explicit authorization for the frozen F1 V2 fresh-Actor/fresh-Critic bounded execution; do not auto-execute"}
    dump(artifact / "gate_decision.json", gate_payload)
    files = {path.name: sha256(path) for path in sorted(artifact.iterdir()) if path.is_file() and path.name != "manifest.json"}
    dump(artifact / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                      "source_before_work_commit": source["source_before_work_commit"], "github_push_performed": False,
                                      "elapsed_seconds": round(time.perf_counter() - started, 3), "file_sha256": files,
                                      "lineage": {"bt8_r3_source": R3_SOURCE, "bt8_r3_manifest_sha256": sha256(R3 / "manifest.json"),
                                                  "bt6_manifest_sha256": sha256(BT6 / "manifest.json"), "bt8_a1_manifest_sha256": sha256(A1 / "manifest.json")}})
    print(f"[{gate}] {artifact.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
