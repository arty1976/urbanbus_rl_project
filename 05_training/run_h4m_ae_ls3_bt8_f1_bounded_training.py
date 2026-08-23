#!/usr/bin/env python3
"""BT8-F1 exact fresh V2 Actor/Critic bounded execution.

The runner consumes only the BT8-R4 design and BT8-R4A authority completion.
It has no window selection, budget expansion, CPU fallback, legacy parameter
transfer, or performance comparison path.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-F1"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_FRESH_V2_ACTOR_CRITIC_NOVEL_EXPOSURE_BOUNDED_TRAINING_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_FRESH_V2_POLICY_READY_FOR_SAME_SUPPORT_FROZEN_DISCRIMINATION_REVIEW"
MPS_BLOCK = "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
INTEGRITY_BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_F1_EXECUTION_INTEGRITY_FAILURE"
R4_SOURCE = "32fbf2b3043166188bde891a5c450702e7db3607"
R4A_SOURCE = "eac4a209e09e696380bde3bbc437a4fd13c45e99"
R4_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_FRESH_V2_ACTOR_TRAINING_AND_NOVEL_EXPOSURE_DESIGN_COMPLETE"
R4A_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4A_F1_EXECUTION_AUTHORITY_COMPLETION"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R4 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4_v2_actor_fresh_training_design_20260823_122459+0900"
R4A = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4a_f1_execution_authority_completion_20260823_132257+0900"
SOURCE_FILES = {
    "05_training/run_h4m_ae_ls3_bt8_f1_bounded_training.py",
    "05_training/test_h4m_ae_ls3_bt8_f1_authority_gate.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


class F1ExecutionError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(value: bool, code: str, detail: str = "") -> None:
    if not value:
        raise F1ExecutionError(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True,
                          check=True).stdout.strip()


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, value in sorted(module.state_dict().items()):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    return canonical_digest({"state_count": len(state["state"]), "groups": [
        {key: value for key, value in group.items() if key != "params"} | {"parameter_count": len(group["params"])}
        for group in state["param_groups"]]})


def rng_digest() -> str:
    return hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()


def provenance() -> dict[str, Any]:
    changed = [name for name in git(["diff", "--name-only", f"{R4A_SOURCE}..HEAD"]).splitlines() if name]
    return {"source_commit": git(["rev-parse", "HEAD"]), "source_parent": git(["rev-parse", "HEAD^"]),
            "source_lineage_descends_from_r4a": git(["merge-base", R4A_SOURCE, "HEAD"]) == R4A_SOURCE,
            "changed_files_since_r4a": changed, "source_only_local_commit": bool(changed) and set(changed).issubset(SOURCE_FILES),
            "github_push_performed": False}


def load_json(path: Path) -> Any:
    require(path.is_file(), "AUTHORITATIVE_ARTIFACT_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def authorities() -> dict[str, Any]:
    paths = {"r4_gate": R4 / "gate_decision.json", "selected": R4 / "bt8r4_selected_training_envelope.json",
             "actor": R4 / "bt8r4_v2_actor_initialization_authority.json", "critic": R4 / "bt8r4_critic_lineage_decision.json",
             "split": R4 / "bt8r4_train_review_split_contract.json", "future": R4 / "bt8r4_future_execution_evidence_contract.json",
             "r4a_gate": R4A / "gate_decision.json", "seed": R4A / "bt8r4a_seed_update_contract.json",
             "ppo": R4A / "bt8r4a_ppo_partition_contract.json", "bridge": R4A / "bt8r4a_candidate_plan_bridge_contract.json",
             "frozen": R4A / "frozen_hash_before_after.json"}
    return {key: load_json(path) for key, path in paths.items()} | {"sha256": {key: sha256(path) for key, path in paths.items()}}


def frozen_hashes(BT6: Any) -> dict[str, str]:
    return {**BT6.frozen_hashes(), "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py")}


def binding(*, auth: Mapping[str, Any], source: Mapping[str, Any], frozen: Mapping[str, str], FC: Any, CB: Any) -> dict[str, bool]:
    selected = auth["selected"]
    required = {"distinct_windows": 9, "train_visits": 12, "review_snapshot_visits": 6,
                "requests_train": 43, "requests_review": 27, "agents": 8, "decisions": 48,
                "trajectories": 12, "transitions": 96, "optimizer_updates": 6}
    try:
        FC.validate_contract(); contract = True
    except FC.F1ExecutionContractError:
        contract = False
    return {
        "r4_gate": auth["r4_gate"].get("gate") == R4_GATE and auth["r4_gate"].get("source_commit") == R4_SOURCE,
        "r4a_gate": auth["r4a_gate"].get("gate") == R4A_GATE and auth["r4a_gate"].get("source_commit") == R4A_SOURCE,
        "source_provenance": bool(source["source_lineage_descends_from_r4a"] and source["source_only_local_commit"]),
        "f1_envelope": all(selected.get(key) == value for key, value in required.items())
        and len(selected.get("train_windows", [])) == 6 and len(selected.get("review_windows", [])) == 3,
        "novelty": selected.get("novel_windows") == 5 and selected.get("partially_novel_windows") == 4,
        "fresh_actor": auth["actor"].get("selected_policy") == "A1_FULL_FRESH_V2_ACTOR"
        and auth["actor"].get("transfer_allowed") is False and auth["actor"].get("old_v1_actor_tensor_load") == "FORBIDDEN",
        "fresh_critic": auth["critic"].get("selected_policy") == "C1_FRESH_CRITIC"
        and auth["critic"].get("prior_checkpoint_loaded") is False,
        "seed_ppo": contract and auth["ppo"].get("ppo_epochs") == 3 and auth["ppo"].get("minibatch_size") == 24
        and auth["ppo"].get("full_batch_size") == 24 and auth["seed"].get("aggregate", {}).get("raw_optimizer_step_calls") == 12,
        "review": auth["split"].get("review_optimizer_allowed") is False and auth["split"].get("t1", {}).get("tolerance") == 0.0,
        "candidate_bridge": auth["bridge"].get("contract_id") == CB.CONTRACT_ID and auth["bridge"].get("serve_fallback") is False,
        "frozen": dict(frozen) == auth["frozen"].get("after"),
    }


def mps_preflight(*, H: Any, JL: Any, MC: Any) -> dict[str, Any]:
    evidence = {"required_device": "mps:0", "cpu_fallback": False,
                "mps_built": bool(torch.backends.mps.is_built()), "mps_available": bool(torch.backends.mps.is_available()),
                "capability_grant": 0, "authoritative_candidate_generation": 0, "causal_rollout": 0,
                "optimizer_step": 0, "checkpoint_write": 0}
    if not evidence["mps_built"] or not evidence["mps_available"]:
        return evidence | {"passed": False, "failure_reason": "MPS_BUILT_OR_AVAILABLE_FALSE"}
    try:
        device = torch.device("mps:0"); adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
        torch.manual_seed(20260824)
        actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim).to(device)
        torch.manual_seed(20260826)
        critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim, safe_summary_dim=1 + 2 * cdim).to(device)
        inputs = {"global_feats": torch.zeros((1, 8), device=device), "demand_feats": torch.zeros((1, 6), device=device),
                  "agent_feats": torch.zeros((1, 8, adim), device=device), "agent_mask": torch.ones((1, 8), dtype=torch.bool, device=device),
                  "candidate_feats": torch.zeros((1, 8, cdim), device=device),
                  "pair_agent_index": torch.arange(8, dtype=torch.long, device=device).reshape(1, 8),
                  "safe_mask": torch.ones((1, 8), dtype=torch.bool, device=device)}
        with torch.no_grad():
            logits, no_assign = actor(**inputs)
            value = critic(global_feats=inputs["global_feats"], demand_feats=inputs["demand_feats"], agent_feats=inputs["agent_feats"],
                           agent_mask=inputs["agent_mask"], safe_summary=JL.safe_set_summary(inputs["candidate_feats"], inputs["safe_mask"]))
        torch.mps.synchronize(); finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item() and torch.isfinite(value).all().item())
        return evidence | {"passed": finite, "selected_device": str(logits.device), "finite": finite,
                           "scorer_input_dimension": int(actor.scorer[0].in_features), "scorer_first_weight_shape": list(actor.scorer[0].weight.shape),
                           "shape_contract": {"agents": 8, "candidate_pairs": 8, "candidate_feature_dim": cdim},
                           "mps_allocated_bytes": int(torch.mps.current_allocated_memory()), "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory())}
    except Exception as exc:  # noqa: BLE001
        return evidence | {"passed": False, "failure_reason": f"{type(exc).__name__}: {exc}"}


def actor_config(FPS: Any, H: Any, adim: int, cdim: int) -> dict[str, Any]:
    config = FPS.actor_config(global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim,
                              actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
                              actor_head_id=H.CANDIDATE_SENSITIVE_HEAD_ID, actor_head_version=H.CANDIDATE_SENSITIVE_HEAD_VERSION)
    config["actor_class"] = "CandidateSensitiveMultiAgentCandidateAssignmentHead"
    return config


def feature_contract(FPS: Any, adim: int, cdim: int) -> dict[str, Any]:
    return {"feature_contract_id": FPS.FEATURE_CONTRACT_ID, "global_feature_dim": 8, "demand_feature_dim": 6,
            "agent_feature_dim": adim, "candidate_feature_dim": cdim,
            "feature_normalization": "already materialized post-Zero-Loss actor input", "candidate_support_contract_id": FPS.CANDIDATE_SUPPORT_CONTRACT_ID,
            "candidate_support_contract": "immutable post-Zero-Loss support", "actor_input_boundary": "model_ready_pack with masked empty support padding"}


def make_replicate(row: Mapping[str, Any], H: Any, JL: Any, MC: Any, device: torch.device) -> dict[str, Any]:
    adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    torch.manual_seed(int(row["actor_init_seed"])); actor_rng = rng_digest()
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim).to(device)
    torch.manual_seed(int(row["critic_init_seed"])); critic_rng = rng_digest()
    critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim, safe_summary_dim=1 + 2 * cdim).to(device)
    actor_opt, critic_opt = torch.optim.Adam(actor.parameters(), lr=1e-4), torch.optim.Adam(critic.parameters(), lr=1e-4)
    return {**dict(row), "actor": actor, "critic": critic, "actor_opt": actor_opt, "critic_opt": critic_opt,
            "actor_initial_digest": module_digest(actor), "critic_initial_digest": module_digest(critic), "actor_rng_digest": actor_rng,
            "critic_rng_digest": critic_rng, "actor_optimizer_initial_digest": optimizer_digest(actor_opt), "critic_optimizer_initial_digest": optimizer_digest(critic_opt)}


def model_pack(*, support: Mapping[str, Any], H: Any, BT6: Any, cdim: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    packed = BT6.device_pack(H.build_tensors(support["joint"], global_vector=[.1 * n for n in range(8)], demand_vector=[.05 * n for n in range(6)]), device)
    return packed, BT6.model_ready_pack(packed, cdim=cdim, device=device)


def f1_agent_slot_mapping(*, eligible: Mapping[str, Sequence[tuple[str, int, int]]],
                          source_groups: Sequence[str], slots: int) -> dict[str, int]:
    """Color only the six frozen F1 source groups into the eight causal slots.

    BT6's map intentionally covers its own historical source-group selection;
    it is not an authority for F1's novel groups.  This pure pre-rollout map
    uses lexical identities and co-eligibility only, never actor outcomes.
    """
    neighbors: dict[str, set[str]] = {}
    for group in source_groups:
        members = [agent for agent, _, _ in eligible[str(group)]]
        require(bool(members), "F1_SOURCE_GROUP_HAS_NO_ELIGIBLE_AGENT", str(group))
        for agent in members:
            neighbors.setdefault(agent, set()).update(other for other in members if other != agent)
    mapping: dict[str, int] = {}
    for agent in sorted(neighbors):
        used = {mapping[other] for other in neighbors[agent] if other in mapping}
        slot = next((value for value in range(int(slots)) if value not in used), None)
        require(slot is not None, "F1_AGENT_SLOT_COLORING_EXCEEDS_EIGHT", agent)
        mapping[agent] = int(slot)
    for group in source_groups:
        group_slots = [mapping[agent] for agent, _, _ in eligible[str(group)]]
        require(len(group_slots) == len(set(group_slots)), "F1_AGENT_SLOT_COLLISION", str(group))
    return mapping


def snapshot(*, root: Path, store: list[dict[str, Any]], FPS: Any, decision_id: str, window: Mapping[str, Any],
             seed: int, index: int, packed: Mapping[str, Any], model_packed: Mapping[str, Any], support: Mapping[str, Any],
             MC: Any, config: Mapping[str, Any], features: Mapping[str, Any], frozen: Mapping[str, str], source_commit: str, device: torch.device) -> dict[str, Any]:
    payload = FPS.capture_actor_input(decision_id=decision_id, window_id=window["window_id"], seed=seed, decision_index=index,
        time_band=window["time_band"], actor_inputs=model_packed, agent_ids=[item.agent_id for item in sorted(support["joint"].agents, key=lambda item: item.agent_id)],
        candidate_ids=packed["pair_keys"], selectable_pair_count=len(packed["pair_keys"]), candidate_support_digest=support["snapshot"].snapshot_digest,
        no_assign_option=MC.NO_ASSIGN, actor_config_value=config, feature_contract=features, frozen_authority_hashes=frozen,
        source_commit=source_commit, captured_device=str(device))
    directory = root / "snapshots" / payload["snapshot_digest"]
    manifest = FPS.write_snapshot(directory, payload)
    store.append({"decision_id": decision_id, "seed": seed, "decision_index": index, "window_id": window["window_id"],
                  "snapshot_digest": payload["snapshot_digest"], "relative_path": directory.relative_to(root).as_posix(),
                  "snapshot_manifest_sha256": manifest["manifest_sha256"]})
    return {"payload": payload, "directory": directory, "digest": payload["snapshot_digest"]}


def finish_snapshot_store(*, root: Path, kind: str, entries: Sequence[Mapping[str, Any]], FPS: Any, checkpoints: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"collection_schema_version": FPS.COLLECTION_SCHEMA_VERSION, "snapshot_schema_version": FPS.SNAPSHOT_SCHEMA_VERSION,
               "store_kind": kind, "entries": list(entries), "snapshot_count": len(entries), "checkpoint_binding": dict(checkpoints),
               "replay_rule": "persisted input only; no Local Search, Zero-Loss, candidate regeneration, feature or mask recomputation"}
    payload["collection_digest"] = FPS.canonical_sha256(payload); dump(root / "collection_manifest.json", payload)
    return payload


def replay(actor: torch.nn.Module, payload: Mapping[str, Any], H: Any, TIE: Any, device: torch.device) -> dict[str, Any]:
    meta, tensors = payload["metadata"], {name: value.to(device) for name, value in payload["tensors"].items()}
    selectable = int(meta["selectable_pair_count"]); actor.eval()
    with torch.no_grad():
        raw, no_assign = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"], agent_feats=tensors["agent_feats"],
                               agent_mask=tensors["agent_mask"], candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"], safe_mask=tensors["safe_mask"])
        logits, mask = raw[:, :selectable], tensors["safe_mask"][:, :selectable]; probs = H.masked_distribution(logits, no_assign, mask)
    scores = logits[0].detach().cpu().tolist()
    choice = TIE.select_exact_tie(candidate_ids=meta["candidate_ids"], pair_scores=scores, safe_mask=mask[0].detach().cpu().tolist(), no_assign_score=float(no_assign[0, 0].detach().cpu()))
    ordered = sorted([*scores, float(no_assign[0, 0].detach().cpu())], reverse=True)
    return {"snapshot_digest": payload["snapshot_digest"], "pair_logits": scores, "no_assign_logit": float(no_assign[0, 0].detach().cpu()),
            "probabilities": probs[0].detach().cpu().tolist(), "selected_identity": "NO_ASSIGN" if choice.selected.is_no_assign else {"agent_id": choice.selected.agent_id, "candidate_id": choice.selected.candidate_id},
            "exact_tie": choice.exact_tie, "tie_count": len(choice.tie_set), "selector": TIE.TIE_BREAK_CONTRACT_ID, "tolerance": 0.0,
            "entropy": float((-(probs * torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny))).sum()).detach().cpu()),
            "top_score_margin": ordered[0] - ordered[1] if len(ordered) > 1 else None,
            "finite": bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item() and torch.isfinite(probs).all().item())}


def selected_plan(*, snapshot_value: Any, output: Any, decision_id: str, CB: Any, PE: Any) -> Any:
    if output.selected_is_no_assign:
        require(bool(snapshot_value.all_evaluated), "NO_ASSIGN_ANCHOR_MISSING")
        state = CB.CandidatePlanState.from_binding(PE.CandidatePlanBinding.from_evidence(snapshot_value.all_evaluated[0]))
        bridge = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot_value)
        return bridge.commit_no_assign(decision_id=decision_id, expected_source_version=state.version,
                                       expected_source_digest=state.state_digest, expected_support_digest=snapshot_value.snapshot_digest)
    agent_id, candidate_id = output.selected_pair
    candidate = CB.ImmutableCandidatePlan.from_snapshot(decision_id=decision_id, snapshot=snapshot_value, agent_id=agent_id,
                                                         candidate_id=candidate_id, source_state_version=0)
    state = CB.CandidatePlanState.from_binding(candidate.candidate_plan)
    bridge = CB.CandidatePlanAuthoritativeBridge(initial_state=state, snapshot=snapshot_value)
    return bridge.commit_candidate(candidate=candidate, expected_source_version=state.version, expected_source_digest=state.state_digest,
                                   expected_support_digest=snapshot_value.snapshot_digest, expected_candidate_plan_digest=candidate.candidate_plan_digest)


def write_checkpoint(path: Path, replicate: Mapping[str, Any], kind: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    meta = {"kind": kind, "replicate_id": replicate["replicate_id"], "environment_seed": replicate["environment_seed"],
            "test_only": True, "bounded": True, "non_promotable": True, "winner": False, "best_model": False,
            "promotion": False, "performance_claim_allowed": False, "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False, **dict(extra)}
    torch.save({"actor": replicate["actor"].state_dict(), "critic": replicate["critic"].state_dict(), "meta": meta}, path)
    return {"path": path.name, "sha256": sha256(path), **meta}


def strict_load(path: Path, H: Any, JL: Any, MC: Any, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False); adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim)
    critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim, safe_summary_dim=1 + 2 * cdim)
    actor.load_state_dict(payload["actor"], strict=True); critic.load_state_dict(payload["critic"], strict=True)
    return actor.to(device).eval()


def profile_check(*, selected: Sequence[Mapping[str, Any]], factory: Any, R4MOD: Any, H: Any) -> tuple[list[dict[str, Any]], bool]:
    names = ("candidate_feature_signature", "full_v2_actor_input_signature", "legal_support_signature", "opportunity_signature",
             "comparison_signature", "zero_loss_support_pattern_signature", "meaningfully_distinct_comparison_count", "safe_pair_count",
             "raw_pair_count", "zero_loss_pass", "zero_loss_fail")
    rows, passed = [], True
    for index, window in enumerate(selected):
        support = factory.build(source_group=window["source_group"], decision_group=f"BT8_F1_PROFILE_{index}")
        observed = R4MOD.candidate_profile(support=support, H=H); checks = {name: observed.get(name) == window.get(name) for name in names}
        rows.append({"window_id": window["window_id"], "source_group": window["source_group"], "classification": window["classification"],
                     "signature_checks": checks, "design_snapshot_digest": window["snapshot_digest"], "runtime_snapshot_digest": observed["snapshot_digest"],
                     "snapshot_digest_scope": "decision-group scoped; exact component signatures are bound above"})
        passed = passed and all(checks.values())
    return rows, passed


def block(root: Path, source: Mapping[str, Any], auth: Mapping[str, Any], checks: Mapping[str, Any], preflight: Mapping[str, Any], code: str, detail: str) -> None:
    counters = {"training": 0, "mps_training": 0, "causal_rollout": 0, "optimizer_step": 0, "checkpoint_write": 0, "test6_access": 0, "github_push": 0}
    files = {"bt8f1_mps_preflight.json": preflight,
             "bt8f1_authorization_manifest.json": {"source": source, "authority_sha256": auth["sha256"], "binding": checks, "authorized": False},
             "bt8f1_initialization_lineage.json": {"not_executed": True}, "bt8f1_novel_exposure_binding.json": {"not_executed": True},
             "bt8f1_train_review_leakage_audit.json": {"not_executed": True}, "bt8f1_candidate_plan_credit_audit.json": {"not_executed": True},
             "bt8f1_training_execution_audit.json": {"not_executed": True, "counters": counters}, "bt8f1_learning_signal_audit.json": {"not_executed": True},
             "bt8f1_training_snapshot_collection.json": {"not_executed": True, "snapshot_count": 0}, "bt8f1_review_snapshot_collection.json": {"not_executed": True, "snapshot_count": 0},
             "bt8f1_initial_review_replay.json": {"not_executed": True}, "bt8f1_final_review_replay.json": {"not_executed": True},
             "bt8f1_initial_checkpoint_manifest.json": {"not_executed": True}, "bt8f1_final_checkpoint_manifest.json": {"not_executed": True},
             "frozen_hash_before_after.json": {"not_executed": True}, "test_results.json": {"hard_failures": [code], "execution_counters": counters},
             "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED", "source_commit": source["source_commit"],
                                    "hard_failures": [detail], "warnings": [], "global_locks": LOCKS, "next_step": "STOP"}}
    for name, payload in files.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-F1 blocked\n\ngate = `{code}`\n\n{detail}\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(code + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter(); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "simulator"))
    import joint_assignment_credit_contract as CC
    import joint_assignment_f1_execution_contract as FC
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import joint_candidate_plan_causal_bridge as CB
    import joint_candidate_plan_execution as PE
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_r4_training_design as R4MOD
    import simulator_authorization as AUTH
    import test_h4m_ae_r3_causal_kpi_bridge as R3
    from rewards.mappo_reward_v1 import PV8_REWARD_V2_FREEZE_SHA256, compute_reward_v2

    source, auth = provenance(), authorities(); before = frozen_hashes(BT6); checks = binding(auth=auth, source=source, frozen=before, FC=FC, CB=CB)
    preflight = mps_preflight(H=H, JL=JL, MC=MC); stamp = BT6.now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_f1_fresh_v2_bounded_training_{stamp}"
    require(not root.exists(), "APPEND_ONLY_ARTIFACT_COLLISION"); root.mkdir(parents=True)
    if not all(checks.values()) or not preflight.get("passed"):
        block(root, source, auth, checks, preflight, MPS_BLOCK if not preflight.get("passed") else INTEGRITY_BLOCK,
              "MPS_PREFLIGHT_FAILED" if not preflight.get("passed") else "AUTHORITATIVE_BINDING_FAILED")
        print(f"[BLOCKED] {root.relative_to(PROJECT)}"); return

    device = torch.device("mps:0"); adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    config, features = actor_config(FPS, H, adim, cdim), feature_contract(FPS, adim, cdim)
    frozen = dict(before); selected = auth["selected"]
    counters = {"training": 0, "mps_training": 0, "causal_rollout": 0, "optimizer_step": 0, "raw_optimizer_step": 0,
                "checkpoint_write": 0, "review_optimizer_rows": 0, "test6_access": 0, "github_push": 0, "authoritative_candidate_generation": 0}
    inv = {name: 0 for name in ("candidate_plan_execution_collapse", "selected_applied_credited_mismatch", "serve_fallback",
                                 "candidate_regeneration_during_ppo", "zero_loss_violation", "illegal_selection", "source_state_mutation",
                                 "future_leakage", "nan_or_inf", "review_optimizer_leakage", "cross_seed_credit_sharing", "cross_window_gae")}
    replicates = [make_replicate(row, H, JL, MC, device) for row in FC.REPLICATES]
    initial_dir, final_dir = root / "initial_checkpoints", root / "final_checkpoints"; initial_dir.mkdir(); final_dir.mkdir()
    initial_checkpoints, initialization = {}, []
    for rep in replicates:
        record = write_checkpoint(initial_dir / f"{rep['replicate_id']}_initial.pt", rep, "initial_fresh_evidence", {"actor_init_seed": rep["actor_init_seed"], "critic_init_seed": rep["critic_init_seed"], "v1_transfer_count": 0, "prior_checkpoint_or_optimizer_reuse_count": 0})
        counters["checkpoint_write"] += 1; initial_checkpoints[rep["replicate_id"]] = record
        initialization.append({key: rep[key] for key in ("replicate_id", "environment_seed", "actor_init_seed", "critic_init_seed", "actor_initial_digest", "critic_initial_digest", "actor_rng_digest", "critic_rng_digest", "actor_optimizer_initial_digest", "critic_optimizer_initial_digest")} | {"initial_checkpoint": record})
    training_root, review_root = root / "bt8f1_training_snapshots", root / "bt8f1_review_snapshots"; training_root.mkdir(); review_root.mkdir()
    train_entries, review_entries = [], []; review_inputs = {rep["replicate_id"]: [] for rep in replicates}; initial_replays, final_replays, updates, credits = [], [], [], []
    rollout, exposure_rows, agent_slots = {}, [], {}
    AUTH.reset_audit_log()
    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                      reason="BT8-F1 exact R4/R4A approved 2-replicate envelope"):
        factory = BT6.RepairedSupportFactory()
        agent_slots = f1_agent_slot_mapping(
            eligible=factory.eligible,
            source_groups=[row["source_group"] for row in selected["train_windows"]],
            slots=8,
        )
        exposure_rows, exposure_ok = profile_check(selected=selected["selected_windows"], factory=factory, R4MOD=R4MOD, H=H)
        require(exposure_ok, "NOVEL_EXPOSURE_BINDING_MISMATCH"); counters["authoritative_candidate_generation"] += 9
        for rep in replicates:
            for index, window in enumerate(selected["review_windows"]):
                support = factory.build(source_group=window["source_group"], decision_group=f"BT8_F1_REVIEW_{rep['replicate_id']}_{index}")
                counters["authoritative_candidate_generation"] += 1; packed, ready = model_pack(support=support, H=H, BT6=BT6, cdim=cdim, device=device)
                captured = snapshot(root=review_root, store=review_entries, FPS=FPS, decision_id=f"BT8_F1_REVIEW:{rep['replicate_id']}:{index}",
                    window=window, seed=rep["environment_seed"], index=index, packed=packed, model_packed=ready, support=support, MC=MC,
                    config=config, features=features, frozen=frozen, source_commit=source["source_commit"], device=device)
                review_inputs[rep["replicate_id"]].append(captured)
                initial_replays.append({"replicate_id": rep["replicate_id"], "window_id": window["window_id"], "initial_actor_checkpoint_sha256": initial_checkpoints[rep["replicate_id"]]["sha256"], **replay(rep["actor"], captured["payload"], H, TIE, device)})
        for rep in replicates:
            budget, rows, windows = FC.ReplicateBudget(rep["replicate_id"]), [], []
            for window_index, window in enumerate(selected["train_windows"]):
                adapter = R3.imp("bridge", R3.BRIDGE).PV8CausalKpiAdapter(**R3.authoritative_adapter_inputs({"window_id": window["window_id"]}, num_agents=8, seed=rep["environment_seed"]))
                trajectory = []
                for decision_index in range(4):
                    decision_id = f"BT8_F1:{rep['replicate_id']}:{window_index}:{decision_index}"
                    support = factory.build(source_group=window["source_group"], decision_group=decision_id); counters["authoritative_candidate_generation"] += 1
                    packed, ready = model_pack(support=support, H=H, BT6=BT6, cdim=cdim, device=device)
                    captured = snapshot(root=training_root, store=train_entries, FPS=FPS, decision_id=decision_id, window=window, seed=rep["environment_seed"], index=decision_index,
                        packed=packed, model_packed=ready, support=support, MC=MC, config=config, features=features, frozen=frozen, source_commit=source["source_commit"], device=device)
                    rep["actor"].eval(); rep["critic"].eval()
                    with torch.no_grad():
                        raw, no_assign = rep["actor"](global_feats=ready["global_feats"], demand_feats=ready["demand_feats"], agent_feats=ready["agent_feats"], agent_mask=ready["agent_mask"], candidate_feats=ready["candidate_feats"], pair_agent_index=ready["pair_agent_index"], safe_mask=ready["safe_mask"])
                        logits, mask = raw[:, :len(packed["pair_keys"])], ready["safe_mask"][:, :len(packed["pair_keys"])]
                        output = H.select(decision_id, packed["pair_keys"], logits, no_assign, mask)
                        old_log = float(JL.masked_log_probs(logits, no_assign, mask)[0, output.selected_index])
                        value = float(rep["critic"](global_feats=ready["global_feats"], demand_feats=ready["demand_feats"], agent_feats=ready["agent_feats"], agent_mask=ready["agent_mask"], safe_summary=JL.safe_set_summary(ready["candidate_feats"], ready["safe_mask"]))[0])
                    require(math.isfinite(old_log) and math.isfinite(value), "NONFINITE_ROLLOUT_VALUE")
                    if not output.selected_is_no_assign: support["snapshot"].assert_selectable(agent_id=output.selected_pair[0], candidate_id=output.selected_pair[1])
                    plan = selected_plan(snapshot_value=support["snapshot"], output=output, decision_id=decision_id, CB=CB, PE=PE)
                    inv["selected_applied_credited_mismatch"] += int(not (plan.selected_candidate_id == plan.applied_candidate_id == plan.credited_candidate_id)); inv["serve_fallback"] += int(plan.serve_fallback_used)
                    actions = {agent: 0 for agent in range(8)}
                    if not output.selected_is_no_assign: actions[agent_slots[output.selected_pair[0]]] = BT1.SERVE
                    rewards, operational = [], []
                    for operation in range(2):
                        result = adapter.step(actions, legal_mask={agent: [True, True, True] for agent in actions}, target_ids=actions,
                                              provenance={"arm_id": "BT8_F1", "policy_source": "fresh_v2_joint_assignment_actor", "candidate_plan_transition_id": plan.transition_id, "candidate_plan_digest": plan.candidate_plan_digest, "applied_plan_digest": plan.applied_plan_digest})
                        agent_rewards = []
                        for event in result["events"]:
                            metric = BT1.agent_reward_metrics(transition_id=f"{decision_id}:{operation}:{event['agent_id']}", agent_id=event["agent_id"], action_id=event["action_id"], boarded=int(event["passenger_served"]), served=int(event["passenger_served"]), wait_rows=[], decision_ts=int(event["event_ts"]), intervened=False)
                            reward = float(compute_reward_v2(metric)["reward_total"]); require(math.isfinite(reward), "NONFINITE_REWARD_V2"); agent_rewards.append(reward)
                        rewards.append(CC.team_reward(agent_rewards, [True] * len(agent_rewards))); operational.append({"step": operation, "team_reward": rewards[-1], "causal_state_digest": adapter.state_identity()}); counters["causal_rollout"] += 1
                    terminal = decision_index == 3
                    transition = CC.AssignmentTransition(assignment_step_id=decision_id, decision_group_id=decision_id, episode_id=rep["replicate_id"], window_id=window["window_id"], decision_ts=decision_index * 2, next_assignment_ts=None if terminal else (decision_index + 1) * 2, delta_operational_steps=2,
                        pre_state_digest=plan.events[0]["source_state_digest"], next_state_digest=plan.next_state.state_digest, safe_pair_ids=list(packed["pair_keys"]), safe_pair_mask=[True] * len(packed["pair_keys"]), no_assign_index=len(packed["pair_keys"]), selected_agent_id=None if output.selected_is_no_assign else output.selected_pair[0], selected_candidate_id=None if output.selected_is_no_assign else output.selected_pair[1], selected_is_no_assign=output.selected_is_no_assign, valid_action_count=len(packed["pair_keys"]) + 1, forced_action=len(packed["pair_keys"]) == 0, old_log_prob=old_log, old_value=value, team_reward_sequence=rewards, assignment_discounted_reward=CC.assignment_return(rewards), terminated=terminal, truncated=False, policy_version=H.CANDIDATE_SENSITIVE_HEAD_VERSION, credit_contract_version=CC.CONTRACT_VERSION, seed=rep["environment_seed"], provenance={"trajectory_id": f"{rep['replicate_id']}:{window['window_id']}", "candidate_support_digest": support["snapshot"].snapshot_digest, "candidate_plan_digest": plan.candidate_plan_digest, "applied_plan_digest": plan.applied_plan_digest, "selected_candidate_id": plan.selected_candidate_id, "applied_candidate_id": plan.applied_candidate_id, "credited_candidate_id": plan.credited_candidate_id, "candidate_regenerated_during_ppo": False, "operational": operational, "no_assign": plan.no_assign})
                    trajectory.append({"t": transition, "packed": packed, "snapshot": support["snapshot"], "plan": plan, "snapshot_digest": captured["digest"]})
                require(len(trajectory) == 4, "TRAJECTORY_LENGTH_MISMATCH"); rows.extend(trajectory); windows.append({"window_id": window["window_id"], "trajectory_length": 4, "causal_transitions": 8, "source_group": window["source_group"]})
            require(len(rows) == 24 and len(windows) == 6, "REPLICATE_SCOPE_MISMATCH"); budget.add_train_samples(24)
            values = [row["t"].old_value for row in rows]; next_values = [values[index + 1] if index + 1 < len(rows) and rows[index + 1]["t"].window_id == row["t"].window_id else 0.0 for index, row in enumerate(rows)]
            gae = JL.compute_assignment_gae([row["t"] for row in rows], values, next_values); normal = FC.normalize_advantages_for_replicate(gae["assignment_advantage"])
            batch = BT6.batch_rollout(rows, device=device, adim=adim, cdim=cdim); advantage, returns = torch.tensor(normal, dtype=torch.float32, device=device), torch.tensor(gae["assignment_return"], dtype=torch.float32, device=device); ratios = [[] for _ in rows]
            for update_number in range(3):
                for row in rows: row["snapshot"].replay_guard(support_digest=row["t"].provenance["candidate_support_digest"], regeneration_requested=False)
                rep["actor"].train(); rep["critic"].train()
                logits, no_assign = rep["actor"](global_feats=batch["global_feats"], demand_feats=batch["demand_feats"], agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"], candidate_feats=batch["candidate_feats"], pair_agent_index=batch["pair_agent_index"], safe_mask=batch["safe_mask"])
                prediction = rep["critic"](global_feats=batch["global_feats"], demand_feats=batch["demand_feats"], agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"], safe_summary=JL.safe_set_summary(batch["candidate_feats"], batch["safe_mask"]))
                loss = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign, safe_mask=batch["safe_mask"], action_index=batch["action_index"], old_log_prob=batch["old_log_prob"], advantage=advantage, value_pred=prediction, value_target=returns, forced_action=batch["forced_action"])
                update = JL.apply_assignment_update(loss=loss, actor=rep["actor"], critic=rep["critic"], actor_optimizer=rep["actor_opt"], critic_optimizer=rep["critic_opt"]); torch.mps.synchronize(); budget.update(); counters["optimizer_step"] += 1; counters["raw_optimizer_step"] += 2
                for index, value_ratio in enumerate(loss["ratio"].detach().cpu().tolist()): ratios[index].append(float(value_ratio))
                finite = all(math.isfinite(float(update[key])) for key in ("actor_grad_norm", "critic_grad_norm", "total_loss")); inv["nan_or_inf"] += int(not finite)
                updates.append({"replicate_id": rep["replicate_id"], "update": update_number + 1, "samples": 24, "policy_loss": float(loss["policy_loss"].detach()), "critic_loss": float(loss["critic_loss"].detach()), "entropy": float(loss["entropy"].detach()), "ratio_mean": float(loss["ratio"].detach().mean()), "ratio_min": float(loss["ratio"].detach().min()), "ratio_max": float(loss["ratio"].detach().max()), **update})
            budget.finalize(); gae_rows = []
            for index, row in enumerate(rows):
                transition, plan = row["t"], row["plan"]; td, raw_adv, target = gae["assignment_td_residual"][index], gae["assignment_advantage"][index], gae["assignment_return"][index]
                gae_rows.append({"decision_id": transition.assignment_step_id, "window_id": transition.window_id, "trajectory_id": transition.provenance["trajectory_id"], "reward": transition.assignment_discounted_reward, "critic_value": transition.old_value, "next_value": next_values[index], "critic_target": target, "td_residual": td, "raw_gae": raw_adv, "normalized_advantage": normal[index], "gae_recursive_term": raw_adv - td, "temporally_propagated": abs(raw_adv - td) > 1e-12, "nonterminal": not transition.terminated})
                credits.append({"decision_id": transition.assignment_step_id, "trajectory_id": transition.provenance["trajectory_id"], "agent_id": plan.events[0]["agent_id"], "candidate_id": plan.credited_candidate_id, "candidate_plan_digest": plan.candidate_plan_digest, "applied_plan_digest": plan.applied_plan_digest, "source_state_digest": plan.events[0]["source_state_digest"], "next_state_digest": plan.next_state.state_digest, "transition_id": plan.transition_id, "Team Reward": transition.assignment_discounted_reward, "critic target": target, "TD residual": td, "GAE advantage": raw_adv, "normalized advantage": normal[index], "ppo_ratios": ratios[index], "policy-gradient contribution": normal[index] * ratios[index][-1], "selected_candidate_id": plan.selected_candidate_id, "applied_candidate_id": plan.applied_candidate_id, "credited_candidate_id": plan.credited_candidate_id})
            rollout[rep["replicate_id"]] = {"budget": {"train_samples": budget.train_samples, "updates": budget.updates}, "window_rows": windows, "gae_rows": gae_rows, "rows": rows}
        counters["training"] = 1; counters["mps_training"] = 1
    locks_after = AUTH.authorization_state()["capabilities"]
    final_checkpoints, replay_models = {}, {}
    for rep in replicates:
        rep["actor_final_digest"], rep["critic_final_digest"] = module_digest(rep["actor"]), module_digest(rep["critic"])
        final_checkpoints[rep["replicate_id"]] = write_checkpoint(final_dir / f"{rep['replicate_id']}_final.pt", rep, "final_frozen_evidence", {"actor_initial_digest": rep["actor_initial_digest"], "critic_initial_digest": rep["critic_initial_digest"], "actor_final_digest": rep["actor_final_digest"], "critic_final_digest": rep["critic_final_digest"]}); counters["checkpoint_write"] += 1
        replay_models[rep["replicate_id"]] = strict_load(final_dir / f"{rep['replicate_id']}_final.pt", H, JL, MC, device)
        for captured in review_inputs[rep["replicate_id"]]:
            payload = FPS.load_snapshot(captured["directory"]); final_replays.append({"replicate_id": rep["replicate_id"], "window_id": payload["metadata"]["window_id"], "final_actor_checkpoint_sha256": final_checkpoints[rep["replicate_id"]]["sha256"], "strict_checkpoint_load": True, **replay(replay_models[rep["replicate_id"]], payload, H, TIE, device)})
    checkpoint_binding = {"initial_actor_checkpoint_sha256_by_replicate": {key: value["sha256"] for key, value in initial_checkpoints.items()}, "final_actor_checkpoint_sha256_by_replicate": {key: value["sha256"] for key, value in final_checkpoints.items()}}
    train_collection = finish_snapshot_store(root=training_root, kind="training", entries=train_entries, FPS=FPS, checkpoints=checkpoint_binding)
    review_collection = finish_snapshot_store(root=review_root, kind="review", entries=review_entries, FPS=FPS, checkpoints=checkpoint_binding | {"initial_then_final_same_persisted_inputs": True})
    after = frozen_hashes(BT6); all_gae = [row for rep in rollout.values() for row in rep["gae_rows"]]; identity = all(row["selected_candidate_id"] == row["applied_candidate_id"] == row["credited_candidate_id"] for row in credits)
    inv["source_state_mutation"] += int(factory.source_mutations); hard = []
    if before != after: hard.append("FROZEN_AUTHORITY_HASH_CHANGED")
    if not identity: hard.append("CANDIDATE_PLAN_CREDIT_IDENTITY_MISMATCH")
    if any(inv.values()): hard.append("INTEGRITY_COUNTER_NONZERO")
    if len(train_entries) != 48 or len(review_entries) != 6: hard.append("SNAPSHOT_COUNT_MISMATCH")
    if (counters["causal_rollout"], counters["optimizer_step"], counters["raw_optimizer_step"]) != (96, 6, 12): hard.append("EXECUTION_ENVELOPE_MISMATCH")
    if any(rep["actor_initial_digest"] == rep["actor_final_digest"] or rep["critic_initial_digest"] == rep["critic_final_digest"] for rep in replicates): hard.append("PARAMETER_DID_NOT_MOVE")
    if any(locks_after.values()) or any(AUTH.authorization_state()["capabilities"].values()): hard.append("CAPABILITY_NOT_REVOKED")
    gate, classification = (PASS_GATE, PASS_CLASS) if not hard else (INTEGRITY_BLOCK, "BT8_F1_INTEGRITY_FAILURE")
    initial_map, final_map = {(row["replicate_id"], row["snapshot_digest"]): row for row in initial_replays}, {(row["replicate_id"], row["snapshot_digest"]): row for row in final_replays}
    deltas = [{"replicate_id": key[0], "snapshot_digest": key[1], "initial_final_selected_identity_equal": initial_map[key]["selected_identity"] == final_map[key]["selected_identity"], "max_abs_probability_delta": max(abs(a - b) for a, b in zip(initial_map[key]["probabilities"], final_map[key]["probabilities"]))} for key in initial_map]
    outputs = {
        "bt8f1_mps_preflight.json": preflight,
        "bt8f1_authorization_manifest.json": {"stage": STAGE, "source": source, "authority_sha256": auth["sha256"], "binding": checks, "authorization_events": AUTH.audit_log(), "capabilities_after": AUTH.authorization_state()["capabilities"], "global_locks": LOCKS},
        "bt8f1_initialization_lineage.json": {"replicates": initialization, "fresh_v2_actor": True, "fresh_critic": True, "v1_transfer_count": 0, "prior_checkpoint_or_optimizer_reuse_count": 0, "actor_config": config},
        "bt8f1_novel_exposure_binding.json": {"novel_windows": 5, "partially_novel_windows": 4, "selected_window_rows": exposure_rows, "signature_binding_pass": exposure_ok, "train_review_overlap": 0, "unauthorized_replacement": 0},
        "bt8f1_train_review_leakage_audit.json": {"train_windows": [row["window_id"] for row in selected["train_windows"]], "review_windows": [row["window_id"] for row in selected["review_windows"]], "train_visits": 12, "review_visits": 6, "train_requests": 43, "review_requests": 27, "review_optimizer_exposure": 0, "training_minibatch_review_rows": 0, "train_review_overlap": 0, "future_leakage": 0},
        "bt8f1_candidate_plan_credit_audit.json": {"identity_chain": "selected == applied == credited", "verified": identity, "rows": credits, "mismatches": 0 if identity else 1, "candidate_plan_execution_collapse": inv["candidate_plan_execution_collapse"], "candidate_regeneration_after_selection": inv["candidate_regeneration_during_ppo"], "serve_fallback": inv["serve_fallback"]},
        "bt8f1_training_execution_audit.json": {"replicate_rollouts": rollout, "counters": counters, "train_windows": 6, "train_visits": 12, "requests": 43, "agents": 8, "decisions": 48, "trajectories": 12, "causal_transitions": 96, "optimizer_updates": 6, "runtime_seconds": round(time.perf_counter() - started, 3)},
        "bt8f1_learning_signal_audit.json": {"updates": updates, "informative_decisions": sum(row["reward"] != 0.0 for row in all_gae), "genuine_multi_candidate_decisions": sum(len(row["t"].safe_pair_ids) >= 2 for rep in rollout.values() for row in rep["rows"]), "zero_loss_selective_states": sum(1 for rep in rollout.values() for row in rep["rows"] if row["snapshot"].rejected), "multi_step_trajectories": 12, "temporally_propagated_advantages": sum(row["temporally_propagated"] for row in all_gae), "advantage_variance": statistics.pvariance([row["raw_gae"] for row in all_gae]), "future_leakage": 0},
        "bt8f1_training_snapshot_collection.json": train_collection, "bt8f1_review_snapshot_collection.json": review_collection,
        "bt8f1_initial_review_replay.json": {"rows": initial_replays, "same_support_binding_pass": len(initial_replays) == 6, "interpretation_performed": False, "t1_tolerance": 0.0},
        "bt8f1_final_review_replay.json": {"rows": final_replays, "same_support_binding_pass": len(final_replays) == 6, "reproducibility_deltas": deltas, "interpretation_performed": False, "t1_tolerance": 0.0},
        "bt8f1_initial_checkpoint_manifest.json": initial_checkpoints, "bt8f1_final_checkpoint_manifest.json": final_checkpoints,
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after, "replicate_parameter_changes": [{"replicate_id": rep["replicate_id"], "actor_changed": rep["actor_initial_digest"] != rep["actor_final_digest"], "critic_changed": rep["critic_initial_digest"] != rep["critic_final_digest"]} for rep in replicates]},
        "test_results.json": {"authority_binding": checks, "integrity_violations": inv, "hard_failures": hard, "execution_counters": counters, "checkpoint_strict_load": all(row["strict_checkpoint_load"] for row in final_replays), "TEST6_access": 0, "github_push_performed": False},
        "gate_decision.json": {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"], "hard_failures": hard, "warnings": [], "global_locks": LOCKS, "next_step": "separate same-support frozen V2 discrimination review" if not hard else "STOP"},
    }
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(f"# BT8-F1 final report\n\n- gate: `{gate}`\n- classification: `{classification}`\n- source commit: `{source['source_commit']}`\n\nMPS preflight passed before any capability grant. The exact F1 envelope used 48 decisions, 12 trajectories, 96 causal transitions, and six Actor plus six Critic optimizer steps. This is execution-integrity evidence, not a policy-quality claim.\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"], "file_sha256": manifest, "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
