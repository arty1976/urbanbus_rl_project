#!/usr/bin/env python3
"""BT8-S0: physical distinctness audit and additional-exposure design only.

The audit reads the preserved BT6 actor-input snapshots, the bound checkpoint,
and append-only prior artifacts.  It does not construct candidates, run
Zero-Loss, step a simulator, train, or write a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-S0"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_S0_ADDITIONAL_TRAINING_ADEQUACY_AND_DISCRIMINATION_EXPOSURE_DESIGN_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_ADDITIONAL_BOUNDED_TRAINING_ENVELOPE_READY_FOR_SEPARATE_EXECUTION_AUTHORIZATION"
BLOCK = "BLOCKED_INSUFFICIENT_CANDIDATE_DISTINCTNESS_EVIDENCE"
BT6_SOURCE = "183e0dae6075e3038305686b151db0c3767212c8"
BT7_R1_SOURCE = "8a4c645906bc3967e0d1d03fe2c7e2912289baf2"
BT7_RERUN2_SOURCE = "dfa18a837d716dfc180418a239ac661aaf1544a3"
T1_TOLERANCE = 0.0
SEEDS = [20260822, 20260823]
DECISIONS_PER_VISIT = 4
TRANSITIONS_PER_VISIT = 8

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = "05_training/run_h4m_ae_ls3_bt8_s0_training_adequacy_design.py"
BT6 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_20260822_200221+09:00"
BT7_R1 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_r1_tie_break_selection_20260822_205519+09:00"
BT7_RERUN2 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt7_rerun2_t1_frozen_policy_review_20260822_211215+09:00"
BT6_S0 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_s0_postrepair_scale_redesign_20260822_152944+09:00"
SNAPSHOT_ROOT = BT6 / "bt7_frozen_policy_snapshots"
COLLECTION_PATH = SNAPSHOT_ROOT / "collection_manifest.json"
CHECKPOINT_PATH = BT6 / "bt6_r2_joint_assignment_checkpoint.pt"
FROZEN = {
    "gatv2_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2": "rewards/mappo_reward_v1.py", "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py", "candidate_support_deconfounding": "joint_candidate_support_snapshot.py",
    "causal_bridge": "causal_kpi_bridge.py", "r9_8_authorization": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py", "joint_assignment_learning": "joint_assignment_learning.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py", "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def provenance() -> dict[str, Any]:
    changed = [row for row in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if row]
    return {"source_commit": git(["rev-parse", "HEAD"]), "source_parent": git(["rev-parse", "HEAD^"]),
            "changed_files": changed, "source_only_local_commit": changed == [SOURCE_REL], "github_push_performed": False}


def frozen_hashes() -> dict[str, str]:
    out = {name: sha256(ROOT / relative) for name, relative in FROZEN.items()}
    out["joint_actor_head"] = sha256(ROOT / "multi_agent_candidate_assignment_head.py")
    return out


def module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.mean(values)) if values else None


def density(rank: int) -> str:
    return "low" if rank == 0 else "high" if rank == 3 else "medium"


def action_identity(selection: Any) -> str | tuple[str, str]:
    return "NO_ASSIGN" if selection.selected.is_no_assign else (str(selection.selected.agent_id), str(selection.selected.candidate_id))


def forward(actor: torch.nn.Module, payload: Mapping[str, Any], *, device: torch.device, head: Any, tie: Any) -> dict[str, Any]:
    meta, cpu = payload["metadata"], payload["tensors"]
    tensors = {name: value.to(device) for name, value in cpu.items()}
    count = int(meta["selectable_pair_count"])
    with torch.no_grad():
        raw, no_assign = actor(global_feats=tensors["global_feats"], demand_feats=tensors["demand_feats"],
                               agent_feats=tensors["agent_feats"], agent_mask=tensors["agent_mask"],
                               candidate_feats=tensors["candidate_feats"], pair_agent_index=tensors["pair_agent_index"], safe_mask=tensors["safe_mask"])
        logits, safe = raw[:, :count], tensors["safe_mask"][:, :count]
        probabilities = head.masked_distribution(logits, no_assign, safe)
    selection = tie.select_exact_tie(candidate_ids=meta["candidate_ids"], pair_scores=logits[0].detach().cpu().tolist(),
                                     safe_mask=safe[0].detach().cpu().tolist(), no_assign_score=float(no_assign[0, 0].detach().cpu()))
    finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item() and torch.isfinite(probabilities).all().item())
    return {"selection": selection, "logits": logits.detach().cpu(), "probabilities": probabilities.detach().cpu(), "finite": finite}


def distinctness_record(payload: Mapping[str, Any], result: Mapping[str, Any], *, feature_names: Sequence[str], agent_feature_names: Sequence[str]) -> dict[str, Any]:
    meta, tensors, selection = payload["metadata"], payload["tensors"], result["selection"]
    ties = [row for row in selection.tie_set if not row.is_no_assign]
    common = {"snapshot_digest": payload["snapshot_digest"], "time_band": meta["time_band"], "d": int(meta["decision_index"]),
              "support_size": int(meta["selectable_pair_count"]), "tie_set_size": len(selection.tie_set), "t1_tolerance": T1_TOLERANCE,
              "candidate_identities": [{"agent_id": row.agent_id, "candidate_id": row.candidate_id} for row in ties]}
    if len(ties) < 2:
        return {**common, "classification": "INSUFFICIENT_METADATA", "reason": "exact top-score tie includes fewer than two physical candidate actions"}
    candidate_values = {index: [float(value) for value in tensors["candidate_feats"][0, row.source_index].tolist()] for index, row in enumerate(ties)}
    agent_values = {index: [float(value) for value in tensors["agent_feats"][0, int(tensors["pair_agent_index"][0, row.source_index])].tolist()] for index, row in enumerate(ties)}
    varying_candidate = [name for offset, name in enumerate(feature_names) if len({values[offset] for values in candidate_values.values()}) > 1]
    varying_agent = [name for offset, name in enumerate(agent_feature_names) if len({values[offset] for values in agent_values.values()}) > 1]
    distinct_agents = len({row.agent_id for row in ties}) > 1
    if varying_candidate or varying_agent:
        label, reason = "MEANINGFULLY_DISTINCT", "preserved physical/operational quantities differ by exact value; no magnitude threshold was introduced"
    elif distinct_agents:
        label, reason = "INSUFFICIENT_METADATA", "different agent identities remain, but all preserved numeric candidate and agent context fields are equal and route/path identity was not retained"
    else:
        label, reason = "PHYSICALLY_EQUIVALENT_OR_NEAR_EQUIVALENT", "all preserved numeric physical/operational fields are exactly equal"
    return {**common, "classification": label, "reason": reason, "distinct_agent_identity": distinct_agents,
            "varying_candidate_features": varying_candidate, "varying_agent_features": varying_agent,
            "unavailable_preserved_fields": ["route/path identity", "stop sequence", "pickup/dropoff stop ids"],
            "candidate_feature_vectors": {str(index): value for index, value in candidate_values.items()},
            "agent_feature_vectors": {str(index): value for index, value in agent_values.items()}}


def counts_by(rows: Sequence[Mapping[str, Any]], key: str, values: Sequence[Any]) -> dict[str, Any]:
    return {str(value): summary([row for row in rows if row[key] == value]) for value in values}


def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows); counts = Counter(row["classification"] for row in rows)
    return {"states": len(rows), "physically_equivalent_or_near_equivalent": counts["PHYSICALLY_EQUIVALENT_OR_NEAR_EQUIVALENT"],
            "meaningfully_distinct": counts["MEANINGFULLY_DISTINCT"], "insufficient_metadata": counts["INSUFFICIENT_METADATA"],
            "meaningfully_distinct_rate": rate(counts["MEANINGFULLY_DISTINCT"], len(rows))}


def selected_window_pool() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ladder = json.loads((BT6_S0 / "bt6s0_scale_candidate_ladder.json").read_text(encoding="utf-8"))["ladder"]
    r3 = next(row for row in ladder if row["label"] == "R3_UPPER_BOUNDED_POST_REPAIR")["windows"]
    used = json.loads((BT6 / "bt6_execution_manifest.json").read_text(encoding="utf-8"))["selected_windows"]
    used_ids = {row["window_id"] for row in used}
    new = [row for row in r3 if row["window_id"] not in used_ids]
    # A1 uses every previously unused, pre-policy-ranked candidate from the
    # authoritative pool plus three rank-extreme repeats solely to retain all
    # density classes.  No policy score, reward, or tie outcome enters here.
    repeats = [
        "PV8_R3R_WEEKDAY_NIGHT_D0_EARLIEST_20230102_0700",
        "PV8_R3R_WEEKDAY_OFFPEAK_D1_EARLIEST_20230102_1000",
        "PV8_R3R_SATURDAY_PEAK_D1_MEDIAN_CHRONOLOGICAL_20230701_1700",
    ]
    lookup = {row["window_id"]: row for row in r3}
    a1 = [*new, *(lookup[row] for row in repeats)]
    if len(new) != 9 or len(a1) != 12 or len({row["window_id"] for row in a1}) != 12:
        raise RuntimeError("BT8S0_NONOUTCOME_WINDOW_POOL_BINDING_FAILURE")
    for row in a1:
        row["exposure_origin"] = "new_since_BT6" if row["window_id"] not in used_ids else "balanced_repeat_for_density_coverage"
        row["density_class_from_prepolicy_rank"] = "low" if row["prepolicy_density_rank"] == 0 else ("high" if row["prepolicy_density_rank"] == 4 else "medium")
    return a1, {"bt6s0_ladder_sha256": sha256(BT6_S0 / "bt6s0_scale_candidate_ladder.json"), "bt6_selected_window_ids": sorted(used_ids),
                "candidate_pool_label": "R3_UPPER_BOUNDED_POST_REPAIR", "candidate_pool_size": len(r3), "new_window_count": len(new), "repeated_window_count": len(repeats),
                "selection_rule": "pre-existing within-band request-count rank and window id only; T1 tie evidence, reward, advantage, KPI, and winner data excluded"}


def expected_counts(*, decisions: int, trajectories: int, transitions: int, updates: int, windows: Sequence[Mapping[str, Any]], current: Mapping[str, Any], repeat_visits: int = 0) -> dict[str, Any]:
    visits = trajectories
    return {"additional_distinct_windows": len(windows), "additional_window_visits": visits, "additional_requests_unique_window_sum": sum(int(row["requests"]) for row in windows),
            "assignment_decisions": decisions, "trajectories": trajectories, "causal_transitions": transitions, "optimizer_updates": updates,
            "expected_informative_decisions": decisions * current["informative_rate"], "expected_genuine_multi_candidate_exposures": decisions * current["multi_rate"],
            "expected_meaningfully_distinct_exact_tie_exposures": decisions * current["meaningful_distinct_tie_decision_rate"],
            "expectation_basis": "current BT6/BT7 preserved observed rates; planning reference only, not an outcome guarantee",
            "all_time_bands": sorted({row["time_band"] for row in windows}) == ["night", "offpeak", "peak"],
            "density_classes": sorted({row["density_class_from_prepolicy_rank"] for row in windows}),
            "mps_runtime_reference": {"BT6_observed_seconds": 3.463, "BT6_observed_decisions": 96, "status": "new-window MPS runtime must be preflight-measured; no unsupported extrapolated ceiling is claimed"},
            "peak_memory_reference": {"BT6_observed_maxrss_bytes": 772440064, "status": "new-window support shapes require MPS preflight; no unsupported memory upper bound is claimed"},
            "repeat_visits_beyond_two_seed_once_per_window": repeat_visits}


def main() -> None:
    started = time.perf_counter(); sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_s0_training_adequacy_design_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists(): raise SystemExit("append-only artifact collision")
    before, source, hard = frozen_hashes(), provenance(), []
    integrity = {key: 0 for key in ("optimizer_steps", "causal_simulator_rollouts", "training_visits", "candidate_regeneration", "zero_loss_reevaluation", "checkpoint_writes", "parameter_mutation", "source_mutation", "checkpoint_mutation", "TEST6_access", "github_push", "nan_or_inf")}
    if not torch.backends.mps.is_available(): hard.append("MPS_FROZEN_REPLAY_ENVIRONMENT_UNAVAILABLE")
    bt6_gate = json.loads((BT6 / "gate_decision.json").read_text(encoding="utf-8"))
    r1_gate = json.loads((BT7_R1 / "gate_decision.json").read_text(encoding="utf-8"))
    rerun2_gate = json.loads((BT7_RERUN2 / "gate_decision.json").read_text(encoding="utf-8"))
    decisions = json.loads((BT6 / "bt6_candidate_support_audit.json").read_text(encoding="utf-8"))["decisions"]
    learning = json.loads((BT6 / "bt6_learning_signal_audit.json").read_text(encoding="utf-8"))
    try:
        collection = FPS.load_collection_manifest(COLLECTION_PATH)
        snapshots = [FPS.load_snapshot(SNAPSHOT_ROOT / entry["relative_path"]) for entry in collection["entries"]]
    except Exception as exc:  # noqa: BLE001
        collection, snapshots = {}, []
        hard.append(f"SNAPSHOT_LOAD_OR_TAMPER_FAILURE:{type(exc).__name__}")
    checkpoint_before = sha256(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    binding: dict[str, Any] = {"bt6_gate": bt6_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_POST_REPAIR_EXTENDED_BOUNDED_TRAINING_AND_BT7_READINESS_COMPLETE", "bt6_source": bt6_gate.get("source_commit") == BT6_SOURCE,
                               "bt7_r1_gate": r1_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT7_R1_ORDER_INVARIANT_TIE_BREAK_SELECTION_SEMANTICS_COMPLETE", "bt7_r1_source": r1_gate.get("source_commit") == BT7_R1_SOURCE,
                               "bt7_rerun2_gate": rerun2_gate.get("gate") == "PASS_WITH_FROZEN_POLICY_DISCRIMINATION_CAUTION", "bt7_rerun2_source": rerun2_gate.get("source_commit") == BT7_RERUN2_SOURCE,
                               "t1_contract": TIE.TIE_BREAK_CONTRACT_ID == "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1", "t1_tolerance_zero": T1_TOLERANCE == 0.0,
                               "source_parent_is_rerun2": source["source_parent"] == BT7_RERUN2_SOURCE, "source_only_local_commit": source["source_only_local_commit"], "checkpoint_exists": CHECKPOINT_PATH.is_file()}
    expected: dict[str, Any] = {}
    if collection:
        cfg = collection["actor_config"]
        expected = FPS.actor_config(global_dim=cfg["global_dim"], demand_dim=cfg["demand_dim"], agent_dim=cfg["agent_dim"], candidate_dim=cfg["candidate_dim"], hidden=cfg["hidden"], heads=cfg["heads"], actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"), actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
        binding.update({"checkpoint_sha": checkpoint_before == collection.get("checkpoint_sha256"), "snapshot_schema": collection.get("snapshot_schema_version") == FPS.SNAPSHOT_SCHEMA_VERSION,
                        "collection_schema": collection.get("collection_schema_version") == FPS.COLLECTION_SCHEMA_VERSION, "snapshot_count_96": len(snapshots) == collection.get("snapshot_count") == 96,
                        "unique_snapshots": len({row["snapshot_digest"] for row in snapshots}) == 96, "actor_config": cfg == expected and collection.get("actor_config_sha256") == FPS.actor_config_sha256(expected), "frozen_authorities": collection.get("frozen_authority_hashes") == before})
    decision_by_digest = {row["frozen_actor_snapshot_digest"]: row for row in decisions}
    binding["decision_snapshot_one_to_one"] = len(decision_by_digest) == 96 and set(decision_by_digest) == {row["snapshot_digest"] for row in snapshots}
    if not all(binding.values()): hard.append("AUTHORITATIVE_BINDING_FAILURE")

    ties: list[dict[str, Any]] = []
    if not hard:
        device = torch.device("mps")
        actor = H.MultiAgentCandidateAssignmentHead(global_dim=expected["global_dim"], demand_dim=expected["demand_dim"], agent_dim=expected["agent_dim"], candidate_dim=expected["candidate_dim"], hidden=expected["hidden"], heads=expected["heads"]).to(device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or "actor" not in checkpoint:
            hard.append("CHECKPOINT_ACTOR_STATE_MISSING")
        else:
            actor.load_state_dict(checkpoint["actor"], strict=True); actor.eval(); parameter_before = module_digest(actor)
            for payload in snapshots:
                result = forward(actor, payload, device=device, head=H, tie=TIE)
                integrity["nan_or_inf"] += int(not result["finite"])
                if result["selection"].exact_tie:
                    record = distinctness_record(payload, result, feature_names=MC.LOCAL_SEARCH_FEATURE_NAMES, agent_feature_names=MC.AgentContext.FEATURE_NAMES)
                    decision = decision_by_digest[payload["snapshot_digest"]]
                    record["density"] = density(int(decision["request_density_stratum"]))
                    record["agent_opportunity_pattern"] = tuple(sorted(pair["agent_id"] for pair in payload["metadata"]["candidate_ids"]))
                    ties.append(record)
            torch.mps.synchronize(); integrity["parameter_mutation"] = int(module_digest(actor) != parameter_before)
    after = frozen_hashes(); integrity["source_mutation"] = int(before != after); integrity["checkpoint_mutation"] = int(checkpoint_before is not None and checkpoint_before != sha256(CHECKPOINT_PATH))
    if len(ties) != 26: hard.append("EXACT_TIE_EVIDENCE_COUNT_MISMATCH")
    if any(integrity.values()): hard.append("FORBIDDEN_EXECUTION_OR_INTEGRITY_COUNTER_NONZERO")

    overall = summary(ties); multi = [row for row in ties if row["support_size"] >= 2]
    feature_prevalence = Counter(feature for row in ties for feature in row.get("varying_candidate_features", []) + row.get("varying_agent_features", []))
    repeated_signatures = Counter(tuple(sorted((tuple(values) for values in row.get("candidate_feature_vectors", {}).values()))) for row in ties)
    tied_audit = {"method": {"allowed_snapshot_fields": ["candidate identity", *MC.LOCAL_SEARCH_FEATURE_NAMES, *MC.AgentContext.FEATURE_NAMES, "support size", "time band", "density", "decision position"],
                              "unavailable_fields_not_inferred": ["route/path identity", "stop sequence", "pickup/dropoff stop ids", "fresh Local Search", "fresh Zero-Loss"],
                              "classification_rule": "exact equality/difference of preserved physical and operational quantities only; no magnitude tolerance or threshold"},
                  "overall": overall, "multi_candidate_only": summary(multi), "feature_difference_prevalence": dict(feature_prevalence), "repeated_exact_feature_signatures": {"unique": len(repeated_signatures), "largest_repeat_count": max(repeated_signatures.values(), default=0)},
                  "by_time_band": counts_by(ties, "time_band", ("night", "offpeak", "peak")), "by_density": counts_by(ties, "density", ("low", "medium", "high")), "by_position": counts_by(ties, "d", range(4)),
                  "by_support_size": {str(size): summary([row for row in ties if row["support_size"] == size]) for size in sorted({row["support_size"] for row in ties})},
                  "by_agent_opportunity_pattern": {str(key): summary([row for row in ties if row["agent_opportunity_pattern"] == key]) for key in sorted({row["agent_opportunity_pattern"] for row in ties})}, "state_records": ties}
    current = {"decisions": 96, "trajectories": 24, "transitions": 192, "optimizer_updates": 10, "informative_decisions": int(learning["informative_decisions"]), "genuine_multi_candidate_decisions": 48,
               "zero_loss_selective_states": sum(row["zero_loss_selective"] for row in decisions), "informative_rate": int(learning["informative_decisions"]) / 96, "multi_rate": 48 / 96,
               "meaningful_distinct_given_exact_tie_rate": overall["meaningfully_distinct_rate"],
               "meaningful_distinct_tie_decision_rate": overall["meaningfully_distinct"] / 96,
               "repeated_exact_feature_signatures": tied_audit["repeated_exact_feature_signatures"]}
    adequate = {"current_exposure": current, "assessment": "INSUFFICIENT_FOR_MEANINGFULLY_DISTINCT_TIE_DISCRIMINATION" if overall["meaningfully_distinct"] else "TIE_RATE_ALONE_DOES_NOT_JUSTIFY_MORE_TRAINING",
                "main_evidence": "meaningfully distinct preserved physical/operational candidates receive exact equal learned scores, while current exposure contains only 20 informative decisions across 24 trajectories and 10 updates", "not_inferred_from_total_decisions_alone": True,
                "no_tie_rate_zero_target": True, "additional_training_justified": bool(overall["meaningfully_distinct"])}
    try:
        a1_windows, pool_binding = selected_window_pool()
    except Exception as exc:  # noqa: BLE001
        a1_windows, pool_binding = [], {"error": type(exc).__name__}
        hard.append("NONOUTCOME_WINDOW_SELECTION_BINDING_FAILURE")
    r3_windows = json.loads((BT6_S0 / "bt6s0_scale_candidate_ladder.json").read_text(encoding="utf-8"))["ladder"][-1]["windows"]
    a2_windows = [dict(row, exposure_origin="new_or_balanced_repeat_from_prior_nonoutcome_pool", density_class_from_prepolicy_rank=("low" if row["prepolicy_density_rank"] == 0 else ("high" if row["prepolicy_density_rank"] == 4 else "medium"))) for row in r3_windows]
    a3_repeat_visits = 6
    ladder = {"selection_pool_binding": pool_binding, "A1_small_extension": expected_counts(decisions=96, trajectories=24, transitions=192, updates=6, windows=a1_windows, current=current),
              "A2_moderate_extension": expected_counts(decisions=120, trajectories=30, transitions=240, updates=8, windows=a2_windows, current=current),
              "A3_upper_bounded_extension": expected_counts(decisions=144, trajectories=36, transitions=288, updates=8, windows=a2_windows, current=current, repeat_visits=a3_repeat_visits),
              "optimizer_design": "A1=3 full same-seed rollout passes per seed; A2/A3=4 per seed. This is below BT6's five passes per seed and keeps PPO reuse bounded by actual per-seed rows.",
              "selection_prohibitions": ["future reward", "future advantage", "future policy outcome", "performance KPI", "winner information", "current-policy tie status"], "all_candidates_are_designs_only": True}
    selected = {"selected_label": "A1_small_extension", "reason": "smallest envelope that adds all nine previously unused pre-policy-ranked windows, retains all bands and low/medium/high density coverage, and adds only three non-outcome balanced repeats", "windows": a1_windows, **ladder["A1_small_extension"],
                "authorization": "NOT_GRANTED", "must_not_execute_automatically": True, "checkpoint_policy": {"test_only": True, "non_promotable": True, "winner": False, "promotion": False, "performance_claim": False}}
    post = {"selector": {"contract_id": TIE.TIE_BREAK_CONTRACT_ID, "tolerance": T1_TOLERANCE, "positive_margin_rule": "positive margins remain model-controlled; no near-tie tolerance"},
            "snapshot_preservation": {"snapshot_schema": "LS3_BT7_FROZEN_POLICY_SNAPSHOT_V1", "collection_schema": "LS3_BT7_FROZEN_POLICY_SNAPSHOT_COLLECTION_V1", "required": ["100% actor-input snapshots", "checkpoint SHA binding", "collection manifest binding", "canonical digest", "T1 selector authority"], "metadata_only_fallback_forbidden": True},
            "next_review": ["order invariance 100% candidate/agent/combined", "integrity violations 0", "report informative and genuine multi-candidate exposure", "report meaningful-distinct exact-tie rate and unique-score-winner fraction", "positive non-tie margins remain model-controlled", "opportunity-adjusted agent concentration re-evaluated", "entropy must not collapse", "persistent physically equivalent ties are explainable and not automatically failure"],
            "decision_rule": "do not require tie rate zero; assess change in meaningfully-distinct ties against preserved physical-distinctness audit"}
    if hard:
        gate, classification, next_step = BLOCK, "BT8S0_BLOCKED", "STOP"
    elif adequate["additional_training_justified"]:
        gate, classification, next_step = PASS_GATE, PASS_CLASS, "separate A1 bounded-training execution authorization; do not execute automatically"
    else:
        gate, classification, next_step = "PASS_SUSEONG_LS3_TIES_PRIMARILY_EQUIVALENT_ADDITIONAL_TRAINING_NOT_REQUIRED_FOR_DISCRIMINATION", "TIES_PRIMARILY_EQUIVALENT", "separate concentration or performance-readiness design gate"
    root.mkdir(parents=True)
    outputs = {"bt8s0_tied_candidate_distinctness_audit.json": tied_audit, "bt8s0_discrimination_adequacy_audit.json": adequate, "bt8s0_exposure_distribution.json": {"current": current, "meaningfully_distinct_tie_by_time_band": tied_audit["by_time_band"], "by_density": tied_audit["by_density"], "by_position": tied_audit["by_position"], "by_support_size": tied_audit["by_support_size"]},
               "bt8s0_additional_training_ladder.json": ladder, "bt8s0_selected_training_envelope.json": selected, "bt8s0_posttraining_review_contract.json": post,
               "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after, "checkpoint_before": checkpoint_before, "checkpoint_after": sha256(CHECKPOINT_PATH)},
               "test_results.json": {"binding": binding, "integrity_counters": integrity, "hard_failures": hard, "warnings": [], "github_push_performed": False},
               "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"], "hard_failures": hard, "warnings": [], "global_locks": LOCKS, "next_step": next_step}}
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(f"# {STAGE} — Additional training adequacy and discrimination exposure design\n\ngate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"Exact ties = {len(ties)}; meaningfully distinct by preserved physical/operational evidence = {overall['meaningfully_distinct']}. "
        f"No candidate was regenerated and no training occurred. A1 is a design only, requiring separate authorization and MPS preflight.\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"], "file_sha256": manifest, "elapsed_seconds": round(time.perf_counter() - started, 3), "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
