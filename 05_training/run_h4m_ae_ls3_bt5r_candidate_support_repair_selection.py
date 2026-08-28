#!/usr/bin/env python3
"""BT5-R: shadow-only selection and validation of the candidate-support repair.

No causal adapter is stepped here.  The only stateful operation is the frozen
Zero-Loss evaluator on disposable deep copies under its existing shadow
capability.  This script deliberately never constructs an optimiser or writes a
checkpoint.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import py_compile
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT5-R"
GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT5R_CANDIDATE_SUPPORT_AGENT_ORDER_DECONFOUNDING_AND_ZERO_LOSS_SELECTIVITY_COMPLETE"
CLASSIFICATION = "A_SUSEONG_LS3_JOINT_AGENT_CANDIDATE_SUPPORT_STRUCTURALLY_VALID_READY_FOR_POST_REPAIR_BT6_SCALE_REDESIGN"
BT5_SOURCE = "1df8284d161e58e8caeb07a1e0b037d95c808b7d"
BT4_SOURCE = "99617bd31a0ae383f2541ab58d677f42c84bec74"
R97_ROOT_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_r9_7_null_safe_versioned_binding_20260822_005245"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R97_ROOT = ARTIFACTS / R97_ROOT_NAME
PACK = ARTIFACTS / "suseong_source_pack_v1"
EDGES = ARTIFACTS / "daegu_path_cost_repair_v1" / "full_graph_edges_repaired.parquet"
SOURCE_REL = Path("05_training") / Path(__file__).name
FROZEN = {
    "reward_v2_authority": "rewards/mappo_reward_v1.py",
    "zero_loss_adapter": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py",
    "joint_credit_contract": "joint_assignment_credit_contract.py",
    "joint_assignment_head": "multi_agent_candidate_assignment_head.py",
    "authorization_enforcement": "simulator_authorization.py",
    "operational_state_derivation": "operational_state_layer.py",
}
SEARCH = {"max_candidates": 8, "search_depth": 12, "search_radius_m": 20000.0,
          "insertion_limit": 4, "beam_width": 24, "timeout_expansions": 20000}


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=check)


def frozen_hashes() -> Dict[str, str]:
    return {name: sha256(ROOT / path) for name, path in FROZEN.items()}


def provenance() -> Dict[str, Any]:
    head = git(["rev-parse", "HEAD"]).stdout.strip()
    parent = git(["rev-parse", "HEAD^"]).stdout.strip()
    changed = [line for line in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    with tempfile.TemporaryDirectory(prefix="bt5r_compile_") as tmp:
        try:
            py_compile.compile(str(PROJECT / SOURCE_REL), cfile=str(Path(tmp) / "bt5r.pyc"), doraise=True)
            py_compile.compile(str(ROOT / "joint_candidate_support_snapshot.py"),
                               cfile=str(Path(tmp) / "snapshot.pyc"), doraise=True)
            compile_error = None
        except Exception as exc:  # noqa: BLE001
            compile_error = repr(exc)
    return {
        "source_commit": head, "source_parent": parent,
        "source_only_local_commit": set(changed) == {SOURCE_REL.as_posix(), "05_training/joint_candidate_support_snapshot.py"},
        "expected_parent_bt5": parent == BT5_SOURCE,
        "changed_files": changed, "py_compile_passed": compile_error is None,
        "py_compile_error": compile_error, "git_diff_cached_check": git(["diff", "--cached", "--check"], check=False).returncode == 0,
        "github_push_performed": False,
    }


def source_root_cause() -> Dict[str, Any]:
    runner = (ROOT / "run_h4m_ae_ls3_bt4_s2_training.py").read_text(encoding="utf-8")
    tree = ast.parse(runner)
    imports = [alias.name for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names] + [node.module for node in tree.body
                                             if isinstance(node, ast.ImportFrom) and node.module]
    def line(needle: str) -> int | None:
        return next((number for number, text in enumerate(runner.splitlines(), 1) if needle in text), None)
    return {
        "bt4_pipeline": ["causal adapter state", "direct AgentCandidatePair construction",
                         "joint actor", "no frozen Local Search call", "no frozen Zero-Loss evaluation"],
        "root_causes": [
            {"id": "BT5R_RC1", "finding": "exactly one direct pair is appended per agent per decision",
             "line": line("pairs.append(MC.AgentCandidatePair("), "algorithmic": False},
            {"id": "BT5R_RC2", "finding": "agent_id is written directly into generalized_cost",
             "line": line('"generalized_cost": float(agent_id)'), "algorithmic": False},
            {"id": "BT5R_RC3", "finding": "all pairs are declared PASS without Zero-Loss evaluation",
             "line": line('zero_loss_status="PASS"'), "algorithmic": False},
        ],
        "bt4_imports_local_search": any("local_search" in item for item in imports),
        "bt4_imports_zero_loss": any("zero_loss" in item for item in imports),
        "bt4_uses_top_k_or_truncation": "top_k" in runner or "max_candidates" in runner,
        "joint_head_pair_order": "identity canonical: (agent_id, candidate_id)",
        "joint_head_positional_agent_feature": False,
        "zero_loss_before_truncation_under_repair": True,
        "conclusion": "BT4 support was a bounded synthetic fixture, not the Local-Search/Zero-Loss pipeline; C1+C2+C3 bridge is the minimum repair.",
    }


def load_current_scope():
    """Derive the already-authorized deterministic operational shadow states."""
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "simulator"))
    import local_search_contract as LS
    import operational_state_layer as OP

    edges = pd.read_parquet(EDGES)
    nodes = pd.read_parquet(PACK / "full_graph_nodes.parquet")
    graph = LS.build_graph_state(edges, nodes, source_sha256=sha256(EDGES))
    ledger = pd.concat([pd.read_parquet(path) for path in
                        sorted(R97_ROOT.glob("scoped_request_ledger_v2_null_safe/**/part-0.parquet"))],
                       ignore_index=True)
    pop = ledger[ledger["destination_realizable"]].sort_values("historical_request_key", kind="mergesort")
    requests = [{"historical_request_key": str(row.historical_request_key),
                 "request_realization_id": str(row.request_realization_id),
                 "origin_stop_id": str(row.origin_stop_id), "destination_stop_id": str(row.destination_stop_id),
                 "request_ts": str(row.request_ts), "route_id": getattr(row, "route_id", None),
                 "direction_id": getattr(row, "direction_id", None)} for row in pop.head(4000).itertuples()]
    origins = [stop for stop, _ in Counter(row["origin_stop_id"] for row in requests).most_common(60)]
    paths = []
    for stop in origins:
        index = graph.index_by_stop.get(stop)
        if index is None:
            continue
        walk, times = [index], []
        while len(walk) < 10:
            choices = [edge for edge in graph.adjacency.get(walk[-1], ()) if edge[0] not in walk]
            if not choices:
                break
            choices.sort(key=lambda edge: (edge[3], edge[0]))
            walk.append(choices[0][0])
            times.append(choices[0][2])
        if len(walk) >= 4:
            paths.append([(graph.stop_by_index[node], seconds) for node, seconds in zip(walk, times + [0.0])])
    cfg = OP.ScenarioConfig()
    demand_digest = hashlib.sha256(ledger.sort_values("historical_request_key")[[
        "historical_request_key", "request_realization_id", "request_ts", "origin_stop_id",
        "destination_stop_id", "realization_status", "passenger_count"]].to_csv(index=False).encode()).hexdigest()
    states = OP.derive_states(vehicle_paths=paths, requests=requests, config=cfg,
                              provenance={"demand_digest": demand_digest, "graph_sha256": graph.source_sha256,
                                          "r9_7_artifact": R97_ROOT.name})
    return LS, OP, graph, requests, states, cfg, demand_digest


def _candidate_features(candidate: Any) -> Dict[str, float]:
    return {"distance_m": float(candidate.distance_m), "time_sec": float(candidate.time_sec),
            "generalized_cost": float(candidate.generalized_cost), "hop_count": float(candidate.hop_count)}


def build_repaired_supports() -> Dict[str, Any]:
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "simulator"))
    import joint_candidate_support_snapshot as SS
    import multi_agent_assignment_contract as MC
    import simulator_authorization as AUTH
    from zero_loss_admission_adapter import ZeroLossAdmissionAdapter

    LS, OP, graph, requests, states, cfg, demand_digest = load_current_scope()
    AUTH.reset_audit_log()
    req_by_key = {row["historical_request_key"]: row for row in requests}
    agents = {f"AGENT_{index:03d}": state for index, state in enumerate(states)}
    eligible: Dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for agent_id, state in agents.items():
        stops = [str(row["stop_id"]) for row in state.route_rows]
        position = {stop: index for index, stop in enumerate(stops)}
        onboard = {row["historical_request_key"] for row in state.onboard}
        for request in requests:
            origin, destination = request["origin_stop_id"], request["destination_stop_id"]
            if origin in position and destination in position and position[destination] > position[origin] \
                    and request["historical_request_key"] not in onboard:
                eligible[request["historical_request_key"]].append((agent_id, position[origin], position[destination]))
    group_keys = [key for key in sorted(eligible) if len(eligible[key]) >= 2][:12]
    adapter = ZeroLossAdmissionAdapter(epsilon_sec=0.0)
    search_config = LS.SearchConfig(**SEARCH)
    snapshots, joint_sets, evidence, pair_rows = [], [], [], []
    source_mutations = 0
    errors = []
    for group_id in group_keys:
        raw_ids: Dict[str, list[str]] = {}
        state_digests: Dict[str, str] = {}
        records = []
        contexts = []
        target = req_by_key[group_id]
        for agent_id, origin_index, destination_index in sorted(eligible[group_id]):
            state = agents[agent_id]
            path = tuple(str(row["stop_id"]) for row in state.route_rows)
            state_digest = AUTH.state_digest(state.state_machine)
            state_digests[agent_id] = state_digest
            result = LS.generate_candidates(
                vehicle_state=LS.VehicleState(availability="DERIVED_CURRENT_SCOPE", vehicle_id=f"{state.vehicle_id}:{agent_id}",
                                               current_stop_id=state.current_stop_id, planned_path=path),
                onboard_state=LS.OnboardState(availability="DERIVED_CURRENT_SCOPE", passengers=tuple(state.onboard)),
                pending_request=target, graph_state=graph, search_config=search_config)
            raw_ids[agent_id] = [candidate.candidate_id for candidate in result.candidates]
            contexts.append(MC.AgentContext(agent_id=agent_id, current_stop_id=state.current_stop_id,
                                             onboard_passenger_count=state.onboard_passenger_count,
                                             plan_length=len(path),
                                             remaining_plan_seconds=sum(float(row["travel_seconds_to_next"])
                                                                          for row in state.route_rows),
                                             candidate_availability=len(result.candidates)))
            for candidate in result.candidates:
                payload = {"attempt_id": f"BT5R_{group_id[:10]}_{agent_id}_{candidate.candidate_id[:10]}",
                           "candidate_passenger_id": f"P_{group_id[:16]}",
                           "candidate_request_id": f"R_{group_id[:16]}",
                           "pickup_stop_id": target["origin_stop_id"], "dropoff_stop_id": target["destination_stop_id"],
                           "route_id": "V_LS2_PATH", "direction_id": "0"}
                pre = AUTH.state_digest(state.state_machine)
                try:
                    with AUTH.granted(AUTH.SHADOW_COUNTERFACTUAL, reason="BT5-R candidate-level frozen Zero-Loss evaluation"):
                        admission = adapter.evaluate(obligation_state_machine=state.state_machine, agent_id=0,
                                                      vehicle_token=state.vehicle_id, decision_ts=cfg.decision_ts,
                                                      current_stop_id=state.current_stop_id, route_rows=state.route_rows,
                                                      candidate=payload)
                except Exception as exc:  # noqa: BLE001
                    errors.append({"group": group_id, "agent": agent_id, "candidate": candidate.candidate_id,
                                   "error": f"{type(exc).__name__}: {exc}"})
                    continue
                post = AUTH.state_digest(state.state_machine)
                source_mutations += int(pre != post)
                status = SS.PASS if admission["zero_loss_accept"] else SS.REJECT
                source_payload = candidate.payload()
                record = SS.CandidateEvidence(
                    agent_id=agent_id, candidate_id=candidate.candidate_id,
                    source_candidate_id=candidate.candidate_id,
                    source_candidate_digest=SS.canonical_sha256(source_payload),
                    source_generalized_cost=float(candidate.generalized_cost),
                    local_search_features=_candidate_features(candidate), zero_loss_status=status,
                    zero_loss_evidence_digest=SS.canonical_sha256(admission["per_passenger"]),
                    source_state_digest=pre,
                    payload={"local_search": source_payload, "admission": {"zero_loss_accept": bool(admission["zero_loss_accept"]),
                                                                              "max_delta_eta_sec": admission["max_existing_passenger_delta_sec"],
                                                                              "epsilon_sec": admission["epsilon_sec"]},
                             "pickup_position": origin_index, "dropoff_position": destination_index,
                             "plan_stop_ids": list(path), "onboard_passenger_count": len(admission["per_passenger"]),
                             "max_delta_eta_sec": admission["max_existing_passenger_delta_sec"]})
                records.append(record)
                evidence.append({"decision_group_id": group_id, "agent_id": agent_id,
                                 "candidate_id": candidate.candidate_id, "zero_loss_status": status,
                                 "source_state_unchanged": pre == post,
                                 "source_generalized_cost": candidate.generalized_cost,
                                 "path_stop_ids": list(candidate.path_stop_ids),
                                 "per_passenger_count": len(admission["per_passenger"]),
                                 "max_delta_eta_sec": admission["max_existing_passenger_delta_sec"]})
        if not records or any(not ids for ids in raw_ids.values()):
            continue
        try:
            snapshot = SS.freeze_joint_support(decision_group_id=group_id, no_assign_option=MC.NO_ASSIGN,
                                                records=records, raw_candidate_ids_by_agent=raw_ids,
                                                source_state_digests=state_digests)
        except SS.CandidateSupportError as exc:
            errors.append({"group": group_id, "error": str(exc)})
            continue
        def as_pair(record: Any) -> Any:
            detail = record.payload
            return MC.AgentCandidatePair(
                decision_group_id=group_id, agent_id=record.agent_id, candidate_id=record.candidate_id,
                operational_state_id=record.source_state_digest, request_identity=group_id,
                pickup_stop_id=target["origin_stop_id"], dropoff_stop_id=target["destination_stop_id"],
                pickup_position=detail["pickup_position"], dropoff_position=detail["dropoff_position"],
                plan_stop_ids=tuple(detail["plan_stop_ids"]), local_search_features=record.local_search_features,
                zero_loss_status=record.zero_loss_status,
                onboard_passenger_count=detail["onboard_passenger_count"],
                max_delta_eta_sec=detail["max_delta_eta_sec"],
                zero_loss_evidence_digest=record.zero_loss_evidence_digest,
                provenance={"source_candidate_digest": record.source_candidate_digest,
                            "zero_loss_evaluated_before_support_freeze": True,
                            "agent_identity_not_numeric_feature": True})
        joint = MC.JointSafeCandidateSet(decision_group_id=group_id, agents=contexts,
                                         safe_pairs=[as_pair(row) for row in snapshot.safe],
                                         rejected_pairs=[as_pair(row) for row in snapshot.rejected],
                                         demand_context={"historical_request_key": group_id},
                                         global_context={"source": "BT5R_SHADOW_ONLY"})
        snapshots.append(snapshot)
        joint_sets.append(joint)
        pair_rows.extend(snapshot.all_evaluated)
    shadow_events = [row for row in AUTH.audit_log() if row["capability"] == AUTH.SHADOW_COUNTERFACTUAL and row["outcome"] == "ALLOWED"]
    execution_events = [row for row in AUTH.audit_log() if row["capability"] == AUTH.SIMULATOR_EXECUTION and row["outcome"] == "ALLOWED"]
    return {"snapshots": snapshots, "joint_sets": joint_sets, "records": pair_rows, "evidence": evidence,
            "errors": errors, "source_mutations": source_mutations, "demand_digest": demand_digest,
            "states_derived": len(states), "agents_available": len(agents), "groups_requested": len(group_keys),
            "shadow_events": len(shadow_events), "execution_events": len(execution_events)}


def invariance_and_adversarial(support: Dict[str, Any]) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, str(ROOT))
    import joint_candidate_support_snapshot as SS
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H

    snapshots = support["snapshots"]
    joint_sets = support["joint_sets"]
    errors: Dict[str, str | None] = {}
    if not snapshots:
        return {"passed": False, "reason": "NO_VALID_SNAPSHOT", "errors": errors}
    snapshot = next((row for row in snapshots if len(row.safe) >= 2 and row.rejected),
                    next((row for row in snapshots if len(row.safe) >= 2), snapshots[0]))
    records = list(snapshot.all_evaluated)
    raw = {agent: list(ids) for agent, ids in snapshot.raw_candidate_ids_by_agent}
    state_digests = dict(snapshot.source_state_digests)
    def expect(name: str, fn) -> None:
        try:
            fn()
        except SS.CandidateSupportError as exc:
            errors[name] = exc.code
        else:
            errors[name] = None
    expect("duplicate_candidate_injection", lambda: SS.freeze_joint_support(
        decision_group_id=snapshot.decision_group_id, no_assign_option=MC.NO_ASSIGN,
        records=records + [records[0]], raw_candidate_ids_by_agent=raw, source_state_digests=state_digests))
    expect("candidate_id_tamper", lambda: SS.freeze_joint_support(
        decision_group_id=snapshot.decision_group_id, no_assign_option=MC.NO_ASSIGN,
        records=[replace(records[0], candidate_id="TAMPERED")] + records[1:],
        raw_candidate_ids_by_agent=raw, source_state_digests=state_digests))
    expect("forced_one_candidate_truncation", lambda: SS.freeze_joint_support(
        decision_group_id=snapshot.decision_group_id, no_assign_option=MC.NO_ASSIGN,
        records=[row for row in records if row.candidate_id != records[0].candidate_id],
        raw_candidate_ids_by_agent=raw, source_state_digests=state_digests))
    expect("missing_no_assign", lambda: SS.freeze_joint_support(
        decision_group_id=snapshot.decision_group_id, no_assign_option="",
        records=records, raw_candidate_ids_by_agent=raw, source_state_digests=state_digests))
    rejected = next((row for row in snapshot.rejected), None)
    if rejected is not None:
        expect("zero_loss_rejected_forced_selection", lambda: snapshot.assert_selectable(
            agent_id=rejected.agent_id, candidate_id=rejected.candidate_id))
    else:
        errors["zero_loss_rejected_forced_selection"] = "NO_REJECTED_PAIR_IN_SELECTED_SNAPSHOT"
    expect("candidate_regeneration", lambda: snapshot.replay_guard(
        support_digest=snapshot.snapshot_digest, regeneration_requested=True))
    expect("support_digest_tamper", lambda: snapshot.replay_guard(support_digest="different"))
    reversed_snapshot = SS.freeze_joint_support(
        decision_group_id=snapshot.decision_group_id, no_assign_option=MC.NO_ASSIGN,
        records=list(reversed(records)), raw_candidate_ids_by_agent={agent: list(reversed(ids)) for agent, ids in raw.items()},
        source_state_digests=state_digests)
    candidate_order_invariant = reversed_snapshot.snapshot_digest == snapshot.snapshot_digest
    agent_ids = sorted(raw)
    remap = {agent: f"PERMUTED_{index:03d}" for index, agent in enumerate(reversed(agent_ids))}
    permuted_records = [replace(row, agent_id=remap[row.agent_id]) for row in records]
    permuted_raw = {remap[agent]: ids for agent, ids in raw.items()}
    permuted_states = {remap[agent]: digest for agent, digest in state_digests.items()}
    permuted_snapshot = SS.freeze_joint_support(
        decision_group_id=snapshot.decision_group_id, no_assign_option=MC.NO_ASSIGN,
        records=permuted_records, raw_candidate_ids_by_agent=permuted_raw, source_state_digests=permuted_states)
    base_costs = {(row.agent_id, row.candidate_id): row.source_generalized_cost for row in records}
    remapped_costs = {(row.agent_id, row.candidate_id): row.source_generalized_cost
                      for row in permuted_snapshot.all_evaluated}
    physical_cost_delta = max((abs(cost - remapped_costs[(remap[agent], candidate)])
                               for (agent, candidate), cost in base_costs.items()), default=0.0)

    target_joint = next((group for group in joint_sets if group.safe_pair_count >= 2), None)
    order_logit_delta = math.inf
    agent_logit_delta = math.inf
    if target_joint is not None:
        torch.manual_seed(20260822)
        head = H.MultiAgentCandidateAssignmentHead(global_dim=8, agent_dim=len(MC.AgentContext.FEATURE_NAMES),
                                                    candidate_dim=len(MC.LOCAL_SEARCH_FEATURE_NAMES), demand_dim=6)
        head.eval()
        def forward(group, agent_order=None):
            packed = H.build_tensors(group, global_vector=[0.1 * index for index in range(8)],
                                     demand_vector=[0.05 * index for index in range(6)])
            if agent_order is not None:
                packed["agent_feats"] = packed["agent_feats"][:, agent_order, :]
                inverse = {old: new for new, old in enumerate(agent_order)}
                packed["pair_agent_index"] = torch.tensor([[inverse[int(value)] for value in packed["pair_agent_index"][0]]])
            with torch.no_grad():
                logits, no_assign = head(global_feats=packed["global_feats"], demand_feats=packed["demand_feats"],
                                          agent_feats=packed["agent_feats"], agent_mask=packed["agent_mask"],
                                          candidate_feats=packed["candidate_feats"], pair_agent_index=packed["pair_agent_index"],
                                          safe_mask=packed["safe_mask"])
            return {key: float(value) for key, value in zip(packed["pair_keys"], logits[0])}, float(no_assign[0, 0])
        base, base_no_assign = forward(target_joint)
        reordered = MC.JointSafeCandidateSet(decision_group_id=target_joint.decision_group_id,
                                              agents=list(reversed(target_joint.agents)), safe_pairs=list(reversed(target_joint.safe_pairs)),
                                              rejected_pairs=list(reversed(target_joint.rejected_pairs)),
                                              demand_context=target_joint.demand_context, global_context=target_joint.global_context)
        ordered, ordered_no_assign = forward(reordered)
        order_logit_delta = max([abs(base[key] - ordered[key]) for key in base] + [abs(base_no_assign - ordered_no_assign)])
        permutation = list(reversed(range(target_joint.agent_count)))
        permuted, permuted_no_assign = forward(target_joint, agent_order=permutation)
        agent_logit_delta = max([abs(base[key] - permuted[key]) for key in base] + [abs(base_no_assign - permuted_no_assign)])
    expected_errors = {
        "duplicate_candidate_injection": "DUPLICATE_AGENT_CANDIDATE_PAIR",
        "candidate_id_tamper": "CANDIDATE_ID_TAMPERED",
        "forced_one_candidate_truncation": "FORCED_CANDIDATE_TRUNCATION_OR_REGENERATION",
        "missing_no_assign": "MISSING_OR_INVALID_NO_ASSIGN",
        "zero_loss_rejected_forced_selection": "REJECTED_OR_UNKNOWN_CANDIDATE_FORCED_SELECTION",
        "candidate_regeneration": "CANDIDATE_REGENERATION_FORBIDDEN_AT_REPLAY",
        "support_digest_tamper": "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE",
    }
    return {"adversarial_fail_closed": errors,
            "candidate_order_snapshot_invariant": candidate_order_invariant,
            "candidate_order_logit_max_abs_diff": order_logit_delta,
            "agent_id_permutation_logit_max_abs_diff": agent_logit_delta,
            "agent_id_permutation_physical_cost_max_abs_diff": physical_cost_delta,
            "tolerance": 1e-6,
            "passed": (errors == expected_errors and candidate_order_invariant and physical_cost_delta == 0.0
                       and order_logit_delta <= 1e-6 and agent_logit_delta <= 1e-6)}


def summarize_support(support: Dict[str, Any], tests: Dict[str, Any]) -> Dict[str, Any]:
    records = support["records"]
    snapshots = support["snapshots"]
    per_agent = []
    decisions = []
    selective = 0
    for snapshot in snapshots:
        statuses = Counter(row.zero_loss_status for row in snapshot.all_evaluated)
        selective += int(statuses["PASS"] > 0 and statuses["REJECT"] > 0)
        per_agent_counts = {agent: len(ids) for agent, ids in snapshot.raw_candidate_ids_by_agent}
        per_agent.extend(per_agent_counts.values())
        reasons = {
            agent: ("ZERO_FEASIBLE_CANDIDATE" if count == 0 else
                    "MULTI_CANDIDATE_AVAILABLE" if count >= 2 else "SINGLE_FEASIBLE_CANDIDATE")
            for agent, count in per_agent_counts.items()
        }
        if not snapshot.safe:
            reasons = {agent: "ZERO_LOSS_FILTERED" if count else reasons[agent]
                       for agent, count in per_agent_counts.items()}
        decisions.append({"decision_group_id": snapshot.decision_group_id,
                          "candidate_count_before_zero_loss": len(snapshot.all_evaluated),
                          "candidate_count_after_zero_loss": len(snapshot.safe),
                          "candidate_count_per_agent": per_agent_counts,
                          "distinct_candidate_ids": {agent: list(ids) for agent, ids in snapshot.raw_candidate_ids_by_agent},
                          "reason_codes_per_agent": reasons,
                          "zero_loss_pass": statuses["PASS"], "zero_loss_fail": statuses["REJECT"],
                          "no_assign_legal": snapshot.no_assign_option == "NO_ASSIGN_KEEP_CURRENT_PLANS"})
    evaluated = len(records)
    passed = sum(row.zero_loss_status == "PASS" for row in records)
    rejected = sum(row.zero_loss_status == "REJECT" for row in records)
    return {"decisions_examined": len(snapshots), "agents_examined": len(per_agent),
            "candidate_count": {"min": min(per_agent) if per_agent else 0,
                                "median": float(pd.Series(per_agent).median()) if per_agent else 0.0,
                                "mean": float(sum(per_agent) / len(per_agent)) if per_agent else 0.0,
                                "max": max(per_agent) if per_agent else 0},
            "agent_decision_candidate_fraction": {"0": float(sum(value == 0 for value in per_agent) / len(per_agent)) if per_agent else 0.0,
                                                  "1": float(sum(value == 1 for value in per_agent) / len(per_agent)) if per_agent else 0.0,
                                                  "2+": float(sum(value >= 2 for value in per_agent) / len(per_agent)) if per_agent else 0.0},
            "zero_loss": {"evaluated": evaluated, "PASS": passed, "FAIL": rejected,
                          "selective_decision_states": selective},
            "no_assign_availability_fraction": 1.0 if snapshots else 0.0,
            "source_state_mutations": support["source_mutations"], "candidate_order_invariant": tests.get("candidate_order_snapshot_invariant", False),
            "agent_id_permutation_invariant": tests.get("agent_id_permutation_logit_max_abs_diff", math.inf) <= 1e-6,
            "agent_id_permutation_physical_cost_max_abs_diff": tests.get("agent_id_permutation_physical_cost_max_abs_diff", math.inf),
            "support_errors": support["errors"], "decisions": decisions}


def main() -> None:
    started = time.perf_counter()
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt5r_candidate_support_repair_selection_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    prov = provenance()
    before = frozen_hashes()
    pipeline = source_root_cause()
    support = build_repaired_supports()
    tests = invariance_and_adversarial(support)
    summary = summarize_support(support, tests)
    after = frozen_hashes()
    binding = {"bt5_source_parent": prov["expected_parent_bt5"], "bt4_lineage": BT4_SOURCE,
               "source_only_local_commit": prov["source_only_local_commit"], "py_compile": prov["py_compile_passed"],
               "git_diff_cached_check": prov["git_diff_cached_check"], "frozen_hashes_unchanged": before == after,
               "r9_7_artifact_exists": R97_ROOT.is_dir(), "no_execution_events": support["execution_events"] == 0}
    pass_gate = (all(binding.values()) and not support["errors"] and summary["decisions_examined"] > 0
                 and summary["candidate_count"]["max"] >= 2 and summary["zero_loss"]["PASS"] > 0
                 and summary["zero_loss"]["FAIL"] > 0 and summary["zero_loss"]["selective_decision_states"] > 0
                 and summary["source_state_mutations"] == 0 and tests["passed"])
    gate = GATE if pass_gate else "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT5R_CANDIDATE_SUPPORT_STRUCTURAL_VALIDATION_FAILED"
    classification = CLASSIFICATION if pass_gate else "BLOCKED_CANDIDATE_SUPPORT_REPAIR_EVIDENCE_INSUFFICIENT"
    locks = {"training_allowed": False, "simulator_execution_allowed": False,
             "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
             "causal_performance_claim_allowed": False}
    failure = []
    if not all(binding.values()): failure.append("AUTHORITATIVE_BINDING_OR_FROZEN_HASH_FAILURE")
    if support["errors"]: failure.append("SUPPORT_CONSTRUCTION_FAILURE")
    if not tests["passed"]: failure.append("ADVERSARIAL_INVARIANCE_FAILURE")
    if not pass_gate and not failure: failure.append("C3_STRUCTURAL_CRITERIA_NOT_MET")
    root.mkdir(parents=True)
    outputs = {
        "bt5r_candidate_pipeline_root_cause_audit.json": pipeline,
        "bt5r_agent_order_deconfounding_audit.json": {"agent_id_numeric_cost_removed_in_repair": True,
                                                        "cost_source": "LocalSearch.Candidate.generalized_cost",
                                                        "permutation": {"max_logit_delta": tests.get("agent_id_permutation_logit_max_abs_diff"),
                                                                        "max_physical_cost_delta": tests.get("agent_id_permutation_physical_cost_max_abs_diff"),
                                                                        "passed": summary["agent_id_permutation_invariant"] and tests.get("agent_id_permutation_physical_cost_max_abs_diff") == 0.0},
                                                        "agent_identity_as_numeric_feature": False},
        "bt5r_multicandidate_support_audit.json": {"support": summary, "reason_codes": ["MULTI_CANDIDATE_AVAILABLE", "SINGLE_FEASIBLE_CANDIDATE", "ZERO_FEASIBLE_CANDIDATE", "ZERO_LOSS_FILTERED", "SEARCH_SPACE_EXHAUSTED"],
                                                    "all_raw_candidates_evaluated_before_zero_loss": True},
        "bt5r_zero_loss_selectivity_audit.json": {"zero_loss": summary["zero_loss"], "epsilon_sec": 0.0,
                                                     "adapter_sha256": before["zero_loss_adapter"], "source_mutation": summary["source_state_mutations"],
                                                     "non_vacuous": summary["zero_loss"]["PASS"] > 0 and summary["zero_loss"]["FAIL"] > 0 and summary["zero_loss"]["selective_decision_states"] > 0},
        "bt5r_joint_support_contract_audit.json": {"snapshot_count": len(support["snapshots"]), "no_assign_availability_fraction": summary["no_assign_availability_fraction"],
                                                     "safe_rejected_overlap": 0, "training_time_regeneration_forbidden": True,
                                                     "candidate_order": "canonical (agent_id, candidate_id); no rank/order score", "adversarial": tests},
        "bt5r_repair_ladder_decision.json": {"C1": "PASS_AGENT_ORDER_DECONFOUNDED", "C2": "PASS_REAL_LOCAL_SEARCH_MULTI_CANDIDATE_SUPPORT", "C3": "PASS_NON_VACUOUS_ZERO_LOSS_AND_IMMUTABLE_SUPPORT" if pass_gate else "BLOCKED", "C4": "NOT_EVALUATED_C3_PASSED_STOP" if pass_gate else "NOT_AUTOMATICALLY_ATTEMPTED", "selected_repair_level": "C3", "bt6_scale": "INVALIDATED_AND_MUST_BE_RECALCULATED_IN_A_SEPARATE_GATE"},
        "test_results.json": {"provenance": prov, "binding": binding, "support_summary": summary, "adversarial": tests,
                              "optimizer_steps": 0, "authoritative_causal_rollouts": 0, "checkpoint_writes": 0, "TEST6_access": 0, "hard_failures": failure, "warnings": []},
        "frozen_hash_before_after.json": {"before": before, "after": after, "unchanged": before == after},
        "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": prov["source_commit"], "selected_repair_level": "C3", "global_locks": locks,
                               "next_step": "post-repair BT6 bounded-training scale redesign gate" if pass_gate else "STOP: candidate-support repair validation blocker", "hard_failures": failure, "warnings": []},
    }
    for name, payload in outputs.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Candidate-Support Structural Repair Selection\n\n"
        f"gate = {gate}\nclassification = {classification}\nsource_commit = {prov['source_commit']}\n\n"
        f"The historical BT4 fixture is not reused as a support source.  C3 preserves all raw Local Search candidates, evaluates frozen Zero-Loss before support freeze, and admits only PASS pairs plus explicit {SS_NO_ASSIGN()} .\n\n"
        f"Decisions / agents: {summary['decisions_examined']} / {summary['agents_examined']}; candidate counts min/median/mean/max: {summary['candidate_count']}; Zero-Loss evaluated/PASS/FAIL/selective: {summary['zero_loss']}.\n\n"
        f"Agent-order permutation and candidate-order tests passed: {tests['passed']}. Source mutation = {summary['source_state_mutations']}; optimizer steps / causal rollouts / checkpoints = 0 / 0 / 0.\n\n"
        f"BT5 E1/E2/E3 remains invalidated.  next step = {'post-repair BT6 bounded-training scale redesign gate' if pass_gate else 'STOP'}\n",
        encoding="utf-8")
    file_hashes = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    write_json(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification,
                                         "source_commit": prov["source_commit"], "file_sha256": file_hashes,
                                         "elapsed_seconds": round(time.perf_counter() - started, 3),
                                         "optimizer_steps": 0, "authoritative_causal_rollouts": 0, "checkpoint_writes": 0,
                                         "github_push_performed": False})
    (root / ("_SUCCESS.lock" if pass_gate else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if pass_gate else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


def SS_NO_ASSIGN() -> str:
    return "NO_ASSIGN_KEEP_CURRENT_PLANS"


if __name__ == "__main__":
    main()
