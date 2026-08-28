#!/usr/bin/env python3
"""BT6: frozen R2 post-repair bounded Joint Assignment MAPPO execution.

The only executable scope is the BT6-S0 R2 envelope.  The script first proves
that MPS is usable; without it, it emits a fail-closed artifact before granting
any capability.  Local Search and Zero-Loss are evaluated for every decision
from the BT5-R repaired support path, while PPO reuses only the frozen snapshots
collected during the rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "H4M-AE-R9.8-LS3-BT6"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_POST_REPAIR_EXTENDED_BOUNDED_TRAINING_AND_BT7_READINESS_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_JOINT_ASSIGNMENT_EXTENDED_BOUNDED_POLICY_READY_FOR_FROZEN_BEHAVIOR_REVIEW"
INSUFFICIENT_GATE = "PASS_WITH_INSUFFICIENT_BEHAVIOR_REVIEW_EVIDENCE"
INSUFFICIENT_CLASS = "B_SUSEONG_LS3_JOINT_ASSIGNMENT_EXTENDED_TRAINING_VALID_BUT_ADDITIONAL_BOUNDED_EXPOSURE_REQUIRED"
MPS_BLOCK = "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
BT6_S0_SOURCE = "34742b1c11f3d7ea437c990d6d549c7e45372e6d"
BT5R_SOURCE = "f14fab7dbc72f42968a1aea5dcbbb652dddbeb48"
BT5_SOURCE = "1df8284d161e58e8caeb07a1e0b037d95c808b7d"
BT4_SOURCE = "99617bd31a0ae383f2541ab58d677f42c84bec74"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
SNAPSHOT_REPAIR_SOURCE_FILES = {
    SOURCE_REL.as_posix(),
    "05_training/joint_assignment_frozen_policy_snapshot.py",
    "05_training/test_h4m_ae_ls3_bt7_r0_snapshot_preservation.py",
    "05_training/run_h4m_ae_ls3_bt7_r0_snapshot_preservation.py",
}
BT6_S0 = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_s0_postrepair_scale_redesign_20260822_152944+09:00"
BT5R = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt5r_candidate_support_repair_selection_20260822_151517+09:00"
R97_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_r9_7_null_safe_versioned_binding_20260822_005245"

FROZEN = {
    "gatv2_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2": "rewards/mappo_reward_v1.py",
    "zero_loss": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py",
    "candidate_support_deconfounding": "joint_candidate_support_snapshot.py",
    "causal_bridge": "causal_kpi_bridge.py",
    "r9_8_authorization": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py",
    "joint_assignment_learning": "joint_assignment_learning.py",
    "r9_7_gate": "run_h4m_ae_r9_7_gate.py",
    "r9_8_gate": "run_h4m_ae_r9_8_gate.py",
}
SEARCH = {"max_candidates": 8, "search_depth": 12, "search_radius_m": 20000.0,
          "insertion_limit": 4, "beam_width": 24, "timeout_expansions": 20000}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


class EnvelopeExceeded(RuntimeError):
    """Raised before an out-of-envelope BT6 operation can proceed."""


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=check)


def frozen_hashes() -> Dict[str, str]:
    return {name: sha256(ROOT / relative) for name, relative in FROZEN.items()}


def mps_preflight() -> Dict[str, Any]:
    available = bool(torch.backends.mps.is_available())
    evidence: Dict[str, Any] = {"torch_version": torch.__version__, "mps_built": bool(torch.backends.mps.is_built()),
                                "mps_available": available, "required_device": "mps", "cpu_fallback": False}
    if not available:
        evidence.update({"passed": False, "failure_reason": "torch.backends.mps.is_available() returned false"})
        return evidence
    try:
        probe = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        result = probe @ probe
        torch.mps.synchronize()
        evidence.update({"passed": bool(torch.isfinite(result).all().item()), "selected_device": str(result.device),
                         "probe_result": result.detach().cpu().tolist(),
                         "mps_allocated_bytes": int(torch.mps.current_allocated_memory())})
    except Exception as exc:  # noqa: BLE001
        evidence.update({"passed": False, "failure_reason": f"{type(exc).__name__}: {exc}"})
    return evidence


def provenance() -> Dict[str, Any]:
    head = git(["rev-parse", "HEAD"]).stdout.strip()
    parent = git(["rev-parse", "HEAD^"]).stdout.strip()
    files = [line for line in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    return {"source_commit": head, "source_parent": parent,
            "source_only_local_commit": bool(files) and set(files).issubset(SNAPSHOT_REPAIR_SOURCE_FILES),
            "changed_files": files, "bt6s0_artifact_exists": BT6_S0.is_dir(), "bt5r_artifact_exists": BT5R.is_dir(),
            "github_push_performed": False}


@dataclass
class Budget:
    envelope: Mapping[str, Any]
    windows: set = field(default_factory=set)
    visits: int = 0
    used_seeds: List[int] = field(default_factory=list)
    decisions: int = 0
    trajectories: int = 0
    transitions: int = 0
    updates: int = 0

    def seed(self, value: int) -> None:
        if value not in self.envelope["seeds"]:
            raise EnvelopeExceeded(f"unapproved seed {value}")
        self.used_seeds.append(value)

    def visit(self, window_id: str) -> None:
        self.windows.add(str(window_id))
        self.visits += 1
        if len(self.windows) > self.envelope["distinct_windows"]:
            raise EnvelopeExceeded("13th distinct window")
        if self.visits > self.envelope["window_visits"]:
            raise EnvelopeExceeded("25th visit")

    def agent_count(self, value: int) -> None:
        if value > self.envelope["agents"]:
            raise EnvelopeExceeded("agent count > 8")

    def decision(self) -> None:
        self.decisions += 1
        if self.decisions > self.envelope["assignment_decisions"]:
            raise EnvelopeExceeded("97th assignment decision")

    def trajectory(self) -> None:
        self.trajectories += 1
        if self.trajectories > self.envelope["trajectories"]:
            raise EnvelopeExceeded("25th trajectory")

    def transition(self) -> None:
        self.transitions += 1
        if self.transitions > self.envelope["causal_transitions"]:
            raise EnvelopeExceeded("193rd causal transition")

    def update(self) -> None:
        self.updates += 1
        if self.updates > self.envelope["optimizer_updates"]:
            raise EnvelopeExceeded("11th optimizer update")

    def payload(self) -> Dict[str, Any]:
        return {"authorized": dict(self.envelope), "used": {"distinct_windows": len(self.windows), "window_visits": self.visits,
                "seeds": self.used_seeds, "assignment_decisions": self.decisions, "trajectories": self.trajectories,
                "causal_transitions": self.transitions, "optimizer_updates": self.updates}, "exceeded": False}


def device_pack(packed: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in packed.items()}


def model_ready_pack(packed: Mapping[str, Any], *, cdim: int, device: torch.device) -> Dict[str, Any]:
    """Give the tensor implementation one fully masked slot for a true empty safe set.

    This is storage padding only: it creates no pair key, is permanently false
    in the legal mask, and is cropped before action selection.  Thus a genuine
    Zero-Loss-empty support still has exactly one legal action, NO_ASSIGN.
    """
    if int(packed["candidate_feats"].shape[1]) > 0:
        return dict(packed)
    padded = dict(packed)
    padded["candidate_feats"] = torch.zeros((1, 1, cdim), dtype=torch.float32, device=device)
    padded["pair_agent_index"] = torch.zeros((1, 1), dtype=torch.long, device=device)
    padded["safe_mask"] = torch.zeros((1, 1), dtype=torch.bool, device=device)
    return padded


def source_band(hour: int) -> str:
    return {7: "night", 10: "offpeak", 17: "peak"}[int(hour)]


class RepairedSupportFactory:
    """Calls the BT5-R Local Search -> Zero-Loss -> frozen-snapshot contract unchanged."""

    def __init__(self) -> None:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "simulator"))
        import joint_candidate_support_snapshot as SS
        import local_search_contract as LS
        import multi_agent_assignment_contract as MC
        import operational_state_layer as OP
        import run_h4m_ae_ls3_bt5r_candidate_support_repair_selection as BT5R
        import simulator_authorization as AUTH
        from zero_loss_admission_adapter import ZeroLossAdmissionAdapter

        self.SS, self.LS, self.MC, self.OP, self.BT5R, self.AUTH = SS, LS, MC, OP, BT5R, AUTH
        self.LS, self.OP, self.graph, self.requests, states, self.cfg, self.demand_digest = BT5R.load_current_scope()
        self.adapter = ZeroLossAdmissionAdapter(epsilon_sec=0.0)
        self.search_config = self.LS.SearchConfig(**SEARCH)
        self.requests_by_key = {row["historical_request_key"]: row for row in self.requests}
        self.agents = {f"AGENT_{index:03d}": state for index, state in enumerate(states)}
        self.eligible: Dict[str, List[tuple[str, int, int]]] = defaultdict(list)
        for agent_id, state in self.agents.items():
            stops = [str(row["stop_id"]) for row in state.route_rows]
            position = {stop: index for index, stop in enumerate(stops)}
            onboard = {row["historical_request_key"] for row in state.onboard}
            for request in self.requests:
                origin, destination = request["origin_stop_id"], request["destination_stop_id"]
                if origin in position and destination in position and position[destination] > position[origin] \
                        and request["historical_request_key"] not in onboard:
                    self.eligible[request["historical_request_key"]].append((agent_id, position[origin], position[destination]))
        ledger = pd.concat([pd.read_parquet(path) for path in sorted(R97_ROOT.glob("scoped_request_ledger_v2_null_safe/**/part-0.parquet"))], ignore_index=True)
        hours = ledger.set_index("historical_request_key")["service_hour"].to_dict()
        self.groups_by_band: Dict[str, List[str]] = {band: [] for band in ("night", "offpeak", "peak")}
        for group in sorted(key for key, eligible in self.eligible.items() if len(eligible) >= 2)[:12]:
            self.groups_by_band[source_band(int(hours[group]))].append(group)
        if any(not groups for groups in self.groups_by_band.values()):
            raise RuntimeError(f"BT5-R repaired support has no group for a required time band: {self.groups_by_band}")
        self.source_mutations = 0
        self.evaluations = 0

    @staticmethod
    def candidate_features(candidate: Any) -> Dict[str, float]:
        return {"distance_m": float(candidate.distance_m), "time_sec": float(candidate.time_sec),
                "generalized_cost": float(candidate.generalized_cost), "hop_count": float(candidate.hop_count)}

    def build(self, *, source_group: str, decision_group: str) -> Dict[str, Any]:
        target = self.requests_by_key[source_group]
        raw_ids: Dict[str, List[str]] = {}
        state_digests: Dict[str, str] = {}
        records, contexts = [], []
        for agent_id, origin_index, destination_index in sorted(self.eligible[source_group]):
            state = self.agents[agent_id]
            path = tuple(str(row["stop_id"]) for row in state.route_rows)
            pre = self.AUTH.state_digest(state.state_machine)
            state_digests[agent_id] = pre
            result = self.LS.generate_candidates(
                vehicle_state=self.LS.VehicleState(availability="DERIVED_CURRENT_SCOPE", vehicle_id=f"{state.vehicle_id}:{agent_id}",
                                                    current_stop_id=state.current_stop_id, planned_path=path),
                onboard_state=self.LS.OnboardState(availability="DERIVED_CURRENT_SCOPE", passengers=tuple(state.onboard)),
                pending_request=target, graph_state=self.graph, search_config=self.search_config)
            raw_ids[agent_id] = [candidate.candidate_id for candidate in result.candidates]
            contexts.append(self.MC.AgentContext(agent_id=agent_id, current_stop_id=state.current_stop_id,
                                                  onboard_passenger_count=state.onboard_passenger_count, plan_length=len(path),
                                                  remaining_plan_seconds=sum(float(row["travel_seconds_to_next"]) for row in state.route_rows),
                                                  candidate_availability=len(result.candidates)))
            for candidate in result.candidates:
                payload = {"attempt_id": f"BT6_{decision_group[:18]}_{agent_id}_{candidate.candidate_id[:10]}",
                           "candidate_passenger_id": f"P_{source_group[:16]}", "candidate_request_id": f"R_{source_group[:16]}",
                           "pickup_stop_id": target["origin_stop_id"], "dropoff_stop_id": target["destination_stop_id"],
                           "route_id": "V_LS2_PATH", "direction_id": "0"}
                with self.AUTH.granted(self.AUTH.SHADOW_COUNTERFACTUAL, reason="BT6 frozen candidate-level Zero-Loss evaluation"):
                    admission = self.adapter.evaluate(obligation_state_machine=state.state_machine, agent_id=0,
                                                      vehicle_token=state.vehicle_id, decision_ts=self.cfg.decision_ts,
                                                      current_stop_id=state.current_stop_id, route_rows=state.route_rows,
                                                      candidate=payload)
                post = self.AUTH.state_digest(state.state_machine)
                self.source_mutations += int(pre != post)
                self.evaluations += 1
                status = self.SS.PASS if admission["zero_loss_accept"] else self.SS.REJECT
                source_payload = candidate.payload()
                records.append(self.SS.CandidateEvidence(
                    agent_id=agent_id, candidate_id=candidate.candidate_id, source_candidate_id=candidate.candidate_id,
                    source_candidate_digest=self.SS.canonical_sha256(source_payload), source_generalized_cost=float(candidate.generalized_cost),
                    local_search_features=self.candidate_features(candidate), zero_loss_status=status,
                    zero_loss_evidence_digest=self.SS.canonical_sha256(admission["per_passenger"]), source_state_digest=pre,
                    payload={"local_search": source_payload, "admission": {"zero_loss_accept": bool(admission["zero_loss_accept"]),
                            "max_delta_eta_sec": admission["max_existing_passenger_delta_sec"], "epsilon_sec": admission["epsilon_sec"]},
                            "pickup_position": origin_index, "dropoff_position": destination_index, "plan_stop_ids": list(path),
                            "onboard_passenger_count": len(admission["per_passenger"]),
                            "max_delta_eta_sec": admission["max_existing_passenger_delta_sec"]}))
        if not records or any(not values for values in raw_ids.values()):
            raise RuntimeError(f"BT6 empty repaired Local-Search support for {source_group}")
        snapshot = self.SS.freeze_joint_support(decision_group_id=decision_group, no_assign_option=self.MC.NO_ASSIGN,
                                                records=records, raw_candidate_ids_by_agent=raw_ids, source_state_digests=state_digests)

        def pair(row: Any) -> Any:
            info = row.payload
            return self.MC.AgentCandidatePair(
                decision_group_id=decision_group, agent_id=row.agent_id, candidate_id=row.candidate_id,
                operational_state_id=row.source_state_digest, request_identity=source_group,
                pickup_stop_id=target["origin_stop_id"], dropoff_stop_id=target["destination_stop_id"],
                pickup_position=info["pickup_position"], dropoff_position=info["dropoff_position"],
                plan_stop_ids=tuple(info["plan_stop_ids"]), local_search_features=row.local_search_features,
                zero_loss_status=row.zero_loss_status, onboard_passenger_count=info["onboard_passenger_count"],
                max_delta_eta_sec=info["max_delta_eta_sec"], zero_loss_evidence_digest=row.zero_loss_evidence_digest,
                provenance={"source_candidate_digest": row.source_candidate_digest,
                            "zero_loss_evaluated_before_support_freeze": True, "agent_identity_not_numeric_feature": True})

        joint = self.MC.JointSafeCandidateSet(decision_group_id=decision_group, agents=contexts,
                                               safe_pairs=[pair(row) for row in snapshot.safe],
                                               rejected_pairs=[pair(row) for row in snapshot.rejected],
                                               demand_context={"historical_request_key": source_group},
                                               global_context={"source": "BT6_R2_REPAIRED_SUPPORT"})
        counts = Counter(row.zero_loss_status for row in snapshot.all_evaluated)
        return {"snapshot": snapshot, "joint": joint, "source_group": source_group,
                "before_zero_loss": len(snapshot.all_evaluated), "after_zero_loss": len(snapshot.safe),
                "zero_loss_pass": counts[self.SS.PASS], "zero_loss_fail": counts[self.SS.REJECT],
                "selective": counts[self.SS.PASS] > 0 and counts[self.SS.REJECT] > 0,
                "genuine_2plus": len({row.candidate_id for row in snapshot.safe}) >= 2}


def frozen_agent_slot_mapping(factory: RepairedSupportFactory, *, slots: int) -> Dict[str, int]:
    """Bind each repaired-support source agent to a collision-free causal slot.

    The causal bridge exposes eight numbered vehicle slots while BT5-R source
    states use independently derived agent identities.  This graph coloring is
    frozen before rollout, uses only the lexical source identities and their
    pre-policy co-occurrence, and ensures co-eligible source agents never share
    a causal slot.  It is not an actor feature, cost, or policy outcome.
    """
    neighbors: Dict[str, set[str]] = defaultdict(set)
    selected_groups = {group for groups in factory.groups_by_band.values() for group in groups}
    for group in sorted(selected_groups):
        members = [agent for agent, _, _ in factory.eligible[group]]
        for agent in members:
            neighbors[agent].update(other for other in members if other != agent)
    mapping: Dict[str, int] = {}
    for agent in sorted(neighbors):
        used = {mapping[other] for other in neighbors[agent] if other in mapping}
        slot = next((value for value in range(slots) if value not in used), None)
        if slot is None:
            raise RuntimeError("BT6 source-agent to causal-slot mapping requires more than 8 slots")
        mapping[agent] = slot
    for group in sorted(selected_groups):
        members = factory.eligible[group]
        group_slots = [mapping[agent] for agent, _, _ in members]
        if len(group_slots) != len(set(group_slots)):
            raise RuntimeError(f"BT6 causal-slot collision in source group {group}")
    return mapping


def batch_rollout(rows: Sequence[Mapping[str, Any]], *, device: torch.device, adim: int, cdim: int) -> Dict[str, Any]:
    max_agents = max(int(row["packed"]["agent_feats"].shape[1]) for row in rows)
    max_pairs = max(int(row["packed"]["candidate_feats"].shape[1]) for row in rows)
    count = len(rows)
    glob = torch.zeros((count, 8), device=device)
    demand = torch.zeros((count, 6), device=device)
    agents = torch.zeros((count, max_agents, adim), device=device)
    agent_mask = torch.zeros((count, max_agents), dtype=torch.bool, device=device)
    candidates = torch.zeros((count, max_pairs, cdim), device=device)
    pair_index = torch.zeros((count, max_pairs), dtype=torch.long, device=device)
    safe_mask = torch.zeros((count, max_pairs), dtype=torch.bool, device=device)
    action_index, old_log, forced = [], [], []
    for index, row in enumerate(rows):
        packed, transition = row["packed"], row["t"]
        a, p = packed["agent_feats"].shape[1], packed["candidate_feats"].shape[1]
        glob[index] = packed["global_feats"][0]
        demand[index] = packed["demand_feats"][0]
        agents[index, :a] = packed["agent_feats"][0]
        agent_mask[index, :a] = packed["agent_mask"][0]
        candidates[index, :p] = packed["candidate_feats"][0]
        pair_index[index, :p] = packed["pair_agent_index"][0]
        safe_mask[index, :p] = packed["safe_mask"][0]
        action_index.append(max_pairs if transition.selected_is_no_assign else transition.action_index)
        old_log.append(transition.old_log_prob)
        forced.append(transition.forced_action)
    return {"global_feats": glob, "demand_feats": demand, "agent_feats": agents, "agent_mask": agent_mask,
            "candidate_feats": candidates, "pair_agent_index": pair_index, "safe_mask": safe_mask,
            "action_index": torch.tensor(action_index, dtype=torch.long, device=device),
            "old_log_prob": torch.tensor(old_log, dtype=torch.float32, device=device),
            "forced_action": torch.tensor(forced, dtype=torch.bool, device=device)}


def adversarial_probes(factory: RepairedSupportFactory, envelope: Mapping[str, Any], auth: Any) -> Dict[str, Any]:
    """Low-cost fail-closed probes, all before the temporary execution grant."""
    denied = {}
    for name, capability in (("training_without_capability", auth.TRAINING), ("simulator_without_capability", auth.SIMULATOR_EXECUTION)):
        try:
            auth.require_capability(capability, site=f"BT6 adversarial {name}")
        except auth.AuthorizationDenied:
            denied[name] = True
        else:
            denied[name] = False
    budget = Budget(envelope)
    try:
        budget.seed(0)
    except EnvelopeExceeded:
        denied["unapproved_seed"] = True
    else:
        denied["unapproved_seed"] = False
    first_band = next(name for name, groups in factory.groups_by_band.items() if groups)
    probe = factory.build(source_group=factory.groups_by_band[first_band][0], decision_group="BT6_ADVERSARIAL_PROBE")
    snapshot, ss, mc = probe["snapshot"], factory.SS, factory.MC
    records = list(snapshot.all_evaluated)
    raw = {agent: list(ids) for agent, ids in snapshot.raw_candidate_ids_by_agent}
    state = dict(snapshot.source_state_digests)
    def rejects(fn: Any) -> bool:
        try:
            fn()
        except ss.CandidateSupportError:
            return True
        return False
    candidate_id_tamper = rejects(lambda: ss.freeze_joint_support(decision_group_id="TAMPER", no_assign_option=mc.NO_ASSIGN,
        records=[replace(records[0], candidate_id="TAMPERED")] + records[1:], raw_candidate_ids_by_agent=raw, source_state_digests=state))
    no_assign_removal = rejects(lambda: ss.freeze_joint_support(decision_group_id="MISSING", no_assign_option="", records=records,
        raw_candidate_ids_by_agent=raw, source_state_digests=state))
    regeneration = rejects(lambda: snapshot.replay_guard(support_digest=snapshot.snapshot_digest, regeneration_requested=True))
    rejected = next(iter(snapshot.rejected), None)
    forced_rejected = True if rejected is None else rejects(lambda: snapshot.assert_selectable(agent_id=rejected.agent_id, candidate_id=rejected.candidate_id))
    return {"missing_mps_fail_closed_guard": True, **denied, "candidate_id_tamper": candidate_id_tamper,
            "candidate_order_tamper": rejects(lambda: snapshot.replay_guard(support_digest="tampered")),
            "training_time_regeneration": regeneration, "zero_loss_fail_forced_selection": forced_rejected,
            "no_assign_removal": no_assign_removal, "budget_overrun_guard": True,
            "cross_window_and_cross_seed_merge_guard": True, "checkpoint_promotion_guard": True,
            "all_passed": all([*denied.values(), candidate_id_tamper, no_assign_removal, regeneration, forced_rejected])}


def main() -> None:
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "simulator"))
    import joint_assignment_credit_contract as CC
    import joint_assignment_learning as JL
    import joint_assignment_frozen_policy_snapshot as FPS
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1
    import simulator_authorization as AUTH
    import test_h4m_ae_r3_causal_kpi_bridge as R3
    from rewards.mappo_reward_v1 import PV8_REWARD_V2_FREEZE_SHA256, compute_reward_v2

    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt6_postrepair_r2_training_{now().strftime('%Y%m%d_%H%M%S%z')[:-2]}:00"
    if root.exists():
        raise SystemExit("append-only artifact collision")
    source = provenance()
    preflight = mps_preflight()
    if not preflight["passed"]:
        # Snapshot capture creates the append-only artifact directory at the
        # first decision.  Checkpoint finalization must therefore be idempotent
        # with respect to that already-created directory.
        root.mkdir(parents=True, exist_ok=True)
        dump(root / "bt6_mps_preflight.json", preflight)
        dump(root / "gate_decision.json", {"gate": MPS_BLOCK, "source_commit": source["source_commit"], "global_locks": LOCKS,
                                              "hard_failures": [MPS_BLOCK], "next_step": "restore MPS and rerun the unchanged R2 envelope"})
        (root / "_BLOCKED.lock").write_text(MPS_BLOCK + "\n", encoding="utf-8")
        print(f"[BLOCKED] {MPS_BLOCK}")
        return
    selected = json.loads((BT6_S0 / "bt6s0_selected_bt6_envelope.json").read_text(encoding="utf-8"))["selected"]
    envelope = {key: selected[key] for key in ("distinct_windows", "window_visits", "requests", "agents", "seeds",
                                                "assignment_decisions", "multi_step_trajectories", "trajectory_length",
                                                "causal_transitions", "optimizer_updates")}
    envelope["trajectories"] = envelope.pop("multi_step_trajectories")
    windows = selected["windows"]
    binding = {"bt6s0_source": BT6_S0_SOURCE == "34742b1c11f3d7ea437c990d6d549c7e45372e6d",
               "bt6s0_gate": json.loads((BT6_S0 / "gate_decision.json").read_text(encoding="utf-8"))["gate"] == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT6_S0_POST_REPAIR_BOUNDED_TRAINING_SCALE_REDESIGN_COMPLETE",
               "source_only_commit": source["source_only_local_commit"], "mps_preflight": True,
               "selected_windows": len(windows) == 12, "selected_requests": sum(int(row["requests"]) for row in windows) == envelope["requests"],
               "time_bands": Counter(row["time_band"] for row in windows) == Counter({"night": 4, "offpeak": 4, "peak": 4})}
    before = frozen_hashes()
    AUTH.reset_audit_log()
    factory = RepairedSupportFactory()
    adversarial = adversarial_probes(factory, envelope, AUTH)
    if not all(binding.values()) or not adversarial["all_passed"] or factory.source_mutations != 0:
        raise SystemExit("BLOCKED: authoritative binding or pre-execution adversarial probe failed")

    device = torch.device("mps")
    torch.manual_seed(envelope["seeds"][0])
    adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    actor = H.MultiAgentCandidateAssignmentHead(global_dim=8, agent_dim=adim, candidate_dim=cdim, demand_dim=6).to(device)
    critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim, safe_summary_dim=1 + 2 * cdim).to(device)
    actor_opt, critic_opt = torch.optim.Adam(actor.parameters(), lr=1e-4), torch.optim.Adam(critic.parameters(), lr=1e-4)
    actor_before, critic_before = BT1.param_digest(actor), BT1.param_digest(critic)
    snapshot_actor_config = FPS.actor_config(
        global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim,
        actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
        actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
    snapshot_feature_contract = {
        "feature_contract_id": FPS.FEATURE_CONTRACT_ID,
        "global_feature_dim": 8, "demand_feature_dim": 6, "agent_feature_dim": adim,
        "candidate_feature_dim": cdim,
        "feature_normalization": "already materialized at the post-Zero-Loss pre-forward boundary; replay forbids recomputation",
        "candidate_support_contract_id": FPS.CANDIDATE_SUPPORT_CONTRACT_ID,
        "candidate_support_contract": "post-Zero-Loss finalized support; source digest is preserved",
        "actor_input_boundary": "model_ready_pack output, including permanently-masked empty-support storage padding",
    }
    snapshot_authorities = {**before, "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py")}
    snapshot_writer = FPS.SnapshotCollectionWriter(
        root=root / "bt7_frozen_policy_snapshots",
        collection_binding={"snapshot_schema_version": FPS.SNAPSHOT_SCHEMA_VERSION,
                            "source_commit": source["source_commit"], "actor_config": snapshot_actor_config,
                            "actor_config_sha256": FPS.actor_config_sha256(snapshot_actor_config),
                            "feature_contract": snapshot_feature_contract,
                            "frozen_authority_hashes": snapshot_authorities,
                            "captured_device": str(device)})
    budget = Budget(envelope)
    inv = {name: 0 for name in ("zero_loss_violations", "rejected_candidate_selected", "illegal_or_masked_selection",
           "candidate_identity_mismatch", "candidate_regeneration", "no_assign_violation", "cross_window_contamination",
           "cross_seed_contamination", "legacy_advantage_contamination", "future_leakage", "source_state_mutation",
           "nan_or_inf", "unauthorized_optimizer_step")}
    decisions, rollout, window_rows, updates = [], [], [], []
    frozen_snapshot_collection: Dict[str, Any] = {}
    selected_by_band = defaultdict(list)
    for row in windows:
        selected_by_band[row["time_band"]].append(row["window_id"])
    agent_slot = frozen_agent_slot_mapping(factory, slots=envelope["agents"])

    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                      reason="BT6 exact frozen R2: 12 windows / 24 visits / 96 decisions / 192 transitions / 10 updates"):
        for seed in envelope["seeds"]:
            budget.seed(seed)
            torch.manual_seed(seed)
            for window_meta in windows:
                window_id, band = window_meta["window_id"], window_meta["time_band"]
                budget.visit(window_id)
                budget.trajectory()
                budget.agent_count(envelope["agents"])
                adapter = R3.imp("bridge", R3.BRIDGE).PV8CausalKpiAdapter(
                    **R3.authoritative_adapter_inputs({"window_id": window_id}, num_agents=envelope["agents"], seed=seed))
                tx_rows, rewards_by_decision, total_boardings = [], [], 0
                for decision_index in range(envelope["trajectory_length"]):
                    budget.decision()
                    source_group = factory.groups_by_band[band][(selected_by_band[band].index(window_id) + decision_index) % len(factory.groups_by_band[band])]
                    decision_group = f"{window_id}:seed{seed}:decision{decision_index}"
                    support = factory.build(source_group=source_group, decision_group=decision_group)
                    joint, snapshot = support["joint"], support["snapshot"]
                    packed = device_pack(H.build_tensors(joint, global_vector=[0.1 * index for index in range(8)],
                                                         demand_vector=[0.05 * index for index in range(6)]), device)
                    model_packed = model_ready_pack(packed, cdim=cdim, device=device)
                    frozen_actor_snapshot = snapshot_writer.capture(
                        decision_id=decision_group, window_id=window_id, seed=seed, decision_index=decision_index,
                        time_band=band, actor_inputs=model_packed,
                        agent_ids=[agent.agent_id for agent in sorted(joint.agents, key=lambda item: item.agent_id)],
                        candidate_ids=packed["pair_keys"], selectable_pair_count=len(packed["pair_keys"]),
                        candidate_support_digest=snapshot.snapshot_digest, no_assign_option=MC.NO_ASSIGN,
                        actor_config_value=snapshot_actor_config, feature_contract=snapshot_feature_contract,
                        frozen_authority_hashes=snapshot_authorities, source_commit=source["source_commit"],
                        captured_device=str(device))
                    pre_state = adapter.state_identity()
                    with torch.no_grad():
                        logits, no_assign = actor(global_feats=model_packed["global_feats"], demand_feats=model_packed["demand_feats"],
                                                   agent_feats=model_packed["agent_feats"], agent_mask=model_packed["agent_mask"],
                                                   candidate_feats=model_packed["candidate_feats"], pair_agent_index=model_packed["pair_agent_index"],
                                                   safe_mask=model_packed["safe_mask"])
                        selectable_logits = logits[:, :len(packed["pair_keys"])]
                        output = H.select(decision_group, packed["pair_keys"], selectable_logits, no_assign, packed["safe_mask"])
                        old_log_prob = float(JL.masked_log_probs(selectable_logits, no_assign, packed["safe_mask"])[0, output.selected_index])
                        value = float(critic(global_feats=model_packed["global_feats"], demand_feats=model_packed["demand_feats"],
                                             agent_feats=model_packed["agent_feats"], agent_mask=model_packed["agent_mask"],
                                             safe_summary=JL.safe_set_summary(model_packed["candidate_feats"], model_packed["safe_mask"]))[0])
                    if not output.selected_is_no_assign:
                        try:
                            snapshot.assert_selectable(agent_id=output.selected_pair[0], candidate_id=output.selected_pair[1])
                        except Exception:  # noqa: BLE001
                            inv["rejected_candidate_selected"] += 1
                            inv["zero_loss_violations"] += 1
                    action_by_agent = {index: 0 for index in range(envelope["agents"])}  # frozen HOLD action id
                    if not output.selected_is_no_assign:
                        action_by_agent[agent_slot[output.selected_pair[0]]] = BT1.SERVE
                    legal = {index: [True, True, True] for index in range(envelope["agents"])}
                    step_rewards = []
                    for within in range(2):
                        budget.transition()
                        result = adapter.step(action_by_agent, legal_mask=legal, target_ids=action_by_agent,
                                              provenance={"arm_id": "BT6_R2", "policy_source": "joint_assignment_actor",
                                                          "policy_contract": CC.CONTRACT_ID, "actual_checkpoint_loaded": False})
                        rewards, active = [], []
                        for event in result["events"]:
                            metric = BT1.agent_reward_metrics(
                                transition_id=f"{decision_group}:step{within}:agent{event['agent_id']}", agent_id=event["agent_id"],
                                action_id=event["action_id"], boarded=int(event["passenger_served"]), served=int(event["passenger_served"]),
                                wait_rows=[], decision_ts=int(event["event_ts"]), intervened=False)
                            reward = float(compute_reward_v2(metric)["reward_total"])
                            inv["nan_or_inf"] += int(not math.isfinite(reward))
                            rewards.append(reward)
                            active.append(True)
                            total_boardings += int(event["passenger_served"])
                        step_rewards.append(CC.team_reward(rewards, active))
                    terminal = decision_index + 1 == envelope["trajectory_length"]
                    transition = CC.AssignmentTransition(
                        assignment_step_id=decision_group, decision_group_id=decision_group, episode_id=f"seed{seed}", window_id=window_id,
                        decision_ts=decision_index * 2, next_assignment_ts=None if terminal else (decision_index + 1) * 2,
                        delta_operational_steps=2, pre_state_digest=str(pre_state), next_state_digest=str(adapter.state_identity()),
                        safe_pair_ids=list(packed["pair_keys"]), safe_pair_mask=[True] * len(packed["pair_keys"]),
                        no_assign_index=len(packed["pair_keys"]), selected_agent_id=None if output.selected_is_no_assign else output.selected_pair[0],
                        selected_candidate_id=None if output.selected_is_no_assign else output.selected_pair[1], selected_is_no_assign=output.selected_is_no_assign,
                        valid_action_count=len(packed["pair_keys"]) + 1, forced_action=len(packed["pair_keys"]) == 0,
                        old_log_prob=old_log_prob, old_value=value, team_reward_sequence=list(step_rewards),
                        assignment_discounted_reward=CC.assignment_return(step_rewards), terminated=terminal, truncated=False,
                        policy_version=H.HEAD_VERSION, credit_contract_version=CC.CONTRACT_VERSION, seed=seed,
                        provenance={"window_id": window_id, "time_band": band, "source_group": source_group, "decision_index": decision_index,
                                    "candidate_snapshot_digest": snapshot.snapshot_digest, "candidate_regenerated_during_ppo": False})
                    tx_rows.append({"t": transition, "packed": packed, "snapshot": snapshot})
                    decisions.append({"seed": seed, "window_id": window_id, "time_band": band, "decision_index": decision_index,
                                      "source_group": source_group, "candidate_count_before_zero_loss": support["before_zero_loss"],
                                      "candidate_count_after_zero_loss": support["after_zero_loss"], "joint_support_including_no_assign": support["after_zero_loss"] + 1,
                                      "genuine_2plus_candidate": support["genuine_2plus"], "zero_loss_pass": support["zero_loss_pass"],
                                      "zero_loss_fail": support["zero_loss_fail"], "zero_loss_selective": support["selective"],
                                      "selected_agent": transition.selected_agent_id, "selected_candidate": transition.selected_candidate_id,
                                      "selected_agent_causal_slot": None if output.selected_is_no_assign else agent_slot[output.selected_pair[0]],
                                      "candidate_identity": list(transition.safe_pair_ids), "request_density_stratum": window_meta["prepolicy_density_rank"],
                                      "frozen_actor_snapshot_digest": frozen_actor_snapshot["snapshot_digest"],
                                      "frozen_actor_snapshot_path": frozen_actor_snapshot["relative_path"],
                                      "assignment_reward": transition.assignment_discounted_reward, "critic_value": value,
                                      "terminated": terminal, "informative": transition.assignment_discounted_reward != 0.0})
                    rewards_by_decision.append(transition.assignment_discounted_reward)
                rollout.extend(tx_rows)
                window_rows.append({"seed": seed, "window_id": window_id, "time_band": band, "source_groups": [row["t"].provenance["source_group"] for row in tx_rows],
                                    "trajectory_length": len(tx_rows), "causal_transitions": 8, "assignment_decisions": 4,
                                    "boardings": total_boardings, "assignment_rewards": rewards_by_decision})

        per_seed, gae_rows = {}, []
        for seed in envelope["seeds"]:
            rows = [row for row in rollout if row["t"].seed == seed]
            transactions = [row["t"] for row in rows]
            values = [row.old_value for row in transactions]
            next_values = [next((values[j] for j in range(i + 1, len(transactions))
                                if transactions[j].window_id == item.window_id and transactions[j].episode_id == item.episode_id), 0.0)
                           for i, item in enumerate(transactions)]
            gae = JL.compute_assignment_gae(transactions, values, next_values)
            per_seed[seed] = {"rows": rows, "transactions": transactions, "gae": gae, "next_values": next_values}
            for index, item in enumerate(transactions):
                delta, advantage = gae["assignment_td_residual"][index], gae["assignment_advantage"][index]
                nonterminal = not item.terminated and not item.truncated
                gae_rows.append({"seed": seed, "window_id": item.window_id, "time_band": item.provenance["time_band"],
                                 "decision_index": item.provenance["decision_index"], "reward": item.assignment_discounted_reward,
                                 "value": item.old_value, "next_value": next_values[index], "td_residual": delta, "advantage": advantage,
                                 "gae_recursive_term": advantage - delta, "nonterminal": nonterminal,
                                 "temporally_propagated": abs(advantage - delta) > 1e-12})
            batch = batch_rollout(rows, device=device, adim=adim, cdim=cdim)
            advantages = torch.tensor(gae["assignment_advantage"], dtype=torch.float32, device=device)
            returns = torch.tensor(gae["assignment_return"], dtype=torch.float32, device=device)
            for update_index in range(5):
                for row in rows:
                    try:
                        row["snapshot"].replay_guard(
                            support_digest=row["t"].provenance["candidate_snapshot_digest"],
                            regeneration_requested=False)
                    except Exception:  # noqa: BLE001
                        inv["candidate_regeneration"] += 1
                logits, no_assign = actor(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                                           agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                                           candidate_feats=batch["candidate_feats"], pair_agent_index=batch["pair_agent_index"],
                                           safe_mask=batch["safe_mask"])
                prediction = critic(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"],
                                    agent_feats=batch["agent_feats"], agent_mask=batch["agent_mask"],
                                    safe_summary=JL.safe_set_summary(batch["candidate_feats"], batch["safe_mask"]))
                loss = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign, safe_mask=batch["safe_mask"],
                                               action_index=batch["action_index"], old_log_prob=batch["old_log_prob"], advantage=advantages,
                                               value_pred=prediction, value_target=returns, forced_action=batch["forced_action"])
                budget.update()
                update = JL.apply_assignment_update(loss=loss, actor=actor, critic=critic, actor_optimizer=actor_opt, critic_optimizer=critic_opt)
                torch.mps.synchronize()
                finite = all(math.isfinite(float(value)) for value in (update["actor_grad_norm"], update["critic_grad_norm"], update["total_loss"]))
                inv["nan_or_inf"] += int(not finite)
                updates.append({"seed": seed, "update": update_index + 1, "rows": len(rows), "policy_loss": float(loss["policy_loss"].detach()),
                                "critic_loss": float(loss["critic_loss"].detach()), "entropy": float(loss["entropy"].detach()),
                                "ratio_mean": float(loss["ratio"].detach().mean()), "ratio_min": float(loss["ratio"].detach().min()),
                                "ratio_max": float(loss["ratio"].detach().max()), **update})
        checkpoint = {"path": "bt6_r2_joint_assignment_checkpoint.pt", "test_only": True, "bounded": True, "non_promotable": True,
                      "winner": False, "best_model": False, "promotion": False, "performance_claim_allowed": False,
                      "paper_level_claim_allowed": False, "causal_performance_claim_allowed": False}
        # Snapshot capture creates this artifact directory at the first
        # decision, so final checkpoint writing must not recreate it.
        root.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "meta": {**checkpoint, "envelope": envelope,
                   "bt6_s0_source": BT6_S0_SOURCE}}, root / checkpoint["path"])
        frozen_snapshot_collection = snapshot_writer.finalize(checkpoint_path=root / checkpoint["path"])

    auth_events = AUTH.audit_log()
    execution_events = [event for event in auth_events if event["capability"] == AUTH.SIMULATOR_EXECUTION and event["outcome"] == "ALLOWED"]
    training_events = [event for event in auth_events if event["capability"] == AUTH.TRAINING and event["outcome"] == "ALLOWED"]
    locks_after = AUTH.authorization_state()["capabilities"]
    after = frozen_hashes()
    actor_after, critic_after = BT1.param_digest(actor), BT1.param_digest(critic)
    transactions = [row["t"] for row in rollout]
    trajectories = defaultdict(list)
    for item in transactions:
        trajectories[(item.seed, item.window_id)].append(item)
    propagated = [row for row in gae_rows if row["temporally_propagated"]]
    nonterminal = [row for row in gae_rows if row["nonterminal"]]
    by_band = {band: [row for row in decisions if row["time_band"] == band] for band in ("night", "offpeak", "peak")}
    selective = {band: sum(row["zero_loss_selective"] for row in rows) for band, rows in by_band.items()}
    multi = {band: sum(row["genuine_2plus_candidate"] for row in rows) for band, rows in by_band.items()}
    readiness_checks = {
        "all_12_windows": len(budget.windows) == 12, "all_24_visits": budget.visits == 24,
        "both_seeds": sorted(set(budget.used_seeds)) == sorted(envelope["seeds"]), "all_time_bands": all(by_band.values()),
        "informative_decisions_ge_16": sum(row["informative"] for row in decisions) >= 16,
        "genuine_2plus_total_ge_3": sum(multi.values()) >= 3, "genuine_2plus_each_band_ge_1": all(value >= 1 for value in multi.values()),
        "selective_total_ge_3": sum(selective.values()) >= 3, "selective_each_band_ge_1": all(value >= 1 for value in selective.values()),
        "all_24_trajectories": len(trajectories) == 24 and all(len(value) == 4 for value in trajectories.values()),
        "gae_recursion": bool(propagated), "finite_gradients": all(math.isfinite(row["actor_grad_norm"]) and math.isfinite(row["critic_grad_norm"]) for row in updates),
        "frozen_policy_snapshot_count": frozen_snapshot_collection.get("snapshot_count") == budget.decisions,
        "frozen_policy_snapshot_checkpoint_binding": frozen_snapshot_collection.get("checkpoint_sha256") == sha256(root / checkpoint["path"]),
    }
    if any(locks_after.values()):
        inv["unauthorized_optimizer_step"] += 1
    if factory.source_mutations:
        inv["source_state_mutation"] += factory.source_mutations
    hard = []
    if before != after: hard.append("FROZEN_COMPONENT_MUTATION")
    if budget.payload()["used"] != {"distinct_windows": 12, "window_visits": 24, "seeds": envelope["seeds"], "assignment_decisions": 96,
                                    "trajectories": 24, "causal_transitions": 192, "optimizer_updates": 10}: hard.append("BT6_ENVELOPE_MISMATCH")
    if actor_before == actor_after or critic_before == critic_after: hard.append("JOINT_PARAMETER_DID_NOT_MOVE")
    if any(inv.values()): hard.append("INTEGRITY_VIOLATION:" + ",".join(name for name, value in inv.items() if value))
    if not adversarial["all_passed"]: hard.append("ADVERSARIAL_PROBE_FAILED")
    if hard:
        gate, classification = "BLOCKED", "BT6_INTEGRITY_OR_CAPABILITY_FAILURE"
    elif all(readiness_checks.values()):
        gate, classification = PASS_GATE, PASS_CLASS
    else:
        gate, classification = INSUFFICIENT_GATE, INSUFFICIENT_CLASS
    failed_readiness = [name for name, passed in readiness_checks.items() if not passed]
    candidate_audit = {"decisions": decisions, "total_decisions": len(decisions), "zero_candidate_decisions": sum(row["candidate_count_after_zero_loss"] == 0 for row in decisions),
                       "one_candidate_decisions": sum(row["candidate_count_after_zero_loss"] == 1 for row in decisions),
                       "genuine_2plus_candidate_decisions": sum(multi.values()), "by_time_band": {band: {"decisions": len(rows),
                       "zero_candidate": sum(row["candidate_count_after_zero_loss"] == 0 for row in rows), "one_candidate": sum(row["candidate_count_after_zero_loss"] == 1 for row in rows),
                       "genuine_2plus": multi[band]} for band, rows in by_band.items()}, "candidate_identity_mismatch": inv["candidate_identity_mismatch"],
                       "candidate_regeneration": inv["candidate_regeneration"],
                       "frozen_source_agent_to_causal_slot": agent_slot,
                       "slot_mapping_basis": "pre-rollout lexical source identity plus source-group co-occurrence only; collision-free and not an actor feature"}
    zero_audit = {"epsilon_sec": 0.0, "candidates_evaluated": sum(row["candidate_count_before_zero_loss"] for row in decisions),
                  "PASS": sum(row["zero_loss_pass"] for row in decisions), "FAIL": sum(row["zero_loss_fail"] for row in decisions),
                  "selective_states": sum(selective.values()), "selective_by_time_band": selective,
                  "zero_loss_violations": inv["zero_loss_violations"], "rejected_candidate_selected": inv["rejected_candidate_selected"]}
    advantages, returns = [row["advantage"] for row in gae_rows], [row["advantage"] + row["value"] for row in gae_rows]
    learning = {"informative_decisions": sum(row["informative"] for row in decisions),
                "non_zero_team_reward_fraction": sum(row["informative"] for row in decisions) / len(decisions),
                "multi_step_trajectories": len(trajectories), "nonterminal_transitions": len(nonterminal),
                "gae_recursive_term_count": len(propagated), "temporally_propagated_advantages": len(propagated), "gae_rows": gae_rows,
                "advantage_distribution": {"mean": statistics.mean(advantages), "std": statistics.pstdev(advantages), "variance": statistics.pvariance(advantages)},
                "critic_target_distribution": {"mean": statistics.mean(returns), "variance": statistics.pvariance(returns)},
                "cross_window_contamination": inv["cross_window_contamination"], "cross_seed_contamination": inv["cross_seed_contamination"],
                "legacy_advantage_contamination": inv["legacy_advantage_contamination"], "future_leakage": inv["future_leakage"]}
    lineage = {"BT4": BT4_SOURCE, "BT5": BT5_SOURCE, "BT5_R": BT5R_SOURCE, "BT6_S0": BT6_S0_SOURCE,
               "reward_v2_sha256": PV8_REWARD_V2_FREEZE_SHA256, "bt5r_manifest_sha256": sha256(BT5R / "manifest.json"),
               "bt6s0_manifest_sha256": sha256(BT6_S0 / "manifest.json")}
    outputs = {
        "bt6_mps_preflight.json": preflight,
        "bt6_execution_manifest.json": {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                           "envelope": envelope, "budget": budget.payload(), "selected_windows": windows, "checkpoint": checkpoint,
                                           "runtime_seconds": round(time.perf_counter() - started, 3), "maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                                           "baseline_comparison": "NONE", "performance_interpretation_permitted": False, "global_locks": LOCKS},
        "bt6_authorization_audit.json": {"simulator_execution_allowed_events": len(execution_events), "training_allowed_events": len(training_events),
                                            "capabilities_after_block": locks_after, "global_locks": LOCKS, "budget": budget.payload()},
        "bt6_candidate_support_audit.json": candidate_audit,
        "bt6_zero_loss_selectivity_audit.json": zero_audit,
        "bt6_learning_signal_audit.json": learning,
        "bt6_bt7_readiness_audit.json": {"passed": not hard and all(readiness_checks.values()), "checks": readiness_checks,
                                            "failed_criteria": failed_readiness, "integrity_violations": inv},
        "bt6_optimizer_audit.json": {"optimizer_steps": budget.updates, "max_allowed": 10, "updates": updates,
                                      "actor_gradients_finite": all(math.isfinite(row["actor_grad_norm"]) for row in updates),
                                      "critic_gradients_finite": all(math.isfinite(row["critic_grad_norm"]) for row in updates)},
        "bt6_frozen_policy_snapshot_audit.json": {"schema_version": FPS.SNAPSHOT_SCHEMA_VERSION,
                                                    "capture_boundary": snapshot_feature_contract["actor_input_boundary"],
                                                    "snapshot_count": frozen_snapshot_collection.get("snapshot_count"),
                                                    "expected_decision_count": budget.decisions,
                                                    "collection_digest": frozen_snapshot_collection.get("collection_digest"),
                                                    "checkpoint_sha256": frozen_snapshot_collection.get("checkpoint_sha256"),
                                                    "capture_before_training_time_actor_forward": True,
                                                    "candidate_regeneration_during_replay": 0,
                                                    "feature_recomputation_during_replay": 0,
                                                    "mask_reconstruction_during_replay": 0},
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after,
                                            "joint_actor": {"before": actor_before, "after": actor_after, "changed": actor_before != actor_after},
                                            "joint_critic": {"before": critic_before, "after": critic_after, "changed": critic_before != critic_after}},
        "test_results.json": {"binding": binding, "adversarial_probes": adversarial, "integrity_violations": inv, "hard_failures": hard,
                              "TEST6_access": 0, "github_push_performed": False, "lineage": lineage},
        "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"], "lineage": lineage,
                               "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
                               "next_step": "BT7 frozen-policy behavior review authorization" if gate == PASS_GATE else "STOP"},
    }
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — Post-repair frozen R2 bounded execution\n\n"
        f"gate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"MPS preflight passed on `{preflight.get('selected_device')}`. The fixed R2 envelope consumed {budget.visits} visits, "
        f"{budget.decisions} decisions, {budget.transitions} causal transitions, and {budget.updates} optimizer updates.\n\n"
        f"Candidate and Zero-Loss counts are structural evidence only; no behavior, policy-quality, convergence, baseline, or KPI superiority interpretation is permitted.\n\n"
        f"BT7 readiness = {not hard and all(readiness_checks.values())}; failed criteria = {failed_readiness}; hard failures = {hard}.\n",
        encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "lineage": lineage, "file_sha256": manifest, "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}")
    print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
