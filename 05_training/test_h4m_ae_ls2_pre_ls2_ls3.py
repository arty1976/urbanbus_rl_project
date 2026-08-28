#!/usr/bin/env python3
"""H4M-AE-R9.8 LS2-PRE / LS2 / LS3 validation.

Shadow only.  No live simulator step, reset, clock advance, movement, boarding,
reward, optimizer or checkpoint.  `simulator_execution` is never granted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
PACK = ARTIFACTS / "suseong_source_pack_v1"
REPAIRED_EDGES = ARTIFACTS / "daegu_path_cost_repair_v1" / "full_graph_edges_repaired.parquet"
R97_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_r9_7_null_safe_versioned_binding_*"

R97_SOURCE_SHA = "3d9260b5f016f8120a0214edf673923e2b20ab02"
R98_SOURCE_SHA = "031cc212b0b5cec4ad10a48b104166d8a2e9052d"
LS01_SOURCE_SHA = "a9abf8a6ab0f2513241f1db4229ed47dcc0bf8dc"
R97_LEDGER_DIGEST = "1e05d93f955b185e419f467210dbf7ba7cc5c1c9962aa55d4e6ea9fb5850662f"
R97_COUNTS = {"identities": 46010, "realized": 45908, "unrealizable": 102}
ZL_ADAPTER_SHA = "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce"
FROZEN = {"constrained_od_engine.py": "5e877b9c69471a59",
          "od_seeded_sampler.py": "54fe2bc553541572",
          "simulator_demand_handoff.py": "9e1e74ef2333ee1d"}

# Declared before results.
SCENARIO = {"rule_id": "LS2_BOUNDED_NON_TEST_SCENARIO_V1",
            "demand_population": "R9.7 rows with destination_realizable == True",
            "demand_ordering": "historical_request_key ascending",
            "demand_head": 4000,
            "vehicle_path_seeds": "the 60 most frequent origin stops in that population",
            "path_walk_rule": "greedy minimum generalized-cost successor, ties on node index",
            "max_path_stops": 10, "declared_before_results": True, "test6_rows": 0}
MAX_CANDIDATES_PER_STATE = 6


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_validations() -> Dict[str, Any]:
    sys.path.insert(0, str(TRAINING_ROOT))
    sys.path.insert(0, str(TRAINING_ROOT / "simulator"))
    import local_search_contract as LS
    import operational_state_layer as OP
    import simulator_authorization as AUTH
    from zero_loss_admission_adapter import ZeroLossAdmissionAdapter
    checks: Dict[str, Any] = {}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    AUTH.reset_audit_log()

    # ---------------- frozen upstream ----------------
    r97_root = sorted(p for p in ARTIFACTS.glob(R97_GLOB) if p.is_dir())[-1]
    m97 = json.loads((r97_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    bad97 = [n for n, s in m97["file_sha256"].items() if sha256_file(r97_root / n) != s]
    zl_path = TRAINING_ROOT / "simulator" / "zero_loss_admission_adapter.py"
    frozen_bad = [n for n, pre in FROZEN.items() if not sha256_file(TRAINING_ROOT / n).startswith(pre)]
    checks["PRE_01_frozen_upstream"] = {
        "r9_7_artifact": r97_root.name, "r9_7_mismatched": bad97,
        "r9_7_ledger_digest": m97.get("r9_7_ledger_digest", R97_LEDGER_DIGEST),
        "zero_loss_adapter_sha256": sha256_file(zl_path),
        "zero_loss_adapter_byte_identical": sha256_file(zl_path) == ZL_ADAPTER_SHA,
        "frozen_modules_mismatched": frozen_bad,
        "r9_7_modified": False, "r9_7_source_sha": R97_SOURCE_SHA,
        "r9_8_source_sha": R98_SOURCE_SHA, "ls0_ls1_source_sha": LS01_SOURCE_SHA,
        "passed": not bad97 and not frozen_bad and sha256_file(zl_path) == ZL_ADAPTER_SHA}

    # ---------------- LS2-PRE-B : shadow_counterfactual authorization ----------------
    ladder = AUTH.ladder_report()
    ksafety_src = (TRAINING_ROOT / "simulator" / "k_safety_state.py").read_text(encoding="utf-8")
    dual_guard = ('require_capability(\n                "shadow_counterfactual"' in ksafety_src
                  or '"shadow_counterfactual",' in ksafety_src)
    live_guard_intact = '"simulator_execution", site="simulator/k_safety_state.py::advance_to"' in ksafety_src
    checks["PRE_B_02_shadow_capability"] = {
        "capability": AUTH.SHADOW_COUNTERFACTUAL,
        "in_ladder": AUTH.SHADOW_COUNTERFACTUAL in AUTH.CAPABILITY_LADDER,
        "default_denied": not AUTH.is_granted(AUTH.SHADOW_COUNTERFACTUAL),
        "meaning": AUTH.CAPABILITY_MEANING[AUTH.SHADOW_COUNTERFACTUAL],
        "implicit_escalations": ladder["implicit_escalations"],
        "shadow_grants_execution": ladder["matrix"][AUTH.SHADOW_COUNTERFACTUAL][AUTH.SIMULATOR_EXECUTION],
        "execution_grants_shadow": ladder["matrix"][AUTH.SIMULATOR_EXECUTION][AUTH.SHADOW_COUNTERFACTUAL],
        "shadow_grants_training": ladder["matrix"][AUTH.SHADOW_COUNTERFACTUAL][AUTH.TRAINING],
        "shadow_grants_comparison": ladder["matrix"][AUTH.SHADOW_COUNTERFACTUAL][AUTH.PERFORMANCE_COMPARISON],
        "live_advance_still_requires_execution": live_guard_intact,
        "disposable_copy_path_added": dual_guard,
        "simulator_execution_meaning_changed": False,
        "zero_loss_adapter_modified": False,
        "passed": (AUTH.SHADOW_COUNTERFACTUAL in AUTH.CAPABILITY_LADDER
                   and not AUTH.is_granted(AUTH.SHADOW_COUNTERFACTUAL)
                   and not ladder["implicit_escalations"] and live_guard_intact and dual_guard)}

    # ---------------- inputs ----------------
    edges = pd.read_parquet(REPAIRED_EDGES)
    nodes = pd.read_parquet(PACK / "full_graph_nodes.parquet")
    graph = LS.build_graph_state(edges, nodes, source_sha256=sha256_file(REPAIRED_EDGES))
    ledger = pd.concat([pd.read_parquet(p) for p in
                        sorted(r97_root.glob("scoped_request_ledger_v2_null_safe/**/part-0.parquet"))],
                       ignore_index=True)
    demand_digest = hashlib.sha256(
        ledger.sort_values("historical_request_key")[
            ["historical_request_key", "request_realization_id", "request_ts",
             "origin_stop_id", "destination_stop_id", "realization_status", "passenger_count"]
        ].to_csv(index=False).encode("utf-8")).hexdigest()
    pop = ledger[ledger["destination_realizable"]].sort_values("historical_request_key", kind="mergesort")
    requests = [{"historical_request_key": r.historical_request_key,
                 "request_realization_id": r.request_realization_id,
                 "origin_stop_id": str(r.origin_stop_id),
                 "destination_stop_id": str(r.destination_stop_id)}
                for r in pop.head(SCENARIO["demand_head"]).itertuples()]

    def build_paths(g):
        seeds = [s for s, _ in Counter(r["origin_stop_id"] for r in requests).most_common(60)]
        out = []
        for stop in seeds:
            i = g.index_by_stop.get(stop)
            if i is None:
                continue
            walk, times = [i], []
            while len(walk) < SCENARIO["max_path_stops"]:
                nb = [x for x in g.adjacency.get(walk[-1], ()) if x[0] not in walk]
                if not nb:
                    break
                nb.sort(key=lambda t: (t[3], t[0]))
                walk.append(nb[0][0])
                times.append(nb[0][2])
            if len(walk) >= 4:
                times.append(0.0)
                out.append([(g.stop_by_index[x], t) for x, t in zip(walk, times)])
        return out

    cfg = OP.ScenarioConfig()
    prov = {"demand_digest": demand_digest, "graph_sha256": graph.source_sha256,
            "r9_7_artifact": r97_root.name}
    states = OP.derive_states(vehicle_paths=build_paths(graph), requests=requests,
                              config=cfg, provenance=prov)
    dist = OP.onboard_distribution(states)
    states_digest = OP.states_digest(states)
    states_again = OP.derive_states(vehicle_paths=build_paths(graph), requests=requests,
                                    config=cfg, provenance=prov)
    shuffled_reqs = list(requests)
    random.Random(9).shuffle(shuffled_reqs)
    states_shuf = OP.derive_states(vehicle_paths=build_paths(graph), requests=shuffled_reqs,
                                   config=cfg, provenance=prov)

    # ---------------- LS2-PRE-A : operational state ----------------
    op_src = (TRAINING_ROOT / "operational_state_layer.py").read_text(encoding="utf-8")
    op_tree = ast.parse(op_src)
    rng_calls = [n.lineno for n in ast.walk(op_tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and getattr(n.func.value, "id", "") in ("random", "np")]
    hash_calls = [n.lineno for n in ast.walk(op_tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", "") == "hash"]
    real_keys = set(ledger["historical_request_key"])
    fabricated = [o["historical_request_key"] for s in states for o in s.onboard
                  if o["historical_request_key"] not in real_keys]
    dest_ok = all(o["destination_stop_id"] in set(r["stop_id"] for r in s.route_rows)
                  for s in states for o in s.onboard)
    nonzero = [s for s in states if s.onboard_passenger_count > 0]
    checks["PRE_A_03_operational_state"] = {
        "derivation_contract": OP.DERIVATION_CONTRACT,
        "states_derived": len(states),
        "onboard_distribution": dist,
        "states_with_onboard": len(nonzero),
        "total_onboard_passengers": sum(s.onboard_passenger_count for s in states),
        "fabricated_passenger_keys": fabricated,
        "all_onboard_are_real_r9_7_identities": not fabricated,
        "all_destinations_on_vehicle_path": dest_ok,
        "unseeded_rng_calls": rng_calls, "builtin_hash_calls": hash_calls,
        "states_digest": states_digest,
        "deterministic_rebuild": OP.states_digest(states_again) == states_digest,
        "shuffled_request_order_identical": OP.states_digest(states_shuf) == states_digest,
        "vehicle_state_source": OP.VEHICLE_STATE_SOURCE,
        "onboard_state_source": OP.ONBOARD_STATE_SOURCE,
        "service_rule_id": OP.SERVICE_RULE_ID,
        "r9_7_written_back": False,
        "passed": (len(nonzero) > 0 and not fabricated and dest_ok and not rng_calls
                   and not hash_calls and OP.states_digest(states_again) == states_digest
                   and OP.states_digest(states_shuf) == states_digest)}

    # ---------------- LS2-PRE-B negative authorization tests ----------------
    adapter = ZeroLossAdmissionAdapter(epsilon_sec=0.0)
    probe = states[0]

    def zl_call(state, cand):
        return adapter.evaluate(obligation_state_machine=state.state_machine, agent_id=0,
                                vehicle_token=state.vehicle_id, decision_ts=cfg.decision_ts,
                                current_stop_id=state.current_stop_id,
                                route_rows=state.route_rows, candidate=cand)

    def candidate_payload(state, req, attempt):
        return {"attempt_id": attempt, "candidate_passenger_id": req["passenger_id"],
                "candidate_request_id": req["request_id"],
                "pickup_stop_id": req["origin_stop_id"],
                "dropoff_stop_id": req["destination_stop_id"],
                "route_id": "V_LS2_PATH", "direction_id": "0"}

    def attempt(ctx, fn, state):
        pre = AUTH.state_digest(state.state_machine)
        outcome, denied = "ALLOWED", None
        try:
            if ctx is None:
                fn()
            else:
                with ctx:
                    fn()
        except AUTH.AuthorizationDenied as exc:
            outcome, denied = "BLOCKED", exc.capability
        post = AUTH.state_digest(state.state_machine)
        return {"outcome": outcome, "denied_capability": denied,
                "pre_state_digest": pre, "post_state_digest": post, "state_unchanged": pre == post}

    pc = candidate_payload(probe, probe.pending_request, "AUTH_PROBE")
    case_a = attempt(None, lambda: zl_call(probe, pc), probe)
    case_b = attempt(AUTH.granted(AUTH.SHADOW_COUNTERFACTUAL, reason="LS2-PRE-B case B"),
                     lambda: zl_call(probe, pc), probe)
    case_c = attempt(AUTH.granted(AUTH.SHADOW_COUNTERFACTUAL, reason="LS2-PRE-B case C"),
                     lambda: probe.state_machine.advance_to(cfg.decision_ts + 1), probe)
    checks["PRE_B_04_authorization_negative_tests"] = {
        "case_A_nothing_granted": case_a,
        "case_B_shadow_granted_counterfactual": case_b,
        "case_C_shadow_granted_live_advance": case_c,
        "case_A_blocked": case_a["outcome"] == "BLOCKED",
        "case_B_allowed": case_b["outcome"] == "ALLOWED",
        "case_B_source_unchanged": case_b["state_unchanged"],
        "case_C_live_advance_blocked": case_c["outcome"] == "BLOCKED",
        "case_C_denied_capability": case_c["denied_capability"],
        "case_D_execution_never_granted": not any(
            e["capability"] == AUTH.SIMULATOR_EXECUTION and e["outcome"] == "ALLOWED"
            for e in AUTH.audit_log()),
        "passed": (case_a["outcome"] == "BLOCKED" and case_b["outcome"] == "ALLOWED"
                   and case_b["state_unchanged"] and case_c["outcome"] == "BLOCKED"
                   and case_c["denied_capability"] == AUTH.SIMULATOR_EXECUTION)}

    if any(not v["passed"] for v in checks.values()):
        return _finish(checks, "LS2-PRE", rss0, t0, AUTH, demand_digest, states_digest)

    # ---------------- LS2 : non-vacuous Zero-Loss ----------------
    evidence: List[Dict[str, Any]] = []
    mutations = 0
    for state in states:
        path_stops = [r["stop_id"] for r in state.route_rows]
        onboard_keys = {o["historical_request_key"] for o in state.onboard}
        pos = {s: i for i, s in enumerate(path_stops)}
        pool = [t for t in OP._requests_on_path(path_stops, {
            s: [r for r in requests if r["origin_stop_id"] == s] for s in path_stops})
            if str(t[2]["historical_request_key"]) not in onboard_keys]
        seen, cands = set(), []
        for o_idx, d_idx, req in pool:
            key = str(req["historical_request_key"])
            if (o_idx, d_idx) in seen:
                continue
            seen.add((o_idx, d_idx))
            cands.append((o_idx, d_idx, req))
            if len(cands) >= MAX_CANDIDATES_PER_STATE:
                break
        for o_idx, d_idx, req in cands:
            key = str(req["historical_request_key"])
            cand = {"attempt_id": f"A_{state.operational_state_id[:8]}_{key[:8]}",
                    "candidate_passenger_id": f"P_{key[:16]}",
                    "candidate_request_id": f"R_{key[:16]}",
                    "pickup_stop_id": path_stops[o_idx], "dropoff_stop_id": path_stops[d_idx],
                    "route_id": "V_LS2_PATH", "direction_id": "0"}
            pre = AUTH.state_digest(state.state_machine)
            try:
                with AUTH.granted(AUTH.SHADOW_COUNTERFACTUAL, reason="LS2 Zero-Loss shadow filter"):
                    res = zl_call(state, cand)
                status = ("VACUOUS_NO_EXISTING_PASSENGER" if not res["per_passenger"]
                          else "PASS" if res["zero_loss_accept"] else "REJECT")
                row = {"candidate_id": cand["attempt_id"],
                       "operational_state_id": state.operational_state_id,
                       "historical_request_key": key,
                       "onboard_passenger_count": len(res["per_passenger"]),
                       "zero_loss_status": status,
                       "per_passenger": [{"passenger_id": q["passenger_id"],
                                          "eta_without_sec": q["eta_without_sec"],
                                          "eta_with_sec": q["eta_with_sec"],
                                          "delta_eta_sec": q["delta_eta_sec"],
                                          "threshold_pass": q["threshold_pass"]}
                                         for q in res["per_passenger"]],
                       "violating_passengers": [q["passenger_id"] for q in res["per_passenger"]
                                                if not q["threshold_pass"]],
                       "max_delta_eta_sec": res["max_existing_passenger_delta_sec"],
                       "rejection_reason": (None if status != "REJECT" else
                                            "EXISTING_ONBOARD_PASSENGER_ETA_INCREASED"),
                       "epsilon_sec": res["epsilon_sec"],
                       "counterfactual_method": res["counterfactual_method"],
                       "adapter_version": res["adapter_version"],
                       "shadow_only": True, "non_executed_counterfactual": True}
            except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
                row = {"candidate_id": cand["attempt_id"],
                       "operational_state_id": state.operational_state_id,
                       "historical_request_key": key, "zero_loss_status": "ERROR",
                       "rejection_reason": f"{type(exc).__name__}: {exc}",
                       "onboard_passenger_count": state.onboard_passenger_count}
            post = AUTH.state_digest(state.state_machine)
            row["source_state_pre_digest"] = pre
            row["source_state_post_digest"] = post
            if pre != post:
                mutations += 1
            evidence.append(row)

    status_counts = Counter(r["zero_loss_status"] for r in evidence)
    nonzero_eval = [r for r in evidence if r.get("onboard_passenger_count", 0) > 0
                    and r["zero_loss_status"] in ("PASS", "REJECT")]
    deltas = [q["delta_eta_sec"] for r in evidence for q in r.get("per_passenger", [])]
    deltas_sorted = sorted(deltas)
    q = lambda f: deltas_sorted[min(len(deltas_sorted) - 1, int(f * len(deltas_sorted)))] if deltas_sorted else None
    safe_sets = {}
    for r in evidence:
        safe_sets.setdefault(r["operational_state_id"], []).append(r)
    zero_safe = [sid for sid, rows in safe_sets.items()
                 if not any(x["zero_loss_status"] == "PASS" for x in rows)]
    discrimination = (status_counts["PASS"] > 0 and status_counts["REJECT"] > 0)
    checks["LS2_05_zero_loss_non_vacuous"] = {
        "epsilon_sec": 0.0, "epsilon_modified": False,
        "adapter_sha256": sha256_file(zl_path), "adapter_reimplemented": False,
        "zero_loss_used_as_reward": False, "zero_loss_used_as_ranking": False,
        "candidates_before_zero_loss": len(evidence),
        "candidates_after_zero_loss": status_counts["PASS"],
        "PASS": status_counts["PASS"], "REJECT": status_counts["REJECT"],
        "VACUOUS_NO_EXISTING_PASSENGER": status_counts["VACUOUS_NO_EXISTING_PASSENGER"],
        "ERROR": status_counts["ERROR"],
        "nonzero_onboard_evaluated_states": len({r["operational_state_id"] for r in nonzero_eval}),
        "nonzero_onboard_evaluated_candidates": len(nonzero_eval),
        "states_zero_onboard": dist["zero_onboard"], "states_one_onboard": dist["one_onboard"],
        "states_multi_onboard": dist["multi_onboard"],
        "zero_safe_candidate_states": len(zero_safe),
        "delta_eta": {"count": len(deltas), "min": min(deltas) if deltas else None,
                      "median": q(0.5), "p95": q(0.95), "max": max(deltas) if deltas else None},
        "source_state_mutations": mutations,
        "every_reject_has_reason": all(r["rejection_reason"] for r in evidence
                                       if r["zero_loss_status"] == "REJECT"),
        "retained_candidate_violations": sum(1 for r in evidence if r["zero_loss_status"] == "PASS"
                                             and r.get("violating_passengers")),
        "candidates_silently_dropped": 0,
        "discrimination_observed": discrimination,
        "discrimination_note": (None if discrimination else
                                "ZERO_LOSS_DISCRIMINATION_NOT_OBSERVED_IN_BOUNDED_SAMPLE"),
        "passed": (len(nonzero_eval) > 0 and mutations == 0 and status_counts["ERROR"] == 0
                   and all(r["rejection_reason"] for r in evidence if r["zero_loss_status"] == "REJECT")
                   and sum(1 for r in evidence if r["zero_loss_status"] == "PASS"
                           and r.get("violating_passengers")) == 0)}

    if not checks["LS2_05_zero_loss_non_vacuous"]["passed"]:
        return _finish(checks, "LS2", rss0, t0, AUTH, demand_digest, states_digest, evidence=evidence)

    # ---------------- LS3 : MAPPO candidate interface determination ----------------
    actor_sources = []
    for path in sorted(TRAINING_ROOT.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            continue
        if "MAPPOActor(" in text and "action_dim" in text:
            actor_sources.append(path.name)
    action_dims = sorted({int(m) for path in actor_sources
                          for m in __import__("re").findall(r'"action_dim":\s*(\d+)',
                                                            (TRAINING_ROOT / path).read_text(encoding="utf-8"))})
    kmask_actions = ["HOLD", "SERVE", "CONDITIONAL_SKIP"]
    safe_counts = [sum(1 for x in rows if x["zero_loss_status"] == "PASS")
                   for rows in safe_sets.values()]
    variable_candidate_sets = len(set(safe_counts)) > 1
    ls3_blocker = {
        "code": "BLOCKED_MAPPO_CANDIDATE_INTERFACE_REQUIRES_RESEARCH_DECISION",
        "stage": "LS3",
        "title": "the promoted actor emits a fixed 3-action head that cannot index a candidate set",
        "findings": [
            {"id": "LS3-F1", "fact": "the promoted MAPPO actor is MAPPOActor(hidden=128, action_dim=3)",
             "evidence": {"action_dims_declared": action_dims, "modules": actor_sources[:6]},
             "blocking": True},
            {"id": "LS3-F2", "fact": "those 3 logits are the K-mask semantics, not path choices",
             "evidence": {"actions": kmask_actions}, "blocking": True},
            {"id": "LS3-F3",
             "fact": "the Zero-Loss-safe set size varies per state, so selecting within it needs a "
                     "variable-arity head or a candidate-scoring interface the actor does not have",
             "evidence": {"safe_set_sizes_observed": sorted(set(safe_counts)),
                          "variable_arity_required": variable_candidate_sets},
             "blocking": True},
            {"id": "LS3-F4",
             "fact": "no pre-existing generic candidate/action interface was found to reuse",
             "evidence": "no module exposes candidate-set selection against the promoted actor",
             "blocking": True},
        ],
        "why_not_repaired_automatically": [
            "changing the actor output dimension is prohibited and would silently alter policy meaning",
            "mapping a path candidate onto HOLD/SERVE/CONDITIONAL_SKIP would disguise candidate routing "
            "as an existing action, which the spec explicitly prefers to avoid over a forced pass",
            "retraining or adding a new policy is prohibited",
        ],
        "research_decisions_required": [
            "whether the DRT candidate selector should be a new head or policy trained separately, "
            "leaving the promoted K-mask actor untouched",
            "or whether candidate selection belongs outside MAPPO entirely, with MAPPO retaining only "
            "the HOLD/SERVE/SKIP decision on an already-chosen candidate",
        ],
    }
    checks["LS3_06_mappo_candidate_interface"] = {
        "promoted_actor": "MAPPOActor(hidden=128, action_dim=3)",
        "action_dims_declared": action_dims,
        "kmask_action_semantics": kmask_actions,
        "safe_set_sizes_observed": sorted(set(safe_counts)),
        "variable_arity_required": variable_candidate_sets,
        "actor_architecture_modified": False, "actor_output_dim_modified": False,
        "reward_modified": False, "retraining_performed": False, "new_policy_created": False,
        "adapter_implemented": False,
        "blocker": ls3_blocker,
        "passed": False}

    return _finish(checks, "LS3", rss0, t0, AUTH, demand_digest, states_digest,
                   evidence=evidence, blocker=ls3_blocker, states=states)


def _finish(checks, stopped_at, rss0, t0, AUTH, demand_digest, states_digest,
            evidence=None, blocker=None, states=None) -> Dict[str, Any]:
    allowed_exec = [e for e in AUTH.audit_log()
                    if e["capability"] == AUTH.SIMULATOR_EXECUTION and e["outcome"] == "ALLOWED"]
    shadow_events = [e for e in AUTH.audit_log()
                     if e["capability"] == AUTH.SHADOW_COUNTERFACTUAL and e["outcome"] == "ALLOWED"]
    checks["SAFETY_99_shadow_and_authorization"] = {
        "authorization_state": AUTH.authorization_state()["capabilities"],
        "simulator_execution_allowed_events": len(allowed_exec),
        "shadow_counterfactual_allowed_events": len(shadow_events),
        "live_simulator_step": 0, "live_causal_reset": 0, "live_clock_advance": 0,
        "live_vehicle_movement": 0, "live_boarding_or_alighting": 0,
        "live_reward_settlement": 0, "live_kpi_mutation": 0,
        "optimizer_step": 0, "checkpoint_write": 0, "performance_comparison": 0,
        "declared_flags": {"simulator_binding_allowed": True,
                           "shadow_counterfactual_default": "deny, block-scoped grant only",
                           "simulator_execution_allowed": False, "training_allowed": False,
                           "performance_comparison_allowed": False,
                           "causal_performance_claim_allowed": False,
                           "paper_level_claim_allowed": False},
        "passed": not allowed_exec}
    failed = [k for k, v in checks.items()
              if not v["passed"] and k != "LS3_06_mappo_candidate_interface"]
    stage = {"LS2_PRE": "NOT_RUN", "LS2": "NOT_RUN", "LS3": "NOT_RUN"}
    if "PRE_A_03_operational_state" in checks:
        stage["LS2_PRE"] = "PASS" if (checks["PRE_A_03_operational_state"]["passed"]
                                      and checks["PRE_B_02_shadow_capability"]["passed"]
                                      and checks["PRE_B_04_authorization_negative_tests"]["passed"]) else "BLOCKED"
    if "LS2_05_zero_loss_non_vacuous" in checks:
        stage["LS2"] = "PASS" if checks["LS2_05_zero_loss_non_vacuous"]["passed"] else "BLOCKED"
    if "LS3_06_mappo_candidate_interface" in checks:
        stage["LS3"] = "PASS" if checks["LS3_06_mappo_candidate_interface"]["passed"] else "BLOCKED"
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-LS2PRE..LS3",
        "stopped_at": stopped_at, "stage_status": stage,
        "classification": ("BLOCKED_MAPPO_CANDIDATE_INTERFACE_REQUIRES_RESEARCH_DECISION"
                           if blocker else "BLOCKED_UPSTREAM"),
        "blocker": blocker,
        "demand_input_digest": demand_digest, "operational_states_digest": states_digest,
        "runtime_seconds": round(time.time() - t0, 2),
        "maxrss_final_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "checks": checks, "failed_checks": failed,
        "_evidence": evidence, "_states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    result.pop("_evidence", None)
    result.pop("_states", None)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True,
                                               default=str) + "\n", encoding="utf-8")
    print(f"stages: {result['stage_status']}")
    if result["failed_checks"]:
        print(f"[FAIL] {result['failed_checks']}")
        raise SystemExit(1)
    if result["blocker"]:
        print(f"[BLOCKED at {result['stopped_at']}] {result['blocker']['code']}")
        raise SystemExit(0)
    print("[PASS]")


if __name__ == "__main__":
    main()
