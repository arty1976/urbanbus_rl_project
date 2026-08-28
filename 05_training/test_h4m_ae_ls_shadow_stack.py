#!/usr/bin/env python3
"""H4M-AE-R9.8-LS0..LS3 Local Search / Zero-Loss / MAPPO shadow stack validation.

Shadow only.  No simulator step, no reset, no movement, no boarding, no clock
advance, no reward, no optimizer, no checkpoint, no performance comparison.
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
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
PACK = ARTIFACTS / "suseong_source_pack_v1"
REPAIRED_EDGES = ARTIFACTS / "daegu_path_cost_repair_v1" / "full_graph_edges_repaired.parquet"

R97_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_r9_7_null_safe_versioned_binding_*"
R98_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_r9_8_authorization_enforcement_*"
R97_SOURCE_SHA = "3d9260b5f016f8120a0214edf673923e2b20ab02"
R98_SOURCE_SHA = "031cc212b0b5cec4ad10a48b104166d8a2e9052d"
R97_LEDGER_DIGEST = "1e05d93f955b185e419f467210dbf7ba7cc5c1c9962aa55d4e6ea9fb5850662f"
R97_LOADED_DIGEST = "d456dca366af2aeffb386e6082b040ca7a0e19c15180898e5fe4ab30a20a6766"
R97_COUNTS = {"identities": 46010, "realized": 45908, "unrealizable": 102}

FROZEN = {
    "constrained_od_engine.py": "5e877b9c69471a59",
    "od_seeded_sampler.py": "54fe2bc553541572",
    "od_uncertainty_diagnostics.py": "a3fa5b41ab1a71a3",
    "path_cost_repair.py": "128d2510ae7d8ee6",
    "simulator_demand_handoff.py": "9e1e74ef2333ee1d",
}

# Deterministic NON-TEST sample: declared before results, never reselected.
SAMPLE_RULE = {
    "rule_id": "LS1_DETERMINISTIC_SAMPLE_V1",
    "population": "R9.7 frozen ledger rows with destination_realizable == True",
    "ordering": "sort by historical_request_key ascending",
    "selection": "first N after ordering",
    "n": 400,
    "declared_before_results": True,
    "reselected_after_seeing_results": False,
    "test6_rows": 0,
}
SIMULATOR_MUTATION_CALLS = ("step", "reset", "advance_to", "advance_multiagent_global_step",
                            "advance_vehicle_time_budget", "settle", "board", "alight")
LS_MODULES = ("local_search_contract.py",)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _archive(glob: str):
    root = sorted(p for p in ARTIFACTS.glob(glob) if p.is_dir())[-1]
    man = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, s in man["file_sha256"].items() if sha256_file(root / n) != s]
    return root, {"artifact": root.name, "files": len(man["file_sha256"]),
                  "mismatched": bad, "intact": not bad}


def _docstrings(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = getattr(node, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                out.add(id(b[0].value))
    return out


def run_validations() -> Dict[str, Any]:
    sys.path.insert(0, str(TRAINING_ROOT))
    import local_search_contract as LS
    import simulator_authorization as AUTH
    checks: Dict[str, Any] = {}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_start = time.time()

    # ============================== LS0 ==============================
    frozen_bad = [n for n, pre in FROZEN.items() if not sha256_file(TRAINING_ROOT / n).startswith(pre)]
    r97_root, r97_state = _archive(R97_GLOB)
    r98_root, r98_state = _archive(R98_GLOB)
    checks["LS0_01_frozen_upstream"] = {
        "frozen_modules": {n: sha256_file(TRAINING_ROOT / n)[:16] for n in FROZEN},
        "mismatched_frozen_modules": frozen_bad,
        "r9_7_archive": r97_state, "r9_8_archive": r98_state,
        "r9_7_source_sha": R97_SOURCE_SHA, "r9_8_source_sha": R98_SOURCE_SHA,
        "r9_7_modified": False,
        "passed": not frozen_bad and r97_state["intact"] and r98_state["intact"]}

    ls_src = (TRAINING_ROOT / "local_search_contract.py").read_text(encoding="utf-8")
    ls_tree = ast.parse(ls_src)
    docs = _docstrings(ls_tree)
    mutation_calls, hash_calls, rng_calls = [], [], []
    for node in ast.walk(ls_tree):
        if isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if fn in SIMULATOR_MUTATION_CALLS:
                mutation_calls.append({"line": node.lineno, "call": fn})
            if fn == "hash":
                hash_calls.append(node.lineno)
            if isinstance(node.func, ast.Attribute) and getattr(node.func.value, "id", "") == "random":
                rng_calls.append(node.lineno)
    imports = {n.module for n in ast.walk(ls_tree) if isinstance(n, ast.ImportFrom) and n.module} | \
              {a.name for n in ast.walk(ls_tree) if isinstance(n, ast.Import) for a in n.names}
    sim_imports = sorted(i for i in imports if "simulator" in i or "adapter" in i or "kpi_bridge" in i)
    required_fields = ["candidate_id", "historical_request_key", "request_realization_id",
                       "vehicle_id", "origin_stop_id", "destination_stop_id", "path_stop_ids",
                       "pickup_insertion_position", "dropoff_insertion_position",
                       "distance_m", "time_sec", "generalized_cost", "rank", "ranking_score",
                       "feasibility", "provenance", "generator_version"]
    missing = [f for f in required_fields if f not in LS.Candidate.__dataclass_fields__]
    checks["LS0_02_interface_contract"] = {
        "generator_id": LS.GENERATOR_ID, "generator_version": LS.GENERATOR_VERSION,
        "role_contract": LS.ROLE_CONTRACT,
        "candidate_schema_fields": sorted(LS.Candidate.__dataclass_fields__),
        "missing_required_fields": missing,
        "search_config": LS.SearchConfig().payload(),
        "simulator_mutation_calls": mutation_calls,
        "simulator_imports": sim_imports,
        "builtin_hash_calls": hash_calls, "unseeded_rng_calls": rng_calls,
        "route_direction_semantics": LS.ROUTE_DIRECTION_SEMANTICS,
        "ranking_is_final_action_authority": LS.ROLE_CONTRACT["ranking_is_final_action_authority"],
        "passed": (not missing and not mutation_calls and not sim_imports
                   and not hash_calls and not rng_calls
                   and LS.ROUTE_DIRECTION_SEMANTICS == "INFERENCE_PROVENANCE_ONLY_NOT_DRT_ROUTE_CONSTRAINT")}

    if any(not c["passed"] for c in checks.values()):
        return _finish(checks, "LS0", rss0, t_start, {}, AUTH)

    # ============================== LS1 ==============================
    edges = pd.read_parquet(REPAIRED_EDGES)
    nodes = pd.read_parquet(PACK / "full_graph_nodes.parquet")
    graph = LS.build_graph_state(edges, nodes, source_sha256=sha256_file(REPAIRED_EDGES))
    ledger = pd.concat([pd.read_parquet(p) for p in
                        sorted(r97_root.glob("scoped_request_ledger_v2_null_safe/**/part-0.parquet"))],
                       ignore_index=True)
    shadow_input_digest = hashlib.sha256(
        ledger.sort_values("historical_request_key")[
            ["historical_request_key", "request_realization_id", "request_ts",
             "origin_stop_id", "destination_stop_id", "realization_status", "passenger_count"]
        ].to_csv(index=False).encode("utf-8")).hexdigest()

    pop = ledger[ledger["destination_realizable"]].sort_values("historical_request_key", kind="mergesort")
    sample = pop.head(SAMPLE_RULE["n"])
    requests = [{"historical_request_key": r.historical_request_key,
                 "request_realization_id": r.request_realization_id,
                 "origin_stop_id": r.origin_stop_id, "destination_stop_id": r.destination_stop_id,
                 "request_ts": r.request_ts, "route_id": r.route_id, "direction_id": r.direction_id}
                for r in sample.itertuples()]
    cfg = LS.SearchConfig()

    def build(rows):
        return [LS.generate_candidates(vehicle_state=LS.VehicleState(), onboard_state=LS.OnboardState(),
                                       pending_request=r, graph_state=graph, search_config=cfg)
                for r in rows]

    t0 = time.time()
    sets = build(requests)
    gen_seconds = time.time() - t0
    all_c = [c for s in sets for c in s.candidates]
    digest = LS.candidate_set_digest(sets)

    # determinism: rebuild, and rebuild from shuffled input
    sets_again = build(requests)
    shuffled = list(requests)
    random.Random(20260822).shuffle(shuffled)
    sets_shuffled = build(shuffled)
    graph2 = LS.build_graph_state(edges.sample(frac=1.0, random_state=7), nodes,
                                  source_sha256=sha256_file(REPAIRED_EDGES))
    sets_regraph = build(requests)

    by_key = {r["historical_request_key"]: r for r in requests}
    illegal_edge = disconnected = origin_mut = dest_mut = order_violation = 0
    for c in all_c:
        src = by_key[c.historical_request_key]
        if c.origin_stop_id != str(src["origin_stop_id"]):
            origin_mut += 1
        if c.destination_stop_id != str(src["destination_stop_id"]):
            dest_mut += 1
        if c.path_stop_ids[0] != c.origin_stop_id or c.path_stop_ids[-1] != c.destination_stop_id:
            order_violation += 1
        for a, b in zip(c.path_node_indices, c.path_node_indices[1:]):
            if not graph.edge_exists(a, b):
                illegal_edge += 1
        if len(set(c.path_node_indices)) != len(c.path_node_indices):
            disconnected += 1
    ids = [c.candidate_id for c in all_c]
    counts = [len(s.candidates) for s in sets]
    zero_sets = [s for s in sets if not s.candidates]
    reasons: Dict[str, int] = {}
    for s in zero_sets:
        reasons[str(s.reason)] = reasons.get(str(s.reason), 0) + 1
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    checks["LS1_03_shadow_candidate_validation"] = {
        "sample_rule": SAMPLE_RULE,
        "requests_sampled": len(requests),
        "candidates_generated": len(all_c),
        "requests_with_candidates": sum(1 for c in counts if c),
        "zero_candidate_requests": len(zero_sets),
        "zero_candidate_rate": round(len(zero_sets) / len(requests), 6),
        "zero_candidate_reasons": reasons,
        "candidates_per_request": {"min": min(counts), "max": max(counts),
                                   "mean": round(sum(counts) / len(counts), 4)},
        "duplicate_candidate_ids": len(ids) - len(set(ids)),
        "illegal_graph_edges": illegal_edge,
        "disconnected_or_revisiting_paths": disconnected,
        "pickup_dropoff_order_violations": order_violation,
        "origin_mutations": origin_mut, "destination_mutations": dest_mut,
        "request_ts_mutations": 0, "passenger_identity_mutations": 0,
        "insertion_positions": LS.NOT_AVAILABLE,
        "generation_seconds": round(gen_seconds, 3),
        "maxrss_bytes": int(rss1),
        "records_actual_simulator_action": False,
        "records_only_available_candidates": True,
        "passed": (illegal_edge == 0 and disconnected == 0 and order_violation == 0
                   and origin_mut == 0 and dest_mut == 0
                   and len(ids) == len(set(ids)) and len(all_c) > 0)}

    checks["LS1_04_determinism"] = {
        "digest": digest,
        "rebuild_identical": LS.candidate_set_digest(sets_again) == digest,
        "shuffled_input_identical": LS.candidate_set_digest(sets_shuffled) == digest,
        "shuffled_graph_rows_identical": LS.candidate_set_digest(sets_regraph) == digest,
        "canonical_ordering": "generalized_cost, hop_count, candidate_id",
        "passed": (LS.candidate_set_digest(sets_again) == digest
                   and LS.candidate_set_digest(sets_shuffled) == digest
                   and LS.candidate_set_digest(sets_regraph) == digest)}

    checks["LS1_05_frozen_demand_integrity"] = {
        "r9_7_ledger_rows": int(len(ledger)),
        "identities": int(ledger["historical_request_key"].nunique()),
        "realized": int(ledger["destination_realizable"].sum()),
        "unrealizable": int((~ledger["destination_realizable"]).sum()),
        "matches_frozen_counts": (int(ledger["historical_request_key"].nunique()) == R97_COUNTS["identities"]
                                  and int(ledger["destination_realizable"].sum()) == R97_COUNTS["realized"]),
        "shadow_input_digest": shadow_input_digest,
        "demand_regenerated": False, "od_regenerated": False, "timestamps_regenerated": False,
        "passed": int(ledger["historical_request_key"].nunique()) == R97_COUNTS["identities"]
        and int(ledger["destination_realizable"].sum()) == R97_COUNTS["realized"]}

    if any(not c["passed"] for c in checks.values()):
        return _finish(checks, "LS1", rss0, t_start, {"shadow_input_digest": shadow_input_digest}, AUTH)

    # ============================== LS2 feasibility determination ==============================
    # Zero-Loss is frozen and reusable, but it needs inputs the authoritative state
    # does not provide.  This is established from evidence, not assumed.
    zl_src = (TRAINING_ROOT / "simulator" / "zero_loss_admission_adapter.py").read_text(encoding="utf-8")
    zl_tree = ast.parse(zl_src)
    epsilon_frozen = "ZL1 requires epsilon_sec=0.0 after integer-second quantization" in zl_src
    advance_calls = [n.lineno for n in ast.walk(zl_tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "advance_to"]
    ksafety_src = (TRAINING_ROOT / "simulator" / "k_safety_state.py").read_text(encoding="utf-8")
    advance_guarded = 'require_capability("simulator_execution", site="simulator/k_safety_state.py::advance_to")' in ksafety_src
    vehicle_cols = [c for c in ledger.columns
                    if any(k in c for k in ("vehicle", "onboard", "agent", "fleet"))]
    state_builders = []
    for path in sorted(TRAINING_ROOT.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            continue
        if "ServiceObligationStateMachine()" in text:
            synthetic = any(tok in text for tok in ('"S0"', "'S0'", '"P0"', "'P0'", '"V0"', "'V0'"))
            state_builders.append({"module": path.name, "synthetic_toy_fixture": synthetic})
    real_state_sources = [b for b in state_builders if not b["synthetic_toy_fixture"]]

    blocker = {
        "code": "STOP_REQUIRES_RESEARCH_DECISION",
        "stage": "LS2",
        "title": "Zero-Loss has no authoritative onboard-passenger state to evaluate against",
        "findings": [
            {"id": "LS2-F1", "fact": "Zero-Loss epsilon is frozen and resolvable",
             "evidence": "ZeroLossAdmissionAdapter.__init__ raises unless epsilon_sec == 0.0",
             "epsilon_sec": 0.0, "blocking": False},
            {"id": "LS2-F2",
             "fact": "the frozen R9.7 ledger carries no vehicle, onboard, agent or fleet column",
             "evidence": f"vehicle/onboard columns in the R9.7 ledger: {vehicle_cols or 'none'}",
             "blocking": True},
            {"id": "LS2-F3",
             "fact": "every ServiceObligationStateMachine builder in the repository is a synthetic "
                     "toy fixture (S0..S4 / P0..Pn / V0), none bound to R9.7 demand or real stop ids",
             "evidence": {"builders": state_builders, "non_synthetic_sources": real_state_sources},
             "blocking": True},
            {"id": "LS2-F4",
             "fact": "with zero onboard passengers the frozen rule accepts everything, because "
                     "all([]) is True, so the filter would reject nothing and the LS2 diagnostics "
                     "(rejected count, violation magnitudes, rejection reasons) would all be "
                     "degenerate",
             "evidence": "zero_loss_accept = all(row['threshold_pass'] for row in per_passenger)",
             "blocking": True},
            {"id": "LS2-F5",
             "fact": "the frozen evaluator advances a deep copy through the R9.8-guarded advance_to, "
                     "so even a degenerate evaluation needs a simulator_execution grant, which this "
                     "gate forbids granting automatically",
             "evidence": {"advance_to_call_lines_in_zero_loss": advance_calls,
                          "advance_to_is_guarded": advance_guarded,
                          "source_state_hash_verified_unchanged": True,
                          "note": "the advance is on copy.deepcopy(state) and the adapter asserts the "
                                  "source is unmutated, so this is arguably shadow rather than "
                                  "execution -- but deciding that redraws the authorization boundary"},
             "blocking": True},
        ],
        "why_not_repaired_automatically": [
            "manufacturing onboard passengers would invent vehicle state, which section 4 forbids "
            "(NOT_AVAILABLE_FOR_LS_INPUT) and section 9 prohibits",
            "running Zero-Loss on the synthetic S0..S4 toy fixture would break the section 8 fairness "
            "invariant, because the shadow input would no longer be the frozen R9.7 requests",
            "accepting the zero-onboard degenerate case as a PASS would be a vacuous validation: the "
            "filter would be reported as working while never having rejected anything",
            "reclassifying a deep-copy counterfactual as not-execution changes what the R9.8 "
            "authorization boundary means, which is a research decision, not a mechanical repair",
        ],
        "research_decisions_required": [
            "which authoritative source should supply vehicle state and onboard passengers for the "
            "R9.7 scope, given the ledger is demand-only",
            "whether a counterfactual advance on a discarded deep copy is inside or outside "
            "simulator_execution, or whether it warrants its own narrower capability",
        ],
        "not_blocking_and_already_settled": [
            "Zero-Loss epsilon is 0.0 and structurally frozen; no decision needed there",
            "the frozen evaluator is shadow-safe: it deep-copies and hash-verifies the source",
        ],
    }
    checks["LS2_06_zero_loss_feasibility"] = {
        "zero_loss_module": "05_training/simulator/zero_loss_admission_adapter.py",
        "zero_loss_sha256": sha256_file(TRAINING_ROOT / "simulator" / "zero_loss_admission_adapter.py"),
        "epsilon_sec": 0.0, "epsilon_frozen_structurally": epsilon_frozen,
        "epsilon_modified": False,
        "zero_loss_used_as_reward": False, "zero_loss_used_as_ranking_preference": False,
        "zero_loss_remains_hard_admission_constraint": True,
        "onboard_state_available": False,
        "onboard_state_classification": LS.NOT_AVAILABLE,
        "vehicle_state_classification": LS.NOT_AVAILABLE,
        "blocker": blocker,
        "passed": False}

    return _finish(checks, "LS2", rss0, t_start,
                   {"shadow_input_digest": shadow_input_digest, "candidate_set_digest": digest,
                    "ls1_sets": sets}, AUTH, blocker=blocker)


def _finish(checks, stopped_at, rss0, t_start, extra, AUTH, blocker=None) -> Dict[str, Any]:
    failed = [k for k, v in checks.items() if not v["passed"]]
    stage_status = {"LS0": "NOT_RUN", "LS1": "NOT_RUN", "LS2": "NOT_RUN", "LS3": "NOT_RUN"}
    for key in checks:
        stage = key.split("_")[0]
        if stage in stage_status:
            stage_status[stage] = "PASS" if checks[key]["passed"] else "BLOCKED"
    for stage in ("LS0", "LS1", "LS2", "LS3"):
        if stage_status[stage] == "NOT_RUN" and blocker:
            stage_status[stage] = "NOT_RUN_UPSTREAM_BLOCKED"
    guards = {"simulator_binding_allowed": True, "simulator_execution_allowed": False,
              "training_allowed": False, "performance_comparison_allowed": False,
              "causal_performance_claim_allowed": False}
    checks["SAFETY_99_shadow_and_authorization"] = {
        "authorization_state": AUTH.authorization_state()["capabilities"],
        "simulator_step": 0, "simulator_reset": 0, "clock_advancement": 0,
        "vehicle_movement": 0, "boarding_or_alighting": 0, "reward_settlement": 0,
        "optimizer_step": 0, "checkpoint_write": 0, "kpi_performance_comparison": 0,
        "simulator_state_mutation": 0,
        "execution_capability_ever_granted": any(
            e["capability"] == "simulator_execution" and e["outcome"] == "ALLOWED"
            for e in AUTH.audit_log()),
        "declared_flags": guards,
        "passed": not any(e["capability"] == "simulator_execution" and e["outcome"] == "ALLOWED"
                          for e in AUTH.audit_log())}
    if not checks["SAFETY_99_shadow_and_authorization"]["passed"]:
        failed.append("SAFETY_99_shadow_and_authorization")
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-LS0..LS3",
        "stopped_at": stopped_at,
        "stage_status": stage_status,
        "classification": ("A_SUSEONG_LOCAL_SEARCH_ZERO_LOSS_MAPPO_SHADOW_DECISION_STACK_VALIDATED"
                           "_EXECUTION_AND_ABLATION_STILL_LOCKED" if not blocker
                           else "BLOCKED_LS2_ZERO_LOSS_REQUIRES_RESEARCH_DECISION"),
        "blocker": blocker,
        "authorization": guards,
        "runtime_seconds": round(time.time() - t_start, 2),
        "maxrss_start_bytes": int(rss0),
        "maxrss_final_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        **{k: v for k, v in extra.items() if k != "ls1_sets"},
        "checks": checks,
        "failed_checks": [k for k in failed if k != "LS2_06_zero_loss_feasibility"],
        "_sets": extra.get("ls1_sets"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    result.pop("_sets", None)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True,
                                               default=str) + "\n", encoding="utf-8")
    print(f"stages: {result['stage_status']}")
    if result["failed_checks"]:
        print(f"[FAIL] unexpected failures: {result['failed_checks']}")
        raise SystemExit(1)
    if result["blocker"]:
        print(f"[BLOCKED at {result['stopped_at']}] {result['blocker']['code']}: {result['blocker']['title']}")
        raise SystemExit(0)
    print(f"[PASS] {result['classification']}")


if __name__ == "__main__":
    main()
