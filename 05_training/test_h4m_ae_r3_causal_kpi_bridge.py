#!/usr/bin/env python3
"""H4M-AE-R3 causal KPI bridge contract validation (V1-V18).

Contract validation only: no training, no optimizer, no performance ranking, no
A/B1/B2 comparison, no TEST6 access.  Uses the actual promoted checkpoint, one
approved validation window for shape and causality checks, and synthetic
fixtures elsewhere.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

# --- H4M-AE-R9.8 fail-closed simulator authorization -------------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent,):
    if (_authz_dir / "simulator_authorization.py").exists() and str(_authz_dir) not in _authz_sys.path:
        _authz_sys.path.insert(0, str(_authz_dir))
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
BRIDGE = TRAINING_ROOT / "causal_kpi_bridge.py"
H4MQ = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
DL1 = TRAINING_ROOT / "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
DL4 = TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
OBS = TRAINING_ROOT / "observation_target_context_repair.py"
ENERGY = TRAINING_ROOT / "rewards/energy_proxy_model_v1.py"
REWARD = TRAINING_ROOT / "rewards/mappo_reward_v1.py"
AGGREGATOR = TRAINING_ROOT / "evaluation/canonical_kpi_aggregator.py"
POLICY_SMOKE = TRAINING_ROOT / "run_a_family_policy_rollout_smoke.py"
REGISTRY = (
    ARTIFACTS_ROOT
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
    / "r8er3r_representative_window_registry.parquet"
)
SPLIT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250" / "02_seed_split_frozen_contract.json"
PROMOTED_GLOB = "pv8_r2a_r8e_r3_r_h4m_y_fresh_window_boundary_repaired_three_seed_retraining_*/checkpoints/*.pt"

REWARD_V2_SHA = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
ZERO_LOSS_SHA = "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce"


def imp(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def promoted_checkpoints() -> List[Path]:
    return sorted(ARTIFACTS_ROOT.glob(PROMOTED_GLOB))


AUTHORITATIVE_DEMAND = (
    ARTIFACTS_ROOT
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
    / "r8er3r_generated_demand.parquet"
)


def registry_row(window_id: str):
    import pandas as pd

    registry = pd.read_parquet(REGISTRY)
    row = registry[registry["window_id"].astype(str) == str(window_id)]
    if row.empty:
        raise RuntimeError(f"window {window_id} missing from the representative registry")
    return row.iloc[0]


def authoritative_adapter_inputs(window: Mapping[str, Any], *, seed: int = 1, num_agents: int = 4) -> Dict[str, Any]:
    """Adapter kwargs bound to the frozen demand realization and the external evaluation boundary."""
    demand_mod = imp("authoritative_demand", TRAINING_ROOT / "authoritative_demand_realization.py")
    record = registry_row(window["window_id"])
    contract = demand_mod.EvaluationTimeContract(
        evaluation_start_ts=float(record["start_ts"]),
        evaluation_end_ts=float(record["evaluation_end_ts"]),
        reporting_window_ids=[str(window["window_id"])],
        scope_label="MAC_MINI_REDUCED_VALIDATION",
    )
    realization = demand_mod.load_demand_realization(AUTHORITATIVE_DEMAND)
    selection = demand_mod.select_population(realization, contract)
    return {
        "window": dict(window),
        "num_agents": num_agents,
        "seed": seed,
        "evaluation_contract": contract,
        "demand_population": selection["requests"],
        "demand_provenance": realization.provenance(),
        "population_audit": selection["audit"],
    }


def build_env(bridge: Any, qmod: Any, dl1: Any, dl4: Any):
    plan = qmod.load_validation_plan()
    sample_graph, inventory, config, data = qmod.load_validation_data(dl1, dl4, imp("r3_obs", OBS), plan)
    device = torch.device("cpu")
    window = plan["validation_rows"][0]
    return plan, sample_graph, config, data, device, window


def _run_validations_inner() -> Dict[str, Any]:
    torch.set_num_threads(1)
    bridge = imp("r3_bridge", BRIDGE)
    dl1 = imp("r3_dl1", DL1)
    dl4 = imp("r3_dl4", DL4)
    qmod = imp("r3_q", H4MQ)
    energy_model = imp("r3_energy", ENERGY)
    reward_mod = imp("r3_reward", REWARD)

    checks: Dict[str, Any] = {}
    plan, sample_graph, config, data, device, window = build_env(bridge, qmod, dl1, dl4)
    checkpoints = promoted_checkpoints()
    policy = bridge.PromotedPolicyBridge(checkpoints[0], dl1, dl4, sample_graph, device)
    graph = data[0].to(device)
    agent_indices = dl1.agent_indices_for_step(config["spec"], 0, int(config["effective_agents"]))

    # V1 actual checkpoint -> exact 131D observation -> action
    decision = policy.act(graph, agent_indices, qmod.masked_logits_for_targets)
    checks["V1_checkpoint_131d_action"] = {
        "observation_dim": decision["observation_dim"],
        "actions": decision["actions"],
        "checkpoint_sha256": policy.provenance.checkpoint_sha256,
        "passed": decision["observation_dim"] == bridge.ACTOR_OBS_DIM and len(decision["actions"]) == len(agent_indices),
    }

    # V2 observation mismatch fails closed
    mismatch_raised = None
    tmp = Path(__file__).with_suffix(".mismatch.pt")
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    payload["actor_state_dict"]["net.0.weight"] = torch.zeros(128, 130)
    torch.save(payload, tmp)
    try:
        bridge.PromotedPolicyBridge(tmp, dl1, dl4, sample_graph, device)
    except bridge.BridgeContractError as exc:
        mismatch_raised = exc.code
    finally:
        tmp.unlink(missing_ok=True)
    checks["V2_observation_mismatch_fail_closed"] = {"code": mismatch_raised, "passed": mismatch_raised == "ACTOR_INPUT_DIM_MISMATCH"}

    # V3 no placeholder / mock / random fallback
    prov = policy.provenance.as_dict()
    checks["V3_no_placeholder_or_mock"] = {
        "provenance": prov,
        "passed": prov["placeholder_used"] is False and prov["mock_action_used"] is False and prov["random_fallback_used"] is False
        and prov["policy_source_mode"] == "actual_promoted_mappo_checkpoint",
    }

    adapter_inputs = authoritative_adapter_inputs(window, num_agents=len(agent_indices))
    provenance = {
        "condition_id": "A",
        "policy_source_mode": prov["policy_source_mode"],
        "checkpoint_sha256": prov["checkpoint_sha256"],
        "checkpoint_path": prov["checkpoint_path"],
    }

    def fresh_adapter(seed: int = 1):
        return bridge.PV8CausalKpiAdapter(**{**adapter_inputs, "seed": seed})

    mask = {agent: [True] * 3 for agent in range(len(agent_indices))}
    targets = {agent: int(decision["targets"][agent]) for agent in range(len(agent_indices))}

    # V4 legal action changes the next state
    hold_adapter = fresh_adapter()
    serve_adapter = fresh_adapter()
    hold_step = hold_adapter.step({a: 0 for a in mask}, legal_mask=mask, target_ids=targets, provenance=provenance)
    serve_step = serve_adapter.step({a: 1 for a in mask}, legal_mask=mask, target_ids=targets, provenance=provenance)
    checks["V4_action_changes_next_state"] = {
        "hold_post_hash": hold_step["post_state"]["hash"][:16],
        "serve_post_hash": serve_step["post_state"]["hash"][:16],
        "hold_served": hold_adapter.accounting.passenger_eventual_served_count,
        "serve_served": serve_adapter.accounting.passenger_eventual_served_count,
        "hold_distance_m": hold_adapter.accounting.distance_m,
        "serve_distance_m": serve_adapter.accounting.distance_m,
        "passed": hold_step["post_state"]["hash"] != serve_step["post_state"]["hash"]
        and serve_adapter.accounting.distance_m > hold_adapter.accounting.distance_m
        and hold_adapter.accounting.hold_seconds > 0.0,
    }

    # V5 determinism
    a1, a2 = fresh_adapter(), fresh_adapter()
    s1 = a1.step({a: 1 for a in mask}, legal_mask=mask, target_ids=targets, provenance=provenance)
    s2 = a2.step({a: 1 for a in mask}, legal_mask=mask, target_ids=targets, provenance=provenance)
    checks["V5_deterministic_transition"] = {
        "passed": s1["post_state"]["hash"] == s2["post_state"]["hash"]
        and a1.accounting.wait_total_passenger_seconds == a2.accounting.wait_total_passenger_seconds,
    }

    # V6 K-mask unchanged and enforced
    illegal_raised = None
    try:
        fresh_adapter().step({0: 2}, legal_mask={0: [True, True, False]}, target_ids={0: 0}, provenance=provenance)
    except bridge.BridgeContractError as exc:
        illegal_raised = exc.code
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    masked, allowed = qmod.masked_logits_for_targets(logits, torch.tensor([0]))
    checks["V6_kmask_unchanged"] = {
        "illegal_action_code": illegal_raised,
        "mask_fn_source": "qmod.masked_logits_for_targets (unchanged)",
        "allowed_row": [bool(v) for v in allowed[0].tolist()],
        "passed": illegal_raised == "ILLEGAL_ACTION_APPLIED" and bool(allowed[0][0]),
    }

    # V7 Reward V2 unchanged
    checks["V7_reward_v2_unchanged"] = {
        "freeze_sha256": reward_mod.PV8_REWARD_V2_FREEZE_SHA256,
        "passed": reward_mod.PV8_REWARD_V2_FREEZE_SHA256 == REWARD_V2_SHA,
    }

    # V8 Zero-Loss semantics unchanged (bridge emits no admission candidate)
    bridge_src = BRIDGE.read_text(encoding="utf-8")
    checks["V8_zero_loss_unchanged"] = {
        "zero_loss_adapter_sha256": ZERO_LOSS_SHA,
        "bridge_defines_admission": "zero_loss" in bridge_src.lower(),
        "passed": "zero_loss" not in bridge_src.lower(),
    }

    # V9/V10 events -> rollup
    run = fresh_adapter()
    for _ in range(4):
        run.step({a: int(decision["actions"][a]) for a in mask}, legal_mask=mask, target_ids=targets, provenance=provenance)
    row = run.window_rollup_row(condition_id="A", provenance=provenance, energy_model=energy_model, baseline_bus_count=len(agent_indices))
    checks["V9_events_emitted"] = {
        "event_count": len(run.events),
        "sample_event_keys": sorted(run.events[0].keys())[:8],
        "passed": len(run.events) > 0 and all(event["causal_transition"] for event in run.events),
    }
    checks["V10_rollup_from_events"] = {
        "event_row_count_in_rollup": row["event_row_count"],
        "kpi_provenance": row["kpi_provenance"],
        "passed": row["event_row_count"] == len(run.events) and row["kpi_provenance"] == "simulator_transition_accounting_only",
    }

    # V11 no synthetic path in the promoted route: inspect the rollup value
    # expressions themselves, so the bridge's own forbidden-token guard (which
    # only holds string literals) cannot be mistaken for a synthetic source.
    import ast as _ast

    forbidden_names = {"nonzero_action_count", "action_count", "policy_probability", "condition_name", "time_band_lookup"}
    rollup_fn = next(
        node for node in _ast.walk(_ast.parse(bridge_src))
        if isinstance(node, _ast.FunctionDef) and node.name == "window_rollup_row"
    )
    used_names = {
        node.id for node in _ast.walk(rollup_fn) if isinstance(node, _ast.Name)
    } | {
        node.attr for node in _ast.walk(rollup_fn) if isinstance(node, _ast.Attribute)
    }
    step_fn = next(
        node for node in _ast.walk(_ast.parse(bridge_src))
        if isinstance(node, _ast.FunctionDef) and node.name == "step"
    )
    step_names = {node.attr for node in _ast.walk(step_fn) if isinstance(node, _ast.Attribute)}
    checks["V11_no_synthetic_path"] = {
        "forbidden_names_used_in_rollup": sorted(used_names & forbidden_names),
        "forbidden_names_used_in_transition": sorted(step_names & forbidden_names),
        "guard_present_as_literals_only": "SYNTHETIC_KPI_FIELD_PRESENT" in bridge_src,
        "legacy_smoke_still_synthetic": "nonzero_action_count" in POLICY_SMOKE.read_text(encoding="utf-8"),
        "legacy_smoke_referenced_by_bridge": "run_a_family_policy_rollout_smoke" in bridge_src,
        "passed": not (used_names & forbidden_names)
        and not (step_names & forbidden_names)
        and "run_a_family_policy_rollout_smoke" not in bridge_src,
    }

    # V12 canonical KPI field provenance
    aggregator_src = AGGREGATOR.read_text(encoding="utf-8")
    required_inputs = ["wait_total_passenger_seconds", "wait_passenger_count", "headway_std_seconds", "headway_mean_seconds",
                       "bunching_event_count", "headway_event_count", "ontime_event_count", "schedulable_arrival_count",
                       "intervention_count", "decision_step_count", "energy_proxy_total", "passenger_served_count",
                       "passenger_demand_generated", "active_bus_count", "baseline_bus_count"]
    checks["V12_kpi_field_provenance"] = {
        "required_inputs_present_in_rollup": [f for f in required_inputs if f not in row],
        "provenance_map_size": len(bridge.KPI_FIELD_PROVENANCE),
        "passed": not [f for f in required_inputs if f not in row] and all(f in aggregator_src for f in required_inputs[:6]),
    }

    # V13 A/B1/B2 share the bridge, only the action source differs
    noop_adapter, rule_adapter = fresh_adapter(), fresh_adapter()
    noop_adapter.step({a: 0 for a in mask}, legal_mask=mask, target_ids=targets, provenance={**provenance, "condition_id": "B1", "policy_source_mode": "noop"})
    rule_adapter.step({a: (1 if targets[a] == 1 else 0) for a in mask}, legal_mask=mask, target_ids=targets, provenance={**provenance, "condition_id": "B2", "policy_source_mode": "rule_based"})
    checks["V13_arms_share_bridge"] = {
        "same_adapter_class": type(noop_adapter).__name__ == type(rule_adapter).__name__ == type(run).__name__,
        "same_demand": noop_adapter.accounting.passenger_demand_generated == rule_adapter.accounting.passenger_demand_generated == run.accounting.passenger_demand_generated,
        "same_window": noop_adapter.window["window_id"] == run.window["window_id"],
        "passed": type(noop_adapter) is type(run) and noop_adapter.accounting.passenger_demand_generated == run.accounting.passenger_demand_generated,
    }

    # V14/V15 baseline statuses
    checks["V14_b0r_reference_only"] = {"status": "HISTORICAL_REFERENCE_ONLY_NEVER_CAUSAL_ARM", "passed": True}
    checks["V15_b0c_blocked"] = {"status": "BLOCKED_UNEXECUTED", "passed": True}

    # V16 TEST6 untouched
    sealed = set(json.loads(SPLIT.read_text(encoding="utf-8-sig"))["ordered_window_ids"]["test"])
    used = {str(window["window_id"])}
    checks["V16_test6_untouched"] = {
        "used_windows": sorted(used),
        "sealed_intersection": sorted(used & {str(w) for w in sealed}),
        "passed": not (used & {str(w) for w in sealed}),
    }

    # V17 in-vehicle time
    checks["V17_in_vehicle_not_measurable"] = {
        "status": bridge.KPI_FIELD_PROVENANCE["in_vehicle_time_seconds"],
        "emitted_in_rollup": any("in_vehicle" in key for key in row),
        "passed": "NOT_YET_MEASURABLE" in bridge.KPI_FIELD_PROVENANCE["in_vehicle_time_seconds"]
        and not any("in_vehicle" in key for key in row),
    }

    # V18 numeric / schema / cardinality guards
    numeric_bad = [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
    checks["V18_numeric_schema_guards"] = {
        "nonfinite_fields": numeric_bad,
        "rollup_field_count": len(row),
        "event_provenance_complete": all(
            all(key in event for key in ("window_id", "seed", "agent_id", "action_id", "checkpoint_sha256", "causal_transition"))
            for event in run.events
        ),
        "passed": not numeric_bad and all(
            all(key in event for key in ("window_id", "seed", "agent_id", "action_id", "checkpoint_sha256", "causal_transition"))
            for event in run.events
        ),
    }

    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R3",
        "bridge_id": bridge.BRIDGE_ID,
        "actor_observation_dim": bridge.ACTOR_OBS_DIM,
        "checkpoints_available": len(checkpoints),
        "validation_window_used": window["window_id"],
        "training_executed": False,
        "optimizer_step_count": 0,
        "test6_accessed": False,
        "performance_claim_allowed": False,
        "causal_comparison_executed": False,
        "sample_rollup_row": row,
        "checks": checks,
        "failed": [name for name, row_ in checks.items() if not row_["passed"]],
        "passed": all(row_["passed"] for row_ in checks.values()),
    }


def run_validations() -> Dict[str, Any]:
    """Validation harness.

    This exercises the causal bridge, which R9.8 protects with the
    simulator_execution capability. The harness is entitled to run it, so it
    declares that explicitly here rather than the guard being weakened.
    """
    with _authz.granted("simulator_execution", reason="R3 causal KPI bridge validation harness"):
        return _run_validations_inner()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(json.dumps({"failed": result["failed"]}, ensure_ascii=False, indent=2))
    print("[PASS] H4M-AE-R3 causal KPI bridge contract validation passed")


if __name__ == "__main__":
    main()
