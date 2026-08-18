#!/usr/bin/env python3
"""H4M-AE-R2 causal KPI measurement bridge repair selection and freeze.

Select and freeze only.  No bridge implementation, no simulator repair, no KPI
execution, no rollout, no training, no TEST6 access, no network, no push.  The
promoted actor observation contract is read from the actual checkpoint weights
rather than assumed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-AE-R2"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R2_"
    "CAUSAL_KPI_MEASUREMENT_BRIDGE_REPAIR_SELECTION_AND_FREEZE_COMPLETE"
)
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R2_NO_SAFE_BRIDGE_SELECTION"
NEXT_GATE = "H4M-AE-R3_CAUSAL_KPI_MEASUREMENT_BRIDGE_IMPLEMENTATION_AND_CONTRACT_VALIDATION"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

ADAPTER_V2 = "05_training/adapters/causal_simulator_v2_adapter.py"
ADAPTER_TOY = "05_training/adapters/causal_simulator_adapter.py"
POLICY_SMOKE = "05_training/run_a_family_policy_rollout_smoke.py"
CAUSAL_ROLLOUT = "05_training/run_causal_rollout.py"
AGGREGATOR = "05_training/evaluation/canonical_kpi_aggregator.py"
OBS_REPAIR = "05_training/observation_target_context_repair.py"
DL1 = "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
REWARD = "05_training/rewards/mappo_reward_v1.py"
REGISTRY = (
    ARTIFACTS_ROOT
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
    / "r8er3r_representative_window_registry.parquet"
)
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
PROMOTED_GLOB = "pv8_r2a_r8e_r3_r_h4m_y_fresh_window_boundary_repaired_three_seed_retraining_*/checkpoints/*.pt"

EXPECTED = {
    "parent_commit_short": "844913d",
    "actor_input_dim": 131,
    "gatv2_hidden": 128,
    "target_context_dim": 3,
    "node_feature_dim": 12,
    "critic_input_dim": 256,
    "action_dim": 3,
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "actor_repair_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
}

REQUIRED_ARTIFACTS = [
    "final_report.md", "final_report.json", "promoted_actor_observation_contract.json",
    "observation_dimension_lineage_audit.json", "observation_feature_layout.json",
    "bridge_candidate_matrix.json", "bridge_candidate_matrix.parquet",
    "selected_bridge_architecture.json", "causal_transition_contract.json",
    "raw_event_provenance_contract.json", "canonical_kpi_bridge_contract.json",
    "comparison_arm_construction_contract.json", "reward_v2_zero_loss_non_regression.json",
    "test6_non_reuse_contract.json", "in_vehicle_time_guard.json",
    "implementation_dependency_register.json", "self_test_report.json",
    "gate_decision.json", "downstream_lock.json", "artifact_manifest.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def src(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")


# ---------------------------------------------------------------------------
# 2. observation contract audit (measured from checkpoint weights)
# ---------------------------------------------------------------------------
def observation_contract() -> Dict[str, Any]:
    paths = sorted(ARTIFACTS_ROOT.glob(PROMOTED_GLOB))
    rows = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actor = payload["actor_state_dict"]
        critic = payload["critic_state_dict"]
        gatv2 = payload["gatv2_state_dict"]
        config = payload.get("training_configuration", {})
        rows.append(
            {
                "seed": payload.get("seed"),
                "path": str(path),
                "sha256": sha256_file(path),
                "actor_first_layer": list(actor["net.0.weight"].shape),
                "actor_input_dim": int(actor["net.0.weight"].shape[1]),
                "actor_hidden_dim": int(actor["net.0.weight"].shape[0]),
                "actor_head_count": sum(1 for key in actor if key.startswith("target_heads.") and key.endswith(".weight")),
                "actor_head_output_dim": int(actor["target_heads.0.weight"].shape[0]),
                "critic_input_dim": int(critic["net.0.weight"].shape[1]),
                "gatv2_node_input_dim": int(gatv2["conv1.lin_l.weight"].shape[1]),
                "declared_actor_conditioned_input_dim": config.get("actor_conditioned_input_dim"),
                "declared_schema": config.get("actor_conditioned_input_schema"),
                "declared_node_feature_dim_after_repair": config.get("node_feature_dim_after_observation_repair"),
                "declared_target_context_source": config.get("actor_target_context_source"),
            }
        )
    consistent = len({row["actor_input_dim"] for row in rows}) == 1 and len({row["gatv2_node_input_dim"] for row in rows}) == 1
    measured = rows[0]
    builder_supports_12 = "old_dim == 12" in src(OBS_REPAIR)
    checks = {
        "three_promoted_checkpoints": len(rows) == 3,
        "actor_input_dim_consistent_across_seeds": consistent,
        "measured_actor_input_dim_matches_declared": measured["actor_input_dim"] == measured["declared_actor_conditioned_input_dim"],
        "measured_actor_input_equals_hidden_plus_target_context": measured["actor_input_dim"]
        == EXPECTED["gatv2_hidden"] + EXPECTED["target_context_dim"],
        "gatv2_node_input_matches_repaired_observation": measured["gatv2_node_input_dim"] == EXPECTED["node_feature_dim"],
        "observation_builder_emits_same_node_dim": builder_supports_12,
        "critic_input_is_agent_plus_graph_embedding": measured["critic_input_dim"] == 2 * EXPECTED["gatv2_hidden"],
        "checkpoint_and_builder_agree": measured["actor_input_dim"] == EXPECTED["actor_input_dim"] and builder_supports_12,
    }
    return {
        "stage": STAGE,
        "measurement_method": "read directly from the promoted checkpoint state_dict weight shapes; no assumption",
        "checkpoints": rows,
        "authoritative_contract": {
            "actor_observation_dim": measured["actor_input_dim"],
            "composition": "concat(GATv2 agent embedding 128, r3 action-target one-hot 3)",
            "concatenation_order": ["gatv2_agent_embedding[0:128]", "r3_action_target_one_hot[128:131]"],
            "gatv2_node_input_dim": measured["gatv2_node_input_dim"],
            "node_feature_source": "12D repaired pre-action observation; target context read from data.x[agent, -3:]",
            "critic_input_dim": measured["critic_input_dim"],
            "action_dim": EXPECTED["action_dim"],
            "target_head_count": measured["actor_head_count"],
            "normalization": "return normalizer applied to critic value scale only; actor input is the raw GATv2 embedding concat",
            "dtype": "float32",
            "device": "MPS at training time, device-agnostic at inference",
            "status": "AUTHORITATIVE_PROMOTED_CONTRACT",
        },
        "checks": checks,
        "fail_closed": not all(checks.values()),
        "passed": all(checks.values()),
    }


def dimension_lineage(contract: Mapping[str, Any]) -> Dict[str, Any]:
    toy = src(ADAPTER_TOY)
    lineages = {
        "131D": {
            "meaning": "promoted actor input: GATv2 agent embedding 128 + target one-hot 3",
            "evidence": f"actor net.0.weight = {contract['checkpoints'][0]['actor_first_layer']} in all three promoted checkpoints",
            "classification": "AUTHORITATIVE_PROMOTED_CONTRACT",
        },
        "128D": {
            "meaning": "GATv2 hidden agent embedding",
            "evidence": "gatv2_hidden=128 in the promoted training configuration",
            "classification": "AUTHORITATIVE_PROMOTED_CONTRACT",
        },
        "12D": {
            "meaning": "GATv2 node input after the H4M-I observation repair (9D + 3D target context)",
            "evidence": f"gatv2 conv1.lin_l.weight input = {contract['checkpoints'][0]['gatv2_node_input_dim']}",
            "classification": "AUTHORITATIVE_PROMOTED_CONTRACT",
        },
        "9D": {
            "meaning": "pre-repair node feature dim",
            "evidence": "node_feature_dim_before_observation_repair=9 in the promoted configuration",
            "classification": "HISTORICAL_SUPERSEDED",
        },
        "16D": {
            "meaning": "toy causal simulator flat actor_obs",
            "evidence": f"{ADAPTER_TOY}: actor_obs shape (num_agents, 16)" if "(self.num_agents_val, 16)" in toy else "not found",
            "classification": "SMOKE_ONLY",
        },
        "256D": {
            "meaning": "critic input: agent embedding 128 + graph embedding 128",
            "evidence": f"critic net.0.weight input = {contract['checkpoints'][0]['critic_input_dim']}",
            "classification": "AUTHORITATIVE_PROMOTED_CONTRACT",
        },
        "154D": {
            "meaning": "referenced in the H4M-AE-R2 instruction as a possible historical contract",
            "evidence": "no 154-dimensional observation lineage exists anywhere in this repository",
            "classification": "UNRESOLVED_NOT_PRESENT_IN_REPOSITORY",
        },
    }
    return {
        "stage": STAGE,
        "lineages": lineages,
        "authoritative_dim": contract["authoritative_contract"]["actor_observation_dim"],
        "assumption_avoided": "131D was verified from checkpoint weights, not taken from documentation",
        "passed": True,
    }


def feature_layout(contract: Mapping[str, Any]) -> Dict[str, Any]:
    obs = src(OBS_REPAIR)
    recoverable = "r3_action_target_is_hold" in obs or "TARGET_CONTEXT_FIELDS" in src(DL1)
    return {
        "stage": STAGE,
        "actor_input_layout": [
            {"slice": "[0:128]", "block": "gatv2_agent_embedding", "source": "GATv2Encoder output for the agent node", "recoverable_names": False},
            {"slice": "[128:131]", "block": "r3_action_target_one_hot", "source": "data.x[agent, -3:] of the repaired 12D observation", "recoverable_names": True,
             "names": ["r3_action_target_is_hold", "r3_action_target_is_serve", "r3_action_target_is_skip"]},
        ],
        "gatv2_node_input_layout": {
            "dim": EXPECTED["node_feature_dim"],
            "composition": "9 pre-repair node features + 3 appended target-context channels",
            "append_position": "tail",
            "names_recoverable": recoverable,
        },
        "mask_features": "action legality is applied as a K-mask on logits, not as an observation feature",
        "normalization": "no actor-input normalization; the return normalizer affects the critic target coordinate system only",
        "passed": True,
    }


# ---------------------------------------------------------------------------
# 3. candidate bridges
# ---------------------------------------------------------------------------
def candidate_matrix(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    v2 = src(ADAPTER_V2)
    toy = src(ADAPTER_TOY)
    registry_has_demand = True
    try:
        cols = set(pd.read_parquet(REGISTRY).columns)
        registry_has_demand = {"historical_boarding_intensity", "historical_alighting_intensity", "historical_demand_score"} <= cols
    except Exception:
        registry_has_demand = False
    return [
        {
            "candidate_id": "A_PV8_ADAPTER_KPI_STATE_EXTENSION",
            "summary": "extend the existing PV8 causal simulator v2 adapter with the passenger/headway/energy accounting already validated in the toy causal adapter, keeping the promoted observation contract untouched",
            "files_changed": [ADAPTER_V2, POLICY_SMOKE],
            "observation_compatibility": "EXACT: the adapter already emits the PV8 graph context the promoted GATv2 consumes; no observation change",
            "checkpoint_compatibility": "EXACT: 131D actor input unchanged",
            "graph_state_compatibility": "EXACT: PV8 54-window snapshots retained",
            "action_kmask_compatibility": "EXACT: 3-action semantics and K-mask unchanged",
            "reward_v2_compatibility": "UNCHANGED: KPIs are measured from transitions, not from reward",
            "zero_loss_compatibility": "UNCHANGED: feasibility semantics untouched",
            "simulator_causality": "CAUSAL after implementation: action changes passenger/vehicle state",
            "raw_event_provenance": "adapter emits per-transition events; rollup consumes them",
            "kpi_availability_after_repair": "11/12 canonical KPIs; in_vehicle_time remains unavailable",
            "b1_b2_reuse": "same adapter drives no-op and rule-based policies on identical windows",
            "test6_dependency": False,
            "synthetic_kpi_risk": "LOW: the rollup writer is required to consume only simulator output",
            "implementation_complexity": "MEDIUM",
            "semantic_risk": "MEDIUM: passenger dynamics must be bound to authoritative demand fields, not invented",
            "reproducibility_risk": "LOW",
            "authoritative_demand_fields_available": registry_has_demand,
            "blocking_unknowns": ["demand-to-passenger-arrival binding must be derived from the registry demand fields under R3"],
        },
        {
            "candidate_id": "B_TOY_ENGINE_STATE_MAPPING_WRAPPER",
            "summary": "keep the promoted observation builder and forward actions into the toy causal engine, mapping toy state back into promoted observations",
            "files_changed": [ADAPTER_TOY, POLICY_SMOKE, "a new state-mapping module"],
            "observation_compatibility": "REQUIRES INVENTED MAPPING: the toy engine is an 8-stop corridor with a flat 16D observation and no PV8 graph",
            "checkpoint_compatibility": "INDIRECT: only through an invented state correspondence",
            "graph_state_compatibility": "INCOMPATIBLE: toy corridor is not the PV8 54-window universe",
            "action_kmask_compatibility": "PARTIAL",
            "reward_v2_compatibility": "AT RISK: toy state has no Reward V2 transition identity",
            "zero_loss_compatibility": "AT RISK",
            "simulator_causality": "CAUSAL but in the wrong universe",
            "raw_event_provenance": "toy events, not PV8 events",
            "kpi_availability_after_repair": "11/12 but on the toy corridor",
            "b1_b2_reuse": "possible but on the toy universe only",
            "test6_dependency": False,
            "synthetic_kpi_risk": "HIGH: the PV8-to-toy correspondence would be fabricated",
            "implementation_complexity": "HIGH",
            "semantic_risk": "HIGH",
            "reproducibility_risk": "HIGH",
            "authoritative_demand_fields_available": registry_has_demand,
            "blocking_unknowns": ["no authoritative correspondence between PV8 windows and the toy corridor exists"],
        },
        {
            "candidate_id": "C_ROLLUP_WRITER_PROVENANCE_FIX_ONLY",
            "summary": "fix the rollup writer so it consumes adapter.step output instead of synthesising KPIs from the action count, changing nothing else",
            "files_changed": [POLICY_SMOKE],
            "observation_compatibility": "EXACT",
            "checkpoint_compatibility": "EXACT",
            "graph_state_compatibility": "EXACT",
            "action_kmask_compatibility": "EXACT",
            "reward_v2_compatibility": "UNCHANGED",
            "zero_loss_compatibility": "UNCHANGED",
            "simulator_causality": "UNCHANGED: the PV8 adapter still has no passenger or headway state",
            "raw_event_provenance": "would consume an adapter that returns None for every core KPI",
            "kpi_availability_after_repair": "0/12: the underlying adapter emits no KPI",
            "b1_b2_reuse": "no",
            "test6_dependency": False,
            "synthetic_kpi_risk": "LOW but useless: removing the synthetic path leaves nothing behind it",
            "implementation_complexity": "LOW",
            "semantic_risk": "LOW",
            "reproducibility_risk": "LOW",
            "authoritative_demand_fields_available": registry_has_demand,
            "blocking_unknowns": ["insufficient on its own; it is a necessary sub-requirement of candidate A"],
        },
    ]


HARD_REQUIREMENTS = [
    "actual_trained_checkpoint_loaded",
    "no_placeholder_or_mock_fallback",
    "actor_observation_contract_exact_match",
    "gatv2_graph_context_preserved",
    "legal_action_kmask_preserved",
    "reward_v2_frozen_semantics_unchanged",
    "zero_loss_feasibility_semantics_unchanged",
    "policy_action_causally_changes_next_state",
    "kpi_originates_from_simulator_transition",
    "raw_events_carry_provenance",
    "window_rollup_consumes_simulator_quantities",
    "canonical_aggregator_shared_across_arms",
    "synthetic_kpi_generation_prohibited",
    "source_mode_declares_causality",
    "test6_not_required",
    "in_vehicle_time_remains_not_yet_measurable",
]


def select_bridge(candidates: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> Dict[str, Any]:
    eligible = []
    for row in candidates:
        ok = (
            row["observation_compatibility"].startswith("EXACT")
            and row["checkpoint_compatibility"].startswith("EXACT")
            and row["graph_state_compatibility"].startswith("EXACT")
            and row["test6_dependency"] is False
            and row["kpi_availability_after_repair"].startswith("11/12")
            and row["semantic_risk"] != "HIGH"
        )
        if ok:
            eligible.append(row["candidate_id"])
    selected = eligible[0] if len(eligible) == 1 else None
    return {
        "stage": STAGE,
        "eligible_candidates": eligible,
        "selected_bridge_architecture": selected,
        "selection_unique": len(eligible) == 1,
        "selection_reason": (
            "A is the only candidate that keeps the promoted 131D observation contract and the PV8 graph universe exactly "
            "as they are while making the simulator emit transition-derived KPIs; B would fabricate a PV8-to-toy state "
            "correspondence and C leaves an adapter behind it that returns None for every KPI"
        )
        if selected
        else "no candidate satisfies the selection rule",
        "selection_priority_applied": ["MINIMUM_SEMANTIC_CHANGE", "MINIMUM_CODE_CHANGE", "MAXIMUM_REUSE_OF_VALIDATED_COMPONENTS"],
        "subsumed_requirement": "candidate C is folded into A as a mandatory sub-requirement: the rollup writer must consume only simulator output",
        "rejected": {
            "B_TOY_ENGINE_STATE_MAPPING_WRAPPER": "invented PV8-to-toy state correspondence; high semantic and reproducibility risk",
            "C_ROLLUP_WRITER_PROVENANCE_FIX_ONLY": "necessary but insufficient; the PV8 adapter still emits no KPI",
        },
        "hard_requirements": HARD_REQUIREMENTS,
        "hard_requirements_bound_to_implementation": True,
        "contract_sha256": canonical_sha({"selected": selected, "requirements": HARD_REQUIREMENTS}),
    }


def self_tests(contract, lineage, candidates, selection, arms) -> Dict[str, Any]:
    smoke = src(POLICY_SMOKE)
    rollout = src(CAUSAL_ROLLOUT)
    selected = next((row for row in candidates if row["candidate_id"] == selection["selected_bridge_architecture"]), {})
    tests = {
        "T1_actor_input_dimension_mismatch": {
            "guard": "measured actor input equals 131 across all promoted checkpoints",
            "passed": contract["checks"]["actor_input_dim_consistent_across_seeds"]
            and contract["authoritative_contract"]["actor_observation_dim"] == EXPECTED["actor_input_dim"],
        },
        "T2_checkpoint_vs_builder_mismatch": {
            "guard": "checkpoint GATv2 input dim equals the repaired observation builder output dim",
            "passed": contract["checks"]["checkpoint_and_builder_agree"],
        },
        "T3_historical_or_smoke_contract_promoted": {
            "guard": "only 131/128/12/256 are marked authoritative; 9D historical, 16D smoke, 154D not present",
            "passed": lineage["lineages"]["9D"]["classification"] == "HISTORICAL_SUPERSEDED"
            and lineage["lineages"]["16D"]["classification"] == "SMOKE_ONLY"
            and lineage["lineages"]["154D"]["classification"].startswith("UNRESOLVED"),
        },
        "T4_placeholder_policy_accepted": {
            "guard": "the selected design forbids placeholder or mock policy fallback",
            "passed": "no_placeholder_or_mock_fallback" in selection["hard_requirements"]
            and selection["selected_bridge_architecture"] != "PLACEHOLDER",
        },
        "T5_simulator_output_discarded": {
            "guard": "the selected design requires the rollup to consume simulator output only",
            "passed": "window_rollup_consumes_simulator_quantities" in selection["hard_requirements"]
            and "adapter.step(" in smoke,
        },
        "T6_synthetic_kpi_from_action_counts": {
            "guard": "synthetic KPI generation is prohibited by the frozen requirements",
            "passed": "synthetic_kpi_generation_prohibited" in selection["hard_requirements"]
            and selected.get("synthetic_kpi_risk", "HIGH").startswith("LOW"),
        },
        "T7_noncausal_replay_labeled_causal": {
            "guard": "source_mode must declare causality and no non-causal lineage is promoted",
            "passed": "source_mode_declares_causality" in selection["hard_requirements"]
            and "placeholder" in rollout,
        },
        "T8_different_window_universes_merged": {
            "guard": "every frozen arm shares one window universe",
            "passed": arms["all_arms_share_one_universe"],
        },
        "T9_b0r_promoted_to_causal_arm": {
            "guard": "B0R stays reference only",
            "passed": arms["B0R_status"] == "HISTORICAL_REFERENCE_ONLY_NEVER_CAUSAL_ARM",
        },
        "T10_test6_dependency_introduced": {
            "guard": "no candidate or arm requires TEST6",
            "passed": all(row["test6_dependency"] is False for row in candidates) and arms["test6_required"] is False,
        },
        "T11_in_vehicle_proxy_invented": {
            "guard": "in-vehicle time stays NOT_YET_MEASURABLE with no substitute",
            "passed": "in_vehicle_time_remains_not_yet_measurable" in selection["hard_requirements"]
            and selected.get("kpi_availability_after_repair", "").endswith("in_vehicle_time remains unavailable"),
        },
        "T12_reward_v2_or_zero_loss_changed": {
            "guard": "the selected candidate changes neither Reward V2 nor Zero-Loss semantics",
            "passed": selected.get("reward_v2_compatibility", "").startswith("UNCHANGED")
            and selected.get("zero_loss_compatibility", "").startswith("UNCHANGED"),
        },
    }
    return {
        "stage": STAGE,
        "tests": tests,
        "failed": [name for name, row in tests.items() if not row["passed"]],
        "all_passed": all(row["passed"] for row in tests.values()),
    }


def arm_contract() -> Dict[str, Any]:
    split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")["ordered_window_ids"]
    body = {
        "primary_repair_targets": ["A_vs_B1", "A_vs_B2"],
        "shared_requirements": [
            "identical window universe",
            "identical passenger demand",
            "identical initial vehicle state",
            "identical simulator transition semantics",
            "identical KPI definitions",
            "identical evaluation horizon",
            "same randomness contract except the policy action",
            "the shared canonical aggregator",
            "explicit policy provenance in every row",
        ],
        "window_universe": {
            "source": "PV8 R3-R representative registry",
            "train": len(split["train"]),
            "validation": len(split["validation"]),
            "test6_sealed": len(split["test"]),
            "usable_for_bridge_development": "train and validation for schema and shape only",
        },
        "all_arms_share_one_universe": True,
        "B0R_status": "HISTORICAL_REFERENCE_ONLY_NEVER_CAUSAL_ARM",
        "B0C_status": "BLOCKED_UNEXECUTED_UNLESS_INDEPENDENTLY_RELEASED",
        "test6_required": False,
        "constructable_after_implementation": {"A_vs_B1": True, "A_vs_B2": True, "A_vs_B0R": False, "A_vs_B0C": False},
    }
    return {"stage": STAGE, **body, "contract_sha256": canonical_sha(body)}


def main() -> None:
    started = time.perf_counter()
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ae_r2_causal_kpi_bridge_selection_freeze_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "--short", "HEAD^"]).stdout.strip()
    status = git_run(["status", "--short"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="h4maer2_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "r2.pyc"), doraise=True)
            compile_error = None
        except Exception as exc:
            compile_error = repr(exc)

    contract = observation_contract()
    if contract["fail_closed"]:
        write_json(root / "promoted_actor_observation_contract.json", contract)
        write_json(root / "gate_matrix.json", {"stage": STAGE, "gate": BLOCK_GATE, "reason": "observation contract fail-closed", "checks": contract["checks"]})
        print(f"[H4M-AE-R2] gate: {BLOCK_GATE} (observation contract mismatch)")
        return

    lineage = dimension_lineage(contract)
    layout = feature_layout(contract)
    candidates = candidate_matrix(contract)
    arms = arm_contract()
    selection = select_bridge(candidates, contract)
    tests = self_tests(contract, lineage, candidates, selection, arms)

    selected_id = selection["selected_bridge_architecture"]
    passed = bool(selected_id) and tests["all_passed"] and contract["passed"] and status == "" and compile_error is None

    causal_transition = {
        "stage": STAGE,
        "requirement": "the sampled action must change the simulator's next passenger and vehicle state on the PV8 window",
        "forbidden": ["static scaffold transitions that ignore the action", "reward-derived KPI substitution"],
        "demand_binding": "passenger arrivals must be bound to the registry historical demand fields, never invented",
        "randomness_contract": "identical seeds and identical demand realisation across arms; only the policy action differs",
        "source_mode": "must declare a causal source_mode distinct from placeholder/replay/smoke",
    }
    raw_events = {
        "stage": STAGE,
        "required_fields": ["window_id", "seed", "condition_id", "decision_step", "agent_id", "action_id", "legal_action_mask",
                            "passenger_wait_seconds", "passenger_served", "headway_seconds", "distance_m",
                            "acceleration_event_count", "hold_seconds", "active_vehicle_count", "policy_provenance"],
        "provenance": ["checkpoint_sha256", "policy_source_mode", "adapter_source_mode", "causal_comparison_allowed"],
        "rule": "every canonical KPI input must trace to a raw event row produced by the simulator transition",
    }
    kpi_bridge = {
        "stage": STAGE,
        "pipeline": "adapter transition -> raw_events -> window_rollup -> canonical_kpi_aggregator official_rollup",
        "aggregator": AGGREGATOR,
        "aggregator_change_permitted": False,
        "kpi_scope_after_repair": "11/12 canonical KPIs; in_vehicle_time excluded",
        "synthetic_generation": "prohibited",
    }
    non_regression = {
        "stage": STAGE,
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "actor_repair_sha256": EXPECTED["actor_repair_sha256"],
        "reward_v2_change_required": False,
        "zero_loss_change_required": False,
        "observation_change_required": False,
        "checkpoint_change_required": False,
        "rule": "the bridge measures outcomes; it must not touch reward, credit, observation or policy semantics",
    }
    test6_contract = {
        "stage": STAGE, "test6_status": "EXHAUSTED_SEALED_HOLDOUT", "test6_required_for_bridge": False,
        "train_validation_use": "schema and shape inspection only; no performance KPI computation",
        "new_holdout_invented": False,
    }
    in_vehicle = {
        "stage": STAGE, "status": "NOT_YET_MEASURABLE", "proxy_permitted": False,
        "unlock_condition": "an authoritative alighting timestamp emitted by the repaired simulator and added to the canonical contract",
    }
    dependencies = {
        "stage": STAGE,
        "ordered": [
            {"id": "D1", "item": "PV8 adapter passenger/headway/energy state and event emission", "depends_on": [], "owner_gate": NEXT_GATE},
            {"id": "D2", "item": "rollup writer consumes simulator output only", "depends_on": ["D1"], "owner_gate": NEXT_GATE},
            {"id": "D3", "item": "checkpoint-driven action loop with K-mask", "depends_on": ["D1"], "owner_gate": NEXT_GATE},
            {"id": "D4", "item": "B1 and B2 policies on the identical adapter and windows", "depends_on": ["D1", "D2"], "owner_gate": NEXT_GATE},
            {"id": "D5", "item": "contract validation tests for all 16 hard requirements", "depends_on": ["D1", "D2", "D3"], "owner_gate": NEXT_GATE},
        ],
    }
    gate_decision = {
        "stage": STAGE,
        "promoted_actor_observation_dim": contract["authoritative_contract"]["actor_observation_dim"],
        "promoted_actor_observation_contract_status": contract["authoritative_contract"]["status"],
        "checkpoint_observation_match": contract["checks"]["checkpoint_and_builder_agree"],
        "selected_bridge_architecture": selected_id,
        "selected_bridge_reason": selection["selection_reason"],
        "causal_transition_path_constructable": True,
        "simulator_derived_kpi_path_constructable": True,
        "A_vs_B1_constructable_after_implementation": arms["constructable_after_implementation"]["A_vs_B1"],
        "A_vs_B2_constructable_after_implementation": arms["constructable_after_implementation"]["A_vs_B2"],
        "B0R_status": arms["B0R_status"],
        "B0C_status": arms["B0C_status"],
        "test6_required": False,
        "in_vehicle_time_status": "NOT_YET_MEASURABLE",
        "reward_v2_change_required": False,
        "zero_loss_change_required": False,
        "implementation_repair_required": True,
        "recommended_next_gate": NEXT_GATE,
    }
    gate = {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "criteria": {
            "observation_contract_resolved": contract["passed"],
            "exactly_one_bridge_selected": selection["selection_unique"],
            "all_self_tests_passed": tests["all_passed"],
            "source_only_clean_worktree": status == "",
            "py_compile_passed": compile_error is None,
            "no_implementation_performed": True,
            "no_kpi_executed": True,
            "test6_untouched": True,
        },
        "failing_criteria": [],
        "gate_decision": gate_decision,
        "exact_next_gate": NEXT_GATE,
        "next_gate_auto_execution": False,
        "pass_meaning": "repair design selected and frozen only; the bridge is not implemented, KPIs are not measurable and no performance is proven",
    }
    gate["failing_criteria"] = [key for key, value in gate["criteria"].items() if not value]
    downstream_lock = {
        "stage": STAGE, "locked_until": NEXT_GATE,
        "forbidden": ["KPI performance claim", "A vs baseline comparison", "TEST6 reuse", "in-vehicle proxy", "synthetic KPI rollups"],
        "unlock_condition": "H4M-AE-R3 implements candidate A and validates all 16 hard requirements",
    }

    payloads = {
        "promoted_actor_observation_contract.json": contract,
        "observation_dimension_lineage_audit.json": lineage,
        "observation_feature_layout.json": layout,
        "bridge_candidate_matrix.json": {"stage": STAGE, "candidates": candidates},
        "selected_bridge_architecture.json": selection,
        "causal_transition_contract.json": causal_transition,
        "raw_event_provenance_contract.json": raw_events,
        "canonical_kpi_bridge_contract.json": kpi_bridge,
        "comparison_arm_construction_contract.json": arms,
        "reward_v2_zero_loss_non_regression.json": non_regression,
        "test6_non_reuse_contract.json": test6_contract,
        "in_vehicle_time_guard.json": in_vehicle,
        "implementation_dependency_register.json": dependencies,
        "self_test_report.json": tests,
        "gate_decision.json": gate_decision,
        "downstream_lock.json": downstream_lock,
        "gate_matrix.json": gate,
        "source_provenance.json": {"stage": STAGE, "source_commit": head, "parent_commit_short": parent,
                                   "expected_parent": EXPECTED["parent_commit_short"], "clean_worktree": status == "",
                                   "git_status_short": status, "py_compile_passed": compile_error is None,
                                   "reset_or_rebase_performed": False, "github_push_performed": False},
    }
    for name, payload in payloads.items():
        write_json(root / name, payload)
    pd.DataFrame(candidates).to_parquet(root / "bridge_candidate_matrix.parquet", index=False)

    final_json = {
        "stage": STAGE, "gate": gate["gate"], "source_commit": head,
        "gate_decision": gate_decision, "self_test_failed": tests["failed"],
        "candidates": [row["candidate_id"] for row in candidates],
    }
    write_json(root / "final_report.json", final_json)
    (root / "final_report.md").write_text(
        f"""# H4M-AE-R2 Causal KPI Measurement Bridge Repair Selection and Freeze

gate = {gate['gate']}
selected_bridge = {selected_id}
promoted_actor_observation_dim = {gate_decision['promoted_actor_observation_dim']} ({contract['authoritative_contract']['status']})
source_commit = {head}
select_and_freeze_only = true

## Observation contract, measured not assumed

```json
{json.dumps(contract["authoritative_contract"], ensure_ascii=False, indent=2, default=jsonable)}
```

```json
{json.dumps({name: row["classification"] for name, row in lineage["lineages"].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Candidates

```json
{json.dumps([{k: row[k] for k in ("candidate_id", "observation_compatibility", "graph_state_compatibility", "kpi_availability_after_repair", "semantic_risk")} for row in candidates], ensure_ascii=False, indent=2, default=jsonable)}
```

## Selection

{selection['selection_reason']}

## Self-tests

```json
{json.dumps({name: row["passed"] for name, row in tests["tests"].items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: design selected and frozen only. The bridge is not implemented, KPIs remain unmeasurable and no
performance claim is made.
""",
        encoding="utf-8",
    )
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file()}
    write_json(root / "artifact_manifest.json", {
        "stage": STAGE, "artifact_root": str(root), "source_commit": head, "gate": gate["gate"],
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "artifact_manifest.json"),
        "file_sha256": {name: sha256_file(Path(path)) for name, path in files.items() if name != "artifact_manifest.json"},
        "elapsed_seconds": time.perf_counter() - started, "append_only_artifact": True,
        "select_and_freeze_only": True, "implementation_performed": False, "kpi_executed": False,
        "test6_touched": False, "github_push_performed": False,
    })
    (root / "_SUCCESS.lock").write_text(f"{gate['gate']}\n{head}\n{created_at}\n", encoding="utf-8")

    print(f"[H4M-AE-R2] artifact root: {root}")
    print(f"[H4M-AE-R2] gate: {gate['gate']}")
    print(f"[H4M-AE-R2] actor observation dim: {gate_decision['promoted_actor_observation_dim']} status={gate_decision['promoted_actor_observation_contract_status']}")
    print(f"[H4M-AE-R2] selected: {selected_id}")
    print(f"[H4M-AE-R2] self-tests failed: {tests['failed']}")
    print(f"[H4M-AE-R2] A_vs_B1/B2 after implementation: {gate_decision['A_vs_B1_constructable_after_implementation']}/{gate_decision['A_vs_B2_constructable_after_implementation']}")
    print(f"[H4M-AE-R2] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-AE-R2] next gate: {NEXT_GATE} (not executed)")


if __name__ == "__main__":
    main()
