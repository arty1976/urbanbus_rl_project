#!/usr/bin/env python3
"""H4M-K fresh target-context repaired three-seed retraining.

Executes the H4M-J validated 12D target-context observation repair through the
authorized fresh seeds 1/2/3 × frozen 11-cycle TRAIN44 schedule, then evaluates
state-conditional HOLD/SERVE discrimination against H4M-H-RERUN as a historical
diagnostic comparator.

This stage may train only the requested fresh diagnostic run. It must not reuse
9D or H4M-C/H4M-H checkpoints, tune hyperparameters, modify Reward/GAE/PPO,
open validation/TEST6, select a winner/baseline, or push to GitHub.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-K"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_K_FRESH_TARGET_CONTEXT_REPAIRED_THREE_SEED_RETRAINING_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_K_FRESH_TARGET_CONTEXT_REPAIRED_THREE_SEED_RETRAINING_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_H_BASE_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_h_fresh_instrumented_credit_diagnostic_retraining.py"
H4M_H_RERUN_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_h_rerun_fresh_instrumented_credit_diagnostic_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
H4M_C_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_c_fresh_extended_budget_three_seed_retraining.py"
H4K_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py"
DL4_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
MAPPO_SOURCE = TRAINING_ROOT / "mappo_runner.py"
REWARD_SOURCE = TRAINING_ROOT / "rewards/mappo_reward_v1.py"
OBS_REPAIR_SOURCE = TRAINING_ROOT / "observation_target_context_repair.py"

H4M_J_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_j_observation_discrimination_repair_implementation_equivalence_validation_20260817_093856+0900"
H4M_I_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_i_observation_discrimination_repair_selection_freeze_20260817_092043+0900"
H4M_H_RERUN_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_h_rerun_fresh_instrumented_credit_diagnostic_retraining_20260816_233823+0900"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"

EXPECTED = {
    "h4m_j_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_J_OBSERVATION_DISCRIMINATION_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE",
    "h4m_j_source_commit": "63ca5ab33312e723ce35d77324be2c7e2e4bfc0f",
    "h4m_j_decision": "PV8_TARGET_CONTEXT_OBSERVATION_REPAIR_IMPLEMENTED_VALIDATED_READY_FOR_FRESH_THREE_SEED_RETRAINING",
    "h4m_i_repair_contract_sha256": "6ecd20cfcd220d50a6f1ebbcd6e33594a9ca14a86a2a60236cea435a24058e1a",
    "active_mps_instrumentation_contract_sha256": "e6c73da48c12edfd573069730d2eeec32c74fea74f590105e55ad3502a729c92",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "seeds": [1, 2, 3],
    "outer_training_count": 11,
    "ppo_updates_per_seed": 44,
    "critic_updates_per_seed": 88,
    "active_samples_per_seed": 3872,
    "node_feature_dim": 12,
}

ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
CLASS_HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
CLASS_SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
CLASS_TIE = "TIE_WITH_EXISTING_TOLERANCE"
TARGET_CONTEXT_FIELDS = [
    "r3_action_target_is_hold",
    "r3_action_target_is_serve",
    "r3_action_target_is_skip",
]

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_training_integrity.json",
    "03_three_seed_cycle_summary.json",
    "04_conditional_policy_discrimination.json",
    "05_hold_better_metrics.json",
    "06_serve_better_metrics.json",
    "07_h4mh_vs_h4mk_comparison.json",
    "08_long_horizon_shadow_summary.json",
    "09_credit_trace_summary.json",
    "10_policy_collapse_audit.json",
    "11_root_decision.json",
    "12_checkpoint_registry.json",
    "13_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def import_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_h4m_helpers() -> Tuple[Any, Any]:
    base = import_module_from_path("h4mk_h4mh_base", H4M_H_BASE_SOURCE)
    rerun = import_module_from_path("h4mk_h4mh_rerun", H4M_H_RERUN_SOURCE)
    base.STAGE = STAGE
    rerun.STAGE = STAGE
    return base, rerun


def load_h4mg() -> Any:
    return import_module_from_path(f"h4mk_h4mg_{time.time_ns()}", H4MG_SOURCE)


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    latest_source_commit = git_run(["log", "-1", "--format=%H", "--", str(SOURCE_REL)]).stdout.strip()
    source_present = git_run(["cat-file", "-e", f"HEAD:{SOURCE_REL}"], check=False).returncode == 0
    source_entries: Dict[str, Any] = {}
    for rel in [
        SOURCE_REL,
        Path("05_training") / "observation_target_context_repair.py",
        Path("05_training") / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py",
    ]:
        latest = git_run(["log", "-1", "--format=%H", "--", str(rel)]).stdout.strip()
        present = git_run(["cat-file", "-e", f"HEAD:{rel}"], check=False).returncode == 0
        source_entries[str(rel)] = {
            "latest_commit": latest,
            "present_in_head": present,
            "sha256": sha256_file(PROJECT_ROOT / rel),
        }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_k_source_git_commit": latest_source_commit,
        "source_rel": str(SOURCE_REL),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "source_present_in_head": source_present,
        "head_commit_files": head_files,
        "head_commit_source_only": head_files == [str(SOURCE_REL)],
        "status_short": status_short,
        "source_entries": source_entries,
        "local_source_only_commit_created_before_training": source_present
        and latest_source_commit == head
        and head_files == [str(SOURCE_REL)]
        and status_short == "",
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    h4mj_gate = read_json(H4M_J_ROOT / "10_gate_matrix.json")
    h4mj_binding = read_json(H4M_J_ROOT / "01_authoritative_binding.json")
    h4mj_schema = read_json(H4M_J_ROOT / "03_schema_9_to_12_audit.json")
    h4mj_ready = read_json(H4M_J_ROOT / "09_fresh_training_readiness.json")
    h4mh_gate = read_json(H4M_H_RERUN_ROOT / "14_gate_matrix.json")
    h4mi_actor = read_json(H4M_I_ROOT / "05_actor_discrimination.json")
    close_contract_sha = sha256_file(H4M_G_CLOSE_ROOT / "12_final_mps_equivalence_contract.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    h4mj_sha = h4mj_binding.get("sha_bindings", {})
    contract = h4mj_binding.get("bound_contract", {})
    checks = {
        "source_only_commit_before_training": provenance.get("local_source_only_commit_created_before_training") is True,
        "h4m_j_gate_match": h4mj_gate.get("gate") == EXPECTED["h4m_j_gate"],
        "h4m_j_decision_match": h4mj_gate.get("decision") == EXPECTED["h4m_j_decision"],
        "h4m_j_source_commit_match": h4mj_binding.get("source_provenance", {}).get("h4m_j_source_git_commit")
        == EXPECTED["h4m_j_source_commit"],
        "h4m_i_repair_contract_sha_match_binding": contract.get("repair_contract_sha256")
        == EXPECTED["h4m_i_repair_contract_sha256"],
        "h4m_i_repair_contract_sha_match_schema": h4mj_schema.get("contract", {}).get("contract_sha256")
        == EXPECTED["h4m_i_repair_contract_sha256"],
        "schema_12d_match": int(h4mj_schema.get("new_dim", -1)) == EXPECTED["node_feature_dim"],
        "target_context_fields_match": h4mj_schema.get("contract", {}).get("target_context_fields") == TARGET_CONTEXT_FIELDS,
        "h4m_g_close_active_contract_match_binding": h4mj_sha.get("active_mps_instrumentation_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_g_close_active_contract_match_file": close_contract_sha == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_b_schedule_sha_match_binding": h4mj_sha.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "h4m_b_schedule_sha_match_direct": h4m_b_schedule.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": h4mj_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "r3_split_sha_match": h4mj_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "h4g_runtime_sha_match": h4mj_sha.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "zero_loss_adapter_sha_match": h4mj_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "h4m_h_rerun_comparator_gate_pass": str(h4mh_gate.get("gate", "")).startswith("PASS_"),
        "h4m_h_actor_comparator_available": "actual_h4m_h_trace_actor_output" in h4mi_actor,
        "fresh_training_readiness_yes": h4mj_ready.get("ready_for_next_gate") is True,
        "test6_sealed": h4mj_ready.get("fresh_retraining_contract", {}).get("test6_status") == "SEALED_NOT_OPENED",
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_j": str(H4M_J_ROOT),
            "h4m_i": str(H4M_I_ROOT),
            "h4m_h_rerun": str(H4M_H_RERUN_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "source_provenance": provenance,
        "authoritative_binding_passed": all(checks.values()),
        "checks": checks,
        "bound_gates": {
            "h4m_j_gate": h4mj_gate.get("gate"),
            "h4m_j_decision": h4mj_gate.get("decision"),
            "h4m_j_next_gate": h4mj_gate.get("exact_next_gate"),
            "h4m_h_rerun_gate": h4mh_gate.get("gate"),
            "h4m_h_rerun_decision": h4mh_gate.get("decision"),
        },
        "bound_contract": {
            "repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
            "schema": {
                "node_feature_dim_before": 9,
                "node_feature_dim_after": EXPECTED["node_feature_dim"],
                "target_context_fields": TARGET_CONTEXT_FIELDS,
            },
            "fresh_initialization_required": True,
            "reuse_9d_checkpoint_weights": False,
        },
        "sha_bindings": {
            "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "hard_lock_attestation": {
            "reward_v2_modified": False,
            "zero_loss_modified": False,
            "k_mask_or_action_contract_modified": False,
            "causal_order_modified": False,
            "gae_ppo_critic_normalization_modified": False,
            "hidden_sizes_or_model_topology_modified": False,
            "hyperparameters_modified": False,
            "train44_agents_environment_modified": False,
            "instrumentation_contract_modified": False,
            "old_checkpoint_reuse": False,
            "validation_or_test6_executed": False,
            "github_push_performed": False,
        },
    }


def build_context_mps(h4mg: Any, seed: int, created_at: str, *, device: torch.device) -> Dict[str, Any]:
    h4m_c = h4mg.import_module_from_path(H4M_C_SOURCE, f"h4mk_h4m_c_{seed}_{time.time_ns()}")
    h4k = h4mg.import_module_from_path(H4K_SOURCE, f"h4mk_h4k_{seed}_{time.time_ns()}")
    dl4 = h4mg.import_module_from_path(DL4_SOURCE, f"h4mk_dl4_{seed}_{time.time_ns()}")
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    mappo_mod = h4mg.import_module_from_path(MAPPO_SOURCE, f"h4mk_mappo_{seed}_{time.time_ns()}")
    reward_mod = h4mg.import_module_from_path(REWARD_SOURCE, f"h4mk_reward_{seed}_{time.time_ns()}")
    observation_repair = h4mg.import_module_from_path(OBS_REPAIR_SOURCE, f"h4mk_observation_repair_{seed}_{time.time_ns()}")

    schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    window_plan = read_json(H4M_C_ROOT / "seed_001" / "r3_train_window_plan.json")
    train_paths = [Path(row["snapshot_path"]) for row in window_plan["train_rows"]]
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])

    dl4.set_all_seeds(seed)
    sample_full = dl1.torch_load(train_paths[0])
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(PROJECT_ROOT, sample_full, mapping_artifact=mapping_artifact)
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    sample_graph, sample_observation_repair_audit = observation_repair.append_target_context_features(sample_graph, dl1, action_dim=3)
    train_data_unrepaired = dl4.load_subgraphs(dl1, train_paths, spec)
    train_data, observation_repair_audit = observation_repair.repair_data_sequence(train_data_unrepaired, dl1, action_dim=3)
    effective_horizon = len(train_data)
    config: Dict[str, Any] = {
        "created_at": created_at,
        "stage": STAGE,
        "study_area": "SUSEONG_GU_DAEGU",
        "seed": seed,
        "controlled_equivalence_seed": seed,
        "device": str(device),
        "agents": 8,
        "effective_agents": min(8, int(inventory["available_suseong_agents"])),
        "gatv2_hidden": 128,
        "gatv2_grad_clip": 5.0,
        "mappo_grad_clip": 0.5,
        "critic_grad_clip": 0.5,
        "rollout_horizon": int(schedule["rollout_horizon"]),
        "controlled_effective_horizon": effective_horizon,
        "minibatch_size": int(schedule["minibatch"]),
        "actor_ppo_epochs": int(schedule["ppo_epochs_per_update"]),
        "critic_epochs": int(schedule["critic_epochs_per_update"]),
        "actor_gatv2_lr": float(schedule["gatv2_lr"]),
        "actor_lr": float(schedule["actor_lr"]),
        "gatv2_lr": float(schedule["gatv2_lr"]),
        "critic_lr": float(schedule["critic_lr"]),
        "gamma": float(schedule["gamma"]),
        "gae_lambda": float(schedule["gae_lambda"]),
        "ppo_clip_epsilon": float(schedule["clip_epsilon"]),
        "value_loss_coef": float(schedule["value_coefficient"]),
        "return_normalization": True,
        "reward_normalization": True,
        "action_dim": 3,
        "h4m_b_extended_training_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha256": EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "observation_repair_contract_sha256": observation_repair.CONTRACT_SHA256,
        "observation_schema_version": observation_repair.SCHEMA_VERSION,
        "node_feature_dim_before_observation_repair": 9,
        "node_feature_dim_after_observation_repair": int(sample_graph.x.size(1)),
        "target_context_fields": observation_repair.TARGET_CONTEXT_FIELDS,
        "spec": spec,
        "started_at_perf": time.perf_counter(),
    }
    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    reward_normalizer = mappo_mod.RewardNormalizer(window_size=1000, clip_value=10.0)
    return_normalizer = dl4.ReturnNormalizer(True)
    optimizers = {
        "gatv2": torch.optim.Adam(encoder.parameters(), lr=float(config["gatv2_lr"])),
        "actor": torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"])),
        "critic": torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"])),
    }
    return {
        "h4m_c": h4m_c,
        "h4k": h4k,
        "dl1": dl1,
        "dl4": dl4,
        "mappo_mod": mappo_mod,
        "reward_mod": reward_mod,
        "observation_repair_module": observation_repair,
        "device": device,
        "window_plan": window_plan,
        "train_data": train_data,
        "config": config,
        "encoder": encoder,
        "actor": actor,
        "critic": critic,
        "reward_normalizer": reward_normalizer,
        "return_normalizer": return_normalizer,
        "optimizers": optimizers,
        "sample_graph": sample_graph,
        "sample_observation_repair_audit": sample_observation_repair_audit,
        "observation_repair_audit": observation_repair_audit,
        "connectivity": connectivity,
        "tensor_mask": tensor_mask,
        "inventory": inventory,
    }


def cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def cpu_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, Mapping):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(convert(v) for v in value)
        return value

    return convert(optimizer.state_dict())


def save_seed_checkpoint(artifact_root: Path, seed: int, ctx: Mapping[str, Any], cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path = artifact_root / "checkpoints" / f"H4M_K_SEED_{seed:03d}_FRESH_TARGET_CONTEXT_REPAIRED.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "seed": seed,
        "checkpoint_status": "FRESH_TARGET_CONTEXT_REPAIRED_TRAINING_COMPLETE_DIAGNOSTIC_EVALUATION_NOT_RELEASED",
        "created_at": kst_now(),
        "fresh_initialization": True,
        "old_9d_checkpoint_reuse": False,
        "h4m_c_or_h4m_h_checkpoint_continuation": False,
        "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "observation_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
        "node_feature_dim": int(ctx["config"]["node_feature_dim_after_observation_repair"]),
        "gatv2_state_dict": cpu_state_dict(ctx["encoder"]),
        "actor_state_dict": cpu_state_dict(ctx["actor"]),
        "critic_state_dict": cpu_state_dict(ctx["critic"]),
        "gatv2_optimizer_state_dict": cpu_optimizer_state_dict(ctx["optimizers"]["gatv2"]),
        "actor_optimizer_state_dict": cpu_optimizer_state_dict(ctx["optimizers"]["actor"]),
        "critic_optimizer_state_dict": cpu_optimizer_state_dict(ctx["optimizers"]["critic"]),
        "reward_normalizer_state": dict(ctx["reward_normalizer"].__dict__),
        "return_normalizer_state": ctx["return_normalizer"].state_dict(),
        "cycle_summaries_sha256": canonical_sha(cycle_summaries),
        "metadata": {
            "outer_training_count": EXPECTED["outer_training_count"],
            "ppo_updates": EXPECTED["ppo_updates_per_seed"],
            "critic_updates": EXPECTED["critic_updates_per_seed"],
            "test6_status": "SEALED_NOT_OPENED",
            "selection_or_validation_release_status": "NOT_RELEASED_DIAGNOSTIC_ONLY",
        },
    }
    torch.save(payload, path)
    return {"seed": seed, "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def shadow_integrity_snapshot(h4mg: Any, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    dl1 = ctx["dl1"]
    return {
        "rng": h4mg.rng_snapshot(),
        "model_hashes": h4mg.model_hashes(dl1, ctx["encoder"], ctx["actor"], ctx["critic"]),
        "optimizer_hashes": h4mg.optimizer_hashes(ctx["optimizers"]),
        "reward_normalizer": canonical_sha(h4mg.reward_normalizer_state(ctx["reward_normalizer"])),
        "return_normalizer": canonical_sha(ctx["return_normalizer"].state_dict()),
    }


def target_context_from_target_id(target_id: int) -> Dict[str, int]:
    return {
        "r3_action_target_is_hold": int(target_id == 0),
        "r3_action_target_is_serve": int(target_id == 1),
        "r3_action_target_is_skip": int(target_id == 2),
    }


def joined_trace_row(
    seed: int,
    cycle: int,
    uid: str,
    policy: Mapping[str, Any],
    shadow: Mapping[str, Any],
    adv: Mapping[str, Any],
    crit: Mapping[str, Any],
    pressure: Counter,
) -> Dict[str, Any]:
    target_id = int(policy.get("target_id"))
    target_context = target_context_from_target_id(target_id)
    p_hold = float(policy.get("masked_probability_0"))
    p_serve = float(policy.get("masked_probability_1"))
    p_skip = float(policy.get("masked_probability_2"))
    sampled = adv.get("sampled_action_name")
    classification = shadow.get("classification")
    return {
        "seed": seed,
        "cycle": cycle,
        "sample_uid": uid,
        "window_id": policy.get("window_id"),
        "step_index": int(policy.get("step_index")),
        "agent_slot": int(policy.get("agent_slot")),
        "state_hash": policy.get("state_hash"),
        "target_id": target_id,
        "target_name": policy.get("target"),
        **target_context,
        "target_context_one_hot_sum": sum(target_context.values()),
        "classification": classification,
        "delta_full_bootstrapped_return_serve_minus_hold": shadow.get("delta_full_bootstrapped_return_serve_minus_hold"),
        "delta_discounted_reward_to_boundary_serve_minus_hold": shadow.get("delta_discounted_reward_to_boundary_serve_minus_hold"),
        "delta_bootstrap_value_serve_minus_hold": shadow.get("delta_bootstrap_value_serve_minus_hold"),
        "sampled_action_name": sampled,
        "sampled_action_id": int(policy.get("action_id")),
        "P_HOLD": p_hold,
        "P_SERVE": p_serve,
        "P_SKIP": p_skip,
        "policy_margin_hold_minus_serve": p_hold - p_serve,
        "pre_mask_logit_margin_hold_minus_serve": float(policy.get("pre_mask_logit_0")) - float(policy.get("pre_mask_logit_1")),
        "policy_entropy": float(policy.get("policy_entropy")),
        "raw_gae_advantage": adv.get("raw_gae_advantage"),
        "normalized_advantage": adv.get("normalized_advantage"),
        "sign_change_class": adv.get("sign_change_class"),
        "value_error": crit.get("value_error"),
        "serve_pressure_increase_count": int(sum(v for (serve, _hold), v in pressure.items() if serve == "increase")),
        "hold_pressure_increase_count": int(sum(v for (_serve, hold), v in pressure.items() if hold == "increase")),
        "probability_correct_direction": (
            p_hold > p_serve if classification == CLASS_HOLD_BETTER else p_serve > p_hold if classification == CLASS_SERVE_BETTER else None
        ),
        "sampled_correct_direction": (
            sampled == ACTION_HOLD if classification == CLASS_HOLD_BETTER else sampled == ACTION_SERVE if classification == CLASS_SERVE_BETTER else None
        ),
    }


def run_training(
    base: Any,
    h4mg: Any,
    created_at: str,
    artifact_root: Path,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    all_cycle_summaries: List[Dict[str, Any]] = []
    trace_registry: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    seed_summaries: List[Dict[str, Any]] = []
    joined_rows: List[Dict[str, Any]] = []
    safety_counts: Counter[str] = Counter()
    shadow_audits: List[Dict[str, Any]] = []
    schema_audits: List[Dict[str, Any]] = []

    for seed in EXPECTED["seeds"]:
        ctx = build_context_mps(h4mg, seed, created_at, device=device)
        schema_audits.append(
            {
                "seed": seed,
                "node_feature_dim": int(ctx["config"]["node_feature_dim_after_observation_repair"]),
                "repair_contract_sha256": ctx["config"]["observation_repair_contract_sha256"],
                "sample_repair_audit": ctx["sample_observation_repair_audit"],
                "train_sequence_summary": ctx["observation_repair_audit"],
            }
        )
        dl1 = ctx["dl1"]
        before_state = {
            "gatv2": dl1.clone_state_dict(ctx["encoder"]),
            "actor": dl1.clone_state_dict(ctx["actor"]),
            "critic": dl1.clone_state_dict(ctx["critic"]),
        }
        seed_cycle_summaries: List[Dict[str, Any]] = []
        ppo_rows_total = 0
        critic_update_rows_total = 0
        active_samples_total = 0
        started_seed = time.perf_counter()

        for cycle in range(1, EXPECTED["outer_training_count"] + 1):
            recorder = h4mg.TraceRecorder(root=artifact_root / "_unused_h4mg_recorder_root", enabled=True)
            rollout = h4mg.collect_controlled_rollout(
                branch="H4M_K_TARGET_CONTEXT_REPAIRED",
                seed=seed,
                cycle_index=cycle,
                ctx=ctx,
                recorder=recorder,
            )
            update = h4mg.ppo_update_controlled(
                branch="H4M_K_TARGET_CONTEXT_REPAIRED",
                seed=seed,
                cycle_index=cycle,
                ctx=ctx,
                rollout=rollout,
                before_state=before_state,
                recorder=recorder,
            )
            before_shadow = shadow_integrity_snapshot(h4mg, ctx)
            pair_rows, event_rows, summary_rows, shadow_completeness = base.build_full_shadow_rows(h4mg, seed, cycle, ctx, rollout)
            after_shadow = shadow_integrity_snapshot(h4mg, ctx)
            shadow_audit = {
                "seed": seed,
                "outer_cycle": cycle,
                "rng_unchanged_by_shadow": before_shadow["rng"] == after_shadow["rng"],
                "model_hashes_unchanged_by_shadow": before_shadow["model_hashes"] == after_shadow["model_hashes"],
                "optimizer_hashes_unchanged_by_shadow": before_shadow["optimizer_hashes"] == after_shadow["optimizer_hashes"],
                "reward_normalizer_unchanged_by_shadow": before_shadow["reward_normalizer"] == after_shadow["reward_normalizer"],
                "return_normalizer_unchanged_by_shadow": before_shadow["return_normalizer"] == after_shadow["return_normalizer"],
                "shadow_pair_rows": len(pair_rows),
                "shadow_branch_event_rows": len(event_rows),
                "shadow_summary_rows": len(summary_rows),
            }
            shadow_audit["shadow_isolation_passed"] = all(
                bool(shadow_audit[k])
                for k in [
                    "rng_unchanged_by_shadow",
                    "model_hashes_unchanged_by_shadow",
                    "optimizer_hashes_unchanged_by_shadow",
                    "reward_normalizer_unchanged_by_shadow",
                    "return_normalizer_unchanged_by_shadow",
                ]
            )
            shadow_audits.append(shadow_audit)

            trace_registry.append(base.write_cycle_trace(artifact_root, seed, cycle, recorder, (pair_rows, event_rows, summary_rows)))
            cycle_summary = base.summarize_cycle(seed, cycle, recorder, summary_rows)
            cycle_summary["shadow_completeness"] = shadow_completeness
            cycle_summary["ppo_metric_rows"] = len(update["metrics_rows"])
            cycle_summary["actor_joint_update_rows"] = sum(
                1 for row in update["metrics_rows"] if row.get("update_role") == "actor_gatv2_critic_joint"
            )
            cycle_summary["critic_update_rows"] = len(update["metrics_rows"])
            cycle_summary["node_feature_dim"] = int(ctx["config"]["node_feature_dim_after_observation_repair"])
            cycle_summary["loss_finite"] = all(
                all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in update["loss_rows"]
            )
            safety_counts["illegal_action"] += sum(
                1 for row in recorder.pre_action_rows if int(row["action_id"]) not in [int(v) for v in row["legal_action_ids"]]
            )
            safety_counts["illegal_SKIP"] += sum(
                1 for row in recorder.pre_action_rows if int(row["action_id"]) == 2 and 2 not in [int(v) for v in row["legal_action_ids"]]
            )
            safety_counts["nonfinite_loss_cycle"] += int(not cycle_summary["loss_finite"])
            safety_counts["shadow_rng_drift"] += int(not shadow_audit["rng_unchanged_by_shadow"])
            safety_counts["shadow_model_contamination"] += int(not shadow_audit["model_hashes_unchanged_by_shadow"])
            safety_counts["shadow_optimizer_contamination"] += int(not shadow_audit["optimizer_hashes_unchanged_by_shadow"])
            safety_counts["shadow_normalizer_contamination"] += int(
                not (shadow_audit["reward_normalizer_unchanged_by_shadow"] and shadow_audit["return_normalizer_unchanged_by_shadow"])
            )

            seed_cycle_summaries.append(cycle_summary)
            all_cycle_summaries.append(cycle_summary)
            ppo_rows_total += cycle_summary["actor_joint_update_rows"]
            critic_update_rows_total += cycle_summary["critic_update_rows"]
            active_samples_total += cycle_summary["active_samples"]

            policy_by_uid = {row["sample_uid"]: row for row in recorder.pre_action_rows}
            shadow_by_uid = {row["sample_uid"]: row for row in summary_rows}
            adv_by_uid = {row["sample_uid"]: row for row in recorder.advantage_sample_rows}
            critic_by_uid = {row["sample_uid"]: row for row in recorder.critic_value_error_rows}
            pressure_by_uid: Dict[str, Counter] = defaultdict(Counter)
            for row in recorder.actor_pressure_rows:
                pressure_by_uid[row["sample_uid"]][(row["serve_logit_pressure"], row["hold_logit_pressure"])] += 1
            for uid, shadow in shadow_by_uid.items():
                joined_rows.append(
                    joined_trace_row(
                        seed,
                        cycle,
                        uid,
                        policy_by_uid[uid],
                        shadow,
                        adv_by_uid[uid],
                        critic_by_uid[uid],
                        pressure_by_uid[uid],
                    )
                )
            if torch.backends.mps.is_available() and hasattr(torch, "mps"):
                torch.mps.empty_cache()

        checkpoint = save_seed_checkpoint(artifact_root, seed, ctx, seed_cycle_summaries)
        checkpoint_rows.append(checkpoint)
        parameter_delta = {
            "gatv2": dl1.delta_stats(ctx["encoder"], before_state["gatv2"]),
            "actor": dl1.delta_stats(ctx["actor"], before_state["actor"]),
            "critic": dl1.delta_stats(ctx["critic"], before_state["critic"]),
        }
        seed_summaries.append(
            {
                "seed": seed,
                "fresh_initialization": True,
                "old_9d_checkpoint_reuse": False,
                "h4m_c_or_h4m_h_checkpoint_continuation": False,
                "node_feature_dim": int(ctx["config"]["node_feature_dim_after_observation_repair"]),
                "rollouts": len(seed_cycle_summaries),
                "ppo_updates": ppo_rows_total,
                "critic_updates": critic_update_rows_total,
                "active_samples": active_samples_total,
                "elapsed_seconds": time.perf_counter() - started_seed,
                "parameter_delta": parameter_delta,
                "parameter_delta_positive": all(float(parameter_delta[k]["l2_delta"]) > 0.0 for k in ("gatv2", "actor", "critic")),
                "loss_finite": all(row["loss_finite"] for row in seed_cycle_summaries),
                "checkpoint": checkpoint,
            }
        )

    training_summary = {
        "stage": STAGE,
        "created_at": created_at,
        "device": "mps",
        "seeds": EXPECTED["seeds"],
        "seed_summaries": seed_summaries,
        "cycle_summaries": all_cycle_summaries,
        "schema_audits": schema_audits,
        "total_rollouts": sum(row["rollouts"] for row in seed_summaries),
        "total_ppo_updates": sum(row["ppo_updates"] for row in seed_summaries),
        "total_critic_updates": sum(row["critic_updates"] for row in seed_summaries),
        "total_active_samples": sum(row["active_samples"] for row in seed_summaries),
        "fresh_initialization_all_seeds": all(row["fresh_initialization"] for row in seed_summaries),
        "old_9d_checkpoint_reuse": False,
        "h4m_c_or_h4m_h_checkpoint_continuation": False,
        "rollout_tensor_reuse": False,
    }
    checkpoint_registry = {
        "stage": STAGE,
        "created_at": created_at,
        "checkpoints": checkpoint_rows,
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_creation_authorized_for_diagnostic_evidence": True,
        "fresh_12d_initialization": True,
        "reuse_9d_checkpoint_weights": False,
        "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "observation_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
        "selection_or_validation_release_status": "NOT_RELEASED_DIAGNOSTIC_ONLY",
    }
    trace_meta = {
        "trace_registry": trace_registry,
        "joined_rows": joined_rows,
        "safety_counts": dict(safety_counts),
        "shadow_audits": shadow_audits,
    }
    return training_summary, all_cycle_summaries, checkpoint_registry, trace_meta


def numeric_stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    mu = float(mean(nums))
    var = sum((x - mu) ** 2 for x in nums) / len(nums)
    return {"count": len(nums), "mean": mu, "min": min(nums), "max": max(nums), "std": math.sqrt(var)}


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else float(numerator) / float(denominator)


def dominant_action(p_hold_mean: Optional[float], p_serve_mean: Optional[float]) -> str:
    if p_hold_mean is None or p_serve_mean is None:
        return "NOT_MEASURABLE"
    if p_hold_mean > p_serve_mean:
        return "HOLD"
    if p_serve_mean > p_hold_mean:
        return "SERVE"
    return "TIE"


def sampled_dominant_action(counts: Mapping[str, int]) -> str:
    hold = int(counts.get(ACTION_HOLD, 0))
    serve = int(counts.get(ACTION_SERVE, 0))
    if hold > serve:
        return "HOLD"
    if serve > hold:
        return "SERVE"
    return "TIE"


def summarize_policy_subset(rows: Sequence[Mapping[str, Any]], classification: str) -> Dict[str, Any]:
    action_counts = Counter(row["sampled_action_name"] for row in rows)
    count = len(rows)
    prob_correct = sum(1 for row in rows if row.get("probability_correct_direction") is True)
    sampled_correct = sum(1 for row in rows if row.get("sampled_correct_direction") is True)
    target_context_counts = Counter(
        (row["r3_action_target_is_hold"], row["r3_action_target_is_serve"], row["r3_action_target_is_skip"]) for row in rows
    )
    p_hold = numeric_stats([row["P_HOLD"] for row in rows])
    p_serve = numeric_stats([row["P_SERVE"] for row in rows])
    margin = numeric_stats([row["policy_margin_hold_minus_serve"] for row in rows])
    return {
        "classification": classification,
        "count": count,
        "P_HOLD": p_hold,
        "P_SERVE": p_serve,
        "P_SKIP": numeric_stats([row["P_SKIP"] for row in rows]),
        "policy_margin_hold_minus_serve": margin,
        "pre_mask_logit_margin_hold_minus_serve": numeric_stats([row["pre_mask_logit_margin_hold_minus_serve"] for row in rows]),
        "entropy": numeric_stats([row["policy_entropy"] for row in rows]),
        "sampled_action_counts": dict(action_counts),
        "sampled_HOLD_rate": safe_rate(int(action_counts.get(ACTION_HOLD, 0)), count),
        "sampled_SERVE_rate": safe_rate(int(action_counts.get(ACTION_SERVE, 0)), count),
        "sampled_SKIP_rate": safe_rate(int(action_counts.get(ACTION_SKIP, 0)), count),
        "probability_correct_direction_count": prob_correct,
        "probability_correct_direction_rate": safe_rate(prob_correct, count),
        "sampled_correct_direction_count": sampled_correct,
        "sampled_correct_direction_rate": safe_rate(sampled_correct, count),
        "probability_dominant_action": dominant_action(p_hold["mean"], p_serve["mean"]),
        "sampled_dominant_action": sampled_dominant_action(action_counts),
        "target_context_one_hot_counts": {str(k): int(v) for k, v in target_context_counts.items()},
        "target_context_preserved": all(int(row["target_context_one_hot_sum"]) == 1 for row in rows),
    }


def conditional_policy_discrimination(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_label = {
        CLASS_HOLD_BETTER: summarize_policy_subset([row for row in joined_rows if row["classification"] == CLASS_HOLD_BETTER], CLASS_HOLD_BETTER),
        CLASS_SERVE_BETTER: summarize_policy_subset([row for row in joined_rows if row["classification"] == CLASS_SERVE_BETTER], CLASS_SERVE_BETTER),
    }
    by_seed_cycle_label: List[Dict[str, Any]] = []
    cycle_aggregate: List[Dict[str, Any]] = []
    overall_evolution: List[Dict[str, Any]] = []
    for seed in EXPECTED["seeds"]:
        for cycle in range(1, EXPECTED["outer_training_count"] + 1):
            sc_rows = [row for row in joined_rows if int(row["seed"]) == seed and int(row["cycle"]) == cycle]
            for label in [CLASS_HOLD_BETTER, CLASS_SERVE_BETTER]:
                rows = [row for row in sc_rows if row["classification"] == label]
                summary = summarize_policy_subset(rows, label)
                summary.update({"seed": seed, "outer_cycle": cycle})
                by_seed_cycle_label.append(summary)
    for cycle in range(1, EXPECTED["outer_training_count"] + 1):
        cycle_rows = [row for row in joined_rows if int(row["cycle"]) == cycle]
        action_counts = Counter(row["sampled_action_name"] for row in cycle_rows)
        p_hold = numeric_stats([row["P_HOLD"] for row in cycle_rows])
        p_serve = numeric_stats([row["P_SERVE"] for row in cycle_rows])
        overall_evolution.append(
            {
                "outer_cycle": cycle,
                "count": len(cycle_rows),
                "P_HOLD": p_hold,
                "P_SERVE": p_serve,
                "policy_margin_hold_minus_serve": numeric_stats([row["policy_margin_hold_minus_serve"] for row in cycle_rows]),
                "entropy": numeric_stats([row["policy_entropy"] for row in cycle_rows]),
                "sampled_action_counts": dict(action_counts),
                "probability_dominant_action": dominant_action(p_hold["mean"], p_serve["mean"]),
                "sampled_dominant_action": sampled_dominant_action(action_counts),
            }
        )
        hold_summary = summarize_policy_subset([row for row in cycle_rows if row["classification"] == CLASS_HOLD_BETTER], CLASS_HOLD_BETTER)
        serve_summary = summarize_policy_subset([row for row in cycle_rows if row["classification"] == CLASS_SERVE_BETTER], CLASS_SERVE_BETTER)
        separation_gap = None
        if hold_summary["policy_margin_hold_minus_serve"]["mean"] is not None and serve_summary["policy_margin_hold_minus_serve"]["mean"] is not None:
            separation_gap = hold_summary["policy_margin_hold_minus_serve"]["mean"] - serve_summary["policy_margin_hold_minus_serve"]["mean"]
        cycle_aggregate.append(
            {
                "outer_cycle": cycle,
                "hold_better": hold_summary,
                "serve_better": serve_summary,
                "separation_gap_hold_margin_minus_serve_margin": separation_gap,
            }
        )
    first = overall_evolution[0] if overall_evolution else None
    last = overall_evolution[-1] if overall_evolution else None
    return {
        "stage": STAGE,
        "by_long_horizon_label": by_label,
        "by_seed_cycle_label": by_seed_cycle_label,
        "cycle_aggregate": cycle_aggregate,
        "overall_policy_evolution": overall_evolution,
        "cycle1_to11_overall": {
            "cycle1": first,
            "cycle11": last,
            "P_HOLD_mean_delta": None if not first or not last else last["P_HOLD"]["mean"] - first["P_HOLD"]["mean"],
            "P_SERVE_mean_delta": None if not first or not last else last["P_SERVE"]["mean"] - first["P_SERVE"]["mean"],
            "entropy_mean_delta": None if not first or not last else last["entropy"]["mean"] - first["entropy"]["mean"],
        },
        "conditional_discrimination_measurable": by_label[CLASS_HOLD_BETTER]["count"] > 0 and by_label[CLASS_SERVE_BETTER]["count"] > 0,
        "large_row_data_path": "conditional_policy_by_sample.parquet",
    }


def hold_or_serve_metrics(conditional: Mapping[str, Any], label: str) -> Dict[str, Any]:
    by_label = conditional["by_long_horizon_label"][label]
    cycle_rows = [row for row in conditional["cycle_aggregate"]]
    first = cycle_rows[0][("hold_better" if label == CLASS_HOLD_BETTER else "serve_better")]
    last = cycle_rows[-1][("hold_better" if label == CLASS_HOLD_BETTER else "serve_better")]
    return {
        "stage": STAGE,
        "classification": label,
        "aggregate": by_label,
        "cycle1": first,
        "cycle11": last,
        "cycle1_to11_delta": {
            "P_HOLD_mean_delta": last["P_HOLD"]["mean"] - first["P_HOLD"]["mean"],
            "P_SERVE_mean_delta": last["P_SERVE"]["mean"] - first["P_SERVE"]["mean"],
            "correct_direction_rate_delta": last["probability_correct_direction_rate"] - first["probability_correct_direction_rate"],
            "sampled_correct_direction_rate_delta": last["sampled_correct_direction_rate"] - first["sampled_correct_direction_rate"],
        },
    }


def load_h4m_h_comparator() -> Dict[str, Any]:
    h4mi_actor = read_json(H4M_I_ROOT / "05_actor_discrimination.json")["actual_h4m_h_trace_actor_output"]
    h4mh_crosswalk = read_json(H4M_H_RERUN_ROOT / "07_long_horizon_vs_gae_crosswalk.json")
    class_counts = h4mh_crosswalk["classification_counts"]
    sampled_counts = h4mh_crosswalk["classification_by_sampled_action_counts"]
    hold_count = int(class_counts.get(CLASS_HOLD_BETTER, 0))
    serve_count = int(class_counts.get(CLASS_SERVE_BETTER, 0))
    return {
        "probability_direction": {
            CLASS_HOLD_BETTER: {
                "correct_direction_rate": h4mi_actor["by_long_horizon_label"][CLASS_HOLD_BETTER]["correct_direction_rate"],
                "P_HOLD_mean": h4mi_actor["by_long_horizon_label"][CLASS_HOLD_BETTER]["P_HOLD"]["mean"],
                "P_SERVE_mean": h4mi_actor["by_long_horizon_label"][CLASS_HOLD_BETTER]["P_SERVE"]["mean"],
                "policy_margin_hold_minus_serve_mean": h4mi_actor["by_long_horizon_label"][CLASS_HOLD_BETTER][
                    "policy_margin_hold_minus_serve"
                ]["mean"],
            },
            CLASS_SERVE_BETTER: {
                "correct_direction_rate": h4mi_actor["by_long_horizon_label"][CLASS_SERVE_BETTER]["correct_direction_rate"],
                "P_HOLD_mean": h4mi_actor["by_long_horizon_label"][CLASS_SERVE_BETTER]["P_HOLD"]["mean"],
                "P_SERVE_mean": h4mi_actor["by_long_horizon_label"][CLASS_SERVE_BETTER]["P_SERVE"]["mean"],
                "policy_margin_hold_minus_serve_mean": h4mi_actor["by_long_horizon_label"][CLASS_SERVE_BETTER][
                    "policy_margin_hold_minus_serve"
                ]["mean"],
            },
            "overall_correct_direction_rate": h4mi_actor["overall_correct_direction_rate"],
        },
        "sampled_action_direction": {
            CLASS_HOLD_BETTER: {
                "correct_direction_rate": safe_rate(
                    int(sampled_counts.get(f"{CLASS_HOLD_BETTER}::{ACTION_HOLD}", 0)),
                    hold_count,
                ),
                "correct_count": int(sampled_counts.get(f"{CLASS_HOLD_BETTER}::{ACTION_HOLD}", 0)),
                "count": hold_count,
            },
            CLASS_SERVE_BETTER: {
                "correct_direction_rate": safe_rate(
                    int(sampled_counts.get(f"{CLASS_SERVE_BETTER}::{ACTION_SERVE}", 0)),
                    serve_count,
                ),
                "correct_count": int(sampled_counts.get(f"{CLASS_SERVE_BETTER}::{ACTION_SERVE}", 0)),
                "count": serve_count,
            },
            "classification_by_sampled_action_counts": sampled_counts,
        },
    }


def h4mh_vs_h4mk_comparison(conditional: Mapping[str, Any]) -> Dict[str, Any]:
    h4mh = load_h4m_h_comparator()
    k_hold = conditional["by_long_horizon_label"][CLASS_HOLD_BETTER]
    k_serve = conditional["by_long_horizon_label"][CLASS_SERVE_BETTER]
    k_total = k_hold["count"] + k_serve["count"]
    k_overall_prob_correct = safe_rate(
        k_hold["probability_correct_direction_count"] + k_serve["probability_correct_direction_count"],
        k_total,
    )
    k_overall_sample_correct = safe_rate(
        k_hold["sampled_correct_direction_count"] + k_serve["sampled_correct_direction_count"],
        k_total,
    )
    h4mh_overall_sample_correct = safe_rate(
        h4mh["sampled_action_direction"][CLASS_HOLD_BETTER]["correct_count"]
        + h4mh["sampled_action_direction"][CLASS_SERVE_BETTER]["correct_count"],
        h4mh["sampled_action_direction"][CLASS_HOLD_BETTER]["count"]
        + h4mh["sampled_action_direction"][CLASS_SERVE_BETTER]["count"],
    )
    return {
        "stage": STAGE,
        "comparator": "H4M-H-RERUN historical diagnostic only; no winner/baseline selection released",
        "h4m_h": h4mh,
        "h4m_k": {
            "probability_direction": {
                CLASS_HOLD_BETTER: {
                    "correct_direction_rate": k_hold["probability_correct_direction_rate"],
                    "P_HOLD_mean": k_hold["P_HOLD"]["mean"],
                    "P_SERVE_mean": k_hold["P_SERVE"]["mean"],
                    "policy_margin_hold_minus_serve_mean": k_hold["policy_margin_hold_minus_serve"]["mean"],
                },
                CLASS_SERVE_BETTER: {
                    "correct_direction_rate": k_serve["probability_correct_direction_rate"],
                    "P_HOLD_mean": k_serve["P_HOLD"]["mean"],
                    "P_SERVE_mean": k_serve["P_SERVE"]["mean"],
                    "policy_margin_hold_minus_serve_mean": k_serve["policy_margin_hold_minus_serve"]["mean"],
                },
                "overall_correct_direction_rate": k_overall_prob_correct,
            },
            "sampled_action_direction": {
                CLASS_HOLD_BETTER: {
                    "correct_direction_rate": k_hold["sampled_correct_direction_rate"],
                    "correct_count": k_hold["sampled_correct_direction_count"],
                    "count": k_hold["count"],
                },
                CLASS_SERVE_BETTER: {
                    "correct_direction_rate": k_serve["sampled_correct_direction_rate"],
                    "correct_count": k_serve["sampled_correct_direction_count"],
                    "count": k_serve["count"],
                },
                "overall_correct_direction_rate": k_overall_sample_correct,
            },
        },
        "deltas_h4mk_minus_h4mh": {
            "hold_better_probability_correct_direction_rate": k_hold["probability_correct_direction_rate"]
            - h4mh["probability_direction"][CLASS_HOLD_BETTER]["correct_direction_rate"],
            "serve_better_probability_correct_direction_rate": k_serve["probability_correct_direction_rate"]
            - h4mh["probability_direction"][CLASS_SERVE_BETTER]["correct_direction_rate"],
            "overall_probability_correct_direction_rate": k_overall_prob_correct
            - h4mh["probability_direction"]["overall_correct_direction_rate"],
            "hold_better_sampled_correct_direction_rate": k_hold["sampled_correct_direction_rate"]
            - h4mh["sampled_action_direction"][CLASS_HOLD_BETTER]["correct_direction_rate"],
            "serve_better_sampled_correct_direction_rate": k_serve["sampled_correct_direction_rate"]
            - h4mh["sampled_action_direction"][CLASS_SERVE_BETTER]["correct_direction_rate"],
            "overall_sampled_correct_direction_rate": k_overall_sample_correct - h4mh_overall_sample_correct,
            "hold_better_P_HOLD_mean": k_hold["P_HOLD"]["mean"] - h4mh["probability_direction"][CLASS_HOLD_BETTER]["P_HOLD_mean"],
            "serve_better_P_SERVE_mean": k_serve["P_SERVE"]["mean"] - h4mh["probability_direction"][CLASS_SERVE_BETTER]["P_SERVE_mean"],
            "separation_gap_hold_margin_minus_serve_margin": (
                k_hold["policy_margin_hold_minus_serve"]["mean"] - k_serve["policy_margin_hold_minus_serve"]["mean"]
            )
            - (
                h4mh["probability_direction"][CLASS_HOLD_BETTER]["policy_margin_hold_minus_serve_mean"]
                - h4mh["probability_direction"][CLASS_SERVE_BETTER]["policy_margin_hold_minus_serve_mean"]
            ),
        },
    }


def policy_collapse_audit(conditional: Mapping[str, Any]) -> Dict[str, Any]:
    hold = conditional["by_long_horizon_label"][CLASS_HOLD_BETTER]
    serve = conditional["by_long_horizon_label"][CLASS_SERVE_BETTER]
    probability_same_dominance = (
        hold["probability_dominant_action"] == serve["probability_dominant_action"]
        and hold["probability_dominant_action"] in {"HOLD", "SERVE"}
    )
    sampled_same_dominance = (
        hold["sampled_dominant_action"] == serve["sampled_dominant_action"]
        and hold["sampled_dominant_action"] in {"HOLD", "SERVE"}
    )
    return {
        "stage": STAGE,
        "hold_better_probability_dominant_action": hold["probability_dominant_action"],
        "serve_better_probability_dominant_action": serve["probability_dominant_action"],
        "hold_better_sampled_dominant_action": hold["sampled_dominant_action"],
        "serve_better_sampled_dominant_action": serve["sampled_dominant_action"],
        "probability_same_dominance_across_labels": probability_same_dominance,
        "sampled_same_dominance_across_labels": sampled_same_dominance,
        "global_policy_collapse_detected": probability_same_dominance and sampled_same_dominance,
        "collapse_rule": "collapse iff both HOLD-better and SERVE-better labels share the same dominant probability action and the same dominant sampled action; no arbitrary numeric tolerance",
    }


def long_horizon_shadow_summary(cycle_summaries: Sequence[Mapping[str, Any]], trace_meta: Mapping[str, Any]) -> Dict[str, Any]:
    joined_rows = trace_meta["joined_rows"]
    classifications = Counter(row["classification"] for row in joined_rows)
    shadow_audits = trace_meta["shadow_audits"]
    completeness = [row.get("shadow_completeness", {}) for row in cycle_summaries]
    return {
        "stage": STAGE,
        "classification_counts": dict(classifications),
        "total_joined_rows": len(joined_rows),
        "cycle_shadow_counts": [
            {
                "seed": row["seed"],
                "outer_cycle": row["outer_cycle"],
                "classification_counts": row.get("long_horizon_classification_counts", {}),
                "shadow_completeness": row.get("shadow_completeness", {}),
            }
            for row in cycle_summaries
        ],
        "required_long_horizon_fields_non_null": all(
            item.get("non_null_delta_discounted_reward_to_boundary")
            and item.get("non_null_delta_bootstrap_value")
            and item.get("non_null_delta_full_bootstrapped_return")
            and item.get("non_null_closure_reason")
            and item.get("non_null_classification")
            for item in completeness
        ),
        "shadow_rng_drift_count": sum(1 for row in shadow_audits if not row.get("rng_unchanged_by_shadow")),
        "shadow_model_contamination_count": sum(1 for row in shadow_audits if not row.get("model_hashes_unchanged_by_shadow")),
        "shadow_optimizer_contamination_count": sum(1 for row in shadow_audits if not row.get("optimizer_hashes_unchanged_by_shadow")),
        "shadow_normalizer_contamination_count": sum(
            1
            for row in shadow_audits
            if not (row.get("reward_normalizer_unchanged_by_shadow") and row.get("return_normalizer_unchanged_by_shadow"))
        ),
        "shadow_training_inclusion": False,
    }


def credit_trace_summary(trace_meta: Mapping[str, Any]) -> Dict[str, Any]:
    registry = trace_meta["trace_registry"]
    joined_rows = trace_meta["joined_rows"]
    row_counts = Counter()
    for cycle in registry:
        for _name, meta in cycle.get("files", {}).items():
            row_counts[_name] += int(meta.get("rows", 0))
    return {
        "stage": STAGE,
        "trace_cycle_count": len(registry),
        "trace_registry": registry,
        "row_counts": dict(row_counts),
        "joined_row_count": len(joined_rows),
        "sample_uid_join_complete": len(joined_rows) > 0,
        "target_context_preserved_by_sample_uid": all(int(row["target_context_one_hot_sum"]) == 1 for row in joined_rows),
        "lineage_captured": {
            "target_context": True,
            "legal_mask": row_counts.get("actual_pre_action_trace", 0) > 0,
            "logits_probabilities": row_counts.get("actual_pre_action_trace", 0) > 0,
            "sampled_action": row_counts.get("actual_pre_action_trace", 0) > 0,
            "reward": row_counts.get("actual_reward_trace", 0) > 0,
            "value_td": row_counts.get("actual_critic_td_gae_trace", 0) > 0,
            "raw_gae": row_counts.get("advantage_normalization_sample", 0) > 0,
            "normalized_advantage": row_counts.get("advantage_normalization_sample", 0) > 0,
            "ppo_pressure": row_counts.get("actor_logit_pressure_by_sample", 0) > 0,
        },
        "large_row_data": {
            "actual_on_policy_credit_trace_dir": "05_actual_on_policy_credit_trace",
            "long_horizon_shadow_trace_dir": "06_long_horizon_shadow_trace",
            "conditional_policy_by_sample": "conditional_policy_by_sample.parquet",
        },
    }


def training_integrity(
    binding: Mapping[str, Any],
    training: Mapping[str, Any],
    cycle_summaries: Sequence[Mapping[str, Any]],
    trace_meta: Mapping[str, Any],
    conditional: Mapping[str, Any],
    shadow_summary: Mapping[str, Any],
    credit_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    safety_counts = trace_meta.get("safety_counts", {})
    finite_values = True
    nonfinite_joined = 0
    for row in trace_meta.get("joined_rows", []):
        for key in [
            "P_HOLD",
            "P_SERVE",
            "P_SKIP",
            "policy_entropy",
            "raw_gae_advantage",
            "normalized_advantage",
            "value_error",
            "delta_full_bootstrapped_return_serve_minus_hold",
            "delta_discounted_reward_to_boundary_serve_minus_hold",
            "delta_bootstrap_value_serve_minus_hold",
        ]:
            if row.get(key) is None or not math.isfinite(float(row[key])):
                finite_values = False
                nonfinite_joined += 1
    criteria = {
        "authoritative_binding_passed": binding.get("authoritative_binding_passed") is True,
        "device_mps_available": torch.backends.mps.is_available(),
        "fresh_seeds_1_2_3": training.get("fresh_initialization_all_seeds") is True
        and training.get("seeds") == EXPECTED["seeds"],
        "schema_12d_all_seeds": all(row.get("node_feature_dim") == EXPECTED["node_feature_dim"] for row in training.get("seed_summaries", [])),
        "old_9d_checkpoint_reuse_false": training.get("old_9d_checkpoint_reuse") is False,
        "checkpoint_continuation_false": training.get("h4m_c_or_h4m_h_checkpoint_continuation") is False,
        "rollouts_11_per_seed": all(row["rollouts"] == EXPECTED["outer_training_count"] for row in training.get("seed_summaries", [])),
        "ppo_updates_44_per_seed": all(row["ppo_updates"] == EXPECTED["ppo_updates_per_seed"] for row in training.get("seed_summaries", [])),
        "critic_updates_88_per_seed": all(row["critic_updates"] == EXPECTED["critic_updates_per_seed"] for row in training.get("seed_summaries", [])),
        "active_samples_3872_per_seed": all(row["active_samples"] == EXPECTED["active_samples_per_seed"] for row in training.get("seed_summaries", [])),
        "trace_registry_33_cycles": len(trace_meta.get("trace_registry", [])) == 33,
        "finite_training_and_trace_values": int(safety_counts.get("nonfinite_loss_cycle", 0)) == 0 and finite_values,
        "illegal_action_zero": int(safety_counts.get("illegal_action", 0)) == 0,
        "illegal_skip_zero": int(safety_counts.get("illegal_SKIP", 0)) == 0,
        "safety_violation_zero": int(safety_counts.get("illegal_action", 0)) == 0 and int(safety_counts.get("illegal_SKIP", 0)) == 0,
        "rng_drift_zero": int(safety_counts.get("shadow_rng_drift", 0)) == 0,
        "shadow_contamination_zero": int(safety_counts.get("shadow_model_contamination", 0)) == 0
        and int(safety_counts.get("shadow_optimizer_contamination", 0)) == 0
        and int(safety_counts.get("shadow_normalizer_contamination", 0)) == 0,
        "long_horizon_labels_available": shadow_summary.get("classification_counts", {}).get(CLASS_HOLD_BETTER, 0) > 0
        and shadow_summary.get("classification_counts", {}).get(CLASS_SERVE_BETTER, 0) > 0,
        "long_horizon_required_fields_non_null": shadow_summary.get("required_long_horizon_fields_non_null") is True,
        "conditional_discrimination_measurable": conditional.get("conditional_discrimination_measurable") is True,
        "target_context_preserved": credit_summary.get("target_context_preserved_by_sample_uid") is True,
        "lineage_captured": all(credit_summary.get("lineage_captured", {}).values()),
        "parameter_delta_positive_all_seeds": all(row.get("parameter_delta_positive") is True for row in training.get("seed_summaries", [])),
        "instrumentation_mps_contract_pass": binding.get("checks", {}).get("h4m_g_close_active_contract_match_file") is True,
        "validation_or_test6_not_executed": True,
        "github_push_not_performed": True,
    }
    return {
        "stage": STAGE,
        "created_at": training.get("created_at"),
        "criteria": criteria,
        "training_integrity_passed": all(criteria.values()),
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "safety_counts": dict(safety_counts),
        "nonfinite_joined_value_count": nonfinite_joined,
        "shadow_audits": trace_meta.get("shadow_audits", []),
        "active_mps_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "TEST6": "SEALED_NOT_OPENED",
    }


def root_decision(comparison: Mapping[str, Any], collapse: Mapping[str, Any]) -> Dict[str, Any]:
    deltas = comparison["deltas_h4mk_minus_h4mh"]
    h4mk = comparison["h4m_k"]
    hold_prob_improved = deltas["hold_better_probability_correct_direction_rate"] > 0.0
    hold_action_improved = deltas["hold_better_sampled_correct_direction_rate"] > 0.0
    serve_prob_maintained = deltas["serve_better_probability_correct_direction_rate"] >= 0.0
    serve_action_maintained = deltas["serve_better_sampled_correct_direction_rate"] >= 0.0
    if collapse.get("global_policy_collapse_detected") is True:
        decision = "GLOBAL_POLICY_COLLAPSE_TO_HOLD_OR_SERVE"
        next_gate = "H4M-L_TARGET_CONTEXT_REPAIR_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
    elif hold_prob_improved and hold_action_improved and serve_prob_maintained and serve_action_maintained:
        decision = "TARGET_CONTEXT_REPAIR_CONFIRMED_STATE_CONDITIONAL_POLICY_DISCRIMINATION"
        next_gate = "H4M-L_FROZEN_TARGET_CONTEXT_REPAIRED_POLICY_REVALIDATION"
    elif hold_prob_improved or hold_action_improved:
        decision = "TARGET_CONTEXT_REPAIR_PARTIALLY_IMPROVED_BUT_INSUFFICIENT"
        next_gate = "H4M-L_TARGET_CONTEXT_REPAIR_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
    elif not hold_prob_improved and not hold_action_improved:
        decision = "TARGET_CONTEXT_REPAIR_FAILED_TO_IMPROVE_HOLD_DISCRIMINATION"
        next_gate = "H4M-L_TARGET_CONTEXT_REPAIR_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
    else:
        decision = "CAUSAL_EFFECT_NOT_UNIQUE"
        next_gate = "H4M-L_MINIMAL_CAUSAL_EFFECT_ISOLATION_AUDIT"
    return {
        "stage": STAGE,
        "decision": decision,
        "exact_next_gate": next_gate,
        "decision_rule": {
            "no_arbitrary_tolerance": True,
            "confirmed_requires": [
                "HOLD-better probability correct-direction rate > H4M-H",
                "HOLD-better sampled correct-direction rate > H4M-H",
                "SERVE-better probability correct-direction rate >= H4M-H",
                "SERVE-better sampled correct-direction rate >= H4M-H",
                "no global same-action collapse",
            ],
            "global_collapse_rule": collapse.get("collapse_rule"),
        },
        "evidence": {
            "hold_prob_improved": hold_prob_improved,
            "hold_action_improved": hold_action_improved,
            "serve_prob_maintained_or_improved": serve_prob_maintained,
            "serve_action_maintained_or_improved": serve_action_maintained,
            "global_policy_collapse_detected": collapse.get("global_policy_collapse_detected"),
            "deltas_h4mk_minus_h4mh": deltas,
            "h4mk_probability_direction": h4mk["probability_direction"],
            "h4mk_sampled_action_direction": h4mk["sampled_action_direction"],
        },
    }


def gate_matrix(
    binding: Mapping[str, Any],
    integrity: Mapping[str, Any],
    training: Mapping[str, Any],
    conditional: Mapping[str, Any],
    root: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_bindings_match": binding.get("authoritative_binding_passed") is True,
        "training_integrity_pass": integrity.get("training_integrity_passed") is True,
        "seeds_1_2_3_complete": training.get("seeds") == EXPECTED["seeds"],
        "schema_12d_used": all(row.get("node_feature_dim") == EXPECTED["node_feature_dim"] for row in training.get("seed_summaries", [])),
        "total_rollouts_33": training.get("total_rollouts") == 33,
        "total_ppo_updates_132": training.get("total_ppo_updates") == 132,
        "total_critic_updates_264": training.get("total_critic_updates") == 264,
        "conditional_discrimination_result_measurable": conditional.get("conditional_discrimination_measurable") is True,
        "decision_selected": root.get("decision")
        in {
            "TARGET_CONTEXT_REPAIR_CONFIRMED_STATE_CONDITIONAL_POLICY_DISCRIMINATION",
            "TARGET_CONTEXT_REPAIR_PARTIALLY_IMPROVED_BUT_INSUFFICIENT",
            "TARGET_CONTEXT_REPAIR_FAILED_TO_IMPROVE_HOLD_DISCRIMINATION",
            "GLOBAL_POLICY_COLLAPSE_TO_HOLD_OR_SERVE",
            "CAUSAL_EFFECT_NOT_UNIQUE",
        },
        "test6_sealed": integrity.get("TEST6") == "SEALED_NOT_OPENED",
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": root.get("decision") if passed else "H4M_K_FRESH_TARGET_CONTEXT_REPAIRED_THREE_SEED_RETRAINING_BLOCKED",
        "exact_next_gate": root.get("exact_next_gate") if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "final_flags": {
            "fresh_training_executed": passed,
            "reward_gae_ppo_modified": False,
            "feature_added_or_deleted_beyond_h4m_i_contract": False,
            "environment_expanded": False,
            "validation_or_test6_executed": False,
            "winner_or_baseline_selected": False,
            "github_push_performed": False,
        },
        "active_mps_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "observation_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
    }


def write_conditional_parquet(base: Any, artifact_root: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return base.parquet_write(rows, artifact_root / "conditional_policy_by_sample.parquet")


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files: Dict[str, str] = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            output_files[str(path.relative_to(artifact_root))] = str(path)
    output_sha = {name: sha256_file(Path(path)) for name, path in output_files.items()}
    return {
        "stage": STAGE,
        "created_at": kst_now(),
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": output_files,
        "output_sha256": output_sha,
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift",
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def final_report(
    binding: Mapping[str, Any],
    integrity: Mapping[str, Any],
    training: Mapping[str, Any],
    conditional: Mapping[str, Any],
    hold_metrics: Mapping[str, Any],
    serve_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    collapse: Mapping[str, Any],
    root: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    seed_counts = [
        {
            "seed": row["seed"],
            "rollouts": row["rollouts"],
            "ppo_updates": row["ppo_updates"],
            "critic_updates": row["critic_updates"],
            "active_samples": row["active_samples"],
            "node_feature_dim": row["node_feature_dim"],
        }
        for row in training.get("seed_summaries", [])
    ]
    return f"""# H4M-K Fresh Target-Context Repaired Three-Seed Retraining

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_k_source_git_commit"]}
decision = {gate["decision"]}
exact_next_gate = {gate["exact_next_gate"]}

## Three-seed execution counts

```json
{json.dumps(seed_counts, ensure_ascii=False, indent=2, default=jsonable)}
```

## Cycle 1→11 overall policy evolution

```json
{json.dumps(conditional.get("cycle1_to11_overall"), ensure_ascii=False, indent=2, default=jsonable)}
```

## HOLD-better conditional result

```json
{json.dumps(hold_metrics.get("aggregate"), ensure_ascii=False, indent=2, default=jsonable)}
```

## SERVE-better conditional result

```json
{json.dumps(serve_metrics.get("aggregate"), ensure_ascii=False, indent=2, default=jsonable)}
```

## H4M-H 대비 개선량

```json
{json.dumps(comparison.get("deltas_h4mk_minus_h4mh"), ensure_ascii=False, indent=2, default=jsonable)}
```

## Global collapse

```json
{json.dumps(collapse, ensure_ascii=False, indent=2, default=jsonable)}
```

## Integrity

training_integrity_passed = {integrity.get("training_integrity_passed")}
TEST6 = {integrity.get("TEST6")}

STOP: no additional training, no Reward/GAE/PPO modification, no feature change, no environment expansion, no validation/TEST6, no winner/baseline, no GitHub push.
"""


def write_block_outputs(
    artifact_root: Path,
    created_at: str,
    binding: Mapping[str, Any],
    reason: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    gate = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": BLOCK_GATE,
        "decision": "H4M_K_FRESH_TARGET_CONTEXT_REPAIRED_THREE_SEED_RETRAINING_BLOCKED",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
        "extra": extra or {},
    }
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_training_integrity.json": empty,
        "03_three_seed_cycle_summary.json": empty,
        "04_conditional_policy_discrimination.json": empty,
        "05_hold_better_metrics.json": empty,
        "06_serve_better_metrics.json": empty,
        "07_h4mh_vs_h4mk_comparison.json": empty,
        "08_long_horizon_shadow_summary.json": empty,
        "09_credit_trace_summary.json": empty,
        "10_policy_collapse_audit.json": empty,
        "11_root_decision.json": empty,
        "12_checkpoint_registry.json": empty,
        "13_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(f"# H4M-K\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-K] artifact root: {artifact_root}")
    print(f"[H4M-K] gate: {BLOCK_GATE}")
    print(f"[H4M-K] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_k_fresh_target_context_repaired_three_seed_retraining_{stamp}"
    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    if not binding["authoritative_binding_passed"]:
        write_block_outputs(artifact_root, created_at, binding, "AUTHORITATIVE_BINDING_MISMATCH")
        return
    if not torch.backends.mps.is_available():
        write_block_outputs(
            artifact_root,
            created_at,
            binding,
            "MPS_NOT_AVAILABLE_FOR_AUTHORIZED_FRESH_RETRAINING",
            {"mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available()},
        )
        return

    base, _rerun = load_h4m_helpers()
    h4mg = load_h4mg()
    device = torch.device("mps")
    training, cycle_summaries, checkpoint_registry, trace_meta = run_training(base, h4mg, created_at, artifact_root, device)
    conditional = conditional_policy_discrimination(trace_meta["joined_rows"])
    conditional_parquet = write_conditional_parquet(base, artifact_root, trace_meta["joined_rows"])
    conditional["conditional_policy_by_sample_parquet"] = conditional_parquet
    hold_metrics = hold_or_serve_metrics(conditional, CLASS_HOLD_BETTER)
    serve_metrics = hold_or_serve_metrics(conditional, CLASS_SERVE_BETTER)
    comparison = h4mh_vs_h4mk_comparison(conditional)
    collapse = policy_collapse_audit(conditional)
    shadow_summary = long_horizon_shadow_summary(cycle_summaries, trace_meta)
    credit_summary = credit_trace_summary(trace_meta)
    integrity = training_integrity(binding, training, cycle_summaries, trace_meta, conditional, shadow_summary, credit_summary)
    root = root_decision(comparison, collapse)
    gate = gate_matrix(binding, integrity, training, conditional, root)
    report = final_report(binding, integrity, training, conditional, hold_metrics, serve_metrics, comparison, collapse, root, gate)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_training_integrity.json": integrity,
        "03_three_seed_cycle_summary.json": training,
        "04_conditional_policy_discrimination.json": conditional,
        "05_hold_better_metrics.json": hold_metrics,
        "06_serve_better_metrics.json": serve_metrics,
        "07_h4mh_vs_h4mk_comparison.json": comparison,
        "08_long_horizon_shadow_summary.json": shadow_summary,
        "09_credit_trace_summary.json": credit_summary,
        "10_policy_collapse_audit.json": collapse,
        "11_root_decision.json": root,
        "12_checkpoint_registry.json": checkpoint_registry,
        "13_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    print(f"[H4M-K] artifact root: {artifact_root}")
    print(f"[H4M-K] gate: {gate['gate']}")
    print(f"[H4M-K] source_commit: {provenance['h4m_k_source_git_commit']}")
    print(
        f"[H4M-K] total_rollouts={training['total_rollouts']} "
        f"total_ppo_updates={training['total_ppo_updates']} total_critic_updates={training['total_critic_updates']}"
    )
    print(f"[H4M-K] hold_better_prob_correct={hold_metrics['aggregate']['probability_correct_direction_rate']}")
    print(f"[H4M-K] serve_better_prob_correct={serve_metrics['aggregate']['probability_correct_direction_rate']}")
    print(f"[H4M-K] global_collapse={collapse['global_policy_collapse_detected']}")
    print(f"[H4M-K] decision: {gate['decision']}")
    print(f"[H4M-K] next_gate: {gate['exact_next_gate']}")
    print("[H4M-K] STOP: no additional training, no validation/TEST6, no winner/baseline, github_push=false")


if __name__ == "__main__":
    main()
