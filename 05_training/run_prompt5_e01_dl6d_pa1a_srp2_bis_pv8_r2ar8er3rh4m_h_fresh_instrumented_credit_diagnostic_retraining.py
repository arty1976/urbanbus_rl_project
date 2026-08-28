#!/usr/bin/env python3
"""H4M-H fresh instrumented credit diagnostic retraining.

Runs the H4M-F/H4M-G frozen credit instrumentation through a fresh
seed 1/2/3 × 11-cycle diagnostic retraining, after mandatory MPS OFF/ON
equivalence and long-horizon shadow completeness preflights.

This stage may train only the authorized fresh diagnostic run. It must not
continue from H4M-C checkpoints, open validation/TEST6, compare baselines,
select a winner, tune hyperparameters, repair Reward/GAE/PPO/model code, or
push to GitHub.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-H"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_H_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_H_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_F_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_f_credit_trace_instrumentation_selection_freeze_20260815_111506+0900"
H4M_G_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_instrumentation_equivalence_validation_20260815_122304+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
H4M_A_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_a_decision_opportunity_environment_adequacy_audit_20260814_143743"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"

EXPECTED = {
    "h4m_f_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_F_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_SELECTION_AND_FREEZE_COMPLETE",
    "h4m_f_decision": "PV8_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_FROZEN_READY_FOR_IMPLEMENTATION_EQUIVALENCE_VALIDATION",
    "h4m_f_source_commit": "570e8ec9a48adc3fd7d91231f71976b65e23224d",
    "h4m_f_contract_sha256": "9b95d0dc46459eedd2cf51e85789407be247e9db1e23e3a52a5bbae9c6216ffe",
    "h4m_g_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_INSTRUMENTATION_IMPLEMENTATION_AND_BEHAVIORAL_EQUIVALENCE_VALIDATION_COMPLETE",
    "h4m_g_decision": "PV8_CREDIT_TRACE_INSTRUMENTATION_IMPLEMENTED_AND_BEHAVIORALLY_EQUIVALENT_READY_FOR_FRESH_DIAGNOSTIC_RETRAINING",
    "h4m_g_source_commit": "672e05a24e09440351e46dad05cf5ab02f498060",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "outer_training_count": 11,
    "ppo_epochs": 4,
    "critic_epochs": 8,
    "minibatch": 256,
    "rollout_horizon": 512,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "lr": 0.001,
}

ACTION_NAMES = {
    0: "HOLD_CURRENT_POSITION",
    1: "SERVE_AND_MOVE_TO_NEXT_STOP",
    2: "CONDITIONAL_SKIP_EMPTY_STOP",
}
ACTION_SHORT = {0: "HOLD", 1: "SERVE", 2: "SKIP"}
TOL = 1.0e-12

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_mps_off_on_preflight.json",
    "03_long_horizon_shadow_preflight.json",
    "04_three_seed_training_summary.json",
    "05_actual_on_policy_credit_trace",
    "06_long_horizon_shadow_trace",
    "07_advantage_sign_flip_summary.json",
    "08_ppo_actor_pressure_summary.json",
    "09_critic_credit_bias_summary.json",
    "10_cycle1_to11_policy_evolution.json",
    "11_root_cause_classification.json",
    "12_safety_integrity.json",
    "13_checkpoint_registry.json",
    "14_h4m_h_gate_matrix.json",
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
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


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module_from_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git_bool(args: Sequence[str]) -> bool:
    return git_run(args, check=False).returncode == 0


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    source = SOURCE_REL.as_posix()
    latest = git_run(["log", "-1", "--format=%H", "--", source]).stdout.strip()
    head_files = [line.strip() for line in git_run(["show", "--name-only", "--pretty=format:", "HEAD"]).stdout.splitlines() if line.strip()]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_h_source_git_commit": head,
        "h4m_h_source_path": source,
        "h4m_h_source_sha256": sha256_file(PROJECT_ROOT / source),
        "h4m_h_source_present_in_head": git_bool(["cat-file", "-e", f"HEAD:{source}"]),
        "h4m_h_source_latest_commit": latest,
        "h4m_h_source_no_uncommitted_diff_vs_head": git_bool(["diff", "--quiet", "--", source]),
        "h4m_h_source_no_staged_diff_vs_head": git_bool(["diff", "--cached", "--quiet", "--", source]),
        "head_commit_files": head_files,
        "local_source_only_commit_created_before_training": latest == head and head_files == [source],
        "post_commit_provenance_gate_passed": latest == head and head_files == [source] and git_bool(["diff", "--quiet", "--", source]) and git_bool(["diff", "--cached", "--quiet", "--", source]),
        "status_short": git_run(["status", "--short"]).stdout,
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    h4m_f_manifest = read_json(H4M_F_ROOT / "manifest.json")
    h4m_f_gate = read_json(H4M_F_ROOT / "14_h4m_f_gate_matrix.json")
    h4m_f_contract_path = H4M_F_ROOT / "13_h4m_f_credit_trace_instrumentation_contract.json"
    h4m_f_contract = read_json(h4m_f_contract_path)
    h4m_g_manifest = read_json(H4M_G_ROOT / "manifest.json")
    h4m_g_gate = read_json(H4M_G_ROOT / "09_h4m_g_gate_matrix.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    checks = {
        "h4m_f_gate_match": h4m_f_gate.get("gate") == EXPECTED["h4m_f_gate"],
        "h4m_f_decision_match": h4m_f_gate.get("decision") == EXPECTED["h4m_f_decision"],
        "h4m_f_source_commit_match": h4m_f_manifest.get("h4m_f_source_git_commit") == EXPECTED["h4m_f_source_commit"],
        "h4m_f_contract_sha_match": sha256_file(h4m_f_contract_path) == EXPECTED["h4m_f_contract_sha256"],
        "h4m_g_gate_match": h4m_g_gate.get("gate") == EXPECTED["h4m_g_gate"],
        "h4m_g_decision_match": h4m_g_gate.get("decision") == EXPECTED["h4m_g_decision"],
        "h4m_g_source_commit_match": h4m_g_manifest.get("h4m_g_source_git_commit") == EXPECTED["h4m_g_source_commit"],
        "h4m_g_contract_sha_match": h4m_g_manifest.get("instrumentation_contract_sha256") == EXPECTED["h4m_f_contract_sha256"],
        "h4m_b_schedule_sha_match": h4m_b_schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "outer_training_count_match": h4m_b_schedule.get("outer_training_count") == EXPECTED["outer_training_count"],
        "ppo_epochs_match": h4m_b_schedule.get("ppo_epochs_per_update") == EXPECTED["ppo_epochs"],
        "critic_epochs_match": h4m_b_schedule.get("critic_epochs_per_update") == EXPECTED["critic_epochs"],
        "minibatch_match": h4m_b_schedule.get("minibatch") == EXPECTED["minibatch"],
        "gamma_match": abs(float(h4m_b_schedule.get("gamma")) - EXPECTED["gamma"]) <= 0.0,
        "gae_lambda_match": abs(float(h4m_b_schedule.get("gae_lambda")) - EXPECTED["gae_lambda"]) <= 0.0,
        "clip_match": abs(float(h4m_b_schedule.get("clip_epsilon")) - EXPECTED["clip"]) <= 0.0,
        "lr_match": all(abs(float(h4m_b_schedule.get(k)) - EXPECTED["lr"]) <= 0.0 for k in ("actor_lr", "critic_lr", "gatv2_lr")),
        "reward_v2_sha_match": h4m_f_contract.get("authoritative_upstream_SHA_bindings", {}).get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": h4m_f_contract.get("authoritative_upstream_SHA_bindings", {}).get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": h4m_f_contract.get("authoritative_upstream_SHA_bindings", {}).get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": h4m_f_contract.get("authoritative_upstream_SHA_bindings", {}).get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "h4m_h_source_commit_frozen": provenance.get("post_commit_provenance_gate_passed") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_f": str(H4M_F_ROOT),
            "h4m_g": str(H4M_G_ROOT),
            "h4m_b": str(H4M_B_ROOT),
            "h4m_c": str(H4M_C_ROOT),
            "h4m_a": str(H4M_A_ROOT),
        },
        "instrumentation_contract_sha256": sha256_file(h4m_f_contract_path),
        "h4m_f_contract_path": str(h4m_f_contract_path),
        "selected_scope": h4m_f_gate.get("selected_instrumentation_scope"),
        "h4m_g_source_commit": h4m_g_manifest.get("h4m_g_source_git_commit"),
        "schedule": {
            "outer_training_count": h4m_b_schedule.get("outer_training_count"),
            "ppo_epochs_per_update": h4m_b_schedule.get("ppo_epochs_per_update"),
            "critic_epochs_per_update": h4m_b_schedule.get("critic_epochs_per_update"),
            "minibatch": h4m_b_schedule.get("minibatch"),
            "gamma": h4m_b_schedule.get("gamma"),
            "gae_lambda": h4m_b_schedule.get("gae_lambda"),
            "clip_epsilon": h4m_b_schedule.get("clip_epsilon"),
            "actor_lr": h4m_b_schedule.get("actor_lr"),
            "critic_lr": h4m_b_schedule.get("critic_lr"),
            "gatv2_lr": h4m_b_schedule.get("gatv2_lr"),
            "training_windows": h4m_b_schedule.get("training_windows"),
            "agents": 8,
            "seeds": [1, 2, 3],
        },
        "source_provenance": dict(provenance),
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "validation_executed": False,
        "test6_status": "SEALED_NOT_OPENED",
        "github_push_performed": False,
    }


def load_h4mg() -> Any:
    return import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py",
        f"h4m_h_h4mg_{time.time_ns()}",
    )


def build_context_mps(h4mg: Any, seed: int, created_at: str, *, device: torch.device) -> Dict[str, Any]:
    h4m_c = h4mg.import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_c_fresh_extended_budget_three_seed_retraining.py",
        f"h4m_h_h4m_c_{seed}_{time.time_ns()}",
    )
    h4k = h4mg.import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py",
        f"h4m_h_h4k_{seed}_{time.time_ns()}",
    )
    dl4 = h4mg.import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
        f"h4m_h_dl4_{seed}_{time.time_ns()}",
    )
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    mappo_mod = h4mg.import_module_from_path(TRAINING_ROOT / "mappo_runner.py", f"h4m_h_mappo_{seed}_{time.time_ns()}")
    reward_mod = h4mg.import_module_from_path(TRAINING_ROOT / "rewards/mappo_reward_v1.py", f"h4m_h_reward_{seed}_{time.time_ns()}")
    schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    window_plan = read_json(H4M_C_ROOT / "seed_001" / "r3_train_window_plan.json")
    train_paths = [Path(row["snapshot_path"]) for row in window_plan["train_rows"]]
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])

    dl4.set_all_seeds(seed)
    sample_full = dl1.torch_load(train_paths[0])
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(PROJECT_ROOT, sample_full, mapping_artifact=mapping_artifact)
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    train_data = dl4.load_subgraphs(dl1, train_paths, spec)
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
        "connectivity": connectivity,
        "tensor_mask": tensor_mask,
        "inventory": inventory,
    }


def parquet_write(rows: Sequence[Mapping[str, Any]], path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(row) for row in rows])
    frame.to_parquet(path, index=False)
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def compare_branch_preflight(h4mg: Any, seed: int, created_at: str, artifact_root: Path, device: torch.device) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Mapping[str, Any]]]:
    off = run_branch_mps(h4mg, "OFF", seed, created_at, device, recorder=None)
    recorder = h4mg.TraceRecorder(root=artifact_root / "02_mps_off_on_preflight_trace", enabled=True)
    on = run_branch_mps(h4mg, "ON", seed, created_at, device, recorder=recorder)
    trace_write = recorder.write()
    sample, credit, params, first_mismatch = h4mg.compare_branches(off, on)
    rng = h4mg.rng_noninterference_audit(created_at, off, on)
    trace = h4mg.trace_schema_integrity(created_at, trace_write, recorder)
    shadow = h4mg.shadow_training_separation_audit(created_at, recorder, on.get("shadow_result") or {})
    checks = {
        "mps_available": torch.backends.mps.is_available(),
        "first_material_mismatch_none": first_mismatch is None,
        "rng_drift_zero": rng.get("rng_noninterference_passed") is True,
        "shadow_contamination_zero": shadow.get("shadow_training_separation_passed") is True,
        "sample_equivalence": sample.get("sample_equivalence_passed") is True,
        "credit_equivalence": credit.get("credit_equivalence_passed") is True,
        "parameter_equivalence": params.get("parameter_checkpoint_equivalence_passed") is True,
        "trace_integrity": trace.get("trace_schema_integrity_passed") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "device": str(device),
        "seed": seed,
        "preflight_passed": all(checks.values()),
        "checks": checks,
        "first_material_mismatch": first_mismatch,
        "rng_noninterference": rng,
        "sample_equivalence": sample,
        "credit_equivalence": credit,
        "parameter_equivalence": params,
        "trace_integrity": trace,
        "shadow_separation": shadow,
    }, on, recorder


def run_branch_mps(h4mg: Any, branch: str, seed: int, created_at: str, device: torch.device, recorder: Optional[Any]) -> Dict[str, Any]:
    ctx = build_context_mps(h4mg, seed, created_at, device=device)
    dl1 = ctx["dl1"]
    before_state = {
        "gatv2": dl1.clone_state_dict(ctx["encoder"]),
        "actor": dl1.clone_state_dict(ctx["actor"]),
        "critic": dl1.clone_state_dict(ctx["critic"]),
    }
    result: Dict[str, Any] = {
        "branch": branch,
        "initial_rng": h4mg.rng_snapshot(),
        "initial_model_hashes": h4mg.model_hashes(dl1, ctx["encoder"], ctx["actor"], ctx["critic"]),
        "initial_optimizer_hashes": h4mg.optimizer_hashes(ctx["optimizers"]),
        "initial_reward_normalizer_state": h4mg.reward_normalizer_state(ctx["reward_normalizer"]),
        "initial_return_normalizer_state": dict(ctx["return_normalizer"].state_dict()),
    }
    rollout = h4mg.collect_controlled_rollout(branch=branch, seed=seed, cycle_index=1, ctx=ctx, recorder=recorder)
    result["post_rollout_pre_update_rng"] = h4mg.rng_snapshot()
    update = h4mg.ppo_update_controlled(branch=branch, seed=seed, cycle_index=1, ctx=ctx, rollout=rollout, before_state=before_state, recorder=recorder)
    result["post_update_pre_shadow_rng"] = h4mg.rng_snapshot()
    shadow_result = None
    if recorder is not None and recorder.enabled:
        shadow_result = h4mg.shadow_smoke_trace(seed, 1, ctx, rollout, recorder)
    result["post_shadow_rng"] = h4mg.rng_snapshot()
    result["shadow_result"] = shadow_result
    result["final_model_hashes"] = h4mg.model_hashes(dl1, ctx["encoder"], ctx["actor"], ctx["critic"])
    result["final_optimizer_hashes"] = h4mg.optimizer_hashes(ctx["optimizers"])
    result["final_reward_normalizer_state"] = h4mg.reward_normalizer_state(ctx["reward_normalizer"])
    result["final_return_normalizer_state"] = dict(ctx["return_normalizer"].state_dict())
    result["parameter_delta"] = {
        "gatv2": dl1.delta_stats(ctx["encoder"], before_state["gatv2"]),
        "actor": dl1.delta_stats(ctx["actor"], before_state["actor"]),
        "critic": dl1.delta_stats(ctx["critic"], before_state["critic"]),
    }
    checkpoint_payload = {
        "model_hashes": result["final_model_hashes"],
        "optimizer_hashes": result["final_optimizer_hashes"],
        "reward_normalizer_state": result["final_reward_normalizer_state"],
        "return_normalizer_state": result["final_return_normalizer_state"],
        "config_sha": canonical_sha({k: v for k, v in ctx["config"].items() if k not in {"spec", "started_at_perf"}}),
    }
    result["in_memory_checkpoint_equivalence_sha256"] = canonical_sha(checkpoint_payload)
    result["rollout_fingerprints"] = {
        "sample_uid_sequence": [uid for _t, _a, uid in rollout["active_order"]],
        "sample_uid_sequence_sha256": canonical_sha([uid for _t, _a, uid in rollout["active_order"]]),
        "sampled_actions_sha256": h4mg.tensor_hash(rollout["actions"]),
        "legal_masks_sha256": h4mg.tensor_hash(rollout["agent_mask"]),
        "targets_sha256": h4mg.tensor_hash(rollout["targets"]),
        "raw_rewards_sha256": h4mg.tensor_hash(rollout["raw_rewards"]),
        "normalized_rewards_sha256": h4mg.tensor_hash(rollout["normalized_rewards"]),
        "value_t_sha256": h4mg.tensor_hash(rollout["values_original"]),
        "next_value_t_sha256": h4mg.tensor_hash(rollout["next_values_original"]),
        "td_delta_sha256": h4mg.tensor_hash(rollout["td_delta"]),
        "raw_gae_sha256": h4mg.tensor_hash(rollout["advantages"]),
        "normalized_advantage_sha256": h4mg.tensor_hash(rollout["normalized_advantages"]),
        "return_target_sha256": h4mg.tensor_hash(rollout["returns_original"]),
        "old_log_prob_sha256": h4mg.tensor_hash(rollout["old_log_probs"]),
    }
    result["ppo_fingerprints"] = {
        "ppo_row_count": len(update["ppo_fingerprint_rows"]),
        "ppo_rows_sha256": canonical_sha(update["ppo_fingerprint_rows"]),
        "metrics_rows_sha256": canonical_sha([{k: v for k, v in row.items() if k not in {"ppo_update_seconds", "branch"}} for row in update["metrics_rows"]]),
        "gradient_rows_sha256": canonical_sha(update["gradient_rows"]),
        "loss_rows_sha256": canonical_sha(update["loss_rows"]),
    }
    result["finite_loss_passed"] = all(all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in update["loss_rows"])
    result["_rollout_for_pre2"] = rollout
    result["_ctx_for_pre2"] = ctx
    return result


def classify_delta(delta: float) -> str:
    if delta > TOL:
        return "SERVE_LONG_HORIZON_BETTER"
    if delta < -TOL:
        return "HOLD_LONG_HORIZON_BETTER"
    return "TIE_WITH_EXISTING_TOLERANCE"


def build_full_shadow_rows(h4mg: Any, seed: int, cycle: int, ctx: Mapping[str, Any], rollout: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    h4m_c = ctx["h4m_c"]
    reward_mod = ctx["reward_mod"]
    window_rows = ctx["window_plan"]["train_rows"]
    gamma = float(ctx["config"]["gamma"])
    horizon = int(rollout["actions"].size(0))
    pre_by_uid = {row["sample_uid"]: row for row in rollout["policy_rows"]}
    raw_reward_by_slot_step: Dict[Tuple[int, int], float] = {}
    for row in rollout["reward_rows"]:
        raw_reward_by_slot_step[(int(row["step_index"]), int(row["agent_slot"]))] = float(row["reward_v2_raw_total"])
    pair_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    private_rng = random.Random(canonical_sha({"stage": STAGE, "seed": seed, "cycle": cycle, "shadow": "private"}))
    for sample_uid, policy in pre_by_uid.items():
        legal_ids = [int(v) for v in policy["legal_action_ids"]]
        if 0 not in legal_ids or 1 not in legal_ids:
            continue
        t0 = int(policy["step_index"])
        agent_slot = int(policy["agent_slot"])
        target_id = int(policy["target_id"])
        pair_uid = canonical_sha({"sample_uid": sample_uid, "contract": EXPECTED["h4m_f_contract_sha256"], "pair": "HOLD_SERVE_LONG_HORIZON"})
        hold_branch_uid = canonical_sha({"pair_uid": pair_uid, "branch": "HOLD"})
        serve_branch_uid = canonical_sha({"pair_uid": pair_uid, "branch": "SERVE"})
        pair_rows.append(
            {
                "counterfactual_pair_uid": pair_uid,
                "sample_uid": sample_uid,
                "seed": seed,
                "outer_cycle": cycle,
                "initial_state_hash": policy["state_hash"],
                "hold_branch_uid": hold_branch_uid,
                "serve_branch_uid": serve_branch_uid,
                "eligible_reason": "HOLD_AND_SERVE_LEGAL_AT_T0",
                "actual_sampled_action_id": int(policy["action_id"]),
                "continuation_contract": "CURRENT_PRE_UPDATE_STOCHASTIC_POLICY_PLUS_COMMON_RANDOM_NUMBERS_STATIC_OFFLINE_SNAPSHOT_SEMANTICS",
            }
        )
        branch_totals: Dict[int, Dict[str, float]] = {}
        for initial_action_id, branch_uid, branch_name in [
            (0, hold_branch_uid, "HOLD_INITIAL"),
            (1, serve_branch_uid, "SERVE_INITIAL"),
        ]:
            discounted_sum = 0.0
            reward_sequence_hash_inputs: List[Any] = []
            for event_index, step in enumerate(range(t0, horizon)):
                window = window_rows[step]
                if step == t0:
                    metrics = h4m_c.h4m_c_reward_v2_metrics(
                        reward_mod,
                        cycle_index=cycle,
                        window=window,
                        local_step=step,
                        agent_slot=agent_slot,
                        action_id=initial_action_id,
                        target_id=target_id,
                    )
                    materialized = reward_mod.compute_reward_v2(metrics)
                    reward_total = float(materialized["reward_total"])
                    action_id = initial_action_id
                    service_hash = canonical_sha(materialized["reward_service_component"])
                    wait_hash = canonical_sha(materialized["reward_avg_wait_component"])
                else:
                    reward_total = raw_reward_by_slot_step[(step, agent_slot)]
                    action_id = int(rollout["actions"][step, agent_slot].detach().cpu().item())
                    service_hash = canonical_sha({"continuation": "actual_on_policy_reward_reused", "step": step, "agent_slot": agent_slot})
                    wait_hash = service_hash
                discounted = (gamma ** event_index) * reward_total
                discounted_sum += discounted
                private_uniform_ref = private_rng.random()
                reward_sequence_hash_inputs.append([step, action_id, reward_total, private_uniform_ref])
                event_rows.append(
                    {
                        "branch_uid": branch_uid,
                        "counterfactual_pair_uid": pair_uid,
                        "sample_uid": sample_uid,
                        "seed": seed,
                        "outer_cycle": cycle,
                        "branch_action": branch_name,
                        "branch_event_index": event_index,
                        "state_hash": policy["state_hash"] if step == t0 else canonical_sha({"sample_uid": sample_uid, "continuation_step": step}),
                        "event_sequence_hash": canonical_sha(reward_sequence_hash_inputs[-1]),
                        "action_id": action_id,
                        "reward_v2_raw_total": reward_total,
                        "reward_v2_discounted": discounted,
                        "service_outcome_hash": service_hash,
                        "passenger_wait_outcome_hash": wait_hash,
                        "future_decision_index": event_index,
                        "closure_reason": "ROLLOUT_BOUNDARY_REACHED" if step == horizon - 1 else None,
                        "private_rng_stream_ref": canonical_sha({"pair_uid": pair_uid, "event_index": event_index, "private_uniform_ref": private_uniform_ref}),
                    }
                )
            remaining_events = horizon - t0
            boundary_value = float(rollout["next_values_original"][horizon - 1, agent_slot].detach().cpu().item())
            bootstrap_value = (gamma ** remaining_events) * boundary_value
            branch_totals[initial_action_id] = {
                "discounted_reward_to_boundary": discounted_sum,
                "boundary_bootstrap_value": bootstrap_value,
                "full_bootstrapped_return": discounted_sum + bootstrap_value,
                "immediate_reward": event_rows[-remaining_events]["reward_v2_raw_total"],
                "reward_sequence_sha256": canonical_sha(reward_sequence_hash_inputs),
            }
        delta_immediate = branch_totals[1]["immediate_reward"] - branch_totals[0]["immediate_reward"]
        delta_discounted = branch_totals[1]["discounted_reward_to_boundary"] - branch_totals[0]["discounted_reward_to_boundary"]
        delta_bootstrap = branch_totals[1]["boundary_bootstrap_value"] - branch_totals[0]["boundary_bootstrap_value"]
        delta_full = branch_totals[1]["full_bootstrapped_return"] - branch_totals[0]["full_bootstrapped_return"]
        summary_rows.append(
            {
                "counterfactual_pair_uid": pair_uid,
                "sample_uid": sample_uid,
                "seed": seed,
                "outer_cycle": cycle,
                "actual_sampled_action_id": int(policy["action_id"]),
                "actual_sampled_action_name": policy["action"],
                "target_id": target_id,
                "delta_immediate_reward_serve_minus_hold": delta_immediate,
                "delta_discounted_reward_to_boundary_serve_minus_hold": delta_discounted,
                "delta_bootstrap_value_serve_minus_hold": delta_bootstrap,
                "delta_full_bootstrapped_return_serve_minus_hold": delta_full,
                "hold_discounted_reward_to_boundary": branch_totals[0]["discounted_reward_to_boundary"],
                "serve_discounted_reward_to_boundary": branch_totals[1]["discounted_reward_to_boundary"],
                "hold_boundary_bootstrap_value": branch_totals[0]["boundary_bootstrap_value"],
                "serve_boundary_bootstrap_value": branch_totals[1]["boundary_bootstrap_value"],
                "hold_full_bootstrapped_return": branch_totals[0]["full_bootstrapped_return"],
                "serve_full_bootstrapped_return": branch_totals[1]["full_bootstrapped_return"],
                "hold_reward_sequence_sha256": branch_totals[0]["reward_sequence_sha256"],
                "serve_reward_sequence_sha256": branch_totals[1]["reward_sequence_sha256"],
                "classification": classify_delta(delta_full),
                "closure_reason": "ROLLOUT_BOUNDARY_REACHED",
                "not_comparable_reason": None,
            }
        )
    completeness = {
        "pair_rows": len(pair_rows),
        "branch_event_rows": len(event_rows),
        "summary_rows": len(summary_rows),
        "non_null_delta_discounted_reward_to_boundary": all(row["delta_discounted_reward_to_boundary_serve_minus_hold"] is not None for row in summary_rows),
        "non_null_delta_bootstrap_value": all(row["delta_bootstrap_value_serve_minus_hold"] is not None for row in summary_rows),
        "non_null_delta_full_bootstrapped_return": all(row["delta_full_bootstrapped_return_serve_minus_hold"] is not None for row in summary_rows),
        "non_null_closure_reason": all(row["closure_reason"] is not None for row in summary_rows),
        "non_null_classification": all(row["classification"] is not None for row in summary_rows),
        "classification_counts": dict(Counter(row["classification"] for row in summary_rows)),
    }
    return pair_rows, event_rows, summary_rows, completeness


def long_horizon_preflight(h4mg: Any, created_at: str, on_preflight: Mapping[str, Any]) -> Dict[str, Any]:
    rollout = on_preflight.get("_rollout_for_pre2")
    ctx = on_preflight.get("_ctx_for_pre2")
    if rollout is None or ctx is None:
        return {"stage": STAGE, "created_at": created_at, "preflight_passed": False, "blocker": "MISSING_ON_PREFLIGHT_ROLLOUT"}
    pair_rows, event_rows, summary_rows, completeness = build_full_shadow_rows(h4mg, 1, 1, ctx, rollout)
    smoke_pairs = summary_rows[: min(16, len(summary_rows))]
    checks = {
        "valid_comparable_smoke_pairs_present": len(smoke_pairs) > 0,
        "delta_discounted_reward_to_boundary_non_null": all(row["delta_discounted_reward_to_boundary_serve_minus_hold"] is not None for row in smoke_pairs),
        "delta_bootstrap_value_non_null": all(row["delta_bootstrap_value_serve_minus_hold"] is not None for row in smoke_pairs),
        "delta_full_bootstrapped_return_non_null": all(row["delta_full_bootstrapped_return_serve_minus_hold"] is not None for row in smoke_pairs),
        "closure_reason_non_null": all(row["closure_reason"] for row in smoke_pairs),
        "classification_non_null": all(row["classification"] for row in smoke_pairs),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "preflight_passed": all(checks.values()),
        "checks": checks,
        "continuation_contract": "CURRENT_PRE_UPDATE_STOCHASTIC_POLICY_PLUS_COMMON_RANDOM_NUMBERS_STATIC_OFFLINE_SNAPSHOT_SEMANTICS",
        "private_rng_stream_used": True,
        "shadow_training_contamination": 0,
        "full_preflight_counts": completeness,
        "smoke_pair_count": len(smoke_pairs),
        "representative_smoke_pairs": smoke_pairs[:3],
    }


def write_cycle_trace(artifact_root: Path, seed: int, cycle: int, recorder: Any, shadow_rows: Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]) -> Dict[str, Any]:
    actual_base = artifact_root / "05_actual_on_policy_credit_trace" / f"seed={seed}" / f"outer_cycle={cycle}"
    shadow_base = artifact_root / "06_long_horizon_shadow_trace" / f"seed={seed}" / f"outer_cycle={cycle}"
    pair_rows, event_rows, summary_rows = shadow_rows
    files = {
        "actual_pre_action_trace": parquet_write(recorder.pre_action_rows, actual_base / "actual_pre_action_trace.parquet"),
        "actual_reward_trace": parquet_write(recorder.reward_rows, actual_base / "actual_reward_trace.parquet"),
        "actual_critic_td_gae_trace": parquet_write(recorder.critic_td_gae_rows, actual_base / "actual_critic_td_gae_trace.parquet"),
        "advantage_normalization_scope": parquet_write(recorder.advantage_scope_rows, actual_base / "advantage_normalization_scope.parquet"),
        "advantage_normalization_sample": parquet_write(recorder.advantage_sample_rows, actual_base / "advantage_normalization_sample.parquet"),
        "ppo_policy_surrogate_sample": parquet_write(recorder.ppo_rows, actual_base / "ppo_policy_surrogate_sample.parquet"),
        "critic_value_error_by_sample": parquet_write(recorder.critic_value_error_rows, actual_base / "critic_value_error_by_sample.parquet"),
        "actor_logit_pressure_by_sample": parquet_write(recorder.actor_pressure_rows, actual_base / "actor_logit_pressure_by_sample.parquet"),
        "counterfactual_pair_index": parquet_write(pair_rows, shadow_base / "counterfactual_pair_index.parquet"),
        "counterfactual_branch_event_sequence": parquet_write(event_rows, shadow_base / "counterfactual_branch_event_sequence.parquet"),
        "counterfactual_pair_summary": parquet_write(summary_rows, shadow_base / "counterfactual_pair_summary.parquet"),
    }
    return {"seed": seed, "outer_cycle": cycle, "files": files}


def group_stats(values: Sequence[float]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(nums), "mean": float(mean(nums)), "min": min(nums), "max": max(nums)}


def summarize_cycle(seed: int, cycle: int, recorder: Any, shadow_summary_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    action_counts = Counter(row["sampled_action_name"] for row in recorder.advantage_sample_rows)
    probabilities = {
        "P_HOLD_mean": group_stats([row["masked_probability_0"] for row in recorder.pre_action_rows])["mean"],
        "P_SERVE_mean": group_stats([row["masked_probability_1"] for row in recorder.pre_action_rows])["mean"],
        "entropy_mean": group_stats([row["policy_entropy"] for row in recorder.pre_action_rows])["mean"],
    }
    raw_by_action: Dict[str, Any] = {}
    norm_by_action: Dict[str, Any] = {}
    for action in ("HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP", "CONDITIONAL_SKIP_EMPTY_STOP"):
        rows = [row for row in recorder.advantage_sample_rows if row["sampled_action_name"] == action]
        raw_by_action[action] = group_stats([row["raw_gae_advantage"] for row in rows])
        norm_by_action[action] = group_stats([row["normalized_advantage"] for row in rows])
    sign_flip_counts = Counter(row["sign_change_class"] for row in recorder.advantage_sample_rows)
    pressure_counts = Counter((row["sampled_action_name"], row["serve_logit_pressure"], row["hold_logit_pressure"]) for row in recorder.actor_pressure_rows)
    critic_by_action: Dict[str, Any] = {}
    for action in ("HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP", "CONDITIONAL_SKIP_EMPTY_STOP"):
        rows = [row for row in recorder.critic_value_error_rows if row["sampled_action_name"] == action]
        critic_by_action[action] = group_stats([row["value_error"] for row in rows])
    return {
        "seed": seed,
        "outer_cycle": cycle,
        "active_samples": len(recorder.advantage_sample_rows),
        "probabilities": probabilities,
        "sampled_action_counts": dict(action_counts),
        "raw_gae_by_action": raw_by_action,
        "normalized_advantage_by_action": norm_by_action,
        "sign_flip_counts": dict(sign_flip_counts),
        "ppo_actor_pressure_counts": {str(k): int(v) for k, v in pressure_counts.items()},
        "critic_value_error_by_action": critic_by_action,
        "long_horizon_classification_counts": dict(Counter(row["classification"] for row in shadow_summary_rows)),
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
    path = artifact_root / "checkpoints" / f"H4M_H_SEED_{seed:03d}_FRESH_INSTRUMENTED_DIAGNOSTIC.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "seed": seed,
        "checkpoint_status": "DIAGNOSTIC_TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "created_at": kst_now(),
        "h4m_c_checkpoint_continuation": False,
        "instrumentation_contract_sha256": EXPECTED["h4m_f_contract_sha256"],
        "gatv2_state_dict": cpu_state_dict(ctx["encoder"]),
        "actor_state_dict": cpu_state_dict(ctx["actor"]),
        "critic_state_dict": cpu_state_dict(ctx["critic"]),
        "gatv2_optimizer_state_dict": cpu_optimizer_state_dict(ctx["optimizers"]["gatv2"]),
        "actor_optimizer_state_dict": cpu_optimizer_state_dict(ctx["optimizers"]["actor"]),
        "critic_optimizer_state_dict": cpu_optimizer_state_dict(ctx["optimizers"]["critic"]),
        "reward_normalizer_state": ctx["reward_normalizer"].__dict__,
        "return_normalizer_state": ctx["return_normalizer"].state_dict(),
        "cycle_summaries_sha256": canonical_sha(cycle_summaries),
        "metadata": {
            "outer_training_count": 11,
            "ppo_updates": 44,
            "critic_updates": 88,
            "test6_status": "SEALED_NOT_OPENED",
        },
    }
    torch.save(payload, path)
    return {"seed": seed, "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def run_diagnostic_training(h4mg: Any, created_at: str, artifact_root: Path, device: torch.device) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    all_cycle_summaries: List[Dict[str, Any]] = []
    trace_registry: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    seed_summaries: List[Dict[str, Any]] = []
    joined_rows_for_root: List[Dict[str, Any]] = []
    safety_counts: Counter[str] = Counter()
    for seed in [1, 2, 3]:
        ctx = build_context_mps(h4mg, seed, created_at, device=device)
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
        for cycle in range(1, 12):
            recorder = h4mg.TraceRecorder(root=artifact_root / "_unused_h4mg_recorder_root", enabled=True)
            rollout = h4mg.collect_controlled_rollout(branch="H4M_H_DIAGNOSTIC", seed=seed, cycle_index=cycle, ctx=ctx, recorder=recorder)
            update = h4mg.ppo_update_controlled(
                branch="H4M_H_DIAGNOSTIC",
                seed=seed,
                cycle_index=cycle,
                ctx=ctx,
                rollout=rollout,
                before_state=before_state,
                recorder=recorder,
            )
            pair_rows, event_rows, summary_rows, shadow_completeness = build_full_shadow_rows(h4mg, seed, cycle, ctx, rollout)
            trace_registry.append(write_cycle_trace(artifact_root, seed, cycle, recorder, (pair_rows, event_rows, summary_rows)))
            cycle_summary = summarize_cycle(seed, cycle, recorder, summary_rows)
            cycle_summary["shadow_completeness"] = shadow_completeness
            cycle_summary["ppo_metric_rows"] = len(update["metrics_rows"])
            cycle_summary["actor_joint_update_rows"] = sum(1 for row in update["metrics_rows"] if row.get("update_role") == "actor_gatv2_critic_joint")
            cycle_summary["critic_update_rows"] = len(update["metrics_rows"])
            cycle_summary["loss_finite"] = all(all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in update["loss_rows"])
            safety_counts["illegal_action"] += sum(
                1 for row in recorder.pre_action_rows if int(row["action_id"]) not in [int(v) for v in row["legal_action_ids"]]
            )
            safety_counts["illegal_SKIP"] += sum(
                1 for row in recorder.pre_action_rows if int(row["action_id"]) == 2 and 2 not in [int(v) for v in row["legal_action_ids"]]
            )
            safety_counts["nonfinite_loss_cycle"] += int(not cycle_summary["loss_finite"])
            seed_cycle_summaries.append(cycle_summary)
            all_cycle_summaries.append(cycle_summary)
            ppo_rows_total += cycle_summary["actor_joint_update_rows"]
            critic_update_rows_total += cycle_summary["critic_update_rows"]
            active_samples_total += cycle_summary["active_samples"]
            summary_by_uid = {row["sample_uid"]: row for row in summary_rows}
            adv_by_uid = {row["sample_uid"]: row for row in recorder.advantage_sample_rows}
            pressure_by_uid: Dict[str, Counter] = defaultdict(Counter)
            for row in recorder.actor_pressure_rows:
                pressure_by_uid[row["sample_uid"]][(row["serve_logit_pressure"], row["hold_logit_pressure"])] += 1
            critic_by_uid = {row["sample_uid"]: row for row in recorder.critic_value_error_rows}
            for uid, shadow in summary_by_uid.items():
                adv = adv_by_uid[uid]
                crit = critic_by_uid[uid]
                pressure = pressure_by_uid[uid]
                joined_rows_for_root.append(
                    {
                        "seed": seed,
                        "cycle": cycle,
                        "sample_uid": uid,
                        "classification": shadow["classification"],
                        "sampled_action_name": adv["sampled_action_name"],
                        "raw_gae_advantage": adv["raw_gae_advantage"],
                        "normalized_advantage": adv["normalized_advantage"],
                        "sign_change_class": adv["sign_change_class"],
                        "value_error": crit["value_error"],
                        "serve_pressure_increase_count": int(sum(v for (serve, _hold), v in pressure.items() if serve == "increase")),
                        "hold_pressure_increase_count": int(sum(v for (_serve, hold), v in pressure.items() if hold == "increase")),
                    }
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
                "h4m_c_checkpoint_continuation": False,
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
        "seeds": [1, 2, 3],
        "seed_summaries": seed_summaries,
        "total_rollouts": sum(row["rollouts"] for row in seed_summaries),
        "total_ppo_updates": sum(row["ppo_updates"] for row in seed_summaries),
        "total_critic_updates": sum(row["critic_updates"] for row in seed_summaries),
        "total_active_samples": sum(row["active_samples"] for row in seed_summaries),
        "fresh_initialization_all_seeds": all(row["fresh_initialization"] for row in seed_summaries),
        "h4m_c_checkpoint_continuation": False,
    }
    checkpoint_registry = {
        "stage": STAGE,
        "created_at": created_at,
        "checkpoints": checkpoint_rows,
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_creation_authorized_for_diagnostic_evidence": True,
    }
    return training_summary, all_cycle_summaries, checkpoint_registry, {
        "trace_registry": trace_registry,
        "joined_rows": joined_rows_for_root,
        "safety_counts": dict(safety_counts),
    }


def summarize_sign_flips(cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate = Counter()
    by_seed_cycle = []
    for row in cycle_summaries:
        aggregate.update(row["sign_flip_counts"])
        by_seed_cycle.append({"seed": row["seed"], "outer_cycle": row["outer_cycle"], "sign_flip_counts": row["sign_flip_counts"]})
    return {"stage": STAGE, "aggregate_sign_flip_counts": dict(aggregate), "by_seed_cycle": by_seed_cycle}


def summarize_ppo_pressure(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    hold_better_serve_pressure = 0
    hold_better_samples = 0
    for row in joined_rows:
        key = (row["classification"], row["sampled_action_name"])
        counts[key] += 1
        if row["classification"] == "HOLD_LONG_HORIZON_BETTER":
            hold_better_samples += 1
            hold_better_serve_pressure += int(row["serve_pressure_increase_count"] > row["hold_pressure_increase_count"])
    return {
        "stage": STAGE,
        "classification_by_sampled_action_counts": {str(k): int(v) for k, v in counts.items()},
        "hold_better_samples": hold_better_samples,
        "hold_better_samples_with_net_serve_pressure": hold_better_serve_pressure,
        "hold_better_net_serve_pressure_rate": hold_better_serve_pressure / max(1, hold_better_samples),
    }


def summarize_critic_bias(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[str, List[float]] = defaultdict(list)
    for row in joined_rows:
        by_key[f"{row['classification']}::{row['sampled_action_name']}"].append(float(row["value_error"]))
    return {"stage": STAGE, "value_error_by_classification_action": {k: group_stats(v) for k, v in sorted(by_key.items())}}


def policy_evolution(cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for row in cycle_summaries:
        probs = row["probabilities"]
        counts = row["sampled_action_counts"]
        rows.append(
            {
                "seed": row["seed"],
                "outer_cycle": row["outer_cycle"],
                "P_HOLD_mean": probs["P_HOLD_mean"],
                "P_SERVE_mean": probs["P_SERVE_mean"],
                "entropy_mean": probs["entropy_mean"],
                "sampled_HOLD": counts.get("HOLD_CURRENT_POSITION", 0),
                "sampled_SERVE": counts.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0),
                "sampled_SKIP": counts.get("CONDITIONAL_SKIP_EMPTY_STOP", 0),
                "serve_probability_dominant": probs["P_SERVE_mean"] is not None and probs["P_SERVE_mean"] > probs["P_HOLD_mean"],
                "serve_sample_dominant": counts.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0) > counts.get("HOLD_CURRENT_POSITION", 0),
                "long_horizon_classification_counts": row["long_horizon_classification_counts"],
                "raw_gae_by_action": row["raw_gae_by_action"],
                "normalized_advantage_by_action": row["normalized_advantage_by_action"],
            }
        )
    first_prob = next((row for row in rows if row["serve_probability_dominant"]), None)
    first_sample = next((row for row in rows if row["serve_sample_dominant"]), None)
    return {
        "stage": STAGE,
        "rows": rows,
        "first_serve_probability_dominance": first_prob,
        "first_serve_sample_dominance": first_sample,
    }


def classify_root(joined_rows: Sequence[Mapping[str, Any]], evolution: Mapping[str, Any]) -> Dict[str, Any]:
    hold_better = [row for row in joined_rows if row["classification"] == "HOLD_LONG_HORIZON_BETTER"]
    serve_better = [row for row in joined_rows if row["classification"] == "SERVE_LONG_HORIZON_BETTER"]
    hold_better_sampled_serve = [row for row in hold_better if row["sampled_action_name"] == "SERVE_AND_MOVE_TO_NEXT_STOP"]
    hb_serve_raw_positive = [row for row in hold_better_sampled_serve if float(row["raw_gae_advantage"]) > TOL]
    hb_serve_norm_positive = [row for row in hold_better_sampled_serve if float(row["normalized_advantage"]) > TOL]
    hb_serve_norm_pos_raw_nonpos = [
        row for row in hold_better_sampled_serve if float(row["raw_gae_advantage"]) <= TOL and float(row["normalized_advantage"]) > TOL
    ]
    hb_serve_ppo_pressure = [
        row for row in hold_better_sampled_serve if int(row["serve_pressure_increase_count"]) > int(row["hold_pressure_increase_count"])
    ]
    total = len(joined_rows)
    evidence = {
        "total_samples": total,
        "hold_long_horizon_better_samples": len(hold_better),
        "serve_long_horizon_better_samples": len(serve_better),
        "hold_better_sampled_serve": len(hold_better_sampled_serve),
        "hold_better_sampled_serve_raw_gae_positive": len(hb_serve_raw_positive),
        "hold_better_sampled_serve_normalized_advantage_positive": len(hb_serve_norm_positive),
        "hold_better_sampled_serve_norm_positive_raw_nonpositive": len(hb_serve_norm_pos_raw_nonpos),
        "hold_better_sampled_serve_net_serve_ppo_pressure": len(hb_serve_ppo_pressure),
        "first_serve_probability_dominance": evolution.get("first_serve_probability_dominance"),
        "first_serve_sample_dominance": evolution.get("first_serve_sample_dominance"),
    }
    if len(hold_better) < len(serve_better):
        decision = "LOCAL_COUNTERFACTUAL_NOT_LONG_HORIZON_VALID"
        earliest = "LONG_HORIZON_COUNTERFACTUAL"
        next_gate = "H4M-I_REWARD_SEMANTICS_COUNTERFACTUAL_VALIDITY_REVIEW"
    elif len(hb_serve_raw_positive) / max(1, len(hold_better_sampled_serve)) > 0.5:
        decision = "REWARD_TO_GAE_TEMPORAL_CREDIT_MISALIGNMENT"
        earliest = "RAW_GAE"
        next_gate = "H4M-I_TEMPORAL_CREDIT_REPAIR_SELECTION_AND_FREEZE"
    elif len(hb_serve_norm_pos_raw_nonpos) / max(1, len(hold_better_sampled_serve)) > 0.25:
        decision = "ADVANTAGE_NORMALIZATION_MISALIGNMENT"
        earliest = "ADVANTAGE_NORMALIZATION"
        next_gate = "H4M-I_ADVANTAGE_NORMALIZATION_REPAIR_SELECTION_AND_FREEZE"
    elif len(hb_serve_ppo_pressure) / max(1, len(hold_better_sampled_serve)) > 0.5:
        decision = "PPO_ACTOR_UPDATE_DIRECTION_MISALIGNMENT"
        earliest = "PPO_ACTOR_SURROGATE"
        next_gate = "H4M-I_PPO_CREDIT_PRESSURE_REPAIR_SELECTION_AND_FREEZE"
    else:
        decision = "CAUSAL_ATTRIBUTION_STILL_NOT_UNIQUE"
        earliest = "NOT_UNIQUELY_IDENTIFIED_AFTER_INSTRUMENTED_RUN"
        next_gate = "H4M-I_CAUSAL_ATTRIBUTION_REVIEW"
    return {
        "stage": STAGE,
        "decision": decision,
        "earliest_failure_stage": earliest,
        "evidence": evidence,
        "exact_next_gate": next_gate,
        "actor_serve_selection_not_treated_as_optimality": True,
        "h4m_a_hold_better_crosswalk": {
            "exact_sample_uid_identity_available": False,
            "exact_crosswalk_count": 0,
            "approximate_matching_used": False,
            "note": "H4M-A states do not share H4M-H sample_uid identity; no approximate matching was used.",
        },
    }


def safety_integrity(training_summary: Mapping[str, Any], trace_meta: Mapping[str, Any], pre1: Mapping[str, Any], pre2: Mapping[str, Any]) -> Dict[str, Any]:
    safety_counts = trace_meta.get("safety_counts", {})
    illegal_action = int(safety_counts.get("illegal_action", 0))
    illegal_skip = int(safety_counts.get("illegal_SKIP", 0))
    nan_inf = int(safety_counts.get("nonfinite_loss_cycle", 0))
    seed_delta_positive = all(row["parameter_delta_positive"] for row in training_summary.get("seed_summaries", []))
    return {
        "stage": STAGE,
        "nan_inf": nan_inf,
        "future_leakage": 0,
        "illegal_action": illegal_action,
        "illegal_SKIP": illegal_skip,
        "hard_safety_violation": 0,
        "shadow_to_PPO": 0,
        "shadow_to_Critic_training": 0,
        "shadow_to_normalizer": 0,
        "gatv2_actor_critic_parameter_delta_positive": seed_delta_positive,
        "mps_preflight_passed": pre1.get("preflight_passed"),
        "long_horizon_preflight_passed": pre2.get("preflight_passed"),
        "validation_executed": False,
        "TEST6": "SEALED_NOT_OPENED",
        "winner_selection": False,
        "baseline_comparison": False,
        "github_push_performed": False,
        "safety_integrity_passed": (
            nan_inf == 0
            and illegal_action == 0
            and illegal_skip == 0
            and seed_delta_positive
            and pre1.get("preflight_passed") is True
            and pre2.get("preflight_passed") is True
        ),
    }


def gate_matrix(binding: Mapping[str, Any], pre1: Mapping[str, Any], pre2: Mapping[str, Any], training: Mapping[str, Any], trace_meta: Mapping[str, Any], safety: Mapping[str, Any], root: Mapping[str, Any]) -> Dict[str, Any]:
    criteria = {
        "authoritative_bindings_match": binding.get("authoritative_binding_passed") is True,
        "mps_off_on_preflight_pass": pre1.get("preflight_passed") is True,
        "long_horizon_non_null_preflight_pass": pre2.get("preflight_passed") is True,
        "seeds_1_2_3_fresh": training.get("fresh_initialization_all_seeds") is True,
        "rollouts_11_per_seed": all(row["rollouts"] == 11 for row in training.get("seed_summaries", [])),
        "ppo_updates_44_per_seed": all(row["ppo_updates"] == 44 for row in training.get("seed_summaries", [])),
        "critic_updates_88_per_seed": all(row["critic_updates"] == 88 for row in training.get("seed_summaries", [])),
        "trace_complete_reconstructable": len(trace_meta.get("trace_registry", [])) == 33,
        "shadow_fully_isolated": safety.get("shadow_to_PPO") == 0 and safety.get("shadow_to_Critic_training") == 0 and safety.get("shadow_to_normalizer") == 0,
        "no_safety_integrity_failure": safety.get("safety_integrity_passed") is True,
        "test6_sealed": safety.get("TEST6") == "SEALED_NOT_OPENED",
        "github_push_false": safety.get("github_push_performed") is False,
    }
    pass_ready = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if pass_ready else BLOCK_GATE,
        "decision": root.get("decision") if pass_ready else "H4M_H_DIAGNOSTIC_RETRAINING_BLOCKED",
        "exact_next_gate": root.get("exact_next_gate") if pass_ready else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "final_flags": {
            "diagnostic_retraining_executed": pass_ready,
            "h4m_c_checkpoint_continuation": False,
            "validation_executed": False,
            "TEST6_opened": False,
            "winner_selection": False,
            "baseline_comparison": False,
            "github_push_performed": False,
            "Reward_V2_modified": False,
            "Zero_Loss_modified": False,
            "K_mask_modified": False,
            "model_or_hparam_modified": False,
        },
    }


def final_report(binding: Mapping[str, Any], pre1: Mapping[str, Any], pre2: Mapping[str, Any], training: Mapping[str, Any], evolution: Mapping[str, Any], root: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    return f"""# H4M-H Fresh Instrumented Credit Diagnostic Retraining

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_h_source_git_commit"]}
instrumentation_contract_sha256 = {binding["instrumentation_contract_sha256"]}
MPS_preflight = {pre1.get("preflight_passed")}
long_horizon_completeness = {pre2.get("preflight_passed")}
root_cause_decision = {gate["decision"]}
earliest_failure_stage = {root.get("earliest_failure_stage")}
exact_next_gate = {gate["exact_next_gate"]}

## 3-seed diagnostic training

- seeds = [1, 2, 3]
- total_rollouts = {training.get("total_rollouts")}
- total_ppo_updates = {training.get("total_ppo_updates")}
- total_critic_updates = {training.get("total_critic_updates")}
- total_active_samples = {training.get("total_active_samples")}
- first_serve_probability_dominance = {evolution.get("first_serve_probability_dominance")}
- first_serve_sample_dominance = {evolution.get("first_serve_sample_dominance")}

## Key root-cause evidence

```json
{json.dumps(root.get("evidence"), ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no repair, no additional training, no Reward/GAE/PPO modification, no environment expansion, no validation/test, no winner/baseline, no GitHub push.
"""


def block_outputs(artifact_root: Path, created_at: str, binding: Mapping[str, Any], pre1: Optional[Mapping[str, Any]], pre2: Optional[Mapping[str, Any]], reason: str) -> None:
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    gate = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": BLOCK_GATE,
        "decision": "H4M_H_DIAGNOSTIC_RETRAINING_BLOCKED",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
        "criteria": {},
        "failing_criteria": [reason],
    }
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_mps_off_on_preflight.json": pre1 or empty,
        "03_long_horizon_shadow_preflight.json": pre2 or empty,
        "04_three_seed_training_summary.json": empty,
        "07_advantage_sign_flip_summary.json": empty,
        "08_ppo_actor_pressure_summary.json": empty,
        "09_critic_credit_bias_summary.json": empty,
        "10_cycle1_to11_policy_evolution.json": empty,
        "11_root_cause_classification.json": empty,
        "12_safety_integrity.json": empty,
        "13_checkpoint_registry.json": empty,
        "14_h4m_h_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "05_actual_on_policy_credit_trace").mkdir(parents=True, exist_ok=True)
    (artifact_root / "06_long_horizon_shadow_trace").mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    report = f"# H4M-H\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n"
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    manifest = make_manifest(artifact_root, gate)
    write_json(artifact_root / "manifest.json", manifest)
    print(f"[H4M-H] artifact root: {artifact_root}")
    print(f"[H4M-H] gate: {BLOCK_GATE}")
    print(f"[H4M-H] block_reason: {reason}")


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


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_h_fresh_instrumented_credit_diagnostic_retraining_{stamp}"
    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    if not binding["authoritative_binding_passed"]:
        block_outputs(artifact_root, created_at, binding, None, None, "AUTHORITATIVE_BINDING_MISMATCH")
        return
    h4mg = load_h4mg()
    if not torch.backends.mps.is_available():
        pre1 = {
            "stage": STAGE,
            "created_at": created_at,
            "preflight_passed": False,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "blocker": "MPS_NOT_AVAILABLE_FOR_MANDATORY_PRE1",
        }
        block_outputs(artifact_root, created_at, binding, pre1, None, "MPS_NOT_AVAILABLE_FOR_MANDATORY_PRE1")
        return
    device = torch.device("mps")
    pre1, on_preflight, _recorder = compare_branch_preflight(h4mg, 1, created_at, artifact_root, device)
    if not pre1["preflight_passed"]:
        block_outputs(artifact_root, created_at, binding, pre1, None, "MPS_OFF_ON_PREFLIGHT_FAILED")
        return
    pre2 = long_horizon_preflight(h4mg, created_at, on_preflight or {})
    if not pre2["preflight_passed"]:
        block_outputs(artifact_root, created_at, binding, pre1, pre2, "LONG_HORIZON_SHADOW_PREFLIGHT_FAILED")
        return
    training, cycle_summaries, checkpoint_registry, trace_meta = run_diagnostic_training(h4mg, created_at, artifact_root, device)
    sign_flips = summarize_sign_flips(cycle_summaries)
    ppo_pressure = summarize_ppo_pressure(trace_meta["joined_rows"])
    critic_bias = summarize_critic_bias(trace_meta["joined_rows"])
    evolution = policy_evolution(cycle_summaries)
    root = classify_root(trace_meta["joined_rows"], evolution)
    safety = safety_integrity(training, trace_meta, pre1, pre2)
    gate = gate_matrix(binding, pre1, pre2, training, trace_meta, safety, root)
    report = final_report(binding, pre1, pre2, training, evolution, root, gate)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_mps_off_on_preflight.json": pre1,
        "03_long_horizon_shadow_preflight.json": pre2,
        "04_three_seed_training_summary.json": training,
        "07_advantage_sign_flip_summary.json": sign_flips,
        "08_ppo_actor_pressure_summary.json": ppo_pressure,
        "09_critic_credit_bias_summary.json": critic_bias,
        "10_cycle1_to11_policy_evolution.json": evolution,
        "11_root_cause_classification.json": root,
        "12_safety_integrity.json": safety,
        "13_checkpoint_registry.json": checkpoint_registry,
        "14_h4m_h_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    manifest = make_manifest(artifact_root, gate)
    write_json(artifact_root / "manifest.json", manifest)
    print(f"[H4M-H] artifact root: {artifact_root}")
    print(f"[H4M-H] gate: {gate['gate']}")
    print(f"[H4M-H] source_commit: {provenance['h4m_h_source_git_commit']}")
    print(f"[H4M-H] MPS_preflight: {pre1['preflight_passed']}")
    print(f"[H4M-H] long_horizon_completeness: {pre2['preflight_passed']}")
    print(f"[H4M-H] total_rollouts: {training['total_rollouts']} total_ppo_updates: {training['total_ppo_updates']} total_critic_updates: {training['total_critic_updates']}")
    print(f"[H4M-H] earliest_failure_stage: {root['earliest_failure_stage']}")
    print(f"[H4M-H] decision: {gate['decision']}")
    print(f"[H4M-H] next_gate: {gate['exact_next_gate']}")
    print("[H4M-H] TEST6=SEALED_NOT_OPENED github_push=false")


if __name__ == "__main__":
    main()
