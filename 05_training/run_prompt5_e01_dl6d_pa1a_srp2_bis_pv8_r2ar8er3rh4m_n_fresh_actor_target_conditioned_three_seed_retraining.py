#!/usr/bin/env python3
"""H4M-N fresh actor-target-conditioned three-seed retraining.

Runs the H4M-M validated Actor target direct-conditioning repair through the
frozen seeds 1/2/3 × 11-cycle TRAIN44 schedule. This stage performs training
and diagnostic evaluation only: no additional repair, no validation/TEST6, no
winner/baseline selection, and no GitHub push.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import py_compile
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-N"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_N_FRESH_ACTOR_TARGET_CONDITIONED_THREE_SEED_RETRAINING_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_N_FRESH_ACTOR_TARGET_CONDITIONED_THREE_SEED_RETRAINING_FAILED"
PASS_DECISION = "H4M_N_FRESH_ACTOR_TARGET_CONDITIONED_THREE_SEED_RETRAINING_EXECUTION_COMPLETE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MK_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
H4M_C_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_c_fresh_extended_budget_three_seed_retraining.py"
H4K_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py"
DL4_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
MAPPO_SOURCE = TRAINING_ROOT / "mappo_runner.py"
REWARD_SOURCE = TRAINING_ROOT / "rewards/mappo_reward_v1.py"
OBS_REPAIR_SOURCE = TRAINING_ROOT / "observation_target_context_repair.py"
DL1_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"

H4M_M_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_m_actor_target_sensitivity_repair_implementation_equivalence_validation_20260817_114202+0900"
H4M_L_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_l_target_context_repair_outcome_review_next_decision_20260817_105652+0900"
H4M_K_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_k_fresh_target_context_repaired_three_seed_retraining_20260817_100428+0900"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"

EXPECTED = {
    "h4m_m_gate": PASS_GATE.replace(
        "H4M_N_FRESH_ACTOR_TARGET_CONDITIONED_THREE_SEED_RETRAINING_COMPLETE",
        "H4M_M_ACTOR_TARGET_SENSITIVITY_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE",
    ),
    "h4m_m_source_commit": "8390bffd96e161ac7113eb2a58411414e9dc9529",
    "h4m_l_repair_contract_sha256": "5637381f6f450cbda6f01a76126a5940b31984edf6b782d71ca224c4accf8754",
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
    "actor_embedding_dim": 128,
    "actor_target_context_dim": 3,
    "actor_conditioned_input_dim": 131,
}

ACTION_HOLD = "HOLD_CURRENT_POSITION"
ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
ACTION_SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
ACTION_NAMES = {0: ACTION_HOLD, 1: ACTION_SERVE, 2: ACTION_SKIP}
CLASS_HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
CLASS_SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
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
    "05_actor_target_utilization.json",
    "06_target_conditioned_credit.json",
    "07_h4mh_h4mk_h4mn_comparison.json",
    "08_long_horizon_shadow_summary.json",
    "09_policy_collapse_audit.json",
    "10_root_decision.json",
    "11_checkpoint_registry.json",
    "12_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]

DECISION_CLASSES = {
    "A": "ACTOR_DIRECT_CONDITIONING_CONFIRMED_STATE_CONDITIONAL_DISCRIMINATION",
    "B": "ACTOR_DIRECT_CONDITIONING_PARTIAL_IMPROVEMENT",
    "C": "ACTOR_DIRECT_CONDITIONING_FAILED_POLICY_COLLAPSE_PERSISTS",
    "D": "GLOBAL_COLLAPSE_REVERSED_TO_HOLD",
    "E": "TARGET_CONDITIONED_CREDIT_OR_GRADIENT_IMBALANCE_REMAINS",
    "F": "CAUSAL_EFFECT_NOT_UNIQUE",
}


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


def tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def numeric_stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    mu = float(mean(nums))
    var = sum((x - mu) ** 2 for x in nums) / len(nums)
    return {
        "count": len(nums),
        "mean": mu,
        "median": float(median(nums)),
        "min": min(nums),
        "max": max(nums),
        "std": math.sqrt(var),
    }


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else float(numerator) / float(denominator)


def py_compile_audit() -> Dict[str, Any]:
    rows = []
    passed = True
    with tempfile.TemporaryDirectory(prefix="h4m_n_pycompile_") as tmp:
        for rel in [SOURCE_REL]:
            source = PROJECT_ROOT / rel
            cfile = Path(tmp) / (rel.name + ".pyc")
            try:
                py_compile.compile(str(source), cfile=str(cfile), doraise=True)
                ok = True
                error = None
            except Exception as exc:
                ok = False
                error = repr(exc)
            passed = passed and ok
            rows.append({"source_rel": str(rel), "passed": ok, "error": error, "bytecode_target_outside_repo": str(cfile)})
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {
        "py_compile_passed": passed,
        "py_compile_rows": rows,
        "git_diff_cached_check_passed": cached.returncode == 0,
        "git_diff_cached_check_stdout": cached.stdout.strip(),
        "git_diff_cached_check_stderr": cached.stderr.strip(),
    }


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
        Path("05_training") / Path(H4MK_SOURCE).name,
        Path("05_training") / Path(H4MG_SOURCE).name,
        Path("05_training") / Path(DL1_SOURCE).name,
        Path("05_training") / "observation_target_context_repair.py",
        Path("05_training") / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_m_actor_target_sensitivity_repair_implementation_equivalence_validation.py",
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
        "h4m_n_source_git_commit": latest_source_commit,
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


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_static: Mapping[str, Any]) -> Dict[str, Any]:
    h4mm_gate = read_json(H4M_M_ROOT / "09_gate_matrix.json")
    h4mm_binding = read_json(H4M_M_ROOT / "01_authoritative_binding.json")
    h4mm_impl = read_json(H4M_M_ROOT / "02_actor_direct_conditioning_implementation.json")
    h4mm_grad = read_json(H4M_M_ROOT / "05_gradient_path_mps_smoke.json")
    h4mm_instr = read_json(H4M_M_ROOT / "07_instrumentation_compatibility.json")
    h4mm_ready = read_json(H4M_M_ROOT / "08_fresh_training_readiness.json")
    h4mm_manifest = read_json(H4M_M_ROOT / "manifest.json")
    h4ml_contract = read_json(H4M_L_ROOT / "09_h4m_l_repair_contract.json")
    h4mg_close_gate = read_json(H4M_G_CLOSE_ROOT / "14_gate_matrix.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    close_contract_sha = sha256_file(H4M_G_CLOSE_ROOT / "12_final_mps_equivalence_contract.json")
    upstream = h4mm_binding.get("sha_bindings", {})
    checks = {
        "source_only_commit_before_training": provenance.get("local_source_only_commit_created_before_training") is True,
        "py_compile_passed": compile_static.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_static.get("git_diff_cached_check_passed") is True,
        "h4m_m_gate_match": h4mm_gate.get("gate") == EXPECTED["h4m_m_gate"],
        "h4m_m_source_commit_match": h4mm_binding.get("source_provenance", {}).get("h4m_m_source_git_commit")
        == EXPECTED["h4m_m_source_commit"],
        "h4m_m_repair_decision_match": h4mm_gate.get("decision")
        == "PV8_ACTOR_TARGET_DIRECT_CONDITIONING_IMPLEMENTED_VALIDATED_READY_FOR_FRESH_THREE_SEED_RETRAINING",
        "h4m_l_repair_contract_sha_match": h4ml_contract.get("contract_sha256")
        == EXPECTED["h4m_l_repair_contract_sha256"]
        and upstream.get("h4m_l_repair_contract_sha256") == EXPECTED["h4m_l_repair_contract_sha256"],
        "actor_direct_conditioning_repair_active": h4mm_impl.get("actor_direct_conditioning_implementation_passed") is True
        and h4mm_impl.get("actor_input_schema", {}).get("conditioned_input_dim") == EXPECTED["actor_conditioned_input_dim"],
        "actor_input_128_plus_3_equals_131": h4mm_impl.get("actor_input_schema", {}).get("gatv2_agent_embedding_dim") == 128
        and h4mm_impl.get("actor_input_schema", {}).get("target_context_dim") == 3
        and h4mm_impl.get("actor_input_schema", {}).get("conditioned_input_dim") == 131,
        "h4m_i_observation_repair_sha_match": upstream.get("observation_repair_contract_sha256")
        == EXPECTED["h4m_i_repair_contract_sha256"],
        "mps_gradient_smoke_pass": h4mm_grad.get("mps_gradient_path_smoke_passed") is True,
        "instrumentation_compatibility_pass": h4mm_instr.get("instrumentation_compatibility_passed") is True,
        "fresh_training_readiness_pass": h4mm_ready.get("fresh_training_readiness_passed") is True,
        "active_mps_contract_sha_match_binding": upstream.get("active_mps_instrumentation_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "active_mps_contract_sha_match_gate": h4mg_close_gate.get("final_active_contract_sha256")
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "active_mps_contract_sha_match_file": close_contract_sha == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_g_close_active_contract_match_file": close_contract_sha
        == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_b_schedule_sha_match_binding": upstream.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "h4m_b_schedule_sha_match_direct": h4m_b_schedule.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": upstream.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "r3_split_sha_match": upstream.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "h4g_runtime_sha_match": upstream.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "zero_loss_adapter_sha_match": upstream.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "test6_sealed": h4mm_manifest.get("TEST6_opened") is False,
        "github_push_false": h4mm_manifest.get("github_push_performed") is False,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_m": str(H4M_M_ROOT),
            "h4m_l": str(H4M_L_ROOT),
            "h4m_k": str(H4M_K_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "source_provenance": provenance,
        "compile_static_checks": compile_static,
        "authoritative_binding_passed": all(checks.values()),
        "checks": checks,
        "bound_contract": {
            "active_repair": "ACTOR_TARGET_CONTEXT_DIRECT_CONDITIONING_REPAIR",
            "h4m_l_repair_contract_sha256": EXPECTED["h4m_l_repair_contract_sha256"],
            "observation_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
            "node_feature_dim": EXPECTED["node_feature_dim"],
            "actor_input_schema": {
                "gatv2_embedding_dim": 128,
                "target_context_dim": 3,
                "conditioned_input_dim": 131,
                "target_context_fields": TARGET_CONTEXT_FIELDS,
            },
            "fresh_initialization_required": True,
            "checkpoint_reuse_allowed": False,
        },
        "sha_bindings": {
            "h4m_m_source_commit": EXPECTED["h4m_m_source_commit"],
            "h4m_l_repair_contract_sha256": EXPECTED["h4m_l_repair_contract_sha256"],
            "h4m_i_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
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
            "observation_12d_modified": False,
            "actor_direct_conditioning_architecture_modified": False,
            "gatv2_or_critic_architecture_modified": False,
            "gae_ppo_normalization_modified": False,
            "hyperparameters_modified": False,
            "train44_agents_environment_modified": False,
            "instrumentation_contract_modified": False,
            "old_checkpoint_reuse": False,
            "validation_or_test6_executed": False,
            "github_push_performed": False,
        },
    }


def configure_h4mk_module(kmod: Any) -> Any:
    kmod.STAGE = STAGE
    kmod.PASS_GATE = PASS_GATE
    kmod.BLOCK_GATE = BLOCK_GATE
    kmod.SOURCE_REL = SOURCE_REL
    kmod.REQUIRED_ARTIFACTS = REQUIRED_ARTIFACTS
    kmod.EXPECTED = {
        "h4m_i_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
        "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha256": EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "seeds": EXPECTED["seeds"],
        "outer_training_count": EXPECTED["outer_training_count"],
        "ppo_updates_per_seed": EXPECTED["ppo_updates_per_seed"],
        "critic_updates_per_seed": EXPECTED["critic_updates_per_seed"],
        "active_samples_per_seed": EXPECTED["active_samples_per_seed"],
        "node_feature_dim": EXPECTED["node_feature_dim"],
    }
    kmod.build_context_mps = build_context_mps_h4mn
    kmod.save_seed_checkpoint = save_seed_checkpoint_h4mn
    return kmod


def build_context_mps_h4mn(h4mg: Any, seed: int, created_at: str, *, device: torch.device) -> Dict[str, Any]:
    h4m_c = h4mg.import_module_from_path(H4M_C_SOURCE, f"h4mn_h4m_c_{seed}_{time.time_ns()}")
    h4k = h4mg.import_module_from_path(H4K_SOURCE, f"h4mn_h4k_{seed}_{time.time_ns()}")
    dl4 = h4mg.import_module_from_path(DL4_SOURCE, f"h4mn_dl4_{seed}_{time.time_ns()}")
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    mappo_mod = h4mg.import_module_from_path(MAPPO_SOURCE, f"h4mn_mappo_{seed}_{time.time_ns()}")
    reward_mod = h4mg.import_module_from_path(REWARD_SOURCE, f"h4mn_reward_{seed}_{time.time_ns()}")
    observation_repair = h4mg.import_module_from_path(OBS_REPAIR_SOURCE, f"h4mn_observation_repair_{seed}_{time.time_ns()}")

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
        "actor_target_context_direct_conditioning_repair_active": True,
        "actor_target_context_direct_conditioning_repair_contract_sha256": EXPECTED["h4m_l_repair_contract_sha256"],
        "actor_target_context_source": "data.x[agent_id, -3:] existing pre-action 12D observation fields",
        "actor_target_context_dim": 3,
        "actor_conditioned_input_dim": 131,
        "actor_conditioned_input_schema": "concat(GATv2_agent_embedding_128d, r3_action_target_one_hot_3d)",
        "spec": spec,
        "started_at_perf": time.perf_counter(),
    }
    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(
        int(config["gatv2_hidden"]),
        int(config["action_dim"]),
        target_context_dim=int(config["actor_target_context_dim"]),
    ).to(device)
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


def save_seed_checkpoint_h4mn(artifact_root: Path, seed: int, ctx: Mapping[str, Any], cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path = artifact_root / "checkpoints" / f"H4M_N_SEED_{seed:03d}_FRESH_ACTOR_TARGET_CONDITIONED.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "seed": seed,
        "checkpoint_status": "FRESH_ACTOR_TARGET_CONDITIONED_TRAINING_COMPLETE_DIAGNOSTIC_EVALUATION_NOT_RELEASED",
        "created_at": kst_now(),
        "fresh_initialization": True,
        "old_9d_checkpoint_reuse": False,
        "h4m_h_or_h4m_k_checkpoint_continuation": False,
        "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "observation_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
        "actor_direct_conditioning_repair_contract_sha256": EXPECTED["h4m_l_repair_contract_sha256"],
        "node_feature_dim": int(ctx["config"]["node_feature_dim_after_observation_repair"]),
        "actor_target_context_dim": int(getattr(ctx["actor"], "target_context_dim", -1)),
        "actor_conditioned_input_dim": int(getattr(ctx["actor"], "actor_conditioned_input_dim", -1)),
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
    return {
        "seed": seed,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "actor_target_context_dim": payload["actor_target_context_dim"],
        "actor_conditioned_input_dim": payload["actor_conditioned_input_dim"],
    }


def trace_conditioned_input_audit(credit_summary: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    required_columns = [
        "actor_direct_conditioning_active",
        "actor_target_context_dim",
        "actor_target_context_fields",
        "actor_target_context_values",
        "actor_target_context_one_hot_sum",
        "actor_conditioned_input_shape",
        "actor_conditioned_input_schema",
    ]
    total_rows = 0
    for registry in credit_summary.get("trace_registry", []):
        meta = registry.get("files", {}).get("actual_pre_action_trace", {})
        path = Path(meta.get("path", ""))
        if not path.exists():
            rows.append({"seed": registry.get("seed"), "outer_cycle": registry.get("outer_cycle"), "path": str(path), "exists": False})
            continue
        frame = pd.read_parquet(path)
        total_rows += len(frame)
        present = {col: col in frame.columns for col in required_columns}

        def shape_ok(value: Any) -> bool:
            if isinstance(value, (list, tuple)):
                return list(value) == [131]
            return "131" in str(value)

        active_ok = bool(present["actor_direct_conditioning_active"] and frame["actor_direct_conditioning_active"].map(bool).all())
        dim_ok = bool(present["actor_target_context_dim"] and (frame["actor_target_context_dim"].astype(int) == 3).all())
        one_hot_ok = bool(
            present["actor_target_context_one_hot_sum"]
            and (frame["actor_target_context_one_hot_sum"].astype(float) == 1.0).all()
        )
        conditioned_shape_ok = bool(
            present["actor_conditioned_input_shape"] and frame["actor_conditioned_input_shape"].map(shape_ok).all()
        )
        rows.append(
            {
                "seed": registry.get("seed"),
                "outer_cycle": registry.get("outer_cycle"),
                "path": str(path),
                "exists": True,
                "rows": len(frame),
                "required_columns_present": present,
                "direct_conditioning_active_all_rows": active_ok,
                "target_context_dim_3_all_rows": dim_ok,
                "target_context_one_hot_sum_1_all_rows": one_hot_ok,
                "actor_conditioned_input_shape_131_all_rows": conditioned_shape_ok,
                "actor_conditioned_input_schema_examples": sorted(set(str(v) for v in frame["actor_conditioned_input_schema"].head(3).tolist()))
                if present["actor_conditioned_input_schema"]
                else [],
            }
        )
    checks = {
        "all_trace_files_exist": all(row.get("exists") for row in rows) and len(rows) == 33,
        "all_required_columns_present": all(all(row.get("required_columns_present", {}).values()) for row in rows),
        "direct_conditioning_active_all_rows": all(row.get("direct_conditioning_active_all_rows") for row in rows),
        "target_context_dim_3_all_rows": all(row.get("target_context_dim_3_all_rows") for row in rows),
        "target_context_one_hot_sum_1_all_rows": all(row.get("target_context_one_hot_sum_1_all_rows") for row in rows),
        "actor_conditioned_input_shape_131_all_rows": all(row.get("actor_conditioned_input_shape_131_all_rows") for row in rows),
        "total_pre_action_rows_11616": total_rows == 11616,
    }
    return {
        "stage": STAGE,
        "trace_file_count": len(rows),
        "total_pre_action_rows": total_rows,
        "checks": checks,
        "conditioned_actor_input_trace_passed": all(checks.values()),
        "rows": rows,
    }


def load_checkpoint_context(h4mg: Any, checkpoint: Mapping[str, Any], seed: int, created_at: str, device: torch.device) -> Dict[str, Any]:
    ctx = build_context_mps_h4mn(h4mg, seed, created_at, device=device)
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    ctx["encoder"].load_state_dict(payload["gatv2_state_dict"])
    ctx["actor"].load_state_dict(payload["actor_state_dict"])
    ctx["critic"].load_state_dict(payload["critic_state_dict"])
    return ctx


def paired_context_probe(dl1: Any, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    encoder = ctx["encoder"]
    actor = ctx["actor"]
    data = ctx["train_data"][0].to(ctx["device"])
    indices = dl1.agent_indices_for_step(ctx["config"]["spec"], 0, int(ctx["config"]["effective_agents"]))
    encoder.eval()
    actor.eval()
    with torch.no_grad():
        node_embeddings = encoder(data)
        idx = torch.tensor(indices, dtype=torch.long, device=node_embeddings.device)
        embeddings = node_embeddings[idx][: min(8, len(indices))]
        contexts = torch.eye(3, dtype=embeddings.dtype, device=embeddings.device)
        max_logit_diff = 0.0
        max_prob_diff = 0.0
        prefer_rows = []
        for sample_idx in range(embeddings.size(0)):
            repeated = embeddings[sample_idx : sample_idx + 1].expand(3, -1)
            logits = actor(repeated, contexts)
            probs = torch.softmax(logits, dim=-1)
            for i in range(3):
                for j in range(i + 1, 3):
                    max_logit_diff = max(max_logit_diff, float(torch.max(torch.abs(logits[i] - logits[j])).detach().cpu().item()))
                    max_prob_diff = max(max_prob_diff, float(torch.max(torch.abs(probs[i] - probs[j])).detach().cpu().item()))
            prefer_rows.append(
                {
                    "sample_index": sample_idx,
                    "hold_context_prefer_hold": bool(probs[0, 0] > probs[0, 1]),
                    "hold_context_prefer_serve": bool(probs[0, 1] > probs[0, 0]),
                    "serve_context_prefer_hold": bool(probs[1, 0] > probs[1, 1]),
                    "serve_context_prefer_serve": bool(probs[1, 1] > probs[1, 0]),
                    "hold_context_margin_hold_minus_serve": float((probs[0, 0] - probs[0, 1]).detach().cpu().item()),
                    "serve_context_margin_hold_minus_serve": float((probs[1, 0] - probs[1, 1]).detach().cpu().item()),
                }
            )
    return {
        "max_logit_diff": max_logit_diff,
        "max_probability_diff": max_prob_diff,
        "hold_context_prefer_hold_rate": safe_rate(sum(row["hold_context_prefer_hold"] for row in prefer_rows), len(prefer_rows)),
        "hold_context_prefer_serve_rate": safe_rate(sum(row["hold_context_prefer_serve"] for row in prefer_rows), len(prefer_rows)),
        "serve_context_prefer_serve_rate": safe_rate(sum(row["serve_context_prefer_serve"] for row in prefer_rows), len(prefer_rows)),
        "rows": prefer_rows,
    }


def target_gradient_probe(dl1: Any, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    encoder = ctx["encoder"]
    actor = ctx["actor"]
    critic = ctx["critic"]
    data = ctx["train_data"][0].to(ctx["device"])
    indices = dl1.agent_indices_for_step(ctx["config"]["spec"], 0, int(ctx["config"]["effective_agents"]))
    encoder.train()
    actor.train()
    critic.train()
    for module in (encoder, actor, critic):
        module.zero_grad(set_to_none=True)
    logits, _values, mask, _node_embeddings = dl1.forward_policy(data, indices, encoder, actor, critic)
    loss = logits[mask.bool()].sum()
    loss.backward()
    target_grad = actor.net[0].weight.grad[:, -3:].detach()
    return {
        "loss": float(loss.detach().cpu().item()),
        "target_weight_grad_shape": list(target_grad.shape),
        "target_weight_grad_sha256": tensor_hash(target_grad),
        "target_weight_grad_finite": bool(torch.isfinite(target_grad).all().detach().cpu().item()),
        "target_weight_grad_nonzero_count": int((target_grad != 0).sum().detach().cpu().item()),
        "target_weight_grad_l2_norm": float(torch.linalg.vector_norm(target_grad.float()).detach().cpu().item()),
        "optimizer_step_executed": False,
    }


def actor_target_utilization(
    h4mg: Any,
    checkpoint_registry: Mapping[str, Any],
    joined_rows: Sequence[Mapping[str, Any]],
    created_at: str,
    device: torch.device,
) -> Dict[str, Any]:
    cycle_rows = []
    for cycle in range(1, EXPECTED["outer_training_count"] + 1):
        for target_id in [0, 1]:
            rows = [row for row in joined_rows if int(row["cycle"]) == cycle and int(row["target_id"]) == target_id]
            action_counts = Counter(row["sampled_action_name"] for row in rows)
            prefer_hold = sum(1 for row in rows if float(row["P_HOLD"]) > float(row["P_SERVE"]))
            prefer_serve = sum(1 for row in rows if float(row["P_SERVE"]) > float(row["P_HOLD"]))
            cycle_rows.append(
                {
                    "outer_cycle": cycle,
                    "target_id": target_id,
                    "target_name": ACTION_NAMES[target_id],
                    "count": len(rows),
                    "P_HOLD": numeric_stats([row["P_HOLD"] for row in rows]),
                    "P_SERVE": numeric_stats([row["P_SERVE"] for row in rows]),
                    "policy_margin_hold_minus_serve": numeric_stats([row["policy_margin_hold_minus_serve"] for row in rows]),
                    "pre_mask_logit_margin_hold_minus_serve": numeric_stats(
                        [row["pre_mask_logit_margin_hold_minus_serve"] for row in rows]
                    ),
                    "prefer_hold_rate": safe_rate(prefer_hold, len(rows)),
                    "prefer_serve_rate": safe_rate(prefer_serve, len(rows)),
                    "sampled_action_counts": dict(action_counts),
                }
            )
    checkpoints = []
    for checkpoint in checkpoint_registry.get("checkpoints", []):
        seed = int(checkpoint["seed"])
        ctx = load_checkpoint_context(h4mg, checkpoint, seed, created_at, device)
        dl1 = ctx["dl1"]
        paired = paired_context_probe(dl1, ctx)
        gradient = target_gradient_probe(dl1, ctx)
        target_weight = ctx["actor"].net[0].weight.detach().cpu()[:, -3:]
        checkpoints.append(
            {
                "seed": seed,
                "checkpoint_path": checkpoint["path"],
                "actor_target_context_dim": int(getattr(ctx["actor"], "target_context_dim", -1)),
                "actor_conditioned_input_dim": int(getattr(ctx["actor"], "actor_conditioned_input_dim", -1)),
                "target_weight_stats": {
                    "shape": list(target_weight.shape),
                    "sha256": tensor_hash(target_weight),
                    "nonzero_count": int((target_weight != 0).sum().item()),
                    "l2_norm": float(torch.linalg.vector_norm(target_weight.float()).item()),
                    "finite": bool(torch.isfinite(target_weight).all().item()),
                },
                "paired_context_sensitivity": paired,
                "target_conditioning_gradient_probe": gradient,
            }
        )
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    h4ml_pattern = read_json(H4M_L_ROOT / "09_h4m_l_repair_contract.json")["evidence"]["actor_paired_context_sensitivity"]
    cycle11_hold = next(row for row in cycle_rows if row["outer_cycle"] == 11 and row["target_id"] == 0)
    cycle11_serve = next(row for row in cycle_rows if row["outer_cycle"] == 11 and row["target_id"] == 1)
    checks = {
        "actor_131d_all_checkpoints": all(row["actor_conditioned_input_dim"] == 131 for row in checkpoints),
        "actor_target_context_dim_3_all_checkpoints": all(row["actor_target_context_dim"] == 3 for row in checkpoints),
        "paired_context_sensitivity_positive": all(row["paired_context_sensitivity"]["max_logit_diff"] > 0.0 for row in checkpoints),
        "target_weight_finite_nonzero": all(
            row["target_weight_stats"]["finite"] and row["target_weight_stats"]["nonzero_count"] > 0 for row in checkpoints
        ),
        "target_conditioning_gradient_finite_nonzero": all(
            row["target_conditioning_gradient_probe"]["target_weight_grad_finite"]
            and row["target_conditioning_gradient_probe"]["target_weight_grad_nonzero_count"] > 0
            for row in checkpoints
        ),
        "optimizer_step_not_executed_in_gradient_probe": all(
            row["target_conditioning_gradient_probe"]["optimizer_step_executed"] is False for row in checkpoints
        ),
    }
    return {
        "stage": STAGE,
        "cycle_target_context_evolution": cycle_rows,
        "cycle1_to11_target_hold": {
            "cycle1": next(row for row in cycle_rows if row["outer_cycle"] == 1 and row["target_id"] == 0),
            "cycle11": cycle11_hold,
        },
        "cycle1_to11_target_serve": {
            "cycle1": next(row for row in cycle_rows if row["outer_cycle"] == 1 and row["target_id"] == 1),
            "cycle11": cycle11_serve,
        },
        "h4m_l_baseline_pattern": {
            "hold_context_prefer_serve_rate": h4ml_pattern["hold_context_prefer_serve_rate"],
            "serve_context_prefer_serve_rate": h4ml_pattern["serve_context_prefer_serve_rate"],
            "paired_preference_flip_rate": h4ml_pattern["paired_preference_flip_rate"],
        },
        "h4m_n_cycle11_observed_target_preference": {
            "target_hold_prefer_hold_rate": cycle11_hold["prefer_hold_rate"],
            "target_hold_prefer_serve_rate": cycle11_hold["prefer_serve_rate"],
            "target_serve_prefer_serve_rate": cycle11_serve["prefer_serve_rate"],
            "target_serve_prefer_hold_rate": cycle11_serve["prefer_hold_rate"],
        },
        "final_checkpoint_utilization": checkpoints,
        "checks": checks,
        "actor_target_utilization_passed": all(checks.values()),
    }


def target_conditioned_credit(joined_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for target_id in [0, 1, 2]:
        for sampled_action in [ACTION_HOLD, ACTION_SERVE, ACTION_SKIP]:
            subset = [
                row
                for row in joined_rows
                if int(row["target_id"]) == target_id and row["sampled_action_name"] == sampled_action
            ]
            positive_raw = sum(1 for row in subset if float(row["raw_gae_advantage"]) > 0.0)
            positive_norm = sum(1 for row in subset if float(row["normalized_advantage"]) > 0.0)
            rows.append(
                {
                    "target_id": target_id,
                    "target_name": ACTION_NAMES[target_id],
                    "sampled_action": sampled_action,
                    "count": len(subset),
                    "raw_gae_advantage": numeric_stats([row["raw_gae_advantage"] for row in subset]),
                    "raw_gae_positive_rate": safe_rate(positive_raw, len(subset)),
                    "normalized_advantage": numeric_stats([row["normalized_advantage"] for row in subset]),
                    "normalized_advantage_positive_rate": safe_rate(positive_norm, len(subset)),
                    "value_error": numeric_stats([row["value_error"] for row in subset]),
                    "serve_pressure_increase_count": numeric_stats([row["serve_pressure_increase_count"] for row in subset]),
                    "hold_pressure_increase_count": numeric_stats([row["hold_pressure_increase_count"] for row in subset]),
                }
            )
    def get(target_id: int, action: str) -> Mapping[str, Any]:
        return next(row for row in rows if row["target_id"] == target_id and row["sampled_action"] == action)

    target_hold_hold = get(0, ACTION_HOLD)
    target_hold_serve = get(0, ACTION_SERVE)
    target_serve_hold = get(1, ACTION_HOLD)
    target_serve_serve = get(1, ACTION_SERVE)
    hold_credit_ok = (
        target_hold_hold["raw_gae_advantage"]["mean"] is not None
        and target_hold_serve["raw_gae_advantage"]["mean"] is not None
        and target_hold_hold["raw_gae_advantage"]["mean"] > target_hold_serve["raw_gae_advantage"]["mean"]
    )
    serve_credit_ok = (
        target_serve_hold["raw_gae_advantage"]["mean"] is not None
        and target_serve_serve["raw_gae_advantage"]["mean"] is not None
        and target_serve_serve["raw_gae_advantage"]["mean"] > target_serve_hold["raw_gae_advantage"]["mean"]
    )
    return {
        "stage": STAGE,
        "target_sampled_action_rows": rows,
        "target_hold_credit_direction": {
            "sampled_hold_raw_gae_mean": target_hold_hold["raw_gae_advantage"]["mean"],
            "sampled_serve_raw_gae_mean": target_hold_serve["raw_gae_advantage"]["mean"],
            "credit_favors_hold_over_serve": hold_credit_ok,
            "rule": "target=HOLD credit direction OK iff sampled HOLD raw GAE mean > sampled SERVE raw GAE mean; no arbitrary tolerance",
        },
        "target_serve_credit_direction": {
            "sampled_hold_raw_gae_mean": target_serve_hold["raw_gae_advantage"]["mean"],
            "sampled_serve_raw_gae_mean": target_serve_serve["raw_gae_advantage"]["mean"],
            "credit_favors_serve_over_hold": serve_credit_ok,
            "rule": "target=SERVE credit direction OK iff sampled SERVE raw GAE mean > sampled HOLD raw GAE mean; no arbitrary tolerance",
        },
        "target_conditioned_credit_passed": hold_credit_ok and serve_credit_ok,
    }


def weighted_overall_correct(summary: Mapping[str, Any], *, sampled: bool) -> Optional[float]:
    hold = summary["by_long_horizon_label"][CLASS_HOLD_BETTER]
    serve = summary["by_long_horizon_label"][CLASS_SERVE_BETTER]
    count = int(hold["count"]) + int(serve["count"])
    if count == 0:
        return None
    key = "sampled_correct_direction_count" if sampled else "probability_correct_direction_count"
    return float(int(hold[key]) + int(serve[key])) / float(count)


def compact_run_metrics(label: str, conditional: Mapping[str, Any], collapse: Mapping[str, Any]) -> Dict[str, Any]:
    hold = conditional["by_long_horizon_label"][CLASS_HOLD_BETTER]
    serve = conditional["by_long_horizon_label"][CLASS_SERVE_BETTER]
    separation = None
    if hold["policy_margin_hold_minus_serve"]["mean"] is not None and serve["policy_margin_hold_minus_serve"]["mean"] is not None:
        separation = hold["policy_margin_hold_minus_serve"]["mean"] - serve["policy_margin_hold_minus_serve"]["mean"]
    cycle1 = conditional["overall_policy_evolution"][0]
    cycle11 = conditional["overall_policy_evolution"][-1]
    return {
        "label": label,
        "hold_better_probability_correct_direction_rate": hold["probability_correct_direction_rate"],
        "hold_better_sampled_correct_direction_rate": hold["sampled_correct_direction_rate"],
        "serve_better_probability_correct_direction_rate": serve["probability_correct_direction_rate"],
        "serve_better_sampled_correct_direction_rate": serve["sampled_correct_direction_rate"],
        "overall_probability_correct_direction_rate": weighted_overall_correct(conditional, sampled=False),
        "overall_sampled_correct_direction_rate": weighted_overall_correct(conditional, sampled=True),
        "conditional_probability_separation_gap": separation,
        "hold_better_probability_dominant_action": hold["probability_dominant_action"],
        "serve_better_probability_dominant_action": serve["probability_dominant_action"],
        "global_policy_collapse_detected": collapse["global_policy_collapse_detected"],
        "global_probability_dominant_cycle11": cycle11["probability_dominant_action"],
        "entropy_cycle1_mean": cycle1["entropy"]["mean"],
        "entropy_cycle11_mean": cycle11["entropy"]["mean"],
        "cycle1_to11": conditional["cycle1_to11_overall"],
    }


def h4mh_h4mk_h4mn_comparison(kmod: Any, conditional: Mapping[str, Any], collapse: Mapping[str, Any]) -> Dict[str, Any]:
    h4mh_raw = kmod.load_h4m_h_comparator()
    h4mk_conditional = read_json(H4M_K_ROOT / "04_conditional_policy_discrimination.json")
    h4mk_collapse = read_json(H4M_K_ROOT / "10_policy_collapse_audit.json")
    h4mh = {
        "label": "H4M-H-RERUN_pre_observation_repair",
        "hold_better_probability_correct_direction_rate": h4mh_raw["probability_direction"][CLASS_HOLD_BETTER]["correct_direction_rate"],
        "hold_better_sampled_correct_direction_rate": h4mh_raw["sampled_action_direction"][CLASS_HOLD_BETTER]["correct_direction_rate"],
        "serve_better_probability_correct_direction_rate": h4mh_raw["probability_direction"][CLASS_SERVE_BETTER]["correct_direction_rate"],
        "serve_better_sampled_correct_direction_rate": h4mh_raw["sampled_action_direction"][CLASS_SERVE_BETTER]["correct_direction_rate"],
        "overall_probability_correct_direction_rate": h4mh_raw["probability_direction"]["overall_correct_direction_rate"],
        "overall_sampled_correct_direction_rate": safe_rate(
            h4mh_raw["sampled_action_direction"][CLASS_HOLD_BETTER]["correct_count"]
            + h4mh_raw["sampled_action_direction"][CLASS_SERVE_BETTER]["correct_count"],
            h4mh_raw["sampled_action_direction"][CLASS_HOLD_BETTER]["count"]
            + h4mh_raw["sampled_action_direction"][CLASS_SERVE_BETTER]["count"],
        ),
        "conditional_probability_separation_gap": h4mh_raw["probability_direction"][CLASS_HOLD_BETTER][
            "policy_margin_hold_minus_serve_mean"
        ]
        - h4mh_raw["probability_direction"][CLASS_SERVE_BETTER]["policy_margin_hold_minus_serve_mean"],
        "global_policy_collapse_detected": None,
    }
    h4mk = compact_run_metrics("H4M-K_12D_observation_only", h4mk_conditional, h4mk_collapse)
    h4mn = compact_run_metrics("H4M-N_12D_plus_actor_direct_conditioning", conditional, collapse)

    def deltas(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
        keys = [
            "hold_better_probability_correct_direction_rate",
            "hold_better_sampled_correct_direction_rate",
            "serve_better_probability_correct_direction_rate",
            "serve_better_sampled_correct_direction_rate",
            "overall_probability_correct_direction_rate",
            "overall_sampled_correct_direction_rate",
            "conditional_probability_separation_gap",
        ]
        return {
            key: None if left.get(key) is None or right.get(key) is None else float(left[key]) - float(right[key])
            for key in keys
        }

    return {
        "stage": STAGE,
        "comparators": {
            "h4m_h_rerun": h4mh,
            "h4m_k": h4mk,
            "h4m_n": h4mn,
        },
        "deltas_h4mn_minus_h4mk": deltas(h4mn, h4mk),
        "deltas_h4mn_minus_h4mh": deltas(h4mn, h4mh),
        "identical_sample_identity_crosswalk": {
            "applied": False,
            "reason": "H4M-H/H4M-K/H4M-N are fresh independent diagnostic executions with stage-specific sample_uid scopes; aggregate read-only comparator used.",
        },
    }


def training_integrity_h4mn(
    kmod: Any,
    binding: Mapping[str, Any],
    training: Mapping[str, Any],
    cycle_summaries: Sequence[Mapping[str, Any]],
    trace_meta: Mapping[str, Any],
    conditional: Mapping[str, Any],
    shadow_summary: Mapping[str, Any],
    credit_summary: Mapping[str, Any],
    actor_utilization: Mapping[str, Any],
) -> Dict[str, Any]:
    integrity = kmod.training_integrity(binding, training, cycle_summaries, trace_meta, conditional, shadow_summary, credit_summary)
    conditioned_trace = trace_conditioned_input_audit(credit_summary)
    seed_summaries = training.get("seed_summaries", [])
    extra_criteria = {
        "actor_target_context_dim_3_all_seeds": all(
            row.get("checkpoint", {}).get("actor_target_context_dim") == 3 for row in seed_summaries
        ),
        "actor_conditioned_input_dim_131_all_seeds": all(
            row.get("checkpoint", {}).get("actor_conditioned_input_dim") == 131 for row in seed_summaries
        ),
        "conditioned_actor_input_trace_lineage": conditioned_trace["conditioned_actor_input_trace_passed"],
        "actor_target_utilization_pass": actor_utilization.get("actor_target_utilization_passed") is True,
        "target_conditioning_gradients_finite_nonzero": actor_utilization.get("checks", {}).get(
            "target_conditioning_gradient_finite_nonzero"
        )
        is True,
    }
    integrity["criteria"].update(extra_criteria)
    integrity["training_integrity_passed"] = all(integrity["criteria"].values())
    integrity["failing_criteria"] = [k for k, v in integrity["criteria"].items() if not v]
    integrity["conditioned_actor_input_trace_audit"] = conditioned_trace
    integrity["active_actor_direct_conditioning_contract_sha256"] = EXPECTED["h4m_l_repair_contract_sha256"]
    integrity["actor_conditioned_input_dim"] = 131
    return integrity


def root_decision(
    conditional: Mapping[str, Any],
    comparison: Mapping[str, Any],
    collapse: Mapping[str, Any],
    actor_utilization: Mapping[str, Any],
    credit: Mapping[str, Any],
) -> Dict[str, Any]:
    hold = conditional["by_long_horizon_label"][CLASS_HOLD_BETTER]
    serve = conditional["by_long_horizon_label"][CLASS_SERVE_BETTER]
    h4mn = comparison["comparators"]["h4m_n"]
    deltas_k = comparison["deltas_h4mn_minus_h4mk"]
    credit_gradient_imbalance = not (
        credit.get("target_conditioned_credit_passed") is True
        and actor_utilization.get("checks", {}).get("target_conditioning_gradient_finite_nonzero") is True
    )
    if collapse.get("global_policy_collapse_detected") is True:
        dominant = collapse.get("hold_better_probability_dominant_action")
        if dominant == "HOLD":
            decision_class = "D"
        else:
            decision_class = "C"
        next_gate = "H4M-O_ACTOR_TARGET_CONDITIONED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
    elif (
        hold["probability_dominant_action"] == "HOLD"
        and serve["probability_dominant_action"] == "SERVE"
        and h4mn["overall_probability_correct_direction_rate"] is not None
    ):
        decision_class = "A"
        next_gate = "H4M-O_FROZEN_ACTOR_TARGET_CONDITIONED_POLICY_REVALIDATION"
    elif credit_gradient_imbalance:
        decision_class = "E"
        next_gate = "H4M-O_ACTOR_TARGET_CONDITIONED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
    elif any((value is not None and value > 0.0) for value in deltas_k.values()):
        decision_class = "B"
        next_gate = "H4M-O_ACTOR_TARGET_CONDITIONED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"
    else:
        decision_class = "F"
        next_gate = "H4M-O_MINIMAL_CAUSAL_EFFECT_ISOLATION_AUDIT"
    return {
        "stage": STAGE,
        "decision_class": decision_class,
        "decision": DECISION_CLASSES[decision_class],
        "exact_next_gate": next_gate,
        "decision_rule": {
            "no_arbitrary_tolerance": True,
            "A_requires": [
                "no global same-action collapse",
                "HOLD-better probability dominant action is HOLD",
                "SERVE-better probability dominant action is SERVE",
            ],
            "C_or_D_requires": "global collapse detected by identical probability and sampled dominance across HOLD-better/SERVE-better labels",
            "E_requires": "target-conditioned credit direction or target-conditioning gradient usage failure",
            "B_requires": "not A/C/D/E and at least one H4M-N minus H4M-K diagnostic delta is positive",
        },
        "evidence": {
            "hold_better_probability_dominant_action": hold["probability_dominant_action"],
            "serve_better_probability_dominant_action": serve["probability_dominant_action"],
            "global_policy_collapse_detected": collapse.get("global_policy_collapse_detected"),
            "credit_gradient_imbalance": credit_gradient_imbalance,
            "deltas_h4mn_minus_h4mk": deltas_k,
            "h4mn_metrics": h4mn,
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
        "actor_131d_path_active": integrity.get("criteria", {}).get("actor_conditioned_input_dim_131_all_seeds") is True,
        "total_rollouts_33": training.get("total_rollouts") == 33,
        "total_ppo_updates_132": training.get("total_ppo_updates") == 132,
        "total_critic_updates_264": training.get("total_critic_updates") == 264,
        "conditional_discrimination_result_measurable": conditional.get("conditional_discrimination_measurable") is True,
        "decision_selected": root.get("decision") in set(DECISION_CLASSES.values()),
        "test6_sealed": integrity.get("TEST6") == "SEALED_NOT_OPENED",
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": root.get("decision") if passed else "H4M_N_FRESH_ACTOR_TARGET_CONDITIONED_THREE_SEED_RETRAINING_BLOCKED",
        "execution_decision": PASS_DECISION if passed else "H4M_N_EXECUTION_BLOCKED",
        "exact_next_gate": root.get("exact_next_gate") if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "final_flags": {
            "fresh_training_executed": passed,
            "reward_gae_ppo_modified": False,
            "additional_repair_performed": False,
            "feature_added_or_deleted_beyond_h4m_m_contract": False,
            "environment_expanded": False,
            "validation_or_test6_executed": False,
            "winner_or_baseline_selected": False,
            "github_push_performed": False,
        },
        "active_mps_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "actor_direct_conditioning_contract_sha256": EXPECTED["h4m_l_repair_contract_sha256"],
        "observation_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
    }


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
        "execution_decision": gate.get("execution_decision"),
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
    actor_utilization: Mapping[str, Any],
    credit: Mapping[str, Any],
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
            "actor_conditioned_input_dim": row.get("checkpoint", {}).get("actor_conditioned_input_dim"),
        }
        for row in training.get("seed_summaries", [])
    ]
    hold = conditional["by_long_horizon_label"][CLASS_HOLD_BETTER]
    serve = conditional["by_long_horizon_label"][CLASS_SERVE_BETTER]
    return f"""# H4M-N Fresh Actor-Target-Conditioned Three-Seed Retraining

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_n_source_git_commit"]}
execution_decision = {gate["execution_decision"]}
decision = {gate["decision"]}
exact_next_gate = {gate["exact_next_gate"]}

## Three-seed execution counts

```json
{json.dumps(seed_counts, ensure_ascii=False, indent=2, default=jsonable)}
```

## Cycle 1→11 policy evolution

```json
{json.dumps(conditional.get("cycle1_to11_overall"), ensure_ascii=False, indent=2, default=jsonable)}
```

## HOLD-better conditional result

```json
{json.dumps(hold, ensure_ascii=False, indent=2, default=jsonable)}
```

## SERVE-better conditional result

```json
{json.dumps(serve, ensure_ascii=False, indent=2, default=jsonable)}
```

## Actor target-utilization change

```json
{json.dumps(actor_utilization.get("h4m_n_cycle11_observed_target_preference"), ensure_ascii=False, indent=2, default=jsonable)}
```

## H4M-H/H4M-K 대비 개선량

```json
{json.dumps({"h4mn_minus_h4mk": comparison.get("deltas_h4mn_minus_h4mk"), "h4mn_minus_h4mh": comparison.get("deltas_h4mn_minus_h4mh")}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Global collapse

```json
{json.dumps(collapse, ensure_ascii=False, indent=2, default=jsonable)}
```

## Target-conditioned credit

```json
{json.dumps({"target_hold": credit.get("target_hold_credit_direction"), "target_serve": credit.get("target_serve_credit_direction")}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Integrity

training_integrity_passed = {integrity.get("training_integrity_passed")}
TEST6 = {integrity.get("TEST6")}

STOP: no additional training, no additional repair, no Reward/GAE/PPO modification, no environment expansion, no validation/TEST6, no winner/baseline, no GitHub push.
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
        "decision": "H4M_N_FRESH_ACTOR_TARGET_CONDITIONED_THREE_SEED_RETRAINING_BLOCKED",
        "execution_decision": "H4M_N_EXECUTION_BLOCKED",
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
        "05_actor_target_utilization.json": empty,
        "06_target_conditioned_credit.json": empty,
        "07_h4mh_h4mk_h4mn_comparison.json": empty,
        "08_long_horizon_shadow_summary.json": empty,
        "09_policy_collapse_audit.json": empty,
        "10_root_decision.json": empty,
        "11_checkpoint_registry.json": empty,
        "12_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(f"# H4M-N\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-N] artifact root: {artifact_root}")
    print(f"[H4M-N] gate: {BLOCK_GATE}")
    print(f"[H4M-N] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_n_fresh_actor_target_conditioned_three_seed_retraining_{stamp}"
    provenance = source_provenance(created_at)
    compile_static = py_compile_audit()
    binding = authoritative_binding(created_at, provenance, compile_static)
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

    kmod = configure_h4mk_module(import_module_from_path(f"h4mn_h4mk_{time.time_ns()}", H4MK_SOURCE))
    base, _rerun = kmod.load_h4m_helpers()
    h4mg = kmod.load_h4mg()
    device = torch.device("mps")

    training, cycle_summaries, checkpoint_registry, trace_meta = kmod.run_training(base, h4mg, created_at, artifact_root, device)
    for row in training.get("seed_summaries", []):
        row["actor_target_context_dim"] = row.get("checkpoint", {}).get("actor_target_context_dim")
        row["actor_conditioned_input_dim"] = row.get("checkpoint", {}).get("actor_conditioned_input_dim")
    training["actor_architecture"] = "MAPPOActor(hidden=128, action_dim=3, target_context_dim=3)"
    training["actor_conditioned_input_dim"] = 131
    checkpoint_registry["actor_direct_conditioning_repair_contract_sha256"] = EXPECTED["h4m_l_repair_contract_sha256"]
    checkpoint_registry["actor_conditioned_input_dim"] = 131

    conditional = kmod.conditional_policy_discrimination(trace_meta["joined_rows"])
    conditional_parquet = kmod.write_conditional_parquet(base, artifact_root, trace_meta["joined_rows"])
    conditional["conditional_policy_by_sample_parquet"] = conditional_parquet
    collapse = kmod.policy_collapse_audit(conditional)
    shadow_summary = kmod.long_horizon_shadow_summary(cycle_summaries, trace_meta)
    credit_summary = kmod.credit_trace_summary(trace_meta)
    actor_utilization = actor_target_utilization(h4mg, checkpoint_registry, trace_meta["joined_rows"], created_at, device)
    credit = target_conditioned_credit(trace_meta["joined_rows"])
    comparison = h4mh_h4mk_h4mn_comparison(kmod, conditional, collapse)
    integrity = training_integrity_h4mn(
        kmod,
        binding,
        training,
        cycle_summaries,
        trace_meta,
        conditional,
        shadow_summary,
        credit_summary,
        actor_utilization,
    )
    root = root_decision(conditional, comparison, collapse, actor_utilization, credit)
    gate = gate_matrix(binding, integrity, training, conditional, root)
    report = final_report(binding, integrity, training, conditional, actor_utilization, credit, comparison, collapse, root, gate)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_training_integrity.json": integrity,
        "03_three_seed_cycle_summary.json": training,
        "04_conditional_policy_discrimination.json": conditional,
        "05_actor_target_utilization.json": actor_utilization,
        "06_target_conditioned_credit.json": credit,
        "07_h4mh_h4mk_h4mn_comparison.json": comparison,
        "08_long_horizon_shadow_summary.json": shadow_summary,
        "09_policy_collapse_audit.json": collapse,
        "10_root_decision.json": root,
        "11_checkpoint_registry.json": checkpoint_registry,
        "12_gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    hold = conditional["by_long_horizon_label"][CLASS_HOLD_BETTER]
    serve = conditional["by_long_horizon_label"][CLASS_SERVE_BETTER]
    print(f"[H4M-N] artifact root: {artifact_root}")
    print(f"[H4M-N] gate: {gate['gate']}")
    print(f"[H4M-N] source_commit: {provenance['h4m_n_source_git_commit']}")
    print(
        f"[H4M-N] total_rollouts={training['total_rollouts']} "
        f"total_ppo_updates={training['total_ppo_updates']} total_critic_updates={training['total_critic_updates']}"
    )
    print(f"[H4M-N] hold_better_prob_correct={hold['probability_correct_direction_rate']}")
    print(f"[H4M-N] serve_better_prob_correct={serve['probability_correct_direction_rate']}")
    print(f"[H4M-N] global_collapse={collapse['global_policy_collapse_detected']}")
    print(f"[H4M-N] decision: {gate['decision']}")
    print(f"[H4M-N] next_gate: {gate['exact_next_gate']}")
    print("[H4M-N] STOP: no additional training, no validation/TEST6, no winner/baseline, github_push=false")


if __name__ == "__main__":
    main()
