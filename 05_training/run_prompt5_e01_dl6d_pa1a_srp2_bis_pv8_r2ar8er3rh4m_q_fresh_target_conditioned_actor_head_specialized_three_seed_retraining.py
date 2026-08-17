#!/usr/bin/env python3
"""H4M-Q fresh target-conditioned actor-head specialized three-seed retraining.

This stage reuses the frozen H4M-N/H4M-B/H4M-K training lineage and changes
only the actor decision head path validated in H4M-P:

    shared actor trunk -> HOLD/SERVE/SKIP specialized decision heads.

It performs fresh seeds 1/2/3 x 11-cycle MPS retraining, validation
discrimination diagnostics on the frozen H4L validation basis, and gradient
isolation audits. It does not repair, retune, open TEST6, or push GitHub.
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


STAGE = "PV8-R2A-R8E-R3-R-H4M-Q"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Q_"
    "FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_THREE_SEED_RETRAINING_COMPLETE"
)
BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Q_"
    "FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_THREE_SEED_RETRAINING_FAILED"
)
PASS_DECISION = "H4M_Q_FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_RETRAINING_EXECUTION_COMPLETE"
NEXT_DECISION_GATE = "H4M-R_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MN_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_n_fresh_actor_target_conditioned_three_seed_retraining.py"
H4MK_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
H4L_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4l_frozen_three_seed_policy_validation_evaluation.py"
DL1_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"
DL4_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
OBS_REPAIR_SOURCE = TRAINING_ROOT / "observation_target_context_repair.py"

H4M_P_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_p_target_conditioned_actor_head_specialization_"
    "implementation_equivalence_validation_20260817_145922+0900"
)
H4M_N_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_n_fresh_actor_target_conditioned_three_seed_retraining_20260817_134046+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
H4M_G_CLOSE_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_close_instrumentation_diagnosis_repair_validation_closure_20260816_182923+0900"
H4L_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4l_frozen_three_seed_policy_validation_evaluation_20260814_124419"

EXPECTED = {
    "h4m_p_source_commit": "e8d69668e800124f41ff789a32210fb7701aa289",
    "h4m_p_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_P_"
        "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
    ),
    "h4m_p_repair": "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_REPAIR",
    "h4m_p_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
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
CLASS_SKIP_OR_OTHER = "SKIP_OR_OTHER_TARGET_CONTEXT"
TARGET_CONTEXT_FIELDS = ["r3_action_target_is_hold", "r3_action_target_is_serve", "r3_action_target_is_skip"]

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "repair_binding.json",
    "training_summary.json",
    "seed_metrics.json",
    "cycle_action_evolution.json",
    "validation_discrimination.json",
    "parameter_delta.json",
    "gradient_audit.json",
    "checkpoint_manifest.json",
    "training_integrity.json",
    "changed_files.json",
    "test_results.json",
    "gate_matrix.json",
]

BEHAVIORAL_OUTCOMES = {
    "DISCRIMINATION_RESTORED",
    "PARTIAL_DISCRIMINATION",
    "GLOBAL_HOLD_SHIFT_WITHOUT_CONTEXT_DISCRIMINATION",
    "SERVE_COLLAPSE_PERSISTS",
    "NEW_COLLAPSE_OR_INSTABILITY",
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


def dominant_action_from_probs(row: Mapping[str, Any]) -> str:
    values = {
        ACTION_HOLD: row.get("P_HOLD", {}).get("mean"),
        ACTION_SERVE: row.get("P_SERVE", {}).get("mean"),
        ACTION_SKIP: row.get("P_SKIP", {}).get("mean"),
    }
    finite = {key: float(value) for key, value in values.items() if value is not None and math.isfinite(float(value))}
    if not finite:
        return "NOT_MEASURABLE"
    best = max(finite.values())
    winners = [key for key, value in finite.items() if value == best]
    return winners[0] if len(winners) == 1 else "TIE"


def sampled_dominant_action(counts: Mapping[str, int]) -> str:
    if not counts:
        return "NOT_MEASURABLE"
    best = max(int(v) for v in counts.values())
    winners = [str(k) for k, v in counts.items() if int(v) == best]
    return winners[0] if len(winners) == 1 else "TIE"


def masked_logits_for_targets(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    allowed = torch.ones_like(logits, dtype=torch.bool)
    if logits.size(-1) >= 3:
        allowed[:, 2] = targets.eq(2)
        allowed.scatter_(1, targets.reshape(-1, 1).clamp(min=0, max=logits.size(-1) - 1), True)
    return logits.masked_fill(~allowed, -1.0e9), allowed


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


def maybe_mps_rng_state() -> Optional[torch.Tensor]:
    if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            return torch.mps.get_rng_state()
        except Exception:
            return None
    return None


def maybe_restore_mps_rng_state(state: Optional[torch.Tensor]) -> bool:
    if state is None:
        return False
    if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
        try:
            torch.mps.set_rng_state(state)
            return True
        except Exception:
            return False
    return False


def py_compile_audit() -> Dict[str, Any]:
    rows = []
    passed = True
    with tempfile.TemporaryDirectory(prefix="h4m_q_pycompile_") as tmp:
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
        "rows": rows,
        "git_diff_cached_check_passed": cached.returncode == 0,
        "git_diff_cached_check_stdout": cached.stdout.strip(),
        "git_diff_cached_check_stderr": cached.stderr.strip(),
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"], check=False).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    source_commit = git_run(["log", "-1", "--format=%H", "--", str(SOURCE_REL)]).stdout.strip()
    source_present = git_run(["cat-file", "-e", f"HEAD:{SOURCE_REL}"], check=False).returncode == 0
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "parent_commit": parent,
        "h4m_q_source_git_commit": source_commit,
        "h4m_p_source_commit_before_work": EXPECTED["h4m_p_source_commit"],
        "source_rel": str(SOURCE_REL),
        "source_sha256": sha256_file(PROJECT_ROOT / SOURCE_REL),
        "source_present_in_head": source_present,
        "head_commit_files": head_files,
        "head_commit_source_only": head_files == [str(SOURCE_REL)],
        "status_short": status_short,
        "source_only_local_commit_created_before_training": (
            source_present
            and source_commit == head
            and parent == EXPECTED["h4m_p_source_commit"]
            and head_files == [str(SOURCE_REL)]
            and status_short == ""
        ),
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_static: Mapping[str, Any]) -> Dict[str, Any]:
    h4mp_gate = read_json(H4M_P_ROOT / "gate_matrix.json")
    h4mp_repair = read_json(H4M_P_ROOT / "repair_binding.json")
    h4mp_equiv = read_json(H4M_P_ROOT / "equivalence_validation.json")
    h4mp_grad = read_json(H4M_P_ROOT / "gradient_isolation_validation.json")
    h4mp_manifest = read_json(H4M_P_ROOT / "manifest.json")
    h4mn_binding = read_json(H4M_N_ROOT / "01_authoritative_binding.json")
    h4mb_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    active_contract_sha = sha256_file(H4M_G_CLOSE_ROOT / "12_final_mps_equivalence_contract.json")
    upstream = h4mn_binding.get("sha_bindings", {})
    checks = {
        "source_only_commit_before_training": provenance.get("source_only_local_commit_created_before_training") is True,
        "source_parent_is_h4m_p_source_commit": provenance.get("parent_commit") == EXPECTED["h4m_p_source_commit"],
        "py_compile_passed": compile_static.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_static.get("git_diff_cached_check_passed") is True,
        "h4m_p_gate_match": h4mp_gate.get("gate") == EXPECTED["h4m_p_gate"],
        "h4m_p_manifest_source_commit_match": h4mp_manifest.get("source_commit_after_pass") == EXPECTED["h4m_p_source_commit"],
        "h4m_p_repair_contract_match": h4mp_gate.get("repair_contract_sha256") == EXPECTED["h4m_p_repair_contract_sha256"]
        and h4mp_repair.get("repair_binding_passed") is True,
        "h4m_p_selected_repair_match": h4mp_repair.get("observed", {}).get("selected_repair") == EXPECTED["h4m_p_repair"],
        "h4m_p_legacy_equivalence_pass": h4mp_gate.get("criteria", {}).get("legacy_equivalence_within_tolerance") is True
        and h4mp_equiv.get("equivalence_validation_passed") is True,
        "h4m_p_gradient_isolation_pass": h4mp_gate.get("criteria", {}).get("hold_target_loss_isolates_hold_head") is True
        and h4mp_gate.get("criteria", {}).get("serve_target_loss_isolates_serve_head") is True
        and h4mp_grad.get("gradient_isolation_validation_passed") is True,
        "h4m_p_test6_zero_optimizer_zero": h4mp_manifest.get("TEST6_opened") is False
        and h4mp_manifest.get("mappo_training_executed") is False,
        "h4m_n_lineage_binding_passed": h4mn_binding.get("authoritative_binding_passed") is True,
        "h4m_b_schedule_sha_match": upstream.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"]
        and h4mb_schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": upstream.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": upstream.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": upstream.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": upstream.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "active_mps_contract_sha_match": active_contract_sha == EXPECTED["active_mps_instrumentation_contract_sha256"],
        "h4m_g_close_active_contract_match_file": active_contract_sha == EXPECTED["active_mps_instrumentation_contract_sha256"],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_p": str(H4M_P_ROOT),
            "h4m_n": str(H4M_N_ROOT),
            "h4m_b": str(H4M_B_ROOT),
            "h4m_c": str(H4M_C_ROOT),
            "h4m_g_close": str(H4M_G_CLOSE_ROOT),
            "h4l_validation_basis": str(H4L_ROOT),
        },
        "source_provenance": provenance,
        "compile_static_checks": compile_static,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "repair_contract_sha256": EXPECTED["h4m_p_repair_contract_sha256"],
        "active_repair": EXPECTED["h4m_p_repair"],
        "sha_bindings": {
            "h4m_p_source_commit": EXPECTED["h4m_p_source_commit"],
            "h4m_p_repair_contract_sha256": EXPECTED["h4m_p_repair_contract_sha256"],
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
            "simulator_semantics_modified": False,
            "observation_contract_modified": False,
            "critic_architecture_or_targets_modified": False,
            "k_mask_action_legality_modified": False,
            "three_action_policy_semantics_modified": False,
            "zero_loss_modified": False,
            "frozen_split_modified": False,
            "training_schedule_modified": False,
            "hold_logit_bias_workaround_used": False,
            "target_specific_reward_shaping_used": False,
            "test6_opened_or_used": False,
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
    kmod.build_context_mps = build_context_mps_h4mq
    kmod.save_seed_checkpoint = save_seed_checkpoint_h4mq
    return kmod


def configure_h4mn_library(nmod: Any) -> Any:
    nmod.STAGE = STAGE
    nmod.PASS_GATE = PASS_GATE
    nmod.BLOCK_GATE = BLOCK_GATE
    nmod.SOURCE_REL = SOURCE_REL
    nmod.EXPECTED = {**nmod.EXPECTED, **EXPECTED}
    return nmod


def build_context_mps_h4mq(h4mg: Any, seed: int, created_at: str, *, device: torch.device) -> Dict[str, Any]:
    nmod = configure_h4mn_library(import_module_from_path(f"h4mq_h4mn_ctx_{seed}_{time.time_ns()}", H4MN_SOURCE))
    ctx = nmod.build_context_mps_h4mn(h4mg, seed, created_at, device=device)
    dl1 = ctx["dl1"]
    legacy_actor = ctx["actor"]
    cpu_rng_after_legacy_context = torch.random.get_rng_state()
    mps_rng_after_legacy_context = maybe_mps_rng_state()
    specialized_actor = dl1.MAPPOActor(
        int(ctx["config"]["gatv2_hidden"]),
        int(ctx["config"]["action_dim"]),
        target_context_dim=EXPECTED["actor_target_context_dim"],
        target_head_specialization=True,
    )
    specialized_actor.load_state_dict(legacy_actor.state_dict(), strict=True)
    specialized_actor = specialized_actor.to(device)
    torch.random.set_rng_state(cpu_rng_after_legacy_context)
    mps_rng_restored = maybe_restore_mps_rng_state(mps_rng_after_legacy_context)
    ctx["actor"] = specialized_actor
    ctx["optimizers"]["actor"] = torch.optim.Adam(specialized_actor.parameters(), lr=float(ctx["config"]["actor_lr"]))
    ctx["config"].update(
        {
            "stage": STAGE,
            "actor_target_head_specialization_repair_active": True,
            "actor_target_head_specialization_repair_contract_sha256": EXPECTED["h4m_p_repair_contract_sha256"],
            "actor_decision_head_architecture": "shared_trunk(net.0,Tanh)+target_heads[HOLD,SERVE,SKIP]",
            "actor_target_head_count": int(getattr(specialized_actor, "target_head_count", -1)),
            "actor_target_head_specialization_rng_preservation": {
                "legacy_context_constructed_first": True,
                "specialized_heads_initialized_after_critic_then_rng_restored": True,
                "cpu_rng_restored": True,
                "mps_rng_restored": mps_rng_restored,
                "reason": "Preserve H4M-N fresh initialization lineage for encoder/critic and rollout sampling; only actor decision head topology changes.",
            },
        }
    )
    return ctx


def save_seed_checkpoint_h4mq(
    artifact_root: Path,
    seed: int,
    ctx: Mapping[str, Any],
    cycle_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    path = artifact_root / "checkpoints" / f"H4M_Q_SEED_{seed:03d}_FRESH_TARGET_HEAD_SPECIALIZED.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "seed": seed,
        "checkpoint_status": "FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_TRAINING_COMPLETE_DIAGNOSTIC_EVALUATION_NOT_RELEASED",
        "created_at": kst_now(),
        "fresh_initialization": True,
        "old_9d_checkpoint_reuse": False,
        "h4m_c_h4m_h_h4m_n_checkpoint_continuation": False,
        "active_mps_instrumentation_contract_sha256": EXPECTED["active_mps_instrumentation_contract_sha256"],
        "observation_repair_contract_sha256": EXPECTED["h4m_i_repair_contract_sha256"],
        "actor_target_head_specialization_repair_contract_sha256": EXPECTED["h4m_p_repair_contract_sha256"],
        "node_feature_dim": int(ctx["config"]["node_feature_dim_after_observation_repair"]),
        "actor_target_context_dim": int(getattr(ctx["actor"], "target_context_dim", -1)),
        "actor_conditioned_input_dim": int(getattr(ctx["actor"], "actor_conditioned_input_dim", -1)),
        "actor_target_head_specialization_active": bool(getattr(ctx["actor"], "target_head_specialization_active", False)),
        "actor_target_head_count": int(getattr(ctx["actor"], "target_head_count", -1)),
        "actor_decision_head_architecture": ctx["config"].get("actor_decision_head_architecture"),
        "training_configuration": {
            key: value
            for key, value in ctx["config"].items()
            if key not in {"spec"} and not str(key).endswith("_perf")
        },
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
        "actor_target_head_specialization_active": payload["actor_target_head_specialization_active"],
        "actor_target_head_count": payload["actor_target_head_count"],
        "actor_target_head_specialization_repair_contract_sha256": EXPECTED["h4m_p_repair_contract_sha256"],
    }


def load_checkpoint_context_h4mq(h4mg: Any, checkpoint: Mapping[str, Any], seed: int, created_at: str, device: torch.device) -> Dict[str, Any]:
    ctx = build_context_mps_h4mq(h4mg, seed, created_at, device=device)
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    ctx["encoder"].load_state_dict(payload["gatv2_state_dict"])
    ctx["actor"].load_state_dict(payload["actor_state_dict"])
    ctx["critic"].load_state_dict(payload["critic_state_dict"])
    return ctx


def delta_stats_for_state(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor], keys: Iterable[str]) -> Dict[str, Any]:
    selected = [key for key in keys if key in before and key in after]
    l1 = 0.0
    l2_sq = 0.0
    max_abs = 0.0
    parameter_count = 0
    finite = True
    changed = 0
    unchanged = 0
    for key in selected:
        delta = after[key].detach().float() - before[key].detach().float()
        parameter_count += int(delta.numel())
        l1 += float(delta.abs().sum().item())
        l2_sq += float((delta * delta).sum().item())
        tensor_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        max_abs = max(max_abs, tensor_max)
        finite = finite and bool(torch.isfinite(delta).all().item())
        if tensor_max > 1.0e-12:
            changed += 1
        else:
            unchanged += 1
    return {
        "keys": selected,
        "tensor_count": len(selected),
        "parameter_count": parameter_count,
        "l1_delta": l1,
        "l2_delta": math.sqrt(l2_sq),
        "max_abs_delta": max_abs,
        "changed_tensor_count": changed,
        "unchanged_tensor_count": unchanged,
        "finite": finite,
        "delta_tolerance": 1.0e-12,
    }


def parameter_delta_audit(
    h4mg: Any,
    checkpoint_registry: Mapping[str, Any],
    created_at: str,
    device: torch.device,
    joined_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    target_counts = Counter(int(row.get("target_id", -1)) for row in joined_rows)
    rows = []
    for checkpoint in checkpoint_registry.get("checkpoints", []):
        seed = int(checkpoint["seed"])
        init_ctx = build_context_mps_h4mq(h4mg, seed, created_at, device=device)
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        before_actor = cpu_state_dict(init_ctx["actor"])
        after_actor = payload["actor_state_dict"]
        groups = {
            "shared_trunk": ["net.0.weight", "net.0.bias"],
            "legacy_unused_decision_head": ["net.2.weight", "net.2.bias"],
            "hold_context_head": ["target_heads.0.weight", "target_heads.0.bias"],
            "serve_context_head": ["target_heads.1.weight", "target_heads.1.bias"],
            "skip_context_head": ["target_heads.2.weight", "target_heads.2.bias"],
            "all_actor": list(before_actor.keys()),
        }
        branch_delta = {name: delta_stats_for_state(before_actor, after_actor, keys) for name, keys in groups.items()}
        expected_head_activity = {
            "hold_context_head": target_counts.get(0, 0) > 0,
            "serve_context_head": target_counts.get(1, 0) > 0,
            "skip_context_head": target_counts.get(2, 0) > 0,
        }
        observed_head_activity = {
            "hold_context_head": branch_delta["hold_context_head"]["l2_delta"] > 0.0,
            "serve_context_head": branch_delta["serve_context_head"]["l2_delta"] > 0.0,
            "skip_context_head": branch_delta["skip_context_head"]["l2_delta"] > 0.0,
        }
        head_delta_matches_observed_target_support = all(
            observed_head_activity[name] if expected else branch_delta[name]["l2_delta"] == 0.0
            for name, expected in expected_head_activity.items()
        )
        rows.append(
            {
                "seed": seed,
                "checkpoint_path": checkpoint["path"],
                "branch_parameter_delta": branch_delta,
                "target_counts_from_training_trace": {str(key): int(value) for key, value in sorted(target_counts.items())},
                "expected_head_activity_from_target_support": expected_head_activity,
                "observed_head_activity": observed_head_activity,
                "head_delta_matches_observed_target_support": head_delta_matches_observed_target_support,
                "gatv2_parameter_delta": init_ctx["dl1"].delta_stats(init_ctx["encoder"], payload["gatv2_state_dict"]),
                "critic_parameter_delta": init_ctx["dl1"].delta_stats(init_ctx["critic"], payload["critic_state_dict"]),
                "expected_unused_legacy_head_unchanged": branch_delta["legacy_unused_decision_head"]["l2_delta"] == 0.0,
                "hold_and_serve_head_delta_positive": branch_delta["hold_context_head"]["l2_delta"] > 0.0
                and branch_delta["serve_context_head"]["l2_delta"] > 0.0,
                "skip_head_delta_policy": (
                    "SKIP head delta is required only if frozen training trace contains target_id=2 rows; "
                    "absent target support is recorded, not treated as an arbitrary positive-delta failure."
                ),
                "shared_trunk_delta_positive": branch_delta["shared_trunk"]["l2_delta"] > 0.0,
                "finite": all(group["finite"] for group in branch_delta.values()),
            }
        )
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    checks = {
        "rows_for_all_seeds": len(rows) == 3,
        "all_finite": all(row["finite"] for row in rows),
        "shared_trunk_delta_positive_all": all(row["shared_trunk_delta_positive"] for row in rows),
        "hold_and_serve_head_delta_positive_all": all(row["hold_and_serve_head_delta_positive"] for row in rows),
        "head_delta_matches_observed_target_support_all": all(row["head_delta_matches_observed_target_support"] for row in rows),
        "unused_legacy_head_unchanged_all": all(row["expected_unused_legacy_head_unchanged"] for row in rows),
    }
    return {
        "stage": STAGE,
        "target_counts_from_training_trace": {str(key): int(value) for key, value in sorted(target_counts.items())},
        "checks": checks,
        "parameter_delta_passed": all(checks.values()),
        "seed_rows": rows,
    }


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).norm().detach().cpu().item())


def gradient_isolation_probe(dl1: Any, ctx: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    encoder = ctx["encoder"]
    actor = ctx["actor"]
    critic = ctx["critic"]
    results = []
    for target_id in [0, 1, 2]:
        selected_data = None
        selected_indices = None
        selected_targets = None
        selected_step = None
        for step, cpu_data in enumerate(ctx["train_data"]):
            probe_data = cpu_data.to(ctx["device"])
            probe_indices = dl1.agent_indices_for_step(ctx["config"]["spec"], step, int(ctx["config"]["effective_agents"]))
            probe_targets = dl1.action_targets_from_y(
                probe_data.y[torch.tensor(probe_indices, dtype=torch.long, device=ctx["device"])],
                3,
            )
            if int(probe_targets.eq(target_id).sum().detach().cpu().item()) > 0:
                selected_data = probe_data
                selected_indices = probe_indices
                selected_targets = probe_targets
                selected_step = step
                break
        for module in [encoder, actor, critic]:
            module.zero_grad(set_to_none=True)
            module.train()
        if selected_data is None or selected_indices is None or selected_targets is None:
            fallback_data = ctx["train_data"][0].to(ctx["device"])
            fallback_indices = dl1.agent_indices_for_step(ctx["config"]["spec"], 0, int(ctx["config"]["effective_agents"]))
            logits, _values, _mask, _node_embeddings = dl1.forward_policy(fallback_data, fallback_indices, encoder, actor, critic)
            loss = logits[:, target_id].sum() * 0.0
            selected_count = 0
        else:
            logits, _values, mask, _node_embeddings = dl1.forward_policy(selected_data, selected_indices, encoder, actor, critic)
            selected = mask.bool() & selected_targets.eq(target_id)
            loss = logits[selected, target_id].sum()
            selected_count = int(selected.sum().detach().cpu().item())
        loss.backward()
        head_norms = {
            str(idx): grad_norm(ctx["actor"].target_heads[idx].parameters())
            for idx in range(int(ctx["actor"].target_head_count))
        }
        shared_norm = grad_norm(ctx["actor"].net[0].parameters())
        legacy_unused_norm = grad_norm(ctx["actor"].net[2].parameters())
        cross_leakage = {
            str(idx): value
            for idx, value in ((idx, float(head_norms[str(idx)])) for idx in range(int(ctx["actor"].target_head_count)))
            if idx != target_id and value != 0.0
        }
        results.append(
            {
                "seed": seed,
                "target_id": target_id,
                "target_name": ACTION_NAMES[target_id],
                "source_train_step": selected_step,
                "selected_sample_count": selected_count,
                "loss": float(loss.detach().cpu().item()),
                "head_gradient_norms": head_norms,
                "selected_head_gradient_nonzero": head_norms[str(target_id)] > 0.0 if selected_count > 0 else True,
                "cross_specialized_head_gradient_leakage": cross_leakage,
                "cross_specialized_head_gradient_leakage_count": len(cross_leakage),
                "shared_trunk_gradient_norm": shared_norm,
                "shared_trunk_gradient_sharing_explicit": shared_norm > 0.0 if selected_count > 0 else True,
                "legacy_unused_decision_head_gradient_norm": legacy_unused_norm,
                "critic_gradient_norm": grad_norm(ctx["critic"].parameters()),
                "optimizer_step_executed": False,
                "finite": math.isfinite(float(loss.detach().cpu().item()))
                and all(math.isfinite(float(v)) for v in head_norms.values())
                and math.isfinite(shared_norm)
                and math.isfinite(legacy_unused_norm),
            }
        )
    return {"seed": seed, "probe_rows": results}


def pressure_by_target_from_joined_rows(joined_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for target_id in [0, 1, 2]:
        subset = [row for row in joined_rows if int(row.get("target_id", -1)) == target_id]
        rows.append(
            {
                "target_id": target_id,
                "target_name": ACTION_NAMES[target_id],
                "count": len(subset),
                "serve_pressure_increase_count_sum": int(sum(int(row.get("serve_pressure_increase_count", 0)) for row in subset)),
                "hold_pressure_increase_count_sum": int(sum(int(row.get("hold_pressure_increase_count", 0)) for row in subset)),
                "net_serve_minus_hold_pressure": int(
                    sum(int(row.get("serve_pressure_increase_count", 0)) - int(row.get("hold_pressure_increase_count", 0)) for row in subset)
                ),
                "raw_gae_advantage": numeric_stats([row.get("raw_gae_advantage") for row in subset]),
                "normalized_advantage": numeric_stats([row.get("normalized_advantage") for row in subset]),
                "policy_margin_hold_minus_serve": numeric_stats([row.get("policy_margin_hold_minus_serve") for row in subset]),
            }
        )
    return rows


def gradient_audit(
    h4mg: Any,
    checkpoint_registry: Mapping[str, Any],
    joined_rows: Sequence[Mapping[str, Any]],
    created_at: str,
    device: torch.device,
) -> Dict[str, Any]:
    seed_rows = []
    for checkpoint in checkpoint_registry.get("checkpoints", []):
        seed = int(checkpoint["seed"])
        ctx = load_checkpoint_context_h4mq(h4mg, checkpoint, seed, created_at, device)
        seed_rows.append(gradient_isolation_probe(ctx["dl1"], ctx, seed))
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    flat = [row for seed_row in seed_rows for row in seed_row["probe_rows"]]
    by_target: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in flat:
        by_target[int(row["target_id"])].append(row)
    checks = {
        "probe_rows_all_seeds_targets": len(flat) == 9,
        "cross_specialized_head_gradient_leakage_zero": sum(row["cross_specialized_head_gradient_leakage_count"] for row in flat) == 0,
        "legacy_unused_decision_head_gradient_zero": all(row["legacy_unused_decision_head_gradient_norm"] == 0.0 for row in flat),
        "hold_target_probe_samples_present_all_seeds": all(row["selected_sample_count"] > 0 for row in by_target[0]),
        "serve_target_probe_samples_present_all_seeds": all(row["selected_sample_count"] > 0 for row in by_target[1]),
        "selected_head_gradient_nonzero_when_samples_exist": all(
            row["selected_head_gradient_nonzero"] for row in flat if row["selected_sample_count"] > 0
        ),
        "shared_trunk_gradient_sharing_explicit_when_samples_exist": all(
            row["shared_trunk_gradient_sharing_explicit"] for row in flat if row["selected_sample_count"] > 0
        ),
        "optimizer_step_not_executed": all(row["optimizer_step_executed"] is False for row in flat),
        "finite": all(row["finite"] for row in flat),
    }
    return {
        "stage": STAGE,
        "checks": checks,
        "gradient_audit_passed": all(checks.values()),
        "cross_specialized_head_gradient_leakage_count": sum(row["cross_specialized_head_gradient_leakage_count"] for row in flat),
        "seed_gradient_probe_rows": seed_rows,
        "training_pressure_by_target": pressure_by_target_from_joined_rows(joined_rows),
    }


def load_validation_plan() -> Dict[str, Any]:
    plan = read_json(H4L_ROOT / "validation_window_resolution.json")
    return {
        **plan,
        "validation_basis_source": str(H4L_ROOT / "validation_window_resolution.json"),
        "test6_access_count": 0,
        "test_snapshot_paths_loaded": plan.get("test_snapshot_paths_loaded", []),
    }


def load_validation_data(dl1: Any, dl4: Any, observation_repair: Any, validation_plan: Mapping[str, Any]) -> Tuple[Any, Any, Dict[str, Any], List[Any]]:
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])
    validation_paths = [Path(row["snapshot_path"]) for row in validation_plan["validation_rows"]]
    sample_full = dl1.torch_load(validation_paths[0])
    spec, inventory, _connectivity, _tensor_mask = dl1.build_subgraph_spec(PROJECT_ROOT, sample_full, mapping_artifact=mapping_artifact)
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    sample_graph, _sample_audit = observation_repair.append_target_context_features(sample_graph, dl1, action_dim=3)
    validation_data_unrepaired = dl4.load_subgraphs(dl1, validation_paths, spec)
    validation_data, audit = observation_repair.repair_data_sequence(validation_data_unrepaired, dl1, action_dim=3)
    config = {
        "spec": spec,
        "effective_agents": min(8, int(inventory["available_suseong_agents"])),
        "action_dim": 3,
        "gatv2_hidden": 128,
        "return_normalization": True,
        "validation_observation_repair_audit": audit,
    }
    return sample_graph, inventory, config, validation_data


def load_models_for_validation(dl1: Any, dl4: Any, payload: Mapping[str, Any], sample_graph: Any, device: torch.device) -> Tuple[Any, Any, Any, Any]:
    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), 128, sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(128, 3, target_context_dim=3, target_head_specialization=True).to(device)
    critic = dl1.CentralizedCritic(128).to(device)
    encoder.load_state_dict(payload["gatv2_state_dict"])
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    critic.load_state_dict(payload["critic_state_dict"])
    return_normalizer = dl4.ReturnNormalizer(True)
    return_normalizer.load_state_dict(payload.get("return_normalizer_state", {}))
    encoder.eval()
    actor.eval()
    critic.eval()
    return encoder, actor, critic, return_normalizer


def target_label(target_id: int) -> str:
    if target_id == 0:
        return CLASS_HOLD_BETTER
    if target_id == 1:
        return CLASS_SERVE_BETTER
    return CLASS_SKIP_OR_OTHER


def summarize_validation_subset(rows: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    action_counts = Counter(row["action_name"] for row in rows)
    count = len(rows)
    prob_correct = sum(1 for row in rows if row.get("probability_correct_direction") is True)
    sampled_correct = sum(1 for row in rows if row.get("sampled_correct_direction") is True)
    summary = {
        "classification": label,
        "count": count,
        "P_HOLD": numeric_stats([row["P_HOLD"] for row in rows]),
        "P_SERVE": numeric_stats([row["P_SERVE"] for row in rows]),
        "P_SKIP": numeric_stats([row["P_SKIP"] for row in rows]),
        "policy_entropy": numeric_stats([row["policy_entropy"] for row in rows]),
        "policy_margin_hold_minus_serve": numeric_stats([row["P_HOLD"] - row["P_SERVE"] for row in rows]),
        "sampled_action_counts": dict(action_counts),
        "probability_correct_direction_count": prob_correct,
        "probability_correct_direction_rate": safe_rate(prob_correct, count),
        "sampled_correct_direction_count": sampled_correct,
        "sampled_correct_direction_rate": safe_rate(sampled_correct, count),
    }
    summary["probability_dominant_action"] = dominant_action_from_probs(summary)
    summary["sampled_dominant_action"] = sampled_dominant_action(action_counts)
    return summary


def validation_discrimination(checkpoint_registry: Mapping[str, Any], validation_plan: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    dl4 = import_module_from_path(f"h4mq_val_dl4_{time.time_ns()}", DL4_SOURCE)
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    observation_repair = import_module_from_path(f"h4mq_val_obs_{time.time_ns()}", OBS_REPAIR_SOURCE)
    sample_graph, inventory, config, validation_data = load_validation_data(dl1, dl4, observation_repair, validation_plan)
    seed_summaries = []
    by_sample_rows = []
    mutation_rows = []
    nan_inf_count = 0
    illegal_count = 0
    for checkpoint in checkpoint_registry.get("checkpoints", []):
        seed = int(checkpoint["seed"])
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        pre_sha = sha256_file(Path(checkpoint["path"]))
        pre_hashes = {
            "actor": dl1.state_dict_hash(payload["actor_state_dict"]),
            "critic": dl1.state_dict_hash(payload["critic_state_dict"]),
            "gatv2": dl1.state_dict_hash(payload["gatv2_state_dict"]),
        }
        encoder, actor, critic, return_normalizer = load_models_for_validation(dl1, dl4, payload, sample_graph, device)
        action_rows = []
        with torch.no_grad():
            for step, (cpu_data, window) in enumerate(zip(validation_data, validation_plan["validation_rows"])):
                data = cpu_data.to(device)
                indices = dl1.agent_indices_for_step(config["spec"], step, int(config["effective_agents"]))
                logits, _critic_out, value_original, agent_mask = dl4.forward_scaled(
                    dl1, data, indices, encoder, actor, critic, return_normalizer
                )
                targets = dl1.action_targets_from_y(data.y[torch.tensor(indices, dtype=torch.long, device=device)], 3)
                masked_logits, allowed = masked_logits_for_targets(logits, targets)
                probs = torch.softmax(masked_logits, dim=-1)
                actions = torch.argmax(masked_logits, dim=-1)
                entropy = Categorical(logits=masked_logits).entropy()
                finite_ok = bool(torch.isfinite(logits).all().detach().cpu().item())
                finite_ok = finite_ok and bool(torch.isfinite(probs).all().detach().cpu().item())
                finite_ok = finite_ok and bool(torch.isfinite(value_original).all().detach().cpu().item())
                if not finite_ok:
                    nan_inf_count += 1
                active = agent_mask.detach().cpu().bool().tolist()
                for slot, is_active in enumerate(active):
                    if not is_active:
                        continue
                    action_id = int(actions[slot].detach().cpu().item())
                    target_id = int(targets[slot].detach().cpu().item())
                    legal = bool(allowed[slot, action_id].detach().cpu().item())
                    if not legal:
                        illegal_count += 1
                    p_hold = float(probs[slot, 0].detach().cpu().item())
                    p_serve = float(probs[slot, 1].detach().cpu().item())
                    p_skip = float(probs[slot, 2].detach().cpu().item())
                    label = target_label(target_id)
                    record = {
                        "seed": seed,
                        "window_id": window["window_id"],
                        "window_position": int(step),
                        "agent_slot": slot,
                        "target_id": target_id,
                        "target_name": ACTION_NAMES[target_id],
                        "classification": label,
                        "action_id": action_id,
                        "action_name": ACTION_NAMES[action_id],
                        "legal_action": legal,
                        "P_HOLD": p_hold,
                        "P_SERVE": p_serve,
                        "P_SKIP": p_skip,
                        "policy_entropy": float(entropy[slot].detach().cpu().item()),
                        "probability_correct_direction": (
                            p_hold > p_serve if label == CLASS_HOLD_BETTER else p_serve > p_hold if label == CLASS_SERVE_BETTER else None
                        ),
                        "sampled_correct_direction": (
                            action_id == 0 if label == CLASS_HOLD_BETTER else action_id == 1 if label == CLASS_SERVE_BETTER else None
                        ),
                    }
                    action_rows.append(record)
                    by_sample_rows.append(record)
        post_sha = sha256_file(Path(checkpoint["path"]))
        mutation_rows.append(
            {
                "seed": seed,
                "checkpoint_path": checkpoint["path"],
                "pre_eval_sha256": pre_sha,
                "post_eval_sha256": post_sha,
                "checkpoint_sha_unchanged": pre_sha == post_sha,
                "actor_parameter_hash_unchanged": pre_hashes["actor"] == dl1.module_hash(actor),
                "critic_parameter_hash_unchanged": pre_hashes["critic"] == dl1.module_hash(critic),
                "gatv2_parameter_hash_unchanged": pre_hashes["gatv2"] == dl1.module_hash(encoder),
            }
        )
        seed_summaries.append(
            {
                "seed": seed,
                "validation_window_count": len(validation_plan["validation_rows"]),
                "action_count": len(action_rows),
                "by_classification": {
                    CLASS_HOLD_BETTER: summarize_validation_subset([row for row in action_rows if row["classification"] == CLASS_HOLD_BETTER], CLASS_HOLD_BETTER),
                    CLASS_SERVE_BETTER: summarize_validation_subset([row for row in action_rows if row["classification"] == CLASS_SERVE_BETTER], CLASS_SERVE_BETTER),
                    CLASS_SKIP_OR_OTHER: summarize_validation_subset([row for row in action_rows if row["classification"] == CLASS_SKIP_OR_OTHER], CLASS_SKIP_OR_OTHER),
                },
            }
        )
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    aggregate = {
        CLASS_HOLD_BETTER: summarize_validation_subset([row for row in by_sample_rows if row["classification"] == CLASS_HOLD_BETTER], CLASS_HOLD_BETTER),
        CLASS_SERVE_BETTER: summarize_validation_subset([row for row in by_sample_rows if row["classification"] == CLASS_SERVE_BETTER], CLASS_SERVE_BETTER),
        CLASS_SKIP_OR_OTHER: summarize_validation_subset([row for row in by_sample_rows if row["classification"] == CLASS_SKIP_OR_OTHER], CLASS_SKIP_OR_OTHER),
    }
    h4mn_pre = read_json(H4M_N_ROOT / "04_conditional_policy_discrimination.json")["by_long_horizon_label"]
    comparison = {
        "pre_repair_behavior_source": str(H4M_N_ROOT / "04_conditional_policy_discrimination.json"),
        "pre_repair_h4m_n_hold_better": h4mn_pre[CLASS_HOLD_BETTER],
        "pre_repair_h4m_n_serve_better": h4mn_pre[CLASS_SERVE_BETTER],
        "h4mq_validation_minus_h4mn_training_probability_correct_direction": {
            CLASS_HOLD_BETTER: (
                aggregate[CLASS_HOLD_BETTER]["probability_correct_direction_rate"]
                - h4mn_pre[CLASS_HOLD_BETTER]["probability_correct_direction_rate"]
            )
            if aggregate[CLASS_HOLD_BETTER]["probability_correct_direction_rate"] is not None
            else None,
            CLASS_SERVE_BETTER: (
                aggregate[CLASS_SERVE_BETTER]["probability_correct_direction_rate"]
                - h4mn_pre[CLASS_SERVE_BETTER]["probability_correct_direction_rate"]
            )
            if aggregate[CLASS_SERVE_BETTER]["probability_correct_direction_rate"] is not None
            else None,
        },
    }
    checks = {
        "validation_scope_passed": validation_plan.get("validation_scope_passed") is True,
        "validation_rows_four": len(validation_plan.get("validation_rows", [])) == 4,
        "test_snapshot_paths_loaded_empty": validation_plan.get("test_snapshot_paths_loaded", []) == [],
        "seed_summaries_all_seeds": len(seed_summaries) == 3,
        "hold_and_serve_labels_measurable": aggregate[CLASS_HOLD_BETTER]["count"] > 0 and aggregate[CLASS_SERVE_BETTER]["count"] > 0,
        "legal_actions_all": illegal_count == 0,
        "nan_inf_zero": nan_inf_count == 0,
        "checkpoint_unmodified_by_validation": all(
            row["checkpoint_sha_unchanged"]
            and row["actor_parameter_hash_unchanged"]
            and row["critic_parameter_hash_unchanged"]
            and row["gatv2_parameter_hash_unchanged"]
            for row in mutation_rows
        ),
    }
    return {
        "stage": STAGE,
        "classification_source": (
            "Frozen H4L validation windows; HOLD_BETTER/SERVE_BETTER labels are derived from "
            "the immutable pre-action target one-hot context appended by H4M-I observation repair."
        ),
        "validation_plan": validation_plan,
        "seed_summaries": seed_summaries,
        "aggregate_by_classification": aggregate,
        "comparison_to_pre_repair_h4m_n_training_behavior": comparison,
        "by_sample_rows": by_sample_rows,
        "mutation_rows": mutation_rows,
        "test6_access_count": 0,
        "illegal_action_count": illegal_count,
        "nan_inf_count": nan_inf_count,
        "checks": checks,
        "validation_discrimination_passed": all(checks.values()),
    }


def seed_metrics(training: Mapping[str, Any], validation: Mapping[str, Any]) -> Dict[str, Any]:
    val_by_seed = {int(row["seed"]): row for row in validation.get("seed_summaries", [])}
    rows = []
    for row in training.get("seed_summaries", []):
        seed = int(row["seed"])
        rows.append(
            {
                "seed": seed,
                "training": row,
                "validation_discrimination": val_by_seed.get(seed),
            }
        )
    return {"stage": STAGE, "seed_rows": rows}


def cycle_action_evolution(training: Mapping[str, Any], conditional: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "cycle_summaries": training.get("cycle_summaries", []),
        "overall_policy_evolution": conditional.get("overall_policy_evolution", []),
        "by_seed_cycle_label": conditional.get("by_seed_cycle_label", []),
    }


def checkpoint_manifest(checkpoint_registry: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for checkpoint in checkpoint_registry.get("checkpoints", []):
        path = Path(checkpoint["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                **checkpoint,
                "sha256_recomputed": sha256_file(path),
                "sha256_match": sha256_file(path) == checkpoint.get("sha256"),
                "stage_in_payload": payload.get("stage"),
                "fresh_initialization": payload.get("fresh_initialization"),
                "actor_target_head_specialization_active": payload.get("actor_target_head_specialization_active"),
                "actor_target_head_count": payload.get("actor_target_head_count"),
                "repair_contract_sha256": payload.get("actor_target_head_specialization_repair_contract_sha256"),
            }
        )
    checks = {
        "checkpoint_count_three": len(rows) == 3,
        "all_sha_match": all(row["sha256_match"] for row in rows),
        "all_fresh": all(row["fresh_initialization"] is True for row in rows),
        "all_specialized": all(row["actor_target_head_specialization_active"] is True for row in rows),
        "all_contract_match": all(row["repair_contract_sha256"] == EXPECTED["h4m_p_repair_contract_sha256"] for row in rows),
    }
    return {"stage": STAGE, "checks": checks, "checkpoint_manifest_passed": all(checks.values()), "checkpoints": rows}


def changed_files() -> Dict[str, Any]:
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    return {
        "stage": STAGE,
        "source_commit": git_run(["rev-parse", "HEAD"]).stdout.strip(),
        "parent_commit": git_run(["rev-parse", "HEAD^"]).stdout.strip(),
        "head_commit_files": head_files,
        "source_only_local_commit": head_files == [str(SOURCE_REL)],
        "artifact_or_log_files_committed": any("artifacts/" in path for path in head_files),
        "github_push_performed": False,
    }


def training_integrity_h4mq(
    kmod: Any,
    binding: Mapping[str, Any],
    training: Mapping[str, Any],
    cycle_summaries: Sequence[Mapping[str, Any]],
    trace_meta: Mapping[str, Any],
    conditional: Mapping[str, Any],
    shadow_summary: Mapping[str, Any],
    credit_summary: Mapping[str, Any],
    validation: Mapping[str, Any],
    gradient: Mapping[str, Any],
    parameter_delta: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
) -> Dict[str, Any]:
    base = kmod.training_integrity(binding, training, cycle_summaries, trace_meta, conditional, shadow_summary, credit_summary)
    extra = {
        "actor_target_head_specialization_active_all_seeds": all(
            row.get("checkpoint", {}).get("actor_target_head_specialization_active") is True
            for row in training.get("seed_summaries", [])
        ),
        "actor_target_head_count_three_all_seeds": all(
            row.get("checkpoint", {}).get("actor_target_head_count") == 3 for row in training.get("seed_summaries", [])
        ),
        "actor_target_head_contract_bound_all_seeds": all(
            row.get("checkpoint", {}).get("actor_target_head_specialization_repair_contract_sha256")
            == EXPECTED["h4m_p_repair_contract_sha256"]
            for row in training.get("seed_summaries", [])
        ),
        "gradient_cross_head_leakage_zero": gradient.get("cross_specialized_head_gradient_leakage_count") == 0,
        "gradient_audit_passed": gradient.get("gradient_audit_passed") is True,
        "parameter_delta_passed": parameter_delta.get("parameter_delta_passed") is True,
        "validation_discrimination_passed": validation.get("validation_discrimination_passed") is True,
        "checkpoint_manifest_passed": checkpoints.get("checkpoint_manifest_passed") is True,
        "test6_access_count_zero": validation.get("test6_access_count") == 0,
    }
    base["criteria"].update(extra)
    base["training_integrity_passed"] = all(base["criteria"].values())
    base["failing_criteria"] = [key for key, value in base["criteria"].items() if not value]
    base["TEST6"] = "SEALED_NOT_OPENED"
    base["optimizer_step_count_training"] = training.get("total_ppo_updates")
    base["optimizer_step_count_gradient_validation"] = 0
    base["nan_inf_count_validation"] = validation.get("nan_inf_count")
    base["illegal_action_count_validation"] = validation.get("illegal_action_count")
    base["cross_specialized_head_gradient_leakage_count"] = gradient.get("cross_specialized_head_gradient_leakage_count")
    return base


def behavioral_classification(validation: Mapping[str, Any], gradient: Mapping[str, Any], integrity: Mapping[str, Any]) -> Dict[str, Any]:
    hold = validation["aggregate_by_classification"][CLASS_HOLD_BETTER]
    serve = validation["aggregate_by_classification"][CLASS_SERVE_BETTER]
    hold_dom = hold["probability_dominant_action"]
    serve_dom = serve["probability_dominant_action"]
    hold_sample_dom = hold["sampled_dominant_action"]
    serve_sample_dom = serve["sampled_dominant_action"]
    h4mn_delta_hold = validation["comparison_to_pre_repair_h4m_n_training_behavior"][
        "h4mq_validation_minus_h4mn_training_probability_correct_direction"
    ][CLASS_HOLD_BETTER]
    serve_retained = serve_dom == ACTION_SERVE or serve["probability_correct_direction_rate"] == 1.0
    if integrity.get("training_integrity_passed") is not True or gradient.get("gradient_audit_passed") is not True:
        outcome = "NEW_COLLAPSE_OR_INSTABILITY"
    elif hold_dom == ACTION_HOLD and serve_dom == ACTION_SERVE:
        outcome = "DISCRIMINATION_RESTORED"
    elif hold_dom == ACTION_HOLD and serve_dom == ACTION_HOLD:
        outcome = "GLOBAL_HOLD_SHIFT_WITHOUT_CONTEXT_DISCRIMINATION"
    elif hold_dom == ACTION_SERVE and serve_dom == ACTION_SERVE and hold_sample_dom == ACTION_SERVE and serve_sample_dom == ACTION_SERVE:
        outcome = "SERVE_COLLAPSE_PERSISTS"
    elif h4mn_delta_hold is not None and h4mn_delta_hold > 0.0 and serve_retained:
        outcome = "PARTIAL_DISCRIMINATION"
    else:
        outcome = "NEW_COLLAPSE_OR_INSTABILITY"
    return {
        "stage": STAGE,
        "behavioral_outcome": outcome,
        "exact_next_gate": NEXT_DECISION_GATE,
        "rule": {
            "no_arbitrary_tolerance": True,
            "DISCRIMINATION_RESTORED": "HOLD_BETTER probability dominant action is HOLD and SERVE_BETTER probability dominant action is SERVE.",
            "GLOBAL_HOLD_SHIFT_WITHOUT_CONTEXT_DISCRIMINATION": "HOLD_BETTER and SERVE_BETTER both probability-dominant HOLD.",
            "SERVE_COLLAPSE_PERSISTS": "HOLD_BETTER and SERVE_BETTER both probability- and sampled-dominant SERVE.",
            "PARTIAL_DISCRIMINATION": "HOLD_BETTER correct-direction rate improves over H4M-N pre-repair evidence while SERVE_BETTER is retained.",
            "NEW_COLLAPSE_OR_INSTABILITY": "Integrity/gradient failure or no unique discrimination/collapse class.",
        },
        "evidence": {
            "hold_better_probability_dominant_action": hold_dom,
            "serve_better_probability_dominant_action": serve_dom,
            "hold_better_sampled_dominant_action": hold_sample_dom,
            "serve_better_sampled_dominant_action": serve_sample_dom,
            "hold_better_probability_correct_direction_rate": hold["probability_correct_direction_rate"],
            "serve_better_probability_correct_direction_rate": serve["probability_correct_direction_rate"],
            "h4mq_minus_h4mn_hold_probability_correct_direction_rate": h4mn_delta_hold,
            "cross_specialized_head_gradient_leakage_count": gradient.get("cross_specialized_head_gradient_leakage_count"),
        },
    }


def gate_matrix(
    binding: Mapping[str, Any],
    integrity: Mapping[str, Any],
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    gradient: Mapping[str, Any],
    parameter_delta: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    behavior: Mapping[str, Any],
    changed: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_bindings_match": binding.get("authoritative_binding_passed") is True,
        "source_only_local_commit": changed.get("source_only_local_commit") is True,
        "training_integrity_pass": integrity.get("training_integrity_passed") is True,
        "seeds_1_2_3_complete": training.get("seeds") == EXPECTED["seeds"],
        "total_rollouts_33": training.get("total_rollouts") == 33,
        "total_ppo_updates_132": training.get("total_ppo_updates") == 132,
        "total_critic_updates_264": training.get("total_critic_updates") == 264,
        "actor_head_specialization_active": integrity.get("criteria", {}).get("actor_target_head_specialization_active_all_seeds") is True,
        "gradient_cross_head_leakage_zero": gradient.get("cross_specialized_head_gradient_leakage_count") == 0,
        "parameter_delta_pass": parameter_delta.get("parameter_delta_passed") is True,
        "validation_discrimination_pass": validation.get("validation_discrimination_passed") is True,
        "checkpoint_manifest_pass": checkpoints.get("checkpoint_manifest_passed") is True,
        "behavioral_outcome_classified": behavior.get("behavioral_outcome") in BEHAVIORAL_OUTCOMES,
        "test6_access_count_zero": validation.get("test6_access_count") == 0,
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "execution_decision": PASS_DECISION if passed else "H4M_Q_EXECUTION_BLOCKED",
        "behavioral_outcome": behavior.get("behavioral_outcome"),
        "decision": behavior.get("behavioral_outcome") if passed else "H4M_Q_BLOCKED_REVIEW_EVIDENCE",
        "exact_next_gate": NEXT_DECISION_GATE if passed else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "repair_contract_sha256": EXPECTED["h4m_p_repair_contract_sha256"],
        "final_flags": {
            "fresh_three_seed_training_executed": passed,
            "additional_repair_performed": False,
            "reward_gae_ppo_modified": False,
            "training_budget_changed": False,
            "validation_discrimination_audit_executed": passed,
            "test6_opened_or_used": False,
            "github_push_performed": False,
        },
    }


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files = {}
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
        "behavioral_outcome": gate.get("behavioral_outcome"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": output_files,
        "output_sha256": output_sha,
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift",
        "github_push_performed": False,
        "TEST6_opened": False,
        "test6_access_count": 0,
    }


def final_report(
    binding: Mapping[str, Any],
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    gradient: Mapping[str, Any],
    parameter_delta: Mapping[str, Any],
    behavior: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    hold = validation["aggregate_by_classification"][CLASS_HOLD_BETTER]
    serve = validation["aggregate_by_classification"][CLASS_SERVE_BETTER]
    pressure = gradient.get("training_pressure_by_target", [])
    seed_counts = [
        {
            "seed": row["seed"],
            "rollouts": row["rollouts"],
            "ppo_updates": row["ppo_updates"],
            "critic_updates": row["critic_updates"],
            "active_samples": row["active_samples"],
            "actor_target_head_specialization_active": row.get("checkpoint", {}).get("actor_target_head_specialization_active"),
        }
        for row in training.get("seed_summaries", [])
    ]
    return f"""# H4M-Q Fresh Target-Conditioned Actor-Head Specialized Three-Seed Retraining

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_q_source_git_commit"]}
execution_decision = {gate["execution_decision"]}
behavioral_outcome = {gate["behavioral_outcome"]}
exact_next_gate = {gate["exact_next_gate"]}
repair_contract_sha256 = {EXPECTED["h4m_p_repair_contract_sha256"]}

## Three-seed execution counts

```json
{json.dumps(seed_counts, ensure_ascii=False, indent=2, default=jsonable)}
```

## Frozen validation discrimination — HOLD_BETTER

```json
{json.dumps(hold, ensure_ascii=False, indent=2, default=jsonable)}
```

## Frozen validation discrimination — SERVE_BETTER

```json
{json.dumps(serve, ensure_ascii=False, indent=2, default=jsonable)}
```

## Gradient isolation

```json
{json.dumps({"checks": gradient.get("checks"), "cross_specialized_head_gradient_leakage_count": gradient.get("cross_specialized_head_gradient_leakage_count"), "training_pressure_by_target": pressure}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Parameter delta

```json
{json.dumps({"checks": parameter_delta.get("checks")}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Behavioral classification

```json
{json.dumps(behavior, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no automatic repair, no retune, no training extension, no Reward/GAE/PPO modification, no TEST6, no GitHub push.
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
        "execution_decision": "H4M_Q_EXECUTION_BLOCKED",
        "decision": "H4M_Q_BLOCKED_REVIEW_EVIDENCE",
        "behavioral_outcome": "NEW_COLLAPSE_OR_INSTABILITY",
        "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "block_reason": reason,
        "extra": extra or {},
    }
    empty = {"stage": STAGE, "created_at": created_at, "not_executed_due_to": reason}
    payloads = {
        "repair_binding.json": binding,
        "training_summary.json": empty,
        "seed_metrics.json": empty,
        "cycle_action_evolution.json": empty,
        "validation_discrimination.json": empty,
        "parameter_delta.json": empty,
        "gradient_audit.json": empty,
        "checkpoint_manifest.json": empty,
        "training_integrity.json": empty,
        "changed_files.json": changed_files() if (PROJECT_ROOT / SOURCE_REL).exists() else empty,
        "test_results.json": binding.get("compile_static_checks", empty),
        "gate_matrix.json": gate,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(f"# H4M-Q\n\ngate = {BLOCK_GATE}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-Q] artifact root: {artifact_root}")
    print(f"[H4M-Q] gate: {BLOCK_GATE}")
    print(f"[H4M-Q] block_reason: {reason}")


def write_parquet_large_rows(artifact_root: Path, validation: Mapping[str, Any]) -> Dict[str, str]:
    outputs = {}
    rows = validation.get("by_sample_rows", [])
    if rows:
        path = artifact_root / "validation_discrimination_by_sample.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        outputs["validation_discrimination_by_sample.parquet"] = str(path)
    return outputs


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_{stamp}"

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

    kmod = configure_h4mk_module(import_module_from_path(f"h4mq_h4mk_{time.time_ns()}", H4MK_SOURCE))
    base, _rerun = kmod.load_h4m_helpers()
    h4mg = kmod.load_h4mg()
    device = torch.device("mps")

    training, cycle_summaries, checkpoint_registry_raw, trace_meta = kmod.run_training(base, h4mg, created_at, artifact_root, device)
    for row in training.get("seed_summaries", []):
        row["actor_target_context_dim"] = row.get("checkpoint", {}).get("actor_target_context_dim")
        row["actor_conditioned_input_dim"] = row.get("checkpoint", {}).get("actor_conditioned_input_dim")
        row["actor_target_head_specialization_active"] = row.get("checkpoint", {}).get("actor_target_head_specialization_active")
        row["actor_target_head_count"] = row.get("checkpoint", {}).get("actor_target_head_count")
    training["actor_architecture"] = "MAPPOActor(hidden=128, action_dim=3, target_context_dim=3, target_head_specialization=True)"
    training["actor_target_head_specialization_repair_contract_sha256"] = EXPECTED["h4m_p_repair_contract_sha256"]
    training["only_intended_training_change"] = "legacy/shared decision head replaced by H4M-P target-conditioned specialized decision heads"
    checkpoint_registry_raw["actor_target_head_specialization_repair_contract_sha256"] = EXPECTED["h4m_p_repair_contract_sha256"]
    checkpoint_registry_raw["actor_architecture"] = training["actor_architecture"]

    conditional = kmod.conditional_policy_discrimination(trace_meta["joined_rows"])
    conditional["conditional_policy_by_sample_parquet"] = kmod.write_conditional_parquet(base, artifact_root, trace_meta["joined_rows"])
    shadow_summary = kmod.long_horizon_shadow_summary(cycle_summaries, trace_meta)
    credit_summary = kmod.credit_trace_summary(trace_meta)
    collapse = kmod.policy_collapse_audit(conditional)

    validation_plan = load_validation_plan()
    validation = validation_discrimination(checkpoint_registry_raw, validation_plan, device)
    large_rows = write_parquet_large_rows(artifact_root, validation)
    validation["large_row_data"] = large_rows
    checkpoints = checkpoint_manifest(checkpoint_registry_raw)
    parameter_delta = parameter_delta_audit(h4mg, checkpoint_registry_raw, created_at, device, trace_meta["joined_rows"])
    gradient = gradient_audit(h4mg, checkpoint_registry_raw, trace_meta["joined_rows"], created_at, device)
    integrity = training_integrity_h4mq(
        kmod,
        binding,
        training,
        cycle_summaries,
        trace_meta,
        conditional,
        shadow_summary,
        credit_summary,
        validation,
        gradient,
        parameter_delta,
        checkpoints,
    )
    behavior = behavioral_classification(validation, gradient, integrity)
    changed = changed_files()
    test_results = {
        "stage": STAGE,
        "py_compile": compile_static,
        "commands": [
            f"{sys.executable} -m py_compile {SOURCE_REL}",
            "git diff --cached --check",
            f"{sys.executable} {SOURCE_REL}",
        ],
        "training_optimizer_step_count": training.get("total_ppo_updates"),
        "gradient_validation_optimizer_step_count": 0,
        "test6_access_count": 0,
    }
    gate = gate_matrix(binding, integrity, training, validation, gradient, parameter_delta, checkpoints, behavior, changed)

    payloads = {
        "repair_binding.json": binding,
        "training_summary.json": training,
        "seed_metrics.json": seed_metrics(training, validation),
        "cycle_action_evolution.json": cycle_action_evolution(training, conditional),
        "validation_discrimination.json": validation,
        "parameter_delta.json": parameter_delta,
        "gradient_audit.json": gradient,
        "checkpoint_manifest.json": checkpoints,
        "training_integrity.json": integrity,
        "changed_files.json": changed,
        "test_results.json": test_results,
        "gate_matrix.json": gate,
        "conditional_policy_discrimination_training_trace.json": conditional,
        "long_horizon_shadow_summary.json": shadow_summary,
        "credit_trace_summary.json": credit_summary,
        "policy_collapse_audit_training_trace.json": collapse,
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(
        final_report(binding, training, validation, gradient, parameter_delta, behavior, gate),
        encoding="utf-8",
    )
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))

    hold = validation["aggregate_by_classification"][CLASS_HOLD_BETTER]
    serve = validation["aggregate_by_classification"][CLASS_SERVE_BETTER]
    print(f"[H4M-Q] artifact root: {artifact_root}")
    print(f"[H4M-Q] gate: {gate['gate']}")
    print(f"[H4M-Q] source_commit: {provenance['h4m_q_source_git_commit']}")
    print(
        f"[H4M-Q] total_rollouts={training['total_rollouts']} "
        f"total_ppo_updates={training['total_ppo_updates']} total_critic_updates={training['total_critic_updates']}"
    )
    print(
        "[H4M-Q] validation_hold_better="
        f"prob_dom={hold['probability_dominant_action']} prob_correct={hold['probability_correct_direction_rate']} "
        f"sampled_correct={hold['sampled_correct_direction_rate']}"
    )
    print(
        "[H4M-Q] validation_serve_better="
        f"prob_dom={serve['probability_dominant_action']} prob_correct={serve['probability_correct_direction_rate']} "
        f"sampled_correct={serve['sampled_correct_direction_rate']}"
    )
    print(f"[H4M-Q] cross_head_gradient_leakage={gradient['cross_specialized_head_gradient_leakage_count']}")
    print(f"[H4M-Q] behavioral_outcome: {gate['behavioral_outcome']}")
    print(f"[H4M-Q] exact_next_gate: {gate['exact_next_gate']}")


if __name__ == "__main__":
    main()
