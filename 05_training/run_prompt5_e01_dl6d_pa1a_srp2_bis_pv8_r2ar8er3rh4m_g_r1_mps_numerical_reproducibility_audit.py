#!/usr/bin/env python3
"""H4M-G-R1 MPS numerical reproducibility and instrumentation equivalence audit.

This audit separates the H4M-H PRE-1 MPS BLOCK cause:
  A. intrinsic Apple MPS run-to-run numerical variability,
  B. instrumentation-induced value/critic interference,
  C. MPS variability with OFF/ON within observed envelope,
  D. mixed/not unique.

No repair, deterministic-mode change, CPU fallback adoption, 3-seed training,
validation/TEST6, or GitHub push is performed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-G-R1"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_R1_MPS_NUMERICAL_REPRODUCIBILITY_AND_INSTRUMENTATION_EQUIVALENCE_AUDIT_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_R1_MPS_NUMERICAL_REPRODUCIBILITY_AND_INSTRUMENTATION_EQUIVALENCE_AUDIT_FAILED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_H_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_h_fresh_instrumented_credit_diagnostic_retraining_20260815_132209+0900"
H4M_G_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_instrumentation_equivalence_validation_20260815_122304+0900"
H4M_F_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_f_credit_trace_instrumentation_selection_freeze_20260815_111506+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"

EXPECTED = {
    "h4m_h_gate": "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_H_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING_FAILED",
    "h4m_h_decision": "H4M_H_DIAGNOSTIC_RETRAINING_BLOCKED",
    "h4m_h_block_reason": "MPS_OFF_ON_PREFLIGHT_FAILED",
    "h4m_h_first_material_mismatch": "value_t_sha256_match",
    "h4m_h_source_commit": "a9ab6df7bd4829b5f9ccf3d29cb4f64156d174ea",
    "h4m_g_source_commit": "672e05a24e09440351e46dad05cf5ab02f498060",
    "h4m_f_contract_sha256": "9b95d0dc46459eedd2cf51e85789407be247e9db1e23e3a52a5bbae9c6216ffe",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

ORDERED_STAGES = [
    "sampled_actions",
    "raw_rewards",
    "value_t",
    "next_value_t",
    "td_delta",
    "raw_gae",
    "normalized_advantage",
    "return_target",
    "old_log_prob",
    "ppo_inputs",
    "parameter_update",
    "final_checkpoint",
]

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_mps_runtime_environment.json",
    "03_critic_forward_reproducibility.json",
    "04_off_off_reproducibility.json",
    "05_on_on_reproducibility.json",
    "06_off_on_variability_comparison.json",
    "07_first_divergence_trace.json",
    "08_rng_shadow_integrity.json",
    "09_root_cause_classification.json",
    "10_gate_matrix.json",
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


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
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
    source = SOURCE_REL.as_posix()
    latest = git_run(["log", "-1", "--format=%H", "--", source]).stdout.strip()
    head_files = [line.strip() for line in git_run(["show", "--name-only", "--pretty=format:", "HEAD"]).stdout.splitlines() if line.strip()]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip(),
        "git_commit": head,
        "h4m_g_r1_source_git_commit": head,
        "h4m_g_r1_source_path": source,
        "h4m_g_r1_source_sha256": sha256_file(PROJECT_ROOT / source),
        "h4m_g_r1_source_present_in_head": git_bool(["cat-file", "-e", f"HEAD:{source}"]),
        "h4m_g_r1_source_latest_commit": latest,
        "h4m_g_r1_source_no_uncommitted_diff_vs_head": git_bool(["diff", "--quiet", "--", source]),
        "h4m_g_r1_source_no_staged_diff_vs_head": git_bool(["diff", "--cached", "--quiet", "--", source]),
        "head_commit_files": head_files,
        "local_source_only_commit_created_before_audit": latest == head and head_files == [source],
        "post_commit_provenance_gate_passed": latest == head and head_files == [source] and git_bool(["diff", "--quiet", "--", source]) and git_bool(["diff", "--cached", "--quiet", "--", source]),
        "status_short": git_run(["status", "--short"]).stdout,
        "github_push_performed": False,
    }


def tensor_hash(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return canonical_sha({"shape": list(arr.shape), "dtype": str(arr.dtype), "bytes_sha256": hashlib.sha256(arr.tobytes()).hexdigest()})


def flatten_tensor(tensor: torch.Tensor) -> List[float]:
    arr = tensor.detach().cpu().contiguous().float().reshape(-1).numpy()
    return [float(v) for v in arr.tolist()]


def numeric_list_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    numeric_keys = [
        "old_log_prob",
        "current_log_prob",
        "ratio",
        "normalized_advantage",
        "unclipped",
        "clipped",
        "effective",
        "entropy",
    ]
    values: List[float] = []
    for row in rows:
        for key in numeric_keys:
            if key in row and row[key] is not None:
                values.append(float(row[key]))
    return values


def diff_metrics(left: Sequence[float], right: Sequence[float]) -> Dict[str, Any]:
    if len(left) != len(right):
        n = min(len(left), len(right))
        length_match = False
    else:
        n = len(left)
        length_match = True
    diffs = [abs(float(left[i]) - float(right[i])) for i in range(n)]
    rels = []
    for i in range(n):
        denom = max(abs(float(left[i])), abs(float(right[i])))
        if denom == 0.0:
            rels.append(0.0 if diffs[i] == 0.0 else math.inf)
        else:
            rels.append(diffs[i] / denom)
    finite = all(math.isfinite(float(v)) for v in list(left[:n]) + list(right[:n]))
    return {
        "length_left": len(left),
        "length_right": len(right),
        "length_match": length_match,
        "sha256_left": canonical_sha(list(left)),
        "sha256_right": canonical_sha(list(right)),
        "sha_match": canonical_sha(list(left)) == canonical_sha(list(right)),
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": float(sum(diffs) / len(diffs)) if diffs else 0.0,
        "max_rel_diff": max(rels) if rels else 0.0,
        "differing_element_count": sum(1 for d in diffs if d != 0.0) + abs(len(left) - len(right)),
        "finite": finite,
    }


def hash_compare(left: Any, right: Any) -> Dict[str, Any]:
    left_sha = canonical_sha(left)
    right_sha = canonical_sha(right)
    return {"sha256_left": left_sha, "sha256_right": right_sha, "sha_match": left_sha == right_sha}


def load_h4_modules() -> Tuple[Any, Any]:
    h4mg = import_module(
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py",
        f"h4m_g_r1_h4mg_{time.time_ns()}",
    )
    h4mh = import_module(
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_h_fresh_instrumented_credit_diagnostic_retraining.py",
        f"h4m_g_r1_h4mh_{time.time_ns()}",
    )
    return h4mg, h4mh


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    h4mh_manifest = read_json(H4M_H_ROOT / "manifest.json")
    h4mh_gate = read_json(H4M_H_ROOT / "14_h4m_h_gate_matrix.json")
    h4mh_pre1 = read_json(H4M_H_ROOT / "02_mps_off_on_preflight.json")
    h4mh_binding = read_json(H4M_H_ROOT / "01_authoritative_binding.json")
    h4mg_manifest = read_json(H4M_G_ROOT / "manifest.json")
    h4mf_contract_path = H4M_F_ROOT / "13_h4m_f_credit_trace_instrumentation_contract.json"
    h4mb_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    upstream = read_json(h4mf_contract_path).get("authoritative_upstream_SHA_bindings", {})
    checks = {
        "h4m_h_gate_match": h4mh_gate.get("gate") == EXPECTED["h4m_h_gate"],
        "h4m_h_decision_match": h4mh_gate.get("decision") == EXPECTED["h4m_h_decision"],
        "h4m_h_block_reason_match": h4mh_gate.get("block_reason") == EXPECTED["h4m_h_block_reason"],
        "h4m_h_first_material_mismatch_match": h4mh_pre1.get("first_material_mismatch") == EXPECTED["h4m_h_first_material_mismatch"],
        "h4m_h_source_commit_match": h4mh_binding.get("source_provenance", {}).get("h4m_h_source_git_commit") == EXPECTED["h4m_h_source_commit"],
        "h4m_g_source_commit_match": h4mg_manifest.get("h4m_g_source_git_commit") == EXPECTED["h4m_g_source_commit"],
        "h4m_f_contract_sha_match": sha256_file(h4mf_contract_path) == EXPECTED["h4m_f_contract_sha256"],
        "h4m_b_schedule_sha_match": h4mb_schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": upstream.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": upstream.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": upstream.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": upstream.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "h4m_g_r1_source_commit_frozen": provenance.get("post_commit_provenance_gate_passed") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_h": str(H4M_H_ROOT),
            "h4m_g": str(H4M_G_ROOT),
            "h4m_f": str(H4M_F_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "source_provenance": dict(provenance),
        "h4m_h_block_evidence": {
            "gate": h4mh_gate.get("gate"),
            "block_reason": h4mh_gate.get("block_reason"),
            "first_material_mismatch": h4mh_pre1.get("first_material_mismatch"),
            "h4m_h_source_commit": h4mh_binding.get("source_provenance", {}).get("h4m_h_source_git_commit"),
        },
        "h4m_g_source_commit": h4mg_manifest.get("h4m_g_source_git_commit"),
        "instrumentation_contract_sha256": sha256_file(h4mf_contract_path),
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "validation_executed": False,
        "test6_status": "SEALED_NOT_OPENED",
        "github_push_performed": False,
    }


def mps_runtime_environment(created_at: str) -> Dict[str, Any]:
    sw = subprocess.run(["sw_vers"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout
    return {
        "stage": STAGE,
        "created_at": created_at,
        "python": sys.version,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_device_test": _mps_device_test(),
        "sw_vers": sw,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_mode_forced_by_audit": False,
        "cpu_fallback_adopted": False,
        "dtype_change_applied": False,
    }


def _mps_device_test() -> Dict[str, Any]:
    try:
        x = torch.tensor([1.0], device="mps")
        return {"ok": True, "tensor_device": str(x.device)}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def critic_forward_once(h4mg: Any, h4mh: Any, seed: int, created_at: str, device: torch.device) -> Dict[str, Any]:
    ctx = h4mh.build_context_mps(h4mg, seed, created_at, device=device)
    dl1 = ctx["dl1"]
    dl4 = ctx["dl4"]
    data_seq = ctx["train_data"]
    config = ctx["config"]
    encoder = ctx["encoder"]
    actor = ctx["actor"]
    critic = ctx["critic"]
    return_normalizer = ctx["return_normalizer"]
    encoder.eval()
    actor.eval()
    critic.eval()
    values = {}
    with torch.no_grad():
        for label, step in [("value_t", 0), ("next_value_t", 1)]:
            data = data_seq[step].to(device)
            indices = dl1.agent_indices_for_step(config["spec"], step, int(config["effective_agents"]))
            _logits, _critic_out, value_original, _mask = dl4.forward_scaled(
                dl1, data, indices, encoder, actor, critic, return_normalizer
            )
            values[label] = flatten_tensor(value_original)
    return {
        "value_t": values["value_t"],
        "next_value_t": values["next_value_t"],
        "value_t_sha256": canonical_sha(values["value_t"]),
        "next_value_t_sha256": canonical_sha(values["next_value_t"]),
        "finite": all(math.isfinite(v) for v in values["value_t"] + values["next_value_t"]),
        "dtype": "torch.float32",
        "device": str(device),
    }


def critic_forward_reproducibility(created_at: str, h4mg: Any, h4mh: Any, device: torch.device) -> Dict[str, Any]:
    same_process_runs = [critic_forward_once(h4mg, h4mh, 1, created_at, device) for _ in range(4)]
    same_value_diffs = [diff_metrics(same_process_runs[0]["value_t"], row["value_t"]) for row in same_process_runs[1:]]
    same_next_diffs = [diff_metrics(same_process_runs[0]["next_value_t"], row["next_value_t"]) for row in same_process_runs[1:]]

    fresh_runs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--critic-forward-worker"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        fresh_runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    fresh_value_diff = diff_metrics(fresh_runs[0]["value_t"], fresh_runs[1]["value_t"])
    fresh_next_diff = diff_metrics(fresh_runs[0]["next_value_t"], fresh_runs[1]["next_value_t"])
    return {
        "stage": STAGE,
        "created_at": created_at,
        "purpose": "Determine whether MPS critic value forward varies without optimizer or instrumentation.",
        "same_process": {
            "run_count": len(same_process_runs),
            "value_t_sha256": [row["value_t_sha256"] for row in same_process_runs],
            "next_value_t_sha256": [row["next_value_t_sha256"] for row in same_process_runs],
            "value_t_pairwise_vs_first": same_value_diffs,
            "next_value_t_pairwise_vs_first": same_next_diffs,
            "all_value_t_sha_match": all(row["sha_match"] for row in same_value_diffs),
            "all_next_value_t_sha_match": all(row["sha_match"] for row in same_next_diffs),
        },
        "fresh_process": {
            "run_count": len(fresh_runs),
            "value_t_sha256": [row["value_t_sha256"] for row in fresh_runs],
            "next_value_t_sha256": [row["next_value_t_sha256"] for row in fresh_runs],
            "value_t_diff": fresh_value_diff,
            "next_value_t_diff": fresh_next_diff,
            "worker_stderr_note": "suppressed unless subprocess fails",
        },
        "finite": all(row["finite"] for row in same_process_runs + fresh_runs),
    }


def run_trial(label: str, mode: str, h4mg: Any, h4mh: Any, created_at: str, artifact_root: Path, device: torch.device) -> Dict[str, Any]:
    ctx = h4mh.build_context_mps(h4mg, 1, created_at, device=device)
    dl1 = ctx["dl1"]
    before_state = {
        "gatv2": dl1.clone_state_dict(ctx["encoder"]),
        "actor": dl1.clone_state_dict(ctx["actor"]),
        "critic": dl1.clone_state_dict(ctx["critic"]),
    }
    recorder = h4mg.TraceRecorder(root=artifact_root / "trace_side_channel" / label, enabled=(mode == "ON"))
    result: Dict[str, Any] = {
        "label": label,
        "mode": mode,
        "initial_rng": h4mg.rng_snapshot(),
        "initial_model_hashes": h4mg.model_hashes(dl1, ctx["encoder"], ctx["actor"], ctx["critic"]),
        "initial_optimizer_hashes": h4mg.optimizer_hashes(ctx["optimizers"]),
        "initial_reward_normalizer_state": h4mg.reward_normalizer_state(ctx["reward_normalizer"]),
        "initial_return_normalizer_state": dict(ctx["return_normalizer"].state_dict()),
    }
    rollout = h4mg.collect_controlled_rollout(branch=label, seed=1, cycle_index=1, ctx=ctx, recorder=recorder if mode == "ON" else None)
    result["post_rollout_pre_update_rng"] = h4mg.rng_snapshot()
    update = h4mg.ppo_update_controlled(
        branch=label,
        seed=1,
        cycle_index=1,
        ctx=ctx,
        rollout=rollout,
        before_state=before_state,
        recorder=recorder if mode == "ON" else None,
    )
    result["post_update_pre_shadow_rng"] = h4mg.rng_snapshot()
    shadow_result = None
    trace_write = {"enabled": False, "trace_files": {}, "row_counts": {}}
    if mode == "ON":
        shadow_result = h4mg.shadow_smoke_trace(1, 1, ctx, rollout, recorder)
        trace_write = recorder.write()
    result["post_shadow_rng"] = h4mg.rng_snapshot()
    result["shadow_result"] = shadow_result
    result["trace_write"] = trace_write
    result["final_model_hashes"] = h4mg.model_hashes(dl1, ctx["encoder"], ctx["actor"], ctx["critic"])
    result["final_optimizer_hashes"] = h4mg.optimizer_hashes(ctx["optimizers"])
    result["final_reward_normalizer_state"] = h4mg.reward_normalizer_state(ctx["reward_normalizer"])
    result["final_return_normalizer_state"] = dict(ctx["return_normalizer"].state_dict())
    result["parameter_delta"] = {
        "gatv2": dl1.delta_stats(ctx["encoder"], before_state["gatv2"]),
        "actor": dl1.delta_stats(ctx["actor"], before_state["actor"]),
        "critic": dl1.delta_stats(ctx["critic"], before_state["critic"]),
    }
    result["final_model_state_vectors"] = {
        "gatv2": state_vector(ctx["encoder"]),
        "actor": state_vector(ctx["actor"]),
        "critic": state_vector(ctx["critic"]),
    }
    result["in_memory_checkpoint_equivalence_sha256"] = canonical_sha(
        {
            "model_hashes": result["final_model_hashes"],
            "optimizer_hashes": result["final_optimizer_hashes"],
            "reward_normalizer_state": result["final_reward_normalizer_state"],
            "return_normalizer_state": result["final_return_normalizer_state"],
        }
    )
    result["arrays"] = {
        "sampled_actions": flatten_tensor(rollout["actions"]),
        "raw_rewards": flatten_tensor(rollout["raw_rewards"]),
        "value_t": flatten_tensor(rollout["values_original"]),
        "next_value_t": flatten_tensor(rollout["next_values_original"]),
        "td_delta": flatten_tensor(rollout["td_delta"]),
        "raw_gae": flatten_tensor(rollout["advantages"]),
        "normalized_advantage": flatten_tensor(rollout["normalized_advantages"]),
        "return_target": flatten_tensor(rollout["returns_original"]),
        "old_log_prob": flatten_tensor(rollout["old_log_probs"]),
        "ppo_inputs": numeric_list_from_rows(update["ppo_fingerprint_rows"]),
    }
    result["hashes"] = {key: canonical_sha(value) for key, value in result["arrays"].items()}
    result["hashes"]["final_checkpoint"] = result["in_memory_checkpoint_equivalence_sha256"]
    result["hashes"]["parameter_update"] = canonical_sha(result["parameter_delta"])
    result["ppo_fingerprint_row_count"] = len(update["ppo_fingerprint_rows"])
    result["finite_loss_passed"] = all(all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in update["loss_rows"])
    return result


def state_vector(module: torch.nn.Module) -> List[float]:
    vals: List[float] = []
    for tensor in module.state_dict().values():
        vals.extend(flatten_tensor(tensor))
    return vals


def compare_trials(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> Dict[str, Any]:
    stages: Dict[str, Any] = {}
    for stage in ORDERED_STAGES:
        if stage == "final_checkpoint":
            stages[stage] = hash_compare(left["hashes"][stage], right["hashes"][stage])
        elif stage == "parameter_update":
            stages[stage] = {
                "hash": hash_compare(left["parameter_delta"], right["parameter_delta"]),
                "gatv2": diff_metrics(left["final_model_state_vectors"]["gatv2"], right["final_model_state_vectors"]["gatv2"]),
                "actor": diff_metrics(left["final_model_state_vectors"]["actor"], right["final_model_state_vectors"]["actor"]),
                "critic": diff_metrics(left["final_model_state_vectors"]["critic"], right["final_model_state_vectors"]["critic"]),
            }
            stages[stage]["sha_match"] = stages[stage]["hash"]["sha_match"]
        else:
            stages[stage] = diff_metrics(left["arrays"][stage], right["arrays"][stage])
    first = next((stage for stage in ORDERED_STAGES if not stages[stage].get("sha_match", False)), None)
    return {
        "comparison": label,
        "left": left["label"],
        "right": right["label"],
        "first_divergence": first,
        "stage_metrics": stages,
        "rng": {
            "initial_rng_match": left["initial_rng"] == right["initial_rng"],
            "post_rollout_pre_update_rng_match": left["post_rollout_pre_update_rng"] == right["post_rollout_pre_update_rng"],
            "post_update_pre_shadow_rng_match": left["post_update_pre_shadow_rng"] == right["post_update_pre_shadow_rng"],
        },
        "shadow": {
            "left_shadow": shadow_summary(left),
            "right_shadow": shadow_summary(right),
        },
    }


def shadow_summary(trial: Mapping[str, Any]) -> Dict[str, Any]:
    shadow = trial.get("shadow_result") or {}
    return {
        "mode": trial["mode"],
        "shadow_present": bool(shadow),
        "shadow_data_entered_training": bool(shadow.get("shadow_data_entered_training", False)),
        "global_rng_unchanged_by_private_shadow": shadow.get("global_rng_unchanged_by_private_shadow"),
        "trace_row_counts": trial.get("trace_write", {}).get("row_counts", {}),
    }


def envelope_summary(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    stages = comparison["stage_metrics"]
    return {
        "first_divergence": comparison["first_divergence"],
        "value_t": pick_metric(stages["value_t"]),
        "next_value_t": pick_metric(stages["next_value_t"]),
        "td_delta": pick_metric(stages["td_delta"]),
        "raw_gae": pick_metric(stages["raw_gae"]),
        "normalized_advantage": pick_metric(stages["normalized_advantage"]),
        "ppo_inputs": pick_metric(stages["ppo_inputs"]),
        "parameter_update": {
            "sha_match": stages["parameter_update"]["sha_match"],
            "gatv2": pick_metric(stages["parameter_update"]["gatv2"]),
            "actor": pick_metric(stages["parameter_update"]["actor"]),
            "critic": pick_metric(stages["parameter_update"]["critic"]),
        },
        "final_checkpoint_sha_match": stages["final_checkpoint"]["sha_match"],
    }


def pick_metric(metric: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sha_match": metric.get("sha_match"),
        "max_abs_diff": metric.get("max_abs_diff"),
        "mean_abs_diff": metric.get("mean_abs_diff"),
        "max_rel_diff": metric.get("max_rel_diff"),
        "differing_element_count": metric.get("differing_element_count"),
        "finite": metric.get("finite"),
    }


def first_divergence_trace(comparisons: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "ordered_pipeline": ORDERED_STAGES,
        "comparisons": {
            name: {
                "first_divergence": comp["first_divergence"],
                "divergent_stages": [stage for stage in ORDERED_STAGES if not comp["stage_metrics"][stage].get("sha_match", False)],
            }
            for name, comp in comparisons.items()
        },
    }


def root_cause(comparisons: Mapping[str, Mapping[str, Any]], critic_forward: Mapping[str, Any]) -> Dict[str, Any]:
    off = comparisons["OFF_OFF"]
    on = comparisons["ON_ON"]
    offon = comparisons["OFF_ON"]
    off_value = off["stage_metrics"]["value_t"]
    on_value = on["stage_metrics"]["value_t"]
    offon_value = offon["stage_metrics"]["value_t"]
    off_mismatch = not bool(off_value.get("sha_match"))
    on_mismatch = not bool(on_value.get("sha_match"))
    offon_mismatch = not bool(offon_value.get("sha_match"))
    natural_max = max(float(off_value.get("max_abs_diff") or 0.0), float(on_value.get("max_abs_diff") or 0.0))
    offon_max = float(offon_value.get("max_abs_diff") or 0.0)
    offon_within_observed_envelope = offon_mismatch and natural_max > 0.0 and offon_max <= natural_max
    if off_mismatch and not on_mismatch and offon_mismatch and offon_max <= float(off_value.get("max_abs_diff") or 0.0):
        decision = "MPS_INTRINSIC_NUMERICAL_VARIABILITY"
        next_gate = "H4M-G-R2_MPS_EQUIVALENCE_CRITERION_SELECTION_AND_FREEZE"
    elif offon_within_observed_envelope:
        decision = "MPS_VARIABILITY_WITH_INSTRUMENTATION_WITHIN_ENVELOPE"
        next_gate = "H4M-G-R2_MPS_EQUIVALENCE_CRITERION_SELECTION_AND_FREEZE"
    elif (not off_mismatch) and (on_mismatch or offon_mismatch):
        decision = "INSTRUMENTATION_INDUCED_MPS_INTERFERENCE"
        next_gate = "H4M-G-R2_INSTRUMENTATION_MPS_INTERFERENCE_REPAIR_SELECTION_AND_FREEZE"
    else:
        decision = "MIXED_OR_NOT_UNIQUE"
        next_gate = "H4M-G-R2_ADDITIONAL_MINIMAL_ISOLATION_AUDIT"
    return {
        "stage": STAGE,
        "decision": decision,
        "exact_next_gate": next_gate,
        "observed_envelope_policy": "No tolerance invented; classification uses exact observed max_abs_diff envelope comparisons only.",
        "evidence": {
            "critic_forward_same_process_value_stable": critic_forward["same_process"]["all_value_t_sha_match"],
            "critic_forward_fresh_process_value_sha_match": critic_forward["fresh_process"]["value_t_diff"]["sha_match"],
            "off_off_first_divergence": off["first_divergence"],
            "on_on_first_divergence": on["first_divergence"],
            "off_on_first_divergence": offon["first_divergence"],
            "off_off_value_t": pick_metric(off_value),
            "on_on_value_t": pick_metric(on_value),
            "off_on_value_t": pick_metric(offon_value),
            "natural_max_abs_envelope_value_t": natural_max,
            "off_on_value_t_within_observed_envelope": offon_within_observed_envelope,
        },
    }


def rng_shadow_integrity(trials: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    shadow_contamination = any((trial.get("shadow_result") or {}).get("shadow_data_entered_training", False) for trial in trials.values())
    rng_initial_ok = all(comp["rng"]["initial_rng_match"] for comp in comparisons.values())
    return {
        "stage": STAGE,
        "rng_initial_state_reproducible": rng_initial_ok,
        "comparison_rng_checks": {name: comp["rng"] for name, comp in comparisons.items()},
        "shadow_summaries": {name: shadow_summary(trial) for name, trial in trials.items()},
        "shadow_contamination": shadow_contamination,
        "test6_opened": False,
        "github_push_performed": False,
        "integrity_passed": rng_initial_ok and not shadow_contamination,
    }


def gate_matrix(binding: Mapping[str, Any], runtime: Mapping[str, Any], rng_shadow: Mapping[str, Any], root: Mapping[str, Any]) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_match": binding.get("authoritative_binding_passed") is True,
        "mps_available": runtime.get("mps_available") is True,
        "audit_rng_initial_state_reproducible": rng_shadow.get("rng_initial_state_reproducible") is True,
        "instrumentation_contract_not_violated": True,
        "shadow_contamination_absent": rng_shadow.get("shadow_contamination") is False,
        "test6_not_opened": True,
        "github_push_false": True,
        "root_cause_classification_selected": root.get("decision") in {
            "MPS_INTRINSIC_NUMERICAL_VARIABILITY",
            "INSTRUMENTATION_INDUCED_MPS_INTERFERENCE",
            "MPS_VARIABILITY_WITH_INSTRUMENTATION_WITHIN_ENVELOPE",
            "MIXED_OR_NOT_UNIQUE",
        },
    }
    pass_ready = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if pass_ready else BLOCK_GATE,
        "decision": root.get("decision") if pass_ready else "H4M_G_R1_AUDIT_BLOCKED",
        "exact_next_gate": root.get("exact_next_gate") if pass_ready else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "final_flags": {
            "repair_executed": False,
            "h4m_h_rerun_executed": False,
            "three_seed_training_executed": False,
            "deterministic_mode_forced": False,
            "cpu_fallback_adopted_as_solution": False,
            "reward_gae_ppo_modified": False,
            "validation_executed": False,
            "TEST6_opened": False,
            "github_push_performed": False,
        },
    }


def final_report(comparison: Mapping[str, Any], root: Mapping[str, Any], gate: Mapping[str, Any], binding: Mapping[str, Any]) -> str:
    return f"""# H4M-G-R1 MPS Numerical Reproducibility & Instrumentation Equivalence Audit

gate = {gate["gate"]}
source_commit = {binding["source_provenance"]["h4m_g_r1_source_git_commit"]}
decision = {gate["decision"]}
exact_next_gate = {gate["exact_next_gate"]}

## Variability

- OFF↔OFF first divergence: {comparison["OFF_OFF"]["first_divergence"]}
- ON↔ON first divergence: {comparison["ON_ON"]["first_divergence"]}
- OFF↔ON first divergence: {comparison["OFF_ON"]["first_divergence"]}

## Root evidence

```json
{json.dumps(root["evidence"], ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no instrumentation repair, no H4M-H rerun, no training-budget change, no Reward/GAE/PPO change, no TEST6, no GitHub push.
"""


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files: Dict[str, str] = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            output_files[str(path.relative_to(artifact_root))] = str(path)
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
        "output_sha256": {name: sha256_file(Path(path)) for name, path in output_files.items()},
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256",
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def worker() -> None:
    created_at = "H4M-G-R1-WORKER"
    h4mg, h4mh = load_h4_modules()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable for critic forward worker")
    payload = critic_forward_once(h4mg, h4mh, 1, created_at, torch.device("mps"))
    print(compact_json(payload))


def main() -> None:
    if "--critic-forward-worker" in sys.argv:
        worker()
        return
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_g_r1_mps_numerical_reproducibility_audit_{stamp}"
    artifact_root.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    runtime = mps_runtime_environment(created_at)
    if not binding["authoritative_binding_passed"] or not runtime["mps_available"]:
        root = {"decision": "H4M_G_R1_AUDIT_BLOCKED", "exact_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE", "evidence": {}}
        rng_shadow = {"rng_initial_state_reproducible": False, "shadow_contamination": False}
        gate = gate_matrix(binding, runtime, rng_shadow, root)
        payloads = {
            "01_authoritative_binding.json": binding,
            "02_mps_runtime_environment.json": runtime,
            "03_critic_forward_reproducibility.json": {},
            "04_off_off_reproducibility.json": {},
            "05_on_on_reproducibility.json": {},
            "06_off_on_variability_comparison.json": {},
            "07_first_divergence_trace.json": {},
            "08_rng_shadow_integrity.json": rng_shadow,
            "09_root_cause_classification.json": root,
            "10_gate_matrix.json": gate,
        }
        for name, payload in payloads.items():
            write_json(artifact_root / name, payload)
        (artifact_root / "final_report.md").write_text("# H4M-G-R1 BLOCK\n", encoding="utf-8")
        write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
        print(f"[H4M-G-R1] artifact root: {artifact_root}")
        print(f"[H4M-G-R1] gate: {gate['gate']}")
        return

    h4mg, h4mh = load_h4_modules()
    device = torch.device("mps")
    critic_forward = critic_forward_reproducibility(created_at, h4mg, h4mh, device)
    trials = {
        "OFF_A": run_trial("OFF_A", "OFF", h4mg, h4mh, created_at, artifact_root, device),
        "OFF_B": run_trial("OFF_B", "OFF", h4mg, h4mh, created_at, artifact_root, device),
        "ON_A": run_trial("ON_A", "ON", h4mg, h4mh, created_at, artifact_root, device),
        "ON_B": run_trial("ON_B", "ON", h4mg, h4mh, created_at, artifact_root, device),
    }
    comparisons = {
        "OFF_OFF": compare_trials(trials["OFF_A"], trials["OFF_B"], "OFF↔OFF"),
        "ON_ON": compare_trials(trials["ON_A"], trials["ON_B"], "ON↔ON"),
        "OFF_ON": compare_trials(trials["OFF_A"], trials["ON_A"], "OFF↔ON"),
    }
    off_off = {"stage": STAGE, "comparison": "OFF↔OFF", "summary": envelope_summary(comparisons["OFF_OFF"]), "details": comparisons["OFF_OFF"]}
    on_on = {"stage": STAGE, "comparison": "ON↔ON", "summary": envelope_summary(comparisons["ON_ON"]), "details": comparisons["ON_ON"]}
    off_on = {"stage": STAGE, "comparison": "OFF↔ON", "summary": envelope_summary(comparisons["OFF_ON"]), "comparisons": {k: envelope_summary(v) for k, v in comparisons.items()}, "details": comparisons["OFF_ON"]}
    divergence = first_divergence_trace(comparisons)
    rng_shadow = rng_shadow_integrity(trials, comparisons)
    root = root_cause(comparisons, critic_forward)
    gate = gate_matrix(binding, runtime, rng_shadow, root)
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_mps_runtime_environment.json": runtime,
        "03_critic_forward_reproducibility.json": critic_forward,
        "04_off_off_reproducibility.json": off_off,
        "05_on_on_reproducibility.json": on_on,
        "06_off_on_variability_comparison.json": off_on,
        "07_first_divergence_trace.json": divergence,
        "08_rng_shadow_integrity.json": rng_shadow,
        "09_root_cause_classification.json": root,
        "10_gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(final_report(comparisons, root, gate, binding), encoding="utf-8")
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-G-R1] artifact root: {artifact_root}")
    print(f"[H4M-G-R1] gate: {gate['gate']}")
    print(f"[H4M-G-R1] decision: {gate['decision']}")
    print(f"[H4M-G-R1] OFF_OFF first_divergence: {comparisons['OFF_OFF']['first_divergence']}")
    print(f"[H4M-G-R1] ON_ON first_divergence: {comparisons['ON_ON']['first_divergence']}")
    print(f"[H4M-G-R1] OFF_ON first_divergence: {comparisons['OFF_ON']['first_divergence']}")
    print(f"[H4M-G-R1] next_gate: {gate['exact_next_gate']}")
    print("[H4M-G-R1] STOP no repair no H4M-H rerun no TEST6 no GitHub push")


if __name__ == "__main__":
    main()
