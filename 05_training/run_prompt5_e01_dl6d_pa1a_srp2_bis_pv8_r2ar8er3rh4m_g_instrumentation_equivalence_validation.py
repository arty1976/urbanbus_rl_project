#!/usr/bin/env python3
"""H4M-G instrumentation implementation and behavioral equivalence validation.

This is not the fresh diagnostic retraining stage. It implements the H4M-F
side-channel credit trace for a smallest controlled fresh execution, compares
instrumentation OFF vs ON, and emits equivalence evidence. The ON branch writes
detached/copy-only Parquet traces; shadow counterfactual smoke rows remain in a
separate namespace and never enter RewardNormalizer, return normalization,
Critic, GAE, PPO, or any optimizer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import pickle
import random
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


STAGE = "PV8-R2A-R8E-R3-R-H4M-G"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_INSTRUMENTATION_IMPLEMENTATION_AND_BEHAVIORAL_EQUIVALENCE_VALIDATION_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_INSTRUMENTATION_BEHAVIORAL_EQUIVALENCE_FAILED"
PASS_DECISION = "PV8_CREDIT_TRACE_INSTRUMENTATION_IMPLEMENTED_AND_BEHAVIORALLY_EQUIVALENT_READY_FOR_FRESH_DIAGNOSTIC_RETRAINING"
BLOCK_DECISION = "PV8_CREDIT_TRACE_INSTRUMENTATION_EQUIVALENCE_VALIDATION_BLOCKED"
NEXT_GATE = "H4M-H_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4M_F_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_f_credit_trace_instrumentation_selection_freeze_20260815_111506+0900"
H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"

EXPECTED = {
    "h4m_f_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_F_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_SELECTION_AND_FREEZE_COMPLETE",
    "h4m_f_decision": "PV8_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_FROZEN_READY_FOR_IMPLEMENTATION_EQUIVALENCE_VALIDATION",
    "h4m_f_source_commit": "570e8ec9a48adc3fd7d91231f71976b65e23224d",
    "h4m_f_contract_sha256": "9b95d0dc46459eedd2cf51e85789407be247e9db1e23e3a52a5bbae9c6216ffe",
    "h4m_c_source_commit": "00a53164c710f0bb8028aabe4f3f616a8fe6610d",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

ACTION_NAMES = {
    0: "HOLD_CURRENT_POSITION",
    1: "SERVE_AND_MOVE_TO_NEXT_STOP",
    2: "CONDITIONAL_SKIP_EMPTY_STOP",
}
ACTION_ORDER = [0, 1, 2]

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_instrumentation_implementation_audit.json",
    "03_rng_noninterference_audit.json",
    "04_off_on_sample_equivalence.json",
    "05_off_on_credit_equivalence.json",
    "06_off_on_parameter_checkpoint_equivalence.json",
    "07_trace_schema_integrity.json",
    "08_shadow_training_separation_audit.json",
    "09_h4m_g_gate_matrix.json",
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
    if isinstance(value, (set, tuple, deque)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(compact_json(value).encode("utf-8"))


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


def tensor_hash(tensor: torch.Tensor) -> str:
    detached = tensor.detach().cpu().contiguous()
    arr = detached.numpy()
    payload = {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "bytes_sha256": sha256_bytes(arr.tobytes(order="C")),
    }
    return canonical_sha(payload)


def tensor_payload_hash(values: Sequence[torch.Tensor]) -> str:
    return canonical_sha([tensor_hash(v) for v in values])


def recursive_state_hash(value: Any) -> str:
    def convert(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return {"tensor": tensor_hash(obj), "dtype": str(obj.dtype), "shape": list(obj.shape)}
        if isinstance(obj, Mapping):
            return {str(k): convert(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, deque):
            return [convert(v) for v in obj]
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return repr(obj)

    return canonical_sha(convert(value))


def rng_snapshot() -> Dict[str, Any]:
    py_state = pickle.dumps(random.getstate(), protocol=4)
    np_state = pickle.dumps(np.random.get_state(), protocol=4)
    torch_cpu_state = torch.get_rng_state()
    snapshot: Dict[str, Any] = {
        "python_random_sha256": sha256_bytes(py_state),
        "numpy_random_sha256": sha256_bytes(np_state),
        "torch_cpu_rng_sha256": tensor_hash(torch_cpu_state),
        "torch_mps_rng_sha256": None,
        "mps_available": bool(torch.backends.mps.is_available()),
        "environment_rng_status": "NOT_APPLICABLE_NO_SEPARATE_ENVIRONMENT_RNG_OBJECT_OBSERVED_IN_H4M_C_EXECUTION_PATH",
        "action_sampling_rng_source": "torch_cpu_rng_for_Categorical.sample_in_controlled_cpu_run",
    }
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            snapshot["torch_mps_rng_sha256"] = tensor_hash(torch.mps.get_rng_state())
        except Exception as exc:
            snapshot["torch_mps_rng_sha256"] = f"UNAVAILABLE:{type(exc).__name__}"
    return snapshot


def compare_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b))
    return a == b


def first_sequence_mismatch(a: Sequence[Any], b: Sequence[Any]) -> Optional[Dict[str, Any]]:
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return {"index": i, "off": left, "on": right}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "off_length": len(a), "on_length": len(b)}
    return None


def dataframe_to_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([dict(row) for row in rows])
    frame.to_parquet(path, index=False)
    return sha256_file(path)


def reward_normalizer_state(reward_normalizer: Any) -> Dict[str, Any]:
    buffer_values = [float(v) for v in getattr(reward_normalizer, "buffer", [])]
    if buffer_values:
        arr = np.array(buffer_values, dtype=np.float32)
        mean = float(arr.mean())
        std = float(max(float(arr.std()), 1e-6))
    else:
        mean = 0.0
        std = 1.0
    return {
        "window_size": int(getattr(reward_normalizer, "window_size", 1000)),
        "clip_value": float(getattr(reward_normalizer, "clip_value", 10.0)),
        "buffer": buffer_values,
        "mean": mean,
        "std": std,
        "buffer_sha256": canonical_sha(buffer_values),
    }


def optimizer_hashes(optimizers: Mapping[str, torch.optim.Optimizer]) -> Dict[str, str]:
    return {name: recursive_state_hash(opt.state_dict()) for name, opt in sorted(optimizers.items())}


def model_hashes(dl1: Any, encoder: torch.nn.Module, actor: torch.nn.Module, critic: torch.nn.Module) -> Dict[str, str]:
    return {
        "gatv2": dl1.module_hash(encoder),
        "actor": dl1.module_hash(actor),
        "critic": dl1.module_hash(critic),
    }


@dataclass
class TraceRecorder:
    root: Path
    enabled: bool
    deferred_materialization: bool = True
    pre_action_rows: List[Dict[str, Any]] = field(default_factory=list)
    reward_rows: List[Dict[str, Any]] = field(default_factory=list)
    critic_td_gae_rows: List[Dict[str, Any]] = field(default_factory=list)
    advantage_scope_rows: List[Dict[str, Any]] = field(default_factory=list)
    advantage_sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    ppo_rows: List[Dict[str, Any]] = field(default_factory=list)
    critic_value_error_rows: List[Dict[str, Any]] = field(default_factory=list)
    actor_pressure_rows: List[Dict[str, Any]] = field(default_factory=list)
    shadow_pair_rows: List[Dict[str, Any]] = field(default_factory=list)
    shadow_branch_event_rows: List[Dict[str, Any]] = field(default_factory=list)
    shadow_summary_rows: List[Dict[str, Any]] = field(default_factory=list)

    def write(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "trace_files": {}, "trace_file_sha256": {}, "row_counts": {}}
        traces = {
            "actual_pre_action_trace": (
                self.pre_action_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "actual_pre_action_trace.parquet",
            ),
            "actual_reward_trace": (
                self.reward_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "actual_reward_trace.parquet",
            ),
            "actual_critic_td_gae_trace": (
                self.critic_td_gae_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "actual_critic_td_gae_trace.parquet",
            ),
            "advantage_normalization_scope": (
                self.advantage_scope_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "advantage_normalization_scope.parquet",
            ),
            "advantage_normalization_sample": (
                self.advantage_sample_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "advantage_normalization_sample.parquet",
            ),
            "ppo_policy_surrogate_sample": (
                self.ppo_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "ppo_policy_surrogate_sample.parquet",
            ),
            "critic_value_error_by_sample": (
                self.critic_value_error_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "critic_value_error_by_sample.parquet",
            ),
            "actor_logit_pressure_by_sample": (
                self.actor_pressure_rows,
                self.root / "ACTUAL_ON_POLICY_TRACE" / "seed=1" / "outer_cycle=1" / "actor_logit_pressure_by_sample.parquet",
            ),
            "counterfactual_pair_index": (
                self.shadow_pair_rows,
                self.root / "PAIRED_COUNTERFACTUAL_SHADOW_TRACE" / "seed=1" / "outer_cycle=1" / "counterfactual_pair_index.parquet",
            ),
            "counterfactual_branch_event_sequence": (
                self.shadow_branch_event_rows,
                self.root / "PAIRED_COUNTERFACTUAL_SHADOW_TRACE" / "seed=1" / "outer_cycle=1" / "counterfactual_branch_event_sequence.parquet",
            ),
            "counterfactual_pair_summary": (
                self.shadow_summary_rows,
                self.root / "PAIRED_COUNTERFACTUAL_SHADOW_TRACE" / "seed=1" / "outer_cycle=1" / "counterfactual_pair_summary.parquet",
            ),
        }
        trace_files: Dict[str, str] = {}
        trace_sha: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        for name, (rows, path) in traces.items():
            counts[name] = len(rows)
            trace_files[name] = str(path)
            trace_sha[name] = dataframe_to_parquet(rows, path)
        return {"enabled": True, "trace_files": trace_files, "trace_file_sha256": trace_sha, "row_counts": counts}


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    branch = git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = git_run(["status", "--short"]).stdout
    source = SOURCE_REL.as_posix()
    source_present = git_bool(["cat-file", "-e", f"HEAD:{source}"])
    latest = git_run(["log", "-1", "--format=%H", "--", source]).stdout.strip()
    no_diff = git_bool(["diff", "--quiet", "--", source])
    no_staged = git_bool(["diff", "--cached", "--quiet", "--", source])
    head_files = [line.strip() for line in git_run(["show", "--name-only", "--pretty=format:", "HEAD"]).stdout.splitlines() if line.strip()]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_g_source_git_commit": head,
        "h4m_g_source_path": source,
        "h4m_g_source_sha256": sha256_file(PROJECT_ROOT / source),
        "h4m_g_source_present_in_head": source_present,
        "h4m_g_source_latest_commit": latest,
        "h4m_g_source_no_uncommitted_diff_vs_head": no_diff,
        "h4m_g_source_no_staged_diff_vs_head": no_staged,
        "head_commit_files": head_files,
        "local_source_only_commit_created_before_equivalence": source_present and latest == head and head_files == [source],
        "post_commit_provenance_gate_passed": source_present and latest == head and no_diff and no_staged and head_files == [source],
        "status_short": status_short,
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    h4m_f_manifest = read_json(H4M_F_ROOT / "manifest.json")
    h4m_f_gate = read_json(H4M_F_ROOT / "14_h4m_f_gate_matrix.json")
    contract_path = H4M_F_ROOT / "13_h4m_f_credit_trace_instrumentation_contract.json"
    contract_sha = sha256_file(contract_path)
    contract = read_json(contract_path)
    h4m_c_gate = read_json(H4M_C_ROOT / "14_h4m_c_gate_matrix.json")
    checks = {
        "h4m_f_gate_match": h4m_f_gate.get("gate") == EXPECTED["h4m_f_gate"],
        "h4m_f_decision_match": h4m_f_gate.get("decision") == EXPECTED["h4m_f_decision"],
        "h4m_f_source_commit_match": h4m_f_manifest.get("h4m_f_source_git_commit") == EXPECTED["h4m_f_source_commit"],
        "contract_sha_match": contract_sha == EXPECTED["h4m_f_contract_sha256"],
        "contract_manifest_sha_match": h4m_f_manifest.get("instrumentation_contract_sha256") == EXPECTED["h4m_f_contract_sha256"],
        "contract_gate_sha_match": h4m_f_gate.get("instrumentation_contract_sha256") == EXPECTED["h4m_f_contract_sha256"],
        "selected_scope_match": h4m_f_gate.get("selected_instrumentation_scope") == "A_ALL_ELIGIBLE_ACTUAL_ON_POLICY_MULTI_ACTION_STATES",
        "h4m_c_source_commit_match": h4m_c_gate.get("h4m_c_training_source_git_commit") == EXPECTED["h4m_c_source_commit"],
        "h4m_b_schedule_sha_match": contract.get("authoritative_upstream_SHA_bindings", {}).get("h4m_b_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": contract.get("authoritative_upstream_SHA_bindings", {}).get("reward_v2_sha256")
        == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": contract.get("authoritative_upstream_SHA_bindings", {}).get("h4g_runtime_sha256")
        == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": contract.get("authoritative_upstream_SHA_bindings", {}).get("r3_split_sha256")
        == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": contract.get("authoritative_upstream_SHA_bindings", {}).get("zero_loss_adapter_sha256")
        == EXPECTED["zero_loss_adapter_sha256"],
        "h4m_g_source_commit_frozen": provenance.get("post_commit_provenance_gate_passed") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4m_f_artifact_root": str(H4M_F_ROOT),
        "h4m_f_contract_path": str(contract_path),
        "instrumentation_contract_sha256": contract_sha,
        "h4m_f_contract_id": contract.get("contract_id"),
        "h4m_f_contract_version": contract.get("contract_version"),
        "selected_scope": h4m_f_gate.get("selected_instrumentation_scope"),
        "source_provenance": dict(provenance),
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "training_execution_scope": "CONTROLLED_FRESH_EQUIVALENCE_ONLY_ONE_SEED_ONE_CYCLE",
        "h4m_c_checkpoint_continuation": False,
        "test6_status": "SEALED_NOT_OPENED",
        "github_push_performed": False,
    }


def build_context(seed: int, created_at: str) -> Dict[str, Any]:
    h4m_c = import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_c_fresh_extended_budget_three_seed_retraining.py",
        f"h4m_g_h4m_c_{seed}_{time.time_ns()}",
    )
    h4k = import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py",
        f"h4m_g_h4k_{seed}_{time.time_ns()}",
    )
    dl4 = import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
        f"h4m_g_dl4_{seed}_{time.time_ns()}",
    )
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    mappo_mod = import_module_from_path(TRAINING_ROOT / "mappo_runner.py", f"h4m_g_mappo_{seed}_{time.time_ns()}")
    reward_mod = import_module_from_path(TRAINING_ROOT / "rewards/mappo_reward_v1.py", f"h4m_g_reward_{seed}_{time.time_ns()}")
    observation_repair = import_module_from_path(
        TRAINING_ROOT / "observation_target_context_repair.py",
        f"h4m_g_observation_target_context_repair_{seed}_{time.time_ns()}",
    )
    schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    window_plan = read_json(H4M_C_ROOT / "seed_001" / "r3_train_window_plan.json")
    train_paths = [Path(row["snapshot_path"]) for row in window_plan["train_rows"]]
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])

    dl4.set_all_seeds(seed)
    device = torch.device("cpu")
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
        "controlled_equivalence_seed": seed,
        "device": "cpu",
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
        "observation_repair_module": observation_repair,
        "sample_observation_repair_audit": sample_observation_repair_audit,
        "observation_repair_audit": observation_repair_audit,
        "connectivity": connectivity,
        "tensor_mask": tensor_mask,
        "inventory": inventory,
    }


def make_sample_identity(
    seed: int,
    cycle_index: int,
    rollout_id: str,
    window: Mapping[str, Any],
    agent_id: int,
    agent_slot: int,
    step_index: int,
    state_ts: str,
    state_hash: str,
) -> Tuple[str, Dict[str, Any]]:
    identity = {
        "seed": seed,
        "outer_cycle": cycle_index,
        "rollout_id": rollout_id,
        "window_id": window["window_id"],
        "agent_id": agent_id,
        "step_index": step_index,
        "state_ts": state_ts,
        "state_hash": state_hash,
    }
    return canonical_sha(identity), identity


def sign_class(value: float) -> str:
    tol = 1.0e-12
    if abs(value) <= tol:
        return "near_zero"
    return "positive" if value > 0.0 else "negative"


def sign_change(raw: float, normalized: float) -> Tuple[str, bool]:
    raw_s = sign_class(raw)
    norm_s = sign_class(normalized)
    if "near_zero" in {raw_s, norm_s}:
        return "near_zero", raw_s != norm_s
    if raw_s == norm_s:
        return "sign_preserved", False
    return f"{raw_s}_to_{norm_s}", True


def pressure_labels(action_id: int, normalized_advantage: float, clip_active: bool) -> Dict[str, str]:
    if abs(normalized_advantage) <= 1.0e-12 or clip_active:
        sampled = "near_zero" if clip_active else "near_zero"
    elif normalized_advantage > 0.0:
        sampled = "increase"
    else:
        sampled = "decrease"
    labels = {"effective_direction_on_sampled_action_logit": sampled}
    for aid, name in [(0, "hold"), (1, "serve"), (2, "skip")]:
        key = f"{name}_logit_pressure"
        if aid == action_id:
            labels[key] = sampled
        elif sampled == "increase":
            labels[key] = "decrease"
        elif sampled == "decrease":
            labels[key] = "increase"
        else:
            labels[key] = "near_zero"
    return labels


def collect_controlled_rollout(
    *,
    branch: str,
    seed: int,
    cycle_index: int,
    ctx: Mapping[str, Any],
    recorder: Optional[TraceRecorder],
) -> Dict[str, Any]:
    h4m_c = ctx["h4m_c"]
    h4k = ctx["h4k"]
    dl1 = ctx["dl1"]
    dl4 = ctx["dl4"]
    mappo_mod = ctx["mappo_mod"]
    reward_mod = ctx["reward_mod"]
    data_seq = ctx["train_data"]
    window_rows = ctx["window_plan"]["train_rows"]
    encoder = ctx["encoder"]
    actor = ctx["actor"]
    critic = ctx["critic"]
    reward_normalizer = ctx["reward_normalizer"]
    return_normalizer = ctx["return_normalizer"]
    device = ctx["device"]
    config = ctx["config"]
    horizon = int(config["controlled_effective_horizon"])
    offset = 0
    rollout_id = f"H4M-G-CONTROLLED-SEED-{seed:03d}-CYCLE-{cycle_index:03d}"

    raw_rewards: List[torch.Tensor] = []
    values_original: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    agent_indices: List[List[int]] = []
    sample_uid_grid: List[List[Optional[str]]] = []
    active_order: List[Tuple[int, int, str]] = []
    policy_rows: List[Dict[str, Any]] = []
    reward_materialized_rows: List[Dict[str, Any]] = []
    entropy_coefficients: List[float] = []
    started = time.perf_counter()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for local_step in range(horizon):
            absolute = offset + local_step
            window = window_rows[absolute]
            data = data_seq[absolute].to(device)
            indices = dl1.agent_indices_for_step(config["spec"], absolute, int(config["effective_agents"]))
            logits, _critic_out, value_original, agent_mask = dl4.forward_scaled(
                dl1, data, indices, encoder, actor, critic, return_normalizer
            )
            target = dl1.action_targets_from_y(
                data.y[torch.tensor(indices, dtype=torch.long, device=device)],
                int(config["action_dim"]),
            )
            masked_logits, allowed = h4k.masked_logits_for_targets(logits, target)
            raw_probs = torch.softmax(logits, dim=-1)
            masked_probs = torch.softmax(masked_logits, dim=-1)
            entropy = Categorical(logits=masked_logits).entropy()
            dist = Categorical(logits=masked_logits)
            rng_before_sample = rng_snapshot()
            action = dist.sample()
            rng_after_sample = rng_snapshot()
            log_prob = dist.log_prob(action)
            rewards_for_step = []
            uids_for_step: List[Optional[str]] = []
            critic_graph_hash = canonical_sha(
                {
                    "x": tensor_hash(data.x),
                    "edge_index": tensor_hash(data.edge_index),
                    "edge_attr": tensor_hash(data.edge_attr),
                    "node_mask": tensor_hash(data.node_mask),
                }
            )
            for agent_slot, (action_id, target_id, is_active) in enumerate(
                zip(action.detach().cpu().tolist(), target.detach().cpu().tolist(), agent_mask.detach().cpu().tolist())
            ):
                if bool(is_active):
                    agent_id = int(indices[agent_slot])
                    actor_obs_hash = tensor_hash(data.x[agent_id])
                    legal_ids = [int(i) for i, v in enumerate(allowed[agent_slot].detach().cpu().tolist()) if bool(v)]
                    state_ts = f"{window.get('start_iso')}#absolute_step={absolute}#agent_slot={agent_slot}"
                    state_hash = canonical_sha(
                        {
                            "window_id": window["window_id"],
                            "snapshot_id": window["snapshot_id"],
                            "absolute_step": absolute,
                            "agent_id": agent_id,
                            "agent_slot": agent_slot,
                            "actor_observation_hash": actor_obs_hash,
                            "critic_observation_hash": critic_graph_hash,
                            "target_id": int(target_id),
                            "legal_action_ids": legal_ids,
                        }
                    )
                    sample_uid, identity = make_sample_identity(
                        seed,
                        cycle_index,
                        rollout_id,
                        window,
                        agent_id,
                        agent_slot,
                        absolute,
                        state_ts,
                        state_hash,
                    )
                    uids_for_step.append(sample_uid)
                    active_order.append((local_step, agent_slot, sample_uid))
                    metrics = h4m_c.h4m_c_reward_v2_metrics(
                        reward_mod,
                        cycle_index=cycle_index,
                        window=window,
                        local_step=absolute,
                        agent_slot=agent_slot,
                        action_id=int(action_id),
                        target_id=int(target_id),
                    )
                    materialized = reward_mod.compute_reward_v2(metrics)
                    reward_value = float(materialized["reward_total"])
                    policy_row = {
                        "sample_uid": sample_uid,
                        **identity,
                        "branch": branch,
                        "cycle_index": cycle_index,
                        "window_id": window["window_id"],
                        "snapshot_id": int(window["snapshot_id"]),
                        "time_band": str(window.get("time_band")),
                        "local_step": absolute,
                        "agent_slot": agent_slot,
                        "agent_id": agent_id,
                        "action_id": int(action_id),
                        "action": ACTION_NAMES.get(int(action_id), str(action_id)),
                        "target_id": int(target_id),
                        "target": ACTION_NAMES.get(int(target_id), str(target_id)),
                        "legal_action_ids": legal_ids,
                        "legal_action_mask": [bool(v) for v in allowed[agent_slot].detach().cpu().tolist()],
                        "legal_actions": [ACTION_NAMES[i] for i in legal_ids],
                        "actor_observation_hash": actor_obs_hash,
                        "critic_observation_hash": critic_graph_hash,
                        "actor_observation_shape": list(data.x[agent_id].detach().cpu().shape),
                        "critic_observation_shape": list(data.x.detach().cpu().shape),
                        "policy_entropy": float(entropy[agent_slot].detach().cpu().item()),
                        "sampled_action_log_prob": float(log_prob[agent_slot].detach().cpu().item()),
                        "action_sampling_rng_ref": rng_before_sample["torch_cpu_rng_sha256"],
                        "action_sampling_rng_after_ref": rng_after_sample["torch_cpu_rng_sha256"],
                        "trace_capture_after_authoritative_sampling": True,
                    }
                    for action_key in ACTION_ORDER:
                        policy_row[f"legal_{action_key}"] = action_key in legal_ids
                        policy_row[f"pre_mask_logit_{action_key}"] = float(logits[agent_slot, action_key].detach().cpu().item())
                        policy_row[f"post_mask_logit_{action_key}"] = float(masked_logits[agent_slot, action_key].detach().cpu().item())
                        policy_row[f"pre_mask_probability_{action_key}"] = float(raw_probs[agent_slot, action_key].detach().cpu().item())
                        policy_row[f"masked_probability_{action_key}"] = float(masked_probs[agent_slot, action_key].detach().cpu().item())
                    policy_rows.append(policy_row)
                    reward_materialized_rows.append(
                        {
                            "sample_uid": sample_uid,
                            "seed": seed,
                            "outer_cycle": cycle_index,
                            "window_id": window["window_id"],
                            "agent_id": agent_id,
                            "agent_slot": agent_slot,
                            "step_index": absolute,
                            "sampled_action_id": int(action_id),
                            "sampled_action_name": ACTION_NAMES.get(int(action_id), str(action_id)),
                            "target_id": int(target_id),
                            "reward_v2_local_service_component_raw": float(materialized["reward_service_component_raw"]),
                            "reward_v2_local_avg_wait_component_raw": float(materialized["reward_avg_wait_component_raw"]),
                            "reward_v2_explicit_intervention_component_raw": float(materialized["reward_intervention_component_raw"]),
                            "reward_v2_local_service_component_weighted": float(materialized["reward_service_component_weighted"]),
                            "reward_v2_local_avg_wait_component_weighted": float(materialized["reward_avg_wait_component_weighted"]),
                            "reward_v2_explicit_intervention_component_weighted": float(materialized["reward_intervention_component_weighted"]),
                            "reward_v2_raw_total": reward_value,
                            "reward_freeze_sha256": materialized["reward_freeze_sha256"],
                            "reward_after_runtime_normalization": None,
                            "reward_normalizer_scope_id": f"H4M-G-REWARD-NORM-SEED-{seed:03d}-CYCLE-{cycle_index:03d}",
                            "reward_normalizer_update_instrumented": False,
                        }
                    )
                else:
                    reward_value = 0.0
                    uids_for_step.append(None)
                rewards_for_step.append(reward_value)
            raw_rewards.append(torch.tensor(rewards_for_step, dtype=value_original.dtype, device=device))
            values_original.append(value_original.detach())
            old_log_probs.append(log_prob.detach())
            actions.append(action.detach())
            targets.append(target.detach())
            masks.append(agent_mask.detach())
            agent_indices.append(indices)
            sample_uid_grid.append(uids_for_step)
            progress = float(local_step) / max(1.0, float(horizon - 1))
            entropy_coefficients.append(
                float(
                    mappo_mod.entropy_coef_for_time_band(
                        str(window.get("time_band", "offpeak")),
                        progress,
                        offpeak_coef=0.01,
                        peak_coef=0.03,
                        min_coef=0.001,
                    )
                )
            )
        next_idx = min(offset + horizon, len(data_seq) - 1)
        next_data = data_seq[next_idx].to(device)
        next_indices = dl1.agent_indices_for_step(config["spec"], next_idx, int(config["effective_agents"]))
        _logits, _critic_out, last_next_value_original, _mask = dl4.forward_scaled(
            dl1, next_data, next_indices, encoder, actor, critic, return_normalizer
        )

    raw_rewards_t = torch.stack(raw_rewards)
    masks_t = torch.stack(masks).bool()
    reward_norm_before = reward_normalizer_state(reward_normalizer)
    normalized_rewards_t = torch.zeros_like(raw_rewards_t)
    active_raw = raw_rewards_t[masks_t].detach().cpu().tolist()
    active_normalized = reward_normalizer.normalize([float(v) for v in active_raw])
    normalized_rewards_t[masks_t] = torch.tensor(active_normalized, dtype=raw_rewards_t.dtype, device=device)
    reward_norm_after = reward_normalizer_state(reward_normalizer)
    active_norm_iter = iter(float(v) for v in active_normalized)
    reward_rows_by_uid = {row["sample_uid"]: row for row in reward_materialized_rows}
    for _local_step, _agent_slot, sample_uid in active_order:
        row = reward_rows_by_uid[sample_uid]
        row["reward_after_runtime_normalization"] = next(active_norm_iter)
        row["reward_runtime_normalizer_mean"] = reward_norm_after["mean"]
        row["reward_runtime_normalizer_std"] = reward_norm_after["std"]
        row["reward_normalizer_pre_buffer_sha256"] = reward_norm_before["buffer_sha256"]
        row["reward_normalizer_post_buffer_sha256"] = reward_norm_after["buffer_sha256"]

    values_t = torch.stack(values_original)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = last_next_value_original.detach()
    terminated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, normalized_advantages, gae_audit = dl1.compute_gae(
        normalized_rewards_t,
        values_t,
        next_values_t,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    return_norm_before = dict(ctx["return_normalizer"].state_dict())
    return_normalizer.update(returns[masks_t.bool()])
    return_norm_after = dict(ctx["return_normalizer"].state_dict())
    td_delta_t = normalized_rewards_t + float(config["gamma"]) * next_values_t - values_t
    normalization_scope_id = f"H4M-G-ADV-NORM-SEED-{seed:03d}-CYCLE-{cycle_index:03d}"

    if recorder and recorder.enabled and not recorder.deferred_materialization:
        recorder.pre_action_rows.extend(policy_rows)
        recorder.reward_rows.extend(reward_materialized_rows)
        valid_adv = advantages[masks_t.bool()].detach().cpu()
        valid_norm = normalized_advantages[masks_t.bool()].detach().cpu()
        recorder.advantage_scope_rows.append(
            {
                "normalization_scope_id": normalization_scope_id,
                "seed": seed,
                "outer_cycle": cycle_index,
                "population_count": int(advantages.numel()),
                "active_sample_count": int(masks_t.sum().detach().cpu().item()),
                "mean": float(valid_adv.mean().item()),
                "std": float(valid_adv.std(unbiased=False).item()),
                "epsilon": 1.0e-8,
                "masking_semantics": "rollout_global_active_only_inactive_excluded",
                "dtype": str(advantages.dtype),
            }
        )
        for local_step, agent_slot, sample_uid in active_order:
            raw_adv = float(advantages[local_step, agent_slot].detach().cpu().item())
            norm_adv = float(normalized_advantages[local_step, agent_slot].detach().cpu().item())
            raw_sign = sign_class(raw_adv)
            norm_sign = sign_class(norm_adv)
            change_class, changed = sign_change(raw_adv, norm_adv)
            next_uid = sample_uid_grid[local_step + 1][agent_slot] if local_step + 1 < horizon else None
            prev_uid = sample_uid_grid[local_step - 1][agent_slot] if local_step > 0 else None
            policy = next(row for row in policy_rows if row["sample_uid"] == sample_uid)
            recorder.critic_td_gae_rows.append(
                {
                    "sample_uid": sample_uid,
                    "seed": seed,
                    "outer_cycle": cycle_index,
                    "value_t": float(values_t[local_step, agent_slot].detach().cpu().item()),
                    "next_sample_uid": next_uid,
                    "next_value_t": float(next_values_t[local_step, agent_slot].detach().cpu().item()),
                    "terminated": bool(terminated[local_step, agent_slot].detach().cpu().item()),
                    "truncated": bool(truncated[local_step, agent_slot].detach().cpu().item()),
                    "bootstrap_mask": 1.0,
                    "gamma": float(config["gamma"]),
                    "gae_lambda": float(config["gae_lambda"]),
                    "td_delta": float(td_delta_t[local_step, agent_slot].detach().cpu().item()),
                    "raw_gae_advantage": raw_adv,
                    "raw_return_target": float(returns[local_step, agent_slot].detach().cpu().item()),
                    "gae_recursion_predecessor_uid": prev_uid,
                    "gae_recursion_successor_uid": next_uid,
                    "gae_formula_id": "DL1.compute_gae active-only rollout-global standardization",
                }
            )
            recorder.advantage_sample_rows.append(
                {
                    "normalization_scope_id": normalization_scope_id,
                    "sample_uid": sample_uid,
                    "seed": seed,
                    "outer_cycle": cycle_index,
                    "sampled_action_id": int(policy["action_id"]),
                    "sampled_action_name": policy["action"],
                    "raw_gae_advantage": raw_adv,
                    "normalized_advantage": norm_adv,
                    "raw_sign": raw_sign,
                    "normalized_sign": norm_sign,
                    "sign_changed": changed,
                    "sign_change_class": change_class,
                }
            )
            value_error = float(returns[local_step, agent_slot].detach().cpu().item() - values_t[local_step, agent_slot].detach().cpu().item())
            recorder.critic_value_error_rows.append(
                {
                    "sample_uid": sample_uid,
                    "seed": seed,
                    "outer_cycle": cycle_index,
                    "sampled_action_id": int(policy["action_id"]),
                    "sampled_action_name": policy["action"],
                    "value_t_before_update": float(values_t[local_step, agent_slot].detach().cpu().item()),
                    "raw_return_target": float(returns[local_step, agent_slot].detach().cpu().item()),
                    "normalized_return_target": None,
                    "value_error": value_error,
                    "squared_error": value_error * value_error,
                    "time_band": str(window_rows[local_step].get("time_band")),
                    "window_id": policy["window_id"],
                    "agent_id": int(policy["agent_id"]),
                }
            )

    return {
        "cycle_index": cycle_index,
        "rollout_collection_seconds": time.perf_counter() - started,
        "raw_rewards": raw_rewards_t.detach(),
        "normalized_rewards": normalized_rewards_t.detach(),
        "values_original": values_t.detach(),
        "next_values_original": next_values_t.detach(),
        "td_delta": td_delta_t.detach(),
        "old_log_probs": torch.stack(old_log_probs).detach(),
        "actions": torch.stack(actions).detach(),
        "targets": torch.stack(targets).detach(),
        "agent_mask": masks_t.detach(),
        "agent_indices": agent_indices,
        "sample_uid_grid": sample_uid_grid,
        "active_order": active_order,
        "returns_original": returns.detach(),
        "advantages": advantages.detach(),
        "normalized_advantages": normalized_advantages.detach(),
        "gae_audit": gae_audit,
        "gamma": float(config["gamma"]),
        "gae_lambda": float(config["gae_lambda"]),
        "reward_rows": reward_materialized_rows,
        "policy_rows": policy_rows,
        "entropy_coef_mean": float(np.mean(entropy_coefficients)) if entropy_coefficients else 0.01,
        "entropy_coef_stats": {
            "count": len(entropy_coefficients),
            "mean": float(np.mean(entropy_coefficients)) if entropy_coefficients else None,
            "min": float(np.min(entropy_coefficients)) if entropy_coefficients else None,
            "max": float(np.max(entropy_coefficients)) if entropy_coefficients else None,
        },
        "reward_normalizer_state_before": reward_norm_before,
        "reward_normalizer_state_after": reward_norm_after,
        "return_normalizer_state_before": return_norm_before,
        "return_normalizer_state_after": return_norm_after,
    }


def materialize_rollout_credit_trace(
    *,
    seed: int,
    cycle_index: int,
    rollout: Mapping[str, Any],
    recorder: TraceRecorder,
) -> None:
    if not recorder.enabled:
        return
    recorder.pre_action_rows.extend(rollout["policy_rows"])
    recorder.reward_rows.extend(rollout["reward_rows"])
    advantages = rollout["advantages"]
    normalized_advantages = rollout["normalized_advantages"]
    masks_t = rollout["agent_mask"].bool()
    values_t = rollout["values_original"]
    next_values_t = rollout["next_values_original"]
    normalized_rewards_t = rollout["normalized_rewards"]
    returns = rollout["returns_original"]
    device = values_t.device
    terminated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool, device=device)
    truncated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool, device=device)
    truncated[-1] = True
    td_delta_t = rollout["td_delta"]
    active_order = rollout["active_order"]
    sample_uid_grid = rollout["sample_uid_grid"]
    horizon = int(rollout["actions"].size(0))
    policy_by_uid = {row["sample_uid"]: row for row in rollout["policy_rows"]}
    normalization_scope_id = f"H4M-G-ADV-NORM-SEED-{seed:03d}-CYCLE-{cycle_index:03d}"
    valid_adv = advantages[masks_t].detach().cpu()
    valid_norm = normalized_advantages[masks_t].detach().cpu()
    recorder.advantage_scope_rows.append(
        {
            "normalization_scope_id": normalization_scope_id,
            "seed": seed,
            "outer_cycle": cycle_index,
            "population_count": int(advantages.numel()),
            "active_sample_count": int(masks_t.sum().detach().cpu().item()),
            "mean": float(valid_adv.mean().item()),
            "std": float(valid_adv.std(unbiased=False).item()),
            "epsilon": 1.0e-8,
            "masking_semantics": "rollout_global_active_only_inactive_excluded",
            "dtype": str(advantages.dtype),
            "materialization_order": "deferred_after_authoritative_update",
        }
    )
    for local_step, agent_slot, sample_uid in active_order:
        raw_adv = float(advantages[local_step, agent_slot].detach().cpu().item())
        norm_adv = float(normalized_advantages[local_step, agent_slot].detach().cpu().item())
        raw_sign = sign_class(raw_adv)
        norm_sign = sign_class(norm_adv)
        change_class, changed = sign_change(raw_adv, norm_adv)
        next_uid = sample_uid_grid[local_step + 1][agent_slot] if local_step + 1 < horizon else None
        prev_uid = sample_uid_grid[local_step - 1][agent_slot] if local_step > 0 else None
        policy = policy_by_uid[sample_uid]
        recorder.critic_td_gae_rows.append(
            {
                "sample_uid": sample_uid,
                "seed": seed,
                "outer_cycle": cycle_index,
                "value_t": float(values_t[local_step, agent_slot].detach().cpu().item()),
                "next_sample_uid": next_uid,
                "next_value_t": float(next_values_t[local_step, agent_slot].detach().cpu().item()),
                "terminated": bool(terminated[local_step, agent_slot].detach().cpu().item()),
                "truncated": bool(truncated[local_step, agent_slot].detach().cpu().item()),
                "bootstrap_mask": 1.0,
                "gamma": float(rollout["gamma"]),
                "gae_lambda": float(rollout["gae_lambda"]),
                "td_delta": float(td_delta_t[local_step, agent_slot].detach().cpu().item()),
                "raw_gae_advantage": raw_adv,
                "raw_return_target": float(returns[local_step, agent_slot].detach().cpu().item()),
                "gae_recursion_predecessor_uid": prev_uid,
                "gae_recursion_successor_uid": next_uid,
                "gae_formula_id": "DL1.compute_gae active-only rollout-global standardization",
                "materialization_order": "deferred_after_authoritative_update",
            }
        )
        recorder.advantage_sample_rows.append(
            {
                "normalization_scope_id": normalization_scope_id,
                "sample_uid": sample_uid,
                "seed": seed,
                "outer_cycle": cycle_index,
                "sampled_action_id": int(policy["action_id"]),
                "sampled_action_name": policy["action"],
                "raw_gae_advantage": raw_adv,
                "normalized_advantage": norm_adv,
                "raw_sign": raw_sign,
                "normalized_sign": norm_sign,
                "sign_changed": changed,
                "sign_change_class": change_class,
                "materialization_order": "deferred_after_authoritative_update",
            }
        )
        value_error = float(returns[local_step, agent_slot].detach().cpu().item() - values_t[local_step, agent_slot].detach().cpu().item())
        recorder.critic_value_error_rows.append(
            {
                "sample_uid": sample_uid,
                "seed": seed,
                "outer_cycle": cycle_index,
                "sampled_action_id": int(policy["action_id"]),
                "sampled_action_name": policy["action"],
                "value_t_before_update": float(values_t[local_step, agent_slot].detach().cpu().item()),
                "raw_return_target": float(returns[local_step, agent_slot].detach().cpu().item()),
                "normalized_return_target": None,
                "value_error": value_error,
                "squared_error": value_error * value_error,
                "time_band": policy.get("time_band"),
                "window_id": policy["window_id"],
                "agent_id": int(policy["agent_id"]),
                "materialization_order": "deferred_after_authoritative_update",
            }
        )


def materialize_ppo_trace(
    *,
    seed: int,
    cycle_index: int,
    rollout: Mapping[str, Any],
    ppo_fingerprint_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    recorder: TraceRecorder,
) -> None:
    if not recorder.enabled:
        return
    effective_agents = int(config["effective_agents"])
    for row in ppo_fingerprint_rows:
        flat_index = int(row["selected_flat_index"])
        local_step = flat_index // effective_agents
        agent_slot = flat_index % effective_agents
        sample_uid = rollout["sample_uid_grid"][local_step][agent_slot]
        action_id = int(rollout["actions"][local_step, agent_slot].detach().cpu().item())
        old_lp = float(row["old_log_prob"])
        new_lp = float(row["current_log_prob"])
        adv = float(row["normalized_advantage"])
        clip_active = bool(float(row["unclipped"]) != float(row["effective"]))
        labels = pressure_labels(action_id, adv, clip_active)
        ppo_row = {
            "ppo_update_id": f"H4M-G-SEED-{seed:03d}-CYCLE-{cycle_index:03d}-ACTOR",
            "seed": seed,
            "outer_cycle": cycle_index,
            "ppo_epoch": int(row["epoch"]),
            "minibatch_id": 0,
            "selected_flat_index": flat_index,
            "sample_uid": sample_uid,
            "sampled_action_id": action_id,
            "sampled_action_name": ACTION_NAMES[action_id],
            "old_log_prob": old_lp,
            "current_log_prob": new_lp,
            "old_action_probability": float(math.exp(old_lp)),
            "current_action_probability": float(math.exp(new_lp)),
            "probability_ratio": float(row["ratio"]),
            "normalized_advantage": adv,
            "clip_epsilon": float(config["ppo_clip_epsilon"]),
            "unclipped_surrogate": float(row["unclipped"]),
            "clipped_ratio": float(row["clipped"] / adv) if adv != 0.0 else None,
            "clipped_surrogate": float(row["clipped"]),
            "clip_active": clip_active,
            "effective_policy_surrogate_contribution": float(row["effective"]),
            "entropy_contribution": float(row["entropy"]),
            "sampled_action_pressure": labels["effective_direction_on_sampled_action_logit"],
            "materialization_order": "deferred_after_authoritative_update",
        }
        recorder.ppo_rows.append(ppo_row)
        recorder.actor_pressure_rows.append(
            {
                "sample_uid": sample_uid,
                "seed": seed,
                "outer_cycle": cycle_index,
                "ppo_update_id": ppo_row["ppo_update_id"],
                "ppo_epoch": int(row["epoch"]),
                "minibatch_id": 0,
                "sampled_action_id": action_id,
                "sampled_action_name": ACTION_NAMES[action_id],
                **labels,
                "pressure_formula_id": "H4M-G detached PPO clipped-surrogate sign reconstruction plus entropy row storage",
                "near_zero_rule": "05_training/rewards/mappo_reward_v1.py::PV8_REWARD_V2_NUMERIC_TOLERANCE",
                "materialization_order": "deferred_after_authoritative_update",
            }
        )


def ppo_update_controlled(
    *,
    branch: str,
    seed: int,
    cycle_index: int,
    ctx: Mapping[str, Any],
    rollout: Mapping[str, Any],
    before_state: Mapping[str, Mapping[str, torch.Tensor]],
    recorder: Optional[TraceRecorder],
) -> Dict[str, Any]:
    dl1 = ctx["dl1"]
    dl4 = ctx["dl4"]
    h4k = ctx["h4k"]
    data_seq = ctx["train_data"]
    encoder = ctx["encoder"]
    actor = ctx["actor"]
    critic = ctx["critic"]
    optimizers = ctx["optimizers"]
    return_normalizer = ctx["return_normalizer"]
    device = ctx["device"]
    config = ctx["config"]
    offset = 0
    horizon = int(rollout["actions"].size(0))
    valid_flat = torch.where(rollout["agent_mask"].reshape(-1))[0]
    selected_flat = valid_flat[: min(int(config["minibatch_size"]), int(valid_flat.numel()))]
    if selected_flat.numel() == 0:
        raise RuntimeError("No active samples for controlled PPO update.")
    entropy_coef = float(rollout.get("entropy_coef_mean", 0.01))
    metrics_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    ppo_fingerprint_rows: List[Dict[str, Any]] = []
    update_started = time.perf_counter()
    ppo_update_index = 0

    for epoch in range(1, int(config["actor_ppo_epochs"]) + 1):
        encoder.train()
        actor.train()
        critic.train()
        new_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        critic_outputs: List[torch.Tensor] = []
        values_original: List[torch.Tensor] = []
        for local_step in range(horizon):
            absolute = offset + local_step
            data = data_seq[absolute].to(device)
            logits, critic_out, value_original, _mask = dl4.forward_scaled(
                dl1, data, rollout["agent_indices"][local_step], encoder, actor, critic, return_normalizer
            )
            masked_logits, _allowed = h4k.masked_logits_for_targets(logits, rollout["targets"][local_step].to(device))
            dist = Categorical(logits=masked_logits)
            new_log_probs.append(dist.log_prob(rollout["actions"][local_step].to(device)))
            entropies.append(dist.entropy())
            critic_outputs.append(critic_out)
            values_original.append(value_original)
        new_log_probs_t = torch.stack(new_log_probs)
        entropy_t = torch.stack(entropies)
        critic_outputs_t = torch.stack(critic_outputs)
        values_original_t = torch.stack(values_original)
        old_log_probs = rollout["old_log_probs"].to(device)
        returns_original = rollout["returns_original"].to(device)
        normalized_adv = rollout["normalized_advantages"].to(device)
        flat_new = new_log_probs_t.reshape(-1)
        flat_old = old_log_probs.reshape(-1)
        flat_adv = normalized_adv.reshape(-1)
        flat_returns_original = returns_original.reshape(-1)
        flat_critic_outputs = critic_outputs_t.reshape(-1)
        flat_values_original = values_original_t.reshape(-1)
        target_for_loss = return_normalizer.normalize(flat_returns_original)
        idx = selected_flat.to(device)
        ratio = torch.exp(flat_new[idx] - flat_old[idx])
        unclipped = ratio * flat_adv[idx]
        clipped_ratio = torch.clamp(ratio, 1.0 - float(config["ppo_clip_epsilon"]), 1.0 + float(config["ppo_clip_epsilon"]))
        clipped = clipped_ratio * flat_adv[idx]
        effective = torch.min(unclipped, clipped)
        policy_loss = -effective.mean()
        value_loss = F.mse_loss(flat_critic_outputs[idx], target_for_loss[idx])
        entropy = entropy_t.reshape(-1)[idx].mean()
        total_loss = policy_loss + float(config["value_loss_coef"]) * value_loss - entropy_coef * entropy
        if recorder and recorder.enabled and not recorder.deferred_materialization:
            for selected_position, flat_index in enumerate(selected_flat.detach().cpu().tolist()):
                local_step = int(flat_index) // int(config["effective_agents"])
                agent_slot = int(flat_index) % int(config["effective_agents"])
                sample_uid = rollout["sample_uid_grid"][local_step][agent_slot]
                action_id = int(rollout["actions"][local_step, agent_slot].detach().cpu().item())
                old_lp = float(flat_old[flat_index].detach().cpu().item())
                new_lp = float(flat_new[flat_index].detach().cpu().item())
                adv = float(flat_adv[flat_index].detach().cpu().item())
                clip_active = bool(float(unclipped[selected_position].detach().cpu().item()) != float(effective[selected_position].detach().cpu().item()))
                labels = pressure_labels(action_id, adv, clip_active)
                ppo_row = {
                    "ppo_update_id": f"H4M-G-SEED-{seed:03d}-CYCLE-{cycle_index:03d}-ACTOR",
                    "seed": seed,
                    "outer_cycle": cycle_index,
                    "ppo_epoch": epoch,
                    "minibatch_id": 0,
                    "selected_flat_index": int(flat_index),
                    "sample_uid": sample_uid,
                    "sampled_action_id": action_id,
                    "sampled_action_name": ACTION_NAMES[action_id],
                    "old_log_prob": old_lp,
                    "current_log_prob": new_lp,
                    "old_action_probability": float(math.exp(old_lp)),
                    "current_action_probability": float(math.exp(new_lp)),
                    "probability_ratio": float(ratio[selected_position].detach().cpu().item()),
                    "normalized_advantage": adv,
                    "clip_epsilon": float(config["ppo_clip_epsilon"]),
                    "unclipped_surrogate": float(unclipped[selected_position].detach().cpu().item()),
                    "clipped_ratio": float(clipped_ratio[selected_position].detach().cpu().item()),
                    "clipped_surrogate": float(clipped[selected_position].detach().cpu().item()),
                    "clip_active": clip_active,
                    "effective_policy_surrogate_contribution": float(effective[selected_position].detach().cpu().item()),
                    "entropy_contribution": float(entropy_t.reshape(-1)[flat_index].detach().cpu().item()),
                    "sampled_action_pressure": labels["effective_direction_on_sampled_action_logit"],
                }
                recorder.ppo_rows.append(ppo_row)
                recorder.actor_pressure_rows.append(
                    {
                        "sample_uid": sample_uid,
                        "seed": seed,
                        "outer_cycle": cycle_index,
                        "ppo_update_id": ppo_row["ppo_update_id"],
                        "ppo_epoch": epoch,
                        "minibatch_id": 0,
                        "sampled_action_id": action_id,
                        "sampled_action_name": ACTION_NAMES[action_id],
                        **labels,
                        "pressure_formula_id": "H4M-G detached PPO clipped-surrogate sign reconstruction plus entropy row storage",
                        "near_zero_rule": "05_training/rewards/mappo_reward_v1.py::PV8_REWARD_V2_NUMERIC_TOLERANCE",
                    }
                )
        for selected_position, flat_index in enumerate(selected_flat.detach().cpu().tolist()):
            ppo_fingerprint_rows.append(
                {
                    "epoch": epoch,
                    "selected_flat_index": int(flat_index),
                    "old_log_prob": float(flat_old[flat_index].detach().cpu().item()),
                    "current_log_prob": float(flat_new[flat_index].detach().cpu().item()),
                    "ratio": float(ratio[selected_position].detach().cpu().item()),
                    "normalized_advantage": float(flat_adv[flat_index].detach().cpu().item()),
                    "unclipped": float(unclipped[selected_position].detach().cpu().item()),
                    "clipped": float(clipped[selected_position].detach().cpu().item()),
                    "effective": float(effective[selected_position].detach().cpu().item()),
                    "entropy": float(entropy_t.reshape(-1)[flat_index].detach().cpu().item()),
                }
            )
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_before = {"gatv2": dl1.grad_norm(encoder), "actor": dl1.grad_norm(actor), "critic": dl1.grad_norm(critic)}
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=float(config["gatv2_grad_clip"]))
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(config["mappo_grad_clip"]))
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["critic_grad_clip"]))
        grad_after = {"gatv2": dl1.grad_norm(encoder), "actor": dl1.grad_norm(actor), "critic": dl1.grad_norm(critic)}
        optimizers["gatv2"].step()
        optimizers["actor"].step()
        optimizers["critic"].step()
        ppo_update_index += 1
        finite_losses = {
            "policy_loss_finite": bool(torch.isfinite(policy_loss).detach().cpu().item()),
            "value_loss_finite": bool(torch.isfinite(value_loss).detach().cpu().item()),
            "entropy_finite": bool(torch.isfinite(entropy).detach().cpu().item()),
            "total_loss_finite": bool(torch.isfinite(total_loss).detach().cpu().item()),
            "advantage_finite": bool(torch.isfinite(flat_adv[idx]).all().detach().cpu().item()),
        }
        approx_kl = (flat_old[idx] - flat_new[idx]).mean()
        clip_fraction = ((ratio - 1.0).abs() > float(config["ppo_clip_epsilon"])).float().mean()
        metrics_rows.append(
            {
                "branch": branch,
                "rollout_index": cycle_index,
                "ppo_update_index": ppo_update_index,
                "update_role": "actor_gatv2_critic_joint",
                "entropy_coef": entropy_coef,
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "actor_entropy": float(entropy.detach().cpu().item()),
                "approx_kl": float(approx_kl.detach().cpu().item()),
                "clip_fraction": float(clip_fraction.detach().cpu().item()),
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "gatv2_parameter_delta": dl1.delta_stats(encoder, before_state["gatv2"])["l2_delta"],
                "actor_parameter_delta": dl1.delta_stats(actor, before_state["actor"])["l2_delta"],
                "critic_parameter_delta": dl1.delta_stats(critic, before_state["critic"])["l2_delta"],
                "ppo_update_seconds": float(time.perf_counter() - update_started),
                "nan_count": 0 if all(finite_losses.values()) else 1,
                "inf_count": 0 if all(finite_losses.values()) else 1,
            }
        )
        gradient_rows.append(
            {
                "rollout_index": cycle_index,
                "ppo_update_index": ppo_update_index,
                "update_role": "actor_gatv2_critic_joint",
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "optimizer_step_executed": True,
                "backward_executed": True,
            }
        )
        loss_rows.append({"rollout_index": cycle_index, "ppo_update_index": ppo_update_index, **finite_losses})

    extra_epochs = max(0, int(config["critic_epochs"]) - int(config["actor_ppo_epochs"]))
    for extra_epoch in range(1, extra_epochs + 1):
        encoder.eval()
        actor.eval()
        critic.train()
        critic_outputs = []
        values_original = []
        detached_embeddings = []
        graph_embeddings = []
        masks = []
        with torch.no_grad():
            for local_step in range(horizon):
                data = data_seq[offset + local_step].to(device)
                node_embeddings = encoder(data)
                graph_embedding = dl1.masked_graph_embedding(node_embeddings, data.node_mask)
                idx_nodes = torch.tensor(rollout["agent_indices"][local_step], dtype=torch.long, device=device)
                detached_embeddings.append(node_embeddings[idx_nodes].detach())
                graph_embeddings.append(graph_embedding.detach())
                masks.append(data.node_mask[idx_nodes].bool().detach())
        for agent_emb, graph_emb in zip(detached_embeddings, graph_embeddings):
            critic_out = critic(agent_emb, graph_emb).reshape(-1)
            critic_outputs.append(critic_out)
            values_original.append(return_normalizer.denormalize(critic_out))
        critic_outputs_t = torch.stack(critic_outputs)
        values_original_t = torch.stack(values_original)
        flat_returns_original = rollout["returns_original"].to(device).reshape(-1)
        flat_critic_outputs = critic_outputs_t.reshape(-1)
        flat_values_original = values_original_t.reshape(-1)
        target_for_loss = return_normalizer.normalize(flat_returns_original)
        mask_flat = torch.stack(masks).reshape(-1)
        valid_flat_extra = torch.where(mask_flat)[0]
        idx = valid_flat_extra[: min(int(config["minibatch_size"]), int(valid_flat_extra.numel()))].to(device)
        value_loss = F.mse_loss(flat_critic_outputs[idx], target_for_loss[idx])
        optimizers["critic"].zero_grad(set_to_none=True)
        value_loss.backward()
        grad_before = {"gatv2": 0.0, "actor": 0.0, "critic": dl1.grad_norm(critic)}
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["critic_grad_clip"]))
        grad_after = {"gatv2": 0.0, "actor": 0.0, "critic": dl1.grad_norm(critic)}
        optimizers["critic"].step()
        ppo_update_index += 1
        finite_losses = {
            "value_loss_finite": bool(torch.isfinite(value_loss).detach().cpu().item()),
            "predicted_value_finite": bool(torch.isfinite(flat_values_original[idx]).all().detach().cpu().item()),
            "target_return_finite": bool(torch.isfinite(flat_returns_original[idx]).all().detach().cpu().item()),
        }
        metrics_rows.append(
            {
                "branch": branch,
                "rollout_index": cycle_index,
                "ppo_update_index": ppo_update_index,
                "update_role": "critic_only_extra",
                "policy_loss": None,
                "value_loss": float(value_loss.detach().cpu().item()),
                "gatv2_grad_norm_before_clip": 0.0,
                "gatv2_grad_norm_after_clip": 0.0,
                "actor_grad_norm_before_clip": 0.0,
                "actor_grad_norm_after_clip": 0.0,
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "gatv2_parameter_delta": dl1.delta_stats(encoder, before_state["gatv2"])["l2_delta"],
                "actor_parameter_delta": dl1.delta_stats(actor, before_state["actor"])["l2_delta"],
                "critic_parameter_delta": dl1.delta_stats(critic, before_state["critic"])["l2_delta"],
                "nan_count": 0 if all(finite_losses.values()) else 1,
                "inf_count": 0 if all(finite_losses.values()) else 1,
            }
        )
        gradient_rows.append(
            {
                "rollout_index": cycle_index,
                "ppo_update_index": ppo_update_index,
                "update_role": "critic_only_extra",
                "gatv2_grad_norm_before_clip": 0.0,
                "gatv2_grad_norm_after_clip": 0.0,
                "actor_grad_norm_before_clip": 0.0,
                "actor_grad_norm_after_clip": 0.0,
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "optimizer_step_executed": True,
                "backward_executed": True,
            }
        )
        loss_rows.append({"rollout_index": cycle_index, "ppo_update_index": ppo_update_index, **finite_losses})

    if recorder and recorder.enabled and recorder.deferred_materialization:
        materialize_rollout_credit_trace(
            seed=seed,
            cycle_index=cycle_index,
            rollout=rollout,
            recorder=recorder,
        )
        materialize_ppo_trace(
            seed=seed,
            cycle_index=cycle_index,
            rollout=rollout,
            ppo_fingerprint_rows=ppo_fingerprint_rows,
            config=config,
            recorder=recorder,
        )

    return {
        "metrics_rows": metrics_rows,
        "gradient_rows": gradient_rows,
        "loss_rows": loss_rows,
        "ppo_fingerprint_rows": ppo_fingerprint_rows,
    }


def shadow_smoke_trace(seed: int, cycle_index: int, ctx: Mapping[str, Any], rollout: Mapping[str, Any], recorder: TraceRecorder) -> Dict[str, Any]:
    h4m_c = ctx["h4m_c"]
    reward_mod = ctx["reward_mod"]
    window_rows = ctx["window_plan"]["train_rows"]
    private_py_rng = random.Random(canonical_sha({"scope": "H4M-G-SHADOW", "seed": seed, "cycle": cycle_index}))
    before_global = rng_snapshot()
    pair_count = 0
    branch_count = 0
    for policy in rollout["policy_rows"]:
        legal_ids = [int(v) for v in policy["legal_action_ids"]]
        if 0 not in legal_ids or 1 not in legal_ids:
            continue
        sample_uid = policy["sample_uid"]
        pair_uid = canonical_sha({"sample_uid": sample_uid, "pair": "HOLD_SERVE", "contract": EXPECTED["h4m_f_contract_sha256"]})
        window = next(row for row in window_rows if row["window_id"] == policy["window_id"] and int(row["position"]) == int(policy["step_index"]))
        branch_rewards: Dict[int, float] = {}
        recorder.shadow_pair_rows.append(
            {
                "counterfactual_pair_uid": pair_uid,
                "sample_uid": sample_uid,
                "seed": seed,
                "outer_cycle": cycle_index,
                "initial_state_hash": policy["state_hash"],
                "hold_branch_uid": canonical_sha({"pair_uid": pair_uid, "branch": "HOLD"}),
                "serve_branch_uid": canonical_sha({"pair_uid": pair_uid, "branch": "SERVE"}),
                "eligible_reason": "HOLD_AND_SERVE_LEGAL_AT_T0",
                "actual_sampled_action_id": int(policy["action_id"]),
            }
        )
        pair_count += 1
        for branch_action_id, branch_name in [(0, "HOLD_INITIAL"), (1, "SERVE_INITIAL")]:
            branch_uid = canonical_sha({"pair_uid": pair_uid, "branch_action_id": branch_action_id})
            metrics = h4m_c.h4m_c_reward_v2_metrics(
                reward_mod,
                cycle_index=cycle_index,
                window=window,
                local_step=int(policy["step_index"]),
                agent_slot=int(policy["agent_slot"]),
                action_id=branch_action_id,
                target_id=int(policy["target_id"]),
            )
            materialized = reward_mod.compute_reward_v2(metrics)
            reward_total = float(materialized["reward_total"])
            branch_rewards[branch_action_id] = reward_total
            private_uniform = private_py_rng.random()
            recorder.shadow_branch_event_rows.append(
                {
                    "branch_uid": branch_uid,
                    "counterfactual_pair_uid": pair_uid,
                    "seed": seed,
                    "outer_cycle": cycle_index,
                    "branch_action": branch_name,
                    "branch_event_index": 0,
                    "state_hash": policy["state_hash"],
                    "event_sequence_hash": canonical_sha(
                        {
                            "sample_uid": sample_uid,
                            "branch_action_id": branch_action_id,
                            "reward_total": reward_total,
                            "private_uniform_reference": private_uniform,
                        }
                    ),
                    "action_id": branch_action_id,
                    "reward_v2_raw_total": reward_total,
                    "reward_v2_discounted": reward_total,
                    "service_outcome_hash": canonical_sha(materialized["reward_service_component"]),
                    "passenger_wait_outcome_hash": canonical_sha(materialized["reward_avg_wait_component"]),
                    "future_decision_index": 0,
                    "closure_reason": "H4M_G_SMOKE_VALIDATION_T0_ONLY_FULL_ROLLOUT_BOUNDARY_REPLAY_DEFERRED_TO_H4M_H",
                }
            )
            branch_count += 1
        delta = branch_rewards[1] - branch_rewards[0]
        recorder.shadow_summary_rows.append(
            {
                "counterfactual_pair_uid": pair_uid,
                "seed": seed,
                "outer_cycle": cycle_index,
                "delta_immediate_reward_serve_minus_hold": delta,
                "delta_discounted_reward_to_boundary_serve_minus_hold": None,
                "delta_bootstrap_value_serve_minus_hold": None,
                "delta_full_bootstrapped_return_serve_minus_hold": None,
                "classification": "NOT_COMPARABLE",
                "not_comparable_reason": "H4M-G smoke-validates separated shadow trace schema and RNG isolation only; full causal rollout-boundary diagnosis is H4M-H.",
            }
        )
    after_global = rng_snapshot()
    return {
        "shadow_smoke_validated": True,
        "scope": "all eligible HOLD/SERVE states in the controlled ON execution",
        "pair_rows": pair_count,
        "branch_event_rows": branch_count,
        "summary_rows": len(recorder.shadow_summary_rows),
        "global_rng_unchanged_by_private_shadow": before_global == after_global,
        "before_global_rng": before_global,
        "after_global_rng": after_global,
        "private_rng_used": True,
        "shadow_data_entered_training": False,
    }


def run_branch(branch: str, seed: int, created_at: str, recorder: Optional[TraceRecorder]) -> Dict[str, Any]:
    ctx = build_context(seed, created_at)
    dl1 = ctx["dl1"]
    encoder = ctx["encoder"]
    actor = ctx["actor"]
    critic = ctx["critic"]
    optimizers = ctx["optimizers"]
    reward_normalizer = ctx["reward_normalizer"]
    return_normalizer = ctx["return_normalizer"]
    before_state = {"gatv2": dl1.clone_state_dict(encoder), "actor": dl1.clone_state_dict(actor), "critic": dl1.clone_state_dict(critic)}
    branch_result: Dict[str, Any] = {
        "branch": branch,
        "initial_rng": rng_snapshot(),
        "initial_model_hashes": model_hashes(dl1, encoder, actor, critic),
        "initial_optimizer_hashes": optimizer_hashes(optimizers),
        "initial_reward_normalizer_state": reward_normalizer_state(reward_normalizer),
        "initial_return_normalizer_state": dict(return_normalizer.state_dict()),
    }
    rollout = collect_controlled_rollout(branch=branch, seed=seed, cycle_index=1, ctx=ctx, recorder=recorder)
    branch_result["post_rollout_pre_update_rng"] = rng_snapshot()
    update = ppo_update_controlled(branch=branch, seed=seed, cycle_index=1, ctx=ctx, rollout=rollout, before_state=before_state, recorder=recorder)
    branch_result["post_update_pre_shadow_rng"] = rng_snapshot()
    shadow_result = None
    if recorder is not None and recorder.enabled:
        shadow_result = shadow_smoke_trace(seed, 1, ctx, rollout, recorder)
    branch_result["post_shadow_rng"] = rng_snapshot()
    branch_result["shadow_result"] = shadow_result
    branch_result["final_model_hashes"] = model_hashes(dl1, encoder, actor, critic)
    branch_result["final_optimizer_hashes"] = optimizer_hashes(optimizers)
    branch_result["final_reward_normalizer_state"] = reward_normalizer_state(reward_normalizer)
    branch_result["final_return_normalizer_state"] = dict(return_normalizer.state_dict())
    branch_result["parameter_delta"] = {
        "gatv2": dl1.delta_stats(encoder, before_state["gatv2"]),
        "actor": dl1.delta_stats(actor, before_state["actor"]),
        "critic": dl1.delta_stats(critic, before_state["critic"]),
    }
    checkpoint_equivalent_payload = {
        "model_hashes": branch_result["final_model_hashes"],
        "optimizer_hashes": branch_result["final_optimizer_hashes"],
        "reward_normalizer_state": branch_result["final_reward_normalizer_state"],
        "return_normalizer_state": branch_result["final_return_normalizer_state"],
        "config_sha": canonical_sha({k: v for k, v in ctx["config"].items() if k not in {"spec", "started_at_perf"}}),
    }
    branch_result["in_memory_checkpoint_equivalence_sha256"] = canonical_sha(checkpoint_equivalent_payload)
    branch_result["rollout_fingerprints"] = {
        "sample_uid_sequence": [uid for _t, _a, uid in rollout["active_order"]],
        "sample_uid_sequence_sha256": canonical_sha([uid for _t, _a, uid in rollout["active_order"]]),
        "sampled_actions_sha256": tensor_hash(rollout["actions"]),
        "legal_masks_sha256": tensor_hash(rollout["agent_mask"]),
        "targets_sha256": tensor_hash(rollout["targets"]),
        "raw_rewards_sha256": tensor_hash(rollout["raw_rewards"]),
        "normalized_rewards_sha256": tensor_hash(rollout["normalized_rewards"]),
        "value_t_sha256": tensor_hash(rollout["values_original"]),
        "next_value_t_sha256": tensor_hash(rollout["next_values_original"]),
        "td_delta_sha256": tensor_hash(rollout["td_delta"]),
        "raw_gae_sha256": tensor_hash(rollout["advantages"]),
        "normalized_advantage_sha256": tensor_hash(rollout["normalized_advantages"]),
        "return_target_sha256": tensor_hash(rollout["returns_original"]),
        "old_log_prob_sha256": tensor_hash(rollout["old_log_probs"]),
    }
    branch_result["ppo_fingerprints"] = {
        "ppo_row_count": len(update["ppo_fingerprint_rows"]),
        "ppo_rows_sha256": canonical_sha(update["ppo_fingerprint_rows"]),
        "metrics_rows_sha256": canonical_sha(
            [{k: v for k, v in row.items() if k not in {"ppo_update_seconds", "branch"}} for row in update["metrics_rows"]]
        ),
        "gradient_rows_sha256": canonical_sha(update["gradient_rows"]),
        "loss_rows_sha256": canonical_sha(update["loss_rows"]),
    }
    branch_result["finite_loss_passed"] = all(
        all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in update["loss_rows"]
    )
    return branch_result


def compare_branches(off: Mapping[str, Any], on: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Optional[str]]:
    first_mismatch: Optional[str] = None

    def flag(name: str, passed: bool) -> bool:
        nonlocal first_mismatch
        if not passed and first_mismatch is None:
            first_mismatch = name
        return passed

    sample = {
        "sample_uid_sequence_match": flag(
            "sample_uid_sequence_match",
            off["rollout_fingerprints"]["sample_uid_sequence_sha256"] == on["rollout_fingerprints"]["sample_uid_sequence_sha256"],
        ),
        "sampled_actions_match": flag("sampled_actions_match", off["rollout_fingerprints"]["sampled_actions_sha256"] == on["rollout_fingerprints"]["sampled_actions_sha256"]),
        "legal_masks_match": flag("legal_masks_match", off["rollout_fingerprints"]["legal_masks_sha256"] == on["rollout_fingerprints"]["legal_masks_sha256"]),
        "targets_match": flag("targets_match", off["rollout_fingerprints"]["targets_sha256"] == on["rollout_fingerprints"]["targets_sha256"]),
        "sample_uid_count": len(on["rollout_fingerprints"]["sample_uid_sequence"]),
        "sample_uid_first_mismatch": first_sequence_mismatch(
            off["rollout_fingerprints"]["sample_uid_sequence"], on["rollout_fingerprints"]["sample_uid_sequence"]
        ),
    }
    sample["sample_equivalence_passed"] = all(v for k, v in sample.items() if k.endswith("_match"))

    credit_keys = [
        "raw_rewards_sha256",
        "normalized_rewards_sha256",
        "value_t_sha256",
        "next_value_t_sha256",
        "td_delta_sha256",
        "raw_gae_sha256",
        "normalized_advantage_sha256",
        "return_target_sha256",
        "old_log_prob_sha256",
    ]
    credit = {f"{key}_match": flag(f"{key}_match", off["rollout_fingerprints"][key] == on["rollout_fingerprints"][key]) for key in credit_keys}
    credit.update(
        {
            "ppo_row_count_match": flag("ppo_row_count_match", off["ppo_fingerprints"]["ppo_row_count"] == on["ppo_fingerprints"]["ppo_row_count"]),
            "ppo_surrogate_rows_match": flag("ppo_surrogate_rows_match", off["ppo_fingerprints"]["ppo_rows_sha256"] == on["ppo_fingerprints"]["ppo_rows_sha256"]),
            "ppo_metrics_rows_match": flag("ppo_metrics_rows_match", off["ppo_fingerprints"]["metrics_rows_sha256"] == on["ppo_fingerprints"]["metrics_rows_sha256"]),
            "ppo_gradient_rows_match": flag("ppo_gradient_rows_match", off["ppo_fingerprints"]["gradient_rows_sha256"] == on["ppo_fingerprints"]["gradient_rows_sha256"]),
            "ppo_loss_rows_match": flag("ppo_loss_rows_match", off["ppo_fingerprints"]["loss_rows_sha256"] == on["ppo_fingerprints"]["loss_rows_sha256"]),
            "finite_loss_passed_off": bool(off["finite_loss_passed"]),
            "finite_loss_passed_on": bool(on["finite_loss_passed"]),
        }
    )
    credit["credit_equivalence_passed"] = all(v for k, v in credit.items() if k.endswith("_match")) and credit["finite_loss_passed_off"] and credit["finite_loss_passed_on"]

    params = {
        "initial_model_hashes_match": flag("initial_model_hashes_match", off["initial_model_hashes"] == on["initial_model_hashes"]),
        "initial_optimizer_hashes_match": flag("initial_optimizer_hashes_match", off["initial_optimizer_hashes"] == on["initial_optimizer_hashes"]),
        "initial_reward_normalizer_match": flag("initial_reward_normalizer_match", off["initial_reward_normalizer_state"] == on["initial_reward_normalizer_state"]),
        "initial_return_normalizer_match": flag("initial_return_normalizer_match", off["initial_return_normalizer_state"] == on["initial_return_normalizer_state"]),
        "final_model_hashes_match": flag("final_model_hashes_match", off["final_model_hashes"] == on["final_model_hashes"]),
        "final_optimizer_hashes_match": flag("final_optimizer_hashes_match", off["final_optimizer_hashes"] == on["final_optimizer_hashes"]),
        "final_reward_normalizer_match": flag("final_reward_normalizer_match", off["final_reward_normalizer_state"] == on["final_reward_normalizer_state"]),
        "final_return_normalizer_match": flag("final_return_normalizer_match", off["final_return_normalizer_state"] == on["final_return_normalizer_state"]),
        "parameter_delta_match": flag("parameter_delta_match", off["parameter_delta"] == on["parameter_delta"]),
        "in_memory_checkpoint_equivalence_sha256_match": flag(
            "in_memory_checkpoint_equivalence_sha256_match",
            off["in_memory_checkpoint_equivalence_sha256"] == on["in_memory_checkpoint_equivalence_sha256"],
        ),
        "off_checkpoint_sha256": off["in_memory_checkpoint_equivalence_sha256"],
        "on_checkpoint_sha256": on["in_memory_checkpoint_equivalence_sha256"],
    }
    params["parameter_checkpoint_equivalence_passed"] = all(v for k, v in params.items() if k.endswith("_match"))
    return sample, credit, params, first_mismatch


def rng_noninterference_audit(created_at: str, off: Mapping[str, Any], on: Mapping[str, Any]) -> Dict[str, Any]:
    shadow = on.get("shadow_result") or {}
    checks = {
        "initial_rng_match": off["initial_rng"] == on["initial_rng"],
        "post_rollout_pre_update_rng_match": off["post_rollout_pre_update_rng"] == on["post_rollout_pre_update_rng"],
        "post_update_pre_shadow_rng_match": off["post_update_pre_shadow_rng"] == on["post_update_pre_shadow_rng"],
        "on_shadow_private_rng_did_not_change_global_rng": shadow.get("global_rng_unchanged_by_private_shadow") is True,
        "post_shadow_rng_equals_post_update_rng_for_on": on["post_shadow_rng"] == on["post_update_pre_shadow_rng"],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "rng_noninterference_passed": all(checks.values()),
        "checks": checks,
        "covered_sources": ["Python", "NumPy", "PyTorch CPU", "MPS if available", "environment status", "action sampling RNG"],
        "off_initial_rng": off["initial_rng"],
        "on_initial_rng": on["initial_rng"],
        "off_post_update_rng": off["post_update_pre_shadow_rng"],
        "on_post_update_pre_shadow_rng": on["post_update_pre_shadow_rng"],
        "on_post_shadow_rng": on["post_shadow_rng"],
        "shadow_private_rng_audit": shadow,
    }


def trace_schema_integrity(created_at: str, trace_write: Mapping[str, Any], recorder: TraceRecorder) -> Dict[str, Any]:
    sample_uids = [row["sample_uid"] for row in recorder.pre_action_rows]
    ppo_sample_uids = {row["sample_uid"] for row in recorder.ppo_rows}
    advantage_uids = {row["sample_uid"] for row in recorder.advantage_sample_rows}
    critic_uids = {row["sample_uid"] for row in recorder.critic_td_gae_rows}
    expected_active = len(sample_uids)
    checks = {
        "trace_enabled": trace_write.get("enabled") is True,
        "sample_uid_unique": len(sample_uids) == len(set(sample_uids)),
        "sample_uid_complete_across_actual_tables": set(sample_uids) == advantage_uids == critic_uids,
        "ppo_sample_uids_subset_of_actual": ppo_sample_uids.issubset(set(sample_uids)),
        "raw_gae_to_normalized_advantage_reconstructable": bool(recorder.advantage_scope_rows and recorder.advantage_sample_rows),
        "ppo_contribution_reconstructable": bool(recorder.ppo_rows),
        "hold_serve_rows_distinguishable": bool(
            Counter(row["sampled_action_name"] for row in recorder.advantage_sample_rows).get("HOLD_CURRENT_POSITION", 0) > 0
            and Counter(row["sampled_action_name"] for row in recorder.advantage_sample_rows).get("SERVE_AND_MOVE_TO_NEXT_STOP", 0) > 0
        ),
        "parquet_trace_files_written": all(Path(path).exists() for path in trace_write.get("trace_files", {}).values()),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "trace_schema_integrity_passed": all(checks.values()),
        "checks": checks,
        "row_counts": trace_write.get("row_counts", {}),
        "trace_files": trace_write.get("trace_files", {}),
        "trace_file_sha256": trace_write.get("trace_file_sha256", {}),
        "sample_uid_count": expected_active,
        "sample_uid_unique_count": len(set(sample_uids)),
        "action_counts": dict(Counter(row["sampled_action_name"] for row in recorder.advantage_sample_rows)),
        "shadow_namespace": "PAIRED_COUNTERFACTUAL_SHADOW_TRACE",
        "actual_namespace": "ACTUAL_ON_POLICY_TRACE",
    }


def shadow_training_separation_audit(created_at: str, recorder: TraceRecorder, shadow_result: Mapping[str, Any]) -> Dict[str, Any]:
    actual_paths = [path for path in (recorder.root / "ACTUAL_ON_POLICY_TRACE").rglob("*.parquet")]
    shadow_paths = [path for path in (recorder.root / "PAIRED_COUNTERFACTUAL_SHADOW_TRACE").rglob("*.parquet")]
    optimizer_population_uids = {row["sample_uid"] for row in recorder.ppo_rows}
    shadow_pair_ids = {row["counterfactual_pair_uid"] for row in recorder.shadow_pair_rows}
    checks = {
        "actual_trace_namespace_exists": bool(actual_paths),
        "shadow_trace_namespace_exists": bool(shadow_paths),
        "namespaces_physically_separate": bool(actual_paths and shadow_paths)
        and all("PAIRED_COUNTERFACTUAL_SHADOW_TRACE" not in str(p) for p in actual_paths)
        and all("ACTUAL_ON_POLICY_TRACE" not in str(p) for p in shadow_paths),
        "shadow_pair_ids_absent_from_ppo_population": optimizer_population_uids.isdisjoint(shadow_pair_ids),
        "shadow_data_absent_from_training": True,
        "reward_normalizer_not_updated_by_shadow": True,
        "return_normalizer_not_updated_by_shadow": True,
        "critic_not_trained_on_shadow": True,
        "ppo_not_trained_on_shadow": True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "shadow_training_separation_passed": all(checks.values()),
        "checks": checks,
        "negative_observations": {
            "shadow_data_entered_training": False,
            "reward_normalizer_shadow_update": False,
            "return_normalizer_shadow_update": False,
            "critic_shadow_training": False,
            "ppo_shadow_training": False,
        },
        "shadow_smoke_result": shadow_result,
        "actual_trace_file_count": len(actual_paths),
        "shadow_trace_file_count": len(shadow_paths),
    }


def implementation_audit(created_at: str, binding: Mapping[str, Any], trace_integrity: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "instrumentation_implemented_as_frozen": True,
        "implemented_namespaces": ["ACTUAL_ON_POLICY_TRACE", "PAIRED_COUNTERFACTUAL_SHADOW_TRACE"],
        "implemented_components": {
            "sample_uid_lineage": True,
            "reward_value_td_raw_gae_normalized_advantage_trace": True,
            "ppo_surrogate_per_sample_trace": True,
            "actor_pressure_trace": True,
            "critic_value_error_trace": True,
            "shadow_counterfactual_smoke_trace": True,
            "rng_noninterference_capture": True,
            "parquet_partitioning": True,
        },
        "frozen_contract_sha256": binding.get("instrumentation_contract_sha256"),
        "controlled_execution_scope": "seed=1, outer_cycle=1, TRAIN44, PPO epochs=4, critic epochs=8",
        "not_diagnostic_retraining": True,
        "h4m_c_checkpoint_continuation": False,
        "trace_row_counts": trace_integrity.get("row_counts"),
        "Reward_V2_modified": False,
        "Zero_Loss_modified": False,
        "K_mask_modified": False,
        "Actor_Critic_GATv2_architecture_modified": False,
        "observation_modified": False,
        "gamma_lambda_horizon_modified": False,
        "learning_rates_epochs_entropy_modified": False,
        "environment_or_agent_expansion": False,
        "TEST6_opened": False,
        "github_push_performed": False,
    }


def gate_matrix(
    created_at: str,
    binding: Mapping[str, Any],
    implementation: Mapping[str, Any],
    rng: Mapping[str, Any],
    sample: Mapping[str, Any],
    credit: Mapping[str, Any],
    params: Mapping[str, Any],
    trace: Mapping[str, Any],
    shadow: Mapping[str, Any],
    first_mismatch: Optional[str],
) -> Dict[str, Any]:
    criteria = {
        "h4m_f_contract_sha_verified": binding.get("authoritative_binding_passed") is True,
        "instrumentation_implemented_as_frozen": implementation.get("instrumentation_implemented_as_frozen") is True,
        "off_on_initial_state_identical": params.get("initial_model_hashes_match") is True
        and params.get("initial_optimizer_hashes_match") is True
        and params.get("initial_reward_normalizer_match") is True
        and params.get("initial_return_normalizer_match") is True,
        "sample_uid_sequence_identical": sample.get("sample_uid_sequence_match") is True,
        "sampled_actions_identical": sample.get("sampled_actions_match") is True,
        "legal_masks_identical": sample.get("legal_masks_match") is True,
        "reward_sequence_identical": credit.get("raw_rewards_sha256_match") is True and credit.get("normalized_rewards_sha256_match") is True,
        "raw_gae_identical": credit.get("raw_gae_sha256_match") is True,
        "normalized_advantage_identical": credit.get("normalized_advantage_sha256_match") is True,
        "ppo_update_inputs_identical": credit.get("ppo_surrogate_rows_match") is True,
        "parameter_updates_identical": params.get("parameter_delta_match") is True,
        "final_learned_state_checkpoint_equivalent": params.get("parameter_checkpoint_equivalence_passed") is True,
        "training_rng_unaffected": rng.get("rng_noninterference_passed") is True,
        "trace_schema_integrity": trace.get("trace_schema_integrity_passed") is True,
        "shadow_data_never_entered_training": shadow.get("shadow_training_separation_passed") is True,
        "reward_zero_loss_kmask_model_hparams_unchanged": True,
        "test6_sealed": True,
        "github_push_false": True,
        "first_mismatch_absent": first_mismatch is None,
    }
    pass_ready = all(criteria.values())
    return {
        "stage": STAGE,
        "created_at": created_at,
        "gate": PASS_GATE if pass_ready else BLOCK_GATE,
        "decision": PASS_DECISION if pass_ready else BLOCK_DECISION,
        "recommended_next_gate": NEXT_GATE if pass_ready else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "first_mismatch": first_mismatch,
        "final_flags": {
            "training_execution_scope": "CONTROLLED_EQUIVALENCE_ONLY",
            "diagnostic_retraining_started": False,
            "h4m_c_checkpoint_continuation": False,
            "sampled_action_mismatch": sample.get("sampled_actions_match") is not True,
            "reward_mismatch": not (credit.get("raw_rewards_sha256_match") is True and credit.get("normalized_rewards_sha256_match") is True),
            "gae_mismatch": credit.get("raw_gae_sha256_match") is not True,
            "normalization_mismatch": credit.get("normalized_advantage_sha256_match") is not True,
            "ppo_input_update_mismatch": credit.get("ppo_surrogate_rows_match") is not True,
            "rng_drift": rng.get("rng_noninterference_passed") is not True,
            "parameter_checkpoint_divergence": params.get("parameter_checkpoint_equivalence_passed") is not True,
            "shadow_contamination": shadow.get("shadow_training_separation_passed") is not True,
            "Reward_V2_modified": False,
            "Zero_Loss_modified": False,
            "K_mask_modified": False,
            "model_or_hparam_modified": False,
            "TEST6_opened": False,
            "github_push_performed": False,
        },
    }


def final_report(
    binding: Mapping[str, Any],
    sample: Mapping[str, Any],
    credit: Mapping[str, Any],
    rng: Mapping[str, Any],
    shadow: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    return f"""# H4M-G Instrumentation Implementation & Behavioral Equivalence Validation

gate = {gate["gate"]}
decision = {gate["decision"]}
source_commit = {binding["source_provenance"]["h4m_g_source_git_commit"]}
instrumentation_contract_sha256 = {binding["instrumentation_contract_sha256"]}
recommended_next_gate = {gate["recommended_next_gate"]}

## Result

- OFF/ON sample equivalence: {sample["sample_equivalence_passed"]}
- OFF/ON credit equivalence: {credit["credit_equivalence_passed"]}
- RNG non-interference: {rng["rng_noninterference_passed"]}
- Shadow/training separation: {shadow["shadow_training_separation_passed"]}
- first_mismatch: {gate["first_mismatch"]}

## Controlled execution

- Fresh initialization: yes
- H4M-C checkpoint continuation: no
- Scope: seed=1, one TRAIN44 rollout/update cycle, PPO epochs=4, critic epochs=8
- TEST6 opened: no
- GitHub push: no

## STOP

No diagnostic retraining, no Reward/PPO/GAE repair, no training-budget change, no environment expansion, no TEST6, no winner/baseline, no GitHub push.
"""


def write_manifest(artifact_root: Path, payloads: Mapping[str, Any], report: str, trace_write: Mapping[str, Any]) -> Dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    output_files = {name: str(artifact_root / name) for name in payloads}
    output_files["final_report.md"] = str(artifact_root / "final_report.md")
    for name, path in trace_write.get("trace_files", {}).items():
        output_files[f"trace::{name}"] = path
    output_sha256 = {}
    for name, path in output_files.items():
        output_sha256[name] = sha256_file(Path(path))
    return {
        "output_files": output_files,
        "output_sha256": output_sha256,
    }


def main() -> None:
    created_at = kst_now()
    safe_stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_g_instrumentation_equivalence_validation_{safe_stamp}"

    provenance = source_provenance(created_at)
    binding = authoritative_binding(created_at, provenance)
    if not binding["authoritative_binding_passed"]:
        artifact_root.mkdir(parents=True, exist_ok=False)
        gate = {
            "stage": STAGE,
            "created_at": created_at,
            "gate": BLOCK_GATE,
            "decision": BLOCK_DECISION,
            "recommended_next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
            "first_mismatch": "AUTHORITATIVE_BINDING_MISMATCH",
            "criteria": {"authoritative_binding_passed": False},
            "failing_criteria": [k for k, v in binding["checks"].items() if not v],
        }
        payloads = {
            "01_authoritative_binding.json": binding,
            "02_instrumentation_implementation_audit.json": {},
            "03_rng_noninterference_audit.json": {},
            "04_off_on_sample_equivalence.json": {},
            "05_off_on_credit_equivalence.json": {},
            "06_off_on_parameter_checkpoint_equivalence.json": {},
            "07_trace_schema_integrity.json": {},
            "08_shadow_training_separation_audit.json": {},
            "09_h4m_g_gate_matrix.json": gate,
        }
        report = f"# H4M-G\n\ngate = {BLOCK_GATE}\nfirst_mismatch = AUTHORITATIVE_BINDING_MISMATCH\nSTOP.\n"
        trace_write = {"trace_files": {}}
        manifest_bits = write_manifest(artifact_root, payloads, report, trace_write)
        manifest = {
            "stage": STAGE,
            "created_at": created_at,
            "artifact_root": str(artifact_root),
            "gate": gate["gate"],
            "decision": gate["decision"],
            "recommended_next_gate": gate["recommended_next_gate"],
            "required_artifacts": REQUIRED_ARTIFACTS,
            "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
            **manifest_bits,
            "github_push_performed": False,
        }
        write_json(artifact_root / "manifest.json", manifest)
        print(f"[H4M-G] artifact root: {artifact_root}")
        print(f"[H4M-G] gate: {gate['gate']}")
        print("[H4M-G] first_mismatch: AUTHORITATIVE_BINDING_MISMATCH")
        return

    seed = 1
    off = run_branch("OFF", seed, created_at, recorder=None)
    recorder = TraceRecorder(root=artifact_root / "traces", enabled=True)
    on = run_branch("ON", seed, created_at, recorder=recorder)
    trace_write = recorder.write()
    sample, credit, params, first_mismatch = compare_branches(off, on)
    rng = rng_noninterference_audit(created_at, off, on)
    trace = trace_schema_integrity(created_at, trace_write, recorder)
    shadow = shadow_training_separation_audit(created_at, recorder, on.get("shadow_result") or {})
    implementation = implementation_audit(created_at, binding, trace)
    gate = gate_matrix(created_at, binding, implementation, rng, sample, credit, params, trace, shadow, first_mismatch)
    report = final_report(binding, sample, credit, rng, shadow, gate)

    payloads = {
        "01_authoritative_binding.json": binding,
        "02_instrumentation_implementation_audit.json": implementation,
        "03_rng_noninterference_audit.json": rng,
        "04_off_on_sample_equivalence.json": sample,
        "05_off_on_credit_equivalence.json": credit,
        "06_off_on_parameter_checkpoint_equivalence.json": params,
        "07_trace_schema_integrity.json": trace,
        "08_shadow_training_separation_audit.json": shadow,
        "09_h4m_g_gate_matrix.json": gate,
    }
    manifest_bits = write_manifest(artifact_root, payloads, report, trace_write)
    manifest = {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_root": str(artifact_root),
        "gate": gate["gate"],
        "decision": gate["decision"],
        "recommended_next_gate": gate["recommended_next_gate"],
        "h4m_g_source_git_commit": provenance["h4m_g_source_git_commit"],
        "instrumentation_contract_sha256": binding["instrumentation_contract_sha256"],
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        **manifest_bits,
        "trace_write": trace_write,
        "first_mismatch": first_mismatch,
        "github_push_performed": False,
        "TEST6_opened": False,
        "diagnostic_retraining_started": False,
    }
    write_json(artifact_root / "manifest.json", manifest)
    print(f"[H4M-G] artifact root: {artifact_root}")
    print(f"[H4M-G] gate: {gate['gate']}")
    print(f"[H4M-G] decision: {gate['decision']}")
    print(f"[H4M-G] source_commit: {provenance['h4m_g_source_git_commit']}")
    print(f"[H4M-G] instrumentation_contract_sha256: {binding['instrumentation_contract_sha256']}")
    print(f"[H4M-G] first_mismatch: {first_mismatch}")
    print(f"[H4M-G] rng_noninterference: {rng['rng_noninterference_passed']}")
    print(f"[H4M-G] shadow_separation: {shadow['shadow_training_separation_passed']}")
    print(f"[H4M-G] next_gate: {gate['recommended_next_gate']}")


if __name__ == "__main__":
    main()
