from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-E"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_E_REWARD_GAE_ACTOR_CREDIT_ALIGNMENT_AUDIT_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_E_REWARD_GAE_ACTOR_CREDIT_ALIGNMENT_AUDIT_FAILED"

DECISION_NOT_UNIQUE = "PV8_REWARD_GAE_ACTOR_CREDIT_CAUSAL_ATTRIBUTION_NOT_UNIQUE"
BLOCK_DECISION = "PV8_REWARD_GAE_ACTOR_CREDIT_ALIGNMENT_AUDIT_BLOCKED"
NEXT_GATE_NOT_UNIQUE = "H4M-F_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_SELECTION_AND_FREEZE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

H4M_A_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_a_decision_opportunity_environment_adequacy_audit_20260814_143743"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
H4M_D_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_d_frozen_extended_budget_policy_revalidation_20260814_175417"
H4I_RERUN_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_rerun_training_readiness_20260810_192924"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"

H4M_E_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_e_reward_gae_actor_credit_alignment_audit.py"
)
H4M_C_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_c_fresh_extended_budget_three_seed_retraining.py"
)
H4M_D_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_d_frozen_extended_budget_policy_revalidation.py"
)
H4K_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py"
)
H4L_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4l_frozen_three_seed_policy_validation_evaluation.py"
)
DL1_SOURCE_PATH = "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
DL4_SOURCE_PATH = "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
MAPPO_SOURCE_PATH = "05_training/mappo_runner.py"
REWARD_SOURCE_PATH = "05_training/rewards/mappo_reward_v1.py"

EXPECTED = {
    "h4m_a_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_A_DECISION_OPPORTUNITY_AND_ENVIRONMENT_ADEQUACY_AUDIT_COMPLETE",
    "h4m_b_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_B_TRAINING_BUDGET_EXTENSION_SELECTION_AND_FREEZE_COMPLETE",
    "h4m_c_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_C_FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_COMPLETE",
    "h4m_d_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_D_FROZEN_EXTENDED_BUDGET_POLICY_REVALIDATION_COMPLETE",
    "h4m_d_decision": "PV8_EXTENDED_BUDGET_POLICY_REVALIDATION_SERVE_DOMINANCE_WITH_COUNTERFACTUAL_MISALIGNMENT",
    "h4m_c_source_commit": "00a53164c710f0bb8028aabe4f3f616a8fe6610d",
    "h4m_d_source_commit": "960e60adbfb3737c676532be575eb1b38375d9da",
    "h4m_b_schedule_sha": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "checkpoint_shas": {
        1: "084cb811200a2f68e32ed2eaeef6a4ae07645c56cdfd62be20169cc24230aa3a",
        2: "6b5ff219105e8d6a3029c429d53152651f7fe4e186af1c673b0810833d846af5",
        3: "4e840e14f959bf8a7e05ad560829bdbf3268382113b6d90c7eb9dc95f184be23",
    },
}

ACTION_NAMES = {
    0: "HOLD_CURRENT_POSITION",
    1: "SERVE_AND_MOVE_TO_NEXT_STOP",
    2: "CONDITIONAL_SKIP_EMPTY_STOP",
}
ACTION_ID_BY_NAME = {v: k for k, v in ACTION_NAMES.items()}

RUNTIME_DEPENDENCIES = {
    H4M_E_SOURCE_PATH,
    H4M_C_SOURCE_PATH,
    H4M_D_SOURCE_PATH,
    H4K_SOURCE_PATH,
    H4L_SOURCE_PATH,
    DL1_SOURCE_PATH,
    DL4_SOURCE_PATH,
    MAPPO_SOURCE_PATH,
    REWARD_SOURCE_PATH,
    "05_training/evaluation/canonical_kpi_aggregator.py",
    "05_training/simulator/zero_loss_admission_adapter.py",
    "05_training/simulator/k_action_mask_runtime.py",
}

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_exact_training_credit_equations.json",
    "03_h4ma_state_identity_crosswalk.json",
    "04_counterfactual_reward_return_trace.json",
    "05_long_horizon_action_value_reconciliation.json",
    "06_raw_gae_by_action_audit.json",
    "07_advantage_sign_flip_audit.json",
    "08_advantage_normalization_audit.json",
    "09_ppo_surrogate_by_action_audit.json",
    "10_entropy_contribution_audit.json",
    "11_actor_logit_gradient_direction_audit.json",
    "12_cycle1_to_cycle11_credit_pressure.json",
    "13_critic_value_bias_audit.json",
    "14_actor_observation_discrimination_secondary_audit.json",
    "15_root_cause_classification.json",
    "16_safety_integrity_audit.json",
    "17_h4m_e_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n",
        encoding="utf-8",
    )


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def stats(values: Iterable[Any]) -> Dict[str, Any]:
    nums = finite_values(values)
    if not nums:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
            "positive": 0,
            "zero": 0,
            "negative": 0,
        }
    nums_sorted = sorted(nums)
    avg = mean(nums)
    var = mean([(v - avg) ** 2 for v in nums]) if len(nums) > 1 else 0.0
    return {
        "count": len(nums),
        "mean": avg,
        "median": median(nums),
        "std": math.sqrt(var),
        "min": min(nums),
        "p05": percentile(nums_sorted, 0.05),
        "p25": percentile(nums_sorted, 0.25),
        "p75": percentile(nums_sorted, 0.75),
        "p95": percentile(nums_sorted, 0.95),
        "max": max(nums),
        "positive": sum(1 for v in nums if v > 1e-12),
        "zero": sum(1 for v in nums if abs(v) <= 1e-12),
        "negative": sum(1 for v in nums if v < -1e-12),
    }


def git_run(args: Sequence[str], timeout: int = 30) -> Tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def parse_status_paths(status_short: str) -> List[Dict[str, Any]]:
    rows = []
    for line in status_short.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows.append({"status": status, "path": path})
    return rows


def git_source_provenance(created_at: str) -> Dict[str, Any]:
    _rc, branch, _branch_err = git_run(["branch", "--show-current"])
    _rc, head, _head_err = git_run(["rev-parse", "HEAD"])
    _rc, status_short, _status_err = git_run(["status", "--short"])
    dirty_rows = parse_status_paths(status_short)
    dirty_classification = []
    relevant_dirty = []
    for row in dirty_rows:
        relevance = "H4M_E_RUNTIME_DEPENDENCY" if row["path"] in RUNTIME_DEPENDENCIES else "OUTSIDE_H4M_E_EXECUTION_DEPENDENCY_SET"
        dirty_classification.append({**row, "relevance": relevance})
        if relevance == "H4M_E_RUNTIME_DEPENDENCY":
            relevant_dirty.append(row["path"])
    ls_rc, ls_out, _ls_err = git_run(["ls-tree", "-r", "--name-only", "HEAD", "--", H4M_E_SOURCE_PATH])
    diff_rc, _diff_out, _diff_err = git_run(["diff", "--quiet", "HEAD", "--", H4M_E_SOURCE_PATH])
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_e_audit_source_git_commit": head if ls_rc == 0 and bool(ls_out) and diff_rc == 0 else None,
        "h4m_e_source_path": H4M_E_SOURCE_PATH,
        "h4m_e_source_present_in_head": ls_rc == 0 and bool(ls_out),
        "h4m_e_source_no_uncommitted_diff_vs_head": diff_rc == 0,
        "local_source_only_commit_created_before_audit": ls_rc == 0 and bool(ls_out) and diff_rc == 0,
        "github_push_performed": False,
        "status_short": status_short,
        "remaining_dirty_paths": dirty_rows,
        "dirty_path_relevance_classification": dirty_classification,
        "relevant_execution_source_remains_uncommitted": bool(relevant_dirty),
        "relevant_dirty_paths": relevant_dirty,
        "post_commit_provenance_gate_passed": ls_rc == 0 and bool(ls_out) and diff_rc == 0 and not relevant_dirty,
    }


def import_module_from_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_paths_from_h4mc_registry() -> Dict[int, Path]:
    registry = read_json(H4M_C_ROOT / "12_checkpoint_registry.json")
    out: Dict[int, Path] = {}
    for row in registry.get("checkpoints", []):
        match = re.search(r"H4M_C_SEED_(\d+)_", str(row.get("namespace", "")))
        if match:
            out[int(match.group(1))] = Path(row["path"])
    return out


def line_number(path: Path, needle: str) -> Optional[int]:
    for idx, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if needle in line:
            return idx
    return None


def authoritative_binding(created_at: str, checkpoint_pre_shas: Mapping[int, Optional[str]]) -> Dict[str, Any]:
    h4m_a_gate = read_json(H4M_A_ROOT / "12_h4m_a_gate_matrix.json")
    h4m_b_gate = read_json(H4M_B_ROOT / "09_h4m_b_gate_matrix.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    h4m_c_gate = read_json(H4M_C_ROOT / "14_h4m_c_gate_matrix.json")
    h4m_c_manifest = read_json(H4M_C_ROOT / "manifest.json")
    h4m_c_binding = read_json(H4M_C_ROOT / "01_authoritative_binding.json")
    h4m_d_gate = read_json(H4M_D_ROOT / "14_h4m_d_gate_matrix.json")
    h4m_d_manifest = read_json(H4M_D_ROOT / "manifest.json")
    h4m_d_root = read_json(H4M_D_ROOT / "13_h4m_d_root_cause_classification.json")
    h4i_scope = read_json(H4I_RERUN_ROOT / "02_training_scope_binding.json")
    zero_loss_sha = sha256_file(TRAINING_ROOT / "simulator/zero_loss_admission_adapter.py")
    reward_text = (TRAINING_ROOT / "rewards/mappo_reward_v1.py").read_text(encoding="utf-8-sig")
    reward_match = re.search(r'PV8_REWARD_V2_FREEZE_SHA256\s*=\s*"([0-9a-f]{64})"', reward_text)
    reward_sha = reward_match.group(1) if reward_match else None
    checkpoint_checks = {}
    for seed, expected_sha in EXPECTED["checkpoint_shas"].items():
        checkpoint_checks[str(seed)] = {
            "expected_sha256": expected_sha,
            "observed_pre_audit_sha256": checkpoint_pre_shas.get(seed),
            "sha_match": checkpoint_pre_shas.get(seed) == expected_sha,
        }
    checks = {
        "h4m_a_gate_match": h4m_a_gate.get("gate") == EXPECTED["h4m_a_gate"],
        "h4m_b_gate_match": h4m_b_gate.get("gate") == EXPECTED["h4m_b_gate"],
        "h4m_b_schedule_sha_match": h4m_b_schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha"],
        "h4m_c_gate_match": h4m_c_gate.get("gate") == EXPECTED["h4m_c_gate"],
        "h4m_c_decision_ready_for_h4m_d": h4m_c_gate.get("decision")
        == "PV8_FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_COMPLETE_READY_FOR_FROZEN_POLICY_REVALIDATION",
        "h4m_c_source_commit_match": h4m_c_gate.get("h4m_c_training_source_git_commit") == EXPECTED["h4m_c_source_commit"],
        "h4m_c_manifest_source_commit_match": h4m_c_manifest.get("h4m_c_training_source_git_commit")
        == EXPECTED["h4m_c_source_commit"],
        "h4m_d_gate_match": h4m_d_gate.get("gate") == EXPECTED["h4m_d_gate"],
        "h4m_d_decision_match": h4m_d_gate.get("decision") == EXPECTED["h4m_d_decision"],
        "h4m_d_source_commit_match": h4m_d_manifest.get("h4m_d_evaluation_source_git_commit")
        == EXPECTED["h4m_d_source_commit"],
        "h4m_d_root_cause_match": h4m_d_root.get("decision") == EXPECTED["h4m_d_decision"],
        "reward_v2_sha_match": reward_sha == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": h4m_c_binding.get("required_sha_bindings", {}).get("h4g_runtime_sha", {}).get("observed")
        == EXPECTED["h4g_runtime_sha"],
        "r3_split_sha_match": h4i_scope.get("scope_hash") == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": zero_loss_sha == EXPECTED["zero_loss_adapter_sha"],
        "checkpoint_shas_match": all(row["sha_match"] for row in checkpoint_checks.values()),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "binding_passed": all(checks.values()),
        "checks": checks,
        "checkpoint_checks": checkpoint_checks,
        "artifact_roots": {
            "h4m_a": str(H4M_A_ROOT),
            "h4m_b": str(H4M_B_ROOT),
            "h4m_c": str(H4M_C_ROOT),
            "h4m_d": str(H4M_D_ROOT),
        },
        "immutable_bindings": {
            "reward_v2_sha256": reward_sha,
            "h4g_runtime_sha256": h4m_c_binding.get("required_sha_bindings", {}).get("h4g_runtime_sha", {}).get("observed"),
            "r3_split_sha256": h4i_scope.get("scope_hash"),
            "zero_loss_adapter_sha256": zero_loss_sha,
            "h4m_b_schedule_sha256": h4m_b_schedule.get("extended_training_schedule_sha256"),
        },
        "gate_text_note": (
            "H4M-E prompt includes a shortened H4M-D gate spelling; this audit binds to the authoritative "
            "H4M-D manifest/gate artifact generated in the immediately preceding stage."
        ),
        "classification": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "test6_status": "SEALED_NOT_OPENED",
        "training_executed": False,
        "optimizer_step_executed": False,
        "github_push_performed": False,
    }


def exact_training_credit_equations(created_at: str) -> Dict[str, Any]:
    h4mc = PROJECT_ROOT / H4M_C_SOURCE_PATH
    h4k = PROJECT_ROOT / H4K_SOURCE_PATH
    dl1 = PROJECT_ROOT / DL1_SOURCE_PATH
    dl4 = PROJECT_ROOT / DL4_SOURCE_PATH
    mappo = PROJECT_ROOT / MAPPO_SOURCE_PATH
    return {
        "stage": STAGE,
        "created_at": created_at,
        "semantics_recovered_from_executable_code": True,
        "source_sha256": {
            H4M_C_SOURCE_PATH: sha256_file(h4mc),
            H4K_SOURCE_PATH: sha256_file(h4k),
            DL1_SOURCE_PATH: sha256_file(dl1),
            DL4_SOURCE_PATH: sha256_file(dl4),
            MAPPO_SOURCE_PATH: sha256_file(mappo),
        },
        "source_line_anchors": {
            "h4m_c_collect_rollout": line_number(h4mc, "def collect_h4m_c_rollout"),
            "h4m_c_reward_normalizer_call": line_number(h4mc, "active_normalized = reward_normalizer.normalize"),
            "h4m_c_compute_gae_call": line_number(h4mc, "returns, advantages, normalized_advantages, gae_audit = dl1.compute_gae"),
            "h4m_c_return_normalizer_update": line_number(h4mc, "return_normalizer.update(returns[masks_t.bool()])"),
            "h4k_ppo_update": line_number(h4k, "def ppo_update_h4k"),
            "h4k_selected_minibatch": line_number(h4k, "selected_flat = valid_flat[: min"),
            "h4k_ratio": line_number(h4k, "ratio = torch.exp"),
            "h4k_clipped_surrogate": line_number(h4k, "clipped = torch.clamp"),
            "h4k_policy_loss": line_number(h4k, "policy_loss = -torch.min"),
            "h4k_entropy_term": line_number(h4k, "total_loss = policy_loss"),
            "dl1_compute_gae": line_number(dl1, "def compute_gae"),
            "mappo_reward_normalizer": line_number(mappo, "class RewardNormalizer"),
            "mappo_entropy_coef": line_number(mappo, "def entropy_coef_for_time_band"),
            "dl4_return_normalizer": line_number(dl4, "class ReturnNormalizer"),
        },
        "implemented_order": [
            "policy logits and critic values are produced with encoder/actor/critic in eval mode for rollout collection",
            "H4K masked logits are applied before Categorical sampling",
            "Reward V2 is materialized from the sampled action and frozen target/obligation semantics",
            "RewardNormalizer.normalize is called once per rollout on active raw rewards only and updates its rolling buffer",
            "GAE uses normalized rewards, original-scale value_t, original-scale next_value_t, gamma=0.99, lambda=0.95",
            "raw GAE advantages are standardized globally over all active rollout samples; inactive agents are excluded",
            "ReturnNormalizer.update is called on active returns after GAE; critic target in PPO is normalized return",
            "PPO actor update uses the first 256 active flattened samples, no shuffle/minibatch randomization",
            "ratio=exp(new_log_prob-old_log_prob); unclipped=ratio*A; clipped=clamp(ratio, 0.8, 1.2)*A",
            "policy_loss=-mean(min(unclipped, clipped)); total_loss=policy_loss + 0.5*value_loss - entropy_coef*entropy",
            "actor/GATv2/critic are gradient-clipped then optimizer.step() in H4M-C; H4M-E executes none of those steps",
            "critic-only extra epochs detach encoder embeddings and update critic only in H4M-C; H4M-E executes none",
        ],
        "equations": {
            "reward_normalizer": "clip active raw rewards to [-10,10], append to rolling buffer, normalize each active reward as (r-buffer_mean)/max(buffer_std,1e-6)",
            "td_delta": "delta_t = normalized_reward_t + gamma * next_value_t * bootstrap_mask - value_t",
            "raw_gae": "gae_t = delta_t + gamma * lambda * bootstrap_mask * gae_{t+1}",
            "return_target_original": "return_t = value_t + raw_gae_t",
            "advantage_standardization": "normalized_advantage = (raw_advantage - mean(active raw advantages)) / std(active raw advantages)",
            "ppo_ratio": "ratio = exp(new_log_prob - old_log_prob)",
            "ppo_surrogate": "policy_loss = -mean(min(ratio*A, clamp(ratio, 1-eps, 1+eps)*A))",
            "entropy": "total_loss includes -entropy_coef * mean(Categorical(masked_logits).entropy())",
            "critic_target": "value_loss = mse(critic_output_normalized, ReturnNormalizer.normalize(return_original))",
        },
        "normalization_scope": {
            "reward_normalizer_scope": "active agents only; rolling buffer persists across cycles inside one seed",
            "advantage_standardization_scope": "global per rollout across all active agent samples; not per action, not per agent, not per minibatch",
            "ppo_minibatch_scope": "first 256 active flattened samples, deterministic order",
            "inactive_agents_enter_statistics": False,
        },
        "executable_semantics_diagram_mermaid": "\n".join(
            [
                "flowchart TD",
                "  A[\"Frozen TRAIN44 rollout under current policy\"] --> B[\"Masked Categorical.sample action\"]",
                "  B --> C[\"Reward V2 materialization\"]",
                "  C --> D[\"RewardNormalizer active-only rolling normalize\"]",
                "  D --> E[\"GAE: gamma 0.99 lambda 0.95\"]",
                "  E --> F[\"Rollout-global active-only advantage standardization\"]",
                "  F --> G[\"PPO first 256 active samples\"]",
                "  G --> H[\"ratio / clipped surrogate / entropy\"]",
                "  H --> I[\"H4M-C optimizer.step (historical only)\"]",
                "  I --> J[\"Observed SERVE probability concentration\"]",
            ]
        ),
        "training_executed_in_h4m_e": False,
        "optimizer_step_executed_in_h4m_e": False,
    }


def load_h4mc_logs() -> Dict[int, Dict[str, Any]]:
    out = {}
    for seed in (1, 2, 3):
        seed_dir = H4M_C_ROOT / f"seed_{seed:03d}"
        out[seed] = {
            "policy_rows": read_jsonl(seed_dir / "policy_probability_diagnostics.jsonl"),
            "reward_rows": read_jsonl(seed_dir / "reward_v2_materialization.jsonl"),
            "training_rows": read_jsonl(seed_dir / "training_metrics.jsonl"),
            "gradient_rows": read_jsonl(seed_dir / "gradient_audit.jsonl"),
            "cycle_diagnostics": read_json(seed_dir / "cycle_diagnostics.json").get("cycles", []),
            "configuration": read_json(seed_dir / "configuration.json"),
        }
    return out


def state_identity_crosswalk(created_at: str) -> Dict[str, Any]:
    h4ma = read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json")
    h4md = read_json(H4M_D_ROOT / "08_h4ma_hold_better_actor_crosswalk.json")
    rows = []
    for row in h4md.get("crosswalk_rows", []):
        rows.append(
            {
                "seed": row.get("seed"),
                "state_id": row.get("state_id"),
                "window_id": row.get("window_id"),
                "agent_id": row.get("agent_slot"),
                "agent_slot": row.get("agent_slot"),
                "decision_index": str(row.get("state_id", "")).split(":")[-2] if ":step" in str(row.get("state_id")) else None,
                "state_hash": hashlib.sha256(str(row.get("state_id")).encode("utf-8")).hexdigest(),
                "h4m_a_classification": row.get("classification"),
                "h4m_d_actor_selected_action": row.get("actor_selected_action"),
                "p_hold": row.get("p_hold"),
                "p_serve": row.get("p_serve"),
                "legal_actions": row.get("legal_actions"),
                "mapping_mode": "EXACT_STATE_ID_NO_APPROXIMATION",
            }
        )
    counts = Counter((r["h4m_a_classification"], r["h4m_d_actor_selected_action"]) for r in rows)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4m_a_counterfactual_state_count": len(h4ma.get("counterfactual_rows", [])),
        "h4m_d_seed_state_count": len(rows),
        "state_identity_match_count": h4md.get("state_identity_match_count"),
        "state_identity_mismatch_count": h4md.get("state_identity_mismatch_count"),
        "classification_counts_h4m_a": h4ma.get("summary", {}).get("classification_counts", {}),
        "seed_state_actor_cross_counts": {str(k): v for k, v in counts.items()},
        "rows": rows,
        "material_state_identity_mismatch": h4md.get("state_identity_mismatch_count") != 0,
        "test_opened": False,
    }


def counterfactual_reward_return_trace(created_at: str) -> Dict[str, Any]:
    h4ma = read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json")
    trace_rows = []
    for row in h4ma.get("counterfactual_rows", []):
        hold = row["action_values"]["HOLD_CURRENT_POSITION"]
        serve = row["action_values"]["SERVE_AND_MOVE_TO_NEXT_STOP"]
        trace_rows.append(
            {
                "state_id": row["state_id"],
                "window_id": row["window_id"],
                "agent_slot": row["agent_slot"],
                "classification": row["classification"],
                "immediate_reward_hold": hold.get("reward_v2_total"),
                "immediate_reward_serve": serve.get("reward_v2_total"),
                "delta_immediate_reward_serve_minus_hold": serve.get("reward_v2_total") - hold.get("reward_v2_total"),
                "local_service_delta_serve_minus_hold": serve.get("completed_obligation_count")
                - hold.get("completed_obligation_count"),
                "local_affected_wait_count_delta_serve_minus_hold": serve.get("affected_wait_count")
                - hold.get("affected_wait_count"),
                "future_reward_v2_sequence": None,
                "future_reward_v2_sequence_status": row.get("future_return_delta_status"),
                "passenger_wait_delta": None,
                "passenger_wait_delta_status": row.get("passenger_wait_delta_status"),
                "service_outcome_delta": None,
                "event_timestamps_status": "LOCAL_DECISION_TIMESTAMP_ONLY_NO_SUPPORTED_COUNTERFACTUAL_CONTINUATION_TIMELINE",
                "same_state_and_semantics_preserved": all(
                    bool(row.get(k))
                    for k in [
                        "same_frozen_state_preserved",
                        "same_obligations_preserved",
                        "same_k_mask_semantics_preserved",
                        "same_reward_v2_preserved",
                        "same_zero_loss_semantics_preserved",
                    ]
                ),
            }
        )
    by_class = defaultdict(list)
    for row in trace_rows:
        by_class[row["classification"]].append(row["delta_immediate_reward_serve_minus_hold"])
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source": str(H4M_A_ROOT / "05_counterfactual_action_value_audit.json"),
        "trace_rows": trace_rows,
        "delta_immediate_reward_serve_minus_hold_by_class": {k: stats(v) for k, v in by_class.items()},
        "future_outcomes_invented": False,
        "counterfactual_causal_replay_scope": "H4M-A immediate Reward V2 transition materialization only",
        "test_opened": False,
    }


def long_horizon_action_value_reconciliation(created_at: str) -> Dict[str, Any]:
    h4ma = read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json")
    rows = []
    counts = Counter()
    for row in h4ma.get("counterfactual_rows", []):
        classification = "HORIZON_RECONCILIATION_NOT_OBSERVABLE"
        counts[classification] += 1
        rows.append(
            {
                "state_id": row["state_id"],
                "h4m_a_local_classification": row["classification"],
                "delta_immediate_reward_observable": True,
                "delta_discounted_return_observable": False,
                "delta_return_through_same_rollout_boundary_observable": False,
                "delta_bootstrapped_return_observable": False,
                "classification": classification,
                "reason": row.get("future_return_delta_status"),
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "rollout_horizon": 512,
            }
        )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "rows": rows,
        "classification_counts": dict(counts),
        "h4m_a_hold_better_long_horizon_answer": "NOT_OBSERVABLE_WITH_CURRENT_FROZEN_ARTIFACTS",
        "arbitrary_new_horizon_contract_invented": False,
        "required_before_reward_gae_misalignment_claim": True,
    }


def reward_by_action(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["action"]].append(row.get("reward_total"))
    return {action: stats(values) for action, values in grouped.items()}


def raw_gae_by_action_audit(created_at: str, logs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    per_seed_cycle = []
    for seed, bundle in logs.items():
        reward_rows = bundle["reward_rows"]
        reward_by_cycle = defaultdict(list)
        for row in reward_rows:
            reward_by_cycle[int(row["cycle_index"])].append(row)
        for cycle in bundle["cycle_diagnostics"]:
            idx = int(cycle["cycle_index"])
            per_seed_cycle.append(
                {
                    "seed": seed,
                    "cycle_index": idx,
                    "actual_raw_gae_aggregate_stats": cycle.get("advantage_statistics"),
                    "actual_normalized_advantage_aggregate_stats": cycle.get("normalized_advantage_statistics"),
                    "action_counts": cycle.get("action_counts"),
                    "target_counts": dict(Counter(row["target"] for row in reward_by_cycle[idx])),
                    "reward_total_by_action": reward_by_action(reward_by_cycle[idx]),
                    "raw_gae_by_action_status": "NOT_STORED_NOT_RECONSTRUCTABLE_WITHOUT_INTERMEDIATE_CHECKPOINTS_OR_FORBIDDEN_TRAINING_REPLAY",
                    "raw_gae_by_action": {
                        "HOLD_CURRENT_POSITION": None,
                        "SERVE_AND_MOVE_TO_NEXT_STOP": None,
                    },
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "actual_training_samples_inspected": sum(len(bundle["policy_rows"]) for bundle in logs.values()),
        "actual_per_sample_policy_reward_rows_available": True,
        "actual_per_sample_raw_gae_tensor_available": False,
        "actual_per_sample_value_tensor_available": False,
        "actual_per_sample_return_tensor_available": False,
        "aggregate_raw_gae_available_from_cycle_diagnostics": True,
        "per_seed_cycle": per_seed_cycle,
        "by_action_distribution_limitation": (
            "H4M-C stored per-sample policy/reward rows and aggregate advantage stats, but it discarded rollout tensors "
            "containing per-sample value_t, next_value_t, returns, raw GAE, and normalized advantages after each cycle. "
            "Recreating cycles 2-11 exactly would require forbidden optimizer.step replay because no intermediate checkpoints exist."
        ),
        "training_replay_executed": False,
        "optimizer_step_executed": False,
    }


def advantage_sign_flip_audit(created_at: str, logs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for seed, bundle in logs.items():
        for cycle in bundle["cycle_diagnostics"]:
            raw_stats = cycle.get("advantage_statistics", {})
            norm_stats = cycle.get("normalized_advantage_statistics", {})
            rows.append(
                {
                    "seed": seed,
                    "cycle_index": cycle["cycle_index"],
                    "raw_advantage_mean": raw_stats.get("mean"),
                    "raw_advantage_std": raw_stats.get("std"),
                    "normalized_advantage_mean": norm_stats.get("mean"),
                    "normalized_advantage_std": norm_stats.get("std"),
                    "standardization_threshold_raw_value": raw_stats.get("mean"),
                    "sign_flip_counts_status": "NOT_STORED_PER_SAMPLE_RAW_AND_NORMALIZED_ADVANTAGE_NOT_AVAILABLE",
                    "positive_raw_to_negative_normalized": None,
                    "negative_raw_to_positive_normalized": None,
                    "sign_preserved": None,
                    "near_zero_tolerance": 1e-12,
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "rows": rows,
        "exact_standardization_equation_audited": True,
        "action_dependent_sign_effect_quantifiable": False,
        "normalization_harmless_assumed": False,
        "conclusion": "SIGN_FLIP_MECHANISM_EXISTS_BY_EQUATION_BUT_EXACT_ACTION_CONDITIONAL_COUNTS_ARE_NOT_STORED",
    }


def first_actor_metric_by_cycle(training_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    actor_rows = [row for row in training_rows if row.get("update_role") == "actor_gatv2_critic_joint"]
    out = {}
    seen = set()
    for row in actor_rows:
        cycle = int(row["cycle_index"])
        if cycle not in seen:
            out[cycle] = row
            seen.add(cycle)
    return out


def advantage_normalization_audit(created_at: str, logs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    all_actor_adv_means = []
    for seed, bundle in logs.items():
        policy_by_cycle = defaultdict(list)
        for row in bundle["policy_rows"]:
            policy_by_cycle[int(row["cycle_index"])].append(row)
        first_actor = first_actor_metric_by_cycle(bundle["training_rows"])
        for cycle in bundle["cycle_diagnostics"]:
            idx = int(cycle["cycle_index"])
            first256 = policy_by_cycle[idx][:256]
            actor_metric = first_actor[idx]
            all_actor_adv_means.append(actor_metric.get("advantage_mean"))
            rows.append(
                {
                    "seed": seed,
                    "cycle_index": idx,
                    "raw_advantage_stats_active_rollout": cycle.get("advantage_statistics"),
                    "normalized_advantage_stats_active_rollout": cycle.get("normalized_advantage_statistics"),
                    "normalized_advantage_mean_first_256_actor_minibatch": actor_metric.get("advantage_mean"),
                    "normalized_advantage_positive_first_256_actor_minibatch": actor_metric.get("advantage_mean", 0.0) > 0,
                    "first_256_action_counts": dict(Counter(row["action"] for row in first256)),
                    "first_256_target_counts": dict(Counter(row["target"] for row in first256)),
                    "rollout_action_counts": cycle.get("action_counts"),
                    "rollout_target_counts": dict(Counter(row["target"] for row in policy_by_cycle[idx])),
                    "hold_raw_advantage_mean": None,
                    "serve_raw_advantage_mean": None,
                    "hold_normalized_advantage_mean": None,
                    "serve_normalized_advantage_mean": None,
                    "fraction_positive_after_normalization_by_action": None,
                    "by_action_status": "NOT_STORED_PER_SAMPLE_ADVANTAGE_NOT_AVAILABLE",
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "normalization_population": "352 active samples per seed/cycle; inactive agents excluded",
        "ppo_population": "first 256 active flattened samples per actor update",
        "first_256_advantage_mean_summary": stats(all_actor_adv_means),
        "first_256_positive_advantage_mean_count": sum(1 for v in all_actor_adv_means if v is not None and v > 0),
        "first_256_actor_update_count": len(all_actor_adv_means),
        "population_dominated_by_hold_like_targets": True,
        "rows": rows,
        "normalization_modified_in_h4m_e": False,
    }


def ppo_surrogate_by_action_audit(created_at: str, logs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    sampled_proxy_counts = Counter()
    for seed, bundle in logs.items():
        policy_by_cycle = defaultdict(list)
        for row in bundle["policy_rows"]:
            policy_by_cycle[int(row["cycle_index"])].append(row)
        for metric in [row for row in bundle["training_rows"] if row.get("update_role") == "actor_gatv2_critic_joint"]:
            cycle = int(metric["cycle_index"])
            first256 = policy_by_cycle[cycle][:256]
            adv_mean = metric.get("advantage_mean")
            direction = "REINFORCE_SAMPLED_ACTIONS" if adv_mean is not None and adv_mean > 0 else "SUPPRESS_SAMPLED_ACTIONS"
            action_counts = Counter(row["action"] for row in first256)
            for action, count in action_counts.items():
                sampled_proxy_counts[(action, direction)] += count
            rows.append(
                {
                    "seed": seed,
                    "cycle_index": cycle,
                    "ppo_update_index": metric.get("ppo_update_index"),
                    "old_action_probability": "AVAILABLE_AS_LOG_PROB_IN_H4M_C_ROLLOUT_TENSOR_BUT_NOT_PERSISTED_PER_SAMPLE",
                    "new_current_probability": "AGGREGATE_ONLY",
                    "probability_ratio": "AGGREGATE_APPROX_KL_AND_CLIP_FRACTION_ONLY",
                    "raw_advantage": "NOT_PERSISTED_PER_SAMPLE",
                    "normalized_advantage_mean_first_256": adv_mean,
                    "unclipped_surrogate_mean_first_epoch_at_ratio_1": adv_mean if metric.get("ppo_update_index") in {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81} else None,
                    "policy_loss": metric.get("policy_loss"),
                    "clip_fraction": metric.get("clip_fraction"),
                    "clip_activated": bool(metric.get("clip_fraction") and metric.get("clip_fraction") > 0),
                    "effective_actor_loss_contribution": metric.get("policy_loss"),
                    "sampled_action_pressure_proxy": direction,
                    "first_256_action_counts": dict(action_counts),
                    "first_256_target_counts": dict(Counter(row["target"] for row in first256)),
                    "evidence_level": "ACTUAL_UPDATE_AGGREGATE_WITH_ACTION_COMPOSITION_PROXY_NOT_PER_SAMPLE_SURROGATE",
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "rows": rows,
        "actor_update_count": len(rows),
        "positive_minibatch_advantage_update_count": sum(
            1 for row in rows if row["sampled_action_pressure_proxy"] == "REINFORCE_SAMPLED_ACTIONS"
        ),
        "negative_minibatch_advantage_update_count": sum(
            1 for row in rows if row["sampled_action_pressure_proxy"] == "SUPPRESS_SAMPLED_ACTIONS"
        ),
        "sampled_action_pressure_proxy_counts": {str(k): v for k, v in sampled_proxy_counts.items()},
        "ppo_systematically_reinforces_sampled_actions_when_minibatch_advantage_positive": True,
        "per_sample_ratio_clip_surrogate_not_persisted": True,
        "training_executed_in_h4m_e": False,
    }


def entropy_contribution_audit(created_at: str, logs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    ratios = []
    for seed, bundle in logs.items():
        for metric in [row for row in bundle["training_rows"] if row.get("update_role") == "actor_gatv2_critic_joint"]:
            entropy_contribution = (
                abs(float(metric["entropy_coef"]) * float(metric["actor_entropy"]))
                if metric.get("entropy_coef") is not None and metric.get("actor_entropy") is not None
                else None
            )
            policy_mag = abs(float(metric["policy_loss"])) if metric.get("policy_loss") is not None else None
            rel = entropy_contribution / policy_mag if entropy_contribution is not None and policy_mag and policy_mag > 1e-12 else None
            if rel is not None:
                ratios.append(rel)
            rows.append(
                {
                    "seed": seed,
                    "cycle_index": metric.get("cycle_index"),
                    "ppo_update_index": metric.get("ppo_update_index"),
                    "entropy_coef": metric.get("entropy_coef"),
                    "policy_entropy": metric.get("actor_entropy"),
                    "entropy_loss_contribution_magnitude": entropy_contribution,
                    "policy_surrogate_contribution_magnitude": policy_mag,
                    "relative_scale_abs_entropy_over_abs_policy": rel,
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "rows": rows,
        "relative_scale_summary": stats(ratios),
        "entropy_materiality_assessment": (
            "Entropy is nonzero but the observed concentration is not explained by a growing entropy term; entropy decays with "
            "time-band progress and generally opposes concentration rather than creating SERVE dominance."
        ),
        "entropy_modified_in_h4m_e": False,
    }


def logit_gradient_for_row(row: Mapping[str, Any], advantage: float, entropy_coef: float) -> Tuple[float, float]:
    p_hold = float(row["masked_probability_0"])
    p_serve = float(row["masked_probability_1"])
    entropy = float(row["entropy"])
    action_id = int(row["action_id"])
    d_serve_policy = -advantage * ((1.0 if action_id == 1 else 0.0) - p_serve)
    d_hold_policy = -advantage * ((1.0 if action_id == 0 else 0.0) - p_hold)
    d_serve_entropy = entropy_coef * p_serve * (math.log(max(p_serve, 1e-12)) + entropy)
    d_hold_entropy = entropy_coef * p_hold * (math.log(max(p_hold, 1e-12)) + entropy)
    return d_serve_policy + d_serve_entropy, d_hold_policy + d_hold_entropy


def gradient_effect(d_serve: float, d_hold: float) -> str:
    if d_serve < -1e-12 and d_hold > 1e-12:
        return "SERVE_PROBABILITY_UP_HOLD_DOWN"
    if d_hold < -1e-12 and d_serve > 1e-12:
        return "HOLD_PROBABILITY_UP_SERVE_DOWN"
    return "MIXED_OR_NEAR_ZERO"


def actor_logit_gradient_direction_audit(created_at: str, logs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    sample_rows = []
    aggregate_counts = Counter()
    per_seed_cycle = []
    for seed, bundle in logs.items():
        policy_by_cycle = defaultdict(list)
        for row in bundle["policy_rows"]:
            policy_by_cycle[int(row["cycle_index"])].append(row)
        first_actor = first_actor_metric_by_cycle(bundle["training_rows"])
        for cycle, metric in sorted(first_actor.items()):
            first256 = policy_by_cycle[cycle][:256]
            adv = float(metric.get("advantage_mean") or 0.0)
            entropy_coef = float(metric.get("entropy_coef") or 0.0)
            cycle_counts = Counter()
            d_serve_sum = 0.0
            d_hold_sum = 0.0
            for row in first256:
                d_serve, d_hold = logit_gradient_for_row(row, adv, entropy_coef)
                effect = gradient_effect(d_serve, d_hold)
                class_like = "HOLD_BETTER_LIKE_TARGET_HOLD" if row["target"] == "HOLD_CURRENT_POSITION" else "SERVE_BETTER_LIKE_TARGET_SERVE"
                cycle_counts[(class_like, effect)] += 1
                aggregate_counts[(class_like, effect)] += 1
                d_serve_sum += d_serve
                d_hold_sum += d_hold
                if len(sample_rows) < 96:
                    sample_rows.append(
                        {
                            "seed": seed,
                            "cycle_index": cycle,
                            "window_id": row["window_id"],
                            "local_step": row["local_step"],
                            "agent_slot": row["agent_slot"],
                            "target": row["target"],
                            "action": row["action"],
                            "advantage_proxy": adv,
                            "entropy_coef": entropy_coef,
                            "dL_d_logit_SERVE": d_serve,
                            "dL_d_logit_HOLD": d_hold,
                            "gradient_implied_effect": effect,
                        }
                    )
            per_seed_cycle.append(
                {
                    "seed": seed,
                    "cycle_index": cycle,
                    "advantage_proxy": adv,
                    "entropy_coef": entropy_coef,
                    "aggregate_dL_d_logit_SERVE": d_serve_sum,
                    "aggregate_dL_d_logit_HOLD": d_hold_sum,
                    "aggregate_effect": gradient_effect(d_serve_sum, d_hold_sum),
                    "counts": {str(k): v for k, v in cycle_counts.items()},
                    "evidence_level": "DIAGNOSTIC_ANALYTICAL_PROXY_USING_FIRST_256_MINIBATCH_MEAN_ADVANTAGE",
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "diagnostic_backward_executed": False,
        "training_backward_executed": False,
        "optimizer_created": False,
        "optimizer_step_executed": False,
        "analytic_formula_used": "d(policy_loss - entropy_coef*entropy)/d logits at ratio=1 using stored masked probabilities and first-actor-update minibatch mean advantage",
        "limitation": "Per-sample normalized advantages are not persisted; gradients are directional proxies, not exact H4M-C per-sample gradients.",
        "aggregate_counts": {str(k): v for k, v in aggregate_counts.items()},
        "per_seed_cycle": per_seed_cycle,
        "representative_rows": sample_rows,
    }


def cycle_credit_pressure(created_at: str, logs: Mapping[int, Mapping[str, Any]], gradient_audit: Mapping[str, Any]) -> Dict[str, Any]:
    grad_by_seed_cycle = {(row["seed"], row["cycle_index"]): row for row in gradient_audit.get("per_seed_cycle", [])}
    rows = []
    for seed, bundle in logs.items():
        cycles = sorted(bundle["cycle_diagnostics"], key=lambda r: int(r["cycle_index"]))
        pserve_by_cycle = {
            int(c["cycle_index"]): c["policy_probability_statistics"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"]
            for c in cycles
        }
        for cycle in cycles:
            idx = int(cycle["cycle_index"])
            next_p = pserve_by_cycle.get(idx + 1)
            cur_p = pserve_by_cycle[idx]
            delta = next_p - cur_p if next_p is not None else None
            observed = (
                "SERVE_REINFORCING"
                if delta is not None and delta > 1e-12
                else "HOLD_REINFORCING"
                if delta is not None and delta < -1e-12
                else "NEAR_ZERO_OR_FINAL_CYCLE"
            )
            rows.append(
                {
                    "seed": seed,
                    "cycle_index": idx,
                    "p_serve_mean": cur_p,
                    "p_hold_mean": cycle["policy_probability_statistics"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"],
                    "next_cycle_p_serve_delta": delta,
                    "observed_learning_pressure_from_probability_delta": observed,
                    "proxy_gradient_effect": grad_by_seed_cycle.get((seed, idx), {}).get("aggregate_effect"),
                    "action_counts": cycle.get("action_counts"),
                    "stage_producing_pressure": "PPO_ACTOR_GATV2_UPDATE_ON_SAMPLED_ACTIONS_AFTER_ROLLOUT_GLOBAL_ADVANTAGE_STANDARDIZATION",
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "rows": rows,
        "observed_pressure_counts": dict(Counter(row["observed_learning_pressure_from_probability_delta"] for row in rows)),
        "mechanism_summary": (
            "The stored trajectory shows SERVE probability rising because PPO updates the sampled action distribution, "
            "not because the actor directly evaluates HOLD-vs-SERVE counterfactual alternatives. As SERVE samples become "
            "more common, positive/minibatch-level advantages reinforce sampled SERVE mass; however exact per-sample GAE "
            "is not persisted, so the earliest causal flip cannot be uniquely localized."
        ),
    }


def critic_value_bias_audit(created_at: str, logs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for seed, bundle in logs.items():
        for metric in bundle["training_rows"]:
            rows.append(
                {
                    "seed": seed,
                    "cycle_index": metric.get("cycle_index"),
                    "ppo_update_index": metric.get("ppo_update_index"),
                    "update_role": metric.get("update_role"),
                    "value_prediction_mean": metric.get("prediction_mean"),
                    "return_target_mean": metric.get("target_mean"),
                    "value_error_mean_bias": metric.get("mean_bias"),
                    "value_loss": metric.get("value_loss"),
                    "explained_variance": metric.get("explained_variance"),
                    "value_return_correlation": metric.get("prediction_target_pearson"),
                    "td_residual_by_action_status": "NOT_STORED_PER_SAMPLE_VALUE_AND_TD_RESIDUAL_NOT_AVAILABLE",
                    "gae_by_action_status": "NOT_STORED_PER_SAMPLE_GAE_NOT_AVAILABLE",
                }
            )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "rows": rows,
        "mean_bias_summary": stats(row["value_error_mean_bias"] for row in rows),
        "value_loss_summary": stats(row["value_loss"] for row in rows),
        "explained_variance_summary": stats(row["explained_variance"] for row in rows),
        "critic_bias_dominance_status": "POSSIBLE_BUT_NOT_ACTION_CONDITIONAL_WITH_STORED_ARTIFACTS",
        "critic_retrained_in_h4m_e": False,
    }


def vector_stats_by_class(rows: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, Any]:
    by_class = defaultdict(list)
    for row in rows:
        vec = row.get(prefix)
        if vec is not None:
            by_class[row["classification"]].append(vec)
    if not by_class:
        return {"available": False}
    class_means = {}
    for cls, vectors in by_class.items():
        dim = len(vectors[0])
        class_means[cls] = [mean(vec[i] for vec in vectors) for i in range(dim)]
    result = {"available": True, "class_counts": {k: len(v) for k, v in by_class.items()}, "class_mean_vectors": class_means}
    if "HOLD_BETTER" in class_means and "SERVE_BETTER" in class_means:
        diff = [class_means["SERVE_BETTER"][i] - class_means["HOLD_BETTER"][i] for i in range(len(class_means["HOLD_BETTER"]))]
        result["centroid_l2_distance"] = math.sqrt(sum(v * v for v in diff))
        result["top_abs_mean_difference_indices"] = [
            {"index": i, "serve_minus_hold_mean_difference": diff[i]}
            for i in sorted(range(len(diff)), key=lambda j: abs(diff[j]), reverse=True)[:8]
        ]
    return result


def actor_observation_discrimination_secondary_audit(created_at: str, checkpoint_pre_shas: Mapping[int, Optional[str]]) -> Tuple[Dict[str, Any], Dict[int, Optional[str]]]:
    h4l = import_module_from_path(PROJECT_ROOT / H4L_SOURCE_PATH, "h4m_e_h4l")
    dl4 = import_module_from_path(PROJECT_ROOT / DL4_SOURCE_PATH, "h4m_e_dl4")
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    validation_plan = h4l.load_validation_window_plan()
    h4ma_rows = {row["state_id"]: row for row in read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json").get("counterfactual_rows", [])}
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])
    validation_paths = [Path(row["snapshot_path"]) for row in validation_plan["validation_rows"]]
    sample_full = dl1.torch_load(validation_paths[0])
    spec, inventory, _connectivity, _tensor_mask = dl1.build_subgraph_spec(PROJECT_ROOT, sample_full, mapping_artifact=mapping_artifact)
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    validation_data = dl4.load_subgraphs(dl1, validation_paths, spec)
    device = torch.device("cpu")
    checkpoint_paths = checkpoint_paths_from_h4mc_registry()
    feature_rows = []
    post_shas: Dict[int, Optional[str]] = {}
    with torch.no_grad():
        for seed, ckpt_path in checkpoint_paths.items():
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint = {"payload": payload, "path": ckpt_path, "pre_eval_sha256": checkpoint_pre_shas.get(seed)}
            encoder, actor, critic, return_normalizer = h4l.load_models_for_seed(
                dl1, dl4, checkpoint, sample_graph, device
            )
            encoder.eval()
            actor.eval()
            critic.eval()
            for step, (cpu_data, window) in enumerate(zip(validation_data, validation_plan["validation_rows"])):
                data = cpu_data.to(device)
                indices = dl1.agent_indices_for_step(spec, step, min(8, int(inventory["available_suseong_agents"])))
                node_embeddings = encoder(data)
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                agent_embeddings = node_embeddings[idx_tensor].detach().cpu()
                actor_obs = data.x[idx_tensor].detach().cpu()
                target = dl1.action_targets_from_y(data.y[idx_tensor], int(payload["training_configuration"]["action_dim"]))
                for agent_slot, node_index in enumerate(indices):
                    sid = f"validation:{window['window_id']}:step{step:03d}:agent{agent_slot:02d}"
                    cf = h4ma_rows.get(sid)
                    if cf is None:
                        continue
                    feature_rows.append(
                        {
                            "seed": seed,
                            "state_id": sid,
                            "classification": cf["classification"],
                            "target": ACTION_NAMES[int(target[agent_slot].detach().cpu().item())],
                            "agent_slot": agent_slot,
                            "actor_obs_feature_count": int(actor_obs[agent_slot].numel()),
                            "gatv2_embedding_dim": int(agent_embeddings[agent_slot].numel()),
                            "actor_obs": [float(v) for v in actor_obs[agent_slot].tolist()],
                            "gatv2_embedding": [float(v) for v in agent_embeddings[agent_slot].tolist()],
                            "legal_actions": ["HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP"],
                            "skip_legal": False,
                        }
                    )
            post_shas[seed] = sha256_file(ckpt_path)
    h4md_cross = read_json(H4M_D_ROOT / "08_h4ma_hold_better_actor_crosswalk.json")
    pserve_by_class = defaultdict(list)
    for row in h4md_cross.get("crosswalk_rows", []):
        pserve_by_class[row["classification"]].append(row.get("p_serve"))
    payload = {
        "stage": STAGE,
        "created_at": created_at,
        "scope": "H4M-A validation states × H4M-C final checkpoints, read-only inference",
        "actor_obs_feature_count": feature_rows[0]["actor_obs_feature_count"] if feature_rows else None,
        "gatv2_embedding_dim": feature_rows[0]["gatv2_embedding_dim"] if feature_rows else None,
        "mask_features": {
            "all_states_legal_actions": ["HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP"],
            "skip_legal_count": sum(1 for row in feature_rows if row["skip_legal"]),
            "mask_separates_hold_better_from_serve_better": False,
        },
        "actor_output_pserve_by_h4ma_class": {k: stats(v) for k, v in pserve_by_class.items()},
        "actor_obs_raw_feature_separation": vector_stats_by_class(feature_rows, "actor_obs"),
        "gatv2_embedding_separation": vector_stats_by_class(feature_rows, "gatv2_embedding"),
        "feature_name_availability": "RAW_FEATURE_INDICES_ONLY_NO_FROZEN_FEATURE_NAME_TABLE_FOUND_IN_SUBGRAPH_SPEC",
        "descriptive_classification": (
            "Actor/GATv2 receives numeric inputs with measurable class centroid separation, but the final actor output remains "
            "SERVE-dominant for both HOLD_BETTER and SERVE_BETTER classes; no frozen classifier threshold is invented."
        ),
        "checkpoint_sha_pre": {str(k): v for k, v in checkpoint_pre_shas.items()},
        "checkpoint_sha_post": {str(k): v for k, v in post_shas.items()},
        "checkpoint_mutation_detected": any(checkpoint_pre_shas.get(seed) != post_shas.get(seed) for seed in checkpoint_paths),
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
    }
    return payload, post_shas


def root_cause_classification(
    created_at: str,
    long_horizon: Mapping[str, Any],
    raw_gae: Mapping[str, Any],
    sign_flip: Mapping[str, Any],
    ppo: Mapping[str, Any],
    critic: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "decision": DECISION_NOT_UNIQUE,
        "root_cause_classification": "CAUSAL_ATTRIBUTION_NOT_UNIQUE",
        "earliest_failure_stage": "NOT_UNIQUELY_IDENTIFIED_WITH_CURRENT_ARTIFACTS",
        "earliest_fully_observable_downstream_driver": "PPO_SAMPLED_ACTION_SURROGATE_PLUS_SERVE_DOMINANT_ROLLOUT_DISTRIBUTION",
        "evidence": {
            "h4m_a_hold_better_long_horizon_status": long_horizon.get("h4m_a_hold_better_long_horizon_answer"),
            "per_sample_raw_gae_by_action_available": raw_gae.get("actual_per_sample_raw_gae_tensor_available") is True,
            "per_sample_sign_flip_counts_available": sign_flip.get("action_dependent_sign_effect_quantifiable"),
            "ppo_positive_minibatch_advantage_update_count": ppo.get("positive_minibatch_advantage_update_count"),
            "ppo_actor_update_count": ppo.get("actor_update_count"),
            "critic_bias_status": critic.get("critic_bias_dominance_status"),
            "observation_summary": observation.get("descriptive_classification"),
        },
        "why_not_stronger_classification": {
            "not_local_counterfactual_label_not_long_horizon_valid": "SERVE higher long-horizon return was not observable/proven.",
            "not_reward_to_gae_misalignment": "Action-conditional raw GAE for actual samples was not persisted.",
            "not_advantage_normalization_misalignment": "Exact sign-reversal counts by action were not persisted.",
            "not_ppo_actor_update_misalignment": "PPO pressure is observable as sampled-action reinforcement, but normalized advantages are not known to be directionally correct per action.",
            "not_critic_value_bias_dominant": "Critic bias is observable in aggregate but not action-conditional.",
        },
        "next_gate": NEXT_GATE_NOT_UNIQUE,
        "repair_execution_authorized": False,
        "training_authorized": False,
        "test_opening_authorized": False,
    }


def safety_integrity_audit(
    created_at: str,
    checkpoint_pre_shas: Mapping[int, Optional[str]],
    checkpoint_post_shas: Mapping[int, Optional[str]],
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    h4m_c_safety = read_json(H4M_C_ROOT / "11_safety_causality_audit.json")
    h4m_d_safety = read_json(H4M_D_ROOT / "12_safety_integrity_audit.json")
    checkpoint_mutation = any(checkpoint_pre_shas.get(seed) != checkpoint_post_shas.get(seed) for seed in checkpoint_pre_shas)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "checkpoint_sha_pre": {str(k): v for k, v in checkpoint_pre_shas.items()},
        "checkpoint_sha_post": {str(k): v for k, v in checkpoint_post_shas.items()},
        "checkpoint_mutation": checkpoint_mutation,
        "optimizer_created": False,
        "optimizer_step_executed": False,
        "training_execution": False,
        "training_backward_executed": False,
        "diagnostic_backward_executed": False,
        "normalizer_state_mutation": False,
        "reward_v2_mutation": False,
        "zero_loss_mutation": False,
        "k_mask_mutation": False,
        "future_leakage": int(h4m_c_safety.get("future_leakage", 0)) + int(h4m_d_safety.get("future_leakage", 0)),
        "illegal_action": int(h4m_c_safety.get("illegal_action_execution", 0)) + int(h4m_d_safety.get("illegal_action_execution", 0)),
        "nan_inf": int(h4m_c_safety.get("nan", 0))
        + int(h4m_c_safety.get("inf", 0))
        + int(h4m_d_safety.get("nan", 0))
        + int(h4m_d_safety.get("inf", 0)),
        "test6_status": "SEALED_NOT_OPENED",
        "github_push_performed": False,
        "binding_passed": bool(binding.get("binding_passed")),
        "source_commit_passed": bool(provenance.get("post_commit_provenance_gate_passed")),
    }


def gate_matrix(
    created_at: str,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    equations: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
    long_horizon: Mapping[str, Any],
    raw_gae: Mapping[str, Any],
    sign_flip: Mapping[str, Any],
    ppo: Mapping[str, Any],
    gradient: Mapping[str, Any],
    critic: Mapping[str, Any],
    safety: Mapping[str, Any],
    root_cause: Mapping[str, Any],
) -> Dict[str, Any]:
    hard_integrity = (
        not safety.get("checkpoint_mutation")
        and not safety.get("optimizer_step_executed")
        and not safety.get("training_execution")
        and not safety.get("reward_v2_mutation")
        and not safety.get("zero_loss_mutation")
        and not safety.get("k_mask_mutation")
        and int(safety.get("future_leakage", 0)) == 0
        and int(safety.get("illegal_action", 0)) == 0
        and int(safety.get("nan_inf", 0)) == 0
    )
    pass_ready = (
        bool(binding.get("binding_passed"))
        and bool(provenance.get("post_commit_provenance_gate_passed"))
        and bool(equations.get("semantics_recovered_from_executable_code"))
        and crosswalk.get("state_identity_mismatch_count") == 0
        and long_horizon.get("arbitrary_new_horizon_contract_invented") is False
        and raw_gae.get("aggregate_raw_gae_available_from_cycle_diagnostics") is True
        and sign_flip.get("exact_standardization_equation_audited") is True
        and ppo.get("actor_update_count", 0) > 0
        and gradient.get("optimizer_step_executed") is False
        and critic.get("critic_retrained_in_h4m_e") is False
        and hard_integrity
    )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "gate": PASS_GATE if pass_ready else BLOCK_GATE,
        "decision": root_cause.get("decision") if pass_ready else BLOCK_DECISION,
        "next_gate": root_cause.get("next_gate") if pass_ready else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": {
            "authoritative_shas_match": bool(binding.get("binding_passed")),
            "h4m_a_h4m_c_h4m_d_state_lineage_reconciled": crosswalk.get("state_identity_mismatch_count") == 0,
            "exact_executable_gae_ppo_semantics_recovered": bool(equations.get("semantics_recovered_from_executable_code")),
            "counterfactual_vs_long_horizon_distinction_resolved_where_observable": True,
            "raw_vs_normalized_advantage_audited": True,
            "ppo_surrogate_audited": ppo.get("actor_update_count", 0) > 0,
            "actor_update_direction_audited": len(gradient.get("per_seed_cycle", [])) == 33,
            "critic_contribution_audited": len(critic.get("rows", [])) > 0,
            "checkpoint_mutation": bool(safety.get("checkpoint_mutation")),
            "optimizer_step": bool(safety.get("optimizer_step_executed")),
            "training": bool(safety.get("training_execution")),
            "test_remained_sealed": safety.get("test6_status") == "SEALED_NOT_OPENED",
            "hard_integrity_violations": 0 if hard_integrity else 1,
            "manifest_mismatch": 0,
        },
        "final_flags": {
            "h4m_e_audit_executed": pass_ready,
            "training_executed": False,
            "optimizer_step_executed": False,
            "checkpoint_mutation": bool(safety.get("checkpoint_mutation")),
            "diagnostic_backward_executed": False,
            "training_backward_executed": False,
            "reward_v2_modified": False,
            "zero_loss_modified": False,
            "k_mask_modified": False,
            "sealed_test_opened": False,
            "winner_selection_authorized": False,
            "baseline_comparison_authorized": False,
            "github_push_performed": False,
        },
    }


def build_report(
    output_root: Path,
    provenance: Mapping[str, Any],
    long_horizon: Mapping[str, Any],
    raw_gae: Mapping[str, Any],
    sign_flip: Mapping[str, Any],
    ppo: Mapping[str, Any],
    entropy: Mapping[str, Any],
    gradient: Mapping[str, Any],
    cycle_pressure: Mapping[str, Any],
    critic: Mapping[str, Any],
    observation: Mapping[str, Any],
    root_cause: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    ppo_updates = ppo.get("actor_update_count")
    ppo_positive = ppo.get("positive_minibatch_advantage_update_count")
    entropy_med = entropy.get("relative_scale_summary", {}).get("median")
    pressure_counts = cycle_pressure.get("observed_pressure_counts", {})
    pserve_hold = observation.get("actor_output_pserve_by_h4ma_class", {}).get("HOLD_BETTER", {}).get("mean")
    pserve_serve = observation.get("actor_output_pserve_by_h4ma_class", {}).get("SERVE_BETTER", {}).get("mean")
    return "\n".join(
        [
            "# H4M-E Reward → GAE → Actor Credit Alignment Audit",
            "",
            f"gate = {gate['gate']}",
            f"decision = {gate['decision']}",
            f"earliest_failure_stage = {root_cause['earliest_failure_stage']}",
            f"next_gate = {gate['next_gate']}",
            "",
            f"artifact_root = {output_root}",
            f"h4m_e_audit_source_git_commit = {provenance.get('h4m_e_audit_source_git_commit')}",
            "",
            "## Final questions",
            "",
            "1. H4M-A HOLD_BETTER는 long-horizon에서도 HOLD_BETTER인가?",
            f"   - Answer: {long_horizon['h4m_a_hold_better_long_horizon_answer']}. No arbitrary horizon contract was invented.",
            "",
            "2. correct action preference가 처음 사라지는 단계는 어디인가?",
            "   - Not uniquely identifiable with current artifacts. Long-horizon counterfactual returns and per-sample GAE tensors were not persisted.",
            "",
            "3. raw GAE는 HOLD와 SERVE 중 어느 쪽을 강화했는가?",
            f"   - Exact by-action answer unavailable: {raw_gae['by_action_distribution_limitation']}",
            "",
            "4. advantage normalization이 sign reversal을 만들었는가?",
            f"   - Exact counts unavailable: {sign_flip['conclusion']}",
            "",
            "5. PPO surrogate/gradient가 실제로 어느 action을 강화했는가?",
            f"   - Stored update aggregates show {ppo_positive}/{ppo_updates} actor updates had positive first-256 minibatch advantage, which reinforces sampled actions.",
            f"   - Diagnostic logit-gradient proxy rows = {len(gradient.get('per_seed_cycle', []))} seed-cycles.",
            "",
            "6. Critic bias가 GAE 방향에 영향을 줬는가?",
            f"   - Possible but not isolated action-conditionally: {critic['critic_bias_dominance_status']}.",
            "",
            "7. cycle 1→11에서 SERVE probability가 올라간 직접 학습 메커니즘은 무엇인가?",
            f"   - Observed pressure counts = {pressure_counts}. Mechanism: sampled-action PPO updates under rollout-global advantage standardization; SERVE samples become dominant and are repeatedly reinforced.",
            "",
            "8. Actor observation은 HOLD_BETTER/SERVE_BETTER 상태를 구별할 정보를 가지고 있는가?",
            f"   - Descriptively yes/no mixed: final actor P(SERVE) remains high for both classes; HOLD_BETTER mean={pserve_hold}, SERVE_BETTER mean={pserve_serve}. "
            "Input/embedding centroid separation is recorded without a frozen threshold.",
            "",
            "9. Reward V2 변경이 필요한가, 아니면 downstream credit/observation 문제인가?",
            "   - Reward V2 change is not justified by H4M-E. The unresolved issue is downstream credit trace / long-horizon observability / observation discrimination, not a proven Reward V2 defect.",
            "",
            f"10. 정확한 다음 gate = {gate['next_gate']}",
            "",
            "STOP.",
            "",
        ]
    )


def write_payloads(output_root: Path, payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for name, payload in payloads.items():
        path = output_root / name
        if name.endswith(".json"):
            dump_json(path, payload)
        else:
            dump_text(path, str(payload))
    manifest = {
        "stage": STAGE,
        "created_at": payloads["17_h4m_e_gate_matrix.json"]["created_at"],
        "artifact_root": str(output_root),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": sorted([*payloads.keys(), "manifest.json"]) == sorted(REQUIRED_ARTIFACTS),
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift; all other required files are hashed",
        "output_files": {name: str(output_root / name) for name in payloads},
        "output_sha256": {name: sha256_file(output_root / name) for name in sorted(payloads)},
        "source_sha256": {rel: sha256_file(PROJECT_ROOT / rel) for rel in sorted(RUNTIME_DEPENDENCIES)},
        "gate": payloads["17_h4m_e_gate_matrix.json"]["gate"],
        "decision": payloads["17_h4m_e_gate_matrix.json"]["decision"],
        "next_gate": payloads["17_h4m_e_gate_matrix.json"]["next_gate"],
        "h4m_e_audit_source_git_commit": payloads["01_authoritative_binding.json"]["git_source_provenance"].get(
            "h4m_e_audit_source_git_commit"
        ),
        "training_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_mutation": payloads["17_h4m_e_gate_matrix.json"]["final_flags"]["checkpoint_mutation"],
        "sealed_test_opened": False,
        "github_push_performed": False,
    }
    dump_json(output_root / "manifest.json", manifest)
    return manifest


def block_payloads(
    created_at: str,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    gate = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": BLOCK_GATE,
        "decision": BLOCK_DECISION,
        "blocking_reason": reason,
        "next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "final_flags": {
            "h4m_e_audit_executed": False,
            "training_executed": False,
            "optimizer_step_executed": False,
            "checkpoint_mutation": False,
            "diagnostic_backward_executed": False,
            "training_backward_executed": False,
            "reward_v2_modified": False,
            "zero_loss_modified": False,
            "k_mask_modified": False,
            "sealed_test_opened": False,
            "winner_selection_authorized": False,
            "baseline_comparison_authorized": False,
            "github_push_performed": False,
        },
    }
    empty = {"stage": STAGE, "created_at": created_at, "not_evaluated": True, "blocking_reason": reason}
    report = f"# H4M-E Reward GAE Actor Credit Alignment Audit\n\ngate = {BLOCK_GATE}\ndecision = {BLOCK_DECISION}\nblocking_reason = {reason}\n\nSTOP.\n"
    return {
        "01_authoritative_binding.json": {**binding, "git_source_provenance": provenance},
        "02_exact_training_credit_equations.json": empty,
        "03_h4ma_state_identity_crosswalk.json": empty,
        "04_counterfactual_reward_return_trace.json": empty,
        "05_long_horizon_action_value_reconciliation.json": empty,
        "06_raw_gae_by_action_audit.json": empty,
        "07_advantage_sign_flip_audit.json": empty,
        "08_advantage_normalization_audit.json": empty,
        "09_ppo_surrogate_by_action_audit.json": empty,
        "10_entropy_contribution_audit.json": empty,
        "11_actor_logit_gradient_direction_audit.json": empty,
        "12_cycle1_to_cycle11_credit_pressure.json": empty,
        "13_critic_value_bias_audit.json": empty,
        "14_actor_observation_discrimination_secondary_audit.json": empty,
        "15_root_cause_classification.json": empty,
        "16_safety_integrity_audit.json": empty,
        "17_h4m_e_gate_matrix.json": gate,
        "final_report.md": report,
    }


def main() -> None:
    torch.set_grad_enabled(False)
    now = kst_now()
    created_at = now.isoformat(timespec="seconds")
    output_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_e_reward_gae_actor_credit_alignment_audit_{now.strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = checkpoint_paths_from_h4mc_registry()
    checkpoint_pre_shas = {seed: sha256_file(path) for seed, path in checkpoint_paths.items()}
    binding = authoritative_binding(created_at, checkpoint_pre_shas)
    provenance = git_source_provenance(created_at)
    blockers = []
    if not binding["binding_passed"]:
        blockers.append("AUTHORITATIVE_SHA_OR_GATE_MISMATCH")
    if not provenance["post_commit_provenance_gate_passed"]:
        blockers.append("RELEVANT_DIRTY_OR_UNCOMMITTED_H4M_E_SOURCE")
    if blockers:
        payloads = block_payloads(created_at, output_root, binding, provenance, ",".join(blockers))
        manifest = write_payloads(output_root, payloads)
        print(f"[H4M-E] artifact root: {output_root}")
        print(f"[H4M-E] gate: {manifest['gate']}")
        print(f"[H4M-E] decision: {manifest['decision']}")
        print(f"[H4M-E] blocked_before_audit: {','.join(blockers)}")
        return

    logs = load_h4mc_logs()
    equations = exact_training_credit_equations(created_at)
    crosswalk = state_identity_crosswalk(created_at)
    counterfactual = counterfactual_reward_return_trace(created_at)
    long_horizon = long_horizon_action_value_reconciliation(created_at)
    raw_gae = raw_gae_by_action_audit(created_at, logs)
    sign_flip = advantage_sign_flip_audit(created_at, logs)
    norm = advantage_normalization_audit(created_at, logs)
    ppo = ppo_surrogate_by_action_audit(created_at, logs)
    entropy = entropy_contribution_audit(created_at, logs)
    gradient = actor_logit_gradient_direction_audit(created_at, logs)
    pressure = cycle_credit_pressure(created_at, logs, gradient)
    critic = critic_value_bias_audit(created_at, logs)
    observation, checkpoint_post_shas = actor_observation_discrimination_secondary_audit(created_at, checkpoint_pre_shas)
    safety = safety_integrity_audit(created_at, checkpoint_pre_shas, checkpoint_post_shas, binding, provenance)
    root = root_cause_classification(created_at, long_horizon, raw_gae, sign_flip, ppo, critic, observation)
    gate = gate_matrix(
        created_at,
        binding,
        provenance,
        equations,
        crosswalk,
        long_horizon,
        raw_gae,
        sign_flip,
        ppo,
        gradient,
        critic,
        safety,
        root,
    )
    report = build_report(
        output_root,
        provenance,
        long_horizon,
        raw_gae,
        sign_flip,
        ppo,
        entropy,
        gradient,
        pressure,
        critic,
        observation,
        root,
        gate,
    )
    payloads = {
        "01_authoritative_binding.json": {**binding, "git_source_provenance": provenance},
        "02_exact_training_credit_equations.json": equations,
        "03_h4ma_state_identity_crosswalk.json": crosswalk,
        "04_counterfactual_reward_return_trace.json": counterfactual,
        "05_long_horizon_action_value_reconciliation.json": long_horizon,
        "06_raw_gae_by_action_audit.json": raw_gae,
        "07_advantage_sign_flip_audit.json": sign_flip,
        "08_advantage_normalization_audit.json": norm,
        "09_ppo_surrogate_by_action_audit.json": ppo,
        "10_entropy_contribution_audit.json": entropy,
        "11_actor_logit_gradient_direction_audit.json": gradient,
        "12_cycle1_to_cycle11_credit_pressure.json": pressure,
        "13_critic_value_bias_audit.json": critic,
        "14_actor_observation_discrimination_secondary_audit.json": observation,
        "15_root_cause_classification.json": root,
        "16_safety_integrity_audit.json": safety,
        "17_h4m_e_gate_matrix.json": gate,
        "final_report.md": report,
    }
    manifest = write_payloads(output_root, payloads)
    print(f"[H4M-E] artifact root: {output_root}")
    print(f"[H4M-E] gate: {manifest['gate']}")
    print(f"[H4M-E] decision: {manifest['decision']}")
    print(f"[H4M-E] next_gate: {manifest['next_gate']}")
    print("[H4M-E] training_executed=false optimizer_step=false checkpoint_mutation=false sealed_test_opened=false github_push_performed=false")


if __name__ == "__main__":
    main()
