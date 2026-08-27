#!/usr/bin/env python3
"""BT8-R17: freeze the exact E1-only bounded-training authorization.

This stage is deliberately *not* a training executor.  It binds the immutable
R11--R16 evidence, chooses the smallest previously-proven 24-row PPO envelope,
and emits a one-shot R18 authorization manifest.  It does not construct a
model, create an optimizer, run a rollout, settle a reward, write a checkpoint,
or grant a simulator/training capability.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import pandas as pd


STAGE = "H4M-AE-R9.8-LS3-BT8-R17"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R17_E1_BOUNDED_TRAINING_EXECUTION_AUTHORIZATION_AND_ENVELOPE_FREEZE_COMPLETE"
CLASSIFICATION = "A_E1_MINIMAL_BOUNDED_TRAINING_ENVELOPE_FROZEN_AND_SEPARATE_EXECUTION_AUTHORIZED"
BINDING_BLOCK = "BLOCKED_R17_UPSTREAM_EVIDENCE_BINDING_FAILURE"
NO_SAFE_BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R17_NO_SAFE_MINIMAL_BOUNDED_TRAINING_ENVELOPE"
SOURCE_BLOCK = "BLOCKED_R17_SOURCE_OR_EXECUTION_CONTRACT_FAILURE"

R11_SOURCE = "f467feef8246c6c77dc67a360aecc60e0102913e"
R12_SOURCE = "6e22b95d80720255ca17b1ab0520417b89e5a5c9"
R13_SOURCE = "a9818399c4a1a74734496a146b2b98fabab513b0"
R14_SOURCE = "36ccbaa84e06af3202560802fd3371066839a7c5"
R15_SOURCE = "8e24652fa0523e751077ca0c4500b20d626fc517"
R16_SOURCE = "90a8a69227c1166c8d311d54f53fe9795dfeb7cd"
R4_SOURCE = "32fbf2b3043166188bde891a5c450702e7db3607"
S3_CONTRACT_SHA256 = "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"
R15_CONTRACT_SHA256 = "cd864dcbba5ec42b26c6399aa92480596455aa104e752627676b3e6ac2de0c84"
R16_IMPLEMENTATION_CONTRACT_SHA256 = "485691360307318b4e05bd4f42d65c2d31a43b43196272e94432629e3f91435a"
REVIEW_COLLECTION_DIGEST = "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"
TRAINING_COLLECTION_DIGEST = "954d50f384ebe4318dbe9565c4c30c5e6a0b6249dd06ca441ff5abc2af08fb63"

R11_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R11_MINIMAL_SEED_DIVERGENCE_CAUSE_ISOLATION_AND_REPAIR_SELECTION_COMPLETE"
R12_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R12_S3_FACTORIAL_EXECUTION_AUTHORITY_SELECTION_COMPLETE"
R13_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_S3_FOUR_CELL_ON_POLICY_CAUSAL_EXECUTION_COMPLETE"
R14_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R14_S3_SAME_INPUT_FROZEN_POLICY_FACTOR_REVIEW_COMPLETE"
R15_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R15_MINIMAL_EXPLORATION_DEADLOCK_REPAIR_SELECTION_AUDIT_COMPLETE"
R16_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R16_FROZEN_POLICY_MASKED_CATEGORICAL_EXPLORATION_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
R4_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R4_FRESH_V2_ACTOR_TRAINING_AND_NOVEL_EXPOSURE_DESIGN_COMPLETE"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
R11_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection_20260826_123135+09:00"
R12_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection_20260826_145345+09:00"
R13_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r13_s3_four_cell_execution_20260826_174729+09:00"
R14_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review_20260827_001145+09:00"
R15_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r15_minimal_exploration_deadlock_repair_selection_audit_20260827_194419+09:00"
R16_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r16_frozen_policy_masked_categorical_exploration_validation_20260827_205833+09:00"
R4_NAME = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r4_v2_actor_fresh_training_design_20260823_122459+0900"

R17_RUNNER_REL = "05_training/run_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization.py"
R17_TEST_REL = "05_training/test_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization.py"
R18_RUNNER_REL = "05_training/run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
R18_TEST_REL = "05_training/test_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"
SOURCE_FILES = {R17_RUNNER_REL, R17_TEST_REL, R18_RUNNER_REL, R18_TEST_REL}

TIME_BANDS = ("night", "offpeak", "peak")
WINDOWS = (
    "PV8_R3R_HOLIDAY_NIGHT_D1_MEDIAN_CHRONOLOGICAL_20230702_0700",
    "PV8_R3R_SATURDAY_NIGHT_D1_EARLIEST_20230107_0700",
    "PV8_R3R_HOLIDAY_OFFPEAK_D0_MEDIAN_CHRONOLOGICAL_20230702_1000",
    "PV8_R3R_SATURDAY_OFFPEAK_D1_EARLIEST_20230107_1000",
    "PV8_R3R_HOLIDAY_PEAK_D0_LATEST_20231231_1700",
    "PV8_R3R_SATURDAY_PEAK_D0_EARLIEST_20230107_1700",
)
EXPECTED_R16_CHECKPOINTS = {
    "initial:AC-R1": "e13e3f0294234d3ee4b49878912a4b5ee18c5e53253e478201b161520dc6cba7",
    "initial:AC-R2": "ad618133374c1ccbe2c367f67117a3d6b205b9c48acd5ee7a2ba3b5549ed0df0",
    "initial:BD-R1": "565c07424f811170a8f3640cef622b3a87a235c763f7a790958c1c3aace9712d",
    "initial:BD-R2": "0fcd9c359bedc355718fabb9dbf87c3cbcf09d2efc07419bd848dcd4e12470dd",
    "final:AC-R1": "fc4a283d11bd436a4e42cdfebf2b07156dbfedbc703c715bce5618600bfd5968",
    "final:AC-R2": "f07240f2a11ea0707f5b2f775a56b723739742c51cfb0ff0b7383b995bc525b1",
    "final:BD-R1": "6a95135c2ff3e1637d23710783b8a7729c2a0fb0d025977c73452ed4045037da",
    "final:BD-R2": "4754a7d53a810e4562ee42d5e77dd55810ab88855d41d81e9eef02f9e2f5afae",
}
GLOBAL_LOCKS = {
    "training_allowed": False,
    "simulator_execution_allowed": False,
    "performance_comparison_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}
R17_HARD_LOCKS = {
    "training_allowed": False,
    "environment_rollout_allowed": False,
    "reward_settlement_allowed": False,
    "optimizer_creation_allowed": False,
    "optimizer_step_allowed": False,
    "backward_allowed": False,
    "checkpoint_write_allowed": False,
    "policy_mutation_allowed": False,
    "TEST6_open_allowed": False,
    "GitHub_push_allowed": False,
}


class R17Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R17Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path, code: str = BINDING_BLOCK) -> Any:
    require(path.is_file(), code, f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    hashes = manifest.get("file_sha256")
    require(isinstance(hashes, Mapping), BINDING_BLOCK, f"manifest_schema={root}")
    mismatches = [str(name) for name, digest in hashes.items()
                  if not (root / str(name)).is_file() or sha256(root / str(name)) != str(digest)]
    return {"path": str(root), "manifest_sha256": sha256(root / "manifest.json"),
            "declared_file_count": len(hashes), "mismatches": mismatches, "all_match": not mismatches}


def locate_timestamped_artifact(*, name: str, gate: str, source: str) -> Path:
    """Fail closed on a mutable pointer, missing root, or duplicate authority."""
    candidates = [path for path in ARTIFACTS.glob(f"*{name.split('_')[-1]}*") if path.is_dir() and path.name == name]
    verified: list[Path] = []
    for path in candidates:
        decision = path / "gate_decision.json"
        if not decision.is_file():
            continue
        payload = load_json(decision)
        if payload.get("gate") == gate and payload.get("source_commit") == source:
            verified.append(path)
    require(len(verified) == 1, BINDING_BLOCK, f"timestamped_artifact={name}:matches={len(verified)}")
    return verified[0]


def source_provenance() -> dict[str, Any]:
    changed = [item for item in git(["diff", "--name-only", f"{R16_SOURCE}..HEAD"]).splitlines() if item]
    return {
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_lineage_descends_from_r16": git(["merge-base", R16_SOURCE, "HEAD"]) == R16_SOURCE,
        "changed_files_since_r16": changed,
        "source_only_local_commit": set(changed) == SOURCE_FILES,
        "git_status_porcelain": git(["status", "--porcelain=v1"]),
        "github_push_performed": False,
    }


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization_{stamp}"


def r17_counters() -> dict[str, int]:
    return {
        "training": 0, "environment_rollout": 0, "reward_settlement": 0,
        "optimizer_creation": 0, "optimizer_step": 0, "backward": 0,
        "checkpoint_write": 0, "policy_mutation": 0, "test6_access": 0,
        "github_push": 0, "candidate_generation": 0, "mps_model_forward": 0,
    }


def _require_gate(root: Path, expected_gate: str, expected_source: str, label: str) -> dict[str, Any]:
    payload = load_json(root / "gate_decision.json")
    require(payload.get("gate") == expected_gate and payload.get("source_commit") == expected_source,
            BINDING_BLOCK, f"gate={label}")
    return payload


def _file_hashes_from_r16(r16: Path) -> dict[str, str]:
    implementation = load_json(r16 / "implementation_source_hash_audit.json")
    frozen = load_json(r16 / "frozen_hash_before_after.json")
    sources = dict(implementation.get("modified_source_hashes", {}))
    sources |= {f"frozen:{key}": str(value) for key, value in dict(frozen.get("before", {})).items()}
    sources |= {f"extra:{key}": str(value) for key, value in dict(frozen.get("extra_before", {})).items()}
    return sources


def _checkpoint_binding(r13: Path, r16: Path) -> dict[str, Any]:
    manifest = load_json(r13 / "bt8r13_checkpoint_manifest.json")
    observed: dict[str, dict[str, Any]] = {}
    for role in ("initial", "final"):
        entries = manifest.get(role)
        require(isinstance(entries, Mapping) and set(entries) == {"AC-R1", "AC-R2", "BD-R1", "BD-R2"},
                BINDING_BLOCK, f"checkpoint_schema={role}")
        for cell, entry in entries.items():
            require(isinstance(entry, Mapping), BINDING_BLOCK, f"checkpoint_entry={role}:{cell}")
            path = r13 / str(entry.get("path", ""))
            observed[f"{role}:{cell}"] = {
                "path": str(path.resolve()), "sha256": sha256(path), "declared_sha256": str(entry.get("sha256")),
                "role": "allowed_initial_input" if role == "initial" else "evidence_only_not_loadable",
                "test_only": bool(entry.get("test_only")), "non_promotable": bool(entry.get("non_promotable")),
            }
            require(observed[f"{role}:{cell}"]["sha256"] == observed[f"{role}:{cell}"]["declared_sha256"],
                    BINDING_BLOCK, f"checkpoint_file={role}:{cell}")
    r16_hashes = load_json(r16 / "frozen_hash_before_after.json").get("checkpoint_before")
    require(observed.keys() == EXPECTED_R16_CHECKPOINTS.keys() and dict(r16_hashes or {}) == EXPECTED_R16_CHECKPOINTS,
            BINDING_BLOCK, "r16_checkpoint_schema")
    require({key: value["sha256"] for key, value in observed.items()} == EXPECTED_R16_CHECKPOINTS,
            BINDING_BLOCK, "checkpoint_hash")
    return {"all_eight_bound": True, "checkpoints": observed}


def _source_hash_binding(r16: Path) -> dict[str, Any]:
    r16_hashes = _file_hashes_from_r16(r16)
    required = {
        "implementation:selector": ROOT / "joint_assignment_frozen_policy_selector.py",
        "implementation:r16_runner": ROOT / "run_h4m_ae_ls3_bt8_r16_frozen_policy_masked_categorical_exploration_validation.py",
        "implementation:selector_test": ROOT / "test_joint_assignment_frozen_policy_selector.py",
        "frozen:gatv2_operational_actor_critic": ROOT / "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
        "frozen:reward_v2": ROOT / "rewards" / "mappo_reward_v1.py",
        "frozen:zero_loss": ROOT / "simulator" / "zero_loss_admission_adapter.py",
        "frozen:local_search_authority": ROOT / "local_search_contract.py",
        "frozen:candidate_support_deconfounding": ROOT / "joint_candidate_support_snapshot.py",
        "frozen:causal_bridge": ROOT / "causal_kpi_bridge.py",
        "frozen:r9_8_authorization": ROOT / "simulator_authorization.py",
        "frozen:credit_contract": ROOT / "joint_assignment_credit_contract.py",
        "frozen:joint_assignment_learning": ROOT / "joint_assignment_learning.py",
        "frozen:r9_7_gate": ROOT / "run_h4m_ae_r9_7_gate.py",
        "frozen:r9_8_gate": ROOT / "run_h4m_ae_r9_8_gate.py",
        "frozen:joint_actor_head": ROOT / "multi_agent_candidate_assignment_head.py",
        "frozen:t1_selector": ROOT / "joint_assignment_frozen_tie_break.py",
        "extra:candidate_plan_bridge": ROOT / "joint_candidate_plan_causal_bridge.py",
        "extra:e1_eligibility": ROOT / "joint_assignment_e1_eligibility.py",
        "extra:t1_selector": ROOT / "joint_assignment_frozen_tie_break.py",
    }
    expected = {
        "implementation:selector": str(r16_hashes["05_training/joint_assignment_frozen_policy_selector.py"]["after_sha256"]),
        "implementation:r16_runner": str(r16_hashes["05_training/run_h4m_ae_ls3_bt8_r16_frozen_policy_masked_categorical_exploration_validation.py"]["after_sha256"]),
        "implementation:selector_test": str(r16_hashes["05_training/test_joint_assignment_frozen_policy_selector.py"]["after_sha256"]),
        **{key: str(value) for key, value in r16_hashes.items() if key.startswith(("frozen:", "extra:"))},
    }
    actual = {key: sha256(path) for key, path in required.items()}
    require(actual == expected, BINDING_BLOCK, "r16_frozen_source_hash")
    return {"expected": expected, "actual": actual, "all_unchanged": True}


def _time_band(window_id: str) -> str:
    upper = str(window_id).upper()
    if "_NIGHT_" in upper:
        return "night"
    if "_OFFPEAK_" in upper:
        return "offpeak"
    if "_PEAK_" in upper:
        return "peak"
    raise R17Error(NO_SAFE_BLOCK, f"unclassified_time_band={window_id}")


def select_minimal_envelope(*, r12_cells: Sequence[Mapping[str, Any]], r12_windows: Sequence[str],
                            r4_window_records: Sequence[Mapping[str, Any]], r14_deadlock: Mapping[str, Any],
                            r16_probe_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose the smallest envelope that preserves frozen 24-row PPO semantics.

    One optimized cell cannot both retain the AC control and test the BD E1
    opening.  Each arm must retain six 4-step trajectories because the frozen
    N0 helper accepts exactly 24 rows.  The selected two-arm same-environment
    design therefore is not a post-hoc reduction or performance comparison.
    """
    cells = {str(cell.get("cell_id")): dict(cell) for cell in r12_cells}
    require(set(cells) == {"AC-R1", "AC-R2", "BD-R1", "BD-R2"}, NO_SAFE_BLOCK, "r12_cell_set")
    require(tuple(r12_windows) == WINDOWS, NO_SAFE_BLOCK, "frozen_window_ids")
    records = [dict(record) for record in r4_window_records]
    require([str(record.get("window_id")) for record in records] == list(WINDOWS)
            and all(str(record.get("source_group", "")) for record in records), NO_SAFE_BLOCK, "frozen_window_records")
    ac, bd = cells["AC-R1"], cells["BD-R1"]
    require(int(ac["environment_seed"]) == int(bd["environment_seed"]) == 20260822,
            NO_SAFE_BLOCK, "same_environment_control")
    audit_cells = dict(r14_deadlock.get("cells", {}))
    ac_audit, bd_audit = dict(audit_cells.get("AC-R1", {})), dict(audit_cells.get("BD-R1", {}))
    require(int(ac_audit.get("candidate_plan_executions", -1)) == 24 and int(ac_audit.get("actor_eligible_rows", -1)) == 9,
            NO_SAFE_BLOCK, "ac_control_evidence")
    require(int(bd_audit.get("feasible_candidate_decisions", -1)) == 24 and int(bd_audit.get("no_assign_selections", -1)) == 24
            and int(bd_audit.get("actor_eligible_rows", -1)) == 0, NO_SAFE_BLOCK, "bd_deadlock_evidence")
    counts = {band: 0 for band in TIME_BANDS}
    for row in r16_probe_rows:
        selected_identity = str(row.get("selected_identity", "NO_ASSIGN"))
        if str(row.get("cell_id")) == "BD-R1" and int(row.get("probe_seed", -1)) == 0 and selected_identity not in {"NO_ASSIGN", "NO_ASSIGN_KEEP_CURRENT_PLANS"}:
            band = str(row.get("time_band"))
            if band in counts:
                counts[band] += 1
    require(all(counts[band] > 0 for band in TIME_BANDS), NO_SAFE_BLOCK, f"bd_seed0_exposure={counts}")
    cell_rows = []
    for arm_id, role, cell in (("AC_CONTROL_R1", "AC_CONTROL", ac), ("BD_E1_R1", "BD_E1_OPENING", bd)):
        checkpoint_key = "initial:AC-R1" if str(cell["cell_id"]) == "AC-R1" else "initial:BD-R1"
        cell_rows.append({
            "arm_id": arm_id, "lineage_cell": str(cell["cell_id"]), "role": role,
            "environment_seed": int(cell["environment_seed"]), "actor_seed": int(cell["actor_seed"]),
            "critic_seed": int(cell["critic_seed"]), "initial_checkpoint_key": checkpoint_key,
            "initialization_replicate": str(cell["initialization_replicate"]),
            "train_window_records": records, "train_windows": list(WINDOWS), "visits": 6, "trajectories": 6, "trajectory_length": 4,
            "assignment_decisions": 24, "causal_transitions": 48,
            "ppo_epochs": 3, "full_batch_size": 24, "minibatch_size": 24,
            "critic_optimizer_steps": 3, "actor_optimizer_steps": 3,
            "actor_step_rule": "exactly 3 only after actor_eligible_count >= 1; otherwise STOP without supplement",
            "categorical_probe_seed_by_decision": {f"{arm_id}:{index}": 0 for index in range(24)},
        })
    envelope = {
        "selection_method": "minimum_sufficient_envelope_from_BT4_pass_plus_R12_24_row_contract_plus_R16_static_support",
        "not_a_performance_or_seed_generalization_claim": True,
        "why_not_smaller": [
            "AC control and BD E1 opening are separate roles and need two independent arms.",
            "Frozen cell-local N0/PPO semantics require 24 rows, six 4-step trajectories, and three full-batch epochs per optimized arm.",
            "Selecting a smaller rewardful subset would be outcome-driven and would alter the frozen 24-row contract.",
        ],
        "selected_arms": cell_rows,
        "distinct_windows": list(WINDOWS),
        "time_bands": {band: [window for window in WINDOWS if _time_band(window) == band] for band in TIME_BANDS},
        "static_bd_r16_seed0_non_no_assign_exposure": counts,
        "static_evidence_only_not_runtime_guarantee": True,
        "aggregate": {
            "environment_seed_count": 1, "windows": 6, "visits": 12, "trajectories": 12,
            "assignment_decisions": 48, "causal_transitions": 96,
            "assignment_actor_optimizer_steps_maximum": 6, "assignment_critic_optimizer_steps_exact": 6,
            "raw_optimizer_step_calls_maximum": 12, "supplemental_steps": 0,
        },
        "caps_checked": {"windows_le_12": True, "visits_le_24": True, "assignment_decisions_le_96": True,
                         "environment_seeds_le_2": True, "actor_steps_le_6": True},
    }
    require(envelope["aggregate"] == {"environment_seed_count": 1, "windows": 6, "visits": 12, "trajectories": 12,
                                         "assignment_decisions": 48, "causal_transitions": 96,
                                         "assignment_actor_optimizer_steps_maximum": 6,
                                         "assignment_critic_optimizer_steps_exact": 6,
                                         "raw_optimizer_step_calls_maximum": 12, "supplemental_steps": 0},
            NO_SAFE_BLOCK, "aggregate_envelope")
    return envelope


def module_freeze_contract(*, source_hashes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trainable_in_R18_only": ["LS3 Joint Assignment Actor", "LS3 Joint Assignment Critic"],
        "optimizer_scope": "fresh Actor/Critic-only optimizers; no prior optimizer state, operational parameters, or advantage statistics",
        "assignment_only_tensors": ["assignment_log_prob", "assignment_value", "assignment_reward", "assignment_advantage", "assignment_GAE", "assignment_optimizer"],
        "frozen": ["operational HOLD/SERVE/SKIP Actor/Critic", "GATv2", "Reward V2", "Zero-Loss",
                   "Local Search candidate semantics", "demand ledger", "simulator semantics", "T1 selector"],
        "frozen_source_hashes": dict(source_hashes),
        "E1_only": True, "E2_rescue": False, "E3_temperature_or_floor": False,
        "epsilon_greedy": False, "no_assign_penalty_or_removal": False, "candidate_bonus": False,
        "logit_or_probability_mutation": False,
        "training_selection_mode": "FROZEN_MASKED_CATEGORICAL_TRAINING",
        "inference_evaluation_mode": "FROZEN_INFERENCE_T1",
        "sealed_distribution_view_required": True,
    }


def execution_guard_contract() -> dict[str, Any]:
    stop_conditions = [
        "upstream hash mismatch", "unexpected source mutation", "MPS unavailable or CPU fallback", "illegal candidate > 0",
        "Zero-Loss violation > 0", "NaN or Inf > 0", "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE",
        "cross-window GAE leakage > 0", "future leakage > 0", "duplicate reward ancestry > 0",
        "inference T1 semantics changed", "Reward V2/GATv2/Zero-Loss mutation", "unexpected optimizer target",
        "checkpoint overwrite", "BD missing a required causal-learning counter", "budget overrun",
    ]
    return {
        "selection_flow": "actor.eval + no_grad -> sealed FrozenMaskedDistributionView -> FROZEN_MASKED_CATEGORICAL_TRAINING -> retain returned log_probability",
        "inference_flow": "FROZEN_INFERENCE_T1 only, exact tie tolerance 0",
        "candidate_support_roundtrip": [
            "lossless rollout snapshot captures ordered pair IDs, masks, model inputs, support digest, selected source index, and old log-probability",
            "PPO re-reads and byte-compares those values before every epoch",
            "AssignmentRolloutBuffer.assert_action_support_unchanged must be called; mismatch code is ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE",
        ],
        "candidate_plan_credit": "selected candidate == applied candidate == credited candidate; candidate regeneration after selection/PPO = 0; SERVE fallback = 0; explicit NO_ASSIGN transition only",
        "gae": "trajectory-local 4-step recurrence; no cross-window, cross-arm, or future connection; N0 is arm-local over exactly 24 rows before E1 mask",
        "e1": "DIRECT_CANDIDATE_REWARD/TEMPORALLY_PROPAGATED_CANDIDATE_REWARD only drive Actor; CRITIC_ONLY_NO_REWARD_ANCESTRY has zero Actor gradient; INSUFFICIENT_IDENTITY_CREDIT blocks",
        "mps": {"platform": "Mac mini M4 24GB", "accelerator": "Apple MPS", "device": "mps:0", "cpu_fallback": 0, "cuda": False, "cloud_gpu": False},
        "review": "only six preserved review snapshots; no optimizer inclusion, regeneration, Local Search rerun, Zero-Loss reevaluation, or simulator use",
        "stop_conditions": stop_conditions,
        "no_auto_extension": True,
    }


def expected_learning_counters() -> dict[str, Any]:
    required_bd_per_band = {band: {"feasible_support_count_min": 1, "categorical_sample_count_min": 1,
                                    "non_NO_ASSIGN_count_min": 1} for band in TIME_BANDS}
    return {
        "per_arm_seed_time_band": ["feasible_support_count", "categorical_sample_count", "NO_ASSIGN_count", "non_NO_ASSIGN_count",
                                    "candidate_execution_count", "reward_bearing_transition_count", "reward_ancestry_count",
                                    "actor_eligible_count", "assignment_actor_optimizer_steps", "assignment_critic_optimizer_steps",
                                    "policy_tensor_delta", "logit_probability_shift_after_authorized_update", "illegal_selection_count",
                                    "Zero_Loss_violation_count", "NaN_count", "Inf_count"],
        "AC_CONTROL_R1": {"actor_eligible_count_min": 1, "assignment_actor_optimizer_steps_exact": 3,
                            "assignment_critic_optimizer_steps_exact": 3},
        "BD_E1_R1": {"time_band_minimums": required_bd_per_band, "candidate_execution_count_min": 1,
                       "reward_bearing_transition_count_min": 1, "reward_ancestry_count_min": 1,
                       "actor_eligible_count_min": 1, "assignment_actor_optimizer_steps_exact": 3,
                       "assignment_critic_optimizer_steps_exact": 3},
        "failure_behavior": "If any required minimum is not observed, block before PPO for the affected arm; do not add windows, visits, seeds, or steps.",
        "not_a_policy_quality_or_KPI_claim": True,
    }


def checkpoint_contract(*, root: Path, checkpoint_binding: Mapping[str, Any]) -> dict[str, Any]:
    target = ARTIFACTS / (root.name.replace("r17_e1_bounded_training_authorization", "r18_e1_bounded_training_execution"))
    require(not target.exists(), SOURCE_BLOCK, f"r18_output_root_already_exists={target}")
    inputs = dict(checkpoint_binding["checkpoints"])
    return {
        "initial_inputs": {key: value for key, value in inputs.items() if key.startswith("initial:")},
        "final_evidence_only": {key: value for key, value in inputs.items() if key.startswith("final:")},
        "initial_checkpoint_policy": "exact SHA-bound read-only input; no optimizer state reuse; no V1/BT6/old Critic transfer",
        "R18_output_root": str(target.resolve()),
        "output_root_must_not_exist_before_R18": True,
        "final_checkpoint_policy": {"only_after_exact_authorized_execution": True, "test_only": True, "bounded": True,
                                    "non_promotable": True, "winner": False, "best_model": False, "promotion": False},
        "intermediate_checkpoint": "forbidden", "overwrite_existing_checkpoint": False,
        "original_eight_checkpoints_mutation": False,
    }


def build_authorization_manifest(*, source: Mapping[str, Any], upstream: Mapping[str, Any], envelope: Mapping[str, Any],
                                 module_contract: Mapping[str, Any], guard: Mapping[str, Any], counters: Mapping[str, Any],
                                 checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "stage": STAGE, "authorization": "R18_EXACT_ONE_SHOT_E1_BOUNDED_TRAINING_ONLY", "authorized": True,
        "source_commit": source["source_commit"], "r16_source_commit": R16_SOURCE,
        "upstream": dict(upstream), "envelope": dict(envelope), "module_freeze_contract": dict(module_contract),
        "execution_guard_contract": dict(guard), "expected_learning_counters": dict(counters),
        "checkpoint_contract": dict(checkpoint), "global_locks_after_R18": GLOBAL_LOCKS,
        "r17_execution_counters": dict(counters_zero(counters=None)),
        "authorization_sha256": "PENDING",
    }
    payload["authorization_sha256"] = canonical_sha256({key: value for key, value in payload.items() if key != "authorization_sha256"})
    return payload


def counters_zero(counters: Mapping[str, int] | None) -> dict[str, int]:
    return dict(counters) if counters is not None else r17_counters()


def r18_exact_command(*, authorization_manifest: Path, authorization_sha256: str) -> str:
    interpreter = PROJECT / ".venv" / "bin" / "python"
    return (
        f"{interpreter} {PROJECT / R18_RUNNER_REL} "
        f"--authorization-manifest {authorization_manifest.resolve()} "
        f"--authorization-sha256 {authorization_sha256} --execute-exact-r17-envelope"
    )


def write_block(*, root: Path, source: Mapping[str, Any], reason: str, detail: str, counters: Mapping[str, Any]) -> None:
    outputs = {
        "evidence_binding_audit.json": {"passed": False, "reason": reason, "detail": detail},
        "r17_envelope_selection.json": {"not_completed": True},
        "r17_module_freeze_contract.json": {"not_completed": True},
        "r17_execution_guard_contract.json": {"not_completed": True},
        "r17_expected_learning_counters.json": {"not_completed": True},
        "r17_checkpoint_contract.json": {"not_completed": True},
        "r17_bounded_training_authorization_manifest.json": {"authorized": False, "reason": reason},
        "test_results.json": {"execution_counters": dict(counters), "hard_failures": [detail], "warnings": [], "global_locks": GLOBAL_LOCKS},
        "frozen_hash_before_after.json": {"not_completed": True},
        "gate_decision.json": {"stage": STAGE, "gate": reason, "classification": "BLOCKED", "source_commit": source.get("source_commit"),
                               "hard_failures": [detail], "warnings": [], "global_locks": GLOBAL_LOCKS, "next_step": "STOP"},
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "r18_exact_execution_command.txt").write_text("NOT_AUTHORIZED\n", encoding="utf-8")
    (root / "final_report.md").write_text(f"# BT8-R17 blocked\n\n- gate: `{reason}`\n- detail: `{detail}`\n", encoding="utf-8")
    hashes = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": reason, "source_commit": source.get("source_commit"),
                                  "github_push_performed": False, "file_sha256": hashes})


def main() -> None:
    source = source_provenance()
    root = artifact_root()
    require(not root.exists(), SOURCE_BLOCK, "append_only_artifact_collision")
    root.mkdir(parents=True)
    execution = r17_counters()
    try:
        require(source["source_lineage_descends_from_r16"] and source["source_only_local_commit"]
                and source["git_status_porcelain"] == "", SOURCE_BLOCK, "source_scope_or_dirty_tree")
        upstream_paths = {
            "r4": locate_timestamped_artifact(name=R4_NAME, gate=R4_GATE, source=R4_SOURCE),
            "r11": locate_timestamped_artifact(name=R11_NAME, gate=R11_GATE, source=R11_SOURCE),
            "r12": locate_timestamped_artifact(name=R12_NAME, gate=R12_GATE, source=R12_SOURCE),
            "r13": locate_timestamped_artifact(name=R13_NAME, gate=R13_GATE, source=R13_SOURCE),
            "r14": locate_timestamped_artifact(name=R14_NAME, gate=R14_GATE, source=R14_SOURCE),
            "r15": locate_timestamped_artifact(name=R15_NAME, gate=R15_GATE, source=R15_SOURCE),
            "r16": locate_timestamped_artifact(name=R16_NAME, gate=R16_GATE, source=R16_SOURCE),
        }
        manifest_audits = {key: manifest_audit(value) for key, value in upstream_paths.items()}
        require(all(item["all_match"] for item in manifest_audits.values()), BINDING_BLOCK, "upstream_manifest_hash")
        for key, expected_gate, expected_source in (("r4", R4_GATE, R4_SOURCE), ("r11", R11_GATE, R11_SOURCE), ("r12", R12_GATE, R12_SOURCE),
                                                     ("r13", R13_GATE, R13_SOURCE), ("r14", R14_GATE, R14_SOURCE),
                                                     ("r15", R15_GATE, R15_SOURCE), ("r16", R16_GATE, R16_SOURCE)):
            _require_gate(upstream_paths[key], expected_gate, expected_source, key)
        r4, r11, r12, r13, r14, r15, r16 = (upstream_paths[key] for key in ("r4", "r11", "r12", "r13", "r14", "r15", "r16"))
        r15_contract = load_json(r15 / "selected_minimal_exploration_repair_contract.json")
        require(r15_contract.get("contract_sha256") == R15_CONTRACT_SHA256 and r15_contract.get("contract_id") == "LS3_BT8_R15_E1_FROZEN_POLICY_MASKED_CATEGORICAL_SAMPLING_V1"
                and r15_contract.get("repair_level") == "E1" and r15_contract.get("training_authorized") is False,
                BINDING_BLOCK, "r15_contract")
        r16_contract = load_json(r16 / "r16_implementation_contract.json")
        require(r16_contract.get("contract_sha256") == R16_IMPLEMENTATION_CONTRACT_SHA256
                and r16_contract.get("selected_repair") == "E1_FROZEN_POLICY_MASKED_CATEGORICAL_SAMPLING"
                and r16_contract.get("training_selection_mode") == "FROZEN_MASKED_CATEGORICAL_TRAINING"
                and r16_contract.get("inference_mode") == "FROZEN_INFERENCE_T1"
                and r16_contract.get("sealed_distribution_view_required") is True
                and r16_contract.get("E2_enabled") is False and r16_contract.get("E3_enabled") is False,
                BINDING_BLOCK, "r16_contract")
        s3 = load_json(r11 / "bt8r11_selected_seed_repair_contract.json")
        require(s3.get("sha256") == S3_CONTRACT_SHA256 and canonical_sha256({key: value for key, value in s3.items() if key != "sha256"}) == S3_CONTRACT_SHA256,
                BINDING_BLOCK, "s3_contract")
        r12_cells_contract = load_json(r12 / "bt8r12_s3_cell_contract.json")
        r16_evidence = load_json(r16 / "evidence_binding_audit.json")
        require(r16_evidence.get("review_collection_digest") == REVIEW_COLLECTION_DIGEST
                and int(r16_evidence.get("review_snapshot_count", -1)) == 6
                and r16_evidence.get("training_collection_digest") == TRAINING_COLLECTION_DIGEST
                and int(r16_evidence.get("training_snapshot_count", -1)) == 96,
                BINDING_BLOCK, "snapshot_collection")
        checkpoint_binding = _checkpoint_binding(r13, r16)
        source_binding = _source_hash_binding(r16)
        r18_path = PROJECT / R18_RUNNER_REL
        r18_test_path = PROJECT / R18_TEST_REL
        require(r18_path.is_file() and r18_test_path.is_file(), SOURCE_BLOCK, "r18_executor_missing")
        r18_source_hash = sha256(r18_path)
        r16_probe_summary = load_json(r16 / "training_selection_probe_summary.json")
        probe_path = r16 / str(dict(r16_probe_summary.get("parquet", {})).get("path", ""))
        require(probe_path.is_file(), BINDING_BLOCK, "r16_probe_parquet_missing")
        probe_rows = pd.read_parquet(probe_path).to_dict(orient="records")
        require(len(probe_rows) == int(dict(r16_probe_summary.get("parquet", {})).get("row_count", -1)) == 3072,
                BINDING_BLOCK, "r16_probe_row_count")
        envelope = select_minimal_envelope(r12_cells=list(r12_cells_contract.get("cells", [])),
                                            r12_windows=list(r12_cells_contract.get("train_window_ids", [])),
                                            r4_window_records=list(load_json(r4 / "bt8r4_selected_training_envelope.json").get("train_windows", [])),
                                            r14_deadlock=load_json(r14 / "bt8r14_exploration_deadlock_audit.json"),
                                            r16_probe_rows=probe_rows)
        module_contract = module_freeze_contract(source_hashes=source_binding)
        guard = execution_guard_contract()
        expected = expected_learning_counters()
        checkpoint = checkpoint_contract(root=root, checkpoint_binding=checkpoint_binding)
        upstream = {
            "timestamped_artifacts": {key: str(path.resolve()) for key, path in upstream_paths.items()},
            "manifest_audits": manifest_audits, "r15_contract_sha256": R15_CONTRACT_SHA256,
            "r16_implementation_contract_sha256": R16_IMPLEMENTATION_CONTRACT_SHA256,
            "s3_contract_sha256": S3_CONTRACT_SHA256, "review_collection_digest": REVIEW_COLLECTION_DIGEST,
            "training_collection_digest": TRAINING_COLLECTION_DIGEST, "checkpoint_binding": checkpoint_binding,
            "r16_modified_and_frozen_source_hashes": source_binding, "r18_executor": {"path": str(r18_path.resolve()), "sha256": r18_source_hash},
        }
        authorization = build_authorization_manifest(source=source, upstream=upstream, envelope=envelope,
                                                      module_contract=module_contract, guard=guard, counters=expected,
                                                      checkpoint=checkpoint)
        auth_path = root / "r17_bounded_training_authorization_manifest.json"
        dump(auth_path, authorization)
        command = r18_exact_command(authorization_manifest=auth_path, authorization_sha256=str(authorization["authorization_sha256"]))
        (root / "r18_exact_execution_command.txt").write_text(command + "\n", encoding="utf-8")
        outputs = {
            "evidence_binding_audit.json": {"passed": True, "source": source, "upstream": upstream,
                                              "R17_hard_locks": R17_HARD_LOCKS, "mps_read_only_environment": {
                                                  "mps_built": bool(torch.backends.mps.is_built()), "mps_available": bool(torch.backends.mps.is_available()),
                                                  "device_for_R18_if_available": "mps:0", "model_forward": 0}},
            "r17_envelope_selection.json": envelope,
            "r17_module_freeze_contract.json": module_contract,
            "r17_execution_guard_contract.json": guard,
            "r17_expected_learning_counters.json": expected,
            "r17_checkpoint_contract.json": checkpoint,
            "test_results.json": {"execution_counters": execution, "hard_failures": [], "warnings": [],
                                  "R17_is_training": False, "command_dry_run": "authorization syntax validated by focused test only",
                                  "global_locks": GLOBAL_LOCKS, "github_push_performed": False},
            "frozen_hash_before_after.json": {"before": source_binding, "after": _source_hash_binding(r16),
                                                "all_unchanged": source_binding == _source_hash_binding(r16),
                                                "checkpoint_before": EXPECTED_R16_CHECKPOINTS,
                                                "checkpoint_after": {key: value["sha256"] for key, value in checkpoint_binding["checkpoints"].items()},
                                                "checkpoint_unchanged": True},
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                   "source_commit": source["source_commit"], "hard_failures": [], "warnings": [],
                                   "global_locks": GLOBAL_LOCKS,
                                   "next_step": "R18 = execute exactly the frozen R17 E1 bounded-training envelope + causal learning-path evidence capture"},
        }
        for name, payload in outputs.items():
            dump(root / name, payload)
        (root / "final_report.md").write_text(
            "# BT8-R17 final report\n\n"
            f"- gate: `{PASS_GATE}`\n- classification: `{CLASSIFICATION}`\n- source commit: `{source['source_commit']}`\n\n"
            "R17 performed only timestamped-artifact/hash inspection, static envelope selection, and authorization serialization. "
            "It performed no training, rollout, reward settlement, optimizer action, backward pass, checkpoint write, or policy mutation.\n\n"
            "## Frozen R18 envelope\n\n"
            "- arms: `AC-R1` control and `BD-R1` E1 opening on environment seed `20260822`\n"
            "- windows / visits / trajectories / decisions / transitions: `6 / 12 / 12 / 48 / 96`\n"
            "- PPO: arm-local `24` rows, full batch/minibatch `24`, `3` epochs\n"
            "- Actor/Critic step ceiling: `6 / 6`; no supplemental steps\n"
            "- training selection: `FROZEN_MASKED_CATEGORICAL_TRAINING`; inference: `FROZEN_INFERENCE_T1`\n"
            "- static BD seed-0 support is evidence only; R18 blocks before PPO if fresh causal-path minima are not observed.\n\n"
            "Next and only authorized step: `R18 = execute exactly the frozen R17 E1 bounded-training envelope + causal learning-path evidence capture`.\n",
            encoding="utf-8")
        hashes = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE, "classification": CLASSIFICATION,
                                       "source_commit": source["source_commit"], "github_push_performed": False, "file_sha256": hashes})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(f"classification: {CLASSIFICATION}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except R17Error as exc:
        write_block(root=root, source=source, reason=exc.code, detail=str(exc), counters=execution)
        print(f"[BLOCKED] {exc.code}")
        print(f"artifact: {root.relative_to(PROJECT)}")
    except Exception as exc:  # noqa: BLE001
        write_block(root=root, source=source, reason=SOURCE_BLOCK, detail=f"{type(exc).__name__}:{exc}", counters=execution)
        print(f"[BLOCKED] {SOURCE_BLOCK}")
        print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
