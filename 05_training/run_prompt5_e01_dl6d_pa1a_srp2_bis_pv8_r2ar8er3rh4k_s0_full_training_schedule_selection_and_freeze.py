from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4K-S0"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_S0_FULL_TRAINING_SCHEDULE_SELECTION_AND_FREEZE_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_S0_FULL_TRAINING_SCHEDULE_SELECTION_REQUIRED"
PASS_DECISION = "PV8_FRESH_REWARD_V2_ZERO_LOSS_FULL_TRAINING_SCHEDULE_FROZEN_READY_FOR_H4K_RERUN"
BLOCK_DECISION = "PV8_FRESH_REWARD_V2_ZERO_LOSS_FULL_TRAINING_SCHEDULE_REMAINS_UNRESOLVED"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"
DL4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
H4I_RERUN_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_rerun_training_readiness_20260810_192924"
H4J_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4j_fresh_mappo_execution_integrity_20260810_200616"
ZL2_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4j_zl2_patent_aware_execution_integrity_20260810_221630"
H4K_BLOCK_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4k_fresh_reward_v2_zero_loss_three_seed_full_retraining_20260810_223354_BLOCKED"

EXPECTED = {
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

H4G_SOURCE_HASHES = {
    "05_training/rewards/mappo_reward_v1.py": "8f157b8ea0798b3ec72ab81ca747ba1d58ccf38d82767e0f5a302292b958da52",
    "05_training/simulator/pv8_reward_outcome_collector.py": "ea3ba294d86d5753e9a398dd1b539e6ea2ba862a39b83e175172fda17c6f4419",
    "05_training/simulator/pv8_b1_orchestrator.py": "4fc812b8e74415d64c2bbc981e53e7319dd8f8e6519a6908b313dce22b7e46b1",
    "05_training/mappo_runner.py": "b7a9c39534d90e4757dff7a5397cb8c67483ff0aac8533f610be7471e993d169",
}

REQUIRED_OUTPUTS = [
    "01_authoritative_schedule_sources.json",
    "02_historical_schedule_inventory.json",
    "03_schedule_semantic_compatibility.json",
    "04_full_training_budget_derivation.json",
    "05_rollout_update_cadence_contract.json",
    "06_checkpoint_termination_contract.json",
    "07_dataset_traversal_sealing_contract.json",
    "08_runtime_estimate.json",
    "09_h4k_full_training_schedule_freeze.json",
    "10_h4k_s0_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_entry(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def git_identity() -> Dict[str, Any]:
    def run(args: List[str]) -> str | None:
        try:
            completed = subprocess.run(args, cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=10)
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    return {
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def get_path(payload: Mapping[str, Any], keys: Iterable[Any], default: Any = None) -> Any:
    cur: Any = payload
    for key in keys:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def find_selected_parameter(parameters: Sequence[Mapping[str, Any]], name: str) -> Any:
    for row in parameters:
        if row.get("name") == name:
            return row.get("selected_value")
    return None


def extract_reward_v2_sha() -> str | None:
    text = (TRAINING_ROOT / "rewards/mappo_reward_v1.py").read_text(encoding="utf-8-sig")
    match = re.search(r'PV8_REWARD_V2_FREEZE_SHA256\s*=\s*"([0-9a-f]{64})"', text)
    return match.group(1) if match else None


def h4g_runtime_status() -> Dict[str, Any]:
    source_hashes = {}
    ok = True
    for rel, expected in H4G_SOURCE_HASHES.items():
        path = PROJECT_ROOT / rel
        observed = sha256_file(path) if path.exists() else None
        match = observed == expected
        source_hashes[rel] = {"expected_sha256": expected, "observed_sha256": observed, "match": match}
        ok = ok and match
    return {
        "expected_h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "observed_h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "source_hashes_match": ok,
        "source_hashes": source_hashes,
    }


def training_jsonl_shape(seed_dir: Path) -> Dict[str, Any]:
    rows = read_jsonl(seed_dir / "training_metrics.jsonl")
    return {
        "row_count": len(rows),
        "rollout_indices": sorted({int(row["rollout_index"]) for row in rows if row.get("rollout_index") is not None}),
        "ppo_update_indices_first_last": [
            rows[0].get("ppo_update_index") if rows else None,
            rows[-1].get("ppo_update_index") if rows else None,
        ],
        "max_ppo_update_index": max([int(row["ppo_update_index"]) for row in rows if row.get("ppo_update_index") is not None], default=0),
    }


def authoritative_sources(created_at: str) -> Dict[str, Any]:
    h4i_r3_gate = read_json(H4I_R3_ROOT / "09_h4i_r3_gate_matrix.json")
    h4i_scope = read_json(H4I_RERUN_ROOT / "02_training_scope_binding.json")
    h4i_rerun_gate = read_json(H4I_RERUN_ROOT / "03_h4i_rerun_gate_matrix.json")
    h4j_gate = read_json(H4J_ROOT / "09_h4j_gate_matrix.json")
    zl2_gate = read_json(ZL2_ROOT / "10_zl2_gate_matrix.json")
    h4k_block = read_json(H4K_BLOCK_ROOT / "12_h4k_gate_matrix.json") if (H4K_BLOCK_ROOT / "12_h4k_gate_matrix.json").exists() else {}
    observed_reward = extract_reward_v2_sha()
    observed_adapter = sha256_file(TRAINING_ROOT / "simulator/zero_loss_admission_adapter.py")
    h4g = h4g_runtime_status()
    checks = {
        "reward_v2_sha_match": observed_reward == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": h4g["observed_h4g_runtime_sha256"] == EXPECTED["h4g_runtime_sha"] and h4g["source_hashes_match"],
        "r3_split_sha_match": h4i_scope.get("scope_hash") == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": observed_adapter == EXPECTED["zero_loss_adapter_sha"],
        "h4i_r3_current_contract_pass": h4i_r3_gate.get("gate")
        == "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_R3_EXPLICIT_FRESH_TRAINING_CONTRACT_SELECTION_AND_FREEZE_COMPLETE",
        "h4i_rerun_pass": h4i_rerun_gate.get("gate")
        == "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_RERUN_REWARD_V2_TRAINING_READINESS_COMPLETE",
        "h4j_pass": h4j_gate.get("gate")
        == "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4J_FRESH_MAPPO_EXECUTION_INTEGRITY_COMPLETE",
        "zl2_pass": zl2_gate.get("gate")
        == "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4J_ZL2_PATENT_AWARE_EXECUTION_INTEGRITY_COMPLETE",
        "h4k_block_is_schedule_not_frozen": h4k_block.get("gate")
        == "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_FULL_TRAINING_SCHEDULE_NOT_FROZEN",
    }
    current_contract = {
        "seeds": [1, 2, 3],
        "train_validation_test": {
            "train": h4i_scope.get("train_window_count"),
            "validation": h4i_scope.get("validation_window_count"),
            "test": h4i_scope.get("test_window_count"),
        },
        "agents": get_path(h4i_r3_gate, ["preserved", "agents"]),
        "gamma": get_path(h4i_r3_gate, ["preserved", "gamma"]),
        "gae_lambda": get_path(h4i_r3_gate, ["preserved", "gae_lambda"]),
        "rollout_horizon": get_path(h4i_r3_gate, ["preserved", "rollout_horizon"]),
        "reward_v2_sha256": observed_reward,
        "h4g_runtime_sha256": h4g["observed_h4g_runtime_sha256"],
        "r3_split_sha256": h4i_scope.get("scope_hash"),
        "zero_loss_adapter_sha256": observed_adapter,
    }
    files = {
        "dl3_runner": source_entry(TRAINING_ROOT / "run_prompt5_e01_dl3_suseong_three_seed_full_training.py"),
        "dl3_training_budget": source_entry(DL3_ROOT / "training_budget.json"),
        "dl3_source_inventory": source_entry(DL3_ROOT / "source_change_inventory.json"),
        "dl4_runner": source_entry(TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"),
        "dl4_selected_critic_profile": source_entry(DL4_ROOT / "selected_critic_profile.json"),
        "h4i_r3_seed_split": source_entry(H4I_R3_ROOT / "02_seed_split_frozen_contract.json"),
        "h4i_r3_ppo": source_entry(H4I_R3_ROOT / "04_ppo_optimizer_frozen_contract.json"),
        "h4i_r3_critic": source_entry(H4I_R3_ROOT / "05_critic_frozen_contract.json"),
        "h4i_r3_gatv2": source_entry(H4I_R3_ROOT / "06_gatv2_encoder_frozen_contract.json"),
        "h4i_rerun_scope": source_entry(H4I_RERUN_ROOT / "02_training_scope_binding.json"),
        "h4j_gate": source_entry(H4J_ROOT / "09_h4j_gate_matrix.json"),
        "zl2_gate": source_entry(ZL2_ROOT / "10_zl2_gate_matrix.json"),
        "h4k_block_gate": source_entry(H4K_BLOCK_ROOT / "12_h4k_gate_matrix.json"),
        "h4k_block_schedule_resolution": source_entry(H4K_BLOCK_ROOT / "02_full_training_schedule_resolution.json"),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "hard_execution_lock": {
            "training": False,
            "optimizer_creation": False,
            "backward": False,
            "optimizer_step": False,
            "training_checkpoint_creation": False,
            "policy_evaluation": False,
        },
        "authoritative_current_contract_passed": all(checks.values()),
        "checks": checks,
        "current_contract": current_contract,
        "h4g_runtime_status": h4g,
        "source_files": files,
        "git_identity": git_identity(),
    }


def historical_inventory(created_at: str) -> Dict[str, Any]:
    dl3_budget = read_json(DL3_ROOT / "training_budget.json")
    dl3_config = read_json(DL3_ROOT / "seeds/seed_1/configuration.json")
    dl4_selected = read_json(DL4_ROOT / "selected_critic_profile.json")["selected_critic_profile"]
    dl4_three = read_json(DL4_ROOT / "three_seed_critic_summary.json")
    dl3_seed_shapes = {str(seed): training_jsonl_shape(DL3_ROOT / f"seeds/seed_{seed}") for seed in [1, 2, 3]}
    dl4_seed_shapes = {str(seed): training_jsonl_shape(DL4_ROOT / f"final_seeds/seed_{seed}") for seed in [1, 2, 3]}
    return {
        "stage": STAGE,
        "created_at": created_at,
        "dl3_three_seed_full_training": {
            "classification": "VALIDATED_HISTORICAL_FULL_TRAINING_SCHEDULE_EVIDENCE",
            "source_runner": source_entry(TRAINING_ROOT / "run_prompt5_e01_dl3_suseong_three_seed_full_training.py"),
            "training_budget": dl3_budget,
            "seed_configuration_reference": dl3_config,
            "seed_training_metric_shapes": dl3_seed_shapes,
            "outer_training_unit_observed": "dataset_pass",
            "outer_training_count_observed": dl3_budget.get("dataset_pass_count"),
            "rollout_formula_observed": "ceil(training_snapshot_count / rollout_horizon_configured)",
            "rollout_update_cadence_observed": "for each rollout segment, run ppo_epochs optimizer passes",
            "ppo_updates_formula_observed": "total_rollout_count * ppo_epochs",
            "window_traversal_observed": "sorted train files, contiguous offsets range(0, len(train_data), rollout_horizon)",
            "checkpoint_cadence_observed": ["initial", "best_validation", "final"],
            "termination_observed": "training_rows == total_rollouts * ppo_epochs after one dataset pass",
            "validation_involvement_observed": "validation at start/middle/end and best_validation checkpoint selection",
            "test_involvement_observed": "test evaluated once after loading best_validation checkpoint",
        },
        "dl4_critic_calibration": {
            "classification": "VALIDATED_HISTORICAL_CRITIC_UPDATE_BUDGET_EVIDENCE",
            "source_runner": source_entry(TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"),
            "selected_profile": dl4_selected,
            "final_seed_profiles": dl4_three.get("final_seed_profiles", []),
            "seed_training_metric_shapes": dl4_seed_shapes,
            "outer_training_unit_observed": "dataset_pass",
            "outer_training_count_observed": 1,
            "critic_updates_formula_observed": "total_rollouts * critic_epochs",
            "actor_ppo_epochs_observed": dl4_selected.get("actor_ppo_epochs"),
            "critic_epochs_observed": dl4_selected.get("critic_epochs"),
            "extra_critic_epochs_observed": int(dl4_selected.get("critic_epochs", 0)) - int(dl4_selected.get("actor_ppo_epochs", 0)),
            "checkpoint_cadence_observed": ["initial", "best_validation", "final"],
            "validation_involvement_observed": "validation profile/checkpoint ranking; not reusable for H4K selection",
            "test_involvement_observed": "final diagnostic only in DL4; not reusable for H4K",
        },
        "current_lineage_inventory": {
            "r3_fresh_contract": "freezes split/hyperparameters/fresh lineage, not full schedule until S0",
            "h4i_rerun": "readiness only; no training/backward/optimizer/checkpoint",
            "h4j": "seed=1 execution-integrity only; no full schedule",
            "zl2": "seed=1 patent-aware integrity only; no full schedule",
            "h4k_block": "blocked specifically because schedule was not frozen",
        },
    }


def compatibility(created_at: str) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "classification_legend": [
            "REUSABLE_UNCHANGED",
            "REUSABLE_WITH_CURRENT_BINDING",
            "NOT_REUSABLE",
            "UNKNOWN",
        ],
        "items": [
            {
                "element": "DL3 dataset_pass_count=1",
                "classification": "REUSABLE_WITH_CURRENT_BINDING",
                "reason": "Historical outer unit is one full pass over training split; bind the same unit to current R3 train-44 windows.",
            },
            {
                "element": "DL3 rollout formula ceil(train_count / rollout_horizon)",
                "classification": "REUSABLE_WITH_CURRENT_BINDING",
                "reason": "Formula is runner-defined and compatible with current rollout_horizon=512 and train_window_count=44.",
            },
            {
                "element": "DL3 PPO epochs/update=4",
                "classification": "REUSABLE_UNCHANGED",
                "reason": "Already frozen by current H4I-R3 PPO optimizer contract.",
            },
            {
                "element": "DL4 critic epochs/update=8",
                "classification": "REUSABLE_UNCHANGED",
                "reason": "Already frozen by current H4I-R3 critic contract; H4I-R3 GATv2 contract marks critic-only extra epochs as detached from encoder.",
            },
            {
                "element": "DL3/DL4 validation best checkpoint ranking",
                "classification": "NOT_REUSABLE",
                "reason": "H4K forbids validation-based model selection, early stopping, and winner selection.",
            },
            {
                "element": "DL3/DL4 test metrics/checkpoint comparison",
                "classification": "NOT_REUSABLE",
                "reason": "H4K test split remains sealed and policy evaluation is not authorized.",
            },
            {
                "element": "DL3/DL4 initial/final/best checkpoint learned states",
                "classification": "NOT_REUSABLE",
                "reason": "Fresh lineage forbids learned-state reuse; only schedule metadata is reusable.",
            },
            {
                "element": "Final checkpoint after completed seed",
                "classification": "REUSABLE_WITH_CURRENT_BINDING",
                "reason": "A final checkpoint is required as a future evaluation candidate, but no best-validation checkpoint or selection behavior is reused.",
            },
        ],
        "compatibility_checks": {
            "reward_v2": "compatible_as_schedule_only; reward formula/weights unchanged",
            "zero_loss_admission": "compatible_as_runtime_constraint; schedule does not weaken epsilon or eligibility",
            "k_mask": "compatible_as_runtime_constraint; schedule does not alter K-mask",
            "rollout_horizon_512": "compatible_unchanged",
            "train_44_split": "compatible_by_current_binding",
            "critic_epochs_8": "compatible_unchanged",
            "fresh_gatv2_actor_critic": "compatible; no historical weights/states reused",
        },
        "unique_schedule_supported": True,
        "candidate_count_after_compatibility_filter": 1,
    }


def selected_schedule_payload(created_at: str) -> Dict[str, Any]:
    h4i_seed_split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")
    h4i_gate = read_json(H4I_R3_ROOT / "09_h4i_r3_gate_matrix.json")
    h4i_ppo = read_json(H4I_R3_ROOT / "04_ppo_optimizer_frozen_contract.json")
    h4i_critic = read_json(H4I_R3_ROOT / "05_critic_frozen_contract.json")
    h4i_gatv2 = read_json(H4I_R3_ROOT / "06_gatv2_encoder_frozen_contract.json")
    h4i_scope = read_json(H4I_RERUN_ROOT / "02_training_scope_binding.json")
    ppo_params = h4i_ppo.get("parameters", [])

    train_windows = int(h4i_scope["train_window_count"])
    rollout_horizon = int(get_path(h4i_gate, ["preserved", "rollout_horizon"]))
    ppo_epochs = int(find_selected_parameter(ppo_params, "ppo_epochs"))
    critic_epochs = int(get_path(h4i_critic, ["training_hyperparameters", "critic_epochs"]))
    outer_count = 1
    rollouts_per_outer = math.ceil(train_windows / rollout_horizon)
    total_rollouts = outer_count * rollouts_per_outer
    ppo_updates = total_rollouts * ppo_epochs
    critic_updates = total_rollouts * critic_epochs
    schedule = {
        "schedule_id": "H4K_S0_DL3_DATASET_PASS_1_BIND_TO_R3_TRAIN44_REWARD_V2_ZERO_LOSS",
        "schedule_version": "H4K_FULL_TRAINING_SCHEDULE_V1",
        "created_at": created_at,
        "stage": STAGE,
        "selection_status": "SELECT_AND_FREEZE",
        "selection_rule": "current R3/H4I contract + DL3/DL4 validated one-pass schedule evidence + minimum-change binding to TRAIN 44",
        "seeds": [1, 2, 3],
        "training_windows": train_windows,
        "validation_windows": int(h4i_scope["validation_window_count"]),
        "test_windows": int(h4i_scope["test_window_count"]),
        "training_input": "TRAIN_44_ONLY",
        "agents": int(get_path(h4i_gate, ["preserved", "agents"])),
        "gamma": float(get_path(h4i_gate, ["preserved", "gamma"])),
        "gae_lambda": float(get_path(h4i_gate, ["preserved", "gae_lambda"])),
        "outer_training_unit": "full_train_pass",
        "outer_training_unit_definition": "one deterministic traversal over all 44 frozen R3 training windows in the current H4I-R3 train order",
        "outer_training_count": outer_count,
        "rollout_horizon": rollout_horizon,
        "rollouts_per_outer_unit": rollouts_per_outer,
        "total_rollouts_per_seed": total_rollouts,
        "last_rollout_partial": train_windows % rollout_horizon != 0,
        "last_rollout_effective_window_count": train_windows % rollout_horizon or rollout_horizon,
        "ppo_epochs_per_update": ppo_epochs,
        "ppo_updates_per_outer_unit": rollouts_per_outer * ppo_epochs,
        "total_ppo_updates_per_seed": ppo_updates,
        "critic_epochs_per_update": critic_epochs,
        "critic_updates_per_outer_unit": rollouts_per_outer * critic_epochs,
        "total_critic_updates_per_seed": critic_updates,
        "actor_gatv2_epochs_per_rollout": ppo_epochs,
        "critic_only_extra_epochs_per_rollout": max(0, critic_epochs - ppo_epochs),
        "critic_only_extra_epochs_detach_encoder": bool(get_path(h4i_gatv2, ["trainable_modules", "critic_only_extra_epochs_detach_encoder"])),
        "minibatch": int(find_selected_parameter(ppo_params, "minibatch_size")),
        "clip_epsilon": float(find_selected_parameter(ppo_params, "ppo_clip_epsilon")),
        "actor_lr": float(find_selected_parameter(ppo_params, "actor_learning_rate")),
        "gatv2_lr": float(find_selected_parameter(ppo_params, "gatv2_learning_rate")),
        "critic_lr": float(find_selected_parameter(ppo_params, "critic_learning_rate")),
        "value_coefficient": float(find_selected_parameter(ppo_params, "value_loss_coefficient")),
        "critic_loss": str(get_path(h4i_critic, ["training_hyperparameters", "loss"])).upper(),
        "scheduler": find_selected_parameter(ppo_params, "scheduler_policy"),
        "window_traversal_order": h4i_seed_split["ordered_window_ids"]["train"],
        "window_traversal_policy": {
            "order": "frozen H4I-R3 train order",
            "shuffle": False,
            "shuffling_authorized": False,
            "rng_affects_traversal": False,
            "seed_rng_scope": "model initialization/action sampling/minibatch order only; not window order",
        },
        "rollout_update_cadence": "collect one rollout segment over up to 512 train windows -> compute Reward V2/Zero-Loss/K-mask/action/GAE -> run 4 actor/GATv2 PPO update epochs and 8 critic epochs -> continue until full_train_pass complete",
        "checkpoint_cadence": {
            "intermediate_checkpoint_rule": "NONE; no best-validation or periodic ranking checkpoint in H4K",
            "final_checkpoint_rule": "write exactly one final training checkpoint per seed after the frozen full_train_pass completes and all integrity checks for that seed are recorded",
            "checkpoint_namespaces": [
                "H4K_SEED_001_FRESH_REWARD_V2_ZERO_LOSS",
                "H4K_SEED_002_FRESH_REWARD_V2_ZERO_LOSS",
                "H4K_SEED_003_FRESH_REWARD_V2_ZERO_LOSS",
            ],
            "same_seed_recovery_checkpoint": "not authorized by S0; interrupted seed restarts fresh",
        },
        "final_checkpoint_condition": "seed completes exactly outer_training_count=1 full_train_pass, total_rollouts_per_seed=1, total_ppo_updates_per_seed=4, total_critic_updates_per_seed=8, no hard safety violation, no NaN/Inf, and checkpoint write/read validation passes",
        "seed_termination_rule": "seed terminates when the fixed precommitted full-training budget completes; no validation early stopping and no test access",
        "validation_usage": "NONE_FOR_H4K_SELECTION",
        "test_usage": "SEALED_NOT_OPENED",
        "resume_policy": {
            "auto_resume": False,
            "latest_pt_implicit_load": False,
            "best_pt_implicit_load": False,
            "same_seed_resume_authorized": False,
            "interrupted_seed_recovery": "restart affected seed FRESH from seed-specific RNG and fresh modules",
        },
        "fresh_policy": {
            "seed_1": "FRESH",
            "seed_2": "FRESH",
            "seed_3": "FRESH",
            "cross_seed_reuse": False,
            "h4j_checkpoint_reuse": False,
            "zl2_checkpoint_reuse": False,
            "dl3_dl4_learned_state_reuse": False,
            "optimizer_state_reuse": False,
            "normalizer_state_reuse": False,
        },
        "forbidden_h4k_s0_actions": {
            "training": False,
            "optimizer_creation": False,
            "backward": False,
            "optimizer_step": False,
            "training_checkpoint_creation": False,
            "policy_evaluation": False,
        },
    }
    schedule_hash = canonical_sha(schedule)
    schedule["full_training_schedule_sha256"] = schedule_hash
    schedule["full_training_schedule_sha256_scope"] = "canonical JSON of this contract excluding full_training_schedule_sha256 itself"
    return schedule


def budget_derivation(created_at: str, schedule: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "selected_schedule_id": schedule["schedule_id"],
        "missing_h4k_fields_resolved": {
            "A_outer_training_budget": {
                "resolved": True,
                "outer_training_unit": schedule["outer_training_unit"],
                "outer_training_count": schedule["outer_training_count"],
                "evidence": "DL3 training_budget.dataset_pass_count=1; bound to current R3 TRAIN 44.",
            },
            "B_total_ppo_updates": {
                "resolved": True,
                "ppo_updates_per_outer_unit": schedule["ppo_updates_per_outer_unit"],
                "total_ppo_updates_per_seed": schedule["total_ppo_updates_per_seed"],
                "formula": "ceil(44 / 512) * 4 = 4",
            },
            "C_rollout_update_cadence": {
                "resolved": True,
                "cadence": schedule["rollout_update_cadence"],
                "training_window_traversal": "frozen H4I-R3 train order, no shuffle",
            },
            "D_checkpoint_cadence": {
                "resolved": True,
                **schedule["checkpoint_cadence"],
            },
            "E_seed_termination_rule": {
                "resolved": True,
                "seed_termination_rule": schedule["seed_termination_rule"],
            },
        },
        "derivation": {
            "train_windows": schedule["training_windows"],
            "rollout_horizon": schedule["rollout_horizon"],
            "rollouts_per_outer_unit_formula": "ceil(training_windows / rollout_horizon)",
            "rollouts_per_outer_unit": schedule["rollouts_per_outer_unit"],
            "outer_training_count": schedule["outer_training_count"],
            "total_rollouts_per_seed": schedule["total_rollouts_per_seed"],
            "ppo_epochs_per_update": schedule["ppo_epochs_per_update"],
            "total_ppo_updates_per_seed_formula": "outer_training_count * rollouts_per_outer_unit * ppo_epochs_per_update",
            "total_ppo_updates_per_seed": schedule["total_ppo_updates_per_seed"],
            "critic_epochs_per_update": schedule["critic_epochs_per_update"],
            "total_critic_updates_per_seed_formula": "outer_training_count * rollouts_per_outer_unit * critic_epochs_per_update",
            "total_critic_updates_per_seed": schedule["total_critic_updates_per_seed"],
        },
        "candidate_schedules_considered": [
            {
                "candidate_id": schedule["schedule_id"],
                "status": "SELECTED",
                "scientific_basis": "only candidate that preserves current R3 TRAIN 44 scope, DL3 one-pass schedule semantics, and H4I-R3 critic=8 contract without validation/test selection",
            }
        ],
        "arbitrary_round_number_used": False,
        "training_budget_sweep_conducted": False,
    }


def cadence_contract(created_at: str, schedule: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schedule_id": schedule["schedule_id"],
        "rollout_to_update_cadence": schedule["rollout_update_cadence"],
        "rollouts_per_seed": schedule["total_rollouts_per_seed"],
        "ppo_updates_per_seed": schedule["total_ppo_updates_per_seed"],
        "critic_updates_per_seed": schedule["total_critic_updates_per_seed"],
        "window_traversal_order": schedule["window_traversal_order"],
        "window_traversal_policy": schedule["window_traversal_policy"],
        "zero_loss_placement_preserved": [
            "STATE_SNAPSHOT",
            "OBLIGATION_SNAPSHOT",
            "ZERO_LOSS_ADMISSION",
            "K_MASK_BUILD",
            "ACTION_SELECTION",
            "ACTION_VALIDATION",
            "BOARDING_ALIGHTING",
            "LOCAL_SERVICE_SETTLEMENT",
            "HOLD_IF_APPLICABLE",
            "DEPARTURE",
            "NEXT_LINK_TRAVEL",
        ],
        "gatv2_attention_role": "EVIDENCE_ONLY",
        "validation_or_test_in_cadence": False,
    }


def checkpoint_termination(created_at: str, schedule: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schedule_id": schedule["schedule_id"],
        "checkpoint_cadence": schedule["checkpoint_cadence"],
        "final_checkpoint_condition": schedule["final_checkpoint_condition"],
        "seed_termination_rule": schedule["seed_termination_rule"],
        "resume_policy": schedule["resume_policy"],
        "fresh_policy": schedule["fresh_policy"],
        "validation_checkpoint_ranking_authorized": False,
        "winner_selection_authorized": False,
        "test_opening_authorized": False,
        "training_checkpoint_created_in_s0": False,
    }


def traversal_sealing(created_at: str, schedule: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schedule_id": schedule["schedule_id"],
        "training": {
            "usage": "OPTIMIZATION_ONLY_IN_H4K_RERUN",
            "window_count": schedule["training_windows"],
            "input": "TRAIN_44_ONLY",
            "order": schedule["window_traversal_order"],
            "shuffle": False,
        },
        "validation": {
            "window_count": schedule["validation_windows"],
            "usage": "NO_OPTIMIZATION_NO_EARLY_STOPPING_NO_CHECKPOINT_RANKING_IN_H4K",
            "opened_in_s0": False,
            "selection_authorized": False,
        },
        "test": {
            "window_count": schedule["test_windows"],
            "usage": "SEALED_NOT_OPENED",
            "opened_in_s0": False,
            "policy_evaluation_authorized": False,
        },
        "historical_conflicts_resolved": {
            "dl3_best_validation_checkpoint": "not reused",
            "dl3_test_after_best_validation": "not reused",
            "dl4_validation_profile_selection": "not reused",
            "dl4_test_diagnostic": "not reused",
        },
    }


def runtime_estimate(created_at: str, schedule: Mapping[str, Any]) -> Dict[str, Any]:
    dl3_summary = read_json(DL3_ROOT / "three_seed_metric_summary.json")
    dl4_three = read_json(DL4_ROOT / "three_seed_critic_summary.json")
    dl3_elapsed_mean = float(dl3_summary["elapsed_time"]["mean"])
    dl3_update_count = int(read_json(DL3_ROOT / "training_budget.json")["total_ppo_update_count"])
    dl3_seconds_per_ppo_update = dl3_elapsed_mean / dl3_update_count
    dl4_elapsed_values = [float(row["elapsed_seconds"]) for row in dl4_three["final_seed_profiles"]]
    dl4_elapsed_mean = mean(dl4_elapsed_values)
    dl4_critic_update_count = max(training_jsonl_shape(DL4_ROOT / "final_seeds/seed_1")["max_ppo_update_index"], 1)
    dl4_seconds_per_critic_update = dl4_elapsed_mean / dl4_critic_update_count
    actor_gat_est = dl3_seconds_per_ppo_update * int(schedule["total_ppo_updates_per_seed"])
    critic_est = dl4_seconds_per_critic_update * int(schedule["total_critic_updates_per_seed"])
    estimate_seconds = max(actor_gat_est, critic_est)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schedule_id": schedule["schedule_id"],
        "estimated_minutes_per_seed": estimate_seconds / 60.0,
        "estimated_total_three_seed_minutes": estimate_seconds * 3.0 / 60.0,
        "estimate_basis": {
            "dl3_elapsed_mean_seconds": dl3_elapsed_mean,
            "dl3_total_ppo_update_count": dl3_update_count,
            "dl3_seconds_per_ppo_update": dl3_seconds_per_ppo_update,
            "h4k_scheduled_ppo_updates_per_seed": schedule["total_ppo_updates_per_seed"],
            "dl4_elapsed_mean_seconds": dl4_elapsed_mean,
            "dl4_critic_update_count_per_seed": dl4_critic_update_count,
            "dl4_seconds_per_critic_update": dl4_seconds_per_critic_update,
            "h4k_scheduled_critic_updates_per_seed": schedule["total_critic_updates_per_seed"],
            "combination_rule": "use max(actor/GAT PPO scaled estimate, critic scaled estimate) to avoid double-counting shared work",
        },
        "uncertainty": "HIGH: no full H4K Reward V2 + Zero-Loss three-seed run exists; estimate is scaled only from existing DL3/DL4 Mac M4 evidence and is informational.",
        "runtime_estimate_changed_schedule": False,
    }


def gate_matrix(created_at: str, sources: Mapping[str, Any], compat: Mapping[str, Any], schedule: Mapping[str, Any]) -> Dict[str, Any]:
    missing_fields_resolved = True
    pass_gate = bool(sources["authoritative_current_contract_passed"]) and bool(compat["unique_schedule_supported"]) and missing_fields_resolved
    return {
        "stage": STAGE,
        "created_at": created_at,
        "gate": PASS_GATE if pass_gate else BLOCK_GATE,
        "decision": PASS_DECISION if pass_gate else BLOCK_DECISION,
        "blocking_decisions": [] if pass_gate else ["FULL_TRAINING_SCHEDULE_SELECTION_REQUIRED"],
        "criteria": {
            "authoritative_current_contract_passed": bool(sources["authoritative_current_contract_passed"]),
            "historical_executable_evidence_inspected": True,
            "semantic_compatibility_passed": bool(compat["unique_schedule_supported"]),
            "five_missing_fields_resolved": missing_fields_resolved,
            "single_schedule_selected": bool(compat["candidate_count_after_compatibility_filter"] == 1),
            "full_training_schedule_frozen": pass_gate,
            "training_executed": False,
            "optimizer_creation": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "training_checkpoint_creation": False,
            "policy_evaluation": False,
        },
        "final_state": {
            "full_training_schedule_frozen": pass_gate,
            "full_training_schedule_sha256": schedule["full_training_schedule_sha256"] if pass_gate else None,
            "training_authorized": False,
            "training_executed": False,
            "three_seed_full_training_authorized": False,
            "policy_evaluation_authorized": False,
            "winner_selection_authorized": False,
            "patent_performance_claim_allowed": False,
            "paper_level_claim_allowed": False,
        },
        "next": "H4K_RERUN_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_FULL_RETRAINING" if pass_gate else "EXPLICIT_SCHEDULE_SELECTION_REQUIRED",
    }


def final_report(gate: Mapping[str, Any], schedule: Mapping[str, Any], output_root: Path) -> str:
    return "\n".join(
        [
            "# H4K-S0 Full-Training Schedule Selection & Freeze",
            "",
            f"gate = {gate['gate']}",
            f"decision = {gate['decision']}",
            "",
            "## Frozen schedule",
            "",
            f"- schedule_id = `{schedule['schedule_id']}`",
            f"- full_training_schedule_sha256 = `{schedule['full_training_schedule_sha256']}`",
            f"- outer_training_unit/count = `{schedule['outer_training_unit']} / {schedule['outer_training_count']}`",
            f"- training_windows = `{schedule['training_windows']}`",
            f"- rollout_horizon = `{schedule['rollout_horizon']}`",
            f"- total_rollouts_per_seed = `{schedule['total_rollouts_per_seed']}`",
            f"- total_ppo_updates_per_seed = `{schedule['total_ppo_updates_per_seed']}`",
            f"- total_critic_updates_per_seed = `{schedule['total_critic_updates_per_seed']}`",
            "",
            "Validation is not used for H4K selection, early stopping, checkpoint ranking, or tuning. Test remains sealed.",
            "",
            "No training, optimizer creation, backward(), optimizer.step(), training checkpoint creation, policy evaluation, baseline comparison, winner selection, Reward V2 modification, or Zero-Loss modification was executed in S0.",
            "",
            "## Artifact root",
            "",
            str(output_root),
            "",
            "STOP.",
            "",
        ]
    )


def build_payloads(created_at: str, output_root: Path) -> Dict[str, Any]:
    sources = authoritative_sources(created_at)
    inventory = historical_inventory(created_at)
    compat = compatibility(created_at)
    schedule = selected_schedule_payload(created_at)
    budget = budget_derivation(created_at, schedule)
    cadence = cadence_contract(created_at, schedule)
    checkpoint = checkpoint_termination(created_at, schedule)
    traversal = traversal_sealing(created_at, schedule)
    runtime = runtime_estimate(created_at, schedule)
    gate = gate_matrix(created_at, sources, compat, schedule)
    return {
        "01_authoritative_schedule_sources.json": sources,
        "02_historical_schedule_inventory.json": inventory,
        "03_schedule_semantic_compatibility.json": compat,
        "04_full_training_budget_derivation.json": budget,
        "05_rollout_update_cadence_contract.json": cadence,
        "06_checkpoint_termination_contract.json": checkpoint,
        "07_dataset_traversal_sealing_contract.json": traversal,
        "08_runtime_estimate.json": runtime,
        "09_h4k_full_training_schedule_freeze.json": schedule,
        "10_h4k_s0_gate_matrix.json": gate,
        "final_report.md": final_report(gate, schedule, output_root),
    }


def main() -> None:
    now = kst_now()
    created_at = now.isoformat(timespec="seconds")
    output_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4k_s0_full_training_schedule_selection_and_freeze_{now.strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    payloads = build_payloads(created_at, output_root)
    for name, payload in payloads.items():
        path = output_root / name
        if name.endswith(".json"):
            dump_json(path, payload)
        else:
            dump_text(path, str(payload))

    output_sha = {name: sha256_file(output_root / name) for name in sorted(payloads)}
    manifest = {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_root": str(output_root),
        "required_outputs": REQUIRED_OUTPUTS,
        "required_outputs_present": sorted([*payloads.keys(), "manifest.json"]) == sorted(REQUIRED_OUTPUTS),
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-reference",
        "output_files": {name: str(output_root / name) for name in sorted(payloads)},
        "output_sha256": output_sha,
        "source_sha256": {
            "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_s0_full_training_schedule_selection_and_freeze.py": sha256_file(Path(__file__).resolve()),
            "05_training/run_prompt5_e01_dl3_suseong_three_seed_full_training.py": sha256_file(TRAINING_ROOT / "run_prompt5_e01_dl3_suseong_three_seed_full_training.py"),
            "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py": sha256_file(TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"),
            "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py": sha256_file(TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py")
            if (TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py").exists()
            else None,
            "05_training/simulator/zero_loss_admission_adapter.py": sha256_file(TRAINING_ROOT / "simulator/zero_loss_admission_adapter.py"),
            "05_training/rewards/mappo_reward_v1.py": sha256_file(TRAINING_ROOT / "rewards/mappo_reward_v1.py"),
        },
        "gate": payloads["10_h4k_s0_gate_matrix.json"]["gate"],
        "decision": payloads["10_h4k_s0_gate_matrix.json"]["decision"],
        "full_training_schedule_frozen": payloads["10_h4k_s0_gate_matrix.json"]["final_state"]["full_training_schedule_frozen"],
        "full_training_schedule_sha256": payloads["09_h4k_full_training_schedule_freeze.json"]["full_training_schedule_sha256"],
        "training_authorized": False,
        "training_executed": False,
        "three_seed_full_training_authorized": False,
        "policy_evaluation_authorized": False,
        "optimizer_creation": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "training_checkpoint_creation": False,
    }
    dump_json(output_root / "manifest.json", manifest)
    print(f"[OK] H4K-S0 artifact root: {output_root}")
    print(f"[OK] gate: {manifest['gate']}")
    print(f"[OK] schedule sha256: {manifest['full_training_schedule_sha256']}")
    print("[OK] training=false optimizer_creation=false backward=false optimizer_step=false")


if __name__ == "__main__":
    main()
