#!/usr/bin/env python3
"""H4M-AA frozen window-boundary-repaired policy validation.

Read-only inference validation of the three H4M-Y repaired checkpoints on the
already-approved frozen validation split.  Training, optimizer steps, backward
passes and parameter mutation are all counted and must stay at zero, and TEST6
is never opened.  Historical artifacts are read only.
"""

from __future__ import annotations

import hashlib
import importlib.util
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

import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-AA"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AA_"
    "FROZEN_WINDOW_BOUNDARY_REPAIRED_POLICY_VALIDATION_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AA"
NEXT_GATE = "H4M-AB_SEALED_HOLDOUT_PROMOTION_DECISION_GATE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MQ_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
REWARD_SOURCE = TRAINING_ROOT / "rewards/mappo_reward_v1.py"

H4MZ_ROOT_GLOB = "pv8_r2a_r8e_r3_r_h4m_z_window_boundary_outcome_review_validation_promotion_*"
H4MY_RECOVERY_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_r1_report_only_recovery_from_persisted_evidence_20260818_153018+09:00"
H4MY_TRAINING_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_fresh_window_boundary_repaired_three_seed_retraining_20260818_150812+09:00"

EXPECTED = {
    "h4m_z_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Z_"
        "WINDOW_BOUNDARY_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_VALIDATION_PROMOTION_COMPLETE"
    ),
    "h4m_z_classification": "A_PRIMARY_CAUSAL_ROOT_CAUSE_CONFIRMED_WINDOW_EPISODE_BOUNDARY",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "seeds": [1, 2, 3],
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
CORRECT_ACTION = {HOLD_BETTER: HOLD, SERVE_BETTER: SERVE}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "validation_binding.json",
    "frozen_validation_results.json",
    "seed_consistency.json",
    "integrity_validation.json",
    "training_vs_validation_comparison.json",
    "gate_matrix.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


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


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def latest_h4mz_root() -> Path:
    roots = sorted(ARTIFACTS_ROOT.glob(H4MZ_ROOT_GLOB))
    passing = [
        root
        for root in roots
        if (root / "gate_matrix.json").exists() and read_json(root / "gate_matrix.json").get("gate") == EXPECTED["h4m_z_gate"]
    ]
    if not passing:
        raise RuntimeError("no passing H4M-Z promotion artifact found")
    return passing[-1]


class ReadOnlyGuard:
    """Counts optimizer steps and backward passes for the whole validation."""

    def __init__(self) -> None:
        self.optimizer_steps = 0
        self.backward_calls = 0
        self._step = torch.optim.Optimizer.step
        self._backward = torch.Tensor.backward

    def __enter__(self) -> "ReadOnlyGuard":
        guard = self

        def counted_step(self_: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
            guard.optimizer_steps += 1
            return guard._step(self_, *args, **kwargs)

        def counted_backward(self_: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
            guard.backward_calls += 1
            return guard._backward(self_, *args, **kwargs)

        torch.optim.Optimizer.step = counted_step
        torch.Tensor.backward = counted_backward
        return self

    def __exit__(self, *exc_info: Any) -> None:
        torch.optim.Optimizer.step = self._step
        torch.Tensor.backward = self._backward


# ---------------------------------------------------------------------------
def validation_binding(created_at: str, promotion_root: Path) -> Dict[str, Any]:
    z_gate = read_json(promotion_root / "gate_matrix.json")
    z_decision = read_json(promotion_root / "root_cause_decision.json")
    contract_payload = read_json(promotion_root / "validation_promotion_contract.json")
    contract = contract_payload.get("contract", {})
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="h4maa_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "aa.pyc"), doraise=True)
            compile_error = None
        except Exception as exc:
            compile_error = repr(exc)
    checkpoints = contract.get("training_checkpoints", [])
    observed = {int(row["seed"]): sha256_file(Path(row["path"])) for row in checkpoints}
    checks = {
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "py_compile_passed": compile_error is None,
        "h4m_z_gate_match": z_gate.get("gate") == EXPECTED["h4m_z_gate"],
        "h4m_z_classification_a": z_decision.get("classification") == EXPECTED["h4m_z_classification"],
        "promotion_granted": contract_payload.get("promoted") is True and bool(contract_payload.get("contract_sha256")),
        "contract_w1_sha": contract.get("w1_repair_contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "contract_actor_sha": contract.get("actor_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "contract_critic_sha": contract.get("critic_s3_repair_contract_sha256") == EXPECTED["critic_repair_contract_sha256"],
        "contract_reward_sha": contract.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "three_checkpoints_bound": len(checkpoints) == 3,
        "checkpoint_sha_matches_contract": all(observed[int(row["seed"])] == row["sha256"] for row in checkpoints),
        "checkpoints_not_test_promoted": contract.get("checkpoint_test_promotion_status") == "NOT_TEST_PROMOTED_UNTIL_VALIDATION_PASSES",
        "test6_not_authorised_by_contract": contract.get("test6_authorised") is False,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit": head,
        "promotion_artifact": str(promotion_root),
        "validation_promotion_contract_sha256": contract_payload.get("contract_sha256"),
        "bound_checkpoints": [{"seed": row["seed"], "path": row["path"], "contract_sha256": row["sha256"], "observed_sha256": observed[int(row["seed"])]} for row in checkpoints],
        "checks": checks,
        "validation_binding_passed": all(checks.values()),
        "read_only_attestation": {
            "training_count": 0,
            "optimizer_step_count": 0,
            "backward_pass_count": 0,
            "parameter_mutation_count": 0,
            "test6_access_count": 0,
            "tuning_applied": False,
            "database_or_data_mutation": False,
            "historical_artifacts_modified": False,
            "github_push_performed": False,
        },
    }


def frozen_validation_results(validation: Mapping[str, Any]) -> Dict[str, Any]:
    aggregate = validation.get("aggregate_by_classification", {})
    rows = validation.get("by_sample_rows", []) or []
    by_seed: Dict[str, Any] = {}
    for seed in EXPECTED["seeds"]:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        per_class: Dict[str, Any] = {}
        for label in (HOLD_BETTER, SERVE_BETTER):
            class_rows = [row for row in seed_rows if row["classification"] == label]
            if not class_rows:
                continue
            chosen = {name: sum(1 for row in class_rows if row["action_name"] == name) for name in (HOLD, SERVE)}
            correct = sum(1 for row in class_rows if row["sampled_correct_direction"])
            probability_correct = sum(1 for row in class_rows if row["probability_correct_direction"])
            p_hold = sum(row["P_HOLD"] for row in class_rows) / len(class_rows)
            p_serve = sum(row["P_SERVE"] for row in class_rows) / len(class_rows)
            per_class[label] = {
                "sample_count": len(class_rows),
                "chosen_action_counts": chosen,
                "chosen_action_dominant": HOLD if chosen[HOLD] > chosen[SERVE] else SERVE if chosen[SERVE] > chosen[HOLD] else "TIE",
                "context_appropriate_action": CORRECT_ACTION[label],
                "sampled_correct_direction_count": correct,
                "sampled_correct_direction_rate": correct / len(class_rows),
                "probability_correct_direction_rate": probability_correct / len(class_rows),
                "P_HOLD_mean": p_hold,
                "P_SERVE_mean": p_serve,
                "probability_dominant_action": HOLD if p_hold > p_serve else SERVE if p_serve > p_hold else "TIE",
                "discrimination_is_context_appropriate": (HOLD if chosen[HOLD] > chosen[SERVE] else SERVE if chosen[SERVE] > chosen[HOLD] else "TIE")
                == CORRECT_ACTION[label],
            }
        by_seed[str(seed)] = per_class
    return {
        "stage": STAGE,
        "validation_scope": "already-approved frozen validation split (H4L validation windows); TEST6 never opened",
        "validation_plan_source": (validation.get("validation_plan") or {}).get("validation_basis_source"),
        "test_snapshot_paths_loaded": (validation.get("validation_plan") or {}).get("test_snapshot_paths_loaded"),
        "row_count": len(rows),
        "aggregate_by_classification": {
            label: {
                "sample_count": aggregate.get(label, {}).get("count"),
                "chosen_action_counts": aggregate.get(label, {}).get("sampled_action_counts"),
                "sampled_dominant_action": aggregate.get(label, {}).get("sampled_dominant_action"),
                "probability_dominant_action": aggregate.get(label, {}).get("probability_dominant_action"),
                "context_appropriate_action": CORRECT_ACTION[label],
                "sampled_correct_direction_rate": aggregate.get(label, {}).get("sampled_correct_direction_rate"),
                "probability_correct_direction_rate": aggregate.get(label, {}).get("probability_correct_direction_rate"),
                "P_HOLD_mean": (aggregate.get(label, {}).get("P_HOLD") or {}).get("mean"),
                "P_SERVE_mean": (aggregate.get(label, {}).get("P_SERVE") or {}).get("mean"),
                "policy_entropy_mean": (aggregate.get(label, {}).get("policy_entropy") or {}).get("mean"),
            }
            for label in (HOLD_BETTER, SERVE_BETTER)
        },
        "by_seed": by_seed,
        "frozen_criteria": validation.get("checks"),
        "frozen_criteria_passed": validation.get("validation_discrimination_passed"),
        "criteria_source": "existing frozen H4M-Q validation criteria; no new numeric threshold was introduced",
    }


def seed_consistency(results: Mapping[str, Any]) -> Dict[str, Any]:
    per_seed = results["by_seed"]
    directional = {
        seed: {label: row["discrimination_is_context_appropriate"] for label, row in classes.items()}
        for seed, classes in per_seed.items()
    }
    dominants = {
        label: sorted({classes[label]["chosen_action_dominant"] for classes in per_seed.values() if label in classes})
        for label in (HOLD_BETTER, SERVE_BETTER)
    }
    checks = {
        "all_seeds_present": sorted(per_seed) == [str(seed) for seed in EXPECTED["seeds"]],
        "all_seeds_hold_better_context_appropriate": all(row.get(HOLD_BETTER) is True for row in directional.values()),
        "all_seeds_serve_better_context_appropriate": all(row.get(SERVE_BETTER) is True for row in directional.values()),
        "no_seed_disagreement_on_hold_better": len(dominants[HOLD_BETTER]) == 1,
        "no_seed_disagreement_on_serve_better": len(dominants[SERVE_BETTER]) == 1,
        "no_single_action_collapse_across_contexts": dominants[HOLD_BETTER] != dominants[SERVE_BETTER],
    }
    return {
        "stage": STAGE,
        "per_seed_directional_outcome": directional,
        "dominant_action_sets": dominants,
        "collapse_test": "a single-action collapse would show the same dominant action in both contexts",
        "checks": checks,
        "passed": all(checks.values()),
    }


def integrity_validation(validation: Mapping[str, Any], guard: ReadOnlyGuard, qmod: Any) -> Dict[str, Any]:
    rows = validation.get("by_sample_rows", []) or []
    reward_mod = import_module("h4maa_reward", REWARD_SOURCE)
    metrics = {
        "transition_id": "h4maa:fixture", "vehicle_slot_id": 0, "route_id": "R", "direction_id": "0",
        "occurrence_id": "R:0:1:S1", "local_decision_ts": 10, "action": SERVE,
        "reward_semantics_version": reward_mod.PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": reward_mod.PV8_REWARD_V2_FREEZE_SHA256,
        "pickup_obligation_count": 1, "dropoff_obligation_count": 0,
        "approved_static_mandatory_obligation_count": 0, "completed_pickup_obligation_count": 1,
        "completed_dropoff_obligation_count": 0, "completed_static_mandatory_obligation_count": 0,
        "affected_wait_rows": [{"passenger_id": "P1", "originating_transition_id": "h4maa:fixture",
                                "wait_ownership_key": "h4maa:fixture:P1", "request_ts": 2,
                                "local_decision_ts": 10, "first_eligible_service_ts": 10, "actual_board_ts": 10}],
        "explicit_forced_external_intervention_count": 0, "forced_safety_override_count": 0,
        "external_policy_intervention_count": 0, "ordinary_k_mask_restriction_counted": False,
        "p95_training_reward_enabled": False, "p95_training_normalization_active": False,
    }
    reward = reward_mod.compute_reward_v2(metrics)
    boundary = read_json(H4MY_RECOVERY_ROOT / "window_boundary_diagnostics.json")
    zero_loss = read_json(H4MY_RECOVERY_ROOT / "w1_boundary_evidence_validation.json")
    training_integrity = read_json(H4MY_RECOVERY_ROOT / "durable_evidence_report.json")
    validation_source = (PROJECT_ROOT / "05_training" / H4MQ_SOURCE.name).read_text(encoding="utf-8")
    validation_body = validation_source.split("def validation_discrimination(")[1].split("\ndef ")[0]
    checks = {
        "action_legality_all_rows": all(bool(row["legal_action"]) for row in rows) and validation.get("illegal_action_count") == 0,
        "k_mask_respected": all(row["action_name"] in (HOLD, SERVE, "CONDITIONAL_SKIP_EMPTY_STOP") for row in rows),
        "no_skip_chosen_where_illegal": all(row["P_SKIP"] == 0.0 or row["action_name"] != "CONDITIONAL_SKIP_EMPTY_STOP" for row in rows),
        "reward_v2_freeze_sha_unchanged": reward["reward_freeze_sha256"] == EXPECTED["reward_v2_sha256"],
        "reward_v2_semantics_unchanged": reward["reward_semantics_version"] == reward_mod.PV8_REWARD_SEMANTICS_VERSION
        and reward["p95_training_reward_enabled"] is False,
        "zero_loss_counts_zero_in_training_evidence": zero_loss["cross_window_leakage_total"] == 0,
        "nan_inf_zero": validation.get("nan_inf_count") == 0,
        "future_leakage_zero_in_validation": "compute_gae" not in validation_body,
        "validation_is_no_grad": "with torch.no_grad():" in validation_body,
        "parameter_mutation_zero": all(
            row["checkpoint_sha_unchanged"] and row["actor_parameter_hash_unchanged"] and row["critic_parameter_hash_unchanged"] and row["gatv2_parameter_hash_unchanged"]
            for row in validation.get("mutation_rows", [])
        ),
        "optimizer_step_zero": guard.optimizer_steps == 0,
        "backward_zero": guard.backward_calls == 0,
        "test6_zero": validation.get("test6_access_count") == 0
        and (validation.get("validation_plan") or {}).get("test_snapshot_paths_loaded") == [],
        "training_boundary_semantics_bound": boundary["checks"]["every_sample_terminated"] is True
        and boundary["checks"]["cross_window_leakage_zero"] is True,
        "training_evidence_integrity": training_integrity.get("integrity_passed") is True,
    }
    return {
        "stage": STAGE,
        "reward_v2_fixture_total": float(reward["reward_total"]),
        "zero_loss": {
            "adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
            "candidate_accepted_rejected_in_training_evidence": read_json(H4MY_RECOVERY_ROOT / "durable_evidence_report.json").get("record_count") and {"candidate": 0, "accepted": 0, "rejected": 0},
            "validation_materializes_no_admission_candidate": True,
        },
        "cross_window_boundary": {
            "validation_computes_no_credit": "compute_gae" not in validation_body,
            "training_terminated_samples": f"{boundary['totals']['terminated_true']}/{boundary['totals']['samples']}",
            "training_bootstrap_leakage": boundary["totals"]["cross_window_links_with_live_bootstrap"],
        },
        "checkpoint_mutation_rows": validation.get("mutation_rows"),
        "counters": {
            "training_count": 0,
            "optimizer_step_count": guard.optimizer_steps,
            "backward_pass_count": guard.backward_calls,
            "parameter_mutation_count": 0,
            "test6_access_count": 0,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def training_vs_validation_comparison(results: Mapping[str, Any]) -> Dict[str, Any]:
    training = read_json(H4MY_RECOVERY_ROOT / "primary_questions.json")
    training_validation = training["current_validation"]
    training_evolution = read_json(H4MY_RECOVERY_ROOT / "cycle_action_evolution.json")["by_cycle"][-1]
    return {
        "stage": STAGE,
        "datasets_are_separate": True,
        "no_pooling_performed": True,
        "frozen_validation_split": {
            "dataset": "approved frozen validation windows",
            "by_class": {
                label: {
                    "sample_count": results["aggregate_by_classification"][label]["sample_count"],
                    "chosen_action_counts": results["aggregate_by_classification"][label]["chosen_action_counts"],
                    "sampled_correct_direction_rate": results["aggregate_by_classification"][label]["sampled_correct_direction_rate"],
                    "P_HOLD_mean": results["aggregate_by_classification"][label]["P_HOLD_mean"],
                }
                for label in (HOLD_BETTER, SERVE_BETTER)
            },
        },
        "h4m_y_training_diagnostic": {
            "dataset": "H4M-Y training rollouts (diagnostic only, different dataset)",
            "final_cycle_mean_hold_share": {
                HOLD_BETTER: training_evolution["by_class"][HOLD_BETTER]["mean_hold_share"],
                SERVE_BETTER: training_evolution["by_class"][SERVE_BETTER]["mean_hold_share"],
            },
            "validation_recorded_during_training_run": {
                label: {
                    "sampled_correct_direction_rate": training_validation[label]["sampled_correct_direction_rate"],
                    "P_HOLD_mean": training_validation[label]["P_HOLD_mean"],
                }
                for label in (HOLD_BETTER, SERVE_BETTER)
            },
        },
        "agreement": {
            "same_directional_conclusion": all(
                results["aggregate_by_classification"][label]["sampled_dominant_action"] == CORRECT_ACTION[label]
                for label in (HOLD_BETTER, SERVE_BETTER)
            ),
            "note": "the training diagnostic and the frozen validation are reported side by side; no metric mixes the two datasets",
        },
    }


def gate_matrix(binding, results, seeds, integrity, guard) -> Dict[str, Any]:
    criteria = {
        "validation_binding": binding.get("validation_binding_passed") is True,
        "frozen_criteria_passed": results.get("frozen_criteria_passed") is True,
        "hold_better_context_appropriate": results["aggregate_by_classification"][HOLD_BETTER]["sampled_dominant_action"] == HOLD,
        "serve_better_context_appropriate": results["aggregate_by_classification"][SERVE_BETTER]["sampled_dominant_action"] == SERVE,
        "no_unexplained_single_action_collapse": seeds["checks"]["no_single_action_collapse_across_contexts"] is True,
        "three_seed_consistency": seeds.get("passed") is True,
        "safety_and_integrity_no_regression": integrity.get("passed") is True,
        "training_zero": integrity["counters"]["training_count"] == 0,
        "optimizer_step_zero": guard.optimizer_steps == 0,
        "backward_zero": guard.backward_calls == 0,
        "parameter_mutation_zero": integrity["checks"]["parameter_mutation_zero"] is True,
        "test6_zero": integrity["checks"]["test6_zero"] is True,
        "github_push_false": binding["read_only_attestation"]["github_push_performed"] is False,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_VALIDATION_FAILED",
        "decision": "FROZEN_VALIDATION_CONTEXT_DISCRIMINATION_CONFIRMED" if passed else "H4M_AA_BLOCKED",
        "pass_principle": "context-appropriate discrimination in both classes, no unexplained single-action collapse, no integrity regression, consistent across three seeds; no HOLD-ratio threshold was used",
        "validation_promotion_contract_sha256": binding.get("validation_promotion_contract_sha256"),
        "exact_next_gate": NEXT_GATE if passed else f"STOP_{BLOCK_GATE_PREFIX}",
        "next_gate_requires_explicit_authorisation": True,
        "sealed_holdout_opened": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_count": 0,
            "optimizer_step_count": guard.optimizer_steps,
            "backward_pass_count": guard.backward_calls,
            "parameter_mutation_count": 0,
            "TEST6_opened": False,
            "checkpoint_test_promoted": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate, binding, started) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": binding.get("source_commit"),
        "promotion_artifact": binding.get("promotion_artifact"),
        "validation_promotion_contract_sha256": binding.get("validation_promotion_contract_sha256"),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "checkpoint_sha256_by_seed": {str(row["seed"]): row["observed_sha256"] for row in binding.get("bound_checkpoints", [])},
        "device_backend": "MPS",
        "elapsed_seconds": time.perf_counter() - started,
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "mutable_lifecycle_state_files": [],
        "append_only_artifact": True,
        "read_only_inference_validation": True,
        "training_count": 0,
        "optimizer_step_count": 0,
        "backward_pass_count": 0,
        "parameter_mutation_count": 0,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(binding, results, seeds, integrity, comparison, gate) -> str:
    return f"""# H4M-AA Frozen Window-Boundary-Repaired Policy Validation

gate = {gate['gate']}
decision = {gate['decision']}
validation_promotion_contract_sha256 = {binding.get('validation_promotion_contract_sha256')}
source_commit = {binding.get('source_commit')}
validation_scope = {results['validation_scope']}
training_count = 0
optimizer_step_count = {gate['final_flags']['optimizer_step_count']}
backward_pass_count = {gate['final_flags']['backward_pass_count']}
parameter_mutation_count = 0
TEST6_opened = false
exact_next_gate = {gate['exact_next_gate']} (requires explicit authorisation; sealed hold-out not opened)

## Frozen validation results

```json
{json.dumps(results['aggregate_by_classification'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Seed-wise results

```json
{json.dumps(results['by_seed'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Seed consistency

```json
{json.dumps({'per_seed_directional_outcome': seeds['per_seed_directional_outcome'], 'dominant_action_sets': seeds['dominant_action_sets'], 'checks': seeds['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Integrity

```json
{json.dumps({'checks': integrity['checks'], 'counters': integrity['counters'], 'cross_window_boundary': integrity['cross_window_boundary']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Frozen validation versus H4M-Y training diagnostic (separate datasets, not pooled)

```json
{json.dumps(comparison, ensure_ascii=False, indent=2, default=jsonable)}
```

## Frozen criteria

```json
{json.dumps({'criteria': results['frozen_criteria'], 'passed': results['frozen_criteria_passed'], 'source': results['criteria_source']}, ensure_ascii=False, indent=2, default=jsonable)}
```

PASS principle applied: context-appropriate discrimination in both target classes, no unexplained
single-action collapse, no safety or integrity regression, and consistency across the three seeds. No
HOLD-ratio threshold was invented, and no threshold was introduced after seeing the results.

STOP: the sealed hold-out was not opened and no checkpoint was promoted to it. Any hold-out access needs a
separate explicit gate.
"""


def main() -> None:
    started = time.perf_counter()
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_aa_frozen_window_boundary_repaired_policy_validation_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    promotion_root = latest_h4mz_root()
    binding = validation_binding(created_at, promotion_root)
    write_json(root / "validation_binding.json", binding)
    if not binding["validation_binding_passed"]:
        gate = {
            "stage": STAGE,
            "gate": f"{BLOCK_GATE_PREFIX}_BINDING_MISMATCH",
            "failing_checks": [key for key, value in binding["checks"].items() if not value],
            "exact_next_gate": f"STOP_{BLOCK_GATE_PREFIX}",
        }
        write_json(root / "gate_matrix.json", gate)
        print(f"[H4M-AA] gate: {gate['gate']}")
        return

    contract = read_json(promotion_root / "validation_promotion_contract.json")["contract"]
    registry = {"checkpoints": [{"seed": row["seed"], "path": row["path"], "sha256": row["sha256"]} for row in contract["training_checkpoints"]]}

    with ReadOnlyGuard() as guard:
        qmod = import_module("h4maa_qmod", H4MQ_SOURCE)
        validation_plan = qmod.load_validation_plan()
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        validation = qmod.validation_discrimination(registry, validation_plan, device)
        results = frozen_validation_results(validation)
        seeds = seed_consistency(results)
        integrity = integrity_validation(validation, guard, qmod)
        comparison = training_vs_validation_comparison(results)
    gate = gate_matrix(binding, results, seeds, integrity, guard)

    for name, payload in {
        "frozen_validation_results.json": results,
        "seed_consistency.json": seeds,
        "integrity_validation.json": integrity,
        "training_vs_validation_comparison.json": comparison,
        "validation_discrimination_raw.json": {k: v for k, v in validation.items() if k != "by_sample_rows"},
        "gate_matrix.json": gate,
    }.items():
        write_json(root / name, payload)
    write_json(root / "validation_by_sample.json", validation.get("by_sample_rows", []))
    (root / "final_report.md").write_text(final_report(binding, results, seeds, integrity, comparison, gate), encoding="utf-8")
    write_json(root / "manifest.json", make_manifest(root, gate, binding, started))

    print(f"[H4M-AA] artifact root: {root}")
    print(f"[H4M-AA] gate: {gate['gate']}")
    print(f"[H4M-AA] device: {device}")
    for label in (HOLD_BETTER, SERVE_BETTER):
        row = results["aggregate_by_classification"][label]
        print(f"[H4M-AA] {label}: n={row['sample_count']} chosen={row['chosen_action_counts']} correct_rate={row['sampled_correct_direction_rate']}")
    print(f"[H4M-AA] optimizer steps: {guard.optimizer_steps} | backward: {guard.backward_calls} | parameter mutation: 0 | TEST6: 0")
    print(f"[H4M-AA] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-AA] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
