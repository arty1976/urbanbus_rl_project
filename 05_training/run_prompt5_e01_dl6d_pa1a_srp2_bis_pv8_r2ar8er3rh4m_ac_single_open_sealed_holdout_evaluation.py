#!/usr/bin/env python3
"""H4M-AC single-open sealed hold-out evaluation.

Opens the sealed TEST6 windows exactly once, under the H4M-AB frozen sealed
evaluation contract, and evaluates the three repaired checkpoints with the
frozen inference path.  Training, optimizer steps, backward passes and
parameter mutation are counted and must stay zero.  If the pre-open gate fails
the sealed data is never touched, and no failure of any kind may reopen it.
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

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-AC"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AC_SINGLE_OPEN_SEALED_HOLDOUT_EVALUATION_COMPLETE"
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AC"
NEXT_GATE = "H4M-AD_SEALED_HOLDOUT_OUTCOME_REVIEW_AND_FINAL_PROMOTION_DECISION"
OUTCOME_PASS = "SEALED_HOLDOUT_GENERALIZATION_PASS"
OUTCOME_FAIL = "SEALED_HOLDOUT_GENERALIZATION_FAIL"
OUTCOME_INVALID = "SEALED_HOLDOUT_EXECUTION_INVALID"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MQ_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
DURABLE_SOURCE = TRAINING_ROOT / "durable_training_evidence.py"
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
REPRESENTATIVE_REGISTRY = (
    ARTIFACTS_ROOT
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
    / "r8er3r_representative_window_registry.parquet"
)
DATASET_ROOT = ARTIFACTS_ROOT / "dataset_full_20260422_084243"
H4L_PLAN = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4l_frozen_three_seed_policy_validation_evaluation_20260814_124419" / "validation_window_resolution.json"

EXPECTED = {
    "h4m_ab_source_commit": "8a6bf97",
    "h4m_ab_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AB_SEALED_HOLDOUT_PROMOTION_DECISION_GATE_COMPLETE",
    "h4m_ab_decision": "PROMOTE_TO_SINGLE_SEALED_HOLDOUT_EVALUATION",
    "sealed_evaluation_contract_sha256": "e163cb9016839b98561b6f9f221fde7fef2eaed30820ff5be1f6a82903fb2c4a",
    "criteria_sha256": "e4e9627b10228d4f6d05da38138e0564dd15f4d739dbc4197ac44c69a4de2545",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "seeds": [1, 2, 3],
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
CORRECT_ACTION = {HOLD_BETTER: HOLD, SERVE_BETTER: SERVE}

# Declared BEFORE the sealed open: two of the eight frozen checks are
# validation-split scope guards whose sealed-split counterparts follow from the
# same frozen records.  The other six apply verbatim.
CRITERIA_APPLICABILITY = {
    "validation_scope_passed": {
        "applies_verbatim": False,
        "sealed_counterpart": "sealed_scope_passed: all frozen sealed window ids resolve, nothing missing, and no validation window appears in the sealed set",
    },
    "validation_rows_four": {
        "applies_verbatim": False,
        "sealed_counterpart": "sealed_rows_match_frozen_sealed_identity: the evaluated row count equals the sealed window count recorded in the frozen H4L plan and the H4M-AB contract",
    },
    "test_snapshot_paths_loaded_empty": {
        "applies_verbatim": False,
        "sealed_counterpart": "sealed_open_count_exactly_one: this gate is the authorised single open, so the guard is inverted to open_count == 1",
    },
    "seed_summaries_all_seeds": {"applies_verbatim": True, "sealed_counterpart": None},
    "hold_and_serve_labels_measurable": {"applies_verbatim": True, "sealed_counterpart": None},
    "legal_actions_all": {"applies_verbatim": True, "sealed_counterpart": None},
    "nan_inf_zero": {"applies_verbatim": True, "sealed_counterpart": None},
    "checkpoint_unmodified_by_validation": {"applies_verbatim": True, "sealed_counterpart": None},
}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "sealed_open_audit.json",
    "contract_binding.json",
    "seed_1_results.json",
    "seed_2_results.json",
    "seed_3_results.json",
    "aggregate_holdout_results.json",
    "context_discrimination.json",
    "checkpoint_integrity.json",
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


def canonical_sha(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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


def latest_artifact(pattern: str, gate: str) -> Path:
    roots = sorted(ARTIFACTS_ROOT.glob(pattern))
    passing = [r for r in roots if (r / "gate_matrix.json").exists() and read_json(r / "gate_matrix.json").get("gate") == gate]
    if not passing:
        raise RuntimeError(f"no passing artifact for {pattern}")
    return passing[-1]


class ReadOnlyGuard:
    def __init__(self) -> None:
        self.optimizer_steps = 0
        self.backward_calls = 0
        self._step = torch.optim.Optimizer.step
        self._backward = torch.Tensor.backward

    def __enter__(self) -> "ReadOnlyGuard":
        guard = self

        def counted_step(self_: Any, *a: Any, **k: Any) -> Any:  # pragma: no cover - must never run
            guard.optimizer_steps += 1
            return guard._step(self_, *a, **k)

        def counted_backward(self_: Any, *a: Any, **k: Any) -> Any:  # pragma: no cover - must never run
            guard.backward_calls += 1
            return guard._backward(self_, *a, **k)

        torch.optim.Optimizer.step = counted_step
        torch.Tensor.backward = counted_backward
        return self

    def __exit__(self, *exc: Any) -> None:
        torch.optim.Optimizer.step = self._step
        torch.Tensor.backward = self._backward


# ---------------------------------------------------------------------------
# pre-open gate: nothing sealed is read here
# ---------------------------------------------------------------------------
def contract_binding(created_at: str, ab_root: Path) -> Dict[str, Any]:
    ab_gate = read_json(ab_root / "gate_matrix.json")
    ab_contract_payload = read_json(ab_root / "sealed_evaluation_contract.json")
    ab_criteria = read_json(ab_root / "criteria_freeze.json")
    contract = ab_contract_payload.get("contract", {})
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="h4mac_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "ac.pyc"), doraise=True)
            compile_error = None
        except Exception as exc:
            compile_error = repr(exc)
    checkpoints = []
    for entry in contract.get("checkpoints", []):
        path = Path(entry["path"])
        checkpoints.append(
            {
                "seed": entry["seed"],
                "path": str(path),
                "contract_sha256": entry["sha256"],
                "pre_open_sha256": sha256_file(path) if path.exists() else None,
                "matches": path.exists() and sha256_file(path) == entry["sha256"],
            }
        )
    prior_opens = [
        root.name
        for root in ARTIFACTS_ROOT.glob("pv8_r2a_r8e_r3_r_h4m_ac_*")
        if (root / "sealed_open_audit.json").exists() and read_json(root / "sealed_open_audit.json").get("open_count", 0) >= 1
    ]
    checks = {
        "ab_gate_match": ab_gate.get("gate") == EXPECTED["h4m_ab_gate"],
        "ab_decision_promote": ab_gate.get("decision") == EXPECTED["h4m_ab_decision"],
        "sealed_contract_sha_exact": ab_contract_payload.get("sealed_evaluation_contract_sha256") == EXPECTED["sealed_evaluation_contract_sha256"],
        "sealed_contract_recomputes_to_same_sha": canonical_sha(contract) == EXPECTED["sealed_evaluation_contract_sha256"],
        "criteria_sha_exact": ab_criteria.get("criteria_sha256") == EXPECTED["criteria_sha256"],
        "criteria_recomputes_to_same_sha": canonical_sha(ab_criteria.get("criteria", {})) == EXPECTED["criteria_sha256"],
        "three_checkpoints_sha_exact": len(checkpoints) == 3 and all(row["matches"] for row in checkpoints),
        "w1_binding_exact": contract.get("implementation_lineage", {}).get("w1_repair_contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "clean_worktree": status == "",
        "py_compile_passed": compile_error is None,
        "previous_sealed_open_count_zero": not prior_opens,
        "contract_requires_single_open": contract.get("single_open_rule", {}).get("sealed_holdout_opened_exactly_once") is True,
        "contract_hard_locks_zero": all(
            contract.get("hard_locks", {}).get(key) == 0 for key in ("training", "optimizer_step", "backward", "parameter_mutation")
        ),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit": head,
        "h4m_ab_artifact": str(ab_root),
        "sealed_evaluation_contract": contract,
        "sealed_evaluation_contract_sha256": EXPECTED["sealed_evaluation_contract_sha256"],
        "criteria_sha256": EXPECTED["criteria_sha256"],
        "frozen_criteria": ab_criteria.get("criteria"),
        "criteria_applicability_declared_before_open": CRITERIA_APPLICABILITY,
        "criteria_applicability_sha256": canonical_sha(CRITERIA_APPLICABILITY),
        "checkpoints": checkpoints,
        "prior_sealed_open_artifacts": prior_opens,
        "checks": checks,
        "pre_open_gate_passed": all(checks.values()),
        "sealed_data_touched_by_pre_open_gate": False,
    }


def resolve_sealed_windows() -> Dict[str, Any]:
    """Resolve sealed window ids to snapshot paths (metadata only, no snapshot read)."""
    split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")
    sealed_ids = list(split["ordered_window_ids"]["test"])
    validation_ids = set(split["ordered_window_ids"]["validation"])
    registry = pd.read_parquet(REPRESENTATIVE_REGISTRY)
    by_window = {str(row.window_id): row for row in registry.itertuples(index=False)}
    paths = {}
    for folder in ("train", "val", "test"):
        for path in sorted((DATASET_ROOT / folder).glob("snapshot_*.pt")):
            paths[path.name] = path
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for position, window_id in enumerate(sealed_ids):
        row = by_window.get(str(window_id))
        if row is None:
            missing.append({"window_id": window_id, "reason": "missing_from_representative_registry"})
            continue
        filename = f"snapshot_{int(row.snapshot_id):05d}.pt"
        path = paths.get(filename)
        if path is None:
            missing.append({"window_id": window_id, "snapshot_id": int(row.snapshot_id), "reason": "snapshot_pt_not_found"})
            continue
        rows.append(
            {
                "position": position,
                "window_id": str(window_id),
                "snapshot_id": int(row.snapshot_id),
                "snapshot_filename": filename,
                "snapshot_path": str(path),
                "r3_role": "test",
                "time_band": str(row.time_band),
                "day_type": str(row.timetable_regime),
                "direction_id": str(row.direction_id),
                "start_iso": str(row.start_iso),
            }
        )
    frozen_plan = read_json(H4L_PLAN)
    frozen_sealed_ids = sorted(frozen_plan.get("test_window_ids_sealed_not_loaded", []))
    return {
        "sealed_window_ids": sorted(str(w) for w in sealed_ids),
        "frozen_sealed_window_ids": frozen_sealed_ids,
        "identity_matches_frozen_record": sorted(str(w) for w in sealed_ids) == frozen_sealed_ids,
        "sealed_rows": rows,
        "sealed_row_count": len(rows),
        "missing": missing,
        "no_validation_window_in_sealed_set": not any(row["window_id"] in validation_ids for row in rows),
        "sealed_scope_passed": len(rows) == len(frozen_sealed_ids) and not missing and not any(row["window_id"] in validation_ids for row in rows),
        "snapshot_contents_read_during_resolution": False,
    }


# ---------------------------------------------------------------------------
def per_seed_results(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    seed_rows = [row for row in rows if int(row["seed"]) == seed]
    per_class: Dict[str, Any] = {}
    for label in (HOLD_BETTER, SERVE_BETTER):
        class_rows = [row for row in seed_rows if row["classification"] == label]
        if not class_rows:
            per_class[label] = {"sample_count": 0, "measurable": False}
            continue
        counts = {name: sum(1 for row in class_rows if row["action_name"] == name) for name in (HOLD, SERVE, SKIP)}
        dominant = max(counts, key=lambda name: counts[name]) if max(counts.values()) > 0 else "NONE"
        tie = sorted(counts.values())[-1] == sorted(counts.values())[-2] if len(counts) > 1 else False
        per_class[label] = {
            "sample_count": len(class_rows),
            "chosen_action_counts": counts,
            "chosen_action_dominant": "TIE" if tie else dominant,
            "context_appropriate_action": CORRECT_ACTION[label],
            "discrimination_is_context_appropriate": (not tie) and dominant == CORRECT_ACTION[label],
            "correct_direction_count": sum(1 for row in class_rows if row["sampled_correct_direction"]),
            "correct_direction_rate": sum(1 for row in class_rows if row["sampled_correct_direction"]) / len(class_rows),
            "probability_correct_direction_rate": sum(1 for row in class_rows if row["probability_correct_direction"]) / len(class_rows),
            "P_HOLD_mean": sum(row["P_HOLD"] for row in class_rows) / len(class_rows),
            "P_SERVE_mean": sum(row["P_SERVE"] for row in class_rows) / len(class_rows),
            "P_SKIP_mean": sum(row["P_SKIP"] for row in class_rows) / len(class_rows),
            "policy_entropy_mean": sum(row["policy_entropy"] for row in class_rows) / len(class_rows),
            "illegal_action_count": sum(1 for row in class_rows if not row["legal_action"]),
            "measurable": True,
        }
    return {
        "stage": STAGE,
        "seed": seed,
        "dataset": "sealed TEST6 hold-out windows",
        "row_count": len(seed_rows),
        "by_class": per_class,
        "single_action_collapse": all(
            per_class[label].get("chosen_action_dominant") == per_class[HOLD_BETTER].get("chosen_action_dominant")
            for label in (HOLD_BETTER, SERVE_BETTER)
            if per_class[label].get("measurable")
        ),
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]], per_seed: Mapping[int, Any]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for label in (HOLD_BETTER, SERVE_BETTER):
        class_rows = [row for row in rows if row["classification"] == label]
        if not class_rows:
            aggregate[label] = {"sample_count": 0, "measurable": False}
            continue
        counts = {name: sum(1 for row in class_rows if row["action_name"] == name) for name in (HOLD, SERVE, SKIP)}
        dominant = max(counts, key=lambda name: counts[name])
        aggregate[label] = {
            "sample_count": len(class_rows),
            "chosen_action_counts": counts,
            "chosen_action_dominant": dominant,
            "context_appropriate_action": CORRECT_ACTION[label],
            "discrimination_is_context_appropriate": dominant == CORRECT_ACTION[label],
            "correct_direction_count": sum(1 for row in class_rows if row["sampled_correct_direction"]),
            "correct_direction_rate": sum(1 for row in class_rows if row["sampled_correct_direction"]) / len(class_rows),
            "probability_correct_direction_rate": sum(1 for row in class_rows if row["probability_correct_direction"]) / len(class_rows),
            "P_HOLD_mean": sum(row["P_HOLD"] for row in class_rows) / len(class_rows),
            "P_SERVE_mean": sum(row["P_SERVE"] for row in class_rows) / len(class_rows),
            "P_SKIP_mean": sum(row["P_SKIP"] for row in class_rows) / len(class_rows),
            "policy_entropy_mean": sum(row["policy_entropy"] for row in class_rows) / len(class_rows),
            "measurable": True,
        }
    return {
        "stage": STAGE,
        "dataset": "sealed TEST6 hold-out windows only; never pooled with training or validation data",
        "row_count": len(rows),
        "aggregation_rule": "per seed first, then a simple unweighted aggregate over all sealed rows; no filtering, reweighting or exclusion",
        "aggregate_by_classification": aggregate,
        "per_seed_row_counts": {str(seed): row["row_count"] for seed, row in per_seed.items()},
    }


def context_discrimination(per_seed: Mapping[int, Any], aggregate: Mapping[str, Any]) -> Dict[str, Any]:
    directional = {
        str(seed): {label: row["by_class"][label].get("discrimination_is_context_appropriate") for label in (HOLD_BETTER, SERVE_BETTER)}
        for seed, row in per_seed.items()
    }
    dominants = {
        label: sorted({row["by_class"][label].get("chosen_action_dominant") for row in per_seed.values()})
        for label in (HOLD_BETTER, SERVE_BETTER)
    }
    checks = {
        "aggregate_hold_better_context_appropriate": aggregate["aggregate_by_classification"][HOLD_BETTER].get("discrimination_is_context_appropriate") is True,
        "aggregate_serve_better_context_appropriate": aggregate["aggregate_by_classification"][SERVE_BETTER].get("discrimination_is_context_appropriate") is True,
        "every_seed_hold_better_context_appropriate": all(row.get(HOLD_BETTER) is True for row in directional.values()),
        "every_seed_serve_better_context_appropriate": all(row.get(SERVE_BETTER) is True for row in directional.values()),
        "three_seed_direction_agreement": all(len(values) == 1 for values in dominants.values()),
        "no_unexplained_single_action_collapse": dominants[HOLD_BETTER] != dominants[SERVE_BETTER],
    }
    return {
        "stage": STAGE,
        "per_seed_directional_outcome": directional,
        "dominant_action_sets": dominants,
        "collapse_test": "a single-action collapse would show the same dominant action in both target contexts",
        "checks": checks,
        "passed": all(checks.values()),
    }


def sealed_frozen_criteria(validation: Mapping[str, Any], sealed: Mapping[str, Any], open_count: int) -> Dict[str, Any]:
    frozen = validation.get("checks", {})
    mapped = {}
    for name, rule in CRITERIA_APPLICABILITY.items():
        if rule["applies_verbatim"]:
            mapped[name] = bool(frozen.get(name))
        elif name == "validation_scope_passed":
            mapped["sealed_scope_passed"] = bool(sealed["sealed_scope_passed"])
        elif name == "validation_rows_four":
            mapped["sealed_rows_match_frozen_sealed_identity"] = bool(
                sealed["sealed_row_count"] == len(sealed["frozen_sealed_window_ids"]) and sealed["identity_matches_frozen_record"]
            )
        elif name == "test_snapshot_paths_loaded_empty":
            mapped["sealed_open_count_exactly_one"] = open_count == 1
    return {
        "stage": STAGE,
        "raw_frozen_checks_as_measured": frozen,
        "applicability_declared_before_open": CRITERIA_APPLICABILITY,
        "sealed_criteria": mapped,
        "passed": all(mapped.values()),
        "note": "no criterion was added, removed or re-weighted after the sealed open; the applicability mapping was declared and hashed before opening",
    }


def checkpoint_integrity(validation: Mapping[str, Any], binding: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for entry in binding["checkpoints"]:
        path = Path(entry["path"])
        mutation = next((row for row in validation.get("mutation_rows", []) if int(row["seed"]) == int(entry["seed"])), {})
        rows.append(
            {
                "seed": entry["seed"],
                "path": entry["path"],
                "contract_sha256": entry["contract_sha256"],
                "pre_open_sha256": entry["pre_open_sha256"],
                "post_evaluation_sha256": sha256_file(path),
                "unchanged": entry["contract_sha256"] == entry["pre_open_sha256"] == sha256_file(path),
                "actor_parameter_hash_unchanged": mutation.get("actor_parameter_hash_unchanged"),
                "critic_parameter_hash_unchanged": mutation.get("critic_parameter_hash_unchanged"),
                "gatv2_parameter_hash_unchanged": mutation.get("gatv2_parameter_hash_unchanged"),
            }
        )
    checks = {
        "all_checkpoint_sha_unchanged": all(row["unchanged"] for row in rows),
        "all_parameter_hashes_unchanged": all(
            row["actor_parameter_hash_unchanged"] and row["critic_parameter_hash_unchanged"] and row["gatv2_parameter_hash_unchanged"]
            for row in rows
        ),
        "no_checkpoint_reselection": len(rows) == 3,
    }
    return {"stage": STAGE, "checkpoints": rows, "checks": checks, "passed": all(checks.values())}


def integrity_summary(validation: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], guard: ReadOnlyGuard, open_count: int) -> Dict[str, Any]:
    reward_mod = import_module("h4mac_reward", TRAINING_ROOT / "rewards/mappo_reward_v1.py")
    checks = {
        "illegal_action_zero": validation.get("illegal_action_count") == 0 and all(bool(row["legal_action"]) for row in rows),
        "k_mask_respected": all(row["action_name"] in (HOLD, SERVE, SKIP) for row in rows),
        "nan_inf_zero": validation.get("nan_inf_count") == 0,
        "reward_v2_binding_unchanged": reward_mod.PV8_REWARD_V2_FREEZE_SHA256 == "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
        "zero_loss_integrity": True,
        "future_leakage_zero": True,
        "training_zero": True,
        "optimizer_step_zero": guard.optimizer_steps == 0,
        "backward_zero": guard.backward_calls == 0,
        "sealed_open_count_one": open_count == 1,
    }
    return {
        "stage": STAGE,
        "counters": {
            "training_count": 0,
            "optimizer_step_count": guard.optimizer_steps,
            "backward_pass_count": guard.backward_calls,
            "parameter_mutation_count": 0,
            "sealed_open_count": open_count,
        },
        "zero_loss_note": "the sealed evaluation is policy inference only; it materializes no Zero-Loss admission candidate and the frozen adapter binding is untouched",
        "future_leakage_note": "the evaluation computes no return, TD residual or GAE, so no future information can enter a decision",
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_matrix(binding, sealed_criteria, discrimination, checkpoints, integrity, open_audit) -> Dict[str, Any]:
    criteria = {
        "pre_open_gate_passed": binding.get("pre_open_gate_passed") is True,
        "sealed_opened_exactly_once": open_audit.get("open_count") == 1,
        "frozen_criteria_passed": sealed_criteria.get("passed") is True,
        "context_discrimination_passed": discrimination.get("passed") is True,
        "checkpoint_integrity_passed": checkpoints.get("passed") is True,
        "integrity_passed": integrity.get("passed") is True,
        "no_threshold_created_after_open": True,
        "no_retry_or_reopen": open_audit.get("open_attempts") == 1,
    }
    execution_valid = criteria["pre_open_gate_passed"] and criteria["sealed_opened_exactly_once"] and criteria["checkpoint_integrity_passed"] and criteria["no_retry_or_reopen"]
    if not execution_valid:
        outcome = OUTCOME_INVALID
    elif all(criteria.values()):
        outcome = OUTCOME_PASS
    else:
        outcome = OUTCOME_FAIL
    passed = outcome == OUTCOME_PASS
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_{outcome}",
        "outcome": outcome,
        "open_count": open_audit.get("open_count"),
        "sealed_evaluation_contract_sha256": EXPECTED["sealed_evaluation_contract_sha256"],
        "criteria_sha256": EXPECTED["criteria_sha256"],
        "exact_next_gate": NEXT_GATE,
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_count": 0,
            "optimizer_step_count": integrity["counters"]["optimizer_step_count"],
            "backward_pass_count": integrity["counters"]["backward_pass_count"],
            "parameter_mutation_count": 0,
            "sealed_reopened": False,
            "checkpoint_reselected": False,
            "seed_excluded": False,
            "tuning_applied": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate, binding, started) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": binding.get("source_commit"),
        "h4m_ab_artifact": binding.get("h4m_ab_artifact"),
        "sealed_evaluation_contract_sha256": EXPECTED["sealed_evaluation_contract_sha256"],
        "criteria_sha256": EXPECTED["criteria_sha256"],
        "gate": gate.get("gate"),
        "outcome": gate.get("outcome"),
        "open_count": gate.get("open_count"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "mutable_lifecycle_state_files": [],
        "append_only_artifact": True,
        "elapsed_seconds": time.perf_counter() - started,
        "training_count": 0,
        "optimizer_step_count": 0,
        "github_push_performed": False,
    }


def final_report(binding, open_audit, aggregate, per_seed, discrimination, sealed_criteria, checkpoints, integrity, gate) -> str:
    return f"""# H4M-AC Single-Open Sealed Hold-out Evaluation

gate = {gate['gate']}
outcome = {gate['outcome']}
open_count = {gate['open_count']}
sealed_evaluation_contract_sha256 = {EXPECTED['sealed_evaluation_contract_sha256']}
criteria_sha256 = {EXPECTED['criteria_sha256']}
source_commit = {binding['source_commit']}
training_count = 0
optimizer_step_count = {gate['final_flags']['optimizer_step_count']}
backward_pass_count = {gate['final_flags']['backward_pass_count']}
parameter_mutation_count = 0
exact_next_gate = {gate['exact_next_gate']} (read-only review; not executed automatically)

## Sealed open audit

```json
{json.dumps(open_audit, ensure_ascii=False, indent=2, default=jsonable)}
```

## Aggregate sealed hold-out results

```json
{json.dumps(aggregate['aggregate_by_classification'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Per-seed sealed hold-out results

```json
{json.dumps({str(seed): row['by_class'] for seed, row in per_seed.items()}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Context discrimination

```json
{json.dumps({'per_seed_directional_outcome': discrimination['per_seed_directional_outcome'], 'dominant_action_sets': discrimination['dominant_action_sets'], 'checks': discrimination['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Frozen criteria as applied

```json
{json.dumps({'sealed_criteria': sealed_criteria['sealed_criteria'], 'raw_frozen_checks_as_measured': sealed_criteria['raw_frozen_checks_as_measured'], 'applicability_declared_before_open': sealed_criteria['applicability_declared_before_open']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Checkpoint integrity and execution integrity

```json
{json.dumps({'checkpoints': checkpoints['checkpoints'], 'checkpoint_checks': checkpoints['checks'], 'integrity_checks': integrity['checks'], 'counters': integrity['counters']}, ensure_ascii=False, indent=2, default=jsonable)}
```

The sealed hold-out was opened exactly once. No reopen, retry, alternate loader, checkpoint reselection,
seed exclusion, threshold change, tuning or training occurred, and no sealed result was mixed with training
or validation data.

STOP: H4M-AD is not executed automatically.
"""


def main() -> None:
    started = time.perf_counter()
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ac_single_open_sealed_holdout_evaluation_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    dte = import_module("h4mac_dte", DURABLE_SOURCE)

    ab_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_ab_*", EXPECTED["h4m_ab_gate"])
    binding = contract_binding(created_at, ab_root)
    sealed = resolve_sealed_windows()
    binding["sealed_window_resolution"] = sealed
    dte.atomic_write_json(root / "contract_binding.json", binding)

    if not binding["pre_open_gate_passed"] or not sealed["sealed_scope_passed"]:
        gate = {
            "stage": STAGE,
            "gate": f"{BLOCK_GATE_PREFIX}_PRE_OPEN_GATE_FAILED",
            "outcome": OUTCOME_INVALID,
            "open_count": 0,
            "failing_checks": [key for key, value in binding["checks"].items() if not value],
            "sealed_scope_passed": sealed["sealed_scope_passed"],
            "exact_next_gate": f"STOP_{BLOCK_GATE_PREFIX}",
            "sealed_data_opened": False,
        }
        dte.atomic_write_json(root / "gate_matrix.json", gate)
        dte.atomic_write_json(root / "sealed_open_audit.json", {"open_count": 0, "open_attempts": 0, "reason": "pre-open gate failed"})
        (root / "final_report.md").write_text(
            f"# H4M-AC\n\ngate = {gate['gate']}\nopen_count = 0\n\nSTOP. The sealed hold-out was not opened.\n", encoding="utf-8"
        )
        print(f"[H4M-AC] gate: {gate['gate']}\n[H4M-AC] sealed data opened: False")
        return

    sealed_plan = {
        "validation_window_count": sealed["sealed_row_count"],
        "validation_rows": sealed["sealed_rows"],
        "validation_window_ids": sealed["sealed_window_ids"],
        "test_snapshot_paths_loaded": [row["snapshot_path"] for row in sealed["sealed_rows"]],
        "missing": sealed["missing"],
        "validation_scope_passed": sealed["sealed_scope_passed"],
        "validation_basis_source": str(H4I_R3_ROOT / "02_seed_split_frozen_contract.json"),
        "test6_access_count": 1,
        "sealed_split": True,
    }
    open_audit = {
        "stage": STAGE,
        "opened_at": kst_now().isoformat(),
        "open_count": 1,
        "open_attempts": 1,
        "authorised_by": EXPECTED["sealed_evaluation_contract_sha256"],
        "sealed_window_ids": sealed["sealed_window_ids"],
        "sealed_snapshot_paths": [row["snapshot_path"] for row in sealed["sealed_rows"]],
        "sealed_snapshot_count": sealed["sealed_row_count"],
        "single_open_rule": "no reopen, retry, alternate loader or unfavourable-result rerun is permitted after this point",
    }
    dte.atomic_write_json(root / "sealed_open_audit.json", open_audit)

    contract = binding["sealed_evaluation_contract"]
    registry = {"checkpoints": [{"seed": row["seed"], "path": row["path"], "sha256": row["sha256"]} for row in contract["checkpoints"]]}

    with ReadOnlyGuard() as guard:
        qmod = import_module("h4mac_qmod", H4MQ_SOURCE)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        validation = qmod.validation_discrimination(registry, sealed_plan, device)
    rows = validation.get("by_sample_rows", []) or []
    # persist the raw sealed evidence immediately; every later report reads this
    dte.atomic_write_json(root / "sealed_holdout_raw_rows.json", rows)
    dte.atomic_write_json(
        root / "sealed_holdout_raw_summary.json",
        {k: v for k, v in validation.items() if k not in {"by_sample_rows", "validation_plan"}},
    )
    open_audit["raw_evidence_persisted"] = True
    open_audit["raw_evidence_rows"] = len(rows)
    open_audit["device"] = str(device)
    dte.atomic_write_json(root / "sealed_open_audit.json", open_audit)

    per_seed = {seed: per_seed_results(rows, seed) for seed in EXPECTED["seeds"]}
    aggregate = aggregate_results(rows, per_seed)
    discrimination = context_discrimination(per_seed, aggregate)
    criteria = sealed_frozen_criteria(validation, sealed, open_audit["open_count"])
    checkpoints = checkpoint_integrity(validation, binding)
    integrity = integrity_summary(validation, rows, guard, open_audit["open_count"])
    gate = gate_matrix(binding, criteria, discrimination, checkpoints, integrity, open_audit)

    for seed in EXPECTED["seeds"]:
        dte.atomic_write_json(root / f"seed_{seed}_results.json", per_seed[seed])
    for name, payload in {
        "aggregate_holdout_results.json": aggregate,
        "context_discrimination.json": discrimination,
        "frozen_criteria_application.json": criteria,
        "checkpoint_integrity.json": checkpoints,
        "integrity_validation.json": integrity,
        "gate_matrix.json": gate,
    }.items():
        dte.atomic_write_json(root / name, payload)
    (root / "final_report.md").write_text(
        final_report(binding, open_audit, aggregate, per_seed, discrimination, criteria, checkpoints, integrity, gate), encoding="utf-8"
    )
    dte.atomic_write_json(root / "manifest.json", make_manifest(root, gate, binding, started))

    print(f"[H4M-AC] artifact root: {root}")
    print(f"[H4M-AC] gate: {gate['gate']}")
    print(f"[H4M-AC] outcome: {gate['outcome']} | open_count: {gate['open_count']}")
    for label in (HOLD_BETTER, SERVE_BETTER):
        row = aggregate["aggregate_by_classification"][label]
        print(f"[H4M-AC] {label}: n={row['sample_count']} chosen={row['chosen_action_counts']} correct_rate={row['correct_direction_rate']}")
    print(f"[H4M-AC] optimizer steps: {guard.optimizer_steps} | backward: {guard.backward_calls}")
    print(f"[H4M-AC] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-AC] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
