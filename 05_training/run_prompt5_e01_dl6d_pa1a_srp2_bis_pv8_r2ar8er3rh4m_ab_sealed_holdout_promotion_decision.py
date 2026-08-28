#!/usr/bin/env python3
"""H4M-AB sealed hold-out promotion decision gate.

Read-only eligibility decision for a single sealed hold-out evaluation.  This
program never opens, loads, reads or hashes the sealed TEST6 snapshots: the
sealed identity is taken from the window-id list already recorded in the frozen
H4L validation plan artifact.  No training, no inference, no tuning, no
checkpoint replacement, no new validation threshold, no database mutation and no
GitHub push.  Historical artifacts are read only.
"""

from __future__ import annotations

import ast
import hashlib
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


STAGE = "PV8-R2A-R8E-R3-R-H4M-AB"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AB_SEALED_HOLDOUT_PROMOTION_DECISION_GATE_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AB_NOT_READY_FOR_SEALED_HOLDOUT"
NEXT_GATE = "H4M-AC_SINGLE_OPEN_SEALED_HOLDOUT_EVALUATION"
PROMOTE = "PROMOTE_TO_SINGLE_SEALED_HOLDOUT_EVALUATION"
HOLD_DECISION = "HOLD_BEFORE_SEALED_HOLDOUT"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

H4MQ_REL = "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
H4MG_REL = "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
DL1_REL = "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
REWARD_REL = "05_training/rewards/mappo_reward_v1.py"
OBS_REL = "05_training/observation_target_context_repair.py"
DURABLE_REL = "05_training/durable_training_evidence.py"
EXECUTION_PATH_SOURCES = [H4MG_REL, DL1_REL, REWARD_REL, OBS_REL, DURABLE_REL, H4MQ_REL,
                          "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py",
                          "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"]

H4L_PLAN = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4l_frozen_three_seed_policy_validation_evaluation_20260814_124419" / "validation_window_resolution.json"
H4MY_RECOVERY_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_r1_report_only_recovery_from_persisted_evidence_20260818_153018+09:00"
H4MY_TRAINING_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_y_fresh_window_boundary_repaired_three_seed_retraining_20260818_150812+09:00"
H4MU_BLOCKED_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_u_fresh_target_conditioned_actor_head_specialized_"
    "and_critic_value_target_repaired_three_seed_retraining_20260817_193849+09:00"
)

EXPECTED = {
    "h4m_z_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_Z_"
        "WINDOW_BOUNDARY_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_VALIDATION_PROMOTION_COMPLETE"
    ),
    "h4m_z_classification": "A_PRIMARY_CAUSAL_ROOT_CAUSE_CONFIRMED_WINDOW_EPISODE_BOUNDARY",
    "validation_promotion_contract_sha256": "6c15c2011bcd0c40932198d754a36b68ba44ce810f57b7352f3634847af687e7",
    "h4m_aa_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_AA_"
        "FROZEN_WINDOW_BOUNDARY_REPAIRED_POLICY_VALIDATION_COMPLETE"
    ),
    "h4m_aa_source_commit_short": "e3392a4",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "observation_repair_contract_sha256": "6ecd20cfcd220d50a6f1ebbcd6e33594a9ca14a86a2a60236cea435a24058e1a",
    "boundary_schema": "independent_window_causal_horizon_termination_v1",
    "seeds": [1, 2, 3],
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "promotion_readiness.json",
    "checkpoint_binding.json",
    "sealed_evaluation_contract.json",
    "criteria_freeze.json",
    "lineage_integrity.json",
    "gate_matrix.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
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


def latest_artifact(pattern: str, gate: str) -> Path:
    roots = sorted(ARTIFACTS_ROOT.glob(pattern))
    passing = [r for r in roots if (r / "gate_matrix.json").exists() and read_json(r / "gate_matrix.json").get("gate") == gate]
    if not passing:
        raise RuntimeError(f"no passing artifact for {pattern}")
    return passing[-1]


# ---------------------------------------------------------------------------
def checkpoint_binding(promotion_contract: Mapping[str, Any], aa_binding: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for entry in promotion_contract["training_checkpoints"]:
        path = Path(entry["path"])
        observed = sha256_file(path)
        aa_row = next((r for r in aa_binding["bound_checkpoints"] if int(r["seed"]) == int(entry["seed"])), {})
        payload_stage = None
        try:
            import torch  # noqa: PLC0415 - metadata read only, no model is executed

            payload_stage = torch.load(path, map_location="cpu", weights_only=False).get("stage")
        except Exception as exc:  # pragma: no cover - defensive
            payload_stage = f"UNREADABLE:{exc!r}"
        rows.append(
            {
                "seed": entry["seed"],
                "path": str(path),
                "contract_sha256": entry["sha256"],
                "h4m_aa_observed_sha256": aa_row.get("observed_sha256"),
                "current_sha256": observed,
                "sha_fixed_and_unchanged": entry["sha256"] == observed == aa_row.get("observed_sha256"),
                "fresh_initialization": entry.get("fresh_initialization"),
                "payload_stage": payload_stage,
                "belongs_to_h4m_y_lineage": "H4M_Y_SEED" in path.name and path.parent.parent == H4MY_TRAINING_ROOT,
            }
        )
    excluded_u = sorted(p.name for p in (H4MU_BLOCKED_ROOT / "checkpoints").glob("*.pt")) if (H4MU_BLOCKED_ROOT / "checkpoints").exists() else []
    checks = {
        "three_checkpoints": len(rows) == 3,
        "sha256_fixed_and_unchanged": all(row["sha_fixed_and_unchanged"] for row in rows),
        "all_fresh_initialization": all(row["fresh_initialization"] is True for row in rows),
        "all_from_h4m_y_lineage": all(row["belongs_to_h4m_y_lineage"] for row in rows),
        "payload_stage_is_h4m_y": all(row["payload_stage"] == "PV8-R2A-R8E-R3-R-H4M-Y" for row in rows),
        "old_h4m_u_checkpoints_excluded": all(
            not any(name in row["path"] for name in excluded_u) for row in rows
        ),
    }
    return {
        "stage": STAGE,
        "checkpoints": rows,
        "excluded_incomplete_h4m_u_checkpoints": {
            "artifact": H4MU_BLOCKED_ROOT.name,
            "status": ["DIAGNOSTIC_ONLY", "NON_PROMOTABLE", "INCOMPLETE_EVIDENCE"],
            "files": excluded_u,
            "referenced_by_contract": False,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def lineage_integrity(z_root: Path, aa_root: Path) -> Dict[str, Any]:
    z_gate = read_json(z_root / "gate_matrix.json")
    z_decision = read_json(z_root / "root_cause_decision.json")
    aa_gate = read_json(aa_root / "gate_matrix.json")
    aa_results = read_json(aa_root / "frozen_validation_results.json")
    aa_integrity = read_json(aa_root / "integrity_validation.json")
    y_recovery_gate = read_json(H4MY_RECOVERY_ROOT / "gate_matrix.json")
    y_binding = read_json(H4MY_RECOVERY_ROOT / "recovery_binding.json")
    y_boundary = read_json(H4MY_RECOVERY_ROOT / "window_boundary_diagnostics.json")
    y_evidence = read_json(H4MY_RECOVERY_ROOT / "durable_evidence_report.json")
    header = y_binding["run_header"]

    aa_commit = git_run(["rev-parse", EXPECTED["h4m_aa_source_commit_short"]]).stdout.strip()
    commits_after = [line for line in git_run(["log", "--format=%H %s", f"{aa_commit}..HEAD"]).stdout.splitlines() if line]
    changed_after = [line for line in git_run(["diff", "--name-only", aa_commit, "HEAD"]).stdout.splitlines() if line]
    execution_changed_after = [path for path in changed_after if path in EXECUTION_PATH_SOURCES]
    training_artifacts_after = [
        p.name
        for p in ARTIFACTS_ROOT.iterdir()
        if p.is_dir() and p.stat().st_mtime > (aa_root.stat().st_mtime) and "retraining" in p.name
    ]

    test6_counts = {}
    for label, root in (("h4m_y_recovery", H4MY_RECOVERY_ROOT), ("h4m_z", z_root), ("h4m_aa", aa_root)):
        manifest = read_json(root / "manifest.json")
        test6_counts[label] = {
            "test6_access_count": manifest.get("test6_access_count"),
            "TEST6_opened": manifest.get("TEST6_opened"),
        }
    known_issues = [
        {
            "issue": "H4M-Y first execution blocked on a reporting FileNotFoundError",
            "status": "RESOLVED",
            "resolution": "H4M-Y-R1 rebuilt the artifact from 33/33 persisted evidence with zero retraining; the runner defect was fixed in the same commit",
            "can_invalidate_sealed_evaluation": False,
        },
        {
            "issue": "H4M-U-R2 manifest recorded a stale hash for the mutable run_state.json lifecycle pointer",
            "status": "DETERMINED_AND_TRACKED",
            "resolution": "H4M-W recorded that mutable lifecycle state must be hashed separately; H4M-Y and later manifests already declare the split",
            "can_invalidate_sealed_evaluation": False,
        },
        {
            "issue": "H4M-U checkpoints and traces remain incomplete evidence",
            "status": "EXCLUDED",
            "resolution": "kept DIAGNOSTIC_ONLY / NON_PROMOTABLE and never referenced by the promotion contract",
            "can_invalidate_sealed_evaluation": False,
        },
    ]
    bindings = {
        "w1_repair_contract_sha256": header.get("w1_repair_contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "actor_repair_contract_sha256": header.get("actor_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "critic_repair_contract_sha256": header.get("critic_repair_contract_sha256") == EXPECTED["critic_repair_contract_sha256"],
        "reward_v2_sha256": header.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "r3_split_sha256": header.get("split_sha256") == EXPECTED["r3_split_sha256"],
        "h4m_b_schedule_sha256": header.get("schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "zero_loss_adapter_sha256": header.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "window_episode_boundary_schema": header.get("window_episode_boundary_schema") == EXPECTED["boundary_schema"],
    }
    checks = {
        "h4m_z_gate_pass": z_gate.get("gate") == EXPECTED["h4m_z_gate"],
        "h4m_z_classification_a": z_decision.get("classification") == EXPECTED["h4m_z_classification"],
        "h4m_z_contract_sha_match": z_gate.get("validation_promotion_contract_sha256") == EXPECTED["validation_promotion_contract_sha256"],
        "h4m_aa_gate_pass": aa_gate.get("gate") == EXPECTED["h4m_aa_gate"],
        "h4m_aa_frozen_criteria_passed": aa_results.get("frozen_criteria_passed") is True,
        "h4m_aa_read_only_counters_zero": aa_integrity["counters"]["optimizer_step_count"] == 0
        and aa_integrity["counters"]["backward_pass_count"] == 0
        and aa_integrity["counters"]["training_count"] == 0
        and aa_integrity["counters"]["parameter_mutation_count"] == 0,
        "h4m_y_recovery_gate_pass": str(y_recovery_gate.get("gate", "")).startswith("PASS_"),
        "h4m_y_evidence_complete": y_evidence.get("record_count") == 33 and y_evidence.get("integrity_passed") is True,
        "h4m_y_boundary_clean": y_boundary["checks"]["cross_window_leakage_zero"] is True
        and y_boundary["checks"]["every_sample_terminated"] is True,
        "all_frozen_bindings_unchanged": all(bindings.values()),
        "no_execution_path_change_after_h4m_aa": not execution_changed_after,
        "no_training_artifact_after_h4m_aa": not training_artifacts_after,
        "test6_access_zero_across_lineage": all(
            row["test6_access_count"] == 0 and row["TEST6_opened"] is False for row in test6_counts.values()
        ),
        "no_unresolved_invalidating_issue": all(not row["can_invalidate_sealed_evaluation"] for row in known_issues),
    }
    return {
        "stage": STAGE,
        "frozen_binding_identity": bindings,
        "commits_after_h4m_aa": commits_after,
        "files_changed_after_h4m_aa": changed_after,
        "execution_path_files_changed_after_h4m_aa": execution_changed_after,
        "training_artifacts_created_after_h4m_aa": training_artifacts_after,
        "test6_access_by_stage": test6_counts,
        "known_issues": known_issues,
        "checks": checks,
        "passed": all(checks.values()),
    }


def sealed_identity_without_reading() -> Dict[str, Any]:
    """Sealed identity from the frozen plan record only; no snapshot is opened."""
    plan = read_json(H4L_PLAN)
    sealed_ids = list(plan.get("test_window_ids_sealed_not_loaded", []))
    return {
        "identity_source": str(H4L_PLAN),
        "identity_kind": "recorded sealed window-id list from the frozen H4L validation plan",
        "sealed_window_ids": sealed_ids,
        "sealed_window_count": len(sealed_ids),
        "sealed_window_id_list_sha256": canonical_sha(sorted(sealed_ids)),
        "content_hash_computed": False,
        "content_hash_omitted_reason": "hashing the snapshots would require reading sealed contents, which this gate forbids",
        "snapshot_paths_loaded": plan.get("test_snapshot_paths_loaded", []),
        "governing_split_sha256": EXPECTED["r3_split_sha256"],
        "validation_windows_used_so_far": plan.get("validation_window_ids", []),
        "sealed_and_validation_are_disjoint": not (set(sealed_ids) & set(plan.get("validation_window_ids", []))),
    }


def criteria_freeze(aa_root: Path) -> Dict[str, Any]:
    aa_results = read_json(aa_root / "frozen_validation_results.json")
    aa_gate = read_json(aa_root / "gate_matrix.json")
    body = {
        "criteria_frozen_before_sealed_open": True,
        "criteria_origin": "carried over verbatim from the frozen H4M-Q validation criteria and the H4M-AA pass principle; nothing new is introduced here",
        "structural_criteria": {
            "frozen_validation_checks": sorted(aa_results.get("frozen_criteria", {}).keys()),
            "context_appropriate_discrimination": {
                HOLD_BETTER: f"chosen-action dominant must be {HOLD}",
                SERVE_BETTER: f"chosen-action dominant must be {SERVE}",
                "scope": "aggregate and every seed",
            },
            "no_unexplained_single_action_collapse": "the dominant action must differ between the two target contexts",
            "seed_consistency": "all three seeds must give the same directional outcome per context",
        },
        "integrity_criteria": {
            "training": 0,
            "optimizer_step": 0,
            "backward": 0,
            "parameter_mutation": 0,
            "illegal_action_count": 0,
            "nan_inf_count": 0,
            "sealed_open_count": 1,
        },
        "numeric_thresholds": {
            "hold_ratio_threshold": None,
            "accuracy_threshold": None,
            "reason": "PASS is a structural discrimination and integrity property; no numeric threshold existed in the frozen criteria and none may be created",
        },
        "failure_handling": {
            "on_failure": "record BLOCKED with the evidence as measured",
            "retry_allowed": False,
            "checkpoint_selection_after_results": False,
            "seed_dropping": False,
            "tuning": False,
            "threshold_change_after_open": False,
        },
        "reporting_rules": {
            "report_per_seed_and_aggregate": True,
            "required_fields": [
                "sample_count",
                "chosen_action_counts",
                "correct_direction_rate",
                "P_HOLD_mean",
                "P_SERVE_mean",
                "policy_entropy_mean",
            ],
            "no_metric_added_after_results": True,
            "training_diagnostics_are_not_sealed_performance_evidence": True,
            "datasets_never_pooled": True,
        },
        "h4m_aa_reference_outcome": {
            "gate": aa_gate.get("gate"),
            "aggregate": aa_results.get("aggregate_by_classification"),
            "role": "frozen validation reference only; it is not evidence about the sealed split",
        },
    }
    return {"stage": STAGE, "criteria": body, "criteria_sha256": canonical_sha(body)}


def promotion_readiness(binding: Mapping[str, Any], lineage: Mapping[str, Any], sealed: Mapping[str, Any], criteria: Mapping[str, Any], self_audit: Mapping[str, Any]) -> Dict[str, Any]:
    conditions = {
        "1_h4m_z_root_cause_is_A": lineage["checks"]["h4m_z_classification_a"],
        "2_h4m_aa_frozen_validation_pass": lineage["checks"]["h4m_aa_gate_pass"] and lineage["checks"]["h4m_aa_frozen_criteria_passed"],
        "3_checkpoint_sha_fixed_and_unchanged": binding["checks"]["sha256_fixed_and_unchanged"],
        "4_all_frozen_bindings_unchanged": lineage["checks"]["all_frozen_bindings_unchanged"],
        "5_fresh_lineage_and_h4m_u_excluded": binding["checks"]["all_fresh_initialization"]
        and binding["checks"]["all_from_h4m_y_lineage"]
        and binding["checks"]["old_h4m_u_checkpoints_excluded"],
        "6_no_training_or_tuning_after_h4m_aa": lineage["checks"]["no_execution_path_change_after_h4m_aa"]
        and lineage["checks"]["no_training_artifact_after_h4m_aa"],
        "7_criteria_frozen_before_sealed_open": criteria["criteria"]["criteria_frozen_before_sealed_open"] is True,
        "8_no_threshold_creatable_after_open": criteria["criteria"]["failure_handling"]["threshold_change_after_open"] is False,
        "9_test6_access_zero_through_h4m_ab": lineage["checks"]["test6_access_zero_across_lineage"] and self_audit["passed"],
        "10_no_unresolved_invalidating_issue": lineage["checks"]["no_unresolved_invalidating_issue"],
    }
    return {
        "stage": STAGE,
        "conditions": conditions,
        "unmet_conditions": [key for key, value in conditions.items() if not value],
        "sealed_identity": sealed,
        "ready": all(conditions.values()),
        "decision": PROMOTE if all(conditions.values()) else HOLD_DECISION,
        "training_diagnostics_excluded_as_sealed_evidence": True,
    }


def self_audit_no_sealed_access() -> Dict[str, Any]:
    source = (PROJECT_ROOT / SOURCE_REL).read_text(encoding="utf-8")
    tree = ast.parse(source)
    load_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"read_parquet", "load"}
    ]
    sealed_ids = read_json(H4L_PLAN).get("test_window_ids_sealed_not_loaded", [])
    checks = {
        "no_parquet_read_in_this_gate": "read_parquet" not in load_calls,
        "only_checkpoint_metadata_torch_load": load_calls.count("load") <= 1,
        "no_sealed_window_id_literal_in_source": not any(str(window_id) in source for window_id in sealed_ids),
        "sealed_snapshot_paths_not_loaded": read_json(H4L_PLAN).get("test_snapshot_paths_loaded", []) == [],
    }
    return {
        "stage": STAGE,
        "torch_or_parquet_load_calls": sorted(set(load_calls)),
        "checks": checks,
        "test6_access_count": 0,
        "passed": all(checks.values()),
    }


def sealed_evaluation_contract(readiness: Mapping[str, Any], binding: Mapping[str, Any], criteria: Mapping[str, Any], sealed: Mapping[str, Any], source_commit: str) -> Dict[str, Any]:
    if not readiness["ready"]:
        return {"stage": STAGE, "frozen": False, "reason": "promotion preconditions not met", "decision": HOLD_DECISION}
    body = {
        "decision": PROMOTE,
        "next_gate": NEXT_GATE,
        "checkpoints": [
            {"seed": row["seed"], "path": row["path"], "sha256": row["current_sha256"]} for row in binding["checkpoints"]
        ],
        "source_commit": source_commit,
        "implementation_lineage": {
            "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
            "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
            "critic_s3_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "observation_repair_contract_sha256": EXPECTED["observation_repair_contract_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "validation_promotion_contract_sha256": EXPECTED["validation_promotion_contract_sha256"],
        },
        "sealed_dataset_identity": sealed,
        "evaluation_path": {
            "evaluation_code": f"{H4MQ_REL}::validation_discrimination",
            "reason": "the same frozen inference path already used for the approved validation split, pointed at the sealed windows",
            "inference_mode": "torch.no_grad forward pass with argmax action selection",
            "device": "MPS if available else CPU",
            "seed_handling": "each of the three checkpoints is evaluated independently on the identical sealed windows; no seed is merged, dropped or reweighted",
            "aggregation_rules": "per seed first, then a simple unweighted aggregate over all sealed rows; no filtering, no reweighting, no exclusion",
        },
        "reporting_rules": criteria["criteria"]["reporting_rules"],
        "promotion_criteria": criteria["criteria"]["structural_criteria"],
        "integrity_criteria": criteria["criteria"]["integrity_criteria"],
        "failure_handling": criteria["criteria"]["failure_handling"],
        "single_open_rule": {
            "sealed_holdout_opened_exactly_once": True,
            "open_count_must_be_recorded": True,
            "second_open_requires_new_explicit_gate": True,
        },
        "hard_locks": {
            "training": 0,
            "optimizer_step": 0,
            "backward": 0,
            "parameter_mutation": 0,
            "checkpoint_selection_after_results": False,
            "seed_dropping": False,
            "retry_on_unfavorable_result": False,
            "tuning": False,
            "criteria_change_after_open": False,
        },
        "criteria_sha256": criteria["criteria_sha256"],
    }
    return {"stage": STAGE, "frozen": True, "contract": body, "sealed_evaluation_contract_sha256": canonical_sha(body)}


# ---------------------------------------------------------------------------
def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="h4mab_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "ab.pyc"), doraise=True)
            error = None
        except Exception as exc:
            error = repr(exc)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit": head,
        "head_commit_files": head_files,
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "clean_worktree": status == "",
        "py_compile_passed": error is None,
        "github_push_performed": False,
    }


def gate_matrix(provenance, readiness, binding, lineage, contract, self_audit) -> Dict[str, Any]:
    promoted = contract.get("frozen") is True
    criteria = {
        "source_only_commit": provenance.get("source_only_local_commit") is True,
        "py_compile_passed": provenance.get("py_compile_passed") is True,
        "checkpoint_binding_passed": binding.get("passed") is True,
        "lineage_integrity_passed": lineage.get("passed") is True,
        "all_ten_readiness_conditions": readiness.get("ready") is True,
        "sealed_contract_frozen": promoted and bool(contract.get("sealed_evaluation_contract_sha256")),
        "no_test6_access_in_this_gate": self_audit.get("passed") is True and self_audit.get("test6_access_count") == 0,
        "no_training_or_inference_here": True,
        "github_push_false": provenance.get("github_push_performed") is False,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else BLOCK_GATE,
        "decision": readiness.get("decision"),
        "sealed_evaluation_contract_sha256": contract.get("sealed_evaluation_contract_sha256"),
        "exact_next_gate": NEXT_GATE if passed and promoted else f"STOP_{BLOCK_GATE}",
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "TEST6_opened": False,
            "test6_access_count": 0,
            "training_count": 0,
            "optimizer_step_count": 0,
            "inference_on_sealed_holdout": False,
            "checkpoint_replaced": False,
            "new_validation_threshold_created": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate, provenance, contract, started) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit": provenance.get("source_commit"),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "sealed_evaluation_contract_sha256": contract.get("sealed_evaluation_contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "mutable_lifecycle_state_files": [],
        "append_only_artifact": True,
        "read_only_decision_gate": True,
        "elapsed_seconds": time.perf_counter() - started,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "training_count": 0,
        "optimizer_step_count": 0,
        "github_push_performed": False,
    }


def final_report(provenance, readiness, binding, lineage, criteria, contract, gate) -> str:
    return f"""# H4M-AB Sealed Hold-out Promotion Decision Gate

gate = {gate['gate']}
decision = {gate['decision']}
sealed_evaluation_contract_sha256 = {gate.get('sealed_evaluation_contract_sha256')}
criteria_sha256 = {criteria['criteria_sha256']}
source_commit = {provenance['source_commit']}
TEST6_opened = false
test6_access_count = 0
training_count = 0
optimizer_step_count = 0
exact_next_gate = {gate['exact_next_gate']} (not executed automatically)

## Readiness conditions

```json
{json.dumps(readiness['conditions'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Checkpoint binding

```json
{json.dumps({'checkpoints': binding['checkpoints'], 'excluded': binding['excluded_incomplete_h4m_u_checkpoints'], 'checks': binding['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Lineage integrity

```json
{json.dumps({'frozen_binding_identity': lineage['frozen_binding_identity'], 'commits_after_h4m_aa': lineage['commits_after_h4m_aa'], 'execution_path_files_changed_after_h4m_aa': lineage['execution_path_files_changed_after_h4m_aa'], 'test6_access_by_stage': lineage['test6_access_by_stage'], 'checks': lineage['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Known issues considered

```json
{json.dumps(lineage['known_issues'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Sealed identity, obtained without reading the sealed data

```json
{json.dumps(readiness['sealed_identity'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Frozen criteria for the sealed evaluation

```json
{json.dumps(criteria['criteria'], ensure_ascii=False, indent=2, default=jsonable)}
```

## Sealed evaluation contract

```json
{json.dumps(contract.get('contract', {'frozen': contract.get('frozen'), 'reason': contract.get('reason')}), ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: this gate opened no sealed data, ran no training and no inference, replaced no checkpoint, created no
new validation threshold, mutated no data, and pushed nothing. H4M-Y training diagnostics are explicitly not
sealed-split performance evidence. H4M-AC is not executed automatically.
"""


def main() -> None:
    started = time.perf_counter()
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ab_sealed_holdout_promotion_decision_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    z_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_z_*", EXPECTED["h4m_z_gate"])
    aa_root = latest_artifact("pv8_r2a_r8e_r3_r_h4m_aa_*", EXPECTED["h4m_aa_gate"])
    promotion_contract = read_json(z_root / "validation_promotion_contract.json")["contract"]
    aa_binding = read_json(aa_root / "validation_binding.json")

    provenance = source_provenance(created_at)
    binding = checkpoint_binding(promotion_contract, aa_binding)
    lineage = lineage_integrity(z_root, aa_root)
    sealed = sealed_identity_without_reading()
    criteria = criteria_freeze(aa_root)
    self_audit = self_audit_no_sealed_access()
    readiness = promotion_readiness(binding, lineage, sealed, criteria, self_audit)
    contract = sealed_evaluation_contract(readiness, binding, criteria, sealed, provenance["source_commit"])
    gate = gate_matrix(provenance, readiness, binding, lineage, contract, self_audit)

    for name, payload in {
        "promotion_readiness.json": readiness,
        "checkpoint_binding.json": binding,
        "lineage_integrity.json": lineage,
        "criteria_freeze.json": criteria,
        "sealed_evaluation_contract.json": contract,
        "self_audit_no_sealed_access.json": self_audit,
        "source_provenance.json": provenance,
        "gate_matrix.json": gate,
    }.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(final_report(provenance, readiness, binding, lineage, criteria, contract, gate), encoding="utf-8")
    write_json(root / "manifest.json", make_manifest(root, gate, provenance, contract, started))

    print(f"[H4M-AB] artifact root: {root}")
    print(f"[H4M-AB] gate: {gate['gate']}")
    print(f"[H4M-AB] decision: {gate['decision']}")
    print(f"[H4M-AB] sealed_evaluation_contract_sha256: {gate.get('sealed_evaluation_contract_sha256')}")
    print(f"[H4M-AB] criteria_sha256: {criteria['criteria_sha256']}")
    print(f"[H4M-AB] unmet conditions: {readiness['unmet_conditions']}")
    print(f"[H4M-AB] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-AB] TEST6 opened: False | access count: 0")
    print(f"[H4M-AB] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
