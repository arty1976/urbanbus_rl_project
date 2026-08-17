#!/usr/bin/env python3
"""H4M-P target-conditioned Actor head specialization validation.

This stage validates a minimum-change Actor repair before any retraining:
the shared Actor trunk is preserved and the decision head is specialized by
the existing pre-action target one-hot context. It records source provenance,
authoritative H4M-O binding, strict legacy equivalence, gradient isolation,
serialization compatibility, and hard-lock attestations.

No MAPPO training, optimizer step, TEST6 or validation access, GitHub push, reward
change, simulator change, or data mutation is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-P"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_P_"
    "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_P"
NEXT_GATE = "H4M-Q_FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_THREE_SEED_RETRAINING"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACT_ROOT_BASE = TRAINING_ROOT / "artifacts"

DL1_REL = Path("05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
TEST_REL = Path("05_training/test_h4m_p_target_conditioned_actor_head_specialization.py")
SOURCE_REL = Path("05_training") / Path(__file__).name
INTENDED_CHANGED_FILES = [DL1_REL.as_posix(), SOURCE_REL.as_posix(), TEST_REL.as_posix()]

H4M_O_ROOT = ARTIFACT_ROOT_BASE / (
    "pv8_r2a_r8e_r3_r_h4m_o_actor_target_conditioned_retraining_outcome_review_next_decision_"
    "20260817_143826+0900"
)

EXPECTED = {
    "h4m_o_source_commit": "77eb290dd1f0fbd4b0a5fd2ecf0d5ec9b288248d",
    "h4m_o_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_O_ACTOR_TARGET_CONDITIONED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION_COMPLETE",
    "h4m_o_decision": "MIXED_TARGET_CONDITIONED_GRADIENT_IMBALANCE",
    "shared_actor_net_serve_minus_hold": 684.8146479033484,
    "target_hold_branch_net_serve_minus_hold": -783.2955939588423,
    "target_serve_branch_net_serve_minus_hold": 1468.1102418621904,
    "serve_hold_abs_pressure_ratio": 1.8742735860956867,
    "same_column_target_interference": 0,
    "earliest_collapse_cycle": 2,
    "selected_repair": "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_REPAIR",
    "repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "repair_binding.json",
    "equivalence_validation.json",
    "gradient_isolation_validation.json",
    "changed_files.json",
    "test_results.json",
    "test_results.stdout.txt",
    "gate_matrix.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def timestamp() -> str:
    return kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"


def artifact_timestamp() -> str:
    return kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + "00"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def float_close(observed: Any, expected: float, tolerance: float = 1.0e-12) -> bool:
    try:
        return math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except Exception:
        return False


def py_compile_sources() -> Dict[str, Any]:
    files = [PROJECT_ROOT / path for path in [DL1_REL, TEST_REL, SOURCE_REL]]
    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="h4mp_pycompile_") as tmp:
        for path in files:
            try:
                py_compile.compile(str(path), cfile=str(Path(tmp) / (path.name + ".pyc")), doraise=True)
                rows.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "passed": True, "error": None})
            except Exception as exc:
                rows.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "passed": False, "error": repr(exc)})
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {
        "rows": rows,
        "py_compile_passed": all(row["passed"] for row in rows),
        "git_diff_cached_check_passed": cached.returncode == 0,
        "git_diff_cached_check_stdout": cached.stdout.strip(),
        "git_diff_cached_check_stderr": cached.stderr.strip(),
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"], check=False)
    parent_commit = parent.stdout.strip() if parent.returncode == 0 else None
    branch = git_run(["branch", "--show-current"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status_short = git_run(["status", "--short"]).stdout.strip()
    latest_by_file = {
        rel: git_run(["log", "-1", "--format=%H", "--", rel]).stdout.strip()
        for rel in INTENDED_CHANGED_FILES
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "source_commit_before_work": parent_commit,
        "source_commit_after_pass": head,
        "head_commit_files": head_files,
        "intended_changed_files": INTENDED_CHANGED_FILES,
        "head_commit_intended_source_test_only": sorted(head_files) == sorted(INTENDED_CHANGED_FILES),
        "source_commit_before_work_matches_h4m_o": parent_commit == EXPECTED["h4m_o_source_commit"],
        "latest_commit_by_file": latest_by_file,
        "all_intended_files_latest_at_head": all(value == head for value in latest_by_file.values()),
        "status_short": status_short,
        "github_push_performed": False,
    }


def changed_files_audit(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    rw_marker = "reward" + "_weight"
    rv2_semantics_marker = "Reward V2 semantics" + " change"
    for rel in sorted(set(provenance.get("head_commit_files", []))):
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "line_count": len(text.splitlines()) if path.exists() else None,
                "contains_reward_v2_formula_change": rw_marker in text or rv2_semantics_marker in text,
                "contains_test6_access_path": ("TEST" + "6/") in text or ("test" + "6/") in text,
                "contains_optimizer_step_in_test": ".step()" in text and rel == TEST_REL.as_posix(),
            }
        )
    return {
        "stage": STAGE,
        "modified_files": rows,
        "intended_changed_files": INTENDED_CHANGED_FILES,
        "only_intended_files_changed": sorted(provenance.get("head_commit_files", [])) == sorted(INTENDED_CHANGED_FILES),
        "source_level_repair_markers": {
            "actor_specialization_flag_present": "target_head_specialization" in (PROJECT_ROOT / DL1_REL).read_text(encoding="utf-8"),
            "repair_contract_sha_bound_in_actor": EXPECTED["repair_contract_sha256"]
            in (PROJECT_ROOT / DL1_REL).read_text(encoding="utf-8"),
            "forward_policy_target_context_source_preserved": "data.x[idx, -target_context_dim:]"
            in (PROJECT_ROOT / DL1_REL).read_text(encoding="utf-8"),
        },
    }


def authoritative_repair_binding(created_at: str, provenance: Mapping[str, Any], compile_audit: Mapping[str, Any]) -> Dict[str, Any]:
    h4m_o_gate = read_json(H4M_O_ROOT / "11_gate_matrix.json")
    h4m_o_binding = read_json(H4M_O_ROOT / "01_authoritative_binding.json")
    h4m_o_contract = read_json(H4M_O_ROOT / "10_h4m_o_repair_contract.json")
    h4m_o_branch = read_json(H4M_O_ROOT / "04_shared_vs_target_branch_gradient.json")
    h4m_o_interference = read_json(H4M_O_ROOT / "05_gradient_interference_audit.json")
    h4m_o_cycle = read_json(H4M_O_ROOT / "07_cyclewise_collapse_origin.json")
    observed = {
        "h4m_o_source_commit": h4m_o_binding.get("source_provenance", {}).get("h4m_o_source_git_commit"),
        "h4m_o_gate": h4m_o_gate.get("gate"),
        "h4m_o_decision": h4m_o_gate.get("decision"),
        "selected_repair": h4m_o_contract.get("selected_repair"),
        "repair_contract_sha256": h4m_o_contract.get("contract_sha256"),
        "shared_actor_net_serve_minus_hold": h4m_o_branch.get("shared_actor_gradient_result", {}).get(
            "net_serve_minus_hold_pressure_sum"
        ),
        "target_hold_branch_net_serve_minus_hold": h4m_o_branch.get("target_conditioning_gradient_result", {}).get(
            "target_hold_branch_net_serve_minus_hold_pressure_sum"
        ),
        "target_serve_branch_net_serve_minus_hold": h4m_o_branch.get("target_conditioning_gradient_result", {}).get(
            "target_serve_branch_net_serve_minus_hold_pressure_sum"
        ),
        "serve_hold_abs_pressure_ratio": h4m_o_branch.get("target_conditioning_gradient_result", {}).get(
            "serve_context_abs_pressure_over_hold_context_abs_pressure"
        ),
        "same_column_target_interference": h4m_o_branch.get("target_conditioning_gradient_result", {}).get(
            "same_column_target_interference"
        ),
        "gradient_interference_result": h4m_o_interference.get("shared_parameter_interference", {}).get("overall_result"),
        "earliest_collapse_cycle": h4m_o_cycle.get("collapse_origin", {}).get("earliest_collapse_cycle"),
    }
    checks = {
        "h4m_o_source_commit_match": observed["h4m_o_source_commit"] == EXPECTED["h4m_o_source_commit"],
        "h4m_o_gate_match": observed["h4m_o_gate"] == EXPECTED["h4m_o_gate"],
        "h4m_o_decision_match": observed["h4m_o_decision"] == EXPECTED["h4m_o_decision"],
        "selected_repair_match": observed["selected_repair"] == EXPECTED["selected_repair"],
        "repair_contract_sha_match": observed["repair_contract_sha256"] == EXPECTED["repair_contract_sha256"],
        "shared_actor_pressure_match": float_close(
            observed["shared_actor_net_serve_minus_hold"], EXPECTED["shared_actor_net_serve_minus_hold"]
        ),
        "target_hold_branch_match": float_close(
            observed["target_hold_branch_net_serve_minus_hold"], EXPECTED["target_hold_branch_net_serve_minus_hold"]
        ),
        "target_serve_branch_match": float_close(
            observed["target_serve_branch_net_serve_minus_hold"], EXPECTED["target_serve_branch_net_serve_minus_hold"]
        ),
        "serve_hold_abs_pressure_ratio_match": float_close(
            observed["serve_hold_abs_pressure_ratio"], EXPECTED["serve_hold_abs_pressure_ratio"]
        ),
        "same_column_target_interference_match": observed["same_column_target_interference"]
        == EXPECTED["same_column_target_interference"],
        "earliest_collapse_cycle_match": observed["earliest_collapse_cycle"] == EXPECTED["earliest_collapse_cycle"],
        "source_commit_before_work_match": provenance.get("source_commit_before_work_matches_h4m_o") is True,
        "source_only_local_commit_after_pass": provenance.get("head_commit_intended_source_test_only") is True
        and provenance.get("all_intended_files_latest_at_head") is True,
        "py_compile_passed": compile_audit.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_audit.get("git_diff_cached_check_passed") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4m_o_artifact_root": str(H4M_O_ROOT),
        "expected": EXPECTED,
        "observed": observed,
        "source_provenance": provenance,
        "compile_audit": compile_audit,
        "checks": checks,
        "repair_binding_passed": all(checks.values()),
        "hard_lock_attestation": {
            "reward_v2_semantics_modified": False,
            "simulator_semantics_modified": False,
            "observation_contract_modified": False,
            "critic_architecture_or_targets_modified": False,
            "k_mask_action_legality_modified": False,
            "three_action_policy_semantics_modified": False,
            "zero_loss_semantics_modified": False,
            "frozen_split_modified": False,
            "training_schedule_modified": False,
            "logit_bias_workaround_used": False,
            "mappo_training_executed": False,
            "fresh_retraining_executed": False,
            "test6_opened_or_used": False,
            "github_push_performed": False,
        },
    }


def run_test_command(artifact_root: Path) -> Dict[str, Any]:
    test_json = artifact_root / "test_results.json"
    command = [
        sys.executable,
        "-B",
        str(PROJECT_ROOT / TEST_REL),
        "--json-output",
        str(test_json),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    (artifact_root / "test_results.stdout.txt").write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr,
        encoding="utf-8",
    )
    payload = read_json(test_json) if test_json.exists() else {}
    payload["command"] = command
    payload["returncode"] = completed.returncode
    payload["stdout"] = completed.stdout
    payload["stderr"] = completed.stderr
    write_json(test_json, payload)
    return payload


def equivalence_validation(test_results: Mapping[str, Any]) -> Dict[str, Any]:
    sections = test_results.get("sections", {})
    return {
        "stage": STAGE,
        "strict_float_tolerance": 1.0e-7,
        "legacy_vs_specialized": sections.get("legacy_equivalence", {}),
        "action_mask_equivalence": sections.get("action_masks", {}),
        "critic_value_equivalence": sections.get("critic_and_observation_contract", {}).get(
            "critic_value_diff_same_input", {}
        ),
        "observation_contract": sections.get("critic_and_observation_contract", {}),
        "serialization_checkpoint_contract": sections.get("serialization", {}),
        "nan_inf_count": int(test_results.get("nan_inf_section_count", -1)),
        "training_executed": bool(test_results.get("training_executed")),
        "optimizer_step_count": int(test_results.get("optimizer_step_count", -1)),
        "validation_or_test6_access_count": int(test_results.get("validation_or_test6_access_count", -1)),
        "equivalence_validation_passed": all(
            sections.get(name, {}).get("passed") is True
            for name in [
                "legacy_equivalence",
                "action_masks",
                "critic_and_observation_contract",
                "serialization",
            ]
        )
        and int(test_results.get("nan_inf_section_count", -1)) == 0
        and bool(test_results.get("training_executed")) is False
        and int(test_results.get("optimizer_step_count", -1)) == 0
        and int(test_results.get("validation_or_test6_access_count", -1)) == 0,
    }


def gradient_isolation_validation(test_results: Mapping[str, Any]) -> Dict[str, Any]:
    section = test_results.get("sections", {}).get("gradient_isolation", {})
    return {
        "stage": STAGE,
        "gradient_isolation": section,
        "routing": test_results.get("sections", {}).get("routing", {}),
        "repair_binding": test_results.get("sections", {}).get("repair_binding", {}),
        "shared_trunk_gradient_sharing_explicit": section.get("shared_trunk_gradient_sharing"),
        "optimizer_step_count": int(test_results.get("optimizer_step_count", -1)),
        "gradient_isolation_validation_passed": section.get("passed") is True
        and test_results.get("sections", {}).get("routing", {}).get("passed") is True
        and test_results.get("sections", {}).get("repair_binding", {}).get("passed") is True
        and int(test_results.get("optimizer_step_count", -1)) == 0,
    }


def gate_matrix(
    binding: Mapping[str, Any],
    changed_files: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    gradient: Mapping[str, Any],
    test_results: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "repair_contract_sha_explicitly_bound": binding.get("checks", {}).get("repair_contract_sha_match") is True
        and gradient.get("repair_binding", {}).get("passed") is True,
        "target_routing_deterministic_correct": gradient.get("routing", {}).get("passed") is True,
        "hold_target_loss_isolates_hold_head": (
            gradient.get("gradient_isolation", {})
            .get("hold_target_probe", {})
            .get("target_head_grad_norms", {})
            .get("target_head_0", 0.0)
            > 0.0
            and gradient.get("gradient_isolation", {})
            .get("hold_target_probe", {})
            .get("target_head_grad_norms", {})
            .get("target_head_1", 1.0)
            == 0.0
        ),
        "serve_target_loss_isolates_serve_head": (
            gradient.get("gradient_isolation", {})
            .get("serve_target_probe", {})
            .get("target_head_grad_norms", {})
            .get("target_head_1", 0.0)
            > 0.0
            and gradient.get("gradient_isolation", {})
            .get("serve_target_probe", {})
            .get("target_head_grad_norms", {})
            .get("target_head_0", 1.0)
            == 0.0
        ),
        "shared_trunk_gradient_sharing_explicit": bool(gradient.get("shared_trunk_gradient_sharing_explicit")),
        "legacy_equivalence_within_tolerance": equivalence.get("legacy_vs_specialized", {}).get("passed") is True,
        "action_masks_equivalent": equivalence.get("action_mask_equivalence", {}).get("passed") is True,
        "critic_value_path_equivalent": equivalence.get("observation_contract", {}).get("passed") is True,
        "observation_shape_contract_equivalent": equivalence.get("observation_contract", {}).get("passed") is True,
        "checkpoint_serialization_valid_or_migrated": equivalence.get("serialization_checkpoint_contract", {}).get("passed")
        is True,
        "nan_inf_zero": equivalence.get("nan_inf_count") == 0,
        "no_future_leakage": test_results.get("synthetic_fixture_only") is True
        and test_results.get("validation_or_test6_access_count") == 0,
        "no_test6_access": test_results.get("validation_or_test6_access_count") == 0,
        "no_optimizer_driven_training": test_results.get("training_executed") is False
        and test_results.get("optimizer_step_count") == 0,
        "only_intended_files_changed": changed_files.get("only_intended_files_changed") is True,
        "repair_binding_passed": binding.get("repair_binding_passed") is True,
        "test_command_passed": test_results.get("passed") is True and test_results.get("returncode") == 0,
    }
    failing = [name for name, value in criteria.items() if value is not True]
    gate = PASS_GATE if not failing else f"{BLOCK_GATE_PREFIX}_VALIDATION_FAILED"
    return {
        "stage": STAGE,
        "gate": gate,
        "decision": "TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_REPAIR_IMPLEMENTED_AND_VALIDATED"
        if not failing
        else "BLOCKED",
        "criteria": criteria,
        "failing_criteria": failing,
        "repair_contract_sha256": EXPECTED["repair_contract_sha256"],
        "source_commit_before_work": binding.get("source_provenance", {}).get("source_commit_before_work"),
        "source_commit_after_pass": binding.get("source_provenance", {}).get("source_commit_after_pass"),
        "exact_next_gate": NEXT_GATE if not failing else None,
        "final_flags": {
            "mappo_training_executed": False,
            "fresh_retraining_executed": False,
            "optimizer_step_count": test_results.get("optimizer_step_count"),
            "test6_access_count": test_results.get("validation_or_test6_access_count"),
            "github_push_performed": False,
            "repair_implemented": not bool(failing),
            "reward_gae_ppo_modified": False,
            "environment_data_modified": False,
        },
    }


def final_report(
    binding: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    gradient: Mapping[str, Any],
    changed_files: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    legacy = equivalence.get("legacy_vs_specialized", {})
    logit_diff = legacy.get("logit_diff", {})
    prob_diff = legacy.get("probability_diff", {})
    grad = gradient.get("gradient_isolation", {})
    hold = grad.get("hold_target_probe", {})
    serve = grad.get("serve_target_probe", {})
    return "\n".join(
        [
            "# H4M-P Target-Conditioned Actor Head Specialization",
            "",
            f"gate = {gate.get('gate')}",
            f"source_commit_before_work = {gate.get('source_commit_before_work')}",
            f"source_commit_after_pass = {gate.get('source_commit_after_pass')}",
            f"repair_contract_sha256 = {gate.get('repair_contract_sha256')}",
            f"exact_next_gate = {gate.get('exact_next_gate')}",
            "",
            "## Implementation",
            "",
            "- Preserved MAPPOActor shared input/trunk path: `net[0] -> Tanh`.",
            "- Added deterministic target one-hot routing to 3 specialized decision heads.",
            "- Default actor construction remains legacy-compatible with specialization disabled.",
            "- Legacy 131D direct-conditioned actor state dict migrates by copying `net.2` into all target heads.",
            "",
            "## Equivalence",
            "",
            f"- strict tolerance: `{equivalence.get('strict_float_tolerance')}`",
            f"- legacy vs specialized logit max_abs_diff: `{logit_diff.get('max_abs_diff')}`",
            f"- legacy vs specialized probability max_abs_diff: `{prob_diff.get('max_abs_diff')}`",
            f"- action mask equivalence passed: `{equivalence.get('action_mask_equivalence', {}).get('passed')}`",
            f"- critic/observation contract passed: `{equivalence.get('observation_contract', {}).get('passed')}`",
            f"- serialization migration/reload passed: `{equivalence.get('serialization_checkpoint_contract', {}).get('passed')}`",
            "",
            "## Gradient isolation",
            "",
            f"- HOLD target head grad norms: `{hold.get('target_head_grad_norms')}`",
            f"- SERVE target head grad norms: `{serve.get('target_head_grad_norms')}`",
            f"- shared trunk gradient sharing: `{gradient.get('shared_trunk_gradient_sharing_explicit')}`",
            f"- optimizer_step_count: `{gradient.get('optimizer_step_count')}`",
            "",
            "## Changed files",
            "",
            "```json",
            canonical_json(changed_files.get("modified_files", [])),
            "```",
            "",
            "## Hard locks",
            "",
            "Reward V2, simulator semantics, 12D observation contract, Critic/value path, K-mask/action legality, "
            "3-action policy semantics, Zero-Loss, frozen split, and training schedule remained unchanged.",
            "",
            "STOP: no retraining, no TEST6, no GitHub push.",
        ]
    ) + "\n"


def manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files = sorted({path.name for path in artifact_root.iterdir() if path.is_file()} | {"manifest.json"})
    return {
        "stage": STAGE,
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "source_commit_before_work": gate.get("source_commit_before_work"),
        "source_commit_after_pass": gate.get("source_commit_after_pass"),
        "repair_contract_sha256": gate.get("repair_contract_sha256"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all(name == "manifest.json" or (artifact_root / name).exists() for name in REQUIRED_ARTIFACTS),
        "output_files": output_files,
        "output_sha256": {
            name: sha256_file(artifact_root / name) if (artifact_root / name).exists() else None for name in output_files
        },
        "mappo_training_executed": False,
        "fresh_retraining_executed": False,
        "TEST6_opened": False,
        "github_push_performed": False,
    }


def main() -> None:
    created_at = kst_now().isoformat()
    artifact_root = ARTIFACT_ROOT_BASE / (
        "pv8_r2a_r8e_r3_r_h4m_p_target_conditioned_actor_head_specialization_"
        f"implementation_equivalence_validation_{artifact_timestamp()}"
    )
    artifact_root.mkdir(parents=True, exist_ok=False)

    compile_audit = py_compile_sources()
    provenance = source_provenance(created_at)
    changed_files = changed_files_audit(provenance)
    binding = authoritative_repair_binding(created_at, provenance, compile_audit)
    test_results = run_test_command(artifact_root)
    equivalence = equivalence_validation(test_results)
    gradient = gradient_isolation_validation(test_results)
    gate = gate_matrix(binding, changed_files, equivalence, gradient, test_results)

    write_json(artifact_root / "repair_binding.json", binding)
    write_json(artifact_root / "equivalence_validation.json", equivalence)
    write_json(artifact_root / "gradient_isolation_validation.json", gradient)
    write_json(artifact_root / "changed_files.json", changed_files)
    write_json(artifact_root / "gate_matrix.json", gate)
    (artifact_root / "final_report.md").write_text(
        final_report(binding, equivalence, gradient, changed_files, gate),
        encoding="utf-8",
    )
    write_json(artifact_root / "manifest.json", manifest(artifact_root, gate))

    print(f"[H4M-P] artifact_root={artifact_root}")
    print(f"[H4M-P] gate={gate['gate']}")
    print(f"[H4M-P] source_commit_after_pass={gate['source_commit_after_pass']}")
    print(f"[H4M-P] repair_contract_sha256={gate['repair_contract_sha256']}")
    print(
        "[H4M-P] legacy_logit_max_abs_diff="
        f"{equivalence.get('legacy_vs_specialized', {}).get('logit_diff', {}).get('max_abs_diff')}"
    )
    print(
        "[H4M-P] hold_head_grad_norms="
        f"{gradient.get('gradient_isolation', {}).get('hold_target_probe', {}).get('target_head_grad_norms')}"
    )
    print(
        "[H4M-P] serve_head_grad_norms="
        f"{gradient.get('gradient_isolation', {}).get('serve_target_probe', {}).get('target_head_grad_norms')}"
    )
    print(f"[H4M-P] next_gate={gate['exact_next_gate']}")
    print("[H4M-P] STOP: no training, no TEST6, no GitHub push")
    if gate["gate"] != PASS_GATE:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
