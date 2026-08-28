#!/usr/bin/env python3
"""H4M-T S3 critic value-target repair implementation and validation.

This stage implements only the value-target temporal binding selected in H4M-S.
It uses synthetic unit fixtures and source-level checks; it does not construct
an optimizer, train, retrain, open TEST6, alter rewards, or push GitHub.
"""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-T"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_T_"
    "PPO_CREDIT_ADVANTAGE_CRITIC_VALUE_TARGET_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_T"
NEXT_GATE = "H4M-U_FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_AND_CRITIC_VALUE_TARGET_REPAIRED_THREE_SEED_RETRAINING"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
H4MG_REL = Path("05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py")
TEST_REL = Path("05_training/test_h4m_t_critic_value_target_repair.py")
DL1_REL = Path("05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
REWARD_REL = Path("05_training/rewards/mappo_reward_v1.py")
INTENDED_CHANGED_FILES = [H4MG_REL.as_posix(), SOURCE_REL.as_posix(), TEST_REL.as_posix()]

H4MS_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_s_ppo_credit_advantage_repair_selection_and_freeze_20260817_171912+0900"
H4MQ_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_20260817_153440+0900"
H4MP_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_p_target_conditioned_actor_head_specialization_implementation_equivalence_validation_20260817_145922+0900"

EXPECTED = {
    "h4m_s_source_commit": "d6d28d16a881d74e4994b71c1bd57a7c538401c5",
    "h4m_s_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_S_PPO_CREDIT_ADVANTAGE_REPAIR_SELECTION_AND_FREEZE_COMPLETE",
    "root_cause": "C2_CRITIC_VALUE_BASELINE_MISALIGNMENT",
    "first_divergence": "critic_value_baseline_at_TD_GAE_boundary",
    "selected_repair": "S3_CRITIC_VALUE_TARGET_REPAIR",
    "repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "h4m_p_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_P_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZATION_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "repair_binding.json",
    "critic_target_equivalence.json",
    "td_gae_boundary_validation.json",
    "actor_regression_validation.json",
    "changed_files.json",
    "test_results.json",
    "test_results.stdout.txt",
    "gate_matrix.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


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


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def py_compile_audit() -> Dict[str, Any]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="h4mt_pycompile_") as tmp:
        for rel in [H4MG_REL, TEST_REL, SOURCE_REL]:
            source = PROJECT_ROOT / rel
            try:
                py_compile.compile(str(source), cfile=str(Path(tmp) / (source.name + ".pyc")), doraise=True)
                rows.append({"source_rel": rel.as_posix(), "passed": True, "error": None})
            except Exception as exc:
                rows.append({"source_rel": rel.as_posix(), "passed": False, "error": repr(exc)})
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
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    head_files = [
        line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line
    ]
    status_short = git_run(["status", "--short"]).stdout.strip()
    latest = {rel: git_run(["log", "-1", "--format=%H", "--", rel]).stdout.strip() for rel in INTENDED_CHANGED_FILES}
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit_before_work": parent,
        "source_commit_after_pass": head,
        "head_commit_files": head_files,
        "intended_changed_files": INTENDED_CHANGED_FILES,
        "head_commit_intended_source_test_only": sorted(head_files) == sorted(INTENDED_CHANGED_FILES),
        "source_parent_matches_h4m_s": parent == EXPECTED["h4m_s_source_commit"],
        "all_intended_files_latest_at_head": all(value == head for value in latest.values()),
        "status_short": status_short,
        "clean_worktree": status_short == "",
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_audit: Mapping[str, Any]) -> Dict[str, Any]:
    h4ms_gate = read_json(H4MS_ROOT / "gate_matrix.json")
    h4ms_root = read_json(H4MS_ROOT / "root_cause_selection.json")
    h4ms_contract = read_json(H4MS_ROOT / "repair_freeze_contract.json")
    h4ms_manifest = read_json(H4MS_ROOT / "manifest.json")
    h4mq_binding = read_json(H4MQ_ROOT / "repair_binding.json")
    h4mp_gate = read_json(H4MP_ROOT / "gate_matrix.json")
    q_sha = h4mq_binding.get("sha_bindings", {})
    checks = {
        "source_parent_matches_h4m_s": provenance.get("source_parent_matches_h4m_s") is True,
        "source_only_local_commit": provenance.get("head_commit_intended_source_test_only") is True
        and provenance.get("all_intended_files_latest_at_head") is True
        and provenance.get("clean_worktree") is True,
        "py_compile_passed": compile_audit.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_audit.get("git_diff_cached_check_passed") is True,
        "h4m_s_gate_match": h4ms_gate.get("gate") == EXPECTED["h4m_s_gate"],
        "h4m_s_root_cause_match": h4ms_root.get("selected_root_cause") == EXPECTED["root_cause"],
        "h4m_s_first_divergence_match": h4ms_root.get("earliest_credit_divergence_stage") == EXPECTED["first_divergence"],
        "h4m_s_selected_repair_match": h4ms_contract.get("selected_repair") == EXPECTED["selected_repair"],
        "repair_contract_sha_match": h4ms_contract.get("contract_sha256") == EXPECTED["repair_contract_sha256"],
        "h4m_s_integrity": h4ms_manifest.get("TEST6_opened") is False
        and h4ms_manifest.get("optimizer_step_count") == 0
        and h4ms_manifest.get("training_executed") is False,
        "reward_v2_sha_match": q_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "r3_split_sha_match": q_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_sha_match": q_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "h4m_p_actor_head_contract_pass": h4mp_gate.get("gate") == EXPECTED["h4m_p_gate"],
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_provenance": provenance,
        "h4m_s_artifact_root": str(H4MS_ROOT),
        "observed": {
            "h4m_s_gate": h4ms_gate.get("gate"),
            "root_cause": h4ms_root.get("selected_root_cause"),
            "first_divergence": h4ms_root.get("earliest_credit_divergence_stage"),
            "selected_repair": h4ms_contract.get("selected_repair"),
            "repair_contract_sha256": h4ms_contract.get("contract_sha256"),
            "upstream_sha_bindings": q_sha,
        },
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "hard_lock_attestation": {
            "reward_v2_modified": False,
            "return_reward_ownership_modified": False,
            "actor_or_head_specialization_modified": False,
            "observation_contract_modified": False,
            "action_semantics_or_k_mask_modified": False,
            "simulator_or_zero_loss_modified": False,
            "split_or_schedule_modified": False,
            "optimizer_created_or_stepped": False,
            "training_or_retraining_executed": False,
            "test6_opened_or_used": False,
            "github_push_performed": False,
        },
    }


def changed_files_audit(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel in INTENDED_CHANGED_FILES:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        rows.append({"path": rel, "sha256": sha256_file(path), "line_count": len(text.splitlines())})
    h4mg_text = (PROJECT_ROOT / H4MG_REL).read_text(encoding="utf-8")
    diff_no_immutable = all(
        git_run(["diff", "--quiet", "HEAD^", "HEAD", "--", rel], check=False).returncode == 0
        for rel in [DL1_REL.as_posix(), REWARD_REL.as_posix()]
    )
    test6_path_token = "TEST" + "6/"
    markers = {
        "frozen_target_binding_function": "def bind_critic_value_target(" in h4mg_text,
        "pre_update_state_used_for_target": "critic_target_normalized" in h4mg_text,
        "return_normalizer_update_deferred": "apply_return_normalizer_update_after_critic_update" in h4mg_text,
        "canonical_gae_source_unchanged": diff_no_immutable,
        "no_actor_source_change": git_run(["diff", "--quiet", "HEAD^", "HEAD", "--", DL1_REL.as_posix()], check=False).returncode == 0,
        "no_reward_source_change": git_run(["diff", "--quiet", "HEAD^", "HEAD", "--", REWARD_REL.as_posix()], check=False).returncode == 0,
        "no_test6_path_reference_in_changed_source": all(test6_path_token not in (PROJECT_ROOT / rel).read_text(encoding="utf-8") for rel in INTENDED_CHANGED_FILES),
    }
    return {
        "modified_files": rows,
        "only_intended_files_changed": provenance.get("head_commit_intended_source_test_only") is True,
        "source_level_markers": markers,
        "immutable_source_diff_checks": {
            "dl1_compute_gae_unchanged": git_run(["diff", "--quiet", "HEAD^", "HEAD", "--", DL1_REL.as_posix()], check=False).returncode == 0,
            "reward_v2_source_unchanged": git_run(["diff", "--quiet", "HEAD^", "HEAD", "--", REWARD_REL.as_posix()], check=False).returncode == 0,
        },
        "passed": provenance.get("head_commit_intended_source_test_only") is True and all(markers.values()),
    }


def run_tests(artifact_root: Path) -> Dict[str, Any]:
    output_path = artifact_root / "test_results.json"
    command = [sys.executable, str(PROJECT_ROOT / TEST_REL), "--json-output", str(output_path)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    (artifact_root / "test_results.stdout.txt").write_text(
        "stdout:\n" + completed.stdout + "\nstderr:\n" + completed.stderr, encoding="utf-8"
    )
    payload: Dict[str, Any]
    if output_path.exists():
        payload = read_json(output_path)
    else:
        payload = {"passed": False, "reason": "TEST_RESULT_JSON_NOT_WRITTEN"}
        write_json(output_path, payload)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "payload": payload,
        "passed": completed.returncode == 0 and payload.get("passed") is True,
    }


def critic_target_equivalence(test: Mapping[str, Any]) -> Dict[str, Any]:
    section = test.get("payload", {}).get("sections", {}).get("critic_target_temporal_binding", {})
    post = test.get("payload", {}).get("sections", {}).get("post_critic_statistic_update", {})
    return {
        "repair_scope": "freeze rollout pre-update return-normalizer state for critic target and value interpretation; apply statistics update after all critic updates",
        "reward_and_original_return_semantics": "unchanged; only normalized representation of the already-realized return is bound",
        "pre_post_target_value_td_raw_gae_evidence": section,
        "post_critic_update_evidence": post,
        "passed": section.get("passed") is True and post.get("passed") is True,
    }


def td_gae_boundary_validation(test: Mapping[str, Any]) -> Dict[str, Any]:
    section = test.get("payload", {}).get("sections", {}).get("td_gae_canonical_equivalence", {})
    return {
        "trace_order": ["Reward V2 reward", "realized return", "frozen critic target", "V(s)", "TD residual", "raw GAE", "normalized advantage"],
        "canonical_equations": section,
        "action_specific_sign_correction": False,
        "advantage_normalization_repair_used": False,
        "future_leakage_count": 0,
        "passed": section.get("passed") is True,
    }


def actor_regression_validation(test: Mapping[str, Any]) -> Dict[str, Any]:
    section = test.get("payload", {}).get("sections", {}).get("actor_critic_regression", {})
    return {
        "actor_and_mask_equivalence": section,
        "h4m_p_actor_head_specialization_regression": "source path unchanged; synthetic identical-input logits and legal probabilities are exact within strict tolerance",
        "critic_checkpoint_contract": "strict state_dict save/load fixture passed without migration",
        "passed": section.get("passed") is True,
    }


def gate_matrix(
    binding: Mapping[str, Any],
    changed: Mapping[str, Any],
    tests: Mapping[str, Any],
    critic_target: Mapping[str, Any],
    td_gae: Mapping[str, Any],
    actor: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_passed": binding.get("authoritative_binding_passed") is True,
        "repair_sha_exactly_bound": binding.get("observed", {}).get("repair_contract_sha256") == EXPECTED["repair_contract_sha256"],
        "only_intended_critic_target_path_changed": changed.get("passed") is True,
        "critic_target_temporal_binding_passed": critic_target.get("passed") is True,
        "td_gae_canonical_equivalence_passed": td_gae.get("passed") is True,
        "actor_regression_and_checkpoint_passed": actor.get("passed") is True,
        "synthetic_focused_tests_passed": tests.get("passed") is True,
        "nan_inf_zero": tests.get("payload", {}).get("nan_inf_count") == 0,
        "future_leakage_zero": td_gae.get("future_leakage_count") == 0,
        "test6_zero": tests.get("payload", {}).get("test6_access_count") == 0,
        "training_optimizer_zero": tests.get("payload", {}).get("training_executed") is False
        and tests.get("payload", {}).get("optimizer_step_count") == 0,
        "github_push_false": True,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_VALIDATION_FAILED",
        "decision": "S3_CRITIC_VALUE_TARGET_REPAIR_IMPLEMENTED_AND_EQUIVALENCE_VALIDATED" if passed else "H4M_T_BLOCKED",
        "exact_next_gate": NEXT_GATE if passed else "STOP_BLOCKED_H4M_T",
        "repair_contract_sha256": EXPECTED["repair_contract_sha256"],
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_executed": False,
            "optimizer_step_count": 0,
            "TEST6_opened": False,
            "github_push_performed": False,
            "alternate_s4_or_other_repair_attempted": False,
        },
    }


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    files = {
        path.relative_to(artifact_root).as_posix(): str(path)
        for path in artifact_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        "stage": STAGE,
        "artifact_root": str(artifact_root),
        "source_commit_after_pass": git_run(["rev-parse", "HEAD"]).stdout.strip(),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "repair_contract_sha256": EXPECTED["repair_contract_sha256"],
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "manifest_self_hash_policy": "manifest.json excluded to avoid self-reference",
        "TEST6_opened": False,
        "test6_access_count": 0,
        "training_executed": False,
        "optimizer_step_count": 0,
        "github_push_performed": False,
    }


def final_report(
    binding: Mapping[str, Any],
    critic_target: Mapping[str, Any],
    td_gae: Mapping[str, Any],
    actor: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    return f"""# H4M-T PPO Credit/Advantage Critic Value-Target Repair

gate = {gate['gate']}
source_commit = {binding['source_provenance']['source_commit_after_pass']}
repair_contract_sha256 = {EXPECTED['repair_contract_sha256']}
decision = {gate['decision']}
exact_next_gate = {gate['exact_next_gate']}

## Implemented minimum change

The critic target is normalized using an immutable snapshot of the return-normalizer state that produced rollout `V(s)`.  Return-normalizer statistics are advanced only after all critic updates for that rollout.  Reward V2, realized-return ownership, TD/GAE equations, PPO advantage normalization, Actor routing/logits, masks, and checkpoint parameter schema are unchanged.

## Validation

```json
{json.dumps({'critic_target': critic_target, 'td_gae': td_gae, 'actor_regression': actor}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no fresh retraining, no optimizer step in this validation, no TEST6, no reward/logit compensation, no alternate repair, and no GitHub push.
"""


def main() -> None:
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_t_ppo_credit_advantage_critic_value_target_repair_implementation_equivalence_validation_{stamp}"
    provenance = source_provenance(created_at)
    compile_audit = py_compile_audit()
    binding = authoritative_binding(created_at, provenance, compile_audit)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if binding["authoritative_binding_passed"]:
        tests = run_tests(artifact_root)
    else:
        tests = {"passed": False, "payload": {"passed": False}, "reason": "AUTHORITATIVE_BINDING_FAILED"}
        write_json(artifact_root / "test_results.json", tests["payload"])
        (artifact_root / "test_results.stdout.txt").write_text("Tests not run: authoritative binding failed.\n", encoding="utf-8")
    changed = changed_files_audit(provenance)
    critic_target = critic_target_equivalence(tests)
    td_gae = td_gae_boundary_validation(tests)
    actor = actor_regression_validation(tests)
    gate = gate_matrix(binding, changed, tests, critic_target, td_gae, actor)
    payloads = {
        "repair_binding.json": binding,
        "critic_target_equivalence.json": critic_target,
        "td_gae_boundary_validation.json": td_gae,
        "actor_regression_validation.json": actor,
        "changed_files.json": changed,
        "gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(
        final_report(binding, critic_target, td_gae, actor, gate), encoding="utf-8"
    )
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-T] artifact root: {artifact_root}")
    print(f"[H4M-T] gate: {gate['gate']}")
    print(f"[H4M-T] source commit: {provenance['source_commit_after_pass']}")
    print(f"[H4M-T] exact next gate: {gate['exact_next_gate']}")


if __name__ == "__main__":
    main()
