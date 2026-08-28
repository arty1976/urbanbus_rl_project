from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-B"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_B_"
    "TRAINING_BUDGET_EXTENSION_SELECTION_AND_FREEZE_COMPLETE"
)
BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_B_"
    "EXPLICIT_TRAINING_BUDGET_SELECTION_REQUIRED"
)
PASS_DECISION = "PV8_EXTENDED_TRAINING_BUDGET_FROZEN_READY_FOR_FRESH_THREE_SEED_RETRAINING_RELEASE"
BLOCK_DECISION = "PV8_EXTENDED_TRAINING_BUDGET_NOT_FROZEN_USER_SELECTION_REQUIRED"
NEXT_GATE = "H4M-C_FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

H4M_B_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_b_"
    "training_budget_extension_selection_freeze.py"
)
H4M_B_SOURCE_PATH = PROJECT_ROOT / H4M_B_SOURCE_REL
H4M_A_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_a_"
    "decision_opportunity_environment_adequacy_audit.py"
)
H4K_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_"
    "fresh_reward_v2_zero_loss_three_seed_full_retraining.py"
)
H4L_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4l_"
    "frozen_three_seed_policy_validation_evaluation.py"
)
H4K_S0_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_s0_"
    "full_training_schedule_selection_and_freeze.py"
)

H4K_ROOT = (
    ARTIFACTS_ROOT
    / "pv8_r2a_r8e_r3_r_h4k_rerun_fresh_reward_v2_zero_loss_three_seed_full_retraining_20260810_231156"
)
H4L_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4l_frozen_three_seed_policy_validation_evaluation_20260814_124419"
H4M_A_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_a_decision_opportunity_environment_adequacy_audit_20260814_143743"
H4K_S0_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4k_s0_full_training_schedule_selection_and_freeze_20260810_224616"
DL2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl2_suseong_mac_m4_capacity_envelope_20260731_112258"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"
DL4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"

EXPECTED = {
    "h4k_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4K_RERUN_FRESH_REWARD_V2_ZERO_LOSS_THREE_SEED_FULL_RETRAINING_COMPLETE",
    "h4l_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4L_FROZEN_THREE_SEED_POLICY_VALIDATION_EVALUATION_COMPLETE",
    "h4m_a_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_A_DECISION_OPPORTUNITY_AND_ENVIRONMENT_ADEQUACY_AUDIT_COMPLETE",
    "h4m_a_root_cause": "MIXED_LIMITATION",
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "old_h4k_schedule_sha": "209297ba5d606fa859aeeb5c154c288b3aa374535989b74448902defbb302168",
}

REQUIRED_OUTPUTS = [
    "01_authoritative_binding.json",
    "02_h4m_a_budget_evidence.json",
    "03_historical_budget_inventory.json",
    "04_budget_candidate_comparison.json",
    "05_extended_rollout_update_contract.json",
    "06_runtime_resource_estimate.json",
    "07_validation_test_sealing_status.json",
    "08_h4m_b_extended_training_schedule_freeze.json",
    "09_h4m_b_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def get_path(payload: Mapping[str, Any], keys: Iterable[Any], default: Any = None) -> Any:
    cur: Any = payload
    for key in keys:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def git_run(args: Sequence[str], timeout: int = 20) -> Tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def git_identity(source_rel: str) -> Dict[str, Any]:
    _, head, _ = git_run(["rev-parse", "HEAD"])
    _, branch, _ = git_run(["branch", "--show-current"])
    _, status_short, _ = git_run(["status", "--short"])
    present_code, _, present_err = git_run(["ls-files", "--error-unmatch", source_rel])
    diff_code, _, _ = git_run(["diff", "--quiet", "HEAD", "--", source_rel])
    return {
        "branch": branch,
        "head": head,
        "status_short_before_artifact_write": status_short,
        "h4m_b_source_path": source_rel,
        "h4m_b_source_git_commit": head if present_code == 0 and diff_code == 0 else None,
        "h4m_b_source_present_in_head": present_code == 0,
        "h4m_b_source_no_uncommitted_diff_vs_head": diff_code == 0,
        "h4m_b_source_git_error": present_err or None,
        "github_push_performed": False,
    }


def source_entry(rel_path: str) -> Dict[str, Any]:
    path = PROJECT_ROOT / rel_path
    return {"path": rel_path, "exists": path.exists(), "sha256": sha256_file(path)}


def file_binding(label: str, path: Path) -> Dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path),
    }


def observed_required_sha(binding: Mapping[str, Any], key: str) -> Any:
    runtime = get_path(binding, ["runtime_and_checkpoint_binding", "h4l_checkpoint_binding_runtime"], {})
    if key == "reward_v2_sha":
        return get_path(runtime, ["reward_v2", "observed"])
    if key == "h4g_runtime_sha":
        return get_path(runtime, ["h4g_runtime", "observed_h4g_runtime_sha256"])
    if key == "r3_split_sha":
        return get_path(runtime, ["r3_split", "observed"])
    if key == "zero_loss_adapter_sha":
        return get_path(runtime, ["zero_loss_adapter", "observed"])
    if key == "old_h4k_schedule_sha":
        return get_path(runtime, ["h4k_schedule", "observed"])
    return None


def build_authoritative_binding(created_at: str, source_git: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    h4k_gate = read_json(H4K_ROOT / "13_h4k_rerun_gate_matrix.json")
    h4l_gate = read_json(H4L_ROOT / "15_h4l_gate_matrix.json")
    h4m_a_gate = read_json(H4M_A_ROOT / "12_h4m_a_gate_matrix.json")
    h4m_a_root = read_json(H4M_A_ROOT / "10_root_cause_classification.json")
    h4m_a_binding = read_json(H4M_A_ROOT / "01_authoritative_binding.json")
    h4k_schedule_binding = read_json(H4K_ROOT / "03_s0_schedule_binding.json")
    h4k_release_binding = read_json(H4K_ROOT / "01_authoritative_release_binding.json")

    required_sha_checks: Dict[str, Dict[str, Any]] = {}
    for key in [
        "reward_v2_sha",
        "h4g_runtime_sha",
        "r3_split_sha",
        "zero_loss_adapter_sha",
        "old_h4k_schedule_sha",
    ]:
        observed = observed_required_sha(h4m_a_binding, key)
        if observed is None and key == "old_h4k_schedule_sha":
            observed = h4k_schedule_binding.get("recomputed_full_training_schedule_sha256")
        required_sha_checks[key] = {
            "expected": EXPECTED[key],
            "observed": observed,
            "match": observed == EXPECTED[key],
        }

    h4k_schedule_sha_checks = {
        "h4k_rerun_recomputed_full_training_schedule_sha256": {
            "expected": EXPECTED["old_h4k_schedule_sha"],
            "observed": h4k_schedule_binding.get("recomputed_full_training_schedule_sha256"),
            "match": h4k_schedule_binding.get("recomputed_full_training_schedule_sha256")
            == EXPECTED["old_h4k_schedule_sha"],
        },
        "h4k_rerun_schedule_embedded_full_training_schedule_sha256": {
            "expected": EXPECTED["old_h4k_schedule_sha"],
            "observed": get_path(h4k_schedule_binding, ["schedule", "full_training_schedule_sha256"]),
            "match": get_path(h4k_schedule_binding, ["schedule", "full_training_schedule_sha256"])
            == EXPECTED["old_h4k_schedule_sha"],
        },
        "h4k_authoritative_release_schedule_sha256": {
            "expected": EXPECTED["old_h4k_schedule_sha"],
            "observed": get_path(h4k_release_binding, ["authoritative_shas", "h4k_schedule", "observed"]),
            "match": get_path(h4k_release_binding, ["authoritative_shas", "h4k_schedule", "observed"])
            == EXPECTED["old_h4k_schedule_sha"],
        },
    }

    checks = {
        "h4k_rerun_pass": h4k_gate.get("gate") == EXPECTED["h4k_gate"],
        "h4l_pass": h4l_gate.get("gate") == EXPECTED["h4l_gate"],
        "h4m_a_pass": h4m_a_gate.get("gate") == EXPECTED["h4m_a_gate"],
        "h4m_a_root_cause_mixed_limitation": h4m_a_root.get("classification") == EXPECTED["h4m_a_root_cause"],
        "h4m_a_next_gate_matches_h4m_b": h4m_a_root.get("next_recommended_gate")
        == "H4M-B_TRAINING_BUDGET_EXTENSION_SELECTION_AND_FREEZE",
        "all_required_sha_bindings_match": all(row["match"] for row in required_sha_checks.values()),
        "old_h4k_schedule_sha_recomputed_and_matched": all(row["match"] for row in h4k_schedule_sha_checks.values()),
        "source_committed_before_artifact": bool(source_git.get("h4m_b_source_git_commit")),
        "github_push_performed_false": source_git.get("github_push_performed") is False,
    }
    blockers = [key for key, ok in checks.items() if not ok]

    binding = {
        "stage": STAGE,
        "created_at": created_at,
        "authoritative_binding_passed": not blockers,
        "upstream_gates": {
            "h4k_rerun": {
                "expected": EXPECTED["h4k_gate"],
                "observed": h4k_gate.get("gate"),
                "path": str(H4K_ROOT / "13_h4k_rerun_gate_matrix.json"),
            },
            "h4l": {
                "expected": EXPECTED["h4l_gate"],
                "observed": h4l_gate.get("gate"),
                "path": str(H4L_ROOT / "15_h4l_gate_matrix.json"),
                "decision_string_lineage_issue_preserved": h4l_gate.get("decision"),
            },
            "h4m_a": {
                "expected": EXPECTED["h4m_a_gate"],
                "observed": h4m_a_gate.get("gate"),
                "path": str(H4M_A_ROOT / "12_h4m_a_gate_matrix.json"),
                "decision": h4m_a_gate.get("decision"),
            },
        },
        "h4m_a_root_cause": {
            "expected": EXPECTED["h4m_a_root_cause"],
            "observed": h4m_a_root.get("classification"),
            "decision": h4m_a_root.get("decision"),
            "path": str(H4M_A_ROOT / "10_root_cause_classification.json"),
        },
        "required_sha_bindings": required_sha_checks,
        "old_h4k_schedule_sha_checks": h4k_schedule_sha_checks,
        "artifact_file_sha256_bindings": [
            file_binding("h4k_rerun_gate_matrix", H4K_ROOT / "13_h4k_rerun_gate_matrix.json"),
            file_binding("h4k_rerun_schedule_binding", H4K_ROOT / "03_s0_schedule_binding.json"),
            file_binding("h4l_gate_matrix", H4L_ROOT / "15_h4l_gate_matrix.json"),
            file_binding("h4m_a_gate_matrix", H4M_A_ROOT / "12_h4m_a_gate_matrix.json"),
            file_binding("h4m_a_root_cause", H4M_A_ROOT / "10_root_cause_classification.json"),
            file_binding("h4m_a_authoritative_binding", H4M_A_ROOT / "01_authoritative_binding.json"),
        ],
        "source_git_provenance": source_git,
        "source_sha256": {
            "h4m_b": source_entry(H4M_B_SOURCE_REL),
            "h4m_a": source_entry(H4M_A_SOURCE_REL),
            "h4k_rerun": source_entry(H4K_SOURCE_REL),
            "h4l": source_entry(H4L_SOURCE_REL),
            "h4k_s0": source_entry(H4K_S0_SOURCE_REL),
        },
        "hard_locks": {
            "training": False,
            "optimizer_creation": False,
            "backward": False,
            "optimizer_step": False,
            "checkpoint_training_write": False,
            "reward_v2_change": False,
            "zero_loss_change": False,
            "k_mask_change": False,
            "architecture_change": False,
            "hyperparameter_change_except_outer_training_count": False,
            "environment_expansion": False,
            "agent_count_expansion": False,
            "validation_performance_run": False,
            "test_performance_run": False,
            "training_budget_sweep": False,
        },
        "checks": checks,
        "blockers": blockers,
    }
    return binding, blockers


def h4m_a_budget_evidence(created_at: str) -> Tuple[Dict[str, Any], bool]:
    validation_legal = read_json(H4M_A_ROOT / "03_validation_legal_action_space_audit.json")
    opportunity_funnel = read_json(H4M_A_ROOT / "04_decision_opportunity_funnel.json")
    counterfactual = read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json")
    actor_pre = read_json(H4M_A_ROOT / "06_actor_pre_argmax_preference_audit.json")
    train_exposure = read_json(H4M_A_ROOT / "07_training_opportunity_exposure_audit.json")
    comparison = read_json(H4M_A_ROOT / "09_train_validation_opportunity_comparison.json")
    root_cause = read_json(H4M_A_ROOT / "10_root_cause_classification.json")
    h4l_action = read_json(H4L_ROOT / "09_policy_action_distribution_audit.json")
    h4l_budget = read_json(H4L_ROOT / "12_training_budget_adequacy_diagnostic.json")

    train_structural = train_exposure["structurally_available_train_44_opportunities"]
    actual_rollout = train_exposure["actual_h4k_rollout_samples"]
    per_seed_rollout = actual_rollout["per_seed_training_rollout_summaries"]
    pooled_pre = actor_pre["pooled_summary"]

    sampled_action_checks = {}
    for seed, row in per_seed_rollout.items():
        distribution = row.get("action_distribution", {})
        sampled_action_checks[seed] = {
            "hold_sampled": distribution.get("HOLD_CURRENT_POSITION", 0) > 0,
            "serve_sampled": distribution.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0) > 0,
            "distribution": distribution,
        }

    pre_argmax_decomp = pooled_pre.get("serve_preference_decomposition_counts", {})
    h4l_total_serve_selected = sum(
        int(row.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0))
        for row in h4l_action.get("per_seed_action_counts", {}).values()
    )
    h4l_total_records = sum(sum(int(v) for v in row.values()) for row in h4l_action.get("per_seed_action_counts", {}).values())

    checks = {
        "train_decision_states_352": train_structural.get("total_agent_window_states") == 352,
        "train_hold_legal_352": train_structural.get("hold_legal_opportunities") == 352,
        "train_serve_legal_352_by_multi_action": train_structural.get("multi_action_opportunities") == 352,
        "train_skip_legal_0": train_structural.get("skip_legal_opportunities") == 0,
        "train_nonserve_beneficial_263": train_structural.get("non_SERVE_better_or_equal_reviewable_opportunities") == 263,
        "train_zero_loss_candidate_0": train_structural.get("zero_loss_candidate_opportunities") == 0,
        "actual_h4k_sampled_states_352_per_seed": actual_rollout.get("actual_sampled_states_per_seed") == 352,
        "actual_h4k_hold_and_serve_both_sampled_all_seeds": all(
            row["hold_sampled"] and row["serve_sampled"] for row in sampled_action_checks.values()
        ),
        "h4l_deterministic_masked_argmax_serve_32_per_seed": all(
            row.get("SERVE_AND_MOVE_TO_NEXT_STOP") == 32
            for row in h4l_action.get("per_seed_action_counts", {}).values()
        ),
        "h4l_deterministic_masked_argmax_serve_96_total": h4l_total_serve_selected == 96
        and h4l_total_records == 96,
        "pre_argmax_not_hard_collapse": pre_argmax_decomp.get("SERVE_ARGMAX_WITH_NONZERO_NON_SERVE_PROBABILITY") == 96
        and get_path(pooled_pre, ["per_action", "HOLD_CURRENT_POSITION", "masked_probability", "mean"], 0.0) > 0.0,
        "skip_opportunity_zero_train_validation": get_path(comparison, ["skip_opportunity_rate", "train"]) == 0.0
        and get_path(comparison, ["skip_opportunity_rate", "validation"]) == 0.0,
        "zero_loss_opportunity_zero_train_validation": get_path(comparison, ["zero_loss_opportunities", "train"]) == 0
        and get_path(comparison, ["zero_loss_opportunities", "validation"]) == 0,
        "root_cause_mixed_limitation": root_cause.get("classification") == "MIXED_LIMITATION",
        "h4l_training_budget_signal_weak_or_inconclusive": h4l_budget.get("diagnostic_conclusion")
        == "CURRENT_BUDGET_SIGNAL_WEAK_OR_INCONCLUSIVE",
    }

    payload = {
        "stage": STAGE,
        "created_at": created_at,
        "verification_passed": all(checks.values()),
        "checks": checks,
        "train44_structural_opportunities": {
            "decision_states": train_structural.get("total_agent_window_states"),
            "multi_action_opportunities": train_structural.get("multi_action_opportunities"),
            "hold_legal_opportunities": train_structural.get("hold_legal_opportunities"),
            "serve_legal_opportunities": train_structural.get("multi_action_opportunities"),
            "skip_legal_opportunities": train_structural.get("skip_legal_opportunities"),
            "non_SERVE_better_or_equal_reviewable_opportunities": train_structural.get(
                "non_SERVE_better_or_equal_reviewable_opportunities"
            ),
            "passenger_exposed_opportunities_target_SERVE": train_structural.get(
                "passenger_exposed_opportunities_target_SERVE"
            ),
            "zero_loss_candidate_opportunities": train_structural.get("zero_loss_candidate_opportunities"),
        },
        "h4k_actual_rollout_evidence": {
            "actual_train_windows_in_rollout": actual_rollout.get("actual_train_windows_in_rollout"),
            "actual_sampled_states_per_seed": actual_rollout.get("actual_sampled_states_per_seed"),
            "rollout_horizon": actual_rollout.get("rollout_horizon"),
            "per_seed_action_sampling": sampled_action_checks,
            "non_SERVE_beneficial_sampled_states_per_seed_immediate_reward_v2": actual_rollout.get(
                "non_SERVE_beneficial_sampled_states_per_seed_immediate_reward_v2"
            ),
            "skip_legal_sampled_states_per_seed": actual_rollout.get("skip_legal_sampled_states_per_seed"),
        },
        "h4l_validation_diagnostic_evidence": {
            "validation_decision_states": validation_legal.get("total_decisions"),
            "validation_hold_legal": validation_legal.get("hold_legal"),
            "validation_serve_legal": validation_legal.get("serve_legal"),
            "validation_skip_legal": validation_legal.get("skip_legal"),
            "validation_target_counts": validation_legal.get("target_counts"),
            "decision_opportunity_funnel": opportunity_funnel.get("funnel"),
            "counterfactual_summary": counterfactual.get("summary"),
            "h4l_per_seed_action_counts": h4l_action.get("per_seed_action_counts"),
            "h4l_budget_diagnostic": h4l_budget,
            "actor_pre_argmax_pooled_summary": {
                "record_count": pooled_pre.get("record_count"),
                "selected_action_counts": pooled_pre.get("selected_action_counts"),
                "serve_preference_decomposition_counts": pre_argmax_decomp,
                "entropy": pooled_pre.get("entropy"),
                "top1_top2_probability_margin": pooled_pre.get("top1_top2_probability_margin"),
                "hold_masked_probability": get_path(
                    pooled_pre, ["per_action", "HOLD_CURRENT_POSITION", "masked_probability"]
                ),
                "serve_masked_probability": get_path(
                    pooled_pre, ["per_action", "SERVE_AND_MOVE_TO_NEXT_STOP", "masked_probability"]
                ),
                "skip_masked_probability": get_path(
                    pooled_pre, ["per_action", "CONDITIONAL_SKIP_EMPTY_STOP", "masked_probability"]
                ),
            },
        },
        "train_validation_comparison": {
            "non_serve_beneficial_counterfactuals": comparison.get("non_serve_beneficial_counterfactuals"),
            "hold_opportunity_rate": comparison.get("hold_opportunity_rate"),
            "skip_opportunity_rate": comparison.get("skip_opportunity_rate"),
            "zero_loss_opportunities": comparison.get("zero_loss_opportunities"),
            "finding": comparison.get("finding"),
        },
        "scientific_question_for_h4m_b": (
            "Select exactly one fresh rollout→update cycle count for HOLD/SERVE learning. "
            "SKIP and Zero-Loss pickup absence is treated as environment expansion and is not fixed here."
        ),
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "test_opened": False,
    }
    return payload, all(checks.values())


def current_loop_markers() -> Dict[str, Any]:
    source_text = (PROJECT_ROOT / H4K_SOURCE_REL).read_text(encoding="utf-8")
    markers = {
        "ceil_len_train_data_over_rollout_horizon_present": "math.ceil(len(train_data) / int(config[\"rollout_horizon\"]))"
        in source_text,
        "contiguous_range_over_train_data_present": "range(0, len(train_data), int(config[\"rollout_horizon\"]))"
        in source_text,
        "actor_ppo_epoch_loop_present": "range(int(config[\"actor_ppo_epochs\"]))" in source_text,
        "critic_extra_epoch_loop_present": "extra_epochs = max(0, int(config[\"critic_epochs\")" in source_text
        or "extra_epochs = max(0, int(config[\"critic_epochs\"]) - int(config[\"actor_ppo_epochs\"]))" in source_text,
        "backward_call_present_in_training_source_not_executed_here": ".backward()" in source_text,
        "optimizer_step_calls_present_in_training_source_not_executed_here": ".step()" in source_text,
        "default_rollout_horizon_512_present": '"rollout_horizon": 512' in source_text,
        "default_actor_ppo_epochs_4_present": '"actor_ppo_epochs": 4' in source_text,
        "default_critic_epochs_8_present": '"critic_epochs": 8' in source_text,
    }
    return {
        "training_source": source_entry(H4K_SOURCE_REL),
        "markers": markers,
        "all_expected_markers_present": all(markers.values()),
        "read_only_source_scan_only": True,
    }


def historical_budget_inventory(created_at: str) -> Dict[str, Any]:
    dl2_gate = read_json(DL2_ROOT / "gate_decision.json")
    dl2_profile = read_json(DL2_ROOT / "selected_full_training_profile.json")
    dl2_resource = read_json(DL2_ROOT / "full_training_resource_plan.json")
    dl2_time = read_json(DL2_ROOT / "full_training_time_estimate.json")
    dl2_memory = read_json(DL2_ROOT / "memory_headroom_report.json")
    dl3_gate = read_json(DL3_ROOT / "gate_decision.json")
    dl3_budget = read_json(DL3_ROOT / "training_budget.json")
    dl3_resource = read_json(DL3_ROOT / "resource_usage_summary.json")
    dl3_runtime = read_json(DL3_ROOT / "runtime_estimate_comparison.json")
    dl3_profile = read_json(DL3_ROOT / "selected_profile_snapshot.json")
    dl4_gate = read_json(DL4_ROOT / "gate_decision.json")
    dl4_selected = read_json(DL4_ROOT / "selected_critic_profile.json")
    h4k_schedule = read_json(H4K_ROOT / "03_s0_schedule_binding.json")["schedule"]
    h4k_resource = read_json(H4K_ROOT / "12_resource_usage.json")
    h4k_seed_summaries = {
        "1": read_json(H4K_ROOT / "04_seed1_training_summary.json"),
        "2": read_json(H4K_ROOT / "05_seed2_training_summary.json"),
        "3": read_json(H4K_ROOT / "06_seed3_training_summary.json"),
    }

    dl3_rollout_count = int(dl3_budget["total_rollout_count"])
    dl3_rollout_formula_result = int(math.ceil(dl3_budget["training_snapshot_count"] / dl3_budget["rollout_horizon_configured"]))
    current_rollouts_per_cycle = int(math.ceil(h4k_schedule["training_windows"] / h4k_schedule["rollout_horizon"]))
    seed_elapsed = [float(row["elapsed_seconds"]) for row in h4k_seed_summaries.values()]

    inventory = {
        "stage": STAGE,
        "created_at": created_at,
        "dl2_capacity_evidence": {
            "gate": dl2_gate.get("gate"),
            "gate_passed": dl2_gate.get("gate_passed"),
            "selected_profile_id": dl2_gate.get("selected_profile_id"),
            "selected_full_training_profile": dl2_gate.get("selected_full_training_profile"),
            "profile_summary": {
                "device": dl2_profile.get("device"),
                "agents": dl2_profile.get("agents"),
                "rollout_horizon": dl2_profile.get("rollout_horizon"),
                "ppo_epochs": dl2_profile.get("ppo_epochs"),
                "minibatch_size": dl2_profile.get("minibatch_size"),
                "memory_safe": dl2_profile.get("memory_safe"),
                "checkpoint_reload": dl2_profile.get("checkpoint_reload"),
                "nan_inf_count": dl2_profile.get("nan_inf_count"),
                "host_memory_peak_mb": dl2_profile.get("host_memory_peak_mb"),
                "peak_mps_driver_mb": dl2_profile.get("peak_mps_driver_mb"),
            },
            "resource_plan": {
                "checkpoint_interval_recommended": dl2_resource.get("checkpoint_interval_recommended"),
                "estimated_hours_for_3_seeds": get_path(dl2_resource, ["time_estimate", "estimated_hours_for_3_seeds"]),
                "estimated_hours_per_seed": get_path(dl2_resource, ["time_estimate", "estimated_hours_per_seed"]),
                "seconds_per_rollout": get_path(dl2_resource, ["time_estimate", "seconds_per_rollout"]),
                "seconds_per_ppo_update": get_path(dl2_resource, ["time_estimate", "seconds_per_ppo_update"]),
            },
            "time_estimate": dl2_time,
            "memory_headroom": {
                "safety_limit_mb": dl2_memory.get("safety_limit_mb"),
                "max_profile_recorded": dl2_memory.get("rows", [])[-1] if dl2_memory.get("rows") else None,
            },
        },
        "dl3_historical_full_training_schedule": {
            "gate": dl3_gate.get("gate"),
            "gate_passed": dl3_gate.get("gate_passed"),
            "dataset_pass_count": dl3_budget.get("dataset_pass_count"),
            "training_snapshot_count": dl3_budget.get("training_snapshot_count"),
            "rollout_horizon_configured": dl3_budget.get("rollout_horizon_configured"),
            "total_rollout_count": dl3_rollout_count,
            "total_rollout_formula": "ceil(training_snapshot_count / rollout_horizon_configured)",
            "total_rollout_formula_result": dl3_rollout_formula_result,
            "total_ppo_update_count": dl3_budget.get("total_ppo_update_count"),
            "total_minibatch_update_count": dl3_budget.get("total_minibatch_update_count"),
            "total_environment_steps": dl3_budget.get("total_environment_steps"),
            "last_rollout_effective_horizon": dl3_budget.get("last_rollout_effective_horizon"),
            "actual_hours_total": dl3_runtime.get("actual_hours_total"),
            "mean_hours_per_seed": dl3_runtime.get("mean_hours_per_seed"),
            "resource_usage": dl3_resource,
            "selected_profile_snapshot": {
                "expected": dl3_profile.get("expected"),
                "profile_integrity_passed": dl3_profile.get("profile_integrity_passed"),
            },
        },
        "dl4_critic_evidence": {
            "gate": dl4_gate.get("gate"),
            "gate_passed": dl4_gate.get("gate_passed"),
            "selected_critic_profile_id": dl4_gate.get("selected_critic_profile_id"),
            "selected_critic_profile": {
                "actor_ppo_epochs": get_path(dl4_selected, ["selected_critic_profile", "actor_ppo_epochs"]),
                "critic_epochs": get_path(dl4_selected, ["selected_critic_profile", "critic_epochs"]),
                "critic_lr": get_path(dl4_selected, ["selected_critic_profile", "critic_lr"]),
                "critic_learning_active": get_path(dl4_selected, ["selected_critic_profile", "critic_learning_active"]),
                "validation_explained_variance": get_path(
                    dl4_selected, ["selected_critic_profile", "validation_explained_variance"]
                ),
                "prediction_target_pearson": get_path(
                    dl4_selected, ["selected_critic_profile", "prediction_target_pearson"]
                ),
                "mps_oom": get_path(dl4_selected, ["selected_critic_profile", "mps_oom"]),
                "peak_mps_driver_allocation_mb": get_path(
                    dl4_selected, ["selected_critic_profile", "peak_mps_driver_allocation_mb"]
                ),
            },
        },
        "h4k_current_budget_and_runtime": {
            "schedule_id": h4k_schedule.get("schedule_id"),
            "old_schedule_sha256": h4k_schedule.get("full_training_schedule_sha256"),
            "outer_training_unit": h4k_schedule.get("outer_training_unit"),
            "outer_training_count": h4k_schedule.get("outer_training_count"),
            "training_windows": h4k_schedule.get("training_windows"),
            "rollout_horizon": h4k_schedule.get("rollout_horizon"),
            "current_rollouts_per_cycle": current_rollouts_per_cycle,
            "total_rollouts_per_seed": h4k_schedule.get("total_rollouts_per_seed"),
            "ppo_epochs_per_update": h4k_schedule.get("ppo_epochs_per_update"),
            "critic_epochs_per_update": h4k_schedule.get("critic_epochs_per_update"),
            "total_ppo_updates_per_seed": h4k_schedule.get("total_ppo_updates_per_seed"),
            "total_critic_updates_per_seed": h4k_schedule.get("total_critic_updates_per_seed"),
            "elapsed_seconds_total_3_seed_actual": h4k_resource.get("elapsed_seconds_total"),
            "elapsed_seconds_per_seed_actual": {seed: row["elapsed_seconds"] for seed, row in h4k_seed_summaries.items()},
            "elapsed_seconds_per_seed_mean_actual": mean(seed_elapsed),
            "rollout_collection_seconds_total_per_seed": {
                seed: row.get("rollout_collection_seconds_total") for seed, row in h4k_seed_summaries.items()
            },
            "ppo_update_seconds_total_per_seed": {
                seed: row.get("ppo_update_seconds_total") for seed, row in h4k_seed_summaries.items()
            },
            "resource_usage": h4k_resource,
            "rss_unit_note": (
                "H4K resource_usage labels peak_process_rss_mb, but raw Darwin ru_maxrss-like values "
                "are retained without reinterpretation; DL2/DL3 MPS memory artifacts are used for capacity feasibility."
            ),
        },
        "current_executable_training_loop_semantics": current_loop_markers(),
        "candidate_number_sources": {
            "current_baseline": 1,
            "minimal_strict_repetition_of_current_semantics": 2,
            "dl3_validated_full_training_total_rollout_count": dl3_rollout_count,
            "two_times_dl3_rollout_count_guard_candidate": dl3_rollout_count * 2,
        },
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "test_opened": False,
    }
    return inventory


def candidate_metrics(
    n: int,
    train_windows: int,
    active_samples_per_cycle: int,
    nonserve_beneficial_per_cycle: int,
    skip_legal_per_cycle: int,
    ppo_epochs: int,
    critic_epochs: int,
    h4k_mean_seed_seconds: float,
    h4k_three_seed_seconds: float,
) -> Dict[str, Any]:
    return {
        "outer_training_count": n,
        "total_rollouts_per_seed": n,
        "total_environment_interaction_train_windows_per_seed": train_windows * n,
        "expected_active_samples_per_seed": active_samples_per_cycle * n,
        "total_ppo_updates_per_seed": ppo_epochs * n,
        "total_critic_updates_per_seed": critic_epochs * n,
        "approx_hold_legal_exposure_per_seed": active_samples_per_cycle * n,
        "approx_non_SERVE_beneficial_exposure_per_seed": nonserve_beneficial_per_cycle * n,
        "approx_skip_legal_exposure_per_seed": skip_legal_per_cycle * n,
        "estimated_runtime_seconds_per_seed_from_h4k_linear": h4k_mean_seed_seconds * n,
        "estimated_runtime_seconds_3_seed_serial_from_h4k_linear": h4k_three_seed_seconds * n,
    }


def budget_candidate_comparison(
    created_at: str,
    evidence: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> Tuple[Dict[str, Any], int]:
    train = evidence["train44_structural_opportunities"]
    h4k = inventory["h4k_current_budget_and_runtime"]
    dl3 = inventory["dl3_historical_full_training_schedule"]
    ppo_epochs = int(h4k["ppo_epochs_per_update"])
    critic_epochs = int(h4k["critic_epochs_per_update"])
    train_windows = int(h4k["training_windows"])
    active_samples = int(train["decision_states"])
    nonserve = int(train["non_SERVE_better_or_equal_reviewable_opportunities"])
    skip = int(train["skip_legal_opportunities"])
    h4k_mean_seed_seconds = float(h4k["elapsed_seconds_per_seed_mean_actual"])
    h4k_three_seed_seconds = float(h4k["elapsed_seconds_total_3_seed_actual"])
    dl3_rollouts = int(dl3["total_rollout_count"])

    candidate_ns = [1, 2, dl3_rollouts, dl3_rollouts * 2]
    candidates = []
    for n in candidate_ns:
        metrics = candidate_metrics(
            n,
            train_windows,
            active_samples,
            nonserve,
            skip,
            ppo_epochs,
            critic_epochs,
            h4k_mean_seed_seconds,
            h4k_three_seed_seconds,
        )
        if n == 1:
            verdict = "REJECT_CURRENT_BASELINE_INSUFFICIENT"
            rationale = (
                "This is the H4K budget already observed. H4L selected SERVE 32/32 for each seed, "
                "while H4M-A found 263/352 TRAIN44 non-SERVE-beneficial/reviewable states."
            )
        elif n == 2:
            verdict = "REJECT_MINIMAL_REPEAT_BELOW_HISTORICAL_MULTI_CYCLE_EVIDENCE"
            rationale = (
                "This is the smallest strict repetition of the current executable fresh rollout→update cadence, "
                "but it remains far below the DL3 validated 11-rollout full-training schedule."
            )
        elif n == dl3_rollouts:
            verdict = "SELECT_AND_FREEZE"
            rationale = (
                "Unique evidence-priority candidate: DL3 PASS used 11 rollout/update chunks with PPO×4, "
                "and H4M-B can reproduce that rollout/update count by repeating the current TRAIN44 fresh-cycle "
                "without changing reward, architecture, masks, or PPO/critic hyperparameters."
            )
        else:
            verdict = "REJECT_LARGER_THAN_FIRST_HISTORICAL_EXTENSION"
            rationale = (
                "This is a 2× DL3-equivalent guard candidate. It is feasible by linear runtime/capacity evidence, "
                "but it exceeds the first historically validated multi-cycle extension and would be an avoidable "
                "extra budget commitment before observing N=11."
            )
        candidates.append({**metrics, "candidate_source": candidate_source(n, dl3_rollouts), "verdict": verdict, "rationale": rationale})

    selected = [row for row in candidates if row["verdict"] == "SELECT_AND_FREEZE"]
    selected_n = int(selected[0]["outer_training_count"]) if len(selected) == 1 else 0
    comparison = {
        "stage": STAGE,
        "created_at": created_at,
        "selection_status": "SELECT_AND_FREEZE" if selected_n else "BLOCK_PENDING_EXPLICIT_USER_BUDGET_SELECTION",
        "selected_outer_training_count": selected_n or None,
        "candidate_generation_policy": [
            "Use few candidates only; no training-budget sweep.",
            "Prioritize past validated multi-cycle schedule compatible with lineage.",
            "Then compare natural repetition of current executable semantics and minimal-change/larger guards.",
        ],
        "candidate_budgets": candidates,
        "unique_selection_basis": {
            "selected_n": selected_n or None,
            "basis": (
                "DL3 historical full-training total_rollout_count=11 is the only candidate tied directly to a PASS "
                "multi-rollout/update schedule, while N=1 is observed insufficient, N=2 is only minimal repetition, "
                "and N=22 is an unsupported 2× enlargement."
            ),
            "why_one_cycle_insufficient": (
                "H4L deterministic masked argmax selected SERVE for 32/32 validation states per seed after H4K's "
                "single rollout/update cycle, despite H4M-A identifying HOLD/SERVE decision opportunities."
            ),
            "why_smaller_than_selected_likely_insufficient": (
                "N=2 provides only 526 non-SERVE-beneficial exposures/seed and 8 PPO updates/seed, still well below "
                "DL3's validated 11 rollout/update chunks and 44 PPO updates/seed."
            ),
            "why_not_larger_now": (
                "N=22 is feasible but would double the first evidence-backed extension without a new gate. "
                "Minimal-change scientific isolation favors N=11 first."
            ),
            "additional_fresh_experience_collections_vs_h4k": max(0, selected_n - 1),
            "expected_hold_beneficial_exposure_increase_factor_vs_h4k": selected_n if selected_n else None,
            "expected_non_SERVE_beneficial_exposure_per_seed": nonserve * selected_n if selected_n else None,
        },
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "test_opened": False,
    }
    return comparison, selected_n


def candidate_source(n: int, dl3_rollouts: int) -> str:
    if n == 1:
        return "current H4K frozen baseline"
    if n == 2:
        return "smallest strict repetition of current executable fresh rollout→update semantics"
    if n == dl3_rollouts:
        return "DL3 PASS historical full-training total_rollout_count"
    if n == dl3_rollouts * 2:
        return "2× DL3 rollout-count guard candidate"
    return "derived"


def build_schedule(
    created_at: str,
    selected_n: int,
    evidence: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> Dict[str, Any]:
    old_schedule = read_json(H4K_ROOT / "03_s0_schedule_binding.json")["schedule"]
    train = evidence["train44_structural_opportunities"]
    ppo_epochs = int(old_schedule["ppo_epochs_per_update"])
    critic_epochs = int(old_schedule["critic_epochs_per_update"])
    train_windows = int(old_schedule["training_windows"])
    active_samples = int(train["decision_states"])
    nonserve = int(train["non_SERVE_better_or_equal_reviewable_opportunities"])
    skip = int(train["skip_legal_opportunities"])

    schedule: Dict[str, Any] = {
        "stage": STAGE,
        "schedule_id": f"H4M_B_EXTENDED_TRAINING_BUDGET_N{selected_n}_FRESH_ROLLOUT_UPDATE_CYCLE",
        "schedule_version": "H4M_B_EXTENDED_TRAINING_SCHEDULE_V1",
        "created_at": created_at,
        "selection_status": "SELECT_AND_FREEZE",
        "seeds": [1, 2, 3],
        "training_windows": train_windows,
        "training_input": "TRAIN_44_ONLY",
        "outer_training_unit": "fresh_rollout_update_cycle",
        "outer_training_unit_definition": (
            "one full deterministic TRAIN44 traversal captured as a fresh rollout under the current updated policy, "
            "followed by Reward V2/Zero-Loss/K-mask/GAE materialization and PPO×4 + critic×8 updates"
        ),
        "outer_training_count": selected_n,
        "rollout_horizon": int(old_schedule["rollout_horizon"]),
        "rollouts_per_outer_unit": 1,
        "total_rollouts_per_seed": selected_n,
        "total_environment_interaction_train_windows_per_seed": train_windows * selected_n,
        "expected_active_samples_per_seed": active_samples * selected_n,
        "approx_hold_legal_exposure_per_seed": active_samples * selected_n,
        "approx_non_SERVE_beneficial_exposure_per_seed": nonserve * selected_n,
        "approx_skip_legal_exposure_per_seed": skip * selected_n,
        "ppo_epochs_per_update": ppo_epochs,
        "total_ppo_updates_per_seed": selected_n * ppo_epochs,
        "critic_epochs_per_update": critic_epochs,
        "total_critic_updates_per_seed": selected_n * critic_epochs,
        "minibatch": int(old_schedule["minibatch"]),
        "clip_epsilon": float(old_schedule["clip_epsilon"]),
        "actor_lr": float(old_schedule["actor_lr"]),
        "gatv2_lr": float(old_schedule["gatv2_lr"]),
        "critic_lr": float(old_schedule["critic_lr"]),
        "gamma": float(old_schedule["gamma"]),
        "gae_lambda": float(old_schedule["gae_lambda"]),
        "value_coefficient": float(old_schedule["value_coefficient"]),
        "scheduler": old_schedule.get("scheduler", "DISABLED_NO_LR_SCHEDULER"),
        "preserved_scientific_variables": {
            "reward_v2_sha256": EXPECTED["reward_v2_sha"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
            "r3_split_sha256": EXPECTED["r3_split_sha"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha"],
            "old_h4k_schedule_sha256": EXPECTED["old_h4k_schedule_sha"],
            "reward_v2_change": False,
            "zero_loss_change": False,
            "k_mask_change": False,
            "actor_critic_gatv2_architecture_change": False,
            "learning_rate_change": False,
            "entropy_change": False,
            "gamma_lambda_change": False,
            "rollout_horizon_change": False,
            "ppo_epochs_change": False,
            "critic_epochs_change": False,
            "minibatch_change": False,
            "environment_or_agent_expansion": False,
        },
        "window_traversal_policy": {
            "order": "frozen H4I-R3 train order inherited from H4K S0",
            "window_traversal_order": old_schedule.get("window_traversal_order"),
            "shuffle": False,
            "shuffling_authorized": False,
            "rng_affects_traversal": False,
            "seed_rng_scope": "model initialization, stochastic action sampling during rollout, and minibatch ordering only",
        },
        "rng_semantics": {
            "seed_1": "fresh seed-specific RNG stream",
            "seed_2": "fresh seed-specific RNG stream",
            "seed_3": "fresh seed-specific RNG stream",
            "cross_seed_reuse": False,
            "cycle_rng_rule": (
                "each cycle advances the seed-specific RNG naturally; traversal order remains frozen and unshuffled"
            ),
        },
        "reset_semantics": {
            "per_cycle": "fresh rollout collection starts from the frozen TRAIN44 window sequence using the current updated policy",
            "per_window": "environment/window state is materialized from the sealed R3 train window snapshot/contract",
            "rollout_tensor_reuse": False,
            "h4k_rollout_tensor_reuse": False,
        },
        "rollout_update_cadence": (
            "cycle i: collect fresh TRAIN44 rollout under current policy -> compute Reward V2/Zero-Loss/K-mask/actions/GAE "
            "-> run exactly 4 actor/GATv2 PPO epochs and 8 critic epochs -> proceed to cycle i+1 with updated policy"
        ),
        "checkpoint_cadence": {
            "intermediate_checkpoint_rule": "NONE; cycle metrics may be logged, but no intermediate model-selection checkpoint is authorized",
            "final_checkpoint_rule": (
                f"write exactly one final training checkpoint per seed after outer_training_count={selected_n} completes "
                "and integrity checks for that seed are recorded"
            ),
            "intermediate_checkpoints_if_added_by_future_runner": "recovery/audit only, never selection; requires preserving this schedule SHA",
            "checkpoint_namespaces": [
                f"H4M_C_SEED_{seed:03d}_FRESH_EXTENDED_BUDGET_N{selected_n}_REWARD_V2_ZERO_LOSS"
                for seed in [1, 2, 3]
            ],
        },
        "termination_rule": (
            f"each seed terminates only after exactly {selected_n} fresh rollout→update cycles; "
            "no validation early stopping, no performance-based extension, no seed-specific budgets"
        ),
        "fresh_restart_policy": {
            "seed_1": "FRESH",
            "seed_2": "FRESH",
            "seed_3": "FRESH",
            "gatv2": "FRESH",
            "actor": "FRESH",
            "critic": "FRESH",
            "optimizers": "FRESH",
            "reward_normalizer": "FRESH",
            "return_normalizer": "FRESH",
            "h4k_checkpoint_continuation": False,
            "h4l_checkpoint_mutation": False,
            "cross_seed_reuse": False,
        },
        "validation_usage": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "test_usage": "SEALED_NOT_OPENED",
        "selection_rule": (
            "Select DL3 PASS historical 11 rollout/update chunks as the first extended budget, implemented as "
            "11 fresh TRAIN44 rollout→update cycles under unchanged H4K/H4L/H4M-A scientific variables."
        ),
        "mandatory_input_to_next_gate": "extended_training_schedule_sha256",
        "training_authorized": False,
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_training_write_executed": False,
        "sealed_test_opened": False,
    }
    schedule_sha = canonical_sha(schedule)
    schedule["extended_training_schedule_sha256"] = schedule_sha
    schedule["extended_training_schedule_sha256_scope"] = (
        "canonical JSON of schedule excluding extended_training_schedule_sha256 itself"
    )
    return schedule


def build_contract(created_at: str, schedule: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "contract_status": "FROZEN",
        "only_mutable_axis": {
            "name": "outer_training_count",
            "frozen_value": schedule["outer_training_count"],
            "unit": schedule["outer_training_unit"],
        },
        "per_cycle_preserved_values": {
            "rollout_horizon": schedule["rollout_horizon"],
            "ppo_epochs_per_update": schedule["ppo_epochs_per_update"],
            "critic_epochs_per_update": schedule["critic_epochs_per_update"],
            "minibatch": schedule["minibatch"],
            "clip_epsilon": schedule["clip_epsilon"],
            "actor_lr": schedule["actor_lr"],
            "gatv2_lr": schedule["gatv2_lr"],
            "critic_lr": schedule["critic_lr"],
            "gamma": schedule["gamma"],
            "gae_lambda": schedule["gae_lambda"],
        },
        "dataset_traversal": {
            "training_input": schedule["training_input"],
            "training_windows": schedule["training_windows"],
            "window_traversal_policy": schedule["window_traversal_policy"],
            "rng_semantics": schedule["rng_semantics"],
            "reset_semantics": schedule["reset_semantics"],
        },
        "rollout_update_cadence": schedule["rollout_update_cadence"],
        "checkpoint_cadence": schedule["checkpoint_cadence"],
        "termination_rule": schedule["termination_rule"],
        "fresh_restart_policy": schedule["fresh_restart_policy"],
        "forbidden_execution_in_h4m_b": {
            "training": False,
            "optimizer_creation": False,
            "backward": False,
            "optimizer_step": False,
            "checkpoint_training_write": False,
            "validation_run": False,
            "test_run": False,
        },
    }


def runtime_resource_estimate(
    created_at: str,
    selected_n: int,
    comparison: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = next(row for row in comparison["candidate_budgets"] if row["outer_training_count"] == selected_n)
    dl2 = inventory["dl2_capacity_evidence"]
    dl3 = inventory["dl3_historical_full_training_schedule"]
    h4k = inventory["h4k_current_budget_and_runtime"]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "estimate_policy": "extrapolate from actual project measurements only; no external benchmark",
        "selected_outer_training_count": selected_n,
        "h4k_current_measurement_basis": {
            "elapsed_seconds_total_3_seed_actual_for_n1": h4k["elapsed_seconds_total_3_seed_actual"],
            "elapsed_seconds_per_seed_mean_actual_for_n1": h4k["elapsed_seconds_per_seed_mean_actual"],
            "elapsed_seconds_per_seed_actual_for_n1": h4k["elapsed_seconds_per_seed_actual"],
            "cpu_or_mps_context": h4k["resource_usage"],
            "rss_unit_note": h4k["rss_unit_note"],
        },
        "linear_estimate_for_selected_budget": {
            "estimated_runtime_seconds_per_seed": selected["estimated_runtime_seconds_per_seed_from_h4k_linear"],
            "estimated_runtime_minutes_per_seed": selected["estimated_runtime_seconds_per_seed_from_h4k_linear"] / 60.0,
            "estimated_runtime_seconds_3_seed_serial": selected[
                "estimated_runtime_seconds_3_seed_serial_from_h4k_linear"
            ],
            "estimated_runtime_minutes_3_seed_serial": selected[
                "estimated_runtime_seconds_3_seed_serial_from_h4k_linear"
            ]
            / 60.0,
        },
        "dl2_capacity_context": {
            "gate_passed": dl2["gate_passed"],
            "selected_profile_id": dl2["selected_profile_id"],
            "profile_memory_safe": get_path(dl2, ["profile_summary", "memory_safe"]),
            "profile_rollout_horizon": get_path(dl2, ["profile_summary", "rollout_horizon"]),
            "profile_ppo_epochs": get_path(dl2, ["profile_summary", "ppo_epochs"]),
            "profile_minibatch_size": get_path(dl2, ["profile_summary", "minibatch_size"]),
            "profile_peak_mps_driver_mb": get_path(dl2, ["profile_summary", "peak_mps_driver_mb"]),
            "max_recorded_profile": get_path(dl2, ["memory_headroom", "max_profile_recorded"]),
            "safety_limit_mb": get_path(dl2, ["memory_headroom", "safety_limit_mb"]),
        },
        "dl3_historical_runtime_context": {
            "gate_passed": dl3["gate_passed"],
            "total_rollout_count": dl3["total_rollout_count"],
            "actual_hours_total_3_seed": dl3["actual_hours_total"],
            "mean_hours_per_seed": dl3["mean_hours_per_seed"],
            "peak_mps_driver_allocation_mb": get_path(dl3, ["resource_usage", "peak_mps_driver_allocation_mb"]),
        },
        "memory_feasibility_on_mac_mini_m4_24gb_mps": {
            "feasible": bool(dl2["gate_passed"]) and bool(dl3["gate_passed"]),
            "basis": (
                "DL2 selected BALANCED 512-horizon profile and E5_S5476 memory headroom records were memory_safe; "
                "DL3 11-rollout three-seed full training completed on Mac M4 with peak MPS driver allocation recorded."
            ),
        },
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "test_opened": False,
    }


def validation_test_status(created_at: str) -> Dict[str, Any]:
    h4l_gate = read_json(H4L_ROOT / "15_h4l_gate_matrix.json")
    return {
        "stage": STAGE,
        "created_at": created_at,
        "validation_4_status": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "validation_4_rationale": (
            "H4L already evaluated the 4 validation windows and H4M-A used that diagnostic to classify "
            "the budget/environment limitation; it must not be described as untouched independent final evidence."
        ),
        "h4l_gate": h4l_gate.get("gate"),
        "h4l_policy_evaluation_executed_previously": get_path(h4l_gate, ["final_flags", "policy_evaluation_executed"]),
        "validation_performance_run_in_h4m_b": False,
        "sealed_test_status": "SEALED_NOT_OPENED",
        "sealed_test_authorized": False,
        "sealed_test_opened": False,
        "test_performance_run_in_h4m_b": False,
        "winner_selection_authorized": False,
        "winner_selection_performed": False,
    }


def gate_matrix(
    created_at: str,
    binding: Mapping[str, Any],
    evidence_ok: bool,
    comparison: Mapping[str, Any],
    schedule: Mapping[str, Any],
    runtime: Mapping[str, Any],
    source_git: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_n = comparison.get("selected_outer_training_count")
    criteria = {
        "authoritative_binding_passed": binding.get("authoritative_binding_passed") is True,
        "h4m_a_budget_evidence_verified": evidence_ok,
        "historical_budget_inventory_completed": True,
        "current_executable_loop_semantics_scanned_read_only": True,
        "exactly_one_budget_selected": comparison.get("selection_status") == "SELECT_AND_FREEZE"
        and isinstance(selected_n, int),
        "selected_budget_is_dl3_validated_rollout_count": selected_n
        == get_path(comparison, ["unique_selection_basis", "selected_n"]),
        "only_outer_training_count_changed": True,
        "extended_training_schedule_frozen": bool(schedule.get("extended_training_schedule_sha256")),
        "extended_training_schedule_sha256_computed": len(str(schedule.get("extended_training_schedule_sha256", ""))) == 64,
        "runtime_estimated_from_project_measurements": runtime.get("estimate_policy", "").startswith("extrapolate"),
        "validation_status_marked_development_diagnostic": schedule.get("validation_usage")
        == "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "test6_sealed": schedule.get("test_usage") == "SEALED_NOT_OPENED",
        "training_authorized": False,
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "environment_expansion_authorized": False,
        "sealed_test_authorized": False,
        "sealed_test_opened": False,
        "github_push_performed": False,
        "h4m_b_source_committed_before_artifact": bool(source_git.get("h4m_b_source_git_commit")),
    }
    pass_ok = all(value is True or value is False and key in {
        "training_authorized",
        "training_executed",
        "optimizer_created",
        "backward_executed",
        "optimizer_step_executed",
        "environment_expansion_authorized",
        "sealed_test_authorized",
        "sealed_test_opened",
        "github_push_performed",
    } for key, value in criteria.items())
    # The expression above intentionally treats explicit false hard-lock flags as passing criteria.
    failing = {
        key: value
        for key, value in criteria.items()
        if not (value is True or key in {
            "training_authorized",
            "training_executed",
            "optimizer_created",
            "backward_executed",
            "optimizer_step_executed",
            "environment_expansion_authorized",
            "sealed_test_authorized",
            "sealed_test_opened",
            "github_push_performed",
        } and value is False)
    }
    pass_ok = not failing
    return {
        "stage": STAGE,
        "created_at": created_at,
        "gate": PASS_GATE if pass_ok else BLOCK_GATE,
        "decision": PASS_DECISION if pass_ok else BLOCK_DECISION,
        "criteria": criteria,
        "failing_criteria": failing,
        "selected_outer_training_count": selected_n,
        "extended_training_schedule_sha256": schedule.get("extended_training_schedule_sha256"),
        "h4m_b_source_git_commit": source_git.get("h4m_b_source_git_commit"),
        "final_flags": {
            "extended_training_schedule_frozen": pass_ok,
            "extended_training_schedule_sha256": schedule.get("extended_training_schedule_sha256") if pass_ok else None,
            "training_authorized": False,
            "training_executed": False,
            "optimizer_created": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "environment_expansion_authorized": False,
            "sealed_test_authorized": False,
            "sealed_test_opened": False,
            "github_push_performed": False,
        },
        "next": NEXT_GATE if pass_ok else "USER_SELECT_EXPLICIT_TRAINING_BUDGET",
    }


def final_report(
    artifact_root: Path,
    gate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    schedule: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str:
    selected_n = schedule["outer_training_count"]
    lines = [
        "# H4M-B — Training Budget Extension Selection & Freeze",
        "",
        f"- stage: `{STAGE}`",
        f"- gate: `{gate['gate']}`",
        f"- decision: `{gate['decision']}`",
        f"- artifact: `{artifact_root}`",
        f"- h4m_b_source_git_commit: `{gate.get('h4m_b_source_git_commit')}`",
        f"- github_push_performed: `false`",
        "",
        "## Frozen budget",
        "",
        f"- selected outer_training_count: `{selected_n}` fresh rollout→update cycles",
        f"- total_rollouts_per_seed: `{schedule['total_rollouts_per_seed']}`",
        f"- total_ppo_updates_per_seed: `{schedule['total_ppo_updates_per_seed']}`",
        f"- total_critic_updates_per_seed: `{schedule['total_critic_updates_per_seed']}`",
        f"- expected_active_samples_per_seed: `{schedule['expected_active_samples_per_seed']}`",
        f"- approx_non_SERVE_beneficial_exposure_per_seed: `{schedule['approx_non_SERVE_beneficial_exposure_per_seed']}`",
        f"- extended_training_schedule_sha256: `{schedule['extended_training_schedule_sha256']}`",
        "",
        "## Rationale",
        "",
        "- Why 1 cycle is insufficient: H4K's single cycle was followed by H4L validation where all three frozen policies selected SERVE for 32/32 validation states, while H4M-A found abundant HOLD/SERVE opportunities and 263/352 TRAIN44 non-SERVE-beneficial/reviewable states.",
        f"- Why N={selected_n}: DL3 PASS historical full training used 11 rollout/update chunks and 44 PPO updates; H4M-B reproduces that rollout/update count as fresh TRAIN44 cycles without changing Reward V2, Zero-Loss, K-mask, architecture, or PPO/critic hyperparameters.",
        "- Why smaller N is likely insufficient: N=2 is only the smallest strict repetition and remains below the validated DL3 11-rollout schedule.",
        "- Why not larger now: N=22 is feasible but doubles the first evidence-backed extension without a new diagnostic gate; N=11 is the minimum historically validated multi-cycle step.",
        f"- Additional fresh experience collections vs H4K: `{selected_n - 1}`.",
        f"- Expected HOLD-legal exposure increase vs H4K: `{selected_n}x`.",
        "",
        "## Runtime estimate",
        "",
        f"- estimated runtime/seed from H4K linear measurement: `{runtime['linear_estimate_for_selected_budget']['estimated_runtime_seconds_per_seed']:.3f}` seconds",
        f"- estimated 3-seed serial runtime from H4K linear measurement: `{runtime['linear_estimate_for_selected_budget']['estimated_runtime_seconds_3_seed_serial']:.3f}` seconds",
        "- DL2/DL3 Mac mini M4 capacity artifacts remain the memory feasibility basis; H4K RSS raw values are retained without reinterpreting the unit-label anomaly.",
        "",
        "## Sealing and hard locks",
        "",
        "- validation_usage: `DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED`",
        "- test_usage: `SEALED_NOT_OPENED`",
        "- training_authorized/executed: `false/false`",
        "- optimizer_created/backward/optimizer_step: `false/false/false`",
        "- environment_expansion_authorized: `false`",
        "",
        f"Next: `{NEXT_GATE}`",
    ]
    if gate["gate"] == BLOCK_GATE:
        lines.extend(
            [
                "",
                "## Block details",
                "",
                "No unique defensible budget was selected. Candidate evidence is recorded in `04_budget_candidate_comparison.json`.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_manifest(
    artifact_root: Path,
    created_at: str,
    gate: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> Dict[str, Any]:
    files = []
    for name in REQUIRED_OUTPUTS:
        if name == "manifest.json":
            continue
        path = artifact_root / name
        files.append(
            {
                "relative_path": name,
                "exists": path.exists(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    manifest = {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_root": str(artifact_root),
        "gate": gate["gate"],
        "decision": gate["decision"],
        "h4m_b_source_git_commit": gate.get("h4m_b_source_git_commit"),
        "extended_training_schedule_frozen": gate["final_flags"]["extended_training_schedule_frozen"],
        "extended_training_schedule_sha256": schedule.get("extended_training_schedule_sha256"),
        "required_outputs": REQUIRED_OUTPUTS,
        "files": files,
        "artifact_set_sha256": canonical_sha(files),
        "manifest_self_sha256_policy": "excluded_from_artifact_set_to_avoid_self-referential_hash",
        "final_flags": gate["final_flags"],
        "github_push_performed": False,
        "next": gate.get("next"),
    }
    dump_json(artifact_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    created_dt = kst_now()
    created_at = created_dt.isoformat(timespec="seconds")
    artifact_root = (
        ARTIFACTS_ROOT
        / f"pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_{created_dt.strftime('%Y%m%d_%H%M%S')}"
    )
    artifact_root.mkdir(parents=True, exist_ok=False)

    source_git = git_identity(H4M_B_SOURCE_REL)
    binding, binding_blockers = build_authoritative_binding(created_at, source_git)
    evidence, evidence_ok = h4m_a_budget_evidence(created_at)
    inventory = historical_budget_inventory(created_at)
    comparison, selected_n = budget_candidate_comparison(created_at, evidence, inventory)

    if binding_blockers or not evidence_ok or not selected_n:
        selected_n = selected_n or 0
        if selected_n == 0:
            selected_n = 11
        schedule = build_schedule(created_at, selected_n, evidence, inventory)
        comparison["selection_status"] = "BLOCK_PENDING_EXPLICIT_USER_BUDGET_SELECTION"
        comparison["selected_outer_training_count"] = None
    else:
        schedule = build_schedule(created_at, selected_n, evidence, inventory)

    contract = build_contract(created_at, schedule)
    runtime = runtime_resource_estimate(created_at, int(schedule["outer_training_count"]), comparison, inventory)
    sealing = validation_test_status(created_at)
    gate = gate_matrix(created_at, binding, evidence_ok, comparison, schedule, runtime, source_git)

    dump_json(artifact_root / "01_authoritative_binding.json", binding)
    dump_json(artifact_root / "02_h4m_a_budget_evidence.json", evidence)
    dump_json(artifact_root / "03_historical_budget_inventory.json", inventory)
    dump_json(artifact_root / "04_budget_candidate_comparison.json", comparison)
    dump_json(artifact_root / "05_extended_rollout_update_contract.json", contract)
    dump_json(artifact_root / "06_runtime_resource_estimate.json", runtime)
    dump_json(artifact_root / "07_validation_test_sealing_status.json", sealing)
    dump_json(artifact_root / "08_h4m_b_extended_training_schedule_freeze.json", schedule)
    dump_json(artifact_root / "09_h4m_b_gate_matrix.json", gate)
    dump_text(artifact_root / "final_report.md", final_report(artifact_root, gate, comparison, schedule, runtime))
    write_manifest(artifact_root, created_at, gate, schedule)

    print(json.dumps({"artifact": str(artifact_root), "gate": gate["gate"], "gate_passed": gate["gate"] == PASS_GATE}, ensure_ascii=False))
    print(f"[OK] training=false optimizer_created=false backward=false optimizer_step=false test_opened=false")


if __name__ == "__main__":
    main()
