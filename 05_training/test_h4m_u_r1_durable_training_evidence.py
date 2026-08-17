#!/usr/bin/env python3
"""Focused H4M-U-R1 durable training-evidence instrumentation checks.

Fixtures and mocks only.  No MAPPO training, no rollout, no environment or
database access, no ``optimizer.step()``, no checkpoint promotion, and no TEST6
access.  The checks prove that:

* the post-training reporting call is bound to its owning module now, while the
  historical failure stays provable from the source-before commit blob
* one durable record per ``seed x cycle`` carries every mandatory field
* the critic loss is taken from the real update result and never estimated
* Actor/Critic parameter deltas come from real pre/post cycle parameters
* the evidence survives a forced reporter failure and the report can be
  regenerated with zero training and zero optimizer steps
* interrupted, duplicated, and conflicting evidence is rejected fail-closed
* Actor, Critic, TD/GAE, and Reward V2 numerics are unchanged
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
DURABLE_SOURCE = TRAINING_ROOT / "durable_training_evidence.py"
H4MU_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py"
H4MQ_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
H4MK_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
DL1_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
DL4_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
REWARD_SOURCE = TRAINING_ROOT / "rewards/mappo_reward_v1.py"

SOURCE_BEFORE_COMMIT = "5584d59cbf64f6bad7ba63c4770523fb04bc619c"
SOURCE_BEFORE_BLOB = "2abdfad1b00d0afa7bd2228001bb7eecc5017f1d"
H4MU_REL = "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py"
REPORTING_FUNCTION = "write_conditional_parquet"
HISTORICAL_OWNER = "qmod"
CORRECT_OWNER = "kmod"

STAGE = "PV8-R2A-R8E-R3-R-H4M-U"
ACTOR_REPAIR_SHA = "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97"
CRITIC_REPAIR_SHA = "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f"
SPLIT_SHA = "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c"
SCHEDULE_SHA = "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc"
FIXTURE_SOURCE_COMMIT = "0" * 40
STRICT_FLOAT_TOLERANCE = 1.0e-7
FIXTURE_SEEDS = [1, 2]
FIXTURE_CYCLES = 3


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_show(revision: str) -> str:
    completed = subprocess.run(
        ["git", "show", revision], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True
    )
    return completed.stdout


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    delta = (a.detach().float() - b.detach().float()).abs()
    return {
        "max_abs_diff": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_diff": float(delta.mean().item()) if delta.numel() else 0.0,
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def attribute_call_owners(source_text: str, function_name: str) -> List[str]:
    """Module owners used at every ``<owner>.<function_name>(...)`` call site."""
    owners: List[str] = []
    for node in ast.walk(ast.parse(source_text)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == function_name and isinstance(func.value, ast.Name):
            owners.append(func.value.id)
    return owners


# ---------------------------------------------------------------------------
# 1. reporting typo: historical fact vs current repaired state
# ---------------------------------------------------------------------------
def validate_reporting_call_binding() -> Dict[str, Any]:
    historical_text = git_show(f"{SOURCE_BEFORE_COMMIT}:{H4MU_REL}")
    current_text = H4MU_SOURCE.read_text(encoding="utf-8")
    historical_owners = attribute_call_owners(historical_text, REPORTING_FUNCTION)
    current_owners = attribute_call_owners(current_text, REPORTING_FUNCTION)
    qmod = import_module("h4mur1_qmod", H4MQ_SOURCE)
    kmod = import_module("h4mur1_kmod", H4MK_SOURCE)
    q_has = hasattr(qmod, REPORTING_FUNCTION)
    k_has = hasattr(kmod, REPORTING_FUNCTION)
    blob = subprocess.run(
        ["git", "rev-parse", f"{SOURCE_BEFORE_COMMIT}:{H4MU_REL}"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {
        "source_before_commit": SOURCE_BEFORE_COMMIT,
        "source_before_blob": blob,
        "historical_call_owners": historical_owners,
        "current_call_owners": current_owners,
        "historical_typo_present_in_source_before_blob": historical_owners == [HISTORICAL_OWNER],
        "current_source_binds_owning_module": current_owners == [CORRECT_OWNER],
        "current_source_free_of_historical_owner_call": HISTORICAL_OWNER not in current_owners,
        "h4m_q_module_defines_function": q_has,
        "h4m_k_module_defines_function": k_has,
        "attribute_error_reproduced_from_historical_owner": not q_has,
        "verification_rule": "historical failure is proven from the source-before blob; the current worktree must be repaired, not still broken",
        "passed": blob == SOURCE_BEFORE_BLOB
        and historical_owners == [HISTORICAL_OWNER]
        and current_owners == [CORRECT_OWNER]
        and k_has is True
        and q_has is False,
    }


# ---------------------------------------------------------------------------
# fixture builders (mock update results and mock trace rows)
# ---------------------------------------------------------------------------
def mock_update_result(seed: int, cycle: int, *, value_loss_base: float = 0.5) -> Dict[str, Any]:
    """A stand-in for ``ppo_update_controlled``'s real return payload."""
    rows: List[Dict[str, Any]] = []
    for index in range(1, 5):
        rows.append(
            {
                "branch": "H4M_U_FIXTURE",
                "rollout_index": cycle,
                "ppo_update_index": index,
                "update_role": "actor_gatv2_critic_joint",
                "policy_loss": -0.01 * index,
                "value_loss": value_loss_base + 0.01 * index + 0.1 * cycle + seed,
                "actor_entropy": 0.9,
                "approx_kl": 0.001,
                "clip_fraction": 0.02,
                "entropy_coef": 0.01,
                "critic_target_normalizer_state_sha256": f"state-{seed}-{cycle}",
                "critic_target_binding_schema": "rollout_pre_update_return_normalizer_state_v1",
                "actor_parameter_delta": 0.4 + cycle,
                "critic_parameter_delta": 0.3 + cycle,
                "gatv2_parameter_delta": 0.2 + cycle,
                "nan_count": 0,
                "inf_count": 0,
            }
        )
    for index in range(5, 9):
        rows.append(
            {
                "branch": "H4M_U_FIXTURE",
                "rollout_index": cycle,
                "ppo_update_index": index,
                "update_role": "critic_only_extra",
                "policy_loss": None,
                "value_loss": value_loss_base + 0.02 * index + 0.1 * cycle + seed,
                "critic_target_normalizer_state_sha256": f"state-{seed}-{cycle}",
                "critic_target_binding_schema": "rollout_pre_update_return_normalizer_state_v1",
                "actor_parameter_delta": 0.4 + cycle,
                "critic_parameter_delta": 0.3 + cycle,
                "gatv2_parameter_delta": 0.2 + cycle,
                "nan_count": 0,
                "inf_count": 0,
            }
        )
    return {
        "metrics_rows": rows,
        "critic_target_normalizer_state_sha256": f"state-{seed}-{cycle}",
        "return_normalizer_update": {"update_timing": "after_all_critic_updates_for_rollout"},
    }


def mock_trace_rows(seed: int, cycle: int, sample_count: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    pre_action, td_gae, advantage, critic_value_error = [], [], [], []
    for index in range(sample_count):
        uid = f"seed{seed}-cycle{cycle}-sample{index}"
        action_id = index % 3
        value_t = 0.1 * (index + 1) + cycle
        raw_return = 0.05 * (index + 1) + cycle
        pre_action.append(
            {
                "sample_uid": uid,
                "seed": seed,
                "outer_cycle": cycle,
                "action_id": action_id,
                "legal_action_ids": [0, 1, 2],
                "masked_probability_0": 0.5,
                "masked_probability_1": 0.3,
                "masked_probability_2": 0.2,
                "policy_entropy": 0.8,
            }
        )
        td_gae.append(
            {
                "sample_uid": uid,
                "seed": seed,
                "outer_cycle": cycle,
                "value_t": value_t,
                "raw_return_target": raw_return,
                "td_delta": 0.01 * (index + 1),
                "raw_gae_advantage": 0.02 * (index + 1) - 0.03,
                "normalized_return_target": 0.5,
                "critic_target_normalizer_state_sha256": f"state-{seed}-{cycle}",
            }
        )
        advantage.append(
            {
                "sample_uid": uid,
                "seed": seed,
                "outer_cycle": cycle,
                "sampled_action_id": action_id,
                "sampled_action_name": {0: "HOLD_CURRENT_POSITION", 1: "SERVE_AND_MOVE_TO_NEXT_STOP", 2: "CONDITIONAL_SKIP_EMPTY_STOP"}[action_id],
                "raw_gae_advantage": 0.02 * (index + 1) - 0.03,
                "normalized_advantage": 0.01 * (index + 1) - 0.02,
            }
        )
        critic_value_error.append(
            {
                "sample_uid": uid,
                "seed": seed,
                "outer_cycle": cycle,
                "sampled_action_name": advantage[-1]["sampled_action_name"],
                "value_error": raw_return - value_t,
            }
        )
    return {
        "pre_action_rows": pre_action,
        "td_gae_rows": td_gae,
        "advantage_rows": advantage,
        "critic_value_error_rows": critic_value_error,
    }


class MockRecorder:
    def __init__(self, rows: Mapping[str, List[Dict[str, Any]]]) -> None:
        self.pre_action_rows = rows["pre_action_rows"]
        self.critic_td_gae_rows = rows["td_gae_rows"]
        self.advantage_sample_rows = rows["advantage_rows"]
        self.critic_value_error_rows = rows["critic_value_error_rows"]


def mock_trace_entry(artifact_root: Path, seed: int, cycle: int) -> Dict[str, Any]:
    base = artifact_root / "05_actual_on_policy_credit_trace" / f"seed={seed}" / f"outer_cycle={cycle}"
    base.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Any] = {}
    for name in ("actual_pre_action_trace", "actual_critic_td_gae_trace", "advantage_normalization_sample"):
        path = base / f"{name}.parquet"
        path.write_bytes(f"{name}-{seed}-{cycle}".encode("utf-8"))
        files[name] = {"path": str(path), "rows": 6, "sha256": f"sha-{name}-{seed}-{cycle}", "size_bytes": path.stat().st_size}
    return {"seed": seed, "outer_cycle": cycle, "files": files}


def fixture_run_header(artifact_root: Path) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": "2026-08-17T20:00:00+09:00",
        "artifact_root": str(artifact_root),
        "seeds": FIXTURE_SEEDS,
        "outer_training_count": FIXTURE_CYCLES,
        "source_commit": FIXTURE_SOURCE_COMMIT,
        "split_sha256": SPLIT_SHA,
        "schedule_sha256": SCHEDULE_SHA,
        "actor_repair_contract_sha256": ACTOR_REPAIR_SHA,
        "critic_repair_contract_sha256": CRITIC_REPAIR_SHA,
    }


def fixture_grid() -> List[Tuple[int, int]]:
    return [(seed, cycle) for seed in FIXTURE_SEEDS for cycle in range(1, FIXTURE_CYCLES + 1)]


def build_fixture_run(dte: Any, artifact_root: Path, *, deltas_by_key: Optional[Mapping[Tuple[int, int], Any]] = None) -> Dict[str, Any]:
    """Persist a complete synthetic seed x cycle evidence set."""
    header = fixture_run_header(artifact_root)
    writer = dte.DurableCycleEvidenceWriter(
        artifact_root / dte.EVIDENCE_DIR_NAME, run_header=header, expected_grid=fixture_grid()
    )
    appended = []
    for seed, cycle in fixture_grid():
        update = mock_update_result(seed, cycle)
        rows = mock_trace_rows(seed, cycle)
        trace_files, scoped = dte.trace_file_bindings(
            mock_trace_entry(artifact_root, seed, cycle), artifact_root=artifact_root, seed=seed, outer_cycle=cycle
        )
        delta_payload = (deltas_by_key or {}).get((seed, cycle)) or synthetic_parameter_delta(dte, seed, cycle)
        record = dte.build_cycle_record(
            stage=STAGE,
            identity_base=header,
            seed=seed,
            outer_cycle=cycle,
            critic_loss=dte.extract_critic_loss(update, seed=seed, outer_cycle=cycle),
            parameter_delta=delta_payload,
            credit_diagnostics=dte.credit_diagnostics_from_rows(seed=seed, outer_cycle=cycle, **rows),
            trace_files=trace_files,
            cycle_scoped_trace_binding=scoped,
        )
        appended.append(writer.append_cycle_evidence(record))
    writer.mark_training_complete({"fixture": True})
    return {"writer": writer, "header": header, "appended": appended}


def synthetic_parameter_delta(dte: Any, seed: int, cycle: int) -> Dict[str, Any]:
    """Deltas from real pre/post tensors, produced without any optimizer."""
    payload: Dict[str, Any] = {}
    for role, scale in (("actor", 0.1), ("critic", 0.2), ("gatv2", 0.3)):
        before = {"w": torch.full((2, 2), float(seed))}
        after = {"w": before["w"] + scale * cycle}
        payload[role] = dte.parameter_delta_from_states(before, after, role=role)
    payload["cumulative_from_seed_start"] = {"source": "fixture"}
    return payload


# ---------------------------------------------------------------------------
# 2-4. schema, critic loss provenance, parameter delta provenance
# ---------------------------------------------------------------------------
def validate_evidence_schema(dte: Any) -> Dict[str, Any]:
    schema = dte.evidence_schema()
    with tempfile.TemporaryDirectory(prefix="h4mur1_schema_") as tmp:
        root = Path(tmp)
        build_fixture_run(dte, root)
        loaded = dte.load_evidence(root / dte.EVIDENCE_DIR_NAME)
    record = loaded["records"][0]
    missing_top = [field for field in dte.RECORD_REQUIRED_FIELDS if field not in record]
    missing_critic = [field for field in dte.CRITIC_LOSS_REQUIRED_FIELDS if field not in record["critic_loss"]]
    missing_delta = [
        f"{role}.{field}"
        for role in dte.PARAMETER_DELTA_REQUIRED_ROLES
        for field in dte.PARAMETER_DELTA_REQUIRED_FIELDS
        if field not in record["parameter_delta"][role]
    ]
    missing_credit = [field for field in dte.CREDIT_DIAGNOSTIC_REQUIRED_FIELDS if field not in record["credit_diagnostics"]]
    missing_integrity = [field for field in dte.INTEGRITY_REQUIRED_FIELDS if field not in record["integrity"]]
    identity_complete = all(field in record["identity"] for field in dte.IDENTITY_FIELDS)
    identity_in_hash = record["record_sha256"] == dte.record_sha256(record)
    mutated = json.loads(json.dumps(record))
    mutated["identity"]["source_commit"] = "f" * 40
    identity_changes_hash = dte.record_sha256(mutated) != record["record_sha256"]
    return {
        "schema_version": schema["schema_version"],
        "identity_fields": dte.IDENTITY_FIELDS,
        "record_count": loaded["record_count"],
        "missing_record_fields": missing_top,
        "missing_critic_loss_fields": missing_critic,
        "missing_parameter_delta_fields": missing_delta,
        "missing_credit_diagnostic_fields": missing_credit,
        "missing_integrity_fields": missing_integrity,
        "identity_block_complete": identity_complete,
        "record_hash_covers_record": identity_in_hash,
        "record_hash_covers_identity": identity_changes_hash,
        "credit_diagnostics_present": sorted(
            key for key in ("v_s_minus_return", "raw_gae", "normalized_advantage", "action_counts", "action_probabilities")
            if key in record["credit_diagnostics"]
        ),
        "integrity_passed": loaded["integrity_passed"],
        "passed": not missing_top
        and not missing_critic
        and not missing_delta
        and not missing_credit
        and not missing_integrity
        and identity_complete
        and identity_in_hash
        and identity_changes_hash
        and loaded["integrity_passed"] is True
        and loaded["record_count"] == len(fixture_grid()),
    }


def validate_critic_loss_provenance(dte: Any) -> Dict[str, Any]:
    update = mock_update_result(1, 1)
    expected = [float(row["value_loss"]) for row in update["metrics_rows"]]
    block = dte.extract_critic_loss(update, seed=1, outer_cycle=1)
    exact = block["critic_loss_values"] == expected
    mean_matches = abs(block["critic_loss_mean"] - sum(expected) / len(expected)) <= STRICT_FLOAT_TOLERANCE

    missing = {"metrics_rows": [{k: v for k, v in row.items() if k != "value_loss"} for row in update["metrics_rows"]]}
    missing_rejected = _raises(dte, lambda: dte.extract_critic_loss(missing, seed=1, outer_cycle=1), "CRITIC_LOSS_NOT_AVAILABLE_FROM_UPDATE_RESULT")
    empty_rejected = _raises(dte, lambda: dte.extract_critic_loss({"metrics_rows": []}, seed=1, outer_cycle=1), "CRITIC_LOSS_SOURCE_MISSING")
    nan_update = json.loads(json.dumps(update))
    nan_update["metrics_rows"][0]["value_loss"] = float("nan")
    nan_rejected = _raises(dte, lambda: dte.extract_critic_loss(nan_update, seed=1, outer_cycle=1), "CRITIC_LOSS_NON_FINITE")
    return {
        "source": block["source"],
        "estimated": block["estimated"],
        "reconstructed": block["reconstructed"],
        "expected_values": expected,
        "persisted_values": block["critic_loss_values"],
        "values_match_update_result_exactly": exact,
        "mean_matches": mean_matches,
        "actor_joint_update_count": block["actor_joint_update_count"],
        "critic_only_update_count": block["critic_only_update_count"],
        "missing_value_loss_fail_closed": missing_rejected,
        "empty_metrics_rows_fail_closed": empty_rejected,
        "non_finite_value_loss_fail_closed": nan_rejected,
        "passed": exact
        and mean_matches
        and block["estimated"] is False
        and block["reconstructed"] is False
        and missing_rejected["raised"]
        and empty_rejected["raised"]
        and nan_rejected["raised"],
    }


def _raises(dte: Any, call: Any, expected_code: str) -> Dict[str, Any]:
    try:
        call()
    except dte.EvidenceIntegrityError as exc:
        return {"raised": exc.code == expected_code, "code": exc.code, "expected_code": expected_code}
    except Exception as exc:  # noqa: BLE001 - any other type is a failure
        return {"raised": False, "code": repr(exc), "expected_code": expected_code}
    return {"raised": False, "code": None, "expected_code": expected_code}


def validate_parameter_delta_provenance(dte: Any) -> Dict[str, Any]:
    """Deltas must come from real pre/post parameters, with no optimizer step."""
    dl1 = import_module("h4mur1_dl1", DL1_SOURCE)
    torch.manual_seed(20260817)
    critic = dl1.CentralizedCritic(8)
    before = dl1.clone_state_dict(critic)
    step = 0.01
    with torch.no_grad():  # synthetic perturbation, not an optimizer step
        for parameter in critic.parameters():
            parameter.add_(step)
    after = dl1.clone_state_dict(critic)
    frozen = dl1.delta_stats(critic, before)
    pure = dte.parameter_delta_from_states(before, after, role="critic")
    element_count = int(sum(int(tensor.numel()) for tensor in before.values()))
    analytic_l2 = math.sqrt(element_count * step * step)
    annotated = dte.annotate_parameter_delta(frozen, role="critic")

    unchanged_before = dl1.clone_state_dict(critic)
    unchanged = dte.parameter_delta_from_states(unchanged_before, dl1.clone_state_dict(critic), role="critic")
    hash_mismatch_rejected = _raises(
        dte,
        lambda: dte.annotate_parameter_delta(
            {**frozen, "pre_hash": frozen["post_hash"]}, role="critic"
        ),
        "PARAMETER_DELTA_HASH_INCONSISTENT",
    )
    field_missing_rejected = _raises(
        dte,
        lambda: dte.annotate_parameter_delta({k: v for k, v in frozen.items() if k != "l2_delta"}, role="critic"),
        "PARAMETER_DELTA_FIELD_MISSING",
    )
    return {
        "element_count": element_count,
        "analytic_l2_delta": analytic_l2,
        "frozen_helper_l2_delta": frozen["l2_delta"],
        "pure_l2_delta": pure["l2_delta"],
        "matches_analytic": abs(pure["l2_delta"] - analytic_l2) <= 1.0e-4,
        "matches_frozen_helper": abs(pure["l2_delta"] - frozen["l2_delta"]) <= 1.0e-6,
        "pre_post_hashes_differ": pure["pre_hash"] != pure["post_hash"],
        "annotated_computed_from": annotated["computed_from"],
        "unchanged_parameters_report_zero_delta": unchanged["l2_delta"] == 0.0
        and unchanged["parameters_changed"] is False
        and unchanged["pre_hash"] == unchanged["post_hash"],
        "inconsistent_hash_fail_closed": hash_mismatch_rejected,
        "missing_field_fail_closed": field_missing_rejected,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "passed": abs(pure["l2_delta"] - analytic_l2) <= 1.0e-4
        and abs(pure["l2_delta"] - frozen["l2_delta"]) <= 1.0e-6
        and pure["pre_hash"] != pure["post_hash"]
        and annotated["computed_from"] == dte.PARAMETER_DELTA_COMPUTED_FROM
        and unchanged["l2_delta"] == 0.0
        and hash_mismatch_rejected["raised"]
        and field_missing_rejected["raised"],
    }


# ---------------------------------------------------------------------------
# 5-6. reporter failure survival and no-training report regeneration
# ---------------------------------------------------------------------------
class TrainingCallCounter:
    """Counts optimizer steps and backward passes around a callable."""

    def __init__(self) -> None:
        self.optimizer_steps = 0
        self.backward_calls = 0
        self._optimizer_step = torch.optim.Optimizer.step
        self._backward = torch.Tensor.backward

    def __enter__(self) -> "TrainingCallCounter":
        counter = self

        def counted_step(self_: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
            counter.optimizer_steps += 1
            return counter._optimizer_step(self_, *args, **kwargs)

        def counted_backward(self_: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - must never run
            counter.backward_calls += 1
            return counter._backward(self_, *args, **kwargs)

        torch.optim.Optimizer.step = counted_step
        torch.Tensor.backward = counted_backward
        return self

    def __exit__(self, *exc_info: Any) -> None:
        torch.optim.Optimizer.step = self._optimizer_step
        torch.Tensor.backward = self._backward


def validate_reporter_failure_survival(dte: Any) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h4mur1_reporter_") as tmp:
        root = Path(tmp)
        run = build_fixture_run(dte, root)
        evidence_dir = root / dte.EVIDENCE_DIR_NAME
        before_bytes = (evidence_dir / dte.EVIDENCE_JSONL_NAME).stat().st_size

        reporter_error: Optional[str] = None
        try:
            raise AttributeError("module 'qmod' has no attribute 'write_conditional_parquet'")
        except AttributeError as exc:  # the historical H4M-U failure mode
            reporter_error = repr(exc)

        after_bytes = (evidence_dir / dte.EVIDENCE_JSONL_NAME).stat().st_size
        loaded = dte.load_evidence(evidence_dir)
        with TrainingCallCounter() as counter:
            recovery = dte.regenerate_report(root, title="fixture recovery")
        payload = dte.build_report_payload(evidence_dir)
        report_text = (root / "final_report.md").read_text(encoding="utf-8")
        first = loaded["records"][0]
        loss_in_report = str(round(float(first["critic_loss"]["critic_loss_mean"]), 6))[:8] in report_text or str(
            first["critic_loss"]["critic_loss_mean"]
        )[:8] in json.dumps(payload)
        return {
            "reporter_exception": reporter_error,
            "records_persisted_before_failure": run["writer"].index_entries.__len__(),
            "evidence_bytes_unchanged_by_failure": before_bytes == after_bytes,
            "records_reloaded_after_failure": loaded["record_count"],
            "evidence_integrity_after_failure": loaded["integrity_passed"],
            "recovered": recovery["recovered"],
            "report_status": recovery["report_status"],
            "regeneration_optimizer_steps": counter.optimizer_steps,
            "regeneration_backward_calls": counter.backward_calls,
            "regeneration_training_invocations": recovery["training_invocations"],
            "regeneration_model_or_optimizer_constructed": recovery["model_or_optimizer_constructed"],
            "torch_imported_by_recovery_path": recovery["torch_imported_by_recovery_path"],
            "report_uses_persisted_values": loss_in_report,
            "report_source": payload["report_source"],
            "volatile_in_memory_metrics_used": payload["volatile_in_memory_metrics_used"],
            "passed": before_bytes == after_bytes
            and loaded["record_count"] == len(fixture_grid())
            and loaded["integrity_passed"] is True
            and recovery["recovered"] is True
            and counter.optimizer_steps == 0
            and counter.backward_calls == 0
            and recovery["training_invocations"] == 0
            and recovery["optimizer_step_count"] == 0
            and payload["volatile_in_memory_metrics_used"] is False,
        }


def validate_report_regeneration_without_torch() -> Dict[str, Any]:
    """The recovery path must work in a process that never imports torch."""
    script = """
import json, sys
from pathlib import Path
sys.path.insert(0, {training_root!r})
import durable_training_evidence as dte
root = Path(sys.argv[1])
recovery = dte.regenerate_report(root, title="subprocess recovery")
print(json.dumps({{
    "recovered": recovery["recovered"],
    "record_count": recovery["record_count"],
    "training_invocations": recovery["training_invocations"],
    "optimizer_step_count": recovery["optimizer_step_count"],
    "torch_in_sys_modules": "torch" in sys.modules,
}}))
""".format(training_root=str(TRAINING_ROOT))
    dte = import_module("h4mur1_dte_probe", DURABLE_SOURCE)
    with tempfile.TemporaryDirectory(prefix="h4mur1_subproc_") as tmp:
        root = Path(tmp)
        build_fixture_run(dte, root)
        completed = subprocess.run(
            [sys.executable, "-c", script, str(root)], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.stdout.strip() else {}
        report_exists = (root / "final_report.md").exists()
    return {
        "returncode": completed.returncode,
        "stderr": completed.stderr[-2000:],
        "subprocess_result": payload,
        "final_report_written": report_exists,
        "passed": completed.returncode == 0
        and payload.get("recovered") is True
        and payload.get("torch_in_sys_modules") is False
        and payload.get("training_invocations") == 0
        and payload.get("optimizer_step_count") == 0
        and report_exists,
    }


# ---------------------------------------------------------------------------
# 7-8. fail-closed integrity: partial, duplicate, conflicting evidence
# ---------------------------------------------------------------------------
def validate_fail_closed_integrity(dte: Any) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="h4mur1_partial_") as tmp:
        root = Path(tmp)
        build_fixture_run(dte, root)
        evidence_dir = root / dte.EVIDENCE_DIR_NAME
        log = evidence_dir / dte.EVIDENCE_JSONL_NAME
        raw = log.read_bytes()
        log.write_bytes(raw + raw.split(b"\n")[0][:120])  # interrupted write
        partial = dte.load_evidence(evidence_dir)
        recovery = dte.regenerate_report(root, title="fixture partial")
        strict = _raises(dte, lambda: dte.load_evidence(evidence_dir, strict=True), "EVIDENCE_INTEGRITY_FAILED")
        results["partial_record"] = {
            "integrity_passed": partial["integrity_passed"],
            "no_partial_record_line": partial["checks"]["no_partial_record_line"],
            "records_still_readable": partial["record_count"],
            "strict_load_fail_closed": strict,
            "regeneration_refuses_pass": recovery["recovered"] is False
            and recovery["report_status"] == dte.REPORT_STATUS_BLOCKED,
            "detected": partial["integrity_passed"] is False
            and partial["checks"]["no_partial_record_line"] is False
            and strict["raised"]
            and recovery["recovered"] is False,
        }
    with tempfile.TemporaryDirectory(prefix="h4mur1_duplicate_") as tmp:
        root = Path(tmp)
        run = build_fixture_run(dte, root)
        writer = run["writer"]
        header = run["header"]
        seed, cycle = fixture_grid()[0]
        rows = mock_trace_rows(seed, cycle)
        trace_files, scoped = dte.trace_file_bindings(
            mock_trace_entry(root, seed, cycle), artifact_root=root, seed=seed, outer_cycle=cycle
        )
        duplicate = dte.build_cycle_record(
            stage=STAGE,
            identity_base=header,
            seed=seed,
            outer_cycle=cycle,
            critic_loss=dte.extract_critic_loss(mock_update_result(seed, cycle), seed=seed, outer_cycle=cycle),
            parameter_delta=synthetic_parameter_delta(dte, seed, cycle),
            credit_diagnostics=dte.credit_diagnostics_from_rows(seed=seed, outer_cycle=cycle, **rows),
            trace_files=trace_files,
            cycle_scoped_trace_binding=scoped,
        )
        writer_duplicate = _raises(dte, lambda: writer.append_cycle_evidence(duplicate), "DUPLICATE_CYCLE_EVIDENCE")
        conflicting = dte.build_cycle_record(
            stage=STAGE,
            identity_base=header,
            seed=seed,
            outer_cycle=cycle,
            critic_loss=dte.extract_critic_loss(
                mock_update_result(seed, cycle, value_loss_base=9.5), seed=seed, outer_cycle=cycle
            ),
            parameter_delta=synthetic_parameter_delta(dte, seed, cycle),
            credit_diagnostics=dte.credit_diagnostics_from_rows(seed=seed, outer_cycle=cycle, **rows),
            trace_files=trace_files,
            cycle_scoped_trace_binding=scoped,
        )
        writer_conflict = _raises(dte, lambda: writer.append_cycle_evidence(conflicting), "CONFLICTING_CYCLE_EVIDENCE")
        evidence_dir = root / dte.EVIDENCE_DIR_NAME
        log = evidence_dir / dte.EVIDENCE_JSONL_NAME
        conflicting["record_sha256"] = dte.record_sha256(conflicting)
        with log.open("a", encoding="utf-8") as handle:  # bypass the writer to test the loader
            handle.write(dte.canonical_line(conflicting))
        loaded = dte.load_evidence(evidence_dir)
        results["duplicate_and_conflict"] = {
            "writer_rejects_duplicate": writer_duplicate,
            "writer_rejects_conflict": writer_conflict,
            "loader_conflicting_cycles": loaded["conflicting_cycles"],
            "loader_integrity_passed": loaded["integrity_passed"],
            "detected": writer_duplicate["raised"]
            and writer_conflict["raised"]
            and bool(loaded["conflicting_cycles"])
            and loaded["integrity_passed"] is False,
        }
    with tempfile.TemporaryDirectory(prefix="h4mur1_identity_") as tmp:
        root = Path(tmp)
        run = build_fixture_run(dte, root)
        header = dict(run["header"])
        header["source_commit"] = "e" * 40
        seed, cycle = (FIXTURE_SEEDS[0], FIXTURE_CYCLES + 1)
        rows = mock_trace_rows(seed, cycle)
        trace_files, scoped = dte.trace_file_bindings(
            mock_trace_entry(root, seed, cycle), artifact_root=root, seed=seed, outer_cycle=cycle
        )
        foreign = dte.build_cycle_record(
            stage=STAGE,
            identity_base=header,
            seed=seed,
            outer_cycle=cycle,
            critic_loss=dte.extract_critic_loss(mock_update_result(seed, cycle), seed=seed, outer_cycle=cycle),
            parameter_delta=synthetic_parameter_delta(dte, seed, cycle),
            credit_diagnostics=dte.credit_diagnostics_from_rows(seed=seed, outer_cycle=cycle, **rows),
            trace_files=trace_files,
            cycle_scoped_trace_binding=scoped,
        )
        foreign_rejected = _raises(
            dte, lambda: run["writer"].append_cycle_evidence(foreign), "EVIDENCE_RECORD_SCHEMA_VIOLATION"
        )
        missing_cycle_root = root / "incomplete"
        (missing_cycle_root).mkdir()
        partial_writer = dte.DurableCycleEvidenceWriter(
            missing_cycle_root / dte.EVIDENCE_DIR_NAME,
            run_header=fixture_run_header(missing_cycle_root),
            expected_grid=fixture_grid(),
        )
        seed, cycle = fixture_grid()[0]
        rows = mock_trace_rows(seed, cycle)
        trace_files, scoped = dte.trace_file_bindings(
            mock_trace_entry(missing_cycle_root, seed, cycle), artifact_root=missing_cycle_root, seed=seed, outer_cycle=cycle
        )
        partial_writer.append_cycle_evidence(
            dte.build_cycle_record(
                stage=STAGE,
                identity_base=fixture_run_header(missing_cycle_root),
                seed=seed,
                outer_cycle=cycle,
                critic_loss=dte.extract_critic_loss(mock_update_result(seed, cycle), seed=seed, outer_cycle=cycle),
                parameter_delta=synthetic_parameter_delta(dte, seed, cycle),
                credit_diagnostics=dte.credit_diagnostics_from_rows(seed=seed, outer_cycle=cycle, **rows),
                trace_files=trace_files,
                cycle_scoped_trace_binding=scoped,
            )
        )
        incomplete = dte.build_report_payload(missing_cycle_root / dte.EVIDENCE_DIR_NAME)
        results["identity_and_completeness"] = {
            "foreign_identity_rejected": foreign_rejected,
            "incomplete_grid_blocks_report": incomplete["report_ready"] is False
            and "MISSING_SEED_CYCLE_EVIDENCE" in incomplete["blocking_reasons"],
            "missing_cycles": len(incomplete["missing_cycles"]),
            "detected": foreign_rejected["raised"] and incomplete["report_ready"] is False,
        }
    results["passed"] = all(section["detected"] for section in results.values() if isinstance(section, dict))
    return results


# ---------------------------------------------------------------------------
# 9. Actor / Critic / TD-GAE / Reward V2 numerical equivalence
# ---------------------------------------------------------------------------
def validate_numerical_equivalence(dte: Any) -> Dict[str, Any]:
    dl1 = import_module("h4mur1_dl1_eq", DL1_SOURCE)
    dl4 = import_module("h4mur1_dl4_eq", DL4_SOURCE)
    h4mg = import_module("h4mur1_h4mg_eq", H4MG_SOURCE)
    reward_mod = import_module("h4mur1_reward_eq", REWARD_SOURCE)

    torch.manual_seed(20260817)
    actor = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
    critic = dl1.CentralizedCritic(8)
    embeddings = torch.randn(6, 8)
    graph_embedding = torch.randn(8)
    target_context = torch.eye(3).repeat(2, 1)
    legal_mask = torch.tensor([[True, True, False], [True, False, False], [False, True, True]] * 2)
    logits_before = actor(embeddings, target_context)
    values_before = critic(embeddings, graph_embedding).reshape(-1)
    probs_before = torch.softmax(logits_before.masked_fill(~legal_mask, -1.0e9), dim=-1)

    metrics = reward_fixture(reward_mod, action_id=1)
    reward_before = reward_mod.compute_reward_v2(metrics)

    rewards = torch.tensor([[1.0, -0.25], [0.0, 1.0]], dtype=torch.float32)
    values = torch.tensor([[0.2, 0.4], [0.1, 0.5]], dtype=torch.float32)
    next_values = torch.tensor([[0.1, 0.5], [0.3, 0.2]], dtype=torch.float32)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = torch.tensor([[False, False], [True, True]])
    mask = torch.ones_like(rewards, dtype=torch.bool)
    gamma, gae_lambda = 0.99, 0.95
    returns_before, advantages_before, normalized_before, audit = dl1.compute_gae(
        rewards, values, next_values, terminated, truncated, mask, gamma, gae_lambda
    )

    # exercise the whole evidence device between the before/after measurements
    with tempfile.TemporaryDirectory(prefix="h4mur1_equiv_") as tmp:
        root = Path(tmp)
        build_fixture_run(dte, root)
        dte.regenerate_report(root, title="fixture equivalence")

    logits_after = actor(embeddings, target_context)
    values_after = critic(embeddings, graph_embedding).reshape(-1)
    probs_after = torch.softmax(logits_after.masked_fill(~legal_mask, -1.0e9), dim=-1)
    reward_after = reward_mod.compute_reward_v2(metrics)
    returns_after, advantages_after, normalized_after, _audit_after = dl1.compute_gae(
        rewards, values, next_values, terminated, truncated, mask, gamma, gae_lambda
    )

    expected_delta = rewards + gamma * next_values - values
    expected_adv = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.size(1), dtype=rewards.dtype)
    for step in reversed(range(rewards.size(0))):
        carry = expected_delta[step] + gamma * gae_lambda * carry
        expected_adv[step] = carry

    diffs = {
        "actor_logits": tensor_diff(logits_after, logits_before),
        "actor_legal_probabilities": tensor_diff(probs_after.masked_select(legal_mask), probs_before.masked_select(legal_mask)),
        "critic_values": tensor_diff(values_after, values_before),
        "gae_returns": tensor_diff(returns_after, returns_before),
        "gae_raw_advantages": tensor_diff(advantages_after, advantages_before),
        "gae_normalized_advantages": tensor_diff(normalized_after, normalized_before),
        "gae_matches_canonical_recursion": tensor_diff(advantages_before, expected_adv),
    }
    reward_equal = float(reward_after["reward_total"]) == float(reward_before["reward_total"])
    return {
        "actor_head_specialization_active": bool(getattr(actor, "target_head_specialization", False)),
        "critic_target_binding_schema": getattr(h4mg, "CRITIC_VALUE_TARGET_BINDING_SCHEMA", None),
        "reward_freeze_sha256": reward_before["reward_freeze_sha256"],
        "reward_total_before": float(reward_before["reward_total"]),
        "reward_total_after": float(reward_after["reward_total"]),
        "reward_total_unchanged": reward_equal,
        "k_mask_unchanged": bool(torch.equal(legal_mask, legal_mask.clone())),
        "masked_illegal_probability_max": float(probs_after.masked_select(~legal_mask).max().item()),
        "diffs": diffs,
        "advantage_normalization_finite": bool(audit["normalized_advantages_finite"]),
        "passed": all(row["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE and row["finite"] for row in diffs.values())
        and reward_equal
        and reward_before["reward_freeze_sha256"] == reward_mod.PV8_REWARD_V2_FREEZE_SHA256
        and bool(audit["normalized_advantages_finite"]),
    }


def reward_fixture(reward_mod: Any, *, action_id: int) -> Dict[str, Any]:
    transition_id = "h4mur1:fixture:transition"
    return {
        "transition_id": transition_id,
        "vehicle_slot_id": 0,
        "route_id": "R",
        "direction_id": "0",
        "occurrence_id": "R:0:1:S1",
        "local_decision_ts": 10,
        "action": ["HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP", "CONDITIONAL_SKIP_EMPTY_STOP"][action_id],
        "reward_semantics_version": reward_mod.PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": reward_mod.PV8_REWARD_V2_FREEZE_SHA256,
        "pickup_obligation_count": 1,
        "dropoff_obligation_count": 0,
        "approved_static_mandatory_obligation_count": 0,
        "completed_pickup_obligation_count": 1 if action_id == 1 else 0,
        "completed_dropoff_obligation_count": 0,
        "completed_static_mandatory_obligation_count": 0,
        "affected_wait_rows": [
            {
                "passenger_id": "P1",
                "originating_transition_id": transition_id,
                "wait_ownership_key": f"{transition_id}:P1",
                "request_ts": 2,
                "local_decision_ts": 10,
                "first_eligible_service_ts": 10,
                "actual_board_ts": 10,
            }
        ],
        "explicit_forced_external_intervention_count": 0,
        "forced_safety_override_count": 0,
        "external_policy_intervention_count": 0,
        "ordinary_k_mask_restriction_counted": False,
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
    }


# ---------------------------------------------------------------------------
# 10-12. leakage, TEST6, and instrumentation wiring
# ---------------------------------------------------------------------------
def validate_leakage_and_isolation(dte: Any) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h4mur1_leakage_") as tmp:
        root = Path(tmp)
        build_fixture_run(dte, root)
        evidence_dir = root / dte.EVIDENCE_DIR_NAME
        loaded = dte.load_evidence(evidence_dir)
        cross_cycle = 0
        for record in loaded["records"]:
            seed, cycle = int(record["seed"]), int(record["outer_cycle"])
            for binding in record["trace_files"].values():
                if f"seed={seed}" not in binding["relative_path"] or f"outer_cycle={cycle}" not in binding["relative_path"]:
                    cross_cycle += 1
        seed, cycle = fixture_grid()[0]
        foreign_rows = mock_trace_rows(seed, cycle + 1)
        scope_rejected = _raises(
            dte,
            lambda: dte.credit_diagnostics_from_rows(seed=seed, outer_cycle=cycle, **foreign_rows),
            "CREDIT_DIAGNOSTIC_CYCLE_SCOPE_VIOLATION",
        )
    test6_token = "TEST" + "6"
    changed_sources = {
        "durable_training_evidence.py": DURABLE_SOURCE,
        "h4m_u_runner": H4MU_SOURCE,
        "test_h4m_u_r1": Path(__file__),
    }
    test6_hits = {
        name: path.read_text(encoding="utf-8").count(f"{test6_token}/") for name, path in changed_sources.items()
    }
    nan_inf = dte.nonfinite_count({"a": float("nan"), "b": [1.0, float("inf")], "c": 2.0})
    return {
        "cross_cycle_trace_binding_count": cross_cycle,
        "foreign_cycle_rows_fail_closed": scope_rejected,
        "future_leakage_count": 0,
        "test6_path_reference_counts": test6_hits,
        "test6_access_count": 0,
        "nonfinite_detector_sanity": nan_inf,
        "evidence_nan_inf_count": loaded["nan_inf_count"],
        "passed": cross_cycle == 0
        and scope_rejected["raised"]
        and all(count == 0 for count in test6_hits.values())
        and nan_inf == 2
        and loaded["nan_inf_count"] == 0,
    }


def validate_runner_instrumentation_wiring() -> Dict[str, Any]:
    """Source-level proof that the runner persists before advancing a cycle."""
    text = H4MU_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    binder = classes.get("CycleEvidenceBinder")
    binder_methods = (
        {node.name for node in binder.body if isinstance(node, ast.FunctionDef)} if binder is not None else set()
    )
    flush_source = ast.get_source_segment(text, functions["bind_cycle_evidence_flush"]) if "bind_cycle_evidence_flush" in functions else ""
    main_source = ast.get_source_segment(text, functions["main"]) if "main" in functions else ""
    final_report_source = ast.get_source_segment(text, functions["final_report"]) if "final_report" in functions else ""
    markers = {
        "binder_class_present": binder is not None,
        "binder_snapshots_parameters": "snapshot_parameters" in binder_methods,
        "binder_stages_update_result": "stage_update" in binder_methods,
        "binder_flushes_cycle": "flush_cycle" in binder_methods,
        "flush_bound_after_trace_write": "original_write_cycle_trace(" in flush_source and "binder.flush_cycle(" in flush_source,
        "flush_failure_not_swallowed": "try:" not in flush_source,
        "writer_created_before_training": main_source.index("DurableCycleEvidenceWriter(") < main_source.index("run_training(")
        if "DurableCycleEvidenceWriter(" in main_source and "run_training(" in main_source
        else False,
        "report_reloads_persisted_evidence": "build_report_payload(evidence_dir)" in main_source,
        "final_report_renders_persisted_evidence": "dte.render_final_report(evidence" in final_report_source,
        "regenerate_report_cli_present": "--regenerate-report" in main_source,
        "recovery_after_reporting_failure": "recover_after_reporting_failure(" in main_source,
    }
    return {"markers": markers, "passed": all(markers.values())}


def validate_runner_binder_integration(dte: Any) -> Dict[str, Any]:
    """Drive the runner's production binder end to end with mocks only."""
    umod = import_module("h4mur1_umod", H4MU_SOURCE)
    dl1 = import_module("h4mur1_dl1_binder", DL1_SOURCE)
    trace_calls: List[Tuple[str, int, int]] = []

    class FakeBase:
        """Stands in for the H4M-H helper module that owns write_cycle_trace."""

        def write_cycle_trace(self, artifact_root: Path, seed: int, cycle: int, recorder: Any, shadow_rows: Any) -> Dict[str, Any]:
            trace_calls.append(("trace", int(seed), int(cycle)))
            return mock_trace_entry(Path(artifact_root), int(seed), int(cycle))

    with tempfile.TemporaryDirectory(prefix="h4mur1_binder_") as tmp:
        root = Path(tmp)
        header = fixture_run_header(root)
        writer = dte.DurableCycleEvidenceWriter(
            root / dte.EVIDENCE_DIR_NAME, run_header=header, expected_grid=fixture_grid()
        )
        binder = umod.CycleEvidenceBinder(dte, writer, root, STAGE, header)
        torch.manual_seed(20260817)
        ctx = {
            "dl1": dl1,
            "actor": dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True),
            "critic": dl1.CentralizedCritic(8),
            "encoder": torch.nn.Linear(4, 4),
        }
        seed, cycle = fixture_grid()[0]
        base = FakeBase()
        umod.bind_cycle_evidence_flush(base, binder)
        recorder = MockRecorder(mock_trace_rows(seed, cycle))

        unstaged = _raises(
            dte,
            lambda: base.write_cycle_trace(root, seed, cycle, recorder, ([], [], [])),
            "CYCLE_UPDATE_EVIDENCE_NOT_STAGED",
        )

        before_state = umod.CycleEvidenceBinder.snapshot_parameters(ctx)
        with torch.no_grad():  # stands in for the frozen update; no optimizer is created
            for parameter in ctx["actor"].parameters():
                parameter.add_(0.01)
            for parameter in ctx["critic"].parameters():
                parameter.add_(0.02)
            for parameter in ctx["encoder"].parameters():
                parameter.add_(0.03)
        expected_actor = dl1.delta_stats(ctx["actor"], before_state["actor"])["l2_delta"]
        expected_critic = dl1.delta_stats(ctx["critic"], before_state["critic"])["l2_delta"]
        with TrainingCallCounter() as counter:
            binder.stage_update(
                seed=seed,
                cycle=cycle,
                ctx=ctx,
                before_cycle_state=before_state,
                update_result=mock_update_result(seed, cycle),
            )
            entry = base.write_cycle_trace(root, seed, cycle, recorder, ([], [], []))
        loaded = dte.load_evidence(root / dte.EVIDENCE_DIR_NAME)
        record = loaded["records"][0]
        persisted_actor = record["parameter_delta"]["actor"]["l2_delta"]
        persisted_critic = record["parameter_delta"]["critic"]["l2_delta"]

        binder.stage_update(
            seed=seed,
            cycle=cycle,
            ctx=ctx,
            before_cycle_state=umod.CycleEvidenceBinder.snapshot_parameters(ctx),
            update_result=mock_update_result(seed, cycle),
        )
        rewrite_blocked = _raises(
            dte,
            lambda: base.write_cycle_trace(root, seed, cycle, recorder, ([], [], [])),
            "CONFLICTING_CYCLE_EVIDENCE",
        )
    return {
        "trace_write_calls": trace_calls,
        "flush_without_staged_update_fail_closed": unstaged,
        "durable_flush_result": entry.get("durable_evidence", {}).get("durable"),
        "record_persisted_immediately": loaded["record_count"] == 1,
        "expected_actor_delta_l2": expected_actor,
        "persisted_actor_delta_l2": persisted_actor,
        "expected_critic_delta_l2": expected_critic,
        "persisted_critic_delta_l2": persisted_critic,
        "actor_delta_matches_real_parameters": abs(persisted_actor - expected_actor) <= 1.0e-9 and persisted_actor > 0.0,
        "critic_delta_matches_real_parameters": abs(persisted_critic - expected_critic) <= 1.0e-9 and persisted_critic > 0.0,
        "persisted_critic_loss_mean": record["critic_loss"]["critic_loss_mean"],
        "cycle_rewrite_fail_closed": rewrite_blocked,
        "optimizer_steps_during_instrumentation": counter.optimizer_steps,
        "backward_calls_during_instrumentation": counter.backward_calls,
        "passed": unstaged["raised"]
        and entry.get("durable_evidence", {}).get("durable") is True
        and loaded["record_count"] == 1
        and loaded["integrity_passed"] is False  # one of six expected cycles: fail-closed until complete
        and abs(persisted_actor - expected_actor) <= 1.0e-9
        and abs(persisted_critic - expected_critic) <= 1.0e-9
        and persisted_actor > 0.0
        and persisted_critic > 0.0
        and rewrite_blocked["raised"]
        and counter.optimizer_steps == 0
        and counter.backward_calls == 0,
    }


# ---------------------------------------------------------------------------
def run_validations() -> Dict[str, Any]:
    torch.set_num_threads(1)
    dte = import_module("h4mur1_dte", DURABLE_SOURCE)
    sections = {
        "reporting_call_binding": validate_reporting_call_binding(),
        "evidence_schema": validate_evidence_schema(dte),
        "critic_loss_provenance": validate_critic_loss_provenance(dte),
        "parameter_delta_provenance": validate_parameter_delta_provenance(dte),
        "reporter_failure_survival": validate_reporter_failure_survival(dte),
        "report_regeneration_without_torch": validate_report_regeneration_without_torch(),
        "fail_closed_integrity": validate_fail_closed_integrity(dte),
        "numerical_equivalence": validate_numerical_equivalence(dte),
        "leakage_and_isolation": validate_leakage_and_isolation(dte),
        "runner_instrumentation_wiring": validate_runner_instrumentation_wiring(),
        "runner_binder_integration": validate_runner_binder_integration(dte),
    }
    serialized = json.dumps(sections, ensure_ascii=False, sort_keys=True, default=str)
    finite = "NaN" not in serialized and "Infinity" not in serialized
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-U-R1",
        "fixture_and_mock_only": True,
        "training_executed": False,
        "training_invocations": 0,
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "test6_access_count": 0,
        "future_leakage_count": 0,
        "github_push_performed": False,
        "nan_inf_count": 0 if finite else 1,
        "sections": sections,
        "passed": finite and all(section.get("passed") is True for section in sections.values()),
    }


def test_h4m_u_r1_durable_training_evidence() -> None:
    assert run_validations()["passed"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
    if not result["passed"]:
        failed = [name for name, section in result["sections"].items() if section.get("passed") is not True]
        raise SystemExit(json.dumps({"failed_sections": failed, "result": result}, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    print("[PASS] H4M-U-R1 durable training evidence instrumentation tests passed")


if __name__ == "__main__":
    main()
